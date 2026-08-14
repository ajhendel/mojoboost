"""End-to-end GPU training.

`train_gpu` mirrors `train` in boosting.mojo but grows every tree through the
GPU backend: one persistent `GpuHistogramBuilder` holds the binned matrix,
gradients/hessians, and a per-row leaf-assignment array device-resident for
the whole session. Per boosting round, the host uploads gradients once; per
split, the device reassigns rows and builds the smaller child's histogram
(the sibling comes from the subtraction trick on the host, where histograms
are small: n_features * n_bins).

Division of labor:
  CPU  boosting coordination, split selection over downloaded histograms,
       leaf-value renewal (quantile/L1), prediction, the tree model itself
  GPU  binned features, gradients/hessians, leaf assignments, histogram
       accumulation, row partitioning

No row-index lists or per-node gradient vectors ever cross the host/device
boundary. GPU histograms carry Float32 precision (see histogram_gpu.mojo), so
trained models agree with the CPU trainer's predictions to Float32-level
tolerance, not bit-exactly; the GPU trainer itself is bit-deterministic run
to run.
"""

from std.sys import has_accelerator

from .binning import BinnedMatrix
from .boosting import (
    L1,
    QUANTILE,
    Booster,
    BoosterParams,
    _base_score,
    _check_objective,
    _check_sample_weight,
    _fill_grad_hess,
    _renew_leaf_values,
)
from .histogram import Histogram, subtract_histogram
from .histogram_gpu import GpuHistogramBuilder
from .split import SplitInfo
from .tree import Tree, TreeParams, _leaf_value, _search


struct _GpuLeafState(Movable):
    """A grown-but-unsplit leaf: its node id (also its device-side leaf id),
    row count, histogram, and the best split available from it."""

    var node: Int
    var n_rows: Int
    var hist: Histogram
    var split: SplitInfo

    def __init__(
        out self,
        node: Int,
        n_rows: Int,
        var hist: Histogram,
        var split: SplitInfo,
    ):
        self.node = node
        self.n_rows = n_rows
        self.hist = hist^
        self.split = split^


def _count_left(hist: Histogram, feature: Int, threshold_bin: Int) -> Int:
    """Rows going left under (feature, threshold_bin), from the exact integer
    counts of the node's histogram — no host-side row partitioning needed."""
    var total = 0
    var base = feature * hist.n_bins
    for b in range(threshold_bin + 1):
        total += hist.count[base + b]
    return total


def grow_tree_gpu(
    mut builder: GpuHistogramBuilder, params: TreeParams
) raises -> Tree:
    """Grow one tree, leaf-wise, with histogram accumulation and row
    partitioning on the GPU. Gradients for this round must already be
    uploaded via `builder.upload_gradients`. Node ids double as device-side
    leaf ids, and nodes are created in the same order as the CPU
    `grow_tree`, so equal split decisions yield identical tree layouts."""
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )

    builder.begin_tree()
    var root = tree._add_node(0.0)
    var root_hist = builder.build_leaf(root)
    tree.value[root] = _leaf_value(root_hist, params.lambda_reg)
    var root_split = _search(root_hist, builder.n_rows, params)

    var frontier = List[_GpuLeafState]()
    frontier.append(
        _GpuLeafState(root, builder.n_rows, root_hist^, root_split^)
    )
    var n_leaves = 1

    while n_leaves < params.num_leaves:
        # Pick the leaf with the best gain anywhere in the tree.
        var best_i = -1
        var best_gain = 0.0
        for i in range(len(frontier)):
            if frontier[i].split.found and frontier[i].split.gain > best_gain:
                best_gain = frontier[i].split.gain
                best_i = i
        if best_i < 0:
            break

        var parent_node = frontier[best_i].node
        var split = frontier[best_i].split.copy()
        var n_left = _count_left(
            frontier[best_i].hist, split.feature, split.bin
        )
        var n_right = frontier[best_i].n_rows - n_left

        var left_node = tree._add_node(0.0)
        var right_node = tree._add_node(0.0)
        builder.apply_split(
            split.feature, split.bin, parent_node, left_node, right_node
        )

        # Histogram subtraction trick: build the smaller child directly.
        var left_hist: Histogram
        var right_hist: Histogram
        if n_left <= n_right:
            left_hist = builder.build_leaf(left_node)
            right_hist = subtract_histogram(frontier[best_i].hist, left_hist)
        else:
            right_hist = builder.build_leaf(right_node)
            left_hist = subtract_histogram(frontier[best_i].hist, right_hist)

        tree.value[left_node] = _leaf_value(left_hist, params.lambda_reg)
        tree.value[right_node] = _leaf_value(right_hist, params.lambda_reg)
        tree.feature[parent_node] = split.feature
        tree.threshold_bin[parent_node] = split.bin
        tree.left[parent_node] = left_node
        tree.right[parent_node] = right_node
        tree.split_gain[parent_node] = split.gain

        var left_split = _search(left_hist, n_left, params)
        var right_split = _search(right_hist, n_right, params)

        frontier[best_i] = _GpuLeafState(
            left_node, n_left, left_hist^, left_split^
        )
        frontier.append(
            _GpuLeafState(right_node, n_right, right_hist^, right_split^)
        )
        n_leaves += 1

    tree.n_leaves = n_leaves
    return tree^


def train_gpu(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
) raises -> Booster:
    """Train a boosted ensemble with tree growth on the GPU. Same contract
    as `train` (objectives, sample_weight, alpha semantics); requires an
    accelerator at runtime and at most 256 bins."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(target) != data.n_rows:
            raise Error("target length must equal n_rows")
        _check_objective(objective, target, alpha)
        _check_sample_weight(sample_weight, data.n_rows)

        var n = data.n_rows
        var base_score = _base_score(target, objective, sample_weight, alpha)
        var raw = List[Float64](capacity=n)
        for _ in range(n):
            raw.append(base_score)

        var builder = GpuHistogramBuilder(data)
        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        for _ in range(params.n_estimators):
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, grad, hess
            )
            builder.upload_gradients(grad, hess)
            var tree = grow_tree_gpu(builder, params.tree)
            if objective == QUANTILE:
                _renew_leaf_values(
                    tree, data, target, raw, sample_weight, alpha
                )
            elif objective == L1:
                _renew_leaf_values(tree, data, target, raw, sample_weight, 0.5)

            # A single-leaf tree with a near-zero value means the objective
            # has converged; further rounds cannot make progress.
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                break

            for r in range(n):
                raw[r] += params.learning_rate * tree.predict_row(data, r)
            trees.append(tree^)

        return Booster(trees^, base_score, params.learning_rate, objective)
