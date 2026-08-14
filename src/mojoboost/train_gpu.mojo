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

Row bagging is the one exception to the no-row-lists rule, and only at the
start of a tree: the bag decides which rows sit at the root and which sit
out of bag, so the leaf-assignment array is written once per tree instead of
memset. Both trainers draw bags from bagging.mojo with the same seed and
schedule, so CPU and GPU rounds are grown on identical rows.

No per-node row lists or per-node gradient vectors ever cross the
host/device boundary. GPU histograms carry Float32 precision (see
histogram_gpu.mojo), so
trained models agree with the CPU trainer's predictions to Float32-level
tolerance, not bit-exactly; the GPU trainer itself is bit-deterministic run
to run.
"""

from std.math import log
from std.sys import has_accelerator

from .bagging import BaggingParams, bagging_enabled, check_bagging, refresh_bag
from .binning import BinnedMatrix
from .boosting import (
    CUSTOM,
    L1,
    QUANTILE,
    Booster,
    BoosterParams,
    MulticlassBooster,
    _base_score,
    _check_objective,
    _check_sample_weight,
    _clamp_prob,
    _fill_grad_hess,
    _renew_leaf_values,
    _softmax_inplace,
)
from .histogram import Histogram, subtract_histogram
from .histogram_gpu import GpuHistogramBuilder
from .objective import (
    GradHessFn,
    _apply_sample_weight,
    check_custom_grad_hess,
)
from .interaction import extend_branch
from .sampling import select_node_features, select_tree_features
from .split import SplitInfo
from .tree import Tree, TreeParams, _leaf_value, _search


struct _GpuLeafState(Movable):
    """A grown-but-unsplit leaf: its node id (also its device-side leaf id),
    row count, histogram, the best split available from it, the features
    split on between the root and it (empty when no interaction constraints
    are configured), and its depth in edges from the root."""

    var node: Int
    var n_rows: Int
    var hist: Histogram
    var split: SplitInfo
    var branch: List[Int]
    var depth: Int

    def __init__(
        out self,
        node: Int,
        n_rows: Int,
        var hist: Histogram,
        var split: SplitInfo,
        var branch: List[Int] = [],
        depth: Int = 0,
    ):
        self.node = node
        self.n_rows = n_rows
        self.hist = hist^
        self.split = split^
        self.branch = branch^
        self.depth = depth


def _count_left(hist: Histogram, feature: Int, threshold_bin: Int) -> Int:
    """Rows going left under (feature, threshold_bin), from the exact integer
    counts of the node's histogram — no host-side row partitioning needed."""
    var total = 0
    var base = feature * hist.n_bins
    for b in range(threshold_bin + 1):
        total += hist.count[base + b]
    return total


def grow_tree_gpu(
    mut builder: GpuHistogramBuilder,
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
) raises -> Tree:
    """Grow one tree, leaf-wise, with histogram accumulation and row
    partitioning on the GPU. Gradients for this round must already be
    uploaded via `builder.upload_gradients`. Node ids double as device-side
    leaf ids, and nodes are created in the same order as the CPU
    `grow_tree`, so equal split decisions yield identical tree layouts.

    A non-empty `bag` restricts growth to those rows, exactly as in
    `grow_tree`: the device parks the rest out of bag (see
    histogram_gpu.mojo), so bagged rows are the only rows any histogram,
    count, or split on this tree sees. Both backends take the bag from the
    same sampler, so the two grow on identical rows.

    Interaction constraints are tracked exactly as in `grow_tree`: the same
    branch feature sets, the same allow masks, and the same `_search` entry
    point. Constraint enforcement is therefore identical on both backends,
    independent of the Float32 histogram precision the GPU accumulates in.

    `params.max_depth` is tracked the same way, as a per-frontier-leaf depth
    incremented on each split and checked inside `_search`. Since the depth
    limit depends only on tree shape and not on histogram values, the two
    backends cut growth at exactly the same leaves.

    Feature subsampling likewise goes through the same sampler as the CPU
    grower: `tree_index` and `params.feature_fraction_seed` fix the tree's
    feature set, which is handed to the device once per tree so its
    histogram kernel accumulates exactly those features, and the per-node
    sets (drawn from the node ids, which both growers assign in the same
    order) narrow each split search identically."""
    params.constraints.check_features(builder.n_features)
    var tree_features = select_tree_features(
        builder.n_features,
        params.feature_fraction,
        params.feature_fraction_seed,
        tree_index,
    )
    builder.set_features(tree_features)
    # Leaf-value totals must come from a feature the histograms accumulated.
    var value_feature = tree_features[0]
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )

    builder.begin_tree(bag)
    var n_root = len(bag) if len(bag) > 0 else builder.n_rows
    var root = tree._add_node(0.0)
    var root_hist = builder.build_leaf(root)
    tree.value[root] = _leaf_value(
        root_hist, params.lambda_reg, params.lambda_l1, value_feature
    )
    var root_branch = List[Int]()
    var root_split = _search(
        root_hist,
        n_root,
        params,
        params.constraints.allowed_features(root_branch),
        select_node_features(
            tree_features,
            params.feature_fraction_bynode,
            params.feature_fraction_seed,
            tree_index,
            root,
        ),
        depth=0,
    )

    var frontier = List[_GpuLeafState]()
    frontier.append(
        _GpuLeafState(
            root, n_root, root_hist^, root_split^, root_branch^, depth=0
        )
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

        tree.value[left_node] = _leaf_value(
            left_hist, params.lambda_reg, params.lambda_l1, value_feature
        )
        tree.value[right_node] = _leaf_value(
            right_hist, params.lambda_reg, params.lambda_l1, value_feature
        )
        tree.feature[parent_node] = split.feature
        tree.threshold_bin[parent_node] = split.bin
        tree.left[parent_node] = left_node
        tree.right[parent_node] = right_node
        tree.split_gain[parent_node] = split.gain

        # Both children inherit the same branch feature set, so they share one
        # allow mask, and both sit one edge below the leaf that was split.
        var branch = extend_branch(frontier[best_i].branch, split.feature)
        var allowed = params.constraints.allowed_features(branch)
        var child_depth = frontier[best_i].depth + 1
        # Each child draws its own per-node feature set from its node id, the
        # same id the CPU grower would assign it.
        var left_split = _search(
            left_hist,
            n_left,
            params,
            allowed,
            select_node_features(
                tree_features,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                left_node,
            ),
            depth=child_depth,
        )
        var right_split = _search(
            right_hist,
            n_right,
            params,
            allowed,
            select_node_features(
                tree_features,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                right_node,
            ),
            depth=child_depth,
        )

        frontier[best_i] = _GpuLeafState(
            left_node,
            n_left,
            left_hist^,
            left_split^,
            branch.copy(),
            depth=child_depth,
        )
        frontier.append(
            _GpuLeafState(
                right_node,
                n_right,
                right_hist^,
                right_split^,
                branch^,
                depth=child_depth,
            )
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
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> Booster:
    """Train a boosted ensemble with tree growth on the GPU. Same contract
    as `train` (objectives, sample_weight, alpha, and bagging semantics);
    requires an accelerator at runtime and at most 256 bins. Bags come from
    the same sampler and the same schedule as on the CPU, so both backends
    grow round i on exactly the same rows."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(target) != data.n_rows:
            raise Error("target length must equal n_rows")
        _check_objective(objective, target, alpha)
        _check_sample_weight(sample_weight, data.n_rows)
        check_bagging(bagging)

        var n = data.n_rows
        var base_score = _base_score(target, objective, sample_weight, alpha)
        var raw = List[Float64](capacity=n)
        for _ in range(n):
            raw.append(base_score)

        var builder = GpuHistogramBuilder(data)
        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        var bag = List[Int]()
        for i in range(params.n_estimators):
            refresh_bag(bag, bagging, n, i)
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, grad, hess
            )
            builder.upload_gradients(grad, hess)
            var tree = grow_tree_gpu(builder, params.tree, bag, i)
            if objective == QUANTILE:
                _renew_leaf_values(
                    tree, data, target, raw, sample_weight, alpha, bag
                )
            elif objective == L1:
                _renew_leaf_values(
                    tree, data, target, raw, sample_weight, 0.5, bag
                )

            # A single-leaf tree with a near-zero value means the objective
            # has converged; further rounds cannot make progress. Under
            # bagging it only means this bag had nothing to give, so the
            # round is skipped and the next bag gets its turn.
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                if bagging_enabled(bagging):
                    continue
                break

            for r in range(n):
                raw[r] += params.learning_rate * tree.predict_row(data, r)
            trees.append(tree^)

        return Booster(trees^, base_score, params.learning_rate, objective)


def train_custom_gpu[F: GradHessFn](
    data: BinnedMatrix,
    target: List[Float64],
    grad_hess: F,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
) raises -> Booster:
    """`train_custom` with tree growth on the GPU: the objective callback
    stays on the host (it is one call per round over host-side raw scores),
    and only the resulting gradients cross to the device, exactly as the
    built-in objectives do in `train_gpu`. Same contract and validation as
    `train_custom` in objective.mojo; requires an accelerator at runtime and
    at most 256 bins."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(target) != data.n_rows:
            raise Error("target length must equal n_rows")
        _check_sample_weight(sample_weight, data.n_rows)

        var n = data.n_rows
        var raw = List[Float64](capacity=n)
        for _ in range(n):
            raw.append(base_score)

        var builder = GpuHistogramBuilder(data)
        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        for i in range(params.n_estimators):
            grad_hess(raw, target, grad, hess)
            check_custom_grad_hess(grad, hess, n)
            _apply_sample_weight(grad, hess, sample_weight)
            builder.upload_gradients(grad, hess)
            var tree = grow_tree_gpu(builder, params.tree, [], i)

            # A single-leaf tree with a near-zero value means the objective
            # has converged; further rounds cannot make progress.
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                break

            for r in range(n):
                raw[r] += params.learning_rate * tree.predict_row(data, r)
            trees.append(tree^)

        return Booster(trees^, base_score, params.learning_rate, CUSTOM)


def train_multiclass_gpu(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> MulticlassBooster:
    """`train_multiclass` with tree growth on the GPU.

    Softmax is the last objective that shares the per-row gradient/hessian
    interface, so it needs no new device machinery: one class's tree is one
    ordinary `grow_tree_gpu` call over that class's gradients. One builder
    serves every class of every round, so the binned matrix is uploaded once
    for the whole ensemble and each round costs n_classes gradient uploads.

    Same contract as `train_multiclass` (labels in 0..n_classes-1,
    sample_weight, bagging semantics); requires an accelerator at runtime and
    at most 256 bins. Softmax probabilities are computed on the host, exactly
    as on the CPU, so the only backend difference remains the Float32
    histogram precision."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(labels) != data.n_rows:
            raise Error("labels length must equal n_rows")
        if n_classes < 2:
            raise Error("n_classes must be at least 2")
        _check_sample_weight(sample_weight, data.n_rows)
        check_bagging(bagging)
        var n = data.n_rows

        # Base scores are log priors (weighted when sample_weight is given).
        var class_w = List[Float64](capacity=n_classes)
        for _ in range(n_classes):
            class_w.append(0.0)
        var total_w = 0.0
        for r in range(n):
            if labels[r] < 0 or labels[r] >= n_classes:
                raise Error("label out of range")
            var w = sample_weight[r] if len(sample_weight) > 0 else 1.0
            class_w[labels[r]] += w
            total_w += w
        if total_w <= 0.0:
            raise Error("sample_weight must have a positive sum")
        var base_scores = List[Float64](capacity=n_classes)
        for k in range(n_classes):
            base_scores.append(log(_clamp_prob(class_w[k] / total_w)))

        # Row-major raw scores and softmax scratch: raw[r * n_classes + k].
        var raw = List[Float64](capacity=n * n_classes)
        for _ in range(n):
            for k in range(n_classes):
                raw.append(base_scores[k])
        var prob = List[Float64](capacity=n * n_classes)
        for _ in range(n * n_classes):
            prob.append(0.0)

        var builder = GpuHistogramBuilder(data)
        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        var bag = List[Int]()
        for i in range(params.n_estimators):
            refresh_bag(bag, bagging, n, i)
            for r in range(n):
                for k in range(n_classes):
                    prob[r * n_classes + k] = raw[r * n_classes + k]
                _softmax_inplace(prob, r * n_classes, n_classes)

            var made_progress = False
            for k in range(n_classes):
                grad.clear()
                hess.clear()
                for r in range(n):
                    var p = prob[r * n_classes + k]
                    var y = 1.0 if labels[r] == k else 0.0
                    var w = (
                        sample_weight[r] if len(sample_weight) > 0 else 1.0
                    )
                    grad.append(w * (p - y))
                    # LightGBM/XGBoost softmax hessian: 2 * p * (1 - p),
                    # floored.
                    var h = 2.0 * p * (1.0 - p)
                    if h < 1e-16:
                        h = 1e-16
                    hess.append(w * h)
                builder.upload_gradients(grad, hess)
                # Feature subsampling draws once per tree, so each class's
                # tree in a round gets its own feature set; the same index
                # the CPU grower uses keeps the two backends on identical
                # feature sets.
                var tree = grow_tree_gpu(
                    builder, params.tree, bag, i * n_classes + k
                )
                if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                    made_progress = True
                for r in range(n):
                    raw[r * n_classes + k] += (
                        params.learning_rate * tree.predict_row(data, r)
                    )
                trees.append(tree^)

            # No class made progress: with bagging that is a statement about
            # this bag, so the round is dropped and the next bag gets its
            # turn.
            if not made_progress:
                for _ in range(n_classes):
                    _ = trees.pop()
                if bagging_enabled(bagging):
                    continue
                break

        return MulticlassBooster(
            trees^, base_scores^, n_classes, params.learning_rate
        )
