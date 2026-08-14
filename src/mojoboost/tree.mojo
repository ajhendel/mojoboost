"""Leaf-wise (best-first) tree growth.

Grows a single regression tree the way LightGBM does. At each step the leaf
with the highest split gain anywhere in the tree is split, until num_leaves
is reached or no leaf has a positive-gain split. For each split, the smaller
child's histogram is built directly and the larger child's is derived by
subtraction from the parent's.

Leaf values use the second-order Newton step: -G / (H + lambda).
"""

from .binning import BinnedMatrix
from .histogram import (
    Histogram,
    build_histogram,
    build_histogram_subset,
    subtract_histogram,
)
from .split import SplitInfo, find_best_split


@fieldwise_init
struct TreeParams(Copyable, Movable):
    var num_leaves: Int
    var min_data_in_leaf: Int
    var lambda_reg: Float64
    var min_child_hess: Float64

    @staticmethod
    def default() -> TreeParams:
        # LightGBM defaults (min_child_hess mirrors min_sum_hessian_in_leaf).
        return TreeParams(31, 20, 1.0, 1e-3)


@fieldwise_init
struct Tree(Copyable, Movable):
    """Flat-array tree. Node i is internal when feature[i] >= 0; then rows
    with bin(feature[i]) <= threshold_bin[i] go to left[i], the rest to
    right[i]. Node i is a leaf when feature[i] < 0; its output is value[i]."""

    var feature: List[Int]
    var threshold_bin: List[Int]
    var left: List[Int]
    var right: List[Int]
    var value: List[Float64]
    var split_gain: List[Float64]
    var n_leaves: Int

    def _add_node(mut self, value: Float64) -> Int:
        var node = len(self.feature)
        self.feature.append(-1)
        self.threshold_bin.append(-1)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(value)
        # Recorded when the node is split; stays 0.0 for leaves (and for
        # every node of a model loaded from disk).
        self.split_gain.append(0.0)
        return node

    def predict_row(self, data: BinnedMatrix, row: Int) -> Float64:
        var node = 0
        while self.feature[node] >= 0:
            if data.bin_at(row, self.feature[node]) <= self.threshold_bin[node]:
                node = self.left[node]
            else:
                node = self.right[node]
        return self.value[node]

    def predict_bins(self, bins: List[Int]) -> Float64:
        """Predict one example given its per-feature bin ids."""
        var node = 0
        while self.feature[node] >= 0:
            if bins[self.feature[node]] <= self.threshold_bin[node]:
                node = self.left[node]
            else:
                node = self.right[node]
        return self.value[node]

    def leaf_index_row(self, data: BinnedMatrix, row: Int) -> Int:
        """The node index of the leaf this row lands in."""
        var node = 0
        while self.feature[node] >= 0:
            if data.bin_at(row, self.feature[node]) <= self.threshold_bin[node]:
                node = self.left[node]
            else:
                node = self.right[node]
        return node


struct _LeafState(Movable):
    """A grown-but-unsplit leaf: its node id, rows, histogram, and the best
    split available from it."""

    var node: Int
    var rows: List[Int]
    var hist: Histogram
    var split: SplitInfo

    def __init__(
        out self,
        node: Int,
        var rows: List[Int],
        var hist: Histogram,
        var split: SplitInfo,
    ):
        self.node = node
        self.rows = rows^
        self.hist = hist^
        self.split = split^


def _leaf_value(hist: Histogram, lambda_reg: Float64) -> Float64:
    # Totals over feature 0's bins (every feature's bins sum to the same
    # totals; feature 0 is at offset 0).
    var g = 0.0
    var h = 0.0
    for b in range(hist.n_bins):
        g += hist.grad[b]
        h += hist.hess[b]
    return -g / (h + lambda_reg)


def _search(hist: Histogram, n_rows: Int, params: TreeParams) -> SplitInfo:
    if n_rows < 2 * params.min_data_in_leaf or n_rows < 2:
        return SplitInfo(-1, -1, 0.0, False)
    return find_best_split(
        hist,
        lambda_reg=params.lambda_reg,
        min_child_hess=params.min_child_hess,
        min_data_in_leaf=params.min_data_in_leaf,
    )


def grow_tree(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
) raises -> Tree:
    """Grow one tree, leaf-wise, on the full dataset."""
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )

    var all_rows = List[Int](capacity=data.n_rows)
    for r in range(data.n_rows):
        all_rows.append(r)

    var root_hist = build_histogram(data, grad, hess)
    var root_split = _search(root_hist, data.n_rows, params)
    var root = tree._add_node(_leaf_value(root_hist, params.lambda_reg))

    var frontier = List[_LeafState]()
    frontier.append(_LeafState(root, all_rows^, root_hist^, root_split^))
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

        # Partition the parent's rows by the chosen split.
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        for i in range(len(frontier[best_i].rows)):
            var r = frontier[best_i].rows[i]
            if data.bin_at(r, split.feature) <= split.bin:
                left_rows.append(r)
            else:
                right_rows.append(r)

        # Histogram subtraction trick: build the smaller child directly.
        var left_hist: Histogram
        var right_hist: Histogram
        if len(left_rows) <= len(right_rows):
            left_hist = build_histogram_subset(data, grad, hess, left_rows)
            right_hist = subtract_histogram(frontier[best_i].hist, left_hist)
        else:
            right_hist = build_histogram_subset(data, grad, hess, right_rows)
            left_hist = subtract_histogram(frontier[best_i].hist, right_hist)

        var left_node = tree._add_node(
            _leaf_value(left_hist, params.lambda_reg)
        )
        var right_node = tree._add_node(
            _leaf_value(right_hist, params.lambda_reg)
        )
        tree.feature[parent_node] = split.feature
        tree.threshold_bin[parent_node] = split.bin
        tree.left[parent_node] = left_node
        tree.right[parent_node] = right_node
        tree.split_gain[parent_node] = split.gain

        var left_split = _search(left_hist, len(left_rows), params)
        var right_split = _search(right_hist, len(right_rows), params)

        frontier[best_i] = _LeafState(
            left_node, left_rows^, left_hist^, left_split^
        )
        frontier.append(
            _LeafState(right_node, right_rows^, right_hist^, right_split^)
        )
        n_leaves += 1

    tree.n_leaves = n_leaves
    return tree^
