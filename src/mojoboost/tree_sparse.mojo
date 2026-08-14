"""Leaf-wise tree growth on sparse data.

Same algorithm as `tree.mojo` (best-first growth, Newton leaf values,
sibling histogram subtraction) over the sparse accumulator. The two
sparse-specific pieces are:

- entries stay grouped by node in a shared permutation
  (`SparseEntryOrder`), so a node's histogram costs O(nnz_in_node) and a
  split costs one in-place partition of the parent's entry ranges;
- the split itself is applied to rows through the split feature's stored
  entries: every row of the node takes the side of `default_bin[f]` unless
  it has a stored entry for that feature.

Growth returns the row-to-leaf assignment alongside the tree, which lets the
boosting loop update raw scores in O(n_rows) without walking the tree, and
lets LightGBM-style leaf renewal group residuals without random access into
the sparse matrix.

Results do not depend on the number of workers: partitioning preserves entry
order within each child, and accumulation is per-feature disjoint.
"""

from .histogram import Histogram, subtract_histogram
from .histogram_sparse import (
    NodeTotals,
    SparseEntryOrder,
    SparseNodeEntries,
    build_histogram_sparse_node,
    sum_all,
    sum_rows,
)
from .sparse import SparseBinnedMatrix, SparseBinnedRows
from .split import SplitInfo
from .tree import Tree, TreeParams, _leaf_value, _search


@fieldwise_init
struct SparseTreeResult(Copyable, Movable):
    """A grown tree plus the leaf node index of every training row."""

    var tree: Tree
    var row_leaf: List[Int]


struct _SparseLeafState(Movable):
    var node: Int
    var rows: List[Int]
    var entries: SparseNodeEntries
    var hist: Histogram
    var split: SplitInfo

    def __init__(
        out self,
        node: Int,
        var rows: List[Int],
        var entries: SparseNodeEntries,
        var hist: Histogram,
        var split: SplitInfo,
    ):
        self.node = node
        self.rows = rows^
        self.entries = entries^
        self.hist = hist^
        self.split = split^


def predict_row_sparse(
    tree: Tree, data: SparseBinnedRows, row: Int
) -> Float64:
    """Tree output for one row of a row-oriented sparse binned matrix."""
    var node = 0
    while tree.feature[node] >= 0:
        if data.bin_at(row, tree.feature[node]) <= tree.threshold_bin[node]:
            node = tree.left[node]
        else:
            node = tree.right[node]
    return tree.value[node]


def grow_tree_sparse(
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
) raises -> SparseTreeResult:
    """Grow one tree, leaf-wise, on the full sparse dataset."""
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")

    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )
    var n_features = data.n_features
    var order = SparseEntryOrder(data.nnz())

    var all_rows = List[Int](capacity=data.n_rows)
    for r in range(data.n_rows):
        all_rows.append(r)

    var root_entries = SparseNodeEntries.root(data)
    var root_hist = build_histogram_sparse_node(
        data, grad, hess, order, root_entries, sum_all(grad, hess)
    )
    var root_split = _search(root_hist, data.n_rows, params)
    var root = tree._add_node(
        _leaf_value(root_hist, params.lambda_reg), Float64(data.n_rows)
    )

    var frontier = List[_SparseLeafState]()
    frontier.append(
        _SparseLeafState(
            root, all_rows^, root_entries^, root_hist^, root_split^
        )
    )
    var n_leaves = 1

    # 1 = the row goes left, 0 = right. Only the rows of the leaf being split
    # are read, and they are all written first, so one buffer serves every
    # split.
    var row_side = List[UInt8](capacity=data.n_rows)
    row_side.resize(data.n_rows, 0)

    while n_leaves < params.num_leaves:
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

        # Rows without a stored entry for the split feature carry its
        # implicit zero and follow default_bin.
        var default_left: UInt8 = 1 if Int(
            data.default_bin[split.feature]
        ) <= split.bin else 0
        for i in range(len(frontier[best_i].rows)):
            row_side[frontier[best_i].rows[i]] = default_left
        for i in range(
            frontier[best_i].entries.starts[split.feature],
            frontier[best_i].entries.ends[split.feature],
        ):
            var e = order.order[i]
            row_side[data.row_index[e]] = (
                1 if Int(data.bin[e]) <= split.bin else 0
            )

        var left_rows = List[Int]()
        var right_rows = List[Int]()
        for i in range(len(frontier[best_i].rows)):
            var r = frontier[best_i].rows[i]
            if row_side[r] != 0:
                left_rows.append(r)
            else:
                right_rows.append(r)

        var left_entries = SparseNodeEntries.empty(n_features)
        var right_entries = SparseNodeEntries.empty(n_features)
        order.partition(
            data,
            frontier[best_i].entries,
            row_side,
            left_entries,
            right_entries,
        )

        # Build the smaller child directly, derive its sibling by subtraction.
        var left_hist: Histogram
        var right_hist: Histogram
        if len(left_rows) <= len(right_rows):
            left_hist = build_histogram_sparse_node(
                data,
                grad,
                hess,
                order,
                left_entries,
                sum_rows(grad, hess, left_rows),
            )
            right_hist = subtract_histogram(frontier[best_i].hist, left_hist)
        else:
            right_hist = build_histogram_sparse_node(
                data,
                grad,
                hess,
                order,
                right_entries,
                sum_rows(grad, hess, right_rows),
            )
            left_hist = subtract_histogram(frontier[best_i].hist, right_hist)

        var left_node = tree._add_node(
            _leaf_value(left_hist, params.lambda_reg),
            Float64(len(left_rows)),
        )
        var right_node = tree._add_node(
            _leaf_value(right_hist, params.lambda_reg),
            Float64(len(right_rows)),
        )
        tree.feature[parent_node] = split.feature
        tree.threshold_bin[parent_node] = split.bin
        tree.left[parent_node] = left_node
        tree.right[parent_node] = right_node
        tree.split_gain[parent_node] = split.gain

        var left_split = _search(left_hist, len(left_rows), params)
        var right_split = _search(right_hist, len(right_rows), params)

        frontier[best_i] = _SparseLeafState(
            left_node, left_rows^, left_entries^, left_hist^, left_split^
        )
        frontier.append(
            _SparseLeafState(
                right_node,
                right_rows^,
                right_entries^,
                right_hist^,
                right_split^,
            )
        )
        n_leaves += 1

    tree.n_leaves = n_leaves

    var row_leaf = List[Int](capacity=data.n_rows)
    row_leaf.resize(data.n_rows, 0)
    for i in range(len(frontier)):
        for j in range(len(frontier[i].rows)):
            row_leaf[frontier[i].rows[j]] = frontier[i].node
    return SparseTreeResult(tree^, row_leaf^)
