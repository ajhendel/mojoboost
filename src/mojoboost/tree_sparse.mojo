"""Leaf-wise tree growth on sparse data.

Same algorithm as `tree.mojo`: best-first growth, Newton leaf values with L1
soft-thresholding, sibling histogram subtraction, the `max_depth` bound,
interaction constraints, monotonic constraints, feature subsampling, row
bagging, and missing-value routing all behave identically, because the split
search, leaf-value, and constraint code are shared verbatim. Only the
accumulator differs. The mirroring follows the precedent set by
`grow_tree_gpu` in `train_gpu.mojo`.

Two pieces are sparse-specific:

- entries stay grouped by node in a shared permutation
  (`SparseEntryOrder`), so a node's histogram costs O(nnz_in_node) and a
  split costs one in-place partition of the parent's entry ranges;
- a split is applied to rows through the split feature's stored entries
  alone: every row of the node takes the side of `default_bin[f]` unless it
  has a stored entry for that feature.

Growth returns the row-to-leaf assignment alongside the tree. That lets the
boosting loop update raw scores in O(n_rows) without walking the tree, and
lets LightGBM-style leaf renewal group residuals without random access into
the sparse matrix. Rows the tree was not grown on (everything outside a
non-empty bag) are marked -1.

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
from .interaction import extend_branch
from .monotone import (
    MONOTONE_FREE,
    OutputBounds,
    child_bounds,
    midpoint,
    monotone_sign,
)
from .sampling import (
    check_feature_fractions,
    select_split_features,
    select_tree_features,
)
from .sparse import SparseBinnedMatrix, SparseBinnedRows
from .split import SplitInfo
from .tree import Tree, TreeParams, _leaf_value, _search


@fieldwise_init
struct SparseTreeResult(Copyable, Movable):
    """A grown tree plus, per training row, the node index of the leaf it
    lands in, or -1 for a row the tree was not grown on."""

    var tree: Tree
    var row_leaf: List[Int]


struct _SparseLeafState(Movable):
    """A grown-but-unsplit leaf. Mirrors `tree._LeafState`, with the node's
    per-feature entry ranges in place of nothing (the dense grower re-reads
    the matrix instead)."""

    var node: Int
    var rows: List[Int]
    var entries: SparseNodeEntries
    var hist: Histogram
    var split: SplitInfo
    var branch: List[Int]
    var depth: Int
    var bounds: OutputBounds

    def __init__(
        out self,
        node: Int,
        var rows: List[Int],
        var entries: SparseNodeEntries,
        var hist: Histogram,
        var split: SplitInfo,
        var branch: List[Int] = [],
        depth: Int = 0,
        var bounds: OutputBounds = OutputBounds.unbounded(),
    ):
        self.node = node
        self.rows = rows^
        self.entries = entries^
        self.hist = hist^
        self.split = split^
        self.branch = branch^
        self.depth = depth
        self.bounds = bounds^


def predict_row_sparse(
    tree: Tree, data: SparseBinnedRows, row: Int
) -> Float64:
    """Tree output for one row of a row-oriented sparse binned matrix. Each
    node's test binary-searches that row's own stored entries."""
    var node = 0
    while tree.feature[node] >= 0:
        if tree.goes_left(node, data.bin_at(row, tree.feature[node])):
            node = tree.left[node]
        else:
            node = tree.right[node]
    return tree.value[node]


def predict_row_sparse_csc(
    tree: Tree, data: SparseBinnedMatrix, row: Int
) -> Float64:
    """Tree output for one row of a column-oriented sparse binned matrix.
    Each node's test binary-searches the split feature's whole column, so
    prefer `predict_row_sparse` when predicting many rows."""
    var node = 0
    while tree.feature[node] >= 0:
        if tree.goes_left(node, data.bin_at(row, tree.feature[node])):
            node = tree.left[node]
        else:
            node = tree.right[node]
    return tree.value[node]


def _bag_entries(
    data: SparseBinnedMatrix,
    mut order: SparseEntryOrder,
    bag: List[Int],
    mut row_side: List[UInt8],
) raises -> SparseNodeEntries:
    """Root entry ranges restricted to the bagged rows.

    Partitions every feature's whole column into (in bag, out of bag) and
    keeps the in-bag side, which is exactly the root range the full-dataset
    path gets for free. O(nnz).
    """
    for r in range(data.n_rows):
        row_side[r] = 0
    for i in range(len(bag)):
        if bag[i] < 0 or bag[i] >= data.n_rows:
            raise Error("bag row index out of range")
        if row_side[bag[i]] != 0:
            raise Error("bag row indices must be unique on the sparse path")
        row_side[bag[i]] = 1
    var inside = SparseNodeEntries.empty(data.n_features)
    var outside = SparseNodeEntries.empty(data.n_features)
    order.partition(
        data, SparseNodeEntries.root(data), row_side, inside, outside
    )
    return inside^


def grow_tree_sparse(
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
) raises -> SparseTreeResult:
    """Grow one tree, leaf-wise, on sparse data.

    Arguments and semantics match `grow_tree`: a non-empty `bag` restricts
    growth to those rows (they must be unique here, which is what
    `bagging.mojo` and `goss.mojo` produce), and `tree_index` together with
    `params.feature_fraction_seed` fixes which features the tree and its
    nodes may split on.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    params.constraints.check_features(data.n_features)
    params.monotone.check_features(data.n_features)
    check_feature_fractions(
        params.feature_fraction,
        params.feature_fraction_bynode,
        params.feature_fraction_bylevel,
    )
    # This grower applies the whole `extra` bundle, so it validates it the way
    # the dense grower does: against this dataset, before the first histogram.
    params.extra.check(
        data.n_features,
        params.num_leaves,
        params.max_depth,
        params.min_data_in_leaf,
    )
    var max_delta_step = params.extra.max_delta_step
    var path_smooth = params.extra.path_smooth
    var signs = params.monotone.active_signs()
    var tree_features = select_tree_features(
        data.n_features,
        params.feature_fraction,
        params.feature_fraction_seed,
        tree_index,
    )
    var value_feature = tree_features[0]

    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )
    var n_features = data.n_features
    var order = SparseEntryOrder(data.nnz())

    # 1 = the row goes left, 0 = right. Only the rows of the node being split
    # are read, and they are all written first, so one buffer serves every
    # split (and the initial bag partition).
    var row_side = List[UInt8](capacity=data.n_rows)
    row_side.resize(data.n_rows, 0)

    var root_rows: List[Int]
    var root_entries: SparseNodeEntries
    var root_totals: NodeTotals
    if len(bag) == 0:
        root_rows = List[Int](capacity=data.n_rows)
        for r in range(data.n_rows):
            root_rows.append(r)
        root_entries = SparseNodeEntries.root(data)
        root_totals = sum_all(grad, hess)
    else:
        root_entries = _bag_entries(data, order, bag, row_side)
        root_rows = bag.copy()
        root_totals = sum_rows(grad, hess, bag)

    var root_hist = build_histogram_sparse_node(
        data, grad, hess, order, root_entries, root_totals, tree_features
    )
    var root_branch = List[Int]()
    # The root's value comes before its search, because path smoothing makes a
    # candidate's children shrink toward it; the root has no parent and so
    # smooths toward 0.0. Same ordering as `tree.grow_tree`.
    var root = tree._add_node(
        _leaf_value(
            root_hist,
            params.lambda_reg,
            params.lambda_l1,
            value_feature,
            len(root_rows),
            0.0,
            max_delta_step,
            path_smooth,
        ),
        Float64(len(root_rows)),
    )
    var root_split = _search(
        root_hist,
        len(root_rows),
        params,
        params.constraints.allowed_features(root_branch),
        select_split_features(
            tree_features,
            params.feature_fraction_bylevel,
            params.feature_fraction_bynode,
            params.feature_fraction_seed,
            tree_index,
            0,
            0,
        ),
        depth=0,
        missing_bins=data.missing_bin,
        monotone=signs,
        cats=data.cats,
        node=root,
        tree_index=tree_index,
        parent_output=tree.value[root],
        grower_applies_extra=True,
    )

    var frontier = List[_SparseLeafState]()
    frontier.append(
        _SparseLeafState(
            root,
            root_rows^,
            root_entries^,
            root_hist^,
            root_split^,
            root_branch^,
            depth=0,
        )
    )
    var n_leaves = 1

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
        var split_missing_bin = data.missing_bin[split.feature]

        # Rows with no stored entry for the split feature carry its implicit
        # zero, so they all take default_bin's side; the feature's stored
        # entries in this node then override their own rows. Missing rows
        # follow the split's default direction, as in the dense grower.
        var default_bin = Int(data.default_bin[split.feature])
        var default_left: UInt8
        if default_bin == split_missing_bin:
            default_left = 1 if split.default_left else 0
        else:
            default_left = 1 if split.goes_left(default_bin) else 0
        for i in range(len(frontier[best_i].rows)):
            row_side[frontier[best_i].rows[i]] = default_left
        for i in range(
            frontier[best_i].entries.starts[split.feature],
            frontier[best_i].entries.ends[split.feature],
        ):
            var e = order.order[i]
            var bin = Int(data.bin[e])
            var go_left: Bool
            if bin == split_missing_bin:
                go_left = split.default_left
            else:
                go_left = split.goes_left(bin)
            row_side[data.row_index[e]] = 1 if go_left else 0

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

        # Histogram subtraction trick: build the smaller child directly.
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
                tree_features,
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
                tree_features,
            )
            left_hist = subtract_histogram(frontier[best_i].hist, right_hist)

        var parent_bounds = frontier[best_i].bounds.copy()
        var split_sign = monotone_sign(signs, split.feature)
        # Cap and smoothing first, monotone interval on the result, the order
        # the candidate was scored with. Both children smooth toward the value
        # the parent already emits.
        var parent_output = tree.value[parent_node]
        var left_value = parent_bounds.clamp(
            _leaf_value(
                left_hist,
                params.lambda_reg,
                params.lambda_l1,
                value_feature,
                len(left_rows),
                parent_output,
                max_delta_step,
                path_smooth,
            )
        )
        var right_value = parent_bounds.clamp(
            _leaf_value(
                right_hist,
                params.lambda_reg,
                params.lambda_l1,
                value_feature,
                len(right_rows),
                parent_output,
                max_delta_step,
                path_smooth,
            )
        )
        if split_sign != MONOTONE_FREE and left_value > right_value:
            var mid = midpoint(left_value, right_value)
            left_value = mid
            right_value = mid
        var children = child_bounds(
            parent_bounds, split_sign, left_value, right_value
        )
        var left_node = tree._add_node(left_value, Float64(len(left_rows)))
        var right_node = tree._add_node(right_value, Float64(len(right_rows)))
        tree.left[parent_node] = left_node
        tree.right[parent_node] = right_node
        # Routing is recorded through the shared helper, so a sparse-grown
        # tree carries exactly the node layout a dense-grown one does.
        tree._set_split(parent_node, split, split_missing_bin)

        var branch = extend_branch(frontier[best_i].branch, split.feature)
        var allowed = params.constraints.allowed_features(branch)
        var child_depth = frontier[best_i].depth + 1
        var left_split = _search(
            left_hist,
            len(left_rows),
            params,
            allowed,
            select_split_features(
                tree_features,
                params.feature_fraction_bylevel,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                child_depth,
                left_node,
            ),
            depth=child_depth,
            missing_bins=data.missing_bin,
            monotone=signs,
            bounds=children.left.copy(),
            cats=data.cats,
            node=left_node,
            tree_index=tree_index,
            parent_output=left_value,
            grower_applies_extra=True,
        )
        var right_split = _search(
            right_hist,
            len(right_rows),
            params,
            allowed,
            select_split_features(
                tree_features,
                params.feature_fraction_bylevel,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                child_depth,
                right_node,
            ),
            depth=child_depth,
            missing_bins=data.missing_bin,
            monotone=signs,
            bounds=children.right.copy(),
            cats=data.cats,
            node=right_node,
            tree_index=tree_index,
            parent_output=right_value,
            grower_applies_extra=True,
        )

        frontier[best_i] = _SparseLeafState(
            left_node,
            left_rows^,
            left_entries^,
            left_hist^,
            left_split^,
            branch.copy(),
            depth=child_depth,
            bounds=children.left.copy(),
        )
        frontier.append(
            _SparseLeafState(
                right_node,
                right_rows^,
                right_entries^,
                right_hist^,
                right_split^,
                branch^,
                depth=child_depth,
                bounds=children.right.copy(),
            )
        )
        n_leaves += 1

    tree.n_leaves = n_leaves

    var row_leaf = List[Int](capacity=data.n_rows)
    row_leaf.resize(data.n_rows, -1)
    for i in range(len(frontier)):
        for j in range(len(frontier[i].rows)):
            row_leaf[frontier[i].rows[j]] = frontier[i].node
    return SparseTreeResult(tree^, row_leaf^)
