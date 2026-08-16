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

Exclusive feature bundling
--------------------------
Optional, off by default, and a histogram layout rather than a change of
hypothesis space (see efb.mojo). With an active plan the matrix handed to
`grow_tree_sparse` is the *bundled* one, and the accumulation half works
exactly as it does on the dense path: a node's histogram is built per bundle
column and then expanded back into one slice per original feature
(`_node_histogram`, on top of `efb.expand_bundled_histogram`), so nothing
below that point -- split search, leaf values, sibling subtraction, the
`Tree` itself -- ever sees a bundle. Splits therefore name original features
and original bins.

Where the sparse path differs from the dense one is in applying that split.
`tree.grow_tree` keeps both matrices and partitions rows on the original;
the sparse grower keeps only the bundled matrix, because holding the
original CSC too would give back the memory bundling saves. So it routes
rows through the bundle column and decodes each stored entry to the split
feature's own local bin (`SparseBundling.local_bin`). A row sitting in a bin
that belongs to another member of the column is a row where the split
feature is at its default -- which is precisely what the expansion folded
into that feature's default bin when it recovered the histogram this split
was chosen from -- so routing and accumulation agree bin for bin. Since only
lossless plans are currently constructible, they agree with the unbundled
matrix too, and a bundled fit produces the tree an unbundled fit produces.
"""

from .categorical import CategoricalSpec
from .efb import (
    EFB_NONE,
    FeatureBundling,
    columns_for_features,
    expand_bundled_histogram,
)
from .histogram import ConstHessianSettings, Histogram, subtract_histogram
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
from .growth_policy import GrowthSchedule, LeafCandidate, check_grow_policy
from .tree import Tree, TreeParams, _leaf_value, _search


@fieldwise_init
struct SparseTreeResult(Copyable, Movable):
    """A grown tree plus, per training row, the node index of the leaf it
    lands in, or -1 for a row the tree was not grown on."""

    var tree: Tree
    var row_leaf: List[Int]


@fieldwise_init
struct SparseBundling(Copyable, Movable):
    """A bundling plan resolved against the bundled matrix it produced.

    This is the sparse counterpart of `efb.BundledMatrix`, and it differs
    from it in one way that decides the whole design. The dense grower keeps
    *both* matrices: it accumulates histograms from the bundled one and
    partitions rows on the original one, reading the original bins directly.
    The sparse grower keeps only the bundled matrix, because holding the
    original CSC as well would give back exactly the memory bundling was
    meant to save. So a sparse split has to route rows through the bundle
    column, and this is what makes that exact:

        column(f)             the bundle column feature f lives in
        local_bin(f, b)       f's own local bin for a row sitting in bundle
                              bin b of that column

    `local_bin` is the whole trick. A bundle bin belongs to at most one
    member; a row in a bin belonging to some *other* member, or in the
    shared bin 0, is a row where f takes its default bin, which is exactly
    what `efb.unbundle_histogram` folds into f's default when it recovers
    f's histogram. Routing and accumulation therefore agree bin for bin, and
    with the lossless plans `efb.check_bundling_params` currently allows
    (`max_conflict_rate` must be 0.0) they agree with the *unbundled* matrix
    too: a bundled sparse fit produces the tree an unbundled one produces.

    Everything else here is original-feature-space metadata that the split
    search needs and that a bundled matrix no longer carries per feature:

        missing[f]    f's own local missing bin, or -1
        default[f]    f's own local default bin
        cats          the categorical spec indexed by original feature

    They are derived from the plan and the bundled matrix rather than passed
    in, because `efb.mojo` guarantees what makes that possible: a
    categorical feature is always a singleton bundle under identity
    encoding, so its category table survives bundling unchanged.

    Two predicates, and they mean different things. `active` is "bundling is
    on"; `resolved()` is "this view was built against a matrix". The
    inactive-but-resolved view is what a non-bundled fit uses, and it makes
    `grow_tree_sparse` one code path rather than two: `column` is the
    identity, `local_bin` is the identity, and the metadata is read off the
    matrix. `none()` is the unresolved value a *predictor* takes as its
    default, where only the two mappings are ever read.
    """

    var plan: FeatureBundling
    var active: Bool
    var n_features: Int
    """Original feature count. 0 marks an unresolved view (`none()`)."""
    var n_columns: Int
    """Columns of the matrix this view was resolved against: the bundle
    count when active, the feature count when not."""
    var missing: List[Int]
    var default: List[Int]
    var cats: CategoricalSpec
    var slot_at: List[Int]
    """Flat `(column, column bin) -> owning member slot`, `EFB_NONE` for a
    multi-member bundle's shared bin. Empty when inactive."""
    var local_at: List[Int]
    """Flat `(column, column bin) -> that member's local bin`, `EFB_NONE`
    for the shared bin. Empty when inactive."""
    var bin_offsets: List[Int]
    """Per-column offsets into `slot_at` / `local_at`. Empty when inactive."""

    @staticmethod
    def none() -> SparseBundling:
        """The unresolved, inactive view. `column` and `local_bin` are the
        identity on it, which is all a predictor on an unbundled matrix
        needs; the metadata lists are empty and must not be read."""
        return SparseBundling(
            FeatureBundling.none(),
            False,
            0,
            0,
            List[Int](),
            List[Int](),
            CategoricalSpec.none(),
            List[Int](),
            List[Int](),
            List[Int](),
        )

    @staticmethod
    def of(
        plan: FeatureBundling, data: SparseBinnedMatrix
    ) raises -> SparseBundling:
        """Resolve `plan` against the matrix it produced.

        `data` must be the *bundled* matrix when the plan is active, which
        is what the three shape checks below enforce: pairing a plan with
        the matrix it was fitted on rather than the one it produced would
        decode every bin against the wrong column widths, and would do it
        silently.
        """
        if not plan.active():
            var missing = data.missing_bin.copy()
            var default = List[Int](capacity=data.n_features)
            for f in range(data.n_features):
                default.append(Int(data.default_bin[f]))
            return SparseBundling(
                FeatureBundling.none(),
                False,
                data.n_features,
                data.n_features,
                missing^,
                default^,
                data.cats.copy(),
                List[Int](),
                List[Int](),
                List[Int](),
            )

        plan.validate()
        if plan.n_bundles() != data.n_features:
            raise Error(
                "a bundling plan must be resolved against the bundled"
                " matrix: it has one column per bundle"
            )
        if plan.n_rows != data.n_rows:
            raise Error("bundling plan and matrix disagree on n_rows")
        if plan.max_bundle_bins() > data.n_bins:
            raise Error(
                "bundled matrix is narrower than the plan's widest bundle"
            )

        var n_orig = plan.n_features
        var missing = List[Int](capacity=n_orig)
        var default = List[Int](capacity=n_orig)
        var flags = List[Bool](capacity=n_orig)
        var codes = List[Int]()
        var offsets = List[Int](capacity=n_orig + 1)
        offsets.append(0)
        for f in range(n_orig):
            var slot = plan.slot_of[f]
            missing.append(plan.slot_missing[slot])
            default.append(plan.slot_default[slot])
            # A categorical feature is a singleton bundle under identity
            # encoding (efb.mojo guarantees it unconditionally), so its
            # column *is* its own column and its table transfers verbatim.
            # A bundled column can therefore never be categorical, and this
            # loop rebuilds the original-space spec exactly.
            var column = plan.bundle_of[f]
            var is_cat = plan.bundle_size(column) == 1 and data.cats.is_cat(
                column
            )
            flags.append(is_cat)
            if is_cat:
                for i in range(
                    data.cats.offsets[column], data.cats.offsets[column + 1]
                ):
                    codes.append(data.cats.codes[i])
            offsets.append(len(codes))

        var bin_offsets = List[Int](capacity=data.n_features + 1)
        bin_offsets.append(0)
        var slot_at = List[Int]()
        var local_at = List[Int]()
        for column in range(data.n_features):
            for b in range(plan.bundle_bins[column]):
                var slot = plan.slot_containing(column, b)
                slot_at.append(slot)
                if slot == EFB_NONE:
                    local_at.append(EFB_NONE)
                else:
                    local_at.append(plan.decode_bin(column, b))
            bin_offsets.append(len(slot_at))

        return SparseBundling(
            plan.copy(),
            True,
            n_orig,
            data.n_features,
            missing^,
            default^,
            CategoricalSpec(flags^, codes^, offsets^),
            slot_at^,
            local_at^,
            bin_offsets^,
        )

    def resolved(self) -> Bool:
        """Whether this view was built against a matrix, and so carries the
        original-space metadata a grower reads."""
        return self.n_features > 0

    def check_matrix(self, data: SparseBinnedMatrix) raises:
        """Reject a view resolved against a different matrix."""
        if not self.resolved():
            raise Error("bundling view was never resolved against a matrix")
        if self.n_columns != data.n_features:
            raise Error("bundling view and matrix disagree on column count")
        if self.active and self.plan.n_rows != data.n_rows:
            raise Error("bundling plan and matrix disagree on n_rows")

    def source_bins(self, data: SparseBinnedMatrix) -> Int:
        """The bin count of the matrix the *original* features were binned
        into.

        A bundled matrix is only as wide as the widest bundle, so its
        `n_bins` is not the width a second matrix binned by the same mapper
        would have. A trainer comparing a validation matrix against its
        training matrix has to compare against this instead.
        """
        if not self.active:
            return data.n_bins
        return self.plan.n_bins

    def column(self, feature: Int) -> Int:
        """The matrix column `feature`'s entries live in."""
        if not self.active:
            return feature
        return self.plan.bundle_of[feature]

    def local_bin(self, feature: Int, column_bin: Int) -> Int:
        """`feature`'s own local bin for a row sitting in `column_bin` of its
        column.

        A bin owned by another member, the shared bin, and a bin past the
        column's width all resolve to `feature`'s default bin: in every one
        of those cases the row stores nothing for this feature, which is the
        definition of it being at its default.
        """
        if not self.active:
            return column_bin
        var column = self.plan.bundle_of[feature]
        var base = self.bin_offsets[column]
        var width = self.bin_offsets[column + 1] - base
        if column_bin < 0 or column_bin >= width:
            return self.default[feature]
        if self.slot_at[base + column_bin] != self.plan.slot_of[feature]:
            return self.default[feature]
        return self.local_at[base + column_bin]

    def local_table(self, feature: Int, n_bins: Int) -> List[Int]:
        """`local_bin` for every bin of `feature`'s column, as a table.

        Built once per split and read once per stored entry, so routing a
        bundled split costs one table lookup per entry -- the same as the
        direct bin read it replaces. At most 256 entries.
        """
        var out = List[Int](capacity=n_bins)
        for b in range(n_bins):
            out.append(self.local_bin(feature, b))
        return out^

    def columns_for(self, features: List[Int]) raises -> List[Int]:
        """The matrix columns a set of original features occupies.

        A column has to be accumulated when *any* of its members was picked
        by feature subsampling; the extra members' statistics ride along in
        it and are never scanned, because `_node_histogram` expands only the
        picked features out of it.
        """
        if not self.active:
            return features.copy()
        return columns_for_features(self.plan, features)


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
    tree: Tree,
    data: SparseBinnedRows,
    row: Int,
    bundling: SparseBundling = SparseBundling.none(),
) -> Float64:
    """Tree output for one row of a row-oriented sparse binned matrix. Each
    node's test binary-searches that row's own stored entries.

    A grown tree names original features and original bins whether or not it
    was grown on a bundled matrix, so `bundling` is needed only when `data`
    *is* the bundled matrix: it says which column to look the feature up in
    and how to read the bin back. Predicting an unbundled matrix leaves it
    at `none()`, which is the identity on both.
    """
    var node = 0
    while tree.feature[node] >= 0:
        var feature = tree.feature[node]
        var column_bin = data.bin_at(row, bundling.column(feature))
        if tree.goes_left(node, bundling.local_bin(feature, column_bin)):
            node = tree.left[node]
        else:
            node = tree.right[node]
    return tree.value[node]


def predict_row_sparse_csc(
    tree: Tree,
    data: SparseBinnedMatrix,
    row: Int,
    bundling: SparseBundling = SparseBundling.none(),
) -> Float64:
    """Tree output for one row of a column-oriented sparse binned matrix.
    Each node's test binary-searches the split feature's whole column, so
    prefer `predict_row_sparse` when predicting many rows. `bundling` carries
    the meaning it has there."""
    var node = 0
    while tree.feature[node] >= 0:
        var feature = tree.feature[node]
        var column_bin = data.bin_at(row, bundling.column(feature))
        if tree.goes_left(node, bundling.local_bin(feature, column_bin)):
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


def _node_histogram(
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    mut order: SparseEntryOrder,
    entries: SparseNodeEntries,
    totals: NodeTotals,
    bundles: SparseBundling,
    features: List[Int],
    columns: List[Int],
    narrow: Bool = True,
) raises -> Histogram:
    """One node's histogram, always in the original per-feature shape.

    `narrow` is `histogram.derivative_precision`, and it has to be the same
    value `totals` was summed at: the leftover assigned to the default bin is
    `totals` minus what the stored entries accumulated, so a mismatch would
    put the whole node's rounding difference into one bin
    (`histogram_sparse.build_histogram_sparse_node` states the pairing). Every
    caller below takes both from the same `narrow` local for that reason.

    Without bundling that is one call to the sparse accumulator over the
    node's entry ranges. With it, the accumulation runs over the bundle
    columns -- O(nnz_in_node) either way, but over as many columns as there
    are bundles rather than features, which is the saving -- and the result
    is expanded straight back into per-feature shape. That expansion is
    where a bundle stops existing: `_leaf_value`, `_search`, and
    `subtract_histogram` below all read the shape they always read.

    Sibling subtraction survives it because the expansion is linear:
    expanding a parent and a child and subtracting gives the numbers
    subtracting first and expanding the difference would.
    """
    var raw = build_histogram_sparse_node(
        data, grad, hess, order, entries, totals, columns, narrow
    )
    if not bundles.active:
        return raw^
    var out = Histogram.zeroed(bundles.n_features, bundles.source_bins(data))
    expand_bundled_histogram(
        out._grad,
        out._hess,
        out._count,
        out.n_bins,
        bundles.plan,
        raw._grad,
        raw._hess,
        raw._count,
        raw.n_bins,
        features,
    )
    return out^


def grow_tree_sparse(
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
    bundling: SparseBundling = SparseBundling.none(),
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
) raises -> SparseTreeResult:
    """Grow one tree on sparse data, leaf-wise by default or depth-wise
    under `params.grow_policy == GROW_DEPTHWISE` (the same `GrowthSchedule`
    order the dense grower follows, so both growers make the same choice).

    `const_hessian_env` is the fit's histogram snapshot
    (`histogram.ConstHessianSettings`), and it is here for one field:
    `narrow`, which is `derivative_precision`. **Before it existed this
    grower passed the `narrow = True` default to every sparse builder and
    every node total, so a sparse fit ignored
    `MOJOTREES_DERIVATIVE_PRECISION=float64` at the histogram while the
    objective honored it** -- the objective stopped narrowing, the grower
    re-narrowed, and a GOSS or weighted round accumulated `Float32(w * g)`
    where the arm the caller asked for accumulates `w * g`. That is the
    accepted-then-ignored failure this argument closes; the semantics now
    match `tree.grow_tree_leaves_profiled` exactly, including the sentinel
    resolving once per tree and the `derivative_precision` parameter being
    folded on top of whichever snapshot we end up with.

    The two constant-hessian fields are carried but unused here: the sparse
    accumulator has no three-plane elision to switch off, so a resolved
    snapshot and the sentinel choose the same kernel. They are on the type
    and threading them costs nothing, and a sparse const-hessian elision
    would find them already in place.

    Arguments and semantics match `grow_tree`: a non-empty `bag` restricts
    growth to those rows (they must be unique here, which is what
    `bagging.mojo` and `goss.mojo` produce), and `tree_index` together with
    `params.feature_fraction_seed` fixes which features the tree and its
    nodes may split on.

    `bundling` is an exclusive-feature-bundling plan resolved against `data`
    (see `SparseBundling`). With an active one, `data` is the *bundled*
    matrix: histograms are accumulated per bundle column, which is the whole
    point, and `_node_histogram` expands each one back into the original
    per-feature shape before anything reads it. Everything on this side of
    that expansion stays in the original feature space -- the feature
    sample, the interaction and monotonic constraints, the missing-bin
    table, the categorical spec, the split search, the leaf values, and the
    `SplitInfo` that comes back -- so **the tree that comes out names
    original features and original bins** and no consumer of it ever sees a
    bundle id. An unresolved view (the default) is resolved here against
    `data` as an inactive one, which makes the bundled and unbundled paths
    one path rather than two.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")

    var bundles: SparseBundling
    if bundling.resolved():
        bundling.check_matrix(data)
        bundles = bundling.copy()
    else:
        bundles = SparseBundling.of(FeatureBundling.none(), data)
    # Two feature counts from here on, and mixing them is the one mistake
    # this grower can make: `n_features` is the original space every
    # parameter, constraint, and emitted split is indexed by, and
    # `n_columns` is the matrix's own, which only the entry bookkeeping and
    # the histogram layout use. They are equal when bundling is off.
    var n_features = bundles.n_features
    var n_columns = data.n_features

    check_grow_policy(params.grow_policy)
    params.constraints.check_features(n_features)
    params.monotone.check_features(n_features)
    check_feature_fractions(
        params.feature_fraction,
        params.feature_fraction_bynode,
        params.feature_fraction_bylevel,
    )
    # This grower applies the whole `extra` bundle, so it validates it the way
    # the dense grower does: against this dataset, before the first histogram.
    params.extra.check(
        n_features,
        params.num_leaves,
        params.max_depth,
        params.min_data_in_leaf,
    )
    var max_delta_step = params.extra.max_delta_step
    var path_smooth = params.extra.path_smooth
    var signs = params.monotone.active_signs()
    var tree_features = select_tree_features(
        n_features,
        params.feature_fraction,
        params.feature_fraction_seed,
        tree_index,
    )
    # Subsampling picks original features; accumulation is by column. A
    # column is accumulated when any of its members was picked.
    var tree_columns = bundles.columns_for(tree_features)
    # Any accumulated feature answers the node's totals, since every row
    # occupies exactly one bin of every feature. It has to be one of the
    # accumulated ones, because the rest are left at zero. Histograms reach
    # this point already expanded, so this is an original feature id, the
    # same as the dense grower's.
    var value_feature = tree_features[0]

    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )
    var order = SparseEntryOrder(data.nnz())

    # 1 = the row goes left, 0 = right. Only the rows of the node being split
    # are read, and they are all written first, so one buffer serves every
    # split (and the initial bag partition).
    var row_side = List[UInt8](capacity=data.n_rows)
    row_side.resize(data.n_rows, 0)

    # The fit's histogram snapshot, resolved at most once for this tree, and
    # then the `derivative_precision` parameter folded on top of it. Same two
    # steps and the same precedence as `tree.grow_tree_leaves_profiled`:
    # `float64` wins from either the parameter or the environment.
    #
    # `narrow` is read out once here and passed to every builder and every
    # node total below, rather than each site reaching for the snapshot. That
    # is what makes it impossible for a total and its accumulation to
    # disagree, which is the one pairing the sparse builder requires.
    var const_h_env = const_hessian_env.copy()
    if not const_h_env.resolved:
        const_h_env = ConstHessianSettings.resolve()
    var narrow = const_h_env.widened(
        params.extra.wants_float64_derivatives()
    ).narrow

    var root_rows: List[Int]
    var root_entries: SparseNodeEntries
    var root_totals: NodeTotals
    if len(bag) == 0:
        root_rows = List[Int](capacity=data.n_rows)
        for r in range(data.n_rows):
            root_rows.append(r)
        root_entries = SparseNodeEntries.root(data)
        root_totals = sum_all(grad, hess, narrow)
    else:
        root_entries = _bag_entries(data, order, bag, row_side)
        root_rows = bag.copy()
        root_totals = sum_rows(grad, hess, bag, narrow)

    var root_hist = _node_histogram(
        data,
        grad,
        hess,
        order,
        root_entries,
        root_totals,
        bundles,
        tree_features,
        tree_columns,
        narrow,
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
        missing_bins=bundles.missing,
        monotone=signs,
        cats=bundles.cats,
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
    var schedule = GrowthSchedule(params.grow_policy)

    while n_leaves < params.num_leaves:
        # The growth policy picks (growth_policy.mojo): best gain anywhere in
        # the tree under leaf-wise growth, the planned level's next node
        # under depth-wise growth.
        var cands = List[LeafCandidate](capacity=len(frontier))
        for i in range(len(frontier)):
            cands.append(
                LeafCandidate(
                    frontier[i].node,
                    frontier[i].depth,
                    frontier[i].split.gain,
                    frontier[i].split.found and frontier[i].split.gain > 0.0,
                )
            )
        var best_i = schedule.next_leaf(
            cands, n_leaves, params.num_leaves, params.max_depth
        )
        if best_i < 0:
            break

        var parent_node = frontier[best_i].node
        var split = frontier[best_i].split.copy()
        # The split names an original feature and an original bin, so
        # everything it is applied through is original-space too: the
        # feature's own missing bin, its own default bin, and its own local
        # bin for whatever the column happens to store. Only the *column*
        # the entries are read from is a matrix coordinate.
        var split_column = bundles.column(split.feature)
        var split_missing_bin = bundles.missing[split.feature]
        var default_bin = bundles.default[split.feature]
        # Column bin -> this feature's local bin, once per split rather than
        # once per entry. Under bundling this is where a bin belonging to
        # another member of the column becomes "this feature is at its
        # default", which is exactly what it means and exactly what
        # `_node_histogram` folded into the default bin when the histogram
        # this split was chosen from was expanded.
        var local_of = bundles.local_table(split.feature, data.n_bins)

        # Rows with no stored entry for the split feature carry its implicit
        # zero, so they all take default_bin's side; the feature's stored
        # entries in this node then override their own rows. Missing rows
        # follow the split's default direction, as in the dense grower.
        var default_left: UInt8
        if default_bin == split_missing_bin:
            default_left = 1 if split.default_left else 0
        else:
            default_left = 1 if split.goes_left(default_bin) else 0
        for i in range(len(frontier[best_i].rows)):
            row_side[frontier[best_i].rows[i]] = default_left
        for i in range(
            frontier[best_i].entries.starts[split_column],
            frontier[best_i].entries.ends[split_column],
        ):
            var e = order.order[i]
            var bin = local_of[Int(data.bin[e])]
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

        var left_entries = SparseNodeEntries.empty(n_columns)
        var right_entries = SparseNodeEntries.empty(n_columns)
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
            left_hist = _node_histogram(
                data,
                grad,
                hess,
                order,
                left_entries,
                sum_rows(grad, hess, left_rows, narrow),
                bundles,
                tree_features,
                tree_columns,
                narrow,
            )
            right_hist = subtract_histogram(frontier[best_i].hist, left_hist)
        else:
            right_hist = _node_histogram(
                data,
                grad,
                hess,
                order,
                right_entries,
                sum_rows(grad, hess, right_rows, narrow),
                bundles,
                tree_features,
                tree_columns,
                narrow,
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
            missing_bins=bundles.missing,
            monotone=signs,
            bounds=children.left.copy(),
            cats=bundles.cats,
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
            missing_bins=bundles.missing,
            monotone=signs,
            bounds=children.right.copy(),
            cats=bundles.cats,
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
