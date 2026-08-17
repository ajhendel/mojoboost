"""Sparse histogram accumulation.

Produces the same `Histogram` the dense builders produce, so split finding,
the sibling subtraction trick, and leaf values are shared verbatim with the
dense path.

The accumulator never visits the implicit zeros. For each feature it walks
only that feature's stored entries in the node, accumulates them into their
bins, and assigns the leftover (node total minus the stored entries' sum) to
`default_bin[f]`, the bin holding 0.0. Cost per node is therefore
O(nnz_in_node) rather than O(n_rows_in_node * n_features).

Accumulation parallelizes across features exactly like the dense path: each
feature owns the `[f * n_bins, (f + 1) * n_bins)` output slice and its own
entry range, so workers never write the same location.

Every builder takes the same optional list of feature ids the dense builders
take (empty means all). Under feature subsampling only those features are
accumulated and the rest of the output stays zero, so sibling subtraction
stays exact on either representation. A repeated id in that list names the
same slice twice and is accumulated once, on the same rule the dense builders
follow; `_DUPLICATE_FEATURE_IDS` below carries what a surviving repeat would
do to the accumulation, and why the "workers never write the same location"
sentence above depends on the strip.

Numerical note: the default bin is derived by subtraction, so a sparse
histogram is not bit-identical to the dense histogram of the same data. The
two agree to floating-point rounding (counts agree exactly), the same
trade-off the dense path already makes for sibling subtraction.
"""

from .histogram import (
    Histogram,
    _check_features,
    _has_duplicate_features,
    _unique_features,
    derivative,
)
from .parallel import dispatch_features
from .sparse import SparseBinnedMatrix


# --- _DUPLICATE_FEATURE_IDS ------------------------------------------------
#
# What a repeated id in `features` does to the accumulators below, and why
# every entry point strips one before it reaches them.
#
# Both accumulators dispatch over *active slots* rather than over features:
# slot `i` resolves to `features[i]`, and its body accumulates that feature's
# entries into `features[i] * n_bins` and then folds the node's leftover into
# that feature's default bin. The output planes are allocated zeroed once and
# every write is a read-modify-write, so nothing is ever overwritten. A list
# holding one id `k` times therefore runs the same body `k` times over the
# same cells, and the feature's whole slice -- gradient, hessian and count
# alike -- comes out as exactly `k` times the truth. Its counts exceed the
# node's row count, so the histogram is not a histogram of anything.
#
# **The sparse shape is wrong at every `k >= 2`, unlike the dense one.** The
# dense kernels re-zero each lane's slice at the top of a feature group, so a
# repeat there yielded the *group width* times the truth and was accidentally
# right at a width of 1 (see `histogram._check_features`). Nothing here zeroes
# per feature, so there is no width at which the sparse answer is right and no
# machine on which the defect hides.
#
# And once the dispatch goes parallel it is worse than a factor. Two slots
# naming the same feature can land in different workers, and the docstring
# invariant at the top of this module -- "each feature owns the
# `[f * n_bins, (f + 1) * n_bins)` output slice ... so workers never write the
# same location" -- is exactly what a duplicate breaks. The reads and writes
# are unsynchronized, so the result is not even a stable multiple.
#
# `gpu_sparse.SparseDeviceLayout.set_active_features` describes this same
# arithmetic for the device mirror of these kernels, and refuses the list
# there rather than stripping it. The CPU builders strip because they are
# re-exported beside the dense builders, which strip; the device setter is an
# internal wiring point with no such neighbor.
# ---------------------------------------------------------------------------


@fieldwise_init
struct NodeTotals(Copyable, Movable):
    """Summed gradient, hessian, and row count over a node's rows."""

    var grad: Float64
    var hess: Float64
    var count: Int


def sum_rows(
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    narrow: Bool = True,
) -> NodeTotals:
    """`narrow` is `histogram.derivative_precision`; True is the default and
    the loop it selects is the one that shipped, instruction for
    instruction. Two loops rather than one loop with a test in it, which is
    the shape `histogram._accumulate_subset_at` uses for `compact` and for
    the same reason: a branch that is constant for the whole call has no
    business being re-evaluated once per row."""
    var g = 0.0
    var h = 0.0
    if narrow:
        for i in range(len(rows)):
            g += derivative[True](grad[rows[i]])
            h += derivative[True](hess[rows[i]])
    else:
        for i in range(len(rows)):
            g += derivative[False](grad[rows[i]])
            h += derivative[False](hess[rows[i]])
    return NodeTotals(g, h, len(rows))


def sum_all(
    grad: List[Float64], hess: List[Float64], narrow: Bool = True
) -> NodeTotals:
    """`sum_rows` over every row; `narrow` means what it means there."""
    var g = 0.0
    var h = 0.0
    if narrow:
        for r in range(len(grad)):
            g += derivative[True](grad[r])
            h += derivative[True](hess[r])
    else:
        for r in range(len(grad)):
            g += derivative[False](grad[r])
            h += derivative[False](hess[r])
    return NodeTotals(g, h, len(grad))


@fieldwise_init
struct SparseNodeEntries(Copyable, Movable):
    """One tree node's stored-entry range per feature, as offsets into the
    shared `SparseEntryOrder.order` array.

    Feature f's entries for this node are `order[starts[f] : ends[f]]`. A
    node's range for a feature is always a sub-range of its parent's, so the
    two children of a split partition the parent's range in place.
    """

    var starts: List[Int]
    var ends: List[Int]

    @staticmethod
    def root(data: SparseBinnedMatrix) -> SparseNodeEntries:
        var starts = List[Int](capacity=data.n_features)
        var ends = List[Int](capacity=data.n_features)
        for f in range(data.n_features):
            starts.append(data.col_offsets[f])
            ends.append(data.col_offsets[f + 1])
        return SparseNodeEntries(starts^, ends^)

    @staticmethod
    def empty(n_features: Int) -> SparseNodeEntries:
        var starts = List[Int](capacity=n_features)
        starts.resize(n_features, 0)
        var ends = List[Int](capacity=n_features)
        ends.resize(n_features, 0)
        return SparseNodeEntries(starts^, ends^)

    def n_entries(self) -> Int:
        var total = 0
        for f in range(len(self.starts)):
            total += self.ends[f] - self.starts[f]
        return total


struct SparseEntryOrder(Movable):
    """Permutation of stored-entry ids, grouped by tree node.

    Starts as the identity, which already groups entries by feature because
    that is the CSC layout. Splitting a node partitions each of its
    per-feature ranges in place so that the left child's entries come first;
    the children's ranges are then contiguous sub-ranges of the parent's.
    """

    var order: List[Int]
    var scratch: List[Int]

    def __init__(out self, nnz: Int):
        self.order = List[Int](capacity=nnz)
        for i in range(nnz):
            self.order.append(i)
        self.scratch = List[Int](capacity=nnz)
        self.scratch.resize(nnz, 0)

    def partition(
        mut self,
        data: SparseBinnedMatrix,
        node: SparseNodeEntries,
        row_side: List[UInt8],
        mut left: SparseNodeEntries,
        mut right: SparseNodeEntries,
    ) raises:
        """Split `node`'s entry ranges by `row_side` (1 = left, 0 = right).

        `row_side` must be set for every row of `node`; entries in `node`'s
        ranges only ever reference those rows. Order within each child is
        preserved, so row indices stay ascending and results do not depend on
        how the work was scheduled.
        """
        _partition_ranges(
            self.order,
            self.scratch,
            data,
            node,
            row_side,
            left.starts,
            left.ends,
            right.starts,
            right.ends,
        )


def _partition_ranges(
    mut order: List[Int],
    mut scratch: List[Int],
    data: SparseBinnedMatrix,
    node: SparseNodeEntries,
    row_side: List[UInt8],
    mut left_starts: List[Int],
    mut left_ends: List[Int],
    mut right_starts: List[Int],
    mut right_ends: List[Int],
) raises:
    """Stable in-place partition of every per-feature range of `node`.

    The children's four range arrays are passed as separate `mut` lists
    rather than reached through their `SparseNodeEntries`: a pointer taken
    from a struct field carries that field's origin, which a worker closure
    cannot capture (the dense builders in histogram.mojo pass their output
    buffers the same way, for the same reason).
    """
    var n_features = data.n_features
    var order_p = order.unsafe_ptr()
    var scratch_p = scratch.unsafe_ptr()
    var side_p = row_side.unsafe_ptr()
    var entry_row_p = data.row_index.unsafe_ptr()
    var start_p = node.starts.unsafe_ptr()
    var end_p = node.ends.unsafe_ptr()
    var ls_p = left_starts.unsafe_ptr()
    var le_p = left_ends.unsafe_ptr()
    var rs_p = right_starts.unsafe_ptr()
    var re_p = right_ends.unsafe_ptr()

    def do_feature(f: Int) {imm}:
        var lo = start_p.unsafe_load(f)
        var hi = end_p.unsafe_load(f)
        for i in range(lo, hi):
            scratch_p.unsafe_store(i, order_p.unsafe_load(i))
        var w = lo
        for i in range(lo, hi):
            var e = scratch_p.unsafe_load(i)
            if side_p.unsafe_load(entry_row_p.unsafe_load(e)) != 0:
                order_p.unsafe_store(w, e)
                w += 1
        var mid = w
        for i in range(lo, hi):
            var e = scratch_p.unsafe_load(i)
            if side_p.unsafe_load(entry_row_p.unsafe_load(e)) == 0:
                order_p.unsafe_store(w, e)
                w += 1
        ls_p.unsafe_store(f, lo)
        le_p.unsafe_store(f, mid)
        rs_p.unsafe_store(f, mid)
        re_p.unsafe_store(f, hi)

    dispatch_features(do_feature, n_features, node.n_entries() + n_features)


def build_histogram_sparse_node(
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    order: SparseEntryOrder,
    node: SparseNodeEntries,
    totals: NodeTotals,
    features: List[Int] = [],
    narrow: Bool = True,
) raises -> Histogram:
    """Histogram of one tree node from its grouped entry ranges.

    `narrow` is `histogram.derivative_precision`, and the one runtime test
    it costs is taken here, once per build, to select between two
    compile-time instantiations of the accumulation. Neither arm's entry
    loop contains a branch the other put there; at the default the emitted
    loop is the one that shipped. `totals` must have been summed at the same
    setting (`sum_rows(..., narrow)`), because the leftover assigned to the
    default bin is `totals` minus what the stored entries accumulated, and a
    mismatch would put the rounding difference of the whole node into one
    bin.
    """
    if narrow:
        return _build_histogram_sparse_node_at[True](
            data, grad, hess, order, node, totals, features
        )
    return _build_histogram_sparse_node_at[False](
        data, grad, hess, order, node, totals, features
    )


def _build_histogram_sparse_node_at[
    NARROW: Bool
](
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    order: SparseEntryOrder,
    node: SparseNodeEntries,
    totals: NodeTotals,
    features: List[Int] = [],
) raises -> Histogram:
    """`build_histogram_sparse_node` at one derivative precision.

    `totals` must be the node's summed gradient, hessian, and row count; the
    leftover after the stored entries is what lands in each feature's default
    bin. With a non-empty `features`, only those features are accumulated and
    the rest of the output stays zero.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    _check_features(features, data.n_features)
    # A repeated feature id is accumulated once, which is the only answer that
    # is a histogram of anything. See `_DUPLICATE_FEATURE_IDS` above for what
    # this accumulator does with a repeat that survives, and `_check_features`
    # for what the dense kernels do with theirs.
    #
    # Strip rather than raise, matching
    # `histogram.build_histogram_into_scratch` and for its reasons. A feature
    # list names a *set*, the natural reading of a set given twice is the set,
    # and this entry point is re-exported from `__init__.mojo` beside the
    # dense builder that already strips, so a caller swapping one for the
    # other must not meet an error the other never raised. Neither branch
    # costs anything on the production path, because no id list this project
    # generates can repeat.
    #
    # Re-entry rather than a local rebind, so the strip cannot drift out of
    # step with the accumulation below: the second pass takes exactly the path
    # the caller would have taken had it handed in the deduplicated list, and
    # it terminates because `_unique_features` cannot return a list with a
    # repeat in it.
    if _has_duplicate_features(features, data.n_features):
        var unique = _unique_features(features, data.n_features)
        return _build_histogram_sparse_node_at[NARROW](
            data, grad, hess, order, node, totals, unique
        )

    var n_bins = data.n_bins
    var size = data.n_features * n_bins
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    var h = List[Float64](capacity=size)
    h.resize(size, 0.0)
    var c = List[Int](capacity=size)
    c.resize(size, 0)

    var gp = g.unsafe_ptr()
    var hp = h.unsafe_ptr()
    var cp = c.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var order_p = order.order.unsafe_ptr()
    var start_p = node.starts.unsafe_ptr()
    var end_p = node.ends.unsafe_ptr()
    var entry_row_p = data.row_index.unsafe_ptr()
    var entry_bin_p = data.bin.unsafe_ptr()
    var default_p = data.default_bin.unsafe_ptr()
    var total_g = totals.grad
    var total_h = totals.hess
    var total_c = totals.count
    var use_all = len(features) == 0
    var n_active = data.n_features if use_all else len(features)
    var feat_p = features.unsafe_ptr()

    def do_feature(i_feature: Int) {imm}:
        var f = i_feature if use_all else feat_p.unsafe_load(i_feature)
        var base = f * n_bins
        var stored_g = 0.0
        var stored_h = 0.0
        var stored_c = 0
        for i in range(start_p.unsafe_load(f), end_p.unsafe_load(f)):
            var e = order_p.unsafe_load(i)
            var r = entry_row_p.unsafe_load(e)
            var b = base + Int(entry_bin_p.unsafe_load(e))
            var gr = derivative[NARROW](grad_p.unsafe_load(r))
            var he = derivative[NARROW](hess_p.unsafe_load(r))
            gp.unsafe_store(b, gp.unsafe_load(b) + gr)
            hp.unsafe_store(b, hp.unsafe_load(b) + he)
            cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            stored_g += gr
            stored_h += he
            stored_c += 1
        # Whatever the stored entries did not account for belongs to the
        # implicit zeros. The default bin may already hold stored entries
        # that happened to bin there, hence the read-modify-write.
        var db = base + Int(default_p.unsafe_load(f))
        gp.unsafe_store(db, gp.unsafe_load(db) + (total_g - stored_g))
        hp.unsafe_store(db, hp.unsafe_load(db) + (total_h - stored_h))
        cp.unsafe_store(db, cp.unsafe_load(db) + (total_c - stored_c))

    dispatch_features(
        do_feature, n_active, node.n_entries() + data.n_features
    )

    return Histogram.from_planes(g^, h^, c^, data.n_features, n_bins)


def build_histogram_sparse(
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int] = [],
    narrow: Bool = True,
) raises -> Histogram:
    """Full-dataset sparse histogram. `narrow` is
    `histogram.derivative_precision` and reaches the totals and the
    accumulation together, which is the pairing
    `build_histogram_sparse_node` requires."""
    var order = SparseEntryOrder(data.nnz())
    return build_histogram_sparse_node(
        data,
        grad,
        hess,
        order,
        SparseNodeEntries.root(data),
        sum_all(grad, hess, narrow),
        features,
        narrow,
    )


def build_histogram_sparse_subset(
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    features: List[Int] = [],
    narrow: Bool = True,
) raises -> Histogram:
    """Sparse histogram over an arbitrary row subset.

    `narrow` is `histogram.derivative_precision`; see
    `build_histogram_sparse_node`. It reaches the node totals as well as the
    accumulation, which is the part that has to agree.
    """
    if narrow:
        return _build_histogram_sparse_subset_at[True](
            data, grad, hess, rows, features
        )
    return _build_histogram_sparse_subset_at[False](
        data, grad, hess, rows, features
    )


def _build_histogram_sparse_subset_at[
    NARROW: Bool
](
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    features: List[Int] = [],
) raises -> Histogram:
    """`build_histogram_sparse_subset` at one derivative precision.

    Filters each feature's whole column against a row-membership mask, so it
    costs O(nnz) regardless of subset size. Tree growth instead keeps the
    entries grouped by node (`SparseEntryOrder`) and pays only
    O(nnz_in_node); this entry point exists for callers holding a plain row
    list. With a non-empty `features`, only those features are accumulated
    and the rest of the output stays zero.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    _check_features(features, data.n_features)
    # Repeats stripped before anything reads `len(features)` as an active
    # count; `_build_histogram_sparse_node_at` carries the argument for
    # stripping rather than refusing, and `_DUPLICATE_FEATURE_IDS` carries what
    # a repeat that survived would do to this accumulator.
    #
    # A repeated *row* below is refused rather than stripped, and the two rules
    # are not in tension. A duplicate row is a statement about the data being
    # summarized, and dropping it would silently change which rows the caller
    # asked about; a duplicate feature id is a statement about which slices to
    # fill, and naming a slice twice asks for the same slice.
    if _has_duplicate_features(features, data.n_features):
        var unique = _unique_features(features, data.n_features)
        return _build_histogram_sparse_subset_at[NARROW](
            data, grad, hess, rows, unique
        )

    var member = List[UInt8](capacity=data.n_rows)
    member.resize(data.n_rows, 0)
    for i in range(len(rows)):
        if rows[i] < 0 or rows[i] >= data.n_rows:
            raise Error("row index out of range")
        # A membership mask cannot represent a row twice, and the totals it
        # is subtracted from would count it twice, so reject rather than
        # accumulate a histogram that does not add up. A node's row list
        # never holds duplicates.
        if member[rows[i]] != 0:
            raise Error("duplicate row index in subset")
        member[rows[i]] = 1

    var n_bins = data.n_bins
    var size = data.n_features * n_bins
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    var h = List[Float64](capacity=size)
    h.resize(size, 0.0)
    var c = List[Int](capacity=size)
    c.resize(size, 0)

    var totals = sum_rows(grad, hess, rows, NARROW)
    var gp = g.unsafe_ptr()
    var hp = h.unsafe_ptr()
    var cp = c.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var member_p = member.unsafe_ptr()
    var off_p = data.col_offsets.unsafe_ptr()
    var entry_row_p = data.row_index.unsafe_ptr()
    var entry_bin_p = data.bin.unsafe_ptr()
    var default_p = data.default_bin.unsafe_ptr()
    var total_g = totals.grad
    var total_h = totals.hess
    var total_c = totals.count
    var use_all = len(features) == 0
    var n_active = data.n_features if use_all else len(features)
    var feat_p = features.unsafe_ptr()

    def do_feature(i_feature: Int) {imm}:
        var f = i_feature if use_all else feat_p.unsafe_load(i_feature)
        var base = f * n_bins
        var stored_g = 0.0
        var stored_h = 0.0
        var stored_c = 0
        for e in range(off_p.unsafe_load(f), off_p.unsafe_load(f + 1)):
            var r = entry_row_p.unsafe_load(e)
            if member_p.unsafe_load(r) == 0:
                continue
            var b = base + Int(entry_bin_p.unsafe_load(e))
            var gr = derivative[NARROW](grad_p.unsafe_load(r))
            var he = derivative[NARROW](hess_p.unsafe_load(r))
            gp.unsafe_store(b, gp.unsafe_load(b) + gr)
            hp.unsafe_store(b, hp.unsafe_load(b) + he)
            cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            stored_g += gr
            stored_h += he
            stored_c += 1
        var db = base + Int(default_p.unsafe_load(f))
        gp.unsafe_store(db, gp.unsafe_load(db) + (total_g - stored_g))
        hp.unsafe_store(db, hp.unsafe_load(db) + (total_h - stored_h))
        cp.unsafe_store(db, cp.unsafe_load(db) + (total_c - stored_c))

    dispatch_features(
        do_feature, n_active, data.nnz() + data.n_features
    )

    return Histogram.from_planes(g^, h^, c^, data.n_features, n_bins)
