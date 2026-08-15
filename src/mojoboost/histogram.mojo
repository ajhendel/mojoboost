"""Histogram accumulation.

For each (feature, bin) pair, accumulates the sum of gradients, the sum of
hessians, and the row count. Split finding then scans these fixed-size
histograms instead of the raw data.

The accumulation loops are scatter-bound (bin indices collide), so they use
pointer-based scalar stores; the elementwise kernels (zeroing, sibling
subtraction) are SIMD-vectorized. `SIMD_LANES` is sized above the hardware
width so the compiler can keep several vector operations in flight.

Accumulation parallelizes across features: each feature owns the
`[f * n_bins, (f + 1) * n_bins)` slice of the output, so per-feature workers
never write the same location and need no atomics. Nodes too small to amortize
task-scheduling overhead take the serial path.

Every builder takes an optional list of feature ids (empty means all). Under
feature subsampling only those features are accumulated; the output keeps its
full `n_features * n_bins` shape with the excluded slices left at zero, so no
dataset is copied or re-indexed and sibling subtraction stays exact. The ids
must be distinct: two tasks handed the same feature would write the same
slice, which is the one thing the no-atomics argument above rules out.

Each builder comes in two forms. The plain one allocates and returns a fresh
`Histogram` and is what callers outside tree growth want. The `_into` one
writes a caller-owned buffer, so a grower that visits hundreds of nodes can
recycle a handful of histograms instead of allocating three arrays per node
(see `Histogram.zeroed` / `Histogram.reset`). The two forms run the same
kernels and produce bit-identical results; only the allocation differs.

Three things about the CPU shape of these kernels, all of them scheduling or
memory traffic and none of them arithmetic (see
`handoffs/performance_13_apple_cpu.md` for the mechanism behind each and for
what still needs measuring):

- **Zeroing is fused into the feature pass, not run before it.** A build
  writes every one of the `n_features * n_bins` cells and then accumulates
  into a subset of them. Done as a separate pass, the write streams a buffer
  that at 100 features and 255 bins is around 600 KB out of cache and back
  in; done inside each feature's task it happens on the slice that task is
  about to accumulate into, in parallel with every other task. It is also
  counted in the work estimate, so a small node whose accumulation is tiny
  but whose output is large no longer runs a multi-megabyte write serially.
- **A node's gradients and hessians are gathered once, not once per
  feature.** Accumulating a tree node reads `grad[r]` and `hess[r]` through
  a row id, so each feature pays two indirect loads per row over buffers far
  larger than cache. `build_histogram_subset_into_scratch` gathers them into
  one contiguous `(g, h)` pair buffer first, which turns those two loads per
  feature into two sequential loads per feature and leaves one gather for
  the whole node. The threshold and the scratch reuse are the caller's, via
  the scratch form.
- **Two features are accumulated in one inner loop.** Their slices are
  disjoint, so the two read-modify-write chains are independent, and the row
  id and the gradient pair are loaded once for both. Whether the hardware
  actually overlaps them is a claim for a profiler, not for this comment.

None of that changes a value or an order: each feature still sums its bins
over rows in ascending order inside a single task.
"""

from std.math import round
from std.sys.info import simd_width_of

from .apple_cpu_policy import (
    cpu_profile,
    derive_accumulation_plan,
    subtract_ops,
)
from .binning import BinnedMatrix
from .parallel import dispatch_feature_ranges, dispatch_features, dispatch_rows

comptime SIMD_LANES = 4 * simd_width_of[DType.float64]()


@fieldwise_init
struct Histogram(Copyable, Movable):
    """Per-(feature, bin) statistics, flattened as `[f * n_bins + b]`."""

    var grad: List[Float64]
    var hess: List[Float64]
    var count: List[Int]
    var n_features: Int
    var n_bins: Int

    @staticmethod
    def zeroed(n_features: Int, n_bins: Int) -> Histogram:
        """An all-zero histogram of the given shape, ready to accumulate
        into. Callers that build many histograms of one shape allocate once
        with this and recycle with `reset`."""
        var size = n_features * n_bins
        return Histogram(
            _zeroed_f64(size), _zeroed_f64(size), _zeroed_int(size),
            n_features, n_bins,
        )

    def reset(mut self):
        """Zero every bin in place, keeping the allocation. Cheaper than a
        fresh `zeroed` by exactly one malloc/free per buffer, which is what
        tree growth spends most of its allocator time on.

        Serial by construction, and the builders below no longer call it:
        they zero each feature's slice inside that feature's task instead. It
        stays because it is the right shape for a caller that needs a zeroed
        buffer without a build (`histogram_gpu`, `distributed`) and for any
        caller already running inside a parallel task, where a dispatching
        zero pass would nest one `sync_parallelize` inside another.
        """
        var size = self.n_features * self.n_bins
        var gp = self.grad.unsafe_ptr()
        var hp = self.hess.unsafe_ptr()
        var cp = self.count.unsafe_ptr()
        comptime W = SIMD_LANES
        var i = 0
        while i + W <= size:
            gp.unsafe_store(i, SIMD[DType.float64, W](0.0))
            hp.unsafe_store(i, SIMD[DType.float64, W](0.0))
            cp.unsafe_store(i, SIMD[DType.int, W](0))
            i += W
        while i < size:
            gp.unsafe_store(i, 0.0)
            hp.unsafe_store(i, 0.0)
            cp.unsafe_store(i, 0)
            i += 1

    def matches(self, n_features: Int, n_bins: Int) -> Bool:
        return self.n_features == n_features and self.n_bins == n_bins


@fieldwise_init
struct FeatureTotals(Copyable, Movable):
    """One feature's totals over its bins."""

    var grad: Float64
    var hess: Float64
    var count: Int


def _zeroed_f64(size: Int) -> List[Float64]:
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    return g^


def _zeroed_int(size: Int) -> List[Int]:
    var c = List[Int](capacity=size)
    c.resize(size, 0)
    return c^


def _check_features(features: List[Int], n_features: Int) raises:
    """Feature ids must be in range; an empty list means every feature."""
    for i in range(len(features)):
        if features[i] < 0 or features[i] >= n_features:
            raise Error("feature index out of range")


def ensure_pair_capacity(mut pairs: List[Float64], n_rows: Int):
    """Size a gradient/hessian pair buffer for a node of `n_rows` rows.

    Grows only. A grower that keeps one buffer across a whole tree allocates
    it once, at the size of the largest node it meets (the root), and every
    later node reuses that allocation: the buffer is written before it is
    read, so stale contents beyond the current node are never observed.
    """
    var wanted = 2 * n_rows
    if len(pairs) < wanted:
        pairs.resize(wanted, 0.0)


def _zero_excluded(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    n_features: Int,
    n_bins: Int,
    features: List[Int],
    total_ops: Int,
) raises:
    """Zero the slices of every feature this build will not accumulate.

    Only reached under feature subsampling. The active features' slices are
    zeroed by the accumulation pass itself, on the task that is about to fill
    them, so this covers exactly the rest.
    """
    if len(features) == 0 or n_features <= 0 or n_bins <= 0:
        return

    # Mojo's scalar pointer API cannot load `Bool`. A byte preserves the
    # compact active mask and keeps the parallel zeroing pass unchanged.
    var active = List[UInt8](capacity=n_features)
    active.resize(n_features, UInt8(0))
    for i in range(len(features)):
        active[features[i]] = UInt8(1)

    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var active_p = active.unsafe_ptr()
    comptime W = SIMD_LANES

    def zero_range(f_start: Int, f_end: Int) {imm}:
        for f in range(f_start, f_end):
            if active_p.unsafe_load(f) == UInt8(0):
                var base = f * n_bins
                var b = 0
                while b + W <= n_bins:
                    gp.unsafe_store(
                        base + b, SIMD[DType.float64, W](0.0)
                    )
                    hp.unsafe_store(
                        base + b, SIMD[DType.float64, W](0.0)
                    )
                    cp.unsafe_store(base + b, SIMD[DType.int, W](0))
                    b += W
                while b < n_bins:
                    gp.unsafe_store(base + b, 0.0)
                    hp.unsafe_store(base + b, 0.0)
                    cp.unsafe_store(base + b, 0)
                    b += 1

    dispatch_feature_ranges(zero_range, n_features, total_ops)


def build_histogram(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int] = [],
) raises -> Histogram:
    """Build a full-dataset histogram from per-row gradients and hessians.
    With a non-empty `features`, only those features are accumulated and the
    rest of the output stays zero."""
    var out = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_into(out, data, grad, hess, features)
    return out^


def build_histogram_into(
    mut out: Histogram,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int] = [],
) raises:
    """`build_histogram` into a caller-owned buffer, every cell of which is
    written: the accumulation pass zeroes each slice before filling it, so a
    buffer holding another node's histogram comes back holding exactly this
    one. Identical results to the allocating form, one fewer allocation."""
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    _check_features(features, data.n_features)

    # The three output buffers are passed as separate `mut` lists rather than
    # reached through `out`: a pointer taken from a struct field carries that
    # field's origin, which a worker closure cannot capture.
    _accumulate_full(
        out.grad, out.hess, out.count, data, grad, hess, features
    )


def _accumulate_full(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int],
) raises:
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_features = data.n_features
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)
    var plan = derive_accumulation_plan(
        cpu_profile(), n_features, n_active, n_bins, n_rows, False
    )

    _zero_excluded(
        out_grad, out_hess, out_count,
        n_features, n_bins, features, plan.excluded_ops,
    )

    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var bins_p = data.bins.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    var pair_features = plan.group_width >= 2
    comptime W = SIMD_LANES

    def accumulate_range(i_start: Int, i_end: Int) {imm}:
        var i = i_start
        while i < i_end:
            # Two features share one walk of the row range when the policy
            # allows it and the task has two left. Every row of the dataset is
            # visited in ascending order either way, which is the whole of the
            # ordering invariant.
            if pair_features and i + 1 < i_end:
                var f0 = i if use_all else feat_p.unsafe_load(i)
                var f1 = (i + 1) if use_all else feat_p.unsafe_load(i + 1)
                var base0 = f0 * n_bins
                var base1 = f1 * n_bins
                var zb = 0
                while zb + W <= n_bins:
                    gp.unsafe_store(base0 + zb, SIMD[DType.float64, W](0.0))
                    hp.unsafe_store(base0 + zb, SIMD[DType.float64, W](0.0))
                    cp.unsafe_store(base0 + zb, SIMD[DType.int, W](0))
                    gp.unsafe_store(base1 + zb, SIMD[DType.float64, W](0.0))
                    hp.unsafe_store(base1 + zb, SIMD[DType.float64, W](0.0))
                    cp.unsafe_store(base1 + zb, SIMD[DType.int, W](0))
                    zb += W
                while zb < n_bins:
                    gp.unsafe_store(base0 + zb, 0.0)
                    hp.unsafe_store(base0 + zb, 0.0)
                    cp.unsafe_store(base0 + zb, 0)
                    gp.unsafe_store(base1 + zb, 0.0)
                    hp.unsafe_store(base1 + zb, 0.0)
                    cp.unsafe_store(base1 + zb, 0)
                    zb += 1
                var col0 = bins_p.unsafe_offset(f0 * n_rows)
                var col1 = bins_p.unsafe_offset(f1 * n_rows)
                for r in range(n_rows):
                    # Contiguous: the whole dataset is one ascending walk, so
                    # the gradients, the hessians, and both binned columns are
                    # sequential streams.
                    var g = grad_p.unsafe_load(r)
                    var h = hess_p.unsafe_load(r)
                    var b0 = base0 + Int(col0.unsafe_load(r))
                    var b1 = base1 + Int(col1.unsafe_load(r))
                    gp.unsafe_store(b0, gp.unsafe_load(b0) + g)
                    hp.unsafe_store(b0, hp.unsafe_load(b0) + h)
                    cp.unsafe_store(b0, cp.unsafe_load(b0) + 1)
                    gp.unsafe_store(b1, gp.unsafe_load(b1) + g)
                    hp.unsafe_store(b1, hp.unsafe_load(b1) + h)
                    cp.unsafe_store(b1, cp.unsafe_load(b1) + 1)
                i += 2
            else:
                var f = i if use_all else feat_p.unsafe_load(i)
                var base = f * n_bins
                var zb = 0
                while zb + W <= n_bins:
                    gp.unsafe_store(base + zb, SIMD[DType.float64, W](0.0))
                    hp.unsafe_store(base + zb, SIMD[DType.float64, W](0.0))
                    cp.unsafe_store(base + zb, SIMD[DType.int, W](0))
                    zb += W
                while zb < n_bins:
                    gp.unsafe_store(base + zb, 0.0)
                    hp.unsafe_store(base + zb, 0.0)
                    cp.unsafe_store(base + zb, 0)
                    zb += 1
                var col = bins_p.unsafe_offset(f * n_rows)
                for r in range(n_rows):
                    var g = grad_p.unsafe_load(r)
                    var h = hess_p.unsafe_load(r)
                    var b = base + Int(col.unsafe_load(r))
                    gp.unsafe_store(b, gp.unsafe_load(b) + g)
                    hp.unsafe_store(b, hp.unsafe_load(b) + h)
                    cp.unsafe_store(b, cp.unsafe_load(b) + 1)
                i += 1

    dispatch_feature_ranges(accumulate_range, n_active, plan.active_ops)


def build_histogram_subset(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    features: List[Int] = [],
) raises -> Histogram:
    """Build a histogram over a subset of rows (one tree node's rows). With a
    non-empty `features`, only those features are accumulated and the rest of
    the output stays zero."""
    var out = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_subset_into(
        out, data, grad, hess, rows, 0, len(rows), features
    )
    return out^


def build_histogram_subset_into(
    mut out: Histogram,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int] = [],
) raises:
    """`build_histogram_subset` over the window `rows[row_start :
    row_start + row_count]`, into a caller-owned buffer every cell of which
    is written.

    The window lets tree growth keep every node's row ids in one shared arena
    instead of allocating a fresh `List[Int]` per node; passing
    `(0, len(rows))` is exactly the whole-list behaviour.

    This form owns the gradient/hessian pair buffer for the duration of the
    call, which means allocating and freeing it per node. A grower visiting
    hundreds of nodes should hold one buffer and call
    `build_histogram_subset_into_scratch` instead; the two produce identical
    results.
    """
    var pairs = List[Float64]()
    build_histogram_subset_into_scratch(
        out, pairs, data, grad, hess, rows, row_start, row_count, features
    )


def build_histogram_subset_into_scratch(
    mut out: Histogram,
    mut pairs: List[Float64],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int] = [],
) raises:
    """`build_histogram_subset_into` with a caller-owned scratch buffer.

    `pairs` is the node's gradients and hessians, interleaved, and is grown
    (never shrunk) to fit. Passing the same list across every node of a tree
    turns a per-node allocation of `16 * row_count` bytes into one allocation
    per tree; passing a fresh empty list is exactly
    `build_histogram_subset_into`. Its contents on entry are irrelevant and
    its contents on exit are unspecified, so it carries no state between
    calls beyond its capacity.

    Whether the buffer is filled at all is a policy decision
    (`apple_cpu_policy.derive_accumulation_plan`): a node with one active
    feature, or too few rows for the gather to pay for itself, reads the
    gradients through the row ids as before. Both paths accumulate the same
    values in the same order.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    if row_start < 0 or row_count < 0 or row_start + row_count > len(rows):
        raise Error("row window out of range")
    _check_features(features, data.n_features)

    _accumulate_subset(
        out.grad, out.hess, out.count, pairs,
        data, grad, hess, rows, row_start, row_count, features,
    )


def _accumulate_subset(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    mut pairs: List[Float64],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int],
) raises:
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_features = data.n_features
    var n_sub = row_count
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)
    var plan = derive_accumulation_plan(
        cpu_profile(), n_features, n_active, n_bins, n_sub, True
    )

    _zero_excluded(
        out_grad, out_hess, out_count,
        n_features, n_bins, features, plan.excluded_ops,
    )
    if plan.compact_rows:
        ensure_pair_capacity(pairs, n_sub)

    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    var pairs_p = pairs.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    var compact = plan.compact_rows
    var pair_features = plan.group_width >= 2
    comptime W = SIMD_LANES

    # One gather for the whole node instead of one per feature. Disjoint
    # ascending blocks, elementwise, so the buffer comes out the same
    # whatever the block count.
    def fill_pairs(start: Int, end: Int) {imm}:
        for i in range(start, end):
            var r = rows_p.unsafe_load(i)
            pairs_p.unsafe_store(2 * i, grad_p.unsafe_load(r))
            pairs_p.unsafe_store(2 * i + 1, hess_p.unsafe_load(r))

    if compact:
        dispatch_rows(fill_pairs, n_sub, plan.gather_ops)

    def accumulate_range(i_start: Int, i_end: Int) {imm}:
        var i = i_start
        while i < i_end:
            if compact and pair_features and i + 1 < i_end:
                var f0 = i if use_all else feat_p.unsafe_load(i)
                var f1 = (i + 1) if use_all else feat_p.unsafe_load(i + 1)
                var base0 = f0 * n_bins
                var base1 = f1 * n_bins
                var zb = 0
                while zb + W <= n_bins:
                    gp.unsafe_store(base0 + zb, SIMD[DType.float64, W](0.0))
                    hp.unsafe_store(base0 + zb, SIMD[DType.float64, W](0.0))
                    cp.unsafe_store(base0 + zb, SIMD[DType.int, W](0))
                    gp.unsafe_store(base1 + zb, SIMD[DType.float64, W](0.0))
                    hp.unsafe_store(base1 + zb, SIMD[DType.float64, W](0.0))
                    cp.unsafe_store(base1 + zb, SIMD[DType.int, W](0))
                    zb += W
                while zb < n_bins:
                    gp.unsafe_store(base0 + zb, 0.0)
                    hp.unsafe_store(base0 + zb, 0.0)
                    cp.unsafe_store(base0 + zb, 0)
                    gp.unsafe_store(base1 + zb, 0.0)
                    hp.unsafe_store(base1 + zb, 0.0)
                    cp.unsafe_store(base1 + zb, 0)
                    zb += 1
                var col0 = bins_all_p.unsafe_offset(f0 * n_rows)
                var col1 = bins_all_p.unsafe_offset(f1 * n_rows)
                for i_row in range(n_sub):
                    var r = rows_p.unsafe_load(i_row)
                    # Adjacent, so one cache line carries both.
                    var g = pairs_p.unsafe_load(2 * i_row)
                    var h = pairs_p.unsafe_load(2 * i_row + 1)
                    var b0 = base0 + Int(col0.unsafe_load(r))
                    var b1 = base1 + Int(col1.unsafe_load(r))
                    gp.unsafe_store(b0, gp.unsafe_load(b0) + g)
                    hp.unsafe_store(b0, hp.unsafe_load(b0) + h)
                    cp.unsafe_store(b0, cp.unsafe_load(b0) + 1)
                    gp.unsafe_store(b1, gp.unsafe_load(b1) + g)
                    hp.unsafe_store(b1, hp.unsafe_load(b1) + h)
                    cp.unsafe_store(b1, cp.unsafe_load(b1) + 1)
                i += 2
                continue

            var f = i if use_all else feat_p.unsafe_load(i)
            var base = f * n_bins
            var zb = 0
            while zb + W <= n_bins:
                gp.unsafe_store(base + zb, SIMD[DType.float64, W](0.0))
                hp.unsafe_store(base + zb, SIMD[DType.float64, W](0.0))
                cp.unsafe_store(base + zb, SIMD[DType.int, W](0))
                zb += W
            while zb < n_bins:
                gp.unsafe_store(base + zb, 0.0)
                hp.unsafe_store(base + zb, 0.0)
                cp.unsafe_store(base + zb, 0)
                zb += 1
            var col = bins_all_p.unsafe_offset(f * n_rows)
            if compact:
                for i_row in range(n_sub):
                    var r = rows_p.unsafe_load(i_row)
                    var g = pairs_p.unsafe_load(2 * i_row)
                    var h = pairs_p.unsafe_load(2 * i_row + 1)
                    var b = base + Int(col.unsafe_load(r))
                    gp.unsafe_store(b, gp.unsafe_load(b) + g)
                    hp.unsafe_store(b, hp.unsafe_load(b) + h)
                    cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            else:
                # Small node, or one active feature: the gather would cost a
                # pass and save nothing, so the gradients are read through the
                # row ids as they always were.
                for i_row in range(n_sub):
                    var r = rows_p.unsafe_load(i_row)
                    var g = grad_p.unsafe_load(r)
                    var h = hess_p.unsafe_load(r)
                    var b = base + Int(col.unsafe_load(r))
                    gp.unsafe_store(b, gp.unsafe_load(b) + g)
                    hp.unsafe_store(b, hp.unsafe_load(b) + h)
                    cp.unsafe_store(b, cp.unsafe_load(b) + 1)
            i += 1

    dispatch_feature_ranges(accumulate_range, n_active, plan.active_ops)


def build_histogram_subset_replica_into[
    grad_origin: ImmOrigin, hess_origin: ImmOrigin, //
](
    mut out: Histogram,
    mut fixed: List[Int32],
    data: BinnedMatrix,
    grad_f32: Span[Float32, grad_origin],
    hess_f32: Span[Float32, hess_origin],
    rows: List[Int32],
    row_start: Int,
    row_count: Int,
    g_scale: Float64,
    h_scale: Float64,
    features: List[Int] = [],
) raises:
    """The host replica of the device's fixed-point histogram build, over
    the window `rows[row_start : row_start + row_count]`.

    This is the builder `docs/design/HYBRID_TRAINING.md` §8.2 specifies: it
    must reproduce the device pipeline number for number, so a histogram it
    produces can substitute for a downloaded one without changing a split.
    Per (row, feature) it computes

        gq = Int32(round(grad_f32[r] * Float32(g_scale)))

    exactly as `_range_hist_atomic_kernel` does — the *same* Float32 inputs
    the kernel reads (the staged conversions, not the Float64 originals) and
    the same Float32 scale — accumulates in Int32, and dequantizes with the
    same `Float64(sum) * (1.0 / scale)` as `histogram_from_host`. Integer
    addition is associative and commutative, so the per-feature tasks and
    their order cannot change a sum; whether the host's multiply-and-round
    matches the device's bit for bit is a hardware claim this function
    cannot make, and `MODE_MIRROR` in hybrid_leaf_scheduler.mojo is how it
    is established before a replica is allowed to substitute.

    `fixed` is the caller-owned Int32 scratch holding the three planes in
    the device's `[grad | hess | count]` layout, followed by the node's
    quantized (g, h) pairs; it is grown to
    `3 * n_features * n_bins + 2 * row_count` as needed and its contents on
    entry are irrelevant. Excluded features' output slices are zeroed, as every
    builder here guarantees, so the result has the full dataset shape.
    """
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_features = data.n_features
    if len(grad_f32) != n_rows or len(hess_f32) != n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(n_features, n_bins):
        raise Error("output histogram shape must match the data")
    if row_start < 0 or row_count < 0 or row_start + row_count > len(rows):
        raise Error("row window out of range")
    if g_scale <= 0.0 or h_scale <= 0.0:
        raise Error("fixed-point scales must be positive")
    _check_features(features, n_features)
    for j in range(row_start, row_start + row_count):
        var r = Int(rows[j])
        if r < 0 or r >= n_rows:
            raise Error("row id out of range")

    var hist_size = n_features * n_bins
    if len(fixed) < 3 * hist_size + 2 * row_count:
        fixed.resize(3 * hist_size + 2 * row_count, Int32(0))

    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)
    _zero_excluded(
        out.grad, out.hess, out.count,
        n_features, n_bins, features,
        (n_features - n_active) * n_bins,
    )

    # The three output buffers are passed as separate `mut` lists rather than
    # reached through `out`, exactly as `build_histogram_into` explains: a
    # pointer taken from a struct field carries that field's origin, which a
    # worker closure cannot capture.
    _accumulate_replica(
        out.grad, out.hess, out.count, fixed,
        data, grad_f32, hess_f32, rows, row_start, row_count,
        g_scale, h_scale, features,
    )


def _accumulate_replica[
    grad_origin: ImmOrigin, hess_origin: ImmOrigin, //
](
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    mut fixed: List[Int32],
    data: BinnedMatrix,
    grad_f32: Span[Float32, grad_origin],
    hess_f32: Span[Float32, hess_origin],
    rows: List[Int32],
    row_start: Int,
    row_count: Int,
    g_scale: Float64,
    h_scale: Float64,
    features: List[Int],
) raises:
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_features = data.n_features
    var hist_size = n_features * n_bins
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)

    var gsf = Float32(g_scale)
    var hsf = Float32(h_scale)
    var g_inv = 1.0 / g_scale
    var h_inv = 1.0 / h_scale
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var fixed_p = fixed.unsafe_ptr()
    var grad_p = grad_f32.unsafe_ptr()
    var hess_p = hess_f32.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    var feat_p = features.unsafe_ptr()

    # Quantize once per row, not once per (row, feature). The kernel rounds
    # inside its accumulation loop, but the rounded value is a pure function
    # of the row, so hoisting it changes no Int32 the sums are made of --
    # while removing 2 * (n_active - 1) multiply-and-rounds per row, which
    # measured as the difference between this builder running at the Float64
    # builder's rate and running twenty times slower. The pairs land in the
    # tail of the caller's scratch, interleaved like the gather in
    # `_accumulate_subset`.
    var pairs_off = 3 * hist_size

    def fill_quantized(start: Int, end: Int) {imm}:
        for i in range(start, end):
            var r = Int(rows_p.unsafe_load(i))
            fixed_p.unsafe_store(
                pairs_off + 2 * i,
                Int32(round(grad_p.unsafe_load(r) * gsf)),
            )
            fixed_p.unsafe_store(
                pairs_off + 2 * i + 1,
                Int32(round(hess_p.unsafe_load(r) * hsf)),
            )

    dispatch_rows(fill_quantized, row_count, row_count)

    def accumulate_feature(i: Int) {imm}:
        var f = i if use_all else feat_p.unsafe_load(i)
        var base = f * n_bins
        for b in range(n_bins):
            fixed_p.unsafe_store(base + b, Int32(0))
            fixed_p.unsafe_store(hist_size + base + b, Int32(0))
            fixed_p.unsafe_store(2 * hist_size + base + b, Int32(0))
        var col = bins_all_p.unsafe_offset(f * n_rows)
        for j in range(row_count):
            var r = Int(rows_p.unsafe_load(j))
            var slot = base + Int(col.unsafe_load(r))
            var gq = fixed_p.unsafe_load(pairs_off + 2 * j)
            var hq = fixed_p.unsafe_load(pairs_off + 2 * j + 1)
            fixed_p.unsafe_store(slot, fixed_p.unsafe_load(slot) + gq)
            fixed_p.unsafe_store(
                hist_size + slot, fixed_p.unsafe_load(hist_size + slot) + hq
            )
            fixed_p.unsafe_store(
                2 * hist_size + slot,
                fixed_p.unsafe_load(2 * hist_size + slot) + Int32(1),
            )
        for b in range(n_bins):
            gp.unsafe_store(
                base + b, Float64(fixed_p.unsafe_load(base + b)) * g_inv
            )
            hp.unsafe_store(
                base + b,
                Float64(fixed_p.unsafe_load(hist_size + base + b)) * h_inv,
            )
            cp.unsafe_store(
                base + b, Int(fixed_p.unsafe_load(2 * hist_size + base + b))
            )

    dispatch_features(
        accumulate_feature,
        n_active,
        row_count * n_active + 2 * n_active * n_bins,
    )


def feature_totals(hist: Histogram, feature: Int) raises -> FeatureTotals:
    """Totals over one feature's bins.

    Split finding computes exactly this inline before it scans a feature, and
    tree growth computes it again for the leaf value. The order here is the
    order `split.mojo` uses today and must stay that way: `SIMD_LANES` lanes
    accumulated in parallel over the whole vector blocks, reduced, then the
    scalar tail added in ascending bin order. Anything else would move a
    result, not just its cost.

    Nothing calls this yet. It exists so that a split scan parallelized
    across features has a totals pass it can hoist without changing a single
    gain (see the handoff for the call sites).
    """
    if feature < 0 or feature >= hist.n_features:
        raise Error("feature index out of range")
    comptime W = SIMD_LANES
    var grad_p = hist.grad.unsafe_ptr()
    var hess_p = hist.hess.unsafe_ptr()
    var count_p = hist.count.unsafe_ptr()
    var base = feature * hist.n_bins
    var vg = SIMD[DType.float64, W](0.0)
    var vh = SIMD[DType.float64, W](0.0)
    var vc = SIMD[DType.int, W](0)
    var b = 0
    while b + W <= hist.n_bins:
        vg += grad_p.unsafe_load[width=W](base + b)
        vh += hess_p.unsafe_load[width=W](base + b)
        vc += count_p.unsafe_load[width=W](base + b)
        b += W
    var total_g = vg.reduce_add()
    var total_h = vh.reduce_add()
    var total_c = Int(vc.reduce_add())
    while b < hist.n_bins:
        total_g += grad_p.unsafe_load(base + b)
        total_h += hess_p.unsafe_load(base + b)
        total_c += count_p.unsafe_load(base + b)
        b += 1
    return FeatureTotals(total_g, total_h, total_c)


def subtract_histogram(parent: Histogram, child: Histogram) raises -> Histogram:
    """Sibling histogram via the subtraction trick: build the smaller child
    directly, get the larger one as parent - child for free."""
    var out = Histogram.zeroed(parent.n_features, parent.n_bins)
    subtract_histogram_into(out, parent, child)
    return out^


def _subtract_histogram_arrays(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    parent_grad: List[Float64],
    parent_hess: List[Float64],
    parent_count: List[Int],
    child_grad: List[Float64],
    child_hess: List[Float64],
    child_count: List[Int],
    size: Int,
) raises:
    """Parallel SIMD sibling subtraction over independent array borrows.

    Keeping this worker outside `Histogram` field access gives each buffer a
    stable origin that Mojo's ownership checker can carry into the parallel
    closure. The dispatch, range partitioning, SIMD width, and arithmetic are
    identical to the original optimized implementation.
    """
    var pg = parent_grad.unsafe_ptr()
    var ph = parent_hess.unsafe_ptr()
    var pc = parent_count.unsafe_ptr()
    var cg = child_grad.unsafe_ptr()
    var ch = child_hess.unsafe_ptr()
    var cc = child_count.unsafe_ptr()
    var og = out_grad.unsafe_ptr()
    var oh = out_hess.unsafe_ptr()
    var oc = out_count.unsafe_ptr()

    def subtract_block(start: Int, end: Int) {imm}:
        comptime W = SIMD_LANES
        var i = start
        while i + W <= end:
            og.unsafe_store(
                i, pg.unsafe_load[width=W](i) - cg.unsafe_load[width=W](i)
            )
            oh.unsafe_store(
                i, ph.unsafe_load[width=W](i) - ch.unsafe_load[width=W](i)
            )
            oc.unsafe_store(
                i, pc.unsafe_load[width=W](i) - cc.unsafe_load[width=W](i)
            )
            i += W
        while i < end:
            og.unsafe_store(i, pg.unsafe_load(i) - cg.unsafe_load(i))
            oh.unsafe_store(i, ph.unsafe_load(i) - ch.unsafe_load(i))
            oc.unsafe_store(i, pc.unsafe_load(i) - cc.unsafe_load(i))
            i += 1

    dispatch_rows(subtract_block, size, subtract_ops(size))


def subtract_histogram_into(
    mut out: Histogram, parent: Histogram, child: Histogram
) raises:
    """`subtract_histogram` into a caller-owned buffer. Every element is
    written, so unlike the accumulating builders this one needs no zeroing
    pass at all.

    "For free" was always an overstatement: the pass reads two histograms and
    writes a third, six streams over `n_features * n_bins` cells each, and at
    the default shape that is megabytes per node against a node whose
    accumulation the trick just avoided. It is elementwise, so it splits into
    contiguous blocks that reproduce the serial result exactly, and the work
    estimate counts one op per cell per stream rather than per cell (see
    `apple_cpu_policy.subtract_ops`), which is what lets a shape this
    memory-bound reach the parallel path at all.
    """
    if (
        parent.n_features != child.n_features
        or parent.n_bins != child.n_bins
    ):
        raise Error("histogram shapes must match")
    if not out.matches(parent.n_features, parent.n_bins):
        raise Error("output histogram shape must match the operands")

    _subtract_histogram_arrays(
        out.grad,
        out.hess,
        out.count,
        parent.grad,
        parent.hess,
        parent.count,
        child.grad,
        child.hess,
        child.count,
        parent.n_features * parent.n_bins,
    )
