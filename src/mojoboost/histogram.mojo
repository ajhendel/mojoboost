"""Histogram accumulation.

For each (feature, bin) pair, accumulates the sum of gradients, the sum of
hessians, and the row count. Split finding then scans these fixed-size
histograms instead of the raw data.

The accumulation loops are scatter-bound (bin indices collide), so they use
pointer-based scalar stores; the elementwise kernels (sibling subtraction)
are SIMD-vectorized. `SIMD_LANES` is sized above the hardware width so the
compiler can keep several vector operations in flight.

Accumulation parallelizes across features: each feature owns the
`[f * n_bins, (f + 1) * n_bins)` slice of the output, so per-feature workers
never write the same location and need no atomics. Nodes too small to amortize
task-scheduling overhead take the serial path.

Every builder takes an optional list of feature ids (empty means all). Under
feature subsampling only those features are accumulated; the output keeps its
full `n_features * n_bins` shape with the excluded slices left at zero, so no
dataset is copied or re-indexed and sibling subtraction stays exact.

Each builder comes in two forms. The plain one allocates and returns a fresh
`Histogram` and is what callers outside tree growth want. The `_into` one
writes a caller-owned buffer, so a grower that visits hundreds of nodes can
recycle a handful of histograms instead of allocating three arrays per node
(see `Histogram.zeroed` / `Histogram.reset`). The two forms run the same
kernels and produce bit-identical results; only the allocation differs.
"""

from std.sys.info import simd_width_of

from .binning import BinnedMatrix
from .parallel import PARALLEL_MIN_OPS, dispatch_features

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
        tree growth spends most of its allocator time on."""
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
    """`build_histogram` into a caller-owned buffer, which is zeroed first.
    Identical results, one fewer allocation per call."""
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    _check_features(features, data.n_features)

    out.reset()
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
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var bins_p = data.bins.unsafe_ptr()
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var use_all = len(features) == 0
    var n_active = data.n_features if use_all else len(features)
    var feat_p = features.unsafe_ptr()

    def do_feature(i: Int) {imm}:
        var f = i if use_all else feat_p.unsafe_load(i)
        var col = bins_p.unsafe_offset(f * n_rows)
        var base = f * n_bins
        for r in range(n_rows):
            var b = base + Int(col.unsafe_load(r))
            gp.unsafe_store(b, gp.unsafe_load(b) + grad_p.unsafe_load(r))
            hp.unsafe_store(b, hp.unsafe_load(b) + hess_p.unsafe_load(r))
            cp.unsafe_store(b, cp.unsafe_load(b) + 1)

    dispatch_features(do_feature, n_active, n_active * n_rows)


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
    row_start + row_count]`, into a caller-owned buffer that is zeroed first.

    The window lets tree growth keep every node's row ids in one shared arena
    instead of allocating a fresh `List[Int]` per node; passing
    `(0, len(rows))` is exactly the whole-list behaviour."""
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    if row_start < 0 or row_count < 0 or row_start + row_count > len(rows):
        raise Error("row window out of range")
    _check_features(features, data.n_features)

    out.reset()
    _accumulate_subset(
        out.grad, out.hess, out.count,
        data, grad, hess, rows, row_start, row_count, features,
    )


def _accumulate_subset(
    mut out_grad: List[Float64],
    mut out_hess: List[Float64],
    mut out_count: List[Int],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int],
) raises:
    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_sub = row_count
    var use_all = len(features) == 0
    var n_active = data.n_features if use_all else len(features)
    var feat_p = features.unsafe_ptr()

    def do_feature(i_feature: Int) {imm}:
        var f = i_feature if use_all else feat_p.unsafe_load(i_feature)
        var col = bins_all_p.unsafe_offset(f * n_rows)
        var base = f * n_bins
        for i in range(n_sub):
            var r = rows_p.unsafe_load(i)
            var b = base + Int(col.unsafe_load(r))
            gp.unsafe_store(b, gp.unsafe_load(b) + grad_p.unsafe_load(r))
            hp.unsafe_store(b, hp.unsafe_load(b) + hess_p.unsafe_load(r))
            cp.unsafe_store(b, cp.unsafe_load(b) + 1)

    dispatch_features(do_feature, n_active, n_active * n_sub)


def subtract_histogram(parent: Histogram, child: Histogram) raises -> Histogram:
    """Sibling histogram via the subtraction trick: build the smaller child
    directly, get the larger one as parent - child for free."""
    var out = Histogram.zeroed(parent.n_features, parent.n_bins)
    subtract_histogram_into(out, parent, child)
    return out^


def subtract_histogram_into(
    mut out: Histogram, parent: Histogram, child: Histogram
) raises:
    """`subtract_histogram` into a caller-owned buffer. Every element is
    written, so unlike the accumulating builders this one needs no zeroing
    pass at all."""
    if (
        parent.n_features != child.n_features
        or parent.n_bins != child.n_bins
    ):
        raise Error("histogram shapes must match")
    if not out.matches(parent.n_features, parent.n_bins):
        raise Error("output histogram shape must match the operands")

    var size = parent.n_features * parent.n_bins
    var pg = parent.grad.unsafe_ptr()
    var ph = parent.hess.unsafe_ptr()
    var pc = parent.count.unsafe_ptr()
    var cg = child.grad.unsafe_ptr()
    var ch = child.hess.unsafe_ptr()
    var cc = child.count.unsafe_ptr()
    var og = out.grad.unsafe_ptr()
    var oh = out.hess.unsafe_ptr()
    var oc = out.count.unsafe_ptr()

    comptime W = SIMD_LANES
    var i = 0
    while i + W <= size:
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
    while i < size:
        og.unsafe_store(i, pg.unsafe_load(i) - cg.unsafe_load(i))
        oh.unsafe_store(i, ph.unsafe_load(i) - ch.unsafe_load(i))
        oc.unsafe_store(i, pc.unsafe_load(i) - cc.unsafe_load(i))
        i += 1
