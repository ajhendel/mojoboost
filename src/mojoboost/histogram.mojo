"""Histogram accumulation.

For each (feature, bin) pair, accumulates the sum of gradients, the sum of
hessians, and the row count. Split finding then scans these fixed-size
histograms instead of the raw data.

The accumulation loops are scatter-bound (bin indices collide), so they use
pointer-based scalar stores; the elementwise kernels (sibling subtraction)
are SIMD-vectorized. `SIMD_LANES` is sized above the hardware width so the
compiler can keep several vector operations in flight.
"""

from std.sys.info import simd_width_of

from .binning import BinnedMatrix

comptime SIMD_LANES = 4 * simd_width_of[DType.float64]()


@fieldwise_init
struct Histogram(Copyable, Movable):
    """Per-(feature, bin) statistics, flattened as `[f * n_bins + b]`."""

    var grad: List[Float64]
    var hess: List[Float64]
    var count: List[Int]
    var n_features: Int
    var n_bins: Int


def _zeroed_f64(size: Int) -> List[Float64]:
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    return g^


def _zeroed_int(size: Int) -> List[Int]:
    var c = List[Int](capacity=size)
    c.resize(size, 0)
    return c^


def build_histogram(
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64]
) raises -> Histogram:
    """Build a full-dataset histogram from per-row gradients and hessians."""
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")

    var size = data.n_features * data.n_bins
    var g = _zeroed_f64(size)
    var h = _zeroed_f64(size)
    var c = _zeroed_int(size)

    var gp = g.unsafe_ptr()
    var hp = h.unsafe_ptr()
    var cp = c.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    for f in range(data.n_features):
        var bins_p = data.bins.unsafe_ptr().unsafe_offset(f * data.n_rows)
        var base = f * data.n_bins
        for r in range(data.n_rows):
            var b = base + Int(bins_p.unsafe_load(r))
            gp.unsafe_store(b, gp.unsafe_load(b) + grad_p.unsafe_load(r))
            hp.unsafe_store(b, hp.unsafe_load(b) + hess_p.unsafe_load(r))
            cp.unsafe_store(b, cp.unsafe_load(b) + 1)

    return Histogram(g^, h^, c^, data.n_features, data.n_bins)


def build_histogram_subset(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
) raises -> Histogram:
    """Build a histogram over a subset of rows (one tree node's rows)."""
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")

    var size = data.n_features * data.n_bins
    var g = _zeroed_f64(size)
    var h = _zeroed_f64(size)
    var c = _zeroed_int(size)

    var gp = g.unsafe_ptr()
    var hp = h.unsafe_ptr()
    var cp = c.unsafe_ptr()
    var grad_p = grad.unsafe_ptr()
    var hess_p = hess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr()
    var n_sub = len(rows)
    for f in range(data.n_features):
        var bins_p = data.bins.unsafe_ptr().unsafe_offset(f * data.n_rows)
        var base = f * data.n_bins
        for i in range(n_sub):
            var r = rows_p.unsafe_load(i)
            var b = base + Int(bins_p.unsafe_load(r))
            gp.unsafe_store(b, gp.unsafe_load(b) + grad_p.unsafe_load(r))
            hp.unsafe_store(b, hp.unsafe_load(b) + hess_p.unsafe_load(r))
            cp.unsafe_store(b, cp.unsafe_load(b) + 1)

    return Histogram(g^, h^, c^, data.n_features, data.n_bins)


def subtract_histogram(parent: Histogram, child: Histogram) raises -> Histogram:
    """Sibling histogram via the subtraction trick: build the smaller child
    directly, get the larger one as parent - child for free."""
    if (
        parent.n_features != child.n_features
        or parent.n_bins != child.n_bins
    ):
        raise Error("histogram shapes must match")

    var size = parent.n_features * parent.n_bins
    var g = _zeroed_f64(size)
    var h = _zeroed_f64(size)
    var c = _zeroed_int(size)

    var pg = parent.grad.unsafe_ptr()
    var ph = parent.hess.unsafe_ptr()
    var pc = parent.count.unsafe_ptr()
    var cg = child.grad.unsafe_ptr()
    var ch = child.hess.unsafe_ptr()
    var cc = child.count.unsafe_ptr()
    var og = g.unsafe_ptr()
    var oh = h.unsafe_ptr()
    var oc = c.unsafe_ptr()

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

    return Histogram(g^, h^, c^, parent.n_features, parent.n_bins)
