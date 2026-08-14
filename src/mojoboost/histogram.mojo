"""Histogram accumulation.

For each (feature, bin) pair, accumulates the sum of gradients, the sum of
hessians, and the row count. Split finding then scans these fixed-size
histograms instead of the raw data.

v0 is a scalar reference implementation. SIMD accumulation and row-subset
(node-level) histograms are the next steps.
"""

from .binning import BinnedMatrix


@fieldwise_init
struct Histogram(Copyable, Movable):
    """Per-(feature, bin) statistics, flattened as `[f * n_bins + b]`."""

    var grad: List[Float64]
    var hess: List[Float64]
    var count: List[Int]
    var n_features: Int
    var n_bins: Int


def build_histogram(
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64]
) raises -> Histogram:
    """Build a full-dataset histogram from per-row gradients and hessians."""
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")

    var size = data.n_features * data.n_bins
    var g = List[Float64](capacity=size)
    var h = List[Float64](capacity=size)
    var c = List[Int](capacity=size)
    for _ in range(size):
        g.append(0.0)
        h.append(0.0)
        c.append(0)

    for f in range(data.n_features):
        var col = f * data.n_rows
        var base = f * data.n_bins
        for r in range(data.n_rows):
            var b = base + Int(data.bins[col + r])
            g[b] += grad[r]
            h[b] += hess[r]
            c[b] += 1

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
    var g = List[Float64](capacity=size)
    var h = List[Float64](capacity=size)
    var c = List[Int](capacity=size)
    for _ in range(size):
        g.append(0.0)
        h.append(0.0)
        c.append(0)

    for f in range(data.n_features):
        var col = f * data.n_rows
        var base = f * data.n_bins
        for i in range(len(rows)):
            var r = rows[i]
            var b = base + Int(data.bins[col + r])
            g[b] += grad[r]
            h[b] += hess[r]
            c[b] += 1

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
    var g = List[Float64](capacity=size)
    var h = List[Float64](capacity=size)
    var c = List[Int](capacity=size)
    for i in range(size):
        g.append(parent.grad[i] - child.grad[i])
        h.append(parent.hess[i] - child.hess[i])
        c.append(parent.count[i] - child.count[i])

    return Histogram(g^, h^, c^, parent.n_features, parent.n_bins)
