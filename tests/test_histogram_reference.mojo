"""Serial-reference equivalence for the CPU histogram kernels.

The optimized builders (pointer stores, SIMD subtraction, per-feature
multicore) must be bit-identical to a plain List-indexing serial reference:
parallelism is per-feature and every feature accumulates its rows in the
same order as the reference, so no floating-point reassociation occurs.

Shapes are chosen to be odd with respect to any plausible SIMD width
(NEON, AVX2, AVX-512) and to fall on both sides of the multicore
threshold, so both the serial and sync_parallelize paths are covered on
every architecture CI runs on.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojoboost.binning import bin_equal_width, BinnedMatrix
from mojoboost.histogram import (
    build_histogram,
    build_histogram_subset,
    subtract_histogram,
    Histogram,
)
from mojoboost.parallel import PARALLEL_MIN_OPS


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _make_data(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises -> BinnedMatrix:
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(seed + UInt64(k)))
    return bin_equal_width(features, n_rows, n_features, n_bins)


def _grads(n_rows: Int, seed: UInt64) -> List[Float64]:
    var g = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        g.append(2.0 * _uniform(seed + UInt64(r)) - 1.0)
    return g^


def _hessians(n_rows: Int, seed: UInt64) -> List[Float64]:
    var h = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        h.append(_uniform(seed + UInt64(r)) + 0.01)
    return h^


def _reference_full(
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64]
) raises -> Histogram:
    var size = data.n_features * data.n_bins
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    var h = List[Float64](capacity=size)
    h.resize(size, 0.0)
    var c = List[Int](capacity=size)
    c.resize(size, 0)
    for f in range(data.n_features):
        for r in range(data.n_rows):
            var b = f * data.n_bins + data.bin_at(r, f)
            g[b] += grad[r]
            h[b] += hess[r]
            c[b] += 1
    return Histogram(g^, h^, c^, data.n_features, data.n_bins)


def _reference_subset(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
) raises -> Histogram:
    var size = data.n_features * data.n_bins
    var g = List[Float64](capacity=size)
    g.resize(size, 0.0)
    var h = List[Float64](capacity=size)
    h.resize(size, 0.0)
    var c = List[Int](capacity=size)
    c.resize(size, 0)
    for f in range(data.n_features):
        for i in range(len(rows)):
            var r = rows[i]
            var b = f * data.n_bins + data.bin_at(r, f)
            g[b] += grad[r]
            h[b] += hess[r]
            c[b] += 1
    return Histogram(g^, h^, c^, data.n_features, data.n_bins)


def _assert_same(a: Histogram, b: Histogram) raises:
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    for i in range(a.n_features * a.n_bins):
        assert_equal(a.grad[i], b.grad[i])
        assert_equal(a.hess[i], b.hess[i])
        assert_equal(a.count[i], b.count[i])


def test_full_matches_reference_serial_odd_shapes() raises:
    var rows_l = [101, 997, 1023, 511]
    var feats_l = [1, 3, 5, 7]
    var bins_l = [2, 2, 17, 3]
    for k in range(len(rows_l)):
        var n_rows = rows_l[k]
        var n_features = feats_l[k]
        # Guard: these shapes must exercise the serial path.
        assert_true(n_features * n_rows < PARALLEL_MIN_OPS)
        var data = _make_data(n_rows, n_features, bins_l[k], UInt64(1000 * k))
        var grad = _grads(n_rows, UInt64(7_000_000 + 1000 * k))
        var hess = _hessians(n_rows, UInt64(8_000_000 + 1000 * k))
        _assert_same(
            _reference_full(data, grad, hess),
            build_histogram(data, grad, hess),
        )


def test_full_matches_reference_parallel_odd_shape() raises:
    var n_rows = 6247
    var n_features = 21
    # Guard: this shape must exercise the sync_parallelize path.
    assert_true(n_features * n_rows >= PARALLEL_MIN_OPS)
    var data = _make_data(n_rows, n_features, 255, UInt64(1))
    var grad = _grads(n_rows, UInt64(9_000_000))
    var hess = _hessians(n_rows, UInt64(10_000_000))
    _assert_same(
        _reference_full(data, grad, hess),
        build_histogram(data, grad, hess),
    )


def test_subset_matches_reference_serial_and_parallel() raises:
    var n_rows = 12_000
    var n_features = 25
    var data = _make_data(n_rows, n_features, 64, UInt64(2))
    var grad = _grads(n_rows, UInt64(11_000_000))
    var hess = _hessians(n_rows, UInt64(12_000_000))

    # Tiny odd subset: serial path, SIMD-width-free row count.
    var tiny = List[Int]()
    for i in range(7):
        tiny.append(3 * i + 1)
    _assert_same(
        _reference_subset(data, grad, hess, tiny),
        build_histogram_subset(data, grad, hess, tiny),
    )

    # Every 3rd row: still below the multicore threshold.
    var third = List[Int]()
    for r in range(0, n_rows, 3):
        third.append(r)
    assert_true(n_features * len(third) < PARALLEL_MIN_OPS)
    _assert_same(
        _reference_subset(data, grad, hess, third),
        build_histogram_subset(data, grad, hess, third),
    )

    # Every 2nd row: above the multicore threshold.
    var half = List[Int]()
    for r in range(0, n_rows, 2):
        half.append(r)
    assert_true(n_features * len(half) >= PARALLEL_MIN_OPS)
    _assert_same(
        _reference_subset(data, grad, hess, half),
        build_histogram_subset(data, grad, hess, half),
    )


def test_subtract_matches_reference_odd_tail() raises:
    # 3 features x 17 bins = 51 entries: odd against 8/16/32-lane widths,
    # so the SIMD loop leaves a scalar tail on every architecture.
    var n_rows = 907
    var n_features = 3
    var data = _make_data(n_rows, n_features, 17, UInt64(3))
    var grad = _grads(n_rows, UInt64(13_000_000))
    var hess = _hessians(n_rows, UInt64(14_000_000))

    var left = List[Int]()
    for r in range(0, n_rows, 2):
        left.append(r)

    var parent = build_histogram(data, grad, hess)
    var child = build_histogram_subset(data, grad, hess, left)
    var sibling = subtract_histogram(parent, child)

    var size = n_features * parent.n_bins
    for i in range(size):
        assert_equal(sibling.grad[i], parent.grad[i] - child.grad[i])
        assert_equal(sibling.hess[i], parent.hess[i] - child.hess[i])
        assert_equal(sibling.count[i], parent.count[i] - child.count[i])

    # The two children partition the parent's rows.
    var total = 0
    for i in range(size):
        total += child.count[i] + sibling.count[i]
    assert_equal(total, n_features * n_rows)


def test_env_worker_and_threshold_overrides() raises:
    # One test owns all env mutation so no other test sees a dirty
    # environment regardless of suite ordering; empty string means unset.
    var n_rows = 997
    var n_features = 3
    var data = _make_data(n_rows, n_features, 17, UInt64(4))
    var grad = _grads(n_rows, UInt64(15_000_000))
    var hess = _hessians(n_rows, UInt64(16_000_000))
    var expected = _reference_full(data, grad, hess)

    # Lowered threshold forces the per-feature parallel path on a shape
    # that would otherwise run serial.
    _ = setenv("MOJOBOOST_PARALLEL_MIN_OPS", "1")
    _assert_same(expected, build_histogram(data, grad, hess))

    # Explicit worker count takes the chunked path (2 tasks over 3 features).
    _ = setenv("MOJOBOOST_NUM_WORKERS", "2")
    _assert_same(expected, build_histogram(data, grad, hess))

    # More workers than features clamps to one feature per task.
    _ = setenv("MOJOBOOST_NUM_WORKERS", "8")
    _assert_same(expected, build_histogram(data, grad, hess))

    # workers=1 forces serial even with the threshold floored.
    _ = setenv("MOJOBOOST_NUM_WORKERS", "1")
    _assert_same(expected, build_histogram(data, grad, hess))

    _ = setenv("MOJOBOOST_NUM_WORKERS", "")
    _ = setenv("MOJOBOOST_PARALLEL_MIN_OPS", "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
