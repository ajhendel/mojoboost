"""Equivalence tests for the pointerized/SIMD histogram and split kernels.

Each test compares the optimized kernel against a naive scalar reference on
deterministic pseudo-random data (splitmix64), sized so the SIMD main loops
and their scalar tails both execute.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix
from mojotrees.histogram import (
    Histogram,
    build_histogram,
    build_histogram_subset,
    subtract_histogram,
)
from mojotrees.split import find_best_split


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _make_data(n_rows: Int, n_features: Int, n_bins: Int) -> BinnedMatrix:
    var bins = List[UInt8](capacity=n_rows * n_features)
    for i in range(n_rows * n_features):
        bins.append(UInt8(Int(_uniform(UInt64(i)) * Float64(n_bins))))
    return BinnedMatrix(bins^, n_rows, n_features, n_bins)


def _make_grad_hess(n_rows: Int, mut grad: List[Float64], mut hess: List[Float64]):
    for r in range(n_rows):
        grad.append(_uniform(UInt64(100_000 + r)) * 2.0 - 1.0)
        hess.append(_uniform(UInt64(200_000 + r)) + 0.01)


def _reference_histogram(
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64]
) -> Histogram:
    var size = data.n_features * data.n_bins
    var g = List[Float64]()
    var h = List[Float64]()
    var c = List[Int]()
    g.resize(size, 0.0)
    h.resize(size, 0.0)
    c.resize(size, 0)
    for f in range(data.n_features):
        for r in range(data.n_rows):
            var b = f * data.n_bins + data.bin_at(r, f)
            g[b] += grad[r]
            h[b] += hess[r]
            c[b] += 1
    return Histogram(g^, h^, c^, data.n_features, data.n_bins)


def _assert_hist_close(a: Histogram, b: Histogram) raises:
    assert_equal(a.n_features, b.n_features)
    assert_equal(a.n_bins, b.n_bins)
    var size = a.n_features * a.n_bins
    for i in range(size):
        assert_true(abs(a.grad[i] - b.grad[i]) < 1e-9)
        assert_true(abs(a.hess[i] - b.hess[i]) < 1e-9)
        assert_equal(a.count[i], b.count[i])


def test_build_histogram_matches_reference() raises:
    # 137 rows x 3 features x 61 bins: odd sizes exercise SIMD tails.
    var data = _make_data(137, 3, 61)
    var grad = List[Float64]()
    var hess = List[Float64]()
    _make_grad_hess(137, grad, hess)

    var got = build_histogram(data, grad, hess)
    var want = _reference_histogram(data, grad, hess)
    _assert_hist_close(got, want)


def test_subset_plus_subtract_matches_full() raises:
    var n_rows = 201
    var data = _make_data(n_rows, 4, 33)
    var grad = List[Float64]()
    var hess = List[Float64]()
    _make_grad_hess(n_rows, grad, hess)

    var left = List[Int]()
    var right = List[Int]()
    for r in range(n_rows):
        if _uniform(UInt64(300_000 + r)) < 0.4:
            left.append(r)
        else:
            right.append(r)

    var full = build_histogram(data, grad, hess)
    var left_hist = build_histogram_subset(data, grad, hess, left)
    var right_direct = build_histogram_subset(data, grad, hess, right)
    var right_subtracted = subtract_histogram(full, left_hist)
    _assert_hist_close(right_subtracted, right_direct)


def test_find_best_split_agrees_with_scalar_totals() raises:
    # The vectorized totals must leave split selection identical to a
    # scalar rescan of the winning feature's histogram.
    var n_rows = 500
    var data = _make_data(n_rows, 5, 64)
    var grad = List[Float64]()
    var hess = List[Float64]()
    # Correlate the gradient with feature 0's bin so a real split exists.
    for r in range(n_rows):
        var pull = Float64(data.bin_at(r, 0)) / 64.0 - 0.5
        grad.append(pull + (_uniform(UInt64(400_000 + r)) - 0.5) * 0.1)
        hess.append(1.0)

    var hist = build_histogram(data, grad, hess)
    var split = find_best_split(hist, 1.0, 1e-3, 1)
    assert_true(split.found)
    assert_equal(split.feature, 0)

    # Recompute the gain of the chosen split with plain scalar sums.
    var base = split.feature * hist.n_bins
    var total_g = 0.0
    var total_h = 0.0
    for b in range(hist.n_bins):
        total_g += hist.grad[base + b]
        total_h += hist.hess[base + b]
    var left_g = 0.0
    var left_h = 0.0
    for b in range(split.bin + 1):
        left_g += hist.grad[base + b]
        left_h += hist.hess[base + b]
    var right_g = total_g - left_g
    var right_h = total_h - left_h
    var gain = (
        left_g * left_g / (left_h + 1.0)
        + right_g * right_g / (right_h + 1.0)
        - total_g * total_g / (total_h + 1.0)
    )
    assert_true(abs(gain - split.gain) < 1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
