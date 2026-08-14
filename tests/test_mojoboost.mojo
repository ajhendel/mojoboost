from std.testing import assert_equal, assert_true, assert_false, TestSuite

from mojoboost import (
    BinnedMatrix,
    Histogram,
    SplitInfo,
    bin_equal_width,
    build_histogram,
    find_best_split,
)


def make_toy() raises -> BinnedMatrix:
    # One feature, 8 rows, values 0..7 binned into 8 equal-width bins.
    # Equal-width binning maps value v to bin v exactly for this input.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    return bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)


def test_binning_identity() raises:
    var data = make_toy()
    assert_equal(data.n_rows, 8)
    assert_equal(data.n_features, 1)
    assert_equal(data.n_bins, 8)
    for r in range(8):
        assert_equal(data.bin_at(r, 0), r)


def test_binning_constant_feature() raises:
    var features: List[Float64] = [3.0, 3.0, 3.0, 3.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)
    for r in range(4):
        assert_equal(data.bin_at(r, 0), 0)


def test_binning_validates_input() raises:
    var features: List[Float64] = [1.0, 2.0]
    var raised = False
    try:
        _ = bin_equal_width(features, n_rows=2, n_features=1, n_bins=1)
    except:
        raised = True
    assert_true(raised)


def test_histogram_sums() raises:
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var hist = build_histogram(data, grad, hess)
    for b in range(8):
        assert_equal(hist.count[b], 1)
        assert_equal(hist.hess[b], 1.0)
        if b < 4:
            assert_equal(hist.grad[b], -1.0)
        else:
            assert_equal(hist.grad[b], 1.0)


def test_best_split_separates_gradients() raises:
    # Gradients flip sign between bins 3 and 4, so the best split must be
    # at bin 3 (rows with bin <= 3 go left).
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var hist = build_histogram(data, grad, hess)
    var split = find_best_split(hist, lambda_reg=1.0)
    assert_true(split.found)
    assert_equal(split.feature, 0)
    assert_equal(split.bin, 3)
    # GL=-4, HL=4, GR=4, HR=4, G=0: gain = 16/5 + 16/5 - 0 = 6.4
    assert_true(abs(split.gain - 6.4) < 1e-12)


def test_no_split_on_uniform_gradients() raises:
    var data = make_toy()
    var grad: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var hist = build_histogram(data, grad, hess)
    var split = find_best_split(hist, lambda_reg=1.0)
    assert_false(split.found)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
