"""Tests for the poisson objective."""

from std.testing import assert_true, TestSuite

from mojoboost import POISSON, BoosterParams, TreeParams, bin_equal_width, train


def _params(n_rounds: Int) -> BoosterParams:
    return BoosterParams(n_rounds, 0.3, TreeParams(4, 1, 1.0, 1e-3))


def test_poisson_fits_group_means() raises:
    # Two groups of counts; a converged poisson model predicts each
    # group's mean on the response scale.
    var features: List[Float64] = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
    var target: List[Float64] = [1.0, 2.0, 3.0, 8.0, 10.0, 12.0]
    var data = bin_equal_width(features, n_rows=6, n_features=1, n_bins=2)
    var model = train(data, target, POISSON, _params(300))
    for r in range(3):
        assert_true(abs(model.predict_row(data, r) - 2.0) < 0.02)
    for r in range(3, 6):
        assert_true(abs(model.predict_row(data, r) - 10.0) < 0.05)


def test_poisson_zero_count_group_stays_positive() raises:
    # A group of all-zero counts drives its prediction toward zero but
    # the exp link keeps it strictly positive.
    var features: List[Float64] = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
    var target: List[Float64] = [0.0, 0.0, 0.0, 4.0, 5.0, 6.0]
    var data = bin_equal_width(features, n_rows=6, n_features=1, n_bins=2)
    var model = train(data, target, POISSON, _params(300))
    for r in range(3):
        var p = model.predict_row(data, r)
        assert_true(p > 0.0)
        assert_true(p < 0.05)
    for r in range(3, 6):
        assert_true(abs(model.predict_row(data, r) - 5.0) < 0.05)


def test_poisson_validates_negative_target() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var target: List[Float64] = [1.0, 2.0, -1.0, 2.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)
    var raised = False
    try:
        _ = train(data, target, POISSON, _params(5))
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
