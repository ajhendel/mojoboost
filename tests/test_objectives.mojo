"""Tests for the poisson, huber, quantile, and L1 objectives."""

from std.testing import assert_true, TestSuite

from mojoboost import (
    HUBER,
    L1,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    BoosterParams,
    TreeParams,
    bin_equal_width,
    train,
)


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


def test_huber_large_alpha_equals_squared_error() raises:
    # With alpha larger than any residual the huber gradient and hessian
    # equal the squared-error ones exactly, so the models must match.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var target: List[Float64] = [0.5, 0.5, 1.5, 1.5, 4.0, 4.0, 9.0, 9.0]
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var params = BoosterParams(20, 0.3, TreeParams(4, 1, 1.0, 1e-3))
    var huber = train(data, target, HUBER, params, alpha=1e9)
    var l2 = train(data, target, SQUARED_ERROR, params)
    for r in range(8):
        var ph = huber.predict_row(data, r)
        var pl = l2.predict_row(data, r)
        assert_true(abs(ph - pl) < 1e-12)


def test_huber_resists_outlier() raises:
    # One group with a gross outlier: [1, 1, 1, 1, 100]. Squared error is
    # dragged to the mean (20.8). Huber with alpha = 1 converges to the
    # M-estimate where four clipped-in residuals balance the outlier's
    # clipped gradient: 4 * (pred - 1) = 1, so pred = 1.25.
    var features: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0]
    var target: List[Float64] = [1.0, 1.0, 1.0, 1.0, 100.0]
    var data = bin_equal_width(features, n_rows=5, n_features=1, n_bins=2)
    var params = BoosterParams(400, 0.3, TreeParams(4, 1, 1.0, 1e-3))
    var huber = train(data, target, HUBER, params, alpha=1.0)
    var l2 = train(data, target, SQUARED_ERROR, params)
    assert_true(abs(huber.predict_row(data, 0) - 1.25) < 1e-3)
    assert_true(l2.predict_row(data, 0) > 15.0)


def test_quantile_fits_group_quantiles() raises:
    # Two groups with different sizes so the boundary split keeps a
    # positive gain at the fixed point. Leaf renewal must converge each
    # group to LightGBM's interpolated percentile: at alpha = 0.75 the
    # position is (n - 1) * alpha, so group A [1..5] gives 4.0 and
    # group B [10, 20, 30, 40] gives 30 + 0.25 * 10 = 32.5.
    var features: List[Float64] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0,
    ]
    var target: List[Float64] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0,
    ]
    var data = bin_equal_width(features, n_rows=9, n_features=1, n_bins=2)
    var params = BoosterParams(80, 0.3, TreeParams(4, 1, 1.0, 1e-3))
    var model = train(data, target, QUANTILE, params, alpha=0.75)
    for r in range(5):
        assert_true(abs(model.predict_row(data, r) - 4.0) < 1e-6)
    for r in range(5, 9):
        assert_true(abs(model.predict_row(data, r) - 32.5) < 1e-6)

    # alpha = 0.5 is the median: 3.0 and interpolated 25.0.
    var median = train(data, target, QUANTILE, params, alpha=0.5)
    for r in range(5):
        assert_true(abs(median.predict_row(data, r) - 3.0) < 1e-6)
    for r in range(5, 9):
        assert_true(abs(median.predict_row(data, r) - 25.0) < 1e-6)


def test_l1_fits_group_medians() raises:
    # L1 is quantile at alpha = 0.5 with sign gradients; each group must
    # converge to its median regardless of the alpha argument (ignored).
    var features: List[Float64] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0,
    ]
    var target: List[Float64] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0,
    ]
    var data = bin_equal_width(features, n_rows=9, n_features=1, n_bins=2)
    var params = BoosterParams(80, 0.3, TreeParams(4, 1, 1.0, 1e-3))
    var model = train(data, target, L1, params)
    for r in range(5):
        assert_true(abs(model.predict_row(data, r) - 3.0) < 1e-6)
    for r in range(5, 9):
        assert_true(abs(model.predict_row(data, r) - 25.0) < 1e-6)


def test_weighted_quantile_median() raises:
    # Weighted median of [1, 2, 3] with weights [1, 1, 10]: the cdf is
    # [1, 2, 12] and the threshold 6 lands past the last gap, so the
    # answer is exactly 3.0. The base score already equals it, the first
    # residual percentile is zero, and training converges immediately.
    var features: List[Float64] = [0.0, 0.0, 0.0]
    var target: List[Float64] = [1.0, 2.0, 3.0]
    var weights: List[Float64] = [1.0, 1.0, 10.0]
    var data = bin_equal_width(features, n_rows=3, n_features=1, n_bins=2)
    var params = BoosterParams(20, 0.3, TreeParams(4, 1, 1.0, 1e-3))
    var model = train(data, target, QUANTILE, params, weights, alpha=0.5)
    for r in range(3):
        assert_true(abs(model.predict_row(data, r) - 3.0) < 1e-12)


def test_alpha_validation() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var target: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)
    var params = BoosterParams(5, 0.3, TreeParams(4, 1, 1.0, 1e-3))

    var raised = False
    try:
        _ = train(data, target, QUANTILE, params, alpha=1.0)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = train(data, target, QUANTILE, params, alpha=0.0)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = train(data, target, HUBER, params, alpha=0.0)
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
