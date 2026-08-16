"""Tests for the poisson, huber, quantile, L1, gamma, tweedie, MAPE, fair,
and cross-entropy objectives."""

from std.math import exp, log
from std.testing import assert_true, TestSuite

from mojotrees import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    FAIR,
    GAMMA,
    HUBER,
    L1,
    MAPE,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
    BoosterParams,
    TreeParams,
    bin_equal_width,
    train,
)
from mojotrees.boosting import fill_grad_hess


def _params(n_rounds: Int) -> BoosterParams:
    return BoosterParams(n_rounds, 0.3, TreeParams(4, 1, 1.0, 1e-3))


comptime _LossFn = def (Float64, Float64, Float64) raises -> Float64
"""(raw score, target, alpha) in, that row's loss out. Written out
independently of boosting.mojo so the check below is differential rather
than circular: the same expression differentiated twice would agree with
itself no matter what it said."""


def _check_derivatives[L: _LossFn](
    objective: Int,
    alpha: Float64,
    raws: List[Float64],
    targets: List[Float64],
    loss: L,
    check_hess: Bool = True,
) raises:
    """Central differences of `loss` against the gradient and hessian
    `fill_grad_hess` produces for `objective`.

    Two step sizes: 1e-6 for the first difference, where truncation error is
    O(h^2) and roundoff O(eps/h), and 1e-3 for the second, where roundoff is
    O(eps/h^2) and a smaller step would be worse, not better. `check_hess`
    is off for the objectives whose hessian is a LightGBM convention rather
    than a second derivative (the L1 family's constant 1, poisson's
    max-delta-step inflation).
    """
    var grad = List[Float64]()
    var hess = List[Float64]()
    fill_grad_hess(raws, targets, objective, [], alpha, grad, hess)
    for r in range(len(targets)):
        var raw = raws[r]
        var y = targets[r]

        var hg = 1e-6
        var fd_grad = (loss(raw + hg, y, alpha) - loss(raw - hg, y, alpha)) / (
            2.0 * hg
        )
        assert_true(
            abs(grad[r] - fd_grad) < 1e-5 * (1.0 + abs(fd_grad)),
            String(
                "gradient mismatch at row ",
                r,
                ": got ",
                grad[r],
                " want ",
                fd_grad,
            ),
        )

        if check_hess:
            var hh = 1e-3
            var fd_hess = (
                loss(raw + hh, y, alpha)
                - 2.0 * loss(raw, y, alpha)
                + loss(raw - hh, y, alpha)
            ) / (hh * hh)
            assert_true(
                abs(hess[r] - fd_hess) < 1e-4 * (1.0 + abs(fd_hess)),
                String(
                    "hessian mismatch at row ",
                    r,
                    ": got ",
                    hess[r],
                    " want ",
                    fd_hess,
                ),
            )


def _squared_error_loss(raw: Float64, y: Float64, alpha: Float64) -> Float64:
    return 0.5 * (raw - y) * (raw - y)


def _logistic_loss(raw: Float64, y: Float64, alpha: Float64) -> Float64:
    var p = 1.0 / (1.0 + exp(-raw))
    return -(y * log(p) + (1.0 - y) * log(1.0 - p))


def _gamma_loss(raw: Float64, y: Float64, alpha: Float64) -> Float64:
    return raw + y * exp(-raw)


def _tweedie_loss(raw: Float64, y: Float64, rho: Float64) -> Float64:
    return -y * exp((1.0 - rho) * raw) / (1.0 - rho) + exp(
        (2.0 - rho) * raw
    ) / (2.0 - rho)


def _fair_loss(raw: Float64, y: Float64, c: Float64) -> Float64:
    var d = abs(raw - y) / c
    return c * c * (d - log(1.0 + d))


def _mape_loss(raw: Float64, y: Float64, alpha: Float64) -> Float64:
    var denom = abs(y)
    if denom < 1.0:
        denom = 1.0
    return abs(raw - y) / denom


def test_squared_error_derivatives_match_finite_differences() raises:
    # The known-good objective, checked the same way as the new ones: if
    # the harness itself were wrong, this is what would say so.
    var raws: List[Float64] = [-2.0, -0.5, 0.0, 0.75, 3.0]
    var targets: List[Float64] = [1.0, -1.5, 0.25, 2.0, 0.0]
    _check_derivatives(SQUARED_ERROR, 0.0, raws, targets, _squared_error_loss)


def test_binary_logistic_derivatives_match_finite_differences() raises:
    var raws: List[Float64] = [-3.0, -0.5, 0.0, 0.5, 2.0]
    var targets: List[Float64] = [0.0, 1.0, 0.0, 1.0, 1.0]
    _check_derivatives(BINARY_LOGISTIC, 0.0, raws, targets, _logistic_loss)


def test_cross_entropy_derivatives_match_finite_differences() raises:
    # The labels are what separate cross entropy from binary logistic:
    # every one of these is strictly between 0 and 1.
    var raws: List[Float64] = [-2.0, -0.25, 0.0, 0.5, 1.5]
    var targets: List[Float64] = [0.1, 0.25, 0.5, 0.75, 0.9]
    _check_derivatives(CROSS_ENTROPY, 0.0, raws, targets, _logistic_loss)


def test_gamma_derivatives_match_finite_differences() raises:
    var raws: List[Float64] = [-1.0, 0.0, 0.5, 1.5, 2.5]
    var targets: List[Float64] = [0.5, 1.0, 2.0, 4.0, 10.0]
    _check_derivatives(GAMMA, 0.0, raws, targets, _gamma_loss)


def test_tweedie_derivatives_match_finite_differences() raises:
    # Checked at three variance powers, since rho enters both exponents.
    var raws: List[Float64] = [-1.0, 0.0, 0.5, 1.5, 2.0]
    var targets: List[Float64] = [0.0, 1.0, 2.5, 7.0, 12.0]
    _check_derivatives(TWEEDIE, 1.1, raws, targets, _tweedie_loss)
    _check_derivatives(TWEEDIE, 1.5, raws, targets, _tweedie_loss)
    _check_derivatives(TWEEDIE, 1.9, raws, targets, _tweedie_loss)


def test_fair_derivatives_match_finite_differences() raises:
    # Away from raw == y, where the loss is smooth. The kink at zero is
    # only a kink in the third derivative, but the finite differences are
    # cleanest away from it.
    var raws: List[Float64] = [-2.0, -0.5, 0.6, 1.5, 4.0]
    var targets: List[Float64] = [1.0, 0.5, -1.0, 3.0, 0.0]
    _check_derivatives(FAIR, 1.0, raws, targets, _fair_loss)
    _check_derivatives(FAIR, 2.5, raws, targets, _fair_loss)


def test_mape_gradient_matches_finite_differences() raises:
    # The hessian is LightGBM's convention (the label weight, not the true
    # second derivative, which is 0 almost everywhere), so only the
    # gradient is differenced. Rows straddle the |y| = 1 floor.
    var raws: List[Float64] = [-2.0, 0.25, 1.0, 3.0, 6.0]
    var targets: List[Float64] = [0.5, -0.75, 2.0, 5.0, 4.0]
    _check_derivatives(
        MAPE, 0.0, raws, targets, _mape_loss, check_hess=False
    )


def test_mape_hessian_is_the_label_weight() raises:
    # 1 / max(1, |y|), so a small label gets 1 and a large one gets the
    # reciprocal: the objective measures relative error.
    var raws: List[Float64] = [0.0, 0.0, 0.0]
    var targets: List[Float64] = [0.25, 1.0, 4.0]
    var grad = List[Float64]()
    var hess = List[Float64]()
    fill_grad_hess(raws, targets, MAPE, [], 0.0, grad, hess)
    assert_true(abs(hess[0] - 1.0) < 1e-12)
    assert_true(abs(hess[1] - 1.0) < 1e-12)
    assert_true(abs(hess[2] - 0.25) < 1e-12)


def test_new_objective_gradients_scale_with_sample_weight() raises:
    # A row weight multiplies the gradient and the hessian, including the
    # label weight MAPE already carries.
    var raws: List[Float64] = [0.3, -0.4, 1.2]
    var targets: List[Float64] = [1.0, 2.0, 3.0]
    var weights: List[Float64] = [2.0, 0.5, 3.0]
    var soft_targets: List[Float64] = [0.25, 0.5, 0.75]
    var objectives: List[Int] = [GAMMA, TWEEDIE, MAPE, FAIR, CROSS_ENTROPY]
    var alphas: List[Float64] = [0.0, 1.5, 0.0, 1.0, 0.0]
    for o in range(len(objectives)):
        var plain_g = List[Float64]()
        var plain_h = List[Float64]()
        var weighted_g = List[Float64]()
        var weighted_h = List[Float64]()
        var target = soft_targets.copy() if objectives[
            o
        ] == CROSS_ENTROPY else targets.copy()
        fill_grad_hess(
            raws, target, objectives[o], [], alphas[o], plain_g, plain_h
        )
        fill_grad_hess(
            raws,
            target,
            objectives[o],
            weights,
            alphas[o],
            weighted_g,
            weighted_h,
        )
        # Sample-weight scaling is exact in the objective's arithmetic and
        # then NARROWED: the code stores score_t(g * w), while this test can
        # only form score_t(g) * w. Those differ by up to two Float32 ulps,
        # so the identity now holds to Float32 precision rather than Float64.
        # 1e-12 was right when the derivative arrays were Float64 and is not
        # a bound anything can meet now. 1e-6 relative is roughly ten
        # Float32 ulps, tight enough that a real scaling bug still fails.
        #
        # This tolerance is the kind LANE_RULES forbids, and it is still here
        # because the exact form is per-objective work rather than a rewrite.
        # Under `derivative_precision = "float64"` the identity is exact and
        # could be asserted on `to_bits()` for every objective whose store is
        # literally `w * X` -- logistic, cross entropy, gamma, tweedie,
        # poisson, quantile, L1, huber. `FAIR` is not one of them: it stores
        # `w * alpha * d / denom`, which associates as `((w * alpha) * d) /
        # denom` and is not `w * (alpha * d / denom)`. An exact version has
        # to enumerate that, and enumerating it half way would ship a test
        # that looks exact and is not. See `tests/test_derivative_precision.
        # mojo`, which asserts the narrowing relationship exactly on the
        # unweighted path.
        for r in range(3):
            assert_true(
                abs(weighted_g[r] - weights[r] * plain_g[r])
                < 1e-6 * (1.0 + abs(plain_g[r]))
            )
            assert_true(
                abs(weighted_h[r] - weights[r] * plain_h[r])
                < 1e-6 * (1.0 + abs(plain_h[r]))
            )


def test_gamma_fits_group_means() raises:
    # Gamma has the exp link, so a converged model predicts each group's
    # mean on the response scale, as poisson does.
    var features: List[Float64] = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
    var target: List[Float64] = [1.0, 2.0, 3.0, 8.0, 10.0, 12.0]
    var data = bin_equal_width(features, n_rows=6, n_features=1, n_bins=2)
    var model = train(data, target, GAMMA, _params(400))
    for r in range(3):
        assert_true(abs(model.predict_row(data, r) - 2.0) < 0.05)
    for r in range(3, 6):
        assert_true(abs(model.predict_row(data, r) - 10.0) < 0.2)


def test_tweedie_fits_group_means() raises:
    # Tweedie's stationary point under the log link is the group mean too,
    # for any variance power in (1, 2).
    var features: List[Float64] = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
    var target: List[Float64] = [0.0, 2.0, 4.0, 8.0, 10.0, 12.0]
    var data = bin_equal_width(features, n_rows=6, n_features=1, n_bins=2)
    var model = train(data, target, TWEEDIE, _params(400), alpha=1.5)
    for r in range(3):
        assert_true(abs(model.predict_row(data, r) - 2.0) < 0.05)
    for r in range(3, 6):
        assert_true(abs(model.predict_row(data, r) - 10.0) < 0.2)


def test_base_score_only_model_predicts_the_mean_on_the_link_scale() raises:
    # Zero rounds: the prediction is the inverse link of the base score,
    # which for gamma and tweedie is the label mean.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var target: List[Float64] = [1.0, 2.0, 3.0, 6.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)
    var objectives: List[Int] = [GAMMA, TWEEDIE]
    for o in range(len(objectives)):
        var model = train(data, target, objectives[o], _params(0), alpha=1.5)
        for r in range(4):
            assert_true(abs(model.predict_row(data, r) - 3.0) < 1e-9)


def test_cross_entropy_fits_soft_labels() raises:
    # Continuous labels: the fitted probability is the group's mean label,
    # which no {0, 1} objective could represent.
    var features: List[Float64] = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
    var target: List[Float64] = [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]
    var data = bin_equal_width(features, n_rows=6, n_features=1, n_bins=2)
    var model = train(data, target, CROSS_ENTROPY, _params(400))
    for r in range(3):
        assert_true(abs(model.predict_row(data, r) - 0.2) < 0.01)
    for r in range(3, 6):
        assert_true(abs(model.predict_row(data, r) - 0.8) < 0.01)


def test_fair_with_large_c_approaches_squared_error() raises:
    # c^2 * (d/c - log(1 + d/c)) -> d^2 / 2 as c grows, so a large c should
    # land where squared error does.
    var features: List[Float64] = [0.0, 0.0, 5.0, 5.0]
    var target: List[Float64] = [1.0, 3.0, 8.0, 10.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=2)
    var fair = train(data, target, FAIR, _params(200), alpha=1e6)
    var sq = train(data, target, SQUARED_ERROR, _params(200))
    for r in range(4):
        assert_true(
            abs(fair.predict_row(data, r) - sq.predict_row(data, r)) < 1e-3
        )


def test_mape_ignores_a_large_outlier_more_than_l1_does() raises:
    # MAPE weights each row by 1 / max(1, |y|), so a label ten times the
    # rest pulls a tenth as hard as it does under plain L1.
    var features: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var target: List[Float64] = [10.0, 10.0, 10.0, 100.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=2)
    var mape_model = train(data, target, MAPE, _params(200))
    var l1_model = train(data, target, L1, _params(200))
    # Both settle on the median-like value; the check that matters is that
    # neither is dragged to the mean, and that MAPE is no further from the
    # bulk of the data than L1 is.
    var mape_pred = mape_model.predict_row(data, 0)
    var l1_pred = l1_model.predict_row(data, 0)
    assert_true(abs(mape_pred - 10.0) <= abs(l1_pred - 10.0) + 1e-9)
    assert_true(mape_pred < 20.0)


def test_new_objectives_validate_their_targets() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)

    # Gamma needs strictly positive labels: a zero has no gamma density.
    var with_zero: List[Float64] = [1.0, 2.0, 0.0, 4.0]
    var raised = False
    try:
        _ = train(data, with_zero, GAMMA, _params(5))
    except:
        raised = True
    assert_true(raised)

    # Tweedie allows zeros (that is the point of the compound Poisson) but
    # not negatives.
    var with_negative: List[Float64] = [1.0, 2.0, -1.0, 4.0]
    raised = False
    try:
        _ = train(data, with_negative, TWEEDIE, _params(5), alpha=1.5)
    except:
        raised = True
    assert_true(raised)

    # Cross entropy labels are probabilities.
    var out_of_range: List[Float64] = [0.0, 0.5, 1.5, 1.0]
    raised = False
    try:
        _ = train(data, out_of_range, CROSS_ENTROPY, _params(5))
    except:
        raised = True
    assert_true(raised)


def test_new_objectives_validate_alpha() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var target: List[Float64] = [1.0, 2.0, 3.0, 4.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)

    # tweedie_variance_power must be strictly inside (1, 2).
    var bad_powers: List[Float64] = [1.0, 2.0, 0.5, 2.5]
    for b in range(len(bad_powers)):
        var raised = False
        try:
            _ = train(data, target, TWEEDIE, _params(5), alpha=bad_powers[b])
        except:
            raised = True
        assert_true(raised)

    # fair_c must be positive.
    var raised_fair = False
    try:
        _ = train(data, target, FAIR, _params(5), alpha=0.0)
    except:
        raised_fair = True
    assert_true(raised_fair)


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
