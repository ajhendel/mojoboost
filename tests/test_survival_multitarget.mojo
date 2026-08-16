"""CatBoost's `Cox`, `SurvivalAft` and `MultiRMSE`
(src/mojotrees/survival.mojo, src/mojotrees/multi_target.mojo,
src/mojotrees/target_matrix.mojo; docs/design/CATBOOST_CATALOG.md A26-A28).

Every expected number below was derived by hand from CatBoost's formulas or
computed independently in Python from the transcribed distribution
definitions, never read out of a mojotrees run.

Tolerances. The Cox assertions are at 1e-12 because their expected values
(-1/2, +1/2, 1/4) are exact binary fractions and the arithmetic that produces
them is exact. The `SurvivalAft` assertions are at **1e-7**, and that is a
measured toolchain fact rather than slack: Mojo's `std.math` transcendentals
are not libm. `exp(log(5.0))` is 4.999999998698298 here (2.6e-10 relative),
and the Normal cdf at 1 is 0.841344750494095 against libm's
0.8413447460685429 (5.3e-9 relative). The expected values below came from
Python, so every AFT comparison is a mojotrees-vs-libm comparison and 1e-7
is roughly twenty times the observed gap.

The tests that carry weight are the ones that would still pass a *plausible*
wrong implementation:

- `test_cox_ties_are_not_breslow` -- Breslow's correction gives both tied
  events the full risk set and therefore gradient 0 apiece. CatBoost gives
  the second tied event a risk set that excludes the first, and the answer is
  (-0.5, +0.5). Getting the tie rule wrong is the single most likely Cox bug
  and this is what catches it.
- `test_multi_target_gain_is_not_the_gain_of_summed_planes` -- the summed-
  plane gain is 0 on this candidate and the sum-of-gains is 4. Handing a
  single-plane grower the elementwise sum of the planes is the easy wrong
  answer for `MultiRMSE`, and it is 0 where the right answer is 4.
- `test_survival_aft_normal_hessian_is_one_over_scale_squared` -- an
  analytic identity of the Normal AFT loss that holds for every uncensored
  row at every approx, so it checks the whole pdf/pdf'/pdf'' triple at once
  rather than one sampled point.
"""

from std.math import exp, log, sqrt
from std.testing import TestSuite, assert_equal, assert_false, assert_raises
from std.testing import assert_true

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.boosting import BoosterParams, train
from mojotrees.multi_target import (
    MULTI_RMSE,
    MULTI_RMSE_WITH_MISSING,
    check_multi_target_hessian_declaration,
    multi_rmse,
    multi_rmse_grad_hess,
    multi_target_leaf_values,
    multi_target_split_gain,
    multi_target_varies_hessian,
    train_multi_rmse,
)
from mojotrees.survival import (
    AFT_DIST_EXTREME,
    AFT_DIST_LOGISTIC,
    AFT_DIST_NORMAL,
    AFT_LEFT_CENSORED,
    AFT_MIN_SECOND_DER,
    AFT_RIGHT_CENSORED,
    AFT_UNCENSORED,
    COX,
    SURVIVAL_AFT,
    SurvivalAftParams,
    aft_derivative_limit,
    aft_dist_from_name,
    check_cox_hessian_declaration,
    check_cox_sample_weight,
    check_survival_aft_hessian_declaration,
    cox_grad_hess,
    cox_partial_log_likelihood,
    cox_signed_target,
    cox_sort_order,
    cox_varies_hessian,
    survival_aft_grad_hess,
    survival_aft_interval_distance,
    survival_aft_predicted_time,
    survival_aft_row_censoring,
    survival_aft_row_grad_hess,
    survival_aft_varies_hessian,
    train_cox,
    train_survival_aft,
)
from mojotrees.objective_registry import SQUARED_ERROR
from mojotrees.target_matrix import (
    AFT_UNBOUNDED,
    TargetMatrix,
    multi_targets,
    survival_aft_targets,
)
from mojotrees.tree import TreeParams


def close(a: Float64, b: Float64, tol: Float64 = 1e-12) -> Bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# The input contract
# ---------------------------------------------------------------------------


def test_target_matrix_layout_is_row_major() raises:
    var m = multi_targets([[1.0, 3.0], [2.0, 4.0]])
    assert_equal(m.n_rows, 2)
    assert_equal(m.n_targets, 2)
    # Row 0 is (1, 2), row 1 is (3, 4): the two columns interleaved.
    assert_true(close(m.values[0], 1.0))
    assert_true(close(m.values[1], 2.0))
    assert_true(close(m.values[2], 3.0))
    assert_true(close(m.values[3], 4.0))
    assert_true(close(m.get(1, 0), 3.0))
    var col1 = m.column(1)
    assert_true(close(col1[0], 2.0))
    assert_true(close(col1[1], 4.0))


def test_single_column_target_is_the_old_contract() raises:
    var m = TargetMatrix.from_single([1.0, 2.0, 3.0])
    assert_equal(m.n_rows, 3)
    assert_equal(m.n_targets, 1)


def test_target_matrix_refuses_a_ragged_length() raises:
    with assert_raises():
        _ = TargetMatrix([1.0, 2.0, 3.0], 2)
    with assert_raises():
        _ = TargetMatrix([1.0], 0)


def test_survival_aft_target_validation() raises:
    # The four censoring shapes all build.
    var ok = survival_aft_targets(
        [2.0, 2.0, AFT_UNBOUNDED, 1.0], [2.0, AFT_UNBOUNDED, 4.0, 4.0]
    )
    assert_equal(ok.n_rows, 4)
    assert_equal(survival_aft_row_censoring(2.0, 2.0), AFT_UNCENSORED)
    assert_equal(
        survival_aft_row_censoring(2.0, AFT_UNBOUNDED), AFT_RIGHT_CENSORED
    )
    assert_equal(
        survival_aft_row_censoring(AFT_UNBOUNDED, 4.0), AFT_LEFT_CENSORED
    )

    # (-1, -1) is read by CatBoost as an exact event at time -1 because the
    # equality test comes first. Refused here.
    with assert_raises():
        _ = survival_aft_targets([AFT_UNBOUNDED], [AFT_UNBOUNDED])
    # Non-positive bounds cannot go through log().
    with assert_raises():
        _ = survival_aft_targets([0.0], [1.0])
    # Inverted interval.
    with assert_raises():
        _ = survival_aft_targets([4.0], [2.0])
    # Wrong column count.
    with assert_raises():
        var m = TargetMatrix([1.0, 2.0, 3.0], 3)
        m.check_survival_aft()


# ---------------------------------------------------------------------------
# Cox
# ---------------------------------------------------------------------------


def test_cox_sort_is_by_absolute_time_and_stable() raises:
    # Keys are (3, 1, 3, 1); ascending with the original index as tiebreak
    # gives rows 1 and 3 (time 1) before rows 0 and 2 (time 3).
    var order = cox_sort_order([-3.0, 1.0, 3.0, -1.0])
    assert_equal(order[0], 1)
    assert_equal(order[1], 3)
    assert_equal(order[2], 0)
    assert_equal(order[3], 2)


def test_cox_signed_target_encoding() raises:
    var y = cox_signed_target([3.0, 5.0], [1, 0])
    assert_true(close(y[0], 3.0))
    assert_true(close(y[1], -5.0))
    # An event at time 0 would encode as y == 0 and read back as censored.
    with assert_raises():
        _ = cox_signed_target([0.0], [1])
    with assert_raises():
        _ = cox_signed_target([-1.0], [0])


def test_cox_derivatives_one_event_one_later_censor() raises:
    # Event at t = 1 (row 0), censored at t = 2 (row 1), both approx 0.
    # Sorted order is (0, 1). The event's risk set is both rows, S = 2:
    #   rk = 1/2, sk = 1/4
    #   row 0: grad = e^0 * 1/2 - 1 = -1/2, hess = 1/2 - 1/4 = 1/4
    #   row 1: grad = e^0 * 1/2     =  1/2, hess = 1/2 - 1/4 = 1/4
    var target: List[Float64] = [1.0, -2.0]
    var raw: List[Float64] = [0.0, 0.0]
    var grad = List[Float64]()
    var hess = List[Float64]()
    cox_grad_hess(raw, target, grad, hess)
    assert_true(close(grad[0], -0.5))
    assert_true(close(grad[1], 0.5))
    assert_true(close(hess[0], 0.25))
    assert_true(close(hess[1], 0.25))
    # The gradients of a Cox round always sum to zero: every event
    # contributes -1, and the e^{p_k}/S_i terms of one risk set sum to +1.
    assert_true(close(grad[0] + grad[1], 0.0))


def test_cox_ties_are_not_breslow() raises:
    # Two events at the same time. Breslow gives each the full risk set
    # S = 2, hence rk = 1/2 + 1/2 = 1 for both and grad = 1 - 1 = 0 apiece.
    # CatBoost's suffix rule gives the second event a risk set of one row:
    #   row 0: S = 2, rk = 1/2,       grad = 1/2 - 1 = -1/2
    #   row 1: S = 1, rk = 1/2 + 1,   grad = 3/2 - 1 = +1/2
    # so a Breslow implementation reads (0, 0) here and this one reads
    # (-1/2, +1/2).
    var target: List[Float64] = [1.0, 1.0]
    var raw: List[Float64] = [0.0, 0.0]
    var grad = List[Float64]()
    var hess = List[Float64]()
    cox_grad_hess(raw, target, grad, hess)
    assert_true(close(grad[0], -0.5))
    assert_true(close(grad[1], 0.5))
    assert_false(close(grad[0], 0.0))
    # hess: row 0 is 1/2 - 1/4 = 1/4; row 1 is 3/2 - (1/4 + 1) = 1/4.
    assert_true(close(hess[0], 0.25))
    assert_true(close(hess[1], 0.25))


def test_cox_is_invariant_to_a_constant_shift_of_every_score() raises:
    # (p_i + c) - log(sum_j e^{p_j + c}) is p_i - log(sum_j e^{p_j}), which
    # is why train_cox has no base score. The max-subtraction makes it exact
    # rather than merely analytic.
    var target: List[Float64] = [1.0, -2.0, 3.0]
    var g0 = List[Float64]()
    var h0 = List[Float64]()
    var g1 = List[Float64]()
    var h1 = List[Float64]()
    cox_grad_hess([0.0, 0.0, 0.0], target, g0, h0)
    cox_grad_hess([7.0, 7.0, 7.0], target, g1, h1)
    for r in range(3):
        assert_equal(g0[r], g1[r])
        assert_equal(h0[r], h1[r])


def test_cox_hessian_is_nonnegative_and_declared() raises:
    assert_true(cox_varies_hessian())
    with assert_raises():
        check_cox_hessian_declaration(True)
    check_cox_hessian_declaration(False)

    var target: List[Float64] = [1.0, -2.0, 3.0, -4.0]
    var grad = List[Float64]()
    var hess = List[Float64]()
    cox_grad_hess([0.3, -0.2, 1.5, 0.0], target, grad, hess)
    for r in range(4):
        assert_true(hess[r] >= 0.0)
    # And it is not the constant 1: declaring CONSTANT_HESSIAN beside Cox
    # would be a silently wrong hessian plane, which is what the refusal
    # above exists to stop.
    assert_false(close(hess[0], 1.0))


def test_cox_refuses_sample_weight_rather_than_dropping_it() raises:
    with assert_raises():
        check_cox_sample_weight([1.0, 1.0])
    check_cox_sample_weight([])


def test_cox_refuses_a_target_with_no_events() raises:
    var grad = List[Float64]()
    var hess = List[Float64]()
    # The derivative function itself does not check (it is on the round
    # loop's path); the trainer's check_cox_target does, and so does this.
    cox_grad_hess([0.0, 0.0], [-1.0, -2.0], grad, hess)
    assert_true(close(grad[0], 0.0))
    assert_true(close(grad[1], 0.0))


def test_cox_partial_log_likelihood() raises:
    # One event with a risk set of two equal scores: 0 - log(2).
    var ll = cox_partial_log_likelihood([0.0, 0.0], [1.0, -2.0])
    assert_true(close(ll, -log(2.0)))
    # Two tied events: the first sees S = 2, the second sees S = 1, so the
    # total is -log(2) - log(1) = -log(2) again.
    var tied = cox_partial_log_likelihood([0.0, 0.0], [1.0, 1.0])
    assert_true(close(tied, -log(2.0)))
    # Higher is better, and separating the event from the censored row in
    # the right direction raises it.
    var better = cox_partial_log_likelihood([1.0, 0.0], [1.0, -2.0])
    assert_true(better > ll)


# ---------------------------------------------------------------------------
# SurvivalAft
# ---------------------------------------------------------------------------


def test_aft_params_and_dist_names() raises:
    var p = SurvivalAftParams.catboost_default()
    assert_equal(p.dist, AFT_DIST_NORMAL)
    assert_true(close(p.scale, 1.0))
    p.validate()
    assert_equal(aft_dist_from_name("Normal"), AFT_DIST_NORMAL)
    assert_equal(aft_dist_from_name("logistic"), AFT_DIST_LOGISTIC)
    assert_equal(aft_dist_from_name("EXTREME"), AFT_DIST_EXTREME)
    with assert_raises():
        _ = aft_dist_from_name("weibull")
    with assert_raises():
        SurvivalAftParams(AFT_DIST_NORMAL, 0.0).validate()
    with assert_raises():
        SurvivalAftParams(7, 1.0).validate()


def test_survival_aft_normal_hessian_is_one_over_scale_squared() raises:
    # For the Normal AFT loss, an uncensored row has
    #   g = -z  and  h = 1/scale^2
    # identically, because f' = -z f and f'' = (z^2 - 1) f give
    #   -(f f'' - f'^2) / (scale f)^2 = -((z^2-1) - z^2) / scale^2.
    # So this checks pdf, pdf' and pdf'' together at three approxes rather
    # than one sampled point.
    var p = SurvivalAftParams(AFT_DIST_NORMAL, 1.0)
    var t = exp(2.0)
    for i in range(3):
        var a = -1.0 + Float64(i)
        var gh = survival_aft_row_grad_hess(a, t, t, p)
        assert_true(close(gh[0], -(2.0 - a), 1e-7))
        assert_true(close(gh[1], 1.0, 1e-7))

    var half = SurvivalAftParams(AFT_DIST_NORMAL, 0.5)
    var gh2 = survival_aft_row_grad_hess(0.0, exp(1.0), exp(1.0), half)
    # z = (1 - 0) / 0.5 = 2, so g = -z / scale = -4 and h = 1/scale^2 = 4.
    assert_true(close(gh2[0], -4.0, 1e-7))
    assert_true(close(gh2[1], 4.0, 1e-7))


def test_survival_aft_censored_derivatives() raises:
    var p = SurvivalAftParams(AFT_DIST_NORMAL, 1.0)

    # Right-censored at t = 1 with approx 0: z_L = 0, so
    #   P = 0 - phi(0) = -1/sqrt(2 pi),  D = 1 - 1/2 = 1/2,  Q = 0
    #   g = P / D = -2 phi(0) = -0.7978845608028654
    #   h = P^2 / D^2 = 4 phi(0)^2 = 2 / pi = 0.6366197723675814
    var right = survival_aft_row_grad_hess(0.0, 1.0, AFT_UNBOUNDED, p)
    assert_true(close(right[0], -0.7978845608028654, 1e-7))
    assert_true(close(right[1], 0.6366197723675814, 1e-7))

    # Left-censored at t = 1 is the exact mirror in the gradient and the
    # same curvature.
    var left = survival_aft_row_grad_hess(0.0, AFT_UNBOUNDED, 1.0, p)
    assert_true(close(left[0], 0.7978845608028654, 1e-7))
    assert_true(close(left[1], 0.6366197723675814, 1e-7))

    # Interval (1, e) with approx 0, computed independently in Python from
    # the transcribed Normal pdf/cdf.
    var interval = survival_aft_row_grad_hess(0.0, 1.0, exp(1.0), p)
    assert_true(close(interval[0], -0.45986222928642656, 1e-7))
    assert_true(close(interval[1], 0.9203481751514889, 1e-7))


def test_survival_aft_logistic_and_extreme_at_the_median() raises:
    # Logistic at z = 0: pdf = 1/4, pdf' = 0, pdf'' = -1/8, so g = 0 and
    #   h = -(1/4 * -1/8 - 0) / (1/4)^2 = 1/2.
    var logistic = SurvivalAftParams(AFT_DIST_LOGISTIC, 1.0)
    var gl = survival_aft_row_grad_hess(0.0, 1.0, 1.0, logistic)
    assert_true(close(gl[0], 0.0, 1e-7))
    assert_true(close(gl[1], 0.5, 1e-7))

    # Extreme at z = 0: pdf = e^-1, pdf' = 0, pdf'' = -e^-1, so g = 0, h = 1.
    var extreme = SurvivalAftParams(AFT_DIST_EXTREME, 1.0)
    var ge = survival_aft_row_grad_hess(0.0, 1.0, 1.0, extreme)
    assert_true(close(ge[0], 0.0, 1e-7))
    assert_true(close(ge[1], 1.0, 1e-7))


def test_survival_aft_hessian_is_positive_and_declared() raises:
    assert_true(survival_aft_varies_hessian())
    with assert_raises():
        check_survival_aft_hessian_declaration(True)
    check_survival_aft_hessian_declaration(False)

    # The 1e-16 floor is what makes a leaf denominator safe before
    # reg_lambda: drive the approx far from the data and the hessian bottoms
    # out at the floor rather than reaching zero.
    var p = SurvivalAftParams(AFT_DIST_LOGISTIC, 1.0)
    var far = survival_aft_row_grad_hess(-400.0, 1.0, AFT_UNBOUNDED, p)
    assert_true(far[1] >= AFT_MIN_SECOND_DER)
    assert_true(far[1] <= 15.0)
    assert_true(far[0] >= -15.0)
    assert_true(far[0] <= 15.0)


def test_aft_derivative_limit_table_pair_order_is_preserved() raises:
    # Two entries of CatBoost's table put the larger value in the `min` slot
    # and are transcribed as written, because the caller uses the pair as
    # `targetSign ? min : max` and normalizing them would change answers.
    var n2 = aft_derivative_limit(AFT_DIST_NORMAL, 1, AFT_RIGHT_CENSORED, 0.5)
    assert_true(close(n2[0], 4.0))
    assert_true(close(n2[1], AFT_MIN_SECOND_DER))
    var e2 = aft_derivative_limit(AFT_DIST_EXTREME, 1, AFT_UNCENSORED, 1.0)
    assert_true(close(e2[0], 15.0))
    assert_true(close(e2[1], AFT_MIN_SECOND_DER))
    # And the ordinary entries.
    var n1 = aft_derivative_limit(AFT_DIST_NORMAL, 0, AFT_RIGHT_CENSORED, 1.0)
    assert_true(close(n1[0], -15.0))
    assert_true(close(n1[1], 0.0))
    var l1 = aft_derivative_limit(AFT_DIST_LOGISTIC, 0, AFT_LEFT_CENSORED, 2.0)
    assert_true(close(l1[0], 0.0))
    assert_true(close(l1[1], 0.5))


def test_survival_aft_applies_sample_weight() raises:
    # CatBoost's TSurvivalAftError::CalcDers takes `float /*weight*/` and
    # drops it. We multiply both derivatives by it, which is the deliberate
    # divergence recorded in catalog A27.
    var targets = survival_aft_targets([1.0, 1.0], [1.0, 1.0])
    var p = SurvivalAftParams(AFT_DIST_NORMAL, 1.0)
    var raw: List[Float64] = [1.0, 1.0]
    var g_unw = List[Float64]()
    var h_unw = List[Float64]()
    survival_aft_grad_hess(raw, targets, p, [], g_unw, h_unw)
    var g_w = List[Float64]()
    var h_w = List[Float64]()
    survival_aft_grad_hess(raw, targets, p, [1.0, 3.0], g_w, h_w)
    assert_true(close(g_w[0], g_unw[0], 1e-12))
    assert_true(close(g_w[1], 3.0 * g_unw[1], 1e-6))
    assert_true(close(h_w[1], 3.0 * h_unw[1], 1e-6))


def test_survival_aft_interval_distance_metric() raises:
    # Row 0: exact event at 2, predicted time exactly 2 -> distance 0.
    # Row 1: right-censored at 2, predicted time 1 -> below the bound, so
    #        the penalty is |1 - 2| = 1 (the +infinity upper bound loses the
    #        min).
    # Row 2: interval (1, 4), predicted time 2 -> strictly inside, no
    #        penalty.
    var targets = survival_aft_targets(
        [2.0, 2.0, 1.0], [2.0, AFT_UNBOUNDED, 4.0]
    )
    var raw: List[Float64] = [log(2.0), 0.0, log(2.0)]
    var m = survival_aft_interval_distance(raw, targets)
    assert_true(close(m, 1.0 / 3.0, 1e-7))
    # Lower is better: moving the censored row's prediction above its lower
    # bound removes its penalty entirely.
    var raw2: List[Float64] = [log(2.0), log(3.0), log(2.0)]
    assert_true(close(survival_aft_interval_distance(raw2, targets), 0.0, 1e-7))


def test_survival_aft_predicted_time_is_the_exp_link() raises:
    assert_true(close(survival_aft_predicted_time(log(5.0)), 5.0, 1e-7))
    # The two objective codes staged in survival.mojo are distinct and sit
    # after CROSS_ENTROPY = 12. **16 and 17, not the 13 and 14 this lane
    # staged**: `catboost_ranking.mojo` took 13/14/15 for QueryRMSE,
    # PairLogit and YetiRank in the same round, and the two lanes could not
    # see each other. Renumbered on merge.
    #
    # This assertion is worth keeping precisely because it caught that. An
    # objective code is a number in a serialized model, so two objectives
    # sharing one is not a merge conflict -- it is a model that loads as the
    # wrong loss, applies the wrong inverse link, and raises nothing.
    assert_equal(COX, 16)
    assert_equal(SURVIVAL_AFT, 17)


# ---------------------------------------------------------------------------
# MultiRMSE
# ---------------------------------------------------------------------------


def test_multi_rmse_derivatives() raises:
    # grad = w * (raw - y), hess = w, one entry per (row, target).
    var targets = multi_targets([[1.0, 3.0], [2.0, 4.0]])
    var raw: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var grad = List[Float64]()
    var hess = List[Float64]()
    multi_rmse_grad_hess(raw, targets, [], grad, hess)
    assert_equal(len(grad), 4)
    assert_true(close(grad[0], -1.0))
    assert_true(close(grad[1], -2.0))
    assert_true(close(grad[2], -3.0))
    assert_true(close(grad[3], -4.0))
    for i in range(4):
        assert_true(close(hess[i], 1.0))

    var gw = List[Float64]()
    var hw = List[Float64]()
    multi_rmse_grad_hess(raw, targets, [2.0, 0.5], gw, hw)
    assert_true(close(gw[0], -2.0))
    assert_true(close(hw[0], 2.0))
    assert_true(close(gw[2], -1.5))
    assert_true(close(hw[3], 0.5))


def test_multi_rmse_missing_values_zero_both_derivatives() raises:
    var nan = Float64(0.0) / Float64(0.0)
    var targets = TargetMatrix([1.0, nan, 3.0, 4.0], 2)
    var raw: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var grad = List[Float64]()
    var hess = List[Float64]()
    multi_rmse_grad_hess(raw, targets, [], grad, hess, True)
    assert_true(close(grad[0], -1.0))
    assert_true(close(grad[1], 0.0))
    assert_true(close(hess[1], 0.0))
    assert_true(close(grad[2], -3.0))
    assert_true(close(hess[2], 1.0))


def test_multi_rmse_metric_is_the_euclidean_norm_not_the_mean() raises:
    # sum over targets and rows of (0 - y)^2 is 1 + 4 + 9 + 16 = 30; the
    # denominator is the ROW weight sum, 2, not 2 * T. So the metric is
    # sqrt(15), not sqrt(30/4).
    var targets = multi_targets([[1.0, 3.0], [2.0, 4.0]])
    var raw: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var m = multi_rmse(raw, targets)
    assert_true(close(m, sqrt(15.0), 1e-12))
    assert_false(close(m, sqrt(7.5), 1e-6))
    # At T = 1 it is exactly RMSE.
    var one = TargetMatrix([3.0, 4.0], 1)
    assert_true(close(multi_rmse([0.0, 0.0], one), sqrt(12.5), 1e-12))


def test_multi_target_gain_is_not_the_gain_of_summed_planes() raises:
    # Two targets that disagree about which side of the split they favor.
    # Per plane: 1^2/1 + 1^2/1 - 0^2/2 = 2, so the sum over targets is 4.
    # The elementwise sum of the planes has G_left = G_right = 0, so the
    # gain of the summed planes is 0. CatBoost computes 4.
    var left_g: List[Float64] = [1.0, -1.0]
    var left_h: List[Float64] = [1.0, 1.0]
    var right_g: List[Float64] = [-1.0, 1.0]
    var right_h: List[Float64] = [1.0, 1.0]
    var parent_g: List[Float64] = [0.0, 0.0]
    var parent_h: List[Float64] = [2.0, 2.0]
    var gain = multi_target_split_gain(
        left_g, left_h, right_g, right_h, parent_g, parent_h, 0.0, 0.0
    )
    assert_true(close(gain, 4.0, 1e-12))

    # The wrong answer, spelled out so the contrast is in the test and not
    # only in a comment: fold the planes first and the same candidate scores
    # zero.
    var folded = multi_target_split_gain(
        [0.0], [2.0], [0.0], [2.0], [0.0], [4.0], 0.0, 0.0
    )
    assert_true(close(folded, 0.0, 1e-12))


def test_multi_target_leaf_values_are_one_newton_step_per_target() raises:
    var v = multi_target_leaf_values([2.0, -4.0], [4.0, 4.0], 0.0, 0.0)
    assert_equal(len(v), 2)
    assert_true(close(v[0], -0.5))
    assert_true(close(v[1], 1.0))
    # lambda_l2 lands in each plane's denominator separately.
    var reg = multi_target_leaf_values([2.0, -4.0], [4.0, 4.0], 0.0, 1.0)
    assert_true(close(reg[0], -0.4))
    assert_true(close(reg[1], 0.8))
    with assert_raises():
        _ = multi_target_leaf_values([1.0, 2.0], [1.0], 0.0, 0.0)


def test_multi_target_hessian_declaration_is_conditional() raises:
    # Unweighted and complete, the hessian is the literal 1.0, so a constant
    # declaration is legal -- unlike Cox and SurvivalAft.
    assert_false(multi_target_varies_hessian(False, False))
    check_multi_target_hessian_declaration(False, False, True)
    assert_true(multi_target_varies_hessian(True, False))
    assert_true(multi_target_varies_hessian(False, True))
    with assert_raises():
        check_multi_target_hessian_declaration(True, False, True)
    with assert_raises():
        check_multi_target_hessian_declaration(False, True, True)
    check_multi_target_hessian_declaration(True, True, False)
    assert_equal(MULTI_RMSE, -2)
    assert_equal(MULTI_RMSE_WITH_MISSING, -3)


# ---------------------------------------------------------------------------
# End to end, on eight rows. These exist to prove the three trainers run and
# that the three input contracts reach a grower, not to measure anything.
# ---------------------------------------------------------------------------


def _binned() raises -> BinnedMatrix:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    return bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)


def _params(n_rounds: Int) raises -> BoosterParams:
    return BoosterParams(n_rounds, 0.3, TreeParams(4, 1, 1.0, 1e-3))


def test_train_cox_runs_and_orders_the_risk() raises:
    # Later feature values are events at earlier times, so a higher hazard
    # should be learned for them: predictions must be increasing in the
    # feature by the end.
    var data = _binned()
    var y = cox_signed_target(
        [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [1, 1, 1, 1, 1, 1, 1, 1]
    )
    var booster = train_cox(data, y, _params(20))
    assert_equal(booster.objective, COX)
    assert_true(len(booster.trees) > 0)
    var first = booster.predict_raw_row(data, 0)
    var last = booster.predict_raw_row(data, 7)
    assert_true(last > first)
    # And the partial log likelihood improved on the zero start.
    var raw = List[Float64]()
    var start = List[Float64]()
    for r in range(8):
        raw.append(booster.predict_raw_row(data, r))
        start.append(0.0)
    assert_true(
        cox_partial_log_likelihood(raw, y)
        > cox_partial_log_likelihood(start, y)
    )


def test_train_cox_refuses_weights_and_an_all_censored_target() raises:
    var data = _binned()
    var y = cox_signed_target([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                              [1, 0, 1, 0, 1, 0, 1, 0])
    with assert_raises():
        _ = train_cox(
            data,
            y,
            _params(2),
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
    var censored = cox_signed_target(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [0, 0, 0, 0, 0, 0, 0, 0]
    )
    with assert_raises():
        _ = train_cox(data, censored, _params(2))


def test_train_survival_aft_runs_on_a_two_column_target() raises:
    # Exact events whose times rise with the feature, plus one right- and
    # one left-censored row, so all three censoring branches are on the
    # training path.
    var data = _binned()
    var lower: List[Float64] = [
        1.0, 2.0, 3.0, AFT_UNBOUNDED, 5.0, 6.0, 7.0, 8.0
    ]
    var upper: List[Float64] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, AFT_UNBOUNDED, 8.0
    ]
    var targets = survival_aft_targets(lower, upper)
    var booster = train_survival_aft(data, targets, _params(30))
    assert_equal(booster.objective, SURVIVAL_AFT)
    assert_true(len(booster.trees) > 0)
    # The predicted time is exp(raw), and it rises with the feature.
    var t0 = survival_aft_predicted_time(booster.predict_raw_row(data, 0))
    var t7 = survival_aft_predicted_time(booster.predict_raw_row(data, 7))
    assert_true(t7 > t0)
    # The interval metric improved on the zero start.
    var raw = List[Float64]()
    var start = List[Float64]()
    for r in range(8):
        raw.append(booster.predict_raw_row(data, r))
        start.append(0.0)
    assert_true(
        survival_aft_interval_distance(raw, targets)
        < survival_aft_interval_distance(start, targets)
    )


def test_train_multi_rmse_is_t_independent_squared_error_fits() raises:
    # The documented consequence of growing one tree per target instead of
    # one tree with a vector leaf value: because the MultiRMSE derivative has
    # no cross-target term, this shape IS T separate SQUARED_ERROR boosters,
    # bit for bit. If that ever stops being true, the shape has changed and
    # the A28 caveat has to be rewritten.
    var data = _binned()
    var c0: List[Float64] = [0.5, 0.5, 1.5, 1.5, 4.0, 4.0, 9.0, 9.0]
    var c1: List[Float64] = [9.0, 9.0, 4.0, 4.0, 1.5, 1.5, 0.5, 0.5]
    var targets = multi_targets([c0.copy(), c1.copy()])
    var multi = train_multi_rmse(data, targets, _params(12))
    assert_equal(multi.n_targets, 2)
    assert_equal(multi.n_iterations(), 12)
    # 24 trees for 12 iterations: the number comparable to CatBoost's
    # tree_count_ is 12, not 24.
    assert_equal(len(multi.trees), 24)

    var solo0 = train(data, c0, SQUARED_ERROR, _params(12))
    var solo1 = train(data, c1, SQUARED_ERROR, _params(12))
    for r in range(8):
        var p = multi.predict_row(data, r)
        assert_equal(p[0], solo0.predict_raw_row(data, r))
        assert_equal(p[1], solo1.predict_raw_row(data, r))

    # And the metric it optimizes fell.
    var raw = List[Float64]()
    var start = List[Float64]()
    for r in range(8):
        var p = multi.predict_row(data, r)
        raw.append(p[0])
        raw.append(p[1])
        start.append(0.0)
        start.append(0.0)
    assert_true(multi_rmse(raw, targets) < multi_rmse(start, targets))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
