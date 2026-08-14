from std.math import log, sqrt
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojoboost.metrics import (
    average_precision,
    binary_error,
    check_metric_weight,
    cross_entropy_loss,
    fair_loss,
    gamma_deviance,
    gamma_loss,
    huber_loss,
    kullback_leibler,
    l1,
    l2,
    mape,
    multiclass_error,
    poisson_loss,
    quantile_loss,
    tweedie_loss,
)

from mojoboost import (
    BoosterParams,
    TreeParams,
    binary_accuracy,
    binary_auc,
    binary_log_loss,
    fit_bins,
    multiclass_accuracy,
    multiclass_log_loss,
    rmse,
    train_multiclass_with_valid,
)


def small_tree_params() -> TreeParams:
    return TreeParams(4, 1, 1.0, 1e-3)


def test_rmse_known() raises:
    var pred: List[Float64] = [0.0, 0.0]
    var target: List[Float64] = [3.0, 4.0]
    assert_true(abs(rmse(pred, target) - sqrt(12.5)) < 1e-12)
    assert_true(abs(rmse(target, target)) < 1e-12)


def test_binary_log_loss_known() raises:
    var probs: List[Float64] = [0.5, 0.5]
    var labels: List[Float64] = [0.0, 1.0]
    assert_true(abs(binary_log_loss(probs, labels) - log(2.0)) < 1e-8)
    var confident: List[Float64] = [0.001, 0.999]
    assert_true(binary_log_loss(confident, labels) < 0.01)


def test_binary_accuracy_known() raises:
    var probs: List[Float64] = [0.9, 0.4, 0.6, 0.2]
    var labels: List[Float64] = [1.0, 0.0, 0.0, 0.0]
    assert_true(abs(binary_accuracy(probs, labels) - 0.75) < 1e-12)


def test_binary_auc_perfect_and_reversed() raises:
    var scores: List[Float64] = [0.1, 0.2, 0.8, 0.9]
    var labels: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    assert_true(abs(binary_auc(scores, labels) - 1.0) < 1e-12)
    var flipped: List[Float64] = [1.0, 1.0, 0.0, 0.0]
    assert_true(abs(binary_auc(scores, flipped)) < 1e-12)


def test_binary_auc_known() raises:
    # sklearn's doc example: roc_auc_score = 0.75. One positive (0.35)
    # loses to one negative (0.4); the other three pairs are ordered right.
    var scores: List[Float64] = [0.1, 0.4, 0.35, 0.8]
    var labels: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    assert_true(abs(binary_auc(scores, labels) - 0.75) < 1e-12)


def test_binary_auc_ties() raises:
    # A constant score carries no information: AUC must be exactly 0.5.
    var scores: List[Float64] = [0.3, 0.3, 0.3, 0.3]
    var labels: List[Float64] = [0.0, 1.0, 0.0, 1.0]
    assert_true(abs(binary_auc(scores, labels) - 0.5) < 1e-12)


def test_binary_auc_validates() raises:
    var scores: List[Float64] = [0.1, 0.2]
    var labels: List[Float64] = [1.0, 1.0]
    var raised = False
    try:
        _ = binary_auc(scores, labels)
    except:
        raised = True
    assert_true(raised)


def test_mape_known() raises:
    # 0.5/1 (the |y| < 1 floor applies), 0/2, 7/10.
    var pred: List[Float64] = [1.0, 2.0, 3.0]
    var target: List[Float64] = [0.5, 2.0, 10.0]
    assert_true(abs(mape(pred, target) - 0.4) < 1e-12)
    assert_true(abs(mape(target, target)) < 1e-12)


def test_mape_weight_equals_duplicating_the_row() raises:
    # The weighted mean is the mean over a sample with that row repeated,
    # which is what a weight means.
    var pred: List[Float64] = [1.0, 3.0]
    var target: List[Float64] = [2.0, 4.0]
    var weight: List[Float64] = [2.0, 1.0]
    var dup_pred: List[Float64] = [1.0, 1.0, 3.0]
    var dup_target: List[Float64] = [2.0, 2.0, 4.0]
    assert_true(
        abs(mape(pred, target, weight) - mape(dup_pred, dup_target)) < 1e-12
    )


def test_fair_loss_known() raises:
    # c = 1, residual 1: 1 - log 2.
    var pred: List[Float64] = [1.0]
    var target: List[Float64] = [0.0]
    assert_true(abs(fair_loss(pred, target, 1.0) - (1.0 - log(2.0))) < 1e-12)
    # A perfect prediction is exactly 0 at any c.
    var same: List[Float64] = [2.5, -1.0]
    assert_true(abs(fair_loss(same, same, 3.0)) < 1e-12)


def test_fair_loss_large_c_approaches_half_squared_error() raises:
    # c^2 * (d/c - log(1 + d/c)) -> d^2 / 2.
    var pred: List[Float64] = [1.0, -2.0]
    var target: List[Float64] = [0.0, 0.0]
    var expected = 0.5 * (1.0 + 4.0) / 2.0
    assert_true(abs(fair_loss(pred, target, 1e6) - expected) < 1e-4)


def test_poisson_loss_is_minimized_at_the_label() raises:
    # mu - y log mu, minimized at mu = y.
    var target: List[Float64] = [3.0]
    var at: List[Float64] = [3.0]
    var below: List[Float64] = [2.0]
    var above: List[Float64] = [4.0]
    var best = poisson_loss(at, target)
    assert_true(best < poisson_loss(below, target))
    assert_true(best < poisson_loss(above, target))
    assert_true(abs(best - (3.0 - 3.0 * log(3.0))) < 1e-10)


def test_poisson_loss_rejects_a_nonpositive_prediction() raises:
    # The metric takes the response scale, exp(raw), which is positive; a
    # zero here means a caller passed raw scores by mistake.
    var pred: List[Float64] = [1.0, 0.0]
    var target: List[Float64] = [1.0, 1.0]
    var raised = False
    try:
        _ = poisson_loss(pred, target)
    except:
        raised = True
    assert_true(raised)


def test_gamma_loss_is_minimized_at_the_label() raises:
    # y / mu + log mu, minimized at mu = y with value 1 + log y.
    var target: List[Float64] = [2.0]
    var at: List[Float64] = [2.0]
    var below: List[Float64] = [1.0]
    var above: List[Float64] = [4.0]
    assert_true(abs(gamma_loss(at, target) - (1.0 + log(2.0))) < 1e-10)
    assert_true(gamma_loss(at, target) < gamma_loss(below, target))
    assert_true(gamma_loss(at, target) < gamma_loss(above, target))


def test_gamma_deviance_known() raises:
    # 2 * (r - log r - 1) at r = y / mu, and exactly 0 when they agree.
    var target: List[Float64] = [2.0]
    var pred: List[Float64] = [1.0]
    var expected = 2.0 * (2.0 - log(2.0) - 1.0)
    assert_true(abs(gamma_deviance(pred, target) - expected) < 1e-10)
    assert_true(abs(gamma_deviance(target, target)) < 1e-10)


def test_tweedie_loss_known_at_variance_power_1_5() raises:
    # rho = 1.5 collapses to 2y / sqrt(mu) + 2 sqrt(mu), which at mu = y = 4
    # is 4 + 4.
    var target: List[Float64] = [4.0]
    var at: List[Float64] = [4.0]
    assert_true(abs(tweedie_loss(at, target, 1.5) - 8.0) < 1e-10)
    var away: List[Float64] = [1.0]
    assert_true(tweedie_loss(at, target, 1.5) < tweedie_loss(away, target, 1.5))


def test_tweedie_loss_validates_variance_power() raises:
    var pred: List[Float64] = [1.0]
    var target: List[Float64] = [1.0]
    var bad: List[Float64] = [1.0, 2.0, 0.5]
    for b in range(len(bad)):
        var raised = False
        try:
            _ = tweedie_loss(pred, target, bad[b])
        except:
            raised = True
        assert_true(raised)


def test_cross_entropy_matches_binary_log_loss_on_hard_labels() raises:
    # With {0, 1} labels one of the two terms vanishes, so the continuous
    # metric has to reduce to the binary one.
    var probs: List[Float64] = [0.2, 0.7, 0.5, 0.9]
    var labels: List[Float64] = [0.0, 1.0, 1.0, 0.0]
    assert_true(
        abs(cross_entropy_loss(probs, labels) - binary_log_loss(probs, labels))
        < 1e-12
    )


def test_cross_entropy_known_on_soft_labels() raises:
    # Both terms count: at p = 0.5 every label scores log 2, whatever it is.
    var probs: List[Float64] = [0.5, 0.5, 0.5]
    var labels: List[Float64] = [0.25, 0.5, 0.75]
    assert_true(abs(cross_entropy_loss(probs, labels) - log(2.0)) < 1e-10)


def test_cross_entropy_validates_labels() raises:
    var probs: List[Float64] = [0.5, 0.5]
    var labels: List[Float64] = [0.5, 1.5]
    var raised = False
    try:
        _ = cross_entropy_loss(probs, labels)
    except:
        raised = True
    assert_true(raised)


def test_kullback_leibler_is_cross_entropy_minus_label_entropy() raises:
    # The defining identity, checked on soft labels where the entropy term
    # is not zero.
    var probs: List[Float64] = [0.25, 0.6, 0.5]
    var labels: List[Float64] = [0.5, 0.5, 0.25]
    var entropy = 0.0
    for r in range(3):
        var y = labels[r]
        entropy -= y * log(y) + (1.0 - y) * log(1.0 - y)
    entropy /= 3.0
    var expected = cross_entropy_loss(probs, labels) - entropy
    assert_true(abs(kullback_leibler(probs, labels) - expected) < 1e-10)


def test_kullback_leibler_is_zero_on_a_perfect_prediction() raises:
    var labels: List[Float64] = [0.0, 0.25, 0.5, 1.0]
    assert_true(abs(kullback_leibler(labels, labels)) < 1e-10)


def test_average_precision_known() raises:
    # scikit-learn's documented example for average_precision_score:
    # 0.83333... on these four rows.
    var scores: List[Float64] = [0.1, 0.4, 0.35, 0.8]
    var labels: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    assert_true(abs(average_precision(scores, labels) - 5.0 / 6.0) < 1e-12)


def test_average_precision_perfect_and_tied() raises:
    var scores: List[Float64] = [0.1, 0.2, 0.8, 0.9]
    var perfect: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    assert_true(abs(average_precision(scores, perfect) - 1.0) < 1e-12)
    # A constant score is one tied block: precision is the positive rate.
    var flat: List[Float64] = [0.3, 0.3, 0.3, 0.3]
    var mixed: List[Float64] = [0.0, 1.0, 0.0, 1.0]
    assert_true(abs(average_precision(flat, mixed) - 0.5) < 1e-12)


def test_average_precision_validates() raises:
    var scores: List[Float64] = [0.1, 0.2]
    var labels: List[Float64] = [0.0, 0.0]
    var raised = False
    try:
        _ = average_precision(scores, labels)
    except:
        raised = True
    assert_true(raised)


def test_multiclass_log_loss_known() raises:
    # Uniform probabilities over 3 classes give log 3. Tolerance is loose
    # because std.math log is only accurate to about 1e-10, so log(1/3)
    # and -log(3) disagree beyond that.
    var probs = List[Float64]()
    for _ in range(6):
        probs.append(1.0 / 3.0)
    var labels: List[Int] = [0, 2]
    assert_true(
        abs(multiclass_log_loss(probs, labels, 3) - log(3.0)) < 1e-8
    )


def test_multiclass_accuracy_known() raises:
    var probs: List[Float64] = [
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3,
        0.5, 0.3, 0.2,
    ]
    var labels: List[Int] = [0, 1, 2]
    assert_true(abs(multiclass_accuracy(probs, labels, 3) - 2.0 / 3.0) < 1e-12)


def test_multiclass_early_stopping_truncates() raises:
    # Same three-cluster problem the plain multiclass test fits. With the
    # training set as validation, improvement drops below min_delta once
    # the clusters are separated, so training must stop well short of
    # n_estimators while still classifying every row correctly. min_delta
    # is 1e-4 because the softmax hessian 2p(1-p) damps the Newton step,
    # so per-round log loss improvement shrinks slowly.
    var features: List[Float64] = [
        0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0,
    ]
    var labels: List[Int] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    var mapper = fit_bins(features, n_rows=9, n_features=1, max_bins=16)
    var data = mapper.transform(features, 9)
    var params = BoosterParams(300, 0.2, small_tree_params())
    var model = train_multiclass_with_valid(
        data, labels, data, labels, 3, params,
        early_stopping_rounds=5, min_delta=1e-4,
    )
    assert_true(len(model.trees) > 0)
    assert_true(len(model.trees) < 300 * 3)
    assert_equal(len(model.trees) % 3, 0)
    for r in range(9):
        var row: List[Float64] = [features[r]]
        var proba = model.predict_proba_bins(mapper.bin_row(row))
        var argmax = 0
        for k in range(3):
            if proba[k] > proba[argmax]:
                argmax = k
        assert_equal(argmax, labels[r])


def test_multiclass_early_stopping_prevents_overfit() raises:
    # Validation labels are rotated one class over, so every round of
    # fitting the training labels makes validation loss worse. The
    # returned ensemble must be empty (base scores only).
    var features: List[Float64] = [
        0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0,
    ]
    var labels: List[Int] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    var rotated: List[Int] = [1, 1, 1, 2, 2, 2, 0, 0, 0]
    var mapper = fit_bins(features, n_rows=9, n_features=1, max_bins=16)
    var data = mapper.transform(features, 9)
    var params = BoosterParams(300, 0.2, small_tree_params())
    var model = train_multiclass_with_valid(
        data, labels, data, rotated, 3, params, early_stopping_rounds=3,
    )
    assert_equal(len(model.trees), 0)


def test_weighted_means_are_weighted() raises:
    var pred: List[Float64] = [0.0, 0.0, 0.0]
    var target: List[Float64] = [1.0, 2.0, 3.0]
    var weight: List[Float64] = [1.0, 2.0, 3.0]
    # (1*1 + 2*4 + 3*9) / 6 and (1*1 + 2*2 + 3*3) / 6.
    assert_true(abs(l2(pred, target, weight) - 6.0) < 1e-12)
    assert_true(abs(l1(pred, target, weight) - 14.0 / 6.0) < 1e-12)
    assert_true(
        abs(rmse(pred, target, weight) - sqrt(6.0)) < 1e-12
    )


def test_uniform_weights_match_the_unweighted_metric() raises:
    var pred: List[Float64] = [0.5, 1.5, 2.5]
    var target: List[Float64] = [1.0, 2.0, 3.0]
    var weight: List[Float64] = [3.0, 3.0, 3.0]
    assert_true(abs(l2(pred, target, weight) - l2(pred, target)) < 1e-12)
    assert_true(abs(l1(pred, target, weight) - l1(pred, target)) < 1e-12)
    var probs: List[Float64] = [0.8, 0.3, 0.6]
    var labels: List[Float64] = [1.0, 0.0, 1.0]
    assert_true(
        abs(
            binary_log_loss(probs, labels, weight)
            - binary_log_loss(probs, labels)
        )
        < 1e-12
    )
    assert_true(
        abs(binary_auc(probs, labels, weight) - binary_auc(probs, labels))
        < 1e-12
    )


def test_integer_weights_match_duplicated_rows() raises:
    # An AUC row of weight 2 must count exactly as two copies of that row,
    # ties included; that is what makes the weighted rank statistic the
    # same statistic.
    var scores: List[Float64] = [0.1, 0.4, 0.35, 0.8]
    var labels: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    var weight: List[Float64] = [1.0, 2.0, 1.0, 1.0]
    var doubled_scores: List[Float64] = [0.1, 0.4, 0.4, 0.35, 0.8]
    var doubled_labels: List[Float64] = [0.0, 0.0, 0.0, 1.0, 1.0]
    assert_true(
        abs(
            binary_auc(scores, labels, weight)
            - binary_auc(doubled_scores, doubled_labels)
        )
        < 1e-12
    )


def test_tied_scores_split_the_pair() raises:
    # One positive and one negative at the same score: the pair is half won.
    var scores: List[Float64] = [1.0, 1.0]
    var labels: List[Float64] = [0.0, 1.0]
    assert_true(abs(binary_auc(scores, labels) - 0.5) < 1e-12)


def test_bad_weights_raise() raises:
    var pred: List[Float64] = [0.0, 0.0]
    var target: List[Float64] = [1.0, 2.0]
    var short: List[Float64] = [1.0]
    with assert_raises(contains="weight length"):
        _ = l2(pred, target, short)
    var negative: List[Float64] = [1.0, -1.0]
    with assert_raises(contains="nonnegative"):
        _ = l2(pred, target, negative)
    var zeros: List[Float64] = [0.0, 0.0]
    with assert_raises(contains="positive sum"):
        _ = l2(pred, target, zeros)
    assert_equal(check_metric_weight(List[Float64](), 4), 4.0)


def test_quantile_and_huber_losses_known() raises:
    var over: List[Float64] = [1.0]
    var under: List[Float64] = [0.0]
    var zero: List[Float64] = [0.0]
    var one: List[Float64] = [1.0]
    # Over-predicting costs (1 - alpha) per unit, under-predicting alpha.
    assert_true(abs(quantile_loss(over, zero, 0.25) - 0.75) < 1e-12)
    assert_true(abs(quantile_loss(under, one, 0.25) - 0.25) < 1e-12)
    with assert_raises(contains="alpha"):
        _ = quantile_loss(over, zero, 1.0)
    # Quadratic inside the transition point, linear outside it.
    var far: List[Float64] = [3.0]
    var near: List[Float64] = [0.5]
    assert_true(abs(huber_loss(far, zero, 1.0) - 2.5) < 1e-12)
    assert_true(abs(huber_loss(near, zero, 1.0) - 0.125) < 1e-12)


def test_error_rates_complement_accuracy() raises:
    var probs: List[Float64] = [0.9, 0.2, 0.6, 0.4]
    var labels: List[Float64] = [1.0, 0.0, 0.0, 1.0]
    assert_true(
        abs(
            binary_error(probs, labels) + binary_accuracy(probs, labels) - 1.0
        )
        < 1e-12
    )
    var multi: List[Float64] = [0.7, 0.2, 0.1, 0.1, 0.8, 0.1]
    var codes: List[Int] = [0, 2]
    assert_true(
        abs(
            multiclass_error(multi, codes, 3)
            + multiclass_accuracy(multi, codes, 3)
            - 1.0
        )
        < 1e-12
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
