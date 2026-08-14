from std.math import log, sqrt
from std.testing import assert_equal, assert_true, TestSuite

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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
