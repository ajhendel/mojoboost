from std.testing import assert_equal, assert_true, TestSuite

from mojoboost import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BoosterParams,
    Tree,
    TreeParams,
    balanced_class_weights,
    balanced_sample_weight,
    bin_equal_width,
    binary_labels_to_codes,
    check_class_balance_params,
    class_weight_rows,
    gain_importance,
    grow_tree,
    scale_pos_weight_rows,
    split_importance,
    train,
    train_multiclass,
    unbalance_scale,
    unbalanced_sample_weight,
)


def small_tree_params() -> TreeParams:
    return TreeParams(4, 1, 1.0, 1e-3)


def test_weight_two_equals_duplicated_row() raises:
    # Weighting a row by 2 must produce the exact same model as training
    # on a dataset where that row appears twice: gradient and hessian
    # sums per bin are identical, so every split and leaf value matches.
    var features_a: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var target_a: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var weights_a: List[Float64] = [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    var data_a = bin_equal_width(features_a, n_rows=8, n_features=1, n_bins=8)

    var features_b: List[Float64] = [
        0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    ]
    var target_b: List[Float64] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ]
    var data_b = bin_equal_width(
        features_b, n_rows=12, n_features=1, n_bins=8
    )

    var params = BoosterParams(20, 0.3, small_tree_params())
    var model_a = train(data_a, target_a, SQUARED_ERROR, params, weights_a)
    var model_b = train(data_b, target_b, SQUARED_ERROR, params)
    assert_equal(len(model_a.trees), len(model_b.trees))
    for v in range(8):
        var bins: List[Int] = [v]
        var pa = model_a.predict_bins(bins)
        var pb = model_b.predict_bins(bins)
        assert_true(abs(pa - pb) < 1e-12)


def test_zero_weight_rows_are_ignored() raises:
    # Rows 3 and 4 carry flipped labels but weight zero, and each shares
    # a bin with a clean row (features 2.0 and 5.0 repeat). The model
    # must fit the clean step function and contradict the flipped
    # labels. Zero-weight rows must share a bin with a weighted row for
    # this to be determined: an empty bin carries no gradient, so the
    # split search may place it on either side of a threshold tie.
    var features: List[Float64] = [0.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 7.0]
    var labels: List[Float64] = [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    var weights: List[Float64] = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var params = BoosterParams(60, 0.3, small_tree_params())
    var model = train(data, labels, BINARY_LOGISTIC, params, weights)
    for r in range(8):
        var p = model.predict_row(data, r)
        if r <= 3:
            assert_true(p < 0.2)
        else:
            assert_true(p > 0.8)


def test_sample_weight_validation() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var target: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)
    var params = BoosterParams(5, 0.3, small_tree_params())

    var wrong_len: List[Float64] = [1.0, 1.0]
    var raised = False
    try:
        _ = train(data, target, SQUARED_ERROR, params, wrong_len)
    except:
        raised = True
    assert_true(raised)

    var negative: List[Float64] = [1.0, -1.0, 1.0, 1.0]
    raised = False
    try:
        _ = train(data, target, SQUARED_ERROR, params, negative)
    except:
        raised = True
    assert_true(raised)

    var all_zero: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    raised = False
    try:
        _ = train(data, target, SQUARED_ERROR, params, all_zero)
    except:
        raised = True
    assert_true(raised)


def test_multiclass_zero_weight_rows_are_ignored() raises:
    # One row per cluster has a wrong label but weight zero, and each
    # shares a feature value (so a bin) with a clean row of its cluster;
    # predictions for every row must still follow the true cluster.
    var features: List[Float64] = [
        0.0, 1.0, 1.0, 10.0, 11.0, 11.0, 20.0, 21.0, 21.0,
    ]
    var labels: List[Int] = [0, 0, 2, 1, 1, 0, 2, 2, 1]
    var weights: List[Float64] = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    var truth: List[Int] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    var data = bin_equal_width(features, n_rows=9, n_features=1, n_bins=16)
    var params = BoosterParams(150, 0.2, small_tree_params())
    var model = train_multiclass(data, labels, 3, params, weights)
    for r in range(9):
        var bins: List[Int] = [data.bin_at(r, 0)]
        var proba = model.predict_proba_bins(bins)
        var argmax = 0
        for k in range(3):
            if proba[k] > proba[argmax]:
                argmax = k
        assert_equal(argmax, truth[r])


def test_split_importance_single_tree() raises:
    # A step-function gradient on feature 0 admits exactly one split, so
    # with two declared features the counts must be [1, 0].
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var tree = grow_tree(data, grad, hess, small_tree_params())
    var trees = List[Tree]()
    trees.append(tree^)
    var counts = split_importance(trees, 2)
    assert_equal(counts[0], 1)
    assert_equal(counts[1], 0)


def test_split_importance_ensemble() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var params = BoosterParams(10, 0.3, small_tree_params())
    var model = train(data, target, SQUARED_ERROR, params)
    var counts = split_importance(model.trees, 2)
    assert_true(counts[0] > 0)
    assert_equal(counts[1], 0)


def test_gain_importance_single_tree() raises:
    # Same single-split tree as the split-count test. The recorded gain
    # must equal the hand-computed value: GL = -4, GR = 4, H = 4 per
    # side, lambda = 1, so gain = 16/5 + 16/5 - 0 = 6.4.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var tree = grow_tree(data, grad, hess, small_tree_params())
    var trees = List[Tree]()
    trees.append(tree^)
    var gains = gain_importance(trees, 2)
    assert_true(abs(gains[0] - 6.4) < 1e-12)
    assert_true(gains[1] == 0.0)


def test_gain_importance_ensemble() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var params = BoosterParams(10, 0.3, small_tree_params())
    var model = train(data, target, SQUARED_ERROR, params)
    var gains = gain_importance(model.trees, 2)
    assert_true(gains[0] > 0.0)
    assert_true(gains[1] == 0.0)


def test_balanced_class_weights_equalize_class_totals() raises:
    # scikit-learn's rule: total / (n_classes * count_k). With 6 rows in
    # class 0 and 2 in class 1, the weights are 8/(2*6) and 8/(2*2).
    var labels: List[Int] = [0, 0, 0, 0, 0, 0, 1, 1]
    var weights = balanced_class_weights(labels, 2)
    assert_true(abs(weights[0] - 8.0 / 12.0) < 1e-12)
    assert_true(abs(weights[1] - 2.0) < 1e-12)

    # Every class ends up with the same total weight, and the mean row
    # weight stays 1: that is what distinguishes `balanced` from
    # `scale_pos_weight`.
    var rows = balanced_sample_weight(labels, 2)
    var totals: List[Float64] = [0.0, 0.0]
    var total = 0.0
    for r in range(len(labels)):
        totals[labels[r]] += rows[r]
        total += rows[r]
    assert_true(abs(totals[0] - totals[1]) < 1e-12)
    assert_true(abs(total - 8.0) < 1e-12)


def test_balanced_class_weights_use_weighted_counts() raises:
    # With sample weights the counts are weighted counts, so balancing runs
    # on the sample the model actually sees.
    var labels: List[Int] = [0, 0, 1]
    var sample_weight: List[Float64] = [1.0, 1.0, 2.0]
    var weights = balanced_class_weights(labels, 2, sample_weight)
    # Class totals are 2 and 2 already, so both weights are 4/(2*2) = 1.
    assert_true(abs(weights[0] - 1.0) < 1e-12)
    assert_true(abs(weights[1] - 1.0) < 1e-12)


def test_class_weight_rows_multiply_the_sample_weight() raises:
    var labels: List[Int] = [0, 1, 1]
    var class_weights: List[Float64] = [1.0, 3.0]
    var sample_weight: List[Float64] = [2.0, 2.0, 0.5]
    var rows = class_weight_rows(labels, 2, class_weights, sample_weight)
    assert_true(abs(rows[0] - 2.0) < 1e-12)
    assert_true(abs(rows[1] - 6.0) < 1e-12)
    assert_true(abs(rows[2] - 1.5) < 1e-12)


def test_class_weighting_is_ordinary_row_weighting() raises:
    # The whole mechanism: a class-weighted fit is the same model as the
    # fit with those row weights passed by hand, tree for tree.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    var labels: List[Int] = [0, 0, 0, 0, 0, 0, 1, 1]
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var params = BoosterParams(10, 0.3, small_tree_params())

    var rows = balanced_sample_weight(labels, 2)
    var by_class = train(data, target, BINARY_LOGISTIC, params, rows)
    var by_hand_weights: List[Float64] = [
        8.0 / 12.0,
        8.0 / 12.0,
        8.0 / 12.0,
        8.0 / 12.0,
        8.0 / 12.0,
        8.0 / 12.0,
        2.0,
        2.0,
    ]
    var by_hand = train(
        data, target, BINARY_LOGISTIC, params, by_hand_weights
    )
    assert_equal(len(by_class.trees), len(by_hand.trees))
    assert_true(abs(by_class.base_score - by_hand.base_score) < 1e-15)
    for r in range(8):
        assert_true(
            abs(
                by_class.predict_row(data, r) - by_hand.predict_row(data, r)
            )
            < 1e-15
        )


def test_scale_pos_weight_lifts_only_the_positives() raises:
    var labels: List[Float64] = [0.0, 0.0, 1.0]
    var rows = scale_pos_weight_rows(labels, 4.0)
    assert_true(abs(rows[0] - 1.0) < 1e-12)
    assert_true(abs(rows[1] - 1.0) < 1e-12)
    assert_true(abs(rows[2] - 4.0) < 1e-12)


def test_unbalance_scale_is_the_negative_to_positive_ratio() raises:
    var labels: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert_true(abs(unbalance_scale(labels) - 3.0) < 1e-12)
    var rows = unbalanced_sample_weight(labels)
    assert_true(abs(rows[0] - 1.0) < 1e-12)
    assert_true(abs(rows[6] - 3.0) < 1e-12)


def test_class_weight_validation() raises:
    var labels: List[Int] = [0, 1, 1]

    # A weight per class, nonnegative, not all zero.
    var wrong_length: List[Float64] = [1.0]
    var raised = False
    try:
        _ = class_weight_rows(labels, 2, wrong_length)
    except:
        raised = True
    assert_true(raised)

    var negative: List[Float64] = [1.0, -1.0]
    raised = False
    try:
        _ = class_weight_rows(labels, 2, negative)
    except:
        raised = True
    assert_true(raised)

    var all_zero: List[Float64] = [0.0, 0.0]
    raised = False
    try:
        _ = class_weight_rows(labels, 2, all_zero)
    except:
        raised = True
    assert_true(raised)

    # A class with no rows cannot be balanced against.
    var one_class: List[Int] = [0, 0, 0]
    raised = False
    try:
        _ = balanced_class_weights(one_class, 2)
    except:
        raised = True
    assert_true(raised)

    # is_unbalance and scale_pos_weight are two ways to set one number.
    raised = False
    try:
        check_class_balance_params(True, 4.0)
    except:
        raised = True
    assert_true(raised)
    # Either alone is fine.
    check_class_balance_params(True, 1.0)
    check_class_balance_params(False, 4.0)

    # A soft label has no class to weight.
    var soft: List[Float64] = [0.0, 0.5, 1.0]
    raised = False
    try:
        _ = binary_labels_to_codes(soft)
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
