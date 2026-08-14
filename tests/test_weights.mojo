from std.testing import assert_equal, assert_true, TestSuite

from mojoboost import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BoosterParams,
    Tree,
    TreeParams,
    bin_equal_width,
    gain_importance,
    grow_tree,
    split_importance,
    train,
    train_multiclass,
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
