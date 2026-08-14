from std.testing import assert_equal, assert_true, assert_false, TestSuite

from mojoboost import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BinnedMatrix,
    Booster,
    BoosterParams,
    Histogram,
    SplitInfo,
    Tree,
    TreeParams,
    bin_equal_width,
    build_histogram,
    build_histogram_subset,
    find_best_split,
    fit,
    fit_bins,
    grow_tree,
    subtract_histogram,
    train,
)
from mojoboost import BinMapper, Model, MulticlassBooster
from mojoboost import train_multiclass, train_with_valid


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


def make_additive() raises -> BinnedMatrix:
    # Two binary features, 8 rows, column-major. Gradients are additive in
    # both features with feature 0 dominant, so leaf-wise growth must split
    # feature 0 at the root and then feature 1 in each child.
    var features: List[Float64] = [
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,  # feature 0
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,  # feature 1
    ]
    return bin_equal_width(features, n_rows=8, n_features=2, n_bins=2)


def additive_grad() -> List[Float64]:
    # g(row) = (f0 ? +2 : -2) + (f1 ? +1 : -1)
    return [-3.0, -1.0, -3.0, -1.0, 1.0, 3.0, 1.0, 3.0]


def ones(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(1.0)
    return out^


def test_histogram_subtraction() raises:
    var data = make_additive()
    var grad = additive_grad()
    var hess = ones(8)
    var parent = build_histogram(data, grad, hess)
    var left_rows: List[Int] = [0, 1, 2, 3]
    var right_rows: List[Int] = [4, 5, 6, 7]
    var left = build_histogram_subset(data, grad, hess, left_rows)
    var derived_right = subtract_histogram(parent, left)
    var direct_right = build_histogram_subset(data, grad, hess, right_rows)
    for i in range(2 * 2):
        assert_true(abs(derived_right.grad[i] - direct_right.grad[i]) < 1e-12)
        assert_true(abs(derived_right.hess[i] - direct_right.hess[i]) < 1e-12)
        assert_equal(derived_right.count[i], direct_right.count[i])


def test_tree_single_split() raises:
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hess = ones(8)
    var params = TreeParams(2, 1, 1.0, 1e-3)
    var tree = grow_tree(data, grad, hess, params)
    assert_equal(tree.n_leaves, 2)
    assert_equal(tree.feature[0], 0)
    assert_equal(tree.threshold_bin[0], 3)
    # Leaf values are the Newton step -G/(H+lambda): -(-4)/5 and -4/5.
    for r in range(8):
        var expected = 0.8 if r < 4 else -0.8
        assert_true(abs(tree.predict_row(data, r) - expected) < 1e-12)


def test_tree_leafwise_depth2() raises:
    var data = make_additive()
    var grad = additive_grad()
    var hess = ones(8)
    var params = TreeParams(4, 1, 1.0, 1e-3)
    var tree = grow_tree(data, grad, hess, params)
    assert_equal(tree.n_leaves, 4)
    # Root must split the dominant feature 0.
    assert_equal(tree.feature[0], 0)
    # Each (f0, f1) group of 2 rows has value -G/(H+1) = -2g/3.
    var expected: List[Float64] = [2.0, 2.0 / 3.0, 2.0, 2.0 / 3.0,
                                   -2.0 / 3.0, -2.0, -2.0 / 3.0, -2.0]
    for r in range(8):
        assert_true(abs(tree.predict_row(data, r) - expected[r]) < 1e-12)


def test_tree_min_data_in_leaf() raises:
    # min_data_in_leaf=3 allows the root 4/4 split but blocks the 2/2
    # child splits, so growth stops at 2 leaves despite num_leaves=31.
    var data = make_additive()
    var grad = additive_grad()
    var hess = ones(8)
    var params = TreeParams(31, 3, 1.0, 1e-3)
    var tree = grow_tree(data, grad, hess, params)
    assert_equal(tree.n_leaves, 2)


def small_tree_params() -> TreeParams:
    return TreeParams(4, 1, 1.0, 1e-3)


def test_boosting_regression_step_function() raises:
    var data = make_toy()
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var params = BoosterParams(100, 0.1, small_tree_params())
    var model = train(data, target, SQUARED_ERROR, params)
    for r in range(8):
        assert_true(abs(model.predict_row(data, r) - target[r]) < 0.05)


def test_boosting_binary_logistic() raises:
    # Label is feature 0; a boosted logistic model should become confident.
    var data = make_additive()
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var params = BoosterParams(200, 0.2, small_tree_params())
    var model = train(data, target, BINARY_LOGISTIC, params)
    for r in range(8):
        var p = model.predict_row(data, r)
        assert_true(p >= 0.0 and p <= 1.0)
        if target[r] > 0.5:
            assert_true(p > 0.8)
        else:
            assert_true(p < 0.2)


def test_boosting_converged_early_stop() raises:
    # A constant target is fit exactly by the base score; no trees needed.
    var data = make_toy()
    var target: List[Float64] = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    var params = BoosterParams(50, 0.1, small_tree_params())
    var model = train(data, target, SQUARED_ERROR, params)
    assert_equal(len(model.trees), 0)
    assert_true(abs(model.predict_row(data, 0) - 2.5) < 1e-12)


def test_boosting_validates_objective() raises:
    var data = make_toy()
    var target = ones(8)
    var raised = False
    try:
        _ = train(data, target, 99, BoosterParams(1, 0.1, small_tree_params()))
    except:
        raised = True
    assert_true(raised)


def test_quantile_binning_identity() raises:
    # 8 distinct values into 8 quantile bins: edges land between each pair,
    # so binning is the identity, same as the equal-width toy.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var mapper = fit_bins(features, n_rows=8, n_features=1, max_bins=8)
    var data = mapper.transform(features, 8)
    for r in range(8):
        assert_equal(data.bin_at(r, 0), r)
    # Unseen values: below range, between training values, above range.
    assert_equal(mapper.bin_value(0, -100.0), 0)
    assert_equal(mapper.bin_value(0, 2.4), 2)
    assert_equal(mapper.bin_value(0, 100.0), 7)


def test_quantile_binning_duplicates() raises:
    # A binary feature must collapse to a single edge (2 used bins) even
    # with max_bins much larger than n_rows.
    var features: List[Float64] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    var mapper = fit_bins(features, n_rows=6, n_features=1, max_bins=255)
    assert_equal(mapper.edge_offsets[1] - mapper.edge_offsets[0], 1)
    for r in range(3):
        assert_equal(mapper.bin_value(0, features[r]), 0)
    for r in range(3, 6):
        assert_equal(mapper.bin_value(0, features[r]), 1)


def test_quantile_binning_skewed() raises:
    # Equal-frequency binning must separate a dense cluster that
    # equal-width binning would collapse into one bin.
    var features: List[Float64] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1000000.0,
    ]
    var mapper = fit_bins(features, n_rows=8, n_features=1, max_bins=4)
    var bins_seen = List[Int]()
    for r in range(8):
        var b = mapper.bin_value(0, features[r])
        var new = True
        for i in range(len(bins_seen)):
            if bins_seen[i] == b:
                new = False
        if new:
            bins_seen.append(b)
    assert_equal(len(bins_seen), 4)


def test_model_predicts_raw_data() raises:
    # End to end on raw features: fit a regression on a step function and
    # predict unseen raw values on both sides of the step.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var params = BoosterParams(100, 0.1, small_tree_params())
    var model = fit(features, 8, 1, target, SQUARED_ERROR, params, max_bins=8)
    var low: List[Float64] = [1.4]
    var high: List[Float64] = [6.3]
    assert_true(abs(model.predict(low) - 0.0) < 0.05)
    assert_true(abs(model.predict(high) - 1.0) < 0.05)


def test_multiclass_three_classes() raises:
    # 9 rows, one feature with three separated clusters; the model must
    # put high probability on the right class for every training row.
    var features: List[Float64] = [
        0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0,
    ]
    var labels: List[Int] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    var mapper = fit_bins(features, n_rows=9, n_features=1, max_bins=16)
    var data = mapper.transform(features, 9)
    var params = BoosterParams(150, 0.2, small_tree_params())
    var model = train_multiclass(data, labels, 3, params)
    for r in range(9):
        var row: List[Float64] = [features[r]]
        var proba = model.predict_proba_bins(mapper.bin_row(row))
        var total = 0.0
        var argmax = 0
        for k in range(3):
            total += proba[k]
            if proba[k] > proba[argmax]:
                argmax = k
        assert_true(abs(total - 1.0) < 1e-9)
        assert_equal(argmax, labels[r])
        assert_true(proba[labels[r]] > 0.7)


def test_early_stopping_truncates() raises:
    # Once the step function is fit, per-round validation improvement
    # shrinks below min_delta, so training must stop well short of
    # n_estimators while still keeping enough rounds to fit the step.
    var data = make_toy()
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var params = BoosterParams(500, 0.3, small_tree_params())
    var model = train_with_valid(
        data, target, data, target, SQUARED_ERROR, params,
        early_stopping_rounds=5, min_delta=1e-6,
    )
    assert_true(len(model.trees) < 100)
    assert_true(len(model.trees) > 0)
    for r in range(8):
        assert_true(abs(model.predict_row(data, r) - target[r]) < 0.05)


def test_early_stopping_prevents_overfit_to_noise() raises:
    # Validation labels flip the training labels, so validation loss only
    # degrades as training fits: the returned ensemble must be tiny.
    var data = make_toy()
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var flipped: List[Float64] = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    var params = BoosterParams(500, 0.3, small_tree_params())
    var model = train_with_valid(
        data, target, data, flipped, SQUARED_ERROR, params,
        early_stopping_rounds=3,
    )
    assert_equal(len(model.trees), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
