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
from mojoboost import (
    IterationRange,
    MulticlassModel,
    fit_multiclass,
    load_model,
    save_model,
)

from std.math import exp


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


# -- iteration-ranged and leaf-index prediction ---------------------------

comptime _LEAF_TMP_PATH = "./.test_leaf_indices.tmp"


def _sliceable_model() raises -> Model:
    """A step-function regression fitted with enough iterations that there
    is something to slice."""
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var params = BoosterParams(20, 0.1, small_tree_params())
    return fit(features, 8, 1, target, SQUARED_ERROR, params, max_bins=8)


def _sliceable_multiclass() raises -> MulticlassModel:
    var features: List[Float64] = [
        0.0, 1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0,
    ]
    var labels: List[Int] = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    var params = BoosterParams(12, 0.2, small_tree_params())
    return fit_multiclass(features, 9, 1, labels, 3, params, max_bins=16)


def test_iteration_range_clamps_like_lightgbm() raises:
    # slice() takes an explicit half-open pair.
    var r = IterationRange.slice(10, -3, 4)
    assert_equal(r.start, 0)
    assert_equal(r.stop, 4)
    r = IterationRange.slice(10, 8, 99)
    assert_equal(r.start, 8)
    assert_equal(r.stop, 10)
    # A start past the last iteration is an empty range at the end, not an
    # error, so slicing a shorter ensemble than expected still predicts.
    r = IterationRange.slice(10, 12, 14)
    assert_equal(r.start, 10)
    assert_equal(r.stop, 10)
    assert_true(r.is_empty())
    # A stop at or below the start is empty rather than negative-length.
    r = IterationRange.slice(10, 6, 2)
    assert_equal(r.start, 6)
    assert_equal(r.stop, 6)
    assert_true(r.is_empty())

    # clamp() takes LightGBM's (start_iteration, num_iteration) pair, where
    # a nonpositive count means every remaining iteration.
    assert_equal(IterationRange.clamp(10, 0, 3).stop, 3)
    r = IterationRange.clamp(10, 2, 0)
    assert_equal(r.start, 2)
    assert_equal(r.stop, 10)
    assert_equal(IterationRange.clamp(10, 2, -1).stop, 10)
    r = IterationRange.clamp(10, -5, 4)
    assert_equal(r.start, 0)
    assert_equal(r.stop, 4)
    assert_equal(IterationRange.clamp(10, 4, 100).stop, 10)

    # The base score sits in iteration 0, so only a range starting there
    # carries it.
    assert_true(IterationRange.slice(10, 0, 0).includes_base())
    assert_false(IterationRange.slice(10, 1, 3).includes_base())
    assert_equal(IterationRange.slice(10, 2, 7).n_iterations(), 5)


def test_iteration_slice_matches_a_manual_tree_sum() raises:
    # The defining property: slicing [0, k) is the base score plus the first
    # k shrunken tree outputs, summed by hand.
    var model = _sliceable_model()
    var n = model.n_iterations()
    assert_true(n >= 4)
    var row: List[Float64] = [6.3]
    var bins = model.mapper.bin_row(row)
    for k in range(n + 1):
        var manual = model.booster.base_score
        for i in range(k):
            manual += (
                model.booster.learning_rate
                * model.booster.trees[i].predict_bins(bins)
            )
        var sliced = model.predict_raw_range(row, IterationRange.slice(n, 0, k))
        assert_true(abs(sliced - manual) < 1e-12)


def test_iteration_slices_partition_the_raw_score() raises:
    # [0, k) and [k, n) sum to the whole raw score for every k >= 1: the
    # base score belongs to the head, which is the only range starting at 0.
    var model = _sliceable_model()
    var n = model.n_iterations()
    var row: List[Float64] = [2.5]
    var whole = model.predict_raw(row)
    for k in range(1, n + 1):
        var head = model.predict_raw_range(row, IterationRange.slice(n, 0, k))
        var tail = model.predict_raw_range(row, IterationRange.slice(n, k, n))
        assert_true(abs(head + tail - whole) < 1e-9)


def test_full_range_reproduces_plain_prediction() raises:
    var model = _sliceable_model()
    var n = model.n_iterations()
    var full = IterationRange.slice(n, 0, n)
    for r in range(8):
        var row: List[Float64] = [Float64(r)]
        assert_true(
            abs(model.predict_raw_range(row, full) - model.predict_raw(row))
            < 1e-12
        )
        assert_true(
            abs(model.predict_range(row, full) - model.predict(row)) < 1e-12
        )


def test_empty_iteration_ranges_predict_the_base_score() raises:
    var model = _sliceable_model()
    var n = model.n_iterations()
    var row: List[Float64] = [6.3]
    # [0, 0) selects no trees but does start at iteration 0, so it is the
    # base-score-only model: what a zero-iteration ensemble predicts.
    var head = model.predict_raw_range(row, IterationRange.slice(n, 0, 0))
    assert_true(abs(head - model.booster.base_score) < 1e-12)
    # An empty range that starts later carries no base score at all.
    var tail = model.predict_raw_range(row, IterationRange.slice(n, n, n))
    assert_true(abs(tail) < 1e-12)
    # Either way it names no trees, so it produces no leaf indices.
    assert_equal(len(model.leaf_indices(row, IterationRange.slice(n, 0, 0))), 0)
    assert_equal(len(model.leaf_indices(row, IterationRange.slice(n, n, n))), 0)


def test_range_response_scale_transforms_the_raw_score() raises:
    # A binary model's response scale is the logistic of its raw score at
    # every slice, not only for the whole ensemble.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var params = BoosterParams(15, 0.3, small_tree_params())
    var model = fit(
        features, 8, 1, target, BINARY_LOGISTIC, params, max_bins=8
    )
    var n = model.n_iterations()
    var row: List[Float64] = [5.5]
    for k in range(n + 1):
        var rng = IterationRange.slice(n, 0, k)
        var raw = model.predict_raw_range(row, rng)
        var response = model.predict_range(row, rng)
        assert_true(abs(response - 1.0 / (1.0 + exp(-raw))) < 1e-12)
        # A probability, at every truncation.
        assert_true(response > 0.0 and response < 1.0)


def test_leaf_ordinals_are_dense_and_number_only_leaves() raises:
    var model = _sliceable_model()
    for i in range(model.n_iterations()):
        ref tree = model.booster.trees[i]
        var table = tree.leaf_ordinals()
        assert_equal(len(table), len(tree.feature))
        var seen = 0
        for node in range(len(table)):
            if tree.feature[node] < 0:
                assert_equal(table[node], seen)
                seen += 1
            else:
                assert_equal(table[node], -1)
        # Every leaf got an ordinal, and the ordinals fill [0, n_leaves).
        assert_equal(seen, tree.n_leaves)


def test_leaf_indices_agree_with_the_ordinal_table() raises:
    # The per-row walk and the precomputed table are two implementations of
    # the same numbering; the binding uses the table, so they must agree.
    var model = _sliceable_model()
    var n = model.n_iterations()
    var full = IterationRange.slice(n, 0, n)
    var tables = model.booster.leaf_ordinals_range(full)
    for r in range(8):
        var row: List[Float64] = [Float64(r)]
        var bins = model.mapper.bin_row(row)
        var idx = model.leaf_indices(row, full)
        assert_equal(len(idx), n)
        for i in range(n):
            ref tree = model.booster.trees[i]
            var node = tree.leaf_index_bins(bins)
            assert_true(tree.feature[node] < 0)
            assert_equal(idx[i], tables[i][node])
            assert_true(idx[i] >= 0)
            assert_true(idx[i] < tree.n_leaves)


def test_leaf_index_identifies_the_predicting_leaf() raises:
    # An ordinal names the leaf whose value the tree contributes, so two
    # rows share an ordinal only when they share that tree's output.
    var model = _sliceable_model()
    var n = model.n_iterations()
    var full = IterationRange.slice(n, 0, n)
    var indices = List[List[Int]]()
    var outputs = List[List[Float64]]()
    for r in range(8):
        var row: List[Float64] = [Float64(r)]
        var bins = model.mapper.bin_row(row)
        indices.append(model.leaf_indices(row, full))
        var per_tree = List[Float64](capacity=n)
        for i in range(n):
            per_tree.append(model.booster.trees[i].predict_bins(bins))
        outputs.append(per_tree^)
    var distinct_somewhere = False
    for a in range(8):
        for b in range(8):
            for i in range(n):
                if indices[a][i] == indices[b][i]:
                    assert_true(abs(outputs[a][i] - outputs[b][i]) < 1e-15)
                else:
                    distinct_somewhere = True
    # The model is not a single-leaf ensemble, so the check has teeth.
    assert_true(distinct_somewhere)


def test_leaf_indices_follow_the_iteration_range() raises:
    var model = _sliceable_model()
    var n = model.n_iterations()
    var row: List[Float64] = [3.7]
    var full = model.leaf_indices(row, IterationRange.slice(n, 0, n))
    var tail = model.leaf_indices(row, IterationRange.slice(n, 2, n))
    assert_equal(len(tail), n - 2)
    for i in range(len(tail)):
        assert_equal(tail[i], full[i + 2])


def test_multiclass_range_matches_a_manual_per_class_sum() raises:
    var model = _sliceable_multiclass()
    var n = model.n_iterations()
    assert_true(n >= 4)
    var row: List[Float64] = [11.0]
    var bins = model.mapper.bin_row(row)
    for k in range(n + 1):
        var rng = IterationRange.slice(n, 0, k)
        var raw = model.predict_raw_range(row, rng)
        assert_equal(len(raw), 3)
        for c in range(3):
            var manual = model.booster.base_scores[c]
            for i in range(k):
                manual += (
                    model.booster.learning_rate
                    * model.booster.trees[i * 3 + c].predict_bins(bins)
                )
            assert_true(abs(raw[c] - manual) < 1e-9)
        # The softmax is taken over the sliced scores, so the truncated
        # ensemble still returns a distribution.
        var proba = model.predict_proba_range(row, rng)
        assert_equal(len(proba), 3)
        var total = 0.0
        for c in range(3):
            assert_true(proba[c] >= 0.0)
            total += proba[c]
        assert_true(abs(total - 1.0) < 1e-9)


def test_multiclass_leaf_indices_are_round_major() raises:
    # One tree per class per iteration, laid out so entry i * n_classes + k
    # is class k's tree in the range's iteration i.
    var model = _sliceable_multiclass()
    var n = model.n_iterations()
    var full = IterationRange.slice(n, 0, n)
    for r in range(3):
        var row: List[Float64] = [Float64(r) * 10.0 + 1.0]
        var bins = model.mapper.bin_row(row)
        var idx = model.leaf_indices(row, full)
        assert_equal(len(idx), n * 3)
        for i in range(n):
            for k in range(3):
                ref tree = model.booster.trees[i * 3 + k]
                assert_equal(idx[i * 3 + k], tree.leaf_ordinal_bins(bins))
    # A sliced range drops whole iterations, three columns at a time.
    var probe: List[Float64] = [1.0]
    var sliced = model.leaf_indices(probe, IterationRange.slice(n, 1, n))
    assert_equal(len(sliced), (n - 1) * 3)
    assert_equal(
        len(model.leaf_indices(probe, IterationRange.slice(n, n, n))), 0
    )


def test_leaf_indices_and_slices_survive_serialization() raises:
    # Leaf ordinals are derived from node order, and the model format writes
    # nodes in array order, so a reloaded model must number leaves the same
    # way and slice to the same scores.
    var model = _sliceable_model()
    save_model(model, _LEAF_TMP_PATH)
    var loaded = load_model(_LEAF_TMP_PATH)
    var n = model.n_iterations()
    assert_equal(loaded.n_iterations(), n)
    var full = IterationRange.slice(n, 0, n)
    for r in range(8):
        var row: List[Float64] = [Float64(r) + 0.5]
        var before = model.leaf_indices(row, full)
        var after = loaded.leaf_indices(row, full)
        assert_equal(len(before), len(after))
        for i in range(len(before)):
            assert_equal(before[i], after[i])
        for k in range(n + 1):
            var rng = IterationRange.slice(n, 0, k)
            assert_true(
                abs(
                    model.predict_raw_range(row, rng)
                    - loaded.predict_raw_range(row, rng)
                )
                < 1e-12
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
