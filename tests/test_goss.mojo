"""Gradient-based One-Side Sampling.

Covers the selection rule itself (which rows are kept, how many, with what
multiplier), the statistical property that justifies GOSS (the compensated
sample reproduces the full-data gradient histogram in expectation), the
integration points (sample weights, leaf row counts, objectives, leaf
renewal, multiclass), determinism, validation, and the equivalence that
matters most: with GOSS disabled, training is bit-for-bit what it was
before GOSS existed.
"""

from std.os import remove
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees import (
    BINARY_LOGISTIC,
    CPU_DEVICE,
    QUANTILE,
    SQUARED_ERROR,
    BaggingParams,
    BoosterParams,
    GossParams,
    TreeParams,
    apply_goss_scaling,
    bin_equal_width,
    build_histogram,
    build_histogram_subset,
    fit,
    goss_importance,
    goss_select,
    load_model,
    save_model,
    train,
    train_multiclass,
)

comptime _TMP_PATH = "./.test_goss_roundtrip.tmp"


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _features(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major deterministic features in [0, 1)."""
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))
    return features^


def _regression_target(
    features: List[Float64], n_rows: Int
) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        y.append(3.0 * x0 - 2.0 * x1 + 4.0 * (x2 - 0.5) * (x2 - 0.5))
    return y^


def _rmse_on_train(
    predictions: List[Float64], target: List[Float64]
) -> Float64:
    var total = 0.0
    for r in range(len(target)):
        var d = predictions[r] - target[r]
        total += d * d
    return (total / Float64(len(target))) ** 0.5


def test_selection_keeps_top_and_samples_other() raises:
    # 100 rows whose importances are 5 distinct values, 20 rows each, so the
    # 20th largest is the smallest of the top block and `>= threshold` keeps
    # exactly the top 20. top_rate 0.2 and other_rate 0.1 then give
    # top_k = 20, other_k = 10 and multiplier (100 - 20) / 10 = 8.
    var grad = List[Float64](capacity=100)
    var hess = List[Float64](capacity=100)
    for r in range(100):
        grad.append(Float64(r % 5) + 1.0)
        hess.append(1.0)
    var importance = goss_importance(grad, hess)
    var selection = goss_select(importance, GossParams.enable(0.2, 0.1), 0)

    assert_equal(selection.n_top, 20)
    assert_equal(selection.n_other, 10)
    assert_equal(len(selection.rows), 30)
    assert_true(abs(selection.multiplier - 8.0) < 1e-12)

    # Every kept row is a top-importance row, every sampled row is not, and
    # the scales say which is which.
    var n_top_seen = 0
    for i in range(len(selection.rows)):
        var r = selection.rows[i]
        if selection.scale[i] == 1.0:
            assert_true(importance[r] >= 5.0)
            n_top_seen += 1
        else:
            assert_true(importance[r] < 5.0)
            assert_true(abs(selection.scale[i] - 8.0) < 1e-12)
    assert_equal(n_top_seen, 20)

    # Rows come back ascending, which is what the histogram builders want.
    for i in range(1, len(selection.rows)):
        assert_true(selection.rows[i] > selection.rows[i - 1])

    # Every row of the top block is present, not just 20 rows of some kind.
    var present = List[Bool]()
    for _ in range(100):
        present.append(False)
    for i in range(len(selection.rows)):
        present[selection.rows[i]] = True
    for r in range(100):
        if importance[r] >= 5.0:
            assert_true(present[r])


def test_selection_is_deterministic() raises:
    var grad = List[Float64](capacity=500)
    var hess = List[Float64](capacity=500)
    for r in range(500):
        grad.append(2.0 * _uniform(UInt64(r)) - 1.0)
        hess.append(_uniform(UInt64(1000 + r)) + 0.5)
    var importance = goss_importance(grad, hess)
    var params = GossParams.enable(0.2, 0.1, 7)

    var a = goss_select(importance, params, 3)
    var b = goss_select(importance, params, 3)
    assert_equal(len(a.rows), len(b.rows))
    for i in range(len(a.rows)):
        assert_equal(a.rows[i], b.rows[i])

    # A different round reseeds the stream, so the sampled half moves while
    # the kept half does not.
    var c = goss_select(importance, params, 4)
    assert_equal(a.n_top, c.n_top)
    var differences = 0
    for i in range(len(a.rows)):
        if a.rows[i] != c.rows[i]:
            differences += 1
    assert_true(differences > 0)

    # So does a different seed.
    var d = goss_select(importance, GossParams.enable(0.2, 0.1, 8), 3)
    differences = 0
    for i in range(len(a.rows)):
        if a.rows[i] != d.rows[i]:
            differences += 1
    assert_true(differences > 0)


def test_compensated_sample_reproduces_full_histogram() raises:
    # The statistical claim GOSS rests on: scaling the sampled small-gradient
    # rows by (n - top_k) / other_k makes the sampled histogram an unbiased
    # estimate of the full-data histogram. Averaged over rounds (each round
    # reseeds), every bin of the sampled histogram must converge to the full
    # one. Tolerance is relative to the total gradient magnitude, so it does
    # not depend on how the mass happens to fall across bins.
    var n_rows = 400
    var n_bins = 8
    var features = _features(n_rows, 1)
    var data = bin_equal_width(features, n_rows, 1, n_bins)

    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    var total_magnitude = 0.0
    for r in range(n_rows):
        var g = 2.0 * _uniform(UInt64(7000 + r)) - 1.0
        grad.append(g)
        hess.append(_uniform(UInt64(9000 + r)) + 0.25)
        total_magnitude += abs(g)
    var full = build_histogram(data, grad, hess)
    var importance = goss_importance(grad, hess)

    var n_rounds = 200
    var accumulated = List[Float64](capacity=n_bins)
    for _ in range(n_bins):
        accumulated.append(0.0)
    var counts_ok = True
    for round in range(n_rounds):
        var selection = goss_select(
            importance, GossParams.enable(0.2, 0.1, 11), round
        )
        var sampled_grad = grad.copy()
        var sampled_hess = hess.copy()
        apply_goss_scaling(selection, sampled_grad, sampled_hess)
        var sampled = build_histogram_subset(
            data, sampled_grad, sampled_hess, selection.rows
        )
        var total_count = 0
        for b in range(n_bins):
            accumulated[b] += sampled.grad[b]
            total_count += sampled.count[b]
        # Counts are sample counts, never full-data counts: that is what
        # min_data_in_leaf sees under GOSS.
        if total_count != len(selection.rows):
            counts_ok = False
    assert_true(counts_ok)

    for b in range(n_bins):
        var mean = accumulated[b] / Float64(n_rounds)
        assert_true(abs(mean - full.grad[b]) <= 0.02 * total_magnitude)


def test_zero_importance_rows_are_never_kept() raises:
    # Sample weights ride into GOSS through the gradients, so a zero-weight
    # row has zero importance and can never enter the kept top set. It may
    # still be drawn into the sampled set, where it contributes nothing.
    var n_rows = 200
    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        if r % 2 == 0:
            grad.append(0.0)
            hess.append(0.0)
        else:
            grad.append(1.0 + _uniform(UInt64(r)))
            hess.append(1.0)
    var importance = goss_importance(grad, hess)
    var selection = goss_select(importance, GossParams.enable(0.2, 0.1), 0)
    for i in range(len(selection.rows)):
        if selection.scale[i] == 1.0:
            assert_true(importance[selection.rows[i]] > 0.0)


def test_rate_validation() raises:
    var cases = List[GossParams]()
    cases.append(GossParams.enable(1.5, 0.1))
    cases.append(GossParams.enable(-0.1, 0.1))
    cases.append(GossParams.enable(0.2, 1.5))
    cases.append(GossParams.enable(0.2, -0.1))
    cases.append(GossParams.enable(0.7, 0.4))
    cases.append(GossParams.enable(0.0, 0.0))
    cases.append(GossParams.enable(0.2, 0.1, -1))
    cases.append(GossParams.enable(0.2, 0.1, 3, -2))
    for i in range(len(cases)):
        var raised = False
        try:
            cases[i].validate()
        except:
            raised = True
        assert_true(raised)

    # Rates that sum to exactly 1 are legal (every row is used).
    GossParams.enable(0.7, 0.3).validate()
    # Disabled parameters are never validated: the defaults are inert.
    GossParams(False, 5.0, 5.0, -1, -7).validate()


def test_training_rejects_bad_rates_and_bagging() raises:
    var features = _features(80, 3)
    var target = _regression_target(features, 80)
    var data = bin_equal_width(features, 80, 3, 16)
    var params = BoosterParams(5, 0.3, TreeParams(8, 5, 1.0, 1e-3))

    var raised = False
    try:
        _ = train(
            data,
            target,
            SQUARED_ERROR,
            params,
            [],
            0.9,
            BaggingParams.disabled(),
            GossParams.enable(0.9, 0.2),
        )
    except:
        raised = True
    assert_true(raised)

    # GOSS and row bagging both own the row list, so the combination is
    # rejected rather than silently resolved.
    raised = False
    try:
        _ = train(
            data,
            target,
            SQUARED_ERROR,
            params,
            [],
            0.9,
            BaggingParams(0.5, 1, 3),
            GossParams.enable(),
        )
    except:
        raised = True
    assert_true(raised)


def test_disabled_matches_plain_training() raises:
    # The default is ordinary GBDT and must stay bit-identical to training
    # with no GOSS argument at all.
    var n_rows = 600
    var features = _features(n_rows, 4)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, 4, 32)
    var params = BoosterParams(20, 0.2, TreeParams(15, 10, 1.0, 1e-3))

    var plain = train(data, target, SQUARED_ERROR, params)
    var explicit = train(
        data,
        target,
        SQUARED_ERROR,
        params,
        [],
        0.9,
        BaggingParams.disabled(),
        GossParams.disabled(),
    )
    assert_equal(len(plain.trees), len(explicit.trees))
    for r in range(n_rows):
        assert_equal(
            plain.predict_raw_row(data, r), explicit.predict_raw_row(data, r)
        )


def test_warmup_rounds_train_on_every_row() raises:
    # LightGBM trains the first int(1 / learning_rate) rounds on all rows.
    # Those trees must therefore be exactly the full-data trees.
    var n_rows = 400
    var features = _features(n_rows, 3)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, 3, 32)
    var params = BoosterParams(4, 0.25, TreeParams(8, 10, 1.0, 1e-3))

    var full = train(data, target, SQUARED_ERROR, params)
    # learning_rate 0.25 means 4 warmup rounds, which is the whole run.
    var warming = train(
        data,
        target,
        SQUARED_ERROR,
        params,
        [],
        0.9,
        BaggingParams.disabled(),
        GossParams.enable(),
    )
    assert_equal(len(full.trees), len(warming.trees))
    for r in range(n_rows):
        assert_equal(
            full.predict_raw_row(data, r), warming.predict_raw_row(data, r)
        )

    # An explicit warmup of 2 keeps only the first two trees identical.
    var partial = train(
        data,
        target,
        SQUARED_ERROR,
        params,
        [],
        0.9,
        BaggingParams.disabled(),
        GossParams.enable(0.2, 0.1, 3, 2),
    )
    var same_prefix = True
    for r in range(n_rows):
        var a = (
            full.trees[0].predict_row(data, r)
            + full.trees[1].predict_row(data, r)
        )
        var b = (
            partial.trees[0].predict_row(data, r)
            + partial.trees[1].predict_row(data, r)
        )
        if a != b:
            same_prefix = False
    assert_true(same_prefix)

    var diverged = False
    for r in range(n_rows):
        if full.predict_raw_row(data, r) != partial.predict_raw_row(data, r):
            diverged = True
    assert_true(diverged)


def test_training_is_deterministic() raises:
    var n_rows = 500
    var features = _features(n_rows, 4)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, 4, 32)
    var params = BoosterParams(15, 0.3, TreeParams(15, 10, 1.0, 1e-3))
    var goss = GossParams.enable(0.2, 0.1, 5, 0)

    var a = train(
        data, target, SQUARED_ERROR, params, [], 0.9,
        BaggingParams.disabled(), goss,
    )
    var b = train(
        data, target, SQUARED_ERROR, params, [], 0.9,
        BaggingParams.disabled(), goss,
    )
    assert_equal(len(a.trees), len(b.trees))
    for r in range(n_rows):
        assert_equal(a.predict_raw_row(data, r), b.predict_raw_row(data, r))

    # A different seed samples different rows and trains a different model.
    var c = train(
        data, target, SQUARED_ERROR, params, [], 0.9,
        BaggingParams.disabled(), GossParams.enable(0.2, 0.1, 6, 0),
    )
    var differs = False
    for r in range(n_rows):
        if a.predict_raw_row(data, r) != c.predict_raw_row(data, r):
            differs = True
    assert_true(differs)


def test_leaf_row_counts_come_from_the_sample() raises:
    # 300 rows sampled at 0.1 + 0.1 leave 60 rows at the root, so a
    # min_data_in_leaf of 40 (which needs 80 rows to split) forbids every
    # split. Full-data training with the same threshold splits freely. This
    # is only true if min_data_in_leaf counts sampled rows.
    var n_rows = 300
    var features = _features(n_rows, 3)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, 3, 32)
    var params = BoosterParams(3, 0.5, TreeParams(15, 40, 1.0, 1e-3))

    var sampled = train(
        data, target, SQUARED_ERROR, params, [], 0.9,
        BaggingParams.disabled(), GossParams.enable(0.1, 0.1, 3, 0),
    )
    assert_true(len(sampled.trees) > 0)
    for t in range(len(sampled.trees)):
        assert_equal(sampled.trees[t].n_leaves, 1)

    var full = train(data, target, SQUARED_ERROR, params)
    assert_true(full.trees[0].n_leaves > 1)


def test_sampled_training_fits_the_signal() raises:
    # GOSS on a third of the rows must stay in the neighborhood of full-data
    # training, and must be far better than the constant base-score model.
    var n_rows = 1_500
    var features = _features(n_rows, 4)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, 4, 64)
    var params = BoosterParams(60, 0.2, TreeParams(15, 20, 1.0, 1e-3))

    var full = train(data, target, SQUARED_ERROR, params)
    var sampled = train(
        data, target, SQUARED_ERROR, params, [], 0.9,
        BaggingParams.disabled(), GossParams.enable(0.2, 0.1, 3, 0),
    )

    var full_pred = List[Float64](capacity=n_rows)
    var goss_pred = List[Float64](capacity=n_rows)
    var mean_pred = List[Float64](capacity=n_rows)
    var mean_target = 0.0
    for r in range(n_rows):
        mean_target += target[r]
    mean_target /= Float64(n_rows)
    for r in range(n_rows):
        full_pred.append(full.predict_row(data, r))
        goss_pred.append(sampled.predict_row(data, r))
        mean_pred.append(mean_target)

    var full_rmse = _rmse_on_train(full_pred, target)
    var goss_rmse = _rmse_on_train(goss_pred, target)
    var base_rmse = _rmse_on_train(mean_pred, target)
    assert_true(goss_rmse < 0.5 * base_rmse)
    assert_true(goss_rmse < 2.0 * full_rmse)


def test_zero_weight_rows_stay_ignored_under_goss() raises:
    # Half the rows carry a wrecked target and weight zero. GOSS ranks them
    # last (zero gradient, zero importance) and, if drawn, they contribute
    # nothing, so the fit must follow the weighted rows only.
    var n_rows = 600
    var features = _features(n_rows, 3)
    var clean = _regression_target(features, n_rows)
    var target = List[Float64](capacity=n_rows)
    var weights = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        if r % 2 == 0:
            target.append(clean[r])
            weights.append(1.0)
        else:
            target.append(clean[r] + 50.0)
            weights.append(0.0)
    var data = bin_equal_width(features, n_rows, 3, 32)
    var params = BoosterParams(40, 0.2, TreeParams(15, 10, 1.0, 1e-3))

    var model = train(
        data, target, SQUARED_ERROR, params, weights, 0.9,
        BaggingParams.disabled(), GossParams.enable(0.2, 0.1, 3, 0),
    )
    var weighted_error = 0.0
    var n_weighted = 0
    for r in range(n_rows):
        if weights[r] > 0.0:
            var d = model.predict_row(data, r) - target[r]
            weighted_error += d * d
            n_weighted += 1
    var rmse = (weighted_error / Float64(n_weighted)) ** 0.5
    # The wrecked rows sit 50 away; anything close to them would blow this up.
    assert_true(rmse < 1.0)


def test_binary_and_quantile_objectives_train_under_goss() raises:
    # Every objective feeds GOSS the same way, and QUANTILE additionally
    # renews leaf values from the sampled rows, so both paths need a run.
    var n_rows = 800
    var features = _features(n_rows, 3)
    var signal = _regression_target(features, n_rows)
    var labels = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        labels.append(1.0 if signal[r] > 0.5 else 0.0)
    var data = bin_equal_width(features, n_rows, 3, 32)
    var params = BoosterParams(40, 0.2, TreeParams(15, 20, 1.0, 1e-3))
    var goss = GossParams.enable(0.2, 0.1, 3, 0)

    var binary = train(
        data, labels, BINARY_LOGISTIC, params, [], 0.9,
        BaggingParams.disabled(), goss,
    )
    var correct = 0
    for r in range(n_rows):
        var predicted = 1.0 if binary.predict_row(data, r) >= 0.5 else 0.0
        if predicted == labels[r]:
            correct += 1
    assert_true(Float64(correct) / Float64(n_rows) > 0.9)

    var quantile = train(
        data, signal, QUANTILE, params, [], 0.9,
        BaggingParams.disabled(), goss,
    )
    # A 0.9-quantile fit must sit above the target for most rows.
    var above = 0
    for r in range(n_rows):
        if quantile.predict_row(data, r) >= signal[r]:
            above += 1
    assert_true(Float64(above) / Float64(n_rows) > 0.7)


def test_multiclass_shares_one_sample_per_round() raises:
    var n_rows = 600
    var features = _features(n_rows, 3)
    var signal = _regression_target(features, n_rows)
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        if signal[r] < 0.2:
            labels.append(0)
        elif signal[r] < 1.2:
            labels.append(1)
        else:
            labels.append(2)
    var data = bin_equal_width(features, n_rows, 3, 32)
    var params = BoosterParams(30, 0.3, TreeParams(15, 10, 1.0, 1e-3))
    var goss = GossParams.enable(0.2, 0.1, 3, 0)

    var model = train_multiclass(
        data, labels, 3, params, [], BaggingParams.disabled(), goss
    )
    var repeat = train_multiclass(
        data, labels, 3, params, [], BaggingParams.disabled(), goss
    )

    var correct = 0
    for r in range(n_rows):
        var bins: List[Int] = [
            data.bin_at(r, 0), data.bin_at(r, 1), data.bin_at(r, 2)
        ]
        var proba = model.predict_proba_bins(bins)
        var argmax = 0
        for k in range(3):
            if proba[k] > proba[argmax]:
                argmax = k
        if argmax == labels[r]:
            correct += 1
        # One sample per round, reused by every class, is also what makes a
        # repeated run identical.
        var repeated = repeat.predict_raw_bins(bins)
        var original = model.predict_raw_bins(bins)
        for k in range(3):
            assert_equal(original[k], repeated[k])
    assert_true(Float64(correct) / Float64(n_rows) > 0.85)


def test_sampled_model_round_trips() raises:
    # GOSS changes which rows a tree is grown on, not what a tree is, so the
    # model format is untouched and a sampled model must save and load
    # bit-exactly like any other.
    var n_rows = 400
    var n_features = 3
    var features = _features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var params = BoosterParams(20, 0.2, TreeParams(15, 10, 1.0, 1e-3))

    var model = fit(
        features,
        n_rows,
        n_features,
        target,
        SQUARED_ERROR,
        params,
        64,
        [],
        0.9,
        CPU_DEVICE,
        BaggingParams.disabled(),
        GossParams.enable(0.2, 0.1, 3, 0),
    )
    save_model(model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)

    assert_equal(len(model.booster.trees), len(loaded.booster.trees))
    for r in range(n_rows):
        var row = List[Float64](capacity=n_features)
        for f in range(n_features):
            row.append(features[f * n_rows + r])
        assert_equal(model.predict(row), loaded.predict(row))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
