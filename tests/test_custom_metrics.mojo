"""Tests for custom validation metrics (src/mojoboost/custom_metric.mojo).

The equivalence anchor is `train_with_valid`: a custom metric that computes
the objective's own loss must stop at the same round and produce the same
trees. The rest covers direction, ties, min_delta, several metrics with an
explicit primary, several validation sets, and callback failure.
"""

from std.math import log
from std.os import remove
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojoboost import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BinnedMatrix,
    BoosterParams,
    TreeParams,
    bin_equal_width,
    load_model,
    save_model,
    train_custom_with_valid,
    train_with_valid,
)
from mojoboost.custom_metric import (
    CustomMetric,
    MetricSuite,
    RawValidSet,
    ValidSet,
    fit_with_metrics,
    response_scale,
    train_custom_with_metrics,
    train_with_metric,
    train_with_metrics,
)
from mojoboost.objective import (
    mean_label,
    squared_error_grad_hess,
    squared_error_loss,
)

comptime _NAN = Float64(0.0) / Float64(0.0)
comptime _TMP_PATH = "./.test_custom_metric_model.tmp"


def _features() -> List[Float64]:
    return [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def _target() -> List[Float64]:
    return [0.5, 0.5, 1.5, 1.5, 4.0, 4.0, 9.0, 9.0]


def _params(n_rounds: Int) -> BoosterParams:
    return BoosterParams(n_rounds, 0.3, TreeParams(4, 1, 1.0, 1e-3))


def _train_data() raises -> BinnedMatrix:
    return bin_equal_width(_features(), n_rows=8, n_features=1, n_bins=8)


def _valid_data() raises -> BinnedMatrix:
    var valid_features: List[Float64] = [0.5, 2.5, 4.5, 6.5]
    return bin_equal_width(valid_features, n_rows=4, n_features=1, n_bins=8)


def _valid_target() -> List[Float64]:
    return [0.5, 1.5, 4.0, 6.0]


def _one_valid() raises -> List[ValidSet]:
    var out: List[ValidSet] = [
        ValidSet("valid", _valid_data(), _valid_target())
    ]
    return out^


def mse(pred: List[Float64], target: List[Float64]) raises -> Float64:
    """The squared-error objective's own validation loss."""
    var total = 0.0
    for r in range(len(pred)):
        var d = pred[r] - target[r]
        total += d * d
    return total / Float64(len(pred))


def neg_mse(pred: List[Float64], target: List[Float64]) raises -> Float64:
    return -mse(pred, target)


def mean_pred(pred: List[Float64], target: List[Float64]) raises -> Float64:
    var total = 0.0
    for r in range(len(pred)):
        total += pred[r]
    return total / Float64(len(pred))


def constant_metric(
    pred: List[Float64], target: List[Float64]
) raises -> Float64:
    """Never moves, so it never improves: the tie case."""
    return 1.0


def failing_metric(
    pred: List[Float64], target: List[Float64]
) raises -> Float64:
    raise Error("metric callback exploded")


def nan_metric(pred: List[Float64], target: List[Float64]) raises -> Float64:
    return _NAN


def test_single_metric_matches_builtin_early_stopping() raises:
    # A custom metric that recomputes the built-in validation loss must
    # drive the identical early-stopping decision, tree for tree.
    var data = _train_data()
    var target = _target()
    var params = _params(60)

    var builtin = train_with_valid(
        data,
        target,
        _valid_data(),
        _valid_target(),
        SQUARED_ERROR,
        params,
        early_stopping_rounds=3,
    )
    var custom = train_with_metric(
        data,
        target,
        _one_valid(),
        SQUARED_ERROR,
        params,
        "mse",
        mse,
        early_stopping_rounds=3,
    )
    assert_equal(len(custom.booster.trees), len(builtin.trees))
    assert_true(len(builtin.trees) < 60)
    assert_equal(custom.best_iteration, len(builtin.trees))
    for r in range(8):
        assert_equal(
            custom.booster.predict_row(data, r), builtin.predict_row(data, r)
        )

    # The history holds every round that ran, round 0 (no trees) included,
    # and its last entry is the last round trained, not the truncated one.
    assert_true(custom.history.n_rounds() > len(custom.booster.trees))
    assert_equal(custom.history.n_metrics(), 1)
    assert_equal(custom.history.n_valid(), 1)
    assert_true(custom.stopped_early)


def test_maximize_matches_minimize() raises:
    # Negating a lower-is-better metric and flipping higher_is_better must
    # select exactly the same rounds.
    var data = _train_data()
    var target = _target()
    var params = _params(60)

    var minimized = train_with_metric(
        data, target, _one_valid(), SQUARED_ERROR, params, "mse", mse,
        early_stopping_rounds=3,
    )
    var maximized = train_with_metric(
        data, target, _one_valid(), SQUARED_ERROR, params, "neg_mse", neg_mse,
        higher_is_better=True,
        early_stopping_rounds=3,
    )
    assert_equal(maximized.best_iteration, minimized.best_iteration)
    assert_equal(
        len(maximized.booster.trees), len(minimized.booster.trees)
    )
    assert_equal(maximized.best_score, -minimized.best_score)
    for r in range(8):
        assert_equal(
            maximized.booster.predict_row(data, r),
            minimized.booster.predict_row(data, r),
        )


def test_wrong_direction_keeps_no_trees() raises:
    # Declaring a falling loss as higher-is-better means it never improves,
    # so the best round stays 0 and the ensemble comes back empty.
    var data = _train_data()
    var result = train_with_metric(
        data, _target(), _one_valid(), SQUARED_ERROR, _params(60),
        "mse", mse,
        higher_is_better=True,
        early_stopping_rounds=3,
    )
    assert_equal(result.best_iteration, 0)
    assert_equal(len(result.booster.trees), 0)
    assert_true(result.stopped_early)


def test_ties_are_not_improvements() raises:
    # A metric that returns the same value every round never improves, so
    # patience runs out after exactly early_stopping_rounds rounds.
    var data = _train_data()
    var result = train_with_metric(
        data, _target(), _one_valid(), SQUARED_ERROR, _params(60),
        "constant", constant_metric,
        early_stopping_rounds=4,
    )
    assert_equal(result.best_iteration, 0)
    assert_equal(len(result.booster.trees), 0)
    assert_true(result.stopped_early)
    # Round 0 plus the four rounds it took to run out of patience.
    assert_equal(result.history.n_rounds(), 5)
    assert_equal(result.best_score, 1.0)


def test_min_delta_stops_sooner() raises:
    var data = _train_data()
    var target = _target()
    var params = _params(60)

    var strict = train_with_metric(
        data, target, _one_valid(), SQUARED_ERROR, params, "mse", mse,
        early_stopping_rounds=3, min_delta=1.0,
    )
    var loose = train_with_metric(
        data, target, _one_valid(), SQUARED_ERROR, params, "mse", mse,
        early_stopping_rounds=3, min_delta=0.0,
    )
    assert_true(len(strict.booster.trees) < len(loose.booster.trees))

    # min_delta applies in the maximizing direction too, symmetrically.
    var strict_max = train_with_metric(
        data, target, _one_valid(), SQUARED_ERROR, params, "neg_mse", neg_mse,
        higher_is_better=True, early_stopping_rounds=3, min_delta=1.0,
    )
    assert_equal(
        len(strict_max.booster.trees), len(strict.booster.trees)
    )


def test_primary_metric_selects_the_round() raises:
    # Two metrics, one improving and one frozen. Only the primary decides
    # which round the ensemble is truncated to; the frozen one is recorded
    # but, with its early-stopping flag off, never stops the run.
    var data = _train_data()
    var target = _target()
    var params = _params(20)

    def pair(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        if metric == 0:
            return mse(pred, y)
        return constant_metric(pred, y)

    var metrics: List[CustomMetric] = [
        CustomMetric("mse"),
        CustomMetric("frozen", use_for_early_stopping=False),
    ]

    var by_mse = train_with_metrics(
        data, target, _one_valid(), SQUARED_ERROR, params,
        MetricSuite(metrics.copy(), pair, 0),
        early_stopping_rounds=5,
    )
    var by_frozen = train_with_metrics(
        data, target, _one_valid(), SQUARED_ERROR, params,
        MetricSuite(metrics.copy(), pair, 1),
        early_stopping_rounds=5,
    )

    assert_true(by_mse.best_iteration > 0)
    assert_equal(len(by_mse.booster.trees), by_mse.best_iteration)
    # The frozen metric peaks at round 0, so choosing it as primary keeps
    # nothing, even though training itself ran exactly as far.
    assert_equal(by_frozen.best_iteration, 0)
    assert_equal(len(by_frozen.booster.trees), 0)
    assert_equal(by_frozen.history.n_rounds(), by_mse.history.n_rounds())
    assert_equal(by_frozen.best_score, 1.0)

    # Both metrics are recorded for every round either way.
    assert_equal(by_mse.history.n_metrics(), 2)
    for round in range(by_mse.history.n_rounds()):
        assert_equal(by_mse.history.value(round, 0, 1), 1.0)
        assert_equal(
            by_mse.history.value(round, 0, 0),
            by_frozen.history.value(round, 0, 0),
        )


def test_early_stopping_flag_selects_the_watchers() raises:
    # The primary metric keeps improving, but a second, watched metric is
    # frozen: patience belongs to the watched set, so the run stops even
    # though the primary is still getting better.
    var data = _train_data()
    var target = _target()
    var params = _params(60)

    def pair(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        if metric == 0:
            return mse(pred, y)
        return constant_metric(pred, y)

    var watched: List[CustomMetric] = [
        CustomMetric("mse", use_for_early_stopping=False),
        CustomMetric("frozen"),
    ]
    var stopped = train_with_metrics(
        data, target, _one_valid(), SQUARED_ERROR, params,
        MetricSuite(watched^, pair, 0),
        early_stopping_rounds=4,
    )
    assert_true(stopped.stopped_early)
    assert_equal(stopped.history.n_rounds(), 5)
    # Truncation still follows the primary metric, which improved every
    # round, so all four trees survive.
    assert_equal(stopped.best_iteration, 4)
    assert_equal(len(stopped.booster.trees), 4)

    # With nothing watched at all, early stopping has no signal to act on.
    var unwatched: List[CustomMetric] = [
        CustomMetric("mse", use_for_early_stopping=False)
    ]

    def only_mse(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        return mse(pred, y)

    with assert_raises(contains="use_for_early_stopping"):
        _ = train_with_metrics(
            data, target, _one_valid(), SQUARED_ERROR, params,
            MetricSuite(unwatched^, only_mse, 0),
            early_stopping_rounds=3,
        )


def test_multiple_validation_sets() raises:
    # Two validation sets: the training data itself, which keeps improving,
    # and a flipped-label set, which only degrades. Watching both stops the
    # run early; watching only the first does not.
    var data = _train_data()
    var target = _target()
    var flipped: List[Float64] = [9.0, 9.0, 4.0, 4.0, 1.5, 1.5, 0.5, 0.5]
    var params = _params(40)

    var both: List[ValidSet] = [
        ValidSet("train", data.copy(), target.copy()),
        ValidSet("flipped", data.copy(), flipped.copy()),
    ]
    var result = train_with_metric(
        data, target, both^, SQUARED_ERROR, params, "mse", mse,
        early_stopping_rounds=3,
    )
    assert_true(result.stopped_early)
    assert_equal(result.history.n_valid(), 2)
    assert_equal(len(result.history.series(0, 0)), result.history.n_rounds())
    assert_equal(len(result.history.series(1, 0)), result.history.n_rounds())

    # The first set improves round over round; the second gets worse. The
    # second is what runs out of patience.
    assert_true(result.history.value(1, 0, 0) < result.history.value(0, 0, 0))
    assert_true(result.history.value(1, 1, 0) > result.history.value(0, 1, 0))
    # Truncation follows the primary metric on the first set, which was
    # still improving when the second set stopped the run.
    assert_equal(result.best_iteration, len(result.booster.trees))
    assert_equal(result.best_iteration, 3)

    var only_first: List[ValidSet] = [
        ValidSet("train", data.copy(), target.copy())
    ]
    var longer = train_with_metric(
        data, target, only_first^, SQUARED_ERROR, params, "mse", mse,
        early_stopping_rounds=3,
    )
    assert_true(len(longer.booster.trees) > len(result.booster.trees))


def test_history_matches_direct_scoring() raises:
    # With early stopping off, nothing is truncated, so the last recorded
    # round must equal the metric recomputed on the returned model.
    var data = _train_data()
    var target = _target()
    var valid_data = _valid_data()
    var valid_target = _valid_target()
    var result = train_with_metric(
        data, target, _one_valid(), SQUARED_ERROR, _params(12),
        "mse", mse,
    )
    assert_equal(len(result.booster.trees), 12)
    assert_equal(result.history.n_rounds(), 13)
    assert_true(not result.stopped_early)

    var pred = List[Float64](capacity=4)
    for r in range(4):
        pred.append(result.booster.predict_raw_row(valid_data, r))
    var direct = mse(pred, valid_target)
    assert_true(abs(result.history.value(12, 0, 0) - direct) < 1e-12)

    # Round 0 is the base-score-only model.
    var base = List[Float64](capacity=4)
    for _ in range(4):
        base.append(result.booster.base_score)
    assert_true(
        abs(result.history.value(0, 0, 0) - mse(base, valid_target)) < 1e-12
    )
    # best_iteration still reports where the metric peaked.
    for round in range(result.history.n_rounds()):
        assert_true(
            result.history.value(round, 0, 0)
            >= result.history.value(result.best_iteration, 0, 0)
        )


def test_metrics_receive_raw_scores() raises:
    # LightGBM's feval convention: predictions arrive before the inverse
    # link. For binary logistic the base score is the log-odds of the label
    # mean, not the mean itself.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var labels: List[Float64] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var valid: List[ValidSet] = [
        ValidSet("valid", data.copy(), labels.copy())
    ]
    var result = train_with_metric(
        data, labels, valid^, BINARY_LOGISTIC, _params(3),
        "mean_raw", mean_pred,
    )
    var p = 0.25
    var expected = log(p / (1.0 - p))
    assert_true(abs(result.history.value(0, 0, 0) - expected) < 1e-12)

    # response_scale is the documented way back to probabilities. The
    # tolerance is the accuracy of the library's exp, not of the link.
    var raw: List[Float64] = [expected, 0.0]
    var probs = response_scale(BINARY_LOGISTIC, raw)
    assert_true(abs(probs[0] - p) < 1e-8)
    assert_true(abs(probs[1] - 0.5) < 1e-12)
    var identity = response_scale(SQUARED_ERROR, raw)
    assert_equal(identity[0], expected)


def test_custom_objective_with_custom_metric() raises:
    # Metrics and objectives are independent: a custom objective driven by
    # a custom metric must match train_custom_with_valid driven by the
    # equivalent scalar loss.
    var data = _train_data()
    var target = _target()
    var params = _params(60)
    var base = mean_label(target, [])

    var reference = train_custom_with_valid(
        data,
        target,
        _valid_data(),
        _valid_target(),
        squared_error_grad_hess,
        squared_error_loss,
        params,
        early_stopping_rounds=3,
        base_score=base,
    )

    def single(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        return mse(pred, y)

    var metrics: List[CustomMetric] = [CustomMetric("mse")]
    var custom = train_custom_with_metrics(
        data,
        target,
        _one_valid(),
        squared_error_grad_hess,
        params,
        MetricSuite(metrics^, single, 0),
        early_stopping_rounds=3,
        base_score=base,
    )
    assert_equal(len(custom.booster.trees), len(reference.trees))
    assert_true(len(reference.trees) < 60)
    for r in range(8):
        assert_equal(
            custom.booster.predict_row(data, r), reference.predict_row(data, r)
        )


def test_callback_errors_propagate() raises:
    var data = _train_data()
    with assert_raises(contains="metric callback exploded"):
        _ = train_with_metric(
            data, _target(), _one_valid(), SQUARED_ERROR, _params(5),
            "boom", failing_metric,
            early_stopping_rounds=3,
        )

    # A metric that fails only after a few rounds still aborts training.
    def late_failure(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        var value = mse(pred, y)
        if value < 4.0:
            raise Error("metric callback exploded")
        return value

    var metrics: List[CustomMetric] = [CustomMetric("late")]
    with assert_raises(contains="metric callback exploded"):
        _ = train_with_metrics(
            data, _target(), _one_valid(), SQUARED_ERROR, _params(60),
            MetricSuite(metrics^, late_failure, 0),
            early_stopping_rounds=3,
        )


def test_non_finite_metric_raises() raises:
    var data = _train_data()
    with assert_raises(contains="non-finite"):
        _ = train_with_metric(
            data, _target(), _one_valid(), SQUARED_ERROR, _params(5),
            "nan", nan_metric,
            early_stopping_rounds=3,
        )


def test_metric_suite_validation() raises:
    var data = _train_data()
    var target = _target()
    var params = _params(5)

    def any_metric(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        return mse(pred, y)

    var empty = List[CustomMetric]()
    with assert_raises(contains="at least one metric"):
        _ = train_with_metrics(
            data, target, _one_valid(), SQUARED_ERROR, params,
            MetricSuite(empty^, any_metric, 0),
        )

    var duplicate: List[CustomMetric] = [
        CustomMetric("mse"), CustomMetric("mse")
    ]
    with assert_raises(contains="duplicate metric name"):
        _ = train_with_metrics(
            data, target, _one_valid(), SQUARED_ERROR, params,
            MetricSuite(duplicate^, any_metric, 0),
        )

    var unnamed: List[CustomMetric] = [CustomMetric("")]
    with assert_raises(contains="metric names must not be empty"):
        _ = train_with_metrics(
            data, target, _one_valid(), SQUARED_ERROR, params,
            MetricSuite(unnamed^, any_metric, 0),
        )

    var one: List[CustomMetric] = [CustomMetric("mse")]
    with assert_raises(contains="primary metric index out of range"):
        _ = train_with_metrics(
            data, target, _one_valid(), SQUARED_ERROR, params,
            MetricSuite(one.copy(), any_metric, 1),
        )
    with assert_raises(contains="primary metric index out of range"):
        _ = train_with_metrics(
            data, target, _one_valid(), SQUARED_ERROR, params,
            MetricSuite(one.copy(), any_metric, -1),
        )

    with assert_raises(contains="early_stopping_rounds must not be negative"):
        _ = train_with_metrics(
            data, target, _one_valid(), SQUARED_ERROR, params,
            MetricSuite(one.copy(), any_metric, 0),
            early_stopping_rounds=-1,
        )
    with assert_raises(contains="min_delta must not be negative"):
        _ = train_with_metrics(
            data, target, _one_valid(), SQUARED_ERROR, params,
            MetricSuite(one.copy(), any_metric, 0),
            early_stopping_rounds=3,
            min_delta=-1e-9,
        )


def test_validation_set_validation() raises:
    var data = _train_data()
    var target = _target()
    var params = _params(5)

    def any_metric(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        return mse(pred, y)

    var one: List[CustomMetric] = [CustomMetric("mse")]

    var none = List[ValidSet]()
    with assert_raises(contains="at least one validation set"):
        _ = train_with_metrics(
            data, target, none^, SQUARED_ERROR, params,
            MetricSuite(one.copy(), any_metric, 0),
        )

    var duplicate: List[ValidSet] = [
        ValidSet("v", _valid_data(), _valid_target()),
        ValidSet("v", _valid_data(), _valid_target()),
    ]
    with assert_raises(contains="duplicate validation set name"):
        _ = train_with_metrics(
            data, target, duplicate^, SQUARED_ERROR, params,
            MetricSuite(one.copy(), any_metric, 0),
        )

    var unnamed: List[ValidSet] = [
        ValidSet("", _valid_data(), _valid_target())
    ]
    with assert_raises(contains="validation set names must not be empty"):
        _ = train_with_metrics(
            data, target, unnamed^, SQUARED_ERROR, params,
            MetricSuite(one.copy(), any_metric, 0),
        )

    var short_target: List[Float64] = [0.5, 1.5]
    var mismatched: List[ValidSet] = [
        ValidSet("v", _valid_data(), short_target^)
    ]
    with assert_raises(contains="target length must equal its n_rows"):
        _ = train_with_metrics(
            data, target, mismatched^, SQUARED_ERROR, params,
            MetricSuite(one.copy(), any_metric, 0),
        )

    var wide_features: List[Float64] = [
        0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0,
    ]
    var wide: List[ValidSet] = [
        ValidSet(
            "v",
            bin_equal_width(wide_features, n_rows=4, n_features=2, n_bins=4),
            _valid_target(),
        )
    ]
    with assert_raises(contains="same features as the training data"):
        _ = train_with_metrics(
            data, target, wide^, SQUARED_ERROR, params,
            MetricSuite(one.copy(), any_metric, 0),
        )


def test_fit_with_metrics_bins_validation_sets() raises:
    # The raw-feature entry point bins validation rows with the mapper
    # fitted on the training data, so a validation value between two
    # training values lands in the bin a deployed model would give it.
    var features = _features()
    var target = _target()
    var valid_features: List[Float64] = [0.5, 2.5, 4.5, 6.5]
    var valid_target = _valid_target()

    def single(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        return mse(pred, y)

    var metrics: List[CustomMetric] = [CustomMetric("mse")]
    var valid_sets: List[RawValidSet] = [
        RawValidSet("valid", valid_features.copy(), 4, valid_target.copy())
    ]
    var result = fit_with_metrics(
        features,
        8,
        1,
        target,
        valid_sets^,
        SQUARED_ERROR,
        _params(60),
        MetricSuite(metrics^, single, 0),
        early_stopping_rounds=3,
    )
    assert_true(result.best_iteration > 0)
    assert_true(result.stopped_early)
    assert_equal(result.history.n_rounds() > result.best_iteration, True)

    # The kept model is the one the metric picked: scoring it directly
    # reproduces the recorded best value.
    var pred = List[Float64](capacity=4)
    for r in range(4):
        var row: List[Float64] = [valid_features[r]]
        pred.append(result.model.predict_raw(row))
    assert_true(abs(mse(pred, valid_target) - result.best_score) < 1e-12)

    # Nothing about the model changed, so it round-trips through the
    # existing serialization format bit for bit.
    save_model(result.model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)
    for r in range(4):
        var row: List[Float64] = [valid_features[r]]
        assert_equal(loaded.predict_raw(row), result.model.predict_raw(row))


def test_mismatched_raw_validation_shape_raises() raises:
    def single(
        metric: Int, valid: Int, pred: List[Float64], y: List[Float64]
    ) raises -> Float64:
        return mse(pred, y)

    var metrics: List[CustomMetric] = [CustomMetric("mse")]
    var short_features: List[Float64] = [0.5, 2.5]
    var valid_sets: List[RawValidSet] = [
        RawValidSet("valid", short_features^, 4, _valid_target())
    ]
    with assert_raises(contains="n_rows * n_features feature values"):
        _ = fit_with_metrics(
            _features(),
            8,
            1,
            _target(),
            valid_sets^,
            SQUARED_ERROR,
            _params(5),
            MetricSuite(metrics^, single, 0),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
