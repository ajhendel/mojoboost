"""Tests for per-iteration training callbacks (src/mojoboost/callback.mojo
and `train_with_callbacks` in src/mojoboost/custom_metric.mojo).

The equivalence anchor is `train_with_metrics`: a callback that only watches
must leave the model exactly as it was, because `train_with_metrics` is now
`train_with_callbacks` with `no_callback` and a logging callback that changed
the model would be a bug in the loop rather than in the callback.

The learning-rate schedule is checked against that same anchor tree by tree:
before the round a schedule first changes the rate, the baked trees must be
the unbaked ones times the original rate, exactly, with no tolerance.
"""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojoboost import (
    SQUARED_ERROR,
    BinnedMatrix,
    BoosterParams,
    TreeParams,
    bin_equal_width,
)
from mojoboost.callback import (
    ABORT,
    AFTER_ITERATION,
    BEFORE_ITERATION,
    CONTINUE,
    IterationEnv,
    STOP,
    check_resettable,
    scale_tree_values,
)
from mojoboost.custom_metric import (
    CustomMetric,
    MetricSuite,
    ValidSet,
    train_with_callbacks,
    train_with_metrics,
)

comptime _RATE = 0.3


def _features() -> List[Float64]:
    return [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def _target() -> List[Float64]:
    return [0.5, 0.5, 1.5, 1.5, 4.0, 4.0, 9.0, 9.0]


def _valid_features() -> List[Float64]:
    return [0.5, 2.5, 4.5, 6.5]


def _valid_target() -> List[Float64]:
    return [0.5, 1.5, 4.0, 9.0]


def _params(n_rounds: Int) -> BoosterParams:
    return BoosterParams(n_rounds, _RATE, TreeParams(4, 1, 1.0, 1e-3))


def _train_data() raises -> BinnedMatrix:
    return bin_equal_width(_features(), n_rows=8, n_features=1, n_bins=8)


def _valid_data() raises -> BinnedMatrix:
    return bin_equal_width(_valid_features(), n_rows=4, n_features=1, n_bins=8)


def _valid_sets() raises -> List[ValidSet]:
    return [ValidSet("valid", _valid_data(), _valid_target())]


def _mse(pred: List[Float64], target: List[Float64]) raises -> Float64:
    var total = 0.0
    for i in range(len(target)):
        var d = pred[i] - target[i]
        total += d * d
    return total / Float64(len(target))


def _mse_metric(
    metric: Int, valid: Int, pred: List[Float64], target: List[Float64]
) raises -> Float64:
    return _mse(pred, target)


def _metrics() -> List[CustomMetric]:
    return [CustomMetric("mse")]


def test_watching_callback_leaves_the_model_alone() raises:
    """A callback that only observes must not perturb training: same trees,
    same learning rate, same predictions to the bit."""
    var data = _train_data()

    def watch(phase: Int, mut env: IterationEnv) raises -> Int:
        return CONTINUE

    var plain = train_with_metrics(
        data, _target(), _valid_sets(), SQUARED_ERROR, _params(6), MetricSuite(_metrics(), _mse_metric, 0)
    )
    var watched = train_with_callbacks(
        data,
        _target(),
        _valid_sets(),
        SQUARED_ERROR,
        _params(6),
        MetricSuite(_metrics(), _mse_metric, 0),
        watch,
    )
    assert_equal(len(watched.booster.trees), len(plain.booster.trees))
    assert_equal(watched.booster.learning_rate, plain.booster.learning_rate)
    for r in range(8):
        assert_equal(
            watched.booster.predict_raw_row(data, r),
            plain.booster.predict_raw_row(data, r),
        )


def test_phases_alternate_in_order() raises:
    """Every round calls the callback twice, before then after, with the
    round's 0-based index both times."""
    var phases = List[Int]()
    var rounds = List[Int]()

    def record(phase: Int, mut env: IterationEnv) raises {
        mut phases, mut rounds
    } -> Int:
        phases.append(phase)
        rounds.append(env.iteration)
        return CONTINUE

    var result = train_with_callbacks(
        _train_data(),
        _target(),
        _valid_sets(),
        SQUARED_ERROR,
        _params(3),
        MetricSuite(_metrics(), _mse_metric, 0),
        record,
    )
    assert_equal(len(result.booster.trees), 3)
    assert_equal(len(phases), 6)
    for i in range(3):
        assert_equal(phases[2 * i], BEFORE_ITERATION)
        assert_equal(phases[2 * i + 1], AFTER_ITERATION)
        assert_equal(rounds[2 * i], i)
        assert_equal(rounds[2 * i + 1], i)


def test_environment_reports_the_round_range() raises:
    """The round range bounds every call, in both phases."""
    var seen = List[Int]()

    def record(phase: Int, mut env: IterationEnv) raises {mut seen} -> Int:
        seen.append(env.begin_iteration)
        seen.append(env.end_iteration)
        return CONTINUE

    _ = train_with_callbacks(
        _train_data(),
        _target(),
        _valid_sets(),
        SQUARED_ERROR,
        _params(4),
        MetricSuite(_metrics(), _mse_metric, 0),
        record,
    )
    for i in range(len(seen) // 2):
        assert_equal(seen[2 * i], 0)
        assert_equal(seen[2 * i + 1], 4)


def test_evaluation_is_empty_before_and_scored_after() raises:
    """Metrics exist only in the after phase, and match the history."""
    var before_lengths = List[Int]()
    var after_values = List[Float64]()

    def record(phase: Int, mut env: IterationEnv) raises {
        mut before_lengths, mut after_values
    } -> Int:
        if phase == BEFORE_ITERATION:
            before_lengths.append(len(env.evaluation))
        else:
            after_values.append(env.value(0, 0))
        return CONTINUE

    var result = train_with_callbacks(
        _train_data(),
        _target(),
        _valid_sets(),
        SQUARED_ERROR,
        _params(4),
        MetricSuite(_metrics(), _mse_metric, 0),
        record,
    )
    for i in range(len(before_lengths)):
        assert_equal(before_lengths[i], 0)
    # History index 0 is the base-score-only model, so the callback's value
    # for round i is history index i + 1.
    assert_equal(len(after_values), 4)
    for i in range(4):
        assert_equal(after_values[i], result.history.value(i + 1, 0, 0))


def test_value_raises_before_the_round_is_scored() raises:
    """Reading a metric in the before phase is an error, not a stale value."""

    def peek(phase: Int, mut env: IterationEnv) raises -> Int:
        if phase == BEFORE_ITERATION:
            _ = env.value(0, 0)
        return CONTINUE

    with assert_raises(contains="before-iteration phase"):
        _ = train_with_callbacks(
            _train_data(),
            _target(),
            _valid_sets(),
            SQUARED_ERROR,
            _params(3),
            MetricSuite(_metrics(), _mse_metric, 0),
            peek,
        )


def test_stop_after_iteration_truncates_to_best() raises:
    """STOP ends the run and rolls back to the primary metric's best round,
    the way LightGBM's EarlyStopException does."""

    def stop_at_two(phase: Int, mut env: IterationEnv) raises -> Int:
        if phase == AFTER_ITERATION and env.iteration == 2:
            return STOP
        return CONTINUE

    var result = train_with_callbacks(
        _train_data(),
        _target(),
        _valid_sets(),
        SQUARED_ERROR,
        _params(20),
        MetricSuite(_metrics(), _mse_metric, 0),
        stop_at_two,
    )
    assert_true(result.stopped_early)
    # Three rounds ran; the ensemble keeps the best of them, which on this
    # monotonically improving fit is all three.
    assert_equal(result.best_iteration, 3)
    assert_equal(len(result.booster.trees), 3)


def test_stop_before_iteration_grows_nothing_more() raises:
    """STOP in the before phase ends the run without growing that round."""

    def stop_at_two(phase: Int, mut env: IterationEnv) raises -> Int:
        if phase == BEFORE_ITERATION and env.iteration == 2:
            return STOP
        return CONTINUE

    var result = train_with_callbacks(
        _train_data(),
        _target(),
        _valid_sets(),
        SQUARED_ERROR,
        _params(20),
        MetricSuite(_metrics(), _mse_metric, 0),
        stop_at_two,
    )
    assert_true(result.stopped_early)
    assert_equal(len(result.booster.trees), 2)


def test_abort_raises_from_either_phase() raises:
    """ABORT fails the run; nothing is returned."""

    def abort_before(phase: Int, mut env: IterationEnv) raises -> Int:
        return ABORT if phase == BEFORE_ITERATION else CONTINUE

    def abort_after(phase: Int, mut env: IterationEnv) raises -> Int:
        return ABORT if phase == AFTER_ITERATION else CONTINUE

    with assert_raises(contains="before-iteration phase"):
        _ = train_with_callbacks(
            _train_data(),
            _target(),
            _valid_sets(),
            SQUARED_ERROR,
            _params(5),
            MetricSuite(_metrics(), _mse_metric, 0),
            abort_before,
        )
    with assert_raises(contains="after-iteration phase"):
        _ = train_with_callbacks(
            _train_data(),
            _target(),
            _valid_sets(),
            SQUARED_ERROR,
            _params(5),
            MetricSuite(_metrics(), _mse_metric, 0),
            abort_after,
        )


def test_a_callback_error_propagates() raises:
    """A callback that raises fails the run with its own message rather than
    being swallowed into a control code."""

    def boom(phase: Int, mut env: IterationEnv) raises -> Int:
        raise Error("metric store unreachable")

    with assert_raises(contains="metric store unreachable"):
        _ = train_with_callbacks(
            _train_data(),
            _target(),
            _valid_sets(),
            SQUARED_ERROR,
            _params(5),
            MetricSuite(_metrics(), _mse_metric, 0),
            boom,
        )


def test_a_schedule_that_never_moves_leaves_the_rate_alone() raises:
    """Assigning the same rate back is not a change, so nothing is baked and
    the model is the one `train_with_metrics` builds."""
    var data = _train_data()

    def hold(phase: Int, mut env: IterationEnv) raises -> Int:
        if phase == BEFORE_ITERATION:
            env.params.learning_rate = _RATE
        return CONTINUE

    var plain = train_with_metrics(
        data, _target(), _valid_sets(), SQUARED_ERROR, _params(5), MetricSuite(_metrics(), _mse_metric, 0)
    )
    var held = train_with_callbacks(
        data,
        _target(),
        _valid_sets(),
        SQUARED_ERROR,
        _params(5),
        MetricSuite(_metrics(), _mse_metric, 0),
        hold,
    )
    assert_equal(held.booster.learning_rate, _RATE)
    for r in range(8):
        assert_equal(
            held.booster.predict_raw_row(data, r),
            plain.booster.predict_raw_row(data, r),
        )


def test_learning_rate_schedule_bakes_shrinkage_exactly() raises:
    """A rate change moves shrinkage into the leaf values and the stored rate
    to 1.0. The rounds before the change must survive that move untouched:
    each baked tree is the unbaked one times the original rate, exactly."""
    var data = _train_data()

    def halve_after_two(phase: Int, mut env: IterationEnv) raises -> Int:
        if phase == BEFORE_ITERATION and env.iteration >= 2:
            env.params.learning_rate = _RATE / 2.0
        return CONTINUE

    var plain = train_with_metrics(
        data, _target(), _valid_sets(), SQUARED_ERROR, _params(5), MetricSuite(_metrics(), _mse_metric, 0)
    )
    var scheduled = train_with_callbacks(
        data,
        _target(),
        _valid_sets(),
        SQUARED_ERROR,
        _params(5),
        MetricSuite(_metrics(), _mse_metric, 0),
        halve_after_two,
    )
    assert_equal(scheduled.booster.learning_rate, 1.0)
    assert_equal(len(scheduled.booster.trees), len(plain.booster.trees))
    # Rounds 0 and 1 trained at the original rate, so their trees are the
    # plain ones with the rate folded in. No tolerance: the rewrite is the
    # same multiplication prediction would have done.
    for t in range(2):
        for r in range(8):
            assert_equal(
                scheduled.booster.trees[t].predict_row(data, r),
                _RATE * plain.booster.trees[t].predict_row(data, r),
            )
    # The schedule has to have actually changed the fit.
    var differs = False
    for r in range(8):
        if scheduled.booster.predict_raw_row(
            data, r
        ) != plain.booster.predict_raw_row(data, r):
            differs = True
    assert_true(differs)


def test_tree_params_reset_takes_effect() raises:
    """A schedule may also move the tree hyperparameters the loop re-reads."""
    var data = _train_data()

    def widen(phase: Int, mut env: IterationEnv) raises -> Int:
        if phase == BEFORE_ITERATION:
            env.params.tree.num_leaves = 2
        return CONTINUE

    var narrowed = train_with_callbacks(
        data,
        _target(),
        _valid_sets(),
        SQUARED_ERROR,
        _params(4),
        MetricSuite(_metrics(), _mse_metric, 0),
        widen,
    )
    for t in range(len(narrowed.booster.trees)):
        assert_true(narrowed.booster.trees[t].n_leaves <= 2)


def test_resetting_a_fixed_parameter_is_an_error() raises:
    """Silently ignoring a reset the loop cannot honor would train the wrong
    model and say nothing, so it raises instead."""

    def retarget(phase: Int, mut env: IterationEnv) raises -> Int:
        if phase == BEFORE_ITERATION:
            env.params.n_estimators = 99
        return CONTINUE

    with assert_raises(contains="n_estimators is not resettable"):
        _ = train_with_callbacks(
            _train_data(),
            _target(),
            _valid_sets(),
            SQUARED_ERROR,
            _params(5),
            MetricSuite(_metrics(), _mse_metric, 0),
            retarget,
        )


def test_reset_values_are_range_checked() raises:
    """A schedule that walks a parameter out of range fails the run."""

    def to_zero(phase: Int, mut env: IterationEnv) raises -> Int:
        if phase == BEFORE_ITERATION:
            env.params.learning_rate = 0.0
        return CONTINUE

    with assert_raises(contains="learning_rate must be positive"):
        _ = train_with_callbacks(
            _train_data(),
            _target(),
            _valid_sets(),
            SQUARED_ERROR,
            _params(5),
            MetricSuite(_metrics(), _mse_metric, 0),
            to_zero,
        )


def test_check_resettable_accepts_the_resettable_set() raises:
    """The documented set passes; a seed change does not."""
    var before = _params(5)
    var after = _params(5)
    after.learning_rate = 0.01
    after.tree.num_leaves = 8
    after.tree.max_depth = 3
    after.tree.min_data_in_leaf = 2
    after.tree.min_child_hess = 0.5
    after.tree.lambda_l1 = 0.25
    after.tree.lambda_reg = 2.0
    after.tree.feature_fraction = 0.5
    after.tree.feature_fraction_bynode = 0.5
    check_resettable(before, after)

    var reseeded = _params(5)
    reseeded.tree.feature_fraction_seed += 1
    with assert_raises(contains="feature_fraction_seed is not resettable"):
        check_resettable(before, reseeded)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
