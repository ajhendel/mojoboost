"""Training callbacks through the Python API.

The loop-level rules (phase order, control codes, parameter resets, the
learning-rate baking) are tested in tests/test_callbacks.mojo. These cover
the Python surface: the environment a callback receives, the four factories,
ordering across a list, what a raising callback leaves behind, and how the
callback spelling of early stopping lines up with the keyword one.
"""

import numpy as np
import pytest

from mojoboost import MojoBoostClassifier, MojoBoostRegressor
from mojoboost.callback import (
    CallbackEnv,
    EarlyStopException,
    RESETTABLE,
    early_stopping,
    log_evaluation,
    record_evaluation,
    reset_parameter,
)


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def _split(X, y, n_valid=100):
    return X[:-n_valid], y[:-n_valid], X[-n_valid:], y[-n_valid:]


def _fit(regression, callbacks, n_estimators=20, **kwargs):
    X, y, Xv, yv = _split(*regression)
    return MojoBoostRegressor(n_estimators=n_estimators).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=("mse", mse),
        callbacks=callbacks,
        **kwargs,
    )


# -- the environment ------------------------------------------------------


def test_environment_fields(regression):
    # Recorded at call time: there is one TrainingHandle per run, as
    # LightGBM has one booster, so keeping the env objects would only show
    # the last round's handle state.
    seen = []

    def watch(env):
        assert isinstance(env, CallbackEnv)
        seen.append(
            (
                env.iteration,
                env.begin_iteration,
                env.end_iteration,
                env.model.current_iteration,
                env.evaluation_result_list,
            )
        )

    _fit(regression, [watch], n_estimators=5)
    assert len(seen) == 5
    for i, (it, begin, end, handle_it, results) in enumerate(seen):
        assert (it, begin, end, handle_it) == (i, 0, 5, i)
        assert len(results) == 1
        data_name, metric_name, value, higher = results[0]
        assert (data_name, metric_name, higher) == ("valid_0", "mse", False)
        assert isinstance(value, float)


def test_the_handle_is_one_object_for_the_run(regression):
    handles = []

    def watch(env):
        handles.append(env.model)

    _fit(regression, [watch], n_estimators=4)
    assert all(h is handles[0] for h in handles)
    assert handles[0].estimator.__class__ is MojoBoostRegressor


def test_after_phase_values_match_evals_result(regression):
    seen = []

    def watch(env):
        seen.append(env.evaluation_result_list[0][2])

    model = _fit(regression, [watch], n_estimators=8)
    # evals_result_ counts the base-score-only model as round 0, so the
    # callback's round i is history index i + 1.
    history = model.evals_result_["valid_0"]["mse"]
    assert seen == pytest.approx(history[1 : len(seen) + 1])


def test_before_phase_sees_no_results(regression):
    lengths = []

    def watch(env):
        lengths.append(len(env.evaluation_result_list))

    watch.before_iteration = True
    _fit(regression, [watch], n_estimators=4)
    assert lengths == [0, 0, 0, 0]


def test_params_expose_only_the_resettable_set(regression):
    seen = {}

    def watch(env):
        seen.update(env.params)

    watch.before_iteration = True
    _fit(regression, [watch], n_estimators=3)
    assert set(seen) == set(RESETTABLE)
    assert seen["learning_rate"] == pytest.approx(0.1)


# -- ordering -------------------------------------------------------------


def test_phase_split_and_order(regression):
    calls = []

    def make(name, order, before):
        def cb(env):
            calls.append((name, env.iteration))

        cb.order = order
        cb.before_iteration = before
        return cb

    # Listed out of order on purpose; `order` decides, not position.
    callbacks = [
        make("after_late", 30, False),
        make("before_b", 10, True),
        make("after_early", 5, False),
        make("before_a", 1, True),
    ]
    _fit(regression, callbacks, n_estimators=2)
    assert calls == [
        ("before_a", 0),
        ("before_b", 0),
        ("after_early", 0),
        ("after_late", 0),
        ("before_a", 1),
        ("before_b", 1),
        ("after_early", 1),
        ("after_late", 1),
    ]


def test_equal_order_keeps_listed_order(regression):
    calls = []

    def make(name):
        def cb(env):
            calls.append(name)

        cb.order = 7
        return cb

    _fit(regression, [make("first"), make("second")], n_estimators=1)
    assert calls == ["first", "second"]


# -- record_evaluation ----------------------------------------------------


def test_record_evaluation_matches_evals_result(regression):
    history = {}
    model = _fit(regression, [record_evaluation(history)], n_estimators=10)
    recorded = history["valid_0"]["mse"]
    # record_evaluation starts at the first round, as LightGBM's does;
    # evals_result_ starts one earlier, at the base-score-only model.
    assert recorded == pytest.approx(
        model.evals_result_["valid_0"]["mse"][1:]
    )


def test_record_evaluation_rejects_a_non_dict():
    with pytest.raises(TypeError):
        record_evaluation([])


# -- log_evaluation -------------------------------------------------------


def test_log_evaluation_period(regression, capsys):
    _fit(regression, [log_evaluation(period=5)], n_estimators=12)
    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split("]")[0] + "]" for line in lines] == ["[5]", "[10]"]
    assert "valid_0's mse:" in lines[0]


def test_log_evaluation_disabled(regression, capsys):
    _fit(regression, [log_evaluation(period=0)], n_estimators=6)
    assert capsys.readouterr().out == ""


# -- reset_parameter ------------------------------------------------------


def test_learning_rate_schedule_changes_the_model(regression):
    plain = _fit(regression, None, n_estimators=12)
    scheduled = _fit(
        regression,
        [reset_parameter(learning_rate=lambda i: 0.1 * (0.5**i))],
        n_estimators=12,
    )
    X, _, Xv, _ = _split(*regression)
    assert not np.allclose(plain.predict(Xv), scheduled.predict(Xv))


def test_schedule_holding_the_rate_is_a_no_op(regression):
    plain = _fit(regression, None, n_estimators=10)
    held = _fit(
        regression,
        [reset_parameter(learning_rate=[0.1] * 10)],
        n_estimators=10,
    )
    _, _, Xv, _ = _split(*regression)
    # The rate never moves, so nothing is baked and the fit is untouched.
    assert held.predict(Xv) == pytest.approx(plain.predict(Xv), abs=0.0)


def test_schedule_reaches_prediction(regression):
    """A rate driven to almost nothing after the first round must leave a
    model close to one trained for a single round."""
    one_round = _fit(regression, None, n_estimators=1)
    decayed = _fit(
        regression,
        [reset_parameter(learning_rate=lambda i: 0.1 if i == 0 else 1e-9)],
        n_estimators=15,
    )
    _, _, Xv, _ = _split(*regression)
    assert decayed.predict(Xv) == pytest.approx(
        one_round.predict(Xv), abs=1e-6
    )


def test_num_leaves_schedule(regression):
    seen = []

    def watch(env):
        seen.append(env.params["num_leaves"])

    watch.before_iteration = True
    watch.order = 99  # after reset_parameter, so it sees the new value
    _fit(
        regression,
        [reset_parameter(num_leaves=[4, 8, 16]), watch],
        n_estimators=3,
    )
    assert seen == [4, 8, 16]


def test_alias_resolves(regression):
    seen = []

    def watch(env):
        seen.append(env.params["min_data_in_leaf"])

    watch.before_iteration = True
    watch.order = 99
    _fit(regression, [reset_parameter(min_child_samples=[3, 3]), watch],
         n_estimators=2)
    assert seen == [3, 3]


def test_unresettable_parameter_raises():
    with pytest.raises(ValueError, match="cannot be reset"):
        reset_parameter(n_estimators=[10, 20])


def test_short_schedule_raises(regression):
    with pytest.raises(ValueError, match="too few for round"):
        _fit(
            regression,
            [reset_parameter(learning_rate=[0.1, 0.05])],
            n_estimators=6,
        )


def test_reset_outside_the_before_phase_raises(regression):
    def late(env):
        env.model.reset_parameter({"learning_rate": 0.01})

    with pytest.raises(RuntimeError, match="only be reset before"):
        _fit(regression, [late], n_estimators=3)


# -- early stopping -------------------------------------------------------


def test_early_stopping_callback_matches_the_keyword(regression):
    X, y, Xv, yv = _split(*regression)
    by_keyword = MojoBoostRegressor(n_estimators=200).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric=("mse", mse),
        early_stopping_rounds=5,
    )
    by_callback = MojoBoostRegressor(n_estimators=200).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric=("mse", mse),
        callbacks=[early_stopping(5, verbose=False)],
    )
    assert by_callback.best_iteration_ == by_keyword.best_iteration_
    assert by_callback.stopped_early_ == by_keyword.stopped_early_
    assert by_callback.predict(Xv) == pytest.approx(by_keyword.predict(Xv))


def test_early_stopping_reports_when_verbose(regression, capsys):
    _fit(regression, [early_stopping(3, verbose=True)], n_estimators=200)
    out = capsys.readouterr().out
    assert "Early stopping, best iteration is:" in out
    assert "valid_0's mse:" in out


def test_early_stopping_silent(regression, capsys):
    _fit(regression, [early_stopping(3, verbose=False)], n_estimators=200)
    assert capsys.readouterr().out == ""


def test_early_stopping_conflicts_with_the_keyword(regression):
    with pytest.raises(ValueError, match="not both"):
        _fit(
            regression,
            [early_stopping(5, verbose=False)],
            n_estimators=50,
            early_stopping_rounds=5,
        )


def test_only_one_early_stopping_callback(regression):
    with pytest.raises(ValueError, match="at most one"):
        _fit(
            regression,
            [early_stopping(5, verbose=False), early_stopping(3, verbose=False)],
            n_estimators=50,
        )


def test_custom_callback_can_stop(regression):
    def stop_at_four(env):
        if env.iteration == 4:
            raise EarlyStopException(env.iteration, 0.0)

    model = _fit(regression, [stop_at_four], n_estimators=100)
    assert model.stopped_early_
    assert model.best_iteration_ <= 5
    assert model.n_iter_ == 5


# -- exceptions -----------------------------------------------------------


class Boom(Exception):
    pass


def test_callback_exception_propagates_by_type(regression):
    def boom(env):
        raise Boom("no thanks")

    with pytest.raises(Boom, match="no thanks"):
        _fit(regression, [boom], n_estimators=10)


def test_a_failed_fit_leaves_the_estimator_unfitted(regression):
    from mojoboost import NotFittedError

    X, y, Xv, yv = _split(*regression)
    model = MojoBoostRegressor(n_estimators=10)

    def boom(env):
        raise Boom("stop")

    with pytest.raises(Boom):
        model.fit(
            X, y, eval_set=[(Xv, yv)], eval_metric=("mse", mse),
            callbacks=[boom],
        )
    assert not model.__sklearn_is_fitted__()
    with pytest.raises(NotFittedError):
        model.predict(Xv)


def test_a_failed_refit_does_not_keep_the_old_model(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoBoostRegressor(n_estimators=10).fit(X, y)
    assert model.__sklearn_is_fitted__()

    def boom(env):
        raise Boom("stop")

    with pytest.raises(Boom):
        model.fit(
            X, y, eval_set=[(Xv, yv)], eval_metric=("mse", mse),
            callbacks=[boom],
        )
    # A half-updated estimator would be worse than an unfitted one.
    assert not model.__sklearn_is_fitted__()


def test_a_failing_before_callback_also_propagates(regression):
    def boom(env):
        raise Boom("early")

    boom.before_iteration = True
    with pytest.raises(Boom, match="early"):
        _fit(regression, [boom], n_estimators=10)


# -- interaction with the rest of the API ---------------------------------


def test_callbacks_need_an_eval_set(regression):
    X, y = regression
    with pytest.raises(ValueError, match="callbacks need an eval_set"):
        MojoBoostRegressor(n_estimators=5).fit(
            X, y, callbacks=[log_evaluation(period=1)]
        )


def test_no_callbacks_is_unchanged(regression):
    """The bridge must not cross the boundary when there is nothing to call,
    and the model must be the one the metric trainer built before callbacks
    existed."""
    without = _fit(regression, None, n_estimators=12)
    empty = _fit(regression, [], n_estimators=12)
    _, _, Xv, _ = _split(*regression)
    assert empty.predict(Xv) == pytest.approx(without.predict(Xv), abs=0.0)


def test_binary_classifier_callbacks(binary):
    X, y, Xv, yv = _split(*binary)
    history = {}
    model = MojoBoostClassifier(n_estimators=8).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric=("mse", mse),
        callbacks=[record_evaluation(history)],
    )
    assert len(history["valid_0"]["mse"]) == model.n_iter_


def test_multiclass_callbacks_are_refused(multiclass):
    X, y, Xv, yv = _split(*multiclass, n_valid=90)
    with pytest.raises(NotImplementedError, match="multiclass"):
        MojoBoostClassifier(n_estimators=5).fit(
            X, y, eval_set=[(Xv, yv)], eval_metric=("mse", mse),
            callbacks=[log_evaluation(period=1)],
        )
