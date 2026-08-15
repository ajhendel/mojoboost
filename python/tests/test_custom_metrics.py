"""Custom validation metrics through the Python API.

The Mojo-level rules (direction, ties, min_delta, primary metric, several
validation sets) are tested in tests/test_custom_metrics.mojo. These cover
the Python surface: how metrics are declared, what the callback receives,
what `evals_result_` holds, and which combinations raise.
"""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRegressor


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def neg_mse(y_true, y_pred):
    return -mse(y_true, y_pred)


def _split(X, y, n_valid=100):
    return X[:-n_valid], y[:-n_valid], X[-n_valid:], y[-n_valid:]


def test_metric_drives_early_stopping(regression):
    X, y, Xv, yv = _split(*regression)
    # The default learning rate keeps improving on this fixture for all 200
    # rounds, so overfit faster to guarantee a five-round plateau.
    model = MojoTreesRegressor(n_estimators=200, learning_rate=0.3).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=mse,
        early_stopping_rounds=5,
    )
    assert model.stopped_early_
    assert 0 < model.best_iteration_ < 200
    history = model.evals_result_["valid_0"]["mse"]
    # Round 0 is the base-score-only model, so the record is one longer
    # than the number of rounds trained, and longer still than the kept
    # ensemble.
    assert len(history) > model.best_iteration_ + 1
    assert history[model.best_iteration_] == pytest.approx(model.best_score_)
    assert model.best_score_ == pytest.approx(min(history))
    # The kept model is the one the metric picked.
    assert mse(yv, model.predict(Xv)) == pytest.approx(model.best_score_)


def test_direction_declared_by_the_tuple_form(regression):
    X, y, Xv, yv = _split(*regression)
    lower = MojoTreesRegressor(n_estimators=60).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric=mse, early_stopping_rounds=5
    )
    higher = MojoTreesRegressor(n_estimators=60).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=("neg_mse", neg_mse, True),
        early_stopping_rounds=5,
    )
    assert higher.best_iteration_ == lower.best_iteration_
    assert higher.best_score_ == pytest.approx(-lower.best_score_)
    assert np.allclose(higher.predict(Xv), lower.predict(Xv))


def test_several_metrics_and_an_explicit_primary(regression):
    X, y, Xv, yv = _split(*regression)

    def frozen(y_true, y_pred):
        return 1.0

    metrics = [
        ("mse", mse),
        {"name": "frozen", "func": frozen, "early_stopping": False},
    ]
    by_mse = MojoTreesRegressor(n_estimators=40).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=metrics,
        early_stopping_rounds=5,
        primary_metric="mse",
    )
    by_frozen = MojoTreesRegressor(n_estimators=40).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=metrics,
        early_stopping_rounds=5,
        primary_metric="frozen",
    )
    assert set(by_mse.evals_result_["valid_0"]) == {"mse", "frozen"}
    assert by_mse.best_iteration_ > 0
    # A metric that never improves peaks at round 0, so selecting it keeps
    # nothing, while the run itself is identical.
    assert by_frozen.best_iteration_ == 0
    assert by_frozen.best_score_ == 1.0
    assert (
        by_frozen.evals_result_["valid_0"]["mse"]
        == by_mse.evals_result_["valid_0"]["mse"]
    )


def test_several_validation_sets_are_named_and_recorded(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=25).fit(
        X,
        y,
        eval_set=[(Xv, yv), (X, y)],
        eval_names=["holdout", "train"],
        eval_metric=mse,
    )
    assert list(model.evals_result_) == ["holdout", "train"]
    rounds = len(model.evals_result_["train"]["mse"])
    assert rounds == len(model.evals_result_["holdout"]["mse"]) == 26
    # Nothing is truncated without early stopping, and the training set
    # keeps improving.
    assert model.best_iteration_ == 25
    assert not model.stopped_early_
    train_curve = model.evals_result_["train"]["mse"]
    assert train_curve[-1] < train_curve[0]


def test_callback_receives_raw_scores_from_the_classifier(binary):
    X, y = binary
    seen = {}

    def capture(y_true, y_pred):
        seen["y_true"] = np.asarray(y_true).copy()
        seen["y_pred"] = np.asarray(y_pred).copy()
        return float(np.mean(np.asarray(y_pred)))

    model = MojoTreesClassifier(n_estimators=5).fit(
        X, y, eval_set=[(X, y)], eval_metric=capture
    )
    # Labels arrive encoded, predictions as log-odds: the sigmoid of the
    # last raw vector is what predict_proba returns.
    assert set(np.unique(seen["y_true"])) <= {0.0, 1.0}
    proba = model.predict_proba(X)[:, 1]
    assert np.allclose(1.0 / (1.0 + np.exp(-seen["y_pred"])), proba, atol=1e-8)
    assert not np.all((seen["y_pred"] >= 0.0) & (seen["y_pred"] <= 1.0))


def test_lightgbm_style_tuple_return_is_accepted(regression):
    X, y, Xv, yv = _split(*regression)

    def feval(y_true, y_pred):
        return "mse", mse(y_true, y_pred), False

    model = MojoTreesRegressor(n_estimators=10).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric=("mse", feval)
    )
    assert model.evals_result_["valid_0"]["mse"][-1] == pytest.approx(
        mse(yv, model.predict(Xv))
    )


def test_callback_errors_propagate(regression):
    X, y, Xv, yv = _split(*regression)

    def boom(y_true, y_pred):
        raise ValueError("metric callback exploded")

    with pytest.raises(ValueError, match="metric callback exploded"):
        MojoTreesRegressor(n_estimators=10).fit(
            X, y, eval_set=[(Xv, yv)], eval_metric=boom
        )


def test_refit_without_eval_set_clears_the_record(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=10)
    model.fit(X, y, eval_set=[(Xv, yv)], eval_metric=mse)
    assert hasattr(model, "evals_result_")
    model.fit(X, y)
    assert not hasattr(model, "evals_result_")
    assert not hasattr(model, "best_score_")


def test_validation_errors(regression, multiclass):
    X, y, Xv, yv = _split(*regression)
    reg = MojoTreesRegressor(n_estimators=5)

    with pytest.raises(ValueError, match="eval_metric needs an eval_set"):
        reg.fit(X, y, eval_metric=mse)
    # An eval_set with no eval_metric falls back to the objective's own
    # loss (python/tests/test_eval_set.py covers the built-in names).
    assert list(
        reg.fit(X, y, eval_set=[(Xv, yv)]).evals_result_["valid_0"]
    ) == ["l2"]
    with pytest.raises(ValueError, match="must not be empty"):
        reg.fit(X, y, eval_set=[], eval_metric=mse)
    with pytest.raises(ValueError, match="names must be unique"):
        reg.fit(
            X,
            y,
            eval_set=[(Xv, yv)],
            eval_metric=[("mse", mse), ("mse", neg_mse)],
        )
    with pytest.raises(ValueError, match="one name per eval_set entry"):
        reg.fit(X, y, eval_set=[(Xv, yv)], eval_names=["a", "b"],
                eval_metric=mse)
    with pytest.raises(ValueError, match="is not one of the eval_metric"):
        reg.fit(
            X, y, eval_set=[(Xv, yv)], eval_metric=mse, primary_metric="nope"
        )
    with pytest.raises(ValueError, match="out of range"):
        reg.fit(X, y, eval_set=[(Xv, yv)], eval_metric=mse, primary_metric=3)
    with pytest.raises(ValueError, match="features, but X has"):
        reg.fit(X, y, eval_set=[(Xv[:, :2], yv)], eval_metric=mse)
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        reg.fit(
            X, y, eval_set=[(Xv, yv)], eval_metric=mse,
            early_stopping_rounds=-1,
        )
    with pytest.raises(ValueError, match="min_delta"):
        reg.fit(X, y, eval_set=[(Xv, yv)], eval_metric=mse, min_delta=-1.0)
    with pytest.raises(ValueError, match="needs a callable"):
        reg.fit(X, y, eval_set=[(Xv, yv)], eval_metric=("mse", 3.0))

    # A Python objective callback cannot be paired with them yet.
    def grad_hess(raw, target):
        return raw - target, np.ones_like(raw)

    with pytest.raises(ValueError, match="cannot be combined"):
        MojoTreesRegressor(objective=grad_hess, n_estimators=5).fit(
            X, y, eval_set=[(Xv, yv)], eval_metric=mse
        )

    # Multiclass has its own trainer, and its metrics see one block of
    # raw scores per row rather than one number, so a single-output
    # callable like `mse` cannot score it.
    Xm, ym = multiclass
    scored = MojoTreesClassifier(n_estimators=5).fit(
        Xm, ym, eval_set=[(Xm, ym)], eval_metric="multi_logloss"
    )
    assert len(scored.evals_result_["valid_0"]["multi_logloss"]) == 6
