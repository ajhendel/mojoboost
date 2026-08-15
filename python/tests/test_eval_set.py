"""Validation sets, built-in eval metrics, and early stopping.

The Mojo-level rules (direction, ties, min_delta, primary metric, several
validation sets, truncation) are tested in tests/test_custom_metrics.mojo.
python/tests/test_custom_metrics.py covers the callable-metric surface.
These cover what the estimators add on top: the `eval_set` spellings,
`eval_names`, `eval_sample_weight`, LightGBM's metric names and their
values, the fitted attributes, and the multiclass and ranking paths.
"""

import math
import pickle

import numpy as np
import pytest

from mojotrees import (
    MojoTreesClassifier,
    MojoTreesRanker,
    MojoTreesRegressor,
    gpu_available,
    ndcg_score,
)


def _split(X, y, n_valid=60):
    return X[:-n_valid], y[:-n_valid], X[-n_valid:], y[-n_valid:]


def _ranking_data(n_queries=40, per_query=5, n_features=3, seed=5):
    gen = np.random.default_rng(seed)
    n_rows = n_queries * per_query
    X = gen.random((n_rows, n_features))
    y = np.minimum((X[:, 0] * 3.0).astype(np.int64), 2)
    return X, y, [per_query] * n_queries


# -- the eval_set spellings ----------------------------------------------


def test_eval_set_spellings_agree(regression):
    """A list of pairs, one bare pair, and eval_X/eval_y are one set."""
    X, y, Xv, yv = _split(*regression)
    histories = []
    for kwargs in (
        {"eval_set": [(Xv, yv)]},
        {"eval_set": (Xv, yv)},
        {"eval_X": Xv, "eval_y": yv},
    ):
        model = MojoTreesRegressor(n_estimators=15).fit(X, y, **kwargs)
        histories.append(model.evals_result_["valid_0"]["l2"])
    assert histories[0] == histories[1] == histories[2]


def test_eval_X_and_eval_set_together_raise(regression):
    X, y, Xv, yv = _split(*regression)
    with pytest.raises(ValueError, match="either eval_set or eval_X"):
        MojoTreesRegressor(n_estimators=5).fit(
            X, y, eval_set=[(Xv, yv)], eval_X=Xv, eval_y=yv
        )


def test_eval_X_without_eval_y_raises(regression):
    X, y, Xv, _ = _split(*regression)
    with pytest.raises(ValueError, match="must be given together"):
        MojoTreesRegressor(n_estimators=5).fit(X, y, eval_X=Xv)


def test_several_eval_sets_are_named_and_scored(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=12).fit(
        X, y, eval_set=[(Xv, yv), (X, y)], eval_names=["holdout", "train"]
    )
    assert set(model.evals_result_) == {"holdout", "train"}
    # The training set is what the trees were fitted to, so it scores better.
    assert (
        model.evals_result_["train"]["l2"][-1]
        < model.evals_result_["holdout"]["l2"][-1]
    )


def test_default_eval_names(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=5).fit(
        X, y, eval_set=[(Xv, yv), (Xv, yv)]
    )
    assert list(model.evals_result_) == ["valid_0", "valid_1"]


@pytest.mark.parametrize(
    "names, message",
    [
        (["only_one"], "one name per eval_set entry"),
        (["same", "same"], "must be unique"),
    ],
)
def test_bad_eval_names_raise(regression, names, message):
    X, y, Xv, yv = _split(*regression)
    with pytest.raises(ValueError, match=message):
        MojoTreesRegressor(n_estimators=5).fit(
            X, y, eval_set=[(Xv, yv), (Xv, yv)], eval_names=names
        )


def test_malformed_eval_sets_raise(regression):
    X, y, Xv, yv = _split(*regression)
    with pytest.raises(ValueError, match="must not be empty"):
        MojoTreesRegressor(n_estimators=5).fit(X, y, eval_set=[])
    with pytest.raises(ValueError, match="an .X, y. pair"):
        MojoTreesRegressor(n_estimators=5).fit(X, y, eval_set=[Xv])
    with pytest.raises(ValueError, match="features, but X has"):
        MojoTreesRegressor(n_estimators=5).fit(
            X, y, eval_set=[(Xv[:, :2], yv)]
        )
    with pytest.raises(ValueError, match="shape"):
        MojoTreesRegressor(n_estimators=5).fit(X, y, eval_set=[(Xv, yv[:-1])])


def test_validation_arguments_need_an_eval_set(regression):
    X, y = regression
    with pytest.raises(ValueError, match="eval_metric needs an eval_set"):
        MojoTreesRegressor(n_estimators=5).fit(X, y, eval_metric="l2")
    with pytest.raises(ValueError, match="eval_sample_weight needs"):
        MojoTreesRegressor(n_estimators=5).fit(
            X, y, eval_sample_weight=np.ones(len(y))
        )
    with pytest.raises(ValueError, match="early_stopping_rounds needs"):
        MojoTreesRegressor(n_estimators=5).fit(X, y, early_stopping_rounds=3)


# -- built-in metrics ----------------------------------------------------


def test_builtin_regression_metrics_match_numpy(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=20).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric=["l2", "rmse", "l1"]
    )
    pred = model.predict(Xv)
    last = {k: v[-1] for k, v in model.evals_result_["valid_0"].items()}
    assert last["l2"] == pytest.approx(np.mean((pred - yv) ** 2))
    assert last["rmse"] == pytest.approx(math.sqrt(last["l2"]))
    assert last["l1"] == pytest.approx(np.mean(np.abs(pred - yv)))


def test_metric_aliases_resolve_to_one_name(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=8).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric="mse"
    )
    assert list(model.evals_result_["valid_0"]) == ["l2"]


def test_default_metric_follows_the_objective(regression):
    X, y, Xv, yv = _split(*regression)
    for objective, metric in (
        ("regression", "l2"),
        ("mae", "l1"),
        ("huber", "huber"),
        ("quantile", "quantile"),
    ):
        model = MojoTreesRegressor(
            objective=objective, alpha=0.3, n_estimators=6
        ).fit(X, y, eval_set=[(Xv, yv)])
        assert list(model.evals_result_["valid_0"]) == [metric]


def test_quantile_metric_is_the_pinball_loss(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(
        objective="quantile", alpha=0.3, n_estimators=10
    ).fit(X, y, eval_set=[(Xv, yv)])
    pred = model.predict(Xv)
    d = pred - yv
    expected = np.mean(np.where(d >= 0.0, 0.7 * d, -0.3 * d))
    assert model.evals_result_["valid_0"]["quantile"][-1] == pytest.approx(
        expected
    )


def test_unknown_and_wrong_task_metrics_raise(regression):
    X, y, Xv, yv = _split(*regression)
    with pytest.raises(ValueError, match="unknown eval_metric"):
        MojoTreesRegressor(n_estimators=5).fit(
            X, y, eval_set=[(Xv, yv)], eval_metric="nope"
        )
    with pytest.raises(ValueError, match="scores binary models"):
        MojoTreesRegressor(n_estimators=5).fit(
            X, y, eval_set=[(Xv, yv)], eval_metric="auc"
        )


def test_builtin_and_callable_metrics_mix(regression):
    X, y, Xv, yv = _split(*regression)

    def worst(y_true, y_pred):
        return float(np.max(np.abs(np.asarray(y_pred) - y_true)))

    model = MojoTreesRegressor(n_estimators=12).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric=["l2", ("worst", worst)]
    )
    history = model.evals_result_["valid_0"]
    assert set(history) == {"l2", "worst"}
    assert history["worst"][-1] == pytest.approx(
        np.max(np.abs(model.predict(Xv) - yv))
    )


# -- eval_sample_weight ---------------------------------------------------


def test_eval_sample_weight_weights_the_metric(regression, rng):
    X, y, Xv, yv = _split(*regression)
    w = rng.uniform(0.5, 2.0, size=len(yv))
    model = MojoTreesRegressor(n_estimators=12).fit(
        X, y, eval_set=[(Xv, yv)], eval_sample_weight=[w], eval_metric="l2"
    )
    pred = model.predict(Xv)
    assert model.evals_result_["valid_0"]["l2"][-1] == pytest.approx(
        np.average((pred - yv) ** 2, weights=w)
    )
    # A single vector is accepted for a single set, as is a list of one.
    nested = MojoTreesRegressor(n_estimators=12).fit(
        X, y, eval_set=[(Xv, yv)], eval_sample_weight=w, eval_metric="l2"
    )
    assert (
        nested.evals_result_["valid_0"]["l2"]
        == model.evals_result_["valid_0"]["l2"]
    )


def test_uniform_eval_weights_change_nothing(regression):
    X, y, Xv, yv = _split(*regression)
    plain = MojoTreesRegressor(n_estimators=10).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric="l1"
    )
    weighted = MojoTreesRegressor(n_estimators=10).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_sample_weight=[np.full(len(yv), 3.0)],
        eval_metric="l1",
    )
    assert weighted.evals_result_["valid_0"]["l1"] == pytest.approx(
        plain.evals_result_["valid_0"]["l1"]
    )


def test_bad_eval_sample_weight_raises(regression):
    X, y, Xv, yv = _split(*regression)
    with pytest.raises(ValueError, match="one entry per eval_set entry"):
        MojoTreesRegressor(n_estimators=5).fit(
            X,
            y,
            eval_set=[(Xv, yv), (Xv, yv)],
            eval_sample_weight=[np.ones(len(yv))],
        )
    with pytest.raises(ValueError, match="shape"):
        MojoTreesRegressor(n_estimators=5).fit(
            X, y, eval_set=[(Xv, yv)], eval_sample_weight=[np.ones(3)]
        )
    with pytest.raises(ValueError, match="nonnegative"):
        MojoTreesRegressor(n_estimators=5).fit(
            X, y, eval_set=[(Xv, yv)], eval_sample_weight=[-np.ones(len(yv))]
        )


def test_eval_weights_with_a_callable_metric_raise(regression):
    X, y, Xv, yv = _split(*regression)
    with pytest.raises(ValueError, match="callable metric"):
        MojoTreesRegressor(n_estimators=5).fit(
            X,
            y,
            eval_set=[(Xv, yv)],
            eval_metric=lambda t, p: 0.0,
            eval_sample_weight=[np.ones(len(yv))],
        )


# -- early stopping and the fitted attributes -----------------------------


def test_early_stopping_rolls_back_to_the_best_iteration(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=200).fit(
        X, y, eval_set=[(Xv, yv)], early_stopping_rounds=5
    )
    assert model.stopped_early_
    assert 0 < model.best_iteration_ < model.n_iter_ < 200
    history = model.evals_result_["valid_0"]["l2"]
    assert len(history) == model.n_iter_ + 1
    assert model.best_score_ == pytest.approx(min(history))
    assert history[model.best_iteration_] == pytest.approx(model.best_score_)
    # Rolling back is not bookkeeping: the kept ensemble is the one a
    # shorter fit would have produced.
    truncated = MojoTreesRegressor(
        n_estimators=model.best_iteration_
    ).fit(X, y)
    assert model.predict(Xv) == pytest.approx(truncated.predict(Xv))


def test_without_early_stopping_nothing_is_rolled_back(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=30).fit(
        X, y, eval_set=[(Xv, yv)], early_stopping_rounds=0
    )
    assert not model.stopped_early_
    assert model.n_iter_ == 30
    history = model.evals_result_["valid_0"]["l2"]
    assert len(history) == 31
    assert model.best_iteration_ == history.index(min(history))


def test_n_iter_without_validation(regression):
    X, y = regression
    model = MojoTreesRegressor(n_estimators=17).fit(X, y)
    assert model.n_iter_ == 17
    assert model.best_iteration_ == 17
    assert not hasattr(model, "evals_result_")


def test_refit_clears_the_validation_record(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=25)
    model.fit(X, y, eval_set=[(Xv, yv)], early_stopping_rounds=3)
    model.fit(X, y)
    assert not hasattr(model, "evals_result_")
    assert not hasattr(model, "best_score_")
    assert model.best_iteration_ == model.n_iter_ == 25


def test_ties_are_not_improvements(regression):
    """A metric that never moves stops after exactly the patience."""
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=50).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=("flat", lambda t, p: 1.0),
        early_stopping_rounds=4,
    )
    assert model.stopped_early_
    assert model.best_iteration_ == 0
    assert model.n_iter_ == 4
    assert model.best_score_ == 1.0


def test_min_delta_stops_sooner(regression):
    X, y, Xv, yv = _split(*regression)
    kwargs = dict(eval_set=[(Xv, yv)], early_stopping_rounds=3)
    patient = MojoTreesRegressor(n_estimators=200).fit(X, y, **kwargs)
    impatient = MojoTreesRegressor(n_estimators=200).fit(
        X, y, min_delta=0.05, **kwargs
    )
    assert impatient.n_iter_ < patient.n_iter_
    assert impatient.best_iteration_ <= patient.best_iteration_


def test_higher_is_better_metrics_select_the_maximum(binary):
    X, y, Xv, yv = _split(*binary)
    model = MojoTreesClassifier(n_estimators=40).fit(
        X, y, eval_set=[(Xv, yv)], eval_metric="auc", early_stopping_rounds=5
    )
    history = model.evals_result_["valid_0"]["auc"]
    assert model.best_score_ == pytest.approx(max(history))
    assert history[model.best_iteration_] == pytest.approx(model.best_score_)


def test_primary_metric_selects_the_round(binary):
    X, y, Xv, yv = _split(*binary)
    model = MojoTreesClassifier(n_estimators=40).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=["binary_logloss", "auc"],
        primary_metric="auc",
        early_stopping_rounds=5,
    )
    auc = model.evals_result_["valid_0"]["auc"]
    assert model.best_score_ == pytest.approx(max(auc))
    assert model.best_iteration_ == auc.index(max(auc))


def test_negative_early_stopping_rounds_raise(regression):
    X, y, Xv, yv = _split(*regression)
    with pytest.raises(ValueError, match="must not be negative"):
        MojoTreesRegressor(n_estimators=5).fit(
            X, y, eval_set=[(Xv, yv)], early_stopping_rounds=-1
        )


# -- classifiers ----------------------------------------------------------


def test_binary_metrics_match_an_independent_computation(binary):
    X, y, Xv, yv = _split(*binary)
    model = MojoTreesClassifier(n_estimators=20).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=["binary_logloss", "binary_error", "auc"],
    )
    proba = model.predict_proba(Xv)[:, 1]
    last = {k: v[-1] for k, v in model.evals_result_["valid_0"].items()}
    expected = -np.mean(
        yv * np.log(proba) + (1 - yv) * np.log(1.0 - proba)
    )
    assert last["binary_logloss"] == pytest.approx(expected, rel=1e-9)
    assert last["binary_error"] == pytest.approx(
        np.mean((proba >= 0.5) != (yv > 0.5))
    )
    order = np.argsort(proba)
    ranks = np.empty(len(proba))
    ranks[order] = np.arange(1, len(proba) + 1)
    n_pos = int(yv.sum())
    n_neg = len(yv) - n_pos
    auc = (ranks[yv == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    assert last["auc"] == pytest.approx(auc)


def test_multiclass_validation(multiclass):
    X, y, Xv, yv = _split(*multiclass, n_valid=80)
    model = MojoTreesClassifier(n_estimators=40).fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        eval_metric=["multi_logloss", "multi_error"],
        early_stopping_rounds=5,
    )
    assert model.n_classes_ == 3
    proba = model.predict_proba(Xv)
    history = model.evals_result_["valid_0"]
    expected = -np.mean(np.log(proba[np.arange(len(yv)), yv]))
    assert history["multi_logloss"][model.best_iteration_] == pytest.approx(
        expected, rel=1e-9
    )
    assert history["multi_error"][model.best_iteration_] == pytest.approx(
        np.mean(model.predict(Xv) != yv)
    )


def test_multiclass_rollback_matches_a_shorter_fit(multiclass):
    X, y, Xv, yv = _split(*multiclass, n_valid=80)
    model = MojoTreesClassifier(n_estimators=60).fit(
        X, y, eval_set=[(Xv, yv)], early_stopping_rounds=4
    )
    assert model.best_iteration_ > 0
    truncated = MojoTreesClassifier(n_estimators=model.best_iteration_).fit(
        X, y
    )
    assert model.predict_proba(Xv) == pytest.approx(
        truncated.predict_proba(Xv)
    )


def test_string_labels_are_encoded_for_validation(binary):
    X, y, Xv, yv = _split(*binary)
    names = np.array(["no", "yes"])
    model = MojoTreesClassifier(n_estimators=10).fit(
        X, names[y], eval_set=[(Xv, names[yv])]
    )
    plain = MojoTreesClassifier(n_estimators=10).fit(
        X, y, eval_set=[(Xv, yv)]
    )
    assert (
        model.evals_result_["valid_0"]["binary_logloss"]
        == plain.evals_result_["valid_0"]["binary_logloss"]
    )


def test_unseen_validation_label_raises(binary):
    X, y, Xv, yv = _split(*binary)
    stranger = yv.copy()
    stranger[0] = 7
    with pytest.raises(ValueError, match="not one of the classes"):
        MojoTreesClassifier(n_estimators=5).fit(
            X, y, eval_set=[(Xv, stranger)]
        )


# -- ranker ---------------------------------------------------------------


def test_ranker_validation_scores_ndcg():
    X, y, group = _ranking_data()
    Xv, yv, gv = _ranking_data(n_queries=12, seed=9)
    model = MojoTreesRanker(n_estimators=30, ndcg_eval_at=3).fit(
        X,
        y,
        group=group,
        eval_set=[(Xv, yv)],
        eval_group=[gv],
        early_stopping_rounds=5,
    )
    history = model.evals_result_["valid_0"]["ndcg"]
    assert model.best_score_ == pytest.approx(max(history))
    # The kept ensemble is the best round, so scoring it reproduces the
    # value the history recorded there, at the estimator's own cutoff.
    assert ndcg_score(model.predict(Xv), yv, gv, at=3) == pytest.approx(
        history[model.best_iteration_]
    )


def test_ranker_eval_group_is_required():
    X, y, group = _ranking_data(n_queries=10)
    Xv, yv, gv = _ranking_data(n_queries=4, seed=9)
    with pytest.raises(ValueError, match="eval_set needs eval_group"):
        MojoTreesRanker(n_estimators=5).fit(
            X, y, group=group, eval_set=[(Xv, yv)]
        )
    with pytest.raises(ValueError, match="eval_group needs an eval_set"):
        MojoTreesRanker(n_estimators=5).fit(
            X, y, group=group, eval_group=[gv]
        )


def test_ranker_eval_group_must_cover_its_rows():
    X, y, group = _ranking_data(n_queries=10)
    Xv, yv, gv = _ranking_data(n_queries=4, seed=9)
    with pytest.raises(ValueError, match="group counts sum to"):
        MojoTreesRanker(n_estimators=5).fit(
            X, y, group=group, eval_set=[(Xv, yv)], eval_group=[gv[:-1]]
        )


def test_ranker_rejects_eval_sample_weight():
    X, y, group = _ranking_data(n_queries=10)
    Xv, yv, gv = _ranking_data(n_queries=4, seed=9)
    with pytest.raises(ValueError, match="no weighted definition"):
        MojoTreesRanker(n_estimators=5).fit(
            X,
            y,
            group=group,
            eval_set=[(Xv, yv)],
            eval_group=[gv],
            eval_sample_weight=[np.ones(len(yv))],
        )


def test_ranker_validation_relevance_is_checked():
    X, y, group = _ranking_data(n_queries=10)
    Xv, yv, gv = _ranking_data(n_queries=4, seed=9)
    bad = yv.astype(np.float64)
    bad[0] = 31.0
    with pytest.raises(ValueError, match="relevance labels"):
        MojoTreesRanker(n_estimators=5).fit(
            X, y, group=group, eval_set=[(Xv, bad)], eval_group=[gv]
        )


# -- devices and serialization -------------------------------------------


def test_auto_device_matches_cpu(regression):
    X, y, Xv, yv = _split(*regression)
    kwargs = dict(eval_set=[(Xv, yv)], early_stopping_rounds=5)
    cpu = MojoTreesRegressor(n_estimators=40, device="cpu").fit(X, y, **kwargs)
    auto = MojoTreesRegressor(n_estimators=40, device="auto").fit(
        X, y, **kwargs
    )
    assert auto.device_ == "cpu"
    assert auto.best_iteration_ == cpu.best_iteration_
    assert auto.evals_result_ == cpu.evals_result_
    assert auto.predict(Xv) == pytest.approx(cpu.predict(Xv))


def test_gpu_with_an_eval_set_raises_rather_than_falling_back(regression):
    """Validation is scored on the CPU. `device="gpu"` never silently uses
    another backend, so it says so instead."""
    X, y, Xv, yv = _split(*regression)
    with pytest.raises(RuntimeError):
        MojoTreesRegressor(n_estimators=5, device="gpu").fit(
            X, y, eval_set=[(Xv, yv)]
        )


def test_gpu_availability_is_reported():
    assert isinstance(gpu_available(), bool)


def test_validation_record_survives_pickling(regression):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=40).fit(
        X, y, eval_set=[(Xv, yv)], early_stopping_rounds=5
    )
    revived = pickle.loads(pickle.dumps(model))
    assert revived.evals_result_ == model.evals_result_
    assert revived.best_iteration_ == model.best_iteration_
    assert revived.best_score_ == model.best_score_
    assert revived.n_iter_ == model.n_iter_
    assert revived.stopped_early_ == model.stopped_early_
    assert revived.predict(Xv) == pytest.approx(model.predict(Xv))


def test_saved_model_is_the_rolled_back_one(regression, tmp_path):
    X, y, Xv, yv = _split(*regression)
    model = MojoTreesRegressor(n_estimators=40).fit(
        X, y, eval_set=[(Xv, yv)], early_stopping_rounds=5
    )
    path = tmp_path / "model.mbst"
    model.save(path)
    loaded = MojoTreesRegressor.load(path)
    assert loaded.best_iteration_ == model.best_iteration_
    assert loaded.predict(Xv) == pytest.approx(model.predict(Xv))
