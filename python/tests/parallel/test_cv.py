"""Cross-validation: `mojoboost.cv.cv` and `CVBooster`.

The load-bearing tests here are the leakage ones and the equivalence one.
`cv()` builds each fold's data from the raw matrix so every fold bins itself
over its own rows, and it grows each fold a round at a time so the history
has a row per round; both claims are checked rather than assumed. The rest
covers the fold generators, the metric plumbing, the callback forwarding,
and the stopping consensus.
"""

import pytest

np = pytest.importorskip("numpy")

import mojoboost
from mojoboost import Dataset, train
from mojoboost import _eval
from mojoboost.cv import CVBooster, cv
from mojoboost.cv import _fold_dataset, _generated_splits, _query_of_row


REG = {"objective": "regression", "num_leaves": 7, "learning_rate": 0.2}
BIN = {"objective": "binary", "num_leaves": 7, "learning_rate": 0.2}


def _ranking_data(n_queries=12, per_query=6, n_features=4, seed=3):
    gen = np.random.default_rng(seed)
    n_rows = n_queries * per_query
    X = gen.random((n_rows, n_features))
    score = 2.0 * X[:, 0] + X[:, 1]
    y = np.zeros(n_rows, dtype=np.int64)
    for q in range(n_queries):
        lo = q * per_query
        block = score[lo : lo + per_query]
        y[lo : lo + per_query] = np.argsort(np.argsort(block)) // 2
    return X, y, [per_query] * n_queries


def _weak_signal(n_rows=120, seed=13):
    """Almost all noise, so the held-out loss bottoms out within a handful
    of rounds. Early stopping needs something it can actually stop on."""
    gen = np.random.default_rng(seed)
    X = gen.random((n_rows, 4))
    y = 0.3 * X[:, 0] + gen.standard_normal(n_rows)
    return X, y


#: Overfits fast, which is what gives the stopping rule something to see.
OVERFIT = {
    "objective": "regression",
    "num_leaves": 15,
    "learning_rate": 0.3,
    "min_data_in_leaf": 2,
}


def _imbalanced(n_rows=120, seed=5):
    gen = np.random.default_rng(seed)
    X = gen.random((n_rows, 3))
    y = np.zeros(n_rows, dtype=np.int64)
    y[:9] = 1  # one rare class, fewer members than some fold counts
    return X, y


# -- the history ---------------------------------------------------------


def test_history_has_a_row_per_round(regression):
    X, y = regression
    results = cv(REG, Dataset(X, label=y), num_boost_round=6, nfold=3)
    assert results["iterations"] == [1, 2, 3, 4, 5, 6]
    assert set(results) == {
        "iterations",
        "valid l2-mean",
        "valid l2-stdv",
    }
    assert len(results["valid l2-mean"]) == 6
    assert all(v >= 0.0 for v in results["valid l2-stdv"])
    # Six rounds of boosting on a learnable problem improve the held-out
    # loss; this is the one thing a cv history must get right.
    assert results["valid l2-mean"][-1] < results["valid l2-mean"][0]


def test_a_single_fold_matches_train_at_the_same_rounds(regression):
    """The incremental path is `train()`'s model.

    `cv()` grows each fold with `Booster.update(1)` so it can report every
    round. That is only sound if the ensemble after R such calls is the
    ensemble `train(..., R)` builds, so one explicit fold is compared with
    the booster `train()` returns on exactly those rows.
    """
    X, y = regression
    rows = list(range(len(y)))
    train_rows, test_rows = rows[:200], rows[200:]
    results = cv(
        REG,
        Dataset(X, label=y),
        num_boost_round=5,
        folds=[(train_rows, test_rows)],
    )
    dtrain = Dataset(X[train_rows], label=y[train_rows])
    dvalid = Dataset(X[test_rows], label=y[test_rows])
    booster = train(REG, dtrain, 5)
    expected = booster.eval(dvalid, "valid")[0][2]
    assert results["valid l2-mean"][-1] == pytest.approx(expected, rel=1e-9)
    # One fold has nothing to disagree with itself about.
    assert results["valid l2-stdv"] == [0.0] * 5


def test_train_metrics_are_reported_when_asked(regression):
    X, y = regression
    results = cv(
        REG,
        Dataset(X, label=y),
        num_boost_round=4,
        nfold=3,
        eval_train_metric=True,
    )
    assert "train l2-mean" in results and "valid l2-mean" in results
    # The model saw the training rows and did not see the others.
    assert results["train l2-mean"][-1] < results["valid l2-mean"][-1]


# -- fold generation -----------------------------------------------------


def test_shuffled_folds_are_deterministic_in_the_seed(regression):
    X, y = regression
    dataset = Dataset(X, label=y)
    same = [
        cv(REG, dataset, num_boost_round=3, nfold=3, seed=7)["valid l2-mean"]
        for _ in range(2)
    ]
    assert same[0] == same[1]
    other = cv(REG, dataset, num_boost_round=3, nfold=3, seed=8)
    assert other["valid l2-mean"] != same[0]


def test_generated_folds_partition_the_rows(regression):
    X, y = regression
    splits = _generated_splits(
        Dataset(X, label=y), _eval.REGRESSION, 4, False, True, 0
    )
    held = []
    for split in splits:
        assert set(split.train_rows).isdisjoint(split.test_rows)
        assert len(split.train_rows) + len(split.test_rows) == len(y)
        held.extend(split.test_rows)
    assert sorted(held) == list(range(len(y)))


def test_stratified_folds_keep_the_label_mix():
    X, y = _imbalanced()
    with pytest.warns(UserWarning, match="fewer than nfold"):
        splits = _generated_splits(
            Dataset(X, label=y), _eval.BINARY, 10, True, True, 0
        )
    rare = [sum(int(y[r]) for r in split.test_rows) for split in splits]
    # Nine positives dealt round-robin over ten folds: one each, and one
    # fold without. Pooling them into a single fold is the failure this
    # rules out.
    assert sorted(rare) == [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_unstratified_folds_may_pool_a_rare_label():
    X, y = _imbalanced()
    splits = _generated_splits(
        Dataset(X, label=y), _eval.BINARY, 10, False, False, 0
    )
    rare = [sum(int(y[r]) for r in split.test_rows) for split in splits]
    assert max(rare) > 1  # the contrast the stratified test is measuring


def test_explicit_folds_are_used_as_given(regression):
    X, y = regression
    folds = [
        (list(range(0, 100)), list(range(100, 150))),
        (list(range(100, 200)), list(range(0, 50))),
    ]
    results = cv(REG, Dataset(X, label=y), num_boost_round=2, folds=folds)
    assert len(results["valid l2-mean"]) == 2


def test_a_sklearn_splitter_is_asked_to_split(regression):
    model_selection = pytest.importorskip("sklearn.model_selection")
    X, y = regression
    splitter = model_selection.KFold(n_splits=3, shuffle=True, random_state=0)
    results = cv(REG, Dataset(X, label=y), num_boost_round=3, folds=splitter)
    assert results["iterations"] == [1, 2, 3]
    assert len(results["valid l2-mean"]) == 3


def test_folds_that_train_on_what_they_score_are_refused(regression):
    X, y = regression
    with pytest.raises(ValueError, match="not be out of sample"):
        cv(
            REG,
            Dataset(X, label=y),
            num_boost_round=2,
            folds=[([0, 1, 2, 3], [3, 4, 5])],
        )


def test_a_fold_left_empty_says_which_way_out():
    """Six rows in two classes cannot fill six stratified folds: each class
    reaches only the first three, and a fold with nothing held out has no
    score to contribute."""
    X = np.random.default_rng(1).random((6, 2))
    y = np.array([0, 0, 0, 1, 1, 1])
    with pytest.warns(UserWarning, match="fewer than nfold"):
        with pytest.raises(ValueError, match="use fewer folds"):
            cv(BIN, Dataset(X, label=y), num_boost_round=2, nfold=6)


def test_nfold_beyond_the_row_count_is_refused():
    X = np.arange(12.0).reshape(4, 3)
    y = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="at least that many rows"):
        cv(REG, Dataset(X, label=y), num_boost_round=2, nfold=5)


# -- leakage -------------------------------------------------------------


def test_each_fold_bins_itself():
    """A fold's bins come from its own rows.

    The held-out rows here are the only ones above 100, so a dataset binned
    over everything would place split points the training rows alone could
    never justify. Slicing a constructed dataset would carry those over;
    building the fold from the raw matrix does not.
    """
    X = np.concatenate(
        [np.linspace(0.0, 1.0, 40), np.linspace(500.0, 900.0, 10)]
    ).reshape(-1, 1)
    y = X[:, 0].copy()
    source = Dataset(X, label=y, params={"max_bin": 16})
    fold = _fold_dataset(source, list(range(40)), None)
    assert fold.num_data() == 40
    assert float(np.max(fold.get_data())) <= 1.0
    # The source is untouched, and neither dataset was constructed by the
    # other's construction.
    assert source.num_data() == 50


def test_a_fold_carries_every_column_the_source_had(regression):
    """A fold is the source dataset restricted to its rows.

    The row-shaped columns are sliced with the matrix and the
    column-shaped declarations are copied whole, because those describe the
    features rather than the rows.
    """
    X, y = regression
    weight = np.linspace(0.5, 1.5, len(y))
    init = np.full(len(y), 0.25)
    source = Dataset(
        X,
        label=y,
        weight=weight,
        init_score=init,
        feature_name=["a", "b", "c", "d"],
        categorical_feature=["a"],
        params={"max_bin": 31},
    )
    rows = [3, 1, 7, 2]
    fold = _fold_dataset(source, rows, None)
    np.testing.assert_array_equal(fold.get_label(), y[rows])
    np.testing.assert_array_equal(fold.get_weight(), weight[rows])
    np.testing.assert_array_equal(fold.get_init_score(), init[rows])
    np.testing.assert_array_equal(np.asarray(fold.get_data()), X[rows])
    assert fold.feature_name == ["a", "b", "c", "d"]
    assert fold.categorical_feature == [0]
    assert fold.params == {"max_bin": 31, "use_missing": True}


def test_a_frame_keeps_its_column_names_through_the_folds(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    results = cv(
        REG, Dataset(frame, label=y), num_boost_round=2, nfold=2,
        return_cvbooster=True,
    )
    assert results["cvbooster"].feature_name() == [["a", "b", "c", "d"]] * 2


def test_fpreproc_sees_one_fold_at_a_time(regression):
    X, y = regression
    seen = []

    def fpreproc(dtrain, dvalid, params):
        assert isinstance(dtrain, Dataset) and isinstance(dvalid, Dataset)
        assert dtrain.num_data() + dvalid.num_data() == len(y)
        seen.append((dtrain.num_data(), dvalid.num_data()))
        return dtrain, dvalid, params

    cv(REG, Dataset(X, label=y), num_boost_round=2, nfold=3, fpreproc=fpreproc)
    assert len(seen) == 3
    assert sum(valid for _, valid in seen) == len(y)


def test_fpreproc_must_return_the_triple(regression):
    X, y = regression
    with pytest.raises(ValueError, match="must return"):
        cv(
            REG,
            Dataset(X, label=y),
            num_boost_round=2,
            nfold=2,
            fpreproc=lambda dtrain, dvalid, params: dtrain,
        )


def test_a_freed_dataset_cannot_be_split(regression):
    X, y = regression
    dataset = Dataset(X, label=y, free_raw_data=True).construct()
    with pytest.raises(ValueError, match="free_raw_data=False"):
        cv(REG, dataset, num_boost_round=2, nfold=2)


# -- metrics -------------------------------------------------------------


def test_several_metrics_are_scored_together(regression):
    X, y = regression
    results = cv(
        REG, Dataset(X, label=y), num_boost_round=3, nfold=3,
        metrics=("l2", "l1"),
    )
    for key in ("valid l2-mean", "valid l1-mean", "valid l2-stdv"):
        assert len(results[key]) == 3
    assert results["valid l2-mean"] != results["valid l1-mean"]


def test_a_custom_metric_is_handed_raw_scores(regression):
    X, y = regression
    calls = []

    def mean_bias(y_true, y_pred):
        calls.append(len(y_true))
        return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))

    results = cv(
        REG,
        Dataset(X, label=y),
        num_boost_round=3,
        nfold=2,
        metrics="l2",
        feval=("bias", mean_bias, False),
    )
    assert len(results["valid bias-mean"]) == 3
    assert len(calls) == 6  # two folds, three rounds
    assert all(n == len(y) // 2 for n in calls)


def test_a_binary_run_scores_its_own_metric(binary):
    X, y = binary
    results = cv(BIN, Dataset(X, label=y), num_boost_round=4, nfold=3)
    history = results["valid binary_logloss-mean"]
    assert len(history) == 4
    assert history[-1] < history[0]


def test_a_multiclass_run_scores_its_own_metric(multiclass):
    X, y = multiclass
    results = cv(
        {"objective": "multiclass", "num_class": 3, "num_leaves": 7},
        Dataset(X, label=y),
        num_boost_round=3,
        nfold=3,
    )
    assert len(results["valid multi_logloss-mean"]) == 3


def test_a_regression_metric_on_a_binary_run_is_refused(binary):
    X, y = binary
    with pytest.raises(ValueError, match="scores regression models"):
        cv(BIN, Dataset(X, label=y), num_boost_round=2, nfold=2, metrics="l2")


# -- ranking -------------------------------------------------------------


def test_ranking_folds_take_whole_queries():
    X, y, group = _ranking_data()
    splits = _generated_splits(
        Dataset(X, label=y, group=group), _eval.RANKING, 3, True, True, 0
    )
    owner = _query_of_row(group)
    for split in splits:
        assert sum(split.test_group) == len(split.test_rows)
        assert sum(split.train_group) == len(split.train_rows)
        train_queries = {owner[r] for r in split.train_rows}
        test_queries = {owner[r] for r in split.test_rows}
        assert train_queries.isdisjoint(test_queries)
        assert len(train_queries) + len(test_queries) == len(group)


def test_a_ranking_cv_reports_its_final_round():
    X, y, group = _ranking_data()
    results = cv(
        {"objective": "lambdarank", "num_leaves": 7},
        Dataset(X, label=y, group=group),
        num_boost_round=5,
        nfold=3,
    )
    assert results["iterations"] == [5]
    assert len(results["valid ndcg-mean"]) == 1


def test_a_ranking_fold_that_splits_a_query_is_refused():
    X, y, group = _ranking_data(n_queries=4, per_query=5)
    dataset = Dataset(X, label=y, group=group)
    with pytest.raises(ValueError, match="splits query"):
        cv(
            {"objective": "lambdarank"},
            dataset,
            num_boost_round=2,
            folds=[(list(range(0, 12)), list(range(12, 20)))],
        )


def test_ranking_refuses_the_knobs_it_cannot_honor():
    X, y, group = _ranking_data()
    dataset = Dataset(X, label=y, group=group)
    params = {"objective": "lambdarank"}
    with pytest.raises(NotImplementedError, match="per-round history"):
        cv(params, dataset, num_boost_round=4, nfold=3,
           early_stopping_rounds=2)
    with pytest.raises(NotImplementedError, match="per-round history"):
        cv(params, dataset, num_boost_round=4, nfold=3,
           callbacks=[mojoboost.log_evaluation(period=0)])


# -- callbacks -----------------------------------------------------------


def test_callbacks_see_the_across_fold_means(regression):
    X, y = regression
    history = {}
    seen = []

    def spy(env):
        seen.append((env.iteration, env.end_iteration,
                     list(env.evaluation_result_list)))

    results = cv(
        REG,
        Dataset(X, label=y),
        num_boost_round=4,
        nfold=3,
        callbacks=[mojoboost.record_evaluation(history), spy],
    )
    assert [iteration for iteration, _, _ in seen] == [0, 1, 2, 3]
    assert all(end == 4 for _, end, _ in seen)
    assert seen[0][2] == [("cv_agg", "valid l2", results["valid l2-mean"][0],
                           False)]
    assert history["cv_agg"]["valid l2"] == results["valid l2-mean"]


def test_a_before_iteration_callback_runs_before_the_round(regression):
    X, y = regression
    order = []

    def before(env):
        order.append(
            ("before", env.iteration, len(env.evaluation_result_list))
        )

    before.before_iteration = True

    def after(env):
        order.append(
            ("after", env.iteration, len(env.evaluation_result_list))
        )

    cv(REG, Dataset(X, label=y), num_boost_round=2, nfold=2,
       callbacks=[before, after])
    assert order == [
        ("before", 0, 0),
        ("after", 0, 1),
        ("before", 1, 0),
        ("after", 1, 1),
    ]


def test_a_callback_that_stops_truncates_the_history(regression):
    X, y = regression

    def stop_at_two(env):
        if env.iteration == 2:
            raise mojoboost.EarlyStopException(1, 0.0)

    results = cv(REG, Dataset(X, label=y), num_boost_round=8, nfold=2,
                 callbacks=[stop_at_two])
    assert results["iterations"] == [1, 2]


def test_reset_parameter_is_refused_rather_than_ignored(regression):
    X, y = regression
    with pytest.raises(NotImplementedError, match="reset_parameter"):
        cv(
            REG,
            Dataset(X, label=y),
            num_boost_round=3,
            nfold=2,
            callbacks=[mojoboost.reset_parameter(learning_rate=[0.1] * 3)],
        )


# -- early stopping ------------------------------------------------------


def test_early_stopping_reaches_one_consensus_round():
    X, y = _weak_signal()
    results = cv(
        OVERFIT,
        Dataset(X, label=y),
        num_boost_round=60,
        nfold=3,
        early_stopping_rounds=3,
        return_cvbooster=True,
    )
    rounds = len(results["valid l2-mean"])
    assert rounds < 60
    booster = results["cvbooster"]
    assert booster.best_iteration == results["iterations"][-1] == rounds
    # The truncation keeps the winning round, which is therefore the best
    # one in what is returned.
    assert results["valid l2-mean"][-1] == min(results["valid l2-mean"])


def test_early_stopping_can_watch_the_first_metric_only():
    X, y = _weak_signal()
    results = cv(
        OVERFIT,
        Dataset(X, label=y),
        num_boost_round=40,
        nfold=3,
        metrics=("l2", "l1"),
        early_stopping_rounds=3,
        first_metric_only=True,
    )
    assert results["valid l2-mean"][-1] == min(results["valid l2-mean"])
    assert len(results["valid l1-mean"]) == len(results["valid l2-mean"])


def test_early_stopping_reads_a_higher_is_better_metric_the_right_way(
    regression,
):
    """A metric that improves upward, on a curve chosen rather than hoped for.

    A direction the rule reads backwards stops on the first round every time
    and still looks plausible, and a real metric that happens to peak on
    round 1 cannot tell the two apart. So the curve here is scripted: it
    rises to a peak on round 3 and falls after, and patience of 2 puts the
    stop on round 5 with round 3 the winner. Read backwards, the same run
    would stop on round 3 and keep round 1.
    """
    X, y = regression
    curve = [0.1, 0.4, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1]
    calls = []

    def scripted(y_true, y_pred):
        value = curve[len(calls) // 2]  # two folds, so two calls per round
        calls.append(value)
        return value

    results = cv(
        REG,
        Dataset(X, label=y),
        num_boost_round=8,
        nfold=2,
        metrics=("scripted", scripted, True),
        early_stopping_rounds=2,
    )
    assert results["iterations"] == [1, 2, 3]
    assert results["valid scripted-mean"] == [0.1, 0.4, 0.9]


def test_a_higher_is_better_metric_is_reported(binary):
    X, y = binary
    results = cv(BIN, Dataset(X, label=y), num_boost_round=3, nfold=3,
                 metrics="auc")
    history = results["valid auc-mean"]
    assert len(history) == 3
    assert all(0.0 <= value <= 1.0 for value in history)


def test_min_delta_raises_the_bar_an_improvement_must_clear(regression):
    X, y = regression
    results = cv(
        REG,
        Dataset(X, label=y),
        num_boost_round=20,
        nfold=2,
        early_stopping_rounds=2,
        min_delta=1e6,
    )
    # Nothing can improve by a million, so the first round is the best one
    # and patience runs out two rounds later.
    assert results["iterations"] == [1]


def test_a_metric_can_opt_out_of_early_stopping(regression):
    X, y = regression
    with pytest.raises(ValueError, match="no metric to watch"):
        cv(
            REG,
            Dataset(X, label=y),
            num_boost_round=10,
            nfold=2,
            metrics=[("l2", lambda a, b: 0.0, False, False)],
            early_stopping_rounds=2,
        )


def test_the_early_stopping_callback_and_the_argument_are_one_knob(regression):
    X, y = regression
    with pytest.raises(ValueError, match="not both"):
        cv(
            REG,
            Dataset(X, label=y),
            num_boost_round=10,
            nfold=2,
            early_stopping_rounds=3,
            callbacks=[mojoboost.early_stopping(3, verbose=False)],
        )


# -- CVBooster -----------------------------------------------------------


def test_cvbooster_holds_one_model_per_fold(regression):
    X, y = regression
    results = cv(REG, Dataset(X, label=y), num_boost_round=4, nfold=3,
                 return_cvbooster=True)
    booster = results["cvbooster"]
    assert isinstance(booster, CVBooster)
    assert len(booster) == 3
    assert [b.current_iteration() for b in booster] == [4, 4, 4]
    assert booster.fold_names == ["fold_0", "fold_1", "fold_2"]
    assert booster.best_iteration == -1
    assert "3 folds" in repr(booster)


def test_cvbooster_forwards_what_it_is_asked(regression):
    X, y = regression
    booster = cv(REG, Dataset(X, label=y), num_boost_round=3, nfold=2,
                 return_cvbooster=True)["cvbooster"]
    assert booster.num_trees() == [3, 3]
    assert booster.num_feature() == [X.shape[1], X.shape[1]]
    predictions = booster.predict(X)
    assert len(predictions) == 2
    assert all(len(p) == len(y) for p in predictions)
    with pytest.raises(AttributeError):
        booster.no_such_method


def test_cvbooster_predicts_through_the_best_iteration():
    X, y = _weak_signal()
    results = cv(
        OVERFIT,
        Dataset(X, label=y),
        num_boost_round=60,
        nfold=3,
        early_stopping_rounds=3,
        return_cvbooster=True,
    )
    booster = results["cvbooster"]
    best = booster.best_iteration
    assert 0 < best < 60
    np.testing.assert_allclose(
        booster.predict(X)[0], booster[0].predict(X, num_iteration=best)
    )


def test_the_cvbooster_is_only_returned_when_asked(regression):
    X, y = regression
    results = cv(REG, Dataset(X, label=y), num_boost_round=2, nfold=2)
    assert "cvbooster" not in results


# -- parameters ----------------------------------------------------------


def test_round_aliases_reach_the_folds(regression):
    X, y = regression
    results = cv(
        dict(REG, n_estimators=3), Dataset(X, label=y), nfold=2,
        return_cvbooster=True,
    )
    assert results["iterations"] == [1, 2, 3]
    assert results["cvbooster"].num_trees() == [3, 3]


def test_a_dataset_parameter_in_params_is_refused(regression):
    X, y = regression
    with pytest.raises(ValueError, match="describes the data"):
        cv(dict(REG, max_bin=31), Dataset(X, label=y), num_boost_round=2)


def test_a_labelless_dataset_is_refused(regression):
    X, _ = regression
    with pytest.raises(ValueError, match="needs a Dataset with a label"):
        cv(REG, Dataset(X), num_boost_round=2)


def test_init_model_is_refused_with_its_reason(regression):
    """`init_model` cannot reach a fold, and says so once.

    Continued training checks that the dataset is the one the model was
    trained on, by comparing the binning; a fold bins itself over its own
    rows, so the trainer refuses it. `cv()` reports that up front rather
    than letting the same refusal arrive once per fold from the bindings.
    """
    X, y = regression
    dataset = Dataset(X, label=y)
    start = train(REG, dataset, 4)
    with pytest.raises(NotImplementedError, match="init_model"):
        cv(REG, dataset, num_boost_round=3, nfold=2, init_model=start)
