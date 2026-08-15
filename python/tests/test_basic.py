"""The functional API: Dataset, Booster, and train().

The load-bearing tests here are the equivalence ones. `train()` and the
estimators are two doors into one trainer, so a model trained through either
must be the same model, bit for bit; and an estimator's `booster_` must be
the object that holds its model rather than a copy of it. The rest covers
the dataset's ownership rules, the booster's lifecycle, and what survives a
pickle.
"""

import gc
import pickle

import pytest

np = pytest.importorskip("numpy")

import mojotrees
from mojotrees import Booster, Dataset, MojoTreesClassifier
from mojotrees import MojoTreesRanker, MojoTreesRegressor, train


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


# -- Dataset ------------------------------------------------------------


def test_dataset_describes_itself(regression):
    X, y = regression
    ds = Dataset(X, label=y, params={"max_bin": 63})
    assert ds.num_data() == X.shape[0]
    assert ds.num_feature() == X.shape[1]
    assert ds.feature_name == [f"Column_{i}" for i in range(X.shape[1])]
    np.testing.assert_array_equal(ds.get_label(), y)
    assert ds.get_weight() is None
    assert ds.get_group() is None
    assert ds.get_init_score() is None
    assert ds.get_data() is X
    # Binning is lazy, and the bin count is a property of fitted bins.
    assert ds.num_bin() <= 63


def test_dataset_keeps_its_fields(regression):
    X, y = regression
    w = np.linspace(0.5, 1.5, len(y))
    init = np.full(len(y), 0.25)
    ds = Dataset(X, label=y, weight=w, init_score=init)
    np.testing.assert_array_equal(ds.get_weight(), w)
    np.testing.assert_array_equal(ds.get_init_score(), init)
    np.testing.assert_array_equal(ds.get_field("weight"), w)
    with pytest.raises(ValueError, match="unknown field"):
        ds.get_field("position")


def test_dataset_validates_its_columns(regression):
    X, y = regression
    with pytest.raises(ValueError):
        Dataset(X, label=y[:-1])
    with pytest.raises(ValueError):
        Dataset(X, label=y, weight=-np.ones(len(y)))
    with pytest.raises(ValueError, match="group counts"):
        Dataset(X, label=y, group=[3, 4])
    with pytest.raises(ValueError, match="feature_name"):
        Dataset(X, label=y, feature_name=["a", "b"])
    with pytest.raises(ValueError, match="out of range"):
        Dataset(X, label=y, categorical_feature=[99])


def test_dataset_rejects_training_parameters(regression):
    X, y = regression
    with pytest.raises(ValueError, match="belong to train"):
        Dataset(X, label=y, params={"num_leaves": 7})


def test_dataset_names_resolve_categorical_features(regression):
    X, y = regression
    names = [f"f{i}" for i in range(X.shape[1])]
    ds = Dataset(X, label=y, feature_name=names, categorical_feature=["f2"])
    assert ds.feature_name == names
    assert ds.categorical_feature == [2]


def test_dataset_reference_must_agree_on_binning(regression):
    X, y = regression
    train_set = Dataset(X, label=y, params={"max_bin": 31})
    # A reference with the same binning is accepted and changes nothing.
    Dataset(X, label=y, params={"max_bin": 31}, reference=train_set)
    with pytest.raises(ValueError, match="binning parameters"):
        Dataset(X, label=y, params={"max_bin": 255}, reference=train_set)


def test_free_raw_data_drops_the_matrix_on_construct(regression):
    X, y = regression
    ds = Dataset(X, label=y, free_raw_data=True)
    assert ds.get_data() is X
    ds.construct()
    assert ds.get_data() is None
    # The binning is still there, and so is everything derived from it.
    assert ds.num_data() == X.shape[0]
    booster = train({"objective": "regression"}, ds, 5)
    assert booster.current_iteration() == 5
    with pytest.raises(ValueError, match="freed"):
        booster.predict(ds)


def test_construct_is_idempotent(regression):
    X, y = regression
    ds = Dataset(X, label=y)
    assert ds.construct() is ds
    first = ds._handle
    ds.construct()
    assert ds._handle is first


# -- train() ------------------------------------------------------------


def test_train_matches_the_regressor(regression):
    """The estimator and train() are two doors into one trainer."""
    X, y = regression
    params = {"objective": "regression", "num_leaves": 8, "max_depth": 4}
    booster = train(params, Dataset(X, label=y), num_boost_round=20)
    estimator = MojoTreesRegressor(
        num_leaves=8, max_depth=4, n_estimators=20
    ).fit(X, y)
    np.testing.assert_array_equal(booster.predict(X), estimator.predict(X))
    assert booster.current_iteration() == estimator.best_iteration_


def test_train_matches_the_binary_classifier(binary):
    X, y = binary
    booster = train(
        {"objective": "binary", "num_leaves": 8}, Dataset(X, label=y), 15
    )
    estimator = MojoTreesClassifier(num_leaves=8, n_estimators=15).fit(X, y)
    np.testing.assert_array_equal(
        booster.predict(X), estimator.predict_proba(X)[:, 1]
    )


def test_train_matches_the_multiclass_classifier(multiclass):
    X, y = multiclass
    booster = train(
        {"objective": "multiclass", "num_class": 3, "num_leaves": 8},
        Dataset(X, label=y),
        12,
    )
    estimator = MojoTreesClassifier(num_leaves=8, n_estimators=12).fit(X, y)
    np.testing.assert_array_equal(
        booster.predict(X), estimator.predict_proba(X)
    )
    assert booster.num_model_per_iteration() == 3
    assert booster.num_trees() == 12 * 3


def test_train_matches_the_ranker():
    X, y, group = _ranking_data()
    booster = train(
        {"objective": "lambdarank", "num_leaves": 8},
        Dataset(X, label=y, group=group),
        10,
    )
    estimator = MojoTreesRanker(num_leaves=8, n_estimators=10).fit(
        X, y, group=group
    )
    np.testing.assert_array_equal(booster.predict(X), estimator.predict(X))


def test_train_reads_lightgbm_round_aliases(regression):
    X, y = regression
    booster = train(
        {"objective": "regression", "num_iterations": 7}, Dataset(X, label=y)
    )
    assert booster.current_iteration() == 7
    with pytest.raises(ValueError, match="disagree"):
        train(
            {"objective": "regression", "num_iterations": 7},
            Dataset(X, label=y),
            num_boost_round=9,
        )


def test_train_rejects_data_parameters(regression):
    X, y = regression
    with pytest.raises(ValueError, match="describes the data"):
        train(
            {"objective": "regression", "max_bin": 31}, Dataset(X, label=y), 5
        )


def test_train_rejects_unknown_parameters(regression):
    X, y = regression
    with pytest.raises(ValueError, match="unknown or unsupported"):
        train({"objective": "regression", "nonesuch": 1}, Dataset(X, label=y))


def test_train_needs_a_labelled_dataset(regression):
    X, _ = regression
    with pytest.raises(ValueError, match="needs a Dataset with a label"):
        train({"objective": "regression"}, Dataset(X), 5)


def test_binary_objective_checks_its_labels(multiclass):
    X, y = multiclass
    with pytest.raises(ValueError, match=r"labels in \{0, 1\}"):
        train({"objective": "binary"}, Dataset(X, label=y), 5)


def test_multiclass_needs_num_class(multiclass):
    X, y = multiclass
    with pytest.raises(ValueError, match="num_class"):
        train({"objective": "multiclass"}, Dataset(X, label=y), 5)


def test_lambdarank_needs_group():
    X, y, _ = _ranking_data()
    with pytest.raises(ValueError, match="needs a Dataset with group"):
        train({"objective": "lambdarank"}, Dataset(X, label=y), 5)


def test_dataset_is_reusable(regression):
    """One binning, two models: the point of a Dataset."""
    X, y = regression
    ds = Dataset(X, label=y)
    shallow = train({"objective": "regression", "num_leaves": 4}, ds, 10)
    deep = train({"objective": "regression", "num_leaves": 31}, ds, 10)
    assert ds._handle is not None
    assert not np.array_equal(shallow.predict(X), deep.predict(X))


# -- Booster ------------------------------------------------------------


def test_booster_starts_empty_and_grows(regression):
    X, y = regression
    ds = Dataset(X, label=y)
    booster = Booster({"objective": "regression"}, ds)
    assert booster.current_iteration() == 0
    assert booster.update() is False
    assert booster.current_iteration() == 1
    booster.update(4)
    assert booster.current_iteration() == 5


def test_continued_training_equals_one_run(regression):
    """40 rounds then 60 more are the 100-round model."""
    X, y = regression
    params = {"objective": "regression", "num_leaves": 8}
    whole = train(params, Dataset(X, label=y), 100)
    staged = train(params, Dataset(X, label=y), 40)
    staged.update(60)
    assert staged.current_iteration() == whole.current_iteration()
    np.testing.assert_array_equal(staged.predict(X), whole.predict(X))


def test_continued_multiclass_training_equals_one_run(multiclass):
    X, y = multiclass
    params = {"objective": "multiclass", "num_class": 3, "num_leaves": 8}
    whole = train(params, Dataset(X, label=y), 20)
    staged = train(params, Dataset(X, label=y), 8)
    staged.update(12)
    np.testing.assert_array_equal(staged.predict(X), whole.predict(X))


def test_init_model_leaves_the_original_alone(regression):
    X, y = regression
    params = {"objective": "regression", "num_leaves": 8}
    ds = Dataset(X, label=y)
    first = train(params, ds, 40)
    first_predictions = first.predict(X)
    second = train(params, ds, 60, init_model=first)
    assert first.current_iteration() == 40
    np.testing.assert_array_equal(first.predict(X), first_predictions)
    whole = train(params, ds, 100)
    np.testing.assert_array_equal(second.predict(X), whole.predict(X))


def test_update_needs_a_training_set(regression):
    X, y = regression
    booster = train({"objective": "regression"}, Dataset(X, label=y), 5)
    revived = Booster(model_str=booster.model_to_string())
    with pytest.raises(ValueError, match="no training set"):
        revived.update()


def test_ranking_cannot_be_continued():
    X, y, group = _ranking_data()
    booster = train(
        {"objective": "lambdarank"}, Dataset(X, label=y, group=group), 5
    )
    with pytest.raises(NotImplementedError, match="ranking"):
        booster.update()


def test_booster_keeps_its_dataset_alive(regression):
    """The training set outlives the caller's reference to it, because the
    booster that grows on it holds one."""
    X, y = regression
    booster = train({"objective": "regression"}, Dataset(X, label=y), 5)
    gc.collect()
    booster.update(5)
    assert booster.current_iteration() == 10


def test_predict_raw_and_response_agree(binary):
    X, y = binary
    booster = train({"objective": "binary"}, Dataset(X, label=y), 10)
    raw = booster.predict(X, raw_score=True)
    proba = booster.predict(X)
    np.testing.assert_allclose(proba, 1.0 / (1.0 + np.exp(-raw)), rtol=1e-12)


def test_predict_slices_iterations(regression):
    X, y = regression
    ds = Dataset(X, label=y)
    booster = train({"objective": "regression"}, ds, 20)
    short = train({"objective": "regression"}, ds, 5)
    np.testing.assert_array_equal(
        booster.predict(X, num_iteration=5), short.predict(X)
    )
    # LightGBM clamps rather than raising, and the halves sum to the whole.
    head = booster.predict(X, num_iteration=8, raw_score=True)
    tail = booster.predict(X, start_iteration=8, raw_score=True)
    whole = booster.predict(X, num_iteration=500, raw_score=True)
    np.testing.assert_allclose(head + tail, whole, rtol=1e-12)


def test_predict_takes_a_dataset(regression):
    X, y = regression
    ds = Dataset(X, label=y)
    booster = train({"objective": "regression"}, ds, 5)
    np.testing.assert_array_equal(booster.predict(ds), booster.predict(X))


def test_predict_checks_the_feature_count(regression):
    X, y = regression
    booster = train({"objective": "regression"}, Dataset(X, label=y), 5)
    with pytest.raises(ValueError, match="features"):
        booster.predict(X[:, :2])


def test_eval_scores_the_objectives_own_metric(regression):
    X, y = regression
    ds = Dataset(X, label=y)
    booster = train({"objective": "regression"}, ds, 20)
    result = booster.eval(ds, "training")
    assert len(result) == 1
    name, metric, value, higher = result[0]
    assert (name, metric, higher) == ("training", "l2", False)
    expected = float(np.mean((booster.predict(X) - y) ** 2))
    assert value == pytest.approx(expected, rel=1e-12)


def test_eval_train_and_eval_valid(regression):
    X, y = regression
    ds = Dataset(X, label=y)
    booster = train({"objective": "regression"}, ds, 10)
    holdout = Dataset(X[:50], label=y[:50])
    booster.add_valid(holdout, "holdout")
    assert booster.eval_train()[0][0] == "training"
    valid = booster.eval_valid()
    assert [row[0] for row in valid] == ["holdout"]
    assert booster.eval_valid(metric="rmse")[0][1] == "rmse"


def test_valid_sets_register_through_train(regression):
    X, y = regression
    booster = train(
        {"objective": "regression"},
        Dataset(X, label=y),
        5,
        valid_sets=[Dataset(X[:40], label=y[:40])],
        valid_names=["holdout"],
    )
    assert [row[0] for row in booster.eval_valid()] == ["holdout"]


def test_eval_rejects_a_metric_from_another_task(regression):
    X, y = regression
    ds = Dataset(X, label=y)
    booster = train({"objective": "regression"}, ds, 5)
    with pytest.raises(ValueError, match="scores binary models"):
        booster.eval(ds, "training", metric="auc")


def test_eval_is_weighted_by_the_dataset(regression):
    X, y = regression
    w = np.linspace(0.1, 2.0, len(y))
    ds = Dataset(X, label=y, weight=w)
    booster = train({"objective": "regression"}, ds, 10)
    value = booster.eval(ds, "training")[0][2]
    pred = booster.predict(X)
    expected = float(np.sum(w * (pred - y) ** 2) / np.sum(w))
    assert value == pytest.approx(expected, rel=1e-12)


def test_feature_importance(regression):
    X, y = regression
    booster = train({"objective": "regression"}, Dataset(X, label=y), 20)
    splits = booster.feature_importance("split")
    gains = booster.feature_importance("gain")
    assert len(splits) == X.shape[1]
    assert splits.sum() > 0
    assert gains.sum() > 0
    with pytest.raises(ValueError, match="unknown importance_type"):
        booster.feature_importance("weight")


def test_model_round_trip(regression, tmp_path):
    X, y = regression
    booster = train({"objective": "regression"}, Dataset(X, label=y), 10)
    path = tmp_path / "model.mbst"
    booster.save_model(path)
    from_file = Booster(model_file=str(path))
    from_string = Booster.model_from_string(booster.model_to_string())
    np.testing.assert_array_equal(from_file.predict(X), booster.predict(X))
    np.testing.assert_array_equal(from_string.predict(X), booster.predict(X))
    assert from_file.current_iteration() == 10
    assert from_file.num_feature() == X.shape[1]


def test_multiclass_model_round_trip(multiclass, tmp_path):
    X, y = multiclass
    booster = train(
        {"objective": "multiclass", "num_class": 3}, Dataset(X, label=y), 8
    )
    path = tmp_path / "model.mbst"
    booster.save_model(path)
    revived = Booster(model_file=str(path))
    assert revived.num_model_per_iteration() == 3
    np.testing.assert_array_equal(revived.predict(X), booster.predict(X))


def test_booster_pickles(regression):
    X, y = regression
    booster = train({"objective": "regression"}, Dataset(X, label=y), 10)
    revived = pickle.loads(pickle.dumps(booster))
    np.testing.assert_array_equal(revived.predict(X), booster.predict(X))
    assert revived.current_iteration() == 10
    # Model format v4 preserves split gains. The training set deliberately
    # does not pickle, so continued training still says so clearly.
    np.testing.assert_array_equal(
        revived.feature_importance("gain"), booster.feature_importance("gain")
    )
    with pytest.raises(ValueError, match="no training set"):
        revived.update()


def test_booster_needs_exactly_one_source(regression):
    X, y = regression
    with pytest.raises(ValueError, match="exactly one"):
        Booster({"objective": "regression"})
    with pytest.raises(ValueError, match="exactly one"):
        Booster({"objective": "regression"}, Dataset(X, label=y), "model.txt")


# -- the estimators hold the same Booster --------------------------------


def test_estimator_booster_is_the_same_model(regression):
    X, y = regression
    estimator = MojoTreesRegressor(n_estimators=12).fit(X, y)
    booster = estimator.booster_
    assert isinstance(booster, Booster)
    assert booster is estimator.booster_
    np.testing.assert_array_equal(booster.predict(X), estimator.predict(X))
    assert booster.current_iteration() == estimator.best_iteration_
    assert booster.num_feature() == estimator.n_features_in_
    np.testing.assert_array_equal(
        booster.feature_importance("split"), estimator.feature_importances_
    )


def test_estimator_booster_reports_class_shape(multiclass):
    X, y = multiclass
    estimator = MojoTreesClassifier(n_estimators=10).fit(X, y)
    booster = estimator.booster_
    assert booster.num_model_per_iteration() == 3
    np.testing.assert_array_equal(
        booster.predict(X), estimator.predict_proba(X)
    )


def test_estimator_booster_carries_feature_names(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    estimator = MojoTreesRegressor(n_estimators=5).fit(frame, y)
    assert estimator.booster_.feature_name() == list(frame.columns)


def test_estimator_booster_cannot_continue_training(regression):
    X, y = regression
    estimator = MojoTreesRegressor(n_estimators=5).fit(X, y)
    with pytest.raises(ValueError, match="no training set"):
        estimator.booster_.update()


def test_unfitted_estimator_has_no_booster(regression):
    with pytest.raises(mojotrees.NotFittedError):
        MojoTreesRegressor().booster_


def test_refit_replaces_the_booster(regression):
    X, y = regression
    estimator = MojoTreesRegressor(n_estimators=5).fit(X, y)
    first = estimator.booster_
    estimator.fit(X, y)
    assert estimator.booster_ is not first
    assert estimator.booster_.current_iteration() == 5


def test_estimator_still_pickles(regression):
    X, y = regression
    estimator = MojoTreesRegressor(n_estimators=8).fit(X, y)
    revived = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_array_equal(revived.predict(X), estimator.predict(X))
    assert isinstance(revived.booster_, Booster)
