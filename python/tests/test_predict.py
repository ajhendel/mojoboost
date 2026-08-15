"""predict/predict_proba consistency and score()."""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRegressor


def test_predict_shape(fitted_regressor, regression):
    X, _ = regression
    pred = fitted_regressor.predict(X)
    assert np.asarray(pred).shape == (len(X),)


@pytest.mark.parametrize("fixture", ["fitted_binary", "fitted_multiclass"])
def test_predict_proba_is_a_distribution(request, fixture):
    est = request.getfixturevalue(fixture)
    X = request.getfixturevalue(
        "binary" if fixture == "fitted_binary" else "multiclass"
    )[0]
    proba = np.asarray(est.predict_proba(X))
    assert proba.shape == (len(X), est.n_classes_)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("fixture", ["fitted_binary", "fitted_multiclass"])
def test_predict_is_the_argmax_of_predict_proba(request, fixture):
    est = request.getfixturevalue(fixture)
    X = request.getfixturevalue(
        "binary" if fixture == "fitted_binary" else "multiclass"
    )[0]
    proba = np.asarray(est.predict_proba(X))
    expected = np.asarray(est.classes_)[proba.argmax(axis=1)]
    assert np.array_equal(np.asarray(est.predict(X)), expected)


def test_proba_columns_follow_classes_order(binary):
    """Column k is the probability of classes_[k], so relabeling must
    permute the columns and nothing else."""
    X, y = binary
    plain = MojoTreesClassifier(n_estimators=10).fit(X, y)
    renamed = MojoTreesClassifier(n_estimators=10).fit(
        X, np.where(y == 1, "b", "a")
    )
    assert list(renamed.classes_) == ["a", "b"]
    assert np.allclose(plain.predict_proba(X), renamed.predict_proba(X))


def test_binary_proba_columns_are_complementary(fitted_binary, binary):
    X, _ = binary
    proba = np.asarray(fitted_binary.predict_proba(X))
    assert np.allclose(proba[:, 0], 1.0 - proba[:, 1])


# -- score ---------------------------------------------------------------


def test_regressor_score_matches_r2(fitted_regressor, regression):
    X, y = regression
    pred = np.asarray(fitted_regressor.predict(X))
    expected = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert fitted_regressor.score(X, y) == pytest.approx(expected)


def test_regressor_score_matches_sklearn(fitted_regressor, regression):
    metrics = pytest.importorskip("sklearn.metrics")
    X, y = regression
    weights = np.linspace(0.5, 2.0, len(y))
    assert fitted_regressor.score(X, y) == pytest.approx(
        metrics.r2_score(y, fitted_regressor.predict(X))
    )
    assert fitted_regressor.score(X, y, sample_weight=weights) == pytest.approx(
        metrics.r2_score(
            y, fitted_regressor.predict(X), sample_weight=weights
        )
    )


def test_r2_of_a_constant_target():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.full(4, 2.0)
    est = MojoTreesRegressor(n_estimators=3, min_data_in_leaf=1).fit(X, y)
    # Fitting a constant lands exactly on it, which is the one case where a
    # zero total sum of squares still counts as a perfect score.
    assert est.score(X, y) == 1.0


def test_classifier_score_is_accuracy(fitted_binary, binary):
    X, y = binary
    pred = np.asarray(fitted_binary.predict(X))
    assert fitted_binary.score(X, y) == pytest.approx((pred == y).mean())


def test_classifier_score_weighted(fitted_binary, binary):
    X, y = binary
    weights = np.linspace(0.1, 3.0, len(y))
    pred = np.asarray(fitted_binary.predict(X))
    expected = (weights * (pred == y)).sum() / weights.sum()
    assert fitted_binary.score(X, y, sample_weight=weights) == pytest.approx(
        expected
    )


def test_classifier_score_on_string_labels(binary):
    X, y = binary
    labels = np.where(y == 1, "yes", "no")
    est = MojoTreesClassifier(n_estimators=10).fit(X, labels)
    assert est.score(X, labels) == pytest.approx(
        (np.asarray(est.predict(X)) == labels).mean()
    )


def test_score_checks_the_target_length(fitted_regressor, regression):
    X, y = regression
    with pytest.raises(ValueError):
        fitted_regressor.score(X, y[:-1])


def _missingness_data(n_rows=300):
    """One feature whose missingness alone drives the target: the observed
    values carry no signal."""
    gen = np.random.default_rng(11)
    x = gen.random(n_rows)
    missing = np.zeros(n_rows, dtype=bool)
    missing[::3] = True
    x[missing] = np.nan
    y = np.where(missing, 5.0, -5.0)
    return x.reshape(-1, 1), y


def test_nan_is_predicted_as_missing():
    """NaN trains and predicts as a missing value, so a target driven by
    missingness alone is learnable and an unseen NaN follows the same route."""
    X, y = _missingness_data()
    est = MojoTreesRegressor(
        num_leaves=4, n_estimators=60, learning_rate=0.3,
        min_data_in_leaf=1, max_bin=16,
    ).fit(X, y)
    pred = np.asarray(est.predict(np.array([[np.nan], [0.25], [0.75]])))
    assert abs(pred[0] - 5.0) < 0.5
    assert abs(pred[1] + 5.0) < 0.5
    assert abs(pred[2] + 5.0) < 0.5


def test_use_missing_false_treats_nan_as_zero():
    """LightGBM's use_missing=false: NaN becomes the value 0.0, so it can no
    longer be told apart from an observed 0.0."""
    X, y = _missingness_data()
    est = MojoTreesRegressor(
        num_leaves=4, n_estimators=60, learning_rate=0.3,
        min_data_in_leaf=1, max_bin=16, use_missing=False,
    ).fit(X, y)
    pred = np.asarray(est.predict(np.array([[np.nan], [0.0]])))
    assert pred[0] == pred[1]


# -- raw_score -----------------------------------------------------------


def test_regressor_raw_score_is_the_response_scale(fitted_regressor,
                                                   regression):
    """Squared error has no link, so the regressor's raw and response scales
    are the same quantity. The flag is accepted so the signature matches
    LightGBM's, not because it changes anything here."""
    X, _ = regression
    plain = np.asarray(fitted_regressor.predict(X))
    raw = np.asarray(fitted_regressor.predict(X, raw_score=True))
    assert raw.shape == plain.shape
    assert np.array_equal(raw, plain)


def test_binary_raw_score_is_the_log_odds(fitted_binary, binary):
    """A binary model returns one raw score per row, not one per class, and
    the logistic of it is the positive class probability."""
    X, _ = binary
    raw = np.asarray(fitted_binary.predict_proba(X, raw_score=True))
    assert raw.shape == (len(X),)
    proba = np.asarray(fitted_binary.predict_proba(X))
    assert np.allclose(1.0 / (1.0 + np.exp(-raw)), proba[:, 1])
    # predict() passes the raw scores through instead of taking an argmax.
    assert np.array_equal(np.asarray(fitted_binary.predict(X, raw_score=True)),
                          raw)


def test_multiclass_raw_score_is_pre_softmax(fitted_multiclass, multiclass):
    X, _ = multiclass
    raw = np.asarray(fitted_multiclass.predict_proba(X, raw_score=True))
    assert raw.shape == (len(X), fitted_multiclass.n_classes_)
    shifted = np.exp(raw - raw.max(axis=1, keepdims=True))
    assert np.allclose(shifted / shifted.sum(axis=1, keepdims=True),
                       np.asarray(fitted_multiclass.predict_proba(X)))


# -- start_iteration / num_iteration -------------------------------------


def test_num_iteration_matches_a_shorter_ensemble(regression):
    """Boosting is sequential and deterministic, so the first k trees of a
    20-round model are the trees of a k-round model: slicing must reproduce
    the shorter fit exactly."""
    X, y = regression
    full = MojoTreesRegressor(n_estimators=20).fit(X, y)
    short = MojoTreesRegressor(n_estimators=7).fit(X, y)
    assert np.allclose(np.asarray(full.predict(X, num_iteration=7)),
                       np.asarray(short.predict(X)))


def test_iteration_slices_sum_to_the_whole_raw_score(fitted_regressor,
                                                     regression):
    """The manual-sum property from the Mojo tests, seen through the Python
    API: [0, k) and [k, n) partition the ensemble for every k >= 1, because
    the base score belongs to iteration 0 and so to the head alone."""
    X, _ = regression
    est = fitted_regressor
    whole = np.asarray(est.predict(X, raw_score=True))
    for k in range(1, est.best_iteration_ + 1):
        head = np.asarray(est.predict(X, raw_score=True, num_iteration=k))
        tail = np.asarray(est.predict(X, raw_score=True, start_iteration=k))
        assert np.allclose(head + tail, whole)


def test_num_iteration_none_uses_every_kept_iteration(fitted_regressor,
                                                      regression):
    X, _ = regression
    est = fitted_regressor
    plain = np.asarray(est.predict(X))
    assert np.array_equal(
        np.asarray(est.predict(X, num_iteration=est.best_iteration_)), plain)
    # A nonpositive count means "all remaining", as in LightGBM.
    assert np.array_equal(np.asarray(est.predict(X, num_iteration=0)), plain)
    assert np.array_equal(np.asarray(est.predict(X, num_iteration=-1)), plain)
    # A count past the end clamps instead of raising.
    assert np.array_equal(np.asarray(est.predict(X, num_iteration=10_000)),
                          plain)


def test_num_iteration_default_follows_best_iteration(regression):
    """LightGBM's default is best_iteration, not n_estimators. mojotrees
    truncates the ensemble at its best iteration when training stops early,
    so the default and best_iteration_ agree by construction."""
    X, y = regression
    noise = np.random.default_rng(3).permutation(y)
    est = MojoTreesRegressor(n_estimators=40).fit(
        X, y,
        eval_set=[(X, noise)],
        eval_metric=lambda truth, pred: float(np.mean((truth - pred) ** 2)),
        early_stopping_rounds=3,
    )
    assert est.best_iteration_ < 40
    plain = np.asarray(est.predict(X))
    assert np.array_equal(
        np.asarray(est.predict(X, num_iteration=est.best_iteration_)), plain)
    # Asking for the rounds early stopping discarded cannot resurrect them.
    assert np.array_equal(np.asarray(est.predict(X, num_iteration=40)), plain)


def test_empty_iteration_range_predicts_the_base_score(fitted_regressor,
                                                       regression):
    """Starting past the last iteration selects no trees. The base score
    sits in iteration 0, which such a range excludes, so the raw score is
    exactly zero."""
    X, _ = regression
    est = fitted_regressor
    empty = np.asarray(
        est.predict(X, raw_score=True, start_iteration=est.best_iteration_))
    assert empty.shape == (len(X),)
    assert np.array_equal(empty, np.zeros(len(X)))
    # A start past the end clamps to that same empty range.
    assert np.array_equal(
        np.asarray(est.predict(X, raw_score=True, start_iteration=99_999)),
        np.zeros(len(X)))
    # Leaf prediction over an empty range keeps its rows and loses its
    # columns rather than failing.
    assert np.asarray(
        est.predict(X, pred_leaf=True, start_iteration=99_999)
    ).shape == (len(X), 0)


def test_negative_start_iteration_clamps_to_zero(fitted_regressor,
                                                 regression):
    X, _ = regression
    est = fitted_regressor
    assert np.array_equal(
        np.asarray(est.predict(X, start_iteration=-5)),
        np.asarray(est.predict(X)))


def test_multiclass_iteration_slice_is_a_truncated_softmax(multiclass):
    X, y = multiclass
    full = MojoTreesClassifier(n_estimators=12).fit(X, y)
    short = MojoTreesClassifier(n_estimators=5).fit(X, y)
    sliced = np.asarray(full.predict_proba(X, num_iteration=5))
    assert sliced.shape == (len(X), full.n_classes_)
    assert np.allclose(sliced.sum(axis=1), 1.0)
    assert np.allclose(sliced, np.asarray(short.predict_proba(X)))


# -- pred_leaf -----------------------------------------------------------


def test_pred_leaf_shape_and_range(fitted_regressor, regression):
    X, _ = regression
    est = fitted_regressor
    leaves = np.asarray(est.predict(X, pred_leaf=True))
    assert leaves.shape == (len(X), est.best_iteration_)
    assert np.issubdtype(leaves.dtype, np.integer)
    assert (leaves >= 0).all()
    # num_leaves bounds every tree, so it bounds every ordinal.
    assert (leaves < est.num_leaves).all()


def test_pred_leaf_follows_the_iteration_range(fitted_regressor, regression):
    X, _ = regression
    est = fitted_regressor
    full = np.asarray(est.predict(X, pred_leaf=True))
    head = np.asarray(est.predict(X, pred_leaf=True, num_iteration=4))
    tail = np.asarray(est.predict(X, pred_leaf=True, start_iteration=4))
    assert np.array_equal(head, full[:, :4])
    assert np.array_equal(tail, full[:, 4:])
    empty = np.asarray(
        est.predict(X, pred_leaf=True, start_iteration=est.best_iteration_))
    assert empty.shape == (len(X), 0)


def test_pred_leaf_agrees_with_predictions(fitted_regressor, regression):
    """Leaf ordinals identify the leaves a row lands in, so two rows that
    land in the same leaf of every tree must get the same prediction."""
    X, _ = regression
    est = fitted_regressor
    leaves = np.asarray(est.predict(X, pred_leaf=True))
    pred = np.asarray(est.predict(X))
    groups = {}
    for row in range(len(X)):
        groups.setdefault(tuple(leaves[row]), []).append(row)
    for rows in groups.values():
        assert np.allclose(pred[rows], pred[rows[0]])
    # The routing is not degenerate: the rows do not all share one path.
    assert len(groups) > 1


def test_binary_pred_leaf_is_single_output(fitted_binary, binary):
    """A binary classifier is one ensemble, not one per class, so it has one
    leaf column per iteration, as LightGBM does."""
    X, _ = binary
    est = fitted_binary
    leaves = np.asarray(est.predict(X, pred_leaf=True))
    assert leaves.shape == (len(X), est.best_iteration_)
    assert np.array_equal(
        leaves, np.asarray(est.predict_proba(X, pred_leaf=True)))


def test_multiclass_pred_leaf_is_round_major(fitted_multiclass, multiclass):
    """One tree per class per iteration, so column i * n_classes + k is
    class k's tree in iteration i."""
    X, _ = multiclass
    est = fitted_multiclass
    k = est.n_classes_
    leaves = np.asarray(est.predict(X, pred_leaf=True))
    assert leaves.shape == (len(X), est.best_iteration_ * k)
    assert np.issubdtype(leaves.dtype, np.integer)
    assert (leaves >= 0).all() and (leaves < est.num_leaves).all()
    # Slicing drops whole iterations, k columns at a time.
    head = np.asarray(est.predict(X, pred_leaf=True, num_iteration=2))
    assert head.shape == (len(X), 2 * k)
    assert np.array_equal(head, leaves[:, : 2 * k])


def test_pred_leaf_is_stable_across_calls(fitted_multiclass, multiclass):
    X, _ = multiclass
    first = np.asarray(fitted_multiclass.predict(X, pred_leaf=True))
    second = np.asarray(fitted_multiclass.predict(X, pred_leaf=True))
    assert np.array_equal(first, second)


def test_pred_leaf_survives_save_and_load(fitted_regressor, regression,
                                          tmp_path):
    """Leaf ordinals come from node order, which the model format preserves,
    so a reloaded model numbers leaves exactly as the saved one did."""
    X, _ = regression
    path = tmp_path / "model.txt"
    fitted_regressor.save(path)
    loaded = MojoTreesRegressor.load(path)
    assert loaded.best_iteration_ == fitted_regressor.best_iteration_
    assert np.array_equal(
        np.asarray(loaded.predict(X, pred_leaf=True)),
        np.asarray(fitted_regressor.predict(X, pred_leaf=True)))
    assert np.allclose(
        np.asarray(loaded.predict(X, raw_score=True, num_iteration=6)),
        np.asarray(fitted_regressor.predict(X, raw_score=True,
                                            num_iteration=6)))


def test_pred_leaf_survives_pickle(fitted_binary, binary):
    import pickle

    X, _ = binary
    twin = pickle.loads(pickle.dumps(fitted_binary))
    assert np.array_equal(
        np.asarray(twin.predict(X, pred_leaf=True)),
        np.asarray(fitted_binary.predict(X, pred_leaf=True)))


# -- validate_features ---------------------------------------------------


def test_validate_features_requires_names_on_both_sides(fitted_regressor,
                                                        regression):
    """Fitted from a bare array, so there are no recorded names to check
    against: asking for validation must say so rather than pass silently."""
    X, _ = regression
    with pytest.raises(ValueError, match="validate_features=True"):
        fitted_regressor.predict(X, validate_features=True)


def test_validate_features_raises_where_it_would_warn(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    names = [f"f{i}" for i in range(X.shape[1])]
    est = MojoTreesRegressor(n_estimators=5).fit(pd.DataFrame(X, columns=names),
                                                 y)
    # Without the flag a nameless matrix only warns.
    with pytest.warns(UserWarning):
        est.predict(X)
    with pytest.raises(ValueError, match="does not have valid feature names"):
        est.predict(X, validate_features=True)


def test_validate_features_accepts_matching_names(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    names = [f"f{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=names)
    est = MojoTreesRegressor(n_estimators=5).fit(frame, y)
    assert np.allclose(np.asarray(est.predict(frame, validate_features=True)),
                       np.asarray(est.predict(frame)))


def test_mismatched_names_raise_with_or_without_validation(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    names = [f"f{i}" for i in range(X.shape[1])]
    est = MojoTreesRegressor(n_estimators=5).fit(pd.DataFrame(X, columns=names),
                                                 y)
    renamed = pd.DataFrame(X, columns=[n.upper() for n in names])
    for flag in (False, True):
        with pytest.raises(ValueError, match="feature names should match"):
            est.predict(renamed, validate_features=flag)


# -- incompatible flags and bad arguments --------------------------------


@pytest.mark.parametrize(
    "fixture,data",
    [("fitted_regressor", "regression"), ("fitted_binary", "binary"),
     ("fitted_multiclass", "multiclass")],
)
def test_raw_score_and_pred_leaf_are_exclusive(request, fixture, data):
    est = request.getfixturevalue(fixture)
    X = request.getfixturevalue(data)[0]
    with pytest.raises(ValueError, match="at most one"):
        est.predict(X, raw_score=True, pred_leaf=True)


def test_classifier_proba_rejects_the_same_combination(fitted_multiclass,
                                                       multiclass):
    X, _ = multiclass
    with pytest.raises(ValueError, match="at most one"):
        fitted_multiclass.predict_proba(X, raw_score=True, pred_leaf=True)


@pytest.mark.parametrize("bad", ["3", 2.5, None])
def test_start_iteration_must_be_an_integer(fitted_regressor, regression, bad):
    X, _ = regression
    with pytest.raises(TypeError, match="start_iteration must be an integer"):
        fitted_regressor.predict(X, start_iteration=bad)


@pytest.mark.parametrize("bad", ["3", 2.5])
def test_num_iteration_must_be_an_integer(fitted_regressor, regression, bad):
    X, _ = regression
    with pytest.raises(TypeError, match="num_iteration must be an integer"):
        fitted_regressor.predict(X, num_iteration=bad)


def test_bool_is_not_an_iteration_count(fitted_regressor, regression):
    """A misplaced flag is far likelier than a request for one iteration."""
    X, _ = regression
    with pytest.raises(TypeError, match="got a bool"):
        fitted_regressor.predict(X, num_iteration=True)


def test_numpy_integers_are_accepted(fitted_regressor, regression):
    X, _ = regression
    assert np.array_equal(
        np.asarray(fitted_regressor.predict(X, num_iteration=np.int64(5))),
        np.asarray(fitted_regressor.predict(X, num_iteration=5)))


# -- ranker and device coverage ------------------------------------------


def test_ranker_supports_the_same_options(regression):
    from mojotrees import MojoTreesRanker

    X, _ = regression
    gen = np.random.default_rng(5)
    y = gen.integers(0, 3, size=len(X))
    group = [len(X) // 4] * 4
    est = MojoTreesRanker(n_estimators=8).fit(X, y, group=group)
    plain = np.asarray(est.predict(X))
    # Lambdarank has no inverse link, so raw and response coincide.
    assert np.array_equal(np.asarray(est.predict(X, raw_score=True)), plain)
    leaves = np.asarray(est.predict(X, pred_leaf=True))
    assert leaves.shape == (len(X), est.best_iteration_)
    head = np.asarray(est.predict(X, raw_score=True, num_iteration=3))
    tail = np.asarray(est.predict(X, raw_score=True, start_iteration=3))
    assert np.allclose(head + tail, plain)


def test_gpu_trained_model_slices_and_reports_leaves(regression):
    """The device chooses the trainer, not the model: a GPU-trained ensemble
    must answer the same prediction contract."""
    from mojotrees import gpu_available

    if not gpu_available():
        pytest.skip("no accelerator available for training")
    X, y = regression
    est = MojoTreesRegressor(n_estimators=10, device="gpu").fit(X, y)
    assert est.device_ == "gpu"
    whole = np.asarray(est.predict(X, raw_score=True))
    head = np.asarray(est.predict(X, raw_score=True, num_iteration=4))
    tail = np.asarray(est.predict(X, raw_score=True, start_iteration=4))
    assert np.allclose(head + tail, whole)
    leaves = np.asarray(est.predict(X, pred_leaf=True))
    assert leaves.shape == (len(X), est.best_iteration_)
    assert (leaves >= 0).all() and (leaves < est.num_leaves).all()
