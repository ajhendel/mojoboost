"""Native categorical features through the Python API.

The split search itself is covered in Mojo (tests/test_categorical.mojo).
What is under test here is the contract the Python layer adds on top: which
columns `categorical_feature` resolves to, how a pandas `category` column
becomes integer codes, and that prediction encodes through the tables `fit`
recorded rather than through whatever the incoming frame happens to carry.

The categorical hyperparameters are sized down from LightGBM's defaults in
most tests here, the way `_small_cat_params` does in the Mojo suite:
`min_data_per_group=100` alone rejects every split in a few-hundred-row
dataset.
"""

import pickle

import pytest

from mojoboost import (
    MojoBoostClassifier,
    MojoBoostRanker,
    MojoBoostRegressor,
)

np = pytest.importorskip("numpy")

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is optional
    pd = None

needs_pandas = pytest.mark.skipif(pd is None, reason="pandas is not installed")

# Category effects that alternate with the code, so no `code <= t` threshold
# separates them: a model that treats the codes as ordinal has to carve out
# one code at a time.
LABELS = ["a", "b", "c", "d", "e", "f"]
CAT_KWARGS = dict(
    num_leaves=8,
    n_estimators=40,
    learning_rate=0.2,
    min_data_in_leaf=5,
    max_cat_to_onehot=4,
    max_cat_threshold=32,
    cat_smooth=2.0,
    cat_l2=10.0,
    min_data_per_group=5,
)


def _codes(n_rows=600, n_categories=6):
    return np.arange(n_rows) % n_categories


def _effect(codes):
    """The signal: alternating in the code, so it is invisible to a
    threshold on it."""
    return np.where(codes % 3 == 0, 1.0, -1.0)


def _numeric_data(n_rows=600, n_categories=6, seed=3):
    """(X, y) with an integer-coded category column and a numerical one."""
    gen = np.random.default_rng(seed)
    codes = _codes(n_rows, n_categories)
    x = gen.random(n_rows)
    y = _effect(codes) + 2.0 * x
    return np.column_stack([codes.astype(float), x]), y


def _frame(codes, x, categories=None):
    """The same matrix as a DataFrame whose first column is a pandas
    categorical of `LABELS`, optionally with its categories in a different
    order than the labels appear."""
    labels = [LABELS[c] for c in codes]
    if categories is None:
        categories = LABELS
    return pd.DataFrame(
        {
            "color": pd.Categorical(labels, categories=list(categories)),
            "x": x,
        }
    )


def _frame_data(n_rows=600, n_categories=6, seed=3, categories=None):
    gen = np.random.default_rng(seed)
    codes = _codes(n_rows, n_categories)
    x = gen.random(n_rows)
    y = _effect(codes) + 2.0 * x
    return _frame(codes, x, categories), y, codes, x


# --- resolving categorical_feature -----------------------------------------


@needs_pandas
def test_auto_marks_every_pandas_category_column():
    """LightGBM's default: 'auto' means the pandas `category` columns and
    nothing else."""
    X, y, _, _ = _frame_data()
    est = MojoBoostRegressor(**CAT_KWARGS).fit(X, y)
    assert est.categorical_feature_ == [0]


def test_auto_marks_nothing_on_an_array():
    """A numpy matrix carries no dtype to read, so 'auto' is no columns and
    the default estimator trains exactly as it did before."""
    X, y = _numeric_data()
    est = MojoBoostRegressor(n_estimators=5).fit(X, y)
    assert est.categorical_feature_ == []


def test_indices_mark_columns_of_an_array():
    X, y = _numeric_data()
    est = MojoBoostRegressor(categorical_feature=[0], **CAT_KWARGS).fit(X, y)
    assert est.categorical_feature_ == [0]


@needs_pandas
def test_a_name_and_its_index_resolve_to_the_same_model():
    X, y, _, _ = _frame_data()
    by_name = MojoBoostRegressor(
        categorical_feature=["color"], **CAT_KWARGS
    ).fit(X, y)
    by_index = MojoBoostRegressor(
        categorical_feature=[0], **CAT_KWARGS
    ).fit(X, y)
    assert by_name.categorical_feature_ == by_index.categorical_feature_ == [0]
    assert list(by_name.predict(X)) == list(by_index.predict(X))


@needs_pandas
def test_names_and_indices_can_be_mixed():
    X, y, codes, x = _frame_data()
    X = X.assign(other=codes.astype(float))
    est = MojoBoostRegressor(
        categorical_feature=["color", 2], **CAT_KWARGS
    ).fit(X, y)
    assert est.categorical_feature_ == [0, 2]


def test_the_plural_alias_is_accepted():
    X, y = _numeric_data()
    est = MojoBoostRegressor(categorical_features=[0], **CAT_KWARGS).fit(X, y)
    assert est.categorical_feature_ == [0]


def test_conflicting_aliases_raise():
    X, y = _numeric_data()
    est = MojoBoostRegressor(
        categorical_feature=[0], categorical_features=[1], **CAT_KWARGS
    )
    with pytest.raises(ValueError, match="aliases"):
        est.fit(X, y)


def test_none_means_no_categorical_feature():
    X, y = _numeric_data()
    est = MojoBoostRegressor(categorical_feature=None, n_estimators=5)
    assert est.fit(X, y).categorical_feature_ == []


@pytest.mark.parametrize(
    "spec, match",
    [
        ("categorical", "expected 'auto'"),
        ("color", "expected 'auto'"),
        ([0.5], "whole feature indices"),
        ([True], "bool"),
        ([None], "neither a feature name nor an index"),
        ([9], "out of range"),
        ([0, 0], "twice"),
    ],
)
def test_bad_categorical_feature_raises(spec, match):
    X, y = _numeric_data(n_rows=60)
    est = MojoBoostRegressor(categorical_feature=spec, n_estimators=3)
    with pytest.raises(ValueError, match=match):
        est.fit(X, y)


def test_a_name_needs_a_matrix_that_carries_names():
    X, y = _numeric_data(n_rows=60)
    est = MojoBoostRegressor(categorical_feature=["color"], n_estimators=3)
    with pytest.raises(ValueError, match="carries no feature names"):
        est.fit(X, y)


@needs_pandas
def test_an_unknown_name_raises():
    X, y, _, _ = _frame_data(n_rows=60)
    est = MojoBoostRegressor(categorical_feature=["shade"], n_estimators=3)
    with pytest.raises(ValueError, match="not a column of X"):
        est.fit(X, y)


@needs_pandas
def test_a_name_listed_twice_by_name_and_index_raises():
    X, y, _, _ = _frame_data(n_rows=60)
    est = MojoBoostRegressor(
        categorical_feature=["color", 0], n_estimators=3
    )
    with pytest.raises(ValueError, match="twice"):
        est.fit(X, y)


@needs_pandas
def test_a_category_column_left_out_of_the_list_raises():
    """LightGBM would feed its codes to the numerical scan. A declared
    category is never an ordered number, so this is an error."""
    X, y, _, _ = _frame_data(n_rows=60)
    est = MojoBoostRegressor(categorical_feature=[], n_estimators=3)
    with pytest.raises(ValueError, match="not in categorical_feature"):
        est.fit(X, y)


# --- what the codes mean ---------------------------------------------------


def test_declared_categories_are_not_treated_as_ordered():
    """The point of the feature: with twelve alternating codes and four
    leaves, ordinal thresholds cannot separate the groups within the leaf
    budget and category sets can."""
    n_rows = 1200
    codes = _codes(n_rows, 12)
    y = _effect(codes)
    X = codes.astype(float).reshape(-1, 1)
    kwargs = dict(CAT_KWARGS, num_leaves=4, n_estimators=20)
    as_cat = MojoBoostRegressor(categorical_feature=[0], **kwargs).fit(X, y)
    as_num = MojoBoostRegressor(categorical_feature=None, **kwargs).fit(X, y)
    cat_sse = float(np.sum((np.asarray(as_cat.predict(X)) - y) ** 2))
    num_sse = float(np.sum((np.asarray(as_num.predict(X)) - y) ** 2))
    assert cat_sse / n_rows < 0.02
    assert num_sse > 10.0 * cat_sse


def test_unseen_missing_and_negative_codes_share_one_route():
    """All three land in the reserved unknown bin, which is in no split's
    category set, so all three route right at every categorical node."""
    X, y = _numeric_data()
    est = MojoBoostRegressor(categorical_feature=[0], **CAT_KWARGS).fit(X, y)
    rows = np.array(
        [[99.0, 0.5], [-1.0, 0.5], [-12345.0, 0.5], [np.nan, 0.5]]
    )
    pred = np.asarray(est.predict(rows))
    assert np.all(pred == pred[0])
    # And that route is not the one a seen category takes, or the assertion
    # above would hold for a model with no categorical split at all.
    seen = np.asarray(est.predict(np.array([[0.0, 0.5], [1.0, 0.5]])))
    assert not np.allclose(seen[0], seen[1])


@pytest.mark.parametrize("bad", [2.0**31, 2.0**40])
def test_oversized_codes_are_rejected(bad):
    """Codes have to stay representable as Int32, LightGBM's category type.
    Rejecting them at both ends beats binning them as unseen at predict
    time and raising at fit time."""
    X, y = _numeric_data(n_rows=120)
    est = MojoBoostRegressor(categorical_feature=[0], **CAT_KWARGS)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        est.fit(np.column_stack([np.full(120, bad), X[:, 1]]), y)
    est.fit(X, y)
    with pytest.raises(ValueError, match="not a category code"):
        est.predict(np.array([[bad, 0.5]]))


def test_fractional_codes_are_rejected():
    """`bin_of` truncates toward zero, so 1.5 and 1 would be one category.
    Merging them silently is worse than saying so."""
    X, y = _numeric_data(n_rows=120)
    est = MojoBoostRegressor(categorical_feature=[0], **CAT_KWARGS)
    with pytest.raises(ValueError, match="not a category code"):
        est.fit(np.column_stack([X[:, 0] + 0.5, X[:, 1]]), y)
    est.fit(X, y)
    with pytest.raises(ValueError, match="not a category code"):
        est.predict(np.array([[1.5, 0.5]]))


def test_a_numerical_column_is_left_alone():
    """Fractional and huge values are only rejected in a declared
    categorical column; the numerical column beside it is untouched."""
    X, y = _numeric_data(n_rows=120)
    est = MojoBoostRegressor(categorical_feature=[0], **CAT_KWARGS).fit(X, y)
    assert len(est.predict(np.array([[0.0, 1e30], [1.0, 0.25]]))) == 2


# --- pandas categorical columns --------------------------------------------


@needs_pandas
def test_a_category_column_trains_on_its_labels():
    X, y, codes, x = _frame_data()
    est = MojoBoostRegressor(**CAT_KWARGS).fit(X, y)
    # The alternating effect is recovered per label, which no threshold on
    # the codes could do.
    pred = np.asarray(est.predict(X))
    for c in range(6):
        rows = codes == c
        assert abs(np.mean(pred[rows] - 2.0 * x[rows]) - _effect(c)) < 0.15


@needs_pandas
def test_prediction_re_encodes_through_the_fitted_categories():
    """The same label must reach the same category whatever the prediction
    frame numbers it, which is the whole reason the tables are kept."""
    X, y, codes, x = _frame_data()
    est = MojoBoostRegressor(**CAT_KWARGS).fit(X, y)
    reordered = _frame(codes, x, categories=list(reversed(LABELS)))
    # The frame's own codes really are different, so a model that trusted
    # them would predict something else here.
    assert not np.array_equal(
        np.asarray(X["color"].cat.codes), np.asarray(reordered["color"].cat.codes)
    )
    assert list(est.predict(reordered)) == list(est.predict(X))


@needs_pandas
def test_a_prediction_frame_may_carry_plain_labels():
    """A column of the labels themselves, not yet a category dtype, is
    encoded through the same tables."""
    X, y, codes, x = _frame_data()
    est = MojoBoostRegressor(**CAT_KWARGS).fit(X, y)
    plain = pd.DataFrame({"color": [LABELS[c] for c in codes], "x": x})
    assert list(est.predict(plain)) == list(est.predict(X))


@needs_pandas
def test_a_label_never_seen_predicts_as_unseen():
    X, y, codes, x = _frame_data()
    est = MojoBoostRegressor(**CAT_KWARGS).fit(X, y)
    new = pd.DataFrame({"color": ["zzz"] * len(codes), "x": x})
    missing = pd.DataFrame({"color": [None] * len(codes), "x": x})
    assert list(est.predict(new)) == list(est.predict(missing))


@needs_pandas
def test_a_label_fitted_model_refuses_an_array():
    """Only a frame carries labels, and the codes of an array are not the
    ones the model was fitted on."""
    X, y, codes, x = _frame_data()
    est = MojoBoostRegressor(**CAT_KWARGS).fit(X, y)
    with pytest.raises(ValueError, match="DataFrame"):
        est.predict(np.column_stack([codes.astype(float), x]))


@needs_pandas
def test_a_code_fitted_model_refuses_a_category_frame():
    """The reverse: a model with no label mapping cannot be handed one."""
    X, y = _numeric_data()
    est = MojoBoostRegressor(categorical_feature=[0], **CAT_KWARGS).fit(X, y)
    frame, _, _, _ = _frame_data()
    with pytest.raises(ValueError, match="no category mapping"):
        est.predict(frame)


# --- other estimators ------------------------------------------------------


@needs_pandas
def test_the_classifier_takes_categorical_columns():
    X, y, codes, _ = _frame_data()
    labels = (codes % 3).astype(np.int64)
    est = MojoBoostClassifier(**CAT_KWARGS).fit(X, labels)
    assert est.categorical_feature_ == [0]
    assert est.n_classes_ == 3
    assert np.mean(np.asarray(est.predict(X)) == labels) > 0.95


def test_the_ranker_takes_categorical_columns():
    """The ranker's own fit path passes categorical indices to one tree.

    Keep this to one estimator: a categorical split can group all even codes
    at once, while one four-leaf numerical tree cannot represent all seven
    alternating boundaries.  With many boosting rounds the numerical model
    can eventually compose enough thresholds to score perfectly too, which
    tests boosting capacity rather than categorical plumbing.
    """
    n_queries, per_query = 60, 8
    codes = np.tile(np.arange(per_query), n_queries)
    X = codes.astype(float).reshape(-1, 1)
    y = (codes % 2 == 0).astype(np.int64)
    group = [per_query] * n_queries
    kwargs = dict(CAT_KWARGS, num_leaves=4, n_estimators=1, ndcg_eval_at=4)
    as_cat = MojoBoostRanker(categorical_feature=[0], **kwargs).fit(
        X, y, group=group
    )
    as_num = MojoBoostRanker(categorical_feature=None, **kwargs).fit(
        X, y, group=group
    )
    assert as_cat.categorical_feature_ == [0]
    assert as_cat.score(X, y, group=group) > as_num.score(X, y, group=group)
    assert as_cat.score(X, y, group=group) == pytest.approx(1.0)


# --- fitted state ----------------------------------------------------------


@needs_pandas
def test_pickle_keeps_the_label_mapping():
    X, y, _, _ = _frame_data()
    est = MojoBoostRegressor(**CAT_KWARGS).fit(X, y)
    clone = pickle.loads(pickle.dumps(est))
    assert clone.categorical_feature_ == [0]
    assert list(clone.predict(X)) == list(est.predict(X))


def test_save_and_load_keep_the_category_tables(tmp_path):
    """The model file carries the tables, so a loaded model splits and
    routes exactly as it did; what it cannot carry is a label encoding."""
    X, y = _numeric_data()
    est = MojoBoostRegressor(categorical_feature=[0], **CAT_KWARGS).fit(X, y)
    path = tmp_path / "model.mbst"
    est.save(path)
    loaded = MojoBoostRegressor.load(path)
    assert loaded.categorical_feature_ == [0]
    assert list(loaded.predict(X)) == list(est.predict(X))
    rows = np.array([[99.0, 0.5], [np.nan, 0.5]])
    assert list(loaded.predict(rows)) == list(est.predict(rows))


@needs_pandas
def test_a_loaded_model_has_no_labels_to_encode_with(tmp_path):
    X, y, codes, x = _frame_data()
    est = MojoBoostRegressor(**CAT_KWARGS).fit(X, y)
    path = tmp_path / "model.mbst"
    est.save(path)
    loaded = MojoBoostRegressor.load(path)
    assert loaded.categorical_feature_ == [0]
    with pytest.raises(ValueError, match="no category mapping"):
        loaded.predict(X)
    # It still predicts on the codes the labels were encoded to, which is
    # what the file does hold.
    codes_only = np.column_stack(
        [np.asarray(X["color"].cat.codes, dtype=float), x]
    )
    assert list(loaded.predict(codes_only)) == list(est.predict(X))


@needs_pandas
def test_refitting_replaces_the_mapping():
    """Training again is training from scratch, so the second fit's
    categories are the ones prediction encodes through."""
    X, y, codes, x = _frame_data()
    est = MojoBoostRegressor(**CAT_KWARGS).fit(X, y)
    renamed = [label.upper() for label in LABELS]
    other = pd.DataFrame(
        {
            "color": pd.Categorical(
                [renamed[c] for c in codes], categories=renamed
            ),
            "x": x,
        }
    )
    est.fit(other, y)
    assert est.categorical_feature_ == [0]
    # 'a'..'f' are not categories of this model any more, so they are unseen
    # rather than reinterpreted through the frame's own codes: exactly the
    # route a label the model never met takes.
    unseen = pd.DataFrame({"color": ["zzz"] * len(codes), "x": x})
    assert list(est.predict(X)) == list(est.predict(unseen))
    # And the refit did learn the new labels, or everything above would be
    # unseen and the equality would be vacuous.
    assert not np.allclose(est.predict(other), est.predict(unseen))


def test_refitting_without_categorical_features_clears_the_state():
    X, y = _numeric_data()
    est = MojoBoostRegressor(categorical_feature=[0], **CAT_KWARGS).fit(X, y)
    assert est.categorical_feature_ == [0]
    est.set_params(categorical_feature=None).fit(X, y)
    assert est.categorical_feature_ == []
    # The fractional value would be rejected while column 0 was categorical.
    assert len(est.predict(np.array([[1.5, 0.5]]))) == 1
