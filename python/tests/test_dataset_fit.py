"""Fitting an estimator on a prepared `mojotrees.Dataset`.

`fit(dataset)` trains on a matrix that is already binned, which is what
every peer's estimator surface takes (CatBoost's `Pool`, and LightGBM's and
XGBoost's through their functional APIs). Two claims are worth a test and
both are checked here by comparing whole model files rather than
predictions alone.

- **Reuse is free of consequence.** The model a Dataset fit produces is the
  model the same fit on the raw matrix produces, tree for tree, for every
  estimator and for weights, categorical columns and query groups alike.
- **A Dataset that does not match the estimator is refused.** Everything
  that shapes the binning (`max_bin`, `use_missing`, the categorical
  declaration, the `ctr` rule) is compared against the dataset's, and a
  disagreement names both values instead of letting one of them win.

Runs under pytest.
"""

import numpy as np
import pytest

from mojotrees import (
    Dataset,
    MojoTreesClassifier,
    MojoTreesRanker,
    MojoTreesRegressor,
)

#: Small on purpose. Bit identity is a property of the code path, not of the
#: shape, and these fits run once per assertion.
_ROUNDS = 3


def _model_text(estimator):
    """The whole fitted model as text, which is what makes the comparison
    bit identity rather than agreement to a tolerance."""
    return estimator.booster_.model_to_string()


def _ranking_data(rng):
    X = rng.random((300, 4))
    labels = rng.integers(0, 4, 300).astype(float)
    return X, labels, [10] * 30


def test_dataset_fit_matches_raw_matrix(regression):
    X, y = regression
    raw = MojoTreesRegressor(n_estimators=_ROUNDS, random_state=1).fit(X, y)
    prepared = MojoTreesRegressor(n_estimators=_ROUNDS, random_state=1).fit(
        Dataset(X, label=y)
    )
    assert _model_text(prepared) == _model_text(raw)
    assert np.array_equal(prepared.predict(X), raw.predict(X))
    assert prepared.n_features_in_ == raw.n_features_in_
    assert prepared.device_ == raw.device_


def test_dataset_is_binned_once_and_reused(regression):
    """The point of the door. Two fits on one dataset are two fits on one
    binning, and both are the raw matrix's model."""
    X, y = regression
    dataset = Dataset(X, label=y)
    first = MojoTreesRegressor(n_estimators=_ROUNDS, random_state=1).fit(
        dataset
    )
    second = MojoTreesRegressor(n_estimators=_ROUNDS, random_state=1).fit(
        dataset
    )
    raw = MojoTreesRegressor(n_estimators=_ROUNDS, random_state=1).fit(X, y)
    assert _model_text(first) == _model_text(raw)
    assert _model_text(second) == _model_text(raw)


def test_dataset_fit_matches_with_weights(regression, rng):
    X, y = regression
    weight = rng.random(len(y)) + 0.5
    raw = MojoTreesRegressor(n_estimators=_ROUNDS, random_state=1).fit(
        X, y, sample_weight=weight
    )
    prepared = MojoTreesRegressor(n_estimators=_ROUNDS, random_state=1).fit(
        Dataset(X, label=y, weight=weight)
    )
    assert _model_text(prepared) == _model_text(raw)


def test_dataset_fit_matches_with_categorical(regression, rng):
    X, y = regression
    Xc = X.copy()
    Xc[:, 3] = rng.integers(0, 6, len(y))
    raw = MojoTreesRegressor(
        n_estimators=_ROUNDS, random_state=1, categorical_feature=[3]
    ).fit(Xc, y)
    dataset = Dataset(Xc, label=y, categorical_feature=[3])
    prepared = MojoTreesRegressor(
        n_estimators=_ROUNDS, random_state=1, categorical_feature=[3]
    ).fit(dataset)
    assert _model_text(prepared) == _model_text(raw)
    assert prepared.categorical_feature_ == [3]
    # `categorical_feature='auto'` declares nothing at this door, because
    # there is no frame to read dtypes off, so the dataset's declaration
    # stands rather than being overruled by an empty one.
    inherited = MojoTreesRegressor(
        n_estimators=_ROUNDS, random_state=1
    ).fit(dataset)
    assert inherited.categorical_feature_ == [3]
    assert _model_text(inherited) == _model_text(raw)


def test_dataset_fit_matches_for_binary(binary):
    X, y = binary
    raw = MojoTreesClassifier(n_estimators=_ROUNDS, random_state=1).fit(X, y)
    prepared = MojoTreesClassifier(n_estimators=_ROUNDS, random_state=1).fit(
        Dataset(X, label=y)
    )
    assert _model_text(prepared) == _model_text(raw)
    assert np.array_equal(
        prepared.predict_proba(X), raw.predict_proba(X)
    )
    assert list(prepared.classes_) == [0, 1]


def test_dataset_fit_matches_for_multiclass(multiclass):
    X, y = multiclass
    raw = MojoTreesClassifier(n_estimators=_ROUNDS, random_state=1).fit(X, y)
    prepared = MojoTreesClassifier(n_estimators=_ROUNDS, random_state=1).fit(
        Dataset(X, label=y)
    )
    assert _model_text(prepared) == _model_text(raw)
    assert prepared.n_classes_ == raw.n_classes_


def test_dataset_fit_matches_for_ranker(rng):
    X, labels, group = _ranking_data(rng)
    raw = MojoTreesRanker(n_estimators=_ROUNDS, random_state=1).fit(
        X, labels, group=group
    )
    prepared = MojoTreesRanker(n_estimators=_ROUNDS, random_state=1).fit(
        Dataset(X, label=labels, group=group)
    )
    assert _model_text(prepared) == _model_text(raw)


def test_dataset_fit_matches_for_sparse(rng):
    """A sparse Dataset stays sparse and trains through the sparse grower,
    which is the trainer `fit` on a SciPy matrix reaches too."""
    sparse = pytest.importorskip("scipy.sparse")
    X = sparse.random(300, 6, density=0.3, format="csc", random_state=1)
    y = np.asarray(X.todense())[:, 0] * 2.0 + 0.05 * rng.standard_normal(300)
    raw = MojoTreesRegressor(n_estimators=_ROUNDS, random_state=1).fit(X, y)
    prepared = MojoTreesRegressor(n_estimators=_ROUNDS, random_state=1).fit(
        Dataset(X, label=y)
    )
    assert _model_text(prepared) == _model_text(raw)


def test_dataset_feature_names_reach_the_estimator(regression):
    X, y = regression
    names = [f"f{i}" for i in range(X.shape[1])]
    prepared = MojoTreesRegressor(n_estimators=_ROUNDS).fit(
        Dataset(X, label=y, feature_name=names)
    )
    assert list(prepared.feature_names_in_) == names
    assert prepared.feature_name_ == names


# -- what a mismatched Dataset does --------------------------------------


def test_max_bin_mismatch_raises(regression):
    X, y = regression
    dataset = Dataset(X, label=y, params={"max_bin": 63})
    with pytest.raises(ValueError, match="max_bin disagrees"):
        MojoTreesRegressor(n_estimators=_ROUNDS, max_bin=255).fit(dataset)


def test_use_missing_mismatch_raises(regression):
    X, y = regression
    dataset = Dataset(X, label=y)
    with pytest.raises(ValueError, match="use_missing disagrees"):
        MojoTreesRegressor(n_estimators=_ROUNDS, use_missing=False).fit(
            dataset
        )


def test_categorical_mismatch_raises(regression):
    X, y = regression
    dataset = Dataset(X, label=y)
    with pytest.raises(ValueError, match="categorical declaration disagrees"):
        MojoTreesRegressor(
            n_estimators=_ROUNDS, categorical_feature=[0]
        ).fit(dataset)


def test_ctr_mismatch_raises(regression):
    """`grow_policy='symmetrictree'` resolves `ctr` to CatBoost's rule, and
    a dataset built without one has no CTR columns to train on."""
    X, y = regression
    dataset = Dataset(X, label=y)
    with pytest.raises(ValueError, match="ctr rule disagrees"):
        MojoTreesRegressor(
            n_estimators=_ROUNDS, grow_policy="symmetrictree"
        ).fit(dataset)


def test_y_beside_a_dataset_raises(regression):
    X, y = regression
    dataset = Dataset(X, label=y)
    with pytest.raises(ValueError, match="carries its own label"):
        MojoTreesRegressor(n_estimators=_ROUNDS).fit(dataset, y)


def test_sample_weight_beside_a_dataset_raises(regression):
    X, y = regression
    dataset = Dataset(X, label=y)
    with pytest.raises(ValueError, match="carries its own weights"):
        MojoTreesRegressor(n_estimators=_ROUNDS).fit(
            dataset, None, np.ones(len(y))
        )


def test_unlabeled_dataset_raises(regression):
    X, _ = regression
    with pytest.raises(ValueError, match="has no label"):
        MojoTreesRegressor(n_estimators=_ROUNDS).fit(Dataset(X))


def test_eval_set_beside_a_dataset_raises(regression):
    X, y = regression
    dataset = Dataset(X, label=y)
    with pytest.raises(ValueError, match="eval_set is not available"):
        MojoTreesRegressor(n_estimators=_ROUNDS).fit(
            dataset, eval_set=[(X, y)]
        )


def test_linear_tree_beside_a_dataset_raises(regression):
    X, y = regression
    dataset = Dataset(X, label=y)
    with pytest.raises(ValueError, match="linear_tree=True is not available"):
        MojoTreesRegressor(n_estimators=_ROUNDS, linear_tree=True).fit(
            dataset
        )


def test_unencoded_classifier_labels_raise(regression):
    """A Dataset's label is a numeric column, so the classifier takes the
    codes the trainer reads and says so when they are not."""
    X, y = regression
    dataset = Dataset(X, label=np.where(y > y.mean(), 5.0, 2.0))
    with pytest.raises(ValueError, match="class codes 0 through"):
        MojoTreesClassifier(n_estimators=_ROUNDS).fit(dataset)


def test_class_weight_beside_a_dataset_raises(binary):
    X, y = binary
    dataset = Dataset(X, label=y)
    with pytest.raises(ValueError, match="class_weight is not available"):
        MojoTreesClassifier(
            n_estimators=_ROUNDS, class_weight="balanced"
        ).fit(dataset)


def test_group_beside_a_dataset_raises(rng):
    X, labels, group = _ranking_data(rng)
    dataset = Dataset(X, label=labels, group=group)
    with pytest.raises(ValueError, match="carries its own query counts"):
        MojoTreesRanker(n_estimators=_ROUNDS).fit(dataset, group=group)
