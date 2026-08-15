"""Fitted attributes: n_features_in_, feature_names_in_, classes_,
feature_importances_, and best_iteration_."""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRegressor, NotFittedError

FITTED_ATTRS = [
    "n_features_in_",
    "best_iteration_",
    "feature_importances_",
    "device_",
]


@pytest.mark.parametrize("attr", FITTED_ATTRS + ["classes_", "n_classes_"])
def test_fitted_attributes_are_absent_before_fit(attr):
    """scikit-learn's convention: a trailing-underscore attribute exists
    only once the estimator is fitted."""
    est = MojoTreesClassifier()
    if attr == "feature_importances_":
        with pytest.raises(NotFittedError):
            getattr(est, attr)
    else:
        assert not hasattr(est, attr)


def test_sklearn_is_fitted_hook():
    est = MojoTreesRegressor(n_estimators=2)
    assert est.__sklearn_is_fitted__() is False
    est.fit(np.array([[0.0], [1.0], [2.0]]), np.array([0.0, 1.0, 2.0]))
    assert est.__sklearn_is_fitted__() is True


def test_n_features_in(fitted_regressor, fitted_multiclass):
    assert fitted_regressor.n_features_in_ == 4
    assert fitted_multiclass.n_features_in_ == 4


def test_classes_and_n_classes(fitted_binary, fitted_multiclass):
    assert list(fitted_binary.classes_) == [0, 1]
    assert fitted_binary.n_classes_ == 2
    assert list(fitted_multiclass.classes_) == [0, 1, 2]
    assert fitted_multiclass.n_classes_ == 3


def test_classes_preserves_label_dtype(binary):
    X, y = binary
    labels = np.where(y == 1, "yes", "no")
    est = MojoTreesClassifier(n_estimators=5).fit(X, labels)
    assert list(est.classes_) == ["no", "yes"]
    assert set(est.predict(X)) <= {"no", "yes"}


def test_refit_replaces_the_fitted_state(regression, binary):
    X, y = binary
    est = MojoTreesClassifier(n_estimators=5).fit(X, np.where(y == 1, 5, 4))
    assert list(est.classes_) == [4, 5]
    est.fit(X[:, :2], y)
    assert list(est.classes_) == [0, 1]
    assert est.n_features_in_ == 2


# -- feature names -------------------------------------------------------


def test_feature_names_in_from_a_dataframe(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    est = MojoTreesRegressor(n_estimators=5).fit(frame, y)
    assert list(est.feature_names_in_) == ["a", "b", "c", "d"]
    # Names are recorded, not used: the fit is the same as on the array.
    plain = MojoTreesRegressor(n_estimators=5).fit(X, y)
    assert np.allclose(est.predict(frame), plain.predict(X))


def test_no_feature_names_without_string_columns(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    frame = pd.DataFrame(X)  # integer column labels
    est = MojoTreesRegressor(n_estimators=5).fit(frame, y)
    assert not hasattr(est, "feature_names_in_")


def test_renamed_columns_raise(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    est = MojoTreesRegressor(n_estimators=5).fit(
        pd.DataFrame(X, columns=["a", "b", "c", "d"]), y
    )
    with pytest.raises(ValueError, match="feature names"):
        est.predict(pd.DataFrame(X, columns=["a", "b", "c", "z"]))


def test_missing_or_unexpected_names_warn(regression):
    pd = pytest.importorskip("pandas")
    X, y = regression
    named = MojoTreesRegressor(n_estimators=5).fit(
        pd.DataFrame(X, columns=["a", "b", "c", "d"]), y
    )
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        named.predict(X)

    plain = MojoTreesRegressor(n_estimators=5).fit(X, y)
    with pytest.warns(UserWarning, match="fitted without feature names"):
        plain.predict(pd.DataFrame(X, columns=["a", "b", "c", "d"]))


# -- feature importances -------------------------------------------------


@pytest.mark.parametrize("importance_type", ["split", "gain"])
def test_feature_importances_shape_and_sign(regression, importance_type):
    X, y = regression
    est = MojoTreesRegressor(
        n_estimators=10, importance_type=importance_type
    ).fit(X, y)
    values = np.asarray(est.feature_importances_)
    assert values.shape == (4,)
    assert (values >= 0).all()
    assert values.sum() > 0


def test_split_importance_counts_every_split(regression):
    """Split counts sum to the number of internal nodes in the ensemble, so
    a bigger ensemble strictly increases the total."""
    X, y = regression
    small = MojoTreesRegressor(n_estimators=5).fit(X, y)
    large = MojoTreesRegressor(n_estimators=20).fit(X, y)
    assert sum(large.feature_importances_) > sum(small.feature_importances_)


def test_importance_type_is_read_at_access_time(regression):
    X, y = regression
    est = MojoTreesRegressor(n_estimators=10).fit(X, y)
    split = np.asarray(est.feature_importances_)
    est.importance_type = "gain"
    gain = np.asarray(est.feature_importances_)
    assert not np.array_equal(split, gain)
    # Split counts are integers; gains are not.
    assert np.allclose(split, np.round(split))


def test_unknown_importance_type_raises(fitted_regressor):
    fitted_regressor.importance_type = "cover"
    with pytest.raises(ValueError, match="unknown importance_type"):
        fitted_regressor.feature_importances_
    fitted_regressor.importance_type = "split"


def test_importances_are_a_copy(fitted_regressor):
    first = fitted_regressor.feature_importances_
    first[0] = 12345.0
    assert fitted_regressor.feature_importances_[0] != 12345.0


def test_multiclass_importance_covers_every_class(multiclass):
    X, y = multiclass
    est = MojoTreesClassifier(n_estimators=10).fit(X, y)
    values = np.asarray(est.feature_importances_)
    assert values.shape == (4,)
    assert values.sum() > 0


# -- best_iteration_ -----------------------------------------------------


def test_best_iteration_matches_n_estimators(regression):
    X, y = regression
    est = MojoTreesRegressor(n_estimators=17).fit(X, y)
    assert est.best_iteration_ == 17


def test_best_iteration_reports_an_early_stop(regression):
    """A penalty larger than any gradient sum zeroes the root value, so
    training stops with an empty ensemble instead of running the full
    n_estimators rounds."""
    X, y = regression
    est = MojoTreesRegressor(n_estimators=30, lambda_l1=1e6).fit(X, y)
    assert est.best_iteration_ == 0


def test_multiclass_best_iteration_counts_rounds_not_trees(multiclass):
    """A multiclass round grows one tree per class; best_iteration_ counts
    rounds, as LightGBM's does."""
    X, y = multiclass
    est = MojoTreesClassifier(n_estimators=8).fit(X, y)
    assert est.best_iteration_ == 8
