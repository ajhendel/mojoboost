"""Input validation: dtype, shape, finiteness, labels, and weights."""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRegressor, NotFittedError

X_OK = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.75]])
Y_OK = np.array([0.0, 1.0, 0.5, 0.25])
LABELS_OK = np.array([0, 1, 0, 1])


def _fit(cls=MojoTreesRegressor, **kwargs):
    est = cls(n_estimators=2, min_data_in_leaf=1)
    y = Y_OK if cls is MojoTreesRegressor else LABELS_OK
    return est.fit(kwargs.pop("X", X_OK), kwargs.pop("y", y), **kwargs)


# -- X ------------------------------------------------------------------


def test_x_must_be_two_dimensional():
    with pytest.raises(ValueError, match="2-dimensional"):
        _fit(X=np.array([1.0, 2.0, 3.0, 4.0]))


def test_one_dimensional_x_error_suggests_reshaping():
    with pytest.raises(ValueError, match=r"reshape"):
        _fit(X=np.array([1.0, 2.0, 3.0, 4.0]))


def test_x_must_have_rows():
    with pytest.raises(ValueError, match="at least one row"):
        _fit(X=np.empty((0, 2)), y=np.empty(0))


def test_x_must_have_features():
    with pytest.raises(ValueError, match="at least one feature"):
        _fit(X=np.empty((4, 0)))


def test_x_rejects_infinities():
    bad = X_OK.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        _fit(X=bad)


def test_x_allows_nan_as_the_missing_marker():
    """NaN is mojotrees's missing value (src/mojotrees/binning.mojo), so it
    must survive validation, unlike an infinity."""
    with_nan = X_OK.copy()
    with_nan[0, 0] = np.nan
    est = _fit(X=with_nan)
    assert np.isfinite(est.predict(with_nan)).all()


def test_x_rejects_non_numeric_dtype():
    with pytest.raises(ValueError, match="float64"):
        _fit(X=np.array([["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]))


def test_x_accepts_sparse_input():
    sparse = pytest.importorskip("scipy.sparse")
    est = _fit(X=sparse.csr_matrix(X_OK))
    assert np.isfinite(est.predict(sparse.csr_matrix(X_OK))).all()


def test_ragged_rows_are_rejected():
    with pytest.raises(ValueError):
        _fit(X=[[1.0, 2.0], [3.0], [4.0, 5.0], [6.0, 7.0]])


# -- y ------------------------------------------------------------------


def test_y_length_must_match_x():
    with pytest.raises(ValueError, match="one entry per row"):
        _fit(y=np.array([1.0, 2.0]))


def test_regression_target_must_be_finite():
    for bad in (np.nan, np.inf):
        y = Y_OK.copy()
        y[1] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            _fit(y=y)


def test_classifier_needs_at_least_two_classes():
    with pytest.raises(ValueError, match="at least 2 classes"):
        _fit(MojoTreesClassifier, y=np.zeros(4, dtype=np.int64))


def test_classifier_rejects_mixed_label_types():
    with pytest.raises(ValueError, match="comparable"):
        _fit(MojoTreesClassifier, y=np.array([0, "a", 1, "b"], dtype=object))


def test_classifier_rejects_non_finite_numeric_labels():
    with pytest.raises(ValueError, match="NaN or infinite"):
        _fit(MojoTreesClassifier, y=np.array([0.0, 1.0, np.nan, 1.0]))


def test_classifier_accepts_gappy_and_string_labels():
    """Labels of any comparable type work; only `classes_` remembers them."""
    gappy = _fit(MojoTreesClassifier, y=np.array([3, 7, 3, 7]))
    assert list(gappy.classes_) == [3, 7]
    assert set(gappy.predict(X_OK)) <= {3, 7}

    strings = _fit(MojoTreesClassifier, y=np.array(["no", "yes", "no", "yes"]))
    assert list(strings.classes_) == ["no", "yes"]
    assert set(strings.predict(X_OK)) <= {"no", "yes"}


# -- sample_weight -------------------------------------------------------


def test_sample_weight_length_must_match():
    with pytest.raises(ValueError, match="sample_weight"):
        _fit(sample_weight=np.ones(3))


def test_sample_weight_must_be_finite():
    with pytest.raises(ValueError, match="NaN or infinite"):
        _fit(sample_weight=np.array([1.0, np.nan, 1.0, 1.0]))


def test_sample_weight_must_be_nonnegative():
    with pytest.raises(ValueError, match="nonnegative"):
        _fit(sample_weight=np.array([1.0, -1.0, 1.0, 1.0]))


def test_sample_weight_must_not_be_all_zeros():
    with pytest.raises(ValueError, match="all zeros"):
        _fit(sample_weight=np.zeros(4))


def test_zero_weights_are_allowed_when_some_row_survives():
    est = _fit(sample_weight=np.array([0.0, 1.0, 0.0, 1.0]))
    assert est.n_features_in_ == 2


# -- predict-time checks -------------------------------------------------


def test_predict_before_fit_raises_not_fitted():
    est = MojoTreesRegressor()
    with pytest.raises(NotFittedError):
        est.predict(X_OK)


def test_not_fitted_error_is_still_a_runtime_error():
    """mojotrees raised RuntimeError before it had a named exception, and
    callers written against that keep working."""
    assert issubclass(NotFittedError, RuntimeError)
    with pytest.raises(RuntimeError):
        MojoTreesRegressor().predict(X_OK)


def test_predict_rejects_a_different_feature_count(fitted_regressor):
    with pytest.raises(ValueError, match="expecting 4 features"):
        fitted_regressor.predict(np.zeros((3, 5)))


def test_predict_validates_like_fit(fitted_regressor):
    bad = np.zeros((3, 4))
    bad[0, 0] = -np.inf
    with pytest.raises(ValueError, match="infinite"):
        fitted_regressor.predict(bad)


def test_failed_refit_clears_the_previous_model(regression):
    X, y = regression
    est = MojoTreesRegressor(n_estimators=3).fit(X, y)
    with pytest.raises(ValueError):
        est.fit(X, np.full(len(y), np.nan))
    with pytest.raises(NotFittedError):
        est.predict(X)
