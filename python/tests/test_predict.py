"""predict/predict_proba consistency and score()."""

import numpy as np
import pytest

from mojoboost import MojoBoostClassifier, MojoBoostRegressor


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
    plain = MojoBoostClassifier(n_estimators=10).fit(X, y)
    renamed = MojoBoostClassifier(n_estimators=10).fit(
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
    est = MojoBoostRegressor(n_estimators=3, min_data_in_leaf=1).fit(X, y)
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
    est = MojoBoostClassifier(n_estimators=10).fit(X, labels)
    assert est.score(X, labels) == pytest.approx(
        (np.asarray(est.predict(X)) == labels).mean()
    )


def test_score_checks_the_target_length(fitted_regressor, regression):
    X, y = regression
    with pytest.raises(ValueError):
        fitted_regressor.score(X, y[:-1])
