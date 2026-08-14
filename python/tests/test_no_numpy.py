"""The stdlib fallback: everything the estimators do without numpy.

numpy is optional, and the wheel installs without it, so the plain-Python
buffer path has to stay working. These tests hide numpy from both modules
that consult it rather than requiring a numpy-free interpreter.
"""

import pickle

import pytest

import mojoboost
from mojoboost import MojoBoostClassifier, MojoBoostRegressor, _arrays


@pytest.fixture
def no_numpy(monkeypatch):
    """Run the body as if numpy were not installed."""
    monkeypatch.setattr(_arrays, "np", None)
    monkeypatch.setattr(mojoboost, "_np", None)


def _rows(n=60):
    X = [[i / n, (i % 7) / 7.0] for i in range(n)]
    y = [3.0 * r[0] + r[1] for r in X]
    return X, y


def test_regressor_round_trip(no_numpy):
    X, y = _rows()
    est = MojoBoostRegressor(n_estimators=10, min_data_in_leaf=2).fit(X, y)
    pred = est.predict(X)
    assert isinstance(pred, list) and len(pred) == len(X)
    assert est.n_features_in_ == 2
    assert est.best_iteration_ == 10
    assert est.score(X, y) > 0.5


def test_regressor_importances_are_lists(no_numpy):
    X, y = _rows()
    est = MojoBoostRegressor(n_estimators=10, min_data_in_leaf=2).fit(X, y)
    values = est.feature_importances_
    assert isinstance(values, list) and len(values) == 2
    assert sum(values) > 0


def test_classifier_labels_and_proba(no_numpy):
    X, _ = _rows()
    y = ["lo" if r[0] < 0.5 else "hi" for r in X]
    est = MojoBoostClassifier(n_estimators=10, min_data_in_leaf=2).fit(X, y)
    assert est.classes_ == ["hi", "lo"]
    proba = est.predict_proba(X)
    assert all(abs(sum(row) - 1.0) < 1e-12 for row in proba)
    pred = est.predict(X)
    assert set(pred) <= {"hi", "lo"}
    assert est.score(X, y) > 0.8


def test_multiclass_without_numpy(no_numpy):
    X, _ = _rows(90)
    y = [0 if r[0] < 0.33 else (1 if r[0] < 0.66 else 2) for r in X]
    est = MojoBoostClassifier(n_estimators=8, min_data_in_leaf=2).fit(X, y)
    assert est.n_classes_ == 3
    proba = est.predict_proba(X)
    assert len(proba) == len(X) and len(proba[0]) == 3


def test_validation_still_applies(no_numpy):
    X, y = _rows()
    with pytest.raises(ValueError, match="infinite"):
        MojoBoostRegressor(n_estimators=2).fit(
            [[float("inf"), 0.0]] + X[1:], y
        )
    with pytest.raises(ValueError, match="NaN or infinite"):
        MojoBoostRegressor(n_estimators=2).fit(X, [float("nan")] + y[1:])
    with pytest.raises(ValueError, match="all zeros"):
        MojoBoostRegressor(n_estimators=2).fit(
            X, y, sample_weight=[0.0] * len(y)
        )


def test_pickle_without_numpy(no_numpy):
    X, y = _rows()
    est = MojoBoostRegressor(n_estimators=10, min_data_in_leaf=2).fit(X, y)
    twin = pickle.loads(pickle.dumps(est))
    assert twin.predict(X) == est.predict(X)
