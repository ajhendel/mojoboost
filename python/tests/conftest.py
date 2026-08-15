"""Shared fixtures for the pytest suite.

Run from the repository root with `pixi run -e pytest test-python`, or
against an installed wheel with `pytest python/tests` from anywhere. The
`sys.path` insertion below only matters in the first case: it puts the
source tree's `python/` directory ahead of site-packages so the tests
exercise the working copy.

python/test_python_api.py is the other suite: a dependency-free script that
covers training behavior (objectives, weights, devices, regularization).
These tests cover the estimator surface instead, and use pytest, numpy, and
optionally scikit-learn and pandas.
"""

import os
import sys

import pytest

_PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_PYTHON_DIR, "mojotrees")):
    sys.path.insert(0, _PYTHON_DIR)

np = pytest.importorskip("numpy")


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


def _regression_data(n_rows=300, n_features=4, seed=7):
    gen = np.random.default_rng(seed)
    X = gen.random((n_rows, n_features))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] * X[:, 2] + 0.05 * gen.standard_normal(
        n_rows
    )
    return X, y


def _classification_data(n_rows=300, n_classes=2, n_features=4, seed=11):
    gen = np.random.default_rng(seed)
    X = gen.random((n_rows, n_features))
    score = X[:, 0] + 0.3 * X[:, 1]
    edges = np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1])
    y = np.searchsorted(edges, score)
    return X, y.astype(np.int64)


@pytest.fixture(scope="session")
def regression():
    """(X, y) for a 4-feature regression problem."""
    return _regression_data()


@pytest.fixture(scope="session")
def binary():
    """(X, y) with labels 0 and 1."""
    return _classification_data(n_classes=2)


@pytest.fixture(scope="session")
def multiclass():
    """(X, y) with labels 0, 1, and 2."""
    return _classification_data(n_rows=360, n_classes=3)


@pytest.fixture(scope="session")
def fitted_regressor(regression):
    from mojotrees import MojoTreesRegressor

    X, y = regression
    return MojoTreesRegressor(n_estimators=20).fit(X, y)


@pytest.fixture(scope="session")
def fitted_binary(binary):
    from mojotrees import MojoTreesClassifier

    X, y = binary
    return MojoTreesClassifier(n_estimators=20).fit(X, y)


@pytest.fixture(scope="session")
def fitted_multiclass(multiclass):
    from mojotrees import MojoTreesClassifier

    X, y = multiclass
    return MojoTreesClassifier(n_estimators=15).fit(X, y)
