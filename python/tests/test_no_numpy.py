"""The stdlib fallback: everything the estimators do without numpy.

numpy is optional, and the wheel installs without it, so the plain-Python
buffer path has to stay working. These tests hide numpy from both modules
that consult it rather than requiring a numpy-free interpreter.
"""

import importlib
import pkgutil
import sys

import pickle

import pytest

import mojoboost
from mojoboost import MojoBoostClassifier, MojoBoostRegressor, _arrays

#: Modules that read numpy through a `_np = _arrays.np` snapshot taken at
#: import. Patching `_arrays.np` does not reach a snapshot, so each one
#: has to be patched by name.
_SNAPSHOT_ATTR = "_np"


def _import_every_submodule():
    """Import the whole package so no snapshot is taken after patching.

    A module imported inside a test body would snapshot the real numpy
    after the fixture had already run and would silently keep it.
    Modules that need an absent optional dependency are skipped: they
    cannot hold a snapshot if they cannot be imported.
    """
    for info in pkgutil.iter_modules(mojoboost.__path__):
        try:
            importlib.import_module("mojoboost." + info.name)
        except ImportError:
            continue


_import_every_submodule()


def _snapshot_modules():
    """Every imported mojoboost module holding an `_np` snapshot.

    Discovered rather than listed, so a module that starts snapshotting
    numpy is covered the day it is written instead of the day someone
    remembers this file.
    """
    return [
        module
        for name, module in sorted(sys.modules.items())
        if name == "mojoboost" or name.startswith("mojoboost.")
        if getattr(module, _SNAPSHOT_ATTR, None) is not None
    ]


@pytest.fixture
def no_numpy(monkeypatch):
    """Run the body as if numpy were not installed.

    Patching `_arrays` alone is not enough and is worse than doing
    nothing: the shared buffer helpers switch to the stdlib `array`
    while every `_np` snapshot still points at the real numpy. Under
    that mismatch the numpy-free branches in `basic.py` were not merely
    untested, they were unreachable, because reaching one meant calling
    a numpy method on an `array.array`.
    """
    monkeypatch.setattr(_arrays, "np", None)
    for module in _snapshot_modules():
        monkeypatch.setattr(module, _SNAPSHOT_ATTR, None)


def _rows(n=60):
    X = [[i / n, (i % 7) / 7.0] for i in range(n)]
    y = [3.0 * r[0] + r[1] for r in X]
    return X, y


def test_the_fixture_reaches_every_snapshot(no_numpy):
    """Guard the fixture itself.

    `basic` is named explicitly because it is the module whose numpy-free
    branches this suite exists to run, and it went uncovered for as long
    as the fixture patched only the package. If discovery ever stops
    finding it, that has to fail here rather than quietly downgrade every
    other test in this file back to the numpy path.
    """
    from mojoboost import basic

    assert basic._np is None
    assert mojoboost._np is None
    assert _arrays.np is None
    assert _snapshot_modules() == []


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


# The estimators above route around the Booster, so on their own they
# leave every numpy-optional branch in basic.py unrun. These drive the
# low-level API, which is the one that actually forks on `_np`.


def test_booster_multiclass_predict_returns_rows_of_lists(no_numpy):
    X, _ = _rows(90)
    y = [0.0 if r[0] < 0.33 else (1.0 if r[0] < 0.66 else 2.0) for r in X]
    booster = mojoboost.train(
        {
            "objective": "multiclass",
            "num_class": 3,
            "min_data_in_leaf": 2,
        },
        mojoboost.Dataset(X, label=y),
        num_boost_round=8,
    )
    proba = booster.predict(X)
    assert isinstance(proba, list) and len(proba) == len(X)
    assert all(isinstance(row, list) and len(row) == 3 for row in proba)
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in proba)


def test_sparse_input_is_refused_rather_than_served(no_numpy):
    """Sparse input needs numpy, and says so.

    This is what makes the numpy-free half of the multiclass fork in
    `_predict_sparse` unreachable: `check_X_sparse` refuses first. The
    assertion is here so that if sparse input ever grows a numpy-free
    path, this fails and the fallback gets written deliberately instead
    of being assumed to already exist.
    """
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _rows()
    booster = mojoboost.train(
        {"objective": "regression", "min_data_in_leaf": 2},
        mojoboost.Dataset(X, label=y),
        num_boost_round=8,
    )
    with pytest.raises(TypeError, match="needs numpy"):
        booster.predict(sparse.csr_matrix(X))


def test_booster_eval_scores_both_widths(no_numpy):
    """The width==1 and width>1 halves of the fallback flatten differ."""
    X, y = _rows()
    data = mojoboost.Dataset(X, label=y)
    booster = mojoboost.train(
        {"objective": "regression", "min_data_in_leaf": 2},
        data,
        num_boost_round=10,
    )
    scored = booster.eval(data, "train")
    assert scored and scored[0][0] == "train"
    assert scored[0][2] >= 0.0

    labels = [0.0 if r[0] < 0.5 else 1.0 for r in X]
    wide_data = mojoboost.Dataset(X, label=labels)
    wide = mojoboost.train(
        {
            "objective": "multiclass",
            "num_class": 2,
            "min_data_in_leaf": 2,
        },
        wide_data,
        num_boost_round=8,
    )
    wide_scored = wide.eval(wide_data, "train")
    assert wide_scored and wide_scored[0][2] >= 0.0


def test_booster_importance_is_a_fresh_list(no_numpy):
    X, y = _rows()
    booster = mojoboost.train(
        {"objective": "regression", "min_data_in_leaf": 2},
        mojoboost.Dataset(X, label=y),
        num_boost_round=10,
    )
    values = booster.feature_importance("split")
    assert isinstance(values, list) and len(values) == 2
    # The cache must hand out a copy, not the entry itself.
    values[0] = -1.0
    assert booster.feature_importance("split")[0] != -1.0


def test_binary_label_check_names_the_offender(no_numpy):
    X, _ = _rows()
    labels = [0.0 if r[0] < 0.5 else 1.0 for r in X]
    labels[7] = 2.0
    with pytest.raises(ValueError, match=r"labels in \{0, 1\}"):
        mojoboost.train(
            {"objective": "binary", "min_data_in_leaf": 2},
            mojoboost.Dataset(X, label=labels),
            num_boost_round=4,
        )
