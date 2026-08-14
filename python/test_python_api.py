"""End-to-end tests for the mojoboost Python API.

Runs with or without numpy (the wrapper falls back to stdlib buffers).
Usage: build the extension with bindings/build.sh, then
    pixi run python python/test_python_api.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mojoboost import MojoBoostClassifier, MojoBoostRegressor


def _rand_stream(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield (state >> 11) / 9007199254740992.0


def make_regression(n_rows, n_features=4, seed=7):
    rng = _rand_stream(seed)
    X = [[next(rng) for _ in range(n_features)] for _ in range(n_rows)]
    y = [
        3.0 * r[0] + 2.0 * r[1] * r[2] + 0.05 * (next(rng) - 0.5) for r in X
    ]
    return X, y


def make_classification(n_rows, n_classes, n_features=4, seed=11):
    rng = _rand_stream(seed)
    X = [[next(rng) for _ in range(n_features)] for _ in range(n_rows)]
    y = [
        min(n_classes - 1, int((r[0] + 0.3 * r[1]) / 1.3 * n_classes))
        for r in X
    ]
    return X, y


def test_regressor():
    X, y = make_regression(800)
    model = MojoBoostRegressor(n_estimators=50)
    assert model.fit(X, y) is model
    pred = model.predict(X)
    mse = sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y)
    assert mse < 0.01, f"regressor train MSE too high: {mse}"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "reg.mbst")
        model.save(path)
        loaded = MojoBoostRegressor.load(path)
        pred2 = loaded.predict(X)
    assert all(a == b for a, b in zip(pred, pred2)), "round-trip not exact"
    print(f"regressor ok (train MSE {mse:.5f})")


def test_binary_classifier():
    X, y = make_classification(800, 2)
    model = MojoBoostClassifier(n_estimators=50).fit(X, y)
    assert model.n_classes_ == 2
    pred = list(model.predict(X))
    acc = sum(int(p == t) for p, t in zip(pred, y)) / len(y)
    assert acc > 0.95, f"binary accuracy too low: {acc}"
    proba = model.predict_proba(X)
    for row in list(proba)[:20]:
        assert abs(row[0] + row[1] - 1.0) < 1e-12

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "bin.mbst")
        model.save(path)
        loaded = MojoBoostClassifier.load(path)
        assert loaded.n_classes_ == 2
        assert list(loaded.predict(X)) == pred, "round-trip not exact"
    print(f"binary classifier ok (train acc {acc:.3f})")


def test_multiclass_classifier():
    X, y = make_classification(900, 3)
    model = MojoBoostClassifier(n_estimators=30).fit(X, y)
    assert model.n_classes_ == 3
    pred = list(model.predict(X))
    acc = sum(int(p == t) for p, t in zip(pred, y)) / len(y)
    assert acc > 0.9, f"multiclass accuracy too low: {acc}"
    proba = model.predict_proba(X)
    for row in list(proba)[:20]:
        assert abs(sum(row) - 1.0) < 1e-9

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "multi.mbst")
        model.save(path)
        loaded = MojoBoostClassifier.load(path)
        assert loaded.n_classes_ == 3
        assert list(loaded.predict(X)) == pred, "round-trip not exact"
    print(f"multiclass classifier ok (train acc {acc:.3f})")


def test_sample_weight():
    # (Weight-2 == duplicated-row exactness is tested at the Mojo level on
    # pre-binned data; through fit() an extra row shifts the quantile bin
    # edges, so here we assert the properties that hold at this level.)
    X, y = make_regression(300)

    # All-ones weights must be bit-identical to no weights.
    plain = MojoBoostRegressor(n_estimators=20).fit(X, y)
    ones = MojoBoostRegressor(n_estimators=20).fit(
        X, y, sample_weight=[1.0] * len(y)
    )
    pa = list(plain.predict(X[:50]))
    pb = list(ones.predict(X[:50]))
    assert all(a == b for a, b in zip(pa, pb)), "unit weights changed the fit"

    # Non-uniform weights must change the fit.
    w = [10.0 if r[0] > 0.5 else 0.1 for r in X]
    skewed = MojoBoostRegressor(n_estimators=20).fit(X, y, sample_weight=w)
    pc = list(skewed.predict(X[:50]))
    assert any(a != c for a, c in zip(pa, pc)), "weights had no effect"
    print("sample_weight ok")


def test_input_validation():
    model = MojoBoostRegressor()
    try:
        model.predict([[1.0]])
        raise AssertionError("predict before fit should raise")
    except RuntimeError:
        pass
    try:
        MojoBoostClassifier().fit([[1.0], [2.0]], [1, 3])
        raise AssertionError("gappy labels should raise")
    except ValueError:
        pass
    print("validation ok")


if __name__ == "__main__":
    try:
        import numpy

        print(f"numpy {numpy.__version__}")
    except ImportError:
        print("numpy not available, using stdlib fallback")
    test_regressor()
    test_binary_classifier()
    test_multiclass_classifier()
    test_sample_weight()
    test_input_validation()
    print("all python API tests passed")
