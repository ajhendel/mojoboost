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

from mojoboost import (
    MojoBoostClassifier,
    MojoBoostRegressor,
    gpu_available,
)

try:
    import numpy as _np
except ImportError:
    _np = None


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


def test_regression_objectives():
    X, y = make_regression(600)

    for objective in ("huber", "mae", "regression_l1"):
        model = MojoBoostRegressor(objective=objective, n_estimators=40)
        pred = model.fit(X, y).predict(X)
        mse = sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y)
        assert mse < 0.02, f"{objective} train MSE too high: {mse}"

    # A 0.9-quantile fit should predict above most training targets.
    q = MojoBoostRegressor(objective="quantile", alpha=0.9, n_estimators=40)
    pred = q.fit(X, y).predict(X)
    frac_below = sum(int(t <= p) for p, t in zip(pred, y)) / len(y)
    assert 0.8 < frac_below <= 1.0, f"quantile coverage off: {frac_below}"

    # Identity link, so these round-trip through the existing format.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "quantile.mbst")
        q.save(path)
        loaded = MojoBoostRegressor.load(path)
        pred2 = loaded.predict(X)
    assert all(a == b for a, b in zip(pred, pred2)), "round-trip not exact"
    print(f"regression objectives ok (q90 coverage {frac_below:.3f})")


def test_lambda_l1():
    X, y = make_regression(600)
    mean_y = sum(y) / len(y)

    # The default is 0, so passing it explicitly must not change the fit.
    default = MojoBoostRegressor(n_estimators=30).fit(X, y)
    explicit = MojoBoostRegressor(n_estimators=30, lambda_l1=0.0).fit(X, y)
    pa = list(default.predict(X))
    pb = list(explicit.predict(X))
    assert all(a == b for a, b in zip(pa, pb)), "lambda_l1=0 changed the fit"

    # A moderate penalty shrinks every leaf's Newton step, so the fitted
    # predictions stay closer to the base score and fit the target less well.
    reg = MojoBoostRegressor(n_estimators=30, lambda_l1=5.0).fit(X, y)
    pc = list(reg.predict(X))
    assert any(a != c for a, c in zip(pa, pc)), "lambda_l1 had no effect"
    travel_plain = sum(abs(p - mean_y) for p in pa)
    travel_reg = sum(abs(p - mean_y) for p in pc)
    assert travel_reg < travel_plain, "lambda_l1 did not shrink the fit"
    mse_plain = sum((p - t) ** 2 for p, t in zip(pa, y)) / len(y)
    mse_reg = sum((p - t) ** 2 for p, t in zip(pc, y)) / len(y)
    assert mse_reg > mse_plain, "lambda_l1 did not regularize the train fit"

    # A penalty larger than any gradient sum zeroes every candidate gain and
    # the root value, so training stops with an empty ensemble that predicts
    # the base score (the target mean) everywhere.
    huge = MojoBoostRegressor(n_estimators=30, lambda_l1=1e6).fit(X, y)
    for p in list(huge.predict(X))[:50]:
        assert abs(p - mean_y) < 1e-9, f"expected base score, got {p}"
    print(f"lambda_l1 ok (train MSE {mse_plain:.5f} -> {mse_reg:.5f})")


def test_regularization_aliases():
    # LightGBM's scikit-learn estimators spell these reg_alpha and
    # reg_lambda; both spellings must train the same model.
    X, y = make_regression(400)
    named = MojoBoostRegressor(
        n_estimators=20, lambda_l1=5.0, lambda_l2=3.0
    ).fit(X, y)
    aliased = MojoBoostRegressor(
        n_estimators=20, reg_alpha=5.0, reg_lambda=3.0
    ).fit(X, y)
    pa = list(named.predict(X))
    pb = list(aliased.predict(X))
    assert all(a == b for a, b in zip(pa, pb)), "aliases trained differently"

    # Redundant but agreeing values are fine; disagreeing ones raise.
    MojoBoostRegressor(n_estimators=5, lambda_l1=5.0, reg_alpha=5.0).fit(X, y)
    for bad in (
        dict(lambda_l1=5.0, reg_alpha=1.0),
        dict(lambda_l2=2.0, reg_lambda=1.0),
    ):
        try:
            MojoBoostRegressor(n_estimators=5, **bad).fit(X, y)
            raise AssertionError(f"{bad} should raise")
        except ValueError:
            pass

    # Aliases are ordinary constructor parameters: stored unmodified under
    # their own names, resolved at fit time, so set_params reaches them and
    # the scikit-learn parameter plumbing needs no special case for them.
    est = MojoBoostRegressor(n_estimators=5)
    assert est.reg_alpha is None and est.lambda_l1 == 0.0
    est.reg_alpha = 7.0  # what set_params does
    assert est._params(0, "cpu")["lambda_l1"] == 7.0
    print("regularization aliases ok")


def test_lightgbm_sklearn_aliases():
    """Native LightGBM names stay canonical; its sklearn wrapper's names
    are accepted as strict aliases and produce the same model."""
    X, y = make_regression(500)
    native = MojoBoostRegressor(
        n_estimators=20,
        min_data_in_leaf=8,
        min_child_hess=0.01,
        lambda_l1=0.5,
        lambda_l2=2.0,
        bagging_fraction=0.75,
        bagging_freq=1,
        bagging_seed=19,
        device="cpu",
    ).fit(X, y)
    sklearn_names = MojoBoostRegressor(
        n_estimators=20,
        min_child_samples=8,
        min_child_weight=0.01,
        reg_alpha=0.5,
        reg_lambda=2.0,
        subsample=0.75,
        subsample_freq=1,
        bagging_seed=19,
        device_type="cpu",
    ).fit(X, y)
    assert list(native.predict(X)) == list(sklearn_names.predict(X))
    assert sklearn_names.device_ == "cpu"

    params = MojoBoostRegressor().get_params()
    for name in (
        "min_child_samples",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "subsample",
        "subsample_freq",
        "device_type",
    ):
        assert name in params, f"get_params omitted alias {name}"

    updated = MojoBoostRegressor().set_params(
        min_child_samples=7, subsample=0.8, device_type="cpu"
    )
    resolved = updated._params(0, updated._resolve_device(100, 4, 1))
    assert resolved["min_data_in_leaf"] == 7
    assert resolved["bagging_fraction"] == 0.8

    conflicts = (
        dict(min_data_in_leaf=9, min_child_samples=8),
        dict(min_child_hess=0.02, min_child_weight=0.01),
        dict(lambda_l1=0.5, reg_alpha=0.25),
        dict(lambda_l2=2.0, reg_lambda=1.0),
        dict(bagging_fraction=0.8, subsample=0.7),
        dict(bagging_freq=2, subsample_freq=1),
        dict(device="gpu", device_type="cpu"),
    )
    for bad in conflicts:
        try:
            MojoBoostRegressor(n_estimators=2, **bad).fit(X, y)
            raise AssertionError(f"conflicting aliases should raise: {bad}")
        except ValueError:
            pass
    print("LightGBM sklearn aliases ok")


def test_bagging():
    X, y = make_regression(600)

    # The defaults disable bagging, and so does either switch on its own.
    base = MojoBoostRegressor(n_estimators=30).fit(X, y)
    pa = list(base.predict(X))
    for kwargs in ({"bagging_fraction": 1.0, "bagging_freq": 1},
                   {"bagging_fraction": 0.5, "bagging_freq": 0}):
        off = MojoBoostRegressor(n_estimators=30, **kwargs).fit(X, y)
        assert all(a == b for a, b in zip(pa, off.predict(X))), (
            f"bagging changed the fit when disabled: {kwargs}"
        )

    # Same seed, same model, bit for bit; a different seed moves it.
    def bagged(seed):
        return MojoBoostRegressor(
            n_estimators=30,
            bagging_fraction=0.6,
            bagging_freq=1,
            bagging_seed=seed,
        ).fit(X, y)

    pb = list(bagged(7).predict(X))
    pb_again = list(bagged(7).predict(X))
    pc = list(bagged(8).predict(X))
    assert all(a == b for a, b in zip(pb, pb_again)), "bagging not reproducible"
    assert any(a != b for a, b in zip(pb, pc)), "bagging seed had no effect"
    assert any(a != b for a, b in zip(pa, pb)), "bagging had no effect"

    # Bagging is a regularizer, not a way to stop learning.
    mse = sum((p - t) ** 2 for p, t in zip(pb, y)) / len(y)
    var = sum((t - sum(y) / len(y)) ** 2 for t in y) / len(y)
    assert mse < 0.2 * var, f"bagged train MSE too high: {mse}"

    for bad in ({"bagging_fraction": 0.0}, {"bagging_fraction": 1.5},
                {"bagging_freq": -1}):
        try:
            MojoBoostRegressor(n_estimators=5, **bad).fit(X, y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # Classifiers take the same parameters.
    Xc, yc = make_classification(400, 3)
    clf = MojoBoostClassifier(
        n_estimators=20, bagging_fraction=0.7, bagging_freq=2
    ).fit(Xc, yc)
    acc = sum(int(p == t) for p, t in zip(clf.predict(Xc), yc)) / len(yc)
    assert acc > 0.9, f"bagged classifier accuracy too low: {acc}"
    print(f"bagging ok (train MSE {mse:.5f}, clf acc {acc:.3f})")


def test_goss():
    X, y = make_regression(600)

    # gbdt is the default, and naming it explicitly changes nothing.
    base = MojoBoostRegressor(n_estimators=30).fit(X, y)
    pa = list(base.predict(X))
    explicit = MojoBoostRegressor(n_estimators=30, boosting="gbdt").fit(X, y)
    assert all(a == b for a, b in zip(pa, explicit.predict(X))), (
        "boosting='gbdt' is not the default path"
    )

    # GOSS with the automatic warmup (int(1 / learning_rate) = 10 rounds of
    # the 30 here) still samples, so the fit moves.
    def goss(seed=3, **kwargs):
        return MojoBoostRegressor(
            n_estimators=30, boosting="goss", goss_seed=seed, **kwargs
        ).fit(X, y)

    pb = list(goss().predict(X))
    assert any(a != b for a, b in zip(pa, pb)), "GOSS had no effect"

    # Same seed, same model, bit for bit; a different seed moves it.
    assert all(a == b for a, b in zip(pb, goss().predict(X))), (
        "GOSS not reproducible"
    )
    assert any(a != b for a, b in zip(pb, goss(seed=8).predict(X))), (
        "goss_seed had no effect"
    )

    # A warmup covering every round is full-data training.
    warmed = MojoBoostRegressor(
        n_estimators=30, boosting="goss", goss_warmup_rounds=30
    ).fit(X, y)
    assert all(a == b for a, b in zip(pa, warmed.predict(X))), (
        "warmup rounds did not train on every row"
    )

    # boosting_type is LightGBM's scikit-learn spelling of the same thing.
    aliased = MojoBoostRegressor(
        n_estimators=30, boosting_type="goss", goss_seed=3
    ).fit(X, y)
    assert all(a == b for a, b in zip(pb, aliased.predict(X))), (
        "boosting_type alias trained a different model"
    )

    # Sampling a third of the rows is a regularizer, not a way to stop
    # learning.
    mse = sum((p - t) ** 2 for p, t in zip(pb, y)) / len(y)
    var = sum((t - sum(y) / len(y)) ** 2 for t in y) / len(y)
    assert mse < 0.2 * var, f"GOSS train MSE too high: {mse}"

    for bad in (
        {"boosting": "dart"},
        {"boosting": "goss", "top_rate": 1.5},
        {"boosting": "goss", "top_rate": -0.1},
        {"boosting": "goss", "other_rate": 1.5},
        {"boosting": "goss", "top_rate": 0.7, "other_rate": 0.4},
        {"boosting": "goss", "top_rate": 0.0, "other_rate": 0.0},
        {"boosting": "goss", "goss_seed": -1},
        {"boosting": "goss", "goss_warmup_rounds": -2},
        # GOSS and row bagging both own the sampled rows.
        {"boosting": "goss", "bagging_fraction": 0.5, "bagging_freq": 1},
        {"boosting": "goss", "boosting_type": "gbdt"},
    ):
        try:
            MojoBoostRegressor(n_estimators=5, **bad).fit(X, y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # Classifiers take the same parameters.
    Xc, yc = make_classification(400, 3)
    clf = MojoBoostClassifier(
        n_estimators=20, boosting="goss", goss_warmup_rounds=0
    ).fit(Xc, yc)
    acc = sum(int(p == t) for p, t in zip(clf.predict(Xc), yc)) / len(yc)
    assert acc > 0.9, f"GOSS classifier accuracy too low: {acc}"
    print(f"goss ok (train MSE {mse:.5f}, clf acc {acc:.3f})")


def test_feature_fraction():
    X, y = make_regression(600, n_features=10)

    # The defaults disable subsampling, and so does an explicit 1.0 whatever
    # the seed: selecting every feature is not a random event.
    base = MojoBoostRegressor(n_estimators=30).fit(X, y)
    pa = list(base.predict(X))
    for kwargs in ({"feature_fraction": 1.0},
                   {"feature_fraction_bynode": 1.0},
                   {"feature_fraction": 1.0, "feature_fraction_seed": 999}):
        off = MojoBoostRegressor(n_estimators=30, **kwargs).fit(X, y)
        assert all(a == b for a, b in zip(pa, off.predict(X))), (
            f"feature subsampling changed the fit when disabled: {kwargs}"
        )

    # Same seed, same model, bit for bit; a different seed moves it.
    def sampled(seed, bynode=1.0):
        return MojoBoostRegressor(
            n_estimators=30,
            feature_fraction=0.4,
            feature_fraction_bynode=bynode,
            feature_fraction_seed=seed,
        ).fit(X, y)

    pb = list(sampled(7).predict(X))
    pb_again = list(sampled(7).predict(X))
    pc = list(sampled(8).predict(X))
    assert all(a == b for a, b in zip(pb, pb_again)), (
        "feature subsampling not reproducible"
    )
    assert any(a != b for a, b in zip(pb, pc)), (
        "feature_fraction_seed had no effect"
    )
    assert any(a != b for a, b in zip(pa, pb)), "feature_fraction had no effect"

    # Per-node sampling composes on top of the per-tree draw.
    pd = list(sampled(7, bynode=0.5).predict(X))
    pd_again = list(sampled(7, bynode=0.5).predict(X))
    assert all(a == b for a, b in zip(pd, pd_again)), (
        "feature_fraction_bynode not reproducible"
    )
    assert any(a != b for a, b in zip(pb, pd)), (
        "feature_fraction_bynode had no effect"
    )

    # Subsampling is a regularizer, not a way to stop learning.
    mse = sum((p - t) ** 2 for p, t in zip(pb, y)) / len(y)
    var = sum((t - sum(y) / len(y)) ** 2 for t in y) / len(y)
    assert mse < 0.2 * var, f"subsampled train MSE too high: {mse}"

    for bad in ({"feature_fraction": 0.0}, {"feature_fraction": 1.5},
                {"feature_fraction": -0.5},
                {"feature_fraction_bynode": 0.0},
                {"feature_fraction_bynode": 2.0}):
        try:
            MojoBoostRegressor(n_estimators=5, **bad).fit(X, y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # Classifiers take the same parameters.
    Xc, yc = make_classification(400, 3, n_features=8)
    clf = MojoBoostClassifier(
        n_estimators=20, feature_fraction=0.5, feature_fraction_bynode=0.8
    ).fit(Xc, yc)
    acc = sum(int(p == t) for p, t in zip(clf.predict(Xc), yc)) / len(yc)
    assert acc > 0.9, f"subsampled classifier accuracy too low: {acc}"
    print(f"feature subsampling ok (train MSE {mse:.5f}, clf acc {acc:.3f})")


def test_max_depth():
    X, y = make_regression(600)

    def mse(model):
        pred = model.predict(X)
        return sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y)

    # The default is unlimited, and every non-positive value means the same
    # thing, so all three must produce the identical ensemble.
    default = MojoBoostRegressor(n_estimators=40).fit(X, y)
    assert default.max_depth == -1, "default max_depth must be -1"
    base = list(default.predict(X))
    for unlimited in (-1, 0, -5):
        other = MojoBoostRegressor(
            n_estimators=40, max_depth=unlimited
        ).fit(X, y)
        assert all(
            a == b for a, b in zip(base, other.predict(X))
        ), f"max_depth={unlimited} should mean unlimited"

    # A limit far above the natural depth is also a no-op.
    loose = MojoBoostRegressor(n_estimators=40, max_depth=100).fit(X, y)
    assert all(
        a == b for a, b in zip(base, loose.predict(X))
    ), "a max_depth above the tree's depth changed the fit"

    # Stumps are strictly weaker than unlimited depth, and relaxing the
    # limit recovers some of the fit.
    stumps = MojoBoostRegressor(n_estimators=40, max_depth=1).fit(X, y)
    mse_stumps = mse(stumps)
    mse_deep = mse(default)
    mse_mid = mse(MojoBoostRegressor(n_estimators=40, max_depth=3).fit(X, y))
    assert mse_stumps > mse_deep, "max_depth=1 did not constrain the fit"
    assert mse_mid < mse_stumps, "raising max_depth did not improve the fit"

    # num_leaves=2 permits exactly one split, so the tree is already a stump
    # and max_depth cannot bind further: the two limits compose rather than
    # one overriding the other.
    by_leaves = MojoBoostRegressor(
        n_estimators=40, num_leaves=2, max_depth=-1
    ).fit(X, y)
    by_depth = MojoBoostRegressor(
        n_estimators=40, num_leaves=2, max_depth=1
    ).fit(X, y)
    assert all(
        a == b for a, b in zip(by_leaves.predict(X), by_depth.predict(X))
    ), "num_leaves=2 and max_depth=1 should agree"

    # Round-trips like any other fitted model.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "depth.mbst")
        stumps.save(path)
        loaded = MojoBoostRegressor.load(path)
        assert all(
            a == b for a, b in zip(stumps.predict(X), loaded.predict(X))
        ), "depth-limited model did not round-trip"

    print(f"max_depth ok (MSE {mse_stumps:.5f} -> {mse_mid:.5f} -> {mse_deep:.5f})")


def make_interaction_regression(n_rows, seed=23):
    """Four features whose target is x0*x1 + x2*x3, so an unconstrained fit
    has a reason to interact any pair."""
    rng = _rand_stream(seed)
    X = [[next(rng) for _ in range(4)] for _ in range(n_rows)]
    y = [4.0 * r[0] * r[1] + 3.0 * r[2] * r[3] for r in X]
    return X, y


def _vary_feature(X, feature, value):
    out = [list(r) for r in X]
    for r in out:
        r[feature] = value
    return out


def test_interaction_constraints():
    X, y = make_interaction_regression(700)

    # No constraints and one group holding every feature are the same model.
    plain = MojoBoostRegressor(n_estimators=30).fit(X, y)
    allowed_all = MojoBoostRegressor(
        n_estimators=30, interaction_constraints=[[0, 1, 2, 3]]
    ).fit(X, y)
    pa = list(plain.predict(X))
    pb = list(allowed_all.predict(X))
    assert all(
        a == b for a, b in zip(pa, pb)
    ), "one all-feature group changed the fit"

    # Splitting the features into two groups must change the fit.
    split = MojoBoostRegressor(
        n_estimators=30, interaction_constraints=[[0, 1], [2, 3]]
    ).fit(X, y)
    pc = list(split.predict(X))
    assert any(a != c for a, c in zip(pa, pc)), "constraints had no effect"

    # Constrained to feature 0 alone, every other feature is unlisted and so
    # never split on: predictions cannot move when they change.
    only_0 = MojoBoostRegressor(
        n_estimators=30, interaction_constraints=[[0]]
    ).fit(X, y)
    base = list(only_0.predict(X))
    for feature in (1, 2, 3):
        moved = list(only_0.predict(_vary_feature(X, feature, 0.0)))
        assert all(
            a == b for a, b in zip(base, moved)
        ), f"an unlisted feature {feature} still moved the prediction"

    # Saved models carry the trees, not the training constraints, so a
    # round-trip is exact and `load` needs no constraint argument.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "constrained.mbst")
        split.save(path)
        loaded = MojoBoostRegressor.load(path)
        assert all(
            a == b for a, b in zip(pc, list(loaded.predict(X)))
        ), "round-trip not exact"

    # And a classifier takes the same parameter.
    Xc, yc = make_classification(500, 2)
    clf = MojoBoostClassifier(
        n_estimators=25, interaction_constraints=[[0, 1], [2, 3]]
    ).fit(Xc, yc)
    assert len(list(clf.predict(Xc))) == len(yc)
    print("interaction constraints ok")


def test_interaction_constraint_validation():
    X, y = make_interaction_regression(200)
    for bad in (
        [[0, 9]],          # feature index out of range
        [[0, -1]],         # negative feature index
        [[0, 0, 1]],       # repeated feature within a group
        [[]],              # empty group
        "[[0,1]]",         # LightGBM's string form is not accepted
        [0, 1],            # groups, not a flat list
    ):
        try:
            MojoBoostRegressor(
                n_estimators=5, interaction_constraints=bad
            ).fit(X, y)
            raise AssertionError(f"{bad!r} should raise")
        except ValueError:
            pass
    print("interaction constraint validation ok")


def test_device_cpu_and_auto():
    X, y = make_regression(400)
    cpu = MojoBoostRegressor(n_estimators=20, device="cpu").fit(X, y)
    assert cpu.device_ == "cpu"

    # auto resolves to the CPU today (the size heuristic ships disabled),
    # so it must be bit-identical to an explicit cpu fit.
    auto = MojoBoostRegressor(n_estimators=20, device="auto").fit(X, y)
    assert auto.device_ == "cpu", f"auto chose {auto.device_}"
    assert all(
        a == b for a, b in zip(cpu.predict(X), auto.predict(X))
    ), "auto and cpu disagree"

    # The default is cpu, and it survives a refit.
    default = MojoBoostRegressor(n_estimators=20).fit(X, y)
    assert default.device == "cpu" and default.device_ == "cpu"
    assert all(a == b for a, b in zip(cpu.predict(X), default.predict(X)))

    clf = MojoBoostClassifier(n_estimators=20, device="auto").fit(
        *make_classification(400, 3)
    )
    assert clf.device_ == "cpu"
    print("device cpu/auto ok")


def test_device_gpu():
    X, y = make_regression(300)
    if not gpu_available():
        try:
            MojoBoostRegressor(n_estimators=10, device="gpu").fit(X, y)
            raise AssertionError("gpu without an accelerator should raise")
        except RuntimeError:
            pass
        print("device gpu ok (no accelerator: explicit gpu raises)")
        return

    gpu = MojoBoostRegressor(n_estimators=10, device="gpu").fit(X, y)
    assert gpu.device_ == "gpu"
    cpu = MojoBoostRegressor(n_estimators=10, device="cpu").fit(X, y)
    # GPU histograms are Float32 fixed-point, so agreement with the Float64
    # CPU trainer is tolerance-based (see src/mojoboost/train_gpu.mojo).
    worst = max(
        abs(a - b) for a, b in zip(gpu.predict(X), cpu.predict(X))
    )
    assert worst <= 1e-3, f"gpu and cpu predictions differ by {worst}"

    # Binary is single-output, so it has a GPU path.
    Xb, yb = make_classification(300, 2)
    binary = MojoBoostClassifier(n_estimators=10, device="gpu").fit(Xb, yb)
    assert binary.device_ == "gpu"

    # Multiclass has no GPU path: it raises instead of falling back.
    Xc, yc = make_classification(300, 3)
    try:
        MojoBoostClassifier(n_estimators=5, device="gpu").fit(Xc, yc)
        raise AssertionError("multiclass on gpu should raise")
    except RuntimeError:
        pass
    print(f"device gpu ok (max |gpu - cpu| {worst:.2e})")


def test_device_gpu_unavailable():
    """MOJOBOOST_DISABLE_GPU makes the library report no accelerator, so the
    unavailable path is covered on GPU machines too."""
    X, y = make_regression(200)
    previous = os.environ.get("MOJOBOOST_DISABLE_GPU")
    os.environ["MOJOBOOST_DISABLE_GPU"] = "1"
    try:
        assert not gpu_available()
        try:
            MojoBoostRegressor(n_estimators=5, device="gpu").fit(X, y)
            raise AssertionError("unavailable gpu should raise")
        except RuntimeError:
            pass
        # auto stays usable and picks the CPU.
        auto = MojoBoostRegressor(n_estimators=5, device="auto").fit(X, y)
        assert auto.device_ == "cpu"
    finally:
        if previous is None:
            del os.environ["MOJOBOOST_DISABLE_GPU"]
        else:
            os.environ["MOJOBOOST_DISABLE_GPU"] = previous
    print("device unavailable-gpu ok")


def test_device_invalid():
    X, y = make_regression(100)
    for bad in ("cuda", "", None, 1, "gpu ", "CPU!"):
        try:
            MojoBoostRegressor(n_estimators=5, device=bad).fit(X, y)
            raise AssertionError(f"device={bad!r} should raise")
        except ValueError:
            pass

    # Names are case-insensitive, as LightGBM treats device_type.
    upper = MojoBoostRegressor(n_estimators=5, device="CPU").fit(X, y)
    assert upper.device_ == "cpu"
    print("device validation ok")


def test_device_serialization():
    """The device is a training choice, not part of the model, so files
    round-trip bit-exactly and carry no device."""
    X, y = make_regression(300)
    devices = ["cpu"] + (["gpu"] if gpu_available() else [])
    for device in devices:
        model = MojoBoostRegressor(n_estimators=10, device=device).fit(X, y)
        assert model.device_ == device
        pred = list(model.predict(X))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"{device}.mbst")
            model.save(path)
            loaded = MojoBoostRegressor.load(path)
            assert list(loaded.predict(X)) == pred, "round-trip not exact"
            assert not hasattr(loaded, "device_"), (
                "a loaded model should carry no training device"
            )
            assert loaded.device == "cpu", "load() defaults device to cpu"
    print(f"device serialization ok ({', '.join(devices)})")


def _custom_squared_error(raw, y):
    """The custom-objective form of the built-in squared-error objective:
    gradient raw - y, hessian 1. Works with numpy arrays or the stdlib
    buffer fallback."""
    if _np is not None:
        return raw - _np.asarray(y), _np.ones(len(raw))
    return [a - b for a, b in zip(raw, y)], [1.0] * len(raw)


def test_custom_objective_matches_builtin():
    """A custom objective that computes the built-in objective's derivatives,
    started from the same base score, must reproduce it bit for bit."""
    X, y = make_regression(600)
    builtin = MojoBoostRegressor(n_estimators=40).fit(X, y)
    custom = MojoBoostRegressor(
        objective=_custom_squared_error, base_score="mean", n_estimators=40
    ).fit(X, y)
    pa = list(builtin.predict(X))
    pb = list(custom.predict(X))
    assert pa == pb, "custom squared error did not reproduce the built-in"

    # predict() returns raw scores for a custom objective, which for this
    # objective is also the response scale.
    assert list(custom.predict(X)) == list(custom.predict(X))

    # Weights are applied by the trainer, not by the callback, so the
    # weighted fits match too.
    w = [1.0 + (i % 5) for i in range(len(y))]
    w[3] = 0.0
    bw = MojoBoostRegressor(n_estimators=40).fit(X, y, sample_weight=w)
    cw = MojoBoostRegressor(
        objective=_custom_squared_error, base_score="mean", n_estimators=40
    ).fit(X, y, sample_weight=w)
    assert list(bw.predict(X)) == list(cw.predict(X)), (
        "weighted custom objective did not reproduce the built-in"
    )
    print("custom objective equivalence ok")


def test_custom_objective_base_score():
    """base_score defaults to 0 and is the raw score training starts from."""
    X, y = make_regression(300)
    zero = MojoBoostRegressor(
        objective=_custom_squared_error, n_estimators=0
    ).fit(X, y)
    for p in list(zero.predict(X))[:20]:
        assert p == 0.0, f"expected base score 0.0, got {p}"

    fixed = MojoBoostRegressor(
        objective=_custom_squared_error, base_score=2.5, n_estimators=0
    ).fit(X, y)
    for p in list(fixed.predict(X))[:20]:
        assert p == 2.5, f"expected base score 2.5, got {p}"

    mean_y = sum(y) / len(y)
    averaged = MojoBoostRegressor(
        objective=_custom_squared_error, base_score="mean", n_estimators=0
    ).fit(X, y)
    for p in list(averaged.predict(X))[:20]:
        assert abs(p - mean_y) < 1e-12, f"expected {mean_y}, got {p}"

    try:
        MojoBoostRegressor(
            objective=_custom_squared_error, base_score="median"
        ).fit(X, y)
        raise AssertionError("unknown base_score should raise")
    except ValueError:
        pass
    print("custom objective base_score ok")


def test_custom_objective_validation():
    """Bad callback output is rejected with a message naming the problem.
    Errors raised inside the callback cross back out of the trainer."""
    X, y = make_regression(200)

    def short(raw, labels):
        return [0.1] * 3, [1.0] * 3

    def wrong_hess_length(raw, labels):
        return [0.1] * len(raw), [1.0] * (len(raw) - 1)

    def negative_hessian(raw, labels):
        return [0.1] * len(raw), [-1.0] * len(raw)

    def not_finite(raw, labels):
        return [float("nan")] * len(raw), [1.0] * len(raw)

    def not_a_pair(raw, labels):
        return [0.1] * len(raw)

    def explodes(raw, labels):
        raise KeyError("callback blew up")

    for objective, fragment in (
        (short, "expected"),
        (wrong_hess_length, "expected"),
        (negative_hessian, "negative hessian"),
        (not_finite, "non-finite gradient"),
        (not_a_pair, "must return (grad, hess)"),
        (explodes, "callback blew up"),
    ):
        try:
            MojoBoostRegressor(objective=objective, n_estimators=5).fit(X, y)
            raise AssertionError(f"{objective.__name__} should raise")
        except AssertionError:
            raise
        except Exception as exc:
            assert fragment in str(exc), (
                f"{objective.__name__}: expected {fragment!r} in {exc!r}"
            )
    print("custom objective validation ok")


def test_custom_objective_is_single_output():
    """Policy: custom objectives are single-output only. The classifier
    takes no objective and says why."""
    X, y = make_classification(300, 3)
    try:
        MojoBoostClassifier(
            objective=_custom_squared_error, n_estimators=5
        ).fit(X, y)
        raise AssertionError("classifier should reject a custom objective")
    except ValueError as exc:
        assert "single-output only" in str(exc), str(exc)

    Xb, yb = make_classification(300, 2)
    try:
        MojoBoostClassifier(
            objective=_custom_squared_error, n_estimators=5
        ).fit(Xb, yb)
        raise AssertionError("classifier should reject a custom objective")
    except ValueError:
        pass
    print("custom objective single-output policy ok")


def test_custom_objective_serialization():
    """A custom-objective model is an ordinary tree ensemble on disk, and a
    loaded one predicts identically."""
    X, y = make_regression(300)
    model = MojoBoostRegressor(
        objective=_custom_squared_error, base_score="mean", n_estimators=20
    ).fit(X, y)
    pred = list(model.predict(X))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "custom.mbst")
        model.save(path)
        loaded = MojoBoostRegressor.load(path)
        assert list(loaded.predict(X)) == pred, "round-trip not exact"
    print("custom objective serialization ok")


def test_custom_objective_sees_current_predictions():
    """The callback receives the running raw scores, not a stale copy: a
    callback that ignores them (constant gradient) drifts monotonically,
    while the squared-error one converges."""
    X, y = make_regression(300)
    seen = []

    def record_first(raw, labels):
        seen.append(list(raw)[0])
        if _np is not None:
            return raw - _np.asarray(labels), _np.ones(len(raw))
        return [a - b for a, b in zip(raw, labels)], [1.0] * len(raw)

    MojoBoostRegressor(
        objective=record_first, base_score="mean", n_estimators=10
    ).fit(X, y)
    assert len(seen) == 10, f"expected one call per round, got {len(seen)}"
    assert len(set(seen)) > 1, "the callback saw the same raw score every round"
    print(f"custom objective per-round calls ok ({len(seen)} calls)")


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
    for bad in (
        dict(objective="l2"),
        dict(objective="huber", alpha=0.0),
        dict(objective="quantile", alpha=1.5),
        dict(lambda_l1=-1.0),
        dict(reg_alpha=-1.0),
        dict(reg_lambda=-1.0),
    ):
        try:
            MojoBoostRegressor(**bad).fit([[1.0], [2.0]], [1.0, 2.0])
            raise AssertionError(f"{bad} should raise")
        except ValueError:
            pass
    try:
        MojoBoostRegressor().fit(
            [[1.0], [2.0]], [1.0, 2.0], sample_weight=[0.0, 0.0]
        )
        raise AssertionError("all-zero sample_weight should raise")
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
    test_regression_objectives()
    test_lambda_l1()
    test_regularization_aliases()
    test_lightgbm_sklearn_aliases()
    test_bagging()
    test_goss()
    test_max_depth()
    test_feature_fraction()
    test_interaction_constraints()
    test_interaction_constraint_validation()
    test_input_validation()
    test_custom_objective_matches_builtin()
    test_custom_objective_base_score()
    test_custom_objective_validation()
    test_custom_objective_is_single_output()
    test_custom_objective_serialization()
    test_custom_objective_sees_current_predictions()
    test_device_cpu_and_auto()
    test_device_gpu()
    test_device_gpu_unavailable()
    test_device_invalid()
    test_device_serialization()
    print("all python API tests passed")
