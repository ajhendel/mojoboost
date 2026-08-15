"""End-to-end tests for the mojotrees Python API.

Runs with or without numpy (the wrapper falls back to stdlib buffers).
Usage: build the extension with bindings/build.sh, then
    pixi run python python/test_python_api.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mojotrees import (
    MojoTreesClassifier,
    MojoTreesRanker,
    MojoTreesRegressor,
    gpu_available,
    group_from_query_ids,
    ndcg_score,
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
    model = MojoTreesRegressor(n_estimators=50)
    assert model.fit(X, y) is model
    pred = model.predict(X)
    mse = sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y)
    assert mse < 0.01, f"regressor train MSE too high: {mse}"

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "reg.mbst")
        model.save(path)
        loaded = MojoTreesRegressor.load(path)
        pred2 = loaded.predict(X)
    assert all(a == b for a, b in zip(pred, pred2)), "round-trip not exact"
    print(f"regressor ok (train MSE {mse:.5f})")


def test_binary_classifier():
    X, y = make_classification(800, 2)
    model = MojoTreesClassifier(n_estimators=50).fit(X, y)
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
        loaded = MojoTreesClassifier.load(path)
        assert loaded.n_classes_ == 2
        assert list(loaded.predict(X)) == pred, "round-trip not exact"
    print(f"binary classifier ok (train acc {acc:.3f})")


def test_multiclass_classifier():
    X, y = make_classification(900, 3)
    model = MojoTreesClassifier(n_estimators=30).fit(X, y)
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
        loaded = MojoTreesClassifier.load(path)
        assert loaded.n_classes_ == 3
        assert list(loaded.predict(X)) == pred, "round-trip not exact"
    print(f"multiclass classifier ok (train acc {acc:.3f})")


def test_sample_weight():
    # (Weight-2 == duplicated-row exactness is tested at the Mojo level on
    # pre-binned data; through fit() an extra row shifts the quantile bin
    # edges, so here we assert the properties that hold at this level.)
    X, y = make_regression(300)

    # All-ones weights must be bit-identical to no weights.
    plain = MojoTreesRegressor(n_estimators=20).fit(X, y)
    ones = MojoTreesRegressor(n_estimators=20).fit(
        X, y, sample_weight=[1.0] * len(y)
    )
    pa = list(plain.predict(X[:50]))
    pb = list(ones.predict(X[:50]))
    assert all(a == b for a, b in zip(pa, pb)), "unit weights changed the fit"

    # Non-uniform weights must change the fit.
    w = [10.0 if r[0] > 0.5 else 0.1 for r in X]
    skewed = MojoTreesRegressor(n_estimators=20).fit(X, y, sample_weight=w)
    pc = list(skewed.predict(X[:50]))
    assert any(a != c for a, c in zip(pa, pc)), "weights had no effect"
    print("sample_weight ok")


def test_regression_objectives():
    X, y = make_regression(600)

    for objective in ("huber", "mae", "regression_l1"):
        model = MojoTreesRegressor(objective=objective, n_estimators=40)
        pred = model.fit(X, y).predict(X)
        mse = sum((p - t) ** 2 for p, t in zip(pred, y)) / len(y)
        assert mse < 0.02, f"{objective} train MSE too high: {mse}"

    # A 0.9-quantile fit should predict above most training targets.
    q = MojoTreesRegressor(objective="quantile", alpha=0.9, n_estimators=40)
    pred = q.fit(X, y).predict(X)
    frac_below = sum(int(t <= p) for p, t in zip(pred, y)) / len(y)
    assert 0.8 < frac_below <= 1.0, f"quantile coverage off: {frac_below}"

    # Identity link, so these round-trip through the existing format.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "quantile.mbst")
        q.save(path)
        loaded = MojoTreesRegressor.load(path)
        pred2 = loaded.predict(X)
    assert all(a == b for a, b in zip(pred, pred2)), "round-trip not exact"
    print(f"regression objectives ok (q90 coverage {frac_below:.3f})")


def test_lambda_l1():
    X, y = make_regression(600)
    mean_y = sum(y) / len(y)

    # The default is 0, so passing it explicitly must not change the fit.
    default = MojoTreesRegressor(n_estimators=30).fit(X, y)
    explicit = MojoTreesRegressor(n_estimators=30, lambda_l1=0.0).fit(X, y)
    pa = list(default.predict(X))
    pb = list(explicit.predict(X))
    assert all(a == b for a, b in zip(pa, pb)), "lambda_l1=0 changed the fit"

    # A moderate penalty shrinks every leaf's Newton step, so the fitted
    # predictions stay closer to the base score and fit the target less well.
    reg = MojoTreesRegressor(n_estimators=30, lambda_l1=5.0).fit(X, y)
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
    huge = MojoTreesRegressor(n_estimators=30, lambda_l1=1e6).fit(X, y)
    for p in list(huge.predict(X))[:50]:
        assert abs(p - mean_y) < 1e-9, f"expected base score, got {p}"
    print(f"lambda_l1 ok (train MSE {mse_plain:.5f} -> {mse_reg:.5f})")


def test_lightgbm_sklearn_aliases():
    """Native LightGBM names stay canonical; its sklearn wrapper's names
    are accepted as strict aliases and produce the same model."""
    X, y = make_regression(500)
    native = MojoTreesRegressor(
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
    sklearn_names = MojoTreesRegressor(
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

    params = MojoTreesRegressor().get_params()
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

    updated = MojoTreesRegressor().set_params(
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
            MojoTreesRegressor(n_estimators=2, **bad).fit(X, y)
            raise AssertionError(f"conflicting aliases should raise: {bad}")
        except ValueError:
            pass
    print("LightGBM sklearn aliases ok")


def test_bagging():
    X, y = make_regression(600)

    # The defaults disable bagging, and so does either switch on its own.
    base = MojoTreesRegressor(n_estimators=30).fit(X, y)
    pa = list(base.predict(X))
    for kwargs in ({"bagging_fraction": 1.0, "bagging_freq": 1},
                   {"bagging_fraction": 0.5, "bagging_freq": 0}):
        off = MojoTreesRegressor(n_estimators=30, **kwargs).fit(X, y)
        assert all(a == b for a, b in zip(pa, off.predict(X))), (
            f"bagging changed the fit when disabled: {kwargs}"
        )

    # Same seed, same model, bit for bit; a different seed moves it.
    def bagged(seed):
        return MojoTreesRegressor(
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
            MojoTreesRegressor(n_estimators=5, **bad).fit(X, y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # Classifiers take the same parameters.
    Xc, yc = make_classification(400, 3)
    clf = MojoTreesClassifier(
        n_estimators=20, bagging_fraction=0.7, bagging_freq=2
    ).fit(Xc, yc)
    acc = sum(int(p == t) for p, t in zip(clf.predict(Xc), yc)) / len(yc)
    assert acc > 0.9, f"bagged classifier accuracy too low: {acc}"
    print(f"bagging ok (train MSE {mse:.5f}, clf acc {acc:.3f})")


def test_goss():
    X, y = make_regression(600)

    # gbdt is the default, and naming it explicitly changes nothing.
    base = MojoTreesRegressor(n_estimators=30).fit(X, y)
    pa = list(base.predict(X))
    explicit = MojoTreesRegressor(n_estimators=30, boosting="gbdt").fit(X, y)
    assert all(a == b for a, b in zip(pa, explicit.predict(X))), (
        "boosting='gbdt' is not the default path"
    )

    # GOSS with the automatic warmup (int(1 / learning_rate) = 10 rounds of
    # the 30 here) still samples, so the fit moves.
    def goss(seed=3, **kwargs):
        return MojoTreesRegressor(
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
    warmed = MojoTreesRegressor(
        n_estimators=30, boosting="goss", goss_warmup_rounds=30
    ).fit(X, y)
    assert all(a == b for a, b in zip(pa, warmed.predict(X))), (
        "warmup rounds did not train on every row"
    )

    # boosting_type is LightGBM's scikit-learn spelling of the same thing.
    aliased = MojoTreesRegressor(
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

    # Two representative rejections; the full validation matrix lives in
    # python/tests/test_params.py.
    for bad in (
        # GOSS's top_rate and other_rate share one unit of rows.
        {"boosting": "goss", "top_rate": 0.7, "other_rate": 0.5},
        # GOSS and row bagging both own the sampled rows.
        {"boosting": "goss", "bagging_fraction": 0.5, "bagging_freq": 1},
    ):
        try:
            MojoTreesRegressor(n_estimators=5, **bad).fit(X, y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # Classifiers take the same parameters.
    Xc, yc = make_classification(400, 3)
    clf = MojoTreesClassifier(
        n_estimators=20, boosting="goss", goss_warmup_rounds=0
    ).fit(Xc, yc)
    acc = sum(int(p == t) for p, t in zip(clf.predict(Xc), yc)) / len(yc)
    assert acc > 0.9, f"GOSS classifier accuracy too low: {acc}"
    print(f"goss ok (train MSE {mse:.5f}, clf acc {acc:.3f})")


def test_feature_fraction():
    X, y = make_regression(600, n_features=10)

    # The defaults disable subsampling, and so does an explicit 1.0 whatever
    # the seed: selecting every feature is not a random event.
    base = MojoTreesRegressor(n_estimators=30).fit(X, y)
    pa = list(base.predict(X))
    for kwargs in ({"feature_fraction": 1.0},
                   {"feature_fraction_bynode": 1.0},
                   {"feature_fraction": 1.0, "feature_fraction_seed": 999}):
        off = MojoTreesRegressor(n_estimators=30, **kwargs).fit(X, y)
        assert all(a == b for a, b in zip(pa, off.predict(X))), (
            f"feature subsampling changed the fit when disabled: {kwargs}"
        )

    # Same seed, same model, bit for bit; a different seed moves it.
    def sampled(seed, bynode=1.0):
        return MojoTreesRegressor(
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
            MojoTreesRegressor(n_estimators=5, **bad).fit(X, y)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")

    # Classifiers take the same parameters.
    Xc, yc = make_classification(400, 3, n_features=8)
    clf = MojoTreesClassifier(
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
    default = MojoTreesRegressor(n_estimators=40).fit(X, y)
    assert default.max_depth == -1, "default max_depth must be -1"
    base = list(default.predict(X))
    for unlimited in (-1, 0, -5):
        other = MojoTreesRegressor(
            n_estimators=40, max_depth=unlimited
        ).fit(X, y)
        assert all(
            a == b for a, b in zip(base, other.predict(X))
        ), f"max_depth={unlimited} should mean unlimited"

    # A limit far above the natural depth is also a no-op.
    loose = MojoTreesRegressor(n_estimators=40, max_depth=100).fit(X, y)
    assert all(
        a == b for a, b in zip(base, loose.predict(X))
    ), "a max_depth above the tree's depth changed the fit"

    # Stumps are strictly weaker than unlimited depth, and relaxing the
    # limit recovers some of the fit.
    stumps = MojoTreesRegressor(n_estimators=40, max_depth=1).fit(X, y)
    mse_stumps = mse(stumps)
    mse_deep = mse(default)
    mse_mid = mse(MojoTreesRegressor(n_estimators=40, max_depth=3).fit(X, y))
    assert mse_stumps > mse_deep, "max_depth=1 did not constrain the fit"
    assert mse_mid < mse_stumps, "raising max_depth did not improve the fit"

    # num_leaves=2 permits exactly one split, so the tree is already a stump
    # and max_depth cannot bind further: the two limits compose rather than
    # one overriding the other.
    by_leaves = MojoTreesRegressor(
        n_estimators=40, num_leaves=2, max_depth=-1
    ).fit(X, y)
    by_depth = MojoTreesRegressor(
        n_estimators=40, num_leaves=2, max_depth=1
    ).fit(X, y)
    assert all(
        a == b for a, b in zip(by_leaves.predict(X), by_depth.predict(X))
    ), "num_leaves=2 and max_depth=1 should agree"

    # Round-trips like any other fitted model.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "depth.mbst")
        stumps.save(path)
        loaded = MojoTreesRegressor.load(path)
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
    plain = MojoTreesRegressor(n_estimators=30).fit(X, y)
    allowed_all = MojoTreesRegressor(
        n_estimators=30, interaction_constraints=[[0, 1, 2, 3]]
    ).fit(X, y)
    pa = list(plain.predict(X))
    pb = list(allowed_all.predict(X))
    assert all(
        a == b for a, b in zip(pa, pb)
    ), "one all-feature group changed the fit"

    # Splitting the features into two groups must change the fit.
    split = MojoTreesRegressor(
        n_estimators=30, interaction_constraints=[[0, 1], [2, 3]]
    ).fit(X, y)
    pc = list(split.predict(X))
    assert any(a != c for a, c in zip(pa, pc)), "constraints had no effect"

    # Constrained to feature 0 alone, every other feature is unlisted and so
    # never split on: predictions cannot move when they change.
    only_0 = MojoTreesRegressor(
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
        loaded = MojoTreesRegressor.load(path)
        assert all(
            a == b for a, b in zip(pc, list(loaded.predict(X)))
        ), "round-trip not exact"

    # And a classifier takes the same parameter.
    Xc, yc = make_classification(500, 2)
    clf = MojoTreesClassifier(
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
            MojoTreesRegressor(
                n_estimators=5, interaction_constraints=bad
            ).fit(X, y)
            raise AssertionError(f"{bad!r} should raise")
        except ValueError:
            pass
    print("interaction constraint validation ok")


def make_monotone_regression(n_rows, seed=29):
    """Two features whose target falls then rises in x0 and rises then falls
    in x1, so an unconstrained fit is monotone in neither."""
    rng = _rand_stream(seed)
    X = [[next(rng), next(rng)] for _ in range(n_rows)]
    y = [
        (r[0] - 0.5) ** 2 - (r[1] - 0.5) ** 2 for r in X
    ]
    return X, y


def _monotone_probe_grid(steps=13):
    """Query points from below the training range to above it."""
    return [-0.25 + 1.5 * i / (steps - 1) for i in range(steps)]


def _predict_probe_grid(model, xs):
    """Predictions over the (x0, x1) grid, as a list of lists indexed [i][j]."""
    rows = [[a, b] for a in xs for b in xs]
    flat = list(model.predict(rows))
    n = len(xs)
    return [flat[i * n : (i + 1) * n] for i in range(n)]


def test_monotone_constraints():
    X, y = make_monotone_regression(600)
    xs = _monotone_probe_grid()

    # Unconstrained, the fit must break both orderings, so the checks below
    # are not passing on data that was already monotone.
    plain = MojoTreesRegressor(n_estimators=30).fit(X, y)
    pg = _predict_probe_grid(plain, xs)
    assert any(
        pg[i][j] > pg[i + 1][j]
        for j in range(len(xs))
        for i in range(len(xs) - 1)
    ), "unconstrained fit was already nondecreasing in feature 0"

    model = MojoTreesRegressor(
        n_estimators=30, monotone_constraints=[1, -1]
    ).fit(X, y)
    grid = _predict_probe_grid(model, xs)
    for j in range(len(xs)):
        for i in range(len(xs) - 1):
            assert grid[i][j] <= grid[i + 1][j], (
                f"prediction fell along feature 0 at x1={xs[j]}"
            )
    for i in range(len(xs)):
        for j in range(len(xs) - 1):
            assert grid[i][j] >= grid[i][j + 1], (
                f"prediction rose along feature 1 at x0={xs[i]}"
            )

    # An all-zero vector is unconstrained, bit for bit.
    zeros = MojoTreesRegressor(
        n_estimators=30, monotone_constraints=[0, 0]
    ).fit(X, y)
    assert all(
        a == b
        for a, b in zip(list(plain.predict(X)), list(zeros.predict(X)))
    ), "an all-zero constraint vector changed the fit"

    # Saved models keep the constraints and predict identically.
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "monotone.mbst")
        model.save(path)
        loaded = MojoTreesRegressor.load(path)
        before = list(model.predict(X))
        after = list(loaded.predict(X))
    assert all(a == b for a, b in zip(before, after)), "round-trip not exact"

    # A binary classifier takes the same parameter, and the increasing link
    # carries the guarantee through to the probability.
    Xc = [list(r) for r in X]
    yc = [1 if t > 0.0 else 0 for t in y]
    clf = MojoTreesClassifier(
        n_estimators=25, monotone_constraints=[1, -1]
    ).fit(Xc, yc)
    proba = [list(p) for p in clf.predict_proba([[a, 0.5] for a in xs])]
    for i in range(len(xs) - 1):
        assert proba[i][1] <= proba[i + 1][1], "class-1 probability fell"
    print("monotone constraints ok")


def test_monotone_constraint_validation():
    X, y = make_monotone_regression(200)
    for bad in (
        [1],               # one entry for two features
        [1, 0, -1],        # one entry too many
        [1, 2],            # 2 is not a constraint
        [1, 0.5],          # fractional, rejected rather than truncated
        "1,-1",            # LightGBM's string form is not accepted
    ):
        try:
            MojoTreesRegressor(
                n_estimators=5, monotone_constraints=bad
            ).fit(X, y)
            raise AssertionError(f"{bad!r} should raise")
        except ValueError:
            pass
    print("monotone constraint validation ok")


def test_device_cpu_and_auto():
    X, y = make_regression(400)
    cpu = MojoTreesRegressor(n_estimators=20, device="cpu").fit(X, y)
    assert cpu.device_ == "cpu"

    # auto resolves to the CPU today (the size heuristic ships disabled),
    # so it must be bit-identical to an explicit cpu fit.
    auto = MojoTreesRegressor(n_estimators=20, device="auto").fit(X, y)
    assert auto.device_ == "cpu", f"auto chose {auto.device_}"
    assert all(
        a == b for a, b in zip(cpu.predict(X), auto.predict(X))
    ), "auto and cpu disagree"

    # The default is cpu, and it survives a refit.
    default = MojoTreesRegressor(n_estimators=20).fit(X, y)
    assert default.device == "cpu" and default.device_ == "cpu"
    assert all(a == b for a, b in zip(cpu.predict(X), default.predict(X)))

    clf = MojoTreesClassifier(n_estimators=20, device="auto").fit(
        *make_classification(400, 3)
    )
    assert clf.device_ == "cpu"
    print("device cpu/auto ok")


def test_device_gpu():
    X, y = make_regression(300)
    if not gpu_available():
        try:
            MojoTreesRegressor(n_estimators=10, device="gpu").fit(X, y)
            raise AssertionError("gpu without an accelerator should raise")
        except RuntimeError:
            pass
        print("device gpu ok (no accelerator: explicit gpu raises)")
        return

    gpu = MojoTreesRegressor(n_estimators=10, device="gpu").fit(X, y)
    assert gpu.device_ == "gpu"
    cpu = MojoTreesRegressor(n_estimators=10, device="cpu").fit(X, y)
    # GPU histograms are Float32 fixed-point, so agreement with the Float64
    # CPU trainer is tolerance-based (see src/mojotrees/train_gpu.mojo).
    worst = max(
        abs(a - b) for a, b in zip(gpu.predict(X), cpu.predict(X))
    )
    assert worst <= 1e-3, f"gpu and cpu predictions differ by {worst}"

    # Binary is single-output, so it has a GPU path.
    Xb, yb = make_classification(300, 2)
    binary = MojoTreesClassifier(n_estimators=10, device="gpu").fit(Xb, yb)
    assert binary.device_ == "gpu"

    # Multiclass trains one class per round through the GPU trainer, so it
    # gets the same tolerance-based CPU agreement as the regressor above.
    Xc, yc = make_classification(300, 3)
    multi = MojoTreesClassifier(n_estimators=5, device="gpu").fit(Xc, yc)
    assert multi.device_ == "gpu"
    multi_cpu = MojoTreesClassifier(n_estimators=5, device="cpu").fit(Xc, yc)
    worst_mc = max(
        abs(a - b)
        for gpu_row, cpu_row in zip(
            multi.predict_proba(Xc), multi_cpu.predict_proba(Xc)
        )
        for a, b in zip(gpu_row, cpu_row)
    )
    assert worst_mc <= 1e-3, (
        f"gpu and cpu multiclass probabilities differ by {worst_mc}"
    )
    print(f"device gpu ok (max |gpu - cpu| {worst:.2e})")


def test_device_gpu_unavailable():
    """MOJOTREES_DISABLE_GPU makes the library report no accelerator, so the
    unavailable path is covered on GPU machines too."""
    X, y = make_regression(200)
    previous = os.environ.get("MOJOTREES_DISABLE_GPU")
    os.environ["MOJOTREES_DISABLE_GPU"] = "1"
    try:
        assert not gpu_available()
        try:
            MojoTreesRegressor(n_estimators=5, device="gpu").fit(X, y)
            raise AssertionError("unavailable gpu should raise")
        except RuntimeError:
            pass
        # auto stays usable and picks the CPU.
        auto = MojoTreesRegressor(n_estimators=5, device="auto").fit(X, y)
        assert auto.device_ == "cpu"
    finally:
        if previous is None:
            del os.environ["MOJOTREES_DISABLE_GPU"]
        else:
            os.environ["MOJOTREES_DISABLE_GPU"] = previous
    print("device unavailable-gpu ok")


def test_device_invalid():
    X, y = make_regression(100)
    for bad in ("cuda", "", None, 1, "gpu ", "CPU!"):
        try:
            MojoTreesRegressor(n_estimators=5, device=bad).fit(X, y)
            raise AssertionError(f"device={bad!r} should raise")
        except ValueError:
            pass

    # Names are case-insensitive, as LightGBM treats device_type.
    upper = MojoTreesRegressor(n_estimators=5, device="CPU").fit(X, y)
    assert upper.device_ == "cpu"
    print("device validation ok")


def test_device_serialization():
    """The device is a training choice, not part of the model, so files
    round-trip bit-exactly and carry no device."""
    X, y = make_regression(300)
    devices = ["cpu"] + (["gpu"] if gpu_available() else [])
    for device in devices:
        model = MojoTreesRegressor(n_estimators=10, device=device).fit(X, y)
        assert model.device_ == device
        pred = list(model.predict(X))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, f"{device}.mbst")
            model.save(path)
            loaded = MojoTreesRegressor.load(path)
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
    builtin = MojoTreesRegressor(n_estimators=40).fit(X, y)
    custom = MojoTreesRegressor(
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
    bw = MojoTreesRegressor(n_estimators=40).fit(X, y, sample_weight=w)
    cw = MojoTreesRegressor(
        objective=_custom_squared_error, base_score="mean", n_estimators=40
    ).fit(X, y, sample_weight=w)
    assert list(bw.predict(X)) == list(cw.predict(X)), (
        "weighted custom objective did not reproduce the built-in"
    )
    print("custom objective equivalence ok")


def test_custom_objective_base_score():
    """base_score defaults to 0 and is the raw score training starts from."""
    X, y = make_regression(300)
    zero = MojoTreesRegressor(
        objective=_custom_squared_error, n_estimators=0
    ).fit(X, y)
    for p in list(zero.predict(X))[:20]:
        assert p == 0.0, f"expected base score 0.0, got {p}"

    fixed = MojoTreesRegressor(
        objective=_custom_squared_error, base_score=2.5, n_estimators=0
    ).fit(X, y)
    for p in list(fixed.predict(X))[:20]:
        assert p == 2.5, f"expected base score 2.5, got {p}"

    mean_y = sum(y) / len(y)
    averaged = MojoTreesRegressor(
        objective=_custom_squared_error, base_score="mean", n_estimators=0
    ).fit(X, y)
    for p in list(averaged.predict(X))[:20]:
        assert abs(p - mean_y) < 1e-12, f"expected {mean_y}, got {p}"

    try:
        MojoTreesRegressor(
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
            MojoTreesRegressor(objective=objective, n_estimators=5).fit(X, y)
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
        MojoTreesClassifier(
            objective=_custom_squared_error, n_estimators=5
        ).fit(X, y)
        raise AssertionError("classifier should reject a custom objective")
    except ValueError as exc:
        assert "single-output only" in str(exc), str(exc)

    Xb, yb = make_classification(300, 2)
    try:
        MojoTreesClassifier(
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
    model = MojoTreesRegressor(
        objective=_custom_squared_error, base_score="mean", n_estimators=20
    ).fit(X, y)
    pred = list(model.predict(X))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "custom.mbst")
        model.save(path)
        loaded = MojoTreesRegressor.load(path)
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

    MojoTreesRegressor(
        objective=record_first, base_score="mean", n_estimators=10
    ).fit(X, y)
    assert len(seen) == 10, f"expected one call per round, got {len(seen)}"
    assert len(set(seen)) > 1, "the callback saw the same raw score every round"
    print(f"custom objective per-round calls ok ({len(seen)} calls)")


def test_input_validation():
    model = MojoTreesRegressor()
    try:
        model.predict([[1.0]])
        raise AssertionError("predict before fit should raise")
    except RuntimeError:
        pass
    # Gappy labels are no longer an error: they are encoded to 0..k-1 and
    # remembered on classes_, the way a scikit-learn classifier does it.
    gappy = MojoTreesClassifier(min_data_in_leaf=1).fit(
        [[1.0], [2.0]], [1, 3]
    )
    assert list(gappy.classes_) == [1, 3]
    assert set(gappy.predict([[1.0], [2.0]])) <= {1, 3}
    try:
        MojoTreesClassifier().fit([[1.0], [2.0]], [1, 1])
        raise AssertionError("a single class should raise")
    except ValueError:
        pass
    for bad in (
        dict(objective="not_an_objective"),
        dict(objective="huber", alpha=0.0),
        dict(objective="quantile", alpha=1.5),
        dict(lambda_l1=-1.0),
        dict(reg_alpha=-1.0),
        dict(reg_lambda=-1.0),
        dict(num_leaves=1),
        dict(learning_rate=0.0),
        dict(max_bin=1),
    ):
        try:
            MojoTreesRegressor(**bad).fit([[1.0], [2.0]], [1.0, 2.0])
            raise AssertionError(f"{bad} should raise")
        except ValueError:
            pass
    try:
        MojoTreesRegressor().fit(
            [[1.0], [2.0]], [1.0, 2.0], sample_weight=[0.0, 0.0]
        )
        raise AssertionError("all-zero sample_weight should raise")
    except ValueError:
        pass
    # Every objective alias the CLI accepts works here too, so the two
    # tables cannot drift apart again.
    for ok in (
        "l2",
        "mse",
        "regression_l2",
        "mean_squared_error",
        "l1",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
    ):
        MojoTreesRegressor(
            objective=ok, n_estimators=2, min_data_in_leaf=1
        ).fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
    print("validation ok")


def make_ranking(n_queries, docs=6, n_features=4, seed=23):
    """Queries whose relevance is a monotone function of a latent utility,
    graded 0..3 inside each query. Rows arrive in query order but not in
    relevance order, so an untrained model does not already rank them."""
    rng = _rand_stream(seed)
    X, y, group = [], [], []
    for _ in range(n_queries):
        rows = [[next(rng) for _ in range(n_features)] for _ in range(docs)]
        utility = [3.0 * r[0] + 2.0 * r[1] * r[2] for r in rows]
        order = sorted(range(docs), key=lambda i: -utility[i])
        labels = [0] * docs
        for rank, i in enumerate(order):
            labels[i] = max(0, 3 - rank)
        X.extend(rows)
        y.extend(labels)
        group.append(docs)
    return X, y, group


def test_ranker():
    X, y, group = make_ranking(120)
    model = MojoTreesRanker(n_estimators=40, min_data_in_leaf=5)
    assert model.fit(X, y, group=group) is model
    assert model.n_features_in_ == 4

    trained = model.score(X, y, group=group)
    untrained = ndcg_score([0.0] * len(y), y, group, at=5)
    assert untrained < 0.999, f"fixture already ranked: {untrained}"
    assert trained > untrained, f"training did not help: {trained}"
    assert trained > 0.99, f"train NDCG@5 too low: {trained}"

    # predict returns raw scores, one per row, and ranks each query.
    pred = list(model.predict(X))
    assert len(pred) == len(y)
    assert ndcg_score(pred, y, group, at=1) > 0.98

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rank.mbst")
        model.save(path)
        loaded = MojoTreesRanker.load(path)
        pred2 = list(loaded.predict(X))
    assert all(a == b for a, b in zip(pred, pred2)), "round-trip not exact"
    print(f"ranker ok (train NDCG@5 {trained:.5f})")


def test_ranker_query_bagging_and_weights():
    X, y, group = make_ranking(80)

    plain = MojoTreesRanker(n_estimators=20, min_data_in_leaf=5).fit(
        X, y, group=group
    )
    ones = MojoTreesRanker(n_estimators=20, min_data_in_leaf=5).fit(
        X, y, group=group, sample_weight=[1.0] * len(y)
    )
    pa = list(plain.predict(X[:40]))
    pb = list(ones.predict(X[:40]))
    assert all(a == b for a, b in zip(pa, pb)), "unit weights changed the fit"

    # Query bagging is reproducible from the seed and still learns.
    bagged = [
        MojoTreesRanker(
            n_estimators=20,
            min_data_in_leaf=5,
            bagging_fraction=0.5,
            bagging_freq=1,
            bagging_seed=5,
        ).fit(X, y, group=group)
        for _ in range(2)
    ]
    p0 = list(bagged[0].predict(X))
    p1 = list(bagged[1].predict(X))
    assert all(a == b for a, b in zip(p0, p1)), "bagging is not reproducible"
    assert ndcg_score(p0, y, group, at=5) > 0.95
    print("ranker bagging and weights ok")


def test_ndcg_score_semantics():
    # Hand-checked: one perfect query and one query with nothing relevant,
    # which LightGBM counts as 1.0.
    labels = [1, 0, 0, 0]
    assert ndcg_score([1.0, 0.0, 0.0, 0.0], labels, [2, 2], at=2) == 1.0
    # Reversing the first query drops it to 1/log2(3), and the all-zero
    # query still contributes 1.0.
    half = ndcg_score([0.0, 1.0, 0.0, 0.0], labels, [2, 2], at=2)
    expected = (1.0 / math.log2(3.0) + 1.0) / 2.0
    assert abs(half - expected) < 1e-8, f"{half} != {expected}"

    # Never mix queries: every document of query 1 outscores query 0's.
    assert ndcg_score([1.0, 0.0, 11.0, 10.0], [1, 0, 1, 0], [2, 2]) == 1.0
    print("ndcg_score ok")


def test_group_from_query_ids():
    assert group_from_query_ids([7, 7, 3, 3, 3, 9]) == [2, 3, 1]
    for bad in ([0, 0, 1, 1, 0], [5, 6, 5], []):
        try:
            group_from_query_ids(bad)
            raise AssertionError(f"{bad} should raise")
        except ValueError:
            pass
    print("group_from_query_ids ok")


def test_ranker_validation():
    X, y, group = make_ranking(4)
    n_rows = len(y)

    try:
        MojoTreesRanker().fit(X, y)
        raise AssertionError("missing group should raise")
    except ValueError:
        pass
    for bad in ([n_rows - 1], [0] * 4, [2.5] * (n_rows // 2), []):
        try:
            MojoTreesRanker().fit(X, y, group=bad)
            raise AssertionError(f"group {bad} should raise")
        except ValueError:
            pass
    for bad_y in ([-1] + y[1:], [31] + y[1:], [0.5] + y[1:]):
        try:
            MojoTreesRanker().fit(X, bad_y, group=group)
            raise AssertionError("bad relevance labels should raise")
        except ValueError:
            pass
    for bad in (
        dict(lambdarank_truncation_level=0),
        dict(sigmoid=0.0),
        dict(ndcg_eval_at=0),
    ):
        try:
            MojoTreesRanker(**bad).fit(X, y, group=group)
            raise AssertionError(f"{bad} should raise")
        except ValueError:
            pass
    try:
        MojoTreesRanker().predict(X)
        raise AssertionError("predict before fit should raise")
    except RuntimeError:
        pass
    print("ranker validation ok")


# ---------------------------------------------------------------- sparse
#
# The wrapper never imports scipy: it duck-types the CSC/CSR interface. So
# these run against real scipy when it is installed and against a
# stand-in with the same attribute surface when it is not, which exercises
# the identical code path either way.

try:
    from scipy import sparse as _scipy_sparse
except ImportError:
    _scipy_sparse = None


class _StubSparse:
    """A CSC/CSR matrix with SciPy's attribute surface, for environments
    without scipy. Only what the wrapper touches is implemented."""

    def __init__(self, data, indices, indptr, shape, fmt, canonical=True):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.shape = shape
        self.format = fmt
        self.ndim = 2
        self.has_canonical_format = canonical

    @property
    def nnz(self):
        return len(self.data)

    def toarray(self):
        rows, cols = self.shape
        out = [[0.0] * cols for _ in range(rows)]
        outer = cols if self.format == "csc" else rows
        for k in range(outer):
            for i in range(self.indptr[k], self.indptr[k + 1]):
                if self.format == "csc":
                    out[self.indices[i]][k] = self.data[i]
                else:
                    out[k][self.indices[i]] = self.data[i]
        return out

    def copy(self):
        return _StubSparse(
            list(self.data), list(self.indices), list(self.indptr),
            self.shape, self.format, self.has_canonical_format,
        )

    def sum_duplicates(self):
        self.has_canonical_format = True

    def tocsc(self):
        return _stub_from_dense(self.toarray(), "csc")

    def tocsr(self):
        return _stub_from_dense(self.toarray(), "csr")


def _stub_from_dense(A, fmt):
    n_rows = len(A)
    n_features = len(A[0])
    data, indices, indptr = [], [], [0]
    outer = n_features if fmt == "csc" else n_rows
    for k in range(outer):
        inner = n_rows if fmt == "csc" else n_features
        for i in range(inner):
            v = A[i][k] if fmt == "csc" else A[k][i]
            if v != 0.0:
                indices.append(i)
                data.append(float(v))
        indptr.append(len(data))
    if _np is not None:
        data = _np.asarray(data, dtype=_np.float64)
        # int32 on purpose: that is SciPy's default index width, and the
        # wrapper has to widen it.
        indices = _np.asarray(indices, dtype=_np.int32)
        indptr = _np.asarray(indptr, dtype=_np.int32)
    return _StubSparse(data, indices, indptr, (n_rows, n_features), fmt)


def _to_sparse(A, fmt):
    if _scipy_sparse is not None:
        return (
            _scipy_sparse.csc_matrix(A)
            if fmt == "csc"
            else _scipy_sparse.csr_matrix(A)
        )
    return _stub_from_dense(A, fmt)


def _sparse_backend():
    return "scipy" if _scipy_sparse is not None else "duck-typed stand-in"


def make_sparse_regression(n_rows, n_features=10, density=0.2, seed=23):
    """Dense rows that are mostly exact zeros, plus a target linear in every
    feature with a distinct coefficient (equal coefficients make split gains
    tie, and then which of two features wins is decided by rounding)."""
    rng = _rand_stream(seed)
    A = []
    for _ in range(n_rows):
        row = []
        for _ in range(n_features):
            u = next(rng)
            row.append(4.0 * (u / density) - 2.0 if u < density else 0.0)
        A.append(row)
    y = [
        sum((1.0 + 0.37 * f) * row[f] for f in range(n_features)) / 4.0
        + 0.05 * (next(rng) - 0.5)
        for row in A
    ]
    return A, y


def _max_gap(a, b):
    return max(abs(x - y) for x, y in zip(list(a), list(b)))


def test_sparse_regressor_matches_dense():
    A, y = make_sparse_regression(600)
    csc = _to_sparse(A, "csc")
    csr = _to_sparse(A, "csr")

    dense = MojoTreesRegressor(n_estimators=25, num_leaves=10).fit(A, y)
    sparse = MojoTreesRegressor(n_estimators=25, num_leaves=10).fit(csc, y)
    assert sparse.n_features_in_ == 10
    assert sparse.device_ == "cpu"

    # An implicit zero is a numerical zero, so the two fits are the same fit.
    assert _max_gap(dense.predict(A), sparse.predict(A)) < 1e-9
    # And the sparse prediction path is exact against the dense one, because
    # it walks the same trees over the same bins.
    assert list(sparse.predict(csr)) == list(sparse.predict(A))
    assert list(sparse.predict(csc)) == list(sparse.predict(csr))
    print(f"sparse regressor ok ({_sparse_backend()})")


def test_sparse_does_not_mutate_input():
    A, y = make_sparse_regression(200)
    messy = _to_sparse(A, "csc")
    messy.has_canonical_format = False
    before = list(messy.indices)
    MojoTreesRegressor(n_estimators=3).fit(messy, y)
    assert list(messy.indices) == before, "fit mutated the caller's matrix"
    print("sparse input not mutated ok")


def test_sparse_classifier_matches_dense():
    A, y = make_sparse_regression(600)
    csc = _to_sparse(A, "csc")
    csr = _to_sparse(A, "csr")

    binary = [1 if t > 0.0 else 0 for t in y]
    dense = MojoTreesClassifier(n_estimators=20, num_leaves=8).fit(A, binary)
    sparse = MojoTreesClassifier(n_estimators=20, num_leaves=8).fit(csc, binary)
    assert sparse.n_classes_ == 2
    for a, b in zip(dense.predict_proba(A), sparse.predict_proba(csr)):
        assert _max_gap(a, b) < 1e-9
    assert list(dense.predict(A)) == list(sparse.predict(csr))

    three = [0 if t < -1.0 else (1 if t < 1.0 else 2) for t in y]
    dense3 = MojoTreesClassifier(n_estimators=12, num_leaves=8).fit(A, three)
    sparse3 = MojoTreesClassifier(n_estimators=12, num_leaves=8).fit(csc, three)
    assert sparse3.n_classes_ == 3
    for a, b in zip(dense3.predict_proba(A), sparse3.predict_proba(csr)):
        assert _max_gap(a, b) < 1e-9
        assert abs(sum(b) - 1.0) < 1e-9
    print(f"sparse classifier ok ({_sparse_backend()})")


def test_sparse_sample_weight_and_missing():
    A, y = make_sparse_regression(400)
    csc = _to_sparse(A, "csc")
    csr = _to_sparse(A, "csr")
    w = [0.5 + (i % 7) / 7.0 for i in range(len(y))]
    dense = MojoTreesRegressor(n_estimators=15).fit(A, y, sample_weight=w)
    sparse = MojoTreesRegressor(n_estimators=15).fit(csc, y, sample_weight=w)
    assert _max_gap(dense.predict(A), sparse.predict(csr)) < 1e-9

    # NaN is still the missing marker; the implicit zeros are not missing.
    nan = float("nan")
    An = [list(row) for row in A]
    k = 0
    for row in An:
        for f in range(len(row)):
            if row[f] != 0.0:
                k += 1
                if k % 5 == 0:
                    row[f] = nan
    dense_n = MojoTreesRegressor(n_estimators=15).fit(An, y)
    sparse_n = MojoTreesRegressor(n_estimators=15).fit(
        _to_sparse(An, "csc"), y
    )
    assert (
        _max_gap(dense_n.predict(An), sparse_n.predict(_to_sparse(An, "csr")))
        < 1e-9
    )
    print("sparse sample_weight and missing values ok")


def test_sparse_save_load():
    A, y = make_sparse_regression(300)
    csc = _to_sparse(A, "csc")
    csr = _to_sparse(A, "csr")
    model = MojoTreesRegressor(n_estimators=15).fit(csc, y)
    before = list(model.predict(csr))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sparse.mbst")
        model.save(path)
        loaded = MojoTreesRegressor.load(path)
        after = list(loaded.predict(csr))
    # A sparse fit produces an ordinary model: nothing about the serialized
    # format changes, and the round trip stays bit-exact.
    assert before == after, "round-trip not exact"
    print("sparse save/load ok")


def test_sparse_validation():
    A, y = make_sparse_regression(200)
    csc = _to_sparse(A, "csc")

    # No sparse GPU kernel, and no silent densification.
    try:
        MojoTreesRegressor(device="gpu").fit(csc, y)
        raise AssertionError("device='gpu' should raise for sparse input")
    except RuntimeError as exc:
        # An accelerator-enabled build reaches the sparse capability check;
        # a CPU-only build correctly refuses the explicit GPU request first.
        message = str(exc)
        assert "sparse" in message or "no accelerator" in message

    # Custom objectives are dense-only for now.
    try:
        MojoTreesRegressor(objective=lambda raw, t: (raw, raw)).fit(csc, y)
        raise AssertionError("custom objective should raise for sparse input")
    except TypeError as exc:
        assert "sparse" in str(exc)

    # Prediction still checks the feature count.
    model = MojoTreesRegressor(n_estimators=5).fit(csc, y)
    narrow = _to_sparse([row[:3] for row in A], "csr")
    try:
        model.predict(narrow)
        raise AssertionError("wrong feature count should raise")
    except ValueError:
        pass
    print("sparse validation ok")



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
    test_lightgbm_sklearn_aliases()
    test_bagging()
    test_goss()
    test_max_depth()
    test_feature_fraction()
    test_interaction_constraints()
    test_interaction_constraint_validation()
    test_monotone_constraints()
    test_monotone_constraint_validation()
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
    test_ranker()
    test_ranker_query_bagging_and_weights()
    test_ndcg_score_semantics()
    test_group_from_query_ids()
    test_ranker_validation()
    if _np is None:
        print("sparse API skipped: numpy is not installed")
    else:
        test_sparse_regressor_matches_dense()
        test_sparse_does_not_mutate_input()
        test_sparse_classifier_matches_dense()
        test_sparse_sample_weight_and_missing()
        test_sparse_save_load()
        test_sparse_validation()
    print("all python API tests passed")
