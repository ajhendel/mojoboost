"""Python-side reach for the features the integration round wired: linear
leaves, `tree_learner` over the in-process world, the advanced LambdaRank
parameters with `position`, and Arrow / polars input. Each has a native
suite of its own; what is checked here is that the estimator parameter or
the input type actually reaches it, by moving the fit or by predicting the
same values as the plain path.

Runs under pytest and as a script (`python test_wired_features.py`).
"""

import numpy as np
import pytest

from mojotrees import MojoTreesRanker, MojoTreesRegressor


def _regression(n_rows=300, seed=7):
    gen = np.random.default_rng(seed)
    X = gen.random((n_rows, 4))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] * X[:, 2] + 0.05 * gen.standard_normal(
        n_rows
    )
    return X, y


def _ranking(n_queries=12, per_query=6, seed=3):
    gen = np.random.default_rng(seed)
    n_rows = n_queries * per_query
    X = gen.random((n_rows, 4))
    score = 2.0 * X[:, 0] + X[:, 1]
    y = np.zeros(n_rows, dtype=np.int64)
    for q in range(n_queries):
        lo = q * per_query
        block = score[lo : lo + per_query]
        y[lo : lo + per_query] = np.argsort(np.argsort(block)) // 2
    return X, y, [per_query] * n_queries


def test_linear_tree_moves_the_fit_and_survives_a_round_trip(tmp_path):
    X, y = _regression()
    plain = MojoTreesRegressor(n_estimators=6, num_leaves=5).fit(X, y)
    linear = MojoTreesRegressor(
        n_estimators=6, num_leaves=5, linear_tree=True, linear_lambda=0.1
    ).fit(X, y)
    assert list(linear.predict(X[:20])) != list(plain.predict(X[:20]))
    path = tmp_path / "linear.mbst"
    linear.booster_.save_model(str(path))
    from mojotrees import Booster

    back = Booster(model_file=str(path))
    np.testing.assert_array_equal(back.predict(X), linear.predict(X))


def test_linear_lambda_must_be_nonnegative():
    X, y = _regression()
    with pytest.raises(ValueError):
        MojoTreesRegressor(linear_tree=True, linear_lambda=-1.0).fit(X, y)


@pytest.mark.parametrize("learner", ["data", "feature", "voting"])
def test_tree_learner_trains_over_an_in_process_world(learner):
    X, y = _regression()
    serial = MojoTreesRegressor(n_estimators=5, num_leaves=5).fit(X, y)
    world = MojoTreesRegressor(
        n_estimators=5, num_leaves=5, tree_learner=learner, num_machines=2,
        top_k=2 if learner == "voting" else 20,
    ).fit(X, y)
    got = world.predict(X)
    assert got.shape == (len(y),)
    assert np.isfinite(got).all()
    if learner in ("data", "feature"):
        # Data- and feature-parallel growth over a local world elects the
        # same splits as the serial grower; voting is allowed to differ.
        np.testing.assert_allclose(got, serial.predict(X), rtol=1e-9)


def test_tree_learner_needs_a_world():
    X, y = _regression()
    with pytest.raises(ValueError, match="num_machines"):
        MojoTreesRegressor(tree_learner="data").fit(X, y)
    with pytest.raises(ValueError, match="unknown tree_learner"):
        MojoTreesRegressor(tree_learner="magic", num_machines=2).fit(X, y)


def test_label_gain_and_position_reach_the_ranker():
    X, y, group = _ranking()
    base = MojoTreesRanker(n_estimators=6, num_leaves=6).fit(X, y, group=group)
    gained = MojoTreesRanker(
        n_estimators=6, num_leaves=6, label_gain=[0.0, 5.0, 40.0]
    ).fit(X, y, group=group)
    assert list(gained.predict(X[:12])) != list(base.predict(X[:12]))
    position = np.tile(np.arange(6), 12)
    unbiased = MojoTreesRanker(
        n_estimators=6, num_leaves=6,
        lambdarank_position_bias_regularization=0.5,
    ).fit(X, y, group=group, position=position)
    assert np.isfinite(unbiased.predict(X)).all()
    assert list(unbiased.predict(X[:12])) != list(base.predict(X[:12]))


def test_arrow_and_polars_inputs_predict_like_numpy():
    X, y = _regression()
    plain = MojoTreesRegressor(n_estimators=5, num_leaves=5).fit(X, y)
    pa = pytest.importorskip("pyarrow")
    table = pa.table({f"f{i}": X[:, i] for i in range(4)})
    arrow = MojoTreesRegressor(n_estimators=5, num_leaves=5).fit(table, y)
    np.testing.assert_array_equal(arrow.predict(table), plain.predict(X))
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({f"f{i}": X[:, i] for i in range(4)})
    polars = MojoTreesRegressor(n_estimators=5, num_leaves=5).fit(frame, y)
    np.testing.assert_array_equal(polars.predict(frame), plain.predict(X))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
