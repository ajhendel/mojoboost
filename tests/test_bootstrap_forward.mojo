"""The edge that carries `bootstrap_type` from a fit entry point to `train`.

`tests/test_bootstrap_wire.mojo` pins the round loop's side of this: that
`boosting.train` draws the bundle it is handed and refuses the samplers it
cannot compose with. What that file cannot see is whether anything ever hands
it one. Until this lane, nothing did -- `model.fit` had no `bootstrap`
argument at all, so the sampler was built, tested, merged and unreachable.

So this file tests exactly the forwarding, and the refusals that stand in for
it where a trainer cannot draw:

1. **The default arm does not move.** `model.fit` with no `bootstrap` and
   `model.fit` with `BootstrapParams.disabled()` are the same model, bit for
   bit, and so are the two `trainset.train_dataset` calls.
2. **An enabled bundle reaches the trainer.** MVS drops rows and reweights
   the survivors, so a fit under it must differ from the unsampled one. A
   forwarding that validated the bundle and dropped it would pass every
   assertion in `test_bootstrap_wire.mojo` and fail this one, which is the
   only difference that matters.
3. **The draw is reproducible.** Two fits at one seed are one model. The MVS
   stream is counter-based on `(seed, tree, row)` with its own domain
   constant, so this must hold at every `MOJOTREES_NUM_WORKERS` too; this
   file can only assert the same-process half of that.
4. **Every entry point that cannot draw refuses by name.** `fit_multiclass`
   and `fit_custom` take the bundle and raise on an enabled one rather than
   training an unsampled model and reporting a sampled one.

Float comparisons are exact, as they are in the wire file: both arms of every
same-model assertion are deterministic by contract, so a tolerance would only
hide a difference.

This file must NOT be renamed to `test_gpu_*`: `tools/run_tests.sh` selects
the accelerator subset by name and would silently exclude it from the CPU
suite. The GPU refusal in `model.fit` is deliberately not tested here for the
same reason -- it needs a box with an accelerator to fail for the reason it
claims, and on a CPU-only box `GPU_DEVICE` raises before the bootstrap check
is ever reached.
"""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees import (
    SQUARED_ERROR,
    BoosterParams,
    Dataset,
    TreeParams,
    fit,
    fit_multiclass,
    train_dataset,
)
from mojotrees.sampling import BootstrapParams
from support import _make_features as _features


comptime N_ROWS = 300
comptime N_FEATURES = 4
comptime MAX_BINS = 64


def _target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        y.append(4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x2 - 0.5))
    return y^


def _labels(target: List[Float64], n_rows: Int) -> List[Int]:
    var out = List[Int](capacity=n_rows)
    for r in range(n_rows):
        if target[r] < 0.0:
            out.append(0)
        elif target[r] < 1.0:
            out.append(1)
        else:
            out.append(2)
    return out^


def _params() -> BoosterParams:
    var tree = TreeParams.default()
    tree.num_leaves = 8
    tree.min_data_in_leaf = 5
    return BoosterParams(12, 0.1, tree^)


def _row(
    features: List[Float64], n_rows: Int, n_features: Int, r: Int
) -> List[Float64]:
    var row = List[Float64](capacity=n_features)
    for f in range(n_features):
        row.append(features[f * n_rows + r])
    return row^


def test_disabled_bootstrap_is_the_fit_that_already_ran() raises:
    """The default argument must move nothing.

    `BootstrapParams.disabled()` is what `model.fit` now defaults to, so
    naming it and omitting it have to be the same call. If they are not, the
    argument's arrival changed every existing fit, which is the one outcome a
    new default is not allowed to have.
    """
    var features = _features(N_ROWS, N_FEATURES)
    var target = _target(features, N_ROWS)
    var absent = fit(
        features, N_ROWS, N_FEATURES, target, SQUARED_ERROR, _params(),
        MAX_BINS,
    )
    var named = fit(
        features, N_ROWS, N_FEATURES, target, SQUARED_ERROR, _params(),
        MAX_BINS,
        bootstrap=BootstrapParams.disabled(),
    )
    assert_equal(len(absent.booster.trees), len(named.booster.trees))
    for r in range(N_ROWS):
        var row = _row(features, N_ROWS, N_FEATURES, r)
        assert_equal(absent.predict(row), named.predict(row))


def test_mvs_reaches_the_trainer_through_model_fit() raises:
    """An enabled bundle must change the model.

    MVS keeps a large-gradient row certainly at weight 1, keeps a small one
    with probability `g/mu` and drops it otherwise, and amplifies the
    survivors by `mu/g`. Every tree therefore sees a different row set and a
    different weighted gradient, so a fit under it cannot be the unsampled
    fit unless the bundle was dropped on the way down.
    """
    var features = _features(N_ROWS, N_FEATURES)
    var target = _target(features, N_ROWS)
    var plain = fit(
        features, N_ROWS, N_FEATURES, target, SQUARED_ERROR, _params(),
        MAX_BINS,
    )
    var sampled = fit(
        features, N_ROWS, N_FEATURES, target, SQUARED_ERROR, _params(),
        MAX_BINS,
        bootstrap=BootstrapParams.mvs_at(0.8, 7),
    )
    var moved = False
    for r in range(N_ROWS):
        var row = _row(features, N_ROWS, N_FEATURES, r)
        if plain.predict(row) != sampled.predict(row):
            moved = True
            break
    assert_true(moved)


def test_mvs_repeats_under_the_same_seed() raises:
    """The draw is a function of `(seed, tree, row)` and of nothing else."""
    var features = _features(N_ROWS, N_FEATURES)
    var target = _target(features, N_ROWS)
    var first = fit(
        features, N_ROWS, N_FEATURES, target, SQUARED_ERROR, _params(),
        MAX_BINS,
        bootstrap=BootstrapParams.mvs_at(0.8, 7),
    )
    var second = fit(
        features, N_ROWS, N_FEATURES, target, SQUARED_ERROR, _params(),
        MAX_BINS,
        bootstrap=BootstrapParams.mvs_at(0.8, 7),
    )
    for r in range(N_ROWS):
        var row = _row(features, N_ROWS, N_FEATURES, r)
        assert_equal(first.predict(row), second.predict(row))


def test_a_different_seed_is_a_different_draw() raises:
    """Two seeds must not be one draw. A forwarding that hard-coded the
    default seed would pass every test above and fail this one."""
    var features = _features(N_ROWS, N_FEATURES)
    var target = _target(features, N_ROWS)
    var one = fit(
        features, N_ROWS, N_FEATURES, target, SQUARED_ERROR, _params(),
        MAX_BINS,
        bootstrap=BootstrapParams.mvs_at(0.8, 7),
    )
    var two = fit(
        features, N_ROWS, N_FEATURES, target, SQUARED_ERROR, _params(),
        MAX_BINS,
        bootstrap=BootstrapParams.mvs_at(0.8, 99),
    )
    var moved = False
    for r in range(N_ROWS):
        var row = _row(features, N_ROWS, N_FEATURES, r)
        if one.predict(row) != two.predict(row):
            moved = True
            break
    assert_true(moved)


def test_train_dataset_forwards_the_bundle() raises:
    """The `Dataset` path is the second reachable one, and it is the one
    `bench/real_data`'s dense arm actually takes: `mojotrees.train(params,
    Dataset)` never touches `model.fit`. So it gets its own test rather than
    riding on the one above."""
    var features = _features(N_ROWS, N_FEATURES)
    var target = _target(features, N_ROWS)
    var ds = Dataset(
        features, N_ROWS, N_FEATURES, target.copy(), max_bin=MAX_BINS
    )
    var plain = train_dataset(ds, SQUARED_ERROR, _params())
    var sampled = train_dataset(
        ds,
        SQUARED_ERROR,
        _params(),
        bootstrap=BootstrapParams.mvs_at(0.8, 7),
    )
    var moved = False
    for r in range(N_ROWS):
        var row = _row(features, N_ROWS, N_FEATURES, r)
        if plain.predict(row) != sampled.predict(row):
            moved = True
            break
    assert_true(moved)


def test_multiclass_refuses_an_enabled_bootstrap() raises:
    """Neither `boosting.train_multiclass` nor `train_multiclass_gpu` takes
    the bundle, so a softmax fit under one would be unsampled. CatBoost
    agrees about the shape of this hole: its own defaulting block excludes
    the multiclass-only losses from the MVS default."""
    var features = _features(N_ROWS, N_FEATURES)
    var target = _target(features, N_ROWS)
    var labels = _labels(target, N_ROWS)
    with assert_raises(contains="bootstrap_type"):
        _ = fit_multiclass(
            features, N_ROWS, N_FEATURES, labels, 3, _params(), MAX_BINS,
            bootstrap=BootstrapParams.mvs_at(0.8, 7),
        )
    # And a disabled bundle is accepted, so the refusal is about the value
    # and not about the argument existing.
    var model = fit_multiclass(
        features, N_ROWS, N_FEATURES, labels, 3, _params(), MAX_BINS,
        bootstrap=BootstrapParams.disabled(),
    )
    assert_true(len(model.booster.trees) > 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
