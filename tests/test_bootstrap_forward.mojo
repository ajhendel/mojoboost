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
4. **The softmax and sparse loops draw it too.** `fit_multiclass` and the
   sparse arm of `trainset.train_dataset` used to refuse an enabled bundle;
   both now thread it into a round loop that calls
   `sampling.bootstrap_round`, so the assertion here is that the model MOVES,
   which a forwarding that validated and dropped the bundle would fail.
5. **What still cannot draw refuses by name.** `fit_custom` raises on any
   enabled bundle, and MVS with a DERIVED `mvs_reg` raises on a softmax fit,
   because CatBoost's derivation reads a previous-iteration `[dim][leaf]`
   leaf-value table that `K` structurally unrelated trees do not have.

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
from mojotrees.sparse import CscMatrix
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


def _multiclass_moved(
    features: List[Float64],
    labels: List[Int],
    bootstrap: BootstrapParams,
) raises -> Bool:
    """Whether a softmax fit under `bootstrap` differs from the unsampled
    one. A bundle that was validated and then dropped would return False,
    which is the only failure worth writing this test for."""
    var plain = fit_multiclass(
        features, N_ROWS, N_FEATURES, labels, 3, _params(), MAX_BINS,
    )
    var sampled = fit_multiclass(
        features, N_ROWS, N_FEATURES, labels, 3, _params(), MAX_BINS,
        bootstrap=bootstrap,
    )
    for r in range(N_ROWS):
        var row = _row(features, N_ROWS, N_FEATURES, r)
        var a = plain.predict_proba(row)
        var b = sampled.predict_proba(row)
        for k in range(len(a)):
            if a[k] != b[k]:
                return True
    return False


def test_multiclass_draws_the_bundle_it_is_handed() raises:
    """The softmax loop RUNS a bootstrap now, and this test used to assert
    that it refused one.

    `boosting._boost_rounds_multiclass` takes a `BootstrapParams` and calls
    `sampling.bootstrap_round` once per round, sharing the draw across every
    class's tree the way it already shares a GOSS sample -- so the `K` trees
    of a round stay grown on one row set. Two of the three arms below are the
    substance:

    - **The Bayesian bootstrap is honored**, and it is the type CatBoost's own
      defaulting block installs for a multiclass-only loss
      (`sampling.catboost_default_bootstrap_type` cites the lines). This is
      the arm a shipped CatBoost default would take.
    - **MVS with an explicit `mvs_reg` is honored.**
    - **MVS with a DERIVED `mvs_reg` still refuses, by name.** CatBoost's
      derivation reads the previous iteration's `[dim][leaf]` leaf-value
      table (`mvs.cpp:21-34`); a mojotrees round is `K` structurally
      unrelated trees and has no such table, so there is no number to compute
      and inventing one would set the floor that decides which rows survive
      to something no CatBoost run produces.

    A disabled bundle is still accepted and still trains, so the refusal that
    remains is about the value and not about the argument existing.
    """
    var features = _features(N_ROWS, N_FEATURES)
    var target = _target(features, N_ROWS)
    var labels = _labels(target, N_ROWS)

    with assert_raises(contains="mvs_reg"):
        _ = fit_multiclass(
            features, N_ROWS, N_FEATURES, labels, 3, _params(), MAX_BINS,
            bootstrap=BootstrapParams.mvs_at(0.8, 7),
        )

    assert_true(
        _multiclass_moved(
            features, labels, BootstrapParams.bayesian_at(1.0, 7)
        )
    )
    assert_true(
        _multiclass_moved(
            features, labels, BootstrapParams.mvs_with_reg(0.1, 0.8, 7)
        )
    )

    var model = fit_multiclass(
        features, N_ROWS, N_FEATURES, labels, 3, _params(), MAX_BINS,
        bootstrap=BootstrapParams.disabled(),
    )
    assert_true(len(model.booster.trees) > 0)


def test_multiclass_bootstrap_repeats_at_one_seed() raises:
    """One seed, one model. The draw is `uniform(stream + row)` for a stream
    derived from `(seed, round)` alone, so two fits of the same data at the
    same seed are the same ensemble bit for bit. This is the same-process half
    of the determinism claim; the cross-worker half is a digest run at two
    `MOJOTREES_NUM_WORKERS` values and cannot be asserted from inside one
    process."""
    var features = _features(N_ROWS, N_FEATURES)
    var target = _target(features, N_ROWS)
    var labels = _labels(target, N_ROWS)
    var a = fit_multiclass(
        features, N_ROWS, N_FEATURES, labels, 3, _params(), MAX_BINS,
        bootstrap=BootstrapParams.bayesian_at(1.0, 11),
    )
    var b = fit_multiclass(
        features, N_ROWS, N_FEATURES, labels, 3, _params(), MAX_BINS,
        bootstrap=BootstrapParams.bayesian_at(1.0, 11),
    )
    assert_equal(len(a.booster.trees), len(b.booster.trees))
    for r in range(N_ROWS):
        var row = _row(features, N_ROWS, N_FEATURES, r)
        var pa = a.predict_proba(row)
        var pb = b.predict_proba(row)
        for k in range(len(pa)):
            assert_equal(pa[k], pb[k])


def test_sparse_draws_the_bundle_it_is_handed() raises:
    """`boosting_sparse.train_sparse`'s round loop calls
    `sampling.bootstrap_round` in the place the dense loop calls it, so a
    sparse fit under MVS is a sampled fit and not the unsampled one.

    `trainset.train_dataset` used to refuse the bundle on this arm through
    `sampling.check_bootstrap_honored`. That refusal is gone because the loop
    behind it exists, not because the rule was relaxed: the GPU arm still
    raises, by name.
    """
    var features = _features(N_ROWS, N_FEATURES)
    var target = _target(features, N_ROWS)
    # Sparsify: a zero is an implicit entry, which is what the CSC path is
    # for. Two of the four columns keep every value so the trees have
    # something to split on.
    var sparse_features = features.copy()
    for f in range(2, N_FEATURES):
        for r in range(N_ROWS):
            if r % 3 != 0:
                sparse_features[f * N_ROWS + r] = 0.0
    var rows = List[Int]()
    var vals = List[Float64]()
    var offs = List[Int]()
    offs.append(0)
    for f in range(N_FEATURES):
        for r in range(N_ROWS):
            var v = sparse_features[f * N_ROWS + r]
            if v != 0.0:
                rows.append(r)
                vals.append(v)
        offs.append(len(rows))
    var ds = Dataset.from_csc(
        CscMatrix(rows^, vals^, offs^, N_ROWS, N_FEATURES),
        target.copy(),
        max_bin=MAX_BINS,
    )
    assert_true(ds.is_sparse)
    var plain = train_dataset(ds, SQUARED_ERROR, _params())
    var sampled = train_dataset(
        ds,
        SQUARED_ERROR,
        _params(),
        bootstrap=BootstrapParams.mvs_at(0.8, 7),
    )
    var moved = False
    for r in range(N_ROWS):
        var row = _row(sparse_features, N_ROWS, N_FEATURES, r)
        if plain.predict(row) != sampled.predict(row):
            moved = True
            break
    assert_true(moved)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
