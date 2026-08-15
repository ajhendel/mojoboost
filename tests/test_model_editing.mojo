"""Tests for editing a fitted model in place (model_editing.mojo).

Each test states what an edit must leave true: rollback removes exactly the
last tree's contribution, a leaf write moves every row in that leaf by
exactly the shrunk delta and no other row at all, a shuffle keeps the
ensemble's sum, a refit at decay 1.0 is the identity, and the bounds contain
every prediction the model makes.
"""

from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojotrees.boosting import BoosterParams, IterationRange, SQUARED_ERROR
from mojotrees.model import Model, fit
from mojotrees.model_editing import (
    EDIT_MODE_DART,
    EDIT_MODE_GBDT,
    LEAF_EDIT_REJECT,
    MODEL_EDITING_SUPPORTED,
    RefitParams,
    editing_capabilities,
    get_leaf_output,
    model_editing_status_json,
    raw_score_bounds,
    refit,
    rollback_one_iter,
    set_leaf_output,
    shuffle_iterations,
)
from mojotrees.tree import TreeParams


def _make_dataset(
    n_rows: Int, mut features: List[Float64], mut target: List[Float64]
):
    var state: UInt64 = 42
    for _ in range(n_rows):
        state = state * 6364136223846793005 + 1442695040888963407
        features.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    for _ in range(n_rows):
        state = state * 6364136223846793005 + 1442695040888963407
        features.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    for r in range(n_rows):
        target.append(
            3.0 * features[r] + 2.0 * features[n_rows + r] * features[r]
        )


def _params() -> BoosterParams:
    return BoosterParams(12, 0.1, TreeParams(8, 5, 1.0, 1e-3))


def _row(features: List[Float64], n_rows: Int, r: Int) -> List[Float64]:
    return [features[r], features[n_rows + r]]


def _fit_model(
    n_rows: Int, mut features: List[Float64], mut target: List[Float64]
) raises -> Model:
    _make_dataset(n_rows, features, target)
    return fit(features, n_rows, 2, target, SQUARED_ERROR, _params(), 64)


def test_status_says_supported() raises:
    assert_true(MODEL_EDITING_SUPPORTED)
    var status = model_editing_status_json()
    assert_true(status.startswith("{\"supported\":true"))
    var caps = editing_capabilities()
    var saw_leaf = False
    for i in range(len(caps)):
        if caps[i].operation == "set_leaf_output":
            saw_leaf = True
            assert_true(caps[i].supported)
    assert_true(saw_leaf)


def test_rollback_one_iter_removes_exactly_the_last_tree() raises:
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_model(n_rows, features, target)
    var n = len(model.booster.trees)
    assert_true(n >= 2)
    var data = model.mapper.transform(features, n_rows)
    var last = model.booster.trees[n - 1].copy()
    var lr = model.booster.learning_rate

    var before = List[Float64]()
    for r in range(0, n_rows, 9):
        before.append(model.predict(_row(features, n_rows, r)))

    var remaining = rollback_one_iter(model.booster, EDIT_MODE_GBDT)
    assert_equal(remaining, n - 1)
    assert_equal(len(model.booster.trees), n - 1)

    var i = 0
    for r in range(0, n_rows, 9):
        var after = model.predict(_row(features, n_rows, r))
        var expected = before[i] - lr * last.predict_row(data, r)
        assert_true(abs(after - expected) <= 1e-12 * (1.0 + abs(expected)))
        i += 1

    with assert_raises():
        _ = rollback_one_iter(model.booster, EDIT_MODE_DART)


def test_set_leaf_output_moves_only_that_leaf_by_the_shrunk_delta() raises:
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_model(n_rows, features, target)
    var data = model.mapper.transform(features, n_rows)
    var lr = model.booster.learning_rate
    ref tree0 = model.booster.trees[0]

    # Pick the leaf row 0 lands in, and a row that lands elsewhere.
    var bins0 = List[Int]()
    for f in range(2):
        bins0.append(data.bin_at(0, f))
    var ordinal = tree0.leaf_ordinal_bins(bins0)
    var other = -1
    for r in range(1, n_rows):
        var bins = List[Int]()
        for f in range(2):
            bins.append(data.bin_at(r, f))
        if tree0.leaf_ordinal_bins(bins) != ordinal:
            other = r
            break
    assert_true(other > 0)

    var old = get_leaf_output(model.booster, 0, ordinal)
    var p0 = model.predict(_row(features, n_rows, 0))
    var p_other = model.predict(_row(features, n_rows, other))

    var stored = set_leaf_output(model.booster, 0, ordinal, old + 0.5)
    assert_true(stored == old + 0.5)
    assert_true(get_leaf_output(model.booster, 0, ordinal) == old + 0.5)

    var q0 = model.predict(_row(features, n_rows, 0))
    assert_true(abs((q0 - p0) - lr * 0.5) <= 1e-12)
    assert_true(model.predict(_row(features, n_rows, other)) == p_other)

    # A leaf that does not exist, and a non-finite value, are refused.
    with assert_raises():
        _ = set_leaf_output(model.booster, 0, 10_000, 0.0)
    with assert_raises():
        _ = set_leaf_output(model.booster, 0, ordinal, 0.0 / 0.0)


def test_shuffle_keeps_the_ensemble_sum() raises:
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_model(n_rows, features, target)
    var n = len(model.booster.trees)
    var before = List[Float64]()
    for r in range(0, n_rows, 9):
        before.append(model.predict(_row(features, n_rows, r)))
    shuffle_iterations(model.booster, 7, IterationRange.slice(n, 0, n))
    assert_equal(len(model.booster.trees), n)
    var i = 0
    for r in range(0, n_rows, 9):
        var after = model.predict(_row(features, n_rows, r))
        assert_true(abs(after - before[i]) <= 1e-12 * (1.0 + abs(before[i])))
        i += 1


def test_refit_at_decay_one_is_the_identity_and_bounds_hold() raises:
    var n_rows = 200
    var features = List[Float64]()
    var target = List[Float64]()
    var model = _fit_model(n_rows, features, target)
    var data = model.mapper.transform(features, n_rows)
    var before = List[Float64]()
    for r in range(0, n_rows, 9):
        before.append(model.predict(_row(features, n_rows, r)))

    var params = RefitParams.default()
    params.decay_rate = 1.0
    var report = refit(model.booster, data, target, params)
    assert_equal(report.n_trees, len(model.booster.trees))
    var i = 0
    for r in range(0, n_rows, 9):
        assert_true(model.predict(_row(features, n_rows, r)) == before[i])
        i += 1

    # A real refit (decay 0) changes leaves and keeps every prediction
    # inside the ensemble's bounds.
    params.decay_rate = 0.0
    report = refit(model.booster, data, target, params)
    assert_true(report.n_leaves_updated > 0)
    var bounds = raw_score_bounds(model.booster)
    for r in range(0, n_rows, 9):
        var p = model.predict(_row(features, n_rows, r))
        assert_true(p >= bounds.lower - 1e-12 and p <= bounds.upper + 1e-12)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
