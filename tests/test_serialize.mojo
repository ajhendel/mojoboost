"""Round-trip tests for model serialization.

A saved-then-loaded model must reproduce the original's predictions
bit-exactly, since floats are stored as raw IEEE-754 bit patterns.
"""

from std.os import remove
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojoboost.boosting import BINARY_LOGISTIC, SQUARED_ERROR, BoosterParams
from mojoboost.model import Model, fit
from mojoboost.serialize import load_model, save_model
from mojoboost.tree import TreeParams

comptime _TMP_PATH = "./.test_model_roundtrip.tmp"


def _make_dataset(
    n_rows: Int, mut features: List[Float64], mut target: List[Float64]
):
    # Two features with an interaction; deterministic pseudo-random values.
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


def _small_params() -> BoosterParams:
    return BoosterParams(20, 0.1, TreeParams(8, 5, 1.0, 1e-3))


def test_roundtrip_regression_predictions_exact() raises:
    var n_rows = 300
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(n_rows, features, target)

    var model = fit(
        features, n_rows, 2, target, SQUARED_ERROR, _small_params(), 64
    )
    save_model(model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)

    assert_equal(len(loaded.booster.trees), len(model.booster.trees))
    assert_equal(loaded.booster.objective, model.booster.objective)
    assert_true(loaded.booster.base_score == model.booster.base_score)
    assert_equal(len(loaded.mapper.edges), len(model.mapper.edges))

    for r in range(0, n_rows, 7):
        var row: List[Float64] = [features[r], features[n_rows + r]]
        assert_true(loaded.predict(row) == model.predict(row))


def test_roundtrip_binary_predictions_exact() raises:
    var n_rows = 300
    var features = List[Float64]()
    var raw_target = List[Float64]()
    _make_dataset(n_rows, features, raw_target)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(1.0 if raw_target[r] > 2.0 else 0.0)

    var model = fit(
        features, n_rows, 2, target, BINARY_LOGISTIC, _small_params(), 64
    )
    save_model(model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)

    for r in range(0, n_rows, 7):
        var row: List[Float64] = [features[r], features[n_rows + r]]
        assert_true(loaded.predict(row) == model.predict(row))
        assert_true(loaded.predict_raw(row) == model.predict_raw(row))


def test_load_rejects_garbage() raises:
    with open(_TMP_PATH, "w") as f:
        f.write("not a model file at all\n")
    with assert_raises():
        _ = load_model(_TMP_PATH)
    remove(_TMP_PATH)


def test_load_rejects_truncated() raises:
    var n_rows = 100
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(n_rows, features, target)
    var model = fit(
        features, n_rows, 2, target, SQUARED_ERROR, _small_params(), 32
    )
    save_model(model, _TMP_PATH)
    var content = open(_TMP_PATH, "r").read()
    var truncated = String("")
    var half = content.byte_length() // 2
    for i in range(half):
        truncated += String(content[byte=i])
    with open(_TMP_PATH, "w") as f:
        f.write(truncated)
    with assert_raises():
        _ = load_model(_TMP_PATH)
    remove(_TMP_PATH)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
