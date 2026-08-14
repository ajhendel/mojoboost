"""Round-trip tests for model serialization.

A saved-then-loaded model must reproduce the original's predictions
bit-exactly, since floats are stored as raw IEEE-754 bit patterns.
"""

from std.os import remove
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojoboost.boosting import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    FAIR,
    GAMMA,
    MAPE,
    SQUARED_ERROR,
    TWEEDIE,
    BoosterParams,
    IterationRange,
)
from mojoboost.contrib import predict_contrib
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


def test_roundtrip_preserves_node_covers_exactly() raises:
    # v3 carries per-node training row counts (see tree.mojo). They are not
    # recoverable from a fitted tree, and exact feature contributions divide
    # by them, so the round-trip has to be exact rather than close.
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

    for t in range(len(model.booster.trees)):
        ref want = model.booster.trees[t]
        ref got = loaded.booster.trees[t]
        assert_true(got.has_node_counts())
        assert_equal(len(got.count), len(want.count))
        for i in range(len(want.count)):
            assert_true(got.count[i] == want.count[i])


def test_roundtrip_preserves_contributions_exactly() raises:
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

    var rng = IterationRange.slice(
        model.n_iterations(), 0, model.n_iterations()
    )
    for r in range(0, n_rows, 29):
        var row: List[Float64] = [features[r], features[n_rows + r]]
        var want = predict_contrib(model, row, rng)
        var got = predict_contrib(loaded, row, rng)
        assert_equal(len(got), len(want))
        for f in range(len(want)):
            assert_true(got[f] == want[f])


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


def test_objective_link_survives_a_round_trip() raises:
    # The objective code is what tells a loaded model which inverse link to
    # apply, so a round trip has to preserve the response scale, not just
    # the trees. Gamma and cross entropy are the two links added most
    # recently: exp and the logistic.
    var n_rows = 60
    var features = List[Float64]()
    var target = List[Float64]()
    _make_dataset(n_rows, features, target)

    var counts = List[Float64](capacity=n_rows)
    var probabilities = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        # Strictly positive for gamma, and inside [0, 1] for cross entropy.
        counts.append(1.0 + abs(target[r]))
        probabilities.append(0.5 + 0.25 * (1.0 if target[r] > 1.0 else -1.0))

    var objectives: List[Int] = [GAMMA, TWEEDIE, MAPE, FAIR, CROSS_ENTROPY]
    for o in range(len(objectives)):
        var objective = objectives[o]
        var labels = (
            probabilities.copy() if objective
            == CROSS_ENTROPY else counts.copy()
        )
        var alpha = 1.5 if objective == TWEEDIE else 1.0
        var model = fit(
            features,
            n_rows,
            2,
            labels,
            objective,
            _small_params(),
            32,
            alpha=alpha,
        )
        save_model(model, _TMP_PATH)
        var loaded = load_model(_TMP_PATH)
        remove(_TMP_PATH)
        assert_equal(loaded.booster.objective, objective)
        for r in range(n_rows):
            var row: List[Float64] = [features[r], features[n_rows + r]]
            # Response scale, so a lost objective code would show up as a
            # missing exp or sigmoid rather than as a small numeric drift.
            assert_equal(loaded.predict(row), model.predict(row))
            assert_equal(loaded.predict_raw(row), model.predict_raw(row))


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



# A model file is untrusted input: it may have been written by another tool,
# edited, or truncated in transit. These build malformed files by hand and
# require the loader to raise rather than read past an array, spin forever,
# or decode a wrapped integer into an arbitrary value. The round-trip tests
# above cannot reach any of it, because they only ever load what this
# library just wrote.

def _bits(x: Float64) -> String:
    return String(x.to_bits())


def _write(content: String) raises:
    with open(_TMP_PATH, "w") as f:
        f.write(content)


def _v1_prefix() -> String:
    """A well-formed v1 header, up to but not including the mapper."""
    var s = String("mojoboost v1\nobjective 0\n")
    s += "learning_rate " + _bits(0.1) + "\n"
    s += "base_score " + _bits(0.0) + "\n"
    return s^


def test_loads_v1_file() raises:
    """Version-1 files stay readable: no missing routing, no categories."""
    var s = String("mojoboost v1\nobjective 0\n")
    s += "learning_rate " + _bits(0.5) + "\n"
    s += "base_score " + _bits(1.0) + "\n"
    s += "mapper 1 4 1\n"
    s += _bits(2.0) + "\n"
    s += "0 1\n"
    s += "trees 1\n"
    s += "tree 3 2\n"
    s += "0 -1 -1\n"
    s += "0 -1 -1\n"
    s += "1 -1 -1\n"
    s += "2 -1 -1\n"
    s += _bits(0.0) + " " + _bits(10.0) + " " + _bits(20.0) + "\n"
    _write(s)
    var m = load_model(_TMP_PATH)
    remove(_TMP_PATH)

    assert_equal(m.mapper.n_features, 1)
    assert_equal(m.booster.trees[0].missing_bin[0], -1)
    # Values <= 2.0 take the left leaf, above it the right one, each scaled
    # by the learning rate on top of the base score.
    var low: List[Float64] = [1.0]
    var high: List[Float64] = [3.0]
    assert_equal(m.predict(low), 1.0 + 0.5 * 10.0)
    assert_equal(m.predict(high), 1.0 + 0.5 * 20.0)


def test_load_rejects_unknown_version() raises:
    _write(String("mojoboost v99\nobjective 0\n"))
    with assert_raises():
        _ = load_model(_TMP_PATH)
    remove(_TMP_PATH)


def test_load_rejects_edge_offset_out_of_range() raises:
    """`BinMapper.bin_value` binary-searches the edge slice with no bounds
    check, so an offset past the edge array would read out of bounds."""
    var s = _v1_prefix()
    s += "mapper 2 4 2\n"
    s += _bits(1.0) + " " + _bits(2.0) + "\n"
    s += "0 1000000 2\n"
    s += "trees 0\n"
    _write(s)
    with assert_raises():
        _ = load_model(_TMP_PATH)
    remove(_TMP_PATH)


def test_load_rejects_descending_edge_offsets() raises:
    var s = _v1_prefix()
    s += "mapper 2 4 2\n"
    s += _bits(1.0) + " " + _bits(2.0) + "\n"
    s += "0 2 1\n"
    s += "trees 0\n"
    _write(s)
    with assert_raises():
        _ = load_model(_TMP_PATH)
    remove(_TMP_PATH)


def test_load_rejects_cyclic_child() raises:
    """A node pointing at itself would spin `predict_row` forever, so the
    loader requires children to point strictly forward."""
    var s = _v1_prefix()
    s += "mapper 1 4 1\n"
    s += _bits(1.0) + "\n"
    s += "0 1\n"
    s += "trees 1\n"
    s += "tree 3 2\n"
    s += "0 -1 -1\n"
    s += "0 -1 -1\n"
    s += "0 -1 -1\n"
    s += "2 -1 -1\n"
    s += _bits(0.0) + " " + _bits(1.0) + " " + _bits(2.0) + "\n"
    _write(s)
    with assert_raises():
        _ = load_model(_TMP_PATH)
    remove(_TMP_PATH)


def test_load_rejects_bad_bin_count() raises:
    var s = _v1_prefix()
    s += "mapper 1 -5 1\n"
    s += _bits(1.0) + "\n"
    s += "0 1\n"
    s += "trees 0\n"
    _write(s)
    with assert_raises():
        _ = load_model(_TMP_PATH)
    remove(_TMP_PATH)


def test_load_rejects_oversized_integer_token() raises:
    """A digit run that would wrap the 64-bit accumulator has to raise, not
    decode to an arbitrary value."""
    var s = _v1_prefix()
    s += "mapper 1 4 1\n"
    s += "999999999999999999999999 \n"
    s += "0 1\n"
    s += "trees 0\n"
    _write(s)
    with assert_raises():
        _ = load_model(_TMP_PATH)
    remove(_TMP_PATH)

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
