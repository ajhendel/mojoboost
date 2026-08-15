"""Linear trees (src/mojotrees/linear_tree.mojo) reached through the metric
fit path, `Model` prediction, and the v5 model file.

The dataset is affine in its two features with a mild interaction, so a
constant-leaf ensemble of tiny trees underfits it and a linear-leaf ensemble
of the same trees does not; the test does not fix a number for that, only
that the linear model's training error is strictly smaller and every
prediction is finite. The round trip is exact: a v5 file reloads to a model
whose predictions match the fitted one to the bit, and a constant-leaf model
still writes v4.
"""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees import BoosterParams, SQUARED_ERROR, TreeParams
from mojotrees.boosting import IterationRange
from mojotrees.custom_metric import (
    CustomMetric,
    MetricSuite,
    RawValidSet,
    fit_with_metrics,
)
from mojotrees.linear_tree import LinearParams, linear_tree_available
from mojotrees.model import Model, fit
from mojotrees.serialize import load_model, save_model

comptime _N = 64
comptime _TMP_PATH = "./.test_linear_tree_roundtrip.tmp"


def _features() -> List[Float64]:
    # Column-major: feature 0 is a ramp, feature 1 a slower ramp offset by
    # a period-4 sawtooth so the two are not collinear.
    var out = List[Float64](capacity=2 * _N)
    for r in range(_N):
        out.append(Float64(r) / 8.0)
    for r in range(_N):
        out.append(Float64(r % 4) + Float64(r) / 32.0)
    return out^


def _target(x: List[Float64]) -> List[Float64]:
    var out = List[Float64](capacity=_N)
    for r in range(_N):
        var a = x[r]
        var b = x[_N + r]
        out.append(1.5 + 2.0 * a - 0.75 * b + 0.05 * a * b)
    return out^


def _mse(pred: List[Float64], target: List[Float64]) raises -> Float64:
    var total = 0.0
    for i in range(len(target)):
        var d = pred[i] - target[i]
        total += d * d
    return total / Float64(len(target))


def _mse_metric(
    metric: Int, valid: Int, pred: List[Float64], target: List[Float64]
) raises -> Float64:
    return _mse(pred, target)


def _params(linear: LinearParams) -> BoosterParams:
    return BoosterParams(
        8, 0.5, TreeParams(3, 4, 1.0, 1e-3), linear=linear.copy()
    )


def _fit(
    linear: LinearParams, x: List[Float64], y: List[Float64]
) raises -> Model:
    var valid = List[RawValidSet]()
    valid.append(RawValidSet("train", x.copy(), _N, y.copy()))
    var result = fit_with_metrics(
        x,
        _N,
        2,
        y,
        valid,
        SQUARED_ERROR,
        _params(linear),
        MetricSuite([CustomMetric("mse")], _mse_metric, 0),
        max_bins=16,
    )
    return result.model.copy()


def test_linear_leaves_fit_an_affine_target_better() raises:
    assert_true(linear_tree_available())
    var x = _features()
    var y = _target(x)
    var constant = _fit(LinearParams.disabled(), x, y)
    var linear = _fit(LinearParams(enabled=True, linear_lambda=0.1), x, y)

    assert_true(not constant.booster.linear.is_active())
    assert_true(linear.booster.linear.is_active())
    assert_true(linear.booster.linear.n_linear_leaves() > 0)
    assert_equal(len(linear.booster.trees), len(constant.booster.trees))

    var full = IterationRange(0, len(linear.booster.trees))
    var pred_c = constant.predict_batch(x, _N, full)
    var pred_l = linear.predict_batch(x, _N, full)
    var differ = False
    for r in range(_N):
        assert_true(pred_l[r] == pred_l[r])
        assert_true(abs(pred_l[r]) < 1e6)
        if pred_l[r] != pred_c[r]:
            differ = True
        # The batched and per-row entry points agree exactly.
        var row: List[Float64] = [x[r], x[_N + r]]
        assert_equal(linear.predict(row), pred_l[r])
        assert_equal(linear.predict_raw_range(row, full), pred_l[r])
    assert_true(differ)
    assert_true(_mse(pred_l, y) < _mse(pred_c, y))


def test_linear_model_round_trips_through_a_v5_file() raises:
    var x = _features()
    var y = _target(x)
    var linear = _fit(LinearParams(enabled=True, linear_lambda=0.1), x, y)
    save_model(linear, _TMP_PATH)
    var text = open(_TMP_PATH, "r").read()
    assert_true(text.startswith("mojotrees v5"))
    var loaded = load_model(_TMP_PATH)
    assert_true(loaded.booster.linear.is_active())
    assert_equal(
        loaded.booster.linear.n_linear_leaves(),
        linear.booster.linear.n_linear_leaves(),
    )
    var full = IterationRange(0, len(linear.booster.trees))
    var a = linear.predict_batch(x, _N, full)
    var b = loaded.predict_batch(x, _N, full)
    for r in range(_N):
        assert_equal(a[r], b[r])

    # A constant-leaf model is written exactly as before: v4, no section.
    var constant = _fit(LinearParams.disabled(), x, y)
    save_model(constant, _TMP_PATH)
    var text_c = open(_TMP_PATH, "r").read()
    assert_true(text_c.startswith("mojotrees v4"))
    assert_true(text_c.find("linear") < 0)


def test_binned_only_entry_points_refuse_linear_trees() raises:
    var x = _features()
    var y = _target(x)
    with assert_raises(contains="linear trees are not supported"):
        _ = fit(
            x, _N, 2, y, SQUARED_ERROR,
            _params(LinearParams(enabled=True)),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
