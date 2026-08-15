"""Early stopping on the GPU: the device-gradient path composes with the
validation scorer.

`train_gpu_with_valid` under `OBJECTIVE_SOURCE_AUTO` now generates a plain
run's gradients on the device, exactly as `train_gpu` does. These tests pin
what that composition must preserve: the stopping decision (the number of
trees kept) agrees with the host-gradient path and with the CPU trainer,
the kept models agree to Float32-level tolerance, a bagged run takes the
device round through `GpuTreeRouter` and still grows on the CPU trainer's
rows, and a GOSS run keeps the host path its ranking needs. Skips (passing) when no
accelerator is present.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.bagging import BaggingParams
from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train_with_valid,
)
from mojotrees.goss import GossParams
from mojotrees.train_gpu import (
    OBJECTIVE_SOURCE_AUTO,
    OBJECTIVE_SOURCE_DEVICE,
    OBJECTIVE_SOURCE_HOST,
    VALID_SCORE_DEVICE,
    VALID_SCORE_HOST,
    device_gradients,
    train_gpu_with_valid,
)
from mojotrees.tree import TreeParams


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _features(n_rows: Int, n_features: Int, offset: Int) -> List[Float64]:
    var out = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        out.append(_uniform(UInt64(offset + k)))
    return out^


def _target(
    features: List[Float64], n_rows: Int, binary: Bool
) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        var s = 4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x3 - 0.5)
        if binary:
            y.append(1.0 if s > 0.5 else 0.0)
        else:
            y.append(s)
    return y^


def _max_abs_pred_gap(a: Booster, b: Booster, data: BinnedMatrix) -> Float64:
    var worst = 0.0
    for r in range(data.n_rows):
        var d = abs(a.predict_row(data, r) - b.predict_row(data, r))
        if d > worst:
            worst = d
    return worst


def test_auto_source_is_device_for_plain_valid_run() raises:
    """The composition is on by default: a plain early-stopping run
    generates its gradients on the device, exactly as `train_gpu` does."""
    assert_true(
        device_gradients(
            SQUARED_ERROR,
            1,
            OBJECTIVE_SOURCE_AUTO,
            BaggingParams.disabled(),
            GossParams.disabled(),
        )
    )
    # A bagged run is on the device round too once the trainer routes every
    # row through GpuTreeRouter, which is what the trainers now do under
    # AUTO (`_routes_all_rows`); a GOSS run keeps the host path its ranking
    # needs.
    assert_true(
        device_gradients(
            SQUARED_ERROR,
            1,
            OBJECTIVE_SOURCE_AUTO,
            BaggingParams(0.7, 1, 3),
            GossParams.disabled(),
            routes_all_rows=True,
        )
    )
    assert_true(
        not device_gradients(
            SQUARED_ERROR,
            1,
            OBJECTIVE_SOURCE_AUTO,
            BaggingParams.disabled(),
            GossParams.enable(),
        )
    )


def test_device_gradients_keep_the_stopping_decision() raises:
    """Device gradients versus host gradients, both with the host scorer:
    the same number of trees kept, and models within Float32 tolerance of
    each other and of the CPU trainer's."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4_000
        var n_valid = 1_000
        var n_features = 6
        var x = _features(n_rows, n_features, 0)
        var y = _target(x, n_rows, False)
        var vx = _features(n_valid, n_features, 1_000_003)
        var vy = _target(vx, n_valid, False)
        var data = bin_equal_width(x, n_rows, n_features, 64)
        var valid = bin_equal_width(vx, n_valid, n_features, 64)
        var params = BoosterParams(60, 0.3, TreeParams(15, 20, 1.0, 1e-3))

        var cpu = train_with_valid(
            data, y, valid, vy, SQUARED_ERROR, params, 5
        )
        var host = train_gpu_with_valid(
            data,
            y,
            valid,
            vy,
            SQUARED_ERROR,
            params,
            5,
            valid_scoring=VALID_SCORE_HOST,
            objective_source=OBJECTIVE_SOURCE_HOST,
        )
        var dev = train_gpu_with_valid(
            data,
            y,
            valid,
            vy,
            SQUARED_ERROR,
            params,
            5,
            valid_scoring=VALID_SCORE_HOST,
            objective_source=OBJECTIVE_SOURCE_DEVICE,
        )
        assert_true(len(cpu.trees) > 0)
        assert_equal(len(host.trees), len(cpu.trees))
        assert_equal(len(dev.trees), len(cpu.trees))
        assert_true(_max_abs_pred_gap(host, dev, valid) <= 1e-3)
        assert_true(_max_abs_pred_gap(cpu, dev, valid) <= 1e-3)


def test_device_gradients_with_device_scorer() raises:
    """Both switches on: gradients and validation scores device-resident,
    same stopping decision as the all-host run on this problem."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4_000
        var n_valid = 1_000
        var n_features = 6
        var x = _features(n_rows, n_features, 0)
        var y = _target(x, n_rows, False)
        var vx = _features(n_valid, n_features, 1_000_003)
        var vy = _target(vx, n_valid, False)
        var data = bin_equal_width(x, n_rows, n_features, 64)
        var valid = bin_equal_width(vx, n_valid, n_features, 64)
        var params = BoosterParams(60, 0.3, TreeParams(15, 20, 1.0, 1e-3))

        var host = train_gpu_with_valid(
            data,
            y,
            valid,
            vy,
            SQUARED_ERROR,
            params,
            5,
            valid_scoring=VALID_SCORE_HOST,
            objective_source=OBJECTIVE_SOURCE_HOST,
        )
        var dev = train_gpu_with_valid(
            data,
            y,
            valid,
            vy,
            SQUARED_ERROR,
            params,
            5,
            valid_scoring=VALID_SCORE_DEVICE,
            objective_source=OBJECTIVE_SOURCE_DEVICE,
        )
        assert_equal(len(dev.trees), len(host.trees))
        assert_true(_max_abs_pred_gap(host, dev, valid) <= 1e-3)


def test_binary_device_gradients_stop_like_the_host() raises:
    """Binary logistic: the device derivative kernel and the host
    definition stop at the same round."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4_000
        var n_valid = 1_000
        var n_features = 6
        var x = _features(n_rows, n_features, 7)
        var y = _target(x, n_rows, True)
        var vx = _features(n_valid, n_features, 2_000_003)
        var vy = _target(vx, n_valid, True)
        var data = bin_equal_width(x, n_rows, n_features, 64)
        var valid = bin_equal_width(vx, n_valid, n_features, 64)
        var params = BoosterParams(60, 0.3, TreeParams(15, 20, 1.0, 1e-3))

        var cpu = train_with_valid(
            data, y, valid, vy, BINARY_LOGISTIC, params, 5
        )
        var dev = train_gpu_with_valid(
            data,
            y,
            valid,
            vy,
            BINARY_LOGISTIC,
            params,
            5,
            objective_source=OBJECTIVE_SOURCE_DEVICE,
        )
        assert_equal(len(dev.trees), len(cpu.trees))
        assert_true(_max_abs_pred_gap(cpu, dev, valid) <= 1e-3)


def test_bagged_valid_run_matches_cpu_under_auto() raises:
    """A bagged early-stopping run under AUTO takes the device round and
    still grows on the CPU trainer's rows: same tree count, same shapes."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4_000
        var n_valid = 1_000
        var n_features = 6
        var x = _features(n_rows, n_features, 0)
        var y = _target(x, n_rows, False)
        var vx = _features(n_valid, n_features, 1_000_003)
        var vy = _target(vx, n_valid, False)
        var data = bin_equal_width(x, n_rows, n_features, 64)
        var valid = bin_equal_width(vx, n_valid, n_features, 64)
        var params = BoosterParams(40, 0.3, TreeParams(15, 20, 1.0, 1e-3))
        var bagging = BaggingParams(0.6, 2, 99)

        var cpu = train_with_valid(
            data, y, valid, vy, SQUARED_ERROR, params, 5, bagging=bagging
        )
        var gpu = train_gpu_with_valid(
            data, y, valid, vy, SQUARED_ERROR, params, 5, bagging=bagging
        )
        assert_equal(len(gpu.trees), len(cpu.trees))
        for t in range(len(cpu.trees)):
            assert_equal(cpu.trees[t].n_leaves, gpu.trees[t].n_leaves)
        assert_true(_max_abs_pred_gap(cpu, gpu, valid) <= 1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
