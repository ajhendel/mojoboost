"""End-to-end GPU training equivalence.

GPU histograms carry Float32 fixed-point precision, so trained-model
agreement with the Float64 CPU trainer is tolerance-based, not bit-exact;
counts and tree shapes must match exactly when split decisions agree. The
GPU trainer itself must be bit-deterministic run to run. Skips (passing)
when no accelerator is present so the suite stays green on CPU-only
machines.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojoboost.binning import bin_equal_width
from mojoboost.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BoosterParams,
    train,
)
from mojoboost.histogram import build_histogram_subset
from mojoboost.histogram_gpu import GpuHistogramBuilder
from mojoboost.train_gpu import train_gpu
from mojoboost.tree import TreeParams


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _make_features(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major deterministic features in [0, 1)."""
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))
    return features^


def _regression_target(
    features: List[Float64], n_rows: Int
) -> List[Float64]:
    """Strong, distinct per-feature effects so CPU and GPU split decisions
    do not sit on knife-edge gain ties."""
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        y.append(4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x3 - 0.5))
    return y^


def test_gpu_subset_histogram_and_partition_match_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 8_000
        var n_features = 5
        var n_bins = 64
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, n_bins)

        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        var g_mag = 0.0
        var h_mag = 0.0
        var base = UInt64(n_rows * n_features)
        for r in range(n_rows):
            var g = 2.0 * _uniform(base + UInt64(r)) - 1.0
            var h = _uniform(base + UInt64(n_rows + r)) + 0.01
            grad.append(g)
            hess.append(h)
            g_mag += abs(g)
            h_mag += abs(h)

        # Split the root on (feature 0, bin <= 31) device-side, then build
        # both children's histograms by leaf filter.
        var builder = GpuHistogramBuilder(data)
        builder.upload_gradients(grad, hess)
        builder.begin_tree()
        builder.apply_split(0, 31, 0, 1, 2)
        var gpu_left = builder.build_leaf(1)
        var gpu_right = builder.build_leaf(2)

        # CPU reference: partition rows on the host, build subsets directly.
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        for r in range(n_rows):
            if data.bin_at(r, 0) <= 31:
                left_rows.append(r)
            else:
                right_rows.append(r)
        var cpu_left = build_histogram_subset(data, grad, hess, left_rows)
        var cpu_right = build_histogram_subset(data, grad, hess, right_rows)

        for i in range(n_features * n_bins):
            assert_equal(cpu_left.count[i], gpu_left.count[i])
            assert_equal(cpu_right.count[i], gpu_right.count[i])
            assert_true(
                abs(cpu_left.grad[i] - gpu_left.grad[i]) <= 1e-4 * g_mag + 1e-6
            )
            assert_true(
                abs(cpu_left.hess[i] - gpu_left.hess[i]) <= 1e-4 * h_mag + 1e-6
            )
            assert_true(
                abs(cpu_right.grad[i] - gpu_right.grad[i])
                <= 1e-4 * g_mag + 1e-6
            )
            assert_true(
                abs(cpu_right.hess[i] - gpu_right.hess[i])
                <= 1e-4 * h_mag + 1e-6
            )


def test_gpu_regression_training_matches_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(15, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var cpu = train(data, target, SQUARED_ERROR, params)
        var gpu = train_gpu(data, target, SQUARED_ERROR, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        var cpu_sse = 0.0
        var gpu_sse = 0.0
        for r in range(n_rows):
            var pc = cpu.predict_row(data, r)
            var pg = gpu.predict_row(data, r)
            assert_true(abs(pc - pg) <= 1e-3)
            cpu_sse += (pc - target[r]) * (pc - target[r])
            gpu_sse += (pg - target[r]) * (pg - target[r])
        # Same-quality fit, not just pointwise agreement.
        assert_true(abs(cpu_sse - gpu_sse) <= 1e-3 * (cpu_sse + 1e-12))


def test_gpu_binary_training_matches_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var x0 = features[0 * n_rows + r]
            var x1 = features[1 * n_rows + r]
            var x2 = features[2 * n_rows + r]
            target.append(
                1.0 if 2.0 * x0 - x1 + 0.5 * x2 > 0.7 else 0.0
            )
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(15, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var cpu = train(data, target, BINARY_LOGISTIC, params)
        var gpu = train_gpu(data, target, BINARY_LOGISTIC, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        for r in range(n_rows):
            var pc = cpu.predict_row(data, r)
            var pg = gpu.predict_row(data, r)
            assert_true(abs(pc - pg) <= 1e-3)


def test_gpu_training_is_deterministic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(8, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var a = train_gpu(data, target, SQUARED_ERROR, params)
        var b = train_gpu(data, target, SQUARED_ERROR, params)

        assert_equal(len(a.trees), len(b.trees))
        # Integer accumulation makes repeat GPU training bit-identical.
        for r in range(n_rows):
            assert_equal(
                a.predict_raw_row(data, r), b.predict_raw_row(data, r)
            )


def test_gpu_l1_regularized_training_matches_cpu() raises:
    """`lambda_l1` is applied host-side to downloaded histogram sums, so the
    GPU trainer must reproduce the CPU trainer's regularized splits and
    shrunken leaf values, not just the unregularized ones."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var l1 = BoosterParams(15, 0.1, TreeParams(15, 20, 1.0, 1e-3, 2.0))
        var cpu = train(data, target, SQUARED_ERROR, l1)
        var gpu = train_gpu(data, target, SQUARED_ERROR, l1)

        assert_equal(len(cpu.trees), len(gpu.trees))
        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-3
            )

        # The regularizer must actually bite: leaf values shrink toward
        # zero, so the regularized fit moves less far from the base score.
        var plain = BoosterParams(15, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var gpu_plain = train_gpu(data, target, SQUARED_ERROR, plain)
        var l1_travel = 0.0
        var plain_travel = 0.0
        for r in range(n_rows):
            l1_travel += abs(
                gpu.predict_raw_row(data, r) - gpu.base_score
            )
            plain_travel += abs(
                gpu_plain.predict_raw_row(data, r) - gpu_plain.base_score
            )
        assert_true(l1_travel < plain_travel)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
