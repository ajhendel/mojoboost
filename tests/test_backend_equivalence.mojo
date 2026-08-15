"""CPU vs GPU histogram backend equivalence.

The GPU backend accumulates in Float32 fixed-point, so agreement with the
Float64 CPU backend is tolerance-based, not bit-exact. Counts are integer on
both backends and must match exactly. Skips (passing) when no accelerator is
present so the suite stays green on CPU-only machines.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.backend import CPU, GPU, build_histogram
from mojotrees.binning import bin_equal_width
from mojotrees.boosting import (
    SQUARED_ERROR,
    BoosterParams,
    IterationRange,
    train,
)
from mojotrees.train_gpu import train_gpu
from mojotrees.device import CPU_DEVICE, GPU_DEVICE
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.model import fit, fit_multiclass
from mojotrees.tree import TreeParams


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _close(a: Float64, b: Float64, scale: Float64) -> Bool:
    # Float32 carries ~7 significant digits; the fixed-point quantization is
    # finer than that. Tolerance is relative to the total magnitude.
    return abs(a - b) <= 1e-4 * scale + 1e-6


def test_gpu_matches_cpu_histogram() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 10_000
        var n_features = 13
        var n_bins = 64

        var features = List[Float64](capacity=n_rows * n_features)
        for k in range(n_rows * n_features):
            features.append(_uniform(UInt64(k)))
        var data = bin_equal_width(features, n_rows, n_features, n_bins)

        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        var base = UInt64(n_rows * n_features)
        var g_mag = 0.0
        var h_mag = 0.0
        for r in range(n_rows):
            var g = 2.0 * _uniform(base + UInt64(r)) - 1.0
            var h = _uniform(base + UInt64(n_rows + r)) + 0.01
            grad.append(g)
            hess.append(h)
            g_mag += abs(g)
            h_mag += abs(h)

        var cpu = build_histogram[CPU](data, grad, hess)
        var gpu = build_histogram[GPU](data, grad, hess)

        assert_equal(cpu.n_features, gpu.n_features)
        assert_equal(cpu.n_bins, gpu.n_bins)
        for i in range(cpu.n_features * cpu.n_bins):
            assert_equal(cpu.count[i], gpu.count[i])
            assert_true(_close(cpu.grad[i], gpu.grad[i], g_mag))
            assert_true(_close(cpu.hess[i], gpu.hess[i], h_mag))


def test_gpu_builder_reuse_is_deterministic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 5_000
        var n_features = 7
        var features = List[Float64](capacity=n_rows * n_features)
        for k in range(n_rows * n_features):
            features.append(_uniform(UInt64(k)))
        var data = bin_equal_width(features, n_rows, n_features, 32)

        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(_uniform(UInt64(r)) - 0.5)
            hess.append(1.0)

        var builder = GpuHistogramBuilder(data)
        var a = builder.build(grad, hess)
        var b = builder.build(grad, hess)
        # Integer accumulation makes repeat builds bit-identical.
        for i in range(a.n_features * a.n_bins):
            assert_equal(a.grad[i], b.grad[i])
            assert_equal(a.hess[i], b.hess[i])
            assert_equal(a.count[i], b.count[i])


def test_model_predict_batch_matches_across_devices() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1_200
        var n_features = 5
        var features = List[Float64](capacity=n_rows * n_features)
        for k in range(n_rows * n_features):
            features.append(_uniform(UInt64(k)))
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            target.append(
                3.0 * features[0 * n_rows + r]
                - 2.0 * features[1 * n_rows + r]
                + features[2 * n_rows + r]
            )
        var params = BoosterParams(15, 0.1, TreeParams(15, 5, 1.0, 1e-3, 0.0))
        var model = fit(
            features, n_rows, n_features, target, SQUARED_ERROR, params
        )
        var rng = IterationRange(0, model.n_iterations())

        # Same trees, same bins, same leaves; the only difference is the
        # device's Float32 accumulation of leaf values.
        for raw in [False, True]:
            var cpu = model.predict_batch(
                features, n_rows, rng, raw_score=raw, device=CPU_DEVICE
            )
            var gpu = model.predict_batch(
                features, n_rows, rng, raw_score=raw, device=GPU_DEVICE
            )
            assert_equal(len(cpu), n_rows)
            assert_equal(len(gpu), n_rows)
            for r in range(n_rows):
                assert_true(abs(cpu[r] - gpu[r]) <= 1e-4)

        var labels = List[Int](capacity=n_rows)
        for r in range(n_rows):
            labels.append(Int(3.0 * features[0 * n_rows + r]) % 3)
        var mc = fit_multiclass(features, n_rows, n_features, labels, 3, params)
        var mc_rng = IterationRange(0, mc.n_iterations())
        for raw in [False, True]:
            var cpu = mc.predict_batch(
                features, n_rows, mc_rng, raw_score=raw, device=CPU_DEVICE
            )
            var gpu = mc.predict_batch(
                features, n_rows, mc_rng, raw_score=raw, device=GPU_DEVICE
            )
            assert_equal(len(cpu), n_rows * 3)
            assert_equal(len(gpu), n_rows * 3)
            for i in range(n_rows * 3):
                assert_true(abs(cpu[i] - gpu[i]) <= 1e-4)
        var proba = mc.predict_batch(
            features, n_rows, mc_rng, device=GPU_DEVICE
        )
        for r in range(n_rows):
            var total = 0.0
            for k in range(3):
                total += proba[r * 3 + k]
            assert_true(abs(total - 1.0) <= 1e-5)


def test_device_split_search_trains_a_close_deterministic_model() raises:
    """Device-side split selection (MOJOTREES_GPU_SPLIT_STRATEGY=device)
    scores gains in Float32, so split decisions can differ from the host
    scan on near-ties and tree shapes are deliberately NOT asserted
    identical. What must hold: the trained model is bit-deterministic run
    to run, and it fits the training data about as well as the CPU model,
    measured in aggregate rather than per row."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = List[Float64](capacity=n_rows * n_features)
        for k in range(n_rows * n_features):
            features.append(_uniform(UInt64(k)))
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var x0 = features[0 * n_rows + r]
            var x1 = features[1 * n_rows + r]
            var x2 = features[2 * n_rows + r]
            target.append(3.0 * x0 - 2.0 * x1 + x2 * x2)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var params = BoosterParams(20, 0.1, TreeParams(15, 20, 1.0, 1e-3))

        var cpu = train(data, target, SQUARED_ERROR, params)
        _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "device")
        var gpu_a = train_gpu(data, target, SQUARED_ERROR, params)
        var gpu_b = train_gpu(data, target, SQUARED_ERROR, params)
        _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "")

        # Bit-deterministic run to run: identical structure and values.
        assert_equal(len(gpu_a.trees), len(cpu.trees))
        assert_equal(len(gpu_a.trees), len(gpu_b.trees))
        for t in range(len(gpu_a.trees)):
            assert_equal(
                len(gpu_a.trees[t].value), len(gpu_b.trees[t].value)
            )
            for i in range(len(gpu_a.trees[t].value)):
                assert_equal(
                    gpu_a.trees[t].feature[i], gpu_b.trees[t].feature[i]
                )
                assert_equal(
                    gpu_a.trees[t].value[i], gpu_b.trees[t].value[i]
                )

        var mse_cpu = 0.0
        var mse_gpu = 0.0
        var diff = 0.0
        var bins = List[Int](capacity=n_features)
        for r in range(n_rows):
            bins.clear()
            for f in range(n_features):
                bins.append(Int(data.bins[f * n_rows + r]))
            var p_cpu = cpu.predict_bins(bins)
            var p_gpu = gpu_a.predict_bins(bins)
            mse_cpu += (p_cpu - target[r]) * (p_cpu - target[r])
            mse_gpu += (p_gpu - target[r]) * (p_gpu - target[r])
            diff += abs(p_cpu - p_gpu)
        mse_cpu /= Float64(n_rows)
        mse_gpu /= Float64(n_rows)
        # The two models minimize the same loss over the same bins; a
        # Float32 near-tie can reroute rows, so the claim is aggregate fit,
        # not per-row agreement.
        assert_true(mse_gpu <= mse_cpu * 1.10 + 1e-6)
        assert_true(diff / Float64(n_rows) <= 0.05)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
