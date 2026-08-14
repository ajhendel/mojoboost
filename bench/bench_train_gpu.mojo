"""End-to-end CPU vs GPU training benchmark for mojoboost.

Generates the same deterministic synthetic dataset as bench_train.mojo
(counter-based splitmix64), bins it once, then times complete boosted
training — including GPU initialization and all transfers — through both
the CPU trainer (`train`) and the GPU trainer (`train_gpu`), with the
library defaults (LightGBM-matched). Also reports each model's training
loss so throughput is never read apart from fit quality.

CPU-side threading honors MOJOBOOST_NUM_WORKERS / MOJOBOOST_PARALLEL_MIN_OPS
(see parallel.mojo), so pin those for reproducible comparisons.

Usage: mojo run -I src bench/bench_train_gpu.mojo [n_rows] [n_features] [reg|binary]
"""

from std.math import exp, log
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojoboost.binning import fit_bins
from mojoboost.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train,
)
from mojoboost.binning import BinnedMatrix
from mojoboost.train_gpu import train_gpu


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _sigmoid(x: Float64) -> Float64:
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    var e = exp(x)
    return e / (1.0 + e)


def _train_loss(
    booster: Booster,
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
) -> Float64:
    var loss = 0.0
    for r in range(data.n_rows):
        if objective == BINARY_LOGISTIC:
            var p = booster.predict_row(data, r)
            if p < 1e-15:
                p = 1e-15
            if p > 1.0 - 1e-15:
                p = 1.0 - 1e-15
            if target[r] > 0.5:
                loss -= log(p)
            else:
                loss -= log(1.0 - p)
        else:
            var d = booster.predict_row(data, r) - target[r]
            loss += d * d
    return loss / Float64(data.n_rows)


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; GPU training benchmark skipped")
    else:
        var n_rows = 100_000
        var n_features = 100
        var objective = SQUARED_ERROR
        var obj_name = String("reg")
        var args = argv()
        if len(args) > 1:
            n_rows = Int(String(args[1]))
        if len(args) > 2:
            n_features = Int(String(args[2]))
        if len(args) > 3:
            obj_name = String(args[3])
            if obj_name == "binary":
                objective = BINARY_LOGISTIC
            elif obj_name != "reg":
                raise Error("objective must be 'reg' or 'binary'")
        if n_features < 4:
            raise Error("need at least 4 features")

        # Same data as bench_train.mojo: column-major features, target from
        # features 0..3 plus a noise stream at counters >= n_rows * n_features.
        var features = List[Float64](capacity=n_rows * n_features)
        for k in range(n_rows * n_features):
            features.append(_uniform(UInt64(k)))
        var noise_base = UInt64(n_rows * n_features)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var x0 = features[0 * n_rows + r]
            var x1 = features[1 * n_rows + r]
            var x2 = features[2 * n_rows + r]
            var x3 = features[3 * n_rows + r]
            var signal = (
                5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
            )
            var u = _uniform(noise_base + UInt64(r))
            if objective == BINARY_LOGISTIC:
                var p = _sigmoid(2.0 * (signal - 3.0))
                target.append(1.0 if u < p else 0.0)
            else:
                target.append(signal + 0.1 * (u - 0.5))

        print(
            "mojoboost gpu-vs-cpu bench:",
            n_rows,
            "rows x",
            n_features,
            "features,",
            obj_name,
        )

        var t0 = perf_counter_ns()
        var mapper = fit_bins(features, n_rows, n_features, 255)
        var data = mapper.transform(features, n_rows)
        var t1 = perf_counter_ns()
        print("binning_s:", Float64(t1 - t0) / 1e9)

        var t2 = perf_counter_ns()
        var cpu = train(data, target, objective, BoosterParams.default())
        var t3 = perf_counter_ns()
        var cpu_s = Float64(t3 - t2) / 1e9
        print("cpu_train_s:", cpu_s)
        print("cpu_n_trees:", len(cpu.trees))
        print("cpu_train_loss:", _train_loss(cpu, data, target, objective))

        # Includes GpuHistogramBuilder construction and every transfer.
        var t4 = perf_counter_ns()
        var gpu = train_gpu(data, target, objective, BoosterParams.default())
        var t5 = perf_counter_ns()
        var gpu_s = Float64(t5 - t4) / 1e9
        print("gpu_train_s:", gpu_s)
        print("gpu_n_trees:", len(gpu.trees))
        print("gpu_train_loss:", _train_loss(gpu, data, target, objective))
        print("gpu_speedup_x:", cpu_s / gpu_s)
