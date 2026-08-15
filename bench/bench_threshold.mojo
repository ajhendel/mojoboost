"""Multicore-threshold sweep for histogram accumulation.

At each shape, times the forced-serial path (MOJOTREES_NUM_WORKERS=1)
against the forced-parallel path (auto workers with
MOJOTREES_PARALLEL_MIN_OPS=1) on identical data, then restores auto mode.
The row where the speedup crosses 1.0 is the measured threshold candidate
for this machine; PARALLEL_MIN_OPS should sit near that crossover's
features * rows, tuned by broad CPU family rather than individual chip.

Usage: mojo run -I src bench/bench_threshold.mojo [n_features] [max_rows]
"""

from std.os import setenv
from std.sys import argv
from std.time import perf_counter_ns

from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.histogram import build_histogram


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _timed_builds(
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64], reps: Int
) raises -> Float64:
    var hist = build_histogram(data, grad, hess)  # warm-up
    _ = hist.grad[0]
    var t0 = perf_counter_ns()
    for _ in range(reps):
        hist = build_histogram(data, grad, hess)
    var t1 = perf_counter_ns()
    return Float64(t1 - t0) / 1e9 / Float64(reps)


def main() raises:
    var n_features = 50
    var max_rows = 32_000
    var args = argv()
    if len(args) > 1:
        n_features = Int(String(args[1]))
    if len(args) > 2:
        max_rows = Int(String(args[2]))

    print("threshold sweep:", n_features, "features, serial vs forced-parallel")
    print("n_rows | ops | serial_s | parallel_s | parallel_speedup")

    var n_rows = 250
    while n_rows <= max_rows:
        var features = List[Float64](capacity=n_rows * n_features)
        for k in range(n_rows * n_features):
            features.append(_uniform(UInt64(k)))
        var data = bin_equal_width(features, n_rows, n_features, 255)

        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(2.0 * _uniform(UInt64(n_rows * n_features + r)) - 1.0)
            hess.append(1.0)

        var ops = n_rows * n_features
        # More reps for tiny shapes so timings are stable.
        var reps = 4_000_000 // ops
        if reps < 5:
            reps = 5

        _ = setenv("MOJOTREES_NUM_WORKERS", "1")
        var serial_s = _timed_builds(data, grad, hess, reps)

        _ = setenv("MOJOTREES_NUM_WORKERS", "0")
        _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "1")
        var parallel_s = _timed_builds(data, grad, hess, reps)
        _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")

        print(n_rows, "|", ops, "|", serial_s, "|", parallel_s, "|", serial_s / parallel_s)
        n_rows *= 2

    _ = setenv("MOJOTREES_NUM_WORKERS", "")
