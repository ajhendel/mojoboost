"""Standalone histogram-kernel benchmark: CPU vs GPU.

Times repeated full-dataset histogram builds on synthetic binned data. This
is the go/no-go signal for deeper GPU work: `gpu_build_s` is the
device-resident rebuild time (gradient upload + kernel + download), the part
that recurs every boosting round; `gpu_first_s` additionally carries binned
matrix upload and context setup, paid once per dataset.

Usage: mojo run -I src bench/bench_histogram.mojo [n_rows] [n_features] [reps]
"""

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojoboost.binning import bin_equal_width
from mojoboost.histogram import build_histogram
from mojoboost.histogram_gpu import GpuHistogramBuilder


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def main() raises:
    var n_rows = 100_000
    var n_features = 100
    var reps = 20
    var args = argv()
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        reps = Int(String(args[3]))

    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))
    var data = bin_equal_width(features, n_rows, n_features, 255)

    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    var base = UInt64(n_rows * n_features)
    for r in range(n_rows):
        grad.append(2.0 * _uniform(base + UInt64(r)) - 1.0)
        hess.append(1.0)

    print("histogram bench:", n_rows, "rows x", n_features, "features,", reps, "reps")

    # Warm-up then timed CPU builds (SIMD + multicore path).
    var cpu_hist = build_histogram(data, grad, hess)
    var t0 = perf_counter_ns()
    for _ in range(reps):
        cpu_hist = build_histogram(data, grad, hess)
    var t1 = perf_counter_ns()
    var cpu_s = Float64(t1 - t0) / 1e9 / Float64(reps)
    print("cpu_build_s:", cpu_s)

    comptime if not has_accelerator():
        print("gpu: no accelerator, skipped")
    else:
        var t2 = perf_counter_ns()
        var builder = GpuHistogramBuilder(data)
        var gpu_hist = builder.build(grad, hess)
        var t3 = perf_counter_ns()
        print("gpu_first_s:", Float64(t3 - t2) / 1e9)

        var t4 = perf_counter_ns()
        for _ in range(reps):
            gpu_hist = builder.build(grad, hess)
        var t5 = perf_counter_ns()
        var gpu_s = Float64(t5 - t4) / 1e9 / Float64(reps)
        print("gpu_build_s:", gpu_s)
        print("gpu_speedup_vs_cpu:", cpu_s / gpu_s)

        # Sanity: totals agree.
        var cg = 0.0
        var gg = 0.0
        for b in range(cpu_hist.n_bins):
            cg += cpu_hist.grad[b]
            gg += gpu_hist.grad[b]
        print("feature0 grad totals (cpu, gpu):", cg, gg)

        _feature_group_arms(builder, reps)


def _quiet_band(times: List[Float64]) -> Float64:
    """How far this arm's median sits above its own best run, in percent.

    The spread between the fastest and the slowest run is not a noise floor
    here: a single descheduled run inflates the maximum by 3x and would bury
    any real difference. The band between the minimum and the median is what
    the arm reproduces, so that is what a difference has to clear.
    """
    var lo = times[0]
    var med = times[len(times) // 2]
    if lo <= 0.0:
        return 0.0
    return 100.0 * (med - lo) / lo


def _feature_group_arms(mut builder: GpuHistogramBuilder, reps: Int) raises:
    """Interleaved A/B of the histogram launch shape: one feature per
    threadgroup against two (`gpu_active_rows.mojo`).

    Both arms build the same root histogram, bit for bit, so this times a
    launch shape and nothing else. The arms alternate inside one process
    because this machine's device timings drift several-fold across time
    windows, which makes two runs minutes apart incomparable; only
    back-to-back interleaved repeats resolve a difference.

    The paired arm only reaches its own kernel on the atomic strategy. On
    the tiled strategy both arms run the same partial kernel and the
    comparison should come out flat, which is worth printing rather than
    hiding: a difference there would mean the arm was not what it said.
    """
    var one = List[Float64](capacity=reps)
    var two = List[Float64](capacity=reps)
    builder.begin_tree()
    # First launch of each kernel pays its compilation; neither is timed.
    for g in range(1, 3):
        builder.set_feature_group(g)
        builder.enqueue_leaf(0)
        builder.synchronize()

    for _ in range(reps):
        builder.set_feature_group(1)
        var a0 = perf_counter_ns()
        builder.enqueue_leaf(0)
        builder.synchronize()
        var a1 = perf_counter_ns()
        one.append(Float64(a1 - a0) / 1e9)

        builder.set_feature_group(2)
        var b0 = perf_counter_ns()
        builder.enqueue_leaf(0)
        builder.synchronize()
        var b1 = perf_counter_ns()
        two.append(Float64(b1 - b0) / 1e9)
    builder.set_feature_group(1)

    sort(one)
    sort(two)
    var band_one = _quiet_band(one)
    var band_two = _quiet_band(two)
    var floor = band_one if band_one > band_two else band_two
    print("feature_group_1_min_ms:", one[0] * 1e3)
    print("feature_group_1_median_ms:", one[len(one) // 2] * 1e3)
    print("feature_group_2_min_ms:", two[0] * 1e3)
    print("feature_group_2_median_ms:", two[len(two) // 2] * 1e3)
    var speedup = one[len(one) // 2] / two[len(two) // 2]
    var delta = 100.0 * (speedup - 1.0)
    print("feature_group_2_vs_1_speedup:", speedup)
    if abs(delta) > floor:
        print("feature_group_verdict: resolved, band_pct", floor)
    else:
        print("feature_group_verdict: indistinguishable, band_pct", floor)
