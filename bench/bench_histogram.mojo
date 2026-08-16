"""Standalone histogram-kernel benchmark: CPU vs GPU.

Times repeated full-dataset histogram builds on synthetic binned data. This
is the go/no-go signal for deeper GPU work: `gpu_build_s` is the
device-resident rebuild time (gradient upload + kernel + download), the part
that recurs every boosting round; `gpu_first_s` additionally carries binned
matrix upload and context setup, paid once per dataset.

Two interleaved A/Bs of the histogram launch run after it, both alternating
their arms inside this process because this machine's device timings drift
several-fold across time windows and only adjacent samples compare:
`_feature_group_arms` on how many feature slots one threadgroup accumulates,
and `_row_unroll_arms` on how many rows one thread keeps in flight. Neither
knob can change a histogram, and each function says why in its own docstring.

Usage: mojo run -I src bench/bench_histogram.mojo [n_rows] [n_features] [reps]
"""

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojotrees.binning import bin_equal_width
from mojotrees.histogram import build_histogram
from mojotrees.histogram_gpu import GpuHistogramBuilder


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
            cg += cpu_hist.grad_at(b)
            gg += gpu_hist.grad_at(b)
        print("feature0 grad totals (cpu, gpu):", cg, gg)

        # The group the builder resolved for this dataset, read before the
        # feature-group A/B moves it. `_feature_group_arms` leaves the
        # builder on group 1, which is not the default on every backend, and
        # the row-walk A/B has no business being measured under a launch
        # shape the trainer would not have chosen.
        var default_group = builder.feature_group()
        _feature_group_arms(builder, reps)
        builder.set_feature_group(default_group)
        print("row_unroll_feature_group:", default_group)
        _row_unroll_arms(builder, reps)


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

    Both strategies now resolve a real difference. That was not true while
    the paired arm had its own hand-written kernel and only the atomic
    strategy dispatched to it, so on the tiled strategy both arms ran the
    identical partial kernel and the comparison was expected to come out
    flat. Both strategies now launch one parameterized kernel instantiated
    at the arm's group width, so a flat result here is a measurement rather
    than a tautology, and a difference on either strategy is real.
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


def _join_ms(times: List[Float64]) -> String:
    """Every repeat in milliseconds, space separated, sorted as passed in."""
    var out = String("")
    for i in range(len(times)):
        if i > 0:
            out += " "
        out += String(times[i] * 1e3)
    return out^


def _row_unroll_arms(mut builder: GpuHistogramBuilder, reps: Int) raises:
    """Interleaved A/B of the histogram row walk: `HIST_ROW_UNROLL` rows in
    flight per thread against one row per iteration
    (`GpuActiveRows.set_row_unroll`).

    **This is a launch shape and it cannot change a histogram.** Both arms
    visit exactly the same rows of the same range and add exactly the same
    fixed-point integers into the same bins; only the order of the adds, and
    of the loads that feed them, differs, and integer addition is associative
    and commutative. `set_row_unroll`'s own docstring is where that argument
    is written out, and it is the reason the arm names below say `on` and
    `off` rather than anything suggesting two results. Nothing here checks
    the two histograms against each other, because there is nothing for such
    a check to catch: it would be asserting that integer addition is still
    associative.

    What the arms do differ in is instruction count and memory-level
    parallelism. The row loop's per-row work is a chain of dependent gathers,
    so written one row at a time a thread has one request outstanding and
    runs at the latency of the gather rather than the bandwidth of it; the
    unrolled arm issues four stages' loads together. Against that, the
    unrolled arm holds more live registers, and threadgroup residency on this
    backend is bounded by quantities this project cannot query. Which of
    those wins is exactly the open question, and it is why the knob is a
    runtime argument rather than a comptime one: this machine's device
    timings drift several-fold across time windows, so two builds compared
    minutes apart cannot settle it and only back-to-back interleaved repeats
    can.

    The knob is reached as `builder.rows.set_row_unroll` rather than through
    a forwarder on the builder, because `GpuHistogramBuilder` has no
    `set_row_unroll` of its own the way it has `set_feature_group`, and this
    lane does not own `src/`. A forwarder mirroring `set_feature_group` is
    the clean version and is a two-line edit somebody else should make.

    Every repeat is printed, not just the reduction, for the same reason
    bench_train_gpu.mojo prints its samples: a dispersion that has to be
    recovered by hand is a dispersion that gets dropped when a result is
    copied into a table.
    """
    var on = List[Float64](capacity=reps)
    var off = List[Float64](capacity=reps)
    builder.begin_tree()
    # Both arms are one kernel instantiation reading a runtime flag, so
    # neither pays a compilation the other does not; the untimed pair below
    # is still run so that the first timed sample of each arm is not the
    # first launch of the session.
    for _ in range(2):
        builder.rows.set_row_unroll(True)
        builder.enqueue_leaf(0)
        builder.synchronize()
        builder.rows.set_row_unroll(False)
        builder.enqueue_leaf(0)
        builder.synchronize()

    for _ in range(reps):
        builder.rows.set_row_unroll(True)
        var a0 = perf_counter_ns()
        builder.enqueue_leaf(0)
        builder.synchronize()
        var a1 = perf_counter_ns()
        on.append(Float64(a1 - a0) / 1e9)

        builder.rows.set_row_unroll(False)
        var b0 = perf_counter_ns()
        builder.enqueue_leaf(0)
        builder.synchronize()
        var b1 = perf_counter_ns()
        off.append(Float64(b1 - b0) / 1e9)
    # Leave the builder on the module default rather than on whichever arm
    # ran last, so a later section of this benchmark cannot inherit an arm.
    builder.rows.set_row_unroll(True)

    print("row_unroll_on_samples_ms:", _join_ms(on))
    print("row_unroll_off_samples_ms:", _join_ms(off))
    sort(on)
    sort(off)
    var band_on = _quiet_band(on)
    var band_off = _quiet_band(off)
    var floor = band_on if band_on > band_off else band_off
    print("row_unroll_on_min_ms:", on[0] * 1e3)
    print("row_unroll_on_median_ms:", on[len(on) // 2] * 1e3)
    print("row_unroll_off_min_ms:", off[0] * 1e3)
    print("row_unroll_off_median_ms:", off[len(off) // 2] * 1e3)
    var speedup = off[len(off) // 2] / on[len(on) // 2]
    var delta = 100.0 * (speedup - 1.0)
    print("row_unroll_on_vs_off_speedup:", speedup)
    if abs(delta) > floor:
        print("row_unroll_verdict: resolved, band_pct", floor)
    else:
        print("row_unroll_verdict: indistinguishable, band_pct", floor)
