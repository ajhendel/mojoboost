"""Per-stage CPU profile of a boosting round.

Times every stage the CPU backend spends real work in, each one twice on
identical data: once with `MOJOTREES_NUM_WORKERS=1` (forced serial) and once
in auto mode (the shipped scheduler). The ratio of the two is the multicore
speedup attributable to that stage alone, which is what tuning the scheduler
moves; the serial column is what single-core optimization moves.

Stages, in the order a boosting round touches them:

- `bin_fit`        quantile edge fitting, per feature (sort-bound)
- `bin_transform`  raw values to bin ids, per feature (binary-search-bound)
- `grad_hess`      per-row gradients and hessians (elementwise)
- `hist_full`      root histogram over every row (scatter-bound)
- `hist_subset50`  child histogram over half the rows
- `hist_subset10`  child histogram over a tenth of the rows
- `hist_subtract`  sibling histogram by subtraction (SIMD elementwise)
- `split_scan`     best-split search over a built histogram
- `partition`      routing a node's rows to its two children
- `grow_tree`      one whole tree, as an integration check on the above
- `predict`        scoring every row through a grown tree

Dataset generation matches bench_train.mojo and bench_lightgbm.py, so a
profile and an end-to-end run describe the same data.

Usage: mojo run -I src bench/bench_profile.mojo [n_rows] [n_features] [reps]

Report the machine header with any numbers taken from this tool. The stage
times are wall-clock on a shared machine: run it on an idle box, and treat a
single run as an estimate rather than a measurement.
"""

from std.os import setenv
from std.sys import argv
from std.sys.info import (
    CompilationTarget,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    simd_width_of,
)
from std.time import perf_counter_ns

from mojotrees.binning import BinMapper, BinnedMatrix, fit_bins
from mojotrees.boosting import SQUARED_ERROR, fill_grad_hess
from mojotrees.histogram import (
    Histogram,
    SIMD_LANES,
    build_histogram,
    build_histogram_subset,
    subtract_histogram,
)
from mojotrees.parallel import (
    PARALLEL_MIN_OPS,
    TASKS_PER_CORE,
    env_parallel_min_ops,
    plan_tasks,
)
from mojotrees.split import find_best_split
from mojotrees.tree import TreeParams, grow_tree, partition_rows


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _serial():
    _ = setenv("MOJOTREES_NUM_WORKERS", "1")


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "0")


def _report(name: String, serial_s: Float64, auto_s: Float64):
    var speedup = serial_s / auto_s if auto_s > 0.0 else 0.0
    print(name, "|", serial_s, "|", auto_s, "|", speedup)


def _machine_header() raises:
    var isa = String("unknown")
    if CompilationTarget.has_avx512f():
        isa = String("x86-64 avx512f")
    elif CompilationTarget.has_avx2():
        isa = String("x86-64 avx2")
    elif CompilationTarget.has_neon():
        isa = String("arm64 neon")
    print("isa:", isa)
    print(
        "cores: physical",
        num_physical_cores(),
        "logical",
        num_logical_cores(),
        "performance",
        num_performance_cores(),
    )
    print(
        "simd: float64 width",
        simd_width_of[DType.float64](),
        "kernel lanes",
        SIMD_LANES,
    )
    print(
        "scheduler: tasks_per_core",
        TASKS_PER_CORE,
        "parallel_min_ops",
        env_parallel_min_ops(),
        "(default",
        PARALLEL_MIN_OPS,
        ")",
    )


def main() raises:
    var n_rows = 100_000
    var n_features = 100
    var reps = 3
    var args = argv()
    if len(args) > 1:
        n_rows = Int(String(args[1]))
    if len(args) > 2:
        n_features = Int(String(args[2]))
    if len(args) > 3:
        reps = Int(String(args[3]))
    if n_features < 4:
        raise Error("need at least 4 features")
    if reps < 1:
        raise Error("reps must be positive")

    _machine_header()
    print("shape:", n_rows, "rows x", n_features, "features, reps", reps)
    print(
        "auto-mode task counts: features",
        plan_tasks(n_features, n_features * n_rows),
        "rows",
        plan_tasks(n_rows, n_rows),
    )
    print("")

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
        var signal = 5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
        target.append(signal + 0.1 * (_uniform(noise_base + UInt64(r)) - 0.5))

    print("stage | serial_s | auto_s | speedup")

    # --- binning -----------------------------------------------------------
    var t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var m = fit_bins(features, n_rows, n_features, 255)
        _ = m.n_features
    var t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var m2 = fit_bins(features, n_rows, n_features, 255)
        _ = m2.n_features
    var t2 = perf_counter_ns()
    _report(
        "bin_fit",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var mapper = fit_bins(features, n_rows, n_features, 255)

    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var d = mapper.transform(features, n_rows)
        _ = d.n_rows
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var d2 = mapper.transform(features, n_rows)
        _ = d2.n_rows
    t2 = perf_counter_ns()
    _report(
        "bin_transform",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var data = mapper.transform(features, n_rows)

    # --- gradient generation ----------------------------------------------
    var raw = List[Float64](capacity=n_rows)
    raw.resize(n_rows, 0.5)
    var grad = List[Float64]()
    var hess = List[Float64]()
    var no_weights = List[Float64]()
    fill_grad_hess(raw, target, SQUARED_ERROR, no_weights, 0.9, grad, hess)

    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        fill_grad_hess(raw, target, SQUARED_ERROR, no_weights, 0.9, grad, hess)
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        fill_grad_hess(raw, target, SQUARED_ERROR, no_weights, 0.9, grad, hess)
    t2 = perf_counter_ns()
    _report(
        "grad_hess",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    # --- histograms --------------------------------------------------------
    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var h = build_histogram(data, grad, hess)
        _ = h.n_bins
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var h2 = build_histogram(data, grad, hess)
        _ = h2.n_bins
    t2 = perf_counter_ns()
    _report(
        "hist_full",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var half = List[Int]()
    for r in range(0, n_rows, 2):
        half.append(r)
    var tenth = List[Int]()
    for r in range(0, n_rows, 10):
        tenth.append(r)

    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var h = build_histogram_subset(data, grad, hess, half)
        _ = h.n_bins
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var h2 = build_histogram_subset(data, grad, hess, half)
        _ = h2.n_bins
    t2 = perf_counter_ns()
    _report(
        "hist_subset50",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var h = build_histogram_subset(data, grad, hess, tenth)
        _ = h.n_bins
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var h2 = build_histogram_subset(data, grad, hess, tenth)
        _ = h2.n_bins
    t2 = perf_counter_ns()
    _report(
        "hist_subset10",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var parent = build_histogram(data, grad, hess)
    var child = build_histogram_subset(data, grad, hess, half)

    # Subtraction is a single SIMD sweep, so one rep is far too short to
    # time; scale the rep count up rather than reporting quantization noise.
    var sub_reps = reps * 200
    t0 = perf_counter_ns()
    _serial()
    for _ in range(sub_reps):
        var s = subtract_histogram(parent, child)
        _ = s.n_bins
    t1 = perf_counter_ns()
    _auto()
    for _ in range(sub_reps):
        var s2 = subtract_histogram(parent, child)
        _ = s2.n_bins
    t2 = perf_counter_ns()
    _report(
        "hist_subtract",
        Float64(t1 - t0) / 1e9 / Float64(sub_reps),
        Float64(t2 - t1) / 1e9 / Float64(sub_reps),
    )

    # --- split scanning ----------------------------------------------------
    var scan_reps = reps * 50
    t0 = perf_counter_ns()
    _serial()
    for _ in range(scan_reps):
        var s = find_best_split(parent)
        _ = s.feature
    t1 = perf_counter_ns()
    _auto()
    for _ in range(scan_reps):
        var s2 = find_best_split(parent)
        _ = s2.feature
    t2 = perf_counter_ns()
    _report(
        "split_scan",
        Float64(t1 - t0) / 1e9 / Float64(scan_reps),
        Float64(t2 - t1) / 1e9 / Float64(scan_reps),
    )

    # --- row partitioning --------------------------------------------------
    var best = find_best_split(parent)
    if not best.found:
        raise Error("profile needs a splittable root histogram")
    var all_rows = List[Int](capacity=n_rows)
    for r in range(n_rows):
        all_rows.append(r)
    var missing = data.missing_bin[best.feature]

    var part_reps = reps * 10
    t0 = perf_counter_ns()
    _serial()
    for _ in range(part_reps):
        var p = partition_rows(data, all_rows, best, missing)
        _ = len(p.left)
    t1 = perf_counter_ns()
    _auto()
    for _ in range(part_reps):
        var p2 = partition_rows(data, all_rows, best, missing)
        _ = len(p2.left)
    t2 = perf_counter_ns()
    _report(
        "partition",
        Float64(t1 - t0) / 1e9 / Float64(part_reps),
        Float64(t2 - t1) / 1e9 / Float64(part_reps),
    )

    # --- whole tree and prediction ----------------------------------------
    var params = TreeParams.default()
    t0 = perf_counter_ns()
    _serial()
    for _ in range(reps):
        var t = grow_tree(data, grad, hess, params)
        _ = t.n_leaves
    t1 = perf_counter_ns()
    _auto()
    for _ in range(reps):
        var t2b = grow_tree(data, grad, hess, params)
        _ = t2b.n_leaves
    t2 = perf_counter_ns()
    _report(
        "grow_tree",
        Float64(t1 - t0) / 1e9 / Float64(reps),
        Float64(t2 - t1) / 1e9 / Float64(reps),
    )

    var tree = grow_tree(data, grad, hess, params)
    var pred_reps = reps * 5
    t0 = perf_counter_ns()
    _serial()
    for _ in range(pred_reps):
        var acc = 0.0
        for r in range(n_rows):
            acc += tree.predict_row(data, r)
        _ = acc
    t1 = perf_counter_ns()
    _auto()
    for _ in range(pred_reps):
        var acc2 = 0.0
        for r in range(n_rows):
            acc2 += tree.predict_row(data, r)
        _ = acc2
    t2 = perf_counter_ns()
    _report(
        "predict",
        Float64(t1 - t0) / 1e9 / Float64(pred_reps),
        Float64(t2 - t1) / 1e9 / Float64(pred_reps),
    )

    _ = setenv("MOJOTREES_NUM_WORKERS", "")
