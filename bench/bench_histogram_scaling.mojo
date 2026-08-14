"""GPU histogram scaling benchmark: strategies, tilings, and phases.

Answers two questions the single-number microbenchmark cannot. First, how
the two accumulation strategies compare on the same data: `tiled`
(per-threadgroup partials plus a deterministic reduction kernel) against
`atomic` (the preserved fallback that folds partials in with global integer
atomics). Second, where the time actually goes, because a GPU histogram that
looks slow end to end is usually not slow in the kernel.

Every run reports these phases separately, all host-visible wall clock:

- `setup_s`: `DeviceContext`, every device allocation, and the one-time
  binned matrix upload. Paid once per dataset, never per boosting round.
- `convert_s`: Float64 gradients and hessians to the device's Float32, in
  pinned host memory. Host work, no transfer.
- `upload_s`: that staged pair copied host to device. Once per round.
- `kernel_s`: enqueue the histogram kernels and wait. No transfers. This is
  the number the tiling work is trying to move.
- `download_s`: the fixed-point histogram copied device to host.
- `back_convert_s`: those integers scaled into the Float64 `Histogram`.

Kernel and transfer phases are timed after a warm-up build, so shader or
PTX compilation is never inside a reported number. Timings are averaged over
`reps` repetitions.

The root histogram scans every row; a leaf histogram scans every row and
filters, which is the shape every node after the root has, so both are
reported. Correctness is checked in the same run: the two strategies must
produce bit-identical histograms, and the benchmark prints whether they did.

Usage:
  mojo run -I src bench/bench_histogram_scaling.mojo [reps] [rows features]...

With no shape arguments it runs a small, a medium, and a large shape. All
timings come from one process on one device, so record the accelerator, the
Mojo/MAX version, and these dimensions with any number quoted from it, and
never present one GPU family's numbers as representative of another.
"""

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojoboost.binning import bin_equal_width, BinnedMatrix
from mojoboost.gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    strategy_name,
)
from mojoboost.histogram import Histogram, build_histogram
from mojoboost.histogram_gpu import GpuHistogramBuilder


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _seconds(t0: Int, t1: Int, reps: Int) -> Float64:
    return Float64(t1 - t0) / 1e9 / Float64(reps)


def _identical(a: Histogram, b: Histogram) -> Bool:
    if a.n_features != b.n_features or a.n_bins != b.n_bins:
        return False
    for i in range(a.n_features * a.n_bins):
        if a.grad[i] != b.grad[i]:
            return False
        if a.hess[i] != b.hess[i]:
            return False
        if a.count[i] != b.count[i]:
            return False
    return True


def _run_strategy(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    strategy: Int,
    reps: Int,
) raises -> Histogram:
    """Time every phase of repeated histogram builds under one strategy and
    return the histogram it produced, for the cross-strategy equality check.
    """
    var t0 = perf_counter_ns()
    var builder = GpuHistogramBuilder(data, strategy)
    var t1 = perf_counter_ns()

    print("  strategy:", strategy_name(builder.strategy()))
    print("    block_threads:", builder.tiling.block_threads)
    print("    n_tiles:", builder.tiling.n_tiles)
    print("    rows_per_tile:", builder.tiling.rows_per_tile)
    print(
        "    partial_mib:",
        Float64(3 * builder.tiling.partial_cells * 4) / 1048576.0,
    )
    print("    setup_s:", _seconds(t0, t1, 1))

    # Warm-up: first launch pays kernel compilation, which is not part of
    # any steady-state number below.
    var warm = builder.build(grad, hess)

    var t2 = perf_counter_ns()
    for _ in range(reps):
        builder.stage_gradients(grad, hess)
    var t3 = perf_counter_ns()
    print("    convert_s:", _seconds(t2, t3, reps))

    var t4 = perf_counter_ns()
    for _ in range(reps):
        builder.upload_staged()
        builder.synchronize()
    var t5 = perf_counter_ns()
    print("    upload_s:", _seconds(t4, t5, reps))

    builder.begin_tree()
    builder.synchronize()
    var t6 = perf_counter_ns()
    for _ in range(reps):
        builder.enqueue_leaf(0)
    builder.synchronize()
    var t7 = perf_counter_ns()
    print("    kernel_root_s:", _seconds(t6, t7, reps))

    var t8 = perf_counter_ns()
    for _ in range(reps):
        builder.download_raw()
    var t9 = perf_counter_ns()
    print("    download_s:", _seconds(t8, t9, reps))

    var t10 = perf_counter_ns()
    for _ in range(reps):
        _ = builder.histogram_from_host()
    var t11 = perf_counter_ns()
    print("    back_convert_s:", _seconds(t10, t11, reps))

    # A node below the root: same full-row scan, plus the leaf filter.
    builder.apply_split(0, data.n_bins // 2 - 1, 0, 1, 2)
    builder.synchronize()
    var t12 = perf_counter_ns()
    for _ in range(reps):
        builder.enqueue_leaf(1)
    builder.synchronize()
    var t13 = perf_counter_ns()
    print("    kernel_leaf_s:", _seconds(t12, t13, reps))

    return warm^


def _run_shape(n_rows: Int, n_features: Int, reps: Int) raises:
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))

    var t0 = perf_counter_ns()
    var data = bin_equal_width(features, n_rows, n_features, 255)
    var t1 = perf_counter_ns()

    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    var base = UInt64(n_rows * n_features)
    for r in range(n_rows):
        grad.append(2.0 * _uniform(base + UInt64(r)) - 1.0)
        hess.append(_uniform(base + UInt64(n_rows + r)) + 0.01)

    print("shape:", n_rows, "rows x", n_features, "features, 255 bins")
    print("  binning_s:", _seconds(t0, t1, 1))

    # CPU reference: the multicore SIMD builder, same full-dataset build.
    var cpu = build_histogram(data, grad, hess)
    var t2 = perf_counter_ns()
    for _ in range(reps):
        cpu = build_histogram(data, grad, hess)
    var t3 = perf_counter_ns()
    print("  cpu_build_s:", _seconds(t2, t3, reps))

    var atomic = _run_strategy(data, grad, hess, STRATEGY_ATOMIC, reps)
    var tiled = _run_strategy(data, grad, hess, STRATEGY_TILED, reps)
    print("  strategies_bit_identical:", _identical(atomic, tiled))


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; GPU histogram scaling bench skipped")
    else:
        var reps = 20
        var args = argv()
        if len(args) > 1:
            reps = Int(String(args[1]))
        if reps < 1:
            raise Error("reps must be at least 1")

        var rows = List[Int]()
        var features = List[Int]()
        var i = 2
        while i + 1 < len(args):
            rows.append(Int(String(args[i])))
            features.append(Int(String(args[i + 1])))
            i += 2
        if i < len(args):
            raise Error("shapes are (rows, features) pairs")

        if len(rows) == 0:
            # Small, medium, large. Small is smaller than one threadgroup's
            # worth of work per feature; large is where row tiling has to
            # do something.
            rows.append(20_000)
            features.append(20)
            rows.append(200_000)
            features.append(100)
            rows.append(1_000_000)
            features.append(50)

        print("mojoboost gpu histogram scaling bench, reps:", reps)
        for s in range(len(rows)):
            _run_shape(rows[s], features[s], reps)
