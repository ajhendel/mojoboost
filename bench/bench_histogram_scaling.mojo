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
- `kernel_root_s`: enqueue the histogram kernels over every row and wait.
  No transfers. This is the number the tiling work is trying to move.
- `kernel_leaf_s`: the same, filtered to one child of a root split, which
  is the shape every node below the root has.
- `download_s`: the fixed-point histogram copied device to host.
- `back_convert_s`: those integers scaled into the Float64 `Histogram`.

Two things make the numbers trustworthy on a machine that is not idle. The
strategies are interleaved: within one repetition each phase is timed for
`atomic` and then for `tiled`, so a background load that inflates one
inflates the other. And each phase reports `best` alongside `mean`, since
the fastest observed repetition is the closest estimate of the uncontended
cost while the mean carries whatever else the machine was doing. Compare
`best` to `best`. If `mean` and `best` are far apart, the machine was busy
and the run should be repeated.

Kernel and transfer phases are timed after a warm-up build, so shader or
PTX compilation is never inside a reported number. Correctness is checked in
the same run: the two strategies must produce bit-identical histograms, and
the benchmark prints whether they did.

Usage:
  mojo run -I src bench/bench_histogram_scaling.mojo [reps] [rows features]...

With no shape arguments it runs a small, a medium, and a large shape. Record
the accelerator, the Mojo/MAX version, and these dimensions with any number
quoted from this driver, and never present one GPU family's numbers as
representative of another.
"""

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    strategy_name,
)
from mojotrees.histogram import Histogram, build_histogram
from mojotrees.histogram_gpu import GpuHistogramBuilder


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


struct _Timer(Copyable, Movable):
    """Mean and best of repeated timings of one phase."""

    var total_ns: Float64
    var best_ns: Float64
    var reps: Int

    def __init__(out self):
        self.total_ns = 0.0
        self.best_ns = 0.0
        self.reps = 0

    def add(mut self, elapsed_ns: Int):
        var ns = Float64(elapsed_ns)
        if self.reps == 0 or ns < self.best_ns:
            self.best_ns = ns
        self.total_ns += ns
        self.reps += 1

    def mean_s(self) -> Float64:
        if self.reps == 0:
            return 0.0
        return self.total_ns / Float64(self.reps) / 1e9

    def best_s(self) -> Float64:
        return self.best_ns / 1e9


def _report(label: String, atomic: _Timer, tiled: _Timer):
    print(
        "   ",
        label,
        "atomic mean",
        atomic.mean_s(),
        "best",
        atomic.best_s(),
        "| tiled mean",
        tiled.mean_s(),
        "best",
        tiled.best_s(),
    )


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


def _describe(builder: GpuHistogramBuilder, setup_s: Float64):
    print("  ", strategy_name(builder.strategy()), "geometry:")
    print("      block_threads:", builder.tiling.block_threads)
    print("      n_tiles:", builder.tiling.n_tiles)
    print("      rows_per_tile:", builder.tiling.rows_per_tile)
    print("      threadgroups:", len(builder.active) * builder.tiling.n_tiles)
    print(
        "      partial_mib:",
        Float64(3 * builder.tiling.partial_cells * 4) / 1048576.0,
    )
    print("      setup_s:", setup_s)


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
    print("  binning_s:", Float64(t1 - t0) / 1e9)

    # CPU reference: the multicore SIMD builder, same full-dataset build.
    var cpu_timer = _Timer()
    var cpu = build_histogram(data, grad, hess)
    for _ in range(reps):
        var c0 = perf_counter_ns()
        cpu = build_histogram(data, grad, hess)
        cpu_timer.add(perf_counter_ns() - c0)
    print(
        "  cpu_build_s: mean",
        cpu_timer.mean_s(),
        "best",
        cpu_timer.best_s(),
    )

    var s0 = perf_counter_ns()
    var atomic = GpuHistogramBuilder(data, STRATEGY_ATOMIC)
    var s1 = perf_counter_ns()
    var tiled = GpuHistogramBuilder(data, STRATEGY_TILED)
    var s2 = perf_counter_ns()
    _describe(atomic, Float64(s1 - s0) / 1e9)
    _describe(tiled, Float64(s2 - s1) / 1e9)

    # Warm-up: the first launch of each kernel pays its compilation, which
    # belongs in no steady-state number below.
    var warm_atomic = atomic.build(grad, hess)
    var warm_tiled = tiled.build(grad, hess)
    print("  strategies_bit_identical:", _identical(warm_atomic, warm_tiled))

    var a_convert = _Timer()
    var t_convert = _Timer()
    var a_upload = _Timer()
    var t_upload = _Timer()
    var a_root = _Timer()
    var t_root = _Timer()
    var a_download = _Timer()
    var t_download = _Timer()
    var a_back = _Timer()
    var t_back = _Timer()

    atomic.begin_tree()
    tiled.begin_tree()
    atomic.synchronize()
    tiled.synchronize()

    # One repetition times every phase for both strategies back to back, so
    # a background load lands on both.
    for _ in range(reps):
        var p0 = perf_counter_ns()
        atomic.stage_gradients(grad, hess)
        var p1 = perf_counter_ns()
        tiled.stage_gradients(grad, hess)
        var p2 = perf_counter_ns()
        a_convert.add(p1 - p0)
        t_convert.add(p2 - p1)

        var p3 = perf_counter_ns()
        atomic.upload_staged()
        atomic.synchronize()
        var p4 = perf_counter_ns()
        tiled.upload_staged()
        tiled.synchronize()
        var p5 = perf_counter_ns()
        a_upload.add(p4 - p3)
        t_upload.add(p5 - p4)

        var p6 = perf_counter_ns()
        atomic.enqueue_leaf(0)
        atomic.synchronize()
        var p7 = perf_counter_ns()
        tiled.enqueue_leaf(0)
        tiled.synchronize()
        var p8 = perf_counter_ns()
        a_root.add(p7 - p6)
        t_root.add(p8 - p7)

        var p9 = perf_counter_ns()
        atomic.download_raw()
        var p10 = perf_counter_ns()
        tiled.download_raw()
        var p11 = perf_counter_ns()
        a_download.add(p10 - p9)
        t_download.add(p11 - p10)

        var p12 = perf_counter_ns()
        _ = atomic.histogram_from_host()
        var p13 = perf_counter_ns()
        _ = tiled.histogram_from_host()
        var p14 = perf_counter_ns()
        a_back.add(p13 - p12)
        t_back.add(p14 - p13)

    # A node below the root: the same full-row scan plus the leaf filter.
    var mid = data.n_bins // 2 - 1
    atomic.apply_split(0, mid, 0, 1, 2)
    tiled.apply_split(0, mid, 0, 1, 2)
    atomic.synchronize()
    tiled.synchronize()

    var a_leaf = _Timer()
    var t_leaf = _Timer()
    for _ in range(reps):
        var q0 = perf_counter_ns()
        atomic.enqueue_leaf(1)
        atomic.synchronize()
        var q1 = perf_counter_ns()
        tiled.enqueue_leaf(1)
        tiled.synchronize()
        var q2 = perf_counter_ns()
        a_leaf.add(q1 - q0)
        t_leaf.add(q2 - q1)

    _report("convert_s     ", a_convert, t_convert)
    _report("upload_s      ", a_upload, t_upload)
    _report("kernel_root_s ", a_root, t_root)
    _report("kernel_leaf_s ", a_leaf, t_leaf)
    _report("download_s    ", a_download, t_download)
    _report("back_convert_s", a_back, t_back)

    # The leaf-filtered builds have to agree too, not just the warm-up.
    print(
        "  leaf_bit_identical:",
        _identical(atomic.build_leaf(1), tiled.build_leaf(1)),
    )


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
            # Small, medium, large. Small has less than one threadgroup's
            # worth of rows per feature on a wide device; large is where row
            # tiling has to do something.
            rows.append(20_000)
            features.append(20)
            rows.append(200_000)
            features.append(100)
            rows.append(1_000_000)
            features.append(50)

        print("mojotrees gpu histogram scaling bench, reps:", reps)
        for s in range(len(rows)):
            _run_shape(rows[s], features[s], reps)
