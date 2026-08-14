"""Per-device GPU validation report: device facts, launch geometry, phases.

One driver that produces everything a cross-vendor validation record needs
from the host side, so the same command yields comparable output on Metal,
CUDA, and HIP. It measures nothing vendor-specific and takes no vendor
branch; the whole point is that one source runs everywhere and the numbers
are read side by side afterwards.

What it prints, per run

- device identity (`name`, `api`, `arch_name`, `compute_capability`) and the
  capability attributes the launch geometry depends on. Attributes a backend
  does not implement print as `unavailable` rather than failing the run.

What it prints, per dataset shape

- the launch geometry the histogram kernel will actually use, and the derived
  occupancy, shared-memory, and atomic-traffic quantities a profiler trace
  should be checked against (see docs/GPU_VALIDATION.md)
- wall-clock phases, measured through the public `GpuHistogramBuilder` API:

  | phase          | what it covers                                       |
  |----------------|------------------------------------------------------|
  | `binning`      | host-side quantile binning (no device involvement)    |
  | `setup`        | context creation, device allocation, binned-matrix H2D|
  | `grad_h2d`     | one round of gradient/hessian upload                  |
  | `partition`    | one `apply_split` kernel, synchronized, no transfer   |
  | `node_hist`    | one `build_leaf`: memset + kernel + histogram D2H     |
  | `d2h_probe`    | a same-size device-to-host map, for scale on the above|
  | `gpu_train`    | complete `train_gpu`, uninstrumented                  |
  | `cpu_train`    | complete `train`, for the same shape and parameters   |

`node_hist` fuses kernel and download because the builder's public API does;
`d2h_probe` measures an identical-size map on the same context so the
transfer share of `node_hist` has a measured scale next to it. This is a
host-visible wall-clock decomposition and nothing more. The authoritative
kernel-versus-transfer split comes from the vendor profiler (Nsight Compute,
rocprof), and docs/GPU_VALIDATION.md gives those commands.

Both trainers report training loss, so throughput is never read apart from
fit quality. CPU-side threading honors MOJOBOOST_NUM_WORKERS and
MOJOBOOST_PARALLEL_MIN_OPS (see parallel.mojo); pin them for reproducible
comparisons.

Usage:
    mojo run -I src bench/bench_gpu_validation.mojo [rounds] [rows features]...

With no shape arguments a built-in sweep runs: shapes that are wide, tall,
and square, plus one small enough to be launch-overhead bound. Extra
arguments are read as (rows, features) pairs and replace the sweep.
"""

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceAttribute, DeviceContext

from mojoboost.binning import fit_bins
from mojoboost.boosting import (
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train,
)
from mojoboost.binning import BinnedMatrix
from mojoboost.histogram_gpu import (
    BLOCK_THREADS,
    MAX_BINS,
    ROWS_PER_CHUNK,
    GpuHistogramBuilder,
)
from mojoboost.train_gpu import train_gpu
from mojoboost.tree import TreeParams

# Bins requested from the binner. 255 is the library default and the value
# every cross-device record should use unless it is explicitly sweeping bins.
comptime BENCH_BINS = 255

# Boosting rounds when none is given. Low enough that a four-shape sweep
# finishes on a laptop-class device, high enough that per-round costs
# dominate one-time setup.
comptime DEFAULT_ROUNDS = 20


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _seconds(start: Int, end: Int) -> Float64:
    return Float64(end - start) / 1e9


def _train_mse(
    booster: Booster, data: BinnedMatrix, target: List[Float64]
) -> Float64:
    var loss = 0.0
    for r in range(data.n_rows):
        var d = booster.predict_row(data, r) - target[r]
        loss += d * d
    return loss / Float64(data.n_rows)


def _attribute_line(
    ctx: DeviceContext, label: String, attr: DeviceAttribute
) -> String:
    """One capability attribute, or `unavailable` when this backend does not
    implement it. Metal rejects several that CUDA and HIP answer, and a
    missing attribute is a fact worth recording, not a failure."""
    try:
        return label + ": " + String(ctx.get_attribute(attr))
    except:
        return label + ": unavailable"


def _report_device(ctx: DeviceContext) raises:
    print("== device ==")
    print("name:", ctx.name())
    print("api:", ctx.api())
    print("arch_name:", ctx.arch_name())
    print("compute_capability:", ctx.compute_capability())
    print(
        _attribute_line(
            ctx, "multiprocessor_count", DeviceAttribute.MULTIPROCESSOR_COUNT
        )
    )
    print(_attribute_line(ctx, "warp_size", DeviceAttribute.WARP_SIZE))
    print(
        _attribute_line(
            ctx,
            "max_threads_per_block",
            DeviceAttribute.MAX_THREADS_PER_BLOCK,
        )
    )
    print(
        _attribute_line(
            ctx,
            "max_threads_per_multiprocessor",
            DeviceAttribute.MAX_THREADS_PER_MULTIPROCESSOR,
        )
    )
    print(
        _attribute_line(
            ctx,
            "max_blocks_per_multiprocessor",
            DeviceAttribute.MAX_BLOCKS_PER_MULTIPROCESSOR,
        )
    )
    print(
        _attribute_line(
            ctx,
            "max_shared_memory_per_block",
            DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK,
        )
    )
    print(
        _attribute_line(
            ctx,
            "max_registers_per_block",
            DeviceAttribute.MAX_REGISTERS_PER_BLOCK,
        )
    )
    print(_attribute_line(ctx, "max_grid_dim_x", DeviceAttribute.MAX_GRID_DIM_X))
    print(_attribute_line(ctx, "max_grid_dim_y", DeviceAttribute.MAX_GRID_DIM_Y))
    print(_attribute_line(ctx, "clock_rate_khz", DeviceAttribute.CLOCK_RATE))


def _report_geometry(
    ctx: DeviceContext, n_rows: Int, n_features: Int, n_bins: Int
) raises:
    """Print the histogram launch geometry and the occupancy, shared-memory,
    and atomic-traffic quantities derived from it. These are predictions from
    host-side arithmetic; docs/GPU_VALIDATION.md pairs each with the profiler
    counter that confirms or refutes it on real hardware."""
    var n_chunks = (n_rows + ROWS_PER_CHUNK - 1) // ROWS_PER_CHUNK
    if n_chunks < 1:
        n_chunks = 1
    var blocks = n_features * n_chunks

    # The kernel's shared arrays are MAX_BINS long regardless of the dataset's
    # bin count, so a 32-bin dataset reserves exactly as much shared memory as
    # a 256-bin one. That gap is the first thing to check against the
    # profiler's static shared-memory figure.
    var shared_reserved = 3 * MAX_BINS * 4
    var shared_used = 3 * n_bins * 4

    print("  grid_dim:", n_features, "x", n_chunks)
    print("  block_dim:", BLOCK_THREADS)
    print("  blocks_per_hist_launch:", blocks)
    print("  rows_per_chunk:", ROWS_PER_CHUNK)
    print("  shared_bytes_reserved_per_block:", shared_reserved)
    print("  shared_bytes_used_per_block:", shared_used)

    try:
        var sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        if sm > 0:
            print(
                "  blocks_per_sm_at_launch:",
                Float64(blocks) / Float64(sm),
            )
    except:
        print("  blocks_per_sm_at_launch: unavailable")

    try:
        var smem = ctx.get_attribute(
            DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK
        )
        if smem > 0:
            print(
                "  shared_memory_resident_block_limit:",
                smem // shared_reserved,
            )
    except:
        print("  shared_memory_resident_block_limit: unavailable")

    # Atomic traffic. Every in-range row issues three shared-memory atomics;
    # every block flushes at most n_bins bins as three global atomics each.
    # `threads_per_bin` is the contention proxy: with more threads than bins,
    # concurrent updates to one bin serialize, and that ratio is what a
    # bin-count or privatization change moves.
    var rows_in_chunk = ROWS_PER_CHUNK
    if n_rows < rows_in_chunk:
        rows_in_chunk = n_rows
    print("  shared_atomics_per_block_upper:", 3 * rows_in_chunk)
    print("  global_atomics_per_launch_upper:", 3 * blocks * n_bins)
    print(
        "  threads_per_bin_contention_proxy:",
        Float64(BLOCK_THREADS) / Float64(n_bins),
    )


def _make_dataset(
    n_rows: Int, n_features: Int, mut target: List[Float64]
) -> List[Float64]:
    """The same synthetic stream as bench_train.mojo and bench_train_gpu.mojo,
    so numbers from this driver sit next to those without a caveat."""
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))
    var noise_base = UInt64(n_rows * n_features)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        var signal = 5.0 * x0 + 4.0 * x1 * x2 + 3.0 * (x3 - 0.5) * (x3 - 0.5)
        target.append(signal + 0.1 * (_uniform(noise_base + UInt64(r)) - 0.5))
    return features^


def _time_d2h_probe(mut builder: GpuHistogramBuilder) raises -> Float64:
    """Map a histogram-sized device buffer to the host and read every entry.

    Deliberately the same size and the same access shape as `build_leaf`'s
    download, on the same context, so it gives the transfer term in
    `node_hist` a measured scale. It is a comparable quantity, not a
    subtraction: the profiler is what separates kernel from copy exactly."""
    var hist_size = builder.n_features * builder.n_bins
    var probe = builder.ctx.enqueue_create_buffer[DType.int32](3 * hist_size)
    builder.ctx.enqueue_memset(probe, 0)
    builder.ctx.synchronize()

    var t0 = perf_counter_ns()
    var checksum = 0
    with probe.map_to_host() as host:
        var src = host.unsafe_ptr()
        for i in range(3 * hist_size):
            checksum += Int(src.unsafe_load(i))
    var t1 = perf_counter_ns()
    # Keep the read from being optimized away without printing noise.
    if checksum != 0:
        print("  d2h_probe_checksum_unexpected:", checksum)
    return _seconds(t0, t1)


def _run_shape(n_rows: Int, n_features: Int, rounds: Int) raises:
    print("")
    print("== shape:", n_rows, "rows x", n_features, "features ==")

    var target = List[Float64](capacity=n_rows)
    var features = _make_dataset(n_rows, n_features, target)

    var t0 = perf_counter_ns()
    var mapper = fit_bins(features, n_rows, n_features, BENCH_BINS)
    var data = mapper.transform(features, n_rows)
    var t1 = perf_counter_ns()
    print("  n_bins:", data.n_bins)
    print("  rounds:", rounds)
    print("  binning_s:", _seconds(t0, t1))

    var ctx = DeviceContext()
    _report_geometry(ctx, n_rows, n_features, data.n_bins)

    # Phase: setup. Context creation, every device allocation, and the
    # one-time binned-matrix upload, all inside the constructor.
    var t2 = perf_counter_ns()
    var builder = GpuHistogramBuilder(data)
    builder.ctx.synchronize()
    var t3 = perf_counter_ns()
    print("  setup_s:", _seconds(t2, t3))

    # Phase: gradient upload. One round's worth, the recurring H2D cost.
    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(target[r] - 1.0)
        hess.append(1.0)
    var t4 = perf_counter_ns()
    builder.upload_gradients(grad, hess)
    builder.ctx.synchronize()
    var t5 = perf_counter_ns()
    print("  grad_h2d_s:", _seconds(t4, t5))

    builder.begin_tree()
    builder.ctx.synchronize()

    # Phase: one root histogram. Memset, kernel, and download fused, which is
    # what the public API exposes and what a boosting round actually pays.
    var t6 = perf_counter_ns()
    var root = builder.build_leaf(0)
    var t7 = perf_counter_ns()
    print("  node_hist_root_s:", _seconds(t6, t7))
    print("  node_hist_root_bins:", root.n_features * root.n_bins)

    # Phase: one partition kernel, synchronized. No transfer in this one, so
    # it is the cleanest host-visible kernel time the harness can produce.
    var t8 = perf_counter_ns()
    builder.apply_split(0, data.n_bins // 2, 0, 1, 2)
    builder.ctx.synchronize()
    var t9 = perf_counter_ns()
    print("  partition_kernel_s:", _seconds(t8, t9))

    # Phase: one child histogram, after a real split, so the leaf filter
    # rejects roughly half the rows the way it does mid-tree.
    var t10 = perf_counter_ns()
    var child = builder.build_leaf(1)
    var t11 = perf_counter_ns()
    print("  node_hist_child_s:", _seconds(t10, t11))
    var child_rows = 0
    for b in range(child.n_bins):
        child_rows += child.count[b]
    print("  node_hist_child_rows:", child_rows)

    print("  d2h_probe_s:", _time_d2h_probe(builder))

    var params = BoosterParams(rounds, 0.1, TreeParams.default())

    # Phase: complete training, both backends, uninstrumented. No
    # synchronization the library would not do on its own.
    var t12 = perf_counter_ns()
    var gpu = train_gpu(data, target, SQUARED_ERROR, params)
    var t13 = perf_counter_ns()
    print("  gpu_train_s:", _seconds(t12, t13))
    print("  gpu_n_trees:", len(gpu.trees))
    print("  gpu_train_mse:", _train_mse(gpu, data, target))

    var t14 = perf_counter_ns()
    var cpu = train(data, target, SQUARED_ERROR, params)
    var t15 = perf_counter_ns()
    print("  cpu_train_s:", _seconds(t14, t15))
    print("  cpu_n_trees:", len(cpu.trees))
    print("  cpu_train_mse:", _train_mse(cpu, data, target))


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; GPU validation report skipped")
        print("build a device-enabled toolchain on the target machine first")
    else:
        var args = argv()
        var rounds = DEFAULT_ROUNDS
        if len(args) > 1:
            rounds = Int(String(args[1]))
        if rounds < 1:
            raise Error("rounds must be positive")

        var shapes = List[Int]()
        var i = 2
        while i + 1 < len(args):
            shapes.append(Int(String(args[i])))
            shapes.append(Int(String(args[i + 1])))
            i += 2
        if i != len(args):
            raise Error("dataset shapes must be given as (rows, features) pairs")

        if len(shapes) == 0:
            # Small (launch-overhead bound), square, wide, and tall. Four
            # shapes exercise different ratios of grid.x to grid.y, which is
            # what separates a device that fills from one that starves.
            shapes = [
                10_000, 20,
                100_000, 100,
                50_000, 400,
                1_000_000, 20,
            ]

        var ctx = DeviceContext()
        _report_device(ctx)

        var s = 0
        while s + 1 < len(shapes):
            var n_rows = shapes[s]
            var n_features = shapes[s + 1]
            if n_rows < 1 or n_features < 4:
                raise Error("need at least 1 row and 4 features")
            _run_shape(n_rows, n_features, rounds)
            s += 2
