"""Per-device GPU validation report: device facts, launch geometry, phases.

One driver that produces everything a cross-vendor validation record needs
from the host side, so the same command yields comparable output on Metal,
CUDA, and HIP. It measures nothing vendor-specific and takes no vendor
branch; the whole point is that one source runs everywhere and the numbers
are read side by side afterwards.

Where `bench_histogram.mojo` times the histogram kernel in isolation, this
driver covers a whole training session and reports the device and geometry
facts a validation record has to carry with its numbers.

What it prints, per run

- device identity (`name`, `api`, `arch_name`, `compute_capability`) and the
  capability attributes the tiling policy reads. Attributes a backend does
  not implement print as `unavailable` rather than failing the run, and which
  ones those are is itself a result worth recording.

What it prints, per dataset shape

- the launch geometry `gpu_tiling.mojo` resolved for that shape, and the
  occupancy, shared-memory, and atomic-traffic quantities derived from it,
  each one a prediction a profiler trace should confirm or refute (see
  docs/GPU_VALIDATION.md for which counter tests which line)
- separated wall-clock phases, using the builder's own phase methods so
  transfers, kernels, and host conversion are timed apart rather than
  estimated:

  | phase        | what it covers                                        |
  |--------------|-------------------------------------------------------|
  | `binning`    | host-side quantile binning, no device involvement      |
  | `setup`      | context, capability query, allocation, binned-matrix H2D|
  | `stage`      | Float64 to Float32 gradient conversion, host only      |
  | `grad_h2d`   | staged gradients and hessians copied to the device     |
  | `hist_kernel`| `enqueue_leaf` plus synchronize: kernels, no transfer  |
  | `hist_d2h`   | fixed-point histogram copied back to the host          |
  | `hist_decode`| fixed-point to Float64 conversion, host only           |
  | `partition`  | one `apply_split` plus synchronize: kernel, no transfer|
  | `gpu_train`  | complete `train_gpu`, uninstrumented                   |
  | `cpu_train`  | complete `train`, same shape and parameters            |

Every phase above is measured directly. Nothing is derived by subtraction.
The one thing the harness cannot see is where time goes *inside* a kernel,
which is what the vendor profiler is for.

Both trainers report training MSE, so throughput is never read apart from fit
quality. CPU-side threading honors MOJOTREES_NUM_WORKERS and
MOJOTREES_PARALLEL_MIN_OPS (see parallel.mojo); pin them for reproducible
comparisons. MOJOTREES_GPU_HIST_STRATEGY, MOJOTREES_GPU_ROW_TILE, and
MOJOTREES_GPU_BLOCK_THREADS override the tiling policy, so a sweep needs no
rebuild.

Usage:
    mojo run -I src bench/bench_gpu_validation.mojo [rounds] [rows features]...

With no shape arguments a built-in sweep runs: shapes that are wide, tall,
and square, plus one small enough to be launch-overhead bound. Extra
arguments are read as (rows, features) pairs and replace the sweep.
"""

from std.sys import argv, has_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceAttribute, DeviceContext

from mojotrees.binning import fit_bins
from mojotrees.boosting import (
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    train,
)
from mojotrees.binning import BinnedMatrix
from mojotrees.gpu_tiling import STRATEGY_TILED, strategy_name
from mojotrees.binning import MAX_BINS
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.train_gpu import train_gpu
from mojotrees.tree import TreeParams

# Bins requested from the binner. 255 is the library default and the value
# every cross-device record should use unless it is explicitly sweeping bins.
comptime BENCH_BINS = 255

# Boosting rounds when none is given. Low enough that a four-shape sweep
# finishes on a laptop-class device, high enough that per-round costs
# dominate one-time setup.
comptime DEFAULT_ROUNDS = 20

# Bytes of shared memory the histogram kernels reserve per threadgroup: three
# MAX_BINS-long Int32 planes, sized by the compile-time maximum rather than
# the dataset's bin count.
comptime SHARED_BYTES_RESERVED = 3 * MAX_BINS * 4


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
    print(
        _attribute_line(ctx, "max_grid_dim_x", DeviceAttribute.MAX_GRID_DIM_X)
    )
    print(
        _attribute_line(ctx, "max_grid_dim_y", DeviceAttribute.MAX_GRID_DIM_Y)
    )
    print(_attribute_line(ctx, "clock_rate_khz", DeviceAttribute.CLOCK_RATE))


def _report_geometry(builder: GpuHistogramBuilder) raises:
    """Print the resolved launch geometry and the occupancy, shared-memory,
    and atomic-traffic quantities that follow from it.

    These are read off the builder, not recomputed, so they are the geometry
    the kernels will actually launch with. They are still host-side
    predictions about the device's behavior; docs/GPU_VALIDATION.md pairs
    each with the profiler counter that confirms it on real hardware."""
    var tiling = builder.tiling.copy()
    var caps = builder.caps.copy()
    var blocks = builder.n_features * tiling.n_tiles

    print("  strategy:", strategy_name(tiling.strategy))
    print("  grid_dim:", builder.n_features, "x", tiling.n_tiles)
    print("  block_dim:", tiling.block_threads)
    print("  blocks_per_hist_launch:", blocks)
    print("  rows_per_tile:", tiling.rows_per_tile)
    print("  partial_cells:", tiling.partial_cells)
    print("  caps_multiprocessor_count:", caps.sm_count)
    print("  caps_max_threads_per_block:", caps.max_threads_per_block)
    print(
        "  caps_max_shared_memory_per_block:",
        caps.max_shared_memory_per_block,
    )

    # The kernels reserve three MAX_BINS-long planes regardless of the
    # dataset's bin count, so a 64-bin dataset reserves exactly as much
    # shared memory as a 256-bin one. That gap is the first thing to check
    # against the profiler's static shared-memory figure.
    print("  shared_bytes_reserved_per_block:", SHARED_BYTES_RESERVED)
    print("  shared_bytes_used_per_block:", 3 * builder.n_bins * 4)
    if caps.max_shared_memory_per_block > 0:
        print(
            "  shared_memory_resident_block_limit:",
            caps.max_shared_memory_per_block // SHARED_BYTES_RESERVED,
        )
    if caps.sm_count > 0:
        print(
            "  blocks_per_sm_at_launch:",
            Float64(blocks) / Float64(caps.sm_count),
        )

    # Atomic traffic. Every in-range row issues three shared-memory atomics.
    # The tiled strategy writes partials to their own slots and reduces them
    # in a second kernel, so it issues no global atomics at all; the atomic
    # strategy folds every block's partial into the output.
    var rows_in_tile = tiling.rows_per_tile
    if builder.n_rows < rows_in_tile:
        rows_in_tile = builder.n_rows
    print("  shared_atomics_per_block_upper:", 3 * rows_in_tile)
    var global_atomics = 0
    if tiling.strategy != STRATEGY_TILED:
        global_atomics = 3 * blocks * builder.n_bins
    print("  global_atomics_per_launch_upper:", global_atomics)
    print(
        "  threads_per_bin_contention_proxy:",
        Float64(tiling.block_threads) / Float64(builder.n_bins),
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

    # Phase: setup. Context creation, the capability query, the tiling
    # decision, every device allocation, and the one-time binned-matrix
    # upload, all inside the constructor.
    var t2 = perf_counter_ns()
    var builder = GpuHistogramBuilder(data)
    builder.synchronize()
    var t3 = perf_counter_ns()
    print("  setup_s:", _seconds(t2, t3))

    _report_geometry(builder)

    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(target[r] - 1.0)
        hess.append(1.0)

    # Phase: staging, then the transfer. Split because the first is a host
    # Float64-to-Float32 pass and the second is the actual copy; on a
    # discrete GPU they scale with different things.
    var t4 = perf_counter_ns()
    builder.stage_gradients(grad, hess)
    var t5 = perf_counter_ns()
    builder.upload_staged()
    builder.synchronize()
    var t6 = perf_counter_ns()
    print("  stage_s:", _seconds(t4, t5))
    print("  grad_h2d_s:", _seconds(t5, t6))

    builder.begin_tree()
    builder.synchronize()

    # Phase: the root histogram, split three ways. Kernels, transfer back,
    # and host-side decode are each measured on their own.
    var t7 = perf_counter_ns()
    builder.enqueue_leaf(0)
    builder.synchronize()
    var t8 = perf_counter_ns()
    builder.download_raw()
    var t9 = perf_counter_ns()
    var root = builder.histogram_from_host()
    var t10 = perf_counter_ns()
    print("  hist_kernel_root_s:", _seconds(t7, t8))
    print("  hist_d2h_s:", _seconds(t8, t9))
    print("  hist_decode_s:", _seconds(t9, t10))
    print("  hist_cells:", root.n_features * root.n_bins)

    # Phase: one partition kernel, synchronized. No transfer at all, so it is
    # the cleanest kernel time the harness can produce.
    var t11 = perf_counter_ns()
    builder.apply_split(0, data.n_bins // 2, 0, 1, 2)
    builder.synchronize()
    var t12 = perf_counter_ns()
    print("  partition_kernel_s:", _seconds(t11, t12))

    # A child histogram after a real split, where the leaf filter rejects
    # roughly half the rows the way it does mid-tree.
    var t13 = perf_counter_ns()
    builder.enqueue_leaf(1)
    builder.synchronize()
    var t14 = perf_counter_ns()
    print("  hist_kernel_child_s:", _seconds(t13, t14))
    builder.download_raw()
    var child = builder.histogram_from_host()
    var child_rows = 0
    for b in range(child.n_bins):
        child_rows += child.count_at(b)
    print("  hist_child_rows:", child_rows)

    var params = BoosterParams(rounds, 0.1, TreeParams.default())

    # Phase: complete training, both backends, uninstrumented. No
    # synchronization the library would not do on its own.
    var t15 = perf_counter_ns()
    var gpu = train_gpu(data, target, SQUARED_ERROR, params)
    var t16 = perf_counter_ns()
    print("  gpu_train_s:", _seconds(t15, t16))
    print("  gpu_n_trees:", len(gpu.trees))
    print("  gpu_train_mse:", _train_mse(gpu, data, target))

    var t17 = perf_counter_ns()
    var cpu = train(data, target, SQUARED_ERROR, params)
    var t18 = perf_counter_ns()
    print("  cpu_train_s:", _seconds(t17, t18))
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
            raise Error(
                "dataset shapes must be given as (rows, features) pairs"
            )

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
