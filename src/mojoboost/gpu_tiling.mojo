"""GPU histogram tiling policy.

Chooses the launch geometry for GPU histogram accumulation: how many threads
per threadgroup, how many row tiles each feature is split into, and which of
the two accumulation strategies runs.

The two strategies share the same per-threadgroup work (a shared-memory
partial histogram over one (feature, row-tile) pair) and differ only in how
those partials combine:

- `STRATEGY_ATOMIC` folds every partial into the output with global integer
  atomics. This is the original implementation, kept as the validated
  fallback. It needs no extra memory, so it stays usable when the partial
  buffer would not fit.
- `STRATEGY_TILED` writes every partial to its own slot in a global buffer
  and reduces them in a second kernel that sums tiles in ascending order.
  No global atomics, no output memset, and the reduction order is fixed.

Both produce the identical integer result: accumulation is fixed-point Int32
throughout, and integer addition is associative, so strategy choice never
changes a histogram. Tests assert that bit-exactly.

Tiling is derived from device capabilities at runtime rather than fixed at
compile time, because the same source targets Metal, CUDA, and HIP with
multiprocessor counts spanning more than an order of magnitude. Capability
queries that a backend does not implement fall back to conservative
portable constants (Metal, for instance, rejects `WARP_SIZE`), so a missing
attribute degrades the tiling rather than failing the build.

Two environment variables override the policy for benchmarking and for tests
that must force one path, matching the `MOJOBOOST_` contract in
`parallel.mojo`:

- `MOJOBOOST_GPU_HIST_STRATEGY`: `atomic` or `tiled` forces that strategy;
  `auto`, unset, or unrecognized means the policy below decides.
- `MOJOBOOST_GPU_ROW_TILE`: rows per tile, overriding the derived value.
  Still clamped to the partial-buffer memory budget.
- `MOJOBOOST_GPU_BLOCK_THREADS`: threads per threadgroup, overriding the
  derived value. Clamped to the device maximum and rounded to a warp.
"""

from std.os import getenv
from max.gpu.host import DeviceAttribute, DeviceContext

from .parallel import _env_int


comptime STRATEGY_AUTO = 0
comptime STRATEGY_ATOMIC = 1
comptime STRATEGY_TILED = 2

# Threads per threadgroup to aim for. 256 is a warp multiple on every
# supported backend (32 on NVIDIA and Apple, 64 on AMD) and leaves room for
# several resident blocks per multiprocessor.
comptime TARGET_BLOCK_THREADS = 256

# Warp granularity used to round the block size. 64 is AMD's wavefront, and
# a multiple of 64 is also a multiple of the 32-wide warp elsewhere, so one
# constant is safe for all three backends.
comptime WARP_GRANULARITY = 64

# Resident threadgroups to aim for per multiprocessor. Enough waves that a
# late-finishing tile does not leave multiprocessors idle, small enough that
# the per-tile partial histogram stays a minor cost.
comptime TARGET_BLOCKS_PER_SM = 8

# A tile must scan enough rows to amortize writing (or atomically folding)
# its n_bins-wide partial histogram.
comptime MIN_ROWS_PER_TILE_BIN_FACTOR = 8
comptime MIN_ROWS_PER_TILE_THREAD_FACTOR = 4

# Int32 gradient + Int32 hessian + UInt32 count per (tile, feature, bin).
comptime BYTES_PER_PARTIAL_CELL = 12

# Ceiling on the global partial-histogram buffer. Beyond this the tiled
# strategy would trade more device memory than the atomic one saves.
comptime PARTIAL_BUDGET_BYTES = 64 << 20

# CUDA caps grid.y at 65535; Metal and HIP allow more, so the smallest
# portable bound applies everywhere.
comptime MAX_GRID_DIM_Y = 65535

# Used when a backend does not expose the corresponding attribute. Chosen
# low rather than typical: underestimating the multiprocessor count costs
# some parallelism, overestimating it wastes memory on unused tiles.
comptime FALLBACK_SM_COUNT = 16
comptime FALLBACK_MAX_THREADS_PER_BLOCK = 1024
comptime FALLBACK_SHARED_MEMORY_PER_BLOCK = 16384


@fieldwise_init
struct DeviceCaps(Copyable, Movable):
    """The device properties the tiling policy depends on."""

    var sm_count: Int
    var max_threads_per_block: Int
    var max_shared_memory_per_block: Int

    @staticmethod
    def fallback() -> DeviceCaps:
        """Conservative capabilities for a device that reports nothing."""
        return DeviceCaps(
            FALLBACK_SM_COUNT,
            FALLBACK_MAX_THREADS_PER_BLOCK,
            FALLBACK_SHARED_MEMORY_PER_BLOCK,
        )


@fieldwise_init
struct HistogramTiling(Copyable, Movable):
    """A resolved launch geometry for one dataset shape."""

    var strategy: Int
    var block_threads: Int
    var n_tiles: Int
    var rows_per_tile: Int
    var partial_cells: Int
    """`n_tiles * n_features * n_bins`: one (tile, feature, bin) cell, which
    carries all three Int32 planes, so the buffer holds `3 * partial_cells`
    Int32. Zero when the resolved strategy needs no partial buffer."""


def _attribute_or(
    ctx: DeviceContext, attr: DeviceAttribute, default: Int
) -> Int:
    """One device attribute, or `default` when this backend does not
    implement it or reports a nonsense value."""
    try:
        var value = ctx.get_attribute(attr)
        if value > 0:
            return value
        return default
    except:
        return default


def query_device_caps(ctx: DeviceContext) -> DeviceCaps:
    """Read the tiling-relevant capabilities of `ctx`'s device."""
    return DeviceCaps(
        _attribute_or(
            ctx, DeviceAttribute.MULTIPROCESSOR_COUNT, FALLBACK_SM_COUNT
        ),
        _attribute_or(
            ctx,
            DeviceAttribute.MAX_THREADS_PER_BLOCK,
            FALLBACK_MAX_THREADS_PER_BLOCK,
        ),
        _attribute_or(
            ctx,
            DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK,
            FALLBACK_SHARED_MEMORY_PER_BLOCK,
        ),
    )


def env_strategy() -> Int:
    """`MOJOBOOST_GPU_HIST_STRATEGY` as a strategy constant."""
    var s = getenv("MOJOBOOST_GPU_HIST_STRATEGY")
    if s == "atomic":
        return STRATEGY_ATOMIC
    if s == "tiled":
        return STRATEGY_TILED
    return STRATEGY_AUTO


def shared_bytes_for(n_bins: Int) -> Int:
    """Shared memory one threadgroup needs for its partial histogram."""
    return n_bins * BYTES_PER_PARTIAL_CELL


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def derive_block_threads(caps: DeviceCaps) -> Int:
    """Threads per threadgroup: the target, clamped to the device maximum,
    rounded down to a warp multiple, never below one warp."""
    var requested = _env_int("MOJOBOOST_GPU_BLOCK_THREADS", 0)
    var threads = requested if requested > 0 else TARGET_BLOCK_THREADS
    if threads > caps.max_threads_per_block:
        threads = caps.max_threads_per_block
    threads = (threads // WARP_GRANULARITY) * WARP_GRANULARITY
    if threads < WARP_GRANULARITY:
        threads = WARP_GRANULARITY
    return threads


def derive_tiling(
    caps: DeviceCaps,
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    requested_strategy: Int = STRATEGY_AUTO,
    max_partial_cells: Int = 0,
) raises -> HistogramTiling:
    """Resolve the launch geometry for one (rows, features, bins) shape.

    Pure host-side arithmetic so the policy is testable without a device.

    Row tiles per feature come from three bounds, whichever is tightest:
    enough threadgroups to fill the device (`TARGET_BLOCKS_PER_SM` per
    multiprocessor, spread across the features already in `grid.x`), enough
    rows per tile to amortize the partial histogram, and the memory the
    partial buffer may use.

    `max_partial_cells` caps that last bound at an already allocated buffer
    instead of at `PARTIAL_BUDGET_BYTES`. Feature subsampling re-derives the
    tiling for a narrower `grid.x` without reallocating, so it passes the
    capacity it has.
    """
    if n_rows < 1 or n_features < 1 or n_bins < 1:
        raise Error("tiling needs positive rows, features, and bins")
    if shared_bytes_for(n_bins) > caps.max_shared_memory_per_block:
        raise Error(
            "device shared memory too small for a per-threadgroup histogram"
        )

    var strategy = requested_strategy
    if strategy == STRATEGY_AUTO:
        strategy = env_strategy()

    var block_threads = derive_block_threads(caps)

    var min_rows_per_tile = MIN_ROWS_PER_TILE_BIN_FACTOR * n_bins
    var by_threads = MIN_ROWS_PER_TILE_THREAD_FACTOR * block_threads
    if by_threads > min_rows_per_tile:
        min_rows_per_tile = by_threads
    var tiles_by_rows = _ceil_div(n_rows, min_rows_per_tile)
    if tiles_by_rows < 1:
        tiles_by_rows = 1

    var target_blocks = caps.sm_count * TARGET_BLOCKS_PER_SM
    var tiles_by_occupancy = _ceil_div(target_blocks, n_features)
    if tiles_by_occupancy < 1:
        tiles_by_occupancy = 1

    var wanted = tiles_by_occupancy
    if tiles_by_rows < wanted:
        wanted = tiles_by_rows

    var hist_cells = n_features * n_bins
    var tiles_by_memory: Int
    if max_partial_cells > 0:
        tiles_by_memory = max_partial_cells // hist_cells
    else:
        tiles_by_memory = PARTIAL_BUDGET_BYTES // (
            hist_cells * BYTES_PER_PARTIAL_CELL
        )
    if tiles_by_memory < 1:
        tiles_by_memory = 1
    if tiles_by_memory > MAX_GRID_DIM_Y:
        tiles_by_memory = MAX_GRID_DIM_Y

    var n_tiles = wanted
    var forced_rows = _env_int("MOJOBOOST_GPU_ROW_TILE", 0)
    if forced_rows > 0:
        n_tiles = _ceil_div(n_rows, forced_rows)
    if n_tiles > tiles_by_memory:
        n_tiles = tiles_by_memory
    if n_tiles > MAX_GRID_DIM_Y:
        n_tiles = MAX_GRID_DIM_Y
    if n_tiles < 1:
        n_tiles = 1

    # Re-derive rows per tile from the final count so the last tile is never
    # empty, then re-derive the count so grid.y matches the rows covered.
    var rows_per_tile = _ceil_div(n_rows, n_tiles)
    n_tiles = _ceil_div(n_rows, rows_per_tile)

    if strategy == STRATEGY_AUTO:
        # More than one tile per feature is exactly the case the tiled path
        # exists for: partials that would otherwise contend on the same
        # output bins reduce without atomics. At one tile there is nothing
        # to reduce, the partial buffer is the same size as the output, and
        # the second kernel launch buys nothing, so the preserved atomic
        # path is the better default. That also covers a histogram too wide
        # for the partial budget, which is clamped to one tile above.
        if n_tiles > 1:
            strategy = STRATEGY_TILED
        else:
            strategy = STRATEGY_ATOMIC

    var partial_cells = 0
    if strategy == STRATEGY_TILED:
        partial_cells = n_tiles * hist_cells

    return HistogramTiling(
        strategy, block_threads, n_tiles, rows_per_tile, partial_cells
    )


def strategy_name(strategy: Int) -> String:
    if strategy == STRATEGY_ATOMIC:
        return String("atomic")
    if strategy == STRATEGY_TILED:
        return String("tiled")
    return String("auto")
