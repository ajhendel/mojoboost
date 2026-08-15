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
that must force one path, matching the `MOJOTREES_` contract in
`parallel.mojo`:

- `MOJOTREES_GPU_HIST_STRATEGY`: `atomic` or `tiled` forces that strategy;
  `auto`, unset, or unrecognized means the policy below decides.
- `MOJOTREES_GPU_ROW_TILE`: rows per tile, overriding the derived value.
  Still clamped to the partial-buffer memory budget.
- `MOJOTREES_GPU_BLOCK_THREADS`: threads per threadgroup, overriding the
  derived value. Clamped to the device maximum and rounded to a warp.

Specialization sits above this module, not inside it. `derive_tiling` stays
the one geometry every caller gets by default, on every backend;
`apple_histogram_policy.mojo` can plan a specialized launch from reported
device properties and a node's shape, and its own default is to call
`derive_tiling` and hand back exactly what it returned. Nothing there is on
unless a caller asks for it by level or by
`MOJOTREES_GPU_HIST_SPECIALIZATION`, because nothing there has been measured.

Where this module sits in the scheduling layer
----------------------------------------------
This is the bottom of it, and the only place its geometry rules are written
down:

    gpu_tiling.mojo             the tile arithmetic, the block-width rule,
                                the partial-buffer budget, and the launch
                                count each strategy costs
        ^
    apple_histogram_policy.mojo the same arithmetic re-run against reported
                                device properties and a node's shape, plus
                                the specialization ladder and the multiclass
                                schedule
        ^                                   ^
    gpu_multiclass_batch.mojo       hybrid_leaf_scheduler.mojo
    (per-round class batching)      (per-leaf placement; charges a node the
                                     launches `launches_for_strategy` says
                                     its resolved strategy costs)

`resolve_tiling` below is the one function that decides a tile count. Both
the portable geometry (`derive_tiling`) and the shape-derived one
(`apple_histogram_policy.derive_histogram_plan` at `SPEC_LEVEL_SHAPE`) call
it with different bounds rather than restating the rule, so the two can
disagree about how many threadgroups a device wants but never about what
follows from that answer.
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

# Kernel launches each strategy costs for one node. The atomic path folds
# its partials into the output as it goes, so it is one launch; the tiled
# path writes them to a buffer and reduces them in a second kernel.
comptime LAUNCHES_ATOMIC = 1
comptime LAUNCHES_TILED = 2


def launches_for_strategy(strategy: Int) raises -> Int:
    """Kernel launches one node's histogram costs under this strategy.

    The one place that number is written down. `hybrid_leaf_scheduler.mojo`
    charges a node `launch_nanos * gpu_launches`, and deriving that count
    from the resolved strategy rather than defaulting it keeps the cost model
    and the launch it models from drifting apart.

    `STRATEGY_AUTO` has no launch count: it is a request, not a resolution,
    and a caller holding one has not yet planned a launch.
    """
    if strategy == STRATEGY_ATOMIC:
        return LAUNCHES_ATOMIC
    if strategy == STRATEGY_TILED:
        return LAUNCHES_TILED
    raise Error(
        "an unresolved strategy has no launch count; resolve it with"
        " derive_tiling first"
    )


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

    def launches(self) raises -> Int:
        """Kernel launches a node built with this geometry costs. See
        `launches_for_strategy`; carried as a method so a scheduler that
        holds a tiling never has to know the rule."""
        return launches_for_strategy(self.strategy)


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
    """`MOJOTREES_GPU_HIST_STRATEGY` as a strategy constant."""
    var s = getenv("MOJOTREES_GPU_HIST_STRATEGY")
    if s == "atomic":
        return STRATEGY_ATOMIC
    if s == "tiled":
        return STRATEGY_TILED
    return STRATEGY_AUTO


def shared_bytes_for(n_bins: Int) -> Int:
    """Shared memory one threadgroup needs for one feature slot's partial
    histogram.

    Three Int32 planes over the capacity a histogram kernel is instantiated
    at, which is the footprint one block really occupies now that the kernel
    family in `gpu_active_rows.mojo` takes a `BIN_CAP` parameter. The capacity
    rather than `n_bins` itself because that parameter walks a four-value
    ladder (`histogram_bin_capacity` below), so a 40-bin dataset occupies the
    64-bin footprint and nothing narrower; the previous rounding error, three
    `MAX_BINS`-wide planes at every bin count, is gone in the other direction.

    A block that owns several feature slots occupies a multiple of this.
    `histogram_shared_bytes` is the figure that carries the group width, and
    it is the one a residency estimate has to use.

    The guard in `derive_tiling` reads this, so it asks whether the
    allocation a kernel really makes fits rather than whether an idealized
    `n_bins`-wide one would.
    """
    return histogram_bin_capacity(n_bins) * BYTES_PER_PARTIAL_CELL


# --- The histogram kernel family's two ladders ---
#
# `gpu_active_rows._range_hist_atomic_kernel` and `_range_hist_partial_kernel`
# are parameterized on how many feature slots one threadgroup owns (`GROUP`)
# and how wide its shared planes are (`BIN_CAP`). Both ladders live here
# because this is the module that prices a threadgroup, and both are
# deliberately short: every rung is a kernel instantiation the whole matrix
# pays compile time for on every backend.

comptime HIST_BIN_CAP_MIN = 32
"""Narrowest shared plane a histogram kernel is instantiated at. Below 32
bins the threadgroup memory saved is already small next to the launch, and
the rung would cost a fifth of the matrix to buy it."""

comptime HIST_BIN_CAP_MAX = 256
"""Widest, and the bin ceiling itself: a bin id is a byte, which is what
`gpu_portability.require_bins_supported` refuses more than."""

comptime HIST_FEATURE_GROUP_MAX = 16
"""Widest ladder rung for feature slots per threadgroup. 1, 2, 4, 8, 16."""

comptime HIST_FEATURE_GROUP_LADDER = 5
"""Rungs in that ladder, counting 1."""

comptime HIST_BIN_CAP_LADDER = 4
"""Rungs in the bin-capacity ladder: 32, 64, 128, 256."""


def histogram_bin_capacity(n_bins: Int) -> Int:
    """The shared-plane width a histogram kernel for `n_bins` bins is
    instantiated at: the smallest ladder value at or above `n_bins`, never
    below `HIST_BIN_CAP_MIN`.

    Deliberately not raising, because `shared_bytes_for` above is not, and
    deliberately not saturating at `HIST_BIN_CAP_MAX` either: a bin count past
    the ceiling has no kernel, and returning the next power of two keeps
    `shared_bytes_for` growing so `derive_tiling`'s guard still refuses the
    shape rather than quietly pricing it as a 256-bin one.
    `require_bins_supported` is the check that names the limit.
    """
    var capacity = HIST_BIN_CAP_MIN
    while capacity < n_bins:
        capacity *= 2
    return capacity


def histogram_shared_bytes(bin_cap: Int, group: Int) -> Int:
    """Threadgroup memory one block of the histogram family occupies: three
    Int32 planes of `group * bin_cap` cells each.

    The whole footprint, and the only number a residency estimate for these
    kernels may use. `3 * group * bin_cap * 4` is what the `stack_allocation`
    calls in `gpu_active_rows.mojo` ask for, restated once here so a policy
    layer does not re-derive it from a bin count it happened to hold.
    """
    return group * bin_cap * BYTES_PER_PARTIAL_CELL


def is_feature_group_width(group: Int) -> Bool:
    """Whether `group` is a rung of the feature-group ladder, so a kernel is
    instantiated at it. 3 is not, and is refused rather than rounded, for the
    reason every rung is refused: a width with no kernel cannot launch."""
    var rung = 1
    for _ in range(HIST_FEATURE_GROUP_LADDER):
        if rung == group:
            return True
        rung *= 2
    return False


def feature_group_for_residency(
    caps: DeviceCaps, bin_cap: Int, resident_blocks: Int
) raises -> Int:
    """The widest ladder `GROUP` whose threadgroup footprint at `bin_cap`
    still leaves `resident_blocks` blocks able to share one multiprocessor's
    threadgroup memory.

    The residency question stated in the unit the device reports:
    `resident_blocks * histogram_shared_bytes(bin_cap, group)` must fit in
    `caps.max_shared_memory_per_block`. At one resident block this is simply
    the widest group that launches at all.

    Returns 1 when even a single feature slot does not fit at that residency,
    because 1 is the narrowest kernel there is; whether it launches is
    `gpu_portability`'s question and not this one's. Nothing here selects a
    group. `GpuActiveRows` chooses the default by the free-footprint rule
    below, and this exists for a policy layer that wants to ask the residency
    question directly.
    """
    if bin_cap < 1:
        raise Error("a shared plane covers a positive number of bins")
    if resident_blocks < 1:
        raise Error("a residency target is a positive number of blocks")
    var budget = caps.max_shared_memory_per_block
    var best = 1
    var rung = 1
    for _ in range(HIST_FEATURE_GROUP_LADDER):
        if resident_blocks * histogram_shared_bytes(bin_cap, rung) <= budget:
            best = rung
        rung *= 2
    return best


def free_feature_group(bin_cap: Int, baseline_group: Int) raises -> Int:
    """The widest ladder `GROUP` that costs no more threadgroup memory at
    `bin_cap` than `baseline_group` already costs at the full 256-bin width.

    This is the rule the default group follows, and the reason it can be a
    default rather than a knob. Whatever residency the build had before the
    kernels took a `BIN_CAP` parameter, it had while paying
    `histogram_shared_bytes(256, baseline_group)` per block, because the old
    kernels allocated the maximum width at every bin count. A group whose
    footprint at the real capacity is at or below that number therefore cannot
    reduce the number of resident blocks, whatever the device's threadgroup
    budget turns out to be, so widening to it is free in residency terms and
    provably not a regression on any backend.

    A 64-bin dataset with a baseline of 2 gets 8, and a 32-bin one gets 16,
    for the same bytes. At 256 bins the rule returns the baseline unchanged,
    which is what makes it safe to apply everywhere.

    Anything WIDER than this returns is not free: it trades resident blocks
    for row-side traffic, and that trade is UNMEASURED on every device this
    project runs on. It stays an explicit `MOJOTREES_GPU_FEATURE_GROUP`
    request. What would measure it is an interleaved A/B of the two groups at
    one shape in one process, the protocol `bench_histogram.mojo` already
    uses, on a dataset binned well below 256.
    """
    if bin_cap < 1:
        raise Error("a shared plane covers a positive number of bins")
    if not is_feature_group_width(baseline_group):
        raise Error(
            "a baseline feature group must be a rung of the ladder: 1, 2, 4,"
            " 8, or 16"
        )
    var budget = histogram_shared_bytes(HIST_BIN_CAP_MAX, baseline_group)
    var best = 1
    var rung = 1
    for _ in range(HIST_FEATURE_GROUP_LADDER):
        if histogram_shared_bytes(bin_cap, rung) <= budget:
            best = rung
        rung *= 2
    return best


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def clamp_block_threads(threads: Int, max_threads_per_block: Int) -> Int:
    """A threadgroup width the device can actually launch: clamped to the
    device maximum, rounded down to a warp multiple, never below one warp.

    The rule, separated from the choice of what to clamp.
    `derive_block_threads` clamps the portable target and
    `apple_histogram_policy._shape_block_threads` clamps a row-bounded one;
    both land here, so a width that one layer would refuse is not a width
    the other quietly launches.
    """
    var out = threads
    if out > max_threads_per_block:
        out = max_threads_per_block
    out = (out // WARP_GRANULARITY) * WARP_GRANULARITY
    if out < WARP_GRANULARITY:
        out = WARP_GRANULARITY
    return out


def partial_cell_limit_for(max_partial_cells: Int) -> Int:
    """Partial-histogram cells the tiled strategy may plan against: an
    already-allocated buffer when the caller has one, and otherwise the
    portable byte ceiling converted to cells.

    One (tile, feature, bin) cell carries all three Int32 planes, which is
    what `BYTES_PER_PARTIAL_CELL` counts, so this is the budget in the unit
    `HistogramTiling.partial_cells` reports.
    """
    if max_partial_cells > 0:
        return max_partial_cells
    return PARTIAL_BUDGET_BYTES // BYTES_PER_PARTIAL_CELL


def derive_block_threads(caps: DeviceCaps) -> Int:
    """Threads per threadgroup: the target, clamped to the device maximum,
    rounded down to a warp multiple, never below one warp."""
    var requested = _env_int("MOJOTREES_GPU_BLOCK_THREADS", 0)
    var threads = requested if requested > 0 else TARGET_BLOCK_THREADS
    return clamp_block_threads(threads, caps.max_threads_per_block)


def resolve_tiling(
    n_rows: Int,
    n_slots: Int,
    n_bins: Int,
    block_threads: Int,
    amortize_bins: Int,
    target_blocks: Int,
    partial_cell_limit: Int,
    requested_strategy: Int = STRATEGY_AUTO,
) raises -> HistogramTiling:
    """Resolve a tile count, a row tile, and a strategy from bounds the
    caller has already decided.

    The tile arithmetic itself, factored out of `derive_tiling` so the
    shape-derived policy in `apple_histogram_policy.mojo` can run the same
    rules against different bounds instead of restating them. What a caller
    supplies is where the two layers legitimately differ:

    - `block_threads`: the portable target, or a row-bounded width.
    - `amortize_bins`: the bin width a threadgroup's partial histogram really
      occupies, which is `n_bins` for the shipping kernels and the kernel's
      bin capacity for the specialized ones.
    - `target_blocks`: threadgroups wanted device-wide, which is
      `sm_count * TARGET_BLOCKS_PER_SM` portably and `core_count * resident`
      when residency was derived from the reported threadgroup memory.
    - `partial_cell_limit`: cells the partial buffer may use, from
      `partial_cell_limit_for` or from a reported memory budget.

    Everything after that is one rule, applied once: tiles from the tightest
    of the occupancy, row-amortization, and memory bounds; the
    `MOJOTREES_GPU_ROW_TILE` override; the grid bound; a re-derivation so the
    last tile is never empty; and the `STRATEGY_AUTO` resolution.
    """
    if n_rows < 1 or n_slots < 1 or n_bins < 1:
        raise Error("tiling needs positive rows, features, and bins")
    if block_threads < 1:
        raise Error("a threadgroup needs a positive width")
    if amortize_bins < 1:
        raise Error("a partial histogram covers a positive number of bins")
    if target_blocks < 1:
        raise Error("a device wants a positive number of threadgroups")
    if partial_cell_limit < 1:
        raise Error("the partial buffer budget must admit at least one cell")

    var strategy = requested_strategy
    if strategy == STRATEGY_AUTO:
        strategy = env_strategy()

    var min_rows_per_tile = MIN_ROWS_PER_TILE_BIN_FACTOR * amortize_bins
    var by_threads = MIN_ROWS_PER_TILE_THREAD_FACTOR * block_threads
    if by_threads > min_rows_per_tile:
        min_rows_per_tile = by_threads
    var tiles_by_rows = _ceil_div(n_rows, min_rows_per_tile)
    if tiles_by_rows < 1:
        tiles_by_rows = 1

    var tiles_by_occupancy = _ceil_div(target_blocks, n_slots)
    if tiles_by_occupancy < 1:
        tiles_by_occupancy = 1

    var wanted = tiles_by_occupancy
    if tiles_by_rows < wanted:
        wanted = tiles_by_rows

    var hist_cells = n_slots * n_bins
    var tiles_by_memory = partial_cell_limit // hist_cells
    if tiles_by_memory < 1:
        tiles_by_memory = 1
    if tiles_by_memory > MAX_GRID_DIM_Y:
        tiles_by_memory = MAX_GRID_DIM_Y

    var n_tiles = wanted
    var forced_rows = _env_int("MOJOTREES_GPU_ROW_TILE", 0)
    if forced_rows > 0:
        n_tiles = _ceil_div(n_rows, forced_rows)
    # The atomic path allocates no partial buffer, so the memory bound is
    # not its bound. Under AUTO the clamp still applies, because AUTO may
    # resolve to the tiled path below.
    if strategy != STRATEGY_ATOMIC and n_tiles > tiles_by_memory:
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

    The bounds are this module's; the arithmetic over them is
    `resolve_tiling`'s, which is also what the shape-derived policy runs.
    """
    if n_rows < 1 or n_features < 1 or n_bins < 1:
        raise Error("tiling needs positive rows, features, and bins")
    if shared_bytes_for(n_bins) > caps.max_shared_memory_per_block:
        raise Error(
            "device shared memory too small for a per-threadgroup histogram"
        )
    # A device reporting no multiprocessors is a device that answered
    # nothing, and one threadgroup per feature is what this asked for before
    # the occupancy bound existed. `query_device_caps` substitutes the
    # portable constant for a missing attribute, so this only catches a
    # hand-built `DeviceCaps`.
    var target_blocks = caps.sm_count * TARGET_BLOCKS_PER_SM
    if target_blocks < 1:
        target_blocks = 1
    return resolve_tiling(
        n_rows,
        n_features,
        n_bins,
        derive_block_threads(caps),
        n_bins,
        target_blocks,
        partial_cell_limit_for(max_partial_cells),
        requested_strategy,
    )


def strategy_name(strategy: Int) -> String:
    if strategy == STRATEGY_ATOMIC:
        return String("atomic")
    if strategy == STRATEGY_TILED:
        return String("tiled")
    return String("auto")


def rows_per_thread(tiling: HistogramTiling) -> Int:
    """Rows one lane accumulates over its tile.

    Derived, not chosen: `rows_per_tile` and `block_threads` are the
    decisions and this reports what they imply. It is at least
    `MIN_ROWS_PER_TILE_THREAD_FACTOR` whenever the row bound is what set the
    tile count, and below it only on a node with too few rows to feed one
    threadgroup, where no tiling can raise it. Reported so a benchmark can
    say how much work a lane did per partial-histogram zero and flush, which
    is the ratio the two `MIN_ROWS_PER_TILE_*` constants exist to bound.
    """
    return _ceil_div(tiling.rows_per_tile, tiling.block_threads)


def describe_tiling(tiling: HistogramTiling) -> String:
    """One line for benchmark output and bug reports."""
    return String(
        "strategy=",
        strategy_name(tiling.strategy),
        " threads=",
        tiling.block_threads,
        " tiles=",
        tiling.n_tiles,
        " rows_per_tile=",
        tiling.rows_per_tile,
        " rows_per_thread=",
        rows_per_thread(tiling),
        " partial_cells=",
        tiling.partial_cells,
    )
