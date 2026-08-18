"""Apple-silicon tuning policy.

A pure policy layer: reported device facts and a dataset shape in, a launch
plan out. It computes threadgroup size, rows per tile, histogram strategy,
the partial-buffer limit, and the inputs a CPU/GPU crossover rule would
need. It opens no device, allocates nothing, and launches nothing, so every
decision below is testable on a CPU-only machine.

Why a separate layer at all. `gpu_tiling.mojo` derives its geometry from
three numbers (multiprocessor count, threads per threadgroup, shared memory
per threadgroup) and constants chosen for a discrete GPU. Apple silicon
differs from that shape in ways that are architectural rather than measured,
and those are the only differences this module encodes:

- Unified memory. On Metal the device budget is system RAM, shared with the
  dataset the host is holding, so a fixed 64 MiB partial-histogram buffer is
  a larger claim than the same buffer on a discrete card with its own VRAM.
  The partial budget here is a fraction of the reported budget, with a
  tighter fraction when memory is unified, and the portable 64 MiB ceiling
  still caps it.
- Small core counts. Apple GPUs report core counts an order of magnitude
  below a large discrete part, so a fixed "blocks per multiprocessor" target
  either underfills the small device or overshoots the large one. The
  resident-block target here is derived from what actually limits residency
  and is reported: how many `n_bins`-wide partial histograms fit in the
  threadgroup memory the device advertises.
- Narrow datasets. A 256-thread threadgroup scanning 40 rows leaves most of
  its lanes idle, so the threadgroup size is also bounded by the row count.

What this module deliberately does NOT encode:

- No per-chip constant is used as a default for any other chip. The
  development machine for this project is an M4; `apple_m4_observed()` is
  the one capability triple this repository has read off real hardware, and
  it is a named fixture, never a fallback. Everything else derives from what
  the running device reports, and `GpuProfile.generic()` is the portable
  answer for a device that reports nothing, on Metal, CUDA, or HIP alike.
- No divergence in the strategy choice. Apple GPUs implement global
  integer atomics, but nothing in this project has measured atomic
  throughput against the tiled reduction on any Apple part, so the rule
  here is the same one `gpu_tiling.mojo` already ships: reduce when there is
  more than one tile to reduce, use atomics when there is not.
- No crossover threshold. `CrossoverInputs.min_cells` is
  `CROSSOVER_DISABLED`, because the only end-to-end GPU training
  measurement taken (M4, bench/bench_train_gpu.mojo) is slower than the CPU
  trainer. This module reports the inputs such a rule would key on so the
  benchmark that would justify a threshold has somewhere to put its answer;
  it does not invent the answer, and it does not decide anything with it.
  `device_policy.mojo` is where the crossover evidence table lives and
  where `auto` is resolved; it imports `CROSSOVER_DISABLED` from here so
  the disabled sentinel has one definition rather than two.

Where this module sits. It is the leaf of the device layer: it imports
only `gpu_tiling.mojo` (for the portable geometry constants and the block
clamp; that module opens no device either, its `DeviceContext` use is
behind `query_device_caps`, which nothing here calls), opens nothing, and
is imported by `device_policy.mojo`, which turns a `GpuProfile` into the
hardware half of a device decision. Keeping the dependency in that
direction is what lets the whole policy stack be exercised on a machine
with no accelerator.

The synthetic fixtures. `apple_synthetic()` returns a deliberately
conservative capability profile per M-series generation, and every one of
them is flagged `synthetic`. The core counts are published base-configuration
GPU core counts, not numbers this project read from a device, and the
threadgroup memory is the portable Apple floor rather than what any
particular chip advertises: underestimating costs some parallelism, whereas
overestimating produces a plan the device cannot run. None of them is a
performance measurement, and no measurement of M1, M2, M3, or M5 exists in
this repository. They are here so the policy can be exercised across the
generation range on a machine that has none of them, and each should be
replaced by a reading from real hardware, clearing `synthetic`, as that
hardware becomes available.

The portable geometry constants (`STRATEGY_*`, `TARGET_BLOCK_THREADS`,
`WARP_GRANULARITY`, `BYTES_PER_PARTIAL_CELL`, `MAX_GRID_DIM_Y`, the
`FALLBACK_*` reporting defaults, the `MIN_ROWS_PER_TILE_*` factors) and
`strategy_name` are `gpu_tiling.mojo`'s, imported here and re-exported
under the same names; earlier revisions carried copies, pinned equal by
test_apple_gpu_policy.mojo, until the consolidation round of 2026-08
collapsed them into these imports. What this module defines itself is only
what is Apple-specific or profile-specific: the API and generation
vocabulary, `GpuProfile` and its fixtures, the unified-memory partial
budget, the residency estimate, the row-bounded block width, and the
tuning plan over them.
"""


from .gpu_tiling import (
    BYTES_PER_PARTIAL_CELL,
    FALLBACK_MAX_THREADS_PER_BLOCK,
    FALLBACK_SHARED_MEMORY_PER_BLOCK,
    FALLBACK_SM_COUNT,
    MAX_GRID_DIM_Y,
    MIN_ROWS_PER_TILE_BIN_FACTOR,
    MIN_ROWS_PER_TILE_THREAD_FACTOR,
    PARTIAL_BUDGET_BYTES,
    STRATEGY_ATOMIC,
    STRATEGY_AUTO,
    STRATEGY_TILED,
    TARGET_BLOCK_THREADS,
    TARGET_BLOCKS_PER_SM,
    WARP_GRANULARITY,
    clamp_block_threads,
    row_tile_floor,
    shared_bytes_for as tiling_shared_bytes_for,
    strategy_name,
)

# The portable geometry constants above are gpu_tiling.mojo's, imported and
# re-exported rather than restated: the strategy codes, the block target and
# warp granularity, the partial-cell size, the grid.y bound, the reporting
# fallbacks, and the rows-per-tile factors have one definition. The three
# below are the same facts under the names this layer has always used.

# Absolute ceiling on the partial-histogram buffer, whatever the device
# budget is: past this the tiled strategy trades more memory than the atomic
# one saves.
comptime PARTIAL_BUDGET_CEILING_BYTES = PARTIAL_BUDGET_BYTES

# Used when a backend reports nothing. Low rather than typical, for the
# reason given in gpu_tiling.mojo. Cores on Apple, multiprocessors elsewhere:
# the same reported quantity.
comptime FALLBACK_CORE_COUNT = FALLBACK_SM_COUNT


comptime API_UNKNOWN = 0
comptime API_METAL = 1
comptime API_CUDA = 2
comptime API_HIP = 3

comptime APPLE_GEN_UNKNOWN = 0
comptime APPLE_GEN_M1 = 1
comptime APPLE_GEN_M2 = 2
comptime APPLE_GEN_M3 = 3
comptime APPLE_GEN_M4 = 4
comptime APPLE_GEN_M5 = 5

# Ceiling on threadgroups resident per core. The real bound is derived from
# the reported threadgroup memory; this caps it so a narrow histogram cannot
# ask for unbounded residency. It is gpu_tiling.mojo's fixed target, so a
# device whose shared memory fits that many partials plans exactly as the
# portable rule does.
comptime MAX_RESIDENT_BLOCKS_PER_CORE = TARGET_BLOCKS_PER_SM

# Fraction of the reported memory budget the partial buffer may claim.
# Unified memory gets the tighter one: that budget is also holding the
# dataset, the binned matrix, and the host's copy of both.
comptime UNIFIED_PARTIAL_BUDGET_DIVISOR = 32
comptime DISCRETE_PARTIAL_BUDGET_DIVISOR = 16

# `CrossoverInputs.min_cells` when no measurement supports a threshold,
# which is every case today. Negative disables. The one definition of the
# disabled sentinel: device_policy.mojo imports it rather than restating
# it, so the two layers cannot drift into disagreeing about what "no
# crossover" means.
comptime CROSSOVER_DISABLED = -1

# Threadgroup memory the synthetic Apple fixtures claim: the portable Apple
# floor, not what any measured chip advertises. See the module docstring.
comptime SYNTHETIC_APPLE_SHARED_BYTES = 16384

# The one Apple triple this repository has read from hardware: the M4 the
# project is developed on, the same numbers tests/test_gpu_tiling.mojo
# already pins as "Apple M4 reports these exactly".
comptime OBSERVED_M4_CORE_COUNT = 10
comptime OBSERVED_M4_MAX_THREADS_PER_BLOCK = 1024
comptime OBSERVED_M4_SHARED_BYTES = 32768

# The architecture string that same M4 reports, verbatim.
#
# MEASURED, 2026-08-16: `DeviceContext().arch_name()` on the development M4
# returns exactly `4-metal4`, and `std.sys.info._accelerator_arch()` on a
# build targeting it returns `metal:4-metal4`. Both were read, not recalled;
# `gpu_split_policy._is_observed_m4` had already hardcoded the same literal
# for the same reason and from the same reading.
#
# It is a constant rather than a pattern because one reading is all this
# project has. `parse_apple_generation` below matches it exactly and
# extrapolates nothing: it is not known whether the leading digit is the
# chip generation or the Metal feature-set family, and those two readings
# disagree about what an M3 or an M5 would report. Guessing wrong there does
# not fail loudly, it silently applies M4-measured evidence to a chip that
# was never measured, which is the one outcome the crossover scope exists to
# prevent. A second reading, from any other generation, is what would turn
# this into a grammar.
comptime OBSERVED_M4_ARCH_NAME = String("4-metal4")


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


def _positive_or(value: Int, default: Int) -> Int:
    """A reported attribute, or `default` when it is absent or nonsense."""
    if value > 0:
        return value
    return default


def _mentions(text: String, spellings: List[String]) -> Bool:
    """Whether `text` contains any of `spellings`.

    Case is handled by listing the spellings rather than by folding the
    string: what a backend returns from an API-name query is not a
    normalized identifier, and this keeps the parse to `String.find`, which
    is the only string search this package relies on elsewhere.
    """
    for i in range(len(spellings)):
        if text.find(spellings[i]) >= 0:
            return True
    return False


def parse_api(api_name: String) -> Int:
    """The GPU API a reported name denotes, or `API_UNKNOWN`.

    `API_UNKNOWN` is not an error: it selects the portable path, which is
    the same path a recognized API takes minus the unified-memory
    adjustment.
    """
    var metal: List[String] = ["metal", "Metal", "METAL"]
    if _mentions(api_name, metal):
        return API_METAL
    var cuda: List[String] = ["cuda", "Cuda", "CUDA"]
    if _mentions(api_name, cuda):
        return API_CUDA
    var hip: List[String] = ["hip", "Hip", "HIP", "rocm", "ROCm", "ROCM"]
    if _mentions(api_name, hip):
        return API_HIP
    return API_UNKNOWN


def api_name(api: Int) -> String:
    if api == API_METAL:
        return String("metal")
    if api == API_CUDA:
        return String("cuda")
    if api == API_HIP:
        return String("hip")
    return String("unknown")


def parse_apple_generation(arch_name: String) -> Int:
    """The M-series generation an architecture string names, or
    `APPLE_GEN_UNKNOWN`.

    Only meaningful for Metal, and `GpuProfile.from_reported` only calls it
    there: a CUDA or HIP architecture string is not an Apple part number and
    must not be mined for one. An ambiguous string that names two
    generations resolves to unknown rather than to whichever matched first,
    because a wrong generation is worse than none: none falls back to
    reported capabilities, which are correct by construction.

    The generation is metadata for the crossover inputs and for diagnostics.
    Nothing in `derive_policy` branches on it, and nothing should until a
    measurement distinguishes the generations.

    THE SPELLING A REAL DEVICE USES IS NOT THE SPELLING THIS PARSED. Until
    2026-08-16 this function recognized only human-readable forms ("Apple
    M4", "m4"), and the string an actual Metal device reports is
    `4-metal4`, which contains no "m4" substring: "metal4" ends in "l4".
    So every profile built by `GpuProfile.from_reported` from a real Apple
    reading came back with `APPLE_GEN_UNKNOWN`, and the one
    generation-scoped crossover rule in device_policy.mojo could not fire
    for any caller, including `capabilities_from_reported` and
    `decide_device_report_reported`, which exist for precisely that caller.
    Nothing caught it because `apple_m4_observed()` builds its profile with
    the fieldwise constructor and hands `APPLE_GEN_M4` in directly, so the
    tests that assert the rule fires were asserting it against a value this
    parser could never produce.

    `OBSERVED_M4_ARCH_NAME` fixes that for the one string this project has
    read, exactly, and for no other. See that constant for why it is not a
    pattern.
    """
    if arch_name == OBSERVED_M4_ARCH_NAME:
        return APPLE_GEN_M4
    var found = APPLE_GEN_UNKNOWN
    for generation in range(APPLE_GEN_M1, APPLE_GEN_M5 + 1):
        var lower = String("m", generation)
        var upper = String("M", generation)
        if arch_name.find(lower) >= 0 or arch_name.find(upper) >= 0:
            if found != APPLE_GEN_UNKNOWN:
                return APPLE_GEN_UNKNOWN
            found = generation
    return found


def apple_generation_name(generation: Int) -> String:
    if generation >= APPLE_GEN_M1 and generation <= APPLE_GEN_M5:
        return String("m", generation)
    return String("unknown")


@fieldwise_init
struct GpuProfile(Copyable, Movable):
    """Everything the policy is allowed to know about a device."""

    var api: Int
    var apple_generation: Int
    """`APPLE_GEN_UNKNOWN` on every non-Metal device, and on a Metal device
    whose architecture string does not name one generation."""

    var core_count: Int
    """GPU cores on Apple, multiprocessors elsewhere: the same reported
    quantity, `MULTIPROCESSOR_COUNT`."""

    var max_threads_per_block: Int
    var max_shared_memory_per_block: Int

    var memory_budget_bytes: Int
    """Device memory the policy may plan against. Zero means unreported, in
    which case the partial buffer falls back to the portable ceiling."""

    var unified_memory: Bool
    """True when the budget above is shared with the host, which is every
    Apple silicon GPU and no discrete one."""

    var synthetic: Bool
    """True when these numbers were constructed rather than read from a
    device. Never affects a decision; it exists so a caller can tell a
    fixture from a reading, and so tests can assert that every fixture is
    labeled."""

    @staticmethod
    def generic() -> GpuProfile:
        """The portable fallback: a device that reports nothing useful, on
        any of Metal, CUDA, or HIP. Deliberately not Apple-shaped."""
        return GpuProfile(
            API_UNKNOWN,
            APPLE_GEN_UNKNOWN,
            FALLBACK_CORE_COUNT,
            FALLBACK_MAX_THREADS_PER_BLOCK,
            FALLBACK_SHARED_MEMORY_PER_BLOCK,
            0,
            False,
            True,
        )

    @staticmethod
    def from_reported(
        reported_api: String,
        reported_arch: String,
        core_count: Int,
        max_threads_per_block: Int,
        max_shared_memory_per_block: Int,
        memory_budget_bytes: Int = 0,
    ) -> GpuProfile:
        """A profile from what a device actually reported.

        Every numeric argument is sanitized the way `gpu_tiling.mojo`
        sanitizes an attribute query: absent or nonsense falls back to the
        conservative portable constant, so a backend that answers nothing
        degrades the plan instead of producing an unlaunchable one.

        The caller does the querying. Keeping `DeviceContext` out of this
        module is what lets the whole policy be tested without an
        accelerator. The call site reads the attributes and hands them here.
        """
        var api = parse_api(reported_api)
        var generation = APPLE_GEN_UNKNOWN
        if api == API_METAL:
            generation = parse_apple_generation(reported_arch)
        return GpuProfile(
            api,
            generation,
            _positive_or(core_count, FALLBACK_CORE_COUNT),
            _positive_or(
                max_threads_per_block, FALLBACK_MAX_THREADS_PER_BLOCK
            ),
            _positive_or(
                max_shared_memory_per_block, FALLBACK_SHARED_MEMORY_PER_BLOCK
            ),
            _positive_or(memory_budget_bytes, 0),
            api == API_METAL,
            False,
        )

    def is_apple(self) -> Bool:
        return self.api == API_METAL


def synthetic_apple_core_count(generation: Int) raises -> Int:
    """Published base-configuration GPU core count for an M-series
    generation, taking the lowest published base configuration where a
    generation ships more than one.

    Specifications, not measurements, and not performance figures of any
    kind: this sets how many threadgroups the policy aims for, nothing else.
    Erring low is the safe direction, for the reason `gpu_tiling.mojo` gives
    for its own fallback core count.
    """
    if generation == APPLE_GEN_M1:
        return 7
    if generation == APPLE_GEN_M2:
        return 8
    if generation == APPLE_GEN_M3:
        return 8
    if generation == APPLE_GEN_M4:
        return 10
    if generation == APPLE_GEN_M5:
        return 10
    raise Error(
        "no synthetic core count for Apple generation ",
        generation,
        "; expected ",
        APPLE_GEN_M1,
        " through ",
        APPLE_GEN_M5,
    )


def apple_synthetic(generation: Int) raises -> GpuProfile:
    """A conservative, explicitly synthetic profile for an M-series
    generation.

    Marked `synthetic` because it is: only the core count varies by
    generation, the threadgroup memory is the portable Apple floor rather
    than what the chip advertises, and no part of it was read from hardware.
    Use it to exercise the policy across the generation range, never as a
    substitute for querying the device in front of you. See
    `apple_m4_observed()` for the one profile that is a reading.
    """
    return GpuProfile(
        API_METAL,
        generation,
        synthetic_apple_core_count(generation),
        FALLBACK_MAX_THREADS_PER_BLOCK,
        SYNTHETIC_APPLE_SHARED_BYTES,
        0,
        True,
        True,
    )


def apple_m4_observed() -> GpuProfile:
    """The M4 this project is developed on, as that device reports itself.

    The only Apple capability reading in this repository, and a named
    fixture rather than any kind of default: no other device, Apple or not,
    inherits these numbers. The memory budget is zero because it has not
    been read, and nothing here is a performance measurement.
    """
    return GpuProfile(
        API_METAL,
        APPLE_GEN_M4,
        OBSERVED_M4_CORE_COUNT,
        OBSERVED_M4_MAX_THREADS_PER_BLOCK,
        OBSERVED_M4_SHARED_BYTES,
        0,
        True,
        False,
    )


def shared_bytes_for_bins(n_bins: Int) -> Int:
    """Threadgroup memory one block needs for its partial histogram, for a
    block that owns one feature slot.

    Delegates rather than restating the arithmetic. The two used to agree by
    both spelling `n_bins * BYTES_PER_PARTIAL_CELL`, and they stopped
    agreeing when the histogram kernels became capacity-sized: a launch at
    63 bins now allocates the 64-bin rung, so the modeled footprint has to
    round up the same ladder or it under-reports what the kernel really
    takes. `gpu_tiling.shared_bytes_for` is where that rounding lives.

    One slot, not one block: a threadgroup that owns a feature group of G
    occupies G times this. Nothing here knows the group, which is why
    `GpuActiveRows.set_feature_group` is what actually refuses a width the
    device cannot hold.
    """
    return tiling_shared_bytes_for(n_bins)


def partial_budget_bytes(profile: GpuProfile) -> Int:
    """Bytes the partial-histogram buffer may claim on this device.

    A fraction of the reported budget, tighter when memory is unified,
    capped by the portable ceiling. An unreported budget gets the ceiling,
    which is what `gpu_tiling.mojo` uses unconditionally today, so a device
    that says nothing about its memory plans exactly as it does now.
    """
    if profile.memory_budget_bytes <= 0:
        return PARTIAL_BUDGET_CEILING_BYTES
    var divisor = DISCRETE_PARTIAL_BUDGET_DIVISOR
    if profile.unified_memory:
        divisor = UNIFIED_PARTIAL_BUDGET_DIVISOR
    var bytes = profile.memory_budget_bytes // divisor
    if bytes > PARTIAL_BUDGET_CEILING_BYTES:
        bytes = PARTIAL_BUDGET_CEILING_BYTES
    if bytes < 0:
        bytes = 0
    return bytes


def resident_blocks_for_bytes(profile: GpuProfile, per_block: Int) raises -> Int:
    """Threadgroups the policy expects to be resident on one core, for a
    block that occupies `per_block` bytes of threadgroup memory.

    The one residency rule: how many such blocks fit in the advertised
    threadgroup memory, capped at `MAX_RESIDENT_BLOCKS_PER_CORE` so a narrow
    histogram does not ask for unbounded residency, and floored at one so a
    block that only just fits still gets scheduled. `resident_blocks_per_core`
    below applies it to the modeled `n_bins * BYTES_PER_PARTIAL_CELL`
    footprint; `apple_histogram_policy.resident_blocks_for_kernel` applies
    it to the footprint the compiled kernel really has.
    """
    if per_block < 1:
        raise Error("a histogram block occupies a positive number of bytes")
    var fits = profile.max_shared_memory_per_block // per_block
    if fits > MAX_RESIDENT_BLOCKS_PER_CORE:
        fits = MAX_RESIDENT_BLOCKS_PER_CORE
    if fits < 1:
        fits = 1
    return fits


def resident_blocks_per_core(profile: GpuProfile, n_bins: Int) raises -> Int:
    """Threadgroups the policy expects to be resident on one core.

    Bounded by the thing that actually limits residency and is reported:
    how many `n_bins`-wide partial histograms fit in the advertised
    threadgroup memory. `resident_blocks_for_bytes` over the modeled
    per-block footprint.
    """
    if n_bins < 1:
        raise Error("histogram needs a positive bin count")
    return resident_blocks_for_bytes(profile, shared_bytes_for_bins(n_bins))


def shape_block_threads(profile: GpuProfile, n_rows: Int) -> Int:
    """Threads per threadgroup: a warp multiple, at least one warp, never
    above the device maximum, and never so wide that most lanes would sit
    idle over the rows available.

    The row bound is the shape-driven part, and the only part that is this
    layer's: a 256-thread block scanning 40 rows leaves seven eighths of its
    lanes with nothing to accumulate, which matters most on the small core
    counts Apple reports, where those idle lanes are a large fraction of the
    device. The clamp itself is `gpu_tiling.clamp_block_threads`, the one
    rule for a launchable width, so a width this layer would produce is
    never one the portable layer would refuse. (Formerly named
    `derive_block_threads` here; renamed so the portable
    `gpu_tiling.derive_block_threads`, which takes a `DeviceCaps` and reads
    the block-threads override, is the only function of that name.)
    """
    var threads = TARGET_BLOCK_THREADS
    if n_rows > 0 and n_rows < threads:
        threads = n_rows
    return clamp_block_threads(threads, profile.max_threads_per_block)


@fieldwise_init
struct CrossoverInputs(Copyable, Movable):
    """What a CPU/GPU crossover rule would key on, and the threshold it
    would produce.

    Reported so a crossover benchmark has a defined set of inputs to regress
    against and a defined place to put its answer. `min_cells` is
    `CROSSOVER_DISABLED` in every case this module can construct, because no
    such benchmark has run: see the module docstring, and
    `crossover_rules()` in device_policy.mojo, which is empty for the same
    reason and which is where a measured rule is installed.

    Reporting only. Nothing reads `min_cells` off this value to decide a
    device; `decide_device` consults the evidence table instead.
    """

    var api: Int
    var apple_generation: Int
    var core_count: Int
    var unified_memory: Bool
    var cells: Int
    """`n_rows * n_features`, the size measure `device.mojo` already uses."""

    var n_bins: Int
    var device_parallel_width: Int
    """`core_count * block_threads`: the thread count the device would run
    concurrently for this shape. Carried because a bare cell count cannot
    stand in for it: the same shape means very different things on a 7-core
    part and on a 108-core one, and a crossover rule keyed on cells alone
    would have to be refit per device."""

    var min_cells: Int
    """Cells at or above which the GPU would be chosen. Negative disables,
    and it is always negative here."""


@fieldwise_init
struct TuningPolicy(Copyable, Movable):
    """A resolved plan for one device and one dataset shape."""

    var block_threads: Int
    var n_tiles: Int
    var rows_per_tile: Int
    var strategy: Int
    var partial_cell_limit: Int
    """Ceiling on `partial_cells` from the memory budget, in (tile, feature,
    bin) cells. This is the value to hand to the tiling module as its
    already-allocated capacity."""

    var partial_cells: Int
    """`n_tiles * n_features * n_bins`, or zero when the resolved strategy
    allocates no partial buffer."""

    var resident_blocks_per_core: Int
    var crossover: CrossoverInputs


def derive_policy(
    profile: GpuProfile,
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    requested_strategy: Int = STRATEGY_AUTO,
) raises -> TuningPolicy:
    """Resolve a launch plan for one device profile and one dataset shape.

    Pure host-side arithmetic. Row tiles come from the tightest of three
    bounds, as in `gpu_tiling.mojo`: enough threadgroups to fill the device,
    enough rows per tile to amortize the partial histogram, and the memory
    the partial buffer may claim. What differs is where the first and third
    bounds come from, which is the whole subject of this module: residency
    derived from reported threadgroup memory rather than fixed, and a budget
    derived from reported device memory rather than a constant.

    An explicit `requested_strategy` wins, as it does in the tiling module.
    Requesting the tiled strategy on a shape whose partial buffer exceeds
    `partial_cell_limit` is honored and reported, not clamped: the caller
    asked, `partial_cell_limit` is on the returned plan, and the caller can
    see it was exceeded.
    """
    if n_rows < 1 or n_features < 1 or n_bins < 1:
        raise Error("policy needs positive rows, features, and bins")
    if shared_bytes_for_bins(n_bins) > profile.max_shared_memory_per_block:
        raise Error(
            "device shared memory too small for a per-threadgroup histogram"
        )

    var block_threads = shape_block_threads(profile, n_rows)
    var resident = resident_blocks_per_core(profile, n_bins)

    # Enough threadgroups to fill the device. `row_tile_floor` is the shared
    # rule, called rather than restated: dividing the device-wide block
    # target by the feature count, as this used to, makes the feature count a
    # ceiling on the whole launch instead of a floor under it, so a wide
    # dataset collapsed to a single row tile however many rows a node held.
    # This file reaching its own answer was how the specialized Apple path
    # kept the old geometry after the shared rule was fixed.
    var target_blocks = profile.core_count * resident
    var tiles_by_occupancy = row_tile_floor(target_blocks, n_features)

    # Enough rows per tile to pay for writing or folding the partial.
    var min_rows_per_tile = MIN_ROWS_PER_TILE_BIN_FACTOR * n_bins
    var by_threads = MIN_ROWS_PER_TILE_THREAD_FACTOR * block_threads
    if by_threads > min_rows_per_tile:
        min_rows_per_tile = by_threads
    var tiles_by_rows = _ceil_div(n_rows, min_rows_per_tile)
    if tiles_by_rows < 1:
        tiles_by_rows = 1

    var wanted = tiles_by_occupancy
    if tiles_by_rows < wanted:
        wanted = tiles_by_rows

    # What the memory budget allows.
    var hist_cells = n_features * n_bins
    var partial_cell_limit = (
        partial_budget_bytes(profile) // BYTES_PER_PARTIAL_CELL
    )
    var tiles_by_memory = partial_cell_limit // hist_cells
    if tiles_by_memory < 1:
        tiles_by_memory = 1
    if tiles_by_memory > MAX_GRID_DIM_Y:
        tiles_by_memory = MAX_GRID_DIM_Y

    var strategy = requested_strategy
    var n_tiles = wanted
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
        # More than one tile is exactly what the tiled path exists for; at
        # one tile there is nothing to reduce and the preserved atomic path
        # is the better default. Same rule as gpu_tiling.mojo, deliberately:
        # no Apple measurement supports diverging from it.
        if n_tiles > 1:
            strategy = STRATEGY_TILED
        else:
            strategy = STRATEGY_ATOMIC

    var partial_cells = 0
    if strategy == STRATEGY_TILED:
        partial_cells = n_tiles * hist_cells

    return TuningPolicy(
        block_threads,
        n_tiles,
        rows_per_tile,
        strategy,
        partial_cell_limit,
        partial_cells,
        resident,
        CrossoverInputs(
            profile.api,
            profile.apple_generation,
            profile.core_count,
            profile.unified_memory,
            n_rows * n_features,
            n_bins,
            profile.core_count * block_threads,
            CROSSOVER_DISABLED,
        ),
    )


def describe_gpu_policy(profile: GpuProfile, policy: TuningPolicy) -> String:
    """One line for benchmark output and bug reports: which device the plan
    was made for, whether that device was real, and what it decided."""
    var origin = String("reported")
    if profile.synthetic:
        origin = String("synthetic")
    return String(
        "api=",
        api_name(profile.api),
        " apple_gen=",
        apple_generation_name(profile.apple_generation),
        " cores=",
        profile.core_count,
        " (",
        origin,
        ") strategy=",
        strategy_name(policy.strategy),
        " block_threads=",
        policy.block_threads,
        " tiles=",
        policy.n_tiles,
        " rows_per_tile=",
        policy.rows_per_tile,
        " partial_cells=",
        policy.partial_cells,
        "/",
        policy.partial_cell_limit,
    )
