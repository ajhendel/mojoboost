"""Which GPU histogram specialization to use, and when not to.

`gpu_histogram_specializations.mojo` says what a specialized histogram path
could be built out of. This module decides whether any of it applies to one
device and one node, and its answer today, on every device, is **no**.

That is the point of the module rather than a gap in it. Nothing in this
repository has measured a specialized histogram kernel against the shipping
one on any hardware, and a launch policy that is faster in principle is not a
launch policy that is faster. So the specialization ladder below is
default-off: with no explicit request and no environment override, a plan's
launch geometry is `gpu_tiling.derive_tiling`'s, field for field, obtained by
calling it rather than by reproducing it.

The ladder
----------
Cumulative, each level a superset of the one below:

- `SPEC_LEVEL_BASELINE` -- today's geometry, verbatim from `derive_tiling`.
  The default.
- `SPEC_LEVEL_SHAPE` -- geometry re-derived from reported device properties
  and the node's own shape: threadgroup residency from the threadgroup memory
  a block really occupies, threadgroup width bounded by the rows available,
  row tiles amortized against the kernel's bin capacity, partial budget as a
  fraction of reported device memory. No new kernel: every one of these is a
  number handed to the launch that exists.
- `SPEC_LEVEL_PACKED` -- additionally reads four bins per load over a
  contiguous aligned run of row ids. Needs the packed kernel variant and a
  device that has shown the wide load pays.
- `SPEC_LEVEL_BATCHED` -- additionally covers several small leaves in one
  launch. Needs the batched kernel variant. The launch itself is planned by
  `gpu_leaf_batching.plan_batch`, which owns the tile sharing, the partial
  buffer bound, and the kernels; this module contributes only the decision of
  whether a frontier wants batching at all.

Fallback order, applied in this order and reported on the plan
--------------------------------------------------------------
1. A level above what `KernelFeatures` says is compiled in falls to the
   highest level that is.
2. `SPEC_LEVEL_PACKED` falls to `SPEC_LEVEL_SHAPE` when the device has not
   reported that wide loads pay, when the node's rows are not a contiguous
   run, when the column stride is not a multiple of the pack width, or when
   the aligned body is too short to pay for the head and tail.
3. `SPEC_LEVEL_SHAPE` falls to `SPEC_LEVEL_BASELINE` when the shape-derived
   geometry cannot be built (a bin count outside the class ladder, a block
   that does not fit threadgroup memory).
4. `SPEC_LEVEL_BASELINE` is `derive_tiling`, which has its own fallback
   already: the tiled strategy falls to the atomic one whenever the partial
   buffer will not fit, and every device attribute the query refuses falls to
   the portable constant.

Every step down is recorded in `HistogramPlan.reason`, so a plan that did
less than was asked says why rather than looking like the plan that was
asked for.

`HistogramPlan.level_applied` is a contiguous position on that ladder, and
it is a statement about **one node**: a node whose rows are not a contiguous
run stops at `SPEC_LEVEL_SHAPE` however capable the build is. Batching is a
decision across nodes, so `batching_declined_reason` gates on
`level_requested` and `KernelFeatures` rather than on `level_applied`;
otherwise a frontier would lose its batching because one node in it happened
not to be contiguous.

What else is planned here
-------------------------
Two decisions join the ladder because they need the same resolved geometry
and must not re-derive it:

- `plan_class_schedule` groups a softmax round's classes. How many classes
  may be resident depends on the tile count and on the device's threadgroup
  memory, which is what a plan already resolved, so the class grouping
  (`gpu_output_planes.plan_class_batches`) is run from the plan rather than
  from a second guess at the geometry. Its default is the sequential path:
  one class at a time, exactly what the trainer does today, unless a caller
  or `MOJOBOOST_GPU_CLASS_BATCH` asks for more.
- `plan_from_caps` is the bridge for a caller holding only the three
  attributes `gpu_tiling.query_device_caps` reads, which is every GPU call
  site in this repository today. It plans with no compiled specializations
  and no proven device capability, so its geometry is `derive_tiling`'s
  unless a level is requested.

`HistogramPlan.tiling()` projects a plan back to the `HistogramTiling` every
launch site already accepts, and `HistogramPlan.gpu_launches()` reports what
`hybrid_leaf_scheduler.mojo` must charge a node for the launches this plan
will actually issue. Between them a caller can adopt a plan without learning
the ladder, and a caller that does not want one never sees it.

What this module will not decide
--------------------------------
It does not branch on a model string, a part number, or an Apple generation.
`GpuProfile` carries a generation and `apple_gpu_policy.mojo` explains at
length why nothing reads it; the same holds here, for the same reason. Every
input below is either a reported device property or a property of the
workload.

It does not change accumulation. Fixed-point Int32 gradients and hessians,
integer atomics into threadgroup memory, ascending-order reduction: none of
that is a launch decision and none of it is touched. Every specialization
here is a decision about how many threadgroups run, how wide they are, which
rows each one reads, and how many bins its threadgroup allocation holds. The
integers that come out are the same integers, which is the property that
makes the whole ladder testable against the baseline by bit comparison.

It does not choose between the CPU and the GPU. That is `device.mojo` and
`apple_gpu_policy.CrossoverInputs`, and it stays there.

Environment
-----------
`MOJOBOOST_GPU_HIST_SPECIALIZATION`, following the `MOJOBOOST_` contract in
`parallel.mojo`: `off` or `baseline` (the default, and the value of anything
unrecognized), `shape`, `packed`, `batched`. The existing
`MOJOBOOST_GPU_ROW_TILE` and `MOJOBOOST_GPU_BLOCK_THREADS` overrides are
honored at every level, so a benchmark can pin a geometry across the ladder
and compare only the specialization.
"""

from std.os import getenv

from .apple_gpu_policy import (
    API_UNKNOWN,
    APPLE_GEN_UNKNOWN,
    MAX_RESIDENT_BLOCKS_PER_CORE,
    GpuProfile,
    derive_block_threads,
    partial_budget_bytes,
)
from .gpu_histogram_specializations import (
    WINDOW_NOT_A_RUN,
    WINDOW_OK,
    DeviceHistogramCapabilities,
    KernelFeatures,
    PackedLoadWindow,
    bin_capacity_for,
    kernel_shared_bytes,
    plan_packed_window,
    unspecialized_kernel_shared_bytes,
)
from .gpu_output_planes import (
    BatchEligibility,
    ClassBatchPlan,
    env_class_batch,
    plan_class_batches,
)
from .gpu_tiling import (
    BYTES_PER_PARTIAL_CELL,
    STRATEGY_AUTO,
    TARGET_BLOCKS_PER_SM,
    DeviceCaps,
    HistogramTiling,
    clamp_block_threads,
    derive_tiling,
    launches_for_strategy,
    partial_cell_limit_for,
    resolve_tiling,
    strategy_name,
)
from .parallel import _env_int


comptime SPEC_LEVEL_UNSET = -1
comptime SPEC_LEVEL_BASELINE = 0
comptime SPEC_LEVEL_SHAPE = 1
comptime SPEC_LEVEL_PACKED = 2
comptime SPEC_LEVEL_BATCHED = 3

# A frontier below this many leaves is not a batch. The upper bound on a
# batch is not a policy question and is not set here: `gpu_leaf_batching`
# owns it, along with everything else about how a batch is laid out.
comptime MIN_BATCH_LEAVES = 2

# Why a plan stopped where it did.
comptime REASON_AS_REQUESTED = 0
comptime REASON_NOT_REQUESTED = 1
comptime REASON_KERNEL_ABSENT = 2
comptime REASON_DEVICE_UNPROVEN = 3
comptime REASON_ROWS_NOT_A_RUN = 4
comptime REASON_WINDOW_UNUSABLE = 5
comptime REASON_SHARED_MEMORY = 6
comptime REASON_SINGLE_LEAF_FILLS_DEVICE = 7


def level_name(level: Int) -> String:
    if level == SPEC_LEVEL_BASELINE:
        return String("baseline")
    if level == SPEC_LEVEL_SHAPE:
        return String("shape")
    if level == SPEC_LEVEL_PACKED:
        return String("packed")
    if level == SPEC_LEVEL_BATCHED:
        return String("batched")
    return String("unset")


def reason_name(reason: Int) -> String:
    if reason == REASON_AS_REQUESTED:
        return String("as_requested")
    if reason == REASON_NOT_REQUESTED:
        return String("not_requested")
    if reason == REASON_KERNEL_ABSENT:
        return String("kernel_variant_not_compiled")
    if reason == REASON_DEVICE_UNPROVEN:
        return String("device_capability_unproven")
    if reason == REASON_ROWS_NOT_A_RUN:
        return String("rows_not_a_contiguous_run")
    if reason == REASON_WINDOW_UNUSABLE:
        return String("packed_window_unusable")
    if reason == REASON_SHARED_MEMORY:
        return String("threadgroup_memory_too_small")
    if reason == REASON_SINGLE_LEAF_FILLS_DEVICE:
        return String("single_leaf_already_fills_device")
    return String("unknown")


def env_specialization_level() -> Int:
    """`MOJOBOOST_GPU_HIST_SPECIALIZATION` as a level constant.

    Unset, empty, and unrecognized all mean `SPEC_LEVEL_BASELINE`. There is
    no `auto`: a level that turned itself on when it liked the shape would be
    exactly the unmeasured default this module exists to refuse.
    """
    var s = getenv("MOJOBOOST_GPU_HIST_SPECIALIZATION")
    if s == "shape":
        return SPEC_LEVEL_SHAPE
    if s == "packed":
        return SPEC_LEVEL_PACKED
    if s == "batched":
        return SPEC_LEVEL_BATCHED
    return SPEC_LEVEL_BASELINE


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


@fieldwise_init
struct HistogramWorkload(Copyable, Movable):
    """The shape one histogram is being planned for."""

    var dataset_rows: Int
    """Rows in the binned matrix, which is also the column stride: the bin of
    row `r` of feature `f` is at `f * dataset_rows + r`."""

    var node_rows: Int
    """Rows this node owns. Zero is allowed and plans at one row, matching
    `GpuActiveRows.range_tiling`, so an empty node still gets a launchable
    geometry."""

    var n_slots: Int
    """Active features, which is `grid.x`. The full feature count unless
    feature subsampling narrowed it."""

    var n_bins: Int

    var first_row: Int
    """Row id at the start of the node's run. Meaningful only when the run
    flag below is true."""

    var rows_are_contiguous_run: Bool
    """The caller's assertion that this node's slice of the active-row
    permutation holds `first_row` through `first_row + node_rows - 1` in
    ascending order.

    It is a property of the permutation, so it cannot be derived from a
    shape and is never inferred here. False costs only the packed path; a
    wrongly-true answer would make the kernel read the wrong rows, which is
    why the caller has to state it and why `node()` below defaults it off.
    """

    @staticmethod
    def node(
        dataset_rows: Int, node_rows: Int, n_slots: Int, n_bins: Int
    ) -> HistogramWorkload:
        """A node whose rows are not known to be a contiguous run, which is
        every node until the caller proves otherwise."""
        return HistogramWorkload(
            dataset_rows, node_rows, n_slots, n_bins, 0, False
        )


@fieldwise_init
struct HistogramPlan(Copyable, Movable):
    """A resolved histogram launch for one device and one node."""

    var level_requested: Int
    var level_applied: Int
    var reason: Int
    """`REASON_AS_REQUESTED` when the applied level is the requested one, and
    otherwise why the plan stopped lower."""

    var strategy: Int
    var block_threads: Int
    var n_tiles: Int
    var rows_per_tile: Int

    var rows_per_thread: Int
    """`ceil(rows_per_tile / block_threads)`: rows one lane accumulates per
    tile. Reported rather than chosen. It is bounded below by
    `MIN_ROWS_PER_TILE_THREAD_FACTOR` whenever the row bound is what set the
    tile count, and falls below that only on a node with too few rows to feed
    one threadgroup, where nothing can raise it."""

    var bin_capacity: Int
    """The class the bin count falls in (16, 32, 64, 128, or 256)."""

    var shared_bytes_per_block: Int
    """Threadgroup memory one block of the kernel that will actually run
    occupies. The specialized kernels occupy `kernel_shared_bytes(capacity)`;
    the shipping ones occupy the `MAX_BINS` width at every bin count, and
    this reports whichever is true of the build in hand."""

    var resident_blocks_per_core: Int
    var partial_cell_limit: Int
    var partial_cells: Int

    var packed_loads: Bool
    var packed_window: PackedLoadWindow

    var baseline: HistogramTiling
    """What `derive_tiling` would launch for this node, whatever level was
    applied. Carried so a caller can compare, a benchmark can report both,
    and a specialization that misbehaves can be dropped for it at the call
    site without re-deriving anything."""

    def matches_baseline(self) -> Bool:
        """Whether this plan's launch geometry is the shipping one. True at
        `SPEC_LEVEL_BASELINE` by construction, and possibly true above it,
        since a shape-derived geometry may land on the same numbers."""
        return (
            self.strategy == self.baseline.strategy
            and self.block_threads == self.baseline.block_threads
            and self.n_tiles == self.baseline.n_tiles
            and self.rows_per_tile == self.baseline.rows_per_tile
        )

    def tiling(self) -> HistogramTiling:
        """This plan's geometry as the `HistogramTiling` every launch site
        already takes.

        The projection that makes the ladder consumable without teaching a
        call site about it: `gpu_active_rows` and the batched paths accept a
        tiling, and a caller that has planned one hands over this instead of
        re-deriving a second geometry. At `SPEC_LEVEL_BASELINE` it is
        `self.baseline` field for field.
        """
        return HistogramTiling(
            self.strategy,
            self.block_threads,
            self.n_tiles,
            self.rows_per_tile,
            self.partial_cells,
        )

    def gpu_launches(self) raises -> Int:
        """Kernel launches this plan's node costs on the device, from the
        resolved strategy. The number `hybrid_leaf_scheduler.LeafWork` needs
        and must not guess."""
        return launches_for_strategy(self.strategy)


def caps_from_profile(profile: GpuProfile) -> DeviceCaps:
    """The three attributes `gpu_tiling.mojo` plans from, taken off a
    profile. Both layers sanitize their inputs the same way, so this is a
    projection and not a conversion."""
    return DeviceCaps(
        profile.core_count,
        profile.max_threads_per_block,
        profile.max_shared_memory_per_block,
    )


def profile_from_caps(caps: DeviceCaps) -> GpuProfile:
    """A profile from the three attributes a caller already queried.

    The bridge for a call site that holds a `DeviceCaps` and has no API or
    architecture string to hand, which is every call site in this repository
    today. The API is `API_UNKNOWN` and the generation is unknown, which
    costs nothing: no decision in this module or in `apple_gpu_policy.mojo`
    branches on either.

    `unified_memory` is false because a `DeviceCaps` does not say. It feeds
    only the partial-budget fraction, and that fraction applies only to a
    reported memory budget, which is zero here, so the plan is identical
    either way. Once a budget accessor exists (see
    handoffs/apple_a6_policy.md step 3), a call site that knows it is on
    Metal should build the profile with `GpuProfile.from_reported` instead of
    this.

    `synthetic` is false because a `DeviceCaps` is normally a reading. One
    built from `DeviceCaps.fallback()` is not, and this cannot tell the
    difference, so a caller that knows it fell back should construct the
    profile itself rather than mislabel a guess as a reading here.
    """
    return GpuProfile(
        API_UNKNOWN,
        APPLE_GEN_UNKNOWN,
        caps.sm_count,
        caps.max_threads_per_block,
        caps.max_shared_memory_per_block,
        0,
        False,
        False,
    )


def kernel_block_bytes(features: KernelFeatures, capacity: Int) -> Int:
    """Threadgroup memory one block occupies in the build in hand.

    The distinction the bin-capacity specialization is about: without the
    specialized kernels a block occupies the full `MAX_BINS` width whatever
    `n_bins` is, so planning residency from `n_bins * 12` overstates how many
    blocks fit. With them, the narrower footprint is real.
    """
    if features.specialized_bin_kernels:
        return kernel_shared_bytes(capacity)
    return unspecialized_kernel_shared_bytes()


def resident_blocks_per_core(
    profile: GpuProfile, features: KernelFeatures, capacity: Int
) raises -> Int:
    """Threadgroups the policy expects resident on one core: how many really
    fit in the advertised threadgroup memory, capped at
    `MAX_RESIDENT_BLOCKS_PER_CORE` and floored at one.

    Same rule as `apple_gpu_policy.resident_blocks_per_core`, against the
    footprint the compiled kernel has instead of the modeled one.
    """
    var per_block = kernel_block_bytes(features, capacity)
    if per_block < 1:
        raise Error("a histogram block occupies a positive number of bytes")
    var fits = profile.max_shared_memory_per_block // per_block
    if fits > MAX_RESIDENT_BLOCKS_PER_CORE:
        fits = MAX_RESIDENT_BLOCKS_PER_CORE
    if fits < 1:
        fits = 1
    return fits


def baseline_partial_cell_limit(max_partial_cells: Int) -> Int:
    """Partial-histogram cells `derive_tiling` plans against: the flat
    portable ceiling, or an already-allocated buffer when the caller has
    one. `gpu_tiling.partial_cell_limit_for` is that rule; this name is kept
    so a plan reads as reporting the baseline's budget alongside the
    shape-derived one below."""
    return partial_cell_limit_for(max_partial_cells)


def shape_partial_cell_limit(
    profile: GpuProfile, max_partial_cells: Int
) -> Int:
    """Partial-histogram cells `SPEC_LEVEL_SHAPE` plans against: the reported
    device budget's fraction, tightened to an already-allocated buffer when
    there is one.

    `derive_tiling` treats a supplied capacity as replacing its budget; here
    it narrows it, so a plan can never exceed either bound. The two agree
    whenever the allocated buffer was itself sized under the budget, which is
    how it comes to exist. With no reported device memory, which is every
    device today, `partial_budget_bytes` returns the same portable ceiling
    `derive_tiling` uses, so the two limits are equal.
    """
    var cells = partial_budget_bytes(profile) // BYTES_PER_PARTIAL_CELL
    if max_partial_cells > 0 and max_partial_cells < cells:
        cells = max_partial_cells
    if cells < 1:
        cells = 1
    return cells


def _shape_block_threads(profile: GpuProfile, node_rows: Int) -> Int:
    """Threads per threadgroup at `SPEC_LEVEL_SHAPE`: the environment
    override when set, and otherwise the row-bounded Apple rule, which is a
    warp multiple, at least one warp, and never above the device maximum."""
    var requested = _env_int("MOJOBOOST_GPU_BLOCK_THREADS", 0)
    if requested < 1:
        return derive_block_threads(profile, node_rows)
    return clamp_block_threads(requested, profile.max_threads_per_block)


def derive_histogram_plan(
    profile: GpuProfile,
    device: DeviceHistogramCapabilities,
    features: KernelFeatures,
    work: HistogramWorkload,
    requested_strategy: Int = STRATEGY_AUTO,
    requested_level: Int = SPEC_LEVEL_UNSET,
    max_partial_cells: Int = 0,
) raises -> HistogramPlan:
    """Resolve one node's histogram launch.

    Pure host arithmetic: no device is opened, nothing is allocated, and
    nothing is enqueued, so the whole ladder is exercisable on a machine with
    no accelerator.

    `requested_level` of `SPEC_LEVEL_UNSET` consults
    `MOJOBOOST_GPU_HIST_SPECIALIZATION` and then defaults to
    `SPEC_LEVEL_BASELINE`. An explicit level is a ceiling, not a floor: the
    plan may come back lower, with `reason` saying which check failed.

    `max_partial_cells` is an already-allocated partial buffer, as in
    `derive_tiling`. It tightens the memory bound here rather than replacing
    it, so a plan can never ask for more tiles than either the buffer holds
    or the device budget allows.

    Raises on a shape that cannot be planned at all, matching `derive_tiling`
    and `derive_policy`: a nonpositive feature count or bin count, a bin
    count past `MAX_BINS`, or a device whose threadgroup memory cannot hold
    one histogram.
    """
    if work.n_slots < 1:
        raise Error("histogram plan needs at least one active feature")
    if work.dataset_rows < 1:
        raise Error("histogram plan needs a positive dataset row count")
    if work.node_rows < 0:
        raise Error("a node cannot own a negative number of rows")

    # An empty node still needs a launchable geometry, and gets the one a
    # single row would produce. Matching `GpuActiveRows.range_tiling` exactly
    # keeps the plan comparable to the launch it is planning for.
    var rows = work.node_rows
    if rows < 1:
        rows = 1

    var caps = caps_from_profile(profile)
    var baseline = derive_tiling(
        caps,
        rows,
        work.n_slots,
        work.n_bins,
        requested_strategy,
        max_partial_cells,
    )

    var level = requested_level
    if level == SPEC_LEVEL_UNSET:
        level = env_specialization_level()
    if level < SPEC_LEVEL_BASELINE:
        level = SPEC_LEVEL_BASELINE
    if level > SPEC_LEVEL_BATCHED:
        level = SPEC_LEVEL_BATCHED

    # Every rung below either reaches the level asked for or overwrites this
    # with the check that stopped it, so `REASON_AS_REQUESTED` is the state
    # of a plan that has not yet been refused anything.
    var capacity = bin_capacity_for(work.n_bins)
    var applied = SPEC_LEVEL_BASELINE
    var reason = REASON_AS_REQUESTED

    # --- Level 1: geometry from reported properties and the node's shape ---

    if level >= SPEC_LEVEL_SHAPE:
        if (
            kernel_block_bytes(features, capacity)
            > profile.max_shared_memory_per_block
        ):
            reason = REASON_SHARED_MEMORY
        else:
            applied = SPEC_LEVEL_SHAPE
            reason = REASON_AS_REQUESTED

    # Baseline geometry until a level above it replaces it. The residency
    # reported alongside it is the fixed target `derive_tiling` assumes rather
    # than a derived one, because that assumption is what produced this
    # geometry.
    var block_threads = baseline.block_threads
    var n_tiles = baseline.n_tiles
    var rows_per_tile = baseline.rows_per_tile
    var strategy = baseline.strategy
    var partial_cells = baseline.partial_cells
    var resident = TARGET_BLOCKS_PER_SM
    var block_bytes = unspecialized_kernel_shared_bytes()
    var partial_cell_limit = baseline_partial_cell_limit(max_partial_cells)

    if applied >= SPEC_LEVEL_SHAPE:
        partial_cell_limit = shape_partial_cell_limit(
            profile, max_partial_cells
        )
        block_threads = _shape_block_threads(profile, rows)
        resident = resident_blocks_per_core(profile, features, capacity)
        block_bytes = kernel_block_bytes(features, capacity)

        # The same tile arithmetic the baseline runs, over the three bounds
        # this level derives differently, and `gpu_tiling.resolve_tiling` is
        # where that arithmetic lives so the two can never drift:
        #
        # - threadgroups wanted device-wide is `core_count * resident`, with
        #   residency derived from the threadgroup memory a block really
        #   occupies rather than assumed;
        # - a tile amortizes the kernel's bin *capacity*, not the bin count,
        #   which is the same substitution the shared footprint above makes;
        # - the partial buffer is bounded by the reported device budget as
        #   well as by any buffer already allocated.
        #
        # The `MOJOBOOST_GPU_ROW_TILE` and `MOJOBOOST_GPU_HIST_STRATEGY`
        # overrides therefore reach this level too, which they did not when
        # the arithmetic was restated here.
        var target_blocks = profile.core_count * resident
        if target_blocks < 1:
            # A profile reporting no cores is a device that answered nothing;
            # one threadgroup per active feature is the floor, as in
            # `derive_tiling`.
            target_blocks = 1
        var shaped = resolve_tiling(
            rows,
            work.n_slots,
            work.n_bins,
            block_threads,
            capacity,
            target_blocks,
            partial_cell_limit,
            requested_strategy,
        )
        strategy = shaped.strategy
        n_tiles = shaped.n_tiles
        rows_per_tile = shaped.rows_per_tile
        partial_cells = shaped.partial_cells

    # --- Level 2: packed bin loads ---

    # A node that is not a run needs no window arithmetic, and asking for it
    # would validate a `first_row` the caller never claimed was meaningful.
    var window = PackedLoadWindow(False, rows, 0, 0, WINDOW_NOT_A_RUN)
    if work.rows_are_contiguous_run:
        window = plan_packed_window(
            work.dataset_rows, work.first_row, rows, True
        )
    var packed = False
    if level >= SPEC_LEVEL_PACKED and applied >= SPEC_LEVEL_SHAPE:
        if not features.packed_bin_loads:
            reason = REASON_KERNEL_ABSENT
        elif not device.wide_byte_loads:
            reason = REASON_DEVICE_UNPROVEN
        elif not work.rows_are_contiguous_run:
            reason = REASON_ROWS_NOT_A_RUN
        elif window.reason != WINDOW_OK:
            reason = REASON_WINDOW_UNUSABLE
        else:
            packed = True
            applied = SPEC_LEVEL_PACKED
            reason = REASON_AS_REQUESTED

    # --- Level 3: batched small leaves ---

    # `level_applied` is a contiguous ladder position, so this level is only
    # reached when the packed one was. That is a statement about this node,
    # and batching is a decision across nodes, which is why
    # `batching_declined_reason` gates on `level_requested` and
    # `KernelFeatures` instead: a frontier must not lose its batching because
    # one node's rows happened not to be contiguous. The launch a batched
    # frontier gets is `gpu_leaf_batching.plan_batch`'s, not this plan's.
    if level >= SPEC_LEVEL_BATCHED and applied >= SPEC_LEVEL_PACKED:
        if not features.batched_leaf_kernel:
            reason = REASON_KERNEL_ABSENT
        else:
            applied = SPEC_LEVEL_BATCHED
            reason = REASON_AS_REQUESTED

    return HistogramPlan(
        level,
        applied,
        reason,
        strategy,
        block_threads,
        n_tiles,
        rows_per_tile,
        _ceil_div(rows_per_tile, block_threads),
        capacity,
        block_bytes,
        resident,
        partial_cell_limit,
        partial_cells,
        packed,
        window^,
        baseline^,
    )


def batching_declined_reason(
    plan: HistogramPlan,
    profile: GpuProfile,
    features: KernelFeatures,
    row_counts: List[Int],
    n_slots: Int,
) raises -> Int:
    """Why a frontier will not be batched, or `REASON_AS_REQUESTED` when it
    wants to be.

    This is a decision, not a plan. It answers the one question that belongs
    to a launch policy, which is whether this frontier leaves the device
    underfilled, and it deliberately does not group the leaves, hand out
    tiles, or size a partial buffer. `gpu_leaf_batching.plan_batch` does all
    three, from the same `DeviceCaps` and the same amortization constants,
    and a second planner here that disagreed with it by a tile would be worse
    than no planner at all.

    `row_counts` is the frontier in launch order. A count below one is read
    as one, matching `GpuActiveRows.range_tiling`.
    """
    if n_slots < 1:
        raise Error("batching needs at least one active feature")
    if plan.level_requested < SPEC_LEVEL_BATCHED:
        return REASON_NOT_REQUESTED
    if not features.batched_leaf_kernel:
        return REASON_KERNEL_ABSENT
    if len(row_counts) < MIN_BATCH_LEAVES:
        return REASON_SINGLE_LEAF_FILLS_DEVICE

    # One leaf too small to fill the device on its own is the whole case for
    # batching. A frontier where every leaf already fills it has nothing to
    # gain and would only pay the packed tile axis's binary search.
    var target = profile.core_count * plan.resident_blocks_per_core
    for i in range(len(row_counts)):
        var rows = row_counts[i]
        if rows < 1:
            rows = 1
        var tiles = _ceil_div(rows, plan.rows_per_tile)
        if n_slots * tiles < target:
            return REASON_AS_REQUESTED
    return REASON_SINGLE_LEAF_FILLS_DEVICE


def plan_from_caps(
    caps: DeviceCaps,
    work: HistogramWorkload,
    requested_strategy: Int = STRATEGY_AUTO,
    requested_level: Int = SPEC_LEVEL_UNSET,
    max_partial_cells: Int = 0,
) raises -> HistogramPlan:
    """One node's plan for a caller that queried a `DeviceCaps` and nothing
    else, which is every GPU call site in this repository today.

    The bridge, and deliberately the conservative one: with no compiled
    specialized kernels (`KernelFeatures.none()`) and no device that has
    demonstrated wide byte loads (`DeviceHistogramCapabilities.portable()`),
    the packed and batched rungs cannot be reached whatever is requested, and
    the plan's geometry at the default level is `derive_tiling`'s field for
    field. A caller that has built the specialized kernels, or that holds a
    real `GpuProfile` with a memory budget, calls `derive_histogram_plan`
    directly with what it knows rather than through this.
    """
    return derive_histogram_plan(
        profile_from_caps(caps),
        DeviceHistogramCapabilities.portable(),
        KernelFeatures.none(),
        work,
        requested_strategy,
        requested_level,
        max_partial_cells,
    )


@fieldwise_init
struct ClassSchedule(Copyable, Movable):
    """How one softmax round's classes are grouped, and the launch geometry
    each group runs at.

    The join between the two halves of the multiclass decision, which are
    otherwise planned by modules that cannot see each other:
    `gpu_output_planes.plan_class_batches` needs a tile count and the
    device's threadgroup memory before it can say how many classes fit, and
    those are exactly what a histogram plan resolves. Holding both together
    is what lets a caller ask the one question it actually has -- how many
    classes may I make resident, and how many of them share a bin read --
    without planning a geometry twice.

    It groups classes; it does not reorder them. The batches are contiguous
    ascending runs and slot `s` of batch `b` is class
    `b * batch_size + s` (`ClassBatchPlan.class_at`), so a round still grows
    one tree per class and still stores it at `round * n_classes + k`.
    """

    var plan: HistogramPlan
    var batches: ClassBatchPlan
    var eligibility: BatchEligibility

    def shared_group_size(self) -> Int:
        """Classes one threadgroup serves from a single bin read.

        `ClassBatchPlan.per_block`, which is 1 whenever the classes do not
        share both their rows and their features, and otherwise the number
        that fits in threadgroup memory, capped at the batched kernel's
        static `SHARED_CLASS_CAP`. A batch wider than this is histogrammed in
        `ceil(batch / per_block)` shared launches, which is what
        `ClassBatchPlan.bin_passes_per_round` counts.
        """
        return self.batches.per_block

    def shares_bin_reads(self) -> Bool:
        """Whether any bin read is shared at all. False for a sequential
        plan and for any node below a round's root."""
        return (
            self.eligibility.bin_reads_shared() and self.batches.per_block > 1
        )

    def is_sequential(self) -> Bool:
        """True when the schedule degenerates to the shipped one-class-at-a-
        time loop, which is the default: `plan_class_schedule` asks for a
        batch of one unless a caller or `MOJOBOOST_GPU_CLASS_BATCH` asks for
        more."""
        return self.batches.is_sequential()

    def bin_passes_per_round(self) -> Int:
        """Times a round reads the binned matrix at this level.
        `n_classes` sequentially, fewer only where bin reads are shared."""
        return self.batches.bin_passes_per_round()


def plan_class_schedule(
    profile: GpuProfile,
    device: DeviceHistogramCapabilities,
    features: KernelFeatures,
    work: HistogramWorkload,
    n_features: Int,
    n_classes: Int,
    eligibility: BatchEligibility,
    requested_strategy: Int = STRATEGY_AUTO,
    requested_level: Int = SPEC_LEVEL_UNSET,
    max_partial_cells: Int = 0,
    budget_bytes: Int = 0,
    requested_batch: Int = 0,
) raises -> ClassSchedule:
    """Plan one softmax round's class grouping and its launch geometry.

    Pure host arithmetic, like everything else here: no device is opened and
    nothing is allocated, so a caller can ask what a round would cost before
    committing to it.

    `work.n_slots` is the active feature count, which is `grid.x` and what
    the geometry is derived from; `n_features` is the dataset's full feature
    count, which is what the batch's output planes are allocated at and so
    what the memory budget must cover. They differ under feature subsampling
    and the distinction is load-bearing, which is why both are arguments.

    **The default is the sequential path.** A `requested_batch` of zero
    consults `MOJOBOOST_GPU_CLASS_BATCH` and then asks for a batch of one,
    rather than for the widest batch the budget admits. Batching changes no
    number a round produces -- the fixed-point scales are per class and
    integer accumulation is associative, which is what
    `gpu_multiclass_batch.mojo` is built around -- but it does change the
    memory a fit holds and the order work reaches the queue in, and nothing
    in this repository has measured either against the sequential loop. So it
    is opt-in, on the same grounds as the specialization ladder above.

    An explicitly requested batch that does not fit the budget raises rather
    than silently shrinking, because a benchmark that asked for a geometry
    must not be handed a different one; that refusal is
    `plan_class_batches`'s and is left to it.
    """
    if n_features < work.n_slots:
        raise Error(
            "the active feature count cannot exceed the dataset's feature"
            " count"
        )
    var plan = derive_histogram_plan(
        profile,
        device,
        features,
        work,
        requested_strategy,
        requested_level,
        max_partial_cells,
    )
    var requested = requested_batch
    if requested <= 0:
        requested = env_class_batch()
    if requested <= 0:
        # Default off. One class at a time is what the trainer does today.
        requested = 1
    var batches = plan_class_batches(
        n_classes,
        work.dataset_rows,
        n_features,
        work.n_bins,
        plan.n_tiles,
        profile.max_shared_memory_per_block,
        eligibility,
        budget_bytes,
        requested,
    )
    return ClassSchedule(plan^, batches^, eligibility.copy())


def _yes_no(value: Bool) -> String:
    if value:
        return String("yes")
    return String("no")


def describe_plan(plan: HistogramPlan) -> String:
    """One line for benchmark output and bug reports: what was asked for,
    what was applied, why, and the geometry that resulted."""
    return String(
        "level=",
        level_name(plan.level_applied),
        "/",
        level_name(plan.level_requested),
        " why=",
        reason_name(plan.reason),
        " strategy=",
        strategy_name(plan.strategy),
        " threads=",
        plan.block_threads,
        " tiles=",
        plan.n_tiles,
        " rows_per_tile=",
        plan.rows_per_tile,
        " rows_per_thread=",
        plan.rows_per_thread,
        " bin_capacity=",
        plan.bin_capacity,
        " shared=",
        plan.shared_bytes_per_block,
        " resident=",
        plan.resident_blocks_per_core,
        " partial=",
        plan.partial_cells,
        "/",
        plan.partial_cell_limit,
        " packed=",
        _yes_no(plan.packed_loads),
        " baseline=",
        _yes_no(plan.matches_baseline()),
    )


def describe_schedule(schedule: ClassSchedule) -> String:
    """One line for a trace: how the round's classes were grouped, whether
    any bin read is shared, and what a group's launch looks like."""
    return String(
        "classes=",
        schedule.batches.n_classes,
        " batch=",
        schedule.batches.batch_size,
        " per_block=",
        schedule.batches.per_block,
        " batches=",
        schedule.batches.n_batches(),
        " bin_passes=",
        schedule.bin_passes_per_round(),
        " shared_rows=",
        _yes_no(schedule.batches.shared_rows),
        " shared_counts=",
        _yes_no(schedule.batches.shared_counts),
        " shares_bin_reads=",
        _yes_no(schedule.shares_bin_reads()),
        " bytes=",
        schedule.batches.bytes_per_batch,
        " | ",
        describe_plan(schedule.plan),
    )
