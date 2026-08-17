"""GPU histogram accumulation and row partitioning.

Same conceptual inputs and outputs as the CPU builders in `histogram.mojo`,
different algorithm for the hardware. The dataset stays device-resident for
a whole training session: the binned matrix is uploaded once at construction,
gradients and hessians once per boosting round, and a device-resident
permutation of the row indices (see gpu_active_rows.mojo) gives every live
leaf a contiguous half-open range of rows, so tree growth never copies
row-index lists across the PCIe/unified-memory boundary.

Histograms build with a 2D grid: `grid.x` is the active feature, `grid.y` a
tile of rows, so a device gets `n_active * n_tiles` threadgroups of parallel
work instead of just `n_active`. Every threadgroup accumulates a partial
histogram for its (feature, row-tile) in shared memory, reading only the
node's own compacted row range. The launch geometry (threads per group,
tiles per feature) is derived per node from device capabilities and the
node's own row count rather than fixed at compile time; see
`gpu_tiling.mojo` and `gpu_active_rows.mojo`.

Two strategies combine those partials, both selectable and both tested:

- `STRATEGY_TILED` writes each partial to its own slot of a global buffer
  and sums the slots in a second kernel, in ascending tile order. No global
  atomics and no contention on hot bins, at the cost of one extra kernel
  launch and a partial buffer bounded by a memory budget.
- `STRATEGY_ATOMIC` folds each partial into the output with global integer
  atomics. This is the original implementation, preserved as the fallback
  for hardware where the tiled path is not yet validated, and chosen
  automatically when the partial buffer would not fit the budget. The
  kernel is unchanged; only its launch geometry is now device-derived
  instead of a fixed constant, which cannot affect an atomic accumulation.

The two return bit-identical histograms, which the tests assert directly:
accumulation is exact fixed-point Int32 throughout and integer addition is
associative, so how the partials combine cannot change the result.

Gradient, hessian, and count planes share one Int32 device buffer laid out as
`[grad | hess | count]`, so a whole node's histogram costs one kernel launch
and one device-to-host copy instead of three of each. Per-node launch and
synchronization overhead dominates on small nodes, so this is the difference
between three host synchronizations per tree node and one. The partial buffer
repeats that layout once per row tile.

Within and across blocks, gradients and hessians accumulate as fixed-point
Int32 via integer atomics rather than float atomics. Metal has no float
atomic add, and integer accumulation is portable (CUDA/ROCm/Metal) and
order-independent, so GPU histograms are bit-deterministic run to run. The
fixed-point scale is chosen on the host from the global gradient/hessian
magnitude sums, which bound every partial sum (any leaf's rows are a subset
of all rows), so scaled accumulation cannot overflow.

The kernels use only primitives all three backends provide: shared memory,
`barrier()`, integer atomics on shared memory, and plain global loads and
stores. No warp shuffles, no float atomics, no vendor intrinsics, and no
per-architecture code paths. That portable baseline is the same source on
Metal, CUDA, and HIP; only the tiling numbers differ per device.

Gradients are carried as Float32 on the device: Apple GPUs have no Float64.
Results convert back to the Float64 `Histogram` on download; agreement with
the CPU builder is to Float32 precision, not bit-exact. Counts are exact.

Row bagging rides on the same permutation: `begin_tree` seeds the root
range with the bag's rows in the caller's order, and a row outside the bag
is simply not inside any range, so no kernel ever iterates it. A bagged
tree costs one n_rows Int32 staged copy; an unbagged one seeds the
identity permutation with a kernel and costs no transfer at all.

Every node's histogram is built by scanning exactly that node's rows, so a
tree costs on the order of `sum over built nodes of node_rows * n_features`
bin reads, the same asymptotic cost as the CPU builder's row-index lists,
instead of the `nodes_built * n_rows * n_features` the pre-compaction
filtering kernels paid. The partition that maintains the ranges is a
stable four-launch scan/scatter that is bit-deterministic and keeps each
node's rows in the exact order the CPU grower's row lists hold them.

Transfers stage through pinned host buffers and one-way copies rather than
`map_to_host`, which copies in both directions on every use. That also makes
the phases separately timeable, which is what `bench/bench_histogram.mojo`
reports: `stage_gradients` (Float64 to Float32 conversion), `upload_staged`
(host to device), `enqueue_leaf` (kernels), `download_raw` (device to host),
and `histogram_from_host` (fixed-point to Float64 conversion).

One-way is the whole claim, and in particular it is not a claim that the
upload is asynchronous. On Metal `enqueue_copy` is a synchronous full-queue
drain in **both** directions: it commits an empty command buffer, waits for
it, and then memcpys. That was **measured** by disassembling the shipped MAX
Metal runtime and is written out with its consequences in
`docs/GPU_PORTABILITY.md` section 6.1. So `upload_staged` is a full-queue
drain exactly as `download_raw` is, the `stage_gradients` and `upload_staged`
split above is a split between host conversion work and a drain rather than
between two enqueues, and any per-round staging (gradients, a bag mask, a
GOSS row vector, a scale word) is a per-round ordering point that has to be
counted as one. What the staged route still buys over `map_to_host` is the
second direction's bytes and a separately timeable phase, which is why it
stays.

A drain is not a wait, and the difference is the whole of section 6.1.1,
withdrawn 2026-08-16. `download_raw` is a **round trip**: the host reads a
histogram and then decides what to enqueue next, so it blocks on unfinished
device work and costs whatever that work had left. `upload_staged` is not;
nothing is queued behind it and nothing reads a device answer. Draining a
queue that holds nothing costs nothing, and the **measured** null is on the
record: thirteen copies removed per tree elsewhere on this plane bought 0.016
seconds at 1,000,000 x 50 against a registered prediction of 0.64
(`bench/results/session3_2026-08-16/RESULTS.md`). Count round trips here to
predict time; count copies to predict portability risk and staleness.

Counting the ordinary round's drains under that rule gives three, all of them
on the `upload_gradients` path: the `synchronize` in `stage_gradients`, and
one copy per derivative plane in `upload_staged`. The gradient and hessian
planes now share one device allocation and one pinned host arena laid out as
`[grad | hess]`, so `upload_staged` moves both with a single copy and the
round has two ordering points instead of three; `grad_dev` and `hess_dev` are
`create_sub_buffer` windows onto that allocation, so nothing that consumes
them -- kernels, the device objectives, the multiclass scatter -- sees a
change. **Three to two is one fewer copy per round, not a predicted saving**;
it is one fewer staging lifetime and one fewer place a plane can go stale.
The remaining `synchronize` stays on purpose and `stage_gradients` says why.
`stage_from_device`, which only the replica tests reach, adds two copies and
a synchronize of its own on top.

Which transfer route the binned matrix takes is resolved once per builder
through `unified_memory_policy.resolve_from_env`, so `MOJOTREES_GPU_TRANSFER`
is answered by the one route policy in the package rather than ignored here.
Every route but the staged copy is structurally blocked or unproven today, so
the resolver returns `ROUTE_COPY_STAGED` unless a caller asked for something
else, and an explicit request it cannot honor raises there rather than being
quietly downgraded. The upload below is that route; a second route would be a
second branch here, not a second builder.

Several leaves in one launch
----------------------------
`build_leaf` builds one node per launch, which is right for the root and
wastes the device on the tail of a tree (see gpu_leaf_batching.mojo).
`build_leaves` is the multi-leaf entry: it turns the caller's node ids into
`LeafWorkItem`s off the device-resident row ranges `GpuActiveRows` already
maintains, plans one packed launch with `gpu_leaf_batching.plan_batch`,
launches it through `enqueue_frontier_batch` (which checks every item's window
against the tree's live active prefix), and decodes each leaf's slot with the
same fixed-point arithmetic `histogram_from_host` uses. No second row model
and no second decoder: the ranges are the ones the partition kernel wrote, so
a batched histogram covers exactly the rows the single-leaf build would have
covered. The slot stamp, the per-item scale table, and the multi-slot download
all come from `gpu_leaf_batching` rather than being written again here.

The constant-hessian plane
--------------------------
Four of the built-in objectives put the same hessian into every row when the
fit carries no sample weights, and `set_constant_hessian` is how a caller
declares that. The range histogram kernels in `gpu_active_rows.mojo` then stop
accumulating that plane and reconstruct it as `hq_const * count`, which is the
exact Int32 they would have accumulated; that kernel's docstring carries the
argument and this module's `histogram_from_host` says why the download is
unaffected.

It reaches the single-leaf and resident-frontier builds, which are the ones
that run through `enqueue_leaf`. It does **not** reach the batched multi-leaf
path: those kernels live in `gpu_leaf_batching.mojo`, are never handed the
flag, and therefore keep accumulating three planes. That is a difference in
work and not in output, since both paths write the same three planes into the
same layout, and a build is free to take either.

Batching is off unless it is asked for and the launch policy agrees, and
both halves of that are existing code rather than a new switch.
`MOJOTREES_GPU_HIST_SPECIALIZATION=batched` (`apple_histogram_policy`) is the
request, and `apple_histogram_policy.batching_declined_reason` is the
decision, which declines a frontier whose every leaf already fills the device
and a batch of fewer than two leaves. Declined or unrequested, `build_leaves`
falls back to a `build_leaf` per node, which is byte for byte the path that
shipped. Nothing here defaults to a claim no benchmark has made.
"""

from std.math import isfinite
from std.memory import unsafe_memcpy
from std.os import getenv
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from .apple_gpu_policy import API_METAL, parse_api
from .apple_histogram_policy import (
    REASON_AS_REQUESTED,
    SPEC_LEVEL_BATCHED,
    ClassSchedule,
    HistogramPlan,
    HistogramWorkload,
    batching_declined_reason,
    derive_histogram_plan,
    env_specialization_level,
    plan_class_schedule,
    profile_from_caps,
)
from .binning import BinnedMatrix
from .categorical import CatBitset, CategoricalSpec, cat_empty
from .gpu_active_rows import MAX_ROWS, GpuActiveRows, LeafRange, RowRouting
from .gpu_binned_layout import check_layout_support
from .gpu_frontier import LeafWorkItem
from .gpu_tree_tables import DeviceTreeTables, TreeTablesSnapshot
from .gpu_multiclass_batch import GpuClassBatch
from .gpu_output_planes import BatchEligibility
from .gpu_histogram_specializations import (
    DeviceHistogramCapabilities,
    KernelFeatures,
)
from .gpu_leaf_batching import (
    DEFAULT_MAX_ITEMS,
    GpuLeafBatcher,
    batch_const_hessian_forward_requested,
    oblivious_subtract_requested,
    plan_batch,
    slots_for_budget,
    subtraction_stamp,
    uniform_scales,
)
from .gpu_portability import (
    BackendContract,
    contract_from_profile,
    require_bins_supported,
    require_device_can_host_kernels,
    require_histogram_launchable,
)
from .gpu_objectives_native import (
    DEFAULT_MAX_NODES,
    SCALE_WINDOW_MAX,
    GpuObjectiveState,
    GpuRankingState,
)
from .ranking import RankGroups
from .ranking_pairwise import check_rank_kind, describe_rank_kind
from .parallel import _env_int, dispatch_rows
from .quantized_gradient import (
    DEFAULT_SCALE_SHAPE,
    FIXED_ONE,
    SCALE_SHAPE_ARBITRARY,
    SCALE_SHAPE_POW2,
    describe_scale_shape,
    fixed_point_scale_shaped,
    magnitude_sum,
)
from .unified_memory_policy import (
    ROLE_BINS,
    ROUTE_COPY_STAGED,
    describe_decision,
    resolve_from_env,
)
from .gpu_tiling import (
    STRATEGY_AUTO,
    DeviceCaps,
    HistogramTiling,
    derive_tiling,
    free_feature_group,
    query_device_caps,
)
from .gpu_runtime import (
    PHASE_ALLOC,
    ROLE_TRAIN,
    SLOT_BINS,
    SLOT_FEAT,
    SLOT_GRAD,
    SLOT_HESS,
    SLOT_HOST_OUT,
    SLOT_OUT,
    SLOT_PART,
    SLOT_STAGE,
    GpuSession,
    MatrixIdentity,
    bins_fingerprint,
)
from .histogram import (
    Histogram,
    SIMD_LANES,
    _zeroed_f64,
    _zeroed_int,
    build_histogram_subset_replica_into,
)

# The bin ceiling the kernels size threadgroup planes by is `binning.MAX_BINS`
# (imported by gpu_histogram_specializations, which owns the kernels), and the
# Int32 row-index limit is `gpu_active_rows.MAX_ROWS`, imported above.

# Histogram slots the batched path may hold at once, and the byte budget it
# is capped against. A slot is a full-width `3 * n_features * n_bins` Int32
# histogram, so the budget is what keeps a wide dataset from asking for
# hundreds of megabytes to hold a frontier; see
# `gpu_leaf_batching.slots_for_budget`.
comptime DEFAULT_BATCH_SLOTS = 4
comptime MAX_BATCH_SLOTS = 64
comptime BATCH_POOL_BUDGET_BYTES = 64 * 1024 * 1024

# The byte budget the *resident frontier* is sized against, which is a
# different question from the batch pool's and so a different number. A batch
# pool holds as many leaves as one launch is worth widening to, and four is a
# guess about launch width; a resident frontier holds one slot per live leaf
# for the whole tree, so its depth is `num_leaves` and nothing smaller works
# at all (see `open_resident`). Four times the batch budget is what a default
# 31-leaf frontier costs up to about 850 features at 256 bins, which covers
# the dense workloads the GPU trainer is benchmarked on; wider datasets fall
# back to the incremental device search rather than allocating past it.
comptime RESIDENT_POOL_BUDGET_BYTES = 256 * 1024 * 1024


def env_batch_slots() -> Int:
    """`MOJOTREES_GPU_BATCH_SLOTS`, the histogram slot pool depth, clamped to
    a usable range. Only read when batching was requested at all, so the
    default path never consults it."""
    var n = _env_int("MOJOTREES_GPU_BATCH_SLOTS", DEFAULT_BATCH_SLOTS)
    if n < 2:
        return 2
    if n > MAX_BATCH_SLOTS:
        return MAX_BATCH_SLOTS
    return n


def build_kernel_features() -> KernelFeatures:
    """Which specialized histogram kernel variants this build compiles in.

    `batched_leaf_kernel` is true because `build_leaves` below instantiates
    `gpu_leaf_batching`'s kernels, so on any build that links this module
    they exist. That is a fact about compilation and nothing more: no
    benchmark and no hardware run has compared a batched launch against the
    single-leaf one here, which is exactly why the level that selects it
    (`SPEC_LEVEL_BATCHED`) has to be asked for and never resolves from
    `auto`. `packed_bin_loads` does not exist: the row loop reads one bin per
    load.

    `specialized_bin_kernels` stays false, and this is now a deliberate
    understatement rather than a description. The range histogram family in
    `gpu_active_rows.mojo` *is* instantiated per bin capacity, so the flag is
    literally true of the build. It is left false because of what consumes it:
    `gpu_portability.kernel_shared_request` turns it into a threadgroup
    footprint of one capacity-sized block, and that expression carries no
    feature-group width, so at any group past 1 it would report a fraction of
    what the launch really allocates and the geometry gate would be checking a
    number that does not bound the launch. Reporting the smaller, more
    flattering figure to a gate is the wrong direction to be wrong in.
    Flipping this belongs with a `kernel_shared_request` that takes the group,
    which is `gpu_portability.mojo`'s to change. Until then the real bound is
    enforced where the group is known: `GpuActiveRows.set_feature_group`
    refuses a width whose `gpu_tiling.histogram_shared_bytes` exceeds what the
    device reported, and the constructor clamps an environment request to the
    widest rung that fits.
    """
    return KernelFeatures(False, False, True)


def _fixed_scale(
    values: List[Float64], shape: Int = DEFAULT_SCALE_SHAPE
) raises -> Float32:
    """Fixed-point scale derived from a host-side value list.

    `shape` selects the scale arm and defaults to the package default, which
    is `SCALE_SHAPE_POW2`: `2^30 / sum|v|` rounded *down* to a power of two.
    `quantized_gradient.fixed_point_scale_pow2` is the rule and carries the
    whole argument for it -- the exactness it buys, the overflow bound it
    tightens rather than loosens, and the up-to-one-bit of lattice resolution
    it costs. Nothing about the rule lives here; this function is a magnitude
    sum and a forward.
    """
    return fixed_point_scale_shaped(magnitude_sum(values), shape)


def _check_window_bound(total: Float64, scale: Float64, plane: String) raises:
    """The fixed-point overflow inequality, applied to a round that has
    already run.

    `sum_i |v_i| * s <= 2^30` is the whole of the argument in
    `quantized_gradient.fixed_point_scale_pow2`: the exact scaled sum of every
    row's magnitude is at most 2^30, any node holds a subset of the rows, and
    deterministic rounding adds at most 1/2 per row, so no Int32 cell exceeds
    `2^30 + n/2`. A round that derives its own scale satisfies it by
    construction. A round that reuses an earlier round's scale
    (`GpuHistogramBuilder.set_scale_refresh`) satisfies it only if its
    magnitudes did not outgrow the ones the scale came from, and this is where
    that is established rather than assumed.

    Raising is the right answer and a quiet clamp is not: by the time this
    runs the histograms of the offending round have already been accumulated,
    so there is nothing left to fix and the only choice is between saying so
    and not saying so. The message carries both numbers so the caller can see
    how far past the bound it went and pick a `headroom_bits` that prevents
    it next time.

    The comparison is `>` and not `>=` because the bound is inclusive: a
    freshly derived power-of-two scale can land exactly on `2^30` and does so
    whenever `2^30 / T` is itself a power of two.

    THE SLACK, AND WHY IT IS NOT A FUDGE
    ------------------------------------
    The comparison is against `2^30 (1 + 2^-24)` rather than `2^30`, and
    without that this check would fire on rounds that are perfectly safe.
    `fixed_point_scale_pow2` derives `s` as the largest power of two at most
    `fl(2^30 / T)`, and `fl` rounds to nearest, so the *derivation itself*
    admits an exact scaled total of `2^30 (1 + 2^-53)` -- one ulp of the
    quotient above the round number. `total * scale` is an exact product here
    (a Float64 times a power of two is exact), so that ulp survives into this
    comparison rather than being rounded away by it, and a round that derived
    its own scale would be reported as having outgrown it.

    `2^-24` and not `2^-53` because `2^-24` is the slack the *previously
    shipped* scale rule admitted: `SCALE_SHAPE_ARBITRARY` narrows through
    Float32 and rounds to nearest, so it admits `2^30 (1 + 2^-24)`, and the
    repository's overflow proof (`docs/GPU_PORTABILITY.md`,
    `test_gpu_portability.test_fixed_point_accumulation_cannot_overflow_int32`)
    is stated to survive it. So this tolerance is not a number chosen to make
    a test pass; it is the bound the package already lives with, and it
    covers both scale shapes with one constant. In units it is 64 lattice
    units above 2^30 against an Int32 headroom of about 2^31 - 2^30, which no
    accumulation can see. **Derived bound**, not measured.
    """
    var scaled = total * scale
    if scaled > FIXED_ONE * (1.0 + 1.0 / Float64(1 << 24)):
        raise Error(
            String(
                "a round reusing an earlier fixed-point scale outgrew it: the",
                " ",
                plane,
                " magnitude sum ",
                String(total),
                " times the scale in force ",
                String(scale),
                " is ",
                String(scaled),
                ", past the 2^30 bound the Int32 histogram rests on."
                " Lower the scale refresh cadence or raise the headroom"
                " (GpuHistogramBuilder.set_scale_refresh).",
            )
        )


struct GpuHistogramBuilder(Movable):
    """Device-resident histogram builder and row partitioner for one binned
    dataset. Construct once per training session, `upload_gradients` once per
    boosting round, `begin_tree` + `build_leaf`/`apply_split` per tree."""

    var ctx: DeviceContext
    var bins_dev: DeviceBuffer[DType.uint8]
    # The device-resident active-row permutation and its per-leaf ranges;
    # every histogram and every partition works through it.
    var rows: GpuActiveRows
    # One `2 * n_rows` Float32 device allocation holding this round's
    # derivatives as `[grad | hess]`, adjacent and in that order, and the two
    # windows onto it that every kernel and every consumer still sees.
    #
    # Why one allocation. `enqueue_copy` is a full-queue drain on Metal
    # whatever the byte count is (`docs/GPU_PORTABILITY.md` section 6.1,
    # **measured** by disassembly), so two plane copies per round are two
    # drains and one is one. The two planes are the same dtype and the same
    # length, so concatenating them is a plain adjacency: no bitcast, no
    # conversion, no padding, and the bytes each plane's window receives are
    # exactly the bytes its separate buffer used to receive.
    #
    # Two drains to one is not two waits to one. Section 6.1.1, withdrawn
    # 2026-08-16, took back the step that priced a drain: nothing is queued
    # behind these uploads, and draining an empty queue costs nothing. So the
    # argument for one allocation is one staging lifetime instead of two and
    # one ordering point instead of two, plus a `[grad | hess]` pair that
    # cannot arrive half fresh. No time may be predicted from it.
    #
    # Why windows rather than an offset pointer. `grad_dev` and `hess_dev`
    # leave this file as `DeviceBuffer`s (`gpu_objectives_native`'s fills and
    # reductions, `gpu_multiclass_batch.scatter_slot`) and as raw pointers
    # (the range histogram kernels, `gpu_categorical`). A `create_sub_buffer`
    # window is a `DeviceBuffer`, so both kinds of call site keep the exact
    # signature and the exact argument they had, and no file outside this one
    # changes. An offset pointer would have served the second kind only.
    # Offsets are counted in elements, not bytes.
    #
    # Why the parent is a field. Holding `gh_dev` for the life of the builder
    # makes the parent allocation's lifetime the builder's, so nothing here
    # depends on whether a window keeps its parent alive on its own.
    var gh_dev: DeviceBuffer[DType.float32]
    var grad_dev: DeviceBuffer[DType.float32]
    var hess_dev: DeviceBuffer[DType.float32]
    # The feature ids builds accumulate, device side; the first `n_active`
    # entries are live (see `set_features`).
    var feat_dev: DeviceBuffer[DType.int32]
    # [grad | hess | count], each n_features * n_bins entries.
    var out_dev: DeviceBuffer[DType.int32]
    # The same three planes once per row tile, indexed by active-feature
    # slot. One element when the resolved strategy needs no partial buffer.
    var part_dev: DeviceBuffer[DType.int32]
    var part_capacity: Int
    # The host side of the same adjacency: `2 * n_rows` pinned Float32 with
    # the gradient plane at 0 and the hessian plane at `n_rows`, so one copy
    # of the whole arena is one copy of both planes and the fused upload has
    # a contiguous source. Nothing outside this file reads it; the two halves
    # are taken as `unsafe_ptr()` and `unsafe_ptr().unsafe_offset(n_rows)` at
    # the four sites that need them.
    var stage_gh: HostBuffer[DType.float32]
    # Whether `upload_staged` issues one copy of the whole `[grad | hess]`
    # arena or the two per-plane copies that shipped. See
    # `set_fused_gradient_upload`.
    var fused_upload: Bool
    var host_out: HostBuffer[DType.int32]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    # The dataset's per-feature missing bins, so the grower can route missing
    # rows without holding on to the BinnedMatrix.
    var missing_bin: List[Int]
    var cats: CategoricalSpec
    var active: List[Int]
    var caps: DeviceCaps
    # Stable strings reported by the context when the builder opens.  Split
    # strategy policy consumes them without reopening or re-querying a
    # device for every tree.
    var device_api: String
    var device_arch: String
    # What this backend is required to provide, derived once from `caps` when
    # the builder opens. Every launch geometry this builder resolves is
    # checked against it by `require_histogram_launchable`: at construction,
    # whenever `set_features` narrows the grid, and once more before the
    # batched pool is allocated. See gpu_portability.mojo, which is the one
    # place that knows what a backend must have for these kernels to run.
    var contract: BackendContract
    var tiling: HistogramTiling
    var g_scale: Float64
    var h_scale: Float64
    # Which shape this builder's fixed-point scales take:
    # `SCALE_SHAPE_POW2` (the default and the accurate arm) or
    # `SCALE_SHAPE_ARBITRARY` (what shipped). A *numeric* option, unlike
    # every other arm on this builder, so it is the one that changes
    # histogram bits; `set_scale_shape` says what that means for a fixture.
    var fixed_scale_shape: Int
    # --- The scale window: how often the host waits for a magnitude sum ----
    #
    # `set_scale_refresh` is the whole argument. In one line: the round's
    # fixed-point scale comes from a device reduction whose answer the host
    # has to hold before it can enqueue a single histogram, so it is a round
    # trip and not a drain, and it is one of exactly two per round on the
    # default arm. These four fields are what let a fit pay it every `N`
    # rounds instead of every round.
    var scale_refresh: Int
    """`N`. 1 is the shipped cadence (fold every round). 0 selects the
    unwindowed `magnitude_sums` call, expression for expression, which is the
    reference arm the identity test compares against."""
    var scale_headroom: Int
    """`H`, in bits. The derived scale is divided by `2^H`, which buys the
    reused rounds room to grow their magnitudes by `2^H` before the Int32
    overflow bound is at risk, and costs `H` bits of lattice resolution."""
    var scale_ref_total_g: Float64
    var scale_ref_total_h: Float64
    """The magnitude sums the scale currently in force was derived from, kept
    so a closed window can be checked against the scale that was actually
    applied to it rather than against the one that replaced it."""
    var scale_readbacks: Int
    """How many times this builder has folded a magnitude window. The fit's
    scale round-trip count, exactly: one fold is one `synchronize` on one
    device answer the host could not proceed without."""
    var has_gradients: Bool
    # Whether the Float32 gradients the kernels read this round are also
    # sitting in the `stage_gh` arena on the host: True after
    # `stage_gradients` (the host produced them) or `stage_from_device` (the
    # device produced them and the host read them back), False after a
    # device fill until one of those runs. A host replica build
    # (`build_leaf_host_replica`) is only possible when this is True.
    var gradients_host: Bool
    # The batched multi-leaf launcher, held as a zero-or-one list so a
    # builder that never batches allocates none of its buffers. `List` is
    # what holds a move-only value here; there is no second batcher and no
    # second output pool anywhere.
    var batcher: List[GpuLeafBatcher]
    # The device tree tables a device-owned growth loop commits into, held
    # for the life of the fit in a one-element list exactly as `batcher` is.
    # Empty unless `open_resident_tables` was called, which only
    # `gpu_resident_round.mojo` does; nothing on the shipping path allocates
    # or reads it.
    var resident_tables: List[DeviceTreeTables]
    # The specialization level asked for, from
    # `MOJOTREES_GPU_HIST_SPECIALIZATION`. `SPEC_LEVEL_BASELINE` is the
    # default and means every launch below is the one that shipped.
    var spec_level: Int
    # The transfer route `unified_memory_policy` resolved for the binned
    # matrix. `ROUTE_COPY_STAGED` today; carried so the upload site and a
    # trace name the same route.
    var bins_route: Int
    # The two counters `gpu_leaf_batching.subtraction_stamp` folds into a
    # histogram slot's stamp: one bumped whenever the fixed-point scales are
    # re-derived (once per round, and once per class in a multiclass round),
    # one whenever the active feature set moves. Kept as counters rather than
    # as a stamp so the encoding lives in one place, next to the subtraction
    # that depends on it.
    var round_epoch: Int
    var feat_epoch: Int
    # The stamp the batcher's feature table was last staged at, or -1 when it
    # has never been staged.
    var batch_feat_stamp: Int

    def __init__(
        out self, data: BinnedMatrix, strategy: Int = STRATEGY_AUTO
    ) raises:
        """Upload `data` and resolve the launch geometry for its shape.

        `strategy` forces `STRATEGY_ATOMIC` or `STRATEGY_TILED`; the default
        `STRATEGY_AUTO` lets `MOJOTREES_GPU_HIST_STRATEGY` and then the
        device-capability policy in `gpu_tiling.mojo` decide.

        Opens a private `DeviceContext`. A builder that should share one
        context (and one queue) with other device work takes a `GpuSession`
        instead; see the session overload below.
        """
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        self = Self(ctx, caps, data, strategy)

    def __init__(
        out self,
        mut session: GpuSession,
        data: BinnedMatrix,
        strategy: Int = STRATEGY_AUTO,
    ) raises:
        """Build on `session`'s context instead of opening a private one, so
        every buffer this builder creates shares the session's in-order
        queue, and record the construction in the session's ledgers.

        Bookkeeping only, no behavior change: the binned matrix is uploaded
        and drained exactly as the private-context form does, and the
        residency and pool entries record what a pooled path could later
        skip (a second builder on the same session and matrix still
        re-uploads today, because the buffers live in the builder rather
        than the session). Construction is charged to `PHASE_ALLOC`, which
        `session.trace()` reports under `MOJOTREES_GPU_TRACE=1`.
        """
        var started = session.clock()
        self = Self(session.ctx, session.caps, data, strategy)
        _ = session.admit_matrix(
            ROLE_TRAIN,
            MatrixIdentity(
                data.n_rows,
                data.n_features,
                data.n_bins,
                bins_fingerprint(
                    data.bins, data.n_rows, data.n_features, data.n_bins
                ),
            ),
        )
        var hist_cells = 3 * data.n_features * data.n_bins
        _ = session.request_buffer(
            SLOT_BINS, data.n_rows * data.n_features, 1
        )
        # Two ledger entries, one allocation: the gradient and hessian planes
        # are windows onto a single `2 * n_rows` buffer (see `gh_dev`). The
        # bytes each role is charged are unchanged, which is what these
        # entries are for; the allocation count they imply is now one high.
        _ = session.request_buffer(SLOT_GRAD, data.n_rows, 4)
        _ = session.request_buffer(SLOT_HESS, data.n_rows, 4)
        _ = session.request_buffer(SLOT_FEAT, data.n_features, 4)
        _ = session.request_buffer(SLOT_OUT, hist_cells, 4)
        _ = session.request_buffer(SLOT_PART, 3 * self.part_capacity, 4)
        _ = session.request_buffer(SLOT_STAGE, 2 * data.n_rows, 4)
        _ = session.request_buffer(SLOT_HOST_OUT, hist_cells, 4)
        session.record(PHASE_ALLOC, started)

    def __init__(
        out self,
        ctx: DeviceContext,
        caps: DeviceCaps,
        data: BinnedMatrix,
        strategy: Int = STRATEGY_AUTO,
    ) raises:
        """Build on a caller-supplied context and its queried capabilities;
        the two public forms above both land here."""
        if data.n_rows < 1:
            raise Error("GPU backend requires at least one row")
        if data.n_features < 1:
            raise Error("GPU backend requires at least one feature")
        if data.n_rows > MAX_ROWS:
            raise Error("GPU backend supports at most 2^31 - 1 rows")

        # The bin count is the portability module's limit to state, not this
        # one's: the kernels index a `MAX_BINS`-wide threadgroup plane by a
        # UInt8 bin, and `require_bins_supported` is where that structural
        # fact is written down. This replaces the two hand-rolled bin checks
        # that used to sit here and said the same thing in a second place.
        require_bins_supported(data.n_bins)

        # What this backend must provide, asked once when the builder opens.
        # This is the question the builder never asked at all, and the one
        # worth asking early: a device that cannot host these kernels fails
        # here rather than after the binned matrix has been uploaded to it.
        self.contract = contract_from_profile(profile_from_caps(caps))
        require_device_can_host_kernels(self.contract, caps)
        # Whether this shape can be laid out on a device at all is
        # `gpu_binned_layout`'s question, and it asks two the checks above do
        # not: a feature count past the Int32 index range, and an
        # `n_rows * n_features` cell count that overflows it even though each
        # factor fits. Those are the shapes whose flat bin index would wrap.
        # The specific tests run first so a bad row or bin count is still
        # reported as the number it is rather than as an unsupported layout,
        # which is the order `_check_device_search_supported` uses in
        # train_gpu.mojo for the same reason.
        check_layout_support(data.n_rows, data.n_features, data.n_bins)
        if len(data.bins) != data.n_rows * data.n_features:
            raise Error("binned matrix size must equal n_rows * n_features")

        # The one route policy in the package answers for the binned
        # matrix's upload. Every route but the staged copy is blocked or
        # unproven today, so this returns the staged copy unless a caller
        # set `MOJOTREES_GPU_TRANSFER`, and an explicit request it cannot
        # honor raises here rather than silently taking the default.
        # `unified_memory` is False because a `DeviceCaps` does not report
        # it, which is the same conservative answer `profile_from_caps`
        # gives from the same three attributes. It can only widen what the
        # resolver allows, so answering False cannot enable a route on a
        # device that has not been shown to support it.
        var route = resolve_from_env(ROLE_BINS, False)
        if route.selected != ROUTE_COPY_STAGED:
            raise Error(
                "the GPU histogram builder implements the staged copy only,"
                " and this run resolved to a different transfer route: ",
                describe_decision(route),
            )
        self.bins_route = route.selected
        self.spec_level = env_specialization_level()

        var device_api = ctx.api()
        var device_arch = ctx.arch_name()
        self.ctx = ctx
        self.n_rows = data.n_rows
        self.n_features = data.n_features
        self.n_bins = data.n_bins
        self.missing_bin = data.missing_bin.copy()
        self.cats = data.cats.copy()
        self.caps = caps.copy()
        self.device_api = device_api^
        self.device_arch = device_arch^
        self.batcher = List[GpuLeafBatcher]()
        self.resident_tables = List[DeviceTreeTables]()
        self.tiling = derive_tiling(
            self.caps, data.n_rows, data.n_features, data.n_bins, strategy
        )
        # The whole-matrix launch this tiling describes, checked before
        # anything is allocated for it. The grid's x axis carries the feature
        # count on this path.
        require_histogram_launchable(
            self.contract,
            self.caps,
            self.tiling,
            data.n_features,
            data.n_bins,
            build_kernel_features(),
        )
        self.part_capacity = self.tiling.partial_cells
        self.rows = GpuActiveRows(
            self.ctx, data.n_rows, data.n_features, data.n_bins, self.caps
        )
        # The default feature group, by the one rule that cannot cost
        # residency. Metal's baseline is the pairing: measured on an Apple M4
        # at 1.39x end to end for a 5M x 50 fit with byte-identical
        # predictions (see _range_hist_partial_kernel), and pairing changes no
        # histogram it produces on any backend. CUDA/HIP/unknown keep the
        # single slot, because the doubling is exactly the kind of change that
        # can invert on a different threadgroup budget and nobody has measured
        # one.
        #
        # What is new is that the baseline is a *footprint* rather than a
        # width. The kernels used to allocate three MAX_BINS-wide Int32 planes
        # at every bin count, so the baseline group was paid at 256 bins
        # whatever the dataset was binned at; now that they are sized to the
        # real capacity, free_feature_group returns the widest rung that costs
        # no more bytes than that baseline already cost. A 64-bin dataset gets
        # 8 on Metal for the same 6 KiB, a 32-bin one gets 16, and a 256-bin
        # one gets the baseline back unchanged. Since the bytes per block do
        # not rise, the resident blocks per core cannot fall, so this widening
        # is not an occupancy regression on any device and does not need a
        # measurement to ship.
        #
        # Anything wider than the rule allows does trade residency for
        # row-side traffic, and that trade is UNMEASURED here. It stays an
        # explicit MOJOTREES_GPU_FEATURE_GROUP request, which still wins in
        # both directions.
        if getenv("MOJOTREES_GPU_FEATURE_GROUP") == "":
            var baseline = 1
            if parse_api(self.device_api) == API_METAL:
                baseline = 2
            self.rows.set_feature_group(
                free_feature_group(self.rows.bin_cap, baseline)
            )
        self.g_scale = 1.0
        self.h_scale = 1.0
        self.fixed_scale_shape = DEFAULT_SCALE_SHAPE
        # The shipped cadence, and the shipped arithmetic: one fold per
        # round, no headroom. `set_scale_refresh` is what moves it, and its
        # docstring is where the case for moving it is made and bounded.
        self.scale_refresh = 1
        self.scale_headroom = 0
        self.scale_ref_total_g = 0.0
        self.scale_ref_total_h = 0.0
        self.scale_readbacks = 0
        self.has_gradients = False
        self.gradients_host = False
        self.round_epoch = 0
        self.feat_epoch = 0
        self.batch_feat_stamp = -1
        # One copy per round rather than two. Both arms write the same bytes
        # to the same device addresses, so this is not a numeric option; the
        # argument for the default is in `set_fused_gradient_upload`.
        self.fused_upload = True
        self.active = List[Int](capacity=data.n_features)
        for f in range(data.n_features):
            self.active.append(f)

        var n_cells = data.n_rows * data.n_features
        var hist_size = data.n_features * data.n_bins
        self.bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](n_cells)
        # One derivative allocation, two windows onto it. The windows are
        # taken once here and never retaken: a window is a value, the parent
        # never moves or reallocates, and the buffer this builder hands to a
        # kernel has to be the same buffer for the life of the fit.
        self.gh_dev = self.ctx.enqueue_create_buffer[DType.float32](
            2 * data.n_rows
        )
        self.grad_dev = self.gh_dev.create_sub_buffer[DType.float32](
            0, data.n_rows
        )
        self.hess_dev = self.gh_dev.create_sub_buffer[DType.float32](
            data.n_rows, data.n_rows
        )
        self.out_dev = self.ctx.enqueue_create_buffer[DType.int32](
            3 * hist_size
        )
        self.feat_dev = self.ctx.enqueue_create_buffer[DType.int32](
            data.n_features
        )

        # Zero-length device buffers are not portable, so the atomic
        # strategy still allocates a one-element placeholder.
        var part_size = 3 * self.part_capacity
        if part_size < 1:
            part_size = 1
        self.part_dev = self.ctx.enqueue_create_buffer[DType.int32](part_size)

        self.stage_gh = self.ctx.enqueue_create_host_buffer[DType.float32](
            2 * data.n_rows
        )
        self.host_out = self.ctx.enqueue_create_host_buffer[DType.int32](
            3 * hist_size
        )

        # Upload the binned matrix once; it is reused every call. This is
        # `ROUTE_COPY_STAGED`, the route resolved above, and the only one
        # implemented here. The copy reads host memory owned by the caller's
        # `data`, so it has to complete before the constructor returns.
        self.ctx.enqueue_copy(
            dst_buf=self.bins_dev, src_ptr=data.bins.unsafe_ptr()
        )
        self.ctx.synchronize()

        # Every feature is active until `set_features` narrows it.
        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(data.n_features):
                dst.unsafe_store(f, Int32(f))

    def strategy(self) -> Int:
        """The accumulation strategy this builder resolved to."""
        return self.tiling.strategy

    def set_feature_group(mut self, group: Int) raises:
        """How many feature slots one histogram threadgroup accumulates: a
        rung of the ladder 1, 2, 4, 8, 16, each of which the kernel family in
        gpu_active_rows.mojo is instantiated at. Every rung builds the same
        integer histogram, so this is a launch shape and not a numeric option;
        it exists so a benchmark can hold two arms in one process instead of
        reading its arm from the environment. A rung whose threadgroup
        footprint at this dataset's bin capacity exceeds what the device
        reported is refused rather than launched."""
        self.rows.set_feature_group(group)

    def feature_group(self) -> Int:
        """The launch shape `set_feature_group` last chose."""
        return self.rows.feature_group

    def set_row_unroll(mut self, on: Bool):
        """Whether the histogram row loop keeps `HIST_ROW_UNROLL` rows in
        flight or walks one row per iteration.

        The forwarder exists for the same reason `set_feature_group`'s does:
        `train_gpu` constructs its own builder, so a benchmark that wants both
        arms end to end has no other way to reach the setting, and the knob is
        deliberately a runtime argument rather than an environment variable
        because this machine's device timings drift several-fold between time
        windows and only interleaved arms compare.

        Like the feature group this is a launch shape and not a numeric
        option. Both arms visit the same rows of the same range and add the
        same fixed-point integers into the same bins, and integer addition is
        associative and commutative, so the histogram is identical either way.
        The argument is written out at `GpuActiveRows.set_row_unroll`, which
        this forwards to unchanged.

        Why it is worth reaching end to end rather than trusting the isolated
        histogram benchmark: the row-tile floor measured well in isolation and
        was a 22 to 36 percent regression in a whole fit, because the isolated
        shape did not carry the partial traffic the real round does. A kernel
        arm that does not show up in a fit is not a win in a fit.
        """
        self.rows.set_row_unroll(on)

    def row_unroll(self) -> Bool:
        """The row-walk arm `set_row_unroll` last chose."""
        return self.rows.row_unroll

    def set_scale_shape(mut self, shape: Int) raises:
        """Which shape this builder's fixed-point scales take from the next
        `upload_gradients` or device fill on.

        `SCALE_SHAPE_POW2` is the default: `2^30 / sum|g|` rounded *down* to
        a power of two. `SCALE_SHAPE_ARBITRARY` is what shipped before it.
        The rule, the exactness argument, the overflow proof, and the
        one-bit resolution cost are all at
        `quantized_gradient.fixed_point_scale_pow2`, which is the single
        statement of the rule for the CPU and the GPU alike; nothing about it
        is restated here.

        **This is the one arm on this builder that changes histogram bits.**
        `set_row_unroll`, `set_feature_group`, `set_narrow_index`,
        `set_pair_alignment`, `set_row_tiling`, and
        `set_fused_gradient_upload` are launch shapes and transfer shapes:
        every one of them accumulates the same fixed-point integers into the
        same bins, and their docstrings say so. This one changes the unit
        those integers count in, so it changes every cell, therefore possibly
        a split, therefore possibly a tree. A fixture taken under one shape
        does not describe the other and must not be diffed against it.

        It exists as a runtime arm for the same two reasons the others do.
        First, `train_gpu` constructs its own builder, so an end-to-end
        comparison has no other way to reach the setting -- though reaching
        it end to end still needs a `train_gpu` parameter, exactly as
        `set_fused_gradient_upload` records for itself. Second, an
        environment variable would compare two arms across two processes, and
        this machine's device timings drift several-fold between time
        windows, so only interleaved arms inside one process resolve
        anything.

        The accuracy case for the default is the argument for the change, and
        it is an argument rather than a measurement: three roundings deleted
        against up to one bit of lattice step, worked through per bin at
        `fixed_point_scale_pow2`. **The speed effect is unmeasured**, in
        either direction, and this arm is what makes it measurable: a
        power-of-two multiply is an exponent add where an arbitrary one is a
        full multiply, and the reciprocal on the download path becomes exact,
        but nothing here predicts a time from that and nothing may.

        Refuses an unknown code rather than defaulting to one, because
        silently quantizing on a lattice the caller did not ask for is the
        failure this whole file is arranged to prevent.
        """
        if shape != SCALE_SHAPE_POW2 and shape != SCALE_SHAPE_ARBITRARY:
            raise Error(
                String(
                    "unknown fixed-point scale shape: ",
                    String(shape),
                    "; expected SCALE_SHAPE_POW2 or SCALE_SHAPE_ARBITRARY",
                )
            )
        self.fixed_scale_shape = shape

    def scale_shape(self) -> Int:
        """The scale arm `set_scale_shape` last chose."""
        return self.fixed_scale_shape

    def set_scale_refresh(mut self, rounds: Int, headroom_bits: Int = 0
    ) raises:
        """How often the host waits for a magnitude sum, and how much room it
        leaves the rounds that do not.

        `rounds = 1, headroom_bits = 0` is the shipped cadence and the shipped
        arithmetic. `rounds = 0` selects the unwindowed `magnitude_sums` call
        instead of the split enqueue/read pair, expression for expression, and
        exists so a test can compare the two rather than trust that a window
        of one is the same thing.

        WHAT THIS ARM IS FOR
        --------------------
        `train_gpu.mojo`'s census counts two round trips per round on the
        default arm, where a round trip is host code blocking on a device
        answer it needs before it can enqueue anything else. One of them is
        this: `fill_gradients_device` reduces `|grad|` and `|hess|` on the
        device, the host folds the partials in Float64, and every histogram
        launch of the round takes the resulting scale as a launch argument, so
        **nothing can be enqueued until it lands.** At `R = 100` that is 100
        of the fit's 200 round trips, **estimated** at 0.046 seconds against a
        fit **measured** at about 2.58 seconds, which is below what this
        machine can resolve in one window.

        The obvious removal is not available here and it is worth saying why,
        because it is the first thing a reader will ask. Making the scale
        device-resident means the *host* never needs its value, and the host
        needs its value in three places outside this file: as a `Float32`
        launch argument to nine kernels across `gpu_active_rows.mojo`,
        `gpu_gradient_stream.mojo`, `gpu_categorical.mojo` and
        `gpu_sparse.mojo`; as a staged `Float32` table in
        `gpu_leaf_batching.mojo` and `gpu_multiclass_batch.mojo`; and as
        `Float32(1.0 / g_scale)` written into the split searcher's parameter
        block at `gpu_split_search._stage_params`. The last one is decisive
        and is not a plumbing problem: the gain and the leaf value are not
        homogeneous in the scale (the regularizer `lambda` is not scaled), so
        a kernel handed pre-scaled gradients and told the scale is 1.0 does
        not compute the same gain. The host has to know the number. This arm
        therefore changes **how often the host waits for it**, which is the
        part that is reachable, and not **whether it needs it**, which is not.

        WHAT A WINDOW COSTS, AND WHY IT IS SAFE
        ---------------------------------------
        With `rounds = N`, the host folds once every `N` rounds and the
        rounds in between quantize on the scale the last fold produced. Three
        separate claims hold that up, and none of them is an assumption about
        how gradients behave.

        1. **Every round's magnitudes are still measured exactly.** The
           reduction runs every round; only the *readback* is deferred, into
           its own slot of a pinned window
           (`GpuObjectiveState.enqueue_magnitudes`). When the window closes,
           the host has round `j`'s totals bit for bit as it would have had
           them at round `j`. Nothing is estimated and nothing is skipped.

        2. **A closed window is checked against the scale that was applied to
           it.** The overflow argument the whole fixed-point path rests on is
           `sum_i |g_i| * s <= 2^30` (`quantized_gradient.fixed_point_scale_pow2`).
           On a fold, every round in the window that ran under the previous
           scale is tested against exactly that inequality, and a violation
           **raises**. So a stale scale cannot silently overflow an Int32
           cell; it can only end the fit with a message naming the round. The
           detection is one window late by construction, which is why 3 exists
           to make it not fire.

        3. **`headroom_bits` prevents rather than detects.** The scale is
           derived from `T * 2^H` instead of `T`, so it is smaller by exactly
           `2^H` -- exactly, because `T * 2^H` is an exact Float64 product and
           the power-of-two rule commutes with a power-of-two rescaling of its
           input. The window is then safe as long as no round in it exceeds
           `2^H` times the largest magnitude sum the previous window saw. At
           `H = 1` that is a doubling, which a boosting round whose residuals
           are shrinking does not do. **Derived bound**, not measured.

        The reference the next window is derived from is the **maximum** over
        the closing window, not its last round, which is the conservative
        choice: a window is sized against the worst round it has seen rather
        than the most recent one.

        WHAT IT COSTS IN ACCURACY, STATED PLAINLY
        -----------------------------------------
        **This arm moves histogram bits at any setting other than
        `rounds <= 1, headroom_bits = 0`, and therefore moves trees.** Two
        separate mechanisms, and they are different in kind:

        - `headroom_bits = H` gives up exactly `H` bits of lattice
          resolution on every round, always. The exact scaled total moves
          from `(2^29, 2^30]` to `(2^(29-H), 2^(30-H)]`. That is on top of
          the up-to-one bit the power-of-two rule already gives up, so `H = 1`
          means a lattice holding between 28 and 30 bits below the total
          where the arbitrary rule held 30.
        - `rounds = N > 1` gives up whatever `log2(T_ref / T_j)` is on each
          reused round, which is zero whenever the magnitudes stay inside one
          binade of the reference and is otherwise unbounded below. Because
          the rule is a *step function*, most reused rounds land on the same
          power of two the fresh derivation would have chosen and are
          bit-identical to it. That is a property of a step function and not a
          guarantee: a round whose magnitude sum crosses a binade boundary
          gets a scale a factor of two from the one it would have had, and the
          histogram, the split, and the tree can all differ. Which rounds
          those are is a property of the data.

        So the default stays 1 and 0, and this is an arm rather than a
        change. It is a runtime argument and not an environment variable for
        the reason `set_row_unroll` gives: this machine's device timings drift
        several-fold between time windows, so only two arms interleaved inside
        one process compare.

        Refuses out-of-range values rather than clamping, on the same grounds
        `set_scale_shape` refuses an unknown shape: quantizing on a lattice
        the caller did not ask for is the failure this file is arranged to
        prevent.
        """
        if rounds < 0 or rounds > SCALE_WINDOW_MAX:
            raise Error(
                String(
                    "scale refresh cadence must be 0 (the unwindowed call) or",
                    " 1..",
                    String(SCALE_WINDOW_MAX),
                    "; got ",
                    String(rounds),
                )
            )
        # A headroom past 30 bits would leave no lattice at all: the exact
        # scaled total would sit at or below 1, so every gradient in the
        # round would quantize to zero or one unit and no split could be
        # told from any other.
        if headroom_bits < 0 or headroom_bits > 30:
            raise Error(
                String(
                    "scale headroom must be 0..30 bits; got ",
                    String(headroom_bits),
                )
            )
        self.scale_refresh = rounds
        self.scale_headroom = headroom_bits

    def scale_refresh_rounds(self) -> Int:
        """The cadence `set_scale_refresh` last chose."""
        return self.scale_refresh

    def scale_readback_count(self) -> Int:
        """How many magnitude windows this builder has folded.

        The fit's scale round-trip count, as a number rather than as an
        argument: one fold is one `synchronize` on one device answer the host
        could not proceed without. A caller charging a phase profile reads it
        before and after a fill and charges `syncs=1` only when it moved,
        which is what keeps a census taken off a profile honest when the
        cadence is not one.
        """
        return self.scale_readbacks

    def _scale_from_total(self, total: Float64) raises -> Float64:
        """This builder's scale for a magnitude sum, with the window's
        headroom applied.

        `total * 2^H` rather than `scale / 2^H`, because the first is exact
        in Float64 for every `H` this setter admits and the second would
        narrow through Float32 twice. Under `SCALE_SHAPE_POW2` the two agree
        anyway -- the power-of-two rule commutes with a power-of-two
        rescaling of its input -- and under `SCALE_SHAPE_ARBITRARY` the first
        is the one that keeps a single rounding.

        At `H = 0` the multiplier is exactly 1.0 and this is
        `fixed_point_scale_shaped(total, shape)`, which is what makes the
        default arm's arithmetic the shipped arithmetic and not a
        reconstruction of it.
        """
        var t = total
        if self.scale_headroom > 0:
            t = total * Float64(1 << self.scale_headroom)
        return Float64(fixed_point_scale_shaped(t, self.fixed_scale_shape))

    def _close_scale_window(
        mut self, mut state: GpuObjectiveState, quantized_all: Bool = False
    ) raises:
        """Fold every pending magnitude slot, check the window that is
        closing against the scale it ran under, and derive the scale for the
        window that opens.

        The one wait, and the one place `scale_readbacks` moves. Order
        matters and is the order below: check first, then derive, so a
        violation is reported against the scale that caused it rather than
        against its replacement.

        `quantized_all` says whether the newest slot is a round that has
        already built histograms. It is False on the round path, where the
        newest slot is the round whose gradients were just filled and which
        is about to run under the scale this call derives -- the derivation is
        what makes that round safe, so testing it against the outgoing scale
        would be testing the wrong inequality. It is True on the end-of-fit
        flush, where every pending slot is a round that has finished, and
        leaving the last one out there would be the one round the check never
        covered.
        """
        var window = state.read_magnitudes(self.ctx)
        self.scale_readbacks += 1
        var max_g = 0.0
        var max_h = 0.0
        for i in range(len(window)):
            if window[i].grad > max_g:
                max_g = window[i].grad
            if window[i].hess > max_h:
                max_h = window[i].hess
        # The rounds that already quantized under the scale in force. See the
        # docstring for why the newest is normally not one of them.
        var quantized = len(window) if quantized_all else len(window) - 1
        if quantized > 0:
            var ran_g = 0.0
            var ran_h = 0.0
            for i in range(quantized):
                if window[i].grad > ran_g:
                    ran_g = window[i].grad
                if window[i].hess > ran_h:
                    ran_h = window[i].hess
            # Guarded per plane and on the *reference* rather than on the
            # readback count, because the two planes can differ: a round
            # whose hessians are all zero has no hessian lattice to outgrow
            # and its scale came off the magnitude floor rather than off a
            # measurement, so the inequality has nothing to say about it. A
            # zero reference is also what the very first fold of a fit
            # leaves, which is the case where `g_scale` is still the
            # constructor's 1.0 and checking against it would be checking
            # against a number no round ever quantized with.
            if self.scale_ref_total_g > 0.0:
                _check_window_bound(ran_g, self.g_scale, "gradient")
            if self.scale_ref_total_h > 0.0:
                _check_window_bound(ran_h, self.h_scale, "hessian")
        self.scale_ref_total_g = max_g
        self.scale_ref_total_h = max_h
        self.g_scale = self._scale_from_total(max_g)
        self.h_scale = self._scale_from_total(max_h)

    def _refresh_scales(mut self, mut state: GpuObjectiveState) raises:
        """This round's scales, at whatever cadence `set_scale_refresh`
        chose. Shared by the plain and the softmax device fills so the two
        cannot drift apart on when the host waits.

        Three arms, and the first two are the same arithmetic:

        - `scale_refresh == 0`: the unwindowed `magnitude_sums` call, which
          enqueues, copies, synchronizes and folds in one go. The reference
          arm.
        - `scale_refresh == 1`: the split pair with a window of one, which is
          that call in two halves. One wait per round, same scale.
        - `scale_refresh > 1`: one wait per `N` rounds. The first round of a
          fit always folds, because there is no scale to reuse yet.
        """
        if self.scale_refresh == 0:
            var sums = state.magnitude_sums(
                self.ctx, self.grad_dev, self.hess_dev
            )
            self.scale_readbacks += 1
            self.scale_ref_total_g = sums.grad
            self.scale_ref_total_h = sums.hess
            self.g_scale = self._scale_from_total(sums.grad)
            self.h_scale = self._scale_from_total(sums.hess)
            return
        var pending = state.enqueue_magnitudes(
            self.ctx, self.grad_dev, self.hess_dev
        )
        # `scale_readbacks == 0` is the first fill of the fit: there is no
        # scale to reuse, so the window closes immediately whatever the
        # cadence says.
        if pending >= self.scale_refresh or self.scale_readbacks == 0:
            self._close_scale_window(state)

    def flush_scale_window(mut self, mut state: GpuObjectiveState) raises:
        """Fold and check any magnitude slots the fit left pending.

        Called once at the end of a fit. A window wider than one leaves the
        last few rounds unread when the loop stops, and those rounds are
        exactly the ones the overflow check in `_close_scale_window` has not
        run against yet. Flushing costs one round trip per fit and is what
        makes the check cover **every** round rather than every round but the
        last few, which is the difference between a guarantee and a habit.

        A no-op when nothing is pending, so the default cadence pays nothing
        for it: at `scale_refresh <= 1` the window is always empty here.
        """
        if state.magnitudes_pending() < 1:
            return
        self._close_scale_window(state, quantized_all=True)

    def describe_scale(self) -> String:
        """One phrase naming the scale arm, for a trace line."""
        return describe_scale_shape(self.fixed_scale_shape)

    def set_narrow_index(mut self, on: Bool) raises:
        """Whether the histogram row loop forms its two data-dependent
        indices in Int32 rather than in Int.

        A forwarder for the reason `set_row_unroll`'s docstring gives:
        `train_gpu` builds its own builder, so an end-to-end interleaved A/B
        has no other way to reach the arm.

        Off by default and refused outright on a dataset whose shape does not
        admit it, which `GpuActiveRows.narrow_index_supported` states as a
        bound and `GpuActiveRows.set_narrow_index` enforces. Under that bound
        the two arms address the same bytes and accumulate the same integers,
        so the histogram is identical; above it the narrow arm would wrap an
        index, which is why it raises rather than degrading.
        """
        self.rows.set_narrow_index(on)

    def narrow_index(self) -> Bool:
        """The index-width arm `set_narrow_index` last chose."""
        return self.rows.narrow_index

    def set_pair_alignment(mut self, on: Bool):
        """Whether the width-2 load of the quantized gradient pair states the
        8-byte alignment its address actually has.

        A forwarder, for the same reason the others here are. On by default:
        the unannotated spelling emits `align 4` for a `<2 x i32>` load, and
        an under-aligned vector load is one a backend may split back into the
        two scalar loads that width-2 spelling exists to replace. Both read
        the same eight bytes, so this cannot change a histogram.
        """
        self.rows.set_pair_alignment(on)

    def pair_alignment(self) -> Bool:
        """The pair-load arm `set_pair_alignment` last chose."""
        return self.rows.pair_alignment

    def set_row_tiling(
        mut self, min_tiles: Int = 0, rows_per_tile: Int = 0
    ) raises:
        """Request a row-tile floor, a rows-per-tile length, or neither, for
        every node this builder's device-resident path plans from here on.

        Zero on both, the default, is the geometry this builder always
        produced. A forwarder for the reason the others here are, and this one
        matters more than most: the row-tile floor has to be re-measured
        against the unrolled row walk, and the earlier measurement that made
        it opt-in was taken across processes through an environment variable.

        Affects only `GpuActiveRows.range_tiling`, which is the per-node
        geometry the device-resident path derives. The whole-dataset
        `self.tiling` this builder resolved in its constructor is not
        re-derived, because it is a property of the dataset rather than of a
        node and nothing here changes the dataset.

        Cannot change a histogram: tiling is a launch geometry and
        accumulation is fixed-point Int32, so two geometries over the same
        rows sum the same bins in a different order to the same value.
        """
        self.rows.set_row_tiling(min_tiles, rows_per_tile)

    def set_fused_gradient_upload(mut self, on: Bool):
        """Whether `upload_staged` moves this round's two derivative planes
        with one copy of their shared allocation or with one copy each.

        Not a numeric option and not a launch shape either: the two arms
        write identical bytes to identical device addresses, and the argument
        for that is at `upload_staged`. The only difference is the number of
        drains the round issues, because on Metal `enqueue_copy` is a
        synchronous full-queue drain whose behavior does not scale with the
        byte count (`docs/GPU_PORTABILITY.md` section 6.1).

        **What that difference is worth is not a time.** An earlier version of
        this docstring called the two copies two host waits and justified the
        default on the ground that removing a copy removes a synchronization
        by an established fact. Section 6.1.1 withdrew that inference on
        2026-08-16: a drain of a queue holding nothing costs nothing, and
        nothing is queued behind a gradient upload. The nearest **measured**
        point is the thirteen-copy collapse on the device-resident plane,
        which bought 0.016 seconds at 1,000,000 x 50 against a registered
        prediction of 0.64 and did not resolve under M0
        (`bench/results/session3_2026-08-16/RESULTS.md`). Nothing here
        predicts that fusing these two is different.

        Why fused is still the default rather than the requested arm. The
        usual rule is that an unmeasured arm has to be asked for, and that
        rule is about arms whose sign is unknown. This one's sign is known in
        the dimension that is left: the bytes that move are the same bytes,
        and one copy is one staging lifetime, one ordering point, and one
        place a `[grad | hess]` pair can arrive half fresh, where two are two.
        It is a hazard and portability default, not a speed default, and no
        speed claim may be attached to it. The split arm survives so an
        interleaved A/B can be run if anyone wants it; the **estimate**, from
        the null above, is that it will not resolve.

        Reachable in process, and only in process. `train_gpu` builds its own
        builder and takes its arm knobs as function parameters (`row_unroll`
        is the pattern), so an end-to-end interleaved comparison needs either
        such a parameter in `train_gpu.mojo` or an environment variable read
        in this constructor. Both were out of this lane's file budget -- the
        second because a new `MOJOTREES_*` literal under `src/` moves
        `compatibility/api_snapshot.json`, which is a gate artifact and a
        third file. Neither is more than a line; see the lane report.
        """
        self.fused_upload = on

    def fused_gradient_upload(self) -> Bool:
        """The upload arm `set_fused_gradient_upload` last chose."""
        return self.fused_upload

    def set_constant_hessian(mut self, on: Bool):
        """Declare that this round's objective guarantees a per-row hessian
        of exactly `histogram.CONSTANT_HESSIAN`, so the histogram kernels may
        stop accumulating that plane and reconstruct it from the count.

        The declaration and its hazards belong to
        `GpuActiveRows.set_constant_hessian`, which this forwards to; the
        predicate that answers it correctly is
        `histogram.objective_has_constant_hessian`, and it is false for every
        weighted fit and every GOSS round whatever the objective code says.
        `MOJOTREES_CONST_HESSIAN=0` refuses the declaration outright, and
        `constant_hessian` below reports what was actually adopted rather
        than what was asked for.

        Held on the builder rather than passed per node because it is a
        property of the round, exactly like `g_scale` and `h_scale`. A
        trainer should set it once where it computes this round's gradients
        and clear it when it does anything -- weights, GOSS, a different
        objective, a softmax class -- that breaks the guarantee. Nothing in
        this package calls it yet.
        """
        self.rows.set_constant_hessian(on)
        # **THE BATCHER USED TO BE LEFT OUT, AND THAT IS WHY THE BATCHED
        # ELISION HAS NEVER RUN.** This method forwarded to `self.rows` and
        # stopped, so `GpuLeafBatcher.constant_hessian` stayed False for the
        # life of every fit and the `celide` arm of `_batch_hist_atomic_kernel`,
        # of its subtracting twin, and of `_plan_hist_kernel` was dead code on
        # the oblivious level build. Behind `MOJOTREES_GPU_BATCH_CONST_HESS=1`
        # so the elision is measured against the arm that has been shipping
        # rather than arriving inside somebody else's number.
        #
        # Not sufficient on its own and not meant to be: `train_gpu` makes the
        # declaration at fit setup, before `open_resident` has allocated any
        # batcher, so `self.batcher` is usually empty here and this line does
        # nothing. `enqueue_desc_level_children` repeats it where the batcher
        # certainly exists, and that is the forward that reaches a kernel.
        #
        # `self.constant_hessian()` and not `on`: what is forwarded is what
        # `GpuActiveRows` ADOPTED, so `MOJOTREES_CONST_HESSIAN=0` withdraws the
        # permission from both sides at once and the two cannot disagree.
        if len(self.batcher) > 0 and batch_const_hessian_forward_requested():
            self.batcher[0].set_constant_hessian(self.constant_hessian())

    def constant_hessian(self) -> Bool:
        """Whether the constant-hessian specialization is actually in force
        for the next build: the declaration ANDed with the environment's
        permission. A benchmark arm should read this rather than assume its
        request was honored."""
        return self.rows.constant_hessian

    def synchronize(self) raises:
        """Block until every enqueued device operation has completed."""
        self.ctx.synchronize()

    def range_of(self, node: Int) raises -> LeafRange:
        """The half-open window of the device-resident active-row
        permutation `node` currently owns.

        The grower needs this to describe a leaf to a batched launch, and it
        is the same table the partition kernel maintains rather than a second
        host-side model of it, so a batch cannot disagree with the device
        about which rows a node holds. `count()` is the node's row count and
        equals the count the parent histogram's integer bins sum to.
        """
        return self.rows.range_of(node)

    def set_features(mut self, features: List[Int]) raises:
        """Restrict later `build_leaf` calls to `features` (global feature
        ids, one entry each). This is how the GPU grower consumes the same
        subsampled feature set as the CPU grower: the dataset stays whole and
        device-resident, only the launch grid narrows. Slices of features not
        listed here stay zero in every histogram built afterwards, which is
        what keeps sibling subtraction exact as long as one tree keeps one
        feature set."""
        if len(features) == 0:
            raise Error("active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        var changed = len(features) != len(self.active)
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
            if not changed and self.active[i] != features[i]:
                changed = True
        if not changed:
            return

        self.active = features.copy()
        self.feat_epoch += 1
        # Fewer features in grid.x means fewer threadgroups, so the row
        # tiling is re-derived for the narrowed grid. The partial buffer is
        # never reallocated: its construction-time capacity is the cap.
        self.tiling = derive_tiling(
            self.caps,
            self.n_rows,
            len(self.active),
            self.n_bins,
            self.tiling.strategy,
            self.part_capacity,
        )
        # A narrowed grid is a different launch and gets the same gate the
        # full one got. Feature subsampling only ever shrinks grid.x, so this
        # cannot start failing mid-tree on a builder that opened cleanly, but
        # checking is what makes that a property rather than an assumption.
        require_histogram_launchable(
            self.contract,
            self.caps,
            self.tiling,
            len(self.active),
            self.n_bins,
            build_kernel_features(),
        )
        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(features)):
                dst.unsafe_store(i, Int32(features[i]))

    def stage_gradients(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises:
        """Convert this round's gradients and hessians into the device's
        Float32 in pinned host memory, laid out as one `[grad | hess]` arena.
        Host work only, no transfer.

        The device's copy is stale from here until `upload_staged`, and the
        scales below already describe the new values, so builds are refused
        in between rather than mixing one round's scale with another's data.

        Why the synchronize below is still here
        ---------------------------------------
        It guards a host-write hazard: the previous round's upload reads this
        arena, and this method is about to overwrite it. Rotating two or more
        arenas would retire the hazard structurally instead of by waiting,
        and was considered and declined, for two reasons that both have to
        fail before it is worth doing.

        First, on the one backend where the cost of a wait is established, it
        is the cheap one of this round's three. `enqueue_copy` on Metal
        commits an empty command buffer and waits for it before it memcpys
        (`docs/GPU_PORTABILITY.md` section 6.1), so the upload that read this
        arena had already completed before it returned, and this synchronize
        has nothing of its own to wait for. What it drains is whatever the
        previous round left enqueued, and the previous round's last device
        operation is a copy on every path in this file -- `download_raw`,
        `readback_range`, the device split search's own readbacks -- each of
        which drained the queue on the way out. The wait removed would
        therefore be **estimated** near zero on Metal, and **unestablished**
        on CUDA and HIP, where nothing in this repository has shown what MAX
        does with an asynchronous copy family.

        Second, on a backend where copies really are asynchronous this
        synchronize is the entire guarantee, and a rotation's replacement
        guarantee has to be named rather than assumed. "Two rounds have
        passed" is not one: nothing in this class's contract forces a drain
        between two `upload_gradients` calls, so a caller that uploads twice
        with no build in between would come back to an arena whose copy is
        still in flight. A watermark (a copy sequence number against the
        sequence a synchronize last drained through) would be a real
        guarantee and would cut the drains to one per arena-count rounds, but
        it buys a wait that is near zero where we can measure it, costs a
        pinned `2 * n_rows` Float32 per extra arena, and fails silently in
        exactly the way a stale host table failed silently in this project
        this week. The fused upload below removes one drain unconditionally
        and with no hazard at all, which is the better trade of the two.

        Section 6.1.1, on 2026-08-16, strengthened the first reason rather
        than weakening it: a copy is a drain but a drain is not a wait, and
        the only **measured** point puts thirteen such copies per tree at
        0.016 seconds, a null under M0. So the rotation is buying even less
        than the paragraph above conceded, and the hazard argument in the
        second reason is now the whole of the case. That argument is about
        ordering and it stands unchanged.
        """
        if len(grad) != self.n_rows or len(hess) != self.n_rows:
            raise Error("gradient/hessian length must equal n_rows")
        self.has_gradients = False

        var g_scale = _fixed_scale(grad, self.fixed_scale_shape)
        var h_scale = _fixed_scale(hess, self.fixed_scale_shape)
        self.g_scale = Float64(g_scale)
        self.h_scale = Float64(h_scale)
        self.round_epoch += 1

        # Any copy still reading the staging arena has to finish before it is
        # overwritten. See the docstring for why this stays.
        self.ctx.synchronize()

        var dst_g = self.stage_gh.unsafe_ptr()
        var dst_h = dst_g.unsafe_offset(self.n_rows)
        var src_g = grad.unsafe_ptr()
        var src_h = hess.unsafe_ptr()
        for r in range(self.n_rows):
            dst_g.unsafe_store(r, Float32(src_g.unsafe_load(r)))
            dst_h.unsafe_store(r, Float32(src_h.unsafe_load(r)))
        self.gradients_host = True

    def upload_staged(mut self) raises:
        """Copy the staged gradients and hessians to the device: one copy of
        the whole `[grad | hess]` arena, or the two per-plane copies that
        shipped, per `fused_upload`.

        The two arms are byte for byte the same upload. The host arena and
        the device allocation carry the same two planes at the same two
        offsets in the same order, so the fused arm's single copy writes
        `stage_gh[0 : n_rows]` to `gh_dev[0 : n_rows]` and
        `stage_gh[n_rows : 2 * n_rows]` to `gh_dev[n_rows : 2 * n_rows]`,
        which is exactly what the split arm's two copies write into the two
        windows. Neither arm converts, reorders, or pads anything. What
        differs is one drain, because on Metal a copy drains the whole queue
        whatever its byte count is (`docs/GPU_PORTABILITY.md` section 6.1,
        **measured** by disassembly).

        **One drain, not one wait, and the ~458 microseconds that used to be
        written here has been taken off it.** That constant is **derived**
        from the depthwise A/B and it is the price of a *round trip*; neither
        of these copies is one, because nothing is queued behind them and no
        host decision reads a device answer. Section 6.1.1 records the
        withdrawal and the data: thirteen copies per tree removed elsewhere on
        this plane **measured** 0.016 seconds against a registered prediction
        of 0.64, a null under M0
        (`bench/results/session3_2026-08-16/RESULTS.md`). The fused arm is
        kept because one copy is one ordering point and one staging lifetime
        where two are two, not because it is faster.
        """
        if self.fused_upload:
            self.ctx.enqueue_copy(
                dst_buf=self.gh_dev, src_ptr=self.stage_gh.unsafe_ptr()
            )
            self.has_gradients = True
            return
        var stage = self.stage_gh.unsafe_ptr()
        self.ctx.enqueue_copy(dst_buf=self.grad_dev, src_ptr=stage)
        self.ctx.enqueue_copy(
            dst_buf=self.hess_dev, src_ptr=stage.unsafe_offset(self.n_rows)
        )
        self.has_gradients = True

    def upload_gradients(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises:
        """Upload this round's per-row gradients and hessians (once per
        boosting round, not per node)."""
        self.stage_gradients(grad, hess)
        self.upload_staged()

    def stage_from_device(mut self) raises:
        """Read this round's device-produced Float32 gradients and hessians
        back into the pinned staging buffers, so a host replica build can
        run on a device-objective round.

        The values that land are the exact Float32 the kernels read (a copy,
        no arithmetic), and `g_scale`/`h_scale` were already set by the
        device fill, so `build_leaf_host_replica` afterwards accumulates
        exactly what `build_leaf` does — the same replica claim, no new one.
        Costs `2 * 4 * n_rows` bytes and one synchronize per round (per
        class, on a softmax round). No grower calls this: it is how the
        equivalence tests reach a device-objective round from the host.

        A no-op when the gradients are already host-side. Refused before any
        gradients exist.
        """
        if not self.has_gradients:
            raise Error("no gradients to read back this round")
        if self.gradients_host:
            return
        # Two copies and a synchronize, deliberately left as they were. The
        # planes are now adjacent in both directions, so this is one
        # `dst_ptr=stage_gh, src_buf=gh_dev` away from costing one wait
        # instead of two, exactly as `upload_staged` is; it is not fused here
        # because this path is verification-only and no arm of it
        # has been measured, and a second unmeasured arm in the same commit
        # would make the first one harder to attribute.
        var stage = self.stage_gh.unsafe_ptr()
        self.ctx.enqueue_copy(dst_ptr=stage, src_buf=self.grad_dev)
        self.ctx.enqueue_copy(
            dst_ptr=stage.unsafe_offset(self.n_rows), src_buf=self.hess_dev
        )
        self.ctx.synchronize()
        self.gradients_host = True

    def objective_state(
        mut self,
        target: List[Float64],
        sample_weight: List[Float64] = [],
        n_classes: Int = 1,
        max_nodes: Int = DEFAULT_MAX_NODES,
    ) raises -> GpuObjectiveState:
        """A device objective state on this builder's context, so its
        gradients land in this builder's buffers. See
        gpu_objectives_native.mojo."""
        return GpuObjectiveState(
            self.ctx, target, sample_weight, n_classes, max_nodes
        )

    def refresh_objective_weights(
        mut self, mut state: GpuObjectiveState, weights: List[Float64]
    ) raises:
        """Upload this tree's per-row weights into `state`'s weight plane, on
        this builder's context, and refuse the one combination that would make
        the upload produce a wrong histogram.

        The upload itself is `GpuObjectiveState.refresh_weights`; what this
        adds is the builder's half of the contract, which the state cannot
        see. A per-row weight multiplies both derivatives, so under squared
        error, L1, huber and quantile the hessian *is* the weight (the CPU
        stores exactly `w` there, `boosting._fill_grad_hess_into`). A builder
        holding `set_constant_hessian(True)` rebuilds the hessian plane from
        the row count instead of accumulating it, and against a weighted
        round it would rebuild the wrong plane silently, in the histogram,
        with nothing downstream able to tell. So it is refused here.

        This is the same rule as `boosting.round_has_constant_hessian` (which
        ends in `objective_has_constant_hessian(objective, len(sample_weight)
        > 0)`, false for every objective under a non-empty weight vector) and
        `sampling.check_bayesian_bootstrap_hessian_declaration`, reached from
        the third side. Those two are host predicates evaluated once per fit;
        this is the device state, and a declaration once made is held for the
        whole fit and cannot be withdrawn mid-loop, so the ordering a caller
        must follow is: decide the declaration, then set it, then refresh
        weights -- never a refresh into a builder already declared constant.

        Only the objective plane is touched. The histogram kernels take no
        weight argument and none is added: the weight is applied per row in
        the gradient kernel, before quantization, exactly as on the CPU. See
        the gpu_objectives_native.mojo module docstring for that argument and
        for what the refused declaration costs the Int16 staging arm (9 bytes
        per (row, feature) visit against 7).
        """
        if self.constant_hessian():
            raise Error(
                "a constant-hessian declaration is in force: a per-row weight"
                " makes the hessian the weight, so the plane cannot be"
                " rebuilt from the row count. Clear the declaration before"
                " refreshing weights"
            )
        state.refresh_weights(self.ctx, weights)

    def fill_gradients_device(
        mut self,
        mut state: GpuObjectiveState,
        objective: Int,
        alpha: Float64,
    ) raises:
        """This round's gradients, computed on the device straight into the
        histogram buffers. Replaces `upload_gradients` for the built-in
        objectives; the fixed-point scales come from a device reduction
        instead of a host pass, and nothing per-row crosses to the device.

        The scale is derived by `quantized_gradient.fixed_point_scale_shaped`
        rather than by `gpu_objectives_native.device_fixed_scale`, which is
        the same rule reached one forward earlier: `device_fixed_scale` takes
        no shape argument and this builder's arm has to reach the derivation.
        Only where the magnitude sum comes from differs between this path and
        the host one, and that difference is the point of the device
        reduction, not of the scale.

        **Whether this call waits is `set_scale_refresh`'s answer, not
        this method's.** At the shipped cadence it does, once, and that wait
        is one of the two round trips `train_gpu.mojo`'s census counts per
        round. A caller that needs to know whether a particular call waited
        reads `scale_readback_count` before and after; the count is what a
        phase profile's `syncs` column should be charged from, because at any
        cadence but the default a fixed `syncs=1` would be a fiction.
        """
        state.fill_grad_hess(
            self.ctx, objective, alpha, self.grad_dev, self.hess_dev
        )
        self._refresh_scales(state)
        self.has_gradients = True
        self.gradients_host = False
        self.round_epoch += 1

    def ranking_state(mut self, groups: RankGroups) raises -> GpuRankingState:
        """A device ranking state on this builder's context, so its gradients
        land in this builder's buffers. See gpu_objectives_native.mojo, and
        ranking_pairwise.mojo for what the objectives are.

        The query boundaries are uploaded here, once per fit. A pairwise fit
        then calls `GpuRankingState.refresh_pairs` -- once for PairLogit,
        once per round for YetiRank.
        """
        return GpuRankingState(self.ctx, groups)

    def fill_rank_gradients_device(
        mut self,
        mut ranking: GpuRankingState,
        mut state: GpuObjectiveState,
        kind: Int,
        round_index: Int = 0,
    ) raises:
        """This round's ranking gradients, computed on the device straight
        into the histogram buffers. The ranking twin of
        `fill_gradients_device`, and it shares that method's scale derivation
        rather than repeating it, so a ranking round and a regression round
        quantize by the same rule at the same cadence.

        What the builder adds to `GpuRankingState.fill_grad_hess`
        --------------------------------------------------------
        One refusal the state cannot make for itself, and it is the same
        refusal `refresh_objective_weights` makes from the other side. A
        builder holding `set_constant_hessian(True)` rebuilds the hessian plane
        from the row count instead of accumulating it, and **no ranking
        objective may be accumulated that way**:

        - PairLogit and YetiRank have `hess_r = sum over the row's pairs of
          w rho (1 - rho)`, which varies per row, on every round, at every raw
          score. It is never the constant.
        - QueryRMSE has `hess_r = w_r`, so it is exactly
          `histogram.CONSTANT_HESSIAN` when the fit is unweighted and is the
          weight when it is not -- the same shape squared error has, and
          `objective_has_constant_hessian` refuses the declaration for squared
          error under weights for exactly this reason. The unweighted case
          would qualify, and it is still refused here, because
          `objective_has_constant_hessian` is histogram.mojo's statement of
          which objectives qualify and this lane does not extend it. A
          declaration this path accepted without that function's agreement
          would be a second answer to the same question.

        So every ranking round on this path stages both derivative planes. Per
        `GpuActiveRows.staged_gradient_bytes_per_row` that is 4 bytes per row
        rather than 2, and at the default feature group of one each (row,
        feature) visit fetches 4 bytes of row index plus the staged derivative
        plus 1 bin byte: **9 bytes per visit where an unweighted squared-error
        round is on 7.** That is the arithmetic of the declaration, by
        construction, and it is not a regression in anything measured here. The
        gpu_objectives_native.mojo and ranking_pairwise.mojo module docstrings
        state the same sum from the objective's side.
        """
        check_rank_kind(kind)
        if self.constant_hessian():
            raise Error(
                "a constant-hessian declaration is in force and objective '",
                describe_rank_kind(kind),
                "' does not guarantee a per-row hessian of 1: its hessian is"
                " the row weight (QueryRMSE) or a per-round sum over the"
                " row's pairs (PairLogit, YetiRank). Clear the declaration"
                " before filling ranking gradients, or train with"
                " device='cpu'",
            )
        ranking.fill_grad_hess(
            self.ctx,
            state,
            kind,
            self.grad_dev,
            self.hess_dev,
            round_index,
        )
        self._refresh_scales(state)
        self.has_gradients = True
        self.gradients_host = False
        self.round_epoch += 1

    def fill_softmax_gradients_device(
        mut self, mut state: GpuObjectiveState, k: Int
    ) raises:
        """Class `k`'s softmax gradients into the histogram buffers; call
        `state.refresh_softmax` once per round first.

        **The scale window is not used here and `set_scale_refresh` does not
        reach this path**, which is deliberate rather than an omission. A
        softmax round fills, reduces and grows once per class, so the window's
        slots would hold a mixture of classes and the maximum taken over them
        would size one class's lattice by another class's magnitudes. Classes
        do not share a scale anywhere else in this package
        (`gpu_multiclass_batch` keeps one per class) and they must not start
        here. The multiclass mitigation that does exist is
        `_train_multiclass_gpu_batched`, which reduces a batch of classes
        together and so pays one readback per batch instead of one per class;
        it is opt-in and nothing has measured it.
        """
        state.fill_softmax_grad_hess(
            self.ctx, k, self.grad_dev, self.hess_dev
        )
        var sums = state.magnitude_sums(
            self.ctx, self.grad_dev, self.hess_dev
        )
        self.scale_readbacks += 1
        self.g_scale = Float64(
            fixed_point_scale_shaped(sums.grad, self.fixed_scale_shape)
        )
        self.h_scale = Float64(
            fixed_point_scale_shaped(sums.hess, self.fixed_scale_shape)
        )
        self.has_gradients = True
        self.gradients_host = False
        self.round_epoch += 1

    def class_schedule(
        self,
        n_classes: Int,
        eligibility: BatchEligibility,
        requested_batch: Int = 0,
    ) raises -> ClassSchedule:
        """How a softmax round's classes may be grouped on this device.

        The companion of `histogram_plan`, from the same profile and the same
        dataset shape: the class grouping needs a resolved tile count and the
        device's threadgroup memory, and planning it here is what keeps it
        from being derived a second time somewhere else. The workload is the
        round's *root*, because that is the level a batch is formed at.

        Its default is the sequential path -- one class at a time, exactly
        what this trainer does today -- unless a caller or
        `MOJOTREES_GPU_CLASS_BATCH` asks for more. Nothing is allocated here;
        the schedule is a value, and `GpuClassBatch.for_plan` is what spends
        against it.
        """
        return plan_class_schedule(
            profile_from_caps(self.caps),
            DeviceHistogramCapabilities.portable(),
            build_kernel_features(),
            HistogramWorkload.node(
                self.n_rows, self.n_rows, len(self.active), self.n_bins
            ),
            self.n_features,
            n_classes,
            eligibility,
            self.tiling.strategy,
            self.spec_level,
            self.part_capacity,
            0,
            requested_batch,
        )

    def fill_batched_gradients(
        mut self, mut batch: GpuClassBatch, slot: Int
    ) raises:
        """One batched class's gradients into the histogram buffers.

        The batched counterpart of `fill_softmax_gradients_device`, and
        deliberately the same four statements: the class's Float32 gradients
        and hessians arrive in this builder's buffers, the fixed-point scales
        that quantize them are set, and the round epoch advances because a
        new gradient set invalidates every cached histogram.

        What differs is only where the two come from. In the sequential form
        each class runs its own reduction and its own readback, and a
        readback drains the queue; here `GpuClassBatch.refresh_scales` has
        already reduced every class of the batch in one launch and read the
        partials back once, so this class's scale is looked up rather than
        waited for. The numbers are the same either way: the batched
        reduction uses the same blocks, the same grid stride, and the same
        ascending Float64 host fold per class as the single-class one, so a
        class's scale does not depend on the batch it was reduced in.

        The plane itself is copied rather than pointed at
        (`GpuClassBatch.scatter_slot`). This builder owns its gradient
        buffers for the whole session and every enqueue below reads them, so
        adopting a pointer into someone else's allocation would make every
        later build depend on that allocation outliving it. The copy is
        `2 * 4 * n_rows` bytes, device to device, once per class per round --
        against a histogram pass that reads `n_rows * n_features` bins per
        node -- and it buys the removal of one host synchronization per
        class, which is the cost the batch exists to remove. That trade has
        not been measured on any device.

        `set_scale_shape` does not reach this path, and the boundary is real
        rather than an oversight. The scales come from `GpuClassBatch`, which
        derives its own through `gpu_objectives_native.device_fixed_scale`
        and so gets `DEFAULT_SCALE_SHAPE` -- the same shape this builder
        defaults to, so the shipping behavior of the two agrees. Asking this
        builder for `SCALE_SHAPE_ARBITRARY` and then adopting a batched
        class's scale would silently get the power-of-two arm anyway. The A/B
        this arm exists for is a single-output one; a multiclass A/B needs
        the shape threaded through `gpu_multiclass_batch.refresh_scales`,
        which is a file this lane does not own.
        """
        if batch.n_rows != self.n_rows:
            raise Error("class batch and histogram builder disagree on n_rows")
        batch.scatter_slot(slot, self.grad_dev, self.hess_dev)
        self.g_scale = batch.scale_of(slot)
        self.h_scale = batch.hess_scale_of(slot)
        self.has_gradients = True
        self.gradients_host = False
        self.round_epoch += 1

    def update_raw_device(
        mut self,
        mut state: GpuObjectiveState,
        values: List[Float64],
        learning_rate: Float64,
        k: Int = 0,
    ) raises:
        """Advance the device raw scores by the tree just grown, from the
        leaf ranges it left behind. Call after `grow_tree_gpu` returns and
        before the next `begin_tree`."""
        state.update_raw_ranges(
            self.ctx, self.rows, values, learning_rate, k
        )

    def begin_tree(mut self, bag: List[Int] = []) raises:
        """Seed the tree's active rows and make the root (node 0) own all of
        them.

        Unbagged, the root range is the identity permutation, written by a
        kernel so a tree costs no host-to-device row transfer at all. With a
        non-empty `bag`, the bag's rows are staged in the caller's order and
        the rows left out are simply not inside the root range: no sentinel
        leaf id, no per-node filtering, and no cost for a row this tree
        ignores. See gpu_active_rows.mojo.
        """
        self.rows.begin_tree(bag)

    def apply_split(
        mut self,
        feature: Int,
        threshold_bin: Int,
        parent: Int,
        left: Int,
        right: Int,
        missing_bin: Int = -1,
        default_left: Bool = False,
        is_categorical: Bool = False,
        cat_bitset: CatBitset = cat_empty(),
        expected_left: Int = -1,
    ) raises:
        """Reassign rows of `parent` to `left`/`right` by the chosen split,
        entirely on the device: the parent's contiguous row range is stably
        partitioned into the two children's. Rows in `missing_bin` follow
        `default_left` instead of the threshold; -1 (the default) means the
        feature has no missing bin and every row goes by the threshold.

        With `is_categorical`, `cat_bitset` is the node's category set and
        `threshold_bin`/`missing_bin`/`default_left` are ignored: a row goes
        left exactly when its bin is in the set, which is what
        `Tree.goes_left` does on the host.

        `expected_left` is the left row count the caller already knows
        exactly (the grower has it from the parent histogram's integer
        counts). Passing it keeps the split fully enqueued; the default -1
        downloads the device's own count, which synchronizes. With
        `MOJOTREES_GPU_VERIFY_ROWS=1` a supplied count is checked against
        the device's anyway."""
        if feature < 0 or feature >= self.n_features:
            raise Error("split feature out of range")
        if not is_categorical and (
            threshold_bin < 0 or threshold_bin >= self.n_bins
        ):
            raise Error("split threshold bin out of range")
        if missing_bin >= self.n_bins:
            raise Error("split missing bin out of range")
        if parent < 0 or left < 0 or right < 0:
            raise Error("leaf ids must be nonnegative")
        if left > MAX_ROWS or right > MAX_ROWS or parent > MAX_ROWS:
            raise Error("leaf ids must fit in Int32")
        if left == parent or right == parent or left == right:
            raise Error(
                "child leaf ids must differ from the parent and each other"
            )
        var routing: RowRouting
        if is_categorical:
            routing = RowRouting.categorical(feature, cat_bitset)
        else:
            routing = RowRouting.numerical(
                feature, threshold_bin, missing_bin, default_left
            )
        _ = self.rows.partition(
            self.bins_dev.unsafe_ptr(),
            parent,
            left,
            right,
            routing,
            expected_left,
        )

    def enqueue_leaf(
        mut self,
        leaf: Int,
        resident_slot: Int = -1,
        subtract_from_slot: Int = -1,
    ) raises:
        """Enqueue the kernels building the histogram of the rows `leaf`
        currently owns, reading only that node's compacted row range. Does
        not transfer or synchronize. A small node gets a grid sized for its
        own rows rather than for the dataset; see gpu_active_rows.mojo.

        `resident_slot` names where the result goes: -1, the default, is this
        builder's own single-node output buffer, and a nonnegative value is
        that slot of the resident frontier pool (`enqueue_resident_leaf`,
        which is the entry point that checks the slot is live). Only the
        destination pointer differs, which is the point: a resident histogram
        is built by the same kernel at the same launch geometry over the same
        rows as the shipping one, so residency cannot be a second histogram
        implementation whose cost has to be measured on its own. The two
        calls below are written out rather than sharing one because the
        destination's origin is part of its type, so no single variable holds
        either pointer.

        `subtract_from_slot`, valid only alongside a `resident_slot`, is a
        second live slot to subtract this histogram from as it is built. It
        is the sibling subtraction of `enqueue_resident_subtract` folded into
        the build, so a split costs one kernel where it used to cost two;
        `enqueue_resident_leaf_subtracting` is the entry point that checks
        the two slots may be subtracted at all.
        """
        if not self.has_gradients:
            raise Error("call upload_gradients before build_leaf")
        if leaf < 0 or leaf > MAX_ROWS:
            raise Error("leaf id must be nonnegative and fit in Int32")
        if resident_slot >= 0 and len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        if subtract_from_slot >= 0 and resident_slot < 0:
            raise Error(
                "a fused subtraction needs a resident destination slot"
            )
        if subtract_from_slot >= 0 and subtract_from_slot == resident_slot:
            raise Error("a histogram slot cannot be subtracted from itself")

        var n_slots = len(self.active)
        # Deliberately not gated by `require_histogram_launchable`, and this
        # is the reason rather than an oversight. This runs once per node, and
        # the gate allocates a `List[Int]` of required primitives per call. It
        # would also have nothing new to check: `n_slots` is `len(self.active)`,
        # which `set_features` gated when it last changed, and a node's tiling
        # only ever shrinks the full-matrix one this builder opened with, since
        # a node holds a subset of the rows. Every dimension here was bounded
        # by a gate that already ran.
        var tiling = self.rows.range_tiling(
            self.caps,
            leaf,
            n_slots,
            self.tiling.strategy,
            self.part_capacity,
        )
        if resident_slot >= 0:
            var cells = 3 * self.n_features * self.n_bins
            var pool = self.batcher[0].out_dev.unsafe_ptr()
            # Where the sibling to derive sits relative to this build's own
            # slot, in Int32 words. Both are slots of one pool buffer, so the
            # kernels reach the second through this offset rather than
            # through a second pointer into the same allocation.
            var sub_offset = (subtract_from_slot - resident_slot) * cells
            self.rows.enqueue_range_histogram(
                tiling,
                leaf,
                self.bins_dev.unsafe_ptr(),
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                pool.unsafe_offset(resident_slot * cells),
                self.part_dev.unsafe_ptr(),
                n_slots,
                Float32(self.g_scale),
                Float32(self.h_scale),
                sub_offset,
                subtract_from_slot >= 0,
            )
            return
        self.rows.enqueue_range_histogram(
            tiling,
            leaf,
            self.bins_dev.unsafe_ptr(),
            self.grad_dev.unsafe_ptr(),
            self.hess_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            self.out_dev.unsafe_ptr(),
            self.part_dev.unsafe_ptr(),
            n_slots,
            Float32(self.g_scale),
            Float32(self.h_scale),
        )

    def download_raw(mut self) raises:
        """Copy the fixed-point histogram into pinned host memory and wait.
        One host synchronization per node, not one per plane."""
        self.ctx.enqueue_copy(
            dst_ptr=self.host_out.unsafe_ptr(), src_buf=self.out_dev
        )
        self.ctx.synchronize()

    def histogram_from_host(self) raises -> Histogram:
        """Convert the downloaded fixed-point planes into the Float64
        `Histogram`. Host work only; call after `download_raw`.

        Which path this is on, and which it is not
        ------------------------------------------
        This runs once per node on the **host-search** path only. The
        device-resident split search never calls it: the histogram stays in
        device memory and is scanned there, and only a 136-byte record
        crosses, which is the whole point of that path. So this conversion
        is a cost of the host scan and of nothing else.

        That matters for how a change here may be described, and what it
        used to say was that our headline benchmark sat one row away from
        this cost: at 1,000,000 x 50, 255 bins and 31 leaves
        `normalized_split_work` is exactly 50,000,000.0, which cleared the
        crossover by floating-point equality, so one row fewer, one feature
        fewer, or any `feature_fraction` below 1 fell back to the host scan
        and started paying this per node.

        **THAT KNIFE EDGE IS GONE, AND NOT BECAUSE IT MOVED.** The crossover
        was measured on 2026-08-16 at four shapes from 5.0M to 70.0M
        normalized work, arms interleaved, and it does not exist: the device
        search wins at every one and wins by MORE at the smaller shapes
        (1.85x at 5.0M against 1.29x at 70.0M). Both thresholds were
        withdrawn, so `gpu_split_policy` sends every eligible shape on
        measured hardware to the device search, and this function is off the
        automatic path at every size rather than above one.

        What still reaches it: a fit the policy declines for ELIGIBILITY --
        `TreeParams.extra`, `feature_fraction_bylevel`, a resident frontier
        that does not fit, or hardware nobody has measured -- plus
        `train_gpu_sparse`, which has its own host scan, and
        `distributed_gpu`, whose histogram exchange is host arithmetic by
        construction. Those are the callers a change here should be weighed
        against, and none of them is a size.

        On the host-scan path the cost is real. It covers
        `3 * n_features * n_bins` cells, and the (now deleted) hybrid
        scheduler's calibrated cost model priced the conversion at
        `convert_nanos_per_kcell = 10024`, about 10 ns per cell, against a
        modeled device fixed cost per node of roughly 263 microseconds
        (bench/results/apple_m4_hybrid_costs_2026-08-15.md).
        Nothing here has been measured by this lane, and no speedup is
        claimed on either path; what changed is only the shape of the loop.

        Why the shape may change freely. Every output cell is a function of
        exactly one input cell: `Float64(Int32) * (1.0 / scale)` for the two
        Float64 planes and a widening integer conversion for the count
        plane. There is no accumulation, no reassociation, and no
        cross-cell dependence, so the same value lands in the same slot
        whatever the lane width and whatever the task count. That is a
        property of the arithmetic, not of the schedule, which is what makes
        both the SIMD body and the block split exactness-neutral by
        construction rather than by measurement.

        Two things do change. The body loads `SIMD_LANES` cells at a time
        and falls back to a scalar tail, which applies at every size. The
        block split goes through `dispatch_rows` under the ordinary
        `MOJOTREES_NUM_WORKERS` / `MOJOTREES_PARALLEL_MIN_OPS` contract, and
        that contract is deliberately not bent here: the work estimate is
        the honest `3 * hist_size` cell conversions, so a 50-feature, 255-bin
        node (38,250 ops) sits below the default 65,536-op grain and stays
        serial exactly as it did before. Forcing workers is how the parallel
        arm is reached at that shape. Moving the grain is a measured
        decision and is not taken here.

        **The constant-hessian specialization does not reach this
        conversion, and it is worth saying so rather than leaving it to be
        inferred.** When the hessian plane is elided, it is elided from the
        *accumulation*: the kernels stop accumulating it and refill it in
        device memory before this function ever sees the buffer. The
        download therefore still moves three planes and this loop still
        converts three, unchanged, byte for byte. It could not be otherwise
        without changing the device-side layout of `out_dev`, which
        `gpu_split_search.mojo` scans in place across all three planes and
        which the resident pool in `gpu_leaf_batching.mojo` strides by, and
        neither is this lane's to change. The hessian plane also sits in the
        middle of `[grad | hess | count]`, so even skipping it in the copy
        would need two transfers rather than one, against the "one host
        synchronization per node, not one per plane" rule `download_raw`
        exists to keep.
        """
        var hist_size = self.n_features * self.n_bins
        var g = _zeroed_f64(hist_size)
        var h = _zeroed_f64(hist_size)
        var c = _zeroed_int(hist_size)
        # Reciprocal once, multiply per cell -- and under
        # `SCALE_SHAPE_POW2`, the default, this whole conversion is *exact*.
        # `g_scale` is a power of two, so `1.0 / g_scale` is representable in
        # Float64 with no rounding, an Int32 cell is representable exactly,
        # and a Float64 times a power of two is exact. Under
        # `SCALE_SHAPE_ARBITRARY` neither the reciprocal nor the product is,
        # and every cell carried both roundings. Trading a division per cell
        # for a multiply is why the reciprocal is hoisted; that it now costs
        # nothing in accuracy to hoist it is new.
        var g_inv = 1.0 / self.g_scale
        var h_inv = 1.0 / self.h_scale
        var gp = g.unsafe_ptr()
        var hp = h.unsafe_ptr()
        var cp = c.unsafe_ptr()
        var src = self.host_out.unsafe_ptr()

        def decode(start: Int, end: Int) {imm}:
            comptime W = SIMD_LANES
            var i = start
            while i + W <= end:
                gp.unsafe_store(
                    i,
                    src.unsafe_load[width=W](i).cast[DType.float64]() * g_inv,
                )
                hp.unsafe_store(
                    i,
                    src.unsafe_load[width=W](hist_size + i).cast[
                        DType.float64
                    ]()
                    * h_inv,
                )
                cp.unsafe_store(
                    i,
                    src.unsafe_load[width=W](2 * hist_size + i).cast[
                        DType.int
                    ](),
                )
                i += W
            while i < end:
                gp.unsafe_store(i, Float64(src.unsafe_load(i)) * g_inv)
                hp.unsafe_store(
                    i, Float64(src.unsafe_load(hist_size + i)) * h_inv
                )
                cp.unsafe_store(i, Int(src.unsafe_load(2 * hist_size + i)))
                i += 1

        dispatch_rows(decode, hist_size, 3 * hist_size)
        return Histogram.from_planes(g^, h^, c^, self.n_features, self.n_bins)

    def build_leaf(mut self, leaf: Int) raises -> Histogram:
        """Build the histogram of the rows currently assigned to `leaf`, over
        the currently active feature set (every feature unless `set_features`
        narrowed it). The returned histogram always has the dataset's full
        `n_features * n_bins` shape; inactive features' slices are zero."""
        self.enqueue_leaf(leaf)
        self.download_raw()
        return self.histogram_from_host()

    def build_leaf_host_replica(
        self,
        mut out: Histogram,
        mut fixed_scratch: List[Int32],
        data: BinnedMatrix,
        rows: List[Int32],
        row_start: Int,
        row_count: Int,
    ) raises:
        """The host build of one node's histogram through the device's exact
        fixed-point pipeline: this round's staged Float32 gradients, this
        round's scales, Int32 accumulation, the same dequantization. Same
        output contract as `build_leaf` — full dataset shape, the active
        feature set accumulated, inactive slices zero.

        `rows[row_start : row_start + row_count]` must be the node's rows in
        the device's compacted order (a host mirror of the permutation, or a
        `download_rows` snapshot; see docs/design/HYBRID_TRAINING.md §3).
        Only legal while the staged gradients are this round's host uploads:
        on the device-objective paths (`gradients_host` False) the Float32
        values the kernels read never existed host-side, and this raises
        rather than accumulating a stale round.

        Whether the result is bit-identical to `build_leaf`'s is a claim
        about this device's multiply-and-round, never assumed. It is what
        `tests/test_host_replica.mojo` asserts, and this builder is the
        oracle side of the CPU/GPU equivalence docs/ARCHITECTURE.md names.
        """
        if not self.has_gradients or not self.gradients_host:
            raise Error(
                "host replica build needs this round's host-staged gradients"
            )
        if (
            data.n_rows != self.n_rows
            or data.n_features != self.n_features
            or data.n_bins != self.n_bins
        ):
            raise Error("binned matrix does not match this builder")
        # The two halves of the one staging arena. Same values the two
        # separate staging buffers held: `stage_gradients` writes the same
        # Float32 to the same indices of each plane, only adjacently.
        var stage = self.stage_gh.unsafe_ptr()
        var grad_span = Span(unsafe_ptr=stage, length=self.n_rows)
        var hess_span = Span(
            unsafe_ptr=stage.unsafe_offset(self.n_rows), length=self.n_rows
        )
        build_histogram_subset_replica_into(
            out,
            fixed_scratch,
            data,
            grad_span,
            hess_span,
            rows,
            row_start,
            row_count,
            self.g_scale,
            self.h_scale,
            self.active,
            self.rows.constant_hessian,
        )

    def readback_range(
        mut self, begin: Int, count: Int, mut out: List[Int32]
    ) raises:
        """One node's rows — `rows_dev[begin : begin + count]` — into `out`,
        resized to `count`, in the device's compacted order.

        The per-range readback docs/design/HYBRID_TRAINING.md §3 assumed was
        not expressible: `DeviceBuffer.create_sub_buffer` views a window of
        the row buffer without allocating or launching anything, and
        `enqueue_copy` moves exactly that window, so a host build of a
        four-row leaf reads back sixteen bytes and not the whole permutation.
        One synchronize, which also drains the partition that produced the
        window, so the caller may take the range from `range_of` immediately
        after `split`.

        A whole-permutation snapshot (`snapshot_rows`) is the same copy at
        `begin=0, count=n_rows`.
        """
        if begin < 0 or count < 0 or begin + count > self.n_rows:
            raise Error("readback window is outside the active-row buffer")
        if len(out) != count:
            out.resize(count, Int32(0))
        if count == 0:
            return
        var window = self.rows.rows_dev.create_sub_buffer[DType.int32](
            begin, count
        )
        self.ctx.enqueue_copy(dst_ptr=out.unsafe_ptr(), src_buf=window)
        self.ctx.synchronize()

    def snapshot_rows(mut self, mut out: List[Int32]) raises:
        """The whole active-row permutation into `out`, resized to `n_rows`.

        The whole-permutation snapshot of docs/design/HYBRID_TRAINING.md §3,
        which the grower no longer needs (`readback_range` moves one node's
        window instead) and tests still use: one whole-buffer copy through
        the row machinery's pinned staging buffer, one synchronize, one
        memcpy into the caller's list. `GpuActiveRows.download_rows` answers
        the same question but builds its result a row at a time, which
        measured at an order of magnitude over the copy itself -- fine for
        the tests it serves, and the wrong price for a cost the scheduler
        charges to a single leaf.
        """
        self.ctx.enqueue_copy(
            dst_ptr=self.rows.host_rows.unsafe_ptr(),
            src_buf=self.rows.rows_dev,
        )
        self.ctx.synchronize()
        if len(out) != self.n_rows:
            out.resize(self.n_rows, Int32(0))
        unsafe_memcpy(
            dest=out.unsafe_ptr(),
            src=self.rows.host_rows.unsafe_ptr(),
            count=self.n_rows,
        )

    # -- batched multi-leaf construction ----------------------------------

    def batching_live(self) -> Bool:
        """Whether this builder holds a batched launcher."""
        return len(self.batcher) > 0

    def histogram_plan(self, node_rows: Int) raises -> HistogramPlan:
        """The launch policy's plan for a node of `node_rows` rows on this
        device, at the level this run asked for.

        The one bridge from the builder's state to
        `apple_histogram_policy`: the profile is this builder's own
        `DeviceCaps`, the workload is this builder's dataset shape and active
        feature count, and the partial budget is the buffer this builder
        already allocated. Nothing is opened, allocated, or enqueued.

        `DeviceHistogramCapabilities.portable()` is the honest answer for a
        device that has reported nothing beyond being launchable, which is
        every device here: `query_device_caps` reads three attributes and
        neither a subgroup width nor a wide-load result is among them.
        """
        return derive_histogram_plan(
            profile_from_caps(self.caps),
            DeviceHistogramCapabilities.portable(),
            build_kernel_features(),
            HistogramWorkload.node(
                self.n_rows, node_rows, len(self.active), self.n_bins
            ),
            self.tiling.strategy,
            self.spec_level,
            self.part_capacity,
        )

    def open_batching(mut self, pool_slots: Int = 0) raises -> Bool:
        """Allocate the batched launcher, at most once per builder.

        Returns True when a batcher is live afterwards. Refuses to allocate
        anything unless `MOJOTREES_GPU_HIST_SPECIALIZATION=batched` asked for
        the batched level, so the default path pays neither the histogram
        slot pool nor the second partial buffer.

        The pool depth is `MOJOTREES_GPU_BATCH_SLOTS` capped by what
        `BATCH_POOL_BUDGET_BYTES` buys at this dataset's shape: a slot is a
        full-width `3 * n_features * n_bins` Int32 histogram, so a wide
        dataset gets fewer slots rather than a surprise allocation. A budget
        that does not buy two slots leaves batching off, because a batch of
        one is what the single-leaf path already does.
        """
        if self.spec_level < SPEC_LEVEL_BATCHED:
            return False
        if len(self.batcher) > 0:
            return True
        var want = pool_slots if pool_slots > 0 else env_batch_slots()
        var affordable = slots_for_budget(
            BATCH_POOL_BUDGET_BYTES, self.n_features, self.n_bins
        )
        var slots = want if want < affordable else affordable
        if slots < 2:
            return False
        # The one launch on this builder whose selected variants differ from
        # the default, and so the one place the specialization gate has teeth:
        # a backend that has never run the batched kernel refuses it here,
        # before the slot pool is allocated, rather than at the launch. The
        # `selected` argument is what makes this different from the two gates
        # above, which pass the conservative `KernelFeatures.none()`.
        require_histogram_launchable(
            self.contract,
            self.caps,
            self.tiling,
            len(self.active),
            self.n_bins,
            build_kernel_features(),
            KernelFeatures(False, False, True),
        )
        self.batcher.append(
            GpuLeafBatcher(
                self.ctx,
                self.caps,
                self.n_rows,
                self.n_features,
                self.n_bins,
                slots,
                self.part_capacity,
                DEFAULT_MAX_ITEMS,
                1,
            )
        )
        self.batch_feat_stamp = -1
        return True

    # -- device-resident frontier -----------------------------------------
    #
    # The batched launcher's slot pool, used as *storage for a frontier*
    # rather than as a wider launch. The two uses share every buffer and
    # every kernel; what differs is who reads a slot. `build_leaves`
    # downloads a slot and throws it away, so it holds a slot only across
    # one launch. The device split search reads a slot where it lies, so a
    # leaf's histogram is worth keeping until that leaf splits — and once a
    # parent's histogram is still on the device when its children are built,
    # `enqueue_subtract` derives the larger child instead of accumulating it,
    # which is the same halving of histogram work the host-search grower gets
    # from `subtract_histogram`. See `train_gpu._grow_tree_gpu_device_search`.

    def resident_slots(self) -> Int:
        """Slots the resident frontier pool holds, or zero when none is
        open. A grower asks before it commits to the resident path."""
        return self.batcher[0].pool.capacity if len(self.batcher) > 0 else 0

    def resident_frontier_fits(self, want_slots: Int) raises -> Bool:
        """Whether the resident pool can hold the complete leaf frontier.

        The automatic split policy asks this before selecting device search;
        `open_resident` asks it again before allocation.  Keeping the budget
        arithmetic here prevents policy and allocation from drifting.
        """
        if want_slots < 2:
            return False
        if len(self.batcher) > 0:
            return self.batcher[0].pool.capacity >= want_slots
        return (
            slots_for_budget(
                RESIDENT_POOL_BUDGET_BYTES, self.n_features, self.n_bins
            )
            >= want_slots
        )

    def oblivious_level_fits(self, n_children: Int) raises -> Bool:
        """Whether an open batcher can build a whole level of `n_children` in
        one batch, and hold a slot for every one of them.

        Two questions and both have to answer yes, because an oblivious level
        builds every child from its own rows: `max_items` bounds how many item
        rows one batch covers, and the pool bounds how many histograms can be
        live at once. The widest level of a depth-`d` tree has `1 << d` children
        and that is also the tree's leaf count, so the two bounds are asked
        against the same number.

        A builder with no batcher open answers False rather than guessing at the
        budget: this is asked by `gpu_resident_round.oblivious_device_supported`
        *after* `open_resident` has run, so "not open" here means the pool
        declined, and a mode whose census depends on the batch width should not
        be routed to on the strength of an allocation nobody made.

        The item bound is the one that is easy to get wrong and expensive to get
        wrong quietly; see `gpu_leaf_batching.OBLIVIOUS_MAX_ITEMS`.
        """
        if n_children < 2:
            return False
        if len(self.batcher) == 0:
            return False
        return (
            self.batcher[0].max_items >= n_children
            and self.batcher[0].pool.capacity >= n_children
        )

    def open_resident(
        mut self, want_slots: Int, max_items: Int = DEFAULT_MAX_ITEMS
    ) raises -> Bool:
        """Allocate a slot pool deep enough to hold `want_slots` leaves at
        once, and report whether the resident path is available afterwards.

        `max_items` is the widest batch the pool's batcher will ever be asked to
        build, and it is a parameter rather than a constant because
        `grow_policy = oblivious` needs it to be 64 and the default is 32. That
        is not a preference: a depth-6 level's last generation has 64 children,
        at the default bound it needs two batches, and
        `gpu_resident_round.oblivious_launch_census(6, batch_max_items=32)` lands
        the tree at exactly 64 command buffers -- on the queue-depth knee rather
        than under it. See `gpu_leaf_batching.OBLIVIOUS_MAX_ITEMS`. The leaf-wise
        plane passes the default and is unchanged.

        All or nothing, and deliberately so. A leaf-wise frontier holds a
        slot per live leaf for the whole tree, so a pool one slot short does
        not degrade gracefully: it strands a leaf whose histogram has to be
        rebuilt from its rows, which is the accumulation the residency was
        bought to avoid. Rather than carry a second, slower path through the
        grower for a case that only arises on very wide datasets, this
        answers False and the caller keeps the incremental search it already
        has.

        Unlike `open_batching` this is not gated on
        `MOJOTREES_GPU_HIST_SPECIALIZATION=batched`, and the reason is that it
        selects no specialized kernel. That level gates a *launch policy*
        nothing has benchmarked (is a wide batch faster than several
        single-leaf builds?); residency asks where a histogram lives, and
        `enqueue_resident_leaf` builds one with the kernel that already
        ships. So the specialization half of the gate is inert here, exactly
        as it is for `set_features`, and only the geometry and primitive
        checks below have anything to say.

        A pool already open is reused when it is deep enough, since a builder
        holds at most one batcher for its whole life and a tree's slots are
        released back to it rather than reallocated.

        That last paragraph described an intention rather than the code until
        this guard was added. `open_resident` is called once per tree from
        `_grow_tree_gpu_device_search`, and it appended a new `GpuLeafBatcher`
        every time, while `enqueue_leaf` reads `self.batcher[0]` and nothing
        reads any other. So a hundred-round fit allocated a hundred slot
        pools, used the first, and never freed or read the other ninety-nine,
        each one a full-width `3 * n_features * n_bins` Int32 slot per leaf
        plus its own partial buffer, on the default device path. The guard is
        the same one `open_batching` has always had; the two are written the
        same way now so the difference cannot come back.

        The depth test is what makes reuse safe rather than merely cheap. A
        pool shallower than this tree's frontier would strand a leaf, which is
        precisely the failure the all-or-nothing contract above exists to
        refuse, so a request deeper than the open pool declines rather than
        reusing it. In a single fit `want_slots` is `params.num_leaves` and
        does not move, so the deeper-request branch is unreachable today; it
        is written because a caller that varied the leaf budget between trees
        would otherwise get a silently stranded leaf instead of a decline.
        """
        if not self.resident_frontier_fits(want_slots):
            return False
        if len(self.batcher) > 0:
            # Both dimensions, and the item bound is not decoration: a batcher
            # opened for the leaf-wise plane holds 32 items and an oblivious
            # tree would silently split its widest level into two batches,
            # which is the one thing the census cannot absorb. A shallower
            # request reuses; a wider one declines, exactly as a deeper slot
            # request does.
            return (
                self.batcher[0].pool.capacity >= want_slots
                and self.batcher[0].max_items >= max_items
            )
        require_histogram_launchable(
            self.contract,
            self.caps,
            self.tiling,
            len(self.active),
            self.n_bins,
            build_kernel_features(),
        )
        self.batcher.append(
            GpuLeafBatcher(
                self.ctx,
                self.caps,
                self.n_rows,
                self.n_features,
                self.n_bins,
                want_slots,
                self.part_capacity,
                max_items,
                1,
            )
        )
        self.batch_feat_stamp = -1
        return True

    def resident_stamp(self) raises -> Int:
        """The stamp this builder's slots are acquired under right now: the
        round's scales and the tree's feature set, in the one encoding
        `gpu_leaf_batching` defines. Two slots may only be subtracted when
        these agree, which within one tree they always do."""
        return subtraction_stamp(self.round_epoch, self.feat_epoch)

    def acquire_resident(mut self, node: Int) raises -> Int:
        """A pool slot owned by `node`, or -1 when the pool is full."""
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        return self.batcher[0].pool.acquire(node, self.resident_stamp())

    def release_resident(mut self, slot: Int) raises:
        """Give one leaf's slot back."""
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        self.batcher[0].pool.release(slot)

    def release_resident_all(mut self):
        """Give every slot back, which is what the end of a tree means: no
        histogram from one tree is readable by the next, since the next tree
        repartitions every row. Cheap and idempotent, so a grower calls it
        when it starts as well, and an error that escapes mid-tree cannot
        leak the frontier's slots into the following one."""
        if len(self.batcher) > 0:
            self.batcher[0].pool.release_all()

    def reown_resident(mut self, slot: Int, node: Int) raises:
        """Hand a live slot to `node`. The bookkeeping half of an in-place
        subtraction, whose result is the sibling's histogram sitting in the
        parent's slot."""
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        self.batcher[0].pool.reassign(slot, node)

    def enqueue_resident_leaf(mut self, node: Int, slot: Int) raises:
        """Build `node`'s histogram into pool slot `slot` and leave it there.
        Enqueues only: no transfer and no synchronization.

        `enqueue_leaf`'s kernel, its Apple-aware launch geometry, and its
        rows, aimed at a pool slot instead of the builder's own output
        buffer. Nothing about the accumulation moves, so a resident histogram
        is the one the single-leaf path would have produced, bin for bin, and
        the resident grower's cost is that path's cost plus a subtraction.

        Not the batched multi-leaf kernel, deliberately. A resident frontier
        builds exactly one leaf per split (the other is subtracted), so a
        batch would be a batch of one: it would pay `_stage_plan`'s two
        copies and the batched kernel's per-item indirection to launch the
        same work, and it would make the resident path's histograms a
        different kernel's output from the host-search path's, which is a
        variable a comparison between them does not need. The slot pool is
        what this borrows from `gpu_leaf_batching`; the launch is not.
        """
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        self.batcher[0].pool.check_live(slot)
        self.enqueue_leaf(node, resident_slot=slot)

    def enqueue_resident_leaf_subtracting(
        mut self, node: Int, slot: Int, parent_slot: Int
    ) raises:
        """Build `node` into `slot` and derive its sibling into `parent_slot`
        in the same launch: `enqueue_resident_leaf` followed by
        `enqueue_resident_subtract(parent_slot, slot, parent_slot)`, with the
        subtraction folded into the kernel that writes the build.

        A split is where a device-resident frontier spends its fixed cost.
        Every kernel launch on this machine's GPU costs about twenty
        microseconds of enqueue whatever it does, and the standalone
        subtraction is a launch that reads and writes every cell of two whole
        slots -- `3 * n_features * n_bins` words apiece, a size set by the
        dataset and not by the node -- to add nothing to the histogram the
        build already computed. Folding it in costs the build one extra
        store per cell it was already writing, or three extra global atomics
        per populated bin per block on the atomic path, and it drops the
        launch. Deep in a tree, where nodes are small and that slot-sized
        pass dominates the build itself, that is the difference worth having.

        Nothing about the result moves. Both operands are fixed-point Int32
        under one scale, so the difference is exact, which is the same reason
        `enqueue_resident_subtract` and the host `subtract_histogram` are
        exact; the stamps are checked here exactly as that path checks them;
        and the cells the fused form skips (a narrowed feature set's inactive
        slices, and bins this node put no rows in) are cells where the
        subtrahend is zero. The two paths therefore leave the pool holding
        the same words, which `test_gpu_active_rows` asserts bin for bin.

        Enqueues only: no transfer and no synchronization.
        """
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        self.batcher[0].pool.check_live(slot)
        self.batcher[0].pool.check_subtractable(parent_slot, slot)
        self.enqueue_leaf(node, resident_slot=slot, subtract_from_slot=parent_slot)

    def open_resident_tables(mut self, num_leaves: Int) raises -> Bool:
        """Open, or reuse, the device tree tables a device-owned growth loop
        commits into, and report whether they are available.

        The same shape and the same reasoning as `open_resident`, which is
        deliberate: both are per-fit state whose dimensions come from the leaf
        budget and not from the data, both are held in a one-element list so
        that "not open" is a length rather than a sentinel, and both reuse an
        already-open instance when it is deep enough. `DeviceTreeTables` costs
        eleven device allocations and one synchronization to construct, so
        building one per tree would put eleven allocations and a drain back
        into every tree. The allocations are the part that is real work; the
        drain is an ordering point and, under `docs/GPU_PORTABILITY.md`
        section 6.1.1, is not by itself a time. Neither is a round trip, so
        nothing here should be quoted in seconds. Reusing is right on
        allocation grounds and on the ordinary grounds that per-fit state
        belongs on a per-fit object.

        It lives on the builder rather than on the growth loop for one
        practical reason: the builder is the object a fit already threads
        through every tree, and giving the new loop somewhere to keep per-fit
        state without adding a parameter to the shipping grower is what keeps
        that grower to a single new call site.

        A shallower request reuses; a deeper one declines rather than
        reusing, since tables sized for a smaller budget would overflow and
        the commit kernel would report `TREE_OVERFLOW` rather than corrupt
        anything. In a single fit `num_leaves` does not move, so the deeper
        branch is unreachable today and is written so that a caller who
        varied the budget between trees gets a decline instead of a stopped
        tree.
        """
        if num_leaves < 2:
            return False
        if len(self.resident_tables) > 0:
            return self.resident_tables[0].leaf_capacity >= num_leaves
        self.resident_tables.append(
            DeviceTreeTables(
                self.ctx, num_leaves, self.n_features, self.missing_bin.copy()
            )
        )
        return True

    def enqueue_desc_child(mut self, max_rows: Int) raises:
        """Build the child named by the step descriptor into the pool slot the
        step descriptor names, subtracting it from its sibling's slot.

        The device-owned counterpart of `enqueue_resident_leaf_subtracting`,
        and the place where the second holdout of `gpu_tree_tables` is
        closed. That method takes a node, a destination slot and a slot to
        subtract from, and the caller gets all three from the *host* slot pool
        through `acquire_resident` and `reown_resident`. On this path the
        device's `slot_owner` vector is the authority instead: the commit
        kernel takes the lowest free slot, reassigns the parent's, and writes
        both into the descriptor, so there is nothing left for the host pool
        to decide and it is not consulted. `pool.check_live` is therefore not
        called here, and could not be: the host pool holds no owners at all on
        this path, because nothing on it ever acquired.

        That is a real loss of a check and it is worth naming. On the host
        path a slot handed to two live leaves would be caught by
        `HistogramSlotPool.check_live` at the launch that did it. Here the
        equivalent check is `TreeTablesSnapshot.check_invariants`, which
        asserts that no two live leaves share a histogram slot and that the
        slot pool and the frontier agree about every owner; the difference is
        that it runs once per tree, at the download, rather than once per
        split. That is the trade the whole lane is: checks that cost a
        synchronization move to the one synchronization there is.

        `max_rows` is an upper bound on the built child's rows, which the
        active row count supplies. See `GpuActiveRows.enqueue_desc_histogram`
        for why a bound is enough and for why this path always takes the
        atomic strategy.

        Enqueues only: no transfer and no synchronization.
        """
        if not self.has_gradients:
            raise Error("call upload_gradients before enqueue_desc_child")
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        var n_slots = len(self.active)
        var pool_slots = self.batcher[0].pool.capacity
        var pool = self.batcher[0].out_dev.unsafe_ptr()
        self.rows.enqueue_desc_histogram(
            pool_slots,
            max_rows,
            self.bins_dev.unsafe_ptr(),
            self.grad_dev.unsafe_ptr(),
            self.hess_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            pool,
            n_slots,
            Float32(self.g_scale),
            Float32(self.h_scale),
            self.caps,
        )

    def enqueue_desc_partition(mut self, max_count: Int) raises:
        """Partition the window the step descriptor names, by the split the
        step descriptor names.

        `apply_split` with every argument removed, because on this path a
        kernel chose all of them. It is the first holdout of
        `gpu_tree_tables` closed, and it is the one that actually blocked a
        one-wait-per-tree loop: everything else the host does per split is
        bookkeeping it could in principle defer, while the partition genuinely
        needs the chosen split before it can run.

        No range table is updated, because there is no host range table on
        this path: the frontier's windows live in the device tree tables and
        the commit kernel moves them. `GpuActiveRows.ranges` therefore still
        describes the root and nothing else for the whole of a device-owned
        tree, and `range_of`, `check_frontier`, `download_range` and
        `snapshot_rows` must not be used against it. That is stated here
        rather than defended against, because defending against it would mean
        maintaining a second copy of the windows on the host, which is exactly
        the bookkeeping the lane removes.
        """
        self.rows.enqueue_partition_desc(
            self.bins_dev.unsafe_ptr(), max_count
        )

    # --- The device-owned tree seam ---------------------------------------
    #
    # Six forwarders that belong, conceptually, to `gpu_resident_round.mojo`:
    # they are the growth loop's launches, in the loop's order, and this
    # struct has no opinion about any of them. They are methods here because
    # every one of them needs two different parts of this builder at once --
    # the tree tables in `resident_tables` and the step descriptor in
    # `rows.step_dev` -- and Mojo will not let a caller outside hold a mutable
    # borrow of one field of an object while passing a pointer derived from
    # another. Inside a method body the two are disjoint field borrows and the
    # call is fine. That is the whole reason, and it is a language constraint
    # rather than a design: the growth loop reads better in one place, and
    # `gpu_resident_round.mojo` is where it lives.

    def desc_tables_open(self) -> Bool:
        """Whether `open_resident_tables` has run on this builder."""
        return len(self.resident_tables) > 0

    def enqueue_desc_begin_tree(mut self, n_active: Int) raises:
        """Reset the device tree tables to a one-leaf frontier owning
        `[0, n_active)` with the root's histogram in pool slot 0.

        Slot 0 is not a convention this picks, it is what
        `HistogramSlotPool.acquire` and the commit kernel's own upward scan
        both hand out first, so a tree that starts anywhere else would have a
        frontier the commit kernel could not have produced.

        Does not synchronize; see `DeviceTreeTables.begin_tree` for the
        staging-lifetime argument that makes that safe here.
        """
        if len(self.resident_tables) == 0:
            raise Error("no device tree tables are open")
        self.resident_tables[0].begin_tree(n_active, root_slot=0, wait=False)

    def enqueue_desc_seed_root(
        mut self, mut rec_f: DeviceBuffer[DType.float32], record: Int
    ) raises:
        """Copy the root's Newton value from its search record into node 0."""
        if len(self.resident_tables) == 0:
            raise Error("no device tree tables are open")
        self.resident_tables[0].enqueue_seed_root_value(rec_f, record)

    def enqueue_desc_step(
        mut self,
        mut rec_i: DeviceBuffer[DType.int32],
        mut rec_f: DeviceBuffer[DType.float32],
        num_leaves: Int,
        max_depth: Int,
        min_data_in_leaf: Int,
    ) raises:
        """One pick-and-commit step, writing the descriptor the partition and
        the child histogram will read."""
        if len(self.resident_tables) == 0:
            raise Error("no device tree tables are open")
        self.resident_tables[0].enqueue_step(
            rec_i,
            rec_f,
            self.rows.step_dev.unsafe_ptr(),
            num_leaves,
            max_depth,
            min_data_in_leaf,
        )

    def enqueue_desc_stage_search(
        mut self,
        mut node_tbl: DeviceBuffer[DType.int32],
        slot_cells: Int,
        left_record: Int,
        right_record: Int,
    ) raises:
        """Point the searcher's two scratch records at the children's pool
        slots."""
        if len(self.resident_tables) == 0:
            raise Error("no device tree tables are open")
        self.resident_tables[0].enqueue_stage_child_search(
            node_tbl,
            self.rows.step_dev.unsafe_ptr(),
            slot_cells,
            left_record,
            right_record,
        )

    def enqueue_desc_copy_records(
        mut self,
        mut rec_i: DeviceBuffer[DType.int32],
        mut rec_f: DeviceBuffer[DType.float32],
        left_record: Int,
        right_record: Int,
    ) raises:
        """Move the two scratch records into the frontier slots that own
        them."""
        if len(self.resident_tables) == 0:
            raise Error("no device tree tables are open")
        self.resident_tables[0].enqueue_copy_records(
            rec_i,
            rec_f,
            self.rows.step_dev.unsafe_ptr(),
            left_record,
            right_record,
        )

    # --- The oblivious level seam -----------------------------------------
    #
    # Four more forwarders, and they are the whole of `enqueue_desc_child`'s
    # replacement under `grow_policy = oblivious`. The single-child build is
    # two launches per parent; these are two launches per *level*, which is the
    # difference between 176 command buffers on a depth-6 tree and 62. They sit
    # here for the same language reason the six above do: each one needs the
    # tree tables in `resident_tables`, the slot pool in `batcher`, and the row
    # state in `rows` at once, and a caller outside cannot hold a mutable
    # borrow of one field while passing a pointer derived from another.

    def stage_desc_level_plan(
        mut self, n_items: Int, max_rows: Int
    ) raises -> Int:
        """Fix the host half of the level plan for this tree, once.

        Forwards to `GpuLeafBatcher.stage_device_plan`, which stages every item
        dead and returns the packed tile count the batch's grid uses. Called
        once per tree, before the first level commit, and not again: the
        geometry is a function of the row bound, the feature width and the item
        count, and none of the three moves inside a tree.

        `n_items` is the widest level's child count, `1 << max_depth`, and not
        the first level's. Every level fills or kills the same width; see
        `gpu_tree_tables._kill_level_plan` for why the killing has to cover the
        whole staged width and not just the level's own prefix.
        """
        if not self.has_gradients:
            raise Error("call upload_gradients before staging a level plan")
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        # The batcher's feature table, and WITHOUT THIS LINE THE OBLIVIOUS
        # DEVICE PATH RETURNS A WRONG ANSWER whenever `feature_fraction < 1`.
        # Fixed 2026-08-17, found by an audit of declined optimizations rather
        # than by a test, which is the part worth remembering.
        #
        # `GpuLeafBatcher.feat_dev` is written in exactly three places: the
        # constructor, which fills it with the identity; `set_shared_features`,
        # whose only other caller is the host path's `_build_leaves_batched`;
        # and `set_item_features`, which has no caller at all. The oblivious
        # level build reaches `enqueue_device_plan_batch_fused` through none of
        # them, so `_batch_hist_atomic_kernel` read `f == slot` and both read
        # column `slot` and wrote slice `slot`. Meanwhile the split searcher's
        # own table IS set from `tree_features`, so the searcher read slice
        # `active[slot]` of a histogram written at slice `slot`, and the root,
        # built through this builder's table, disagreed with its own children.
        # At `feature_fraction = 1.0` the identity is correct and nothing shows.
        #
        # No refusal made this unreachable. `oblivious_device_supported` refuses
        # `feature_fraction_bylevel` and `feature_fraction_bynode` and never the
        # per-tree `feature_fraction`, and `select_tree_features` returns the
        # identity only at `fraction >= 1.0`.
        #
        # Here rather than in the level loop because this runs once per tree,
        # the feature set cannot move inside a tree (the docstring above says
        # so and the geometry depends on it), and this point is already before
        # anything is in flight, so the drain inside `set_shared_features` costs
        # a wait that is already paid.
        self.batcher[0].set_shared_features(self.active)
        return self.batcher[0].stage_device_plan(
            n_items,
            max_rows,
            len(self.active),
            self.caps,
            Float32(self.g_scale),
            Float32(self.h_scale),
        )

    def enqueue_desc_level(
        mut self,
        mut rec_i: DeviceBuffer[DType.int32],
        mut rec_f: DeviceBuffer[DType.float32],
        mut fparams: DeviceBuffer[DType.float32],
        level_record: Int,
        level_depth: Int,
        max_depth: Int,
        gain_form: Int,
    ) raises:
        """Commit one oblivious level, writing the descriptor its partition
        routes by and the plan its batched build accumulates from.

        One launch. `gpu_tree_tables._commit_level_kernel` is the rule.
        """
        if len(self.resident_tables) == 0:
            raise Error("no device tree tables are open")
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        var pool = self.batcher[0].out_dev.copy()
        var plan_items = self.batcher[0].plan_items
        if plan_items < 2:
            raise Error(
                "no device-written plan is staged; call"
                " stage_desc_level_plan before committing a level"
            )
        # The pair count the batch that follows this commit may hold, for the
        # pair-indexed grid alone. A level at depth `l` commits `1 << l`
        # parents and `2 << l` children (`_commit_level_kernel` writes `2L`
        # items as `L` left children then `L` right ones), so this is exact
        # whenever the level commits and an over-estimate when growth already
        # stopped, which is the only direction that is safe and is the
        # direction `GpuLeafBatcher.set_level_pairs` clamps in. Zero for an
        # implausible depth, which means "assume the staged width" and is what
        # every arm but the pair-indexed one does unconditionally.
        #
        # Written here rather than passed to `enqueue_desc_level_children`
        # because this is the one call in the level loop that is told the
        # depth, and the file that runs that loop belongs to another lane.
        var level_pairs = 0
        if level_depth >= 0 and level_depth < 30:
            level_pairs = 1 << level_depth
        self.batcher[0].set_level_pairs(level_pairs)
        self.resident_tables[0].enqueue_level(
            rec_i,
            rec_f,
            fparams,
            pool.unsafe_ptr(),
            self.rows.step_dev.unsafe_ptr(),
            self.batcher[0].items_dev.unsafe_ptr(),
            True,
            level_record,
            self.n_bins,
            self.n_features * self.n_bins,
            level_depth,
            max_depth,
            plan_items,
            gain_form,
        )

    def enqueue_desc_stage_level_search(
        mut self,
        mut node_tbl: DeviceBuffer[DType.int32],
        slot_cells: Int,
        leaf_base: Int,
        max_leaves: Int,
    ) raises:
        """Point the level's leaf search records at the level's pool slots."""
        if len(self.resident_tables) == 0:
            raise Error("no device tree tables are open")
        self.resident_tables[0].enqueue_stage_level_search(
            node_tbl, slot_cells, leaf_base, max_leaves
        )

    def enqueue_desc_level_children(mut self) raises:
        """Build the level's children. **Two launches, whatever the level
        holds**, and they also pay the partition's deferred copy-back.

        This is the call `enqueue_desc_child` is replaced by, and the two are
        alternatives rather than stages: that one builds the smaller child of one
        parent and derives the larger by subtraction, which is two launches per
        parent and 126 per depth-6 tree; this one covers all `2^(l+1)` children
        of a level from the windows the level commit wrote, which is two
        launches per level and 12 per tree. Both accumulate the same
        per-`(row, feature)` quantized value into the same bin of the same slot
        in fixed-point Int32, so they are bit-identical where they overlap; see
        the plan-writing block of `gpu_tree_tables._pick_and_commit_kernel`.

        TWO ARMS, AND LOW LAUNCH COUNT IS NOT WHAT DECIDES BETWEEN THEM
        ---------------------------------------------------------------
        This docstring used to say "build every child of the level from its own
        rows" and to present that as the price of two launches per level. The
        two are not exclusive and the trade as stated was badly priced. At
        every level the children partition the active rows, so building all of
        them from their own rows reads EVERY active row once per level,
        permanently, while sibling subtraction reads at most half of them and
        derives the rest by exact integer arithmetic over the parent. The
        launches that bought were worth roughly 1.1 ms against a tree measured
        at 170 to 312 ms, and histogram construction is 86 percent of a
        symmetric fit (`bench/results`, the Aug 17 symmetric diagnosis).

        The subtracting arm accumulates only the smaller child of each pair and
        derives the sibling inside the same two launches:
        `GpuLeafBatcher.enqueue_device_plan_batch_fused_subtracting`, whose
        kernels fold the subtraction into the accumulation and the lopsided
        case's parent copy into the zeroing. The two arms leave the pool holding
        the same words, and the argument is written leg by leg at
        `gpu_leaf_batching._batch_hist_atomic_subtract_kernel`.

        **IT IS THE DEFAULT, and `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT=0` is the
        escape hatch that turns it off.** This paragraph read
        "`MOJOTREES_GPU_OBLIVIOUS_SUBTRACT=1` selects the subtracting arm ...
        Default off, because no benchmark has priced it; the numbers above say
        where the loss is, not that this removes it." Both halves stopped being
        true on 2026-08-17: the predicate at
        `gpu_leaf_batching.oblivious_subtract_requested` is `!= "0"`, and the arm
        was priced twice, once at 21.97 s to 12.34 s and once at 22.76 s to
        14.39 s over three interleaved round-robin cycles, both at
        799,110 x 100 x 100 trees, rmse identical to nine decimals. Corrected
        2026-08-17 by the GPU histogram lane; the measurement is recorded at that
        function's flip comment and in `docs/design/SWITCH_GRID.md`.

        **Quote the speedup as a RANGE of 1.58x to 1.78x.** This paragraph said
        "priced at 1.78x and then at 22.76 s to 14.39 s", which reads as one
        figure plus a detail when the second pair is 1.58x and is a second
        figure. Neither reading has been withdrawn, so reconciling them is
        another lane's item and no single multiple is quoted here.

        **A third arm exists behind its own switches and is default off.**
        `gpu_leaf_batching.plan_lean_requested` routes the accumulation to
        `_plan_hist_kernel` instead of either kernel above, which is the same
        integer accumulation with private threadgroup accumulators
        (`MOJOTREES_GPU_HIST_PRIVATE`), a per-item row split
        (`MOJOTREES_GPU_HIST_ROW_SPLIT`), and a feature group
        (`MOJOTREES_GPU_HIST_GROUP`) as knobs, plus three unconditional
        strictly-less-work removals reachable on their own with
        `MOJOTREES_GPU_HIST_LEAN=1`. It is still two launches per level, so the
        census does not move. Nothing it does can move a bit and the argument is
        leg by leg at that kernel.

        The copy-back is carried inside the zeroing pass the batch has to launch
        anyway (`gpu_leaf_batching._batch_copy_back_zero_kernel`, or
        `_batch_copy_back_zero_subtract_kernel` on the arm above, which carries
        it in the same statements), and paying it
        anywhere else would cost a third partition launch per level -- six per
        depth-6 tree, which is exactly the margin between 62 command buffers and
        68. `GpuActiveRows.mark_copy_back_fused` is the bookkeeping half and is
        called after the launch, never before: four refusals on that struct read
        the flag this clears.

        No `max_rows` argument, and that is not an omission. The batch's grid
        comes from the geometry `stage_desc_level_plan` fixed for the tree, so
        there is no per-level bound left for a caller to get wrong.
        """
        if not self.has_gradients:
            raise Error(
                "call upload_gradients before enqueue_desc_level_children"
            )
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        # The round's constant-hessian declaration, forwarded to the batcher
        # where the batcher certainly exists. THIS IS THE FORWARD THAT REACHES
        # A KERNEL; the one inside `set_constant_hessian` runs at fit setup,
        # before any pool is open, and does nothing on every path this package
        # actually takes. A host field assignment per level, six per tree, no
        # launch and no allocation. See `set_constant_hessian` for why the
        # forward is a switch and why it cannot move a bit when it is true.
        if batch_const_hessian_forward_requested():
            self.batcher[0].set_constant_hessian(self.constant_hessian())
        var blocks = self.rows.copy_back_debt_blocks()
        var rows_ptr = self.rows.rows_dev.copy()
        var scratch_ptr = self.rows.scratch_dev.copy()
        var desc_ptr = self.rows.step_dev.copy()
        # The branch is here rather than inside the batcher, so that only an
        # oblivious level plan can reach the subtracting arm. A leaf-wise
        # two-item plan also goes through `enqueue_device_plan_batch_fused`,
        # and its commit reassigns the PARENT's slot to one child and hands the
        # other the lowest free slot, so the slot relationship the subtracting
        # kernels rest on does not hold there. See
        # `gpu_leaf_batching.oblivious_subtract_requested`.
        if oblivious_subtract_requested():
            self.batcher[0].enqueue_device_plan_batch_fused_subtracting(
                self.bins_dev.unsafe_ptr(),
                rows_ptr.unsafe_ptr(),
                scratch_ptr.unsafe_ptr(),
                desc_ptr.unsafe_ptr(),
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
                blocks,
            )
        else:
            self.batcher[0].enqueue_device_plan_batch_fused(
                self.bins_dev.unsafe_ptr(),
                rows_ptr.unsafe_ptr(),
                scratch_ptr.unsafe_ptr(),
                desc_ptr.unsafe_ptr(),
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
                blocks,
            )
        self.rows.mark_copy_back_fused()

    def download_desc_tables(mut self) raises -> TreeTablesSnapshot:
        """Bring the whole device tree state home.

        **This is the device-owned growth loop's one host synchronization per
        tree.** Everything the loop enqueued finishes here, and everything it
        decided is decoded from what comes back.
        """
        if len(self.resident_tables) == 0:
            raise Error("no device tree tables are open")
        return self.resident_tables[0].download()

    def enqueue_resident_subtract(
        mut self, parent_slot: Int, child_slot: Int, dst_slot: Int
    ) raises:
        """`dst = parent - child` over resident slots, on the device.

        The device-side counterpart of `histogram.subtract_histogram`, and
        exact for the same reason the host one is: accumulation is fixed-point
        Int32 under one scale for the whole tree, so a parent's bins are the
        exact integer sum of its children's. `dst_slot` may be `parent_slot`,
        which is how a frontier keeps exactly one slot per live leaf.
        """
        if len(self.batcher) == 0:
            raise Error("no resident histogram pool is open")
        self.batcher[0].enqueue_subtract(parent_slot, child_slot, dst_slot)

    def _histogram_from_words(
        self, words: List[Int32], offset: Int = 0
    ) raises -> Histogram:
        """One slot's fixed-point words as the Float64 `Histogram`.

        The same arithmetic and the same current scales as
        `histogram_from_host`, over a list instead of the pinned output
        buffer, so a batched leaf and a single-leaf one decode identically.
        `offset` is where this slot starts in a multi-slot download, which
        `download_slots` returns concatenated at `slot_cells()` apiece.
        """
        var hist_size = self.n_features * self.n_bins
        if offset < 0 or len(words) < offset + 3 * hist_size:
            raise Error(
                "a histogram slot must hold 3 * n_features * n_bins words"
            )
        var g = _zeroed_f64(hist_size)
        var h = _zeroed_f64(hist_size)
        var c = _zeroed_int(hist_size)
        # Exact under `SCALE_SHAPE_POW2`, for the reason spelled out at the
        # same two lines in `histogram_from_host`.
        var g_inv = 1.0 / self.g_scale
        var h_inv = 1.0 / self.h_scale
        var gp = g.unsafe_ptr()
        var hp = h.unsafe_ptr()
        var cp = c.unsafe_ptr()
        var src = words.unsafe_ptr()
        var g0 = offset
        var h0 = offset + hist_size
        var c0 = offset + 2 * hist_size
        for i in range(hist_size):
            gp.unsafe_store(i, Float64(src.unsafe_load(g0 + i)) * g_inv)
            hp.unsafe_store(i, Float64(src.unsafe_load(h0 + i)) * h_inv)
            cp.unsafe_store(i, Int(src.unsafe_load(c0 + i)))
        return Histogram.from_planes(g^, h^, c^, self.n_features, self.n_bins)

    def _build_leaves_batched(
        mut self, nodes: List[Int]
    ) raises -> List[Histogram]:
        """Every node's histogram, several leaves per launch.

        Chunked at the smaller of the batcher's item bound and its slot pool,
        so a frontier larger than either is served by several launches rather
        than refused. Each chunk stages its plan, launches, and downloads,
        which is also what upholds `_stage_plan`'s ordering contract: a
        chunk's copies have retired before the next chunk stages over the
        pinned tables.
        """
        if not self.has_gradients:
            raise Error("call upload_gradients before build_leaves")
        var n = len(nodes)
        var out = List[Histogram](capacity=n)
        # The stamp two histograms must share to be subtractable, in the one
        # encoding `gpu_leaf_batching` defines for it.
        var stamp = subtraction_stamp(self.round_epoch, self.feat_epoch)
        var g32 = Float32(self.g_scale)
        var h32 = Float32(self.h_scale)
        var chunk = self.batcher[0].max_items
        var pool_cap = self.batcher[0].pool.capacity
        if pool_cap < chunk:
            chunk = pool_cap
        if chunk < 1:
            raise Error("the batched launcher holds no usable slots")

        # The active feature set reaches the batch through its own per-item
        # table, so it is staged when it (or the round's scales) last moved
        # rather than once per launch.
        if self.batch_feat_stamp != stamp:
            self.batcher[0].set_shared_features(self.active)
            self.batch_feat_stamp = stamp

        var taken = 0
        while taken < n:
            var take = n - taken
            if take > chunk:
                take = chunk
            var items = List[LeafWorkItem](capacity=take)
            for j in range(take):
                var node = nodes[taken + j]
                var r = self.rows.range_of(node)
                var slot = self.batcher[0].pool.acquire(node, stamp)
                if slot < 0:
                    raise Error(
                        "the batched histogram slot pool is full; raise"
                        " MOJOTREES_GPU_BATCH_SLOTS or build fewer leaves at"
                        " once"
                    )
                items.append(
                    LeafWorkItem(taken + j, node, r.begin, r.count(), slot, 0)
                )
            var plan = plan_batch(
                self.caps,
                items,
                uniform_scales(take, g32, h32),
                len(self.active),
                self.n_bins,
                self.tiling.strategy,
                self.part_capacity,
                self.batcher[0].max_items,
            )
            # Against the permutation itself rather than a raw pointer: this
            # entry checks every item's window against the tree's live active
            # prefix before it launches, which is the difference between a
            # batch that reads only its leaves' rows and one that silently
            # accumulates over rows this tree does not grow on.
            self.batcher[0].enqueue_frontier_batch(
                plan,
                self.bins_dev.unsafe_ptr(),
                self.rows,
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
            )
            # One mapping for the whole chunk, not one per leaf: a launch
            # that is not paid per leaf would be given straight back by a
            # synchronization that is.
            var slots = List[Int](capacity=take)
            for j in range(take):
                slots.append(items[j].out_slot)
            var words = self.batcher[0].download_slots(slots)
            var cells = self.batcher[0].slot_cells()
            for j in range(take):
                out.append(self._histogram_from_words(words, j * cells))
                self.batcher[0].pool.release(slots[j])
            taken += take
        return out^

    def batches(mut self, counts: List[Int]) raises -> Bool:
        """Whether leaves of these row counts would be built in one launch.

        The whole batching decision, in one place, so a grower choosing
        between a batch and the subtraction trick and `build_leaves` itself
        cannot answer it differently. Three conditions, none of them new:
        the batched level was asked for
        (`MOJOTREES_GPU_HIST_SPECIALIZATION=batched`), the launcher and its
        slot pool are affordable at this dataset's shape, and
        `apple_histogram_policy.batching_declined_reason` endorses this
        frontier, which it does not for fewer than two leaves and does not
        when every leaf already fills the device on its own.

        The plan the verdict is read off belongs to the batch's *largest*
        leaf, which is the one with the best chance of filling the device
        alone: if even its geometry leaves the device short, no smaller leaf
        in the batch fills it either.
        """
        if self.spec_level < SPEC_LEVEL_BATCHED or len(counts) < 2:
            return False
        if not self.open_batching():
            return False
        var widest = 0
        for i in range(len(counts)):
            if counts[i] > widest:
                widest = counts[i]
        var verdict = batching_declined_reason(
            self.histogram_plan(widest),
            profile_from_caps(self.caps),
            build_kernel_features(),
            counts,
            len(self.active),
        )
        return verdict == REASON_AS_REQUESTED

    def batches_nodes(mut self, nodes: List[Int]) raises -> Bool:
        """`batches` for a list of node ids, with the row counts looked up
        here. The level check comes first so the default path answers
        without touching the range table."""
        if self.spec_level < SPEC_LEVEL_BATCHED or len(nodes) < 2:
            return False
        return self.batches(self.leaf_rows(nodes))

    def leaf_rows(self, nodes: List[Int]) raises -> List[Int]:
        """Each node's current row count, off the device-resident ranges.
        What a grower hands `batches` before it decides."""
        var counts = List[Int](capacity=len(nodes))
        for i in range(len(nodes)):
            counts.append(self.rows.range_of(nodes[i]).count())
        return counts^

    def build_leaves(mut self, nodes: List[Int]) raises -> List[Histogram]:
        """Every node's histogram, in `nodes` order.

        The multi-leaf entry point, and the one a grower with more than one
        pending leaf should call. Whether it batches is `batches`'s answer
        and not a switch of its own; where that answer is no, each node goes
        through `build_leaf` exactly as before, so a caller may always call
        this and never has to branch on the policy itself.
        """
        if len(nodes) < 1:
            raise Error("build_leaves needs at least one node")
        for i in range(len(nodes)):
            for k in range(i):
                if nodes[k] == nodes[i]:
                    raise Error("build_leaves may not hold a node twice")
        var counts = self.leaf_rows(nodes)
        if self.batches(counts):
            return self._build_leaves_batched(nodes)

        var out = List[Histogram](capacity=len(nodes))
        for i in range(len(nodes)):
            out.append(self.build_leaf(nodes[i]))
        return out^

    def build(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises -> Histogram:
        """Build a full-dataset histogram on the GPU (uploads gradients and
        resets leaf assignments; use the finer-grained methods when training
        whole trees)."""
        self.upload_gradients(grad, hess)
        self.begin_tree()
        return self.build_leaf(0)


def build_histogram_gpu(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    strategy: Int = STRATEGY_AUTO,
) raises -> Histogram:
    """One-shot GPU histogram build (uploads the binned matrix every call;
    use `GpuHistogramBuilder` for repeated builds on one dataset)."""
    var builder = GpuHistogramBuilder(data, strategy)
    return builder.build(grad, hess)
