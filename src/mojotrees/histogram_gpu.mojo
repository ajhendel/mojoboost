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
`stage_from_device`, which only hybrid leaf scheduling reaches, adds two
copies and a synchronize of its own on top.

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
    GpuObjectiveState,
    device_fixed_scale,
)
from .parallel import _env_int, dispatch_rows
from .quantized_gradient import fixed_point_scale, magnitude_sum
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


def _fixed_scale(values: List[Float64]) raises -> Float32:
    """Fixed-point scale derived from a host-side value list."""
    return fixed_point_scale(magnitude_sum(values))


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
    var has_gradients: Bool
    # Whether the Float32 gradients the kernels read this round are also
    # sitting in the `stage_gh` arena on the host: True after
    # `stage_gradients` (the host produced them) or `stage_from_device` (the
    # device produced them and the host read them back), False after a
    # device fill until one of those runs. A host replica build
    # (`build_leaf_host_replica`) is only possible when this is True.
    var gradients_host: Bool
    # Whether the host fixed-point replica has been shown to reproduce this
    # device's histograms bit for bit: 0 untested, 1 verified, 2 refuted.
    # Set by the grower's mirror comparison (see hybrid_leaf_scheduler.mojo)
    # and kept on the builder so one fit verifies once, not once per tree.
    var replica_state: Int
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
        self.has_gradients = False
        self.gradients_host = False
        self.replica_state = 0
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

        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)
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
        exactly what `build_leaf` does — the same replica claim
        `replica_state` records, no new one. Costs `2 * 4 * n_rows` bytes
        and one synchronize per round (per class, on a softmax round), which
        is the price of reaching the built-in-objective path with hybrid
        leaf scheduling at all; the grower pays it only when that scheduling
        is switched on and measured.

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
        # because this path is hybrid-leaf-scheduling only and no arm of it
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

    def fill_gradients_device(
        mut self,
        mut state: GpuObjectiveState,
        objective: Int,
        alpha: Float64,
    ) raises:
        """This round's gradients, computed on the device straight into the
        histogram buffers. Replaces `upload_gradients` for the built-in
        objectives; the fixed-point scales come from a device reduction
        instead of a host pass, and nothing per-row crosses to the device."""
        state.fill_grad_hess(
            self.ctx, objective, alpha, self.grad_dev, self.hess_dev
        )
        var sums = state.magnitude_sums(
            self.ctx, self.grad_dev, self.hess_dev
        )
        self.g_scale = Float64(device_fixed_scale(sums.grad))
        self.h_scale = Float64(device_fixed_scale(sums.hess))
        self.has_gradients = True
        self.gradients_host = False
        self.round_epoch += 1

    def fill_softmax_gradients_device(
        mut self, mut state: GpuObjectiveState, k: Int
    ) raises:
        """Class `k`'s softmax gradients into the histogram buffers; call
        `state.refresh_softmax` once per round first."""
        state.fill_softmax_grad_hess(
            self.ctx, k, self.grad_dev, self.hess_dev
        )
        var sums = state.magnitude_sums(
            self.ctx, self.grad_dev, self.hess_dev
        )
        self.g_scale = Float64(device_fixed_scale(sums.grad))
        self.h_scale = Float64(device_fixed_scale(sums.hess))
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

        That matters for how a change here may be described, because our
        headline benchmark does not take this path. At 1,000,000 rows, 50
        features, 255 bins and 31 leaves `normalized_split_work` is exactly
        50,000,000.0, which is not less than `M4_MIN_NORMALIZED_WORK`, so
        `gpu_split_policy` resolves that shape to the device-resident
        search and this function is never reached. Nothing done here can
        move that number, and nothing here claims to.

        What it does reach is everything on the other side of that
        crossover, which is a knife edge rather than a broad margin: one
        row fewer, one feature fewer, one leaf fewer, or any
        `feature_fraction` below 1 all fall back to the host scan and start
        paying this per node. `SplitSearchDecision.uses_device` is exactly
        the predicate "this function is off the path"; `margin` and
        `on_crossover_boundary` say how far a shape is from flipping it,
        and `tests/test_gpu_split_launch_overhead.mojo` pins both at the
        shapes either side of the edge.

        On the host-scan path the cost is real. It covers
        `3 * n_features * n_bins` cells, and the hybrid scheduler's
        calibrated cost model prices the conversion at
        `convert_nanos_per_kcell = 10024`, about 10 ns per cell, against a
        modeled device fixed cost per node of roughly 263 microseconds.
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
        return Histogram(g^, h^, c^, self.n_features, self.n_bins)

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
        about this device's multiply-and-round, established by the grower's
        mirror comparison and recorded in `replica_state`, never assumed.
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

    def open_resident(mut self, want_slots: Int) raises -> Bool:
        """Allocate a slot pool deep enough to hold `want_slots` leaves at
        once, and report whether the resident path is available afterwards.

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
            return self.batcher[0].pool.capacity >= want_slots
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
                DEFAULT_MAX_ITEMS,
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
        return Histogram(g^, h^, c^, self.n_features, self.n_bins)

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
