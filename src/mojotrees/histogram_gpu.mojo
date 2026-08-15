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
from .gpu_active_rows import GpuActiveRows, LeafRange, RowRouting
from .gpu_binned_layout import check_layout_support
from .gpu_frontier import LeafWorkItem
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
from .parallel import _env_int
from .quantized_gradient import fixed_point_scale, magnitude_sum
from .unified_memory_policy import (
    ROLE_BINS,
    ROUTE_COPY_STAGED,
    describe_decision,
    resolve_from_env,
)
from .gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_AUTO,
    STRATEGY_TILED,
    DeviceCaps,
    HistogramTiling,
    derive_tiling,
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
    _zeroed_f64,
    _zeroed_int,
    build_histogram_subset_replica_into,
)

comptime MAX_BINS = 256

# Row indices and leaf ids cross into the kernels as Int32.
comptime MAX_ROWS = Int(Int32.MAX)

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
    `auto`. The other two variants do not exist: the shipping kernels
    allocate the `MAX_BINS` threadgroup width at every bin count and read one
    bin per load.
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
    var stage_g: HostBuffer[DType.float32]
    var stage_h: HostBuffer[DType.float32]
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
    # Whether this round's gradients came through `stage_gradients`, so the
    # Float32 conversions the kernels read are still sitting in `stage_g` /
    # `stage_h` on the host. False on the device-objective paths, where the
    # gradients never exist host-side; a host replica build
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
        # Metal defaults to the paired histogram kernels: measured on an
        # Apple M4 at 1.39x end to end for a 5M x 50 fit with byte-identical
        # predictions (see _range_hist_partial_g2_kernel), and pairing
        # changes no histogram it produces on any backend. An explicit
        # MOJOTREES_GPU_FEATURE_GROUP still wins in both directions, and
        # CUDA/HIP/unknown keep one feature per threadgroup until someone
        # measures them — the shared-memory doubling is exactly the kind of
        # change that can invert on a different threadgroup budget.
        if (
            getenv("MOJOTREES_GPU_FEATURE_GROUP") == ""
            and parse_api(self.device_api) == API_METAL
        ):
            self.rows.set_feature_group(2)
        self.g_scale = 1.0
        self.h_scale = 1.0
        self.has_gradients = False
        self.gradients_host = False
        self.replica_state = 0
        self.round_epoch = 0
        self.feat_epoch = 0
        self.batch_feat_stamp = -1
        self.active = List[Int](capacity=data.n_features)
        for f in range(data.n_features):
            self.active.append(f)

        var n_cells = data.n_rows * data.n_features
        var hist_size = data.n_features * data.n_bins
        self.bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](n_cells)
        self.grad_dev = self.ctx.enqueue_create_buffer[DType.float32](
            data.n_rows
        )
        self.hess_dev = self.ctx.enqueue_create_buffer[DType.float32](
            data.n_rows
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

        self.stage_g = self.ctx.enqueue_create_host_buffer[DType.float32](
            data.n_rows
        )
        self.stage_h = self.ctx.enqueue_create_host_buffer[DType.float32](
            data.n_rows
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
        """How many features one histogram threadgroup accumulates: 1, the
        shipping launch, or 2, the paired one (gpu_active_rows.mojo). Both
        build the same integer histogram, so this is a launch shape and not
        a numeric option; it exists so a benchmark can hold both arms in one
        process instead of reading its arm from the environment."""
        self.rows.set_feature_group(group)

    def feature_group(self) -> Int:
        """The launch shape `set_feature_group` last chose."""
        return self.rows.feature_group

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
        Float32 in pinned host memory. Host work only, no transfer.

        The device's copy is stale from here until `upload_staged`, and the
        scales below already describe the new values, so builds are refused
        in between rather than mixing one round's scale with another's
        data."""
        if len(grad) != self.n_rows or len(hess) != self.n_rows:
            raise Error("gradient/hessian length must equal n_rows")
        self.has_gradients = False

        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)
        self.g_scale = Float64(g_scale)
        self.h_scale = Float64(h_scale)
        self.round_epoch += 1

        # Any copy still reading the staging buffers has to finish before
        # they are overwritten.
        self.ctx.synchronize()

        var dst_g = self.stage_g.unsafe_ptr()
        var dst_h = self.stage_h.unsafe_ptr()
        var src_g = grad.unsafe_ptr()
        var src_h = hess.unsafe_ptr()
        for r in range(self.n_rows):
            dst_g.unsafe_store(r, Float32(src_g.unsafe_load(r)))
            dst_h.unsafe_store(r, Float32(src_h.unsafe_load(r)))
        self.gradients_host = True

    def upload_staged(mut self) raises:
        """Copy the staged gradients and hessians to the device."""
        self.ctx.enqueue_copy(
            dst_buf=self.grad_dev, src_ptr=self.stage_g.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.hess_dev, src_ptr=self.stage_h.unsafe_ptr()
        )
        self.has_gradients = True

    def upload_gradients(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises:
        """Upload this round's per-row gradients and hessians (once per
        boosting round, not per node)."""
        self.stage_gradients(grad, hess)
        self.upload_staged()

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
        `Histogram`. Host work only; call after `download_raw`."""
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
        for i in range(hist_size):
            gp.unsafe_store(i, Float64(src.unsafe_load(i)) * g_inv)
            hp.unsafe_store(i, Float64(src.unsafe_load(hist_size + i)) * h_inv)
            cp.unsafe_store(i, Int(src.unsafe_load(2 * hist_size + i)))
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
        var grad_span = Span(
            unsafe_ptr=self.stage_g.unsafe_ptr(), length=self.n_rows
        )
        var hess_span = Span(
            unsafe_ptr=self.stage_h.unsafe_ptr(), length=self.n_rows
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
        )

    def snapshot_rows(mut self, mut out: List[Int32]) raises:
        """The whole active-row permutation into `out`, resized to `n_rows`.

        The hybrid grower's once-per-tree snapshot
        (docs/design/HYBRID_TRAINING.md §3): one whole-buffer copy through
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
        """
        if not self.resident_frontier_fits(want_slots):
            return False
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
