"""End-to-end GPU training.

`train_gpu` mirrors `train` in boosting.mojo but grows every tree through the
GPU backend: one persistent `GpuHistogramBuilder` holds the binned matrix,
gradients/hessians, and a device-resident active-row permutation in which
every live leaf owns a contiguous row range (see gpu_active_rows.mojo).
For the built-in objectives without row sampling, the round's gradients are
generated on the device from device-resident labels and raw scores, and
each grown tree advances those raw scores from its leaf ranges (see
gpu_objectives_native.mojo), so nothing per-row crosses the host/device
boundary in a plain round. Under bagging or GOSS the host generates and
uploads the gradients instead, because the row sample is ranked and drawn
host-side. Per split, the device stably partitions the parent's range and
builds the smaller child's histogram over exactly that child's rows (the
sibling comes from the subtraction trick on the host, where histograms are
small: n_features * n_bins). The grower hands the partition its exact left
count from the parent histogram's integer counts, so a split enqueues
without a host synchronization.

Division of labor (the GPU owns the data plane; the CPU owns the control
plane, small data, and verification; see docs/ARCHITECTURE.md seam 4):
  CPU  boosting coordination, split selection over downloaded histograms,
       leaf-value renewal (quantile/L1), prediction, the tree model itself,
       host row sampling under bagging/GOSS, validation scoring, and the
       bit-exact reference the device path is verified against; below the
       launch-cost crossover the CPU also builds individual small leaves
       (hybrid_leaf_scheduler.mojo) or the whole fit (device_policy.mojo)
  GPU  binned features, gradients/hessians, leaf assignments, histogram
       accumulation, row partitioning, native objective evaluation and
       score advancement, and the split scan when selected

`train_custom_gpu` is the same loop with the gradients coming from a
caller-supplied callable instead of a built-in objective (see
objective.mojo). The callback stays on the host, where the raw scores live,
and only the gradients it produces cross to the device, so a custom
objective costs no more on the GPU than a built-in one does.

Row bagging is the one exception to the no-row-lists rule, and only at the
start of a tree: the bag decides which rows sit at the root and which sit
out of bag, so the leaf-assignment array is written once per tree instead of
memset. Both trainers draw bags from bagging.mojo with the same seed and
schedule, so CPU and GPU rounds are grown on identical rows.

No per-node row lists or per-node gradient vectors ever cross the
host/device boundary. GPU histograms carry Float32 precision (see
histogram_gpu.mojo), so
trained models agree with the CPU trainer's predictions to Float32-level
tolerance, not bit-exactly; the GPU trainer itself is bit-deterministic run
to run.

Every device stage this file reaches has a switch back to the path it
replaced, and none of them defaults to a claim no benchmark has made:

  gradients      `objective_source` / `MOJOTREES_GPU_OBJECTIVE`, one of
                 `auto` (the shipped behavior: the device kernels whenever
                 the objective has them and no row sampling is configured),
                 `host` (never; upload from `_fill_grad_hess` instead), or
                 `device` (a hard requirement, which raises with a specific
                 reason when the kernels cannot serve it)
  split search   `split_search` / `MOJOTREES_GPU_SPLIT_STRATEGY`; explicit
                 host/device requests are exact, while AUTO uses the pure
                 workload and hardware policy in gpu_split_policy.mojo and
                 conservatively falls back to the host scan
  validation     `valid_scoring` / `MOJOTREES_GPU_VALID_SCORING`, defaulting
                 to the host tree walk; see `train_gpu_with_valid` and the
                 VALID_SCORE_* constants
  histograms     `MOJOTREES_GPU_HIST_SPECIALIZATION=batched` asks for
                 several leaves per launch (gpu_leaf_batching.mojo, gated by
                 `apple_histogram_policy`), which the leaf-wise grower can
                 feed with a split's two children. Unset, every histogram is
                 the single-leaf launch that shipped and the sibling still
                 comes from the subtraction trick; see `grow_tree_gpu`.
  bagged rounds  a bagged run reaches the device objective path only under
                 an explicit `objective_source=OBJECTIVE_SOURCE_DEVICE`,
                 where `GpuTreeRouter` (gpu_fused_round.mojo) advances every
                 row's raw score, in bag or not. Under AUTO a bagged run
                 keeps the host path and its Float64 raw scores, which is
                 what shipped.
  session        each trainer has an overload taking a `GpuSession`
                 (gpu_runtime.mojo). Without one the trainers run on
                 `NoLifecycle` and execute exactly the device calls they did
                 before the seam existed; with one, the builder borrows the
                 session's context, the round and tree boundaries are
                 announced to it, and the fit is timed against its cold/warm
                 split. The session is bookkeeping today: nothing yet owns
                 one across two fits, which is what would let its pool and
                 residency ledgers skip an upload rather than only record
                 that they could have.

Which configurations the device round can serve is not decided here. That
question has one answer in the package, `gpu_fused_round.round_eligibility`,
and `device_gradients` below is the trainer's binding of it rather than a
second list of blockers that could drift from it.
"""

from std.math import log, min
from std.os import getenv
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from .apple_histogram_policy import ClassSchedule
from .bagging import BaggingParams, bagging_enabled, check_bagging, refresh_bag
from .binning import BinnedMatrix
from .boosting import (
    CUSTOM,
    L1,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    MulticlassBooster,
    _base_score,
    _check_objective,
    _check_sample_weight,
    _clamp_prob,
    _fill_grad_hess,
    _mean_loss,
    _renew_leaf_values,
    _check_goss,
    _fill_softmax_grad_hess,
    _multiclass_goss_select,
    _softmax_inplace,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
)
from .goss import GossParams, GossSelection, apply_goss_scaling, goss_round
from .gpu_frontier import subtraction_builds_left
from .gpu_fused_round import (
    ROUND_OK,
    GpuTreeRouter,
    round_eligibility,
    round_eligibility_reason,
)
from .gpu_multiclass_batch import GpuClassBatch, MulticlassRoundGuard
from .gpu_objectives_native import GpuObjectiveState
from .gpu_output_planes import BatchEligibility
from .gpu_predict import (
    DEVICE_METRIC_L1,
    DEVICE_METRIC_L2,
    RESPONSE_IDENTITY,
    GpuPredictor,
    flatten_trees,
)
from .gpu_runtime import GpuSession, NoLifecycle, RoundLifecycle
from .gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    SplitNodeRequest,
)
from .gpu_split_policy import decide_split_search
from .gpu_tiling import DeviceCaps
from .histogram import Histogram, subtract_histogram
from .histogram_gpu import GpuHistogramBuilder
from .hybrid_leaf_scheduler import (
    MODE_MIRROR,
    MODE_OFF,
    MODE_REPLICA,
    PLACE_BOTH,
    REPLICA_REFUTED,
    REPLICA_UNTESTED,
    REPLICA_VERIFIED,
    ROWS_DEVICE_COPY,
    HybridContext,
    HybridCosts,
    env_hybrid_costs,
    env_hybrid_mode,
    mode_name,
    plan_split,
)
from .objective import (
    GradHessFn,
    _apply_sample_weight,
    check_custom_grad_hess,
)
from .interaction import extend_branch
from .monotone import (
    MONOTONE_FREE,
    ChildBounds,
    OutputBounds,
    child_bounds,
    midpoint,
    monotone_sign,
)
from .sampling import (
    check_feature_fractions,
    select_node_features,
    select_split_features,
    select_tree_features,
)
from .split import SplitInfo
from .growth_policy import (
    GROW_DEPTHWISE,
    GrowthSchedule,
    LeafCandidate,
    check_grow_policy,
)
from .tree import Tree, TreeParams, _leaf_value, _search


struct _GpuLeafState(Movable):
    """A grown-but-unsplit leaf: its node id (also its device-side leaf id),
    row count, histogram, the best split available from it, the features
    split on between the root and it (empty when no interaction constraints
    are configured), its depth in edges from the root, and the interval its
    output must lie in (unbounded when no monotonic constraint above it
    applies)."""

    var node: Int
    var n_rows: Int
    var hist: Histogram
    var split: SplitInfo
    var branch: List[Int]
    var depth: Int
    var bounds: OutputBounds

    def __init__(
        out self,
        node: Int,
        n_rows: Int,
        var hist: Histogram,
        var split: SplitInfo,
        var branch: List[Int] = [],
        depth: Int = 0,
        var bounds: OutputBounds = OutputBounds.unbounded(),
    ):
        self.node = node
        self.n_rows = n_rows
        self.hist = hist^
        self.split = split^
        self.branch = branch^
        self.depth = depth
        self.bounds = bounds^


def _histograms_match(a: Histogram, b: Histogram) -> Bool:
    """Whether two histograms carry exactly the same numbers, cell for cell.

    The mirror comparison behind the replica claim (HYBRID_TRAINING.md §4):
    both sides dequantize with the same `Float64(Int32) * (1 / scale)`, so
    equal Float64 planes here is equivalent to equal Int32 planes there —
    the multiply is injective over the Int32 range for any positive scale.
    Exact equality on purpose; a tolerance would verify nothing.
    """
    if a.n_features != b.n_features or a.n_bins != b.n_bins:
        return False
    var size = a.n_features * a.n_bins
    var ag = a.grad.unsafe_ptr()
    var bg = b.grad.unsafe_ptr()
    var ah = a.hess.unsafe_ptr()
    var bh = b.hess.unsafe_ptr()
    var ac = a.count.unsafe_ptr()
    var bc = b.count.unsafe_ptr()
    for i in range(size):
        if (
            ag.unsafe_load(i) != bg.unsafe_load(i)
            or ah.unsafe_load(i) != bh.unsafe_load(i)
            or ac.unsafe_load(i) != bc.unsafe_load(i)
        ):
            return False
    return True


def _count_left(
    hist: Histogram,
    split: SplitInfo,
    missing_bin: Int = -1,
) -> Int:
    """Rows going left under `split`, from the exact integer counts of the
    node's histogram — no host-side row partitioning needed. Every bin is
    routed by `split.goes_left`, the same rule the device partition kernel
    and `Tree.goes_left` apply, so the three cannot disagree; rows in
    `missing_bin` follow the split's default direction instead."""
    var total = 0
    var base = split.feature * hist.n_bins
    for b in range(hist.n_bins):
        var go_left: Bool
        if not split.is_categorical and b == missing_bin:
            go_left = split.default_left
        else:
            go_left = split.goes_left(b)
        if go_left:
            total += hist.count[base + b]
    return total


# Where each node's best split is chosen. HOST downloads the node's
# histogram and scans it in Float64 with `_search`, which is what keeps
# CPU/GPU split decisions identical and is the conservative fallback. DEVICE
# scans the histogram where it was accumulated (see
# gpu_split_search.mojo) and downloads one 136-byte record per node instead
# of the whole histogram; its gains and leaf values are Float32, so split
# decisions can differ from the host's on near-ties. AUTO reads
# `MOJOTREES_GPU_SPLIT_STRATEGY` (`host` or `device`) and otherwise resolves
# through gpu_split_policy.mojo from the workload and reported hardware.
comptime SPLIT_SEARCH_AUTO = 0
comptime SPLIT_SEARCH_HOST = 1
comptime SPLIT_SEARCH_DEVICE = 2


def env_split_search() -> Int:
    """`MOJOTREES_GPU_SPLIT_STRATEGY` as a split-search constant."""
    var s = getenv("MOJOTREES_GPU_SPLIT_STRATEGY")
    if s == "device":
        return SPLIT_SEARCH_DEVICE
    if s == "host":
        return SPLIT_SEARCH_HOST
    return SPLIT_SEARCH_AUTO


def resolve_split_search(strategy: Int) -> Int:
    """Legacy request-only resolution without workload or hardware facts.

    Production tree growth uses `resolve_split_search_for`; this helper stays
    conservative for callers that cannot supply a builder and therefore
    cannot justify an automatic device choice.
    """
    var s = strategy
    if s == SPLIT_SEARCH_AUTO:
        s = env_split_search()
    if s == SPLIT_SEARCH_DEVICE:
        return SPLIT_SEARCH_DEVICE
    return SPLIT_SEARCH_HOST


def _device_search_semantics_supported(params: TreeParams) -> Bool:
    """Question form of `_check_device_search_supported` for AUTO.

    Explicit device selection still calls the raising check and reports the
    exact unsupported setting.  AUTO needs a non-raising eligibility answer
    so it can retain the fully featured host scan instead of failing a fit.
    """
    return (
        not params.extra.is_active()
        and params.feature_fraction_bylevel == 1.0
    )


def _estimated_active_features(params: TreeParams, n_features: Int) -> Int:
    """Features a tree-level histogram is expected to scan.

    `feature_fraction` is the only draw known before the tree seed is applied;
    per-node sampling is intentionally not folded in because the resident
    frontier holds full-width slots and the policy must not understate its
    memory shape.
    """
    var active = Int(Float64(n_features) * params.feature_fraction)
    if active < 1:
        return 1
    if active > n_features:
        return n_features
    return active


def resolve_split_search_for(
    builder: GpuHistogramBuilder, params: TreeParams
) raises -> Int:
    """Resolve explicit/environment requests, then workload-aware AUTO.

    Explicit `host` and `device` retain their old meanings.  With neither
    present, the pure policy sees the reported device signature, actual
    matrix shape, tree budget, semantic eligibility, and the builder's own
    resident-memory calculation.  Unknown or marginal cases stay on host.
    """
    var decision = decide_split_search(
        builder.device_api,
        builder.device_arch,
        builder.n_rows,
        _estimated_active_features(params, builder.n_features),
        builder.n_bins,
        params.num_leaves,
        _device_search_semantics_supported(params),
        builder.resident_frontier_fits(params.num_leaves),
    )
    return (
        SPLIT_SEARCH_DEVICE if decision.uses_device() else SPLIT_SEARCH_HOST
    )


def resolve_split_search_for(
    builder: GpuHistogramBuilder, params: TreeParams, strategy: Int
) raises -> Int:
    """Explicit request wrapper around workload-aware AUTO."""
    var requested = strategy
    if requested == SPLIT_SEARCH_AUTO:
        requested = env_split_search()
    if requested == SPLIT_SEARCH_DEVICE:
        return SPLIT_SEARCH_DEVICE
    if requested == SPLIT_SEARCH_HOST:
        return SPLIT_SEARCH_HOST
    return resolve_split_search_for(builder, params)


# Where a round's gradients come from. DEVICE generates them on the device
# from device-resident labels and raw scores (gpu_objectives_native.mojo) and
# advances the raw scores there too, so a round moves nothing per row. HOST
# evaluates the objective in `_fill_grad_hess` over host-side raw scores and
# uploads the result, which is the path row sampling and custom objectives
# need and the one every GPU round took before the device objectives landed.
# AUTO reads `MOJOTREES_GPU_OBJECTIVE` (`host` or `device`) and then takes
# the device path wherever it is available.
comptime OBJECTIVE_SOURCE_AUTO = 0
comptime OBJECTIVE_SOURCE_HOST = 1
comptime OBJECTIVE_SOURCE_DEVICE = 2


def env_objective_source() -> Int:
    """`MOJOTREES_GPU_OBJECTIVE` as an objective-source constant."""
    var s = getenv("MOJOTREES_GPU_OBJECTIVE")
    if s == "device":
        return OBJECTIVE_SOURCE_DEVICE
    if s == "host":
        return OBJECTIVE_SOURCE_HOST
    return OBJECTIVE_SOURCE_AUTO


def resolve_objective_source(source: Int) -> Int:
    """An explicit source outranks the environment; AUTO resolves through
    `MOJOTREES_GPU_OBJECTIVE` and then stays AUTO, since whether the device
    can serve it is a property of the objective and the sampling, not of the
    request. `device_gradients` answers that."""
    var s = source
    if s == OBJECTIVE_SOURCE_AUTO:
        s = env_objective_source()
    if s == OBJECTIVE_SOURCE_HOST:
        return OBJECTIVE_SOURCE_HOST
    if s == OBJECTIVE_SOURCE_DEVICE:
        return OBJECTIVE_SOURCE_DEVICE
    return OBJECTIVE_SOURCE_AUTO


# Softmax has no id in the objective registry: boosting.mojo numbers the
# single-output objectives and a multiclass run is selected by `n_classes`
# rather than by an objective code. `round_eligibility` reaches the
# device-kernel question only at `n_classes == 1`, so the multiclass entry
# points name this placeholder instead of borrowing a regression code at the
# call site and leaving a reader to work out that it is never read.
comptime _SOFTMAX_OBJECTIVE = SQUARED_ERROR


def device_gradients(
    objective: Int,
    n_classes: Int,
    source: Int,
    bagging: BaggingParams,
    goss: GossParams,
    routes_all_rows: Bool = False,
) raises -> Bool:
    """Whether this run generates its gradients on the device.

    The question itself belongs to `gpu_fused_round.round_eligibility`,
    which is the package's one answer to which configurations every per-row
    stage of a round can serve, and this is the trainer's binding of it: the
    caller's `objective_source` decides whether the answer is consulted at
    all, and a caller that asked for the device path explicitly is raised at
    with that module's own reason rather than being quietly downgraded.
    There is deliberately no second list of blockers here; a blocker added
    there reaches this trainer without an edit.

    `routes_all_rows` is the trainer stating that it advances the raw scores
    with `GpuTreeRouter.update_all_rows`, which covers the rows a bag left
    out and is the one thing that keeps a bagged run off the device round.
    The trainers below pass it only under an explicit
    `OBJECTIVE_SOURCE_DEVICE`, so AUTO keeps bagging on the host path and
    its Float64 raw scores.

    GOSS stays blocked either way: its sample is a ranking of
    `|grad * hess|`, and ranking Float32 device scores can put a different
    row across the threshold, so `allow_device_ranking` is left False and
    both backends keep sampling identically.
    """
    var s = resolve_objective_source(source)
    if s == OBJECTIVE_SOURCE_HOST:
        return False
    var code = round_eligibility(
        objective,
        n_classes,
        bagging_enabled(bagging),
        goss.enabled,
        False,
        routes_all_rows,
    )
    if code == ROUND_OK:
        return True
    if s == OBJECTIVE_SOURCE_DEVICE:
        raise Error(
            round_eligibility_reason(code),
            ". Use objective_source=OBJECTIVE_SOURCE_HOST (or",
            " MOJOTREES_GPU_OBJECTIVE=host), which uploads host-computed",
            " gradients and grows the trees on the device exactly as before",
        )
    return False


struct _GpuRecordLeafState(Movable):
    """A grown-but-unsplit leaf under device split selection: the compact
    search record stands in for the histogram the host-search frontier
    carries, since the record already holds the split, both children's
    counts and Newton values, and the parent's value.

    `slot` is where this leaf's histogram still lives on the device, or -1
    when nothing kept it. The resident loop keeps one, so that a split can
    derive its larger child by subtraction instead of accumulating it; the
    incremental loop keeps none, because its histograms are overwritten in
    the builder's single-node buffer by the next node's build."""

    var node: Int
    var n_rows: Int
    var rec: GpuSplitRecord
    var branch: List[Int]
    var depth: Int
    var bounds: OutputBounds
    var slot: Int

    def __init__(
        out self,
        node: Int,
        n_rows: Int,
        var rec: GpuSplitRecord,
        var branch: List[Int] = [],
        depth: Int = 0,
        var bounds: OutputBounds = OutputBounds.unbounded(),
        slot: Int = -1,
    ):
        self.node = node
        self.n_rows = n_rows
        self.rec = rec^
        self.branch = branch^
        self.depth = depth
        self.bounds = bounds^
        self.slot = slot


# Ceilings on the searcher's record capacity under depth-wise growth, where
# one batch is a whole planned level. A level wider than this is searched in
# several batches, which costs one extra wait apiece and nothing else, so
# both are budget decisions rather than correctness ones.
#
# The record count itself, which bounds `grid.y` of the search launch and the
# 136 bytes per record `download_frontier` brings home.
comptime MAX_LEVEL_RECORDS = 512
# Cells in one per-record table. The searcher strides `feat_dev` and
# `allow_dev` by `n_features` rather than by a batch's slot count, so their
# size is `records * n_features` and a wide dataset has to buy its capacity
# in records. 2^20 cells is 4 MiB per table.
comptime MAX_LEVEL_TABLE_CELLS = 1 << 20


def _search_record_slots(params: TreeParams, n_features: Int) -> Int:
    """Record slots the device-search searcher is constructed with.

    Two is what a split needs: the resident loop searches a split's two
    children in one launch pair, and the incremental loop uses the first
    slot only. Depth-wise growth searches a whole planned level in one pair
    instead (`GrowthSchedule.plan_level`), so it buys room for one, bounded
    by both ceilings above and never below the two a single split needs.
    """
    if params.grow_policy != GROW_DEPTHWISE:
        return 2
    var slots = 2 * params.num_leaves
    if slots > MAX_LEVEL_RECORDS:
        slots = MAX_LEVEL_RECORDS
    var width = n_features if n_features > 0 else 1
    var by_cells = MAX_LEVEL_TABLE_CELLS // width
    if slots > by_cells:
        slots = by_cells
    if slots < 2:
        slots = 2
    return slots


struct _GpuPendingSplit(Movable):
    """A split whose device work is enqueued and whose children's records
    have not come home yet.

    `_device_search_resident` commits a batch of splits before it waits, so
    everything the frontier update needs after the wait has to survive the
    enqueue: which frontier slot the parent held, the two child node ids and
    their exact row counts, the branch and depth both children inherit, the
    monotone interval each child's own search must respect, and the pool
    slot each child's histogram landed in. The two records themselves arrive
    from `download_frontier` in the order the requests were staged, which is
    what pairs a pending split with `recs[2 * k]` and `recs[2 * k + 1]`.
    """

    var index: Int
    var left_node: Int
    var right_node: Int
    var n_left: Int
    var n_right: Int
    var depth: Int
    var branch: List[Int]
    var left_bounds: OutputBounds
    var right_bounds: OutputBounds
    var left_slot: Int
    var right_slot: Int

    def __init__(
        out self,
        index: Int,
        left_node: Int,
        right_node: Int,
        n_left: Int,
        n_right: Int,
        depth: Int,
        var branch: List[Int],
        var left_bounds: OutputBounds,
        var right_bounds: OutputBounds,
        left_slot: Int,
        right_slot: Int,
    ):
        self.index = index
        self.left_node = left_node
        self.right_node = right_node
        self.n_left = n_left
        self.n_right = n_right
        self.depth = depth
        self.branch = branch^
        self.left_bounds = left_bounds^
        self.right_bounds = right_bounds^
        self.left_slot = left_slot
        self.right_slot = right_slot


def _apply_shape_rules(
    mut rec: GpuSplitRecord, n_rows: Int, depth: Int, params: TreeParams
):
    """The rules `_search` applies before it ever looks at bins, applied to a
    record the device produced.

    The depth limit and the minimum-row rules are properties of the tree, not
    of a histogram, so no kernel is told about them and every device-search
    loop clears `found` here instead. Written once so the incremental and the
    resident loop cannot cut growth at different leaves."""
    if params.max_depth > 0 and depth >= params.max_depth:
        rec.found = False
    if n_rows < 2 * params.min_data_in_leaf or n_rows < 2:
        rec.found = False


def _search_leaf_device(
    mut builder: GpuHistogramBuilder,
    mut searcher: GpuSplitSearcher,
    split_params: GpuSplitParams,
    node: Int,
    n_rows: Int,
    depth: Int,
    params: TreeParams,
    tree_features: List[Int],
    allowed: List[Bool],
    tree_index: Int,
    bounds: OutputBounds,
) raises -> GpuSplitRecord:
    """Build `node`'s histogram and search it on the device, then apply the
    shape rules `_search` applies before it ever looks at bins (the depth
    limit and the minimum-row rules are properties of the tree, not of the
    histogram, so they stay host decisions).

    The builder and the searcher share one device context, so the search
    kernels are queued behind the histogram kernels with no fence; the
    record download is the node's one host synchronization, which also
    upholds the searcher's staging contract (one node's `enqueue` completes
    before the next node's `set_allowed` restages the pinned buffers)."""
    builder.enqueue_leaf(node)
    searcher.set_features(
        select_node_features(
            tree_features,
            params.feature_fraction_bynode,
            params.feature_fraction_seed,
            tree_index,
            node,
        )
    )
    searcher.set_allowed(allowed)
    searcher.enqueue(
        builder.out_dev,
        split_params,
        builder.g_scale,
        builder.h_scale,
        bounds,
    )
    var rec = searcher.download()
    _apply_shape_rules(rec, n_rows, depth, params)
    return rec^


def _check_device_search_supported(params: TreeParams) raises:
    """Refuse a configuration the device split kernel cannot score.

    The kernel reads `GpuSplitParams`, which carries the two lambdas, the two
    child floors, and the categorical parameters. Everything in
    `TreeParams.extra`, and the per-level feature draw, would have to move
    into the kernel or into the record it returns. Until one of those
    happens, asking for them under `SPLIT_SEARCH_DEVICE` is an error, not a
    silently different tree. The host scan (the default) honors all of them.

    The range checks run first, so an out-of-range value is reported as the
    bad number it is rather than as an unsupported strategy. `is_active` tests
    for a value that would change a fit, which a negative one would not.
    """
    params.extra.check_scalars(params.min_data_in_leaf)
    if params.extra.is_active():
        raise Error(
            "the device split search does not implement min_gain_to_split,"
            " max_delta_step, path_smooth, extra_trees, monotone_penalty,"
            " feature_contri, or the CEGB costs; the kernel scores from"
            " GpuSplitParams alone. Use the host split scan, which is the"
            " default (MOJOTREES_GPU_SPLIT_STRATEGY=host, or"
            " split_search=SPLIT_SEARCH_HOST)"
        )
    if params.feature_fraction_bylevel != 1.0:
        raise Error(
            "the device split search does not implement"
            " feature_fraction_bylevel; the per-node draw it stages is taken"
            " from the tree's feature set directly. Use the host split scan,"
            " which is the default"
        )


def resident_frontier_disabled() -> Bool:
    """`MOJOTREES_GPU_SPLIT_RESIDENT=0`, which forces the device split search
    back onto its incremental loop even where the slot pool would open.

    An escape hatch and a measurement handle, not a tuning knob. It exists so
    the two device-search loops can be compared on one machine in one thermal
    state, which is the comparison the resident loop was written to win, and
    so a run that hits a slot-pool problem has somewhere to go that is not
    "use the host scan". Unset, the resident loop runs wherever it fits."""
    return getenv("MOJOTREES_GPU_SPLIT_RESIDENT") == "0"


def _node_features(
    params: TreeParams,
    tree_features: List[Int],
    tree_index: Int,
    node: Int,
) raises -> List[Int]:
    """One node's search feature set: the per-node draw
    (`feature_fraction_bynode`) taken from the node id the CPU grower would
    have assigned it. Written once so both device-search loops narrow a
    node's scan to the same features."""
    return select_node_features(
        tree_features,
        params.feature_fraction_bynode,
        params.feature_fraction_seed,
        tree_index,
        node,
    )


def _commit_device_split(
    mut tree: Tree,
    rec: GpuSplitRecord,
    split: SplitInfo,
    split_missing_bin: Int,
    parent_node: Int,
    left_node: Int,
    right_node: Int,
    parent_bounds: OutputBounds,
    signs: List[Int],
) raises -> ChildBounds:
    """Write a device-chosen split and its two child values into the tree,
    and return the intervals the children's own searches must respect.

    The same clamp-and-divide the host-search grower does, over the raw
    Newton values the record already carries instead of over a downloaded
    histogram: no-ops when unconstrained, and on a constrained feature whose
    two outputs a rounding step inverted, both collapse to their midpoint so
    the ordering stays exact. Shared by both device-search loops, which is
    what keeps a monotone fit from depending on where the histograms lived.
    """
    var split_sign = monotone_sign(signs, split.feature)
    var left_value = parent_bounds.clamp(rec.left_value)
    var right_value = parent_bounds.clamp(rec.right_value)
    if split_sign != MONOTONE_FREE and left_value > right_value:
        var mid = midpoint(left_value, right_value)
        left_value = mid
        right_value = mid
    var children = child_bounds(
        parent_bounds, split_sign, left_value, right_value
    )
    tree.value[left_node] = left_value
    tree.value[right_node] = right_value
    tree.left[parent_node] = left_node
    tree.right[parent_node] = right_node
    tree._set_split(parent_node, split, split_missing_bin)
    return children^


def _grow_tree_gpu_device_search(
    mut builder: GpuHistogramBuilder,
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
) raises -> Tree:
    """`grow_tree_gpu` with split selection on the device: every node's
    histogram is built and searched where it lives, and only a 136-byte
    record crosses to the host per node instead of the whole
    `3 * n_features * n_bins` histogram.

    Two loops implement that, and which one runs is a memory question the
    builder answers. `_device_search_resident` keeps every live leaf's
    histogram in a device slot, so a split builds only its smaller child and
    derives the larger by subtracting on the device, and searches both
    children in one launch pair. `_device_search_incremental` keeps nothing,
    so a split builds both children and searches them one at a time. The
    resident loop is the one to want: it does the same histogram work the
    host-search grower does and pays neither the per-node download nor the
    per-node wait, while the incremental loop does roughly twice the
    accumulation and waits twice per split, which is measurably slower than
    the host scan it was meant to beat.

    Residency needs one slot per live leaf for the whole tree, so it is all
    or nothing: `builder.open_resident(params.num_leaves)` declines when a
    dataset is wide enough that `num_leaves` full-width histograms exceed the
    pool budget, and the incremental loop is the fallback rather than a
    stranded leaf. It also needs `min_data_in_leaf >= 1`, since a subtraction
    is only worth taking when the child that *is* built is nonempty, and that
    floor is what the search kernel enforces to guarantee it.
    `MOJOTREES_GPU_SPLIT_RESIDENT=0` forces the incremental loop where the
    resident one would have fit, which is how the two are compared.

    Gains, hessian tests, and leaf values are Float32 on the device, so a
    near-tie between two candidates can resolve differently than the host
    scan and CPU/GPU tree shapes can differ there; child row counts are
    exact integers either way. Selection is still bit-deterministic run to
    run. Shape rules (depth limit, minimum rows), monotone clamping with
    the midpoint collapse, interaction masks, and per-node feature
    subsampling all stay identical to the host path.

    What this path does *not* implement is the `params.extra` bundle and the
    per-level feature draw. `GpuSplitParams` carries lambda_l1, lambda_l2, the
    two child floors, and the categorical parameters, and the kernel scores
    from those alone: there is nowhere in it to charge a gain floor, a
    per-feature multiplier, a CEGB cost, a monotone penalty, a drawn
    threshold, or a capped and smoothed child output. Rather than return
    trees that quietly ignore what the caller asked for, an active setting is
    refused here and the caller is pointed at the host scan, which honors all
    of it. `_check_device_search_supported` is that refusal."""
    _check_device_search_supported(params)
    check_grow_policy(params.grow_policy)
    params.constraints.check_features(builder.n_features)
    params.monotone.check_features(builder.n_features)
    var signs = params.monotone.active_signs()
    var tree_features = select_tree_features(
        builder.n_features,
        params.feature_fraction,
        params.feature_fraction_seed,
        tree_index,
    )
    builder.set_features(tree_features)
    var searcher = GpuSplitSearcher(
        builder.ctx,
        builder.n_features,
        builder.n_bins,
        builder.missing_bin,
        builder.cats,
        max_records=_search_record_slots(params, builder.n_features),
    )
    searcher.set_monotone(signs)
    var split_params = GpuSplitParams(
        params.lambda_reg,
        params.lambda_l1,
        params.min_child_hess,
        params.min_data_in_leaf,
        params.cat.copy(),
    )

    builder.begin_tree(bag)
    var n_root = len(bag) if len(bag) > 0 else builder.n_rows
    if (
        not resident_frontier_disabled()
        and params.min_data_in_leaf >= 1
        and builder.open_resident(params.num_leaves)
    ):
        return _device_search_resident(
            builder,
            searcher,
            split_params,
            params,
            tree_features,
            signs,
            tree_index,
            n_root,
        )
    return _device_search_incremental(
        builder,
        searcher,
        split_params,
        params,
        tree_features,
        signs,
        tree_index,
        n_root,
    )


def _device_search_incremental(
    mut builder: GpuHistogramBuilder,
    mut searcher: GpuSplitSearcher,
    split_params: GpuSplitParams,
    params: TreeParams,
    tree_features: List[Int],
    signs: List[Int],
    tree_index: Int,
    n_root: Int,
) raises -> Tree:
    """Device split selection with nothing kept between nodes.

    Both of a split's children are accumulated from their own rows, because
    the parent's histogram is gone by then: it was written into the builder's
    single-node output buffer, which the next node's build overwrites. That
    is roughly twice the accumulation the subtraction trick needs, and each
    child's record is a wait of its own, so this is the slower of the two
    device-search loops by a wide margin. It stays because it needs no slot
    pool at all, which is what a dataset wide enough to price residency out
    is left with; see `_grow_tree_gpu_device_search`.

    The caller has already opened the tree (`begin_tree`), narrowed the
    feature set, and staged the searcher's monotone vector.
    """
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )
    var root = tree._add_node(0.0, Float64(n_root))
    var root_branch = List[Int]()
    var root_rec = _search_leaf_device(
        builder,
        searcher,
        split_params,
        root,
        n_root,
        0,
        params,
        tree_features,
        params.constraints.allowed_features(root_branch),
        tree_index,
        OutputBounds.unbounded(),
    )
    tree.value[root] = root_rec.parent_value

    var frontier = List[_GpuRecordLeafState]()
    frontier.append(
        _GpuRecordLeafState(root, n_root, root_rec^, root_branch^, depth=0)
    )
    var n_leaves = 1
    var schedule = GrowthSchedule(params.grow_policy)

    while n_leaves < params.num_leaves:
        # The growth policy picks (growth_policy.mojo), exactly as the
        # host-search loop does: best gain anywhere in the tree, ties to the
        # lower frontier index, under leaf-wise growth; the planned level's
        # next node under depth-wise growth.
        var cands = List[LeafCandidate](capacity=len(frontier))
        for i in range(len(frontier)):
            cands.append(
                LeafCandidate(
                    frontier[i].node,
                    frontier[i].depth,
                    frontier[i].rec.gain,
                    frontier[i].rec.found and frontier[i].rec.gain > 0.0,
                )
            )
        var best_i = schedule.next_leaf(
            cands, n_leaves, params.num_leaves, params.max_depth
        )
        if best_i < 0:
            break

        var parent_node = frontier[best_i].node
        var rec = frontier[best_i].rec.copy()
        var split = rec.to_split_info()
        var split_missing_bin = -1 if split.is_categorical else (
            builder.missing_bin[split.feature]
        )
        # Exact integers off the record, from the same histogram counts the
        # host `_count_left` would sum.
        var n_left = rec.left.count
        var n_right = rec.right.count

        var left_node = tree._add_node(0.0, Float64(n_left))
        var right_node = tree._add_node(0.0, Float64(n_right))
        builder.apply_split(
            split.feature,
            split.bin,
            parent_node,
            left_node,
            right_node,
            split_missing_bin,
            split.default_left,
            split.is_categorical,
            split.cat_bitset,
            expected_left=n_left,
        )

        var children = _commit_device_split(
            tree,
            rec,
            split,
            split_missing_bin,
            parent_node,
            left_node,
            right_node,
            frontier[best_i].bounds,
            signs,
        )

        var branch = extend_branch(frontier[best_i].branch, split.feature)
        var allowed = params.constraints.allowed_features(branch)
        var child_depth = frontier[best_i].depth + 1
        var left_rec = _search_leaf_device(
            builder,
            searcher,
            split_params,
            left_node,
            n_left,
            child_depth,
            params,
            tree_features,
            allowed,
            tree_index,
            children.left,
        )
        var right_rec = _search_leaf_device(
            builder,
            searcher,
            split_params,
            right_node,
            n_right,
            child_depth,
            params,
            tree_features,
            allowed,
            tree_index,
            children.right,
        )

        frontier[best_i] = _GpuRecordLeafState(
            left_node,
            n_left,
            left_rec^,
            branch.copy(),
            depth=child_depth,
            bounds=children.left.copy(),
        )
        frontier.append(
            _GpuRecordLeafState(
                right_node,
                n_right,
                right_rec^,
                branch^,
                depth=child_depth,
                bounds=children.right.copy(),
            )
        )
        n_leaves += 1

    tree.n_leaves = n_leaves
    return tree^


def _device_search_resident(
    mut builder: GpuHistogramBuilder,
    mut searcher: GpuSplitSearcher,
    split_params: GpuSplitParams,
    params: TreeParams,
    tree_features: List[Int],
    signs: List[Int],
    tree_index: Int,
    n_root: Int,
) raises -> Tree:
    """Device split selection over a device-resident frontier.

    Every live leaf holds a histogram slot for as long as it is a leaf, which
    is what turns a split into one histogram build instead of two: the
    smaller child is accumulated from its own rows into a fresh slot, and the
    larger is derived by subtracting it from the parent's slot, in place, on
    the device. That is exactly the arithmetic the host-search grower does
    with `subtract_histogram`, moved to where the histograms already are, and
    it is exact for the same reason — accumulation is fixed-point Int32 under
    one scale for the whole tree, so a parent's bins are the exact integer
    sum of its children's. `subtraction_builds_left` picks which child is
    built, the same test `grow_tree` and `grow_tree_gpu` use, so no two
    growers can disagree about which histogram a slot holds.

    A batch of splits is committed, enqueued, and searched together, and the
    batch is what `GrowthSchedule.plan_level` hands over: one split under
    leaf-wise growth, because the next pick depends on the frontier this one
    changes, and a whole planned level under depth-wise growth, whose
    admissions and order are all decided before any of them runs. Each
    split in a batch costs one histogram build with the sibling subtraction
    folded into it; the batch as a whole costs one search launch pair, one
    wait, and 136 bytes per child across the bus. The host-search grower
    pays one build, one wait, a `3 * n_features * n_bins` download, a host
    subtraction, and two host scans per split; the incremental device loop
    pays two builds and two waits per split.

    Batching a level is safe because a level's splits are independent: their
    parents own disjoint row windows, so no two partitions touch the same
    rows; each child's histogram lands in its own pool slot; and no split in
    a batch reads a frontier entry another writes, since the frontier is
    updated only after the wait. The in-order queue (gpu_runtime.mojo) is
    what serializes the scratch buffers the partitions share. Nothing about
    a decision changes, so the tree is the one the same schedule would have
    grown one split at a time.

    What a split's fixed cost actually is
    -------------------------------------
    Under leaf-wise growth, eight launches and one wait, and on an Apple M4
    (`pixi run bench-launch-cost`, which measures both directly) that is
    about 20us of enqueue per launch and about 126us for the wait: roughly
    280us a split, or 0.85s of a 3.05s run at 50000 x 100. Read those two
    numbers before proposing anything whose whole benefit is fewer launches
    or fewer waits, because they set the price. One launch removed is worth
    about 60ms over a default run, near 2%, which is inside the benchmark
    harness's noise floor -- so a fusion has to justify itself as strictly
    less work for an identical result, not by a measured speedup.

    The eight are four for the row partition (flag scan, block-sum scan,
    scatter, copy back), two or three for the histogram (a conditional
    zeroing, then either the atomic kernel or the partial and reduce pair),
    and two for the split search. Six of them are per split whatever the
    batch; the two search launches and the wait are per batch. A depth-wise
    level of `L` splits therefore pays `6L + 2` launches and one wait rather
    than `8L` and `L`, which at the default 31 leaves is 5 waits for a tree
    instead of 30. That is a count, not a measurement: no benchmark of
    depth-wise growth on the device has been run, and the launch and wait
    prices above are what it would have to be read against.

    The wait cannot go below one per batch: the host chooses the next
    leaves, writes the tree, and draws per-node feature subsets from the
    records the batch brought home.

    Three fusions have already been examined and are not open. The output
    zeroing is skipped already whenever the tiled path builds a full feature
    set (`gpu_active_rows.enqueue_range_histogram`). Folding the split
    search's per-record reduce into its per-slot scan needs a wider
    threadgroup, and each scan thread owns a `MAX_SPLIT_BINS` categorical
    sort scratch, so the shared allocation grows with the block -- the same
    occupancy trap that made the earlier scan reshape measure inside noise.
    Dropping the partition's copy-back needs a per-buffer staleness parity
    carried across splits, because the scatter writes only the parent's
    window and swapping whole buffers would invalidate every other leaf's
    range; that is bookkeeping, not fusion.

    Nothing about a decision changes. The batch stages each node's own
    feature set, allow mask, and monotone interval into its own record, the
    scan order inside a node is unchanged, and the records are the ones the
    same nodes searched one at a time would produce. Slot residency is a
    memory decision, not a numeric one.

    The caller has already opened the tree (`begin_tree`), narrowed the
    feature set, staged the searcher's monotone vector, and confirmed with
    `builder.open_resident` that the pool is deep enough for `num_leaves`.
    """
    # A tree's slots die with it: the next tree repartitions every row, so no
    # histogram here is readable by it. Releasing on the way in as well as on
    # the way out means an error that escapes mid-tree cannot leak a frontier
    # into the following one.
    builder.release_resident_all()
    # Wall-clock phase attribution, printed per tree under
    # `MOJOTREES_GPU_PHASE_TRACE=1`. The kernels here queue with no fence and
    # normally collapse into `download_frontier`'s one wait, so a traced run
    # inserts a sync after the partition and after the histogram build to
    # separate the phases, paying two extra device waits per split that an
    # untraced run does not.
    var phase_trace = getenv("MOJOTREES_GPU_PHASE_TRACE") == "1"
    var t_partition = 0.0
    var t_hist = 0.0
    var t_search = 0.0
    var tree_t0 = perf_counter_ns()
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )
    var root = tree._add_node(0.0, Float64(n_root))
    var root_branch = List[Int]()
    var root_slot = builder.acquire_resident(root)
    if root_slot < 0:
        raise Error("the resident histogram pool is full at the root")
    var hist_t0 = perf_counter_ns()
    builder.enqueue_resident_leaf(root, root_slot)
    if phase_trace:
        builder.ctx.synchronize()
    t_hist += Float64(perf_counter_ns() - hist_t0) / 1e9
    var root_batch = List[SplitNodeRequest]()
    root_batch.append(
        SplitNodeRequest(
            root_slot,
            _node_features(params, tree_features, tree_index, root),
            params.constraints.allowed_features(root_branch),
            OutputBounds.unbounded(),
        )
    )
    # The searcher shares the builder's context, so these kernels queue
    # behind the histogram build with no fence, and they read the pool
    # buffer the build just wrote rather than a copy of it.
    var search_t0 = perf_counter_ns()
    searcher.enqueue_frontier(
        builder.batcher[0].out_dev,
        root_batch,
        split_params,
        builder.g_scale,
        builder.h_scale,
    )
    var root_recs = searcher.download_frontier(1)
    t_search += Float64(perf_counter_ns() - search_t0) / 1e9
    var root_rec = root_recs[0].copy()
    _apply_shape_rules(root_rec, n_root, 0, params)
    tree.value[root] = root_rec.parent_value

    var frontier = List[_GpuRecordLeafState]()
    frontier.append(
        _GpuRecordLeafState(
            root, n_root, root_rec^, root_branch^, depth=0, slot=root_slot
        )
    )
    var n_leaves = 1
    var schedule = GrowthSchedule(params.grow_policy)
    # A batch is a host wait, so a level wider than the searcher's record
    # capacity becomes several of them rather than one oversized launch.
    # Two records per split, since both children are searched.
    var per_batch = searcher.max_records // 2
    if per_batch < 1:
        per_batch = 1

    while n_leaves < params.num_leaves:
        # The growth policy picks (growth_policy.mojo), exactly as the
        # host-search loop does: best gain anywhere in the tree, ties to the
        # lower frontier index, under leaf-wise growth; the whole planned
        # level, in ascending node id, under depth-wise growth.
        var cands = List[LeafCandidate](capacity=len(frontier))
        for i in range(len(frontier)):
            cands.append(
                LeafCandidate(
                    frontier[i].node,
                    frontier[i].depth,
                    frontier[i].rec.gain,
                    frontier[i].rec.found and frontier[i].rec.gain > 0.0,
                )
            )
        var picks = schedule.plan_level(
            cands, n_leaves, params.num_leaves, params.max_depth
        )
        if len(picks) == 0:
            break

        # Everything a batch's splits enqueue is independent: their parents
        # hold disjoint row windows, so the partitions do not overlap, and
        # each child's histogram lands in its own pool slot. The queue is in
        # order (gpu_runtime.mojo), so the scratch every partition shares is
        # drained by one before the next writes it, and no split in a batch
        # reads a frontier entry another split in the same batch writes: the
        # frontier updates all happen below, after the batch's one wait.
        var taken = 0
        while taken < len(picks):
            var upto = taken + per_batch
            if upto > len(picks):
                upto = len(picks)
            var batch = List[SplitNodeRequest](capacity=2 * (upto - taken))
            var pending = List[_GpuPendingSplit](capacity=upto - taken)

            for pick in range(taken, upto):
                _enqueue_resident_split(
                    builder,
                    tree,
                    frontier,
                    picks[pick],
                    params,
                    tree_features,
                    signs,
                    tree_index,
                    batch,
                    pending,
                    t_partition,
                    t_hist,
                    phase_trace,
                )

            search_t0 = perf_counter_ns()
            searcher.enqueue_frontier(
                builder.batcher[0].out_dev,
                batch,
                split_params,
                builder.g_scale,
                builder.h_scale,
            )
            # The batch's one wait, and what upholds the staging contracts on
            # both sides: the batcher's pinned item table and the searcher's
            # pinned node tables are only restaged after this returns.
            var recs = searcher.download_frontier(len(batch))
            t_search += Float64(perf_counter_ns() - search_t0) / 1e9

            for k in range(len(pending)):
                var left_rec = recs[2 * k].copy()
                var right_rec = recs[2 * k + 1].copy()
                _apply_shape_rules(
                    left_rec, pending[k].n_left, pending[k].depth, params
                )
                _apply_shape_rules(
                    right_rec, pending[k].n_right, pending[k].depth, params
                )
                frontier[pending[k].index] = _GpuRecordLeafState(
                    pending[k].left_node,
                    pending[k].n_left,
                    left_rec^,
                    pending[k].branch.copy(),
                    depth=pending[k].depth,
                    bounds=pending[k].left_bounds.copy(),
                    slot=pending[k].left_slot,
                )
                frontier.append(
                    _GpuRecordLeafState(
                        pending[k].right_node,
                        pending[k].n_right,
                        right_rec^,
                        pending[k].branch.copy(),
                        depth=pending[k].depth,
                        bounds=pending[k].right_bounds.copy(),
                        slot=pending[k].right_slot,
                    )
                )
                n_leaves += 1
            taken = upto

    if phase_trace:
        var tree_s = Float64(perf_counter_ns() - tree_t0) / 1e9
        print(
            "phase_trace tree",
            tree_index,
            "total_s",
            tree_s,
            "hist_s",
            t_hist,
            "partition_s",
            t_partition,
            "search_s",
            t_search,
            "other_s",
            tree_s - t_hist - t_partition - t_search,
        )

    tree.n_leaves = n_leaves
    builder.release_resident_all()
    return tree^


def _enqueue_resident_split(
    mut builder: GpuHistogramBuilder,
    mut tree: Tree,
    mut frontier: List[_GpuRecordLeafState],
    index: Int,
    params: TreeParams,
    tree_features: List[Int],
    signs: List[Int],
    tree_index: Int,
    mut batch: List[SplitNodeRequest],
    mut pending: List[_GpuPendingSplit],
    mut t_partition: Float64,
    mut t_hist: Float64,
    phase_trace: Bool,
) raises:
    """Commit one split of `frontier[index]` into `tree` and enqueue the
    device work its two children need, without waiting for any of it.

    This is the body of `_device_search_resident`'s loop up to but not
    including the search: the row partition, the tree write, the built
    child's histogram with the sibling subtraction folded in, and the two
    search requests appended to `batch`. What the caller still owes is one
    `enqueue_frontier` over `batch` and one `download_frontier`, after which
    `pending` says where each pair of records belongs.

    It is a separate function because a batch calls it several times before
    it waits, and because the frontier entry it reads must not be one
    another call in the same batch has written: nothing here writes
    `frontier` at all, which is what makes that true by construction rather
    than by reading the loop.
    """
    var parent_node = frontier[index].node
    var parent_slot = frontier[index].slot
    var rec = frontier[index].rec.copy()
    var split = rec.to_split_info()
    var split_missing_bin = -1 if split.is_categorical else (
        builder.missing_bin[split.feature]
    )
    # Exact integers off the record, from the same histogram counts the
    # host `_count_left` would sum.
    var n_left = rec.left.count
    var n_right = rec.right.count

    var left_node = tree._add_node(0.0, Float64(n_left))
    var right_node = tree._add_node(0.0, Float64(n_right))
    var part_t0 = perf_counter_ns()
    builder.apply_split(
        split.feature,
        split.bin,
        parent_node,
        left_node,
        right_node,
        split_missing_bin,
        split.default_left,
        split.is_categorical,
        split.cat_bitset,
        expected_left=n_left,
    )
    if phase_trace:
        builder.ctx.synchronize()
    t_partition += Float64(perf_counter_ns() - part_t0) / 1e9

    var children = _commit_device_split(
        tree,
        rec,
        split,
        split_missing_bin,
        parent_node,
        left_node,
        right_node,
        frontier[index].bounds,
        signs,
    )

    # The subtraction trick, device side, folded into the build. The built
    # child gets a fresh slot; the derived one takes over the parent's,
    # which is what keeps the pool at one slot per live leaf rather than one
    # per node. The subtraction rides along inside the histogram kernel, so
    # a split spends one launch here rather than two and never makes a
    # slot-sized pass over the pool to do it.
    var build_left = subtraction_builds_left(n_left, n_right)
    var built_node = left_node if build_left else right_node
    var derived_node = right_node if build_left else left_node
    var hist_t0 = perf_counter_ns()
    var built_slot = builder.acquire_resident(built_node)
    if built_slot < 0:
        raise Error(
            "the resident histogram pool ran out mid-tree; it is sized"
            " for num_leaves slots, so this means the pool and the leaf"
            " budget disagree"
        )
    builder.enqueue_resident_leaf_subtracting(
        built_node, built_slot, parent_slot
    )
    builder.reown_resident(parent_slot, derived_node)
    if phase_trace:
        builder.ctx.synchronize()
    t_hist += Float64(perf_counter_ns() - hist_t0) / 1e9
    var left_slot = built_slot if build_left else parent_slot
    var right_slot = parent_slot if build_left else built_slot

    # Both children inherit the same branch feature set, so they share one
    # allow mask, and both sit one edge below the leaf that split.
    var branch = extend_branch(frontier[index].branch, split.feature)
    var allowed = params.constraints.allowed_features(branch)
    var child_depth = frontier[index].depth + 1
    batch.append(
        SplitNodeRequest(
            left_slot,
            _node_features(params, tree_features, tree_index, left_node),
            allowed.copy(),
            children.left.copy(),
        )
    )
    batch.append(
        SplitNodeRequest(
            right_slot,
            _node_features(params, tree_features, tree_index, right_node),
            allowed^,
            children.right.copy(),
        )
    )
    pending.append(
        _GpuPendingSplit(
            index,
            left_node,
            right_node,
            n_left,
            n_right,
            child_depth,
            branch^,
            children.left.copy(),
            children.right.copy(),
            left_slot,
            right_slot,
        )
    )


def grow_tree_gpu(
    mut builder: GpuHistogramBuilder,
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
    split_search: Int = SPLIT_SEARCH_AUTO,
    data: BinnedMatrix = BinnedMatrix(List[UInt8](), 0, 0, 0),
) raises -> Tree:
    """Grow one tree, leaf-wise, with histogram accumulation and row
    partitioning on the GPU. Gradients for this round must already be
    uploaded via `builder.upload_gradients`. Node ids double as device-side
    leaf ids, and nodes are created in the same order as the CPU
    `grow_tree`, so equal split decisions yield identical tree layouts.

    A non-empty `bag` restricts growth to those rows, exactly as in
    `grow_tree`: only the bag's rows are seeded into the root's device-side
    row range (see gpu_active_rows.mojo), so bagged rows are the only rows
    any histogram, count, or split on this tree sees. Both backends take the
    bag from the same sampler, so the two grow on identical rows.

    Interaction constraints are tracked exactly as in `grow_tree`: the same
    branch feature sets, the same allow masks, and the same `_search` entry
    point. Constraint enforcement is therefore identical on both backends,
    independent of the Float32 histogram precision the GPU accumulates in.

    `params.max_depth` is tracked the same way, as a per-frontier-leaf depth
    incremented on each split and checked inside `_search`. Since the depth
    limit depends only on tree shape and not on histogram values, the two
    backends cut growth at exactly the same leaves.

    Feature subsampling likewise goes through the same sampler as the CPU
    grower: `tree_index` and `params.feature_fraction_seed` fix the tree's
    feature set, which is handed to the device once per tree so its
    histogram kernel accumulates exactly those features, and the per-node
    sets (drawn from the node ids, which both growers assign in the same
    order) narrow each split search identically.

    Monotonic constraints go through the same `_search` and the same interval
    bookkeeping as on the CPU. Split search, leaf clamping, and candidate
    rejection all run host-side on downloaded histograms, so the constraint is
    enforced identically on both backends; only the histogram sums the
    decisions are made from carry the GPU's Float32 precision.

    `split_search` picks where that split selection runs (see the
    SPLIT_SEARCH_* constants above): explicit requests are honored, while
    AUTO consults `gpu_split_policy` and otherwise stays on this host scan.
    The device-side scan trades the identical-split guarantee for compact
    records and resident frontier processing.

    Where a split's two children's histograms come from is the builder's
    launch decision, not this grower's: `builder.batches_nodes` answers it
    from `apple_histogram_policy`, and this loop either takes both children
    from one batched launch or builds the smaller one and subtracts. The two
    produce the same pair of histograms exactly, since fixed-point Int32
    accumulation under one scale makes a parent's bins the exact integer sum
    of its children's, so no split decision can tell which ran.

    `data` is the caller's host-resident binned matrix, and passing it is
    what makes hybrid CPU/GPU leaf scheduling possible at all
    (`MOJOTREES_HYBRID_LEAVES` + `MOJOTREES_HYBRID_COSTS`; see
    hybrid_leaf_scheduler.mojo). The default is an empty matrix, under which
    hybrid scheduling stays off and this grower is exactly the pure-device
    one — as it also is whenever the environment does not opt in."""
    if (
        resolve_split_search_for(builder, params, split_search)
        == SPLIT_SEARCH_DEVICE
    ):
        return _grow_tree_gpu_device_search(builder, params, bag, tree_index)
    check_grow_policy(params.grow_policy)
    params.constraints.check_features(builder.n_features)
    params.monotone.check_features(builder.n_features)
    check_feature_fractions(
        params.feature_fraction,
        params.feature_fraction_bynode,
        params.feature_fraction_bylevel,
    )
    # This grower applies the whole `extra` bundle, so it validates it against
    # this dataset before the first histogram, as the CPU grower does.
    params.extra.check(
        builder.n_features,
        params.num_leaves,
        params.max_depth,
        params.min_data_in_leaf,
    )
    var max_delta_step = params.extra.max_delta_step
    var path_smooth = params.extra.path_smooth
    var signs = params.monotone.active_signs()
    var tree_features = select_tree_features(
        builder.n_features,
        params.feature_fraction,
        params.feature_fraction_seed,
        tree_index,
    )
    builder.set_features(tree_features)
    # Leaf-value totals must come from a feature the histograms accumulated.
    var value_feature = tree_features[0]
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )

    # Wall-clock phase attribution, printed per tree under
    # `MOJOTREES_GPU_PHASE_TRACE=1`. The traced sync after `apply_split` is
    # what separates partition time from the histogram build that would
    # otherwise absorb it, so a traced run pays one extra device wait per
    # split (~0.15ms) that an untraced run does not.
    var phase_trace = getenv("MOJOTREES_GPU_PHASE_TRACE") == "1"
    var t_partition = 0.0
    var t_hist = 0.0
    var t_search = 0.0
    var tree_t0 = perf_counter_ns()

    builder.begin_tree(bag)
    var n_root = len(bag) if len(bag) > 0 else builder.n_rows

    # --- Hybrid leaf scheduling (MOJOTREES_HYBRID_LEAVES) -----------------
    # Off unless the environment asks, and a no-op when it does not: the
    # branch below is the only cost an ordinary tree pays. This grower wires
    # the two fit-preserving modes only. `mirror` builds a host replica
    # alongside accepted device builds and compares them bit for bit, with
    # the device's histogram always the one consumed; `replica` substitutes
    # host builds — but only after a mirror comparison on this builder has
    # verified the replica (`replica_state`), so a substituted tree is
    # bit-identical to the one the pure-device grower grows. `float64`
    # changes the fit and is deliberately not wired here.
    #
    # Rows for a host build are read back per node
    # (`builder.readback_range`, a sub-buffer view of exactly the node's
    # window: `4 * node_rows` bytes and one synchronize), never from a
    # maintained mirror and never from a whole-permutation snapshot. The
    # design (HYBRID_TRAINING.md §3, §8.1) assumed a per-range readback was
    # not expressible and built the snapshot around that; it is, so every
    # leaf is planned as `ROWS_DEVICE_COPY` and charged only its own
    # transfer. Nothing is amortized across leaves, so a host placement can
    # never make a tree slower than the pure-device path, and the election
    # no longer depends on the dataset's size at all — only on the leaf's.
    #
    # `guard_transfer_dominates` is off here: under a per-range readback the
    # "transfer" is the fixed synchronize plus a few bytes, and the device
    # path pays that same synchronize on its histogram download, so the
    # guard would only refuse the small leaves the comparison exists to
    # move (measured either way in
    # bench/results/apple_m4_hybrid_costs_2026-08-15.md; predictions are
    # identical either way, only the time differs).
    var hy_mode = env_hybrid_mode()
    var hy_costs = HybridCosts.unmeasured()
    var hybrid_live = False
    if hy_mode == MODE_MIRROR or hy_mode == MODE_REPLICA:
        hy_costs = env_hybrid_costs()
        var hy_shape_ok = (
            data.n_rows == builder.n_rows
            and data.n_features == builder.n_features
            and data.n_bins == builder.n_bins
        )
        if (
            hy_costs.measured
            and hy_shape_ok
            and builder.replica_state != REPLICA_REFUTED
            and builder.has_gradients
            and not builder.gradients_host
        ):
            # Device-produced gradients (the unbagged built-in objectives,
            # the native softmax paths): read the exact Float32 the kernels
            # use back into the staging buffers once per tree, so the host
            # replica has the same inputs it has on the upload path. That
            # is `2 * 4 * n_rows` bytes and one synchronize per tree, the
            # only price this path pays for host builds; a `replica` run
            # that never elects a leaf is slower by exactly that.
            builder.stage_from_device()
        hybrid_live = (
            hy_costs.measured
            and hy_shape_ok
            and builder.gradients_host
            and builder.replica_state != REPLICA_REFUTED
        )
    var hy_rows = List[Int32]()
    var hy_fixed = List[Int32]()
    var hy_hist = Histogram.zeroed(1, 1)
    if hybrid_live:
        hy_hist = Histogram.zeroed(builder.n_features, builder.n_bins)
    var hy_host_builds = 0
    var hy_mirror_matches = 0
    var hy_mirror_mismatches = 0
    var hybrid_trace = getenv("MOJOTREES_HYBRID_TRACE") == "1"
    var hy_guard_transfer = getenv("MOJOTREES_HYBRID_GUARD_TRANSFER") == "1"

    var root = tree._add_node(0.0, Float64(n_root))
    var hist_t0 = perf_counter_ns()
    var root_hist = builder.build_leaf(root)
    t_hist += Float64(perf_counter_ns() - hist_t0) / 1e9
    # Valued before the search, because path smoothing makes a candidate's
    # children shrink toward this value; the root smooths toward 0.0.
    tree.value[root] = _leaf_value(
        root_hist,
        params.lambda_reg,
        params.lambda_l1,
        value_feature,
        n_root,
        0.0,
        max_delta_step,
        path_smooth,
    )
    var root_branch = List[Int]()
    var search_t0 = perf_counter_ns()
    var root_split = _search(
        root_hist,
        n_root,
        params,
        params.constraints.allowed_features(root_branch),
        select_split_features(
            tree_features,
            params.feature_fraction_bylevel,
            params.feature_fraction_bynode,
            params.feature_fraction_seed,
            tree_index,
            0,
            root,
        ),
        depth=0,
        missing_bins=builder.missing_bin,
        monotone=signs,
        cats=builder.cats,
        node=root,
        tree_index=tree_index,
        parent_output=tree.value[root],
        grower_applies_extra=True,
    )
    t_search += Float64(perf_counter_ns() - search_t0) / 1e9

    var frontier = List[_GpuLeafState]()
    frontier.append(
        _GpuLeafState(
            root, n_root, root_hist^, root_split^, root_branch^, depth=0
        )
    )
    var n_leaves = 1
    var schedule = GrowthSchedule(params.grow_policy)

    while n_leaves < params.num_leaves:
        # The growth policy picks (growth_policy.mojo): best gain anywhere in
        # the tree under leaf-wise growth, the planned level's next node
        # under depth-wise growth.
        var cands = List[LeafCandidate](capacity=len(frontier))
        for i in range(len(frontier)):
            cands.append(
                LeafCandidate(
                    frontier[i].node,
                    frontier[i].depth,
                    frontier[i].split.gain,
                    frontier[i].split.found and frontier[i].split.gain > 0.0,
                )
            )
        var best_i = schedule.next_leaf(
            cands, n_leaves, params.num_leaves, params.max_depth
        )
        if best_i < 0:
            break

        var parent_node = frontier[best_i].node
        var split = frontier[best_i].split.copy()
        var split_missing_bin = -1 if split.is_categorical else (
            builder.missing_bin[split.feature]
        )
        var n_left = _count_left(
            frontier[best_i].hist, split, split_missing_bin
        )
        var n_right = frontier[best_i].n_rows - n_left

        # The row counts come off the parent's exact histogram counts, the
        # same numbers the CPU grower gets from its row lists, so node covers
        # match across backends.
        var left_node = tree._add_node(0.0, Float64(n_left))
        var right_node = tree._add_node(0.0, Float64(n_right))
        var part_t0 = perf_counter_ns()
        builder.apply_split(
            split.feature,
            split.bin,
            parent_node,
            left_node,
            right_node,
            split_missing_bin,
            split.default_left,
            split.is_categorical,
            split.cat_bitset,
            expected_left=n_left,
        )
        if phase_trace:
            builder.ctx.synchronize()
        t_partition += Float64(perf_counter_ns() - part_t0) / 1e9

        # Both children, however the launch policy wants them. Batched,
        # they go up in one packed launch over exactly their own row ranges
        # (gpu_leaf_batching.mojo); otherwise the subtraction trick builds
        # the smaller child and derives the sibling on the host.
        #
        # The two produce the same pair of histograms, and exactly, not to a
        # tolerance: accumulation is fixed-point Int32 under one scale for
        # the whole tree, and a parent's bins are the exact integer sum of
        # its children's, so subtracting one built child and building both
        # children agree bin for bin. Which one runs is therefore a launch
        # decision no split can observe.
        var left_hist = Histogram.zeroed(1, 1)
        var right_hist = Histogram.zeroed(1, 1)
        var child_nodes: List[Int] = [left_node, right_node]
        hist_t0 = perf_counter_ns()
        var hy_direct_done = False
        if hybrid_live and not builder.batches_nodes(child_nodes):
            var verified = builder.replica_state == REPLICA_VERIFIED
            # While the replica claim is untested, a `replica` run places as
            # a mirror: its first accepted leaf is the comparison that can
            # verify the claim, and the device's histogram is still the one
            # consumed, so the tree cannot depend on the outcome.
            var eff_mode = hy_mode
            if hy_mode == MODE_REPLICA and not verified:
                eff_mode = MODE_MIRROR
            var hy_ctx = HybridContext(
                eff_mode,
                False,  # this is the host-search path
                True,  # gradients are host-staged (gated at tree start)
                True,  # the caller's binned matrix is in hand
                verified,
                hy_guard_transfer,
                n_root,
                False,  # no snapshot exists or is ever taken
                hy_costs.copy(),
            )
            # `plan_split`'s direct child is `n_left <= n_right`, the same
            # tie `subtraction_builds_left` breaks, so the host builds
            # exactly the child the device would have built.
            var hy_plan = plan_split(
                hy_ctx,
                left_node,
                n_left,
                right_node,
                n_right,
                len(builder.active),
                builder.n_features,
                builder.n_bins,
                builder.n_rows,
                left_row_source=ROWS_DEVICE_COPY,
                right_row_source=ROWS_DEVICE_COPY,
                gpu_launches=builder.histogram_plan(
                    min(n_left, n_right)
                ).tiling().launches(),
                parent_materialized=True,
            )
            if hy_plan.placement.builds_on_host():
                # Per-range readback: the partition just enqueued is drained
                # by the copy's synchronize, so the child's range indexes the
                # device permutation exactly and only its window moves.
                var hy_win = builder.range_of(hy_plan.direct_node)
                if hy_win.count() == hy_plan.direct_rows:
                    builder.readback_range(
                        hy_win.begin, hy_win.count(), hy_rows
                    )
                    builder.build_leaf_host_replica(
                        hy_hist,
                        hy_fixed,
                        data,
                        hy_rows,
                        0,
                        hy_win.count(),
                    )
                    hy_host_builds += 1
                    if hy_plan.placement.device == PLACE_BOTH:
                        # Mirror: the host build is compared and discarded;
                        # the device's histogram is the one consumed, so a
                        # mismatch is a diagnostic, not a fork in the tree.
                        var dev_hist = builder.build_leaf(hy_plan.direct_node)
                        if _histograms_match(dev_hist, hy_hist):
                            hy_mirror_matches += 1
                            if builder.replica_state == REPLICA_UNTESTED:
                                builder.replica_state = REPLICA_VERIFIED
                        else:
                            hy_mirror_mismatches += 1
                            builder.replica_state = REPLICA_REFUTED
                            hybrid_live = False
                        if hy_plan.direct_is_left:
                            left_hist = dev_hist^
                            right_hist = subtract_histogram(
                                frontier[best_i].hist, left_hist
                            )
                        else:
                            right_hist = dev_hist^
                            left_hist = subtract_histogram(
                                frontier[best_i].hist, right_hist
                            )
                    else:
                        # Verified replica: the host histogram substitutes,
                        # and the launch, download, synchronize, and
                        # conversion the device build would have cost never
                        # happen. Bit-identical by the verified claim, so
                        # the sibling subtraction is the same arithmetic it
                        # always was.
                        if hy_plan.direct_is_left:
                            left_hist = hy_hist.copy()
                            right_hist = subtract_histogram(
                                frontier[best_i].hist, left_hist
                            )
                        else:
                            right_hist = hy_hist.copy()
                            left_hist = subtract_histogram(
                                frontier[best_i].hist, right_hist
                            )
                    hy_direct_done = True
                else:
                    # The device's range table and the plan disagree about
                    # this node — nothing has been consumed yet, so fall
                    # back to the device path and stop scheduling.
                    hybrid_live = False
        if not hy_direct_done:
            if builder.batches_nodes(child_nodes):
                var pair = builder.build_leaves(child_nodes)
                left_hist = pair[0].copy()
                right_hist = pair[1].copy()
            elif subtraction_builds_left(n_left, n_right):
                # Which child is built and which is derived is
                # `gpu_frontier.subtraction_builds_left`, whose docstring
                # names this test and `grow_tree`'s as the two it matches.
                # Written once so a batched grower and this one cannot pick
                # different children and then disagree about which histogram
                # a slot holds.
                left_hist = builder.build_leaf(left_node)
                right_hist = subtract_histogram(
                    frontier[best_i].hist, left_hist
                )
            else:
                right_hist = builder.build_leaf(right_node)
                left_hist = subtract_histogram(
                    frontier[best_i].hist, right_hist
                )
        t_hist += Float64(perf_counter_ns() - hist_t0) / 1e9

        # Same clamp-and-divide as the CPU grower: no-ops when unconstrained.
        # The cap and the smoothing come first and the interval is enforced on
        # the result, the order the candidate was scored with, and both
        # children smooth toward the value the parent already emits.
        var parent_bounds = frontier[best_i].bounds.copy()
        var split_sign = monotone_sign(signs, split.feature)
        var parent_output = tree.value[parent_node]
        var left_value = parent_bounds.clamp(
            _leaf_value(
                left_hist,
                params.lambda_reg,
                params.lambda_l1,
                value_feature,
                n_left,
                parent_output,
                max_delta_step,
                path_smooth,
            )
        )
        var right_value = parent_bounds.clamp(
            _leaf_value(
                right_hist,
                params.lambda_reg,
                params.lambda_l1,
                value_feature,
                n_right,
                parent_output,
                max_delta_step,
                path_smooth,
            )
        )
        if split_sign != MONOTONE_FREE and left_value > right_value:
            # A rounding step can invert the two outputs after the candidate
            # check; collapsing both to their midpoint keeps the ordering
            # exact and leaves the midpoint unchanged.
            var mid = midpoint(left_value, right_value)
            left_value = mid
            right_value = mid
        var children = child_bounds(
            parent_bounds, split_sign, left_value, right_value
        )
        tree.value[left_node] = left_value
        tree.value[right_node] = right_value
        tree.left[parent_node] = left_node
        tree.right[parent_node] = right_node
        tree._set_split(parent_node, split, split_missing_bin)

        # Both children inherit the same branch feature set, so they share one
        # allow mask, and both sit one edge below the leaf that was split.
        var branch = extend_branch(frontier[best_i].branch, split.feature)
        var allowed = params.constraints.allowed_features(branch)
        var child_depth = frontier[best_i].depth + 1
        # Each child draws its own per-node feature set from its node id, the
        # same id the CPU grower would assign it.
        search_t0 = perf_counter_ns()
        var left_split = _search(
            left_hist,
            n_left,
            params,
            allowed,
            select_split_features(
                tree_features,
                params.feature_fraction_bylevel,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                child_depth,
                left_node,
            ),
            depth=child_depth,
            missing_bins=builder.missing_bin,
            monotone=signs,
            cats=builder.cats,
            bounds=children.left,
            node=left_node,
            tree_index=tree_index,
            parent_output=left_value,
            grower_applies_extra=True,
        )
        var right_split = _search(
            right_hist,
            n_right,
            params,
            allowed,
            select_split_features(
                tree_features,
                params.feature_fraction_bylevel,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                child_depth,
                right_node,
            ),
            depth=child_depth,
            missing_bins=builder.missing_bin,
            monotone=signs,
            cats=builder.cats,
            bounds=children.right,
            node=right_node,
            tree_index=tree_index,
            parent_output=right_value,
            grower_applies_extra=True,
        )
        t_search += Float64(perf_counter_ns() - search_t0) / 1e9

        frontier[best_i] = _GpuLeafState(
            left_node,
            n_left,
            left_hist^,
            left_split^,
            branch.copy(),
            depth=child_depth,
            bounds=children.left.copy(),
        )
        frontier.append(
            _GpuLeafState(
                right_node,
                n_right,
                right_hist^,
                right_split^,
                branch^,
                depth=child_depth,
                bounds=children.right.copy(),
            )
        )
        n_leaves += 1

    if phase_trace:
        var tree_s = Float64(perf_counter_ns() - tree_t0) / 1e9
        print(
            "phase_trace tree",
            tree_index,
            "total_s",
            tree_s,
            "hist_s",
            t_hist,
            "partition_s",
            t_partition,
            "search_s",
            t_search,
            "other_s",
            tree_s - t_hist - t_partition - t_search,
        )
    if hybrid_trace and hy_mode != MODE_OFF:
        print(
            "hybrid_trace tree",
            tree_index,
            "mode",
            mode_name(hy_mode),
            "live",
            hybrid_live,
            "host_builds",
            hy_host_builds,
            "mirror_matches",
            hy_mirror_matches,
            "mirror_mismatches",
            hy_mirror_mismatches,
            "replica_state",
            builder.replica_state,
        )

    tree.n_leaves = n_leaves
    return tree^


def _check_train_gpu(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64],
    alpha: Float64,
    bagging: BaggingParams,
    goss: GossParams,
) raises:
    """Everything the trainer refuses before a byte reaches the device: the
    same checks, in the same order, that `train` makes. Shared by both
    `train_gpu` entry points so the session overload cannot drift from the
    plain one."""
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    params.tree.monotone.check_features(data.n_features)


def _train_gpu_rounds[
    S: RoundLifecycle
](
    mut builder: GpuHistogramBuilder,
    mut life: S,
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64],
    alpha: Float64,
    bagging: BaggingParams,
    goss: GossParams,
    device_grads: Bool,
    split_search: Int,
    route_all_rows: Bool = False,
) raises -> Booster:
    """The boosting loop both `train_gpu` entry points run, over a builder
    the caller already constructed and a lifecycle it already chose.

    `route_all_rows` is the bagged device round: the bag comes from the same
    sampler and the same schedule as on the host, and the tree's contribution
    reaches every row's device raw score through `GpuTreeRouter`, in bag or
    not, which is what the leaf-range update cannot do. It is set only when
    the caller asked for `OBJECTIVE_SOURCE_DEVICE` explicitly, so the shipped
    AUTO behavior for a bagged run is unchanged.

    `life` is `NoLifecycle` for the session-free entry point, which makes
    every hook below two integer increments and no device work, so this loop
    issues exactly the sequence it issued before the session seam existed.
    A `GpuSession` in its place moves the session's state machine and gives
    `MOJOTREES_GPU_TRACE=1` its per-round and per-tree counts."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        var n = data.n_rows
        var base_score = _base_score(target, objective, sample_weight, alpha)

        var signs = params.tree.monotone.active_signs()
        var renews = objective_renews_leaves(objective)
        var renew_w = renewal_weights(objective, target, sample_weight)
        var renew_a = renewal_alpha(objective, alpha)
        var trees = List[Tree]()

        # Built-in objectives without row sampling generate their gradients
        # on the device and advance the raw scores there too, so a round
        # uploads nothing per row: labels and weights cross once at state
        # construction, and only the tree's node-value table (a few hundred
        # bytes) crosses per tree. Bagging and GOSS stay on the host path,
        # which needs the gradients host-side to rank and sample rows, and so
        # does an explicit `objective_source=OBJECTIVE_SOURCE_HOST`.
        if device_grads:
            var state = builder.objective_state(
                target, sample_weight, 1, 2 * params.tree.num_leaves
            )
            state.init_raw(builder.ctx, [base_score])
            # One router per fit, never per tree: it holds the flattened
            # tree tables and one leaf ordinal per row, all sized by the
            # largest tree this run can grow. Empty on the unbagged path,
            # where the leaf ranges already cover every row and are cheaper,
            # so an explicit device request on an unbagged run still takes
            # the range update and allocates nothing extra.
            var router = List[GpuTreeRouter]()
            if route_all_rows and bagging_enabled(bagging):
                router.append(
                    GpuTreeRouter(
                        builder.ctx, n, 2 * params.tree.num_leaves
                    )
                )
            var dev_bag = List[Int]()
            for i in range(params.n_estimators):
                life.begin_round()
                # Same sampler, same schedule, same seed as the host path
                # and as the CPU trainer, so round i grows on identical
                # rows whichever path produced its gradients.
                refresh_bag(dev_bag, bagging, n, i)
                builder.fill_gradients_device(state, objective, alpha)
                life.begin_tree()
                var tree = grow_tree_gpu(
                    builder, params.tree, dev_bag, i, split_search, data=data
                )
                life.end_tree()
                if renews:
                    # Renewal is a host-side weighted percentile, so the
                    # renewing objectives pay one raw-score download per
                    # tree; the scores come back through Float32, which is
                    # the device path's documented precision.
                    var raw = state.download_raw(builder.ctx)
                    _renew_leaf_values(
                        tree, data, target, raw, renew_w, renew_a, dev_bag,
                        signs, params.tree.extra,
                    )
                if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                    life.end_round()
                    # Under bagging a degenerate tree indicts this sample,
                    # not the run, exactly as on the host path.
                    if bagging_enabled(bagging):
                        continue
                    break
                if len(router) > 0:
                    # Every row, in bag or not. Never both this and the
                    # range update for one tree: each adds a full
                    # `learning_rate * value` step.
                    router[0].update_all_rows(
                        builder.ctx,
                        state,
                        tree,
                        builder.bins_dev,
                        params.learning_rate,
                    )
                else:
                    builder.update_raw_device(
                        state, tree.value, params.learning_rate
                    )
                trees.append(tree^)
                life.end_round()
            return Booster(
                trees^,
                base_score,
                params.learning_rate,
                objective,
                params.tree.monotone.copy(),
            )

        var raw = List[Float64](capacity=n)
        for _ in range(n):
            raw.append(base_score)
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        var bag = List[Int]()
        for i in range(params.n_estimators):
            life.begin_round()
            refresh_bag(bag, bagging, n, i)
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, grad, hess
            )
            # GOSS rescales the sampled rows' gradients before they are
            # uploaded, so the device histograms already carry the
            # compensation multiplier.
            goss_round(bag, grad, hess, goss, i, params.learning_rate)
            builder.upload_gradients(grad, hess)
            life.begin_tree()
            var tree = grow_tree_gpu(
                builder, params.tree, bag, i, split_search, data=data
            )
            life.end_tree()
            if renews:
                _renew_leaf_values(
                    tree, data, target, raw, renew_w, renew_a, bag, signs,
                    params.tree.extra,
                )

            # A single-leaf tree with a near-zero value means the objective
            # has converged; further rounds cannot make progress. Under
            # bagging or GOSS it only means this sample had nothing to give,
            # so the round is skipped and the next sample gets its turn.
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                life.end_round()
                if bagging_enabled(bagging) or goss.enabled:
                    continue
                break

            for r in range(n):
                raw[r] += params.learning_rate * tree.predict_row(data, r)
            trees.append(tree^)
            life.end_round()

        return Booster(
            trees^,
            base_score,
            params.learning_rate,
            objective,
            params.tree.monotone.copy(),
        )


def train_gpu(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    objective_source: Int = OBJECTIVE_SOURCE_AUTO,
    split_search: Int = SPLIT_SEARCH_AUTO,
) raises -> Booster:
    """Train a boosted ensemble with tree growth on the GPU. Same contract
    as `train` (objectives, sample_weight, alpha, bagging, and GOSS
    semantics); requires an accelerator at runtime and at most 256 bins.
    Bags come from the same sampler and the same schedule as on the CPU, so
    both backends grow round i on exactly the same rows. GOSS ranks rows on
    the host from the same Float64 gradients the CPU trainer uses, so its
    sample matches the CPU sample exactly as well; only the histograms the
    sample feeds carry the GPU's Float32 precision.

    `objective_source` and `split_search` are the two device stages'
    switches (see the OBJECTIVE_SOURCE_* and SPLIT_SEARCH_* constants);
    both default to the behavior this trainer already shipped. The overload
    below takes a `GpuSession` and is otherwise identical."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        _check_train_gpu(
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            goss,
        )
        # Only an explicit device request routes every row through
        # `GpuTreeRouter`; AUTO keeps a bagged run on the host path and its
        # Float64 raw scores.
        var routes_all = (
            resolve_objective_source(objective_source)
            == OBJECTIVE_SOURCE_DEVICE
        )
        var device_grads = device_gradients(
            objective, 1, objective_source, bagging, goss, routes_all
        )
        var builder = GpuHistogramBuilder(data)
        var life = NoLifecycle()
        return _train_gpu_rounds(
            builder,
            life,
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            goss,
            device_grads,
            split_search,
            routes_all,
        )


def train_gpu(
    mut session: GpuSession,
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    objective_source: Int = OBJECTIVE_SOURCE_AUTO,
    split_search: Int = SPLIT_SEARCH_AUTO,
) raises -> Booster:
    """`train_gpu` on a caller-owned session: the builder borrows the
    session's context and its ledgers record the construction, and every
    round and tree boundary is announced to the session's state machine.

    The device work is identical to the session-free form. What a session
    adds is an owner: it outlives this call, so a later fit or a validation
    matrix can share the context, and `session.trace()` reports the phases
    under `MOJOTREES_GPU_TRACE=1`. The trainer never closes it."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        _check_train_gpu(
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            goss,
        )
        var routes_all = (
            resolve_objective_source(objective_source)
            == OBJECTIVE_SOURCE_DEVICE
        )
        var device_grads = device_gradients(
            objective, 1, objective_source, bagging, goss, routes_all
        )
        # The session's own fit latency: the first fit through a session
        # pays the one-time costs and every later one does not.
        var fit = session.begin_fit()
        var builder = GpuHistogramBuilder(session, data)
        # `MOJOTREES_HYBRID_LEAVES` against this run's real facts. Reports
        # the workload-aware split resolution rather than treating AUTO as
        # the historical host default.
        session.note_hybrid(
            resolve_split_search_for(builder, params.tree, split_search)
            == SPLIT_SEARCH_DEVICE,
            not device_grads,
            True,
            data.n_rows,
            data.n_features,
            data.n_bins,
            data.n_rows,
        )
        var booster = _train_gpu_rounds(
            builder,
            session,
            data,
            target,
            objective,
            params,
            sample_weight,
            alpha,
            bagging,
            goss,
            device_grads,
            split_search,
            routes_all,
        )
        session.end_fit(fit)
        return booster^


def _train_custom_gpu_rounds[
    S: RoundLifecycle, F: GradHessFn
](
    mut builder: GpuHistogramBuilder,
    mut life: S,
    data: BinnedMatrix,
    target: List[Float64],
    grad_hess: F,
    params: BoosterParams,
    sample_weight: List[Float64],
    base_score: Float64,
    split_search: Int,
) raises -> Booster:
    """The custom-objective loop both `train_custom_gpu` entry points run.
    There is no device-gradient branch here by construction: the callback
    lives on the host, which is where the raw scores it reads live."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        var n = data.n_rows
        var raw = List[Float64](capacity=n)
        for _ in range(n):
            raw.append(base_score)

        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        for i in range(params.n_estimators):
            life.begin_round()
            grad_hess(raw, target, grad, hess)
            check_custom_grad_hess(grad, hess, n)
            _apply_sample_weight(grad, hess, sample_weight)
            builder.upload_gradients(grad, hess)
            life.begin_tree()
            var tree = grow_tree_gpu(
                builder, params.tree, [], i, split_search, data=data
            )
            life.end_tree()

            # A single-leaf tree with a near-zero value means the objective
            # has converged; further rounds cannot make progress.
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                life.end_round()
                break

            for r in range(n):
                raw[r] += params.learning_rate * tree.predict_row(data, r)
            trees.append(tree^)
            life.end_round()

        return Booster(
            trees^,
            base_score,
            params.learning_rate,
            CUSTOM,
            params.tree.monotone.copy(),
        )


def train_custom_gpu[F: GradHessFn](
    data: BinnedMatrix,
    target: List[Float64],
    grad_hess: F,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
    split_search: Int = SPLIT_SEARCH_AUTO,
) raises -> Booster:
    """`train_custom` with tree growth on the GPU: the objective callback
    stays on the host (it is one call per round over host-side raw scores),
    and only the resulting gradients cross to the device, exactly as the
    built-in objectives do in `train_gpu`. Same contract and validation as
    `train_custom` in objective.mojo; requires an accelerator at runtime and
    at most 256 bins."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(target) != data.n_rows:
            raise Error("target length must equal n_rows")
        _check_sample_weight(sample_weight, data.n_rows)

        var builder = GpuHistogramBuilder(data)
        var life = NoLifecycle()
        return _train_custom_gpu_rounds(
            builder,
            life,
            data,
            target,
            grad_hess,
            params,
            sample_weight,
            base_score,
            split_search,
        )


def train_custom_gpu[F: GradHessFn](
    mut session: GpuSession,
    data: BinnedMatrix,
    target: List[Float64],
    grad_hess: F,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
    split_search: Int = SPLIT_SEARCH_AUTO,
) raises -> Booster:
    """`train_custom_gpu` on a caller-owned session; see the `train_gpu`
    session overload. The device work is identical to the session-free
    form."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(target) != data.n_rows:
            raise Error("target length must equal n_rows")
        _check_sample_weight(sample_weight, data.n_rows)

        var fit = session.begin_fit()
        var builder = GpuHistogramBuilder(session, data)
        # A custom objective evaluates on the host, so this run's gradients
        # are host resident by construction.
        session.note_hybrid(
            resolve_split_search_for(builder, params.tree, split_search)
            == SPLIT_SEARCH_DEVICE,
            True,
            True,
            data.n_rows,
            data.n_features,
            data.n_bins,
            data.n_rows,
        )
        var booster = _train_custom_gpu_rounds(
            builder,
            session,
            data,
            target,
            grad_hess,
            params,
            sample_weight,
            base_score,
            split_search,
        )
        session.end_fit(fit)
        return booster^


def _multiclass_base_scores(
    labels: List[Int],
    n_classes: Int,
    sample_weight: List[Float64],
) raises -> List[Float64]:
    """Per-class log priors (weighted when `sample_weight` is given), which
    is where a softmax run starts. Also the point every label is range
    checked, so both entry points refuse a bad label before the binned
    matrix is uploaded."""
    var class_w = List[Float64](capacity=n_classes)
    for _ in range(n_classes):
        class_w.append(0.0)
    var total_w = 0.0
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        var w = sample_weight[r] if len(sample_weight) > 0 else 1.0
        class_w[labels[r]] += w
        total_w += w
    if total_w <= 0.0:
        raise Error("sample_weight must have a positive sum")
    var base_scores = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        base_scores.append(log(_clamp_prob(class_w[k] / total_w)))
    return base_scores^


def _train_multiclass_gpu_batched[
    S: RoundLifecycle
](
    mut builder: GpuHistogramBuilder,
    mut life: S,
    mut state: GpuObjectiveState,
    data: BinnedMatrix,
    n_classes: Int,
    params: BoosterParams,
    var base_scores: List[Float64],
    schedule: ClassSchedule,
    split_search: Int,
) raises -> MulticlassBooster:
    """The device softmax loop with a batch of classes resident at once.

    The same rounds `_train_multiclass_gpu_rounds` runs, with one step
    hoisted. Sequentially, each class fills its gradients, reduces its own
    magnitudes, and *waits* for them, because the fixed-point scale has to be
    on the host before a histogram can be quantized; that wait drains the
    queue once per class. Here a batch's classes fill together and reduce
    together, so a round pays one readback per batch instead of one per
    class, and the classes of a batch keep their gradients resident while
    their trees grow one after another.

    Everything else is the sequential loop, unmoved: the same grower, the
    same seed `round * n_classes + k`, the same score update through
    `update_raw_device`, the same `close_round` reasoning, the same
    no-progress truncation, and trees appended in ascending `k` so tree
    `(i, k)` still lands at `i * n_classes + k` in the serialized ensemble.

    Nor does it change a number. Batches are contiguous ascending runs
    (`ClassBatchPlan`), slot `s` of batch `b` is class `b * batch + s`, and
    the batched gradient kernel and magnitude reduction are the single-class
    ones with the class moved into `grid.y`: same arithmetic per row, same
    blocks, same grid stride, same ascending Float64 host fold. So a class's
    gradients, its scale, and every histogram quantized with it are what the
    sequential loop would have produced. `MulticlassRoundGuard` checks the
    part of that which is an ordering rather than an arithmetic: one
    snapshot per round, one tree per class, every commit before the next
    snapshot, and batches consumed in the plan's order.

    Not reached unless a caller or `MOJOTREES_GPU_CLASS_BATCH` asked for a
    batch above one. No measurement supports the default moving.
    """
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        var trees = List[Tree]()
        var plan = schedule.batches.copy()
        var batch = GpuClassBatch.for_plan(
            builder.ctx,
            builder.n_rows,
            builder.n_features,
            builder.n_bins,
            plan,
        )
        var guard = MulticlassRoundGuard(n_classes)
        for i in range(params.n_estimators):
            life.begin_round()
            guard.open_round()
            state.refresh_softmax(builder.ctx)
            guard.note_probs()
            var made_progress = False
            for b in range(plan.n_batches()):
                var k_begin = plan.batch_begin(b)
                var k_count = plan.batch_count(b)
                # One launch for the batch's gradients, then the round's one
                # readback for its scales. The guard checks that the batch
                # is the next one the plan expects.
                guard.note_batch(plan, b)
                batch.fill_gradients(state, k_begin, k_count)
                batch.refresh_scales(k_count)
                for slot in range(k_count):
                    var k = plan.class_at(b, slot)
                    # This slot's plane and its already-reduced scale into
                    # the builder's own buffers; no wait, because the scale
                    # is known.
                    builder.fill_batched_gradients(batch, slot)
                    life.begin_tree()
                    var tree = grow_tree_gpu(
                        builder,
                        params.tree,
                        [],
                        i * n_classes + k,
                        split_search,
                        data=data,
                    )
                    life.end_tree()
                    guard.note_tree(k)
                    if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                        made_progress = True
                    # Before the next class's begin_tree resets the ranges.
                    builder.update_raw_device(
                        state, tree.value, params.learning_rate, k
                    )
                    guard.note_commit(k)
                    trees.append(tree^)
            guard.close_round()
            life.end_round()
            if not made_progress:
                for _ in range(n_classes):
                    _ = trees.pop()
                break
        return MulticlassBooster(
            trees^,
            base_scores^,
            n_classes,
            params.learning_rate,
            params.tree.monotone.copy(),
        )


def _train_multiclass_gpu_rounds[
    S: RoundLifecycle
](
    mut builder: GpuHistogramBuilder,
    mut life: S,
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64],
    bagging: BaggingParams,
    goss: GossParams,
    var base_scores: List[Float64],
    device_grads: Bool,
    split_search: Int,
) raises -> MulticlassBooster:
    """The softmax loop both `train_multiclass_gpu` entry points run. A
    round opens once and each class's tree inside it opens and closes, which
    is the tree-to-tree transition `SessionLifecycle` allows without an
    intervening round boundary.

    Both paths below drive one `MulticlassRoundGuard`
    (gpu_multiclass_batch.mojo), which is where the softmax round contract
    lives: the probability snapshot is taken once, when the raw scores hold
    every previous round's trees and none of this round's; every class then
    grows exactly one tree from that snapshot and folds it into the scores
    exactly once before the next snapshot. That rule is what the two loops
    have always obeyed by construction, and it is the one a batched or
    reordered class schedule can break silently, so it is checked here rather
    than restated as a comment. The guard owns no device state and allocates
    two flag lists."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        var n = data.n_rows
        var trees = List[Tree]()

        # Softmax without row sampling runs the whole objective on the
        # device: probabilities refresh once per round, each class's
        # gradients land straight in the histogram buffers, and each class's
        # tree advances the device raw scores from its leaf ranges. Bagging
        # and GOSS keep the host path, which owns the row sample.
        if device_grads:
            var labels_f = List[Float64](capacity=n)
            for r in range(n):
                labels_f.append(Float64(labels[r]))
            var state = builder.objective_state(
                labels_f, sample_weight, n_classes, 2 * params.tree.num_leaves
            )
            state.init_raw(builder.ctx, base_scores)

            # How many classes may hold their gradients on the device at
            # once. The default is one -- the loop below, unchanged -- and a
            # wider batch is opt-in through `MOJOTREES_GPU_CLASS_BATCH`,
            # because batching changes the memory a fit holds and nothing in
            # this repository has measured it against the sequential loop.
            #
            # `deeper_node()` rather than `round_root()`: eligibility says
            # what a batch's classes *share*, and this path shares nothing
            # but the launch. Each class's histogram is still built by the
            # builder, one class at a time, out of the buffers it owns; the
            # bin reads are shared only by the batched histogram kernels,
            # which need a grower that consumes a whole frontier level at
            # once and are not driven from here. Claiming shared rows would
            # allocate a batch whose single count plane this path never
            # writes.
            var schedule = builder.class_schedule(
                n_classes, BatchEligibility.deeper_node()
            )
            if not schedule.is_sequential():
                return _train_multiclass_gpu_batched(
                    builder,
                    life,
                    state,
                    data,
                    n_classes,
                    params,
                    base_scores^,
                    schedule,
                    split_search,
                )

            var guard = MulticlassRoundGuard(n_classes)
            for i in range(params.n_estimators):
                life.begin_round()
                guard.open_round()
                state.refresh_softmax(builder.ctx)
                guard.note_probs()
                var made_progress = False
                for k in range(n_classes):
                    builder.fill_softmax_gradients_device(state, k)
                    guard.note_gradients(k, 1)
                    life.begin_tree()
                    var tree = grow_tree_gpu(
                        builder,
                        params.tree,
                        [],
                        i * n_classes + k,
                        split_search,
                        data=data,
                    )
                    life.end_tree()
                    guard.note_tree(k)
                    if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                        made_progress = True
                    # Before the next class's begin_tree resets the ranges.
                    builder.update_raw_device(
                        state, tree.value, params.learning_rate, k
                    )
                    guard.note_commit(k)
                    trees.append(tree^)
                # `close_round`, not `abandon_round`, even when the trees are
                # about to be dropped: they already reached the raw scores,
                # so the round is finished and owes nothing. `abandon_round`
                # is for a schedule that drops trees before committing them,
                # which neither of these loops does.
                guard.close_round()
                life.end_round()
                if not made_progress:
                    for _ in range(n_classes):
                        _ = trees.pop()
                    break
            return MulticlassBooster(
                trees^,
                base_scores^,
                n_classes,
                params.learning_rate,
                params.tree.monotone.copy(),
            )

        # Row-major raw scores and softmax scratch: raw[r * n_classes + k].
        var raw = List[Float64](capacity=n * n_classes)
        for _ in range(n):
            for k in range(n_classes):
                raw.append(base_scores[k])
        var prob = List[Float64](capacity=n * n_classes)
        for _ in range(n * n_classes):
            prob.append(0.0)
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        var bag = List[Int]()
        var guard = MulticlassRoundGuard(n_classes)
        for i in range(params.n_estimators):
            life.begin_round()
            guard.open_round()
            refresh_bag(bag, bagging, n, i)
            for r in range(n):
                for k in range(n_classes):
                    prob[r * n_classes + k] = raw[r * n_classes + k]
                _softmax_inplace(prob, r * n_classes, n_classes)
            # The host path's probability snapshot, taken from raw scores
            # that hold every earlier round's trees and none of this one's,
            # which is the same instant `refresh_softmax` captures on the
            # device path.
            guard.note_probs()

            # One shared sample for the whole round, drawn before any class's
            # tree, exactly as on the CPU.
            var selection = GossSelection.all_rows()
            if goss.active(i, params.learning_rate):
                selection = _multiclass_goss_select(
                    prob, labels, n_classes, sample_weight, goss, i
                )
                bag = selection.rows.copy()

            var made_progress = False
            for k in range(n_classes):
                _fill_softmax_grad_hess(
                    prob, labels, k, n_classes, sample_weight, grad, hess
                )
                apply_goss_scaling(selection, grad, hess)
                builder.upload_gradients(grad, hess)
                guard.note_gradients(k, 1)
                # Feature subsampling draws once per tree, so each class's
                # tree in a round gets its own feature set; the same index
                # the CPU grower uses keeps the two backends on identical
                # feature sets.
                life.begin_tree()
                var tree = grow_tree_gpu(
                    builder,
                    params.tree,
                    bag,
                    i * n_classes + k,
                    split_search,
                    data=data,
                )
                life.end_tree()
                guard.note_tree(k)
                if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                    made_progress = True
                for r in range(n):
                    raw[r * n_classes + k] += (
                        params.learning_rate * tree.predict_row(data, r)
                    )
                guard.note_commit(k)
                trees.append(tree^)
            guard.close_round()
            life.end_round()

            # No class made progress: with bagging or GOSS that is a
            # statement about this sample, so the round is dropped and the
            # next sample gets its turn.
            if not made_progress:
                for _ in range(n_classes):
                    _ = trees.pop()
                if bagging_enabled(bagging) or goss.enabled:
                    continue
                break

        return MulticlassBooster(
            trees^,
            base_scores^,
            n_classes,
            params.learning_rate,
            params.tree.monotone.copy(),
        )


def train_multiclass_gpu(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    objective_source: Int = OBJECTIVE_SOURCE_AUTO,
    split_search: Int = SPLIT_SEARCH_AUTO,
) raises -> MulticlassBooster:
    """`train_multiclass` with tree growth on the GPU.

    Softmax is the last objective that shares the per-row gradient/hessian
    interface, so it needs no new device machinery: one class's tree is one
    ordinary `grow_tree_gpu` call over that class's gradients. One builder
    serves every class of every round, so the binned matrix is uploaded once
    for the whole ensemble and each round costs n_classes gradient uploads.

    Same contract as `train_multiclass` (labels in 0..n_classes-1,
    sample_weight, bagging, and GOSS semantics, including the one shared row
    sample per round); requires an accelerator at runtime and at most 256
    bins. On the host gradient path softmax probabilities are computed on
    the host, exactly as on the CPU, so the only backend difference remains
    the Float32 histogram precision."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(labels) != data.n_rows:
            raise Error("labels length must equal n_rows")
        if n_classes < 2:
            raise Error("n_classes must be at least 2")
        _check_sample_weight(sample_weight, data.n_rows)
        check_bagging(bagging)
        _check_goss(goss, bagging)
        var base_scores = _multiclass_base_scores(
            labels, n_classes, sample_weight
        )
        var device_grads = device_gradients(
            _SOFTMAX_OBJECTIVE, n_classes, objective_source, bagging, goss
        )
        var builder = GpuHistogramBuilder(data)
        var life = NoLifecycle()
        return _train_multiclass_gpu_rounds(
            builder,
            life,
            data,
            labels,
            n_classes,
            params,
            sample_weight,
            bagging,
            goss,
            base_scores^,
            device_grads,
            split_search,
        )


def train_multiclass_gpu(
    mut session: GpuSession,
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    objective_source: Int = OBJECTIVE_SOURCE_AUTO,
    split_search: Int = SPLIT_SEARCH_AUTO,
) raises -> MulticlassBooster:
    """`train_multiclass_gpu` on a caller-owned session; see the `train_gpu`
    session overload. The device work is identical to the session-free
    form."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(labels) != data.n_rows:
            raise Error("labels length must equal n_rows")
        if n_classes < 2:
            raise Error("n_classes must be at least 2")
        _check_sample_weight(sample_weight, data.n_rows)
        check_bagging(bagging)
        _check_goss(goss, bagging)
        var base_scores = _multiclass_base_scores(
            labels, n_classes, sample_weight
        )
        var device_grads = device_gradients(
            _SOFTMAX_OBJECTIVE, n_classes, objective_source, bagging, goss
        )
        var fit = session.begin_fit()
        var builder = GpuHistogramBuilder(session, data)
        session.note_hybrid(
            resolve_split_search_for(builder, params.tree, split_search)
            == SPLIT_SEARCH_DEVICE,
            not device_grads,
            True,
            data.n_rows,
            data.n_features,
            data.n_bins,
            data.n_rows,
        )
        var booster = _train_multiclass_gpu_rounds(
            builder,
            session,
            data,
            labels,
            n_classes,
            params,
            sample_weight,
            bagging,
            goss,
            base_scores^,
            device_grads,
            split_search,
        )
        session.end_fit(fit)
        return booster^


# ---------------------------------------------------------------------------
# Validation scoring
# ---------------------------------------------------------------------------

# Where a validation set's running raw scores are maintained. HOST walks the
# round's tree over every validation row on the host, which is what
# `train_with_valid` in boosting.mojo does and what this trainer does until a
# benchmark says otherwise. DEVICE keeps the validation matrix, its labels,
# and the running raw-score vector resident on the training context and folds
# each round's tree in with one kernel (see gpu_predict.mojo), downloading the
# scores for the loss. AUTO reads `MOJOTREES_GPU_VALID_SCORING` (`host` or
# `device`) and then defaults to HOST.
comptime VALID_SCORE_AUTO = 0
comptime VALID_SCORE_HOST = 1
comptime VALID_SCORE_DEVICE = 2


def env_valid_scoring() -> Int:
    """`MOJOTREES_GPU_VALID_SCORING` as a validation-scoring constant."""
    var s = getenv("MOJOTREES_GPU_VALID_SCORING")
    if s == "device":
        return VALID_SCORE_DEVICE
    if s == "host":
        return VALID_SCORE_HOST
    return VALID_SCORE_AUTO


def resolve_valid_scoring(scoring: Int) -> Int:
    """An explicit choice outranks the environment; AUTO resolves through
    `MOJOTREES_GPU_VALID_SCORING` and then to the host walk."""
    var s = scoring
    if s == VALID_SCORE_AUTO:
        s = env_valid_scoring()
    if s == VALID_SCORE_DEVICE:
        return VALID_SCORE_DEVICE
    return VALID_SCORE_HOST


trait GpuValidScorer:
    """The three things an early-stopping loop asks of a validation set.

    Both implementations hold the same quantity, a running raw score per
    validation row, and both hand it to the same host loss function, so the
    stopping rule is one definition rather than two. They differ only in
    where the running score is kept and how a round's tree is added to it.
    """

    def start(mut self, base_score: Float64) raises:
        """Seed every row's raw score with the ensemble's base score."""
        ...

    def observe(
        mut self, tree: Tree, data: BinnedMatrix, learning_rate: Float64
    ) raises:
        """Add `learning_rate * tree(row)` to every row's raw score."""
        ...

    def loss(
        mut self, target: List[Float64], objective: Int, alpha: Float64
    ) raises -> Float64:
        """The objective's mean loss over the current raw scores."""
        ...


struct _HostValidScorer(GpuValidScorer, Movable):
    """The established path: raw scores in a host list, one `predict_row`
    per validation row per round. Identical arithmetic to
    `train_with_valid`, including its Float64 accumulation, and it touches
    no device code at all, which is what makes it a usable fallback for the
    device scorer below."""

    var raw: List[Float64]
    var n_rows: Int

    def __init__(out self, n_rows: Int):
        self.n_rows = n_rows
        self.raw = List[Float64](capacity=n_rows)
        for _ in range(n_rows):
            self.raw.append(0.0)

    def start(mut self, base_score: Float64) raises:
        for r in range(self.n_rows):
            self.raw[r] = base_score

    def observe(
        mut self, tree: Tree, data: BinnedMatrix, learning_rate: Float64
    ) raises:
        for r in range(self.n_rows):
            self.raw[r] += learning_rate * tree.predict_row(data, r)

    def loss(
        mut self, target: List[Float64], objective: Int, alpha: Float64
    ) raises -> Float64:
        return _mean_loss(self.raw, target, objective, alpha)


def device_loss_metric(objective: Int) -> Int:
    """The METRIC_* code whose device definition is `_mean_loss`'s for
    `objective`, term for term, or -1 when the device has no equal and the
    loss has to be computed on the host from the downloaded raw scores.

    Two objectives qualify today, and the agreement is exact rather than
    approximate. `_mean_loss`'s squared-error branch sums `(raw - y)^2` and
    divides by the row count; `DEVICE_METRIC_L2` under `RESPONSE_IDENTITY` sums
    `w * d * d` and divides by `check_metric_weight([], n)`, which is `n`.
    `_mean_loss`'s L1 branch and `DEVICE_METRIC_L1` line up the same way. The
    remaining difference is the one this whole path already carries: the
    device sums Float32 terms over Float32 labels, so the value agrees to
    Float32 tolerance, not bit for bit.

    Binary logistic is deliberately **not** here even though
    `METRIC_BINARY_LOG_LOSS` exists and looks like a match. The two clamp
    probabilities at different floors, `_clamp_prob` at 1e-15 and `_clamp32`
    at 1e-7, because Float32 cannot hold `1 - 1e-15` apart from 1. On a
    confidently wrong row the host reports `-log(1e-15)` and the device
    saturates at `-log(1e-7)`, and confidently wrong rows are exactly the
    ones a log-loss stopping decision turns on. Scoring it on the host from
    `validation_raw()` keeps the run's loss definition intact. Anyone adding
    a code here should be able to write the host expression and the kernel
    expression side by side and see them agree, including the clamps.
    """
    if objective == SQUARED_ERROR:
        return DEVICE_METRIC_L2
    if objective == L1:
        return DEVICE_METRIC_L1
    return -1


def _open_valid_predictor(
    ctx: DeviceContext,
    caps: DeviceCaps,
    data: BinnedMatrix,
    target: List[Float64],
) raises -> GpuPredictor:
    """A single-output predictor on `ctx`, with `data` and `target` made
    resident. The comptime guard keeps the device instantiation out of
    CPU-only builds, the same shape the guarded device helpers in
    tests/parallel use; only a caller that resolved to VALID_SCORE_DEVICE
    reaches it."""
    comptime if not has_accelerator():
        raise Error("GPU validation scoring requires an accelerator")
    else:
        var predictor = GpuPredictor(ctx, caps, data.n_features, 1)
        predictor.set_validation(data, target)
        return predictor^


struct _DeviceValidScorer(GpuValidScorer, Movable):
    """Validation scores kept on the training context.

    The validation matrix, its labels, and the running raw-score vector are
    uploaded once and stay resident; a round uploads only the tree it grew
    (kilobytes) and adds it in with one kernel, so scoring round i costs one
    tree walk per row rather than i of them. The predictor shares the
    builder's `DeviceContext`, so its kernels queue behind the round's
    training kernels in the same in-order queue and need no fence between
    them.

    The loss is reduced wherever its definition is the run's own.
    `device_loss_metric` answers that per objective: squared error and L1
    reduce on the device, so a round moves `n_valid / REDUCE_BLOCK` floats
    home instead of `n_valid`, and every other objective downloads the raw
    scores and is scored by the same `_mean_loss` the CPU trainer stops on.
    The device's metric set is smaller than the objective loss set, and a
    stopping decision made from a different loss definition is a different
    run, not a faster one, so the fallback is the rule rather than the
    exception.

    The one limit that no dispatch removes: the raw scores accumulate in
    Float32, since Apple GPUs have no Float64. Two rounds within Float32
    noise of each other can therefore order differently than they would on
    the host path and pick a different `best_iteration`. That is why this
    scorer is not the default.
    """

    var predictor: GpuPredictor

    def __init__(
        out self,
        ctx: DeviceContext,
        caps: DeviceCaps,
        data: BinnedMatrix,
        target: List[Float64],
    ) raises:
        self.predictor = _open_valid_predictor(ctx, caps, data, target)

    def start(mut self, base_score: Float64) raises:
        comptime if not has_accelerator():
            raise Error("GPU validation scoring requires an accelerator")
        else:
            var base: List[Float64] = [base_score]
            self.predictor.reset_validation(base)

    def observe(
        mut self, tree: Tree, data: BinnedMatrix, learning_rate: Float64
    ) raises:
        comptime if not has_accelerator():
            raise Error("GPU validation scoring requires an accelerator")
        else:
            # `data` is already device-resident from `set_validation`, so
            # the host matrix is not read here; the argument keeps one trait
            # signature for both scorers.
            var round_trees = List[Tree]()
            round_trees.append(tree.copy())
            # Base scores of zero: `start` put the ensemble's base score
            # into the resident vector once, which is where
            # `IterationRange` puts it too, and `accumulate_round` never
            # adds it again.
            var zero: List[Float64] = [0.0]
            self.predictor.upload_ensemble(
                flatten_trees(round_trees, zero, 1, learning_rate)
            )
            self.predictor.accumulate_round(0)

    def loss(
        mut self, target: List[Float64], objective: Int, alpha: Float64
    ) raises -> Float64:
        comptime if not has_accelerator():
            raise Error("GPU validation scoring requires an accelerator")
        else:
            # Reduce on the device when the device's definition of this
            # objective's loss is `_mean_loss`'s, which turns a per-round
            # n_valid download into an n_blocks one. Otherwise the raw
            # scores come home and the host owns the loss, which is what
            # keeps the eight objectives the device has no kernel for on
            # exactly the definition the CPU trainer stops on.
            var metric = device_loss_metric(objective)
            if metric >= 0:
                return self.predictor.validation_metric(
                    metric, RESPONSE_IDENTITY
                )
            var raw = self.predictor.validation_raw()
            return _mean_loss(raw, target, objective, alpha)


def _train_gpu_valid_rounds[
    V: GpuValidScorer
](
    mut builder: GpuHistogramBuilder,
    mut scorer: V,
    data: BinnedMatrix,
    target: List[Float64],
    valid_data: BinnedMatrix,
    valid_target: List[Float64],
    objective: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    min_delta: Float64,
    sample_weight: List[Float64],
    alpha: Float64,
    bagging: BaggingParams,
    goss: GossParams,
    split_search: Int,
) raises -> Booster:
    """`train_with_valid`'s loop with `grow_tree` replaced by
    `grow_tree_gpu`, over whichever scorer the caller chose.

    Every decision the loop makes is made from the same numbers as on the
    CPU: the same `_fill_grad_hess`, the same bags from the same sampler,
    the same `_mean_loss`, and the same compare-against-best rule with the
    same `min_delta`. Only the histograms the splits are chosen from carry
    the GPU's Float32 precision, plus the validation raw scores when the
    device scorer is selected."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        var n = data.n_rows
        var base_score = _base_score(target, objective, sample_weight, alpha)
        var raw = List[Float64](capacity=n)
        for _ in range(n):
            raw.append(base_score)
        scorer.start(base_score)

        var signs = params.tree.monotone.active_signs()
        var renews = objective_renews_leaves(objective)
        var renew_w = renewal_weights(objective, target, sample_weight)
        var renew_a = renewal_alpha(objective, alpha)
        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        # The base-score-only model is the run's incumbent, exactly as it is on
        # the CPU: a run whose first round does not beat it keeps no trees.
        var best_loss = scorer.loss(valid_target, objective, alpha)
        var best_n_trees = 0
        var bag = List[Int]()
        for i in range(params.n_estimators):
            refresh_bag(bag, bagging, n, i)
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, grad, hess
            )
            goss_round(bag, grad, hess, goss, i, params.learning_rate)
            builder.upload_gradients(grad, hess)
            var tree = grow_tree_gpu(
                builder, params.tree, bag, i, split_search, data=data
            )
            if renews:
                _renew_leaf_values(
                    tree, data, target, raw, renew_w, renew_a, bag, signs,
                    params.tree.extra,
                )

            # Under bagging or GOSS a degenerate tree indicts the sample, not
            # the run.
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                if bagging_enabled(bagging) or goss.enabled:
                    continue
                break

            for r in range(n):
                raw[r] += params.learning_rate * tree.predict_row(data, r)
            # After renewal, which rewrote the leaf values the scorer
            # folds in.
            scorer.observe(tree, valid_data, params.learning_rate)
            trees.append(tree^)

            var loss = scorer.loss(valid_target, objective, alpha)
            if loss < best_loss - min_delta:
                best_loss = loss
                best_n_trees = len(trees)
            elif len(trees) - best_n_trees >= early_stopping_rounds:
                break

        while len(trees) > best_n_trees:
            _ = trees.pop()
        return Booster(
            trees^,
            base_score,
            params.learning_rate,
            objective,
            params.tree.monotone.copy(),
        )


def train_gpu_with_valid(
    data: BinnedMatrix,
    target: List[Float64],
    valid_data: BinnedMatrix,
    valid_target: List[Float64],
    objective: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    valid_scoring: Int = VALID_SCORE_AUTO,
    split_search: Int = SPLIT_SEARCH_AUTO,
) raises -> Booster:
    """`train_with_valid` with tree growth on the GPU: validation-set early
    stopping, the same stopping rule, and the ensemble truncated to its best
    round. Same contract as `train_with_valid` (objectives, sample_weight,
    alpha, bagging, and GOSS semantics, validation rows never sampled);
    requires an accelerator at runtime and at most 256 bins. `valid_data`
    must be binned by the same `BinMapper` as `data`, which is what makes a
    tree's threshold bins mean the same thing on both matrices.

    `valid_scoring` picks where the running validation scores are kept (see
    the VALID_SCORE_* constants): the default resolves to the host tree
    walk, byte for byte what `train_with_valid` does, and VALID_SCORE_DEVICE
    keeps them on the training context instead.

    Gradients are host-side here whichever scorer is chosen, because early
    stopping needs the host raw scores that `_fill_grad_hess` reads. The
    device objective path (`train_gpu`'s `objective_source`) keeps its raw
    scores on the device, so composing the two is a further stage, not a
    parameter this function takes."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(target) != data.n_rows:
            raise Error("target length must equal n_rows")
        if len(valid_target) != valid_data.n_rows:
            raise Error("valid_target length must equal valid n_rows")
        if valid_data.n_features != data.n_features:
            raise Error("valid_data must have the same features")
        _check_objective(objective, target, alpha)
        if early_stopping_rounds < 1:
            raise Error("early_stopping_rounds must be positive")
        _check_sample_weight(sample_weight, data.n_rows)
        check_bagging(bagging)
        _check_goss(goss, bagging)
        params.tree.monotone.check_features(data.n_features)

        var builder = GpuHistogramBuilder(data)
        if resolve_valid_scoring(valid_scoring) == VALID_SCORE_DEVICE:
            # On the builder's own context, so the validation kernels queue
            # behind the round's training kernels rather than racing them
            # from a second context.
            var on_device = _DeviceValidScorer(
                builder.ctx, builder.caps, valid_data, valid_target
            )
            return _train_gpu_valid_rounds(
                builder,
                on_device,
                data,
                target,
                valid_data,
                valid_target,
                objective,
                params,
                early_stopping_rounds,
                min_delta,
                sample_weight,
                alpha,
                bagging,
                goss,
                split_search,
            )
        var on_host = _HostValidScorer(valid_data.n_rows)
        return _train_gpu_valid_rounds(
            builder,
            on_host,
            data,
            target,
            valid_data,
            valid_target,
            objective,
            params,
            early_stopping_rounds,
            min_delta,
            sample_weight,
            alpha,
            bagging,
            goss,
            split_search,
        )
