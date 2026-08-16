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
       launch-cost crossover the CPU also builds the whole fit
       (device_policy.mojo)
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

Round trips per fit, which is the count that predicts time
----------------------------------------------------------

A **round trip** is host code that blocks on a device answer it needs before
it can decide what to enqueue next. A **copy** is an `enqueue_copy` or a
`synchronize`, which on Metal drains the queue (`docs/GPU_PORTABILITY.md`
section 6.1, **measured** by disassembly) but which costs nothing when the
queue holds nothing. Section 6.1.1 records why the two counts may not be
added: removing thirteen copies per tree, about 1,300 per fit, **measured**
0.016 seconds against a registered prediction of 0.64, while removing about
thirty round trips per tree **measured** 0.75 in the same session on the same
machine. Count round trips to predict time; count copies to predict
portability risk and ordering hazards.

Counted in source on 2026-08-16, over a whole fit rather than over one tree,
because the per-tree counts this file kept were what hid the per-round ones.
`R` is `n_estimators` and `K` is `n_classes`. The default shape this
repository benchmarks is 1,000,000 x 50 with `R = 100`, squared error, no
bagging and no GOSS, which resolves to device gradients and the
device-resident split plane.

  phase                                    round trips        default arm
  ---------------------------------------  -----------------  -----------
  builder, searcher and table construction  0                  0
  per round, magnitude window fold          ceil(R/N) [+1]     100
  per round, `upload_gradients`             0                  0
  per tree, `download_desc_tables`          R                  100
  per tree, `update_raw_device`             0                  0
  per tree, `begin_tree`                    0                  0
  per round, device validation scorer       R                  0, off
  ---------------------------------------  -----------------  -----------
  total on the default arm                  R + ceil(R/N) [+1] 200

`N` is `GpuHistogramBuilder.set_scale_refresh`'s cadence, the `scale_refresh`
argument to `train_gpu`, and it is **1 by default**, which is why the default
column still reads 200. The `+1` is the end-of-fit flush, which is a no-op
and costs nothing at `N <= 1`. At `N = 8` the same fit is 100 + 13 + 1 = 114
and at `N = 64`, the ceiling, it is 100 + 2 + 1 = 103. **Counted in source**,
and countable on a run: `scale_readback_count` is the left column's first row
and the round loop charges the profile's `syncs` from it rather than from a
constant, so a profile taken at any cadence reports that cadence's count.

Multiply by `K` for a multiclass fit, since a class is a tree and each pays
both; the window does not reach the softmax path, and
`fill_softmax_gradients_device` says why. **Estimated** at the ~458
microsecond per-round-trip constant **derived** from the depthwise A/B, 200
round trips is 0.09 seconds against a fit **measured** at about 2.58 seconds,
or 3.5%. Each half on its own is about 0.046 seconds, which M0 does not
resolve here: arm spreads run from 0.02 in a quiet window to several tenths
in a slow one, and the machine drifts two- to threefold between windows.
Nothing was cut on the strength of this table and nothing should be until
something can measure it. That applies to the cadence too: it is an arm and
not a default, and it stays an arm until a measurement says otherwise.

The two remaining trips, what each would cost to remove, and which of them
is actually reachable:

  **The scale readback**, once per round on the device-gradient arm.
  `GpuHistogramBuilder.fill_gradients_device` reduces `|grad|` and `|hess|` on
  the device and the host folds the partials in Float64 into
  `g_scale`/`h_scale`, which every histogram launch afterwards takes as a
  launch argument. Nothing can be enqueued until that answer is home, so it
  is a round trip and not a drain.

  Eliminating it outright -- the scale living in a device buffer the kernels
  read, so the host never needs its value -- is **not possible**, and this is
  a finding rather than a scoping decision. The host needs the number in
  three places: as a `Float32` launch argument to nine kernels across
  `gpu_active_rows`, `gpu_gradient_stream`, `gpu_categorical` and
  `gpu_sparse`; as a staged `Float32` table in `gpu_leaf_batching` and
  `gpu_multiclass_batch`; and as `Float32(1.0 / g_scale)` written into the
  split searcher's parameter block by `gpu_split_search._stage_params`. The
  first two are plumbing. The third is not: the obvious dodge -- pre-scale
  the gradients on the device, which the power-of-two rule makes *bit-exact*
  since `g * 2^k` is an exact Float32 product, and hand the kernels a scale of
  1.0 -- fails because the gain and the leaf value are not homogeneous in the
  scale. `G^2 / (H + lambda)` and `-G / (H + lambda)` both carry an unscaled
  `lambda`, so a searcher told the scale is 1.0 computes a different gain and
  chooses a different split. The host has to know the number.

  So what this file does instead is change **how often** the host waits for
  it, not whether it needs it: `set_scale_refresh` defers the readback across
  `N` rounds, keeps every round's magnitudes measured exactly in its own slot
  of a pinned window, and checks each closed window against the `2^30`
  overflow bound so a reused scale can end a fit loudly but cannot corrupt
  one quietly. The whole argument, including the two distinct ways it moves
  histogram bits, is at `set_scale_refresh`. `_train_multiclass_gpu_batched`
  remains the only mitigation on the softmax path: a batch of classes reduces
  together, so a round pays one readback per batch instead of one per class.
  It is opt-in and nothing has measured it.

  **The tree download**, once per tree, and it is **still R**. The obvious
  amortization -- download every `N` trees, since the host only needs a tree
  to append it to the ensemble, to test it for degeneracy, and to hand its
  leaf values to `update_raw_device` -- is blocked by where the code lives
  rather than by what it does. Two hard dependencies, both outside this file:

  - `grow_tree_device_resident` (`gpu_resident_round.mojo`) *always* ends in
    `download_desc_tables` and returns a host `Tree`. A deferred mode is a
    change to that function, not to this loop.
  - `update_raw_device` needs `builder.rows.ranges`, the **host** leaf-range
    mirror, and on the resident plane that mirror is only correct because
    `_publish_row_ranges` replays the downloaded commit order onto it. Its own
    docstring records the bug from assuming otherwise: without the replay the
    update added the root's value to every row and every tree after the first
    diverged. A device-sourced raw update reading `front_dev` and `node_f_dev`
    directly is writable here, but it removes no round trip on its own,
    because the tree still has to come home for the ensemble.

  The rule the deferral would need, recorded so the lane that owns those
  files does not have to re-derive it: **`N` must not exceed the early
  stopping patience**, and where both are set it must be
  `min(requested_N, early_stopping_rounds)`, falling back to 1 whenever
  patience is 1. Early stopping compares `len(trees) - best_n_trees` against
  `early_stopping_rounds` once per round (`_train_gpu_valid_rounds`), so a
  batch of `N` undownloaded trees is `N` rounds in which that comparison
  cannot be made and the stop is up to `N - 1` rounds late; the ensemble is
  then truncated to `best_n_trees` anyway, so the *model* is unchanged and
  only the work is wasted -- but only if `N <= patience`, because at `N >
  patience` the loop can pass the stopping point and keep going for a whole
  further batch, which is a silently defeated stop and worse than no change.
  As it happens the validation loop is host-gradient only today, so it does
  not reach the resident plane at all, which is exactly why the rule has to
  be written down before the two paths meet rather than after.

What is **not** a round trip, checked rather than assumed, because all three
have been reported as waits at some point:

  `upload_gradients` on the host-gradient arm is one `synchronize` protecting
  the staging arena and one copy. That synchronize runs after the previous
  tree's download has already drained the queue, and no host decision reads a
  device answer through it. `GpuObjectiveState.update_raw_ranges` is one copy
  of the range descriptors and one launch per tree. `GpuActiveRows.begin_tree`
  is one launch unbagged, and one synchronize plus one copy bagged.

The contrast is the two paths the resident plane replaced, and it is what
makes 2R a small number rather than a suspicious one.
`_device_search_resident` downloads a frontier per split and decides the next
split from what it read, which is `num_leaves` round trips per tree and 3,100
per fit at the default budget; `grow_tree_gpu_profiled`'s host scan downloads
a histogram per histogram it builds. Those are the counts a 0.75 second
result came off, and there is no third such reduction left in this file.
Under depth-wise growth `_device_search_resident` searches a whole planned
level at once, which its docstring counts as 5 waits per tree rather than 30;
that is the figure the resident plane's 1 already beats, and depth-wise is
one of the configurations the plane refuses, so the 5 is what a depth-wise
fit actually pays.

Taking this census on a run rather than reading it here
-------------------------------------------------------

`MOJOTREES_PHASE_PROFILE` prints a `syncs` column per phase, and as of
2026-08-16 every round trip in the table above increments it: the scale
readback through `PROF_GRAD_FILL` and the resident plane's download through
`PROF_TRANSFER`, the latter counted with no clock and no fence so that a
count nobody timed is still true. Before that both read zero, the resident
plane emitted no rows at all, and a profile could not tell a fit making one
wait per tree from one making thirty-one. A wait count is the right first
instrument here precisely because it does not need a quiet machine: it is a
static property of the schedule, so it says whether a change is worth timing
before anything is timed.

One refinement since, and it is what keeps that true now that the schedule is
not fixed. `PROF_GRAD_FILL` used to be charged `syncs=1` unconditionally,
which was right while every round folded and became a fiction the moment
`set_scale_refresh` let a fit fold every `N` rounds instead. The charge is now
taken from `GpuHistogramBuilder.scale_readback_count`, read before and after
the fill, so the column counts the waits that happened rather than the waits
the default cadence would have made. That is the same failure this docstring
names below for `arm_conditions` -- an instrument reporting a condition it
cannot see -- caught on the way in rather than after it had cost a number.
The cadence is a `train_gpu` argument and not an environment variable, so
unlike every switch named below it is at least visible in the call.

Two blind spots remain around it, both outside this file and both worth
naming here because this is where the census lives.
`bench/bench_train_gpu.mojo`'s `arm_conditions` line records the trainer, the
split strategy, the growth policy and the four launch shapes, and records
none of the switches that decide the wait count: `MOJOTREES_GPU_TREE_RESIDENT`,
`MOJOTREES_GPU_SPLIT_RESIDENT` and `MOJOTREES_GPU_OBJECTIVE` among them. Two
arms whose round-trip counts differ thirty-fold print the same line, which is
the same "conditions inherited from an environment variable" failure that
docstring says has already cost this file a number once. It also records
neither the gain form nor the fixed-point scale shape, which are the arms in
this wave that change arithmetic rather than shape. A phase profile now
distinguishes those arms by their `syncs` column even though the conditions
line does not, which makes the profile the instrument to quote until the line
is widened.
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
    _check_leaf_estimation_config,
    _clamp_prob,
    _estimate_leaf_values,
    _fill_grad_hess,
    _mean_loss,
    _refuse_leaf_estimation,
    _renew_leaf_values,
    _check_goss,
    _fill_softmax_grad_hess,
    _multiclass_goss_select,
    _softmax_inplace,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
    round_has_constant_hessian,
)
from .efb import check_bundling_honored
from .goss import GossParams, GossSelection, apply_goss_scaling, goss_round
from .gpu_frontier import subtraction_builds_left
from .gpu_fused_round import (
    ROUND_OK,
    GpuTreeRouter,
    round_eligibility,
    round_eligibility_reason,
)
from .gpu_multiclass_batch import GpuClassBatch, MulticlassRoundGuard
from .gpu_objectives_native import GpuLeafEstimator, GpuObjectiveState
from .gpu_output_planes import BatchEligibility
from .gpu_predict import (
    DEVICE_METRIC_L1,
    DEVICE_METRIC_L2,
    RESPONSE_IDENTITY,
    GpuPredictor,
    flatten_trees,
)
from .gpu_resident_round import (
    RESIDENT_NO_POOL,
    RESIDENT_OK,
    RESIDENT_TABLES,
    OBLIVIOUS_OK,
    OBLIVIOUS_RECORDS,
    grow_tree_device_oblivious,
    grow_tree_device_resident,
    oblivious_device_supported,
    oblivious_leaf_budget,
    oblivious_reason_name,
    oblivious_records_needed,
    resident_round_enabled,
    resident_round_reason_name,
    resident_round_record_slots,
    resident_round_refusal_detail,
    resident_round_report_refusal,
    resident_round_supported,
)
from .gpu_runtime import GpuSession, NoLifecycle, RoundLifecycle
from .gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
    SplitNodeRequest,
)
from .gpu_split_policy import (
    SPLIT_POLICY_DEVICE_RESIDENT,
    SPLIT_POLICY_HOST,
    SPLIT_REASON_ENVIRONMENT_REQUEST,
    SPLIT_REASON_EXPLICIT_REQUEST,
    SplitSearchDecision,
    decide_split_search,
    normalized_split_work,
)
from .gpu_tiling import DeviceCaps
from .histogram import (
    Histogram,
    check_device_derivative_precision,
    const_hessian_verify,
    subtract_histogram,
)
from .histogram_gpu import GpuHistogramBuilder
from .linear_tree import check_linear_tree_unconnected
from .phase_profile import (
    PARTITION_LAUNCHES,
    PROF_CONVERT,
    PROF_GRAD_FILL,
    PROF_HISTOGRAM,
    PROF_HOST_SYNC,
    PROF_PARTITION,
    PROF_SCORE_UPDATE,
    PROF_SPLIT_SEARCH,
    PROF_SUBTRACT,
    PROF_TRANSFER,
    SCOPE_FIT,
    SCOPE_TREE,
    SPLIT_SEARCH_DEVICE_LAUNCHES,
    PhaseProfile,
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
    GROW_OBLIVIOUS,
    GrowthSchedule,
    LeafCandidate,
    check_grow_policy,
)
from .gpu_leaf_batching import OBLIVIOUS_MAX_ITEMS
from .tree import Tree, TreeParams, _leaf_value, _search, node_bounds


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
            total += hist.count_at(base + b)
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


def split_search_decision_for(
    builder: GpuHistogramBuilder, params: TreeParams
) raises -> SplitSearchDecision:
    """The workload-aware AUTO decision, kept whole rather than reduced to a
    path constant.

    This is the call `resolve_split_search_for` has always made; what is new
    is that the reason, the normalized work, the threshold, and the evidence
    id survive it. They are the only record of *why* a fit took the path it
    took, and until now they were computed and dropped one line later, which
    left a user with no way to tell a device run from a host run except by
    timing it.
    """
    return decide_split_search(
        builder.device_api,
        builder.device_arch,
        builder.n_rows,
        _estimated_active_features(params, builder.n_features),
        builder.n_bins,
        params.num_leaves,
        _device_search_semantics_supported(params),
        builder.resident_frontier_fits(params.num_leaves),
        params.grow_policy,
    )


def split_search_decision_for(
    builder: GpuHistogramBuilder, params: TreeParams, strategy: Int
) raises -> SplitSearchDecision:
    """The resolved decision including a path the caller named outright.

    A named path skips the policy entirely, which is the behavior it has
    always had. It still comes back as a `SplitSearchDecision` so that one
    accessor answers "what is this fit doing and why" in every case; the
    reason is then `explicit-request` or `environment-request` rather than
    anything about a crossover, and the normalized work is reported for
    context without having been compared to anything.
    """
    var requested = strategy
    var reason = SPLIT_REASON_EXPLICIT_REQUEST
    if requested == SPLIT_SEARCH_AUTO:
        requested = env_split_search()
        reason = SPLIT_REASON_ENVIRONMENT_REQUEST
    if requested == SPLIT_SEARCH_DEVICE or requested == SPLIT_SEARCH_HOST:
        return SplitSearchDecision(
            SPLIT_POLICY_DEVICE_RESIDENT
            if requested == SPLIT_SEARCH_DEVICE
            else SPLIT_POLICY_HOST,
            reason,
            normalized_split_work(
                builder.n_rows,
                _estimated_active_features(params, builder.n_features),
                builder.n_bins,
                params.num_leaves,
            ),
        )
    return split_search_decision_for(builder, params)


def describe_split_search(
    builder: GpuHistogramBuilder,
    params: TreeParams,
    strategy: Int = SPLIT_SEARCH_AUTO,
) raises -> String:
    """One line naming the split-search path this shape resolves to, and
    why, without growing a tree.

    The accessor a benchmark prints in its header and a bug report quotes.
    It answers from the same call the trainer makes, so what it prints is
    what a fit at that shape will do, and it includes the boundary marker
    when the workload sits exactly on the measured crossover. Reading it
    costs a policy evaluation and no device work.
    """
    var decision = split_search_decision_for(builder, params, strategy)
    return String(
        decision.describe(),
        " rows=",
        builder.n_rows,
        " active_features=",
        _estimated_active_features(params, builder.n_features),
        " bins=",
        builder.n_bins,
        " leaves=",
        params.num_leaves,
        " margin=",
        decision.margin(),
    )


def split_trace_enabled() -> Bool:
    """Whether to print the resolved split-search decision per tree.

    `MOJOTREES_GPU_SPLIT_TRACE=1` asks for it directly;
    `MOJOTREES_GPU_PHASE_TRACE=1` also turns it on, because a phase trace
    that does not say which split path produced the phases it is attributing
    is a trace of an unnamed run.
    """
    return (
        getenv("MOJOTREES_GPU_SPLIT_TRACE") == "1"
        or getenv("MOJOTREES_GPU_PHASE_TRACE") == "1"
    )


def resolve_split_search_for(
    builder: GpuHistogramBuilder, params: TreeParams
) raises -> Int:
    """Resolve explicit/environment requests, then workload-aware AUTO.

    Explicit `host` and `device` retain their old meanings.  With neither
    present, the pure policy sees the reported device signature, actual
    matrix shape, tree budget, semantic eligibility, and the builder's own
    resident-memory calculation.  Unknown or marginal cases stay on host.
    """
    var decision = split_search_decision_for(builder, params)
    return (
        SPLIT_SEARCH_DEVICE if decision.uses_device() else SPLIT_SEARCH_HOST
    )


def resolve_split_search_for(
    builder: GpuHistogramBuilder, params: TreeParams, strategy: Int
) raises -> Int:
    """Explicit request wrapper around workload-aware AUTO."""
    var decision = split_search_decision_for(builder, params, strategy)
    return (
        SPLIT_SEARCH_DEVICE if decision.uses_device() else SPLIT_SEARCH_HOST
    )


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
    if params.grow_policy == GROW_OBLIVIOUS:
        # `grow_policy = oblivious` searches a whole *level* in one launch
        # pair, so it needs one record per leaf of the widest level plus the
        # one level record the cross-feature reduction folds into --
        # `oblivious_records_needed`, which is `(1 << max_depth) + 1` and is 65
        # at CatBoost's default depth.
        #
        # **Sized from the depth and not from `num_leaves`, which does not bind
        # under this mode.** At the default budget of 31 leaves the leaf-wise
        # arithmetic below would ask for 33 records, the plane would refuse with
        # `OBLIVIOUS_RECORDS`, and the fit would fall back for a reason that was
        # created here rather than found. That is the exact shape of mistake
        # the leaf-wise branch's own comment warns about: a capacity decision
        # made from a different question than the routing decision.
        var want = oblivious_records_needed(params)
        if want > MAX_LEVEL_RECORDS:
            want = MAX_LEVEL_RECORDS
        var ob_width = n_features if n_features > 0 else 1
        var ob_by_cells = MAX_LEVEL_TABLE_CELLS // ob_width
        if want > ob_by_cells:
            want = ob_by_cells
        if want < 2:
            want = 2
        return want
    if params.grow_policy != GROW_DEPTHWISE:
        # The device-owned growth plane (gpu_resident_round.mojo) reduces
        # over the whole frontier on the device, so every live leaf needs a
        # record of its own, plus two scratch records the child searches
        # write into. That plane is the default, so this is the ordinary
        # size; `MOJOTREES_GPU_TREE_RESIDENT=0` takes the two-slot branch.
        # A searcher this size is harmless to the shipping loops either way,
        # since they use the first two slots and leave the rest staged and
        # unread.
        #
        # This asks the same question `_grow_tree_gpu_device_search` asks
        # before it routes, and it has to: `resident_round_supported`
        # refuses with `RESIDENT_RECORDS` when the searcher holds fewer
        # records than one per live leaf plus scratch, so a capacity
        # decision made from a different predicate than the routing decision
        # would refuse the plane on the grounds that it had not been given
        # room for it.
        if resident_round_enabled():
            return resident_round_record_slots(params.num_leaves, n_features)
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


def verify_rows_requested() -> Bool:
    """`MOJOTREES_GPU_VERIFY_ROWS=1`, the per-split row-count cross-check.

    A second read of a variable `gpu_active_rows.GpuActiveRows.__init__`
    already reads, and it is deliberate rather than an oversight: that read
    lands on a field of a device object this module cannot reach from the
    routing decision below, and the routing decision is the only place the
    request can be honored or refused. The two agree by construction because
    both are the same one-line `== "1"` test on the same name, and neither
    interprets it.
    """
    return getenv("MOJOTREES_GPU_VERIFY_ROWS") == "1"


def _check_verify_rows_reachable() raises:
    """Refuse the row-count cross-check on the plane that cannot perform it.

    `MOJOTREES_GPU_VERIFY_ROWS=1` asks for one thing: after each split, the
    device's left-count is downloaded and compared against the histogram's,
    and a disagreement raises. `GpuActiveRows.partition` does exactly that
    (gpu_active_rows.mojo, `verify_counts`), and it is the arm the
    `apply_split` / `finish_split` loops take.

    The device-owned growth plane has no such comparison and cannot grow one
    cheaply: `enqueue_partition_desc` still *writes* the count, and its own
    docstring says the value "stays written so that
    `MOJOTREES_GPU_VERIFY_ROWS` remains meaningful for anyone who wants to
    check" -- but nothing on that plane ever reads it back, because reading it
    per step is precisely the host wait the plane exists to remove. So the
    flag was accepted and did nothing, on what is now the default path.

    That is the worst shape a silent ignore can take. A knob that quietly
    fails to change a fit costs the user a wrong number; a *verification*
    knob that quietly fails to verify costs them a wrong number they have
    been told is checked. It is refused here rather than warned about for the
    same reason.

    `MOJOTREES_GPU_TREE_RESIDENT=0` is the answer and is named in the
    message: it takes the incremental loop, where the cross-check is real.
    """
    if not verify_rows_requested():
        return
    raise Error(
        "MOJOTREES_GPU_VERIFY_ROWS=1 asks for the per-split row-count"
        " cross-check, and the device-owned growth plane never downloads the"
        " count to compare, so the check would silently not run. Set"
        " MOJOTREES_GPU_TREE_RESIDENT=0 to take the incremental loop, which"
        " performs it, or unset MOJOTREES_GPU_VERIFY_ROWS"
    )


def _check_gpu_forced_splits(params: TreeParams, grower: String) raises:
    """Refuse a forced-split document on a grower that does not apply one.

    `tree.grow_tree` is the only grower in this package that reads
    `params.extra.forced`: it seeds the growth loop from the document
    (tree.mojo, `forced=` on the root candidate) and follows it down. No GPU
    grower does, dense or sparse, and none ever has.

    WHY THE GUARD EVERYONE READS DID NOT CATCH THIS, because it is the whole
    lesson of the 2026-08-16 refusal sweep and it will be read as redundant
    otherwise. `ExtraTreeParams.is_active()` *does* name `forced`, so the
    parameter looks covered. It is covered on exactly one path:
    `_check_device_search_supported` refuses the entire bundle, and that path
    is `MOJOTREES_GPU_SPLIT_STRATEGY=device`, which is not the default. The
    shipping host split scan routes through `tree._search`, and `tree._search`
    refuses `ExtraTreeParams.needs_grower_support()`, which is
    `max_delta_step`, `path_smooth`, `extra_trees`, and `random_strength` and
    is strictly smaller than `is_active()`. `forced` fell in the gap between
    the two.

    It is worse than a gap, because AUTO steers into it. `is_active()` being
    True is exactly what makes `split_search_decision_for` decline the device
    arm (`_device_search_semantics_supported`), so a fit that set forced
    splits was routed *onto* the host arm, which is the arm that drops them,
    by the same predicate that would have refused it on the other one.

    Called before the two arms split rather than inside either, so both raise
    the same sentence and a caller cannot be told a different story by a
    strategy switch. `binning.map_forced_splits` has already turned thresholds
    into bins by the time a document gets this far, which is why
    `check_scalars`' unmapped-document refusal does not fire: a correctly
    mapped document is precisely the case that used to be dropped.
    """
    if params.extra.forced.is_empty():
        return
    raise Error(
        "forced splits are applied by tree.grow_tree, the dense CPU grower,"
        " and by no other grower in this package; ",
        grower,
        " never reads the document and would return an unforced tree. Train"
        " on the CPU (device='cpu', or device='auto', which routes around"
        " this), or leave forced_splits unset",
    )


def _check_gpu_const_hessian_verify(trainer: String) raises:
    """Refuse the constant-hessian audit on a backend that does not run it.

    `MOJOTREES_CONST_HESSIAN_VERIFY=1` makes a builder that was told the
    hessians are constant walk the hessian array and raise if any entry is
    not exactly `CONSTANT_HESSIAN` (`histogram._check_constant_hessian`).
    Every CPU builder honors it. No GPU builder does: the device path reads
    its sibling `MOJOTREES_CONST_HESSIAN` through
    `GpuActiveRows.const_hessian_allowed` and takes or declines the shortcut
    on it, and then verifies nothing, because the audit is a host walk over a
    host array and the plane the hessians are on is the device's.

    So the two halves of one diagnostic pair came apart: the half that
    *enables* an optimization is live on both backends and the half that
    *checks* it is live on one. A GPU fit under this flag took the shortcut
    and reported that it had been audited. Refused rather than warned for the
    reason `_check_verify_rows_reachable` gives at length: an unperformed
    check is worse than an unset one, because the user has stopped looking.
    """
    if not const_hessian_verify():
        return
    raise Error(
        "MOJOTREES_CONST_HESSIAN_VERIFY=1 audits a declared constant hessian"
        " by walking the host hessian array, which only the CPU histogram"
        " builders do; ",
        trainer,
        " accumulates on the device and would take the shortcut unaudited."
        " Train on the CPU (device='cpu', or device='auto', which routes"
        " around this), or unset MOJOTREES_CONST_HESSIAN_VERIFY",
    )


def _check_gpu_booster_params(
    params: BoosterParams, trainer: String
) raises:
    """Refuse what this trainer cannot honor: two parameters and one knob.

    The parameters first.

    Both were accepted and silently dropped until 2026-08-16, and both are
    ensemble-level rather than per-tree, which is why they are refused at the
    trainer rather than at the grower: a grower is handed a `TreeParams` and
    never sees either one.

    `enable_bundle` is built by the dense CPU trainers in boosting.mojo, which
    fit a bundling plan once per training call and grow every tree on the
    bundled matrix. A GPU trainer accumulates its histograms from the
    unbundled binned matrix through histogram_gpu.mojo and has nowhere to
    apply a plan. `efb.check_bundling_honored` is the refusal boosting.mojo's
    own docstring asks every other trainer to make;
    `train_gpu_sparse._refuse_bundling` has made it since it shipped, and the
    dense trainers here did not, which is how the gap was found.

    `linear_tree` fits an affine model per leaf from the *raw* feature matrix.
    Every GPU trainer here takes a `BinnedMatrix`, exactly as `boosting.train`
    does, and `boosting.train` refuses it for that reason. `model.fit` also
    refuses it before it dispatches, which is why this was invisible from the
    Python surface; `model.fit_multiclass`, `external_memory.train_external*`,
    and any direct call to a trainer in this file went straight past it.

    And then the knob, which is here rather than in a separate call at every
    entry point because a caller of this function is asking one question --
    "is there anything I asked for that you will not do" -- and the answer
    must not depend on whether the thing they asked for arrived as a field or
    as an environment variable.
    """
    check_bundling_honored(params.bundling, trainer)
    if params.linear.is_active():
        check_linear_tree_unconnected(trainer)
    _check_gpu_const_hessian_verify(trainer)


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


struct GpuSplitSearcherCache(Movable):
    """One `GpuSplitSearcher` held across a fit's trees instead of rebuilt
    per tree.

    What a searcher's shape depends on is the dataset and the tree budget:
    `n_features`, `n_bins`, the missing-bin table, the categorical spec, and
    the record capacity `_search_record_slots` derives from the growth
    policy and the leaf budget. None of those move between the trees of one
    fit, yet the device-search grower constructed a fresh searcher for every
    tree. That construction is not free. Counted from the constructor: twelve
    device buffers and six pinned host buffers, so eighteen allocations;
    three staged table copies; three `map_to_host` mappings, each of which
    blocks until the device is idle; and one explicit `synchronize`. The
    `set_monotone` call the grower makes immediately afterwards is a fourth
    mapping and a fourth hidden block. At the shipped 100-round default that
    was 100 rebuilds of an object whose contents were about to be
    overwritten node by node anyway.

    The eighteen allocations are the part of that which is unambiguously
    work. The copies and mappings are drains, and under
    `docs/GPU_PORTABILITY.md` section 6.1.1 a drain is an ordering point
    rather than a price: none of them is a round trip, and a queue with
    nothing in it drains for nothing. Read them here as part of the hazard
    surface a per-tree rebuild kept re-creating, not as a wait budget the
    cache recovers.

    One of those eighteen is dead weight on this path in particular.
    `hist_dev` is `3 * n_features * n_bins` Int32, about 150 KB at 50
    features and 255 bins, and it is read by exactly two methods,
    `upload_histogram` and `search`, which exist so the searcher is
    exercisable on its own. Neither device-search loop calls either: both go
    through `enqueue`/`enqueue_frontier` against the histogram builder's own
    buffer. Hoisting takes that allocation from once per tree to once per
    fit; removing it outright means making it lazy inside
    `GpuSplitSearcher`, which is a different file and a different lane.

    Reuse is exact, not approximate. Every table a search reads is restaged
    before every launch: `enqueue` and `enqueue_frontier` write each
    record's feature slots, allow mask, float parameters, and histogram base
    themselves, and `download` clears the base again. The only searcher
    state that outlives a launch is the monotone vector, which is a property
    of the fit rather than of the tree, and `reset_for_tree` restages it
    whenever it differs from what this cache last uploaded. To make the
    equivalence obvious rather than merely true, the reset also rebroadcasts
    the tree's feature set to every record slot, which is the same
    "every listed feature, all allowed" state a fresh construction leaves
    behind and costs host stores only.

    A shape mismatch rebuilds rather than raising, so a caller that reuses
    one cache across two builders or two leaf budgets gets a correct
    searcher and pays what it would have paid anyway.
    """

    var searchers: List[GpuSplitSearcher]
    var n_features: Int
    var n_bins: Int
    var max_records: Int
    var signs: List[Int]
    var signs_staged: Bool

    def __init__(out self):
        """An empty cache. The first tree builds the searcher."""
        self.searchers = List[GpuSplitSearcher]()
        self.n_features = 0
        self.n_bins = 0
        self.max_records = 0
        self.signs = List[Int]()
        self.signs_staged = False

    def _same_signs(self, signs: List[Int]) -> Bool:
        if not self.signs_staged or len(self.signs) != len(signs):
            return False
        for f in range(len(signs)):
            if self.signs[f] != signs[f]:
                return False
        return True

    def reset_for_tree(
        mut self,
        mut builder: GpuHistogramBuilder,
        params: TreeParams,
        tree_features: List[Int],
        signs: List[Int],
    ) raises:
        """Make the held searcher ready for this tree, building it only if
        this cache holds nothing usable for the shape.

        The monotone upload is skipped when the vector is the one already on
        the device. That is the one place reuse removes work a fresh
        searcher would still have done, and it is sound because `mono_dev`
        is written by nothing else: `set_monotone` is its only writer, and
        the vector comes from `params.monotone`, which does not change
        within a fit.
        """
        var want_records = _search_record_slots(params, builder.n_features)
        if (
            len(self.searchers) == 0
            or self.n_features != builder.n_features
            or self.n_bins != builder.n_bins
            or self.max_records != want_records
        ):
            self.searchers.clear()
            self.searchers.append(
                GpuSplitSearcher(
                    builder.ctx,
                    builder.n_features,
                    builder.n_bins,
                    builder.missing_bin,
                    builder.cats,
                    max_records=want_records,
                )
            )
            self.n_features = builder.n_features
            self.n_bins = builder.n_bins
            self.max_records = want_records
            self.signs_staged = False
        if not self._same_signs(signs):
            self.searchers[0].set_monotone(signs)
            self.signs = signs.copy()
            self.signs_staged = True
        self.searchers[0].set_features(tree_features)


def _oblivious_route_reason(
    params: TreeParams,
    builder: GpuHistogramBuilder,
    opened: Bool,
    searcher_records: Int,
) raises -> Int:
    """Why an oblivious fit cannot take the level schedule, or `OBLIVIOUS_OK`.

    Three questions in the order a reader wants them answered: did the pool and
    the tables open at all, does the configuration fit the plane, and is the
    searcher wide enough for the widest level. The first comes back as a
    `RESIDENT_*` code because it is the pool's answer and not this mode's, which
    is why the caller names the two code spaces apart when it reports.

    A free function rather than three lines at the call site so that the
    "resident pool declined" case has one spelling; it is also the only place
    that knows both code spaces.
    """
    if not opened:
        return RESIDENT_NO_POOL
    var why = oblivious_device_supported(params, builder)
    if why != OBLIVIOUS_OK:
        return why
    if searcher_records < oblivious_records_needed(params):
        return OBLIVIOUS_RECORDS
    return OBLIVIOUS_OK


def _grow_tree_gpu_device_search(
    mut profile: PhaseProfile,
    mut builder: GpuHistogramBuilder,
    mut cache: GpuSplitSearcherCache,
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
    # Hoisted out of the per-tree loop: see `GpuSplitSearcherCache` for why
    # a searcher may be reused and what the reset restores.
    cache.reset_for_tree(builder, params, tree_features, signs)
    var split_params = GpuSplitParams(
        params.lambda_reg,
        params.lambda_l1,
        params.min_child_hess,
        params.min_data_in_leaf,
        params.cat.copy(),
    )

    builder.begin_tree(bag)
    var n_root = len(bag) if len(bag) > 0 else builder.n_rows
    if params.grow_policy == GROW_OBLIVIOUS:
        # CatBoost's `SymmetricTree`, on the device: one split per level
        # applied to every leaf of that level. It is reached here, through the
        # ordinary per-tree dispatch and the ordinary `grow_policy` parameter,
        # rather than behind a switch of its own -- a parallel switch is how the
        # CPU half of this mode came to be built, tested and uncallable.
        #
        # **Every table is sized from `1 << max_depth` and not from
        # `num_leaves`.** A level splits entirely or not at all, so a leaf
        # budget cannot be met exactly; `oblivious_leaf_budget` is the one place
        # that arithmetic lives and `tree._check_oblivious` says the same thing
        # on the host.
        var budget = oblivious_leaf_budget(params)
        var opened = (
            not resident_frontier_disabled()
            and budget >= 2
            and builder.open_resident(budget, OBLIVIOUS_MAX_ITEMS)
            and builder.open_resident_tables(budget)
        )
        var why = _oblivious_route_reason(
            params, builder, opened, cache.searchers[0].max_records
        )
        if why != OBLIVIOUS_OK:
            # A named raise rather than a fallback, because there is nothing on
            # this backend to fall back *to*: no other GPU grower implements a
            # symmetric tree. The CPU grower does, so the message names it.
            raise Error(
                String(
                    "grow_policy=oblivious cannot be grown on this device: ",
                    oblivious_reason_name(why)
                    if opened
                    else resident_round_reason_name(why),
                    ". The CPU backend grows the same tree; pass"
                    " device='cpu', or use max_depth in [1, 6] with no"
                    " categorical feature, no monotone or interaction"
                    " constraint, and no extra tree parameters",
                )
            )
        # The symmetric plane is the same descriptor plane as the leaf-wise
        # one and downloads the split count no more often, so the row-count
        # cross-check is equally unavailable here. See
        # `_check_verify_rows_reachable`.
        _check_verify_rows_reachable()
        profile.begin_tree(n_root, builder.n_rows)
        var oblivious_started = profile.clock()
        var symmetric = grow_tree_device_oblivious(
            builder,
            cache.searchers[0],
            split_params,
            params,
            tree_features,
            n_root,
        )
        # Counted, not timed, exactly as the leaf-wise plane's bracket is: one
        # round trip per tree, `download_desc_tables` at the end, and no clock
        # inside a loop whose whole claim is that it does not wait.
        profile.charge(PROF_TRANSFER, n_root, 0, syncs=1)
        profile.end_tree(oblivious_started)
        return symmetric^
    if (
        not resident_frontier_disabled()
        and params.min_data_in_leaf >= 1
        and builder.open_resident(params.num_leaves)
    ):
        # The device-owned growth plane, which is a second control plane
        # beside this one rather than a change to it: it grows the same tree
        # with the frontier, the tree and the slot pool all resident, and it
        # waits once per tree where the loop below waits once per split.
        #
        # **It is the default.** `MOJOTREES_GPU_TREE_RESIDENT=0` forces the
        # loop below instead, which is the A/B handle and stays because on
        # this hardware only interleaved arms compare. It refuses by name
        # every configuration it cannot express, and every one of those
        # falls through to the loop below rather than being approximated.
        # See gpu_resident_round.mojo.
        #
        # **It is not timed, and it is now counted.** `profile` is not
        # threaded into it and should not be: it has no per-node host phases
        # to time, because the whole point of it is that the host does not
        # see the nodes, and separating device phases needs fences that would
        # measure the instrument. So a `PhaseProfile` of a device-search fit
        # attributes none of the tree's time to a phase, which is the honest
        # failure rather than a wrong one.
        #
        # What changed on 2026-08-16 is that it no longer comes back *empty*.
        # The bracket below opens and closes the tree and charges the plane's
        # one round trip, with no clock and no fence; see there for why one.
        # An empty table could not distinguish a plane making one wait per
        # tree from the loop below making thirty-one, which made the profile
        # useless for the one question a control-plane lane asks of it.
        # `MOJOTREES_GPU_TREE_RESIDENT=0` gets the fully instrumented loop
        # back; the plane's own trace (`MOJOTREES_GPU_TREE_RESIDENT_TRACE`)
        # is what replaces the per-node detail, and a Metal timeline from
        # outside the process is what sees the rest.
        if resident_round_enabled():
            var why = resident_round_supported(
                params, builder, cache.searchers[0].max_records
            )
            if why == RESIDENT_OK and not builder.open_resident_tables(
                params.num_leaves
            ):
                why = RESIDENT_NO_POOL
            if why == RESIDENT_OK:
                # The plane is elected here and nowhere else, so this is the
                # one place a request the plane cannot honor can be refused
                # rather than dropped. See `_check_verify_rows_reachable`.
                _check_verify_rows_reachable()
                # Counted, not timed, and the distinction is the whole of
                # what this bracket claims. The plane takes no `profile` and
                # will not: its phases are device phases and separating them
                # needs fences, which would measure the instrument rather
                # than the plane. But its **wait count** needs no clock and
                # no fence, and a wait count is a static reproducible fact
                # where a time on this machine is not. So the tree is opened
                # and closed here for the wall total and the tree count, and
                # exactly one `PROF_TRANSFER` charge is made with a zero
                # start, which `PhaseProfile.charge` documents as charging
                # the counts and no time.
                #
                # One, because `grow_tree_device_resident` makes exactly one
                # round trip: `download_desc_tables` at the end, after every
                # step is enqueued. The step trace can add one per step and
                # is off unless a variable turns it on, which is why this
                # counts the contract rather than reading a counter back out
                # of the plane. What the report shows for a default fit is
                # therefore `trees` trees, one transfer call and one sync
                # apiece, and the whole of the tree's time in the
                # unattributed remainder, which is true. Before this it
                # showed nothing at all, and a census taken off the profile
                # could not tell a fit that made one wait per tree from a
                # fit that made thirty-one.
                profile.begin_tree(n_root, builder.n_rows)
                var resident_started = profile.clock()
                var grown = grow_tree_device_resident(
                    builder,
                    cache.searchers[0],
                    split_params,
                    params,
                    tree_features,
                    n_root,
                )
                profile.charge(PROF_TRANSFER, n_root, 0, syncs=1)
                profile.end_tree(resident_started)
                return grown^
            if tree_index == 0:
                # Once per fit rather than once per tree, and named down to
                # the layer that refused.
                #
                # Where it goes changed with the default. While the plane
                # was opt-in this printed unconditionally, which was right:
                # the only way to reach it was to have asked. Now that the
                # plane is the default, an unconditional print would put a
                # line on standard output for every GPU fit with monotone
                # constraints, depth-wise growth or a categorical column,
                # all of which are refusing correctly and none of which
                # asked. `resident_round_report_refusal` keeps the print for
                # the caller who still sets `=1` by hand and otherwise
                # routes the reason to the trace sink, so a default-path
                # refusal is inspectable without being noisy.
                var detail = resident_round_reason_name(why)
                if why == RESIDENT_TABLES:
                    detail = resident_round_refusal_detail(params)
                resident_round_report_refusal(detail)
        return _device_search_resident(
            profile,
            builder,
            cache.searchers[0],
            split_params,
            params,
            tree_features,
            signs,
            tree_index,
            n_root,
        )
    # Not instrumented: the incremental loop is the fallback the resident one
    # replaced and is reachable only when residency declines or
    # `MOJOTREES_GPU_SPLIT_RESIDENT=0` forces it. A profile of a run that took
    # this path reports an empty table rather than a wrong one, which is the
    # honest failure; wiring it is a small job and was left undone rather than
    # half done.
    return _device_search_incremental(
        builder,
        cache.searchers[0],
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
    mut profile: PhaseProfile,
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
    # The per-phase, per-node-size instrument (phase_profile.mojo), which is
    # the same three phases broken down by node class and carrying launch and
    # synchronization counts beside the time. Its `fenced` mode inserts the
    # same drains `phase_trace` does, which is why the two share one flag
    # here: either instrument asking for a fence gets one, and neither
    # changes the tree.
    var fence = phase_trace or profile.fenced()
    var n_slots = len(builder.active)
    profile.begin_tree(n_root, builder.n_rows)
    var profile_t0 = profile.clock()
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
    var root_hist_started = profile.clock()
    builder.enqueue_resident_leaf(root, root_slot)
    if fence:
        builder.ctx.synchronize()
    # The launch count comes off the resolved plan rather than a constant,
    # because the tiled and atomic strategies cost different numbers of them
    # and the policy is the only thing that knows which ran. Behind
    # `enabled()` because deriving a plan is real work this run has no other
    # reason to do.
    var root_launches = 0
    if profile.enabled():
        root_launches = builder.histogram_plan(n_root).gpu_launches()
    profile.note_node()
    profile.charge(
        PROF_HISTOGRAM,
        n_root,
        root_hist_started,
        dispatches=root_launches,
        syncs=1 if fence else 0,
        slots_per_row=n_slots,
        cells=builder.n_features * builder.n_bins,
    )
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
    var root_search_started = profile.clock()
    searcher.enqueue_frontier(
        builder.batcher[0].out_dev,
        root_batch,
        split_params,
        builder.g_scale,
        builder.h_scale,
    )
    profile.charge(
        PROF_SPLIT_SEARCH,
        n_root,
        root_search_started,
        dispatches=SPLIT_SEARCH_DEVICE_LAUNCHES,
        cells=builder.n_features * builder.n_bins,
    )
    # The batch's one *download*. `download_frontier` copies the record
    # buffer and synchronizes inside the same call, so this charge is the
    # copy and the wait together and the sync count is what separates them.
    # Under `async` this is also where every kernel enqueued since the last
    # wait actually finishes, which is why a device arm's `transfer` line is
    # large and its `histogram` line is small: see the mode note in
    # phase_profile.mojo.
    #
    # It is the batch's one *round trip*, and that is the count that predicts
    # time (docs/GPU_PORTABILITY.md section 6.1.1): host code reads a device
    # answer here and then decides what to enqueue next. It is not the batch's
    # only drain. The `enqueue_frontier` above copies the staged tables across
    # first, and on Metal each copy is itself a full-queue drain (**measured**
    # by disassembly, section 6.1), so the batch has those upload drains plus
    # this download. Only the download is charged with `syncs=1`, because only
    # the download calls `synchronize` by name.
    #
    # Two counts, and this profile carries one of them. `syncs` is a count of
    # explicit synchronizations and is closest to the round-trip count, which
    # is the one that predicts seconds. The upload drains belong on the copy
    # count, which predicts portability risk and ordering hazards and does not
    # convert to time: removing thirteen such copies per tree **measured**
    # 0.016 seconds at 1,000,000 x 50 against a registered prediction of 0.64
    # (bench/results/session3_2026-08-16/RESULTS.md), a null under M0. An
    # earlier version of this comment called every one of them a wait the
    # batch paid for; section 6.1.1 withdrew that on 2026-08-16.
    var root_dl_started = profile.clock()
    var root_recs = searcher.download_frontier(1)
    profile.charge(PROF_TRANSFER, n_root, root_dl_started, syncs=1)
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
    # A batch ends in a download, so a batch is a round trip: the count that
    # predicts time. A level wider than the searcher's record capacity
    # therefore becomes several of them rather than one oversized launch.
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

            var batch_rows = 0
            for pick in range(taken, upto):
                batch_rows += frontier[picks[pick]].n_rows
                _enqueue_resident_split(
                    profile,
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
                    fence,
                )

            search_t0 = perf_counter_ns()
            # A batch is charged at the rows its parents held, so a level's
            # search lands in the class those parents were in. Two launches
            # per batch however wide it is, which under leaf-wise growth is
            # two per split and under depth-wise growth is two per level; that
            # difference is exactly what a depth-wise comparison would read
            # off the `dispatches` column.
            var batch_search_started = profile.clock()
            searcher.enqueue_frontier(
                builder.batcher[0].out_dev,
                batch,
                split_params,
                builder.g_scale,
                builder.h_scale,
            )
            profile.charge(
                PROF_SPLIT_SEARCH,
                batch_rows,
                batch_search_started,
                dispatches=SPLIT_SEARCH_DEVICE_LAUNCHES,
                cells=len(batch) * builder.n_features * builder.n_bins,
            )
            # The batch's one wait, and what upholds the staging contracts on
            # both sides: the batcher's pinned item table and the searcher's
            # pinned node tables are only restaged after this returns.
            var batch_dl_started = profile.clock()
            var recs = searcher.download_frontier(len(batch))
            profile.charge(
                PROF_TRANSFER, batch_rows, batch_dl_started, syncs=1
            )
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
    profile.end_tree(profile_t0)
    return tree^


def _enqueue_resident_split(
    mut profile: PhaseProfile,
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
    fence: Bool,
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
    # Charged at the parent's rows, because the partition walks the parent's
    # window and writes it back; filing it under a child would split one cost
    # across two classes.
    var part_rows = n_left + n_right
    var part_started = profile.clock()
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
    if fence:
        builder.ctx.synchronize()
    profile.charge(
        PROF_PARTITION,
        part_rows,
        part_started,
        dispatches=PARTITION_LAUNCHES,
        syncs=1 if fence else 0,
    )
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
    var built_rows = n_left if build_left else n_right
    var derived_rows = n_right if build_left else n_left
    var hist_started = profile.clock()
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
    if fence:
        builder.ctx.synchronize()
    var hist_launches = 0
    if profile.enabled():
        hist_launches = builder.histogram_plan(built_rows).gpu_launches()
    profile.note_node()
    profile.charge(
        PROF_HISTOGRAM,
        built_rows,
        hist_started,
        dispatches=hist_launches,
        syncs=1 if fence else 0,
        slots_per_row=len(builder.active),
        cells=builder.n_features * builder.n_bins,
    )
    # The sibling subtraction rides inside that same kernel here rather than
    # costing a launch of its own, which is why it is charged with no time,
    # no launch, and only a `calls` entry at the derived child's size. That
    # zero is the finding: on the host path this phase is a whole per-cell
    # pass and on this path it is free, and the two profiles put those side
    # by side under one name.
    profile.note_node()
    profile.charge(PROF_SUBTRACT, derived_rows, 0, cells=0)
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


def _build_leaf_profiled(
    mut profile: PhaseProfile,
    mut builder: GpuHistogramBuilder,
    leaf: Int,
    node_rows: Int,
    fence: Bool,
) raises -> Histogram:
    """`builder.build_leaf(leaf)`, with its three parts charged to three
    phases instead of one.

    `build_leaf` is `enqueue_leaf` then `download_raw` then
    `histogram_from_host`, and those are a per-row-slot accumulate, a fixed
    `3 * n_features * n_bins` copy with a host wait inside it, and a per-cell
    dequantization. Charging them together would say a small node's histogram
    is expensive without saying that almost none of the expense is the rows,
    which is the question this instrument exists for.

    An off profile takes the `build_leaf` call itself, unchanged, so the
    default path is not merely equivalent to what shipped -- it is the same
    call. The profiled arm is that method's body written out, and if
    `build_leaf` ever gains a fourth step this function will silently stop
    matching it. That is the price of not editing histogram_gpu.mojo from
    this lane, and it is noted here so the next reader can pay it or fix it.
    """
    if not profile.enabled():
        return builder.build_leaf(leaf)
    var cells = builder.n_features * builder.n_bins
    var launches = builder.histogram_plan(node_rows).gpu_launches()
    var acc_started = profile.clock()
    builder.enqueue_leaf(leaf)
    if fence:
        builder.ctx.synchronize()
    profile.note_node()
    profile.charge(
        PROF_HISTOGRAM,
        node_rows,
        acc_started,
        dispatches=launches,
        syncs=1 if fence else 0,
        slots_per_row=len(builder.active),
        cells=cells,
    )
    # The copy and the wait it carries. `download_raw` does both in one call,
    # so this is one charge with a sync count of one rather than two charges;
    # separating them needs a change inside histogram_gpu.mojo.
    var dl_started = profile.clock()
    builder.download_raw()
    profile.charge(
        PROF_TRANSFER, node_rows, dl_started, syncs=1, cells=cells
    )
    var cv_started = profile.clock()
    var hist = builder.histogram_from_host()
    profile.charge(PROF_CONVERT, node_rows, cv_started, cells=cells)
    return hist^


def grow_tree_gpu(
    mut builder: GpuHistogramBuilder,
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
    split_search: Int = SPLIT_SEARCH_AUTO,
) raises -> Tree:
    """`grow_tree_gpu` for a caller that grows one tree and keeps nothing.

    The signature that shipped, unchanged. It owns both of the things a
    boosting loop would rather own itself: a searcher cache, which lives for
    exactly this call so the device-search path builds and tears down a
    searcher as it always did, and a phase profile, which reports one
    `scope=tree` block. `_train_gpu_rounds` holds both across a whole fit and
    calls `grow_tree_gpu_profiled` directly, which is why a `train_gpu` run
    reports one table and a direct caller reports one per tree.

    Neither is a defaulted parameter, because a `mut` argument cannot be
    defaulted. Hence the chain of three entry points rather than one with
    optional arguments. The two paths differ in nothing a tree can observe.
    """
    var cache = GpuSplitSearcherCache()
    return grow_tree_gpu(
        builder, cache, params, bag, tree_index, split_search
    )


def grow_tree_gpu(
    mut builder: GpuHistogramBuilder,
    mut cache: GpuSplitSearcherCache,
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
    split_search: Int = SPLIT_SEARCH_AUTO,
) raises -> Tree:
    """`grow_tree_gpu` with a caller-owned searcher cache, profiled per tree.

    The overload a loop takes when it wants the searcher to outlive the tree
    but does not accumulate a profile across the fit.
    """
    var profile = PhaseProfile.from_env(SCOPE_TREE, String("grow_tree_gpu"))
    var tree = grow_tree_gpu_profiled(
        profile, builder, cache, params, bag, tree_index, split_search
    )
    profile.print_report()
    return tree^


def grow_tree_gpu_profiled(
    mut profile: PhaseProfile,
    mut builder: GpuHistogramBuilder,
    mut cache: GpuSplitSearcherCache,
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
    split_search: Int = SPLIT_SEARCH_AUTO,
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

    There is no `data` parameter, and its absence is the record of something
    this grower stopped needing. All three entry points took the caller's
    host-resident `BinnedMatrix` for the hybrid CPU/GPU leaf scheduler, which
    was deleted 2026-08-16 once the device-resident tree plane beat the host
    path at every measured shape (bench/results/session3_2026-08-16). After
    that deletion no arm read it, and it was kept for one commit only so the
    seven call sites threading it were not rewritten in the same change; it
    came out on 2026-08-16 with the round-trip census. A GPU grower that
    needs the host matrix is a grower whose data plane is not on the device,
    so a future edit that wants it back should say which arm needs it and
    why, rather than restoring the argument. Every arm here reaches the bins
    through `builder`, where they are already resident."""
    # Resolved once, kept whole, and printed under trace. The reason, the
    # normalized work, the threshold, and whether the workload sits exactly
    # on the crossover are the only record of why this fit is taking the
    # path it is taking, and they used to be discarded on the line that
    # computed them.
    # Before the arms split, so a strategy switch cannot change the answer.
    # See `_check_gpu_forced_splits` for why AUTO steers a forced-split fit
    # onto the one arm that drops them.
    _check_gpu_forced_splits(params, String("the GPU grower"))
    var decision = split_search_decision_for(builder, params, split_search)
    if split_trace_enabled():
        print("split_trace tree", tree_index, decision.describe())
    if params.grow_policy == GROW_OBLIVIOUS:
        # `grow_policy = oblivious` has exactly one GPU grower -- the level
        # schedule in `gpu_resident_round.mojo` -- and it lives behind the
        # device-search entry point below. There is no host-scan arm to weigh
        # it against on this backend: the grower under `decision.uses_device()
        # == False` builds its frontier with `GrowthSchedule`, and a symmetric
        # tree is not a frontier order at all (`growth_policy.mojo`), so that
        # arm raises rather than growing a different tree.
        #
        # So the AUTO crossover has nothing to decide here and is not consulted
        # for this policy. It is still *resolved* above, and printed under
        # `split_trace`, because the reason a fit took a path is worth having in
        # the record even when the path was not in question.
        return _grow_tree_gpu_device_search(
            profile, builder, cache, params, bag, tree_index
        )
    if decision.uses_device():
        return _grow_tree_gpu_device_search(
            profile, builder, cache, params, bag, tree_index
        )
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
    # `derivative_precision = float64` is refused rather than ignored here:
    # the device carries derivatives as Float32 and has no Float64 to carry
    # them in, so this grower cannot honor the setting by any amount of
    # threading. See `histogram.check_device_derivative_precision`.
    check_device_derivative_precision(
        params.extra.wants_float64_derivatives()
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
    # See `_device_search_resident`: one flag for both instruments' fences, so
    # a fenced profile and the older trace never disagree about how many
    # drains a split performed.
    var fence = phase_trace or profile.fenced()
    var t_partition = 0.0
    var t_hist = 0.0
    var t_search = 0.0
    var tree_t0 = perf_counter_ns()

    builder.begin_tree(bag)
    var n_root = len(bag) if len(bag) > 0 else builder.n_rows
    profile.begin_tree(n_root, builder.n_rows)
    var profile_t0 = profile.clock()
    var hist_cells = builder.n_features * builder.n_bins

    var root = tree._add_node(0.0, Float64(n_root))
    var hist_t0 = perf_counter_ns()
    var root_hist = _build_leaf_profiled(
        profile, builder, root, n_root, fence
    )
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
    var root_search_started = profile.clock()
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
    # The host scan, on this path, over the node's own feature draw. Same
    # phase name the device scan uses, which is the point of a shared
    # vocabulary: a host-search arm and a device-search arm put their split
    # selection on the same line.
    profile.charge(
        PROF_SPLIT_SEARCH, n_root, root_search_started, cells=hist_cells
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
        var part_started = profile.clock()
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
        if fence:
            builder.ctx.synchronize()
        # At the parent's rows, which is the window the partition walks.
        profile.charge(
            PROF_PARTITION,
            frontier[best_i].n_rows,
            part_started,
            dispatches=PARTITION_LAUNCHES,
            syncs=1 if fence else 0,
        )
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
        # No placeholder value: every arm below assigns both. The
        # `Histogram.zeroed(1, 1)` stubs these used to hold existed only so
        # the deleted hybrid branch could leave one unwritten.
        var left_hist: Histogram
        var right_hist: Histogram
        var child_nodes: List[Int] = [left_node, right_node]
        hist_t0 = perf_counter_ns()
        if builder.batches_nodes(child_nodes):
            # One launch group for both children, so both are charged to
            # `histogram` at their own sizes and neither pays a
            # subtraction. The launch count belongs to the pair, so it is
            # charged once, against the larger child.
            var batch_started = profile.clock()
            var pair = builder.build_leaves(child_nodes)
            left_hist = pair[0].copy()
            right_hist = pair[1].copy()
            var batch_launches = 0
            if profile.enabled():
                batch_launches = builder.histogram_plan(
                    n_left if n_left > n_right else n_right
                ).gpu_launches()
            profile.note_node()
            profile.note_node()
            profile.charge(
                PROF_HISTOGRAM,
                n_left + n_right,
                batch_started,
                dispatches=batch_launches,
                syncs=1,
                slots_per_row=len(builder.active),
                cells=2 * hist_cells,
            )
        elif subtraction_builds_left(n_left, n_right):
            # Which child is built and which is derived is
            # `gpu_frontier.subtraction_builds_left`, whose docstring
            # names this test and `grow_tree`'s as the two it matches.
            # Written once so a batched grower and this one cannot pick
            # different children and then disagree about which histogram
            # a slot holds.
            left_hist = _build_leaf_profiled(
                profile, builder, left_node, n_left, fence
            )
            var sub_started = profile.clock()
            right_hist = subtract_histogram(
                frontier[best_i].hist, left_hist
            )
            profile.note_node()
            profile.charge(
                PROF_SUBTRACT, n_right, sub_started, cells=hist_cells
            )
        else:
            right_hist = _build_leaf_profiled(
                profile, builder, right_node, n_right, fence
            )
            var sub_started = profile.clock()
            left_hist = subtract_histogram(
                frontier[best_i].hist, right_hist
            )
            profile.note_node()
            profile.charge(
                PROF_SUBTRACT, n_left, sub_started, cells=hist_cells
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
        var left_search_started = profile.clock()
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
        profile.charge(
            PROF_SPLIT_SEARCH, n_left, left_search_started, cells=hist_cells
        )
        var right_search_started = profile.clock()
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
        profile.charge(
            PROF_SPLIT_SEARCH, n_right, right_search_started, cells=hist_cells
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

    tree.n_leaves = n_leaves
    profile.end_tree(profile_t0)
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
    _check_gpu_booster_params(params, String("train_gpu"))


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
        # CatBoost's extra Newton steps. Read and checked once per fit, not
        # once per tree, exactly as `boosting._boost_rounds` does it: nothing
        # the check reads moves inside the loop, and at the default of 1 both
        # the count and the check are one integer comparison. The check
        # refuses the renewing objectives and GOSS, which are the two
        # configurations where a second step would be minimizing a different
        # quantity than the first; both arms below reach it.
        var leaf_iters = params.tree.extra.leaf_estimation_iterations
        _check_leaf_estimation_config(params.tree.extra, objective, goss)
        var trees = List[Tree]()
        # One profile for the whole fit rather than one per tree, so a
        # hundred rounds produce one table to diff instead of a hundred
        # (phase_profile.mojo). Off unless `MOJOTREES_PHASE_PROFILE` says
        # otherwise, and an off profile reads no clock and writes no counter,
        # so the ensemble is the same ensemble either way.
        var profile = PhaseProfile.from_env(SCOPE_FIT, String("train_gpu"))
        # One searcher for the fit, not one per tree. Its shape comes from
        # the builder and the tree budget, neither of which moves between the
        # rounds below; `GpuSplitSearcherCache` states what the per-tree reset
        # restores and why reuse cannot change a tree. Stays empty, and
        # allocates nothing, on the host-search path.
        var searcher_cache = GpuSplitSearcherCache()

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
            # One estimator per fit, and only when a fit actually asked for
            # extra Newton steps: it owns three `n_rows` Float32 planes, which
            # is 12 MB at a million rows and is not worth allocating for the
            # default. Empty is the shipped path, and an empty list issues
            # nothing.
            var estimator = List[GpuLeafEstimator]()
            if params.tree.extra.leaf_estimation_active():
                estimator.append(
                    GpuLeafEstimator(
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
                var dev_grad_started = profile.clock()
                var scale_reads_before = builder.scale_readback_count()
                builder.fill_gradients_device(state, objective, alpha)
                # Charged from the builder's own readback counter rather than
                # from the constant 1 that used to sit here.
                # `GpuHistogramBuilder.set_scale_refresh` lets a fit fold the
                # magnitude window every `N` rounds instead of every round, so
                # the number of rounds that wait is no longer the number of
                # rounds. A fixed `syncs=1` would report the shipped cadence's
                # count whatever cadence ran, which is exactly the "conditions
                # inherited from a switch the instrument cannot see" failure
                # this file's docstring says has already cost it a number once.
                var scale_synced = (
                    builder.scale_readback_count() != scale_reads_before
                )
                # `syncs=1`, and the comment this replaced said the opposite.
                # It read "enqueued, not waited for", borrowed from the device
                # histogram line, and it was wrong about this call:
                # `fill_gradients_device` enqueues the derivative kernels and
                # then calls `magnitude_sums`, which reduces `|grad|` and
                # `|hess|` on the device, copies the partials home and
                # synchronizes, because the host folds them in Float64 into
                # `g_scale`/`h_scale` and every histogram launch after this
                # takes those as launch arguments. Nothing can be enqueued
                # until the answer is home, so this is a **round trip** in the
                # sense of `docs/GPU_PORTABILITY.md` section 6.1.1 and not a
                # drain, and it is one of the two the module docstring's
                # census counts. The clock stays as it was; what was missing
                # was the count, so a wait census taken off a phase profile
                # read zero here and the fit's most reducible round trip was
                # invisible to the instrument that exists to find it.
                profile.charge(
                    PROF_GRAD_FILL,
                    n,
                    dev_grad_started,
                    syncs=1 if scale_synced else 0,
                )
                profile.note_wall(dev_grad_started)
                life.begin_tree()
                var tree = grow_tree_gpu_profiled(
                    profile,
                    builder,
                    searcher_cache,
                    params.tree,
                    dev_bag,
                    i,
                    split_search,
                )
                life.end_tree()
                var dev_post_started = profile.clock()
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
                if len(estimator) > 0:
                    # CatBoost's extra Newton steps, on the device, from the
                    # leaf ranges the grower left behind. The structure is
                    # fixed and no histogram is rebuilt: what moves is each
                    # leaf's `G` and `H`, and only because the point they are
                    # taken at moved. Placed before the degenerate-tree test
                    # below so that test sees the value the ensemble is about
                    # to carry, which is where `boosting._boost_rounds` places
                    # it too. Exclusive with renewal, which
                    # `_check_leaf_estimation_config` refuses above rather
                    # than ordering against.
                    var est_started = profile.clock()
                    var is_leaf = List[Bool](capacity=len(tree.feature))
                    for node in range(len(tree.feature)):
                        is_leaf.append(tree.feature[node] < 0)
                    # Read off the tree *before* its value list is handed over
                    # mutably: `node_bounds` reads `tree.value`, and the two
                    # borrows cannot overlap.
                    var est_bounds = node_bounds(tree, signs)
                    estimator[0].estimate(
                        builder.ctx,
                        state,
                        builder.rows,
                        tree.value,
                        is_leaf,
                        est_bounds,
                        objective,
                        alpha,
                        leaf_iters,
                        params.tree.lambda_l1,
                        params.tree.lambda_reg,
                        params.tree.extra.max_delta_step,
                    )
                    # Charged to the gradient phase because that is what the
                    # extra iterations overwhelmingly are: `leaf_iters - 1`
                    # more full-row derivative passes through the round's own
                    # kernel. `syncs=1` is the one readback per tree, which is
                    # a genuine round trip and would otherwise be invisible to
                    # the instrument that exists to find them.
                    profile.charge(
                        PROF_GRAD_FILL,
                        n * (leaf_iters - 1),
                        est_started,
                        syncs=1,
                    )
                if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                    life.end_round()
                    profile.note_wall(dev_post_started)
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
                profile.note_wall(dev_post_started)
            # The rounds the window was still holding when the loop stopped.
            # A no-op at the shipped cadence, where the window is always
            # empty here; at a wider one it is the round trip that makes the
            # overflow check cover the *last* few rounds rather than every
            # round but them. Outside the `for`, so a `break` on a degenerate
            # tree and a fit that ran its full budget both reach it.
            builder.flush_scale_window(state)
            profile.print_report()
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
            var pre_started = profile.clock()
            refresh_bag(bag, bagging, n, i)
            var grad_started = profile.clock()
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, grad, hess
            )
            # GOSS rescales the sampled rows' gradients before they are
            # uploaded, so the device histograms already carry the
            # compensation multiplier.
            goss_round(bag, grad, hess, goss, i, params.learning_rate)
            profile.charge(PROF_GRAD_FILL, n, grad_started)
            # The round's one per-row upload, which the device-objective path
            # above does not pay at all. Its own charge so the two paths can
            # be told apart by what crosses rather than by which branch ran.
            var upload_started = profile.clock()
            builder.upload_gradients(grad, hess)
            profile.charge(PROF_TRANSFER, n, upload_started)
            profile.note_wall(pre_started)
            life.begin_tree()
            var tree = grow_tree_gpu_profiled(
                profile,
                builder,
                searcher_cache,
                params.tree,
                bag,
                i,
                split_search,
            )
            life.end_tree()
            var post_started = profile.clock()
            if renews:
                _renew_leaf_values(
                    tree, data, target, raw, renew_w, renew_a, bag, signs,
                    params.tree.extra,
                )
            # The host arm's raw scores are the same `List[Float64]` the CPU
            # trainer carries, so this arm takes the CPU implementation
            # itself rather than a second one: same fold, same order, same
            # bits. Honoring the parameter on only one of this function's two
            # arms would be worse than not honoring it at all, which is the
            # rule `params.parse_params` states for this name.
            _estimate_leaf_values(
                tree, data, target, raw, objective, sample_weight, alpha,
                leaf_iters, params.tree.lambda_l1, params.tree.lambda_reg,
                params.tree.extra.max_delta_step, bag, signs,
            )

            # A single-leaf tree with a near-zero value means the objective
            # has converged; further rounds cannot make progress. Under
            # bagging or GOSS it only means this sample had nothing to give,
            # so the round is skipped and the next sample gets its turn.
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                life.end_round()
                profile.note_wall(post_started)
                if bagging_enabled(bagging) or goss.enabled:
                    continue
                break

            # The same serial full-tree traversal of every row the CPU
            # trainer performs, on the arm of the GPU trainer whose raw
            # scores live on the host. Same phase name, so the two backends'
            # score updates sit on the same line.
            var update_started = profile.clock()
            for r in range(n):
                raw[r] += params.learning_rate * tree.predict_row(data, r)
            profile.charge(PROF_SCORE_UPDATE, n, update_started)
            trees.append(tree^)
            life.end_round()
            profile.note_wall(post_started)

        profile.print_report()
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
    row_unroll: Bool = True,
    row_compaction: Bool = False,
    narrow_index: Bool = False,
    pair_alignment: Bool = True,
    min_tiles: Int = 0,
    rows_per_tile: Int = 0,
    scale_refresh: Int = 1,
    scale_headroom: Int = 0,
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
    below takes a `GpuSession` and is otherwise identical.

    `row_unroll` is a launch shape rather than a numeric option and defaults
    to the shape this trainer ships. It is an argument at all so that a
    benchmark can hold both arms in one process: this machine's device
    timings drift several-fold between time windows, so only interleaved arms
    compare, and an environment variable would force a two-build comparison.
    It cannot change a model. Both arms visit the same rows and add the same
    fixed-point integers into the same bins, and integer addition is
    associative, so the histograms are identical and therefore so is every
    split chosen from them. See `GpuActiveRows.set_row_unroll`.

    `narrow_index`, `pair_alignment`, `min_tiles`, and `rows_per_tile` are
    the K1 hist-latency lane's arms, here for the identical reason and with
    the identical guarantee: none of them can change a model.

    - `narrow_index` forms the histogram row loop's two data-dependent
      indices in Int32. Off by default, because nothing has measured it and
      the wide form's expensive term may already be hoisted; see
      `GpuActiveRows.set_narrow_index`. Exact under a bound on the dataset
      shape, which that setter refuses to launch outside of, so this argument
      raises rather than degrading on an oversized fit.
    - `pair_alignment` states the 8-byte alignment the quantized gradient
      pair's address has, instead of letting the width-2 load be annotated
      `align 4`. On by default: it reads the same eight bytes and a truer
      alignment cannot cost a backend anything.
    - `min_tiles` and `rows_per_tile` request a row-tile geometry per node.
      Zero on both, the default, is the geometry this trainer already
      produced. `rows_per_tile` is the only one that can ask for FEWER tiles
      than the occupancy term gives, which is what a re-test of the row-tile
      floor needs; see `gpu_tiling.row_tile_floor`.

    `scale_refresh` and `scale_headroom` are the **one pair here that is not
    a launch shape**, and they are on the same footing as `set_scale_shape`
    rather than as the arms above: they can change a model, and at any
    setting but the default they do. `scale_refresh` is how many rounds share
    one fixed-point scale, so it is how many of the fit's `2R` round trips the
    scale accounts for -- `R` at the default 1, `ceil(R/N) + 1` at `N`.
    `scale_headroom` is how many bits of lattice resolution every round gives
    up in exchange for room to reuse a scale safely. `1, 0` is the shipped
    cadence and the shipped arithmetic; `0` selects the unwindowed reference
    call. The rule, the overflow argument, the check that enforces it and the
    two distinct ways the bits move are all at
    `GpuHistogramBuilder.set_scale_refresh`, and nothing about them is
    restated here.

    They reach only the device-gradient arm, because the host-gradient arm
    derives its scale from a host pass over the gradient lists and never
    waits on the device for it. A bagged or GOSS fit therefore ignores both,
    silently, which is correct and is the reason a fit's readback count has
    to be read off the builder rather than derived from this signature.

    One hazard in how these defaults are spelled, since the calls below are
    unconditional: `GpuActiveRows.__init__` also sets every one of them, and
    for any fit that comes through here those initializations are dead. The
    shipped defaults are therefore written in two places that agree today and
    could silently stop agreeing. If either moves, move both, and prefer
    changing the constructor and passing that through to changing only this
    signature."""
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
        # This fit's constant-hessian declaration, made once, next to where
        # the trainer decides whether to pass weights. It holds for every
        # round because the three things it reads do: the objective code, the
        # weight vector, and the GOSS configuration are all arguments to this
        # function and none of them moves inside `_train_gpu_rounds`.
        #
        # It is correct on both of that loop's arms. On the host-gradient arm
        # `_fill_grad_hess` writes the constant, which is what the predicate
        # was read off. On the device-gradient arm the derivative kernel in
        # `gpu_objectives_native.mojo` writes `w` into `hess` in the same four
        # objective arms and `w` is the Float32 literal 1.0 when the state
        # carries no weights, so the two arms agree on the value as well as on
        # the objective. Bagging restricts which rows are accumulated and does
        # not touch a hessian, so it is not an exclusion; GOSS is, and
        # `round_has_constant_hessian` refuses it whether or not this run
        # would have reached the device round.
        builder.set_constant_hessian(
            round_has_constant_hessian(objective, sample_weight, goss)
        )
        builder.set_row_unroll(row_unroll)
        # A trade, not a strictly-less-work change, so it defaults
        # OFF and the window decides: compaction adds 4*L*(nf+8)
        # bytes of contiguous moves per split against scattered
        # gather traffic removed, and its own author put break-even
        # near depth 3 with a thin margin on a 31-leaf tree.
        # `or` the environment, not override it. An unconditional set here
        # made this parameter's default silently DISABLE
        # `MOJOTREES_GPU_ROW_COMPACTION=1` for every fit through
        # `train_gpu`, which is exactly how the arm's own test caught it:
        # "the on arm never reached the compaction; the environment
        # variable did not select it". A parameter that defaults off must
        # not out-rank an explicit request.
        builder.rows.set_row_compaction(
            row_compaction or builder.rows.row_compaction_requested()
        )
        # The three hist-latency arms, on the same footing and for the same
        # reason: a launch shape reachable in the call so a benchmark can
        # interleave arms in one process. None can change a model. The index
        # width is exact under a bound on the dataset shape that
        # `set_narrow_index` refuses to launch outside of; the pair alignment
        # is an assertion about an address over the same eight bytes; the tile
        # requests only change the order in which fixed-point Int32 adds are
        # issued. See `GpuActiveRows`.
        builder.set_narrow_index(narrow_index)
        builder.set_pair_alignment(pair_alignment)
        builder.set_row_tiling(min_tiles, rows_per_tile)
        # Not a launch shape. See the paragraph in this docstring and the
        # whole argument at `GpuHistogramBuilder.set_scale_refresh`.
        builder.set_scale_refresh(scale_refresh, scale_headroom)
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
    row_unroll: Bool = True,
    row_compaction: Bool = False,
    narrow_index: Bool = False,
    pair_alignment: Bool = True,
    min_tiles: Int = 0,
    rows_per_tile: Int = 0,
    scale_refresh: Int = 1,
    scale_headroom: Int = 0,
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
        # The same declaration the session-free overload makes, from the same
        # predicate over the same three arguments. A session owns the context
        # and not the round's numbers, so it cannot change the answer.
        builder.set_constant_hessian(
            round_has_constant_hessian(objective, sample_weight, goss)
        )
        # A launch shape, not a number: see the session-free overload. A
        # session owns the context, so it cannot change this answer either.
        builder.set_row_unroll(row_unroll)
        # A trade, not a strictly-less-work change, so it defaults
        # OFF and the window decides: compaction adds 4*L*(nf+8)
        # bytes of contiguous moves per split against scattered
        # gather traffic removed, and its own author put break-even
        # near depth 3 with a thin margin on a 31-leaf tree.
        # `or` the environment, not override it. An unconditional set here
        # made this parameter's default silently DISABLE
        # `MOJOTREES_GPU_ROW_COMPACTION=1` for every fit through
        # `train_gpu`, which is exactly how the arm's own test caught it:
        # "the on arm never reached the compaction; the environment
        # variable did not select it". A parameter that defaults off must
        # not out-rank an explicit request.
        builder.rows.set_row_compaction(
            row_compaction or builder.rows.row_compaction_requested()
        )
        # The same three arms the session-free overload sets, from the same
        # arguments. A session owns the context, not the geometry, so it
        # cannot change these answers either.
        builder.set_narrow_index(narrow_index)
        builder.set_pair_alignment(pair_alignment)
        builder.set_row_tiling(min_tiles, rows_per_tile)
        # A session owns the context, not the round's numbers, so it cannot
        # change this answer either -- but unlike the four above, this one is
        # a number and not a shape. See `set_scale_refresh`.
        builder.set_scale_refresh(scale_refresh, scale_headroom)
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
        # A custom objective has no second derivative this trainer can
        # recompute: the callback is asked once per round for the whole row
        # set, and `boosting._estimate_leaf_values` needs a *leaf's* rows
        # re-differentiated at a shifted score, which is a call shape
        # `GradHessFn` does not have. Refused by name rather than ignored.
        _refuse_leaf_estimation(params.tree.extra, "train_custom_gpu")
        var n = data.n_rows
        var raw = List[Float64](capacity=n)
        for _ in range(n):
            raw.append(base_score)

        var trees = List[Tree]()
        # One searcher for the fit, not one per tree. Its shape comes from
        # the builder and the tree budget, neither of which moves between the
        # rounds below; `GpuSplitSearcherCache` states what the per-tree reset
        # restores and why reuse cannot change a tree. Stays empty, and
        # allocates nothing, on the host-search path.
        var searcher_cache = GpuSplitSearcherCache()
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
                builder,
                searcher_cache,
                params.tree,
                [],
                i,
                split_search,
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
        _check_gpu_booster_params(params, String("train_custom_gpu"))

        var builder = GpuHistogramBuilder(data)
        # No constant-hessian declaration on a custom objective, and the
        # builder's default of False is what leaves the three-plane path in
        # force. The hessians here are whatever `grad_hess` returns, from a
        # callback this package cannot read, and `_apply_sample_weight`
        # multiplies them by the row weight afterwards on top of that. Neither
        # is something a predicate over an objective code could rule on, which
        # is exactly why `objective_has_constant_hessian` returns False for
        # `CUSTOM` and why nothing here tries to be cleverer than that.
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
        _check_gpu_booster_params(params, String("train_custom_gpu"))

        var fit = session.begin_fit()
        var builder = GpuHistogramBuilder(session, data)
        # No constant-hessian declaration, for the reason the session-free
        # overload gives: a custom objective's hessians come from a caller's
        # callback and no predicate here can rule on them.
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
        # One searcher for the fit, not one per tree. Its shape comes from
        # the builder and the tree budget, neither of which moves between the
        # rounds below; `GpuSplitSearcherCache` states what the per-tree reset
        # restores and why reuse cannot change a tree. Stays empty, and
        # allocates nothing, on the host-search path.
        var searcher_cache = GpuSplitSearcherCache()
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
                        searcher_cache,
                        params.tree,
                        [],
                        i * n_classes + k,
                        split_search,
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
        # The same refusal `boosting._boost_rounds_multiclass` makes, for the
        # same reason: class k's softmax derivative reads every class's raw
        # score, so re-estimating class k's leaves would have to hold this
        # round's not-yet-grown trees for classes k+1.. fixed at a value they
        # do not have. CatBoost defaults `MultiClass` to 1 Newton iteration
        # anyway (`boosting.catboost_leaf_estimation_iterations`).
        _refuse_leaf_estimation(
            params.tree.extra, "the multiclass GPU trainers"
        )
        var n = data.n_rows
        var trees = List[Tree]()
        # One searcher for the fit, not one per tree. Its shape comes from
        # the builder and the tree budget, neither of which moves between the
        # rounds below; `GpuSplitSearcherCache` states what the per-tree reset
        # restores and why reuse cannot change a tree. Stays empty, and
        # allocates nothing, on the host-search path.
        var searcher_cache = GpuSplitSearcherCache()

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
                        searcher_cache,
                        params.tree,
                        [],
                        i * n_classes + k,
                        split_search,
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
                    searcher_cache,
                    params.tree,
                    bag,
                    i * n_classes + k,
                    split_search,
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
        _check_gpu_booster_params(params, String("train_multiclass_gpu"))
        var base_scores = _multiclass_base_scores(
            labels, n_classes, sample_weight
        )
        var device_grads = device_gradients(
            _SOFTMAX_OBJECTIVE, n_classes, objective_source, bagging, goss
        )
        var builder = GpuHistogramBuilder(data)
        # No constant-hessian declaration on a softmax run, at any class
        # count. `boosting._fill_softmax_grad_hess` and the device kernels it
        # mirrors write `(k / (k - 1)) * p * (1 - p)`, floored, which is a
        # function of that row's class probability, so the hessian plane is
        # genuinely per row. `_SOFTMAX_OBJECTIVE` binds `SQUARED_ERROR` as a
        # placeholder for the device-round eligibility question, and handing
        # that placeholder to the predicate would read it as a regression
        # objective and declare a guarantee no softmax round makes; nothing
        # here does.
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
        _check_gpu_booster_params(params, String("train_multiclass_gpu"))
        var base_scores = _multiclass_base_scores(
            labels, n_classes, sample_weight
        )
        var device_grads = device_gradients(
            _SOFTMAX_OBJECTIVE, n_classes, objective_source, bagging, goss
        )
        var fit = session.begin_fit()
        var builder = GpuHistogramBuilder(session, data)
        # No constant-hessian declaration, for the reason the session-free
        # overload gives: a softmax hessian varies with the row's class
        # probability.
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
        # The same once-per-fit check and the same count `_train_gpu_rounds`
        # makes. This loop's raw scores are host-side throughout, so the extra
        # Newton steps are the CPU implementation itself.
        var leaf_iters = params.tree.extra.leaf_estimation_iterations
        _check_leaf_estimation_config(params.tree.extra, objective, goss)
        var trees = List[Tree]()
        # One searcher for the fit, not one per tree. Its shape comes from
        # the builder and the tree budget, neither of which moves between the
        # rounds below; `GpuSplitSearcherCache` states what the per-tree reset
        # restores and why reuse cannot change a tree. Stays empty, and
        # allocates nothing, on the host-search path.
        var searcher_cache = GpuSplitSearcherCache()
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
                builder,
                searcher_cache,
                params.tree,
                bag,
                i,
                split_search,
            )
            if renews:
                _renew_leaf_values(
                    tree, data, target, raw, renew_w, renew_a, bag, signs,
                    params.tree.extra,
                )
            # The same extra Newton steps, in the same place and with the same
            # arguments; a no-op at the default of 1, and taken before the
            # scorer folds the tree in so early stopping judges the values the
            # ensemble will carry.
            _estimate_leaf_values(
                tree, data, target, raw, objective, sample_weight, alpha,
                leaf_iters, params.tree.lambda_l1, params.tree.lambda_reg,
                params.tree.extra.max_delta_step, bag, signs,
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
        _check_gpu_booster_params(params, String("train_gpu_with_valid"))

        var builder = GpuHistogramBuilder(data)
        # The same declaration `train_gpu` makes, over the same three inputs.
        # `_train_gpu_valid_rounds` has one arm and it is the host-gradient
        # one, so the hessians this covers are exactly the ones
        # `_fill_grad_hess` writes. The validation matrix is scored and never
        # accumulated into a histogram, so neither its rows nor its labels
        # bear on this; early stopping truncates the ensemble afterwards and
        # likewise changes nothing about how a tree was grown.
        builder.set_constant_hessian(
            round_has_constant_hessian(objective, sample_weight, goss)
        )
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
