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

Division of labor:
  CPU  boosting coordination, split selection over downloaded histograms,
       leaf-value renewal (quantile/L1), prediction, the tree model itself
  GPU  binned features, gradients/hessians, leaf assignments, histogram
       accumulation, row partitioning

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

  gradients      `objective_source` / `MOJOBOOST_GPU_OBJECTIVE`, one of
                 `auto` (the shipped behavior: the device kernels whenever
                 the objective has them and no row sampling is configured),
                 `host` (never; upload from `_fill_grad_hess` instead), or
                 `device` (a hard requirement, which raises with a specific
                 reason when the kernels cannot serve it)
  split search   `split_search` / `MOJOBOOST_GPU_SPLIT_STRATEGY`, defaulting
                 to the host scan over downloaded histograms; see the
                 SPLIT_SEARCH_* constants
  validation     `valid_scoring` / `MOJOBOOST_GPU_VALID_SCORING`, defaulting
                 to the host tree walk; see `train_gpu_with_valid` and the
                 VALID_SCORE_* constants
  session        each trainer has an overload taking a `GpuSession`
                 (gpu_runtime.mojo). Without one the trainers run on
                 `NoLifecycle` and execute exactly the device calls they did
                 before the seam existed; with one, the builder borrows the
                 session's context and the round and tree boundaries are
                 announced to it. The session is bookkeeping today: nothing
                 yet owns one across two fits, which is what would let its
                 pool and residency ledgers skip an upload rather than only
                 record that they could have.
"""

from std.math import log
from std.os import getenv
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from .bagging import BaggingParams, bagging_enabled, check_bagging, refresh_bag
from .binning import BinnedMatrix
from .boosting import (
    CUSTOM,
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
from .gpu_objectives_native import supports_device_objective
from .gpu_predict import GpuPredictor, flatten_trees
from .gpu_runtime import GpuSession, NoLifecycle, RoundLifecycle
from .gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
)
from .gpu_tiling import DeviceCaps
from .histogram import Histogram, subtract_histogram
from .histogram_gpu import GpuHistogramBuilder
from .objective import (
    GradHessFn,
    _apply_sample_weight,
    check_custom_grad_hess,
)
from .interaction import extend_branch
from .monotone import (
    MONOTONE_FREE,
    OutputBounds,
    child_bounds,
    midpoint,
    monotone_sign,
)
from .sampling import select_node_features, select_tree_features
from .split import SplitInfo
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
# CPU/GPU split decisions identical and is the default until a benchmark
# says otherwise. DEVICE scans the histogram where it was accumulated (see
# gpu_split_search.mojo) and downloads one 136-byte record per node instead
# of the whole histogram; its gains and leaf values are Float32, so split
# decisions can differ from the host's on near-ties. AUTO reads
# `MOJOBOOST_GPU_SPLIT_STRATEGY` (`host` or `device`) and then defaults to
# HOST.
comptime SPLIT_SEARCH_AUTO = 0
comptime SPLIT_SEARCH_HOST = 1
comptime SPLIT_SEARCH_DEVICE = 2


def env_split_search() -> Int:
    """`MOJOBOOST_GPU_SPLIT_STRATEGY` as a split-search constant."""
    var s = getenv("MOJOBOOST_GPU_SPLIT_STRATEGY")
    if s == "device":
        return SPLIT_SEARCH_DEVICE
    if s == "host":
        return SPLIT_SEARCH_HOST
    return SPLIT_SEARCH_AUTO


def resolve_split_search(strategy: Int) -> Int:
    """An explicit strategy outranks the environment; AUTO resolves through
    `MOJOBOOST_GPU_SPLIT_STRATEGY` and then to the host scan."""
    var s = strategy
    if s == SPLIT_SEARCH_AUTO:
        s = env_split_search()
    if s == SPLIT_SEARCH_DEVICE:
        return SPLIT_SEARCH_DEVICE
    return SPLIT_SEARCH_HOST


# Where a round's gradients come from. DEVICE generates them on the device
# from device-resident labels and raw scores (gpu_objectives_native.mojo) and
# advances the raw scores there too, so a round moves nothing per row. HOST
# evaluates the objective in `_fill_grad_hess` over host-side raw scores and
# uploads the result, which is the path row sampling and custom objectives
# need and the one every GPU round took before the device objectives landed.
# AUTO reads `MOJOBOOST_GPU_OBJECTIVE` (`host` or `device`) and then takes
# the device path wherever it is available.
comptime OBJECTIVE_SOURCE_AUTO = 0
comptime OBJECTIVE_SOURCE_HOST = 1
comptime OBJECTIVE_SOURCE_DEVICE = 2


def env_objective_source() -> Int:
    """`MOJOBOOST_GPU_OBJECTIVE` as an objective-source constant."""
    var s = getenv("MOJOBOOST_GPU_OBJECTIVE")
    if s == "device":
        return OBJECTIVE_SOURCE_DEVICE
    if s == "host":
        return OBJECTIVE_SOURCE_HOST
    return OBJECTIVE_SOURCE_AUTO


def resolve_objective_source(source: Int) -> Int:
    """An explicit source outranks the environment; AUTO resolves through
    `MOJOBOOST_GPU_OBJECTIVE` and then stays AUTO, since whether the device
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


def device_gradients(
    supported: Bool,
    source: Int,
    bagging: BaggingParams,
    goss: GossParams,
) raises -> Bool:
    """Whether this run generates its gradients on the device.

    Two things can rule the device path out, and a caller that asked for it
    explicitly is told which one rather than being quietly downgraded: an
    objective with no device kernel (`supported` comes from
    `supports_device_objective`, and softmax passes True since
    `fill_softmax_gradients_device` covers it), and row sampling, whose
    sample is ranked and drawn on the host, so its gradients have to exist
    there to be scaled and uploaded. Under AUTO either one falls back
    silently, which is what the shipped trainers already do."""
    var s = resolve_objective_source(source)
    if s == OBJECTIVE_SOURCE_HOST:
        return False
    if not supported:
        if s == OBJECTIVE_SOURCE_DEVICE:
            raise Error(
                "this objective has no device gradient kernel; train it with"
                " objective_source=OBJECTIVE_SOURCE_HOST (or"
                " MOJOBOOST_GPU_OBJECTIVE=host), which uploads host-computed"
                " gradients and grows the trees on the device exactly as"
                " before"
            )
        return False
    if bagging_enabled(bagging) or goss.enabled:
        if s == OBJECTIVE_SOURCE_DEVICE:
            raise Error(
                "row sampling draws its sample from host-side gradients, so"
                " bagging and GOSS cannot take the device objective path;"
                " use objective_source=OBJECTIVE_SOURCE_AUTO or"
                " OBJECTIVE_SOURCE_HOST"
            )
        return False
    return True


struct _GpuRecordLeafState(Movable):
    """A grown-but-unsplit leaf under device split selection: the compact
    search record stands in for the histogram the host-search frontier
    carries, since the record already holds the split, both children's
    counts and Newton values, and the parent's value."""

    var node: Int
    var n_rows: Int
    var rec: GpuSplitRecord
    var branch: List[Int]
    var depth: Int
    var bounds: OutputBounds

    def __init__(
        out self,
        node: Int,
        n_rows: Int,
        var rec: GpuSplitRecord,
        var branch: List[Int] = [],
        depth: Int = 0,
        var bounds: OutputBounds = OutputBounds.unbounded(),
    ):
        self.node = node
        self.n_rows = n_rows
        self.rec = rec^
        self.branch = branch^
        self.depth = depth
        self.bounds = bounds^


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
    if params.max_depth > 0 and depth >= params.max_depth:
        rec.found = False
    if n_rows < 2 * params.min_data_in_leaf or n_rows < 2:
        rec.found = False
    return rec^


def _grow_tree_gpu_device_search(
    mut builder: GpuHistogramBuilder,
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
) raises -> Tree:
    """`grow_tree_gpu` with split selection on the device (stage 1 of
    handoffs/apple_a2_split_search.md): every node's histogram is built and
    searched where it lives, and only a 136-byte record crosses to the host
    per node instead of the whole `3 * n_features * n_bins` histogram.

    Both children's histograms are built on the device, which replaces the
    host-side sibling subtraction: one extra histogram build per split in
    exchange for the removed per-node transfer. That is a tradeoff, not a
    claimed speedup; no benchmark has compared the two paths yet.

    Gains, hessian tests, and leaf values are Float32 on the device, so a
    near-tie between two candidates can resolve differently than the host
    scan and CPU/GPU tree shapes can differ there; child row counts are
    exact integers either way. Selection is still bit-deterministic run to
    run. Shape rules (depth limit, minimum rows), monotone clamping with
    the midpoint collapse, interaction masks, and per-node feature
    subsampling all stay identical to the host path."""
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
    )
    searcher.set_monotone(signs)
    var split_params = GpuSplitParams(
        params.lambda_reg,
        params.lambda_l1,
        params.min_child_hess,
        params.min_data_in_leaf,
        params.cat.copy(),
    )
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )

    builder.begin_tree(bag)
    var n_root = len(bag) if len(bag) > 0 else builder.n_rows
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

    while n_leaves < params.num_leaves:
        # Pick the leaf with the best gain anywhere in the tree, ties to
        # the lower frontier index, exactly as the host-search loop does.
        var best_i = -1
        var best_gain = 0.0
        for i in range(len(frontier)):
            if frontier[i].rec.found and frontier[i].rec.gain > best_gain:
                best_gain = frontier[i].rec.gain
                best_i = i
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

        # Same clamp-and-divide as the host paths: no-ops when
        # unconstrained. The record carries the raw Newton values.
        var parent_bounds = frontier[best_i].bounds.copy()
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


def grow_tree_gpu(
    mut builder: GpuHistogramBuilder,
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
    SPLIT_SEARCH_* constants above): the default resolves to this host
    scan, and SPLIT_SEARCH_DEVICE routes to the device-side scan, which
    trades the identical-split guarantee for a fixed 136-byte per-node
    readback."""
    if resolve_split_search(split_search) == SPLIT_SEARCH_DEVICE:
        return _grow_tree_gpu_device_search(builder, params, bag, tree_index)
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
    # Leaf-value totals must come from a feature the histograms accumulated.
    var value_feature = tree_features[0]
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )

    builder.begin_tree(bag)
    var n_root = len(bag) if len(bag) > 0 else builder.n_rows
    var root = tree._add_node(0.0, Float64(n_root))
    var root_hist = builder.build_leaf(root)
    tree.value[root] = _leaf_value(
        root_hist, params.lambda_reg, params.lambda_l1, value_feature
    )
    var root_branch = List[Int]()
    var root_split = _search(
        root_hist,
        n_root,
        params,
        params.constraints.allowed_features(root_branch),
        select_node_features(
            tree_features,
            params.feature_fraction_bynode,
            params.feature_fraction_seed,
            tree_index,
            root,
        ),
        depth=0,
        missing_bins=builder.missing_bin,
        monotone=signs,
        cats=builder.cats,
    )

    var frontier = List[_GpuLeafState]()
    frontier.append(
        _GpuLeafState(
            root, n_root, root_hist^, root_split^, root_branch^, depth=0
        )
    )
    var n_leaves = 1

    while n_leaves < params.num_leaves:
        # Pick the leaf with the best gain anywhere in the tree.
        var best_i = -1
        var best_gain = 0.0
        for i in range(len(frontier)):
            if frontier[i].split.found and frontier[i].split.gain > best_gain:
                best_gain = frontier[i].split.gain
                best_i = i
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

        # Histogram subtraction trick: build the smaller child directly.
        var left_hist: Histogram
        var right_hist: Histogram
        if n_left <= n_right:
            left_hist = builder.build_leaf(left_node)
            right_hist = subtract_histogram(frontier[best_i].hist, left_hist)
        else:
            right_hist = builder.build_leaf(right_node)
            left_hist = subtract_histogram(frontier[best_i].hist, right_hist)

        # Same clamp-and-divide as the CPU grower: no-ops when unconstrained.
        var parent_bounds = frontier[best_i].bounds.copy()
        var split_sign = monotone_sign(signs, split.feature)
        var left_value = parent_bounds.clamp(
            _leaf_value(
                left_hist, params.lambda_reg, params.lambda_l1, value_feature
            )
        )
        var right_value = parent_bounds.clamp(
            _leaf_value(
                right_hist, params.lambda_reg, params.lambda_l1, value_feature
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
        var left_split = _search(
            left_hist,
            n_left,
            params,
            allowed,
            select_node_features(
                tree_features,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                left_node,
            ),
            depth=child_depth,
            missing_bins=builder.missing_bin,
            monotone=signs,
            cats=builder.cats,
            bounds=children.left,
        )
        var right_split = _search(
            right_hist,
            n_right,
            params,
            allowed,
            select_node_features(
                tree_features,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                right_node,
            ),
            depth=child_depth,
            missing_bins=builder.missing_bin,
            monotone=signs,
            cats=builder.cats,
            bounds=children.right,
        )

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
) raises -> Booster:
    """The boosting loop both `train_gpu` entry points run, over a builder
    the caller already constructed and a lifecycle it already chose.

    `life` is `NoLifecycle` for the session-free entry point, which makes
    every hook below two integer increments and no device work, so this loop
    issues exactly the sequence it issued before the session seam existed.
    A `GpuSession` in its place moves the session's state machine and gives
    `MOJOBOOST_GPU_TRACE=1` its per-round and per-tree counts."""
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
            for i in range(params.n_estimators):
                life.begin_round()
                builder.fill_gradients_device(state, objective, alpha)
                life.begin_tree()
                var tree = grow_tree_gpu(
                    builder, params.tree, [], i, split_search
                )
                life.end_tree()
                if renews:
                    # Renewal is a host-side weighted percentile, so the
                    # renewing objectives pay one raw-score download per
                    # tree; the scores come back through Float32, which is
                    # the device path's documented precision.
                    var raw = state.download_raw(builder.ctx)
                    _renew_leaf_values(
                        tree, data, target, raw, renew_w, renew_a, [], signs
                    )
                if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                    life.end_round()
                    break
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
                builder, params.tree, bag, i, split_search
            )
            life.end_tree()
            if renews:
                _renew_leaf_values(
                    tree, data, target, raw, renew_w, renew_a, bag, signs
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
        var device_grads = device_gradients(
            supports_device_objective(objective),
            objective_source,
            bagging,
            goss,
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
    under `MOJOBOOST_GPU_TRACE=1`. The trainer never closes it."""
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
        var device_grads = device_gradients(
            supports_device_objective(objective),
            objective_source,
            bagging,
            goss,
        )
        var builder = GpuHistogramBuilder(session, data)
        return _train_gpu_rounds(
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
        )


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
            var tree = grow_tree_gpu(builder, params.tree, [], i, split_search)
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

        var builder = GpuHistogramBuilder(session, data)
        return _train_custom_gpu_rounds(
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
    intervening round boundary."""
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
            for i in range(params.n_estimators):
                life.begin_round()
                state.refresh_softmax(builder.ctx)
                var made_progress = False
                for k in range(n_classes):
                    builder.fill_softmax_gradients_device(state, k)
                    life.begin_tree()
                    var tree = grow_tree_gpu(
                        builder,
                        params.tree,
                        [],
                        i * n_classes + k,
                        split_search,
                    )
                    life.end_tree()
                    if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                        made_progress = True
                    # Before the next class's begin_tree resets the ranges.
                    builder.update_raw_device(
                        state, tree.value, params.learning_rate, k
                    )
                    trees.append(tree^)
                life.end_round()
                if not made_progress:
                    for _ in range(n_classes):
                        _ = trees.pop()
                    break
            return MulticlassBooster(
                trees^, base_scores^, n_classes, params.learning_rate
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
        for i in range(params.n_estimators):
            life.begin_round()
            refresh_bag(bag, bagging, n, i)
            for r in range(n):
                for k in range(n_classes):
                    prob[r * n_classes + k] = raw[r * n_classes + k]
                _softmax_inplace(prob, r * n_classes, n_classes)

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
                # Feature subsampling draws once per tree, so each class's
                # tree in a round gets its own feature set; the same index
                # the CPU grower uses keeps the two backends on identical
                # feature sets.
                life.begin_tree()
                var tree = grow_tree_gpu(
                    builder, params.tree, bag, i * n_classes + k, split_search
                )
                life.end_tree()
                if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                    made_progress = True
                for r in range(n):
                    raw[r * n_classes + k] += (
                        params.learning_rate * tree.predict_row(data, r)
                    )
                trees.append(tree^)
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
            trees^, base_scores^, n_classes, params.learning_rate
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
            True, objective_source, bagging, goss
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
            True, objective_source, bagging, goss
        )
        var builder = GpuHistogramBuilder(session, data)
        return _train_multiclass_gpu_rounds(
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


# ---------------------------------------------------------------------------
# Validation scoring
# ---------------------------------------------------------------------------

# Where a validation set's running raw scores are maintained. HOST walks the
# round's tree over every validation row on the host, which is what
# `train_with_valid` in boosting.mojo does and what this trainer does until a
# benchmark says otherwise. DEVICE keeps the validation matrix, its labels,
# and the running raw-score vector resident on the training context and folds
# each round's tree in with one kernel (see gpu_predict.mojo), downloading the
# scores for the loss. AUTO reads `MOJOBOOST_GPU_VALID_SCORING` (`host` or
# `device`) and then defaults to HOST.
comptime VALID_SCORE_AUTO = 0
comptime VALID_SCORE_HOST = 1
comptime VALID_SCORE_DEVICE = 2


def env_valid_scoring() -> Int:
    """`MOJOBOOST_GPU_VALID_SCORING` as a validation-scoring constant."""
    var s = getenv("MOJOBOOST_GPU_VALID_SCORING")
    if s == "device":
        return VALID_SCORE_DEVICE
    if s == "host":
        return VALID_SCORE_HOST
    return VALID_SCORE_AUTO


def resolve_valid_scoring(scoring: Int) -> Int:
    """An explicit choice outranks the environment; AUTO resolves through
    `MOJOBOOST_GPU_VALID_SCORING` and then to the host walk."""
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

    Two deliberate limits. The loss itself is computed on the host, from the
    downloaded raw scores, through the same `_mean_loss` the CPU trainer and
    the host scorer use: gpu_predict.mojo can reduce a metric on the device,
    but its metric set is not the objective loss set `_mean_loss` covers, and
    a stopping decision made from a different loss definition is not the same
    run. The raw scores themselves accumulate in Float32 (Apple GPUs have no
    Float64), so two rounds within Float32 noise of each other can order
    differently than they would on the host path and pick a different
    `best_iteration`. That is why this is not the default.
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
                builder, params.tree, bag, i, split_search
            )
            if renews:
                _renew_leaf_values(
                    tree, data, target, raw, renew_w, renew_a, bag, signs
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
