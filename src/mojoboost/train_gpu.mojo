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
"""

from std.math import log
from std.os import getenv
from std.sys import has_accelerator

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
from .gpu_split_search import (
    GpuSplitParams,
    GpuSplitRecord,
    GpuSplitSearcher,
)
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


def train_gpu(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Booster:
    """Train a boosted ensemble with tree growth on the GPU. Same contract
    as `train` (objectives, sample_weight, alpha, bagging, and GOSS
    semantics); requires an accelerator at runtime and at most 256 bins.
    Bags come from the same sampler and the same schedule as on the CPU, so
    both backends grow round i on exactly the same rows. GOSS ranks rows on
    the host from the same Float64 gradients the CPU trainer uses, so its
    sample matches the CPU sample exactly as well; only the histograms the
    sample feeds carry the GPU's Float32 precision."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(target) != data.n_rows:
            raise Error("target length must equal n_rows")
        _check_objective(objective, target, alpha)
        _check_sample_weight(sample_weight, data.n_rows)
        check_bagging(bagging)
        _check_goss(goss, bagging)
        params.tree.monotone.check_features(data.n_features)

        var n = data.n_rows
        var base_score = _base_score(target, objective, sample_weight, alpha)

        var signs = params.tree.monotone.active_signs()
        var renews = objective_renews_leaves(objective)
        var renew_w = renewal_weights(objective, target, sample_weight)
        var renew_a = renewal_alpha(objective, alpha)
        var builder = GpuHistogramBuilder(data)
        var trees = List[Tree]()

        # Built-in objectives without row sampling generate their gradients
        # on the device and advance the raw scores there too, so a round
        # uploads nothing per row: labels and weights cross once at state
        # construction, and only the tree's node-value table (a few hundred
        # bytes) crosses per tree. Bagging and GOSS stay on the host path,
        # which needs the gradients host-side to rank and sample rows.
        if not bagging_enabled(bagging) and not goss.enabled and (
            supports_device_objective(objective)
        ):
            var state = builder.objective_state(
                target, sample_weight, 1, 2 * params.tree.num_leaves
            )
            state.init_raw(builder.ctx, [base_score])
            for i in range(params.n_estimators):
                builder.fill_gradients_device(state, objective, alpha)
                var tree = grow_tree_gpu(builder, params.tree, [], i)
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
                    break
                builder.update_raw_device(
                    state, tree.value, params.learning_rate
                )
                trees.append(tree^)
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
            refresh_bag(bag, bagging, n, i)
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, grad, hess
            )
            # GOSS rescales the sampled rows' gradients before they are
            # uploaded, so the device histograms already carry the
            # compensation multiplier.
            goss_round(bag, grad, hess, goss, i, params.learning_rate)
            builder.upload_gradients(grad, hess)
            var tree = grow_tree_gpu(builder, params.tree, bag, i)
            if renews:
                _renew_leaf_values(
                    tree, data, target, raw, renew_w, renew_a, bag, signs
                )

            # A single-leaf tree with a near-zero value means the objective
            # has converged; further rounds cannot make progress. Under
            # bagging or GOSS it only means this sample had nothing to give,
            # so the round is skipped and the next sample gets its turn.
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                if bagging_enabled(bagging) or goss.enabled:
                    continue
                break

            for r in range(n):
                raw[r] += params.learning_rate * tree.predict_row(data, r)
            trees.append(tree^)

        return Booster(
            trees^,
            base_score,
            params.learning_rate,
            objective,
            params.tree.monotone.copy(),
        )


def train_custom_gpu[F: GradHessFn](
    data: BinnedMatrix,
    target: List[Float64],
    grad_hess: F,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
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

        var n = data.n_rows
        var raw = List[Float64](capacity=n)
        for _ in range(n):
            raw.append(base_score)

        var builder = GpuHistogramBuilder(data)
        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        for i in range(params.n_estimators):
            grad_hess(raw, target, grad, hess)
            check_custom_grad_hess(grad, hess, n)
            _apply_sample_weight(grad, hess, sample_weight)
            builder.upload_gradients(grad, hess)
            var tree = grow_tree_gpu(builder, params.tree, [], i)

            # A single-leaf tree with a near-zero value means the objective
            # has converged; further rounds cannot make progress.
            if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
                break

            for r in range(n):
                raw[r] += params.learning_rate * tree.predict_row(data, r)
            trees.append(tree^)

        return Booster(
            trees^,
            base_score,
            params.learning_rate,
            CUSTOM,
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
    bins. Softmax probabilities are computed on the host, exactly
    as on the CPU, so the only backend difference remains the Float32
    histogram precision."""
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
        var n = data.n_rows

        # Base scores are log priors (weighted when sample_weight is given).
        var class_w = List[Float64](capacity=n_classes)
        for _ in range(n_classes):
            class_w.append(0.0)
        var total_w = 0.0
        for r in range(n):
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

        var builder = GpuHistogramBuilder(data)
        var trees = List[Tree]()

        # Softmax without row sampling runs the whole objective on the
        # device: probabilities refresh once per round, each class's
        # gradients land straight in the histogram buffers, and each class's
        # tree advances the device raw scores from its leaf ranges. Bagging
        # and GOSS keep the host path, which owns the row sample.
        if not bagging_enabled(bagging) and not goss.enabled:
            var labels_f = List[Float64](capacity=n)
            for r in range(n):
                labels_f.append(Float64(labels[r]))
            var state = builder.objective_state(
                labels_f, sample_weight, n_classes, 2 * params.tree.num_leaves
            )
            state.init_raw(builder.ctx, base_scores)
            for i in range(params.n_estimators):
                state.refresh_softmax(builder.ctx)
                var made_progress = False
                for k in range(n_classes):
                    builder.fill_softmax_gradients_device(state, k)
                    var tree = grow_tree_gpu(
                        builder, params.tree, [], i * n_classes + k
                    )
                    if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                        made_progress = True
                    # Before the next class's begin_tree resets the ranges.
                    builder.update_raw_device(
                        state, tree.value, params.learning_rate, k
                    )
                    trees.append(tree^)
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
                var tree = grow_tree_gpu(
                    builder, params.tree, bag, i * n_classes + k
                )
                if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                    made_progress = True
                for r in range(n):
                    raw[r * n_classes + k] += (
                        params.learning_rate * tree.predict_row(data, r)
                    )
                trees.append(tree^)

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
