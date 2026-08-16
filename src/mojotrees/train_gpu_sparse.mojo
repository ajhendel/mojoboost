"""End-to-end GPU training on sparse input.

`train_gpu_sparse` mirrors `train_sparse` in boosting_sparse.mojo the way
`train_gpu` mirrors `train`: the objective layer, the bag and GOSS
schedules, leaf renewal, and the raw-score update are the CPU sparse
trainer's own, imported and called unchanged, and only the tree grower
differs. `grow_tree_gpu_sparse` drives one `GpuSparseHistogramBuilder`
(gpu_sparse.mojo) for a whole session: the CSC binned matrix is uploaded
once, every live node owns a device-resident row range and per-feature
entry windows, a node's histogram is accumulated on the device from its
own stored entries with the implicit zeros folded in by subtraction, and a
split partitions both index structures on the device by one side mask.
Nothing is densified anywhere: a `SparseBinnedMatrix` goes to the device
compressed and comes back, per node, as an ordinary `Histogram`.

Division of labor, the same one the dense GPU trainer keeps:
  CPU  boosting coordination, split selection over downloaded histograms
       (`tree._search`, so CPU and GPU sparse fits choose from identical
       rules), the sibling histogram by subtraction, leaf values, leaf
       renewal, the raw scores, validation scoring, and the tree itself
  GPU  the compressed matrix, gradients and hessians, the row permutation
       and the entry permutation, node totals, histogram accumulation, the
       default-bin leftover, and both partitions at every split

The accumulation half of that skips one more bin than it used to. On a
column where the default bin already holds a majority of the stored entries,
those entries are not accumulated either and the same leftover recovers
them, which is LightGBM's `FixHistogramKernel` rule and is exact in this
fixed point rather than approximate; gpu_sparse.mojo derives it. Nothing
here has to know: the histograms are unchanged cell for cell, so split
selection, sibling subtraction, and leaf values all read the same numbers
they did before. `MOJOTREES_GPU_SPARSE_SKIP_FREQ` is the threshold and 0
turns it off. Whether it is faster has not been measured on any device.

Categorical splits go through gpu_categorical.mojo: the set the host search
produced is staged in a `CatSetPool`, checked (`check_cat_bitset`, so a set
that would reverse the routing of every missing and unseen row is refused
before it reaches a kernel), uploaded, and applied by
`apply_categorical_split_pooled`, whose side kernel reads the set from the
device pool. Numerical splits take `GpuSparseHistogramBuilder.apply_split`
directly. Both end in the builder's `finish_split`, so the two arms share
one partition.

What comes out is the ordinary `Booster` / `MulticlassBooster` the CPU
sparse trainer returns: same serialization, same prediction, no sparse or
device residue in the model. Trees agree with `train_sparse`'s to Float32
level (the histograms are fixed-point Int32 on the device, see
histogram_gpu.mojo), not bit-exactly, exactly as `train_gpu` agrees with
`train`; a split decision on a near tie can therefore differ. The device
path itself is bit-deterministic run to run.

Not on this path, and refused rather than approximated:
  - exclusive feature bundling. The device split kernel routes on the
    stored column bin and knows no bundle's local-bin table, so a bundled
    matrix cannot be split correctly here; `train_gpu_sparse` raises for
    an active `SparseBundling` and for an unhonored `enable_bundle` alike.
  - custom objectives (`objective.mojo` takes a `BinnedMatrix`), as on the
    CPU sparse path.
  - a validation matrix on the device: `train_gpu_sparse_with_valid`
    scores the validation set on the host by the same row walk
    `train_sparse_with_valid` uses.
The device path's crossover against the CPU sparse trainer is unmeasured
(docs/GPU_SPARSE_CATEGORICAL_DESIGN.md section 10), which is why
`device_policy` never selects it under `auto`; it runs on an explicit
`device='gpu'`, or on `MOJOTREES_AUTO_MIN_CELLS`, which exists to run that
benchmark.
"""

from std.math import log
from std.sys import has_accelerator

from .bagging import BaggingParams, bagging_enabled, check_bagging, refresh_bag
from .boosting import (
    Booster,
    BoosterParams,
    MulticlassBooster,
    _base_score,
    _check_class_bagging,
    _check_goss,
    _check_objective,
    _check_sample_weight,
    _clamp_prob,
    _fill_grad_hess,
    _fill_softmax_grad_hess,
    _mean_loss,
    _multiclass_goss_select,
    _softmax_inplace,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
)
from .boosting_sparse import _add_tree_scores, _renew_leaf_values_sparse
from .categorical import cat_empty
from .efb import check_bundling_honored
from .goss import GossParams, GossSelection, apply_goss_scaling, goss_round
from .gpu_categorical import CatSetPool, apply_categorical_split_pooled
from .gpu_sparse import GpuSparseHistogramBuilder
from .growth_policy import GrowthSchedule, LeafCandidate, check_grow_policy
from .histogram import (
    Histogram,
    check_device_derivative_precision,
    subtract_histogram,
)
from .interaction import extend_branch
from .monotone import (
    MONOTONE_FREE,
    OutputBounds,
    child_bounds,
    midpoint,
    monotone_sign,
)
from .sampling import (
    ClassBaggingParams,
    check_feature_fractions,
    has_positive_rows,
    refresh_class_bag,
    select_split_features,
    select_tree_features,
)
from .sparse import SparseBinnedMatrix
from .split import SplitInfo
from .train_gpu import _count_left
from .tree import Tree, TreeParams, _leaf_value, _search
from .tree_sparse import SparseBundling, SparseTreeResult, predict_row_sparse


# One node id per tree node plus the builder's reserved id. Tree nodes are
# numbered densely from 0 as `Tree._add_node` hands them out, so a tree with
# `num_leaves` leaves uses ids `0 .. 2 * num_leaves - 2` and the reserved id
# is `2 * num_leaves - 1`.
def _max_nodes_for(num_leaves: Int) -> Int:
    return 2 * num_leaves if num_leaves > 1 else 2


struct _GpuSparseLeaf(Movable):
    """A grown-but-unsplit leaf. Mirrors `tree_sparse._SparseLeafState`
    without the row list and the entry ranges, which the device holds."""

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


def _refuse_bundling(
    bundling: SparseBundling, params: BoosterParams, trainer: String
) raises:
    """The sparse GPU path grows on the unbundled matrix only (module
    docstring). An active plan is refused by name; an `enable_bundle` the
    caller set without preparing a plan is refused by
    `check_bundling_honored`, exactly as the other non-bundling trainers
    refuse it."""
    if bundling.resolved() and bundling.active:
        raise Error(
            "exclusive feature bundling is not available on the sparse GPU"
            " path (",
            trainer,
            "); fit with enable_bundle=False, or on the CPU",
        )
    check_bundling_honored(params.bundling, trainer)


def _open_builder(
    data: SparseBinnedMatrix, num_leaves: Int
) raises -> GpuSparseHistogramBuilder:
    """One builder for a training session, sized for the trees it will
    grow, and the capability check the design reserves for an explicit
    sparse `device='gpu'` request: the builder records what the device path
    can do with this dataset, and a record that says the primitives run but
    training is not wired is a wiring error here, not a fallback."""
    var builder = GpuSparseHistogramBuilder(data, _max_nodes_for(num_leaves))
    if not builder.capability.training:
        raise Error(builder.capability.explain())
    return builder^


def grow_tree_gpu_sparse(
    mut builder: GpuSparseHistogramBuilder,
    mut pool: CatSetPool,
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
) raises -> SparseTreeResult:
    """Grow one tree on sparse data with the device builder.

    Same arguments and semantics as `grow_tree_sparse` less `bundling`,
    which this grower does not take (module docstring), and the same
    growth: leaf-wise by default, depth-wise under `GROW_DEPTHWISE`, the
    subtraction trick for the larger child, and every `TreeParams` field
    including the `extra` bundle applied by the grower. `builder` must have
    been constructed on `data`; `pool` on the builder's context. Gradients
    are uploaded here, once per tree, so a caller growing several trees on
    one round's gradients (the multiclass loop grows one per class) pays one
    upload per tree, as the CPU grower pays one accumulation per tree.

    Returns the tree and, per training row, the leaf it landed in (-1
    outside the bag), read back from the device row permutation once at
    the end of the tree.
    """
    if len(grad) != data.n_rows or len(hess) != data.n_rows:
        raise Error("gradient/hessian length must equal n_rows")
    if builder.n_rows != data.n_rows or builder.n_features != data.n_features:
        raise Error("builder was constructed on a different matrix")
    if builder.max_nodes < _max_nodes_for(params.num_leaves):
        raise Error("builder holds too few node ids for num_leaves")
    var n_features = data.n_features

    check_grow_policy(params.grow_policy)
    params.constraints.check_features(n_features)
    params.monotone.check_features(n_features)
    check_feature_fractions(
        params.feature_fraction,
        params.feature_fraction_bynode,
        params.feature_fraction_bylevel,
    )
    params.extra.check(
        n_features,
        params.num_leaves,
        params.max_depth,
        params.min_data_in_leaf,
    )
    # Refused, not ignored: the device has no Float64 to carry a derivative
    # in. See `histogram.check_device_derivative_precision`.
    check_device_derivative_precision(
        params.extra.wants_float64_derivatives()
    )
    var max_delta_step = params.extra.max_delta_step
    var path_smooth = params.extra.path_smooth
    var signs = params.monotone.active_signs()
    var tree_features = select_tree_features(
        n_features,
        params.feature_fraction,
        params.feature_fraction_seed,
        tree_index,
    )
    # Any accumulated feature answers a node's totals, since every row
    # occupies exactly one bin of every feature (the default bin, for a row
    # with no stored entry). It has to be an accumulated one; the rest are
    # left at zero by `set_features`.
    var value_feature = tree_features[0]

    builder.upload_gradients(grad, hess)
    builder.begin_tree(bag)
    builder.set_features(tree_features)
    pool.clear()

    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )
    var n_root = len(bag) if len(bag) > 0 else data.n_rows
    var root_hist = builder.build_leaf(0)
    var root_branch = List[Int]()
    var root = tree._add_node(
        _leaf_value(
            root_hist,
            params.lambda_reg,
            params.lambda_l1,
            value_feature,
            n_root,
            0.0,
            max_delta_step,
            path_smooth,
        ),
        Float64(n_root),
    )
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
            0,
        ),
        depth=0,
        missing_bins=data.missing_bin,
        monotone=signs,
        cats=data.cats,
        node=root,
        tree_index=tree_index,
        parent_output=tree.value[root],
        grower_applies_extra=True,
    )

    var frontier = List[_GpuSparseLeaf]()
    frontier.append(
        _GpuSparseLeaf(
            root, n_root, root_hist^, root_split^, root_branch^, depth=0
        )
    )
    var n_leaves = 1
    var schedule = GrowthSchedule(params.grow_policy)

    while n_leaves < params.num_leaves:
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
            data.missing_bin[split.feature]
        )
        # Exact integers off the parent histogram, the same counts the CPU
        # grower's row lists would give, so the device partition enqueues
        # without a synchronization.
        var n_left = _count_left(
            frontier[best_i].hist, split, split_missing_bin
        )
        var n_right = frontier[best_i].n_rows - n_left

        var left_node = tree._add_node(0.0, Float64(n_left))
        var right_node = tree._add_node(0.0, Float64(n_right))
        if split.is_categorical:
            var offset = pool.push(
                split.cat_bitset, data.cats.n_categories(split.feature)
            )
            pool.upload()
            apply_categorical_split_pooled(
                builder,
                pool,
                offset,
                split.feature,
                parent_node,
                left_node,
                right_node,
                expected_left=n_left,
            )
        else:
            builder.apply_split(
                split.feature,
                split.bin,
                parent_node,
                left_node,
                right_node,
                split_missing_bin,
                split.default_left,
                False,
                cat_empty(),
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
            missing_bins=data.missing_bin,
            monotone=signs,
            bounds=children.left.copy(),
            cats=data.cats,
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
            missing_bins=data.missing_bin,
            monotone=signs,
            bounds=children.right.copy(),
            cats=data.cats,
            node=right_node,
            tree_index=tree_index,
            parent_output=right_value,
            grower_applies_extra=True,
        )

        frontier[best_i] = _GpuSparseLeaf(
            left_node,
            n_left,
            left_hist^,
            left_split^,
            branch.copy(),
            depth=child_depth,
            bounds=children.left.copy(),
        )
        frontier.append(
            _GpuSparseLeaf(
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

    # One readback of the row permutation per tree: every live leaf's range
    # names its rows in compacted order, and a row outside every range (out
    # of bag) keeps -1.
    var perm = builder.rows.download_rows()
    var row_leaf = List[Int](capacity=data.n_rows)
    row_leaf.resize(data.n_rows, -1)
    for i in range(len(frontier)):
        var rng = builder.rows.range_of(frontier[i].node)
        for j in range(rng.begin, rng.end):
            row_leaf[Int(perm[j])] = frontier[i].node
    return SparseTreeResult(tree^, row_leaf^)


def train_gpu_sparse(
    data: SparseBinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    init_score: List[Float64] = [],
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
    bundling: SparseBundling = SparseBundling.none(),
) raises -> Booster:
    """`train_sparse` with tree growth on the GPU. Same arguments and
    semantics, including `init_score` and `class_bagging`; requires an
    accelerator at runtime and refuses bundling (module docstring). Bags
    and GOSS samples come from the same samplers and schedules as on the
    CPU, so both backends grow round i on exactly the same rows.
    """
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(target) != data.n_rows:
            raise Error("target length must equal n_rows")
        data.validate()
        _refuse_bundling(bundling, params, "train_gpu_sparse")
        var n_features = data.n_features
        _check_objective(objective, target, alpha)
        _check_sample_weight(sample_weight, data.n_rows)
        check_bagging(bagging)
        _check_goss(goss, bagging)
        _check_class_bagging(class_bagging, bagging, goss, objective)
        params.tree.monotone.check_features(n_features)
        if len(init_score) != 0 and len(init_score) != data.n_rows:
            raise Error("init_score length must equal n_rows")

        var builder = _open_builder(data, params.tree.num_leaves)
        var pool = CatSetPool(builder.ctx, params.tree.num_leaves)

        var n = data.n_rows
        var raw = List[Float64](capacity=n)
        var base_score = 0.0
        if len(init_score) == n:
            for r in range(n):
                raw.append(init_score[r])
        else:
            base_score = _base_score(target, objective, sample_weight, alpha)
            for _ in range(n):
                raw.append(base_score)

        var signs = params.tree.monotone.active_signs()
        var renews = objective_renews_leaves(objective)
        var renew_w = renewal_weights(objective, target, sample_weight)
        var renew_a = renewal_alpha(objective, alpha)
        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        var bag = List[Int]()
        var balanced = class_bagging.enabled() and has_positive_rows(target)
        for i in range(params.n_estimators):
            if balanced:
                refresh_class_bag(bag, class_bagging, target, i)
            else:
                refresh_bag(bag, bagging, n, i)
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, grad, hess
            )
            goss_round(bag, grad, hess, goss, i, params.learning_rate)
            var grown = grow_tree_gpu_sparse(
                builder, pool, data, grad, hess, params.tree, bag, i
            )
            if renews:
                _renew_leaf_values_sparse(
                    grown.tree,
                    grown.row_leaf,
                    target,
                    raw,
                    renew_w,
                    renew_a,
                    signs,
                    params.tree.extra,
                )

            if grown.tree.n_leaves == 1 and abs(grown.tree.value[0]) < 1e-12:
                if bagging_enabled(bagging) or goss.enabled or balanced:
                    continue
                break

            _add_tree_scores(
                raw, params.learning_rate, grown.tree, grown.row_leaf, data
            )
            trees.append(grown.tree.copy())

        return Booster(
            trees^,
            base_score,
            params.learning_rate,
            objective,
            params.tree.monotone.copy(),
        )


def train_gpu_sparse_with_valid(
    data: SparseBinnedMatrix,
    target: List[Float64],
    valid_data: SparseBinnedMatrix,
    valid_target: List[Float64],
    objective: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
    bundling: SparseBundling = SparseBundling.none(),
) raises -> Booster:
    """`train_sparse_with_valid` with tree growth on the GPU. The
    validation matrix stays on the host and is scored by the same row walk;
    the early-stopping rule, the returned tree count, and the loss it is
    judged by are the CPU trainer's."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(target) != data.n_rows:
            raise Error("target length must equal n_rows")
        if len(valid_target) != valid_data.n_rows:
            raise Error("valid_target length must equal valid n_rows")
        data.validate()
        valid_data.validate()
        _refuse_bundling(bundling, params, "train_gpu_sparse_with_valid")
        var n_features = data.n_features
        if valid_data.n_features != n_features:
            raise Error("valid_data must have the same features")
        if valid_data.n_bins != data.n_bins:
            raise Error("valid_data must be binned with the same bin count")
        _check_objective(objective, target, alpha)
        if early_stopping_rounds < 1:
            raise Error("early_stopping_rounds must be positive")
        _check_sample_weight(sample_weight, data.n_rows)
        check_bagging(bagging)
        _check_goss(goss, bagging)
        _check_class_bagging(class_bagging, bagging, goss, objective)
        params.tree.monotone.check_features(n_features)

        var builder = _open_builder(data, params.tree.num_leaves)
        var pool = CatSetPool(builder.ctx, params.tree.num_leaves)

        var n = data.n_rows
        var n_valid = valid_data.n_rows
        var valid_rows = valid_data.to_rows()
        var base_score = _base_score(target, objective, sample_weight, alpha)
        var raw = List[Float64](capacity=n)
        for _ in range(n):
            raw.append(base_score)
        var valid_raw = List[Float64](capacity=n_valid)
        for _ in range(n_valid):
            valid_raw.append(base_score)

        var signs = params.tree.monotone.active_signs()
        var renews = objective_renews_leaves(objective)
        var renew_w = renewal_weights(objective, target, sample_weight)
        var renew_a = renewal_alpha(objective, alpha)
        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        var best_loss = _mean_loss(valid_raw, valid_target, objective, alpha)
        var best_n_trees = 0
        var bag = List[Int]()
        var balanced = class_bagging.enabled() and has_positive_rows(target)
        for i in range(params.n_estimators):
            if balanced:
                refresh_class_bag(bag, class_bagging, target, i)
            else:
                refresh_bag(bag, bagging, n, i)
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, grad, hess
            )
            goss_round(bag, grad, hess, goss, i, params.learning_rate)
            var grown = grow_tree_gpu_sparse(
                builder, pool, data, grad, hess, params.tree, bag, i
            )
            if renews:
                _renew_leaf_values_sparse(
                    grown.tree,
                    grown.row_leaf,
                    target,
                    raw,
                    renew_w,
                    renew_a,
                    signs,
                    params.tree.extra,
                )
            if grown.tree.n_leaves == 1 and abs(grown.tree.value[0]) < 1e-12:
                if bagging_enabled(bagging) or goss.enabled or balanced:
                    continue
                break

            _add_tree_scores(
                raw, params.learning_rate, grown.tree, grown.row_leaf, data
            )
            for r in range(n_valid):
                valid_raw[r] += (
                    params.learning_rate
                    * predict_row_sparse(grown.tree, valid_rows, r)
                )
            trees.append(grown.tree.copy())

            var loss = _mean_loss(valid_raw, valid_target, objective, alpha)
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


def train_multiclass_gpu_sparse(
    data: SparseBinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    bundling: SparseBundling = SparseBundling.none(),
) raises -> MulticlassBooster:
    """`train_multiclass_sparse` with tree growth on the GPU: one tree per
    class per round, every class's tree grown by the same builder on the
    same rows, the shared GOSS sample drawn once per round as on the CPU."""
    comptime if not has_accelerator():
        raise Error("GPU training requires an accelerator")
    else:
        if len(labels) != data.n_rows:
            raise Error("labels length must equal n_rows")
        if n_classes < 2:
            raise Error("n_classes must be at least 2")
        data.validate()
        _refuse_bundling(bundling, params, "train_multiclass_gpu_sparse")
        _check_sample_weight(sample_weight, data.n_rows)
        check_bagging(bagging)
        _check_goss(goss, bagging)
        params.tree.monotone.check_features(data.n_features)
        var n = data.n_rows

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

        var builder = _open_builder(data, params.tree.num_leaves)
        var pool = CatSetPool(builder.ctx, params.tree.num_leaves)

        var raw = List[Float64](capacity=n * n_classes)
        for _ in range(n):
            for k in range(n_classes):
                raw.append(base_scores[k])
        var prob = List[Float64](capacity=n * n_classes)
        for _ in range(n * n_classes):
            prob.append(0.0)

        var trees = List[Tree]()
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        var bag = List[Int]()
        for i in range(params.n_estimators):
            refresh_bag(bag, bagging, n, i)
            for r in range(n):
                for k in range(n_classes):
                    prob[r * n_classes + k] = raw[r * n_classes + k]
                _softmax_inplace(prob, r * n_classes, n_classes)

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
                var grown = grow_tree_gpu_sparse(
                    builder,
                    pool,
                    data,
                    grad,
                    hess,
                    params.tree,
                    bag,
                    i * n_classes + k,
                )
                if grown.tree.n_leaves > 1 or abs(grown.tree.value[0]) >= 1e-12:
                    made_progress = True
                _add_tree_scores(
                    raw,
                    params.learning_rate,
                    grown.tree,
                    grown.row_leaf,
                    data,
                    SparseBundling.none(),
                    n_classes,
                    k,
                )
                trees.append(grown.tree.copy())

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
