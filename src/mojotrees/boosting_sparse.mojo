"""Gradient boosting on sparse data.

The objective layer is shared with `boosting.mojo` verbatim: base scores,
gradients and hessians, losses, sample weights, LightGBM leaf renewal
(QUANTILE, L1, and MAPE, via `objective_renews_leaves`), bagging, and GOSS
all come from there, so a sparse fit
differs from a dense one only in how histograms are accumulated. Only the
loop skeleton is mirrored here, the way `train_gpu.mojo` mirrors it for the
GPU path; the dense loop is left untouched.

Two sparse-specific shortcuts:

- `grow_tree_sparse` hands back the row-to-leaf assignment, so the
  per-round raw-score update is one lookup per row instead of a tree walk.
  Rows outside a bag come back as -1 and fall back to a walk.
- leaf renewal groups residuals by that same assignment.

What the sparse path does and does not accept
---------------------------------------------
Available and shared with the dense path: every built-in objective, sample
weights, early stopping, uniform bagging, LightGBM's balanced
(class-conditional) bagging, GOSS in both the single-output and the softmax
form, categorical features (`fit_categorical_spec_csc` and `transform_csc`
fit and bin them, and `tree._search` searches them through `data.cats`, so a
sparse-grown categorical split is the dense one), missing values, monotonic
and interaction constraints, feature subsampling, and every `TreeParams`
field including the whole `extra` bundle -- `min_gain_to_split`,
`max_delta_step`, `path_smooth`, `extra_trees`, the monotone penalty, and the
feature penalties all reach `grow_tree_sparse`, which opts into
`grower_applies_extra` and applies the leaf-finishing half itself, renewal
included.

Exclusive feature bundling is available too, through
`prepare_bundling_csc`: it fits a plan with `efb.fit_bundles`, applies it
with `efb.bundle_csc`, and hands back the bundled matrix under the
`SparseBundling` view that reads it. The trainers take that view and pass it
straight to `grow_tree_sparse`; the trees that come out name original
features and original bins, so nothing downstream of training can tell a
bundled fit from an unbundled one. A caller that sets
`BoosterParams.bundling` but hands over a plain binned matrix gets an error
rather than an unbundled fit, which is what `efb.check_bundling_honored` is
for.

Not available: custom objectives (`objective.mojo` takes a `BinnedMatrix`)
and the GPU backend. The sparse GPU primitives exist -- see `gpu_sparse.mojo`
and `gpu_categorical.mojo` -- but no training path is wired to them, and
`gpu_sparse_layout.sparse_gpu_capability` is the record that says so rather
than a trainer that silently falls back.

Every entry point here validates its matrix once, through
`SparseBinnedMatrix.validate`, before the first histogram: a structurally
broken matrix or a categorical column whose default bin is not category 0's
would otherwise produce a fitted model rather than an error.
"""

from std.math import exp, log

from .bagging import (
    BaggingParams,
    bagging_enabled,
    check_bagging,
    refresh_bag,
)
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
    _multiclass_mean_loss,
    _percentile,
    _softmax_inplace,
    _weighted_percentile,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
)
from .binning import BinMapper
from .efb import (
    EfbSettings,
    FeatureBundling,
    bundle_csc,
    check_bundling_honored,
    fit_bundles,
)
from .goss import GossParams, GossSelection, apply_goss_scaling, goss_round
from .sampling import (
    ClassBaggingParams,
    has_positive_rows,
    refresh_class_bag,
)
from .sparse import SparseBinnedMatrix, SparseBinnedRows
from .tree import Tree, node_bounds
from .tree_parameters_extra import ExtraTreeParams, finish_leaf_output
from .tree_sparse import (
    SparseBundling,
    grow_tree_sparse,
    predict_row_sparse,
    predict_row_sparse_csc,
)


struct SparseBundledData(Movable):
    """A binned sparse matrix together with the bundling view that reads it.

    The sparse counterpart of `efb.BundledMatrix`, and it holds *one* matrix
    where that holds two: the dense grower keeps the original matrix beside
    the bundled one to partition rows on, and the sparse grower cannot,
    because a second CSC copy is exactly the memory bundling saves. So the
    matrix in here is the one the trainer sees -- bundled when the plan is
    active, the original when it is not -- and `bundling` is what turns a
    bundle column back into an original feature (see `SparseBundling`).

    Training-time only. Nothing here reaches the fitted model: a tree grown
    through this view names original features and original bins, so the
    `Booster` that comes out is byte-for-byte the one an unbundled fit
    produces and serializes with no new fields.
    """

    var data: SparseBinnedMatrix
    var bundling: SparseBundling

    def __init__(
        out self,
        var data: SparseBinnedMatrix,
        var bundling: SparseBundling,
    ):
        self.data = data^
        self.bundling = bundling^

    def active(self) -> Bool:
        """Whether a plan was fitted *and* applied."""
        return self.bundling.active


def prepare_bundling_csc(
    mapper: BinMapper,
    var data: SparseBinnedMatrix,
    settings: EfbSettings = EfbSettings.disabled(),
) raises -> SparseBundledData:
    """Fit and apply a bundling plan to a binned sparse matrix, or decide
    not to.

    The sparse mirror of `efb.prepare_bundling`, with the same three ways to
    decline and the same conservative meaning for declining:

    - `settings.enabled` is False, which is the default;
    - the matrix has no rows or fewer than two features to bundle;
    - `fit_bundles` packs nothing, or the packing does not clear
      `EfbParams.min_reduction`, so it reports `use_bundling = False`.

    Each of those returns the caller's own matrix under an inactive view,
    which grows exactly the trees it grew before bundling existed.

    `data` is taken by value and either handed straight back or replaced by
    the bundled matrix, so declining costs no copy and accepting frees the
    original as soon as `bundle_csc` has read it -- the two matrices are
    never both alive after this returns.

    The plan comes from the *mapper*'s bin widths rather than the matrix's
    (see `fit_bundles`), and is fitted once per training call rather than
    once per tree: it is a deterministic function of the matrix and the
    parameters, so a continued run rebuilds the plan the first call used.
    """
    data.validate()
    if not settings.enabled:
        var view = SparseBundling.of(FeatureBundling.none(), data)
        return SparseBundledData(data^, view^)
    settings.check()
    if mapper.n_features != data.n_features:
        raise Error("mapper and matrix must agree on n_features")
    if data.n_rows < 1 or data.n_features < 2:
        var view = SparseBundling.of(FeatureBundling.none(), data)
        return SparseBundledData(data^, view^)
    var plan = fit_bundles(mapper, data, settings.params)
    if not plan.use_bundling:
        var view = SparseBundling.of(FeatureBundling.none(), data)
        return SparseBundledData(data^, view^)
    var bundled = bundle_csc(data, plan)
    # The bundled matrix is what every later histogram is read from, so it is
    # validated here rather than trusted: `bundle_csc` is the one place a
    # structurally sound matrix could be turned into an unsound one.
    bundled.validate()
    var view = SparseBundling.of(plan, bundled)
    return SparseBundledData(bundled^, view^)


def _resolve_bundling(
    data: SparseBinnedMatrix,
    bundling: SparseBundling,
    settings: EfbSettings,
    trainer: String,
) raises -> SparseBundling:
    """The bundling view a sparse trainer grows under.

    A resolved view (from `prepare_bundling_csc`) is checked against the
    matrix and used. An unresolved one means the caller never prepared a
    plan, so this is also where an *unhonored* `enable_bundle` is caught:
    without it, setting the switch and passing the plain binned matrix would
    silently produce an unbundled fit that looks exactly like a bundled one.
    """
    if bundling.resolved():
        bundling.check_matrix(data)
        return bundling.copy()
    check_bundling_honored(settings, trainer)
    return SparseBundling.of(FeatureBundling.none(), data)


def _renew_leaf_values_sparse(
    mut tree: Tree,
    row_leaf: List[Int],
    target: List[Float64],
    raw: List[Float64],
    weights: List[Float64],
    alpha: Float64,
    monotone: List[Int] = [],
    extra: ExtraTreeParams = ExtraTreeParams(),
) raises:
    """LightGBM's RenewTreeOutput over the assignment `grow_tree_sparse`
    already produced: replace each leaf's Newton value with the
    alpha-percentile of the residuals of the rows in it. Rows marked -1 are
    outside the bag the tree was grown on and are skipped, which is what the
    dense path does by renewing over the bag.

    A non-empty `monotone` clamps every renewed value back into its leaf's
    monotone interval, exactly as `_renew_leaf_values` does on the dense
    path: renewal without the clamp would discard the constraint the tree
    was grown under (see monotone.mojo).

    `extra` carries `max_delta_step` and `path_smooth`, applied to a renewed
    value as they are to a grown one and in the same order (cap and smooth
    first, monotone interval on the result). Without this the three renewing
    objectives would be the only ones to escape the cap the caller asked for.
    An internal node still holds the finished value it was grown with, since
    renewal rewrites leaves only, so it is the parent output its children
    smooth toward."""
    var n_nodes = len(tree.feature)
    var bounds = node_bounds(tree, monotone)
    var finish = extra.needs_leaf_finish()
    var parent_output = List[Float64]()
    if finish:
        parent_output.resize(n_nodes, 0.0)
        for node in range(n_nodes):
            if tree.feature[node] < 0:
                continue
            parent_output[tree.left[node]] = tree.value[node]
            parent_output[tree.right[node]] = tree.value[node]
    var leaf_residuals = List[List[Float64]]()
    var leaf_weights = List[List[Float64]]()
    for _ in range(n_nodes):
        leaf_residuals.append(List[Float64]())
        leaf_weights.append(List[Float64]())
    for r in range(len(row_leaf)):
        var node = row_leaf[r]
        if node < 0:
            continue
        leaf_residuals[node].append(target[r] - raw[r])
        if len(weights) > 0:
            leaf_weights[node].append(weights[r])
    for node in range(n_nodes):
        if tree.feature[node] >= 0 or len(leaf_residuals[node]) == 0:
            continue
        var renewed: Float64
        if len(weights) > 0:
            renewed = _weighted_percentile(
                leaf_residuals[node], leaf_weights[node], alpha
            )
        else:
            renewed = _percentile(leaf_residuals[node], alpha)
        if finish:
            renewed = finish_leaf_output(
                renewed,
                extra.max_delta_step,
                extra.path_smooth,
                len(leaf_residuals[node]),
                parent_output[node],
            )
        if len(bounds) > 0:
            renewed = bounds[node].clamp(renewed)
        tree.value[node] = renewed


def _add_tree_scores(
    mut raw: List[Float64],
    learning_rate: Float64,
    tree: Tree,
    row_leaf: List[Int],
    data: SparseBinnedMatrix,
    bundling: SparseBundling = SparseBundling.none(),
    stride: Int = 1,
    offset: Int = 0,
) raises:
    """Add this tree's shrunken output to every row's raw score.

    A row the tree was grown on is looked up in `row_leaf` directly; a row
    outside the bag (-1) is walked through the tree instead, which is what
    the dense loop does for every row. `bundling` is what that walk needs
    when `data` is a bundled matrix, since the tree names original features
    and the matrix holds bundle columns."""
    for r in range(len(row_leaf)):
        var node = row_leaf[r]
        var value: Float64
        if node >= 0:
            value = tree.value[node]
        else:
            value = predict_row_sparse_csc(tree, data, r, bundling)
        raw[r * stride + offset] += learning_rate * value


def train_sparse(
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
    """Sparse counterpart of `train`, with identical arguments and
    semantics. The returned `Booster` is an ordinary one: it serializes,
    loads, and predicts on dense rows exactly like a densely trained
    model.

    A non-empty `init_score` starts boosting from those raw scores instead
    of from the objective's own base score, exactly as in `train`: the
    returned ensemble has a base score of 0 and predicts the trees alone.

    `class_bagging` is LightGBM's balanced bagging, as in `train`: it
    replaces the uniform bag with one drawn per label class, produces the
    same shape (one ascending, duplicate-free row list), and so reaches
    `grow_tree_sparse` exactly as a uniform bag does. `_check_class_bagging`
    is what makes it exclusive with `bagging` and with `goss`, and what
    restricts it to binary classification.

    `bundling` is an exclusive-feature-bundling view from
    `prepare_bundling_csc`; with an active one, `data` is the bundled matrix
    it produced. Everything the caller passes and everything that comes back
    stays in the original feature space, so a bundled fit differs from an
    unbundled one only in how histograms are laid out. Leaving it at
    `none()` while `params.bundling.enabled` is set is an error rather than
    a silent unbundled fit -- see `_resolve_bundling`."""
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    data.validate()
    var bundles = _resolve_bundling(
        data, bundling, params.bundling, "train_sparse"
    )
    var n_features = bundles.n_features
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    _check_class_bagging(class_bagging, bagging, goss, objective)
    params.tree.monotone.check_features(n_features)
    if len(init_score) != 0 and len(init_score) != data.n_rows:
        raise Error("init_score length must equal n_rows")

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
    # LightGBM turns balanced bagging on only when the dataset holds a
    # positive row, and falls back to plain `bagging_fraction` when it does
    # not. One label sweep, hoisted out of the round loop, exactly as in
    # `boosting._boost_rounds`.
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
        var grown = grow_tree_sparse(
            data, grad, hess, params.tree, bag, i, bundles
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
            raw, params.learning_rate, grown.tree, grown.row_leaf, data,
            bundles,
        )
        trees.append(grown.tree.copy())

    return Booster(
        trees^,
        base_score,
        params.learning_rate,
        objective,
        params.tree.monotone.copy(),
    )


def train_sparse_with_valid(
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
    """Sparse counterpart of `train_with_valid`. The validation matrix is
    turned into a row view once, so scoring it each round costs one binary
    search per node over that row's own entries.

    `class_bagging` carries the meaning it has in `train_sparse`; the
    validation loss is untouched by it, exactly as on the dense path.

    `bundling` also carries its `train_sparse` meaning, and it applies to the
    *training* matrix alone: `valid_data` stays unbundled, because a grown
    tree names original features and the validation walk therefore reads it
    directly. That is also why the two shape checks below compare the
    validation matrix against the original feature count and the original
    bin width rather than against the bundled matrix's own, which are the
    bundle count and the widest bundle."""
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if len(valid_target) != valid_data.n_rows:
        raise Error("valid_target length must equal valid n_rows")
    data.validate()
    valid_data.validate()
    var bundles = _resolve_bundling(
        data, bundling, params.bundling, "train_sparse_with_valid"
    )
    var n_features = bundles.n_features
    if valid_data.n_features != n_features:
        raise Error("valid_data must have the same features")
    if valid_data.n_bins != bundles.source_bins(data):
        raise Error("valid_data must be binned with the same bin count")
    _check_objective(objective, target, alpha)
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    _check_class_bagging(class_bagging, bagging, goss, objective)
    params.tree.monotone.check_features(n_features)

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
        var grown = grow_tree_sparse(
            data, grad, hess, params.tree, bag, i, bundles
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
            raw, params.learning_rate, grown.tree, grown.row_leaf, data,
            bundles,
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


def train_multiclass_sparse(
    data: SparseBinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    bundling: SparseBundling = SparseBundling.none(),
) raises -> MulticlassBooster:
    """Sparse counterpart of `train_multiclass`, with identical arguments
    and semantics.

    `goss` draws one gradient-based sample per round, shared by every class's
    tree in that round, through the same `_multiclass_goss_select` and
    `apply_goss_scaling` the dense multiclass loop uses -- so the sample, the
    scaling, and the round-skipping rule are one implementation, not two.

    `bundling` carries its `train_sparse` meaning, and one plan is shared by
    every class's tree in every round, as on the dense multiclass path."""
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    data.validate()
    var bundles = _resolve_bundling(
        data, bundling, params.bundling, "train_multiclass_sparse"
    )
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    params.tree.monotone.check_features(bundles.n_features)
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

        # One shared sample for the whole round, drawn before any class's
        # tree so that every class is grown on the same rows.
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
            var grown = grow_tree_sparse(
                data,
                grad,
                hess,
                params.tree,
                bag,
                i * n_classes + k,
                bundles,
            )
            if grown.tree.n_leaves > 1 or abs(grown.tree.value[0]) >= 1e-12:
                made_progress = True
            _add_tree_scores(
                raw,
                params.learning_rate,
                grown.tree,
                grown.row_leaf,
                data,
                bundles,
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

    # The constraints every per-class tree was grown under travel with the
    # ensemble, as they do on the dense path: a sparse-trained multiclass
    # model that dropped them would serialize as an unconstrained one and
    # `train_multiclass_more` would then accept a different constraint
    # vector for its continued rounds.
    return MulticlassBooster(
        trees^,
        base_scores^,
        n_classes,
        params.learning_rate,
        params.tree.monotone.copy(),
    )
