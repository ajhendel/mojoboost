"""Gradient boosting on sparse data.

The objective layer is shared with `boosting.mojo` verbatim: base scores,
gradients and hessians, losses, sample weights, LightGBM leaf renewal for
QUANTILE and L1, bagging, and GOSS all come from there, so a sparse fit
differs from a dense one only in how histograms are accumulated. Only the
loop skeleton is mirrored here, the way `train_gpu.mojo` mirrors it for the
GPU path; the dense loop is left untouched.

Two sparse-specific shortcuts:

- `grow_tree_sparse` hands back the row-to-leaf assignment, so the
  per-round raw-score update is one lookup per row instead of a tree walk.
  Rows outside a bag come back as -1 and fall back to a walk.
- leaf renewal groups residuals by that same assignment.

Not available on the sparse path: categorical features (rejected when the
data is binned, see sparse.mojo), custom objectives (`objective.mojo` takes
a `BinnedMatrix`), and the GPU backend. Everything else the dense trainer
accepts -- every built-in objective, sample weights, early stopping,
bagging, GOSS, and every `TreeParams` field -- behaves the same here.
"""

from std.math import exp, log

from .bagging import (
    BaggingParams,
    bagging_enabled,
    check_bagging,
    refresh_bag,
)
from .boosting import (
    L1,
    QUANTILE,
    Booster,
    BoosterParams,
    MulticlassBooster,
    _base_score,
    _check_goss,
    _check_objective,
    _check_sample_weight,
    _clamp_prob,
    _fill_grad_hess,
    _mean_loss,
    _multiclass_mean_loss,
    _percentile,
    _softmax_inplace,
    _weighted_percentile,
)
from .goss import GossParams, goss_round
from .sparse import SparseBinnedMatrix, SparseBinnedRows
from .tree import Tree
from .tree_sparse import (
    grow_tree_sparse,
    predict_row_sparse,
    predict_row_sparse_csc,
)


def _renew_leaf_values_sparse(
    mut tree: Tree,
    row_leaf: List[Int],
    target: List[Float64],
    raw: List[Float64],
    weights: List[Float64],
    alpha: Float64,
) raises:
    """LightGBM's RenewTreeOutput over the assignment `grow_tree_sparse`
    already produced: replace each leaf's Newton value with the
    alpha-percentile of the residuals of the rows in it. Rows marked -1 are
    outside the bag the tree was grown on and are skipped, which is what the
    dense path does by renewing over the bag."""
    var n_nodes = len(tree.feature)
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
        if len(weights) > 0:
            tree.value[node] = _weighted_percentile(
                leaf_residuals[node], leaf_weights[node], alpha
            )
        else:
            tree.value[node] = _percentile(leaf_residuals[node], alpha)


def _add_tree_scores(
    mut raw: List[Float64],
    learning_rate: Float64,
    tree: Tree,
    row_leaf: List[Int],
    data: SparseBinnedMatrix,
    stride: Int = 1,
    offset: Int = 0,
) raises:
    """Add this tree's shrunken output to every row's raw score.

    A row the tree was grown on is looked up in `row_leaf` directly; a row
    outside the bag (-1) is walked through the tree instead, which is what
    the dense loop does for every row."""
    for r in range(len(row_leaf)):
        var node = row_leaf[r]
        var value: Float64
        if node >= 0:
            value = tree.value[node]
        else:
            value = predict_row_sparse_csc(tree, data, r)
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
) raises -> Booster:
    """Sparse counterpart of `train`, with identical arguments and
    semantics. The returned `Booster` is an ordinary one: it serializes,
    loads, and predicts on dense rows exactly like a densely trained
    model."""
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)

    var n = data.n_rows
    var base_score = _base_score(target, objective, sample_weight, alpha)
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    for i in range(params.n_estimators):
        refresh_bag(bag, bagging, n, i)
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess
        )
        goss_round(bag, grad, hess, goss, i, params.learning_rate)
        var grown = grow_tree_sparse(data, grad, hess, params.tree, bag, i)
        if objective == QUANTILE:
            _renew_leaf_values_sparse(
                grown.tree, grown.row_leaf, target, raw, sample_weight, alpha
            )
        elif objective == L1:
            _renew_leaf_values_sparse(
                grown.tree, grown.row_leaf, target, raw, sample_weight, 0.5
            )

        if grown.tree.n_leaves == 1 and abs(grown.tree.value[0]) < 1e-12:
            if bagging_enabled(bagging) or goss.enabled:
                continue
            break

        _add_tree_scores(
            raw, params.learning_rate, grown.tree, grown.row_leaf, data
        )
        trees.append(grown.tree.copy())

    return Booster(trees^, base_score, params.learning_rate, objective)


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
) raises -> Booster:
    """Sparse counterpart of `train_with_valid`. The validation matrix is
    turned into a row view once, so scoring it each round costs one binary
    search per node over that row's own entries."""
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

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var best_loss = _mean_loss(valid_raw, valid_target, objective, alpha)
    var best_n_trees = 0
    var bag = List[Int]()
    for i in range(params.n_estimators):
        refresh_bag(bag, bagging, n, i)
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess
        )
        goss_round(bag, grad, hess, goss, i, params.learning_rate)
        var grown = grow_tree_sparse(data, grad, hess, params.tree, bag, i)
        if objective == QUANTILE:
            _renew_leaf_values_sparse(
                grown.tree, grown.row_leaf, target, raw, sample_weight, alpha
            )
        elif objective == L1:
            _renew_leaf_values_sparse(
                grown.tree, grown.row_leaf, target, raw, sample_weight, 0.5
            )
        if grown.tree.n_leaves == 1 and abs(grown.tree.value[0]) < 1e-12:
            if bagging_enabled(bagging) or goss.enabled:
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
    return Booster(trees^, base_score, params.learning_rate, objective)


def train_multiclass_sparse(
    data: SparseBinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> MulticlassBooster:
    """Sparse counterpart of `train_multiclass`, with identical arguments
    and semantics."""
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
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

        var made_progress = False
        for k in range(n_classes):
            grad.clear()
            hess.clear()
            for r in range(n):
                var p = prob[r * n_classes + k]
                var y = 1.0 if labels[r] == k else 0.0
                var w = sample_weight[r] if len(sample_weight) > 0 else 1.0
                grad.append(w * (p - y))
                var h = 2.0 * p * (1.0 - p)
                if h < 1e-16:
                    h = 1e-16
                hess.append(w * h)
            var grown = grow_tree_sparse(
                data, grad, hess, params.tree, bag, i * n_classes + k
            )
            if grown.tree.n_leaves > 1 or abs(grown.tree.value[0]) >= 1e-12:
                made_progress = True
            _add_tree_scores(
                raw,
                params.learning_rate,
                grown.tree,
                grown.row_leaf,
                data,
                n_classes,
                k,
            )
            trees.append(grown.tree.copy())

        if not made_progress:
            for _ in range(n_classes):
                _ = trees.pop()
            if bagging_enabled(bagging):
                continue
            break

    return MulticlassBooster(
        trees^, base_scores^, n_classes, params.learning_rate
    )
