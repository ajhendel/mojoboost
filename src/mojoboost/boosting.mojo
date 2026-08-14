"""Gradient boosting loop.

Trains an additive ensemble of leaf-wise trees on second-order gradients.
Objectives: squared error (regression), binary logistic (labels in {0, 1}),
poisson counts, huber, quantile, L1 (mean absolute error), and multiclass
softmax (labels in 0..n_classes-1, via train_multiclass).

Caller-supplied objectives live in objective.mojo (`train_custom`); a
booster trained that way carries the CUSTOM objective code.

QUANTILE and L1 follow LightGBM's RenewTreeOutput: after each tree is
grown, every leaf's Newton value is replaced by the alpha-percentile
(median for L1) of the residuals of the rows in that leaf, and shrinkage
is applied to the renewed value. `alpha` is the target quantile for
QUANTILE and the transition point of the huber loss for HUBER; the other
objectives ignore it.
"""

from std.math import exp, log

from .binning import BinnedMatrix
from .metrics import _argsort
from .bagging import (
    BaggingParams,
    bagging_enabled,
    check_bagging,
    refresh_bag,
)
from .goss import (
    GossParams,
    GossSelection,
    apply_goss_scaling,
    goss_round,
    goss_select,
)
from .monotone import MonotoneConstraints
from .tree import Tree, TreeParams, grow_tree, node_bounds

comptime SQUARED_ERROR = 0
comptime BINARY_LOGISTIC = 1
comptime POISSON = 2
comptime HUBER = 3
comptime QUANTILE = 4
comptime L1 = 5

# Marks a booster trained through `train_custom` in objective.mojo. It is not
# a built-in objective: `train` and `train_gpu` reject it, because the
# gradients come from a caller-supplied callable rather than from
# `_fill_grad_hess`. Predictions for it are raw scores (no known link).
comptime CUSTOM = 6

# LightGBM's poisson_max_delta_step: the hessian is exp(raw + this), which
# caps the Newton step for rows with tiny predicted means.
comptime _POISSON_MAX_DELTA_STEP = 0.7


def _sign(x: Float64) -> Float64:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def _percentile(values: List[Float64], alpha: Float64) -> Float64:
    """LightGBM's PercentileFun: linear interpolation at position
    (n - 1) * alpha of the ascending sorted values."""
    var n = len(values)
    if n <= 1:
        return values[0]
    var order = _argsort(values)
    var float_pos = Float64(n - 1) * alpha
    var pos = Int(float_pos)
    if pos >= n - 1:
        return values[order[n - 1]]
    var bias = float_pos - Float64(pos)
    var v1 = values[order[pos]]
    var v2 = values[order[pos + 1]]
    return v1 + (v2 - v1) * bias


def _weighted_percentile(
    values: List[Float64], weights: List[Float64], alpha: Float64
) -> Float64:
    """LightGBM's WeightedPercentileFun: stable-sort by value, walk the
    weighted cdf to threshold = alpha * total weight, and interpolate
    between the straddling values only when their cdf gap is at least
    1.0 (otherwise the lower value is returned)."""
    var n = len(values)
    if n <= 1:
        return values[0]
    var order = _argsort(values)
    var cdf = List[Float64](capacity=n)
    var total = 0.0
    for i in range(n):
        total += weights[order[i]]
        cdf.append(total)
    var threshold = total * alpha
    var pos = n
    for i in range(n):
        if cdf[i] > threshold:
            pos = i
            break
    if pos > n - 1:
        pos = n - 1
    if pos == 0 or pos == n - 1:
        return values[order[pos]]
    var v1 = values[order[pos - 1]]
    var v2 = values[order[pos]]
    if cdf[pos] - cdf[pos - 1] >= 1.0:
        return (
            (threshold - cdf[pos - 1]) / (cdf[pos] - cdf[pos - 1]) * (v2 - v1)
            + v1
        )
    return v1


def _sigmoid(x: Float64) -> Float64:
    if x >= 0.0:
        var e = exp(-x)
        return 1.0 / (1.0 + e)
    var e = exp(x)
    return e / (1.0 + e)


def _clamp_prob(p: Float64) -> Float64:
    if p < 1e-15:
        return 1e-15
    if p > 1.0 - 1e-15:
        return 1.0 - 1e-15
    return p


def _check_objective(
    objective: Int, target: List[Float64], alpha: Float64
) raises:
    if objective == CUSTOM:
        raise Error(
            "custom objectives must be trained with train_custom (or"
            " train_custom_with_valid / fit_custom / train_custom_gpu)"
        )
    if (
        objective != SQUARED_ERROR
        and objective != BINARY_LOGISTIC
        and objective != POISSON
        and objective != HUBER
        and objective != QUANTILE
        and objective != L1
    ):
        raise Error("unknown objective")
    if objective == POISSON:
        for r in range(len(target)):
            if target[r] < 0.0:
                raise Error("poisson target values must be nonnegative")
    if objective == HUBER and alpha <= 0.0:
        raise Error("huber requires alpha > 0")
    if objective == QUANTILE and (alpha <= 0.0 or alpha >= 1.0):
        raise Error("quantile requires 0 < alpha < 1")


def _check_sample_weight(weights: List[Float64], n: Int) raises:
    """An empty list means unweighted; otherwise one nonnegative weight
    per training row."""
    if len(weights) == 0:
        return
    if len(weights) != n:
        raise Error("sample_weight length must equal n_rows")
    var total = 0.0
    for r in range(n):
        if weights[r] < 0.0:
            raise Error("sample_weight entries must be nonnegative")
        total += weights[r]
    if total <= 0.0:
        raise Error("sample_weight must have a positive sum")


def _check_goss(goss: GossParams, bagging: BaggingParams) raises:
    """Validate the GOSS rates and reject GOSS together with row bagging.

    Both strategies own the row list a tree is grown on, and LightGBM makes
    them exclusive too: it silently turns bagging off under GOSS. mojoboost
    raises instead, so a configuration that would quietly ignore half of
    itself is reported."""
    goss.validate()
    if goss.enabled and bagging_enabled(bagging):
        raise Error("goss and bagging cannot both be enabled")


def _base_score(
    target: List[Float64],
    objective: Int,
    weights: List[Float64],
    alpha: Float64,
) raises -> Float64:
    # QUANTILE and L1 boost from the target percentile, LightGBM style;
    # every other objective boosts from the (weighted) mean.
    if objective == QUANTILE or objective == L1:
        var q = alpha if objective == QUANTILE else 0.5
        if len(weights) > 0:
            return _weighted_percentile(target, weights, q)
        return _percentile(target, q)
    var mean = 0.0
    var total_w = 0.0
    for r in range(len(target)):
        var w = weights[r] if len(weights) > 0 else 1.0
        mean += w * target[r]
        total_w += w
    if total_w <= 0.0:
        raise Error("sample_weight must have a positive sum")
    mean /= total_w
    if objective == BINARY_LOGISTIC:
        var p = _clamp_prob(mean)
        return log(p / (1.0 - p))
    if objective == POISSON:
        if mean <= 0.0:
            raise Error("poisson requires a positive mean target")
        return log(mean)
    return mean


def _fill_grad_hess(
    raw: List[Float64],
    target: List[Float64],
    objective: Int,
    weights: List[Float64],
    alpha: Float64,
    mut grad: List[Float64],
    mut hess: List[Float64],
):
    grad.clear()
    hess.clear()
    if objective == BINARY_LOGISTIC:
        for r in range(len(target)):
            var w = weights[r] if len(weights) > 0 else 1.0
            var p = _sigmoid(raw[r])
            grad.append(w * (p - target[r]))
            var h = p * (1.0 - p)
            if h < 1e-16:
                h = 1e-16
            hess.append(w * h)
    elif objective == POISSON:
        for r in range(len(target)):
            var w = weights[r] if len(weights) > 0 else 1.0
            var mu = exp(raw[r])
            grad.append(w * (mu - target[r]))
            hess.append(w * exp(raw[r] + _POISSON_MAX_DELTA_STEP))
    elif objective == HUBER:
        for r in range(len(target)):
            var w = weights[r] if len(weights) > 0 else 1.0
            var diff = raw[r] - target[r]
            if abs(diff) <= alpha:
                grad.append(w * diff)
            else:
                grad.append(w * _sign(diff) * alpha)
            hess.append(w)
    elif objective == QUANTILE:
        for r in range(len(target)):
            var w = weights[r] if len(weights) > 0 else 1.0
            var diff = raw[r] - target[r]
            if diff >= 0.0:
                grad.append(w * (1.0 - alpha))
            else:
                grad.append(w * -alpha)
            hess.append(w)
    elif objective == L1:
        for r in range(len(target)):
            var w = weights[r] if len(weights) > 0 else 1.0
            grad.append(w * _sign(raw[r] - target[r]))
            hess.append(w)
    else:
        for r in range(len(target)):
            var w = weights[r] if len(weights) > 0 else 1.0
            grad.append(w * (raw[r] - target[r]))
            hess.append(w)


def _mean_loss(
    raw: List[Float64], target: List[Float64], objective: Int, alpha: Float64
) -> Float64:
    var total = 0.0
    if objective == BINARY_LOGISTIC:
        for r in range(len(target)):
            var p = _clamp_prob(_sigmoid(raw[r]))
            if target[r] > 0.5:
                total -= log(p)
            else:
                total -= log(1.0 - p)
    elif objective == POISSON:
        # Poisson negative log likelihood up to a constant in the target.
        for r in range(len(target)):
            total += exp(raw[r]) - target[r] * raw[r]
    elif objective == HUBER:
        for r in range(len(target)):
            var d = abs(raw[r] - target[r])
            if d <= alpha:
                total += 0.5 * d * d
            else:
                total += alpha * (d - 0.5 * alpha)
    elif objective == QUANTILE:
        # Pinball loss.
        for r in range(len(target)):
            var d = raw[r] - target[r]
            if d >= 0.0:
                total += (1.0 - alpha) * d
            else:
                total += -alpha * d
    elif objective == L1:
        for r in range(len(target)):
            total += abs(raw[r] - target[r])
    else:
        for r in range(len(target)):
            var d = raw[r] - target[r]
            total += d * d
    return total / Float64(len(target))


def _renew_leaf_values(
    mut tree: Tree,
    data: BinnedMatrix,
    target: List[Float64],
    raw: List[Float64],
    weights: List[Float64],
    alpha: Float64,
    bag: List[Int] = [],
    monotone: List[Int] = [],
) raises:
    """LightGBM's RenewTreeOutput for QUANTILE and L1: replace each leaf's
    Newton value with the alpha-percentile of the residuals
    (target - current raw score) of the rows in that leaf. Runs before
    shrinkage, which then applies to the renewed value.

    A non-empty `bag` renews from bagged rows only, the rows the tree was
    grown on; LightGBM likewise renews over its bagged partition.

    A non-empty `monotone` clamps every renewed value back into its leaf's
    monotone interval, without which renewal would discard the constraint the
    tree was grown under (see monotone.mojo)."""
    var n_nodes = len(tree.feature)
    var bounds = node_bounds(tree, monotone)
    var leaf_residuals = List[List[Float64]]()
    var leaf_weights = List[List[Float64]]()
    for _ in range(n_nodes):
        leaf_residuals.append(List[Float64]())
        leaf_weights.append(List[Float64]())
    var n_used = len(bag) if len(bag) > 0 else data.n_rows
    for i in range(n_used):
        var r = bag[i] if len(bag) > 0 else i
        var node = tree.leaf_index_row(data, r)
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
        if len(bounds) > 0:
            renewed = bounds[node].clamp(renewed)
        tree.value[node] = renewed


@fieldwise_init
struct BoosterParams(Copyable, Movable):
    var n_estimators: Int
    var learning_rate: Float64
    var tree: TreeParams

    @staticmethod
    def default() -> BoosterParams:
        return BoosterParams(100, 0.1, TreeParams.default())


struct Booster(Copyable, Movable):
    """A fitted single-output ensemble.

    `monotone` records the monotonic constraints the ensemble was trained
    under. Unlike the other training-time restrictions it is kept with the
    model and serialized, because it is a property the fitted model satisfies
    rather than only a knob that shaped it: predictions are monotone in those
    features, and a consumer cannot recover that claim from the trees.
    """

    var trees: List[Tree]
    var base_score: Float64
    var learning_rate: Float64
    var objective: Int
    var monotone: MonotoneConstraints

    def __init__(
        out self,
        var trees: List[Tree],
        base_score: Float64,
        learning_rate: Float64,
        objective: Int,
        var monotone: MonotoneConstraints = MonotoneConstraints(),
    ):
        self.trees = trees^
        self.base_score = base_score
        self.learning_rate = learning_rate
        self.objective = objective
        self.monotone = monotone^

    def predict_raw_row(self, data: BinnedMatrix, row: Int) -> Float64:
        """Raw ensemble output (log-odds for BINARY_LOGISTIC)."""
        var s = self.base_score
        for i in range(len(self.trees)):
            s += self.learning_rate * self.trees[i].predict_row(data, row)
        return s

    def predict_row(self, data: BinnedMatrix, row: Int) -> Float64:
        """Prediction on the response scale (probability for logistic,
        expected count for poisson). For CUSTOM this is the raw score:
        the framework does not know the objective's inverse link, so the
        caller applies it."""
        var raw = self.predict_raw_row(data, row)
        if self.objective == BINARY_LOGISTIC:
            return _sigmoid(raw)
        if self.objective == POISSON:
            return exp(raw)
        return raw

    def predict_raw_bins(self, bins: List[Int]) -> Float64:
        var s = self.base_score
        for i in range(len(self.trees)):
            s += self.learning_rate * self.trees[i].predict_bins(bins)
        return s

    def predict_bins(self, bins: List[Int]) -> Float64:
        var raw = self.predict_raw_bins(bins)
        if self.objective == BINARY_LOGISTIC:
            return _sigmoid(raw)
        if self.objective == POISSON:
            return exp(raw)
        return raw


def train(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> Booster:
    """Train a boosted ensemble. `target` is the regression target for
    SQUARED_ERROR, HUBER, QUANTILE, and L1, {0, 1} labels for
    BINARY_LOGISTIC, or nonnegative counts for POISSON. A non-empty
    sample_weight scales each row's gradient and hessian, LightGBM style;
    a row with weight zero is ignored. `alpha` is the target quantile for
    QUANTILE and the huber transition point for HUBER (LightGBM's alpha,
    default 0.9); other objectives ignore it. `bagging` grows each tree on
    a seeded row sample (see bagging.mojo); the base score and the
    per-round score update stay on the full dataset. `goss` grows each tree
    on a gradient-based sample instead (see goss.mojo), which the same row
    list carries; the two samplers are mutually exclusive."""
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    params.tree.monotone.check_features(data.n_features)

    var n = data.n_rows
    var base_score = _base_score(target, objective, sample_weight, alpha)
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)

    var signs = params.tree.monotone.active_signs()
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
        var tree = grow_tree(data, grad, hess, params.tree, bag, i)
        if objective == QUANTILE:
            _renew_leaf_values(
                tree, data, target, raw, sample_weight, alpha, bag, signs
            )
        elif objective == L1:
            _renew_leaf_values(
                tree, data, target, raw, sample_weight, 0.5, bag, signs
            )

        # A single-leaf tree with a near-zero value means the objective has
        # converged; further rounds cannot make progress. Under bagging or
        # GOSS one such tree only says this sample had nothing to give
        # (every sampled row zero-weight, say), so the round is skipped and
        # the next sample gets its turn.
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


def train_with_valid(
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
) raises -> Booster:
    """Train with validation-set early stopping. Stops when the validation
    loss (MSE / log loss / huber / pinball / MAE) has not improved by more
    than min_delta for early_stopping_rounds consecutive rounds and
    truncates the ensemble to its best round. sample_weight applies to
    training rows only; the validation loss is unweighted. `bagging`
    samples training rows per tree (see bagging.mojo); validation rows are
    never bagged, so the early-stopping signal stays out of sample. `goss`
    samples training rows by gradient magnitude instead (see goss.mojo) and
    leaves the validation loss untouched in the same way."""
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

    var n = data.n_rows
    var base_score = _base_score(target, objective, sample_weight, alpha)
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)
    var valid_raw = List[Float64](capacity=valid_data.n_rows)
    for _ in range(valid_data.n_rows):
        valid_raw.append(base_score)

    var signs = params.tree.monotone.active_signs()
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
        var tree = grow_tree(data, grad, hess, params.tree, bag, i)
        if objective == QUANTILE:
            _renew_leaf_values(
                tree, data, target, raw, sample_weight, alpha, bag, signs
            )
        elif objective == L1:
            _renew_leaf_values(
                tree, data, target, raw, sample_weight, 0.5, bag, signs
            )
        # Under bagging or GOSS a degenerate tree indicts the sample, not
        # the run.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            if bagging_enabled(bagging) or goss.enabled:
                continue
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        for r in range(valid_data.n_rows):
            valid_raw[r] += (
                params.learning_rate * tree.predict_row(valid_data, r)
            )
        trees.append(tree^)

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


def _multiclass_mean_loss(
    raw: List[Float64], labels: List[Int], n_classes: Int
) -> Float64:
    """Mean negative log likelihood of the true class from row-major raw
    scores, via a max-subtracted log-sum-exp."""
    var total = 0.0
    for r in range(len(labels)):
        var m = raw[r * n_classes]
        for k in range(1, n_classes):
            if raw[r * n_classes + k] > m:
                m = raw[r * n_classes + k]
        var denom = 0.0
        for k in range(n_classes):
            denom += exp(raw[r * n_classes + k] - m)
        total -= raw[r * n_classes + labels[r]] - m - log(denom)
    return total / Float64(len(labels))


def _softmax_inplace(mut scores: List[Float64], start: Int, k: Int):
    var m = scores[start]
    for i in range(1, k):
        if scores[start + i] > m:
            m = scores[start + i]
    var total = 0.0
    for i in range(k):
        var e = exp(scores[start + i] - m)
        scores[start + i] = e
        total += e
    for i in range(k):
        scores[start + i] /= total


def _fill_softmax_grad_hess(
    prob: List[Float64],
    labels: List[Int],
    k: Int,
    n_classes: Int,
    weights: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
):
    """One-vs-rest gradients and hessians for class `k` from row-major
    softmax probabilities."""
    grad.clear()
    hess.clear()
    for r in range(len(labels)):
        var p = prob[r * n_classes + k]
        var y = 1.0 if labels[r] == k else 0.0
        var w = weights[r] if len(weights) > 0 else 1.0
        grad.append(w * (p - y))
        # LightGBM/XGBoost softmax hessian: 2 * p * (1 - p), floored.
        var h = 2.0 * p * (1.0 - p)
        if h < 1e-16:
            h = 1e-16
        hess.append(w * h)


def _multiclass_goss_select(
    prob: List[Float64],
    labels: List[Int],
    n_classes: Int,
    weights: List[Float64],
    goss: GossParams,
    round: Int,
) raises -> GossSelection:
    """One row sample for a whole multiclass round. LightGBM sums the
    per-row `|grad * hess|` over the round's trees before sampling, so every
    class's tree is grown on the same rows and their leaf counts stay
    comparable."""
    var n = len(labels)
    var importance = List[Float64](capacity=n)
    for _ in range(n):
        importance.append(0.0)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for k in range(n_classes):
        _fill_softmax_grad_hess(
            prob, labels, k, n_classes, weights, grad, hess
        )
        for r in range(n):
            importance[r] += abs(grad[r] * hess[r])
    return goss_select(importance, goss, round)


struct MulticlassBooster(Copyable, Movable):
    """Softmax ensemble: one tree per class per round, round-major, so the
    tree for (round i, class k) is trees[i * n_classes + k].

    `monotone` records the monotonic constraints every per-class tree was
    grown under, which makes each class's raw score monotone in the
    constrained features. Softmax probabilities are not guaranteed monotone;
    see monotone.mojo for the policy.
    """

    var trees: List[Tree]
    var base_scores: List[Float64]
    var n_classes: Int
    var learning_rate: Float64
    var monotone: MonotoneConstraints

    def __init__(
        out self,
        var trees: List[Tree],
        var base_scores: List[Float64],
        n_classes: Int,
        learning_rate: Float64,
        var monotone: MonotoneConstraints = MonotoneConstraints(),
    ):
        self.trees = trees^
        self.base_scores = base_scores^
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.monotone = monotone^

    def predict_raw_bins(self, bins: List[Int]) -> List[Float64]:
        var raw = List[Float64](capacity=self.n_classes)
        for k in range(self.n_classes):
            raw.append(self.base_scores[k])
        var n_rounds = len(self.trees) // self.n_classes
        for i in range(n_rounds):
            for k in range(self.n_classes):
                raw[k] += self.learning_rate * self.trees[
                    i * self.n_classes + k
                ].predict_bins(bins)
        return raw^

    def predict_proba_bins(self, bins: List[Int]) -> List[Float64]:
        var raw = self.predict_raw_bins(bins)
        _softmax_inplace(raw, 0, self.n_classes)
        return raw^


def train_multiclass(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MulticlassBooster:
    """Train a softmax multiclass ensemble on labels in 0..n_classes-1.
    A non-empty sample_weight scales each row's gradient and hessian and
    weights the class priors; a row with weight zero is ignored. `bagging`
    draws one bag per round and every class's tree in that round is grown
    on it, so the per-class trees stay comparable. `goss` samples the round's
    rows by summed per-class gradient magnitude instead (see goss.mojo), one
    sample per round for the same reason."""
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    params.tree.monotone.check_features(data.n_features)
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

    # Row-major raw scores and softmax scratch: raw[r * n_classes + k].
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
            # Feature subsampling draws once per tree, so each class's tree
            # in a round gets its own feature set.
            var tree = grow_tree(
                data, grad, hess, params.tree, bag, i * n_classes + k
            )
            if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                made_progress = True
            for r in range(n):
                raw[r * n_classes + k] += (
                    params.learning_rate * tree.predict_row(data, r)
                )
            trees.append(tree^)

        # No class made progress: with bagging or GOSS that is a statement
        # about this sample, so the round is dropped and the next sample
        # gets its turn.
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


def train_multiclass_with_valid(
    data: BinnedMatrix,
    labels: List[Int],
    valid_data: BinnedMatrix,
    valid_labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MulticlassBooster:
    """Train a softmax multiclass ensemble with validation-set early
    stopping. Stops when the validation multiclass log loss has not
    improved by more than min_delta for early_stopping_rounds consecutive
    rounds and truncates the ensemble to its best round (a round is one
    tree per class). sample_weight applies to training rows only; the
    validation loss is unweighted. `bagging` samples training rows per
    round; validation rows are never bagged. `goss` is the gradient-based
    alternative sampler (see goss.mojo), also drawn once per round."""
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if len(valid_labels) != valid_data.n_rows:
        raise Error("valid_labels length must equal valid n_rows")
    if valid_data.n_features != data.n_features:
        raise Error("valid_data must have the same features")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    params.tree.monotone.check_features(data.n_features)
    var n = data.n_rows
    var n_valid = valid_data.n_rows

    # Base scores are log priors of the TRAINING labels (weighted when
    # sample_weight is given).
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
    for r in range(n_valid):
        if valid_labels[r] < 0 or valid_labels[r] >= n_classes:
            raise Error("valid label out of range")
    var base_scores = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        base_scores.append(log(_clamp_prob(class_w[k] / total_w)))

    # Row-major raw scores for both sets: raw[r * n_classes + k].
    var raw = List[Float64](capacity=n * n_classes)
    for _ in range(n):
        for k in range(n_classes):
            raw.append(base_scores[k])
    var valid_raw = List[Float64](capacity=n_valid * n_classes)
    for _ in range(n_valid):
        for k in range(n_classes):
            valid_raw.append(base_scores[k])
    var prob = List[Float64](capacity=n * n_classes)
    for _ in range(n * n_classes):
        prob.append(0.0)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var best_loss = _multiclass_mean_loss(valid_raw, valid_labels, n_classes)
    var best_n_rounds = 0
    var n_rounds = 0
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
            # Feature subsampling draws once per tree, so each class's tree
            # in a round gets its own feature set.
            var tree = grow_tree(
                data, grad, hess, params.tree, bag, i * n_classes + k
            )
            if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                made_progress = True
            for r in range(n):
                raw[r * n_classes + k] += (
                    params.learning_rate * tree.predict_row(data, r)
                )
            for r in range(n_valid):
                valid_raw[r * n_classes + k] += (
                    params.learning_rate * tree.predict_row(valid_data, r)
                )
            trees.append(tree^)

        # Popped trees are all single-leaf with value ~0, so the score
        # updates above were no-ops. Under bagging or GOSS the next sample
        # still gets its turn.
        if not made_progress:
            for _ in range(n_classes):
                _ = trees.pop()
            if bagging_enabled(bagging) or goss.enabled:
                continue
            break
        n_rounds += 1

        var loss = _multiclass_mean_loss(valid_raw, valid_labels, n_classes)
        if loss < best_loss - min_delta:
            best_loss = loss
            best_n_rounds = n_rounds
        elif n_rounds - best_n_rounds >= early_stopping_rounds:
            break

    while len(trees) > best_n_rounds * n_classes:
        _ = trees.pop()
    return MulticlassBooster(
        trees^,
        base_scores^,
        n_classes,
        params.learning_rate,
        params.tree.monotone.copy(),
    )
