"""Gradient boosting loop.

Trains an additive ensemble of leaf-wise trees on second-order gradients.
Objectives: squared error (regression), binary logistic (labels in {0, 1}),
and multiclass softmax (labels in 0..n_classes-1, via train_multiclass).
"""

from std.math import exp, log

from .binning import BinnedMatrix
from .tree import Tree, TreeParams, grow_tree

comptime SQUARED_ERROR = 0
comptime BINARY_LOGISTIC = 1


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


def _base_score(target: List[Float64], objective: Int) -> Float64:
    var mean = 0.0
    for r in range(len(target)):
        mean += target[r]
    mean /= Float64(len(target))
    if objective == BINARY_LOGISTIC:
        var p = _clamp_prob(mean)
        return log(p / (1.0 - p))
    return mean


def _fill_grad_hess(
    raw: List[Float64],
    target: List[Float64],
    objective: Int,
    mut grad: List[Float64],
    mut hess: List[Float64],
):
    grad.clear()
    hess.clear()
    if objective == BINARY_LOGISTIC:
        for r in range(len(target)):
            var p = _sigmoid(raw[r])
            grad.append(p - target[r])
            var h = p * (1.0 - p)
            if h < 1e-16:
                h = 1e-16
            hess.append(h)
    else:
        for r in range(len(target)):
            grad.append(raw[r] - target[r])
            hess.append(1.0)


def _mean_loss(
    raw: List[Float64], target: List[Float64], objective: Int
) -> Float64:
    var total = 0.0
    if objective == BINARY_LOGISTIC:
        for r in range(len(target)):
            var p = _clamp_prob(_sigmoid(raw[r]))
            if target[r] > 0.5:
                total -= log(p)
            else:
                total -= log(1.0 - p)
    else:
        for r in range(len(target)):
            var d = raw[r] - target[r]
            total += d * d
    return total / Float64(len(target))


@fieldwise_init
struct BoosterParams(Copyable, Movable):
    var n_estimators: Int
    var learning_rate: Float64
    var tree: TreeParams

    @staticmethod
    def default() -> BoosterParams:
        return BoosterParams(100, 0.1, TreeParams.default())


@fieldwise_init
struct Booster(Copyable, Movable):
    var trees: List[Tree]
    var base_score: Float64
    var learning_rate: Float64
    var objective: Int

    def predict_raw_row(self, data: BinnedMatrix, row: Int) -> Float64:
        """Raw ensemble output (log-odds for BINARY_LOGISTIC)."""
        var s = self.base_score
        for i in range(len(self.trees)):
            s += self.learning_rate * self.trees[i].predict_row(data, row)
        return s

    def predict_row(self, data: BinnedMatrix, row: Int) -> Float64:
        """Prediction on the response scale (probability for logistic)."""
        var raw = self.predict_raw_row(data, row)
        if self.objective == BINARY_LOGISTIC:
            return _sigmoid(raw)
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
        return raw


def train(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
) raises -> Booster:
    """Train a boosted ensemble. `target` is the regression target for
    SQUARED_ERROR or {0, 1} labels for BINARY_LOGISTIC."""
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if objective != SQUARED_ERROR and objective != BINARY_LOGISTIC:
        raise Error("unknown objective")

    var n = data.n_rows
    var base_score = _base_score(target, objective)
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for _ in range(params.n_estimators):
        _fill_grad_hess(raw, target, objective, grad, hess)
        var tree = grow_tree(data, grad, hess, params.tree)

        # A single-leaf tree with a near-zero value means the objective has
        # converged; further rounds cannot make progress.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        trees.append(tree^)

    return Booster(trees^, base_score, params.learning_rate, objective)


def train_with_valid(
    data: BinnedMatrix,
    target: List[Float64],
    valid_data: BinnedMatrix,
    valid_target: List[Float64],
    objective: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    min_delta: Float64 = 0.0,
) raises -> Booster:
    """Train with validation-set early stopping. Stops when the validation
    loss (MSE / log loss) has not improved by more than min_delta for
    early_stopping_rounds consecutive rounds and truncates the ensemble
    to its best round."""
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if len(valid_target) != valid_data.n_rows:
        raise Error("valid_target length must equal valid n_rows")
    if valid_data.n_features != data.n_features:
        raise Error("valid_data must have the same features")
    if objective != SQUARED_ERROR and objective != BINARY_LOGISTIC:
        raise Error("unknown objective")
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")

    var n = data.n_rows
    var base_score = _base_score(target, objective)
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)
    var valid_raw = List[Float64](capacity=valid_data.n_rows)
    for _ in range(valid_data.n_rows):
        valid_raw.append(base_score)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var best_loss = _mean_loss(valid_raw, valid_target, objective)
    var best_n_trees = 0
    for _ in range(params.n_estimators):
        _fill_grad_hess(raw, target, objective, grad, hess)
        var tree = grow_tree(data, grad, hess, params.tree)
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        for r in range(valid_data.n_rows):
            valid_raw[r] += (
                params.learning_rate * tree.predict_row(valid_data, r)
            )
        trees.append(tree^)

        var loss = _mean_loss(valid_raw, valid_target, objective)
        if loss < best_loss - min_delta:
            best_loss = loss
            best_n_trees = len(trees)
        elif len(trees) - best_n_trees >= early_stopping_rounds:
            break

    while len(trees) > best_n_trees:
        _ = trees.pop()
    return Booster(trees^, base_score, params.learning_rate, objective)


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


@fieldwise_init
struct MulticlassBooster(Copyable, Movable):
    """Softmax ensemble: one tree per class per round, round-major, so the
    tree for (round i, class k) is trees[i * n_classes + k]."""

    var trees: List[Tree]
    var base_scores: List[Float64]
    var n_classes: Int
    var learning_rate: Float64

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
) raises -> MulticlassBooster:
    """Train a softmax multiclass ensemble on labels in 0..n_classes-1."""
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    var n = data.n_rows

    # Base scores are log priors.
    var counts = List[Int](capacity=n_classes)
    for _ in range(n_classes):
        counts.append(0)
    for r in range(n):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        counts[labels[r]] += 1
    var base_scores = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        base_scores.append(log(_clamp_prob(Float64(counts[k]) / Float64(n))))

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
    for _ in range(params.n_estimators):
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
                grad.append(p - y)
                # LightGBM/XGBoost softmax hessian: 2 * p * (1 - p), floored.
                var h = 2.0 * p * (1.0 - p)
                if h < 1e-16:
                    h = 1e-16
                hess.append(h)
            var tree = grow_tree(data, grad, hess, params.tree)
            if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                made_progress = True
            for r in range(n):
                raw[r * n_classes + k] += (
                    params.learning_rate * tree.predict_row(data, r)
                )
            trees.append(tree^)

        if not made_progress:
            for _ in range(n_classes):
                _ = trees.pop()
            break

    return MulticlassBooster(
        trees^, base_scores^, n_classes, params.learning_rate
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
) raises -> MulticlassBooster:
    """Train a softmax multiclass ensemble with validation-set early
    stopping. Stops when the validation multiclass log loss has not
    improved by more than min_delta for early_stopping_rounds consecutive
    rounds and truncates the ensemble to its best round (a round is one
    tree per class)."""
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
    var n = data.n_rows
    var n_valid = valid_data.n_rows

    # Base scores are log priors of the TRAINING labels.
    var counts = List[Int](capacity=n_classes)
    for _ in range(n_classes):
        counts.append(0)
    for r in range(n):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        counts[labels[r]] += 1
    for r in range(n_valid):
        if valid_labels[r] < 0 or valid_labels[r] >= n_classes:
            raise Error("valid label out of range")
    var base_scores = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        base_scores.append(log(_clamp_prob(Float64(counts[k]) / Float64(n))))

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
    for _ in range(params.n_estimators):
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
                grad.append(p - y)
                var h = 2.0 * p * (1.0 - p)
                if h < 1e-16:
                    h = 1e-16
                hess.append(h)
            var tree = grow_tree(data, grad, hess, params.tree)
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

        if not made_progress:
            for _ in range(n_classes):
                _ = trees.pop()
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
        trees^, base_scores^, n_classes, params.learning_rate
    )
