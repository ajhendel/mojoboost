"""Gradient boosting loop.

Trains an additive ensemble of leaf-wise trees on second-order gradients.
Supported objectives are squared error (regression) and binary logistic
(classification, labels in {0, 1}).
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

    # Base score: mean target for regression, prior log-odds for logistic.
    var mean = 0.0
    for r in range(n):
        mean += target[r]
    mean /= Float64(n)
    var base_score: Float64
    if objective == BINARY_LOGISTIC:
        var p = mean
        if p < 1e-15:
            p = 1e-15
        if p > 1.0 - 1e-15:
            p = 1.0 - 1e-15
        base_score = log(p / (1.0 - p))
    else:
        base_score = mean

    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)

    var trees = List[Tree]()
    for _ in range(params.n_estimators):
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        if objective == BINARY_LOGISTIC:
            for r in range(n):
                var p = _sigmoid(raw[r])
                grad.append(p - target[r])
                var h = p * (1.0 - p)
                if h < 1e-16:
                    h = 1e-16
                hess.append(h)
        else:
            for r in range(n):
                grad.append(raw[r] - target[r])
                hess.append(1.0)

        var tree = grow_tree(data, grad, hess, params.tree)

        # A single-leaf tree with a near-zero value means the objective has
        # converged; further rounds cannot make progress.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        trees.append(tree^)

    return Booster(trees^, base_score, params.learning_rate, objective)
