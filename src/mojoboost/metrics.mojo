"""Evaluation metrics.

Standalone helpers for scoring predictions: squared error and its root,
absolute error, the pinball, Huber, and fair losses, MAPE, the poisson,
gamma, and tweedie deviances, log loss (binary, continuous-label cross
entropy, and multiclass), KL divergence, accuracy, error rate, ROC AUC, and
average precision. Probability inputs follow the library conventions
elsewhere: binary labels are Float64 in {0, 1}, multiclass labels are Int in
0..n_classes-1, and multiclass probabilities are row-major,
probs[r * n_classes + k].

Scale. Every metric takes predictions on the scale a user reads them on,
which is the objective's response scale: `l2` and its relatives take the
regression prediction, `binary_log_loss` takes a probability, and the
poisson, gamma, and tweedie metrics take the *expected value* (`exp(raw)`),
not the raw score. `response_scale` in custom_metric.mojo is the one call
that converts, and it applies exactly the links `Booster.response` does.

Direction. All of these are losses (lower is better) except
`binary_accuracy`, `multiclass_accuracy`, `binary_auc`, and
`average_precision`, which are scores (higher is better). Nothing here
carries its direction as data; `CustomMetric` in custom_metric.mojo is where
a metric is paired with its direction for early stopping.

Every metric takes a trailing `weight` vector. An empty one, the default,
weights every row equally and reproduces the unweighted definition exactly.
A non-empty one must have one finite, nonnegative entry per row and a
positive sum, and turns the metric into the weighted mean LightGBM reports
for a validation set that carries weights. Weighted AUC is the weighted
rank statistic: a tie between a positive and a negative row contributes half
of the product of their weights, which is what the unweighted average-rank
formula does when every weight is 1.

Metric codes, and why they live here
------------------------------------
`METRIC_L2` ... `METRIC_MAP` below are the twenty-one built-in metric codes.
A code names a *function*, and nineteen of those functions are in this file,
so this is where the numbering lives; `objective_registry.mojo` re-exports
every one of them and owns everything a code *means* (its canonical name and
aliases, the task it scores, which direction is better, what it needs beyond
predictions and labels, and which transform predictions must have been
through). Two codes name functions in ranking.mojo (`METRIC_NDCG`,
`METRIC_MAP`) and two need a class count (`METRIC_MULTI_LOGLOSS`,
`METRIC_MULTI_ERROR`); `eval_metric_by_code` rejects all four by name and
points at the entry point that does handle them.

The values are fixed by `bindings/_mojoboost.mojo` and
`python/mojoboost/_eval.py` and must not be renumbered: a code crosses the
Python boundary as an integer.

This module deliberately does **not** import `objective_registry`. It cannot:
boosting.mojo imports `_argsort` from here, the registry imports boosting,
and the three would form a cycle. That constraint is also the right layering.
Nothing here decides what a metric means, only what it computes, so a metric
never applies a transform, never chooses a direction, and never rejects a
metric that does not suit a model's task. Those are registry questions, and
`eval_builtin_metric` in custom_metric.mojo is the one call path that asks
them and then lands here.
"""

from std.math import exp, isfinite, log, sqrt

# ---------------------------------------------------------------------------
# Metric codes
# ---------------------------------------------------------------------------

comptime METRIC_L2 = 0
comptime METRIC_RMSE = 1
comptime METRIC_L1 = 2
comptime METRIC_QUANTILE = 3
comptime METRIC_HUBER = 4
comptime METRIC_BINARY_LOGLOSS = 5
comptime METRIC_BINARY_ERROR = 6
comptime METRIC_AUC = 7
comptime METRIC_MULTI_LOGLOSS = 8
comptime METRIC_MULTI_ERROR = 9
comptime METRIC_NDCG = 10
comptime METRIC_MAPE = 11
comptime METRIC_FAIR = 12
comptime METRIC_POISSON = 13
comptime METRIC_GAMMA = 14
comptime METRIC_GAMMA_DEVIANCE = 15
comptime METRIC_TWEEDIE = 16
comptime METRIC_CROSS_ENTROPY = 17
comptime METRIC_KLDIV = 18
comptime METRIC_AVERAGE_PRECISION = 19
comptime METRIC_MAP = 20

comptime N_BUILTIN_METRICS = 21

# The default threshold `binary_error` scores at, LightGBM's. Named because
# `eval_metric_by_code` has to pass one and a caller reading the dispatch
# should see which one rather than a bare 0.5.
comptime DEFAULT_BINARY_THRESHOLD = 0.5


def _clamp(p: Float64) -> Float64:
    if p < 1e-15:
        return 1e-15
    if p > 1.0 - 1e-15:
        return 1.0 - 1e-15
    return p


def _check_same_length(a: Int, b: Int) raises:
    if a != b:
        raise Error("predictions and labels must have the same length")
    if a == 0:
        raise Error("metrics need at least one row")


def check_metric_weight(weight: List[Float64], n: Int) raises -> Float64:
    """Total weight of `n` rows: `n` itself when `weight` is empty, and the
    validated sum otherwise. Entries must be finite and nonnegative and
    their sum positive, since a zero total leaves nothing to average over."""
    if len(weight) == 0:
        return Float64(n)
    if len(weight) != n:
        raise Error("weight length must equal the number of rows")
    var total = 0.0
    for r in range(n):
        if not isfinite(weight[r]):
            raise Error("weights must be finite")
        if weight[r] < 0.0:
            raise Error("weights must be nonnegative")
        total += weight[r]
    if total <= 0.0:
        raise Error("weights must have a positive sum")
    return total


def _w(weight: List[Float64], r: Int) -> Float64:
    return weight[r] if len(weight) > 0 else 1.0


def l2(
    pred: List[Float64], target: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Mean squared error, LightGBM's `l2`."""
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    var total = 0.0
    for r in range(len(pred)):
        var d = pred[r] - target[r]
        total += _w(weight, r) * d * d
    return total / total_w


def rmse(
    pred: List[Float64], target: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Root mean squared error, LightGBM's `rmse`."""
    return sqrt(l2(pred, target, weight))


def l1(
    pred: List[Float64], target: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Mean absolute error, LightGBM's `l1`."""
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    var total = 0.0
    for r in range(len(pred)):
        total += _w(weight, r) * abs(pred[r] - target[r])
    return total / total_w


def quantile_loss(
    pred: List[Float64],
    target: List[Float64],
    alpha: Float64,
    weight: List[Float64] = [],
) raises -> Float64:
    """Mean pinball loss at level `alpha`, the loss the QUANTILE objective
    minimizes (boosting.mojo's `_mean_loss` computes the same quantity)."""
    if not 0.0 < alpha < 1.0:
        raise Error("quantile alpha must be in (0, 1)")
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    var total = 0.0
    for r in range(len(pred)):
        var d = pred[r] - target[r]
        var loss = (1.0 - alpha) * d if d >= 0.0 else -alpha * d
        total += _w(weight, r) * loss
    return total / total_w


def huber_loss(
    pred: List[Float64],
    target: List[Float64],
    alpha: Float64,
    weight: List[Float64] = [],
) raises -> Float64:
    """Mean Huber loss with transition point `alpha`, the loss the HUBER
    objective minimizes (boosting.mojo's `_mean_loss` computes the same
    quantity)."""
    if alpha <= 0.0:
        raise Error("huber alpha must be positive")
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    var total = 0.0
    for r in range(len(pred)):
        var d = abs(pred[r] - target[r])
        var loss = 0.5 * d * d if d <= alpha else alpha * (d - 0.5 * alpha)
        total += _w(weight, r) * loss
    return total / total_w


def fair_loss(
    pred: List[Float64],
    target: List[Float64],
    c: Float64,
    weight: List[Float64] = [],
) raises -> Float64:
    """Mean fair loss with parameter `c` (LightGBM's `fair_c`), the loss the
    FAIR objective minimizes: `c^2 * (|d|/c - log(1 + |d|/c))`. Quadratic in
    the residual near zero and linear far from it, like Huber, but smooth
    everywhere."""
    if c <= 0.0:
        raise Error("fair c must be positive")
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    var total = 0.0
    for r in range(len(pred)):
        var d = abs(pred[r] - target[r]) / c
        total += _w(weight, r) * c * c * (d - log(1.0 + d))
    return total / total_w


def mape(
    pred: List[Float64], target: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Mean absolute percentage error, LightGBM's `mape`:
    `|pred - target| / max(1, |target|)`.

    The denominator's floor at 1 is LightGBM's, and it is why this is not
    quite the textbook MAPE: without it a label at zero would divide by
    zero, and one near zero would swamp the mean. It matches the label
    weight the MAPE objective trains against, so the metric and the
    objective measure the same thing.
    """
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    var total = 0.0
    for r in range(len(pred)):
        var denom = abs(target[r])
        if denom < 1.0:
            denom = 1.0
        total += _w(weight, r) * abs(pred[r] - target[r]) / denom
    return total / total_w


def _check_positive_response(pred: List[Float64], name: String) raises:
    for r in range(len(pred)):
        if pred[r] <= 0.0:
            raise Error(
                String(
                    name,
                    " needs positive predictions (the response scale,"
                    " exp(raw))",
                )
            )


def poisson_loss(
    pred: List[Float64], target: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Mean poisson negative log likelihood up to a constant in the target,
    LightGBM's `poisson` metric: `mu - y * log(mu)`.

    `pred` is on the response scale, the expected count `exp(raw)`. The
    dropped constant is `log(y!)`, which no prediction can change; the
    metric is therefore comparable across models but not an absolute
    likelihood."""
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    _check_positive_response(pred, "poisson")
    var total = 0.0
    for r in range(len(pred)):
        total += _w(weight, r) * (pred[r] - target[r] * log(pred[r]))
    return total / total_w


def gamma_loss(
    pred: List[Float64], target: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Mean gamma negative log likelihood up to a constant in the target,
    LightGBM's `gamma` metric: `y / mu + log(mu)` at unit dispersion.

    `pred` is the expected value `exp(raw)`. Labels must be positive, as
    for the objective."""
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    _check_positive_response(pred, "gamma")
    var total = 0.0
    for r in range(len(pred)):
        if target[r] <= 0.0:
            raise Error("gamma needs positive labels")
        total += _w(weight, r) * (target[r] / pred[r] + log(pred[r]))
    return total / total_w


def gamma_deviance(
    pred: List[Float64], target: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Mean gamma deviance, LightGBM's `gamma_deviance`:
    `2 * (y/mu - log(y/mu) - 1)`.

    The same likelihood as `gamma_loss` recentred so that a perfect
    prediction scores 0, which is what makes a deviance readable on its own
    rather than only in comparison."""
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    _check_positive_response(pred, "gamma_deviance")
    var total = 0.0
    for r in range(len(pred)):
        if target[r] <= 0.0:
            raise Error("gamma_deviance needs positive labels")
        var ratio = target[r] / pred[r]
        total += _w(weight, r) * (ratio - log(ratio) - 1.0)
    return 2.0 * total / total_w


def tweedie_loss(
    pred: List[Float64],
    target: List[Float64],
    variance_power: Float64,
    weight: List[Float64] = [],
) raises -> Float64:
    """Mean tweedie negative log likelihood up to a constant in the target,
    LightGBM's `tweedie` metric:
    `-y * mu^(1-rho) / (1-rho) + mu^(2-rho) / (2-rho)`.

    `pred` is the expected value `exp(raw)` and `variance_power` is rho, the
    same value the objective trains with; scoring at a different rho scores
    a different loss."""
    if not 1.0 < variance_power < 2.0:
        raise Error("tweedie variance_power must be in (1, 2)")
    _check_same_length(len(pred), len(target))
    var total_w = check_metric_weight(weight, len(pred))
    _check_positive_response(pred, "tweedie")
    var total = 0.0
    for r in range(len(pred)):
        var ln_mu = log(pred[r])
        var a = target[r] * exp((1.0 - variance_power) * ln_mu) / (
            1.0 - variance_power
        )
        var b = exp((2.0 - variance_power) * ln_mu) / (2.0 - variance_power)
        total += _w(weight, r) * (-a + b)
    return total / total_w


def binary_log_loss(
    probs: List[Float64], labels: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Mean negative log likelihood of {0, 1} labels under probabilities."""
    _check_same_length(len(probs), len(labels))
    var total_w = check_metric_weight(weight, len(probs))
    var total = 0.0
    for r in range(len(probs)):
        var p = _clamp(probs[r])
        if labels[r] > 0.5:
            total -= _w(weight, r) * log(p)
        else:
            total -= _w(weight, r) * log(1.0 - p)
    return total / total_w


def _check_probability_labels(labels: List[Float64], name: String) raises:
    for r in range(len(labels)):
        if labels[r] < 0.0 or labels[r] > 1.0:
            raise Error(String(name, " labels must be in [0, 1]"))


def cross_entropy_loss(
    probs: List[Float64], labels: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Mean cross entropy against labels anywhere in [0, 1], LightGBM's
    `cross_entropy` (`xentropy`) metric:
    `-(y * log(p) + (1 - y) * log(1 - p))`.

    `binary_log_loss` is this metric restricted to {0, 1} labels, where one
    of the two terms always vanishes. Kept apart because a label strictly
    between 0 and 1 is a soft target rather than a misread class, and
    validating it here catches the swapped-argument mistake that a lenient
    binary log loss would score without complaint."""
    _check_same_length(len(probs), len(labels))
    var total_w = check_metric_weight(weight, len(probs))
    _check_probability_labels(labels, "cross entropy")
    var total = 0.0
    for r in range(len(probs)):
        var p = _clamp(probs[r])
        total -= _w(weight, r) * (
            labels[r] * log(p) + (1.0 - labels[r]) * log(1.0 - p)
        )
    return total / total_w


def kullback_leibler(
    probs: List[Float64], labels: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Mean Kullback-Leibler divergence from the labels to the predictions,
    LightGBM's `kullback_leibler` (`kldiv`).

    This is `cross_entropy_loss` minus the labels' own entropy, so a perfect
    prediction scores 0 rather than the irreducible entropy the cross
    entropy keeps. The entropy term is a constant of the data: the two
    metrics rank models identically and differ only in where zero sits.
    `0 * log 0` is taken as 0, its limit."""
    _check_same_length(len(probs), len(labels))
    var total_w = check_metric_weight(weight, len(probs))
    _check_probability_labels(labels, "kullback_leibler")
    var total = 0.0
    for r in range(len(probs)):
        var p = _clamp(probs[r])
        var y = labels[r]
        var d = 0.0
        if y > 0.0:
            d += y * log(y / p)
        if y < 1.0:
            d += (1.0 - y) * log((1.0 - y) / (1.0 - p))
        total += _w(weight, r) * d
    return total / total_w


def binary_accuracy(
    probs: List[Float64],
    labels: List[Float64],
    threshold: Float64 = 0.5,
    weight: List[Float64] = [],
) raises -> Float64:
    """Fraction of rows where thresholded probability matches the label."""
    _check_same_length(len(probs), len(labels))
    var total_w = check_metric_weight(weight, len(probs))
    var correct = 0.0
    for r in range(len(probs)):
        var predicted = 1.0 if probs[r] >= threshold else 0.0
        if abs(predicted - labels[r]) < 0.5:
            correct += _w(weight, r)
    return correct / total_w


def binary_error(
    probs: List[Float64],
    labels: List[Float64],
    threshold: Float64 = 0.5,
    weight: List[Float64] = [],
) raises -> Float64:
    """Misclassification rate, LightGBM's `binary_error`: one minus
    `binary_accuracy` at the same threshold."""
    return 1.0 - binary_accuracy(probs, labels, threshold, weight)


def multiclass_log_loss(
    probs: List[Float64],
    labels: List[Int],
    n_classes: Int,
    weight: List[Float64] = [],
) raises -> Float64:
    """Mean negative log likelihood of the true class, row-major probs."""
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if len(probs) != len(labels) * n_classes:
        raise Error("probs must have n_rows * n_classes entries")
    if len(labels) == 0:
        raise Error("metrics need at least one row")
    var total_w = check_metric_weight(weight, len(labels))
    var total = 0.0
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        total -= _w(weight, r) * log(_clamp(probs[r * n_classes + labels[r]]))
    return total / total_w


def multiclass_accuracy(
    probs: List[Float64],
    labels: List[Int],
    n_classes: Int,
    weight: List[Float64] = [],
) raises -> Float64:
    """Fraction of rows where the argmax class matches the label."""
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if len(probs) != len(labels) * n_classes:
        raise Error("probs must have n_rows * n_classes entries")
    if len(labels) == 0:
        raise Error("metrics need at least one row")
    var total_w = check_metric_weight(weight, len(labels))
    var correct = 0.0
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        var argmax = 0
        for k in range(1, n_classes):
            if probs[r * n_classes + k] > probs[r * n_classes + argmax]:
                argmax = k
        if argmax == labels[r]:
            correct += _w(weight, r)
    return correct / total_w


def multiclass_error(
    probs: List[Float64],
    labels: List[Int],
    n_classes: Int,
    weight: List[Float64] = [],
) raises -> Float64:
    """Misclassification rate, LightGBM's `multi_error`: one minus
    `multiclass_accuracy`."""
    return 1.0 - multiclass_accuracy(probs, labels, n_classes, weight)


def _argsort(values: List[Float64]) -> List[Int]:
    """Indices that sort values ascending (stable bottom-up merge sort)."""
    var n = len(values)
    var idx = List[Int](capacity=n)
    for i in range(n):
        idx.append(i)
    var buf = List[Int](capacity=n)
    for _ in range(n):
        buf.append(0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            if mid > n:
                mid = n
            var hi = lo + 2 * width
            if hi > n:
                hi = n
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if values[idx[i]] <= values[idx[j]]:
                    buf[k] = idx[i]
                    i += 1
                else:
                    buf[k] = idx[j]
                    j += 1
                k += 1
            while i < mid:
                buf[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                buf[k] = idx[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(n):
            idx[t] = buf[t]
        width *= 2
    return idx^


def binary_auc(
    scores: List[Float64], labels: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Area under the ROC curve, the probability that a random positive
    outranks a random negative.

    Works on any monotone score (raw margin or probability). Rows are swept
    in score order and each tied block contributes half of its own
    positive-negative weight product, which is the average-rank rule
    sklearn's `roc_auc_score` uses, generalized to weights.
    """
    _check_same_length(len(scores), len(labels))
    var n = len(scores)
    _ = check_metric_weight(weight, n)
    var w_pos = 0.0
    var w_neg = 0.0
    for r in range(n):
        if labels[r] > 0.5:
            w_pos += _w(weight, r)
        else:
            w_neg += _w(weight, r)
    if w_pos <= 0.0 or w_neg <= 0.0:
        raise Error("AUC needs at least one positive and one negative")

    var order = _argsort(scores)
    var seen_neg = 0.0
    var total = 0.0
    var i = 0
    while i < n:
        var j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # One tied block [i, j]: its positives beat every negative already
        # swept and tie with the negatives inside the block.
        var block_pos = 0.0
        var block_neg = 0.0
        for t in range(i, j + 1):
            if labels[order[t]] > 0.5:
                block_pos += _w(weight, order[t])
            else:
                block_neg += _w(weight, order[t])
        total += block_pos * (seen_neg + 0.5 * block_neg)
        seen_neg += block_neg
        i = j + 1

    return total / (w_pos * w_neg)


def average_precision(
    scores: List[Float64], labels: List[Float64], weight: List[Float64] = []
) raises -> Float64:
    """Area under the precision-recall curve by the step-wise rule,
    LightGBM's `average_precision` and scikit-learn's
    `average_precision_score`: `sum over thresholds of
    (recall_k - recall_{k-1}) * precision_k`.

    Thresholds are the distinct scores in descending order, so a block of
    tied scores contributes one step and its whole positive mass is counted
    at the precision that block reaches. That is the interpolation-free
    rule; it is why this is not the trapezoid under the same curve, which
    would give tied blocks credit for a diagonal they do not earn.

    Higher is better, unlike almost everything else in this module, and
    unlike AUC it needs no negatives-and-positives-both-present guarantee
    beyond at least one positive: with no positive there is no recall axis.
    """
    _check_same_length(len(scores), len(labels))
    var n = len(scores)
    _ = check_metric_weight(weight, n)
    var w_pos = 0.0
    for r in range(n):
        if labels[r] > 0.5:
            w_pos += _w(weight, r)
    if w_pos <= 0.0:
        raise Error("average_precision needs at least one positive")

    # Ascending order reversed: the sweep runs from the highest score down,
    # which is the order the precision-recall curve is traced in.
    var order = _argsort(scores)
    var tp = 0.0
    var fp = 0.0
    var prev_recall = 0.0
    var total = 0.0
    var i = n - 1
    while i >= 0:
        var j = i
        while j - 1 >= 0 and scores[order[j - 1]] == scores[order[i]]:
            j -= 1
        for t in range(j, i + 1):
            if labels[order[t]] > 0.5:
                tp += _w(weight, order[t])
            else:
                fp += _w(weight, order[t])
        var seen = tp + fp
        if seen > 0.0:
            var recall = tp / w_pos
            total += (recall - prev_recall) * (tp / seen)
            prev_recall = recall
        i = j - 1

    return total


# ---------------------------------------------------------------------------
# Code dispatch
# ---------------------------------------------------------------------------


def eval_metric_by_code(
    metric: Int,
    pred: List[Float64],
    target: List[Float64],
    weight: List[Float64] = [],
    param: Float64 = 0.0,
) raises -> Float64:
    """One of the nineteen single-output built-in metrics, selected by code.

    This is the code-to-call map, the one thing the registry does not do:
    the registry says a code is `METRIC_TWEEDIE` and that it needs the
    objective's scalar parameter, and this turns that into a call to
    `tweedie_loss`. It is the only such map in Mojo, so a metric added here
    becomes reachable from every caller at once rather than from whichever
    ones remembered to grow a branch.

    `pred` must already be on the scale the metric reads, which for every
    code here is the *objective's* response scale (`metric_transform` returns
    `TRANSFORM_OBJECTIVE_LINK` for all nineteen). Applying the link is the
    caller's job because only the caller knows the objective;
    `eval_builtin_metric` in custom_metric.mojo is the caller that does it.

    `param` is the objective's scalar parameter and is read only by the four
    codes whose loss is a family rather than a single function: `quantile`
    and `huber` read it as `alpha`, `fair` as `fair_c`, and `tweedie` as
    `tweedie_variance_power`. Those four validate their own domains. The
    other fifteen ignore it, so passing the objective's parameter
    unconditionally is safe and is what the dispatcher does.

    The four codes this does not compute raise rather than return a wrong
    number: the two multiclass metrics need a class count and row-major
    probabilities, and the two ranking metrics need the validation set's own
    query groups and a cutoff. `eval_builtin_metric` routes all four; nothing
    else should special-case them.
    """
    if metric == METRIC_L2:
        return l2(pred, target, weight)
    if metric == METRIC_RMSE:
        return rmse(pred, target, weight)
    if metric == METRIC_L1:
        return l1(pred, target, weight)
    if metric == METRIC_QUANTILE:
        return quantile_loss(pred, target, param, weight)
    if metric == METRIC_HUBER:
        return huber_loss(pred, target, param, weight)
    if metric == METRIC_MAPE:
        return mape(pred, target, weight)
    if metric == METRIC_FAIR:
        return fair_loss(pred, target, param, weight)
    if metric == METRIC_POISSON:
        return poisson_loss(pred, target, weight)
    if metric == METRIC_GAMMA:
        return gamma_loss(pred, target, weight)
    if metric == METRIC_GAMMA_DEVIANCE:
        return gamma_deviance(pred, target, weight)
    if metric == METRIC_TWEEDIE:
        return tweedie_loss(pred, target, param, weight)
    if metric == METRIC_CROSS_ENTROPY:
        return cross_entropy_loss(pred, target, weight)
    if metric == METRIC_KLDIV:
        return kullback_leibler(pred, target, weight)
    if metric == METRIC_BINARY_LOGLOSS:
        return binary_log_loss(pred, target, weight)
    if metric == METRIC_BINARY_ERROR:
        return binary_error(pred, target, DEFAULT_BINARY_THRESHOLD, weight)
    if metric == METRIC_AUC:
        # AUC and average precision read only the order of the scores, which
        # every link the objectives use preserves, so whether the caller
        # applied the transform cannot change either number.
        return binary_auc(pred, target, weight)
    if metric == METRIC_AVERAGE_PRECISION:
        return average_precision(pred, target, weight)
    if metric == METRIC_MULTI_LOGLOSS or metric == METRIC_MULTI_ERROR:
        raise Error(
            "the multiclass metrics need a class count and row-major"
            " probabilities; call multiclass_log_loss / multiclass_error, or"
            " eval_builtin_metric, which routes them"
        )
    if metric == METRIC_NDCG or metric == METRIC_MAP:
        raise Error(
            "the ranking metrics need the validation set's query groups and"
            " a cutoff; call ndcg / mean_average_precision in ranking.mojo,"
            " or eval_builtin_metric, which routes them"
        )
    raise Error(String("unknown metric code ", metric))
