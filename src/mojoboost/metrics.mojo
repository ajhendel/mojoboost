"""Evaluation metrics.

Standalone helpers for scoring predictions: squared error and its root,
absolute error, the pinball and Huber losses, log loss (binary and
multiclass), accuracy, error rate, and ROC AUC. Probability inputs follow
the library conventions elsewhere: binary labels are Float64 in {0, 1},
multiclass labels are Int in 0..n_classes-1, and multiclass probabilities
are row-major, probs[r * n_classes + k].

Every metric takes a trailing `weight` vector. An empty one, the default,
weights every row equally and reproduces the unweighted definition exactly.
A non-empty one must have one finite, nonnegative entry per row and a
positive sum, and turns the metric into the weighted mean LightGBM reports
for a validation set that carries weights. Weighted AUC is the weighted
rank statistic: a tie between a positive and a negative row contributes half
of the product of their weights, which is what the unweighted average-rank
formula does when every weight is 1.
"""

from std.math import isfinite, log, sqrt


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
