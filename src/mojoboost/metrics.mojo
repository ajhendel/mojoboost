"""Evaluation metrics.

Standalone helpers for scoring predictions: RMSE, log loss (binary and
multiclass), accuracy, and ROC AUC. Probability inputs follow the library
conventions elsewhere: binary labels are Float64 in {0, 1}, multiclass
labels are Int in 0..n_classes-1, and multiclass probabilities are
row-major, probs[r * n_classes + k].
"""

from std.math import log, sqrt


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


def rmse(pred: List[Float64], target: List[Float64]) raises -> Float64:
    """Root mean squared error."""
    _check_same_length(len(pred), len(target))
    var total = 0.0
    for r in range(len(pred)):
        var d = pred[r] - target[r]
        total += d * d
    return sqrt(total / Float64(len(pred)))


def binary_log_loss(
    probs: List[Float64], labels: List[Float64]
) raises -> Float64:
    """Mean negative log likelihood of {0, 1} labels under probabilities."""
    _check_same_length(len(probs), len(labels))
    var total = 0.0
    for r in range(len(probs)):
        var p = _clamp(probs[r])
        if labels[r] > 0.5:
            total -= log(p)
        else:
            total -= log(1.0 - p)
    return total / Float64(len(probs))


def binary_accuracy(
    probs: List[Float64], labels: List[Float64], threshold: Float64 = 0.5
) raises -> Float64:
    """Fraction of rows where thresholded probability matches the label."""
    _check_same_length(len(probs), len(labels))
    var correct = 0
    for r in range(len(probs)):
        var predicted = 1.0 if probs[r] >= threshold else 0.0
        if abs(predicted - labels[r]) < 0.5:
            correct += 1
    return Float64(correct) / Float64(len(probs))


def multiclass_log_loss(
    probs: List[Float64], labels: List[Int], n_classes: Int
) raises -> Float64:
    """Mean negative log likelihood of the true class, row-major probs."""
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if len(probs) != len(labels) * n_classes:
        raise Error("probs must have n_rows * n_classes entries")
    if len(labels) == 0:
        raise Error("metrics need at least one row")
    var total = 0.0
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        total -= log(_clamp(probs[r * n_classes + labels[r]]))
    return total / Float64(len(labels))


def multiclass_accuracy(
    probs: List[Float64], labels: List[Int], n_classes: Int
) raises -> Float64:
    """Fraction of rows where the argmax class matches the label."""
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    if len(probs) != len(labels) * n_classes:
        raise Error("probs must have n_rows * n_classes entries")
    if len(labels) == 0:
        raise Error("metrics need at least one row")
    var correct = 0
    for r in range(len(labels)):
        var argmax = 0
        for k in range(1, n_classes):
            if probs[r * n_classes + k] > probs[r * n_classes + argmax]:
                argmax = k
        if argmax == labels[r]:
            correct += 1
    return Float64(correct) / Float64(len(labels))


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
    scores: List[Float64], labels: List[Float64]
) raises -> Float64:
    """Area under the ROC curve via the rank statistic.

    Works on any monotone score (raw margin or probability). Ties receive
    average ranks, matching sklearn's roc_auc_score.
    """
    _check_same_length(len(scores), len(labels))
    var n = len(scores)
    var n_pos = 0
    for r in range(n):
        if labels[r] > 0.5:
            n_pos += 1
    var n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise Error("AUC needs at least one positive and one negative")

    var order = _argsort(scores)
    var rank_sum_pos = 0.0
    var i = 0
    while i < n:
        var j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        # Average 1-based rank of the tied block [i, j].
        var avg_rank = Float64(i + j + 2) / 2.0
        for t in range(i, j + 1):
            if labels[order[t]] > 0.5:
                rank_sum_pos += avg_rank
        i = j + 1

    var pos = Float64(n_pos)
    return (rank_sum_pos - pos * (pos + 1.0) / 2.0) / (pos * Float64(n_neg))
