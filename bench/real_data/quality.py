"""One implementation of every quality metric, applied to both engines.

This is the rule that makes the harness differential rather than
anecdotal: no engine's self-reported metric is ever compared against
another engine's self-reported metric. Both engines are asked only for
predictions, and the numbers that go into a result record are computed
here, by this code, from those predictions.

The reason is that a metric name does not pin down a metric. Log loss
differs in its clipping, AUC differs in its tie handling, NDCG differs in
what it does with a query that has no relevant document, and average
precision differs in whether it interpolates. Any of those is worth a
few thousandths, which is the same size as the differences the harness is
trying to detect. Computing both sides here removes the question.

Conventions, all deliberate and all recorded in the result:

- Probabilities are clipped to [1e-15, 1 - 1e-15] before any logarithm.
- AUC uses average ranks for tied scores, which is the Mann-Whitney
  estimator and matches scikit-learn.
- Average precision is the step-wise sum used by scikit-learn, not the
  interpolated eleven-point version.
- NDCG uses LightGBM's gains, 2**label - 1, and 1 / log2(rank + 2)
  discounts. A query with no positive label has no defined NDCG and is
  skipped by default rather than counted as a perfect 1.0.

`scipy` and `scikit-learn` are not imported here on purpose. Fewer moving
parts between the predictions and the number.
"""

import numpy as np

EPS = 1e-15


def _clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


def rmse(y_true, y_pred, weight=None):
    err = (np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) ** 2
    return float(np.sqrt(np.average(err, weights=weight)))


def mae(y_true, y_pred, weight=None):
    err = np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
    return float(np.average(err, weights=weight))


def r2(y_true, y_pred, weight=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    resid = np.average((y_true - np.asarray(y_pred, dtype=np.float64)) ** 2, weights=weight)
    total = np.average((y_true - np.average(y_true, weights=weight)) ** 2, weights=weight)
    return float(1.0 - resid / total) if total > 0 else float("nan")


def logloss(y_true, y_prob, weight=None):
    p = _clip(y_prob)
    y = np.asarray(y_true, dtype=np.float64)
    return float(np.average(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)), weights=weight))


def multi_logloss(y_true, y_prob, weight=None):
    """`y_prob` is (n_rows, n_classes) and `y_true` holds class indices."""
    p = _clip(y_prob)
    p = p / p.sum(axis=1, keepdims=True)
    rows = np.arange(len(y_true))
    picked = p[rows, np.asarray(y_true, dtype=np.int64)]
    return float(np.average(-np.log(picked), weights=weight))


def accuracy(y_true, y_pred_label, weight=None):
    hit = (np.asarray(y_pred_label) == np.asarray(y_true)).astype(np.float64)
    return float(np.average(hit, weights=weight))


def error_rate(y_true, y_pred_label, weight=None):
    return 1.0 - accuracy(y_true, y_pred_label, weight)


def _average_ranks(scores):
    """Ranks 1..n with ties averaged, which is what the rank formulation of
    AUC needs to be exact in the presence of tied scores."""
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auc(y_true, y_score):
    """Binary ROC AUC by the rank formulation. Undefined and returned as
    NaN when one class is absent, which is a real possibility on a small
    validation slice of an imbalanced problem."""
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = float((y > 0.5).sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true, y_score):
    """Area under the precision-recall curve, summed step-wise.

    On a badly imbalanced problem this is the metric that moves. AUC can sit
    at 0.99 while the model is useless at the top of the ranking; average
    precision cannot.
    """
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = float((y > 0.5).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable")
    hits = (y[order] > 0.5).astype(np.float64)
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, len(y) + 1, dtype=np.float64)
    return float((precision * hits).sum() / n_pos)


def ndcg(y_true, y_score, group, k=10, empty_query="skip"):
    """Mean NDCG at `k` over queries, with LightGBM's gains and discounts.

    `group` is the per-query row count, in row order. `empty_query` decides
    what a query with no positive label contributes: "skip" leaves it out of
    the mean, "one" counts it as 1.0 the way LightGBM's own metric does. The
    default is "skip" because a query nobody can get wrong should not raise
    the score, and because whichever choice is made it is applied to both
    engines identically.
    """
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(y_score, dtype=np.float64)
    starts = np.concatenate(([0], np.cumsum(np.asarray(group, dtype=np.int64))))
    if int(starts[-1]) != len(y):
        raise ValueError(
            f"group sums to {int(starts[-1])} rows but there are {len(y)}"
        )
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))

    total, counted = 0.0, 0
    for q in range(len(starts) - 1):
        lo, hi = int(starts[q]), int(starts[q + 1])
        labels = y[lo:hi]
        if not (labels > 0).any():
            if empty_query == "one":
                total += 1.0
                counted += 1
            continue
        top = min(k, hi - lo)
        gains = np.exp2(labels) - 1.0
        ranked = gains[np.argsort(-s[lo:hi], kind="stable")][:top]
        ideal = np.sort(gains)[::-1][:top]
        denom = float((ideal * discounts[:top]).sum())
        if denom > 0:
            total += float((ranked * discounts[:top]).sum()) / denom
            counted += 1
    if counted == 0:
        return float("nan")
    return total / counted


#: What each task is scored on. The first entry is the primary metric, the
#: one thresholds.json gates and reports lead with; the rest are recorded
#: because they answer different questions about the same model.
TASK_METRICS = {
    "regression": ("rmse", "mae", "r2"),
    "binary": ("auc", "average_precision", "logloss"),
    "multiclass": ("multi_logloss", "accuracy"),
    "ranking": ("ndcg@10", "ndcg@5", "ndcg@1"),
}

#: Whether a larger value is better. verify.py needs this to know which
#: direction of a difference is a regression and which is an improvement.
HIGHER_IS_BETTER = {
    "rmse": False,
    "mae": False,
    "r2": True,
    "logloss": False,
    "auc": True,
    "average_precision": True,
    "multi_logloss": False,
    "accuracy": True,
    "error_rate": False,
    "ndcg@1": True,
    "ndcg@3": True,
    "ndcg@5": True,
    "ndcg@10": True,
}


def score(task, y_true, prediction, group=None, weight=None):
    """Every metric for `task`, as a dict.

    `prediction` is whatever the engine's `predict` returned on the
    response scale: a vector for regression, positive-class probabilities
    for binary, an (n_rows, n_classes) matrix for multiclass, and a score
    per document for ranking.
    """
    if task == "regression":
        return {
            "rmse": rmse(y_true, prediction, weight),
            "mae": mae(y_true, prediction, weight),
            "r2": r2(y_true, prediction, weight),
        }
    if task == "binary":
        return {
            "auc": auc(y_true, prediction),
            "average_precision": average_precision(y_true, prediction),
            "logloss": logloss(y_true, prediction, weight),
        }
    if task == "multiclass":
        probs = np.asarray(prediction, dtype=np.float64)
        if probs.ndim != 2:
            raise ValueError("multiclass predictions must be (n_rows, n_classes)")
        return {
            "multi_logloss": multi_logloss(y_true, probs, weight),
            "accuracy": accuracy(y_true, probs.argmax(axis=1), weight),
        }
    if task == "ranking":
        return {
            "ndcg@10": ndcg(y_true, prediction, group, 10),
            "ndcg@5": ndcg(y_true, prediction, group, 5),
            "ndcg@1": ndcg(y_true, prediction, group, 1),
        }
    raise ValueError(f"unknown task {task!r}")


def trivial_baseline(task, y_train, y_test, group_test=None):
    """The score of the least interesting model that could be fitted: the
    training mean for regression, the training positive rate for binary and
    multiclass, and a constant score for ranking.

    verify.py uses this as a floor. Two engines that agree because both
    produced rubbish would pass a pure agreement check, so agreement is not
    enough: each engine also has to beat the baseline by the margin in
    thresholds.json.
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    if task == "regression":
        const = np.full(len(y_test), y_train.mean())
        return score("regression", y_test, const)
    if task == "binary":
        const = np.full(len(y_test), float((y_train > 0.5).mean()))
        return score("binary", y_test, const)
    if task == "multiclass":
        n_classes = int(y_train.max()) + 1
        prior = np.bincount(y_train.astype(np.int64), minlength=n_classes).astype(np.float64)
        prior /= prior.sum()
        const = np.tile(prior, (len(y_test), 1))
        return score("multiclass", y_test, const)
    if task == "ranking":
        # A constant score leaves every query in its stored order, which is
        # the honest "no model" ranking baseline for these files.
        return score("ranking", y_test, np.zeros(len(y_test)), group_test)
    raise ValueError(f"unknown task {task!r}")
