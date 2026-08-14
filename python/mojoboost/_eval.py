"""Built-in validation metrics for the Python estimators.

`eval_metric` takes LightGBM's metric names as well as callables. A name
resolves here to a code, a direction, and the task it belongs to; the value
itself is computed by `_mojoboost.eval_metric`, which calls the same
functions in src/mojoboost/metrics.mojo that the Mojo API exposes. Nothing
in this module computes a metric, so the Python and Mojo answers cannot
drift apart.

Names, aliases, and directions follow LightGBM:

    l2          mean_squared_error, mse, regression_l2, regression   lower
    rmse        root_mean_squared_error, l2_root                     lower
    l1          mean_absolute_error, mae, regression_l1              lower
    quantile                                                         lower
    huber                                                            lower
    binary_logloss  binary                                           lower
    binary_error                                                     lower
    auc                                                              higher
    multi_logloss   multiclass, softmax                              lower
    multi_error                                                      lower
    ndcg        lambdarank                                           higher

Each metric belongs to one task, and an estimator only accepts its own:
the regressor takes the regression metrics, the classifier takes the binary
ones for two classes and the multiclass ones beyond that, and the ranker
takes `ndcg`. That is stricter than LightGBM, which accepts any name and
scores whatever it can; here a metric that cannot mean anything for the
model being fitted is a mistake worth reporting. Pass a callable when you
want something else.

`quantile` and `huber` read the estimator's `alpha`, so they score the loss
the objective was actually trained on. `ndcg` reads the ranker's
`ndcg_eval_at` and each validation set's own `group`.
"""

# Metric codes: the mirror of the table in bindings/_mojoboost.mojo. The two
# are one contract and must move together.
L2 = 0
RMSE = 1
L1 = 2
QUANTILE = 3
HUBER = 4
BINARY_LOGLOSS = 5
BINARY_ERROR = 6
AUC = 7
MULTI_LOGLOSS = 8
MULTI_ERROR = 9
NDCG = 10

REGRESSION = "regression"
BINARY = "binary"
MULTICLASS = "multiclass"
RANKING = "ranking"

#: canonical name -> (code, higher_is_better, task)
_METRICS = {
    "l2": (L2, False, REGRESSION),
    "rmse": (RMSE, False, REGRESSION),
    "l1": (L1, False, REGRESSION),
    "quantile": (QUANTILE, False, REGRESSION),
    "huber": (HUBER, False, REGRESSION),
    "binary_logloss": (BINARY_LOGLOSS, False, BINARY),
    "binary_error": (BINARY_ERROR, False, BINARY),
    "auc": (AUC, True, BINARY),
    "multi_logloss": (MULTI_LOGLOSS, False, MULTICLASS),
    "multi_error": (MULTI_ERROR, False, MULTICLASS),
    "ndcg": (NDCG, True, RANKING),
}

#: alias -> canonical name
_ALIASES = {
    "mean_squared_error": "l2",
    "mse": "l2",
    "regression_l2": "l2",
    "regression": "l2",
    "root_mean_squared_error": "rmse",
    "l2_root": "rmse",
    "mean_absolute_error": "l1",
    "mae": "l1",
    "regression_l1": "l1",
    "binary": "binary_logloss",
    "multiclass": "multi_logloss",
    "softmax": "multi_logloss",
    "lambdarank": "ndcg",
}

#: The metric LightGBM scores when none is named, by objective. A custom
#: objective has no default: only its author knows what it optimizes.
_DEFAULTS = {
    "regression": "l2",
    "huber": "huber",
    "quantile": "quantile",
    "mae": "l1",
    "regression_l1": "l1",
}


def task_metrics(task):
    """The metric names an estimator of this task accepts, sorted."""
    return sorted(
        name for name, spec in _METRICS.items() if spec[2] == task
    )


def resolve(name, task):
    """`(canonical_name, code, higher_is_better)` for a metric name, or a
    ValueError naming what this task does accept."""
    key = str(name).strip().lower()
    canonical = _ALIASES.get(key, key)
    spec = _METRICS.get(canonical)
    if spec is None:
        raise ValueError(
            f"unknown eval_metric {name!r}; expected one of "
            + ", ".join(task_metrics(task))
            + ", or a callable"
        )
    code, higher, metric_task = spec
    if metric_task != task:
        raise ValueError(
            f"eval_metric {canonical!r} scores {metric_task} models; this "
            "one takes " + ", ".join(task_metrics(task)) + ", or a callable"
        )
    return canonical, code, higher


def default_metric(task, objective=None):
    """The metric to score when `eval_metric` is not given.

    LightGBM's rule: the objective's own loss. A callable objective has
    none, and says so rather than picking one.
    """
    if task == BINARY:
        return "binary_logloss"
    if task == MULTICLASS:
        return "multi_logloss"
    if task == RANKING:
        return "ndcg"
    if callable(objective):
        raise ValueError(
            "eval_set with a Python objective callback needs an explicit "
            "eval_metric: a custom objective has no default loss to score"
        )
    name = _DEFAULTS.get(objective)
    if name is None:
        raise ValueError(
            f"no default eval_metric for objective {objective!r}; pass "
            "eval_metric explicitly"
        )
    return name
