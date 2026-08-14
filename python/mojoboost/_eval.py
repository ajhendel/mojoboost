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
    mape        mean_absolute_percentage_error                       lower
    fair                                                             lower
    poisson                                                          lower
    gamma                                                            lower
    gamma_deviance  gamma_dev                                        lower
    tweedie                                                          lower
    cross_entropy   xentropy                                         lower
    kullback_leibler  kldiv                                          lower
    binary_logloss  binary                                           lower
    binary_error                                                     lower
    auc                                                              higher
    average_precision                                                higher
    multi_logloss   multiclass, softmax                              lower
    multi_error                                                      lower
    ndcg        lambdarank                                           higher
    map         mean_average_precision                               higher

Each metric belongs to one task, and an estimator only accepts its own:
the regressor takes the regression metrics, the classifier takes the binary
ones for two classes and the multiclass ones beyond that, and the ranker
takes `ndcg` and `map`. That is stricter than LightGBM, which accepts any
name and scores whatever it can; here a metric that cannot mean anything
for the model being fitted is a mistake worth reporting. Pass a callable
when you want something else.

`cross_entropy` and `kullback_leibler` sit under the regression task rather
than the binary one because the objective they belong to does: soft labels
anywhere in [0, 1] are the regressor's business, since the classifier's
labels are classes.

`quantile`, `huber`, `fair`, and `tweedie` read the estimator's `alpha`
slot, so they score the loss the objective was actually trained on; see the
`alpha` / `fair_c` / `tweedie_variance_power` discussion in the estimator
docstrings. `ndcg` and `map` read the ranker's `ndcg_eval_at` (LightGBM's
one `eval_at` serves both) and each validation set's own `group`.

Predictions are transformed by the *objective's* inverse link before any
metric sees them, so `l2` on a poisson model scores expected counts and
`binary_logloss` scores probabilities. The transform belongs to the
objective, as in LightGBM; a metric never applies a second one.
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
MAPE = 11
FAIR = 12
POISSON = 13
GAMMA = 14
GAMMA_DEVIANCE = 15
TWEEDIE = 16
CROSS_ENTROPY = 17
KLDIV = 18
AVERAGE_PRECISION = 19
MAP = 20

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
    "mape": (MAPE, False, REGRESSION),
    "fair": (FAIR, False, REGRESSION),
    "poisson": (POISSON, False, REGRESSION),
    "gamma": (GAMMA, False, REGRESSION),
    "gamma_deviance": (GAMMA_DEVIANCE, False, REGRESSION),
    "tweedie": (TWEEDIE, False, REGRESSION),
    "cross_entropy": (CROSS_ENTROPY, False, REGRESSION),
    "kullback_leibler": (KLDIV, False, REGRESSION),
    "binary_logloss": (BINARY_LOGLOSS, False, BINARY),
    "binary_error": (BINARY_ERROR, False, BINARY),
    "auc": (AUC, True, BINARY),
    "average_precision": (AVERAGE_PRECISION, True, BINARY),
    "multi_logloss": (MULTI_LOGLOSS, False, MULTICLASS),
    "multi_error": (MULTI_ERROR, False, MULTICLASS),
    "ndcg": (NDCG, True, RANKING),
    "map": (MAP, True, RANKING),
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
    "mean_absolute_percentage_error": "mape",
    "gamma_dev": "gamma_deviance",
    "xentropy": "cross_entropy",
    "kldiv": "kullback_leibler",
    "binary": "binary_logloss",
    "multiclass": "multi_logloss",
    "softmax": "multi_logloss",
    "lambdarank": "ndcg",
    "mean_average_precision": "map",
}

#: The metric LightGBM scores when none is named, by objective. A custom
#: objective has no default: only its author knows what it optimizes.
_DEFAULTS = {
    "regression": "l2",
    "huber": "huber",
    "quantile": "quantile",
    "mae": "l1",
    "regression_l1": "l1",
    "poisson": "poisson",
    "gamma": "gamma",
    "tweedie": "tweedie",
    "mape": "mape",
    "fair": "fair",
    "cross_entropy": "cross_entropy",
    "xentropy": "cross_entropy",
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
