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

Where these facts belong
------------------------
Names, aliases, directions, tasks, and per-objective defaults are *not*
Python's to know. They are one registry in Mojo,
src/mojoboost/objective_registry.mojo, alongside the objective facts that
go with them (links, scalar parameters, leaf renewal, backend support), so
that a metric added to metrics.mojo cannot be invisible from Python and a
direction cannot disagree with the code that computes the number.

This module is on its way to being a facade over that registry. Until the
binding exists, the tables below are a *mirror* of it, isolated in one
clearly marked block so that wiring is a deletion rather than a rewrite:
everything public here already goes through `_TABLE`, whose five methods
are one-for-one with registry queries. `_CompatTable` answers them from the
mirrored dicts today; `_NativeTable` will answer them from
`_mojoboost.objective_registry_*` and the dicts go away. The mapping is:

    _TABLE.metric_code(name)         metric_code_from_name
    _TABLE.metric_task(code)         metric_task
    _TABLE.higher_is_better(code)    metric_higher_is_better
    _TABLE.names_for_task(task)      metric_names_for_task
    _TABLE.default_metric(task, o)   objective_default_metric

What stays in Python after that: turning a Python callable into a metric
spec (python/mojoboost/__init__.py), and formatting the ValueErrors below.
Both are presentation, not semantics.

handoffs/migration_21_objective_metric_registry.md carries the binding API
this needs, the lines that get deleted, and the disagreements found between
this file, the Mojo tables, the README, and docs/LIGHTGBM_PARITY.md.
"""

# ===========================================================================
# Compatibility table. Mirrors src/mojoboost/objective_registry.mojo.
#
# Nothing outside this block reads these dicts: `_CompatTable` below is the
# only consumer, so deleting the block means deleting `_CompatTable` with
# it. Do not add a fact here that the registry does not carry, and do not
# change one here alone.
# ===========================================================================

# Metric codes: the mirror of the table in bindings/_mojoboost.mojo and of
# `METRIC_*` in src/mojoboost/objective_registry.mojo. The three are one
# contract and must move together.
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

# Task names, `objective_registry.task_name` spelling for spelling.
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
#:
#: Keyed by the objective *name the user typed*, which is why an objective
#: alias missing from this table has no default even though the objective
#: itself does; the registry keys the same rule on the objective code and so
#: has no such hole. Closing it changes behavior, so it is a handoff item
#: rather than an edit here.
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

#: The task defaults that do not depend on the objective name: a classifier
#: scores its log loss and a ranker its NDCG whatever spelling reached them.
_TASK_DEFAULTS = {
    BINARY: "binary_logloss",
    MULTICLASS: "multi_logloss",
    RANKING: "ndcg",
}


class _CompatTable:
    """The registry queries, answered from the mirrored dicts above.

    One method per registry function, taking and returning what the native
    call will take and return, so `_NativeTable` can replace this class
    without touching a caller. `metric_code` returns None rather than
    raising for an unknown name because the caller builds the message.
    """

    def metric_code(self, canonical):
        spec = _METRICS.get(canonical)
        return None if spec is None else spec[0]

    def canonical_name(self, name):
        """The alias resolution, which the native side folds into
        `metric_code_from_name`; kept separate here because the ValueError
        below names the canonical spelling, not the one that was typed."""
        return _ALIASES.get(name, name)

    def metric_task(self, code):
        for _, (metric_code, _higher, task) in _METRICS.items():
            if metric_code == code:
                return task
        raise ValueError(f"unknown metric code {code!r}")

    def higher_is_better(self, code):
        for _, (metric_code, higher, _task) in _METRICS.items():
            if metric_code == code:
                return higher
        raise ValueError(f"unknown metric code {code!r}")

    def names_for_task(self, task):
        return sorted(
            name for name, spec in _METRICS.items() if spec[2] == task
        )

    def default_metric(self, task, objective):
        """The metric name for a task and objective, or None when the
        objective has no default loss to score."""
        by_task = _TASK_DEFAULTS.get(task)
        if by_task is not None:
            return by_task
        if callable(objective):
            return None
        return _DEFAULTS.get(objective)


#: The one indirection. Swapping this for a `_NativeTable()` is the whole
#: of the Python side of the migration.
_TABLE = _CompatTable()

# ===========================================================================
# Facade. Everything below is presentation: normalizing what a user typed
# and turning a registry answer into the ValueError that names what this
# estimator does accept.
# ===========================================================================


def task_metrics(task):
    """The metric names an estimator of this task accepts, sorted."""
    return _TABLE.names_for_task(task)


def resolve(name, task):
    """`(canonical_name, code, higher_is_better)` for a metric name, or a
    ValueError naming what this task does accept."""
    key = str(name).strip().lower()
    canonical = _TABLE.canonical_name(key)
    code = _TABLE.metric_code(canonical)
    if code is None:
        raise ValueError(
            f"unknown eval_metric {name!r}; expected one of "
            + ", ".join(task_metrics(task))
            + ", or a callable"
        )
    metric_task = _TABLE.metric_task(code)
    if metric_task != task:
        raise ValueError(
            f"eval_metric {canonical!r} scores {metric_task} models; this "
            "one takes " + ", ".join(task_metrics(task)) + ", or a callable"
        )
    return canonical, code, _TABLE.higher_is_better(code)


def default_metric(task, objective=None):
    """The metric to score when `eval_metric` is not given.

    LightGBM's rule: the objective's own loss. A callable objective has
    none, and says so rather than picking one.
    """
    name = _TABLE.default_metric(task, objective)
    if name is not None:
        return name
    if callable(objective):
        raise ValueError(
            "eval_set with a Python objective callback needs an explicit "
            "eval_metric: a custom objective has no default loss to score"
        )
    raise ValueError(
        f"no default eval_metric for objective {objective!r}; pass "
        "eval_metric explicitly"
    )
