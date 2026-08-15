"""Built-in validation metrics for the Python estimators.

`eval_metric` takes LightGBM's metric names as well as callables. A name
resolves here to a code, a direction, and the task it belongs to; the value
itself is computed by `_mojotrees.eval_metric`, which calls the same
functions in src/mojotrees/metrics.mojo that the Mojo API exposes. Nothing
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

Names, aliases, directions, tasks, and per-objective defaults are *not*
Python's to know. They are one registry in Mojo,
src/mojotrees/objective_registry.mojo, alongside the objective facts that
go with them (links, scalar parameters, leaf renewal, backend support), so
that a metric added to metrics.mojo cannot be invisible from Python and a
direction cannot disagree with the code that computes the number.

This module is a facade over that registry. `_TABLE` is one snapshot of it,
taken at import through `_mojotrees.registry_metrics`,
`registry_metric_aliases`, `registry_objectives`, and
`registry_objective_aliases`, and its methods are one-for-one with
registry queries:

    _TABLE.metric_code(name)         metric_code_from_name
    _TABLE.canonical_name(name)      (the alias half of the same query)
    _TABLE.metric_task(code)         metric_task
    _TABLE.higher_is_better(code)    metric_higher_is_better
    _TABLE.names_for_task(task)      metric_names_for_task
    _TABLE.default_metric(task, o)   objective_default_metric

`resolve()` asks the registry itself for the one name it holds, through
`_mojotrees.metric_code_of_name`, and reads the rest off the snapshot; the
two cannot disagree because the snapshot is derived from the same
registry. There is no Python mirror of any of it any more: the mirrored
tables this module carried while the binding was landing were compared
against the native answers, spelling for spelling, and deleted. A build
whose extension cannot answer the registry queries fails at import with
a message that says so, rather than answering from a copy that could
drift.

The metric code constants below (`L2` ... `MAP`) are read off the same
snapshot, so they are names for the registry's numbers, not a second
statement of them. `REGRESSION`, `BINARY`, `MULTICLASS`, `RANKING` are the
task spellings `objective_registry.task_name` uses.

What stays in Python: turning a Python callable into a metric spec
(python/mojotrees/_fit_args.py), and formatting the ValueErrors below.
Both are presentation, not semantics.
"""

# Task names, `objective_registry.task_name` spelling for spelling.
REGRESSION = "regression"
BINARY = "binary"
MULTICLASS = "multiclass"
RANKING = "ranking"


# ===========================================================================
# The native table. Reads the registry, holds no facts of its own.
# ===========================================================================

#: The four registry snapshot entry points, plus the single lookup
#: `resolve()` makes per name. All are required of the extension.
_REGISTRY_HOOKS = (
    "registry_metrics",
    "registry_metric_aliases",
    "registry_objectives",
    "registry_objective_aliases",
    "metric_code_of_name",
)

#: Field positions in the registry tuples, so a reader can check them
#: against section 4 without counting.
_METRIC_CODE, _METRIC_NAME, _METRIC_TASK, _METRIC_HIGHER = 0, 1, 2, 3
_OBJECTIVE_CODE, _OBJECTIVE_TASK, _OBJECTIVE_DEFAULT = 0, 2, 12

#: `default_metric_code` for the one objective that has no default.
_NO_DEFAULT_METRIC = -1


class _NativeTable:
    """The registry queries, answered from one snapshot of the registry.

    A snapshot is not a second table: it is derived at import from the
    native tuples, never edited, and nothing that could disagree with the
    registry can reach it. Taking it once is deliberate -- `resolve()` runs
    inside `fit`, and a call per lookup would put the boundary in the
    training loop.
    """

    source = "native"

    def __init__(self, metrics, aliases, task_defaults, objective_defaults):
        #: canonical name -> (code, higher_is_better, task)
        self._metrics = metrics
        #: alias -> canonical name (a canonical name maps to itself)
        self._aliases = aliases
        #: task -> canonical metric name, for the tasks whose default does
        #: not depend on which objective spelling reached them
        self._task_defaults = task_defaults
        #: objective name or alias -> canonical metric name
        self._objective_defaults = objective_defaults
        self._by_code = {
            spec[0]: (name, spec[1], spec[2])
            for name, spec in metrics.items()
        }

    def metric_code(self, canonical):
        spec = self._metrics.get(canonical)
        return None if spec is None else spec[0]

    def canonical_name(self, name):
        return self._aliases.get(name, name)

    def metric_task(self, code):
        record = self._by_code.get(code)
        if record is None:
            raise ValueError(f"unknown metric code {code!r}")
        return record[2]

    def higher_is_better(self, code):
        record = self._by_code.get(code)
        if record is None:
            raise ValueError(f"unknown metric code {code!r}")
        return record[1]

    def names_for_task(self, task):
        return sorted(
            name for name, spec in self._metrics.items() if spec[2] == task
        )

    def default_metric(self, task, objective):
        by_task = self._task_defaults.get(task)
        if by_task is not None:
            return by_task
        if callable(objective):
            return None
        return self._objective_defaults.get(objective)


def _native_snapshot():
    """The registry as the dicts `_NativeTable` needs.

    Raises `_RegistryUnavailable` when this build cannot answer the queries
    or answers them inconsistently; `_selected()` turns that into the
    import-time error, since there is no other source to fall back on.
    """
    from . import _mojotrees

    hooks = {}
    for name in _REGISTRY_HOOKS:
        hook = getattr(_mojotrees, name, None)
        if hook is None:
            raise _RegistryUnavailable(f"the extension lacks {name}")
        hooks[name] = hook
    try:
        metric_records = list(hooks["registry_metrics"]())
        metric_aliases = list(hooks["registry_metric_aliases"]())
        objective_records = list(hooks["registry_objectives"]())
        objective_aliases = list(hooks["registry_objective_aliases"]())
    except Exception as exc:
        raise _RegistryUnavailable(f"a registry query failed: {exc}")

    metrics = {}
    name_of_code = {}
    for record in metric_records:
        code = int(record[_METRIC_CODE])
        name = str(record[_METRIC_NAME])
        metrics[name] = (
            code,
            bool(record[_METRIC_HIGHER]),
            str(record[_METRIC_TASK]),
        )
        name_of_code[code] = name
    if not metrics:
        raise _RegistryUnavailable("registry_metrics returned no metrics")

    aliases = {name: name for name in metrics}
    for alias, code in metric_aliases:
        canonical = name_of_code.get(int(code))
        if canonical is None:
            raise _RegistryUnavailable(
                f"metric alias {alias!r} names code {code!r}, which "
                "registry_metrics did not list"
            )
        aliases[str(alias)] = canonical

    # The per-task default, derived rather than mirrored: every builtin
    # objective of a task carries the same `default_metric_code` for the
    # three tasks whose default does not depend on the objective spelling,
    # so "the one all of them agree on" is the registry's own answer. A
    # task whose objectives disagree gets no task default and falls through
    # to the per-objective lookup, which is what the mirror does too.
    per_task = {}
    for record in objective_records:
        task = str(record[_OBJECTIVE_TASK])
        if task == REGRESSION:
            continue
        code = int(record[_OBJECTIVE_DEFAULT])
        if code == _NO_DEFAULT_METRIC:
            continue
        per_task.setdefault(task, set()).add(code)
    task_defaults = {}
    for task, codes in per_task.items():
        if len(codes) != 1:
            continue
        canonical = name_of_code.get(next(iter(codes)))
        if canonical is not None:
            task_defaults[task] = canonical

    default_of_objective = {}
    for record in objective_records:
        code = int(record[_OBJECTIVE_DEFAULT])
        if code != _NO_DEFAULT_METRIC:
            default_of_objective[int(record[_OBJECTIVE_CODE])] = code
    objective_defaults = {}
    for alias, code in objective_aliases:
        metric_code = default_of_objective.get(int(code))
        if metric_code is None:
            continue
        canonical = name_of_code.get(metric_code)
        if canonical is not None:
            objective_defaults[str(alias)] = canonical

    table = _NativeTable(metrics, aliases, task_defaults, objective_defaults)
    table.lookup = hooks["metric_code_of_name"]
    return table


class _RegistryUnavailable(RuntimeError):
    """The compiled registry could not be read; there is no fallback."""


def _selected():
    """The live table: the native registry, or an ImportError that says
    what the extension is missing."""
    try:
        return _native_snapshot()
    except _RegistryUnavailable as exc:
        raise ImportError(
            "mojotrees._mojotrees cannot answer the metric registry queries "
            f"({exc}); the Python package and the extension are different "
            "builds. Rebuild the extension (pixi run build-python)."
        ) from None


#: The one indirection.
_TABLE = _selected()

# The metric codes, as names. Read off the registry so they are the
# registry's numbers; nothing here states a code.
L2 = _TABLE.metric_code("l2")
RMSE = _TABLE.metric_code("rmse")
L1 = _TABLE.metric_code("l1")
QUANTILE = _TABLE.metric_code("quantile")
HUBER = _TABLE.metric_code("huber")
BINARY_LOGLOSS = _TABLE.metric_code("binary_logloss")
BINARY_ERROR = _TABLE.metric_code("binary_error")
AUC = _TABLE.metric_code("auc")
MULTI_LOGLOSS = _TABLE.metric_code("multi_logloss")
MULTI_ERROR = _TABLE.metric_code("multi_error")
NDCG = _TABLE.metric_code("ndcg")
MAPE = _TABLE.metric_code("mape")
FAIR = _TABLE.metric_code("fair")
POISSON = _TABLE.metric_code("poisson")
GAMMA = _TABLE.metric_code("gamma")
GAMMA_DEVIANCE = _TABLE.metric_code("gamma_deviance")
TWEEDIE = _TABLE.metric_code("tweedie")
CROSS_ENTROPY = _TABLE.metric_code("cross_entropy")
KLDIV = _TABLE.metric_code("kullback_leibler")
AVERAGE_PRECISION = _TABLE.metric_code("average_precision")
MAP = _TABLE.metric_code("map")


def registry_source():
    """`"native"`: the metric facts come from the compiled registry. Kept
    for the tests that asserted which of two sources was live while a
    Python mirror existed; there is one source now."""
    return _TABLE.source


def metric_names():
    """Every canonical metric name the registry knows, sorted."""
    return sorted(_TABLE._metrics)


def metric_aliases():
    """Every accepted spelling mapped to its canonical name, aliases and
    canonical names alike, as the registry reports them."""
    return dict(sorted(_TABLE._aliases.items()))


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
    try:
        code = int(_TABLE.lookup(key))
    except Exception:
        code = None
    canonical = _TABLE.canonical_name(key)
    if code is None or _TABLE.metric_code(canonical) != code:
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
