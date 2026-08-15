"""Objective and metric registry queries, for the Python facade.

`src/mojotrees/objective_registry.mojo` is the one native registry of what
objectives and metrics exist, what they are called, what they need, what
they optimize, and which backends can train them. Python holds a second
copy of most of it today (`_METRICS`, `_ALIASES`, `_DEFAULTS` and
`_TASK_DEFAULTS` in `python/mojotrees/_eval.py`, `_OBJECTIVES` and
`_UNIMPLEMENTED_OBJECTIVES` in `python/mojotrees/__init__.py`). These
functions are the seam that lets those tables be deleted:
`handoffs/migration_21_objective_metric_registry.md` section 4 specified
them and this module implements that specification.

Read this once, at `_eval` import, and cache the result. Not once per
lookup: `resolve()` runs inside `fit`, and a snapshot taken at import is
also what makes "does the Python table agree with the registry" a
one-line differential rather than a per-name loop across the boundary.

A cached snapshot is not a second table. It is derived from the registry
at import, never edited, and nothing that could disagree with it can
reach it.

Two conventions the snapshot keeps, both from the handoff:

- An unknown name is *absent from the alias list*, never a sentinel code.
  "Unknown objective" stays a Python message about a Python argument.
- Records cross as lists, in the field order documented on each function.
  A consumer reads them positionally; adding a field is a change to this
  module's contract and to `_eval`'s reader in the same commit.

Nothing here decides anything. Every value is read out of the registry.
"""

from std.python import Python, PythonObject

from binding_support import py_dict, py_pair

from mojotrees.boosting import CUSTOM
from mojotrees.objective_registry import (
    N_BUILTIN_METRICS,
    NAME_SUPPORTED,
    NAME_UNIMPLEMENTED,
    NAME_UNKNOWN,
    all_objective_codes,
    check_objective_param as registry_check_objective_param,
    metric_alias_names,
    metric_canonical_name,
    metric_code_from_name,
    metric_spec,
    objective_alias_names,
    objective_canonical_name,
    objective_code_from_name,
    objective_default_metric,
    objective_name_status,
    objective_param_name,
    objective_spec,
    objective_unimplemented_canonical,
    objective_unimplemented_reason,
    task_name,
    unimplemented_objective_alias_names,
)


# -- objectives ----------------------------------------------------------


def _objective_record(objective: Int) raises -> PythonObject:
    """One objective as the documented list:

    `[code, canonical_name, task_name, link, param_name, default_param,
      renews_leaves, multi_output, needs_groups, gradients_on_device,
      backends, builtin, default_metric_code]`

    `default_metric_code` is -1 for the one objective with no default, the
    custom one, whose author is the only party that knows what it
    optimizes. Every other field comes straight off `objective_spec`.
    """
    var spec = objective_spec(objective)
    # `objective_default_metric` raises for the custom objective rather than
    # picking one, and -1 is how this boundary spells "there is none". The
    # branch is on the code rather than on an exception because that is the
    # one objective it can be, and a `try` here would also swallow the
    # "unknown code" error, which is not this function's to hide.
    var default_metric: Int = -1
    if objective != CUSTOM:
        default_metric = objective_default_metric(objective)
    var out = Python.list()
    out.append(PythonObject(spec.code))
    out.append(PythonObject(objective_canonical_name(objective)))
    out.append(PythonObject(task_name(spec.task)))
    out.append(PythonObject(spec.link))
    out.append(PythonObject(objective_param_name(objective)))
    out.append(PythonObject(spec.default_param))
    out.append(PythonObject(spec.renews_leaves))
    out.append(PythonObject(spec.multi_output))
    out.append(PythonObject(spec.needs_groups))
    out.append(PythonObject(spec.gradients_on_device))
    out.append(PythonObject(spec.backends))
    out.append(PythonObject(spec.builtin))
    out.append(PythonObject(default_metric))
    return out^


def registry_objectives() raises -> PythonObject:
    """Every objective the trainer implements, one record per code.

    The list `all_objective_codes` walks, built-ins first and then the
    three with trainers of their own. See `_objective_record` for the field
    order.
    """
    var codes = all_objective_codes()
    var out = Python.list()
    for i in range(len(codes)):
        out.append(_objective_record(codes[i]))
    return out^


def registry_objective_aliases() raises -> PythonObject:
    """Every objective spelling the registry resolves, as `[alias, code]`
    pairs. Replaces `MojoTreesRegressor._OBJECTIVES`."""
    var names = objective_alias_names()
    var out = Python.list()
    for i in range(len(names)):
        out.append(
            py_pair(
                PythonObject(names[i]),
                PythonObject(objective_code_from_name(names[i])),
            )
        )
    return out^


def registry_objective_unimplemented() raises -> PythonObject:
    """Every LightGBM objective mojotrees reports by name as not
    implemented, as `[alias, canonical, reason]`.

    Python keeps its own sentence about which estimator to use instead,
    because that is a Python fact; the reason comes from here, because it
    is not.
    """
    var names = unimplemented_objective_alias_names()
    var out = Python.list()
    for i in range(len(names)):
        var record = Python.list()
        record.append(PythonObject(names[i]))
        record.append(
            PythonObject(objective_unimplemented_canonical(names[i]))
        )
        record.append(PythonObject(objective_unimplemented_reason(names[i])))
        out.append(record^)
    return out^


# -- metrics -------------------------------------------------------------


def registry_metrics() raises -> PythonObject:
    """Every built-in metric, one record per code 0..N-1:

    `[code, canonical_name, task_name, higher_is_better, needs, transform]`

    Replaces `_METRICS` in `python/mojotrees/_eval.py`.
    """
    var out = Python.list()
    for code in range(N_BUILTIN_METRICS):
        var spec = metric_spec(code)
        var record = Python.list()
        record.append(PythonObject(spec.code))
        record.append(PythonObject(metric_canonical_name(code)))
        record.append(PythonObject(task_name(spec.task)))
        record.append(PythonObject(spec.higher_is_better))
        record.append(PythonObject(spec.needs))
        record.append(PythonObject(spec.transform))
        out.append(record^)
    return out^


def registry_metric_aliases() raises -> PythonObject:
    """Every metric spelling the registry resolves, as `[alias, code]`
    pairs. Replaces `_ALIASES` in `python/mojotrees/_eval.py`."""
    var names = metric_alias_names()
    var out = Python.list()
    for i in range(len(names)):
        out.append(
            py_pair(
                PythonObject(names[i]),
                PythonObject(metric_code_from_name(names[i])),
            )
        )
    return out^


def registry_vocabulary() raises -> PythonObject:
    """The scalar vocabularies the records above are written in, so a
    consumer reads names rather than restating the numbers.

    Keys: `n_builtin_metrics`, and the three name-status codes a resolver
    can report (`name_supported`, `name_unimplemented`, `name_unknown`).
    """
    var out = py_dict()
    out["n_builtin_metrics"] = PythonObject(N_BUILTIN_METRICS)
    out["name_supported"] = PythonObject(NAME_SUPPORTED)
    out["name_unimplemented"] = PythonObject(NAME_UNIMPLEMENTED)
    out["name_unknown"] = PythonObject(NAME_UNKNOWN)
    return out^


# -- single lookups ------------------------------------------------------
#
# For the callers that hold one name and no snapshot. They resolve through
# the same registry the snapshot is built from, so a build with both
# cannot answer two ways.


def objective_code_of_name(name: PythonObject) raises -> PythonObject:
    """The objective code a public objective name denotes.

    Named for what it takes, because `objective_code` is the accessor that
    takes a *model handle* (see `inspection_bindings`). The two are
    different questions and shared one name in two earlier plans; see the
    handoff.

    Raises for a name the registry does not resolve, including a real
    LightGBM objective mojotrees has not implemented, whose message names
    it and says why.
    """
    return PythonObject(objective_code_from_name(String(py=name)))


def metric_code_of_name(name: PythonObject) raises -> PythonObject:
    """The metric code a public metric name denotes. Raises for a name the
    registry does not resolve."""
    return PythonObject(metric_code_from_name(String(py=name)))


def objective_name_status_of(name: PythonObject) raises -> PythonObject:
    """Whether a name is implemented, a real LightGBM objective that is
    not, or nothing the registry knows. The three codes are in
    `registry_vocabulary`. Does not raise: this is the question a caller
    asks *before* deciding whether an exception is warranted."""
    return PythonObject(objective_name_status(String(py=name)))


def check_objective_param(
    objective: PythonObject, value: PythonObject
) raises -> PythonObject:
    """Validate an objective's scalar parameter (`alpha`, `fair_c`, or
    `tweedie_variance_power`) without looking at data.

    The trainer's own range checks, so there is no second copy of the
    ranges in Python to drift from them. Returns None and raises with the
    trainer's message.
    """
    registry_check_objective_param(
        Int(py=objective), Float64(py=value)
    )
    return PythonObject(None)
