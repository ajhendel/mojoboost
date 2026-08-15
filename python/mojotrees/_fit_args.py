"""Vocabularies and argument helpers the estimators share.

Moved here from the package `__init__` in the consolidation round: the
parameter vocabularies (`_DEVICES`, `_BOOSTING_TYPES`, `_GROW_POLICIES`,
`_IMPORTANCE_TYPES`), the unimplemented-objective notes, and the small
functions `fit` and `predict` use to coerce and check their arguments.
Every name is bound back into the package namespace, so `from mojotrees
import _as_iteration` (basic.py) and `from mojotrees import _metric_specs`
(cv.py) are unchanged.

The objective codes and `_LAMBDA_L1` / `_LAMBDA_L2` live in `sklearn.py`:
tools/api_snapshot.py resolves estimator defaults and `_OBJECTIVES` from
that file's literals. Objective name status, unimplemented-objective
reasons, and scalar-parameter ranges come from the compiled registry
through `_objective_status`, `_unimplemented_objectives`, and
`_check_objective_param`; Python holds no copy of them.
"""

import operator as _operator

from . import _arrays, _compat, _eval

_np = _arrays.np
_mojotrees = _compat.import_extension()
_as_f64_vector = _arrays.f64_vector


_IMPORTANCE_TYPES = {"split": 0, "gain": 1}

_DEVICES = ("cpu", "gpu", "auto")
_BOOSTING_TYPES = ("gbdt", "goss", "dart", "rf")

#: What to say when a caller asks a prediction to run somewhere and this
#: build has no entry point that can. The alternative -- running on the CPU
#: and reporting the device -- is the one outcome an explicit `device=`
#: must never produce, so this is an error and not a warning.
#:
#: The device-aware entry points are `predict_batch`,
#: `predict_proba_batch`, `predict_leaf_batch`, and
#: `predict_leaf_multiclass_batch` in bindings/_mojotrees.mojo. Each takes
#: the requested device in a params dict, asks gpu_predict.mojo whether the
#: GPU path covers the request, resolves through device.mojo, and returns
#: the backend that ran. `_Base._predict_batch` uses one the moment the
#: build has it and this message when it does not.
_NO_DEVICE_PREDICT = (
    "this build predicts on the CPU only: the extension does not expose "
    "%s, so device=%r cannot be honored without silently predicting "
    "somewhere else. Rebuild with the device prediction bindings, or pass "
    "device='cpu' (or leave device unset)."
)

#: LightGBM objectives that resolve here to another estimator: the name is
#: implemented natively, and the sentence says which estimator owns it.
#: This is a Python fact (which class trains what), so it lives here.
_OTHER_ESTIMATOR_OBJECTIVES = {
    "lambdarank": "use MojoTreesRanker, which takes the query groups",
    "multiclass": "use MojoTreesClassifier, which derives the task from y",
    "softmax": "use MojoTreesClassifier, which derives the task from y",
    "binary": "use MojoTreesClassifier, which derives the task from y",
    "ova": "use MojoTreesClassifier, which trains a shared softmax model",
    "ovr": "use MojoTreesClassifier, which trains a shared softmax model",
}


def _unimplemented_objectives():
    """LightGBM objectives mojotrees does not implement, alias -> reason,
    read from the compiled registry (`registry_objective_unimplemented`);
    the reason is the trainer's own sentence. Nothing here restates it."""
    out = {}
    for record in _mojotrees.registry_objective_unimplemented():
        alias, _canonical, reason = (str(v) for v in record)
        out[alias] = reason
    return out


#: The three answers `objective_name_status` gives, by their registry
#: names, so a caller branches on a word rather than a number.
def _objective_status_codes():
    vocab = _mojotrees.registry_vocabulary()
    return {
        "supported": int(vocab["name_supported"]),
        "unimplemented": int(vocab["name_unimplemented"]),
        "unknown": int(vocab["name_unknown"]),
    }


def _objective_status(name):
    """`"supported"`, `"unimplemented"`, or `"unknown"` for an objective
    spelling, from the registry, without raising."""
    if not isinstance(name, str):
        return "unknown"
    code = int(_mojotrees.objective_name_status(name.strip().lower()))
    for word, value in _objective_status_codes().items():
        if value == code:
            return word
    return "unknown"


def _objective_code_of_name(name):
    """The registry's objective code for a spelling, or None when the
    registry does not resolve it (unknown or unimplemented)."""
    if not isinstance(name, str):
        return None
    try:
        return int(_mojotrees.objective_code_of_name(name.strip().lower()))
    except Exception:
        return None


def _check_objective_param(code, value):
    """The trainer's own range check for an objective's scalar parameter
    (`alpha`, `fair_c`, `tweedie_variance_power`), as a ValueError carrying
    the trainer's message. There is no second copy of the ranges here."""
    try:
        _mojotrees.check_objective_param(int(code), float(value))
    except Exception as exc:
        raise ValueError(str(exc)) from None


def _unimplemented_objective_note(objective):
    """The trailing half of an unknown-objective message when the name is a
    LightGBM objective mojotrees does not implement here, or one another
    estimator owns; empty otherwise."""
    if not isinstance(objective, str):
        return ""
    key = objective.strip().lower()
    reasons = [
        r
        for r in (
            _unimplemented_objectives().get(key),
            _OTHER_ESTIMATOR_OBJECTIVES.get(key),
        )
        if r
    ]
    if not reasons:
        return ""
    return f". {objective!r} is not available here: " + "; ".join(reasons)

#: `grow_policy` spellings and the canonical name each resolves to. XGBoost's
#: names, since LightGBM has no such parameter; "lossguide" is XGBoost's word
#: for leaf-wise growth (src/mojotrees/levelwise_policy.mojo).
_GROW_POLICIES = {
    "leafwise": "leafwise",
    "leaf_wise": "leafwise",
    "lossguide": "leafwise",
    "depthwise": "depthwise",
    "depth_wise": "depthwise",
}

# The largest relevance label a ranker accepts, the range of LightGBM's
# default label_gain (src/mojotrees/ranking.mojo).
_MAX_RELEVANCE_LABEL = 30


def _as_iteration(value, name):
    """Coerce a `start_iteration`/`num_iteration` argument to an int.

    `bool` is rejected rather than accepted as 0/1: `num_iteration=True`
    is far more likely to be a misplaced flag than a request for one
    iteration. numpy integers pass through `__index__`."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got a bool")
    try:
        return int(_operator.index(value))
    except TypeError:
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}"
        ) from None


def _store_vector(out, values, n_rows, name):
    """Copy a custom objective's returned array into the buffer the trainer
    reads, checking the length here so the error names the offending array
    instead of surfacing as a generic shape error."""
    if _np is not None:
        arr = _np.asarray(values, dtype=_np.float64)
        if arr.shape != (n_rows,):
            raise ValueError(
                f"custom objective returned {name} with shape "
                f"{arr.shape}, expected ({n_rows},)"
            )
        out[:] = arr
        return
    if len(values) != n_rows:
        raise ValueError(
            f"custom objective returned {len(values)} {name} values, "
            f"expected {n_rows}"
        )
    for i in range(n_rows):
        out[i] = float(values[i])


def _metric_spec(item, index, task):
    """One `eval_metric` entry as `(name, func, higher_is_better,
    use_for_early_stopping, code)`.

    Accepted forms: one of LightGBM's metric names, a callable, a
    `(name, func[, higher_is_better[, use_for_early_stopping]])` tuple, or a
    dict with those keys. A name resolves to a built-in metric, whose code
    is what `_mojotrees.eval_metric` computes and whose `func` is None; a
    callable keeps `code` None instead. Unlike LightGBM, a callable's
    direction is declared here rather than returned by the callback, because
    early stopping needs it before the first evaluation.
    """
    higher = False
    early_stopping = True
    if isinstance(item, str):
        name, code, higher = _eval.resolve(item, task)
        return name, None, higher, True, code
    if callable(item):
        func = item
        name = getattr(item, "__name__", None) or f"metric_{index}"
    elif isinstance(item, dict):
        spec = dict(item)
        func = spec.pop("func", spec.pop("metric", None))
        name = spec.pop("name", None)
        higher = bool(spec.pop("higher_is_better", False))
        early_stopping = bool(spec.pop("early_stopping", True))
        if spec:
            raise ValueError(
                f"unknown eval_metric keys {sorted(spec)}; expected name, "
                "func, higher_is_better, early_stopping"
            )
        if name is None:
            name = getattr(func, "__name__", None) or f"metric_{index}"
    elif isinstance(item, (tuple, list)):
        if not 2 <= len(item) <= 4:
            raise ValueError(
                "an eval_metric tuple must be (name, func[, "
                "higher_is_better[, use_for_early_stopping]])"
            )
        name, func = item[0], item[1]
        if len(item) > 2:
            higher = bool(item[2])
        if len(item) > 3:
            early_stopping = bool(item[3])
    else:
        raise ValueError(
            f"eval_metric entry {item!r} must be a callable, a "
            "(name, func, ...) tuple, or a dict"
        )
    if not callable(func):
        raise ValueError("each eval_metric needs a callable")
    if not isinstance(name, str) or not name:
        raise ValueError("each eval_metric needs a non-empty name")
    return name, func, higher, early_stopping, None


def _metric_specs(eval_metric, task, objective=None):
    """Every `eval_metric` entry normalized, with unique names.

    `eval_metric=None` falls back to the metric LightGBM would score for
    this task and objective (see `_eval.default_metric`).
    """
    if eval_metric is None:
        eval_metric = _eval.default_metric(task, objective)
    single = (
        isinstance(eval_metric, str)
        or callable(eval_metric)
        or isinstance(eval_metric, dict)
        or (
            isinstance(eval_metric, tuple)
            and eval_metric
            and isinstance(eval_metric[0], str)
            and not all(isinstance(entry, str) for entry in eval_metric)
        )
    )
    items = [eval_metric] if single else list(eval_metric)
    if not items:
        raise ValueError("eval_metric must not be empty")
    specs = [_metric_spec(item, i, task) for i, item in enumerate(items)]
    names = [spec[0] for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("eval_metric names must be unique")
    return specs


def _primary_index(primary_metric, specs):
    """The index of the metric that selects the best round, by position or
    by name."""
    if primary_metric is None:
        return 0
    if isinstance(primary_metric, str):
        for i, spec in enumerate(specs):
            if spec[0] == primary_metric:
                return i
        raise ValueError(
            f"primary_metric {primary_metric!r} is not one of the "
            "eval_metric names"
        )
    index = int(primary_metric)
    if not 0 <= index < len(specs):
        raise ValueError(
            f"primary_metric {index} is out of range for {len(specs)} metrics"
        )
    return index


def _eval_pairs(eval_set, eval_X, eval_y):
    """The validation sets as a list of `(X, y)` pairs, or None when there
    are none.

    Three spellings are accepted. `eval_set=[(X, y), ...]` is LightGBM's,
    `eval_set=(X, y)` is the one-set shorthand (a *tuple* of length two is
    one pair, a list is a list of pairs), and `eval_X=` / `eval_y=` is the
    keyword form, which cannot be combined with `eval_set`.
    """
    if eval_X is not None or eval_y is not None:
        if eval_set is not None:
            raise ValueError(
                "pass either eval_set or eval_X/eval_y, not both"
            )
        if eval_X is None or eval_y is None:
            raise ValueError("eval_X and eval_y must be given together")
        return [(eval_X, eval_y)]
    if eval_set is None:
        return None
    if isinstance(eval_set, tuple) and len(eval_set) == 2:
        return [eval_set]
    pairs = list(eval_set)
    if not pairs:
        raise ValueError("eval_set must not be empty")
    return pairs


def _per_set(value, n_sets, name):
    """One entry per validation set. A single vector is accepted as well as
    a list of them, so one eval set needs no nesting."""
    if value is None:
        return [None] * n_sets
    entries = list(value)
    if entries and not hasattr(entries[0], "__len__"):
        # A bare vector of numbers rather than a list of vectors.
        entries = [value]
    if len(entries) != n_sets:
        raise ValueError(
            f"{name} must have one entry per eval_set entry "
            f"({len(entries)} given for {n_sets})"
        )
    return entries


def _encode_like(y, n_rows, classes, name):
    """Validation labels as the class codes the trainer uses, encoded
    through the classes `fit` recorded.

    A label the training set never held has no code and no meaning here, so
    it raises instead of being folded into a neighboring class.
    """
    known = list(classes.tolist() if hasattr(classes, "tolist") else classes)
    index = {label: i for i, label in enumerate(known)}
    values = list(y.tolist() if hasattr(y, "tolist") else y)
    if len(values) != n_rows:
        raise ValueError(
            f"{name} labels must have length {n_rows}, got {len(values)}"
        )
    codes = []
    for value in values:
        if value not in index:
            raise ValueError(
                f"{name} has label {value!r}, which is not one of the "
                "classes seen during fit"
            )
        codes.append(float(index[value]))
    return _as_f64_vector(codes, n_rows, name)


def _early_stopping_rounds(value):
    """`early_stopping_rounds` as a nonnegative int, with `None` meaning
    off, as LightGBM's callback treats it."""
    rounds = 0 if value is None else int(value)
    if rounds < 0:
        raise ValueError("early_stopping_rounds must not be negative")
    return rounds


def _check_eval_arguments(
    eval_set,
    eval_metric,
    eval_sample_weight,
    early_stopping_rounds,
    callbacks=None,
):
    """Reject the arguments that only mean something with validation data.

    Early stopping in particular needs something to stop on: LightGBM's
    callback raises when no validation set is present, and so does this,
    rather than quietly training the full ensemble the caller did not ask
    for.
    """
    if eval_set is not None:
        return
    if eval_metric is not None:
        raise ValueError("eval_metric needs an eval_set to score")
    if eval_sample_weight is not None:
        raise ValueError("eval_sample_weight needs an eval_set to weight")
    if _early_stopping_rounds(early_stopping_rounds) > 0:
        raise ValueError(
            "early_stopping_rounds needs an eval_set to stop on; pass "
            "eval_set=[(X_valid, y_valid)]"
        )
    if callbacks:
        raise ValueError(
            "callbacks need an eval_set: the per-iteration hook lives in the "
            "trainer that scores validation metrics; pass "
            "eval_set=[(X_valid, y_valid)]"
        )


def _device_name(device):
    """A `predict(device=...)` argument, validated and lowercased.

    `None` passes through as `None`, which means the established path: the
    CPU, with no policy consulted. The names are case-insensitive, as
    LightGBM treats `device_type`, and the spelling check is here rather
    than across the boundary only so that a typo names the alternatives.
    """
    if device is None:
        return None
    if not isinstance(device, str) or device.lower() not in _DEVICES:
        raise ValueError(
            f"unknown device {device!r}; expected one of "
            + ", ".join(_DEVICES)
        )
    return device.lower()
