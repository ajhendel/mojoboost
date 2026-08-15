"""Editing a fitted model from Python: LightGBM's `Booster` mutators.

`src/mojotrees/model_editing.mojo` implements `rollback_one_iter`,
`set_leaf_output`, `shuffle_models`, `refit`, and the prediction bounds
`lower_bound` / `upper_bound`, each checked against the invariants a fitted
model records (routing, covers, the monotone claim, loadability). This
module is the boundary that lets `python/mojotrees/basic.py`'s `Booster`
call them on the handle it already holds.

Conventions, shared with `inspection_bindings.mojo`:

- one entry point per model kind (`_multiclass` suffix), because a handle is
  a `Model` or a `MulticlassModel` and the Python layer knows which;
- every mutator writes through the handle in place and returns what the
  native function returns, so a Python `Booster` sees the edit on its next
  call without rebuilding anything;
- leaf values cross on LightGBM's scale, the value a leaf contributes to a
  raw score. Natively a leaf stores the unshrunk value and the ensemble
  multiplies by `learning_rate` at predict time, so `set_leaf_output`
  divides on the way in and `get_leaf_output` multiplies on the way out.
  That is the one conversion this file owns, and it is here rather than in
  Python because only the native ensemble knows its own rate.

Refit reads the leaf-shaping parameters (`lambda_l2`, `lambda_l1`,
`max_delta_step`, `path_smooth`) as plain scalars rather than through the
full parameter parser, because a refit never grows anything and those four
are all a leaf formula consumes.
"""

from std.python import Python, PythonObject

from binding_support import py_dict, py_f64_list, py_pair

from mojotrees.boosting import IterationRange
from mojotrees.model import Model, MulticlassModel
from mojotrees.model_editing import (
    EDIT_MODE_GBDT,
    LEAF_EDIT_CLAMP,
    RefitParams,
    RefitReport,
    ScoreBounds,
    editing_capabilities,
    get_leaf_output as mojo_get_leaf_output,
    get_leaf_output_multiclass as mojo_get_leaf_output_multiclass,
    model_editing_status_json as mojo_model_editing_status_json,
    probability_bounds_multiclass as mojo_probability_bounds_multiclass,
    raw_score_bounds as mojo_raw_score_bounds,
    raw_score_bounds_multiclass as mojo_raw_score_bounds_multiclass,
    refit_dataset as mojo_refit_dataset,
    refit_dataset_multiclass as mojo_refit_dataset_multiclass,
    response_bounds as mojo_response_bounds,
    rollback_one_iter as mojo_rollback_one_iter,
    rollback_one_iter_multiclass as mojo_rollback_one_iter_multiclass,
    rollback_to as mojo_rollback_to,
    rollback_to_multiclass as mojo_rollback_to_multiclass,
    set_leaf_output as mojo_set_leaf_output,
    set_leaf_output_multiclass as mojo_set_leaf_output_multiclass,
    shuffle_iterations as mojo_shuffle_iterations,
    shuffle_iterations_multiclass as mojo_shuffle_iterations_multiclass,
)
from mojotrees.tree import TreeParams
from mojotrees.trainset import Dataset


# -- status ----------------------------------------------------------------


def model_editing_status() raises -> PythonObject:
    """The native editing status as JSON text: whether this build edits a
    fitted model in place and which operations that covers."""
    return PythonObject(mojo_model_editing_status_json())


def model_editing_operations() raises -> PythonObject:
    """The operations `editing_capabilities` lists, as a list of
    `(name, supported, reason)` triples."""
    var caps = editing_capabilities()
    var out = Python.list()
    for i in range(len(caps)):
        var row = Python.list()
        row.append(PythonObject(caps[i].operation))
        row.append(PythonObject(caps[i].supported))
        row.append(PythonObject(caps[i].reason))
        out.append(row)
    return out^


# -- rollback ---------------------------------------------------------------


def rollback_one_iter(model: PythonObject, mode: PythonObject) raises -> PythonObject:
    """Drop the last iteration; returns how many remain. `mode` is the
    boosting mode the caller states (0 gbdt, 1 rf, 2 dart)."""
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(mojo_rollback_one_iter(m[].booster, Int(py=mode)))


def rollback_one_iter_multiclass(
    model: PythonObject, mode: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(
        mojo_rollback_one_iter_multiclass(m[].booster, Int(py=mode))
    )


def rollback_to(
    model: PythonObject, n_iterations: PythonObject, mode: PythonObject
) raises -> PythonObject:
    """Truncate to the first `n_iterations`; returns how many remain."""
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(
        mojo_rollback_to(m[].booster, Int(py=n_iterations), Int(py=mode))
    )


def rollback_to_multiclass(
    model: PythonObject, n_iterations: PythonObject, mode: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(
        mojo_rollback_to_multiclass(
            m[].booster, Int(py=n_iterations), Int(py=mode)
        )
    )


# -- leaf outputs -------------------------------------------------------------


def get_leaf_output(
    model: PythonObject, tree_index: PythonObject, leaf_index: PythonObject
) raises -> PythonObject:
    """One leaf's contribution to a raw score (LightGBM's scale)."""
    var m = model.downcast_value_ptr[Model]()
    var v = mojo_get_leaf_output(
        m[].booster, Int(py=tree_index), Int(py=leaf_index)
    )
    return PythonObject(m[].booster.learning_rate * v)


def get_leaf_output_multiclass(
    model: PythonObject,
    iteration: PythonObject,
    class_id: PythonObject,
    leaf_index: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    var v = mojo_get_leaf_output_multiclass(
        m[].booster, Int(py=iteration), Int(py=class_id), Int(py=leaf_index)
    )
    return PythonObject(m[].booster.learning_rate * v)


def set_leaf_output(
    model: PythonObject,
    tree_index: PythonObject,
    leaf_index: PythonObject,
    value: PythonObject,
    policy: PythonObject,
) raises -> PythonObject:
    """Set one leaf's contribution to the raw score and return what was
    stored, on the same scale. The two differ only when a monotone claim
    clamped the write (`policy` 0) rather than refusing it (`policy` 1)."""
    var m = model.downcast_value_ptr[Model]()
    var rate = m[].booster.learning_rate
    var stored = mojo_set_leaf_output(
        m[].booster,
        Int(py=tree_index),
        Int(py=leaf_index),
        Float64(py=value) / rate,
        Int(py=policy),
    )
    return PythonObject(rate * stored)


def set_leaf_output_multiclass(
    model: PythonObject,
    iteration: PythonObject,
    class_id: PythonObject,
    leaf_index: PythonObject,
    value: PythonObject,
    policy: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    var rate = m[].booster.learning_rate
    var stored = mojo_set_leaf_output_multiclass(
        m[].booster,
        Int(py=iteration),
        Int(py=class_id),
        Int(py=leaf_index),
        Float64(py=value) / rate,
        Int(py=policy),
    )
    return PythonObject(rate * stored)


# -- order ------------------------------------------------------------------


def shuffle_models(
    model: PythonObject,
    seed: PythonObject,
    start: PythonObject,
    stop: PythonObject,
) raises:
    """Shuffle iterations `[start, stop)`, clamped to the ensemble, under a
    seeded counter-based draw. LightGBM's `shuffle_models`."""
    var m = model.downcast_value_ptr[Model]()
    var n = m[].booster.n_iterations()
    mojo_shuffle_iterations(
        m[].booster,
        Int(py=seed),
        IterationRange.slice(n, Int(py=start), Int(py=stop)),
    )


def shuffle_models_multiclass(
    model: PythonObject,
    seed: PythonObject,
    start: PythonObject,
    stop: PythonObject,
) raises:
    var m = model.downcast_value_ptr[MulticlassModel]()
    var n = m[].booster.n_iterations()
    mojo_shuffle_iterations_multiclass(
        m[].booster,
        Int(py=seed),
        IterationRange.slice(n, Int(py=start), Int(py=stop)),
    )


# -- refit ------------------------------------------------------------------


def _refit_params(
    decay_rate: PythonObject,
    min_leaf_rows: PythonObject,
    recount: PythonObject,
    lambda_l2: PythonObject,
    lambda_l1: PythonObject,
    max_delta_step: PythonObject,
    path_smooth: PythonObject,
) raises -> RefitParams:
    var tree = TreeParams.default()
    tree.lambda_reg = Float64(py=lambda_l2)
    tree.lambda_l1 = Float64(py=lambda_l1)
    tree.extra.max_delta_step = Float64(py=max_delta_step)
    tree.extra.path_smooth = Float64(py=path_smooth)
    return RefitParams(
        Float64(py=decay_rate),
        tree^,
        Int(py=min_leaf_rows),
        Int(py=recount) != 0,
    )


def _py_report(report: RefitReport) raises -> PythonObject:
    var out = py_dict()
    out["n_trees"] = PythonObject(report.n_trees)
    out["n_leaves_updated"] = PythonObject(report.n_leaves_updated)
    out["n_leaves_kept"] = PythonObject(report.n_leaves_kept)
    out["n_leaves_clamped"] = PythonObject(report.n_leaves_clamped)
    out["n_trees_recounted"] = PythonObject(report.n_trees_recounted)
    return out^


def refit(
    model: PythonObject,
    dataset: PythonObject,
    decay_rate: PythonObject,
    min_leaf_rows: PythonObject,
    recount: PythonObject,
    lambda_l2: PythonObject,
    lambda_l1: PythonObject,
    max_delta_step: PythonObject,
    path_smooth: PythonObject,
    alpha: PythonObject,
) raises -> PythonObject:
    """Rebuild every leaf value from `dataset`, keeping every tree's shape.
    Returns the refit report as a dict. LightGBM's `Booster.refit`."""
    var m = model.downcast_value_ptr[Model]()
    var d = dataset.downcast_value_ptr[Dataset]()
    var report = mojo_refit_dataset(
        m[],
        d[],
        _refit_params(
            decay_rate,
            min_leaf_rows,
            recount,
            lambda_l2,
            lambda_l1,
            max_delta_step,
            path_smooth,
        ),
        Float64(py=alpha),
    )
    return _py_report(report)


def refit_multiclass(
    model: PythonObject,
    dataset: PythonObject,
    decay_rate: PythonObject,
    min_leaf_rows: PythonObject,
    recount: PythonObject,
    lambda_l2: PythonObject,
    lambda_l1: PythonObject,
    max_delta_step: PythonObject,
    path_smooth: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    var d = dataset.downcast_value_ptr[Dataset]()
    var report = mojo_refit_dataset_multiclass(
        m[],
        d[],
        _refit_params(
            decay_rate,
            min_leaf_rows,
            recount,
            lambda_l2,
            lambda_l1,
            max_delta_step,
            path_smooth,
        ),
    )
    return _py_report(report)


# -- bounds -----------------------------------------------------------------


def _py_bounds(b: ScoreBounds) raises -> PythonObject:
    return py_pair(PythonObject(b.lower), PythonObject(b.upper))


def score_bounds(model: PythonObject, response: PythonObject) raises -> PythonObject:
    """`(lower, upper)` on the raw score, or on the response scale when
    `response` is nonzero. LightGBM's `lower_bound` / `upper_bound`."""
    var m = model.downcast_value_ptr[Model]()
    if Int(py=response) != 0:
        return _py_bounds(mojo_response_bounds(m[].booster))
    return _py_bounds(mojo_raw_score_bounds(m[].booster))


def score_bounds_multiclass(
    model: PythonObject, response: PythonObject
) raises -> PythonObject:
    """Per-class `(lower, upper)` pairs on the raw scores, or on the softmax
    probabilities when `response` is nonzero."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var bounds: List[ScoreBounds]
    if Int(py=response) != 0:
        bounds = mojo_probability_bounds_multiclass(m[].booster)
    else:
        bounds = mojo_raw_score_bounds_multiclass(m[].booster)
    var out = Python.list()
    for k in range(len(bounds)):
        out.append(_py_bounds(bounds[k]))
    return out^
