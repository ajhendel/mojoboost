"""CPython extension module for mojoboost.

Built with `bindings/build.sh` into `python/mojoboost/_mojoboost.so`; the
public Python surface is the sklearn-style wrapper in `python/mojoboost/`.

Data crosses the boundary as raw buffer addresses (integers) plus lengths:
the wrapper passes float64-contiguous buffers (column-major for feature
matrices) and keeps them alive for the duration of each call. Copies into
Mojo Lists happen here, so no Python buffer is retained after a call
returns. Trained models are returned as opaque handles owned by Python.
"""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from mojoboost.boosting import BoosterParams
from mojoboost.device import (
    device_name as mojo_device_name,
    gpu_available as mojo_gpu_available,
    parse_device,
    resolve_device as mojo_resolve_device,
)
from mojoboost.importance import gain_importance, split_importance
from mojoboost.interaction import InteractionConstraints
from mojoboost.model import Model, MulticlassModel
from mojoboost.model import fit as mojo_fit
from mojoboost.model import fit_custom as mojo_fit_custom
from mojoboost.model import fit_multiclass as mojo_fit_multiclass
from mojoboost.serialize import (
    load_model,
    load_multiclass_model,
    save_model,
    save_multiclass_model,
)
from mojoboost.tree import Tree, TreeParams


@export
def PyInit__mojoboost() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojoboost")
        _ = m.add_type[Model]("Model")
        _ = m.add_type[MulticlassModel]("MulticlassModel")
        m.def_function[fit]("fit")
        m.def_function[fit_custom]("fit_custom")
        m.def_function[fit_multiclass]("fit_multiclass")
        m.def_function[predict]("predict")
        m.def_function[predict_raw]("predict_raw")
        m.def_function[predict_proba]("predict_proba")
        m.def_function[num_trees]("num_trees")
        m.def_function[num_iterations]("num_iterations")
        m.def_function[num_iterations_multiclass]("num_iterations_multiclass")
        m.def_function[n_classes]("n_classes")
        m.def_function[n_features]("n_features")
        m.def_function[n_features_multiclass]("n_features_multiclass")
        m.def_function[feature_importance]("feature_importance")
        m.def_function[feature_importance_multiclass](
            "feature_importance_multiclass"
        )
        m.def_function[save]("save")
        m.def_function[load]("load")
        m.def_function[save_multiclass]("save_multiclass")
        m.def_function[load_multiclass]("load_multiclass")
        m.def_function[gpu_available]("gpu_available")
        m.def_function[resolve_device]("resolve_device")
        return m.finalize()
    except e:
        abort(String("failed to create _mojoboost module: ", e))


def _f64_list(addr: Int, n: Int) raises -> List[Float64]:
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(p.unsafe_load(i))
    return out^


def _int_list_from_f64(addr: Int, n: Int) raises -> List[Int]:
    if addr == 0 or n < 0:
        raise Error("invalid buffer")
    var p = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)
    var out = List[Int](capacity=n)
    for i in range(n):
        out.append(Int(p.unsafe_load(i)))
    return out^


def _row(
    features: List[Float64], n_rows: Int, n_features: Int, r: Int
) -> List[Float64]:
    var row = List[Float64](capacity=n_features)
    for f in range(n_features):
        row.append(features[f * n_rows + r])
    return row^


def _parse_constraints(
    params: PythonObject, n_features: Int
) raises -> InteractionConstraints:
    """Interaction constraint groups from the params dict, flattened by the
    wrapper into two float64 buffers (group features, and one more offset
    than there are groups). A zero address means unconstrained. The wrapper
    validates too; this is the check that also covers direct callers."""
    var flat_addr = Int(py=params["interaction_flat_addr"])
    if flat_addr == 0:
        return InteractionConstraints()
    var flat = _int_list_from_f64(
        flat_addr, Int(py=params["interaction_flat_len"])
    )
    var offsets = _int_list_from_f64(
        Int(py=params["interaction_offsets_addr"]),
        Int(py=params["interaction_offsets_len"]),
    )
    return InteractionConstraints.from_flat(flat, offsets, n_features)


def _parse_params(
    params: PythonObject, n_features: Int
) raises -> BoosterParams:
    var tree = TreeParams(
        Int(py=params["num_leaves"]),
        Int(py=params["min_data_in_leaf"]),
        Float64(py=params["lambda_l2"]),
        Float64(py=params["min_child_hess"]),
        Float64(py=params["lambda_l1"]),
        _parse_constraints(params, n_features),
    )
    return BoosterParams(
        Int(py=params["n_estimators"]),
        Float64(py=params["learning_rate"]),
        tree^,
    )


def _parse_device(params: PythonObject) raises -> Int:
    """Device code from the params dict. The wrapper sends the name it
    already resolved ("cpu" or "gpu"); the trainer resolves it again, so
    the policy in device.mojo stays the only one."""
    return parse_device(String(py=params["device"]))


def _parse_weights(params: PythonObject, n_rows: Int) raises -> List[Float64]:
    var weight_addr = Int(py=params["sample_weight_addr"])
    if weight_addr == 0:
        return List[Float64]()
    return _f64_list(weight_addr, n_rows)


def fit(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    objective: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a single-output model. Buffers are float64; X is column-major."""
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var target = _f64_list(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf)
    var weights = _parse_weights(params, nr)
    var model = mojo_fit(
        features,
        nr,
        nf,
        target,
        Int(py=objective),
        bp,
        Int(py=params["max_bin"]),
        weights,
        Float64(py=params["alpha"]),
        _parse_device(params),
    )
    return PythonObject(alloc=model^)


def fit_multiclass(
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    y_addr: PythonObject,
    n_classes: PythonObject,
    params: PythonObject,
) raises -> PythonObject:
    """Train a multiclass model. Labels arrive as float64 in 0..n_classes-1."""
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var labels = _int_list_from_f64(Int(py=y_addr), nr)
    var bp = _parse_params(params, nf)
    var weights = _parse_weights(params, nr)
    var model = mojo_fit_multiclass(
        features,
        nr,
        nf,
        labels,
        Int(py=n_classes),
        bp,
        Int(py=params["max_bin"]),
        weights,
        _parse_device(params),
    )
    return PythonObject(alloc=model^)


def predict(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Response-scale predictions into a preallocated float64 buffer."""
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    for r in range(nr):
        out.unsafe_store(r, m[].predict(_row(features, nr, nf, r)))
    return PythonObject(None)


def predict_raw(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Raw-score predictions into a preallocated float64 buffer."""
    var m = model.downcast_value_ptr[Model]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    for r in range(nr):
        out.unsafe_store(r, m[].predict_raw(_row(features, nr, nf, r)))
    return PythonObject(None)


def predict_proba(
    model: PythonObject,
    x_addr: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Multiclass probabilities, row-major `[r * n_classes + k]`, into a
    preallocated float64 buffer of size n_rows * n_classes."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nr = Int(py=n_rows)
    var nf = Int(py=n_features)
    var features = _f64_list(Int(py=x_addr), nr * nf)
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=Int(py=out_addr)
    )
    var k = m[].booster.n_classes
    for r in range(nr):
        var proba = m[].predict_proba(_row(features, nr, nf, r))
        for c in range(k):
            out.unsafe_store(r * k + c, proba[c])
    return PythonObject(None)


def gpu_available() raises -> PythonObject:
    """True when training can run on an accelerator (see device.mojo)."""
    return PythonObject(mojo_gpu_available())


def resolve_device(
    device: PythonObject,
    n_rows: PythonObject,
    n_features: PythonObject,
    n_outputs: PythonObject,
) raises -> PythonObject:
    """Resolve a requested device name ("cpu", "gpu", or "auto") to the
    backend that will run: "cpu" or "gpu". Raises on an unknown name, on
    "gpu" without an accelerator, and on "gpu" for a workload the GPU path
    does not cover. `n_outputs` is 1 for single-output training and the
    class count for multiclass."""
    var resolved = mojo_resolve_device(
        parse_device(String(py=device)),
        Int(py=n_rows),
        Int(py=n_features),
        Int(py=n_outputs),
    )
    return PythonObject(mojo_device_name(resolved))


def num_trees(model: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(len(m[].booster.trees))


def num_iterations(model: PythonObject) raises -> PythonObject:
    """Boosting iterations the fitted ensemble kept. One iteration is one
    tree for a single-output model, so this is the tree count."""
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(len(m[].booster.trees))


def num_iterations_multiclass(model: PythonObject) raises -> PythonObject:
    """Boosting iterations a multiclass ensemble kept. One iteration is one
    tree per class, so this is the tree count divided by the class count."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(len(m[].booster.trees) // m[].booster.n_classes)


def n_classes(model: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(m[].booster.n_classes)


def n_features(model: PythonObject) raises -> PythonObject:
    """Features the model was trained on, from its bin mapper."""
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(m[].mapper.n_features)


def n_features_multiclass(model: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(m[].mapper.n_features)


def _write_importance(
    trees: List[Tree], nf: Int, kind: Int, out_addr: Int
) raises:
    """Per-feature importance into a preallocated float64 buffer of length
    `nf`. `kind` is 0 for split counts and 1 for total gain."""
    if kind != 0 and kind != 1:
        raise Error("importance kind must be 0 (split) or 1 (gain)")
    var out = Pointer[Float64, MutUntrackedOrigin](
        unsafe_from_address=out_addr
    )
    if kind == 0:
        var counts = split_importance(trees, nf)
        for f in range(nf):
            out.unsafe_store(f, Float64(counts[f]))
    else:
        var gains = gain_importance(trees, nf)
        for f in range(nf):
            out.unsafe_store(f, gains[f])


def feature_importance(
    model: PythonObject,
    n_features: PythonObject,
    kind: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Split-count (kind 0) or total-gain (kind 1) importance for a
    single-output model. Gains are not part of the serialized format, so a
    model read back from disk reports zero gain importance."""
    var m = model.downcast_value_ptr[Model]()
    _write_importance(
        m[].booster.trees,
        Int(py=n_features),
        Int(py=kind),
        Int(py=out_addr),
    )
    return PythonObject(None)


def feature_importance_multiclass(
    model: PythonObject,
    n_features: PythonObject,
    kind: PythonObject,
    out_addr: PythonObject,
) raises -> PythonObject:
    """Importance summed over every class's trees, LightGBM style."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    _write_importance(
        m[].booster.trees,
        Int(py=n_features),
        Int(py=kind),
        Int(py=out_addr),
    )
    return PythonObject(None)


def save(model: PythonObject, path: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    save_model(m[], String(py=path))
    return PythonObject(None)


def load(path: PythonObject) raises -> PythonObject:
    var model = load_model(String(py=path))
    return PythonObject(alloc=model^)


def save_multiclass(
    model: PythonObject, path: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    save_multiclass_model(m[], String(py=path))
    return PythonObject(None)


def load_multiclass(path: PythonObject) raises -> PythonObject:
    var model = load_multiclass_model(String(py=path))
    return PythonObject(alloc=model^)
