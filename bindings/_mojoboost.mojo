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
from mojoboost.model import Model, MulticlassModel
from mojoboost.model import fit as mojo_fit
from mojoboost.model import fit_multiclass as mojo_fit_multiclass
from mojoboost.serialize import (
    load_model,
    load_multiclass_model,
    save_model,
    save_multiclass_model,
)
from mojoboost.tree import TreeParams


@export
def PyInit__mojoboost() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojoboost")
        _ = m.add_type[Model]("Model")
        _ = m.add_type[MulticlassModel]("MulticlassModel")
        m.def_function[fit]("fit")
        m.def_function[fit_multiclass]("fit_multiclass")
        m.def_function[predict]("predict")
        m.def_function[predict_raw]("predict_raw")
        m.def_function[predict_proba]("predict_proba")
        m.def_function[num_trees]("num_trees")
        m.def_function[n_classes]("n_classes")
        m.def_function[save]("save")
        m.def_function[load]("load")
        m.def_function[save_multiclass]("save_multiclass")
        m.def_function[load_multiclass]("load_multiclass")
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


def _parse_params(params: PythonObject) raises -> BoosterParams:
    var tree = TreeParams(
        Int(py=params["num_leaves"]),
        Int(py=params["min_data_in_leaf"]),
        Float64(py=params["lambda_l2"]),
        Float64(py=params["min_child_hess"]),
    )
    return BoosterParams(
        Int(py=params["n_estimators"]),
        Float64(py=params["learning_rate"]),
        tree^,
    )


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
    var bp = _parse_params(params)
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
    var bp = _parse_params(params)
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


def num_trees(model: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(len(m[].booster.trees))


def n_classes(model: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(m[].booster.n_classes)


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
