"""Implementation of the mojoboost C ABI declared in capi/mojoboost.h.

Build with `capi/build.sh`, which emits a shared library exporting exactly
the symbols in that header.

Nothing in here is a mojoboost type as far as a C caller is concerned:
models and error objects are heap allocated Mojo structs handed out as
opaque pointers, and every other argument is a C scalar, a C string, or a
buffer of doubles the caller owns. That is what lets the Mojo side change
freely without breaking compiled callers, so keep it that way: never add a
struct to a signature here, and put new hyperparameters in the parameter
string (see params.mojo) instead.

Two rules keep the boundary safe. First, every exported function catches
everything: a Mojo error must never unwind into C, so each body is one
try/except that turns an error into a status code and a message. Second,
pointers are checked before use, including the ones C code is most likely
to get wrong, so a NULL is a status code rather than a crash.
"""

from std.memory.alloc import unsafe_alloc

from mojoboost.model import Model, MulticlassModel, fit, fit_multiclass
from mojoboost.params import (
    TrainConfig,
    params_names_mojo_api_only,
    parse_params,
)
from mojoboost.serialize import (
    load_model,
    load_multiclass_model,
    model_file_kind,
    save_model,
    save_multiclass_model,
)

# Keep in sync with MOJOBOOST_ABI_VERSION in capi/mojoboost.h.
comptime ABI_VERSION: Int32 = 1

# Keep in sync with python/pyproject.toml and pixi.toml.
comptime VERSION_MAJOR: Int32 = 0
comptime VERSION_MINOR: Int32 = 1
comptime VERSION_PATCH: Int32 = 0

comptime OK: Int32 = 0
comptime ERROR_INVALID_ARGUMENT: Int32 = -1
comptime ERROR_TRAINING: Int32 = -2
comptime ERROR_IO: Int32 = -3
comptime ERROR_UNSUPPORTED: Int32 = -4

comptime _SINGLE = 0
comptime _MULTICLASS = 1

comptime _CharPtr = Pointer[UInt8, MutUntrackedOrigin]
comptime _F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime _I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime _I64Ptr = Pointer[Int64, MutUntrackedOrigin]


struct _ErrorBox(Movable):
    """The storage behind a `MojoBoostError`.

    `text` is always NUL terminated so its buffer can be handed straight to
    C as a `const char *`. Rebuilding the list on every message is what
    makes the documented lifetime true: the pointer a caller holds stays
    valid exactly until the next call that writes this object.
    """

    var text: List[UInt8]

    def __init__(out self):
        self.text = [0]

    def set(mut self, message: String):
        var text = List[UInt8](capacity=message.byte_length() + 1)
        for b in message.as_bytes():
            text.append(b)
        text.append(0)
        self.text = text^

    def clear(mut self):
        self.text = [0]


struct _ModelBox(Movable):
    """The storage behind a `MojoBoostModel`.

    Single-output and multiclass models are different Mojo types but one
    handle type in C, because a caller loading a file does not necessarily
    know which it has. `kind` says which of the two is populated.
    """

    var kind: Int
    var single: Optional[Model]
    var multi: Optional[MulticlassModel]

    def __init__(out self, var model: Model):
        self.kind = _SINGLE
        self.single = Optional(model^)
        self.multi = Optional[MulticlassModel]()

    def __init__(out self, var model: MulticlassModel):
        self.kind = _MULTICLASS
        self.single = Optional[Model]()
        self.multi = Optional(model^)

    def n_features(self) raises -> Int:
        if self.kind == _MULTICLASS:
            return self.multi.value().mapper.n_features
        return self.single.value().mapper.n_features

    def n_classes(self) raises -> Int:
        if self.kind == _MULTICLASS:
            return self.multi.value().booster.n_classes
        return 1

    def n_trees(self) raises -> Int:
        if self.kind == _MULTICLASS:
            return len(self.multi.value().booster.trees)
        return len(self.single.value().booster.trees)


comptime _ModelPtr = Pointer[_ModelBox, MutUntrackedOrigin]
comptime _ErrorPtr = Pointer[_ErrorBox, MutUntrackedOrigin]
comptime _ModelOutPtr = Pointer[_ModelPtr, MutUntrackedOrigin]


def _is_null[T: AnyType, //](p: Pointer[T, MutUntrackedOrigin]) -> Bool:
    """Whether a pointer that arrived from C is NULL. `Pointer` is
    non-nullable in Mojo, so a NULL from C is only ever inspected as an
    address and never dereferenced."""
    return Int(p) == 0


def _clear(error: _ErrorPtr):
    if not _is_null(error):
        error[].clear()


def _fail(error: _ErrorPtr, code: Int32, message: String) -> Int32:
    if not _is_null(error):
        error[].set(message)
    return code


def _invalid(error: _ErrorPtr, message: String) -> Int32:
    return _fail(error, ERROR_INVALID_ARGUMENT, message)


def _c_string(p: _CharPtr) -> String:
    """Copy a NUL terminated C string into an owned Mojo `String`. The
    caller must have checked for NULL."""
    var bytes = List[UInt8]()
    var i = 0
    while True:
        var b = p.unsafe_load(i)
        if b == 0:
            break
        bytes.append(b)
        i += 1
    return String(StringSlice(unsafe_from_utf8=Span(bytes)))


def _copy_f64(p: _F64Ptr, n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for i in range(n):
        out.append(p.unsafe_load(i))
    return out^


def _row(
    features: List[Float64], n_rows: Int, n_features: Int, r: Int
) -> List[Float64]:
    """Row `r` of a column-major matrix, which is what `Model.predict`
    takes."""
    var row = List[Float64](capacity=n_features)
    for f in range(n_features):
        row.append(features[f * n_rows + r])
    return row^


def _check_matrix(
    error: _ErrorPtr, data: _F64Ptr, n_rows: Int64, n_features: Int64
) -> Int32:
    """Shared validation for the two dense-matrix entry points."""
    if n_rows <= 0:
        return _invalid(error, String("n_rows must be positive"))
    if n_features <= 0:
        return _invalid(error, String("n_features must be positive"))
    if _is_null(data):
        return _invalid(error, String("data must not be NULL"))
    return OK


def _param_error_code(spec: String) -> Int32:
    """Whether a rejected parameter string was merely wrong, or named a
    feature that exists but is not reachable from a parameter string."""
    if params_names_mojo_api_only(spec):
        return ERROR_UNSUPPORTED
    return ERROR_INVALID_ARGUMENT


def _new_handle[T: Movable, //](var box: T) -> Pointer[T, MutUntrackedOrigin]:
    var p = unsafe_alloc[T](1)
    p.unsafe_write(box^)
    return p


# ------------------------------------------------------------------ version


@export
def mojoboost_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def mojoboost_library_version(
    major: _I32Ptr, minor: _I32Ptr, patch: _I32Ptr
) abi("C"):
    if not _is_null(major):
        major[] = VERSION_MAJOR
    if not _is_null(minor):
        minor[] = VERSION_MINOR
    if not _is_null(patch):
        patch[] = VERSION_PATCH


# -------------------------------------------------------------------- error


@export
def mojoboost_error_create() abi("C") -> _ErrorPtr:
    return _new_handle(_ErrorBox())


@export
def mojoboost_error_free(error: _ErrorPtr) abi("C"):
    if _is_null(error):
        return
    error.unsafe_deinit_pointee()
    error.unsafe_free()


@export
def mojoboost_error_message(error: _ErrorPtr) abi("C") -> Int:
    """Returns `const char *` as an address. Mojo's `Pointer` is
    non-nullable by construction, so the address is the only way to spell
    the NULL this returns for a NULL error object; `Int` and a pointer are
    returned in the same register on every ABI mojoboost targets."""
    if _is_null(error):
        return 0
    return Int(error[].text.unsafe_ptr())


# -------------------------------------------------------------------- train


@export
def mojoboost_train_dense(
    data: _F64Ptr,
    n_rows: Int64,
    n_features: Int64,
    labels: _F64Ptr,
    weights: _F64Ptr,
    parameters: _CharPtr,
    out_model: _ModelOutPtr,
    error: _ErrorPtr,
) abi("C") -> Int32:
    _clear(error)
    var bad = _check_matrix(error, data, n_rows, n_features)
    if bad != OK:
        return bad
    if _is_null(labels):
        return _invalid(error, String("labels must not be NULL"))
    if _is_null(out_model):
        return _invalid(error, String("out_model must not be NULL"))

    var spec = String("")
    if not _is_null(parameters):
        spec = _c_string(parameters)

    var config: TrainConfig
    try:
        config = parse_params(spec)
    except e:
        return _fail(error, _param_error_code(spec), String(e))

    var n = Int(n_rows)
    var f = Int(n_features)
    var features = _copy_f64(data, n * f)
    var target = _copy_f64(labels, n)
    var sample_weight = List[Float64]()
    if not _is_null(weights):
        sample_weight = _copy_f64(weights, n)

    var box: _ModelBox
    try:
        if config.is_multiclass():
            var classes = List[Int](capacity=n)
            for r in range(n):
                var v = target[r]
                var k = Int(v)
                if Float64(k) != v or k < 0 or k >= config.n_classes:
                    raise Error(
                        "multiclass labels must be integers in"
                        " 0..num_class-1; row ",
                        r,
                        " is ",
                        v,
                    )
                classes.append(k)
            box = _ModelBox(
                fit_multiclass(
                    features,
                    n,
                    f,
                    classes,
                    config.n_classes,
                    config.booster,
                    config.max_bin,
                    sample_weight,
                    device=config.device,
                    use_missing=config.use_missing,
                )
            )
        else:
            box = _ModelBox(
                fit(
                    features,
                    n,
                    f,
                    target,
                    config.objective,
                    config.booster,
                    config.max_bin,
                    sample_weight,
                    config.alpha,
                    device=config.device,
                    use_missing=config.use_missing,
                )
            )
    except e:
        return _fail(error, ERROR_TRAINING, String(e))

    out_model[] = _new_handle(box^)
    return OK


# ------------------------------------------------------------------ predict


def _predict_into(
    model: _ModelPtr,
    data: _F64Ptr,
    n_rows: Int64,
    n_features: Int64,
    out_values: _F64Ptr,
    out_len: Int64,
    error: _ErrorPtr,
    raw: Bool,
) -> Int32:
    """The body shared by `mojoboost_predict` and `mojoboost_predict_raw`.
    Predictions are computed in full before anything is written, so a
    failure leaves the caller's buffer untouched."""
    _clear(error)
    try:
        if _is_null(model):
            return _invalid(error, String("model must not be NULL"))
        var bad = _check_matrix(error, data, n_rows, n_features)
        if bad != OK:
            return bad
        if _is_null(out_values):
            return _invalid(error, String("out_values must not be NULL"))

        var expected = model[].n_features()
        if Int(n_features) != expected:
            return _invalid(
                error,
                String(
                    "n_features is ",
                    n_features,
                    " but the model was trained on ",
                    expected,
                ),
            )
        var width = model[].n_classes()
        var needed = Int(n_rows) * width
        if Int(out_len) < needed:
            return _invalid(
                error,
                String(
                    "out_len is ",
                    out_len,
                    " but ",
                    needed,
                    " values are needed (n_rows * num_classes)",
                ),
            )

        var n = Int(n_rows)
        var features = _copy_f64(data, n * Int(n_features))
        var values = List[Float64](capacity=needed)
        if model[].kind == _MULTICLASS:
            ref multi = model[].multi.value()
            for r in range(n):
                var row = _row(features, n, expected, r)
                var scores: List[Float64]
                if raw:
                    scores = multi.predict_raw(row)
                else:
                    scores = multi.predict_proba(row)
                for k in range(width):
                    values.append(scores[k])
        else:
            ref single = model[].single.value()
            for r in range(n):
                var row = _row(features, n, expected, r)
                if raw:
                    values.append(single.predict_raw(row))
                else:
                    values.append(single.predict(row))

        for i in range(needed):
            out_values.unsafe_store(i, values[i])
        return OK
    except e:
        return _fail(error, ERROR_INVALID_ARGUMENT, String(e))


@export
def mojoboost_predict(
    model: _ModelPtr,
    data: _F64Ptr,
    n_rows: Int64,
    n_features: Int64,
    out_values: _F64Ptr,
    out_len: Int64,
    error: _ErrorPtr,
) abi("C") -> Int32:
    return _predict_into(
        model, data, n_rows, n_features, out_values, out_len, error, False
    )


@export
def mojoboost_predict_raw(
    model: _ModelPtr,
    data: _F64Ptr,
    n_rows: Int64,
    n_features: Int64,
    out_values: _F64Ptr,
    out_len: Int64,
    error: _ErrorPtr,
) abi("C") -> Int32:
    return _predict_into(
        model, data, n_rows, n_features, out_values, out_len, error, True
    )


# -------------------------------------------------------------- save / load


@export
def mojoboost_save_model(
    model: _ModelPtr, path: _CharPtr, error: _ErrorPtr
) abi("C") -> Int32:
    _clear(error)
    if _is_null(model):
        return _invalid(error, String("model must not be NULL"))
    if _is_null(path):
        return _invalid(error, String("path must not be NULL"))
    var target = _c_string(path)
    if target.byte_length() == 0:
        return _invalid(error, String("path must not be empty"))
    try:
        if model[].kind == _MULTICLASS:
            save_multiclass_model(model[].multi.value(), target)
        else:
            save_model(model[].single.value(), target)
    except e:
        return _fail(error, ERROR_IO, String(e))
    return OK


@export
def mojoboost_load_model(
    path: _CharPtr, out_model: _ModelOutPtr, error: _ErrorPtr
) abi("C") -> Int32:
    _clear(error)
    if _is_null(path):
        return _invalid(error, String("path must not be NULL"))
    if _is_null(out_model):
        return _invalid(error, String("out_model must not be NULL"))
    var source = _c_string(path)
    if source.byte_length() == 0:
        return _invalid(error, String("path must not be empty"))

    var box: _ModelBox
    try:
        if model_file_kind(source) == "multiclass":
            box = _ModelBox(load_multiclass_model(source))
        else:
            box = _ModelBox(load_model(source))
    except e:
        return _fail(error, ERROR_IO, String(e))

    out_model[] = _new_handle(box^)
    return OK


# ---------------------------------------------------------------- accessors


def _accessor(
    model: _ModelPtr, out_value: _I64Ptr, error: _ErrorPtr, which: Int
) -> Int32:
    _clear(error)
    try:
        if _is_null(model):
            return _invalid(error, String("model must not be NULL"))
        if _is_null(out_value):
            return _invalid(error, String("out_value must not be NULL"))
        if which == 0:
            out_value[] = Int64(model[].n_features())
        elif which == 1:
            out_value[] = Int64(model[].n_classes())
        else:
            out_value[] = Int64(model[].n_trees())
        return OK
    except e:
        return _invalid(error, String(e))


@export
def mojoboost_model_num_features(
    model: _ModelPtr, out_value: _I64Ptr, error: _ErrorPtr
) abi("C") -> Int32:
    return _accessor(model, out_value, error, 0)


@export
def mojoboost_model_num_classes(
    model: _ModelPtr, out_value: _I64Ptr, error: _ErrorPtr
) abi("C") -> Int32:
    return _accessor(model, out_value, error, 1)


@export
def mojoboost_model_num_trees(
    model: _ModelPtr, out_value: _I64Ptr, error: _ErrorPtr
) abi("C") -> Int32:
    return _accessor(model, out_value, error, 2)


# ------------------------------------------------------------------ destroy


@export
def mojoboost_model_free(model: _ModelPtr) abi("C"):
    if _is_null(model):
        return
    model.unsafe_deinit_pointee()
    model.unsafe_free()
