"""Implementation of the mojotrees C ABI declared in capi/mojotrees.h.

Build with `capi/build.sh`, which emits a shared library exporting exactly
the symbols in that header.

Nothing in here is a mojotrees type as far as a C caller is concerned:
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

A third rule keeps it honest: nothing here reimplements mojotrees. Training
goes to `fit`/`fit_multiclass`, prediction to `Model.predict_batch`,
inspection to `dump_model`, the device decision to `resolve_device`, and
the file format to serialize.mojo. This file only translates C arguments
into those calls and their failures into status codes, which is why a
change to how mojotrees bins, walks trees, or picks a device reaches C
callers without anyone editing the ABI.
"""

from std.memory.alloc import unsafe_alloc

from mojotrees.boosting import IterationRange
from mojotrees.device import (
    AUTO_DEVICE,
    CPU_DEVICE,
    GPU_DEVICE,
    gpu_available,
    resolve_device,
)
from mojotrees.importance import gain_importance, split_importance
from mojotrees.inspection import dump_model, dump_multiclass_model
from mojotrees.model import Model, MulticlassModel, fit, fit_multiclass
from mojotrees.params import (
    SUPPORTED_KEYS,
    TrainConfig,
    params_names_mojo_api_only,
    parse_params,
)
from mojotrees.serialize import (
    load_model,
    load_multiclass_model,
    model_file_kind,
    save_model,
    save_multiclass_model,
)

# Keep in sync with MOJOTREES_ABI_VERSION in capi/mojotrees.h. Each version is
# a strict superset of the one before: they only add declarations, so a caller
# compiled against any earlier version keeps working unchanged.
comptime ABI_VERSION: Int32 = 3

# Importance types, matching MOJOTREES_IMPORTANCE_* in capi/mojotrees.h.
comptime IMPORTANCE_SPLIT: Int32 = 0
comptime IMPORTANCE_GAIN: Int32 = 1

# Keep in sync with python/pyproject.toml and pixi.toml.
comptime VERSION_MAJOR: Int32 = 0
comptime VERSION_MINOR: Int32 = 1
comptime VERSION_PATCH: Int32 = 0

comptime OK: Int32 = 0
comptime ERROR_INVALID_ARGUMENT: Int32 = -1
comptime ERROR_TRAINING: Int32 = -2
comptime ERROR_IO: Int32 = -3
comptime ERROR_UNSUPPORTED: Int32 = -4

# The C spelling of the device vocabulary. These are defined from the codes
# in device.mojo rather than written out, so the ABI cannot drift from the
# policy module that owns the meaning of each value.
comptime DEVICE_CPU: Int32 = Int32(CPU_DEVICE)
comptime DEVICE_GPU: Int32 = Int32(GPU_DEVICE)
comptime DEVICE_AUTO: Int32 = Int32(AUTO_DEVICE)

# Prediction flags. Only one bit is defined; anything else is rejected, so a
# caller cannot silently get response-scale output by setting a flag this
# build does not know about.
comptime PREDICT_RESPONSE: Int32 = 0
comptime PREDICT_RAW: Int32 = 1

comptime _SINGLE = 0
comptime _MULTICLASS = 1

comptime _CharPtr = Pointer[UInt8, MutUntrackedOrigin]
comptime _F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime _I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime _I64Ptr = Pointer[Int64, MutUntrackedOrigin]


struct _ErrorBox(Movable):
    """The storage behind a `MojoTreesError`.

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
    """The storage behind a `MojoTreesModel`.

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

    def n_iterations(self) raises -> Int:
        """Boosting iterations, which is the tree count for a single-output
        model and the tree count over the class count for a multiclass one.
        This, not the tree count, is what an iteration range is expressed
        in."""
        if self.kind == _MULTICLASS:
            return self.multi.value().booster.n_iterations()
        return self.single.value().booster.n_iterations()

    def predict_batch(
        self,
        features: List[Float64],
        n_rows: Int,
        rng: IterationRange,
        raw: Bool,
        device: Int,
    ) raises -> List[Float64]:
        """Row-major `[r * num_classes + k]` predictions for a raw
        column-major matrix, straight from the model's own batched
        prediction path (model.mojo). Going through `predict_batch` rather
        than looping over the per-row `predict` is what puts the C ABI on
        the same code as the Mojo and Python front ends: one binning pass
        for the whole matrix, iteration ranges, and the device dispatch."""
        if self.kind == _MULTICLASS:
            return self.multi.value().predict_batch(
                features, n_rows, rng, raw, device
            )
        return self.single.value().predict_batch(
            features, n_rows, rng, raw, device
        )

    def dump_json(self) raises -> String:
        """The model in the inspection schema (docs/MODEL_INSPECTION_SCHEMA.md),
        from the one implementation in inspection.mojo. A model carries no
        feature names, so the dump uses the default `Column_0`, `Column_1`,
        ... names."""
        if self.kind == _MULTICLASS:
            return dump_multiclass_model(self.multi.value())
        return dump_model(self.single.value())

    def importance(self, kind: Int32) raises -> List[Float64]:
        """Per-feature importance, from the one implementation in
        importance.mojo rather than a count kept here.

        Both importance functions take a flat tree list and neither cares
        which class a tree belongs to, so the multiclass sum over classes
        falls out of passing the whole ensemble instead of needing a second
        code path. Split counts come back as integers and are widened here,
        which is what lets one C signature serve both types."""
        var n = self.n_features()
        if self.kind == _MULTICLASS:
            if kind == IMPORTANCE_GAIN:
                return gain_importance(self.multi.value().booster.trees, n)
            return _widen(
                split_importance(self.multi.value().booster.trees, n)
            )
        if kind == IMPORTANCE_GAIN:
            return gain_importance(self.single.value().booster.trees, n)
        return _widen(split_importance(self.single.value().booster.trees, n))


def _widen(var counts: List[Int]) -> List[Float64]:
    """Split counts as doubles, so one C signature serves both importance
    types."""
    var out = List[Float64](capacity=len(counts))
    for i in range(len(counts)):
        out.append(Float64(counts[i]))
    return out^


comptime _ModelPtr = Pointer[_ModelBox, MutUntrackedOrigin]
comptime _ErrorPtr = Pointer[_ErrorBox, MutUntrackedOrigin]
comptime _ModelOutPtr = Pointer[_ModelPtr, MutUntrackedOrigin]
comptime _CharOutPtr = Pointer[_CharPtr, MutUntrackedOrigin]


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


def _new_c_string(text: String) -> _CharPtr:
    """A freshly allocated, NUL terminated copy of `text`, to be released by
    `mojotrees_string_free`. This is the ABI's second ownership class: a
    plain `char *` rather than an opaque handle, because a caller wants to
    read it with ordinary string functions."""
    var n = text.byte_length()
    var p = unsafe_alloc[UInt8](n + 1)
    var i = 0
    for b in text.as_bytes():
        p.unsafe_store(i, b)
        i += 1
    p.unsafe_store(n, 0)
    return p


# ------------------------------------------------------------------ version


@export
def mojotrees_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export
def mojotrees_library_version(
    major: _I32Ptr, minor: _I32Ptr, patch: _I32Ptr
) abi("C"):
    if not _is_null(major):
        major[] = VERSION_MAJOR
    if not _is_null(minor):
        minor[] = VERSION_MINOR
    if not _is_null(patch):
        patch[] = VERSION_PATCH


@export
def mojotrees_gpu_available() abi("C") -> Int32:
    """1 when `MOJOTREES_DEVICE_GPU` can be honored by this build on this
    machine, 0 otherwise. It answers exactly what `gpu_available()` in
    device.mojo answers, so the C ABI and the Mojo API agree on whether a
    request is worth making at all; whether a *particular* workload is
    covered is still decided per call by the policy (device_policy.mojo),
    which is why a 1 here is not a promise that every GPU request
    succeeds."""
    if gpu_available():
        return 1
    return 0


# -------------------------------------------------------------------- error


@export
def mojotrees_error_create() abi("C") -> _ErrorPtr:
    return _new_handle(_ErrorBox())


@export
def mojotrees_error_free(error: _ErrorPtr) abi("C"):
    if _is_null(error):
        return
    error.unsafe_deinit_pointee()
    error.unsafe_free()


@export
def mojotrees_error_message(error: _ErrorPtr) abi("C") -> Int:
    """Returns `const char *` as an address. Mojo's `Pointer` is
    non-nullable by construction, so the address is the only way to spell
    the NULL this returns for a NULL error object; `Int` and a pointer are
    returned in the same register on every ABI mojotrees targets."""
    if _is_null(error):
        return 0
    return Int(error[].text.unsafe_ptr())


# -------------------------------------------------------------------- train


@export
def mojotrees_train_dense(
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
    # CatBoost's data-dependent learning rate, if the spec asked for it
    # (`auto_learning_rate=true`, src/mojotrees/auto_learning_rate.mojo,
    # docs/design/CATBOOST_CATALOG.md A9). It needs the train row count, so
    # it cannot be resolved inside `parse_params`, and this is the first
    # point that has one. A no-op for every spec that did not ask: the
    # method returns `booster.learning_rate` untouched when the derivation
    # is disabled, which is the default.
    try:
        config.booster.learning_rate = config.resolved_learning_rate(n)
    except e:
        return _invalid(error, String(e))
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
    start_iteration: Int64,
    num_iteration: Int64,
    flags: Int32,
    device: Int32,
    out_values: _F64Ptr,
    out_len: Int64,
    error: _ErrorPtr,
) -> Int32:
    """The body behind all three prediction entry points.

    Everything the model knows how to do is reached from here through
    `_ModelBox.predict_batch`, which is `Model.predict_batch` in model.mojo:
    the ABI does not walk trees, slice iterations, or choose a device
    itself, it translates C arguments into that call. Predictions are
    computed in full before anything is written, so a failure leaves the
    caller's buffer untouched.
    """
    _clear(error)
    try:
        if _is_null(model):
            return _invalid(error, String("model must not be NULL"))
        var bad = _check_matrix(error, data, n_rows, n_features)
        if bad != OK:
            return bad
        if _is_null(out_values):
            return _invalid(error, String("out_values must not be NULL"))
        if flags != PREDICT_RESPONSE and flags != PREDICT_RAW:
            return _invalid(
                error,
                String(
                    "flags is ",
                    flags,
                    " but only 0 (response scale) and 1 (raw score) are"
                    " defined",
                ),
            )
        if (
            device != DEVICE_CPU
            and device != DEVICE_GPU
            and device != DEVICE_AUTO
        ):
            return _invalid(
                error,
                String(
                    "device is ",
                    device,
                    " but only 0 (cpu), 1 (gpu), and 2 (auto) are defined",
                ),
            )

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
        # Resolve the device before predicting, so a request the policy
        # refuses comes back as MOJOTREES_ERROR_UNSUPPORTED carrying the
        # policy's own reason, rather than as a generic invalid argument.
        # What comes back is CPU_DEVICE or GPU_DEVICE, never AUTO_DEVICE.
        var resolved: Int
        try:
            resolved = resolve_device(Int(device), n, expected, width)
        except e:
            return _fail(error, ERROR_UNSUPPORTED, String(e))

        # LightGBM's (start_iteration, num_iteration) convention, clamped by
        # the ensemble: a nonpositive num_iteration means every remaining
        # iteration, which is what the two legacy entry points pass.
        var rng = IterationRange.clamp(
            model[].n_iterations(), Int(start_iteration), Int(num_iteration)
        )

        var features = _copy_f64(data, n * expected)
        var values = model[].predict_batch(
            features, n, rng, flags == PREDICT_RAW, resolved
        )
        if len(values) != needed:
            return _invalid(
                error,
                String(
                    "internal error: prediction produced ",
                    len(values),
                    " values but ",
                    needed,
                    " were expected",
                ),
            )
        for i in range(needed):
            out_values.unsafe_store(i, values[i])
        return OK
    except e:
        return _fail(error, ERROR_INVALID_ARGUMENT, String(e))


@export
def mojotrees_predict(
    model: _ModelPtr,
    data: _F64Ptr,
    n_rows: Int64,
    n_features: Int64,
    out_values: _F64Ptr,
    out_len: Int64,
    error: _ErrorPtr,
) abi("C") -> Int32:
    return _predict_into(
        model,
        data,
        n_rows,
        n_features,
        0,
        0,
        PREDICT_RESPONSE,
        DEVICE_CPU,
        out_values,
        out_len,
        error,
    )


@export
def mojotrees_predict_raw(
    model: _ModelPtr,
    data: _F64Ptr,
    n_rows: Int64,
    n_features: Int64,
    out_values: _F64Ptr,
    out_len: Int64,
    error: _ErrorPtr,
) abi("C") -> Int32:
    return _predict_into(
        model,
        data,
        n_rows,
        n_features,
        0,
        0,
        PREDICT_RAW,
        DEVICE_CPU,
        out_values,
        out_len,
        error,
    )


@export
def mojotrees_predict_ex(
    model: _ModelPtr,
    data: _F64Ptr,
    n_rows: Int64,
    n_features: Int64,
    start_iteration: Int64,
    num_iteration: Int64,
    flags: Int32,
    device: Int32,
    out_values: _F64Ptr,
    out_len: Int64,
    error: _ErrorPtr,
) abi("C") -> Int32:
    """The full prediction surface: `mojotrees_predict` and
    `mojotrees_predict_raw` are this call with the whole ensemble, the CPU,
    and one flag fixed. Those two keep the established behavior exactly, so
    reaching the iteration range or the accelerator is an explicit choice a
    caller makes rather than something a rebuild changes underneath it."""
    return _predict_into(
        model,
        data,
        n_rows,
        n_features,
        start_iteration,
        num_iteration,
        flags,
        device,
        out_values,
        out_len,
        error,
    )


# -------------------------------------------------------------- save / load


@export
def mojotrees_save_model(
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
def mojotrees_load_model(
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
        elif which == 2:
            out_value[] = Int64(model[].n_trees())
        else:
            out_value[] = Int64(model[].n_iterations())
        return OK
    except e:
        return _invalid(error, String(e))


@export
def mojotrees_model_num_features(
    model: _ModelPtr, out_value: _I64Ptr, error: _ErrorPtr
) abi("C") -> Int32:
    return _accessor(model, out_value, error, 0)


@export
def mojotrees_model_num_classes(
    model: _ModelPtr, out_value: _I64Ptr, error: _ErrorPtr
) abi("C") -> Int32:
    return _accessor(model, out_value, error, 1)


@export
def mojotrees_model_num_trees(
    model: _ModelPtr, out_value: _I64Ptr, error: _ErrorPtr
) abi("C") -> Int32:
    return _accessor(model, out_value, error, 2)


@export
def mojotrees_model_num_iterations(
    model: _ModelPtr, out_value: _I64Ptr, error: _ErrorPtr
) abi("C") -> Int32:
    return _accessor(model, out_value, error, 3)


# ----------------------------------------------------------------- inspect


@export
def mojotrees_model_dump_json(
    model: _ModelPtr, out_text: _CharOutPtr, error: _ErrorPtr
) abi("C") -> Int32:
    """The model as inspection-schema JSON, allocated here and released with
    `mojotrees_string_free`. `out_text` is untouched on failure."""
    _clear(error)
    if _is_null(model):
        return _invalid(error, String("model must not be NULL"))
    if _is_null(out_text):
        return _invalid(error, String("out_text must not be NULL"))
    var text: String
    try:
        text = model[].dump_json()
    except e:
        return _fail(error, ERROR_INVALID_ARGUMENT, String(e))
    out_text[] = _new_c_string(text)
    return OK


@export
def mojotrees_feature_importance(
    model: _ModelPtr,
    importance_type: Int32,
    out_values: _F64Ptr,
    out_len: Int64,
    error: _ErrorPtr,
) abi("C") -> Int32:
    """Per-feature importance into a caller-owned buffer.

    The buffer is filled only after the whole computation has succeeded, so
    a failure leaves whatever the caller had there untouched rather than
    half-written."""
    _clear(error)
    if _is_null(model):
        return _invalid(error, String("model must not be NULL"))
    if _is_null(out_values):
        return _invalid(error, String("out_values must not be NULL"))
    if (
        importance_type != IMPORTANCE_SPLIT
        and importance_type != IMPORTANCE_GAIN
    ):
        return _invalid(
            error,
            String(
                "importance_type must be MOJOTREES_IMPORTANCE_SPLIT (0) or"
                " MOJOTREES_IMPORTANCE_GAIN (1), got ",
                importance_type,
            ),
        )
    var values: List[Float64]
    try:
        values = model[].importance(importance_type)
    except e:
        return _invalid(error, String(e))
    if out_len < Int64(len(values)):
        return _invalid(
            error,
            String(
                "out_len is ",
                out_len,
                " but this model has ",
                len(values),
                " features",
            ),
        )
    for i in range(len(values)):
        out_values.unsafe_store(i, values[i])
    return OK


@export
def mojotrees_parameter_keys(
    out_text: _CharOutPtr, error: _ErrorPtr
) abi("C") -> Int32:
    """The parser's own key list, so a binding never keeps a second copy."""
    _clear(error)
    if _is_null(out_text):
        return _invalid(error, String("out_text must not be NULL"))
    out_text[] = _new_c_string(SUPPORTED_KEYS)
    return OK


# ------------------------------------------------------------------ destroy


@export
def mojotrees_string_free(text: _CharPtr) abi("C"):
    if _is_null(text):
        return
    text.unsafe_free()


@export
def mojotrees_model_free(model: _ModelPtr) abi("C"):
    if _is_null(model):
        return
    model.unsafe_deinit_pointee()
    model.unsafe_free()
