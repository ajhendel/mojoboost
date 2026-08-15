"""Tests for the parameter strings and the C ABI.

The C caller's view is tested from C, in capi/test_capi.c, which is the only
place a NULL pointer or a stale handle can be constructed at all: Mojo's
`Pointer` is non-nullable, so a Mojo test cannot even express those calls.
What is tested here is what C cannot see: that the ABI produces exactly what
the Mojo API produces, that the header and the implementation agree on their
constants, and that the parameter string is parsed the way both front ends
document.

Run with `mojo run -I src -I capi tests/test_capi.mojo`.
"""

from std.os import remove
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.boosting import BoosterParams, L1, QUANTILE, SQUARED_ERROR
from mojotrees.device import CPU_DEVICE
from mojotrees.model import fit, fit_multiclass
from mojotrees.params import (
    MULTICLASS,
    objective_display_name,
    objective_from_name,
    params_names_mojo_api_only,
    parse_params,
)

from mojotrees_capi import (
    ABI_VERSION,
    ERROR_INVALID_ARGUMENT,
    ERROR_IO,
    ERROR_TRAINING,
    ERROR_UNSUPPORTED,
    OK,
    _ErrorPtr,
    _ModelOutPtr,
    _ModelPtr,
    mojotrees_abi_version,
    mojotrees_error_create,
    mojotrees_error_free,
    mojotrees_error_message,
    mojotrees_library_version,
    mojotrees_load_model,
    mojotrees_model_free,
    mojotrees_model_num_classes,
    mojotrees_model_num_features,
    mojotrees_model_num_trees,
    mojotrees_predict,
    mojotrees_predict_raw,
    mojotrees_save_model,
    mojotrees_train_dense,
)

comptime _MODEL_PATH = "./.test_capi_model.tmp"

comptime _F64Ptr = Pointer[Float64, MutUntrackedOrigin]
comptime _I64Ptr = Pointer[Int64, MutUntrackedOrigin]
comptime _I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime _CharPtr = Pointer[UInt8, MutUntrackedOrigin]


# ----------------------------------------------------------------- helpers


def _f64_ptr(mut buf: List[Float64]) -> _F64Ptr:
    return _F64Ptr(unsafe_from_address=Int(buf.unsafe_ptr()))


def _c_bytes(s: String) -> List[UInt8]:
    """A NUL terminated copy of `s`, kept alive by the caller while its
    address is in use, exactly as a C caller would."""
    var bytes = List[UInt8](capacity=s.byte_length() + 1)
    for b in s.as_bytes():
        bytes.append(b)
    bytes.append(0)
    return bytes^


def _c_ptr(mut bytes: List[UInt8]) -> _CharPtr:
    return _CharPtr(unsafe_from_address=Int(bytes.unsafe_ptr()))


def _message(error: _ErrorPtr) -> String:
    """The error text, read back the way C reads it: a NUL terminated
    string at the address the ABI hands out."""
    var address = mojotrees_error_message(error)
    if address == 0:
        return String("")
    var p = _CharPtr(unsafe_from_address=address)
    var bytes = List[UInt8]()
    var i = 0
    while True:
        var b = p.unsafe_load(i)
        if b == 0:
            break
        bytes.append(b)
        i += 1
    return String(StringSlice(unsafe_from_utf8=Span(bytes)))


struct _Slot(Movable):
    """One pointer-sized cell to receive a handle, which is how a C caller
    passes `MojoTreesModel **`."""

    var cell: List[Int]

    def __init__(out self):
        self.cell = [0]

    def out_ptr(mut self) -> _ModelOutPtr:
        return _ModelOutPtr(unsafe_from_address=Int(self.cell.unsafe_ptr()))

    def address(self) -> Int:
        return self.cell[0]

    def handle(self) raises -> _ModelPtr:
        if self.cell[0] == 0:
            raise Error("no handle was written")
        return _ModelPtr(unsafe_from_address=self.cell[0])


def _int_out(mut cell: List[Int64]) -> _I64Ptr:
    return _I64Ptr(unsafe_from_address=Int(cell.unsafe_ptr()))


def _dataset(n_rows: Int, n_features: Int) -> List[Float64]:
    """Deterministic column-major features from the splitmix64 style stream
    the rest of the suite uses."""
    var features = List[Float64](capacity=n_rows * n_features)
    var state: UInt64 = 0x9E3779B97F4A7C15
    for _ in range(n_rows * n_features):
        state = state * 6364136223846793005 + 1442695040888963407
        features.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    return features^


def _regression_target(n_rows: Int, features: List[Float64]) -> List[Float64]:
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(2.0 * features[r] - features[n_rows + r])
    return target^


def _ones(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(1.0)
    return out^


# -------------------------------------------------------- parameter string


def test_params_defaults_match_the_mojo_defaults() raises:
    var config = parse_params(String(""))
    var defaults = BoosterParams.default()
    assert_equal(config.objective, SQUARED_ERROR)
    assert_equal(config.n_classes, 1)
    assert_equal(config.device, CPU_DEVICE)
    assert_true(config.use_missing)
    assert_equal(config.max_bin, 255)
    assert_equal(config.booster.n_estimators, defaults.n_estimators)
    assert_equal(config.booster.learning_rate, defaults.learning_rate)
    assert_equal(config.booster.tree.num_leaves, defaults.tree.num_leaves)
    assert_equal(
        config.booster.tree.min_data_in_leaf, defaults.tree.min_data_in_leaf
    )
    assert_equal(config.booster.tree.lambda_reg, defaults.tree.lambda_reg)
    assert_equal(config.booster.tree.lambda_l1, defaults.tree.lambda_l1)


def test_params_parses_every_supported_key() raises:
    var config = parse_params(
        String(
            "objective=quantile alpha=0.25 num_iterations=7"
            " learning_rate=0.03 num_leaves=9 min_data_in_leaf=3"
            " min_sum_hessian_in_leaf=0.5 lambda_l1=0.25 lambda_l2=2.5"
            " max_depth=4 feature_fraction=0.8 feature_fraction_bynode=0.9"
            " feature_fraction_seed=11 max_bin=63 device=cpu"
            " use_missing=false"
        )
    )
    assert_equal(config.objective, QUANTILE)
    assert_equal(config.alpha, 0.25)
    assert_equal(config.booster.n_estimators, 7)
    assert_equal(config.booster.learning_rate, 0.03)
    assert_equal(config.booster.tree.num_leaves, 9)
    assert_equal(config.booster.tree.min_data_in_leaf, 3)
    assert_equal(config.booster.tree.min_child_hess, 0.5)
    assert_equal(config.booster.tree.lambda_l1, 0.25)
    assert_equal(config.booster.tree.lambda_reg, 2.5)
    assert_equal(config.booster.tree.max_depth, 4)
    assert_equal(config.booster.tree.feature_fraction, 0.8)
    assert_equal(config.booster.tree.feature_fraction_bynode, 0.9)
    assert_equal(config.booster.tree.feature_fraction_seed, 11)
    assert_equal(config.max_bin, 63)
    assert_equal(config.device, CPU_DEVICE)
    assert_true(not config.use_missing)


def test_params_accepts_lightgbm_aliases() raises:
    var canonical = parse_params(
        String(
            "objective=mae num_iterations=5 learning_rate=0.2 lambda_l1=0.5"
            " lambda_l2=3.0 min_data_in_leaf=7"
        )
    )
    var aliased = parse_params(
        String(
            "application=regression_l1 num_round=5 eta=0.2 reg_alpha=0.5"
            " reg_lambda=3.0 min_child_samples=7"
        )
    )
    assert_equal(canonical.objective, L1)
    assert_equal(aliased.objective, canonical.objective)
    assert_equal(aliased.booster.n_estimators, canonical.booster.n_estimators)
    assert_equal(
        aliased.booster.learning_rate, canonical.booster.learning_rate
    )
    assert_equal(
        aliased.booster.tree.lambda_l1, canonical.booster.tree.lambda_l1
    )
    assert_equal(
        aliased.booster.tree.lambda_reg, canonical.booster.tree.lambda_reg
    )
    assert_equal(
        aliased.booster.tree.min_data_in_leaf,
        canonical.booster.tree.min_data_in_leaf,
    )


def test_params_objective_names_round_trip() raises:
    var names = [
        String("regression"),
        String("binary"),
        String("poisson"),
        String("huber"),
        String("quantile"),
        String("mae"),
        String("multiclass"),
    ]
    for i in range(len(names)):
        var code = objective_from_name(names[i])
        assert_equal(objective_display_name(code), names[i])
    assert_equal(objective_from_name(String("softmax")), MULTICLASS)
    assert_equal(
        objective_from_name(String("l2")), objective_from_name(String("mse"))
    )


def test_params_rejects_malformed_and_out_of_range() raises:
    with assert_raises():
        _ = parse_params(String("num_leaves"))
    with assert_raises():
        _ = parse_params(String("num_leaves="))
    with assert_raises():
        _ = parse_params(String("=31"))
    with assert_raises():
        _ = parse_params(String("not_a_parameter=1"))
    with assert_raises():
        _ = parse_params(String("num_leaves=many"))
    with assert_raises():
        _ = parse_params(String("num_leaves=1"))
    with assert_raises():
        _ = parse_params(String("learning_rate=0"))
    with assert_raises():
        _ = parse_params(String("num_iterations=-1"))
    with assert_raises():
        _ = parse_params(String("lambda_l1=-0.5"))
    with assert_raises():
        _ = parse_params(String("max_bin=1"))
    with assert_raises():
        _ = parse_params(String("feature_fraction=0"))
    with assert_raises():
        _ = parse_params(String("feature_fraction=1.5"))
    with assert_raises():
        _ = parse_params(String("use_missing=maybe"))
    with assert_raises():
        _ = parse_params(String("device=tpu"))
    with assert_raises():
        _ = parse_params(String("objective=nonesuch"))


def test_params_multiclass_requires_num_class() raises:
    with assert_raises():
        _ = parse_params(String("objective=multiclass"))
    with assert_raises():
        _ = parse_params(String("objective=multiclass num_class=1"))
    with assert_raises():
        # num_class means nothing without the multiclass objective, so
        # accepting it silently would hide a mistake.
        _ = parse_params(String("objective=binary num_class=3"))
    var config = parse_params(String("objective=multiclass num_class=4"))
    assert_true(config.is_multiclass())
    assert_equal(config.n_classes, 4)


def test_params_separates_unsupported_from_unknown() raises:
    # Real LightGBM features that mojotrees has, reachable from the Mojo API
    # but not from a parameter string.
    assert_true(params_names_mojo_api_only(String("bagging_fraction=0.5")))
    assert_true(params_names_mojo_api_only(String("objective=lambdarank")))
    assert_true(params_names_mojo_api_only(String("objective=custom")))
    assert_true(
        params_names_mojo_api_only(String("num_leaves=7 top_rate=0.2"))
    )
    # Typos and plain mistakes are not.
    assert_true(not params_names_mojo_api_only(String("not_a_parameter=1")))
    assert_true(not params_names_mojo_api_only(String("num_leaves=7")))
    assert_true(not params_names_mojo_api_only(String("objective=binary")))
    assert_true(not params_names_mojo_api_only(String("")))
    # It must not raise on input parse_params rejects outright, since it is
    # called while already handling that error.
    assert_true(not params_names_mojo_api_only(String("garbage")))


# ------------------------------------------------------------ ABI contract


def _header_define(source: String, name: String) raises -> Int:
    """The integer a `#define NAME value` line gives `name`, with C's
    parentheses around negative constants removed."""
    var prefix = String("#define ", name, " ")
    for line_slice in source.split("\n"):
        var line = String(String(line_slice).strip())
        if not line.startswith(prefix):
            continue
        var text = String(String(line[byte = prefix.byte_length() :]).strip())
        if text.startswith(String("(")) and text.endswith(String(")")):
            var inner = String(text[byte = 1 : text.byte_length() - 1])
            text = inner^
        return Int(text)
    raise Error("capi/mojotrees.h does not define ", name)


def test_header_and_implementation_agree() raises:
    """The header is the contract a compiled caller was built against, so
    drift between it and the exported constants is a defect even though
    nothing links the two."""
    var header = open("capi/mojotrees.h", "r").read()
    assert_equal(
        _header_define(header, String("MOJOTREES_ABI_VERSION")),
        Int(ABI_VERSION),
    )
    assert_equal(Int(mojotrees_abi_version()), Int(ABI_VERSION))
    assert_equal(_header_define(header, String("MOJOTREES_OK")), Int(OK))
    assert_equal(
        _header_define(header, String("MOJOTREES_ERROR_INVALID_ARGUMENT")),
        Int(ERROR_INVALID_ARGUMENT),
    )
    assert_equal(
        _header_define(header, String("MOJOTREES_ERROR_TRAINING")),
        Int(ERROR_TRAINING),
    )
    assert_equal(
        _header_define(header, String("MOJOTREES_ERROR_IO")), Int(ERROR_IO)
    )
    assert_equal(
        _header_define(header, String("MOJOTREES_ERROR_UNSUPPORTED")),
        Int(ERROR_UNSUPPORTED),
    )


def test_library_version_fills_every_out_pointer() raises:
    var parts: List[Int32] = [0, 0, 0]
    var base = Int(parts.unsafe_ptr())
    mojotrees_library_version(
        _I32Ptr(unsafe_from_address=base),
        _I32Ptr(unsafe_from_address=base + 4),
        _I32Ptr(unsafe_from_address=base + 8),
    )
    assert_true(parts[0] >= 0)
    assert_true(parts[1] >= 0)
    assert_true(parts[2] >= 0)


# ----------------------------------------------------- ABI matches the API


def test_capi_regression_matches_the_mojo_api() raises:
    """Bit-exact, not approximate: the ABI is a call path into the same
    trainer, so any difference is a bug in the boundary."""
    var n_rows = 200
    var n_features = 3
    var features = _dataset(n_rows, n_features)
    var target = _regression_target(n_rows, features)
    var weights = _ones(n_rows)

    var spec = String(
        "objective=regression num_iterations=12 learning_rate=0.2"
        " num_leaves=7 min_data_in_leaf=5"
    )
    var config = parse_params(spec)
    var reference = fit(
        features,
        n_rows,
        n_features,
        target,
        config.objective,
        config.booster,
        config.max_bin,
        weights,
        config.alpha,
    )

    var error = mojotrees_error_create()
    var params = _c_bytes(spec)
    var slot = _Slot()
    assert_equal(
        Int(
            mojotrees_train_dense(
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(target),
                _f64_ptr(weights),
                _c_ptr(params),
                slot.out_ptr(),
                error,
            )
        ),
        Int(OK),
    )
    assert_equal(_message(error), String(""))
    var model = slot.handle()

    var counts: List[Int64] = [0]
    assert_equal(
        Int(mojotrees_model_num_features(model, _int_out(counts), error)),
        Int(OK),
    )
    assert_equal(Int(counts[0]), n_features)
    assert_equal(
        Int(mojotrees_model_num_classes(model, _int_out(counts), error)),
        Int(OK),
    )
    assert_equal(Int(counts[0]), 1)
    assert_equal(
        Int(mojotrees_model_num_trees(model, _int_out(counts), error)),
        Int(OK),
    )
    assert_equal(Int(counts[0]), len(reference.booster.trees))

    var predictions = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        predictions.append(0.0)
    assert_equal(
        Int(
            mojotrees_predict(
                model,
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(predictions),
                Int64(n_rows),
                error,
            )
        ),
        Int(OK),
    )
    for r in range(n_rows):
        var row = List[Float64](capacity=n_features)
        for f in range(n_features):
            row.append(features[f * n_rows + r])
        assert_equal(predictions[r], reference.predict(row))

    var raw = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        raw.append(0.0)
    assert_equal(
        Int(
            mojotrees_predict_raw(
                model,
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(raw),
                Int64(n_rows),
                error,
            )
        ),
        Int(OK),
    )
    for r in range(n_rows):
        var row = List[Float64](capacity=n_features)
        for f in range(n_features):
            row.append(features[f * n_rows + r])
        assert_equal(raw[r], reference.predict_raw(row))

    mojotrees_model_free(model)
    mojotrees_error_free(error)
    # The ABI takes addresses, and Mojo frees a value at its last use, so
    # every buffer whose address crossed the boundary is named once more
    # here to keep it alive for the calls above.
    _ = features^
    _ = target^
    _ = weights^
    _ = params^
    _ = predictions^
    _ = raw^
    _ = counts^


def test_capi_multiclass_matches_the_mojo_api() raises:
    var n_rows = 150
    var n_features = 3
    var features = _dataset(n_rows, n_features)
    var weights = _ones(n_rows)
    var target = List[Float64](capacity=n_rows)
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        var k = Int(3.0 * features[r]) % 3
        labels.append(k)
        target.append(Float64(k))

    var spec = String(
        "objective=multiclass num_class=3 num_iterations=6 num_leaves=7"
        " min_data_in_leaf=5"
    )
    var config = parse_params(spec)
    var reference = fit_multiclass(
        features,
        n_rows,
        n_features,
        labels,
        config.n_classes,
        config.booster,
        config.max_bin,
        weights,
    )

    var error = mojotrees_error_create()
    var params = _c_bytes(spec)
    var slot = _Slot()
    assert_equal(
        Int(
            mojotrees_train_dense(
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(target),
                _f64_ptr(weights),
                _c_ptr(params),
                slot.out_ptr(),
                error,
            )
        ),
        Int(OK),
    )
    var model = slot.handle()

    var counts: List[Int64] = [0]
    assert_equal(
        Int(mojotrees_model_num_classes(model, _int_out(counts), error)),
        Int(OK),
    )
    assert_equal(Int(counts[0]), 3)

    var proba = List[Float64](capacity=n_rows * 3)
    for _ in range(n_rows * 3):
        proba.append(0.0)
    assert_equal(
        Int(
            mojotrees_predict(
                model,
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(proba),
                Int64(n_rows * 3),
                error,
            )
        ),
        Int(OK),
    )
    for r in range(n_rows):
        var row = List[Float64](capacity=n_features)
        for f in range(n_features):
            row.append(features[f * n_rows + r])
        var expected = reference.predict_proba(row)
        for k in range(3):
            assert_equal(proba[r * 3 + k], expected[k])

    # An output buffer sized for one value per row is short for multiclass.
    var narrow = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        narrow.append(0.0)
    assert_equal(
        Int(
            mojotrees_predict(
                model,
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(narrow),
                Int64(n_rows),
                error,
            )
        ),
        Int(ERROR_INVALID_ARGUMENT),
    )

    mojotrees_model_free(model)
    mojotrees_error_free(error)
    # The ABI takes addresses, and Mojo frees a value at its last use, so
    # every buffer whose address crossed the boundary is named once more
    # here to keep it alive for the calls above.
    _ = features^
    _ = target^
    _ = weights^
    _ = params^
    _ = proba^
    _ = narrow^
    _ = counts^


def test_capi_save_load_round_trip() raises:
    var n_rows = 120
    var n_features = 3
    var features = _dataset(n_rows, n_features)
    var target = _regression_target(n_rows, features)
    var weights = _ones(n_rows)

    var error = mojotrees_error_create()
    var params = _c_bytes(String("num_iterations=8 num_leaves=7"))
    var slot = _Slot()
    assert_equal(
        Int(
            mojotrees_train_dense(
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(target),
                _f64_ptr(weights),
                _c_ptr(params),
                slot.out_ptr(),
                error,
            )
        ),
        Int(OK),
    )
    var model = slot.handle()

    var before = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        before.append(0.0)
    _ = mojotrees_predict(
        model,
        _f64_ptr(features),
        Int64(n_rows),
        Int64(n_features),
        _f64_ptr(before),
        Int64(n_rows),
        error,
    )

    var path = _c_bytes(String(_MODEL_PATH))
    assert_equal(
        Int(mojotrees_save_model(model, _c_ptr(path), error)), Int(OK)
    )
    mojotrees_model_free(model)

    var loaded_slot = _Slot()
    assert_equal(
        Int(
            mojotrees_load_model(
                _c_ptr(path), loaded_slot.out_ptr(), error
            )
        ),
        Int(OK),
    )
    var loaded = loaded_slot.handle()

    var after = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        after.append(0.0)
    assert_equal(
        Int(
            mojotrees_predict(
                loaded,
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(after),
                Int64(n_rows),
                error,
            )
        ),
        Int(OK),
    )
    for r in range(n_rows):
        assert_equal(after[r], before[r])

    mojotrees_model_free(loaded)
    mojotrees_error_free(error)
    remove(_MODEL_PATH)
    # The ABI takes addresses, and Mojo frees a value at its last use, so
    # every buffer whose address crossed the boundary is named once more
    # here to keep it alive for the calls above.
    _ = features^
    _ = target^
    _ = weights^
    _ = params^
    _ = path^
    _ = before^
    _ = after^


def test_capi_status_codes_and_messages() raises:
    """Each failure class has its own status code and a message that says
    what went wrong."""
    var n_rows = 60
    var n_features = 2
    var features = _dataset(n_rows, n_features)
    var target = _regression_target(n_rows, features)
    var weights = _ones(n_rows)
    var error = mojotrees_error_create()

    # A typo is an invalid argument.
    var bad = _c_bytes(String("num_leavs=7"))
    var slot = _Slot()
    assert_equal(
        Int(
            mojotrees_train_dense(
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(target),
                _f64_ptr(weights),
                _c_ptr(bad),
                slot.out_ptr(),
                error,
            )
        ),
        Int(ERROR_INVALID_ARGUMENT),
    )
    assert_true(_message(error).find("num_leavs") >= 0)
    assert_equal(slot.address(), 0)

    # A real feature asked for the wrong way is unsupported.
    var unsupported = _c_bytes(String("bagging_fraction=0.5"))
    assert_equal(
        Int(
            mojotrees_train_dense(
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(target),
                _f64_ptr(weights),
                _c_ptr(unsupported),
                slot.out_ptr(),
                error,
            )
        ),
        Int(ERROR_UNSUPPORTED),
    )
    assert_true(_message(error).find("Mojo API") >= 0)
    assert_equal(slot.address(), 0)

    # Labels the objective cannot accept are a training failure.
    var negative = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        negative.append(-1.0)
    var poisson = _c_bytes(String("objective=poisson num_iterations=2"))
    assert_equal(
        Int(
            mojotrees_train_dense(
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(negative),
                _f64_ptr(weights),
                _c_ptr(poisson),
                slot.out_ptr(),
                error,
            )
        ),
        Int(ERROR_TRAINING),
    )
    assert_equal(slot.address(), 0)

    # A missing file is an I/O failure.
    var missing = _c_bytes(String("./.test_capi_no_such_model.tmp"))
    assert_equal(
        Int(mojotrees_load_model(_c_ptr(missing), slot.out_ptr(), error)),
        Int(ERROR_IO),
    )
    assert_equal(slot.address(), 0)

    # A successful call clears the message the previous failure left.
    var good = _c_bytes(String("num_iterations=2 num_leaves=7"))
    assert_equal(
        Int(
            mojotrees_train_dense(
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(target),
                _f64_ptr(weights),
                _c_ptr(good),
                slot.out_ptr(),
                error,
            )
        ),
        Int(OK),
    )
    assert_equal(_message(error), String(""))
    assert_true(slot.address() != 0)

    var model = slot.handle()
    # Prediction validates its shapes against the model.
    var out = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        out.append(0.0)
    assert_equal(
        Int(
            mojotrees_predict(
                model,
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features - 1),
                _f64_ptr(out),
                Int64(n_rows),
                error,
            )
        ),
        Int(ERROR_INVALID_ARGUMENT),
    )
    assert_true(_message(error).find("trained on") >= 0)
    assert_equal(
        Int(
            mojotrees_predict(
                model,
                _f64_ptr(features),
                Int64(n_rows),
                Int64(n_features),
                _f64_ptr(out),
                Int64(n_rows - 1),
                error,
            )
        ),
        Int(ERROR_INVALID_ARGUMENT),
    )
    assert_equal(out[0], 0.0)

    mojotrees_model_free(model)
    mojotrees_error_free(error)
    # The ABI takes addresses, and Mojo frees a value at its last use, so
    # every buffer whose address crossed the boundary is named once more
    # here to keep it alive for the calls above.
    _ = features^
    _ = target^
    _ = weights^
    _ = negative^
    _ = out^
    _ = bad^
    _ = unsupported^
    _ = poisson^
    _ = missing^
    _ = good^


def test_capi_handle_churn() raises:
    """Many create/destroy cycles, so a handle that outlives its owner or a
    double free shows up here rather than in a caller. capi/test_capi.c runs
    the same shape under a leak checker."""
    for _ in range(200):
        var error = mojotrees_error_create()
        mojotrees_error_free(error)

    var n_rows = 40
    var n_features = 2
    var features = _dataset(n_rows, n_features)
    var target = _regression_target(n_rows, features)
    var weights = _ones(n_rows)
    var params = _c_bytes(String("num_iterations=2 num_leaves=4"))
    for _ in range(10):
        var error = mojotrees_error_create()
        var slot = _Slot()
        assert_equal(
            Int(
                mojotrees_train_dense(
                    _f64_ptr(features),
                    Int64(n_rows),
                    Int64(n_features),
                    _f64_ptr(target),
                    _f64_ptr(weights),
                    _c_ptr(params),
                    slot.out_ptr(),
                    error,
                )
            ),
            Int(OK),
        )
        mojotrees_model_free(slot.handle())
        mojotrees_error_free(error)
    # The ABI takes addresses, and Mojo frees a value at its last use, so
    # every buffer whose address crossed the boundary is named once more
    # here to keep it alive for the calls above.
    _ = features^
    _ = target^
    _ = weights^
    _ = params^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
