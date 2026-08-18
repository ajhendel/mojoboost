"""The objective code across the Python boundary: -1 is softmax, -2 is silence.

WHAT THIS FILE IS FOR. Two negative integers one apart live in the objective
code space and mean opposite things:

    objective_registry.MULTICLASS   = -1   "this fit is softmax"
    device_policy.OBJECTIVE_UNSPECIFIED = -2   "the caller named no objective"

Three separate places have already conflated them, twice in ways that
shipped. `model.fit_multiclass` and `trainset.train_dataset_multiclass` both
passed `-1` to `resolve_device` and made every multiclass GPU fit raise
"objective code -1 is not one the built-in trainers implement"; the comments
at those two sites record it. `decide_device_workload` in
bindings/basic_bindings.mojo then folded every *negative* objective to
`OBJECTIVE_UNSPECIFIED` before calling `decide_device_report`, which folds
below `-1` and preserves `-1` -- two marshallers over one wire, disagreeing
on exactly the one value that matters, so the `-1` branch of
`_normalized_objective` was unreachable from Python and a softmax fit could
not declare itself at all.

HOW IT AVOIDS BEING VACUOUS. Nothing here trains, nothing here times
anything, and nothing here asserts against a fixture that could not be
produced by the real path. The Python-boundary test builds an actual CPython
dict and calls the actual registered binding, so it exercises the fold that
was there; it does not model it. It opens no device: `device="cpu"` and a
tiny shape, so the decision is policy arithmetic and the assertions are on
the request the engine echoed back, which is exactly the value the fold
destroyed.

WHAT IT DOES NOT PROVE. It does not prove any fit changes backend. The one
installed crossover rule is scoped to `SQUARED_ERROR` and `max_outputs = 1`,
so it declines a softmax workload on two counts either way; declaring the
objective honestly changes which reasons the report gives, not which backend
runs.
"""

from std.python import Python, PythonObject
from std.testing import (
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)

from basic_bindings import decide_device_workload

from mojotrees.apple_gpu_policy import GpuProfile
from mojotrees.device_policy import (
    _normalized_bins,
    _normalized_objective,
    AUTO_MIN_CELLS,
    BINS_UNSPECIFIED,
    BLOCK_UNKNOWN_OBJECTIVE,
    CPU_DEVICE,
    DeviceCapabilities,
    DeviceRequest,
    GPU_DEVICE,
    OBJECTIVE_UNSPECIFIED,
    PROFILE_REPORTED,
    WARN_HOST_GRADIENT_PATH,
    decide_device,
)
from mojotrees.initialization import SessionState
from mojotrees.unified_memory_policy import SessionMemoryPlan
from mojotrees.objective_registry import (
    BINARY_LOGISTIC,
    LAMBDARANK,
    MULTICLASS,
    SQUARED_ERROR,
)


# The bin count the workloads below declare. Not under test; passed so the
# request is complete and the assertions are about the objective alone.
comptime _BINS = 255


# --- The native marshaller ---------------------------------------------


def test_normalized_objective_separates_the_two_negatives() raises:
    """`_normalized_objective` is the ONE normalizer, and this is its rule.

    Stated as three assertions rather than one because the boundary between
    "keep" and "fold" sits between two adjacent integers, and a test that
    only checked `-2` would pass against a marshaller that folded `-1` too,
    which is precisely the marshaller this file exists to keep out.
    """
    assert_equal(MULTICLASS, -1)
    assert_equal(OBJECTIVE_UNSPECIFIED, -2)
    # Kept, because it is a code.
    assert_equal(_normalized_objective(MULTICLASS), MULTICLASS)
    # Folded, because it is an absence.
    assert_equal(
        _normalized_objective(OBJECTIVE_UNSPECIFIED), OBJECTIVE_UNSPECIFIED
    )
    # Anything further down is the same absence, however it was spelled.
    assert_equal(_normalized_objective(-3), OBJECTIVE_UNSPECIFIED)
    assert_equal(_normalized_objective(-99), OBJECTIVE_UNSPECIFIED)
    # Real codes cross unchanged, including the two that are not
    # single-output built-ins.
    assert_equal(_normalized_objective(SQUARED_ERROR), SQUARED_ERROR)
    assert_equal(_normalized_objective(BINARY_LOGISTIC), BINARY_LOGISTIC)
    assert_equal(_normalized_objective(LAMBDARANK), LAMBDARANK)


def test_normalized_bins_folds_both_spellings_of_undeclared() raises:
    """The bin count's sentinel has no such near-miss: `BINS_UNSPECIFIED`
    is 0, so `-1` and `0` are both undeclared and neither is a bin count.
    Asserted here so that removing the duplicate fold from the binding is
    covered on this argument too."""
    assert_equal(BINS_UNSPECIFIED, 0)
    assert_equal(_normalized_bins(-1), BINS_UNSPECIFIED)
    assert_equal(_normalized_bins(0), BINS_UNSPECIFIED)
    assert_equal(_normalized_bins(255), 255)


# --- The engine, on a declared softmax objective -----------------------


def _request(
    objective: Int, n_outputs: Int, device: Int = CPU_DEVICE
) raises -> DeviceRequest:
    """A complete request at a shape too small to interest any rule.

    `device` defaults to `cpu` because the blocking reasons are collected
    whatever was requested ("why would the GPU not have worked here" is a
    question the report answers unasked). The WARNINGS are not:
    `_collect_warnings` returns early on `requested_device == CPU_DEVICE`,
    so a test about a warning has to ask for the GPU or it asserts the
    absence of something that was never going to be there. That mistake was
    made in the first draft of this file and caught by running the test
    against the unfixed engine, where it passed.
    """
    return DeviceRequest(device, 1000, 10, n_outputs, _BINS, objective)


def _working_accelerator() raises -> DeviceCapabilities:
    """A machine with a working accelerator, constructed rather than
    detected.

    `_collect_blocks` short-circuits on `not caps.gpu_available` and
    returns before the objective gate is ever reached, and
    `DeviceCapabilities.detect()` reports False for that on every runner
    without a GPU. Detecting here would therefore make the three tests
    below pass vacuously on exactly the CPU-only half of CI that is
    supposed to catch this, which is the failure mode
    tests/test_device_auto_crossover.mojo's header warns about. This is the
    same fixture `_caps_for` in tests/test_device.mojo builds, and for the
    same reason: `gpu_available` is True whatever this build was compiled
    on, so the gate is reached identically everywhere.

    The profile is the portable generic one. No assertion below depends on
    a hardware number: the memory gate stays silent because a generic
    profile reports no budget, and the one installed crossover rule is
    scoped to an Apple M4 this profile is not.
    """
    return DeviceCapabilities(
        True,
        True,
        False,
        GpuProfile.generic(),
        PROFILE_REPORTED,
        AUTO_MIN_CELLS,
        SessionState.cold(),
        SessionMemoryPlan.staged(),
    )


def _has(codes: List[Int], wanted: Int) -> Bool:
    for i in range(len(codes)):
        if codes[i] == wanted:
            return True
    return False


def test_declared_multiclass_is_not_refused_as_an_unknown_objective() raises:
    """THE BUG THE TWO TRAINER CALL SITES WORKED AROUND, AS AN ASSERTION.

    `gpu_trains_objective(MULTICLASS)` is False and stays False: it asks
    whether `train_gpu` itself accepts the code, and it does not. The
    objective gate asks a different question -- whether any backend covers
    the run -- and `train_multiclass_gpu` covers it, reached through the
    `device` setting rather than around it. Until the gate was taught the
    difference, declaring `-1` produced BLOCK_UNKNOWN_OBJECTIVE, which is
    what `model.fit_multiclass` and `trainset.train_dataset_multiclass`
    avoided by declaring nothing instead.
    """
    var decision = decide_device(
        _request(MULTICLASS, 4), _working_accelerator()
    )
    assert_false(
        _has(decision.blocking_reasons.codes, BLOCK_UNKNOWN_OBJECTIVE)
    )
    # And the request kept what the caller said, rather than the engine
    # quietly deciding the caller had said nothing.
    assert_equal(decision.request.objective, MULTICLASS)
    assert_true(decision.request.objective_known())


def test_declared_multiclass_does_not_claim_host_gradients() raises:
    """`objective_gradients_on_device(MULTICLASS)` is False because it is
    about the single-output path's `fill_gradients_device`. Softmax
    derivatives have a device kernel of their own inside
    `train_multiclass_gpu`, so the host-gradient warning would be a false
    statement printed on every declared softmax run.

    Requested as `gpu`, because `_collect_warnings` returns before the
    gradient warning for a `cpu` request and this assertion would otherwise
    hold vacuously.
    """
    var decision = decide_device(
        _request(MULTICLASS, 4, GPU_DEVICE), _working_accelerator()
    )
    # Guard the guard: the warning list has to have been populated at all,
    # or the absence below says nothing.
    assert_true(decision.warnings.count() > 0)
    assert_false(_has(decision.warnings.codes, WARN_HOST_GRADIENT_PATH))


def test_an_objective_no_trainer_implements_is_still_refused() raises:
    """The counterweight. Loosening the gate for `-1` must not loosen it
    for a code that really is unimplemented, or the test above would be
    satisfied by deleting the gate."""
    var decision = decide_device(_request(9999, 1), _working_accelerator())
    assert_true(
        _has(decision.blocking_reasons.codes, BLOCK_UNKNOWN_OBJECTIVE)
    )


# --- The Python boundary ------------------------------------------------


def _workload(objective: Int, n_outputs: Int) raises -> PythonObject:
    """The mapping `_FullNativePolicy.decide` sends, as a real CPython dict.

    Built rather than modelled: the point of this test is what the binding
    does to a value on its way through, so the value has to actually go
    through it. And built BY THE SENDER, through
    `device_selection.workload_mapping`, which is now the one place the key
    set is written down.

    That call is the fix for a defect this fixture had four times. Every key
    is read on the native side with no default, deliberately, so that a stale
    sender cannot silently mean "L2, plain boosting, no split noise, float32".
    The cost of that choice is that a fixture missing a key dies at the CALL,
    before any assertion runs. It happened for two hours when
    `ordered_boosting` and `score_function` became required, for 646 commits
    when `random_strength` and `derivative_precision` did on a branch with no
    continuous integration, and twice on 2026-08-18 when five more keys
    arrived in one day. Each time three tests failed together, every assertion
    in them true the whole time.

    A default `Workload` supplies neutral values, which is a requirement
    rather than a convenience. `_collect_blocks` gates the accelerator on
    `derivative_precision != DERIV_PRECISION_FLOAT32` and on `random_strength`,
    so a fixture that sent something interesting would add a blocking reason
    to every workload here and these tests would be asserting about a run the
    policy had already refused for an unrelated reason. The fixture's job is
    to be complete and boring so the assertions are about the objective alone.
    """
    # BUILT BY THE SENDER, NOT BY HAND. `workload_mapping` in
    # device_selection.py is the one place the key set is written down, and
    # `_FullNativePolicy.decide` builds its mapping with the same call. A
    # field added to the request therefore appears here automatically.
    #
    # This fixture drifted FOUR times before that function existed, three of
    # them to these same three tests and one of them for 646 commits, because
    # the sender built its mapping as a dict literal inline and there was
    # nothing for this file to import. Five keys were added on 2026-08-18
    # alone. The native reader takes every key with no default on purpose, so
    # each drift surfaced as a KeyError hours later rather than as a wrong
    # decision, which is the good failure mode and still cost the time.
    #
    # A default `Workload` carries exactly the neutral values this fixture
    # wants, which is the second reason to build it this way: the fixture must
    # not send anything interesting, or it would add a blocking reason to
    # every workload here and the assertions would be about a run the policy
    # had already refused for an unrelated cause.
    # Loaded as a FILE rather than as `mojotrees.device_selection`, because
    # importing the package would execute `mojotrees/__init__.py` and pull in
    # the compiled extension, which this test does not need and which is not
    # on the interpreter's path when the Mojo suite runs. `device_selection.py`
    # imports only `json` at module scope, so it loads standalone.
    var os = Python.import_module("os")
    var util = Python.import_module("importlib.util")
    var rel = os.path.join("python", "mojotrees", "device_selection.py")
    var root = os.path.abspath(os.getcwd())
    var target = os.path.join(root, rel)
    while not Bool(os.path.exists(target)):
        var parent = os.path.dirname(root)
        if String(parent) == String(root):
            raise Error(
                "could not find ",
                rel,
                " walking up from ",
                String(os.getcwd()),
            )
        root = parent
        target = os.path.join(root, rel)
    var spec = util.spec_from_file_location("mt_device_selection", target)
    var selection = util.module_from_spec(spec)
    spec.loader.exec_module(selection)
    var w = selection.workload_mapping(
        selection.Workload(n_rows=1000, n_features=10)
    )
    w["n_outputs"] = PythonObject(n_outputs)
    w["objective"] = PythonObject(objective)
    w["n_bins"] = PythonObject(_BINS)
    return w^


def _decide(objective: Int, n_outputs: Int) raises -> String:
    """The binding, called on the fixture, with the stale-fixture failure
    named instead of arriving as a CPython `KeyError`.

    Every key `decide_device_workload` reads is read with no default, on
    purpose, so a mapping that has gone stale against the binding fails at the
    CALL rather than at an assertion. That is the right failure, and what it
    never did was say what went wrong: three tests die together, the message
    is a bare key name from CPython, and nothing points at the fixture as the
    thing that is behind rather than at the decision as the thing that is
    wrong.

    So the exception is caught only to be re-raised with that sentence
    attached. Nothing is swallowed. The original message is carried through,
    every test still fails, and they fail for the same reason they did before.
    This is the second line of defence now that `_workload` builds its mapping
    from the sender: it should no longer be reachable by drift, and it stays
    because a mapping can still go stale in ways that call does not cover, for
    instance a key the native reader adds and the sender has not.

    `"cpu"` for all three callers, because the boundary tests are about what
    the marshaller does to the objective on the way through, and asking for a
    device the runner may not have would put a capability detection in the
    middle of that.
    """
    try:
        return String(
            py=decide_device_workload(
                PythonObject("cpu"), _workload(objective, n_outputs)
            )
        )
    except e:
        raise Error(
            "decide_device_workload did not accept the fixture mapping: ",
            String(e),
            ". Every key it reads is required and read with no default, so a"
            " KeyError here means `_workload` has gone stale against"
            " bindings/basic_bindings.mojo rather than that the decision is"
            " wrong. `_workload` sends n_rows, n_features, n_outputs, n_bins,"
            " objective, sparse, categorical, has_missing, uses_validation,"
            " ordered_boosting, score_function, random_strength,"
            " derivative_precision; compare that against the mapping"
            " _FullNativePolicy.decide sends in"
            " python/mojotrees/device_selection.py, which is the only real"
            " sender and is the one that is right",
        )


def _field(report: String, key: String) raises -> String:
    """One `key=value` line's value out of a serialized decision."""
    var needle = String("\n", key, "=")
    var padded = String("\n", report)
    var start = padded.find(needle)
    if start < 0:
        raise Error(String("no ", key, "= line in the decision"))
    var from_value = start + needle.byte_length()
    var end = padded.find(String("\n"), from_value)
    if end < 0:
        end = padded.byte_length()
    return String(padded[byte=from_value:end])


def test_python_boundary_preserves_the_multiclass_marker() raises:
    """THE DEFECT THIS FILE IS NAMED FOR.

    `decide_device_workload` folded `objective < 0` before calling
    `decide_device_report`, so `-1` and `-2` reached the engine as the same
    value. A Python caller had no way to say "softmax":
    `Workload(objective="multiclass")` resolves through the registry to
    `-1` and lost it here.

    The assertion is on the objective the decision echoes back, because
    that is the request the engine actually gated on.
    """
    var report = _decide(MULTICLASS, 4)
    assert_equal(_field(report, String("objective")), String("-1"))
    assert_equal(_field(report, String("objective_known")), String("true"))


def test_python_boundary_keeps_undeclared_distinguishable() raises:
    """The other half of the same claim: fixing `-1` must not make `-2`
    look like a declaration. A test that only checked `-1` would pass
    against a binding that had simply stopped normalizing anything, which
    would hand `-3` and `-99` to the engine as objective codes."""
    var undeclared = _decide(OBJECTIVE_UNSPECIFIED, 1)
    assert_equal(_field(undeclared, String("objective")), String("-2"))
    assert_equal(
        _field(undeclared, String("objective_known")), String("false")
    )

    # Any other spelling of "I did not say" normalizes to the same one.
    var far = _decide(-99, 1)
    assert_equal(_field(far, String("objective")), String("-2"))
    assert_equal(_field(far, String("objective_known")), String("false"))


def test_python_boundary_carries_real_codes_unchanged() raises:
    """A nonnegative code is not a sentinel and nothing may touch it."""
    var report = _decide(BINARY_LOGISTIC, 1)
    assert_equal(_field(report, String("objective")), String("1"))
    assert_equal(_field(report, String("objective_known")), String("true"))

    # The four keys the fixture has at some point stopped sending, asserted on
    # the way back out. `_workload` had gone stale against a required key and
    # the failure that produced was a CPython KeyError at the call, in three
    # tests at once, with no assertion reached and nothing naming the key.
    # These lines are what turns the next such drift into a named, single
    # failure. They read the echo of exactly the fields those keys feed, so a
    # mapping that silently stopped carrying them fails here and says which.
    assert_equal(_field(report, String("ordered_boosting")), String("false"))
    assert_equal(_field(report, String("score_function")), String("0"))
    # `random_strength` and `derivative_precision` joined the required set in
    # `73d808a` and `9d51e29` and the fixture did not follow, which is what
    # took these three tests down when the branch was promoted. Asserted the
    # same way and for the same reason as the two above them, and NOT merely
    # by sending the keys. A binding that accepted them and dropped them on
    # the floor would satisfy the call and fail right here.
    #
    # `random_strength` is compared against `String(Float64(0.0))` rather than
    # against a spelled-out "0.0", because the serializer writes the field by
    # handing the Float64 to the same String conversion. Pinning the digits
    # would be pinning Mojo's float formatting, which is not what this file is
    # about and would go red on a change that moved no value.
    assert_equal(
        _field(report, String("random_strength")), String(Float64(0.0))
    )
    assert_equal(_field(report, String("derivative_precision")), String("0"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
