"""`device='auto'` reaching the GPU: the three things that stopped it, pinned.

WHAT THIS FILE IS FOR. Between the crossover rule being installed and
2026-08-16, `auto` selected the CPU at every shape on every machine,
including the Apple M4 the rule was measured on. Three independent defects
did it, each sufficient alone, and a fix for any one of them would have
looked correct in isolation and changed nothing. Each has a test below that
fails if it comes back.

NAMING, AND WHY THIS IS NOT `test_gpu_*`. `tools/run_tests.sh` derives
GPU-only status from the file name. This file opens no device: every
assertion is either pure policy arithmetic or a read of a comptime build
property, so it belongs in the CPU set and runs on the x86-64 half of CI,
which is where a detection bug shows up as the *other* answer rather than
not at all. Nothing here trains, and nothing here times anything.

HOW IT AVOIDS BEING VACUOUS, WHICH IS THIS PROJECT'S RECURRING TEST FAILURE.
This repository has shipped a test whose six fixtures all ran below the gate
they were testing, and `bench/results/session3_2026-08-16/RESULTS.md` records
the same error twice more in other forms. So:

  - `test_detection_names_the_hardware_when_there_is_any` asserts
    unconditionally, on any machine with an accelerator, that
    `DeviceCapabilities.detect()` does NOT come back `PROFILE_FALLBACK`. That
    is the exact state the bug consisted of. It cannot pass vacuously: on
    this machine it either identifies the device or it fails.
  - the crossover assertions run against `DeviceCapabilities.detect()`, not
    against a constructed fixture. `apple_m4_observed()` builds its profile
    with the fieldwise constructor and hands `APPLE_GEN_M4` in directly, so
    tests written against it were green for two days while no detection path
    in the repository could produce that value. Fixtures are used here only
    for the branches a real machine cannot supply (another vendor, a
    disabled accelerator).
  - `test_reported_path_parses_a_real_arch_string` feeds the exact string a
    Metal device returns, read from one on 2026-08-16, rather than a
    human-readable spelling nobody's driver emits.

WHAT IT DOES NOT PROVE. It does not prove that a 1,000,000 x 50 fit runs on
the accelerator, because that is a training run and this lane was not
permitted one. It proves that `resolve_device`, the function every trainer
calls, returns `GPU_DEVICE` for that workload on this machine, and that the
return is the crossover rule firing rather than a size heuristic or an
explicit request: the decision code is `DECISION_AUTO_GPU_EVIDENCE`,
`validated()` is True, and the evidence identifier is the rule's own.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_true,
    TestSuite,
)

from mojotrees.apple_gpu_policy import (
    API_CUDA,
    API_METAL,
    API_UNKNOWN,
    APPLE_GEN_M4,
    APPLE_GEN_UNKNOWN,
    GpuProfile,
    OBSERVED_M4_ARCH_NAME,
    OBSERVED_M4_CORE_COUNT,
    OBSERVED_M4_MAX_THREADS_PER_BLOCK,
    OBSERVED_M4_SHARED_BYTES,
    parse_apple_generation,
)
from mojotrees.device import (
    AUTO_DEVICE,
    CPU_DEVICE,
    GPU_DEVICE,
    resolve_device,
)
from mojotrees.device_policy import (
    AUTO_GPU_MIN_ROWS,
    BLOCK_NO_ACCELERATOR,
    DECISION_AUTO_CPU_BELOW_EVIDENCE,
    DECISION_AUTO_CPU_BLOCKED,
    DECISION_AUTO_GPU_EVIDENCE,
    DeviceCapabilities,
    DeviceRequest,
    M4_MULTICLASS_EVIDENCE_ID,
    M4_MULTICLASS_MIN_FEATURES,
    M4_TRAINING_EVIDENCE_ID,
    M4_TRAINING_MIN_FEATURES,
    OBJECTIVE_UNSPECIFIED,
    POLICY_VERSION,
    PROFILE_BUILD_TARGET,
    PROFILE_DECLARED,
    PROFILE_FALLBACK,
    PROFILE_NONE,
    PROFILE_REPORTED,
    WARN_BUILD_TARGET_HARDWARE,
    build_accelerator_target,
    build_target_profile,
    capabilities_from_reported,
    decide_device,
    profile_source_name,
)
from mojotrees.objective_registry import (
    BINARY_LOGISTIC,
    MULTICLASS,
    SQUARED_ERROR,
)

comptime _DISABLE_GPU = "MOJOTREES_DISABLE_GPU"
comptime _AUTO_MIN_CELLS = "MOJOTREES_AUTO_MIN_CELLS"
comptime _GPU_BACKEND = "MOJOTREES_GPU_BACKEND"

# The bin count both records behind `M4_TRAINING_EVIDENCE_ID` were taken at,
# and the one this file describes its workloads with. It is not in the rule's
# scope, because it was never swept; it is passed so that the memory estimate
# is complete and the request carries no `WARN_INCOMPLETE_REQUEST`.
comptime _MEASURED_BINS = 255

# The build target this project develops on, MEASURED 2026-08-16 by calling
# `std.sys.info._accelerator_arch()` on this machine. Every assertion scoped
# to it is skipped, passing, on any other build, and each such test says so.
comptime _OBSERVED_M4_TARGET = String("metal:4-metal4")


def _set_env(name: String, value: String):
    """Set an override; the empty string restores the default."""
    _ = setenv(name, value, True)


def _clear_env():
    _set_env(_DISABLE_GPU, "")
    _set_env(_AUTO_MIN_CELLS, "")
    _set_env(_GPU_BACKEND, "")


def _auto(n_rows: Int, n_features: Int) raises -> DeviceRequest:
    """A fully described `auto` request at the objective and bin count the
    crossover records were taken at. Fully described on purpose: an
    incomplete one cannot reach an objective-scoped rule and would make the
    shape assertions below say nothing about the shape."""
    return DeviceRequest(
        AUTO_DEVICE,
        n_rows,
        n_features,
        1,
        _MEASURED_BINS,
        SQUARED_ERROR,
    )


# The row count the multiclass record was taken at
# (`bench/results/profile_2026-08-15/RESULTS.md`, 465,000 x 54 over 7
# classes). Used as a shape well clear of both rules' floors, never as a
# threshold: the multiclass rule's floor is `AUTO_GPU_MIN_ROWS` and the test
# below that asserts the floor uses that constant.
comptime _MULTICLASS_MEASURED_ROWS = 465_000


def _auto_multiclass(
    n_rows: Int, n_classes: Int, objective: Int
) raises -> DeviceRequest:
    """A fully described `auto` softmax request at the multiclass record's
    feature count and the measured bin count.

    `objective` is a parameter rather than a constant because whether the
    caller declared one is the thing under test: the three Mojo multiclass
    entry points send `OBJECTIVE_UNSPECIFIED` and Python sends `MULTICLASS`,
    and the rule is scoped so that both reach it.
    """
    return DeviceRequest(
        AUTO_DEVICE,
        n_rows,
        M4_MULTICLASS_MIN_FEATURES,
        n_classes,
        _MEASURED_BINS,
        objective,
    )


def _on_the_measured_build() -> Bool:
    """Whether this binary targets the accelerator the rule was measured on.

    Not "whether this machine is an M4": nothing here can know that, which is
    the standing caveat on `PROFILE_BUILD_TARGET` and is why the decision it
    produces carries `WARN_BUILD_TARGET_HARDWARE`.
    """
    return build_accelerator_target() == _OBSERVED_M4_TARGET


# --- Defect 1: detection could not name the hardware ------------------


def test_build_target_is_present_exactly_when_an_accelerator_is() raises:
    """The comptime probe, and the CPU-only build's answer to it.

    `build_accelerator_target` is the one call into `_accelerator_arch`, and
    it is wrapped in `comptime if has_accelerator()` with an `else` covering
    the whole body. That is not decoration: a build with no GPU architecture
    fails at compile time rather than skipping at run time, and an Apple
    machine never reproduces it. This test is the run-time half of the same
    claim, and on the x86-64 half of CI it is the half that actually runs.
    """
    var target = build_accelerator_target()
    comptime if has_accelerator():
        assert_true(target.byte_length() > 0)
    else:
        assert_equal(target, String(""))


def test_detection_names_the_hardware_when_there_is_any() raises:
    """THE BUG, STATED AS AN ASSERTION.

    `detect()` used to return `GpuProfile.generic()` unconditionally, whose
    api is `API_UNKNOWN`. `CrossoverEvidence.matches` tests the api before
    anything else, so no hardware-scoped rule could fire for any caller that
    went through `resolve_device`, which is all six trainer entry points.

    This cannot pass vacuously on a machine with an accelerator: either the
    profile source is better than `PROFILE_FALLBACK` or this fails.
    """
    _clear_env()
    var caps = DeviceCapabilities.detect()
    if not caps.gpu_available:
        assert_equal(caps.profile_source, PROFILE_NONE)
        return
    assert_not_equal(caps.profile_source, PROFILE_FALLBACK)
    assert_equal(caps.profile_source, PROFILE_BUILD_TARGET)
    assert_not_equal(caps.profile.api, API_UNKNOWN)
    # Identity only. Every capability number is still the portable fallback,
    # which is what keeps the memory gate silent: `_collect_blocks` requires
    # `memory_budget_known()` and a build target reports no budget.
    assert_false(caps.memory_budget_known())
    assert_true(caps.profile.synthetic)


def test_build_target_profile_carries_identity_and_no_numbers() raises:
    var profile = build_target_profile()
    var generic = GpuProfile.generic()
    comptime if not has_accelerator():
        assert_equal(profile.api, API_UNKNOWN)
        assert_equal(profile.apple_generation, APPLE_GEN_UNKNOWN)
        return
    else:
        assert_equal(profile.core_count, generic.core_count)
        assert_equal(
            profile.max_threads_per_block, generic.max_threads_per_block
        )
        assert_equal(
            profile.max_shared_memory_per_block,
            generic.max_shared_memory_per_block,
        )
        assert_equal(profile.memory_budget_bytes, 0)
        if _on_the_measured_build():
            assert_equal(profile.api, API_METAL)
            assert_equal(profile.apple_generation, APPLE_GEN_M4)
            assert_true(profile.unified_memory)


def test_a_contradicting_declaration_removes_identity_rather_than_adding_one(
) raises:
    """`MOJOTREES_GPU_BACKEND` may take identity away and may never install
    one.

    An operator naming an api that disagrees with the build target is
    asserting that this binary is running somewhere other than where it was
    built, which is the redistributed-wheel case. The honest response is to
    drop the build target's generation, not to keep it and not to believe the
    operator's word about the generation, which they did not give.
    """
    comptime if not has_accelerator():
        return
    else:
        if not _on_the_measured_build():
            return
        _clear_env()
        # Agreement adds nothing and does not downgrade the source.
        _set_env(_GPU_BACKEND, "metal")
        var agreeing = DeviceCapabilities.detect()
        assert_equal(agreeing.profile_source, PROFILE_BUILD_TARGET)
        assert_equal(agreeing.profile.apple_generation, APPLE_GEN_M4)

        _set_env(_GPU_BACKEND, "cuda")
        var conflicting = DeviceCapabilities.detect()
        assert_equal(conflicting.profile_source, PROFILE_DECLARED)
        assert_equal(conflicting.profile.api, API_CUDA)
        assert_equal(
            conflicting.profile.apple_generation, APPLE_GEN_UNKNOWN
        )
        # And with the generation gone, the measured shape falls back to the
        # CPU: hardware cannot be misdeclared into a crossover rule.
        var decision = decide_device(
            _auto(AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES),
            conflicting,
        )
        assert_equal(decision.selected_device, CPU_DEVICE)
        assert_false(decision.validated())
        _clear_env()


# --- Defect 2: the reported arch string did not parse ------------------


def test_reported_path_parses_a_real_arch_string() raises:
    """The string a Metal device actually returns, not a spelling of it.

    MEASURED 2026-08-16: `DeviceContext().arch_name()` on the development M4
    returns `4-metal4`, which contains no "m4" substring because "metal4"
    ends in "l4". `parse_apple_generation` matched only human-readable forms,
    so `GpuProfile.from_reported` turned a genuine M4 reading into
    `APPLE_GEN_UNKNOWN` and the generation scope declined the rule a second
    time, for `capabilities_from_reported` and
    `decide_device_report_reported` as well: the two entry points that exist
    for a caller holding an open `DeviceContext`.
    """
    assert_equal(parse_apple_generation(OBSERVED_M4_ARCH_NAME), APPLE_GEN_M4)
    assert_equal(parse_apple_generation(String("4-metal4")), APPLE_GEN_M4)
    # The human-readable forms still work, and an unrelated architecture is
    # still unknown rather than guessed at.
    assert_equal(parse_apple_generation(String("Apple M4")), APPLE_GEN_M4)
    assert_equal(parse_apple_generation(String("sm_90a")), APPLE_GEN_UNKNOWN)
    # No pattern was installed: a generation this project has never read is
    # still unknown, deliberately, because it is not known whether the
    # leading digit is the chip or the Metal feature set.
    assert_equal(parse_apple_generation(String("3-metal3")), APPLE_GEN_UNKNOWN)

    var profile = GpuProfile.from_reported(
        String("metal"),
        OBSERVED_M4_ARCH_NAME,
        OBSERVED_M4_CORE_COUNT,
        OBSERVED_M4_MAX_THREADS_PER_BLOCK,
        OBSERVED_M4_SHARED_BYTES,
    )
    assert_equal(profile.api, API_METAL)
    assert_equal(profile.apple_generation, APPLE_GEN_M4)


def test_an_open_context_still_outranks_the_build_target() raises:
    """`PROFILE_REPORTED` is unchanged and still authoritative, and now it
    reaches the rule from the strings a real device hands over."""
    var caps = capabilities_from_reported(
        String("metal"),
        OBSERVED_M4_ARCH_NAME,
        OBSERVED_M4_CORE_COUNT,
        OBSERVED_M4_MAX_THREADS_PER_BLOCK,
        OBSERVED_M4_SHARED_BYTES,
    )
    assert_equal(caps.profile_source, PROFILE_REPORTED)
    assert_equal(caps.profile.apple_generation, APPLE_GEN_M4)
    var decision = decide_device(
        _auto(AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES), caps
    )
    assert_equal(decision.selected_device, GPU_DEVICE)
    assert_true(decision.validated())
    assert_equal(decision.evidence_id, M4_TRAINING_EVIDENCE_ID)
    # A reported profile is a reading, so it gets no build-target warning.
    var warned = False
    for i in range(decision.warnings.count()):
        if decision.warnings.codes[i] == WARN_BUILD_TARGET_HARDWARE:
            warned = True
    assert_false(warned)


# --- The gate, opened, and the shapes on either side of it -------------


def test_auto_reaches_the_gpu_at_the_measured_shape() raises:
    """THE GATE, PROVED OPEN, through detected capabilities.

    Not a fixture: `DeviceCapabilities.detect()` is what `resolve_device`
    itself calls, so this passes only if detection named the hardware, the
    architecture string parsed, and the rule matched. The three assertions
    after the device code are what separate "the rule fired" from "something
    returned a value meaning GPU": a size heuristic would report
    `DECISION_AUTO_GPU_ENV_THRESHOLD` and `validated()` False, and an
    explicit request would report `EVIDENCE_EXPLICIT`.

    Skipped, passing, on any build that does not target the measured
    accelerator, which is the whole of CPU-only CI and every non-Apple
    runner. `test_detection_names_the_hardware_when_there_is_any` is the
    assertion that runs everywhere and cannot be skipped into vacuity.
    """
    _clear_env()
    if not _on_the_measured_build():
        return
    var caps = DeviceCapabilities.detect()
    var decision = decide_device(
        _auto(AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES), caps
    )
    assert_equal(decision.selected_device, GPU_DEVICE)
    assert_equal(decision.decision_code, DECISION_AUTO_GPU_EVIDENCE)
    assert_true(decision.validated())
    assert_equal(decision.evidence_id, M4_TRAINING_EVIDENCE_ID)
    # The citation is what makes the claim checkable rather than merely
    # asserted, and `CrossoverEvidence` refuses to construct a rule without
    # one.
    assert_true(decision.crossover_citation.find("Apple M4") >= 0)
    # Selecting a backend on a build property is a stronger claim than
    # reporting availability on one, so it says so.
    var warned = False
    for i in range(decision.warnings.count()):
        if decision.warnings.codes[i] == WARN_BUILD_TARGET_HARDWARE:
            warned = True
    assert_true(warned)


def test_auto_keeps_the_cpu_below_the_row_floor() raises:
    """The other half of the gate: one row under the floor is the CPU.

    The floor is `AUTO_GPU_MIN_ROWS`, a provisional constant at 250,000 rows
    rather than a measured crossover, so this test asserts an *edge* rather
    than a claim about performance. `AUTO_GPU_MIN_ROWS - 1` is the assertion
    that carries the weight: it fails the moment the floor stops being a
    hard comparison against one number, which is what a "roughly" or a
    fitted expression growing in its place would look like.

    50,000 x 50 is the shape where staying on the CPU is also the measured
    answer: `bench/results/profile_2026-08-15/RESULTS.md`, arms interleaved,
    median of three, has the CPU at 0.564 s against the GPU's 1.63 s, a 2.9x
    loss. That is the near end of the interval the floor is deliberately
    kept above.

    `bench/results/session3_2026-08-16/RESULTS.md` has larger and more recent
    figures at 50,000 and 250,000 and they do not bear on this: both arms of
    every one of those pairs had the device split search forced, so they
    compare a GPU path against a GPU path, and that file's own same-night
    correction withdraws the inference that was drawn from them.
    """
    _clear_env()
    if not _on_the_measured_build():
        return
    var caps = DeviceCapabilities.detect()
    var shapes: List[Int] = [50_000, 100_000, AUTO_GPU_MIN_ROWS - 1]
    for i in range(len(shapes)):
        var decision = decide_device(
            _auto(shapes[i], M4_TRAINING_MIN_FEATURES), caps
        )
        assert_equal(decision.selected_device, CPU_DEVICE)
        assert_equal(
            decision.decision_code, DECISION_AUTO_CPU_BELOW_EVIDENCE
        )
        assert_false(decision.validated())
    # And the feature floor, at a shape with far more than enough rows: a
    # 5,000,000 x 10 matrix clears the row floor twenty times over with a
    # fifth of the measured feature count, which is a different ratio of
    # per-node launch cost to per-node work and which the record says
    # nothing about. The row floor moved; the feature scope did not.
    var narrow = decide_device(_auto(5_000_000, 10), caps)
    assert_equal(narrow.selected_device, CPU_DEVICE)
    assert_false(narrow.validated())


def test_resolve_device_reaches_the_gpu_only_with_an_objective() raises:
    """Defect 3, pinned so that nobody closes it by weakening a rule's scope.

    `resolve_device` is what all six Mojo trainer entry points call and it
    declares no objective unless one is passed. Every crossover rule is
    scoped to the objective it was measured on, and an undeclared objective
    must not inherit a squared-error measurement. So the four-argument call
    keeps the CPU at every shape, correctly, and the fix is at the call
    sites, which are holding the objective when they drop it.
    """
    _clear_env()
    if not _on_the_measured_build():
        return
    assert_equal(
        resolve_device(
            AUTO_DEVICE, AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES, 1
        ),
        CPU_DEVICE,
    )
    assert_equal(
        resolve_device(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS,
            M4_TRAINING_MIN_FEATURES,
            1,
            OBJECTIVE_UNSPECIFIED,
        ),
        CPU_DEVICE,
    )
    # The same call, with the objective the caller already knows.
    assert_equal(
        resolve_device(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS,
            M4_TRAINING_MIN_FEATURES,
            1,
            SQUARED_ERROR,
        ),
        GPU_DEVICE,
    )
    # Below the floor it is still the CPU with the objective declared, so
    # what the objective bought is reachability and not a bypass.
    assert_equal(
        resolve_device(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS - 1,
            M4_TRAINING_MIN_FEATURES,
            1,
            SQUARED_ERROR,
        ),
        CPU_DEVICE,
    )


# --- The multiclass rule ------------------------------------------------
#
# `tests/test_gpu_auto_reaches_multiclass.mojo` is the other half of this and
# the half that costs a training run: it proves an `auto` softmax fit produces
# the GPU trainer's model bits and not the CPU trainer's. Everything here is
# policy arithmetic and opens no device, which is why it lives in the CPU set
# beside the single-output assertions.


def test_auto_reaches_the_gpu_for_multiclass_at_the_measured_shape() raises:
    """The multiclass gate, proved open at the shape the record was taken at.

    Until 2026-08-16 the table held one rule, scoped `max_outputs=1`, so a
    softmax fit fell through to `DECISION_AUTO_CPU_BELOW_EVIDENCE` at every
    size on every machine. Measured, that meant `auto` handing multiclass
    users the arm that loses: 465,000 x 54 over 7 classes is CPU 25.47 s
    against GPU 15.30 s on this hardware.

    Same construction as the single-output test above and for the same
    reason: `DeviceCapabilities.detect()`, not a fixture, and three
    assertions past the device code so that "a rule fired" is separated from
    "something returned a value meaning GPU".
    """
    _clear_env()
    if not _on_the_measured_build():
        return
    var caps = DeviceCapabilities.detect()
    var decision = decide_device(
        _auto_multiclass(_MULTICLASS_MEASURED_ROWS, 7, MULTICLASS), caps
    )
    assert_equal(decision.selected_device, GPU_DEVICE)
    assert_equal(decision.decision_code, DECISION_AUTO_GPU_EVIDENCE)
    assert_true(decision.validated())
    assert_equal(decision.evidence_id, M4_MULTICLASS_EVIDENCE_ID)
    assert_true(decision.crossover_citation.find("7 classes") >= 0)
    # And it is not the single-output rule that fired: the two rules are
    # disjoint and a report that named the wrong one would be citing a
    # squared-error measurement for a softmax fit.
    assert_not_equal(decision.evidence_id, M4_TRAINING_EVIDENCE_ID)


def test_multiclass_reaches_the_rule_with_or_without_a_declared_objective(
) raises:
    """The scope choice, pinned: trees-per-round is the multiclass scope and
    the objective code is not.

    This is what makes the rule reachable at all. `model.fit_multiclass`,
    `trainset.train_dataset_multiclass`, and
    `external_memory.train_external_multiclass` each call `resolve_device`
    with `n_classes` as `n_outputs` and no objective, so a rule scoped
    `objective == MULTICLASS` would have been unreachable from every Mojo
    entry point while looking correct in the table. Python's
    `binding_params` does declare `MULTICLASS`. Both spellings must reach the
    same rule, and this asserts they do.

    It is also the test that fails if somebody "tightens" the rule by adding
    an objective scope to it.
    """
    _clear_env()
    if not _on_the_measured_build():
        return
    var caps = DeviceCapabilities.detect()
    var declared = decide_device(
        _auto_multiclass(_MULTICLASS_MEASURED_ROWS, 3, MULTICLASS), caps
    )
    var undeclared = decide_device(
        _auto_multiclass(
            _MULTICLASS_MEASURED_ROWS, 3, OBJECTIVE_UNSPECIFIED
        ),
        caps,
    )
    assert_equal(declared.selected_device, GPU_DEVICE)
    assert_equal(undeclared.selected_device, GPU_DEVICE)
    assert_equal(declared.evidence_id, M4_MULTICLASS_EVIDENCE_ID)
    assert_equal(undeclared.evidence_id, M4_MULTICLASS_EVIDENCE_ID)


def test_resolve_device_reaches_multiclass_without_an_objective() raises:
    """The same claim through the function the trainers actually call.

    `resolve_device` is the narrow entry point and the multiclass call sites
    pass it four arguments. This asserts the rule fires through that exact
    call, so the test cannot pass while the shipped path stays on the CPU --
    which is precisely the shape of the single-output defect this file's
    `test_resolve_device_reaches_the_gpu_only_with_an_objective` records.
    """
    _clear_env()
    if not _on_the_measured_build():
        return
    assert_equal(
        resolve_device(
            AUTO_DEVICE,
            _MULTICLASS_MEASURED_ROWS,
            M4_MULTICLASS_MIN_FEATURES,
            3,
        ),
        GPU_DEVICE,
    )
    # Below the floor it is still the CPU, so what the class count bought is
    # a rule and not a bypass of the row floor.
    assert_equal(
        resolve_device(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS - 1,
            M4_MULTICLASS_MIN_FEATURES,
            3,
        ),
        CPU_DEVICE,
    )


def test_multiclass_keeps_the_cpu_below_its_floor_and_feature_scope() raises:
    """The other side of the multiclass gate, both bounds.

    The row floor is `AUTO_GPU_MIN_ROWS`, shared with the single-output rule
    and provisional for both; the feature scope is
    `M4_MULTICLASS_MIN_FEATURES`, which is 54 because that is what was
    measured and not 50 to match the other rule. One short of either and the
    run keeps the CPU.
    """
    _clear_env()
    if not _on_the_measured_build():
        return
    var caps = DeviceCapabilities.detect()
    var thin = decide_device(
        _auto_multiclass(AUTO_GPU_MIN_ROWS - 1, 3, MULTICLASS), caps
    )
    assert_equal(thin.selected_device, CPU_DEVICE)
    assert_equal(thin.decision_code, DECISION_AUTO_CPU_BELOW_EVIDENCE)
    assert_false(thin.validated())
    # Far more rows than the floor, one feature short of the scope. The
    # multiclass record was taken at 54 features and says nothing about 53.
    var narrow = DeviceRequest(
        AUTO_DEVICE,
        _MULTICLASS_MEASURED_ROWS,
        M4_MULTICLASS_MIN_FEATURES - 1,
        3,
        _MEASURED_BINS,
        MULTICLASS,
    )
    var narrow_decision = decide_device(narrow, caps)
    assert_equal(narrow_decision.selected_device, CPU_DEVICE)
    assert_false(narrow_decision.validated())


def test_the_two_rules_are_disjoint() raises:
    """Neither rule can be reached by a request the other one covers.

    This is what keeps a multiclass rule from moving a single-output fit's
    bits. `max_outputs=1` on the single-output rule and `min_outputs=2` on
    the multiclass one are complementary, so a request matches at most one,
    and a single-output request at a shape well past both floors must still
    cite the squared-error record.
    """
    _clear_env()
    if not _on_the_measured_build():
        return
    var caps = DeviceCapabilities.detect()

    # Single output, well past both row floors and both feature scopes.
    var single = decide_device(
        _auto(_MULTICLASS_MEASURED_ROWS, M4_MULTICLASS_MIN_FEATURES), caps
    )
    assert_equal(single.selected_device, GPU_DEVICE)
    assert_equal(single.evidence_id, M4_TRAINING_EVIDENCE_ID)
    assert_not_equal(single.evidence_id, M4_MULTICLASS_EVIDENCE_ID)

    # A single-output request declaring an objective the single-output rule
    # does not cover reaches neither rule, which is the check that the
    # multiclass rule's unconstrained objective field did not open a hole:
    # binary logistic at 1 output must stay on the CPU.
    var logistic = DeviceRequest(
        AUTO_DEVICE,
        _MULTICLASS_MEASURED_ROWS,
        M4_MULTICLASS_MIN_FEATURES,
        1,
        _MEASURED_BINS,
        BINARY_LOGISTIC,
    )
    var logistic_decision = decide_device(logistic, caps)
    assert_equal(logistic_decision.selected_device, CPU_DEVICE)
    assert_false(logistic_decision.validated())

    # Multiclass at a class count past the measured seven still matches: the
    # rule bounds the class count below and not above, deliberately, and
    # `crossover_rules()` says what measurement would close that.
    var many = decide_device(
        _auto_multiclass(_MULTICLASS_MEASURED_ROWS, 20, MULTICLASS), caps
    )
    assert_equal(many.selected_device, GPU_DEVICE)
    assert_equal(many.evidence_id, M4_MULTICLASS_EVIDENCE_ID)


# --- Degrading on a machine that has no accelerator --------------------


def test_no_accelerator_keeps_the_cpu_at_the_measured_shape() raises:
    """The CPU-only build, exercised on any machine.

    `DeviceCapabilities.unavailable()` is the injected form and reads no
    environment at all; `MOJOTREES_DISABLE_GPU=1` is the same state reached
    through `detect()` on a machine that does have one. Both must give the
    CPU at the shape where the rule otherwise fires, and the first blocking
    reason must be the absence of hardware rather than anything about the
    workload.
    """
    var absent = DeviceCapabilities.unavailable()
    assert_equal(absent.profile_source, PROFILE_NONE)
    var decision = decide_device(
        _auto(AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES), absent
    )
    assert_equal(decision.selected_device, CPU_DEVICE)
    assert_equal(decision.decision_code, DECISION_AUTO_CPU_BLOCKED)
    assert_false(decision.validated())
    assert_equal(decision.blocking_reasons.codes[0], BLOCK_NO_ACCELERATOR)

    _clear_env()
    _set_env(_DISABLE_GPU, "1")
    var pinned = DeviceCapabilities.detect()
    assert_false(pinned.gpu_available)
    assert_equal(pinned.profile_source, PROFILE_NONE)
    assert_equal(
        resolve_device(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS,
            M4_TRAINING_MIN_FEATURES,
            1,
            SQUARED_ERROR,
        ),
        CPU_DEVICE,
    )
    _clear_env()


def test_policy_version_records_that_reachability_changed() raises:
    """A version-2 report saying "no rule covered this" and a version-3 one
    saying it mean different things: under version 2 the table was non-empty
    and unreachable. The number is what lets a reader tell them apart, and
    `profile_source` is what tells them which detector answered.

    Version 4 is the second such distinction and is about the floor rather
    than about reachability: from 4 on, a GPU selection between 250,000 and
    1,000,000 rows rests on `AUTO_GPU_MIN_ROWS`, a provisional constant set
    below the measured evidence, and a report carrying the number says so.
    """
    assert_true(POLICY_VERSION >= 4)
    assert_equal(profile_source_name(PROFILE_BUILD_TARGET), "build-target")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
