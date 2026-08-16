"""Device selection: parsing, resolution policy, and trainer equivalence.

Resolution is exercised on any machine: `MOJOTREES_DISABLE_GPU` simulates a
CPU-only build so the unavailable-GPU path is covered on accelerators too,
and `MOJOTREES_AUTO_MIN_CELLS` enables the otherwise-disabled `auto` size
heuristic so both sides of it are covered. Tests that actually train on the
GPU skip (passing) when no accelerator is present.

The crossover-rule tests below go through `decide_device` with injected
capabilities rather than through `resolve_device`, because that is the only
way to put a *reported* Apple M4 in front of the policy on a machine that
is not one. `DeviceCapabilities` is plain data by design so this works, and
the engine cannot tell an injected device from a read one. Nothing here
trains: a crossover rule chooses which backend runs, and both backends are
already checked against each other by the equivalence tests further down.
"""

from std.os import remove, setenv
from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.apple_gpu_policy import (
    API_CUDA,
    API_METAL,
    API_UNKNOWN,
    APPLE_GEN_M3,
    APPLE_GEN_M4,
    APPLE_GEN_UNKNOWN,
    GpuProfile,
    apple_m4_observed,
)
from mojotrees.boosting import BINARY_LOGISTIC, BoosterParams, SQUARED_ERROR
from mojotrees.device import (
    AUTO_DEVICE,
    CPU_DEVICE,
    GPU_DEVICE,
    NO_DEVICE,
    device_name,
    env_auto_min_cells,
    gpu_available,
    parse_device,
    resolve_device,
)
from mojotrees.device_policy import (
    AUTO_GPU_MIN_ROWS,
    AUTO_MIN_CELLS,
    BLOCK_DERIVATIVE_PRECISION,
    CrossoverEvidence,
    DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD,
    DECISION_AUTO_CPU_BELOW_EVIDENCE,
    DECISION_AUTO_CPU_BLOCKED,
    DECISION_AUTO_GPU_ENV_THRESHOLD,
    DECISION_AUTO_GPU_EVIDENCE,
    DECISION_EXPLICIT_CPU,
    DECISION_EXPLICIT_GPU,
    DECISION_GPU_REFUSED,
    DeviceCapabilities,
    DeviceRequest,
    EVIDENCE_ENV,
    EVIDENCE_EXPLICIT,
    M4_MULTICLASS_EVIDENCE_ID,
    M4_MULTICLASS_MEASURED_ON,
    M4_MULTICLASS_MIN_FEATURES,
    M4_MULTICLASS_MIN_OUTPUTS,
    M4_MULTICLASS_RULE_NAME,
    M4_TRAINING_EVIDENCE_ID,
    M4_TRAINING_MAX_OUTPUTS,
    M4_TRAINING_MEASURED_ON,
    M4_TRAINING_MIN_FEATURES,
    M4_TRAINING_RULE_NAME,
    OBJECTIVE_UNSPECIFIED,
    POLICY_VERSION,
    PROFILE_FALLBACK,
    PROFILE_REPORTED,
    crossover_rules,
    decide_device,
    describe_device_decision,
)
from mojotrees.initialization import SessionState
from mojotrees.model import fit, fit_multiclass
from mojotrees.serialize import load_model, save_model
from mojotrees.tree import TreeParams
from mojotrees.unified_memory_policy import SessionMemoryPlan
from support import _make_features as _features

comptime _DISABLE_GPU = "MOJOTREES_DISABLE_GPU"
comptime _AUTO_MIN_CELLS = "MOJOTREES_AUTO_MIN_CELLS"
comptime _TMP_PATH = "./.test_device_roundtrip.tmp"

# The bin count both records in `M4_TRAINING_EVIDENCE_ID` were taken at.
comptime _MEASURED_BINS = 255


def _set_env(name: String, value: String):
    """Set an override; the empty string restores the default (both knobs
    treat unset and empty alike)."""
    _ = setenv(name, value, True)


def _target(features: List[Float64], n_rows: Int) -> List[Float64]:
    """Distinct per-feature effects, so CPU and GPU split decisions do not
    sit on knife-edge gain ties."""
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(
            4.0 * features[0 * n_rows + r]
            - 3.0 * features[1 * n_rows + r]
            + 2.0
            * (features[2 * n_rows + r] - 0.5)
            * (features[3 * n_rows + r] - 0.5)
        )
    return y^


def _labels(features: List[Float64], n_rows: Int, n_classes: Int) -> List[Int]:
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        var k = Int(features[0 * n_rows + r] * Float64(n_classes))
        if k >= n_classes:
            k = n_classes - 1
        labels.append(k)
    return labels^


def test_device_names_round_trip() raises:
    assert_equal(parse_device("cpu"), CPU_DEVICE)
    assert_equal(parse_device("gpu"), GPU_DEVICE)
    assert_equal(parse_device("auto"), AUTO_DEVICE)
    assert_equal(device_name(CPU_DEVICE), "cpu")
    assert_equal(device_name(GPU_DEVICE), "gpu")
    assert_equal(device_name(AUTO_DEVICE), "auto")


def test_invalid_device_raises() raises:
    with assert_raises():
        _ = parse_device("cuda")
    with assert_raises():
        _ = parse_device("CPU")
    with assert_raises():
        _ = parse_device("")
    with assert_raises():
        _ = device_name(7)
    with assert_raises():
        _ = resolve_device(7, 100, 10, 1)


def test_cpu_always_resolves_to_cpu() raises:
    # Large workload, no accelerator, either way: cpu means cpu.
    assert_equal(resolve_device(CPU_DEVICE, 10_000_000, 100, 1), CPU_DEVICE)
    _set_env(_DISABLE_GPU, "1")
    assert_equal(resolve_device(CPU_DEVICE, 10_000_000, 100, 1), CPU_DEVICE)
    _set_env(_DISABLE_GPU, "")
    assert_equal(resolve_device(CPU_DEVICE, 10, 2, 1), CPU_DEVICE)


def test_explicit_gpu_raises_when_unavailable() raises:
    _set_env(_DISABLE_GPU, "1")
    assert_false(gpu_available())
    # No silent fallback: the request fails instead of returning CPU.
    with assert_raises():
        _ = resolve_device(GPU_DEVICE, 1_000_000, 100, 1)
    _set_env(_DISABLE_GPU, "")

    if gpu_available():
        assert_equal(resolve_device(GPU_DEVICE, 10, 2, 1), GPU_DEVICE)
    else:
        with assert_raises():
            _ = resolve_device(GPU_DEVICE, 10, 2, 1)


def test_gpu_accepts_multiclass() raises:
    # Multiclass is no longer an unsupported workload: it grows one tree per
    # class per round through train_multiclass_gpu. Resolution therefore
    # turns only on whether an accelerator is present, exactly as it does
    # for single-output training.
    comptime if not has_accelerator():
        with assert_raises():
            _ = resolve_device(GPU_DEVICE, 1_000, 10, 3)
    else:
        assert_equal(resolve_device(GPU_DEVICE, 1_000, 10, 3), GPU_DEVICE)


def test_auto_chooses_cpu_by_default() raises:
    assert_equal(env_auto_min_cells(), -1)
    assert_equal(resolve_device(AUTO_DEVICE, 10, 2, 1), CPU_DEVICE)
    assert_equal(
        resolve_device(AUTO_DEVICE, 10_000_000, 100, 1), CPU_DEVICE
    )
    _set_env(_DISABLE_GPU, "1")
    assert_equal(
        resolve_device(AUTO_DEVICE, 10_000_000, 100, 1), CPU_DEVICE
    )
    _set_env(_DISABLE_GPU, "")


def test_auto_honors_the_size_threshold() raises:
    _set_env(_AUTO_MIN_CELLS, "10000")
    assert_equal(env_auto_min_cells(), 10_000)

    var expected_big = GPU_DEVICE if gpu_available() else CPU_DEVICE
    # Below the threshold: CPU regardless of hardware.
    assert_equal(resolve_device(AUTO_DEVICE, 50, 10, 1), CPU_DEVICE)
    # At and above it: GPU when one is available.
    assert_equal(resolve_device(AUTO_DEVICE, 1_000, 10, 1), expected_big)
    assert_equal(resolve_device(AUTO_DEVICE, 100_000, 10, 1), expected_big)
    # Multiclass is inside the GPU path now, so it obeys the same threshold
    # as single-output training rather than being forced to the CPU.
    assert_equal(resolve_device(AUTO_DEVICE, 50, 10, 3), CPU_DEVICE)
    assert_equal(resolve_device(AUTO_DEVICE, 100_000, 10, 3), expected_big)

    # A disabled accelerator sends everything back to the CPU.
    _set_env(_DISABLE_GPU, "1")
    assert_equal(resolve_device(AUTO_DEVICE, 100_000, 10, 1), CPU_DEVICE)
    _set_env(_DISABLE_GPU, "")

    # Unparsable and negative overrides fall back to disabled.
    _set_env(_AUTO_MIN_CELLS, "many")
    assert_equal(env_auto_min_cells(), -1)
    _set_env(_AUTO_MIN_CELLS, "-5")
    assert_equal(resolve_device(AUTO_DEVICE, 100_000, 10, 1), CPU_DEVICE)
    _set_env(_AUTO_MIN_CELLS, "")
    assert_equal(env_auto_min_cells(), -1)


def _caps_for(
    var profile: GpuProfile,
    profile_source: Int = PROFILE_REPORTED,
    auto_min_cells: Int = AUTO_MIN_CELLS,
    derivative_precision_float64: Bool = False,
) raises -> DeviceCapabilities:
    """A machine with a working accelerator and this profile.

    Constructed rather than detected, which is the whole point:
    `DeviceCapabilities.detect` opens no device and so can never produce a
    `PROFILE_REPORTED` profile, and the hardware-scoped rules can only
    match one. `gpu_available` is True regardless of what this build was
    compiled on, so these tests run identically on a laptop with a GPU and
    on a CI runner without one.
    """
    return DeviceCapabilities(
        True,
        True,
        False,
        profile^,
        profile_source,
        auto_min_cells,
        SessionState.cold(),
        SessionMemoryPlan.staged(),
        derivative_precision_float64=derivative_precision_float64,
    )


def _request(
    device: Int,
    n_rows: Int,
    n_features: Int,
    n_outputs: Int = 1,
    objective: Int = SQUARED_ERROR,
) raises -> DeviceRequest:
    """A dense, complete request at the bin count the records were taken
    at, so that only the field a test varies is doing the work."""
    return DeviceRequest(
        device, n_rows, n_features, n_outputs, _MEASURED_BINS, objective
    )


def _cuda_profile() -> GpuProfile:
    """A plausible discrete NVIDIA part, reported. No such device has ever
    run this code (docs/GPU_VALIDATION.md), which is the point of the test
    that uses it."""
    return GpuProfile(
        API_CUDA, APPLE_GEN_UNKNOWN, 108, 1024, 49152, 0, False, False
    )


def _m3_profile() -> GpuProfile:
    """A Metal device that is not the measured generation."""
    return GpuProfile(
        API_METAL, APPLE_GEN_M3, 10, 1024, 32768, 0, True, False
    )


def test_crossover_rules_carry_their_evidence() raises:
    var rules = crossover_rules()
    # Two rules: one single output, one multiclass. A third arriving without
    # this test being edited means somebody installed a threshold and did not
    # think about its scope.
    assert_equal(len(rules), 2)

    var rule = rules[0].copy()
    assert_equal(rule.name, M4_TRAINING_RULE_NAME)
    assert_equal(rule.evidence_id, M4_TRAINING_EVIDENCE_ID)
    assert_equal(rule.measured_on, M4_TRAINING_MEASURED_ON)
    # Scoped to the one device family and generation that has run this
    # code end to end, and to the one objective that was measured.
    assert_equal(rule.api, API_METAL)
    assert_equal(rule.apple_generation, APPLE_GEN_M4)
    assert_equal(rule.objective, SQUARED_ERROR)
    assert_equal(rule.max_outputs, M4_TRAINING_MAX_OUTPUTS)
    # The row floor is the one named constant and nothing else. There is no
    # cell term: `min_cells` was 50,000,000 (the product of the old
    # 1,000,000-row floor and the 50-feature scope) and gated on the same
    # fact twice, so a reader had to divide to find the shipped threshold.
    # It is 0 now, which does not constrain, and `AUTO_GPU_MIN_ROWS` is the
    # only number a reader has to find. A cell term coming back without this
    # test being edited means the threshold has two homes again.
    assert_equal(rule.min_rows, AUTO_GPU_MIN_ROWS)
    assert_equal(rule.min_features, M4_TRAINING_MIN_FEATURES)
    assert_equal(rule.min_cells, 0)
    # And the constant is what this release ships, spelled out rather than
    # referred to, so that lowering it silently is a test edit.
    assert_equal(AUTO_GPU_MIN_ROWS, 250_000)
    # The citation names both the record and the device.
    assert_true(rule.cite().find(M4_TRAINING_EVIDENCE_ID) >= 0)
    assert_true(rule.cite().find("Apple M4") >= 0)

    # The multiclass rule, pinned the same way. Its evidence is one record
    # (`bench/results/profile_2026-08-15/RESULTS.md`, 465,000 x 54 over 7
    # classes, GPU 15.30 s against CPU 25.47 s), and its scope differs from
    # the rule above in three places, each of which is a decision:
    #
    #   - `objective` does not constrain, and `min_outputs` does. Trees per
    #     round is what the trainers branch on and it is what every
    #     multiclass entry point declares; three of the four declare no
    #     objective at all, so an objective-scoped rule would have been
    #     unreachable from them.
    #   - `max_outputs` does not constrain, so the class count is bounded
    #     below and not above. Capping it at the measured seven would send a
    #     ten-class fit to the CPU while a seven-class fit of the same shape
    #     went to the device.
    #   - `min_features` is 54, the feature count the record was taken at,
    #     and deliberately not rounded down to the other rule's 50.
    #
    # Any of the three changing without this test being edited is a rule
    # quietly widening or narrowing past its record.
    var mc = rules[1].copy()
    assert_equal(mc.name, M4_MULTICLASS_RULE_NAME)
    assert_equal(mc.evidence_id, M4_MULTICLASS_EVIDENCE_ID)
    assert_equal(mc.measured_on, M4_MULTICLASS_MEASURED_ON)
    assert_equal(mc.api, API_METAL)
    assert_equal(mc.apple_generation, APPLE_GEN_M4)
    assert_equal(mc.objective, OBJECTIVE_UNSPECIFIED)
    assert_equal(mc.min_outputs, M4_MULTICLASS_MIN_OUTPUTS)
    assert_equal(mc.max_outputs, 0)
    assert_equal(mc.min_rows, AUTO_GPU_MIN_ROWS)
    assert_equal(mc.min_features, M4_MULTICLASS_MIN_FEATURES)
    assert_equal(mc.min_cells, 0)
    assert_equal(M4_MULTICLASS_MIN_FEATURES, 54)
    assert_equal(M4_MULTICLASS_MIN_OUTPUTS, 2)
    assert_true(mc.cite().find(M4_MULTICLASS_EVIDENCE_ID) >= 0)
    assert_true(mc.cite().find("Apple M4") >= 0)

    # The two rules are complementary on trees per round, which is what
    # makes them disjoint: no request can match both, so installing the
    # multiclass rule cannot move a single-output fit.
    assert_true(rule.max_outputs < mc.min_outputs)

    # And the constructor still refuses a rule with nothing under it.
    with assert_raises():
        _ = CrossoverEvidence(
            String("invented"), String(""), String("nowhere")
        )


def test_auto_selects_gpu_at_and_above_the_row_floor() raises:
    """At `AUTO_GPU_MIN_ROWS` and above, `auto` selects the GPU.

    The floor is at 250,000 rows and the two records behind the rule were
    taken at 1,000,000, so this test's lower shape is deliberately *below*
    the measured evidence: that is what `AUTO_GPU_MIN_ROWS` being a
    provisional constant means, and the constant's own comment carries the
    trade. Both shapes are asserted so that raising the floor back to the
    measured point cannot pass this file unedited."""
    var caps = _caps_for(apple_m4_observed())
    var decision = decide_device(
        _request(AUTO_DEVICE, AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES),
        caps,
    )
    assert_equal(decision.selected_device, GPU_DEVICE)
    assert_equal(decision.decision_code, DECISION_AUTO_GPU_EVIDENCE)
    assert_true(decision.validated())
    assert_false(decision.blocked)
    assert_equal(decision.evidence_id, M4_TRAINING_EVIDENCE_ID)
    assert_equal(decision.policy_version, POLICY_VERSION)

    # Larger is still selected: the 2026-08-14 record measured 5,000,000 x
    # 50 as well, and the margin there was wider.
    var bigger = decide_device(
        _request(AUTO_DEVICE, 5_000_000, M4_TRAINING_MIN_FEATURES), caps
    )
    assert_equal(bigger.selected_device, GPU_DEVICE)
    assert_equal(bigger.decision_code, DECISION_AUTO_GPU_EVIDENCE)


def test_a_selected_gpu_says_what_measured_it() raises:
    """The decision has to be findable, not just correct."""
    var caps = _caps_for(apple_m4_observed())
    var decision = decide_device(
        _request(AUTO_DEVICE, AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES),
        caps,
    )

    # The rule that fired, the record it came from, and the device it was
    # taken on all reach the message.
    assert_true(decision.message.find(M4_TRAINING_RULE_NAME) >= 0)
    assert_true(decision.message.find(M4_TRAINING_EVIDENCE_ID) >= 0)
    assert_true(decision.message.find("Apple M4") >= 0)

    var wire = decision.serialize()
    assert_true(wire.find("decision=auto-gpu-evidence\n") >= 0)
    assert_true(wire.find("validated=true\n") >= 0)
    assert_true(wire.find("crossover_rules_installed=2\n") >= 0)
    assert_true(
        wire.find(String("crossover_rule=", decision.crossover_citation, "\n"))
        >= 0
    )
    assert_true(
        wire.find(String("evidence_id=", M4_TRAINING_EVIDENCE_ID, "\n")) >= 0
    )

    var line = describe_device_decision(decision)
    assert_true(line.find("auto -> gpu") >= 0)
    assert_true(line.find("auto-gpu-evidence") >= 0)
    assert_true(line.find(M4_TRAINING_EVIDENCE_ID) >= 0)
    assert_true(line.find("Apple M4") >= 0)


def _declines(request: DeviceRequest, caps: DeviceCapabilities) raises:
    """Auto keeps the CPU here, and says a rule was consulted and declined
    rather than that none exists."""
    var decision = decide_device(request, caps)
    assert_equal(decision.selected_device, CPU_DEVICE)
    assert_equal(decision.decision_code, DECISION_AUTO_CPU_BELOW_EVIDENCE)
    assert_false(decision.validated())
    assert_equal(decision.crossover_citation, String(""))
    assert_true(decision.serialize().find("crossover_rule=") < 0)


def test_auto_declines_beside_the_measured_shape() raises:
    var caps = _caps_for(apple_m4_observed())

    # One row and one feature short of the floor. The floor is
    # `AUTO_GPU_MIN_ROWS`, a provisional constant rather than a measured
    # crossover, but it is still a hard edge: one row under it is the CPU,
    # and a caller cannot get the GPU by rounding.
    _declines(
        _request(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS - 1,
            M4_TRAINING_MIN_FEATURES,
        ),
        caps,
    )
    _declines(
        _request(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS,
            M4_TRAINING_MIN_FEATURES - 1,
        ),
        caps,
    )
    # Same cell count, a tenth of the features: a different ratio of
    # per-node cost to per-node work, and unmeasured.
    _declines(_request(AUTO_DEVICE, 5_000_000, 10), caps)
    # Multiclass grows one tree per class per round through another trainer
    # and is out of THIS rule's scope (`max_outputs=1`). It is no longer
    # unmeasured, which is what this comment used to say: it has a rule of
    # its own, and the two shapes below are on the wrong side of that rule
    # rather than uncovered by any rule.
    #
    # 50 features is one such shape, and the reason is worth spelling out
    # because otherwise this assertion passes for a reason nobody wrote
    # down: `M4_MULTICLASS_MIN_FEATURES` is 54, the feature count the
    # multiclass record was taken at, so a 50-feature softmax fit is below
    # the multiclass scope even though it clears the single-output one.
    _declines(
        _request(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS,
            M4_TRAINING_MIN_FEATURES,
            3,
        ),
        caps,
    )
    # And at the multiclass rule's own feature count, one row under the
    # shared floor. This is the assertion that would fail if the multiclass
    # rule stopped gating on rows, which the one above cannot see.
    _declines(
        _request(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS - 1,
            M4_MULTICLASS_MIN_FEATURES,
            3,
        ),
        caps,
    )
    # The control for both: same fixture, same class count, at the shape the
    # multiclass rule does cover. Without this the two declines above could
    # both be passing because no multiclass request ever reaches a rule.
    var covered = decide_device(
        _request(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS,
            M4_MULTICLASS_MIN_FEATURES,
            3,
        ),
        caps,
    )
    assert_equal(covered.selected_device, GPU_DEVICE)
    assert_equal(covered.evidence_id, M4_MULTICLASS_EVIDENCE_ID)
    # Another objective, and an undeclared one.
    _declines(
        _request(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS,
            M4_TRAINING_MIN_FEATURES,
            1,
            BINARY_LOGISTIC,
        ),
        caps,
    )
    _declines(
        _request(
            AUTO_DEVICE,
            AUTO_GPU_MIN_ROWS,
            M4_TRAINING_MIN_FEATURES,
            1,
            OBJECTIVE_UNSPECIFIED,
        ),
        caps,
    )


def test_unmeasured_hardware_still_gets_no_rule() raises:
    # A device nobody here owns, at a shape far past the measured one. The
    # CUDA and HIP rows of docs/GPU_VALIDATION.md are `not run`, and a rule
    # measured on Metal does not transfer to them.
    _declines(
        _request(AUTO_DEVICE, 5_000_000, 100), _caps_for(_cuda_profile())
    )
    # Metal, but not the measured generation.
    _declines(
        _request(AUTO_DEVICE, 5_000_000, 100), _caps_for(_m3_profile())
    )

    # And the case a defaulted `fit` actually hits: no attributes were read
    # at all, so every hardware-scoped rule is out of reach before the
    # shape is even compared. The message has to say which of the two
    # things went wrong.
    var blind = decide_device(
        _request(
            AUTO_DEVICE, AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES
        ),
        _caps_for(GpuProfile.generic(), PROFILE_FALLBACK),
    )
    assert_equal(blind.selected_device, CPU_DEVICE)
    assert_equal(blind.decision_code, DECISION_AUTO_CPU_BELOW_EVIDENCE)
    assert_true(blind.message.find("No device attributes were read") >= 0)
    assert_equal(blind.capabilities.profile.api, API_UNKNOWN)

    # The narrow entry point keeps the CPU on the measured shape too, and as
    # of 2026-08-16 that is for a different reason than the fixture above.
    # Detection can now name the hardware (`PROFILE_BUILD_TARGET`), so what
    # declines the rule here is the undeclared objective: every rule is
    # scoped to the objective it was measured on and a four-argument call
    # declares none. That is deliberate and must stay; the remaining gap is
    # at the six call sites that hold an objective and drop it.
    # `tests/test_device_auto_crossover.mojo` pins both halves, including the
    # five-argument call that does reach the GPU here.
    assert_equal(
        resolve_device(
            AUTO_DEVICE, AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES, 1
        ),
        CPU_DEVICE,
    )


def test_the_rule_leaves_explicit_requests_alone() raises:
    var caps = _caps_for(apple_m4_observed())

    # Explicit gpu on a tiny shape no rule covers: still the GPU, still
    # unvalidated, still carrying no crossover citation.
    var forced = decide_device(_request(GPU_DEVICE, 10, 2), caps)
    assert_equal(forced.selected_device, GPU_DEVICE)
    assert_equal(forced.decision_code, DECISION_EXPLICIT_GPU)
    assert_equal(forced.evidence_id, EVIDENCE_EXPLICIT)
    assert_false(forced.validated())
    assert_equal(forced.crossover_citation, String(""))

    # Explicit cpu on the shape the rule does cover: still the CPU.
    var pinned = decide_device(
        _request(CPU_DEVICE, AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES),
        caps,
    )
    assert_equal(pinned.selected_device, CPU_DEVICE)
    assert_equal(pinned.decision_code, DECISION_EXPLICIT_CPU)

    # The escape hatch still outranks the table in both directions: it
    # reaches the GPU below every rule, and it holds the CPU above them.
    var by_env = decide_device(
        _request(AUTO_DEVICE, 10, 2),
        _caps_for(apple_m4_observed(), auto_min_cells=0),
    )
    assert_equal(by_env.selected_device, GPU_DEVICE)
    assert_equal(by_env.decision_code, DECISION_AUTO_GPU_ENV_THRESHOLD)
    assert_equal(by_env.evidence_id, EVIDENCE_ENV)
    assert_false(by_env.validated())

    var held = decide_device(
        _request(AUTO_DEVICE, AUTO_GPU_MIN_ROWS, M4_TRAINING_MIN_FEATURES),
        _caps_for(
            apple_m4_observed(),
            auto_min_cells=AUTO_GPU_MIN_ROWS * M4_TRAINING_MIN_FEATURES + 1,
        ),
    )
    assert_equal(held.selected_device, CPU_DEVICE)
    assert_equal(held.decision_code, DECISION_AUTO_CPU_BELOW_ENV_THRESHOLD)


def test_float64_derivatives_route_auto_to_the_cpu() raises:
    """`MOJOTREES_DERIVATIVE_PRECISION=float64` sends `auto` to the CPU, and
    it does so for a reason that is not the shape.

    The shape here is 5,000,000 x 50, twenty times `AUTO_GPU_MIN_ROWS` and
    five times the largest shape any record covers, so "the CPU because it
    is small" is not available as an explanation: the same request with the
    flag off selects the GPU, asserted immediately below the first
    assertion so that the two answers sit side by side. What changed is the
    capability, not the workload.

    Why it is a block and not a rule scope. `gpu_gradient_stream`'s upload
    narrows every per-row derivative to Float32, so the accelerator cannot
    produce the Float64 answer at any size; precision outranks the crossover
    rather than being weighed against it, and `decide_device` reads the
    blocks before it reads the table. `DECISION_AUTO_CPU_BLOCKED` rather
    than `DECISION_AUTO_CPU_BELOW_EVIDENCE` is what says which of those two
    happened, and asserting on it is what stops this test from passing for
    the wrong reason.
    """
    var caps = _caps_for(
        apple_m4_observed(), derivative_precision_float64=True
    )
    var big = _request(AUTO_DEVICE, 5_000_000, M4_TRAINING_MIN_FEATURES)
    var decision = decide_device(big, caps)
    assert_equal(decision.selected_device, CPU_DEVICE)
    assert_equal(decision.decision_code, DECISION_AUTO_CPU_BLOCKED)
    assert_false(decision.validated())
    assert_equal(
        decision.blocking_reasons.codes[0], BLOCK_DERIVATIVE_PRECISION
    )

    # The control, at the identical shape: with the flag off this request
    # reaches the GPU. Without this line the test above would pass on any
    # build where `auto` never reaches the GPU at all.
    var permitted = decide_device(
        _request(AUTO_DEVICE, 5_000_000, M4_TRAINING_MIN_FEATURES),
        _caps_for(apple_m4_observed()),
    )
    assert_equal(permitted.selected_device, GPU_DEVICE)
    assert_equal(permitted.decision_code, DECISION_AUTO_GPU_EVIDENCE)

    # And it holds below the floor too, where the CPU was the answer
    # anyway: the reason has to be the precision either way, because a
    # report that named the shape would send the reader to the wrong knob.
    var small = decide_device(_request(AUTO_DEVICE, 1_000, 4), caps)
    assert_equal(small.selected_device, CPU_DEVICE)
    assert_equal(small.decision_code, DECISION_AUTO_CPU_BLOCKED)
    assert_equal(small.blocking_reasons.codes[0], BLOCK_DERIVATIVE_PRECISION)


def test_float64_derivatives_still_refuse_an_explicit_gpu() raises:
    """The other half of the asymmetry, and it must not soften.

    A caller who wrote `device='gpu'` named a backend, and that backend
    cannot honor a Float64 derivative request. Telling them so is the whole
    correction: the defect being fixed is that the flag was accepted and the
    Float32 answer returned. `auto` is the only request that may be routed,
    because `auto` is the only one that asked us to choose.
    """
    var caps = _caps_for(
        apple_m4_observed(), derivative_precision_float64=True
    )
    var decision = decide_device(
        _request(GPU_DEVICE, 5_000_000, M4_TRAINING_MIN_FEATURES), caps
    )
    assert_true(decision.blocked)
    assert_equal(decision.selected_device, NO_DEVICE)
    assert_equal(decision.decision_code, DECISION_GPU_REFUSED)
    assert_equal(
        decision.blocking_reasons.codes[0], BLOCK_DERIVATIVE_PRECISION
    )
    assert_true(decision.message.find("float64") >= 0)
    with assert_raises():
        decision.raise_if_blocked()

    # An explicit `cpu` is unaffected: it was always going to honor it.
    var pinned = decide_device(
        _request(CPU_DEVICE, 5_000_000, M4_TRAINING_MIN_FEATURES), caps
    )
    assert_equal(pinned.selected_device, CPU_DEVICE)
    assert_false(pinned.blocked)


def test_sparse_input_keeps_the_cpu_under_auto() raises:
    # The rule is measured on dense matrices, and the sparse branch in
    # `decide_device` runs before the table is consulted, so a sparse run
    # at the measured shape must not inherit it.
    var request = DeviceRequest(
        AUTO_DEVICE,
        AUTO_GPU_MIN_ROWS,
        M4_TRAINING_MIN_FEATURES,
        1,
        _MEASURED_BINS,
        SQUARED_ERROR,
        True,
    )
    var decision = decide_device(request, _caps_for(apple_m4_observed()))
    assert_equal(decision.selected_device, CPU_DEVICE)
    assert_false(decision.validated())
    assert_equal(decision.crossover_citation, String(""))


def test_fit_cpu_and_auto_agree() raises:
    var n_rows = 400
    var n_features = 5
    var features = _features(n_rows, n_features)
    var target = _target(features, n_rows)
    var params = BoosterParams(10, 0.1, TreeParams.default())

    var cpu = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params,
        device=CPU_DEVICE,
    )
    var auto = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params,
        device=AUTO_DEVICE,
    )
    # auto resolves to the CPU trainer here, so this is bit-exact.
    assert_equal(len(cpu.booster.trees), len(auto.booster.trees))
    for r in range(n_rows):
        var row = List[Float64](capacity=n_features)
        for f in range(n_features):
            row.append(features[f * n_rows + r])
        assert_equal(cpu.predict(row), auto.predict(row))


def test_fit_rejects_unknown_device() raises:
    var features = _features(50, 4)
    var target = _target(features, 50)
    var params = BoosterParams(2, 0.1, TreeParams.default())
    with assert_raises():
        _ = fit(
            features, 50, 4, target, SQUARED_ERROR, params, device=42
        )


def test_fit_multiclass_device_selection() raises:
    var n_rows = 300
    var n_features = 4
    var features = _features(n_rows, n_features)
    var labels = _labels(features, n_rows, 3)
    var params = BoosterParams(5, 0.1, TreeParams.default())

    # 300 x 4 is three orders of magnitude under `AUTO_GPU_MIN_ROWS` and
    # under the multiclass rule's feature scope, so auto resolves to the CPU
    # here. That is a statement about this shape and not about multiclass:
    # since `apple-m4-metal-dense-multiclass` was installed, `auto` reaches
    # `train_multiclass_gpu` above the floor, and
    # `tests/test_gpu_auto_reaches_multiclass.mojo` proves it does so by the
    # model bits. What this test is for is that `fit_multiclass` dispatches
    # on the backend it resolved at all.
    var model = fit_multiclass(
        features, n_rows, n_features, labels, 3, params, device=AUTO_DEVICE
    )
    assert_equal(model.booster.n_classes, 3)

    # Explicit gpu trains on the device when there is one, and raises only
    # for the absent accelerator, not for the workload.
    comptime if not has_accelerator():
        with assert_raises():
            _ = fit_multiclass(
                features, n_rows, n_features, labels, 3, params,
                device=GPU_DEVICE,
            )
    else:
        var on_gpu = fit_multiclass(
            features, n_rows, n_features, labels, 3, params,
            device=GPU_DEVICE,
        )
        assert_equal(on_gpu.booster.n_classes, 3)
        assert_equal(
            len(on_gpu.booster.trees), len(model.booster.trees)
        )


def test_fit_gpu_matches_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 6
        var features = _features(n_rows, n_features)
        var target = _target(features, n_rows)
        var params = BoosterParams(10, 0.1, TreeParams(15, 20, 1.0, 1e-3))

        var cpu = fit(
            features, n_rows, n_features, target, SQUARED_ERROR, params,
            device=CPU_DEVICE,
        )
        var gpu = fit(
            features, n_rows, n_features, target, SQUARED_ERROR, params,
            device=GPU_DEVICE,
        )
        assert_equal(len(cpu.booster.trees), len(gpu.booster.trees))
        # GPU histograms are Float32 fixed-point, so agreement is
        # tolerance-based (see train_gpu.mojo).
        for r in range(n_rows):
            var row = List[Float64](capacity=n_features)
            for f in range(n_features):
                row.append(features[f * n_rows + r])
            assert_true(abs(cpu.predict(row) - gpu.predict(row)) <= 1e-3)


def test_gpu_trained_model_round_trips() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 500
        var n_features = 4
        var features = _features(n_rows, n_features)
        var target = _target(features, n_rows)
        var params = BoosterParams(8, 0.1, TreeParams.default())
        var model = fit(
            features, n_rows, n_features, target, SQUARED_ERROR, params,
            device=GPU_DEVICE,
        )

        # The device is a training choice, not part of the model, so the
        # file format is unchanged and the round trip stays bit-exact.
        save_model(model, _TMP_PATH)
        var loaded = load_model(_TMP_PATH)
        remove(_TMP_PATH)
        assert_equal(
            len(model.booster.trees), len(loaded.booster.trees)
        )
        for r in range(n_rows):
            var row = List[Float64](capacity=n_features)
            for f in range(n_features):
                row.append(features[f * n_rows + r])
            assert_equal(model.predict(row), loaded.predict(row))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
