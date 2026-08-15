"""The crossover table and the probe that lets it match.

Policy version 2 carries two benchmark-derived rules, both scoped to the
Apple M4 the sweep ran on. These tests inject that device and others as
reported capabilities, so they run the same on a machine with no
accelerator, and pin: the rules fire for the measured objectives from the
measured cell count up, and for nothing else; the fallback profile never
matches a hardware-scoped rule; and the shipped `MOJOTREES_AUTO_MIN_CELLS`
override still outranks the table. The one hardware-touching test checks
that `probe_device_profile` reports an API when an accelerator is present.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.apple_gpu_policy import API_METAL, API_UNKNOWN, APPLE_GEN_M4
from mojotrees.boosting import BINARY_LOGISTIC, HUBER, SQUARED_ERROR
from mojotrees.device_policy import (
    AUTO_DEVICE,
    CPU_DEVICE,
    DECISION_AUTO_CPU_BELOW_EVIDENCE,
    DECISION_AUTO_GPU_EVIDENCE,
    DeviceCapabilities,
    DeviceRequest,
    GPU_DEVICE,
    M4_CROSSOVER_EVIDENCE,
    M4_CROSSOVER_MIN_CELLS,
    M4_CROSSOVER_MIN_ROWS,
    POLICY_VERSION,
    PROFILE_REPORTED,
    capabilities_from_reported,
    crossover_rules,
    decide_device,
    probe_device_profile,
)


def _m4() raises -> DeviceCapabilities:
    """What the development M4 reports: `metal`, `4-metal4`, ten cores."""
    return capabilities_from_reported("metal", "4-metal4", 10, 1024, 32768)


def _request(
    n_rows: Int, n_features: Int, objective: Int, n_outputs: Int = 1
) -> DeviceRequest:
    return DeviceRequest(
        AUTO_DEVICE,
        n_rows,
        n_features,
        n_outputs,
        255,
        objective,
    )


def test_policy_version_and_table_shape() raises:
    assert_equal(POLICY_VERSION, 2)
    var rules = crossover_rules()
    assert_equal(len(rules), 2)
    for i in range(len(rules)):
        assert_equal(rules[i].api, API_METAL)
        assert_equal(rules[i].apple_generation, APPLE_GEN_M4)
        assert_equal(rules[i].min_cells, M4_CROSSOVER_MIN_CELLS)
        assert_equal(rules[i].min_rows, M4_CROSSOVER_MIN_ROWS)
        assert_equal(rules[i].max_outputs, 1)
        assert_equal(rules[i].evidence_id, M4_CROSSOVER_EVIDENCE)


def test_m4_rules_fire_from_the_measured_cell_count() raises:
    var caps = _m4()
    assert_equal(caps.profile_source, PROFILE_REPORTED)
    assert_equal(caps.profile.apple_generation, APPLE_GEN_M4)
    # Exactly at the threshold, and comfortably above it, both objectives.
    var at = decide_device(
        _request(M4_CROSSOVER_MIN_CELLS // 50, 50, SQUARED_ERROR), caps
    )
    assert_equal(at.selected_device, GPU_DEVICE)
    assert_equal(at.decision_code, DECISION_AUTO_GPU_EVIDENCE)
    assert_equal(at.evidence_id, M4_CROSSOVER_EVIDENCE)
    var big = decide_device(
        _request(5_000_000, 50, BINARY_LOGISTIC), caps
    )
    assert_equal(big.selected_device, GPU_DEVICE)
    assert_equal(big.decision_code, DECISION_AUTO_GPU_EVIDENCE)
    # One cell short keeps the CPU, and says a rule fell short.
    var under = decide_device(
        _request(M4_CROSSOVER_MIN_CELLS // 50 - 1, 50, SQUARED_ERROR), caps
    )
    assert_equal(under.selected_device, CPU_DEVICE)
    assert_equal(under.decision_code, DECISION_AUTO_CPU_BELOW_EVIDENCE)
    # Enough cells but too few rows (a wide, short matrix) keeps the CPU
    # too: the row floor is part of what was measured.
    var wide = decide_device(
        _request(
            M4_CROSSOVER_MIN_ROWS - 1,
            M4_CROSSOVER_MIN_CELLS // (M4_CROSSOVER_MIN_ROWS - 1) + 1,
            SQUARED_ERROR,
        ),
        caps,
    )
    assert_equal(wide.selected_device, CPU_DEVICE)
    assert_equal(wide.decision_code, DECISION_AUTO_CPU_BELOW_EVIDENCE)


def test_m4_rules_cover_nothing_they_did_not_measure() raises:
    var caps = _m4()
    # An objective the sweep did not run.
    var huber = decide_device(_request(5_000_000, 50, HUBER), caps)
    assert_equal(huber.selected_device, CPU_DEVICE)
    assert_equal(huber.decision_code, DECISION_AUTO_CPU_BELOW_EVIDENCE)
    # A shape-only request, which names no objective.
    var shape = DeviceRequest(AUTO_DEVICE, 5_000_000, 50)
    assert_equal(decide_device(shape, caps).selected_device, CPU_DEVICE)
    # Multiclass: more than one tree per round.
    var multi = decide_device(
        _request(5_000_000, 50, SQUARED_ERROR, n_outputs=3), caps
    )
    assert_equal(multi.selected_device, CPU_DEVICE)


def test_other_hardware_never_matches_the_m4_rules() raises:
    var request = _request(5_000_000, 50, SQUARED_ERROR)
    # Another Apple generation.
    var m3 = capabilities_from_reported("metal", "3-metal3", 10, 1024, 32768)
    assert_equal(decide_device(request, m3).selected_device, CPU_DEVICE)
    # An unparsable Metal architecture string: the generation is unknown,
    # so a generation-scoped rule cannot claim it.
    var odd = capabilities_from_reported("metal", "apple-gpu", 10, 1024, 32768)
    assert_equal(decide_device(request, odd).selected_device, CPU_DEVICE)
    # CUDA and HIP.
    var cuda = capabilities_from_reported("cuda", "sm_90", 132, 1024, 49152)
    assert_equal(decide_device(request, cuda).selected_device, CPU_DEVICE)
    var hip = capabilities_from_reported("hip", "gfx942", 304, 1024, 65536)
    assert_equal(decide_device(request, hip).selected_device, CPU_DEVICE)
    # The portable fallback profile, which names no API at all.
    var generic = DeviceCapabilities.detect()
    assert_equal(generic.profile.api, API_UNKNOWN)
    assert_equal(decide_device(request, generic).selected_device, CPU_DEVICE)


def test_probe_reports_an_api_when_an_accelerator_is_present() raises:
    var profile = probe_device_profile()
    comptime if has_accelerator():
        assert_true(profile.api != API_UNKNOWN)
        var caps = DeviceCapabilities.detect(probe_device=True)
        assert_equal(caps.profile_source, PROFILE_REPORTED)
    else:
        assert_equal(profile.api, API_UNKNOWN)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
