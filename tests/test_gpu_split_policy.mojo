"""Pure tests for workload-aware GPU split selection."""

# run_tests: cpu-safe -- host arithmetic only, opens no device.

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.gpu_split_policy import (
    M4_EVIDENCE_ID,
    SPLIT_POLICY_DEVICE_RESIDENT,
    SPLIT_POLICY_HOST,
    SPLIT_REASON_BELOW_CROSSOVER,
    SPLIT_REASON_RESIDENT_MEMORY,
    SPLIT_REASON_UNKNOWN_HARDWARE,
    SPLIT_REASON_UNSUPPORTED,
    SPLIT_REASON_VALIDATED_WORKLOAD,
    SplitSearchDecision,
    decide_split_search,
    normalized_split_work,
)


def _m4(
    rows: Int,
    features: Int = 100,
    bins: Int = 255,
    leaves: Int = 31,
    supported: Bool = True,
    fits: Bool = True,
) -> SplitSearchDecision:
    return decide_split_search(
        "metal", "4-metal4", rows, features, bins, leaves, supported, fits
    )


def test_small_m4_workload_stays_on_host() raises:
    var decision = _m4(50_000)
    assert_equal(decision.selected, SPLIT_POLICY_HOST)
    assert_equal(decision.reason, SPLIT_REASON_BELOW_CROSSOVER)
    assert_false(decision.uses_device())


def test_large_validated_m4_workload_uses_resident_device() raises:
    var decision = _m4(1_000_000)
    assert_equal(decision.selected, SPLIT_POLICY_DEVICE_RESIDENT)
    assert_equal(decision.reason, SPLIT_REASON_VALIDATED_WORKLOAD)
    assert_equal(decision.evidence_id, M4_EVIDENCE_ID)
    assert_true(decision.uses_device())


def test_unknown_hardware_is_conservative() raises:
    var decision = decide_split_search(
        "cuda", "sm_100", 5_000_000, 100, 255, 31, True, True
    )
    assert_equal(decision.selected, SPLIT_POLICY_HOST)
    assert_equal(decision.reason, SPLIT_REASON_UNKNOWN_HARDWARE)


def test_semantics_and_memory_gate_before_profitability() raises:
    var unsupported = _m4(1_000_000, supported=False)
    assert_equal(unsupported.selected, SPLIT_POLICY_HOST)
    assert_equal(unsupported.reason, SPLIT_REASON_UNSUPPORTED)

    var too_wide = _m4(1_000_000, fits=False)
    assert_equal(too_wide.selected, SPLIT_POLICY_HOST)
    assert_equal(too_wide.reason, SPLIT_REASON_RESIDENT_MEMORY)


def test_normalization_accounts_for_bins_and_leaves() raises:
    var baseline = normalized_split_work(1_000_000, 100, 255, 31)
    var fewer_bins = normalized_split_work(1_000_000, 100, 63, 31)
    var fewer_leaves = normalized_split_work(1_000_000, 100, 255, 15)
    assert_equal(baseline, 100_000_000.0)
    assert_true(fewer_bins < baseline)
    assert_true(fewer_leaves < baseline)


def test_description_carries_reason_work_threshold_and_evidence() raises:
    var line = _m4(1_000_000).describe()
    assert_true(line.find("split_strategy=device-resident") >= 0)
    assert_true(line.find("reason=validated-workload") >= 0)
    assert_true(line.find("threshold=") >= 0)
    assert_true(line.find(M4_EVIDENCE_ID) >= 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
