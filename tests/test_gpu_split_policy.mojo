"""Pure tests for workload-aware GPU split selection.

Two rules live in the policy and both are asserted here. Rule 1 is the
leaf-wise threshold of 50,000,000 normalized work, unchanged since 2026-08-14.
Rule 2 is the depth-wise floor of 12,500,000 added 2026-08-15, which is the
single interleaved point measured at 250,000 x 50 on an Apple M4 and not a
fitted curve; these tests pin where it applies, where it deliberately does not,
and that it can only ever add a device selection rather than remove one.

Everything here is host arithmetic. No device is opened, nothing is timed, and
no assertion in this file is evidence that either path is faster; the evidence
is cited in the policy module and lives in `bench/results/`.
"""

# run_tests: cpu-safe -- host arithmetic only, opens no device.

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.growth_policy import GROW_DEPTHWISE, GROW_LEAFWISE
from mojotrees.gpu_split_policy import (
    M4_DEPTHWISE_EVIDENCE_ID,
    M4_DEPTHWISE_MIN_FEATURES,
    M4_DEPTHWISE_MIN_NORMALIZED_WORK,
    M4_DEPTHWISE_MIN_ROWS,
    M4_EVIDENCE_ID,
    M4_MIN_NORMALIZED_WORK,
    SPLIT_GROW_UNSPECIFIED,
    SPLIT_POLICY_DEVICE_RESIDENT,
    SPLIT_POLICY_HOST,
    SPLIT_POLICY_VERSION,
    SPLIT_REASON_BELOW_CROSSOVER,
    SPLIT_REASON_RESIDENT_MEMORY,
    SPLIT_REASON_UNKNOWN_HARDWARE,
    SPLIT_REASON_UNSUPPORTED,
    SPLIT_REASON_VALIDATED_WORKLOAD,
    SplitSearchDecision,
    decide_split_search,
    depthwise_floor_applies,
    normalized_split_work,
    split_threshold_for,
)


def _m4(
    rows: Int,
    features: Int = 100,
    bins: Int = 255,
    leaves: Int = 31,
    supported: Bool = True,
    fits: Bool = True,
    grow: Int = SPLIT_GROW_UNSPECIFIED,
) -> SplitSearchDecision:
    return decide_split_search(
        "metal",
        "4-metal4",
        rows,
        features,
        bins,
        leaves,
        supported,
        fits,
        grow,
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


def test_policy_version_records_the_second_rule() raises:
    """The version is bumped when a rule is added, not when a comment is."""
    assert_equal(SPLIT_POLICY_VERSION, 2)


def test_depthwise_floor_is_exactly_the_measured_point() raises:
    """250,000 x 50 at 255 bins and 31 leaves is 12,500,000 normalized work.

    That shape is the one the sweep II addendum forced both paths at, so the
    floor is the measured point rather than a fraction or a multiple of it.
    If `normalized_split_work` ever changes what it measures, this equality
    breaks first and the threshold has to be re-derived rather than carried
    over.
    """
    var measured = normalized_split_work(
        M4_DEPTHWISE_MIN_ROWS, M4_DEPTHWISE_MIN_FEATURES, 255, 31
    )
    assert_equal(measured, 12_500_000.0)
    assert_equal(measured, M4_DEPTHWISE_MIN_NORMALIZED_WORK)
    assert_equal(M4_MIN_NORMALIZED_WORK, 50_000_000.0)


def test_measured_depthwise_shape_takes_the_device_search() raises:
    """The 0.70 second shape: 1.909 s automatic against 1.214 s forced.

    Under rule 1 this workload is a quarter of the threshold and takes the
    host scan, which is what the measurement showed it doing and what the
    addendum called the gate's cost. Under rule 2 it takes the device search,
    cites the depth-wise record, and sits exactly on its own threshold.
    """
    var decision = _m4(250_000, features=50, grow=GROW_DEPTHWISE)
    assert_equal(decision.selected, SPLIT_POLICY_DEVICE_RESIDENT)
    assert_equal(decision.reason, SPLIT_REASON_VALIDATED_WORKLOAD)
    assert_equal(decision.threshold, M4_DEPTHWISE_MIN_NORMALIZED_WORK)
    assert_equal(decision.evidence_id, M4_DEPTHWISE_EVIDENCE_ID)
    assert_true(decision.used_depthwise_floor())
    assert_equal(decision.margin(), 0.0)
    assert_true(decision.on_crossover_boundary())


def test_same_shape_leafwise_keeps_the_host_scan() raises:
    """Leaf-wise was 15 percent worse on the device search at this shape, so
    rule 1 is not touched and the same numbers resolve the other way."""
    var decision = _m4(250_000, features=50, grow=GROW_LEAFWISE)
    assert_equal(decision.selected, SPLIT_POLICY_HOST)
    assert_equal(decision.reason, SPLIT_REASON_BELOW_CROSSOVER)
    assert_equal(decision.threshold, M4_MIN_NORMALIZED_WORK)
    assert_equal(decision.evidence_id, M4_EVIDENCE_ID)
    assert_false(decision.used_depthwise_floor())
    assert_equal(decision.margin(), -37_500_000.0)


def test_unspecified_growth_keeps_version_one_behavior() raises:
    """A caller that does not thread the growth policy gets rule 1.

    This is every caller in the tree today, so this assertion is what says
    the change is inert until somebody passes `TreeParams.grow_policy`
    through. Unspecified resolves to the higher threshold, so an unknown
    caller can only lose a device path it was never measured to want.
    """
    var decision = _m4(250_000, features=50)
    assert_equal(decision.grow_policy, SPLIT_GROW_UNSPECIFIED)
    assert_equal(decision.selected, SPLIT_POLICY_HOST)
    assert_equal(decision.threshold, M4_MIN_NORMALIZED_WORK)
    assert_true(decision.describe().find("grow_policy=unspecified") >= 0)


def test_depthwise_floor_is_a_strict_gate_not_a_region() raises:
    """The measured point clears rule 2 by floating-point equality alone.

    One row more is fifty units of work over and takes the device search; a
    leaf budget of 30 instead of 31 is under and takes the host scan, at a
    shape still inside rule 2's measured rows and features. That is the same
    knife edge rule 1 has at the headline 1,000,000 x 50 benchmark, in the
    same direction, and it is reported rather than filed off.

    Note which way the row count moves the answer. 250,001 rows stays under
    rule 2 because the shape floor is a minimum; 249,999 rows leaves rule 2's
    scope entirely and is weighed against rule 1 instead, which the fallback
    test below asserts. The work comparison and the shape scope are separate
    gates that happen to coincide at exactly the measured shape.
    """
    var above = _m4(250_001, features=50, grow=GROW_DEPTHWISE)
    assert_equal(above.selected, SPLIT_POLICY_DEVICE_RESIDENT)
    assert_equal(above.threshold, M4_DEPTHWISE_MIN_NORMALIZED_WORK)
    assert_equal(above.margin(), 50.0)
    assert_false(above.on_crossover_boundary())

    var fewer_leaves = _m4(
        250_000, features=50, leaves=30, grow=GROW_DEPTHWISE
    )
    assert_equal(fewer_leaves.selected, SPLIT_POLICY_HOST)
    assert_equal(fewer_leaves.reason, SPLIT_REASON_BELOW_CROSSOVER)
    assert_equal(fewer_leaves.threshold, M4_DEPTHWISE_MIN_NORMALIZED_WORK)
    assert_true(fewer_leaves.margin() < 0.0)
    assert_false(fewer_leaves.on_crossover_boundary())


def test_depthwise_outside_the_measured_shape_falls_back_to_rule_one() raises:
    """Same work, different shape, and the record says nothing about it.

    12,500 rows by 1,000 features is also 12,500,000 normalized work and is a
    completely different ratio of per-launch cost to per-launch work. So is
    250,000 rows by 49 features, one feature under the measured width. Both
    are weighed against rule 1, which is the conservative answer and the same
    one they got before this rule existed.
    """
    var narrow_and_short = _m4(12_500, features=1_000, grow=GROW_DEPTHWISE)
    assert_equal(
        normalized_split_work(12_500, 1_000, 255, 31), 12_500_000.0
    )
    assert_equal(narrow_and_short.selected, SPLIT_POLICY_HOST)
    assert_equal(narrow_and_short.threshold, M4_MIN_NORMALIZED_WORK)
    assert_equal(narrow_and_short.evidence_id, M4_EVIDENCE_ID)
    assert_false(narrow_and_short.used_depthwise_floor())

    var one_feature_narrow = _m4(250_000, features=49, grow=GROW_DEPTHWISE)
    assert_equal(one_feature_narrow.selected, SPLIT_POLICY_HOST)
    assert_equal(one_feature_narrow.threshold, M4_MIN_NORMALIZED_WORK)

    var one_row_short = _m4(249_999, features=50, grow=GROW_DEPTHWISE)
    assert_equal(one_row_short.selected, SPLIT_POLICY_HOST)
    assert_equal(one_row_short.threshold, M4_MIN_NORMALIZED_WORK)
    assert_false(one_row_short.used_depthwise_floor())

    assert_false(depthwise_floor_applies(GROW_DEPTHWISE, 12_500, 1_000))
    assert_false(depthwise_floor_applies(GROW_DEPTHWISE, 250_000, 49))
    assert_false(depthwise_floor_applies(GROW_LEAFWISE, 250_000, 50))
    assert_false(
        depthwise_floor_applies(SPLIT_GROW_UNSPECIFIED, 250_000, 50)
    )
    assert_true(depthwise_floor_applies(GROW_DEPTHWISE, 250_000, 50))


def _assert_depthwise_is_at_least_as_device_selecting(
    rows: Int, features: Int
) raises:
    """One shape's worth of the superset property, both policies."""
    var leafwise = _m4(rows, features=features, grow=GROW_LEAFWISE)
    var depthwise = _m4(rows, features=features, grow=GROW_DEPTHWISE)
    assert_true(
        depthwise.threshold <= leafwise.threshold,
        "depth-wise must never be weighed against a higher bar",
    )
    if leafwise.uses_device():
        assert_true(
            depthwise.uses_device(),
            "depth-wise declined a path leaf-wise took",
        )


def test_depthwise_never_declines_what_leafwise_would_take() raises:
    """The policy awareness only ever adds device selection.

    Rule 2's floor is strictly the lower of the two and its shape scope only
    decides which threshold applies, so at every shape the depth-wise answer
    is device wherever the leaf-wise answer is. Checked over a spread of
    shapes including ones inside and outside rule 2's scope.
    """
    _assert_depthwise_is_at_least_as_device_selecting(10_000, 8)
    _assert_depthwise_is_at_least_as_device_selecting(10_000, 100)
    _assert_depthwise_is_at_least_as_device_selecting(250_000, 49)
    _assert_depthwise_is_at_least_as_device_selecting(250_000, 50)
    _assert_depthwise_is_at_least_as_device_selecting(250_000, 100)
    _assert_depthwise_is_at_least_as_device_selecting(1_000_000, 50)
    _assert_depthwise_is_at_least_as_device_selecting(5_000_000, 8)
    _assert_depthwise_is_at_least_as_device_selecting(5_000_000, 100)


def test_eligibility_still_precedes_the_depthwise_floor() raises:
    """A depth-wise fit whose semantics or memory rule out the resident scan
    is refused before any threshold is consulted, and the refusal still
    reports the policy it was made for so a trace line is not ambiguous."""
    var unsupported = _m4(
        250_000, features=50, supported=False, grow=GROW_DEPTHWISE
    )
    assert_equal(unsupported.selected, SPLIT_POLICY_HOST)
    assert_equal(unsupported.reason, SPLIT_REASON_UNSUPPORTED)
    assert_equal(unsupported.grow_policy, GROW_DEPTHWISE)
    assert_false(unsupported.weighed_workload())
    assert_equal(unsupported.margin(), 0.0)
    assert_false(unsupported.used_depthwise_floor())

    var too_wide = _m4(
        250_000, features=50, fits=False, grow=GROW_DEPTHWISE
    )
    assert_equal(too_wide.reason, SPLIT_REASON_RESIDENT_MEMORY)

    var elsewhere = decide_split_search(
        "cuda", "sm_100", 250_000, 50, 255, 31, True, True, GROW_DEPTHWISE
    )
    assert_equal(elsewhere.selected, SPLIT_POLICY_HOST)
    assert_equal(elsewhere.reason, SPLIT_REASON_UNKNOWN_HARDWARE)


def test_threshold_helper_answers_without_a_decision() raises:
    """`split_threshold_for` is the same arithmetic the decision uses, so a
    caller can ask which rule covers a shape without building one."""
    assert_equal(
        split_threshold_for(GROW_DEPTHWISE, 250_000, 50),
        M4_DEPTHWISE_MIN_NORMALIZED_WORK,
    )
    assert_equal(
        split_threshold_for(GROW_LEAFWISE, 250_000, 50),
        M4_MIN_NORMALIZED_WORK,
    )
    assert_equal(
        split_threshold_for(SPLIT_GROW_UNSPECIFIED, 5_000_000, 100),
        M4_MIN_NORMALIZED_WORK,
    )


def test_description_and_citation_identify_which_rule_ran() raises:
    """Two rules exist, so the line has to say which one decided."""
    var line = _m4(250_000, features=50, grow=GROW_DEPTHWISE).describe()
    assert_true(line.find("split_strategy=device-resident") >= 0)
    assert_true(line.find("grow_policy=depthwise") >= 0)
    assert_true(line.find(M4_DEPTHWISE_EVIDENCE_ID) >= 0)
    assert_true(line.find("boundary=exact-threshold") >= 0)

    var fallback = _m4(250_000, features=49, grow=GROW_DEPTHWISE).describe()
    assert_true(fallback.find("grow_policy=depthwise") >= 0)
    assert_true(fallback.find(M4_EVIDENCE_ID) >= 0)

    var cite = _m4(250_000, features=50, grow=GROW_DEPTHWISE).cite()
    assert_true(cite.find("sweep2_2026-08-15/RESULTS.md") >= 0)
    assert_true(cite.find("Apple M4") >= 0)
    assert_equal(
        _m4(250_000, features=50, supported=False, grow=GROW_DEPTHWISE)
        .cite(),
        String("none"),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
