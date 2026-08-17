"""Pure tests for GPU split selection, which is eligibility and nothing else.

This file used to pin two thresholds. Rule 1 was the leaf-wise crossover of
50,000,000 normalized work, installed 2026-08-14 from a single measured point
with a ~2 percent margin. Rule 2 was the depth-wise floor of 12,500,000 added
2026-08-15 from one interleaved point at 250,000 x 50. Seventeen tests held
them still: where each applied, where it deliberately did not, that rule 2
could only ever add a device selection, and that the headline benchmark shape
sat on rule 1's edge by floating-point equality.

**THE SWEEP THOSE RULES WERE OWED WAS RUN ON 2026-08-16 AND THERE IS NO
CROSSOVER.** Four shapes, `gpu-host` against `gpu-device`, interleaved in one
process, five repeats each, the box verified quiet at every shape boundary:
5.0M work 1.695 s host against 0.918 s device, 25.0M 2.494 against 1.862,
41.7M 3.041 against 2.309, 70.0M 4.041 against 3.128. Every shape resolved
with disjoint ranges, and the device margin is LARGEST at the SMALLEST shape
(1.85x falling to 1.29x) -- the signature of a per-node cost, and the exact
inverse of what a profitability gate assumes. The range spans both retired
thresholds, so neither is a boundary the sweep failed to bracket.

So the tests that pinned a threshold are gone rather than retuned: there is no
measured value to install and no evidence file to write. What replaces them is
the *opposite* assertion, which is the one that can now regress -- that no
shape, however small, is ever declined for its size. `test_no_shape_is_ever_
declined_for_being_small` is the guard against a threshold being reintroduced
from an argument, which is the thing this module exists to refuse.

The three eligibility tests survive unchanged in substance, because
eligibility is a different question from profitability: it asks whether the
device search can serve a configuration at all, and that has the same answer
at every size.

Everything here is host arithmetic. No device is opened and nothing is timed.

Scope, because it is easy to overstate: the sweep compared two GPU arms.
Nothing here says anything about the CPU/GPU crossover, which is
`device_policy.AUTO_GPU_MIN_ROWS` and is a separate gate.
"""

# run_tests: cpu-safe -- host arithmetic only, opens no device.

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.growth_policy import GROW_DEPTHWISE, GROW_LEAFWISE
from mojotrees.gpu_split_policy import (
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
    normalized_split_work,
    split_reason_name,
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


def test_no_shape_is_ever_declined_for_being_small() raises:
    """The assertion that replaces both thresholds.

    Five shapes spanning four orders of magnitude in rows, every one of them
    below the retired 50,000,000 gate and four of them below the retired
    12,500,000 one. All five must take the device search, and none may report
    `below-crossover`.

    This is deliberately the mirror image of the tests it replaces. Those
    asserted that a small shape stayed on the host, which was the behavior a
    measurement later contradicted; this asserts that no size is a reason,
    which is what the measurement established. A future edit that
    reintroduces a threshold from an argument rather than from a sweep fails
    here, by name.
    """
    var shapes = [1_000, 10_000, 50_000, 100_000, 250_000]
    for i in range(len(shapes)):
        var decision = _m4(shapes[i])
        assert_true(
            decision.uses_device(),
            String("rows=", shapes[i], " must reach the device search"),
        )
        assert_equal(decision.reason, SPLIT_REASON_VALIDATED_WORKLOAD)
        assert_false(decision.reason == SPLIT_REASON_BELOW_CROSSOVER)


def test_the_smallest_shape_is_not_the_weakest_case() raises:
    """100,000 x 50 is where the device plane won by the most (1.85x).

    Pinned as its own case because it is the shape the retired rule was most
    confident about: it normalizes to 5,000,000, one tenth of the threshold,
    so the old policy sent it to the host scan by a wide margin. It is also
    the shape the host scan lost by the widest measured margin. Both facts in
    one test, so that a reader who reintroduces a gate has to explain this
    one.
    """
    var work = normalized_split_work(100_000, 50, 255, 31)
    assert_equal(work, 5_000_000.0)
    var decision = _m4(100_000, features=50)
    assert_true(decision.uses_device())


def test_eligibility_still_gates_and_reports_its_own_reason() raises:
    """The three tests that survive, because none of them is about size."""
    var unsupported = _m4(1_000_000, supported=False)
    assert_equal(unsupported.selected, SPLIT_POLICY_HOST)
    assert_equal(unsupported.reason, SPLIT_REASON_UNSUPPORTED)
    assert_false(unsupported.uses_device())

    var too_wide = _m4(1_000_000, fits=False)
    assert_equal(too_wide.selected, SPLIT_POLICY_HOST)
    assert_equal(too_wide.reason, SPLIT_REASON_RESIDENT_MEMORY)

    # And they gate a small shape exactly as they gate a large one, which is
    # what makes them eligibility rather than profitability in disguise.
    var small_unsupported = _m4(1_000, supported=False)
    assert_equal(small_unsupported.reason, SPLIT_REASON_UNSUPPORTED)


def test_unmeasured_hardware_still_stays_on_the_host() raises:
    """`_is_observed_m4` survives the threshold it used to scope.

    It looks like the hardware scope of a deleted rule and is not. The sweep
    ran on one machine, and nothing in this repository has ever run the
    device split search on CUDA or HIP (`docs/GPU_VALIDATION.md` reads "not
    run" for both). Routing unmeasured hardware to the device arm on the
    strength of an M4 measurement would install a performance claim about a
    machine nobody owns.

    Asserted at a large shape and a small one, because if this ever became a
    size question it would have stopped being this.
    """
    var cuda = decide_split_search(
        "cuda", "sm_90", 1_000_000, 100, 255, 31, True, True
    )
    assert_equal(cuda.selected, SPLIT_POLICY_HOST)
    assert_equal(cuda.reason, SPLIT_REASON_UNKNOWN_HARDWARE)

    var small_cuda = decide_split_search(
        "cuda", "sm_90", 1_000, 100, 255, 31, True, True
    )
    assert_equal(small_cuda.reason, SPLIT_REASON_UNKNOWN_HARDWARE)


def test_growth_policy_is_reported_and_decides_nothing() raises:
    """Rule 2 selected a threshold from the growth policy. Both are gone.

    The same shape under leaf-wise, depth-wise and unspecified growth must now
    produce the same selection and the same reason, and must still report the
    policy it was given -- reporting it was always worth doing and is the half
    that survives.
    """
    var leafwise = _m4(250_000, features=50, grow=GROW_LEAFWISE)
    var depthwise = _m4(250_000, features=50, grow=GROW_DEPTHWISE)
    var unspecified = _m4(250_000, features=50)

    assert_equal(leafwise.selected, depthwise.selected)
    assert_equal(leafwise.reason, depthwise.reason)
    assert_equal(leafwise.selected, unspecified.selected)
    assert_true(leafwise.uses_device())

    assert_equal(leafwise.grow_policy, GROW_LEAFWISE)
    assert_equal(depthwise.grow_policy, GROW_DEPTHWISE)
    assert_equal(unspecified.grow_policy, SPLIT_GROW_UNSPECIFIED)

    assert_true(leafwise.describe().find("grow_policy=leafwise") >= 0)
    assert_true(depthwise.describe().find("grow_policy=depthwise") >= 0)
    assert_true(unspecified.describe().find("grow_policy=unspecified") >= 0)


def test_normalization_still_accounts_for_bins_and_leaves() raises:
    """The measure survives the threshold, as a reported shape summary."""
    var baseline = normalized_split_work(1_000_000, 100, 255, 31)
    var fewer_bins = normalized_split_work(1_000_000, 100, 63, 31)
    var fewer_leaves = normalized_split_work(1_000_000, 100, 255, 15)
    assert_equal(baseline, 100_000_000.0)
    assert_true(fewer_bins < baseline)
    assert_true(fewer_leaves < baseline)


def test_description_reports_no_threshold_and_no_evidence() raises:
    """A line that named a comparison must stop naming one.

    Both halves are asserted. The positive half is that the path, the reason
    and the shape are still there; the negative half is that `threshold=`,
    `evidence=` and `boundary=` are not, because a reader who sees any of the
    three will infer a boundary is nearby and there is no boundary.
    """
    var line = _m4(1_000_000).describe()
    assert_true(line.find("split_strategy=device-resident") >= 0)
    assert_true(line.find("reason=validated-workload") >= 0)
    assert_true(line.find("normalized_work=") >= 0)

    assert_true(line.find("threshold=") < 0)
    assert_true(line.find("evidence=") < 0)
    assert_true(line.find("boundary=") < 0)


def test_the_retired_reason_code_is_reserved_not_reused() raises:
    """`SPLIT_REASON_BELOW_CROSSOVER` keeps its number and its name.

    No decision returns it any more. It is not deleted and its number is not
    recycled, so a trace line or a serialized decision written before
    2026-08-16 reads back as what it was rather than as something else --
    the same rule this repository applies to retired policy block codes.
    """
    assert_equal(SPLIT_REASON_BELOW_CROSSOVER, 4)
    assert_equal(split_reason_name(SPLIT_REASON_BELOW_CROSSOVER), "below-crossover")

    # And nothing returns it, over a spread of shapes wide enough that the
    # retired thresholds would both have fired somewhere in it.
    var shapes = [1_000, 100_000, 250_000, 1_000_000, 5_000_000]
    for i in range(len(shapes)):
        assert_false(_m4(shapes[i]).reason == SPLIT_REASON_BELOW_CROSSOVER)


def test_policy_version_records_the_withdrawal() raises:
    """The version is bumped when a rule is added OR withdrawn.

    Version 1 was one threshold, version 2 added the depth-wise floor,
    version 3 withdrew both. A withdrawal changes what the policy decides
    exactly as much as an addition does, so it costs a version.
    """
    assert_equal(SPLIT_POLICY_VERSION, 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
