"""Which GPU growth plane runs by default, asserted on a machine with no GPU.

Why this file is separate from `test_gpu_tree_resident.mojo`
-----------------------------------------------------------
That file proves the two planes grow identical trees, and it needs a device
to do it, so every test in it skips on a runner without an accelerator. The
gate that decides *which* of them runs is a string comparison and needs no
device at all. Leaving it in there would have put the entire content of the
2026-08-16 default flip behind an accelerator, which is to say unchecked on
the x86-64 half of this project's CI matrix and unchecked on any contributor
machine without a GPU.

So the polarity lives here, marked `cpu-safe` so `tools/run_tests.sh` puts it
in the CPU set, the same arrangement `test_gpu_tile_floor.mojo` uses for the
same reason: host arithmetic about a device belongs where the host can check
it.

What the flip was
-----------------
`gpu_resident_round.mojo` shipped opt-in behind `MOJOTREES_GPU_TREE_RESIDENT=1`.
Rule S1 in `bench/results/PROFILE_PROTOCOL.md` set three conditions for making
it the default and `bench/results/session3_2026-08-16/RESULTS.md` answers all
three: trees node-identical to the host plane (asserted, no tolerance, six
configurations), **measured** 44 percent faster at 250,000 rows and 24 percent
at 1,000,000, and **measured** 2.2x faster at 50,000 where the rule asked only
for no regression. The gate is now `!= "0"`.

The opt-out is not a courtesy. This machine drifts several-fold between time
windows, so only interleaved arms compare, which means both arms have to be
reachable from one binary without a rebuild: a rebuild between arms puts a
different compile and a different thermal state on either side of the
comparison. `=0` is that handle, and it is also where a run goes that hits a
fault in the plane, since the answer to that must not be "use the host scan".

Why the polarity is worth four assertions
-----------------------------------------
A default is exactly the thing nobody writes down and everybody assumes. The
same day this flip landed it broke `_both_planes` in the sibling file, which
had spelled its off arm as the empty string on the reasoning that unset meant
off. That reasoning was correct for a week. Asserting each value separately is
what makes the next move of this default a failing test rather than a silent
change of which plane the project ships.
"""

# run_tests: cpu-safe -- opens no device; see tools/run_tests.sh gpu_by_content.

from std.os import setenv
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.gpu_resident_round import (
    resident_round_enabled,
    resident_round_explicitly_requested,
    resident_trace_sink,
    resident_trace_steps_requested,
)


def test_the_plane_is_on_unless_it_is_switched_off() raises:
    """Unset is on, `0` is off, and anything unrecognized is on.

    The third is the one a permissive parser gets wrong. An inequality
    against "0" has to land an unrecognized value on the **default**, which
    is what `MOJOTREES_GPU_SPLIT_RESIDENT` does for the same polarity, so
    that a typo in a benchmark script does not silently select the arm
    nobody asked for and then get reported as the default's number.
    """
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
    assert_true(
        resident_round_enabled(),
        "unset must select the device-resident plane; it is the default",
    )

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "0")
    assert_false(
        resident_round_enabled(),
        "0 must force the shipping device-search loop; it is the A/B handle"
        " and the escape hatch, and it has to work from one binary",
    )

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "yes")
    assert_true(
        resident_round_enabled(),
        "an unrecognized value must land on the default, which is on",
    )

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "1")
    assert_true(
        resident_round_enabled(),
        "1 must still select the plane, because every benchmark script and"
        " every results file written before the flip spells the arm that way",
    )

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")


def test_only_an_explicit_request_counts_as_one() raises:
    """`resident_round_explicitly_requested` is not the gate, and the
    difference is whether a refusal prints.

    While the plane was opt-in, "you asked for the resident plane and did not
    get it" was worth a line on standard output, because the only way to
    reach that line was to have asked. On by default, the identical line
    would print on every GPU fit with monotone constraints, depth-wise
    growth, interaction constraints or a categorical column, none of which
    asked for anything and every one of which is falling back correctly. A
    fallback that is correct and expected is silent.

    So this predicate stays strict about "1" while the gate does not, and
    that asymmetry is deliberate rather than an oversight in the flip.
    """
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
    assert_false(
        resident_round_explicitly_requested(),
        "taking the default is not a request and must not print",
    )

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "0")
    assert_false(
        resident_round_explicitly_requested(),
        "opting out is not a request for the thing opted out of",
    )

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "yes")
    assert_false(
        resident_round_explicitly_requested(),
        "only the exact string 1 is an explicit request",
    )

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "1")
    assert_true(
        resident_round_explicitly_requested(),
        "1 is the explicit request and is what gets told about a refusal",
    )

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")


def test_the_trace_did_not_come_on_with_the_plane() raises:
    """The flip moved one variable, and this checks it did not move two.

    The trace costs a file open per tree, and its step form costs a download
    per step, which is precisely the per-split host wait this plane exists to
    remove. It is a debugging instrument and never a measuring one, so "off
    unless asked for" is a precondition of every measurement this repository
    takes. It is a separate variable from the gate exactly so that turning
    the plane on cannot turn the instrument on, and this asserts that the two
    stayed separate.

    Both directions, because an assertion that a predicate is false is worth
    nothing without an assertion that it can be true.
    """
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", "")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS", "")
    assert_equal(
        resident_trace_sink(),
        String(""),
        "the trace sink must be empty unless a sink was named",
    )
    assert_false(
        resident_trace_steps_requested(),
        "per-step tracing must be off unless it was asked for",
    )

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", "1")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS", "1")
    assert_equal(resident_trace_sink(), String("1"))
    assert_true(resident_trace_steps_requested())

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", "")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS", "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
