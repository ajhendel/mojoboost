"""The K=1 speculation census counts consumption, and it can tell a hit from a
miss.

What this file is defending against
-----------------------------------
This project shipped a test whose six fixtures all ran below the gate they
were meant to exercise, so they compared the fallback against itself and
verified nothing across six configurations. The equivalent failure for a
speculation instrument is an instrument that reports the same answer whatever
happened -- always 100 percent, or always 0 -- and passes every assertion that
only checks it produced a number.

So the assertions below are two-sided by construction. A commit log in which
**every** decision consumes and a commit log in which **no** decision consumes
are both here, they differ only in the log, and the census must separate them.
A classifier that returned a constant fails at least one test in every pair.

Why a device is not needed and must not be used
-----------------------------------------------
`speculation_census` is a pure function of `TreeTablesSnapshot.commit_order`,
which is a list of node ids a device kernel wrote and the plane already
downloads once per tree. Making the census a function of the log rather than
of a device counter is what puts it here, in the CPU set, where the x86-64
half of the CI matrix runs it. It also makes it reproducible: the census is a
property of the tree, so a fit answers the same way in a fast window and a
slow one, which is not true of anything else this lane could have measured.

The node arithmetic every case below depends on
-----------------------------------------------
`_reset_tables_kernel` starts the node counter at 1 and `_pick_and_commit_
kernel` advances it by two per commit, so commit `j` splits `commit_order[j]`
and creates nodes `2j + 1` and `2j + 2`. A commit therefore consumes the
build issued during the previous step exactly when its parent is neither of
the two nodes the previous commit created. Every log below is written out
node by node against that rule rather than generated, so a reader can check
the expected counts by hand.

What this file does NOT prove, and what does
--------------------------------------------
It does not prove that a speculative build was consumed. It cannot: it opens
no device and grows no tree, and a consumption is a thing a kernel does.
What it proves is the precondition -- that the quantity gating the decision
is measured by something that can come back with either answer.

The speculative build now exists, behind `MOJOTREES_GPU_SPECULATION=1`, and
the test that proves consumption is `tests/test_gpu_speculation_build.mojo`.
It asserts on an observable only a *consuming* step can produce: a device
counter (`gpu_active_rows.SPEC_STAT_CONSUMED`) incremented on the branch of
`_spec_consume_kernel` that suppresses the real build and nowhere else,
downloaded with the tree and asserted strictly positive. Asserting that the
speculative launches ran would pass on a speculation that never once hit,
which is exactly the shape of the six-fixture failure above.

**This file is what makes that device counter two-sided.** That test also
asserts the device counter equal to `speculation_census`'s `consumed`, per
tree, which is only evidence because the census below is shown here to reach
zero, to reach everything, and to separate hits from misses inside one log.
A counter pinned to a constant instrument would prove nothing; a counter
pinned to this one cannot be constant unless this one is.

One thing the census gets wrong, which the device counter found and which
`SpeculationCensus.builds` now states: `builds` is `steps - 1`, and a commit
log cannot see the four situations in which the shipped speculation declines
to publish at all. On an early-stopping tree it reports 29 builds where the
device issued 2. The counts below are all exactly right *for the design the
census describes*, which is an enqueue-blind one; the shipped design is not
blind, and `wasted` is an upper bound rather than an estimate.
"""

# run_tests: cpu-safe -- host arithmetic only, opens no device.

from std.os import setenv
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.gpu_resident_round import (
    SPECULATION_K,
    speculation_census,
    speculation_census_sink,
)


def _log(*ids: Int) -> List[Int]:
    """A commit log written out node by node."""
    var out = List[Int]()
    for i in range(len(ids)):
        out.append(ids[i])
    return out^


def test_a_deepest_first_tree_consumes_nothing() raises:
    """Every commit splits a node the previous commit just created, which is
    the worst case for a speculation on pre-existing leaves.

    The log is the root, then its left child, then that child's left child,
    and so on: commit 1 splits node 1, which commit 0 created; commit 2 splits
    node 3, which commit 1 created. A K=1 prebuild targets a leaf that was
    live *before* the step, so on this tree it is right zero times.

    This is the pole the census must be able to reach. A hit rate that cannot
    come back as zero is not a measurement of a hit rate.
    """
    var census = speculation_census(_log(0, 1, 3, 5, 7), 5)
    assert_equal(census.commits, 5, "five commits are five commits")
    assert_equal(census.steps, 5, "the tree spent every step it was given")
    assert_equal(census.dead, 0, "no step ran past the end of growth")
    assert_equal(
        census.builds, 4, "every step but the first issues one build"
    )
    assert_equal(
        census.consumed,
        0,
        "a deepest-first tree never picks a leaf that predates the step that"
        " speculated on it",
    )
    assert_equal(
        census.wasted, 4, "every build on this tree is discarded work"
    )


def test_a_tree_that_never_returns_to_a_new_child_consumes_everything() raises:
    """The opposite pole: after the second commit, every pick is a leaf that
    predates the step before it.

    Commit 1 splits node 1 (one of commit 0's two children, so no build could
    have served it, and it is excluded rather than counted as a miss). Commit
    2 splits node 2, which commit 1 did not create -- commit 1 created 3 and
    4 -- so it consumes. Commit 3 splits node 3 while commit 2 created 5 and
    6, so it consumes. Commit 4 splits node 4 while commit 3 created 7 and 8,
    so it consumes.

    Three consumable decisions and three consumed, against four builds: the
    build issued by the last step has no following commit to consume it, which
    is a structural waste rather than a wrong guess and is why `consumed`
    tops out below `builds` even here.
    """
    var census = speculation_census(_log(0, 1, 2, 3, 4), 5)
    assert_equal(census.builds, 4, "four builds")
    assert_equal(
        census.consumed,
        3,
        "every decision a build could have served consumes it",
    )
    assert_equal(
        census.wasted,
        1,
        "the last step's build has no commit after it to consume it",
    )


def test_the_census_separates_hits_from_misses_inside_one_log() raises:
    """One tree, two decisions that consume and two that do not.

    Commit 1 splits node 1 (excluded: step 0 speculates on an empty candidate
    set). Commit 2 splits node 2 against created {3, 4}: consumed. Commit 3
    splits node 5 against created {5, 6}: missed, it is a child commit 2 just
    made. Commit 4 splits node 4 against created {7, 8}: consumed.

    A classifier that returned a constant passes the two single-pole tests
    above only by failing this one, which is the point of writing all three.
    """
    var census = speculation_census(_log(0, 1, 2, 5, 4), 5)
    assert_equal(census.consumed, 2, "two decisions consume a build")
    assert_equal(
        census.wasted, 2, "four builds, two consumed, two discarded"
    )
    assert_equal(
        census.consumed + census.wasted,
        census.builds,
        "consumed and wasted must account for every build",
    )


def test_the_second_commit_of_any_tree_can_never_consume() raises:
    """Before the root split the only live leaf is the root, which is also the
    pick, so step 0 has an empty candidate set and issues no build.

    Commit 1 therefore has nothing to consume whatever it splits, and both of
    the only two nodes it could split are children commit 0 created. The
    census excludes it rather than recording it as a miss, because a miss is
    a wrong guess and there was no guess.

    Asserted on both possible logs, since the exclusion has to be about the
    decision's position and not about which child happened to win.
    """
    for parent in [1, 2]:
        var census = speculation_census(_log(0, parent), 2)
        assert_equal(
            census.builds,
            1,
            "step 1 issues a build even though step 0 did not",
        )
        assert_equal(
            census.consumed,
            0,
            "the second commit has no build to consume",
        )
        assert_equal(
            census.wasted,
            1,
            "step 1's build is wasted: this tree has no third commit",
        )


def test_a_one_split_tree_has_no_speculation_at_all() raises:
    """A tree with a single commit, which is the degenerate case a fit hits
    when the root offers the only admissible split.

    Nothing is speculated and nothing is wasted, and the census must say so
    with zeros rather than by dividing by one of them. This is the case that
    makes `builds` a count and not a fraction: a fit that summed per-tree
    ratios would have to invent an answer here.
    """
    var census = speculation_census(_log(0), 1)
    assert_equal(census.commits, 1, "one commit")
    assert_equal(census.builds, 0, "one step issues no build")
    assert_equal(census.consumed, 0, "nothing to consume")
    assert_equal(census.wasted, 0, "nothing to waste")


def test_dead_steps_are_counted_and_every_one_of_them_is_wasted() raises:
    """A tree that runs out of admissible leaves before it runs out of budget
    still enqueues its whole schedule, because the host does not know how many
    splits there will be and asking would cost the round trip this plane
    exists to remove.

    A dead step's descriptor-aware kernels read `STEP_LIVE` and return, so a
    dead step is nearly free today. A speculative build is not free on a dead
    step: under an enqueue-blind design it issues its own partition and its
    own histogram and cannot possibly be consumed. So `dead` is reported next
    to `builds`, and the wasted count includes those steps.

    **The shipped speculation is not enqueue-blind and this count therefore
    overstates it.** `gpu_tree_tables._pick_runner_up_kernel` returns on
    `STEP_LIVE` before it reads anything, so a dead step publishes nothing
    and the launches behind it read one word and return. The nine below is
    the right answer to the question this function asks; the device issues
    two. Kept as it is, because a census over a commit log cannot answer the
    other question and pretending it could is how an upper bound becomes a
    quoted figure.

    Four commits against ten enqueued steps: six dead, nine builds, and the
    two decisions that could have consumed did.
    """
    var census = speculation_census(_log(0, 1, 2, 3), 10)
    assert_equal(census.commits, 4, "four commits")
    assert_equal(census.steps, 10, "ten steps were enqueued")
    assert_equal(census.dead, 6, "six steps ran past the end of growth")
    assert_equal(census.builds, 9, "every step but the first issues a build")
    assert_equal(census.consumed, 2, "two decisions consumed a build")
    assert_equal(
        census.wasted,
        7,
        "one live miss and six dead steps are all discarded work",
    )


def test_a_log_that_could_not_have_been_grown_is_refused() raises:
    """A census computed from a malformed log is worse than no census, so both
    ways a log can fail to describe a legal leaf-wise growth raise.

    A commit cannot split a node that does not exist yet: after commit 0 the
    tree holds nodes 0, 1 and 2, so a commit 1 naming node 5 is not a tree.
    And a log cannot be longer than the schedule that could have written it,
    since a step commits at most once.
    """
    with assert_raises():
        _ = speculation_census(_log(0, 5), 5)
    with assert_raises():
        _ = speculation_census(_log(0, 1, 2), 2)
    with assert_raises():
        _ = speculation_census(_log(0), -1)


def test_k_is_one_and_the_line_says_so() raises:
    """The census line carries the K it was taken at, and it carries no token
    that another instrument counts.

    The first is so that a result file records which speculation produced its
    numbers rather than leaving a reader to assume. The second is a real
    hazard rather than a style point: `tests/test_gpu_tree_resident.mojo`
    proves the plane executed by counting `plane=device-resident` in the
    trace, and a second per-tree line carrying that substring would double
    every one of those counts and turn a passing test into a wrong one.
    """
    assert_equal(SPECULATION_K, 1, "the shipped census is K=1")
    var line = speculation_census(_log(0, 1, 2, 5, 4), 5).trace_line()
    assert_true(line.find("k=1") >= 0, "the line names its K")
    assert_true(line.find("consumed=2") >= 0, "the line carries the hits")
    assert_true(line.find("wasted=2") >= 0, "the line carries the waste")
    assert_equal(
        line.find("plane=device-resident"),
        -1,
        "the census line must not carry the token that proves the plane ran",
    )


def test_the_census_sink_is_off_unless_it_is_named() raises:
    """Off is the empty string, and the variable is the census's own.

    Its own variable rather than a mode of the trace because the two have
    opposite costs: per-step tracing reinstates a download per split and
    cannot be on while anything is measured, while the census is a host loop
    over about thirty integers. Conflating them would have made the cheap
    instrument unusable during a measurement.
    """
    _ = setenv("MOJOTREES_GPU_SPECULATION_CENSUS", "")
    assert_equal(
        speculation_census_sink(), "", "unset must leave the census off"
    )
    _ = setenv("MOJOTREES_GPU_SPECULATION_CENSUS", "./somewhere.txt")
    assert_equal(
        speculation_census_sink(),
        "./somewhere.txt",
        "a named sink comes back verbatim, path or stream",
    )
    _ = setenv("MOJOTREES_GPU_SPECULATION_CENSUS", "")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
