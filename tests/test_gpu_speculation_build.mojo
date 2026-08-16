"""The K=1 speculative prebuild is consumed, and turning it on moves no bit.

What this file is defending against
-----------------------------------
A speculation that never hits enqueues exactly the same kernels as one that
always hits. So the failure mode here is not a wrong answer, it is a test
that watches the right number of launches go by and concludes a mechanism
works when the mechanism has never once fired. This project has already
shipped the shape of that failure: a test whose six fixtures all ran below
the gate they were meant to exercise, comparing the fallback against itself
and verifying nothing across six configurations.

So the load-bearing assertion in this file is a **device counter**
(`gpu_active_rows.SPEC_STAT_CONSUMED`) that `_spec_consume_kernel`
increments on the branch that suppresses the real build, and nowhere else.
It is downloaded with the tree and asserted strictly positive. A speculation
that launched perfectly and consumed nothing would fail it; a count of
launches would not.

The second assertion is the equality
-------------------------------------
`device_consumed` is asserted **equal** to the host census's `consumed`, per
tree, and that equality is a theorem rather than a coincidence. The host
census, which is a pure function of the commit log, says a commit consumes
when the leaf it splits is not one of the two nodes the previous commit
created. The device counter says a commit consumes when the split it
committed is field for field the split the previous step prebuilt. Those
coincide because a commit leaves every other leaf's slot, record, depth and
row count untouched, so a pre-existing leaf that wins the pick at step k+1
was already the best pre-existing leaf at step k.

That equality is what makes the device counter two-sided without needing a
second device fixture to force a miss. `tests/test_gpu_speculation_census.mojo`
proves the host census can return zero, can return everything, and can
separate hits from misses inside one log, over commit logs written out by
hand and needing no device at all. A device counter pinned to that census is
pinned to an instrument already shown to reach both poles.

The zero pole is here anyway
----------------------------
A two-leaf budget is one growth step, after which the leaf budget is spent,
so `_pick_runner_up_kernel` declines on its budget check and the tree
publishes nothing and consumes nothing. That fixture asserts **zeros** on
both counters while still asserting the plane ran and the tree is identical,
which is the control a positive-only file would be missing.

What identical means
--------------------
Node for node with no tolerance, `value` compared as bit patterns, following
`tests/test_gpu_tree_resident.mojo`. It is a stronger claim than "similar
models": a speculative build is bit-identical to the build it replaces
because the two accumulate the same multiset of rows into the same
fixed-point Int32 bins under the same per-round scales, and integer addition
does not care about order.

The one thing that is *not* identical, stated rather than hidden: the
active-row permutation *within* a leaf's window. A speculative partition
permutes a window whose leaf may never be split, which the unarmed plane
would have left alone. Nothing downstream indexes a row by its position in a
window -- the histogram is a function of the multiset, `update_raw_device`
broadcasts one leaf value over a window, and a later partition is
set-preserving -- so no tree, no score and no later round moves. A test that
compared row buffers would fail, and would be testing an invariant this
package does not hold.
"""

# run_tests: gpu -- opens a device and grows trees on it.

from std.os import setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.boosting import BoosterParams
from mojotrees.objective_registry import SQUARED_ERROR
from mojotrees.train_gpu import train_gpu
from mojotrees.tree import Tree, TreeParams
from support import _make_features, _uniform


comptime _TRACE_PATH = "./.test_gpu_speculation_build_trace.tmp"

comptime _CENSUS_STEM = "./.test_gpu_speculation_build_census."
"""Stem of the per-fixture census file; `_both_arms` appends its label.

One file per fixture rather than one shared file, because a fixture that
fails mid-run would otherwise have its evidence overwritten by the next
one -- and the evidence is the entire point of this file. It also lets a
reader open the census of the fixture that failed rather than of the last
fixture that ran."""

comptime _PLANE_MARK = "plane=device-resident"
"""The token `grow_tree_device_resident` writes once per tree it grows.

Counted before anything else in every fixture below. The caller falls back to
the shipping loop for any configuration this plane refuses, so two fits can
agree perfectly while the plane under test never executed."""


def _regression_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(
            2.0 * features[r] - 1.5 * features[n_rows + r] + _uniform(UInt64(r))
        )
    return target^


def _field(line: String, key: String) raises -> Int:
    """The value of one `key=value` field of a census line.

    Split on spaces and match the key **exactly**, which is not fussiness:
    the line carries both `consumed=` and `device_consumed=`, and both
    `builds=` and `device_builds=`, so a `find` would silently read the wrong
    counter and the equality assertion in this file would compare a number
    with itself.
    """
    var parts = line.split(" ")
    for i in range(len(parts)):
        var part = String(parts[i])
        var eq = part.find("=")
        if eq > 0 and part[byte=0:eq] == key:
            return Int(part[byte = eq + 1 :].strip())
    raise Error(
        String("the census line carries no field named '", key, "': ", line)
    )


@fieldwise_init
struct SpeculationRun(Copyable, Movable):
    """Two forests over one dataset, and the census each arm produced."""

    var off: List[Tree]
    """Grown with `MOJOTREES_GPU_SPECULATION` unset."""

    var on: List[Tree]
    """Grown with it set to `1`, which is the arm under test."""

    var off_trace: String
    var on_trace: String
    var off_census: String
    var on_census: String


def _write(path: String, text: String) raises:
    with open(path, "w") as handle:
        handle.write(text)


def _read(path: String) raises -> String:
    return open(path, "r").read()


def _both_arms(
    data: BinnedMatrix,
    target: List[Float64],
    params: BoosterParams,
    label: String,
) raises -> SpeculationRun:
    """Train twice in one process against one dataset, speculation off then
    on.

    The device-resident plane is forced on for both arms and the split
    strategy is pinned to `device` for both, so the only thing that differs
    between them is the speculation. Without the strategy pin the automatic
    policy sends a fixture this size to the host histogram scan, which
    reaches neither arm and turns the comparison into the fallback against
    itself -- which is exactly the failure this file's header is about.

    Every variable is cleared afterwards whichever way the assertions go. A
    leaked `MOJOTREES_GPU_SPECULATION` would arm every later test in the
    process, and would arm it in the direction of passing.
    """
    _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "device")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "1")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", _TRACE_PATH)
    var census_path = _CENSUS_STEM + label + ".tmp"
    _ = setenv("MOJOTREES_GPU_SPECULATION_CENSUS", census_path)

    _write(_TRACE_PATH, String(""))
    _write(census_path, String(""))
    _ = setenv("MOJOTREES_GPU_SPECULATION", "")
    var off = train_gpu(data, target, SQUARED_ERROR, params)
    var off_trace = _read(_TRACE_PATH)
    var off_census = _read(census_path)

    _write(_TRACE_PATH, String(""))
    _write(census_path, String(""))
    _ = setenv("MOJOTREES_GPU_SPECULATION", "1")
    var on = train_gpu(data, target, SQUARED_ERROR, params)
    var on_trace = _read(_TRACE_PATH)
    var on_census = _read(census_path)

    _ = setenv("MOJOTREES_GPU_SPECULATION", "")
    _ = setenv("MOJOTREES_GPU_SPECULATION_CENSUS", "")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", "")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
    _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "")
    return SpeculationRun(
        off.trees.copy(),
        on.trees.copy(),
        off_trace,
        on_trace,
        off_census,
        on_census,
    )


def _assert_both_arms_took_the_plane(
    run: SpeculationRun, label: String
) raises:
    """Both arms grew every tree through `grow_tree_device_resident`, and both
    arms said which speculation they were running.

    Four claims. The plane ran in both arms, once per tree, so neither arm is
    the shipping loop in disguise; every tree ended in a terminal status; the
    off arm's census lines say `spec=off` and the on arm's say `spec=on`, so
    the environment variable reached the code that reads it rather than being
    set and ignored.
    """
    assert_true(len(run.on) > 0, label + ": the fit grew no trees")
    assert_equal(
        run.on_trace.count(_PLANE_MARK),
        len(run.on),
        label
        + ": the speculation arm did not reach the device-resident plane on"
        + " every tree, so this comparison proves nothing",
    )
    assert_equal(
        run.off_trace.count(_PLANE_MARK),
        len(run.off),
        label + ": the control arm did not reach the plane either",
    )
    assert_equal(
        run.on_trace.count("status=running"),
        0,
        label + ": a tree came home while growth was still running",
    )
    assert_equal(
        run.on_trace.count("status=pool_full")
        + run.on_trace.count("status=overflow"),
        0,
        label + ": the device tree tables were too small for the budget",
    )
    assert_equal(
        run.on_census.count(" spec=on "),
        len(run.on),
        label + ": the speculation arm did not report itself armed",
    )
    assert_equal(
        run.off_census.count(" spec=off "),
        len(run.off),
        label + ": the control arm did not report itself unarmed",
    )
    assert_equal(
        run.on_census.count(" spec=off "),
        0,
        label + ": an armed tree reported itself unarmed",
    )


def _assert_same_forest(a: List[Tree], b: List[Tree], label: String) raises:
    """Every tree, node for node, with no tolerance anywhere.

    `value` is compared as bit patterns rather than as floats. A tolerance
    would defeat the purpose: the question is not whether the two arms
    produce similar models, it is whether they make the same decisions, and a
    decision is discrete.
    """
    assert_equal(len(a), len(b), label + ": tree count")
    for t in range(len(a)):
        var want = a[t].copy()
        var got = b[t].copy()
        assert_equal(got.n_leaves, want.n_leaves, label + ": n_leaves")
        assert_equal(len(got.feature), len(want.feature), label + ": n_nodes")
        for i in range(len(want.feature)):
            assert_equal(got.feature[i], want.feature[i], label + ": feature")
            assert_equal(
                got.threshold_bin[i],
                want.threshold_bin[i],
                label + ": threshold_bin",
            )
            assert_equal(got.left[i], want.left[i], label + ": left")
            assert_equal(got.right[i], want.right[i], label + ": right")
            assert_equal(
                got.value[i].to_bits(),
                want.value[i].to_bits(),
                label + ": value bits",
            )
            assert_equal(
                got.split_gain[i].to_bits(),
                want.split_gain[i].to_bits(),
                label + ": split_gain bits",
            )


def _census_lines(text: String) raises -> List[String]:
    var out = List[String]()
    var lines = text.split("\n")
    for i in range(len(lines)):
        var line = String(lines[i])
        if line.find("mojotrees.speculation ") >= 0:
            out.append(line)
    return out^


def test_a_speculative_build_is_consumed_and_the_counter_says_so() raises:
    """The load-bearing test. A prebuild is consumed, and the evidence is a
    device counter that only a consuming step can move.

    31 leaves, which is the shipped default and the shape every benchmark on
    this project uses, so the frontier is wider than one threadgroup of the
    pick kernel and the tree is deep enough that most picks are of leaves
    that predate the step before them. Eight rounds, so the assertion is over
    eight trees and not one.

    Three assertions in ascending order of what they cost to fake:

    - **`device_consumed > 0` somewhere in the fit.** A speculation that
      launched everything and hit nothing fails here, and a count of launches
      would not have.
    - **`device_consumed == consumed` on every tree.** The device counter and
      the commit-log census are independent derivations of the same quantity
      -- one from a kernel's branch, one from a host loop over thirty
      integers -- and the theorem in `gpu_resident_round.mojo` says they must
      agree. A device counter stuck at a constant fails this unless the
      census is stuck at the same constant, and
      `tests/test_gpu_speculation_census.mojo` proves the census is not.
    - **`device_builds <= builds`.** The two are not the same quantity and
      must not be asserted equal: the host census counts a build on every
      step but the first, because a commit log cannot see a step whose
      pre-existing leaves were all inadmissible, whose leaf budget was
      already spent, or whose slot pool had nothing free.
      `_pick_runner_up_kernel` declines on exactly those. The inequality is
      the census's own documented overcount, checked rather than assumed.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 6_000
        var n_features = 8
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var params = BoosterParams(8, 0.1, TreeParams(31, 20, 1.0, 1e-3))

        var run = _both_arms(data, target, params, "31_leaves")
        _assert_both_arms_took_the_plane(run, "31 leaves")

        var lines = _census_lines(run.on_census)
        assert_equal(
            len(lines),
            len(run.on),
            "31 leaves: one census line per tree the plane grew",
        )
        var total_consumed = 0
        for i in range(len(lines)):
            var line = lines[i]
            var consumed = _field(line, "consumed")
            var device_consumed = _field(line, "device_consumed")
            var builds = _field(line, "builds")
            var device_builds = _field(line, "device_builds")
            total_consumed += device_consumed
            assert_equal(
                device_consumed,
                consumed,
                "31 leaves: the device consumption counter and the commit-log"
                " census disagree, which under the invariance theorem in"
                " gpu_resident_round.mojo cannot happen and is a fault"
                " report: "
                + line,
            )
            assert_true(
                device_builds <= builds,
                "31 leaves: the device issued more speculative builds than"
                " the census's upper bound allows: " + line,
            )
            assert_true(
                device_consumed <= device_builds,
                "31 leaves: more builds were consumed than were issued: "
                + line,
            )
        assert_true(
            total_consumed > 0,
            "31 leaves: not one speculative build was consumed in the whole"
            " fit. The speculation launched and never fired, which is exactly"
            " the failure a launch count would have passed.",
        )


def test_arming_the_speculation_moves_no_bit() raises:
    """The same fit with the speculation off and on produces the same forest,
    node for node, `value` and `split_gain` as bit patterns.

    This is the claim that makes the arm safe to measure at all. It rests on
    four legs, each argued where it lives rather than here: the speculative
    partition permutes rows inside one leaf's window and inside nothing else
    and is idempotent on an already-partitioned window; accumulation is
    fixed-point Int32, so a histogram is a function of the multiset of rows
    and not of their order; the fixed-point scales are per round and a
    prebuild and its consumer are inside one tree; and under this plane's
    refusals a node's feature set is the tree's feature set.

    Eight rounds rather than one, deliberately. A divergence that only showed
    up in the *scores* -- the bug that cost this plane every tree after the
    first when `_publish_row_ranges` was missing -- is invisible in a
    one-round comparison, because round one is right by construction and
    round two is the first that reads a raw score the previous tree wrote.
    The speculative partition reorders rows inside windows that the unarmed
    plane leaves alone, so `update_raw_device`'s window-broadcast is exactly
    the consumer that would show it.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 5_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var params = BoosterParams(8, 0.1, TreeParams(16, 20, 1.0, 1e-3))

        var run = _both_arms(data, target, params, "bit_identity")
        _assert_both_arms_took_the_plane(run, "bit identity")
        _assert_same_forest(run.off, run.on, "bit identity")


def test_a_tree_that_ends_early_still_agrees_and_consumes_nothing_wrongly(
) raises:
    """`max_depth` and a large `min_data_in_leaf`, so growth stops before the
    leaf budget is spent and the schedule runs dead steps.

    A dead step is where an enqueue-blind speculation is at its worst and
    where it could most easily be wrong. It costs one runner-up launch, one
    consume launch, three partition launches, two histogram launches and one
    subtract launch that all read one word and return -- and the registration
    says so in advance, because a fit that stops early pays that on every
    remaining step and can consume none of it.

    The correctness claim is the sharper one: a dead step must publish
    nothing. `_pick_runner_up_kernel` returns on `STEP_LIVE` before it reads
    the frontier, which matters because on a dead step the frontier and the
    counters are the ones the *previous* commit left, and reducing over them
    would publish a descriptor naming a split nothing chose. If that were
    wrong, the trees would differ, which is what the identity assertion here
    is for.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var params = BoosterParams(4, 0.1, TreeParams(31, 400, 1.0, 1e-3))

        var run = _both_arms(data, target, params, "early_stop")
        _assert_both_arms_took_the_plane(run, "early stop")
        _assert_same_forest(run.off, run.on, "early stop")

        var lines = _census_lines(run.on_census)
        var saw_dead = False
        for i in range(len(lines)):
            var line = lines[i]
            if _field(line, "dead") > 0:
                saw_dead = True
            assert_equal(
                _field(line, "device_consumed"),
                _field(line, "consumed"),
                "early stop: the two instruments disagree on a tree with dead"
                " steps: " + line,
            )
        assert_true(
            saw_dead,
            "early stop: no tree ran a dead step, so this fixture exercised"
            " the case it exists for on none of its trees",
        )


def test_a_two_leaf_budget_publishes_nothing_and_consumes_nothing() raises:
    """The zero pole, which is guaranteed by arithmetic rather than by luck.

    A budget of two leaves is one growth step. After that step's commit the
    frontier holds two leaves and the budget is spent, so
    `_pick_runner_up_kernel` returns on its budget check before it reads a
    record: the next step cannot commit, so a prebuild for it is pure waste.
    Both device counters must therefore be exactly zero.

    This is the control that keeps the positive fixture honest. A counter
    that could only come back positive is not a measurement of consumption,
    and the two fixtures together show this one comes back with either
    answer. The identity assertion runs here too, because a tree grown with a
    speculation that published nothing must be the tree grown with no
    speculation at all, and that is the cheapest possible check that arming
    the arm does not perturb the plane by merely existing.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var params = BoosterParams(4, 0.1, TreeParams(2, 20, 1.0, 1e-3))

        var run = _both_arms(data, target, params, "two_leaves")
        _assert_both_arms_took_the_plane(run, "two leaves")
        _assert_same_forest(run.off, run.on, "two leaves")

        var lines = _census_lines(run.on_census)
        assert_true(len(lines) > 0, "two leaves: no census line at all")
        for i in range(len(lines)):
            var line = lines[i]
            assert_equal(
                _field(line, "device_builds"),
                0,
                "two leaves: a step published a prebuild for a step that"
                " cannot commit: " + line,
            )
            assert_equal(
                _field(line, "device_consumed"),
                0,
                "two leaves: a prebuild was consumed on a tree that issued"
                " none, which means the counter is not counting what it"
                " claims: " + line,
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
