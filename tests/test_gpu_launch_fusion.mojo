"""Folding the partition's copy-back into the child build's zeroing moves no bit.

What this file is for
---------------------
`gpu_active_rows._copy_back_zero_slot_kernel` replaces two adjacent launches of
the device-resident growth step with one. On Metal every `enqueue_function` is
its own single-encoder command buffer and the queue is 64 deep, and the
per-launch enqueue cost is flat at 6 to 7 microseconds through a stream of 64
and 14 to 17 beyond it; the resident plane emits 308 command buffers between
waits at the default leaf budget, so it is past that knee for most of every
tree. This fusion takes 30 of those out per tree, which is roughly 3,000 out of
a hundred-tree fit.

That is a PRICE argument and it is the only form the queue depth is now allowed
to take. **The depth was retired as a safety criterion on 2026-08-18**
(`docs/design/SWITCH_GRID.md` section 6 item 8): a full queue blocks the host
thread that enqueues into it rather than dropping work, and this plane, at 308
buffers a tree and 2,303 at a 256-leaf budget, was measured running exactly
that way at commit 1d77414 -- `device_wait` at 0 calls, `encode` at 85.74
percent of host time -- and is the fastest arm this package ships. Nothing in
this file ever argued otherwise, and the paragraph above is left standing
because "past that knee for most of every tree" is the measured fact the
retirement rests on.

The claim under test is not that this is faster. It is that it is **exact**:
the same trees, node for node, with no tolerance anywhere. A schedule change
that moved a bit would be worth nothing however fast it was, and this
repository has been bitten twice this week by changes whose arithmetic moved
under them (`docs/NUMERICS.md` on contraction, and the raw-score divergence
that `_publish_row_ranges` exists to fix, which round one could not see).

Why the exactness is arguable and still tested
----------------------------------------------
The two folded halves write disjoint memory -- the copy-back writes the active
row buffer and reads the scratch buffer, the zeroing writes one histogram pool
slot -- and neither reads what the other writes, so there is no ordering
between them for a launch boundary to have been providing. Both read the same
`STEP_LIVE` word of the same descriptor and neither writes any descriptor, so
the fold does not cross a descriptor write, which is the one thing that would
make a later kernel's read meaningless. Both were already grid-strided over
distinct positions of their own range, so neither has a launch geometry that
reaches its answer and one grid serves both.

That is an argument, and the argument is written out at the kernel. This file
is the assertion, because arguments of exactly this shape have been wrong here
before.

The vacuity trap, which is this repository's standing failure mode
------------------------------------------------------------------
`tests/test_gpu_tree_resident.mojo` shipped a version that passed on a plane
that failed one hundred percent of the times it ran, because it never reached
the plane. Two gates stand between `train_gpu` and `grow_tree_device_resident`
and it was opening one. Everything in that file's header applies here and is
not repeated; what this file adds is a **third** gate that could be silently
shut, and a positive assertion against it.

The third gate is the fusion arm itself. A comparison of two fits that both
took the unfused path would agree perfectly and prove nothing, and nothing
about the trees would show which path either arm took. So the plane's trace
now carries `folds=`, the number of fused launches the tree actually enqueued,
taken from a counter `GpuActiveRows.enqueue_desc_histogram` increments on the
branch that issues one. It is a count of launches made, not a prediction of
launches that should have been made, which is the difference between an
instrument and a tautology. The fused arm must report exactly one fold per
growth step on every tree and the unfused arm must report zero, and both are
asserted, in both directions, before any tree is compared.
"""

from std.os import getenv, setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from std.utils.numerics import nan

from mojotrees.bagging import BaggingParams
from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.boosting import BoosterParams
from mojotrees.gpu_resident_round import (
    PARTITION_FUSION_VAR,
    partition_fusion_enabled,
)
from mojotrees.objective_registry import SQUARED_ERROR
from mojotrees.train_gpu import train_gpu
from mojotrees.tree import Tree, TreeParams
from support import _make_features, _uniform


comptime _TRACE_PATH = "./.test_gpu_launch_fusion_trace.tmp"
"""Where the plane's trace lands for the duration of one comparison.

A file rather than standard output, because every assertion here is a count or
a token and a count wants text a test can read back. Truncated before each arm
and read after it, so the two arms never share a record.
"""

comptime _PLANE_MARK = "plane=device-resident"
"""The token `grow_tree_device_resident` writes once per tree it grows, which
is what proves the plane ran at all. See `tests/test_gpu_tree_resident.mojo`
for why counting it is not optional."""


def _truncate_trace() raises:
    with open(_TRACE_PATH, "w") as handle:
        handle.write(String(""))


def _read_trace() raises -> String:
    return open(_TRACE_PATH, "r").read()


def _regression_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(
            2.0 * features[r] - 1.5 * features[n_rows + r] + _uniform(UInt64(r))
        )
    return target^


@fieldwise_init
struct FusionComparison(Copyable, Movable):
    """Two forests and the trace each arm produced.

    The traces are carried rather than asserted on inside `_both_arms`, because
    how many trees a fit grows and how many steps each tree enqueues are the
    caller's facts: a test that trains four rounds at a budget of sixteen
    leaves knows its own expected fold count and nothing else does.
    """

    var unfused: List[Tree]
    """`MOJOTREES_GPU_FUSE_PARTITION_TAIL=0`: three launches for the partition
    and two for the child build, which is the shape the plane shipped."""

    var fused: List[Tree]
    """`=1`: two and two, with the copy-back folded into the zeroing."""

    var unfused_trace: String
    var fused_trace: String


def _both_arms(
    data: BinnedMatrix,
    target: List[Float64],
    params: BoosterParams,
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> FusionComparison:
    """Train once with the fusion off and once with it on, in one process.

    Both fits happen against one dataset with one compile, so nothing about the
    environment differs between them beyond the arm. That is the same
    requirement every arm in this repository carries, and it is not a
    convenience: this machine drifts two- to threefold between time windows, so
    a rebuild between arms would put a different compile and a different
    thermal state on either side of any comparison anyone later tries to make
    with these same variables.

    Three variables are set and all three are necessary:

    - `MOJOTREES_GPU_SPLIT_STRATEGY=device`, without which the automatic policy
      sends a fixture this size to the host histogram scan and neither arm
      reaches the plane at all;
    - `MOJOTREES_GPU_TREE_RESIDENT=1`, named rather than left to the default,
      because that default has moved once already and both arms here depend on
      the plane running;
    - `MOJOTREES_GPU_FUSE_PARTITION_TAIL`, which is the arm.

    Every one is cleared afterwards whichever way the assertions go, because a
    leaked variable would silently change every later test in the file and
    would change it in the direction of passing.
    """
    _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "device")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "1")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", _TRACE_PATH)

    _truncate_trace()
    _ = setenv(PARTITION_FUSION_VAR, "0")
    var unfused = train_gpu(data, target, SQUARED_ERROR, params, bagging=bagging)
    var unfused_trace = _read_trace()

    _truncate_trace()
    _ = setenv(PARTITION_FUSION_VAR, "1")
    var fused = train_gpu(data, target, SQUARED_ERROR, params, bagging=bagging)
    var fused_trace = _read_trace()

    _ = setenv(PARTITION_FUSION_VAR, "")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", "")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
    _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "")
    return FusionComparison(
        unfused.trees.copy(), fused.trees.copy(), unfused_trace, fused_trace
    )


def _assert_both_arms_ran(
    run: FusionComparison, steps: Int, label: String
) raises:
    """The plane ran on both arms, and the arms actually differed.

    The second half is what this helper exists for. Without it a run in which
    the variable did nothing -- a misspelled name, a gate that stopped reading
    it, a default that moved -- would compare the unfused path against itself,
    agree perfectly, and pass. That is the exact failure
    `tests/test_gpu_tree_resident.mojo` shipped with, one layer down.

    `folds=` is counted at its exact expected value on both arms rather than
    merely being asserted nonzero on one, because the two-sided count is what
    rules out a fold that fired on some steps and not others: the fused arm
    folds once per growth step on every tree, dead steps included, since the
    descriptor histogram is enqueued on every step and the liveness test lives
    inside the kernel rather than around the launch.
    """
    assert_true(len(run.fused) > 0, label + ": the fit grew no trees")
    assert_equal(
        len(run.unfused), len(run.fused), label + ": tree count between arms"
    )
    assert_equal(
        run.fused_trace.count(_PLANE_MARK),
        len(run.fused),
        label
        + ": the fused arm traced "
        + String(run.fused_trace.count(_PLANE_MARK))
        + " trees for a forest of "
        + String(len(run.fused))
        + "; zero means the fit never reached the plane and this comparison"
        + " proves nothing",
    )
    assert_equal(
        run.unfused_trace.count(_PLANE_MARK),
        len(run.unfused),
        label + ": the unfused arm did not reach the plane on every tree",
    )
    # The arm engaged, exactly once per growth step, on every tree.
    assert_equal(
        run.fused_trace.count(" folds=" + String(steps) + "\n"),
        len(run.fused),
        label
        + ": the fused arm should fold once per growth step on every tree,"
        + " which is "
        + String(steps)
        + " folds; a different count means the fusion fired on some steps and"
        + " not others",
    )
    # And the other arm did not engage at all, which is what makes the
    # comparison a comparison.
    assert_equal(
        run.unfused_trace.count(" folds=0\n"),
        len(run.unfused),
        label
        + ": the unfused arm folded something, so the variable is not"
        + " selecting the arm and the two fits took the same path",
    )
    # No tree may have come home mid-growth or with tables under the budget.
    # `grow_tree_device_resident` raises on all three, so this is redundant
    # with a raise and costs nothing; it is here so that this file's account of
    # a good run is complete rather than delegated.
    assert_equal(
        run.fused_trace.count("status=running")
        + run.fused_trace.count("status=pool_full")
        + run.fused_trace.count("status=overflow"),
        0,
        label + ": a fused tree ended abnormally",
    )


def _assert_same_forest(a: List[Tree], b: List[Tree], label: String) raises:
    """Every tree, node for node, with no tolerance anywhere.

    `value` is compared as bit patterns rather than as floats, for the reason
    `tests/test_gpu_tree_resident.mojo` gives: the question is not whether the
    two arms produce similar models, it is whether they make the same
    decisions, and a decision is discrete. A tolerance here would pass a
    fusion that had quietly changed a histogram by one fixed-point unit, which
    is precisely the failure this fold could produce if the two halves were not
    as independent as the argument says.
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


def _assert_arms_agree(
    run: FusionComparison, steps: Int, label: String
) raises:
    """Both halves of what a comparison here means: the arms differed, and what
    they grew is identical. In that order, because the second is worthless
    without the first."""
    _assert_both_arms_ran(run, steps, label)
    _assert_same_forest(run.unfused, run.fused, label)


def test_the_gate_reads_the_variable_with_the_right_polarity() raises:
    """The arm is selected by an inequality against "0", and unset is on.

    No device, no fit. This is the cheapest possible statement of the thing
    every other test in this file depends on, and it is worth stating on its
    own because a gate that had stopped reading its variable would make every
    comparison below vacuous while leaving all of them green.

    The polarity is the content. `MOJOTREES_GPU_SPECULATION` is spelled as an
    equality against "1" because it trades a third more child histogram builds
    for a better launch shape and nobody has weighed the two. This one trades
    nothing: it issues one command buffer where the step used to issue two,
    storing the same values to the same addresses under the same guard. So it
    is spelled the way a default is spelled, and an unrecognized value lands on
    the default rather than on whatever a permissive parser makes of it.
    """
    var restore = getenv(PARTITION_FUSION_VAR)

    _ = setenv(PARTITION_FUSION_VAR, "")
    assert_true(
        partition_fusion_enabled(),
        "unset should select the fusion, which is the default",
    )
    _ = setenv(PARTITION_FUSION_VAR, "0")
    assert_false(
        partition_fusion_enabled(), "0 is the only value that unfolds it"
    )
    _ = setenv(PARTITION_FUSION_VAR, "1")
    assert_true(partition_fusion_enabled(), "1 selects the fusion")
    _ = setenv(PARTITION_FUSION_VAR, "yes")
    assert_true(
        partition_fusion_enabled(),
        "an unrecognized value must land on the default and not on off",
    )

    _ = setenv(PARTITION_FUSION_VAR, restore)


def test_dense_squared_error_trees_are_identical() raises:
    """The day-one shape: dense numeric, squared error, no missing.

    Small on purpose. A disagreement here is a disagreement about the
    permutation or about the zeroing, not about anything statistical, and a
    small fixture makes the first disagreeing node easy to find.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var params = BoosterParams(10, 0.1, TreeParams(8, 20, 1.0, 1e-3))

        var run = _both_arms(data, target, params)
        _assert_arms_agree(run, 7, "dense squared error")


def test_the_default_leaf_budget_is_identical() raises:
    """31 leaves, which is the shipped default and the shape the census is
    quoted at: 30 growth steps, so 30 folds a tree and 308 command buffers
    becoming 278.

    It is also the shape that spends its whole budget, so every step is live
    and every fold has real work on both halves. The previous test grows
    8-leaf trees, whose frontier never exceeds one threadgroup of the pick
    kernel and whose windows are small enough that a grid-stride bug could
    hide inside a single iteration.
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

        var run = _both_arms(data, target, params)
        _assert_arms_agree(run, 30, "31 leaves")
        assert_equal(
            run.fused_trace.count("status=budget_spent"),
            len(run.fused),
            "31 leaves: a full-budget tree did not report a spent budget",
        )


def test_dead_steps_fold_without_touching_anything() raises:
    """A tree that stops early, so most of its folds run on dead steps.

    This is the case the fusion could get wrong in the way that would be
    hardest to see. Both halves are guarded on the descriptor's `STEP_LIVE`
    word and the fused kernel tests it once instead of twice, so a dead step
    must copy nothing back and zero nothing. If the single guard were placed
    wrongly, a dead step would zero whatever slot the descriptor was left
    naming -- which belongs to a **live** leaf -- and that leaf would then be
    searched and would offer a plausible split with no rows behind it. The
    resulting tree is well formed, passes every structural invariant, and is
    wrong, which is why this is asserted rather than argued.

    `min_data_in_leaf` at 900 against 3,000 rows stops growth long before the
    31-leaf budget, so the fold count stays at one per enqueued step while the
    commit count does not. The fold count is asserted at the full 30 for
    exactly that reason: a fusion that only fired on live steps would report
    fewer and would be a different mechanism from the one described.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)

        var by_count = BoosterParams(6, 0.1, TreeParams(31, 900, 1.0, 1e-3))
        var run = _both_arms(data, target, by_count)
        _assert_arms_agree(run, 30, "min_data_in_leaf 900")

        var shallow = BoosterParams(6, 0.1, TreeParams(31, 20, 1.0, 1e-3))
        shallow.tree.max_depth = 3
        var by_depth = _both_arms(data, target, shallow)
        # SEVEN, NOT THIRTY, SINCE 2026-08-18. This expected 30, which was
        # `num_leaves - 1`, because the resident grower used to enqueue a step
        # per leaf of the BUDGET regardless of what the depth allowed. It now
        # enqueues `min(num_leaves, 1 << max_depth) - 1`, which at 31 leaves
        # and depth 3 is 7. See `gpu_resident_round.grow_tree_device_resident`,
        # where the bound carries the reasoning: a tree with max_depth = d
        # cannot commit more than `2^d - 1` splits, so the steps above that
        # were enqueued for nothing at nine command buffers each.
        #
        # The change is why this test is worth keeping rather than relaxing.
        # Its own docstring above says the fused arm folds on every step "dead
        # steps included", and the dead steps are what got deleted, so the
        # premise moved rather than the fusion breaking. A test that had
        # asserted "at least one fold" would have passed through the change
        # and told us nothing; the two-sided exact count is what caught it.
        _assert_arms_agree(by_depth, min(31, 1 << 3) - 1, "max_depth 3")
        assert_equal(
            by_depth.fused_trace.count("status=no_candidate"),
            len(by_depth.fused),
            "max_depth 3: a depth-stopped tree should run out of candidates",
        )


def test_missing_values_route_identically() raises:
    """Missing values, because the fold's copy-back half is what publishes the
    permutation the next histogram reads.

    The routing itself is decided two launches earlier and is untouched here.
    What this fixture adds is a permutation that is not a simple prefix split:
    a missing-heavy column sends a scattered seventh of the rows the other way,
    so the copy-back moves a window whose contents are interleaved rather than
    blocked. A fold that dropped or duplicated a tail element under its
    grid-stride would show as a different tree here and might not at all on a
    cleanly separable fixture.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4_000
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        # The target is computed BEFORE the missing values are injected;
        # `_regression_target` reads feature 1, and injecting first produces a
        # NaN label and a refused fit rather than an exercised routing.
        var target = _regression_target(features, n_rows)
        for r in range(n_rows):
            if r % 7 == 0:
                features[n_rows + r] = nan[DType.float64]()
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var params = BoosterParams(8, 0.1, TreeParams(16, 20, 1.0, 1e-3))

        var run = _both_arms(data, target, params)
        _assert_arms_agree(run, 15, "missing values")


def test_bagging_folds_and_agrees() raises:
    """A bagged root, which is the one supported case where the caller does
    something different per tree.

    Bagging restricts which rows the root range holds and changes nothing else
    about the loop, so the fusion should be indifferent to it. It is included
    because it is also the case where a silent fallback would be easiest to
    mistake for agreement, and because `GpuActiveRows.begin_tree` is one of the
    entry points that refuses while a copy-back is outstanding: a bagged fit
    reaches that refusal once per tree and must pass it every time.
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
        var bagging = BaggingParams(0.6, 1, 7)

        var run = _both_arms(data, target, params, bagging)
        _assert_arms_agree(run, 15, "bagged")


def test_the_speculative_pair_folds_too_and_agrees() raises:
    """Armed with the K=1 speculative prebuild, which folds a second time.

    The armed step enqueues the partition and the child build twice: once
    against `DESC_BUILD` for the real split and once against `DESC_SPEC` for
    the prebuild. Both pairs are adjacent, both read one descriptor throughout,
    and both fold, so an armed step goes from eighteen launches to sixteen and
    reports **two** folds per growth step rather than one.

    That doubling is the assertion. It is also the place the pairing contract
    is most likely to be broken by a later edit, because
    `gpu_resident_round` moves `set_descriptor_target` between the two pairs
    and that setter refuses while a copy-back is outstanding. A schedule that
    put the target change between a partition and its histogram would raise
    here rather than silently pay a copy-back against the wrong window.

    Both arms are armed, so what is being compared is still the fusion and not
    the speculation; `tests/test_gpu_speculation_build.mojo` owns the
    speculation's own exactness.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var params = BoosterParams(6, 0.1, TreeParams(16, 20, 1.0, 1e-3))

        _ = setenv("MOJOTREES_GPU_SPECULATION", "1")
        var run = _both_arms(data, target, params)
        _ = setenv("MOJOTREES_GPU_SPECULATION", "")

        # Fifteen growth steps, two pairs each.
        _assert_arms_agree(run, 30, "speculation armed")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
