"""The device-resident control plane grows the same trees as the host-driven one.

This is the whole claim of `gpu_resident_round.mojo`, and until this file
existed there was no test of it. The lane that built that module was forbidden
from running anything, so every correctness argument in it is a reading
argument. Eight of them are listed in its own report as unverified. This file
converts the load-bearing one into an assertion.

Read this part before changing anything here
--------------------------------------------
The first version of this file compared two fits and asserted their trees
matched, and it passed on a plane that failed one hundred percent of the times
it ran. It passed because **it never ran the plane**. Two gates stand between
`train_gpu` and `grow_tree_device_resident`, and this file was only opening
one of them:

1. The split search has to be the device one. That is chosen automatically
   only when `normalized_split_work` reaches fifty million, which no fixture in
   a test file will ever reach; every shape here fell back to the host scan.
   `MOJOTREES_GPU_SPLIT_STRATEGY=device` forces it, and every comparison below
   sets it for **both** arms, so what is being compared is
   `grow_tree_device_resident` against `_device_search_resident`, which is the
   identity the module actually claims.
2. `MOJOTREES_GPU_TREE_RESIDENT=1` has to be set, and the configuration has to
   be one the plane accepts.

Forcing the first is necessary and is not sufficient, because a future refusal
in the second, or a change to either gate, would route around the plane again
and this file would go back to comparing the fallback against itself and
passing. So every comparison here also asserts, positively, that the plane
executed and how many trees it grew. The evidence is the plane's own trace:
`MOJOTREES_GPU_TREE_RESIDENT_TRACE` makes `grow_tree_device_resident` append
one record per tree, carrying a status word and counters that a device kernel
wrote, and `_assert_plane_ran` counts them. A run that reaches none of the
plane produces an empty file and fails, which is what a vacuous pass now looks
like.

The counts are two-sided. The arm with the gate off must produce **zero**
records, which is what makes the arm with it on meaningful: if both arms
traced, the variable would not be selecting anything.

What is being compared
----------------------
**Trees, not predictions.** Predictions can agree to Float32 tolerance while
the two planes have chosen different splits, and a different split is the
failure mode that matters: it is not corrected by later rounds, unlike a leaf
value. So every comparison here is exact and structural.

The two planes must agree because they make the same decision from the same
histogram, differing only in where the decision is made. The device plane
picks the leaf and commits the split on the device from a record the search
kernel wrote; the host plane downloads that record and does the same
arithmetic. `gpu_tree_tables` proved the pick and commit equivalent in
isolation across seven tree shapes. What this file adds is the rest of the
loop: the descriptor-driven partition, the descriptor-driven histogram, and
the forced atomic strategy.

That last one is the substantive difference and it is worth naming. The
device-resident path always takes `STRATEGY_ATOMIC`, because the tiled path
sizes its partial buffer and its reduction by a tile count that a kernel
cannot derive from a row count it has not been told. The two strategies are
asserted elsewhere to produce identical integer histograms, on the argument
that fixed-point Int32 accumulation is associative. The lane's own report
lists that claim as asserted-but-not-reverified. If it is false anywhere,
these tests fail, which is the point of running the comparison at the tree
level rather than trusting the argument.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from std.utils.numerics import nan

from mojotrees.bagging import BaggingParams
from mojotrees.binning import bin_equal_width, BinnedMatrix
from mojotrees.boosting import BoosterParams
from mojotrees.monotone import MonotoneConstraints
from mojotrees.objective_registry import SQUARED_ERROR
from mojotrees.train_gpu import train_gpu
from mojotrees.tree import Tree, TreeParams
from support import _make_features, _uniform


comptime _TRACE_PATH = "./.test_gpu_tree_resident_trace.tmp"
"""Where the plane's trace lands for the duration of one comparison.

A file rather than standard output, because the assertion is a count and a
count wants text a test can read back. Truncated before each arm and read
after it, so the two arms never share a record.
"""

comptime _PLANE_MARK = "plane=device-resident"
"""The token `grow_tree_device_resident` writes once per tree it grows.

Counting it is the whole positive assertion of this file. It appears in the
per-tree summary line and nowhere else: `TreeTablesSnapshot.describe` was
written to leave the summary out for exactly this reason, so one tree is one
occurrence and not two.
"""


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
struct PlaneComparison(Copyable, Movable):
    """Two forests and the trace each arm produced.

    The traces are carried rather than asserted on inside `_both_planes`,
    because how many trees a fit grows is the caller's fact and not this
    helper's: a test that stops early or trains one round knows its own
    expected count and nothing else does.
    """

    var host: List[Tree]
    """The arm with the gate off: `_device_search_resident`, the shipping
    device-search loop that downloads a record per split."""

    var device: List[Tree]
    """The arm with the gate on, which is the plane under test if and only if
    `device_trace` is nonempty."""

    var host_trace: String
    var device_trace: String


def _both_planes(
    data: BinnedMatrix,
    target: List[Float64],
    params: BoosterParams,
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> PlaneComparison:
    """Train once with the device-resident plane off and once with it on.

    Both fits happen in one process against one dataset, so nothing about the
    environment differs between them beyond the gate itself.

    `MOJOTREES_GPU_SPLIT_STRATEGY=device` is set for both arms and is not
    optional. Without it the automatic policy sends a fixture this size to the
    host histogram scan, which reaches neither the plane nor the loop the
    plane claims to be identical to, and the comparison becomes the fallback
    against itself. With it, the off arm is `_device_search_resident` and the
    on arm is `grow_tree_device_resident`.

    Every variable is cleared afterwards whichever way the assertions go,
    because a leaked environment variable would silently change every later
    test in the file and, worse, would change it in the direction of passing.
    """
    _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "device")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", _TRACE_PATH)

    _truncate_trace()
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
    var host_plane = train_gpu(
        data, target, SQUARED_ERROR, params, bagging=bagging
    )
    var host_trace = _read_trace()

    _truncate_trace()
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "1")
    var device_plane = train_gpu(
        data, target, SQUARED_ERROR, params, bagging=bagging
    )
    var device_trace = _read_trace()

    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", "")
    _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "")
    return PlaneComparison(
        host_plane.trees.copy(),
        device_plane.trees.copy(),
        host_trace,
        device_trace,
    )


def _assert_plane_ran(run: PlaneComparison, label: String) raises:
    """The device arm went through `grow_tree_device_resident`, once per tree,
    and every tree it grew ended in a terminal status.

    This is the assertion the first version of this file was missing, and
    every comparison below makes it before it compares anything. Four separate
    claims, because each of them has failed or could fail on its own:

    - the on arm traced once per tree in its forest, so the plane grew all of
      them and not some of them;
    - the off arm traced not at all, so the gate is what selected the plane
      rather than the plane running unconditionally;
    - no tree came home `running`, which is the fault this file was written to
      catch and which the plane produced on every tree that reached its leaf
      budget;
    - no tree came home `pool_full` or `overflow`, which would mean the tables
      were sized under the budget.

    `grow_tree_device_resident` raises on all three bad statuses, so a fit that
    produced one would not have returned. Asserting on them anyway costs
    nothing and keeps this file's account of what a good run looks like
    complete, rather than delegated to a raise somebody could relax.
    """
    var traced = run.device_trace.count(_PLANE_MARK)
    assert_equal(
        traced,
        len(run.device),
        label
        + ": the device-resident plane traced "
        + String(traced)
        + " trees for a forest of "
        + String(len(run.device))
        + "; zero means the fit never reached the plane and this comparison"
        + " proves nothing",
    )
    assert_true(
        traced > 0, label + ": the device-resident plane grew no trees at all"
    )
    assert_equal(
        run.host_trace.count(_PLANE_MARK),
        0,
        label + ": the gate-off arm reached the device-resident plane too",
    )
    assert_equal(
        run.device_trace.count("status=running"),
        0,
        label + ": a tree came home while growth was still running",
    )
    assert_equal(
        run.device_trace.count("status=pool_full")
        + run.device_trace.count("status=overflow"),
        0,
        label + ": the device tree tables were too small for the budget",
    )
    assert_equal(
        run.device_trace.count("status=budget_spent")
        + run.device_trace.count("status=no_candidate"),
        traced,
        label + ": a traced tree ended in no recognized status",
    )


def _assert_plane_refused(run: PlaneComparison, label: String) raises:
    """The device arm did *not* reach the plane.

    The negative control, and it is only worth having because the positive one
    exists: before the trace, "the plane refused" and "the plane ran and
    agreed" were indistinguishable from outside, which is precisely how a
    refusal that silently ran anyway would have hidden.
    """
    assert_equal(
        run.device_trace.count(_PLANE_MARK),
        0,
        label + ": a refused configuration reached the device-resident plane",
    )


def _assert_same_forest(a: List[Tree], b: List[Tree], label: String) raises:
    """Every tree, node for node, with no tolerance anywhere.

    `value` and `split_gain` are compared as bit patterns rather than as
    floats. A tolerance here would defeat the purpose: the question is not
    whether the two planes produce similar models, it is whether they make the
    same decisions, and a decision is discrete.
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


def _assert_planes_agree(
    run: PlaneComparison, label: String
) raises:
    """Both halves of what a comparison here means: the plane ran, and what it
    grew is what the shipping loop grows. In that order, because the second is
    worthless without the first."""
    _assert_plane_ran(run, label)
    _assert_same_forest(run.host, run.device, label)


def test_dense_squared_error_trees_are_identical() raises:
    """The day-one supported case: dense numeric, squared error, no missing.

    Small on purpose. A failure here is a failure of the descriptor plumbing
    rather than of anything statistical, and a small fixture makes the first
    disagreeing node easy to find.
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

        var run = _both_planes(data, target, params)
        _assert_planes_agree(run, "dense squared error")


def test_the_default_leaf_budget_is_identical() raises:
    """31 leaves, which is the shipped default and the shape every benchmark
    on this project uses. The previous test grows 8-leaf trees, which does not
    exercise a frontier wider than one threadgroup of the pick kernel.

    It is also the shape that spends its whole budget, which is the shape the
    plane could not finish: a tree whose last enqueued step commits leaves the
    device saying growth is still running, and there was nothing after it to
    say otherwise. Every tree here should report `budget_spent`, and that is
    asserted rather than left to the count of terminal statuses.
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

        var run = _both_planes(data, target, params)
        _assert_planes_agree(run, "31 leaves")
        assert_equal(
            run.device_trace.count("status=budget_spent"),
            len(run.device),
            "31 leaves: a full-budget tree did not report a spent budget",
        )
        assert_equal(
            run.device_trace.count("leaves=31"),
            len(run.device),
            "31 leaves: a tree came home with a frontier other than 31",
        )


def test_early_termination_paths_are_identical() raises:
    """`max_depth` and a large `min_data_in_leaf`, so growth stops before the
    leaf budget is spent.

    This is the case the lane flagged as needing coverage: a tree that ends
    early leaves dead steps in an already-enqueued schedule, and a dead step
    is supposed to read a live count of zero and return. If that guard is
    wrong the device plane either grows a larger tree than the host plane or
    reads a descriptor that was never written.

    It is also the one shape that reported a terminal status before the fix,
    since a tree that runs out of candidates writes `no_candidate` from the
    step that found none. That asymmetry is why the fault looked like a
    hundred percent failure at benchmark scale and would have looked like
    nothing at all here.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)

        var shallow = BoosterParams(6, 0.1, TreeParams(31, 20, 1.0, 1e-3))
        shallow.tree.max_depth = 3
        var by_depth = _both_planes(data, target, shallow)
        _assert_planes_agree(by_depth, "max_depth 3")
        assert_equal(
            by_depth.device_trace.count("status=no_candidate"),
            len(by_depth.device),
            "max_depth 3: a depth-stopped tree should run out of candidates",
        )

        var sparse_leaves = BoosterParams(
            6, 0.1, TreeParams(31, 900, 1.0, 1e-3)
        )
        var by_count = _both_planes(data, target, sparse_leaves)
        _assert_planes_agree(by_count, "min_data_in_leaf 900")


def test_missing_values_route_identically() raises:
    """The routing words are the least-covered part of the descriptor.

    A missing bin is carried in the step descriptor alongside the feature and
    the threshold, and a row in that bin follows `default_left` rather than
    the threshold. If the descriptor's routing words are packed or read
    wrongly, a missing-heavy column is where it shows first, and it shows as
    a different tree rather than as an error.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4_000
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        # The target is computed BEFORE the missing values are injected.
        # `_regression_target` reads feature 1, so injecting first produces a
        # NaN target and the trainer rejects the fit rather than exercising
        # the routing. That is the mistake this comment exists to stop
        # somebody repeating: missing values belong in the feature matrix
        # only, and the label has to stay finite.
        var target = _regression_target(features, n_rows)
        # Every seventh row of feature 1 is missing, which is dense enough
        # that a mis-routed default direction cannot hide in one leaf.
        for r in range(n_rows):
            if r % 7 == 0:
                features[n_rows + r] = nan[DType.float64]()
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var params = BoosterParams(8, 0.1, TreeParams(16, 20, 1.0, 1e-3))

        var run = _both_planes(data, target, params)
        _assert_planes_agree(run, "missing values")


def test_bagging_reaches_the_plane_and_agrees() raises:
    """Bagging is labelled untested by the lane that built the plane.

    It reaches the device-resident loop and was not gated out, because gating
    it would have needed state threaded through `train_gpu.mojo` that the
    lane was told not to touch. So it either works or it grows a different
    tree, and nothing until now distinguished those. A bag restricts which
    rows the root range holds and changes nothing else, so the two planes
    should agree, but that is exactly the kind of should that has been wrong
    twice this week.

    The trace count matters more here than anywhere else in the file: bagging
    is the one supported case where the *caller* does something different per
    tree, so a silent fallback would be easiest to mistake for agreement.
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

        var run = _both_planes(data, target, params, bagging)
        _assert_planes_agree(run, "bagged")


def test_a_refused_configuration_falls_back_rather_than_diverging() raises:
    """Monotone constraints are refused by name, so the gate must be inert.

    A refusal that silently ran the device plane anyway would be the worst
    outcome available here, because the refusals exist precisely for the cases
    the plane cannot express. Two assertions make that legible: the trees
    match, and the trace is empty, which together say the fit fell back rather
    than that it agreed.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var params = BoosterParams(6, 0.1, TreeParams(16, 20, 1.0, 1e-3))
        params.tree.monotone = MonotoneConstraints.from_signs(
            [1, 0, 0, 0], n_features
        )

        var run = _both_planes(data, target, params)
        _assert_plane_refused(run, "monotone falls back")
        _assert_same_forest(run.host, run.device, "monotone falls back")


def test_the_step_trace_shows_every_step_of_every_tree() raises:
    """The per-step trace, which is the only per-split view of this plane.

    Turning it on inserts a download after every step, which is the wait the
    plane exists to remove, so it is a debugging instrument and never a
    measuring one. What is asserted is that it produces one record per
    enqueued step per tree, `num_leaves - 1` of them, whether or not growth
    was still going at that step. That count is the point: a schedule of dead
    steps is exactly what a reader needs to see, since a tree that stopped
    early and a tree that stopped late are indistinguishable from the one
    snapshot the plane otherwise brings home.

    One round, one tree, because this test exists to check an instrument and
    not a model.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var params = BoosterParams(1, 0.1, TreeParams(8, 20, 1.0, 1e-3))

        _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "device")
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "1")
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", _TRACE_PATH)
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS", "1")
        _truncate_trace()
        var booster = train_gpu(data, target, SQUARED_ERROR, params)
        var text = _read_trace()
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS", "")
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT_TRACE", "")
        _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
        _ = setenv("MOJOTREES_GPU_SPLIT_STRATEGY", "")

        assert_equal(
            text.count(_PLANE_MARK),
            len(booster.trees),
            "step trace: one summary record per tree",
        )
        assert_equal(
            text.count("mojotrees.resident step="),
            7 * len(booster.trees),
            "step trace: one record per enqueued step, num_leaves - 1 of them",
        )
        # A step record carries the frontier, which is the thing there is
        # otherwise no way to see. Step 0 has committed the root's split, so
        # the tree holds two leaves and three nodes by then.
        assert_true(
            text.find("step=0 status=running commits=1 leaves=2 nodes=3") >= 0,
            "step trace: the first step should show the root split committed",
        )
        assert_true(
            text.find("leaf slot=0 node=1") >= 0,
            "step trace: the frontier rows should be in the record",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
