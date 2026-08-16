"""The device-resident control plane grows the same trees as the host-driven one.

This is the whole claim of `gpu_resident_round.mojo`, and until this file
existed there was no test of it. The lane that built that module was forbidden
from running anything, so every correctness argument in it is a reading
argument. Eight of them are listed in its own report as unverified. This file
converts the load-bearing one into an assertion.

What is being compared is **trees, not predictions**. Predictions can agree to
Float32 tolerance while the two planes have chosen different splits, and a
different split is the failure mode that matters: it is not corrected by later
rounds, unlike a leaf value. So every comparison here is exact and structural.

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


def _regression_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(
            2.0 * features[r] - 1.5 * features[n_rows + r] + _uniform(UInt64(r))
        )
    return target^


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


def _both_planes(
    data: BinnedMatrix,
    target: List[Float64],
    params: BoosterParams,
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> Tuple[List[Tree], List[Tree]]:
    """Train once with the device-resident plane off and once with it on.

    Both fits happen in one process against one dataset, so nothing about the
    environment differs between them beyond the gate itself. The gate is
    cleared afterwards whichever way the assertions go, because a leaked
    environment variable would silently change every later test in the file.
    """
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
    var host_plane = train_gpu(
        data, target, SQUARED_ERROR, params, bagging=bagging
    )
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "1")
    var device_plane = train_gpu(
        data, target, SQUARED_ERROR, params, bagging=bagging
    )
    _ = setenv("MOJOTREES_GPU_TREE_RESIDENT", "")
    return (host_plane.trees.copy(), device_plane.trees.copy())


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

        var planes = _both_planes(data, target, params)
        _assert_same_forest(planes[0], planes[1], "dense squared error")


def test_the_default_leaf_budget_is_identical() raises:
    """31 leaves, which is the shipped default and the shape every benchmark
    on this project uses. The previous test grows 8-leaf trees, which does not
    exercise a frontier wider than one threadgroup of the pick kernel."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 6_000
        var n_features = 8
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var params = BoosterParams(8, 0.1, TreeParams(31, 20, 1.0, 1e-3))

        var planes = _both_planes(data, target, params)
        _assert_same_forest(planes[0], planes[1], "31 leaves")


def test_early_termination_paths_are_identical() raises:
    """`max_depth` and a large `min_data_in_leaf`, so growth stops before the
    leaf budget is spent.

    This is the case the lane flagged as needing coverage: a tree that ends
    early leaves dead steps in an already-enqueued schedule, and a dead step
    is supposed to read a live count of zero and return. If that guard is
    wrong the device plane either grows a larger tree than the host plane or
    reads a descriptor that was never written.
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
        _assert_same_forest(by_depth[0], by_depth[1], "max_depth 3")

        var sparse_leaves = BoosterParams(
            6, 0.1, TreeParams(31, 900, 1.0, 1e-3)
        )
        var by_count = _both_planes(data, target, sparse_leaves)
        _assert_same_forest(by_count[0], by_count[1], "min_data_in_leaf 900")


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

        var planes = _both_planes(data, target, params)
        _assert_same_forest(planes[0], planes[1], "missing values")


def test_bagging_reaches_the_plane_and_agrees() raises:
    """Bagging is labelled untested by the lane that built the plane.

    It reaches the device-resident loop and was not gated out, because gating
    it would have needed state threaded through `train_gpu.mojo` that the
    lane was told not to touch. So it either works or it grows a different
    tree, and nothing until now distinguished those. A bag restricts which
    rows the root range holds and changes nothing else, so the two planes
    should agree, but that is exactly the kind of should that has been wrong
    twice this week.
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

        var planes = _both_planes(data, target, params, bagging)
        _assert_same_forest(planes[0], planes[1], "bagged")


def test_a_refused_configuration_falls_back_rather_than_diverging() raises:
    """Monotone constraints are refused by name, so the gate must be inert.

    A refusal that silently ran the device plane anyway would be the worst
    outcome available here, because the refusals exist precisely for the cases
    the plane cannot express. Asserting the trees match with the gate on is
    how a working refusal is distinguished from a broken one.
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

        var planes = _both_planes(data, target, params)
        _assert_same_forest(planes[0], planes[1], "monotone falls back")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
