"""What the GPU launch-overhead lane changed, and what it must not have.

Three edits are under test here, and they are all changes to how often the
host talks to the device rather than to what the device computes. That is
what makes them testable at all: for each one there is a second arm still in
the process, and the assertion is that the two arms agree exactly rather than
approximately.

  1. The split searcher is now built once per fit and reset per tree
     (`GpuSplitSearcherCache`) instead of being constructed per tree. The
     reference arm is the single-call `grow_tree_gpu` overload, which still
     builds a searcher for one tree and throws it away, so growing the same
     rounds both ways must produce the same trees node for node.
  2. `GpuObjectiveState.update_raw_ranges` closes a whole tree in one launch
     over a device-side range table instead of one launch per leaf. The
     reference arm is `update_raw_ranges_per_leaf`, which is the old launch
     shape, and the assertion is on the raw scores bit for bit.
  3. The split-search decision is now reported rather than discarded.

On the third, and on what used to stand here. This file pinned the
50,000,000 crossover at the exact shape our headline benchmark uses:
`normalized_split_work(1_000_000, 50, 255, 31)` is exactly 50,000,000.0 and
the gate was `work < threshold`, so that shape took the device path by
floating-point equality and nothing else. Three tests held the edge still --
the shape on it, a single row/feature/leaf either side of it, and the
description that reported the margin.

**There is no edge. It was measured on 2026-08-16 and the crossover does not
exist**: four shapes from 5.0M to 70.0M normalized work, the two arms
interleaved in one process, five repeats each, device winning all four with
disjoint ranges and winning by MORE at the smaller shapes (1.85x at 5.0M
falling to 1.29x at 70.0M). Both thresholds were withdrawn and the three
tests went with them; `tests/test_gpu_split_policy.mojo` carries the
replacement assertion, which is that no shape is ever declined for its size.

What that removes from this file specifically is worth naming, because it was
this file's stated job: the host scan's per-node histogram download, host
block, and Float64 conversion of `3 * n_features * n_bins` cells were what
falling off the edge cost, and they are what `histogram_from_host` does. The
benchmark shape provably never called it, which is how a change to that
function was bounded here. That bound is now trivial rather than argued --
no shape calls it on the automatic path, at any size -- so the vectorization
it bounded needs a different justification than this file.

Deliberately not done here: no benchmark shape is trained. A 1,000,000 x 50
fit is a benchmark, and this lane runs none. Both split paths are still
exercised, at a size a test can afford, by naming them outright -- which is
now the only way to reach the host scan on a test-sized fixture, since the
automatic policy sends every shape to the device search.

Skips (passing) when no accelerator is present, so the suite stays green on
CPU-only machines.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.train_gpu import (
    SPLIT_SEARCH_DEVICE,
    SPLIT_SEARCH_HOST,
    GpuSplitSearcherCache,
    grow_tree_gpu,
)
from mojotrees.tree import Tree, TreeParams
from support import _make_features, _uniform


def _fill_round_grad_hess(
    mut grad: List[Float64],
    mut hess: List[Float64],
    n_rows: Int,
    round_index: Int,
):
    """A deterministic gradient pair that differs from round to round.

    Trees must differ across rounds for the searcher-reuse test to mean
    anything: identical rounds would grow identical trees whether or not
    anything leaked between them.
    """
    grad.clear()
    hess.clear()
    var base = UInt64(1_000_003 * (round_index + 1))
    for r in range(n_rows):
        grad.append(2.0 * _uniform(base + UInt64(r)) - 1.0)
        hess.append(_uniform(base + UInt64(n_rows + r)) + 0.25)


def _assert_trees_identical(a: Tree, b: Tree, where: String) raises:
    assert_equal(len(a.feature), len(b.feature), where)
    assert_equal(a.n_leaves, b.n_leaves, where)
    for i in range(len(a.feature)):
        assert_equal(a.feature[i], b.feature[i], where)
        assert_equal(a.threshold_bin[i], b.threshold_bin[i], where)
        assert_equal(a.left[i], b.left[i], where)
        assert_equal(a.right[i], b.right[i], where)
        assert_equal(a.default_left[i], b.default_left[i], where)
        assert_equal(a.missing_bin[i], b.missing_bin[i], where)
        # Exact, not tolerance-based: the hoist may not perturb a value.
        assert_equal(a.value[i], b.value[i], where)
        assert_equal(a.split_gain[i], b.split_gain[i], where)
        assert_equal(a.count[i], b.count[i], where)


def _grow_rounds(
    data: BinnedMatrix,
    params: TreeParams,
    n_rounds: Int,
    split_search: Int,
    reuse_searcher: Bool,
) raises -> List[Tree]:
    """Grow `n_rounds` trees over a fresh builder, either through one
    searcher cache held across the rounds or through the single-call
    overload, which builds and discards a searcher per tree as the trainer
    did before this lane."""
    var builder = GpuHistogramBuilder(data)
    var cache = GpuSplitSearcherCache()
    var trees = List[Tree]()
    var grad = List[Float64]()
    var hess = List[Float64]()
    for i in range(n_rounds):
        _fill_round_grad_hess(grad, hess, data.n_rows, i)
        builder.upload_gradients(grad, hess)
        if reuse_searcher:
            trees.append(
                grow_tree_gpu(builder, cache, params, [], i, split_search)
            )
        else:
            trees.append(
                grow_tree_gpu(builder, params, [], i, split_search)
            )
    return trees^


def test_hoisted_searcher_grows_identical_trees_on_the_device_path() raises:
    """The searcher hoist against its own reference arm, on the device
    split search, with a different feature set per tree and per node.

    Per-tree and per-node feature draws are what a reused searcher could
    plausibly leak: the feature slots and the allow mask are the tables a
    node restages. Drawing a different subset every tree and every node is
    therefore the shape most likely to expose a stale table, which is why
    the fractions below are under 1.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 8
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var params = TreeParams(
            8,
            20,
            1.0,
            1e-3,
            feature_fraction=0.75,
            feature_fraction_bynode=0.5,
        )
        var hoisted = _grow_rounds(
            data, params, 5, SPLIT_SEARCH_DEVICE, True
        )
        var per_tree = _grow_rounds(
            data, params, 5, SPLIT_SEARCH_DEVICE, False
        )
        assert_equal(len(hoisted), len(per_tree))
        for i in range(len(hoisted)):
            _assert_trees_identical(
                hoisted[i], per_tree[i], String("device tree ", i)
            )
        # A degenerate run of single-leaf trees would satisfy the comparison
        # while testing nothing, so require that real splits happened.
        assert_true(hoisted[0].n_leaves > 1)


def test_hoisted_searcher_leaves_the_host_scan_untouched() raises:
    """The other split path, where the cache is carried but never used.

    The host scan does not construct a searcher at all, so this is a
    regression guard on the rewiring of `grow_tree_gpu` rather than on the
    cache itself: the extra parameter must not have changed which grower
    runs or what it produces.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 8
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var params = TreeParams(
            8,
            20,
            1.0,
            1e-3,
            feature_fraction=0.75,
            feature_fraction_bynode=0.5,
        )
        var cached = _grow_rounds(data, params, 5, SPLIT_SEARCH_HOST, True)
        var plain = _grow_rounds(data, params, 5, SPLIT_SEARCH_HOST, False)
        for i in range(len(cached)):
            _assert_trees_identical(
                cached[i], plain[i], String("host tree ", i)
            )
        assert_true(cached[0].n_leaves > 1)


def test_range_update_arms_agree_bit_for_bit() raises:
    """One launch over a device range table against one launch per leaf.

    Both arms run against the same leaf ranges, the same node values, and
    the same starting raw scores, on two objective states over one builder.
    Float32 addition is exact given identical operands in identical order,
    and every row belongs to exactly one range, so agreement here is
    equality and not a tolerance.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            target.append(_uniform(UInt64(7_000 + r)))

        var builder = GpuHistogramBuilder(data)
        # Four splits, so five live leaves and four emptied internal nodes.
        # The emptied ones are what both arms have to skip identically.
        builder.begin_tree()
        builder.apply_split(0, 15, 0, 1, 2)
        builder.apply_split(1, 15, 1, 3, 4)
        builder.apply_split(2, 15, 2, 5, 6)
        builder.apply_split(3, 15, 3, 7, 8)

        var values = List[Float64]()
        for node in range(9):
            values.append(0.125 * Float64(node) + 0.5)
        var learning_rate = 0.3

        var table_arm = builder.objective_state(target)
        table_arm.init_raw(builder.ctx, [0.25])
        var loop_arm = builder.objective_state(target)
        loop_arm.init_raw(builder.ctx, [0.25])

        table_arm.update_raw_ranges(
            builder.ctx, builder.rows, values, learning_rate
        )
        loop_arm.update_raw_ranges_per_leaf(
            builder.ctx, builder.rows, values, learning_rate
        )

        var got = table_arm.download_raw(builder.ctx)
        var want = loop_arm.download_raw(builder.ctx)
        assert_equal(len(got), n_rows)
        assert_equal(len(want), n_rows)
        var moved = 0
        for r in range(n_rows):
            assert_equal(got[r], want[r])
            if got[r] != 0.25:
                moved += 1
        # Without this the test would pass on two arms that both did
        # nothing. Every row is in some leaf's range, and no leaf value here
        # is zero, so every row must have moved.
        assert_equal(moved, n_rows)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
