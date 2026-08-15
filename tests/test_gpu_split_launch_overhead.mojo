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
  3. The split-search decision is now reported rather than discarded, and the
     50,000,000.0 crossover is pinned at the exact shape our headline
     benchmark uses.

On the third: `normalized_split_work(1_000_000, 50, 255, 31)` is exactly
50,000,000.0, which is exactly `M4_MIN_NORMALIZED_WORK`, and the gate is
`work < threshold`, so that shape takes the device path by floating-point
equality and by nothing else. This file pins that arithmetic so a later
change to the comparison, to the normalization, or to the threshold cannot
move the benchmark's path in silence. It is a pin, not an endorsement: which
side of that edge is faster is a measurement, and nothing here measures
anything.

What makes that edge worth pinning is what falls off it. The host scan pays,
per node, a full histogram download, a host synchronization, and a Float64
conversion of `3 * n_features * n_bins` cells; the device-resident scan pays
none of the three. One row, one feature, one leaf, or a `feature_fraction`
of 0.99 moves a fit from one regime to the other, so each boundary case
below asserts which regime it lands in as well as which path it names. That
is also why this file is where the fourth change in this lane, the
vectorization of `histogram_from_host`, is bounded: that function is on the
host scan only, and the benchmark shape provably never calls it.

Deliberately not done here: no shape near the crossover is trained. A
1,000,000 x 50 fit is a benchmark, and this lane runs none. The boundary is
checked as host arithmetic over the pure policy, which is where the decision
is actually made, and both split paths are exercised at a size a test can
afford by naming them outright.

Skips (passing) when no accelerator is present, so the suite stays green on
CPU-only machines. The policy tests need no device and always run.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.gpu_split_policy import (
    M4_MIN_NORMALIZED_WORK,
    SPLIT_POLICY_DEVICE_RESIDENT,
    SPLIT_POLICY_HOST,
    SPLIT_REASON_BELOW_CROSSOVER,
    SPLIT_REASON_VALIDATED_WORKLOAD,
    SplitSearchDecision,
    decide_split_search,
    normalized_split_work,
)
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.train_gpu import (
    SPLIT_SEARCH_DEVICE,
    SPLIT_SEARCH_HOST,
    GpuSplitSearcherCache,
    grow_tree_gpu,
)
from mojotrees.tree import Tree, TreeParams
from support import _make_features, _uniform


def _m4(
    rows: Int,
    features: Int,
    bins: Int = 255,
    leaves: Int = 31,
) -> SplitSearchDecision:
    """The headline benchmark's hardware profile with a variable shape."""
    return decide_split_search(
        "metal", "4-metal4", rows, features, bins, leaves, True, True
    )


def _dequantizes_per_node(decision: SplitSearchDecision) -> Bool:
    """Whether this shape pays the host scan's per-node conversion.

    `GpuHistogramBuilder.histogram_from_host` turns a downloaded
    fixed-point histogram into a Float64 `Histogram`, once per node, and
    the host scan is the only path that calls it: the device-resident
    search scans the histogram where it lives and brings back a 136-byte
    record instead. So "takes the host scan" and "pays a per-node
    download, a host block, and a `3 * n_features * n_bins` conversion"
    are the same statement, and `uses_device` decides both.

    This is spelled out as a named predicate rather than left in a comment
    because it is the reason the crossover deserves the visibility this
    lane gave it: one row either way decides whether that cost is paid at
    all.
    """
    return not decision.uses_device()


def test_headline_shape_sits_exactly_on_the_crossover() raises:
    """1,000,000 x 50 at 255 bins and 31 leaves normalizes to exactly the
    threshold, and therefore takes the device path on equality alone."""
    var work = normalized_split_work(1_000_000, 50, 255, 31)
    assert_equal(work, 50_000_000.0)
    assert_equal(work, M4_MIN_NORMALIZED_WORK)

    var decision = _m4(1_000_000, 50)
    assert_equal(decision.selected, SPLIT_POLICY_DEVICE_RESIDENT)
    assert_equal(decision.reason, SPLIT_REASON_VALIDATED_WORKLOAD)
    assert_true(decision.uses_device())
    # The margin is the whole point: zero means one more subtraction in the
    # normalization, or a `<=` where the `<` is, flips the benchmark's path.
    assert_equal(decision.margin(), 0.0)
    assert_true(decision.on_crossover_boundary())
    assert_true(
        decision.describe().find("boundary=exact-threshold") >= 0
    )
    # And therefore the benchmark never runs the host scan's per-node
    # histogram download, block, and Float64 conversion. Anything done to
    # `histogram_from_host` is invisible at this shape by construction.
    assert_false(_dequantizes_per_node(decision))


def test_one_step_off_the_boundary_falls_back_to_the_host_scan() raises:
    """How wide the cliff is: one row, one feature, or one leaf either way.

    Every case below is the same benchmark with a single input moved by the
    smallest amount that input can move. Each one also asserts what falling
    off the edge costs, which is the host scan's per-node download, host
    block, and Float64 conversion; see `_dequantizes_per_node`.
    """
    # One row fewer.
    var fewer_rows = _m4(999_999, 50)
    assert_equal(fewer_rows.selected, SPLIT_POLICY_HOST)
    assert_equal(fewer_rows.reason, SPLIT_REASON_BELOW_CROSSOVER)
    assert_equal(fewer_rows.margin(), -50.0)
    assert_false(fewer_rows.on_crossover_boundary())
    assert_true(_dequantizes_per_node(fewer_rows))

    # One feature fewer, which is also what any `feature_fraction` below 1
    # produces once `_estimated_active_features` has rounded it down.
    var fewer_features = _m4(1_000_000, 49)
    assert_equal(fewer_features.selected, SPLIT_POLICY_HOST)
    assert_equal(fewer_features.reason, SPLIT_REASON_BELOW_CROSSOVER)
    assert_equal(fewer_features.margin(), -1_000_000.0)
    assert_true(_dequantizes_per_node(fewer_features))

    # One leaf fewer.
    var fewer_leaves = _m4(1_000_000, 50, 255, 30)
    assert_equal(fewer_leaves.selected, SPLIT_POLICY_HOST)
    assert_true(fewer_leaves.margin() < 0.0)
    assert_true(_dequantizes_per_node(fewer_leaves))

    # And one row more, which is the same edge from the other side.
    var more_rows = _m4(1_000_001, 50)
    assert_equal(more_rows.selected, SPLIT_POLICY_DEVICE_RESIDENT)
    assert_equal(more_rows.margin(), 50.0)
    assert_false(more_rows.on_crossover_boundary())
    assert_false(_dequantizes_per_node(more_rows))


def test_reason_survives_to_a_description_a_user_can_read() raises:
    """Every decision, including a refusal, describes itself; only a
    decision that actually weighed a workload reports a margin."""
    var below = _m4(50_000, 50)
    var line = below.describe()
    assert_true(line.find("split_strategy=host") >= 0)
    assert_true(line.find("reason=below-crossover") >= 0)
    assert_true(line.find("boundary=exact-threshold") < 0)

    var unsupported = decide_split_search(
        "metal", "4-metal4", 1_000_000, 50, 255, 31, False, True
    )
    assert_false(unsupported.weighed_workload())
    assert_equal(unsupported.margin(), 0.0)
    assert_false(unsupported.on_crossover_boundary())
    assert_true(
        unsupported.describe().find("reason=unsupported-parameters") >= 0
    )


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
