"""`grow_policy`: leaf-wise (the default) against depth-wise growth.

Depth-wise growth commits every admitted split at one depth before any
deeper one, `num_leaves` staying a hard bound and the last level admitted as
a gain-ranked prefix (src/mojotrees/growth_policy.mojo). These tests pin
the properties that make the mode well defined rather than any comparison
against a reference implementation, since none exists:

- with enough eligible splits the tree fills level by level and every leaf
  sits at the same depth;
- a partial last level takes the highest-gain candidates and leaves the
  rest as leaves one level up;
- `max_depth` and `num_leaves` both bind, whichever comes first;
- the sparse grower and the GPU growers make the same choice as the dense
  CPU grower;
- the default is leaf-wise and bit-identical to a fit that never named the
  parameter;
- the distributed prototype refuses the mode instead of ignoring it.

The main fixture is one feature over 16 identity-binned rows, hessian 1,
`lambda_l2 = 0`, so a split's gain is exactly its variance reduction
`n_L * n_R / n * (mean_L - mean_R)^2` and is invariant to shifting a
segment's gradients by a constant. The gradient is four quarters, each a
ramp of slope 1, 2, 3, 4, offset by 20 per quarter: the offsets make the
midpoint the best split of the root and of each half, and within a quarter
the midpoint split of a ramp with slope `s` gains `4 s^2` against `3 s^2`
for the off-center ones, so every node with two rows is eligible and the
four quarters' gains are strictly ordered by slope. Which quarters a partial
level admits is therefore decided by the data, not by a rounding tie.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees import (
    GROW_DEPTHWISE,
    GROW_LEAFWISE,
    SQUARED_ERROR,
    BinnedMatrix,
    BoosterParams,
    Tree,
    TreeParams,
    bin_equal_width,
    grow_policy_name,
    grow_tree,
    parse_grow_policy,
    train,
)
from mojotrees.growth_policy import (
    BUDGET_WHOLE_LEVEL,
    STOP_LEAF_BUDGET,
    STOP_MAX_DEPTH,
    GrowthSchedule,
    LeafCandidate,
)
from mojotrees.sparse import csc_from_dense, fit_bins_csc, transform_csc
from mojotrees.train_gpu import (
    SPLIT_SEARCH_DEVICE,
    SPLIT_SEARCH_HOST,
    train_gpu,
)
from mojotrees.tree_sparse import grow_tree_sparse


def identity_data() raises -> BinnedMatrix:
    """One feature, 16 rows, values 0..15 into 16 equal-width bins, so the
    bin id is the row id and a threshold names a row boundary."""
    var features = List[Float64](capacity=16)
    for r in range(16):
        features.append(Float64(r))
    return bin_equal_width(features, n_rows=16, n_features=1, n_bins=16)


def quarter_grad() -> List[Float64]:
    """Four ramps of slope 1, 2, 3, 4 on rows 0-3, 4-7, 8-11, 12-15, offset
    by 20 per quarter (negated: the leaf value is `-G/H`)."""
    var grad = List[Float64](capacity=16)
    for r in range(16):
        var q = r // 4
        grad.append(-(20.0 * Float64(q) + Float64(q + 1) * Float64(r % 4)))
    return grad^


def ones(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(1.0)
    return out^


def params(num_leaves: Int, policy: Int, max_depth: Int = -1) -> TreeParams:
    # min_data_in_leaf=1 so only the policy, max_depth, and num_leaves
    # decide the shape.
    return TreeParams(
        num_leaves, 1, 0.0, 1e-3, max_depth=max_depth, grow_policy=policy
    )


def node_depths(tree: Tree) -> List[Int]:
    """Depth of every node, in edges from the root. Children carry larger
    ids than their parent in every grower, so one forward pass suffices."""
    var depths = List[Int](capacity=len(tree.feature))
    depths.resize(len(tree.feature), 0)
    for n in range(len(tree.feature)):
        if tree.feature[n] >= 0:
            depths[tree.left[n]] = depths[n] + 1
            depths[tree.right[n]] = depths[n] + 1
    return depths^


def leaf_depths(tree: Tree) -> List[Int]:
    var depths = node_depths(tree)
    var out = List[Int]()
    for n in range(len(tree.feature)):
        if tree.feature[n] < 0:
            out.append(depths[n])
    return out^


def assert_same_tree(got: Tree, want: Tree) raises:
    assert_equal(got.n_leaves, want.n_leaves)
    assert_equal(len(got.feature), len(want.feature))
    for i in range(len(want.feature)):
        assert_equal(got.feature[i], want.feature[i])
        assert_equal(got.threshold_bin[i], want.threshold_bin[i])
        assert_equal(got.left[i], want.left[i])
        assert_equal(got.right[i], want.right[i])
        assert_true(abs(got.value[i] - want.value[i]) <= 1e-9)


# ------------------------------------------------------------ the schedule


def test_schedule_leafwise_takes_the_best_gain_ties_to_the_lower_slot() raises:
    """The leaf-wise rule every grower ran before the schedule existed:
    strict `>` over the frontier, so equal gains keep the earlier slot, and
    depth plays no part."""
    var schedule = GrowthSchedule(GROW_LEAFWISE)
    var cands: List[LeafCandidate] = [
        LeafCandidate(5, 2, 2.0, True),
        LeafCandidate(3, 1, 9.0, True),
        LeafCandidate(4, 1, 9.0, True),
        LeafCandidate(6, 2, 0.5, False),
    ]
    assert_equal(schedule.next_leaf(cands, 4, 31, -1), 1)
    cands[1] = LeafCandidate.terminal(7, 2)
    assert_equal(schedule.next_leaf(cands, 5, 31, -1), 2)
    cands[2] = LeafCandidate.terminal(9, 2)
    assert_equal(schedule.next_leaf(cands, 6, 31, -1), 0)
    cands[0] = LeafCandidate.terminal(11, 3)
    # An unfound split, and a found split of zero gain, are both terminal.
    cands[3] = LeafCandidate(6, 2, 0.0, True)
    assert_equal(schedule.next_leaf(cands, 7, 31, -1), -1)


def test_schedule_commits_a_level_in_node_order_then_the_next() raises:
    """Membership is gain ranked; order within a level is ascending node id;
    a deeper leaf waits for the shallower level to finish."""
    var schedule = GrowthSchedule(GROW_DEPTHWISE)
    # Frontier: node 5 (depth 2), node 3 (depth 1, gain 1), node 4 (depth 1,
    # gain 9), node 6 (depth 2). Level 1 is planned first, node 3 before 4
    # whatever their gains, then level 2.
    var cands: List[LeafCandidate] = [
        LeafCandidate(5, 2, 2.0, True),
        LeafCandidate(3, 1, 1.0, True),
        LeafCandidate(4, 1, 9.0, True),
        LeafCandidate(6, 2, 0.5, True),
    ]
    assert_equal(schedule.next_leaf(cands, 4, 31, -1), 1)
    assert_equal(schedule.level, 1)
    assert_equal(schedule.next_leaf(cands, 5, 31, -1), 2)
    # A grower replaces a split leaf's slot with its left child and appends
    # the right one; here the two slots become depth-2 leaves with nothing
    # to offer, and the level-2 nodes are next, node 5 first.
    cands[1] = LeafCandidate.terminal(7, 2)
    cands[2] = LeafCandidate.terminal(9, 2)
    assert_equal(schedule.next_leaf(cands, 6, 31, -1), 0)
    assert_equal(schedule.level, 2)
    assert_equal(schedule.next_leaf(cands, 7, 31, -1), 3)
    # Nothing eligible remains once every candidate is marked terminal.
    var dry: List[LeafCandidate] = [
        LeafCandidate.terminal(5, 2),
        LeafCandidate.terminal(3, 2),
        LeafCandidate.terminal(4, 2),
        LeafCandidate.terminal(6, 2),
    ]
    assert_equal(schedule.next_leaf(dry, 8, 31, -1), -1)


def test_schedule_admits_the_highest_gain_prefix_under_the_budget() raises:
    """Four candidates, room for two: the two best gains, handed out in node
    order (not gain order), and the stop reason names the budget."""
    var schedule = GrowthSchedule(GROW_DEPTHWISE)
    var cands: List[LeafCandidate] = [
        LeafCandidate(3, 2, 1.0, True),
        LeafCandidate(4, 2, 4.0, True),
        LeafCandidate(5, 2, 3.0, True),
        LeafCandidate(6, 2, 2.0, True),
    ]
    # 4 leaves now, 6 allowed: two splits.
    assert_equal(schedule.next_leaf(cands, 4, 6, -1), 1)
    assert_equal(schedule.next_leaf(cands, 5, 6, -1), 2)
    assert_equal(schedule.stop_reason, STOP_LEAF_BUDGET)


def test_schedule_whole_level_refuses_a_level_that_does_not_fit() raises:
    var schedule = GrowthSchedule(GROW_DEPTHWISE, BUDGET_WHOLE_LEVEL)
    var cands: List[LeafCandidate] = [
        LeafCandidate(3, 2, 1.0, True),
        LeafCandidate(4, 2, 4.0, True),
        LeafCandidate(5, 2, 3.0, True),
    ]
    assert_equal(schedule.next_leaf(cands, 4, 6, -1), -1)


def test_schedule_reports_max_depth() raises:
    var schedule = GrowthSchedule(GROW_DEPTHWISE)
    var cands: List[LeafCandidate] = [LeafCandidate(0, 0, 1.0, True)]
    assert_equal(schedule.next_leaf(cands, 1, 31, 1), 0)
    assert_equal(schedule.stop_reason, STOP_MAX_DEPTH)


# ---------------------------------------------------------- dense grower


def test_depthwise_fills_levels_completely() raises:
    """8 leaves out of 16 rows: three complete levels, every leaf at depth 3
    with two rows, and no leaf anywhere else."""
    var tree = grow_tree(
        identity_data(), quarter_grad(), ones(16), params(8, GROW_DEPTHWISE)
    )
    assert_equal(tree.n_leaves, 8)
    assert_equal(tree.depth(), 3)
    var depths = leaf_depths(tree)
    assert_equal(len(depths), 8)
    for i in range(len(depths)):
        assert_equal(depths[i], 3)
    for n in range(len(tree.feature)):
        if tree.feature[n] < 0:
            assert_equal(tree.count[n], 2.0)


def test_depthwise_partial_level_takes_the_steepest_quarters() raises:
    """6 leaves: two complete levels (4 quarters) and a third level with room
    for two splits. The quarters' gains rise with their slope, so the two
    admitted are the two steepest, rows 8-11 and 12-15, and the two gentle
    quarters stay leaves at depth 2."""
    var tree = grow_tree(
        identity_data(), quarter_grad(), ones(16), params(6, GROW_DEPTHWISE)
    )
    assert_equal(tree.n_leaves, 6)
    assert_equal(tree.depth(), 3)
    var depths = node_depths(tree)
    var at_two = 0
    var at_three = 0
    for n in range(len(tree.feature)):
        if tree.feature[n] < 0:
            if depths[n] == 2:
                at_two += 1
                assert_equal(tree.count[n], 4.0)
            elif depths[n] == 3:
                at_three += 1
                assert_equal(tree.count[n], 2.0)
            else:
                assert_true(False)
    assert_equal(at_two, 2)
    assert_equal(at_three, 4)
    # The depth-2 internal nodes are the split quarters; their thresholds
    # are the steep quarters' midpoints, both in the upper half of the
    # feature range.
    var split_quarters = 0
    for n in range(len(tree.feature)):
        if tree.feature[n] >= 0 and depths[n] == 2:
            split_quarters += 1
            assert_true(tree.threshold_bin[n] >= 8)
    assert_equal(split_quarters, 2)


def test_depthwise_and_leafwise_disagree_on_a_skewed_fixture() raises:
    """Gradients -2^r reward peeling the largest rows off the top, so
    leaf-wise growth chases the gain down one branch and reaches depth 5
    with 8 leaves (tests/test_max_depth.mojo). Depth-wise growth never
    splits a deeper leaf while a shallower one is eligible, so it spends the
    same budget on a different tree. Both honor num_leaves."""
    var grad = List[Float64](capacity=16)
    for r in range(16):
        grad.append(-(2.0**r))
    var leafwise = grow_tree(
        identity_data(), grad, ones(16), TreeParams(8, 1, 1.0, 1e-3)
    )
    var depthwise = grow_tree(
        identity_data(),
        grad,
        ones(16),
        TreeParams(8, 1, 1.0, 1e-3, grow_policy=GROW_DEPTHWISE),
    )
    assert_equal(leafwise.n_leaves, 8)
    assert_equal(depthwise.n_leaves, 8)
    assert_equal(leafwise.depth(), 5)
    var same = len(leafwise.feature) == len(depthwise.feature)
    if same:
        for i in range(len(leafwise.feature)):
            if (
                leafwise.feature[i] != depthwise.feature[i]
                or leafwise.threshold_bin[i] != depthwise.threshold_bin[i]
                or leafwise.left[i] != depthwise.left[i]
            ):
                same = False
                break
    assert_false(same)


def test_depthwise_max_depth_binds() raises:
    """`max_depth=2` with a generous leaf budget: two full levels, 4 leaves."""
    var tree = grow_tree(
        identity_data(),
        quarter_grad(),
        ones(16),
        params(31, GROW_DEPTHWISE, max_depth=2),
    )
    assert_equal(tree.n_leaves, 4)
    assert_equal(tree.depth(), 2)


def test_default_policy_is_leafwise_and_bit_identical() raises:
    assert_equal(TreeParams.default().grow_policy, GROW_LEAFWISE)
    var plain = TreeParams(8, 1, 0.0, 1e-3)
    assert_equal(plain.grow_policy, GROW_LEAFWISE)
    var want = grow_tree(identity_data(), quarter_grad(), ones(16), plain)
    var got = grow_tree(
        identity_data(), quarter_grad(), ones(16), params(8, GROW_LEAFWISE)
    )
    assert_same_tree(got, want)


def test_policy_names_round_trip() raises:
    assert_equal(parse_grow_policy("leafwise"), GROW_LEAFWISE)
    assert_equal(parse_grow_policy("lossguide"), GROW_LEAFWISE)
    assert_equal(parse_grow_policy("depthwise"), GROW_DEPTHWISE)
    assert_equal(parse_grow_policy("depth_wise"), GROW_DEPTHWISE)
    assert_equal(grow_policy_name(GROW_LEAFWISE), "leafwise")
    assert_equal(grow_policy_name(GROW_DEPTHWISE), "depthwise")
    var raised = False
    try:
        _ = parse_grow_policy("sideways")
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        _ = grow_tree(
            identity_data(), quarter_grad(), ones(16), params(8, 7)
        )
    except:
        raised = True
    assert_true(raised)


# ----------------------------------------------------- the other growers


def _dense_matrix(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major, mostly zero, with values either side of zero, so it is
    a sensible sparse fixture too."""
    var out = List[Float64](capacity=n_rows * n_features)
    var state = UInt64(12345)
    for _ in range(n_rows * n_features):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float64(state >> 11) * (1.0 / 9007199254740992.0)
        if u < 0.25:
            out.append(16.0 * u - 2.0)
        else:
            out.append(0.0)
    return out^


def _grads(n_rows: Int) -> List[Float64]:
    var out = List[Float64](capacity=n_rows)
    var state = UInt64(777)
    for _ in range(n_rows):
        state = state * 6364136223846793005 + 1442695040888963407
        out.append(2.0 * Float64(state >> 11) / 9007199254740992.0 - 1.0)
    return out^


def test_sparse_grower_matches_dense_under_depthwise() raises:
    var n_rows = 1500
    var n_features = 12
    var dense = _dense_matrix(n_rows, n_features)
    var csc = csc_from_dense(dense, n_rows, n_features)
    var mapper = fit_bins_csc(csc, 32)
    var sparse = transform_csc(mapper, csc)
    var binned = mapper.transform(dense, n_rows)
    var grad = _grads(n_rows)
    var hess = ones(n_rows)
    var p = TreeParams(15, 20, 1.0, 1e-3, grow_policy=GROW_DEPTHWISE)
    var want = grow_tree(binned, grad, hess, p)
    var got = grow_tree_sparse(sparse, grad, hess, p)
    assert_same_tree(got.tree, want)
    assert_equal(want.n_leaves, 15)
    # And it really is depth-wise: an internal node is never deeper than a
    # leaf that still had a split to offer would have been, which on the
    # finished tree reads as: no leaf with a split-worthy row count sits
    # more than one level above the deepest internal node's children.
    var depths = node_depths(want)
    var deepest_internal = 0
    for n in range(len(want.feature)):
        if want.feature[n] >= 0 and depths[n] > deepest_internal:
            deepest_internal = depths[n]
    # The last planned level was `deepest_internal`; every leaf shallower
    # than that had no eligible split (else it would have been admitted
    # before anything deeper), which the leaf-wise tree cannot promise.
    assert_true(deepest_internal >= 3)


def test_gpu_growers_follow_the_same_schedule() raises:
    """The device builds Float32 histograms, so leaf values are compared to a
    tolerance; the split structure of every tree must match the CPU's
    depth-wise tree exactly on a fixture with well-separated gains, under
    the host-search loop and both device-search loops."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2000
        var n_features = 6
        var features = List[Float64](capacity=n_rows * n_features)
        var state = UInt64(99)
        for _ in range(n_rows * n_features):
            state = state * 6364136223846793005 + 1442695040888963407
            features.append(Float64(state >> 11) / 9007199254740992.0)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            target.append(
                3.0 * features[r]
                - 2.0 * features[n_rows + r]
                + features[2 * n_rows + r] * features[3 * n_rows + r]
            )
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var tree_params = TreeParams(
            15, 10, 1.0, 1e-3, grow_policy=GROW_DEPTHWISE
        )
        var bp = BoosterParams(4, 0.1, tree_params^)
        var cpu = train(data, target, SQUARED_ERROR, bp)
        for strategy in [SPLIT_SEARCH_HOST, SPLIT_SEARCH_DEVICE]:
            var gpu = train_gpu(
                data, target, SQUARED_ERROR, bp, split_search=strategy
            )
            assert_equal(len(gpu.trees), len(cpu.trees))
            for t in range(len(cpu.trees)):
                ref want = cpu.trees[t]
                ref got = gpu.trees[t]
                assert_equal(got.n_leaves, want.n_leaves)
                assert_equal(len(got.feature), len(want.feature))
                for i in range(len(want.feature)):
                    assert_equal(got.feature[i], want.feature[i])
                    assert_equal(got.threshold_bin[i], want.threshold_bin[i])
                    assert_equal(got.left[i], want.left[i])
                    assert_equal(got.right[i], want.right[i])
                    assert_true(abs(got.value[i] - want.value[i]) <= 1e-3)
                var depths = leaf_depths(want)
                for i in range(len(depths)):
                    assert_true(depths[i] <= 4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
