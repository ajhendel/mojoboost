"""Maximum tree depth under leaf-wise growth.

`max_depth` must bound depth without turning leaf-wise growth into
level-wise growth: trees stay unbalanced, the best-gain leaf is still the
one that splits, and only leaves sitting at the limit are held back. Depth
is counted in edges from the root (root = 0), so `max_depth=1` is a stump
and `max_depth <= 0` is unlimited, matching LightGBM.

The fixture drives growth into a maximally unbalanced chain: with gradients
-2^r on a 16-row identity-binned feature, isolating the largest-magnitude
row always wins, so each split peels one row off the deep side. Unlimited
growth to 8 leaves therefore reaches depth 5, well past the depth 3 a
balanced 8-leaf tree would have. Every gradient and hessian here is an exact
power of two, so sums are exact in Float64 and these are equalities, not
tolerances.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojoboost import (
    SQUARED_ERROR,
    BinnedMatrix,
    BoosterParams,
    TreeParams,
    bin_equal_width,
    grow_tree,
    train,
)


def skewed_data() raises -> BinnedMatrix:
    # One feature, 16 rows, values 0..15 into 16 equal-width bins: binning is
    # the identity, so bin id == row id.
    var features = List[Float64](capacity=16)
    for r in range(16):
        features.append(Float64(r))
    return bin_equal_width(features, n_rows=16, n_features=1, n_bins=16)


def skewed_grad() -> List[Float64]:
    """Geometrically growing gradients: the best split always isolates the
    single largest row, which is what forces a deep, one-sided tree."""
    var grad = List[Float64](capacity=16)
    for r in range(16):
        grad.append(-(2.0**r))
    return grad^


def ones(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(1.0)
    return out^


def depth_params(num_leaves: Int, max_depth: Int) -> TreeParams:
    # min_data_in_leaf=1 so only max_depth and num_leaves bound growth.
    return TreeParams(num_leaves, 1, 1.0, 1e-3, max_depth=max_depth)


def test_unlimited_growth_is_unbalanced() raises:
    """The fixture really is lopsided: 8 leaves at depth 5, not depth 3.

    Without this the depth-limit tests below would be vacuous, since a
    balanced tree would never reach the limits they set.
    """
    var tree = grow_tree(
        skewed_data(), skewed_grad(), ones(16), depth_params(8, -1)
    )
    assert_equal(tree.n_leaves, 8)
    assert_equal(tree.depth(), 5)


def test_max_depth_one_is_a_stump() raises:
    """Depth 1 permits exactly the root split, whatever num_leaves allows."""
    var tree = grow_tree(
        skewed_data(), skewed_grad(), ones(16), depth_params(31, 1)
    )
    assert_equal(tree.n_leaves, 2)
    assert_equal(tree.depth(), 1)
    # Root split, two leaf children, and nothing else.
    assert_equal(len(tree.feature), 3)
    assert_true(tree.feature[0] >= 0)
    assert_true(tree.feature[1] < 0)
    assert_true(tree.feature[2] < 0)


def test_max_depth_bounds_each_level() raises:
    """Depth is the binding constraint here, and leaf-wise growth keeps the
    tree short of the 2^depth leaves a level-wise grower would produce."""
    var depths: List[Int] = [1, 2, 3, 4]
    var expect_leaves: List[Int] = [2, 4, 6, 7]
    for i in range(len(depths)):
        var tree = grow_tree(
            skewed_data(), skewed_grad(), ones(16),
            depth_params(8, depths[i]),
        )
        assert_equal(tree.depth(), depths[i])
        assert_equal(tree.n_leaves, expect_leaves[i])
        # Leaf-wise, so past depth 2 the tree is sparser than a full level.
        assert_true(tree.n_leaves <= 1 << depths[i])


def test_nonpositive_max_depth_means_unlimited() raises:
    """0 and every negative value mean unlimited, as in LightGBM, and match a
    limit set far above the tree's natural depth."""
    var reference = grow_tree(
        skewed_data(), skewed_grad(), ones(16), depth_params(8, 100)
    )
    assert_equal(reference.depth(), 5)

    var unlimited: List[Int] = [0, -1, -7]
    for i in range(len(unlimited)):
        var tree = grow_tree(
            skewed_data(), skewed_grad(), ones(16),
            depth_params(8, unlimited[i]),
        )
        assert_equal(tree.n_leaves, reference.n_leaves)
        assert_equal(tree.depth(), reference.depth())
        assert_equal(len(tree.feature), len(reference.feature))
        for n in range(len(tree.feature)):
            assert_equal(tree.feature[n], reference.feature[n])
            assert_equal(tree.threshold_bin[n], reference.threshold_bin[n])
            assert_equal(tree.left[n], reference.left[n])
            assert_equal(tree.right[n], reference.right[n])
            assert_equal(tree.value[n], reference.value[n])


def test_default_is_unlimited() raises:
    """The default must not silently bound depth."""
    assert_equal(TreeParams.default().max_depth, -1)
    var params = TreeParams(8, 1, 1.0, 1e-3)
    assert_equal(params.max_depth, -1)
    var tree = grow_tree(skewed_data(), skewed_grad(), ones(16), params)
    assert_equal(tree.depth(), 5)


def test_num_leaves_binds_before_max_depth() raises:
    """With a generous depth the leaf budget is what stops growth, and the
    tree is identical to the unlimited-depth one."""
    var capped = grow_tree(
        skewed_data(), skewed_grad(), ones(16), depth_params(3, 20)
    )
    var unlimited = grow_tree(
        skewed_data(), skewed_grad(), ones(16), depth_params(3, -1)
    )
    assert_equal(capped.n_leaves, 3)
    assert_equal(capped.depth(), unlimited.depth())
    assert_equal(capped.n_leaves, unlimited.n_leaves)
    for n in range(len(capped.feature)):
        assert_equal(capped.feature[n], unlimited.feature[n])
        assert_equal(capped.threshold_bin[n], unlimited.threshold_bin[n])


def test_max_depth_binds_before_num_leaves() raises:
    """The reverse: a tight depth stops growth well short of the leaf
    budget, so the two limits compose rather than one masking the other."""
    var tree = grow_tree(
        skewed_data(), skewed_grad(), ones(16), depth_params(31, 2)
    )
    assert_equal(tree.depth(), 2)
    assert_equal(tree.n_leaves, 4)


def test_max_depth_holds_across_boosting_rounds() raises:
    """Every tree in an ensemble respects the limit, and a bounded ensemble
    still fits (it is a constraint on shape, not a break in training)."""
    var data = skewed_data()
    var target = List[Float64](capacity=16)
    for r in range(16):
        target.append(1.0 if r >= 8 else 0.0)

    var params = BoosterParams(30, 0.3, depth_params(31, 2))
    var model = train(data, target, SQUARED_ERROR, params)
    assert_true(len(model.trees) > 0)
    for t in range(len(model.trees)):
        assert_true(model.trees[t].depth() <= 2)
        assert_true(model.trees[t].n_leaves <= 4)
    for r in range(16):
        assert_true(abs(model.predict_row(data, r) - target[r]) < 0.1)


def test_tree_depth_of_single_leaf_is_zero() raises:
    """A tree that never splits has depth 0, so the metric agrees with the
    root-is-depth-0 convention max_depth is checked against."""
    var uniform = ones(16)
    var tree = grow_tree(
        skewed_data(), uniform, ones(16), depth_params(31, -1)
    )
    assert_equal(tree.n_leaves, 1)
    assert_equal(tree.depth(), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
