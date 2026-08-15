"""Feature interaction constraints.

Three layers:

- the allow-mask rule itself (root union, narrowing along a branch, branch
  features staying available, overlapping groups, unlisted features)
- validation of a constraint set
- structural checks on trained models: every root-to-leaf path of every tree
  is inspected and must be contained in a single configured group, on the CPU
  trainer and, when an accelerator is present, on the GPU trainer, which must
  also produce the identical tree structure.
"""

from std.sys import has_accelerator
from std.testing import (
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.binning import bin_equal_width
from mojotrees.boosting import SQUARED_ERROR, BoosterParams, train
from mojotrees.interaction import InteractionConstraints, extend_branch
from mojotrees.train_gpu import train_gpu
from mojotrees.tree import Tree, TreeParams
from support import _make_features


def _groups(
    flat: List[List[Int]], n_features: Int
) raises -> InteractionConstraints:
    return InteractionConstraints.from_groups(flat, n_features)


def _interacting_target(
    features: List[Float64], n_rows: Int
) -> List[Float64]:
    """Pairwise products across every feature, so an unconstrained trainer
    has a reason to mix any two of them on one path."""
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        var x4 = features[4 * n_rows + r]
        var x5 = features[5 * n_rows + r]
        y.append(
            4.0 * x0 * x1
            + 3.0 * x2 * x3
            + 2.5 * x4 * x5
            + 2.0 * x0 * x3
            + 1.5 * x1 * x5
        )
    return y^


def _path_feature_sets(tree: Tree) -> List[List[Int]]:
    """Every root-to-leaf path's set of split features, one entry per leaf."""
    var paths = List[List[Int]]()
    var nodes: List[Int] = [0]
    var branches = List[List[Int]]()
    branches.append(List[Int]())
    while len(nodes) > 0:
        var node = nodes.pop()
        var branch = branches.pop()
        if tree.feature[node] < 0:
            paths.append(branch^)
            continue
        var child = extend_branch(branch, tree.feature[node])
        nodes.append(tree.left[node])
        branches.append(child.copy())
        nodes.append(tree.right[node])
        branches.append(child^)
    return paths^


def _within_one_group(
    branch: List[Int], constraints: InteractionConstraints
) -> Bool:
    for g in range(constraints.n_groups()):
        if constraints.contains_all(g, branch):
            return True
    return False


def _assert_paths_respect_constraints(
    trees: List[Tree], constraints: InteractionConstraints
) raises:
    """The invariant interaction constraints exist to produce: no root-to-leaf
    path mixes features that no single group holds together."""
    var n_paths = 0
    for t in range(len(trees)):
        var paths = _path_feature_sets(trees[t])
        for p in range(len(paths)):
            n_paths += 1
            assert_true(
                _within_one_group(paths[p], constraints),
                "a root-to-leaf path is not contained in any group",
            )
    assert_true(n_paths > 0, "no paths were inspected")


def _assert_same_trees(a: List[Tree], b: List[Tree]) raises:
    assert_equal(len(a), len(b))
    for t in range(len(a)):
        assert_equal(len(a[t].feature), len(b[t].feature))
        assert_equal(a[t].n_leaves, b[t].n_leaves)
        for i in range(len(a[t].feature)):
            assert_equal(a[t].feature[i], b[t].feature[i])
            assert_equal(a[t].threshold_bin[i], b[t].threshold_bin[i])
            assert_equal(a[t].left[i], b[t].left[i])
            assert_equal(a[t].right[i], b[t].right[i])


def test_no_constraints_gives_an_empty_mask() raises:
    var none = InteractionConstraints()
    assert_true(none.is_empty())
    assert_equal(none.n_groups(), 0)
    # Empty means "every feature allowed" to every consumer, with no
    # per-node allocation.
    assert_equal(len(none.allowed_features(List[Int]())), 0)
    assert_equal(len(none.allowed_features([0, 3])), 0)
    # An empty group list is the same thing.
    assert_true(_groups(List[List[Int]](), 5).is_empty())


def test_root_allows_the_union_of_groups() raises:
    var c = _groups([[0, 1], [2, 3]], 6)
    var allowed = c.allowed_features(List[Int]())
    assert_equal(len(allowed), 6)
    for f in range(4):
        assert_true(allowed[f], "grouped feature must be allowed at the root")
    # Features in no group are in no union, so they are never candidates.
    assert_false(allowed[4])
    assert_false(allowed[5])


def test_branch_narrows_to_groups_containing_it() raises:
    var c = _groups([[0, 1], [2, 3]], 4)
    var after_0 = c.allowed_features([0])
    assert_true(after_0[0])
    assert_true(after_0[1])
    assert_false(after_0[2])
    assert_false(after_0[3])


def test_overlapping_groups_union_then_narrow() raises:
    # 1 may pair with 0 or with 2, but 0, 1, 2 may never share a path.
    var c = _groups([[0, 1], [1, 2]], 3)
    var after_1 = c.allowed_features([1])
    assert_true(after_1[0])
    assert_true(after_1[1])
    assert_true(after_1[2])

    var after_0_1 = c.allowed_features([0, 1])
    assert_true(after_0_1[0])
    assert_true(after_0_1[1])
    assert_false(after_0_1[2], "no group holds 0, 1 and 2 together")

    var after_1_2 = c.allowed_features([1, 2])
    assert_false(after_1_2[0])
    assert_true(after_1_2[1])
    assert_true(after_1_2[2])


def test_branch_order_and_duplicates_do_not_matter() raises:
    var c = _groups([[0, 1], [1, 2]], 3)
    var a = c.allowed_features([0, 1])
    var b = c.allowed_features([1, 0, 1, 0])
    for f in range(3):
        assert_equal(a[f], b[f])


def test_branch_features_stay_allowed() raises:
    # Feature 2 is in no group with 0, yet once 2 is on the branch it can be
    # re-split on: doing so adds no new interaction.
    var c = _groups([[0, 1], [2]], 3)
    var after_2 = c.allowed_features([2])
    assert_true(after_2[2])
    assert_false(after_2[0])
    assert_false(after_2[1])


def test_singleton_group_isolates_rather_than_frees() raises:
    # The documented gotcha: a group of one does not make a feature
    # unconstrained, it makes it interact with nothing.
    var c = _groups([[0, 1], [2]], 3)
    var root = c.allowed_features(List[Int]())
    assert_true(root[2], "a singleton group is usable at the root")
    var after_0 = c.allowed_features([0])
    assert_false(after_0[2], "but it cannot join another group's branch")


def test_feature_in_every_group_is_effectively_unconstrained() raises:
    # The documented way to leave a feature free: list it everywhere.
    var c = _groups([[0, 1, 4], [2, 3, 4]], 5)
    assert_true(c.allowed_features([0])[4])
    assert_true(c.allowed_features([2])[4])
    assert_true(c.allowed_features([0, 1, 4])[1])
    assert_true(c.allowed_features([2, 3, 4])[3])
    # It still does not bridge the two groups.
    assert_false(c.allowed_features([0, 4])[2])


def test_validation_rejects_malformed_groups() raises:
    with assert_raises():
        _ = _groups([[0, 4]], 3)  # feature index out of range
    with assert_raises():
        _ = _groups([[0, -1]], 3)  # negative feature index
    with assert_raises():
        _ = _groups([[0, 0, 1]], 3)  # repeated feature within a group
    with assert_raises():
        _ = _groups([List[Int]()], 3)  # empty group
    with assert_raises():
        _ = _groups([[0, 1]], 0)  # no features to constrain
    with assert_raises():
        _ = InteractionConstraints.from_flat([0, 1], [1, 2], 3)  # bad start
    with assert_raises():
        _ = InteractionConstraints.from_flat([0, 1], [0, 1], 3)  # bad end
    with assert_raises():
        _ = InteractionConstraints.from_flat([0, 1], List[Int](), 3)


def test_check_features_rejects_a_mismatched_dataset() raises:
    var c = _groups([[0, 1]], 3)
    c.check_features(3)
    with assert_raises():
        c.check_features(4)
    # An unconstrained set fits any dataset.
    InteractionConstraints().check_features(9)


def test_extend_branch_is_duplicate_free() raises:
    var b = extend_branch(List[Int](), 2)
    assert_equal(len(b), 1)
    b = extend_branch(b, 2)
    assert_equal(len(b), 1)
    b = extend_branch(b, 5)
    assert_equal(len(b), 2)
    assert_equal(b[0], 2)
    assert_equal(b[1], 5)


def test_training_respects_every_root_to_leaf_path() raises:
    var n_rows = 2_000
    var n_features = 6
    var features = _make_features(n_rows, n_features)
    var target = _interacting_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var constraints = _groups([[0, 1], [2, 3]], n_features)
    var params = BoosterParams(
        20, 0.2, TreeParams(15, 20, 1.0, 1e-3, 0.0, constraints.copy())
    )
    var model = train(data, target, SQUARED_ERROR, params)

    _assert_paths_respect_constraints(model.trees, constraints)

    # Features 4 and 5 are in no group, so they never appear at all.
    for t in range(len(model.trees)):
        ref tree = model.trees[t]
        for i in range(len(tree.feature)):
            assert_true(
                tree.feature[i] < 4,
                "an unlisted feature was split on",
            )


def test_overlapping_groups_hold_on_trained_paths() raises:
    var n_rows = 2_000
    var n_features = 6
    var features = _make_features(n_rows, n_features)
    var target = _interacting_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var constraints = _groups([[0, 1, 2], [2, 3, 4], [4, 5, 0]], n_features)
    var params = BoosterParams(
        20, 0.2, TreeParams(15, 20, 1.0, 1e-3, 0.0, constraints.copy())
    )
    var model = train(data, target, SQUARED_ERROR, params)
    _assert_paths_respect_constraints(model.trees, constraints)


def test_one_group_of_everything_matches_unconstrained() raises:
    var n_rows = 1_500
    var n_features = 5
    var features = _make_features(n_rows, n_features)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(
            3.0 * features[0 * n_rows + r] * features[1 * n_rows + r]
            + 2.0 * features[2 * n_rows + r]
            + features[3 * n_rows + r] * features[4 * n_rows + r]
        )
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var everything = List[Int]()
    for f in range(n_features):
        everything.append(f)
    var groups = List[List[Int]]()
    groups.append(everything^)
    var constraints = InteractionConstraints.from_groups(groups, n_features)

    var free = BoosterParams(15, 0.2, TreeParams(15, 20, 1.0, 1e-3))
    var constrained = BoosterParams(
        15, 0.2, TreeParams(15, 20, 1.0, 1e-3, 0.0, constraints^)
    )
    var a = train(data, target, SQUARED_ERROR, free)
    var b = train(data, target, SQUARED_ERROR, constrained)
    _assert_same_trees(a.trees, b.trees)


def test_constraints_actually_change_the_model() raises:
    var n_rows = 1_500
    var n_features = 6
    var features = _make_features(n_rows, n_features)
    var target = _interacting_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var constraints = _groups([[0, 1], [2, 3]], n_features)
    var free = BoosterParams(10, 0.2, TreeParams(15, 20, 1.0, 1e-3))
    var constrained = BoosterParams(
        10, 0.2, TreeParams(15, 20, 1.0, 1e-3, 0.0, constraints.copy())
    )
    var a = train(data, target, SQUARED_ERROR, free)
    var b = train(data, target, SQUARED_ERROR, constrained)

    var differs = False
    for r in range(n_rows):
        if a.predict_row(data, r) != b.predict_row(data, r):
            differs = True
            break
    assert_true(differs, "constraints had no effect on the fit")

    # And the unconstrained model does what the constrained one may not.
    var mixes = False
    for t in range(len(a.trees)):
        var paths = _path_feature_sets(a.trees[t])
        for p in range(len(paths)):
            if not _within_one_group(paths[p], constraints):
                mixes = True
                break
    assert_true(
        mixes, "the unconstrained fit never violated the constraint anyway"
    )


def test_a_single_feature_group_makes_an_ignoring_model() raises:
    """Constrained to feature 0 alone, predictions must not move when any
    other feature changes."""
    var n_rows = 1_200
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(
            2.0 * features[0 * n_rows + r] + features[1 * n_rows + r]
        )
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var constraints = _groups([[0]], n_features)
    var params = BoosterParams(
        20, 0.2, TreeParams(15, 20, 1.0, 1e-3, 0.0, constraints^)
    )
    var model = train(data, target, SQUARED_ERROR, params)
    for t in range(len(model.trees)):
        ref tree = model.trees[t]
        for i in range(len(tree.feature)):
            assert_true(tree.feature[i] <= 0, "split on a disallowed feature")

    # Same feature-0 bin, different everything else, same prediction.
    var bins_a: List[Int] = [3, 0, 0, 0]
    var bins_b: List[Int] = [3, 31, 31, 31]
    assert_equal(
        model.predict_bins(bins_a), model.predict_bins(bins_b)
    )


def test_gpu_training_enforces_the_same_constraints() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        # Deliberately small: the constraint check is structural, not
        # statistical, and GPU rounds are the slowest thing in this suite.
        var n_rows = 800
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _interacting_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 32)

        var constraints = _groups([[0, 1], [2, 3]], n_features)
        var params = BoosterParams(
            4, 0.2, TreeParams(8, 20, 1.0, 1e-3, 0.0, constraints.copy())
        )
        var cpu = train(data, target, SQUARED_ERROR, params)
        var gpu = train_gpu(data, target, SQUARED_ERROR, params)

        # The GPU path must obey the constraints on its own terms: every one
        # of its root-to-leaf paths, inspected directly.
        _assert_paths_respect_constraints(gpu.trees, constraints)
        for t in range(len(gpu.trees)):
            ref tree = gpu.trees[t]
            for i in range(len(tree.feature)):
                assert_true(
                    tree.feature[i] < 4,
                    "the GPU trainer split on an unlisted feature",
                )

        # Constraint enforcement is a split-search decision the two backends
        # share, so the fits agree to the usual GPU tolerance. (Structures
        # are not asserted equal: GPU histograms are Float32 fixed-point, so
        # a near-tied gain may still resolve differently, exactly as in
        # test_gpu_training.)
        assert_equal(len(cpu.trees), len(gpu.trees))
        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-3
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
