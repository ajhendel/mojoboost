"""Deterministic feature subsampling.

Covers the sampler itself (count formula, no replacement, ordering, seeds),
the machinery that carries a feature set into histogram building and split
search, and the training-level guarantees: a fraction of 1.0 reproduces
unsubsampled training bit for bit, the same seed reproduces a model bit for
bit, different seeds give different models, invalid fractions raise, and
importance follows the features a tree was actually allowed to use.
"""

from std.os import setenv
from std.testing import assert_almost_equal, assert_equal, assert_true, TestSuite

from mojotrees.binning import bin_equal_width
from mojotrees.boosting import SQUARED_ERROR, BoosterParams, train
from mojotrees.histogram import build_histogram, build_histogram_subset
from mojotrees.importance import gain_importance, split_importance
from mojotrees.sampling import (
    DEFAULT_FEATURE_FRACTION_SEED,
    sample_without_replacement,
    select_node_features,
    select_tree_features,
    selection_count,
)
from mojotrees.split import find_best_split
from mojotrees.tree import Tree, TreeParams, grow_tree
from support import _make_features, _uniform


def _bits(v: Float64) -> UInt64:
    return v.to_bits().cast[DType.uint64]()


def _assert_same_tree(a: Tree, b: Tree) raises:
    """Two trees, field for field, floats as bits. No tolerance: the changes
    this file guards are supposed to move nothing at all, so a one-ulp
    difference is a defect and not a rounding difference to absorb."""
    assert_equal(a.n_leaves, b.n_leaves)
    assert_equal(len(a.feature), len(b.feature))
    for i in range(len(a.feature)):
        assert_equal(a.feature[i], b.feature[i])
        assert_equal(a.threshold_bin[i], b.threshold_bin[i])
        assert_equal(a.left[i], b.left[i])
        assert_equal(a.right[i], b.right[i])
        assert_equal(_bits(a.value[i]), _bits(b.value[i]))
        assert_equal(_bits(a.split_gain[i]), _bits(b.split_gain[i]))
        assert_equal(Int(a.default_left[i]), Int(b.default_left[i]))
        assert_equal(a.missing_bin[i], b.missing_bin[i])
        assert_equal(a.cat_offset[i], b.cat_offset[i])
        assert_equal(_bits(a.count[i]), _bits(b.count[i]))
    assert_equal(len(a.cat_bitset), len(b.cat_bitset))
    for i in range(len(a.cat_bitset)):
        assert_equal(a.cat_bitset[i], b.cat_bitset[i])


def _target(features: List[Float64], n_rows: Int, n_features: Int) -> List[Float64]:
    """Every feature carries signal, with steadily shrinking coefficients, so
    an unsubsampled ensemble concentrates on the first few features and a
    subsampled one is forced to look further."""
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var v = 0.0
        for f in range(n_features):
            v += (1.0 / Float64(f + 1)) * features[f * n_rows + r]
        y.append(v)
    return y^


def _grad(n_rows: Int) -> List[Float64]:
    var grad = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(2.0 * _uniform(UInt64(1_000_000 + r)) - 1.0)
    return grad^


def _hess(n_rows: Int) -> List[Float64]:
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        hess.append(_uniform(UInt64(2_000_000 + r)) + 0.5)
    return hess^


def _params(
    n_rounds: Int,
    feature_fraction: Float64 = 1.0,
    feature_fraction_bynode: Float64 = 1.0,
    seed: Int = DEFAULT_FEATURE_FRACTION_SEED,
) -> BoosterParams:
    return BoosterParams(
        n_rounds,
        0.2,
        TreeParams(
            8,
            5,
            1.0,
            1e-3,
            0.0,
            feature_fraction=feature_fraction,
            feature_fraction_bynode=feature_fraction_bynode,
            feature_fraction_seed=seed,
        ),
    )


def test_selection_count_matches_lightgbm_formula() raises:
    # round(total * fraction), floored at 2 features.
    assert_equal(selection_count(10, 1.0), 10)
    assert_equal(selection_count(10, 0.5), 5)
    assert_equal(selection_count(10, 0.55), 6)  # 5.5 rounds up
    assert_equal(selection_count(7, 0.3), 2)  # 2.1 truncates to 2
    assert_equal(selection_count(100, 0.345), 35)  # 34.5 rounds up
    # The floor of 2 wins over any smaller rounded count.
    assert_equal(selection_count(10, 0.01), 2)
    assert_equal(selection_count(3, 0.1), 2)
    # ...unless there are fewer than 2 candidates at all.
    assert_equal(selection_count(1, 0.5), 1)
    assert_equal(selection_count(0, 0.5), 0)


def test_selection_is_without_replacement_and_ordered() raises:
    var n_features = 40
    for seed in range(12):
        var chosen = select_tree_features(n_features, 0.35, seed, 0)
        assert_equal(len(chosen), selection_count(n_features, 0.35))
        for i in range(len(chosen)):
            assert_true(chosen[i] >= 0 and chosen[i] < n_features)
            if i > 0:
                # Strictly ascending: ordered, and therefore no repeats.
                assert_true(chosen[i] > chosen[i - 1])


def test_selection_is_deterministic_per_tree_and_node() raises:
    var a = select_tree_features(30, 0.4, 7, 3)
    var b = select_tree_features(30, 0.4, 7, 3)
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])

    var na = select_node_features(a, 0.5, 7, 3, 11)
    var nb = select_node_features(b, 0.5, 7, 3, 11)
    assert_equal(len(na), len(nb))
    for i in range(len(na)):
        assert_equal(na[i], nb[i])


def test_different_trees_nodes_and_seeds_select_differently() raises:
    def same(a: List[Int], b: List[Int]) -> Bool:
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i]:
                return False
        return True

    var base = select_tree_features(40, 0.3, 2, 0)
    # A different tree index, a different seed, and a different node id each
    # move the draw (12 features out of 40: agreement by chance is remote).
    assert_true(not same(base, select_tree_features(40, 0.3, 2, 1)))
    assert_true(not same(base, select_tree_features(40, 0.3, 3, 0)))
    var node_a = select_node_features(base, 0.5, 2, 0, 1)
    var node_b = select_node_features(base, 0.5, 2, 0, 2)
    assert_true(not same(node_a, node_b))


def test_node_selection_is_a_subset_of_the_tree_selection() raises:
    var tree_features = select_tree_features(50, 0.4, 5, 2)
    for node in range(8):
        var node_features = select_node_features(tree_features, 0.5, 5, 2, node)
        assert_equal(
            len(node_features), selection_count(len(tree_features), 0.5)
        )
        var i = 0
        for j in range(len(node_features)):
            while i < len(tree_features) and tree_features[i] != node_features[j]:
                i += 1
            assert_true(i < len(tree_features))
            i += 1


def test_fraction_one_selects_every_feature() raises:
    var all_features = select_tree_features(13, 1.0, 99, 4)
    assert_equal(len(all_features), 13)
    for f in range(13):
        assert_equal(all_features[f], f)
    var by_node = select_node_features(all_features, 1.0, 99, 4, 6)
    assert_equal(len(by_node), 13)
    for f in range(13):
        assert_equal(by_node[f], f)


def test_sample_without_replacement_edge_counts() raises:
    var pool: List[Int] = [2, 4, 6, 8]
    var none = sample_without_replacement(pool, 0, UInt64(1))
    assert_equal(len(none), 0)
    var all_of_them = sample_without_replacement(pool, 4, UInt64(1))
    assert_equal(len(all_of_them), 4)
    for i in range(4):
        assert_equal(all_of_them[i], pool[i])
    var clamped = sample_without_replacement(pool, 9, UInt64(1))
    assert_equal(len(clamped), 4)


def test_invalid_fractions_raise() raises:
    var bad: List[Float64] = [0.0, -0.1, 1.5, 2.0]
    var three: List[Int] = [0, 1, 2]
    for i in range(len(bad)):
        var raised = False
        try:
            _ = select_tree_features(10, bad[i], 1, 0)
        except:
            raised = True
        assert_true(raised)

        raised = False
        try:
            _ = select_node_features(List[Int](copy=three), bad[i], 1, 0, 0)
        except:
            raised = True
        assert_true(raised)

    # And through the trainer, where the tree grower validates them.
    var n_rows = 200
    var n_features = 6
    var features = _make_features(n_rows, n_features)
    var target = _target(features, n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 16)
    var raised_tree = False
    try:
        _ = train(data, target, SQUARED_ERROR, _params(2, 0.0, 1.0))
    except:
        raised_tree = True
    assert_true(raised_tree)
    var raised_node = False
    try:
        _ = train(data, target, SQUARED_ERROR, _params(2, 1.0, 1.5))
    except:
        raised_node = True
    assert_true(raised_node)


def test_selected_histogram_matches_the_full_build() raises:
    """A histogram over an explicit feature list equals the plain build on the
    selected slices and stays zero elsewhere, bit for bit."""
    var n_rows = 500
    var n_features = 9
    var n_bins = 16
    var features = _make_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, n_bins)
    var grad = _grad(n_rows)
    var hess = _hess(n_rows)

    var full = build_histogram(data, grad, hess)

    var everything = List[Int]()
    for f in range(n_features):
        everything.append(f)
    var indexed = build_histogram(data, grad, hess, everything)
    for i in range(n_features * n_bins):
        assert_equal(full.grad_at(i), indexed.grad_at(i))
        assert_equal(full.hess_at(i), indexed.hess_at(i))
        assert_equal(full.count_at(i), indexed.count_at(i))

    var some: List[Int] = [1, 4, 5, 8]
    var partial = build_histogram(data, grad, hess, some)
    for f in range(n_features):
        var selected = False
        for i in range(len(some)):
            if some[i] == f:
                selected = True
        for b in range(n_bins):
            var i = f * n_bins + b
            if selected:
                assert_equal(full.grad_at(i), partial.grad_at(i))
                assert_equal(full.hess_at(i), partial.hess_at(i))
                assert_equal(full.count_at(i), partial.count_at(i))
            else:
                assert_equal(partial.grad_at(i), 0.0)
                assert_equal(partial.hess_at(i), 0.0)
                assert_equal(partial.count_at(i), 0)

    # Same story for a row subset, which is what tree nodes build.
    var rows = List[Int]()
    for r in range(0, n_rows, 3):
        rows.append(r)
    var sub_full = build_histogram_subset(data, grad, hess, rows)
    var sub_partial = build_histogram_subset(data, grad, hess, rows, some)
    for i in range(len(some)):
        var base = some[i] * n_bins
        for b in range(n_bins):
            assert_equal(
                sub_full.grad_at(base + b), sub_partial.grad_at(base + b)
            )
            assert_equal(
                sub_full.count_at(base + b), sub_partial.count_at(base + b)
            )

    var out_of_range: List[Int] = [0, n_features]
    var raised = False
    try:
        _ = build_histogram(data, grad, hess, out_of_range)
    except:
        raised = True
    assert_true(raised)


def test_split_search_ignores_unselected_features() raises:
    """The best split over a feature list equals the best split found by
    scanning every feature and keeping only candidates from that list."""
    var n_rows = 400
    var n_features = 7
    var n_bins = 16
    var features = _make_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, n_bins)
    var grad = _grad(n_rows)
    var hess = _hess(n_rows)
    # Give feature 0 the strongest signal, then exclude it.
    for r in range(n_rows):
        grad[r] += 4.0 * Float64(data.bin_at(r, 0) - n_bins // 2)
    var hist = build_histogram(data, grad, hess)

    var unrestricted = find_best_split(hist, 1.0, 1e-3, 1)
    assert_true(unrestricted.found)
    assert_equal(unrestricted.feature, 0)

    var allowed: List[Int] = [2, 3, 6]
    var restricted = find_best_split(hist, 1.0, 1e-3, 1, 0.0, [], allowed)
    assert_true(restricted.found)
    var is_allowed = False
    for i in range(len(allowed)):
        if allowed[i] == restricted.feature:
            is_allowed = True
    assert_true(is_allowed)

    # Reference: mask the same features by hand with the interaction mask.
    var mask = List[Bool]()
    for f in range(n_features):
        var on = False
        for i in range(len(allowed)):
            if allowed[i] == f:
                on = True
        mask.append(on)
    var by_mask = find_best_split(hist, 1.0, 1e-3, 1, 0.0, mask)
    assert_equal(restricted.feature, by_mask.feature)
    assert_equal(restricted.bin, by_mask.bin)
    assert_equal(restricted.gain, by_mask.gain)


def test_grown_tree_only_splits_on_selected_features() raises:
    var n_rows = 400
    var n_features = 12
    var features = _make_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 16)
    var grad = _grad(n_rows)
    var hess = _hess(n_rows)

    var params = TreeParams(
        16, 5, 1.0, 1e-3, 0.0, feature_fraction=0.35, feature_fraction_seed=11
    )
    for tree_index in range(4):
        var tree = grow_tree(data, grad, hess, params, [], tree_index)
        var selected = select_tree_features(n_features, 0.35, 11, tree_index)
        for node in range(len(tree.feature)):
            var f = tree.feature[node]
            if f < 0:
                continue
            var found = False
            for i in range(len(selected)):
                if selected[i] == f:
                    found = True
            assert_true(found)


def test_fraction_one_matches_unsubsampled_training() raises:
    """A fraction of 1.0 must reproduce the default path exactly, whatever the
    seed, since selecting every feature is not a random event."""
    var n_rows = 250
    var n_features = 10
    var features = _make_features(n_rows, n_features)
    var target = _target(features, n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var plain = BoosterParams(10, 0.2, TreeParams(8, 5, 1.0, 1e-3))
    var base = train(data, target, SQUARED_ERROR, plain)
    var explicit = train(data, target, SQUARED_ERROR, _params(10, 1.0, 1.0, 4242))

    assert_equal(len(base.trees), len(explicit.trees))
    for r in range(n_rows):
        assert_equal(
            base.predict_raw_row(data, r), explicit.predict_raw_row(data, r)
        )


def test_same_seed_reproduces_the_model_bit_for_bit() raises:
    var n_rows = 250
    var n_features = 12
    var features = _make_features(n_rows, n_features)
    var target = _target(features, n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var params = _params(10, 0.4, 0.7, 17)
    var a = train(data, target, SQUARED_ERROR, params)
    var b = train(data, target, SQUARED_ERROR, params)

    assert_equal(len(a.trees), len(b.trees))
    for t in range(len(a.trees)):
        assert_equal(len(a.trees[t].feature), len(b.trees[t].feature))
        for node in range(len(a.trees[t].feature)):
            assert_equal(a.trees[t].feature[node], b.trees[t].feature[node])
            assert_equal(
                a.trees[t].threshold_bin[node], b.trees[t].threshold_bin[node]
            )
            assert_equal(a.trees[t].value[node], b.trees[t].value[node])
    for r in range(n_rows):
        assert_equal(a.predict_raw_row(data, r), b.predict_raw_row(data, r))


def test_different_seeds_give_different_models() raises:
    var n_rows = 250
    var n_features = 12
    var features = _make_features(n_rows, n_features)
    var target = _target(features, n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var a = train(data, target, SQUARED_ERROR, _params(10, 0.4, 1.0, 1))
    var b = train(data, target, SQUARED_ERROR, _params(10, 0.4, 1.0, 2))

    var differs = False
    for r in range(n_rows):
        if a.predict_raw_row(data, r) != b.predict_raw_row(data, r):
            differs = True
    assert_true(differs)


def test_bynode_changes_the_model_and_stays_deterministic() raises:
    var n_rows = 250
    var n_features = 12
    var features = _make_features(n_rows, n_features)
    var target = _target(features, n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var tree_only = train(data, target, SQUARED_ERROR, _params(10, 0.5, 1.0, 3))
    var with_node = train(data, target, SQUARED_ERROR, _params(10, 0.5, 0.5, 3))
    var with_node_again = train(
        data, target, SQUARED_ERROR, _params(10, 0.5, 0.5, 3)
    )

    var differs = False
    for r in range(n_rows):
        if (
            tree_only.predict_raw_row(data, r)
            != with_node.predict_raw_row(data, r)
        ):
            differs = True
        assert_equal(
            with_node.predict_raw_row(data, r),
            with_node_again.predict_raw_row(data, r),
        )
    assert_true(differs)


def test_subsampled_model_still_fits() raises:
    """Subsampling is a regularizer, not a way to stop learning: a subsampled
    ensemble must still explain most of the variance an unsubsampled one
    does."""
    var n_rows = 300
    var n_features = 10
    var features = _make_features(n_rows, n_features)
    var target = _target(features, n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var mean = 0.0
    for r in range(n_rows):
        mean += target[r]
    mean /= Float64(n_rows)
    var total_ss = 0.0
    for r in range(n_rows):
        total_ss += (target[r] - mean) * (target[r] - mean)

    var full = train(data, target, SQUARED_ERROR, _params(30, 1.0, 1.0))
    var sub = train(data, target, SQUARED_ERROR, _params(30, 0.5, 1.0, 8))
    var full_sse = 0.0
    var sub_sse = 0.0
    for r in range(n_rows):
        var df = full.predict_row(data, r) - target[r]
        var ds = sub.predict_row(data, r) - target[r]
        full_sse += df * df
        sub_sse += ds * ds
    assert_true(full_sse < 0.2 * total_ss)
    assert_true(sub_sse < 0.4 * total_ss)


def test_importance_only_credits_selected_features() raises:
    """Split and gain importance follow the sampled sets: a single tree can
    only credit features its draw allowed, and subsampling across an ensemble
    spreads credit onto features the unsubsampled model never splits on."""
    var n_rows = 300
    var n_features = 12
    var features = _make_features(n_rows, n_features)
    var target = _target(features, n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    # One tree: every split must come from that tree's own draw.
    var one = train(data, target, SQUARED_ERROR, _params(1, 0.3, 1.0, 21))
    var selected = select_tree_features(n_features, 0.3, 21, 0)
    var counts = split_importance(one.trees, n_features)
    var gains = gain_importance(one.trees, n_features)
    var total_splits = 0
    for f in range(n_features):
        var in_draw = False
        for i in range(len(selected)):
            if selected[i] == f:
                in_draw = True
        if not in_draw:
            assert_equal(counts[f], 0)
            assert_equal(gains[f], 0.0)
        total_splits += counts[f]
        assert_true(gains[f] >= 0.0)
    assert_equal(total_splits, one.trees[0].n_leaves - 1)

    # An ensemble: subsampling must use strictly more distinct features than
    # the unsubsampled ensemble, which piles onto the strongest few.
    var full = train(data, target, SQUARED_ERROR, _params(25, 1.0, 1.0))
    var sub = train(data, target, SQUARED_ERROR, _params(25, 0.4, 1.0, 21))
    var full_counts = split_importance(full.trees, n_features)
    var sub_counts = split_importance(sub.trees, n_features)
    var full_used = 0
    var sub_used = 0
    for f in range(n_features):
        if full_counts[f] > 0:
            full_used += 1
        if sub_counts[f] > 0:
            sub_used += 1
    assert_true(sub_used >= full_used)
    assert_true(sub_used > 1)

    # Gain importance stays consistent with the recorded per-node gains.
    var sub_gains = gain_importance(sub.trees, n_features)
    var summed = 0.0
    for f in range(n_features):
        summed += sub_gains[f]
    var node_total = 0.0
    for t in range(len(sub.trees)):
        for node in range(len(sub.trees[t].feature)):
            if sub.trees[t].feature[node] >= 0:
                node_total += sub.trees[t].split_gain[node]
    assert_almost_equal(summed, node_total, atol=1e-9)


# ---------------------------------------------------------------------------
# The undrawn node: an empty node feature list is the complete one
# ---------------------------------------------------------------------------
#
# `tree.grow_tree` no longer materializes a per-node feature list when no
# fraction narrows it. At `feature_fraction = 1.0` the tree's own set is every
# feature, and with both inner fractions at 1.0 the two copies
# `select_split_features` would make are copies of that same complete set --
# which is exactly what the empty list already means to `find_best_split`, to
# `cegb.prepare_cegb_node` and to the profile's cell count. The list is
# therefore not built.
#
# The claim under test is that this is the *same selection*, not merely a
# selection that happens to fit as well. The two arms below are built so that
# one takes the undrawn path and the other materializes a node list that is
# provably the complete set, and the trees must agree bit for bit.


comptime _COMPLETE_DRAW_FRACTION = 0.99
"""A bylevel fraction below 1.0 -- which is what makes `grow_tree` draw at all
-- whose `selection_count` at this fixture's width is still every feature, so
the materialized list is `[0, n_features)`. `test_the_undrawn_node_...` asserts
that arithmetic rather than trusting it: if `selection_count` ever changed its
rounding, this constant would silently start comparing two different draws and
the test would become a test of nothing."""


def _signal_grad(
    features: List[Float64], n_rows: Int
) raises -> List[Float64]:
    """Gradients with real structure in the first two features, so growth
    reaches the leaf budget instead of stopping at a stump. A stump would make
    "the two trees agree" a claim about one node."""
    var grad = _grad(n_rows)
    for r in range(n_rows):
        grad[r] += 3.0 * features[r] - 2.0 * features[n_rows + r]
    return grad^


def test_the_undrawn_node_grows_the_tree_a_complete_draw_grows() raises:
    """An undrawn node and a node whose draw is the complete feature set must
    grow the identical tree.

    Arm A is the default: `feature_fraction`, `feature_fraction_bylevel` and
    `feature_fraction_bynode` all 1.0, so no node draws and every node's list
    is empty. Arm B sets the bylevel fraction below 1.0, which is the whole
    of the condition `grow_tree` gates on, so every node materializes a list;
    the assertions below prove that list is `[0, n_features)`, so the two arms
    differ in whether the list exists and in nothing else.

    Both halves of the gate are proved rather than assumed. `selection_count`
    is asserted to return every feature at arm B's fraction, and
    `select_tree_features` at 1.0 is asserted to be the ascending complete
    list, which is the other half of `grow_tree`'s condition.
    """
    var n_rows = 400
    var n_features = 10
    var n_bins = 16
    var features = _make_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, n_bins)
    var grad = _signal_grad(features, n_rows)
    var hess = _hess(n_rows)

    # The gate, both halves, as arithmetic.
    assert_true(_COMPLETE_DRAW_FRACTION < 1.0)
    assert_equal(
        selection_count(n_features, _COMPLETE_DRAW_FRACTION), n_features
    )
    var tree_set = select_tree_features(n_features, 1.0, 11, 0)
    assert_equal(len(tree_set), n_features)
    for f in range(n_features):
        assert_equal(tree_set[f], f)

    var undrawn = TreeParams(
        16, 5, 1.0, 1e-3, 0.0,
        feature_fraction=1.0,
        feature_fraction_bynode=1.0,
        feature_fraction_seed=11,
        feature_fraction_bylevel=1.0,
    )
    var complete_draw = TreeParams(
        16, 5, 1.0, 1e-3, 0.0,
        feature_fraction=1.0,
        feature_fraction_bynode=1.0,
        feature_fraction_seed=11,
        feature_fraction_bylevel=_COMPLETE_DRAW_FRACTION,
    )
    for tree_index in range(3):
        var a = grow_tree(data, grad, hess, undrawn, [], tree_index)
        var b = grow_tree(data, grad, hess, complete_draw, [], tree_index)
        assert_true(a.n_leaves > 1)
        _assert_same_tree(a, b)


def test_the_undrawn_node_is_identical_at_one_three_and_eight_workers() raises:
    """Determinism across `MOJOTREES_NUM_WORKERS` for the undrawn path.

    The crossover is forced to zero so the parallel path is actually taken at
    this fixture's size, rather than all three arms falling serial and quietly
    measuring the same schedule.
    """
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "1")
    var n_rows = 400
    var n_features = 10
    var features = _make_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 16)
    var grad = _signal_grad(features, n_rows)
    var hess = _hess(n_rows)
    var params = TreeParams(16, 5, 1.0, 1e-3, 0.0)

    _ = setenv("MOJOTREES_NUM_WORKERS", "1")
    var base = grow_tree(data, grad, hess, params)
    assert_true(base.n_leaves > 1)

    var workers: List[String] = ["3", "8"]
    for i in range(len(workers)):
        _ = setenv("MOJOTREES_NUM_WORKERS", workers[i])
        _assert_same_tree(grow_tree(data, grad, hess, params), base)

    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    _ = setenv("MOJOTREES_PARALLEL_MIN_OPS", "")


def test_a_complete_feature_list_excludes_nothing_however_it_is_ordered() raises:
    """`histogram._zero_excluded` short-circuits on a strictly ascending list
    that names every feature, and takes the full masked path on any other
    list. Both sides of that gate must produce the same histogram.

    The gate is chosen deliberately: `_check_features` validates range only
    and accepts repeats, so `len(features) == n_features` alone would not
    establish that nothing is excluded. A permuted complete list is the case
    that proves the short-circuit is not being reached by length alone -- it
    is complete, it is the same length, and it is not ascending, so it takes
    the long path and must still leave every slice written.
    """
    var n_rows = 300
    var n_features = 8
    var n_bins = 16
    var features = _make_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, n_bins)
    var grad = _grad(n_rows)
    var hess = _hess(n_rows)

    var full = build_histogram(data, grad, hess)

    var ascending = List[Int]()
    for f in range(n_features):
        ascending.append(f)
    # Complete, same length, not ascending: the long path.
    var permuted = List[Int]()
    for f in range(n_features):
        permuted.append(n_features - 1 - f)

    var by_ascending = build_histogram(data, grad, hess, ascending)
    var by_permuted = build_histogram(data, grad, hess, permuted)
    for i in range(n_features * n_bins):
        assert_equal(_bits(full.grad_at(i)), _bits(by_ascending.grad_at(i)))
        assert_equal(_bits(full.hess_at(i)), _bits(by_ascending.hess_at(i)))
        assert_equal(full.count_at(i), by_ascending.count_at(i))
        assert_equal(_bits(full.grad_at(i)), _bits(by_permuted.grad_at(i)))
        assert_equal(_bits(full.hess_at(i)), _bits(by_permuted.hess_at(i)))
        assert_equal(full.count_at(i), by_permuted.count_at(i))

    # And a repeated id is still accepted by the range check, which is why the
    # short-circuit tests ascendingness and not just length: this list has
    # `n_features` entries and excludes all but one feature.
    var repeated = List[Int]()
    for _ in range(n_features):
        repeated.append(0)
    var by_repeated = build_histogram(data, grad, hess, repeated)
    for b in range(n_bins):
        assert_equal(_bits(full.grad_at(b)), _bits(by_repeated.grad_at(b)))
    for f in range(1, n_features):
        for b in range(n_bins):
            assert_equal(by_repeated.count_at(f * n_bins + b), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
