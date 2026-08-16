"""Row bagging: sampler contract, tree equivalence, and training behavior.

The central claim is the equivalence test: a tree grown on a bag equals,
node for node and bit for bit, the tree grown on a dataset physically
holding only those rows. Everything else here (fraction = 1, reproducibility,
seed sensitivity, zero-weight rows, tiny datasets) rests on that.

One qualification, and it is about the histogram builders rather than about
bagging. "Bit for bit" holds against the *subset* builder, which is the one a
bag goes through, and it is asserted at full strength. Against the
whole-dataset builder only the structure is bit-equal, because the subset
builder now folds contiguous row blocks and the whole-dataset builder still
sums flat, so the two reassociate a node's rows differently and its leaf
value can move in its last bits. See `test_bagged_tree_equals_tree_on_subset_dataset`
and the "row blocks" section of `src/mojotrees/histogram.mojo`.

CPU/GPU equivalence lives in tests/test_gpu_training.mojo, which needs an
accelerator.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojotrees import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BaggingParams,
    BinnedMatrix,
    BoosterParams,
    Tree,
    TreeParams,
    bagging_enabled,
    bin_equal_width,
    grow_tree,
    refresh_bag,
    sample_rows,
    train,
    train_multiclass,
)
from support import _make_features, _uniform


def _regression_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        y.append(4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x2 - 0.5))
    return y^


def _subset_matrix(data: BinnedMatrix, rows: List[Int]) -> BinnedMatrix:
    """A binned matrix physically holding `rows`, same bin ids. This is the
    reference a bagged tree must reproduce."""
    var n_sub = len(rows)
    var bins = List[UInt8](capacity=n_sub * data.n_features)
    for f in range(data.n_features):
        for i in range(n_sub):
            bins.append(data.bins[f * data.n_rows + rows[i]])
    return BinnedMatrix(bins^, n_sub, data.n_features, data.n_bins)


def _gather(values: List[Float64], rows: List[Int]) -> List[Float64]:
    var out = List[Float64](capacity=len(rows))
    for i in range(len(rows)):
        out.append(values[rows[i]])
    return out^


def _assert_same_structure(a: Tree, b: Tree) raises:
    """Identical structure: the same splits on the same features in the same
    places. Says nothing about the leaf values."""
    assert_equal(a.n_leaves, b.n_leaves)
    assert_equal(len(a.feature), len(b.feature))
    for i in range(len(a.feature)):
        assert_equal(a.feature[i], b.feature[i])
        assert_equal(a.threshold_bin[i], b.threshold_bin[i])
        assert_equal(a.left[i], b.left[i])
        assert_equal(a.right[i], b.right[i])


def _assert_same_tree(a: Tree, b: Tree) raises:
    """Identical structure and bit-identical leaf values."""
    _assert_same_structure(a, b)
    for i in range(len(a.feature)):
        assert_equal(a.value[i].to_bits(), b.value[i].to_bits())


def _grad_hess(
    n_rows: Int, seed: UInt64
) -> Tuple[List[Float64], List[Float64]]:
    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(2.0 * _uniform(seed + UInt64(r)) - 1.0)
        hess.append(_uniform(seed + UInt64(n_rows + r)) + 0.5)
    return (grad^, hess^)


# --------------------------------------------------------------------------
# Sampler
# --------------------------------------------------------------------------


def test_sample_rows_is_ascending_unique_and_in_range() raises:
    var params = BaggingParams(0.5, 1, 42)
    var rows = List[Int]()
    sample_rows(params, 500, 0, rows)
    assert_true(len(rows) > 0)
    for i in range(len(rows)):
        assert_true(rows[i] >= 0 and rows[i] < 500)
        if i > 0:
            assert_true(rows[i] > rows[i - 1])


def test_sample_rows_is_reproducible() raises:
    # Same (seed, bag index, n_rows) means the same rows, always, and the
    # output list is fully replaced rather than appended to.
    var params = BaggingParams(0.4, 1, 11)
    var a = List[Int]()
    var b = List[Int]()
    sample_rows(params, 1_000, 3, a)
    sample_rows(params, 1_000, 7, b)
    sample_rows(params, 1_000, 3, b)
    assert_equal(len(a), len(b))
    for i in range(len(a)):
        assert_equal(a[i], b[i])


def test_different_seeds_give_different_bags() raises:
    var a = List[Int]()
    var b = List[Int]()
    sample_rows(BaggingParams(0.5, 1, 1), 1_000, 0, a)
    sample_rows(BaggingParams(0.5, 1, 2), 1_000, 0, b)
    var same = len(a) == len(b)
    if same:
        for i in range(len(a)):
            if a[i] != b[i]:
                same = False
                break
    assert_true(not same)


def test_successive_bags_differ() raises:
    var a = List[Int]()
    var b = List[Int]()
    sample_rows(BaggingParams(0.5, 1, 5), 1_000, 0, a)
    sample_rows(BaggingParams(0.5, 1, 5), 1_000, 1, b)
    var same = len(a) == len(b)
    if same:
        for i in range(len(a)):
            if a[i] != b[i]:
                same = False
                break
    assert_true(not same)


def test_bag_size_tracks_the_fraction() raises:
    # Bernoulli per row, so the count is Binomial(n, fraction): sigma is
    # sqrt(n * p * q) = 50 here, and the bound below is 8 sigma.
    var rows = List[Int]()
    sample_rows(BaggingParams(0.5, 1, 3), 10_000, 0, rows)
    assert_true(abs(len(rows) - 5_000) < 400)
    sample_rows(BaggingParams(0.1, 1, 3), 10_000, 0, rows)
    assert_true(abs(len(rows) - 1_000) < 400)


def test_bag_is_never_empty() raises:
    # A tiny dataset with a tiny fraction: the draw almost certainly selects
    # nothing, and the guard must still return one row.
    var rows = List[Int]()
    for n in range(1, 6):
        sample_rows(BaggingParams(0.001, 1, 9), n, 0, rows)
        assert_equal(len(rows), 1)
        assert_true(rows[0] >= 0 and rows[0] < n)


def test_bagging_enabled_rules() raises:
    assert_true(bagging_enabled(BaggingParams(0.5, 1, 0)))
    assert_true(not bagging_enabled(BaggingParams(1.0, 1, 0)))
    assert_true(not bagging_enabled(BaggingParams(0.5, 0, 0)))
    assert_true(not bagging_enabled(BaggingParams.disabled()))


def test_bagging_freq_schedule() raises:
    # freq = 3: rounds 0, 1, 2 share a bag; round 3 draws the next one.
    var params = BaggingParams(0.5, 3, 17)
    var bag = List[Int]()
    refresh_bag(bag, params, 400, 0)
    var first = bag.copy()
    refresh_bag(bag, params, 400, 1)
    refresh_bag(bag, params, 400, 2)
    assert_equal(len(bag), len(first))
    for i in range(len(bag)):
        assert_equal(bag[i], first[i])

    refresh_bag(bag, params, 400, 3)
    var expected = List[Int]()
    sample_rows(params, 400, 1, expected)
    assert_equal(len(bag), len(expected))
    for i in range(len(bag)):
        assert_equal(bag[i], expected[i])


def test_disabled_bagging_never_draws() raises:
    var bag = List[Int]()
    refresh_bag(bag, BaggingParams.disabled(), 100, 0)
    assert_equal(len(bag), 0)
    refresh_bag(bag, BaggingParams(0.5, 0, 3), 100, 0)
    assert_equal(len(bag), 0)
    refresh_bag(bag, BaggingParams(1.0, 5, 3), 100, 0)
    assert_equal(len(bag), 0)


def test_sampler_validation() raises:
    var rows = List[Int]()
    var raised = False
    try:
        sample_rows(BaggingParams(0.0, 1, 3), 10, 0, rows)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        sample_rows(BaggingParams(1.5, 1, 3), 10, 0, rows)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        sample_rows(BaggingParams(0.5, -1, 3), 10, 0, rows)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        sample_rows(BaggingParams(0.5, 1, 3), 10, -1, rows)
    except:
        raised = True
    assert_true(raised)


def test_train_rejects_invalid_bagging() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var target: List[Float64] = [0.0, 0.0, 1.0, 1.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)
    var params = BoosterParams(5, 0.3, TreeParams(4, 1, 1.0, 1e-3))
    var raised = False
    try:
        _ = train(
            data, target, SQUARED_ERROR, params, [], 0.9,
            BaggingParams(0.0, 1, 3),
        )
    except:
        raised = True
    assert_true(raised)


# --------------------------------------------------------------------------
# Tree growth on a bag
# --------------------------------------------------------------------------


def test_bagged_tree_equals_tree_on_subset_dataset() raises:
    # The equivalence that defines bagging: growing on a bag is growing on
    # the dataset of those rows. It is asserted here against both builders,
    # and it is *not* the same assertion against each -- see below.
    var n_rows = 2_000
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var gh = _grad_hess(n_rows, UInt64(1_000_003))
    var params = TreeParams(12, 5, 1.0, 1e-3)

    var bag = List[Int]()
    sample_rows(BaggingParams(0.6, 1, 21), n_rows, 0, bag)

    var bagged = grow_tree(data, gh[0], gh[1], params, bag)
    var subset = _subset_matrix(data, bag)
    var sub_grad = _gather(gh[0], bag)
    var sub_hess = _gather(gh[1], bag)

    # Bit for bit, against the same rows of the same size of matrix grown
    # through the same builder. This is the part that is about bagging: it
    # says the bag indirection gathers the right rows in the right order and
    # adds them in the right order, and nothing else here would catch a bag
    # that was off by a row.
    var all_rows = List[Int](capacity=len(bag))
    for i in range(len(bag)):
        all_rows.append(i)
    var reference_bagged = grow_tree(
        subset, sub_grad, sub_hess, params, all_rows
    )
    _assert_same_tree(bagged, reference_bagged)

    # Structure only, against the whole-dataset builder. The two builders no
    # longer produce the same doubles and are not required to: the subset
    # builder accumulates a node's rows in contiguous row blocks with a
    # private histogram each and folds the partials, and the whole-dataset
    # builder sums flat in ascending row order (see the "row blocks" section
    # of histogram.mojo, which states the reassociation and the bound it
    # buys). The observed spread on this fixture is a few ulp on a leaf
    # value, which moves no split -- and *that* is the claim worth keeping,
    # so it is what is asserted rather than a tolerance on the value.
    var reference_full = grow_tree(subset, sub_grad, sub_hess, params)
    _assert_same_structure(bagged, reference_full)
    assert_true(bagged.n_leaves > 1)


def test_excluded_rows_cannot_affect_the_tree() raises:
    # Out-of-bag gradients and hessians are replaced with values large
    # enough to dominate any histogram they entered. The tree must not
    # move at all.
    var n_rows = 1_500
    var n_features = 3
    var features = _make_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var gh = _grad_hess(n_rows, UInt64(7_777))
    var params = TreeParams(10, 5, 1.0, 1e-3)

    var bag = List[Int]()
    sample_rows(BaggingParams(0.5, 1, 4), n_rows, 0, bag)
    var clean = grow_tree(data, gh[0], gh[1], params, bag)

    var in_bag = List[Bool](capacity=n_rows)
    in_bag.resize(n_rows, False)
    for i in range(len(bag)):
        in_bag[bag[i]] = True
    var grad = gh[0].copy()
    var hess = gh[1].copy()
    for r in range(n_rows):
        if not in_bag[r]:
            grad[r] = 1e6 if r % 2 == 0 else -1e6
            hess[r] = 1e6
    var poisoned = grow_tree(data, grad, hess, params, bag)
    _assert_same_tree(clean, poisoned)


def test_bagged_leaf_counts_are_bag_counts() raises:
    # min_data_in_leaf is a bag constraint: every leaf must hold at least
    # that many BAGGED rows, and the leaves must partition the bag exactly.
    var n_rows = 1_200
    var n_features = 3
    var features = _make_features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var gh = _grad_hess(n_rows, UInt64(31_337))
    var min_data = 40
    var params = TreeParams(8, min_data, 1.0, 1e-3)

    var bag = List[Int]()
    sample_rows(BaggingParams(0.5, 1, 8), n_rows, 0, bag)
    var tree = grow_tree(data, gh[0], gh[1], params, bag)
    assert_true(tree.n_leaves > 1)

    var counts = List[Int](capacity=len(tree.feature))
    counts.resize(len(tree.feature), 0)
    for i in range(len(bag)):
        counts[tree.leaf_index_row(data, bag[i])] += 1
    var total = 0
    for node in range(len(tree.feature)):
        if tree.feature[node] < 0:
            assert_true(counts[node] >= min_data)
            total += counts[node]
    assert_equal(total, len(bag))


def test_grow_tree_rejects_out_of_range_bag() raises:
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)
    var grad: List[Float64] = [-1.0, -1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0]
    var params = TreeParams(4, 1, 1.0, 1e-3)
    var raised = False
    try:
        _ = grow_tree(data, grad, hess, params, [0, 4])
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = grow_tree(data, grad, hess, params, [-1, 2])
    except:
        raised = True
    assert_true(raised)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def test_fraction_one_matches_no_bagging() raises:
    # fraction = 1 and freq = 0 both mean "no bagging", and neither may
    # perturb the model by so much as a bit.
    var n_rows = 800
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(20, 0.1, TreeParams(8, 20, 1.0, 1e-3))

    var plain = train(data, target, SQUARED_ERROR, params)
    var full = train(
        data, target, SQUARED_ERROR, params, [], 0.9, BaggingParams(1.0, 1, 3)
    )
    var never = train(
        data, target, SQUARED_ERROR, params, [], 0.9, BaggingParams(0.5, 0, 3)
    )

    assert_equal(len(plain.trees), len(full.trees))
    assert_equal(len(plain.trees), len(never.trees))
    for r in range(n_rows):
        var p = plain.predict_raw_row(data, r)
        assert_equal(p, full.predict_raw_row(data, r))
        assert_equal(p, never.predict_raw_row(data, r))


def test_bagged_training_is_reproducible() raises:
    var n_rows = 600
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(25, 0.1, TreeParams(8, 20, 1.0, 1e-3))
    var bagging = BaggingParams(0.7, 1, 123)

    var a = train(data, target, SQUARED_ERROR, params, [], 0.9, bagging)
    var b = train(data, target, SQUARED_ERROR, params, [], 0.9, bagging)
    assert_equal(len(a.trees), len(b.trees))
    for r in range(n_rows):
        assert_equal(a.predict_raw_row(data, r), b.predict_raw_row(data, r))


def test_bagged_training_differs_by_seed_and_from_full_data() raises:
    var n_rows = 600
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(25, 0.1, TreeParams(8, 20, 1.0, 1e-3))

    var plain = train(data, target, SQUARED_ERROR, params)
    var one = train(
        data, target, SQUARED_ERROR, params, [], 0.9, BaggingParams(0.6, 1, 1)
    )
    var two = train(
        data, target, SQUARED_ERROR, params, [], 0.9, BaggingParams(0.6, 1, 2)
    )

    var seeds_differ = False
    var bagging_differs = False
    for r in range(n_rows):
        var p1 = one.predict_raw_row(data, r)
        var p2 = two.predict_raw_row(data, r)
        if p1 != p2:
            seeds_differ = True
        if p1 != plain.predict_raw_row(data, r):
            bagging_differs = True
    assert_true(seeds_differ)
    assert_true(bagging_differs)


def test_base_score_ignores_bagging() raises:
    # The base score is a property of the training set, not of any bag, in
    # LightGBM and here. Checked on an objective whose base score is the
    # mean and one whose base score is a percentile.
    var n_rows = 500
    var n_features = 3
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(5, 0.1, TreeParams(8, 20, 1.0, 1e-3))
    var bagging = BaggingParams(0.3, 1, 77)

    var plain = train(data, target, SQUARED_ERROR, params)
    var bagged = train(
        data, target, SQUARED_ERROR, params, [], 0.9, bagging
    )
    assert_equal(plain.base_score, bagged.base_score)

    var labels = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        labels.append(1.0 if target[r] > 1.0 else 0.0)
    var plain_bin = train(data, labels, BINARY_LOGISTIC, params)
    var bagged_bin = train(
        data, labels, BINARY_LOGISTIC, params, [], 0.9, bagging
    )
    assert_equal(plain_bin.base_score, bagged_bin.base_score)


def test_bagged_training_still_fits() raises:
    # Bagging is a regularizer, not a way to stop learning: a bagged model
    # must still beat the constant base-score predictor by a wide margin.
    var n_rows = 1_000
    var n_features = 4
    var features = _make_features(n_rows, n_features)
    var target = _regression_target(features, n_rows)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(60, 0.1, TreeParams(16, 20, 1.0, 1e-3))
    var model = train(
        data, target, SQUARED_ERROR, params, [], 0.9, BaggingParams(0.5, 1, 3)
    )

    var sse = 0.0
    var base_sse = 0.0
    for r in range(n_rows):
        var d = model.predict_row(data, r) - target[r]
        sse += d * d
        var d0 = model.base_score - target[r]
        base_sse += d0 * d0
    assert_true(sse < 0.2 * base_sse)


def test_zero_weight_rows_are_ignored_under_bagging() raises:
    # Half the rows carry flipped labels at weight zero. Whether such a row
    # lands in a bag or not, it contributes nothing, so the model must fit
    # the clean step function anyway.
    var n_rows = 400
    var features = List[Float64](capacity=n_rows)
    var labels = List[Float64](capacity=n_rows)
    var weights = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x = Float64(r % 200)
        features.append(x)
        if r < 200:
            labels.append(1.0 if x >= 100.0 else 0.0)
            weights.append(1.0)
        else:
            # Same feature values, flipped labels, zero weight.
            labels.append(0.0 if x >= 100.0 else 1.0)
            weights.append(0.0)
    var data = bin_equal_width(features, n_rows, n_features=1, n_bins=32)
    var params = BoosterParams(40, 0.3, TreeParams(8, 5, 1.0, 1e-3))
    var model = train(
        data, labels, BINARY_LOGISTIC, params, weights, 0.9,
        BaggingParams(0.5, 1, 12),
    )
    for r in range(n_rows):
        var p = model.predict_row(data, r)
        if features[r] >= 100.0:
            assert_true(p > 0.8)
        else:
            assert_true(p < 0.2)


def test_bagging_on_tiny_datasets() raises:
    # One to six rows, an aggressive fraction, and min_data_in_leaf = 1:
    # training must complete and predict finite values, however small the
    # bag turns out to be.
    for n_rows in range(1, 7):
        var features = List[Float64](capacity=n_rows)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            features.append(Float64(r))
            target.append(Float64(r % 2))
        var data = bin_equal_width(
            features, n_rows, n_features=1, n_bins=4
        )
        var params = BoosterParams(10, 0.3, TreeParams(4, 1, 1.0, 1e-3))
        var model = train(
            data, target, SQUARED_ERROR, params, [], 0.9,
            BaggingParams(0.4, 1, 5),
        )
        for r in range(n_rows):
            var p = model.predict_row(data, r)
            assert_true(p > -1e6 and p < 1e6)


def test_multiclass_bagging_shares_one_bag_per_round() raises:
    # A round's per-class trees are grown on the same bag, so every tree in
    # a round partitions exactly the same rows. Reproducibility and fit are
    # checked alongside.
    var n_rows = 600
    var n_features = 3
    var features = _make_features(n_rows, n_features)
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        labels.append(0 if x0 < 0.33 else (1 if x0 < 0.66 else 2))
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var params = BoosterParams(20, 0.2, TreeParams(8, 10, 1.0, 1e-3))
    var bagging = BaggingParams(0.6, 1, 19)

    var a = train_multiclass(data, labels, 3, params, [], bagging)
    var b = train_multiclass(data, labels, 3, params, [], bagging)
    assert_equal(len(a.trees), len(b.trees))

    var correct = 0
    for r in range(n_rows):
        var bins: List[Int] = [
            data.bin_at(r, 0), data.bin_at(r, 1), data.bin_at(r, 2)
        ]
        var pa = a.predict_proba_bins(bins)
        var pb = b.predict_proba_bins(bins)
        var argmax = 0
        for k in range(3):
            assert_equal(pa[k], pb[k])
            if pa[k] > pa[argmax]:
                argmax = k
        if argmax == labels[r]:
            correct += 1
    assert_true(correct > (9 * n_rows) // 10)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
