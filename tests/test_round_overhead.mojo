"""Per-round CPU overhead: the score update, the grower's scratch, and batch
prediction.

Three changes are checked here, and all three claim the same thing: they
change what a round costs and nothing about what it produces. So every test
in this file is an equality, and the equalities are exact. Nothing here
measures anything; a faster route is not what is being asserted, only that
the route arrives at the same numbers.

1. The score update. A round used to end by walking the whole tree once per
   training row to recover the leaf that row was already partitioned into.
   It now adds `learning_rate * value` to each leaf's own rows. The two are
   compared inside one build, by forcing the old route with
   `MOJOTREES_LEAF_SCORE_UPDATE=0`, and the comparison is on the trained
   ensembles rather than on one prediction: raw scores feed the next round's
   gradients, so a one-ulp difference in the update would show up as a
   different tree a few rounds later, not as a different last digit.

2. The grower's scratch. The histogram pool and the gradient/hessian gather
   buffer are now the caller's, so a booster holds one of each for a whole
   fit instead of building them per tree and per node. `histogram.mojo`'s own
   docstring promises the scratch form of the subset builder is identical to
   the allocating form; that promise is checked directly against a buffer
   deliberately left holding another node's contents, rather than assumed.

3. Batch prediction. The per-row loop moved from `model.mojo` onto the
   boosters and now runs over row blocks. Each row writes one output slot and
   reads nothing another row writes, so this one is identical by
   construction; the test still checks it against a serial reference, at a
   batch large enough to cross the parallel grain and again with the worker
   count forced.

`MOJOTREES_NUM_WORKERS` is set and cleared around the tests that need a
particular task count, so the file leaves the environment as it found it.
"""

from std.os import setenv
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BaggingParams,
    BinnedMatrix,
    Booster,
    BoosterParams,
    MulticlassBooster,
    Tree,
    TreeParams,
    bin_equal_width,
    train,
    train_multiclass,
)
from mojotrees.boosting import (
    IterationRange,
    _add_by_leaf,
    _add_by_traversal,
)
from mojotrees.cegb import CegbLedger
from mojotrees.histogram import (
    Histogram,
    build_histogram_subset_into,
    build_histogram_subset_into_scratch,
)
from mojotrees.model import Model, MulticlassModel, fit, fit_multiclass
from mojotrees.tree import (
    GrowScratch,
    LeafMembership,
    grow_tree_leaves,
    grow_tree_with_cegb,
)
from support import _make_features, _uniform


def _same_bits(a: Float64, b: Float64) -> Bool:
    """Bit equality, not numeric equality.

    `==` would call +0.0 and -0.0 equal, and the claim being tested is that
    the same Float64 reaches the same accumulator, so the bits are what to
    compare."""
    return a.to_bits() == b.to_bits()


def _assert_same_float(a: Float64, b: Float64) raises:
    assert_true(_same_bits(a, b))


def _assert_same_tree(a: Tree, b: Tree) raises:
    """Every array of the flat tree, bit for bit."""
    assert_equal(a.n_leaves, b.n_leaves)
    assert_equal(len(a.feature), len(b.feature))
    assert_equal(len(a.cat_bitset), len(b.cat_bitset))
    for i in range(len(a.feature)):
        assert_equal(a.feature[i], b.feature[i])
        assert_equal(a.threshold_bin[i], b.threshold_bin[i])
        assert_equal(a.left[i], b.left[i])
        assert_equal(a.right[i], b.right[i])
        assert_equal(a.missing_bin[i], b.missing_bin[i])
        assert_equal(a.cat_offset[i], b.cat_offset[i])
        assert_equal(a.default_left[i], b.default_left[i])
        _assert_same_float(a.value[i], b.value[i])
        _assert_same_float(a.count[i], b.count[i])
        _assert_same_float(a.split_gain[i], b.split_gain[i])
    for i in range(len(a.cat_bitset)):
        assert_equal(a.cat_bitset[i], b.cat_bitset[i])


def _assert_same_booster(
    a: Booster, b: Booster, data: BinnedMatrix
) raises:
    assert_equal(len(a.trees), len(b.trees))
    _assert_same_float(a.base_score, b.base_score)
    for t in range(len(a.trees)):
        _assert_same_tree(a.trees[t], b.trees[t])
    for r in range(data.n_rows):
        _assert_same_float(
            a.predict_raw_row(data, r), b.predict_raw_row(data, r)
        )


def _assert_same_multiclass(
    a: MulticlassBooster, b: MulticlassBooster, data: BinnedMatrix
) raises:
    assert_equal(len(a.trees), len(b.trees))
    assert_equal(a.n_classes, b.n_classes)
    for k in range(a.n_classes):
        _assert_same_float(a.base_scores[k], b.base_scores[k])
    for t in range(len(a.trees)):
        _assert_same_tree(a.trees[t], b.trees[t])


def _features(n_rows: Int, n_features: Int) -> List[Float64]:
    return _make_features(n_rows, n_features)


def _data(n_rows: Int, n_features: Int) raises -> BinnedMatrix:
    return bin_equal_width(
        _features(n_rows, n_features), n_rows, n_features, 32
    )


def _regression_target(
    features: List[Float64], n_rows: Int
) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        y.append(3.0 * x0 - 2.0 * x1 + (x2 - 0.5) * (x2 - 0.5))
    return y^


def _binary_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(
            1.0 if features[0 * n_rows + r] + features[1 * n_rows + r] > 1.0
            else 0.0
        )
    return y^


def _labels(features: List[Float64], n_rows: Int, n_classes: Int) -> List[Int]:
    var out = List[Int](capacity=n_rows)
    for r in range(n_rows):
        var v = features[0 * n_rows + r] * Float64(n_classes)
        var k = Int(v)
        out.append(k if k < n_classes else n_classes - 1)
    return out^


def _params(n_rounds: Int, num_leaves: Int = 8) -> BoosterParams:
    return BoosterParams(
        n_rounds, 0.1, TreeParams(num_leaves, 5, 1.0, 1e-3)
    )


def _force_traversal():
    _ = setenv("MOJOTREES_LEAF_SCORE_UPDATE", "0")


def _allow_leaf_update():
    _ = setenv("MOJOTREES_LEAF_SCORE_UPDATE", "")


def test_leaf_update_matches_traversal_regression() raises:
    """The headline equality on the plain path: no sampler, every row in some
    leaf, so the leaf route is the one actually taken."""
    var n = 600
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var y = _regression_target(f, n)
    var params = _params(25)

    _force_traversal()
    var reference = train(data, y, SQUARED_ERROR, params)
    _allow_leaf_update()
    var updated = train(data, y, SQUARED_ERROR, params)

    assert_equal(len(updated.trees), 25)
    _assert_same_booster(reference, updated, data)


def test_leaf_update_matches_traversal_binary() raises:
    var n = 600
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var y = _binary_target(f, n)
    var params = _params(25)

    _force_traversal()
    var reference = train(data, y, BINARY_LOGISTIC, params)
    _allow_leaf_update()
    var updated = train(data, y, BINARY_LOGISTIC, params)

    _assert_same_booster(reference, updated, data)


def test_leaf_update_matches_traversal_multiclass() raises:
    """One tree per class per round, each writing its own stride of the
    row-major raw scores."""
    var n = 600
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var labels = _labels(f, n, 3)
    var params = _params(15)

    _force_traversal()
    var reference = train_multiclass(data, labels, 3, params)
    _allow_leaf_update()
    var updated = train_multiclass(data, labels, 3, params)

    assert_equal(len(updated.trees), 45)
    _assert_same_multiclass(reference, updated, data)


def test_leaf_update_matches_traversal_under_bagging() raises:
    """Bagging is the path that keeps the traversal: the tree is grown on a
    sample and the rows outside it have no leaf. The two runs must still
    agree, which is what says the fallback is taken rather than the sampled
    rows being updated and the rest quietly skipped."""
    var n = 600
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var y = _regression_target(f, n)
    var params = _params(20)
    var bagging = BaggingParams(0.6, 1, 7)

    _force_traversal()
    var reference = train(
        data, y, SQUARED_ERROR, params, bagging=bagging
    )
    _allow_leaf_update()
    var updated = train(data, y, SQUARED_ERROR, params, bagging=bagging)

    _assert_same_booster(reference, updated, data)
    # And the bagged fit is not accidentally the unbagged one, which would
    # make the comparison above vacuous.
    var unbagged = train(data, y, SQUARED_ERROR, params)
    var differs = False
    for r in range(n):
        if not _same_bits(
            updated.predict_raw_row(data, r),
            unbagged.predict_raw_row(data, r),
        ):
            differs = True
    assert_true(differs)


def test_leaf_update_matches_traversal_single_leaf_stop() raises:
    """A constant target gives a first tree with one leaf and a value of
    zero, which stops the run. The membership then holds the root alone and
    covers every row; the early-stop branch must fire identically either
    way."""
    var n = 200
    var f = _features(n, 3)
    var data = bin_equal_width(f, n, 3, 32)
    var y = List[Float64](capacity=n)
    for _ in range(n):
        y.append(2.5)
    var params = _params(10)

    _force_traversal()
    var reference = train(data, y, SQUARED_ERROR, params)
    _allow_leaf_update()
    var updated = train(data, y, SQUARED_ERROR, params)

    # Nothing to learn, so the run stopped before appending a tree.
    assert_equal(len(reference.trees), 0)
    _assert_same_booster(reference, updated, data)


def test_leaf_update_matches_traversal_forced_parallel() raises:
    """The same equality with the task count forced, so both sides run over
    several blocks rather than falling through the grain to the serial
    path."""
    var n = 600
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var y = _regression_target(f, n)
    var params = _params(20)

    _ = setenv("MOJOTREES_NUM_WORKERS", "3")
    _force_traversal()
    var reference = train(data, y, SQUARED_ERROR, params)
    _allow_leaf_update()
    var updated = train(data, y, SQUARED_ERROR, params)
    _ = setenv("MOJOTREES_NUM_WORKERS", "")

    _assert_same_booster(reference, updated, data)


def test_leaf_membership_partitions_the_rows() raises:
    """What the grower now hands back, checked as a partition rather than
    through a trained model: every row in exactly one leaf, each leaf's rows
    ascending, and every row's leaf value equal to what walking the tree for
    that row returns."""
    var n = 500
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for r in range(n):
        grad.append(2.0 * _uniform(UInt64(r) + 11) - 1.0)
        hess.append(_uniform(UInt64(r) + 4001) + 0.5)

    var leaves = LeafMembership()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    var ledger = CegbLedger.none()
    var tree = grow_tree_leaves(
        leaves,
        ledger,
        scratch,
        data,
        grad,
        hess,
        TreeParams(12, 5, 1.0, 1e-3),
    )

    assert_true(leaves.covers_all_rows)
    assert_equal(leaves.n_leaves(), tree.n_leaves)

    var seen = List[Int](capacity=n)
    seen.resize(n, 0)
    var total = 0
    for l in range(leaves.n_leaves()):
        # A leaf, not an internal node.
        assert_equal(tree.feature[leaves.node[l]], -1)
        var rows = leaves.rows[l].copy()
        assert_true(len(rows) > 0)
        for i in range(len(rows)):
            if i > 0:
                assert_true(rows[i - 1] < rows[i])
            seen[rows[i]] += 1
            total += 1
            # The whole claim of the leaf update in one line.
            _assert_same_float(
                tree.value[leaves.node[l]], tree.predict_row(data, rows[i])
            )
    assert_equal(total, n)
    for r in range(n):
        assert_equal(seen[r], 1)


def test_both_updates_match_the_hand_written_previous_update() raises:
    """The before-and-after at the level of the update itself, with the
    "before" written out here rather than reached through a build flag.

    `raw[r] += learning_rate * tree.predict_row(data, r)` is the line every
    trainer ended a round with, and the trainers this lane did not touch still
    end their rounds with it. It compiles to a fused multiply-add, so the
    product is never rounded on its own; a leaf update that hoists
    `learning_rate * value` out of the loop rounds it and lands one ulp away
    on a large fraction of rows, which feeds the next round's gradients and
    changes the second tree. That is not a hypothetical: it is what the first
    version of `_add_by_leaf` did, and this assertion is what caught it.
    """
    var n = 600
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for r in range(n):
        grad.append(2.0 * _uniform(UInt64(r) + 501) - 1.0)
        hess.append(_uniform(UInt64(r) + 60001) + 0.5)

    var leaves = LeafMembership()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    var ledger = CegbLedger.none()
    var tree = grow_tree_leaves(
        leaves,
        ledger,
        scratch,
        data,
        grad,
        hess,
        TreeParams(8, 5, 1.0, 1e-3),
    )

    var learning_rate = 0.1
    var start = List[Float64](capacity=n)
    for r in range(n):
        start.append(0.25 + 0.001 * Float64(r))

    var previous = start.copy()
    for r in range(n):
        previous[r] += learning_rate * tree.predict_row(data, r)

    var traversed = start.copy()
    _add_by_traversal(traversed, tree, data, learning_rate, 1, 0)

    var by_leaf = start.copy()
    _add_by_leaf(by_leaf, tree, leaves, learning_rate, n, 1, 0)

    for r in range(n):
        _assert_same_float(previous[r], traversed[r])
        _assert_same_float(previous[r], by_leaf[r])

    # And with a class stride, the shape multiclass uses.
    var n_classes = 3
    var strided_start = List[Float64](capacity=n * n_classes)
    for r in range(n):
        for k in range(n_classes):
            strided_start.append(0.5 * Float64(k) + 0.001 * Float64(r))
    var strided_previous = strided_start.copy()
    for r in range(n):
        strided_previous[r * n_classes + 1] += (
            learning_rate * tree.predict_row(data, r)
        )
    var strided_leaf = strided_start.copy()
    _add_by_leaf(strided_leaf, tree, leaves, learning_rate, n, n_classes, 1)
    for i in range(n * n_classes):
        _assert_same_float(strided_previous[i], strided_leaf[i])


def test_leaf_membership_covers_the_bag_only_under_bagging() raises:
    """Grown on a bag, the membership names the bag and says so. A caller
    that took the leaf route here would leave every unsampled row
    unscored."""
    var n = 400
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for r in range(n):
        grad.append(2.0 * _uniform(UInt64(r) + 77) - 1.0)
        hess.append(1.0)
    var bag = List[Int]()
    for r in range(0, n, 2):
        bag.append(r)

    var leaves = LeafMembership()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    var ledger = CegbLedger.none()
    var tree = grow_tree_leaves(
        leaves,
        ledger,
        scratch,
        data,
        grad,
        hess,
        TreeParams(8, 5, 1.0, 1e-3),
        bag,
    )

    assert_true(not leaves.covers_all_rows)
    var total = 0
    for l in range(leaves.n_leaves()):
        total += len(leaves.rows[l])
    assert_equal(total, len(bag))
    assert_equal(leaves.n_leaves(), tree.n_leaves)


def test_shared_scratch_grows_the_same_trees() raises:
    """(2b) and (2a) together at the grower's own level: one scratch reused
    across several trees against a fresh one per tree. A pooled histogram and
    a reused gather buffer both come back holding a previous node's contents,
    so this is where a builder that read before it wrote would show up."""
    var n = 500
    var f = _features(n, 5)
    var data = bin_equal_width(f, n, 5, 32)
    var params = TreeParams(10, 5, 1.0, 1e-3)

    var shared = GrowScratch(data.n_features, data.n_bins)
    for t in range(6):
        var grad = List[Float64](capacity=n)
        var hess = List[Float64](capacity=n)
        for r in range(n):
            grad.append(2.0 * _uniform(UInt64(t * n + r) + 5) - 1.0)
            hess.append(_uniform(UInt64(t * n + r) + 9001) + 0.5)

        var leaves = LeafMembership()
        var ledger = CegbLedger.none()
        var reused = grow_tree_leaves(
            leaves, ledger, shared, data, grad, hess, params, [], t
        )
        # `grow_tree_with_cegb` builds a fresh scratch per call, which is what
        # growth did before the scratch was threaded at all.
        var fresh_ledger = CegbLedger.none()
        var fresh = grow_tree_with_cegb(
            data, grad, hess, params, fresh_ledger, [], t
        )
        _assert_same_tree(fresh, reused)


def test_shared_scratch_across_matrix_shapes() raises:
    """`GrowScratch.prepare` must drop pooled buffers of the wrong shape. A
    scratch that carried them over would hand a builder a histogram too small
    for the new feature count."""
    var wide = _data(300, 6)
    var narrow = _data(300, 2)
    var params = TreeParams(6, 5, 1.0, 1e-3)
    var grad = List[Float64](capacity=300)
    var hess = List[Float64](capacity=300)
    for r in range(300):
        grad.append(2.0 * _uniform(UInt64(r) + 31) - 1.0)
        hess.append(1.0)

    var shared = GrowScratch(wide.n_features, wide.n_bins)
    var leaves = LeafMembership()
    var ledger = CegbLedger.none()
    _ = grow_tree_leaves(leaves, ledger, shared, wide, grad, hess, params)
    var after_wide = grow_tree_leaves(
        leaves, ledger, shared, narrow, grad, hess, params
    )
    var fresh_ledger = CegbLedger.none()
    var reference = grow_tree_with_cegb(
        narrow, grad, hess, params, fresh_ledger
    )
    _assert_same_tree(reference, after_wide)


def test_subset_histogram_scratch_matches_allocating_form() raises:
    """(2a) directly: the promise `build_histogram_subset_into`'s docstring
    makes, checked rather than taken.

    The scratch buffer is handed over holding a larger node's gradients, so
    if the builder ever read a cell it had not written for this node, these
    two histograms would differ.
    """
    var n = 400
    var data = _data(n, 4)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for r in range(n):
        grad.append(2.0 * _uniform(UInt64(r) + 101) - 1.0)
        hess.append(_uniform(UInt64(r) + 7777) + 0.25)

    var big = List[Int](capacity=n)
    for r in range(n):
        big.append(r)
    var small = List[Int]()
    for r in range(0, n, 3):
        small.append(r)

    # Dirty the buffer with the big node, then reuse it for the small one.
    var pairs = List[Float64]()
    var warm = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_subset_into_scratch(
        warm, pairs, data, grad, hess, big, 0, len(big)
    )

    var reused = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_subset_into_scratch(
        reused, pairs, data, grad, hess, small, 0, len(small)
    )
    var allocating = Histogram.zeroed(data.n_features, data.n_bins)
    build_histogram_subset_into(
        allocating, data, grad, hess, small, 0, len(small)
    )

    assert_equal(reused.n_cells(), allocating.n_cells())
    for i in range(allocating.n_cells()):
        _assert_same_float(allocating.grad_at(i), reused.grad_at(i))
        _assert_same_float(allocating.hess_at(i), reused.hess_at(i))
        assert_equal(allocating.count_at(i), reused.count_at(i))


def _serial_predict_reference(
    model: Model, data: BinnedMatrix, rng: IterationRange, raw_score: Bool
) raises -> List[Float64]:
    """The loop `Model.predict_batch` ran before it was split into blocks,
    written out here so the parallel result has something to be equal to."""
    var out = List[Float64](capacity=data.n_rows)
    var bins = List[Int](capacity=data.n_features)
    for r in range(data.n_rows):
        bins.clear()
        for f in range(data.n_features):
            bins.append(Int(data.bins[f * data.n_rows + r]))
        if raw_score:
            out.append(model.booster.predict_raw_bins_range(bins, rng))
        else:
            out.append(model.booster.predict_bins_range(bins, rng))
    return out^


def test_batch_prediction_matches_serial_reference() raises:
    """A batch big enough that the block plan is not the serial one: 4000
    rows over 8 features and 20 iterations is far past the default grain, and
    the worker count is forced afterwards so the multi-block path is exercised
    whatever the machine decides."""
    var n_train = 800
    var n_features = 8
    var train_f = _features(n_train, n_features)
    var y = _regression_target(train_f, n_train)
    var model = fit(
        train_f,
        n_train,
        n_features,
        y,
        SQUARED_ERROR,
        _params(20),
    )

    var n_batch = 4000
    var batch = _features(n_batch, n_features)
    var binned = model.mapper.transform(batch, n_batch)
    var rng = IterationRange(0, model.n_iterations())

    var expected_raw = _serial_predict_reference(model, binned, rng, True)
    var expected_response = _serial_predict_reference(
        model, binned, rng, False
    )

    var got_raw = model.predict_batch(batch, n_batch, rng, True)
    var got_response = model.predict_batch(batch, n_batch, rng, False)
    assert_equal(len(got_raw), n_batch)
    for r in range(n_batch):
        _assert_same_float(expected_raw[r], got_raw[r])
        _assert_same_float(expected_response[r], got_response[r])

    _ = setenv("MOJOTREES_NUM_WORKERS", "5")
    var forced_raw = model.predict_batch(batch, n_batch, rng, True)
    var forced_response = model.predict_batch(batch, n_batch, rng, False)
    _ = setenv("MOJOTREES_NUM_WORKERS", "")
    for r in range(n_batch):
        _assert_same_float(expected_raw[r], forced_raw[r])
        _assert_same_float(expected_response[r], forced_response[r])

    # A truncated range goes down the same path and must slice the same way.
    var half = IterationRange(0, model.n_iterations() // 2)
    var expected_half = _serial_predict_reference(model, binned, half, True)
    var got_half = model.predict_batch(batch, n_batch, half, True)
    for r in range(n_batch):
        _assert_same_float(expected_half[r], got_half[r])


def test_multiclass_batch_prediction_matches_serial_reference() raises:
    """The row-major multiclass batch, including the per-row softmax, which
    must stay inside its own row when rows are handed to different blocks."""
    var n_train = 800
    var n_features = 6
    var n_classes = 3
    var train_f = _features(n_train, n_features)
    var labels = _labels(train_f, n_train, n_classes)
    var model = fit_multiclass(
        train_f, n_train, n_features, labels, n_classes, _params(12)
    )

    var n_batch = 3000
    var batch = _features(n_batch, n_features)
    var binned = model.mapper.transform(batch, n_batch)
    var rng = IterationRange(0, model.n_iterations())

    var expected = List[Float64](capacity=n_batch * n_classes)
    var expected_raw = List[Float64](capacity=n_batch * n_classes)
    var bins = List[Int](capacity=n_features)
    for r in range(n_batch):
        bins.clear()
        for f in range(n_features):
            bins.append(Int(binned.bins[f * n_batch + r]))
        var proba = model.booster.predict_proba_bins_range(bins, rng)
        var raw = model.booster.predict_raw_bins_range(bins, rng)
        for k in range(n_classes):
            expected.append(proba[k])
            expected_raw.append(raw[k])

    _ = setenv("MOJOTREES_NUM_WORKERS", "5")
    var got = model.predict_batch(batch, n_batch, rng, False)
    var got_raw = model.predict_batch(batch, n_batch, rng, True)
    _ = setenv("MOJOTREES_NUM_WORKERS", "")

    assert_equal(len(got), n_batch * n_classes)
    for i in range(n_batch * n_classes):
        _assert_same_float(expected[i], got[i])
        _assert_same_float(expected_raw[i], got_raw[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
