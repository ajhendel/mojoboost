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

4. The two things the round loop does with the membership it now keeps. The
   score update is split by *rows* rather than by leaves, so a round is no
   longer as long as its single largest leaf; and leaf renewal (L1, QUANTILE,
   MAPE) reads each row's leaf off the membership instead of walking the tree
   once per row. Both are equalities again and both are checked against the
   route they replaced. The schedule half of the first is proved rather than
   assumed: the fixture grows a deliberately lopsided tree and the test
   asserts that a block boundary really did fall inside a leaf, which is the
   thing the old leaf-wise split could never do. The renewal half asserts the
   gate that chooses the route, in both directions.

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
    _LEAF_ROW_OPS,
    _add_by_leaf,
    _add_by_traversal,
    _renew_leaf_values,
    _renewal_membership_usable,
)
from mojotrees.parallel import plan_row_blocks
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
from mojotrees.tree_parameters_extra import ExtraTreeParams
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


def _skewed_tree(
    mut leaves: LeafMembership,
    mut scratch: GrowScratch,
    data: BinnedMatrix,
    n: Int,
    num_leaves: Int,
) raises -> Tree:
    """A tree whose leaves are deliberately of very unequal size.

    The gradient is a spike. Rows in the top bins of feature 0 carry a large
    and varying gradient; every other row carries exactly zero. A leaf holding
    only zero-gradient rows has zero gain on every split of it, so growth
    refuses to cut the bulk and spends its whole budget subdividing the spike,
    leaving one leaf with the large majority of the dataset in it. That is the
    shape the row-blocked score update exists for, and a fixture with sixteen
    even leaves would not exercise it.
    """
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for r in range(n):
        var x0 = data.bin_at(r, 0)
        grad.append(
            4.0 + 2.0 * _uniform(UInt64(r) + 13) if x0 >= 28 else 0.0
        )
        hess.append(1.0)
    var ledger = CegbLedger.none()
    return grow_tree_leaves(
        leaves, ledger, scratch, data, grad, hess,
        TreeParams(num_leaves, 8, 1.0, 1e-9),
    )


def _leaf_sizes(leaves: LeafMembership) raises -> List[Int]:
    var out = List[Int](capacity=leaves.n_leaves())
    for l in range(leaves.n_leaves()):
        out.append(len(leaves.rows[l]))
    return out^


def test_leaf_update_cuts_inside_a_leaf_and_moves_no_bits() raises:
    """The score update is split by rows, not by leaves, and the split is
    proved to have landed inside a leaf rather than assumed to have.

    Splitting by leaf made the round's update as long as its largest leaf.
    This fixture has one leaf holding a large majority of the rows, so a
    leaf-wise split could not have used more than a couple of workers on it
    however many were asked for. The assertions are in two halves:

    1. The schedule. `_add_by_leaf` splits [0, total) -- the concatenation of
       the leaves' row lists -- with `plan_row_blocks`, so the same call with
       the same two numbers reproduces its block geometry exactly. At least
       one block boundary is asserted to fall strictly inside a leaf, which is
       the thing the old schedule could never do and is therefore the proof
       that the new path ran. It is asserted, not assumed: if the fixture ever
       stopped producing an uneven tree this test would fail rather than pass
       for nothing.
    2. The bits. The same update written out by hand, serially, one row at a
       time in row order, compared with `to_bits()` and no tolerance.
    """
    var n = 4000
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)

    _ = setenv("MOJOTREES_NUM_WORKERS", "8")
    var leaves = LeafMembership()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    var tree = _skewed_tree(leaves, scratch, data, n, 16)

    var sizes = _leaf_sizes(leaves)
    var total = 0
    var largest = 0
    for l in range(len(sizes)):
        total += sizes[l]
        if sizes[l] > largest:
            largest = sizes[l]
    assert_equal(total, n)
    # The fixture has to be lopsided for this test to mean anything: one leaf
    # holding at least a fifth of the rows is already more than eight even
    # workers' share, so a leaf-wise split of it could not have been balanced.
    assert_true(largest * 5 > n)

    # The block geometry `_add_by_leaf` will use, from the same helper with
    # the same arguments. `MOJOTREES_NUM_WORKERS=8` is set, so this is eight
    # blocks whatever machine it runs on.
    var blocks = plan_row_blocks(total, n * _LEAF_ROW_OPS)
    assert_equal(blocks.n_blocks, 8)

    # Leaf offsets in the concatenation: `starts[l]` is where leaf l begins.
    var starts = List[Int](capacity=len(sizes) + 1)
    starts.append(0)
    var acc = 0
    for l in range(len(sizes)):
        acc += sizes[l]
        starts.append(acc)

    var cuts_inside_a_leaf = 0
    for b in range(1, blocks.n_blocks):
        var boundary = blocks.start(b)
        var on_a_leaf_edge = False
        for l in range(len(starts)):
            if starts[l] == boundary:
                on_a_leaf_edge = True
        if not on_a_leaf_edge:
            cuts_inside_a_leaf += 1
    # The whole claim of the change: a block begins in the middle of a leaf.
    assert_true(cuts_inside_a_leaf > 0)

    var learning_rate = 0.1
    var start = List[Float64](capacity=n)
    for r in range(n):
        start.append(0.25 + 0.001 * Float64(r))

    var previous = start.copy()
    for r in range(n):
        previous[r] += learning_rate * tree.predict_row(data, r)

    var by_leaf = start.copy()
    _add_by_leaf(by_leaf, tree, leaves, learning_rate, n, 1, 0)
    _ = setenv("MOJOTREES_NUM_WORKERS", "")

    for r in range(n):
        _assert_same_float(previous[r], by_leaf[r])


def test_leaf_update_identical_at_one_three_and_eight_workers() raises:
    """Determinism across the worker count, which is the round's contract.

    A row's update is one independent `fma` into its own slot, so cutting a
    leaf's rows across blocks cannot reassociate anything and the answer must
    not depend on how many blocks there are. Three worker counts against one
    serial reference, bit for bit. The strided form is included because
    multiclass rounds take it and its slot arithmetic is the one place a block
    could collide with another block's output.
    """
    var n = 4000
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)

    var leaves = LeafMembership()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    var tree = _skewed_tree(leaves, scratch, data, n, 16)

    var learning_rate = 0.1
    var start = List[Float64](capacity=n)
    for r in range(n):
        start.append(0.25 + 0.001 * Float64(r))

    var previous = start.copy()
    for r in range(n):
        previous[r] += learning_rate * tree.predict_row(data, r)

    var n_classes = 3
    var strided_start = List[Float64](capacity=n * n_classes)
    for r in range(n):
        for k in range(n_classes):
            strided_start.append(0.5 * Float64(k) + 0.001 * Float64(r))
    var strided_previous = strided_start.copy()
    for r in range(n):
        strided_previous[r * n_classes + 2] += (
            learning_rate * tree.predict_row(data, r)
        )

    var counts: List[String] = ["1", "3", "8"]
    for w in range(len(counts)):
        _ = setenv("MOJOTREES_NUM_WORKERS", counts[w])
        var got = start.copy()
        _add_by_leaf(got, tree, leaves, learning_rate, n, 1, 0)
        var strided_got = strided_start.copy()
        _add_by_leaf(
            strided_got, tree, leaves, learning_rate, n, n_classes, 2
        )
        _ = setenv("MOJOTREES_NUM_WORKERS", "")
        for r in range(n):
            _assert_same_float(previous[r], got[r])
        for i in range(n * n_classes):
            _assert_same_float(strided_previous[i], strided_got[i])


def _renewal_column(n: Int, seed: UInt64, scale: Float64, shift: Float64
) raises -> List[Float64]:
    """One deterministic Float64 column for the renewal fixtures."""
    var out = List[Float64](capacity=n)
    for r in range(n):
        out.append(shift + scale * _uniform(UInt64(r) + seed))
    return out^


def _assert_renewal_matches_traversal(
    data: BinnedMatrix,
    n: Int,
    num_leaves: Int,
    alpha: Float64,
    weighted: Bool,
    bag: List[Int],
) raises:
    """Renew the same tree twice, once off the membership and once by walking
    the tree per row, and compare every node bit for bit.

    The tree is grown twice rather than copied, from the same inputs, because
    renewal rewrites `tree.value` in place. Growth is deterministic, so the
    two start from the same tree; `_assert_same_tree` on the unrenewed pair
    would be the same assertion made twice, and the renewed pair is what
    matters.
    """
    var target = _renewal_column(n, 909, 3.0, 0.0)
    var raw = _renewal_column(n, 30011, 0.5, -0.25)
    var weights = List[Float64]()
    if weighted:
        weights = _renewal_column(n, 700001, 1.0, 0.25)

    var leaves_a = LeafMembership()
    var scratch_a = GrowScratch(data.n_features, data.n_bins)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for r in range(n):
        grad.append(2.0 * _uniform(UInt64(r) + 41) - 1.0)
        hess.append(1.0)
    var ledger_a = CegbLedger.none()
    var by_membership = grow_tree_leaves(
        leaves_a, ledger_a, scratch_a, data, grad, hess,
        TreeParams(num_leaves, 8, 1.0, 1e-9), bag,
    )

    var leaves_b = LeafMembership()
    var scratch_b = GrowScratch(data.n_features, data.n_bins)
    var ledger_b = CegbLedger.none()
    var by_traversal = grow_tree_leaves(
        leaves_b, ledger_b, scratch_b, data, grad, hess,
        TreeParams(num_leaves, 8, 1.0, 1e-9), bag,
    )
    _assert_same_tree(by_membership, by_traversal)

    var n_used = len(bag) if len(bag) > 0 else n
    # The gate, asserted rather than assumed: this is what decides which of
    # the two routes each call below takes.
    assert_true(_renewal_membership_usable(by_membership, leaves_a, n_used))

    var no_monotone = List[Int]()
    _renew_leaf_values(
        by_membership, data, target, raw, weights, alpha, bag, no_monotone,
        ExtraTreeParams(), leaves_a,
    )
    # No membership argument: the empty default fails the gate, so this one
    # walks the tree per row exactly as it always did.
    _renew_leaf_values(
        by_traversal, data, target, raw, weights, alpha, bag, no_monotone,
    )
    _assert_same_tree(by_membership, by_traversal)


def test_renewal_off_membership_matches_the_per_row_walk() raises:
    """Unweighted renewal (`_percentile`) and weighted renewal
    (`_weighted_percentile`), at three worker counts.

    The weighted form is the one that can tell the difference: it stable-sorts
    each leaf's residuals and then accumulates a weighted cdf in that order,
    so a bucket that came out permuted would give a different Float64 even
    with the same elements in it. Both routes fill each bucket in ascending
    row order, which is why they agree.

    The worker count is swept because the trees on both sides are grown under
    it, and a renewal route that somehow depended on the schedule would show
    up as a tree difference here.
    """
    var n = 2000
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var empty_bag = List[Int]()

    var counts: List[String] = ["1", "3", "8"]
    for w in range(len(counts)):
        _ = setenv("MOJOTREES_NUM_WORKERS", counts[w])
        _assert_renewal_matches_traversal(
            data, n, 12, 0.5, False, empty_bag
        )
        _assert_renewal_matches_traversal(
            data, n, 12, 0.5, True, empty_bag
        )
        # A quantile other than the median, which is the QUANTILE objective's
        # own case and interpolates between two straddling residuals.
        _assert_renewal_matches_traversal(
            data, n, 12, 0.85, True, empty_bag
        )
        _ = setenv("MOJOTREES_NUM_WORKERS", "")


def test_renewal_off_membership_matches_under_bagging() raises:
    """Renewal over a bag. The membership names the bag alone, the traversal
    route iterates the bag alone, and both fill each leaf in ascending row
    order, so the two agree here for the same reason they do on the full
    dataset. `covers_all_rows` is False on this membership and renewal does
    not care: unlike the score update it never had to reach a row outside the
    bag."""
    var n = 2000
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var bag = List[Int]()
    for r in range(0, n, 3):
        bag.append(r)

    var counts: List[String] = ["1", "3", "8"]
    for w in range(len(counts)):
        _ = setenv("MOJOTREES_NUM_WORKERS", counts[w])
        _assert_renewal_matches_traversal(data, n, 12, 0.5, False, bag)
        _assert_renewal_matches_traversal(data, n, 12, 0.5, True, bag)
        _ = setenv("MOJOTREES_NUM_WORKERS", "")


def test_renewal_gate_refuses_a_membership_that_does_not_fit() raises:
    """The gate is a shape check and it says no when the shape is wrong.

    Three refusals: the empty membership every unwired caller passes, a
    membership grown on a bag when renewal was asked for the whole dataset,
    and the whole-dataset membership when renewal was asked for a bag. Each
    would have bucketed the wrong rows, and each falls back to the traversal
    instead.
    """
    var n = 800
    var f = _features(n, 4)
    var data = bin_equal_width(f, n, 4, 32)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for r in range(n):
        grad.append(2.0 * _uniform(UInt64(r) + 91) - 1.0)
        hess.append(1.0)
    var bag = List[Int]()
    for r in range(0, n, 2):
        bag.append(r)

    var full = LeafMembership()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    var ledger = CegbLedger.none()
    var full_tree = grow_tree_leaves(
        full, ledger, scratch, data, grad, hess, TreeParams(8, 5, 1.0, 1e-3)
    )

    var bagged = LeafMembership()
    var bagged_tree = grow_tree_leaves(
        bagged, ledger, scratch, data, grad, hess,
        TreeParams(8, 5, 1.0, 1e-3), bag,
    )

    var nothing = LeafMembership()
    assert_true(not _renewal_membership_usable(full_tree, nothing, n))
    # Right tree shape, wrong row count on both sides of the bag question.
    assert_true(not _renewal_membership_usable(full_tree, full, len(bag)))
    assert_true(not _renewal_membership_usable(bagged_tree, bagged, n))
    # And the two that do fit.
    assert_true(_renewal_membership_usable(full_tree, full, n))
    assert_true(_renewal_membership_usable(bagged_tree, bagged, len(bag)))


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
