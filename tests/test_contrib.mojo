"""Exact feature contributions (TreeSHAP).

Three kinds of check live here.

1. *Analytical tiny trees.* Hand-built trees whose Shapley values can be
   worked out on paper, so a wrong recursion cannot hide behind a
   self-consistent sum.

2. *Efficiency.* Every row's contributions must sum to that row's raw score
   over the same iteration range. This is the property callers actually rely
   on, and it is checked on trained models across objectives, missing values,
   categorical splits, multiclass, and sliced iteration ranges.

3. *Differential against an independent implementation.* `_brute_contrib`
   below computes the same Shapley values straight from the definition, by
   enumerating all 2^M subsets of features and averaging marginal
   contributions over permutations, evaluating `v(S)` with its own recursive
   cover-weighted traversal. It shares no code with contrib.mojo: no path
   extension, no unwinding, no polynomial-time trick. Agreement between an
   O(2^M) enumeration and an O(L * D^2) recursion on trees with repeated
   features, missing routing, and categorical sets is the evidence that the
   fast path is exact.
"""

from std.utils.numerics import nan
from std.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    IterationRange,
)
from mojotrees.contrib import (
    ContribExplainer,
    predict_contrib,
    predict_contrib_bins,
    predict_contrib_bins_multiclass,
    predict_contrib_multiclass,
    tree_expected_value,
)
from mojotrees.model import Model, fit, fit_multiclass
from mojotrees.tree import Tree, TreeParams
from support import _uniform

comptime _TOL = 1e-9
comptime NAN = nan[DType.float64]()


def _factorial(n: Int) -> Float64:
    var out = 1.0
    for i in range(2, n + 1):
        out *= Float64(i)
    return out


# ----------------------------------------------------------------------
# Independent reference: Shapley values straight from the definition.
# ----------------------------------------------------------------------


def _conditional(
    tree: Tree, bins: List[Int], subset: Int, node: Int
) -> Float64:
    """`v(S)` for one tree: the expected leaf value when the features in
    `subset` (a bitmask) are known to take this row's values and every other
    feature is averaged out over the node covers.

    A known feature routes the row the way prediction would; an unknown one
    splits the mass between both children in proportion to their covers.
    This is the definition contrib.mojo's recursion computes indirectly."""
    var feature = tree.feature[node]
    if feature < 0:
        return tree.value[node]
    if (subset >> feature) & 1 == 1:
        var child = (
            tree.left[node] if tree.goes_left(node, bins[feature])
            else tree.right[node]
        )
        return _conditional(tree, bins, subset, child)
    var left = tree.left[node]
    var right = tree.right[node]
    return (
        tree.count[left] * _conditional(tree, bins, subset, left)
        + tree.count[right] * _conditional(tree, bins, subset, right)
    ) / tree.count[node]


def _brute_contrib(
    booster: Booster, bins: List[Int], n_features: Int, rng: IterationRange
) raises -> List[Float64]:
    """Exact Shapley values by enumeration over all 2^n_features subsets.

        phi_i = sum_{S subset N\\{i}} |S|!(M-|S|-1)!/M! (v(S+i) - v(S))

    Exponential in the feature count, so tests keep n_features small. The
    expected-value entry is `v({})`, which is what the fast path puts in its
    last column."""
    var out = List[Float64](capacity=n_features + 1)
    out.resize(n_features + 1, 0.0)
    if rng.includes_base():
        out[n_features] = booster.base_score
    var n_subsets = 1 << n_features
    var m_fact = _factorial(n_features)
    for t in range(rng.start, rng.stop):
        ref tree = booster.trees[t]
        out[n_features] += (
            booster.learning_rate * _conditional(tree, bins, 0, 0)
        )
        for i in range(n_features):
            var phi = 0.0
            for subset in range(n_subsets):
                if (subset >> i) & 1 == 1:
                    continue
                var size = 0
                for b in range(n_features):
                    if (subset >> b) & 1 == 1:
                        size += 1
                var weight = (
                    _factorial(size)
                    * _factorial(n_features - size - 1)
                    / m_fact
                )
                phi += weight * (
                    _conditional(tree, bins, subset | (1 << i), 0)
                    - _conditional(tree, bins, subset, 0)
                )
            out[i] += booster.learning_rate * phi
    return out^


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _stump(
    feature: Int, threshold: Int, left_value: Float64, right_value: Float64,
    left_count: Float64, right_count: Float64,
) -> Tree:
    """A one-split tree: node 0 splits `feature` at `threshold`, children are
    nodes 1 and 2."""
    return Tree(
        [feature, -1, -1],
        [threshold, -1, -1],
        [1, -1, -1],
        [2, -1, -1],
        [0.0, left_value, right_value],
        [0.0, 0.0, 0.0],
        2,
        [False, False, False],
        [-1, -1, -1],
        [-1, -1, -1],
        [],
        [left_count + right_count, left_count, right_count],
    )


def _leaf_only(value: Float64, count: Float64) -> Tree:
    """A depth-zero tree: one node, no split."""
    return Tree(
        [-1], [-1], [-1], [-1], [value], [0.0], 1,
        [False], [-1], [-1], [], [count],
    )


def _booster(var trees: List[Tree], base: Float64, lr: Float64) raises -> Booster:
    return Booster(trees^, base, lr, SQUARED_ERROR)


def _full(n: Int) -> IterationRange:
    return IterationRange.slice(n, 0, n)


def _dense_dataset(
    n_rows: Int,
    n_features: Int,
    mut features: List[Float64],
    mut target: List[Float64],
    seed: UInt64 = 7,
):
    for k in range(n_rows * n_features):
        features.append(_uniform(seed + UInt64(k)))
    for r in range(n_rows):
        var acc = 0.0
        for f in range(n_features):
            acc += (1.0 + 0.5 * Float64(f)) * features[f * n_rows + r]
        acc += 2.0 * features[r] * features[n_rows + r]
        target.append(acc)


def _row_of(features: List[Float64], n_rows: Int, n_features: Int, r: Int
) -> List[Float64]:
    var row = List[Float64](capacity=n_features)
    for f in range(n_features):
        row.append(features[f * n_rows + r])
    return row^


def _sum_features(contrib: List[Float64], n_features: Int) -> Float64:
    """Contributions plus the expected value: the whole raw score."""
    var total = 0.0
    for i in range(n_features + 1):
        total += contrib[i]
    return total


# ----------------------------------------------------------------------
# Analytical tiny trees
# ----------------------------------------------------------------------


def test_single_split_splits_the_gap_entirely_onto_its_feature() raises:
    # One split on feature 0 at bin 5, leaves -1 and +3, covers 3 and 1.
    # E[f] = (3*(-1) + 1*3)/4 = 0. A row going right scores 3, so feature 0
    # must carry the whole 3 and feature 1, never split on, must carry 0.
    var trees: List[Tree] = [_stump(0, 5, -1.0, 3.0, 3.0, 1.0)]
    var b = _booster(trees^, 0.0, 1.0)
    var right: List[Int] = [9, 0]
    var contrib = predict_contrib_bins(b, right, 2, _full(1))
    assert_almost_equal(contrib[0], 3.0, atol=_TOL)
    assert_almost_equal(contrib[1], 0.0, atol=_TOL)
    assert_almost_equal(contrib[2], 0.0, atol=_TOL)

    var left: List[Int] = [1, 0]
    var lc = predict_contrib_bins(b, left, 2, _full(1))
    assert_almost_equal(lc[0], -1.0, atol=_TOL)
    assert_almost_equal(lc[1], 0.0, atol=_TOL)
    assert_almost_equal(lc[2], 0.0, atol=_TOL)


def test_depth_zero_tree_attributes_nothing_to_any_feature() raises:
    # A constant tree explains nothing about any feature; its whole output is
    # the expected value.
    var trees: List[Tree] = [_leaf_only(2.5, 10.0)]
    var b = _booster(trees^, 0.75, 1.0)
    var bins: List[Int] = [3, 4]
    var contrib = predict_contrib_bins(b, bins, 2, _full(1))
    assert_almost_equal(contrib[0], 0.0, atol=_TOL)
    assert_almost_equal(contrib[1], 0.0, atol=_TOL)
    assert_almost_equal(contrib[2], 0.75 + 2.5, atol=_TOL)


def test_symmetric_and_pays_out_the_learning_rate() raises:
    # Two identical stumps on different features. Each contributes its own
    # gap, scaled by the learning rate.
    var trees: List[Tree] = [
        _stump(0, 5, -1.0, 1.0, 1.0, 1.0),
        _stump(1, 5, -1.0, 1.0, 1.0, 1.0),
    ]
    var b = _booster(trees^, 0.0, 0.5)
    var bins: List[Int] = [9, 1]
    var contrib = predict_contrib_bins(b, bins, 2, _full(2))
    assert_almost_equal(contrib[0], 0.5, atol=_TOL)
    assert_almost_equal(contrib[1], -0.5, atol=_TOL)
    assert_almost_equal(contrib[2], 0.0, atol=_TOL)


def test_and_gate_splits_credit_evenly() raises:
    # An AND gate over two features with equal covers everywhere: the leaf is
    # 1 only when both go right. For the row that reaches it, symmetry forces
    # each feature to carry exactly half of the gap from E[f] = 0.25.
    #   node 0: feature 0        node 2: feature 1
    #   0 -> left leaf (1) value 0, right node 2
    #   node 2 -> leaf 3 value 0, leaf 4 value 1
    var tree = Tree(
        [0, -1, 1, -1, -1],
        [5, -1, 5, -1, -1],
        [1, -1, 3, -1, -1],
        [2, -1, 4, -1, -1],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        3,
        [False, False, False, False, False],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [],
        [4.0, 2.0, 2.0, 1.0, 1.0],
    )
    var trees: List[Tree] = [tree^]
    var b = _booster(trees^, 0.0, 1.0)
    var both_right: List[Int] = [9, 9]
    var contrib = predict_contrib_bins(b, both_right, 2, _full(1))
    assert_almost_equal(contrib[2], 0.25, atol=_TOL)
    assert_almost_equal(contrib[0], 0.375, atol=_TOL)
    assert_almost_equal(contrib[1], 0.375, atol=_TOL)
    assert_almost_equal(_sum_features(contrib, 2), 1.0, atol=_TOL)


def test_repeated_feature_on_one_path() raises:
    # Feature 0 split twice down the same path. The unwind step has to remove
    # the shallower occurrence, so all of the credit lands on feature 0 and
    # none leaks onto feature 1.
    #   node 0: f0 <= 3 ? node 1 : node 2
    #   node 2: f0 <= 7 ? leaf 3 : leaf 4
    var tree = Tree(
        [0, -1, 0, -1, -1],
        [3, -1, 7, -1, -1],
        [1, -1, 3, -1, -1],
        [2, -1, 4, -1, -1],
        [0.0, -2.0, 0.0, 1.0, 5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        3,
        [False, False, False, False, False],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [],
        [4.0, 2.0, 2.0, 1.0, 1.0],
    )
    var trees: List[Tree] = [tree^]
    var b = _booster(trees^, 0.0, 1.0)
    # E[f] = (2*(-2) + 1*1 + 1*5)/4 = 0.5
    var deep_right: List[Int] = [9, 4]
    var contrib = predict_contrib_bins(b, deep_right, 2, _full(1))
    assert_almost_equal(contrib[2], 0.5, atol=_TOL)
    assert_almost_equal(contrib[1], 0.0, atol=_TOL)
    assert_almost_equal(contrib[0], 4.5, atol=_TOL)
    assert_almost_equal(_sum_features(contrib, 2), 5.0, atol=_TOL)


def test_expected_value_is_the_cover_weighted_leaf_mean() raises:
    var tree = _stump(0, 5, -1.0, 3.0, 3.0, 1.0)
    assert_almost_equal(tree_expected_value(tree), 0.0, atol=_TOL)
    var leaf = _leaf_only(2.5, 10.0)
    assert_almost_equal(tree_expected_value(leaf), 2.5, atol=_TOL)


# ----------------------------------------------------------------------
# Differential against the subset-enumeration reference
# ----------------------------------------------------------------------


def _assert_matches_brute(
    model: Model, features: List[Float64], n_rows: Int, n_features: Int,
    rng: IterationRange, rows: List[Int],
) raises:
    for i in range(len(rows)):
        var row = _row_of(features, n_rows, n_features, rows[i])
        var bins = model.mapper.bin_row(row)
        var fast = predict_contrib_bins(
            model.booster, bins, n_features, rng
        )
        var slow = _brute_contrib(model.booster, bins, n_features, rng)
        for f in range(n_features + 1):
            assert_almost_equal(fast[f], slow[f], atol=1e-9)


def test_matches_subset_enumeration_regression() raises:
    var n_rows = 240
    var n_features = 5
    var features = List[Float64]()
    var target = List[Float64]()
    _dense_dataset(n_rows, n_features, features, target)
    var params = BoosterParams(12, 0.3, TreeParams(8, 5, 1.0, 1e-3))
    var model = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params, 32
    )
    var rows: List[Int] = [0, 7, 31, 119, 239]
    _assert_matches_brute(
        model, features, n_rows, n_features,
        _full(model.n_iterations()), rows,
    )


def test_matches_subset_enumeration_with_missing_values() raises:
    var n_rows = 200
    var n_features = 4
    var features = List[Float64]()
    var target = List[Float64]()
    _dense_dataset(n_rows, n_features, features, target, seed=99)
    # Punch NaN into two features so the trees learn missing routing and the
    # explained rows actually travel it.
    for r in range(n_rows):
        if r % 5 == 0:
            features[r] = NAN
        if r % 7 == 0:
            features[n_rows + r] = NAN
    var params = BoosterParams(10, 0.3, TreeParams(8, 5, 1.0, 1e-3))
    var model = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params, 32
    )
    var rows: List[Int] = [0, 5, 7, 14, 35, 101]
    _assert_matches_brute(
        model, features, n_rows, n_features,
        _full(model.n_iterations()), rows,
    )


def test_matches_subset_enumeration_with_categorical_splits() raises:
    var n_rows = 300
    var n_features = 4
    var features = List[Float64]()
    var target = List[Float64]()
    # Feature 0 is an integer-coded category with a non-monotone effect, so a
    # threshold split cannot imitate the set split.
    var effect: List[Float64] = [3.0, -2.0, 0.5, -1.5, 2.0, -3.0]
    for r in range(n_rows):
        features.append(Float64(r % 6))
    for k in range(n_rows, n_rows * n_features):
        features.append(_uniform(UInt64(4242) + UInt64(k)))
    for r in range(n_rows):
        var acc = effect[r % 6]
        for f in range(1, n_features):
            acc += 0.7 * Float64(f) * features[f * n_rows + r]
        target.append(acc)
    var params = BoosterParams(10, 0.3, TreeParams(8, 5, 1.0, 1e-3))
    var cats: List[Int] = [0]
    var model = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params, 32,
        categorical_features=cats,
    )
    var has_cat = False
    for t in range(len(model.booster.trees)):
        for i in range(len(model.booster.trees[t].cat_offset)):
            if model.booster.trees[t].cat_offset[i] >= 0:
                has_cat = True
    assert_true(has_cat, "expected the fit to use a categorical set split")
    var rows: List[Int] = [0, 1, 2, 3, 4, 5, 77]
    _assert_matches_brute(
        model, features, n_rows, n_features,
        _full(model.n_iterations()), rows,
    )


def test_matches_subset_enumeration_on_an_iteration_slice() raises:
    var n_rows = 200
    var n_features = 4
    var features = List[Float64]()
    var target = List[Float64]()
    _dense_dataset(n_rows, n_features, features, target, seed=1234)
    var params = BoosterParams(12, 0.25, TreeParams(8, 5, 1.0, 1e-3))
    var model = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params, 32
    )
    var n = model.n_iterations()
    var rows: List[Int] = [3, 41, 150]
    # A slice that excludes iteration 0, so the base score is excluded too.
    _assert_matches_brute(
        model, features, n_rows, n_features,
        IterationRange.slice(n, 3, 9), rows,
    )


# ----------------------------------------------------------------------
# Efficiency: contributions sum to the raw score
# ----------------------------------------------------------------------


def test_contributions_sum_to_raw_score_regression() raises:
    var n_rows = 400
    var n_features = 8
    var features = List[Float64]()
    var target = List[Float64]()
    _dense_dataset(n_rows, n_features, features, target, seed=55)
    var params = BoosterParams(40, 0.1, TreeParams(31, 20, 1.0, 1e-3))
    var model = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params
    )
    var rng = _full(model.n_iterations())
    for r in range(0, n_rows, 37):
        var row = _row_of(features, n_rows, n_features, r)
        var contrib = predict_contrib(model, row, rng)
        assert_equal(len(contrib), n_features + 1)
        assert_almost_equal(
            _sum_features(contrib, n_features),
            model.predict_raw_range(row, rng),
            atol=1e-9,
        )


def test_contributions_sum_to_raw_score_binary() raises:
    var n_rows = 400
    var n_features = 6
    var features = List[Float64]()
    var scores = List[Float64]()
    _dense_dataset(n_rows, n_features, features, scores, seed=808)
    var labels = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        labels.append(1.0 if scores[r] > 3.0 else 0.0)
    var params = BoosterParams(30, 0.2, TreeParams(15, 10, 1.0, 1e-3))
    var model = fit(
        features, n_rows, n_features, labels, BINARY_LOGISTIC, params
    )
    var rng = _full(model.n_iterations())
    for r in range(0, n_rows, 53):
        var row = _row_of(features, n_rows, n_features, r)
        var contrib = predict_contrib(model, row, rng)
        # Contributions explain the raw log-odds, not the probability.
        assert_almost_equal(
            _sum_features(contrib, n_features),
            model.predict_raw_range(row, rng),
            atol=1e-9,
        )


def test_iteration_slices_partition_the_contributions() raises:
    var n_rows = 250
    var n_features = 5
    var features = List[Float64]()
    var target = List[Float64]()
    _dense_dataset(n_rows, n_features, features, target, seed=17)
    var params = BoosterParams(20, 0.2, TreeParams(15, 10, 1.0, 1e-3))
    var model = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params
    )
    var n = model.n_iterations()
    var head = IterationRange.slice(n, 0, 7)
    var tail = IterationRange.slice(n, 7, n)
    var whole = _full(n)
    for r in range(0, n_rows, 61):
        var row = _row_of(features, n_rows, n_features, r)
        var a = predict_contrib(model, row, head)
        var b = predict_contrib(model, row, tail)
        var c = predict_contrib(model, row, whole)
        for f in range(n_features + 1):
            assert_almost_equal(a[f] + b[f], c[f], atol=1e-9)


def test_empty_range_is_all_zeros_and_base_only_range_is_the_base() raises:
    var n_rows = 150
    var n_features = 3
    var features = List[Float64]()
    var target = List[Float64]()
    _dense_dataset(n_rows, n_features, features, target, seed=3)
    var params = BoosterParams(8, 0.2, TreeParams(8, 5, 1.0, 1e-3))
    var model = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params
    )
    var n = model.n_iterations()
    var row = _row_of(features, n_rows, n_features, 4)

    # [0, 0) is the base-score-only model.
    var base_only = predict_contrib(model, row, IterationRange.slice(n, 0, 0))
    assert_almost_equal(base_only[0], 0.0, atol=_TOL)
    assert_almost_equal(base_only[1], 0.0, atol=_TOL)
    assert_almost_equal(base_only[2], 0.0, atol=_TOL)
    assert_almost_equal(
        base_only[3], model.booster.base_score, atol=_TOL
    )

    # An empty range that starts later carries nothing at all.
    var empty = predict_contrib(model, row, IterationRange.slice(n, 3, 3))
    for f in range(n_features + 1):
        assert_almost_equal(empty[f], 0.0, atol=_TOL)


# ----------------------------------------------------------------------
# Multiclass
# ----------------------------------------------------------------------


def test_multiclass_layout_and_per_class_sums() raises:
    var n_rows = 300
    var n_features = 4
    var n_classes = 3
    var features = List[Float64]()
    var scores = List[Float64]()
    _dense_dataset(n_rows, n_features, features, scores, seed=606)
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        labels.append(r % n_classes)
    var params = BoosterParams(12, 0.2, TreeParams(8, 5, 1.0, 1e-3))
    var model = fit_multiclass(
        features, n_rows, n_features, labels, n_classes, params
    )
    var rng = _full(model.n_iterations())
    var stride = n_features + 1
    for r in range(0, n_rows, 41):
        var row = _row_of(features, n_rows, n_features, r)
        var contrib = predict_contrib_multiclass(model, row, rng)
        assert_equal(len(contrib), n_classes * stride)
        var raw = model.predict_raw_range(row, rng)
        for k in range(n_classes):
            var total = 0.0
            for f in range(stride):
                total += contrib[k * stride + f]
            assert_almost_equal(total, raw[k], atol=1e-9)


def test_multiclass_matches_subset_enumeration_per_class() raises:
    # Each class's block is the single-output problem over that class's
    # trees, so the enumeration reference applies to it unchanged.
    var n_rows = 180
    var n_features = 3
    var n_classes = 3
    var features = List[Float64]()
    var scores = List[Float64]()
    _dense_dataset(n_rows, n_features, features, scores, seed=71)
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        labels.append(Int(scores[r] * 1.7) % n_classes)
    var params = BoosterParams(6, 0.3, TreeParams(8, 5, 1.0, 1e-3))
    var model = fit_multiclass(
        features, n_rows, n_features, labels, n_classes, params, 32
    )
    var n = model.n_iterations()
    var rng = _full(n)
    var stride = n_features + 1
    for r in range(0, n_rows, 47):
        var row = _row_of(features, n_rows, n_features, r)
        var bins = model.mapper.bin_row(row)
        var fast = predict_contrib_bins_multiclass(
            model.booster, bins, n_features, rng
        )
        for k in range(n_classes):
            var per_class = List[Tree]()
            for i in range(n):
                per_class.append(model.booster.trees[i * n_classes + k].copy())
            var b = Booster(
                per_class^,
                model.booster.base_scores[k],
                model.booster.learning_rate,
                SQUARED_ERROR,
            )
            var slow = _brute_contrib(b, bins, n_features, _full(n))
            for f in range(stride):
                assert_almost_equal(
                    fast[k * stride + f], slow[f], atol=1e-9
                )


# ----------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------


def test_model_without_node_counts_is_refused() raises:
    # A tree built without covers is what a v1 or v2 file loads as. Asking it
    # for contributions must say so rather than divide by zero.
    var bare = Tree(
        [0, -1, -1], [5, -1, -1], [1, -1, -1], [2, -1, -1],
        [0.0, -1.0, 3.0], [0.0, 0.0, 0.0], 2,
    )
    assert_true(not bare.has_node_counts())
    var trees: List[Tree] = [bare^]
    var b = _booster(trees^, 0.0, 1.0)
    var bins: List[Int] = [9, 0]
    with assert_raises():
        _ = predict_contrib_bins(b, bins, 2, _full(1))


def test_grown_trees_carry_consistent_covers() raises:
    # Every node's cover is the sum of its children's, on every tree a fit
    # produces. This is what makes the expected value a single pass over the
    # leaves rather than a traversal.
    var n_rows = 200
    var n_features = 4
    var features = List[Float64]()
    var target = List[Float64]()
    _dense_dataset(n_rows, n_features, features, target, seed=21)
    var params = BoosterParams(10, 0.2, TreeParams(15, 5, 1.0, 1e-3))
    var model = fit(
        features, n_rows, n_features, target, SQUARED_ERROR, params
    )
    for t in range(len(model.booster.trees)):
        ref tree = model.booster.trees[t]
        assert_true(tree.has_node_counts())
        assert_almost_equal(tree.count[0], Float64(n_rows), atol=_TOL)
        for i in range(len(tree.feature)):
            if tree.feature[i] < 0:
                continue
            assert_almost_equal(
                tree.count[tree.left[i]] + tree.count[tree.right[i]],
                tree.count[i],
                atol=_TOL,
            )


def test_explainer_rejects_a_zero_feature_model() raises:
    var trees: List[Tree] = [_leaf_only(1.0, 5.0)]
    var b = _booster(trees^, 0.0, 1.0)
    with assert_raises():
        _ = ContribExplainer.for_booster(b, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
