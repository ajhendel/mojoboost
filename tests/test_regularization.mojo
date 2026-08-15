"""L1 split regularization (`lambda_l1`).

Every case here has hand-computed gradient and hessian totals, so the
expected gains and leaf values are analytical rather than golden values.
The soft-threshold operator matches LightGBM's ThresholdL1:

    T(G) = sign(G) * max(0, |G| - lambda_l1)

and is applied to the parent, left, and right gradient sums in the split
gain and to the gradient sum in the Newton leaf value.
"""

from std.testing import assert_equal, assert_true, assert_false, TestSuite

from mojotrees import (
    L1,
    SQUARED_ERROR,
    BinnedMatrix,
    BoosterParams,
    TreeParams,
    bin_equal_width,
    build_histogram,
    find_best_split,
    grow_tree,
    soft_threshold_l1,
    train,
)


def make_toy() raises -> BinnedMatrix:
    # One feature, 8 rows, values 0..7 binned into 8 equal-width bins, so
    # row r lands in bin r exactly.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    return bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)


def ones(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(1.0)
    return out^


def close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) < 1e-12


def test_soft_threshold_values() raises:
    # Positive and negative sums shrink toward zero by lambda_l1.
    assert_true(close(soft_threshold_l1(5.0, 2.0), 3.0))
    assert_true(close(soft_threshold_l1(-5.0, 2.0), -3.0))
    # Sums inside the threshold collapse to exactly zero.
    assert_true(close(soft_threshold_l1(1.0, 2.0), 0.0))
    assert_true(close(soft_threshold_l1(-1.0, 2.0), 0.0))
    assert_true(close(soft_threshold_l1(2.0, 2.0), 0.0))
    assert_true(close(soft_threshold_l1(0.0, 2.0), 0.0))
    # Zero (and any non-positive) lambda_l1 is the identity.
    assert_true(close(soft_threshold_l1(5.0, 0.0), 5.0))
    assert_true(close(soft_threshold_l1(-5.0, 0.0), -5.0))
    assert_true(close(soft_threshold_l1(5.0, -1.0), 5.0))


def test_split_gain_zero_l1_matches_plain_formula() raises:
    # Gradients flip sign between bins 3 and 4. GL=-4, HL=4, GR=4, HR=4,
    # G=0, lambda_l2=1: gain = 16/5 + 16/5 - 0 = 6.4.
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hist = build_histogram(data, grad, ones(8))

    var plain = find_best_split(hist, lambda_reg=1.0)
    var explicit_zero = find_best_split(
        hist, lambda_reg=1.0, lambda_l1=0.0
    )
    assert_true(plain.found)
    assert_equal(plain.bin, 3)
    assert_true(close(plain.gain, 6.4))
    assert_equal(explicit_zero.bin, plain.bin)
    assert_true(close(explicit_zero.gain, plain.gain))


def test_split_gain_with_l1_soft_thresholds_children() raises:
    # Same totals with lambda_l1 = 1: T(-4) = -3, T(4) = 3, T(0) = 0, so
    # gain = 9/5 + 9/5 - 0 = 3.6.
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hist = build_histogram(data, grad, ones(8))

    var split = find_best_split(
        hist, lambda_reg=1.0, min_child_hess=1e-3, min_data_in_leaf=0,
        lambda_l1=1.0,
    )
    assert_true(split.found)
    assert_equal(split.feature, 0)
    assert_equal(split.bin, 3)
    assert_true(close(split.gain, 3.6))


def test_split_gain_thresholds_parent_too() raises:
    # Nonzero parent sum, so the parent term is soft-thresholded as well.
    # Totals: G=8, H=8. At bin 3: GL=-4, HL=4, GR=12, HR=4.
    #   lambda_l1 = 0: 16/5 + 144/5 - 64/9 = 224/9
    #   lambda_l1 = 2: T=-2, 10, 6 -> 4/5 + 100/5 - 36/9 = 16.8
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 3.0, 3.0, 3.0, 3.0]
    var hist = build_histogram(data, grad, ones(8))

    var plain = find_best_split(hist, lambda_reg=1.0)
    assert_true(plain.found)
    assert_equal(plain.bin, 3)
    assert_true(close(plain.gain, 224.0 / 9.0))

    var regularized = find_best_split(
        hist, lambda_reg=1.0, min_child_hess=1e-3, min_data_in_leaf=0,
        lambda_l1=2.0,
    )
    assert_true(regularized.found)
    assert_equal(regularized.bin, 3)
    assert_true(close(regularized.gain, 16.8))
    # Regularization can only reduce the gain of a fixed split here.
    assert_true(regularized.gain < plain.gain)


def test_large_l1_removes_every_split() raises:
    # Every candidate has |GL| <= 4 and |GR| <= 4, so lambda_l1 = 4 zeroes
    # all three terms and no candidate has positive gain.
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hist = build_histogram(data, grad, ones(8))

    var split = find_best_split(
        hist, lambda_reg=1.0, min_child_hess=1e-3, min_data_in_leaf=0,
        lambda_l1=4.0,
    )
    assert_false(split.found)

    # Tree growth therefore stops at the root, whose value is also fully
    # shrunk to zero (T(0) = 0).
    var params = TreeParams(31, 1, 1.0, 1e-3, 4.0)
    var tree = grow_tree(data, grad, ones(8), params)
    assert_equal(tree.n_leaves, 1)
    assert_true(close(tree.value[0], 0.0))


def test_leaf_values_use_thresholded_gradient_sum() raises:
    # num_leaves = 2 splits at bin 3: left G=-4, H=4; right G=4, H=4.
    # With lambda_l2 = 1 the Newton values are -T(G)/(H+1).
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hess = ones(8)

    var plain = grow_tree(data, grad, hess, TreeParams(2, 1, 1.0, 1e-3))
    for r in range(8):
        var expected = 0.8 if r < 4 else -0.8
        assert_true(close(plain.predict_row(data, r), expected))

    # lambda_l1 = 1: T(-4) = -3 -> 3/5, T(4) = 3 -> -3/5.
    var reg = grow_tree(data, grad, hess, TreeParams(2, 1, 1.0, 1e-3, 1.0))
    assert_equal(reg.feature[0], 0)
    assert_equal(reg.threshold_bin[0], 3)
    for r in range(8):
        var expected = 0.6 if r < 4 else -0.6
        assert_true(close(reg.predict_row(data, r), expected))

    # lambda_l1 = 3: T(-4) = -1 -> 1/5, T(4) = 1 -> -1/5.
    var more = grow_tree(data, grad, hess, TreeParams(2, 1, 1.0, 1e-3, 3.0))
    for r in range(8):
        var expected = 0.2 if r < 4 else -0.2
        assert_true(close(more.predict_row(data, r), expected))


def test_leaf_magnitude_is_monotone_in_l1() raises:
    # Growing lambda_l1 can only shrink a leaf's magnitude toward zero.
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 3.0, 3.0, 3.0, 3.0]
    var hess = ones(8)
    var previous = 1e18
    for i in range(9):
        var l1 = 0.5 * Float64(i)
        var tree = grow_tree(
            data, grad, hess, TreeParams(2, 1, 1.0, 1e-3, l1)
        )
        var magnitude = abs(tree.predict_row(data, 0))
        assert_true(magnitude <= previous + 1e-12)
        previous = magnitude
    assert_true(close(previous, 0.0))


def test_zero_l1_is_bit_identical_to_unregularized_growth() raises:
    # Explicit lambda_l1 = 0 must reproduce the pre-existing code path
    # exactly, not just approximately, on a multi-feature dataset.
    var n_rows = 64
    var n_features = 4
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        # Deterministic spread of values; the exact pattern is irrelevant.
        features.append(Float64((k * 37) % 61) * 0.125)
    var data = bin_equal_width(features, n_rows, n_features, 16)
    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(Float64((r * 13) % 17) - 8.0)
        hess.append(1.0 + Float64(r % 3))

    var baseline = grow_tree(data, grad, hess, TreeParams(12, 2, 1.0, 1e-3))
    var explicit = grow_tree(
        data, grad, hess, TreeParams(12, 2, 1.0, 1e-3, 0.0)
    )
    assert_equal(baseline.n_leaves, explicit.n_leaves)
    assert_equal(len(baseline.feature), len(explicit.feature))
    for i in range(len(baseline.feature)):
        assert_equal(baseline.feature[i], explicit.feature[i])
        assert_equal(baseline.threshold_bin[i], explicit.threshold_bin[i])
        assert_equal(baseline.value[i], explicit.value[i])


def test_weighted_rows_threshold_the_weighted_sum() raises:
    # lambda_l1 applies to the weighted gradient sum, so weights change how
    # much of a leaf's gradient survives the threshold.
    #
    # target = 0,0,0,0,1,1,1,1 with weights 1,1,1,1,3,3,3,3:
    #   base score = 12/16 = 0.75, grad = w * (raw - y) = +0.75 / -0.75,
    #   hess = w. At bin 3: GL = 3, HL = 4; GR = -3, HR = 12.
    # With lambda_l1 = 1 and lambda_l2 = 1 the leaf values are
    #   left  = -T(3)/(4+1)  = -2/5
    #   right = -T(-3)/(12+1) = 2/13
    var data = make_toy()
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var weights: List[Float64] = [1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0]
    var params = BoosterParams(1, 0.1, TreeParams(2, 1, 1.0, 1e-3, 1.0))
    var model = train(data, target, SQUARED_ERROR, params, weights)

    assert_equal(len(model.trees), 1)
    assert_equal(model.trees[0].threshold_bin[0], 3)
    assert_true(close(model.base_score, 0.75))
    for r in range(8):
        var leaf = -2.0 / 5.0 if r < 4 else 2.0 / 13.0
        assert_true(close(model.predict_row(data, r), 0.75 + 0.1 * leaf))

    # Unweighted, the same targets give GL = 2, HL = 4 and GR = -2, HR = 4
    # around a base score of 0.5, so the surviving gradient differs.
    var unweighted = train(
        data, target, SQUARED_ERROR,
        BoosterParams(1, 0.1, TreeParams(2, 1, 1.0, 1e-3, 1.0)),
    )
    assert_true(close(unweighted.base_score, 0.5))
    for r in range(8):
        var leaf = -1.0 / 5.0 if r < 4 else 1.0 / 5.0
        assert_true(close(unweighted.predict_row(data, r), 0.5 + 0.1 * leaf))


def test_l1_objective_leaf_renewal_ignores_lambda_l1() raises:
    # LightGBM's RenewTreeOutput replaces Newton leaf values with residual
    # percentiles for the L1 objective, so lambda_l1 shapes the split
    # search but not the final leaf value. Here the split is at bin 3 for
    # both settings, so predictions are identical.
    var data = make_toy()
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var plain = train(
        data, target, L1, BoosterParams(1, 0.1, TreeParams(2, 1, 1.0, 1e-3))
    )
    var reg = train(
        data, target, L1,
        BoosterParams(1, 0.1, TreeParams(2, 1, 1.0, 1e-3, 1.0)),
    )
    assert_equal(len(plain.trees), 1)
    assert_equal(len(reg.trees), 1)
    for r in range(8):
        # Base score 0.5, leaf residual medians -0.5 and +0.5.
        var expected = 0.45 if r < 4 else 0.55
        assert_true(close(plain.predict_row(data, r), expected))
        assert_true(close(reg.predict_row(data, r), expected))


def test_boosting_still_fits_with_l1() raises:
    # Sanity check that a regularized ensemble still learns a step
    # function; L1 shrinks each step, so it needs more rounds than the
    # unregularized fit.
    var data = make_toy()
    var target: List[Float64] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    var params = BoosterParams(400, 0.1, TreeParams(4, 1, 1.0, 1e-3, 0.05))
    var model = train(data, target, SQUARED_ERROR, params)
    for r in range(8):
        assert_true(abs(model.predict_row(data, r) - target[r]) < 0.05)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
