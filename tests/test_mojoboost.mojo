from std.testing import assert_equal, assert_true, assert_false, TestSuite

from mojoboost import (
    BinnedMatrix,
    Histogram,
    SplitInfo,
    Tree,
    TreeParams,
    bin_equal_width,
    build_histogram,
    build_histogram_subset,
    find_best_split,
    grow_tree,
    subtract_histogram,
)


def make_toy() raises -> BinnedMatrix:
    # One feature, 8 rows, values 0..7 binned into 8 equal-width bins.
    # Equal-width binning maps value v to bin v exactly for this input.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    return bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)


def test_binning_identity() raises:
    var data = make_toy()
    assert_equal(data.n_rows, 8)
    assert_equal(data.n_features, 1)
    assert_equal(data.n_bins, 8)
    for r in range(8):
        assert_equal(data.bin_at(r, 0), r)


def test_binning_constant_feature() raises:
    var features: List[Float64] = [3.0, 3.0, 3.0, 3.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)
    for r in range(4):
        assert_equal(data.bin_at(r, 0), 0)


def test_binning_validates_input() raises:
    var features: List[Float64] = [1.0, 2.0]
    var raised = False
    try:
        _ = bin_equal_width(features, n_rows=2, n_features=1, n_bins=1)
    except:
        raised = True
    assert_true(raised)


def test_histogram_sums() raises:
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var hist = build_histogram(data, grad, hess)
    for b in range(8):
        assert_equal(hist.count[b], 1)
        assert_equal(hist.hess[b], 1.0)
        if b < 4:
            assert_equal(hist.grad[b], -1.0)
        else:
            assert_equal(hist.grad[b], 1.0)


def test_best_split_separates_gradients() raises:
    # Gradients flip sign between bins 3 and 4, so the best split must be
    # at bin 3 (rows with bin <= 3 go left).
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var hist = build_histogram(data, grad, hess)
    var split = find_best_split(hist, lambda_reg=1.0)
    assert_true(split.found)
    assert_equal(split.feature, 0)
    assert_equal(split.bin, 3)
    # GL=-4, HL=4, GR=4, HR=4, G=0: gain = 16/5 + 16/5 - 0 = 6.4
    assert_true(abs(split.gain - 6.4) < 1e-12)


def test_no_split_on_uniform_gradients() raises:
    var data = make_toy()
    var grad: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    var hist = build_histogram(data, grad, hess)
    var split = find_best_split(hist, lambda_reg=1.0)
    assert_false(split.found)


def make_additive() raises -> BinnedMatrix:
    # Two binary features, 8 rows, column-major. Gradients are additive in
    # both features with feature 0 dominant, so leaf-wise growth must split
    # feature 0 at the root and then feature 1 in each child.
    var features: List[Float64] = [
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,  # feature 0
        0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,  # feature 1
    ]
    return bin_equal_width(features, n_rows=8, n_features=2, n_bins=2)


def additive_grad() -> List[Float64]:
    # g(row) = (f0 ? +2 : -2) + (f1 ? +1 : -1)
    return [-3.0, -1.0, -3.0, -1.0, 1.0, 3.0, 1.0, 3.0]


def ones(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(1.0)
    return out^


def test_histogram_subtraction() raises:
    var data = make_additive()
    var grad = additive_grad()
    var hess = ones(8)
    var parent = build_histogram(data, grad, hess)
    var left_rows: List[Int] = [0, 1, 2, 3]
    var right_rows: List[Int] = [4, 5, 6, 7]
    var left = build_histogram_subset(data, grad, hess, left_rows)
    var derived_right = subtract_histogram(parent, left)
    var direct_right = build_histogram_subset(data, grad, hess, right_rows)
    for i in range(2 * 2):
        assert_true(abs(derived_right.grad[i] - direct_right.grad[i]) < 1e-12)
        assert_true(abs(derived_right.hess[i] - direct_right.hess[i]) < 1e-12)
        assert_equal(derived_right.count[i], direct_right.count[i])


def test_tree_single_split() raises:
    var data = make_toy()
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    var hess = ones(8)
    var params = TreeParams(2, 1, 1.0, 1e-3)
    var tree = grow_tree(data, grad, hess, params)
    assert_equal(tree.n_leaves, 2)
    assert_equal(tree.feature[0], 0)
    assert_equal(tree.threshold_bin[0], 3)
    # Leaf values are the Newton step -G/(H+lambda): -(-4)/5 and -4/5.
    for r in range(8):
        var expected = 0.8 if r < 4 else -0.8
        assert_true(abs(tree.predict_row(data, r) - expected) < 1e-12)


def test_tree_leafwise_depth2() raises:
    var data = make_additive()
    var grad = additive_grad()
    var hess = ones(8)
    var params = TreeParams(4, 1, 1.0, 1e-3)
    var tree = grow_tree(data, grad, hess, params)
    assert_equal(tree.n_leaves, 4)
    # Root must split the dominant feature 0.
    assert_equal(tree.feature[0], 0)
    # Each (f0, f1) group of 2 rows has value -G/(H+1) = -2g/3.
    var expected: List[Float64] = [2.0, 2.0 / 3.0, 2.0, 2.0 / 3.0,
                                   -2.0 / 3.0, -2.0, -2.0 / 3.0, -2.0]
    for r in range(8):
        assert_true(abs(tree.predict_row(data, r) - expected[r]) < 1e-12)


def test_tree_min_data_in_leaf() raises:
    # min_data_in_leaf=3 allows the root 4/4 split but blocks the 2/2
    # child splits, so growth stops at 2 leaves despite num_leaves=31.
    var data = make_additive()
    var grad = additive_grad()
    var hess = ones(8)
    var params = TreeParams(31, 3, 1.0, 1e-3)
    var tree = grow_tree(data, grad, hess, params)
    assert_equal(tree.n_leaves, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
