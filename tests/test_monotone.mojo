"""Tests for monotonic constraints.

The guarantee under test is exact, not approximate: along a constrained
feature, predictions never step the wrong way, at any raw feature value. So
the grid checks below assert plain `<=` and `>=` with no tolerance. That is
sound rather than lucky: each tree is monotone by construction, floating-point
multiplication by a positive learning rate and floating-point addition are both
monotone in each argument, and the ensemble sums its trees in a fixed order.

The training data is deliberately hostile: a V shape in feature 0 (falling then
rising) and an inverted V in feature 1. An unconstrained fit follows both and
is monotone in neither, which `test_constraints_change_the_fit` asserts
directly, so the grid checks cannot pass by accident on data that was already
monotone.
"""

from std.os import remove
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.binning import bin_equal_width
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    L1,
    QUANTILE,
    SQUARED_ERROR,
    BoosterParams,
    train,
    train_multiclass,
)
from mojotrees.histogram import build_histogram
from mojotrees.model import Model, fit, fit_multiclass
from mojotrees.model_sparse import fit_csc
from mojotrees.sparse import csc_from_dense
from mojotrees.monotone import (
    MONOTONE_DECREASING,
    MONOTONE_FREE,
    MONOTONE_INCREASING,
    MonotoneConstraints,
    OutputBounds,
    child_bounds,
    violates,
)
from mojotrees.serialize import load_model, save_model
from mojotrees.split import find_best_split
from mojotrees.tree import Tree, TreeParams, grow_tree, node_bounds

comptime _TMP_PATH = "./.test_monotone_roundtrip.tmp"

# 6 x 6 grid of integer feature values, so quantile binning gives each feature
# exactly six bins and every grid cell is its own row.
comptime _SIDE = 6


def _grid_features() -> List[Float64]:
    """Column-major 36 x 2 grid: feature 0 varies fastest."""
    var features = List[Float64](capacity=_SIDE * _SIDE * 2)
    for _ in range(_SIDE):
        for i in range(_SIDE):
            features.append(Float64(i))
    for j in range(_SIDE):
        for _ in range(_SIDE):
            features.append(Float64(j))
    return features^


def _v_shape_target(features: List[Float64]) -> List[Float64]:
    """A V in feature 0 and an inverted V in feature 1: monotone in neither."""
    var n = _SIDE * _SIDE
    var target = List[Float64](capacity=n)
    for r in range(n):
        var x0 = features[r] - 2.5
        var x1 = features[n + r] - 2.5
        target.append(x0 * x0 - x1 * x1)
    return target^


def _params(n_rounds: Int, var monotone: MonotoneConstraints) -> BoosterParams:
    var tree = TreeParams(
        8, 1, 1.0, 1e-3, 0.0, monotone=monotone^
    )
    return BoosterParams(n_rounds, 0.2, tree^)


def _query_grid() -> List[Float64]:
    """Raw query values from below the training range to above it, so the
    checks cover unseen values and not just the training grid."""
    var xs = List[Float64]()
    var x = -1.0
    while x <= 6.5:
        xs.append(x)
        x += 0.5
    return xs^


def _predict(model: Model, x0: Float64, x1: Float64) raises -> Float64:
    var row: List[Float64] = [x0, x1]
    return model.predict(row)


def _predict_grid(
    model: Model, xs: List[Float64]
) raises -> List[Float64]:
    """Response-scale predictions over the whole query grid, flattened as
    `[i * len(xs) + j]` for feature values `(xs[i], xs[j])`. One pass, so the
    monotonicity checks below are table comparisons rather than refits."""
    var n = len(xs)
    var out = List[Float64](capacity=n * n)
    for i in range(n):
        for j in range(n):
            out.append(_predict(model, xs[i], xs[j]))
    return out^


def _assert_nondecreasing_in_f0(grid: List[Float64], n: Int) raises:
    """Along feature 0, holding feature 1 fixed."""
    for j in range(n):
        for i in range(n - 1):
            assert_true(grid[i * n + j] <= grid[(i + 1) * n + j])


def _assert_nonincreasing_in_f1(grid: List[Float64], n: Int) raises:
    """Along feature 1, holding feature 0 fixed."""
    for i in range(n):
        for j in range(n - 1):
            assert_true(grid[i * n + j] >= grid[i * n + j + 1])


def _fit_v_shape(
    var monotone: MonotoneConstraints, n_rounds: Int = 25
) raises -> Model:
    var features = _grid_features()
    var target = _v_shape_target(features)
    return fit(
        features,
        _SIDE * _SIDE,
        2,
        target,
        SQUARED_ERROR,
        _params(n_rounds, monotone^),
    )


def test_grid_nondecreasing_and_nonincreasing() raises:
    # Feature 0 constrained nondecreasing, feature 1 nonincreasing. Every
    # query point in the grid is checked against its neighbor along each axis,
    # holding the other feature fixed.
    var model = _fit_v_shape(
        MonotoneConstraints.from_signs(
            [MONOTONE_INCREASING, MONOTONE_DECREASING], 2
        )
    )
    var xs = _query_grid()
    var grid = _predict_grid(model, xs)
    _assert_nondecreasing_in_f0(grid, len(xs))
    _assert_nonincreasing_in_f1(grid, len(xs))


def test_constraints_change_the_fit() raises:
    # The unconstrained fit on this data must violate both constraints,
    # otherwise the grid checks above would prove nothing.
    var xs = _query_grid()
    var n = len(xs)
    var plain = _predict_grid(_fit_v_shape(MonotoneConstraints()), xs)

    var fell_against_f0 = False
    var rose_against_f1 = False
    for j in range(n):
        for i in range(n - 1):
            if plain[i * n + j] > plain[(i + 1) * n + j]:
                fell_against_f0 = True
    for i in range(n):
        for j in range(n - 1):
            if plain[i * n + j] < plain[i * n + j + 1]:
                rose_against_f1 = True
    assert_true(fell_against_f0)
    assert_true(rose_against_f1)

    # And the constrained fit must actually differ from it somewhere.
    var constrained = _predict_grid(
        _fit_v_shape(
            MonotoneConstraints.from_signs(
                [MONOTONE_INCREASING, MONOTONE_DECREASING], 2
            )
        ),
        xs,
    )
    var differs = False
    for k in range(n * n):
        if plain[k] != constrained[k]:
            differs = True
    assert_true(differs)


def test_one_constrained_feature_leaves_the_other_free() raises:
    # Constraining feature 0 alone must not quietly constrain feature 1: the
    # fit stays free to follow the inverted V there.
    var model = _fit_v_shape(
        MonotoneConstraints.from_signs([MONOTONE_INCREASING, MONOTONE_FREE], 2)
    )
    var xs = _query_grid()
    var n = len(xs)
    var grid = _predict_grid(model, xs)
    _assert_nondecreasing_in_f0(grid, n)

    var rose = False
    var fell = False
    for i in range(n):
        for j in range(n - 1):
            if grid[i * n + j + 1] > grid[i * n + j]:
                rose = True
            if grid[i * n + j + 1] < grid[i * n + j]:
                fell = True
    assert_true(rose)
    assert_true(fell)


def test_unconstrained_equivalence_is_bit_exact() raises:
    # No vector, an all-zero vector, and an all-zero vector of the wrong
    # intent are all inactive: split search keeps its unconstrained path and
    # the fits must agree bit for bit, not merely closely.
    var features = _grid_features()
    var target = _v_shape_target(features)
    var n = _SIDE * _SIDE
    var data = bin_equal_width(features, n, 2, 8)

    var plain = train(
        data, target, SQUARED_ERROR, _params(30, MonotoneConstraints())
    )
    var zeros = train(
        data,
        target,
        SQUARED_ERROR,
        _params(
            30,
            MonotoneConstraints.from_signs(
                [MONOTONE_FREE, MONOTONE_FREE], 2
            ),
        ),
    )
    assert_equal(len(plain.trees), len(zeros.trees))
    for r in range(n):
        assert_equal(
            plain.predict_raw_row(data, r), zeros.predict_raw_row(data, r)
        )


def test_invalid_configurations() raises:
    # Values outside {-1, 0, 1}.
    with assert_raises():
        _ = MonotoneConstraints.from_signs([MONOTONE_INCREASING, 2], 2)
    with assert_raises():
        _ = MonotoneConstraints.from_signs([-2], 1)
    # One entry per feature, or none at all.
    with assert_raises():
        _ = MonotoneConstraints.from_signs([MONOTONE_INCREASING], 2)
    with assert_raises():
        _ = MonotoneConstraints.from_signs(
            [MONOTONE_INCREASING, MONOTONE_FREE, MONOTONE_FREE], 2
        )

    # A vector that never went through the validating factory is still caught
    # against the data, before any tree is grown.
    var features = _grid_features()
    var target = _v_shape_target(features)
    var mismatched = MonotoneConstraints()
    mismatched.signs = [
        MONOTONE_INCREASING, MONOTONE_FREE, MONOTONE_DECREASING
    ]
    with assert_raises():
        _ = fit(
            features,
            _SIDE * _SIDE,
            2,
            target,
            SQUARED_ERROR,
            _params(5, mismatched^),
        )

    # And the low-level search rejects a vector of the wrong width.
    var data = bin_equal_width(features, _SIDE * _SIDE, 2, 8)
    var grad = List[Float64](capacity=_SIDE * _SIDE)
    var hess = List[Float64](capacity=_SIDE * _SIDE)
    for r in range(_SIDE * _SIDE):
        grad.append(target[r])
        hess.append(1.0)
    var hist = build_histogram(data, grad, hess)
    with assert_raises():
        _ = find_best_split(hist, monotone=[MONOTONE_INCREASING])


def test_binary_logistic_probability_is_monotone() raises:
    # The link is increasing, so a constrained raw score gives a constrained
    # probability. Labels follow the V shape, which the constraint fights.
    var features = _grid_features()
    var raw_target = _v_shape_target(features)
    var n = _SIDE * _SIDE
    var labels = List[Float64](capacity=n)
    for r in range(n):
        labels.append(1.0 if raw_target[r] > 0.0 else 0.0)

    var model = fit(
        features,
        n,
        2,
        labels,
        BINARY_LOGISTIC,
        _params(
            20,
            MonotoneConstraints.from_signs(
                [MONOTONE_INCREASING, MONOTONE_DECREASING], 2
            ),
        ),
    )
    var xs = _query_grid()
    var grid = _predict_grid(model, xs)
    _assert_nondecreasing_in_f0(grid, len(xs))
    _assert_nonincreasing_in_f1(grid, len(xs))
    for k in range(len(grid)):
        assert_true(grid[k] >= 0.0 and grid[k] <= 1.0)


def test_quantile_and_l1_renewal_stay_monotone() raises:
    # Leaf renewal rewrites every leaf value after the tree is grown, so
    # without the clamp these two objectives would lose the constraint.
    var features = _grid_features()
    var target = _v_shape_target(features)
    var n = _SIDE * _SIDE
    var xs = _query_grid()

    for which in range(2):
        var objective = QUANTILE if which == 0 else L1
        var model = fit(
            features,
            n,
            2,
            target,
            objective,
            _params(
                20,
                MonotoneConstraints.from_signs(
                    [MONOTONE_INCREASING, MONOTONE_DECREASING], 2
                ),
            ),
            alpha=0.7,
        )
        var grid = _predict_grid(model, xs)
        _assert_nondecreasing_in_f0(grid, len(xs))
        _assert_nonincreasing_in_f1(grid, len(xs))


def test_sparse_renewal_stays_monotone_and_records() raises:
    # The sparse trainer shares the renewal layer with the dense one, so the
    # renewal clamp and the recorded constraints must survive the sparse
    # path too: without the clamp a renewed leaf can step outside its
    # monotone interval, and without the record the fitted model cannot
    # state the property its trees satisfy.
    var features = _grid_features()
    var target = _v_shape_target(features)
    var n = _SIDE * _SIDE
    var xs = _query_grid()
    var csc = csc_from_dense(features, n, 2)

    for which in range(2):
        var objective = QUANTILE if which == 0 else L1
        var model = fit_csc(
            csc,
            target,
            objective,
            _params(
                20,
                MonotoneConstraints.from_signs(
                    [MONOTONE_INCREASING, MONOTONE_DECREASING], 2
                ),
            ),
            alpha=0.7,
        )
        assert_equal(model.booster.monotone.signs[0], MONOTONE_INCREASING)
        assert_equal(model.booster.monotone.signs[1], MONOTONE_DECREASING)
        var grid = _predict_grid(model, xs)
        _assert_nondecreasing_in_f0(grid, len(xs))
        _assert_nonincreasing_in_f1(grid, len(xs))


def test_multiclass_raw_scores_are_monotone() raises:
    # The documented multiclass policy: every class's raw score is monotone in
    # a constrained feature. Softmax probabilities are not covered by the
    # guarantee, so they are not asserted here.
    var features = _grid_features()
    var shape = _v_shape_target(features)
    var n = _SIDE * _SIDE
    var labels = List[Int](capacity=n)
    for r in range(n):
        if shape[r] < -3.0:
            labels.append(0)
        elif shape[r] < 3.0:
            labels.append(1)
        else:
            labels.append(2)

    var model = fit_multiclass(
        features,
        n,
        2,
        labels,
        3,
        _params(
            15,
            MonotoneConstraints.from_signs(
                [MONOTONE_INCREASING, MONOTONE_DECREASING], 2
            ),
        ),
    )
    # Raw per-class scores over the grid, flattened as
    # [(i * side + j) * 3 + k], filled in one pass.
    var xs = _query_grid()
    var side = len(xs)
    var raw = List[Float64](capacity=side * side * 3)
    for i in range(side):
        for j in range(side):
            var row: List[Float64] = [xs[i], xs[j]]
            var scores = model.predict_raw(row)
            for k in range(3):
                raw.append(scores[k])
    for k in range(3):
        for j in range(side):
            for i in range(side - 1):
                assert_true(
                    raw[(i * side + j) * 3 + k]
                    <= raw[((i + 1) * side + j) * 3 + k]
                )
        for i in range(side):
            for j in range(side - 1):
                assert_true(
                    raw[(i * side + j) * 3 + k]
                    >= raw[(i * side + j + 1) * 3 + k]
                )


def test_tree_level_subtree_ordering() raises:
    # The structural invariant behind the guarantee: for a node split on a
    # constrained feature, every leaf value in its low-side subtree is at or
    # below every leaf value in its high-side subtree. Checked on one grown
    # tree, together with the intervals `node_bounds` recovers.
    var features = _grid_features()
    var target = _v_shape_target(features)
    var n = _SIDE * _SIDE
    var data = bin_equal_width(features, n, 2, 8)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for r in range(n):
        grad.append(-target[r])
        hess.append(1.0)

    var signs: List[Int] = [MONOTONE_INCREASING, MONOTONE_DECREASING]
    var params = TreeParams(
        16,
        1,
        1.0,
        1e-3,
        0.0,
        monotone=MonotoneConstraints.from_signs(signs, 2),
    )
    var tree = grow_tree(data, grad, hess, params)
    assert_true(tree.n_leaves > 2)

    var bounds = node_bounds(tree, signs)
    assert_equal(len(bounds), len(tree.feature))
    for node in range(len(tree.feature)):
        if tree.feature[node] < 0:
            # Every leaf value sits inside its own interval.
            assert_true(tree.value[node] >= bounds[node].lo)
            assert_true(tree.value[node] <= bounds[node].hi)
            continue
        var sign = signs[tree.feature[node]]
        var low = _subtree_leaves(tree, tree.left[node])
        var high = _subtree_leaves(tree, tree.right[node])
        for a in range(len(low)):
            for b in range(len(high)):
                if sign == MONOTONE_INCREASING:
                    assert_true(low[a] <= high[b])
                else:
                    assert_true(low[a] >= high[b])


def _subtree_leaves(tree: Tree, root: Int) -> List[Float64]:
    var values = List[Float64]()
    var stack: List[Int] = [root]
    while len(stack) > 0:
        var node = stack.pop()
        if tree.feature[node] < 0:
            values.append(tree.value[node])
            continue
        stack.append(tree.left[node])
        stack.append(tree.right[node])
    return values^


def test_serialization_round_trip_keeps_constraints() raises:
    var signs: List[Int] = [MONOTONE_INCREASING, MONOTONE_DECREASING]
    var model = _fit_v_shape(MonotoneConstraints.from_signs(signs, 2), 20)
    save_model(model, _TMP_PATH)
    var content = open(_TMP_PATH, "r").read()
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)

    assert_true(content.find("monotone") >= 0)
    assert_equal(len(loaded.booster.monotone.signs), 2)
    assert_equal(loaded.booster.monotone.signs[0], MONOTONE_INCREASING)
    assert_equal(loaded.booster.monotone.signs[1], MONOTONE_DECREASING)

    # Predictions still round-trip bit-exactly, and are still monotone.
    var xs = _query_grid()
    var before = _predict_grid(model, xs)
    var after = _predict_grid(loaded, xs)
    for k in range(len(before)):
        assert_equal(before[k], after[k])
    _assert_nondecreasing_in_f0(after, len(xs))
    _assert_nonincreasing_in_f1(after, len(xs))


def test_unconstrained_model_file_omits_the_section() raises:
    # A model with no constraints must serialize to the same bytes it did
    # before the section existed, so the section is written only when there is
    # something to record.
    var model = _fit_v_shape(MonotoneConstraints(), 10)
    save_model(model, _TMP_PATH)
    var content = open(_TMP_PATH, "r").read()
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)

    assert_true(content.find("monotone") < 0)
    assert_equal(len(loaded.booster.monotone.signs), 0)
    assert_true(not loaded.booster.monotone.is_active())


def test_bounds_and_child_intervals() raises:
    var unbounded = OutputBounds.unbounded()
    assert_true(not unbounded.is_active())
    assert_equal(unbounded.clamp(1e300), 1e300)
    assert_equal(unbounded.clamp(-1e300), -1e300)

    var narrow = OutputBounds(-1.0, 2.0)
    assert_true(narrow.is_active())
    assert_equal(narrow.clamp(-5.0), -1.0)
    assert_equal(narrow.clamp(5.0), 2.0)
    assert_equal(narrow.clamp(0.5), 0.5)

    # An increasing split gives the low side the lower half.
    var up = child_bounds(narrow, MONOTONE_INCREASING, 0.0, 1.0)
    assert_equal(up.left.lo, -1.0)
    assert_equal(up.left.hi, 0.5)
    assert_equal(up.right.lo, 0.5)
    assert_equal(up.right.hi, 2.0)

    # A decreasing split swaps the halves.
    var down = child_bounds(narrow, MONOTONE_DECREASING, 1.0, 0.0)
    assert_equal(down.left.lo, 0.5)
    assert_equal(down.left.hi, 2.0)
    assert_equal(down.right.lo, -1.0)
    assert_equal(down.right.hi, 0.5)

    # An unconstrained split passes the interval down unchanged.
    var free = child_bounds(narrow, MONOTONE_FREE, 10.0, -10.0)
    assert_equal(free.left.lo, -1.0)
    assert_equal(free.left.hi, 2.0)
    assert_equal(free.right.lo, -1.0)
    assert_equal(free.right.hi, 2.0)

    assert_true(violates(MONOTONE_INCREASING, 1.0, 0.0))
    assert_true(not violates(MONOTONE_INCREASING, 0.0, 1.0))
    assert_true(violates(MONOTONE_DECREASING, 0.0, 1.0))
    assert_true(not violates(MONOTONE_DECREASING, 1.0, 0.0))
    assert_true(not violates(MONOTONE_FREE, 1.0, 0.0))
    assert_true(not violates(MONOTONE_FREE, 0.0, 1.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
