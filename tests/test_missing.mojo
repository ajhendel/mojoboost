"""Native missing-value support for numerical features.

Covers the binning contract (reserved bin, NaN routing, infinities), the
two-direction split search, the per-node default direction, agreement between
raw, binned, CPU, and GPU execution, and v2 serialization with v1
backward compatibility.

The LightGBM behavior each test matches was read off LightGBM 4.7 directly;
`bench/compare_missing_lightgbm.py` is the reproducible comparison.
"""

from std.math import isinf, isnan
from std.sys import has_accelerator
from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojoboost.binning import BinnedMatrix, bin_equal_width, fit_bins
from mojoboost.boosting import SQUARED_ERROR, BoosterParams, train
from mojoboost.histogram import Histogram, build_histogram
from mojoboost.model import Model, fit
from mojoboost.serialize import load_model, save_model
from mojoboost.split import find_best_split
from mojoboost.train_gpu import train_gpu
from mojoboost.tree import TreeParams, grow_tree

from std.utils.numerics import inf, nan

comptime NAN = nan[DType.float64]()
comptime INF = inf[DType.float64]()


def _params(num_leaves: Int = 4, n_estimators: Int = 30) -> BoosterParams:
    return BoosterParams(
        n_estimators, 0.3, TreeParams(num_leaves, 1, 1.0, 1e-3, 0.0)
    )


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def test_missing_bin_is_reserved_only_where_needed() raises:
    # Feature 0 has two NaN rows, feature 1 none. Only feature 0 gives up a
    # bin, and its missing bin sits directly above its ordinary bins.
    var features: List[Float64] = [
        0.0, 1.0, NAN, 2.0, 3.0, NAN,  # feature 0
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0,  # feature 1
    ]
    var mapper = fit_bins(features, n_rows=6, n_features=2, max_bins=8)
    assert_true(mapper.has_missing())

    var n_edges_0 = mapper.edge_offsets[1] - mapper.edge_offsets[0]
    assert_equal(mapper.missing_bin[0], n_edges_0 + 1)
    assert_equal(mapper.missing_bin[1], -1)

    # Every ordinary bin of feature 0 stays below its missing bin.
    for v in [0.0, 1.0, 2.0, 3.0]:
        assert_true(mapper.bin_value(0, v) < mapper.missing_bin[0])
    assert_equal(mapper.bin_value(0, NAN), mapper.missing_bin[0])

    # The transform agrees with the scalar path row for row.
    var data = mapper.transform(features, 6)
    assert_equal(data.missing_bin[0], mapper.missing_bin[0])
    for r in range(6):
        assert_equal(data.bin_at(r, 0), mapper.bin_value(0, features[r]))
    assert_true(data.is_missing(2, 0))
    assert_true(data.is_missing(5, 0))
    assert_false(data.is_missing(0, 0))
    assert_false(data.is_missing(0, 1))


def test_missing_values_leave_the_quantile_edges_alone() raises:
    # The same five finite values, with and without NaN rows mixed in, must
    # produce the same edges: NaN never enters a quantile comparison.
    var clean: List[Float64] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var dirty: List[Float64] = [1.0, NAN, 2.0, 3.0, NAN, 4.0, 5.0]
    var a = fit_bins(clean, n_rows=5, n_features=1, max_bins=6)
    var b = fit_bins(dirty, n_rows=7, n_features=1, max_bins=6)
    assert_equal(len(a.edges), len(b.edges))
    for i in range(len(a.edges)):
        assert_equal(a.edges[i], b.edges[i])
    assert_equal(a.missing_bin[0], -1)
    assert_equal(b.missing_bin[0], len(b.edges) + 1)


def test_infinities_bin_as_finite_extremes() raises:
    # +inf and -inf are ordinary values, not missing. Every edge stays finite
    # so the binary search keeps working (LightGBM's Common::AvoidInf).
    var features: List[Float64] = [-INF, 1.0, 2.0, 3.0, INF, NAN]
    var mapper = fit_bins(features, n_rows=6, n_features=1, max_bins=8)
    for i in range(len(mapper.edges)):
        assert_true(not isinf(mapper.edges[i]))
        assert_true(not isnan(mapper.edges[i]))

    var missing = mapper.missing_bin[0]
    assert_true(missing > 0)
    assert_equal(mapper.bin_value(0, -INF), 0)
    var top = mapper.bin_value(0, INF)
    assert_equal(top, missing - 1)
    # +inf lands with the largest finite value, not in the missing bin.
    assert_equal(mapper.bin_value(0, 1e308), top)
    assert_equal(mapper.bin_value(0, NAN), missing)


def test_nan_without_a_reserved_bin_bins_as_zero() raises:
    # A feature with no missing training value reserves nothing, so a NaN at
    # predict time is binned as 0.0. This is LightGBM's rule for a feature
    # whose missing_type is None.
    var features: List[Float64] = [-3.0, -1.0, 1.0, 3.0]
    var mapper = fit_bins(features, n_rows=4, n_features=1, max_bins=8)
    assert_equal(mapper.missing_bin[0], -1)
    assert_false(mapper.has_missing())
    assert_equal(mapper.bin_value(0, NAN), mapper.bin_value(0, 0.0))


def test_use_missing_false_bins_nan_as_zero() raises:
    var features: List[Float64] = [-3.0, -1.0, NAN, 1.0, 3.0]
    var off = fit_bins(
        features, n_rows=5, n_features=1, max_bins=8, use_missing=False
    )
    assert_false(off.has_missing())
    assert_equal(off.missing_bin[0], -1)
    assert_equal(off.bin_value(0, NAN), off.bin_value(0, 0.0))

    var on = fit_bins(features, n_rows=5, n_features=1, max_bins=8)
    assert_true(on.missing_bin[0] >= 0)
    assert_equal(on.bin_value(0, NAN), on.missing_bin[0])


def test_clean_data_binning_is_unchanged() raises:
    # With missing support on but no NaN anywhere, binning must be exactly
    # what it was before: 8 distinct values into 8 bins is still the identity.
    var features: List[Float64] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    var mapper = fit_bins(features, n_rows=8, n_features=1, max_bins=8)
    assert_equal(mapper.missing_bin[0], -1)
    var data = mapper.transform(features, 8)
    for r in range(8):
        assert_equal(data.bin_at(r, 0), r)


def test_all_missing_column() raises:
    # A column of nothing but NaN has no ordinary values to fit edges from;
    # it gets one empty ordinary bin and its missing bin above it.
    var features: List[Float64] = [NAN, NAN, NAN, NAN]
    var mapper = fit_bins(features, n_rows=4, n_features=1, max_bins=8)
    assert_equal(mapper.edge_offsets[1] - mapper.edge_offsets[0], 0)
    assert_equal(mapper.missing_bin[0], 1)
    var data = mapper.transform(features, 4)
    for r in range(4):
        assert_true(data.is_missing(r, 0))


def test_equal_width_binning_has_no_missing_support() raises:
    var features: List[Float64] = [0.0, 1.0, NAN, 3.0]
    var data = bin_equal_width(features, n_rows=4, n_features=1, n_bins=4)
    assert_equal(data.missing_bin[0], -1)
    assert_equal(data.bin_at(2, 0), 0)


# --------------------------------------------------------------------------
# Split search
# --------------------------------------------------------------------------


def _hist(
    grad: List[Float64], hess: List[Float64], count: List[Int]
) -> Histogram:
    return Histogram(grad.copy(), hess.copy(), count.copy(), 1, len(grad))


def test_split_sends_missing_to_the_better_side() raises:
    # Four ordinary bins plus a missing bin at index 4. Ordinary gradients
    # flip sign between bins 1 and 2, and the missing rows carry the negative
    # (left) gradient, so the search must route missing left.
    var grad: List[Float64] = [-2.0, -2.0, 2.0, 2.0, -2.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0]
    var count: List[Int] = [1, 1, 1, 1, 1]
    var missing_bins: List[Int] = [4]
    var split = find_best_split(
        _hist(grad, hess, count), 1.0, 1e-3, 0, missing_bins=missing_bins
    )
    assert_true(split.found)
    assert_equal(split.bin, 1)
    assert_true(split.default_left)

    # Flip the missing rows to the positive (right) gradient and the
    # direction must flip with them.
    grad[4] = 2.0
    var flipped = find_best_split(
        _hist(grad, hess, count), 1.0, 1e-3, 0, missing_bins=missing_bins
    )
    assert_true(flipped.found)
    assert_equal(flipped.bin, 1)
    assert_false(flipped.default_left)


def test_split_can_isolate_the_missing_rows() raises:
    # Missingness alone carries the signal: every ordinary bin has the same
    # gradient and only the missing bin differs. The winning split puts every
    # ordinary bin left and the missing rows alone on the right, which is the
    # split LightGBM reports as threshold = 1e300.
    var grad: List[Float64] = [-1.0, -1.0, -1.0, -1.0, 12.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 4.0]
    var count: List[Int] = [1, 1, 1, 1, 4]
    var missing_bins: List[Int] = [4]
    var split = find_best_split(
        _hist(grad, hess, count), 1.0, 1e-3, 0, missing_bins=missing_bins
    )
    assert_true(split.found)
    # Bin 3 is the last ordinary bin, so left = all ordinary, right = missing.
    assert_equal(split.bin, 3)
    assert_false(split.default_left)


def test_node_without_missing_rows_defaults_left() raises:
    # A feature with a missing bin whose node holds no missing rows: both
    # directions score identically, and the tie keeps default_left, which is
    # what LightGBM records (verified against LightGBM 4.7).
    var grad: List[Float64] = [-2.0, -2.0, 2.0, 2.0, 0.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0, 0.0]
    var count: List[Int] = [1, 1, 1, 1, 0]
    var missing_bins: List[Int] = [4]
    var split = find_best_split(
        _hist(grad, hess, count), 1.0, 1e-3, 0, missing_bins=missing_bins
    )
    assert_true(split.found)
    assert_equal(split.bin, 1)
    assert_true(split.default_left)


def test_search_without_missing_bins_is_unchanged() raises:
    # The same histogram with no missing bin declared must give the same
    # threshold and gain as the missing-aware scan does over its ordinary
    # bins, and report no direction.
    var grad: List[Float64] = [-2.0, -2.0, 2.0, 2.0]
    var hess: List[Float64] = [1.0, 1.0, 1.0, 1.0]
    var count: List[Int] = [1, 1, 1, 1]
    var plain = find_best_split(_hist(grad, hess, count), 1.0, 1e-3, 0)
    assert_true(plain.found)
    assert_equal(plain.bin, 1)
    assert_false(plain.default_left)

    var with_empty_missing: List[Float64] = [-2.0, -2.0, 2.0, 2.0, 0.0]
    var h5: List[Float64] = [1.0, 1.0, 1.0, 1.0, 0.0]
    var c5: List[Int] = [1, 1, 1, 1, 0]
    var missing_bins: List[Int] = [4]
    var aware = find_best_split(
        _hist(with_empty_missing, h5, c5), 1.0, 1e-3, 0,
        missing_bins=missing_bins,
    )
    assert_equal(aware.bin, plain.bin)
    assert_equal(aware.gain, plain.gain)


def test_search_rejects_a_mismatched_missing_table() raises:
    var grad: List[Float64] = [-1.0, 1.0]
    var hess: List[Float64] = [1.0, 1.0]
    var count: List[Int] = [1, 1]
    var bad: List[Int] = [0, 0, 0]
    var raised = False
    try:
        _ = find_best_split(
            _hist(grad, hess, count), 1.0, 1e-3, 0, missing_bins=bad
        )
    except:
        raised = True
    assert_true(raised)


# --------------------------------------------------------------------------
# Training, prediction, and the CPU/GPU boundary
# --------------------------------------------------------------------------


def _missingness_dataset(
    n_rows: Int,
) -> Tuple[List[Float64], List[Float64]]:
    """One feature over [0, 1) where every third row is missing, and a target
    that depends on missingness alone: the pattern is the signal, and the
    observed values carry none."""
    var features = List[Float64](capacity=n_rows)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var missing = r % 3 == 0
        features.append(NAN if missing else Float64(r % 17) / 17.0)
        target.append(5.0 if missing else -5.0)
    return (features^, target^)


def test_missingness_alone_is_learnable() raises:
    var n_rows = 300
    var pair = _missingness_dataset(n_rows)
    var model = fit(
        pair[0], n_rows, 1, pair[1], SQUARED_ERROR, _params(4, 60), max_bins=16
    )

    # A missing row and an observed row must land on opposite sides.
    var missing_row: List[Float64] = [NAN]
    assert_true(abs(model.predict(missing_row) - 5.0) < 0.5)
    for v in [0.0, 0.25, 0.5, 0.99]:
        var row: List[Float64] = [v]
        assert_true(abs(model.predict(row) + 5.0) < 0.5)

    # Some node actually routes on the missing bin.
    var routes_missing = False
    for t in range(len(model.booster.trees)):
        ref tree = model.booster.trees[t]
        for i in range(len(tree.feature)):
            if tree.feature[i] >= 0 and tree.missing_bin[i] >= 0:
                routes_missing = True
    assert_true(routes_missing)


def test_missingness_is_learnable_alongside_a_real_signal() raises:
    # Feature 0 is a clean linear signal, feature 1 is missing for half the
    # rows and its missingness shifts the target. Both must be picked up.
    var n_rows = 400
    var features = List[Float64](capacity=2 * n_rows)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        features.append(Float64(r) / Float64(n_rows))
    for r in range(n_rows):
        features.append(NAN if r % 2 == 0 else Float64(r % 7))
    for r in range(n_rows):
        var base = 3.0 * Float64(r) / Float64(n_rows)
        target.append(base + (2.0 if r % 2 == 0 else -2.0))

    var model = fit(
        features, n_rows, 2, target, SQUARED_ERROR, _params(16, 80),
        max_bins=32,
    )
    var lo_missing: List[Float64] = [0.1, NAN]
    var lo_present: List[Float64] = [0.1, 3.0]
    var hi_missing: List[Float64] = [0.9, NAN]
    assert_true(model.predict(lo_missing) > model.predict(lo_present))
    assert_true(model.predict(hi_missing) > model.predict(lo_missing))


def test_raw_and_binned_prediction_agree_on_missing_rows() raises:
    # Predicting from raw values (which re-bins through the mapper) must give
    # exactly what predicting from the training matrix gives, missing rows
    # included.
    var n_rows = 200
    var pair = _missingness_dataset(n_rows)
    var mapper = fit_bins(pair[0], n_rows, 1, max_bins=16)
    var data = mapper.transform(pair[0], n_rows)
    var booster = train(data, pair[1], SQUARED_ERROR, _params(4, 20))

    for r in range(n_rows):
        var row: List[Float64] = [pair[0][r]]
        assert_equal(
            booster.predict_raw_bins(mapper.bin_row(row)),
            booster.predict_raw_row(data, r),
        )


def test_default_direction_is_applied_when_training_and_predicting() raises:
    # Every internal node's routing of the training rows must match the leaf
    # the tree assigns them, so the partition used to build histograms and
    # the one used to predict cannot drift apart.
    var n_rows = 120
    var pair = _missingness_dataset(n_rows)
    var mapper = fit_bins(pair[0], n_rows, 1, max_bins=8)
    var data = mapper.transform(pair[0], n_rows)
    var grad = List[Float64](capacity=n_rows)
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(-pair[1][r])
        hess.append(1.0)
    var tree = grow_tree(data, grad, hess, TreeParams(4, 1, 1.0, 1e-3, 0.0))

    for r in range(n_rows):
        var node = 0
        while tree.feature[node] >= 0:
            var f = tree.feature[node]
            var bin = data.bin_at(r, f)
            var expected_left: Bool
            if bin == data.missing_bin[f]:
                expected_left = tree.default_left[node]
            else:
                expected_left = bin <= tree.threshold_bin[node]
            assert_equal(tree.goes_left(node, bin), expected_left)
            node = tree.left[node] if expected_left else tree.right[node]
        assert_equal(node, tree.leaf_index_row(data, r))


def test_use_missing_false_ignores_the_pattern() raises:
    # With use_missing off, NaN is just the value 0.0, so a target driven by
    # missingness alone can no longer be separated the way it is above.
    var n_rows = 300
    var pair = _missingness_dataset(n_rows)
    var model = fit(
        pair[0], n_rows, 1, pair[1], SQUARED_ERROR, _params(4, 60),
        max_bins=16, use_missing=False,
    )
    assert_false(model.mapper.has_missing())
    var missing_row: List[Float64] = [NAN]
    # 0.0 is an observed value here, so the NaN row cannot reach the
    # missing-row target of +5.
    assert_true(model.predict(missing_row) < 0.0)


def test_gpu_training_routes_missing_like_the_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 4000
        var n_features = 4
        var features = List[Float64](capacity=n_rows * n_features)
        for f in range(n_features):
            for r in range(n_rows):
                # Feature 1 and 3 carry missing values, on different patterns.
                var v = Float64((r * (f + 3)) % 101) / 101.0
                if (f == 1 and r % 4 == 0) or (f == 3 and r % 7 == 0):
                    features.append(NAN)
                else:
                    features.append(v)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var shift = 2.0 if r % 4 == 0 else -1.0
            target.append(
                shift + 3.0 * Float64((r * 3) % 101) / 101.0
            )

        var mapper = fit_bins(features, n_rows, n_features, max_bins=32)
        assert_true(mapper.has_missing())
        var data = mapper.transform(features, n_rows)

        var params = _params(8, 12)
        var cpu = train(data, target, SQUARED_ERROR, params)
        var gpu = train_gpu(data, target, SQUARED_ERROR, params)
        assert_equal(len(cpu.trees), len(gpu.trees))

        # Split decisions, including the default directions, must agree.
        for t in range(len(cpu.trees)):
            ref a = cpu.trees[t]
            ref b = gpu.trees[t]
            assert_equal(len(a.feature), len(b.feature))
            for i in range(len(a.feature)):
                assert_equal(a.feature[i], b.feature[i])
                assert_equal(a.threshold_bin[i], b.threshold_bin[i])
                assert_equal(a.missing_bin[i], b.missing_bin[i])
                assert_equal(a.default_left[i], b.default_left[i])

        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_raw_row(data, r) - gpu.predict_raw_row(data, r))
                < 1e-3
            )


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


def test_v2_roundtrip_keeps_missing_routing_exact() raises:
    var n_rows = 200
    var pair = _missingness_dataset(n_rows)
    var model = fit(
        pair[0], n_rows, 1, pair[1], SQUARED_ERROR, _params(4, 25), max_bins=16
    )
    var path = String(".test_missing_roundtrip.tmp")
    save_model(model, path)
    var loaded = load_model(path)

    assert_equal(loaded.mapper.missing_bin[0], model.mapper.missing_bin[0])
    assert_true(loaded.mapper.has_missing())
    assert_equal(len(loaded.booster.trees), len(model.booster.trees))
    for t in range(len(model.booster.trees)):
        ref a = model.booster.trees[t]
        ref b = loaded.booster.trees[t]
        for i in range(len(a.feature)):
            assert_equal(a.missing_bin[i], b.missing_bin[i])
            assert_equal(a.default_left[i], b.default_left[i])

    # Predictions on missing and observed rows must be bit-identical.
    var missing_row: List[Float64] = [NAN]
    assert_equal(model.predict(missing_row), loaded.predict(missing_row))
    for r in range(n_rows):
        var row: List[Float64] = [pair[0][r]]
        assert_equal(model.predict(row), loaded.predict(row))


def test_v1_file_still_loads_and_routes_nothing() raises:
    # A hand-written v1 file: one feature, one edge, a single split. It must
    # load, route no missing values, and bin NaN as 0.0.
    var path = String(".test_missing_v1.tmp")
    var content = String(
        "mojoboost v1\n"
        "objective 0\n"
        # learning_rate 1.0, base_score 0.0 as IEEE-754 bit patterns.
        "learning_rate 4607182418800017408\n"
        "base_score 0\n"
        "mapper 1 8 1\n"
        # A single edge at 2.0.
        "4611686018427387904\n"
        "0 1\n"
        "trees 1\n"
        "tree 3 2\n"
        "0 -1 -1\n"
        "0 -1 -1\n"
        "1 -1 -1\n"
        "2 -1 -1\n"
        # Leaf values 0.0, -1.0, 1.0.
        "0 13830554455654793216 4607182418800017408\n"
    )
    with open(path, "w") as f:
        f.write(content)

    var model = load_model(path)
    assert_false(model.mapper.has_missing())
    assert_equal(model.mapper.missing_bin[0], -1)
    ref tree = model.booster.trees[0]
    for i in range(len(tree.feature)):
        assert_equal(tree.missing_bin[i], -1)
        assert_false(tree.default_left[i])

    var low: List[Float64] = [1.0]
    var high: List[Float64] = [3.0]
    var nan_row: List[Float64] = [NAN]
    assert_equal(model.predict(low), -1.0)
    assert_equal(model.predict(high), 1.0)
    # NaN is binned as 0.0, which is below the edge, so it follows `low`.
    assert_equal(model.predict(nan_row), model.predict(low))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
