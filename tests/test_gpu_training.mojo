"""End-to-end GPU training equivalence.

GPU histograms carry Float32 fixed-point precision, so trained-model
agreement with the Float64 CPU trainer is tolerance-based, not bit-exact;
counts and tree shapes must match exactly when split decisions agree. The
GPU trainer itself must be bit-deterministic run to run. Skips (passing)
when no accelerator is present so the suite stays green on CPU-only
machines.
"""

from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.bagging import BaggingParams, sample_rows
from mojotrees.binning import bin_equal_width
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BoosterParams,
    train,
)
from mojotrees.goss import GossParams
from mojotrees.histogram import build_histogram_subset
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.monotone import MonotoneConstraints
from mojotrees.sampling import select_tree_features, selection_count
from mojotrees.train_gpu import train_gpu
from mojotrees.tree import TreeParams
from support import _make_features, _uniform


def _regression_target(
    features: List[Float64], n_rows: Int
) -> List[Float64]:
    """Strong, distinct per-feature effects so CPU and GPU split decisions
    do not sit on knife-edge gain ties."""
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var x0 = features[0 * n_rows + r]
        var x1 = features[1 * n_rows + r]
        var x2 = features[2 * n_rows + r]
        var x3 = features[3 * n_rows + r]
        y.append(4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x3 - 0.5))
    return y^


def test_gpu_subset_histogram_and_partition_match_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 8_000
        var n_features = 5
        var n_bins = 64
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, n_bins)

        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        var g_mag = 0.0
        var h_mag = 0.0
        var base = UInt64(n_rows * n_features)
        for r in range(n_rows):
            var g = 2.0 * _uniform(base + UInt64(r)) - 1.0
            var h = _uniform(base + UInt64(n_rows + r)) + 0.01
            grad.append(g)
            hess.append(h)
            g_mag += abs(g)
            h_mag += abs(h)

        # Split the root on (feature 0, bin <= 31) device-side, then build
        # both children's histograms by leaf filter.
        var builder = GpuHistogramBuilder(data)
        builder.upload_gradients(grad, hess)
        builder.begin_tree()
        builder.apply_split(0, 31, 0, 1, 2)
        var gpu_left = builder.build_leaf(1)
        var gpu_right = builder.build_leaf(2)

        # CPU reference: partition rows on the host, build subsets directly.
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        for r in range(n_rows):
            if data.bin_at(r, 0) <= 31:
                left_rows.append(r)
            else:
                right_rows.append(r)
        var cpu_left = build_histogram_subset(data, grad, hess, left_rows)
        var cpu_right = build_histogram_subset(data, grad, hess, right_rows)

        for i in range(n_features * n_bins):
            assert_equal(cpu_left.count[i], gpu_left.count[i])
            assert_equal(cpu_right.count[i], gpu_right.count[i])
            assert_true(
                abs(cpu_left.grad[i] - gpu_left.grad[i]) <= 1e-4 * g_mag + 1e-6
            )
            assert_true(
                abs(cpu_left.hess[i] - gpu_left.hess[i]) <= 1e-4 * h_mag + 1e-6
            )
            assert_true(
                abs(cpu_right.grad[i] - gpu_right.grad[i])
                <= 1e-4 * g_mag + 1e-6
            )
            assert_true(
                abs(cpu_right.hess[i] - gpu_right.hess[i])
                <= 1e-4 * h_mag + 1e-6
            )


def test_gpu_regression_training_matches_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(15, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var cpu = train(data, target, SQUARED_ERROR, params)
        var gpu = train_gpu(data, target, SQUARED_ERROR, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        var cpu_sse = 0.0
        var gpu_sse = 0.0
        for r in range(n_rows):
            var pc = cpu.predict_row(data, r)
            var pg = gpu.predict_row(data, r)
            assert_true(abs(pc - pg) <= 1e-3)
            cpu_sse += (pc - target[r]) * (pc - target[r])
            gpu_sse += (pg - target[r]) * (pg - target[r])
        # Same-quality fit, not just pointwise agreement.
        assert_true(abs(cpu_sse - gpu_sse) <= 1e-3 * (cpu_sse + 1e-12))


def test_gpu_max_depth_matches_cpu() raises:
    """The depth limit is a function of tree shape alone, so both backends
    must cut growth at exactly the same leaves. Tree shape is asserted
    exactly (depth and leaf count per tree); only the values carry the GPU's
    Float32 tolerance."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var bounded = BoosterParams(
            15, 0.1, TreeParams(31, 20, 1.0, 1e-3, max_depth=3)
        )
        var cpu = train(data, target, SQUARED_ERROR, bounded)
        var gpu = train_gpu(data, target, SQUARED_ERROR, bounded)

        assert_equal(len(cpu.trees), len(gpu.trees))
        var deepest = 0
        for t in range(len(cpu.trees)):
            assert_true(cpu.trees[t].depth() <= 3)
            assert_equal(cpu.trees[t].depth(), gpu.trees[t].depth())
            assert_equal(cpu.trees[t].n_leaves, gpu.trees[t].n_leaves)
            if cpu.trees[t].depth() > deepest:
                deepest = cpu.trees[t].depth()
        # The limit has to actually bind, or the comparison proves nothing.
        assert_equal(deepest, 3)

        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-3
            )

        # Unlimited depth on the same data grows deeper than the limit, so
        # the bounded run above was genuinely constrained on both backends.
        var free = BoosterParams(15, 0.1, TreeParams(31, 20, 1.0, 1e-3))
        var cpu_free = train(data, target, SQUARED_ERROR, free)
        var gpu_free = train_gpu(data, target, SQUARED_ERROR, free)
        var free_deepest = 0
        for t in range(len(cpu_free.trees)):
            assert_equal(cpu_free.trees[t].depth(), gpu_free.trees[t].depth())
            if cpu_free.trees[t].depth() > free_deepest:
                free_deepest = cpu_free.trees[t].depth()
        assert_true(free_deepest > 3)


def test_gpu_binary_training_matches_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var x0 = features[0 * n_rows + r]
            var x1 = features[1 * n_rows + r]
            var x2 = features[2 * n_rows + r]
            target.append(
                1.0 if 2.0 * x0 - x1 + 0.5 * x2 > 0.7 else 0.0
            )
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(15, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var cpu = train(data, target, BINARY_LOGISTIC, params)
        var gpu = train_gpu(data, target, BINARY_LOGISTIC, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        for r in range(n_rows):
            var pc = cpu.predict_row(data, r)
            var pg = gpu.predict_row(data, r)
            assert_true(abs(pc - pg) <= 1e-3)


def test_gpu_training_is_deterministic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(8, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var a = train_gpu(data, target, SQUARED_ERROR, params)
        var b = train_gpu(data, target, SQUARED_ERROR, params)

        assert_equal(len(a.trees), len(b.trees))
        # Integer accumulation makes repeat GPU training bit-identical.
        for r in range(n_rows):
            assert_equal(
                a.predict_raw_row(data, r), b.predict_raw_row(data, r)
            )


def test_gpu_l1_regularized_training_matches_cpu() raises:
    """`lambda_l1` is applied host-side to downloaded histogram sums, so the
    GPU trainer must reproduce the CPU trainer's regularized splits and
    shrunken leaf values, not just the unregularized ones."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var l1 = BoosterParams(15, 0.1, TreeParams(15, 20, 1.0, 1e-3, 2.0))
        var cpu = train(data, target, SQUARED_ERROR, l1)
        var gpu = train_gpu(data, target, SQUARED_ERROR, l1)

        assert_equal(len(cpu.trees), len(gpu.trees))
        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-3
            )

        # The regularizer must actually bite: leaf values shrink toward
        # zero, so the regularized fit moves less far from the base score.
        var plain = BoosterParams(15, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var gpu_plain = train_gpu(data, target, SQUARED_ERROR, plain)
        var l1_travel = 0.0
        var plain_travel = 0.0
        for r in range(n_rows):
            l1_travel += abs(
                gpu.predict_raw_row(data, r) - gpu.base_score
            )
            plain_travel += abs(
                gpu_plain.predict_raw_row(data, r) - gpu_plain.base_score
            )
        assert_true(l1_travel < plain_travel)


def test_gpu_bagged_leaf_histogram_matches_cpu_subset() raises:
    """Out-of-bag rows are parked at a leaf id no build can target, so a
    bagged root histogram must equal the CPU subset histogram over the bag,
    and a split under it must partition the bag alone."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 8_000
        var n_features = 5
        var n_bins = 64
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, n_bins)

        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        var g_mag = 0.0
        var h_mag = 0.0
        var base = UInt64(n_rows * n_features)
        for r in range(n_rows):
            var g = 2.0 * _uniform(base + UInt64(r)) - 1.0
            var h = _uniform(base + UInt64(n_rows + r)) + 0.01
            grad.append(g)
            hess.append(h)
            g_mag += abs(g)
            h_mag += abs(h)

        var bag = List[Int]()
        sample_rows(BaggingParams(0.5, 1, 2024), n_rows, 0, bag)

        var builder = GpuHistogramBuilder(data)
        builder.upload_gradients(grad, hess)
        builder.begin_tree(bag)
        var gpu_root = builder.build_leaf(0)
        builder.apply_split(0, 31, 0, 1, 2)
        var gpu_left = builder.build_leaf(1)

        # CPU reference over the same bag.
        var cpu_root = build_histogram_subset(data, grad, hess, bag)
        var left_rows = List[Int]()
        for i in range(len(bag)):
            if data.bin_at(bag[i], 0) <= 31:
                left_rows.append(bag[i])
        var cpu_left = build_histogram_subset(data, grad, hess, left_rows)

        for i in range(n_features * n_bins):
            assert_equal(cpu_root.count[i], gpu_root.count[i])
            assert_equal(cpu_left.count[i], gpu_left.count[i])
            assert_true(
                abs(cpu_root.grad[i] - gpu_root.grad[i]) <= 1e-4 * g_mag + 1e-6
            )
            assert_true(
                abs(cpu_root.hess[i] - gpu_root.hess[i]) <= 1e-4 * h_mag + 1e-6
            )
            assert_true(
                abs(cpu_left.grad[i] - gpu_left.grad[i]) <= 1e-4 * g_mag + 1e-6
            )

        # Counts are exact integers: the bag, and only the bag, was counted.
        var counted = 0
        for b in range(n_bins):
            counted += gpu_root.count[b]
        assert_equal(counted, len(bag))


def test_gpu_bagged_training_matches_cpu() raises:
    """Both trainers draw bags from the same sampler with the same seed and
    schedule, so they grow every round on identical rows; only Float32
    histogram precision separates the fitted models."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(15, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var bagging = BaggingParams(0.6, 2, 99)
        var cpu = train(data, target, SQUARED_ERROR, params, [], 0.9, bagging)
        var gpu = train_gpu(
            data, target, SQUARED_ERROR, params, [], 0.9, bagging
        )

        assert_equal(len(cpu.trees), len(gpu.trees))
        for t in range(len(cpu.trees)):
            # Identical rows in, identical tree shape out.
            assert_equal(cpu.trees[t].n_leaves, gpu.trees[t].n_leaves)
            assert_equal(
                len(cpu.trees[t].feature), len(gpu.trees[t].feature)
            )
            for i in range(len(cpu.trees[t].feature)):
                assert_equal(
                    cpu.trees[t].feature[i], gpu.trees[t].feature[i]
                )
                assert_equal(
                    cpu.trees[t].threshold_bin[i],
                    gpu.trees[t].threshold_bin[i],
                )
        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-3
            )


def test_gpu_bagged_training_is_deterministic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(8, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var bagging = BaggingParams(0.5, 1, 7)
        var a = train_gpu(
            data, target, SQUARED_ERROR, params, [], 0.9, bagging
        )
        var b = train_gpu(
            data, target, SQUARED_ERROR, params, [], 0.9, bagging
        )
        assert_equal(len(a.trees), len(b.trees))
        for r in range(n_rows):
            assert_equal(
                a.predict_raw_row(data, r), b.predict_raw_row(data, r)
            )


def test_gpu_goss_first_round_matches_cpu() raises:
    """GOSS ranks rows on the host from Float64 gradients, so at round 0,
    where both backends start from the same base score and therefore the
    same gradients, the two must sample exactly the same rows and grow the
    same tree. Later rounds are gradient-dependent, so Float32 histograms
    can move the sample; that is what the deterministic and quality checks
    below cover instead."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(1, 0.1, TreeParams(15, 20, 1.0, 1e-3))
        var goss = GossParams.enable(0.2, 0.1, 3, 0)
        var cpu = train(
            data, target, SQUARED_ERROR, params, [], 0.9,
            BaggingParams.disabled(), goss,
        )
        var gpu = train_gpu(
            data, target, SQUARED_ERROR, params, [], 0.9,
            BaggingParams.disabled(), goss,
        )

        assert_equal(len(cpu.trees), len(gpu.trees))
        assert_equal(cpu.trees[0].n_leaves, gpu.trees[0].n_leaves)
        assert_equal(len(cpu.trees[0].feature), len(gpu.trees[0].feature))
        for i in range(len(cpu.trees[0].feature)):
            assert_equal(cpu.trees[0].feature[i], gpu.trees[0].feature[i])
            assert_equal(
                cpu.trees[0].threshold_bin[i], gpu.trees[0].threshold_bin[i]
            )
        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-3
            )


def test_gpu_goss_training_is_deterministic_and_fits() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(20, 0.2, TreeParams(15, 20, 1.0, 1e-3))
        var goss = GossParams.enable(0.2, 0.1, 3, 0)
        var a = train_gpu(
            data, target, SQUARED_ERROR, params, [], 0.9,
            BaggingParams.disabled(), goss,
        )
        var b = train_gpu(
            data, target, SQUARED_ERROR, params, [], 0.9,
            BaggingParams.disabled(), goss,
        )
        assert_equal(len(a.trees), len(b.trees))
        for r in range(n_rows):
            assert_equal(
                a.predict_raw_row(data, r), b.predict_raw_row(data, r)
            )

        # Sampled GPU training must still fit: comparing against the
        # constant base-score model keeps this a quality check rather than a
        # bit-comparison the Float32 histograms cannot support.
        var mean_target = 0.0
        for r in range(n_rows):
            mean_target += target[r]
        mean_target /= Float64(n_rows)
        var model_sse = 0.0
        var base_sse = 0.0
        for r in range(n_rows):
            var d = a.predict_row(data, r) - target[r]
            model_sse += d * d
            var db = mean_target - target[r]
            base_sse += db * db
        assert_true(model_sse < 0.25 * base_sse)


def test_gpu_feature_subsampling_matches_cpu() raises:
    """Both backends must consume the same subsampled feature sets: every
    split on either backend comes from that tree's draw, and the two fits
    agree to GPU precision."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 12
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(
            12,
            0.1,
            TreeParams(
                15,
                20,
                1.0,
                1e-3,
                0.0,
                feature_fraction=0.5,
                feature_fraction_bynode=0.7,
                feature_fraction_seed=13,
            ),
        )
        var cpu = train(data, target, SQUARED_ERROR, params)
        var gpu = train_gpu(data, target, SQUARED_ERROR, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        for t in range(len(gpu.trees)):
            var selected = select_tree_features(n_features, 0.5, 13, t)
            assert_equal(len(selected), selection_count(n_features, 0.5))
            var candidates = List[Int]()
            for node in range(len(gpu.trees[t].feature)):
                candidates.append(gpu.trees[t].feature[node])
            for node in range(len(cpu.trees[t].feature)):
                candidates.append(cpu.trees[t].feature[node])
            for i in range(len(candidates)):
                var f = candidates[i]
                if f < 0:
                    continue
                var allowed = False
                for j in range(len(selected)):
                    if selected[j] == f:
                        allowed = True
                assert_true(allowed)

        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-3
            )


def test_gpu_subsampled_training_is_deterministic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 10
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = bin_equal_width(features, n_rows, n_features, 64)

        var params = BoosterParams(
            8,
            0.1,
            TreeParams(
                15,
                20,
                1.0,
                1e-3,
                0.0,
                feature_fraction=0.4,
                feature_fraction_bynode=0.6,
                feature_fraction_seed=5,
            ),
        )
        var a = train_gpu(data, target, SQUARED_ERROR, params)
        var b = train_gpu(data, target, SQUARED_ERROR, params)
        assert_equal(len(a.trees), len(b.trees))
        for r in range(n_rows):
            assert_equal(
                a.predict_raw_row(data, r), b.predict_raw_row(data, r)
            )


def test_gpu_monotone_constraints_match_cpu() raises:
    """Monotonic constraints are enforced host-side, on downloaded histograms,
    so the GPU trainer must produce a monotone model too, and the same one the
    CPU trainer does to Float32 tolerance."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var n_bins = 32
        var features = _make_features(n_rows, n_features)
        # A V shape in feature 0 and an inverted V in feature 1, so an
        # unconstrained fit is monotone in neither.
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var x0 = features[0 * n_rows + r] - 0.5
            var x1 = features[1 * n_rows + r] - 0.5
            target.append(
                4.0 * x0 * x0 - 3.0 * x1 * x1 + features[2 * n_rows + r]
            )
        var data = bin_equal_width(features, n_rows, n_features, n_bins)

        var signs: List[Int] = [1, -1, 0, 0, 0, 0]
        var params = BoosterParams(
            15,
            0.1,
            TreeParams(
                15,
                20,
                1.0,
                1e-3,
                0.0,
                monotone=MonotoneConstraints.from_signs(signs, n_features),
            ),
        )
        var cpu = train(data, target, SQUARED_ERROR, params)
        var gpu = train_gpu(data, target, SQUARED_ERROR, params)
        assert_equal(len(cpu.trees), len(gpu.trees))

        # Walk feature 0's bins upward and feature 1's downward, holding the
        # rest at a fixed bin, and check each model steps the right way. Bin
        # ids are what the trees compare against, so this is exactly the
        # monotonicity the trees encode, and it is asserted per backend: the
        # two need not grow identical trees, they each need to be monotone.
        var bins = List[Int](capacity=n_features)
        for _ in range(n_features):
            bins.append(n_bins // 2)
        for b in range(n_bins - 1):
            bins[0] = b
            var cpu_lo = cpu.predict_bins(bins)
            var gpu_lo = gpu.predict_bins(bins)
            bins[0] = b + 1
            assert_true(cpu_lo <= cpu.predict_bins(bins))
            assert_true(gpu_lo <= gpu.predict_bins(bins))
        bins[0] = n_bins // 2
        for b in range(n_bins - 1):
            bins[1] = b
            var cpu_lo = cpu.predict_bins(bins)
            var gpu_lo = gpu.predict_bins(bins)
            bins[1] = b + 1
            assert_true(cpu_lo >= cpu.predict_bins(bins))
            assert_true(gpu_lo >= gpu.predict_bins(bins))

        # And the two backends agree on the data itself, to the same Float32
        # tolerance the other equivalence tests here use.
        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-3
            )

        # The constraint records on the fitted boosters agree.
        assert_equal(len(cpu.monotone.signs), n_features)
        assert_equal(len(gpu.monotone.signs), n_features)
        for f in range(n_features):
            assert_equal(cpu.monotone.signs[f], signs[f])
            assert_equal(gpu.monotone.signs[f], signs[f])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
