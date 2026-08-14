"""GPU prediction and device-side validation scoring.

Every assertion here is a CPU-equivalence assertion: the device walks the
same trees the host walks, so it must reach the same leaves and, up to the
Float32 accumulation the device does, report the same scores. Routing is
exact (bins are integers), so a routing bug shows up as a large difference
rather than a small one; the tolerances below are loose enough for Float32
accumulation and far too tight to hide a misrouted row.

Skips (passing) when no accelerator is present, so the file stays green on
CPU-only machines.
"""

from std.math import isnan
from std.sys import has_accelerator
from std.testing import assert_almost_equal, assert_equal, assert_true
from std.testing import TestSuite
from std.utils.numerics import nan

from mojoboost.binning import BinnedMatrix, fit_bins
from mojoboost.boosting import (
    BINARY_LOGISTIC,
    SQUARED_ERROR,
    BoosterParams,
    IterationRange,
    train,
    train_multiclass,
)
from mojoboost.gpu_predict import (
    METRIC_BINARY_LOG_LOSS,
    METRIC_L1,
    METRIC_L2,
    METRIC_MULTICLASS_LOG_LOSS,
    RESPONSE_IDENTITY,
    RESPONSE_SIGMOID,
    RESPONSE_SOFTMAX,
    GpuPredictor,
    flatten_booster,
    flatten_multiclass,
    flatten_trees,
    response_for_objective,
)
from mojoboost.metrics import binary_log_loss, l1, l2, multiclass_log_loss
from mojoboost.tree import Tree, TreeParams


# Float32 accumulation over a few dozen trees: predictions of order 1 agree
# to about this much, and a misrouted row differs by a whole leaf value.
comptime TOL = 1e-4

comptime NAN = nan[DType.float64]()


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


def _make_features(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major deterministic features in [0, 1)."""
    var features = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        features.append(_uniform(UInt64(k)))
    return features^


def _regression_target(
    features: List[Float64], n_rows: Int
) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(
            4.0 * features[0 * n_rows + r]
            - 3.0 * features[1 * n_rows + r]
            + 2.0 * features[2 * n_rows + r] * features[3 * n_rows + r]
        )
    return y^


def _binary_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var score = (
            2.0 * features[0 * n_rows + r] - 1.5 * features[1 * n_rows + r]
        )
        y.append(1.0 if score > 0.4 else 0.0)
    return y^


def _bins_of(data: BinnedMatrix, row: Int) -> List[Int]:
    """One row's per-feature bin ids, the input the host predictors take."""
    var bins = List[Int](capacity=data.n_features)
    for f in range(data.n_features):
        bins.append(data.bin_at(row, f))
    return bins^


def _small_params(n_trees: Int = 12) -> BoosterParams:
    return BoosterParams(n_trees, 0.1, TreeParams(8, 5, 1.0, 1e-3, 0.0))


def _binned(
    features: List[Float64],
    n_rows: Int,
    n_features: Int,
    categorical_features: List[Int] = [],
) raises -> BinnedMatrix:
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        32,
        categorical_features=categorical_features,
    )
    return mapper.transform(features, n_rows)


def test_raw_and_response_match_cpu() raises:
    """The whole-ensemble batch path against `Booster.predict_raw_bins` and
    `predict_bins`, on a logistic model so the response transform is a real
    link rather than the identity."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1500
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _binary_target(features, n_rows)
        var data = _binned(features, n_rows, n_features)
        var booster = train(
            data, target, BINARY_LOGISTIC, _small_params()
        )

        var predictor = GpuPredictor(n_features, 1)
        predictor.upload_ensemble(flatten_booster(booster))
        var full = IterationRange.slice(booster.n_iterations(), 0, 10000)
        var raw = predictor.raw_scores(data, full)
        var resp = predictor.response_scores(
            data, full, response_for_objective(booster.objective)
        )
        assert_equal(len(raw), n_rows)
        assert_equal(len(resp), n_rows)

        assert_equal(response_for_objective(BINARY_LOGISTIC), RESPONSE_SIGMOID)
        for r in range(n_rows):
            var bins = _bins_of(data, r)
            assert_almost_equal(
                raw[r], booster.predict_raw_bins(bins), atol=TOL
            )
            assert_almost_equal(resp[r], booster.predict_bins(bins), atol=TOL)
            # A probability, not a raw score: the link really was applied.
            assert_true(resp[r] > 0.0 and resp[r] < 1.0)


def test_iteration_ranges_match_cpu() raises:
    """Slices of the ensemble, including where the base score sits: the
    device follows the same `IterationRange` rule the host does, so [0, k)
    and [k, n) must sum to the full raw score."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 800
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = _binned(features, n_rows, n_features)
        var booster = train(data, target, SQUARED_ERROR, _small_params())
        var n_iter = booster.n_iterations()
        assert_true(n_iter > 4)

        var predictor = GpuPredictor(n_features, 1)
        predictor.upload_ensemble(flatten_booster(booster))
        var cut = n_iter // 2
        var head = IterationRange.slice(n_iter, 0, cut)
        var tail = IterationRange.slice(n_iter, cut, n_iter)
        var full = IterationRange.slice(n_iter, 0, n_iter)
        var head_scores = predictor.raw_scores(data, head)
        var tail_scores = predictor.raw_scores(data, tail)
        var full_scores = predictor.raw_scores(data, full)

        assert_true(head.includes_base())
        assert_true(not tail.includes_base())
        for r in range(n_rows):
            var bins = _bins_of(data, r)
            assert_almost_equal(
                head_scores[r],
                booster.predict_raw_bins_range(bins, head),
                atol=TOL,
            )
            assert_almost_equal(
                tail_scores[r],
                booster.predict_raw_bins_range(bins, tail),
                atol=TOL,
            )
            assert_almost_equal(
                head_scores[r] + tail_scores[r], full_scores[r], atol=TOL
            )

        # An empty range that starts past 0 is the zero prediction, and the
        # empty range at 0 is the base-score-only model.
        var empty = IterationRange.slice(n_iter, cut, cut)
        var base_only = IterationRange.slice(n_iter, 0, 0)
        var empty_scores = predictor.raw_scores(data, empty)
        var base_scores = predictor.raw_scores(data, base_only)
        for r in range(n_rows):
            assert_almost_equal(empty_scores[r], 0.0, atol=TOL)
            assert_almost_equal(base_scores[r], booster.base_score, atol=TOL)


def test_missing_and_categorical_routing_match_cpu() raises:
    """Routing, not arithmetic: a categorical feature split by category set
    and a numerical feature with a reserved missing bin. Both take their own
    branch in the kernel's `goes_left`, and both must send a row to the same
    leaf the host sends it to."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1200
        var n_features = 3
        var features = List[Float64](capacity=n_rows * n_features)
        # Feature 0: integer-coded categorical with 6 categories.
        for r in range(n_rows):
            features.append(Float64(Int(_uniform(UInt64(r)) * 6.0)))
        # Feature 1: numerical with one row in eight missing.
        for r in range(n_rows):
            var v = _uniform(UInt64(n_rows + r))
            features.append(NAN if (r % 8) == 0 else v)
        # Feature 2: ordinary numerical.
        for r in range(n_rows):
            features.append(_uniform(UInt64(2 * n_rows + r)))

        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var cat = features[r]
            var x1 = features[n_rows + r]
            var x2 = features[2 * n_rows + r]
            # A strong per-category effect, so category sets really are the
            # splits that get chosen, and a missing value that carries its
            # own signal, so the grower has a reason to split feature 1 and
            # to send its missing bin somewhere in particular.
            var effect = 6.0 if isnan(x1) else 4.0 * x1
            target.append(2.0 * cat - 3.0 * x2 + effect)

        var cats: List[Int] = [0]
        var data = _binned(features, n_rows, n_features, cats)
        assert_true(data.cats.is_cat(0))
        assert_true(data.missing_bin[1] >= 0)
        var booster = train(data, target, SQUARED_ERROR, _small_params(16))

        # The trees actually exercise both branches, or this test proves
        # nothing about routing.
        var saw_categorical = False
        var saw_missing = False
        for t in range(len(booster.trees)):
            ref tree = booster.trees[t]
            for i in range(len(tree.feature)):
                if tree.cat_offset[i] >= 0:
                    saw_categorical = True
                if tree.feature[i] == 1 and tree.missing_bin[i] >= 0:
                    saw_missing = True
        assert_true(saw_categorical)
        assert_true(saw_missing)

        var predictor = GpuPredictor(n_features, 1)
        predictor.upload_ensemble(flatten_booster(booster))
        var full = IterationRange.slice(booster.n_iterations(), 0, 10000)
        var scores = predictor.raw_scores(data, full)
        for r in range(n_rows):
            assert_almost_equal(
                scores[r],
                booster.predict_raw_bins(_bins_of(data, r)),
                atol=TOL,
            )


def test_leaf_indices_match_cpu() raises:
    """Leaf ordinals over a slice, in the layout the host reports them."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 600
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = _binned(features, n_rows, n_features)
        var booster = train(data, target, SQUARED_ERROR, _small_params(10))
        var n_iter = booster.n_iterations()

        var predictor = GpuPredictor(n_features, 1)
        predictor.upload_ensemble(flatten_booster(booster))
        var rng = IterationRange.slice(n_iter, 2, n_iter)
        var leaves = predictor.leaf_indices(data, rng)
        assert_equal(len(leaves), n_rows * rng.n_iterations())
        for r in range(n_rows):
            var expected = booster.leaf_indices_bins(_bins_of(data, r), rng)
            for i in range(rng.n_iterations()):
                assert_equal(leaves[r * rng.n_iterations() + i], expected[i])


def test_multiclass_proba_matches_cpu() raises:
    """Softmax over a round-major ensemble: one thread per (row, class), and
    the row-wise transform on the device."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 900
        var n_features = 4
        var n_classes = 3
        var features = _make_features(n_rows, n_features)
        var labels = List[Int](capacity=n_rows)
        for r in range(n_rows):
            var s = (
                3.0 * features[0 * n_rows + r] - 2.0 * features[1 * n_rows + r]
            )
            if s < 0.0:
                labels.append(0)
            elif s < 1.2:
                labels.append(1)
            else:
                labels.append(2)
        var data = _binned(features, n_rows, n_features)
        var booster = train_multiclass(
            data, labels, n_classes, _small_params(9)
        )
        var n_iter = booster.n_iterations()

        var predictor = GpuPredictor(n_features, n_classes)
        predictor.upload_ensemble(flatten_multiclass(booster))
        var full = IterationRange.slice(n_iter, 0, n_iter)
        var proba = predictor.response_scores(data, full, RESPONSE_SOFTMAX)
        var raw = predictor.raw_scores(data, full)
        assert_equal(len(proba), n_rows * n_classes)

        for r in range(n_rows):
            var bins = _bins_of(data, r)
            var want_raw = booster.predict_raw_bins(bins)
            var want_proba = booster.predict_proba_bins(bins)
            var total = 0.0
            for k in range(n_classes):
                assert_almost_equal(
                    raw[r * n_classes + k], want_raw[k], atol=TOL
                )
                assert_almost_equal(
                    proba[r * n_classes + k], want_proba[k], atol=TOL
                )
                total += proba[r * n_classes + k]
            assert_almost_equal(total, 1.0, atol=TOL)

        # A slice of the iterations takes its softmax over the sliced raw
        # scores, which is what the host does too.
        var head = IterationRange.slice(n_iter, 0, n_iter // 2)
        var sliced = predictor.response_scores(data, head, RESPONSE_SOFTMAX)
        for r in range(n_rows):
            var want = booster.predict_proba_bins_range(_bins_of(data, r), head)
            for k in range(n_classes):
                assert_almost_equal(
                    sliced[r * n_classes + k], want[k], atol=TOL
                )


def test_incremental_validation_and_metrics() raises:
    """The training-time path: a resident validation set whose raw scores are
    advanced one round at a time, scored on the device.

    This is exactly the wiring a trainer needs, so the loop below is written
    the way a trainer would write it: reset the raw scores to the base score
    once, then per round upload only that round's trees and accumulate.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1000
        var n_valid = 700
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = _binned(features, n_rows, n_features)
        var booster = train(data, target, SQUARED_ERROR, _small_params(14))

        # A separate matrix, binned by the same mapper the model was fit
        # with is what a trainer would do; binning it on its own is enough
        # here, since only the bins reach the device.
        var vfeat = _make_features(n_valid, n_features)
        for i in range(len(vfeat)):
            vfeat[i] = 1.0 - vfeat[i]
        var vtarget = _regression_target(vfeat, n_valid)
        var vdata = _binned(vfeat, n_valid, n_features)

        var predictor = GpuPredictor(n_features, 1)
        predictor.set_validation(vdata, vtarget)
        var base: List[Float64] = [booster.base_score]
        predictor.reset_validation(base)

        var host_raw = List[Float64](capacity=n_valid)
        for _ in range(n_valid):
            host_raw.append(booster.base_score)

        for i in range(booster.n_iterations()):
            var round: List[Tree] = [booster.trees[i].copy()]
            var zero: List[Float64] = [0.0]
            predictor.upload_ensemble(
                flatten_trees(round, zero, 1, booster.learning_rate)
            )
            predictor.accumulate_round()
            for r in range(n_valid):
                host_raw[r] += booster.learning_rate * booster.trees[
                    i
                ].predict_bins(_bins_of(vdata, r))

        var device_raw = predictor.validation_raw()
        assert_equal(len(device_raw), n_valid)
        for r in range(n_valid):
            assert_almost_equal(device_raw[r], host_raw[r], atol=TOL)

        # Squared error is on the raw scale for this objective, so the
        # identity link is the right transform and metrics.mojo's `l2` is
        # the reference.
        var gpu_l2 = predictor.validation_metric(METRIC_L2, RESPONSE_IDENTITY)
        assert_almost_equal(gpu_l2, l2(host_raw, vtarget), atol=1e-3)
        var gpu_l1 = predictor.validation_metric(METRIC_L1, RESPONSE_IDENTITY)
        assert_almost_equal(gpu_l1, l1(host_raw, vtarget), atol=1e-3)

        # Scoring twice does not disturb the resident raw scores: the
        # response transform writes to its own buffer.
        var again = predictor.validation_metric(METRIC_L2, RESPONSE_IDENTITY)
        assert_equal(gpu_l2, again)

        # And the whole-ensemble path reproduces the incremental one.
        predictor.upload_ensemble(flatten_booster(booster))
        predictor.score_validation(
            IterationRange.slice(booster.n_iterations(), 0, 10000)
        )
        var rescored = predictor.validation_raw()
        for r in range(n_valid):
            assert_almost_equal(rescored[r], host_raw[r], atol=TOL)


def test_weighted_and_classification_metrics() raises:
    """Weights and the two log losses, against metrics.mojo term for term."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 800
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _binary_target(features, n_rows)
        var data = _binned(features, n_rows, n_features)
        var booster = train(data, target, BINARY_LOGISTIC, _small_params(12))
        var n_iter = booster.n_iterations()

        var weight = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            weight.append(0.25 + _uniform(UInt64(9000 + r)))

        var predictor = GpuPredictor(n_features, 1)
        predictor.upload_ensemble(flatten_booster(booster))
        predictor.set_validation(data, target, weight)
        predictor.score_validation(IterationRange.slice(n_iter, 0, n_iter))

        var host_prob = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            host_prob.append(booster.predict_bins(_bins_of(data, r)))

        var gpu_ll = predictor.validation_metric(
            METRIC_BINARY_LOG_LOSS, RESPONSE_SIGMOID
        )
        assert_almost_equal(
            gpu_ll, binary_log_loss(host_prob, target, weight), atol=1e-3
        )

        # The same reduction with the weights removed is the unweighted mean.
        var unweighted = GpuPredictor(n_features, 1)
        unweighted.upload_ensemble(flatten_booster(booster))
        unweighted.set_validation(data, target)
        unweighted.score_validation(IterationRange.slice(n_iter, 0, n_iter))
        var plain = unweighted.validation_metric(
            METRIC_BINARY_LOG_LOSS, RESPONSE_SIGMOID
        )
        assert_almost_equal(
            plain, binary_log_loss(host_prob, target), atol=1e-3
        )


def test_multiclass_validation_metric() raises:
    """Multiclass log loss over a resident softmax validation set."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 600
        var n_features = 4
        var n_classes = 3
        var features = _make_features(n_rows, n_features)
        var labels = List[Int](capacity=n_rows)
        var label_f64 = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var s = (
                3.0 * features[0 * n_rows + r] - 2.0 * features[1 * n_rows + r]
            )
            var k: Int
            if s < 0.0:
                k = 0
            elif s < 1.2:
                k = 1
            else:
                k = 2
            labels.append(k)
            label_f64.append(Float64(k))
        var data = _binned(features, n_rows, n_features)
        var booster = train_multiclass(
            data, labels, n_classes, _small_params(9)
        )
        var n_iter = booster.n_iterations()

        var predictor = GpuPredictor(n_features, n_classes)
        predictor.upload_ensemble(flatten_multiclass(booster))
        predictor.set_validation(data, label_f64)
        predictor.score_validation(IterationRange.slice(n_iter, 0, n_iter))
        var gpu_ll = predictor.validation_metric(
            METRIC_MULTICLASS_LOG_LOSS, RESPONSE_SOFTMAX
        )

        var host_probs = List[Float64](capacity=n_rows * n_classes)
        for r in range(n_rows):
            var p = booster.predict_proba_bins(_bins_of(data, r))
            for k in range(n_classes):
                host_probs.append(p[k])
        assert_almost_equal(
            gpu_ll,
            multiclass_log_loss(host_probs, labels, n_classes),
            atol=1e-3,
        )


def test_prediction_is_deterministic_and_buffers_are_reused() raises:
    """One thread sums one row's trees in iteration order, so repeated runs
    must agree bit for bit. The second half also covers batch-buffer reuse:
    a smaller batch after a larger one must not read past its own rows."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1000
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var data = _binned(features, n_rows, n_features)
        var booster = train(data, target, SQUARED_ERROR, _small_params(10))
        var n_iter = booster.n_iterations()
        var full = IterationRange.slice(n_iter, 0, n_iter)

        var predictor = GpuPredictor(n_features, 1)
        predictor.upload_ensemble(flatten_booster(booster))
        var first = predictor.raw_scores(data, full)
        var second = predictor.raw_scores(data, full)
        for r in range(n_rows):
            assert_equal(first[r], second[r])

        # A smaller matrix through the same predictor: the buffers are sized
        # for the larger batch, and the kernel must still index this batch's
        # own column stride.
        var small_rows = 137
        var small_features = _make_features(small_rows, n_features)
        var small = _binned(small_features, small_rows, n_features)
        var small_scores = predictor.raw_scores(small, full)
        assert_equal(len(small_scores), small_rows)
        for r in range(small_rows):
            assert_almost_equal(
                small_scores[r],
                booster.predict_raw_bins(_bins_of(small, r)),
                atol=TOL,
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
