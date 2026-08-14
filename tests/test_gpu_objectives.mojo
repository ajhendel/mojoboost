"""GPU training coverage beyond the squared-error/logistic equivalence pair.

`test_gpu_training.mojo` pins CPU/GPU agreement for the two headline
objectives. This file covers the rest of the contract:

  objectives    every objective that shares the per-row gradient/hessian
                interface, including the two that renew leaf values
                (quantile, L1) and softmax multiclass
  quality       the GPU model fits as well as the CPU model, measured as
                objective loss, not just pointwise prediction agreement
  determinism   repeat GPU training is bit-identical, per objective
  weights       sample weights, including zero-weight rows
  edges         single row, unsplittable data, num_leaves=1, the 256-bin
                boundary, and the explicit errors for the combinations the
                GPU backend does not support

Comparisons are mean-absolute rather than per-row worst case: Float32
histogram accumulation can flip a single near-tied split, which moves a
handful of rows without changing the model. Loss parity is asserted
separately and is the assertion that would catch a real divergence.

Skips (passing) when no accelerator is present so the suite stays green on
CPU-only machines.
"""

from std.math import exp, log
from std.os import remove
from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojoboost.binning import BinnedMatrix, bin_equal_width, fit_bins
from mojoboost.boosting import (
    BINARY_LOGISTIC,
    HUBER,
    L1,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    MulticlassBooster,
    train,
    train_multiclass,
)
from mojoboost.histogram_gpu import GpuHistogramBuilder
from mojoboost.model import Model
from mojoboost.serialize import load_model, save_model
from mojoboost.train_gpu import train_gpu, train_multiclass_gpu
from mojoboost.tree import TreeParams

comptime _TMP_PATH = "./.test_gpu_objectives_roundtrip.tmp"


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


def _signal(features: List[Float64], n_rows: Int, r: Int) -> Float64:
    """Strong, distinct per-feature effects so CPU and GPU split decisions
    do not sit on knife-edge gain ties."""
    var x0 = features[0 * n_rows + r]
    var x1 = features[1 * n_rows + r]
    var x2 = features[2 * n_rows + r]
    var x3 = features[3 * n_rows + r]
    return 4.0 * x0 - 3.0 * x1 + 2.0 * (x2 - 0.5) * (x3 - 0.5)


def _regression_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(_signal(features, n_rows, r))
    return y^


def _params(n_estimators: Int = 15, num_leaves: Int = 15) -> BoosterParams:
    return BoosterParams(
        n_estimators, 0.1, TreeParams(num_leaves, 20, 1.0, 1e-3)
    )


def _mean_abs_diff(
    a: Booster, b: Booster, data: BinnedMatrix
) raises -> Float64:
    var total = 0.0
    for r in range(data.n_rows):
        total += abs(a.predict_row(data, r) - b.predict_row(data, r))
    return total / Float64(data.n_rows)


def _loss(
    booster: Booster,
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    alpha: Float64,
) raises -> Float64:
    """The objective's own loss on the training rows, on the raw scale the
    objective is defined on."""
    var total = 0.0
    for r in range(data.n_rows):
        var raw = booster.predict_raw_row(data, r)
        var d = raw - target[r]
        if objective == POISSON:
            total += exp(raw) - target[r] * raw
        elif objective == HUBER:
            var ad = abs(d)
            if ad <= alpha:
                total += 0.5 * ad * ad
            else:
                total += alpha * (ad - 0.5 * alpha)
        elif objective == QUANTILE:
            total += (1.0 - alpha) * d if d >= 0.0 else -alpha * d
        elif objective == L1:
            total += abs(d)
        else:
            total += d * d
    return total / Float64(data.n_rows)


def _assert_matches_cpu(
    objective: Int,
    target: List[Float64],
    data: BinnedMatrix,
    alpha: Float64 = 0.9,
    tol: Float64 = 1e-3,
) raises:
    """CPU/GPU parity for one objective: same ensemble size, negligible mean
    prediction difference, and equal training loss to within `tol` relative.
    A model that merely predicts the base score would pass the first two, so
    the fit must also be materially better than no trees at all."""
    var params = _params()
    var cpu = train(data, target, objective, params, List[Float64](), alpha)
    var gpu = train_gpu(
        data, target, objective, params, List[Float64](), alpha
    )
    var flat = train(
        data, target, objective, _params(0), List[Float64](), alpha
    )

    assert_equal(len(cpu.trees), len(gpu.trees))
    assert_true(_mean_abs_diff(cpu, gpu, data) <= tol)

    var cpu_loss = _loss(cpu, data, target, objective, alpha)
    var gpu_loss = _loss(gpu, data, target, objective, alpha)
    var flat_loss = _loss(flat, data, target, objective, alpha)
    assert_true(abs(cpu_loss - gpu_loss) <= tol * (abs(cpu_loss) + 1e-12))
    # The trees actually did something, so loss parity is not parity between
    # two models that both learned nothing.
    assert_true(gpu_loss < 0.9 * flat_loss)


def _assert_deterministic(
    objective: Int,
    target: List[Float64],
    data: BinnedMatrix,
    alpha: Float64 = 0.9,
) raises:
    var params = _params(8)
    var a = train_gpu(data, target, objective, params, List[Float64](), alpha)
    var b = train_gpu(data, target, objective, params, List[Float64](), alpha)
    assert_equal(len(a.trees), len(b.trees))
    # Integer accumulation makes repeat GPU training bit-identical.
    for r in range(data.n_rows):
        assert_equal(
            a.predict_raw_row(data, r), b.predict_raw_row(data, r)
        )


# --------------------------------------------------------------------------
# Objective coverage
# --------------------------------------------------------------------------


def test_gpu_poisson_matches_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        # Nonnegative counts driven by the same signal.
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            var mu = exp(0.5 * _signal(features, n_rows, r))
            target.append(Float64(Int(mu)))

        _assert_matches_cpu(POISSON, target, data)
        _assert_deterministic(POISSON, target, data)


def test_gpu_huber_matches_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var target = _regression_target(features, n_rows)
        # Heavy tails are the point of huber: poison every 50th row.
        for r in range(0, n_rows, 50):
            target[r] += 40.0

        _assert_matches_cpu(HUBER, target, data)
        _assert_deterministic(HUBER, target, data)


def test_gpu_quantile_matches_cpu() raises:
    """Quantile renews every leaf value from the rows' residual percentile
    after the tree is grown, so this exercises the host-side leaf renewal
    path on top of device-grown tree structure."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var target = _regression_target(features, n_rows)

        _assert_matches_cpu(QUANTILE, target, data, 0.75)
        _assert_deterministic(QUANTILE, target, data, 0.75)


def test_gpu_l1_matches_cpu() raises:
    """L1 renews leaf values to the residual median (alpha = 0.5)."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var target = _regression_target(features, n_rows)

        _assert_matches_cpu(L1, target, data)
        _assert_deterministic(L1, target, data)


def test_gpu_multiclass_matches_cpu() raises:
    """Softmax is the last objective sharing the gradient/hessian interface:
    one builder serves every class of every round."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 6
        var n_classes = 3
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var labels = List[Int](capacity=n_rows)
        for r in range(n_rows):
            var s = _signal(features, n_rows, r)
            if s < -0.5:
                labels.append(0)
            elif s < 1.0:
                labels.append(1)
            else:
                labels.append(2)

        var params = _params(20)
        var cpu = train_multiclass(data, labels, n_classes, params)
        var gpu = train_multiclass_gpu(data, labels, n_classes, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        assert_equal(gpu.n_classes, n_classes)

        var total_diff = 0.0
        var cpu_correct = 0
        var gpu_correct = 0
        for r in range(n_rows):
            var bins = List[Int](capacity=n_features)
            for f in range(n_features):
                bins.append(data.bin_at(r, f))
            var pc = cpu.predict_proba_bins(bins)
            var pg = gpu.predict_proba_bins(bins)
            var cpu_arg = 0
            var gpu_arg = 0
            for k in range(n_classes):
                total_diff += abs(pc[k] - pg[k])
                if pc[k] > pc[cpu_arg]:
                    cpu_arg = k
                if pg[k] > pg[gpu_arg]:
                    gpu_arg = k
            if cpu_arg == labels[r]:
                cpu_correct += 1
            if gpu_arg == labels[r]:
                gpu_correct += 1

        assert_true(total_diff / Float64(n_rows * n_classes) <= 1e-3)
        # Same model quality, and a model that actually learned the classes.
        assert_true(abs(cpu_correct - gpu_correct) <= n_rows // 100)
        assert_true(gpu_correct * 10 >= n_rows * 7)


def test_gpu_multiclass_is_deterministic() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 1_500
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var labels = List[Int](capacity=n_rows)
        for r in range(n_rows):
            labels.append(r % 3)

        var params = _params(6)
        var a = train_multiclass_gpu(data, labels, 3, params)
        var b = train_multiclass_gpu(data, labels, 3, params)
        assert_equal(len(a.trees), len(b.trees))
        for r in range(0, n_rows, 7):
            var bins = List[Int](capacity=n_features)
            for f in range(n_features):
                bins.append(data.bin_at(r, f))
            var pa = a.predict_raw_bins(bins)
            var pb = b.predict_raw_bins(bins)
            for k in range(3):
                assert_equal(pa[k], pb[k])


# --------------------------------------------------------------------------
# Sample weights
# --------------------------------------------------------------------------


def test_gpu_sample_weights_match_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var target = _regression_target(features, n_rows)

        var weights = List[Float64](capacity=n_rows)
        var base = UInt64(n_rows * n_features)
        for r in range(n_rows):
            weights.append(0.1 + 2.0 * _uniform(base + UInt64(r)))

        var params = _params()
        var cpu = train(data, target, SQUARED_ERROR, params, weights)
        var gpu = train_gpu(data, target, SQUARED_ERROR, params, weights)
        var unweighted = train_gpu(data, target, SQUARED_ERROR, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        assert_true(_mean_abs_diff(cpu, gpu, data) <= 1e-3)
        # The weights must actually change the fit, or this proves nothing.
        assert_true(_mean_abs_diff(gpu, unweighted, data) > 1e-6)


def test_gpu_zero_weight_rows_are_ignored() raises:
    """A zero-weight row contributes zero gradient and zero hessian, so its
    target cannot move the fit no matter how extreme it is. Its bin still
    counts toward min_data_in_leaf, exactly as in LightGBM, which is why the
    comparison is against the CPU trainer on the same poisoned data rather
    than against a trainer run on the clean subset."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 3_000
        var n_features = 6
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 64)
        var clean = _regression_target(features, n_rows)

        var poisoned = clean.copy()
        var weights = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            if r % 3 == 0:
                poisoned[r] = 1.0e6
                weights.append(0.0)
            else:
                weights.append(1.0)

        var params = _params()
        var cpu = train(data, poisoned, SQUARED_ERROR, params, weights)
        var gpu = train_gpu(data, poisoned, SQUARED_ERROR, params, weights)

        assert_equal(len(cpu.trees), len(gpu.trees))
        assert_true(_mean_abs_diff(cpu, gpu, data) <= 1e-3)
        # The 1e6 rows are invisible: predictions stay on the clean scale.
        for r in range(n_rows):
            assert_true(abs(gpu.predict_row(data, r)) < 100.0)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


def test_gpu_single_row_dataset() raises:
    """One row can never be split, and squared error is already solved by the
    base score, so training converges immediately on both backends."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features: List[Float64] = [0.25, 0.75]
        var data = bin_equal_width(features, 1, 2, 8)
        var target: List[Float64] = [3.5]

        var cpu = train(data, target, SQUARED_ERROR, _params())
        var gpu = train_gpu(data, target, SQUARED_ERROR, _params())

        assert_equal(len(cpu.trees), len(gpu.trees))
        assert_equal(gpu.predict_row(data, 0), cpu.predict_row(data, 0))
        assert_true(abs(gpu.predict_row(data, 0) - 3.5) < 1e-12)


def test_gpu_unsplittable_data_matches_cpu() raises:
    """Every feature constant: no split clears min_child_hess, so both
    backends grow single-leaf trees (and squared error converges at once)."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 500
        var n_features = 4
        var features = List[Float64](capacity=n_rows * n_features)
        for _ in range(n_rows * n_features):
            features.append(1.0)
        var data = bin_equal_width(features, n_rows, n_features, 16)
        var target = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            target.append(_uniform(UInt64(r)))

        var cpu = train(data, target, SQUARED_ERROR, _params())
        var gpu = train_gpu(data, target, SQUARED_ERROR, _params())

        assert_equal(len(cpu.trees), len(gpu.trees))
        for r in range(n_rows):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-9
            )


def test_gpu_min_data_in_leaf_blocks_every_split() raises:
    """A min_data_in_leaf above half the rows leaves the root unsplittable, so
    every tree is a single leaf. Huber keeps the gradients from summing to
    zero, so training does not converge away before the case is exercised."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 400
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var target = _regression_target(features, n_rows)

        var params = BoosterParams(
            8, 0.1, TreeParams(15, n_rows, 1.0, 1e-3)
        )
        var cpu = train(data, target, HUBER, params)
        var gpu = train_gpu(data, target, HUBER, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        assert_true(len(gpu.trees) > 0)
        for i in range(len(gpu.trees)):
            assert_equal(gpu.trees[i].n_leaves, 1)
        for r in range(0, n_rows, 5):
            assert_true(
                abs(cpu.predict_row(data, r) - gpu.predict_row(data, r))
                <= 1e-6
            )


def test_gpu_num_leaves_one_matches_cpu() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 800
        var n_features = 5
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var target = _regression_target(features, n_rows)

        var params = BoosterParams(6, 0.1, TreeParams(1, 20, 1.0, 1e-3))
        var cpu = train(data, target, HUBER, params)
        var gpu = train_gpu(data, target, HUBER, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        for i in range(len(gpu.trees)):
            assert_equal(gpu.trees[i].n_leaves, 1)


def test_gpu_max_bin_boundary_matches_cpu() raises:
    """256 bins is the documented GPU limit and the shared-memory histogram
    width; it must work, not merely not crash."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 256)
        var target = _regression_target(features, n_rows)

        assert_equal(data.n_bins, 256)
        var params = _params(10)
        var cpu = train(data, target, SQUARED_ERROR, params)
        var gpu = train_gpu(data, target, SQUARED_ERROR, params)

        assert_equal(len(cpu.trees), len(gpu.trees))
        assert_true(_mean_abs_diff(cpu, gpu, data) <= 1e-3)


def test_gpu_trained_model_serializes() raises:
    """A GPU-trained booster is an ordinary `Booster`: the GPU path adds no
    model structure, so the existing format round-trips it bit-exactly."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 600
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var target = _regression_target(features, n_rows)
        var mapper = fit_bins(features, n_rows, n_features, 64)
        var data = mapper.transform(features, n_rows)
        var booster = train_gpu(data, target, SQUARED_ERROR, _params(10))
        var model = Model(mapper^, booster^)

        save_model(model, _TMP_PATH)
        var loaded = load_model(_TMP_PATH)
        remove(_TMP_PATH)

        assert_equal(len(loaded.booster.trees), len(model.booster.trees))
        for r in range(0, n_rows, 11):
            var row = List[Float64](capacity=n_features)
            for f in range(n_features):
                row.append(features[f * n_rows + r])
            assert_equal(loaded.predict(row), model.predict(row))


# --------------------------------------------------------------------------
# Explicit errors for unsupported combinations
# --------------------------------------------------------------------------


def test_gpu_trainer_rejects_bad_inputs() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 200
        # _signal reads features 0..3, so this is the minimum width.
        var n_features = 4
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 16)
        var target = _regression_target(features, n_rows)

        # Target length must match.
        var short: List[Float64] = [1.0, 2.0]
        with assert_raises():
            _ = train_gpu(data, short, SQUARED_ERROR, _params(1))

        # Unknown objective.
        with assert_raises():
            _ = train_gpu(data, target, 99, _params(1))

        # Poisson requires nonnegative targets.
        var negative = target.copy()
        negative[0] = -1.0
        with assert_raises():
            _ = train_gpu(data, negative, POISSON, _params(1))

        # Quantile requires 0 < alpha < 1.
        with assert_raises():
            _ = train_gpu(
                data, target, QUANTILE, _params(1), List[Float64](), 1.5
            )

        # Sample weights must match n_rows and be nonnegative.
        var bad_len: List[Float64] = [1.0, 1.0]
        with assert_raises():
            _ = train_gpu(data, target, SQUARED_ERROR, _params(1), bad_len)
        var negative_w = List[Float64](capacity=n_rows)
        for _ in range(n_rows):
            negative_w.append(1.0)
        negative_w[0] = -1.0
        with assert_raises():
            _ = train_gpu(data, target, SQUARED_ERROR, _params(1), negative_w)

        # Multiclass needs at least two classes and in-range labels.
        var labels = List[Int](capacity=n_rows)
        for r in range(n_rows):
            labels.append(r % 2)
        with assert_raises():
            _ = train_multiclass_gpu(data, labels, 1, _params(1))
        var out_of_range = labels.copy()
        out_of_range[0] = 5
        with assert_raises():
            _ = train_multiclass_gpu(data, out_of_range, 2, _params(1))


def test_gpu_builder_rejects_unsupported_datasets() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 100
        var n_features = 3
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 16)

        # More bins than the shared-memory histogram holds.
        var wide = BinnedMatrix(data.bins.copy(), n_rows, n_features, 257)
        with assert_raises():
            _ = GpuHistogramBuilder(wide)

        # Empty dataset.
        var empty = BinnedMatrix(List[UInt8](), 0, n_features, 16)
        with assert_raises():
            _ = GpuHistogramBuilder(empty)

        # Bin buffer inconsistent with the declared shape.
        var truncated = BinnedMatrix(
            data.bins.copy(), n_rows, n_features + 1, 16
        )
        with assert_raises():
            _ = GpuHistogramBuilder(truncated)


def test_gpu_builder_rejects_bad_calls() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 100
        var n_features = 3
        var features = _make_features(n_rows, n_features)
        var data = bin_equal_width(features, n_rows, n_features, 16)
        var builder = GpuHistogramBuilder(data)

        # Histograms need gradients first.
        with assert_raises():
            _ = builder.build_leaf(0)

        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(_uniform(UInt64(r)) - 0.5)
            hess.append(1.0)

        # Gradient vectors must be one entry per row.
        var short: List[Float64] = [1.0]
        with assert_raises():
            builder.upload_gradients(short, hess)

        # Non-finite gradients would quantize to garbage rather than fail
        # loudly, so the fixed-point scale rejects them up front.
        var infinite = grad.copy()
        infinite[0] = 1.0 / 0.0
        with assert_raises():
            builder.upload_gradients(infinite, hess)
        var not_a_number = grad.copy()
        not_a_number[0] = (1.0 / 0.0) - (1.0 / 0.0)
        with assert_raises():
            builder.upload_gradients(grad, not_a_number)

        builder.upload_gradients(grad, hess)
        builder.begin_tree()

        # Splits must name a real feature, a real bin, and distinct children.
        with assert_raises():
            builder.apply_split(n_features, 0, 0, 1, 2)
        with assert_raises():
            builder.apply_split(0, 16, 0, 1, 2)
        with assert_raises():
            builder.apply_split(0, 0, 0, 0, 1)
        with assert_raises():
            builder.apply_split(0, 0, 0, 1, 1)
        with assert_raises():
            builder.apply_split(0, 0, -1, 1, 2)
        with assert_raises():
            _ = builder.build_leaf(-1)

        # And the valid call still works afterwards.
        builder.apply_split(0, 7, 0, 1, 2)
        var left = builder.build_leaf(1)
        assert_equal(left.n_features, n_features)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
