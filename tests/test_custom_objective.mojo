"""Tests for caller-supplied objectives (objective.mojo).

The anchor property is exact reproduction: a custom objective that computes
the same derivatives as a built-in one, started from the same base score,
must produce bit-identical predictions, weighted or not. The rest covers the
documented contract: input-shape and hessian validation, sample-weight
semantics, the single-output-only multiclass policy, early-stopping
compatibility, and serialization of the CUSTOM objective code.
"""

from std.math import inf, nan
from std.os import remove
from std.sys import has_accelerator
from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojoboost.binning import BinnedMatrix, bin_equal_width
from mojoboost.boosting import (
    CUSTOM,
    HUBER,
    SQUARED_ERROR,
    BoosterParams,
    train,
    train_with_valid,
)
from mojoboost.model import fit, fit_custom
from mojoboost.objective import (
    check_custom_grad_hess,
    mean_label,
    squared_error_grad_hess,
    squared_error_loss,
    train_custom,
    train_custom_with_valid,
)
from mojoboost.serialize import load_model, save_model
from mojoboost.train_gpu import train_custom_gpu, train_gpu
from mojoboost.tree import TreeParams

comptime _TMP_PATH = "./.test_custom_objective_roundtrip.tmp"


def _params(n_rounds: Int) -> BoosterParams:
    return BoosterParams(n_rounds, 0.3, TreeParams(4, 1, 1.0, 1e-3))


def _features() -> List[Float64]:
    return [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def _target() -> List[Float64]:
    return [0.5, 0.5, 1.5, 1.5, 4.0, 4.0, 9.0, 9.0]


def _binned() raises -> BinnedMatrix:
    var features = _features()
    return bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)


def test_custom_squared_error_matches_builtin_exactly() raises:
    # The reference example: gradient raw - target, hessian 1, started from
    # the label mean, is the built-in SQUARED_ERROR objective. Every
    # prediction must agree bit for bit.
    var data = _binned()
    var target = _target()
    var params = _params(25)

    var builtin = train(data, target, SQUARED_ERROR, params)
    var custom = train_custom(
        data,
        target,
        squared_error_grad_hess,
        params,
        base_score=mean_label(target, []),
    )
    assert_equal(len(custom.trees), len(builtin.trees))
    for r in range(8):
        assert_equal(custom.predict_row(data, r), builtin.predict_row(data, r))


def test_custom_squared_error_matches_builtin_with_weights() raises:
    # Weights are applied by the trainer after the callback returns, in the
    # same operand order as the built-in objectives, so the weighted models
    # are bit-identical too. The zero-weight row must be ignored by both.
    var data = _binned()
    var target = _target()
    var weights: List[Float64] = [1.0, 2.0, 0.5, 3.0, 1.0, 1.0, 0.0, 4.0]
    var params = _params(25)

    var builtin = train(data, target, SQUARED_ERROR, params, weights)
    var custom = train_custom(
        data,
        target,
        squared_error_grad_hess,
        params,
        weights,
        base_score=mean_label(target, weights),
    )
    for r in range(8):
        assert_equal(custom.predict_row(data, r), builtin.predict_row(data, r))


def test_zero_weight_row_does_not_influence_training() raises:
    # A zero-weight row contributes zero gradient and zero hessian, so its
    # label is irrelevant: moving it must not change any prediction.
    var data = _binned()
    var target = _target()
    var poisoned = _target()
    poisoned[6] = -1000.0
    var weights: List[Float64] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    var params = _params(20)

    # Same base score for both so only the gradients can differ.
    var base = mean_label(target, weights)
    var clean = train_custom(
        data, target, squared_error_grad_hess, params, weights, base_score=base
    )
    var dirty = train_custom(
        data,
        poisoned,
        squared_error_grad_hess,
        params,
        weights,
        base_score=base,
    )
    for r in range(8):
        assert_equal(clean.predict_row(data, r), dirty.predict_row(data, r))


def test_closure_captures_hyperparameter_and_matches_huber() raises:
    # A closure capturing alpha is a valid objective: this one is the huber
    # gradient, and with the label mean as the base score it reproduces the
    # built-in HUBER objective exactly.
    var data = _binned()
    var target = _target()
    var params = _params(30)
    var alpha = 1.0

    def huber_grad_hess(
        raw: List[Float64],
        target: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises {imm alpha}:
        grad.clear()
        hess.clear()
        for r in range(len(target)):
            var diff = raw[r] - target[r]
            if abs(diff) <= alpha:
                grad.append(diff)
            elif diff > 0.0:
                grad.append(alpha)
            else:
                grad.append(-alpha)
            hess.append(1.0)

    var builtin = train(data, target, HUBER, params, alpha=alpha)
    var custom = train_custom(
        data,
        target,
        huber_grad_hess,
        params,
        base_score=mean_label(target, []),
    )
    for r in range(8):
        assert_equal(custom.predict_row(data, r), builtin.predict_row(data, r))


def test_rejects_wrong_gradient_length() raises:
    var data = _binned()
    var target = _target()

    def short_grad(
        raw: List[Float64],
        target: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises:
        grad.clear()
        hess.clear()
        for _ in range(len(target) - 1):
            grad.append(0.1)
        for _ in range(len(target)):
            hess.append(1.0)

    with assert_raises(contains="7 gradients for 8 rows"):
        _ = train_custom(data, target, short_grad, _params(5))


def test_rejects_wrong_hessian_length() raises:
    var data = _binned()
    var target = _target()

    def long_hess(
        raw: List[Float64],
        target: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises:
        grad.clear()
        hess.clear()
        for _ in range(len(target)):
            grad.append(0.1)
        for _ in range(len(target) + 2):
            hess.append(1.0)

    with assert_raises(contains="10 hessians for 8 rows"):
        _ = train_custom(data, target, long_hess, _params(5))


def test_rejects_negative_hessian() raises:
    var data = _binned()
    var target = _target()

    def negative_hess(
        raw: List[Float64],
        target: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises:
        grad.clear()
        hess.clear()
        for r in range(len(target)):
            grad.append(raw[r] - target[r])
            hess.append(-1.0 if r == 3 else 1.0)

    with assert_raises(contains="negative hessian at row 3"):
        _ = train_custom(data, target, negative_hess, _params(5))


def test_rejects_non_finite_output() raises:
    var data = _binned()
    var target = _target()

    def nan_grad(
        raw: List[Float64],
        target: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises:
        grad.clear()
        hess.clear()
        for r in range(len(target)):
            grad.append(nan[DType.float64]() if r == 2 else 0.1)
            hess.append(1.0)

    with assert_raises(contains="non-finite gradient at row 2"):
        _ = train_custom(data, target, nan_grad, _params(5))

    def inf_hess(
        raw: List[Float64],
        target: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises:
        grad.clear()
        hess.clear()
        for r in range(len(target)):
            grad.append(0.1)
            hess.append(inf[DType.float64]() if r == 5 else 1.0)

    with assert_raises(contains="non-finite hessian at row 5"):
        _ = train_custom(data, target, inf_hess, _params(5))


def test_zero_hessian_is_allowed() raises:
    # Zero curvature is what a zero-weight row produces, so it must be
    # accepted; lambda_l2 keeps the leaf value finite.
    var grad: List[Float64] = [1.0, -1.0, 0.0]
    var hess: List[Float64] = [0.0, 0.0, 0.0]
    check_custom_grad_hess(grad, hess, 3)

    var data = _binned()
    var target = _target()

    def zero_hess(
        raw: List[Float64],
        target: List[Float64],
        mut grad: List[Float64],
        mut hess: List[Float64],
    ) raises:
        grad.clear()
        hess.clear()
        for r in range(len(target)):
            grad.append(raw[r] - target[r])
            hess.append(0.0)

    var model = train_custom(data, target, zero_hess, _params(5))
    for r in range(8):
        assert_true(model.predict_row(data, r) == model.predict_row(data, r))


def test_target_length_validation() raises:
    var data = _binned()
    var short_target: List[Float64] = [1.0, 2.0, 3.0]

    with assert_raises(contains="target length must equal n_rows"):
        _ = train_custom(
            data, short_target, squared_error_grad_hess, _params(5)
        )


def test_sample_weight_validation() raises:
    var data = _binned()
    var target = _target()
    var short_weights: List[Float64] = [1.0, 1.0]

    with assert_raises(contains="sample_weight length must equal n_rows"):
        _ = train_custom(
            data, target, squared_error_grad_hess, _params(5), short_weights
        )

    var negative: List[Float64] = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
    ]
    with assert_raises(contains="nonnegative"):
        _ = train_custom(
            data, target, squared_error_grad_hess, _params(5), negative
        )


def test_custom_objective_code_is_single_output_only() raises:
    # Policy: CUSTOM is not a built-in objective code. It cannot be smuggled
    # into `train`, `train_gpu`, or `fit`, and there is no multiclass entry
    # point at all (LightGBM's (n_rows * n_classes) custom gradient matrix is
    # deliberately not supported).
    var data = _binned()
    var target = _target()
    var features = _features()

    with assert_raises(contains="must be trained with train_custom"):
        _ = train(data, target, CUSTOM, _params(5))

    with assert_raises(contains="must be trained with train_custom"):
        _ = fit(features, 8, 1, target, CUSTOM, _params(5))

    comptime if has_accelerator():
        with assert_raises(contains="must be trained with train_custom"):
            _ = train_gpu(data, target, CUSTOM, _params(5))


def test_early_stopping_matches_builtin() raises:
    # train_custom_with_valid drives the same early-stopping loop as
    # train_with_valid: with the squared-error gradient and the matching
    # MSE validation loss, both must stop at the same round and agree
    # bit for bit.
    var features = _features()
    var target = _target()
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var valid_features: List[Float64] = [0.5, 2.5, 4.5, 6.5]
    var valid_target: List[Float64] = [0.5, 1.5, 4.0, 6.0]
    var valid_data = bin_equal_width(
        valid_features, n_rows=4, n_features=1, n_bins=8
    )
    var params = _params(60)

    var builtin = train_with_valid(
        data,
        target,
        valid_data,
        valid_target,
        SQUARED_ERROR,
        params,
        early_stopping_rounds=3,
    )
    var custom = train_custom_with_valid(
        data,
        target,
        valid_data,
        valid_target,
        squared_error_grad_hess,
        squared_error_loss,
        params,
        early_stopping_rounds=3,
        base_score=mean_label(target, []),
    )
    assert_equal(len(custom.trees), len(builtin.trees))
    assert_true(len(custom.trees) < 60)
    for r in range(8):
        assert_equal(custom.predict_row(data, r), builtin.predict_row(data, r))


def test_early_stopping_min_delta_stops_sooner() raises:
    var features = _features()
    var target = _target()
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var valid_features: List[Float64] = [0.5, 2.5, 4.5, 6.5]
    var valid_target: List[Float64] = [0.5, 1.5, 4.0, 6.0]
    var valid_data = bin_equal_width(
        valid_features, n_rows=4, n_features=1, n_bins=8
    )
    var params = _params(60)
    var base = mean_label(target, [])

    var strict = train_custom_with_valid(
        data,
        target,
        valid_data,
        valid_target,
        squared_error_grad_hess,
        squared_error_loss,
        params,
        early_stopping_rounds=3,
        min_delta=1.0,
        base_score=base,
    )
    var loose = train_custom_with_valid(
        data,
        target,
        valid_data,
        valid_target,
        squared_error_grad_hess,
        squared_error_loss,
        params,
        early_stopping_rounds=3,
        min_delta=0.0,
        base_score=base,
    )
    assert_true(len(strict.trees) < len(loose.trees))


def test_early_stopping_validates_shapes() raises:
    var features = _features()
    var target = _target()
    var data = bin_equal_width(features, n_rows=8, n_features=1, n_bins=8)
    var valid_features: List[Float64] = [0.5, 2.5, 4.5, 6.5]
    var valid_data = bin_equal_width(
        valid_features, n_rows=4, n_features=1, n_bins=8
    )
    var short_valid_target: List[Float64] = [0.5, 1.5]
    var params = _params(10)

    with assert_raises(contains="valid_target length"):
        _ = train_custom_with_valid(
            data,
            target,
            valid_data,
            short_valid_target,
            squared_error_grad_hess,
            squared_error_loss,
            params,
            early_stopping_rounds=3,
        )

    var valid_target: List[Float64] = [0.5, 1.5, 4.0, 6.0]
    with assert_raises(contains="early_stopping_rounds must be positive"):
        _ = train_custom_with_valid(
            data,
            target,
            valid_data,
            valid_target,
            squared_error_grad_hess,
            squared_error_loss,
            params,
            early_stopping_rounds=0,
        )


def test_fit_custom_roundtrips_through_serialization() raises:
    # A custom-objective model carries the CUSTOM code, predicts raw scores
    # (no known inverse link), and survives save/load bit-exactly.
    var features = _features()
    var target = _target()
    var model = fit_custom(
        features,
        8,
        1,
        target,
        squared_error_grad_hess,
        _params(25),
        base_score=mean_label(target, []),
    )
    assert_equal(model.booster.objective, CUSTOM)

    var row: List[Float64] = [3.0]
    assert_equal(model.predict(row), model.predict_raw(row))

    save_model(model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)
    assert_equal(loaded.booster.objective, CUSTOM)
    for r in range(8):
        var one: List[Float64] = [features[r]]
        assert_equal(loaded.predict(one), model.predict(one))


def test_gpu_custom_matches_cpu_custom() raises:
    # The objective callback runs on the host in both trainers; only the
    # histogram accumulation differs, so agreement is Float32-tolerance
    # (see train_gpu.mojo), not bit-exact. Passes trivially with no
    # accelerator present.
    comptime if not has_accelerator():
        return
    else:
        var features = List[Float64](capacity=512)
        var state: UInt64 = 7
        for _ in range(512):
            state = state * 6364136223846793005 + 1442695040888963407
            features.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
        var target = List[Float64](capacity=512)
        for r in range(512):
            target.append(3.0 * features[r] + 0.5)
        var data = bin_equal_width(
            features, n_rows=512, n_features=1, n_bins=32
        )
        var params = BoosterParams(15, 0.1, TreeParams(8, 5, 1.0, 1e-3))
        var base = mean_label(target, [])

        var cpu = train_custom(
            data, target, squared_error_grad_hess, params, base_score=base
        )
        var gpu = train_custom_gpu(
            data, target, squared_error_grad_hess, params, base_score=base
        )
        assert_equal(len(gpu.trees), len(cpu.trees))
        for r in range(512):
            assert_true(
                abs(gpu.predict_row(data, r) - cpu.predict_row(data, r)) < 1e-4
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
