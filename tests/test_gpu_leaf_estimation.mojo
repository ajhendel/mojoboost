"""CatBoost's `leaf_estimation_iterations` on the GPU trainer.

`tests/test_leaf_estimation.mojo` owns the host implementation,
`boosting._estimate_leaf_values`, and proves the three things that
implementation has to prove. This file owns the device one,
`gpu_objectives_native.GpuLeafEstimator`, and proves the two things a *second*
implementation of an existing mechanism has to prove instead.

1. **At the default it is not there.** A `train_gpu` fit with the parameter
   absent and one with it explicitly 1 are the same ensemble bit for bit, on
   the device-objective arm and on the host-objective arm, compared with
   `to_bits()`. The implementation earns that in two places rather than one:
   `GpuLeafEstimator` is never *constructed* when
   `leaf_estimation_active()` is false, so the default fit allocates no
   device buffer and enqueues no launch, and `estimate` returns before it
   stages a word when `iterations <= 1`, so a caller that constructs one
   anyway still cannot move a value. Iteration 1 is never recomputed on
   either backend: the value the grower wrote from the histogram's own sums
   is the value the tree keeps.

2. **Above 1 it is the host implementation's answer, not a second answer.**
   The gate here is not "the device changed something". One tree is grown
   once, and then the *same* tree, the same rows, the same raw scores and the
   same regularization are handed to the device estimator and to
   `boosting._estimate_leaf_values`, and every leaf is compared. The device
   carries Float32 and folds a leaf's rows in a strided threadgroup
   reduction where the host carries Float64 and folds them sequentially in
   ascending row order, so the comparison is to a Float32 tolerance and not
   to the bit -- that is the trade this whole plane already makes and the
   module docstring of gpu_objectives_native.mojo states. Everything else
   about the two is required to agree.

   The fixture makes the leaf values move at **every** iteration, not only at
   the second. `_labels_skewed` gives each leaf roughly a 7:1 class mix, so a
   Newton step from `p = 0.5` lands short of the leaf's optimum and the next
   two steps close visible fractions of the remaining gap:
   `test_every_extra_iteration_moves_a_device_leaf` asserts a floor on both
   the second step and the *third*. A fixture whose iteration 3 changed
   nothing would let a `k = 2`-and-stop implementation pass, which is the
   failure this repository keeps producing.

Everything the device trainer cannot do it refuses by name rather than
ignoring: the multiclass GPU trainers (a softmax row's derivative for class
`k` reads every class's raw score, including trees this round has not grown),
`train_custom_gpu` (whose callback has no per-leaf call shape), the renewing
objectives and GOSS.

Skips (passing) when no accelerator is present, so the suite stays green on
CPU-only machines.
"""

from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    Booster,
    BoosterParams,
    L1,
    SQUARED_ERROR,
    _estimate_leaf_values,
    fill_grad_hess,
)
from mojotrees.goss import GossParams
from mojotrees.gpu_objectives_native import GpuLeafEstimator
from mojotrees.histogram_gpu import GpuHistogramBuilder
from mojotrees.monotone import OutputBounds
from mojotrees.objective import squared_error_grad_hess
from mojotrees.train_gpu import (
    OBJECTIVE_SOURCE_DEVICE,
    OBJECTIVE_SOURCE_HOST,
    grow_tree_gpu,
    train_custom_gpu,
    train_gpu,
    train_multiclass_gpu,
)
from mojotrees.tree import Tree, TreeParams
from mojotrees.tree_parameters_extra import ExtraTreeParams

from support import _uniform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

comptime N_ROWS = 4_000
comptime N_FEATURES = 4
comptime N_BINS = 32


def _bits(v: Float64) -> UInt64:
    return UInt64(v.to_bits())


def _features() -> List[Float64]:
    var out = List[Float64](capacity=N_ROWS * N_FEATURES)
    for k in range(N_ROWS * N_FEATURES):
        out.append(_uniform(UInt64(k) + 17))
    return out^


def _matrix(features: List[Float64]) raises -> BinnedMatrix:
    return bin_equal_width(features, N_ROWS, N_FEATURES, N_BINS)


def _labels_skewed(features: List[Float64]) -> List[Float64]:
    """A {0, 1} label that a tree can find but cannot make pure.

    The class is decided by the first feature and then flipped on every
    eighth row, so a leaf that isolates one side still holds roughly a 7:1
    mix. That ratio is the whole point of the fixture. A pure leaf's optimum
    is at infinity and a Newton step from `p = 0.5` walks toward it in steps
    that never converge, while a 50:50 leaf is *already* optimal at zero and
    the second step moves nothing; at 7:1 the leaf has a finite optimum some
    way off, so the first step falls short of it, the second closes most of
    the gap and the third still closes a visible piece. Both floors in
    `test_every_extra_iteration_moves_a_device_leaf` come from that.
    """
    var out = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        var positive = features[r] > 0.5
        if r % 8 == 0:
            positive = not positive
        out.append(1.0 if positive else 0.0)
    return out^


def _regression_target(features: List[Float64]) -> List[Float64]:
    var out = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        out.append(
            2.0 * features[r]
            - features[N_ROWS + r]
            + 0.5 * features[2 * N_ROWS + r]
        )
    return out^


def _extra(iterations: Int) -> ExtraTreeParams:
    var e = ExtraTreeParams()
    e.leaf_estimation_iterations = iterations
    return e^


def _params(n_estimators: Int, iterations: Int) raises -> BoosterParams:
    return BoosterParams(
        n_estimators,
        0.1,
        TreeParams(8, 20, 1.0, 1e-3, 0.0, extra=_extra(iterations)),
    )


def _default_params(n_estimators: Int) raises -> BoosterParams:
    """The same parameters with `extra` left entirely alone, so the fit takes
    whatever `ExtraTreeParams()` defaults to rather than an explicit 1."""
    return BoosterParams(n_estimators, 0.1, TreeParams(8, 20, 1.0, 1e-3, 0.0))


def _ensemble_bits(booster: Booster) -> List[UInt64]:
    """Every leaf value and every routing decision in the ensemble, as
    integers, so a leaf that moved and a split that moved are both visible
    and neither can be hidden by the other cancelling it out at a row."""
    var out = List[UInt64]()
    out.append(UInt64(len(booster.trees)))
    out.append(_bits(booster.base_score))
    for t in range(len(booster.trees)):
        ref tree = booster.trees[t]
        out.append(UInt64(len(tree.feature)))
        for node in range(len(tree.feature)):
            out.append(UInt64(tree.feature[node]))
            out.append(UInt64(tree.threshold_bin[node]))
            out.append(_bits(tree.value[node]))
    return out^


def _assert_same_bits(
    a: List[UInt64], b: List[UInt64], what: String
) raises:
    assert_equal(len(a), len(b), what)
    for i in range(len(a)):
        assert_equal(a[i], b[i], what)


def _is_leaf(tree: Tree) -> List[Bool]:
    var out = List[Bool](capacity=len(tree.feature))
    for node in range(len(tree.feature)):
        out.append(tree.feature[node] < 0)
    return out^


# ---------------------------------------------------------------------------
# 1. The default does not move
# ---------------------------------------------------------------------------


def test_one_iteration_is_absent_bit_for_bit_on_both_arms() raises:
    """A fit with the parameter absent and a fit with it explicitly 1 are the
    same ensemble, bit for bit, on the device-objective arm and on the
    host-objective arm.

    This is the guarantee the whole feature is built around: 1 is LightGBM's
    behavior and this project's, and the default path may not move at all.
    Both arms are checked because they take different code: the device arm
    would construct a `GpuLeafEstimator` and the host arm would call
    `boosting._estimate_leaf_values`, and each has its own early return.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features()
        var data = _matrix(features)
        var target = _regression_target(features)

        var absent_dev = train_gpu(
            data, target, SQUARED_ERROR, _default_params(4),
            objective_source=OBJECTIVE_SOURCE_DEVICE,
        )
        var one_dev = train_gpu(
            data, target, SQUARED_ERROR, _params(4, 1),
            objective_source=OBJECTIVE_SOURCE_DEVICE,
        )
        _assert_same_bits(
            _ensemble_bits(absent_dev),
            _ensemble_bits(one_dev),
            String("device arm: absent and 1 must be one ensemble"),
        )

        var absent_host = train_gpu(
            data, target, SQUARED_ERROR, _default_params(4),
            objective_source=OBJECTIVE_SOURCE_HOST,
        )
        var one_host = train_gpu(
            data, target, SQUARED_ERROR, _params(4, 1),
            objective_source=OBJECTIVE_SOURCE_HOST,
        )
        _assert_same_bits(
            _ensemble_bits(absent_host),
            _ensemble_bits(one_host),
            String("host arm: absent and 1 must be one ensemble"),
        )


def test_estimate_at_one_iteration_leaves_every_value_untouched() raises:
    """`GpuLeafEstimator.estimate` at `iterations = 1` returns the values it
    was given, bit for bit, even when it is called.

    The trainer never constructs an estimator at the default, so this is the
    second lock rather than the first: a future caller that constructs one
    unconditionally still cannot move a leaf. Compared with `to_bits()`,
    because "unchanged" here is not a tolerance.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features()
        var data = _matrix(features)
        var target = _labels_skewed(features)
        var builder = GpuHistogramBuilder(data)
        var params = TreeParams(8, 20, 1.0, 1e-3, 0.0)

        var raw = List[Float64](capacity=N_ROWS)
        for _ in range(N_ROWS):
            raw.append(0.0)
        var grad = List[Float64]()
        var hess = List[Float64]()
        fill_grad_hess(
            raw, target, BINARY_LOGISTIC, [], 0.0, grad, hess
        )
        builder.upload_gradients(grad, hess)
        var tree = grow_tree_gpu(builder, params)

        var state = builder.objective_state(target)
        state.init_raw(builder.ctx, [0.0])
        var estimator = GpuLeafEstimator(builder.ctx, N_ROWS, 2 * 8)
        var values = tree.value.copy()
        estimator.estimate(
            builder.ctx, state, builder.rows, values, _is_leaf(tree),
            List[OutputBounds](), BINARY_LOGISTIC, 0.0, 1,
            0.0, params.lambda_reg, 0.0,
        )
        assert_equal(len(values), len(tree.value))
        for node in range(len(values)):
            assert_equal(_bits(values[node]), _bits(tree.value[node]))


# ---------------------------------------------------------------------------
# 2. Above 1 it is the host implementation's answer
# ---------------------------------------------------------------------------


def test_device_matches_the_host_reference_at_three_iterations() raises:
    """The device estimator and `boosting._estimate_leaf_values`, on one
    tree, agree leaf by leaf.

    One tree is grown once and both implementations are handed *that* tree,
    so nothing about tree structure, row membership or the starting leaf
    values can differ between them; the only thing under test is the
    iteration. The raw scores are flat at zero on both sides, which is a
    point both carriers represent exactly, so the comparison measures the
    iteration's arithmetic rather than an input rounding.

    The tolerance is relative with an absolute floor, sized for Float32: the
    device sums a leaf's gradients and hessians in Float32 in a strided
    threadgroup reduction, the host sums them in Float64 sequentially in
    ascending row order. Anything larger than that -- a missed iteration, a
    shift by `learning_rate * v` instead of `v`, a leaf whose rows the device
    mis-identified -- moves a value by a fraction of itself, not by a part in
    ten thousand.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features()
        var data = _matrix(features)
        var target = _labels_skewed(features)
        var builder = GpuHistogramBuilder(data)
        var params = TreeParams(8, 20, 1.0, 1e-3, 0.0)

        var raw = List[Float64](capacity=N_ROWS)
        for _ in range(N_ROWS):
            raw.append(0.0)
        var grad = List[Float64]()
        var hess = List[Float64]()
        fill_grad_hess(
            raw, target, BINARY_LOGISTIC, [], 0.0, grad, hess
        )
        builder.upload_gradients(grad, hess)
        var tree = grow_tree_gpu(builder, params)
        # A tree with something to say. A single-leaf tree would make the
        # comparison below true for reasons that have nothing to do with the
        # iteration.
        assert_true(tree.n_leaves > 1, String("fixture must split"))

        var state = builder.objective_state(target)
        state.init_raw(builder.ctx, [0.0])
        var estimator = GpuLeafEstimator(builder.ctx, N_ROWS, 2 * 8)
        var device_values = tree.value.copy()
        estimator.estimate(
            builder.ctx, state, builder.rows, device_values, _is_leaf(tree),
            List[OutputBounds](), BINARY_LOGISTIC, 0.0, 3,
            0.0, params.lambda_reg, 0.0,
        )

        var host_tree = tree.copy()
        _estimate_leaf_values(
            host_tree, data, target, raw, BINARY_LOGISTIC, [], 0.0, 3,
            0.0, params.lambda_reg, 0.0,
        )

        var compared = 0
        for node in range(len(device_values)):
            if tree.feature[node] >= 0:
                continue
            var got = device_values[node]
            var want = host_tree.value[node]
            var tol = 1e-3 * abs(want) + 1e-5
            assert_true(
                abs(got - want) <= tol,
                String("leaf ", node, ": device ", got, " host ", want),
            )
            compared += 1
        # Without this the loop above would pass on a tree it never entered.
        assert_equal(compared, tree.n_leaves)


def test_every_extra_iteration_moves_a_device_leaf() raises:
    """Iterations 2 and 3 both move a leaf, by margins far above the Float32
    noise.

    A test that only asserted "k = 3 differs from k = 1" would pass an
    implementation that took one extra step and stopped, and a fixture whose
    third iteration changed nothing would pass it too. So the count is walked
    from 1 to 4 over the same tree and the same starting scores, and the
    *second* and *third* steps each get their own floor. The fourth is read
    but not floored: by then the leaf is close enough to its optimum that the
    step is genuinely small, which is what convergence looks like and is not
    something to assert a size for.
    """
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features()
        var data = _matrix(features)
        var target = _labels_skewed(features)
        var builder = GpuHistogramBuilder(data)
        var params = TreeParams(8, 20, 1.0, 1e-3, 0.0)

        var raw = List[Float64](capacity=N_ROWS)
        for _ in range(N_ROWS):
            raw.append(0.0)
        var grad = List[Float64]()
        var hess = List[Float64]()
        fill_grad_hess(
            raw, target, BINARY_LOGISTIC, [], 0.0, grad, hess
        )
        builder.upload_gradients(grad, hess)
        var tree = grow_tree_gpu(builder, params)
        assert_true(tree.n_leaves > 1, String("fixture must split"))

        var estimator = GpuLeafEstimator(builder.ctx, N_ROWS, 2 * 8)
        var leaves = _is_leaf(tree)
        var by_k = List[List[Float64]]()
        for k in range(1, 5):
            # A fresh state per count, so every run starts from the same raw
            # scores; the estimator never writes them, but taking a new one
            # makes that independent of whether it ever does.
            var state = builder.objective_state(target)
            state.init_raw(builder.ctx, [0.0])
            var values = tree.value.copy()
            estimator.estimate(
                builder.ctx, state, builder.rows, values, leaves,
                List[OutputBounds](), BINARY_LOGISTIC, 0.0, k,
                0.0, params.lambda_reg, 0.0,
            )
            by_k.append(values^)

        # k = 1 is the grower's own answer, untouched.
        for node in range(len(tree.value)):
            assert_equal(_bits(by_k[0][node]), _bits(tree.value[node]))

        var move_2 = 0.0
        var move_3 = 0.0
        var move_4 = 0.0
        for node in range(len(tree.value)):
            if tree.feature[node] >= 0:
                continue
            move_2 = max(move_2, abs(by_k[1][node] - by_k[0][node]))
            move_3 = max(move_3, abs(by_k[2][node] - by_k[1][node]))
            move_4 = max(move_4, abs(by_k[3][node] - by_k[2][node]))
        assert_true(
            move_2 > 0.05,
            String("iteration 2 must move a leaf, moved ", move_2),
        )
        assert_true(
            move_3 > 0.005,
            String("iteration 3 must move a leaf too, moved ", move_3),
        )
        # Newton on a convex one-dimensional problem: each step closes most
        # of the gap the last one left, so the moves shrink. A run where they
        # grew would mean the iteration is diverging rather than converging.
        assert_true(
            move_3 < move_2 and move_4 < move_3,
            String("steps must shrink: ", move_2, " ", move_3, " ", move_4),
        )


def test_the_device_round_honors_the_parameter() raises:
    """A whole `train_gpu` fit on the device-objective arm changes when the
    count changes, which is what says the estimator is wired into the round
    and not merely unit-tested beside it."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features()
        var data = _matrix(features)
        var target = _labels_skewed(features)

        var one = train_gpu(
            data, target, BINARY_LOGISTIC, _params(3, 1),
            objective_source=OBJECTIVE_SOURCE_DEVICE,
        )
        var three = train_gpu(
            data, target, BINARY_LOGISTIC, _params(3, 3),
            objective_source=OBJECTIVE_SOURCE_DEVICE,
        )
        assert_equal(len(one.trees), len(three.trees))
        assert_true(len(one.trees) > 0, String("fixture must grow trees"))
        var moved = 0
        for t in range(len(one.trees)):
            ref a = one.trees[t]
            ref b = three.trees[t]
            assert_equal(len(a.value), len(b.value))
            for node in range(len(a.value)):
                if a.feature[node] >= 0:
                    continue
                if _bits(a.value[node]) != _bits(b.value[node]):
                    moved += 1
        assert_true(moved > 0, String("k = 3 must move leaves in the fit"))


# ---------------------------------------------------------------------------
# 3. What the GPU trainers refuse rather than ignore
# ---------------------------------------------------------------------------


def test_multiclass_gpu_refuses_extra_iterations() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features()
        var data = _matrix(features)
        var labels = List[Int](capacity=N_ROWS)
        for r in range(N_ROWS):
            labels.append(r % 3)
        with assert_raises(contains="not implemented by"):
            _ = train_multiclass_gpu(data, labels, 3, _params(2, 2))


def test_custom_gpu_refuses_extra_iterations() raises:
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features()
        var data = _matrix(features)
        var target = _regression_target(features)
        with assert_raises(contains="not implemented by"):
            _ = train_custom_gpu(
                data, target, squared_error_grad_hess, _params(2, 2),
                base_score=0.0,
            )


def test_train_gpu_refuses_a_renewing_objective_and_goss() raises:
    """The two configurations `_check_leaf_estimation_config` refuses, reached
    through the GPU trainer so that the check is proved to be on its path and
    not only on the CPU trainer's."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var features = _features()
        var data = _matrix(features)
        var target = _regression_target(features)
        with assert_raises(contains="renews its leaves"):
            _ = train_gpu(data, target, L1, _params(2, 2))
        with assert_raises(contains="cannot be combined with goss"):
            _ = train_gpu(
                data, target, SQUARED_ERROR, _params(2, 2),
                goss=GossParams.enable(),
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
