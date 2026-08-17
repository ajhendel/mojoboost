"""CatBoost's `leaf_estimation_iterations`, and the three things it has to
prove.

A leaf's value in every other trainer in this repository is one Newton step,
`-T(G) / (H + lambda_l2)`, taken at the raw scores the round started from.
`leaf_estimation_iterations = k` re-evaluates the leaf's rows at the value the
leaf currently holds and takes another step, `k - 1` more times. The default is
1 and stays 1, which is LightGBM's behavior.

1. **At the default it is not there.** Not "close to not there": a fit with the
   parameter absent and one with it explicitly 1 are the same ensemble bit for
   bit, at every worker count, compared with `to_bits()`. Nothing in this file
   compares to a tolerance. This is a stronger claim than it looks, and the
   implementation earns it in one specific way: iteration 1 is *never*
   recomputed. The value the grower wrote from the histogram's bin-order sums
   is kept exactly as written, so there is no second route to the first Newton
   step whose fold order could disagree with the histogram's. A version that
   recomputed step 1 from the rows would move bits at k = 1, and
   `test_one_iteration_is_absent_bit_for_bit` is what would catch it.

2. **Above 1 the extra steps really fire, at the right value.** A test that
   passes whether or not a loop ran is worthless, so the gate assertion here is
   not "something changed". A one-round binary-logistic fit is grown twice, at
   k = 1 and k = 2. Every row's raw score in round 0 is the base score, so the
   test can reconstruct what the second step must be from public parts alone --
   `fill_grad_hess` over the leaf's rows at `base + v1`, summed in ascending row
   order, through `raw_leaf_output` -- and assert the k = 2 leaf equals
   `v1 + step` with `to_bits()`. That equality can only hold if the loop ran,
   once, on those rows, at that point. It reuses `v1` from the k = 1 fit rather
   than recomputing the first step, so it cannot pass by reproducing an
   arithmetic mistake. `test_three_iterations_moves_again_from_two` then shows
   the count is a loop bound and not a flag.

3. **The order the extra steps fold in is fixed.** The per-row derivatives come
   from `fill_grad_hess`, which is bit-identical at any worker count because it
   sums nothing across rows; the reduction to `G` and `H` is a plain sequential
   loop over the leaf's row list, dispatched nowhere. `MOJOTREES_NUM_WORKERS` at
   1, 3 and 8 must therefore give one ensemble, and the test asserts that at
   k = 1 **and** at k = 3, because a worker-dependence introduced by the new
   fold would be invisible in a k = 1 run.

Everything this cannot do it refuses by name rather than ignoring: the renewing
objectives, GOSS, path smoothing, the multiclass trainers, and a parameter
string.
"""

from std.math import log
from std.os import setenv
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.boosting import (
    BINARY_LOGISTIC,
    Booster,
    BoosterParams,
    CROSS_ENTROPY,
    GAMMA,
    L1,
    MAPE,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
    catboost_leaf_estimation_iterations,
    fill_grad_hess,
    train,
    train_multiclass,
)
from mojotrees.goss import GossParams
from mojotrees.params import parse_params
from mojotrees.tree import Tree, TreeParams
from mojotrees.tree_parameters_extra import (
    DEFAULT_LEAF_ESTIMATION_ITERATIONS,
    ExtraTreeParams,
    raw_leaf_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _workers(n: String):
    _ = setenv("MOJOTREES_NUM_WORKERS", n)


def _auto():
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


def _bits(v: Float64) -> UInt64:
    return UInt64(v.to_bits())


def _uniform(seed: UInt64) -> Float64:
    """A cheap deterministic spread in [0, 1); the fixture only needs the
    features to differ from each other in a fixed way."""
    var x = seed * 6364136223846793005 + 1442695040888963407
    x ^= x >> 33
    x *= 0xFF51AFD7ED558CCD
    x ^= x >> 33
    return Float64(x >> 11) * (1.0 / 9007199254740992.0)


def _features(n_rows: Int, n_features: Int) -> List[Float64]:
    var out = List[Float64](capacity=n_rows * n_features)
    for k in range(n_rows * n_features):
        out.append(_uniform(UInt64(k) + 17))
    return out^


def _labels_binary(
    features: List[Float64], n_rows: Int
) -> List[Float64]:
    """A {0, 1} label that depends on the first two features, so the tree has
    something to split on and the leaves are not all one class -- a pure leaf
    would have a gradient that the second Newton step could not move much, and
    the point of the gate fixture is that it moves."""
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        var s = features[r] + 0.5 * features[n_rows + r]
        out.append(1.0 if s > 0.7 else 0.0)
    return out^


def _regression_target(
    features: List[Float64], n_rows: Int
) -> List[Float64]:
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(
            2.0 * features[r] - features[n_rows + r] + 0.25 * features[r] * features[r]
        )
    return out^


def _positive_target(features: List[Float64], n_rows: Int) -> List[Float64]:
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(1.0 + 3.0 * features[r] + features[n_rows + r])
    return out^


def _extra(iterations: Int) -> ExtraTreeParams:
    var e = ExtraTreeParams()
    e.leaf_estimation_iterations = iterations
    return e^


def _params(
    n_estimators: Int, iterations: Int, num_leaves: Int = 8
) raises -> BoosterParams:
    return BoosterParams(
        n_estimators,
        0.1,
        TreeParams(
            num_leaves, 20, 1.0, 1e-3, 0.0,
            extra=_extra(iterations),
        ),
    )


def _ensemble_bits(booster: Booster) -> List[UInt64]:
    """Every leaf value and every routing decision in the ensemble, as
    integers. Comparing this rather than a prediction means a leaf that moved
    and a split that moved are both visible, and neither can be hidden by
    another cancelling it out at a row."""
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


def _binary_base_score(target: List[Float64]) -> Float64:
    """`boosting._base_score` for BINARY_LOGISTIC on an unweighted fit: the
    logit of the mean label. Restated here rather than imported because the
    gate assertion has to know the point the first tree was grown from, and a
    single-round fit's raw score is exactly this for every row."""
    var mean = 0.0
    for r in range(len(target)):
        mean += target[r]
    mean /= Float64(len(target))
    return log(mean / (1.0 - mean))


def _second_step_from_public_parts(
    tree: Tree,
    data: BinnedMatrix,
    target: List[Float64],
    base: Float64,
    objective: Int,
    node: Int,
    lambda_l1: Float64,
    lambda_l2: Float64,
) raises -> Float64:
    """What one more Newton step on leaf `node` must produce, built out of
    `fill_grad_hess` and `raw_leaf_output` and nothing private.

    `base` is every row's raw score, which in round 0 of a fit without an
    `init_score` is the objective's base score. The rows are collected by
    sweeping the dataset in ascending order and keeping the ones that route to
    `node`, which is the same list in the same order the trainer folds -- the
    grower's partition keeps each side of every split ascending.
    """
    var v = tree.value[node]
    var raw_l = List[Float64]()
    var tgt_l = List[Float64]()
    for r in range(data.n_rows):
        if tree.leaf_index_row(data, r) == node:
            raw_l.append(base + v)
            tgt_l.append(target[r])
    var grad = List[Float64]()
    var hess = List[Float64]()
    fill_grad_hess(raw_l, tgt_l, objective, [], 0.9, grad, hess)
    var g_sum = 0.0
    var h_sum = 0.0
    for i in range(len(raw_l)):
        g_sum += grad[i]
        h_sum += hess[i]
    return v + raw_leaf_output(g_sum, h_sum, lambda_l1, lambda_l2)


# ---------------------------------------------------------------------------
# 1. The default moves nothing
# ---------------------------------------------------------------------------


def test_one_iteration_is_absent_bit_for_bit() raises:
    """A fit with the parameter untouched and a fit with it explicitly 1 are
    the same ensemble, to the bit, on four objectives.

    This is the claim the whole feature rests on and it is not a formality:
    the trainer calls `_estimate_leaf_values` unconditionally, so if that
    function recomputed the first Newton step instead of returning at
    `iterations <= 1`, the histogram's bin-order sums and a row-order sum
    would disagree in the last few significand bits and every one of these
    comparisons would fail.
    """
    _auto()
    var n_rows = 500
    var n_features = 4
    var features = _features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)

    var reg = _regression_target(features, n_rows)
    var bin_target = _labels_binary(features, n_rows)
    var pos = _positive_target(features, n_rows)

    var absent = BoosterParams(12, 0.1, TreeParams(8, 20, 1.0, 1e-3))
    var explicit = _params(12, 1)

    # The parameter is untouched on `absent` and set to its own default on
    # `explicit`, which is the pair the claim is about.
    assert_equal(
        absent.tree.extra.leaf_estimation_iterations,
        DEFAULT_LEAF_ESTIMATION_ITERATIONS,
    )
    assert_equal(explicit.tree.extra.leaf_estimation_iterations, 1)
    assert_false(absent.tree.extra.leaf_estimation_active())
    assert_false(explicit.tree.extra.leaf_estimation_active())

    _assert_same_bits(
        _ensemble_bits(train(data, reg, SQUARED_ERROR, absent)),
        _ensemble_bits(train(data, reg, SQUARED_ERROR, explicit)),
        "squared_error",
    )
    _assert_same_bits(
        _ensemble_bits(train(data, bin_target, BINARY_LOGISTIC, absent)),
        _ensemble_bits(train(data, bin_target, BINARY_LOGISTIC, explicit)),
        "binary_logistic",
    )
    _assert_same_bits(
        _ensemble_bits(train(data, pos, POISSON, absent)),
        _ensemble_bits(train(data, pos, POISSON, explicit)),
        "poisson",
    )
    _assert_same_bits(
        _ensemble_bits(train(data, pos, GAMMA, absent)),
        _ensemble_bits(train(data, pos, GAMMA, explicit)),
        "gamma",
    )


def test_one_iteration_is_absent_on_a_renewing_objective() raises:
    """The renewing objectives refuse `> 1`, so the one thing that has to hold
    for them is that 1 is still inert: `_estimate_leaf_values` runs after
    `_renew_leaf_values` on every round of an L1 fit and must not touch the
    percentile the renewal just wrote."""
    _auto()
    var n_rows = 400
    var features = _features(n_rows, 3)
    var data = bin_equal_width(features, n_rows, 3, 16)
    var reg = _regression_target(features, n_rows)

    var absent = BoosterParams(10, 0.1, TreeParams(8, 20, 1.0, 1e-3))
    var explicit = _params(10, 1)
    _assert_same_bits(
        _ensemble_bits(train(data, reg, L1, absent)),
        _ensemble_bits(train(data, reg, L1, explicit)),
        "l1",
    )
    _assert_same_bits(
        _ensemble_bits(train(data, reg, QUANTILE, absent)),
        _ensemble_bits(train(data, reg, QUANTILE, explicit)),
        "quantile",
    )


# ---------------------------------------------------------------------------
# 2. The gate: above 1 the extra steps fire, at the value they must
# ---------------------------------------------------------------------------


def test_two_iterations_equal_one_plus_the_reconstructed_step() raises:
    """The gate assertion. Every leaf of a one-round binary-logistic fit at
    k = 2 must equal, to the bit, that leaf's k = 1 value plus one more Newton
    step reconstructed from `fill_grad_hess` and `raw_leaf_output` over the
    leaf's own rows at `base + v1`.

    One round is what makes the reconstruction possible: every row's raw score
    is still the base score, so the point the second step is taken at is known
    without reproducing the boosting loop. And the assertion is exact, not
    directional -- it fails if the step is taken at `raw` instead of
    `raw + v1`, if it is scaled by the learning rate, if `lambda_l2` is
    dropped, if the fold runs in some other order, or if the loop simply did
    not run.
    """
    _auto()
    var n_rows = 600
    var n_features = 4
    var features = _features(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 32)
    var labels = _labels_binary(features, n_rows)

    var one = train(data, labels, BINARY_LOGISTIC, _params(1, 1))
    var two = train(data, labels, BINARY_LOGISTIC, _params(1, 2))

    assert_equal(len(one.trees), 1)
    assert_equal(len(two.trees), 1)
    ref t1 = one.trees[0]
    ref t2 = two.trees[0]

    # The structure is fixed before leaf estimation runs, so the two trees are
    # the same tree with different leaf values. If this ever fails, leaf
    # estimation has leaked into the split search.
    assert_equal(len(t1.feature), len(t2.feature))
    assert_true(t1.n_leaves > 1)
    for node in range(len(t1.feature)):
        assert_equal(t1.feature[node], t2.feature[node])
        assert_equal(t1.threshold_bin[node], t2.threshold_bin[node])

    var base = _binary_base_score(labels)
    var moved = 0
    for node in range(len(t1.feature)):
        if t1.feature[node] >= 0:
            # An internal node keeps the value it was created with; leaf
            # estimation rewrites leaves only.
            assert_equal(_bits(t1.value[node]), _bits(t2.value[node]))
            continue
        var expected = _second_step_from_public_parts(
            t1, data, labels, base, BINARY_LOGISTIC, node, 0.0, 1.0
        )
        assert_equal(_bits(t2.value[node]), _bits(expected))
        if _bits(t1.value[node]) != _bits(t2.value[node]):
            moved += 1
    # Without this the test would pass on a fixture where every second step
    # happened to be zero, which is exactly the failure mode it exists to
    # rule out.
    assert_true(moved > 0)


def test_three_iterations_moves_again_from_two() raises:
    """`leaf_estimation_iterations` is a loop bound, not a flag: the k = 3
    leaf must be the k = 2 leaf plus one more reconstructed step, which is
    the same exact assertion the gate above makes, applied one level up. A
    single-shot "take one extra step when the parameter is set"
    implementation passes the gate above and fails here.

    One round again, so the three fits share a structure and every leaf can
    be matched by node id.
    """
    _auto()
    var n_rows = 600
    var features = _features(n_rows, 4)
    var data = bin_equal_width(features, n_rows, 4, 32)
    var labels = _labels_binary(features, n_rows)

    var two = train(data, labels, BINARY_LOGISTIC, _params(1, 2))
    var three = train(data, labels, BINARY_LOGISTIC, _params(1, 3))
    ref t2 = two.trees[0]
    ref t3 = three.trees[0]
    assert_equal(len(t2.feature), len(t3.feature))

    var base = _binary_base_score(labels)
    var moved = 0
    for node in range(len(t2.feature)):
        if t2.feature[node] >= 0:
            continue
        var expected = _second_step_from_public_parts(
            t2, data, labels, base, BINARY_LOGISTIC, node, 0.0, 1.0
        )
        assert_equal(_bits(t3.value[node]), _bits(expected))
        if _bits(t2.value[node]) != _bits(t3.value[node]):
            moved += 1
    assert_true(moved > 0)


def test_extra_steps_reach_every_smooth_objective() raises:
    """The mechanism is objective-agnostic because it goes back through
    `fill_grad_hess`, so it must move leaf values on objectives whose
    curvature is nothing like the logistic one. Poisson, gamma and tweedie all
    have the exponential link and a raw-score-dependent hessian; squared
    error's hessian is constant and its second step is driven purely by
    `lambda_l2`, which is the case most likely to be silently inert.

    One round per arm, so the k = 1 and k = 4 trees share a structure and a
    moved leaf cannot be confused with a moved split.
    """
    _auto()
    var n_rows = 500
    var features = _features(n_rows, 4)
    var data = bin_equal_width(features, n_rows, 4, 32)
    var pos = _positive_target(features, n_rows)
    var reg = _regression_target(features, n_rows)

    var objectives: List[Int] = [POISSON, GAMMA, TWEEDIE, SQUARED_ERROR]
    var alphas: List[Float64] = [0.9, 0.9, 1.5, 0.9]
    for i in range(len(objectives)):
        var obj = objectives[i]
        var y = reg.copy() if obj == SQUARED_ERROR else pos.copy()
        var a = train(data, y, obj, _params(1, 1), [], alphas[i])
        var b = train(data, y, obj, _params(1, 4), [], alphas[i])
        ref ta = a.trees[0]
        ref tb = b.trees[0]
        assert_equal(len(ta.feature), len(tb.feature))
        var moved = 0
        for node in range(len(ta.feature)):
            if ta.feature[node] >= 0:
                # Leaf estimation rewrites leaves only.
                assert_equal(_bits(ta.value[node]), _bits(tb.value[node]))
                continue
            if _bits(ta.value[node]) != _bits(tb.value[node]):
                moved += 1
        assert_true(moved > 0)


def test_max_delta_step_still_caps_every_iteration() raises:
    """`max_delta_step` is a projection, so re-applying it after every step is
    the identity on an already-capped value and every intermediate value stays
    inside the cap. The observable consequence is that no leaf of a k = 6 fit
    escapes the cap, which a version that capped only the first step would
    violate."""
    _auto()
    var n_rows = 500
    var features = _features(n_rows, 4)
    var data = bin_equal_width(features, n_rows, 4, 32)
    var labels = _labels_binary(features, n_rows)

    var cap = 0.05
    var e = _extra(6)
    e.max_delta_step = cap
    var params = BoosterParams(
        4, 0.1, TreeParams(8, 20, 1.0, 1e-3, 0.0, extra=e^)
    )
    var booster = train(data, labels, BINARY_LOGISTIC, params)
    var capped = 0
    for t in range(len(booster.trees)):
        ref tree = booster.trees[t]
        for node in range(len(tree.feature)):
            if tree.feature[node] >= 0:
                continue
            assert_true(tree.value[node] <= cap)
            assert_true(tree.value[node] >= -cap)
            if _bits(tree.value[node]) == _bits(cap) or (
                _bits(tree.value[node]) == _bits(-cap)
            ):
                capped += 1
    # The cap has to actually bite somewhere, or the assertions above hold
    # vacuously and prove nothing about the per-iteration projection.
    assert_true(capped > 0)


# ---------------------------------------------------------------------------
# 3. The fold order does not move with the worker count
# ---------------------------------------------------------------------------


def test_worker_count_moves_no_bit_at_one_or_above() raises:
    """1, 3 and 8 workers give one ensemble, at k = 1 and at k = 3.

    Both counts matter. The k = 1 arm is the existing contract and would pass
    even if leaf estimation had a worker-dependent fold, because that fold
    never runs there. The k = 3 arm is the one that exercises it: the extra
    steps call `fill_grad_hess`, which dispatches row blocks, and then reduce
    to `G` and `H` in a sequential loop over the leaf's rows. Only the first
    of those two is dispatched, and it sums nothing across rows.
    """
    var n_rows = 700
    var features = _features(n_rows, 5)
    var data = bin_equal_width(features, n_rows, 5, 32)
    var labels = _labels_binary(features, n_rows)

    var counts: List[String] = ["1", "3", "8"]
    var iterations: List[Int] = [1, 3]
    for k in range(len(iterations)):
        var params = _params(5, iterations[k])
        _workers(counts[0])
        var reference = _ensemble_bits(
            train(data, labels, BINARY_LOGISTIC, params)
        )
        for c in range(1, len(counts)):
            _workers(counts[c])
            _assert_same_bits(
                reference,
                _ensemble_bits(train(data, labels, BINARY_LOGISTIC, params)),
                "workers",
            )
    _auto()


# ---------------------------------------------------------------------------
# Refusals: everything it cannot do, it says so
# ---------------------------------------------------------------------------


def test_zero_and_negative_counts_are_refused() raises:
    with assert_raises():
        _extra(0).check_scalars(20)
    with assert_raises():
        _extra(-1).check_scalars(20)
    # 1 and above pass, and 1 is inert.
    _extra(1).check_scalars(20)
    _extra(9).check_scalars(20)
    assert_false(_extra(1).leaf_estimation_active())
    assert_true(_extra(2).leaf_estimation_active())


def test_path_smooth_is_refused_beside_it() raises:
    """Smoothing is an affine contraction toward the parent, not a projection,
    so applying it once per iteration would walk the leaf to its parent's
    output. Refused rather than silently picking one of two wrong placements.
    Either alone is fine."""
    var both = _extra(4)
    both.path_smooth = 2.0
    with assert_raises():
        both.check_scalars(20)

    var smoothing_only = _extra(1)
    smoothing_only.path_smooth = 2.0
    smoothing_only.check_scalars(20)
    _extra(4).check_scalars(20)


def test_renewing_objectives_are_refused() raises:
    """L1, quantile and MAPE already hold the exact minimizer of the leaf's
    loss. Refused at the trainer, where the objective is known."""
    var n_rows = 300
    var features = _features(n_rows, 3)
    var data = bin_equal_width(features, n_rows, 3, 16)
    var reg = _regression_target(features, n_rows)
    var params = _params(3, 4)

    with assert_raises():
        _ = train(data, reg, L1, params)
    with assert_raises():
        _ = train(data, reg, QUANTILE, params)
    var pos = _positive_target(features, n_rows)
    with assert_raises():
        _ = train(data, pos, MAPE, params)
    # The same objectives at the default train fine, which is what makes the
    # refusal specific to the combination rather than to the objective.
    _ = train(data, reg, L1, _params(3, 1))


def test_goss_is_refused() raises:
    """GOSS amplifies the sampled rows' derivatives after they are computed,
    so a recomputation from the raw scores would be minimizing a differently
    weighted loss than the first step did."""
    var n_rows = 400
    var features = _features(n_rows, 4)
    var data = bin_equal_width(features, n_rows, 4, 32)
    var labels = _labels_binary(features, n_rows)
    with assert_raises():
        _ = train(
            data, labels, BINARY_LOGISTIC, _params(5, 3), [], 0.9,
            goss=GossParams.enable(),
        )
    # Same GOSS run at the default is unaffected.
    _ = train(
        data, labels, BINARY_LOGISTIC, _params(5, 1), [], 0.9,
        goss=GossParams.enable(),
    )


def test_multiclass_refuses_rather_than_ignoring() raises:
    """A softmax row's derivative for class k reads every class's raw score,
    so the single-output recomputation does not carry over. CatBoost defaults
    `MultiClass` to one Newton iteration anyway."""
    var n_rows = 300
    var features = _features(n_rows, 3)
    var data = bin_equal_width(features, n_rows, 3, 16)
    var labels = List[Int](capacity=n_rows)
    for r in range(n_rows):
        labels.append(r % 3)
    with assert_raises():
        _ = train_multiclass(data, labels, 3, _params(3, 2))
    _ = train_multiclass(data, labels, 3, _params(3, 1))


def test_the_parameter_string_carries_the_name() raises:
    """The name parses, 1 is inert, and above 1 is expressible here now, but
    only where the trainer this string routes to implements it.

    The blanket refusal this test used to assert was right for as long as every
    count above 1 was reachable from the Mojo API alone. It stopped being right
    on 2026-08-16 (`e3cfb47`), and the reason is that a shipped default has to
    hold on every surface. CatBoost mode resolves this parameter per objective,
    so `grow_policy=oblivious objective=binary` is a configuration whose
    CatBoost value is 10, and a string that could not express 10 could not
    express the default it is asked to port.

    What replaced the refusal is not permission, so the teeth here moved rather
    than came out. `params._check_leaf_estimation_routing` takes the same
    verdict `_parse_params` in `bindings/_mojotrees.mojo` takes, from the same
    question -- which trainer is this configuration about to reach -- and a
    parameter string reaches exactly two. `model.fit` runs the extra Newton
    steps on both backends; `model.fit_multiclass` reads the field nowhere, and
    that is the one the refusal narrowed to.

    So the property under guard is the one it always was, *a parameter string
    never silently drops this setting*, and it is now asserted in both
    directions. The single-output string must carry the count onto the config,
    which is what a regression that resolved the value and then ignored it
    would fail; the multiclass string must still refuse by name, which is what
    a regression that deleted the routing check along with the blanket one
    would fail. Asserting only that the single-output string parses would leave
    the second half unguarded, and the second half is where the silent drop
    actually lives.

    The range check is a different rule and did not narrow with the routing, so
    it is asserted beside them. 0 is still an error at the surface it was typed
    on rather than at the leaf that would first divide by it.
    """
    var one = parse_params("objective=regression leaf_estimation_iterations=1")
    assert_equal(
        one.booster.tree.extra.leaf_estimation_iterations, 1
    )
    assert_false(one.booster.tree.extra.leaf_estimation_active())
    # It is not folded into `is_active`, which is the split search's gate:
    # extra Newton steps touch no candidate and no gain.
    assert_false(one.booster.tree.extra.is_active())

    # Above 1 on a single-output objective is carried onto the config rather
    # than dropped, because `model.fit` honors it on both backends.
    var two = parse_params("objective=regression leaf_estimation_iterations=2")
    assert_equal(two.booster.tree.extra.leaf_estimation_iterations, 2)
    assert_true(two.booster.tree.extra.leaf_estimation_active())
    # And still not the split search's gate, at the count that actually fires.
    assert_false(two.booster.tree.extra.is_active())

    # Above 1 on multiclass is still refused, and refused BY NAME, because
    # `boosting.train_multiclass`, `train_gpu.train_multiclass_gpu` and
    # `boosting_sparse.train_multiclass_sparse` read the field nowhere. This is
    # the assertion that used to cover every objective and now covers the one
    # objective that still needs it.
    with assert_raises(contains="model.fit_multiclass"):
        _ = parse_params(
            "objective=multiclass num_class=3 leaf_estimation_iterations=2"
        )
    # The same multiclass string at the default parses, which is what makes the
    # refusal specific to the count rather than to the objective.
    var multi = parse_params(
        "objective=multiclass num_class=3 leaf_estimation_iterations=1"
    )
    assert_equal(multi.booster.tree.extra.leaf_estimation_iterations, 1)

    with assert_raises(contains="at least 1"):
        _ = parse_params("objective=regression leaf_estimation_iterations=0")


def test_it_is_not_in_the_split_search_gate() raises:
    """`ExtraTreeParams.is_active` is consumed as "the device split search
    cannot do this" and as `split._feature_gain`'s per-feature cost gate. A
    leaf estimation count belongs in neither, and folding it in would send a
    fit down `passes_min_gain`, which rejects a gain of exactly 0.0 that the
    inactive path admits."""
    assert_false(_extra(1).is_active())
    assert_false(_extra(10).is_active())
    assert_false(_extra(10).needs_grower_support())
    assert_false(_extra(10).needs_leaf_finish())
    assert_true(_extra(10).leaf_estimation_active())


# ---------------------------------------------------------------------------
# The recorded CatBoost defaults, which are a record and not our defaults
# ---------------------------------------------------------------------------


def test_catboost_defaults_are_recorded_and_unused() raises:
    """Verified from `catboost_options.cpp::GetEstimationMethodDefaults`
    (master, read 2026-08-16). Logloss and cross entropy really are 10.
    Poisson is 10 as well, which the brief did not mention. Everything else we
    have is 1, and CatBoost's MultiClass is 1 too -- the widely repeated "10
    for multiclass" reads the `defaultGradientIterations` slot, which the
    default Newton method never selects.

    Our default is 1 for every objective and this function is read by nothing
    in the package, which is the property this test pins.
    """
    assert_equal(catboost_leaf_estimation_iterations(BINARY_LOGISTIC), 10)
    assert_equal(catboost_leaf_estimation_iterations(CROSS_ENTROPY), 10)
    assert_equal(catboost_leaf_estimation_iterations(POISSON), 10)
    assert_equal(catboost_leaf_estimation_iterations(SQUARED_ERROR), 1)
    assert_equal(catboost_leaf_estimation_iterations(L1), 1)
    assert_equal(catboost_leaf_estimation_iterations(QUANTILE), 1)
    assert_equal(catboost_leaf_estimation_iterations(MAPE), 1)
    assert_equal(catboost_leaf_estimation_iterations(GAMMA), 1)
    assert_equal(catboost_leaf_estimation_iterations(TWEEDIE), 1)

    # The record does not move the default.
    assert_equal(DEFAULT_LEAF_ESTIMATION_ITERATIONS, 1)
    assert_equal(ExtraTreeParams().leaf_estimation_iterations, 1)
    assert_equal(
        TreeParams.default().extra.leaf_estimation_iterations, 1
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
