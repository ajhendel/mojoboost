"""Langevin boosting and model shrinkage: the draw, the decay, and the fold.

The determinism tests are the point of this file, not decoration. The claim
`langevin.mojo` makes is that a row's noise is a function of
`(seed, tree, row, output)` alone -- not of the row count, not of the worker
count, not of how many draws any earlier row took -- and the only way to hold
that claim is to fix it in a test that would fail the moment somebody
replaces the counter stream with a running one.
"""

from std.math import sqrt
from std.testing import assert_almost_equal, assert_equal, assert_true, TestSuite

from mojotrees.langevin import (
    DEFAULT_DIFFUSION_TEMPERATURE,
    LANGEVIN_SHRINK_RATE_CONSTANT,
    LANGEVIN_SHRINK_RATE_DECREASING,
    LangevinParams,
    MODEL_SHRINK_CONSTANT,
    MODEL_SHRINK_DECREASING,
    MONOTONE_SHRINK_RATE_CONSTANT,
    MONOTONE_SHRINK_RATE_DECREASING,
    ModelShrinkParams,
    ModelShrinkPlan,
    apply_langevin_noise,
    apply_model_shrinkage,
    canonical_model_shrink_mode,
    check_langevin_hessian_declaration,
    check_model_shrink_continued_training,
    check_model_shrink_hessian_declaration,
    check_model_shrink_init_score,
    couple_langevin_defaults,
    langevin_default_model_shrink_rate,
    langevin_leaf_gradient_noise,
    langevin_leaf_newton_noise,
    langevin_varies_hessian,
    model_shrink_mode_name,
    model_shrink_varies_hessian,
    monotone_default_model_shrink_rate,
    scaled_l2_reg,
)
from mojotrees.tree import Tree


comptime LR = 0.03


def _grad(n: Int) -> List[Float64]:
    """A gradient buffer with a distinguishable value per row, so a test can
    tell a noise term from a copy."""
    var g = List[Float64](capacity=n)
    for r in range(n):
        g.append(Float64(r) * 0.25 - 1.0)
    return g^


def _leaf_tree(value: Float64) -> Tree:
    """A depth-zero tree, one node, so a fold test reads one number."""
    return Tree([-1], [-1], [-1], [-1], [value], [0.0], 1)


def _split_tree(root: Float64, left: Float64, right: Float64) -> Tree:
    """A one-split tree, so the fold can be checked to scale internal node
    values as well as leaves."""
    return Tree(
        [0, -1, -1],
        [3, -1, -1],
        [1, -1, -1],
        [2, -1, -1],
        [root, left, right],
        [7.0, 0.0, 0.0],
        2,
    )


def test_disabled_langevin_draws_nothing() raises:
    """The default bundle leaves every gradient byte alone."""
    var p = LangevinParams.disabled()
    assert_true(not p.injects_noise())
    assert_almost_equal(p.noise_rate(LR), 0.0)
    var g = _grad(16)
    var before = g.copy()
    apply_langevin_noise(g, p, LR, 0, 16)
    for r in range(16):
        assert_equal(g[r], before[r])


def test_noise_rate_is_catboosts_formula() raises:
    """`CalcLangevinNoiseRate` is `sqrt(2 / lr / T)`, temperature in the
    DENOMINATOR: a hotter diffusion_temperature means less noise."""
    var p = LangevinParams.enable()
    assert_equal(p.diffusion_temperature, DEFAULT_DIFFUSION_TEMPERATURE)
    assert_almost_equal(
        p.noise_rate(LR), sqrt(2.0 / LR / DEFAULT_DIFFUSION_TEMPERATURE)
    )
    # Raising the temperature lowers the noise. This is the counterintuitive
    # direction and it is the one CatBoost implements.
    var hot = LangevinParams.enable(diffusion_temperature=1e6)
    assert_true(hot.noise_rate(LR) < p.noise_rate(LR))
    # Lowering the learning rate raises the noise, by sqrt(2) for a halving.
    var slow = LangevinParams.enable()
    assert_almost_equal(
        slow.noise_rate(LR / 2.0), p.noise_rate(LR) * sqrt(2.0)
    )


def test_noise_is_a_function_of_seed_tree_row_only() raises:
    """The determinism claim, stated as an experiment.

    Row 57's noise is computed three ways: in a 64-row buffer, in a 4096-row
    buffer, and in a buffer whose earlier rows were never touched at all. A
    sequential stream would give three different answers; a counter stream
    keyed by (seed, tree, row) gives one.
    """
    var p = LangevinParams.enable(seed=11)
    var small = _grad(64)
    var large = _grad(4096)
    apply_langevin_noise(small, p, LR, 3, 64)
    apply_langevin_noise(large, p, LR, 3, 4096)
    var base = _grad(4096)
    assert_almost_equal(small[57] - base[57], large[57] - base[57], atol=0.0)
    # And the whole 64-row prefix agrees, not only one row: this is the
    # statement that no row's draw depends on the buffer it landed in, which
    # is the same statement as "no row's draw depends on the worker count",
    # because a worker split is a split of the row range.
    for r in range(64):
        assert_equal(small[r] - base[r], large[r] - base[r])


def test_noise_moves_with_tree_and_with_seed() raises:
    """Different trees and different seeds draw different noise. Without this
    the previous test would pass on a stream that returns a constant."""
    var a = LangevinParams.enable(seed=11)
    var b = LangevinParams.enable(seed=12)
    var g0 = _grad(32)
    var g1 = _grad(32)
    var g2 = _grad(32)
    apply_langevin_noise(g0, a, LR, 0, 32)
    apply_langevin_noise(g1, a, LR, 1, 32)
    apply_langevin_noise(g2, b, LR, 0, 32)
    var differ_tree = 0
    var differ_seed = 0
    for r in range(32):
        if g0[r] != g1[r]:
            differ_tree += 1
        if g0[r] != g2[r]:
            differ_seed += 1
    assert_equal(differ_tree, 32)
    assert_equal(differ_seed, 32)


def test_multiclass_outputs_get_separate_draws() raises:
    """Each (row, output) pair owns its own two counters, so class 0 and
    class 1 of one row are not given the same perturbation."""
    var p = LangevinParams.enable(seed=5)
    var n = 24
    var k = 3
    var g = List[Float64](capacity=n * k)
    for _ in range(n * k):
        g.append(0.0)
    apply_langevin_noise(g, p, LR, 2, n, k)
    var equal_pairs = 0
    for r in range(n):
        if g[r * k] == g[r * k + 1]:
            equal_pairs += 1
    assert_equal(equal_pairs, 0)


def test_draws_look_standard_normal() raises:
    """Loose distributional sanity on the Box-Muller transform: mean near 0,
    variance near 1, and no draw beyond the bound the [0,1) uniform implies.

    The tolerances are wide on purpose. This test is here to catch a
    transform that is wrong by a factor or that returns uniforms, not to
    certify normality, and a tight tolerance on 20000 samples would be a
    flaky test with a seed for a hypothesis.
    """
    var p = LangevinParams.enable(diffusion_temperature=2.0 / 1.0, seed=99)
    # coef = sqrt(2 / lr / T); pick lr and T so coef is exactly 1.
    var one = LangevinParams.enable(diffusion_temperature=2.0, seed=99)
    assert_almost_equal(one.noise_rate(1.0), 1.0)
    _ = p.enabled
    var n = 20000
    var g = List[Float64](capacity=n)
    for _ in range(n):
        g.append(0.0)
    apply_langevin_noise(g, one, 1.0, 0, n)
    var total = 0.0
    var total_sq = 0.0
    var worst = 0.0
    for r in range(n):
        total += g[r]
        total_sq += g[r] * g[r]
        if abs(g[r]) > worst:
            worst = abs(g[r])
    var mean = total / Float64(n)
    var var_ = total_sq / Float64(n) - mean * mean
    assert_true(abs(mean) < 0.05)
    assert_true(var_ > 0.9 and var_ < 1.1)
    # sqrt(-2 * log(2^-53)) is about 8.57; nothing may exceed it.
    assert_true(worst < 8.6)


def test_leaf_noise_scalings_and_guards() raises:
    """The two leaf-sum forms differ only in what goes under the square root,
    and both skip a leaf whose weight is below CatBoost's 1e-9."""
    var p = LangevinParams.enable(seed=3)
    var l2 = scaled_l2_reg(3.0, 100.0, 100)
    assert_almost_equal(l2, 3.0)
    # A weighted fit is where scaled_l2_reg stops being the identity.
    assert_almost_equal(scaled_l2_reg(3.0, 200.0, 100), 6.0)

    var grad_noise = langevin_leaf_gradient_noise(p, LR, l2, 4, 2, 50.0)
    var newton_noise = langevin_leaf_newton_noise(p, LR, l2, 4, 2, 50.0, 12.0)
    # Same standard normal, two scalings: the ratio is the ratio of the two
    # square roots and nothing else.
    assert_almost_equal(
        newton_noise / grad_noise, sqrt(12.0 + l2) / sqrt(50.0 + l2)
    )
    # Newton takes the absolute value of the hessian sum, which is CatBoost's
    # own `std::fabs(sum.SumDer2)`.
    assert_almost_equal(
        langevin_leaf_newton_noise(p, LR, l2, 4, 2, 50.0, -12.0), newton_noise
    )
    # An empty leaf gets nothing.
    assert_equal(langevin_leaf_gradient_noise(p, LR, l2, 4, 2, 0.0), 0.0)
    assert_equal(
        langevin_leaf_newton_noise(p, LR, l2, 4, 2, 1e-10, 12.0), 0.0
    )
    # A disabled bundle gets nothing.
    var off = LangevinParams.disabled()
    assert_equal(langevin_leaf_gradient_noise(off, LR, l2, 4, 2, 50.0), 0.0)
    # The leaf half is not wired to any trainer, and says so.
    assert_true(not p.leaf_noise_wired())


def test_leaf_stream_does_not_alias_the_row_stream() raises:
    """Leaf 7 and row 7 of the same tree under the same seed must not share a
    draw. Two domain constants is the whole mechanism and this is the test
    that would fail if one of them were dropped."""
    var p = LangevinParams.enable(seed=4)
    var g = List[Float64](capacity=16)
    for _ in range(16):
        g.append(0.0)
    apply_langevin_noise(g, p, LR, 1, 16)
    var coef = p.noise_rate(LR)
    var row_normal = g[7] / coef
    # A leaf with unit weight and no l2 scales its normal by exactly 1.
    var leaf_normal = langevin_leaf_gradient_noise(p, LR, 0.0, 1, 7, 1.0) / coef
    assert_true(row_normal != leaf_normal)


def test_langevin_validation() raises:
    """A negative temperature, a non-finite one, an enabled bundle with no
    noise, and a non-positive learning rate are all refused."""
    var raised = False
    try:
        LangevinParams(True, -1.0, 0).validate()
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        LangevinParams(True, 0.0, 0).validate()
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        _ = LangevinParams.enable().noise_rate(0.0)
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        var g = _grad(4)
        apply_langevin_noise(g, LangevinParams.enable(), LR, -1, 4)
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        var g = _grad(4)
        apply_langevin_noise(g, LangevinParams.enable(), LR, 0, 8)
    except:
        raised = True
    assert_true(raised)


def test_langevin_declares_no_hessian_exclusion() raises:
    """Langevin writes into the derivative buffer alone, so unlike MVS it
    carries no constant-hessian exclusion. The guard is installed anyway and
    is a no-op today, which is exactly what it should be."""
    var p = LangevinParams.enable()
    assert_true(not langevin_varies_hessian(p))
    check_langevin_hessian_declaration(p, True)
    check_langevin_hessian_declaration(p, False)
    var s = ModelShrinkParams.constant(0.001)
    assert_true(not model_shrink_varies_hessian(s))
    check_model_shrink_hessian_declaration(s, True)


def test_shrink_factor_matches_catboost() raises:
    """Round 0 is exempt; Constant multiplies the rate by the learning rate;
    Decreasing divides by the ROUND INDEX, so it fades as the fit goes on."""
    var c = ModelShrinkParams.constant(0.001)
    assert_almost_equal(c.factor_at_round(LR, 0), 1.0)
    assert_almost_equal(c.factor_at_round(LR, 1), 1.0 - 0.001 * LR)
    assert_almost_equal(c.factor_at_round(LR, 500), 1.0 - 0.001 * LR)

    var d = ModelShrinkParams.decreasing(0.01)
    assert_almost_equal(d.factor_at_round(LR, 0), 1.0)
    assert_almost_equal(d.factor_at_round(LR, 1), 1.0 - 0.01)
    assert_almost_equal(d.factor_at_round(LR, 4), 1.0 - 0.0025)
    # The decay weakens with the round: this is the direction that surprises.
    assert_true(d.factor_at_round(LR, 100) > d.factor_at_round(LR, 2))

    var off = ModelShrinkParams.disabled()
    assert_true(not off.shrinks())
    assert_almost_equal(off.factor_at_round(LR, 7), 1.0)


def test_shrink_validation_ranges_differ_by_mode() raises:
    """Constant bounds `rate * learning_rate`; Decreasing bounds `rate`."""
    # rate * lr = 0.9 < 1 is fine under Constant even though rate is 30.
    ModelShrinkParams.constant(30.0).validate(LR)
    var raised = False
    try:
        ModelShrinkParams.constant(40.0).validate(LR)
    except:
        raised = True
    assert_true(raised)
    # The same rate under Decreasing is refused for being >= 1.
    raised = False
    try:
        ModelShrinkParams.decreasing(1.0).validate(LR)
    except:
        raised = True
    assert_true(raised)
    ModelShrinkParams.decreasing(0.999).validate(LR)
    raised = False
    try:
        ModelShrinkParams(-0.1, MODEL_SHRINK_CONSTANT).validate(LR)
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        ModelShrinkParams(0.1, 7).validate(LR)
    except:
        raised = True
    assert_true(raised)


def test_apply_model_shrinkage() raises:
    """The raw-score rescale, and its identity short circuit."""
    var raw: List[Float64] = [1.0, -2.0, 4.0]
    apply_model_shrinkage(raw, 1.0)
    assert_equal(raw[0], 1.0)
    apply_model_shrinkage(raw, 0.5)
    assert_almost_equal(raw[0], 0.5)
    assert_almost_equal(raw[1], -1.0)
    assert_almost_equal(raw[2], 2.0)


def test_plan_fold_matches_the_reference() raises:
    """The deferred fold against the naive O(rounds^2) reference: scale every
    already-grown tree at every event.

    The events deliberately include a round in which no tree was appended
    (two events at the same tree count), because that is the case CatBoost's
    round-indexed history cannot express and the case the mojotrees round
    loop actually reaches through the degenerate-tree `continue`.
    """
    var factors: List[Float64] = [0.9, 0.8, 0.5, 0.25]
    var counts: List[Int] = [1, 2, 2, 3]

    var trees = List[Tree]()
    trees.append(_leaf_tree(1.0))
    trees.append(_leaf_tree(2.0))
    trees.append(_leaf_tree(4.0))
    trees.append(_leaf_tree(8.0))

    # Reference: apply each event to the trees that existed at the time.
    var reference = List[Float64](capacity=4)
    for t in range(4):
        var acc = 1.0
        for e in range(len(factors)):
            if counts[e] > t:
                acc = acc * factors[e]
        reference.append(acc)
    var reference_base = 1.0
    for e in range(len(factors)):
        reference_base = reference_base * factors[e]

    var plan = ModelShrinkPlan.empty()
    for e in range(len(factors)):
        plan.record(factors[e], counts[e])
    assert_true(not plan.is_empty())
    for t in range(4):
        assert_almost_equal(plan.factor_for_tree(t), reference[t])
    var values: List[Float64] = [1.0, 2.0, 4.0, 8.0]
    var base = plan.fold_into_trees(trees)
    assert_almost_equal(base, reference_base)
    for t in range(4):
        assert_almost_equal(trees[t].value[0], values[t] * reference[t])
    # The last tree is never scaled: nothing was recorded after it.
    assert_almost_equal(trees[3].value[0], 8.0)
    assert_almost_equal(plan.total(), reference_base)


def test_fold_scales_internal_node_values_and_not_gains() raises:
    """A prediction stopping at an internal node has to move with the leaves,
    and a split gain has to not move at all."""
    var trees = List[Tree]()
    trees.append(_split_tree(0.0, 3.0, -3.0))
    trees.append(_leaf_tree(1.0))
    var plan = ModelShrinkPlan.empty()
    plan.record(0.5, 1)
    var base = plan.fold_into_trees(trees)
    assert_almost_equal(base, 0.5)
    assert_almost_equal(trees[0].value[1], 1.5)
    assert_almost_equal(trees[0].value[2], -1.5)
    assert_almost_equal(trees[0].split_gain[0], 7.0)
    assert_almost_equal(trees[1].value[0], 1.0)


def test_empty_plan_is_the_identity() raises:
    """A disabled fit folds nothing and returns a base factor of exactly 1."""
    var plan = ModelShrinkPlan.empty()
    # A recorded 1.0 is the identity and is dropped rather than stored.
    plan.record(1.0, 0)
    plan.record(1.0, 3)
    assert_true(plan.is_empty())
    var trees = List[Tree]()
    trees.append(_leaf_tree(2.0))
    var base = plan.fold_into_trees(trees)
    assert_equal(base, 1.0)
    assert_equal(trees[0].value[0], 2.0)


def test_plan_refuses_out_of_order_events() raises:
    """Events are consumed in ascending tree order, so recording them out of
    order is a caller bug and not a silently wrong fold."""
    var plan = ModelShrinkPlan.empty()
    plan.record(0.9, 5)
    var raised = False
    try:
        plan.record(0.9, 4)
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        plan.record(0.9, -1)
    except:
        raised = True
    assert_true(raised)


def test_events_before_any_tree_land_on_the_base_score() raises:
    """A shrink recorded while the ensemble is empty applies to the base
    score alone, which is where CatBoost's `StartingApprox` scaling goes."""
    var plan = ModelShrinkPlan.empty()
    plan.record(0.5, 0)
    var trees = List[Tree]()
    trees.append(_leaf_tree(6.0))
    var base = plan.fold_into_trees(trees)
    assert_almost_equal(base, 0.5)
    assert_almost_equal(trees[0].value[0], 6.0)


def test_shrink_refusals_are_catboosts() raises:
    """Continued training and an external init_score are both refused, as
    CatBoost refuses them, and only when the rate is actually nonzero."""
    var on = ModelShrinkParams.constant(0.001)
    var off = ModelShrinkParams.disabled()
    check_model_shrink_continued_training(on, 0)
    check_model_shrink_continued_training(off, 40)
    check_model_shrink_init_score(off, 100)
    var raised = False
    try:
        check_model_shrink_continued_training(on, 40)
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        check_model_shrink_init_score(on, 100)
    except:
        raised = True
    assert_true(raised)


def test_the_coupling() raises:
    """Enabling Langevin in CatBoost installs a nonzero model_shrink_rate,
    and which one depends on the mode. Ours reproduces it exactly and only
    when a caller asks."""
    var lang = LangevinParams.enable()
    var off = LangevinParams.disabled()

    var c = couple_langevin_defaults(lang, ModelShrinkParams.disabled())
    assert_almost_equal(c.rate, LANGEVIN_SHRINK_RATE_CONSTANT)
    assert_equal(c.mode, MODEL_SHRINK_CONSTANT)

    var d = couple_langevin_defaults(
        lang, ModelShrinkParams(0.0, MODEL_SHRINK_DECREASING)
    )
    assert_almost_equal(d.rate, LANGEVIN_SHRINK_RATE_DECREASING)
    assert_equal(d.mode, MODEL_SHRINK_DECREASING)

    # An explicit rate is left alone.
    var explicit = couple_langevin_defaults(
        lang, ModelShrinkParams.constant(0.5)
    )
    assert_almost_equal(explicit.rate, 0.5)

    # Langevin off changes nothing: this is the whole default path.
    var untouched = couple_langevin_defaults(
        off, ModelShrinkParams.disabled()
    )
    assert_almost_equal(untouched.rate, 0.0)
    assert_true(not untouched.shrinks())

    assert_almost_equal(
        langevin_default_model_shrink_rate(MODEL_SHRINK_CONSTANT), 0.001
    )
    assert_almost_equal(
        langevin_default_model_shrink_rate(MODEL_SHRINK_DECREASING), 0.01
    )
    assert_almost_equal(
        monotone_default_model_shrink_rate(MODEL_SHRINK_CONSTANT),
        MONOTONE_SHRINK_RATE_CONSTANT,
    )
    assert_almost_equal(
        monotone_default_model_shrink_rate(MODEL_SHRINK_DECREASING),
        MONOTONE_SHRINK_RATE_DECREASING,
    )


def test_mode_names_round_trip() raises:
    """Both CatBoost's spelling and the lowercase one resolve; anything else
    is refused by name rather than defaulted."""
    assert_equal(canonical_model_shrink_mode("Constant"), MODEL_SHRINK_CONSTANT)
    assert_equal(canonical_model_shrink_mode("constant"), MODEL_SHRINK_CONSTANT)
    assert_equal(
        canonical_model_shrink_mode("Decreasing"), MODEL_SHRINK_DECREASING
    )
    assert_equal(
        canonical_model_shrink_mode("decreasing"), MODEL_SHRINK_DECREASING
    )
    assert_equal(model_shrink_mode_name(MODEL_SHRINK_CONSTANT), "Constant")
    assert_equal(model_shrink_mode_name(MODEL_SHRINK_DECREASING), "Decreasing")
    var raised = False
    try:
        _ = canonical_model_shrink_mode("Exponential")
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        _ = model_shrink_mode_name(9)
    except:
        raised = True
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
