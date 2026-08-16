"""Every configuration that must NOT declare a constant hessian, enumerated.

# run_tests: cpu-safe -- host arithmetic only, opens no device.

`boosting.round_has_constant_hessian` is a **safety predicate**, and its own
docstring says why that matters: an overinclusive answer does not raise, does
not crash, and does not move a metric far enough to notice. It silently writes
a wrong hessian plane into the histograms a fit chooses its splits from. The
failure mode is a quietly worse model.

The lane that wired the predicate could not run a single test, by instruction,
so every exclusion it claimed rests on a reading argument. This file turns each
of those arguments into an assertion. It is deliberately an enumeration rather
than a spot check: the risk here is not that one known case is wrong, it is
that some case nobody listed slips through the conjunction, so the test walks
**every objective the registry defines** and asserts the answer for each rather
than asserting only the ones somebody thought of.

Why an unweighted, non-GOSS squared-error fit is the only interesting True: the
hessian of squared error is the weight, which is the literal 1.0 when no
weights are supplied, so the plane is the count plane and storing it twice is
waste. Everything else either varies per row or is made to vary by a sampler.

One thing this file does not test, because it cannot: whether a *declared* fit
actually produces identical models. `tests/test_const_hessian.mojo` does that
by comparing the two accumulation arms as bytes, and `tests/test_golden_bits.mojo`
would catch a wrong declaration on the default path. This file tests only the
decision, which is the part with no other guard.
"""

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees.boosting import round_has_constant_hessian
from mojotrees.goss import GossParams
from mojotrees.histogram import objective_has_constant_hessian
from mojotrees.objective_registry import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    CUSTOM,
    FAIR,
    GAMMA,
    HUBER,
    L1,
    LAMBDARANK,
    MAPE,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
)


def _unweighted() -> List[Float64]:
    return List[Float64]()


def _weighted(n: Int) -> List[Float64]:
    """A weight vector whose entries are all 1.0.

    Deliberately all ones, because that is the adversarial case: a reader
    might reason that weights of 1.0 leave the hessian constant and so could
    be admitted. They do, arithmetically. The predicate still refuses, and it
    is right to, because it is answering a question about the fit's
    configuration rather than about the contents of a vector it would have to
    scan. A predicate that admitted this would have to walk n rows to decide,
    which is the cost the specialization exists to avoid.
    """
    var w = List[Float64](capacity=n)
    for _ in range(n):
        w.append(1.0)
    return w^


def _all_objectives() -> List[Int]:
    return [
        SQUARED_ERROR,
        BINARY_LOGISTIC,
        POISSON,
        HUBER,
        QUANTILE,
        L1,
        CUSTOM,
        LAMBDARANK,
        GAMMA,
        TWEEDIE,
        MAPE,
        FAIR,
        CROSS_ENTROPY,
    ]


def _constant_hessian_objectives() -> List[Int]:
    """The four whose unweighted hessian is the literal 1.0.

    Read off `boosting._fill_grad_hess_into`: each of these ends its row body
    with a store of `w` into `hess` and nothing else. This mirrors LightGBM's
    `IsConstantHessian`, which is `return !weights_` on the L2 loss and is
    inherited unchanged by its L1, Huber and quantile subclasses.
    """
    return [SQUARED_ERROR, L1, HUBER, QUANTILE]


def _is_constant(objective: Int) -> Bool:
    for c in _constant_hessian_objectives():
        if c == objective:
            return True
    return False


def test_every_objective_answers_the_same_as_the_registry() raises:
    """Walk every objective the registry defines, unweighted and non-GOSS.

    The enumeration is the point. A spot check on the four that should be
    True and the two that famously should not would pass even if a thirteenth
    objective were added tomorrow with a per-row hessian and no exclusion.
    """
    var goss = GossParams.disabled()
    for objective in _all_objectives():
        var want = _is_constant(objective)
        assert_equal(
            round_has_constant_hessian(objective, _unweighted(), goss), want
        )
        # The trainers' binding must agree with the histogram-level predicate
        # when neither weights nor a sampler is in play. If these two ever
        # disagree here, one of them has grown a rule the other does not know.
        assert_equal(
            objective_has_constant_hessian(objective, False), want
        )


def test_the_named_non_constant_objectives_are_refused() raises:
    """The ones a reader is most likely to get wrong, called out by name.

    MAPE is the trap: its hessian is `w * label_weight(y)`, which varies with
    the label even when the fit is unweighted, so it looks like a regression
    objective and is not constant. LightGBM excludes it the same way. Poisson,
    gamma, tweedie, fair and cross entropy all have hessians that depend on the
    current raw score, so they change every round as well as every row.
    """
    var goss = GossParams.disabled()
    for objective in [
        MAPE,
        POISSON,
        GAMMA,
        TWEEDIE,
        FAIR,
        CROSS_ENTROPY,
        BINARY_LOGISTIC,
        LAMBDARANK,
        CUSTOM,
    ]:
        assert_false(
            round_has_constant_hessian(objective, _unweighted(), goss)
        )


def test_weights_refuse_every_objective_including_the_constant_ones() raises:
    """A weighted fit has a per-row hessian by construction.

    The hessian of squared error *is* the weight, so this is not a
    conservatism, it is the definition. Class weights, `scale_pos_weight` and
    `is_unbalance` are all expanded into an ordinary per-row `sample_weight`
    by `class_weight.mojo` before any trainer is reached, so covering
    `sample_weight` covers all of them, and that is the claim this asserts.
    """
    var goss = GossParams.disabled()
    var w = _weighted(64)
    for objective in _all_objectives():
        assert_false(round_has_constant_hessian(objective, w, goss))


def test_goss_refuses_even_the_constant_objectives() raises:
    """A configured GOSS run is refused for the whole fit, warmup included.

    `goss.apply_goss_scaling` multiplies the sampled small-gradient rows'
    hessians by an amplification factor and leaves the top rows at 1.0, so a
    rescaled round holds two distinct hessian values under an objective whose
    own code says otherwise. The predicate cannot see that, which is exactly
    why the declaration is the trainer's rather than the objective's.

    The subtlety worth pinning: a configured GOSS fit has warmup rounds during
    which nothing is rescaled and the hessians really are constant, so a
    round-scoped predicate could admit them. This one is deliberately
    fit-scoped, because on the device the declaration is builder state set once
    per fit. Admitting warmup would make the declaration go stale mid-loop with
    no mechanism to withdraw it. That trade costs the specialization on a few
    rounds and buys a declaration that cannot rot.
    """
    var goss = GossParams.enable()
    assert_true(goss.enabled)
    for objective in _all_objectives():
        assert_false(
            round_has_constant_hessian(objective, _unweighted(), goss)
        )
        assert_false(
            round_has_constant_hessian(objective, _weighted(16), goss)
        )


def test_goss_with_warmup_is_still_refused() raises:
    """Even a GOSS configuration whose warmup covers the whole fit.

    `goss.enabled` is the test, not `goss.active(round, learning_rate)`. This
    pins that choice so a later change to `active` cannot quietly widen the
    declaration.
    """
    var goss = GossParams.enable(warmup_rounds=1_000_000)
    for objective in _constant_hessian_objectives():
        assert_false(
            round_has_constant_hessian(objective, _unweighted(), goss)
        )


def test_bagging_is_deliberately_not_an_exclusion() raises:
    """Row sampling does not touch the hessian, so it must not refuse.

    This is the one judgment in the exclusion list rather than a
    transcription, and it is asserted here so it is a decision on the record
    rather than an omission. A bag restricts which rows are accumulated; every
    row it keeps still carries the same hessian it had. The same holds for
    balanced class bagging, which selects rows rather than weighting them.

    `round_has_constant_hessian` takes no bagging argument at all, which is
    the structural expression of that claim, so what this test really pins is
    the signature. If a future sampler introduces a per-row weight rather than
    a per-row selection, this is the assertion that should be made to fail
    on purpose.
    """
    var goss = GossParams.disabled()
    for objective in _constant_hessian_objectives():
        assert_true(
            round_has_constant_hessian(objective, _unweighted(), goss)
        )


def test_the_only_true_case_is_the_one_we_think_it_is() raises:
    """A single positive control, so the file cannot pass by refusing
    everything.

    Every other test here asserts a False. If the predicate were replaced by
    `return False` the rest of this file would still pass, which would make it
    a test of nothing. This is the assertion that fails in that case.
    """
    var goss = GossParams.disabled()
    assert_true(
        round_has_constant_hessian(SQUARED_ERROR, _unweighted(), goss)
    )
    var trues = 0
    for objective in _all_objectives():
        if round_has_constant_hessian(objective, _unweighted(), goss):
            trues += 1
    assert_equal(trues, len(_constant_hessian_objectives()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
