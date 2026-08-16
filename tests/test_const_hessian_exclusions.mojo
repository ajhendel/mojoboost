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
from mojotrees.sampling import (
    BayesianBootstrapParams,
    bayesian_bootstrap_varies_hessian,
    check_bayesian_bootstrap_hessian_declaration,
    refresh_bayesian_bootstrap,
    MvsAudit,
    MvsBootstrapParams,
    check_mvs_hessian_declaration,
    mvs_auto_lambda_from_gradients,
    mvs_varies_hessian,
    refresh_mvs_bootstrap,
)
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


def test_bayesian_bootstrap_is_an_exclusion() raises:
    """The case `test_bagging_is_deliberately_not_an_exclusion` predicted.

    That test's docstring says: "If a future sampler introduces a per-row
    weight rather than a per-row selection, this is the assertion that should
    be made to fail on purpose." CatBoost's Bayesian bootstrap is that sampler.
    It keeps every row and gives each a random weight per tree, and the weight
    multiplies the row's derivatives exactly as a `sample_weight` does
    (CatBoost `CalcWeightedData`), so a bootstrapped fit has a per-row hessian
    under every objective including the four constant ones.

    `round_has_constant_hessian` cannot be handed the bootstrap configuration:
    its three inputs are fixed by a contract the device trainers bind to. So
    the exclusion is asserted here in the two forms a caller can actually
    reach it in, and both are checked because a caller that reaches neither is
    the silent-wrong-hessian failure this file exists to prevent.
    """
    var goss = GossParams.disabled()
    var params = BayesianBootstrapParams.enable()

    # Form one: the effective weights reach the predicate as `sample_weight`,
    # which is what they are. The first exclusion in the predicate's own
    # docstring then refuses every objective, constant ones included.
    var weights = List[Float64]()
    refresh_bayesian_bootstrap(weights, params, _unweighted(), 48, 0)
    assert_equal(len(weights), 48)
    for objective in _all_objectives():
        assert_false(round_has_constant_hessian(objective, weights, goss))

    # Form two: the sampler's own predicate, which is what a caller consults
    # when it has the configuration but not yet the vector. Fit-scoped like
    # `goss.enabled`, so a zero temperature -- CatBoost's early return to
    # all-ones -- is refused too rather than being reasoned about.
    assert_true(bayesian_bootstrap_varies_hessian(params))
    assert_true(
        bayesian_bootstrap_varies_hessian(
            BayesianBootstrapParams.enable(temperature=0.0)
        )
    )
    assert_false(
        bayesian_bootstrap_varies_hessian(BayesianBootstrapParams.disabled())
    )

    # And the guard that makes the exclusion impossible to omit: declaring a
    # constant hessian beside an active bootstrap raises instead of quietly
    # writing the wrong plane.
    var raised = False
    try:
        check_bayesian_bootstrap_hessian_declaration(params, True)
    except:
        raised = True
    assert_true(raised)
    # ... and the same call is silent when the declaration is False, and when
    # the bootstrap is off whatever was declared, so the guard is a refusal of
    # one combination rather than a blanket one.
    check_bayesian_bootstrap_hessian_declaration(params, False)
    check_bayesian_bootstrap_hessian_declaration(
        BayesianBootstrapParams.disabled(), True
    )


def test_mvs_excludes_the_constant_hessian_the_same_way() raises:
    """MVS carries the identical exclusion, and for the identical reason.

    `Bootstrap` calls `CalcWeightedData` for MVS on the same line it does for
    Bayesian, so the derivatives come out multiplied by a per-row weight
    either way. The two samplers differ in how the weight is drawn and not at
    all in what it does to the hessian plane, which is the only thing this
    file is about.

    Asserted in the same two forms as the Bayesian case above, because a
    caller that reaches neither is the silent-wrong-hessian failure this file
    exists to prevent.

    One thing here is NOT a mirror and is worth the extra line. MVS is a row
    *dropper*: a row below the threshold that loses its draw gets weight
    exactly 0.0 and is removed from the fold outright. So the weight vector
    it produces contains zeros where the Bayesian one contains small positive
    numbers, and a reader might reason that a zero weight is the same as an
    absent row and leaves the surviving rows' hessian constant. It does not,
    and the first exclusion refuses the vector for the same reason either
    way -- which is asserted below rather than argued.
    """
    var goss = GossParams.disabled()
    var params = MvsBootstrapParams.enable()

    # Form one: the effective weights reach the predicate as `sample_weight`.
    # A ramp of gradients, so the draw has both certainly-kept rows (above the
    # threshold) and rows decided by the coin, which is the vector shape that
    # makes this exclusion matter.
    var gradients = List[Float64]()
    for i in range(48):
        gradients.append(Float64(i + 1) * 0.25)
    var auto_lambda = mvs_auto_lambda_from_gradients(gradients, 48)
    var weights = List[Float64]()
    var audit = MvsAudit.empty()
    refresh_mvs_bootstrap(
        weights, audit, params, gradients, _unweighted(), 48, 0, auto_lambda
    )
    assert_equal(len(weights), 48)
    # The guard branch must NOT have fired here: the lambda is derived and
    # positive on a nonzero gradient ramp, so a nonzero count would mean this
    # fixture is exercising the degenerate path instead of the ordinary one
    # and the assertions below would be about the wrong thing.
    assert_equal(audit.blocks_guarded, 0)
    for objective in _all_objectives():
        assert_false(round_has_constant_hessian(objective, weights, goss))

    # Form two: the configuration-level predicate, fit-scoped like GOSS's, so
    # a subsample of 1.0 -- which keeps every row and draws nothing -- is
    # still refused rather than reasoned about. That coarseness is deliberate
    # and mirrors the Bayesian temperature=0.0 case.
    assert_true(mvs_varies_hessian(params))
    assert_true(mvs_varies_hessian(MvsBootstrapParams.enable(1.0)))
    assert_false(mvs_varies_hessian(MvsBootstrapParams.disabled()))

    # And the guard that makes the exclusion impossible to omit.
    var raised = False
    try:
        check_mvs_hessian_declaration(params, True)
    except:
        raised = True
    assert_true(raised)
    # Silent when the declaration is False, and when the sampler is off
    # whatever was declared: a refusal of one combination, not a blanket one.
    check_mvs_hessian_declaration(params, False)
    check_mvs_hessian_declaration(MvsBootstrapParams.disabled(), True)


def test_a_disabled_mvs_bootstrap_changes_nothing_either() raises:
    """The MVS twin of the disabled-bundle assertion below.

    `refresh_mvs_bootstrap` with a disabled bundle and no user weights must
    produce an *empty* vector, not a vector of 1.0s. A vector of ones would
    be refused by the weights exclusion and would cost an unbootstrapped,
    unweighted fit its specialization for nothing -- the same trap the
    Bayesian bundle documents, pinned here for MVS because MVS is the one
    that ships as CatBoost's actual default and so is the one most likely to
    be wired up in a hurry.
    """
    var goss = GossParams.disabled()
    var off = MvsBootstrapParams.disabled()
    var gradients = List[Float64]()
    for i in range(48):
        gradients.append(Float64(i + 1) * 0.25)
    var weights = List[Float64]()
    var audit = MvsAudit.empty()
    refresh_mvs_bootstrap(
        weights, audit, off, gradients, _unweighted(), 48, 0, 1.0
    )
    assert_equal(len(weights), 0)
    for objective in _all_objectives():
        assert_equal(
            round_has_constant_hessian(objective, weights, goss),
            _is_constant(objective),
        )


def test_a_disabled_bootstrap_changes_nothing_the_predicate_answers() raises:
    """The default bundle must leave every answer in this file alone.

    `refresh_bayesian_bootstrap` with a disabled bundle and no user weights
    produces an *empty* vector rather than a vector of 1.0s, deliberately: a
    vector of ones would be refused by the weights exclusion and would cost an
    unbootstrapped, unweighted fit its specialization for nothing. This is the
    assertion that pins that choice, because it is the one place the default
    path could have been moved by accident.
    """
    var goss = GossParams.disabled()
    var off = BayesianBootstrapParams.disabled()
    var weights = List[Float64]()
    refresh_bayesian_bootstrap(weights, off, _unweighted(), 48, 0)
    assert_equal(len(weights), 0)
    for objective in _all_objectives():
        assert_equal(
            round_has_constant_hessian(objective, weights, goss),
            _is_constant(objective),
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
