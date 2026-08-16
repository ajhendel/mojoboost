"""Reach for the CatBoost mechanisms, and the standing gate that says which
of them are still unreachable.

**Why this file exists rather than more cases in `test_wired_features.py`.**
Five mechanisms in a row were merged in Mojo, tested in Mojo, and set by no
binding: CatBoost mode itself, oblivious trees, `float64` on the sparse and
distributed paths, `float64` on the GPU, and the whole CatBoost parameter
group. Every one of them passed its own native suite. What none of them had
was a test that ran a *fit through Python* and demanded the answer move.

So every wired case here follows one shape: fit the default arm, fit the arm
with the knob set, and assert the predictions differ. A knob that validates
but does not reach the trainer passes a validation test and fails this one,
which is the only difference that matters.

`test_unreachable_catboost_parameters_refuse_by_name` is the other half and
is the part meant to fail loudly later. It pins the four names that are still
refused. When someone wires one, this test fails, and fixing it means moving
the name into the wired list above -- which is a smaller, louder step than
noticing on your own that a merged mechanism was never connected.
"""

import numpy as np
import pytest

from mojotrees import MojoTreesClassifier, MojoTreesRegressor


def _regression(n_rows=400, n_features=5, seed=11):
    gen = np.random.default_rng(seed)
    X = gen.random((n_rows, n_features))
    y = (
        3.0 * X[:, 0]
        + 2.0 * X[:, 1] * X[:, 2]
        - 1.5 * X[:, 3]
        + 0.10 * gen.standard_normal(n_rows)
    )
    return X, y


def _binary(n_rows=400, n_features=5, seed=13):
    X, y = _regression(n_rows, n_features, seed)
    return X, (y > np.median(y)).astype(np.int64)


# --------------------------------------------------------------------------
# leaf_estimation_iterations
# --------------------------------------------------------------------------


def test_leaf_estimation_iterations_moves_the_leaves():
    """Extra Newton steps change leaf values and nothing else.

    The difference is real rather than noise by construction, not by
    threshold. Both arms are the same estimator with the same seed on the
    same data, so every draw -- feature subsample, bagging, binning -- is
    identical and the two fits are deterministic; `MOJOTREES_NUM_WORKERS` is
    contractually not allowed to move either one. The *structure* is
    identical too: `ExtraTreeParams.is_active` deliberately excludes
    `leaf_estimation_iterations`, so the split search takes the same path and
    elects the same splits, and the only thing that can differ is what
    `boosting._estimate_leaf_values` wrote into `Tree.value`. So a difference
    here has exactly one possible cause.

    The direction is checked too. A logistic objective's leaf value from one
    Newton step is an underestimate of the leaf's own minimizer (the
    quadratic is fitted at the raw score the leaf is about to leave), so
    extra steps make the raw scores *more* extreme. A wiring that reached the
    trainer but scaled the step wrongly could still differ from the default;
    it would not systematically increase |raw|.
    """
    X, y = _binary()
    one = MojoTreesClassifier(
        n_estimators=8, num_leaves=7, random_state=5
    ).fit(X, y)
    many = MojoTreesClassifier(
        n_estimators=8,
        num_leaves=7,
        random_state=5,
        leaf_estimation_iterations=6,
    ).fit(X, y)
    raw_one = one.predict(X, raw_score=True)
    raw_many = many.predict(X, raw_score=True)
    assert not np.array_equal(raw_one, raw_many)
    assert np.abs(raw_many).mean() > np.abs(raw_one).mean()


def test_leaf_estimation_iterations_of_one_is_the_default_bit_for_bit():
    """1 and "absent" are the same code path, so they are the same bits.

    `_estimate_leaf_values` returns before it reads a row at 1, which is the
    whole of that guarantee; this is the assertion that would catch a wiring
    that made the default arm take a second route to its first value.
    """
    X, y = _regression()
    absent = MojoTreesRegressor(n_estimators=6, random_state=5).fit(X, y)
    named = MojoTreesRegressor(
        n_estimators=6, random_state=5, leaf_estimation_iterations=1
    ).fit(X, y)
    np.testing.assert_array_equal(absent.predict(X), named.predict(X))


def test_squared_error_barely_moves_and_that_is_the_arithmetic():
    """Under squared error the extra steps are a no-op up to the width the
    histogram was accumulated at, and this test is the record of why.

    Squared error's loss is *exactly* the quadratic a Newton step minimizes.
    With `g_r = raw_r - y_r`, `h_r = 1` and `lambda_l2 = 0` (this library's
    stock value since 2026-08-16), the first step writes `v = -sum(g)/n`;
    evaluating the leaf again at `raw + v` gives `sum(g') = sum(g) + n*v =
    0`, so every later step should add exactly zero. It adds a little more
    than zero because the two sums are not taken at the same width: the
    grower's `v` comes off the histogram, whose derivative planes are
    `float32` by default, while `_estimate_leaf_values` refills the leaf's
    rows in `float64`. The residual is that mismatch and nothing else.

    Measured on this fixture: the squared-error arms differ by 3.2e-09 on
    predictions of mean magnitude 1.21 (relative 2.6e-09, inside `float32`
    epsilon of 1.2e-07), while the logistic arms differ by 3.21 on
    predictions of mean magnitude 0.98. The assertion is the *ratio*, not
    either number, so it does not pin a tolerance that a later change to the
    accumulator width would falsify for the wrong reason.
    """
    X, y = _regression()
    one = MojoTreesRegressor(n_estimators=8, random_state=5).fit(X, y)
    many = MojoTreesRegressor(
        n_estimators=8, random_state=5, leaf_estimation_iterations=5
    ).fit(X, y)
    moved = np.abs(one.predict(X) - many.predict(X)).max()
    scale = np.abs(one.predict(X)).mean()
    assert moved / scale < 1e-6

    Xb, yb = _binary()
    lo = MojoTreesClassifier(n_estimators=8, random_state=5).fit(Xb, yb)
    hi = MojoTreesClassifier(
        n_estimators=8, random_state=5, leaf_estimation_iterations=5
    ).fit(Xb, yb)
    logit_moved = np.abs(
        lo.predict(Xb, raw_score=True) - hi.predict(Xb, raw_score=True)
    ).max()
    logit_scale = np.abs(lo.predict(Xb, raw_score=True)).mean()
    # Six orders of magnitude between "the accumulator is float32" and "the
    # mechanism ran". The measured gap is nine.
    assert logit_moved / logit_scale > 1e6 * (moved / scale)


def test_leaf_estimation_iterations_refuses_an_eval_set_fit():
    """eval_set refuses it, and the reason is worth stating because the
    first draft of this lane got it wrong.

    `boosting.train_with_valid` does implement the extra Newton steps. The
    eval_set path does not reach it: `MojoTreesRegressor.fit(eval_set=...)`
    calls `_mojotrees.fit_with_metrics`, which is
    `custom_metric.fit_with_metrics`, a different round loop that reads
    `TreeParams.extra.leaf_estimation_iterations` nowhere. This lane opted
    that entry point in on the strength of the name, the arms came back
    bit-identical, and the opt-in was removed. That is exactly the failure
    the file's header describes, caught by the shape the file insists on.
    """
    X, y = _binary()
    with pytest.raises(Exception, match="fit_with_metrics"):
        MojoTreesClassifier(
            n_estimators=8, leaf_estimation_iterations=5
        ).fit(X, y, eval_set=[(X, y)])


def test_leaf_estimation_iterations_refuses_the_trainers_that_drop_it():
    """A multiclass fit names the entry point instead of dropping the
    setting. This is the assertion that distinguishes "wired" from
    "accepted": before this lane, the estimator refused every value above 1
    everywhere, which also passed a refusal test.

    `Exception` rather than `ValueError`: the refusal is the native layer's,
    and a Mojo `Error` crosses the boundary as a plain `Exception`. Catching
    `ValueError` here would pass on the old Python-side refusal and fail on
    the wired one, which is backwards.
    """
    X, y = _regression()
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    with pytest.raises(Exception, match="leaf_estimation_iterations"):
        MojoTreesClassifier(
            n_estimators=4, leaf_estimation_iterations=3
        ).fit(X, labels)


def test_leaf_estimation_iterations_refuses_the_renewing_objectives():
    """`l1` replaces its leaf with the weighted median of the residuals,
    which is the exact minimizer, so a Newton step would walk away from it.
    The native check is `boosting._check_leaf_estimation_config`; reaching it
    from Python is what this asserts."""
    X, y = _regression()
    with pytest.raises(Exception, match="renews its leaves"):
        MojoTreesRegressor(
            n_estimators=4, objective="l1", leaf_estimation_iterations=3
        ).fit(X, y)


# --------------------------------------------------------------------------
# boosting_type="ordered"
# --------------------------------------------------------------------------


def test_ordered_boosting_moves_the_fit():
    """CatBoost's ordered boosting changes the model, and by more than a
    reordering.

    Same seed, same data, same estimator otherwise, both fits deterministic,
    so the difference is the mechanism. It is a large difference rather than
    a marginal one and that is the point: ordered boosting hands the gradient
    fill a *different raw score per row* (the score held by the tightest rung
    of the fold ladder that never saw that row), so the split search sees
    different gradients from round one and the trees themselves differ, not
    only the leaf values. A wiring that built the bundle and dropped it would
    reproduce the plain fit exactly, which is what the first assertion
    rejects.
    """
    X, y = _regression()
    plain = MojoTreesRegressor(
        n_estimators=10, num_leaves=7, random_state=5
    ).fit(X, y)
    ordered = MojoTreesRegressor(
        n_estimators=10,
        num_leaves=7,
        random_state=5,
        boosting_type="ordered",
    ).fit(X, y)
    plain_pred = plain.predict(X)
    ordered_pred = ordered.predict(X)
    assert not np.array_equal(plain_pred, ordered_pred)
    # Not a rounding difference: the ladder changes which splits are elected.
    assert np.abs(plain_pred - ordered_pred).max() > 1e-6


def test_ordered_boosting_is_reproducible_and_the_seed_is_read():
    """The permutation is a pure function of `ordered_seed`, so one seed
    repeats and two seeds differ. Two fits at one seed being equal is what
    rules out "the difference above was an unseeded draw"; two seeds
    differing is what rules out "the seed key is not read"."""
    X, y = _regression()

    def fit(seed):
        return MojoTreesRegressor(
            n_estimators=8,
            num_leaves=7,
            random_state=5,
            boosting_type="ordered",
            ordered_seed=seed,
        ).fit(X, y).predict(X)

    np.testing.assert_array_equal(fit(1), fit(1))
    assert not np.array_equal(fit(1), fit(2))


def test_ordered_boosting_knobs_are_read():
    """`fold_len_multiplier` and `permutation_count` reach the ladder.

    The multiplier sets how fast the rungs grow, so it decides which rung
    each row reads and therefore which raw score its gradient is taken at;
    `permutation_count` decides how many permutations a round draws from.
    Both are checked against the same `ordered_seed`, so the only thing that
    moved is the knob.
    """
    X, y = _regression()

    def fit(**kw):
        return MojoTreesRegressor(
            n_estimators=8,
            num_leaves=7,
            random_state=5,
            boosting_type="ordered",
            **kw,
        ).fit(X, y).predict(X)

    base = fit()
    assert not np.array_equal(base, fit(fold_len_multiplier=4.0))
    assert not np.array_equal(base, fit(permutation_count=3))


def test_has_time_is_one_block():
    """`has_time=True` is the identity permutation, so it is reproducible and
    is not the shuffled fit. It also forces `permutation_count=1`, and
    naming a different count beside it raises rather than being overridden.
    """
    X, y = _regression()

    def fit(**kw):
        return MojoTreesRegressor(
            n_estimators=8,
            num_leaves=7,
            random_state=5,
            boosting_type="ordered",
            **kw,
        ).fit(X, y).predict(X)

    timed = fit(has_time=True)
    np.testing.assert_array_equal(timed, fit(has_time=True))
    assert not np.array_equal(timed, fit())
    with pytest.raises(ValueError, match="has_time"):
        fit(has_time=True, permutation_count=4)


def test_ordered_boosting_refuses_what_it_cannot_compose_with():
    """Every path that would train a plain ensemble and call it ordered."""
    X, y = _regression()
    labels = (y > np.median(y)).astype(np.int64)
    multi = np.digitize(y, np.quantile(y, [0.33, 0.66]))

    with pytest.raises(ValueError, match="row bagging"):
        MojoTreesRegressor(
            boosting_type="ordered", subsample=0.7, subsample_freq=1
        ).fit(X, y)
    with pytest.raises(ValueError, match="eval_set"):
        MojoTreesRegressor(n_estimators=4, boosting_type="ordered").fit(
            X, y, eval_set=[(X, y)]
        )
    with pytest.raises(ValueError, match="multiclass"):
        MojoTreesClassifier(n_estimators=4, boosting_type="ordered").fit(
            X, multi
        )
    assert labels.sum() > 0


def test_ordered_knobs_without_ordered_raise():
    """A knob that would be parsed and never read is reported, not dropped.
    This is the same rule `params.mojo` applies to the string surface."""
    X, y = _regression()
    with pytest.raises(ValueError, match="fold_len_multiplier"):
        MojoTreesRegressor(fold_len_multiplier=3.0).fit(X, y)


# --------------------------------------------------------------------------
# score_function
# --------------------------------------------------------------------------
#
# UNRUN. Written in a session where running anything was forbidden; no arm
# of any test in this section has been executed. The claims below are read
# off the source, not off a run.


def test_score_function_cosine_moves_the_fit():
    """`score_function="Cosine"` elects different splits, at the only
    setting where it can.

    **`reg_lambda` is not decoration here, it is the whole test.** Cosine's
    numerator is the L2 sum, and at `reg_lambda=0` its denominator collapses
    onto the same expression, so it degenerates to `sqrt` of the L2 score;
    `sqrt` is strictly increasing, so the argmax cannot move and the two arms
    would be the same tree (docs/design/CATBOOST_CATALOG.md A10 section 3).
    A moves-the-fit test written at the estimator's default `reg_lambda=0`
    would therefore fail against a *correct* wiring, and passing it would
    mean the arithmetic was wrong. Both arms set `reg_lambda=3.0`, which is
    what the CatBoost-mode comparisons in this repository use and is off
    that degenerate point.

    Everything else is held equal: same estimator, same seed, same data, so
    every draw is identical and both fits are deterministic.
    """
    X, y = _regression()
    l2 = MojoTreesRegressor(
        n_estimators=8, num_leaves=15, reg_lambda=3.0, random_state=5
    ).fit(X, y)
    cosine = MojoTreesRegressor(
        n_estimators=8,
        num_leaves=15,
        reg_lambda=3.0,
        random_state=5,
        score_function="Cosine",
    ).fit(X, y)
    assert not np.array_equal(l2.predict(X), cosine.predict(X))


def test_score_function_l2_is_the_default_bit_for_bit():
    """Naming the default must not move a fit. `SCORE_L2` is the field's
    default and `split.find_best_split`'s, so "absent" and `"L2"` are the
    same integer down the same path."""
    X, y = _regression()
    absent = MojoTreesRegressor(
        n_estimators=6, reg_lambda=3.0, random_state=5
    ).fit(X, y)
    named = MojoTreesRegressor(
        n_estimators=6, reg_lambda=3.0, random_state=5, score_function="L2"
    ).fit(X, y)
    np.testing.assert_array_equal(absent.predict(X), named.predict(X))


@pytest.mark.parametrize("spelling", ["cosine", "COSINE", "CoSiNe"])
def test_score_function_value_is_case_insensitive(spelling):
    """docs/PARAMETER_NAMING.md makes value strings case insensitive, and
    the fold happens once, in the estimator, before the native
    `parse_score_function` (which takes canonical lowercase, as
    `parse_device` and `canonical_bootstrap_type` do). Every spelling must
    therefore reach the same code and produce the same fit."""
    X, y = _regression(n_rows=120)
    canonical = MojoTreesRegressor(
        n_estimators=4,
        reg_lambda=3.0,
        random_state=5,
        score_function="Cosine",
    ).fit(X, y)
    other = MojoTreesRegressor(
        n_estimators=4,
        reg_lambda=3.0,
        random_state=5,
        score_function=spelling,
    ).fit(X, y)
    np.testing.assert_array_equal(canonical.predict(X), other.predict(X))


def test_unknown_score_function_is_refused_by_name():
    """An unknown value is refused rather than resolved to the default: a
    mistake that resolved to L2 would run one arm of an A/B under the
    other's label."""
    X, y = _regression(n_rows=120)
    with pytest.raises(ValueError, match="score_function"):
        MojoTreesRegressor(n_estimators=3, score_function="NewtonCosine").fit(
            X, y
        )


# --------------------------------------------------------------------------
# bootstrap_type
# --------------------------------------------------------------------------
#
# UNRUN. Written in a session where running Python tests was forbidden; no
# arm of any test in this section has been executed. The claims below are
# read off the source, not off a run.


def test_bootstrap_type_mvs_moves_the_fit():
    """`bootstrap_type="MVS"` is CatBoost's own CPU default sampler and it
    must reach the trainer, not merely validate.

    MVS drops rows outright: a zero weight is a dropped row, not a
    down-weighted one (`sampling.mvs_kept_rows`,
    `SetControlNoZeroWeighted`), so a tree grown under it sees a different
    row set and a different weighted gradient. Both arms are the same
    estimator, the same seed and the same data, so the only thing that can
    differ is the draw.
    """
    X, y = _regression()
    plain = MojoTreesRegressor(
        n_estimators=8, num_leaves=15, random_state=5
    ).fit(X, y)
    mvs = MojoTreesRegressor(
        n_estimators=8,
        num_leaves=15,
        random_state=5,
        bootstrap_type="MVS",
        subsample=0.8,
    ).fit(X, y)
    assert not np.array_equal(plain.predict(X), mvs.predict(X))


def test_bootstrap_type_bayesian_moves_the_fit():
    """`bootstrap_type="Bayesian"` with a temperature reaches the same
    argument by the other arm of `_parse_bootstrap`. It keeps every row and
    reweights it, so the fit moves without any row being dropped."""
    X, y = _regression()
    plain = MojoTreesRegressor(
        n_estimators=8, num_leaves=15, random_state=5
    ).fit(X, y)
    bayesian = MojoTreesRegressor(
        n_estimators=8,
        num_leaves=15,
        random_state=5,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
    ).fit(X, y)
    assert not np.array_equal(plain.predict(X), bayesian.predict(X))


def test_bootstrap_type_mvs_repeats_under_the_same_seed():
    """The MVS draw is counter-based and keyed on `(seed, tree, row)` with
    its own domain constant (`sampling._mvs_stream`, `_MVS_DOMAIN`), so two
    fits at the same `random_state` are the same model. A sequential draw
    whose order depended on thread scheduling would fail this under a
    different `MOJOTREES_NUM_WORKERS`, which is why the property is a
    contract and not a hope."""
    X, y = _regression()
    kwargs = dict(
        n_estimators=6,
        num_leaves=15,
        random_state=5,
        bootstrap_type="MVS",
        subsample=0.8,
    )
    first = MojoTreesRegressor(**kwargs).fit(X, y)
    second = MojoTreesRegressor(**kwargs).fit(X, y)
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


@pytest.mark.parametrize("spelling", ["mvs", "MVS", "MvS"])
def test_bootstrap_type_value_is_case_insensitive(spelling):
    """docs/PARAMETER_NAMING.md makes value strings case insensitive and the
    fold happens once, in the estimator, before the native
    `canonical_bootstrap_type`, which takes canonical lowercase."""
    X, y = _regression()
    canonical = MojoTreesRegressor(
        n_estimators=4, random_state=5, bootstrap_type="MVS", subsample=0.8
    ).fit(X, y)
    other = MojoTreesRegressor(
        n_estimators=4, random_state=5, bootstrap_type=spelling, subsample=0.8
    ).fit(X, y)
    np.testing.assert_array_equal(canonical.predict(X), other.predict(X))


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        # `bagging_temperature` belongs to Bayesian and MVS never reads it.
        # CatBoost accepts this pair, ignores the temperature and drops it in
        # Save; mojotrees refuses it (sampling.check_mvs_bagging_temperature).
        (
            {"bootstrap_type": "MVS", "bagging_temperature": 1.0},
            "bagging_temperature",
        ),
        # The Bayesian bootstrap keeps every row, so there is no fraction.
        # CatBoost refuses the same pair.
        ({"bootstrap_type": "Bayesian", "subsample": 0.8}, "subsample"),
        # Two bootstrap types at once: bagging_fraction IS Bernoulli.
        (
            {"bootstrap_type": "MVS", "bagging_fraction": 0.8},
            "bagging_fraction",
        ),
        # A bagging schedule beside a per-tree sampler.
        (
            {"bootstrap_type": "MVS", "subsample": 0.8, "subsample_freq": 1},
            "subsample_freq",
        ),
        # The temperature on its own names nothing.
        ({"bagging_temperature": 1.0}, "bagging_temperature"),
        # Still genuinely unreachable, and refused by name rather than as an
        # unknown value.
        ({"bootstrap_type": "Bernoulli"}, "Bernoulli"),
        ({"bootstrap_type": "Poisson"}, "Poisson"),
        ({"bootstrap_type": "Uniform"}, "bootstrap_type"),
    ],
)
def test_a_bootstrap_configuration_that_names_two_things_is_refused(
    kwargs, fragment
):
    """A parameter that belongs to another `bootstrap_type` is refused by
    name with the reason, never accepted and dropped. CatBoost's own
    `TBootstrapConfig::Validate` misses one of these
    (`bagging_temperature` beside MVS) and that divergence is deliberate."""
    X, y = _regression(n_rows=120)
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        MojoTreesRegressor(n_estimators=3, **kwargs).fit(X, y)
    assert fragment in str(excinfo.value)


def test_bootstrap_type_refuses_on_a_multiclass_fit():
    """Multiclass has no bootstrap and must say so. Neither
    `boosting.train_multiclass` nor `train_multiclass_gpu` takes the bundle,
    and CatBoost agrees about the shape of the hole: its own defaulting
    block excludes the multiclass-only losses from the MVS default."""
    X, y = _regression(n_rows=180)
    labels = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    with pytest.raises((ValueError, RuntimeError)):
        MojoTreesClassifier(
            n_estimators=3, bootstrap_type="MVS", subsample=0.8
        ).fit(X, labels)


def test_bootstrap_type_no_is_bit_for_bit_the_default():
    """Naming the default must not move a fit. `"No"` is
    `BootstrapParams.disabled()`, which is the bundle every fit already
    carries, so "absent" and `"No"` are the same object down the same
    path."""
    X, y = _regression()
    absent = MojoTreesRegressor(n_estimators=6, random_state=5).fit(X, y)
    named = MojoTreesRegressor(
        n_estimators=6, random_state=5, bootstrap_type="No"
    ).fit(X, y)
    np.testing.assert_array_equal(absent.predict(X), named.predict(X))


# --------------------------------------------------------------------------
# The standing gate
# --------------------------------------------------------------------------


#: Every CatBoost parameter this estimator still cannot honor, with the
#: value that triggers the refusal and the piece the message must name.
#: A name leaves this table by being wired, never by being deleted.
UNREACHABLE = [
    ("random_strength", 1.0, "random_score_scale"),
    ("max_ctr_complexity", 2, "ctr.mojo"),
]
# `score_function` left this table by being wired, which is the only way a
# row may leave it. Its moves-the-fit tests are
# `test_score_function_cosine_moves_the_fit` and the two beside it above.
#
# `bagging_temperature` left it the same way on 2026-08-16 and did NOT stop
# raising: set on its own it still refuses, because the temperature belongs
# to `bootstrap_type="Bayesian"` and naming it alone names nothing. What
# changed is that the refusal is now about a missing companion parameter
# rather than about a missing edge, so the row's premise was false. Its
# moves-the-fit tests are the `bootstrap_type` section below.


@pytest.mark.parametrize("name,value,missing", UNREACHABLE)
def test_unreachable_catboost_parameters_refuse_by_name(name, value, missing):
    """The refusal fires, and the message names the missing piece.

    Naming the piece is not decoration. Each of these four is *implemented*
    in Mojo and unreachable for a different structural reason, and a message
    that said only "not implemented" is what let four of them sit merged and
    unused through a whole round. The `missing` string is the symbol a reader
    can grep for to find the one edge that is absent.

    **When this test fails because a refusal stopped firing, do not delete
    the row.** Move it up into a moves-the-fit test like the ones above. A
    parameter that stops raising and gains no such test is exactly the state
    this file exists to prevent.
    """
    X, y = _regression(n_rows=120)
    with pytest.raises(ValueError) as excinfo:
        MojoTreesRegressor(n_estimators=3, **{name: value}).fit(X, y)
    assert missing in str(excinfo.value)


def test_the_values_that_name_current_behavior_are_accepted():
    """`score_function="L2"` and `bootstrap_type="No"` state what a fit
    already does, so honoring them costs nothing and they must not raise."""
    X, y = _regression(n_rows=120)
    MojoTreesRegressor(
        n_estimators=3,
        score_function="L2",
        bootstrap_type="No",
        random_strength=0.0,
        leaf_estimation_iterations=1,
    ).fit(X, y)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
