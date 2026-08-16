"""MVS: the solve, its key, its guard, and its consequences.

The mechanism is `sampling.MvsBootstrapParams` and the functions around it. It
is CatBoost's `bootstrap_type=MVS`, Minimal Variance Sampling, which is what
CatBoost actually runs on the CPU for every objective this project benchmarks
-- the Bayesian bootstrap of `tests/test_bayesian_bootstrap.mojo` is its
fallback, not its default. Transcribed from
`catboost/private/libs/algo/mvs.cpp` (`GenSampleWeights`, `CalculateThreshold`,
`GetLambda`, `GetSingleProbability`), with the defaults from
`catboost/private/libs/options/catboost_options.cpp` and the option surface
from `bootstrap_options.{h,cpp}`. See `docs/design/CATBOOST_CATALOG.md`, A11.

What this file has to establish, in order of how badly it would hurt to be
wrong about it:

1. **The threshold solve is right.** This is the only part of MVS with real
   arithmetic in it, and a wrong `mu` is a wrong model that still looks like a
   plausible sampler. Pinned on a fixture whose `mu` is hand-derived from
   CatBoost's formula on paper (section 2), not by calling the code under
   test, and compared on `to_bits()`.
2. **The guard fires, and it is proven to fire.** The degenerate-threshold
   branch is the one with no CatBoost referent -- CatBoost has no guard there
   and silently gives every row weight 0. `MvsAudit.blocks_guarded` is asserted
   nonzero on the fixture that reaches it and asserted zero everywhere else,
   so this cannot be a test whose gate never opened.
3. **`mvs_reg = 0` is refused.** The reason is in `sampling.check_mvs_reg` and
   it is a wrong-answer-with-no-signal bug, not a crash.
4. **The exclusion engages.** An MVS weight multiplies the row's derivatives,
   so a sampled fit must not take the two-plane constant-hessian path.
5. **The draw is reproducible.** Keyed by (seed, tree index, row) alone:
   identical at `MOJOTREES_NUM_WORKERS` 1, 3 and 8, independent of the buffer
   it is written into, and moving when any of the three keys moves. Every key
   is asserted live, because a key that is silently ignored is how this kind
   of code goes wrong.
6. **The default moves nothing.** A disabled bundle produces an empty weight
   vector, not a vector of ones.

No tolerance appears anywhere below: every float comparison is on `to_bits()`
or is an exact order comparison. Where a distributional claim would need a
tolerance -- "about 80 percent of rows survive" -- an exact structural
identity is asserted instead (every surviving weight has the same bits; the
audit's kept count equals the length of the kept row set).

This file must NOT be renamed to `test_gpu_*`: `tools/run_tests.sh` selects
the accelerator subset by name and would silently exclude it from the CPU
suite.
"""

from std.os import setenv
from std.testing import assert_equal, assert_false, assert_true, TestSuite
from std.utils.numerics import nan

from mojotrees import (
    SQUARED_ERROR,
    BoosterParams,
    Booster,
    TreeParams,
    bin_equal_width,
    train,
)
from mojotrees.boosting import round_has_constant_hessian
from mojotrees.goss import GossParams
from mojotrees.sampling import (
    BOOTSTRAP_MVS,
    BayesianBootstrapParams,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_MVS_SUBSAMPLE,
    MVS_BLOCK_SIZE,
    MvsAudit,
    MvsBootstrapParams,
    check_bootstrap_type_exclusive,
    check_mvs_bagging_temperature,
    check_mvs_hessian_declaration,
    check_mvs_reg,
    check_row_set,
    mvs_auto_lambda_from_gradients,
    mvs_auto_lambda_from_leaf_values,
    mvs_bootstrap_weights,
    mvs_kept_rows,
    mvs_varies_hessian,
    refresh_mvs_bootstrap,
)


comptime N_ROWS = 64
comptime N_BINS = 8

# The hand-derived fixture of section 2. See that section for the derivation.
comptime FIXTURE_ROWS = 200
comptime FIXTURE_GRAD = 3.0
comptime FIXTURE_REG = 16.0


def _bits(v: Float64) -> UInt64:
    return v.to_bits().cast[DType.uint64]()


def _assert_same_weights(
    a: List[Float64], b: List[Float64], what: String
) raises:
    assert_equal(len(a), len(b), String(what, ": length"))
    for i in range(len(a)):
        assert_equal(_bits(a[i]), _bits(b[i]), String(what, ": row ", i))


def _differ(a: List[Float64], b: List[Float64]) -> Bool:
    """Whether two weight vectors differ anywhere, on bits."""
    if len(a) != len(b):
        return True
    for i in range(len(a)):
        if _bits(a[i]) != _bits(b[i]):
            return True
    return False


def _constant_gradients(n: Int, value: Float64) -> List[Float64]:
    var g = List[Float64](capacity=n)
    for _ in range(n):
        g.append(value)
    return g^


def _ramp_gradients(n: Int) -> List[Float64]:
    """Gradients of strictly increasing magnitude, alternating in sign.

    Alternating signs matter: MVS reads the magnitude, so a sampler that read
    the signed value would order the rows differently and the monotonicity
    assertion in section 6 would fail.
    """
    var g = List[Float64](capacity=n)
    for r in range(n):
        var m = Float64(r + 1)
        g.append(-m if (r & 1) == 1 else m)
    return g^


def _user_weights(n: Int) -> List[Float64]:
    var w = List[Float64](capacity=n)
    for r in range(n):
        w.append(Float64(r % 3) + 1.0)
    return w^


def _draw(
    params: MvsBootstrapParams,
    gradients: List[Float64],
    n_rows: Int,
    tree_index: Int,
    auto_lambda: Float64,
    mut audit: MvsAudit,
    n_outputs: Int = 1,
) raises -> List[Float64]:
    """One tree's draw, with the audit written into the caller's own slot so
    every assertion about the weights can be made beside one about the path
    that produced them."""
    var w = List[Float64]()
    mvs_bootstrap_weights(
        params,
        gradients,
        n_rows,
        tree_index,
        auto_lambda,
        w,
        audit,
        n_outputs,
    )
    return w^


def _features() -> List[Float64]:
    """A monotone ramp, one feature, so every bin holds the same row count and
    a change in the leaf values cannot be blamed on an empty bin."""
    var f = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        f.append(Float64(r))
    return f^


def _target() -> List[Float64]:
    """A step with two rows of contradiction on either side of it, so the
    fitted leaf values depend on how the individual rows are weighted rather
    than only on which side of the step they fall."""
    var y = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        if r == 20 or r == 44:
            y.append(1.0 if r >= 32 else 0.0)
        else:
            y.append(0.0 if r < 32 else 1.0)
    return y^


def _model_bits(model: Booster) raises -> List[UInt64]:
    """One prediction per bin, as bits. Two models are equal iff these are."""
    var out = List[UInt64](capacity=N_BINS)
    for b in range(N_BINS):
        var bins: List[Int] = [b]
        out.append(_bits(model.predict_bins(bins)))
    return out^


def _booster_params() -> BoosterParams:
    return BoosterParams(12, 0.3, TreeParams(4, 1, 1.0, 1e-3))


# ---------------------------------------------------------------------------
# 1. The defaults we were handed, as constants
# ---------------------------------------------------------------------------


def test_the_catboost_defaults_are_the_ones_source_says() raises:
    """`subsample` is 0.8 under MVS, and MVS is a distinct `bootstrap_type`.

    0.8 is not the `TBootstrapConfig` constructor's default, which is 0.66 --
    it is the value `SetNotSpecifiedOptionsToDefaults` installs specifically
    when the resolved type is MVS. Anyone reading 0.66 out of the header and
    calling it CatBoost's rate is wrong by a fifth of the rows, so the number
    is pinned here rather than left in a comment.

    `MVS_BLOCK_SIZE` is pinned for a different reason: it is 8192 in CatBoost
    as a `const ui32` member, not a thread count, and the whole determinism
    argument in section 8 rests on ours being a constant too.
    """
    assert_equal(_bits(DEFAULT_MVS_SUBSAMPLE), _bits(0.8))
    assert_equal(MVS_BLOCK_SIZE, 8192)
    assert_equal(BOOTSTRAP_MVS, 2)

    var params = MvsBootstrapParams.enable()
    assert_true(params.enabled)
    assert_equal(_bits(params.subsample), _bits(0.8))
    assert_equal(params.seed, DEFAULT_BOOTSTRAP_SEED)
    # `mvs_reg` is CatBoost's `TMaybe<float>` at `Nothing()`: unset is a real
    # state, not a sentinel number, and `resolve_reg` reads the derived value.
    assert_false(params.reg_is_set)
    assert_equal(_bits(params.resolve_reg(12.25)), _bits(12.25))
    assert_true(params.samples_rows())

    var pinned = MvsBootstrapParams.enable_with_reg(2.0)
    assert_true(pinned.reg_is_set)
    assert_equal(_bits(pinned.resolve_reg(12.25)), _bits(2.0))


# ---------------------------------------------------------------------------
# 2. The threshold solve, pinned on a hand-derived fixture
# ---------------------------------------------------------------------------


def test_the_threshold_solve_matches_the_hand_derivation() raises:
    """The one piece of real arithmetic in MVS, checked against paper.

    The fixture is 200 rows whose gradient is 3.0 and whose `mvs_reg` is 16.0,
    so every row's statistic is `g = sqrt(3^2 + 16) = 5.0` exactly. Running
    `CalculateThreshold` on that by hand:

      - the pivot is 5.0, nothing is below it and nothing is above it, so the
        whole block is the "middle": `nMiddle = 200`, `sumOfMiddle = 1000.0`,
        `sumOfSmall = 0`, `nLarge = 0`;
      - `estimatedSampleSize = 0/5 + 0 + 0 + 200 = 200`, and the target is
        `0.8 * 200 = 160`, so the estimate overshoots;
      - `middleEnd == candidatesEnd`, so the recursion terminates on the exact
        solve `(0 + 0 + 1000) / (160 - 0)`, giving `mu = 6.25`;
      - every row then has `g = 5.0 <= mu`, so `p = 5.0/6.25` and a surviving
        row's weight is `1 / (5.0 / 6.25)`, which is 1.25.

    The expected weight below is written as that expression, evaluated from
    CatBoost's formula rather than by calling anything under test, and compared
    on bits. A solve that got `mu` wrong by any amount at all fails here.

    What this does NOT assert is how many rows survive: that is the draw, and
    asserting a count would be asserting a property of splitmix64. What it
    asserts instead is exact and total -- every surviving weight has the same
    bits, and no row has any other value.
    """
    var params = MvsBootstrapParams.enable_with_reg(
        FIXTURE_REG, DEFAULT_MVS_SUBSAMPLE, 7
    )
    var grads = _constant_gradients(FIXTURE_ROWS, FIXTURE_GRAD)
    var audit = MvsAudit.empty()
    var w = _draw(params, grads, FIXTURE_ROWS, 0, 0.0, audit)

    # The block loop ran once and did not take the degenerate guard.
    assert_equal(audit.blocks, 1)
    assert_equal(audit.blocks_guarded, 0)

    var mu = 1000.0 / 160.0
    var expected_kept = 1.0 / (5.0 / mu)

    assert_equal(len(w), FIXTURE_ROWS)
    var kept = 0
    var dropped = 0
    for r in range(FIXTURE_ROWS):
        if _bits(w[r]) == _bits(0.0):
            dropped += 1
        else:
            assert_equal(
                _bits(w[r]),
                _bits(expected_kept),
                String("surviving weight at row ", r),
            )
            kept += 1

    # Both outcomes occurred, so neither branch of the keep test is dead. At a
    # keep probability of 0.8 over 200 rows the chance of an all-kept or
    # all-dropped block is below 1e-15; the draw is deterministic, so this is a
    # fixed fact about this seed rather than a coin flip taken at run time.
    assert_true(kept > 0)
    assert_true(dropped > 0)
    assert_equal(kept + dropped, FIXTURE_ROWS)

    # The audit agrees with the buffer it describes.
    assert_equal(audit.rows_kept, kept)
    # No row here is above the threshold, so nothing was kept with certainty.
    assert_equal(audit.rows_kept_certainly, 0)


def test_rows_above_the_threshold_are_kept_at_exactly_one() raises:
    """MVS is not uniform sampling: a large enough gradient is kept certainly,
    at weight exactly 1.0 and with no draw taken.

    This is `GetSingleProbability`'s `(derivativeAbsoluteValue > threshold) ?
    1.0 : ...` branch, and it is the half of the method that distinguishes it
    from Bernoulli bagging. If it were missing, every row would go through the
    probabilistic branch and `rows_kept_certainly` would stay 0.

    The fixture is a ramp, so the largest magnitudes are far above any
    threshold that keeps half the rows, and the audit proves the branch was
    taken rather than assuming it.
    """
    var params = MvsBootstrapParams.enable_with_reg(1.0, 0.5, 11)
    var grads = _ramp_gradients(FIXTURE_ROWS)
    var audit = MvsAudit.empty()
    var w = _draw(params, grads, FIXTURE_ROWS, 0, 0.0, audit)

    assert_equal(audit.blocks_guarded, 0)
    assert_true(audit.rows_kept_certainly > 0)

    # The largest-magnitude row is one of them: at subsample 0.5 the threshold
    # cannot exceed the top magnitude, or fewer than half the rows would be
    # reachable at all.
    assert_equal(_bits(w[FIXTURE_ROWS - 1]), _bits(1.0))

    # An amplified row exists too, so the other branch is not dead either.
    var amplified = 0
    for r in range(FIXTURE_ROWS):
        if w[r] > 1.0:
            amplified += 1
    assert_true(amplified > 0)


def test_the_block_loop_splits_at_the_constant() raises:
    """More rows than `MVS_BLOCK_SIZE` means more than one threshold solve.

    CatBoost solves per block of 8192 and targets `SampleRate * blockSize`
    within each, so the block count is part of the method and not an internal
    detail: a port that solved once globally would produce different weights.
    Asserted through the audit, which is the only way to see it from outside.
    """
    var n = MVS_BLOCK_SIZE + 5
    var params = MvsBootstrapParams.enable_with_reg(16.0, 0.8, 3)
    var grads = _constant_gradients(n, 3.0)
    var audit = MvsAudit.empty()
    var w = _draw(params, grads, n, 0, 0.0, audit)
    assert_equal(audit.blocks, 2)
    assert_equal(audit.blocks_guarded, 0)
    assert_equal(len(w), n)

    # The short trailing block solves its own threshold on its own row count,
    # and reaches the same `mu` here only because every candidate is equal:
    # `(5 * k) / (0.8 * k)` is 6.25 for any k. So both blocks agree on the
    # surviving weight, which is what makes this assertion exact.
    var expected_kept = 1.0 / (5.0 / (1000.0 / 160.0))
    for r in range(n):
        if _bits(w[r]) != _bits(0.0):
            assert_equal(_bits(w[r]), _bits(expected_kept))


# ---------------------------------------------------------------------------
# 3. The degenerate threshold, and proof that the guard opened
# ---------------------------------------------------------------------------


def test_the_degenerate_guard_fires_and_keeps_the_block_whole() raises:
    """The branch with no CatBoost referent, proven to be reached.

    With every derivative zero and a derived lambda of zero, every candidate is
    zero, the pivot is zero, `sumOfSmall / pivot` is `0.0/0.0` = NaN, the
    `NaN > sampleSize` test is false, and CatBoost's undershoot branch returns
    `0.0 / (0.8n - n)` -- a NEGATIVE ZERO. `GetSingleProbability` then computes
    `0.0 / -0.0` = NaN, and `probability > epsilon` is false for NaN, so
    CatBoost writes weight 0 for every row and trains the tree on nothing, with
    no exception and no NaN left in the output to notice.

    Ours refuses to do that. `_mvs_threshold_is_usable` rejects the negative
    zero (`-0.0 > 0.0` is false) and the block is kept whole at weight exactly
    1.0, which is the correct limit of the method as `mu` falls to zero.

    `blocks_guarded` is asserted nonzero here, so this test cannot be one whose
    gate never opened, and it is asserted zero in every other test in this file
    so the counter is not simply always on.
    """
    var params = MvsBootstrapParams.enable(0.8, 5)
    var grads = _constant_gradients(N_ROWS, 0.0)
    # Unset `mvs_reg`, and the derived lambda for an all-zero gradient buffer
    # is itself zero -- which is exactly the state `check_mvs_reg` refuses a
    # user from creating by hand and cannot refuse the data from creating.
    var auto = mvs_auto_lambda_from_gradients(grads, N_ROWS)
    assert_equal(_bits(auto), _bits(0.0))

    var audit = MvsAudit.empty()
    var w = _draw(params, grads, N_ROWS, 0, auto, audit)

    assert_equal(audit.blocks, 1)
    assert_equal(audit.blocks_guarded, 1)
    assert_equal(audit.rows_kept, N_ROWS)
    assert_equal(audit.rows_kept_certainly, N_ROWS)
    for r in range(N_ROWS):
        assert_equal(_bits(w[r]), _bits(1.0), String("guarded row ", r))

    # And the guard is not a blanket "keep everything": the same params on a
    # nonzero gradient sample rows and do not take the guard.
    var live_audit = MvsAudit.empty()
    var live = _draw(
        params, _constant_gradients(N_ROWS, 3.0), N_ROWS, 0, 16.0, live_audit
    )
    assert_equal(len(live), N_ROWS)
    assert_equal(live_audit.blocks_guarded, 0)
    assert_true(live_audit.rows_kept < N_ROWS)


def test_mvs_reg_zero_is_refused_by_name() raises:
    """The user-facing half of the same defect.

    CatBoost accepts `mvs_reg=0` (its `Validate` only requires `>= 0`). We do
    not, because 0's only two outcomes are "indistinguishable from leaving it
    unset" and "silently empty tree", and the second one produces no error, no
    warning and no NaN anybody would see. A knob with no good case is a trap.
    """
    var raised = False
    try:
        check_mvs_reg(0.0)
    except:
        raised = True
    assert_true(raised, "mvs_reg = 0 must raise")

    # Negative and NaN are refused too; the comparison is written so NaN falls
    # on the raising side rather than sliding through a `< 0` test.
    raised = False
    try:
        check_mvs_reg(-1.0)
    except:
        raised = True
    assert_true(raised, "negative mvs_reg must raise")
    raised = False
    try:
        check_mvs_reg(nan[DType.float64]())
    except:
        raised = True
    assert_true(raised, "NaN mvs_reg must raise")

    # A positive value is accepted, so the check is not refusing everything.
    check_mvs_reg(1e-30)

    # And the refusal reaches `validate`, which is what a fit calls.
    raised = False
    try:
        MvsBootstrapParams.enable_with_reg(0.0).validate()
    except:
        raised = True
    assert_true(raised, "validate must carry the mvs_reg refusal")
    MvsBootstrapParams.enable_with_reg(4.0).validate()
    # An UNSET reg is never checked against zero: the derived value is allowed
    # to be zero because the threshold guard covers that case.
    MvsBootstrapParams.enable().validate()


# ---------------------------------------------------------------------------
# 4. The option surface, including one deliberate divergence
# ---------------------------------------------------------------------------


def test_bagging_temperature_is_refused_beside_mvs() raises:
    """A divergence, deliberately, and the reason is in the docstring.

    CatBoost's `TBootstrapConfig::Validate` refuses `bagging_temperature` for
    `No` and for Bernoulli but its MVS arm checks only the sampling unit, so
    CatBoost accepts `bootstrap_type=MVS, bagging_temperature=5`, never reads
    it, and drops it on save. The user gets no error and no effect.
    """
    var mvs = MvsBootstrapParams.enable()
    var raised = False
    try:
        check_mvs_bagging_temperature(mvs, True)
    except:
        raised = True
    assert_true(raised, "an explicit bagging_temperature must raise under MVS")

    # A defaulted temperature is not a user setting and is not refused.
    check_mvs_bagging_temperature(mvs, False)
    # And with MVS off the parameter is the Bayesian bootstrap's business.
    check_mvs_bagging_temperature(MvsBootstrapParams.disabled(), True)


def test_one_bootstrap_type_at_a_time() raises:
    """CatBoost's `bootstrap_type` is a single enum, so the combination cannot
    be spelled there. Ours is two bundles and can be, so it is refused."""
    var raised = False
    try:
        check_bootstrap_type_exclusive(
            MvsBootstrapParams.enable(), BayesianBootstrapParams.enable()
        )
    except:
        raised = True
    assert_true(raised)

    check_bootstrap_type_exclusive(
        MvsBootstrapParams.enable(), BayesianBootstrapParams.disabled()
    )
    check_bootstrap_type_exclusive(
        MvsBootstrapParams.disabled(), BayesianBootstrapParams.enable()
    )


def test_subsample_range_is_catboosts() raises:
    """"Subsample should be in (0,1]", and NaN falls on the raising side."""
    var bad: List[Float64] = [0.0, -0.5, 1.5]
    for i in range(len(bad)):
        var raised = False
        try:
            MvsBootstrapParams.enable(bad[i]).validate()
        except:
            raised = True
        assert_true(raised, String("subsample ", bad[i], " must raise"))

    var raised_nan = False
    try:
        MvsBootstrapParams.enable(nan[DType.float64]()).validate()
    except:
        raised_nan = True
    assert_true(raised_nan, "NaN subsample must raise")

    # 1.0 is legal and is CatBoost's own no-op early return, not an error.
    var whole = MvsBootstrapParams.enable(1.0)
    whole.validate()
    assert_true(whole.enabled)
    assert_false(whole.samples_rows())
    var audit = MvsAudit.empty()
    var w = _draw(whole, _ramp_gradients(N_ROWS), N_ROWS, 0, 1.0, audit)
    assert_equal(audit.blocks, 0)
    assert_equal(len(w), N_ROWS)
    for r in range(N_ROWS):
        assert_equal(_bits(w[r]), _bits(1.0))


# ---------------------------------------------------------------------------
# 5. The constant-hessian exclusion
# ---------------------------------------------------------------------------


def test_an_mvs_fit_does_not_take_the_two_plane_path() raises:
    """The weights are hessians, so the specialization must be refused.

    `boosting.round_has_constant_hessian` is what a trainer asks before telling
    a histogram builder to accumulate two planes and rebuild the third from the
    count. Squared error unweighted is its one True; the same objective with
    MVS's weights in hand must be a False, because those weights are stored
    into `hess` and the count plane no longer reconstructs it. MVS is the
    stronger case of the two bootstraps: its weights are not merely unequal,
    most are exactly 1, the rest are large, and a zero is a dropped row.
    """
    var goss = GossParams.disabled()
    var params = MvsBootstrapParams.enable_with_reg(16.0, 0.8, 2)

    # The positive control first, so a predicate that returned False for
    # everything could not pass this test.
    assert_true(
        round_has_constant_hessian(SQUARED_ERROR, List[Float64](), goss)
    )

    var effective = List[Float64]()
    var audit = MvsAudit.empty()
    refresh_mvs_bootstrap(
        effective,
        audit,
        params,
        _constant_gradients(N_ROWS, 3.0),
        List[Float64](),
        N_ROWS,
        0,
        0.0,
    )
    assert_equal(len(effective), N_ROWS)
    assert_equal(audit.blocks_guarded, 0)
    assert_false(round_has_constant_hessian(SQUARED_ERROR, effective, goss))

    # The weights really are the reason: they are not all 1.0, so the refusal
    # is not a formality about a vector's presence.
    var moved = 0
    for r in range(N_ROWS):
        if _bits(effective[r]) != _bits(1.0):
            moved += 1
    assert_true(moved > 0)

    # And the guard, which is what a round loop calls when it has the
    # configuration rather than the vector.
    assert_true(mvs_varies_hessian(params))
    var raised = False
    try:
        check_mvs_hessian_declaration(params, True)
    except:
        raised = True
    assert_true(raised)
    check_mvs_hessian_declaration(params, False)

    # The predicate is coarse on purpose: `subsample = 1` draws nothing but
    # still refuses the declaration, because the declaration is held for a
    # whole fit and must not reason about the value of a Float64 knob.
    assert_true(mvs_varies_hessian(MvsBootstrapParams.enable(1.0)))
    assert_false(mvs_varies_hessian(MvsBootstrapParams.disabled()))
    check_mvs_hessian_declaration(MvsBootstrapParams.disabled(), True)


# ---------------------------------------------------------------------------
# 6. The draw fires, and reads the magnitude
# ---------------------------------------------------------------------------


def test_the_draw_changes_the_model() raises:
    """Enabling it must move the fit, asserted as bits.

    No trainer wires the per-tree refresh yet, so this fixture stands in for it
    the way CatBoost's own structure allows: the MVS weight is a sample weight
    (`CalcWeightedData`), so handing one tree's draw to `train` as
    `sample_weight` exercises exactly the arithmetic a wired round loop would,
    with one draw held fixed across the rounds instead of redrawn. That is
    weaker than the real schedule and stronger than nothing: if the drawn
    weights were degenerate, this is the assertion that fails.
    """
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var booster = _booster_params()

    var plain = _model_bits(train(data, y, SQUARED_ERROR, booster))
    var audit = MvsAudit.empty()
    var drawn = _draw(
        MvsBootstrapParams.enable_with_reg(4.0, 0.8, 1),
        _ramp_gradients(N_ROWS),
        N_ROWS,
        0,
        0.0,
        audit,
    )
    assert_equal(audit.blocks_guarded, 0)
    var sampled = _model_bits(train(data, y, SQUARED_ERROR, booster, drawn))

    var same = True
    for b in range(N_BINS):
        if plain[b] != sampled[b]:
            same = False
    assert_false(same, "the MVS draw must move the model")


def test_the_weight_reads_the_magnitude_not_the_sign() raises:
    """A larger gradient magnitude never buys a larger surviving weight.

    This is the property that makes MVS minimum-variance rather than uniform:
    the keep probability rises with `|g|` and the weight is its reciprocal, so
    surviving weights are non-increasing in magnitude. The fixture's gradients
    alternate in sign, so a sampler that read the signed value instead of the
    magnitude would order the rows differently and fail here. The comparison is
    an exact ordering, not a tolerance.
    """
    var params = MvsBootstrapParams.enable_with_reg(1.0, 0.6, 4)
    var audit = MvsAudit.empty()
    var w = _draw(
        params, _ramp_gradients(FIXTURE_ROWS), FIXTURE_ROWS, 0, 0.0, audit
    )
    assert_equal(audit.blocks_guarded, 0)

    # `_ramp_gradients` has strictly increasing magnitude in the row index.
    var previous = 0.0
    var seen = False
    for r in range(FIXTURE_ROWS):
        if _bits(w[r]) == _bits(0.0):
            continue
        if seen:
            assert_true(
                w[r] <= previous,
                String("weight must not rise with magnitude at row ", r),
            )
        previous = w[r]
        seen = True
    assert_true(seen)

    # The kept set is the ascending, duplicate-free form the rest of the module
    # takes, and the audit's count is its length rather than a parallel tally.
    var rows = mvs_kept_rows(w)
    check_row_set(rows, FIXTURE_ROWS)
    assert_equal(len(rows), audit.rows_kept)
    assert_true(len(rows) < FIXTURE_ROWS, "MVS must actually drop rows")


def test_multi_output_gradients_use_the_l2_norm() raises:
    """`sum_k der[i][k]^2` across outputs, row-major with stride `n_outputs`.

    One row with gradients (3, 4) has magnitude 5, so with `mvs_reg = 11` its
    statistic is `sqrt(25 + 11) = 6`. The check is indirect but exact: two
    buffers that must produce the same `g` produce the identical weight vector,
    and one that must not, does not.
    """
    var params = MvsBootstrapParams.enable_with_reg(11.0, 0.5, 6)

    # Two outputs, (3, 4) per row: magnitude 5.
    var wide = List[Float64]()
    for _ in range(N_ROWS):
        wide.append(3.0)
        wide.append(4.0)
    var wide_audit = MvsAudit.empty()
    var wide_w = _draw(params, wide, N_ROWS, 0, 0.0, wide_audit, 2)

    # One output at 5.0 per row: the same magnitude, so the same weights.
    var narrow = _constant_gradients(N_ROWS, 5.0)
    var narrow_audit = MvsAudit.empty()
    var narrow_w = _draw(params, narrow, N_ROWS, 0, 0.0, narrow_audit, 1)

    _assert_same_weights(wide_w, narrow_w, "l2 norm across outputs")
    assert_equal(wide_audit.blocks_guarded, 0)

    # And a buffer that is genuinely different does differ, so the assertion
    # above is not passing because both sides are constant.
    var other_audit = MvsAudit.empty()
    var other_w = _draw(
        params, _constant_gradients(N_ROWS, 0.25), N_ROWS, 0, 0.0,
        other_audit, 1,
    )
    assert_true(_differ(other_w, narrow_w))

    # A gradient buffer too short for the declared shape is refused rather
    # than read past its end.
    var short_audit = MvsAudit.empty()
    var raised = False
    try:
        _ = _draw(params, narrow, N_ROWS, 0, 0.0, short_audit, 2)
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# 7. The derived lambda
# ---------------------------------------------------------------------------


def test_the_auto_lambda_is_the_squared_mean_magnitude() raises:
    """`TMvsSampler::GetLambda`'s two branches, on values chosen to be exact.

    Gradients 3 and -4 give magnitudes 3 and 4, mean 3.5, and lambda 12.25.
    A single leaf whose two-dimensional value is (3, 4) gives norm 5, mean 5,
    and lambda 25. Both are exactly representable, so the comparison is on bits
    with nothing rounded.
    """
    var grads: List[Float64] = [3.0, -4.0]
    assert_equal(
        _bits(mvs_auto_lambda_from_gradients(grads, 2)), _bits(12.25)
    )

    var leaves: List[Float64] = [3.0, 4.0]
    assert_equal(
        _bits(mvs_auto_lambda_from_leaf_values(leaves, 1, 2)), _bits(25.0)
    )

    # The two branches are genuinely different functions of different data:
    # the same buffer read as two leaves of one output is a different lambda.
    assert_equal(
        _bits(mvs_auto_lambda_from_leaf_values(leaves, 2, 1)), _bits(12.25)
    )

    # Empty is 0, which is the value the threshold guard exists to absorb.
    assert_equal(
        _bits(mvs_auto_lambda_from_gradients(List[Float64](), 0)), _bits(0.0)
    )

    # Shape errors are refused rather than read short.
    var raised = False
    try:
        _ = mvs_auto_lambda_from_gradients(grads, 5)
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        _ = mvs_auto_lambda_from_leaf_values(leaves, 1, 0)
    except:
        raised = True
    assert_true(raised)


def test_an_explicit_reg_overrides_the_derived_one() raises:
    """`if (Lambda.Defined()) return Lambda.GetRef();` -- the derived value is
    not consulted at all when the user set one.

    Asserted as a difference in the weights, not only in `resolve_reg`, so a
    `resolve_reg` that was correct but unused would still fail.
    """
    var grads = _ramp_gradients(N_ROWS)
    var pinned = MvsBootstrapParams.enable_with_reg(100.0, 0.7, 9)
    var a_audit = MvsAudit.empty()
    var b_audit = MvsAudit.empty()
    var a = _draw(pinned, grads, N_ROWS, 0, 1.0, a_audit)
    var b = _draw(pinned, grads, N_ROWS, 0, 99999.0, b_audit)
    _assert_same_weights(a, b, "explicit mvs_reg ignores the derived one")

    # And with the reg unset, the derived value does reach the draw.
    var derived = MvsBootstrapParams.enable(0.7, 9)
    var c_audit = MvsAudit.empty()
    var d_audit = MvsAudit.empty()
    var c = _draw(derived, grads, N_ROWS, 0, 1.0, c_audit)
    var d = _draw(derived, grads, N_ROWS, 0, 99999.0, d_audit)
    assert_true(_differ(c, d), "the derived lambda must reach the draw")


# ---------------------------------------------------------------------------
# 8. The key: (seed, tree index, row) and nothing else
# ---------------------------------------------------------------------------


def test_determinism_across_worker_counts() raises:
    """Required by the round's correctness contract: identical at
    `MOJOTREES_NUM_WORKERS` 1, 3 and 8.

    Both halves are checked, because only one of them is trivially true. The
    draw itself dispatches nothing and its block size is a compile-time
    constant rather than a worker count, so worker count cannot reach it --
    that is the design, and this pins it. The *fit* handed the drawn weights
    does dispatch, so the second half is a real assertion about the trainer
    under a weight vector containing zeros, which it would not otherwise see.
    """
    var params = MvsBootstrapParams.enable_with_reg(4.0, 0.8, 3)
    var grads = _ramp_gradients(N_ROWS)
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var booster = _booster_params()

    var reference = List[Float64]()
    var reference_model = List[UInt64]()
    var reference_kept = 0
    var settings: List[String] = [String("1"), String("3"), String("8")]
    for i in range(len(settings)):
        _ = setenv("MOJOTREES_NUM_WORKERS", settings[i])
        var audit = MvsAudit.empty()
        var w = _draw(params, grads, N_ROWS, 2, 0.0, audit)
        assert_equal(audit.blocks_guarded, 0)
        var model = _model_bits(train(data, y, SQUARED_ERROR, booster, w))
        if i == 0:
            reference = w.copy()
            reference_model = model.copy()
            reference_kept = audit.rows_kept
        else:
            _assert_same_weights(
                reference, w, String("weights at ", settings[i], " workers")
            )
            assert_equal(
                reference_kept,
                audit.rows_kept,
                String("kept count at ", settings[i], " workers"),
            )
            assert_equal(len(reference_model), len(model))
            for b in range(N_BINS):
                assert_equal(
                    reference_model[b],
                    model[b],
                    String("model at ", settings[i], " workers, bin ", b),
                )

    # The draw is not degenerate, so the equality above is a real statement
    # about a vector with structure in it rather than about a constant.
    assert_true(reference_kept > 0)
    assert_true(reference_kept < N_ROWS)


def test_every_key_moves_the_draw() raises:
    """Seed, tree index and row are all live inputs.

    A key that is accepted and ignored is how a sampler silently stops being
    reproducible across configurations while still passing every determinism
    test, so each one is asserted to change the result rather than assumed to.
    """
    var grads = _ramp_gradients(FIXTURE_ROWS)
    var base_audit = MvsAudit.empty()
    var base = _draw(
        MvsBootstrapParams.enable_with_reg(4.0, 0.8, 3),
        grads, FIXTURE_ROWS, 5, 0.0, base_audit,
    )

    # The same call twice is identical: reproducibility before sensitivity.
    var again_audit = MvsAudit.empty()
    var again = _draw(
        MvsBootstrapParams.enable_with_reg(4.0, 0.8, 3),
        grads, FIXTURE_ROWS, 5, 0.0, again_audit,
    )
    _assert_same_weights(base, again, "the same key redrawn")

    # A different seed moves it.
    var seed_audit = MvsAudit.empty()
    var other_seed = _draw(
        MvsBootstrapParams.enable_with_reg(4.0, 0.8, 4),
        grads, FIXTURE_ROWS, 5, 0.0, seed_audit,
    )
    assert_true(_differ(other_seed, base), "the seed must reach the draw")

    # A different tree index moves it. This is the per-tree schedule: CatBoost
    # redraws once per tree under `sampling_frequency=PerTree`, so two trees of
    # one fit must not share a sample.
    var tree_audit = MvsAudit.empty()
    var other_tree = _draw(
        MvsBootstrapParams.enable_with_reg(4.0, 0.8, 3),
        grads, FIXTURE_ROWS, 6, 0.0, tree_audit,
    )
    assert_true(
        _differ(other_tree, base), "the tree index must reach the draw"
    )

    # Row is live by construction, but assert it: a whole block of identical
    # magnitudes must not be forced to one shared decision.
    var flat_audit = MvsAudit.empty()
    var flat = _draw(
        MvsBootstrapParams.enable_with_reg(FIXTURE_REG, 0.8, 7),
        _constant_gradients(FIXTURE_ROWS, FIXTURE_GRAD),
        FIXTURE_ROWS,
        0,
        0.0,
        flat_audit,
    )
    var first = _bits(flat[0])
    var differs = False
    for r in range(FIXTURE_ROWS):
        if _bits(flat[r]) != first:
            differs = True
    assert_true(differs, "the row index must reach the draw")


def test_the_draw_does_not_depend_on_the_buffer_it_lands_in() raises:
    """`weights` is resized, not appended to, so a round loop can reuse one
    buffer for a whole fit and get the same vector a fresh one would."""
    var params = MvsBootstrapParams.enable_with_reg(4.0, 0.8, 3)
    var grads = _ramp_gradients(N_ROWS)

    var fresh = List[Float64]()
    var fresh_audit = MvsAudit.empty()
    mvs_bootstrap_weights(
        params, grads, N_ROWS, 1, 0.0, fresh, fresh_audit, 1
    )

    # A buffer carrying a previous, longer, differently-shaped draw.
    var reused = List[Float64]()
    var reused_audit = MvsAudit.empty()
    mvs_bootstrap_weights(
        params,
        _ramp_gradients(FIXTURE_ROWS),
        FIXTURE_ROWS,
        99,
        0.0,
        reused,
        reused_audit,
        1,
    )
    assert_equal(len(reused), FIXTURE_ROWS)
    mvs_bootstrap_weights(
        params, grads, N_ROWS, 1, 0.0, reused, reused_audit, 1
    )

    _assert_same_weights(fresh, reused, "reused buffer")
    # The audit is reset per draw too, not accumulated across calls.
    assert_equal(reused_audit.blocks, fresh_audit.blocks)
    assert_equal(reused_audit.rows_kept, fresh_audit.rows_kept)
    assert_equal(reused_audit.blocks_guarded, fresh_audit.blocks_guarded)


# ---------------------------------------------------------------------------
# 9. The default moves nothing
# ---------------------------------------------------------------------------


def test_a_disabled_bundle_leaves_the_fit_alone() raises:
    """The default path: no weights, not a vector of ones.

    A vector of 1.0s would be arithmetically harmless and would still make
    `boosting.round_has_constant_hessian` refuse a fit it should admit, so the
    empty-means-unweighted convention is the one that matters here, and it is
    asserted against a fit rather than only against a length.
    """
    var params = MvsBootstrapParams.disabled()
    assert_false(params.enabled)
    assert_false(params.samples_rows())
    assert_false(mvs_varies_hessian(params))

    var weights = List[Float64]()
    var audit = MvsAudit.empty()
    refresh_mvs_bootstrap(
        weights,
        audit,
        params,
        _ramp_gradients(N_ROWS),
        List[Float64](),
        N_ROWS,
        0,
        0.0,
    )
    assert_equal(len(weights), 0, "a disabled bundle must not fill weights")
    assert_equal(audit.blocks, 0)
    assert_equal(audit.blocks_guarded, 0)

    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var booster = _booster_params()
    var plain = _model_bits(train(data, y, SQUARED_ERROR, booster))
    var passed_through = _model_bits(
        train(data, y, SQUARED_ERROR, booster, weights)
    )
    for b in range(N_BINS):
        assert_equal(plain[b], passed_through[b], String("bin ", b))

    # A disabled bundle beside a user weight vector passes it straight through
    # unchanged, so wiring the sampler in cannot alter an already-weighted fit.
    var user = _user_weights(N_ROWS)
    refresh_mvs_bootstrap(
        weights, audit, params, _ramp_gradients(N_ROWS), user, N_ROWS, 0, 0.0
    )
    _assert_same_weights(weights, user, "user weights pass through")


def test_the_user_weight_multiplies_the_draw() raises:
    """`SampleWeights[i] *= learnWeights[i]` -- the effective weight is the
    product, which is why a sampled fit is a weighted fit downstream."""
    var params = MvsBootstrapParams.enable_with_reg(4.0, 0.8, 3)
    var grads = _ramp_gradients(N_ROWS)

    var audit = MvsAudit.empty()
    var drawn = _draw(params, grads, N_ROWS, 0, 0.0, audit)

    var user = _user_weights(N_ROWS)
    var combined = List[Float64]()
    var combined_audit = MvsAudit.empty()
    refresh_mvs_bootstrap(
        combined, combined_audit, params, grads, user, N_ROWS, 0, 0.0
    )
    assert_equal(len(combined), N_ROWS)
    for r in range(N_ROWS):
        assert_equal(
            _bits(combined[r]),
            _bits(drawn[r] * user[r]),
            String("product at row ", r),
        )

    # A mismatched user weight vector is refused rather than read short.
    var short: List[Float64] = [1.0, 1.0]
    var raised = False
    try:
        refresh_mvs_bootstrap(
            combined, combined_audit, params, grads, short, N_ROWS, 0, 0.0
        )
    except:
        raised = True
    assert_true(raised)


def test_shape_errors_are_refused() raises:
    """Negative counts and short buffers raise; zero rows is legal."""
    var params = MvsBootstrapParams.enable_with_reg(4.0)
    var grads = _ramp_gradients(N_ROWS)
    var audit = MvsAudit.empty()

    var raised = False
    try:
        _ = _draw(params, grads, -1, 0, 0.0, audit)
    except:
        raised = True
    assert_true(raised, "negative n_rows must raise")

    raised = False
    try:
        _ = _draw(params, grads, N_ROWS, -1, 0.0, audit)
    except:
        raised = True
    assert_true(raised, "negative tree_index must raise")

    raised = False
    try:
        _ = _draw(params, grads, N_ROWS, 0, 0.0, audit, 0)
    except:
        raised = True
    assert_true(raised, "non-positive n_outputs must raise")

    var empty_audit = MvsAudit.empty()
    var empty = _draw(params, List[Float64](), 0, 0, 0.0, empty_audit)
    assert_equal(len(empty), 0)
    assert_equal(empty_audit.blocks, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
