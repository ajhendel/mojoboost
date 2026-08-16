"""CatBoost's Bayesian bootstrap: the draw, its key, and its consequences.

The mechanism is `sampling.BayesianBootstrapParams` and the three functions
around it. It is CatBoost's `bootstrap_type=Bayesian`: every row is kept and
each is given a random weight, redrawn once per tree, stretched by
`bagging_temperature`. Transcribed from
`catboost/private/libs/algo/tensor_search_helpers.cpp`
(`GenerateBayessianWeight`, `GenerateRandomWeights`, `CalcWeightedData`).

What this file has to establish, in order of how badly it would hurt to be
wrong about it:

1. **The exclusion engages.** A bootstrap weight multiplies the row's
   derivatives, so the hessian is the weight and a bootstrapped fit must not
   take the two-plane constant-hessian path. Asserted here and, against every
   objective, in `tests/test_const_hessian_exclusions.mojo`.
2. **The draw fires.** Turning it on changes the model, asserted as bits
   rather than assumed. A sampling regularizer that silently produced weights
   of 1 would pass every determinism test in this file.
3. **The draw is reproducible.** Keyed by (seed, tree index, row) alone:
   identical at `MOJOTREES_NUM_WORKERS` 1, 3 and 8, independent of the buffer
   length it is written into, and moving only when one of those three inputs
   moves. All three keys are asserted live, because a key that is ignored is
   the way this kind of code goes wrong.
4. **The default moves nothing.** A disabled bundle produces an empty weight
   vector, not a vector of ones, and a fit handed it is byte-identical to a
   fit handed no weights at all.

No tolerance appears anywhere below: every float comparison is on
`to_bits()`. Where a distributional claim would need one -- "the mean weight
is about 1" -- the exact algebraic identity is asserted instead
(`weight at temperature 2 == weight at temperature 1, squared`).
"""

from std.math import isfinite
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
    BayesianBootstrapParams,
    DEFAULT_BAGGING_TEMPERATURE,
    DEFAULT_BOOTSTRAP_SEED,
    bayesian_bootstrap_varies_hessian,
    bayesian_bootstrap_weights,
    canonical_bootstrap_type,
    canonical_sampling_param,
    check_bayesian_bootstrap_hessian_declaration,
    is_sampling_param,
    refresh_bayesian_bootstrap,
)


comptime N_ROWS = 64
comptime N_BINS = 8


def _bits(v: Float64) -> UInt64:
    return v.to_bits().cast[DType.uint64]()


def _assert_same_weights(
    a: List[Float64], b: List[Float64], what: String
) raises:
    assert_equal(len(a), len(b), String(what, ": length"))
    for i in range(len(a)):
        assert_equal(_bits(a[i]), _bits(b[i]), String(what, ": row ", i))


def _weights(
    params: BayesianBootstrapParams, n_rows: Int, tree_index: Int
) raises -> List[Float64]:
    var w = List[Float64]()
    bayesian_bootstrap_weights(params, n_rows, tree_index, w)
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
# 1. The constant-hessian exclusion
# ---------------------------------------------------------------------------


def test_a_bootstrapped_fit_does_not_take_the_two_plane_path() raises:
    """The correctness half: the weights are hessians, so the specialization
    must be refused.

    `boosting.round_has_constant_hessian` is what a trainer asks before
    telling a histogram builder to accumulate two planes and rebuild the third
    from the count. Squared error unweighted is its one True; the same
    objective with the bootstrap's weights in hand must be a False, because
    `_fill_grad_hess_into` stores the weight into `hess` and the count plane no
    longer reconstructs it.
    """
    var goss = GossParams.disabled()
    var params = BayesianBootstrapParams.enable()

    # The positive control first, so a predicate that returned False for
    # everything could not pass this test.
    assert_true(
        round_has_constant_hessian(SQUARED_ERROR, List[Float64](), goss)
    )

    var effective = List[Float64]()
    refresh_bayesian_bootstrap(
        effective, params, List[Float64](), N_ROWS, 0
    )
    assert_equal(len(effective), N_ROWS)
    assert_false(
        round_has_constant_hessian(SQUARED_ERROR, effective, goss)
    )

    # The weights really are the reason: they are not all 1.0, so the refusal
    # is not a formality about a vector's presence.
    var moved = 0
    for r in range(N_ROWS):
        if _bits(effective[r]) != _bits(1.0):
            moved += 1
    assert_true(moved > 0)

    # And the guard, which is what a round loop calls when it has the
    # configuration rather than the vector.
    assert_true(bayesian_bootstrap_varies_hessian(params))
    var raised = False
    try:
        check_bayesian_bootstrap_hessian_declaration(params, True)
    except:
        raised = True
    assert_true(raised)
    check_bayesian_bootstrap_hessian_declaration(params, False)


# ---------------------------------------------------------------------------
# 2. The draw fires
# ---------------------------------------------------------------------------


def test_the_bootstrap_changes_the_model() raises:
    """Enabling it must move the fit, asserted as bits.

    No trainer wires the per-tree refresh yet, so this fixture stands in for
    it the way CatBoost's own structure allows: the bootstrap weight is a
    sample weight (`CalcWeightedData`), so handing one tree's draw to `train`
    as `sample_weight` exercises exactly the arithmetic a wired round loop
    would, with one draw held fixed across the rounds instead of redrawn. That
    is weaker than the real schedule and stronger than nothing: if the drawn
    weights were degenerate, this is the assertion that fails.
    """
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var params = _booster_params()

    var plain = _model_bits(train(data, y, SQUARED_ERROR, params))
    var drawn = _weights(BayesianBootstrapParams.enable(), N_ROWS, 0)
    var bootstrapped = _model_bits(
        train(data, y, SQUARED_ERROR, params, drawn)
    )

    assert_equal(len(plain), len(bootstrapped))
    var differing = 0
    for b in range(len(plain)):
        if plain[b] != bootstrapped[b]:
            differing += 1
    assert_true(differing > 0)


def test_a_disabled_bundle_leaves_the_fit_byte_identical() raises:
    """The default path. `refresh_bayesian_bootstrap` with a disabled bundle
    and no user weights yields an empty vector, and an empty `sample_weight`
    is the unweighted convention everywhere in mojotrees, so the fit is the
    one `train` builds with no weight argument at all -- bit for bit, not
    approximately.
    """
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var params = _booster_params()

    var off = List[Float64]()
    refresh_bayesian_bootstrap(
        off, BayesianBootstrapParams.disabled(), List[Float64](), N_ROWS, 0
    )
    assert_equal(len(off), 0)

    var plain = _model_bits(train(data, y, SQUARED_ERROR, params))
    var with_off = _model_bits(train(data, y, SQUARED_ERROR, params, off))
    for b in range(len(plain)):
        assert_equal(plain[b], with_off[b], String("bin ", b))


def test_zero_temperature_is_catboosts_all_ones_early_return() raises:
    """`if (baggingTemperature == 0) { Fill(..., 1); return; }`.

    Zero is not "off" in CatBoost's option surface -- it is a temperature that
    happens to produce weight 1 for every row -- and it is asserted as exactly
    1.0, not as close to it, because the whole point is that no draw is taken.
    """
    var w = _weights(
        BayesianBootstrapParams.enable(temperature=0.0), N_ROWS, 0
    )
    assert_equal(len(w), N_ROWS)
    for r in range(N_ROWS):
        assert_equal(_bits(w[r]), _bits(1.0), String("row ", r))
    # A disabled bundle fills the same buffer the same way, so a caller that
    # asks for weights while the sampler is off gets the neutral vector rather
    # than an empty one; `refresh_bayesian_bootstrap` is the entry point that
    # collapses it to empty.
    var off = _weights(BayesianBootstrapParams.disabled(), N_ROWS, 0)
    for r in range(N_ROWS):
        assert_equal(_bits(off[r]), _bits(1.0), String("off row ", r))


def test_weights_are_positive_and_finite() raises:
    """The `+ 1e-100` guard CatBoost carries inside the log is why. `uniform`
    is half-open on [0, 1), so a draw of exactly 0 is reachable and
    `-log(0)` would be an infinity that poisons every histogram it reaches.
    """
    var w = _weights(BayesianBootstrapParams.enable(), N_ROWS, 3)
    for r in range(N_ROWS):
        assert_true(w[r] > 0.0, String("row ", r, " not positive"))
        assert_true(isfinite(w[r]), String("row ", r, " not finite"))


def test_temperature_is_the_exponent() raises:
    """`powf(w, baggingTemperature)` -- so doubling the temperature squares
    the weight, exactly, for the same (seed, tree, row).

    This is the distributional claim ("higher temperature stretches the
    weights") stated as an identity that can be asserted on bits, since a test
    that checked the spread of the draw would need a tolerance and would
    establish nothing.
    """
    var one = _weights(
        BayesianBootstrapParams.enable(temperature=1.0), N_ROWS, 0
    )
    var two = _weights(
        BayesianBootstrapParams.enable(temperature=2.0), N_ROWS, 0
    )
    for r in range(N_ROWS):
        assert_equal(
            _bits(one[r] ** 2.0), _bits(two[r]), String("row ", r)
        )


def test_user_weights_multiply_the_draw() raises:
    """CatBoost's `CalcWeightedData` tail: `SampleWeights[i] *=
    learnWeights[i]`. The effective per-row weight is the product, and the
    product is what the derivatives and the leaf denominators carry."""
    var params = BayesianBootstrapParams.enable()
    var base = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        base.append(0.5 if r % 3 == 0 else 2.0)

    var drawn = _weights(params, N_ROWS, 5)
    var effective = List[Float64]()
    refresh_bayesian_bootstrap(effective, params, base, N_ROWS, 5)
    assert_equal(len(effective), N_ROWS)
    for r in range(N_ROWS):
        assert_equal(
            _bits(effective[r]),
            _bits(drawn[r] * base[r]),
            String("row ", r),
        )

    # With the sampler off, the effective weight is the user's, untouched.
    var passthrough = List[Float64]()
    refresh_bayesian_bootstrap(
        passthrough, BayesianBootstrapParams.disabled(), base, N_ROWS, 5
    )
    _assert_same_weights(passthrough, base, "disabled passthrough")


# ---------------------------------------------------------------------------
# 3. The key: (seed, tree index, row) and nothing else
# ---------------------------------------------------------------------------


def test_the_draw_is_independent_of_the_buffer_length() raises:
    """The order-independence proof.

    Row r's weight is `uniform(stream + r)`: a counter read, not a step of a
    running generator. So the first 16 entries of a 64-row draw must be the 16
    entries of a 16-row draw, bit for bit. Nothing about the walk -- its
    length, its direction, or a partition of it across workers -- can reach an
    individual weight. A stream-based sampler would fail this line.
    """
    var params = BayesianBootstrapParams.enable()
    var long = _weights(params, N_ROWS, 7)
    var short = _weights(params, 16, 7)
    assert_equal(len(short), 16)
    for r in range(16):
        assert_equal(_bits(long[r]), _bits(short[r]), String("row ", r))


def test_each_key_component_is_live() raises:
    """Seed, tree index and row each move the draw.

    A key that is silently ignored is how this class of code fails: the
    weights would still look random, still be reproducible, and be wrong (the
    same weights every tree, or the same weights for every seed). Each of the
    three is checked by moving it alone.
    """
    var base = BayesianBootstrapParams.enable()
    var tree0 = _weights(base, N_ROWS, 0)
    var tree1 = _weights(base, N_ROWS, 1)
    var seeded = _weights(
        BayesianBootstrapParams.enable(seed=DEFAULT_BOOTSTRAP_SEED + 1),
        N_ROWS,
        0,
    )

    var tree_moved = 0
    var seed_moved = 0
    var row_moved = 0
    for r in range(N_ROWS):
        if _bits(tree0[r]) != _bits(tree1[r]):
            tree_moved += 1
        if _bits(tree0[r]) != _bits(seeded[r]):
            seed_moved += 1
        if r > 0 and _bits(tree0[r]) != _bits(tree0[r - 1]):
            row_moved += 1
    # Every entry should move; asserting "most" would let a partially wired
    # key through, so these are equalities.
    assert_equal(tree_moved, N_ROWS)
    assert_equal(seed_moved, N_ROWS)
    assert_equal(row_moved, N_ROWS - 1)


def test_the_bootstrap_stream_is_not_the_feature_stream() raises:
    """The domain constant earns its place.

    Feature selection keys its per-tree stream on (seed, tree) too. A caller
    is free to give both samplers the same seed, and without
    `_BOOTSTRAP_DOMAIN` the two would then be reading one counter run: the
    tree's row weights and its feature subset would be correlated draws, which
    is a real statistical defect and an invisible one. The observable
    consequence asserted here is the weakest true statement -- that tree 0 at
    seed s and tree s at seed 0 are different draws -- which fails for the
    obvious wrong derivations (a bare `seed + tree`, or an unmixed xor).
    """
    var a = _weights(BayesianBootstrapParams.enable(seed=0), N_ROWS, 3)
    var b = _weights(BayesianBootstrapParams.enable(seed=3), N_ROWS, 0)
    var differing = 0
    for r in range(N_ROWS):
        if _bits(a[r]) != _bits(b[r]):
            differing += 1
    assert_equal(differing, N_ROWS)


def test_determinism_across_worker_counts() raises:
    """Required by the round's correctness contract: identical at
    `MOJOTREES_NUM_WORKERS` 1, 3 and 8.

    Both halves are checked, because only one of them is trivially true. The
    draw itself dispatches nothing, so worker count cannot reach it -- that is
    the design, and this pins it. The *fit* handed the drawn weights does
    dispatch, so the second half is a real assertion about the trainer under a
    weight vector it would not otherwise see.
    """
    var params = BayesianBootstrapParams.enable()
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var booster = _booster_params()

    var reference = List[Float64]()
    var reference_model = List[UInt64]()
    var settings: List[String] = [
        String("1"), String("3"), String("8")
    ]
    for i in range(len(settings)):
        _ = setenv("MOJOTREES_NUM_WORKERS", settings[i])
        var w = _weights(params, N_ROWS, 2)
        var model = _model_bits(train(data, y, SQUARED_ERROR, booster, w))
        if i == 0:
            reference = w.copy()
            reference_model = model.copy()
        else:
            _assert_same_weights(
                w, reference, String("workers=", settings[i])
            )
            for b in range(len(model)):
                assert_equal(
                    model[b],
                    reference_model[b],
                    String("workers=", settings[i], " bin ", b),
                )
    # Leave the environment as this test found it.
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


# ---------------------------------------------------------------------------
# 4. The option surface
# ---------------------------------------------------------------------------


def test_the_defaults_are_catboosts() raises:
    """`bagging_temperature` 1.0 and a bundle that is off."""
    assert_equal(_bits(DEFAULT_BAGGING_TEMPERATURE), _bits(1.0))
    var off = BayesianBootstrapParams.disabled()
    assert_false(off.enabled)
    assert_false(off.draws_weights())
    assert_false(bayesian_bootstrap_varies_hessian(off))
    var on = BayesianBootstrapParams.enable()
    assert_true(on.enabled)
    assert_true(on.draws_weights())
    assert_equal(_bits(on.temperature), _bits(DEFAULT_BAGGING_TEMPERATURE))
    assert_equal(on.seed, DEFAULT_BOOTSTRAP_SEED)


def test_a_negative_or_nan_temperature_is_refused() raises:
    """CatBoost's `Validate`: "Bagging temperature should be >= 0". The NaN
    case is checked because the comparison is written as `not (t >= 0.0)`
    precisely so that it catches one, and a later simplification to `t < 0.0`
    would not."""
    var refused: List[Float64] = [-1.0, -1e-300, nan[DType.float64]()]
    for i in range(len(refused)):
        var raised = False
        try:
            BayesianBootstrapParams.enable(
                temperature=refused[i]
            ).validate()
        except:
            raised = True
        assert_true(raised, String("accepted temperature ", i))
    # And the accepted end of the range.
    BayesianBootstrapParams.enable(temperature=0.0).validate()
    BayesianBootstrapParams.disabled().validate()


def test_bootstrap_type_values() raises:
    """Only `No` and `Bayesian` are implemented, and Bernoulli is refused by
    name rather than as an unknown value because mojotrees already has that
    draw as `bagging_fraction`."""
    assert_equal(canonical_bootstrap_type(String("no")), "no")
    assert_equal(canonical_bootstrap_type(String("none")), "no")
    assert_equal(canonical_bootstrap_type(String("bayesian")), "bayesian")
    # `mvs` moved out of the refusal list the day the sampler landed. It is
    # CatBoost's actual CPU default for both of our headline objectives, so
    # this assertion is the difference between claiming the peer column and
    # having it.
    assert_equal(canonical_bootstrap_type(String("mvs")), "mvs")
    for value in [
        String("bernoulli"),
        String("poisson"),
        String("Bayesian"),
        String(""),
    ]:
        var raised = False
        try:
            _ = canonical_bootstrap_type(value)
        except:
            raised = True
        assert_true(raised, String("accepted ", value))


def test_parameter_names_resolve() raises:
    """The three names join the sampling table, so the Python layer, the CLI
    and the C API resolve them through the same function as every other
    sampling parameter instead of each keeping a list."""
    for name in [
        String("bootstrap_type"),
        String("bagging_temperature"),
        String("bootstrap_seed"),
    ]:
        assert_true(is_sampling_param(name))
        assert_equal(canonical_sampling_param(name), name)
    # `subsample` is now the CANONICAL name for row bagging rather than an
    # alias resolving to LightGBM's `bagging_fraction`, and it is ONE key
    # shared by every sampler that selects rows. That is CatBoost's own
    # shape: one `subsample` option, read by Bernoulli, MVS and Poisson, and
    # refused by Bayesian -- which is this file's sampler and which weights
    # every row rather than selecting any. So the assertion that matters here
    # is unchanged in force and only in spelling: `subsample` must resolve to
    # the row-selection key and must NOT be captured by the bootstrap names.
    assert_equal(canonical_sampling_param(String("subsample")), "subsample")


def test_row_count_and_tree_index_are_validated() raises:
    """A negative row count or tree index is a caller bug, not a draw."""
    var params = BayesianBootstrapParams.enable()
    var w = List[Float64]()
    var raised = False
    try:
        bayesian_bootstrap_weights(params, -1, 0, w)
    except:
        raised = True
    assert_true(raised)
    raised = False
    try:
        bayesian_bootstrap_weights(params, N_ROWS, -1, w)
    except:
        raised = True
    assert_true(raised)
    # A mismatched user weight vector is refused rather than read short.
    var short: List[Float64] = [1.0, 1.0]
    raised = False
    try:
        refresh_bayesian_bootstrap(w, params, short, N_ROWS, 0)
    except:
        raised = True
    assert_true(raised)
    # Zero rows is legal and yields an empty draw.
    bayesian_bootstrap_weights(params, 0, 0, w)
    assert_equal(len(w), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
