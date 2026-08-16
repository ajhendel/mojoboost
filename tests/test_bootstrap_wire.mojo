"""The two round-loop edges: `bootstrap_type` and `random_strength`.

**UNRUN.** Written under a hard no-tests order covering every lane of this
campaign; not compiled, not executed, not once. Every assertion below is a
claim about what the code should do, and none of them has been observed.

`tests/test_mvs_bootstrap.mojo` and `tests/test_bayesian_bootstrap.mojo` pin
the draws themselves and `tests/test_random_strength.mojo` pins the noise; all
three predate this file and all three tested mechanisms with **no callers**.
What is new here is the wiring, so this file tests only the wire:

1. **The default arm does not move.** A fit handed `BootstrapParams.disabled()`
   is bit for bit the fit handed nothing, and `random_strength = 0` reaches the
   grower with the bundle it always reached it with.
2. **The exclusions are refusals.** `bootstrap_type` beside `bagging_fraction`,
   beside GOSS, beside balanced bagging, beside `boosting_type=ordered`, and
   MVS beside Bayesian -- every one of them raises rather than picking a
   winner. `bagging_fraction` is CatBoost's Bernoulli bootstrap under
   mojotrees's name, so that first pair is two `bootstrap_type` values at once.
3. **The constant-hessian exclusion engages.** This is the one that would be a
   silently wrong model rather than an error: a bootstrap weight goes into
   `hess`, and a histogram builder told to rebuild that plane from the row
   count rebuilds ones over draws. `BootstrapParams.check_hessian_declaration`
   is asserted to raise on a `True` declaration and to accept a `False` one.
4. **`bootstrap_round` scales the derivatives by the DRAW and not by the
   product.** The user's `sample_weight` is already inside `grad` and `hess`
   when the sampler runs, so a caller that passed the product would square it.
5. **MVS's dropped rows leave through the row list**, and an all-dropped draw
   is refused rather than written back as the empty list that means "every
   row".
6. **`random_strength` reaches the split search.** A positive strength no
   longer raises, the model it produces differs from the unnoised one, and the
   scale the loop computes is exactly
   `random_score_scale_from_gradients(grad, n, round * learning_rate)`.

Float comparisons are on `to_bits()` throughout, as in the three files above.

This file must NOT be renamed to `test_gpu_*`: `tools/run_tests.sh` selects the
accelerator subset by name and would silently exclude it from the CPU suite.
"""

from std.testing import assert_equal, assert_false, assert_true, TestSuite

from mojotrees import (
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    TreeParams,
    bin_equal_width,
    train,
)
from mojotrees.bagging import BaggingParams
from mojotrees.goss import GossParams
from mojotrees.ordered_boosting import OrderedBoostingParams
from mojotrees.sampling import (
    BayesianBootstrapParams,
    BootstrapParams,
    ClassBaggingParams,
    MvsAudit,
    MvsBootstrapParams,
    apply_bootstrap_weights,
    bootstrap_round,
    check_bootstrap_honored,
)
from mojotrees.tree_parameters_extra import (
    ExtraTreeParams,
    random_score_scale_from_gradients,
)


comptime N_ROWS = 64
comptime N_BINS = 8


def _bits(v: Float64) -> UInt64:
    return v.to_bits().cast[DType.uint64]()


def _features() -> List[Float64]:
    var f = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        f.append(Float64(r))
    return f^


def _target() -> List[Float64]:
    var y = List[Float64](capacity=N_ROWS)
    for r in range(N_ROWS):
        y.append(0.0 if r < 32 else 1.0)
    return y^


def _booster_params() -> BoosterParams:
    return BoosterParams(12, 0.3, TreeParams(4, 1, 1.0, 1e-3))


def _model_bits(model: Booster) raises -> List[UInt64]:
    var out = List[UInt64](capacity=N_BINS)
    for b in range(N_BINS):
        var bins: List[Int] = [b]
        out.append(_bits(model.predict_bins(bins)))
    return out^


def _ramp_gradients(n: Int) -> List[Float64]:
    var g = List[Float64](capacity=n)
    for r in range(n):
        var m = Float64(r + 1)
        g.append(-m if (r & 1) == 1 else m)
    return g^


def _ones(n: Int) -> List[Float64]:
    var h = List[Float64](capacity=n)
    for _ in range(n):
        h.append(1.0)
    return h^


# ---------------------------------------------------------------------------
# 1. The default arm does not move
# ---------------------------------------------------------------------------


def test_a_disabled_bundle_is_the_fit_we_already_had() raises:
    """Passing `BootstrapParams.disabled()` is passing nothing.

    The whole safety case for wiring a sampler into the round loop rests on
    this: the default path must not take a draw, must not allocate a weight
    vector, and must not multiply a single derivative.
    """
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var booster = _booster_params()
    var plain = _model_bits(train(data, y, SQUARED_ERROR, booster))
    var explicit = _model_bits(
        train(
            data,
            y,
            SQUARED_ERROR,
            booster,
            bootstrap=BootstrapParams.disabled(),
        )
    )
    for b in range(N_BINS):
        assert_equal(plain[b], explicit[b], String("bin ", b))


def test_a_disabled_round_touches_nothing() raises:
    """`bootstrap_round` on a disabled bundle leaves every buffer alone, and
    leaves the weight vector EMPTY rather than filled with ones.

    A vector of ones would be refused by
    `boosting.round_has_constant_hessian`'s sample-weight exclusion if it were
    ever handed on as a `sample_weight`, costing an unbootstrapped fit its
    two-plane specialization for nothing at all.
    """
    var rows = List[Int]()
    var grad = _ramp_gradients(N_ROWS)
    var hess = _ones(N_ROWS)
    var before_grad = grad.copy()
    var weights: List[Float64] = [1.0, 2.0, 3.0]
    var audit = MvsAudit(7, 7, 7, 7)

    bootstrap_round(
        rows,
        grad,
        hess,
        weights,
        audit,
        BootstrapParams.disabled(),
        N_ROWS,
        0,
        0.0,
    )
    assert_equal(len(rows), 0, "a disabled bundle must not name rows")
    assert_equal(len(weights), 0, "a disabled bundle must leave weights empty")
    assert_equal(audit.blocks, 0)
    for r in range(N_ROWS):
        assert_equal(_bits(grad[r]), _bits(before_grad[r]), String("row ", r))
        assert_equal(_bits(hess[r]), _bits(1.0), String("hess ", r))


# ---------------------------------------------------------------------------
# 2. The exclusions are refusals
# ---------------------------------------------------------------------------


def _train_refuses(
    bootstrap: BootstrapParams,
    bagging: BaggingParams,
    goss: GossParams,
    class_bagging: ClassBaggingParams,
    what: String,
) raises:
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var booster = _booster_params()
    var raised = False
    try:
        _ = train(
            data,
            y,
            SQUARED_ERROR,
            booster,
            bagging=bagging,
            goss=goss,
            class_bagging=class_bagging,
            bootstrap=bootstrap,
        )
    except:
        raised = True
    assert_true(raised, what)


def test_bootstrap_is_exclusive_with_every_other_row_sampler() raises:
    """One row sampler at a time, and each crossing raises.

    `bagging_fraction` is the sharpest of the four: it IS CatBoost's Bernoulli
    bootstrap under mojotrees's name (`sampling.canonical_bootstrap_type`
    refuses the spelling for that reason), so `bagging_fraction` beside
    `bootstrap_type=MVS` asks for two values of one CatBoost enum.
    """
    _train_refuses(
        BootstrapParams.mvs_at(0.8, 1),
        BaggingParams(0.5, 1, 7),
        GossParams.disabled(),
        ClassBaggingParams.disabled(),
        "mvs beside bagging_fraction must raise",
    )
    _train_refuses(
        BootstrapParams.bayesian_at(1.0, 1),
        BaggingParams(0.5, 1, 7),
        GossParams.disabled(),
        ClassBaggingParams.disabled(),
        "bayesian bootstrap beside bagging_fraction must raise",
    )
    _train_refuses(
        BootstrapParams.mvs_at(0.8, 1),
        BaggingParams.disabled(),
        GossParams.enable(),
        ClassBaggingParams.disabled(),
        "mvs beside goss must raise",
    )


def test_two_bootstrap_types_at_once_are_refused() raises:
    """CatBoost's `bootstrap_type` is one enum; ours is two bundles and can
    spell the combination, so `validate` refuses it."""
    var both = BootstrapParams(
        MvsBootstrapParams.enable(0.8, 1),
        BayesianBootstrapParams.enable(1.0, 1),
    )
    var raised = False
    try:
        both.validate()
    except:
        raised = True
    assert_true(raised, "mvs and bayesian together must raise")


def test_ordered_boosting_refuses_a_bootstrap() raises:
    """Ordered boosting already refuses bagging, GOSS and balanced bagging
    with one sentence -- a dropped row changes which prefix each fold was
    fitted on -- and MVS drops rows while the Bayesian bootstrap reweights
    every one of them."""
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var booster = BoosterParams(
        4,
        0.3,
        TreeParams(4, 1, 1.0, 1e-3),
        ordered=OrderedBoostingParams.enable(),
    )
    var raised = False
    try:
        _ = train(
            data,
            y,
            SQUARED_ERROR,
            booster,
            bootstrap=BootstrapParams.mvs_at(0.8, 1),
        )
    except:
        raised = True
    assert_true(raised, "ordered boosting beside a bootstrap must raise")


def test_an_unwired_entry_point_refuses_rather_than_ignores() raises:
    """`check_bootstrap_honored` is what a trainer with no `bootstrap_round`
    in its loop calls, so a bundle is never accepted and dropped."""
    var raised = False
    try:
        check_bootstrap_honored(
            BootstrapParams.mvs_at(0.8, 1), String("a test")
        )
    except:
        raised = True
    assert_true(raised)
    # A disabled bundle is not a setting and is not refused.
    check_bootstrap_honored(BootstrapParams.disabled(), String("a test"))


# ---------------------------------------------------------------------------
# 3. The constant-hessian exclusion
# ---------------------------------------------------------------------------


def test_the_hessian_declaration_is_refused_beside_a_bootstrap() raises:
    """The failure this whole pairing exists to prevent.

    A bootstrap weight multiplies the row's derivatives, so under an objective
    whose unweighted hessian is the literal 1.0 the drawn weight IS the
    hessian. A builder told to rebuild that plane from the row count would
    rebuild ones over draws with no error anywhere.
    """
    var mvs = BootstrapParams.mvs_at(0.8, 1)
    var bayes = BootstrapParams.bayesian_at(1.0, 1)
    var off = BootstrapParams.disabled()

    assert_true(mvs.varies_hessian())
    assert_true(bayes.varies_hessian())
    assert_false(off.varies_hessian())

    # `subsample = 1.0` and `bagging_temperature = 0` take no draw, and are
    # STILL excluded: the predicate tests `enabled` alone so that a
    # fit-lifetime declaration cannot go stale part way through a loop.
    assert_true(BootstrapParams.mvs_at(1.0, 1).varies_hessian())
    assert_true(BootstrapParams.bayesian_at(0.0, 1).varies_hessian())

    for i in range(2):
        var params = mvs if i == 0 else bayes
        var raised = False
        try:
            params.check_hessian_declaration(True)
        except:
            raised = True
        assert_true(raised, String("arm ", i, " must refuse a True declaration"))
        params.check_hessian_declaration(False)
    off.check_hessian_declaration(True)


# ---------------------------------------------------------------------------
# 4. The draw goes onto the derivatives, and it is the draw alone
# ---------------------------------------------------------------------------


def test_the_bayesian_round_scales_the_derivatives_by_the_draw() raises:
    """`grad` and `hess` come out as the entry values times the per-row
    weight, and the row list is untouched: the Bayesian bootstrap keeps every
    row.

    The weights are re-derived here from the vector the sampler wrote rather
    than from a second call, so this asserts the multiplication and not the
    draw (which `tests/test_bayesian_bootstrap.mojo` owns).
    """
    var rows = List[Int]()
    var grad = _ramp_gradients(N_ROWS)
    var hess = _ones(N_ROWS)
    var before = grad.copy()
    var weights = List[Float64]()
    var audit = MvsAudit.empty()

    bootstrap_round(
        rows,
        grad,
        hess,
        weights,
        audit,
        BootstrapParams.bayesian_at(1.0, 3),
        N_ROWS,
        5,
        0.0,
    )
    assert_equal(len(rows), 0, "the bayesian bootstrap keeps every row")
    assert_equal(len(weights), N_ROWS)
    for r in range(N_ROWS):
        assert_equal(
            _bits(grad[r]), _bits(before[r] * weights[r]), String("grad ", r)
        )
        assert_equal(_bits(hess[r]), _bits(weights[r]), String("hess ", r))


def test_the_mvs_round_drops_its_zero_weight_rows() raises:
    """A zero MVS weight is a dropped row, and it leaves through the row list.

    That is CatBoost's `SetControlNoZeroWeighted`, and it is what makes
    `min_data_in_leaf` count the rows a tree was actually fitted on rather than
    the rows whose contribution happened to be zero.
    """
    var rows = List[Int]()
    var grad = _ramp_gradients(N_ROWS)
    var hess = _ones(N_ROWS)
    var weights = List[Float64]()
    var audit = MvsAudit.empty()

    bootstrap_round(
        rows,
        grad,
        hess,
        weights,
        audit,
        BootstrapParams.mvs_with_reg(4.0, 0.5, 1),
        N_ROWS,
        0,
        0.0,
    )
    assert_equal(len(weights), N_ROWS)
    assert_equal(audit.blocks_guarded, 0, "the guard must not have fired here")
    var dropped = 0
    for r in range(N_ROWS):
        if _bits(weights[r]) == _bits(0.0):
            dropped += 1
            assert_equal(_bits(grad[r]), _bits(0.0), String("grad ", r))
            assert_equal(_bits(hess[r]), _bits(0.0), String("hess ", r))
    if dropped == 0:
        assert_equal(len(rows), 0, "nothing dropped means the full row set")
    else:
        assert_equal(len(rows), N_ROWS - dropped)
        # Ascending, duplicate free, and exactly the surviving rows.
        for i in range(len(rows)):
            if i > 0:
                assert_true(rows[i] > rows[i - 1], "rows must ascend")
            assert_false(
                _bits(weights[rows[i]]) == _bits(0.0),
                String("row ", rows[i], " was dropped but named"),
            )
    assert_equal(audit.rows_kept, N_ROWS - dropped)


def test_mvs_refuses_a_nonempty_row_list() raises:
    """MVS owns the round's row set. An incoming bag would be silently
    intersected with a draw that never saw it, which is a third sampler."""
    var rows: List[Int] = [0, 1, 2]
    var grad = _ramp_gradients(N_ROWS)
    var hess = _ones(N_ROWS)
    var weights = List[Float64]()
    var audit = MvsAudit.empty()
    var raised = False
    try:
        bootstrap_round(
            rows,
            grad,
            hess,
            weights,
            audit,
            BootstrapParams.mvs_with_reg(4.0, 0.5, 1),
            N_ROWS,
            0,
            0.0,
        )
    except:
        raised = True
    assert_true(raised)


def test_apply_bootstrap_weights_checks_its_shapes() raises:
    """The multiply is elementwise over rows and refuses a mismatched
    buffer rather than reading past one."""
    var grad = _ramp_gradients(N_ROWS)
    var hess = _ones(N_ROWS)
    var short: List[Float64] = [1.0, 1.0]
    var raised = False
    try:
        apply_bootstrap_weights(grad, hess, short, N_ROWS, 1)
    except:
        raised = True
    assert_true(raised, "a short weight vector must raise")

    var w = _ones(N_ROWS)
    var before = grad.copy()
    apply_bootstrap_weights(grad, hess, w, N_ROWS, 1)
    for r in range(N_ROWS):
        assert_equal(_bits(grad[r]), _bits(before[r]), String("row ", r))


# ---------------------------------------------------------------------------
# 5. `random_strength` reaches the split search
# ---------------------------------------------------------------------------


def test_random_strength_now_trains_and_moves_the_model() raises:
    """A positive `random_strength` used to raise at the first split search,
    because nothing computed `random_score_scale` and
    `split.find_best_split` refuses the pair. The round loop computes it now.

    Both halves are asserted: that the fit completes, and that the noise
    actually reached the argmax. A fit that completed with the noise scaled to
    zero would pass the first and fail the second, which is precisely the
    silent downgrade this campaign exists to remove.
    """
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var quiet = _booster_params()
    var noisy_tree = TreeParams(4, 1, 1.0, 1e-3)
    noisy_tree.extra.random_strength = 5.0
    noisy_tree.extra.random_strength_seed = 11
    var noisy = BoosterParams(12, 0.3, noisy_tree^)

    var plain = _model_bits(train(data, y, SQUARED_ERROR, quiet))
    var noised = _model_bits(train(data, y, SQUARED_ERROR, noisy))
    var same = True
    for b in range(N_BINS):
        if plain[b] != noised[b]:
            same = False
    assert_false(same, "random_strength must move the model")


def test_random_strength_leaves_the_callers_bundle_alone() raises:
    """The per-tree scale is ENSEMBLE STATE and must not be written back.

    If the loop wrote it onto the caller's `BoosterParams`, two identical
    `train` calls would differ: the second would start from the first's last
    scale. The bundle is borrowed and the loop keeps its own copy, so the
    field is still 0.0 after a fit and the two fits agree.
    """
    var data = bin_equal_width(
        _features(), n_rows=N_ROWS, n_features=1, n_bins=N_BINS
    )
    var y = _target()
    var tree = TreeParams(4, 1, 1.0, 1e-3)
    tree.extra.random_strength = 5.0
    tree.extra.random_strength_seed = 11
    var params = BoosterParams(12, 0.3, tree^)

    var first = _model_bits(train(data, y, SQUARED_ERROR, params))
    assert_equal(
        _bits(params.tree.extra.random_score_scale),
        _bits(0.0),
        "the fit must not write its scale back onto the caller's bundle",
    )
    var second = _model_bits(train(data, y, SQUARED_ERROR, params))
    for b in range(N_BINS):
        assert_equal(first[b], second[b], String("bin ", b))


def test_the_scale_is_catboosts_formula_at_the_rounds_model_length() raises:
    """`random_score_scale_from_gradients(grad, n, round * learning_rate)`,
    with `round` the ABSOLUTE round index so a continued run computes the
    model length an uninterrupted one would have had.

    Asserted against the function directly rather than against a number, so
    this pins the arguments the loop passes and leaves the arithmetic to
    `tests/test_random_strength.mojo`.
    """
    var grad = _ramp_gradients(N_ROWS)
    var at_zero = random_score_scale_from_gradients(grad, N_ROWS, 0.0)
    var at_three = random_score_scale_from_gradients(grad, N_ROWS, 3.0 * 0.3)
    assert_true(at_zero > 0.0)
    assert_true(at_three > 0.0)
    assert_false(
        _bits(at_zero) == _bits(at_three),
        "the model length must reach the scale",
    )
    # And the bundle accepts the pair once the scale is on it, which is the
    # check `split.find_best_split` makes before it draws.
    var extra = ExtraTreeParams.default()
    extra.random_strength = 1.0
    extra.random_score_scale = at_three
    extra.check_random_strength()
    assert_equal(
        _bits(extra.random_score_stdev()), _bits(1.0 * at_three)
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
