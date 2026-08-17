"""Gradient boosting loop.

Trains an additive ensemble of leaf-wise trees on second-order gradients.
Objectives: squared error (regression), binary logistic (labels in {0, 1}),
poisson counts, huber, quantile, L1 (mean absolute error), gamma, tweedie,
MAPE, fair, cross entropy (labels anywhere in [0, 1]), and multiclass
softmax (labels in 0..n_classes-1, via train_multiclass).

Caller-supplied objectives live in objective.mojo (`train_custom`); a
booster trained that way carries the CUSTOM objective code.

QUANTILE, L1, and MAPE follow LightGBM's RenewTreeOutput: after each tree
is grown, every leaf's Newton value is replaced by the alpha-percentile
(median for L1 and MAPE) of the residuals of the rows in that leaf, and
shrinkage is applied to the renewed value. `objective_renews_leaves`,
`renewal_alpha`, and `renewal_weights` are the three pieces of that rule,
so every trainer that grows a tree applies it the same way.

`alpha` is the objective's scalar parameter, LightGBM's `alpha` where
LightGBM has one and its per-objective parameter otherwise:

- QUANTILE: the target quantile, LightGBM's `alpha` (default 0.9).
- HUBER: the transition point, LightGBM's `alpha` (default 0.9).
- FAIR: the `c` of the fair loss, LightGBM's `fair_c` (default 1.0).
- TWEEDIE: the variance power in (1, 2), LightGBM's
  `tweedie_variance_power` (default 1.5).

Every other objective ignores it. One slot rather than four keeps the
trainer signatures, the C ABI, and the serialized model unchanged as
objectives are added; the Python layer keeps LightGBM's parameter names
and maps them here.
"""

from std.math import exp, fma, log
from std.os import getenv

from .binning import BinnedMatrix
from .efb import EfbSettings, prepare_bundling
from .parallel import (
    DispatchSettings,
    dispatch_rows,
    dispatch_rows_with,
    elementwise_row_ops,
)
from .metrics import _argsort
from .bagging import (
    BaggingParams,
    bagging_enabled,
    check_bagging,
    refresh_bag,
)
from .goss import (
    GossParams,
    GossSelection,
    apply_goss_scaling,
    goss_round,
    goss_select,
)
from .monotone import MonotoneConstraints, OutputBounds
from .ordered_boosting import (
    OrderedBoostingParams,
    check_ordered_hessian_declaration,
    check_ordered_honored,
    fold_ids,
    fold_ladder,
    n_rungs,
    ordered_permutation,
    permutation_choice,
    plane_offsets,
)
from .sampling import (
    BootstrapParams,
    ClassBaggingParams,
    MvsAudit,
    apply_bootstrap_weights,
    bootstrap_round,
    check_mvs_reg_is_set,
    has_positive_rows,
    mvs_auto_lambda_from_gradients,
    mvs_auto_lambda_from_leaf_values,
    refresh_class_bag,
)
from .objective_registry import (
    BINARY_LOGISTIC as _BINARY_LOGISTIC,
    CROSS_ENTROPY as _CROSS_ENTROPY,
    CUSTOM as _CUSTOM,
    DEFAULT_FAIR_C as _DEFAULT_FAIR_C,
    DEFAULT_TWEEDIE_VARIANCE_POWER as _DEFAULT_TWEEDIE_VARIANCE_POWER,
    FAIR as _FAIR,
    GAMMA as _GAMMA,
    HUBER as _HUBER,
    L1 as _L1,
    LINK_EXP,
    LINK_IDENTITY,
    LINK_SIGMOID,
    MAPE as _MAPE,
    MULTICLASS as _MULTICLASS,
    POISSON as _POISSON,
    QUANTILE as _QUANTILE,
    SQUARED_ERROR as _SQUARED_ERROR,
    TWEEDIE as _TWEEDIE,
    check_objective_param,
    objective_link,
    objective_renews_leaves,
)
from .cegb import CegbLedger, check_cegb_continued_training
from .histogram import (
    ConstHessianSettings,
    check_derivative_precision,
    derivative,
    derivative_precision_narrows,
    objective_has_constant_hessian,
)
from .linear_tree import (
    LinearEnsemble,
    LinearParams,
    check_linear_tree_unconnected,
)
from .phase_profile import (
    PROF_GRAD_FILL,
    PROF_SCORE_UPDATE,
    SCOPE_FIT,
    PhaseProfile,
)
from .tree import (
    GrowScratch,
    LeafMembership,
    Tree,
    TreeParams,
    grow_tree_leaves,
    grow_tree_leaves_profiled,
    grow_tree_with_cegb,
    node_bounds,
)
from .tree_parameters_extra import (
    DEFAULT_BOOST_FROM_AVERAGE,
    ExtraTreeParams,
    cap_leaf_output,
    finish_leaf_output,
    random_score_scale_from_gradients,
    raw_leaf_output,
)
from .validation import (
    check_class_code_range,
    check_class_count,
    check_column_length,
    check_weights,
)

# The objective codes, and what they mean, live in objective_registry.mojo.
# They are bound back here under the names this module has always exported,
# the way device.mojo binds device_policy.mojo's vocabulary, so every caller
# that imports them from `.boosting` keeps compiling and keeps reading the
# same values. They are bindings rather than plain re-exports so the symbols
# this module exports are defined in it, whatever an importer's view of a
# re-exported name turns out to be.
#
# They moved so that this module can import the registry. While they lived
# here the registry had to import this file, so this file could not import
# the registry, so `Booster.response` had to carry its own copy of the
# inverse-link table. It no longer does; see `response`.

comptime SQUARED_ERROR = _SQUARED_ERROR
comptime BINARY_LOGISTIC = _BINARY_LOGISTIC
comptime POISSON = _POISSON
comptime HUBER = _HUBER
comptime QUANTILE = _QUANTILE
comptime L1 = _L1

# Marks a booster trained through `train_custom` in objective.mojo. It is not
# a built-in objective: `train` and `train_gpu` reject it, because the
# gradients come from a caller-supplied callable rather than from
# `_fill_grad_hess`. Predictions for it are raw scores (no known link).
comptime CUSTOM = _CUSTOM

# 7 is LAMBDARANK, in ranking.mojo: it continues this one objective registry
# but its gradients come from query groups rather than from
# `_fill_grad_hess`, so it lives with the rest of the ranking code.

# The regression family LightGBM calls gamma, tweedie, mape, and fair, and
# the continuous-label cross entropy it calls xentropy.
comptime GAMMA = _GAMMA
comptime TWEEDIE = _TWEEDIE
comptime MAPE = _MAPE
comptime FAIR = _FAIR
comptime CROSS_ENTROPY = _CROSS_ENTROPY

# LightGBM's poisson_max_delta_step: the hessian is exp(raw + this), which
# caps the Newton step for rows with tiny predicted means. This one is not
# metadata and stays here: it is a term in the hessian, not a fact about the
# objective a caller could ask for.
comptime _POISSON_MAX_DELTA_STEP = 0.7

# LightGBM's fair_c and tweedie_variance_power defaults, the value `alpha`
# takes for FAIR and TWEEDIE when a caller does not set one.
comptime DEFAULT_FAIR_C = _DEFAULT_FAIR_C
comptime DEFAULT_TWEEDIE_VARIANCE_POWER = _DEFAULT_TWEEDIE_VARIANCE_POWER


def _sign(x: Float64) -> Float64:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


def _percentile(values: List[Float64], alpha: Float64) -> Float64:
    """LightGBM's PercentileFun: linear interpolation at position
    (n - 1) * alpha of the ascending sorted values."""
    var n = len(values)
    if n <= 1:
        return values[0]
    var order = _argsort(values)
    var float_pos = Float64(n - 1) * alpha
    var pos = Int(float_pos)
    if pos >= n - 1:
        return values[order[n - 1]]
    var bias = float_pos - Float64(pos)
    var v1 = values[order[pos]]
    var v2 = values[order[pos + 1]]
    return v1 + (v2 - v1) * bias


def _weighted_percentile(
    values: List[Float64], weights: List[Float64], alpha: Float64
) -> Float64:
    """LightGBM's WeightedPercentileFun: stable-sort by value, walk the
    weighted cdf to threshold = alpha * total weight, and interpolate
    between the straddling values only when their cdf gap is at least
    1.0 (otherwise the lower value is returned)."""
    var n = len(values)
    if n <= 1:
        return values[0]
    var order = _argsort(values)
    var cdf = List[Float64](capacity=n)
    var total = 0.0
    for i in range(n):
        total += weights[order[i]]
        cdf.append(total)
    var threshold = total * alpha
    var pos = n
    for i in range(n):
        if cdf[i] > threshold:
            pos = i
            break
    if pos > n - 1:
        pos = n - 1
    if pos == 0 or pos == n - 1:
        return values[order[pos]]
    var v1 = values[order[pos - 1]]
    var v2 = values[order[pos]]
    if cdf[pos] - cdf[pos - 1] >= 1.0:
        return (
            (threshold - cdf[pos - 1]) / (cdf[pos] - cdf[pos - 1]) * (v2 - v1)
            + v1
        )
    return v1


def _sigmoid(x: Float64) -> Float64:
    if x >= 0.0:
        var e = exp(-x)
        return 1.0 / (1.0 + e)
    var e = exp(x)
    return e / (1.0 + e)


def _clamp_prob(p: Float64) -> Float64:
    if p < 1e-15:
        return 1e-15
    if p > 1.0 - 1e-15:
        return 1.0 - 1e-15
    return p


@always_inline
def _mape_label_weight(y: Float64) -> Float64:
    """LightGBM's MAPE label weight `1 / max(1, |y|)`, the per-row scale that
    turns an absolute error into a relative one. The floor at 1 is
    LightGBM's: without it a label near zero would dominate the objective."""
    var m = abs(y)
    return 1.0 / m if m > 1.0 else 1.0


def _check_objective(
    objective: Int, target: List[Float64], alpha: Float64
) raises:
    if objective == CUSTOM:
        raise Error(
            "custom objectives must be trained with train_custom (or"
            " train_custom_with_valid / fit_custom / train_custom_gpu)"
        )
    if (
        objective != SQUARED_ERROR
        and objective != BINARY_LOGISTIC
        and objective != POISSON
        and objective != HUBER
        and objective != QUANTILE
        and objective != L1
        and objective != GAMMA
        and objective != TWEEDIE
        and objective != MAPE
        and objective != FAIR
        and objective != CROSS_ENTROPY
    ):
        raise Error("unknown objective")
    if objective == POISSON:
        for r in range(len(target)):
            if target[r] < 0.0:
                raise Error("poisson target values must be nonnegative")
    if objective == TWEEDIE:
        for r in range(len(target)):
            if target[r] < 0.0:
                raise Error("tweedie target values must be nonnegative")
    if objective == GAMMA:
        # The gamma deviance has a log(y) in it and its gradient divides by
        # the prediction; a zero or negative label has no gamma likelihood.
        for r in range(len(target)):
            if target[r] <= 0.0:
                raise Error("gamma target values must be positive")
    if objective == CROSS_ENTROPY:
        for r in range(len(target)):
            if target[r] < 0.0 or target[r] > 1.0:
                raise Error("cross entropy target values must be in [0, 1]")
    # The four scalar-parameter ranges, and their sentences, are
    # `check_objective_param` in objective_registry.mojo, over the intervals
    # `objective_param_domain` states. They used to be spelled out here as
    # well; a caller that wants the bounds without a training run needs them
    # as data, and two statements of a bound is one too many.
    check_objective_param(objective, alpha)


def _check_sample_weight(weights: List[Float64], n: Int) raises:
    """An empty list means unweighted; otherwise one nonnegative weight
    per training row."""
    if len(weights) == 0:
        return
    if len(weights) != n:
        raise Error("sample_weight length must equal n_rows")
    # Finite, nonnegative, positive sum: `validation.check_weights` is the
    # rule; the length message above is kept because callers match on it.
    _ = check_weights(weights, n)


def _check_goss(goss: GossParams, bagging: BaggingParams) raises:
    """Validate the GOSS rates and reject GOSS together with row bagging.

    Both strategies own the row list a tree is grown on, and LightGBM makes
    them exclusive too: it silently turns bagging off under GOSS. mojotrees
    raises instead, so a configuration that would quietly ignore half of
    itself is reported."""
    goss.validate()
    if goss.enabled and bagging_enabled(bagging):
        raise Error("goss and bagging cannot both be enabled")


def _check_class_bagging(
    class_bagging: ClassBaggingParams,
    bagging: BaggingParams,
    goss: GossParams,
    objective: Int,
) raises:
    """Validate class-conditional bagging and reject it where LightGBM does
    not apply it.

    LightGBM's balanced bagging is a binary-classification feature and its
    `pos_bagging_fraction`/`neg_bagging_fraction` are read only there; it also
    owns the row list a tree is grown on, so it cannot share a run with
    uniform bagging or with GOSS. LightGBM ignores the settings in those
    cases. mojotrees raises instead, on the same reasoning as `_check_goss`:
    a configuration that would quietly drop half of itself is reported.
    """
    class_bagging.validate()
    if not class_bagging.enabled():
        return
    if bagging_enabled(bagging):
        raise Error(
            "pos_bagging_fraction/neg_bagging_fraction and bagging_fraction"
            " cannot both be enabled; both own the row list a tree is grown"
            " on"
        )
    if goss.enabled:
        raise Error(
            "pos_bagging_fraction/neg_bagging_fraction and goss cannot both"
            " be enabled; both own the row list a tree is grown on"
        )
    if objective != BINARY_LOGISTIC:
        raise Error(
            "pos_bagging_fraction/neg_bagging_fraction apply to binary"
            " classification only, as in LightGBM"
        )


def _check_bootstrap(
    bootstrap: BootstrapParams,
    bagging: BaggingParams,
    goss: GossParams,
    class_bagging: ClassBaggingParams,
    objective: Int,
) raises:
    """Validate `bootstrap_type` and reject it beside the other row samplers.

    `BootstrapParams.validate` carries CatBoost's own rules and the one
    exclusion between MVS and the Bayesian bootstrap. What is added here is the
    exclusion against the three samplers mojotrees already had, on exactly the
    reasoning `_check_goss` and `_check_class_bagging` give: each of them owns
    the row list a tree is grown on, MVS owns it too (it drops its
    zero-weight rows outright, `sampling.mvs_kept_rows`), and a configuration
    that would quietly ignore half of itself is reported rather than run.

    Row bagging is the sharpest case and is worth naming rather than lumping
    in: **it is CatBoost's Bernoulli bootstrap under mojotrees's own name**
    (`sampling.canonical_bootstrap_type` refuses the spelling `bernoulli` for
    precisely that reason), so `bagging_fraction` beside `bootstrap_type=MVS`
    is two values of one CatBoost enum at once. The Bayesian bootstrap keeps
    every row and could in principle compose with a bag, but CatBoost cannot
    spell that combination either and nothing here is calibrated against it, so
    it is refused with the rest instead of being the one untested crossing.

    `objective` is refused for `CUSTOM` alone: `train_custom` never reaches the
    loops that call `bootstrap_round`, and a callback's derivatives are
    whatever the caller returns, so the draw's row magnitudes would be scaled
    by an unknown convention.
    """
    bootstrap.validate()
    if not bootstrap.enabled():
        return
    if bagging_enabled(bagging):
        raise Error(
            "bootstrap_type and bagging_fraction cannot both be enabled:"
            " bagging_fraction IS CatBoost's Bernoulli bootstrap under"
            " mojotrees's name, so this asks for two bootstrap types at once,"
            " and both own the row list a tree is grown on"
        )
    if goss.enabled:
        raise Error(
            "bootstrap_type and goss cannot both be enabled; both sample rows"
            " by gradient magnitude and both own the row list a tree is grown"
            " on"
        )
    if class_bagging.enabled():
        raise Error(
            "bootstrap_type and pos_bagging_fraction/neg_bagging_fraction"
            " cannot both be enabled; both own the row list a tree is grown on"
        )
    if objective == CUSTOM:
        raise Error(
            "bootstrap_type is not applied to a custom objective: the draw"
            " scales the callback's own derivatives, whose units and hessian"
            " convention this loop does not know"
        )


# `objective_renews_leaves` is objective_registry's rule (LightGBM's
# `RenewTreeOutput`), imported above and re-exported from here under the
# name the boosting loop and its callers have always used; the leaf renewal
# it decides is `_renew_leaf_values` below.


def _check_bootstrap_multiclass(
    bootstrap: BootstrapParams,
    bagging: BaggingParams,
    goss: GossParams,
    where: String,
) raises:
    """`_check_bootstrap` for the softmax loops, plus the one refusal that is
    theirs alone.

    The three row-sampler exclusions are exactly `_check_bootstrap`'s and are
    delegated to it rather than restated, so the messages a multiclass user
    reads are the messages a single-output user reads. `ClassBaggingParams` is
    passed disabled because balanced bagging is binary-only
    (`_check_class_bagging` restricts it) and a softmax loop never builds one,
    and the objective is passed as `MULTICLASS` -- honest, and not `CUSTOM`,
    which is the only code `_check_bootstrap` tests for.

    What is added is `check_mvs_reg_is_set`: MVS with a **derived** lambda
    reads the previous iteration's `[dim][leaf]` table, which a round of `K`
    structurally different trees does not have. See that function for the
    CatBoost lines. It is the reason `catboost_default_bootstrap_type` exists at
    all -- CatBoost defaults these losses to the Bayesian bootstrap, which
    needs no lambda, so the DEFAULT never reaches this refusal and only an
    explicit `bootstrap_type='mvs'` without `mvs_reg` does.
    """
    _check_bootstrap(
        bootstrap, bagging, goss, ClassBaggingParams.disabled(), _MULTICLASS
    )
    check_mvs_reg_is_set(bootstrap, where)


def renewal_alpha(objective: Int, alpha: Float64) -> Float64:
    """The percentile leaf renewal takes for this objective: the target
    quantile for QUANTILE, the median for L1 and MAPE."""
    return alpha if objective == QUANTILE else 0.5


def renewal_weights(
    objective: Int, target: List[Float64], weights: List[Float64]
) -> List[Float64]:
    """The weights leaf renewal takes for this objective.

    Every objective but MAPE renews on the sample weights themselves.
    MAPE renews on `sample_weight * 1 / max(1, |y|)`, the same product that
    scales its gradient, which is LightGBM's `RegressionMAPELOSS`
    `RenewTreeOutput` override: the leaf value has to be the *relative*
    median, and a weighted percentile with those weights is it.

    Computed once per training run rather than once per tree, since neither
    the labels nor the sample weights change between rounds.
    """
    if objective != MAPE:
        return weights.copy()
    var out = List[Float64](capacity=len(target))
    for r in range(len(target)):
        var w = weights[r] if len(weights) > 0 else 1.0
        out.append(w * _mape_label_weight(target[r]))
    return out^


def _base_score(
    target: List[Float64],
    objective: Int,
    weights: List[Float64],
    alpha: Float64,
    boost_from_average: Bool = DEFAULT_BOOST_FROM_AVERAGE,
) raises -> Float64:
    # `boost_from_average=False` starts every row at 0.0 instead of at the
    # optimal constant. It returns before a label is read, which is what makes
    # it an off switch on the existing mechanism rather than a second one: the
    # True branch below is byte for byte the function this was before the
    # argument existed, and the argument defaults to True, so every caller that
    # does not pass it computes what it always computed.
    #
    # The defaulted argument is deliberate and is not an accept-and-ignore: a
    # trainer that has not been given the flag cannot silently start from zero,
    # because zero is only reachable by passing False. The entry points whose
    # trainers do not thread it refuse `False` by name through
    # `_refuse_boost_from_average` instead.
    if not boost_from_average:
        return 0.0
    # QUANTILE, L1, and MAPE boost from the target percentile, LightGBM
    # style; every other objective boosts from the (weighted) mean. MAPE
    # takes the median under its own label weights, the same weights its
    # leaf renewal uses.
    if objective_renews_leaves(objective):
        var q = renewal_alpha(objective, alpha)
        var renew_w = renewal_weights(objective, target, weights)
        if len(renew_w) > 0:
            return _weighted_percentile(target, renew_w, q)
        return _percentile(target, q)
    var mean = 0.0
    var total_w = 0.0
    for r in range(len(target)):
        var w = weights[r] if len(weights) > 0 else 1.0
        mean += w * target[r]
        total_w += w
    if total_w <= 0.0:
        raise Error("sample_weight must have a positive sum")
    mean /= total_w
    if objective == BINARY_LOGISTIC or objective == CROSS_ENTROPY:
        var p = _clamp_prob(mean)
        return log(p / (1.0 - p))
    if objective == POISSON:
        if mean <= 0.0:
            raise Error("poisson requires a positive mean target")
        return log(mean)
    if objective == GAMMA or objective == TWEEDIE:
        # Both have the exp link, so both start at the log of the mean, as
        # in LightGBM where they inherit RegressionPoissonLoss's
        # BoostFromScore.
        if mean <= 0.0:
            raise Error(
                "gamma and tweedie require a positive mean target"
            )
        return log(mean)
    return mean


def fill_grad_hess(
    raw: List[Float64],
    target: List[Float64],
    objective: Int,
    weights: List[Float64],
    alpha: Float64,
    mut grad: List[Float64],
    mut hess: List[Float64],
    settings: DispatchSettings = DispatchSettings.unresolved(),
    float64_derivatives: Bool = False,
) raises:
    """Per-row first and second derivatives of `objective` at the current raw
    scores, written into `grad` and `hess`.

    Both buffers are resized to `len(target)` and fully overwritten, so the
    boosting loop hands the same two lists back every round and the training
    run allocates them once. Rows are independent, so the work splits across
    contiguous row blocks; each row's arithmetic is unchanged and no value is
    summed across rows, which makes the output bit-identical at every worker
    count.

    **Every value written here is rounded to single precision**, through
    `histogram.score_t`. LightGBM's `score_t` is `float`
    (`include/LightGBM/meta.h`) and every derivative it carries -- from the
    objective's output, through `ordered_gradients`, into the histogram
    accumulate -- is one. Raw scores, leaf values and gains stay Float64 on
    both sides; only the per-row derivative narrows.

    The containers stay `List[Float64]` because their type is fixed by
    signatures in `tree.mojo`, so what lands here is a Float32 quantity in a
    Float64 word. That is not a half measure, it is what makes the histogram's
    gathered pair buffer able to hold two Float32 per row without the gathered
    and un-gathered accumulation paths disagreeing: the narrowing is
    idempotent, both paths apply it, and both therefore add the identical
    Float64. `histogram.score_t` carries the full argument.

    Two consequences worth stating. Every derivative-dependent number in a fit
    moves -- a gradient's low 29 significand bits are now zero -- which is the
    accuracy LightGBM has always had rather than a loss against it. And the
    constant-hessian guarantee is untouched: `histogram.CONSTANT_HESSIAN` is
    1.0, exactly representable in Float32, and the unweighted arms of
    `SQUARED_ERROR`, `L1`, `HUBER` and `QUANTILE` still write exactly it.

    **`derivative_precision` decides whether it narrows at all.** The
    narrowing is the default and is what the paragraphs above describe;
    `float64` stores what the objective computed. The decision is taken here
    **once per call** -- which is once per round, not once per node and not
    once per row -- and it then selects between two compile-time
    instantiations of the row loops below, so neither arm's loop carries a
    test the other put there. This is the objective side of the switch;
    `histogram.ConstHessianSettings.narrow` is the read side, and it is a
    snapshot field rather than a live read for the reason stated there.

    **Both entries are honored here, and `float64` wins from either.**
    `float64_derivatives` is the parameter entry
    (`ExtraTreeParams.wants_float64_derivatives()`), forwarded by every
    trainer; the environment entry is read live. The precedence is
    `histogram.ConstHessianSettings.widened`'s and is deliberately the same
    one, because the objective side and the read side of a single switch
    disagreeing about which entry wins is worse than either rule.

    Reading the environment live here is what makes the setting reach a fit
    that never resolves a snapshot: this entry is on every trainer's path.
    Taking the parameter as an argument is what makes it reach a fit whose
    caller never touched the environment, and the two together are what
    lifted the refusal in
    `ExtraTreeParams.check_derivative_precision`. **A trainer that forgets to
    forward it gets the environment-only behavior**, which is the old,
    documented behavior and not a wrong answer -- but it is a fit that
    ignores a parameter it accepted, so the default here is `False` rather
    than something cleverer precisely so that a missed call site is a missing
    feature and never a silent numerical difference.

    Public because it is the gradient-generation stage the CPU profiler times
    (bench/bench_profile.mojo); training calls it through the same entry.
    """
    var n = len(target)
    if len(raw) < n:
        raise Error("raw scores must be at least as long as the target")
    if len(weights) > 0 and len(weights) < n:
        raise Error("sample_weight must be at least as long as the target")
    if len(grad) != n:
        grad.resize(n, 0.0)
    if len(hess) != n:
        hess.resize(n, 0.0)
    check_derivative_precision()
    if derivative_precision_narrows() and not float64_derivatives:
        _fill_grad_hess_into[True](
            grad, hess, raw, target, objective, weights, alpha, n, settings
        )
    else:
        _fill_grad_hess_into[False](
            grad, hess, raw, target, objective, weights, alpha, n, settings
        )


def _fill_grad_hess_into[
    NARROW: Bool
](
    mut grad: List[Float64],
    mut hess: List[Float64],
    raw: List[Float64],
    target: List[Float64],
    objective: Int,
    weights: List[Float64],
    alpha: Float64,
    n: Int,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    var gp = grad.unsafe_ptr()
    var hp = hess.unsafe_ptr()
    var raw_p = raw.unsafe_ptr()
    var tgt_p = target.unsafe_ptr()
    var w_p = weights.unsafe_ptr()
    var weighted = len(weights) > 0

    # The objective test is hoisted out of the row loop, so a block runs one
    # straight-line body rather than re-dispatching per row.
    def block(start: Int, end: Int) {imm}:
        if objective == BINARY_LOGISTIC or objective == CROSS_ENTROPY:
            # One branch for both: cross entropy is the logistic loss with
            # the {0, 1} label relaxed to any probability in [0, 1], so its
            # derivatives are the same expression. Only the label
            # validation and the reported metric differ.
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var p = _sigmoid(raw_p.unsafe_load(r))
                gp.unsafe_store(r, derivative[NARROW](w * (p - tgt_p.unsafe_load(r))))
                var h = p * (1.0 - p)
                if h < 1e-16:
                    h = 1e-16
                hp.unsafe_store(r, derivative[NARROW](w * h))
        elif objective == GAMMA:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                # d/draw of raw + y * exp(-raw), the gamma deviance under
                # the log link.
                var y_over_mu = tgt_p.unsafe_load(r) * exp(
                    -raw_p.unsafe_load(r)
                )
                gp.unsafe_store(r, derivative[NARROW](w * (1.0 - y_over_mu)))
                hp.unsafe_store(r, derivative[NARROW](w * y_over_mu))
        elif objective == TWEEDIE:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var raw_r = raw_p.unsafe_load(r)
                var y = tgt_p.unsafe_load(r)
                # rho is the variance power in (1, 2), so 1 - rho < 0 and
                # 2 - rho > 0 and the hessian below stays nonnegative.
                var e1 = exp((1.0 - alpha) * raw_r)
                var e2 = exp((2.0 - alpha) * raw_r)
                gp.unsafe_store(r, derivative[NARROW](w * (-y * e1 + e2)))
                hp.unsafe_store(
                    r,
                    derivative[NARROW](
                        w * (-y * (1.0 - alpha) * e1 + (2.0 - alpha) * e2)
                    ),
                )
        elif objective == MAPE:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var y = tgt_p.unsafe_load(r)
                # L1 scaled by the label weight: the absolute error becomes
                # a relative one, which is what MAPE measures.
                var lw = w * _mape_label_weight(y)
                gp.unsafe_store(
                    r, derivative[NARROW](lw * _sign(raw_p.unsafe_load(r) - y))
                )
                hp.unsafe_store(r, derivative[NARROW](lw))
        elif objective == FAIR:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var d = raw_p.unsafe_load(r) - tgt_p.unsafe_load(r)
                var denom = abs(d) + alpha
                gp.unsafe_store(r, derivative[NARROW](w * alpha * d / denom))
                hp.unsafe_store(r, derivative[NARROW](w * alpha * alpha / (denom * denom)))
        elif objective == POISSON:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var raw_r = raw_p.unsafe_load(r)
                var mu = exp(raw_r)
                gp.unsafe_store(r, derivative[NARROW](w * (mu - tgt_p.unsafe_load(r))))
                hp.unsafe_store(
                    r, derivative[NARROW](w * exp(raw_r + _POISSON_MAX_DELTA_STEP))
                )
        elif objective == HUBER:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var diff = raw_p.unsafe_load(r) - tgt_p.unsafe_load(r)
                if abs(diff) <= alpha:
                    gp.unsafe_store(r, derivative[NARROW](w * diff))
                else:
                    gp.unsafe_store(r, derivative[NARROW](w * _sign(diff) * alpha))
                hp.unsafe_store(r, derivative[NARROW](w))
        elif objective == QUANTILE:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var diff = raw_p.unsafe_load(r) - tgt_p.unsafe_load(r)
                if diff >= 0.0:
                    gp.unsafe_store(r, derivative[NARROW](w * (1.0 - alpha)))
                else:
                    gp.unsafe_store(r, derivative[NARROW](w * -alpha))
                hp.unsafe_store(r, derivative[NARROW](w))
        elif objective == L1:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                gp.unsafe_store(
                    r,
                    derivative[NARROW](
                        w * _sign(raw_p.unsafe_load(r) - tgt_p.unsafe_load(r))
                    ),
                )
                hp.unsafe_store(r, derivative[NARROW](w))
        else:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                gp.unsafe_store(
                    r,
                    derivative[NARROW](
                        w * (raw_p.unsafe_load(r) - tgt_p.unsafe_load(r))
                    ),
                )
                hp.unsafe_store(r, derivative[NARROW](w))

    # A row here is a handful of flops on three sequential arrays, perhaps a
    # sixteenth of the cost of a histogram op's scattered read-modify-write,
    # so the work estimate is scaled down to match. The scale matters: with
    # the unscaled count, 100k rows asked for one task per core and
    # bench/bench_profile.mojo timed the fan-out well below the serial path,
    # the work per task being far too small to pay for scheduling it.
    dispatch_rows_with(settings, block, n, elementwise_row_ops(n))


def _fill_grad_hess(
    raw: List[Float64],
    target: List[Float64],
    objective: Int,
    weights: List[Float64],
    alpha: Float64,
    mut grad: List[Float64],
    mut hess: List[Float64],
    settings: DispatchSettings = DispatchSettings.unresolved(),
    float64_derivatives: Bool = False,
) raises:
    fill_grad_hess(
        raw, target, objective, weights, alpha, grad, hess, settings,
        float64_derivatives,
    )


def round_has_constant_hessian(
    objective: Int,
    sample_weight: List[Float64],
    goss: GossParams,
) raises -> Bool:
    """Whether a single-output round configured this way puts exactly
    `histogram.CONSTANT_HESSIAN` into every entry of `hess`, so a histogram
    builder may be told to stop accumulating that plane.

    This is the trainers' binding of
    `histogram.objective_has_constant_hessian`, and the reason it exists as a
    function rather than as an expression repeated at each round loop is that
    it is a **safety predicate**: an overinclusive answer produces a silently
    wrong hessian plane in the histograms a fit's splits are chosen from, with
    no exception raised and no metric moved far enough to notice. One
    definition, used by the CPU trainers here and by the GPU trainers in
    `train_gpu.mojo`, is one thing to review and one thing to change if an
    objective's derivatives ever move.

    Three exclusions, in the order they bite:

    - **Sample weights.** `_fill_grad_hess_into` ends the row body of every
      constant-hessian arm with `hp.unsafe_store(r, w)`, where `w` is
      `weights[r]` when the fit is weighted, so the hessian *is* the weight and
      a weighted fit has a per-row hessian by construction. The device kernel
      in `gpu_objectives_native.mojo` writes the same `w` in the same arms, so
      this holds on both backends. `len(sample_weight) > 0` is the whole test
      at this level because a class weight, `scale_pos_weight`, and
      `is_unbalance` are all expanded into an ordinary per-row `sample_weight`
      by `class_weight.mojo` before any trainer here is called; nothing reaches
      these loops as a weight that is not in this list.
    - **GOSS.** `goss.apply_goss_scaling` multiplies the sampled small-gradient
      rows' hessians by the amplification factor and leaves the top rows at
      1.0, so a GOSS round holds two distinct hessian values under an objective
      whose code says otherwise. That is invisible to
      `objective_has_constant_hessian`, which sees only the objective, and it is
      the reason the declaration has to be made here. `goss.enabled` is used
      rather than `goss.active(round, learning_rate)`: a configured GOSS run has
      warmup rounds during which nothing is rescaled and the hessians really
      are constant, but the declaration these trainers make is held for a whole
      fit (on the GPU it is builder state set once), so a run that will rescale
      on any round must be excluded on all of them. That costs the
      specialization on the warmup rounds of a GOSS fit and buys a declaration
      that cannot go stale part way through a loop.
    - Every objective whose curvature depends on the raw score or the label,
      which `objective_has_constant_hessian` itself rejects. MAPE is the one
      worth naming, because it is the near miss: it is L1 scaled by
      `1 / max(1, |y|)`, so it stores a *label-dependent* hessian even
      unweighted, and it is excluded there for exactly that reason.

    Row bagging is deliberately **not** an exclusion, and that is the one
    judgment in this function rather than a transcription. A bag restricts
    which rows are accumulated; it does not touch `hess`, so every row a bagged
    histogram visits still carries 1.0 and the reconstruction from the count is
    the same integer it would have been. Balanced class bagging
    (`sampling.refresh_class_bag`) is a different draw of the same kind and is
    likewise not an exclusion. If either sampler ever grew a per-row weight,
    this reasoning would fail and this is where it would have to be revisited.

    **A fourth exclusion exists and is not visible from here.** CatBoost's
    Bayesian bootstrap (`sampling.BayesianBootstrapParams`) keeps every row and
    gives each a random weight per tree, and that weight multiplies the row's
    derivatives exactly as a `sample_weight` does, so a bootstrapped fit has a
    per-row hessian under every objective. It is the case the paragraph above
    predicted. This function cannot test for it, because the configuration is
    not among its three inputs and widening the signature would change a
    contract the device trainers bind to; so a caller that turns the bootstrap
    on carries the exclusion itself, either by passing the effective per-row
    weights (the bootstrap draw times the user's weights, which is the vector
    `sampling.refresh_bayesian_bootstrap` builds and what CatBoost's
    `CalcWeightedData` computes) as `sample_weight` -- in which case the first
    exclusion above already refuses -- or by calling
    `sampling.check_bayesian_bootstrap_hessian_declaration` on whatever this
    returned, which raises rather than letting the two-plane path be taken by
    omission. No trainer in the repository enables it today; the wiring lane
    that does must do one of those two things.

    A custom objective is excluded by `objective_has_constant_hessian`
    returning False for `CUSTOM`, but no trainer should rely on that alone: the
    callback's hessians are whatever the caller returns, and
    `train_custom`/`train_custom_gpu` never call this at all.
    """
    if goss.enabled:
        return False
    return objective_has_constant_hessian(objective, len(sample_weight) > 0)


def _mean_loss(
    raw: List[Float64], target: List[Float64], objective: Int, alpha: Float64
) -> Float64:
    var total = 0.0
    if objective == BINARY_LOGISTIC:
        for r in range(len(target)):
            var p = _clamp_prob(_sigmoid(raw[r]))
            if target[r] > 0.5:
                total -= log(p)
            else:
                total -= log(1.0 - p)
    elif objective == CROSS_ENTROPY:
        # The same log loss with a continuous label, so both terms count on
        # every row rather than one being selected by a {0, 1} test.
        for r in range(len(target)):
            var p = _clamp_prob(_sigmoid(raw[r]))
            total -= (
                target[r] * log(p) + (1.0 - target[r]) * log(1.0 - p)
            )
    elif objective == GAMMA:
        # Gamma negative log likelihood up to a constant in the target:
        # raw + y * exp(-raw), the loss the gradient above differentiates.
        for r in range(len(target)):
            total += raw[r] + target[r] * exp(-raw[r])
    elif objective == TWEEDIE:
        # Tweedie negative log likelihood up to a constant in the target.
        for r in range(len(target)):
            total += (
                -target[r] * exp((1.0 - alpha) * raw[r]) / (1.0 - alpha)
                + exp((2.0 - alpha) * raw[r]) / (2.0 - alpha)
            )
    elif objective == MAPE:
        for r in range(len(target)):
            total += (
                abs(raw[r] - target[r]) * _mape_label_weight(target[r])
            )
    elif objective == FAIR:
        for r in range(len(target)):
            var d = abs(raw[r] - target[r]) / alpha
            total += alpha * alpha * (d - log(1.0 + d))
    elif objective == POISSON:
        # Poisson negative log likelihood up to a constant in the target.
        for r in range(len(target)):
            total += exp(raw[r]) - target[r] * raw[r]
    elif objective == HUBER:
        for r in range(len(target)):
            var d = abs(raw[r] - target[r])
            if d <= alpha:
                total += 0.5 * d * d
            else:
                total += alpha * (d - 0.5 * alpha)
    elif objective == QUANTILE:
        # Pinball loss.
        for r in range(len(target)):
            var d = raw[r] - target[r]
            if d >= 0.0:
                total += (1.0 - alpha) * d
            else:
                total += -alpha * d
    elif objective == L1:
        for r in range(len(target)):
            total += abs(raw[r] - target[r])
    else:
        for r in range(len(target)):
            var d = raw[r] - target[r]
            total += d * d
    return total / Float64(len(target))


def _renewal_membership_usable(
    tree: Tree, leaves: LeafMembership, n_used: Int
) -> Bool:
    """Whether `leaves` can stand in for walking the tree once per row.

    Three tests, all O(number of leaves) and none of them touching a row:

    - the membership names as many leaves as the tree has, and at least one;
    - every name is a leaf of *this* tree (in range, and `feature < 0`);
    - the row lists together hold exactly the `n_used` rows renewal would
      otherwise iterate, which is the bag when there is one and the dataset
      when there is not.

    That is a shape check, not a provenance check: a membership left over from
    a differently-grown tree of the same shape and the same row count would
    pass it. It does not need to be more than that, because the only callers
    that pass a membership are the two round loops in this file, which pass the
    one the grower filled in for this tree two statements earlier. Every other
    caller passes nothing, gets the default empty membership, fails the first
    test, and takes the traversal it always took.
    """
    var n_leaves = leaves.n_leaves()
    if n_leaves == 0 or n_leaves != tree.n_leaves:
        return False
    var n_nodes = len(tree.feature)
    var total = 0
    for l in range(n_leaves):
        var node = leaves.node[l]
        if node < 0 or node >= n_nodes or tree.feature[node] >= 0:
            return False
        total += len(leaves.rows[l])
    return total == n_used


def _renew_leaf_values(
    mut tree: Tree,
    data: BinnedMatrix,
    target: List[Float64],
    raw: List[Float64],
    weights: List[Float64],
    alpha: Float64,
    bag: List[Int] = [],
    monotone: List[Int] = [],
    extra: ExtraTreeParams = ExtraTreeParams(),
    leaves: LeafMembership = LeafMembership(),
) raises:
    """LightGBM's RenewTreeOutput for QUANTILE and L1: replace each leaf's
    Newton value with the alpha-percentile of the residuals
    (target - current raw score) of the rows in that leaf. Runs before
    shrinkage, which then applies to the renewed value.

    A non-empty `bag` renews from bagged rows only, the rows the tree was
    grown on; LightGBM likewise renews over its bagged partition.

    A non-empty `monotone` clamps every renewed value back into its leaf's
    monotone interval, without which renewal would discard the constraint the
    tree was grown under (see monotone.mojo).

    `extra` carries `max_delta_step` and `path_smooth`, which apply to a
    renewed value as they do to a grown one: renewal replaces the Newton step
    with a percentile, and without this the two objectives that renew would
    be the only ones to escape the cap and the smoothing the caller asked
    for. The parent's output is the value its node still carries -- renewal
    rewrites leaves only, so an internal node holds the finished value it was
    grown with. The bundle defaults to inactive, so a caller that does not
    pass one renews exactly as before.

    `leaves` is the membership the grower handed back for this very tree (see
    `tree.LeafMembership`), and passing it is what stops this from walking the
    whole tree once per row to recover a leaf the partition already named. It
    is optional and defaults to empty; a caller that passes nothing takes the
    traversal unchanged. `_renewal_membership_usable` decides, and a membership
    that does not pass it falls back rather than raising.

    The two routes build the *same list*, not merely the same multiset, and
    that distinction is the whole of the argument. Renewal buckets residuals
    per leaf and then takes a percentile of each bucket, and
    `_weighted_percentile` accumulates a weighted cdf in the order `_argsort`
    leaves its input in, so a bucket permuted by a tie would give a different
    Float64. The traversal fills bucket `node` by sweeping rows in ascending
    order and appending the ones that route there, so bucket `node` comes out
    in ascending row order. `partition_rows_into` keeps each side ascending at
    every split, so `leaves.rows[l]` is ascending too, and it is exactly the
    set of rows that route to `leaves.node[l]` -- the partition and
    `leaf_index_row` apply the same three tests to the same bins. Same
    elements, same order, therefore the same bits out. The finishing loop below
    is untouched and still runs in node order, so nothing else about this
    function moves either.

    This holds under bagging as well as without it. `bag` is ascending and
    duplicate-free, growth partitions the bag rather than the dataset, and the
    traversal route iterates `bag` in order; so the per-leaf lists agree there
    for the same reason."""
    var n_nodes = len(tree.feature)
    var bounds = node_bounds(tree, monotone)
    var finish = extra.needs_leaf_finish()
    # Built before any leaf is rewritten, though renewal touches leaves only
    # and every parent is internal, so the two orders agree. Left empty, and
    # so unallocated, when there is no smoothing to do.
    var parent_output = List[Float64]()
    if finish:
        parent_output.resize(n_nodes, 0.0)
        for node in range(n_nodes):
            if tree.feature[node] < 0:
                continue
            parent_output[tree.left[node]] = tree.value[node]
            parent_output[tree.right[node]] = tree.value[node]
    var leaf_residuals = List[List[Float64]]()
    var leaf_weights = List[List[Float64]]()
    for _ in range(n_nodes):
        leaf_residuals.append(List[Float64]())
        leaf_weights.append(List[Float64]())
    var n_used = len(bag) if len(bag) > 0 else data.n_rows
    var weighted = len(weights) > 0
    if _renewal_membership_usable(tree, leaves, n_used):
        # The leaf is read off the partition instead of re-derived by walking
        # the tree. Each list is also sized once, at its exact final length,
        # rather than doubled into: a leaf holding a million rows costs one
        # allocation here where the appending form costs about twenty and
        # copies about two million Float64 moving between them. Reserving
        # changes no element and no order.
        for l in range(leaves.n_leaves()):
            var node = leaves.node[l]
            ref rows = leaves.rows[l]
            var n_l = len(rows)
            leaf_residuals[node].reserve(n_l)
            if weighted:
                leaf_weights[node].reserve(n_l)
            for i in range(n_l):
                var r = rows[i]
                leaf_residuals[node].append(target[r] - raw[r])
                if weighted:
                    leaf_weights[node].append(weights[r])
    else:
        for i in range(n_used):
            var r = bag[i] if len(bag) > 0 else i
            var node = tree.leaf_index_row(data, r)
            leaf_residuals[node].append(target[r] - raw[r])
            if weighted:
                leaf_weights[node].append(weights[r])
    for node in range(n_nodes):
        if tree.feature[node] >= 0 or len(leaf_residuals[node]) == 0:
            continue
        var renewed: Float64
        if len(weights) > 0:
            renewed = _weighted_percentile(
                leaf_residuals[node], leaf_weights[node], alpha
            )
        else:
            renewed = _percentile(leaf_residuals[node], alpha)
        if finish:
            renewed = finish_leaf_output(
                renewed,
                extra.max_delta_step,
                extra.path_smooth,
                len(leaf_residuals[node]),
                parent_output[node],
            )
        if len(bounds) > 0:
            renewed = bounds[node].clamp(renewed)
        tree.value[node] = renewed


def catboost_leaf_estimation_iterations(objective: Int) -> Int:
    """What CatBoost would default `leaf_estimation_iterations` to for the
    CatBoost loss that corresponds to `objective`, on **CPU**.

    **A record that is now also a resolver.** `ExtraTreeParams.
    leaf_estimation_iterations` still defaults to 1 for every objective, which
    is LightGBM's behavior and this project's, and every surface except one
    still gets that 1. The exception is `bindings/_mojotrees.mojo`, which calls
    this function to resolve an UNSET `leaf_estimation_iterations` when the fit
    is in CatBoost mode (`grow_policy='symmetrictree'`) and the entry point's
    routing reaches a trainer that implements the extra steps. That is the
    standing rule -- CatBoost mode mirrors CatBoost, `lossguide` mirrors
    LightGBM -- and it is why the numbers below have to be right rather than
    merely recorded.

    **The value below is the OPTION default and CatBoost can still stomp it
    from the data.** `UpdateLeavesEstimationIterations`
    (`catboost/libs/train_lib/options_helper.cpp:290-303`, called at `:429`)
    resets the count to 1, for every loss, when the user did not set it AND the
    run has fewer than 200 iterations (`IsSmallIterationCount`,
    `catboost_options.h:88-90`) AND the pool has fewer than 20 features.
    `SetDefault` does not raise the `IsSetFlag` (`option.h:28-33`), so an
    option-resolved 10 is still "not set" as far as that later pass is
    concerned. So a 100-round CatBoost Logloss fit on a 12-feature pool takes
    ONE Newton step, not ten, and a table that reports 10 for Logloss without
    this paragraph will disagree with a live `get_all_params()` on exactly the
    small cells a benchmark suite is quickest to build. This function does not
    apply that stomp: it is keyed on an objective and knows neither the round
    count nor the feature count, and the caller in the binding has both.

    Verified from source, `github.com/catboost/catboost` at commit
    `58c7bb8be2fbd12f85aae268dd1f5cf3cb211e74`, read 2026-08-16. Note that
    CatBoost's own Python docstring (`python-package/catboost/core.py:4958`,
    "If None, then leaf_estimation_iterations=1") is stale and contradicts the
    C++ below; the C++ is what runs. The table is
    `GetEstimationMethodDefaults` in
    `catboost/private/libs/options/catboost_options.cpp`, and the value that
    takes effect is the one belonging to the loss's default
    `leaf_estimation_method`: that function sets `defaultNewtonIterations`
    *and* `defaultGradientIterations` for every loss, and
    `SetLeavesEstimationDefault` then selects between them by method. A number
    in the unselected slot is dead at the defaults.

    That selection is where the widely repeated "CatBoost takes 10 Newton
    steps for logloss and for multiclass" comes from, and it is **half wrong**:

    - `Logloss`, `CrossEntropy`, `MultiLogloss`, `MultiCrossEntropy` really do
      default to **10**. Their block, `catboost_options.cpp:157-165`, sets
      `defaultNewtonIterations = 10`, `defaultGradientIterations = 40`, method
      `Newton`.
    - `MultiClass` and `MultiClassOneVsAll` default to **1**, not 10. Their
      block, `catboost_options.cpp:106-112`, sets
      `defaultEstimationMethod = Newton`, `defaultNewtonIterations = 1`,
      `defaultGradientIterations = 10`. The 10 is the Gradient slot and is
      unreachable unless the caller passes
      `leaf_estimation_method="Gradient"`. CatBoost's own documentation says
      "Multiclassification mode -- One Newton iteration", which agrees with
      the source; the folklore does not.
    - `RMSE` defaults to **1**, at `catboost_options.cpp:59-64`: method
      `Newton`, both slots 1, so the method selection cannot change it.

    The selection itself is `SetLeavesEstimationDefault` at
    `catboost_options.cpp:273`, and the switch that reads the method is at
    `:315-336`. The static option default before any of this runs is 1
    (`private/libs/options/oblivious_tree_options.cpp:13`).

    The rest of the table, for the losses that have a mojotrees counterpart:

    | mojotrees | CatBoost loss | method | iterations |
    |---|---|---|---|
    | `SQUARED_ERROR` | `RMSE` | Newton | 1 |
    | `BINARY_LOGISTIC` | `Logloss` | Newton | **10** |
    | `CROSS_ENTROPY` | `CrossEntropy` | Newton | **10** |
    | `POISSON` | `Poisson` | Newton | **10** |
    | `HUBER` | `Huber` | Newton | 1 |
    | `QUANTILE` | `Quantile` | Exact | 1 |
    | `L1` | `MAE` | Exact | 1 |
    | `MAPE` | `MAPE` | Exact | 1 |
    | `TWEEDIE` | `Tweedie` | Newton | 1 on CPU (**20** on GPU) |
    | `GAMMA`, `FAIR`, `CUSTOM` | no CatBoost counterpart | -- | 1 |
    | `LAMBDARANK` | `LambdaMart` | Newton | 1 |

    Two rows deserve a sentence. `Quantile`, `MAE` and `MAPE` reach 1 through
    `useExact` in `SetLeavesEstimationDefault`, which switches those losses (on
    a single host, without `approx_on_full_history`, without monotone
    constraints) to `ELeavesEstimation::Exact` and pins both iteration counts
    to 1. Exact is a closed-form weighted-quantile leaf refit -- it is
    `_renew_leaf_values`, and CatBoost reaching for it on exactly the three
    objectives LightGBM renews is the strongest evidence that these are one
    mechanism and not two. And `Tweedie` is the one loss whose count differs
    by task type, which is why this function says CPU in its first line.
    """
    if (
        objective == BINARY_LOGISTIC
        or objective == CROSS_ENTROPY
        or objective == POISSON
    ):
        return 10
    return 1


def _refuse_leaf_estimation(
    extra: ExtraTreeParams, trainer: StringSlice
) raises:
    """What a trainer that does **not** implement `leaf_estimation_iterations`
    calls, so a fit that would have ignored the setting says which entry point
    ignored it instead of training a model that silently took one step per
    leaf. Returns on the first comparison at the default of 1.
    """
    if not extra.leaf_estimation_active():
        return
    raise Error(
        "leaf_estimation_iterations > 1 is not implemented by ",
        trainer,
        "; it is implemented by boosting.train, boosting.train_more and"
        " boosting.train_with_valid, through"
        " TreeParams.extra.leaf_estimation_iterations",
    )


def _refuse_boost_from_average(
    extra: ExtraTreeParams, trainer: StringSlice
) raises:
    """What a trainer that does not thread `boost_from_average` into its
    `_base_score` call calls, so a fit asking to start from zero says which
    entry point would have ignored it instead of training from the label mean
    under a parameter that said otherwise. Returns on the first comparison at
    the default of True, which is every fit that does not name the parameter.

    Only `False` is refusable. `True` is what every trainer in this package
    does and has always done, so there is no trainer for which it is a request
    that cannot be met; the asymmetry is the point, and it is why this takes
    the bundle rather than a Bool, so the test and the default live in one
    place (`ExtraTreeParams.boost_from_average_disabled`).
    """
    if not extra.boost_from_average_disabled():
        return
    raise Error(
        "boost_from_average=false is not implemented by ",
        trainer,
        "; the trainers that thread it into boosting._base_score are"
        " boosting.train, boosting.train_with_valid, train_gpu.train_gpu and"
        " train_gpu.train_gpu_with_valid, which a dense, single-output fit"
        " reaches. Every other round loop in this package seeds its raw"
        " scores from the objective's optimal constant unconditionally, so"
        " accepting false here would start from the label mean under a"
        " parameter that asked for zero. true is LightGBM's default"
        " (config.h:948) and mojotrees's behavior since the beginning",
    )


def _check_leaf_estimation_config(
    extra: ExtraTreeParams, objective: Int, goss: GossParams
) raises:
    """Refuse `leaf_estimation_iterations > 1` in the two configurations where
    an extra Newton step would be evaluating something other than the quantity
    the first step used. Returns immediately at the default of 1, so an unset
    parameter costs one integer comparison and nothing else.

    **Renewing objectives (`L1`, `QUANTILE`, `MAPE`).** These are the
    objectives whose leaf value is not a Newton step at all:
    `_renew_leaf_values` throws the Newton value away and writes the weighted
    alpha-percentile of the leaf's residuals, which is LightGBM's
    `RenewTreeOutput` and is the *exact* minimizer of the leaf's own loss.
    Their hessian is the sample weight and their gradient is a sign, so a
    Newton step on them is `-mean(sign) / (n + lambda)`, a number with no
    relation to the minimum. Iterating from the percentile would walk the leaf
    away from the exact answer it already holds. CatBoost draws the same line, and
    that is source-verified rather than inferred: `SetLeavesEstimationDefault`
    switches `MAE`, `MAPE`, `Quantile` and their relatives to
    `ELeavesEstimation::Exact` with both iteration counts pinned to 1, and a
    `CB_ENSURE` refuses `Exact` for any other loss. Exact is a closed-form
    weighted-quantile refit and is not what its iteration count applies to. So
    the
    relation between the two mechanisms is neither composition nor
    generalization: they are the *same* job, done exactly for the objectives
    where a closed form exists and iteratively for the objectives where one
    does not, and running both on one tree would apply two corrections where
    one is called for.

    **GOSS.** `goss_round` multiplies the sampled small-gradient rows'
    gradients and hessians by the amplification factor after
    `_fill_grad_hess` filled them, so the leaf value the grower wrote is a
    Newton step on *amplified* derivatives. This function recomputes from
    `raw` and the objective, which knows nothing about the amplification, so
    iteration 2 onward would be stepping on a differently scaled loss than
    iteration 1. Reweighting here would mean reproducing the GOSS draw, which
    is `sampling`'s to own.

    A trainer that does not implement the mechanism at all calls
    `_refuse_leaf_estimation` instead, which is the same refusal keyed on the
    entry point rather than on the configuration.
    """
    if not extra.leaf_estimation_active():
        return
    if objective_renews_leaves(objective):
        raise Error(
            "leaf_estimation_iterations > 1 cannot be combined with an"
            " objective that renews its leaves (l1, quantile, mape). Their"
            " leaf value is already the exact minimizer of the leaf's loss"
            " (the weighted percentile of its residuals, LightGBM's"
            " RenewTreeOutput), and a Newton step on a sign gradient and a"
            " constant hessian would move it away from that minimum. Extra"
            " Newton steps apply to the smooth objectives, where no closed"
            " form exists"
        )
    if goss.enabled:
        raise Error(
            "leaf_estimation_iterations > 1 cannot be combined with goss."
            " GOSS amplifies the sampled rows' gradients and hessians after"
            " they are computed, so the first Newton step is taken on"
            " rescaled derivatives; this recomputes them from the raw scores"
            " and the objective, which carry no amplification, and the second"
            " step would be minimizing a differently weighted loss than the"
            " first"
        )


def _estimate_leaf_values(
    mut tree: Tree,
    data: BinnedMatrix,
    target: List[Float64],
    raw: List[Float64],
    objective: Int,
    weights: List[Float64],
    alpha: Float64,
    iterations: Int,
    lambda_l1: Float64,
    lambda_l2: Float64,
    max_delta_step: Float64,
    bag: List[Int] = [],
    monotone: List[Int] = [],
    leaves: LeafMembership = LeafMembership(),
    settings: DispatchSettings = DispatchSettings.unresolved(),
    float64_derivatives: Bool = False,
) raises:
    """CatBoost's `leaf_estimation_iterations`: keep taking Newton steps on a
    leaf's own rows after the tree's structure is fixed.

    A leaf's value from the grower is one Newton step,
    `-T(G) / (H + lambda_l2)`, with `G` and `H` summed at the raw scores the
    round started from. That is the minimizer of a quadratic fitted at a point
    the leaf is about to leave: once the leaf emits a nonzero value, the rows
    in it sit at a different score, where the objective has a different
    gradient and a different curvature. Iteration `k` re-evaluates every row of
    the leaf at `raw[r] + v`, where `v` is the value the leaf currently holds,
    sums the derivatives, and adds another step. It is Newton's method on the
    leaf's one-dimensional problem, and one iteration is the first Newton
    iterate, which is where every other trainer in this repository stops.

    **`iterations <= 1` returns before it touches a row, and that early return
    is the bit-identity guarantee.** Iteration 1 is never recomputed here: the
    value the grower wrote is kept exactly as written, so there is no second
    route to the first Newton step whose fold order could disagree with the
    histogram's. A fit with the parameter absent, a fit with it set to 1, and a
    fit from before this function existed produce the same `Float64` in every
    leaf.

    **The shift is `v`, not `learning_rate * v`, and that is deliberate.** The
    ensemble adds `learning_rate * v` to each row, so evaluating derivatives at
    `raw[r] + v` is asking "what is the best *full* step for this leaf", and
    shrinkage then takes a fixed fraction of that answer. That is the same
    division of labour every step of this loop already makes: the grower's
    Newton step is unshrunk, `_renew_leaf_values` takes the percentile of
    unshrunk residuals, and `_add_tree_scores` applies the rate once, at the
    end. Evaluating at `raw[r] + learning_rate * v` instead would solve for the
    best *shrunken* step, which `_add_tree_scores` would then shrink a second
    time. CatBoost does the same and it is worth saying where: the walker in
    `catboost/private/libs/algo/approx_calcer/gradient_walker.h` carries no
    rate at all, and `approx_updater_helpers.cpp::NormalizeLeafValues`
    multiplies the *accumulated* leaf value by `learning_rate` once, after the
    last iteration. Raising the iteration count therefore does not compound
    the rate on either side.

    **Three deliberate differences from CatBoost, all recorded rather than
    accidental.**

    1. *No line search.* CatBoost defaults `leaf_estimation_backtracking` to
       `AnyImprovement` and, at more than one iteration, halves a step that
       does not lower the loss -- and on CPU a rejected halving consumes an
       iteration, so `leaf_estimation_iterations=10` there is between 1 and 10
       accepted steps. This takes every step. For the smooth objectives this
       function accepts, the leaf's one-dimensional problem is convex and its
       curvature is bounded below by `lambda_l2`, which damps the step; but
       undamped Newton on a convex non-quadratic can still overshoot, and
       nothing here detects that. A backtracking arm needs a *weighted*
       per-leaf loss and `_mean_loss` is unweighted, so it is not built here.
    2. *`lambda_l2` is not rescaled.* CatBoost's denominator is
       `-SumDer2 + l2_leaf_reg * (sumAllWeights / allDocCount)`
       (`online_predictor.h::ScaleL2Reg`), so its regularizer tracks the mean
       sample weight. This uses `raw_leaf_output`, which is LightGBM's
       `-T(G) / (H + lambda_l2)` with the same `lambda_l2` the grower used --
       the point of the iteration is to keep solving *our* leaf problem more
       exactly, not to change which problem it is. `lambda_l1` likewise stays
       in the soft threshold, which CatBoost has no counterpart for.
    3. *No Gradient method.* CatBoost's `ELeavesEstimation::Gradient` replaces
       the curvature with the leaf's weight sum. Only Newton is implemented
       here, which is the method CatBoost itself defaults to for every loss
       this function accepts.

    **What composes and what does not.** `max_delta_step` and the monotone
    interval are re-applied after every step because both are projections onto
    a fixed set: applying one to an already-projected value is the identity, so
    re-applying them costs nothing and keeps every intermediate value -- the
    value the *next* iteration differentiates at -- inside the cap and inside
    the constraint the tree was grown under. `path_smooth` is not a projection
    and is refused beside this parameter in
    `ExtraTreeParams.check_leaf_estimation`, which is why no smoothing appears
    below. The renewing objectives and GOSS are refused by
    `_check_leaf_estimation_supported`.

    **Determinism across `MOJOTREES_NUM_WORKERS`.** Two folds happen here and
    both are worker-independent by construction. The per-row derivatives come
    from `fill_grad_hess`, which splits rows into blocks but sums nothing
    across them, so every row's pair is the same `Float64` at any worker count.
    The reduction to `G` and `H` is then a plain sequential loop over
    `rows` in the order that list holds, dispatched nowhere: no task
    decomposition exists for the worker count to move. That order is ascending
    row index on both routes below, for the reason `_renew_leaf_values`
    records at length -- `partition_rows_into` keeps each side of every split
    ascending, and the traversal fallback sweeps rows in ascending order --
    so the two routes fold the same list, not merely the same multiset, and
    agree bit for bit. The leaf loop itself runs in ascending node order and
    leaves are independent, so nothing about it is order-sensitive at all.
    """
    if iterations <= 1:
        return
    var n_nodes = len(tree.feature)
    var n_used = len(bag) if len(bag) > 0 else data.n_rows
    # One row list per node, ascending. The membership route copies the list
    # the grower's partition already built; the fallback rebuilds it by
    # walking the tree once per row, which is what every caller without a
    # membership gets.
    var node_rows = List[List[Int]]()
    for _ in range(n_nodes):
        node_rows.append(List[Int]())
    if _renewal_membership_usable(tree, leaves, n_used):
        for l in range(leaves.n_leaves()):
            var node = leaves.node[l]
            ref rows = leaves.rows[l]
            node_rows[node].reserve(len(rows))
            for i in range(len(rows)):
                node_rows[node].append(rows[i])
    else:
        for i in range(n_used):
            var r = bag[i] if len(bag) > 0 else i
            node_rows[tree.leaf_index_row(data, r)].append(r)

    var bounds = node_bounds(tree, monotone)
    var weighted = len(weights) > 0
    # Five scratch lists for the whole tree rather than five per leaf per
    # iteration. They are resized to each leaf's row count and then fully
    # overwritten, so nothing survives from one leaf to the next.
    var raw_l = List[Float64]()
    var tgt_l = List[Float64]()
    var w_l = List[Float64]()
    var grad_l = List[Float64]()
    var hess_l = List[Float64]()
    for node in range(n_nodes):
        if tree.feature[node] >= 0:
            continue
        ref rows = node_rows[node]
        var n_l = len(rows)
        if n_l == 0:
            continue
        raw_l.resize(n_l, 0.0)
        tgt_l.resize(n_l, 0.0)
        grad_l.resize(n_l, 0.0)
        hess_l.resize(n_l, 0.0)
        # Left empty, and so read as unweighted by `fill_grad_hess`, exactly
        # when the fit is unweighted.
        if weighted:
            w_l.resize(n_l, 0.0)
        for i in range(n_l):
            tgt_l[i] = target[rows[i]]
            if weighted:
                w_l[i] = weights[rows[i]]
        var v = tree.value[node]
        for _ in range(iterations - 1):
            for i in range(n_l):
                raw_l[i] = raw[rows[i]] + v
            # The extra Newton steps read derivatives at the same precision
            # the round's first step did. A leaf re-estimated at a different
            # precision from the one its structure was chosen at is a leaf
            # value that does not correspond to any single arm of the switch.
            fill_grad_hess(
                raw_l, tgt_l, objective, w_l, alpha, grad_l, hess_l, settings,
                float64_derivatives,
            )
            var g_sum = 0.0
            var h_sum = 0.0
            for i in range(n_l):
                g_sum += grad_l[i]
                h_sum += hess_l[i]
            # The grower refuses a leaf whose hessian sum is below
            # `min_child_hess`, so this is not reachable from a grown tree at
            # the default regularization; it is here because a caller may set
            # `lambda_l2` to 0 and an objective may drive a leaf's curvature
            # to zero, and a division by zero would write an infinity into a
            # model rather than stopping.
            if h_sum + lambda_l2 <= 0.0:
                break
            v = cap_leaf_output(
                v + raw_leaf_output(g_sum, h_sum, lambda_l1, lambda_l2),
                max_delta_step,
            )
            if len(bounds) > 0:
                v = bounds[node].clamp(v)
        tree.value[node] = v


struct BoosterParams(Copyable, Movable):
    """Ensemble-level hyperparameters.

    `bundling` is LightGBM's `enable_bundle` and the knobs it governs (see
    efb.mojo). It sits here rather than on `TreeParams` because a bundling
    plan is a property of the training matrix, fitted once per training call
    and shared by every tree, not a per-tree control. It defaults to disabled,
    and it is appended so that every positional caller of this constructor --
    `bindings/_mojotrees.mojo`, `cli/`, `capi/` -- keeps working unchanged.

    Only the dense CPU trainers in this file honor it: `train`, `train_more`,
    `train_with_valid`, `train_multiclass`, `train_multiclass_more`, and
    `train_multiclass_with_valid`. Every other trainer that takes a
    `BoosterParams` should refuse an active setting with
    `efb.check_bundling_honored` rather than ignore it; see
    `handoffs/connect_09_algorithms.md`.
    """

    var n_estimators: Int
    var learning_rate: Float64
    var tree: TreeParams
    var bundling: EfbSettings
    var linear: LinearParams
    var ordered: OrderedBoostingParams

    def __init__(
        out self,
        n_estimators: Int,
        learning_rate: Float64,
        var tree: TreeParams,
        var bundling: EfbSettings = EfbSettings.disabled(),
        var linear: LinearParams = LinearParams(),
        var ordered: OrderedBoostingParams = OrderedBoostingParams.disabled(),
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree = tree^
        self.bundling = bundling^
        # LightGBM's `linear_tree` / `linear_lambda` (linear_tree.mojo).
        # Off by default; the metric-path trainers in custom_metric.mojo fit
        # linear leaves when it is on, and the binned-only trainers here
        # refuse it because they have no raw matrix to fit them from.
        self.linear = linear^
        # CatBoost's `boosting_type=Ordered` (ordered_boosting.mojo). Off by
        # default, appended last so every positional caller of this
        # constructor -- `bindings/_mojotrees.mojo`, `cli/`, `capi/` -- keeps
        # working unchanged, exactly as `bundling` and `linear` were. Only
        # `train` honors it; every other trainer here refuses it with
        # `check_ordered_honored` rather than ignoring it.
        self.ordered = ordered^

    @staticmethod
    def default() -> BoosterParams:
        return BoosterParams(100, 0.1, TreeParams.default())


@fieldwise_init
struct IterationRange(Copyable, Movable):
    """A half-open `[start, stop)` slice of boosting iterations to predict
    with, already clamped to what an ensemble holds.

    Prediction over a slice follows LightGBM: the range selects whole
    boosting iterations (one tree for a single-output model, one tree per
    class for a multiclass one), and the base score counts as part of
    iteration 0. So a range starting at 0 adds the base score and a range
    starting later does not, exactly as in LightGBM, where the base score is
    folded into the leaf values of the first iteration's trees rather than
    stored apart. `[0, 0)` is therefore the base-score-only model, the same
    prediction a zero-iteration ensemble makes, while an empty range that
    starts later sums to zero.

    Build one with `slice` from an explicit `[start, stop)` pair, or with
    `clamp` from LightGBM's `(start_iteration, num_iteration)` pair, where a
    nonpositive `num_iteration` means every remaining iteration."""

    var start: Int
    var stop: Int

    @staticmethod
    def slice(n_iterations: Int, start: Int, stop: Int) -> IterationRange:
        """Clamp an explicit half-open pair into `[0, n_iterations]`. A stop
        at or below the start yields an empty range rather than an error, so
        callers can slice past the end of a short ensemble."""
        var lo = start
        if lo < 0:
            lo = 0
        if lo > n_iterations:
            lo = n_iterations
        var hi = stop
        if hi > n_iterations:
            hi = n_iterations
        if hi < lo:
            hi = lo
        return IterationRange(lo, hi)

    @staticmethod
    def clamp(n_iterations: Int, start: Int, num: Int) -> IterationRange:
        """LightGBM's `(start_iteration, num_iteration)` convention: a
        negative start clamps to 0, a start past the end clamps to an empty
        range at the end, and `num <= 0` means every iteration from the start
        on."""
        var lo = start
        if lo < 0:
            lo = 0
        if lo > n_iterations:
            lo = n_iterations
        var hi = n_iterations
        if num > 0 and lo + num < n_iterations:
            hi = lo + num
        return IterationRange(lo, hi)

    @always_inline
    def is_empty(self) -> Bool:
        return self.stop <= self.start

    @always_inline
    def includes_base(self) -> Bool:
        """Whether the base score belongs to this range: it sits in
        iteration 0."""
        return self.start == 0

    @always_inline
    def n_iterations(self) -> Int:
        return self.stop - self.start


struct Booster(Copyable, Movable):
    """A fitted single-output ensemble.

    `monotone` records the monotonic constraints the ensemble was trained
    under. Unlike the other training-time restrictions it is kept with the
    model and serialized, because it is a property the fitted model satisfies
    rather than only a knob that shaped it: predictions are monotone in those
    features, and a consumer cannot recover that claim from the trees.
    """

    var trees: List[Tree]
    var base_score: Float64
    var learning_rate: Float64
    var objective: Int
    var monotone: MonotoneConstraints
    var linear: LinearEnsemble

    def __init__(
        out self,
        var trees: List[Tree],
        base_score: Float64,
        learning_rate: Float64,
        objective: Int,
        var monotone: MonotoneConstraints = MonotoneConstraints(),
        var linear: LinearEnsemble = LinearEnsemble(),
    ):
        self.trees = trees^
        self.base_score = base_score
        self.learning_rate = learning_rate
        self.objective = objective
        self.monotone = monotone^
        # The linear-leaf sidecar (linear_tree.mojo), keyed by (tree index,
        # leaf ordinal); inactive for a constant-leaf model. The bins-only
        # prediction methods below read `Tree.value` and so see the constant
        # fallback; `model.Model` evaluates the sidecar on the raw row.
        self.linear = linear^

    def predict_raw_row(self, data: BinnedMatrix, row: Int) -> Float64:
        """Raw ensemble output (log-odds for BINARY_LOGISTIC)."""
        var s = self.base_score
        for i in range(len(self.trees)):
            s += self.learning_rate * self.trees[i].predict_row(data, row)
        return s

    @always_inline
    def response(self, raw: Float64) -> Float64:
        """The objective's inverse link applied to a raw score: the
        probability for logistic and cross entropy, the expected value for
        poisson, gamma, and tweedie, and the raw score itself otherwise. For
        CUSTOM this is the raw score, since the framework does not know the
        objective's link and the caller applies it. Every response-scale
        prediction goes through here, so the raw and response scales cannot
        drift apart.

        Which objective has which link is `objective_link` in
        objective_registry.mojo. This function, `response_scale` in
        custom_metric.mojo, and `response_for_objective` in gpu_predict.mojo
        each used to carry the same table and agree by inspection; all three
        now read it. A metric scoring one link while a prediction applies
        another is the kind of disagreement that produces wrong numbers
        rather than an error.

        `LINK_SOFTMAX` cannot reach here: a `Booster` is single-output, and
        multiclass raw scores go through `MulticlassBooster`, whose softmax
        is over a whole row rather than one score."""
        var link = objective_link(self.objective)
        if link == LINK_SIGMOID:
            return _sigmoid(raw)
        if link == LINK_EXP:
            return exp(raw)
        return raw

    @always_inline
    def response_is_identity(self) -> Bool:
        """Whether `response` is the identity function on every Float64.

        A batch predictor that calls `response` once per ROW pays a call plus
        the two compares above for every row, and for an identity link every
        one of those calls returns its argument unchanged. This is the same
        question hoisted to once per CALL, so the caller can skip the whole
        pass. Nothing here decides a value; it decides whether a provably
        value-preserving loop runs at all, which is why using it cannot move
        a bit.

        This reads `objective_link` rather than listing objectives, for the
        reason `response` gives: the table lives in one place. Note that it is
        NOT "true for every regression objective" -- POISSON, GAMMA, TWEEDIE
        and SURVIVAL_AFT are regression objectives with `LINK_EXP`, and they
        must keep paying the pass.
        """
        return objective_link(self.objective) == LINK_IDENTITY

    @always_inline
    def n_iterations(self) -> Int:
        """Boosting iterations this ensemble holds. One iteration is one tree
        for a single-output model."""
        return len(self.trees)

    def predict_row(self, data: BinnedMatrix, row: Int) -> Float64:
        """Prediction on the response scale (probability for logistic,
        expected count for poisson). For CUSTOM this is the raw score:
        the framework does not know the objective's inverse link, so the
        caller applies it."""
        return self.response(self.predict_raw_row(data, row))

    def predict_raw_bins(self, bins: List[Int]) -> Float64:
        var s = self.base_score
        for i in range(len(self.trees)):
            s += self.learning_rate * self.trees[i].predict_bins(bins)
        return s

    def predict_bins(self, bins: List[Int]) -> Float64:
        return self.response(self.predict_raw_bins(bins))

    def predict_raw_bins_range(
        self, bins: List[Int], rng: IterationRange
    ) -> Float64:
        """Raw output of the boosting iterations in `rng` alone.

        The base score belongs to iteration 0, so it is added only when the
        range starts there; see IterationRange. A full range reproduces
        `predict_raw_bins` exactly, and the ranges [0, k) and [k, n) sum to
        the full raw score for any k."""
        var s = self.base_score if rng.includes_base() else 0.0
        for i in range(rng.start, rng.stop):
            s += self.learning_rate * self.trees[i].predict_bins(bins)
        return s

    def predict_bins_range(
        self, bins: List[Int], rng: IterationRange
    ) -> Float64:
        """Response-scale prediction from the iterations in `rng` alone."""
        return self.response(self.predict_raw_bins_range(bins, rng))

    def predict_batch_range(
        self,
        data: BinnedMatrix,
        rng: IterationRange,
        raw_score: Bool = False,
    ) raises -> List[Float64]:
        """One prediction per row of an already binned matrix, over row
        blocks.

        The body is the per-row body `Model.predict_batch` ran serially, with
        the per-row bin gather taken out: it is `predict_raw_bins_range`
        inlined over `Tree.predict_row`, which reads `data.bin_at(r, f)` where
        the bins version read `bins[f]`. See the comment on the loop for why
        that is the same Int at every node.

        Splitting it over blocks cannot change a number. A row reads the
        ensemble, which no block writes, and writes output slot `r`, which no
        other row touches; nothing is accumulated across rows. So the outputs
        are bit-identical to the serial loop's by construction, at any block
        count, and there is nothing here to argue about ordering.

        Batch prediction was the one phase of the LightGBM head-to-head that
        ran at parallel efficiency 1.00 on every dataset measured, against 6.5
        to 9.5 for LightGBM on the same ten threads, because this loop was
        serial. What that becomes is not measured here and this docstring
        claims no number.
        """
        var n = data.n_rows
        var n_features = data.n_features
        var out = List[Float64](capacity=n)
        out.resize(n, 0.0)
        var out_p = out.unsafe_ptr()

        # THE GATHER IS GONE. This used to materialize a row's whole bin
        # vector -- `bins.clear()` then one `append` per feature, so
        # `n_rows * n_features` of them -- and hand it to
        # `predict_raw_bins_range`. The trees then read at most `depth` of
        # those entries per tree and ignored the rest, which on a 90-feature
        # matrix is most of the work in this loop.
        #
        # So the walk reads the matrix in place instead. `Tree.predict_row` is
        # `Tree.predict_bins` with `data.bin_at(row, f)` where the latter has
        # `bins[f]`, and those are the same Int by the definition of the
        # gather that used to run here, so every node takes the same branch.
        # The sum around it is `predict_raw_bins_range` inlined, unchanged:
        # the base score exactly when the range starts at iteration 0, then
        # one `s += learning_rate * value` per tree in ascending range order.
        # `response` is applied to the finished raw score, which is what
        # `predict_bins_range` did. Bit-identical, and it allocates nothing
        # per block.
        def apply(start: Int, end: Int) {imm}:
            var base = self.base_score if rng.includes_base() else 0.0
            for r in range(start, end):
                var s = base
                for i in range(rng.start, rng.stop):
                    s += self.learning_rate * self.trees[i].predict_row(
                        data, r
                    )
                if raw_score:
                    out_p.unsafe_store(r, s)
                else:
                    out_p.unsafe_store(r, self.response(s))

        dispatch_rows(
            apply,
            n,
            n * (n_features + rng.n_iterations() * _TRAVERSAL_ROW_OPS),
        )
        return out^

    def leaf_ordinals_range(self, rng: IterationRange) -> List[List[Int]]:
        """The per-node leaf ordinal table (see `Tree.leaf_ordinals`) of each
        tree in `rng`, in range order. Build this once and index it with
        `Tree.leaf_index_bins` when predicting leaves for many rows."""
        var tables = List[List[Int]](capacity=rng.n_iterations())
        for i in range(rng.start, rng.stop):
            tables.append(self.trees[i].leaf_ordinals())
        return tables^

    def leaf_indices_bins(
        self, bins: List[Int], rng: IterationRange
    ) -> List[Int]:
        """The leaf ordinal this example reaches in each tree of `rng`, one
        entry per iteration in range order. An empty range yields an empty
        list."""
        var out = List[Int](capacity=rng.n_iterations())
        for i in range(rng.start, rng.stop):
            out.append(self.trees[i].leaf_ordinal_bins(bins))
        return out^


def _same_signs(a: List[Int], b: List[Int]) -> Bool:
    """Whether two monotonic constraint vectors are the same vector."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


# How much work one row of a score update is worth, in the histogram-op
# equivalents `parallel.plan_tasks` compares against its grain. A leaf-window
# row is one indirect load of a row id and one read-modify-write of a Float64
# at a scattered address; a traversal row is that plus a dependent walk down
# the tree, which is several loads deep for a 31-leaf tree. Both numbers are
# scheduling estimates and nothing more: every block writes only its own
# slots, so the result is the same at one task and at sixty, and these can be
# retuned without changing an output. Neither has been measured.
comptime _LEAF_ROW_OPS = 2
comptime _TRAVERSAL_ROW_OPS = 8


def _leaf_score_update_enabled() -> Bool:
    """Whether a round may add a tree's contribution by leaf membership.

    `MOJOTREES_LEAF_SCORE_UPDATE=0` forces the full-tree traversal every
    trainer here used before the membership was kept, the way
    `MOJOTREES_GPU_SPLIT_RESIDENT=0` forces the device split search's old
    loop. The two routes are meant to leave bit-identical raw scores, and a
    switch is what lets one build train both ways and compare, instead of
    comparing across two builds and hoping nothing else moved. It is not a
    tuning knob: there is no workload on which the traversal is the better
    route, and none has been measured either way.
    """
    return getenv("MOJOTREES_LEAF_SCORE_UPDATE") != "0"


def _add_by_traversal(
    mut raw: List[Float64],
    tree: Tree,
    data: BinnedMatrix,
    learning_rate: Float64,
    stride: Int,
    offset: Int,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """`raw[r * stride + offset] += learning_rate * tree.predict_row(data, r)`
    for every row of `data`.

    The pre-existing update, spread over row blocks. Every row reads and
    writes one slot of its own and nothing else, and `data` and `tree` are
    read-only throughout, so the blocks are independent and the result does
    not depend on how many there are: this is bit-identical to the serial loop
    by construction rather than by argument.

    The arithmetic is left as the expression the trainers always wrote, rather
    than fused by hand as in `_add_by_leaf`, so that it keeps tracking
    whatever the compiler does with `raw[r] += learning_rate *
    tree.predict_row(...)` -- which is the same expression the trainers this
    lane did not touch still write. It was observed to contract into a fused
    multiply-add, and `test_round_overhead` compares this against that loop
    written out by hand.

    `stride` and `offset` are 1 and 0 for a single-output ensemble and
    `n_classes` and the class index for the row-major multiclass scores.
    """
    var n = data.n_rows
    var raw_p = raw.unsafe_ptr()

    def apply(start: Int, end: Int) {imm}:
        for r in range(start, end):
            var slot = r * stride + offset
            raw_p.unsafe_store(
                slot,
                raw_p.unsafe_load(slot)
                + learning_rate * tree.predict_row(data, r),
            )

    dispatch_rows_with(settings, apply, n, n * _TRAVERSAL_ROW_OPS)


def _add_by_leaf(
    mut raw: List[Float64],
    tree: Tree,
    leaves: LeafMembership,
    learning_rate: Float64,
    n_rows: Int,
    stride: Int,
    offset: Int,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """The same update, routed through the leaf membership growth handed back
    (see `tree.LeafMembership`).

    A leaf contributes one value, `tree.value[node]`, to every row that landed
    in it. `Tree.predict_row` returns `value[node]` for exactly those rows --
    the partition that built the row lists and the routing `predict_row` walks
    are the same three tests on the same bins, since `Tree._set_split` records
    the very `SplitInfo` and missing bin `partition_rows_into` was handed --
    so the leaf route and the traversal have the same two Float64 operands for
    every row.

    Having the same operands is not yet having the same answer, and this is
    the one place in the change where that mattered. `raw[r] += learning_rate
    * tree.predict_row(...)` compiles to a fused multiply-add: the product is
    never rounded to a Float64 of its own. Hoisting `learning_rate * value`
    out of the loop, which is the obvious way to write the leaf update, rounds
    it, and lands one ulp away on the rows where it matters -- a difference
    that feeds the next round's gradients and gives a different second tree.
    So the multiply and the add are fused here explicitly rather than left to
    whatever the compiler does with the expression, because the compiler was
    observed to contract the traversal's expression and not this loop's.
    `test_round_overhead` pins the result against the pre-existing update
    written out by hand, so a compiler that ever stopped contracting there
    would be caught rather than silently accepted.

    Why the split below cannot move a bit, which is the property to preserve.
    A row's update is a single independent read-modify-write of its own slot:
    one `fma` reading `raw[slot]` and writing `raw[slot]`, with `learning_rate`
    a scalar and `value` a per-leaf constant read out of `tree.value`. Nothing
    is accumulated across rows -- there is no running sum, no shared
    accumulator, no reduction to combine -- and each row appears in exactly one
    leaf's row list, so no row is written twice and no row reads a slot another
    row writes. A partition of the (leaf, row) pairs into blocks therefore
    reassociates nothing: every slot receives the same one `fma` with the same
    two operands whatever block it landed in, so the result is identical at one
    block and at sixty. **Any change here that introduces an accumulation
    spanning two rows breaks this and is wrong.**

    The blocks are row blocks, not leaf blocks. The index space is the
    *concatenation* of the leaves' row lists -- position `p` names leaf `l` and
    its `p - start[l]`-th row -- and `dispatch_rows_with` cuts that flat range
    into equal contiguous pieces. A block may begin and end in the middle of a
    leaf, and may span several whole leaves; it walks the leaves it overlaps
    and does each one's slice.

    Splitting by leaf instead, which is what this did, made the round's score
    update as long as its single largest leaf. Leaf-wise growth does not make
    equal leaves: `num_leaves` is at most a few hundred while `min_data_in_leaf`
    is twenty, so one leaf may legitimately hold nearly every row, and a
    schedule whose smallest indivisible unit is a leaf can never use more
    workers than the reciprocal of that leaf's row share no matter how many
    cores the machine has. Cutting inside a leaf removes that floor: the
    longest block is now `ceil(total / n_blocks)` rows by construction.

    The caller must have checked `leaves.covers_all_rows`; this adds nothing
    to a row no leaf holds.
    """
    var n_leaves = leaves.n_leaves()
    var raw_p = raw.unsafe_ptr()

    # Prefix sum over leaf sizes: `start[l]` is where leaf `l`'s rows begin in
    # the concatenation, and `start[n_leaves]` is the total row count. The tree
    # has at most `num_leaves` leaves -- 31 by default, a few hundred at the
    # extreme -- so this is a few hundred integer adds per round against the
    # millions of row updates below it. The total is taken from the row lists
    # rather than from `n_rows` so that the split covers exactly what is there.
    var start = List[Int](capacity=n_leaves + 1)
    start.append(0)
    var total = 0
    for l in range(n_leaves):
        total += len(leaves.rows[l])
        start.append(total)

    def apply(p_start: Int, p_end: Int) {imm}:
        # First leaf this block reaches. A linear scan over at most a few
        # hundred ascending offsets, run once per block, not once per row.
        # `start` is read through the capture rather than through a raw
        # pointer taken before the dispatch, because a pointer would not keep
        # the list alive past its last use and the block runs after that.
        var l = 0
        while l < n_leaves and start[l + 1] <= p_start:
            l += 1
        while l < n_leaves and start[l] < p_end:
            var value = tree.value[leaves.node[l]]
            # A `ref` binding, because `leaves.rows[l]` in an expression
            # materializes a copy of the row list and a pointer taken from
            # that copy dangles the moment the expression ends. The bound
            # reference is the list the grower handed back.
            ref rows = leaves.rows[l]
            var rows_p = rows.unsafe_ptr()
            var base = start[l]
            var i0 = p_start - base
            if i0 < 0:
                i0 = 0
            var i1 = p_end - base
            var n_l = len(rows)
            if i1 > n_l:
                i1 = n_l
            for i in range(i0, i1):
                var slot = Int(rows_p.unsafe_load(i)) * stride + offset
                raw_p.unsafe_store(
                    slot, fma(learning_rate, value, raw_p.unsafe_load(slot))
                )
            l += 1

    # `total`, not `n_leaves`, is the item count: `plan_tasks` clamps the task
    # count to the number of items, so a 31-leaf tree used to cap the fan-out
    # at 31 tasks (and at one task per leaf, still unequal ones). The work
    # estimate is left at `n_rows * _LEAF_ROW_OPS`, unchanged, so the
    # serial/parallel crossover sits exactly where it did.
    dispatch_rows_with(settings, apply, total, n_rows * _LEAF_ROW_OPS)


def _add_tree_scores(
    mut raw: List[Float64],
    tree: Tree,
    leaves: LeafMembership,
    data: BinnedMatrix,
    learning_rate: Float64,
    by_leaf: Bool,
    stride: Int = 1,
    offset: Int = 0,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """End a boosting round: add `learning_rate` times this tree's output to
    the raw score of every training row.

    Takes the leaf route when the caller allows it and the membership covers
    the whole dataset, and the traversal otherwise. There are two paths rather
    than one because a bagged, GOSS-sampled, or class-balanced round grows its
    tree on a subset while every row of the dataset still needs the tree's
    contribution: the membership names the sampled rows alone, and a row
    outside the sample has no leaf to be found in. Those rounds keep the
    traversal, unchanged, so what a row receives does not depend on how the
    round was sampled.
    """
    if by_leaf and leaves.covers_all_rows:
        _add_by_leaf(
            raw, tree, leaves, learning_rate, data.n_rows, stride, offset,
            settings,
        )
        return
    _add_by_traversal(
        raw, tree, data, learning_rate, stride, offset, settings
    )


# ---------------------------------------------------------------------------
# Ordered boosting (CatBoost `boosting_type=Ordered`)
# ---------------------------------------------------------------------------


def _check_ordered(
    ordered: OrderedBoostingParams,
    objective: Int,
    bagging: BaggingParams,
    goss: GossParams,
    class_bagging: ClassBaggingParams,
    round_offset: Int,
) raises:
    """Everything ordered boosting is exclusive with, refused before the first
    tree rather than silently composed with.

    - **Row samplers.** Bagging, GOSS and balanced bagging all remove rows from
      the round. Ordered boosting's premise is that every row's derivative
      comes from a model fitted on a known prefix of a known permutation; a
      sampler that drops rows changes which prefix each rung was actually
      fitted on and the ladder stops meaning what it says. CatBoost does
      compose ordered boosting with its bootstrap, but it does so by rebuilding
      the sampled derivatives per rung
      (`tensor_search_helpers.cpp::Bootstrap` runs inside the per-rung loop);
      that is a second mechanism and this lane did not build it.
    - **Renewing objectives.** `MAPE`, `L1` and the quantile family replace a
      leaf's value with a percentile of residuals rather than a Newton step
      (`_renew_leaf_values`), and there is no per-rung analogue of that here:
      the rung planes would have to be advanced by a value fitted a different
      way from the model's. Refused rather than approximated.
    - **Continued training.** The rung planes are fit state, not model state,
      and a `Booster` does not carry them. Resuming would restart every rung
      from the ensemble's current score, which is precisely the leakage
      ordered boosting removes. CatBoost snapshots them
      (`TFold::SaveApproxes` / `LoadApproxes`); we do not, so we refuse.
    """
    ordered.validate()
    if not ordered.enabled:
        return
    if bagging_enabled(bagging):
        raise Error(
            "boosting_type=ordered is exclusive with bagging: a dropped row"
            " changes which prefix each fold was fitted on"
        )
    if goss.enabled:
        raise Error(
            "boosting_type=ordered is exclusive with goss: a dropped row"
            " changes which prefix each fold was fitted on"
        )
    if class_bagging.enabled():
        raise Error(
            "boosting_type=ordered is exclusive with balanced bagging: a"
            " dropped row changes which prefix each fold was fitted on"
        )
    if objective_renews_leaves(objective):
        raise Error(
            "boosting_type=ordered is not implemented for objectives that"
            " renew leaf values from residual percentiles"
        )
    if round_offset > 0:
        raise Error(
            "boosting_type=ordered cannot continue training: the per-fold raw"
            " score planes are fit state and are not carried by a model"
        )


def _ordered_leaf_of_row(
    tree: Tree,
    data: BinnedMatrix,
    leaves: LeafMembership,
    n: Int,
    mut leaf_of_row: List[Int],
) raises:
    """Each row's leaf *node index* in this tree, once per round.

    Every ordered consumer below needs the same map -- the model's leaf refit,
    and each rung's own refit and plane advance -- and walking the tree once
    per row per rung would be `K` traversals of the dataset instead of one
    array read. The membership the grower just handed back covers the whole
    dataset whenever it is usable (ordered boosting bags nothing, so it always
    is on the honored path); the traversal is the fallback, and it is the same
    fallback `_estimate_leaf_values` takes.
    """
    if len(leaf_of_row) != n:
        leaf_of_row.resize(n, 0)
    if _renewal_membership_usable(tree, leaves, n):
        for l in range(leaves.n_leaves()):
            var node = leaves.node[l]
            ref rows = leaves.rows[l]
            for i in range(len(rows)):
                leaf_of_row[rows[i]] = node
        return
    for r in range(n):
        leaf_of_row[r] = tree.leaf_index_row(data, r)


def _ordered_fit_leaf_values(
    tree: Tree,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    row_offset: Int,
    n_rows: Int,
    leaf_of_row: List[Int],
    lambda_l1: Float64,
    lambda_l2: Float64,
    max_delta_step: Float64,
    bounds: List[OutputBounds],
    mut values: List[Float64],
) raises:
    """One Newton step per leaf on a chosen row set, written into `values`
    indexed by node.

    `rows[row_offset : row_offset + n_rows]` is the row set: the caller keeps
    every permutation end to end in one buffer and passes the window belonging
    to the one it is advancing, so a rung body is named without materializing
    it. An empty `rows` means "rows `0 .. n_rows - 1` in order", which is the
    model's own refit over the whole dataset.

    **The fold order is the ascending order of the row list handed in, on both
    routes, and nothing is summed across any decomposition** -- there is no
    dispatch here at all -- so every leaf's `G` and `H` are the same `Float64`
    at any `MOJOTREES_NUM_WORKERS`.

    A leaf with no rows keeps value 0, which is what an empty rung body gives
    the leaves it does not reach. That is the correct answer and not a
    fallback: a rung whose body never entered a leaf has learned nothing about
    it and must add nothing there.
    """
    var n_nodes = len(tree.feature)
    if len(values) != n_nodes:
        values.resize(n_nodes, 0.0)
    var g_sum = List[Float64](capacity=n_nodes)
    var h_sum = List[Float64](capacity=n_nodes)
    for _ in range(n_nodes):
        g_sum.append(0.0)
        h_sum.append(0.0)
    var indexed = len(rows) > 0
    for i in range(n_rows):
        var r = rows[row_offset + i] if indexed else i
        var node = leaf_of_row[r]
        g_sum[node] += grad[i]
        h_sum[node] += hess[i]
    for node in range(n_nodes):
        if tree.feature[node] >= 0:
            values[node] = 0.0
            continue
        var h = h_sum[node]
        if h + lambda_l2 <= 0.0:
            values[node] = 0.0
            continue
        var v = cap_leaf_output(
            raw_leaf_output(g_sum[node], h, lambda_l1, lambda_l2),
            max_delta_step,
        )
        if len(bounds) > 0:
            v = bounds[node].clamp(v)
        values[node] = v


def _tree_leaf_values(
    tree: Tree, learning_rate: Float64, mut values: List[Float64]
) raises -> Int:
    """The leaf values of one tree, in node order, scaled by the learning
    rate, and how many there are.

    This is what MVS's derived `mvs_reg` reads on every tree after the first:
    `TMvsSampler::GetLambda` calls `CalculateLastIterMeanLeafValue` on
    `leafValues.back()`, the previous iteration's entry of
    `ctx->LearnProgress->LeafValues`.

    **The learning-rate factor is why this is a function rather than a slice.**
    A CatBoost model's stored leaf values already carry the shrinkage -- a
    saved model scores by summing them with no separate rate -- whereas a
    mojotrees `Tree.value` is the bare Newton step and `_add_tree_scores`
    applies `learning_rate` as it accumulates. The two conventions differ by
    exactly that factor, and the lambda is the squared *mean norm* of these
    values, so getting it wrong scales the MVS floor by `learning_rate^2` and
    quietly changes which rows survive. This is reasoning from the two
    libraries' storage conventions rather than a line of CatBoost this lane
    re-read; it is flagged as such in the lane report.

    A leaf is a node with `feature[node] < 0`, the same test the predictor and
    the renewal use. Single-output trees only, so the stride is 1; the
    multiclass loops do not call this.

    **The reason given here for that was stale and has been corrected.** It
    read "see `_boost_rounds_multiclass`, which takes no bootstrap bundle at
    all". That loop does take one and does draw it -- it calls
    `bootstrap_round` once per round and shares the weights across the round's
    `K` trees. What it cannot do is DERIVE the lambda, so it never needs the
    previous tree's leaf values: `sampling.check_mvs_reg_is_set` refuses
    `bootstrap_type='mvs'` without an explicit `mvs_reg` for the softmax loops
    by name, because a CatBoost multiclass iteration is one tree holding a
    `K`-vector per leaf while a round here is `K` unrelated trees, so the
    `[dim][leaf]` rectangle `CalculateLastIterMeanLeafValue` reads does not
    exist. The conclusion is unchanged; the reason was not.
    """
    values.clear()
    for node in range(len(tree.feature)):
        if tree.feature[node] < 0:
            values.append(tree.value[node] * learning_rate)
    return len(values)


def _round_random_score_scale(
    extra: ExtraTreeParams,
    grad: List[Float64],
    n_rows: Int,
    round: Int,
    learning_rate: Float64,
) raises -> Float64:
    """One tree's `ExtraTreeParams.random_score_scale`, or 0.0 when
    `random_strength` is off.

    CatBoost's `CalcScoreStDev` is
    `random_strength * derivativesStDevFromZero * modelSizeDecrease`; the
    strength is the user's knob and stays on the bundle, and the other two
    factors are `random_score_scale_from_gradients`, whose `modelLength` is
    `train.cpp`'s `iteration_index * learning_rate`. `round` is the **absolute**
    round index, so a continued run (`train_more`) computes the model length
    an uninterrupted run would have had rather than restarting it at zero.

    At `random_strength == 0`, the default, this returns before it reads a row
    and the bundle keeps the 0.0 it was constructed with, so not one
    instruction of this reaches the default arm.

    **The one refusal.** `random_score_scale_from_gradients` can return exactly
    0.0 -- from an all-zero gradient vector, or from a model length past the
    point where `exp(log(n) - L)` underflows, which at `learning_rate = 0.1`
    is somewhere beyond round 7500. In CatBoost that is a well-defined
    `N(0, 0)` and the noise simply vanishes; here `split.find_best_split`
    refuses a positive strength beside a non-positive scale, because that pair
    is also what an unwired caller looks like and it cannot tell the two apart.
    So a genuine zero is refused here instead, at the round that produced it
    and with the cause named, rather than reaching a message that says the
    scale was never computed when it was. Distinguishing "computed and zero"
    from "never set" is a field on `ExtraTreeParams`, which this lane does not
    own; the lane report carries the diff.
    """
    if not (extra.random_strength > 0.0):
        return 0.0
    if n_rows <= 0:
        raise Error(
            "random_strength needs at least one row: its scale is the RMS of"
            " the round's derivatives"
        )
    var scale = random_score_scale_from_gradients(
        grad, n_rows, Float64(round) * learning_rate
    )
    if not (scale > 0.0):
        raise Error(
            "random_strength's per-tree scale came out as ",
            scale,
            " on round ",
            round,
            ", and the split search refuses a positive random_strength beside"
            " a non-positive scale because that is indistinguishable from a"
            " caller that never computed one. CatBoost's scoreStDev is"
            " derivativesStDevFromZero * modelSizeDecrease: the first factor"
            " is zero only when every derivative is exactly zero (the fit has"
            " converged), and the second underflows to zero once"
            " round * learning_rate exceeds about 745 + log(n_rows). Lower"
            " n_estimators or learning_rate, or set random_strength=0",
        )
    return scale


def _boost_rounds(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64],
    alpha: Float64,
    bagging: BaggingParams,
    goss: GossParams,
    learning_rate: Float64,
    round_offset: Int,
    mut raw: List[Float64],
    mut trees: List[Tree],
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises:
    """Grow `params.n_estimators` trees, appending them to `trees` and
    keeping `raw` (the raw score of every training row) in step.

    This is the boosting loop itself, shared by the fit-from-scratch path
    and the continue-training path. `round_offset` is the number of rounds
    already grown: every seeded decision (the bagging draw, the GOSS warmup
    schedule, the per-tree feature sample) reads the absolute round index,
    so continued rounds draw what they would have drawn had the whole run
    been one call. `learning_rate` is passed rather than read from `params`
    because a continued run shrinks with the ensemble's rate.

    `class_bagging` replaces the uniform bag with LightGBM's balanced one
    (see sampling.mojo). It produces the same shape -- one ascending,
    duplicate-free row list per bag -- so nothing downstream of the draw
    changes: the same histograms, `min_data_in_leaf` counts, leaf values, and
    score updates follow. The caller validates it with
    `_check_class_bagging`, which is also what makes it exclusive with the
    other two samplers.

    `params.bundling` is fitted here, once, and handed to every tree. It
    changes only how histograms are laid out (see efb.mojo): the trees, the
    leaf renewal below, and the score update all read the original matrix,
    so a bundled run and an unbundled one differ in cost and not in result.
    Disabled by default, and `prepare_bundling` falls back to the unbundled
    matrix whenever the plan would not pay for itself.

    `bootstrap` is CatBoost's `bootstrap_type` (see sampling.mojo): MVS, which
    is CatBoost's real CPU default, or the Bayesian bootstrap. Both put a
    per-row weight on the round's derivatives, which is why the
    constant-hessian declaration below is withdrawn for them and why
    `BootstrapParams.check_hessian_declaration` is called on the result. It is
    exclusive with the three row samplers above (`_check_bootstrap`) and
    disabled by default, so an untouched bundle takes no draw, allocates no
    weight vector, and leaves every fit byte for byte the fit it is today.

    `params.tree.extra.random_score_scale` is the one field this loop writes
    rather than reads. It is CatBoost's `scoreStDev` divided by the user's
    `random_strength`, a property of the ensemble's current gradients and of
    how long the model already is, so it is per-tree state and not
    configuration -- see `_round_random_score_scale` and the local copy below.
    """
    if params.linear.is_active():
        check_linear_tree_unconnected("the binned-only boosting trainers")
    # `random_score_scale` is per-tree ENSEMBLE STATE, not user configuration:
    # CatBoost recomputes it before every tree from that round's derivatives
    # and the model's length. `params` is borrowed, and it must stay borrowed
    # -- writing the scale back into the caller's `BoosterParams` would make a
    # fit mutate its own argument, so two identical `train` calls would differ
    # (the second would start at the first's last scale) and a `BoosterParams`
    # shared between two fits would carry one fit's gradients into the other.
    # So the loop keeps its own copy of the tree bundle and hands THAT to the
    # grower. One `TreeParams` copy per fit, not per tree, and at
    # `random_strength = 0` -- the default -- the copy is never written and the
    # grower reads exactly what it read before.
    var tree_params = params.tree.copy()
    var noisy = params.tree.extra.random_strength > 0.0
    var bundling = prepare_bundling(data, params.bundling)
    # One CEGB ledger for the whole ensemble, handed to every tree. That
    # lifetime is the mechanism's premise: the model pays for a feature once,
    # so a later tree reusing a feature an earlier one already needed gets it
    # free, where a per-tree ledger would be a much harsher regularizer.
    # `CegbLedger.create` allocates nothing unless a penalty needs it, which
    # is the default, and `grow_tree_with_cegb` refuses the penalties this
    # ledger cannot serve rather than charging them as zero.
    var ledger = CegbLedger.create(
        params.tree.extra.penalties.cegb, data.n_features, data.n_rows
    )
    var n = data.n_rows
    var signs = params.tree.monotone.active_signs()
    var renews = objective_renews_leaves(objective)
    var renew_w = renewal_weights(objective, target, sample_weight)
    var renew_a = renewal_alpha(objective, alpha)
    # Once per fit, not once per tree: nothing it reads moves inside the loop.
    # At the default of 1 both return on the first comparison.
    var leaf_iters = params.tree.extra.leaf_estimation_iterations
    _check_leaf_estimation_config(params.tree.extra, objective, goss)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    # LightGBM turns balanced bagging on only when the dataset holds a
    # positive row, and falls back to plain `bagging_fraction` when it does
    # not. The label pass is one sweep and the labels do not change, so the
    # gate is hoisted out of the round loop.
    var balanced = class_bagging.enabled() and has_positive_rows(target)
    # One profile for the whole loop rather than one per tree, so a hundred
    # rounds produce one table to diff instead of a hundred (phase_profile.mojo).
    # Off unless `MOJOTREES_PHASE_PROFILE` says otherwise, and an off profile
    # reads no clock and writes no counter, so the ensemble this loop grows is
    # the same ensemble either way.
    var profile = PhaseProfile.from_env(SCOPE_FIT, String("train"))
    # One histogram pool and one gather buffer for the whole fit rather than
    # per tree (see `tree.GrowScratch`), and one membership record refilled
    # each round (see `tree.LeafMembership`).
    # The fit's one reading of the dispatch environment and of the machine's
    # core counts (parallel.DispatchSettings). Every stage below that takes a
    # `settings` argument plans its fan-out from this value instead of asking
    # the operating system again, and asking again is what the round loop used
    # to do at every dispatch of every node. It is a snapshot: a `setenv`
    # during a fit is not observed by this fit, which is the documented
    # contract and the reason there is no cache to invalidate. It resolves
    # here rather than at module scope because resolving raises on an
    # off-ladder `MOJOTREES_CPU_FEATURE_GROUP`, and once per fit is where that
    # refusal belongs.
    var settings = DispatchSettings.resolve()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    # The two constant-hessian environment variables, read once for the whole
    # fit rather than once per histogram build and once per subtraction. The
    # grower resolves the sentinel per tree if this is not passed, which is
    # 200 reads a fit instead of 2; that is what this hoist removes. Same
    # decision either way -- the variables do not change mid-fit -- so no bits
    # move.
    # `resolve_with` is `resolve()` plus the `derivative_precision`
    # parameter, on the precedence `ConstHessianSettings.widened` states:
    # `float64` from either the parameter or the environment wins. The grower
    # folds the same parameter in again from its own `params.extra`, which is
    # deliberate and costs nothing -- the fold is monotone and idempotent, so
    # applying it at the fit and again at each tree gives the identical
    # snapshot. It is done here as well so that this loop's snapshot is the
    # one it will actually train under, rather than a value the grower
    # silently corrects a level down.
    var const_hessian_env = ConstHessianSettings.resolve_with(
        params.tree.extra.wants_float64_derivatives()
    )
    var leaves = LeafMembership()
    var by_leaf = _leaf_score_update_enabled()
    # Whether every entry of `hess` below is exactly 1.0 on every round, which
    # is what lets the histogram builders accumulate two planes instead of
    # three and rebuild the hessian plane from the count (histogram.mojo).
    # Evaluated once rather than per round because nothing it reads moves
    # inside the loop: the objective, the weight vector, and the GOSS
    # configuration are fixed for the whole fit.
    # `round_has_constant_hessian` carries the exclusions, including why row
    # bagging is not one of them. The trees are byte for byte the trees this
    # loop grew before the declaration existed.
    var const_hessian = round_has_constant_hessian(
        objective, sample_weight, goss
    )
    # ---- the bootstrap's constant-hessian exclusion ----
    #
    # THE FOURTH EXCLUSION, the one `round_has_constant_hessian` documents and
    # cannot test for. Both bootstraps multiply the round's derivatives by a
    # per-row weight (`sampling.apply_bootstrap_weights`, CatBoost's
    # `CalcWeightedData`), so under an objective whose unweighted hessian is
    # the literal 1.0 a bootstrapped round stores the DRAW into `hess`. A
    # histogram builder told to rebuild the hessian plane from the row count
    # would rebuild a plane of ones over a plane of draws, silently, and every
    # split of the fit would be chosen from it.
    #
    # Withdrawing the declaration is the first line; the second is calling the
    # sampler's own refusals on the result, which is not belt-and-braces. They
    # are the assertion that this line ran: `round_has_constant_hessian` grows
    # arms over time, this loop grows lines between the two, and a future edit
    # that computes the declaration after this point, or adds an arm that
    # re-enables it, turns a silently wrong hessian plane into an exception at
    # fit setup. `mvs_varies_hessian` and `bayesian_bootstrap_varies_hessian`
    # both test `enabled` alone, deliberately, so a `subsample = 1.0` or
    # `bagging_temperature = 0` fit pays for the third plane and gets the same
    # model -- the cheap side of the trade, and the side that cannot go stale
    # part way through a loop.
    if bootstrap.varies_hessian():
        const_hessian = False
    bootstrap.check_hessian_declaration(const_hessian)
    # MVS's lambda is derived from the PREVIOUS tree's leaf values on every
    # tree after the first (`TMvsSampler::GetLambda` reads `leafValues.back()`),
    # and a continued run is handed raw scores rather than the ensemble that
    # produced them, so the tree this loop would need does not arrive. Refused
    # rather than silently restarting the derivation from the gradients, which
    # would give round `k` of a continued run a different lambda from round `k`
    # of an uninterrupted one and quietly change which rows survive. An
    # explicit `mvs_reg` needs no derivation and continues fine, as does the
    # Bayesian bootstrap, which has no lambda at all.
    if (
        round_offset != 0
        and bootstrap.mvs.enabled
        and not bootstrap.mvs.reg_is_set
    ):
        raise Error(
            "continued training with bootstrap_type=mvs needs mvs_reg set"
            " explicitly: the derived value is the squared mean leaf-value"
            " norm of the tree before, and a continued run resumes from raw"
            " scores rather than from the ensemble, so the first new round has"
            " no previous tree to read"
        )
    # One buffer for the whole fit, as the gradient buffers are: the draw is
    # rewritten in place every round. Both stay EMPTY on the default arm --
    # `bootstrap_round` clears rather than fills them -- which is the
    # convention `refresh_mvs_bootstrap` states and the reason an unbootstrapped
    # fit moves no bits here.
    var boot_w = List[Float64]()
    var boot_audit = MvsAudit.empty()
    # The previous tree's leaf values, for MVS's derived lambda. Filled after
    # each accepted tree and read by the next round; empty on the first round
    # of a fresh fit, which is the branch `GetLambda` takes when `leafValues`
    # is empty.
    var last_leaf_values = List[Float64]()
    var n_last_leaves = 0
    # ---- ordered boosting state, all empty and untouched when it is off ----
    # `_check_ordered` runs whether or not it is on, because `validate` refuses
    # a malformed bundle a caller built and then disabled, which is a typo
    # worth reporting at the same place every other configuration error is.
    var ordered = params.ordered.copy()
    _check_ordered(
        ordered, objective, bagging, goss, class_bagging, round_offset
    )
    # The fourth sampler `_check_ordered` does not take. Same refusal and the
    # same sentence: MVS drops rows outright and the Bayesian bootstrap
    # reweights every one of them, and either changes which prefix each fold
    # was fitted on. `_check_ordered`'s signature is not widened from here
    # because it is another lane's shape; this is the one place both
    # configurations are in scope.
    if ordered.enabled and bootstrap.enabled():
        raise Error(
            "boosting_type=ordered is exclusive with bootstrap_type: a dropped"
            " or reweighted row changes which prefix each fold was fitted on"
        )
    check_ordered_hessian_declaration(ordered, const_hessian)
    # `ladder[f]` is rung `f`'s body end and `ladder[f + 1]` its tail end;
    # `plane_off[f]` is where rung `f`'s raw-score plane starts in the one flat
    # buffer; `ord_fold[q]` is which rung permuted position `q` reads.
    var ladder = List[Int]()
    var plane_off = List[Int]()
    var ord_fold = List[Int]()
    var ord_perm = List[Int]()
    var ord_plane = List[Float64]()
    var n_rung = 0
    var plane_stride = 0
    var n_perm = 1
    # Per-round scratch, allocated once for the whole fit.
    var ord_score = List[Float64]()
    var leaf_of_row = List[Int]()
    var model_values = List[Float64]()
    var rung_values = List[Float64]()
    var pgrad = List[Float64]()
    var phess = List[Float64]()
    var sub_raw = List[Float64]()
    var sub_tgt = List[Float64]()
    var sub_w = List[Float64]()
    var sub_grad = List[Float64]()
    var sub_hess = List[Float64]()
    if ordered.enabled:
        ladder = fold_ladder(n, ordered.fold_len_multiplier)
        plane_off = plane_offsets(ladder)
        n_rung = n_rungs(ladder)
        plane_stride = plane_off[n_rung]
        ord_fold = fold_ids(ladder, n)
        n_perm = ordered.permutation_count
        var block = ordered.resolve_block_size(n)
        ord_perm.reserve(n_perm * n)
        for p in range(n_perm):
            var one = ordered_permutation(ordered.seed, p, n, block)
            for q in range(n):
                ord_perm.append(one[q])
        # Every plane starts where the ensemble starts. Seeded from `raw`
        # through the permutation rather than from a scalar base score, so a
        # per-row `init_score` reaches the planes too.
        ord_plane.resize(n_perm * plane_stride, 0.0)
        for p in range(n_perm):
            for f in range(n_rung):
                var base = p * plane_stride + plane_off[f]
                for q in range(ladder[f + 1]):
                    ord_plane[base + q] = raw[ord_perm[p * n + q]]
        ord_score.resize(n, 0.0)
        leaf_of_row.resize(n, 0)
    for i in range(params.n_estimators):
        var round = round_offset + i
        # The round's wall clock is taken in two brackets, before the tree and
        # after it, because `grow_tree_leaves_profiled` adds the tree's own.
        # One outer bracket would count the tree twice and understate what
        # the phases leave unattributed.
        var pre_started = profile.clock()
        if balanced:
            refresh_class_bag(bag, class_bagging, target, round)
        else:
            refresh_bag(bag, bagging, n, round)
        var grad_started = profile.clock()
        if ordered.enabled:
            # Which permutation this tree is grown against. One draw, a pure
            # function of `(seed, round)`, taken before the gradient fill
            # because it selects which raw scores the fill reads.
            var perm_idx = permutation_choice(ordered.seed, round, n_perm)
            # The ordered raw score of every row: the score held by the LAST
            # rung whose body ends at or before this row's permuted position,
            # which is the tightest model in the ladder that has never been
            # fitted on it. Scattered back into row order so the gradient fill,
            # the grower and the score update all keep reading rows the way
            # they already do -- the permutation lives here and nowhere else.
            var pbase = perm_idx * n
            var sbase = perm_idx * plane_stride
            for q in range(n):
                ord_score[ord_perm[pbase + q]] = ord_plane[
                    sbase + plane_off[ord_fold[q]] + q
                ]
            _fill_grad_hess(
                ord_score, target, objective, sample_weight, alpha, grad, hess,
                settings,
                float64_derivatives=(
                    params.tree.extra.wants_float64_derivatives()
                ),
            )
        else:
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, grad, hess,
                settings,
                float64_derivatives=(
                    params.tree.extra.wants_float64_derivatives()
                ),
            )
        goss_round(bag, grad, hess, goss, round, learning_rate)
        # ---- CatBoost's `random_strength`, one scale per tree ----
        #
        # Before the bootstrap, and that order is CatBoost's rather than a
        # convenience: `CalcScoreStDev` reads the fold's `WeightedDerivatives`,
        # and `Bootstrap` writes `SampleWeightedDerivatives` and leaves
        # `WeightedDerivatives` alone, so the noise scale is a property of the
        # round's user-weighted derivatives and not of the draw. Reading it
        # after `bootstrap_round` would scale the noise by the sampler.
        #
        # Written onto the loop's own copy of the tree bundle, never onto
        # `params` -- see `tree_params` above. At the default of
        # `random_strength = 0` the call returns on its first comparison and
        # nothing is written.
        if noisy:
            tree_params.extra.random_score_scale = _round_random_score_scale(
                params.tree.extra, grad, n, round, learning_rate
            )
        # ---- CatBoost's `bootstrap_type` ----
        #
        # `goss_round`'s twin, in `goss_round`'s place: a row sampler that
        # reads this round's derivatives, rewrites them, and may replace the
        # row list. The two are mutually exclusive (`_check_bootstrap`) and a
        # disabled bundle returns without touching `bag`, `grad`, `hess`, or
        # the buffers.
        #
        # The lambda argument is `TMvsSampler::GetLambda`'s branch, made
        # explicit: the squared mean gradient magnitude while no tree exists,
        # the squared mean leaf-value norm of the previous tree afterwards.
        # It is computed here rather than inside the sampler because the
        # second branch reads the ensemble, which sampling.mojo cannot see.
        if bootstrap.enabled():
            # MVS WRITES ITS KEPT ROWS INTO `bag`, so the bag arriving here
            # on round 1 and later is the PREVIOUS round's draw, not a bag.
            # `bootstrap_round` refuses a non-empty row list beside MVS --
            # correctly, because a real bag would be silently intersected
            # with a draw that never saw it -- and without this clear that
            # refusal fires on the sampler's own output and every MVS fit
            # stops after exactly one tree. A real bag cannot reach here:
            # `_check_bootstrap` refuses `bootstrap_type` beside
            # `bagging_fraction`, GOSS and balanced bagging at fit setup, so
            # the only thing this can be discarding is last round's kept set,
            # which this round is about to redraw anyway.
            bag.clear()
            var auto_lambda = 0.0
            if bootstrap.mvs.enabled and not bootstrap.mvs.reg_is_set:
                if n_last_leaves > 0:
                    auto_lambda = mvs_auto_lambda_from_leaf_values(
                        last_leaf_values, n_last_leaves, 1
                    )
                else:
                    auto_lambda = mvs_auto_lambda_from_gradients(grad, n, 1)
            bootstrap_round(
                bag,
                grad,
                hess,
                boot_w,
                boot_audit,
                bootstrap,
                n,
                round,
                auto_lambda,
                1,
            )
        # Round-level work belonging to no node, so it is charged at the
        # tree's root row count and files under `root`. The GOSS rescale is
        # inside the same charge because it is the same pass over the same
        # two vectors and separating it would be a distinction without a
        # phase; it is a no-op when GOSS is off, which is the default.
        profile.charge(
            PROF_GRAD_FILL, len(bag) if len(bag) > 0 else n, grad_started
        )
        profile.note_wall(pre_started)
        var tree = grow_tree_leaves_profiled(
            profile,
            leaves,
            ledger,
            scratch,
            data,
            grad,
            hess,
            # The loop's copy, not the caller's bundle: it is `params.tree` in
            # every field but `extra.random_score_scale`, which this round just
            # computed and which is 0.0 on the default arm.
            tree_params,
            bag,
            round,
            bundling,
            const_hessian,
            const_hessian_env,
        )
        var post_started = profile.clock()
        # Hoisted out of the two ordered blocks below because they are split
        # by the degenerate-tree test: the model's values must be written
        # before that test sees them, and the rung planes must be advanced
        # only if the tree actually joins the ensemble.
        var ord_bounds = List[OutputBounds]()
        if ordered.enabled:
            # ---- the model's leaf values ----
            #
            # **Ordered boosting decides the tree's STRUCTURE, not the values
            # the model carries.** That is CatBoost's own division of labour
            # and it is worth being exact about, because it is the half every
            # summary of the mechanism drops: `train.cpp::TrainOneIteration`
            # searches the structure on `takenFold` (a ladder fold) and then
            # calls `CalcLeafValues`, which reads
            # `ctx->LearnProgress->AveragingFold` -- a plain, single-rung fold
            # over every row. `UpdateLearningFold` afterwards advances each
            # ladder fold's own approxes, which never reach the model.
            #
            # So the value written here is a Newton step on the PLAIN raw
            # scores over every row, which is exactly what the grower would
            # have written in plain mode for this structure. The grower's
            # value, fitted on the ordered scores, is overwritten.
            _ordered_leaf_of_row(tree, data, leaves, n, leaf_of_row)
            ord_bounds = node_bounds(tree, signs)
            _fill_grad_hess(
                raw, target, objective, sample_weight, alpha, pgrad, phess,
                settings,
                float64_derivatives=(
                    params.tree.extra.wants_float64_derivatives()
                ),
            )
            _ordered_fit_leaf_values(
                tree, pgrad, phess, List[Int](), 0, n, leaf_of_row,
                params.tree.lambda_l1, params.tree.lambda_reg,
                params.tree.extra.max_delta_step, ord_bounds, model_values,
            )
            for node in range(len(tree.feature)):
                if tree.feature[node] < 0:
                    tree.value[node] = model_values[node]
        if renews:
            # The membership growth just handed back, so renewal reads each
            # row's leaf off the partition instead of walking the tree once
            # per row for a leaf that was already named. Same residual per
            # row, same per-leaf order, same values (see
            # `_renew_leaf_values`).
            _renew_leaf_values(
                tree, data, target, raw, renew_w, renew_a, bag, signs,
                params.tree.extra, leaves,
            )
        # CatBoost's extra Newton steps, on the same membership and the same
        # bag. At the default of 1 this returns before it reads a row, so the
        # tree keeps exactly the value the grower wrote and no bits move.
        # Placed before the degenerate-tree test below, so that test sees the
        # value the ensemble is about to carry. Exclusive with renewal, which
        # `_check_leaf_estimation_config` refuses above rather than ordering.
        _estimate_leaf_values(
            tree, data, target, raw, objective, sample_weight, alpha,
            leaf_iters, params.tree.lambda_l1, params.tree.lambda_reg,
            params.tree.extra.max_delta_step, bag, signs, leaves, settings,
            float64_derivatives=params.tree.extra.wants_float64_derivatives(),
        )

        # A single-leaf tree with a near-zero value means the objective has
        # converged; further rounds cannot make progress. Under any row
        # sampler one such tree only says this sample had nothing to give
        # (every sampled row zero-weight, say), so the round is skipped and
        # the next sample gets its turn.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            profile.note_wall(post_started)
            if (
                bagging_enabled(bagging)
                or goss.enabled
                or balanced
                or bootstrap.enabled()
            ):
                continue
            break

        # One pass over every training row per round, over every row whether
        # or not it was in the bag, and split into equal row blocks whichever
        # of the two routes it takes. Its own phase because it belongs
        # to no node, is invisible to any per-tree instrument, and would
        # otherwise land in the unattributed remainder where nobody would look
        # for it. Charged at `n`, the dataset, not at the bag, which is what
        # lets the two routes `_add_tree_scores` chooses between -- the leaf
        # walk when the membership covers the dataset, the full-tree traversal
        # per row otherwise -- be read against the same denominator.
        var update_started = profile.clock()
        _add_tree_scores(
            raw, tree, leaves, data, learning_rate, by_leaf, 1, 0, settings
        )
        profile.charge(PROF_SCORE_UPDATE, n, update_started)
        if ordered.enabled:
            # ---- advance every rung of every permutation ----
            #
            # After the score update, and only for a tree the ensemble kept,
            # so a round that broke out above leaves the planes exactly where
            # `raw` was left.
            #
            # Rung `f` fits its own leaf values on its own body -- permuted
            # positions `[0, ladder[f])`, read at rung `f`'s own scores -- and
            # then adds `learning_rate * value` over its own tail,
            # `[0, ladder[f + 1])`. `approx_calcer.cpp::CalcApproxDeltaSimple`
            # is the same two ranges: `CalcLeafDersSimple(..., bt.BodyFinish,
            # ...)` for the fit and `UpdateApproxDeltas(..., bt.TailFinish,
            # ...)` for the apply. **These values are bookkeeping and never
            # reach the model**; the model's are the plain ones written above.
            #
            # Nothing is shared across rungs. Each fits its own `G` and `H` at
            # its own scores with the same `lambda_l2`; no scale, threshold or
            # bound derived on one rung is reused on another, which matters
            # because the rungs are systematically NOT magnitude-comparable --
            # an early rung sees a short prefix, underfits harder, and carries
            # larger residuals, monotonically in the rung index.
            #
            # Cost per round, derived and not measured: the bodies sum to
            # under `n * m / (m - 1)` and the tails to under
            # `n * (2m - 1) / (m - 1)`, so it is a bounded multiple of one
            # dataset pass per permutation -- under `2n` derivative
            # evaluations and under `3n` plane writes at the default
            # multiplier -- and not `K` passes.
            for p in range(n_perm):
                var pbase = p * n
                var sbase = p * plane_stride
                for f in range(n_rung):
                    var body = ladder[f]
                    var tail = ladder[f + 1]
                    var off = sbase + plane_off[f]
                    sub_raw.resize(body, 0.0)
                    sub_tgt.resize(body, 0.0)
                    if len(sample_weight) > 0:
                        sub_w.resize(body, 0.0)
                    for q in range(body):
                        var r = ord_perm[pbase + q]
                        sub_raw[q] = ord_plane[off + q]
                        sub_tgt[q] = target[r]
                        if len(sample_weight) > 0:
                            sub_w[q] = sample_weight[r]
                    _fill_grad_hess(
                        sub_raw, sub_tgt, objective, sub_w, alpha,
                        sub_grad, sub_hess, settings,
                        float64_derivatives=(
                            params.tree.extra.wants_float64_derivatives()
                        ),
                    )
                    # The row window is this permutation's prefix, so
                    # `_ordered_fit_leaf_values` reads `grad[i]` beside
                    # `leaf_of_row[perm[i]]` and folds in permuted order.
                    _ordered_fit_leaf_values(
                        tree, sub_grad, sub_hess, ord_perm, pbase, body,
                        leaf_of_row,
                        params.tree.lambda_l1, params.tree.lambda_reg,
                        params.tree.extra.max_delta_step, ord_bounds,
                        rung_values,
                    )
                    for q in range(tail):
                        ord_plane[off + q] += (
                            learning_rate
                            * rung_values[leaf_of_row[ord_perm[pbase + q]]]
                        )
        # MVS's derived lambda for the NEXT round, read off the tree the
        # ensemble just kept -- `leafValues.back()` in `TMvsSampler::GetLambda`.
        # Recorded here, after the degenerate-tree test, so a round that was
        # dropped does not leave its leaf values as the next round's lambda,
        # which is what `LearnProgress->LeafValues` never receiving them means
        # in CatBoost. Only when the derivation is actually used: an explicit
        # `mvs_reg`, the Bayesian bootstrap and the default arm all skip the
        # walk entirely.
        if bootstrap.mvs.enabled and not bootstrap.mvs.reg_is_set:
            n_last_leaves = _tree_leaf_values(
                tree, learning_rate, last_leaf_values
            )
        trees.append(tree^)
        profile.note_wall(post_started)

    # One block for the whole loop, and nothing at all when the profile is
    # off, so an unprofiled run's stdout is byte identical to what it was.
    profile.print_report()


def train(
    data: BinnedMatrix,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    init_score: List[Float64] = [],
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises -> Booster:
    """Train a boosted ensemble. `target` is the regression target for
    SQUARED_ERROR, HUBER, QUANTILE, and L1, {0, 1} labels for
    BINARY_LOGISTIC, or nonnegative counts for POISSON. A non-empty
    sample_weight scales each row's gradient and hessian, LightGBM style;
    a row with weight zero is ignored. `alpha` is the target quantile for
    QUANTILE and the huber transition point for HUBER (LightGBM's alpha,
    default 0.9); other objectives ignore it. `bagging` grows each tree on
    a seeded row sample (see bagging.mojo); the base score and the
    per-round score update stay on the full dataset. `goss` grows each tree
    on a gradient-based sample instead (see goss.mojo), which the same row
    list carries; the two samplers are mutually exclusive.

    `class_bagging` is LightGBM's balanced bagging: positive and negative
    rows are kept at their own rates (see sampling.mojo). It applies to
    BINARY_LOGISTIC alone and is exclusive with the other two samplers.

    `bootstrap` is CatBoost's `bootstrap_type` (see sampling.mojo): MVS at
    `subsample`, which is CatBoost's own CPU default for these objectives, or
    the Bayesian bootstrap at `bagging_temperature`. Both weight every row's
    derivatives per tree; MVS additionally drops its zero-weight rows. It is
    exclusive with the three samplers above -- `bagging_fraction` is CatBoost's
    Bernoulli bootstrap under mojotrees's name, so it is a `bootstrap_type`
    too -- and disabled by default.

    A non-empty `init_score` starts boosting from those raw scores instead
    of from the objective's own base score, LightGBM's `init_score`. The
    offset is training state, not model state: the returned ensemble has a
    base score of 0 and predicts the trees alone, so scoring new data means
    adding the caller's own offset back."""
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    _check_class_bagging(class_bagging, bagging, goss, objective)
    _check_bootstrap(bootstrap, bagging, goss, class_bagging, objective)
    params.tree.monotone.check_features(data.n_features)
    check_column_length(len(init_score), data.n_rows, "init_score")

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    var base_score = 0.0
    if len(init_score) == n:
        for r in range(n):
            raw.append(init_score[r])
    else:
        # `boost_from_average`. The `init_score` branch above is the reason
        # this is threaded here and not folded into `_base_score`'s callers
        # blindly: a per-row `init_score` ALREADY overrides the optimal
        # constant, so the two mechanisms meet on this `if` and the answer is
        # that the offset wins. That is both engines' rule -- LightGBM guards
        # `GBDT::BoostFromAverage` on `!train_score_updater_->has_init_score()`
        # (gbdt.cpp:328) and CatBoost forces `boost_from_average` to false when
        # a baseline column is present (options_helper.cpp:371-373) -- so an
        # `init_score` beside `boost_from_average=false` is not a contradiction
        # to refuse, it is the same request made twice.
        base_score = _base_score(
            target,
            objective,
            sample_weight,
            alpha,
            params.tree.extra.boost_from_average,
        )
        for _ in range(n):
            raw.append(base_score)

    var trees = List[Tree]()
    _boost_rounds(
        data,
        target,
        objective,
        params,
        sample_weight,
        alpha,
        bagging,
        goss,
        params.learning_rate,
        0,
        raw,
        trees,
        class_bagging,
        bootstrap,
    )

    return Booster(
        trees^,
        base_score,
        params.learning_rate,
        objective,
        params.tree.monotone.copy(),
    )


def train_more(
    mut booster: Booster,
    data: BinnedMatrix,
    target: List[Float64],
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    init_score: List[Float64] = [],
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises -> Int:
    """Append `params.n_estimators` more trees to a fitted ensemble and
    return how many were actually added.

    Continued training resumes from the raw scores the existing trees
    produce on `data`, so 40 rounds followed by 60 more give the ensemble
    100 rounds in one call would have given, provided the data, the
    objective, and the tree parameters are the same. Those raw scores are
    recomputed from the model on every call, which costs one pass over the
    existing trees: adding k rounds in one call is cheaper than k calls of
    one round.

    `params.n_estimators` is the number of NEW rounds, not the total.
    `params.learning_rate` must equal the ensemble's own rate and the
    monotonic constraints must match the ones already recorded on it: a
    `Booster` carries a single shrinkage factor and a single constraint
    vector for all of its trees, so neither can change part way through.
    A non-empty `init_score` is the same offset the first call trained
    under: it is training state that the ensemble does not carry, so a
    continued run has to be handed it again to resume from where the first
    one actually was.

    The objective is the ensemble's, and `data` must be binned by the
    mapper the ensemble was trained under (see `BinMapper.matches`), which
    the callers in trainset.mojo check.
    """
    if booster.linear.is_active() or params.linear.is_active():
        check_linear_tree_unconnected("train_more")
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if params.learning_rate != booster.learning_rate:
        raise Error(
            "continued training cannot change learning_rate: the ensemble"
            " shrinks every tree by one rate"
        )
    if not _same_signs(params.tree.monotone.signs, booster.monotone.signs):
        raise Error(
            "continued training cannot change monotone_constraints: the"
            " ensemble records the constraints all of its trees satisfy"
        )
    if params.n_estimators < 0:
        raise Error("n_estimators must not be negative")
    check_column_length(len(init_score), data.n_rows, "init_score")
    _check_objective(booster.objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    _check_class_bagging(class_bagging, bagging, goss, booster.objective)
    # The exclusions, and then the one refusal continued training adds:
    # `_boost_rounds` rejects a derived `mvs_reg` at a nonzero `round_offset`,
    # because the value is the previous tree's and a continuation resumes from
    # raw scores rather than from the ensemble.
    _check_bootstrap(
        bootstrap, bagging, goss, class_bagging, booster.objective
    )
    params.tree.monotone.check_features(data.n_features)
    # A `Booster` carries the trees a CEGB ledger produced, not the ledger,
    # so these rounds would start from an empty one and charge every feature
    # the first run already paid for a second time. Refused rather than
    # allowed to diverge silently; `cegb_penalty_split` reads no ledger and
    # survives the boundary untouched.
    check_cegb_continued_training(params.tree.extra.penalties.cegb, True)

    var n = data.n_rows
    var has_init = len(init_score) == n
    var raw = List[Float64](capacity=n)
    for r in range(n):
        # Accumulated in the order the first run accumulated it: the offset
        # first, then the trees in sequence. Summing the ensemble separately
        # and adding the offset afterwards is the same arithmetic in a
        # different association, and lands one ulp away often enough to
        # break the claim that 40 rounds plus 60 are the 100-round model.
        var s = booster.base_score
        if has_init:
            s += init_score[r]
        for i in range(len(booster.trees)):
            s += booster.learning_rate * booster.trees[i].predict_row(
                data, r
            )
        raw.append(s)

    var grown = List[Tree]()
    _boost_rounds(
        data,
        target,
        booster.objective,
        params,
        sample_weight,
        alpha,
        bagging,
        goss,
        booster.learning_rate,
        len(booster.trees),
        raw,
        grown,
        class_bagging,
        bootstrap,
    )
    var added = len(grown)
    for i in range(added):
        booster.trees.append(grown[i].copy())
    return added


def train_with_valid(
    data: BinnedMatrix,
    target: List[Float64],
    valid_data: BinnedMatrix,
    valid_target: List[Float64],
    objective: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    class_bagging: ClassBaggingParams = ClassBaggingParams.disabled(),
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises -> Booster:
    """Train with validation-set early stopping. Stops when the validation
    loss (MSE / log loss / huber / pinball / MAE) has not improved by more
    than min_delta for early_stopping_rounds consecutive rounds and
    truncates the ensemble to its best round. sample_weight applies to
    training rows only; the validation loss is unweighted. `bagging`
    samples training rows per tree (see bagging.mojo); validation rows are
    never bagged, so the early-stopping signal stays out of sample. `goss`
    samples training rows by gradient magnitude instead (see goss.mojo) and
    leaves the validation loss untouched in the same way. `class_bagging`
    (see sampling.mojo) is a third, exclusive training-row sampler and is
    likewise never applied to the validation rows. `bootstrap` is CatBoost's
    `bootstrap_type`, MVS or the Bayesian bootstrap, and is a fourth: it
    weights and (under MVS) drops training rows per tree, and the validation
    loss is computed on unweighted, unsampled rows exactly as it is under the
    other three."""
    if params.linear.is_active():
        check_linear_tree_unconnected("train_with_valid")
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if len(valid_target) != valid_data.n_rows:
        raise Error("valid_target length must equal valid n_rows")
    if valid_data.n_features != data.n_features:
        raise Error("valid_data must have the same features")
    _check_objective(objective, target, alpha)
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    _check_class_bagging(class_bagging, bagging, goss, objective)
    _check_bootstrap(bootstrap, bagging, goss, class_bagging, objective)
    params.tree.monotone.check_features(data.n_features)

    # Fitted once, from the training matrix alone: the validation matrix is
    # only ever scored through the trees, which name original features.
    var bundling = prepare_bundling(data, params.bundling)
    # One ledger for this ensemble, as in `_boost_rounds`. Early stopping
    # truncates the ensemble afterwards; the ledger is training state and is
    # not consulted again, so the trees that are kept are the ones this run
    # grew under it.
    var ledger = CegbLedger.create(
        params.tree.extra.penalties.cegb, data.n_features, data.n_rows
    )
    var n = data.n_rows
    # `boost_from_average`, threaded for the reason `train` threads it. This
    # loop has no `init_score` branch, so the parameter is the only thing that
    # can move the start.
    var base_score = _base_score(
        target,
        objective,
        sample_weight,
        alpha,
        params.tree.extra.boost_from_average,
    )
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)
    var valid_raw = List[Float64](capacity=valid_data.n_rows)
    for _ in range(valid_data.n_rows):
        valid_raw.append(base_score)

    var signs = params.tree.monotone.active_signs()
    var renews = objective_renews_leaves(objective)
    var renew_w = renewal_weights(objective, target, sample_weight)
    var renew_a = renewal_alpha(objective, alpha)
    # The same once-per-fit check and the same count `_boost_rounds` makes.
    var leaf_iters = params.tree.extra.leaf_estimation_iterations
    _check_leaf_estimation_config(params.tree.extra, objective, goss)
    # This loop is `_boost_rounds`'s twin and does not carry the ordered
    # ladder, so an active bundle is refused rather than silently ignored.
    check_ordered_honored(params.ordered, String("train_with_valid"))
    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var best_loss = _mean_loss(valid_raw, valid_target, objective, alpha)
    var best_n_trees = 0
    var bag = List[Int]()
    var balanced = class_bagging.enabled() and has_positive_rows(target)
    # The fit's one reading of the dispatch environment and of the machine's
    # core counts (parallel.DispatchSettings). Every stage below that takes a
    # `settings` argument plans its fan-out from this value instead of asking
    # the operating system again, and asking again is what the round loop used
    # to do at every dispatch of every node. It is a snapshot: a `setenv`
    # during a fit is not observed by this fit, which is the documented
    # contract and the reason there is no cache to invalidate. It resolves
    # here rather than at module scope because resolving raises on an
    # off-ladder `MOJOTREES_CPU_FEATURE_GROUP`, and once per fit is where that
    # refusal belongs.
    var settings = DispatchSettings.resolve()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    var leaves = LeafMembership()
    var by_leaf = _leaf_score_update_enabled()
    # The same declaration `_boost_rounds` makes, from the same predicate and
    # the same three inputs. Early stopping changes how many of these trees
    # survive and nothing about how any one of them is grown, so it has no
    # bearing on it.
    var const_hessian = round_has_constant_hessian(
        objective, sample_weight, goss
    )
    # The fourth exclusion, exactly as `_boost_rounds` makes it and for exactly
    # the reason stated there: both bootstraps put the drawn weight into `hess`,
    # so the declaration is withdrawn and then the sampler's own refusals are
    # called on the result as the assertion that it was.
    if bootstrap.varies_hessian():
        const_hessian = False
    bootstrap.check_hessian_declaration(const_hessian)
    # The loop's own copy of the tree bundle, written by nothing but
    # `random_score_scale` below. `params` stays borrowed; see the same block
    # in `_boost_rounds` for why the scale must not be written back.
    var tree_params = params.tree.copy()
    var noisy = params.tree.extra.random_strength > 0.0
    var boot_w = List[Float64]()
    var boot_audit = MvsAudit.empty()
    var last_leaf_values = List[Float64]()
    var n_last_leaves = 0
    for i in range(params.n_estimators):
        if balanced:
            refresh_class_bag(bag, class_bagging, target, i)
        else:
            refresh_bag(bag, bagging, n, i)
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess,
            settings,
            float64_derivatives=params.tree.extra.wants_float64_derivatives(),
        )
        goss_round(bag, grad, hess, goss, i, params.learning_rate)
        # `random_strength`'s per-tree scale, then the bootstrap, in that
        # order and for the reason `_boost_rounds` gives: CatBoost's
        # `CalcScoreStDev` reads the pre-bootstrap `WeightedDerivatives`.
        # This loop starts at round 0 always -- it has no `round_offset` --
        # so the model length is `i * learning_rate`.
        if noisy:
            tree_params.extra.random_score_scale = _round_random_score_scale(
                params.tree.extra, grad, n, i, params.learning_rate
            )
        if bootstrap.enabled():
            # MVS WRITES ITS KEPT ROWS INTO `bag`, so the bag arriving here
            # on round 1 and later is the PREVIOUS round's draw, not a bag.
            # `bootstrap_round` refuses a non-empty row list beside MVS --
            # correctly, because a real bag would be silently intersected
            # with a draw that never saw it -- and without this clear that
            # refusal fires on the sampler's own output and every MVS fit
            # stops after exactly one tree. A real bag cannot reach here:
            # `_check_bootstrap` refuses `bootstrap_type` beside
            # `bagging_fraction`, GOSS and balanced bagging at fit setup, so
            # the only thing this can be discarding is last round's kept set,
            # which this round is about to redraw anyway.
            bag.clear()
            var auto_lambda = 0.0
            if bootstrap.mvs.enabled and not bootstrap.mvs.reg_is_set:
                if n_last_leaves > 0:
                    auto_lambda = mvs_auto_lambda_from_leaf_values(
                        last_leaf_values, n_last_leaves, 1
                    )
                else:
                    auto_lambda = mvs_auto_lambda_from_gradients(grad, n, 1)
            bootstrap_round(
                bag,
                grad,
                hess,
                boot_w,
                boot_audit,
                bootstrap,
                n,
                i,
                auto_lambda,
                1,
            )
        var tree = grow_tree_leaves(
            leaves,
            ledger,
            scratch,
            data,
            grad,
            hess,
            tree_params,
            bag,
            i,
            bundling,
            const_hessian,
        )
        if renews:
            # The membership growth just handed back, so renewal reads each
            # row's leaf off the partition instead of walking the tree once
            # per row for a leaf that was already named. Same residual per
            # row, same per-leaf order, same values (see
            # `_renew_leaf_values`).
            _renew_leaf_values(
                tree, data, target, raw, renew_w, renew_a, bag, signs,
                params.tree.extra, leaves,
            )
        # The same extra Newton steps `_boost_rounds` takes, in the same place
        # and with the same arguments; a no-op at the default of 1.
        _estimate_leaf_values(
            tree, data, target, raw, objective, sample_weight, alpha,
            leaf_iters, params.tree.lambda_l1, params.tree.lambda_reg,
            params.tree.extra.max_delta_step, bag, signs, leaves, settings,
            float64_derivatives=params.tree.extra.wants_float64_derivatives(),
        )
        # Under any row sampler a degenerate tree indicts the sample, not
        # the run.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            if (
                bagging_enabled(bagging)
                or goss.enabled
                or balanced
                or bootstrap.enabled()
            ):
                continue
            break

        # MVS's derived lambda for the next round, off the tree the ensemble
        # kept. After the degenerate-tree test, as in `_boost_rounds`.
        if bootstrap.mvs.enabled and not bootstrap.mvs.reg_is_set:
            n_last_leaves = _tree_leaf_values(
                tree, params.learning_rate, last_leaf_values
            )
        _add_tree_scores(
            raw, tree, leaves, data, params.learning_rate, by_leaf, 1, 0,
            settings,
        )
        # The validation matrix has no membership and cannot: growth
        # partitioned the training rows and knows nothing about these. It
        # keeps the traversal, now over row blocks.
        _add_by_traversal(
            valid_raw, tree, valid_data, params.learning_rate, 1, 0, settings
        )
        trees.append(tree^)

        var loss = _mean_loss(valid_raw, valid_target, objective, alpha)
        if loss < best_loss - min_delta:
            best_loss = loss
            best_n_trees = len(trees)
        elif len(trees) - best_n_trees >= early_stopping_rounds:
            break

    while len(trees) > best_n_trees:
        _ = trees.pop()
    return Booster(
        trees^,
        base_score,
        params.learning_rate,
        objective,
        params.tree.monotone.copy(),
    )


def _multiclass_mean_loss(
    raw: List[Float64], labels: List[Int], n_classes: Int
) -> Float64:
    """Mean negative log likelihood of the true class from row-major raw
    scores, via a max-subtracted log-sum-exp."""
    var total = 0.0
    for r in range(len(labels)):
        var m = raw[r * n_classes]
        for k in range(1, n_classes):
            if raw[r * n_classes + k] > m:
                m = raw[r * n_classes + k]
        var denom = 0.0
        for k in range(n_classes):
            denom += exp(raw[r * n_classes + k] - m)
        total -= raw[r * n_classes + labels[r]] - m - log(denom)
    return total / Float64(len(labels))


def _softmax_inplace(mut scores: List[Float64], start: Int, k: Int):
    var m = scores[start]
    for i in range(1, k):
        if scores[start + i] > m:
            m = scores[start + i]
    var total = 0.0
    for i in range(k):
        var e = exp(scores[start + i] - m)
        scores[start + i] = e
        total += e
    for i in range(k):
        scores[start + i] /= total


def _fill_softmax_grad_hess(
    prob: List[Float64],
    labels: List[Int],
    k: Int,
    n_classes: Int,
    weights: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
    float64_derivatives: Bool = False,
) raises:
    """One-vs-rest gradients and hessians for class `k` from row-major
    softmax probabilities.

    Rounded to `histogram.score_t` for the same reason `fill_grad_hess` is:
    LightGBM's derivatives are `float`, the histogram kernels narrow every
    derivative they read anyway, and narrowing at the source is what keeps
    this class's numbers the same whichever accumulation path a node takes.

    `derivative_precision` reaches this exactly the way it reaches
    `fill_grad_hess`, and on the same precedence: the parameter through
    `float64_derivatives`, the environment through a live read, `float64`
    winning from either. Decided once per call, which is once per class per
    round, and used to select a compile-time instantiation of the row loop.

    **This now `raises`, and that closes the last gap in the typo refusal.**
    `check_derivative_precision` was called from every entry that could
    raise, and the multiclass and random-forest paths whose only derivative
    site is this one could not, so a mistyped
    `MOJOTREES_DERIVATIVE_PRECISION` on those paths silently selected the
    default -- one arm of an A/B running under the other's label, which is
    the exact failure the refusal exists to prevent. The only thing that
    stood in the way was `boosting_rf._multiclass_rf_gradients` not declaring
    `raises`; it does now.
    """
    check_derivative_precision()
    if derivative_precision_narrows() and not float64_derivatives:
        _fill_softmax_grad_hess_at[True](
            prob, labels, k, n_classes, weights, grad, hess
        )
    else:
        _fill_softmax_grad_hess_at[False](
            prob, labels, k, n_classes, weights, grad, hess
        )


def _fill_softmax_grad_hess_at[
    NARROW: Bool
](
    prob: List[Float64],
    labels: List[Int],
    k: Int,
    n_classes: Int,
    weights: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
):
    """`_fill_softmax_grad_hess` at one derivative precision."""
    grad.clear()
    hess.clear()
    var factor = Float64(n_classes) / Float64(n_classes - 1)
    for r in range(len(labels)):
        var p = prob[r * n_classes + k]
        var y = 1.0 if labels[r] == k else 0.0
        var w = weights[r] if len(weights) > 0 else 1.0
        grad.append(derivative[NARROW](w * (p - y)))
        # LightGBM softmax hessian: (k / (k - 1)) * p * (1 - p), floored
        # (multiclass_objective.hpp, factor_). At two classes the factor is
        # 2, which is where the old hardcoded 2.0 came from; at seven
        # classes the true factor is 7/6, and the overscaled hessian shrank
        # every leaf by ~1.7x — the real-data harness caught it as a 14%
        # multi_logloss gap on covertype. XGBoost's max(2p(1-p), eps) is a
        # different convention, not this one.
        var h = factor * p * (1.0 - p)
        if h < 1e-16:
            h = 1e-16
        hess.append(derivative[NARROW](w * h))


def _multiclass_goss_select(
    prob: List[Float64],
    labels: List[Int],
    n_classes: Int,
    weights: List[Float64],
    goss: GossParams,
    round: Int,
    float64_derivatives: Bool = False,
) raises -> GossSelection:
    """One row sample for a whole multiclass round. LightGBM sums the
    per-row `|grad * hess|` over the round's trees before sampling, so every
    class's tree is grown on the same rows and their leaf counts stay
    comparable."""
    var n = len(labels)
    var importance = List[Float64](capacity=n)
    for _ in range(n):
        importance.append(0.0)
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for k in range(n_classes):
        # The GOSS ranking is a function of the derivatives, so it has to be
        # taken at the precision the round will train at. Ranking rows at
        # float32 and then training on them at float64 samples a different
        # set of rows than either arm does on its own.
        _fill_softmax_grad_hess(
            prob, labels, k, n_classes, weights, grad, hess,
            float64_derivatives,
        )
        for r in range(n):
            importance[r] += abs(grad[r] * hess[r])
    return goss_select(importance, goss, round)


struct MulticlassBooster(Copyable, Movable):
    """Softmax ensemble: one tree per class per round, round-major, so the
    tree for (round i, class k) is trees[i * n_classes + k].

    `monotone` records the monotonic constraints every per-class tree was
    grown under, which makes each class's raw score monotone in the
    constrained features. Softmax probabilities are not guaranteed monotone;
    see monotone.mojo for the policy.
    """

    var trees: List[Tree]
    var base_scores: List[Float64]
    var n_classes: Int
    var learning_rate: Float64
    var monotone: MonotoneConstraints
    var linear: LinearEnsemble

    def __init__(
        out self,
        var trees: List[Tree],
        var base_scores: List[Float64],
        n_classes: Int,
        learning_rate: Float64,
        var monotone: MonotoneConstraints = MonotoneConstraints(),
        var linear: LinearEnsemble = LinearEnsemble(),
    ):
        self.trees = trees^
        self.base_scores = base_scores^
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.monotone = monotone^
        # Round-major like `trees`; see `Booster.linear`.
        self.linear = linear^

    def predict_raw_bins(self, bins: List[Int]) -> List[Float64]:
        var raw = List[Float64](capacity=self.n_classes)
        for k in range(self.n_classes):
            raw.append(self.base_scores[k])
        var n_rounds = len(self.trees) // self.n_classes
        for i in range(n_rounds):
            for k in range(self.n_classes):
                raw[k] += self.learning_rate * self.trees[
                    i * self.n_classes + k
                ].predict_bins(bins)
        return raw^

    def predict_proba_bins(self, bins: List[Int]) -> List[Float64]:
        var raw = self.predict_raw_bins(bins)
        _softmax_inplace(raw, 0, self.n_classes)
        return raw^

    @always_inline
    def n_iterations(self) -> Int:
        """Boosting iterations this ensemble holds. One iteration grows one
        tree per class, so this is the tree count over the class count."""
        return len(self.trees) // self.n_classes

    def predict_raw_bins_range(
        self, bins: List[Int], rng: IterationRange
    ) -> List[Float64]:
        """Per-class raw scores from the boosting iterations in `rng` alone.

        Each class's base score belongs to iteration 0 and is added only when
        the range starts there, matching the single-output rule; see
        IterationRange."""
        var raw = List[Float64](capacity=self.n_classes)
        for k in range(self.n_classes):
            raw.append(self.base_scores[k] if rng.includes_base() else 0.0)
        for i in range(rng.start, rng.stop):
            for k in range(self.n_classes):
                raw[k] += self.learning_rate * self.trees[
                    i * self.n_classes + k
                ].predict_bins(bins)
        return raw^

    def predict_proba_bins_range(
        self, bins: List[Int], rng: IterationRange
    ) -> List[Float64]:
        """Class probabilities from the iterations in `rng` alone. The
        softmax is taken over the sliced raw scores, so these are the
        probabilities of the truncated ensemble, not a slice of the full
        model's probabilities."""
        var raw = self.predict_raw_bins_range(bins, rng)
        _softmax_inplace(raw, 0, self.n_classes)
        return raw^

    def predict_batch_range(
        self,
        data: BinnedMatrix,
        rng: IterationRange,
        raw_score: Bool = False,
    ) raises -> List[Float64]:
        """Per-class predictions for every row of an already binned matrix,
        row-major (`out[r * n_classes + k]`), over row blocks.

        The single-output `Booster.predict_batch_range` with a class stride:
        the per-row body is the one `MulticlassModel.predict_batch` ran
        serially, calling the same `predict_raw_bins_range` or
        `predict_proba_bins_range`, so the softmax is still taken over one
        row's own scores and nothing crosses a row boundary. Rows write
        disjoint slots and read only the ensemble, so the block count changes
        no number.
        """
        var n = data.n_rows
        var n_features = data.n_features
        var n_classes = self.n_classes
        var out = List[Float64](capacity=n * n_classes)
        out.resize(n * n_classes, 0.0)
        var out_p = out.unsafe_ptr()

        def apply(start: Int, end: Int) {imm}:
            var bins = List[Int](capacity=n_features)
            for r in range(start, end):
                bins.clear()
                for f in range(n_features):
                    bins.append(data.bin_at(r, f))
                var scores: List[Float64]
                if raw_score:
                    scores = self.predict_raw_bins_range(bins, rng)
                else:
                    scores = self.predict_proba_bins_range(bins, rng)
                for k in range(n_classes):
                    out_p.unsafe_store(r * n_classes + k, scores[k])

        dispatch_rows(
            apply,
            n,
            n
            * (
                n_features
                + rng.n_iterations() * n_classes * _TRAVERSAL_ROW_OPS
            ),
        )
        return out^

    def leaf_ordinals_range(self, rng: IterationRange) -> List[List[Int]]:
        """The per-node leaf ordinal table of every tree in `rng`, flattened
        round-major as the ensemble stores them: entry `i * n_classes + k` is
        the table of class k's tree in the range's iteration i."""
        var tables = List[List[Int]](
            capacity=rng.n_iterations() * self.n_classes
        )
        for i in range(rng.start, rng.stop):
            for k in range(self.n_classes):
                tables.append(self.trees[i * self.n_classes + k].leaf_ordinals())
        return tables^

    def leaf_indices_bins(
        self, bins: List[Int], rng: IterationRange
    ) -> List[Int]:
        """The leaf ordinal this example reaches in each tree of `rng`,
        round-major: entry `i * n_classes + k` is class k's tree in the
        range's iteration i. The length is `rng.n_iterations() * n_classes`,
        which is 0 for an empty range."""
        var out = List[Int](capacity=rng.n_iterations() * self.n_classes)
        for i in range(rng.start, rng.stop):
            for k in range(self.n_classes):
                out.append(
                    self.trees[i * self.n_classes + k].leaf_ordinal_bins(bins)
                )
        return out^


def _boost_rounds_multiclass(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64],
    bagging: BaggingParams,
    goss: GossParams,
    learning_rate: Float64,
    round_offset: Int,
    mut raw: List[Float64],
    mut trees: List[Tree],
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises -> Int:
    """Grow `params.n_estimators` softmax rounds, appending one tree per
    class per round to `trees` and keeping the row-major `raw` scores
    (`raw[r * n_classes + k]`) in step. Returns the number of rounds grown.

    The multiclass counterpart of `_boost_rounds`, shared by the
    fit-from-scratch path and the continue-training path. `round_offset` is
    the number of rounds already grown, so the bagging draw, the GOSS
    schedule, and each tree's feature sample (seeded by
    `round * n_classes + k`) read the absolute round index and a continued
    run draws what an uninterrupted one would have drawn.

    `params.bundling` is fitted once here and shared by every class's tree in
    every round, which is what makes it worth fitting at all: the plan depends
    on the matrix, not on the gradients.

    `bootstrap` is CatBoost's `bootstrap_type` (see sampling.mojo), and this
    loop now runs it. **One draw per ROUND, shared by every class's tree**,
    which is the rule `goss` already keeps here and for the same reason: the
    classes' trees have to be grown on the same rows or their leaf counts stop
    being comparable, and CatBoost calls `Bootstrap` once per iteration, not
    once per approx dimension. The draw is taken from the round's derivatives
    laid out row-major over classes (`[r * n_classes + k]`), which is exactly
    the `n_outputs` layout `sampling.bootstrap_round`,
    `sampling._row_magnitude_squared` and CatBoost's own
    `sum_k der[dim][idx]^2` already take, so a row's keep probability is a
    function of its whole `K`-vector of gradients and not of one class's.

    Then each class's own `grad`/`hess` are multiplied by that one per-row
    weight (`sampling.apply_bootstrap_weights`, CatBoost's `CalcWeightedData`),
    which is what makes a bootstrapped softmax fit a weighted fit everywhere
    downstream -- leaf denominators, `min_child_hess`, and the renewal all see
    the drawn weight without a second vector being carried.

    **Determinism.** The draw is keyed on `(seed, round, row)` through
    `sampling._mvs_stream` / `_bootstrap_stream` and their domain constants,
    with `round` the absolute round index, so it is identical at any
    `MOJOTREES_NUM_WORKERS` and on any machine, and a continued run draws what
    an uninterrupted one would have. **No new domain constant is needed and
    none is added**: the key is `(seed, tree_index)` with `tree_index = round`,
    the same shape the dense single-output loop uses, because the draw is
    per-round there too. The per-class tree seed stays `round * n_classes + k`
    and is a different stream (`sampling._stream`, feature sampling) that this
    does not touch.
    """
    # Multiclass does not take extra Newton steps, and refuses rather than
    # ignoring the setting. The single-output form recomputes one row's
    # derivatives from that row's one raw score; a softmax row has
    # `n_classes` of them and class k's derivative depends on all of them, so
    # re-estimating class k's leaves would have to hold the other K-1 trees'
    # contributions fixed at a value they do not have yet -- this round's
    # trees for classes k+1.. are not grown. CatBoost defaults `MultiClass` to
    # **1** Newton iteration anyway (see
    # `catboost_leaf_estimation_iterations`), so there is nothing lost here
    # that CatBoost has by default.
    _refuse_leaf_estimation(params.tree.extra, "the multiclass trainers")
    # Ordered boosting is single-output here. A softmax row carries
    # `n_classes` raw scores and class k's derivative depends on all of them,
    # so a ladder would need `n_classes` planes per rung and a rule for which
    # class's tree advances which -- a second mechanism, not a widening. It is
    # also the case CatBoost itself excludes from its Ordered default
    # (`catboost_options.cpp:802-806`, `ordered_boosting.catboost_auto_is_ordered`).
    check_ordered_honored(params.ordered, String("the multiclass trainers"))
    # CatBoost's `bootstrap_type`: the three row-sampler exclusions, and the
    # derived-lambda refusal that is the softmax loop's own.
    _check_bootstrap_multiclass(
        bootstrap, bagging, goss, String("the multiclass trainers")
    )
    # ---- the bootstrap's constant-hessian exclusion, stated rather than
    # assumed ----
    #
    # This loop declares NO constant hessian and never has -- the softmax
    # hessian `(K/(K-1)) * p * (1-p)` varies per row at every class count, as
    # the comment beside `grow_tree_leaves` below says at length -- so the
    # literal `False` here is the declaration this fit trains under, and the
    # call is trivially satisfied today.
    #
    # It is made anyway, and that is the point of it. `check_hessian_declaration`
    # is an ASSERTION that the declaration and the sampler were reconciled, and
    # the failure it guards against is a future edit: a lane that gives this
    # loop a constant-hessian fast path (the softmax hessian on round 1 is
    # equal across rows, which is a tempting special case) would otherwise
    # rebuild a plane of ones over a plane of MVS draws and choose every split
    # of the fit from it, silently. With this line, that edit raises at fit
    # setup instead. `boosting.round_has_constant_hessian` cannot carry the
    # exclusion for the reason its own docstring gives: it cannot see a
    # sampler that has not run yet.
    bootstrap.check_hessian_declaration(False)
    var bundling = prepare_bundling(data, params.bundling)
    # ONE ledger across every class, not one per class. A feature computed
    # for class 0's tree is computed for the row, so charging it again for
    # class 1 would make one feature cost `n_classes` times what deploying it
    # costs. This follows from the cost being a property of the served model;
    # it has not been compared against LightGBM's multiclass path, and
    # `docs/CEGB.md` records that.
    var ledger = CegbLedger.create(
        params.tree.extra.penalties.cegb, data.n_features, data.n_rows
    )
    var n = data.n_rows
    var prob = List[Float64](capacity=n * n_classes)
    for _ in range(n * n_classes):
        prob.append(0.0)

    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    var grown = 0
    # ---- the bootstrap's buffers, all empty and untouched when it is off ----
    #
    # `mgrad`/`mhess` are the round's derivatives ROW-MAJOR over classes,
    # `[r * n_classes + k]`, which is the only layout the draw can read: MVS's
    # keep probability is a function of `sqrt(sum_k der_k^2 + lambda)`, so it
    # needs a row's whole vector at once, and the class loop below builds one
    # class at a time. They are scratch for the draw alone -- `bootstrap_round`
    # scales them by the weights it just drew and this loop then discards them,
    # because the numbers the trees are grown on are refilled per class from
    # `prob` and scaled by the same weights there.
    #
    # Allocated once for the whole fit and only when a bootstrap is configured,
    # so the default arm allocates nothing and moves no bits.
    var mgrad = List[Float64]()
    var mhess = List[Float64]()
    var boot_w = List[Float64]()
    var boot_audit = MvsAudit.empty()
    if bootstrap.enabled():
        mgrad.resize(n * n_classes, 0.0)
        mhess.resize(n * n_classes, 0.0)
    # Shared by every class's tree in every round: the histogram shape is the
    # dataset's, not the class's, so one pool serves them all.
    # The fit's one reading of the dispatch environment and of the machine's
    # core counts (parallel.DispatchSettings). Every stage below that takes a
    # `settings` argument plans its fan-out from this value instead of asking
    # the operating system again, and asking again is what the round loop used
    # to do at every dispatch of every node. It is a snapshot: a `setenv`
    # during a fit is not observed by this fit, which is the documented
    # contract and the reason there is no cache to invalidate. It resolves
    # here rather than at module scope because resolving raises on an
    # off-ladder `MOJOTREES_CPU_FEATURE_GROUP`, and once per fit is where that
    # refusal belongs.
    var settings = DispatchSettings.resolve()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    var leaves = LeafMembership()
    var by_leaf = _leaf_score_update_enabled()
    for i in range(params.n_estimators):
        var round = round_offset + i
        refresh_bag(bag, bagging, n, round)
        for r in range(n):
            for k in range(n_classes):
                prob[r * n_classes + k] = raw[r * n_classes + k]
            _softmax_inplace(prob, r * n_classes, n_classes)

        # One shared sample for the whole round, drawn before any class's
        # tree so that every class is grown on the same rows.
        var selection = GossSelection.all_rows()
        if goss.active(round, learning_rate):
            selection = _multiclass_goss_select(
                prob, labels, n_classes, sample_weight, goss, round,
                float64_derivatives=params.tree.extra.wants_float64_derivatives(),
            )
            bag = selection.rows.copy()

        # ---- CatBoost's `bootstrap_type`, ONE draw for the whole round ----
        #
        # In `goss_round`'s place in the single-output loop and for the same
        # reason: a row sampler that reads this round's derivatives, rewrites
        # them, and may replace the row list. The two are mutually exclusive
        # (`_check_bootstrap_multiclass`), as are the bags, so nothing here
        # composes with anything above it.
        #
        # The `auto_lambda` argument is 0.0 and is never read: MVS with a
        # derived lambda is refused for this loop at setup
        # (`check_mvs_reg_is_set`), so `MvsBootstrapParams.resolve_reg` always
        # takes its explicit branch, and the Bayesian bootstrap has no lambda
        # at all. Passing a real derivation here would be inventing CatBoost's
        # number rather than computing it.
        if bootstrap.enabled():
            # MVS writes its kept rows into `bag`, so what arrives here from
            # round 1 onward is the PREVIOUS round's draw and not a bag.
            # `bootstrap_round` refuses a non-empty row list beside MVS --
            # correctly, since a real bag would be silently intersected with a
            # draw that never saw it -- and without this clear that refusal
            # fires on the sampler's own output and every MVS fit stops after
            # one round. A real bag cannot reach here: bagging and GOSS are
            # both refused beside a bootstrap at setup.
            bag.clear()
            for k in range(n_classes):
                _fill_softmax_grad_hess(
                    prob, labels, k, n_classes, sample_weight, grad, hess,
                    float64_derivatives=params.tree.extra.wants_float64_derivatives(),
                )
                for r in range(n):
                    mgrad[r * n_classes + k] = grad[r]
                    mhess[r * n_classes + k] = hess[r]
            bootstrap_round(
                bag,
                mgrad,
                mhess,
                boot_w,
                boot_audit,
                bootstrap,
                n,
                round,
                0.0,
                n_classes,
            )

        var made_progress = False
        for k in range(n_classes):
            _fill_softmax_grad_hess(
                prob, labels, k, n_classes, sample_weight, grad, hess,
                float64_derivatives=params.tree.extra.wants_float64_derivatives(),
            )
            apply_goss_scaling(selection, grad, hess)
            # The round's one draw, applied to this class's derivatives. The
            # weight vector is the same for every class, which is what makes
            # the `K` trees of a round comparable, and it is empty on the
            # default arm (`bootstrap_round` clears rather than fills), so an
            # unbootstrapped fit does not even test a row.
            if len(boot_w) != 0:
                apply_bootstrap_weights(grad, hess, boot_w, n, 1)
            # Feature subsampling draws once per tree, so each class's tree
            # in a round gets its own feature set.
            #
            # No constant-hessian declaration here, and its absence is
            # deliberate rather than an omission. `_fill_softmax_grad_hess`
            # writes `(k / (k - 1)) * p * (1 - p)`, floored, which varies
            # per row with that row's class probability, so a softmax
            # hessian is not constant at any class count and is not
            # constant on the first round either, where every probability
            # is equal but the value is not 1.0. The GOSS rescale and the
            # bootstrap draw on the lines above would each break the
            # guarantee a second time; `bootstrap.check_hessian_declaration`
            # at fit setup is where that is stated in code.
            var tree = grow_tree_leaves(
                leaves,
                ledger,
                scratch,
                data,
                grad,
                hess,
                params.tree,
                bag,
                round * n_classes + k,
                bundling,
            )
            if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                made_progress = True
            # Class k owns slot `r * n_classes + k` of the row-major scores,
            # so the update is the single-output one with a stride. The rounds
            # this loop later drops (no class made progress) have already
            # updated `raw`, exactly as they did before: the trees are popped
            # and the scores are not, and this changes nothing about that.
            _add_tree_scores(
                raw, tree, leaves, data, learning_rate, by_leaf, n_classes, k,
                settings,
            )
            trees.append(tree^)

        # No class made progress: with bagging, GOSS or a bootstrap that is a
        # statement about this sample, so the round is dropped and the next
        # sample gets its turn.
        if not made_progress:
            for _ in range(n_classes):
                _ = trees.pop()
            if (
                bagging_enabled(bagging)
                or goss.enabled
                or bootstrap.enabled()
            ):
                continue
            break
        grown += 1
    return grown


def train_multiclass(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises -> MulticlassBooster:
    """Train a softmax multiclass ensemble on labels in 0..n_classes-1.
    A non-empty sample_weight scales each row's gradient and hessian and
    weights the class priors; a row with weight zero is ignored. `bagging`
    draws one bag per round and every class's tree in that round is grown
    on it, so the per-class trees stay comparable. `goss` samples the round's
    rows by summed per-class gradient magnitude instead (see goss.mojo), one
    sample per round for the same reason.

    `bootstrap` is CatBoost's `bootstrap_type` and is drawn once per round on
    the same schedule and for the same reason (see `_boost_rounds_multiclass`).
    It is exclusive with `bagging` and `goss`. **The Bayesian bootstrap is what
    CatBoost itself defaults a multiclass loss to** -- its MVS default excludes
    the multiclass-only losses, `sampling.catboost_default_bootstrap_type` cites
    the lines -- and MVS here needs an explicit `mvs_reg`, because the derived
    lambda reads a `[dim][leaf]` table that a round of `K` separate trees does
    not have. Disabled by default, so an untouched bundle takes no draw and
    every existing fit is byte for byte what it was."""
    if params.linear.is_active():
        check_linear_tree_unconnected("train_multiclass")
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    check_class_count(n_classes)
    check_class_code_range(labels, n_classes)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    params.tree.monotone.check_features(data.n_features)
    var n = data.n_rows

    # Base scores are log priors (weighted when sample_weight is given).
    var class_w = List[Float64](capacity=n_classes)
    for _ in range(n_classes):
        class_w.append(0.0)
    var total_w = 0.0
    for r in range(n):
        var w = sample_weight[r] if len(sample_weight) > 0 else 1.0
        class_w[labels[r]] += w
        total_w += w
    if total_w <= 0.0:
        raise Error("sample_weight must have a positive sum")
    var base_scores = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        base_scores.append(log(_clamp_prob(class_w[k] / total_w)))

    # Row-major raw scores and softmax scratch: raw[r * n_classes + k].
    var raw = List[Float64](capacity=n * n_classes)
    for _ in range(n):
        for k in range(n_classes):
            raw.append(base_scores[k])

    var trees = List[Tree]()
    _ = _boost_rounds_multiclass(
        data,
        labels,
        n_classes,
        params,
        sample_weight,
        bagging,
        goss,
        params.learning_rate,
        0,
        raw,
        trees,
        bootstrap,
    )

    return MulticlassBooster(
        trees^,
        base_scores^,
        n_classes,
        params.learning_rate,
        params.tree.monotone.copy(),
    )


def train_multiclass_more(
    mut booster: MulticlassBooster,
    data: BinnedMatrix,
    labels: List[Int],
    params: BoosterParams,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    bootstrap: BootstrapParams = BootstrapParams.disabled(),
) raises -> Int:
    """Append `params.n_estimators` more softmax rounds to a fitted
    multiclass ensemble and return how many rounds were actually added.

    The multiclass counterpart of `train_more`, with the same contract: 40
    rounds followed by 60 more give the ensemble 100 rounds in one call
    would have given, for the same data, labels, and tree parameters.
    `params.n_estimators` counts NEW rounds, not total ones, and a round is
    one tree per class, so the ensemble grows by `added * n_classes` trees.

    `params.learning_rate` must equal the ensemble's own rate and the
    monotonic constraints must match the ones recorded on it, for the reason
    `train_more` gives: one shrinkage factor and one constraint vector
    describe every tree in the ensemble, so neither can change part way
    through. The class count and the base scores are the ensemble's own: the
    base scores are the log class priors of the data it was first fitted on
    and are deliberately not recomputed, since re-deriving them from new
    labels would silently rewrite what every existing tree is measured
    against.
    """
    if booster.linear.is_active() or params.linear.is_active():
        check_linear_tree_unconnected("train_multiclass_more")
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if params.learning_rate != booster.learning_rate:
        raise Error(
            "continued training cannot change learning_rate: the ensemble"
            " shrinks every tree by one rate"
        )
    if not _same_signs(params.tree.monotone.signs, booster.monotone.signs):
        raise Error(
            "continued training cannot change monotone_constraints: the"
            " ensemble records the constraints all of its trees satisfy"
        )
    if params.n_estimators < 0:
        raise Error("n_estimators must not be negative")
    var n_classes = booster.n_classes
    check_class_code_range(labels, n_classes)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    params.tree.monotone.check_features(data.n_features)
    # As in `train_more`: the ledger is training state and is not in the
    # model, so a resumed run would recharge every first use.
    check_cegb_continued_training(params.tree.extra.penalties.cegb, True)

    # Rebuild the raw scores the existing rounds produce. Accumulating class
    # by class in round order matches the order `_boost_rounds_multiclass`
    # accumulated them in, so a continued run resumes from bitwise the same
    # scores an uninterrupted one would hold.
    var n = data.n_rows
    var raw = List[Float64](capacity=n * n_classes)
    for _ in range(n):
        for k in range(n_classes):
            raw.append(booster.base_scores[k])
    var n_rounds = len(booster.trees) // n_classes
    for i in range(n_rounds):
        for k in range(n_classes):
            _add_by_traversal(
                raw,
                booster.trees[i * n_classes + k],
                data,
                booster.learning_rate,
                n_classes,
                k,
            )

    # Grown into a list of its own and merged only once the loop has
    # returned, so a round that raises leaves the ensemble as it was.
    var grown = List[Tree]()
    var added = _boost_rounds_multiclass(
        data,
        labels,
        n_classes,
        params,
        sample_weight,
        bagging,
        goss,
        booster.learning_rate,
        n_rounds,
        raw,
        grown,
        # No `round_offset` refusal of its own, unlike the single-output loop:
        # the derived MVS lambda that refusal protects is already refused
        # outright for every softmax round (`check_mvs_reg_is_set`), so an
        # explicit `mvs_reg` -- the only MVS that reaches here -- continues
        # exactly as the Bayesian bootstrap does, and both key their draw on
        # the ABSOLUTE round index, which is what `n_rounds` above passes.
        bootstrap,
    )
    for i in range(len(grown)):
        booster.trees.append(grown[i].copy())
    return added


def train_multiclass_with_valid(
    data: BinnedMatrix,
    labels: List[Int],
    valid_data: BinnedMatrix,
    valid_labels: List[Int],
    n_classes: Int,
    params: BoosterParams,
    early_stopping_rounds: Int,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MulticlassBooster:
    """Train a softmax multiclass ensemble with validation-set early
    stopping. Stops when the validation multiclass log loss has not
    improved by more than min_delta for early_stopping_rounds consecutive
    rounds and truncates the ensemble to its best round (a round is one
    tree per class). sample_weight applies to training rows only; the
    validation loss is unweighted. `bagging` samples training rows per
    round; validation rows are never bagged. `goss` is the gradient-based
    alternative sampler (see goss.mojo), also drawn once per round."""
    if params.linear.is_active():
        check_linear_tree_unconnected("train_multiclass_with_valid")
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if len(valid_labels) != valid_data.n_rows:
        raise Error("valid_labels length must equal valid n_rows")
    if valid_data.n_features != data.n_features:
        raise Error("valid_data must have the same features")
    check_class_count(n_classes)
    check_class_code_range(labels, n_classes)
    check_class_code_range(valid_labels, n_classes, "valid label")
    if early_stopping_rounds < 1:
        raise Error("early_stopping_rounds must be positive")
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    params.tree.monotone.check_features(data.n_features)
    # The same refusal `_boost_rounds_multiclass` makes, for the same reason:
    # a softmax row's derivative for class k reads every class's raw score.
    _refuse_leaf_estimation(params.tree.extra, "the multiclass trainers")
    # And the same ordered refusal, for the same reason.
    check_ordered_honored(params.ordered, String("the multiclass trainers"))
    # Fitted once, from the training matrix alone: the validation matrix is
    # only ever scored through the trees, which name original features.
    var bundling = prepare_bundling(data, params.bundling)
    var n = data.n_rows
    var n_valid = valid_data.n_rows

    # Base scores are log priors of the TRAINING labels (weighted when
    # sample_weight is given).
    var class_w = List[Float64](capacity=n_classes)
    for _ in range(n_classes):
        class_w.append(0.0)
    var total_w = 0.0
    for r in range(n):
        var w = sample_weight[r] if len(sample_weight) > 0 else 1.0
        class_w[labels[r]] += w
        total_w += w
    if total_w <= 0.0:
        raise Error("sample_weight must have a positive sum")
    var base_scores = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        base_scores.append(log(_clamp_prob(class_w[k] / total_w)))

    # One ledger across every round and every class, as in
    # `_boost_rounds_multiclass`.
    var ledger = CegbLedger.create(
        params.tree.extra.penalties.cegb, data.n_features, data.n_rows
    )

    # Row-major raw scores for both sets: raw[r * n_classes + k].
    var raw = List[Float64](capacity=n * n_classes)
    for _ in range(n):
        for k in range(n_classes):
            raw.append(base_scores[k])
    var valid_raw = List[Float64](capacity=n_valid * n_classes)
    for _ in range(n_valid):
        for k in range(n_classes):
            valid_raw.append(base_scores[k])
    var prob = List[Float64](capacity=n * n_classes)
    for _ in range(n * n_classes):
        prob.append(0.0)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var best_loss = _multiclass_mean_loss(valid_raw, valid_labels, n_classes)
    var best_n_rounds = 0
    var n_rounds = 0
    var bag = List[Int]()
    # The fit's one reading of the dispatch environment and of the machine's
    # core counts (parallel.DispatchSettings). Every stage below that takes a
    # `settings` argument plans its fan-out from this value instead of asking
    # the operating system again, and asking again is what the round loop used
    # to do at every dispatch of every node. It is a snapshot: a `setenv`
    # during a fit is not observed by this fit, which is the documented
    # contract and the reason there is no cache to invalidate. It resolves
    # here rather than at module scope because resolving raises on an
    # off-ladder `MOJOTREES_CPU_FEATURE_GROUP`, and once per fit is where that
    # refusal belongs.
    var settings = DispatchSettings.resolve()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    var leaves = LeafMembership()
    var by_leaf = _leaf_score_update_enabled()
    for i in range(params.n_estimators):
        refresh_bag(bag, bagging, n, i)
        for r in range(n):
            for k in range(n_classes):
                prob[r * n_classes + k] = raw[r * n_classes + k]
            _softmax_inplace(prob, r * n_classes, n_classes)

        # One shared sample for the whole round, drawn before any class's
        # tree so that every class is grown on the same rows.
        var selection = GossSelection.all_rows()
        if goss.active(i, params.learning_rate):
            selection = _multiclass_goss_select(
                prob, labels, n_classes, sample_weight, goss, i,
                float64_derivatives=params.tree.extra.wants_float64_derivatives(),
            )
            bag = selection.rows.copy()

        var made_progress = False
        for k in range(n_classes):
            _fill_softmax_grad_hess(
                prob, labels, k, n_classes, sample_weight, grad, hess,
                float64_derivatives=params.tree.extra.wants_float64_derivatives(),
            )
            apply_goss_scaling(selection, grad, hess)
            # Feature subsampling draws once per tree, so each class's tree
            # in a round gets its own feature set.
            #
            # No constant-hessian declaration here, and its absence is
            # deliberate rather than an omission. `_fill_softmax_grad_hess`
            # writes `(k / (k - 1)) * p * (1 - p)`, floored, which varies
            # per row with that row's class probability, so a softmax
            # hessian is not constant at any class count and is not
            # constant on the first round either, where every probability
            # is equal but the value is not 1.0. The GOSS rescale on the
            # line above would break the guarantee a second time.
            var tree = grow_tree_leaves(
                leaves,
                ledger,
                scratch,
                data,
                grad,
                hess,
                params.tree,
                bag,
                i * n_classes + k,
                bundling,
            )
            if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                made_progress = True
            _add_tree_scores(
                raw,
                tree,
                leaves,
                data,
                params.learning_rate,
                by_leaf,
                n_classes,
                k,
                settings,
            )
            # The validation rows were never partitioned, so they keep the
            # traversal; see `train_with_valid`.
            _add_by_traversal(
                valid_raw,
                tree,
                valid_data,
                params.learning_rate,
                n_classes,
                k,
                settings,
            )
            trees.append(tree^)

        # Popped trees are all single-leaf with value ~0, so the score
        # updates above were no-ops. Under bagging or GOSS the next sample
        # still gets its turn.
        if not made_progress:
            for _ in range(n_classes):
                _ = trees.pop()
            if bagging_enabled(bagging) or goss.enabled:
                continue
            break
        n_rounds += 1

        var loss = _multiclass_mean_loss(valid_raw, valid_labels, n_classes)
        if loss < best_loss - min_delta:
            best_loss = loss
            best_n_rounds = n_rounds
        elif n_rounds - best_n_rounds >= early_stopping_rounds:
            break

    while len(trees) > best_n_rounds * n_classes:
        _ = trees.pop()
    return MulticlassBooster(
        trees^,
        base_scores^,
        n_classes,
        params.learning_rate,
        params.tree.monotone.copy(),
    )
