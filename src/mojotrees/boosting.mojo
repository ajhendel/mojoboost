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
from .parallel import dispatch_rows, elementwise_row_ops
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
from .monotone import MonotoneConstraints
from .sampling import (
    ClassBaggingParams,
    has_positive_rows,
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
    LINK_SIGMOID,
    MAPE as _MAPE,
    POISSON as _POISSON,
    QUANTILE as _QUANTILE,
    SQUARED_ERROR as _SQUARED_ERROR,
    TWEEDIE as _TWEEDIE,
    check_objective_param,
    objective_link,
    objective_renews_leaves,
)
from .cegb import CegbLedger, check_cegb_continued_training
from .histogram import objective_has_constant_hessian
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
from .tree_parameters_extra import ExtraTreeParams, finish_leaf_output
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


# `objective_renews_leaves` is objective_registry's rule (LightGBM's
# `RenewTreeOutput`), imported above and re-exported from here under the
# name the boosting loop and its callers have always used; the leaf renewal
# it decides is `_renew_leaf_values` below.


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
) raises -> Float64:
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
) raises:
    """Per-row first and second derivatives of `objective` at the current raw
    scores, written into `grad` and `hess`.

    Both buffers are resized to `len(target)` and fully overwritten, so the
    boosting loop hands the same two lists back every round and the training
    run allocates them once. Rows are independent, so the work splits across
    contiguous row blocks; each row's arithmetic is unchanged and no value is
    summed across rows, which makes the output bit-identical at every worker
    count.

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
    _fill_grad_hess_into(
        grad, hess, raw, target, objective, weights, alpha, n
    )


def _fill_grad_hess_into(
    mut grad: List[Float64],
    mut hess: List[Float64],
    raw: List[Float64],
    target: List[Float64],
    objective: Int,
    weights: List[Float64],
    alpha: Float64,
    n: Int,
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
                gp.unsafe_store(r, w * (p - tgt_p.unsafe_load(r)))
                var h = p * (1.0 - p)
                if h < 1e-16:
                    h = 1e-16
                hp.unsafe_store(r, w * h)
        elif objective == GAMMA:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                # d/draw of raw + y * exp(-raw), the gamma deviance under
                # the log link.
                var y_over_mu = tgt_p.unsafe_load(r) * exp(
                    -raw_p.unsafe_load(r)
                )
                gp.unsafe_store(r, w * (1.0 - y_over_mu))
                hp.unsafe_store(r, w * y_over_mu)
        elif objective == TWEEDIE:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var raw_r = raw_p.unsafe_load(r)
                var y = tgt_p.unsafe_load(r)
                # rho is the variance power in (1, 2), so 1 - rho < 0 and
                # 2 - rho > 0 and the hessian below stays nonnegative.
                var e1 = exp((1.0 - alpha) * raw_r)
                var e2 = exp((2.0 - alpha) * raw_r)
                gp.unsafe_store(r, w * (-y * e1 + e2))
                hp.unsafe_store(
                    r, w * (-y * (1.0 - alpha) * e1 + (2.0 - alpha) * e2)
                )
        elif objective == MAPE:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var y = tgt_p.unsafe_load(r)
                # L1 scaled by the label weight: the absolute error becomes
                # a relative one, which is what MAPE measures.
                var lw = w * _mape_label_weight(y)
                gp.unsafe_store(
                    r, lw * _sign(raw_p.unsafe_load(r) - y)
                )
                hp.unsafe_store(r, lw)
        elif objective == FAIR:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var d = raw_p.unsafe_load(r) - tgt_p.unsafe_load(r)
                var denom = abs(d) + alpha
                gp.unsafe_store(r, w * alpha * d / denom)
                hp.unsafe_store(r, w * alpha * alpha / (denom * denom))
        elif objective == POISSON:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var raw_r = raw_p.unsafe_load(r)
                var mu = exp(raw_r)
                gp.unsafe_store(r, w * (mu - tgt_p.unsafe_load(r)))
                hp.unsafe_store(
                    r, w * exp(raw_r + _POISSON_MAX_DELTA_STEP)
                )
        elif objective == HUBER:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var diff = raw_p.unsafe_load(r) - tgt_p.unsafe_load(r)
                if abs(diff) <= alpha:
                    gp.unsafe_store(r, w * diff)
                else:
                    gp.unsafe_store(r, w * _sign(diff) * alpha)
                hp.unsafe_store(r, w)
        elif objective == QUANTILE:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                var diff = raw_p.unsafe_load(r) - tgt_p.unsafe_load(r)
                if diff >= 0.0:
                    gp.unsafe_store(r, w * (1.0 - alpha))
                else:
                    gp.unsafe_store(r, w * -alpha)
                hp.unsafe_store(r, w)
        elif objective == L1:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                gp.unsafe_store(
                    r,
                    w * _sign(raw_p.unsafe_load(r) - tgt_p.unsafe_load(r)),
                )
                hp.unsafe_store(r, w)
        else:
            for r in range(start, end):
                var w = w_p.unsafe_load(r) if weighted else 1.0
                gp.unsafe_store(
                    r, w * (raw_p.unsafe_load(r) - tgt_p.unsafe_load(r))
                )
                hp.unsafe_store(r, w)

    # A row here is a handful of flops on three sequential arrays, perhaps a
    # sixteenth of the cost of a histogram op's scattered read-modify-write,
    # so the work estimate is scaled down to match. The scale matters: with
    # the unscaled count, 100k rows asked for one task per core and
    # bench/bench_profile.mojo timed the fan-out well below the serial path,
    # the work per task being far too small to pay for scheduling it.
    dispatch_rows(block, n, elementwise_row_ops(n))


def _fill_grad_hess(
    raw: List[Float64],
    target: List[Float64],
    objective: Int,
    weights: List[Float64],
    alpha: Float64,
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    fill_grad_hess(raw, target, objective, weights, alpha, grad, hess)


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
    pass one renews exactly as before."""
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
    for i in range(n_used):
        var r = bag[i] if len(bag) > 0 else i
        var node = tree.leaf_index_row(data, r)
        leaf_residuals[node].append(target[r] - raw[r])
        if len(weights) > 0:
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

    def __init__(
        out self,
        n_estimators: Int,
        learning_rate: Float64,
        var tree: TreeParams,
        var bundling: EfbSettings = EfbSettings.disabled(),
        var linear: LinearParams = LinearParams(),
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

        The body is the per-row body `Model.predict_batch` ran serially, moved
        here unchanged: gather the row's bins, then call the same
        `predict_raw_bins_range` or `predict_bins_range` this ensemble has
        always been asked. `data.bin_at(r, f)` is `bins[f * n_rows + r]`,
        which is the value that loop gathered.

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

        # Per block, not per row: the same reuse the serial loop had.
        def apply(start: Int, end: Int) {imm}:
            var bins = List[Int](capacity=n_features)
            for r in range(start, end):
                bins.clear()
                for f in range(n_features):
                    bins.append(data.bin_at(r, f))
                if raw_score:
                    out_p.unsafe_store(
                        r, self.predict_raw_bins_range(bins, rng)
                    )
                else:
                    out_p.unsafe_store(r, self.predict_bins_range(bins, rng))

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

    dispatch_rows(apply, n, n * _TRAVERSAL_ROW_OPS)


def _add_by_leaf(
    mut raw: List[Float64],
    tree: Tree,
    leaves: LeafMembership,
    learning_rate: Float64,
    n_rows: Int,
    stride: Int,
    offset: Int,
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

    Rows are independent accumulators and each appears in exactly one leaf, so
    a row is added to once here as it was once there. Leaves are handed out to
    blocks, and blocks touch disjoint sets of rows, so the block count changes
    nothing.

    The caller must have checked `leaves.covers_all_rows`; this adds nothing
    to a row no leaf holds.
    """
    var n_leaves = leaves.n_leaves()
    var raw_p = raw.unsafe_ptr()

    def apply(l_start: Int, l_end: Int) {imm}:
        for l in range(l_start, l_end):
            var value = tree.value[leaves.node[l]]
            # A `ref` binding, because `leaves.rows[l]` in an expression
            # materializes a copy of the row list and a pointer taken from
            # that copy dangles the moment the expression ends. The bound
            # reference is the list the grower handed back.
            ref rows = leaves.rows[l]
            var rows_p = rows.unsafe_ptr()
            for i in range(len(rows)):
                var slot = Int(rows_p.unsafe_load(i)) * stride + offset
                raw_p.unsafe_store(
                    slot, fma(learning_rate, value, raw_p.unsafe_load(slot))
                )

    # The index space split over blocks is the leaves, not the rows, because a
    # leaf's rows are the contiguous unit here. Leaf-wise growth does not make
    # equal leaves, so the blocks are not equal work; that imbalance has not
    # been measured and is the obvious thing to revisit.
    dispatch_rows(apply, n_leaves, n_rows * _LEAF_ROW_OPS)


def _add_tree_scores(
    mut raw: List[Float64],
    tree: Tree,
    leaves: LeafMembership,
    data: BinnedMatrix,
    learning_rate: Float64,
    by_leaf: Bool,
    stride: Int = 1,
    offset: Int = 0,
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
            raw, tree, leaves, learning_rate, data.n_rows, stride, offset
        )
        return
    _add_by_traversal(raw, tree, data, learning_rate, stride, offset)


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
    """
    if params.linear.is_active():
        check_linear_tree_unconnected("the binned-only boosting trainers")
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
    var scratch = GrowScratch(data.n_features, data.n_bins)
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
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess
        )
        goss_round(bag, grad, hess, goss, round, learning_rate)
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
            params.tree,
            bag,
            round,
            bundling,
            const_hessian,
        )
        var post_started = profile.clock()
        if renews:
            _renew_leaf_values(
                tree, data, target, raw, renew_w, renew_a, bag, signs,
                params.tree.extra,
            )

        # A single-leaf tree with a near-zero value means the objective has
        # converged; further rounds cannot make progress. Under any row
        # sampler one such tree only says this sample had nothing to give
        # (every sampled row zero-weight, say), so the round is skipped and
        # the next sample gets its turn.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            profile.note_wall(post_started)
            if bagging_enabled(bagging) or goss.enabled or balanced:
                continue
            break

        # One pass over every training row per round, serial, over every row
        # whether or not it was in the bag. Its own phase because it belongs
        # to no node, is invisible to any per-tree instrument, and would
        # otherwise land in the unattributed remainder where nobody would look
        # for it. Charged at `n`, the dataset, not at the bag, which is what
        # lets the two routes `_add_tree_scores` chooses between -- the leaf
        # walk when the membership covers the dataset, the full-tree traversal
        # per row otherwise -- be read against the same denominator.
        var update_started = profile.clock()
        _add_tree_scores(raw, tree, leaves, data, learning_rate, by_leaf)
        profile.charge(PROF_SCORE_UPDATE, n, update_started)
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
    params.tree.monotone.check_features(data.n_features)
    check_column_length(len(init_score), data.n_rows, "init_score")

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    var base_score = 0.0
    if len(init_score) == n:
        for r in range(n):
            raw.append(init_score[r])
    else:
        base_score = _base_score(target, objective, sample_weight, alpha)
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
    likewise never applied to the validation rows."""
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
    var base_score = _base_score(target, objective, sample_weight, alpha)
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
    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var best_loss = _mean_loss(valid_raw, valid_target, objective, alpha)
    var best_n_trees = 0
    var bag = List[Int]()
    var balanced = class_bagging.enabled() and has_positive_rows(target)
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
    for i in range(params.n_estimators):
        if balanced:
            refresh_class_bag(bag, class_bagging, target, i)
        else:
            refresh_bag(bag, bagging, n, i)
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess
        )
        goss_round(bag, grad, hess, goss, i, params.learning_rate)
        var tree = grow_tree_leaves(
            leaves,
            ledger,
            scratch,
            data,
            grad,
            hess,
            params.tree,
            bag,
            i,
            bundling,
            const_hessian,
        )
        if renews:
            _renew_leaf_values(
                tree, data, target, raw, renew_w, renew_a, bag, signs,
                params.tree.extra,
            )
        # Under any row sampler a degenerate tree indicts the sample, not
        # the run.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            if bagging_enabled(bagging) or goss.enabled or balanced:
                continue
            break

        _add_tree_scores(
            raw, tree, leaves, data, params.learning_rate, by_leaf
        )
        # The validation matrix has no membership and cannot: growth
        # partitioned the training rows and knows nothing about these. It
        # keeps the traversal, now over row blocks.
        _add_by_traversal(
            valid_raw, tree, valid_data, params.learning_rate, 1, 0
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
):
    """One-vs-rest gradients and hessians for class `k` from row-major
    softmax probabilities."""
    grad.clear()
    hess.clear()
    var factor = Float64(n_classes) / Float64(n_classes - 1)
    for r in range(len(labels)):
        var p = prob[r * n_classes + k]
        var y = 1.0 if labels[r] == k else 0.0
        var w = weights[r] if len(weights) > 0 else 1.0
        grad.append(w * (p - y))
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
        hess.append(w * h)


def _multiclass_goss_select(
    prob: List[Float64],
    labels: List[Int],
    n_classes: Int,
    weights: List[Float64],
    goss: GossParams,
    round: Int,
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
        _fill_softmax_grad_hess(
            prob, labels, k, n_classes, weights, grad, hess
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
    """
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
    # Shared by every class's tree in every round: the histogram shape is the
    # dataset's, not the class's, so one pool serves them all.
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
                prob, labels, n_classes, sample_weight, goss, round
            )
            bag = selection.rows.copy()

        var made_progress = False
        for k in range(n_classes):
            _fill_softmax_grad_hess(
                prob, labels, k, n_classes, sample_weight, grad, hess
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
                raw, tree, leaves, data, learning_rate, by_leaf, n_classes, k
            )
            trees.append(tree^)

        # No class made progress: with bagging or GOSS that is a statement
        # about this sample, so the round is dropped and the next sample
        # gets its turn.
        if not made_progress:
            for _ in range(n_classes):
                _ = trees.pop()
            if bagging_enabled(bagging) or goss.enabled:
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
) raises -> MulticlassBooster:
    """Train a softmax multiclass ensemble on labels in 0..n_classes-1.
    A non-empty sample_weight scales each row's gradient and hessian and
    weights the class priors; a row with weight zero is ignored. `bagging`
    draws one bag per round and every class's tree in that round is grown
    on it, so the per-class trees stay comparable. `goss` samples the round's
    rows by summed per-class gradient magnitude instead (see goss.mojo), one
    sample per round for the same reason."""
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
                prob, labels, n_classes, sample_weight, goss, i
            )
            bag = selection.rows.copy()

        var made_progress = False
        for k in range(n_classes):
            _fill_softmax_grad_hess(
                prob, labels, k, n_classes, sample_weight, grad, hess
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
