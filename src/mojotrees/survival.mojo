"""CatBoost's two survival objectives: `Cox` and `SurvivalAft`.

Both are **off by default and reachable only from this file**. Nothing in
`boosting.mojo`, `objective_registry.mojo` or any binding changes; no existing
default moves. `docs/design/CATBOOST_CATALOG.md` A17 and A18 carry the source
verification, and every formula below cites the CatBoost function it came
from.

CatBoost's derivative sign convention, once, because it is easy to invert
--------------------------------------------------------------------------
CatBoost stores `Der1 = -dL/da` and `Der2 = -d2L/da2`. Its `TRMSEError`
returns `CalcDer = target - approx` against a loss whose gradient is
`approx - target`, and `TDiagonalHessian::SolveNewtonEquation` then computes
`-SumDer / (SumDer2 - l2)`. mojotrees stores the loss's own `grad` and `hess`
and computes `-SumGrad / (SumHess + l2)`, so **every formula transcribed here
is negated once against the CatBoost line it cites**, and the negation is
noted at each site.

Cox (A17)
---------
One signed label column: `abs(y)` is the time, `y > 0` is an event, `y <= 0`
is right-censored. The risk set is a **suffix of the stable sort by
`abs(y)`**, which is not the same thing as "every row with time at least
`t_i`" and differs from it exactly on ties. CatBoost applies **no tie
correction**: the second of two tied events does not see the first in its
risk set. That is neither Breslow nor Efron, and it is reproduced here
exactly, with the tiebreak stated (original row index) rather than inherited
from whichever sort is linked in.

The gradient of every row depends on every other row, so `cox_grad_hess` is a
single serial scan in sorted order and no part of it is split across workers.
That is deliberate: a suffix-sum plus a prefix-sum would parallelize it, and
both would be summed in a worker-count-dependent order. Determinism across
`MOJOTREES_NUM_WORKERS` is not negotiable and Cox is one round's `O(n log n)`
sort plus one `O(n)` pass, which is not where a fit spends its time.

`cox_varies_hessian` is unconditionally `True`. Cox's hessian is per-row,
non-constant and coupled, and declaring `histogram.CONSTANT_HESSIAN` beside
it would rebuild the hessian plane from the row count and be silently wrong.
`check_cox_hessian_declaration` refuses that pairing, in the shape of
`sampling.check_mvs_hessian_declaration`.

SurvivalAft (A18)
-----------------
**Two label columns per row**, a lower and an upper bound, with `-1` the
unbounded sentinel in either column. `target_matrix.TargetMatrix` is that
contract and `target_matrix.check_survival_aft` is the validation CatBoost
does not do. Approx dimension is 1 (`approx_dimension.cpp`), so the tree
shape is ordinary; the input path is the whole difficulty.

The model predicts `log` of the survival time. `exp(raw)` is a time, which is
`LINK_EXP`, which is a registry fact this lane does not own -- until the glue
lands, use `survival_aft_predicted_time` rather than `Booster.response`.
"""

from std.math import erf, exp, isfinite, isnan, log, sqrt

from .binning import BinnedMatrix
from .boosting import Booster, BoosterParams
from .histogram import (
    check_derivative_precision,
    derivative,
    derivative_precision_narrows,
)
from .target_matrix import AFT_UNBOUNDED, TargetMatrix
from .tree import Tree, grow_tree

# ---------------------------------------------------------------------------
# Proposed objective codes
# ---------------------------------------------------------------------------
# These belong in `objective_registry.mojo`, which is glue this lane does not
# own; the lane report carries the literal diff. They are staged here so this
# module compiles and trains on its own, and they are the next two free
# single-output codes after `CROSS_ENTROPY = 12`. A model serialized with one
# of them before the registry diff lands carries a code the registry does not
# know, which resolves to `LINK_IDENTITY` -- correct for `COX`, wrong for
# `SURVIVAL_AFT`, and the reason `survival_aft_predicted_time` exists.
comptime COX = 13
comptime SURVIVAL_AFT = 14

# `dist`, from `BuildError` in
# `catboost/private/libs/algo/tensor_search_helpers.cpp`, which accepts
# exactly `dist` and `scale` and defaults `dist` to Normal.
comptime AFT_DIST_NORMAL = 0
comptime AFT_DIST_LOGISTIC = 1
comptime AFT_DIST_EXTREME = 2

# `NCB::TDerivativeConstants`, transcribed from
# `catboost/private/libs/algo_helpers/survival_aft_utils.h`.
comptime AFT_MIN_FIRST_DER = -15.0
comptime AFT_MAX_FIRST_DER = 15.0
comptime AFT_MIN_SECOND_DER = 1e-16
comptime AFT_MAX_SECOND_DER = 15.0
comptime AFT_EPSILON = 1e-12

# `NCB::ECensoredType`.
comptime AFT_UNCENSORED = 0
comptime AFT_INTERVAL_CENSORED = 1
comptime AFT_RIGHT_CENSORED = 2
comptime AFT_LEFT_CENSORED = 3

# `NCB::EDerivativeOrder`.
comptime AFT_DER_FIRST = 0
comptime AFT_DER_SECOND = 1

# The floor `TCoxMetric::Eval` puts under the risk-set sum
# (`std::max(expPSum, 1e-20)`). The derivative code in `TCoxError` has no such
# guard and divides by the bare sum; we use CatBoost's own metric constant on
# both sides, because a risk-set sum driven to zero by cancellation at the
# last event would otherwise produce an infinite gradient. Class (3),
# bit-moving, and it moves bits only where CatBoost produces an infinity.
comptime COX_RISK_SUM_FLOOR = 1e-20


# ---------------------------------------------------------------------------
# The stable order Cox's risk set is a suffix of
# ---------------------------------------------------------------------------


def cox_sort_order(target: List[Float64]) -> List[Int]:
    """Row indices ordered ascending by `abs(target)`, ties by row index.

    CatBoost's `ArgSort` (`error_functions.cpp:110-127`) is
    `StableSort(..., abs(targets[lhs]) < abs(targets[rhs]))`, so the tiebreak
    is the original index and the risk set of a tied event depends on it. A
    bottom-up merge sort with a `<=` merge test is stable by construction, so
    the tiebreak here is a property of this function rather than of whichever
    sort implementation is linked in -- which is what makes the result the
    same on every machine.

    The key is `abs(target)`, computed once into a scratch column rather than
    recomputed inside the merge comparison: `n log n` comparisons against `n`
    absolute values.
    """
    var n = len(target)
    var key = List[Float64](capacity=n)
    for i in range(n):
        var v = target[i]
        key.append(-v if v < 0.0 else v)

    var idx = List[Int](capacity=n)
    for i in range(n):
        idx.append(i)
    var buf = List[Int](capacity=n)
    for _ in range(n):
        buf.append(0)

    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            if mid > n:
                mid = n
            var hi = lo + 2 * width
            if hi > n:
                hi = n
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if key[idx[i]] <= key[idx[j]]:
                    buf[k] = idx[i]
                    i += 1
                else:
                    buf[k] = idx[j]
                    j += 1
                k += 1
            while i < mid:
                buf[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                buf[k] = idx[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(n):
            idx[t] = buf[t]
        width *= 2
    return idx^


def cox_signed_target(
    time: List[Float64], event: List[Int]
) raises -> List[Float64]:
    """Build CatBoost's one signed Cox column from a time and an event flag.

    `y = time` for an event and `y = -time` for a censored observation, which
    is what `abs(y)` / `y > 0` decode. The check that matters: the event test
    is `y > 0` **strictly**, so an event at time 0 would encode as `y == 0`
    and be silently read as censored. CatBoost accepts that row and gets a
    wrong answer; this refuses it.
    """
    if len(time) != len(event):
        raise Error("time and event must have the same length")
    var out = List[Float64](capacity=len(time))
    for r in range(len(time)):
        var t = time[r]
        if not isfinite(t):
            raise Error(String("Cox time at row ", r, " is not finite"))
        if t < 0.0:
            raise Error(String("Cox time at row ", r, " is negative"))
        if event[r] != 0 and event[r] != 1:
            raise Error(String("Cox event at row ", r, " must be 0 or 1"))
        if event[r] == 1:
            if not (t > 0.0):
                raise Error(
                    String(
                        "Cox event at row ",
                        r,
                        " has time 0; the encoding is 'y > 0 means event' so"
                        " a zero time cannot be an event",
                    )
                )
            out.append(t)
        else:
            out.append(-t)
    return out^


def check_cox_target(target: List[Float64]) raises:
    """Every entry finite, and at least one event.

    With no event the partial likelihood is empty, every gradient is 0 and
    every tree is a single zero leaf. CatBoost trains that fit and returns a
    model that is exactly the base score; refusing it is one pass and turns a
    silent no-op into a message.
    """
    var events = 0
    for r in range(len(target)):
        if not isfinite(target[r]):
            raise Error(String("Cox target at row ", r, " is not finite"))
        if target[r] > 0.0:
            events += 1
    if events == 0:
        raise Error(
            "Cox target has no events: every row is censored (y <= 0), so the"
            " partial likelihood is empty and every gradient is zero"
        )


# ---------------------------------------------------------------------------
# Cox derivatives
# ---------------------------------------------------------------------------


def cox_varies_hessian() -> Bool:
    """Whether a Cox fit has a per-row hessian, so `histogram.CONSTANT_HESSIAN`
    must not be declared for it. Unconditionally `True`.

    The twin of `sampling.mvs_varies_hessian`, and stronger than it. Cox's
    hessian is `h_k = e^{p_k} r_k - e^{2 p_k} s_k`, where `r_k` and `s_k`
    accumulate over every event whose risk set contains `k`. It is not the
    row weight, it is not 1, it is not even a function of row `k` alone: it
    depends on the current raw scores of every row that shares a risk set with
    `k`, and it changes every round. There is no configuration of a Cox fit
    under which it is constant, which is why this takes no argument.

    It is non-negative -- `h_k = e^{p_k} * sum_i (S_i - e^{p_k}) / S_i^2` and
    `e^{p_k} <= S_i` whenever `k` is in the risk set `R_i` -- so the Newton
    step is well posed. It reaches exactly 0 for a censored row that is in no
    event's risk set, which is a real leaf-denominator case rather than a
    numerical accident, and is what `reg_lambda` is for.
    """
    return True


def check_cox_hessian_declaration(const_hessian: Bool) raises:
    """Refuse a constant-hessian declaration beside a Cox fit.

    Exactly `sampling.check_mvs_hessian_declaration`'s shape and for exactly
    its reason: `boosting.round_has_constant_hessian` is the predicate that
    makes the declaration, its signature is a GPU-visible contract no CPU lane
    may widen, and it cannot see the objective's coupling. So the guard lives
    beside the objective, costs one branch per fit, and converts a quietly
    wrong hessian plane into an exception at fit setup.
    """
    if cox_varies_hessian() and const_hessian:
        raise Error(
            "a Cox fit must not declare a constant hessian: the hessian is"
            " per-row, depends on every row sharing a risk set, and changes"
            " every round"
        )


def check_cox_sample_weight(weights: List[Float64]) raises:
    """Cox takes no sample weights, and this refuses rather than drops them.

    `TCoxError::CalcDersRange` and `TCoxError::CalcFirstDerRange` both name
    the parameter `const float* /*weights*/` and never read it, so CatBoost
    silently ignores a weighted Cox fit. A correctly weighted partial
    likelihood needs the weights *inside* the risk-set sums `S_i`, which is a
    different objective from the one transcribed here. Accepting the argument
    and discarding it is the one behavior that cannot be right.
    """
    if len(weights) > 0:
        raise Error(
            "Cox does not take sample_weight: a weighted partial likelihood"
            " needs the weights inside the risk-set sums, which CatBoost"
            " never does (it silently ignores them)"
        )


def cox_grad_hess_into[
    NARROW: Bool
](
    raw: List[Float64],
    target: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    """One serial pass of `TCoxError::CalcDersRange`, negated into loss sign.

    CatBoost, verbatim (`error_functions.cpp:153-204`):

        accumulatedSum += lastExpP;
        if (y > 0) { expPSum -= accumulatedSum; accumulatedSum = 0;
                     rk += 1/expPSum; sk += 1/(expPSum*expPSum); }
        grad = (y > 0) - expP * rk;
        hess = expP * rk - expP * expP * sk;
        ders[ind].Der1 = grad;  ders[ind].Der2 = -hess;

    so `Der1 = -dL/da` gives our `grad = expP*rk - [y>0]` and `Der2 = -h`
    gives our `hess = expP*rk - expP*expP*sk`, both stored at the row's
    original index rather than at its sorted position.

    `expPSum` starts as the sum over every row and is decremented by a
    one-position-lagged accumulator at each event, so at sorted position `k`
    it is the sum over positions `k..n-1`. That is the risk set, and it is a
    suffix of the sort order rather than a time comparison -- see the module
    docstring and catalog A17 for why that matters on ties.

    Every approx is shifted by the maximum before exponentiating, exactly as
    CatBoost does here (it does *not* in the metric); the shift cancels out of
    `e^{p_k} / S_i` and out of `e^{2 p_k} / S_i^2`, so it costs one pass and
    changes nothing but the exponent range.

    The hessian is clamped at 0 from below. It is provably non-negative (see
    `cox_varies_hessian`) and the clamp only catches the cancellation in
    `expP*rk - expP*expP*sk` when the two terms are equal to within rounding,
    which is the last row's case. Class (3), bit-moving, and only where the
    exact answer is 0.
    """
    var n = len(target)
    if len(raw) < n:
        raise Error("raw scores must be at least as long as the target")
    if len(grad) != n:
        grad.resize(n, 0.0)
    if len(hess) != n:
        hess.resize(n, 0.0)
    if n == 0:
        return

    var order = cox_sort_order(target)

    var max_approx = raw[0]
    for r in range(1, n):
        if raw[r] > max_approx:
            max_approx = raw[r]

    var exp_p_sum = 0.0
    for r in range(n):
        exp_p_sum += exp(raw[r] - max_approx)

    var rk = 0.0
    var sk = 0.0
    var last_exp_p = 0.0
    var accumulated = 0.0
    for i in range(n):
        var ind = order[i]
        var p = raw[ind] - max_approx
        var exp_p = exp(p)
        var y = target[ind]
        accumulated += last_exp_p

        if y > 0.0:
            exp_p_sum -= accumulated
            accumulated = 0.0
            var s = exp_p_sum
            if not (s > COX_RISK_SUM_FLOOR):
                s = COX_RISK_SUM_FLOOR
            rk += 1.0 / s
            sk += 1.0 / (s * s)

        var g = exp_p * rk
        if y > 0.0:
            g -= 1.0
        var h = exp_p * rk - exp_p * exp_p * sk
        if not (h > 0.0):
            h = 0.0
        grad[ind] = derivative[NARROW](g)
        hess[ind] = derivative[NARROW](h)
        last_exp_p = exp_p


def cox_grad_hess(
    raw: List[Float64],
    target: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
    float64_derivatives: Bool = False,
) raises:
    """`cox_grad_hess_into` with `derivative_precision` resolved.

    The same rule and the same precedence `boosting.fill_grad_hess` uses --
    the parameter entry and the environment entry both honored, `float64`
    winning from either -- so a Cox fit narrows its derivatives to single
    precision by default exactly as every built-in objective does, and the
    histogram's gathered and un-gathered accumulate paths cannot disagree
    about what they are adding.
    """
    check_derivative_precision()
    if derivative_precision_narrows() and not float64_derivatives:
        cox_grad_hess_into[True](raw, target, grad, hess)
    else:
        cox_grad_hess_into[False](raw, target, grad, hess)


def cox_partial_log_likelihood(
    raw: List[Float64], target: List[Float64]
) raises -> Float64:
    """CatBoost's `Cox` eval metric: the partial log-likelihood. **Higher is
    better**, best value 0 (`TCoxMetric::GetBestValue` returns `Max`).

    `sum over events i of ( p_i - log S_i )` with `S_i` the same suffix risk
    sum the derivative uses, so the metric and the objective agree about ties
    by construction rather than by inspection.

    Not additive over rows -- CatBoost derives it from
    `TNonAdditiveSingleTargetMetric` for that reason -- so there is no
    per-block partial that could be summed in a worker-dependent order.

    One deliberate divergence: CatBoost's metric does **not** subtract the
    maximum approx before exponentiating where its derivative code does, so
    it overflows to infinity above `|approx| ~ 709`. We subtract it in both.
    `(p_i - m) - log(sum_j e^{p_j - m})` is `p_i - log(sum_j e^{p_j})`
    exactly, so this is analytically the same number: class (3), bit-moving.
    """
    var n = len(target)
    if len(raw) < n:
        raise Error("raw scores must be at least as long as the target")
    if n == 0:
        return 0.0

    var order = cox_sort_order(target)
    var max_approx = raw[0]
    for r in range(1, n):
        if raw[r] > max_approx:
            max_approx = raw[r]
    var exp_p_sum = 0.0
    for r in range(n):
        exp_p_sum += exp(raw[r] - max_approx)

    var total = 0.0
    var last_exp_p = 0.0
    var accumulated = 0.0
    for i in range(n):
        var ind = order[i]
        var p = raw[ind] - max_approx
        var exp_p = exp(p)
        accumulated += last_exp_p
        if target[ind] > 0.0:
            exp_p_sum -= accumulated
            accumulated = 0.0
            var s = exp_p_sum
            if not (s > COX_RISK_SUM_FLOOR):
                s = COX_RISK_SUM_FLOOR
            total += p - log(s)
        last_exp_p = exp_p
    return total


# ---------------------------------------------------------------------------
# SurvivalAft: the three distributions
# ---------------------------------------------------------------------------


@fieldwise_init
struct SurvivalAftParams(Copyable, Movable):
    """`SurvivalAft`'s two loss parameters, `dist` and `scale`.

    `BuildError` accepts exactly these two and no others, defaults `dist` to
    `Normal` and `scale` to 1, and `TSurvivalAftError`'s constructor enforces
    `CB_ENSURE(Scale > 0)`.
    """

    var dist: Int
    var scale: Float64

    @staticmethod
    def catboost_default() -> SurvivalAftParams:
        """`dist=Normal, scale=1`, CatBoost's defaults."""
        return SurvivalAftParams(AFT_DIST_NORMAL, 1.0)

    def validate(self) raises:
        if (
            self.dist != AFT_DIST_NORMAL
            and self.dist != AFT_DIST_LOGISTIC
            and self.dist != AFT_DIST_EXTREME
        ):
            raise Error(
                "SurvivalAft dist must be normal, logistic, or extreme"
            )
        if not (self.scale > 0.0):
            raise Error("SurvivalAft scale must be positive")


def _lower_ascii(value: String) -> String:
    """`value` with A-Z folded to a-z. A local copy of `params._lower_ascii`
    so this module depends on nothing that another lane is editing; the same
    three-line fold, and the same reason (docs/PARAMETER_NAMING.md makes value
    strings case insensitive and CatBoost writes `Normal`, not `normal`)."""
    var out = String("")
    for ch in value.codepoints():
        var c = ch.to_u32()
        if c >= 65 and c <= 90:
            out += String(chr(Int(c) + 32))
        else:
            out += String(ch)
    return out^


def aft_dist_from_name(name: String) raises -> Int:
    """CatBoost's `dist` value strings, case-insensitively."""
    var lowered = _lower_ascii(name)
    if lowered == "normal":
        return AFT_DIST_NORMAL
    if lowered == "logistic":
        return AFT_DIST_LOGISTIC
    if lowered == "extreme":
        return AFT_DIST_EXTREME
    raise Error(
        String(
            "unknown SurvivalAft dist '",
            name,
            "'; CatBoost has normal, logistic, extreme",
        )
    )


comptime _INV_SQRT_2PI = 0.398942280401432677939946
"""`TNormalDistribution::CalcPdf`'s constant, transcribed digit for digit."""


@always_inline
def aft_pdf(dist: Int, x: Float64) -> Float64:
    """`IDistribution::CalcPdf`, `distribution_helpers.cpp`.

    CatBoost calls `fast_exp`; we call `std.math.exp`. **Both are
    approximations**, which is a measured fact rather than an assumption:
    `exp(log(5.0))` on this toolchain is 4.999999998698298, about 2.6e-10
    relative, and `aft_cdf(NORMAL, 1.0)` is 0.841344750494095 against libm's
    0.8413447460685429, about 5.3e-9 relative. So this is class (3),
    bit-moving against CatBoost, and it is **not** a claim to be the more
    accurate of the two -- neither side is exact and the two were not
    compared. Every tolerance in tests/test_survival_multitarget.mojo is
    sized from those two numbers.
    """
    if dist == AFT_DIST_NORMAL:
        return exp(-(x * x) / 2.0) * _INV_SQRT_2PI
    var e = exp(x)
    if dist == AFT_DIST_EXTREME:
        # `!IsFinite(expX) ? 0.0 : expX * fast_exp(-expX)`
        if not isfinite(e):
            return 0.0
        return e * exp(-e)
    # Logistic: `!IsFinite(expX) || !IsFinite(Sqr(expX)) ? 0 : expX / Sqr(1+expX)`
    if not isfinite(e) or not isfinite(e * e):
        return 0.0
    var d = 1.0 + e
    return e / (d * d)


@always_inline
def aft_pdf_der1(dist: Int, pdf: Float64, x: Float64) -> Float64:
    """`IDistribution::CalcPdfDer1`. Takes the pdf as CatBoost's does, so the
    two never disagree about which `x` they were evaluated at."""
    if dist == AFT_DIST_NORMAL:
        return -x * pdf
    var e = exp(x)
    if dist == AFT_DIST_EXTREME:
        if not isfinite(e):
            return 0.0
        return (1.0 - e) * pdf
    if not isfinite(e):
        return 0.0
    return pdf * (1.0 - e) / (1.0 + e)


@always_inline
def aft_pdf_der2(dist: Int, pdf: Float64, x: Float64) -> Float64:
    """`IDistribution::CalcPdfDer2`."""
    if dist == AFT_DIST_NORMAL:
        return (x * x - 1.0) * pdf
    var e = exp(x)
    if dist == AFT_DIST_EXTREME:
        if not isfinite(e) or not isfinite(e * e):
            return 0.0
        return (e * e - 3.0 * e + 1.0) * pdf
    if not isfinite(e) or not isfinite(e * e):
        return 0.0
    var d = 1.0 + e
    return pdf * (e * e - 4.0 * e + 1.0) / (d * d)


@always_inline
def aft_cdf(dist: Int, x: Float64) -> Float64:
    """`IDistribution::CalcCdf`.

    Normal is `0.5 + 0.5 * erf(x / sqrt(2))`; Extreme is `1 - exp(-e^x)` with
    no finiteness guard in CatBoost either (an infinite `e^x` gives `exp(-inf)
    = 0` and a cdf of 1, which is right); Logistic is `e^x / (1 + e^x)`,
    guarded to 1.
    """
    if dist == AFT_DIST_NORMAL:
        return 0.5 + 0.5 * erf(x / sqrt(2.0))
    var e = exp(x)
    if dist == AFT_DIST_EXTREME:
        return 1.0 - exp(-e)
    if not isfinite(e):
        return 1.0
    return e / (1.0 + e)


@always_inline
def aft_inverse_monotone_transform(
    approx: Float64, target: Float64, scale: Float64
) -> Float64:
    """`NCB::InverseMonotoneTransform`: `(log(target) - approx) / scale`.

    CatBoost calls `FastLogf`; we call `std.math.log`, which is itself an
    approximation at roughly 1e-10 relative on this toolchain (see
    `aft_pdf`). Class (3), bit-moving, and no accuracy claim either way.
    """
    return (log(target) - approx) / scale


def aft_derivative_limit(
    dist: Int, order: Int, censored: Int, scale: Float64
) raises -> Tuple[Float64, Float64]:
    """`NCB::DispatchDerivativeLimits`, transcribed verbatim from
    `survival_aft_utils.cpp`, returned as `(min, max)`.

    **Read the pair order, do not normalize it.** The caller uses it as
    `targetSign ? min : max`, so which slot holds the larger number is the
    whole meaning, and two entries in CatBoost's table look inverted and are
    reproduced as written rather than tidied:

    - `Normal` / `Second` / `RightCensored` returns `(1/scale^2, 1e-16)`
    - `Extreme` / `Second` / `Uncensored|Interval|RightCensored` returns
      `(15, 1e-16)`

    Both put the larger value in the `min` slot. Guessing at which was
    intended would change answers on the fallback path, so they stand.

    This table is the **second** fallback and is reached only when the
    derivative's denominator is below `1e-12` *and* the quotient came out NaN
    or infinite. It is not on the ordinary path.
    """
    if dist == AFT_DIST_NORMAL:
        if order == AFT_DER_FIRST:
            if censored == AFT_RIGHT_CENSORED:
                return (AFT_MIN_FIRST_DER, 0.0)
            if censored == AFT_LEFT_CENSORED:
                return (0.0, AFT_MAX_FIRST_DER)
            return (AFT_MIN_FIRST_DER, AFT_MAX_FIRST_DER)
        var inv = 1.0 / (scale * scale)
        if censored == AFT_RIGHT_CENSORED:
            return (inv, AFT_MIN_SECOND_DER)
        if censored == AFT_LEFT_CENSORED:
            return (AFT_MIN_SECOND_DER, inv)
        return (inv, inv)

    if dist == AFT_DIST_EXTREME:
        if order == AFT_DER_FIRST:
            if censored == AFT_RIGHT_CENSORED:
                return (-15.0, 0.0)
            if censored == AFT_LEFT_CENSORED:
                return (0.0, 1.0 / scale)
            return (-15.0, 1.0 / scale)
        if censored == AFT_LEFT_CENSORED:
            return (AFT_MIN_SECOND_DER, AFT_MIN_SECOND_DER)
        return (15.0, AFT_MIN_SECOND_DER)

    if dist == AFT_DIST_LOGISTIC:
        if order == AFT_DER_FIRST:
            if censored == AFT_RIGHT_CENSORED:
                return (-1.0 / scale, 0.0)
            if censored == AFT_LEFT_CENSORED:
                return (0.0, 1.0 / scale)
            return (-1.0 / scale, 1.0 / scale)
        return (AFT_MIN_SECOND_DER, AFT_MIN_SECOND_DER)

    raise Error("SurvivalAft dist must be normal, logistic, or extreme")


@always_inline
def aft_clip_derivative(d: Float64, lo: Float64, hi: Float64) -> Float64:
    """`NCB::ClipDerivatives`, `Max(Min(der, max), min)`.

    Written with the comparison order of Yandex's `Min`/`Max` (the
    `std::min`/`std::max` form, `b < a ? b : a` and `a < b ? b : a`), under
    which a NaN passes through both untouched. That is CatBoost's behavior and
    it means CatBoost can store a NaN derivative; `survival_aft_grad_hess`
    checks for it afterwards and raises, which is this package's house rule
    (`objective.check_custom_grad_hess`) and is the one place we deliberately
    stop where CatBoost continues.
    """
    var m = hi if hi < d else d
    return lo if m < lo else m


# ---------------------------------------------------------------------------
# SurvivalAft derivatives
# ---------------------------------------------------------------------------


def survival_aft_varies_hessian() -> Bool:
    """Whether a `SurvivalAft` fit has a per-row hessian. Unconditionally
    `True`.

    The hessian is `clip((P^2 - D Q) / (scale D)^2, 1e-16, 15)` for a censored
    row and `clip(-(f f'' - f'^2) / (scale f)^2, 1e-16, 15)` for an exact one,
    both functions of the row's own current raw score and its own bounds.
    There is no configuration under which it is the constant 1, which is why
    this takes no argument. The `1e-16` floor makes it strictly positive for
    every row and every distribution, so a leaf denominator can never be zero
    even before `reg_lambda`.
    """
    return True


def check_survival_aft_hessian_declaration(const_hessian: Bool) raises:
    """Refuse a constant-hessian declaration beside a `SurvivalAft` fit.
    `check_cox_hessian_declaration`'s twin; same reason, same shape."""
    if survival_aft_varies_hessian() and const_hessian:
        raise Error(
            "a SurvivalAft fit must not declare a constant hessian: the"
            " hessian is per-row, distribution-dependent, and clipped into"
            " [1e-16, 15]"
        )


def survival_aft_row_censoring(lower: Float64, upper: Float64) -> Int:
    """Which of `ECensoredType`'s four cases a row is, in CatBoost's own
    branch order.

    The order is load-bearing: `target[0] == target[1]` is tested **first**,
    so a row of `(-1, -1)` is an "exact event at time -1" rather than a
    doubly-unbounded one. `TargetMatrix.check_survival_aft` refuses that row
    so the case cannot arise here, and the order is preserved anyway so that
    reading this beside `TSurvivalAftError::CalcDers` is a line-for-line
    comparison.
    """
    if lower == upper:
        return AFT_UNCENSORED
    if upper == AFT_UNBOUNDED:
        return AFT_RIGHT_CENSORED
    if lower == AFT_UNBOUNDED:
        return AFT_LEFT_CENSORED
    return AFT_INTERVAL_CENSORED


def survival_aft_row_grad_hess(
    approx: Float64, lower: Float64, upper: Float64, params: SurvivalAftParams
) raises -> Tuple[Float64, Float64]:
    """One row of `TSurvivalAftError::CalcDers`, negated into loss sign.

    With `z = (log t - a) / scale`, `f` the pdf and `F` the cdf:

        uncensored:  g = f'(z) / (scale f(z))
                     h = -(f(z) f''(z) - f'(z)^2) / (scale f(z))^2
        censored:    P = f(z_U) - f(z_L)
                     D = F(z_U) - F(z_L)
                     Q = f'(z_U) - f'(z_L)
                     g = P / (scale D)
                     h = (P^2 - D Q) / (scale D)^2

    with `f = f' = F = 0` at an unbounded lower end and `f = f' = 0, F = 1` at
    an unbounded upper end. CatBoost writes `-g` and `-h` into `Der1`/`Der2`
    (see the module docstring); the two minus signs it applies are the two we
    do not.

    Both the `< AFT_EPSILON` denominator fallback and the final clip are
    CatBoost's, in CatBoost's order: fallback first, clip second, and the
    fallback value is itself clipped.

    Returns `(grad, hess)` unweighted; the caller multiplies by the row
    weight.
    """
    var scale = params.scale
    var dist = params.dist
    var censored = survival_aft_row_censoring(lower, upper)

    var z_lower = 0.0
    var z_upper = 0.0
    var num1: Float64
    var den1: Float64
    var num2: Float64
    var den2: Float64

    if censored == AFT_UNCENSORED:
        z_lower = aft_inverse_monotone_transform(approx, lower, scale)
        var pdf = aft_pdf(dist, z_lower)
        var d1 = aft_pdf_der1(dist, pdf, z_lower)
        var d2 = aft_pdf_der2(dist, pdf, z_lower)
        num1 = d1
        den1 = scale * pdf
        num2 = -(pdf * d2 - d1 * d1)
        den2 = (scale * pdf) * (scale * pdf)
    else:
        var pdf_upper = 0.0
        var cdf_upper = 1.0
        var d1_upper = 0.0
        var pdf_lower = 0.0
        var cdf_lower = 0.0
        var d1_lower = 0.0
        if upper != AFT_UNBOUNDED:
            z_upper = aft_inverse_monotone_transform(approx, upper, scale)
            pdf_upper = aft_pdf(dist, z_upper)
            cdf_upper = aft_cdf(dist, z_upper)
            d1_upper = aft_pdf_der1(dist, pdf_upper, z_upper)
        if lower != AFT_UNBOUNDED:
            z_lower = aft_inverse_monotone_transform(approx, lower, scale)
            pdf_lower = aft_pdf(dist, z_lower)
            cdf_lower = aft_cdf(dist, z_lower)
            d1_lower = aft_pdf_der1(dist, pdf_lower, z_lower)
        var p_diff = pdf_upper - pdf_lower
        var d1_diff = d1_upper - d1_lower
        var cdf_diff = cdf_upper - cdf_lower
        num1 = p_diff
        den1 = scale * cdf_diff
        num2 = -cdf_diff * d1_diff + p_diff * p_diff
        den2 = (scale * cdf_diff) * (scale * cdf_diff)

    # `targetSign`, exactly CatBoost's: the lower transform alone when the row
    # is exact, otherwise "either transform is positive", with an unbounded
    # end leaving its transform at the initial 0.
    var target_sign: Bool
    if censored == AFT_UNCENSORED:
        target_sign = z_lower > 0.0
    else:
        target_sign = z_lower > 0.0 or z_upper > 0.0

    var g = num1 / den1
    if den1 < AFT_EPSILON and (isnan(g) or not isfinite(g)):
        var limits = aft_derivative_limit(dist, AFT_DER_FIRST, censored, scale)
        g = limits[0] if target_sign else limits[1]
    g = aft_clip_derivative(g, AFT_MIN_FIRST_DER, AFT_MAX_FIRST_DER)

    var h = num2 / den2
    if den2 < AFT_EPSILON and (isnan(h) or not isfinite(h)):
        var limits = aft_derivative_limit(dist, AFT_DER_SECOND, censored, scale)
        h = limits[0] if target_sign else limits[1]
    h = aft_clip_derivative(h, AFT_MIN_SECOND_DER, AFT_MAX_SECOND_DER)

    return (g, h)


def survival_aft_grad_hess_into[
    NARROW: Bool
](
    raw: List[Float64],
    targets: TargetMatrix,
    params: SurvivalAftParams,
    weights: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
) raises:
    """Per-row `SurvivalAft` derivatives for a whole round.

    Rows are independent here -- unlike Cox -- so this loop could be split
    across contiguous row blocks with no cross-block reads and no summation,
    which is what makes it bit-identical at every `MOJOTREES_NUM_WORKERS`.
    It is written serially because the parallel harness lives in
    `boosting._fill_grad_hess_into`, which is not this lane's file; the
    arithmetic here is already in the shape that harness wants.

    **Sample weights are applied**, which is a deliberate divergence:
    `TSurvivalAftError::CalcDers` takes `float /*weight*/` and drops it. A
    weight is a clean multiplier on both derivatives of this loss (unlike
    Cox, where it would have to enter the risk-set sums), so applying it is
    correct and silently discarding it is not.
    """
    params.validate()
    targets.check_survival_aft()
    var n = targets.n_rows
    if len(raw) < n:
        raise Error("raw scores must be at least as long as the target")
    if len(weights) > 0 and len(weights) < n:
        raise Error("sample_weight must be at least as long as the target")
    if len(grad) != n:
        grad.resize(n, 0.0)
    if len(hess) != n:
        hess.resize(n, 0.0)

    var weighted = len(weights) > 0
    for r in range(n):
        var gh = survival_aft_row_grad_hess(
            raw[r], targets.values[2 * r], targets.values[2 * r + 1], params
        )
        var g = gh[0]
        var h = gh[1]
        if not isfinite(g) or not isfinite(h):
            raise Error(
                String(
                    "SurvivalAft produced a non-finite derivative at row ",
                    r,
                    "; CatBoost's ClipDerivatives lets a NaN through and this"
                    " does not",
                )
            )
        var w = weights[r] if weighted else 1.0
        grad[r] = derivative[NARROW](w * g)
        hess[r] = derivative[NARROW](w * h)


def survival_aft_grad_hess(
    raw: List[Float64],
    targets: TargetMatrix,
    params: SurvivalAftParams,
    weights: List[Float64],
    mut grad: List[Float64],
    mut hess: List[Float64],
    float64_derivatives: Bool = False,
) raises:
    """`survival_aft_grad_hess_into` with `derivative_precision` resolved, on
    `boosting.fill_grad_hess`'s rule and precedence."""
    check_derivative_precision()
    if derivative_precision_narrows() and not float64_derivatives:
        survival_aft_grad_hess_into[True](
            raw, targets, params, weights, grad, hess
        )
    else:
        survival_aft_grad_hess_into[False](
            raw, targets, params, weights, grad, hess
        )


def survival_aft_interval_distance(
    raw: List[Float64], targets: TargetMatrix, weights: List[Float64] = []
) raises -> Float64:
    """CatBoost's `SurvivalAft` eval metric. **Lower is better**, best 0.

    It is **not** a log-likelihood, and anyone quoting "SurvivalAft" as a
    number is quoting this. `TSurvivalAftMetric::EvalSingleThread`:

        realApprox = exp(approx)
        realTarget(d) = target[d] == -1 ? +infinity : target[d]
        if realApprox <= lower or realApprox >= upper:
            error += min(|realApprox - lower|, |realApprox - upper|) * weight
        total += weight
        result = error / total

    Mapping `-1` to `+infinity` in **both** columns is what makes the four
    censoring cases fall out of one expression: a right-censored row has
    `upper = +inf`, so only `realApprox <= lower` can fire and the `min`
    picks the finite term; a left-censored row has `lower = +inf`, so the
    first test is vacuously true and the `min` again picks the finite term.

    Rows are independent and the accumulation is a plain sum in row order.
    """
    targets.check_survival_aft()
    var n = targets.n_rows
    if len(raw) < n:
        raise Error("raw scores must be at least as long as the target")
    if len(weights) > 0 and len(weights) < n:
        raise Error("weight must be at least as long as the target")

    var weighted = len(weights) > 0
    var total_error = 0.0
    var total_weight = 0.0
    for r in range(n):
        var a = exp(raw[r])
        var lo = targets.values[2 * r]
        var hi = targets.values[2 * r + 1]
        var w = weights[r] if weighted else 1.0
        var lo_inf = lo == AFT_UNBOUNDED
        var hi_inf = hi == AFT_UNBOUNDED
        # `realApprox <= +infinity` is always true, so an unbounded end makes
        # its own test fire; that is CatBoost's behavior, not an oversight.
        var fires = lo_inf or a <= lo
        if not fires:
            fires = (not hi_inf) and a >= hi
        if fires:
            var d_lo = 1.0e308 if lo_inf else abs(a - lo)
            var d_hi = 1.0e308 if hi_inf else abs(a - hi)
            var d = d_lo if d_lo < d_hi else d_hi
            total_error += d * w
        total_weight += w
    if total_weight == 0.0:
        return 0.0
    return total_error / total_weight


def survival_aft_predicted_time(raw: Float64) -> Float64:
    """`exp(raw)`: the survival time a `SurvivalAft` raw score means.

    `SurvivalAft`'s link is `LINK_EXP` and the registry does not know that
    yet, so `Booster.response` returns the raw score for a model this module
    trained. Use this until the `objective_registry.mojo` glue in the lane
    report lands, after which `Booster.response` becomes the same number.
    """
    return exp(raw)


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------


def train_cox(
    data: BinnedMatrix,
    target: List[Float64],
    params: BoosterParams,
    sample_weight: List[Float64] = [],
) raises -> Booster:
    """Train a Cox proportional-hazards ensemble.

    `target` is CatBoost's one signed column; `cox_signed_target` builds it
    from a time and an event flag.

    **Base score 0, and that is not laziness.** Adding a constant `c` to every
    raw score leaves the partial likelihood exactly unchanged --
    `(p_i + c) - log(sum_j e^{p_j + c})` is `p_i - log(sum_j e^{p_j})` -- so
    Cox has no meaningful starting score and CatBoost gives it none either.

    Row sampling is not offered. Bagging or GOSS would change the risk set
    itself, not merely which rows contribute to a histogram, so a bagged Cox
    round optimizes a different likelihood from the one the metric reports.
    CatBoost says the same thing in its own vocabulary by listing `Cox` in
    `IsPlainOnlyModeLoss`. The place to reconsider that is a design note, not
    a defaulted argument.
    """
    check_cox_target(target)
    check_cox_sample_weight(sample_weight)
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    params.tree.monotone.check_features(data.n_features)

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(0.0)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for i in range(params.n_estimators):
        cox_grad_hess(raw, target, grad, hess)
        var tree = grow_tree(data, grad, hess, params.tree, [], i)
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break
        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        trees.append(tree^)

    return Booster(
        trees^, 0.0, params.learning_rate, COX, params.tree.monotone.copy()
    )


def train_survival_aft(
    data: BinnedMatrix,
    targets: TargetMatrix,
    params: BoosterParams,
    aft: SurvivalAftParams = SurvivalAftParams(AFT_DIST_NORMAL, 1.0),
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
) raises -> Booster:
    """Train an accelerated-failure-time ensemble on a two-column target.

    The model's raw score is `log` of the survival time;
    `survival_aft_predicted_time` is the inverse link until the registry glue
    lands.

    `base_score` defaults to 0, meaning a predicted time of 1. CatBoost
    computes no average start for `SurvivalAft` either (it is not in any of
    the boost-from-average families), and the log of a geometric mean of the
    finite bounds would be a better start -- that is a real improvement and it
    is deliberately not smuggled in as a default, because it would be a
    starting point CatBoost does not use.
    """
    aft.validate()
    targets.check_survival_aft()
    targets.check_rows(data.n_rows)
    if len(sample_weight) > 0 and len(sample_weight) != data.n_rows:
        raise Error("sample_weight length must equal n_rows")
    if not isfinite(base_score):
        raise Error("base_score must be finite")
    params.tree.monotone.check_features(data.n_features)

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    for i in range(params.n_estimators):
        survival_aft_grad_hess(raw, targets, aft, sample_weight, grad, hess)
        var tree = grow_tree(data, grad, hess, params.tree, [], i)
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break
        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        trees.append(tree^)

    return Booster(
        trees^,
        base_score,
        params.learning_rate,
        SURVIVAL_AFT,
        params.tree.monotone.copy(),
    )
