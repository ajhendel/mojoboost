"""Stochastic Gradient Langevin Boosting and model shrinkage: CatBoost's
`langevin` / `diffusion_temperature` and `model_shrink_rate` /
`model_shrink_mode`.

Both are off by default and neither changes any existing default. A
default-constructed `LangevinParams` takes no draw and a default-constructed
`ModelShrinkParams` records no event, so an untouched fit is the fit it was.

Verified against CatBoost `master` (August 2026); see
`docs/design/CATBOOST_CATALOG.md`, the A13/A14 note, for the full citation
list and for every place this file deliberately diverges. The short form:

- `catboost/private/libs/algo_helpers/langevin_utils.cpp` --
  `CalcLangevinNoiseRate` is `sqrt(2.0 / learningRate / diffusionTemperature)`,
  and the three places noise is added.
- `catboost/private/libs/algo/greedy_tensor_search.cpp::DoBootstrap` -- the
  per-row site, after the bootstrap, on `WeightedDerivatives` alone.
- `catboost/private/libs/algo/train.cpp::TrainOneIteration` -- the shrink, at
  the top of every iteration after the first, on the accumulated approxes.
- `catboost/libs/train_lib/train_model.cpp` -- the end-of-fit fold of the
  shrink history into the leaf values, and the two refusals.
- `catboost/private/libs/options/catboost_options.cpp` --
  `SetNotSpecifiedOptionsToDefaults`, which is where the two features are
  coupled: turning `langevin` on installs a nonzero `model_shrink_rate`.

Why one module for two features
-------------------------------
Because CatBoost couples them and a reader who finds only one of them will
draw a wrong conclusion about what a Langevin fit is. `langevin=True` in
CatBoost is not "add noise": it is "add noise AND start decaying the model at
`0.001 * learning_rate` per round", and the rate it installs depends on
`model_shrink_mode`. `couple_langevin_defaults` is that step, spelled out and
opt-in rather than silent.

Determinism
-----------
The noise is drawn from a counter-based splitmix64 stream keyed by
`(seed, tree index, row, output)` on this module's own domain constant, the
same shape `sampling.refresh_mvs_bootstrap` uses. Row `r`, output `k` reads
exactly two counters and nothing advances, so a row's noise is independent of
the worker count, of the row count, of how many draws any earlier row took,
and of whether an earlier row was skipped. It reproduces across
`MOJOTREES_NUM_WORKERS` and across machines.

CatBoost's own is weaker: `AddLangevinNoiseToDerivatives` cuts rows into
blocks of `CB_THREAD_LIMIT` (`restrictions.h`, `constexpr int
CB_THREAD_LIMIT = 128` -- a constant, not a thread count, which is the only
reason theirs does not move with `thread_count`) and runs a sequential
`TFastRng64(seed + blockIdx)` inside each block. Sequential means row `i`'s
noise depends on how many uniforms rows `i-1`, `i-2`, ... consumed, and their
polar Box-Muller rejects, so that count is itself random. Ours cannot have
that property and does not.
"""

from std.math import cos, isfinite, log, sqrt

from .rng import GOLDEN, splitmix64, uniform
from .tree import Tree


# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

# Separates the per-row Langevin stream from every other per-tree stream in
# the package. Bagging, feature sampling, GOSS, MVS and the Bayesian bootstrap
# are all keyed by (seed, tree index) and all default their seed to a small
# integer, so a caller who sets one seed and turns two of them on would draw
# the same uniforms twice without a domain separator. `sampling._MVS_DOMAIN`
# is the same device for the same reason.
comptime _LANGEVIN_ROW_DOMAIN = UInt64(0x1A46E71E5C0FF5ED)

# The leaf-sum noise gets its OWN domain rather than sharing the row domain
# with a different index, because both are keyed by (seed, tree, index) and a
# leaf ordinal and a row index occupy the same small integers. Sharing would
# make leaf 7's noise the same draw as row 7's.
comptime _LANGEVIN_LEAF_DOMAIN = UInt64(0x1A46E71EBEAF0DE5)

# Two uniforms per normal draw, so a row's substream is exactly two counters
# wide and the next row's cannot be reached. See `_std_normal`.
comptime _NORMAL_STRIDE = UInt64(2)

# 2 * pi, for the trigonometric Box-Muller transform.
comptime _TWO_PI = 6.283185307179586


# ---------------------------------------------------------------------------
# Defaults, verified from CatBoost source
# ---------------------------------------------------------------------------

# `boosting_options.cpp`: `DiffusionTemperature("diffusion_temperature", 0.0f)`
# is the constructed default, and 0 means no noise at all. CatBoost then
# raises it to 1e4 in `SetNotSpecifiedOptionsToDefaults` if and only if
# `langevin` is on and the user left the temperature alone. Ours keeps the
# same two-stage shape: the disabled bundle carries 0.0 and `enable()`
# installs 1e4.
comptime DEFAULT_DIFFUSION_TEMPERATURE = 1e4

# No CatBoost equivalent: their seed is one draw from the sequential
# learn-progress generator and cannot be named. Ours is a parameter, because
# a stream that cannot be named cannot be reproduced.
comptime DEFAULT_LANGEVIN_SEED = 0

# `boosting_options.cpp`: `ModelShrinkRate("model_shrink_rate", 0.0f)`, and
# `train.cpp` gates the whole mechanism on `modelShrinkRate > 0`.
comptime DEFAULT_MODEL_SHRINK_RATE = 0.0

# `catboost_options.cpp::SetNotSpecifiedOptionsToDefaults`, the Langevin
# block: `shrinkRate = (mode == Constant) ? 0.001 : 0.01`.
comptime LANGEVIN_SHRINK_RATE_CONSTANT = 0.001
comptime LANGEVIN_SHRINK_RATE_DECREASING = 0.01

# The same block for monotone constraints, which install their own defaults by
# the same shape and are recorded here for completeness. Nothing reads them;
# see `monotone_default_model_shrink_rate`.
comptime MONOTONE_SHRINK_RATE_CONSTANT = 0.01
comptime MONOTONE_SHRINK_RATE_DECREASING = 0.2


# `enums.h`: `enum class EModelShrinkMode { Constant, Decreasing }`. Two
# members, no more, and `Constant` is the constructed default.
comptime MODEL_SHRINK_CONSTANT = 0
comptime MODEL_SHRINK_DECREASING = 1


def canonical_model_shrink_mode(value: String) raises -> Int:
    """The `model_shrink_mode` code one spelling selects.

    CatBoost's enum is `Constant` / `Decreasing` and its JSON reader is case
    sensitive; ours accepts the lowercase spelling as well because every other
    string parameter in this package does, and refuses anything else by name
    rather than falling back to a default. A silently defaulted mode is the
    difference between `1 - rate * learning_rate` and `1 - rate / iteration`,
    which are not close to each other.
    """
    if value == "Constant" or value == "constant":
        return MODEL_SHRINK_CONSTANT
    if value == "Decreasing" or value == "decreasing":
        return MODEL_SHRINK_DECREASING
    raise Error(String("unknown model_shrink_mode ", value))


def model_shrink_mode_name(mode: Int) raises -> String:
    """CatBoost's own spelling of a mode code, for messages and round trips."""
    if mode == MODEL_SHRINK_CONSTANT:
        return String("Constant")
    if mode == MODEL_SHRINK_DECREASING:
        return String("Decreasing")
    raise Error("model_shrink_mode must be Constant or Decreasing")


# ---------------------------------------------------------------------------
# The normal draw
# ---------------------------------------------------------------------------


def _langevin_row_stream(seed: Int, tree_index: Int) -> UInt64:
    """Start of the counter stream for one tree's per-row noise.

    Same shape as `sampling._mvs_stream`: mix the seed against this module's
    domain constant, spread the tree index by the golden-ratio increment, mix
    again. Sign bits are masked off so a negative seed is accepted without
    relying on signed-to-unsigned conversion.

    The result is a *start*, not a running state. Nothing advances it.
    """
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _LANGEVIN_ROW_DOMAIN)
    return splitmix64(h ^ (UInt64(tree_index & 0x7FFFFFFFFFFFFFFF) * GOLDEN))


def _langevin_leaf_stream(seed: Int, tree_index: Int) -> UInt64:
    """Start of the counter stream for one tree's leaf-sum noise. A second
    domain, so leaf `j` and row `j` of the same tree never share a draw."""
    var h = splitmix64(
        UInt64(seed & 0x7FFFFFFFFFFFFFFF) ^ _LANGEVIN_LEAF_DOMAIN
    )
    return splitmix64(h ^ (UInt64(tree_index & 0x7FFFFFFFFFFFFFFF) * GOLDEN))


def _std_normal(counter: UInt64) -> Float64:
    """One standard normal from exactly two counter values, `counter` and
    `counter + 1`, by the trigonometric Box-Muller transform.

    **Diverges from CatBoost on purpose.** `util/random/normal.h`'s
    `StdNormalDistribution` is the polar (Marsaglia) form: it draws pairs and
    rejects until they land in the unit disc, so it consumes an unbounded,
    draw-dependent number of uniforms. A counter-based stream cannot afford
    that. Giving each row a fixed stride and capping the retries would change
    the distribution; letting the retries run past the stride would make row
    `r`'s draws collide with row `r+1`'s, which is a correlation bug that no
    test on a single row could see. The trigonometric form is exact, has no
    rejection, and costs one `log`, one `sqrt` and one `cos`.

    The numbers therefore differ from CatBoost's for the same seed. The
    distribution does not, and bit-identity with CatBoost was never on offer:
    their stream is unnamed.

    `1.0 - u0` rather than `u0` because `rng.uniform` is half-open `[0, 1)`
    and can return exactly 0, where `log(0)` is `-inf`. `1.0 - u0` lands in
    `(0, 1]`, so the logarithm is finite and the largest magnitude this can
    return is bounded by `sqrt(-2 log(2^-53))`, about 8.6 sigma.
    """
    var u0 = uniform(counter)
    var u1 = uniform(counter + 1)
    return sqrt(-2.0 * log(1.0 - u0)) * cos(_TWO_PI * u1)


def langevin_row_noise(
    stream: UInt64, row: Int, output: Int, n_outputs: Int
) -> Float64:
    """The standard normal belonging to `(row, output)` of one tree.

    Exposed rather than hidden inside the apply loop so a test can pin one
    row's draw without running a fit, and so a device lane can reproduce the
    same value from the same three integers.

    The index is `row * n_outputs + output`, the row-major layout the
    multiclass gradient buffer already uses, times `_NORMAL_STRIDE`. Every
    `(row, output)` pair therefore owns two counters and no other pair can
    reach them.
    """
    var slot = UInt64((row * n_outputs + output) & 0x7FFFFFFFFFFFFFFF)
    return _std_normal(stream + slot * _NORMAL_STRIDE)


# ---------------------------------------------------------------------------
# LangevinParams
# ---------------------------------------------------------------------------


@fieldwise_init
struct LangevinParams(Copyable, Movable):
    """CatBoost's `langevin` and `diffusion_temperature`.

    **`diffusion_temperature` is an inverse temperature.**
    `CalcLangevinNoiseRate` is `sqrt(2.0 / learningRate / diffusionTemperature)`,
    so raising the temperature *lowers* the noise. The name says the opposite
    of what the arithmetic does. It is kept because it is CatBoost's name and
    a user porting a configuration must be able to carry the number across
    unchanged; `noise_rate` is where the arithmetic is written down.

    The learning rate is in the same denominator, so the pair sets the noise
    jointly: halving the learning rate raises the noise by `sqrt(2)`. Neither
    knob alone is "the noise knob".

    Disabled by default. `enabled` is CatBoost's `langevin` flag and gates the
    per-row noise; `diffusion_temperature` is the real switch inside every
    `AddLangevinNoise*` function, each of which begins
    `if (diffusionTemperature == 0.0f) return;`. Both are kept because
    CatBoost's own two leaf-sum call sites disagree about which one guards
    them, and a bundle that collapsed them into one flag could not express
    the configuration CatBoost actually reaches.
    """

    var enabled: Bool
    var diffusion_temperature: Float64
    var seed: Int

    @staticmethod
    def disabled() -> LangevinParams:
        """No noise: the library default, and the state a fit that never
        mentions Langevin is in."""
        return LangevinParams(False, 0.0, DEFAULT_LANGEVIN_SEED)

    @staticmethod
    def enable(
        diffusion_temperature: Float64 = DEFAULT_DIFFUSION_TEMPERATURE,
        seed: Int = DEFAULT_LANGEVIN_SEED,
    ) -> LangevinParams:
        """Langevin at CatBoost's own default temperature, `1e4`, which is
        what `SetNotSpecifiedOptionsToDefaults` installs when `langevin` is on
        and the temperature was left alone."""
        return LangevinParams(True, diffusion_temperature, seed)

    def validate(self) raises:
        """CatBoost's one rule, plus two of ours.

        Theirs: `CB_ENSURE(DiffusionTemperature >= 0.0, "Diffusion
        temperature should be non-negative")`. Written so a NaN is rejected.

        Ours: an *enabled* bundle at temperature exactly 0 is refused rather
        than silently doing nothing, because `enabled` with no noise is a
        configuration the user did not mean and CatBoost only reaches it by a
        default it installs for them. And a non-finite temperature is refused
        before it can produce a non-finite noise rate.
        """
        if not (self.diffusion_temperature >= 0.0):
            raise Error("diffusion_temperature must be nonnegative")
        if not isfinite(self.diffusion_temperature):
            raise Error("diffusion_temperature must be finite")
        if self.enabled and self.diffusion_temperature == 0.0:
            raise Error(
                "langevin is enabled with diffusion_temperature 0, which"
                " draws no noise; set a positive temperature or disable it"
            )

    def injects_noise(self) -> Bool:
        """Whether a fit configured this way perturbs any row's gradient.

        Deliberately the conjunction, and deliberately not consulted by
        `langevin_varies_hessian`: a predicate that gates a *safety*
        declaration must not reason about the value of a `Float64` knob (see
        `sampling.mvs_varies_hessian` for the argument in its stronger form).
        This one gates only whether work is done.
        """
        return self.enabled and self.diffusion_temperature > 0.0

    def noise_rate(self, learning_rate: Float64) raises -> Float64:
        """`CalcLangevinNoiseRate`, verbatim:
        `sqrt(2.0 / learningRate / diffusionTemperature)`.

        Raises on a learning rate that is zero or negative rather than
        returning an infinity or a NaN. CatBoost cannot reach that case here
        because `TBoostingOptions::Validate` already refuses
        `|learning_rate| <= epsilon`, but this function is callable on its own
        and the refusal has to live where the division does.

        A disabled bundle returns 0.0, which is the additive identity for the
        noise term, so a caller that multiplies by it unconditionally is
        correct without a branch.
        """
        if not self.injects_noise():
            return 0.0
        if not (learning_rate > 0.0):
            raise Error("learning_rate must be positive to scale langevin noise")
        return sqrt(2.0 / learning_rate / self.diffusion_temperature)

    def leaf_noise_wired(self) -> Bool:
        """Whether the leaf-derivative-sum half of CatBoost's mechanism is
        connected to a trainer in this package.

        **False, and it is a fact rather than a placeholder.** CatBoost adds
        noise twice per tree under `langevin`: once to every row's derivative
        (`DoBootstrap`) and once to every leaf's derivative sum
        (`approx_calcer.cpp`, both `CalcApproxDeltaSimple` and
        `CalcLeafValuesSimple`). `langevin_leaf_gradient_noise` and
        `langevin_leaf_newton_noise` below implement the second, and nothing
        calls them: the leaf sums live in `tree.mojo`'s grower, which this
        lane does not own. A caller reading `True` from `injects_noise` is
        getting half of CatBoost's regularizer, and this is how they find out
        without reading the source.
        """
        return False


def langevin_varies_hessian(params: LangevinParams) -> Bool:
    """Whether a fit configured this way has a per-row hessian, so that
    `histogram.CONSTANT_HESSIAN` must not be declared for it.

    **False, always, and this is a checked claim rather than an omission.**
    The twin of `sampling.mvs_varies_hessian`, and it answers the opposite
    way for a reason that is visible in CatBoost's own signatures:
    `AddLangevinNoiseToDerivatives` takes `TVector<double>* derivatives` and
    is passed `bodyTail.WeightedDerivatives` -- the first-derivative buffer
    and nothing else. The two leaf-sum variants write `sum.SumDer` while only
    *reading* `sum.SumDer2` and `sum.SumWeights`. No weight, no hessian, and
    no second derivative is modified anywhere in `langevin_utils.cpp`.

    So `hess[r]` under an active Langevin fit is exactly what the objective
    wrote, `boosting.round_has_constant_hessian` stays correct without
    knowing this bundle exists, and the two-plane histogram path is admissible
    beside Langevin where it is not admissible beside MVS or the Bayesian
    bootstrap.

    `params` is taken and ignored on purpose. The signature is the one a
    later edit would have to change to make this vary, and
    `check_langevin_hessian_declaration` is what would then fire.
    """
    _ = params.enabled
    return False


def check_langevin_hessian_declaration(
    params: LangevinParams, const_hessian: Bool
) raises:
    """Refuse a constant-hessian declaration beside a Langevin fit that
    perturbs the hessian.

    Today that combination cannot arise, because `langevin_varies_hessian` is
    False by construction. The guard exists anyway, and is called by the
    trainer glue, so that the day someone adds noise to `hess` -- the obvious
    next thing a reader of the SGLB paper reaches for -- the fit raises
    instead of silently rebuilding the hessian plane from the row count. This
    is the same machinery as
    `sampling.check_mvs_hessian_declaration`, deliberately installed while
    the predicate it guards is False, because installing it afterwards
    requires noticing.
    """
    if langevin_varies_hessian(params) and const_hessian:
        raise Error(
            "a fit with langevin noise on the hessian must not declare a"
            " constant hessian"
        )


def apply_langevin_noise(
    mut gradients: List[Float64],
    params: LangevinParams,
    learning_rate: Float64,
    tree_index: Int,
    n_rows: Int,
    n_outputs: Int = 1,
) raises:
    """Add this tree's Langevin noise to every row's gradient, in place.

    This is `AddLangevinNoiseToDerivatives`, and it belongs at the same place
    in the round: **after** the row sampler has written its weights into the
    gradients, because CatBoost calls it from `DoBootstrap` on the line after
    `Bootstrap(...)`. Two consequences of that ordering are load-bearing and
    are reproduced here rather than improved on. The noise is not multiplied
    by the row's sample weight, so a row the sampler zeroed still receives a
    full-size noise term. And the noise is added to the gradient buffer the
    histogram builder will read, so it moves the split search as well as the
    leaf values.

    Under CatBoost's default `sampling_frequency=PerTree` this is called once
    per tree. Under `PerTreeLevel` CatBoost calls `DoBootstrap` again at every
    level and the noise **accumulates within one tree**; mojotrees has no
    per-level bootstrap, so this is called once per tree and that divergence
    is recorded rather than emulated.

    A disabled bundle returns before it touches a row, so an untouched fit
    keeps every byte it had.

    `hess` is not a parameter. That is the whole content of
    `langevin_varies_hessian`.
    """
    params.validate()
    if n_rows < 0:
        raise Error("n_rows must be nonnegative")
    if tree_index < 0:
        raise Error("tree_index must be nonnegative")
    if n_outputs < 1:
        raise Error("n_outputs must be positive")
    if not params.injects_noise():
        return
    if len(gradients) < n_rows * n_outputs:
        raise Error("gradient buffer is shorter than n_rows * n_outputs")
    var coef = params.noise_rate(learning_rate)
    var stream = _langevin_row_stream(params.seed, tree_index)
    for r in range(n_rows):
        var base = r * n_outputs
        for k in range(n_outputs):
            gradients[base + k] = (
                gradients[base + k]
                + coef * langevin_row_noise(stream, r, k, n_outputs)
            )


def scaled_l2_reg(
    lambda_reg: Float64, sum_weights: Float64, n_rows: Int
) raises -> Float64:
    """`online_predictor.h::ScaleL2Reg`:
    `l2Regularizer * (sumAllWeights / allDocCount)`.

    The leaf-sum noise below is scaled by `sqrt(something + this)`, so it has
    to be the same number CatBoost uses and not our raw `lambda_reg`. On an
    unweighted fit `sum_weights == n_rows` and this is the identity, which is
    why the distinction is invisible until someone passes sample weights.
    """
    if n_rows <= 0:
        raise Error("n_rows must be positive to scale l2_leaf_reg")
    return lambda_reg * (sum_weights / Float64(n_rows))


def langevin_leaf_gradient_noise(
    params: LangevinParams,
    learning_rate: Float64,
    scaled_l2: Float64,
    tree_index: Int,
    leaf: Int,
    leaf_sum_weights: Float64,
) raises -> Float64:
    """The term `AddLangevinNoiseToLeafDerivativesSum` adds to one leaf's
    gradient sum: `coef * sqrt(sumWeights + scaledL2Reg) * N(0, 1)`.

    **NOT WIRED.** See `LangevinParams.leaf_noise_wired`. It is written here
    because the mechanism is half-described without it and because the two
    scalings are easy to transcribe wrongly from a paper.

    Returns 0.0 for a leaf with `sumWeights < 1e-9`, which is CatBoost's own
    `if (sum.SumWeights < 1e-9) continue;` -- an empty leaf is skipped rather
    than given noise it cannot divide by anything.

    The `sqrt(sumWeights)` factor is the point of the leaf-sum form: a leaf's
    gradient sum aggregates `n` rows, so the noise `n` independent per-row
    draws would have contributed grows like `sqrt(n)`, and this reproduces
    that at the leaf without visiting a row.
    """
    params.validate()
    if leaf < 0:
        raise Error("leaf must be nonnegative")
    if not params.injects_noise():
        return 0.0
    if leaf_sum_weights < 1e-9:
        return 0.0
    var coef = params.noise_rate(learning_rate)
    var stream = _langevin_leaf_stream(params.seed, tree_index)
    return (
        coef
        * sqrt(leaf_sum_weights + scaled_l2)
        * langevin_row_noise(stream, leaf, 0, 1)
    )


def langevin_leaf_newton_noise(
    params: LangevinParams,
    learning_rate: Float64,
    scaled_l2: Float64,
    tree_index: Int,
    leaf: Int,
    leaf_sum_weights: Float64,
    leaf_sum_hess: Float64,
) raises -> Float64:
    """The term `AddLangevinNoiseToLeafNewtonSum` adds to one leaf's gradient
    sum under Newton leaf estimation:
    `coef * sqrt(|sumDer2| + scaledL2Reg) * N(0, 1)`.

    **NOT WIRED.** Same reason as the gradient variant.

    Note what changes and what does not. The *guard* is still `sumWeights <
    1e-9` -- CatBoost tests the weight and scales by the hessian -- and the
    absolute value on `sumDer2` is theirs, `std::fabs(sum.SumDer2)`, which
    matters for an objective whose hessian can go negative. The perturbation
    still lands on the numerator only; the leaf denominator is untouched, so
    the noise cannot flip a leaf's sign by shrinking its divisor.

    CatBoost is inconsistent about which of the two variants applies. Their
    `CalcApproxDeltaSimple` selects by leaf estimation method, and their
    `CalcLeafValuesSimple` calls the *gradient* variant under Newton too. A
    wiring lane has to choose; the Newton variant is the one that matches the
    estimation method mojotrees actually uses.

    The draw is taken from the same `(seed, tree, leaf)` counter as the
    gradient variant, so a fit cannot get two different leaf noises for the
    same leaf by changing estimation method -- only two different scalings of
    the same standard normal. That is deliberate: it makes the two variants
    comparable on one fit.
    """
    params.validate()
    if leaf < 0:
        raise Error("leaf must be nonnegative")
    if not params.injects_noise():
        return 0.0
    if leaf_sum_weights < 1e-9:
        return 0.0
    var coef = params.noise_rate(learning_rate)
    var stream = _langevin_leaf_stream(params.seed, tree_index)
    return (
        coef
        * sqrt(abs(leaf_sum_hess) + scaled_l2)
        * langevin_row_noise(stream, leaf, 0, 1)
    )


# ---------------------------------------------------------------------------
# Model shrinkage
# ---------------------------------------------------------------------------


@fieldwise_init
struct ModelShrinkParams(Copyable, Movable):
    """CatBoost's `model_shrink_rate` and `model_shrink_mode`.

    What it does, from `train.cpp::TrainOneIteration`: at the top of every
    iteration after the first, and before this iteration's derivatives are
    computed, every accumulated raw score is multiplied by

        Constant:   1 - model_shrink_rate * learning_rate
        Decreasing: 1 - model_shrink_rate / iteration_index

    The first iteration is exempt and records a factor of exactly 1.0.

    **`Decreasing` divides by the iteration index, not the iteration count.**
    So the factor is `1 - rate` on iteration 1 and climbs toward 1 as the fit
    goes on: the decay is strongest at the start and fades. That is the
    opposite of what the name suggests about the multiplier, and it is why the
    two modes have different admissible ranges -- under `Constant` the product
    `rate * learning_rate` must be below 1, under `Decreasing` the rate itself
    must be, because at iteration 1 the divisor is 1.

    Off by default: `model_shrink_rate` defaults to 0 and `train.cpp` gates
    the entire mechanism on `modelShrinkRate > 0`.

    This is CatBoost's own weight decay, and it is worth naming as such. The
    accumulated model is pulled toward zero by a constant factor per round
    while the trees push it back out, which is exactly ridge regularization on
    the ensemble output rather than on any leaf. It is the second half of
    SGLB: the noise explores and the decay confines, and CatBoost turns the
    decay on the moment the noise goes on. See `couple_langevin_defaults`.
    """

    var rate: Float64
    var mode: Int

    @staticmethod
    def disabled() -> ModelShrinkParams:
        """No shrink: the library default, and CatBoost's."""
        return ModelShrinkParams(DEFAULT_MODEL_SHRINK_RATE, MODEL_SHRINK_CONSTANT)

    @staticmethod
    def constant(rate: Float64) -> ModelShrinkParams:
        """`model_shrink_mode=Constant`, the CatBoost default mode."""
        return ModelShrinkParams(rate, MODEL_SHRINK_CONSTANT)

    @staticmethod
    def decreasing(rate: Float64) -> ModelShrinkParams:
        """`model_shrink_mode=Decreasing`."""
        return ModelShrinkParams(rate, MODEL_SHRINK_DECREASING)

    def shrinks(self) -> Bool:
        """CatBoost's `if (modelShrinkRate > 0)`, the whole gate."""
        return self.rate > 0.0

    def validate(self, learning_rate: Float64) raises:
        """`TBoostingOptions::Validate`, both arms, verbatim in force.

        The admissible range depends on the mode, and the learning rate is
        part of it under `Constant`. Comparisons are written so a NaN is
        rejected. An unknown mode code is refused here rather than falling
        through to one of the two arms.
        """
        if self.mode != MODEL_SHRINK_CONSTANT and (
            self.mode != MODEL_SHRINK_DECREASING
        ):
            raise Error("model_shrink_mode must be Constant or Decreasing")
        if not (self.rate >= 0.0):
            raise Error("model_shrink_rate must be nonnegative")
        if not isfinite(self.rate):
            raise Error("model_shrink_rate must be finite")
        if self.mode == MODEL_SHRINK_CONSTANT:
            if not (learning_rate > 0.0):
                raise Error("learning_rate must be positive")
            var coef = self.rate * learning_rate
            if not (coef >= 0.0 and coef < 1.0):
                raise Error(
                    "for Constant shrink mode: (model_shrink_rate *"
                    " learning_rate) should be in [0, 1)"
                )
        else:
            if not (self.rate >= 0.0 and self.rate < 1.0):
                raise Error(
                    "for Decreasing shrink mode: model shrink rate should be"
                    " in [0, 1)"
                )

    def factor_at_round(self, learning_rate: Float64, round: Int) raises -> Float64:
        """The multiplier applied to every accumulated raw score at the start
        of boosting round `round`, counting from 0.

        Exactly `TrainOneIteration`: 1.0 at round 0 whatever the parameters
        say, `1 - rate * learning_rate` under `Constant`, and
        `1 - rate / round` under `Decreasing`. A disabled bundle returns 1.0
        at every round and is the multiplicative identity, so a caller may
        multiply unconditionally.

        `round` is the **absolute** round index of the fit, not an offset
        within a call, which is why `Decreasing` cannot be used with continued
        training here -- see `check_model_shrink_continued_training`.
        """
        self.validate(learning_rate)
        if round < 0:
            raise Error("round must be nonnegative")
        if not self.shrinks():
            return 1.0
        if round == 0:
            return 1.0
        if self.mode == MODEL_SHRINK_CONSTANT:
            return 1.0 - self.rate * learning_rate
        return 1.0 - self.rate / Float64(round)


def check_model_shrink_continued_training(
    params: ModelShrinkParams, round_offset: Int
) raises:
    """Refuse model shrinkage on a continued fit.

    **This is CatBoost's own refusal, not ours.** `train_model.cpp`:
    `CB_ENSURE(!initModel, "Usage of model_shrink_rate option in combination
    with learning continuation is unimplemented yet.")`.

    The reason is visible in `factor_at_round`: the shrink history has to be
    folded into every tree of the ensemble at the end of the fit, and a
    continued fit's earlier trees have already been folded and returned to the
    caller. Folding them twice would decay them twice; not folding them would
    leave the returned model disagreeing with the raw scores the fit is
    boosting from. Neither is defensible, so the combination is refused.
    """
    if params.shrinks() and round_offset != 0:
        raise Error(
            "model_shrink_rate is not supported with continued training"
        )


def check_model_shrink_init_score(
    params: ModelShrinkParams, init_score_len: Int
) raises:
    """Refuse model shrinkage beside an external `init_score`.

    CatBoost's second refusal, same guard block in `train_model.cpp`: "Usage
    of model_shrink_rate option in combination with baseline is unimplemented
    yet", asserted for the learn set and for every test set.

    Ours has a sharper reason than theirs. CatBoost scales `StartingApprox`
    along with the approxes and keeps it, so it *could* have folded a baseline
    the same way. mojotrees returns a `Booster` whose base score is 0 under
    `init_score` -- the offset is training state, not model state, and the
    caller adds it back at scoring time. There is nothing in the returned
    model to fold the accumulated decay of that offset into, so a shrunk fit
    from an `init_score` would predict something the training loop never
    computed.
    """
    if params.shrinks() and init_score_len != 0:
        raise Error(
            "model_shrink_rate is not supported with init_score"
        )


def model_shrink_varies_hessian(params: ModelShrinkParams) -> Bool:
    """Whether model shrinkage makes any row's hessian non-constant.

    **False.** It scales the accumulated raw scores, which does move the
    gradients and, for an objective whose hessian depends on the score, the
    hessians -- but it moves them to the values the objective itself produces
    at those scores. `fill_grad_hess` still writes `histogram.CONSTANT_HESSIAN`
    into every entry for the objectives that have a constant hessian, because
    the constant does not depend on the score at all. The declaration is safe.

    Recorded as a function rather than as a sentence in a comment for the same
    reason `langevin_varies_hessian` is: it is the thing an edit would have to
    change, and `check_model_shrink_hessian_declaration` is what fires if it
    ever does.
    """
    _ = params.rate
    return False


def check_model_shrink_hessian_declaration(
    params: ModelShrinkParams, const_hessian: Bool
) raises:
    """The guard for `model_shrink_varies_hessian`, installed while the
    predicate is False. See `check_langevin_hessian_declaration`."""
    if model_shrink_varies_hessian(params) and const_hessian:
        raise Error(
            "a fit with model shrinkage that varies the hessian must not"
            " declare a constant hessian"
        )


@fieldwise_init
struct ModelShrinkPlan(Copyable, Movable):
    """The shrink factors a fit applied, and how many trees existed when each
    was applied, so they can be folded into the leaf values at the end.

    **Why events rather than CatBoost's one-entry-per-round history.**
    CatBoost's `ModelShrinkHistory` is indexed by iteration and its
    `LeafValues` is indexed by iteration too, because CatBoost appends a tree
    every iteration without exception. The mojotrees round loop does not: the
    degenerate single-leaf guard in `boosting._boost_rounds` `continue`s
    without appending under any active row sampler. A history indexed by round
    would then be one entry longer than the tree list and the fold would scale
    the wrong trees. Recording `(factor, trees grown so far)` is exact under
    both behaviors and costs one extra `Int` per round.

    **Why deferred rather than applied.** Scaling every already-grown tree at
    every round is `O(rounds^2 * leaves)` work for a result the reverse scan
    in `fold_into_trees` reaches in one pass. Strictly less work, and exactly
    the same numbers as CatBoost's own end-of-fit fold, which is also
    deferred.

    A plan that recorded nothing folds nothing and returns a base factor of
    exactly 1.0, so the disabled path allocates one empty list and moves no
    bit.
    """

    var factors: List[Float64]
    var trees_at_factor: List[Int]

    @staticmethod
    def empty() -> ModelShrinkPlan:
        return ModelShrinkPlan(List[Float64](), List[Int]())

    def record(mut self, factor: Float64, n_trees: Int) raises:
        """Note that `factor` was applied to the accumulated scores while
        `n_trees` trees existed.

        A factor of exactly 1.0 is dropped rather than stored: it is the
        identity, CatBoost stores it only to keep its history the same length
        as its tree list, and ours is not length-coupled. That makes the
        round-0 entry free and keeps a disabled fit's plan literally empty.

        `n_trees` must not go backwards; the events are consumed in ascending
        order by `fold_into_trees` and a caller that recorded them out of
        order would get a silently wrong fold.
        """
        if n_trees < 0:
            raise Error("n_trees must be nonnegative")
        if len(self.trees_at_factor) > 0:
            if n_trees < self.trees_at_factor[len(self.trees_at_factor) - 1]:
                raise Error(
                    "model shrink events must be recorded in ascending tree"
                    " order"
                )
        if factor == 1.0:
            return
        self.factors.append(factor)
        self.trees_at_factor.append(n_trees)

    def is_empty(self) -> Bool:
        return len(self.factors) == 0

    def total(self) -> Float64:
        """The product of every recorded factor: what the base score, which
        existed before round 0, has to be multiplied by.

        Accumulated in reverse, the same order `fold_into_trees` walks, so the
        base factor and the tree factors come from one associativity and
        cannot disagree by a rounding step.
        """
        var acc = 1.0
        for i in range(len(self.factors) - 1, -1, -1):
            acc = acc * self.factors[i]
        return acc

    def factor_for_tree(self, tree_index: Int) -> Float64:
        """The product of every factor recorded *after* tree `tree_index` was
        appended: `prod_{j > t} History[j]` in CatBoost's terms.

        One-off form of the reverse scan, for tests and for a caller that
        wants a single tree's factor without folding the ensemble.
        """
        var acc = 1.0
        for i in range(len(self.factors) - 1, -1, -1):
            if self.trees_at_factor[i] > tree_index:
                acc = acc * self.factors[i]
        return acc

    def fold_into_trees(self, mut trees: List[Tree]) raises -> Float64:
        """Scale every tree's values by the decay applied after it was grown,
        and return the factor the base score still needs.

        `train_model.cpp`, verbatim in shape:

            accumulatedTreeShrinkage = 1.0
            for treeIndex = treeCount - 1 down to 0:
                LeafValues[treeIndex] *= accumulatedTreeShrinkage
                accumulatedTreeShrinkage *= ModelShrinkHistory[treeIndex]

        so the last tree is not scaled at all, and the returned base factor is
        the product of everything, which is right because the base score was
        present before round 0 and every shrink hit it.

        The whole `value` array is scaled, not only the leaf entries. Internal
        nodes carry the value a prediction stopping there would take, which
        `contrib.mojo` reads for exact TreeSHAP; scaling the leaves alone
        would leave contributions on a different scale from the predictions
        they are supposed to decompose.

        `split_gain` is deliberately not scaled. A gain is a diagnostic of the
        split search that ran at the unshrunk scale, and rescaling it would
        claim a search happened that did not.

        One pass over the events and one pass over the trees. The events are
        consumed by a cursor rather than rescanned per tree, so this is
        `O(trees + events)` comparisons plus one multiply per node.
        """
        var n_trees = len(trees)
        var acc = 1.0
        var ei = len(self.factors) - 1
        for t in range(n_trees - 1, -1, -1):
            while ei >= 0 and self.trees_at_factor[ei] > t:
                acc = acc * self.factors[ei]
                ei -= 1
            if acc != 1.0:
                for j in range(len(trees[t].value)):
                    trees[t].value[j] = trees[t].value[j] * acc
        # Whatever is left was recorded before any tree existed, and applies
        # to the base score alone.
        while ei >= 0:
            acc = acc * self.factors[ei]
            ei -= 1
        return acc


def apply_model_shrinkage(mut raw: List[Float64], factor: Float64):
    """Multiply every accumulated raw score by `factor`, in place.

    `train.cpp::ScaleAllApproxes`, minus the exp-approx branch: CatBoost
    stores some approxes exponentiated and raises them to the power instead
    (`ApplyLearningRate<true>`), where mojotrees keeps raw scores everywhere
    and a multiply is the whole operation.

    A factor of exactly 1.0 returns without writing, so round 0 and every
    round of a disabled fit cost one comparison.
    """
    if factor == 1.0:
        return
    for r in range(len(raw)):
        raw[r] = raw[r] * factor


# ---------------------------------------------------------------------------
# The coupling
# ---------------------------------------------------------------------------


def langevin_default_model_shrink_rate(mode: Int) raises -> Float64:
    """The `model_shrink_rate` CatBoost installs when `langevin` is on and the
    user left the rate alone.

    `catboost_options.cpp::SetNotSpecifiedOptionsToDefaults`:

        if (Langevin) {
            if (shrinkRate.NotSet()) {
                shrinkRate = (ModelShrinkMode == Constant) ? 0.001 : 0.01;
            }
        }
    """
    if mode == MODEL_SHRINK_CONSTANT:
        return LANGEVIN_SHRINK_RATE_CONSTANT
    if mode == MODEL_SHRINK_DECREASING:
        return LANGEVIN_SHRINK_RATE_DECREASING
    raise Error("model_shrink_mode must be Constant or Decreasing")


def monotone_default_model_shrink_rate(mode: Int) raises -> Float64:
    """The rate CatBoost installs for a fit with monotone constraints and no
    explicit rate: `0.01` for Constant, `0.2` for Decreasing, from the block
    immediately below the Langevin one.

    Nothing in mojotrees reads this. It is recorded because the two blocks are
    adjacent and identical in shape, and because a reader who finds only the
    Langevin one will conclude that Langevin is the only thing that turns
    shrinkage on. CatBoost's ordering matters too: the Langevin block runs
    first, so a fit with both Langevin and monotone constraints gets
    Langevin's smaller rate.

    mojotrees does **not** install this for a monotone fit and will not
    without a decision, because `monotone.mojo`'s constraints are a
    LightGBM-compatible feature here and LightGBM has no model shrinkage.
    Turning one library's regularizer on because another library's option was
    set is exactly the surprise this module refuses elsewhere.
    """
    if mode == MODEL_SHRINK_CONSTANT:
        return MONOTONE_SHRINK_RATE_CONSTANT
    if mode == MODEL_SHRINK_DECREASING:
        return MONOTONE_SHRINK_RATE_DECREASING
    raise Error("model_shrink_mode must be Constant or Decreasing")


def couple_langevin_defaults(
    langevin: LangevinParams, shrink: ModelShrinkParams
) raises -> ModelShrinkParams:
    """CatBoost's coupling, spelled out and **opt-in**.

    Returns the `model_shrink_rate` bundle CatBoost would have installed for a
    fit with this Langevin configuration and this shrink configuration: if
    Langevin injects noise and the shrink rate is still at its default of 0,
    the rate becomes `0.001` (Constant) or `0.01` (Decreasing). Otherwise the
    bundle is returned unchanged.

    **Nothing calls this implicitly, and that is the divergence.** CatBoost
    performs this substitution inside `SetNotSpecifiedOptionsToDefaults`, so a
    user who sets `langevin=True` and nothing else gets a second, different
    regularizer they did not ask for and cannot see in their own parameter
    dict. Two mechanisms with one switch is how a fit becomes unattributable:
    an accuracy change measured against `langevin=True` is a change from noise
    *and* decay, and neither this project nor a user can tell which half moved
    the number.

    So the function exists, it is exact, and a caller who wants CatBoost's
    estimator calls it. A caller who wants to know what the noise alone does
    does not.

    CatBoost's other two Langevin-triggered defaults are deliberately not
    here. `DiffusionTemperature.SetDefault(1e4)` is already
    `LangevinParams.enable`'s default argument, which is the same number in
    the place a mojotrees user would look for it. And
    `LeavesEstimationBacktrackingType.SetDefault(No)` has no referent:
    `boosting._estimate_leaf_values` implements no line search at all (see the
    A6 note), so there is nothing for Langevin to switch off.
    """
    langevin.validate()
    if not langevin.injects_noise():
        return shrink.copy()
    if shrink.shrinks():
        return shrink.copy()
    return ModelShrinkParams(
        langevin_default_model_shrink_rate(shrink.mode), shrink.mode
    )
