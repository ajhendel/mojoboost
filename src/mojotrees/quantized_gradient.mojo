"""Quantized gradient training, and the package's numerical policy for it.

LightGBM's `use_quantized_grad` family (`num_grad_quant_bins`,
`stochastic_rounding`, `quant_train_renew_leaf`) trains on integer gradients:
each row's gradient and hessian are rounded onto a small integer lattice
once per boosting round, histograms accumulate those integers exactly, and
the split gain is reconstructed by multiplying the integer sums back by the
lattice step. The accumulation is associative, so a histogram is
bit-identical however its partial sums are combined, and the per-row payload
shrinks from two Float64 to two narrow integers.

mojotrees already ships half of that. `histogram_gpu.mojo` quantizes
gradients onto a 2^30 fixed-point lattice, accumulates them with Int32
integer atomics, and dequantizes on download, for exactly the reason above:
Metal has no float atomic add, and integer accumulation is the only
order-independent option that is portable across CUDA, ROCm, and Metal. What
it does not have is a *choice* of lattice: the scale is always the
magnitude-sum rule, the rounding is always deterministic, and the integers
are always as wide as the accumulator. This module is the one place that
decides those three things, for the CPU and the GPU alike.

THE SCALE RULE, IN ONE PARAGRAPH
--------------------------------
The magnitude-sum scale is a **power of two**: take `2^30 / sum|g|` and round
it **down** to a power of two -- the largest one at or below it, never the
nearest one. `fixed_point_scale_pow2` is that
rule, stated once, and it is what a CPU fixed-point histogram calls or copies
verbatim -- this module imports nothing from `max.gpu.*` and nothing from any
GPU file, which is why it can be the shared definition at all. Down rather
than to nearest, because the whole overflow argument rests on the exact
scaled total staying at or below 2^30 and rounding up would raise it. A power
of two rather than an arbitrary real, because it deletes three separate
roundings: from the dequantization on download, from the Float32 product
inside the quantization kernel, and from the narrowing of the scale itself.
It costs at most one bit of lattice resolution, and that cost is stated
rather than buried. The bound, the proof, the extremes, and the rejected
alternatives are all at `fixed_point_scale_pow2`; the pre-existing arm
survives as `SCALE_SHAPE_ARBITRARY` so the change can be measured.

WHAT IS HERE, AND WHAT IS DELIBERATELY NOT
------------------------------------------
Here: the parameter bundle and its validation, the two scale rules and the
proof obligations each one carries, the deterministic and stochastic
rounding modes and the counter-based stream that makes the stochastic one
reproducible, the overflow bounds and the accumulator width they imply, the
integer histogram representation and its exact sibling subtraction, the
reconstruction of split gains and leaf outputs from integer sums, the
multiclass and GOSS and sample-weight contracts, and the decision procedure
that falls back to floating accumulation with a named reason.

Not here: gradients (`boosting.mojo`, `objective.mojo`), the float histogram
kernels (`histogram.mojo`), the device kernels (`histogram_gpu.mojo`,
`gpu_leaf_batching.mojo`), the split search (`split.mojo`), tree growth
(`tree.mojo`), and every parameter surface (`params.mojo`, the bindings, the
Python estimators). This module holds no trainer state and opens no device.

THE TWO SCALE RULES
-------------------
Both quantize with a *units-per-unit* factor, so `q = round(x * units)` and
`x ~= q / units`. That is the convention the shipped device kernel already
uses (`gq = Int32(round(grad * g_scale))` in `gpu_leaf_batching.mojo`), and
it is the reciprocal of LightGBM's `grad_scale_`, which is a step size.
Converting between the two is one division; the docstring on
`QuantScales.grad_step` states it.

- `SCALE_MAX_ABS` is LightGBM's rule. With `B = num_grad_quant_bins`,
  `units = (B / 2) / max|g|` for gradients and `units = B / max|h|` for
  hessians, and every quantized value is clamped into `[-B/2, B/2]`
  (gradients) or `[-B, B]` (hessians). The hessian gets the full width
  because hessians are nonnegative for every built-in objective and the
  custom-objective check rejects negative ones, so a one-sided range wastes
  a bit. The bound is *per row*, which is what makes a node's accumulation
  bound `n_rows * B/2` and lets a narrow accumulator be chosen from the node
  size.
- `SCALE_MAGNITUDE_SUM` is the rule `histogram_gpu.mojo` ships:
  `units = 2^30 / sum|g|`, rounded down to a power of two. The bound is on
  the *total*, not per row: any node's rows are a subset of all rows, so no
  partial sum of scaled values can exceed 2^30 plus the rounding residue.
  There is no clamp, because nothing can reach one. Rounding the factor down
  can only lower the exact scaled total (into `(2^29, 2^30]`), so it moves
  that bound in the safe direction and never the other way.

They are not interchangeable, and the difference is the whole point of
having both. `SCALE_MAX_ABS` at B = 4 is lossy by design: it is a
compression scheme with an accuracy cost the LightGBM paper measures.
`SCALE_MAGNITUDE_SUM` at 2^30 is the existing GPU path, whose loss is
already below Float32 and which exists to make an accumulation
order-independent rather than to make it small.

THE ROUNDING RESIDUE, AND WHY STOCHASTIC ROUNDING CHANGES THE BOUND
-------------------------------------------------------------------
Deterministic rounding moves a value by at most 1/2 a unit, so a sum of `n`
of them is off the exact scaled sum by at most `n/2`. Stochastic rounding
(`q = floor(x + u)`, `u` uniform in [0, 1)) is unbiased, which is why the
LightGBM paper uses it, but it moves a value by strictly less than 1 unit,
so the same sum is off by at most `n`. That doubling is not cosmetic:

    SCALE_MAGNITUDE_SUM, deterministic:  |sum| <= 2^30 + n/2
    SCALE_MAGNITUDE_SUM, stochastic:     |sum| <= 2^30 + n

`histogram_gpu.MAX_ROWS` is `Int32.MAX`, so the deterministic bound is
`2^30 + (2^31 - 1)/2 = 2^31 - 1/2`, which floors to exactly `Int32.MAX`. The
shipped Int32 accumulator holds it with zero slack. The stochastic bound is
`2^30 + 2^31 - 1`, which does not fit. So stochastic rounding on the
magnitude-sum rule requires either `n_rows <= 2^30` or a wider accumulator,
and `accumulation_bound` below is what says which. Nothing in this module
quietly narrows an accumulator that a bound does not fit; `decide` falls
back to floating accumulation and names the reason.

DETERMINISTIC SEEDS
-------------------
Stochastic rounding needs a random number per row per plane per round, and
it needs the same one on every backend and at every worker count, or the CPU
and GPU trainers stop agreeing and a rerun stops reproducing. So the draw is
counter-based, exactly as in `bagging.mojo`, `goss.mojo`, and
`sampling.mojo`:

    stream = mix64 over (seed, round, class, plane)
    u(r)   = mix64(stream + r) >> 11, scaled by 2^-53      in [0, 1)

A row's draw depends only on `(seed, round_index, class_index, plane, row)`.
It does not depend on how many rows were quantized before it, on the thread
count, on the bagged or GOSS-sampled subset, or on the backend. A GPU kernel
computing the same three integer multiplies gets the same `u`, because the
mixing is exact 64-bit integer arithmetic on every device this package
targets.

LightGBM has no public seed for this; its stochastic rounding draws from a
per-thread engine, so its rounding is not reproducible across thread counts.
The seed is an mojotrees extension, and `DEFAULT_QUANT_SEED` is its default.

SAMPLE WEIGHTS, GOSS, AND THE ORDER OF OPERATIONS
--------------------------------------------------
mojotrees folds sample weights into the derivatives before anything else
sees them (`boosting._fill_grad_hess_into` multiplies by `w`), and GOSS
multiplies the sampled small-gradient rows by its compensation multiplier
in place (`goss.apply_goss_scaling`). Both change the magnitudes the scale
is derived from, so the scale must be derived *after* them. The contract is:

    fill grad/hess  ->  GOSS scaling  ->  gradient_stats  ->  derive_scales
                    ->  quantize      ->  accumulate

`gradient_stats` takes the row subset it should measure, so a bagged or
GOSS-sampled round measures the rows the tree will actually see rather than
the whole dataset. Measuring the whole dataset would still be *correct* (a
subset's magnitudes are bounded by the full set's) but would throw away
resolution whenever the sample excludes the extremes.

A wide weight range costs resolution and there is no way around it: one row
weighted 10^6 times another sets `max|g|`, and at B = 4 every ordinary row
then quantizes to zero. `count_underflow` reports exactly that, in rows, so
a caller can see the cost instead of inferring it from a worse model.

MULTICLASS
----------
One scale pair per class, never one shared pair. That is already how the GPU
multiclass path works (`gpu_multiclass_batch.mojo`: "each class carries its
own fixed-point scale"), and it matters more here: softmax gradients for a
rare class are orders of magnitude smaller than for a common one, and a
shared scale would quantize the rare class to all zeros. The class index is
mixed into the rounding stream too, so two classes do not share a dither
sequence.

WHAT QUANTIZATION DOES NOT CHANGE
----------------------------------
Counts. A histogram's count plane is exact integers already and is untouched
by any of this, so `min_data_in_leaf`, the leaf-count-dependent path
smoothing, and every count-based guard behave identically.

Sibling subtraction. In floating point the subtraction trick is exact only
up to cancellation; in integers it is exact, full stop. `subtract_quantized`
is therefore the one histogram operation that quantization makes *more*
accurate rather than less. **The scale cannot affect this, and that is worth
stating rather than assuming**, because every other exactness claim in the
package rests on it. The exactness comes from the accumulator: Int32 addition
is associative and commutative, and a parent cell is the exact integer sum of
its two children's cells, so `parent - left == right` identically. A scale is
one fixed multiplier applied to every row of the round before any integer
reaches a bin, so parent and both children are counts in the same unit
whatever that unit is. Changing the unit from an arbitrary real to a power of
two changes which integers are counted and changes nothing about the identity
that relates them. `check_same_lattice` is what refuses a subtraction across
two different scales, and it gets easier rather than harder under the default
shape: two equal powers of two are bit-equal at every float width.

Leaf renewal for `mae`, `quantile`, and `mape`. Those objectives rewrite
every leaf value from residuals after the tree is grown
(`boosting._renew_leaf_values`), and residuals are not gradients, so that
renewal is untouched and supersedes anything quantization did to the leaf
values. `quant_train_renew_leaf` is a *different* renewal, from the
unquantized gradient sums; `leaf_renewal_mode` below is what keeps the two
from being applied twice.

STATUS
------
Disabled, and not reachable from any public entry point. `CONNECTED` is
False, `QuantGradParams.default()` is disabled, and `decide` returns
`MODE_FLOAT` with `QUANT_REASON_NOT_CONNECTED` for any request while `CONNECTED`
is False, whatever the parameters say. A caller that explicitly asked for
quantized training gets `check_supported`'s error rather than a silent
downgrade, which is the same rule `unified_memory_policy` applies to a
transfer route it cannot honor. Nothing here is exported from
`src/mojotrees/__init__.mojo`; see
`handoffs/remaining_06_quantized_gradients.md` for the ordered patch set
that connects it, and `docs/QUANTIZED_GRADIENTS.md` for the numerical
policy in prose.
"""

from std.math import floor, isfinite, round
from std.memory import bitcast

from .binning import BinnedMatrix
from .gain import leaf_score
from .histogram import Histogram, _zeroed_f64, _zeroed_int
from .parallel import dispatch_feature_ranges
from .rng import GOLDEN, splitmix64, uniform
from .tree_parameters_extra import raw_leaf_output


# ---------------------------------------------------------------------------
# Connection gate
# ---------------------------------------------------------------------------

comptime CONNECTED = False
"""Whether a trainer, a histogram builder, and a split search have actually
been wired to this module and validated against the float path.

False, and flipping it is the *last* step of the connection sequence in the
handoff, not the first. While it is False every decision procedure here
returns `MODE_FLOAT`, so no amount of parameter setting can route a training
run through an unvalidated integer accumulator. Everything else in the file
is fully exercisable with it False, because the scale, rounding, bound,
histogram, and reconstruction functions do not consult it; only `decide` and
`check_supported` do.
"""


# ---------------------------------------------------------------------------
# Codes
# ---------------------------------------------------------------------------

comptime MODE_FLOAT = 0
"""Accumulate gradients and hessians in Float64, the shipping behavior."""

comptime MODE_QUANTIZED = 1
"""Accumulate quantized integers and reconstruct on read."""

comptime SCALE_MAX_ABS = 0
"""LightGBM's rule: a per-row lattice from the largest magnitude."""

comptime SCALE_MAGNITUDE_SUM = 1
"""The rule `histogram_gpu.mojo` ships: a total-sum lattice at 2^30."""

comptime SCALE_SHAPE_ARBITRARY = 0
"""The magnitude-sum factor as it shipped: `Float32(2^30 / T)`, whatever
real number that lands on. Kept reachable so the change to
`SCALE_SHAPE_POW2` can be measured rather than asserted; it is never the
more accurate arm."""

comptime SCALE_SHAPE_POW2 = 1
"""The magnitude-sum factor rounded *down* to a power of two. The default.
`fixed_point_scale_pow2` is the rule and states the whole argument."""

comptime DEFAULT_SCALE_SHAPE = SCALE_SHAPE_POW2
"""Which shape `fixed_point_scale` returns when no caller names one.

`SCALE_SHAPE_POW2`, because it deletes three separate roundings outright and
tightens the overflow bound, at a cost of at most one bit of lattice
resolution. Deleting a rounding and coarsening a lattice are not the same
kind of thing and the net is a *trade*, not a free win; the three roundings,
the bound, and the cost are all argued at `fixed_point_scale_pow2`, which
works the trade through per bin rather than asserting a direction.
"""

comptime ROUND_NEAREST = 0
"""Round to nearest through `std.math.round`, the rule the shipped device
kernel applies. Deterministic without a seed."""

comptime ROUND_STOCHASTIC = 1
"""`floor(x + u)` with `u` from the counter stream. Unbiased, reproducible,
and one unit of residue per row instead of half a unit."""

comptime BOUND_PER_ROW = 0
"""Every quantized value is clamped, so a node's bound scales with rows."""

comptime BOUND_TOTAL = 1
"""The scale itself bounds the total, so the bound does not scale with rows
beyond the rounding residue."""

comptime PLANE_GRAD = 0
comptime PLANE_HESS = 1
"""Which of the two planes a rounding stream belongs to. Separate streams,
so a row's gradient dither and hessian dither are independent draws rather
than the same number applied twice."""

comptime WIDTH_NONE = 0
comptime WIDTH_16 = 16
comptime WIDTH_32 = 32
comptime WIDTH_64 = 64
"""Accumulator widths, in bits. `WIDTH_NONE` means no integer accumulator of
any supported width holds the bound, which is a fallback, not an error."""

comptime RENEW_NONE = 0
"""Leaf values stand as the quantized histogram produced them."""

comptime RENEW_FROM_FLOAT = 1
"""LightGBM's `quant_train_renew_leaf`: after growth, recompute every leaf
value from the unquantized gradient and hessian sums of its rows."""

comptime RENEW_BY_OBJECTIVE = 2
"""The objective already rewrites every leaf value from residuals
(`mae`, `quantile`, `mape`). That renewal supersedes both of the above and
must not be doubled up with `RENEW_FROM_FLOAT`."""

# Decision reasons. `QUANT_REASON_OK` is the only one that accompanies
# `MODE_QUANTIZED`; every other value names why the float path was chosen.
comptime QUANT_REASON_OK = 0
comptime QUANT_REASON_NOT_REQUESTED = 1
comptime QUANT_REASON_NOT_CONNECTED = 2
comptime QUANT_REASON_NO_ROWS = 3
comptime QUANT_REASON_NON_FINITE = 4
comptime QUANT_REASON_DEGENERATE = 5
comptime QUANT_REASON_OVERFLOW = 6
comptime QUANT_REASON_BACKEND = 7


def describe_mode(mode: Int) -> String:
    if mode == MODE_QUANTIZED:
        return "quantized"
    return "float"


def describe_scale_rule(rule: Int) -> String:
    if rule == SCALE_MAGNITUDE_SUM:
        return "magnitude-sum"
    return "max-abs"


def describe_rounding(mode: Int) -> String:
    if mode == ROUND_STOCHASTIC:
        return "stochastic"
    return "nearest"


def describe_scale_shape(shape: Int) -> String:
    if shape == SCALE_SHAPE_ARBITRARY:
        return "arbitrary"
    return "power-of-two"


def describe_reason(reason: Int) -> String:
    """One line per reason, phrased so it can be concatenated into a
    trainer's error or trace without further wording."""
    if reason == QUANT_REASON_OK:
        return "quantized gradient accumulation is in use"
    if reason == QUANT_REASON_NOT_REQUESTED:
        return "quantized gradient training was not requested"
    if reason == QUANT_REASON_NOT_CONNECTED:
        return (
            "quantized gradient training is not connected to a trainer in"
            " this build"
        )
    if reason == QUANT_REASON_NO_ROWS:
        return "there are no rows to quantize"
    if reason == QUANT_REASON_NON_FINITE:
        return "a gradient or hessian is not finite"
    if reason == QUANT_REASON_DEGENERATE:
        return (
            "every gradient and hessian magnitude is below the quantization"
            " floor"
        )
    if reason == QUANT_REASON_OVERFLOW:
        return (
            "no supported integer accumulator width holds this round's"
            " accumulation bound"
        )
    if reason == QUANT_REASON_BACKEND:
        return "this backend has no quantized accumulation path"
    return "unknown reason"


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

comptime FIXED_ONE = Float64(1 << 30)
"""Half the Int32 range, the total bound `SCALE_MAGNITUDE_SUM` targets.

The same value as `histogram_gpu._FIXED_ONE` and
`gpu_objectives_native.FIXED_ONE`, and this is the definition the handoff
asks those two to import rather than restate;
`gpu_objectives_native.device_fixed_scale`'s own docstring already says the
duplication is waiting on exactly that refactor.
"""

comptime MAGNITUDE_FLOOR = 1e-12
"""Magnitude sums below this are numerically zero against any
regularization. The floor keeps a scale finite instead of dividing by
(near) zero, and matches `histogram_gpu._fixed_scale` value for value."""

comptime DEFAULT_NUM_GRAD_QUANT_BINS = 4
"""LightGBM's default."""

comptime MIN_NUM_GRAD_QUANT_BINS = 2
comptime MAX_NUM_GRAD_QUANT_BINS = 1 << 20
"""The upper cap is mojotrees's, not LightGBM's. At `Int32.MAX` rows a bin
count this high still leaves the Int64 accumulation bound at 2^51, four
orders of magnitude inside the type, and no reported use of quantized
training goes past a few dozen bins. A caller that wants the full
fixed-point lattice wants `SCALE_MAGNITUDE_SUM`, not a bin count of 2^31."""

comptime DEFAULT_QUANT_SEED = 11
"""Distinct from `bagging.DEFAULT_BAGGING_SEED` (3), `GossParams.disabled()`
(3), and `sampling.DEFAULT_FEATURE_FRACTION_SEED`, so a run that leaves
every seed at its default does not dither, bag, and subsample off related
streams. The streams are independently mixed anyway; distinct defaults make
that visible rather than merely true."""

comptime INT16_LIMIT = Int64(32767)
comptime INT32_LIMIT = Int64(2147483647)
comptime INT64_LIMIT = Int64(9223372036854775807)


# ---------------------------------------------------------------------------
# Counter-based rounding streams
# ---------------------------------------------------------------------------
# The mixer is rng.mojo's splitmix64, the one every sampler draws from; this
# module owns only how (seed, round, class, plane) becomes a stream start.


def quant_uniform(counter: UInt64) -> Float64:
    """Uniform in [0, 1) with 53 significant bits, from a counter value:
    rng.mojo's `uniform`, under the name the rounding code calls it by."""
    return uniform(counter)


def quant_stream(
    seed: Int, round_index: Int, class_index: Int, plane: Int
) -> UInt64:
    """Start of the counter stream for one (round, class, plane) of
    stochastic rounding.

    Sign bits are masked off so negative seeds are accepted, as in
    `sampling._stream`. `class_index` is 0 for a single-output objective and
    the class ordinal for softmax multiclass; it is offset by one so class 0
    does not collapse onto the "no class" case of an unmixed zero. `plane`
    is `PLANE_GRAD` or `PLANE_HESS` and is folded in through the golden
    ratio rather than by XOR, so the two planes are far apart in the mixing
    input rather than one bit apart.
    """
    var h = splitmix64(UInt64(seed & 0x7FFFFFFFFFFFFFFF))
    h = splitmix64(h ^ UInt64(round_index & 0x7FFFFFFFFFFFFFFF))
    h = splitmix64(h ^ UInt64((class_index + 1) & 0x7FFFFFFFFFFFFFFF))
    return splitmix64(h ^ (UInt64(plane) * GOLDEN))


@fieldwise_init
struct QuantRoundKey(Copyable, Movable):
    """Everything the rounding streams of one boosting round depend on.

    Carried as one value so a trainer threads a single argument through the
    gradient fill, the quantization, and any device upload, rather than
    three integers that could drift apart between the CPU and GPU paths.
    """

    var seed: Int
    var round_index: Int
    var class_index: Int

    @staticmethod
    def single(seed: Int, round_index: Int) -> QuantRoundKey:
        """The key for a single-output objective's round."""
        return QuantRoundKey(seed, round_index, 0)

    def grad_stream(self) -> UInt64:
        return quant_stream(
            self.seed, self.round_index, self.class_index, PLANE_GRAD
        )

    def hess_stream(self) -> UInt64:
        return quant_stream(
            self.seed, self.round_index, self.class_index, PLANE_HESS
        )

    def for_class(self, class_index: Int) -> QuantRoundKey:
        """The same round and seed, another class."""
        return QuantRoundKey(self.seed, self.round_index, class_index)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@fieldwise_init
struct QuantGradParams(Copyable, Movable):
    """LightGBM's quantized-training parameters plus the three mojotrees
    needs to make them portable and reproducible.

    `enabled` is `use_quantized_grad`. `num_grad_quant_bins`,
    `stochastic_rounding`, and `renew_leaf` are LightGBM's
    `num_grad_quant_bins`, `stochastic_rounding`, and
    `quant_train_renew_leaf`. `seed` and `scale_rule` are mojotrees's:
    LightGBM has neither a public seed for its rounding nor a second scale
    rule, and both are needed here because the GPU path already has a scale
    rule of its own and because a rounding that is not reproducible across
    thread counts would break the CPU/GPU agreement the package tests.

    `max_width` caps the integer accumulator this round may ask for. The
    default is `WIDTH_64`, meaning "whatever the bound needs"; a caller that
    only has a 32-bit device accumulator sets `WIDTH_32` and gets a float
    fallback rather than a silent overflow whenever the bound does not fit.
    """

    var enabled: Bool
    var num_grad_quant_bins: Int
    var stochastic_rounding: Bool
    var renew_leaf: Bool
    var seed: Int
    var scale_rule: Int
    var max_width: Int

    @staticmethod
    def default() -> QuantGradParams:
        """Disabled, with LightGBM's defaults for everything it names.

        Disabled is not a placeholder: it is LightGBM's default too
        (`use_quantized_grad=false`), and it is what keeps every existing
        training run on the float path byte for byte.
        """
        return QuantGradParams(
            False,
            DEFAULT_NUM_GRAD_QUANT_BINS,
            True,
            False,
            DEFAULT_QUANT_SEED,
            SCALE_MAX_ABS,
            WIDTH_64,
        )

    @staticmethod
    def lightgbm(
        num_grad_quant_bins: Int = DEFAULT_NUM_GRAD_QUANT_BINS,
        stochastic_rounding: Bool = True,
        renew_leaf: Bool = False,
    ) -> QuantGradParams:
        """The LightGBM-parity setting: `use_quantized_grad=true` with the
        max-abs rule, which is the only rule LightGBM has."""
        return QuantGradParams(
            True,
            num_grad_quant_bins,
            stochastic_rounding,
            renew_leaf,
            DEFAULT_QUANT_SEED,
            SCALE_MAX_ABS,
            WIDTH_64,
        )

    @staticmethod
    def fixed_point() -> QuantGradParams:
        """The rule the GPU histogram already applies, expressed as
        parameters: the 2^30 magnitude-sum lattice, rounded to nearest.

        This is what `histogram_gpu.mojo` does today. Naming it here is what
        lets the device path and a quantized CPU path be described by one
        policy instead of two, and it is the setting a CPU replica of a GPU
        histogram has to use to reproduce it (see
        `histogram_cache_policy.ORIGIN_CPU_REPLICA`).
        """
        return QuantGradParams(
            True,
            DEFAULT_NUM_GRAD_QUANT_BINS,
            False,
            False,
            DEFAULT_QUANT_SEED,
            SCALE_MAGNITUDE_SUM,
            WIDTH_32,
        )

    def rounding_mode(self) -> Int:
        return ROUND_STOCHASTIC if self.stochastic_rounding else ROUND_NEAREST

    def validate(self) raises:
        """Range-check the bundle. Runs whether or not it is enabled, so a
        nonsense bin count is reported when it is set rather than when it is
        first used."""
        if (
            self.num_grad_quant_bins < MIN_NUM_GRAD_QUANT_BINS
            or self.num_grad_quant_bins > MAX_NUM_GRAD_QUANT_BINS
        ):
            raise Error(
                "num_grad_quant_bins must be between 2 and 1048576"
            )
        if self.num_grad_quant_bins % 2 != 0:
            # LightGBM computes `num_grad_quant_bins_ / 2` in integer
            # arithmetic, so an odd count silently truncates and the
            # positive and negative halves of the lattice stop matching.
            # Refusing is the difference mojotrees takes: an asymmetric
            # gradient lattice is a bug in every case anyone has reported,
            # not a configuration.
            raise Error(
                "num_grad_quant_bins must be even so the gradient lattice is"
                " symmetric about zero"
            )
        if (
            self.scale_rule != SCALE_MAX_ABS
            and self.scale_rule != SCALE_MAGNITUDE_SUM
        ):
            raise Error("unknown quantized gradient scale rule")
        if (
            self.max_width != WIDTH_16
            and self.max_width != WIDTH_32
            and self.max_width != WIDTH_64
        ):
            raise Error(
                "quantized accumulator max_width must be 16, 32, or 64"
            )

    def grad_max_unit(self) -> Int64:
        """The per-row clamp on a quantized gradient, in lattice units."""
        if self.scale_rule == SCALE_MAGNITUDE_SUM:
            # No per-row clamp exists under this rule: the scale bounds the
            # total. The total bound is reported as the clamp so a caller
            # that clamps anyway cannot change a value.
            return Int64(FIXED_ONE)
        return Int64(self.num_grad_quant_bins // 2)

    def hess_max_unit(self) -> Int64:
        """The per-row clamp on a quantized hessian.

        Twice the gradient's under the max-abs rule, because hessians are
        one-sided: every built-in objective floors them at zero (the
        logistic and softmax ones at 1e-16) and `check_custom_grad_hess`
        rejects a negative one, so the negative half of a symmetric lattice
        would never be reached. The clamp is still symmetric, so a negative
        hessian that somehow arrived would be represented rather than
        folded to zero.
        """
        if self.scale_rule == SCALE_MAGNITUDE_SUM:
            return Int64(FIXED_ONE)
        return Int64(self.num_grad_quant_bins)


# ---------------------------------------------------------------------------
# Gradient statistics
# ---------------------------------------------------------------------------

@fieldwise_init
struct GradientStats(Copyable, Movable):
    """The four magnitudes a scale can be derived from, plus the row count
    they were measured over and whether every value was finite.

    Both magnitudes are carried, not just the one the configured rule needs,
    because the rule is a caller's choice that can change between rounds
    (a fallback to floating accumulation and back), because the diagnostics
    below want both, and because measuring is one pass whichever rule wins.
    """

    var max_abs_grad: Float64
    var max_abs_hess: Float64
    var sum_abs_grad: Float64
    var sum_abs_hess: Float64
    var n_rows: Int
    var finite: Bool

    @staticmethod
    def empty() -> GradientStats:
        return GradientStats(0.0, 0.0, 0.0, 0.0, 0, True)

    def is_degenerate(self) -> Bool:
        """Whether both planes are numerically zero. Nothing to quantize:
        every lattice would put every row on zero, and the split search
        would see an empty histogram whichever path built it."""
        return (
            self.max_abs_grad < MAGNITUDE_FLOOR
            and self.max_abs_hess < MAGNITUDE_FLOOR
        )


def combine_stats(a: GradientStats, b: GradientStats) -> GradientStats:
    """Merge two partial measurements.

    Associative and commutative in every field (max of maxima, sum of sums),
    so a threaded or sharded measurement gives the same answer as a serial
    one at every partition. Two callers need that: a distributed round has
    to agree on one scale across ranks before any rank quantizes anything
    (`distributed.allreduce_histogram` cannot reconcile two lattices), and a
    multi-threaded stats pass must not make the scale depend on the thread
    count.

    The float sums are not bit-associative, so `sum_abs_*` can differ in the
    last ulp between partitions. It is the same tolerance
    `histogram_gpu._fixed_scale` already lives with, since it sums a Float64
    list in row order and the device sums in reduction order.

    What that ulp costs depends on the scale shape, and the two differ in
    kind rather than in degree. Under `SCALE_SHAPE_ARBITRARY` it moves the
    scale by a relative 2^-52, which moves a quantized value by at most one
    unit at 2^30 and by nothing at all at a small bin count: small, and
    always present. Under `SCALE_SHAPE_POW2`, the default, the scale is a
    step function of the sum and is constant across a whole binade, so the
    ulp moves it by *nothing at all* except in the vanishingly rare case
    where the two partitions' quotients straddle a power-of-two boundary --
    in which case it moves it by a factor of two. Almost always better, and
    worse in a measure-2^-52 set of inputs. `fixed_point_scale_pow2` names
    that discontinuity and points at the mitigation, which is this function's
    caller in the distributed case: agree on one total first, then derive one
    scale from it.
    """
    var max_g = a.max_abs_grad
    if b.max_abs_grad > max_g:
        max_g = b.max_abs_grad
    var max_h = a.max_abs_hess
    if b.max_abs_hess > max_h:
        max_h = b.max_abs_hess
    return GradientStats(
        max_g,
        max_h,
        a.sum_abs_grad + b.sum_abs_grad,
        a.sum_abs_hess + b.sum_abs_hess,
        a.n_rows + b.n_rows,
        a.finite and b.finite,
    )


def gradient_stats(
    grad: List[Float64], hess: List[Float64], rows: List[Int] = []
) raises -> GradientStats:
    """Measure this round's gradients and hessians.

    `rows` is the row subset the tree will be grown on: a bag
    (`bagging.refresh_bag`), a GOSS selection (`goss.goss_round`), or empty
    for every row. Measuring the subset is what keeps the lattice tight when
    a sampler excludes the extremes; measuring the full set would also be
    correct, since a subset's magnitudes are bounded by the full set's, and
    would only waste resolution.

    Call this *after* GOSS scaling, never before. `goss.apply_goss_scaling`
    multiplies the sampled small-gradient rows in place, so a scale derived
    before it would be too small by up to the GOSS multiplier and every
    sampled row would clamp.

    Non-finite values are recorded rather than raised on, because the
    decision procedure's answer to them is a float fallback with a named
    reason and not an aborted training run. The float path handles a
    non-finite gradient exactly as it does today.
    """
    if len(grad) != len(hess):
        raise Error("gradient/hessian length must match")
    var n = len(grad)
    var use_all = len(rows) == 0
    var count = n if use_all else len(rows)
    if count == 0:
        return GradientStats.empty()

    var gp = grad.unsafe_ptr()
    var hp = hess.unsafe_ptr()
    var rp = rows.unsafe_ptr()
    var max_g = 0.0
    var max_h = 0.0
    var sum_g = 0.0
    var sum_h = 0.0
    var finite = True
    for i in range(count):
        var r = i if use_all else rp.unsafe_load(i)
        if r < 0 or r >= n:
            raise Error("row index out of range")
        var g = abs(gp.unsafe_load(r))
        var h = abs(hp.unsafe_load(r))
        if not (isfinite(g) and isfinite(h)):
            finite = False
            continue
        if g > max_g:
            max_g = g
        if h > max_h:
            max_h = h
        sum_g += g
        sum_h += h
    return GradientStats(max_g, max_h, sum_g, sum_h, count, finite)


# ---------------------------------------------------------------------------
# Scale selection
# ---------------------------------------------------------------------------

def magnitude_sum(values: List[Float64]) raises -> Float64:
    """Sum of absolute values, raising on a non-finite total.

    The host-side half of `histogram_gpu._fixed_scale`, split out so the
    sum and the scale can be taken from different places: a device reduction
    (`gpu_objectives_native.magnitude_sums`), a distributed all-reduce, or a
    plain host pass all feed the same `fixed_point_scale` below.
    """
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    if not isfinite(total):
        raise Error("gradients and hessians must be finite")
    return total


def floored_magnitude(total: Float64) raises -> Float64:
    """The magnitude sum a scale is actually derived from: `total`, floored
    at `MAGNITUDE_FLOOR` and refused if it is not finite.

    Split out of the two scale rules below so both floor the same value the
    same way and neither can drift from the other. A sum below the floor is
    numerically zero against any regularization; the floor is what keeps the
    scale finite instead of dividing by (near) zero, and it is why the
    all-zero-gradient case has a defined answer rather than an infinity.
    """
    var t = total
    if not isfinite(t):
        raise Error("gradients and hessians must be finite")
    if t < MAGNITUDE_FLOOR:
        t = MAGNITUDE_FLOOR
    return t


def largest_power_of_two_at_most(x: Float64) raises -> Float64:
    """`2^floor(log2(x))` for a positive normal Float64 `x`: the largest
    power of two that does not exceed `x`.

    Exact, by construction rather than by tolerance. A positive normal
    Float64 is `1.m * 2^e` with `1 <= 1.m < 2`, so clearing the mantissa
    field leaves exactly `2^e`, and `2^e <= x < 2^(e+1)` is precisely the
    floor. No `log2`, no `floor` on a float, and therefore no way for the
    answer to come back one power off at an exact power of two, which is the
    failure mode a `floor(log2(x))` spelling has and which would round the
    scale *up* -- the unsafe direction (see `fixed_point_scale_pow2`).

    Rejected alternative: `ldexp(1.0, ilogb(x))`. Same answer where both are
    defined, one more libm dependency, and `ilogb`'s behavior on the
    subnormal and zero edges is exactly the part this function has to be
    explicit about. The bit spelling is the one already used in this
    repository (`binning.mojo`, `distributed_transport.f64_from_bits`).

    Raises on zero, on a subnormal, and on a non-finite input. Zero and
    subnormal are not oversights: a subnormal `x` is below 2^-1022, which is
    hundreds of binades below anything Float32 can hold, so every caller here
    would have had to refuse it one line later anyway.
    """
    if not isfinite(x) or x <= 0.0:
        raise Error("a power-of-two floor needs a positive finite value")
    var bits = x.to_bits().cast[DType.uint64]()
    var biased = (bits >> 52) & 0x7FF
    if biased == 0:
        raise Error(
            "a power-of-two floor needs a normal value; this one is subnormal"
        )
    return bitcast[DType.float64, 1](SIMD[DType.uint64, 1](biased << 52))


def fixed_point_scale_arbitrary(total: Float64) raises -> Float32:
    """The magnitude-sum factor as it shipped: `Float32(2^30 / T)`.

    Kept reachable, unchanged expression for expression, so
    `SCALE_SHAPE_POW2` can be A/B'd against the thing it replaced instead of
    against a reconstruction of it. It is never the more accurate arm; see
    `fixed_point_scale_pow2`.
    """
    var t = floored_magnitude(total)
    var scale = Float32(FIXED_ONE / t)
    if not isfinite(scale) or scale <= 0.0:
        raise Error(
            "gradient/hessian magnitudes are out of range for the"
            " fixed-point histogram"
        )
    return scale


def fixed_point_scale_pow2(total: Float64) raises -> Float32:
    """THE RULE. Given `total = sum|g|` (or `sum|h|`) for the round, return
    the power of two `2^k` with `2^k <= fl(2^30 / T)`, where `T` is `total`
    floored at `MAGNITUDE_FLOOR`.

    A CPU implementation calls this function. It is in this module, which
    imports nothing from `max.gpu.*` and nothing from any GPU file, precisely
    so a CPU-only build can call it; `gpu_objectives_native.device_fixed_scale`
    already forwards here, so host and device get the same `2^k` from the same
    line of code rather than from two spellings that have to be kept in step.
    Copying it verbatim is also fine, and if it is copied, copy
    `largest_power_of_two_at_most` with it: the bit spelling is load-bearing,
    not stylistic.

    WHY A POWER OF TWO: THREE ROUNDINGS DELETED
    -------------------------------------------
    The accumulated cell is an Int32 count of lattice units and every consumer
    has to get back to a Float64 gradient sum. With an arbitrary scale `s`,
    three separate roundings sit between the gradient and the reconstructed
    sum, and all three vanish when `s = 2^k`:

    1. **Dequantization.** `histogram_gpu.download` computes
       `1.0 / self.g_scale` once and multiplies every cell by it. For an
       arbitrary `s` the reciprocal is not representable, so the stored
       inverse is `fl(1/s) = (1/s)(1 + d)` with `|d| <= 2^-53`, and the
       product carries a second rounding on top. For `s = 2^k`, `2^-k` is
       exact in Float64 for every `k` this rule can produce, and a Float64
       times a power of two is exact, so `Float64(cell) * 2^-k` is the exact
       real value of the cell. Zero error, not less error.
    2. **Quantization.** Every histogram kernel computes
       `Int32(round(x * scale))` with `x` and `scale` both Float32
       (`gpu_active_rows.mojo`, `gpu_leaf_batching.mojo`). For an arbitrary
       `s` narrowed to Float32 the product `x * s` is itself rounded before
       `round` sees it, so the kernel rounds a number that is already off the
       true product by a relative 2^-24; the result can therefore differ from
       the correctly-rounded quantization of `x * s` by a whole unit. For
       `s = 2^k` the Float32 product is exact (a binary exponent shift), so
       `round` sees the true product and the quantization is the correctly
       rounded one. This is the rounding that reaches the *bits of the
       histogram*, which is why it is the one worth the most.
    3. **The scale itself.** `Float32(2^30 / T)` is the arbitrary rule's own
       narrowing, a relative 2^-24 the host and device both inherit and which
       the `QuantScales` docstring currently has to warn about ("derived
       through Float32 ... because that is the width the device kernel stores
       the scale in"). `2^k` is exactly representable at Float32, Float64,
       and every device float width, so that warning stops applying: the host
       factor and the device factor are the same number by construction, at
       any width, with nothing to keep in step.

    THE SAFETY BOUND, AND WHY THE ROUNDING GOES DOWN
    ------------------------------------------------
    The repository's overflow argument (`docs/GPU_PORTABILITY.md`,
    `test_gpu_portability.test_fixed_point_accumulation_cannot_overflow_int32`,
    `distributed_gpu.check_fixed_point_headroom`) is: the exact scaled sum of
    every row's magnitude is at most 2^30, any node holds a subset of the
    rows, deterministic rounding adds at most 1/2 per row, so no cell exceeds
    `2^30 + n/2`, and at `n = Int32.MAX` that is exactly `Int32.MAX`.

    Rounding the scale to a power of two changes the first term, and it must
    change it downward or the argument breaks. So the rule floors rather than
    rounds to nearest:

        s2 = 2^k <= fl(2^30 / T)          by construction
        sum_i |g_i| * s2 <= T * s2
                         <= T * fl(2^30 / T)
                         <= T * (2^30 / T)(1 + 2^-53)
                          = 2^30 (1 + 2^-53)

    against the shipped arm, whose Float32 narrowing rounds to *nearest* and
    so admits `2^30 (1 + 2^-24)`. In units: the shipped arm's exact term can
    stand 64 units above 2^30; this one can stand at most 2^-23 of a unit
    above it, which no integer accumulation can see. **Derived bound**, not
    measured. The bound therefore does not merely survive, it tightens by 64
    units, and every downstream statement of `2^30 + n/2` remains true with
    strictly more slack than it had. Nothing in the repository has to be
    re-derived in the loosening direction, because there is no loosening.

    Rejected alternative: round to *nearest* power of two. It is the more
    accurate arm on average (half the resolution loss, below) and it is
    unsafe, because it admits `s2 <= sqrt(2) * (2^30 / T)` and pushes the
    exact term to `2^30 * sqrt(2) ~ 1.52 * 10^9`, which leaves
    `Int32.MAX - 2^30*sqrt(2) ~ 6.3 * 10^8` for the residue and cuts the safe
    row ceiling from about 2.1 billion to about 1.3 billion. Buying half a bit
    by moving a headroom argument in the unsafe direction is the wrong trade
    at any price, and the argument is the thing this file exists to protect.

    THE PRECISION COST, STATED HONESTLY
    -----------------------------------
    `2^k` can be as little as half of `2^30 / T` and as much as all of it, so
    the exact scaled total lands in `(2^29, 2^30]` where the shipped arm put
    it at `2^30`. That is **up to one bit** of lattice resolution given up,
    and on the order of half a bit on average if `log2(2^30 / T)` is taken to
    be uniform in its fractional part (**estimated**; nothing here measured
    the distribution of gradient magnitude sums over real rounds, and it is
    not uniform in general). The lattice goes from 30 bits of headroom below
    the total to between 29 and 30.

    That cost is real and it does not cancel. Worked through at the level
    that matters, one bin of `n_b` rows, in units of value rather than
    lattice units, all four terms **derived bounds**:

        deterministic rounding, either arm:  (n_b / 2) / s
        Float32 product error, arbitrary:    <= 2^30 * 2^-24 / s = 64 / s
                                             (over the whole round, since
                                              sum|x| * s <= 2^30)
        dequantization, arbitrary:           two roundings per cell
        dequantization, power of two:        zero

    With `s2 >= s_arb / 2`, the first term at worst doubles and the second
    and third go to zero. So the bin's error bound *improves* while
    `n_b < 128` and *worsens*, by at most a factor of two, above that. A
    histogram's populated bins near the root are far above 128 rows and its
    bins near the frontier are far below, so both regimes occur in one fit
    and no single sentence covers them.

    What is unconditional: the dequantization error goes to exactly zero
    (item 1), the scale's own narrowing error goes to exactly zero (item 3),
    and the quantization stops being the rounding of an already-wrong product
    (item 2). What is conditional: the lattice step doubles at worst, which
    dominates in large bins. **This is a trade, and calling it anything else
    would be dishonest.** It is taken because exactness is composable and
    resolution at the 29th bit of a Float64-derived gradient is not: an exact
    conversion is a property the CPU and GPU can be held to and a shared
    numeric contract can be written against, and half a bit at 2^-30 is
    already far below the Float32 the device stores the gradient in.

    THE EXTREMES
    ------------
    - **All-zero gradients.** `T` floors to `MAGNITUDE_FLOOR = 1e-12`, so
      `2^30 / T ~ 1.15e21` and `k = 70`. Every value quantizes to 0, the
      histogram is all zeros, and the split search finds no gain -- the same
      outcome the shipped arm produces, reached by the same floor.
    - **A single enormous outlier.** `T ~ |g_max|`, so the outlier alone
      quantizes to between 2^29 and 2^30 and every other row rounds toward
      zero. No clamp is applied and none is needed: the bound above is on the
      total and holds whatever the distribution is. This is the case that
      pays the full one-bit cost, because it is also the case where the
      arbitrary scale was landing exactly on 2^30.
    - **Denormals.** A subnormal `T` is below the floor and never reaches the
      exponent extraction. A subnormal *quotient* means `T` above about
      4.5e291, which no finite gradient sum in a Float64 dataset reaches
      before `magnitude_sum` raises; `largest_power_of_two_at_most` refuses it
      rather than returning a power of two it cannot represent.
    - **Below Float32.** The returned Float32 is checked to be *exactly*
      `2^k` (`Float64(scale) != p` raises), which is what makes every claim
      above true of the value the kernels actually receive. It refuses at
      `k < -149`, where the shipped arm would have rounded a Float64 quotient
      *up* to the smallest Float32 subnormal and quantized with a scale
      larger than the rule asked for. That is the one input on which this arm
      raises and the shipped arm does not; it needs `T` above roughly 7.6e53
      and it is refused rather than accepted precisely because accepting it
      would round the scale up.

    WHAT THIS DOES NOT TOUCH
    ------------------------
    **Sibling subtraction stays exact, and nothing here can make it
    otherwise.** `subtract_quantized`, and the device sibling subtraction in
    `gpu_split_search.mojo`, are exact because Int32 addition is associative
    and commutative and because a parent cell is the exact integer sum of its
    two children's cells -- properties of the *accumulator*, not of the
    scale. A scale is a fixed multiplier applied identically to every row of
    the round before any integer touches a bin, so parent, left child, and
    right child are all counts in the same unit whatever that unit is, and
    `parent - left == right` holds identically. `check_same_lattice` is what
    enforces that two histograms being subtracted came from the same scale,
    and it compares the factors for equality; that comparison gets *easier*
    under this rule, since two equal powers of two are bit-equal at every
    width. This paragraph is here because sibling subtraction is the property
    every other exactness claim in the package rests on, and "a scale change
    cannot affect it" is worth stating rather than assuming.

    Counts are untouched for the same reason they always were: the count
    plane holds no scaled value.

    ONE DISCONTINUITY, NAMED
    ------------------------
    The arbitrary rule is continuous in `T`; this one is a step function. Two
    partitions that sum the same magnitudes in different orders can differ in
    the last ulp (`combine_stats` documents that tolerance), and under the
    arbitrary rule that moved the scale by an ulp. Under this rule it almost
    always moves it by *nothing at all*, because a step function is constant
    over a binade -- but in the vanishingly rare case where the two sums
    straddle a power-of-two boundary of the quotient, it moves the scale by a
    factor of two. The mitigation already exists and does not change:
    `distributed_gpu.agree_fixed_scales` all-reduces the two magnitude sums
    and derives one scale for every rank *before* any rank quantizes, which
    is requirement 1 of `docs/distributed.md` section 5. That reduction was
    belt-and-braces against an ulp; it is load-bearing against a binade, and
    this sentence is the record that its status changed.
    """
    var t = floored_magnitude(total)
    var quotient = FIXED_ONE / t
    if not isfinite(quotient) or quotient <= 0.0:
        raise Error(
            "gradient/hessian magnitudes are out of range for the"
            " fixed-point histogram"
        )
    var p = largest_power_of_two_at_most(quotient)
    var scale = Float32(p)
    # Exactness, checked rather than argued. A power of two below 2^-149 is
    # not a Float32 at all and narrowing rounds it *up* to the smallest
    # subnormal, which is the one direction the bound above forbids. The
    # equality catches that, catches an overflow to infinity, and catches a
    # flush to zero, in one comparison.
    if not isfinite(scale) or scale <= 0.0 or Float64(scale) != p:
        raise Error(
            "gradient/hessian magnitudes are out of range for the"
            " fixed-point histogram"
        )
    return scale


def fixed_point_scale_shaped(total: Float64, shape: Int) raises -> Float32:
    """`fixed_point_scale` with the arm named explicitly.

    The arm is a function argument and not an environment variable and not a
    module-level switch, for the reason `histogram_gpu.set_row_unroll` gives:
    this machine's device timings drift several-fold between time windows, so
    only two arms interleaved inside one process compare, and a module-level
    switch would also be a global, which this package does not have.
    """
    if shape == SCALE_SHAPE_ARBITRARY:
        return fixed_point_scale_arbitrary(total)
    return fixed_point_scale_pow2(total)


def fixed_point_scale(total: Float64) raises -> Float32:
    """The `SCALE_MAGNITUDE_SUM` factor for a magnitude sum, at
    `DEFAULT_SCALE_SHAPE`: every partial sum of scaled values stays within
    +/- 2^30, half the Int32 range.

    Returned as Float32 because that is the precision the device kernel
    multiplies by, so a host-side inverse matches the device quantization
    exactly. Under the default shape the width no longer matters -- the
    factor is a power of two and is the same number at every width -- but the
    signature is what six call sites already import, so it stays.

    This is `gpu_objectives_native.device_fixed_scale` and the scalar core
    of `histogram_gpu._fixed_scale`, expression for expression, with no
    dependency on `max.gpu.*`. That is the point: those two are the same
    arithmetic written twice in modules a CPU-only build cannot import, and
    their own docstrings ask for exactly this single definition.
    """
    return fixed_point_scale_shaped(total, DEFAULT_SCALE_SHAPE)


@fieldwise_init
struct QuantScales(Copyable, Movable):
    """One round's (or one class's) lattice.

    `grad_units` and `hess_units` are units per unit of value, so
    `q = round(x * units)` and `x ~= q / units`. That is the direction the
    shipped device kernel multiplies in, and the reciprocal of LightGBM's
    `grad_scale_` / `hess_scale_`, which are step sizes; `grad_step` and
    `hess_step` below convert.

    `grad_max_unit` and `hess_max_unit` are the per-row clamps. Under
    `SCALE_MAGNITUDE_SUM` they are the total bound instead and are never
    reached, which `bound_kind` records so `accumulation_bound` knows which
    argument applies.

    The units are held as Float64 but derived through Float32 under
    `SCALE_MAGNITUDE_SUM`, because that is the width the device kernel
    stores the scale in; keeping the host and device factors bit-identical
    is what makes a CPU replica of a GPU histogram reproduce it.

    Under `DEFAULT_SCALE_SHAPE` that narrowing is a no-op and the warning
    above no longer has anything to warn about: the factor is a power of two,
    which is exactly representable at Float32, at Float64, and at every
    device float width, so the host factor and the device factor are the same
    number by construction rather than by a convention two files have to
    keep. The Float32 round trip is retained anyway, because it is also the
    range check -- a power of two too small to be a Float32 is refused there
    (`fixed_point_scale_pow2`).
    """

    var grad_units: Float64
    var hess_units: Float64
    var grad_max_unit: Int64
    var hess_max_unit: Int64
    var bound_kind: Int
    var rule: Int

    @staticmethod
    def identity() -> QuantScales:
        """A lattice of one unit per unit, unclamped. Not a useful
        quantization; it exists so a test or a diagnostic can exercise the
        reconstruction arithmetic without a scale in the way."""
        return QuantScales(
            1.0, 1.0, INT64_LIMIT, INT64_LIMIT, BOUND_PER_ROW, SCALE_MAX_ABS
        )

    def grad_step(self) -> Float64:
        """LightGBM's `grad_scale_`: the value one lattice unit stands
        for."""
        return 1.0 / self.grad_units

    def hess_step(self) -> Float64:
        """LightGBM's `hess_scale_`."""
        return 1.0 / self.hess_units

    def dequant_grad(self, q: Int64) -> Float64:
        return Float64(q) / self.grad_units

    def dequant_hess(self, q: Int64) -> Float64:
        return Float64(q) / self.hess_units

    def is_usable(self) -> Bool:
        """Whether both factors are finite and positive. A degenerate round
        can produce a factor that is not, and the decision procedure turns
        that into a float fallback rather than a division by zero deep in an
        accumulation loop."""
        return (
            isfinite(self.grad_units)
            and isfinite(self.hess_units)
            and self.grad_units > 0.0
            and self.hess_units > 0.0
        )


def derive_scales(
    stats: GradientStats, params: QuantGradParams
) raises -> QuantScales:
    """This round's lattice, from the measured magnitudes and the rule.

    Call after `gradient_stats`, which must itself be called after GOSS
    scaling and on the row subset the tree will see. For multiclass, call
    once per class off that class's own stats; `derive_multiclass_scales`
    does that in one call.

    Both rules floor the magnitude they divide by at `MAGNITUDE_FLOOR`, so a
    round whose gradients have all collapsed produces a finite lattice
    rather than an infinity. Such a round is degenerate and
    `decide` sends it to the float path anyway; the floor is what keeps this
    function total so a caller can compute a lattice before deciding whether
    to use one.
    """
    params.validate()
    if params.scale_rule == SCALE_MAGNITUDE_SUM:
        # Float32 on purpose: the device stores the scale at that width and
        # multiplies by it, so a host factor of any other precision would
        # put the CPU and GPU on different lattices. Under
        # `DEFAULT_SCALE_SHAPE` the width is no longer what makes them agree
        # -- a power of two is the same number at every width -- but the call
        # is unchanged so the arbitrary arm keeps the property it needs.
        var g_units = Float64(fixed_point_scale(stats.sum_abs_grad))
        var h_units = Float64(fixed_point_scale(stats.sum_abs_hess))
        return QuantScales(
            g_units,
            h_units,
            Int64(FIXED_ONE),
            Int64(FIXED_ONE),
            BOUND_TOTAL,
            SCALE_MAGNITUDE_SUM,
        )

    var g_max = stats.max_abs_grad
    var h_max = stats.max_abs_hess
    if not (isfinite(g_max) and isfinite(h_max)):
        raise Error("gradients and hessians must be finite")
    if g_max < MAGNITUDE_FLOOR:
        g_max = MAGNITUDE_FLOOR
    if h_max < MAGNITUDE_FLOOR:
        h_max = MAGNITUDE_FLOOR
    var g_units = Float64(params.grad_max_unit()) / g_max
    var h_units = Float64(params.hess_max_unit()) / h_max
    if not (isfinite(g_units) and isfinite(h_units)):
        raise Error(
            "gradient/hessian magnitudes are out of range for a quantized"
            " lattice"
        )
    return QuantScales(
        g_units,
        h_units,
        params.grad_max_unit(),
        params.hess_max_unit(),
        BOUND_PER_ROW,
        SCALE_MAX_ABS,
    )


def derive_multiclass_scales(
    stats: List[GradientStats], params: QuantGradParams
) raises -> List[QuantScales]:
    """One lattice per class, from that class's own magnitudes.

    Never one shared lattice. Softmax gradients for a rare class are orders
    of magnitude smaller than for a common one, so a shared max-abs lattice
    would put every rare-class row on zero and the class would stop being
    fitted at all. The GPU multiclass path already carries a scale per class
    for the same reason (`gpu_multiclass_batch.mojo`), so this is the
    existing shape rather than a new one.
    """
    var out = List[QuantScales](capacity=len(stats))
    for k in range(len(stats)):
        out.append(derive_scales(stats[k], params))
    return out^


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

@always_inline
def quantize_scalar(
    value: Float64,
    units: Float64,
    max_unit: Int64,
    mode: Int,
    counter: UInt64,
) -> Int64:
    """One value onto the lattice.

    `ROUND_NEAREST` applies `std.math.round`, which is the same call
    `gpu_leaf_batching`'s kernels make (`Int32(round(grad * g_scale))`), so
    whatever tie rule that function implements, the host and the device
    apply the identical one and cannot disagree on a half-way value. The
    handoff lists confirming the tie rule as an unrun check; nothing here
    depends on which rule it is, only on both sides calling it.

    `ROUND_STOCHASTIC` applies `floor(x + u)` with `u` from
    `quant_uniform(counter)`. That is unbiased: a value 0.3 of the way to
    the next unit lands there with probability 0.3, so a sum of many
    quantized values converges on the exact scaled sum instead of on a
    systematically rounded one. It also costs twice the residue bound, which
    `accumulation_bound` accounts for.

    Clamping happens before rounding, in float, so the clamp cannot be
    escaped by a rounding that pushes a value one unit past the limit: the
    limit is an integer, and both rounding rules map it to itself.
    """
    var y = value * units
    var lim = Float64(max_unit)
    if y > lim:
        y = lim
    elif y < -lim:
        y = -lim
    if mode == ROUND_STOCHASTIC:
        return Int64(floor(y + quant_uniform(counter)))
    return Int64(round(y))


def quantize_rows(
    grad: List[Float64],
    hess: List[Float64],
    scales: QuantScales,
    params: QuantGradParams,
    key: QuantRoundKey,
    mut qgrad: List[Int64],
    mut qhess: List[Int64],
) raises:
    """Quantize a whole round, in place into caller-owned buffers.

    Both outputs are resized to `len(grad)` and every element is written, so
    a trainer holds two buffers for the run and this allocates nothing after
    the first round, exactly as `boosting.fill_grad_hess` does for the float
    pair.

    Every row of the dataset is quantized, not just the sampled ones. A
    sampled round quantizes the rows it will not use as well, which costs
    one pass and buys the property that the quantized arrays are indexed by
    row id like the float ones: no compaction, no second index space, and a
    bagged and an unbagged round produce the same integer for the same row.
    The unused rows never reach a histogram, so they cannot affect a bound
    either; `accumulation_bound` is asked about the node's row count.

    Row `r` draws from `stream + r`, so a row's dither depends on
    `(seed, round, class, plane, r)` and on nothing else. Quantizing a
    subset, quantizing in blocks across threads, or quantizing on a device
    gives the identical arrays.
    """
    if len(grad) != len(hess):
        raise Error("gradient/hessian length must match")
    if not scales.is_usable():
        raise Error("quantization scales must be finite and positive")
    var n = len(grad)
    if len(qgrad) != n:
        qgrad.resize(n, Int64(0))
    if len(qhess) != n:
        qhess.resize(n, Int64(0))

    var mode = params.rounding_mode()
    var g_stream = key.grad_stream()
    var h_stream = key.hess_stream()
    var gp = grad.unsafe_ptr()
    var hp = hess.unsafe_ptr()
    var qg = qgrad.unsafe_ptr()
    var qh = qhess.unsafe_ptr()
    var g_units = scales.grad_units
    var h_units = scales.hess_units
    var g_lim = scales.grad_max_unit
    var h_lim = scales.hess_max_unit
    for r in range(n):
        qg.unsafe_store(
            r,
            quantize_scalar(
                gp.unsafe_load(r), g_units, g_lim, mode, g_stream + UInt64(r)
            ),
        )
        qh.unsafe_store(
            r,
            quantize_scalar(
                hp.unsafe_load(r), h_units, h_lim, mode, h_stream + UInt64(r)
            ),
        )


# ---------------------------------------------------------------------------
# Overflow bounds and accumulator width
# ---------------------------------------------------------------------------

def residue_per_row(mode: Int) -> Float64:
    """How far one rounded value can sit from its exact scaled value.

    Half a unit for round-to-nearest, and a full unit for stochastic
    rounding, which never rounds away by a whole unit but can round up a
    value that was already just below an integer. The asymmetry is why the
    two modes do not share an overflow bound.
    """
    return 1.0 if mode == ROUND_STOCHASTIC else 0.5


def accumulation_bound(
    n_rows: Int, scales: QuantScales, mode: Int
) raises -> Int64:
    """The largest magnitude any bin of a node of `n_rows` rows can hold.

    Under `BOUND_PER_ROW` every value is clamped, so the bound is
    `n_rows * max(grad_max_unit, hess_max_unit)` and the rounding residue is
    already inside the clamp.

    Under `BOUND_TOTAL` the scale itself bounds the exact scaled sum at
    `FIXED_ONE`, and the rounding adds up to `residue_per_row * n_rows` on
    top. `FIXED_ONE` is the right constant for both scale shapes and is
    conservative for the default one: `SCALE_SHAPE_POW2` floors the factor,
    so the exact scaled total it produces lands in `(2^29, 2^30]` and this
    bound holds with up to a factor of two of unused slack. Using the real
    per-round total instead would tighten the answer and would make the
    accumulator width depend on the round's magnitudes, which is exactly the
    kind of data-dependent width this function refuses to have. That second
    term is the reason this function exists rather than
    returning a constant: at `Int32.MAX` rows the deterministic bound is
    `2^30 + (2^31 - 1)/2`, which floors to exactly `Int32.MAX` and fits the
    shipped Int32 accumulator with zero slack, while the stochastic bound is
    `2^30 + 2^31 - 1`, which does not fit and must promote or fall back.
    The shipped GPU path is the deterministic one, which is why it has never
    had to make this choice.

    Raises only if the bound overflows Int64 itself, which
    `MAX_NUM_GRAD_QUANT_BINS` and `histogram_gpu.MAX_ROWS` together make
    unreachable: 2^31 rows times 2^20 units is 2^51.
    """
    if n_rows < 0:
        raise Error("row count must not be negative")
    var rows = Float64(n_rows)
    var bound: Float64
    if scales.bound_kind == BOUND_TOTAL:
        bound = FIXED_ONE + rows * residue_per_row(mode)
    else:
        var per_row = scales.grad_max_unit
        if scales.hess_max_unit > per_row:
            per_row = scales.hess_max_unit
        bound = rows * Float64(per_row)
    if not isfinite(bound) or bound >= Float64(INT64_LIMIT):
        raise Error(
            "the quantized accumulation bound does not fit a 64-bit"
            " accumulator"
        )
    return Int64(floor(bound))


def width_for_bound(bound: Int64) -> Int:
    """The narrowest supported accumulator that holds `bound`, or
    `WIDTH_NONE` if none does.

    The signed limits are used because a gradient sum is signed. A hessian
    sum is not, but sharing one width across the two planes is what keeps a
    histogram one buffer with a `[grad | hess | count]` layout, which is the
    layout `histogram_gpu` already uses and the reason a whole node's
    histogram is one kernel launch and one copy rather than three.
    """
    if bound <= INT16_LIMIT:
        return WIDTH_16
    if bound <= INT32_LIMIT:
        return WIDTH_32
    if bound <= INT64_LIMIT:
        return WIDTH_64
    return WIDTH_NONE


def accumulator_width(
    n_rows: Int, scales: QuantScales, params: QuantGradParams
) raises -> Int:
    """The accumulator width this node needs, capped by `params.max_width`.

    Returns `WIDTH_NONE` when the bound needs more than the caller allowed,
    which is a fallback condition and not an error: a device with a 32-bit
    integer atomic sets `max_width = WIDTH_32` and gets the float path for
    the nodes that do not fit rather than a wrapped accumulation.

    This is per node, deliberately. LightGBM promotes its histogram bit
    width dynamically for the same reason: the root of a million-row dataset
    needs a wide accumulator and the leaves near the frontier, holding a few
    hundred rows each, do not, and the narrow ones are where the memory
    traffic actually is.
    """
    params.validate()
    var needed = width_for_bound(accumulation_bound(
        n_rows, scales, params.rounding_mode()
    ))
    if needed == WIDTH_NONE or needed > params.max_width:
        return WIDTH_NONE
    return needed


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

@fieldwise_init
struct QuantDecision(Copyable, Movable):
    """What one round will do, and why.

    `mode` is `MODE_FLOAT` or `MODE_QUANTIZED`, `reason` says why, `width`
    is the accumulator the quantized path would use (`WIDTH_NONE` on the
    float path), and `scales` is the lattice (`QuantScales.identity()` on
    the float path, so a caller never reads a lattice that was not derived).
    """

    var mode: Int
    var reason: Int
    var width: Int
    var scales: QuantScales

    @staticmethod
    def floating(reason: Int) -> QuantDecision:
        return QuantDecision(
            MODE_FLOAT, reason, WIDTH_NONE, QuantScales.identity()
        )

    def is_quantized(self) -> Bool:
        return self.mode == MODE_QUANTIZED

    def describe(self) -> String:
        return String(
            describe_mode(self.mode), ": ", describe_reason(self.reason)
        )


def decide(
    stats: GradientStats,
    params: QuantGradParams,
    max_node_rows: Int,
    backend_supported: Bool = True,
) raises -> QuantDecision:
    """Whether this round accumulates quantized integers or Float64.

    `max_node_rows` is the largest node any tree of this round can hold,
    which is the root's row count: the bound has to be checked once against
    the worst node rather than renegotiated per node, or two nodes of one
    tree could end up on different lattices and sibling subtraction would
    stop being exact.

    `backend_supported` is the caller's statement that its accumulation path
    exists. The CPU builder in this file always qualifies; a device path
    qualifies only once its kernel takes an integer gradient array, and a
    distributed round only once the all-reduce agrees on one lattice across
    ranks.

    Order matters and is deliberate: not-requested before not-connected, so
    an ordinary run that never asked reports the ordinary reason; then the
    connection gate; then the data conditions. Every fallback keeps
    training, at the numerics that ship today.
    """
    params.validate()
    if not params.enabled:
        return QuantDecision.floating(QUANT_REASON_NOT_REQUESTED)
    if not CONNECTED:
        return QuantDecision.floating(QUANT_REASON_NOT_CONNECTED)
    if not backend_supported:
        return QuantDecision.floating(QUANT_REASON_BACKEND)
    if stats.n_rows <= 0 or max_node_rows <= 0:
        return QuantDecision.floating(QUANT_REASON_NO_ROWS)
    if not stats.finite:
        return QuantDecision.floating(QUANT_REASON_NON_FINITE)
    if stats.is_degenerate():
        return QuantDecision.floating(QUANT_REASON_DEGENERATE)

    var scales = derive_scales(stats, params)
    if not scales.is_usable():
        return QuantDecision.floating(QUANT_REASON_DEGENERATE)
    var width = accumulator_width(max_node_rows, scales, params)
    if width == WIDTH_NONE:
        return QuantDecision.floating(QUANT_REASON_OVERFLOW)
    return QuantDecision(
        MODE_QUANTIZED, QUANT_REASON_OK, width, scales.copy()
    )


def check_supported(params: QuantGradParams) raises:
    """Raise if quantized training was explicitly asked for and this build
    cannot provide it.

    The rule `unified_memory_policy.resolve_from_env` already applies to a
    transfer route: an explicit request that cannot be honored is refused
    where it was made, not quietly downgraded somewhere the user will never
    look. A parameter surface calls this at parse time, so
    `use_quantized_grad=true` fails with a sentence rather than training a
    model that silently ignored it.

    `decide` deliberately does not call this. Its job is to keep a training
    run going, and a round that falls back because one class's gradients
    collapsed is not a configuration error.
    """
    params.validate()
    if params.enabled and not CONNECTED:
        raise Error(
            String(
                "use_quantized_grad is not available in this build, ",
                describe_reason(QUANT_REASON_NOT_CONNECTED),
            )
        )


# ---------------------------------------------------------------------------
# The quantized histogram
# ---------------------------------------------------------------------------

def _zeroed_i64(size: Int) -> List[Int64]:
    var g = List[Int64](capacity=size)
    g.resize(size, Int64(0))
    return g^


@fieldwise_init
struct QuantTotals(Copyable, Movable):
    """One feature's, or one node's, integer totals. The integer counterpart
    of `histogram.FeatureTotals`."""

    var grad: Int64
    var hess: Int64
    var count: Int


@fieldwise_init
struct QuantizedHistogram(Copyable, Movable):
    """Per-(feature, bin) integer statistics, flattened as
    `[f * n_bins + b]`, in the same layout as `histogram.Histogram`.

    One host representation, always Int64, and a separately declared
    `width`. The width is what a device buffer or a distributed message may
    narrow to, not what the host holds: three parallel host histogram types
    would be three accumulation loops, three subtraction kernels, and three
    dequantizers to keep in step, against a saving that only matters where
    the bytes actually move. `width` travels with the histogram so a
    transport can narrow safely and a reader can tell what it was narrowed
    to.

    `scales` is the lattice these integers were quantized on. It travels
    with the histogram because a histogram is meaningless without it, and
    because it is what makes an accidental mixing of two rounds' histograms
    detectable rather than silently wrong (`check_same_lattice`).
    """

    var grad: List[Int64]
    var hess: List[Int64]
    var count: List[Int]
    var n_features: Int
    var n_bins: Int
    var width: Int
    var scales: QuantScales

    @staticmethod
    def zeroed(
        n_features: Int,
        n_bins: Int,
        scales: QuantScales,
        width: Int = WIDTH_64,
    ) -> QuantizedHistogram:
        var size = n_features * n_bins
        return QuantizedHistogram(
            _zeroed_i64(size),
            _zeroed_i64(size),
            _zeroed_int(size),
            n_features,
            n_bins,
            width,
            scales.copy(),
        )

    def reset(mut self):
        """Zero every bin in place, keeping the allocation and the lattice.

        Serial, and the builder below does not call it: it zeroes each
        feature's slice inside that feature's task, which is what
        `histogram.mojo` does and for the same reason. It stays for a caller
        that needs a zeroed buffer without a build.
        """
        var size = self.n_features * self.n_bins
        var gp = self.grad.unsafe_ptr()
        var hp = self.hess.unsafe_ptr()
        var cp = self.count.unsafe_ptr()
        for i in range(size):
            gp.unsafe_store(i, Int64(0))
            hp.unsafe_store(i, Int64(0))
            cp.unsafe_store(i, 0)

    def matches(self, n_features: Int, n_bins: Int) -> Bool:
        return self.n_features == n_features and self.n_bins == n_bins

    def dequantize(self) raises -> Histogram:
        """The Float64 `Histogram` these integers stand for.

        The same arithmetic `histogram_gpu.histogram_from_host` performs on
        a downloaded fixed-point plane, over Int64 instead of Int32 and with
        the lattice carried by the histogram rather than held on a builder.
        Counts pass through untouched: they were always exact.

        Reconstructing the whole histogram is what a split search that has
        not been taught integers needs. A search that has been taught them
        should call `quantized_split_gain` instead and never materialize
        this, which is the entire point of quantizing.
        """
        if not self.scales.is_usable():
            raise Error("quantization scales must be finite and positive")
        var size = self.n_features * self.n_bins
        var g = _zeroed_f64(size)
        var h = _zeroed_f64(size)
        var c = _zeroed_int(size)
        var g_inv = 1.0 / self.scales.grad_units
        var h_inv = 1.0 / self.scales.hess_units
        var gp = g.unsafe_ptr()
        var hp = h.unsafe_ptr()
        var cp = c.unsafe_ptr()
        var sg = self.grad.unsafe_ptr()
        var sh = self.hess.unsafe_ptr()
        var sc = self.count.unsafe_ptr()
        for i in range(size):
            gp.unsafe_store(i, Float64(sg.unsafe_load(i)) * g_inv)
            hp.unsafe_store(i, Float64(sh.unsafe_load(i)) * h_inv)
            cp.unsafe_store(i, sc.unsafe_load(i))
        return Histogram(g^, h^, c^, self.n_features, self.n_bins)

    def totals(self, feature: Int) raises -> QuantTotals:
        """One feature's integer totals over its bins.

        Exact: this is a sum of integers, so unlike
        `histogram.feature_totals` it does not depend on the order the bins
        are visited in and cannot lose a low bit to cancellation.
        """
        if feature < 0 or feature >= self.n_features:
            raise Error("feature index out of range")
        var base = feature * self.n_bins
        var g = Int64(0)
        var h = Int64(0)
        var c = 0
        for b in range(self.n_bins):
            g += self.grad[base + b]
            h += self.hess[base + b]
            c += self.count[base + b]
        return QuantTotals(g, h, c)


def check_same_lattice(a: QuantScales, b: QuantScales) raises:
    """Refuse to combine two histograms quantized on different lattices.

    Adding, subtracting, or comparing integers from two lattices produces a
    number that means nothing, and the failure is silent: the arithmetic
    succeeds and the gains are wrong. The scale moves every round, so this
    is the check that catches a stale histogram surviving into the next
    round, which is exactly the failure
    `histogram_cache_policy` describes for its own cached planes.
    """
    if (
        a.grad_units != b.grad_units
        or a.hess_units != b.hess_units
        or a.rule != b.rule
    ):
        raise Error(
            "quantized histograms from different lattices cannot be"
            " combined"
        )


def subtract_quantized(
    parent: QuantizedHistogram, child: QuantizedHistogram
) raises -> QuantizedHistogram:
    """The sibling histogram, by subtraction.

    Exact. In Float64 the subtraction trick loses digits to cancellation
    whenever a bin's parent and child sums are close, which is most of them
    near the frontier; in integers `parent - child` is the sibling's
    accumulation, bit for bit, with no error at all. This is the one place
    quantization makes a result *more* accurate than the float path rather
    than less, and it holds at every bin count.

    Requires both operands on the same lattice, which is what
    `check_same_lattice` enforces.
    """
    if parent.n_features != child.n_features or parent.n_bins != child.n_bins:
        raise Error("histogram shapes must match")
    check_same_lattice(parent.scales, child.scales)
    var out = QuantizedHistogram.zeroed(
        parent.n_features, parent.n_bins, parent.scales, parent.width
    )
    var size = parent.n_features * parent.n_bins
    var og = out.grad.unsafe_ptr()
    var oh = out.hess.unsafe_ptr()
    var oc = out.count.unsafe_ptr()
    var pg = parent.grad.unsafe_ptr()
    var ph = parent.hess.unsafe_ptr()
    var pc = parent.count.unsafe_ptr()
    var cg = child.grad.unsafe_ptr()
    var ch = child.hess.unsafe_ptr()
    var cc = child.count.unsafe_ptr()
    for i in range(size):
        og.unsafe_store(i, pg.unsafe_load(i) - cg.unsafe_load(i))
        oh.unsafe_store(i, ph.unsafe_load(i) - ch.unsafe_load(i))
        oc.unsafe_store(i, pc.unsafe_load(i) - cc.unsafe_load(i))
    return out^


def accumulate_quantized(
    mut acc: QuantizedHistogram, src: QuantizedHistogram
) raises:
    """Add `src` into `acc`, bin by bin.

    Integer addition is associative, so a histogram reduced this way is
    bit-identical however the partials were partitioned. That is what makes
    a quantized `distributed.allreduce_histogram` reproducible across rank
    counts, which the Float64 one is not: it sums shard histograms in rank
    order and a different partition gives a different last ulp.

    The ranks must have agreed on one lattice first. `combine_stats` is the
    reduction that produces that agreement, and it has to complete before
    any rank quantizes anything.
    """
    if acc.n_features != src.n_features or acc.n_bins != src.n_bins:
        raise Error("histogram shapes must match")
    check_same_lattice(acc.scales, src.scales)
    var size = acc.n_features * acc.n_bins
    var ag = acc.grad.unsafe_ptr()
    var ah = acc.hess.unsafe_ptr()
    var ac = acc.count.unsafe_ptr()
    var sg = src.grad.unsafe_ptr()
    var sh = src.hess.unsafe_ptr()
    var sc = src.count.unsafe_ptr()
    for i in range(size):
        ag.unsafe_store(i, ag.unsafe_load(i) + sg.unsafe_load(i))
        ah.unsafe_store(i, ah.unsafe_load(i) + sh.unsafe_load(i))
        ac.unsafe_store(i, ac.unsafe_load(i) + sc.unsafe_load(i))


def build_quantized_histogram_into(
    mut out: QuantizedHistogram,
    data: BinnedMatrix,
    qgrad: List[Int64],
    qhess: List[Int64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int] = [],
) raises:
    """Accumulate the quantized histogram of a row window into a
    caller-owned buffer.

    The integer counterpart of
    `histogram.build_histogram_subset_into_scratch`, and deliberately not a
    copy of it. Feature slices are disjoint, so accumulation parallelizes
    across features with no atomics, each feature zeroes its own slice
    inside its own task, and an empty `features` means every feature: those
    three properties are the ones the float builder's correctness argument
    rests on and they are restated here because they are the contract, not
    because the code is shared.

    What is *not* restated is `apple_cpu_policy.derive_accumulation_plan`:
    the gradient-pair gather and the two-features-per-inner-loop grouping
    are tuned against Float64 pair loads and Float64 read-modify-writes, and
    an Int64 pair is a different memory shape with a different crossover.
    Guessing at it here would be a second tuning policy to keep in step with
    the first. This builder reads the quantized values through the row ids,
    which is what the float builder does on every node the plan declines to
    compact, and the handoff's histogram patch is where the two accumulators
    become one templated kernel under one plan.

    The window `rows[row_start : row_start + row_count]` is the same
    convention the float builder takes, so a grower keeps one row arena and
    hands the same window to whichever accumulator the decision picked.
    """
    var n_rows = data.n_rows
    if len(qgrad) != n_rows or len(qhess) != n_rows:
        raise Error("quantized gradient/hessian length must equal n_rows")
    if not out.matches(data.n_features, data.n_bins):
        raise Error("output histogram shape must match the data")
    if row_start < 0 or row_count < 0 or row_start + row_count > len(rows):
        raise Error("row window out of range")
    var n_features = data.n_features
    for i in range(len(features)):
        if features[i] < 0 or features[i] >= n_features:
            raise Error("feature index out of range")

    _accumulate_quantized_subset(
        out.grad, out.hess, out.count,
        data, qgrad, qhess, rows, row_start, row_count, features,
    )


def _accumulate_quantized_subset(
    mut out_grad: List[Int64],
    mut out_hess: List[Int64],
    mut out_count: List[Int],
    data: BinnedMatrix,
    qgrad: List[Int64],
    qhess: List[Int64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int],
) raises:
    """The accumulation pass, with the three output slices as their own
    parameters.

    Splitting it out is not organization: the closure below captures raw
    pointers into these lists, and a pointer taken through `out.grad` of a
    `mut out: QuantizedHistogram` carries `origin_of(out.grad)`, which a
    capture cannot rebind. `histogram._accumulate_subset` is split from
    `build_histogram_subset_into_scratch` for the same reason and is the
    shape this mirrors.
    """
    var n_rows = data.n_rows
    var n_features = data.n_features
    var n_bins = data.n_bins
    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)

    # Every feature not accumulated still has to come out zero, and its
    # slice is not visited by the pass below.
    if not use_all:
        var active = List[Bool](capacity=n_features)
        active.resize(n_features, False)
        for i in range(len(features)):
            active[features[i]] = True
        for f in range(n_features):
            if not active[f]:
                var base = f * n_bins
                for b in range(n_bins):
                    out_grad[base + b] = Int64(0)
                    out_hess[base + b] = Int64(0)
                    out_count[base + b] = 0

    var gp = out_grad.unsafe_ptr()
    var hp = out_hess.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var qg = qgrad.unsafe_ptr()
    var qh = qhess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    var n_sub = row_count

    def accumulate_range(i_start: Int, i_end: Int) {imm}:
        for i in range(i_start, i_end):
            var f = i if use_all else feat_p.unsafe_load(i)
            var base = f * n_bins
            for b in range(n_bins):
                gp.unsafe_store(base + b, Int64(0))
                hp.unsafe_store(base + b, Int64(0))
                cp.unsafe_store(base + b, 0)
            var col = bins_all_p.unsafe_offset(f * n_rows)
            for i_row in range(n_sub):
                var r = rows_p.unsafe_load(i_row)
                var cell = base + Int(col.unsafe_load(r))
                gp.unsafe_store(cell, gp.unsafe_load(cell) + qg.unsafe_load(r))
                hp.unsafe_store(cell, hp.unsafe_load(cell) + qh.unsafe_load(r))
                cp.unsafe_store(cell, cp.unsafe_load(cell) + 1)

    # One op per (feature, row) accumulate plus the zeroing of every active
    # feature's slice, which is the same accounting
    # `apple_cpu_policy.derive_accumulation_plan` uses for the float
    # builder's total.
    var total_ops = n_active * (n_sub + n_bins)
    dispatch_feature_ranges(accumulate_range, n_active, total_ops)


def build_quantized_histogram(
    data: BinnedMatrix,
    qgrad: List[Int64],
    qhess: List[Int64],
    rows: List[Int],
    scales: QuantScales,
    width: Int = WIDTH_64,
    features: List[Int] = [],
) raises -> QuantizedHistogram:
    """`build_quantized_histogram_into` over a whole row list, allocating
    the output. An empty `rows` means every row of `data`."""
    var out = QuantizedHistogram.zeroed(
        data.n_features, data.n_bins, scales, width
    )
    if len(rows) == 0:
        var all_rows = List[Int](capacity=data.n_rows)
        for r in range(data.n_rows):
            all_rows.append(r)
        build_quantized_histogram_into(
            out, data, qgrad, qhess, all_rows, 0, data.n_rows, features
        )
        return out^
    build_quantized_histogram_into(
        out, data, qgrad, qhess, rows, 0, len(rows), features
    )
    return out^


# ---------------------------------------------------------------------------
# Split-gain reconstruction
# ---------------------------------------------------------------------------

def quantized_leaf_score(
    grad_sum: Int64,
    hess_sum: Int64,
    scales: QuantScales,
    lambda_l1: Float64,
    lambda_l2: Float64,
) raises -> Float64:
    """The second-order objective improvement of a leaf holding integer
    sums.

    Dequantize, then call `gain.leaf_score`. The gain formula is not
    restated: `gain.mojo` exists precisely so the ordinal split search and
    the categorical one cannot drift, and a third copy here would defeat
    that. Reconstruction is two divisions per leaf, once per candidate, not
    once per row.

    Dequantizing before the L1 soft threshold is required, not a
    convenience: `lambda_l1` is in gradient units, so thresholding an
    integer sum against it would compare a lattice count with a real
    number and the penalty would scale with the lattice.
    """
    if not scales.is_usable():
        raise Error("quantization scales must be finite and positive")
    return leaf_score(
        scales.dequant_grad(grad_sum),
        scales.dequant_hess(hess_sum),
        lambda_l1,
        lambda_l2,
    )


def quantized_split_gain(
    left: QuantTotals,
    right: QuantTotals,
    parent: QuantTotals,
    scales: QuantScales,
    lambda_l1: Float64,
    lambda_l2: Float64,
) raises -> Float64:
    """`score(L) + score(R) - score(parent)`, from integer sums.

    The whole reason quantization pays: the scan that produced `left` and
    `right` walked integers, and only the three surviving totals per
    candidate are ever converted back. `parent` is passed rather than
    derived as `left + right` so that a search using the subtraction trick
    reports the same gain as one that accumulated both children, which is
    the property `split.mojo` already relies on for its own totals.
    """
    var g_left = quantized_leaf_score(
        left.grad, left.hess, scales, lambda_l1, lambda_l2
    )
    var g_right = quantized_leaf_score(
        right.grad, right.hess, scales, lambda_l1, lambda_l2
    )
    var g_parent = quantized_leaf_score(
        parent.grad, parent.hess, scales, lambda_l1, lambda_l2
    )
    return g_left + g_right - g_parent


def quantized_leaf_output(
    grad_sum: Int64,
    hess_sum: Int64,
    scales: QuantScales,
    lambda_l1: Float64,
    lambda_l2: Float64,
) raises -> Float64:
    """The unconstrained Newton step from integer sums.

    Delegates to `tree_parameters_extra.raw_leaf_output`, so the cap
    (`max_delta_step`), the smoothing (`path_smooth`), and the monotone
    clamp compose onto a quantized leaf exactly as they do onto a float one,
    through the functions that already implement them.
    """
    if not scales.is_usable():
        raise Error("quantization scales must be finite and positive")
    return raw_leaf_output(
        scales.dequant_grad(grad_sum),
        scales.dequant_hess(hess_sum),
        lambda_l1,
        lambda_l2,
    )


def hessian_units_at_least(
    min_child_hess: Float64, scales: QuantScales
) raises -> Int64:
    """`min_sum_hessian_in_leaf` expressed in lattice units, rounded up.

    A search working in integers compares a bin's integer hessian sum
    against this instead of dequantizing every candidate. Rounding *up* is
    what keeps the integer test at least as strict as the float one: a node
    that passes the integer test passes the float test, so quantization can
    never admit a leaf the float path would have rejected. It can reject one
    the float path would have admitted, by less than one lattice unit, and
    that is the direction to err in for a minimum-support guard.
    """
    if not scales.is_usable():
        raise Error("quantization scales must be finite and positive")
    if min_child_hess <= 0.0:
        return Int64(0)
    var units = min_child_hess * scales.hess_units
    if not isfinite(units):
        raise Error("min_sum_hessian_in_leaf is out of range for this lattice")
    var floored = floor(units)
    if floored < units:
        floored += 1.0
    return Int64(floored)


# ---------------------------------------------------------------------------
# Leaf renewal
# ---------------------------------------------------------------------------

def leaf_renewal_mode(
    params: QuantGradParams, objective_renews_leaves: Bool
) -> Int:
    """Which leaf renewal a quantized round performs.

    Three cases, and the middle one is the whole reason this is a function
    rather than a boolean:

    - The objective already renews (`mae`, `quantile`, `mape`, through
      `boosting._renew_leaf_values`). That renewal recomputes every leaf
      from *residuals*, which are not gradients and were never quantized, so
      it supersedes anything quantization did to the leaf values.
      `RENEW_BY_OBJECTIVE`, and `quant_train_renew_leaf` must not also run:
      it would compute a Newton step from gradient sums and then have it
      immediately overwritten, which is one pointless pass over every leaf's
      rows per tree.
    - `quant_train_renew_leaf` is set and the objective does not renew.
      `RENEW_FROM_FLOAT`: after growth, each leaf's value is recomputed from
      the unquantized gradient and hessian sums of its rows. This is
      LightGBM's parameter, and it recovers most of the leaf-value accuracy
      a small bin count costs, at one extra pass over the training rows per
      tree.
    - Neither. `RENEW_NONE`, and the leaf values stand as the quantized
      histogram produced them.

    The objective fact is passed in rather than imported. `boosting.mojo`
    owns `objective_renews_leaves`, and this module must not import
    `boosting` because `boosting` is what will import *this* module when the
    trainer patch lands.
    """
    if objective_renews_leaves:
        return RENEW_BY_OBJECTIVE
    if params.enabled and params.renew_leaf:
        return RENEW_FROM_FLOAT
    return RENEW_NONE


def renewed_leaf_output(
    grad_sum: Float64,
    hess_sum: Float64,
    lambda_l1: Float64,
    lambda_l2: Float64,
) -> Float64:
    """A leaf's value from unquantized sums, for `RENEW_FROM_FLOAT`.

    `tree_parameters_extra.raw_leaf_output` under a name that says where it
    is called from. Re-exported rather than restated so a renewed leaf and a
    grown leaf cannot use two different Newton steps.
    """
    return raw_leaf_output(grad_sum, hess_sum, lambda_l1, lambda_l2)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def count_underflow(
    grad: List[Float64],
    scales: QuantScales,
    rows: List[Int] = [],
) raises -> Int:
    """How many rows have a nonzero gradient that quantizes to zero.

    The honest cost of a small `num_grad_quant_bins`, and the honest cost of
    a wide sample-weight range, reported in rows rather than left to be
    inferred from a worse model. A row whose gradient rounds to zero
    contributes nothing to any split gain, so a large count means the round
    is effectively training on a subset it never chose. One row weighted a
    million times another sets `max_abs_grad`, and at
    `num_grad_quant_bins = 4` every ordinary row then lands on zero; this is
    the number that says so.

    Measured against the deterministic lattice (`|g| * units < 0.5`), which
    is the worst case: stochastic rounding gives such a row a proportional
    chance of landing on one unit instead, which is exactly the bias it
    exists to remove. So this is an upper bound under stochastic rounding
    and an exact count under round-to-nearest.
    """
    if not scales.is_usable():
        raise Error("quantization scales must be finite and positive")
    var n = len(grad)
    var use_all = len(rows) == 0
    var count = n if use_all else len(rows)
    var gp = grad.unsafe_ptr()
    var rp = rows.unsafe_ptr()
    var units = scales.grad_units
    var under = 0
    for i in range(count):
        var r = i if use_all else rp.unsafe_load(i)
        if r < 0 or r >= n:
            raise Error("row index out of range")
        var g = abs(gp.unsafe_load(r))
        if g > 0.0 and g * units < 0.5:
            under += 1
    return under


def lattice_resolution_bits(scales: QuantScales) -> Float64:
    """Roughly how many bits of gradient resolution the lattice carries.

    `log2(grad_max_unit + 1)`, computed without a log by counting bits, so
    this is a report and not an approximation of one. A max-abs lattice at
    `num_grad_quant_bins = 4` reports 2 bits (a gradient is one of -2, -1,
    0, 1, 2 after clamping); the magnitude-sum lattice reports 31.

    Reported for gradients only. The hessian lattice is one bit wider by
    construction under the max-abs rule, and identical under the
    magnitude-sum rule.

    This reports the lattice's *capacity*, not the resolution a given round
    achieves, and the two now differ. `grad_max_unit` is `FIXED_ONE` under
    the magnitude-sum rule whatever the round's magnitudes are, so this still
    answers 31; under `SCALE_SHAPE_POW2` the round's exact scaled total lands
    somewhere in `(2^29, 2^30]`, so between zero and one bit of that capacity
    goes unused. Deliberately not folded in here: this function takes a
    lattice and not a round, and a capacity that moved with the data would be
    a worse diagnostic, not a better one.
    """
    var v = scales.grad_max_unit
    var bits = 0.0
    while v > Int64(0):
        v = v >> 1
        bits += 1.0
    return bits


def describe_quantization_decision(
    decision: QuantDecision, params: QuantGradParams
) -> String:
    """One line naming the mode, the reason, the rule, the rounding, and the
    accumulator width. The shape `unified_memory_policy.describe_quantization_decision`
    uses, so a trace of a training run reads the same whichever policy
    produced the line."""
    if not decision.is_quantized():
        return String(
            "gradient accumulation: float (",
            describe_reason(decision.reason),
            ")",
        )
    return String(
        "gradient accumulation: quantized, ",
        describe_scale_rule(params.scale_rule),
        " lattice, ",
        describe_rounding(params.rounding_mode()),
        " rounding, ",
        String(decision.width),
        "-bit accumulator",
    )
