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
**The CPU histogram builder is here and works; no trainer calls it.** That is
the whole of the change of state. `build_histogram_subset_quantized_into_scratch`
accumulates a node into interleaved Int64 cells over the same row blocks the
Float64 builder uses, `quantize_round_cpu` is the once-per-round map that
feeds it, `decide_cpu_histogram` is the decision for a caller whose backend is
that builder, and `QuantBuildReport` is the marker that says the integer
kernel actually ran. LightGBM's four parameters are on the parameter surface
(`tree_parameters_extra.ExtraTreeParams`, `params.parse_params`) with
LightGBM's four defaults, and `use_quantized_grad=true` is refused there with
a sentence rather than accepted and ignored.

What is still missing is exactly one thing: `boosting.mojo` and `tree.mojo`
have to call `quantize_round_cpu` once per round and
`build_histogram_subset_maybe_quantized` per node, and the split search has to
either read the dequantized `Histogram` (which it can, today, unchanged) or
learn `quantized_split_gain`. Until that lands `CONNECTED` stays False.

Disabled, then, and not reachable from any *training* entry point. `CONNECTED`
is False, `QuantGradParams.default()` is disabled, and `decide` returns
`MODE_FLOAT` with `QUANT_REASON_NOT_CONNECTED` for any request while `CONNECTED`
is False, whatever the parameters say. A caller that explicitly asked for
quantized training gets `check_supported`'s error rather than a silent
downgrade, which is the same rule `unified_memory_policy` applies to a
transfer route it cannot honor. Nothing here is exported from
`src/mojotrees/__init__.mojo`; see
`handoffs/remaining_06_quantized_gradients.md (deleted, recover with git log --all --diff-filter=D -- handoffs/remaining_06_quantized_gradients.md)` for the ordered patch set
that connects it, and `docs/QUANTIZED_GRADIENTS.md` for the numerical
policy in prose.
"""

from std.math import floor, isfinite, round
from std.memory import bitcast
from std.sys.info import simd_width_of

from .apple_cpu_policy import (
    AccumulationPlan,
    derive_accumulation_plan_with,
)
from .binning import BinnedMatrix
from .gain import leaf_score
from .histogram import (
    CONSTANT_HESSIAN,
    Histogram,
    _zeroed_f64,
    _zeroed_int,
    build_histogram_subset_into_scratch,
)
from .parallel import (
    DispatchSettings,
    _env_int,
    dispatch_feature_ranges,
    dispatch_feature_ranges_with,
    dispatch_rows_with,
)
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

comptime QUANT_SIMD_LANES = 4 * simd_width_of[DType.int64]()
"""Lanes the integer zeroing and fold loops use, sized above the hardware
width so several vector operations stay in flight. The Int64 twin of
`histogram.SIMD_LANES`, and the same reasoning: these are the elementwise
kernels, not the scatter, and a scatter cannot be vectorized at all."""

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


# ---------------------------------------------------------------------------
# The Int16 *staging* bound
# ---------------------------------------------------------------------------
#
# WHAT THIS IS NOT. It is not section 5 of `docs/design/ACCURACY_BUDGET.md`.
# That section prices an int16 *accumulator*: a threadgroup histogram cell
# holding the partial sum of many rows, whose bound is
# `sum over the tile's rows in one bin`, which the shipped magnitude-sum scale
# blows through at three to three hundred rows per tile and which therefore
# cannot be had without LightGBM's max-abs per-row clamp and the two percent
# of effective sample size that clamp costs. None of that applies here.
#
# WHAT THIS IS. The bound on one *stored value*. `_quantize_grad_hess_kernel`
# already writes each row's `Int32(round(g_r * s))` once per round into a
# buffer the histogram kernels gather per (row, feature) visit. If every one
# of those integers happens to fit sixteen bits, the buffer can hold them as
# Int16 and the gather reads half as many bytes for **the identical integers**:
# `Int32(Int16(q)) == q` whenever `-32768 <= q <= 32767`, sign extension being
# exact, and every accumulator downstream -- threadgroup Int32, global Int32,
# the sibling subtraction -- is untouched. So this is exact by construction
# given the bound, and it costs no accuracy at all, which is what separates it
# from candidate 2 of the budget.
#
# THE SCOPE, STATED PLAINLY BECAUSE THIS REPOSITORY HAS PAID FOR GETTING IT
# WRONG TWICE. The bound is **per row, over every row of the round**, at the
# round's scale. It is not a per-node bound and it must not be argued as one:
# the staged buffer is built once per round and read by every node of the
# tree, so a bound that held only for one node's rows would not license the
# representation the other nodes read. A per-row bound over the whole round is
# strictly stronger than any per-node bound, and it is the one the check
# enforces.
#
# WHEN IT HOLDS. Under the shipped magnitude-sum scale `s = 2^30 / sum|g|`,
# `q_r = g_r * 2^30 / sum|g|`, so the bound is
#
#     max_r |g_r| / sum_r |g_r| <= 32767 / 2^30 = 3.05e-5
#
# A row carries about `1/n` of the total magnitude, so the condition is
# roughly `n >= 32767 * max|g| / mean|g|`: it fails on small fits and holds on
# large ones. That is the opposite of the usual direction and it is the useful
# direction, because the gather this saves is 57 percent of a histogram phase
# only at the large shapes. It is also why the arm is off by default and why
# a violation raises instead of quietly widening.


def int16_staging_fits(max_abs: Float64, scale: Float64) raises -> Bool:
    """Whether every row of a round whose largest derivative magnitude is
    `max_abs` quantizes, at `scale`, into a signed 16-bit word.

    The inequality is `max_abs * scale <= 32767`, which is the per-row form of
    the block comment above. Stated on the *maximum* rather than the sum
    because that is the quantity the representation constrains; the sum is
    what `fixed_point_scale_pow2` constrains and the two are different bounds
    on different objects.

    No slack term, unlike `histogram_gpu._check_window_bound`. That check
    compares a Float64 product against `2^30` and needs `2^-24` of room
    because the scale's own derivation admits an ulp past the round number.
    Here the comparison is against 32,767, the product is more than fifteen
    orders of magnitude below the point where a Float64 ulp reaches one
    lattice unit, and the device makes the same decision by comparing the
    *integer* it just formed. Adding slack would only let through a value the
    device would then reject.

    Raises rather than returning `False` on a non-finite input, because a
    non-finite magnitude is not a bound that failed, it is a round that
    should never have derived a scale at all.
    """
    if not isfinite(max_abs) or max_abs < 0.0:
        raise Error(
            "an Int16 staging bound needs a finite non-negative magnitude"
        )
    if not isfinite(scale) or scale <= 0.0:
        raise Error("quantization scales must be finite and positive")
    return max_abs * scale <= Float64(INT16_LIMIT)


def check_int16_staging(
    max_abs: Float64, scale: Float64, plane: String
) raises:
    """`int16_staging_fits`, as a refusal.

    The host twin of the device check in
    `gpu_active_rows._quantize_grad_hess_i16_kernel`, which evaluates the same
    inequality per row on the integer it has just formed rather than on a
    maximum it would have to reduce. Both are here so that the rule has one
    written definition and so that it is testable without a device.

    Raising is the only correct answer and a clamp is not. A clamped row is a
    gradient the fit never had, silently, for one round, on one plane, in a
    way no fixture distinguishes from a data change; the whole point of this
    arm is that it is bit-identical or it is nothing.
    """
    if not int16_staging_fits(max_abs, scale):
        raise Error(
            String(
                "the packed Int16 gradient staging arm needs every row's",
                " quantized ",
                plane,
                " to fit a signed 16-bit word: the largest magnitude ",
                String(max_abs),
                " times the scale in force ",
                String(scale),
                " is ",
                String(max_abs * scale),
                ", past the ",
                String(INT16_LIMIT),
                " the Int16 staged buffer holds. This bound tightens as the"
                " row count falls, so the remedy is the Int32 buffer"
                " (GpuActiveRows.set_packed_gradients(False), the default),"
                " not a different scale.",
            )
        )


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
    stats: GradientStats,
    params: QuantGradParams,
    const_hessian: Bool = False,
) raises -> QuantScales:
    """This round's lattice, from the measured magnitudes and the rule.

    `const_hessian` is the objective's guarantee that every row's hessian is
    exactly `histogram.CONSTANT_HESSIAN`, the same declaration the float
    builders take. LightGBM has it too, and it changes the hessian lattice:
    `GradientDiscretizer::DiscretizeGradients` sets
    `hessian_scale_ = max_hessian_abs_` when `is_constant_hessian_` and
    `max_hessian_abs_ / num_grad_quant_bins` otherwise
    (`src/treelearner/gradient_discretizer.cpp`, LightGBM 4.7.0.99), so the
    constant-hessian lattice is exactly one unit wide and every row's hessian
    quantizes to the integer 1. That is what makes the dequantized hessian
    plane come back as `Float64(count)` -- bit for bit the float path's
    hessian plane -- rather than as an approximation of it, and it is why the
    declaration is worth threading this far down. Under
    `SCALE_MAGNITUDE_SUM` the flag changes nothing: that rule derives the
    hessian factor from `sum|h|` and lands on a power of two, which is
    already an exact-integer lattice for a constant hessian.

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
    # LightGBM's constant-hessian hessian lattice, applied under *both* rules
    # rather than only under its own. `hessian_scale_ = max_hessian_abs_`
    # makes one lattice unit stand for exactly the constant, so every row
    # quantizes to the integer 1, a bin's hessian sum is its row count, and
    # the dequantized hessian plane is `Float64(count)` -- bit for bit what
    # the Float64 builder produces. Deriving the hessian factor from
    # `sum|h|` instead would land on some power of two `2^k`, quantize every
    # row to `2^k`, and give back the same Float64 the long way round while
    # costing `k` bits of the packed cell's low field for nothing. This is
    # the one place the magnitude-sum rule adopts LightGBM's, and it adopts
    # it because it is strictly better here.
    var h_const_units = 0.0
    if const_hessian:
        var hm = stats.max_abs_hess
        if not isfinite(hm):
            raise Error("gradients and hessians must be finite")
        if hm < MAGNITUDE_FLOOR:
            hm = MAGNITUDE_FLOOR
        h_const_units = 1.0 / hm

    if params.scale_rule == SCALE_MAGNITUDE_SUM:
        # Float32 on purpose: the device stores the scale at that width and
        # multiplies by it, so a host factor of any other precision would
        # put the CPU and GPU on different lattices. Under
        # `DEFAULT_SCALE_SHAPE` the width is no longer what makes them agree
        # -- a power of two is the same number at every width -- but the call
        # is unchanged so the arbitrary arm keeps the property it needs.
        var g_units = Float64(fixed_point_scale(stats.sum_abs_grad))
        var h_units = Float64(fixed_point_scale(stats.sum_abs_hess))
        if const_hessian:
            return QuantScales(
                g_units,
                h_const_units,
                Int64(FIXED_ONE),
                Int64(1),
                BOUND_TOTAL,
                SCALE_MAGNITUDE_SUM,
            )
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
    # LightGBM's `hessian_scale_`, inverted. One unit for the whole hessian
    # range under a constant hessian, `num_grad_quant_bins` units otherwise.
    var h_max_unit = Int64(1) if const_hessian else params.hess_max_unit()
    var g_units = Float64(params.grad_max_unit()) / g_max
    var h_units = h_const_units if const_hessian else (
        Float64(h_max_unit) / h_max
    )
    if not (isfinite(g_units) and isfinite(h_units)):
        raise Error(
            "gradient/hessian magnitudes are out of range for a quantized"
            " lattice"
        )
    return QuantScales(
        g_units,
        h_units,
        params.grad_max_unit(),
        h_max_unit,
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

    `ROUND_STOCHASTIC` is **LightGBM's rule, magnitude-symmetric**, not
    `floor(x + u)`. Read off `src/treelearner/gradient_discretizer.cpp`
    (`GradientDiscretizer::DiscretizeGradients`, LightGBM 4.7.0.99), which
    spells it

        gradient >= 0 ? int8(g * inv_scale + u) : int8(g * inv_scale - u)

    with `static_cast<int8_t>` truncating toward zero. So the magnitude is
    what gets the dither: `|q| = floor(|y| + u)`, and the sign is carried
    through untouched. That is *not* the same integer as `floor(y + u)` for a
    negative `y` -- at `y = -1.3, u = 0.9` the symmetric rule gives -2 and
    `floor` gives -1 -- and it is the LightGBM one that this package matches,
    because matching a comparator's arithmetic is the whole point of naming
    the parameter after it. Both spellings are unbiased and both move a value
    by strictly less than one unit, so `accumulation_bound` is unaffected.

    That rounding is unbiased in the sense that matters: a value 0.3 of the
    way to the next unit lands there with probability 0.3, so a sum of many
    quantized values converges on the exact scaled sum instead of on a
    systematically rounded one. It costs twice the residue bound of
    round-to-nearest, which `accumulation_bound` accounts for.

    `u` comes from `quant_uniform(counter)`, a counter-based draw and not an
    engine. LightGBM's own stochastic rounding is **not reproducible**: it
    fills `gradient_random_values_` from one `std::mt19937(seed + thread_id)`
    per OpenMP block, so the draw a row gets depends on how many threads ran,
    and it then rotates the array by a fresh `random_values_use_start` every
    round. Neither of those can be reproduced at a different worker count.
    mojotrees keys the draw on `(seed, round, class, plane, row)` instead, so
    the same row gets the same `u` at one worker and at eight, on the CPU and
    on a device. That is a deliberate divergence from LightGBM and it is in
    the safe direction: LightGBM's scheme is a *distribution* over roundings
    and any member of it is as valid as any other, while a distribution that
    depends on the thread count is not something a test can assert on.

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
        var u = quant_uniform(counter)
        if y >= 0.0:
            return Int64(floor(y + u))
        return -Int64(floor(-y + u))
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
        return Histogram.from_planes(g^, h^, c^, self.n_features, self.n_bins)

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
# The CPU quantized accumulation path
# ---------------------------------------------------------------------------
#
# What this is, in one paragraph. `histogram.mojo` accumulates a node into
# three Float64 planes, optionally over contiguous ascending row blocks with
# a private histogram per block and an ascending fold
# (`_accumulate_subset_blocked_at`). Everything below is that same
# decomposition over **integer** cells: the node's rows are quantized once
# onto this round's lattice, the blocks accumulate Int64 counts of lattice
# units, the fold sums the blocks, and the fold dequantizes into the caller's
# `Histogram` on the way out.
#
# WHY IT LIVES HERE AND NOT IN `histogram.mojo`
# ---------------------------------------------
# `quantized_gradient` already imports `histogram` (for `Histogram` itself and
# for the zeroed-plane helpers), so `histogram` cannot import this module back
# without a cycle. The kernel therefore sits on the side of the edge that can
# see both, and it reaches the row-block policy through `apple_cpu_policy`
# directly -- which imports nothing from this package, so nothing here widens
# the dependency graph. The alternative, moving `QuantScales` and the rounding
# into `histogram.mojo`, would put the package's numerical policy inside its
# hottest kernel file and is the wrong direction.
#
# WHAT INTEGER CELLS BUY, STATED EXACTLY
# --------------------------------------
# **The fold becomes exact rather than merely deterministic, and that is a
# real simplification and not a slogan.** The Float64 blocked kernel has to
# argue at length that the block count is a *value* and not a schedule: a fold
# over `B` partial sums is a different Float64 from one ascending sum, so
# `plan_row_block_count` is forbidden to look at the core count and the fold
# is forbidden to run out of order. Integer addition is associative and
# commutative, so none of that argument is needed here. A cell's value is the
# exact integer sum of its rows' quantized gradients at **every** block count,
# in **every** fold order, at **every** worker count, with the blocks visited
# in any order at all. The block count stops being part of the result and goes
# back to being what it looks like: a scheduling knob.
#
# Two consequences worth naming. First, `MOJOTREES_CPU_ROW_BLOCKS` cannot move
# a bit on this path, so it is a pure A/B here where on the float path it is
# the one environment variable that changes an answer. Second, sibling
# subtraction over these cells is exact (`subtract_quantized`), where the
# Float64 one is exact only up to cancellation.
#
# THE CELL LAYOUT IS LIGHTGBM'S, NOT THIS PACKAGE'S
# -------------------------------------------------
# A cell is a **packed `(gradient, hessian)` pair in one integer**: the
# gradient in the high `HIST_BITS` bits, the hessian in the low ones, and one
# row's contribution to a bin is **one integer add**. That is
# `DenseBin::ConstructHistogramIntInner` (`src/io/dense_bin.hpp`), and the two
# widths are LightGBM's two: an int16 pair inside an int32, and an int32 pair
# inside an int64. `serial_tree_learner.cpp` branches on `hist_bits <= 16` and
# so does `build_histogram_subset_quantized_into_scratch`.
#
# The overflow rule is LightGBM's too (`histogram_bits_for_node`, from
# `GradientDiscretizer::SetNumBitsInHistogramBin`) and it is *not* this
# package's 2^30 fixed-point bound, which was derived for a per-plane Int32
# accumulator and does not transfer to a two-field cell. Where they disagree
# the LightGBM rule wins here, and that function names the disagreement.
#
# On the constant-hessian arm LightGBM packs the literal 1 into the low field,
# so the low field accumulates the row **count** and the histogram carries no
# count plane at all. That is the arm `lane/lgbm-cell-layout` is landing on
# the Float64 side, with leaf counts coming from the partition. On the general
# arm this builder does still keep a count plane, because mojotrees's
# `Histogram` promises a per-bin count and a data partition is a per-leaf
# number that cannot answer a per-bin question. **That is the one place this
# lane guessed**: if the incoming layout drops per-bin counts unconditionally,
# the general arm's count plane and its fold loop are what come out, and
# nothing else here changes.


def cpu_quant_grad_allowed() -> Bool:
    """Whether the CPU quantized accumulation path may run at all.

    `MOJOTREES_CPU_QUANT_GRAD=0` forces every build back onto the Float64
    accumulation regardless of what a caller asked for, which is the off
    switch a bisection wants. Anything else, including unset, leaves the path
    available; it still does nothing until a caller enables
    `use_quantized_grad`, which is off by default.

    The convention is `parallel.mojo`'s and `histogram.const_hessian_allowed`'s:
    an integer read through `_env_int`, a default that means "unchanged
    behavior", and zero meaning off. It is an environment override and not a
    public parameter on purpose -- the public surface is exactly LightGBM's
    four names -- for the reason `histogram.const_hessian_allowed` gives.
    """
    return _env_int("MOJOTREES_CPU_QUANT_GRAD", 1) != 0


def env_cpu_quant_scale_rule() -> Int:
    """Which lattice the CPU quantized path derives, as an A/B override.

    `MOJOTREES_CPU_QUANT_SCALE=0` selects `SCALE_MAX_ABS`, which is LightGBM's
    rule and the only one it has: `units = (num_grad_quant_bins / 2) / max|g|`,
    a lattice four units wide at the default bin count. Anything else,
    including unset, selects `SCALE_MAGNITUDE_SUM` at `SCALE_SHAPE_POW2`,
    which is the rule the GPU histogram already ships.

    **The default is the magnitude-sum rule, and that is a decision with a
    consequence worth stating rather than burying.** It puts the CPU on the
    same lattice as the device, which is what makes a CPU histogram and a GPU
    histogram of the same node comparable at all and what
    `histogram.build_histogram_subset_replica_into` already assumes. It also
    means that at the default, `num_grad_quant_bins` **does not affect the
    lattice**: the magnitude-sum rule derives its factor from `sum|g|` and
    ignores the bin count. The parameter is still validated, still carried,
    and becomes load-bearing the moment this variable selects `SCALE_MAX_ABS`.
    A caller comparing against LightGBM's own quantized arm value for value
    wants `MOJOTREES_CPU_QUANT_SCALE=0`; a caller comparing the CPU against
    this package's own GPU wants the default.

    It is an environment variable and not a parameter because the public
    surface is exactly LightGBM's names, and LightGBM has no scale-rule
    parameter to be exactly.
    """
    if _env_int("MOJOTREES_CPU_QUANT_SCALE", 1) == 0:
        return SCALE_MAX_ABS
    return SCALE_MAGNITUDE_SUM


def cpu_quant_params(
    use_quantized_grad: Bool,
    num_grad_quant_bins: Int = DEFAULT_NUM_GRAD_QUANT_BINS,
    quant_train_renew_leaf: Bool = False,
    stochastic_rounding: Bool = True,
) raises -> QuantGradParams:
    """LightGBM's four parameters, with this build's scale rule folded in.

    The one constructor a CPU caller should use: it takes exactly LightGBM's
    names and defaults (`use_quantized_grad=false`,
    `num_grad_quant_bins=4`, `quant_train_renew_leaf=false`,
    `stochastic_rounding=true`, all read off `include/LightGBM/config.h` of
    LightGBM 4.7.0.99) and supplies the two fields LightGBM has no parameter
    for: the rounding seed, and the scale rule, which comes from
    `env_cpu_quant_scale_rule`.

    `max_width` is `WIDTH_64` because the host accumulator here is Int64. A
    narrower device or transport asks `accumulator_width` for its own answer.
    """
    var p = QuantGradParams(
        use_quantized_grad,
        num_grad_quant_bins,
        stochastic_rounding,
        quant_train_renew_leaf,
        DEFAULT_QUANT_SEED,
        env_cpu_quant_scale_rule(),
        WIDTH_64,
    )
    p.validate()
    return p^


def decide_cpu_histogram(
    stats: GradientStats,
    params: QuantGradParams,
    max_node_rows: Int,
    const_hessian: Bool = False,
) raises -> QuantDecision:
    """`decide`, for a caller whose accumulation path is the one below.

    `const_hessian` is threaded through to `derive_scales` and must be the
    same declaration `quantize_round_cpu` was given, or the round's rows are
    quantized on one hessian lattice and its histograms dequantized on
    another. That is a silent wrongness rather than an error, so the two calls
    take the same flag and a test asserts the two lattices come out equal.

    Identical to `decide` in every check except the `CONNECTED` gate, which it
    does not consult, and this is the one place in the module that skips it.
    The distinction is exact and it is not a loophole: `CONNECTED` is the
    package's statement that **a trainer** has been wired to this module and
    validated against the float path, and that is still False -- no trainer
    calls any of this, `boosting.mojo` and `tree.mojo` are untouched, and
    `decide` still refuses every request. What *is* now true is the narrower
    claim this function makes, that a CPU histogram builder exists and holds
    its bounds, and a test of that builder needs a way to say so without
    asserting the wider claim.

    Flipping `CONNECTED` remains the last step of the connection sequence in
    `handoffs/remaining_06_quantized_gradients.md (deleted, recover with git log --all --diff-filter=D -- handoffs/remaining_06_quantized_gradients.md)`, and this function is not
    it.
    """
    params.validate()
    if not params.enabled:
        return QuantDecision.floating(QUANT_REASON_NOT_REQUESTED)
    if not cpu_quant_grad_allowed():
        return QuantDecision.floating(QUANT_REASON_BACKEND)
    if stats.n_rows <= 0 or max_node_rows <= 0:
        return QuantDecision.floating(QUANT_REASON_NO_ROWS)
    if not stats.finite:
        return QuantDecision.floating(QUANT_REASON_NON_FINITE)
    if stats.is_degenerate():
        return QuantDecision.floating(QUANT_REASON_DEGENERATE)

    var scales = derive_scales(stats, params, const_hessian)
    if not scales.is_usable():
        return QuantDecision.floating(QUANT_REASON_DEGENERATE)
    var width = accumulator_width(max_node_rows, scales, params)
    if width == WIDTH_NONE:
        return QuantDecision.floating(QUANT_REASON_OVERFLOW)
    return QuantDecision(MODE_QUANTIZED, QUANT_REASON_OK, width, scales.copy())


@fieldwise_init
struct QuantBuildReport(Copyable, Movable):
    """What one histogram build actually did: **the path marker**.

    A test that compares a quantized histogram against a float one and passes
    whether or not quantization fired establishes nothing, and this package
    has shipped that test three times. So the builder reports, and the
    assertions are on the report rather than on an inference from the values.

    `row_accumulations` is the load-bearing field: the number of
    `(node row, active feature)` integer accumulations the kernel performed.
    It is `n_active * row_count` on the quantized path and **exactly zero** on
    the float path, so no fixture can pass while silently running the float
    builder. `blocks` and `group_width` are the shape the plan chose, so a
    fixture meant to exercise the blocked kernel can assert it got one.

    `hist_bits` is the width of each half of the packed cell, from
    `histogram_bits_for_node`, so a fixture can assert which of LightGBM's two
    arms it exercised. `const_hessian_elided` says whether the low field
    carried the row count instead of a quantized hessian, which is the arm on
    which the count plane disappears entirely.

    There is deliberately no rounding-mode field. Rounding happens once per
    round in `quantize_round_cpu`, not per node, and a builder handed an
    integer array cannot tell how it was rounded. Reporting a guess would be
    exactly the kind of marker that establishes nothing.
    """

    var mode: Int
    var reason: Int
    var scale_rule: Int
    var hist_bits: Int
    var blocks: Int
    var group_width: Int
    var row_accumulations: Int
    var const_hessian_elided: Bool
    var scales: QuantScales

    @staticmethod
    def floating(reason: Int) -> QuantBuildReport:
        return QuantBuildReport(
            MODE_FLOAT, reason, SCALE_MAX_ABS,
            0, 1, 1, 0, False, QuantScales.identity(),
        )

    def is_quantized(self) -> Bool:
        return self.mode == MODE_QUANTIZED

    def describe(self) -> String:
        if not self.is_quantized():
            return String(
                "cpu histogram: float (", describe_reason(self.reason), ")"
            )
        return String(
            "cpu histogram: quantized, ",
            describe_scale_rule(self.scale_rule),
            " lattice, ",
            describe_histogram_bits(self.hist_bits),
            " cells, ",
            String(self.blocks),
            " row blocks, group width ",
            String(self.group_width),
            ", ",
            String(self.row_accumulations),
            " packed accumulations",
        )


def quantize_round_cpu(
    grad: List[Float64],
    hess: List[Float64],
    params: QuantGradParams,
    key: QuantRoundKey,
    mut qgrad: List[Int64],
    mut qhess: List[Int64],
    rows: List[Int] = [],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> QuantScales:
    """One round's quantization, start to finish: measure, derive, quantize.

    The three existing steps in the order this module's contract fixes --
    `gradient_stats` on the row subset the tree will see, then
    `derive_scales`, then the per-row rounding -- composed so a trainer makes
    one call per round instead of three that could be reordered. `rows` is the
    bag or the GOSS selection, empty for every row, and it must already carry
    whatever GOSS scaling this round applied.

    Every row of the dataset is quantized, not just the sampled ones, for the
    reason `quantize_rows` gives: the quantized arrays stay indexed by row id
    like the float ones, so a node's row window indexes them directly and a
    bagged and an unbagged round produce the same integer for the same row.

    Parallel, and deterministic because row `r` draws from `stream + r` and
    from nothing else. The dispatch is elementwise over disjoint ascending
    blocks, so the arrays that come out are identical at every worker count
    and at every task count -- this is not an argument about summation order
    because there is no summation here, only a map.

    **Derived bound on the pass**, not measured: two Float64 reads and two
    Int64 writes per row, 32 bytes, plus the `gradient_stats` pass ahead of it
    at 16 bytes read per row. 48 bytes per row per round against a round whose
    histogram passes read the same gradients `n_active` times.
    """
    if len(grad) != len(hess):
        raise Error("gradient/hessian length must match")
    var stats = gradient_stats(grad, hess, rows)
    var scales = derive_scales(stats, params, const_hessian)
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

    def quantize_range(start: Int, end: Int) {imm}:
        for r in range(start, end):
            qg.unsafe_store(
                r,
                quantize_scalar(
                    gp.unsafe_load(r), g_units, g_lim, mode,
                    g_stream + UInt64(r),
                ),
            )
            qh.unsafe_store(
                r,
                quantize_scalar(
                    hp.unsafe_load(r), h_units, h_lim, mode,
                    h_stream + UInt64(r),
                ),
            )

    dispatch_rows_with(settings, quantize_range, n, 4 * n)
    return scales^

comptime HIST_BITS_16 = 16
"""LightGBM's narrow packed cell: an int16 `(gradient, hessian)` pair inside
one int32. `serial_tree_learner.cpp` branches on `hist_bits <= 16` and this is
that arm."""

comptime HIST_BITS_32 = 32
"""LightGBM's wide packed cell: an int32 pair inside one int64."""

comptime HIST_BITS_NONE = 0
"""No supported packed cell holds this node's bound. A fallback condition,
not an error."""


def histogram_bits_for_node(
    n_rows: Int, scales: QuantScales, mode: Int
) raises -> Int:
    """How wide each half of the packed histogram cell has to be, in bits.

    **This is LightGBM's rule, and it is theirs rather than ours.** Read off
    `GradientDiscretizer::SetNumBitsInHistogramBin`
    (`src/treelearner/gradient_discretizer.cpp`, LightGBM 4.7.0.99), which is
    verbatim:

        max_stat_per_bin = num_data_in_leaf * num_grad_quant_bins
        if      max_stat_per_bin < 256   -> 8
        else if max_stat_per_bin < 65536 -> 16
        else                             -> 32

    per leaf, recomputed for both children at every split. It is a bound on
    the **hessian half**, which is the binding one: with `hess_max_unit = B`
    and `grad_max_unit = B / 2` the hessian sum is at most `n * B` and the
    gradient sum at most `n * B / 2`, so `n * B < 2^bits` bounds the unsigned
    low field and simultaneously puts the signed high field inside
    `2^(bits-1)`. One test, both halves.

    WHY THIS RULE AND NOT THE PACKAGE'S 2^30 ONE
    --------------------------------------------
    The repository's overflow argument (`docs/GPU_PORTABILITY.md`,
    `fixed_point_scale_pow2`) bounds a **per-plane Int32 accumulator** under a
    power-of-two magnitude-sum scale: the exact scaled total is at most 2^30,
    a node is a subset, deterministic rounding adds `n/2`, so a cell fits
    Int32. It is correct and it **does not transfer to a packed pair**,
    because a packed cell has two fields and the argument bounds one plane.
    Where the two disagree, the rule below wins, and the disagreement is worth
    naming: the 2^30 argument would say "Int32 suffices" and, applied
    naively to a packed cell, would produce an int16 pair inside an int32 that
    overflows at the root of any dataset above about 32,000 rows.

    They do not actually conflict once both are stated as field bounds, and
    this function is where that reconciliation lives. It generalizes
    LightGBM's rule to the two lattices this package derives, by bounding each
    half separately:

    - `BOUND_PER_ROW` (LightGBM's `SCALE_MAX_ABS`): low field `n *
      hess_max_unit`, high field `n * grad_max_unit`. At `hess_max_unit = B`
      and `grad_max_unit = B/2` this reduces to LightGBM's single test,
      exactly.
    - `BOUND_TOTAL` (`SCALE_MAGNITUDE_SUM` at `SCALE_SHAPE_POW2`): the scale
      bounds the *total* at `FIXED_ONE`, so each field is at most
      `2^30 + n * residue`. That is above 2^16 for every nonempty node, so
      **the magnitude-sum lattice always lands on the 32-bit arm**, an int32
      pair inside an int64. That is not a defect of the rule, it is what a
      30-bit lattice costs, and it is the honest reason LightGBM can use a
      4-byte cell where this package's default cannot.
    - Under a declared constant hessian the low field accumulates the row
      **count** (LightGBM packs the literal 1; see `_packed_row_value`), so
      its bound is `n` whichever lattice is in use.

    ONE DIVERGENCE, DELIBERATE
    --------------------------
    LightGBM's ladder ends in an unconditional `else 32` and therefore
    silently wraps when `n * B >= 2^32`, which at `B = 4` is any leaf above
    about 1.07 billion rows. This returns `HIST_BITS_NONE` there instead, and
    the caller falls back to Float64 accumulation with a named reason. A
    silent integer wraparound in a histogram is the one failure this whole
    module exists to make impossible, so reproducing it for parity would be
    the wrong kind of faithfulness.

    **Derived bound**, not measured, on where each arm applies at LightGBM's
    defaults (`B = 4`, non-constant hessian): the 16-bit arm holds while
    `4n < 65536`, i.e. **n < 16,384 rows in the node**, and the 32-bit arm
    above it. So a million-row root is on the wide arm and everything below
    roughly the fourteenth level of the tree is on the narrow one.
    """
    if n_rows < 0:
        raise Error("row count must not be negative")
    var rows = Float64(n_rows)
    var hi: Float64
    var lo: Float64
    if scales.bound_kind == BOUND_TOTAL:
        hi = FIXED_ONE + rows * residue_per_row(mode)
        # `hess_max_unit == 1` is how `derive_scales` records a constant
        # hessian: one lattice unit per row, so the low field is the count.
        lo = rows if scales.hess_max_unit == Int64(1) else hi
    else:
        hi = rows * Float64(scales.grad_max_unit)
        lo = rows * Float64(scales.hess_max_unit)
    if not (isfinite(hi) and isfinite(lo)):
        return HIST_BITS_NONE
    # The signed high field needs `hi < 2^(bits-1)`; the unsigned low field
    # needs `lo < 2^bits`. Testing `2 * hi` against the same threshold as `lo`
    # is the same statement with one comparison per width.
    var need = 2.0 * hi
    if lo > need:
        need = lo
    if need < 65536.0:
        return HIST_BITS_16
    if need < 4294967296.0:
        return HIST_BITS_32
    return HIST_BITS_NONE


def describe_histogram_bits(bits: Int) -> String:
    if bits == HIST_BITS_16:
        return "int16 pair in int32"
    if bits == HIST_BITS_32:
        return "int32 pair in int64"
    return "none"


def build_histogram_subset_quantized_into_scratch(
    mut out: Histogram,
    mut qscratch: List[Int64],
    data: BinnedMatrix,
    qgrad: List[Int64],
    qhess: List[Int64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    scales: QuantScales,
    features: List[Int] = [],
    const_hessian: Bool = False,
    rounding: Int = ROUND_NEAREST,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> QuantBuildReport:
    """The packed integer accumulation of one node, dequantized into `out`.

    The quantized twin of `histogram.build_histogram_subset_into_scratch`,
    over the window `rows[row_start : row_start + row_count]`, and it takes
    the same shape of arguments for the same reasons: a caller-owned scratch
    so a grower visiting hundreds of nodes allocates once per tree, a row
    window so node row ids live in one shared arena, an optional feature
    subset whose excluded slices come back zeroed, and a `DispatchSettings`
    snapshot so the build reads no environment variable.

    `qgrad` and `qhess` are the whole round's quantized derivatives, indexed
    by row id, from `quantize_round_cpu`. `scales` is the lattice they were
    quantized on and is what the fold dequantizes with; handing a lattice that
    does not match the arrays produces a wrong histogram and nothing here can
    detect it, which is why `quantize_round_cpu` returns the two together.
    `rounding` is the mode those arrays were rounded with, and it is used for
    one thing only: `histogram_bits_for_node` needs the residue bound.

    THE CELL, WHICH IS LIGHTGBM'S AND NOT THIS PACKAGE'S
    ----------------------------------------------------
    One cell is a **packed `(gradient, hessian)` pair in a single integer**,
    high field gradient and low field hessian, and one row's contribution is
    **one integer add**. That is `DenseBin::ConstructHistogramIntInner`
    (`src/io/dense_bin.hpp`, LightGBM 4.7.0.99) exactly:

        gradient_packed = (int8_t(gradient_16 >> 8) << HIST_BITS)
                          | (gradient_16 & 0xff)
        out_ptr[ti] += gradient_packed

    and the constant-hessian arm of the same function packs the literal `1`
    into the low field instead of the hessian, so the low field accumulates
    the **row count**. Two consequences follow and both are load-bearing here:
    the histogram carries no count plane on that arm, which is what
    `lane/lgbm-cell-layout` is landing on the Float64 side, and unpacking is
    two instructions rather than two loads.

    Componentwise addition of packed values is exact provided the low field
    never carries into the high one, which is precisely what
    `histogram_bits_for_node` guarantees, and it is the reason that rule is
    LightGBM's rather than this package's 2^30 one -- see that function.

    **The fold stays exact.** Packing does not weaken it. Integer addition is
    associative and commutative whether the integers carry one field or two,
    so the row-block fold below is the exact sum at every block count, in
    every order, at every worker count. Nothing in this file has to argue that
    the block count is a value rather than a schedule, which is the argument
    `histogram.mojo` has to make at length for its Float64 twin.

    THE SCRATCH, WHICH THIS FUNCTION DEFINES
    ----------------------------------------
    One `List[Int64]`, grown and never shrunk, contents irrelevant on entry
    and unspecified on exit, viewed through a pointer of the cell type. In
    cell units:

        [0, row_count)                  the node's packed per-row values
        [.., + blocks * cells)          private packed histograms
        [.., + blocks * cells)          private counts, general arm only

    with `cells = n_active * n_bins`, the compact active-slice shape. An
    excluded feature gets no private storage and no fold.

    **Derived bound on the scattered traffic**, not measured, per (row, active
    feature), and this is the whole argument for the packing:

        Float64 builder          3 read-modify-writes, 3 lines, 48 bytes
        packed, general arm      2 read-modify-writes, 2 lines, 32 or 24 bytes
        packed, constant hessian 1 read-modify-write,  1 line,  16 or 8 bytes

    The two figures per packed row are the 32-bit and the 16-bit cell. The
    general arm still pays a count plane because `Histogram` promises a
    per-bin count and LightGBM's does not -- LightGBM takes leaf counts from
    its data partition, which is a per-leaf number and cannot answer a
    per-bin question. On the constant-hessian arm that plane disappears
    entirely, because the low field is the count.

    **Derived bound on the per-row payload**: the Float64 gather writes and
    then re-reads 16 bytes per row (a `(g, h)` Float64 pair); the packed
    gather writes 8 or 4. At a million rows that is 16 MB against 8 MB or
    4 MB, streamed once per feature group.

    Returns a `QuantBuildReport`. Read `row_accumulations` before believing a
    fixture.
    """
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_features = data.n_features
    if len(qgrad) != n_rows or len(qhess) != n_rows:
        raise Error("quantized gradient/hessian length must equal n_rows")
    if not out.matches(n_features, n_bins):
        raise Error("output histogram shape must match the data")
    if row_start < 0 or row_count < 0 or row_start + row_count > len(rows):
        raise Error("row window out of range")
    if not scales.is_usable():
        raise Error("quantization scales must be finite and positive")
    for i in range(len(features)):
        if features[i] < 0 or features[i] >= n_features:
            raise Error("feature index out of range")

    var use_all = len(features) == 0
    var n_active = n_features if use_all else len(features)

    # Excluded features' slices are never accumulated and never folded, so
    # they are zeroed here. Serial and outside the parallel section: it is
    # `(n_features - n_active) * n_bins` cells and it is zero in the common
    # case of no feature subsampling.
    if not use_all:
        var active = List[UInt8](capacity=n_features)
        active.resize(n_features, UInt8(0))
        for i in range(len(features)):
            active[features[i]] = UInt8(1)
        for f in range(n_features):
            if active[f] == UInt8(0):
                var base = f * n_bins
                for b in range(n_bins):
                    out.set_grad_at(base + b, 0.0)
                    out.set_hess_at(base + b, 0.0)
                    out.set_count_at(base + b, 0)

    var bits = histogram_bits_for_node(row_count, scales, rounding)
    if bits == HIST_BITS_NONE:
        # No supported packed cell holds this node's field bounds. A caller
        # that reaches this has asked for a node past a billion rows on
        # LightGBM's lattice; the answer is the float path, named, not a
        # silent wraparound.
        return QuantBuildReport.floating(QUANT_REASON_OVERFLOW)

    var plan = derive_accumulation_plan_with(
        settings.policy, n_features, n_active, n_bins, row_count, True
    )
    var blocks = plan.row_blocks if plan.row_blocks > 0 else 1
    var block_rows = plan.block_rows if blocks > 1 else row_count
    var cells = n_active * n_bins

    if n_active <= 0 or row_count <= 0 or n_bins <= 0:
        # Nothing to accumulate. Every active slice still has to come back
        # zeroed, which the excluded-feature loop above did not cover.
        for j in range(n_active):
            var f = j if use_all else features[j]
            var base = f * n_bins
            for b in range(n_bins):
                out.set_grad_at(base + b, 0.0)
                out.set_hess_at(base + b, 0.0)
                out.set_count_at(base + b, 0)
        return QuantBuildReport(
            MODE_QUANTIZED, QUANT_REASON_OK, scales.rule, bits, 1,
            plan.group_width, 0, False, scales.copy(),
        )

    # `derive_scales` records a declared constant hessian as a one-unit
    # hessian lattice, so this is the check rather than a second flag: the
    # elision is exact only when every row's hessian quantizes to the integer
    # 1, and `hess_max_unit == 1` is precisely that lattice. Checked rather
    # than trusted, because the caller's `const_hessian` and the lattice it
    # derived are two statements that could drift.
    var const_h = const_hessian and scales.hess_max_unit == Int64(1)

    # Scratch, in cell units, then converted to the Int64 slots the caller
    # owns. The count plane is not allocated on the constant-hessian arm at
    # all, which is where the traffic bound above comes from.
    var cell_slots = row_count + blocks * cells
    if not const_h:
        cell_slots += blocks * cells
    var cell_bytes = 4 if bits == HIST_BITS_16 else 8
    var wanted = (cell_slots * cell_bytes + 7) // 8
    if len(qscratch) < wanted:
        qscratch.resize(wanted, Int64(0))

    var group = plan.group_width
    if bits == HIST_BITS_16:
        if group >= 16:
            _accumulate_packed_at[DType.int32, HIST_BITS_16, 16](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )
        elif group >= 8:
            _accumulate_packed_at[DType.int32, HIST_BITS_16, 8](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )
        elif group >= 4:
            _accumulate_packed_at[DType.int32, HIST_BITS_16, 4](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )
        elif group >= 2:
            _accumulate_packed_at[DType.int32, HIST_BITS_16, 2](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )
        else:
            _accumulate_packed_at[DType.int32, HIST_BITS_16, 1](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )
    else:
        if group >= 16:
            _accumulate_packed_at[DType.int64, HIST_BITS_32, 16](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )
        elif group >= 8:
            _accumulate_packed_at[DType.int64, HIST_BITS_32, 8](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )
        elif group >= 4:
            _accumulate_packed_at[DType.int64, HIST_BITS_32, 4](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )
        elif group >= 2:
            _accumulate_packed_at[DType.int64, HIST_BITS_32, 2](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )
        else:
            _accumulate_packed_at[DType.int64, HIST_BITS_32, 1](
                out._gh, out._count, qscratch, data, qgrad, qhess,
                rows, row_start, row_count, features, plan, blocks, block_rows,
                cells, n_active, scales, const_h, settings,
            )

    return QuantBuildReport(
        MODE_QUANTIZED,
        QUANT_REASON_OK,
        scales.rule,
        bits,
        blocks,
        plan.group_width,
        n_active * row_count,
        const_h,
        scales.copy(),
    )


def _accumulate_packed_at[
    CELL: DType, HIST_BITS: Int, GROUP: Int
](
    mut out_gh: List[Float64],
    mut out_count: List[Int],
    mut scratch: List[Int64],
    data: BinnedMatrix,
    qgrad: List[Int64],
    qhess: List[Int64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    features: List[Int],
    plan: AccumulationPlan,
    blocks: Int,
    block_rows: Int,
    cells: Int,
    n_active: Int,
    scales: QuantScales,
    const_h: Bool,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """The packed accumulation at one cell width and one interleave width.

    Structurally `histogram._accumulate_subset_blocked_at` with LightGBM's
    packed cell in place of three Float64 planes: a `(block, group)` dispatch
    unit, a private histogram per block over the active slots, and an
    ascending fold. Four differences, and each of them is a simplification the
    packing or the integers pay for.

    **One kernel, not four.** The Float64 subset builder instantiates a
    blocked arm and an unblocked arm, and each has a gathered and a
    non-gathered row loop. Here there is one arm at `blocks == 1` and at
    `blocks > 1` alike, because a fold of one block is a copy and costs
    nothing to express, and the rows always come through the packed gather,
    because a row's value has to be quantized and packed once rather than once
    per feature -- the same hoist `histogram._accumulate_replica` documents.

    **One read-modify-write per (row, slot).** The gradient and the hessian
    share a cell, so the inner step is `cell += packed` and nothing else. On
    the constant-hessian arm that is the entire inner loop; on the general arm
    there is a second store into the count plane, which exists because
    `Histogram` promises a per-bin count that LightGBM's histogram does not
    have and its data partition cannot supply.

    **The fold folds in place, and it is exact.** Blocks 1 upward are summed
    into block 0's slice, which is then unpacked and dequantized. In Float64
    that would be a summation order worth arguing about; in packed integers it
    is the exact sum whatever order it runs in, so block 0 is simply the
    accumulator and `MOJOTREES_CPU_ROW_BLOCKS` cannot move a cell.

    **Unpacking is arithmetic, not addressing.** `cell >> HIST_BITS` is the
    gradient (an arithmetic shift, so the sign survives) and `cell & mask` is
    the hessian or the count. The low field is nonnegative by construction and
    bounded by `histogram_bits_for_node`, which is exactly what makes the
    shift recover the high field rather than a high field plus a borrow.

    The tail group owning fewer than `GROUP` slots and the SIMD-lane slot
    arrays are as in the Float64 kernel. The single scratch pointer is
    load-bearing rather than stylistic: the packed rows and the private
    histograms live in one `List[Int64]`, and two pointers carrying one origin
    into a parallel closure is a thing Mojo refuses to compile, so the region
    offsets are folded into every index instead.
    """
    var n_rows = data.n_rows
    var n_bins = data.n_bins
    var n_sub = row_count
    var use_all = len(features) == 0
    var n_groups = plan.group_count
    var hist_off = n_sub
    var count_off = n_sub + blocks * cells

    var ghp = out_gh.unsafe_ptr()
    var cp = out_count.unsafe_ptr()
    var sp = scratch.unsafe_ptr().unsafe_bitcast[Scalar[CELL]]()
    var qg = qgrad.unsafe_ptr()
    var qh = qhess.unsafe_ptr()
    var rows_p = rows.unsafe_ptr().unsafe_offset(row_start)
    var bins_all_p = data.bins.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    comptime W = 4 * simd_width_of[CELL]()
    comptime SHIFT = Scalar[CELL](HIST_BITS)
    comptime MASK = (Scalar[CELL](1) << SHIFT) - Scalar[CELL](1)
    comptime ONE = Scalar[CELL](1)
    comptime ZERO = Scalar[CELL](0)

    # The gather, and the pack. `(g << HIST_BITS) | (h & MASK)` is
    # `DenseBin::ConstructHistogramIntInner`'s `gradient_packed`, and the
    # constant-hessian arm's `| 1` is theirs too. Elementwise over disjoint
    # ascending blocks, so the buffer is identical at every task count.
    def gather_packed(start: Int, end: Int) {imm}:
        for i in range(start, end):
            var r = rows_p.unsafe_load(i)
            var g = qg.unsafe_load(r).cast[CELL]()
            var low = ONE if const_h else (qh.unsafe_load(r).cast[CELL]() & MASK)
            sp.unsafe_store(i, (g << SHIFT) | low)

    dispatch_rows_with(settings, gather_packed, n_sub, 3 * n_sub)

    def accumulate_units(u_start: Int, u_end: Int) {imm}:
        for u in range(u_start, u_end):
            var blk = u // n_groups
            var grp = u - blk * n_groups
            var r0 = blk * block_rows
            var r1 = r0 + block_rows
            if r1 > n_sub:
                r1 = n_sub
            var slot0 = grp * GROUP
            var owned = n_active - slot0
            if owned > GROUP:
                owned = GROUP
            # `base` indexes the block's private packed slice by active slot,
            # `cbase` the count slice, `col` the binned matrix by feature id.
            var base = SIMD[DType.int, GROUP](0)
            var cbase = SIMD[DType.int, GROUP](0)
            var col = SIMD[DType.int, GROUP](0)
            comptime for k in range(GROUP):
                if k < owned:
                    var f = (
                        (slot0 + k) if use_all
                        else feat_p.unsafe_load(slot0 + k)
                    )
                    base[k] = hist_off + blk * cells + (slot0 + k) * n_bins
                    cbase[k] = count_off + blk * cells + (slot0 + k) * n_bins
                    col[k] = f * n_rows

            # Zeroing stays fused into the pass that fills the slice.
            comptime for k in range(GROUP):
                if k < owned:
                    var z0 = Int(base[k])
                    var zb = 0
                    while zb + W <= n_bins:
                        sp.unsafe_store(z0 + zb, SIMD[CELL, W](0))
                        zb += W
                    while zb < n_bins:
                        sp.unsafe_store(z0 + zb, ZERO)
                        zb += 1
                    if not const_h:
                        var zc = Int(cbase[k])
                        var cb = 0
                        while cb + W <= n_bins:
                            sp.unsafe_store(zc + cb, SIMD[CELL, W](0))
                            cb += W
                        while cb < n_bins:
                            sp.unsafe_store(zc + cb, ZERO)
                            cb += 1

            if const_h:
                # LightGBM's `USE_HESSIAN=false` arm, cell for cell: one add,
                # and the low field of the accumulated cell is the row count.
                for i_row in range(r0, r1):
                    var r = rows_p.unsafe_load(i_row)
                    var packed = sp.unsafe_load(i_row)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var c = Int(base[k]) + Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            sp.unsafe_store(c, sp.unsafe_load(c) + packed)
            else:
                for i_row in range(r0, r1):
                    var r = rows_p.unsafe_load(i_row)
                    var packed = sp.unsafe_load(i_row)
                    comptime for k in range(GROUP):
                        if k < owned:
                            var bin = Int(
                                bins_all_p.unsafe_load(Int(col[k]) + r)
                            )
                            var c = Int(base[k]) + bin
                            sp.unsafe_store(c, sp.unsafe_load(c) + packed)
                            var cc = Int(cbase[k]) + bin
                            sp.unsafe_store(cc, sp.unsafe_load(cc) + ONE)

    var acc_ops = plan.block_ops if plan.blocked() else plan.active_ops
    dispatch_feature_ranges_with(
        settings, accumulate_units, blocks * n_groups, acc_ops
    )

    var g_inv = 1.0 / scales.grad_units
    var h_inv = 1.0 / scales.hess_units

    # The fold, the unpack and the dequantization, in one pass per active
    # slot. Blocks 1 upward are summed into block 0's slice and block 0 is
    # then read out. Exact at every block count and in every order.
    def fold_slots(s_start: Int, s_end: Int) {imm}:
        for j in range(s_start, s_end):
            var f = j if use_all else feat_p.unsafe_load(j)
            var p0 = hist_off + j * n_bins
            var c0 = count_off + j * n_bins
            for blk in range(1, blocks):
                var po = p0 + blk * cells
                var i = 0
                while i + W <= n_bins:
                    sp.unsafe_store(
                        p0 + i,
                        sp.unsafe_load[width=W](p0 + i)
                        + sp.unsafe_load[width=W](po + i),
                    )
                    i += W
                while i < n_bins:
                    sp.unsafe_store(
                        p0 + i, sp.unsafe_load(p0 + i) + sp.unsafe_load(po + i)
                    )
                    i += 1
                if not const_h:
                    var co = c0 + blk * cells
                    var b2 = 0
                    while b2 + W <= n_bins:
                        sp.unsafe_store(
                            c0 + b2,
                            sp.unsafe_load[width=W](c0 + b2)
                            + sp.unsafe_load[width=W](co + b2),
                        )
                        b2 += W
                    while b2 < n_bins:
                        sp.unsafe_store(
                            c0 + b2,
                            sp.unsafe_load(c0 + b2) + sp.unsafe_load(co + b2),
                        )
                        b2 += 1

            var out0 = f * n_bins
            for b in range(n_bins):
                var cell = sp.unsafe_load(p0 + b)
                # Arithmetic shift for the signed high field, mask for the
                # nonnegative low one. `histogram_bits_for_node` is what makes
                # both exact: no carry ever crossed the boundary.
                var gq = (cell >> SHIFT).cast[DType.int64]()
                var lo = (cell & MASK).cast[DType.int64]()
                # Written as a branch and not a conditional expression: the
                # count plane is not allocated at all on the elided arm, so
                # `sp.unsafe_load(c0 + b)` there would be a read past the end
                # of the caller's scratch even if its value were discarded.
                var cq = lo
                if not const_h:
                    cq = sp.unsafe_load(c0 + b).cast[DType.int64]()
                # One 16-byte store into the interleaved output cell where
                # there were two scalar stores into planes a whole histogram
                # apart. Same two Float64, same dequantization.
                ghp.unsafe_store(
                    2 * (out0 + b),
                    SIMD[DType.float64, 2](
                        Float64(gq) * g_inv, Float64(lo) * h_inv
                    ),
                )
                cp.unsafe_store(out0 + b, Int(cq))

    var fold_ops = plan.fold_ops if plan.blocked() else 3 * cells
    dispatch_feature_ranges_with(settings, fold_slots, n_active, fold_ops)


def build_histogram_subset_maybe_quantized(
    mut out: Histogram,
    mut pairs: List[Float64],
    mut qscratch: List[Int64],
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    qgrad: List[Int64],
    qhess: List[Int64],
    rows: List[Int],
    row_start: Int,
    row_count: Int,
    decision: QuantDecision,
    features: List[Int] = [],
    const_hessian: Bool = False,
    rounding: Int = ROUND_NEAREST,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> QuantBuildReport:
    """The one entry point a node-level caller needs: quantized when the
    round decided to be, Float64 when it did not.

    **Off is not a special case of on, it is the untouched builder.** When
    `decision` is `MODE_FLOAT` this calls
    `histogram.build_histogram_subset_into_scratch`, unchanged, with the same
    arguments it would have received had this function never existed, and
    returns a float report. Nothing about the Float64 path is rewritten,
    re-planned, or re-dispatched, so "bit-identical to today when off" is a
    property of the control flow rather than of an argument about numerics,
    and `tests/test_cpu_quantized_grad.mojo` establishes it with `to_bits()`
    anyway.

    `decision` comes from `decide_cpu_histogram` once per round and is not
    re-derived per node, which is the same rule `decide` states: the bound is
    checked against the worst node once, so two nodes of one tree cannot end
    up on different lattices and sibling subtraction stays exact.

    `qgrad`/`qhess` may be empty when the decision is float; they are not read.
    """
    if not decision.is_quantized():
        build_histogram_subset_into_scratch(
            out, pairs, data, grad, hess, rows, row_start, row_count,
            features, const_hessian, settings,
        )
        return QuantBuildReport.floating(decision.reason)
    return build_histogram_subset_quantized_into_scratch(
        out, qscratch, data, qgrad, qhess, rows, row_start, row_count,
        decision.scales, features, const_hessian, rounding, settings,
    )


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
