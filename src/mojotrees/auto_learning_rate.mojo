"""CatBoost's automatic `learning_rate`, derived from the data.

CatBoost does not use its constant `learning_rate` default of 0.03 on most
runs. When the user leaves the rate unset it fits one from the training row
count and the iteration count, using coefficients regressed offline by the
CatBoost authors. That is why a benchmark harness has to pin the rate on a
CatBoost arm to get a reproducible number, and it is why "our default 0.1
against CatBoost's default" is not a comparison of defaults at all. This
module reproduces the derivation so the comparison can be made honestly.

Everything in this module is inert until a caller enables it, and nothing
here decides when that is. `AutoLearningRateParams` defaults to disabled and
`BoosterParams.default()` is still `learning_rate = 0.1`.

**Who enables it, as of 2026-08-16.** Under `grow_policy=oblivious`,
CatBoost's symmetric tree, this is ON by default, because the standing rule
is that CatBoost mode mirrors CatBoost exactly and CatBoost derives the rate
whenever the user set none of the four gated parameters. Under `lossguide`
and `depthwise` it is OFF by default, because those mirror LightGBM and
LightGBM has no automatic learning rate. `auto_learning_rate=true|false`
overrides the mode default in either direction. The decision is made where
the parameters are parsed and their provenance is still visible --
`params.parse_params` for the string, CLI and C ABI surfaces, and
`_Base._auto_learning_rate_knobs` in python/mojotrees/sklearn.py for the
Python one -- never here.

**Reachability.** All four surfaces reach it. The CLI and the C ABI go
through `TrainConfig.resolved_learning_rate`; the Python extension calls the
free function `resolve_learning_rate` below from `_apply_auto_learning_rate`
in bindings/_mojotrees.mojo, which nine of the fifteen `_parse_params` call
sites hand a row count, an iteration count and an objective, and the other
six refuse by name.

Source, verified
----------------

All line numbers are CatBoost `master`, August 2026.

- `catboost/libs/train_lib/options_helper.cpp`, `UpdateLearningRate`
  (269-288): the gate.
- Same file, `TAutoLRParamsGuesser` (177-266): coefficient table, the
  `ETargetType` collapse (181-194), and the formula (252-262).
- Same file, `SetDataDependentDefaults` (403-435): the one call site, at 418,
  after `UpdateUseBestModel` and `AdjustBoostFromAverageDefaultValue` have
  already resolved the two flags it reads.
- Same file, local `static double Round(double, int)` (15-18).
- `catboost/private/libs/options/boosting_options.cpp`: the static defaults
  this replaces, `learning_rate` 0.03 (line 10) and `iterations` 1000 (13).

It is **not** in `catboost/private/libs/options/catboost_options.cpp`; that
file's `SetNotSpecifiedOptionsToDefaults` handles only defaults that do not
depend on the data. See `docs/design/CATBOOST_CATALOG.md` section A12 for the
full transcription and for the two things this lane could not establish.

The formula
-----------

With coefficients `(A, B, C, D)` for the selected row, `N` train rows and `T`
iterations, `GetLearningRate` computes

    custom  = exp(C * log(T)    + D)
    default = exp(C * log(1000) + D)
    base    = exp(A * log(N)    + B)
    lr      = round6(min(base * custom / default, 0.5))

`D` cancels analytically -- the struct comment in CatBoost writes the closed
form as `exp(B + A log N + C log T - C log 1000)` -- but it does not cancel
in floating point, and neither does the order of the multiply and divide.
This module reproduces the code's ordering rather than the comment's algebra,
so a value printed here can be compared with a value printed by CatBoost
without an argument about rounding.

`round6` is CatBoost's local `Round`: `round(x * 1e6) / 1e6` with C's
`round`, which breaks ties **away from zero**, not to even. The 0.5 cap is
applied before the rounding. There is no lower cap.

When it fires
-------------

CatBoost requires all four of `learning_rate`, `leaf_estimation_method`,
`leaf_estimation_iterations` and `l2_leaf_reg` to be unset, and requires the
`(target type, task type, use_best_model, boost_from_average)` key to exist
in a 20-row table. Setting `l2_leaf_reg` alone silently pins the rate back to
the constant 0.03. `AutoLearningRateParams` carries those three "was it set"
flags so a caller can reproduce the coupling exactly; it is the caller's job
to set them truthfully, because a parameter struct cannot tell a default
apart from a value that happens to equal the default.

The table has no `MultiClass` row with `boost_from_average = true`, and no
row at all for any loss outside `{Logloss, MultiLogloss, MultiCrossEntropy,
MultiClass, RMSE}`. Both cases leave the rate alone.

Determinism
-----------

Scalar Float64 arithmetic on a single thread: no reduction, no parallel
region, no dependence on `MOJOTREES_NUM_WORKERS`. Two `log` and three `exp`
calls go to the platform libm, which may differ by an ulp between machines;
the round to six decimals absorbs that except for a value sitting exactly on
a 5 in the seventh decimal, which is a measure-zero input and is the only
cross-machine hazard here. Optional narrowing to float32 (what CatBoost
actually stores, `boosting_options.h:26` is a `TOption<float>`) absorbs it
further.
"""

from std.math import ceil, exp, floor, log

from .objective_registry import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    MULTICLASS,
    SQUARED_ERROR,
    L1,
    MAPE,
    QUANTILE,
)


# CatBoost's `ETargetType` (`options_helper.cpp:124-129`). The loss function
# is collapsed to one of these before the coefficient table is consulted;
# `UNKNOWN` means no row exists and the rate is left alone.
comptime AUTO_LR_TARGET_UNKNOWN = 0
comptime AUTO_LR_TARGET_LOGLOSS = 1
comptime AUTO_LR_TARGET_MULTICLASS = 2
comptime AUTO_LR_TARGET_RMSE = 3

# CatBoost's `ETaskType`. The coefficients differ between the two, so a GPU
# run gets a different rate from a CPU run on the same data. That is
# CatBoost's design, not a mistake: the fitted rows for GPU Logloss without
# `boost_from_average` even have a negative dataset-size exponent.
comptime AUTO_LR_TASK_CPU = 0
comptime AUTO_LR_TASK_GPU = 1

# `boosting_options.cpp:10` and `:13`. Recorded so a caller can say what the
# derivation is replacing; mojotrees' own defaults are 0.1 and 100 and this
# module never touches them.
comptime CATBOOST_CONSTANT_LEARNING_RATE = 0.03
comptime CATBOOST_DEFAULT_ITERATIONS = 1000

# `Min(..., 0.5)` and `Round(..., /*precision=*/6)`, `options_helper.cpp:261`.
comptime AUTO_LR_CAP = 0.5
comptime AUTO_LR_ROUND_PRECISION = 6


@fieldwise_init
struct AutoLearningRateCoefficients(Copyable, Movable):
    """One row of CatBoost's `TLearningRateCoefficients`
    (`options_helper.cpp:116-122`), named `A, B, C, D` there.

    `iter_count_const` (`D`) is algebraically dead: it appears in both the
    numerator and the denominator of the iteration correction. It is kept
    because it is not dead in floating point and because dropping it would
    make this table impossible to check against the source by eye.
    """

    var dataset_size_coeff: Float64
    """`A`, the exponent on the train row count."""

    var dataset_size_const: Float64
    """`B`, the intercept."""

    var iter_count_coeff: Float64
    """`C`, the exponent on the iteration count. Negative in every row of
    the table: more iterations, smaller steps."""

    var iter_count_const: Float64
    """`D`. Cancels; see the struct docstring."""


@fieldwise_init
struct AutoLearningRateLookup(Copyable, Movable):
    """The result of a coefficient-table lookup.

    `found` false is CatBoost's `NeedToUpdate` returning false
    (`options_helper.cpp:246-249`), which leaves the learning rate at
    whatever it already was.
    """

    var found: Bool
    var coefficients: AutoLearningRateCoefficients


def auto_lr_target_type(objective: Int) -> Int:
    """Collapse a mojotrees objective code to a CatBoost `ETargetType`.

    **This mapping is ours, not CatBoost's**, because the two objective sets
    do not coincide. CatBoost's `GetTargetType`
    (`options_helper.cpp:181-194`) maps `Logloss`, `MultiLogloss` and
    `MultiCrossEntropy` to Logloss, `MultiClass` to MultiClass, `RMSE` to
    RMSE, and everything else to Unknown.

    - `SQUARED_ERROR` -> RMSE. Same loss up to a square root, which does not
      move the minimizer or the Newton step.
    - `BINARY_LOGISTIC` -> Logloss. Same loss.
    - `CROSS_ENTROPY` -> Logloss. **The one deviation.** CatBoost's
      `GetTargetType` lists `MultiCrossEntropy` but not the single-output
      `CrossEntropy`, so strict parity would return Unknown here. Mapping it
      to Logloss is a judgement that the fitted coefficients transfer,
      because soft labels change the target but not the derivative's shape.
      A caller that wants strict parity should pass `AUTO_LR_TARGET_UNKNOWN`
      itself rather than call this.
    - Everything else -> Unknown, matching CatBoost, which leaves 0.03.
    """
    if objective == SQUARED_ERROR:
        return AUTO_LR_TARGET_RMSE
    if objective == BINARY_LOGISTIC:
        return AUTO_LR_TARGET_LOGLOSS
    if objective == CROSS_ENTROPY:
        return AUTO_LR_TARGET_LOGLOSS
    if objective == MULTICLASS:
        return AUTO_LR_TARGET_MULTICLASS
    return AUTO_LR_TARGET_UNKNOWN


def catboost_boost_from_average_default(objective: Int) -> Bool:
    """CatBoost's resolved `boost_from_average` for an unset user option.

    `AdjustBoostFromAverageDefaultValue` (`options_helper.cpp:353-374`) sets
    it true for `RMSE`, `MAE`, `Quantile`, `MAPE`, `MultiQuantile`,
    `MultiRMSE` and `MultiRMSEWithMissingValues` on a single host with no
    model continuation, then forces it false if a baseline column is present
    on either pool. Every other loss keeps the static default of false
    (`boosting_options.cpp:17`) -- including `Logloss` and `MultiClass`.

    Only the four losses mojotrees has are listed here. The three `Multi*`
    ones on CatBoost's list have no mojotrees equivalent. The baseline
    override is the caller's business: mojotrees' `init_score` is the
    analogue, and a caller that supplies one should pass `False` explicitly
    rather than call this.
    """
    if objective == SQUARED_ERROR:
        return True
    if objective == L1:
        return True
    if objective == QUANTILE:
        return True
    if objective == MAPE:
        return True
    return False


def auto_lr_coefficients(
    target_type: Int,
    task_type: Int,
    use_best_model: Bool,
    boost_from_average: Bool,
) -> AutoLearningRateLookup:
    """The coefficient row for a key, or `found = False` if there is none.

    Transcribed from `TAutoLRParamsGuesser::TAutoLRParamsGuesser`
    (`options_helper.cpp:197-244`) in source order: four Logloss rows, two
    MultiClass rows, four RMSE rows, per task type. The MultiClass rows exist
    only for `boost_from_average = false`, which is why a MultiClass run with
    it forced on keeps the constant rate.
    """
    var absent = AutoLearningRateLookup(
        False, AutoLearningRateCoefficients(0.0, 0.0, 0.0, 0.0)
    )

    if task_type == AUTO_LR_TASK_CPU:
        if target_type == AUTO_LR_TARGET_LOGLOSS:
            if use_best_model and boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.246, -5.127, -0.451, 0.978),
                )
            if not use_best_model and boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.408, -7.299, -0.928, 2.701),
                )
            if use_best_model and not boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.247, -5.158, -0.435, 0.934),
                )
            return AutoLearningRateLookup(
                True,
                AutoLearningRateCoefficients(0.427, -7.525, -0.917, 2.63),
            )
        if target_type == AUTO_LR_TARGET_MULTICLASS:
            if boost_from_average:
                return absent.copy()
            if use_best_model:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.02, -2.364, -0.382, 0.924),
                )
            return AutoLearningRateLookup(
                True,
                AutoLearningRateCoefficients(0.051, -2.889, -0.845, 2.928),
            )
        if target_type == AUTO_LR_TARGET_RMSE:
            if use_best_model and boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.157, -4.062, -0.61, 1.557),
                )
            if not use_best_model and boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.158, -4.287, -0.813, 2.571),
                )
            if use_best_model and not boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.189, -4.383, -0.623, 1.439),
                )
            return AutoLearningRateLookup(
                True,
                AutoLearningRateCoefficients(0.178, -4.473, -0.76, 2.133),
            )
        return absent.copy()

    if task_type == AUTO_LR_TASK_GPU:
        if target_type == AUTO_LR_TARGET_LOGLOSS:
            if use_best_model and boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.04, -3.226, -0.488, 0.758),
                )
            if not use_best_model and boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.427, -7.316, -0.907, 2.354),
                )
            if use_best_model and not boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(-0.085, -2.055, -0.414, 0.427),
                )
            return AutoLearningRateLookup(
                True,
                AutoLearningRateCoefficients(-0.055, -3.01, -0.896, 2.366),
            )
        if target_type == AUTO_LR_TARGET_MULTICLASS:
            if boost_from_average:
                return absent.copy()
            if use_best_model:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.101, -2.95, -0.437, 1.136),
                )
            return AutoLearningRateLookup(
                True,
                AutoLearningRateCoefficients(0.204, -4.144, -0.833, 2.889),
            )
        if target_type == AUTO_LR_TARGET_RMSE:
            if use_best_model and boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.108, -3.525, -0.285, 0.058),
                )
            if not use_best_model and boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.131, -4.114, -0.597, 1.693),
                )
            if use_best_model and not boost_from_average:
                return AutoLearningRateLookup(
                    True,
                    AutoLearningRateCoefficients(0.051, -3.001, -0.449, 0.859),
                )
            return AutoLearningRateLookup(
                True,
                AutoLearningRateCoefficients(0.047, -3.034, -0.591, 1.554),
            )
        return absent.copy()

    return absent.copy()


def round_to_precision(value: Float64, precision: Int) -> Float64:
    """CatBoost's local `Round` (`options_helper.cpp:15-18`).

    `round(value * 10**precision) / 10**precision` with C's `round`, which
    breaks ties **away from zero**. Written out rather than delegating to
    Mojo's `round`, whose tie rule is not the one this needs to reproduce.
    """
    var multiplier = 1.0
    for _ in range(precision):
        multiplier *= 10.0
    var scaled = value * multiplier
    var nearest = floor(scaled + 0.5) if scaled >= 0.0 else ceil(scaled - 0.5)
    return nearest / multiplier


def narrow_to_float32(value: Float64) -> Float64:
    """The float32 value CatBoost actually stores.

    `TBoostingOptions::LearningRate` is a `TOption<float>`
    (`boosting_options.h:26`), so the `double` returned by `GetLearningRate`
    is narrowed on assignment at `options_helper.cpp:284` and everything
    downstream uses the narrowed value. Reproducing that keeps a printed
    comparison from disagreeing in the ninth decimal for no reason.
    """
    return Float64(Float32(value))


def catboost_auto_learning_rate(
    coefficients: AutoLearningRateCoefficients,
    n_iterations: Int,
    n_rows: Int,
) raises -> Float64:
    """`TAutoLRParamsGuesser::GetLearningRate` (`options_helper.cpp:252-262`).

    The three `exp` calls, the multiply-then-divide, the 0.5 cap and the
    round to six decimals are in the source's order. Not narrowed to float32
    here; `resolve_learning_rate` does that.

    Raises on a non-positive row or iteration count, where CatBoost would
    take `log(0)` and hand `exp` a negative infinity. CatBoost does not guard
    this because `iterations` is validated elsewhere and an empty train pool
    is rejected before defaults are resolved; we would rather say so than
    return a zero rate.
    """
    if n_rows <= 0:
        raise Error(
            "auto learning rate needs a positive train row count, got ",
            n_rows,
        )
    if n_iterations <= 0:
        raise Error(
            "auto learning rate needs a positive iteration count, got ",
            n_iterations,
        )

    var log_iterations = log(Float64(n_iterations))
    var log_default_iterations = log(Float64(CATBOOST_DEFAULT_ITERATIONS))
    var log_rows = log(Float64(n_rows))

    var custom_iteration_constant = exp(
        coefficients.iter_count_coeff * log_iterations
        + coefficients.iter_count_const
    )
    var default_iteration_constant = exp(
        coefficients.iter_count_coeff * log_default_iterations
        + coefficients.iter_count_const
    )
    var default_learning_rate = exp(
        coefficients.dataset_size_coeff * log_rows
        + coefficients.dataset_size_const
    )

    var scaled = (
        default_learning_rate
        * custom_iteration_constant
        / default_iteration_constant
    )
    var capped = scaled if scaled < AUTO_LR_CAP else AUTO_LR_CAP
    return round_to_precision(capped, AUTO_LR_ROUND_PRECISION)


struct AutoLearningRateParams(Copyable, Movable):
    """Everything `UpdateLearningRate` reads that is not the data itself.

    Defaults to **disabled**. A default-constructed value leaves the learning
    rate exactly as it found it, so adding one of these to a config struct
    changes no behavior until a caller flips `enabled`.

    The three `*_set` fields reproduce CatBoost's gate
    (`options_helper.cpp:276-281`): the derivation fires only if the user set
    none of `leaf_estimation_method`, `leaf_estimation_iterations` and
    `l2_leaf_reg`. A parameter struct cannot see whether a field was assigned
    or left at its default, so the caller has to say. `parse_params` knows
    (it sees the keys) and sets them; a Mojo API caller who does not care can
    leave them false.

    mojotrees has no `leaf_estimation_method` -- Newton only, see catalog A6
    -- so `leaf_estimation_method_set` exists to let a caller that emulates
    CatBoost more fully close the gate, and is false for us.
    """

    var enabled: Bool
    """Off by default. Nothing in this module runs unless this is true."""

    var task_type: Int
    """`AUTO_LR_TASK_CPU` or `AUTO_LR_TASK_GPU`. Selects a different half of
    the coefficient table; CatBoost genuinely derives a different rate for a
    GPU run on the same data."""

    var use_best_model: Bool
    """CatBoost's resolved `use_best_model`. `UpdateUseBestModel`
    (`options_helper.cpp:100-113`) forces it false when there is no eval set,
    so a plain fit is false."""

    var boost_from_average_is_set: Bool
    """True if the caller is overriding `boost_from_average` below rather
    than letting `catboost_boost_from_average_default` derive it from the
    objective, mirroring the `IsSet()` early return at
    `options_helper.cpp:359`."""

    var boost_from_average: Bool
    """Read only when `boost_from_average_is_set`."""

    var leaf_estimation_method_set: Bool
    var leaf_estimation_iterations_set: Bool
    var l2_leaf_reg_set: Bool

    var narrow_result_to_float32: Bool
    """Reproduce the `TOption<float>` narrowing at
    `options_helper.cpp:284`. True by default because that is what CatBoost's
    trainer sees."""

    def __init__(out self):
        self.enabled = False
        self.task_type = AUTO_LR_TASK_CPU
        self.use_best_model = False
        self.boost_from_average_is_set = False
        self.boost_from_average = False
        self.leaf_estimation_method_set = False
        self.leaf_estimation_iterations_set = False
        self.l2_leaf_reg_set = False
        self.narrow_result_to_float32 = True

    @staticmethod
    def disabled() -> Self:
        """The default: leave every learning rate alone."""
        return Self()

    @staticmethod
    def catboost_defaults(task_type: Int = AUTO_LR_TASK_CPU) -> Self:
        """Enabled, with the flags a plain CatBoost `fit` with no eval set
        and no other parameters would resolve to.

        `use_best_model` false because there is no eval set; the three
        `*_set` gates false because nothing was set; `boost_from_average`
        derived from the objective at resolve time.
        """
        var params = Self()
        params.enabled = True
        params.task_type = task_type
        return params^

    def resolved_boost_from_average(self, objective: Int) -> Bool:
        """The `boost_from_average` this will use for an objective."""
        if self.boost_from_average_is_set:
            return self.boost_from_average
        return catboost_boost_from_average_default(objective)

    def fires(self, objective: Int) -> Bool:
        """Whether the derivation would replace the rate for this objective.

        Both halves of CatBoost's condition: the four-unset gate
        (`options_helper.cpp:276-281`) and `NeedToUpdate`
        (`options_helper.cpp:246-249`). Independent of the data, so a caller
        can ask before it has any.
        """
        if not self.enabled:
            return False
        if self.leaf_estimation_method_set:
            return False
        if self.leaf_estimation_iterations_set:
            return False
        if self.l2_leaf_reg_set:
            return False
        var lookup = auto_lr_coefficients(
            auto_lr_target_type(objective),
            self.task_type,
            self.use_best_model,
            self.resolved_boost_from_average(objective),
        )
        return lookup.found


def resolve_learning_rate(
    params: AutoLearningRateParams,
    objective: Int,
    n_iterations: Int,
    n_rows: Int,
    learning_rate: Float64,
) raises -> Float64:
    """`learning_rate` unchanged, or CatBoost's derived rate in its place.

    This is `UpdateLearningRate` (`options_helper.cpp:269-288`) with the
    "was the learning rate set" half of the gate handed to the caller: in
    CatBoost that test is `learningRate.NotSet()`, and a caller here that
    passes an explicit rate should not be enabling this in the first place.
    `parse_params` enforces that by refusing `auto_learning_rate=true`
    alongside an explicit `learning_rate=`.

    Returns `learning_rate` untouched whenever `params.fires(objective)` is
    false, which includes the disabled default. Classification: this is a
    **trade behind a switch** -- it moves every leaf value when it is on, and
    is inert when it is off.
    """
    if not params.fires(objective):
        return learning_rate

    var lookup = auto_lr_coefficients(
        auto_lr_target_type(objective),
        params.task_type,
        params.use_best_model,
        params.resolved_boost_from_average(objective),
    )
    var rate = catboost_auto_learning_rate(
        lookup.coefficients, n_iterations, n_rows
    )
    if params.narrow_result_to_float32:
        return narrow_to_float32(rate)
    return rate
