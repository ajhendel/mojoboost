"""CatBoost's data-dependent `learning_rate`
(src/mojotrees/auto_learning_rate.mojo, docs/design/CATBOOST_CATALOG.md A12).

Every expected number here was computed independently from the transcribed
coefficients using the formula as CatBoost writes it
(`options_helper.cpp:252-262`), not read out of a mojotrees run:

    custom  = exp(C * log(T)    + D)
    default = exp(C * log(1000) + D)
    base    = exp(A * log(N)    + B)
    lr      = round6(min(base * custom / default, 0.5))

The point of the first block is that the derivation is inert: a
default-constructed `AutoLearningRateParams` leaves every rate exactly as it
found it, and so does a default `TrainConfig`. Nothing in mojotrees changes
until a caller asks.
"""

from std.testing import assert_equal, assert_false, assert_raises, assert_true
from std.testing import TestSuite

from mojotrees.auto_learning_rate import (
    AUTO_LR_CAP,
    AUTO_LR_TARGET_LOGLOSS,
    AUTO_LR_TARGET_MULTICLASS,
    AUTO_LR_TARGET_RMSE,
    AUTO_LR_TARGET_UNKNOWN,
    AUTO_LR_TASK_CPU,
    AUTO_LR_TASK_GPU,
    AutoLearningRateParams,
    auto_lr_coefficients,
    auto_lr_target_type,
    catboost_auto_learning_rate,
    catboost_boost_from_average_default,
    narrow_to_float32,
    resolve_learning_rate,
    round_to_precision,
)
from mojotrees.objective_registry import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    GAMMA,
    LAMBDARANK,
    MAPE,
    MULTICLASS,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    L1,
)
from mojotrees.params import parse_params


def close(a: Float64, b: Float64) -> Bool:
    # Six-decimal values narrowed to float32: float32 carries ~7.2 decimal
    # digits, so the narrowing perturbs the eighth significant figure.
    return abs(a - b) < 1e-9


# ---------------------------------------------------------------------------
# Inert by default
# ---------------------------------------------------------------------------


def test_default_params_change_nothing() raises:
    var off = AutoLearningRateParams()
    assert_false(off.enabled)
    assert_false(off.fires(SQUARED_ERROR))
    assert_false(off.fires(BINARY_LOGISTIC))
    assert_false(off.fires(MULTICLASS))
    # The rate comes back byte for byte, for every objective and every shape.
    assert_true(close(resolve_learning_rate(off, SQUARED_ERROR, 100, 1000, 0.1), 0.1))
    assert_true(
        close(resolve_learning_rate(off, BINARY_LOGISTIC, 5000, 2, 0.037), 0.037)
    )
    assert_false(AutoLearningRateParams.disabled().enabled)


def test_default_parse_is_untouched() raises:
    # mojotrees' own default learning rate is 0.1 and stays 0.1: this lane
    # changed no existing default.
    var config = parse_params("")
    assert_false(config.auto_learning_rate.enabled)
    assert_true(close(config.booster.learning_rate, 0.1))
    assert_true(close(config.resolved_learning_rate(1000), 0.1))
    assert_true(close(config.resolved_learning_rate(1000000), 0.1))

    var explicit = parse_params("objective=binary learning_rate=0.05")
    assert_false(explicit.auto_learning_rate.enabled)
    assert_true(close(explicit.resolved_learning_rate(250000), 0.05))


# ---------------------------------------------------------------------------
# The formula
# ---------------------------------------------------------------------------


def test_logloss_cpu_plain_fit() raises:
    # (Logloss, CPU, use_best_model=false, boost_from_average=false):
    # A=0.427 B=-7.525 C=-0.917 D=2.63. This is the row a binary benchmark
    # arm with no eval set lands on.
    var found = auto_lr_coefficients(
        AUTO_LR_TARGET_LOGLOSS, AUTO_LR_TASK_CPU, False, False
    )
    assert_true(found.found)
    assert_true(close(found.coefficients.dataset_size_coeff, 0.427))
    assert_true(close(found.coefficients.dataset_size_const, -7.525))
    assert_true(close(found.coefficients.iter_count_coeff, -0.917))
    assert_true(close(found.coefficients.iter_count_const, 2.63))

    # T = 1000 is CatBoost's own default, where the iteration correction is
    # exactly 1 (numerator and denominator are the same expression), so this
    # value is exp(0.427*log(1e5) - 7.525) rounded to six decimals.
    var rate = catboost_auto_learning_rate(found.coefficients, 1000, 100000)
    assert_true(close(rate, 0.07361))


def test_iteration_correction_raises_the_rate_on_short_runs() raises:
    # C is negative in every row, so fewer iterations means a larger step.
    # 100 iterations on 100k rows overshoots the 0.5 cap for this row.
    var found = auto_lr_coefficients(
        AUTO_LR_TARGET_LOGLOSS, AUTO_LR_TASK_CPU, False, False
    )
    var short_run = catboost_auto_learning_rate(found.coefficients, 100, 100000)
    assert_true(close(short_run, AUTO_LR_CAP))
    var long_run = catboost_auto_learning_rate(found.coefficients, 1000, 100000)
    assert_true(long_run < short_run)
    var longer = catboost_auto_learning_rate(found.coefficients, 10000, 100000)
    assert_true(longer < long_run)


def test_rmse_cpu_plain_fit() raises:
    # (RMSE, CPU, use_best_model=false, boost_from_average=true):
    # A=0.158 B=-4.287 C=-0.813 D=2.571. Regression with no eval set lands
    # here, because AdjustBoostFromAverageDefaultValue turns
    # boost_from_average on for RMSE.
    var found = auto_lr_coefficients(
        AUTO_LR_TARGET_RMSE, AUTO_LR_TASK_CPU, False, True
    )
    assert_true(found.found)
    assert_true(close(catboost_auto_learning_rate(found.coefficients, 1000, 100000), 0.084758))
    assert_true(close(catboost_auto_learning_rate(found.coefficients, 100, 1000), 0.266183))


def test_multiclass_and_gpu_rows() raises:
    var mc = auto_lr_coefficients(
        AUTO_LR_TARGET_MULTICLASS, AUTO_LR_TASK_CPU, False, False
    )
    assert_true(mc.found)
    assert_true(close(catboost_auto_learning_rate(mc.coefficients, 1000, 50000), 0.096599))

    # The GPU table is genuinely different: (Logloss, GPU, ubm=true,
    # bfa=false) has a NEGATIVE dataset-size exponent, A = -0.085.
    var gpu = auto_lr_coefficients(
        AUTO_LR_TARGET_LOGLOSS, AUTO_LR_TASK_GPU, True, False
    )
    assert_true(gpu.found)
    assert_true(gpu.coefficients.dataset_size_coeff < 0.0)
    assert_true(close(catboost_auto_learning_rate(gpu.coefficients, 1000, 100000), 0.048142))

    # Same key, other task type: a different rate, not the same one.
    var cpu = auto_lr_coefficients(
        AUTO_LR_TARGET_LOGLOSS, AUTO_LR_TASK_CPU, True, False
    )
    assert_true(cpu.found)
    assert_false(
        close(
            catboost_auto_learning_rate(cpu.coefficients, 1000, 100000),
            catboost_auto_learning_rate(gpu.coefficients, 1000, 100000),
        )
    )


def test_cap_is_applied_before_rounding_and_there_is_no_floor() raises:
    var found = auto_lr_coefficients(
        AUTO_LR_TARGET_RMSE, AUTO_LR_TASK_CPU, False, True
    )
    # One iteration on a hundred rows blows past 0.5 and is clamped exactly.
    assert_true(close(catboost_auto_learning_rate(found.coefficients, 1, 100), AUTO_LR_CAP))
    # Nothing clamps from below: a huge iteration count drives it toward 0.
    var tiny = catboost_auto_learning_rate(found.coefficients, 1000000, 1000)
    assert_true(tiny < 0.01)
    assert_true(tiny >= 0.0)


def test_row_and_iteration_counts_must_be_positive() raises:
    var found = auto_lr_coefficients(
        AUTO_LR_TARGET_RMSE, AUTO_LR_TASK_CPU, False, True
    )
    with assert_raises():
        _ = catboost_auto_learning_rate(found.coefficients, 1000, 0)
    with assert_raises():
        _ = catboost_auto_learning_rate(found.coefficients, 0, 1000)
    with assert_raises():
        _ = catboost_auto_learning_rate(found.coefficients, -1, 1000)


def test_rounding_breaks_ties_away_from_zero() raises:
    # CatBoost's local Round is C's round(), not banker's rounding. A half at
    # the seventh decimal goes up, not to even.
    assert_true(close(round_to_precision(0.0000005, 6), 0.000001))
    assert_true(close(round_to_precision(-0.0000005, 6), -0.000001))
    assert_true(close(round_to_precision(1.2345675, 6), 1.234568))
    assert_true(close(round_to_precision(0.12345649, 6), 0.123456))


def test_float32_narrowing_is_what_catboost_stores() raises:
    # TOption<float>, boosting_options.h:26. The narrowed value is not the
    # six-decimal double, and a comparison against CatBoost's printout wants
    # the narrowed one.
    var narrowed = narrow_to_float32(0.07361)
    assert_true(abs(narrowed - 0.07361) < 1e-8)
    assert_true(close(narrowed, Float64(Float32(0.07361))))


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def test_absent_table_rows() raises:
    # MultiClass has no boost_from_average=true row on either task type.
    assert_false(
        auto_lr_coefficients(
            AUTO_LR_TARGET_MULTICLASS, AUTO_LR_TASK_CPU, False, True
        ).found
    )
    assert_false(
        auto_lr_coefficients(
            AUTO_LR_TARGET_MULTICLASS, AUTO_LR_TASK_GPU, True, True
        ).found
    )
    # Unknown target type has no row at all.
    assert_false(
        auto_lr_coefficients(
            AUTO_LR_TARGET_UNKNOWN, AUTO_LR_TASK_CPU, False, False
        ).found
    )
    # Every other combination exists: 4 Logloss + 2 MultiClass + 4 RMSE per
    # task type, 20 rows.
    var count = 0
    for task in [AUTO_LR_TASK_CPU, AUTO_LR_TASK_GPU]:
        for target in [
            AUTO_LR_TARGET_LOGLOSS,
            AUTO_LR_TARGET_MULTICLASS,
            AUTO_LR_TARGET_RMSE,
        ]:
            for ubm in [False, True]:
                for bfa in [False, True]:
                    if auto_lr_coefficients(target, task, ubm, bfa).found:
                        count += 1
    assert_equal(count, 20)


def test_setting_any_of_the_three_options_pins_the_constant_rate() raises:
    # options_helper.cpp:276-281. This is the surprising coupling: a user who
    # sets l2_leaf_reg loses the automatic learning rate.
    var base = AutoLearningRateParams.catboost_defaults()
    assert_true(base.fires(SQUARED_ERROR))

    var with_l2 = base.copy()
    with_l2.l2_leaf_reg_set = True
    assert_false(with_l2.fires(SQUARED_ERROR))
    assert_true(close(resolve_learning_rate(with_l2, SQUARED_ERROR, 1000, 100000, 0.1), 0.1))

    var with_iters = base.copy()
    with_iters.leaf_estimation_iterations_set = True
    assert_false(with_iters.fires(SQUARED_ERROR))

    var with_method = base.copy()
    with_method.leaf_estimation_method_set = True
    assert_false(with_method.fires(SQUARED_ERROR))


def test_unsupported_objectives_keep_their_rate() raises:
    var params = AutoLearningRateParams.catboost_defaults()
    for objective in [POISSON, GAMMA, LAMBDARANK, L1, QUANTILE, MAPE]:
        assert_false(params.fires(objective))
        assert_true(
            close(resolve_learning_rate(params, objective, 1000, 100000, 0.1), 0.1)
        )
    # And the supported ones do fire.
    for objective in [SQUARED_ERROR, BINARY_LOGISTIC, CROSS_ENTROPY, MULTICLASS]:
        assert_true(params.fires(objective))


def test_target_type_mapping() raises:
    assert_equal(auto_lr_target_type(SQUARED_ERROR), AUTO_LR_TARGET_RMSE)
    assert_equal(auto_lr_target_type(BINARY_LOGISTIC), AUTO_LR_TARGET_LOGLOSS)
    # Ours, not CatBoost's: CatBoost's GetTargetType lists MultiCrossEntropy
    # but not the single-output CrossEntropy. Marked in the catalog.
    assert_equal(auto_lr_target_type(CROSS_ENTROPY), AUTO_LR_TARGET_LOGLOSS)
    assert_equal(auto_lr_target_type(MULTICLASS), AUTO_LR_TARGET_MULTICLASS)
    assert_equal(auto_lr_target_type(POISSON), AUTO_LR_TARGET_UNKNOWN)
    assert_equal(auto_lr_target_type(LAMBDARANK), AUTO_LR_TARGET_UNKNOWN)


def test_boost_from_average_default_follows_the_objective() raises:
    assert_true(catboost_boost_from_average_default(SQUARED_ERROR))
    assert_true(catboost_boost_from_average_default(L1))
    assert_true(catboost_boost_from_average_default(QUANTILE))
    assert_true(catboost_boost_from_average_default(MAPE))
    # Logloss and MultiClass are NOT on CatBoost's list; they keep the static
    # default of false (boosting_options.cpp:17).
    assert_false(catboost_boost_from_average_default(BINARY_LOGISTIC))
    assert_false(catboost_boost_from_average_default(MULTICLASS))
    assert_false(catboost_boost_from_average_default(POISSON))

    # An explicit override wins, mirroring the IsSet() early return.
    var params = AutoLearningRateParams.catboost_defaults()
    assert_true(params.resolved_boost_from_average(SQUARED_ERROR))
    params.boost_from_average_is_set = True
    params.boost_from_average = False
    assert_false(params.resolved_boost_from_average(SQUARED_ERROR))
    # And forcing it on for MultiClass walks off the table entirely.
    params.boost_from_average = True
    assert_false(params.fires(MULTICLASS))


# ---------------------------------------------------------------------------
# The parameter-string surface
# ---------------------------------------------------------------------------


def test_parameter_string_opt_in() raises:
    var config = parse_params("objective=binary num_iterations=1000 auto_learning_rate=true")
    assert_true(config.auto_learning_rate.enabled)
    assert_equal(config.auto_learning_rate.task_type, AUTO_LR_TASK_CPU)
    assert_false(config.auto_learning_rate.use_best_model)
    # Binary with no eval set: Logloss / CPU / ubm=false / bfa=false, and
    # narrowed to float32 the way CatBoost stores it.
    assert_true(
        close(config.resolved_learning_rate(100000), narrow_to_float32(0.07361))
    )
    # The stored rate is untouched; only the resolved one moves.
    assert_true(close(config.booster.learning_rate, 0.1))
    # It is data-dependent, so a different row count gives a different rate.
    assert_false(
        close(
            config.resolved_learning_rate(100000),
            config.resolved_learning_rate(1000000),
        )
    )


def test_parameter_string_regression_and_false() raises:
    var reg = parse_params("num_iterations=1000 auto_learning_rate=true")
    assert_true(reg.auto_learning_rate.enabled)
    # RMSE / CPU / ubm=false / bfa=TRUE, because the adjuster turns
    # boost_from_average on for RMSE.
    assert_true(reg.auto_learning_rate.resolved_boost_from_average(reg.objective))
    # Narrowed, because `AutoLearningRateParams.narrow_result_to_float32` is
    # on by default: 0.084758 is not a float32, and CatBoost stores a float.
    # The gap is real and about 1.4e-9, which is why this compares against
    # the narrowed constant rather than the six-decimal one.
    assert_true(
        close(reg.resolved_learning_rate(100000), narrow_to_float32(0.084758))
    )
    assert_false(close(reg.resolved_learning_rate(100000), 0.084758))

    var off = parse_params("auto_learning_rate=false")
    assert_false(off.auto_learning_rate.enabled)
    assert_true(close(off.resolved_learning_rate(100000), 0.1))


def test_parameter_string_refuses_the_contradictions() raises:
    with assert_raises():
        _ = parse_params("auto_learning_rate=true learning_rate=0.05")
    with assert_raises():
        _ = parse_params("learning_rate=0.05 auto_learning_rate=true")
    with assert_raises():
        _ = parse_params("auto_learning_rate=true lambda_l2=5")
    with assert_raises():
        _ = parse_params("auto_learning_rate=true leaf_estimation_iterations=1")


def test_catboost_mode_turns_it_on_and_lossguide_does_not() raises:
    """The standing rule, as one test.

    `grow_policy=oblivious` is CatBoost's symmetric tree and mirrors
    CatBoost, which derives the rate whenever the user set none of the four
    gated parameters. Every other grow policy mirrors LightGBM, which has no
    such feature, so the default there is a flat rate.
    """
    var catboost_mode = parse_params("grow_policy=oblivious num_iterations=1000")
    assert_true(catboost_mode.auto_learning_rate.enabled)
    assert_true(
        close(
            catboost_mode.resolved_learning_rate(100000),
            narrow_to_float32(0.084758),
        )
    )

    # Our default. LightGBM has no automatic rate, so neither do we.
    var lossguide = parse_params("num_iterations=1000")
    assert_false(lossguide.auto_learning_rate.enabled)
    assert_true(close(lossguide.resolved_learning_rate(100000), 0.1))

    var depthwise = parse_params("grow_policy=depthwise num_iterations=1000")
    assert_false(depthwise.auto_learning_rate.enabled)


def test_catboost_mode_default_is_silent_on_a_closed_gate() raises:
    """Unset means the key was not named, a closed gate is not an error, and
    the rate the run falls back to is CatBoost's constant rather than ours.

    The three parameters CatBoost's gate reads close it when the string
    names them (`options_helper.cpp:277-280`), and CatBoost keeps its
    constant without saying so. The default here does the same: nothing was
    asked for, so there is nothing to refuse. An explicit
    `auto_learning_rate=true` beside the same key still raises, which is
    `test_parameter_string_refuses_the_contradictions` above.

    **This test asserted 0.1 on every closed-gate case and that was our
    number, not CatBoost's.** It was right until 2026-08-16 (`e3cfb47`), when
    `_apply_catboost_mode_defaults` started supplying CatBoost's own
    `learning_rate` of 0.03 (`boosting_options.cpp:10`, the only initializer
    and unconditional on task type, objective or iteration count) to any
    `grow_policy=oblivious` string that did not name the key. The standing rule
    is that CatBoost mode mirrors CatBoost and every other grow policy mirrors
    LightGBM, so the fallback under this policy is 0.03 and the 0.1 the old
    assertions pinned was the value the mode had not finished replacing.

    The literal 0.03 is deliberate rather than
    `auto_learning_rate.CATBOOST_CONSTANT_LEARNING_RATE`. The number has an
    external source, so a literal asserts against CatBoost; importing the
    symbol would only assert that this package agrees with itself, and no test
    anywhere else pins the constant.

    **The teeth are not in the number, and they were not in 0.1 either.** A
    closed gate has two failure modes and the constant only catches one. The
    interesting one is the defect this whole layer was built to end. The mode
    advertises a derived rate, a key silently closes the gate, and the run
    ships a constant with nothing anywhere saying it did. So every case below
    also asserts `auto_lr_note`, which is the record `e3cfb47` added for
    exactly that, and asserts it names the KEY that closed the gate rather
    than merely reporting that something did. Three keys, three different
    notes, and the order they are tested in is CatBoost's own short-circuit
    order, so a string that closes two is recorded against the one CatBoost
    would have stopped at.

    The other half is that a closed gate must not merely *differ from* the
    derived rate, it must be the constant, and an open gate at the same shape
    must still derive. Both are asserted against one another on strings that
    differ by a single key, so a regression that disabled the derivation
    everywhere would pass the closed-gate half and fail the open one.
    """
    var pinned = parse_params("grow_policy=oblivious learning_rate=0.03")
    assert_false(pinned.auto_learning_rate.enabled)
    assert_true(close(pinned.resolved_learning_rate(100000), 0.03))
    assert_equal(pinned.auto_lr_note, "auto_lr_skipped:learning_rate")

    # Naming the rate closes the gate even at the value the parser would
    # have produced anyway. This is the whole difference between "unset" and
    # "equal to the default", and it is the reading CatBoost's
    # `TOption::NotSet()` has (`option.h:80-85`). 0.1 is now neither the
    # mode's fallback nor the derived rate, which makes it the sharpest
    # possible probe of that distinction, because only a string that named
    # the key can produce it under this policy.
    var named_default = parse_params("grow_policy=oblivious learning_rate=0.1")
    assert_false(named_default.auto_learning_rate.enabled)
    assert_true(close(named_default.resolved_learning_rate(100000), 0.1))
    assert_equal(named_default.auto_lr_note, "auto_lr_skipped:learning_rate")

    # The A/B the constant alone cannot make. Two strings differing by one
    # key. The gate key closes the derivation and the run falls back to
    # CatBoost's 0.03, and without it the same string at the same row count
    # derives. `test_catboost_mode_turns_it_on_and_lossguide_does_not`
    # computes that expected rate from the coefficients independently.
    var open_gate = parse_params("grow_policy=oblivious num_iterations=1000")
    assert_true(open_gate.auto_learning_rate.enabled)
    assert_equal(open_gate.auto_lr_note, "auto_lr_gate_open")
    assert_true(
        close(
            open_gate.resolved_learning_rate(100000),
            narrow_to_float32(0.084758),
        )
    )

    var l2 = parse_params(
        "grow_policy=oblivious num_iterations=1000 lambda_l2=3"
    )
    assert_false(l2.auto_learning_rate.enabled)
    assert_equal(l2.auto_lr_note, "auto_lr_skipped:l2_leaf_reg")
    assert_true(close(l2.resolved_learning_rate(100000), 0.03))
    # The same run, one key apart, is not the derived rate. A derivation that
    # kept firing through a closed gate would satisfy the 0.03 check only by
    # coincidence and this by nothing.
    assert_false(
        close(
            l2.resolved_learning_rate(100000), narrow_to_float32(0.084758)
        )
    )

    # **The gate reads the KEY and not the value.** That is why both spellings
    # are here. `=1` and `=2` are the same act as far as the derivation is
    # concerned, and a gate that had been implemented as "the value differs
    # from the default" would let `=1` through and be caught here.
    #
    # `=2` is also the case that used to raise. `e3cfb47` replaced the blanket
    # refusal of `leaf_estimation_iterations > 1` on this surface with a
    # routing verdict, because a string that could not express a count above 1
    # could not express CatBoost mode's own per-objective default; the refusal
    # narrowed to `model.fit_multiclass`, the one trainer a string can reach
    # that reads the field nowhere. The narrowed refusal is asserted below, so
    # it is not lost with the value that used to trip it.
    var leaves = parse_params(
        "grow_policy=oblivious leaf_estimation_iterations=1"
    )
    assert_false(leaves.auto_learning_rate.enabled)
    assert_true(close(leaves.resolved_learning_rate(100000), 0.03))
    assert_equal(
        leaves.auto_lr_note, "auto_lr_skipped:leaf_estimation_iterations"
    )

    var leaves_two = parse_params(
        "grow_policy=oblivious leaf_estimation_iterations=2"
    )
    assert_equal(
        leaves_two.booster.tree.extra.leaf_estimation_iterations, 2
    )
    assert_false(leaves_two.auto_learning_rate.enabled)
    assert_true(close(leaves_two.resolved_learning_rate(100000), 0.03))
    assert_equal(
        leaves_two.auto_lr_note, "auto_lr_skipped:leaf_estimation_iterations"
    )

    # The gate is still not a way in. What closing it cannot do is smuggle a
    # count past the trainer that drops it, which is the claim the old
    # unconditional refusal was carrying and the only part of it that was
    # ever about this file.
    with assert_raises(contains="model.fit_multiclass"):
        _ = parse_params(
            "grow_policy=oblivious objective=multiclass num_class=3"
            " leaf_estimation_iterations=2"
        )

    # And an explicit false turns the derivation off, which is why "absent"
    # and "auto_learning_rate=false" are tracked separately. It turns off the
    # DERIVATION and not the mode, because `_apply_catboost_mode_defaults`
    # runs before this branch and does not read it, so the run trains at
    # CatBoost's 0.03 rather than LightGBM's 0.1. A reading of `false` as
    # "leave CatBoost mode" would show up right here.
    var off = parse_params("grow_policy=oblivious auto_learning_rate=false")
    assert_false(off.auto_learning_rate.enabled)
    assert_true(close(off.resolved_learning_rate(100000), 0.03))
    assert_equal(off.auto_lr_note, "auto_lr_off:auto_learning_rate=false")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
