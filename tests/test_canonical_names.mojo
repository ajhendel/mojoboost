"""The canonical parameter names, on the parameter-string surface.

docs/PARAMETER_NAMING.md gives one canonical name per parameter, always a
name LightGBM, XGBoost, CatBoost or scikit-learn already uses, with every
other vendor's spelling accepted as an alias. This file is the Mojo half of
the proof; `python/tests/test_vendor_dialects.py` is the other half, on the
estimator surface, and `tests/test_canonical_sampling_names.mojo` covers
the sampling table in `sampling.mojo`.

What is asserted, and why it is worth asserting. The value of an alias
layer is not that the names are pretty, it is that a configuration written
for one library reaches this one *without being retyped*, because retyping
is where two "identical" configurations quietly stop being identical. So
the test is not "the key parses"; it is that a `TrainConfig` built from one
vendor's spelling is field-for-field the same `TrainConfig` built from
another's. A key that parsed and then landed in the wrong slot would pass
the weaker test and fail this one.

Values are compared exactly, floats included: nothing here is computed, so
a tolerance would only hide a slot mixup.

Run with `bash tools/run_tests.sh cpu test_canonical_names`.
"""

from std.testing import assert_equal, assert_raises, assert_true, TestSuite

from mojotrees.boosting import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    GAMMA,
    L1,
    MAPE,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
)
from mojotrees.device import CPU_DEVICE, GPU_DEVICE
from mojotrees.growth_policy import (
    GROW_DEPTHWISE,
    GROW_LEAFWISE,
    GROW_OBLIVIOUS,
)
from mojotrees.params import (
    MULTICLASS,
    TrainConfig,
    objective_from_name,
    params_names_mojo_api_only,
    parse_params,
)


def _assert_same_config(a: TrainConfig, b: TrainConfig, what: String) raises:
    """Every field of a `TrainConfig`, compared exactly.

    Written out field by field rather than through an equality operator so
    that a failure names the parameter that landed in the wrong slot, which
    is the whole failure mode an alias table has.
    """
    assert_equal(a.objective, b.objective, String("objective: ", what))
    assert_equal(a.n_classes, b.n_classes, String("num_class: ", what))
    assert_equal(a.max_bin, b.max_bin, String("max_bin: ", what))
    assert_equal(a.alpha, b.alpha, String("alpha: ", what))
    assert_equal(a.device, b.device, String("device: ", what))
    assert_equal(a.use_missing, b.use_missing, String("use_missing: ", what))
    assert_equal(
        a.booster.n_estimators,
        b.booster.n_estimators,
        String("n_estimators: ", what),
    )
    assert_equal(
        a.booster.learning_rate,
        b.booster.learning_rate,
        String("learning_rate: ", what),
    )
    assert_equal(
        a.booster.tree.num_leaves,
        b.booster.tree.num_leaves,
        String("max_leaves: ", what),
    )
    assert_equal(
        a.booster.tree.max_depth,
        b.booster.tree.max_depth,
        String("max_depth: ", what),
    )
    assert_equal(
        a.booster.tree.min_data_in_leaf,
        b.booster.tree.min_data_in_leaf,
        String("min_child_samples: ", what),
    )
    assert_equal(
        a.booster.tree.min_child_hess,
        b.booster.tree.min_child_hess,
        String("min_child_weight: ", what),
    )
    assert_equal(
        a.booster.tree.lambda_l1,
        b.booster.tree.lambda_l1,
        String("reg_alpha: ", what),
    )
    assert_equal(
        a.booster.tree.lambda_reg,
        b.booster.tree.lambda_reg,
        String("reg_lambda: ", what),
    )
    assert_equal(
        a.booster.tree.feature_fraction,
        b.booster.tree.feature_fraction,
        String("colsample_bytree: ", what),
    )
    assert_equal(
        a.booster.tree.feature_fraction_bynode,
        b.booster.tree.feature_fraction_bynode,
        String("colsample_bynode: ", what),
    )
    assert_equal(
        a.booster.tree.feature_fraction_seed,
        b.booster.tree.feature_fraction_seed,
        String("feature_fraction_seed: ", what),
    )
    assert_equal(
        a.booster.tree.grow_policy,
        b.booster.tree.grow_policy,
        String("grow_policy: ", what),
    )
    assert_equal(
        a.booster.tree.extra.min_gain_to_split,
        b.booster.tree.extra.min_gain_to_split,
        String("min_split_gain: ", what),
    )
    assert_equal(
        a.booster.tree.extra.extra_seed,
        b.booster.tree.extra.extra_seed,
        String("extra_seed: ", what),
    )


# One configuration, spelled four ways. Every number is the same in all
# four; only the keys and the value words differ. Nothing is left at its
# default, because a defaulted parameter would compare equal whether or not
# its alias resolved.
comptime _LIGHTGBM = String(
    "objective=regression num_iterations=40 learning_rate=0.05"
    " num_leaves=15 max_depth=4 min_data_in_leaf=7"
    " min_sum_hessian_in_leaf=0.02 lambda_l1=0.3 lambda_l2=0.7"
    " feature_fraction=0.9 feature_fraction_bynode=0.85"
    " min_gain_to_split=0.01 max_bin=63 boosting_type=gbdt"
    " grow_policy=leafwise device=cpu seed=11"
)

comptime _XGBOOST = String(
    "objective=reg:squarederror n_estimators=40 eta=0.05"
    " max_leaves=15 max_depth=4 min_child_samples=7"
    " min_child_weight=0.02 reg_alpha=0.3 reg_lambda=0.7"
    " colsample_bytree=0.9 colsample_bynode=0.85"
    " gamma=0.01 max_bin=63 booster=gbtree"
    " grow_policy=lossguide device=cpu random_state=11"
)

comptime _CATBOOST = String(
    "loss_function=RMSE iterations=40 learning_rate=0.05"
    " max_leaves=15 depth=4 min_data_in_leaf=7"
    " min_child_weight=0.02 reg_alpha=0.3 l2_leaf_reg=0.7"
    " rsm=0.9 colsample_bynode=0.85"
    " min_split_gain=0.01 border_count=63 boosting_type=Plain"
    " grow_policy=Lossguide task_type=CPU random_seed=11"
)

comptime _SKLEARN = String(
    "loss=squared_error max_iter=40 learning_rate=0.05"
    " max_leaf_nodes=15 max_depth=4 min_samples_leaf=7"
    " min_child_weight=0.02 reg_alpha=0.3 l2_regularization=0.7"
    " colsample_bytree=0.9 max_features=0.85"
    " min_split_gain=0.01 max_bins=63 boosting_type=gbdt"
    " grow_policy=lossguide device=cpu random_state=11"
)


def test_every_vendor_spelling_builds_one_configuration() raises:
    """The lane's whole claim, on this surface: four dialects, one
    `TrainConfig`."""
    var lgbm = parse_params(_LIGHTGBM)
    _assert_same_config(lgbm, parse_params(_XGBOOST), String("xgboost"))
    _assert_same_config(lgbm, parse_params(_CATBOOST), String("catboost"))
    _assert_same_config(lgbm, parse_params(_SKLEARN), String("sklearn"))


def test_the_shared_configuration_is_not_the_default() raises:
    """The guard on the test above.

    Four dialects that all failed to resolve would produce four default
    configurations and compare equal. This asserts the configuration they
    share is not the default one, so a resolution failure shows up as a
    difference rather than as agreement on nothing.
    """
    var configured = parse_params(_LIGHTGBM)
    var default = TrainConfig()
    assert_true(
        configured.booster.n_estimators != default.booster.n_estimators
    )
    assert_true(
        configured.booster.tree.num_leaves != default.booster.tree.num_leaves
    )
    assert_true(configured.max_bin != default.max_bin)
    assert_true(
        configured.booster.tree.feature_fraction
        != default.booster.tree.feature_fraction
    )


def test_objective_takes_every_vendor_loss_name() raises:
    """One objective, every vendor's word for it, case insensitive."""
    assert_equal(
        objective_from_name(String("reg:squarederror")), SQUARED_ERROR
    )
    assert_equal(objective_from_name(String("RMSE")), SQUARED_ERROR)
    assert_equal(objective_from_name(String("rmse")), SQUARED_ERROR)
    assert_equal(objective_from_name(String("squared_error")), SQUARED_ERROR)
    assert_equal(objective_from_name(String("regression")), SQUARED_ERROR)
    assert_equal(objective_from_name(String("binary:logistic")), BINARY_LOGISTIC)
    assert_equal(objective_from_name(String("Logloss")), BINARY_LOGISTIC)
    assert_equal(objective_from_name(String("binary")), BINARY_LOGISTIC)
    assert_equal(objective_from_name(String("count:poisson")), POISSON)
    assert_equal(objective_from_name(String("reg:absoluteerror")), L1)
    assert_equal(objective_from_name(String("MAE")), L1)
    assert_equal(objective_from_name(String("reg:gamma")), GAMMA)
    assert_equal(objective_from_name(String("reg:tweedie")), TWEEDIE)
    assert_equal(objective_from_name(String("reg:quantileerror")), QUANTILE)
    assert_equal(
        objective_from_name(String("reg:absolutepercentageerror")), MAPE
    )
    assert_equal(objective_from_name(String("CrossEntropy")), CROSS_ENTROPY)
    assert_equal(objective_from_name(String("multi:softmax")), MULTICLASS)
    assert_equal(objective_from_name(String("multi:softprob")), MULTICLASS)
    assert_equal(objective_from_name(String("MultiClass")), MULTICLASS)


def test_a_vendor_loss_that_is_a_different_curve_is_refused_by_name() raises:
    """XGBoost's `reg:pseudohubererror` is not a spelling of `huber`, and
    saying it is would be the exact mistake this layer exists to stop."""
    with assert_raises(contains="pseudo-Huber"):
        _ = objective_from_name(String("reg:pseudohubererror"))
    # CatBoost's YetiRank is its own pairwise loss, not LambdaRank spelled
    # differently. That claim is unchanged and is what the `assert_raises`
    # below is for: the name must not RESOLVE, because resolving it to
    # LAMBDARANK would silently fit a different loss.
    #
    # What changed is which refusal it gets. YetiRank used to fall through to
    # "unknown objective"; it is now the reserved code YETI_RANK=15 with a
    # registry entry, so it lands on the "real thing, not implemented" path
    # instead. That is a strictly better message and the old assertion was
    # matching the fall-through -- which means it would have kept passing if
    # the name had been dropped from the registry altogether.
    #
    # Asserted three ways so it still discriminates. It must refuse; the
    # refusal must be the reserved one rather than the unknown one; and it
    # must name its own trainer, which is what distinguishes "we know exactly
    # what this is and have not connected it" from "we have never heard of
    # it". `tests/test_objective_reserved_codes.mojo` pins the same refusal
    # from the registry side.
    with assert_raises(contains="not implemented"):
        _ = objective_from_name(String("YetiRank"))
    with assert_raises(contains="train_catboost_ranker"):
        _ = objective_from_name(String("YetiRank"))
    # And it is still not lambdarank: the connected ranking loss is named as
    # the alternative, not as the answer.
    with assert_raises(contains="lambdarank"):
        _ = objective_from_name(String("YetiRank"))


def test_grow_policy_takes_all_three_canonical_values() raises:
    """`lossguide | depthwise | symmetrictree`, with `leafwise`,
    `oblivious` and `symmetric` as aliases, all case insensitive."""
    assert_equal(
        parse_params(String("grow_policy=lossguide")).booster.tree.grow_policy,
        GROW_LEAFWISE,
    )
    assert_equal(
        parse_params(String("grow_policy=leafwise")).booster.tree.grow_policy,
        GROW_LEAFWISE,
    )
    assert_equal(
        parse_params(String("grow_policy=Lossguide")).booster.tree.grow_policy,
        GROW_LEAFWISE,
    )
    assert_equal(
        parse_params(String("grow_policy=Depthwise")).booster.tree.grow_policy,
        GROW_DEPTHWISE,
    )
    for spelling in [
        String("symmetrictree"),
        String("SymmetricTree"),
        String("oblivious"),
        String("symmetric"),
    ]:
        assert_equal(
            parse_params(
                String("grow_policy=", spelling)
            ).booster.tree.grow_policy,
            GROW_OBLIVIOUS,
            String("grow_policy=", spelling),
        )


def test_boosting_type_is_one_key_with_six_values() raises:
    """`plain` is `gbdt`; `ordered` sets the bundle it names; the three that
    need a parameter bundle a string cannot carry say so; and an unknown
    value is still unknown."""
    # `plain` and `gbdt` both describe what a parameter string already
    # configures, so both parse and change nothing.
    _ = parse_params(String("boosting_type=gbdt"))
    _ = parse_params(String("boosting_type=Plain"))
    _ = parse_params(String("boosting=plain"))
    _ = parse_params(String("booster=gbtree"))
    assert_true(
        not parse_params(String("boosting_type=gbdt")).booster.ordered.enabled
    )
    # `ordered` stopped being a refusal on 2026-08-16. The value is the one
    # thing a caller can check that distinguishes "parsed" from "honored":
    # `_check_boosting_type` accepted it before this change too, and the
    # difference is that `BoosterParams.ordered` is now set from it.
    assert_true(
        parse_params(
            String("boosting_type=Ordered")
        ).booster.ordered.enabled,
        "boosting_type=Ordered must set BoosterParams.ordered.enabled",
    )
    for value in [String("dart"), String("goss"), String("rf")]:
        with assert_raises(contains="Mojo API only"):
            _ = parse_params(String("boosting_type=", value))
        assert_true(
            params_names_mojo_api_only(String("boosting_type=", value)),
            String("boosting_type=", value, " should read as Mojo-API-only"),
        )
    # `ordered` must still NOT read as Mojo-API-only, and the reason inverted:
    # it is not a feature reached the wrong way because this surface reaches
    # it the right way.
    assert_true(
        not params_names_mojo_api_only(String("boosting_type=ordered"))
    )
    with assert_raises(contains="unknown boosting_type"):
        _ = parse_params(String("boosting_type=gblinear"))


def test_ordered_boosting_knobs_reach_the_bundle() raises:
    """The four scalars ordered boosting takes, each read onto
    `BoosterParams.ordered`, and each refused when it would never be read.

    The refusals are the half that matters. A knob parsed onto a disabled
    bundle is a number the user set and the trainer never looks at, which is
    the failure this whole surface's refuse-rather-than-ignore rule exists
    for; and the GPU refusal is there because `train_gpu` reads
    `BoosterParams.ordered` nowhere at all.
    """
    var config = parse_params(
        String(
            "boosting_type=ordered permutation_count=3"
            " fold_len_multiplier=4.0 fold_permutation_block=64"
            " ordered_seed=7"
        )
    )
    assert_true(config.booster.ordered.enabled)
    assert_equal(config.booster.ordered.permutation_count, 3)
    assert_equal(config.booster.ordered.fold_len_multiplier, 4.0)
    assert_equal(config.booster.ordered.permutation_block_size, 64)
    assert_equal(config.booster.ordered.seed, 7)
    # Order does not matter: the knobs set fields and the type flips `enabled`.
    assert_equal(
        parse_params(
            String("fold_len_multiplier=3.0 boosting_type=ordered")
        ).booster.ordered.fold_len_multiplier,
        3.0,
    )
    with assert_raises(contains="read only when boosting_type=ordered"):
        _ = parse_params(String("permutation_count=3"))
    with assert_raises(contains="trains on the CPU only"):
        _ = parse_params(String("boosting_type=ordered device=gpu"))
    with assert_raises(contains="fold_len_multiplier"):
        _ = parse_params(
            String("boosting_type=ordered fold_len_multiplier=1.0")
        )


def test_device_takes_every_vendor_spelling() raises:
    """`device` is canonical; `device_type` is LightGBM's, `task_type`
    CatBoost's, and `tree_method` XGBoost's older algorithm-and-device
    switch."""
    assert_equal(parse_params(String("device=gpu")).device, GPU_DEVICE)
    assert_equal(parse_params(String("device_type=gpu")).device, GPU_DEVICE)
    assert_equal(parse_params(String("task_type=GPU")).device, GPU_DEVICE)
    assert_equal(parse_params(String("task_type=CPU")).device, CPU_DEVICE)
    assert_equal(parse_params(String("tree_method=hist")).device, CPU_DEVICE)
    assert_equal(
        parse_params(String("tree_method=gpu_hist")).device, GPU_DEVICE
    )
    # A different split search is not a spelling of a device.
    with assert_raises(contains="different split search"):
        _ = parse_params(String("tree_method=exact"))
    with assert_raises(contains="different split search"):
        _ = parse_params(String("tree_method=approx"))


def test_random_state_fills_only_the_seeds_the_string_did_not_name() raises:
    """A global seed sets every seed this surface carries, and a seed named
    outright wins whichever order the two appear in."""
    var seeded = parse_params(String("random_state=42"))
    assert_equal(seeded.booster.tree.feature_fraction_seed, 42)
    assert_equal(seeded.booster.tree.extra.extra_seed, 42)
    var mixed = parse_params(String("random_state=42 extra_seed=99"))
    assert_equal(mixed.booster.tree.feature_fraction_seed, 42)
    assert_equal(mixed.booster.tree.extra.extra_seed, 99)
    # Order does not matter: the fill waits until the whole string is read.
    var reversed = parse_params(String("extra_seed=99 random_state=42"))
    assert_equal(reversed.booster.tree.extra.extra_seed, 99)
    assert_equal(reversed.booster.tree.feature_fraction_seed, 42)
    # LightGBM's, XGBoost's and CatBoost's spellings are one parameter.
    for spelling in [
        String("seed"),
        String("random_seed"),
        String("random_state"),
    ]:
        assert_equal(
            parse_params(
                String(spelling, "=7")
            ).booster.tree.feature_fraction_seed,
            7,
            spelling,
        )
    # An untouched string keeps the stock seeds.
    var stock = TrainConfig()
    assert_equal(
        parse_params(String("")).booster.tree.feature_fraction_seed,
        stock.booster.tree.feature_fraction_seed,
    )


def test_n_jobs_accepts_use_the_machine_and_refuses_a_count() raises:
    """A count that would not be honored is refused by name; the values
    that mean "use the machine" describe what already happens."""
    for spelling in [
        String("n_jobs"),
        String("num_threads"),
        String("nthread"),
        String("thread_count"),
    ]:
        _ = parse_params(String(spelling, "=-1"))
        _ = parse_params(String(spelling, "=0"))
        with assert_raises(contains="MOJOTREES_NUM_WORKERS"):
            _ = parse_params(String(spelling, "=4"))


def test_verbose_accepts_silence_and_refuses_a_log() raises:
    """Nothing on this surface writes a training log, so a request for
    silence is honored by doing nothing and a request for output is refused
    rather than producing none."""
    _ = parse_params(String("verbose=0"))
    _ = parse_params(String("verbose=-1"))
    _ = parse_params(String("verbosity=0"))
    _ = parse_params(String("logging_level=Silent"))
    with assert_raises(contains="training log"):
        _ = parse_params(String("verbose=1"))
    with assert_raises(contains="training log"):
        _ = parse_params(String("logging_level=Verbose"))


def test_catboost_only_names_state_or_refuse() raises:
    """`score_function` and `max_ctr_complexity`: the value that names what
    mojotrees already does is accepted, and everything else is refused with
    what it would take.

    UNRUN, as is every change in this file's `score_function` case.
    `score_function` is now honored on this surface rather than accepted
    only at `L2`: it writes `ExtraTreeParams.score_function`, which
    `tree._search` passes into `split.find_best_split`. Both CatBoost
    spellings of both values parse, because the parameter string folds case
    once (`_lower_ascii`) before `parse_score_function`, which takes
    canonical lowercase. An unknown value is still refused rather than
    resolved to the default.
    """
    _ = parse_params(String("score_function=L2"))
    _ = parse_params(String("score_function=l2"))
    _ = parse_params(String("score_function=Cosine"))
    _ = parse_params(String("score_function=cosine"))
    with assert_raises(contains="score_function"):
        _ = parse_params(String("score_function=NewtonCosine"))
    # `max_ctr_complexity=1` is ACCEPTED since the CTR-dataset lane: 1 is the
    # value CatBoost itself resolves to for any fit under 200 iterations. The
    # key used to be refused outright, and the assertion here used to match
    # `contains="CTR"` against the old message's literal `(CTR)`. The reword
    # dropped that spelling while keeping the refusal, so the test failed for
    # prose rather than for behavior -- the same shape as fb356d9.
    #
    # Matched on the key name instead, which the refuse-by-name rule
    # obliges any refusal for this key to carry, and paired with the
    # accepted arm. The pair is what discriminates: matching the key alone
    # would also be satisfied by the `unknown parameter 'max_ctr_complexity'`
    # fall-through, and the `=1` arm is what rules that out.
    _ = parse_params(String("max_ctr_complexity=1"))
    with assert_raises(contains="max_ctr_complexity"):
        _ = parse_params(String("max_ctr_complexity=4"))
    # `random_strength` and `leaf_estimation_iterations` already had this
    # shape and keep it.
    _ = parse_params(String("random_strength=0.0"))
    _ = parse_params(String("leaf_estimation_iterations=1"))
    with assert_raises(contains="leaf_estimation_iterations"):
        _ = parse_params(String("leaf_estimation_iterations=5"))


def test_a_vendor_alias_of_a_mojo_api_parameter_says_so() raises:
    """`subsample` and `rsm`-adjacent names that need the Mojo API get the
    sentence about the feature, not "unknown parameter": what the user needs
    to be told is about the parameter, not about the spelling."""
    for spelling in [
        String("subsample=0.8"),
        String("subsample_freq=1"),
        String("bagging_fraction=0.8"),
        String("one_hot_max_size=5"),
        String("cat_features=0"),
        String("monotonic_cst=1"),
        String("interaction_cst=0"),
        String("od_wait=10"),
        String("n_iter_no_change=10"),
        String("min_data_in_bin=3"),
        String("rate_drop=0.2"),
        String("eval_metric=rmse"),
    ]:
        with assert_raises(contains="Mojo API only"):
            _ = parse_params(spelling)
        assert_true(params_names_mojo_api_only(spelling), spelling)


def test_unknown_parameter_is_still_unknown() raises:
    """The alias table widens what is accepted; it does not make the parser
    tolerant. A typo is still an error."""
    with assert_raises(contains="unknown parameter"):
        _ = parse_params(String("colsample_bytre=0.9"))
    with assert_raises(contains="unknown parameter"):
        _ = parse_params(String("max_leafs=15"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
