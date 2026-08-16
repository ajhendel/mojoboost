"""Training configuration from a LightGBM style parameter string.

A parameter string is whitespace separated `key=value` pairs, the same
shape LightGBM accepts in its config files and in `LGBM_BoosterCreate`:

    objective=binary max_leaves=31 learning_rate=0.05 n_estimators=200

This is the only training surface the C ABI (`capi/`) and the CLI (`cli/`)
expose, which keeps both of them free of struct layouts that would have to
change whenever a hyperparameter is added.

Canonical names, and every vendor's spelling
--------------------------------------------
Keys are the canonical names of docs/PARAMETER_NAMING.md: one name per
parameter, always a name some vendor already uses, chosen for being the
clearest of the four. Every other vendor's spelling for the same parameter
is accepted as an alias, so a LightGBM, XGBoost, CatBoost or scikit-learn
configuration ports across without being retyped -- and retyping is where
two "identical" configurations quietly stop being identical.

The canonical name is what this module's error messages and the package's
documentation use. It is *not* what the struct fields, the LightGBM model
files, or `tools/check_parity.py` are named: those keep LightGBM's
spelling, because that is the wire and a wire does not get renamed for
readability. `n_estimators` therefore sets `BoosterParams.n_estimators`,
`max_leaves` sets `TreeParams.num_leaves`, and `min_child_weight` sets
`TreeParams.min_child_hess`.

**Values are case insensitive.** CatBoost writes `RMSE`, `SymmetricTree`
and `Bayesian`; XGBoost writes `reg:squarederror`; LightGBM writes
everything lowercase. All of them arrive here and are folded to ASCII
lowercase before any comparison (`_lower_ascii`), which is the one place
that fold happens on this surface. Keys are compared as written: no vendor
capitalizes a parameter name.

Intentional differences from LightGBM

- An unknown key is an error. LightGBM warns and ignores it, which silently
  drops typos; a training run is expensive enough that failing fast is
  worth more than tolerance.
- Values are validated here rather than clamped. LightGBM clamps several
  parameters into range.
- Only the parameters listed in `SUPPORTED_KEYS` are accepted. Bagging,
  GOSS, monotone, interaction, and categorical settings, and custom
  objectives, are reachable from the Mojo API only; naming one of those
  keys reports that it is unsupported here rather than ignoring it.
- A key that names a real LightGBM feature mojotrees has not implemented
  gets its own error saying what it would take, never "unknown parameter"
  and never silence. `objective=` values go through
  `_raise_if_unimplemented_objective`, tree options through
  `tree_parameters_extra.check_extra_option_supported`
  (`forcedsplits_filename`, `feature_pre_filter`), and `enable_bundle` through
  `efb.check_bundling_supported`, which accepts it for a CPU run and refuses
  it by name for a device that would ignore it.
"""

from .boosting import (
    BINARY_LOGISTIC,
    BoosterParams,
    CROSS_ENTROPY,
    DEFAULT_FAIR_C,
    DEFAULT_TWEEDIE_VARIANCE_POWER,
    FAIR,
    GAMMA,
    HUBER,
    L1,
    MAPE,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    TWEEDIE,
)
from .auto_learning_rate import (
    AUTO_LR_TASK_CPU,
    AUTO_LR_TASK_GPU,
    AutoLearningRateParams,
    resolve_learning_rate,
)
from .device import CPU_DEVICE, GPU_DEVICE, parse_device
from .objective_registry import MULTICLASS as _MULTICLASS
from .efb import check_bundling_supported
from .sampling import canonical_data_sample_strategy
from .validation import check_booster_ranges, check_max_bin
from .growth_policy import parse_grow_policy
from .tree_parameters_extra import (
    check_extra_option_supported,
    check_feature_pre_filter,
    parse_derivative_precision,
    parse_monotone_method,
    parse_score_function,
)

# `TrainConfig.objective` when the parameter string selects softmax
# multiclass, which `fit_multiclass` handles instead of `fit`. Defined once
# in objective_registry.mojo with the other codes (negative to stay out of
# the single-output space forever) and bound here under the name this
# module's callers import.
comptime MULTICLASS = _MULTICLASS

# Every key `parse_params` accepts, canonical names only, for error
# messages. An alias is deliberately absent here: a user who misspelled
# `colsample_bytre` should be shown the one name to reach for, not four.
comptime SUPPORTED_KEYS = String(
    "objective, num_class, n_estimators, learning_rate, max_leaves,"
    " min_child_samples, min_child_weight, reg_alpha, reg_lambda,"
    " max_depth, grow_policy, boosting_type, colsample_bytree,"
    " colsample_bynode, colsample_bylevel, feature_fraction_seed,"
    " min_split_gain,"
    " max_delta_step, path_smooth, extra_trees, extra_seed, random_strength,"
    " score_function, monotone_penalty, monotone_constraints_method,"
    " cegb_tradeoff,"
    " cegb_penalty_split, linear_tree, linear_lambda, enable_bundle,"
    " max_conflict_rate, feature_pre_filter,"
    " data_sample_strategy, max_bin, alpha, fair_c,"
    " tweedie_variance_power, device, random_state, n_jobs, verbose,"
    " use_missing,"
    " use_quantized_grad, num_grad_quant_bins, quant_train_renew_leaf,"
    " stochastic_rounding, leaf_estimation_iterations,"
    " derivative_precision, auto_learning_rate,"
    " permutation_count, fold_len_multiplier, fold_permutation_block,"
    " ordered_seed"
)

# Parameters that name a real feature this parser does not cover, reported as
# unsupported instead of as unknown so the message can say why. Almost all of
# them are LightGBM's; `bootstrap_type`, `bagging_temperature` and
# `bootstrap_seed` are CatBoost's Bayesian bootstrap (`sampling.mojo`), which
# needs a `BayesianBootstrapParams` handed to a trainer exactly as GOSS needs a
# `GossParams`, so a parameter string cannot select it either.
#
# `feature_contri`, `cegb_penalty_feature_coupled`, and
# `cegb_penalty_feature_lazy` are per-feature vectors, which a
# whitespace-separated parameter string cannot carry any more than it can
# carry `monotone_constraints`. All three are reachable through
# `TreeParams.extra.penalties` in the Mojo API: the first on `contri`, the
# other two on `penalties.cegb` (cegb.mojo).
#
# Every vendor spelling of a Mojo-API-only parameter is listed too, so that
# `subsample` and `rsm` and `cat_features` each get the sentence naming the
# feature they asked for rather than "unknown parameter". A name that is
# only ever an alias still belongs here: what the user needs to be told is
# about the parameter, not about the spelling.
comptime _MOJO_API_ONLY = String(
    "bagging_fraction subsample sub_row bagging"
    " bagging_freq subsample_freq bagging_seed pos_bagging_fraction"
    " neg_bagging_fraction top_rate other_rate"
    " monotone_constraints monotonic_cst interaction_constraints"
    " interaction_cst"
    " categorical_feature categorical_features cat_features"
    " cat_smooth cat_l2 max_cat_threshold"
    " max_cat_to_onehot one_hot_max_size min_data_per_group"
    " early_stopping_round"
    " early_stopping_rounds od_wait n_iter_no_change"
    " first_metric_only lambdarank_truncation_level"
    " label_gain sigmoid eval_at ndcg_eval_at class_weight is_unbalance"
    " unbalance unbalanced_sets scale_pos_weight feature_contri"
    " feature_contrib fc fp feature_penalty cegb_penalty_feature_coupled"
    " cegb_penalty_feature_lazy bootstrap_type bagging_temperature"
    " bootstrap_seed min_data_in_bin bin_construct_sample_cnt"
    " drop_rate rate_drop max_drop skip_drop uniform_drop drop_seed"
    " xgboost_dart_mode metric eval_metric custom_metric"
)


struct TrainConfig(Copyable, Movable):
    """A parsed parameter string, ready to hand to `fit`/`fit_multiclass`.

    `objective` is a boosting.mojo objective code, or `MULTICLASS`, in which
    case `n_classes` is the class count and the caller trains with
    `fit_multiclass`. `n_classes` is 1 for every single-output objective.

    `auto_learning_rate` is CatBoost's data-dependent learning rate
    (auto_learning_rate.mojo, catalog A9) and is **disabled** unless the
    parameter string says `auto_learning_rate=true`. When it is disabled,
    which is the default, `booster.learning_rate` is the whole story and
    `resolved_learning_rate` returns it unchanged.
    """

    var objective: Int
    var n_classes: Int
    var booster: BoosterParams
    var max_bin: Int
    var alpha: Float64
    var device: Int
    var use_missing: Bool
    var auto_learning_rate: AutoLearningRateParams

    def __init__(out self):
        """LightGBM's defaults as mojotrees sets them: squared error on the
        CPU with the `BoosterParams.default()` hyperparameters."""
        self.objective = SQUARED_ERROR
        self.n_classes = 1
        self.booster = BoosterParams.default()
        self.max_bin = 255
        self.alpha = 0.9
        self.device = CPU_DEVICE
        self.use_missing = True
        self.auto_learning_rate = AutoLearningRateParams.disabled()

    def is_multiclass(self) -> Bool:
        return self.objective == MULTICLASS

    def resolved_learning_rate(self, n_rows: Int) raises -> Float64:
        """`booster.learning_rate`, or CatBoost's derived rate in its place.

        Returns `booster.learning_rate` untouched unless the parameter string
        asked for `auto_learning_rate=true`, so a caller can route every
        training entry point through this without changing any behavior it
        already has. `n_rows` is the **train** row count, which is what
        CatBoost reads (`options_helper.cpp:411`); it is not known at parse
        time, which is why this is a method rather than something
        `parse_params` folds into `booster.learning_rate`.
        """
        return resolve_learning_rate(
            self.auto_learning_rate,
            self.objective,
            self.booster.n_estimators,
            n_rows,
            self.booster.learning_rate,
        )


def _lower_ascii(value: String) -> String:
    """`value` with A-Z folded to a-z, every other byte untouched.

    The one place a value string is case folded on this surface. CatBoost
    writes `RMSE`, `SymmetricTree`, `Bayesian` and `Plain`; XGBoost writes
    `reg:squarederror`; LightGBM writes lowercase. docs/PARAMETER_NAMING.md
    makes value strings case insensitive, so every value comparison below
    happens after this fold and every table entry is written lowercase.

    ASCII only, and deliberately: a parameter value is a vocabulary word
    from one of four libraries, all of which spell theirs in ASCII, and a
    Unicode fold would make `objective` depend on a locale table for no
    reachable gain.
    """
    var out = String("")
    for b in value.as_bytes():
        var c = Int(b)
        if c >= 65 and c <= 90:
            c += 32
        out = out + chr(c)
    return out^


def objective_from_name(name: String) raises -> Int:
    """Objective code for an objective name, or `MULTICLASS`.

    Accepts all four vendors' spellings: LightGBM's names and aliases,
    XGBoost's `reg:`/`binary:`/`multi:`/`count:` loss names, CatBoost's
    `loss_function` values, and scikit-learn's `HistGradientBoosting*`
    `loss` values. `name` is folded to lowercase first, so `RMSE`,
    `rmse` and `Rmse` are one name.

    The LightGBM objectives mojotrees does not implement are named
    explicitly in `_raise_if_unimplemented_objective` rather than falling
    into the unknown-name message: a user who asks for `multiclassova` has
    asked for a real thing, and being told it is unknown would be
    misleading.
    """
    var lowered = _lower_ascii(name)
    return _objective_from_lower(lowered)


def _objective_from_lower(name: String) raises -> Int:
    """`objective_from_name` with the case fold already applied.

    Split out so that `parse_params`, which has folded the value once
    already, does not fold it twice; the table is here and nowhere else.

    Each row is one objective and every vendor's word for it. The vendors,
    in column order per docs/PARAMETER_NAMING.md, are LightGBM (`objective`),
    XGBoost (`objective`), CatBoost (`loss_function`) and scikit-learn's
    `HistGradientBoosting*` (`loss`).
    """
    if (
        name == "regression"
        or name == "regression_l2"
        or name == "l2"
        or name == "mean_squared_error"
        or name == "mse"
        # XGBoost, and its pre-1.0 spelling; CatBoost; scikit-learn.
        or name == "reg:squarederror"
        or name == "reg:linear"
        or name == "rmse"
        or name == "squared_error"
    ):
        return SQUARED_ERROR
    if (
        name == "binary"
        # XGBoost; CatBoost; scikit-learn.
        or name == "binary:logistic"
        or name == "logloss"
        or name == "log_loss"
    ):
        return BINARY_LOGISTIC
    if name == "poisson" or name == "count:poisson":
        return POISSON
    # CatBoost spells it `Huber` too, though it carries its delta in the
    # name (`Huber:delta=1.0`) rather than in a separate parameter; the bare
    # word is the spelling both libraries share. XGBoost's nearest thing is
    # `reg:pseudohubererror`, a different curve, refused by name below.
    if name == "huber":
        return HUBER
    if (
        name == "quantile"
        or name == "reg:quantileerror"
        # scikit-learn's HistGradientBoostingRegressor.
        or name == "quantile_loss"
    ):
        return QUANTILE
    if (
        name == "mae"
        or name == "regression_l1"
        or name == "l1"
        or name == "mean_absolute_error"
        # XGBoost; CatBoost; scikit-learn.
        or name == "reg:absoluteerror"
        or name == "absolute_error"
    ):
        return L1
    if name == "gamma" or name == "reg:gamma":
        return GAMMA
    if name == "tweedie" or name == "reg:tweedie":
        return TWEEDIE
    if (
        name == "mape"
        or name == "mean_absolute_percentage_error"
        # XGBoost 2.x.
        or name == "reg:absolutepercentageerror"
    ):
        return MAPE
    if name == "fair":
        return FAIR
    if (
        name == "cross_entropy"
        or name == "xentropy"
        # CatBoost's name for the same loss.
        or name == "crossentropy"
    ):
        return CROSS_ENTROPY
    if (
        name == "multiclass"
        or name == "softmax"
        # XGBoost; CatBoost. `multi:softprob` differs from `multi:softmax`
        # only in what predict returns, which is `predict_proba` here, so
        # both name the same trainer.
        or name == "multi:softmax"
        or name == "multi:softprob"
    ):
        return MULTICLASS
    if name == "reg:pseudohubererror":
        raise Error(
            "objective 'reg:pseudohubererror' is XGBoost's smooth"
            " pseudo-Huber, which is a different curve from LightGBM's"
            " 'huber' (a quadratic spliced to a line at alpha), not a"
            " spelling of it; mojotrees implements 'huber'"
        )
    # XGBoost's `rank:ndcg` is the same LambdaRank-with-NDCG family and
    # resolves here. CatBoost's `YetiRank` is not: it is a different pairwise
    # loss, so it falls through to the unknown-name message rather than being
    # told it is LambdaRank spelled differently.
    if name == "lambdarank" or name == "rank:ndcg":
        raise Error(
            "objective 'lambdarank' needs query groups, which a parameter"
            " string cannot carry; use train_ranker in the Mojo API"
        )
    if name == "custom":
        raise Error(
            "objective 'custom' needs a gradient callback; use fit_custom in"
            " the Mojo API"
        )
    _raise_if_unimplemented_objective(name)
    raise Error(
        "unknown objective '",
        name,
        "'; expected regression, binary, multiclass, poisson, huber,"
        " quantile, mae, gamma, tweedie, mape, fair, or cross_entropy",
    )


def _raise_if_unimplemented_objective(name: String) raises:
    """Report a LightGBM objective mojotrees has not implemented as exactly
    that, with what it would take.

    These are deliberate omissions, not oversights, and each is recorded in
    docs/LIGHTGBM_PARITY.md. Naming them here keeps the parity list and the
    error a user actually sees in one place.
    """
    if name == "cross_entropy_lambda" or name == "xentlambda":
        raise Error(
            "objective 'cross_entropy_lambda' is not implemented; it"
            " parameterizes the rate through log1p(exp(raw)) rather than"
            " the logistic, so it is a separate link, not an alias of"
            " 'cross_entropy'"
        )
    if name == "multiclassova" or name == "multiclass_ova" or (
        name == "ova" or name == "ovr"
        # CatBoost's spelling of the same scheme.
        or name == "multiclassoneversusall"
    ):
        raise Error(
            "objective 'multiclassova' is not implemented; one-vs-rest"
            " needs an independent binary model per class, which is a"
            " different trainer from the shared-softmax 'multiclass'"
        )
    if name == "rank_xendcg" or name == "xendcg":
        raise Error(
            "objective 'rank_xendcg' is not implemented; 'lambdarank' is"
            " the ranking objective mojotrees provides"
        )


def objective_display_name(objective: Int) raises -> String:
    """The LightGBM name for an objective code, for reporting."""
    if objective == SQUARED_ERROR:
        return "regression"
    if objective == BINARY_LOGISTIC:
        return "binary"
    if objective == POISSON:
        return "poisson"
    if objective == HUBER:
        return "huber"
    if objective == QUANTILE:
        return "quantile"
    if objective == L1:
        return "mae"
    if objective == GAMMA:
        return "gamma"
    if objective == TWEEDIE:
        return "tweedie"
    if objective == MAPE:
        return "mape"
    if objective == FAIR:
        return "fair"
    if objective == CROSS_ENTROPY:
        return "cross_entropy"
    if objective == MULTICLASS:
        return "multiclass"
    raise Error("unknown objective code ", objective)


def _parse_bool(key: String, value: String) raises -> Bool:
    """LightGBM's boolean spelling: true/false, plus 1/0."""
    if value == "true" or value == "1":
        return True
    if value == "false" or value == "0":
        return False
    raise Error(
        "parameter '", key, "' expects true or false, got '", value, "'"
    )


def _parse_int(key: String, value: String) raises -> Int:
    try:
        return Int(value)
    except:
        raise Error(
            "parameter '", key, "' expects an integer, got '", value, "'"
        )


def _parse_f64(key: String, value: String) raises -> Float64:
    try:
        return Float64(value)
    except:
        raise Error(
            "parameter '", key, "' expects a number, got '", value, "'"
        )


def _is_mojo_api_only(key: String) -> Bool:
    for known in _MOJO_API_ONLY.split():
        if key == known:
            return True
    return False


def params_names_mojo_api_only(spec: String) -> Bool:
    """Whether `spec` names something mojotrees implements but parameter
    strings cannot carry, rather than something merely invalid.

    A caller that has to turn a rejected parameter string into a status
    code uses this to separate "you asked for a real feature the wrong way"
    from "this is not a parameter". It never raises, so it is safe to call
    while already handling an error.
    """
    for token_slice in spec.split():
        var token = String(token_slice)
        var eq = token.find("=")
        if eq < 0:
            continue
        var key = String(token[byte=0:eq])
        var value = String(token[byte=eq + 1 :])
        if _is_mojo_api_only(key):
            return True
        var lowered = _lower_ascii(value)
        if (
            key == "objective"
            or key == "application"
            or key == "loss_function"
            or key == "loss"
        ) and (
            lowered == "lambdarank"
            or lowered == "rank:ndcg"
            or lowered == "custom"
        ):
            return True
        # `data_sample_strategy=bagging` is accepted; only the GOSS value
        # needs the Mojo API, because selecting it means handing the trainer
        # a `GossParams`.
        if key == "data_sample_strategy" and lowered == "goss":
            return True
        # Same rule for the three `boosting_type` values that name a real
        # trainer needing a parameter bundle. `ordered` is deliberately not
        # here, and the reason changed on 2026-08-16: it used to be absent
        # because it was not implemented, and it is absent now because this
        # surface implements it. Its knobs are four scalars, so there is no
        # bundle a string cannot carry.
        if key == "boosting_type" or key == "boosting" or key == "booster":
            if lowered == "dart" or lowered == "goss" or lowered == "rf":
                return True
    return False


def objective_default_alpha(objective: Int) -> Float64:
    """The value the objective's scalar parameter takes when the parameter
    string does not set one: LightGBM's `fair_c` and
    `tweedie_variance_power` defaults for those two objectives, and
    LightGBM's `alpha` default of 0.9 for the rest (which is the one that
    matters for huber and quantile; the others ignore it)."""
    if objective == FAIR:
        return DEFAULT_FAIR_C
    if objective == TWEEDIE:
        return DEFAULT_TWEEDIE_VARIANCE_POWER
    return 0.9


def _alpha_key_for(objective: Int) -> String:
    """The parameter name that sets `alpha` for this objective, or an empty
    string when the objective has no scalar parameter."""
    if objective == FAIR:
        return "fair_c"
    if objective == TWEEDIE:
        return "tweedie_variance_power"
    if objective == HUBER or objective == QUANTILE:
        return "alpha"
    return ""


def _check_alpha_key(config: TrainConfig, alpha_key: String) raises:
    """Reject a scalar parameter that does not belong to the objective.

    LightGBM accepts `fair_c` alongside `objective=tweedie` and silently
    ignores it, which hides a real mistake: the number the user set is not
    the number the model trains with. Since these three names all land in
    the same slot here, accepting the wrong one would be worse still, so a
    mismatch raises.
    """
    if alpha_key.byte_length() == 0:
        return
    var expected = _alpha_key_for(config.objective)
    if alpha_key == expected:
        return
    if expected.byte_length() == 0:
        raise Error(
            "parameter '",
            alpha_key,
            "' does not apply to objective '",
            objective_display_name(config.objective),
            "'",
        )
    raise Error(
        "parameter '",
        alpha_key,
        "' does not apply to objective '",
        objective_display_name(config.objective),
        "'; it takes '",
        expected,
        "'",
    )


def _check_boosting_type(value: String) raises:
    """The one `boosting_type` vocabulary: gbdt, dart, goss, rf, plain,
    ordered. `value` is already lowercased.

    `plain` is CatBoost's word for boosting without its ordered scheme,
    which is what `gbdt` is, so it resolves to `gbdt` and this surface's
    default configuration answers it by doing nothing.

    `ordered` is CatBoost's ordered boosting: derivatives for row i taken
    from a model that never saw row i. **It is implemented**
    (ordered_boosting.mojo, and the fold ladder inside `boosting.train`), and
    since 2026-08-16 this surface sets `BoosterParams.ordered` rather than
    refusing the value. It differs from `dart`, `goss` and `rf` in the way
    that matters to a parameter string: its four knobs are scalars
    (`permutation_count`, `fold_len_multiplier`, `fold_permutation_block`,
    `ordered_seed`) and a whitespace-separated string can carry every one of
    them, so there is no bundle the string cannot express.

    It is refused on the GPU in `_validate` rather than here, because the
    device is a separate key and is not known at this call.

    `dart`, `goss` and `rf` are implemented, but selecting one means handing
    a trainer a `DartParams`, a `GossParams` or the RF loop, which a
    whitespace-separated string cannot carry, exactly as it cannot carry
    `bagging_fraction`. They are refused with the Mojo API sentence rather
    than as unknown values.
    """
    if value == "gbdt" or value == "plain" or value == "gbrt" or (
        value == "traditional" or value == "gbtree"
    ):
        return
    if value == "ordered":
        return
    if value == "dart" or value == "goss" or value == "rf" or (
        value == "random_forest" or value == "dart_mode"
    ):
        raise Error(
            "boosting_type '",
            value,
            "' is supported by the Mojo API only, not by parameter strings:"
            " it needs a parameter bundle (DartParams, GossParams) or the"
            " random-forest loop, which a whitespace-separated string cannot"
            " carry any more than it can carry bagging_fraction",
        )
    raise Error(
        "unknown boosting_type '",
        value,
        "'; expected gbdt, dart, goss, rf, plain (= gbdt), or ordered",
    )


def _validate(
    config: TrainConfig, saw_num_class: Bool, saw_ordered_knob: Bool = False
) raises:
    """Range checks that do not depend on the training data. Objective
    specific checks on the label values stay in boosting.mojo, which sees
    the labels; the `alpha` range checks are here as well as there, so a
    parameter string is rejected before any data is read."""
    # The data-independent booster ranges, from the one place that holds
    # them; `callback.check_resettable` applies the same call to a reset.
    check_booster_ranges(
        config.booster.n_estimators,
        config.booster.learning_rate,
        config.booster.tree.num_leaves,
        config.booster.tree.max_depth,
        config.booster.tree.min_data_in_leaf,
        config.booster.tree.min_child_hess,
        config.booster.tree.lambda_l1,
        config.booster.tree.lambda_reg,
        config.booster.tree.feature_fraction,
        config.booster.tree.feature_fraction_bynode,
        config.booster.tree.feature_fraction_bylevel,
    )
    # The data-independent half of the remaining tree controls. The per-
    # feature vectors are checked against the dataset later, in
    # `tree.grow_tree`, because a parameter string cannot carry one.
    # `scale_computed_per_tree` on the CPU arm only: the dense CPU round
    # loops compute `random_score_scale` per tree onto their own copy of
    # the bundle, so a positive `random_strength` beside a zero scale is
    # legitimate HERE and a defect anywhere else. The device and
    # distributed loops do not compute it, so they keep the refusal.
    #
    # Known narrow gap, and it is a refusal rather than a silent drop: a
    # CPU MULTICLASS fit passes this and then raises at `split.mojo`'s
    # noise read mid-fit, because `_boost_rounds_multiclass` is not
    # wired. A dedicated refusal here would give a better message.
    config.booster.tree.extra.check_scalars(
        config.booster.tree.min_data_in_leaf,
        scale_computed_per_tree=(config.device == CPU_DEVICE),
    )
    # Exclusive feature bundling: the knobs are range-checked whether or not
    # the switch is on, so a bad value is named here rather than at the first
    # training call that happens to turn bundling on, and the switch itself is
    # checked against the device that would have to honor it.
    check_bundling_supported(
        config.booster.bundling.enabled, config.device == CPU_DEVICE
    )
    config.booster.bundling.check()
    # Ordered boosting: the same two questions bundling is asked, in the same
    # order. Its own range check first (`permutation_count`,
    # `fold_len_multiplier`, the block size), then whether the run that would
    # carry it can honor it.
    config.booster.ordered.validate()
    if config.booster.ordered.enabled and config.device != CPU_DEVICE:
        raise Error(
            "boosting_type=ordered trains on the CPU only: the fold ladder"
            " lives in boosting.train and train_gpu reads"
            " BoosterParams.ordered nowhere, so a GPU run would train a plain"
            " ensemble and report an ordered one. Set device=cpu"
        )
    if saw_ordered_knob and not config.booster.ordered.enabled:
        raise Error(
            "permutation_count, fold_len_multiplier, fold_permutation_block"
            " and ordered_seed configure ordered boosting and are read only"
            " when boosting_type=ordered; set that too, or drop the knob"
        )
    check_max_bin(config.max_bin)

    if config.objective == HUBER and config.alpha <= 0.0:
        raise Error("alpha must be positive for objective 'huber'")
    if config.objective == QUANTILE and not 0.0 < config.alpha < 1.0:
        raise Error("alpha must be in (0, 1) for objective 'quantile'")
    if config.objective == FAIR and config.alpha <= 0.0:
        raise Error("fair_c must be positive")
    if config.objective == TWEEDIE and not 1.0 < config.alpha < 2.0:
        raise Error("tweedie_variance_power must be in (1, 2)")

    if config.is_multiclass():
        if not saw_num_class:
            raise Error("objective 'multiclass' requires num_class")
        if config.n_classes < 2:
            raise Error("num_class must be at least 2")
    elif saw_num_class:
        raise Error(
            "num_class applies to objective 'multiclass' only; got objective"
            " '",
            objective_display_name(config.objective),
            "'",
        )


def parse_params(spec: String) raises -> TrainConfig:
    """Parse a whitespace separated `key=value` parameter string.

    An empty string returns the defaults. Every key is validated, so a
    successful parse means training will not fail on the parameters alone.
    """
    var config = TrainConfig()
    var saw_num_class = False
    var alpha_key = String("")
    # `random_state` sets every seed this surface carries, but only the ones
    # the string did not name outright: an explicit `feature_fraction_seed`
    # wins over a global seed whichever order they appear in, which is the
    # rule a reader expects and the one LightGBM's `seed` follows.
    var random_state = 0
    var saw_random_state = False
    var saw_feature_fraction_seed = False
    var saw_extra_seed = False
    # CatBoost's auto learning rate fires only if the user set none of
    # `learning_rate`, `leaf_estimation_method`, `leaf_estimation_iterations`
    # and `l2_leaf_reg` (`options_helper.cpp:276-281`). A parameter string is
    # the one surface that can tell "set" from "left at the default", so the
    # gate is tracked here and handed to `AutoLearningRateParams` at the end.
    # mojotrees has no `leaf_estimation_method` key at all (Newton only), so
    # that third gate is permanently open for us.
    var saw_learning_rate = False
    var saw_lambda_l2 = False
    var saw_leaf_estimation_iterations = False
    var saw_auto_learning_rate = False
    # Whether the string named any of ordered boosting's four knobs. Tracked
    # for one reason: a knob without `boosting_type=ordered` is a value that
    # would be parsed, stored, and then never read, which is the shape this
    # package refuses everywhere else. `_validate` turns it into a message.
    var saw_ordered_knob = False

    for token_slice in spec.split():
        var token = String(token_slice)
        var eq = token.find("=")
        if eq < 0:
            raise Error(
                "expected key=value in parameter string, got '", token, "'"
            )
        var key = String(token[byte=0:eq])
        var value = String(token[byte=eq + 1 :])
        if key.byte_length() == 0:
            raise Error("empty parameter name in '", token, "'")
        if value.byte_length() == 0:
            raise Error("parameter '", key, "' has an empty value")
        # docs/PARAMETER_NAMING.md: value strings are case insensitive, keys
        # are not. Folded once here so that no branch below has to remember
        # to, and so that a value that is a number or a path is untouched by
        # anything except an A-Z byte, which neither can contain.
        var lowered = _lower_ascii(value)

        # `application` is LightGBM's, `loss_function` CatBoost's, `loss`
        # scikit-learn's.
        if (
            key == "objective"
            or key == "application"
            or key == "loss_function"
            or key == "loss"
        ):
            config.objective = _objective_from_lower(lowered)
        elif (
            key == "num_class"
            or key == "num_classes"
            # CatBoost's name for the same count.
            or key == "classes_count"
        ):
            config.n_classes = _parse_int(key, value)
            saw_num_class = True
        elif (
            key == "n_estimators"
            # LightGBM.
            or key == "num_iterations"
            or key == "num_iteration"
            or key == "num_round"
            or key == "num_rounds"
            or key == "num_boost_round"
            or key == "num_trees"
            or key == "num_tree"
            # CatBoost; scikit-learn's HistGradientBoosting*.
            or key == "iterations"
            or key == "max_iter"
        ):
            config.booster.n_estimators = _parse_int(key, value)
        elif (
            key == "learning_rate"
            or key == "shrinkage_rate"
            # XGBoost.
            or key == "eta"
        ):
            config.booster.learning_rate = _parse_f64(key, value)
            saw_learning_rate = True
        elif (
            key == "max_leaves"
            # LightGBM; scikit-learn's HistGradientBoosting*.
            or key == "num_leaves"
            or key == "num_leaf"
            or key == "max_leaf_nodes"
        ):
            config.booster.tree.num_leaves = _parse_int(key, value)
        elif (
            key == "min_child_samples"
            # LightGBM (and CatBoost, which uses LightGBM's spelling here);
            # scikit-learn's HistGradientBoosting*.
            or key == "min_data_in_leaf"
            or key == "min_data"
            or key == "min_samples_leaf"
        ):
            config.booster.tree.min_data_in_leaf = _parse_int(key, value)
        elif (
            key == "min_child_weight"
            # LightGBM.
            or key == "min_sum_hessian_in_leaf"
            or key == "min_sum_hessian"
        ):
            config.booster.tree.min_child_hess = _parse_f64(key, value)
        elif key == "reg_alpha" or key == "lambda_l1":
            config.booster.tree.lambda_l1 = _parse_f64(key, value)
        elif (
            key == "reg_lambda"
            # LightGBM; CatBoost; scikit-learn's HistGradientBoosting*.
            or key == "lambda_l2"
            or key == "lambda"
            or key == "l2_leaf_reg"
            or key == "l2_regularization"
        ):
            config.booster.tree.lambda_reg = _parse_f64(key, value)
            saw_lambda_l2 = True
        elif key == "max_depth" or key == "depth":
            config.booster.tree.max_depth = _parse_int(key, value)
        elif key == "grow_policy":
            # XGBoost's and CatBoost's parameter, with all three vendors'
            # value spellings and all of them case insensitive
            # (growth_policy.mojo): `lossguide` grows by best gain anywhere,
            # `depthwise` commits a depth at a time, `symmetrictree` grows
            # oblivious trees. LightGBM has no such switch, so this is an
            # extension rather than a parity row.
            config.booster.tree.grow_policy = parse_grow_policy(lowered)
        elif key == "boosting_type" or key == "boosting" or key == "booster":
            _check_boosting_type(lowered)
            # CatBoost's `Ordered`. `_check_boosting_type` has already
            # accepted or refused the vocabulary; this turns the one value
            # that names a mechanism into the bundle that mechanism reads.
            # `enable` supplies the module's defaults, and the four keys
            # below overwrite whichever of them the string names -- in either
            # order, because they set fields on an already-enabled bundle and
            # `boosting_type=ordered` is what flips `enabled`.
            if lowered == "ordered":
                config.booster.ordered.enabled = True
        # The four knobs of ordered boosting (ordered_boosting.mojo). Each is
        # a scalar, which is why this surface can carry the mechanism at all
        # where it cannot carry dart's or GOSS's. They are read whether or not
        # `boosting_type=ordered` appeared, and `_validate` is what refuses a
        # knob set beside a plain fit: a value that would be dropped is the
        # thing this parser reports rather than ignores.
        elif key == "permutation_count":
            config.booster.ordered.permutation_count = _parse_int(key, value)
            saw_ordered_knob = True
        elif key == "fold_len_multiplier":
            config.booster.ordered.fold_len_multiplier = _parse_f64(
                key, value
            )
            saw_ordered_knob = True
        elif key == "fold_permutation_block":
            config.booster.ordered.permutation_block_size = _parse_int(
                key, value
            )
            saw_ordered_knob = True
        elif key == "ordered_seed":
            config.booster.ordered.seed = _parse_int(key, value)
            saw_ordered_knob = True
        elif (
            key == "colsample_bytree"
            # LightGBM; CatBoost.
            or key == "feature_fraction"
            or key == "sub_feature"
            or key == "rsm"
        ):
            config.booster.tree.feature_fraction = _parse_f64(key, value)
        elif (
            key == "colsample_bynode"
            # LightGBM; scikit-learn's HistGradientBoosting*.
            or key == "feature_fraction_bynode"
            or key == "sub_feature_bynode"
            or key == "max_features"
        ):
            config.booster.tree.feature_fraction_bynode = _parse_f64(
                key, value
            )
        elif (
            key == "colsample_bylevel" or key == "feature_fraction_bylevel"
        ):
            # XGBoost's name; LightGBM has no per-level fraction at all, so
            # this is an extension rather than a parity row (sampling.mojo).
            config.booster.tree.feature_fraction_bylevel = _parse_f64(
                key, value
            )
        elif key == "feature_fraction_seed":
            config.booster.tree.feature_fraction_seed = _parse_int(key, value)
            saw_feature_fraction_seed = True
        elif (
            key == "min_split_gain"
            # LightGBM; XGBoost.
            or key == "min_gain_to_split"
            or key == "gamma"
        ):
            config.booster.tree.extra.min_gain_to_split = _parse_f64(
                key, value
            )
        elif (
            key == "max_delta_step"
            or key == "max_tree_output"
            or key == "max_leaf_output"
        ):
            config.booster.tree.extra.max_delta_step = _parse_f64(key, value)
        elif key == "path_smooth":
            config.booster.tree.extra.path_smooth = _parse_f64(key, value)
        elif key == "extra_trees" or key == "extra_tree":
            config.booster.tree.extra.extra_trees = _parse_bool(key, value)
        elif key == "extra_seed":
            config.booster.tree.extra.extra_seed = _parse_int(key, value)
            saw_extra_seed = True
        # `random_state` is scikit-learn's word and the canonical one; the
        # other three are LightGBM's, XGBoost's and CatBoost's. It seeds
        # every draw this surface can reach, which today is the feature
        # sampling seed and the extra-trees seed; the bagging, GOSS, dart and
        # bootstrap seeds belong to samplers a parameter string cannot select
        # at all, so there is nothing here for them to seed.
        #
        # It sets a seed to `random_state` rather than deriving one per
        # component the way LightGBM's `seed` does. That is a deliberate
        # divergence and it is stated rather than hidden: LightGBM derives
        # its per-component seeds by running its own LCG, which mojotrees
        # does not have and would have to reimplement bit for bit to match,
        # and mojotrees's draws are splitmix64 and would not reproduce
        # LightGBM's subsets from a matching seed anyway. What `random_state`
        # buys is reproducibility of a mojotrees fit, which is what a script
        # that sets it is asking for.
        elif (
            key == "random_state"
            or key == "seed"
            or key == "random_seed"
            or key == "data_random_seed"
        ):
            random_state = _parse_int(key, value)
            saw_random_state = True
        # scikit-learn's word; LightGBM's, XGBoost's two, and CatBoost's.
        # There is no worker count on this surface: `parallel.mojo` takes
        # one from `MOJOTREES_NUM_WORKERS` and from the machine, and a fit
        # is bit-identical at every worker count by contract. The values
        # that mean "use the machine" are therefore already satisfied and
        # are accepted; a specific count is refused by name rather than
        # accepted and dropped, because the user asked for a thing that
        # would not happen.
        elif (
            key == "n_jobs"
            or key == "num_threads"
            or key == "num_thread"
            or key == "nthread"
            or key == "thread_count"
        ):
            var jobs = _parse_int(key, value)
            if jobs > 0:
                raise Error(
                    "n_jobs=",
                    jobs,
                    " cannot be set from a parameter string: the worker"
                    " count comes from MOJOTREES_NUM_WORKERS and from the"
                    " machine (parallel.mojo), and a fit is bit-identical"
                    " at every worker count. Set MOJOTREES_NUM_WORKERS=",
                    jobs,
                    " in the environment instead. n_jobs of -1 or 0, which"
                    " means 'use the machine', is what this surface already"
                    " does and is accepted",
                )
        # scikit-learn's and CatBoost's word; LightGBM and XGBoost spell it
        # `verbosity`, CatBoost also takes `logging_level`. Nothing on this
        # surface writes a training log, so silence is what it already does
        # and a request for silence is honored by doing nothing. A request
        # for output is refused by name and pointed at the surface that has
        # one, rather than accepted and quietly producing none.
        elif key == "verbose" or key == "verbosity":
            if _parse_int(key, value) > 0:
                raise Error(
                    "parameter '",
                    key,
                    "' above 0 cannot be honored here: a parameter string"
                    " reaches the C ABI and the CLI, neither of which writes"
                    " a training log. The Python estimator's verbose= does,"
                    " through callback.log_evaluation. A value of 0 or below"
                    " is what this surface already does and is accepted",
                )
        elif key == "logging_level":
            # CatBoost's spelling of the same switch, with its own vocabulary.
            if lowered != "silent":
                raise Error(
                    "logging_level '",
                    value,
                    "' cannot be honored here: a parameter string reaches the"
                    " C ABI and the CLI, neither of which writes a training"
                    " log. 'Silent' is what this surface already does and is"
                    " accepted",
                )
        # CatBoost's `score_function`, the functional a split candidate is
        # scored by. `L2` is `G^2 / (H + lambda)`, which is what mojotrees
        # has always maximized and is the default; `Cosine` is CatBoost's own
        # default, a ratio rather than a sum
        # (`tree_parameters_extra.SCORE_COSINE`, and
        # docs/design/CATBOOST_CATALOG.md A10 for the derivation).
        #
        # Both are now honored rather than one of them refused. The field
        # this writes, `ExtraTreeParams.score_function`, is read by
        # `tree._search` and `tree._grow_oblivious_levels`, which pass it
        # into `split.find_best_split` and `split.find_best_split_shared`;
        # `ExtraTreeParams.is_active()` names it, so the device split search
        # -- which scores `G^2/(H+lambda)` and nothing else -- refuses or
        # declines instead of returning an L2 tree under a Cosine label.
        #
        # Worth stating because it is the reason the two are NOT aliases:
        # Cosine's numerator is the L2 sum and at `lambda_l2 = 0` its
        # denominator collapses onto the same expression, so it degenerates
        # to `sqrt(L2)` and the argmax *within one node* cannot move.
        # `lambda_l2 = 0` is this package's stock value. It is still not an
        # alias there: leaf-wise growth, the default, compares gains from
        # different parents, and `sqrt` does not preserve that ordering
        # (CATBOOST_CATALOG A10 section 5). The CatBoost-mode arm sets
        # `lambda_l2 = 3`, which is off the degenerate point outright.
        elif key == "score_function":
            config.booster.tree.extra.score_function = parse_score_function(
                lowered
            )
        # CatBoost's `max_ctr_complexity`. Refused for any value, its own
        # default included: CTRs -- ordered target statistics over
        # categorical combinations -- are the feature construction the name
        # controls, and mojotrees has none of it (its categorical handling
        # is LightGBM's category-set split). A number here would ask for
        # combinations that are never built.
        elif key == "max_ctr_complexity":
            # Complexity 1 is BUILT now (ctr_columns.mojo, catalog A19) and a
            # simple projection reaches a design matrix. Above 1 needs the
            # candidate enumeration driven by a grow loop, and nothing drives
            # it, so it is refused by name.
            #
            # This key only bounds the arity. It does NOT turn CTRs on, and
            # there is deliberately no parameter-string name that does,
            # because a fit that built CTR columns cannot be SAVED yet -- the
            # tables are model state and the format has no section for them.
            # `ctr.check_ctr_model_support` refuses at every model-producing
            # entry point and `serialize.check_ctr_serializable` refuses at
            # every writer.
            var complexity = _parse_int(key, value)
            if complexity != 1:
                raise Error(
                    "max_ctr_complexity above 1 is not implemented: the"
                    " projection enumeration exists"
                    " (ctr_combinations.grow_tree_ctr_projections) but no"
                    " grow loop drives it, so a combination would never be"
                    " built. 1, the value CatBoost itself resolves to for any"
                    " fit under 200 iterations, is accepted"
                )
        # CatBoost's `random_strength`, the only name on this surface that is
        # not LightGBM's. 0.0, the default, is LightGBM's behavior exactly. A
        # positive value parses and then fails validation with a sentence
        # naming what is missing (`ExtraTreeParams.check_random_strength`),
        # which is the same refuse-rather-than-ignore path
        # `use_quantized_grad` takes below: the noise needs a per-tree scale
        # off the ensemble's gradients that no trainer computes in this
        # build, and a split search cannot reconstruct it from a histogram.
        elif key == "random_strength":
            config.booster.tree.extra.random_strength = _parse_f64(key, value)
        # CatBoost's `leaf_estimation_iterations`, the second name here that
        # is not LightGBM's. 1, the default, is LightGBM's behavior exactly:
        # one Newton step per leaf. It parses and is range-checked, so a
        # configuration that spells out the setting mojotrees matches ports
        # across unchanged.
        #
        # A value above 1 parses and is then refused, by the same
        # refuse-rather-than-ignore rule `random_strength` and
        # `use_quantized_grad` take, for a reason particular to this surface:
        # a parameter string is the entry point for `cli/`, for every
        # `bindings/_mojotrees.mojo` fit including the sparse, custom,
        # multiclass and ranking ones, and for the sklearn wrapper, while the
        # mechanism is reached only from the dense single-output trainers. A
        # string accepted here would therefore be honored by some fits and
        # silently dropped by most, which is worse than either. The Mojo API
        # route named in the message is exact and is not refused.
        #
        # The dense single-output **GPU** trainers left that list on
        # 2026-08-16: `train_gpu` and `train_gpu_with_valid` honor the setting
        # now, through `gpu_objectives_native.GpuLeafEstimator` on the
        # device-objective arm and `boosting._estimate_leaf_values` on the
        # host-objective one. That does not reopen this key, because the
        # refusal is about the *set* a string reaches and not about any one
        # member of it: the sparse, custom-objective, multiclass and ranking
        # trainers still refuse, so a string is still honored by some fits and
        # dropped by others.
        elif key == "leaf_estimation_iterations":
            saw_leaf_estimation_iterations = True
            var iters = _parse_int(key, value)
            if iters > 1:
                raise Error(
                    "leaf_estimation_iterations > 1 cannot be set from a"
                    " parameter string: this string reaches the sparse,"
                    " custom-objective, multiclass and ranking trainers, none"
                    " of which implement extra Newton steps, so it would be"
                    " honored by some fits and dropped by others. Set it from"
                    " the Mojo API instead, on"
                    " TreeParams.extra.leaf_estimation_iterations, which"
                    " boosting.train, boosting.train_more,"
                    " boosting.train_with_valid, train_gpu.train_gpu and"
                    " train_gpu.train_gpu_with_valid honor and every other"
                    " trainer refuses by name"
                )
            config.booster.tree.extra.leaf_estimation_iterations = iters
        # LightGBM's quantized-training family. The four names and no others:
        # LightGBM has no scale-rule, seed, or accumulator-width parameter,
        # and this surface is exactly LightGBM's, so mojotrees's three extra
        # decisions are `MOJOTREES_*` environment overrides instead of keys.
        # `use_quantized_grad=true` parses and then fails validation with a
        # sentence naming what is missing (`ExtraTreeParams.
        # check_quantized_grad`), which is the package's refuse-rather-than-
        # ignore rule; it is not accepted here and dropped later.
        elif key == "use_quantized_grad":
            config.booster.tree.extra.use_quantized_grad = _parse_bool(
                key, value
            )
        elif key == "num_grad_quant_bins":
            config.booster.tree.extra.num_grad_quant_bins = _parse_int(
                key, value
            )
        elif key == "quant_train_renew_leaf":
            config.booster.tree.extra.quant_train_renew_leaf = _parse_bool(
                key, value
            )
        elif key == "stochastic_rounding":
            config.booster.tree.extra.stochastic_rounding = _parse_bool(
                key, value
            )
        # `derivative_precision`, which is not a LightGBM parameter name:
        # LightGBM fixes the same choice at compile time with
        # `SCORE_T_USE_DOUBLE`. It is a key rather than a `MOJOTREES_*`
        # override because it changes a fit's numbers, which is where this
        # package draws that line. `float32` is the default and is LightGBM's
        # profile; `float64` parses and then fails validation with a sentence
        # naming the environment variable that does reach the trainer
        # (`ExtraTreeParams.check_derivative_precision`), which is the same
        # refuse-rather-than-ignore path `use_quantized_grad` takes above.
        elif key == "derivative_precision":
            config.booster.tree.extra.derivative_precision = (
                parse_derivative_precision(value)
            )
        elif (
            key == "monotone_penalty"
            or key == "monotone_splits_penalty"
            or key == "ms_penalty"
            or key == "mc_penalty"
        ):
            config.booster.tree.extra.monotone_penalty = _parse_f64(
                key, value
            )
        elif (
            key == "monotone_constraints_method"
            or key == "monotone_constraining_method"
            or key == "mc_method"
        ):
            # Only `basic` exists here; the other two are named LightGBM
            # methods and are rejected by name rather than downgraded.
            config.booster.tree.extra.monotone_method = parse_monotone_method(
                value
            )
        elif key == "cegb_tradeoff":
            config.booster.tree.extra.penalties.cegb.tradeoff = _parse_f64(
                key, value
            )
        elif key == "cegb_penalty_split":
            config.booster.tree.extra.penalties.cegb.penalty_split = (
                _parse_f64(key, value)
            )
        elif key == "linear_tree":
            # Linear leaves (linear_tree.mojo): fitted by the metric-path
            # trainers, which keep the raw matrix; the binned-only trainers
            # refuse the switch by name rather than dropping it.
            config.booster.linear.enabled = _parse_bool(key, value)
        elif key == "linear_lambda":
            config.booster.linear.linear_lambda = _parse_f64(key, value)
        elif key == "enable_bundle":
            # Exclusive feature bundling, applied by the dense CPU trainers
            # (efb.mojo). Off by default, unlike LightGBM. The device check
            # waits for `_validate`, because `device=` may be named after this
            # key in the same string.
            config.booster.bundling.enabled = _parse_bool(key, value)
        elif key == "feature_pre_filter":
            # `false` is not an unimplemented option, it is the option
            # mojotrees implements: LightGBM prefilters at Dataset
            # construction, and our split search rejects the same candidates
            # as it scans. `true` is still refused, because prefiltering also
            # removes features from the pool `feature_fraction` samples and so
            # can change the trees rather than only their cost.
            #
            # This branch exists so the name reaches its own checker. Without
            # it the key falls through to the unknown-key path and reports
            # `unknown parameter 'feature_pre_filter'`, which is a worse
            # message than the explicit refusal it replaced.
            check_feature_pre_filter(_parse_bool(key, value))
        elif key == "max_conflict_rate":
            config.booster.bundling.params.max_conflict_rate = _parse_f64(
                key, value
            )
        elif key == "data_sample_strategy":
            # The spelling is resolved here so a typo is named, but selecting
            # GOSS needs `GossParams`, which a parameter string cannot carry.
            if canonical_data_sample_strategy(value) != "bagging":
                raise Error(
                    "data_sample_strategy '",
                    value,
                    "' is supported by the Mojo API only, through GossParams",
                )
        elif (
            key == "max_bin"
            # LightGBM's plural; CatBoost; scikit-learn's
            # HistGradientBoosting*.
            or key == "max_bins"
            or key == "border_count"
        ):
            config.max_bin = _parse_int(key, value)
        elif (
            key == "alpha"
            or key == "fair_c"
            or key == "tweedie_variance_power"
        ):
            # Three LightGBM names for one slot: the objective's scalar
            # parameter (see boosting.mojo). Two of them at once would be
            # two different numbers for one slot, so that is rejected rather
            # than resolved by order.
            if alpha_key.byte_length() > 0 and alpha_key != key:
                raise Error(
                    "parameters '",
                    alpha_key,
                    "' and '",
                    key,
                    "' set the same objective parameter; give only one",
                )
            alpha_key = key
            config.alpha = _parse_f64(key, value)
        # `device` is XGBoost 2.x's name and the canonical one; `device_type`
        # is LightGBM's and `task_type` CatBoost's (whose `CPU`/`GPU` fold to
        # our own two words). All three take the same vocabulary.
        elif key == "device" or key == "device_type" or key == "task_type":
            config.device = parse_device(lowered)
        # XGBoost's older switch, which names an algorithm and a device at
        # once. Only the two values that name the histogram algorithm
        # mojotrees implements resolve; `exact` and `approx` are different
        # split searches, not spellings of `device`, and are refused by name.
        elif key == "tree_method":
            if lowered == "hist" or lowered == "auto":
                config.device = parse_device(String("cpu"))
            elif lowered == "gpu_hist":
                config.device = parse_device(String("gpu"))
            elif lowered == "exact" or lowered == "approx":
                raise Error(
                    "tree_method '",
                    value,
                    "' is a different split search, not a spelling of"
                    " device: mojotrees searches a histogram, which is"
                    " XGBoost's 'hist'. Use device=cpu or device=gpu",
                )
            else:
                raise Error(
                    "unknown tree_method '",
                    value,
                    "'; expected 'hist', 'gpu_hist', or 'auto'. device=cpu"
                    " and device=gpu are the canonical spellings",
                )
        elif key == "use_missing":
            config.use_missing = _parse_bool(key, value)
        # CatBoost's data-dependent learning rate (auto_learning_rate.mojo,
        # catalog A9). CatBoost has no such key -- there the derivation is
        # implied by leaving `learning_rate` unset -- so this is an explicit
        # opt-in rather than a parity name, because mojotrees' own default of
        # 0.1 is a real default that a silent override would erase. Off
        # unless the string says so, and it changes nothing at parse time:
        # the rate needs the train row count, so `resolved_learning_rate`
        # applies it.
        elif key == "auto_learning_rate":
            saw_auto_learning_rate = _parse_bool(key, value)
        elif _is_mojo_api_only(key):
            raise Error(
                "parameter '",
                key,
                "' is supported by the Mojo API only, not by parameter"
                " strings",
            )
        else:
            # A real LightGBM tree option that mojotrees does not implement
            # gets its own message saying what it would take, before the
            # unknown-key branch turns it into a typo.
            check_extra_option_supported(key)
            raise Error(
                "unknown parameter '", key, "'; supported: ", SUPPORTED_KEYS
            )

    # The objective may be named after its scalar parameter in the string,
    # so the default and the name check both wait until the whole string is
    # parsed.
    if alpha_key.byte_length() == 0:
        config.alpha = objective_default_alpha(config.objective)
    else:
        _check_alpha_key(config, alpha_key)
    # A global seed fills only the seeds the string did not name, and it
    # waits until the whole string is read so that the two may appear in
    # either order.
    if saw_random_state:
        if not saw_feature_fraction_seed:
            config.booster.tree.feature_fraction_seed = random_state
        if not saw_extra_seed:
            config.booster.tree.extra.extra_seed = random_state
    _validate(config, saw_num_class, saw_ordered_knob)
    if saw_auto_learning_rate:
        _enable_auto_learning_rate(
            config,
            saw_learning_rate,
            saw_lambda_l2,
            saw_leaf_estimation_iterations,
        )
    return config^


def _enable_auto_learning_rate(
    mut config: TrainConfig,
    saw_learning_rate: Bool,
    saw_lambda_l2: Bool,
    saw_leaf_estimation_iterations: Bool,
) raises:
    """Fill in `config.auto_learning_rate` for `auto_learning_rate=true`.

    Two of CatBoost's four gates are reproduced as a refusal rather than as
    silence. CatBoost, handed `learning_rate=0.05 l2_leaf_reg=5`, quietly
    drops back to the constant 0.03 and prints nothing; a user reading their
    own config has no way to see that the rate they thought was derived is
    not. Since asking for `auto_learning_rate=true` here is an explicit act,
    contradicting it is an error:

    - with `learning_rate=` it would be honoring neither one;
    - with `lambda_l2=` or `leaf_estimation_iterations=` it would be a no-op,
      which is the surprising coupling the catalog entry singles out.

    The remaining CatBoost flags come out as they would for a plain fit with
    no eval set: `use_best_model` false (`UpdateUseBestModel` forces it false
    without one, `options_helper.cpp:109-112`) and `boost_from_average`
    derived from the objective. A parameter string carries no eval set and no
    baseline, so neither can be anything else here.
    """
    if saw_learning_rate:
        raise Error(
            "'auto_learning_rate=true' and 'learning_rate=' contradict each"
            " other: the automatic rate exists to replace an unset one."
            " CatBoost resolves this by silently keeping the explicit rate;"
            " give only one"
        )
    if saw_lambda_l2:
        raise Error(
            "'auto_learning_rate=true' with 'lambda_l2=' would do nothing:"
            " CatBoost's derivation is gated on l2_leaf_reg being unset"
            " (options_helper.cpp:280), so setting it pins the rate back to"
            " the constant. Drop one of the two"
        )
    if saw_leaf_estimation_iterations:
        raise Error(
            "'auto_learning_rate=true' with 'leaf_estimation_iterations='"
            " would do nothing: CatBoost's derivation is gated on it being"
            " unset (options_helper.cpp:279). Drop one of the two"
        )

    config.auto_learning_rate = AutoLearningRateParams.catboost_defaults(
        AUTO_LR_TASK_GPU if config.device == GPU_DEVICE else AUTO_LR_TASK_CPU
    )
