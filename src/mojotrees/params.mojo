"""Training configuration from a LightGBM style parameter string.

A parameter string is whitespace separated `key=value` pairs, the same
shape LightGBM accepts in its config files and in `LGBM_BoosterCreate`:

    objective=binary num_leaves=31 learning_rate=0.05 num_iterations=200

This is the only training surface the C ABI (`capi/`) and the CLI (`cli/`)
expose, which keeps both of them free of struct layouts that would have to
change whenever a hyperparameter is added. Keys are LightGBM's names, with
LightGBM's common aliases accepted.

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
from .device import CPU_DEVICE, parse_device
from .objective_registry import MULTICLASS as _MULTICLASS
from .efb import check_bundling_supported
from .sampling import canonical_data_sample_strategy
from .validation import check_booster_ranges, check_max_bin
from .growth_policy import parse_grow_policy
from .tree_parameters_extra import (
    check_extra_option_supported,
    parse_monotone_method,
)

# `TrainConfig.objective` when the parameter string selects softmax
# multiclass, which `fit_multiclass` handles instead of `fit`. Defined once
# in objective_registry.mojo with the other codes (negative to stay out of
# the single-output space forever) and bound here under the name this
# module's callers import.
comptime MULTICLASS = _MULTICLASS

# Every key `parse_params` accepts, primary names only, for error messages.
comptime SUPPORTED_KEYS = String(
    "objective, num_class, num_iterations, learning_rate, num_leaves,"
    " min_data_in_leaf, min_sum_hessian_in_leaf, lambda_l1, lambda_l2,"
    " max_depth, grow_policy, feature_fraction, feature_fraction_bynode,"
    " feature_fraction_bylevel, feature_fraction_seed, min_gain_to_split,"
    " max_delta_step, path_smooth, extra_trees, extra_seed,"
    " monotone_penalty, monotone_constraints_method, cegb_tradeoff,"
    " cegb_penalty_split, linear_tree, linear_lambda, enable_bundle,"
    " max_conflict_rate,"
    " data_sample_strategy, max_bin, alpha, fair_c,"
    " tweedie_variance_power, device, use_missing"
)

# Parameters that name a real LightGBM feature this parser does not cover,
# reported as unsupported instead of as unknown so the message can say why.
#
# `feature_contri`, `cegb_penalty_feature_coupled`, and
# `cegb_penalty_feature_lazy` are per-feature vectors, which a
# whitespace-separated parameter string cannot carry any more than it can
# carry `monotone_constraints`. All three are reachable through
# `TreeParams.extra.penalties` in the Mojo API: the first on `contri`, the
# other two on `penalties.cegb` (cegb.mojo).
comptime _MOJO_API_ONLY = String(
    "bagging_fraction bagging_freq bagging_seed pos_bagging_fraction"
    " neg_bagging_fraction top_rate other_rate boosting boosting_type"
    " monotone_constraints interaction_constraints"
    " categorical_feature cat_smooth cat_l2 max_cat_threshold"
    " max_cat_to_onehot min_data_per_group early_stopping_round"
    " early_stopping_rounds first_metric_only lambdarank_truncation_level"
    " label_gain sigmoid eval_at ndcg_eval_at class_weight is_unbalance"
    " unbalance unbalanced_sets scale_pos_weight feature_contri"
    " feature_contrib fc fp feature_penalty cegb_penalty_feature_coupled"
    " cegb_penalty_feature_lazy"
)


struct TrainConfig(Copyable, Movable):
    """A parsed parameter string, ready to hand to `fit`/`fit_multiclass`.

    `objective` is a boosting.mojo objective code, or `MULTICLASS`, in which
    case `n_classes` is the class count and the caller trains with
    `fit_multiclass`. `n_classes` is 1 for every single-output objective.
    """

    var objective: Int
    var n_classes: Int
    var booster: BoosterParams
    var max_bin: Int
    var alpha: Float64
    var device: Int
    var use_missing: Bool

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

    def is_multiclass(self) -> Bool:
        return self.objective == MULTICLASS


def objective_from_name(name: String) raises -> Int:
    """Objective code for a LightGBM objective name, or `MULTICLASS`.

    Accepts LightGBM's aliases for the objectives mojotrees implements.
    Names are canonical lowercase, as in `parse_device`.

    The LightGBM objectives mojotrees does not implement are named
    explicitly in `_unimplemented_objective_error` rather than falling into
    the unknown-name message: a user who asks for `multiclassova` has asked
    for a real thing, and being told it is unknown would be misleading.
    """
    if (
        name == "regression"
        or name == "regression_l2"
        or name == "l2"
        or name == "mean_squared_error"
        or name == "mse"
    ):
        return SQUARED_ERROR
    if name == "binary":
        return BINARY_LOGISTIC
    if name == "poisson":
        return POISSON
    if name == "huber":
        return HUBER
    if name == "quantile":
        return QUANTILE
    if (
        name == "mae"
        or name == "regression_l1"
        or name == "l1"
        or name == "mean_absolute_error"
    ):
        return L1
    if name == "gamma":
        return GAMMA
    if name == "tweedie":
        return TWEEDIE
    if name == "mape" or name == "mean_absolute_percentage_error":
        return MAPE
    if name == "fair":
        return FAIR
    if name == "cross_entropy" or name == "xentropy":
        return CROSS_ENTROPY
    if name == "multiclass" or name == "softmax":
        return MULTICLASS
    if name == "lambdarank":
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
        if (key == "objective" or key == "application") and (
            value == "lambdarank" or value == "custom"
        ):
            return True
        # `data_sample_strategy=bagging` is accepted; only the GOSS value
        # needs the Mojo API, because selecting it means handing the trainer
        # a `GossParams`.
        if key == "data_sample_strategy" and value == "goss":
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


def _validate(config: TrainConfig, saw_num_class: Bool) raises:
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
    config.booster.tree.extra.check_scalars(
        config.booster.tree.min_data_in_leaf
    )
    # Exclusive feature bundling: the knobs are range-checked whether or not
    # the switch is on, so a bad value is named here rather than at the first
    # training call that happens to turn bundling on, and the switch itself is
    # checked against the device that would have to honor it.
    check_bundling_supported(
        config.booster.bundling.enabled, config.device == CPU_DEVICE
    )
    config.booster.bundling.check()
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

        if key == "objective" or key == "application":
            config.objective = objective_from_name(value)
        elif key == "num_class" or key == "num_classes":
            config.n_classes = _parse_int(key, value)
            saw_num_class = True
        elif (
            key == "num_iterations"
            or key == "num_iteration"
            or key == "n_estimators"
            or key == "num_round"
            or key == "num_rounds"
            or key == "num_boost_round"
        ):
            config.booster.n_estimators = _parse_int(key, value)
        elif (
            key == "learning_rate"
            or key == "shrinkage_rate"
            or key == "eta"
        ):
            config.booster.learning_rate = _parse_f64(key, value)
        elif key == "num_leaves" or key == "num_leaf":
            config.booster.tree.num_leaves = _parse_int(key, value)
        elif (
            key == "min_data_in_leaf"
            or key == "min_data"
            or key == "min_child_samples"
        ):
            config.booster.tree.min_data_in_leaf = _parse_int(key, value)
        elif (
            key == "min_sum_hessian_in_leaf"
            or key == "min_sum_hessian"
            or key == "min_child_weight"
        ):
            config.booster.tree.min_child_hess = _parse_f64(key, value)
        elif key == "lambda_l1" or key == "reg_alpha":
            config.booster.tree.lambda_l1 = _parse_f64(key, value)
        elif key == "lambda_l2" or key == "reg_lambda" or key == "lambda":
            config.booster.tree.lambda_reg = _parse_f64(key, value)
        elif key == "max_depth":
            config.booster.tree.max_depth = _parse_int(key, value)
        elif key == "grow_policy":
            # XGBoost's name and spellings; LightGBM has no such switch, so
            # this is an extension rather than a parity row
            # (growth_policy.mojo). `depthwise` commits a depth at a time.
            config.booster.tree.grow_policy = parse_grow_policy(value)
        elif (
            key == "feature_fraction"
            or key == "sub_feature"
            or key == "colsample_bytree"
        ):
            config.booster.tree.feature_fraction = _parse_f64(key, value)
        elif key == "feature_fraction_bynode" or key == "colsample_bynode":
            config.booster.tree.feature_fraction_bynode = _parse_f64(
                key, value
            )
        elif (
            key == "feature_fraction_bylevel" or key == "colsample_bylevel"
        ):
            # XGBoost's name; LightGBM has no per-level fraction at all, so
            # this is an extension rather than a parity row (sampling.mojo).
            config.booster.tree.feature_fraction_bylevel = _parse_f64(
                key, value
            )
        elif key == "feature_fraction_seed":
            config.booster.tree.feature_fraction_seed = _parse_int(key, value)
        elif key == "min_gain_to_split" or key == "min_split_gain":
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
        elif key == "max_bin":
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
        elif key == "device" or key == "device_type":
            config.device = parse_device(value)
        elif key == "use_missing":
            config.use_missing = _parse_bool(key, value)
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
    _validate(config, saw_num_class)
    return config^
