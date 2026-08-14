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
"""

from .boosting import (
    BINARY_LOGISTIC,
    BoosterParams,
    HUBER,
    L1,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
)
from .device import CPU_DEVICE, parse_device
from .tree import TreeParams

# `TrainConfig.objective` when the parameter string selects softmax
# multiclass, which `fit_multiclass` handles instead of `fit`. Multiclass is
# not one of the single-output objective codes in boosting.mojo, so it needs
# a value of its own; it is negative to keep it out of that space forever.
comptime MULTICLASS = -1

# Every key `parse_params` accepts, primary names only, for error messages.
comptime SUPPORTED_KEYS = String(
    "objective, num_class, num_iterations, learning_rate, num_leaves,"
    " min_data_in_leaf, min_sum_hessian_in_leaf, lambda_l1, lambda_l2,"
    " max_depth, feature_fraction, feature_fraction_bynode,"
    " feature_fraction_seed, max_bin, alpha, device, use_missing"
)

# Parameters that name a real LightGBM feature this parser does not cover,
# reported as unsupported instead of as unknown so the message can say why.
comptime _MOJO_API_ONLY = String(
    "bagging_fraction bagging_freq bagging_seed pos_bagging_fraction"
    " neg_bagging_fraction top_rate other_rate boosting boosting_type"
    " data_sample_strategy monotone_constraints interaction_constraints"
    " categorical_feature cat_smooth cat_l2 max_cat_threshold"
    " max_cat_to_onehot min_data_per_group early_stopping_round"
    " early_stopping_rounds first_metric_only lambdarank_truncation_level"
    " label_gain sigmoid eval_at ndcg_eval_at"
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
        """LightGBM's defaults as mojoboost sets them: squared error on the
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

    Accepts LightGBM's aliases for the objectives mojoboost implements.
    Names are canonical lowercase, as in `parse_device`.
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
    raise Error(
        "unknown objective '",
        name,
        "'; expected regression, binary, multiclass, poisson, huber,"
        " quantile, or mae",
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
    """Whether `spec` names something mojoboost implements but parameter
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
    return False


def _validate(config: TrainConfig, saw_num_class: Bool) raises:
    """Range checks that do not depend on the training data. Objective
    specific checks on `alpha` and on the label values stay in
    boosting.mojo, which sees the labels."""
    if config.booster.n_estimators < 0:
        raise Error("num_iterations must be nonnegative")
    if config.booster.learning_rate <= 0.0:
        raise Error("learning_rate must be positive")
    if config.booster.tree.num_leaves < 2:
        raise Error("num_leaves must be at least 2")
    if config.booster.tree.min_data_in_leaf < 0:
        raise Error("min_data_in_leaf must be nonnegative")
    if config.booster.tree.min_child_hess < 0.0:
        raise Error("min_sum_hessian_in_leaf must be nonnegative")
    if config.booster.tree.lambda_l1 < 0.0:
        raise Error("lambda_l1 must be nonnegative")
    if config.booster.tree.lambda_reg < 0.0:
        raise Error("lambda_l2 must be nonnegative")
    if (
        config.booster.tree.feature_fraction <= 0.0
        or config.booster.tree.feature_fraction > 1.0
    ):
        raise Error("feature_fraction must be in (0, 1]")
    if (
        config.booster.tree.feature_fraction_bynode <= 0.0
        or config.booster.tree.feature_fraction_bynode > 1.0
    ):
        raise Error("feature_fraction_bynode must be in (0, 1]")
    if config.max_bin < 2:
        raise Error("max_bin must be at least 2")

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
        elif key == "feature_fraction_seed":
            config.booster.tree.feature_fraction_seed = _parse_int(key, value)
        elif key == "max_bin":
            config.max_bin = _parse_int(key, value)
        elif key == "alpha":
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
            raise Error(
                "unknown parameter '", key, "'; supported: ", SUPPORTED_KEYS
            )

    _validate(config, saw_num_class)
    return config^
