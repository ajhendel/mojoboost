"""One registry for what objectives and metrics *are*.

Every objective and every built-in metric carries facts that are not
computation: a canonical name and its aliases, the task it belongs to, the
inverse link its predictions go through, the scalar parameter it reads,
whether it replaces its leaf values after each tree, which backends can
train it, and, for a metric, which direction is better, which objective's
loss it is the default for, and what it needs beyond predictions and labels
(a class count, query groups, a cutoff, the objective's scalar parameter).

Before this module those facts were spread across four places that had to
be edited together: `objective_from_name` / `objective_display_name` /
`objective_default_alpha` in params.mojo, `supports_device_objective` in
gpu_objectives_native.mojo, the metric code table in
bindings/_mojoboost.mojo, and `_METRICS` / `_ALIASES` / `_DEFAULTS` in
python/mojoboost/_eval.py. The Python tables are the ones that hurt: they
are a second semantic copy in a second language, so a metric added in Mojo
is invisible from Python until someone remembers, and a direction or a task
can disagree with the code that computes the number. This module is the one
place those facts live; Python's job shrinks to translating Python callables
and formatting error messages.

What this module is not
-----------------------
It is metadata and dispatch *selection*, not a second implementation.
Gradients stay in boosting.mojo, ranking lambdas in ranking.mojo, metric
arithmetic in metrics.mojo, the device kernels in
gpu_objectives_native.mojo. Nothing here computes anything a trainer needs
per row or per round.

It is also not the code-to-call map. `eval_metric_by_code` in metrics.mojo
turns a metric code into a call, and `eval_builtin_metric` in
custom_metric.mojo wraps that with the transform, the class count, and the
query groups this module says each code needs. Both read their rules from
here; neither is duplicated here.

Custom objectives (`CUSTOM`, objective.mojo) and custom metrics
(`CustomMetric`, custom_metric.mojo) are deliberately *outside* the
registry. A caller-supplied callable has no canonical name, no alias, no
link the framework knows, and no default metric, which is exactly why it
carries its own metadata with it. `objective_is_builtin` and
`metric_is_builtin` are the boundary: everything they reject is a callable
whose metadata its author supplies.

Why plain query functions
-------------------------
Every query here is a pure function of an `Int` code over `comptime`
constants: no trait objects, no vtable, no `Dict`, and no allocation except
in the four functions that return a name or a list. Called with a
compile-time-known objective (`objective_link(SQUARED_ERROR)`) the whole
if-chain folds; called with a runtime code it is a handful of integer
compares. That is the reason the registry is functions rather than a table
of `ObjectiveSpec` values behind a trait: a trait would put an indirect call
where a compare is, and a stored table would put a load and a bounds check
there. `ObjectiveSpec` and `MetricSpec` exist for the callers that want
every field at once (a binding marshalling one dict, a Python facade
answering one query); they hold only scalars, so building one allocates
nothing.

No registry call belongs inside a per-row or per-round loop. The trainers
ask once per run, before the first tree.

Temporary mirrors
-----------------
Three things below are duplicated from files this lane does not own, so the
duplicate is spelled out rather than hidden:

- `MULTICLASS` mirrors params.mojo's, `LAMBDARANK` mirrors ranking.mojo's.
- The objective name and alias resolution mirrors `objective_from_name`,
  `objective_display_name`, `objective_default_alpha`, and
  `_raise_if_unimplemented_objective` in params.mojo, message text
  included, so params.mojo can be reduced to delegation without changing a
  single error a user sees.
- `objective_gradients_on_device` mirrors `supports_device_objective` in
  gpu_objectives_native.mojo. It is not imported from there because that
  module pulls in `max.gpu.*`, and a metadata module that params.mojo and
  the CLI depend on must not drag the GPU stack behind it. The dependency
  runs the other way after wiring.

Each mirror is listed in handoffs/migration_21_objective_metric_registry.md
with the exact deletion that removes it. Until those deletions land, this
module is authoritative by intent and the mirrored files are authoritative
in fact.
"""

from .boosting import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    CUSTOM,
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
    _check_objective,
    objective_renews_leaves,
)
from .metrics import (
    METRIC_AUC,
    METRIC_AVERAGE_PRECISION,
    METRIC_BINARY_ERROR,
    METRIC_BINARY_LOGLOSS,
    METRIC_CROSS_ENTROPY,
    METRIC_FAIR,
    METRIC_GAMMA,
    METRIC_GAMMA_DEVIANCE,
    METRIC_HUBER,
    METRIC_KLDIV,
    METRIC_L1,
    METRIC_L2,
    METRIC_MAP,
    METRIC_MAPE,
    METRIC_MULTI_ERROR,
    METRIC_MULTI_LOGLOSS,
    METRIC_NDCG,
    METRIC_POISSON,
    METRIC_QUANTILE,
    METRIC_RMSE,
    METRIC_TWEEDIE,
    N_BUILTIN_METRICS,
)

# Two imports rather than two more tables, and both are re-exported so a
# caller that wants objective or metric metadata needs one import.
#
# `objective_renews_leaves` is the boosting loop's own function: it is called
# once per training run there and the registry answers with the same
# function, so the LightGBM `RenewTreeOutput` rule has exactly one
# definition.
#
# The `METRIC_*` codes and `N_BUILTIN_METRICS` live in metrics.mojo because a
# code names a function and nineteen of those functions are in that file.
# This module owns what a code *means*; metrics.mojo owns what it computes.
# The dependency runs registry -> metrics and never back: metrics.mojo cannot
# import this module, since boosting.mojo imports `_argsort` from it and this
# module imports boosting.

# ---------------------------------------------------------------------------
# Objective codes
# ---------------------------------------------------------------------------

# The single-output objective codes are boosting.mojo's, imported above. Two
# more codes complete the space; both are mirrors (see the module docstring).

# Softmax multiclass, params.mojo's `MULTICLASS`. Negative to stay out of the
# single-output code space forever: multiclass is trained by
# `train_multiclass`, not by `train`, and a model of it holds one tree per
# class per round.
comptime MULTICLASS = -1

# LambdaRank, ranking.mojo's `LAMBDARANK`. Its gradients come from query
# groups rather than from `_fill_grad_hess`, which is why it lives with the
# ranking code, but it is one of the objective codes a fitted model carries.
comptime LAMBDARANK = 7

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

# The task an objective trains and a metric can score. An estimator accepts
# only the metrics of its own task, which is stricter than LightGBM (it
# scores whatever it can) and is the rule `_eval.resolve` enforces from
# Python. `task_name` spells these exactly as the Python layer does.
comptime TASK_REGRESSION = 0
comptime TASK_BINARY = 1
comptime TASK_MULTICLASS = 2
comptime TASK_RANKING = 3

# The inverse link a raw score goes through to become a prediction. These
# are the transforms `Booster.response`, `response_scale` in
# custom_metric.mojo, and `response_for_objective` in gpu_predict.mojo apply;
# all three read the objective, so all three must read the same table.
comptime LINK_IDENTITY = 0
comptime LINK_SIGMOID = 1
comptime LINK_EXP = 2
comptime LINK_SOFTMAX = 3

# The objective's one scalar parameter, under LightGBM's name for it. The
# trainer has a single `alpha` slot whichever name the user spells (see the
# boosting.mojo docstring); this says which name that slot answers to, and
# `PARAM_NONE` says the objective reads no scalar at all.
comptime PARAM_NONE = 0
comptime PARAM_ALPHA = 1
comptime PARAM_FAIR_C = 2
comptime PARAM_TWEEDIE_VARIANCE_POWER = 3

# Backends that can train an objective, as bit flags. Not to be confused
# with backend.mojo's `CPU`/`GPU`, which are the histogram dispatch codes 0
# and 1; these OR together because an objective usually has both.
comptime SUPPORTS_CPU = 1
comptime SUPPORTS_GPU = 2

# What a metric needs besides predictions, labels, and weights. Bit flags,
# because a ranking metric needs two of them.
comptime NEEDS_NOTHING = 0
comptime NEEDS_PARAM = 1
comptime NEEDS_N_CLASSES = 2
comptime NEEDS_GROUPS = 4
comptime NEEDS_CUTOFF = 8

# What a metric expects predictions to have been through. Every single-output
# metric scores the objective's response scale, the multiclass metrics take
# the softmax of the raw row, and the ranking metrics read raw scores because
# only their order matters. This is the branch `eval_metric` in
# bindings/_mojoboost.mojo takes by hand.
comptime TRANSFORM_OBJECTIVE_LINK = 0
comptime TRANSFORM_SOFTMAX = 1
comptime TRANSFORM_RAW = 2

# How a name a user typed relates to what mojoboost implements.
comptime NAME_SUPPORTED = 0
comptime NAME_UNIMPLEMENTED = 1
comptime NAME_UNKNOWN = 2

# Where an objective's per-row first and second derivatives come from. This
# is the question a trainer asks to know which loop it is in, and the one a
# device provider asks before offering to compute a round on the GPU. It is
# finer than `objective_is_builtin`, which answers only "does
# `fill_grad_hess` handle it".
comptime GRAD_BUILTIN = 0
"""`fill_grad_hess` in boosting.mojo, one closed form per row."""
comptime GRAD_SOFTMAX = 1
"""`_fill_softmax_grad_hess` in boosting.mojo, one block of `n_classes` per
row, reached through `train_multiclass`."""
comptime GRAD_LAMBDARANK = 2
"""`_fill_lambdas` in ranking.mojo, pairwise within a query group."""
comptime GRAD_CALLBACK = 3
"""A caller-supplied `GradHessFn`, on the host, once per round. There is no
closed form the framework knows, which is why `CUSTOM` has no device
gradient kernel and no base score the framework can pick."""

# Where an objective's starting raw score comes from before the first tree.
# The trainers compute it (`_base_score` in boosting.mojo,
# `_multiclass_base_scores` in custom_metric.mojo, the literal 0.0 in
# ranking.mojo); this says which of the five rules each objective follows, so
# a caller can answer "what does this model boost from" without running a
# trainer, and so a custom objective can be pointed at the matching rule
# (`matching_base_score` in objective.mojo).
comptime INIT_LINK_MEAN = 0
"""The link of the weighted label mean: the mean itself under the identity
link, its logit under the sigmoid, its logarithm under exp."""
comptime INIT_LABEL_PERCENTILE = 1
"""A weighted percentile of the labels, which is what the objectives that
renew their leaves boost from because their hessian carries no curvature."""
comptime INIT_LOG_CLASS_PRIOR = 2
"""One score per class, the logarithm of that class's weighted prior."""
comptime INIT_ZERO = 3
"""Zero, because the scale is arbitrary: a ranking score means nothing but
its order within a query."""
comptime INIT_CALLER = 4
"""Whatever the caller passes as `base_score`, defaulting to 0.0. The
framework does not know a custom objective's link, so it cannot average
anything sensible; LightGBM's `boost_from_average` likewise does not apply to
a custom objective."""

# Which class-level weighting policy applies to an objective. Every policy is
# expanded into ordinary per-row `sample_weight` by class_weight.mojo before a
# trainer sees it, so this decides which *arguments* a caller may pass, never
# how the weights are applied.
comptime CLASS_WEIGHT_NONE = 0
"""No class exists to weight: the labels are real numbers (regression, cross
entropy's soft targets, a custom objective's) or graded relevances."""
comptime CLASS_WEIGHT_BINARY = 1
"""Two classes, so LightGBM's `scale_pos_weight` and `is_unbalance` apply as
well as an explicit two-entry `class_weight` and `balanced`."""
comptime CLASS_WEIGHT_MULTICLASS = 2
"""`n_classes` classes, so an explicit per-class `class_weight` and
`balanced` apply; `scale_pos_weight` and `is_unbalance` do not, since there
is no single positive class to lift."""

# The metric codes are imported from metrics.mojo above and re-exported, so
# `objective_registry.METRIC_L2` and `metrics.METRIC_L2` are the same
# constant rather than two that have to agree. They are the codes
# `eval_metric` in bindings/_mojoboost.mojo dispatches on and
# python/mojoboost/_eval.py mirrors, and must not be renumbered: a code
# crosses the Python boundary as an integer.

# The objective names the registry resolves, primary spellings only, for the
# unknown-name message. params.mojo names a shorter list because a parameter
# string cannot carry `lambdarank` or `custom`; that list stays there.
comptime KNOWN_OBJECTIVE_NAMES = String(
    "regression, binary, multiclass, poisson, huber, quantile, mae, gamma,"
    " tweedie, mape, fair, cross_entropy, lambdarank, or custom"
)

# Metric names per task, sorted, as the Python error messages print them.
# Sorted rather than declaration order so the message is stable and matches
# `", ".join(sorted(...))` on the Python side exactly.
comptime REGRESSION_METRIC_NAMES = String(
    "cross_entropy, fair, gamma, gamma_deviance, huber, kullback_leibler,"
    " l1, l2, mape, poisson, quantile, rmse, tweedie"
)
comptime BINARY_METRIC_NAMES = String(
    "auc, average_precision, binary_error, binary_logloss"
)
comptime MULTICLASS_METRIC_NAMES = String("multi_error, multi_logloss")
comptime RANKING_METRIC_NAMES = String("map, ndcg")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def task_name(task: Int) raises -> String:
    """The task's name, spelled as the Python layer spells it."""
    if task == TASK_REGRESSION:
        return String("regression")
    if task == TASK_BINARY:
        return String("binary")
    if task == TASK_MULTICLASS:
        return String("multiclass")
    if task == TASK_RANKING:
        return String("ranking")
    raise Error("unknown task code ", task)


def task_from_name(name: String) raises -> Int:
    """The task code for a task name."""
    if name == "regression":
        return TASK_REGRESSION
    if name == "binary":
        return TASK_BINARY
    if name == "multiclass":
        return TASK_MULTICLASS
    if name == "ranking":
        return TASK_RANKING
    raise Error(
        "unknown task '",
        name,
        "'; expected regression, binary, multiclass, or ranking",
    )

# ---------------------------------------------------------------------------
# Objectives: identity and membership
# ---------------------------------------------------------------------------


@always_inline
def objective_is_builtin(objective: Int) -> Bool:
    """Whether `train` and `train_gpu` compute this objective's gradients
    themselves.

    False for `CUSTOM`, whose gradients come from a caller-supplied
    callable, for `LAMBDARANK`, whose come from query groups, and for
    `MULTICLASS`, which `train_multiclass` grows one tree per class for.
    This is the same set `_check_objective` accepts, so a code this rejects
    is a code the single-output trainer refuses.
    """
    return (
        objective == SQUARED_ERROR
        or objective == BINARY_LOGISTIC
        or objective == POISSON
        or objective == HUBER
        or objective == QUANTILE
        or objective == L1
        or objective == GAMMA
        or objective == TWEEDIE
        or objective == MAPE
        or objective == FAIR
        or objective == CROSS_ENTROPY
    )


@always_inline
def objective_is_known(objective: Int) -> Bool:
    """Whether this code names any objective mojoboost has: the built-ins
    plus the three that need their own trainer (`CUSTOM`, `LAMBDARANK`,
    `MULTICLASS`)."""
    return (
        objective_is_builtin(objective)
        or objective == CUSTOM
        or objective == LAMBDARANK
        or objective == MULTICLASS
    )


@always_inline
def objective_is_multi_output(objective: Int) -> Bool:
    """Whether one boosting iteration grows more than one tree, which is
    what makes a model's tree count the round count times the class count
    rather than the round count."""
    return objective == MULTICLASS


def objective_task(objective: Int) raises -> Int:
    """The task this objective trains, which decides the metrics that can
    score it.

    `CROSS_ENTROPY` is a regression objective, not a binary one: its labels
    are soft targets anywhere in [0, 1] rather than classes, which is why
    the Python regressor owns it and the classifier does not. `CUSTOM` is
    reported as regression because that is the family its single real-valued
    output belongs to and the metric set a custom-objective model can be
    scored with; it has no *built-in* loss, which `objective_default_metric`
    is where that shows up.
    """
    if objective == BINARY_LOGISTIC:
        return TASK_BINARY
    if objective == MULTICLASS:
        return TASK_MULTICLASS
    if objective == LAMBDARANK:
        return TASK_RANKING
    if objective == CUSTOM or objective_is_builtin(objective):
        return TASK_REGRESSION
    raise Error("unknown objective code ", objective)


def objective_canonical_name(objective: Int) raises -> String:
    """The LightGBM name mojoboost reports this objective under.

    Two choices here are worth knowing. `L1` reports as `mae` rather than
    LightGBM's own canonical `regression_l1`, which is what
    `objective_display_name` in params.mojo has always reported and what
    error messages therefore say; `lgbm_objective_name` in
    lgbm_model_io.mojo writes `regression_l1` into a LightGBM model file,
    because that is the spelling LightGBM's reader expects. Both spellings
    resolve back to `L1`, so nothing round-trips wrong, but the two
    functions disagree about which is the name.
    """
    if objective == SQUARED_ERROR:
        return String("regression")
    if objective == BINARY_LOGISTIC:
        return String("binary")
    if objective == POISSON:
        return String("poisson")
    if objective == HUBER:
        return String("huber")
    if objective == QUANTILE:
        return String("quantile")
    if objective == L1:
        return String("mae")
    if objective == GAMMA:
        return String("gamma")
    if objective == TWEEDIE:
        return String("tweedie")
    if objective == MAPE:
        return String("mape")
    if objective == FAIR:
        return String("fair")
    if objective == CROSS_ENTROPY:
        return String("cross_entropy")
    if objective == MULTICLASS:
        return String("multiclass")
    if objective == LAMBDARANK:
        return String("lambdarank")
    if objective == CUSTOM:
        return String("custom")
    raise Error("unknown objective code ", objective)


def objective_unimplemented_canonical(name: String) -> String:
    """The primary spelling of a LightGBM objective mojoboost has not
    implemented, or an empty string when `name` is not one of them.

    An alias reports under its primary name, so `xentlambda` is reported as
    `cross_entropy_lambda`: the user asked for one thing under two spellings
    and the answer should name the thing.
    """
    if name == "cross_entropy_lambda" or name == "xentlambda":
        return String("cross_entropy_lambda")
    if (
        name == "multiclassova"
        or name == "multiclass_ova"
        or name == "ova"
        or name == "ovr"
    ):
        return String("multiclassova")
    if name == "rank_xendcg" or name == "xendcg":
        return String("rank_xendcg")
    return String("")


def objective_unimplemented_reason(name: String) -> String:
    """Why a LightGBM objective mojoboost has not implemented is a
    deliberate omission rather than an oversight, or an empty string when
    `name` names something else.

    These are the reasons `_raise_if_unimplemented_objective` in params.mojo
    raises with and `_UNIMPLEMENTED_OBJECTIVES` in
    python/mojoboost/__init__.py repeats in its own words. Each is also a
    row in docs/LIGHTGBM_PARITY.md. Returning the reason rather than raising
    it lets a caller build its own sentence around it, which is what the
    Python estimators need: the same fact reaches a user as "not
    implemented" from a parameter string and as "use MojoBoostClassifier"
    from an estimator.
    """
    if name == "cross_entropy_lambda" or name == "xentlambda":
        return String(
            "it parameterizes the rate through log1p(exp(raw)) rather than"
            " the logistic, so it is a separate link, not an alias of"
            " 'cross_entropy'"
        )
    if (
        name == "multiclassova"
        or name == "multiclass_ova"
        or name == "ova"
        or name == "ovr"
    ):
        return String(
            "one-vs-rest needs an independent binary model per class, which"
            " is a different trainer from the shared-softmax 'multiclass'"
        )
    if name == "rank_xendcg" or name == "xendcg":
        return String(
            "'lambdarank' is the ranking objective mojoboost provides"
        )
    return String("")


def objective_code_from_name(name: String) raises -> Int:
    """The objective code for a LightGBM objective name or alias.

    Every public spelling resolves, including the three that need a trainer
    of their own: `multiclass`/`softmax`, `lambdarank`, and `custom`. Names
    are canonical lowercase, as in `parse_device`; a caller that accepts
    mixed case lowercases first.

    Callers that cannot honor all three narrow the result rather than the
    input, so the name table stays in one place: `objective_from_name` in
    params.mojo resolves here and then refuses `lambdarank` and `custom`,
    because a parameter string can carry neither query groups nor a gradient
    callback.

    A LightGBM objective mojoboost has not implemented is reported by name
    with its reason (see `objective_unimplemented_reason`) rather than as an
    unknown name: a user who asks for `multiclassova` has asked for a real
    thing.
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
        return LAMBDARANK
    if name == "custom":
        return CUSTOM

    var unimplemented = objective_unimplemented_canonical(name)
    if unimplemented.byte_length() > 0:
        raise Error(
            "objective '",
            unimplemented,
            "' is not implemented; ",
            objective_unimplemented_reason(name),
        )
    raise Error(
        "unknown objective '", name, "'; expected ", KNOWN_OBJECTIVE_NAMES
    )


def objective_name_status(name: String) -> Int:
    """Whether a name is supported, is a real LightGBM objective mojoboost
    has not implemented, or is neither.

    Never raises, so it is safe to call while already building an error
    message. It is the query a Python estimator needs: the estimator's own
    message depends on which of the three it is, and on whether the
    objective belongs to a different estimator, which `objective_task`
    answers separately.
    """
    if objective_unimplemented_canonical(name).byte_length() > 0:
        return NAME_UNIMPLEMENTED
    try:
        _ = objective_code_from_name(name)
    except:
        return NAME_UNKNOWN
    return NAME_SUPPORTED


# ---------------------------------------------------------------------------
# Objectives: prediction, parameters, and leaf renewal
# ---------------------------------------------------------------------------


@always_inline
def objective_link(objective: Int) -> Int:
    """The inverse link between a raw score and a prediction.

    The logistic for binary and cross entropy, `exp` for the three
    log-linked regression objectives, softmax for multiclass, and the
    identity for the rest, `CUSTOM` included: the framework does not know a
    caller's link, so it hands back the raw score and the caller applies
    their own. `LAMBDARANK` is the identity because a ranking score is
    meaningful only in its order.

    This is the fact `Booster.response`, `response_scale` in
    custom_metric.mojo, and `response_for_objective` in gpu_predict.mojo
    each decide for themselves today. They agree; the point of naming it
    here is that they cannot stop agreeing.
    """
    if objective == BINARY_LOGISTIC or objective == CROSS_ENTROPY:
        return LINK_SIGMOID
    if (
        objective == POISSON
        or objective == GAMMA
        or objective == TWEEDIE
    ):
        return LINK_EXP
    if objective == MULTICLASS:
        return LINK_SOFTMAX
    return LINK_IDENTITY


@always_inline
def objective_param(objective: Int) -> Int:
    """Which scalar parameter this objective reads, or `PARAM_NONE`."""
    if objective == HUBER or objective == QUANTILE:
        return PARAM_ALPHA
    if objective == FAIR:
        return PARAM_FAIR_C
    if objective == TWEEDIE:
        return PARAM_TWEEDIE_VARIANCE_POWER
    return PARAM_NONE


def objective_param_name(objective: Int) -> String:
    """LightGBM's name for the objective's scalar parameter, or an empty
    string when it reads none.

    This is what makes "parameter 'fair_c' does not apply to objective
    'tweedie'; it takes 'tweedie_variance_power'" possible in one place:
    the three names land in one trainer slot, so accepting the wrong one
    would train with a number the user did not ask for.
    """
    var kind = objective_param(objective)
    if kind == PARAM_ALPHA:
        return String("alpha")
    if kind == PARAM_FAIR_C:
        return String("fair_c")
    if kind == PARAM_TWEEDIE_VARIANCE_POWER:
        return String("tweedie_variance_power")
    return String("")


@always_inline
def objective_default_param(objective: Int) -> Float64:
    """The value the objective's scalar slot takes when nobody sets one:
    LightGBM's `fair_c` and `tweedie_variance_power` defaults for those two,
    and LightGBM's `alpha` default of 0.9 otherwise, which matters for huber
    and quantile and is ignored by everything else.

    The same rule as `objective_default_alpha` in params.mojo, over the same
    two constants from boosting.mojo, extended to `LAMBDARANK`, `CUSTOM`,
    and `MULTICLASS`, which read no scalar and take the shared default.
    """
    if objective == FAIR:
        return DEFAULT_FAIR_C
    if objective == TWEEDIE:
        return DEFAULT_TWEEDIE_VARIANCE_POWER
    return 0.9


@fieldwise_init
struct ParamDomain(Copyable, Movable):
    """The interval an objective's scalar parameter must lie in.

    Declarative, for the callers that need the domain rather than a verdict:
    a Python estimator validating a keyword before it reaches Mojo, a binding
    filling a parameter description, a document listing what `tweedie` takes.
    `check_objective_param` remains the *enforcing* answer, and it delegates
    to the trainer's own check so the message a user sees is the trainer's.

    `applies` is False for the ten objectives that read no scalar, and every
    other field is then meaningless. `has_upper` is False for the two domains
    that are unbounded above (`huber`'s alpha and `fair`'s c), where `upper`
    is 0.0 and must not be read.
    """

    var applies: Bool
    var lower: Float64
    var lower_open: Bool
    var has_upper: Bool
    var upper: Float64
    var upper_open: Bool
    var default: Float64

    @staticmethod
    def none() -> Self:
        """The domain of an objective that reads no scalar."""
        return Self(False, 0.0, False, False, 0.0, False, 0.0)

    def contains(self, value: Float64) -> Bool:
        """Whether `value` lies in the domain. Always True when the
        objective reads no scalar: there is nothing to be outside of."""
        if not self.applies:
            return True
        if self.lower_open:
            if not value > self.lower:
                return False
        elif value < self.lower:
            return False
        if not self.has_upper:
            return True
        if self.upper_open:
            return value < self.upper
        return value <= self.upper

    def describe(self) -> String:
        """The domain in interval notation, for an error message or a
        document: `(0.0, 1.0)`, `(0.0, inf)`, `(1.0, 2.0)`. An objective that
        reads no scalar describes as the empty string."""
        if not self.applies:
            return String("")
        var out = String("[")
        if self.lower_open:
            out = String("(")
        out += String(self.lower)
        out += ", "
        if not self.has_upper:
            return out + "inf)"
        out += String(self.upper)
        if self.upper_open:
            return out + ")"
        return out + "]"


def objective_param_domain(objective: Int) raises -> ParamDomain:
    """The interval this objective's scalar parameter must lie in.

    The four domains are LightGBM's: `huber`'s alpha and `fair`'s c are
    positive, `quantile`'s alpha is a probability strictly inside (0, 1), and
    `tweedie`'s variance power is strictly inside (1, 2), where the
    distribution is the compound Poisson-gamma the objective assumes (at 1 it
    is Poisson and at 2 gamma, and its gradient divides by neither exponent).

    These bounds are stated in `_check_objective` in boosting.mojo as well,
    which is the enforcing copy. `check_objective_param` runs both and raises
    if they ever disagree, so the duplication is loud rather than silent; see
    the handoff for the deletion that removes it.
    """
    var kind = objective_param(objective)
    if kind == PARAM_NONE:
        return ParamDomain.none()
    if objective == QUANTILE:
        return ParamDomain(True, 0.0, True, True, 1.0, True, 0.9)
    if objective == HUBER:
        return ParamDomain(True, 0.0, True, False, 0.0, False, 0.9)
    if objective == FAIR:
        return ParamDomain(
            True, 0.0, True, False, 0.0, False, DEFAULT_FAIR_C
        )
    if objective == TWEEDIE:
        return ParamDomain(
            True, 1.0, True, True, 2.0, True, DEFAULT_TWEEDIE_VARIANCE_POWER
        )
    raise Error("unknown objective code ", objective)


def check_objective_param(objective: Int, value: Float64) raises:
    """Validate the objective's scalar parameter without looking at data.

    Delegates to `_check_objective` with no labels, which runs exactly the
    parameter range checks and skips every label check, so the message a
    user sees is the trainer's own and there is no second copy of the ranges
    to drift. Objectives with no scalar, and the three with their own
    trainers, are accepted unexamined: their trainers validate what they
    read.

    `objective_param_domain` states the same four intervals declaratively,
    for the callers that need the bounds rather than a verdict. That is a
    second statement of the same fact, so this function checks it: a value
    the trainer accepts and the domain rejects raises here rather than
    reaching a Python layer that would reject it later for its own reasons.
    The check costs one comparison and runs once per training run.
    """
    if not objective_is_builtin(objective):
        return
    _check_objective(objective, [], value)
    var domain = objective_param_domain(objective)
    if not domain.contains(value):
        raise Error(
            "registry drift: boosting.mojo accepts ",
            objective_param_name(objective),
            "=",
            value,
            " for objective '",
            objective_canonical_name(objective),
            "' but objective_param_domain says ",
            domain.describe(),
        )


@always_inline
def objective_needs_groups(objective: Int) -> Bool:
    """Whether training this objective needs query groups. LambdaRank's
    gradients are pairwise within a query, so a group array is not optional
    for it and is meaningless for everything else."""
    return objective == LAMBDARANK


def objective_grad_hess_source(objective: Int) raises -> Int:
    """Where this objective's per-row derivatives come from, as a `GRAD_*`
    code.

    Four sources, four loops. `objective_is_builtin` is this question
    collapsed to "is it `GRAD_BUILTIN`", which is what the single-output
    trainer needs; a device provider deciding whether it can own a whole
    round needs the finer answer, because `GRAD_SOFTMAX` has a device kernel
    reached through a different entry point and `GRAD_CALLBACK` has none at
    all.
    """
    if objective == MULTICLASS:
        return GRAD_SOFTMAX
    if objective == LAMBDARANK:
        return GRAD_LAMBDARANK
    if objective == CUSTOM:
        return GRAD_CALLBACK
    if objective_is_builtin(objective):
        return GRAD_BUILTIN
    raise Error("unknown objective code ", objective)


def objective_init_kind(objective: Int) raises -> Int:
    """Which of the five starting-score rules this objective follows, as an
    `INIT_*` code.

    Composed from facts already here rather than restated: an objective that
    renews its leaves boosts from a label percentile (LightGBM's rule, and
    the same percentile leaf renewal takes), softmax multiclass from the log
    class priors, ranking from zero, a custom objective from whatever its
    caller passes, and everything else from the link of the weighted label
    mean. `_base_score` in boosting.mojo computes the first and last of
    those; this only says which.
    """
    if objective == CUSTOM:
        return INIT_CALLER
    if objective == LAMBDARANK:
        return INIT_ZERO
    if objective == MULTICLASS:
        return INIT_LOG_CLASS_PRIOR
    if objective_renews_leaves(objective):
        return INIT_LABEL_PERCENTILE
    if objective_is_builtin(objective):
        return INIT_LINK_MEAN
    raise Error("unknown objective code ", objective)


# ---------------------------------------------------------------------------
# Objectives: weighting
# ---------------------------------------------------------------------------

# Every weighting policy mojoboost has ends as one `sample_weight` vector,
# expanded by class_weight.mojo before a trainer is called: a class weight
# multiplies whatever row weight the caller already had, and nothing
# downstream can tell the product apart from a weight passed by hand. That is
# what makes double application impossible rather than merely discouraged,
# and it is why nothing below returns weights. These functions say which
# *arguments* an objective accepts; class_weight.mojo says what they expand
# to, and the trainers apply the result exactly once.


def objective_class_weight_kind(objective: Int) raises -> Int:
    """Which class-level weighting arguments apply to this objective, as a
    `CLASS_WEIGHT_*` code.

    Keyed on the task, since a class is a task's notion: binary has two,
    multiclass has `n_classes`, and regression and ranking have none.
    `CROSS_ENTROPY` therefore takes none despite its [0, 1] labels, which is
    the point of it being a regression objective: a soft target of 0.3 is not
    a class that could carry a weight.
    """
    var task = objective_task(objective)
    if task == TASK_BINARY:
        return CLASS_WEIGHT_BINARY
    if task == TASK_MULTICLASS:
        return CLASS_WEIGHT_MULTICLASS
    return CLASS_WEIGHT_NONE


@always_inline
def objective_supports_scale_pos_weight(objective: Int) raises -> Bool:
    """Whether LightGBM's `scale_pos_weight` and `is_unbalance` apply.

    Binary only, as in LightGBM: both name *the* positive class, and there is
    no such thing under a multiclass, regression, or ranking objective. The
    two are mutually exclusive with each other as well, which
    `check_class_balance_params` in class_weight.mojo enforces.
    """
    return objective_class_weight_kind(objective) == CLASS_WEIGHT_BINARY


def check_objective_class_weight(
    objective: Int,
    has_class_weight: Bool,
    is_unbalance: Bool,
    scale_pos_weight: Float64,
) raises:
    """Reject a class-weighting argument the objective has no classes for,
    by name, before anything expands it into row weights.

    LightGBM ignores these silently on a regression objective, which is how a
    `scale_pos_weight=4.0` on a poisson model trains unweighted and reports
    nothing. Naming the objective and the argument is the intentional
    difference. `scale_pos_weight` is compared against 1.0 rather than tested
    for presence because 1.0 is its identity: passing it changes nothing, so
    it cannot be an error.
    """
    var kind = objective_class_weight_kind(objective)
    if kind == CLASS_WEIGHT_NONE:
        if has_class_weight:
            raise Error(
                "class_weight does not apply to objective '",
                objective_canonical_name(objective),
                "', which trains a ",
                task_name(objective_task(objective)),
                " task and has no classes to weight; use sample_weight",
            )
    if not objective_supports_scale_pos_weight(objective):
        if is_unbalance:
            raise Error(
                "is_unbalance applies to binary classification only, as in"
                " LightGBM; objective '",
                objective_canonical_name(objective),
                "' has no single positive class",
            )
        if scale_pos_weight != 1.0:
            raise Error(
                "scale_pos_weight applies to binary classification only, as"
                " in LightGBM; objective '",
                objective_canonical_name(objective),
                "' has no single positive class",
            )


# ---------------------------------------------------------------------------
# Objectives: backend support
# ---------------------------------------------------------------------------


@always_inline
def objective_gradients_on_device(objective: Int) -> Bool:
    """Whether this objective's per-row derivatives have a device kernel in
    the single-output path, so a round's gradients never cross the
    host/device boundary.

    True for every built-in single-output objective; false for `CUSTOM`,
    whose callback lives on the host with the raw scores, for `LAMBDARANK`,
    whose lambdas are pairwise within a query rather than a closed form per
    row, and for `MULTICLASS`, whose softmax derivatives have a device
    kernel of their own reached through `train_multiclass_gpu` rather than
    through the single-output `fill_gradients_device`.

    Mirrors `supports_device_objective` in gpu_objectives_native.mojo value
    for value, which is not imported here because it would drag `max.gpu.*`
    into every consumer of this module (see the module docstring).
    """
    return objective_is_builtin(objective)


@always_inline
def objective_backends(objective: Int) -> Int:
    """Which backends can train this objective, as `SUPPORTS_*` bit flags.

    The CPU trains everything. The GPU trains every built-in single-output
    objective (`train_gpu`), softmax multiclass (`train_multiclass_gpu`),
    and custom objectives (`train_custom_gpu`, which grows the trees on the
    device and calls the gradient callback on the host). LambdaRank is CPU
    only.

    A GPU flag here is a claim about a *trainer existing*, not about the
    gradient kernel and not about which entry point accepts the objective.
    Two narrower questions have their own answers and deliberately differ
    from this one on `CUSTOM` and `MULTICLASS`:

    - `objective_gradients_on_device` asks whether the derivatives are
      computed on the device, which a custom objective's never are.
    - `gpu_trains_objective` in device_policy.mojo asks whether `train_gpu`
      *itself* accepts the code, which it does not for either: they have
      their own entry points, `train_custom_gpu` and `train_multiclass_gpu`.

    Do not treat the three as the same predicate. See the handoff.
    """
    if objective == LAMBDARANK:
        return SUPPORTS_CPU
    return SUPPORTS_CPU | SUPPORTS_GPU


def backend_name(backend: Int) raises -> String:
    """A single `SUPPORTS_*` flag's name, for an error message."""
    if backend == SUPPORTS_CPU:
        return String("CPU")
    if backend == SUPPORTS_GPU:
        return String("GPU")
    raise Error("backend must be exactly one of SUPPORTS_CPU, SUPPORTS_GPU")


def check_objective_backend(objective: Int, backend: Int) raises:
    """Refuse an objective the named backend cannot train, by name.

    The one unsupported combination today is LambdaRank on the GPU: its
    lambdas are pairwise within a query group rather than a closed form per
    row, so there is no device kernel and no `train_ranker_gpu`. A device
    provider that would otherwise fall back silently should call this and
    say so instead, which is the difference between a run that is slower than
    the user asked for and a run they can reason about.

    This is deliberately the *wide* question, "does a trainer exist". It is
    not `objective_gradients_on_device`, which asks whether the derivatives
    are computed on the device, and it is not `gpu_trains_objective` in
    device_policy.mojo, which asks whether `train_gpu` itself accepts the
    code. See `objective_backends` for why the three differ.
    """
    if objective_backends(objective) & backend == 0:
        raise Error(
            "objective '",
            objective_canonical_name(objective),
            "' cannot be trained on the ",
            backend_name(backend),
            "; it is ",
            backend_name(SUPPORTS_CPU),
            " only",
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@always_inline
def metric_is_builtin(metric: Int) -> Bool:
    """Whether this code names a metric metrics.mojo and ranking.mojo
    compute. Everything else is a caller-supplied `CustomMetric`, which
    carries its own name and direction."""
    return metric >= 0 and metric < N_BUILTIN_METRICS


def metric_canonical_name(metric: Int) raises -> String:
    """The LightGBM name mojoboost reports this metric under."""
    if metric == METRIC_L2:
        return String("l2")
    if metric == METRIC_RMSE:
        return String("rmse")
    if metric == METRIC_L1:
        return String("l1")
    if metric == METRIC_QUANTILE:
        return String("quantile")
    if metric == METRIC_HUBER:
        return String("huber")
    if metric == METRIC_BINARY_LOGLOSS:
        return String("binary_logloss")
    if metric == METRIC_BINARY_ERROR:
        return String("binary_error")
    if metric == METRIC_AUC:
        return String("auc")
    if metric == METRIC_MULTI_LOGLOSS:
        return String("multi_logloss")
    if metric == METRIC_MULTI_ERROR:
        return String("multi_error")
    if metric == METRIC_NDCG:
        return String("ndcg")
    if metric == METRIC_MAPE:
        return String("mape")
    if metric == METRIC_FAIR:
        return String("fair")
    if metric == METRIC_POISSON:
        return String("poisson")
    if metric == METRIC_GAMMA:
        return String("gamma")
    if metric == METRIC_GAMMA_DEVIANCE:
        return String("gamma_deviance")
    if metric == METRIC_TWEEDIE:
        return String("tweedie")
    if metric == METRIC_CROSS_ENTROPY:
        return String("cross_entropy")
    if metric == METRIC_KLDIV:
        return String("kullback_leibler")
    if metric == METRIC_AVERAGE_PRECISION:
        return String("average_precision")
    if metric == METRIC_MAP:
        return String("map")
    raise Error("unknown metric code ", metric)


def metric_code_from_name(name: String) raises -> Int:
    """The metric code for a LightGBM metric name or alias.

    LightGBM's aliases are accepted under LightGBM's meanings, including the
    ones that collide with objective names: `regression` is the l2 *metric*
    here and the squared-error *objective* in `objective_code_from_name`,
    which is LightGBM's own overloading of the word and not a mistake to
    fix. Names are canonical lowercase.

    A metric that names a task the model does not belong to is a separate
    check (`metric_task`), so a caller can report "this metric scores
    binary models" rather than "unknown metric".
    """
    if (
        name == "l2"
        or name == "mean_squared_error"
        or name == "mse"
        or name == "regression_l2"
        or name == "regression"
    ):
        return METRIC_L2
    if (
        name == "rmse"
        or name == "root_mean_squared_error"
        or name == "l2_root"
    ):
        return METRIC_RMSE
    if (
        name == "l1"
        or name == "mean_absolute_error"
        or name == "mae"
        or name == "regression_l1"
    ):
        return METRIC_L1
    if name == "quantile":
        return METRIC_QUANTILE
    if name == "huber":
        return METRIC_HUBER
    if name == "mape" or name == "mean_absolute_percentage_error":
        return METRIC_MAPE
    if name == "fair":
        return METRIC_FAIR
    if name == "poisson":
        return METRIC_POISSON
    if name == "gamma":
        return METRIC_GAMMA
    if name == "gamma_deviance" or name == "gamma_dev":
        return METRIC_GAMMA_DEVIANCE
    if name == "tweedie":
        return METRIC_TWEEDIE
    if name == "cross_entropy" or name == "xentropy":
        return METRIC_CROSS_ENTROPY
    if name == "kullback_leibler" or name == "kldiv":
        return METRIC_KLDIV
    if name == "binary_logloss" or name == "binary":
        return METRIC_BINARY_LOGLOSS
    if name == "binary_error":
        return METRIC_BINARY_ERROR
    if name == "auc":
        return METRIC_AUC
    if name == "average_precision":
        return METRIC_AVERAGE_PRECISION
    if (
        name == "multi_logloss"
        or name == "multiclass"
        or name == "softmax"
    ):
        return METRIC_MULTI_LOGLOSS
    if name == "multi_error":
        return METRIC_MULTI_ERROR
    if name == "ndcg" or name == "lambdarank":
        return METRIC_NDCG
    if name == "map" or name == "mean_average_precision":
        return METRIC_MAP
    raise Error("unknown metric '", name, "'")


def metric_task(metric: Int) raises -> Int:
    """The task a metric can score.

    `cross_entropy` and `kullback_leibler` are regression metrics because
    the objective they belong to is: soft labels anywhere in [0, 1] are the
    regressor's business, since the classifier's labels are classes.
    """
    if (
        metric == METRIC_BINARY_LOGLOSS
        or metric == METRIC_BINARY_ERROR
        or metric == METRIC_AUC
        or metric == METRIC_AVERAGE_PRECISION
    ):
        return TASK_BINARY
    if metric == METRIC_MULTI_LOGLOSS or metric == METRIC_MULTI_ERROR:
        return TASK_MULTICLASS
    if metric == METRIC_NDCG or metric == METRIC_MAP:
        return TASK_RANKING
    if metric_is_builtin(metric):
        return TASK_REGRESSION
    raise Error("unknown metric code ", metric)


def metric_higher_is_better(metric: Int) raises -> Bool:
    """Which direction early stopping should watch. Everything here is a
    loss except AUC, average precision, NDCG, and MAP."""
    if not metric_is_builtin(metric):
        raise Error("unknown metric code ", metric)
    return (
        metric == METRIC_AUC
        or metric == METRIC_AVERAGE_PRECISION
        or metric == METRIC_NDCG
        or metric == METRIC_MAP
    )


def metric_needs(metric: Int) raises -> Int:
    """What this metric needs besides predictions, labels, and weights, as
    `NEEDS_*` bit flags.

    The four that read the objective's scalar parameter score the loss the
    objective was actually trained on, so scoring `tweedie` at a different
    variance power scores a different loss. The two ranking metrics need
    both their validation set's own groups and a cutoff (LightGBM's one
    `eval_at` serves both).
    """
    if (
        metric == METRIC_QUANTILE
        or metric == METRIC_HUBER
        or metric == METRIC_FAIR
        or metric == METRIC_TWEEDIE
    ):
        return NEEDS_PARAM
    if metric == METRIC_MULTI_LOGLOSS or metric == METRIC_MULTI_ERROR:
        return NEEDS_N_CLASSES
    if metric == METRIC_NDCG or metric == METRIC_MAP:
        return NEEDS_GROUPS | NEEDS_CUTOFF
    if metric_is_builtin(metric):
        return NEEDS_NOTHING
    raise Error("unknown metric code ", metric)


def metric_transform(metric: Int) raises -> Int:
    """What predictions must have been through before this metric sees them.

    Every single-output metric scores the *objective's* response scale, so
    `l2` on a poisson model scores expected counts and `binary_logloss`
    scores probabilities; the transform belongs to the objective, as in
    LightGBM, and a metric never applies a second one. The multiclass
    metrics take the softmax of each raw row themselves. The ranking metrics
    read raw scores, since only their order within a query matters.
    """
    if metric == METRIC_MULTI_LOGLOSS or metric == METRIC_MULTI_ERROR:
        return TRANSFORM_SOFTMAX
    if metric == METRIC_NDCG or metric == METRIC_MAP:
        return TRANSFORM_RAW
    if metric_is_builtin(metric):
        return TRANSFORM_OBJECTIVE_LINK
    raise Error("unknown metric code ", metric)


def metric_names_for_task(task: Int) raises -> String:
    """The metric names an estimator of this task accepts, sorted and comma
    separated, for the "expected one of ..." half of an error message."""
    if task == TASK_REGRESSION:
        return REGRESSION_METRIC_NAMES.copy()
    if task == TASK_BINARY:
        return BINARY_METRIC_NAMES.copy()
    if task == TASK_MULTICLASS:
        return MULTICLASS_METRIC_NAMES.copy()
    if task == TASK_RANKING:
        return RANKING_METRIC_NAMES.copy()
    raise Error("unknown task code ", task)


def metric_codes_for_task(task: Int) raises -> List[Int]:
    """The metric codes an estimator of this task accepts, in the same order
    `metric_names_for_task` prints their names."""
    var out = List[Int]()
    if task == TASK_REGRESSION:
        out.append(METRIC_CROSS_ENTROPY)
        out.append(METRIC_FAIR)
        out.append(METRIC_GAMMA)
        out.append(METRIC_GAMMA_DEVIANCE)
        out.append(METRIC_HUBER)
        out.append(METRIC_KLDIV)
        out.append(METRIC_L1)
        out.append(METRIC_L2)
        out.append(METRIC_MAPE)
        out.append(METRIC_POISSON)
        out.append(METRIC_QUANTILE)
        out.append(METRIC_RMSE)
        out.append(METRIC_TWEEDIE)
        return out^
    if task == TASK_BINARY:
        out.append(METRIC_AUC)
        out.append(METRIC_AVERAGE_PRECISION)
        out.append(METRIC_BINARY_ERROR)
        out.append(METRIC_BINARY_LOGLOSS)
        return out^
    if task == TASK_MULTICLASS:
        out.append(METRIC_MULTI_ERROR)
        out.append(METRIC_MULTI_LOGLOSS)
        return out^
    if task == TASK_RANKING:
        out.append(METRIC_MAP)
        out.append(METRIC_NDCG)
        return out^
    raise Error("unknown task code ", task)


def objective_default_metric(objective: Int) raises -> Int:
    """The metric to score when none is named: LightGBM's rule, the
    objective's own loss.

    A custom objective has none and says so rather than picking one, because
    only its author knows what it optimizes. Note that this is keyed on the
    objective *code*, so every alias of an objective gets its default; the
    Python table it replaces is keyed on the name the user typed, which is
    why `objective="mse"` currently has no default while `objective="l2"`'s
    synonym `objective="regression"` does. See the handoff.
    """
    if objective == SQUARED_ERROR:
        return METRIC_L2
    if objective == L1:
        return METRIC_L1
    if objective == HUBER:
        return METRIC_HUBER
    if objective == QUANTILE:
        return METRIC_QUANTILE
    if objective == POISSON:
        return METRIC_POISSON
    if objective == GAMMA:
        return METRIC_GAMMA
    if objective == TWEEDIE:
        return METRIC_TWEEDIE
    if objective == MAPE:
        return METRIC_MAPE
    if objective == FAIR:
        return METRIC_FAIR
    if objective == CROSS_ENTROPY:
        return METRIC_CROSS_ENTROPY
    if objective == BINARY_LOGISTIC:
        return METRIC_BINARY_LOGLOSS
    if objective == MULTICLASS:
        return METRIC_MULTI_LOGLOSS
    if objective == LAMBDARANK:
        return METRIC_NDCG
    if objective == CUSTOM:
        raise Error(
            "a custom objective has no default metric to score: only its"
            " author knows what it optimizes"
        )
    raise Error("unknown objective code ", objective)


def objective_has_default_metric(objective: Int) raises -> Bool:
    """Whether `objective_default_metric` will answer rather than raise.
    False for `CUSTOM` alone. Useful where a caller is already building an
    error message and cannot afford a second one."""
    return objective_is_known(objective) and objective != CUSTOM


# ---------------------------------------------------------------------------
# Objective and metric compatibility
# ---------------------------------------------------------------------------


def objective_supports_metric(objective: Int, metric: Int) raises -> Bool:
    """Whether this metric can score a model trained with this objective.

    The rule is one line: the tasks must match. It is stricter than LightGBM,
    which scores whatever it can compute, and it is the rule the Python
    estimators already enforce from `_eval.resolve`; stating it here is what
    lets the native path enforce the same thing, so `binary_logloss` on a
    poisson model is refused in Mojo rather than only in Python.

    A custom objective reports as regression (see `objective_task`), so a
    custom-objective model accepts the regression metrics. That is the family
    its single real-valued output belongs to; it does not mean the framework
    knows what the objective optimizes, which is why it still has no default
    metric.
    """
    return objective_task(objective) == metric_task(metric)


def check_objective_metric(objective: Int, metric: Int) raises:
    """Refuse a metric that does not score this objective's task, naming
    both tasks and the metrics that would work.

    "unknown metric" and "wrong metric for this model" are different
    mistakes and get different sentences: `metric_code_from_name` reports the
    first, this reports the second.
    """
    if objective_supports_metric(objective, metric):
        return
    var task = objective_task(objective)
    raise Error(
        "metric '",
        metric_canonical_name(metric),
        "' scores a ",
        task_name(metric_task(metric)),
        " task; objective '",
        objective_canonical_name(objective),
        "' trains a ",
        task_name(task),
        " one, whose metrics are ",
        metric_names_for_task(task),
    )


def metric_code_for_objective(name: String, objective: Int) raises -> Int:
    """Resolve a metric name against an objective in one call: the code, or
    a raise that says whether the name is unknown or merely wrong here.

    This is the query a trainer, a binding, and a Python estimator all need,
    and having it here is what keeps the two error messages identical across
    the three. Names are canonical lowercase, as everywhere in this module.
    """
    var metric = metric_code_from_name(name)
    check_objective_metric(objective, metric)
    return metric


def _metric_param_owner(metric: Int) raises -> Int:
    """The objective whose scalar parameter a parameterized metric reads.
    One of the four; anything else has no parameter and no owner."""
    if metric == METRIC_QUANTILE:
        return QUANTILE
    if metric == METRIC_HUBER:
        return HUBER
    if metric == METRIC_FAIR:
        return FAIR
    if metric == METRIC_TWEEDIE:
        return TWEEDIE
    raise Error(
        "metric '",
        metric_canonical_name(metric),
        "' reads no scalar parameter",
    )


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

# Every spelling the two resolvers accept, and every objective code that
# exists, so a caller can walk the registry instead of re-typing it. The
# binding needs exactly this: without it, a Python-facing table would have
# to spell the names a third time, which is the duplication this module
# exists to remove.
#
# These lists and the `if` chains above are the one duplication inside this
# file, and it is deliberate: the chain resolves without allocating, the
# list enumerates once at start-up. They sit next to each other and the
# round trip between them ("every name resolves, every code names itself")
# is the first check the validation pass runs.

comptime OBJECTIVE_ALIAS_NAMES = String(
    "regression regression_l2 l2 mean_squared_error mse binary poisson"
    " huber quantile mae regression_l1 l1 mean_absolute_error gamma tweedie"
    " mape mean_absolute_percentage_error fair cross_entropy xentropy"
    " multiclass softmax lambdarank custom"
)

comptime UNIMPLEMENTED_OBJECTIVE_ALIAS_NAMES = String(
    "cross_entropy_lambda xentlambda multiclassova multiclass_ova ova ovr"
    " rank_xendcg xendcg"
)

comptime METRIC_ALIAS_NAMES = String(
    "l2 mean_squared_error mse regression_l2 regression rmse"
    " root_mean_squared_error l2_root l1 mean_absolute_error mae"
    " regression_l1 quantile huber mape mean_absolute_percentage_error fair"
    " poisson gamma gamma_deviance gamma_dev tweedie cross_entropy xentropy"
    " kullback_leibler kldiv binary_logloss binary binary_error auc"
    " average_precision multi_logloss multiclass softmax multi_error ndcg"
    " lambdarank map mean_average_precision"
)


def _split_names(names: String) -> List[String]:
    var out = List[String]()
    for token in names.split():
        out.append(String(token))
    return out^


def objective_alias_names() -> List[String]:
    """Every objective spelling `objective_code_from_name` resolves."""
    return _split_names(OBJECTIVE_ALIAS_NAMES)


def unimplemented_objective_alias_names() -> List[String]:
    """Every spelling of a LightGBM objective mojoboost reports by name as
    not implemented."""
    return _split_names(UNIMPLEMENTED_OBJECTIVE_ALIAS_NAMES)


def metric_alias_names() -> List[String]:
    """Every metric spelling `metric_code_from_name` resolves."""
    return _split_names(METRIC_ALIAS_NAMES)


def all_objective_codes() -> List[Int]:
    """Every objective code, built-ins first and then the three with their
    own trainers. Metric codes need no such list: they are 0 through
    `N_BUILTIN_METRICS - 1`."""
    var out = List[Int]()
    out.append(SQUARED_ERROR)
    out.append(BINARY_LOGISTIC)
    out.append(POISSON)
    out.append(HUBER)
    out.append(QUANTILE)
    out.append(L1)
    out.append(GAMMA)
    out.append(TWEEDIE)
    out.append(MAPE)
    out.append(FAIR)
    out.append(CROSS_ENTROPY)
    out.append(MULTICLASS)
    out.append(LAMBDARANK)
    out.append(CUSTOM)
    return out^


# ---------------------------------------------------------------------------
# Whole-record queries
# ---------------------------------------------------------------------------


@fieldwise_init
struct ObjectiveSpec(Copyable, Movable):
    """Every registry fact about one objective, in scalars only.

    For the callers that want the whole record at once (a binding filling a
    dict, a Python facade answering one question per attribute) rather than
    one compare per question. Names are not fields: they allocate, and a
    caller that needs one asks `objective_canonical_name`.
    """

    var code: Int
    var task: Int
    var link: Int
    var param: Int
    var default_param: Float64
    var renews_leaves: Bool
    var multi_output: Bool
    var needs_groups: Bool
    var gradients_on_device: Bool
    var backends: Int
    var builtin: Bool


def objective_spec(objective: Int) raises -> ObjectiveSpec:
    """Every fact about an objective in one record. Raises for a code that
    names no objective, which is the one thing a caller cannot recover
    from."""
    if not objective_is_known(objective):
        raise Error("unknown objective code ", objective)
    return ObjectiveSpec(
        objective,
        objective_task(objective),
        objective_link(objective),
        objective_param(objective),
        objective_default_param(objective),
        objective_renews_leaves(objective),
        objective_is_multi_output(objective),
        objective_needs_groups(objective),
        objective_gradients_on_device(objective),
        objective_backends(objective),
        objective_is_builtin(objective),
    )


@fieldwise_init
struct MetricSpec(Copyable, Movable):
    """Every registry fact about one built-in metric, in scalars only."""

    var code: Int
    var task: Int
    var higher_is_better: Bool
    var needs: Int
    var transform: Int


def metric_spec(metric: Int) raises -> MetricSpec:
    """Every fact about a built-in metric in one record."""
    if not metric_is_builtin(metric):
        raise Error("unknown metric code ", metric)
    return MetricSpec(
        metric,
        metric_task(metric),
        metric_higher_is_better(metric),
        metric_needs(metric),
        metric_transform(metric),
    )


def metric_scoring_param(
    metric: Int, objective: Int, alpha: Float64
) raises -> Float64:
    """The scalar a metric should be scored at, given the objective's own.

    Four metrics are a family rather than a function: `quantile` and `huber`
    read an alpha, `fair` a c, and `tweedie` a variance power. LightGBM
    scores them at the *objective's* parameter, so a tweedie model scored at
    a different variance power is scoring a different loss, and this returns
    that parameter after checking it lies in the objective's domain.

    The awkward case is a metric whose parameter is not the objective's:
    `tweedie` scoring a huber model, say, where both are regression metrics
    so the pair is allowed, but the objective's `alpha` of 0.9 is not a
    variance power and `tweedie_loss` would refuse it. The value is therefore
    always validated against the domain of the objective the *metric* belongs
    to, and the mismatch is named rather than left to surface as a metric
    complaining about a number it was never told the origin of. When the two
    coincide, which is every default metric and every deliberate pairing, the
    check is exactly `check_objective_param` on the objective itself.

    Metrics that read no scalar return 0.0, which the fifteen of them ignore.
    """
    if metric_needs(metric) & NEEDS_PARAM == 0:
        return 0.0
    var owner = _metric_param_owner(metric)
    if objective_param_domain(owner).contains(alpha):
        return alpha
    if owner == objective:
        # The objective's own parameter is out of its own domain. The
        # trainer's message is the right one, and it is the only one a user
        # who set `alpha` themselves should ever see.
        check_objective_param(objective, alpha)
    raise Error(
        "metric '",
        metric_canonical_name(metric),
        "' reads ",
        objective_param_name(owner),
        " in ",
        objective_param_domain(owner).describe(),
        ", but objective '",
        objective_canonical_name(objective),
        "' supplies ",
        alpha,
        "; score a metric of the objective's own family, or pass the"
        " metric's parameter explicitly",
    )
