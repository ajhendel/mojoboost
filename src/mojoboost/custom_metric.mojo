"""Custom validation metrics.

A custom metric is a callable that scores validation predictions against
validation labels and returns one scalar:

    def my_metric(
        pred: List[Float64], target: List[Float64]
    ) raises -> Float64:
        ...

`MetricFn` is that signature (the same shape as `EvalLossFn` in
objective.mojo, without its lower-is-better assumption). Each metric also
carries a name, a direction (`higher_is_better`), and a flag saying whether
early stopping watches it. Several metrics can be evaluated per round, with
one of them named as the primary metric that selects the best round.

Metrics are independent of objectives. Every combination works:

    train_with_metrics          built-in objective  + custom metrics
    train_custom_with_metrics   custom objective    + custom metrics
    train_with_valid            built-in objective  + built-in loss
    train_custom_with_valid     custom objective    + one custom loss

Passing several metrics
-----------------------
Mojo callables of the same signature still have distinct concrete types, so
a `List` cannot hold two different ones. A `MetricSuite` therefore holds one
*dispatching* callable of type `MetricSetFn`, which selects a metric by
index (and is told which validation set it is scoring, which lets a callback
cache per-set state):

    def evaluate(
        metric: Int, valid: Int, pred: List[Float64], target: List[Float64]
    ) raises -> Float64:
        if metric == 0:
            return rmse(pred, target)
        return binary_auc(pred, target)

    var suite = MetricSuite(
        [CustomMetric("rmse"), CustomMetric("auc", higher_is_better=True)],
        evaluate,
        primary=1,
    )

`train_with_metric` takes a single `MetricFn` directly and builds that
one-metric suite for you. The dispatching callable is a compile-time
parameter, so the per-round call is direct and inlinable, and one runtime
callable (a Python callback, say) can serve every metric.

Contract, and how it differs from LightGBM
------------------------------------------
- Predictions are raw scores, before any inverse link, matching LightGBM's
  `feval`: log-odds for BINARY_LOGISTIC, log-mean for POISSON, and the
  prediction itself for the regression objectives and for custom ones.
  `response_scale` converts a raw vector when a metric wants probabilities.
- The callback is invoked once per metric per validation set per round,
  never per row.
- Metric values must be finite. A NaN or infinity raises, naming the metric
  and the round, instead of silently poisoning the comparisons the way a
  NaN would (`NaN < best` is False, so a NaN metric would look like an
  endless run of non-improving rounds). LightGBM does not check.
- Metric metadata is declared up front. LightGBM's `feval` returns
  `(name, value, is_higher_better)` per call, which means the direction is
  only known after the first evaluation; here it is part of `CustomMetric`,
  so validation of the primary metric and of the early-stopping set happens
  before training starts.
- Early stopping tracks every (validation set, flagged metric) pair and
  stops as soon as any one of them goes `early_stopping_rounds` rounds
  without improving, which is what LightGBM's `early_stopping` callback
  does. The ensemble is then truncated to the best round of the *primary*
  metric on the *first* validation set; LightGBM instead truncates to the
  best iteration of the pair that triggered the stop. Naming the round
  selector explicitly is the intentional difference: which model you keep
  should not depend on which pair happened to run out of patience first.
- Improvement is strict and, with a nonzero `min_delta`, must clear it:
  `value > best + min_delta` when higher is better, `value < best -
  min_delta` otherwise. A tie is not an improvement. LightGBM's
  `early_stopping(min_delta=...)` uses the same rule.
- `early_stopping_rounds = 0` disables early stopping: every metric is
  still evaluated and recorded every round, no round can stop training, and
  the full ensemble is returned (`best_iteration` still reports where the
  primary metric peaked). LightGBM's equivalent is omitting the callback.
- The history records round 0, the base-score-only model, at index 0, so
  `value(i, ...)` is the score after `i` trees and `best_iteration` 0 means
  every tree hurt. LightGBM's `evals_result_` starts at the first
  iteration.
- Validation rows are never bagged, matching `train_with_valid`, and the
  framework never weights them either: a metric that should be weighted
  holds its own weights, which is how `metrics.mojo` takes them.
- CPU only. There is no GPU trainer with a validation loop; the metric
  itself is host code either way.

Built-in metrics, by code
-------------------------
Everything above is for a metric the caller writes. For one mojoboost already
has, name it instead:

    var names: List[String] = ["binary_logloss", "auc"]
    var ctx = BuiltinMetricContext.single_output(BINARY_LOGISTIC)
    var codes = resolve_builtin_metrics(names, BINARY_LOGISTIC)
    var result = train_with_builtin_metrics(
        data, y, valid_sets^, params, codes, ctx,
        primary=1, early_stopping_rounds=10,
    )

Nothing about those two metrics is written at the call site. That `auc` is
higher-is-better, that both score probabilities rather than log-odds, that
neither may score a poisson model, and that `auc` is metric code 7 are all
facts of `objective_registry.mojo`, and getting the first one wrong by hand
produces a run that finishes and truncates to the worst round. The four
`*_with_builtin_metrics` entry points read them from the registry:

- `resolve_builtin_metrics` turns names (every LightGBM alias included) into
  codes and refuses one that scores a different task, by name.
- `builtin_metric_metadata` gives each code its canonical name and its
  direction, which is what early stopping compares with.
- `eval_builtin_metric` applies the transform the registry says the metric
  expects and calls the one function that computes it, in metrics.mojo or
  ranking.mojo.
- `BuiltinMetricContext` carries what the registry says a metric needs
  beyond predictions and labels: the objective and its scalar parameter, a
  class count, per-validation-set query groups and a cutoff, and per-
  validation-set weights. `ctx.check` runs the whole request before the
  first tree.

The objective is not a separate argument to those entry points; it comes
from the context, so the metric always scores the objective the model
trained with at the parameter it trained at. A caller who wants a built-in
metric and a hand-written one in the same run builds a `MetricSuite` whose
evaluator calls `eval_builtin_metric` for some indices and their own code
for others; nothing about the two paths is exclusive.

Multiclass and ranking
----------------------
`train_multiclass_with_metrics` and `train_ranker_with_metrics` extend the
same machinery to the two multi-tree-per-round trainers, with the same
early-stopping, truncation, and history rules. A round is one tree per class
for multiclass and one tree for ranking, and `best_iteration` counts rounds
in both cases.

What the metric receives differs, because that is what the model produces:

- multiclass: `pred` holds row-major raw scores, `pred[r * n_classes + k]`,
  so it is `n_classes` times as long as the labels. Apply `_softmax_inplace`
  or your own link to get probabilities. Labels ride in the `ValidSet` as
  Float64 whole numbers in 0..n_classes-1, the same encoding the trainer
  takes.
- ranking: `pred` holds one raw ranking score per row and the labels are
  graded relevances. Query boundaries are *not* passed to the metric: a
  ranking metric needs the validation set's own groups, and the caller
  already holds them, so it captures them rather than being handed them.
"""

from std.math import exp, isfinite, log

from .bagging import BaggingParams, bagging_enabled, check_bagging, refresh_bag
from .binning import BinMapper, BinnedMatrix, fit_bins
from .callback import (
    ABORT,
    AFTER_ITERATION,
    BEFORE_ITERATION,
    CONTINUE,
    IterationEnv,
    IterationFn,
    STOP,
    check_resettable,
    no_callback,
    scale_tree_values,
)
from .boosting import (
    CUSTOM,
    Booster,
    BoosterParams,
    MulticlassBooster,
    _base_score,
    _check_goss,
    _check_objective,
    _check_sample_weight,
    _clamp_prob,
    _fill_grad_hess,
    _fill_softmax_grad_hess,
    _multiclass_goss_select,
    _renew_leaf_values,
    _sigmoid,
    _softmax_inplace,
    objective_renews_leaves,
    renewal_alpha,
    renewal_weights,
)
from .goss import GossParams, GossSelection, apply_goss_scaling, goss_round
from .metrics import (
    METRIC_MAP,
    METRIC_MULTI_ERROR,
    METRIC_MULTI_LOGLOSS,
    METRIC_NDCG,
    eval_metric_by_code,
    multiclass_error,
    multiclass_log_loss,
)
from .model import Model, MulticlassModel
from .objective import (
    GradHessFn,
    _apply_sample_weight,
    check_custom_grad_hess,
)
from .objective_registry import (
    LINK_EXP,
    LINK_IDENTITY,
    LINK_SIGMOID,
    LINK_SOFTMAX,
    MULTICLASS,
    NEEDS_CUTOFF,
    NEEDS_GROUPS,
    NEEDS_N_CLASSES,
    TRANSFORM_OBJECTIVE_LINK,
    TRANSFORM_RAW,
    TRANSFORM_SOFTMAX,
    check_objective_metric,
    metric_canonical_name,
    metric_code_for_objective,
    metric_higher_is_better,
    metric_is_builtin,
    metric_needs,
    metric_scoring_param,
    metric_transform,
    objective_canonical_name,
    objective_default_metric,
    objective_link,
)
from .ranking import (
    DEFAULT_NDCG_EVAL_AT,
    LAMBDARANK,
    RankGroups,
    RankerParams,
    _discounts,
    _fill_lambdas,
    _inverse_max_dcgs,
    _refresh_query_bag,
    check_groups,
    check_labels,
    check_ranker_params,
    groups_from_counts,
    mean_average_precision,
    ndcg,
)
from .tree import Tree, grow_tree

comptime MetricFn = def (List[Float64], List[Float64]) raises -> Float64
"""Raw validation predictions and labels in, one scalar out."""

comptime MetricSetFn = def (
    Int, Int, List[Float64], List[Float64]
) raises -> Float64
"""(metric index, validation-set index, raw predictions, labels) in, one
scalar out. Dispatches a whole set of metrics through one callable."""


struct CustomMetric(Copyable, Movable, Writable):
    """One metric's metadata: what to call it, which direction is better,
    and whether early stopping watches it."""

    var name: String
    var higher_is_better: Bool
    var use_for_early_stopping: Bool

    def __init__(
        out self,
        var name: String,
        higher_is_better: Bool = False,
        use_for_early_stopping: Bool = True,
    ):
        self.name = name^
        self.higher_is_better = higher_is_better
        self.use_for_early_stopping = use_for_early_stopping

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "CustomMetric(",
            self.name,
            ", higher_is_better=",
            self.higher_is_better,
            ", use_for_early_stopping=",
            self.use_for_early_stopping,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


@fieldwise_init
struct ValidSet(Copyable, Movable):
    """A named validation set of pre-binned features and labels."""

    var name: String
    var data: BinnedMatrix
    var target: List[Float64]


@fieldwise_init
struct RawValidSet(Copyable, Movable):
    """A named validation set of raw (unbinned) features, column-major
    `features[f * n_rows + r]`, for the `fit_with_metrics` entry point."""

    var name: String
    var features: List[Float64]
    var n_rows: Int
    var target: List[Float64]


struct MetricSuite[F: MetricSetFn & Copyable](Copyable, Movable):
    """The metrics to evaluate, their dispatching callable, and which one is
    primary (the metric whose best round the ensemble is truncated to)."""

    var metrics: List[CustomMetric]
    var evaluator: Self.F
    var primary: Int

    def __init__(
        out self,
        var metrics: List[CustomMetric],
        var evaluator: Self.F,
        primary: Int = 0,
    ):
        self.metrics = metrics^
        self.evaluator = evaluator^
        self.primary = primary

    def n_metrics(self) -> Int:
        return len(self.metrics)

    def primary_metric(self) raises -> CustomMetric:
        return self.metrics[self.primary].copy()

    def evaluate(
        self,
        metric: Int,
        valid: Int,
        pred: List[Float64],
        target: List[Float64],
    ) raises -> Float64:
        """Score one metric on one validation set."""
        return self.evaluator(metric, valid, pred, target)

    def check(self) raises:
        """Validate the metric metadata: at least one metric, non-empty
        unique names, and a primary index that exists."""
        if len(self.metrics) == 0:
            raise Error("a metric suite needs at least one metric")
        for m in range(len(self.metrics)):
            if self.metrics[m].name.byte_length() == 0:
                raise Error("metric names must not be empty")
            for other in range(m):
                if self.metrics[other].name == self.metrics[m].name:
                    raise Error(
                        String("duplicate metric name ", self.metrics[m].name)
                    )
        if self.primary < 0 or self.primary >= len(self.metrics):
            raise Error("primary metric index out of range")


struct MetricHistory(Copyable, Movable):
    """Every metric value computed during training.

    `value(round, valid, metric)` is the score after `round` trees, so index
    0 is the base-score-only model and `n_rounds() - 1` is the last round
    trained (not necessarily the round the returned ensemble was truncated
    to). Values are stored round-major.
    """

    var metric_names: List[String]
    var valid_names: List[String]
    var values: List[Float64]

    def __init__(
        out self, var metric_names: List[String], var valid_names: List[String]
    ):
        self.metric_names = metric_names^
        self.valid_names = valid_names^
        self.values = List[Float64]()

    def n_metrics(self) -> Int:
        return len(self.metric_names)

    def n_valid(self) -> Int:
        return len(self.valid_names)

    def n_rounds(self) -> Int:
        var per_round = self.n_valid() * self.n_metrics()
        if per_round == 0:
            return 0
        return len(self.values) // per_round

    def _index(self, round: Int, valid: Int, metric: Int) raises -> Int:
        if round < 0 or round >= self.n_rounds():
            raise Error("history round out of range")
        if valid < 0 or valid >= self.n_valid():
            raise Error("history validation-set index out of range")
        if metric < 0 or metric >= self.n_metrics():
            raise Error("history metric index out of range")
        return (
            round * self.n_valid() * self.n_metrics()
            + valid * self.n_metrics()
            + metric
        )

    def value(self, round: Int, valid: Int, metric: Int) raises -> Float64:
        return self.values[self._index(round, valid, metric)]

    def series(self, valid: Int, metric: Int) raises -> List[Float64]:
        """One metric's values on one validation set, round by round."""
        var out = List[Float64](capacity=self.n_rounds())
        for round in range(self.n_rounds()):
            out.append(self.value(round, valid, metric))
        return out^


@fieldwise_init
struct MetricTrainResult(Copyable, Movable):
    """A trained booster plus the metric record that shaped it.

    `best_iteration` is the number of trees kept, the round where the
    primary metric peaked on the first validation set. `stopped_early` says
    whether a tracked metric ran out of patience (False when the run used
    every round, and always False when early stopping is disabled).
    """

    var booster: Booster
    var history: MetricHistory
    var best_iteration: Int
    var best_score: Float64
    var stopped_early: Bool


@fieldwise_init
struct MetricFitResult(Copyable, Movable):
    """`MetricTrainResult` for the raw-feature entry point: the same record
    around a `Model`, which carries the fitted bin mapper."""

    var model: Model
    var history: MetricHistory
    var best_iteration: Int
    var best_score: Float64
    var stopped_early: Bool


def response_scale(objective: Int, raw: List[Float64]) raises -> List[Float64]:
    """Apply an objective's inverse link to raw scores: the sigmoid for
    BINARY_LOGISTIC and CROSS_ENTROPY, exp for POISSON, GAMMA, and TWEEDIE,
    identity otherwise (including CUSTOM, whose link the framework does not
    know).

    Metrics receive raw scores; this is the one call that turns them into
    the probabilities a log loss or a calibration metric wants, and the
    expected values the poisson, gamma, and tweedie metrics want. It applies
    the same links `Booster.response` does, so a metric sees exactly what a
    prediction would return.

    Which link belongs to which objective is `objective_link` in
    objective_registry.mojo and is not decided here. It used to be: this
    function, `Booster.response`, and `response_for_objective` in
    gpu_predict.mojo each carried the same list of objective codes, agreeing
    by inspection. Reading the answer from the registry is what makes them
    unable to stop agreeing.

    `LINK_SOFTMAX` raises rather than returning something: a softmax is not a
    per-row map, so a multiclass raw vector cannot be converted one element
    at a time. `eval_builtin_metric` handles the multiclass metrics on
    row-major blocks instead, and `_softmax_inplace` is the call that does
    it.
    """
    var link = objective_link(objective)
    var out = List[Float64](capacity=len(raw))
    if link == LINK_SIGMOID:
        for r in range(len(raw)):
            out.append(_sigmoid(raw[r]))
    elif link == LINK_EXP:
        for r in range(len(raw)):
            out.append(exp(raw[r]))
    elif link == LINK_IDENTITY:
        for r in range(len(raw)):
            out.append(raw[r])
    elif link == LINK_SOFTMAX:
        raise Error(
            "objective '",
            objective_canonical_name(objective),
            "' has a softmax link, which is not a per-row transform; score"
            " its raw scores with eval_builtin_metric, which takes the"
            " softmax of each row",
        )
    else:
        raise Error("unknown link code ", link)
    return out^


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------
#
# Everything above takes a caller-supplied callable. Everything below takes a
# *metric code* and looks the rest up: the name and the direction from the
# registry, the transform from the registry, the arithmetic from metrics.mojo
# and ranking.mojo. That is the whole point of it. A caller who wants
# `binary_logloss` and `auc` watching an early-stopping run should not have to
# write a dispatching closure, know that AUC is higher-is-better, or remember
# that a binary model's metrics score probabilities rather than log-odds; each
# of those is a fact the registry already holds, and getting one of them wrong
# by hand produces a run that trains to completion and reports the wrong best
# round.
#
# There is no second table here. `eval_builtin_metric` is a router: it asks
# the registry which of the three shapes a code has and calls the one function
# that computes it. Adding a metric means adding it to metrics.mojo and to the
# registry; this file does not change.


def _whole_number_codes(
    target: List[Float64], n_classes: Int, what: String
) raises -> List[Int]:
    """Float64 labels as the Int codes a multiclass or ranking metric needs.

    Labels travel in `ValidSet.target` as Float64 because one `ValidSet` type
    serves every trainer. Only whole numbers mean anything to either metric
    family, and anything else is rejected rather than truncated. `n_classes`
    of 0 means "no upper bound", which is the ranking case: a relevance grade
    is checked by `check_labels` instead.
    """
    var out = List[Int](capacity=len(target))
    for r in range(len(target)):
        var value = target[r]
        var ok = isfinite(value)
        var code = Int(value) if ok else 0
        if ok:
            ok = Float64(code) == value and code >= 0
        if ok and n_classes > 0:
            ok = code < n_classes
        if not ok:
            raise Error(String(what, " at row ", r, " is not a valid label"))
        out.append(code)
    return out^


@fieldwise_init
struct BuiltinMetricContext(Copyable, Movable):
    """Everything a built-in metric needs beyond predictions and labels.

    The registry says *what* each metric needs (`metric_needs`); this carries
    it. One value covers a whole run, so it is built once and captured by the
    dispatching closure rather than rebuilt per round.

    - `objective` decides the transform for every single-output metric, since
      the link belongs to the objective and not to the metric. It is also
      what `check` validates the metric codes against.
    - `alpha` is the objective's scalar parameter, the one the four
      parameterized metrics score at. `metric_scoring_param` is what decides
      whether a given metric may read it.
    - `n_classes` is read by the two multiclass metrics and ignored
      otherwise.
    - `groups` holds one `RankGroups` per validation set, in the same order
      as the validation sets, because a ranking metric ranks within *that*
      set's queries. `cutoff` is LightGBM's `eval_at`, shared by ndcg and map.
    - `weights` holds one weight vector per validation set, again in order,
      and an empty outer list means every validation set is unweighted. This
      is the only way a validation metric is ever weighted: the training
      `sample_weight` is deliberately not reused, matching
      `train_with_valid`, so a class weight folded into the training weights
      cannot leak into the score and be applied a second time. The two
      ranking metrics take no weights at all, which is LightGBM's behavior
      and ranking.mojo's signature.
    """

    var objective: Int
    var alpha: Float64
    var n_classes: Int
    var groups: List[RankGroups]
    var cutoff: Int
    var weights: List[List[Float64]]

    @staticmethod
    def single_output(
        objective: Int, alpha: Float64 = 0.9
    ) -> BuiltinMetricContext:
        """A context for a regression, binary, cross-entropy, or
        custom-objective run: the objective and its scalar parameter, and
        nothing else to carry."""
        return BuiltinMetricContext(
            objective, alpha, 0, List[RankGroups](), 0, List[List[Float64]]()
        )

    @staticmethod
    def multiclass(n_classes: Int) -> BuiltinMetricContext:
        """A context for a softmax run. The objective is `MULTICLASS`, whose
        link is the softmax, so no single-output metric can score it."""
        return BuiltinMetricContext(
            MULTICLASS,
            0.9,
            n_classes,
            List[RankGroups](),
            0,
            List[List[Float64]](),
        )

    @staticmethod
    def ranking(
        var groups: List[RankGroups], cutoff: Int = DEFAULT_NDCG_EVAL_AT
    ) -> BuiltinMetricContext:
        """A context for a LambdaRank run: one query grouping per validation
        set and the cutoff both ranking metrics truncate at."""
        return BuiltinMetricContext(
            LAMBDARANK, 0.9, 0, groups^, cutoff, List[List[Float64]]()
        )

    def set_weights(mut self, var weights: List[List[Float64]]):
        """Give the context one weight vector per validation set, in the same
        order as the validation sets. An entry may be empty, which leaves
        that set unweighted. `ctx.check` rejects a count that does not match
        the number of validation sets."""
        self.weights = weights^

    def weight_for(self, valid: Int) raises -> List[Float64]:
        """This validation set's weights, empty when unweighted."""
        if len(self.weights) == 0:
            return List[Float64]()
        return self.weights[valid].copy()

    def groups_for(self, valid: Int) raises -> RankGroups:
        """This validation set's query grouping."""
        if valid < 0 or valid >= len(self.groups):
            raise Error(
                "a ranking metric needs one RankGroups per validation set;"
                " BuiltinMetricContext carries ",
                len(self.groups),
            )
        return self.groups[valid].copy()

    def check(self, metric_codes: List[Int], n_valid: Int) raises:
        """Validate the whole request before training starts: every code is
        a built-in metric, every one of them scores this objective's task,
        and everything they need is present.

        Every failure here is a failure a user should see before the first
        tree rather than at round zero, which is why it is a separate pass
        over the codes rather than a check inside the dispatch.
        """
        if len(metric_codes) == 0:
            raise Error("a built-in metric suite needs at least one metric")
        if len(self.weights) > 0 and len(self.weights) != n_valid:
            raise Error(
                "BuiltinMetricContext carries ",
                len(self.weights),
                " weight vectors for ",
                n_valid,
                " validation sets",
            )
        var needs = 0
        for i in range(len(metric_codes)):
            var metric = metric_codes[i]
            if not metric_is_builtin(metric):
                raise Error(
                    "metric code ",
                    metric,
                    " is not a built-in metric; a caller-supplied metric goes"
                    " through MetricSuite, which carries its own name and"
                    " direction",
                )
            check_objective_metric(self.objective, metric)
            # Raises here rather than at round zero if the objective's scalar
            # is not the one this metric reads.
            _ = metric_scoring_param(metric, self.objective, self.alpha)
            needs |= metric_needs(metric)
            for other in range(i):
                if metric_codes[other] == metric:
                    raise Error(
                        "metric '",
                        metric_canonical_name(metric),
                        "' is named twice; each metric is scored once per"
                        " validation set per round",
                    )
        if (needs & NEEDS_N_CLASSES) != 0 and self.n_classes < 2:
            raise Error(
                "the multiclass metrics need n_classes >= 2;"
                " BuiltinMetricContext carries ",
                self.n_classes,
            )
        if (needs & NEEDS_GROUPS) != 0 and len(self.groups) != n_valid:
            raise Error(
                "the ranking metrics need one RankGroups per validation set;"
                " BuiltinMetricContext carries ",
                len(self.groups),
                " for ",
                n_valid,
            )
        if (needs & NEEDS_CUTOFF) != 0 and self.cutoff < 1:
            raise Error("the ranking metrics need a positive cutoff")


def builtin_metric_metadata(
    metric_codes: List[Int],
) raises -> List[CustomMetric]:
    """One `CustomMetric` per code: the registry's canonical name and its
    direction.

    This is where the registry reaches early stopping. `higher_is_better`
    decides which way `_StopState.observe` compares, so a metric registered
    with the wrong direction would train to completion and truncate to the
    worst round; reading it from the one place that knows is the difference
    between that and a caller remembering. Every metric is watched for early
    stopping, which is LightGBM's behavior when a metric is named at all.
    """
    var out = List[CustomMetric](capacity=len(metric_codes))
    for i in range(len(metric_codes)):
        var metric = metric_codes[i]
        out.append(
            CustomMetric(
                metric_canonical_name(metric),
                metric_higher_is_better(metric),
                True,
            )
        )
    return out^


def resolve_builtin_metrics(
    names: List[String], objective: Int
) raises -> List[Int]:
    """Metric names to codes, checked against the objective's task.

    Names are canonical lowercase and every LightGBM alias resolves, both of
    which are `metric_code_from_name`'s rules. A name that is not a metric
    and a name that is a metric for a different kind of model get different
    sentences, which is `metric_code_for_objective`'s reason for existing.
    """
    var out = List[Int](capacity=len(names))
    for i in range(len(names)):
        out.append(metric_code_for_objective(names[i], objective))
    return out^


def default_metric_codes(objective: Int) raises -> List[Int]:
    """The one metric to score when the caller names none: the objective's
    own loss, LightGBM's rule.

    A custom objective has none and says so; only its author knows what it
    optimizes, so naming a metric is not optional there.
    """
    var out = List[Int](capacity=1)
    out.append(objective_default_metric(objective))
    return out^


def eval_builtin_metric(
    metric: Int,
    ctx: BuiltinMetricContext,
    valid: Int,
    pred: List[Float64],
    target: List[Float64],
) raises -> Float64:
    """One built-in metric on one validation set's raw predictions.

    The one native call path from a metric code to a number, and the only
    place the three metric shapes are told apart. Which shape a code has is
    `metric_transform`, so the branch below is the registry's answer rather
    than a list of codes maintained here:

    - `TRANSFORM_OBJECTIVE_LINK`, nineteen codes: apply the *objective's*
      inverse link once and hand the result to `eval_metric_by_code`. This is
      LightGBM's rule and the reason `l2` on a poisson model scores expected
      counts rather than log-counts. A metric never applies a second
      transform.
    - `TRANSFORM_SOFTMAX`, the two multiclass codes: take the softmax of each
      row-major block of `n_classes` raw scores, then score the class codes.
    - `TRANSFORM_RAW`, the two ranking codes: rank within the validation
      set's own queries. Only the order matters, so no link is applied.

    `pred` is raw scores, the metric contract in this module's docstring, and
    is left untouched: the transformed copy is local, so the caller's running
    raw-score vector is still raw on the next round.
    """
    var transform = metric_transform(metric)
    var weight = ctx.weight_for(valid)

    if transform == TRANSFORM_SOFTMAX:
        if len(pred) != len(target) * ctx.n_classes:
            raise Error(
                "a multiclass metric needs n_rows * n_classes raw scores;"
                " got ",
                len(pred),
                " for ",
                len(target),
                " rows and ",
                ctx.n_classes,
                " classes",
            )
        var codes = _whole_number_codes(
            target, ctx.n_classes, String("multiclass label")
        )
        var probs = pred.copy()
        for r in range(len(target)):
            _softmax_inplace(probs, r * ctx.n_classes, ctx.n_classes)
        if metric == METRIC_MULTI_LOGLOSS:
            return multiclass_log_loss(probs, codes, ctx.n_classes, weight)
        if metric == METRIC_MULTI_ERROR:
            return multiclass_error(probs, codes, ctx.n_classes, weight)
        raise Error(
            "metric '",
            metric_canonical_name(metric),
            "' claims the softmax transform but is not a multiclass metric",
        )

    if transform == TRANSFORM_RAW:
        var grades = _whole_number_codes(
            target, 0, String("relevance label")
        )
        check_labels(grades)
        var groups = ctx.groups_for(valid)
        if metric == METRIC_NDCG:
            return ndcg(pred, grades, groups, ctx.cutoff)
        if metric == METRIC_MAP:
            return mean_average_precision(pred, grades, groups, ctx.cutoff)
        raise Error(
            "metric '",
            metric_canonical_name(metric),
            "' claims raw scores but is not a ranking metric",
        )

    if transform != TRANSFORM_OBJECTIVE_LINK:
        raise Error(
            "unknown metric transform ",
            transform,
            " for metric '",
            metric_canonical_name(metric),
            "'",
        )
    var scored = response_scale(ctx.objective, pred)
    var param = metric_scoring_param(metric, ctx.objective, ctx.alpha)
    return eval_metric_by_code(metric, scored, target, weight, param)


def _check_valid_sets(valid_sets: List[ValidSet], n_features: Int) raises:
    if len(valid_sets) == 0:
        raise Error("at least one validation set is required")
    for v in range(len(valid_sets)):
        if valid_sets[v].name.byte_length() == 0:
            raise Error("validation set names must not be empty")
        for other in range(v):
            if valid_sets[other].name == valid_sets[v].name:
                raise Error(
                    String(
                        "duplicate validation set name ", valid_sets[v].name
                    )
                )
        if len(valid_sets[v].target) != valid_sets[v].data.n_rows:
            raise Error(
                String(
                    "validation set ",
                    valid_sets[v].name,
                    " target length must equal its n_rows",
                )
            )
        if valid_sets[v].data.n_features != n_features:
            raise Error(
                String(
                    "validation set ",
                    valid_sets[v].name,
                    " must have the same features as the training data",
                )
            )


def _check_early_stopping(
    metrics: List[CustomMetric], early_stopping_rounds: Int, min_delta: Float64
) raises:
    if early_stopping_rounds < 0:
        raise Error("early_stopping_rounds must not be negative")
    if min_delta < 0.0:
        raise Error("min_delta must not be negative")
    if early_stopping_rounds == 0:
        return
    for m in range(len(metrics)):
        if metrics[m].use_for_early_stopping:
            return
    raise Error(
        "early stopping needs at least one metric with"
        " use_for_early_stopping"
    )


def _base_scores(valid_sets: List[ValidSet], base: Float64) -> List[
    List[Float64]
]:
    """One running raw-score vector per validation set."""
    var out = List[List[Float64]]()
    for v in range(len(valid_sets)):
        var scores = List[Float64](capacity=valid_sets[v].data.n_rows)
        for _ in range(valid_sets[v].data.n_rows):
            scores.append(base)
        out.append(scores^)
    return out^


def _update_valid_raw(
    mut valid_raw: List[List[Float64]],
    valid_sets: List[ValidSet],
    tree: Tree,
    learning_rate: Float64,
):
    for v in range(len(valid_sets)):
        for r in range(valid_sets[v].data.n_rows):
            valid_raw[v][r] += (
                learning_rate * tree.predict_row(valid_sets[v].data, r)
            )


def _eval_round[F: MetricSetFn & Copyable](
    metrics: MetricSuite[F],
    valid_sets: List[ValidSet],
    valid_raw: List[List[Float64]],
    round: Int,
    mut history: MetricHistory,
) raises:
    """Score every metric on every validation set and append the round to
    the history, rejecting non-finite values."""
    for v in range(len(valid_sets)):
        for m in range(metrics.n_metrics()):
            var value = metrics.evaluate(
                m, v, valid_raw[v], valid_sets[v].target
            )
            if not isfinite(value):
                raise Error(
                    String(
                        "metric ",
                        metrics.metrics[m].name,
                        " returned a non-finite value on validation set ",
                        valid_sets[v].name,
                        " at round ",
                        round,
                    )
                )
            history.values.append(value)


struct _StopState(Copyable, Movable):
    """Per (validation set, metric) best value and the round it happened,
    laid out as `[valid * n_metrics + metric]`."""

    var best_value: List[Float64]
    var best_round: List[Int]
    var n_metrics: Int

    def __init__(out self, n_valid: Int, n_metrics: Int):
        self.best_value = List[Float64](capacity=n_valid * n_metrics)
        self.best_round = List[Int](capacity=n_valid * n_metrics)
        for _ in range(n_valid * n_metrics):
            self.best_value.append(0.0)
            self.best_round.append(0)
        self.n_metrics = n_metrics

    def observe(
        mut self,
        round: Int,
        history: MetricHistory,
        metrics: List[CustomMetric],
        min_delta: Float64,
    ) raises:
        """Fold one round's values into the running bests. Round 0 seeds
        them; later rounds must improve by more than min_delta."""
        for v in range(history.n_valid()):
            for m in range(self.n_metrics):
                var i = v * self.n_metrics + m
                var value = history.value(round, v, m)
                if round == 0:
                    self.best_value[i] = value
                    self.best_round[i] = 0
                    continue
                var improved: Bool
                if metrics[m].higher_is_better:
                    improved = value > self.best_value[i] + min_delta
                else:
                    improved = value < self.best_value[i] - min_delta
                if improved:
                    self.best_value[i] = value
                    self.best_round[i] = round

    def exhausted(
        self,
        round: Int,
        metrics: List[CustomMetric],
        early_stopping_rounds: Int,
    ) -> Bool:
        """True once any watched pair has gone `early_stopping_rounds`
        rounds without improving."""
        for i in range(len(self.best_round)):
            var m = i % self.n_metrics
            if not metrics[m].use_for_early_stopping:
                continue
            if round - self.best_round[i] >= early_stopping_rounds:
                return True
        return False


def train_with_callbacks[
    F: MetricSetFn & Copyable, C: IterationFn & Copyable
](
    data: BinnedMatrix,
    target: List[Float64],
    valid_sets: List[ValidSet],
    objective: Int,
    params: BoosterParams,
    metrics: MetricSuite[F],
    callback: C,
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MetricTrainResult:
    """`train_with_metrics` with a per-iteration callback.

    `callback` runs twice per round, before the tree is grown and after the
    metrics are scored, and can steer the run: change the next round's
    hyperparameters, stop training, or fail it. See callback.mojo for the
    environment, the control codes, and how a learning-rate schedule reaches
    prediction.

    This is the one training loop for the metric path; `train_with_metrics`
    is this function with `no_callback`, so a run without callbacks and a run
    with inert ones take the same code and return the same model.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    _check_objective(objective, target, alpha)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    _check_valid_sets(valid_sets, data.n_features)
    metrics.check()
    _check_early_stopping(metrics.metrics, early_stopping_rounds, min_delta)

    var n = data.n_rows
    var base_score = _base_score(target, objective, sample_weight, alpha)
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)
    var valid_raw = _base_scores(valid_sets, base_score)

    var history = _new_history(metrics.metrics, valid_sets)
    var stop = _StopState(len(valid_sets), metrics.n_metrics())
    _eval_round(metrics, valid_sets, valid_raw, 0, history)
    stop.observe(0, history, metrics.metrics, min_delta)

    var renews = objective_renews_leaves(objective)
    var renew_w = renewal_weights(objective, target, sample_weight)
    var renew_a = renewal_alpha(objective, alpha)
    var signs = params.tree.monotone.active_signs()

    # The rate the booster was created with. While `baked` is False every
    # tree is shrunk by it at predict time; once a schedule moves off it the
    # shrinkage lives in the leaf values instead. See callback.mojo.
    var lr0 = params.learning_rate
    var baked = False
    var current = params.copy()
    var env = IterationEnv(
        params.copy(), history.valid_names.copy(), history.metric_names.copy()
    )

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    var stopped_early = False
    var callback_stopped = False
    for i in range(params.n_estimators):
        env.iteration = i
        env.evaluation.clear()
        env.params = current.copy()
        var before = callback(BEFORE_ITERATION, env)
        if before == ABORT:
            raise Error(
                String(
                    "training callback failed in the before-iteration phase"
                    " of round ",
                    i,
                )
            )
        if before == STOP:
            callback_stopped = True
            break
        check_resettable(current, env.params)
        current = env.params.copy()

        var lr = current.learning_rate
        if not baked and lr != lr0:
            for t in range(len(trees)):
                scale_tree_values(trees[t], lr0)
            baked = True

        refresh_bag(bag, bagging, n, i)
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess
        )
        goss_round(bag, grad, hess, goss, i, lr)
        var tree = grow_tree(data, grad, hess, current.tree, bag, i)
        if renews:
            _renew_leaf_values(
                tree, data, target, raw, renew_w, renew_a, bag, signs,
                current.tree.extra,
            )
        # Under bagging or GOSS a degenerate tree indicts the sample, not
        # the run, exactly as in train_with_valid. Tested before any
        # shrinkage is baked in, so the threshold means the same thing
        # whether or not a schedule is running.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            if bagging_enabled(bagging) or goss.enabled:
                continue
            break

        # A baked tree carries its own shrinkage, so the step that folds it
        # into the running scores must not apply the rate a second time.
        var step = lr0
        if baked:
            scale_tree_values(tree, lr)
            step = 1.0
        for r in range(n):
            raw[r] += step * tree.predict_row(data, r)
        _update_valid_raw(valid_raw, valid_sets, tree, step)
        trees.append(tree^)

        var round = len(trees)
        _eval_round(metrics, valid_sets, valid_raw, round, history)
        stop.observe(round, history, metrics.metrics, min_delta)

        env.iteration = i
        env.evaluation.clear()
        for v in range(history.n_valid()):
            for m in range(history.n_metrics()):
                env.evaluation.append(history.value(round, v, m))
        var after = callback(AFTER_ITERATION, env)
        if after == ABORT:
            raise Error(
                String(
                    "training callback failed in the after-iteration phase of"
                    " round ",
                    i,
                )
            )
        if after == STOP:
            callback_stopped = True
            break

        if early_stopping_rounds > 0 and stop.exhausted(
            round, metrics.metrics, early_stopping_rounds
        ):
            stopped_early = True
            break

    var best = stop.best_round[metrics.primary]
    var best_score = stop.best_value[metrics.primary]
    # A callback stop rolls back like an early stop: LightGBM's
    # EarlyStopException returns the booster at its best iteration whatever
    # the callback's reason for raising it was.
    if early_stopping_rounds > 0 or callback_stopped:
        while len(trees) > best:
            _ = trees.pop()
    return MetricTrainResult(
        Booster(
            trees^,
            base_score,
            1.0 if baked else lr0,
            objective,
            params.tree.monotone.copy(),
        ),
        history^,
        best,
        best_score,
        stopped_early or callback_stopped,
    )


def train_with_metrics[F: MetricSetFn & Copyable](
    data: BinnedMatrix,
    target: List[Float64],
    valid_sets: List[ValidSet],
    objective: Int,
    params: BoosterParams,
    metrics: MetricSuite[F],
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MetricTrainResult:
    """Train a built-in objective while scoring caller-supplied metrics.

    Same training contract as `train_with_valid` (objectives, sample
    weights, alpha, bagging, GOSS); the difference is that the validation
    signal comes from `metrics` rather than from the objective's own loss,
    that any number of validation sets can be scored, and that every value
    is kept in the returned history. See the module docstring for the
    early-stopping and truncation rules.

    `train_with_callbacks` is the same run with a per-iteration hook.
    """
    return train_with_callbacks(
        data,
        target,
        valid_sets,
        objective,
        params,
        metrics,
        no_callback,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        alpha,
        bagging,
        goss,
    )


def train_custom_with_metrics[G: GradHessFn, F: MetricSetFn & Copyable](
    data: BinnedMatrix,
    target: List[Float64],
    valid_sets: List[ValidSet],
    grad_hess: G,
    params: BoosterParams,
    metrics: MetricSuite[F],
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
) raises -> MetricTrainResult:
    """Train a caller-supplied objective while scoring caller-supplied
    metrics: `train_custom_with_valid` with a full metric suite instead of
    a single lower-is-better loss.

    The objective contract is the one in objective.mojo (unweighted
    derivatives, validated every round, base score 0.0 by default) and the
    metric contract is the one in this module's docstring. The two are
    independent: either can be customized without the other.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    _check_sample_weight(sample_weight, data.n_rows)
    _check_valid_sets(valid_sets, data.n_features)
    metrics.check()
    _check_early_stopping(metrics.metrics, early_stopping_rounds, min_delta)

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    for _ in range(n):
        raw.append(base_score)
    var valid_raw = _base_scores(valid_sets, base_score)

    var history = _new_history(metrics.metrics, valid_sets)
    var stop = _StopState(len(valid_sets), metrics.n_metrics())
    _eval_round(metrics, valid_sets, valid_raw, 0, history)
    stop.observe(0, history, metrics.metrics, min_delta)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var stopped_early = False
    for _ in range(params.n_estimators):
        grad_hess(raw, target, grad, hess)
        check_custom_grad_hess(grad, hess, n)
        _apply_sample_weight(grad, hess, sample_weight)
        var tree = grow_tree(data, grad, hess, params.tree)
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        _update_valid_raw(valid_raw, valid_sets, tree, params.learning_rate)
        trees.append(tree^)

        var round = len(trees)
        _eval_round(metrics, valid_sets, valid_raw, round, history)
        stop.observe(round, history, metrics.metrics, min_delta)
        if early_stopping_rounds > 0 and stop.exhausted(
            round, metrics.metrics, early_stopping_rounds
        ):
            stopped_early = True
            break

    var best = stop.best_round[metrics.primary]
    var best_score = stop.best_value[metrics.primary]
    if early_stopping_rounds > 0:
        while len(trees) > best:
            _ = trees.pop()
    return MetricTrainResult(
        Booster(
            trees^,
            base_score,
            params.learning_rate,
            CUSTOM,
            params.tree.monotone.copy(),
        ),
        history^,
        best,
        best_score,
        stopped_early,
    )


def train_with_metric[G: MetricFn & Copyable](
    data: BinnedMatrix,
    target: List[Float64],
    valid_sets: List[ValidSet],
    objective: Int,
    params: BoosterParams,
    name: String,
    metric: G,
    higher_is_better: Bool = False,
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MetricTrainResult:
    """`train_with_metrics` for the one-metric case: takes the metric
    callable directly, with no index dispatch to write."""

    def dispatch(
        metric_index: Int,
        valid_index: Int,
        pred: List[Float64],
        valid_target: List[Float64],
    ) raises {imm metric} -> Float64:
        return metric(pred, valid_target)

    var suite = MetricSuite(
        [CustomMetric(name, higher_is_better)], dispatch, 0
    )
    return train_with_metrics(
        data,
        target,
        valid_sets,
        objective,
        params,
        suite,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        alpha,
        bagging,
        goss,
    )


def train_with_builtin_metrics(
    data: BinnedMatrix,
    target: List[Float64],
    valid_sets: List[ValidSet],
    params: BoosterParams,
    metric_codes: List[Int],
    ctx: BuiltinMetricContext,
    primary: Int = 0,
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MetricTrainResult:
    """`train_with_metrics` driven by built-in metric codes instead of a
    caller-supplied callable.

    The objective and its scalar parameter come from `ctx` rather than from
    arguments of their own, which is the point: the metric scores the same
    objective at the same parameter the model trained with, and there is no
    pair of arguments that could disagree. `BuiltinMetricContext.single_output`
    builds one.

    Names, directions, transforms, and compatibility all come from the
    registry (see `eval_builtin_metric` and `builtin_metric_metadata`), so a
    metric that cannot score this objective's task is refused here, before
    the first tree, rather than scoring something meaningless for a full run.

    `primary` indexes `metric_codes` and names the metric whose best round
    the ensemble is truncated to. Every named metric is watched for early
    stopping. For a caller-supplied metric, or a mix of the two, build a
    `MetricSuite` and call `train_with_metrics`.
    """
    ctx.check(metric_codes, len(valid_sets))
    var codes = metric_codes.copy()
    var context = ctx.copy()

    def dispatch(
        metric_index: Int,
        valid_index: Int,
        pred: List[Float64],
        valid_target: List[Float64],
    ) raises {imm codes, imm context} -> Float64:
        return eval_builtin_metric(
            codes[metric_index], context, valid_index, pred, valid_target
        )

    var suite = MetricSuite(
        builtin_metric_metadata(metric_codes), dispatch, primary
    )
    return train_with_metrics(
        data,
        target,
        valid_sets,
        ctx.objective,
        params,
        suite,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        ctx.alpha,
        bagging,
        goss,
    )


def train_custom_with_builtin_metrics[G: GradHessFn](
    data: BinnedMatrix,
    target: List[Float64],
    valid_sets: List[ValidSet],
    grad_hess: G,
    params: BoosterParams,
    metric_codes: List[Int],
    ctx: BuiltinMetricContext,
    primary: Int = 0,
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    base_score: Float64 = 0.0,
) raises -> MetricTrainResult:
    """`train_custom_with_metrics` driven by built-in metric codes.

    A custom objective has no default metric, so `metric_codes` is not
    optional and `default_metric_codes(CUSTOM)` raises rather than guessing.
    What it does have is a task: `objective_task(CUSTOM)` is regression, the
    family its single real-valued output belongs to, so the thirteen
    regression metrics score it and the binary, multiclass, and ranking ones
    are refused.

    The predictions those metrics see are raw scores, because
    `objective_link(CUSTOM)` is the identity: the framework does not know the
    caller's link, so it applies none. A metric that wants probabilities
    should be a caller-supplied one that applies its own.
    """
    if ctx.objective != CUSTOM:
        raise Error(
            "train_custom_with_builtin_metrics needs a context built for the"
            " CUSTOM objective; got '",
            objective_canonical_name(ctx.objective),
            "'",
        )
    ctx.check(metric_codes, len(valid_sets))
    var codes = metric_codes.copy()
    var context = ctx.copy()

    def dispatch(
        metric_index: Int,
        valid_index: Int,
        pred: List[Float64],
        valid_target: List[Float64],
    ) raises {imm codes, imm context} -> Float64:
        return eval_builtin_metric(
            codes[metric_index], context, valid_index, pred, valid_target
        )

    var suite = MetricSuite(
        builtin_metric_metadata(metric_codes), dispatch, primary
    )
    return train_custom_with_metrics(
        data,
        target,
        valid_sets,
        grad_hess,
        params,
        suite,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        base_score,
    )


def train_multiclass_with_builtin_metrics(
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    valid_sets: List[ValidSet],
    params: BoosterParams,
    metric_codes: List[Int],
    ctx: BuiltinMetricContext,
    primary: Int = 0,
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MetricMulticlassTrainResult:
    """`train_multiclass_with_metrics` driven by built-in metric codes.

    `multi_logloss` and `multi_error` are the two metrics a softmax model
    accepts, and both take the softmax of each row-major block themselves
    (`metric_transform` is `TRANSFORM_SOFTMAX`), so what reaches them is the
    raw scores the trainer already holds. `BuiltinMetricContext.multiclass`
    builds the context; its `n_classes` must be the one being trained, since
    a metric that read a different one would be scoring a different model.
    """
    if ctx.n_classes != n_classes:
        raise Error(
            "BuiltinMetricContext carries n_classes=",
            ctx.n_classes,
            " but the run trains ",
            n_classes,
        )
    if ctx.objective != MULTICLASS:
        raise Error(
            "train_multiclass_with_builtin_metrics needs a context built for"
            " the multiclass objective; got '",
            objective_canonical_name(ctx.objective),
            "'",
        )
    ctx.check(metric_codes, len(valid_sets))
    var codes = metric_codes.copy()
    var context = ctx.copy()

    def dispatch(
        metric_index: Int,
        valid_index: Int,
        pred: List[Float64],
        valid_target: List[Float64],
    ) raises {imm codes, imm context} -> Float64:
        return eval_builtin_metric(
            codes[metric_index], context, valid_index, pred, valid_target
        )

    var suite = MetricSuite(
        builtin_metric_metadata(metric_codes), dispatch, primary
    )
    return train_multiclass_with_metrics(
        data,
        labels,
        n_classes,
        valid_sets,
        params,
        suite,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        bagging,
        goss,
    )


def train_ranker_with_builtin_metrics(
    data: BinnedMatrix,
    labels: List[Int],
    groups: RankGroups,
    valid_sets: List[ValidSet],
    params: BoosterParams,
    metric_codes: List[Int],
    ctx: BuiltinMetricContext,
    primary: Int = 0,
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    rank_params: RankerParams = RankerParams.default(),
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> MetricTrainResult:
    """`train_ranker_with_metrics` driven by built-in metric codes.

    This is the boilerplate the module docstring warns about, removed: a
    ranking metric needs the *validation* set's own query boundaries, and
    until now the only way to give it them was to capture them in a closure
    by hand. `BuiltinMetricContext.ranking` carries one `RankGroups` per
    validation set instead, in the same order, and `ctx.check` refuses a
    count that does not match rather than letting a metric rank the wrong
    documents together.

    `ndcg` and `map` read raw scores (`metric_transform` is
    `TRANSFORM_RAW`), since only the order within a query matters, and take
    no weights, which is LightGBM's behavior. `groups` here is the *training*
    grouping and is unrelated to the validation ones in `ctx`.
    """
    if ctx.objective != LAMBDARANK:
        raise Error(
            "train_ranker_with_builtin_metrics needs a context built for the"
            " lambdarank objective; got '",
            objective_canonical_name(ctx.objective),
            "'",
        )
    ctx.check(metric_codes, len(valid_sets))
    var codes = metric_codes.copy()
    var context = ctx.copy()

    def dispatch(
        metric_index: Int,
        valid_index: Int,
        pred: List[Float64],
        valid_target: List[Float64],
    ) raises {imm codes, imm context} -> Float64:
        return eval_builtin_metric(
            codes[metric_index], context, valid_index, pred, valid_target
        )

    var suite = MetricSuite(
        builtin_metric_metadata(metric_codes), dispatch, primary
    )
    return train_ranker_with_metrics(
        data,
        labels,
        groups,
        valid_sets,
        params,
        suite,
        early_stopping_rounds,
        min_delta,
        rank_params,
        sample_weight,
        bagging,
    )


def fit_with_metrics[
    F: MetricSetFn & Copyable, features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    target: List[Float64],
    valid_sets: List[RawValidSet],
    objective: Int,
    params: BoosterParams,
    metrics: MetricSuite[F],
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> MetricFitResult:
    """`train_with_metrics` on raw, column-major features
    (`features[f * n_rows + r]`), the `fit` counterpart.

    Validation sets are binned with the mapper fitted on the training data,
    which is what a deployed model would do to them. `max_bins`,
    `use_missing`, and `categorical_features` therefore apply to the
    validation sets as well. CPU only.
    """
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    return fit_with_callbacks(
        features,
        n_rows,
        n_features,
        target,
        valid_sets,
        objective,
        params,
        metrics,
        no_callback,
        early_stopping_rounds,
        min_delta,
        max_bins,
        sample_weight,
        alpha,
        bagging,
        goss,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )


def fit_with_callbacks[
    F: MetricSetFn & Copyable, C: IterationFn & Copyable,
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    target: List[Float64],
    valid_sets: List[RawValidSet],
    objective: Int,
    params: BoosterParams,
    metrics: MetricSuite[F],
    callback: C,
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    alpha: Float64 = 0.9,
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> MetricFitResult:
    """`train_with_callbacks` on raw, column-major features: `fit_with_metrics`
    with a per-iteration hook. See callback.mojo for the contract."""
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var binned = _bin_valid_sets(mapper, valid_sets, n_features)
    var result = train_with_callbacks(
        data,
        target,
        binned,
        objective,
        params,
        metrics,
        callback,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        alpha,
        bagging,
        goss,
    )
    return MetricFitResult(
        Model(mapper^, result.booster.copy()),
        result.history.copy(),
        result.best_iteration,
        result.best_score,
        result.stopped_early,
    )


def _new_history(
    metrics: List[CustomMetric], valid_sets: List[ValidSet]
) -> MetricHistory:
    var metric_names = List[String](capacity=len(metrics))
    for m in range(len(metrics)):
        metric_names.append(metrics[m].name)
    var valid_names = List[String](capacity=len(valid_sets))
    for v in range(len(valid_sets)):
        valid_names.append(valid_sets[v].name)
    return MetricHistory(metric_names^, valid_names^)


@fieldwise_init
struct MetricMulticlassTrainResult(Copyable, Movable):
    """`MetricTrainResult` for the softmax trainer: the same record around a
    `MulticlassBooster`. `best_iteration` counts rounds, and a round is one
    tree per class, so the ensemble it names holds
    `best_iteration * n_classes` trees."""

    var booster: MulticlassBooster
    var history: MetricHistory
    var best_iteration: Int
    var best_score: Float64
    var stopped_early: Bool


@fieldwise_init
struct MetricMulticlassFitResult(Copyable, Movable):
    """`MetricMulticlassTrainResult` for the raw-feature entry point: the
    same record around a `MulticlassModel`, which carries the fitted bin
    mapper."""

    var model: MulticlassModel
    var history: MetricHistory
    var best_iteration: Int
    var best_score: Float64
    var stopped_early: Bool


def _valid_label_codes(
    valid_sets: List[ValidSet], n_classes: Int
) raises -> List[List[Int]]:
    """Validation labels as the Int class codes the softmax trainer needs.

    They travel in `ValidSet.target` as Float64 because one `ValidSet` type
    serves every trainer; only whole numbers in 0..n_classes-1 mean anything
    here, and anything else is rejected rather than truncated.
    """
    var out = List[List[Int]]()
    for v in range(len(valid_sets)):
        var codes = List[Int](capacity=len(valid_sets[v].target))
        for r in range(len(valid_sets[v].target)):
            var value = valid_sets[v].target[r]
            var ok = isfinite(value)
            var code = Int(value) if ok else 0
            if ok:
                ok = Float64(code) == value and 0 <= code < n_classes
            if not ok:
                raise Error(
                    String(
                        "validation set ",
                        valid_sets[v].name,
                        " has a label outside 0..n_classes-1",
                    )
                )
            codes.append(code)
        out.append(codes^)
    return out^


def _relevance_codes(valid_sets: List[ValidSet]) raises -> List[List[Int]]:
    """Validation relevance labels as the Int grades a ranking metric needs,
    checked against the same range `check_labels` enforces for training."""
    var out = List[List[Int]]()
    for v in range(len(valid_sets)):
        var codes = List[Int](capacity=len(valid_sets[v].target))
        for r in range(len(valid_sets[v].target)):
            var value = valid_sets[v].target[r]
            var ok = isfinite(value)
            var code = Int(value) if ok else 0
            if ok:
                ok = Float64(code) == value
            if not ok:
                raise Error(
                    String(
                        "validation set ",
                        valid_sets[v].name,
                        " has a non-integer relevance label",
                    )
                )
            codes.append(code)
        check_labels(codes)
        out.append(codes^)
    return out^


def _multiclass_base_scores(
    labels: List[Int], n_classes: Int, sample_weight: List[Float64]
) raises -> List[Float64]:
    """Log class priors of the training labels, weighted when
    `sample_weight` is given: the base scores `train_multiclass` starts
    from."""
    var class_w = List[Float64](capacity=n_classes)
    for _ in range(n_classes):
        class_w.append(0.0)
    var total_w = 0.0
    for r in range(len(labels)):
        if labels[r] < 0 or labels[r] >= n_classes:
            raise Error("label out of range")
        var w = sample_weight[r] if len(sample_weight) > 0 else 1.0
        class_w[labels[r]] += w
        total_w += w
    if total_w <= 0.0:
        raise Error("sample_weight must have a positive sum")
    var base_scores = List[Float64](capacity=n_classes)
    for k in range(n_classes):
        base_scores.append(log(_clamp_prob(class_w[k] / total_w)))
    return base_scores^


def _multiclass_base_raw(
    valid_sets: List[ValidSet], base_scores: List[Float64], n_classes: Int
) -> List[List[Float64]]:
    """One running row-major raw-score vector per validation set,
    `raw[r * n_classes + k]`, seeded with the training base scores."""
    var out = List[List[Float64]]()
    for v in range(len(valid_sets)):
        var rows = valid_sets[v].data.n_rows
        var scores = List[Float64](capacity=rows * n_classes)
        for _ in range(rows):
            for k in range(n_classes):
                scores.append(base_scores[k])
        out.append(scores^)
    return out^


def _update_multiclass_valid_raw(
    mut valid_raw: List[List[Float64]],
    valid_sets: List[ValidSet],
    tree: Tree,
    learning_rate: Float64,
    n_classes: Int,
    k: Int,
):
    for v in range(len(valid_sets)):
        for r in range(valid_sets[v].data.n_rows):
            valid_raw[v][r * n_classes + k] += (
                learning_rate * tree.predict_row(valid_sets[v].data, r)
            )


def train_multiclass_with_metrics[F: MetricSetFn & Copyable](
    data: BinnedMatrix,
    labels: List[Int],
    n_classes: Int,
    valid_sets: List[ValidSet],
    params: BoosterParams,
    metrics: MetricSuite[F],
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
) raises -> MetricMulticlassTrainResult:
    """Train a softmax ensemble while scoring caller-supplied metrics.

    Same training contract as `train_multiclass_with_valid` (weights, class
    priors as base scores, one shared bag or GOSS sample per round); the
    difference is that the validation signal comes from `metrics` rather
    than from the multiclass log loss, that any number of validation sets
    can be scored, and that every value is kept in the returned history.

    Metrics receive the row-major raw scores of a validation set, one block
    of `n_classes` per row, and its labels as Float64 whole numbers. See the
    module docstring for the early-stopping and truncation rules; a round is
    one tree per class here, so truncating to `best_iteration` drops
    `n_classes` trees at a time.
    """
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    if n_classes < 2:
        raise Error("n_classes must be at least 2")
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_goss(goss, bagging)
    params.tree.monotone.check_features(data.n_features)
    _check_valid_sets(valid_sets, data.n_features)
    _ = _valid_label_codes(valid_sets, n_classes)
    metrics.check()
    _check_early_stopping(metrics.metrics, early_stopping_rounds, min_delta)

    var n = data.n_rows
    var base_scores = _multiclass_base_scores(labels, n_classes, sample_weight)
    var raw = List[Float64](capacity=n * n_classes)
    for _ in range(n):
        for k in range(n_classes):
            raw.append(base_scores[k])
    var prob = List[Float64](capacity=n * n_classes)
    for _ in range(n * n_classes):
        prob.append(0.0)
    var valid_raw = _multiclass_base_raw(valid_sets, base_scores, n_classes)

    var history = _new_history(metrics.metrics, valid_sets)
    var stop = _StopState(len(valid_sets), metrics.n_metrics())
    _eval_round(metrics, valid_sets, valid_raw, 0, history)
    stop.observe(0, history, metrics.metrics, min_delta)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    var n_rounds = 0
    var stopped_early = False
    for i in range(params.n_estimators):
        refresh_bag(bag, bagging, n, i)
        for r in range(n):
            for k in range(n_classes):
                prob[r * n_classes + k] = raw[r * n_classes + k]
            _softmax_inplace(prob, r * n_classes, n_classes)

        # One shared sample for the whole round, drawn before any class's
        # tree so that every class is grown on the same rows.
        var selection = GossSelection.all_rows()
        if goss.active(i, params.learning_rate):
            selection = _multiclass_goss_select(
                prob, labels, n_classes, sample_weight, goss, i
            )
            bag = selection.rows.copy()

        var made_progress = False
        var round_trees = List[Tree]()
        for k in range(n_classes):
            _fill_softmax_grad_hess(
                prob, labels, k, n_classes, sample_weight, grad, hess
            )
            apply_goss_scaling(selection, grad, hess)
            var tree = grow_tree(
                data, grad, hess, params.tree, bag, i * n_classes + k
            )
            if tree.n_leaves > 1 or abs(tree.value[0]) >= 1e-12:
                made_progress = True
            round_trees.append(tree^)

        # A round that moved nothing indicts this sample under bagging or
        # GOSS and the run otherwise, exactly as in train_multiclass. The
        # scores are only updated once the round is known to count, so a
        # dropped round leaves no trace in the history.
        if not made_progress:
            if bagging_enabled(bagging) or goss.enabled:
                continue
            break

        for k in range(n_classes):
            for r in range(n):
                raw[r * n_classes + k] += (
                    params.learning_rate * round_trees[k].predict_row(data, r)
                )
            _update_multiclass_valid_raw(
                valid_raw,
                valid_sets,
                round_trees[k],
                params.learning_rate,
                n_classes,
                k,
            )
            trees.append(round_trees[k].copy())
        n_rounds += 1

        _eval_round(metrics, valid_sets, valid_raw, n_rounds, history)
        stop.observe(n_rounds, history, metrics.metrics, min_delta)
        if early_stopping_rounds > 0 and stop.exhausted(
            n_rounds, metrics.metrics, early_stopping_rounds
        ):
            stopped_early = True
            break

    var best = stop.best_round[metrics.primary]
    var best_score = stop.best_value[metrics.primary]
    if early_stopping_rounds > 0:
        while len(trees) > best * n_classes:
            _ = trees.pop()
    return MetricMulticlassTrainResult(
        MulticlassBooster(
            trees^,
            base_scores^,
            n_classes,
            params.learning_rate,
            params.tree.monotone.copy(),
        ),
        history^,
        best,
        best_score,
        stopped_early,
    )


def train_ranker_with_metrics[F: MetricSetFn & Copyable](
    data: BinnedMatrix,
    labels: List[Int],
    groups: RankGroups,
    valid_sets: List[ValidSet],
    params: BoosterParams,
    metrics: MetricSuite[F],
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    rank_params: RankerParams = RankerParams.default(),
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
) raises -> MetricTrainResult:
    """Train a LambdaRank ensemble while scoring caller-supplied metrics.

    Same training contract as `train_ranker_with_valid` (base score 0,
    query-wise bagging, weights on training rows only); the difference is
    that the validation signal comes from `metrics`, that any number of
    validation sets can be scored, and that every value is kept in the
    returned history.

    A validation set's query boundaries are not passed to the metric, which
    a ranking metric needs: hold them in the callable, as
    `fit_ranker_with_metrics`'s callers do. Relevance labels ride in
    `ValidSet.target` as Float64 whole numbers and are range-checked here.

    Unlike `train_ranker_with_valid`, which starts from "no ranking to
    beat", round 0 is evaluated and recorded like any other round, so a
    `best_iteration` of 0 means no tree beat the all-zero scores the
    ensemble starts from.
    """
    if len(labels) != data.n_rows:
        raise Error("labels length must equal n_rows")
    check_groups(groups, data.n_rows)
    check_labels(labels)
    check_ranker_params(rank_params)
    _check_sample_weight(sample_weight, data.n_rows)
    check_bagging(bagging)
    _check_valid_sets(valid_sets, data.n_features)
    _ = _relevance_codes(valid_sets)
    metrics.check()
    _check_early_stopping(metrics.metrics, early_stopping_rounds, min_delta)

    var n = data.n_rows
    var raw = List[Float64](capacity=n)
    raw.resize(n, 0.0)
    var valid_raw = _base_scores(valid_sets, 0.0)
    var inverse_max_dcg = _inverse_max_dcgs(
        labels, groups, rank_params.truncation_level
    )
    var discounts = _discounts(groups.max_size())

    var history = _new_history(metrics.metrics, valid_sets)
    var stop = _StopState(len(valid_sets), metrics.n_metrics())
    _eval_round(metrics, valid_sets, valid_raw, 0, history)
    stop.observe(0, history, metrics.metrics, min_delta)

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var query_bag = List[Int]()
    var row_bag = List[Int]()
    var stopped_early = False
    for i in range(params.n_estimators):
        _refresh_query_bag(query_bag, row_bag, groups, bagging, i)
        _fill_lambdas(
            raw,
            labels,
            groups,
            inverse_max_dcg,
            discounts,
            rank_params,
            sample_weight,
            grad,
            hess,
        )
        var tree = grow_tree(data, grad, hess, params.tree, row_bag, i)
        # No pair left to separate; under bagging that is a statement about
        # this bag, so the round is skipped and the next bag gets its turn.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            if bagging_enabled(bagging):
                continue
            break

        for r in range(n):
            raw[r] += params.learning_rate * tree.predict_row(data, r)
        _update_valid_raw(valid_raw, valid_sets, tree, params.learning_rate)
        trees.append(tree^)

        var round = len(trees)
        _eval_round(metrics, valid_sets, valid_raw, round, history)
        stop.observe(round, history, metrics.metrics, min_delta)
        if early_stopping_rounds > 0 and stop.exhausted(
            round, metrics.metrics, early_stopping_rounds
        ):
            stopped_early = True
            break

    var best = stop.best_round[metrics.primary]
    var best_score = stop.best_value[metrics.primary]
    if early_stopping_rounds > 0:
        while len(trees) > best:
            _ = trees.pop()
    return MetricTrainResult(
        Booster(
            trees^,
            0.0,
            params.learning_rate,
            LAMBDARANK,
            params.tree.monotone.copy(),
        ),
        history^,
        best,
        best_score,
        stopped_early,
    )


def _bin_valid_sets(
    mapper: BinMapper,
    valid_sets: List[RawValidSet],
    n_features: Int,
) raises -> List[ValidSet]:
    """Bin every raw validation set with the mapper fitted on the training
    data, which is what a deployed model would do to them."""
    var binned = List[ValidSet]()
    for v in range(len(valid_sets)):
        if len(valid_sets[v].features) != valid_sets[v].n_rows * n_features:
            raise Error(
                String(
                    "validation set ",
                    valid_sets[v].name,
                    " must have n_rows * n_features feature values",
                )
            )
        binned.append(
            ValidSet(
                valid_sets[v].name,
                mapper.transform(valid_sets[v].features, valid_sets[v].n_rows),
                valid_sets[v].target.copy(),
            )
        )
    return binned^


def fit_multiclass_with_metrics[
    F: MetricSetFn & Copyable, features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    labels: List[Int],
    n_classes: Int,
    valid_sets: List[RawValidSet],
    params: BoosterParams,
    metrics: MetricSuite[F],
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    goss: GossParams = GossParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> MetricMulticlassFitResult:
    """`train_multiclass_with_metrics` on raw, column-major features
    (`features[f * n_rows + r]`), the `fit_multiclass` counterpart. CPU
    only."""
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var binned = _bin_valid_sets(mapper, valid_sets, n_features)
    var result = train_multiclass_with_metrics(
        data,
        labels,
        n_classes,
        binned,
        params,
        metrics,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        bagging,
        goss,
    )
    return MetricMulticlassFitResult(
        MulticlassModel(mapper^, result.booster.copy()),
        result.history.copy(),
        result.best_iteration,
        result.best_score,
        result.stopped_early,
    )


def fit_ranker_with_metrics[
    F: MetricSetFn & Copyable, features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    labels: List[Int],
    group_counts: List[Int],
    valid_sets: List[RawValidSet],
    params: BoosterParams,
    metrics: MetricSuite[F],
    early_stopping_rounds: Int = 0,
    min_delta: Float64 = 0.0,
    rank_params: RankerParams = RankerParams.default(),
    max_bins: Int = 255,
    sample_weight: List[Float64] = [],
    bagging: BaggingParams = BaggingParams.disabled(),
    use_missing: Bool = True,
    categorical_features: List[Int] = [],
) raises -> MetricFitResult:
    """`train_ranker_with_metrics` on raw, column-major features
    (`features[f * n_rows + r]`), the `fit_ranker` counterpart. CPU only."""
    var groups = groups_from_counts(group_counts)
    var mapper = fit_bins(
        features,
        n_rows,
        n_features,
        max_bins,
        use_missing=use_missing,
        categorical_features=categorical_features,
    )
    var data = mapper.transform(features, n_rows)
    var binned = _bin_valid_sets(mapper, valid_sets, n_features)
    var result = train_ranker_with_metrics(
        data,
        labels,
        groups,
        binned,
        params,
        metrics,
        early_stopping_rounds,
        min_delta,
        rank_params,
        sample_weight,
        bagging,
    )
    return MetricFitResult(
        Model(mapper^, result.booster.copy()),
        result.history.copy(),
        result.best_iteration,
        result.best_score,
        result.stopped_early,
    )
