"""Custom validation metrics.

A custom metric is a callable that scores validation predictions against
validation labels and returns one scalar:

    def my_metric(pred: List[Float64], target: List[Float64]) raises -> Float64:
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
- Validation rows are never weighted or bagged, matching `train_with_valid`.
- Single-output only, as for custom objectives: `train_multiclass` has no
  custom-metric entry point.
- CPU only. There is no GPU trainer with a validation loop; the metric
  itself is host code either way.
"""

from std.math import exp, isfinite

from .bagging import BaggingParams, bagging_enabled, check_bagging, refresh_bag
from .binning import BinMapper, BinnedMatrix, fit_bins
from .boosting import (
    BINARY_LOGISTIC,
    CUSTOM,
    L1,
    POISSON,
    QUANTILE,
    Booster,
    BoosterParams,
    _base_score,
    _check_goss,
    _check_objective,
    _check_sample_weight,
    _fill_grad_hess,
    _renew_leaf_values,
    _sigmoid,
)
from .goss import GossParams, goss_round
from .model import Model
from .objective import (
    GradHessFn,
    _apply_sample_weight,
    check_custom_grad_hess,
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


def response_scale(objective: Int, raw: List[Float64]) -> List[Float64]:
    """Apply an objective's inverse link to raw scores: the sigmoid for
    BINARY_LOGISTIC, exp for POISSON, identity otherwise (including CUSTOM,
    whose link the framework does not know).

    Metrics receive raw scores; this is the one call that turns them into
    the probabilities a log loss or a calibration metric wants.
    """
    var out = List[Float64](capacity=len(raw))
    if objective == BINARY_LOGISTIC:
        for r in range(len(raw)):
            out.append(_sigmoid(raw[r]))
    elif objective == POISSON:
        for r in range(len(raw)):
            out.append(exp(raw[r]))
    else:
        for r in range(len(raw)):
            out.append(raw[r])
    return out^


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

    var trees = List[Tree]()
    var grad = List[Float64](capacity=n)
    var hess = List[Float64](capacity=n)
    var bag = List[Int]()
    var stopped_early = False
    for i in range(params.n_estimators):
        refresh_bag(bag, bagging, n, i)
        _fill_grad_hess(
            raw, target, objective, sample_weight, alpha, grad, hess
        )
        goss_round(bag, grad, hess, goss, i, params.learning_rate)
        var tree = grow_tree(data, grad, hess, params.tree, bag, i)
        if objective == QUANTILE:
            _renew_leaf_values(
                tree, data, target, raw, sample_weight, alpha, bag
            )
        elif objective == L1:
            _renew_leaf_values(
                tree, data, target, raw, sample_weight, 0.5, bag
            )
        # Under bagging or GOSS a degenerate tree indicts the sample, not
        # the run, exactly as in train_with_valid.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            if bagging_enabled(bagging) or goss.enabled:
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
        Booster(trees^, base_score, params.learning_rate, objective),
        history^,
        best,
        best_score,
        stopped_early,
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
        Booster(trees^, base_score, params.learning_rate, CUSTOM),
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


def fit_with_metrics[F: MetricSetFn & Copyable](
    features: List[Float64],
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
) raises -> MetricFitResult:
    """`train_with_metrics` on raw, column-major features
    (`features[f * n_rows + r]`), the `fit` counterpart.

    Validation sets are binned with the mapper fitted on the training data,
    which is what a deployed model would do to them. CPU only.
    """
    var mapper = fit_bins(features, n_rows, n_features, max_bins)
    var data = mapper.transform(features, n_rows)
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
    var result = train_with_metrics(
        data,
        target,
        binned,
        objective,
        params,
        metrics,
        early_stopping_rounds,
        min_delta,
        sample_weight,
        alpha,
        bagging,
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
