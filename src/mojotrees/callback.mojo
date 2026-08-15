"""Per-iteration training callbacks.

A callback observes, and may steer, a training run at two points in every
boosting round:

    BEFORE_ITERATION   before the round's tree is grown
    AFTER_ITERATION    after it is grown and the metrics have been scored

It is one callable of type `IterationFn`, taking the phase and a mutable
`IterationEnv` and returning a control code:

    def watch(phase: Int, mut env: IterationEnv) raises -> Int:
        if phase == AFTER_ITERATION and env.evaluation[0] < 0.01:
            return STOP
        return CONTINUE

`train_with_callbacks` in custom_metric.mojo runs the loop; this module owns
the types the two sides agree on, so the Python bridge in
bindings/_mojotrees.mojo and a native caller see one contract.

The environment
---------------
`IterationEnv` carries what LightGBM's `CallbackEnv` namedtuple carries, with
the names it uses:

- `iteration`, the 0-based round about to run (BEFORE) or just finished
  (AFTER). LightGBM's `log_evaluation` prints `iteration + 1`; this is the
  same 0-based number it prints from.
- `begin_iteration` and `end_iteration`, the half-open range of rounds the
  run will attempt. `end_iteration` is `n_estimators`; a run that stops early
  never reaches it.
- `params`, the hyperparameters this round will train with. Mutable in the
  BEFORE phase, which is how a parameter schedule works; see below.
- `evaluation`, this round's metric values, `[valid * n_metrics + metric]`,
  with `valid_names` and `metric_names` naming the axes. Empty in the BEFORE
  phase, because nothing has been scored yet.

Control codes
-------------
- `CONTINUE` grows the round and keeps going.
- `STOP` ends training. The ensemble is then truncated to the best round of
  the primary metric on the first validation set, the same round
  `early_stopping_rounds` would have kept, and `stopped_early` is True. This
  matches LightGBM, whose `EarlyStopException` also rolls the booster back to
  `best_iteration` no matter why the callback raised it.
- `ABORT` means the callback failed and training must not continue. The
  trainer raises; nothing is returned. The Python bridge uses this to carry a
  Python exception across the boundary intact: it stashes the exception
  object, returns `ABORT`, and re-raises the original on the other side, so
  the caller sees its own exception type rather than a flattened message.

Parameter schedules
-------------------
A BEFORE callback may assign to `env.params` to change the next round's
hyperparameters, which is LightGBM's `reset_parameter`. Only the fields the
loop re-reads every round can change, and `check_resettable` rejects the
rest rather than accepting a value that would be silently ignored:

    learning_rate, num_leaves, max_depth, min_data_in_leaf,
    min_sum_hessian_in_leaf, lambda_l1, lambda_l2, feature_fraction,
    feature_fraction_bynode

`n_estimators` is not resettable: the round count is the loop bound and is
fixed when training starts. Neither are the constraint sets (monotone,
interaction, categorical), which are properties of the fitted model rather
than per-round knobs.

How a learning-rate schedule reaches prediction
-----------------------------------------------
`Booster` applies one learning rate to every tree at predict time
(`predict_raw_row`), so a rate that varies by round cannot be represented
that way. When, and only when, a schedule actually changes the rate, the
shrinkage is instead baked into the leaf values and the booster's stored
rate becomes 1.0 -- which is what LightGBM does for every model.

The switch is lazy, and that is the point. Until a round asks for a
different rate, nothing changes, so a run with callbacks that do not touch
`learning_rate` returns a model identical to the same run without callbacks:
adding a logging callback must not perturb the model. At the first round that
does change it, the trees already grown are multiplied by the original rate
once. Prediction would have computed `rate * value` for those trees, and the
training accumulator did compute exactly that, so the rewrite is the same
arithmetic rather than an approximation of it.
"""

from .boosting import BoosterParams
from .tree import Tree
from .validation import check_booster_ranges

comptime BEFORE_ITERATION = 0
"""Phase: the round's tree has not been grown yet; `params` is mutable."""

comptime AFTER_ITERATION = 1
"""Phase: the tree is grown and `evaluation` holds this round's metrics."""

comptime CONTINUE = 0
"""Control code: grow the round and keep going."""

comptime STOP = 1
"""Control code: stop training and truncate to the primary metric's best
round, as LightGBM's `EarlyStopException` does."""

comptime ABORT = 2
"""Control code: the callback failed; the trainer raises."""


struct IterationEnv(Copyable, Movable):
    """What a callback sees, and the one field it may write.

    `params` is the round's hyperparameters. Assigning to it in the BEFORE
    phase schedules the next round; the trainer validates the result with
    `check_resettable` before using it. Writing it in the AFTER phase has no
    effect, because the round is already grown.
    """

    var iteration: Int
    var begin_iteration: Int
    var end_iteration: Int
    var params: BoosterParams
    var evaluation: List[Float64]
    var valid_names: List[String]
    var metric_names: List[String]

    def __init__(
        out self,
        var params: BoosterParams,
        var valid_names: List[String],
        var metric_names: List[String],
        begin_iteration: Int = 0,
    ):
        self.iteration = begin_iteration
        self.begin_iteration = begin_iteration
        self.end_iteration = params.n_estimators
        self.params = params^
        self.evaluation = List[Float64]()
        self.valid_names = valid_names^
        self.metric_names = metric_names^

    def n_valid(self) -> Int:
        return len(self.valid_names)

    def n_metrics(self) -> Int:
        return len(self.metric_names)

    def value(self, valid: Int, metric: Int) raises -> Float64:
        """This round's value of one metric on one validation set. Raises in
        the BEFORE phase, where nothing has been scored."""
        if len(self.evaluation) == 0:
            raise Error(
                "no evaluation results in the before-iteration phase; metrics"
                " are scored after the round's tree is grown"
            )
        if valid < 0 or valid >= self.n_valid():
            raise Error("validation-set index out of range")
        if metric < 0 or metric >= self.n_metrics():
            raise Error("metric index out of range")
        return self.evaluation[valid * self.n_metrics() + metric]


comptime IterationFn = def (Int, mut IterationEnv) raises -> Int
"""(phase, environment) in, a control code out. Called twice per round."""


def no_callback(phase: Int, mut env: IterationEnv) raises -> Int:
    """The callback a run without callbacks uses, so there is one training
    loop rather than two. Returns `CONTINUE` and touches nothing."""
    return CONTINUE


def check_resettable(before: BoosterParams, after: BoosterParams) raises:
    """Reject a parameter reset the training loop cannot honor, and range
    check the ones it can.

    `before` is what the round was going to use and `after` is what the
    callback left in `env.params`. A field outside the resettable set is an
    error rather than a silent no-op: a schedule that believes it changed
    `n_estimators` and did not would train the wrong model and say nothing.
    """
    if after.n_estimators != before.n_estimators:
        raise Error(
            "n_estimators is not resettable during training; the round count"
            " is fixed when training starts"
        )
    if after.tree.feature_fraction_seed != before.tree.feature_fraction_seed:
        raise Error("feature_fraction_seed is not resettable during training")

    # The same data-independent ranges `params._validate` applies when a
    # parameter string is parsed, from the one place that holds them.
    check_booster_ranges(
        after.n_estimators,
        after.learning_rate,
        after.tree.num_leaves,
        after.tree.max_depth,
        after.tree.min_data_in_leaf,
        after.tree.min_child_hess,
        after.tree.lambda_l1,
        after.tree.lambda_reg,
        after.tree.feature_fraction,
        after.tree.feature_fraction_bynode,
        after.tree.feature_fraction_bylevel,
    )


def scale_tree_values(mut tree: Tree, factor: Float64):
    """Multiply every node value by `factor`, baking a round's shrinkage into
    the tree so prediction does not have to apply it.

    Internal nodes carry values too and are scaled with the leaves: they are
    unused at predict time, and leaving them at a different scale would make
    the tree inconsistent with itself for anything that reads them.
    """
    for i in range(len(tree.value)):
        tree.value[i] *= factor
