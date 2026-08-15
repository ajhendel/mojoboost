"""Training callbacks for the Python API.

    from mojotrees import MojoTreesRegressor
    from mojotrees.callback import log_evaluation, record_evaluation

    history = {}
    model = MojoTreesRegressor(n_estimators=200).fit(
        X, y,
        eval_set=[(X_valid, y_valid)],
        eval_metric=("rmse", rmse),
        callbacks=[log_evaluation(period=10), record_evaluation(history)],
    )

A callback is any callable taking one `CallbackEnv`. The four factories here
build the ones LightGBM ships, with LightGBM's names, signatures, and
scheduling attributes, so a callback list written for LightGBM runs here.

The environment
---------------
`CallbackEnv` is LightGBM's namedtuple, with the same fields:

    model                   the run's handle (see `TrainingHandle`)
    params                  this round's hyperparameters, as a dict
    iteration               the 0-based round
    begin_iteration         the first round of the run, always 0 today
    end_iteration           one past the last round the run may reach
    evaluation_result_list  [(data_name, metric_name, value, higher_better)]

`evaluation_result_list` is empty in the before-iteration phase, because
nothing has been scored yet, and `params` is only writable there, through
`env.model.reset_parameter(...)`.

Ordering
--------
Callbacks run in two groups, as in LightGBM. Those with a truthy
`before_iteration` attribute run before the round's tree is grown; the rest
run after it is grown and the metrics are scored. Within each group they run
in ascending `order` (default 0), and ties keep the order you listed them in.
The factories use LightGBM's numbers: `reset_parameter` and `log_evaluation`
are 10, `record_evaluation` is 20, `early_stopping` is 30.

Cost
----
The training loop is Mojo; a callback costs one crossing of the Python
boundary per phase per round, and nothing per row. Metric values arrive as
one buffer read rather than as per-row objects. `bench/bench_callbacks.py`
measures the per-iteration cost; with no callbacks the bridge does not cross
the boundary at all, so `callbacks=None` runs exactly the loop that ran
before callbacks existed.

Differences from LightGBM
-------------------------
- `early_stopping()` configures the trainer's own early stopping rather than
  reimplementing the rule in Python. The rule is the same one LightGBM's
  callback applies (strict improvement, `min_delta`, per validation-set and
  per-metric patience), and it is described in
  src/mojotrees/custom_metric.mojo. The one behavioral difference is which
  round survives: mojotrees truncates to the best round of the primary
  metric on the first validation set, LightGBM to the best round of the pair
  that ran out of patience first.
- A callback that raises `EarlyStopException` stops the run and rolls the
  ensemble back to that same best round, which is what LightGBM does with
  the exception too.
- Any other exception from a callback propagates unchanged, with its own
  type and traceback, and the estimator is left unfitted. The exception
  object is carried across the Mojo boundary rather than re-raised from its
  message, so `except MyError` still catches it.
- `reset_parameter` accepts only the hyperparameters the training loop
  re-reads each round (listed in `RESETTABLE`). LightGBM accepts any key and
  ignores the ones its loop cannot honor; here an unusable key raises,
  because a schedule that believes it is doing something and is not is worse
  than a failed run.
"""

from collections import namedtuple

__all__ = [
    "CallbackEnv",
    "EarlyStopException",
    "RESETTABLE",
    "TrainingHandle",
    "early_stopping",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
]

CallbackEnv = namedtuple(
    "CallbackEnv",
    [
        "model",
        "params",
        "iteration",
        "begin_iteration",
        "end_iteration",
        "evaluation_result_list",
    ],
)

#: Hyperparameters a before-iteration callback may change, in the slot order
#: the Mojo bridge reads them in. Keep in sync with `RESET_SLOTS` and
#: `_write_reset`/`_read_reset` in bindings/_mojotrees.mojo: the two sides
#: index the same buffer, so a change to one alone reassigns parameters
#: silently.
RESETTABLE = (
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_data_in_leaf",
    "min_sum_hessian_in_leaf",
    "lambda_l1",
    "lambda_l2",
    "feature_fraction",
    "feature_fraction_bynode",
)

#: Slots that must round trip as whole numbers; the buffer is float64.
_INTEGRAL = frozenset(
    {"num_leaves", "max_depth", "min_data_in_leaf"}
)

#: LightGBM's aliases for the resettable parameters, so a schedule written
#: against the scikit-learn spellings works. The estimator resolves the same
#: pairs; see `_Base._resolve_alias`.
_RESET_ALIASES = {
    "min_child_samples": "min_data_in_leaf",
    "min_child_weight": "min_sum_hessian_in_leaf",
    "min_sum_hessian": "min_sum_hessian_in_leaf",
    # mojotrees's own spelling of the same parameter on the estimator.
    "min_child_hess": "min_sum_hessian_in_leaf",
    "reg_alpha": "lambda_l1",
    "reg_lambda": "lambda_l2",
    "colsample_bytree": "feature_fraction",
    "colsample_bynode": "feature_fraction_bynode",
    "sub_feature": "feature_fraction",
    "shrinkage_rate": "learning_rate",
    "eta": "learning_rate",
}

# Phases and control codes; these mirror src/mojotrees/callback.mojo.
BEFORE_ITERATION = 0
AFTER_ITERATION = 1

_CONTINUE = 0
_STOP = 1
_ABORT = 2


class EarlyStopException(Exception):
    """Raised by a callback to stop training.

    The ensemble is rolled back to the best round of the primary metric, as
    LightGBM does. `best_iteration` and `best_score` are recorded for the
    caller but do not override what the trainer measured.
    """

    def __init__(self, best_iteration=None, best_score=None):
        super().__init__(
            f"early stopping requested at iteration {best_iteration}"
        )
        self.best_iteration = best_iteration
        self.best_score = best_score


def canonical_reset_key(key):
    """The primary name of a resettable parameter, or a `ValueError` naming
    the set. Aliases resolve the way the estimator resolves them."""
    name = _RESET_ALIASES.get(key, key)
    if name not in RESETTABLE:
        raise ValueError(
            f"{key!r} cannot be reset during training; the trainer re-reads "
            "only " + ", ".join(RESETTABLE)
        )
    return name


class TrainingHandle:
    """The `model` field of a `CallbackEnv`: what a callback may ask of the
    run in progress.

    This is not the fitted estimator, which does not exist yet. It exposes
    the run's identity and the one mutation the loop supports.
    """

    def __init__(self, estimator, params):
        self.estimator = estimator
        self._params = params
        self._pending = {}
        self._phase = BEFORE_ITERATION
        self.current_iteration = 0
        self.best_iteration = None
        self.best_score = None

    def reset_parameter(self, new_parameters):
        """Change the hyperparameters of the round about to be grown.

        Valid in the before-iteration phase only; afterwards the round is
        already grown and a reset would silently apply to the next one.
        """
        if self._phase != BEFORE_ITERATION:
            raise RuntimeError(
                "parameters can only be reset before an iteration; give the "
                "callback a truthy before_iteration attribute"
            )
        for key, value in new_parameters.items():
            name = canonical_reset_key(key)
            value = int(value) if name in _INTEGRAL else float(value)
            self._pending[name] = value
            # Visible to later callbacks in the same phase straight away,
            # which is what LightGBM's reset_parameter arranges by updating
            # env.params itself. A callback ordered after a schedule should
            # see what the round will actually train with.
            self._params[name] = value

    def __repr__(self):
        return (
            f"TrainingHandle(iteration={self.current_iteration}, "
            f"estimator={type(self.estimator).__name__})"
        )


class CallbackRunner:
    """Runs a callback list against the Mojo training loop.

    One instance serves one `fit`. `__call__` is the function the binding
    invokes once per phase per round; it returns a control code rather than
    letting an exception cross the boundary, and keeps the exception so
    `fit` can re-raise the original object.
    """

    def __init__(
        self,
        callbacks,
        estimator,
        params,
        valid_names,
        metric_names,
        metric_higher,
        reset_buffer,
        evals_buffer,
    ):
        ordered = list(callbacks)
        self.before = _sorted_phase(ordered, before=True)
        self.after = _sorted_phase(ordered, before=False)
        self.params = dict(params)
        self.handle = TrainingHandle(estimator, self.params)
        self.valid_names = list(valid_names)
        self.metric_names = list(metric_names)
        self.metric_higher = list(metric_higher)
        self._reset = reset_buffer
        self._evals = evals_buffer
        self.error = None
        self.stop_exception = None
        # Set by `fit` once it knows the round count; the environment
        # reports it as `end_iteration`.
        self.end_iteration = 0
        # Crossing counters, so the benchmark can assert the "one crossing
        # per phase per round" claim rather than infer it from a timing.
        self.n_before_calls = 0
        self.n_after_calls = 0

    def __call__(self, phase, iteration):
        try:
            if phase == BEFORE_ITERATION:
                self.n_before_calls += 1
                return self._before(iteration)
            self.n_after_calls += 1
            return self._after(iteration)
        except EarlyStopException as exc:
            self.stop_exception = exc
            return _STOP
        except BaseException as exc:  # noqa: BLE001 - re-raised by fit
            self.error = exc
            return _ABORT

    def _before(self, iteration):
        self.handle._phase = BEFORE_ITERATION
        self.handle.current_iteration = iteration
        self.handle._pending.clear()
        # The buffer is what the loop is about to train with, so it, not our
        # last copy, is the truth about the current parameters.
        for slot, name in enumerate(RESETTABLE):
            value = self._reset[slot]
            self.params[name] = int(value) if name in _INTEGRAL else value
        env = CallbackEnv(
            model=self.handle,
            params=self.params,
            iteration=iteration,
            begin_iteration=0,
            end_iteration=self.end_iteration,
            evaluation_result_list=[],
        )
        for cb in self.before:
            cb(env)
        for name, value in self.handle._pending.items():
            slot = RESETTABLE.index(name)
            self._reset[slot] = float(value)
            self.params[name] = value
        return _CONTINUE

    def _after(self, iteration):
        self.handle._phase = AFTER_ITERATION
        self.handle.current_iteration = iteration
        env = CallbackEnv(
            model=self.handle,
            params=self.params,
            iteration=iteration,
            begin_iteration=0,
            end_iteration=self.end_iteration,
            evaluation_result_list=self._results(),
        )
        for cb in self.after:
            cb(env)
        return _CONTINUE

    def _results(self):
        """This round's values as LightGBM's
        `(data_name, metric_name, value, is_higher_better)` tuples."""
        out = []
        n_metrics = len(self.metric_names)
        for v, data_name in enumerate(self.valid_names):
            for m, metric_name in enumerate(self.metric_names):
                out.append(
                    (
                        data_name,
                        metric_name,
                        self._evals[v * n_metrics + m],
                        self.metric_higher[m],
                    )
                )
        return out


def _sorted_phase(callbacks, before):
    """One phase's callbacks in run order: ascending `order`, ties in the
    order the caller listed them. `sorted` is stable, so listing order is
    what breaks a tie."""
    selected = [
        cb
        for cb in callbacks
        if bool(getattr(cb, "before_iteration", False)) is before
    ]
    return sorted(selected, key=lambda cb: getattr(cb, "order", 0))


def log_evaluation(period=1, show_stdv=True):
    """Print the evaluation results every `period` rounds.

    `period=0` (or negative) prints nothing, which is how you silence a run
    without dropping the other callbacks. `show_stdv` is accepted for
    LightGBM compatibility and has no effect: it formats the standard
    deviation of a cross-validation fold, and there is no `cv` here yet.
    """
    period = int(period)

    def _callback(env):
        if period <= 0 or not env.evaluation_result_list:
            return
        if (env.iteration + 1) % period != 0:
            return
        rendered = "\t".join(
            f"{data}'s {name}: {value:g}"
            for data, name, value, _ in env.evaluation_result_list
        )
        print(f"[{env.iteration + 1}]\t{rendered}")

    _callback.order = 10
    return _callback


def record_evaluation(eval_result):
    """Append every round's results into `eval_result`, in place.

    The layout is `evals_result_`'s:
    `{valid_name: {metric_name: [round 1, round 2, ...]}}`. Unlike
    `evals_result_`, which the estimator fills from the trainer's own
    history and which starts at the base-score-only model, this starts at
    the first round, as LightGBM's does.
    """
    if not isinstance(eval_result, dict):
        raise TypeError("record_evaluation expects a dict to fill")
    eval_result.clear()

    def _callback(env):
        for data, name, value, _ in env.evaluation_result_list:
            eval_result.setdefault(data, {}).setdefault(name, []).append(
                value
            )

    _callback.order = 20
    return _callback


def reset_parameter(**kwargs):
    """Change hyperparameters as training runs.

    Each value is either a list with one entry per round, or a callable
    taking the 0-based round and returning the value:

        reset_parameter(learning_rate=lambda i: 0.1 * (0.99 ** i))
        reset_parameter(num_leaves=[15] * 50 + [31] * 50)

    Only the parameters in `RESETTABLE` can be scheduled; anything else
    raises rather than being ignored. See src/mojotrees/callback.mojo for why
    the set is what it is, and for how a learning-rate schedule reaches
    prediction.
    """
    if not kwargs:
        raise ValueError("reset_parameter needs at least one parameter")
    schedule = {canonical_reset_key(k): v for k, v in kwargs.items()}

    def _callback(env):
        updates = {}
        for key, value in schedule.items():
            round_index = env.iteration - env.begin_iteration
            if callable(value):
                new_value = value(round_index)
            else:
                if round_index >= len(value):
                    raise ValueError(
                        f"the {key} schedule has {len(value)} entries, too "
                        f"few for round {round_index}; it needs one per "
                        f"round up to end_iteration={env.end_iteration}"
                    )
                new_value = value[round_index]
            if key in _INTEGRAL:
                new_value = int(new_value)
            else:
                new_value = float(new_value)
            if new_value != env.params.get(key):
                updates[key] = new_value
        if updates:
            env.model.reset_parameter(updates)

    _callback.before_iteration = True
    _callback.order = 10
    return _callback


class _EarlyStopping:
    """`early_stopping()`'s return value.

    It is a callback, and it is also the carrier of `stopping_rounds` and
    `min_delta`: `fit` reads them off the object and hands them to the
    trainer, whose own stopper applies the rule. The `__call__` side only
    reports.
    """

    order = 30
    before_iteration = False

    def __init__(self, stopping_rounds, first_metric_only, verbose, min_delta):
        stopping_rounds = int(stopping_rounds)
        if stopping_rounds < 1:
            raise ValueError("stopping_rounds must be positive")
        min_delta = float(min_delta)
        if min_delta < 0.0:
            raise ValueError("min_delta must not be negative")
        self.stopping_rounds = stopping_rounds
        self.first_metric_only = bool(first_metric_only)
        self.verbose = bool(verbose)
        self.min_delta = min_delta

    def __call__(self, env):
        return

    def report(self, best_iteration, best_score, metric_name, data_name):
        """Print LightGBM's stop line. Called by `fit` once the trainer has
        stopped, because the trainer, not this object, decides the round."""
        if not self.verbose:
            return
        print(
            f"Early stopping, best iteration is:\n[{best_iteration}]\t"
            f"{data_name}'s {metric_name}: {best_score:g}"
        )

    def __repr__(self):
        return (
            f"early_stopping(stopping_rounds={self.stopping_rounds}, "
            f"first_metric_only={self.first_metric_only}, "
            f"verbose={self.verbose}, min_delta={self.min_delta})"
        )


def resolve_early_stopping(callbacks, early_stopping_rounds, min_delta):
    """Fold an `early_stopping()` callback into the trainer's own settings.

    Returns `(rounds, min_delta, first_metric_only, stopper)`. `stopper` is
    the callback object, or None, so `fit` can ask it to report once the
    trainer has decided which round won.

    Passing both the callback and `early_stopping_rounds=` is an error
    rather than a precedence rule: they are two spellings of one knob, and
    silently preferring one would make the other look broken.
    """
    stoppers = [cb for cb in callbacks if isinstance(cb, _EarlyStopping)]
    if len(stoppers) > 1:
        raise ValueError("pass at most one early_stopping() callback")
    if not stoppers:
        return early_stopping_rounds, min_delta, False, None
    stopper = stoppers[0]
    if early_stopping_rounds:
        raise ValueError(
            "pass either the early_stopping() callback or "
            "early_stopping_rounds=, not both; they set the same knob"
        )
    if float(min_delta) != 0.0:
        raise ValueError(
            "min_delta= belongs to early_stopping_rounds=; give it to the "
            "early_stopping() callback instead"
        )
    return (
        stopper.stopping_rounds,
        stopper.min_delta,
        stopper.first_metric_only,
        stopper,
    )


def early_stopping(
    stopping_rounds, first_metric_only=False, verbose=True, min_delta=0.0
):
    """Stop once a watched metric goes `stopping_rounds` rounds without
    improving by more than `min_delta`, and keep the best round.

    `first_metric_only` watches only the first entry of `eval_metric`;
    otherwise every metric that did not opt out is watched, on every
    validation set. Passing this callback and `early_stopping_rounds=` to the
    same `fit` is an error: they set the same knob.
    """
    return _EarlyStopping(
        stopping_rounds, first_metric_only, verbose, min_delta
    )
