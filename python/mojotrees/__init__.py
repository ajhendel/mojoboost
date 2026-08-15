"""scikit-learn style Python API for mojotrees.

Build the extension first: `bindings/build.sh` (produces `_mojotrees.so`
next to this file). numpy is used when available; plain Python sequences
(lists of rows) work without it.

    from mojotrees import MojoTreesRegressor
    model = MojoTreesRegressor().fit(X, y)
    pred = model.predict(X)

LightGBM's functional API is the other door, in `mojotrees.basic`:
`Dataset` owns the training data and its binning, `train()` fits a
`Booster`, and the estimators hold that same `Booster` on `booster_`, so
there is one model object here rather than one per API.

    train_set = mojotrees.Dataset(X, label=y)
    booster = mojotrees.train({"objective": "regression"}, train_set, 100)

scikit-learn conventions
------------------------
The estimators implement `get_params`, `set_params`, `fit`, `predict`,
`predict_proba` (classifier), and `score`, record `n_features_in_`,
`feature_names_in_`, `classes_`, `n_classes_`, `feature_importances_`,
`device_`, `best_iteration_`, and `n_iter_` when fitted, raise
`NotFittedError` before that, and pickle. `clone`, `Pipeline`,
`GridSearchCV`, and `cross_val_score` work.

LightGBM's own fitted attributes are here too, and each is answered by the
model rather than kept in step with it: `booster_` is the `Booster`,
`objective_` the resolved objective's canonical name, `feature_name_` the
training feature names (`Column_0`, `Column_1`, ... when the model carries
none), and `n_features_` the feature count read off the model. With an
`eval_set`, `evals_result_`, `best_score_`, and `stopped_early_` record
what validation saw; without one they are absent, because there was no
metric to report.
scikit-learn is optional: nothing here imports it except the
`__sklearn_tags__` hook that scikit-learn itself calls.

`check_estimator`'s full suite has not been run, so this is "scikit-learn
style" and not a compliance claim. Two known deviations:

- subclasses forward the shared hyperparameters through `**kwargs`, so
  `get_params()` lists them but `inspect.signature` does not
- `best_iteration_` is always set (see below), where LightGBM sets it only
  when early stopping ran

Validation follows LightGBM's scikit-learn wrapper, which validates with
`force_all_finite="allow-nan"`: `X` may hold NaN, mojotrees's missing-value
marker, but not infinities, and `y` and `sample_weight` must be finite.

SciPy sparse input
------------------
`fit` and `predict` accept any SciPy sparse matrix or array and keep it
sparse: nothing is densified, at any point. Whatever format you pass is
converted to the one that side of the boundary wants (CSC to fit, because
histogram accumulation is feature-oriented; CSR to predict, because
prediction is row-oriented), and a non-canonical matrix is copied before its
indices are sorted, so your matrix is never mutated.

An implicit zero is the numerical value 0.0, not a missing value, which
matches LightGBM's default `zero_as_missing=false`: a sparse fit equals the
dense fit of the same matrix with the gaps filled with zeros. Explicitly
stored zeros mean the same thing as the gaps. NaN is still the missing
marker, wherever it is stored.

Sparse input trains on the accelerator too, on an explicit `device="gpu"`
(the sparse GPU trainer grows on the compressed matrix; `device="auto"`
keeps the CPU because that path's crossover is unmeasured). Not available
for sparse input: a Python objective callback, `eval_set` and early
stopping, ranking, and GPU prediction. Each raises rather than densifying
behind your back.

`best_iteration_` is the boosting iteration the model is used at, and
`n_iter_` the number that were trained. They differ only when a validation
metric peaked before the last round with early stopping off; otherwise both
are `n_estimators` unless training stopped early, either because the
objective converged or because a validation metric did.

validation sets and early stopping
----------------------------------
Every estimator takes them, in LightGBM's spelling::

    model.fit(
        X, y,
        eval_set=[(X_valid, y_valid)],   # or (X_valid, y_valid),
                                         # or eval_X=/eval_y=
        eval_names=["holdout"],          # valid_0, valid_1, ... by default
        eval_sample_weight=[w_valid],
        eval_metric=["l2", "l1"],
        early_stopping_rounds=10,
    )

`eval_metric` takes LightGBM's metric names (`l2`, `rmse`, `l1`,
`quantile`, `huber`, `binary_logloss`, `binary_error`, `auc`,
`multi_logloss`, `multi_error`, `ndcg`, and their aliases), callables, or
both, and defaults to the objective's own loss. A built-in name is computed
by src/mojotrees/metrics.mojo, so it agrees with the Mojo API by
construction, and `eval_sample_weight` weights it. A callable is
`f(y_true, y_pred) -> float`, called once per metric per set per round with
raw scores in `y_pred` (log-odds for the binary classifier, one row-major
block of `n_classes_` per row for the softmax one), which is what
LightGBM's `feval` also receives; give it a direction with
`("name", f, True)` for higher-is-better. It is handed unweighted
predictions, so combining one with `eval_sample_weight` raises rather than
quietly dropping the weights.

`early_stopping_rounds` stops training once a watched metric has gone that
many rounds without improving by more than `min_delta`, and rolls the
ensemble back to the best round of `primary_metric` (an index or a name) on
the first validation set. It needs an `eval_set` and says so otherwise.
With `early_stopping_rounds=0` nothing stops and nothing is rolled back,
but every value is still recorded and `best_iteration_` still reports where
the metric peaked.

`evals_result_` holds those values,
`{valid_name: {metric_name: [round 0, round 1, ...]}}`, where index 0 is
the base-score-only model, so `evals_result_[name][metric][i]` is the score
after `i` trees. `best_score_` is the primary metric's best value, and
`stopped_early_` says whether patience ran out. (LightGBM's `best_score_`
is a dict of every set's every metric; here it is the one number
`best_iteration_` was chosen by, and the rest of the grid is in
`evals_result_`.)

The multiclass classifier and the ranker take all of this too. A
classifier's validation labels go through the encoding `classes_` records,
so a label absent from training is an error rather than a silent miscount,
and a ranker's `eval_group` carries each validation set's own query
boundaries. Validation is scored on the CPU, so `device="gpu"` with an
`eval_set` raises rather than falling back. A Python objective callback
cannot be combined with validation yet (the Mojo API pairs them with
`train_custom_with_metrics`). The metric contract, and where it departs
from LightGBM, is in src/mojotrees/custom_metric.mojo.

prediction options
------------------
`predict` (and the classifier's `predict_proba`) take LightGBM's prediction
keywords: `raw_score`, `start_iteration`, `num_iteration`, `pred_leaf`,
`pred_contrib`, and `validate_features`.

`raw_score` returns scores before the objective's inverse link. It changes
nothing for the regressor's objectives or for the ranker, which have no
link; for the binary classifier it returns log-odds of shape
`(n_samples,)`, one score per row rather than one per class, and for the
multiclass classifier the pre-softmax scores of shape
`(n_samples, n_classes)`. Those are LightGBM's shapes.

`start_iteration` and `num_iteration` select a half-open slice of the
boosting iterations, with LightGBM's clamping: a negative start becomes 0, a
start past the end selects nothing, and `num_iteration=None` or any value
<= 0 means every iteration from the start on. The base score belongs to
iteration 0, so a slice starting there carries it and a later slice does
not, which is what makes `predict(num_iteration=k)` equal to the prediction
of a `k`-round fit and makes `[0, k)` and `[k, n)` sum to the whole raw
score. Out-of-range slices clamp rather than raise, because that is what
LightGBM callers slicing a shorter-than-expected ensemble depend on; a
`start_iteration` or `num_iteration` that is not an integer is a `TypeError`.

`num_iteration=None` predicts with `best_iteration_` iterations, LightGBM's
documented default. mojotrees reaches that structurally rather than by
bookkeeping: early stopping truncates the ensemble at its best iteration, so
the trees the model still holds are the best iteration and there is no later
tree for prediction to exclude.

`pred_leaf` returns the leaf each row reaches in each tree, as integers.
The shape is `(n_samples, num_iteration)` for the regressor, the ranker, and
the binary classifier, which are single-output ensembles, and
`(n_samples, num_iteration * n_classes)` for the multiclass classifier,
whose column `i * n_classes + k` is class k's tree in iteration i. A leaf is
named by its ordinal within its own tree, in `[0, num_leaves)`, numbered in
node order. That numbering is fixed once a tree is grown and survives
`save`/`load` and pickling, but it is mojotrees's own: it is not LightGBM's
leaf id and the two agree only by coincidence.

`pred_contrib` returns exact per-feature contributions: one column per
feature plus an expected-value column, so the shape is
`(n_samples, n_features + 1)` for the regressor, the ranker, and the binary
classifier, and `(n_samples, n_classes * (n_features + 1))` in class-major
blocks for the multiclass classifier. Those are LightGBM's shapes. Every
row's entries sum to that row's raw score, exactly: these are TreeSHAP
Shapley values, not a split-gain heuristic, and the sum is a mathematical
identity rather than a normalization. Contributions always explain the raw
score whatever the objective's link, so `raw_score=True` cannot be combined
with them; see src/mojotrees/contrib.mojo for the algorithm and what it
conditions on. A model loaded from a file written before mojotrees recorded
node covers (format v1 or v2) raises here rather than guessing at them.

`validate_features` turns the feature-name checks from warnings into errors:
without it, predicting from a matrix that has names against a model fitted
without them (or the reverse) warns, as scikit-learn does. Names that are
present on both sides and disagree raise either way.

`raw_score=True` and `pred_leaf=True` together raise: they ask for different
dtypes and different shapes. LightGBM lets `pred_leaf` win silently, which
is not recoverable from the output.

`fit(..., eval_set=[(X_valid, y_valid)], eval_metric=my_metric)` scores
validation sets with caller-supplied metrics, one call per metric per set
per round, and `early_stopping_rounds` stops on them. A metric is
`f(y_true, y_pred) -> float`, where `y_pred` holds raw scores (log-odds for
the binary classifier) as LightGBM's `feval` also receives; give it a
direction with `("name", f, True)` for higher-is-better, and pick which
metric truncates the ensemble with `primary_metric`. `evals_result_` holds
every value, `{valid_name: {metric_name: [round 0, round 1, ...]}}`, where
index 0 is the base-score-only model, and `best_score_` the primary
metric's best value. The metric contract, and where it departs from
LightGBM, is in src/mojotrees/custom_metric.mojo. There are no built-in
validation metrics in the Python API yet, and a Python objective callback
cannot be combined with them yet (the Mojo API pairs them with
`train_custom_with_metrics`).

`fit(..., callbacks=[...])` runs callbacks around every boosting round, with
LightGBM's contract: a callable taking one `CallbackEnv`, split into
before-iteration and after-iteration groups by a `before_iteration`
attribute and run in ascending `order` within each. The four factories
LightGBM ships are in `mojotrees.callback` and re-exported here:
`early_stopping`, `log_evaluation`, `record_evaluation`, and
`reset_parameter`. Callbacks need an `eval_set`, because the hook lives in
the trainer that scores validation metrics; they run for the regressor and
the binary classifier, and the softmax and LambdaRank trainers refuse a
callback list rather than ignoring it. A callback costs one crossing of the
Python boundary per phase per round and nothing per row
(bench/bench_callbacks.py); with no callbacks the bridge does not cross the
boundary at all, and the model is unchanged to the bit. An exception from a
callback propagates with its own type and leaves the estimator unfitted.
See python/mojotrees/callback.py for the environment and the differences
from LightGBM, and src/mojotrees/callback.mojo for the loop contract.

Estimators take `device="cpu"` (the default and the dependable backend),
`device="gpu"`, or `device="auto"`. `"gpu"` raises when no accelerator is
available or when the GPU path does not cover the workload, rather than
falling back silently; `"auto"` picks a backend for you and currently
always picks the CPU. `gpu_available()` reports whether this build can
train on an accelerator. Fitting records the backend that ran on
`device_`. The decision is made in Mojo and nowhere else, and
`explain_device_choice(X, y, device="gpu")` returns the report behind it
without training anything. See src/mojotrees/device.mojo for the policy and
`mojotrees.device_selection` for the Python door to it.

`predict`, `predict_proba`, and the ranker's `predict` take their own
`device=` as well, because where a prediction runs is a property of the
call and not of the fit that produced the model: mojotrees stores one
ensemble, so a model fitted on the CPU can be predicted on an accelerator
and the reverse. `device=None`, the default, is the CPU. Scores,
probabilities, and leaf ordinals (`pred_leaf`) all have a device path;
contributions (`pred_contrib`) and sparse input do not, and refuse an
explicit `"gpu"` rather than running on the CPU and reporting otherwise.
`"auto"` is never refused, because it resolves to the CPU there anyway.
Which backend runs is decided in Mojo, not here: the entry point asks
whether the GPU predictor covers the request before it resolves, so an
explicit `"gpu"` that cannot run says why. On a build without the device
prediction entry points, every `device=` other than `"cpu"` raises for the
same reason.

`show_versions()` prints what this installation is, for a bug report, and
`build_info()` returns the same facts as a dict. Both report whether a GPU
path was compiled into the build, which is decided on the machine that
compiled the extension rather than the one that runs it, and which
`gpu_available()` alone cannot distinguish from a GPU build that
`MOJOTREES_DISABLE_GPU=1` is masking.

`MojoTreesRegressor(objective=f)` accepts a Python objective callback,
`f(raw, y) -> (grad, hess)`, called once per boosting round rather than per
row. It costs one Python call and two array passes per round; measure it
with bench/bench_custom_objective.py before reaching for it, and use the
native Mojo interface (src/mojotrees/objective.mojo) when the objective sits
on a hot path.

`monotone_constraints` takes LightGBM's parameter as one entry per feature,
`1` for nondecreasing, `-1` for nonincreasing, `0` for unconstrained, e.g.
`[1, 0, -1]`. Predictions are then monotone in those features at any feature
value, not just on the training data. For a multiclass classifier the
guarantee is on each class's raw score, not on `predict_proba`. See
src/mojotrees/monotone.mojo for the method and its limits.

`interaction_constraints` takes LightGBM's parameter as a list of
feature-index lists, e.g. `[[0, 1], [2, 3]]`. Two features may share a
root-to-leaf path only if one group holds both, and transitively everything
else on that path. Groups may overlap. Note the LightGBM behavior this
matches: a feature in no group is never split on at all, so constraining a
few features drops the rest from the model. To leave a feature free to
interact with everything, put it in every group; a group of its own instead
isolates it. See src/mojotrees/interaction.mojo for the exact rule.

The rest of the package
-----------------------
`import mojotrees` costs the compiled extension and nothing else. Nothing
here imports numpy, pandas, scipy, scikit-learn, or dask, and none of them
has to be installed. Everything below the estimators is reached on first
access, so asking for it is what pays for it:

- `mojotrees.cv(params, train_set, ...)` cross-validates, as `lightgbm.cv`
  does, and `CVBooster` is what `return_cvbooster=True` hands back. Note
  that `mojotrees.cv` is the *function*: it shadows the module of the same
  name, so reach the module as `from mojotrees.cv import ...`.
- `mojotrees.inspection` is structured model inspection, the schema in
  docs/MODEL_INSPECTION_SCHEMA.md. `dump_model`, `trees_to_dataframe`,
  `trees_to_records`, and `get_split_value_histogram` are re-exported here
  and are also methods on `Booster`. pandas is needed for the frame form
  and for nothing else.
- `mojotrees.device_selection` explains what a device request would do.
  `explain_device_choice` is re-exported here.
- `mojotrees.diagnostics` describes the install and the startup phases,
  and is what `build_info()` and `show_versions()` read.
- `mojotrees.dask` is **experimental and cannot train**. No transport
  ships, so every `fit` raises `DistributedNotAvailable`; the module is
  the client-side contract for a backend, not a feature, which is why
  nothing from it is exported here. Importing it does not import dask.
"""

# The interpreter check runs before the extension is named, not around it.
# On an interpreter older than `_compat.EXTENSION_FLOOR` the Mojo runtime
# resolves CPython entry points out of libpython at load time and ends the
# process on the first one it cannot find:
#
#     ABORT: ... symbol not found: Py_NewRef
#
# An abort is not an exception, so a `try` around the import below would
# never run its handler. `_compat.import_extension` raises ImportError first
# and only then touches the extension; see _compat.py and section 10 of
# docs/PYTHON_SUPPORT.md for the measurement. Binding the result here is what
# makes `from . import _mojotrees` elsewhere in the package cheap and safe:
# by the time any submodule runs, this has already either bound the module in
# `sys.modules` or raised.
from . import _compat

_mojotrees = _compat.import_extension()

from . import _arrays, _eval, callback as _callback
from ._sklearn import NotFittedError, ParamsMixin as _ParamsMixin
from ._sklearn import estimator_tags as _estimator_tags

# The functional API. `basic` reaches back into this module for the
# estimators that hold the parameter definitions, but only from inside its
# functions, so importing it here and it importing this is not a cycle.
from .basic import Booster, Dataset, train

_np = _arrays.np

# Keep in sync with python/pyproject.toml.
__version__ = "0.1.0"

__all__ = [
    # The model, and the two doors to it.
    "Booster",
    "Dataset",
    "train",
    "cv",
    "CVBooster",
    "MojoTreesRegressor",
    "MojoTreesClassifier",
    "MojoTreesRanker",
    "NotFittedError",
    # What this installation can say about itself.
    "build_info",
    "gpu_available",
    "show_versions",
    "explain_device_choice",
    # Ranking helpers.
    "group_from_query_ids",
    "ndcg_score",
    # Callbacks, re-exported at the top level the way LightGBM exports
    # them, and also importable from mojotrees.callback.
    "CallbackEnv",
    "EarlyStopException",
    "callback",
    "early_stopping",
    "log_evaluation",
    "record_evaluation",
    "reset_parameter",
    # Structured model inspection. The whole schema is in
    # mojotrees.inspection; these are the four entry points LightGBM has a
    # spelling for, and each resolves the submodule on first access rather
    # than at import.
    "dump_model",
    "trees_to_dataframe",
    "trees_to_records",
    "get_split_value_histogram",
]

from .callback import (  # noqa: E402 - after __all__ for readability
    CallbackEnv,
    EarlyStopException,
    early_stopping,
    log_evaluation,
    record_evaluation,
    reset_parameter,
)

# Buffer plumbing and input validation live in _arrays; these names are the
# spellings the estimator code below has always used.
_as_column_major = _arrays.column_major
_as_f64_vector = _arrays.f64_vector
_addr = _arrays.addr
_out_buffer = _arrays.out_buffer
_finish = _arrays.finish

# Shared vocabularies and fit/predict argument helpers live in
# _fit_args.py; the names stay bound here so the package namespace is
# unchanged.
from ._fit_args import (  # noqa: E402,F401
    _IMPORTANCE_TYPES,
    _DEVICES,
    _BOOSTING_TYPES,
    _NO_DEVICE_PREDICT,
    _OTHER_ESTIMATOR_OBJECTIVES,
    _unimplemented_objectives,
    _objective_status,
    _unimplemented_objective_note,
    _GROW_POLICIES,
    _MAX_RELEVANCE_LABEL,
    _as_iteration,
    _store_vector,
    _metric_spec,
    _metric_specs,
    _primary_index,
    _eval_pairs,
    _per_set,
    _encode_like,
    _early_stopping_rounds,
    _check_eval_arguments,
    _device_name,
)


# What this installation can say about itself lives in _environment.py;
# the names stay bound here so the package namespace is unchanged.
from ._environment import (  # noqa: E402,F401
    gpu_available,
    _OPTIONAL_DEPENDENCIES,
    _distribution_version,
    _optional_dependency_versions,
    _install_layout,
    _BUILD_INFO_FILE,
    _build_provenance,
    build_info,
    show_versions,
)


# Query-group helpers live in _ranking.py; the names stay bound here so
# the package namespace is unchanged.
from ._ranking import (  # noqa: E402,F401
    group_from_query_ids,
    _group_buffer,
    _check_relevance,
    ndcg_score,
)


# The estimators, their base, and the objective-code and regularization
# literals live in sklearn.py (as lightgbm.sklearn holds LightGBM's); the
# names stay bound here so the package namespace is unchanged.
from .sklearn import (  # noqa: E402,F401
    _SQUARED_ERROR,
    _BINARY_LOGISTIC,
    _POISSON,
    _HUBER,
    _QUANTILE,
    _L1,
    _CUSTOM,
    _LAMBDARANK,
    _GAMMA,
    _TWEEDIE,
    _MAPE,
    _FAIR,
    _CROSS_ENTROPY,
    _LAMBDA_L1,
    _LAMBDA_L2,
    _Base,
    MojoTreesRegressor,
    MojoTreesClassifier,
    MojoTreesRanker,
)


# =========================================================================
# The public surface, assembled last
# =========================================================================
#
# Everything below runs after the estimators exist, which is the constraint
# that decides eager from lazy: `mojotrees.cv` and `mojotrees.dask` both
# reach back into this module, so neither can be imported from the top of
# it. `cv` is cheap and unconditional, so it is imported here; the rest are
# resolved on first access by `__getattr__`, so that `import mojotrees`
# costs the extension and nothing else.
#
# `python/mojotrees/_public_api_plan.py` is the same decisions as data,
# with the alternatives that were not taken.

# `mojotrees.cv` is the *function*, as `lightgbm.cv` is, and the module of
# that name is shadowed by it. `from mojotrees.cv import cv, CVBooster`
# still works, because that form resolves the submodule through sys.modules
# rather than through this attribute; `import mojotrees.cv as m` binds the
# function, so use `from mojotrees import cv` instead. The collision is
# written down in `_public_api_plan.NAME_COLLISIONS` with the rename that
# would remove it.
from .cv import CVBooster, cv  # noqa: E402 - the estimators must exist first

#: Submodules the package answers for without importing them. Each is
#: reachable as `mojotrees.<name>` after a plain `import mojotrees`, and
#: none is imported until it is asked for.
#:
#: - `dask` is experimental and cannot train: no transport ships, so every
#:   `fit` raises `DistributedNotAvailable`. It is the client-side contract
#:   for a backend, not a feature, which is why neither it nor its
#:   estimators are in `__all__`. Importing it does not import dask.
#: - `inspection`, `device_selection`, and `diagnostics` are pure mojotrees
#:   and are lazy only to keep this import cheap. `inspection` reaches
#:   pandas from `trees_to_dataframe` alone, and never at import.
#: - `lgbm_model_io` is experimental LightGBM model-file interop, converted
#:   natively and gated by its own `interop_status()`. It exports nothing at
#:   top level: a conversion is asked for by name, never reached by accident.
_LAZY_SUBMODULES = (
    "dask",
    "device_selection",
    "diagnostics",
    "inspection",
    "lgbm_model_io",
)

#: Top-level names that live in a lazy submodule, and the submodule each
#: comes from. They are in `__all__`: the module they come from is an
#: implementation detail of where the code sits, not of what the package
#: offers.
_LAZY_ATTRS = {
    "explain_device_choice": "device_selection",
    "dump_model": "inspection",
    "trees_to_dataframe": "inspection",
    "trees_to_records": "inspection",
    "get_split_value_histogram": "inspection",
}


def __getattr__(name):
    """PEP 562 resolution for the lazy half of the surface.

    Whatever is resolved is written into the module globals, so the cost is
    paid once and every later access is an ordinary attribute lookup.
    """
    import importlib

    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    origin = _LAZY_ATTRS.get(name)
    if origin is not None:
        value = getattr(
            importlib.import_module(f".{origin}", __name__), name
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(
        set(globals()) | set(_LAZY_SUBMODULES) | set(_LAZY_ATTRS)
    )
