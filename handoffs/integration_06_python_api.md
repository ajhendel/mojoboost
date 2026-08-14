# Integration 06 handoff: the public Python surface for cv and dask

Files this lane touched, and the only ones:

- `python/mojoboost/cv.py`
- `python/mojoboost/dask.py`
- `python/mojoboost/_public_api_plan.py` (new)
- `handoffs/integration_06_python_api.md` (this file)

Nothing central, nothing native, no test, no documentation, no packaging.
Nothing was committed or staged. No Python, Mojo, pixi, pytest, or build
command was run in this lane, so every claim below is from reading the
code, and the validation commands at the end are the ones that would
confirm it.

This lane sits downstream of `handoffs/task15_cv.md` (which built `cv.py`)
and `handoffs/task17_dask.md` (which built `dask.py`). Where it disagrees
with either, the disagreement is called out by name below.

## 1. What changed, and why

### 1.1 One `CVBooster` per run, instead of one per callback phase

`cv()` used to build a fresh `CVBooster` around the fold `Booster`s every
time it ran a callback phase, and then a final, different one for
`results["cvbooster"]`. Three or four wrappers per round, all aliasing the
same models, and the object a callback saw as `env.model` was never the
object the caller got back. A callback that recorded anything on it lost it
at the end of the phase.

`cv()` now owns exactly one `CVBooster` for the run. It is created once
after the folds are built, refreshed by `_sync_folds` at each point where a
fold's `Booster` can have appeared (a `_RetrainFold` has none until it is
advanced), handed to the callbacks as `env.model`, given its
`best_iteration` when the run ends, and returned. Same models, one owner.

Visible differences, none of which any current test asserts against:

- `env.model` is the same object every round, and is the object in
  `results["cvbooster"]` when `return_cvbooster=True`.
- `env.model.fold_names` is now populated during the run; it used to be
  empty until the final wrapper was built.
- `env.model.best_iteration` is `-1` during the run and is filled in
  afterward, which is what it always was for the callback-visible wrapper.

### 1.2 A fitted Dask estimator holds one model, not two

`__init__.py` states the rule for the package: "There is one model object in
this package, `mojoboost.Booster`, and an estimator holds one rather than a
second abstraction of its own." A fitted `DaskMojoBoost*` broke it. It kept
the backend's bytes on `self._model_bytes_` **and** the parsed handle in the
`Booster` behind `_model`, and `_Base.__getstate__` writes the Booster out
as bytes when pickling, so a pickled distributed estimator carried the same
model twice, in two fields, with nothing keeping them in step.

Now `_model_blob_cache` is a cache of the one model and is documented as
one:

- it is dropped from `__getstate__`, so a pickle carries the model once;
- `_model_blob()` returns it, and rewrites it from the Booster through
  `_Base._model_bytes()` when it is absent;
- `to_local()` and `_distributed_predict()` go through `_model_blob()`.

Side effect worth having: an **unpickled** distributed estimator can now
predict on a Dask collection and can answer `to_local()`. Before, both
needed `_model_bytes_`, which survived pickling only by accident of it not
being cleared, and would have been absent from any estimator whose model
arrived any other way.

### 1.3 The worker model cache was keyed on too little and bounded by nothing

`_MODEL_CACHE` mapped `(digest, kind, multiclass, classes)` to a parsed
estimator, but `_estimator_from_bytes` also reads `encoders`, the pandas
category tables. Two fits with identical trees over differently ordered
category tables share a digest, so the second fit would have predicted
through the first fit's encoders, on a pandas frame, silently and wrongly.
`encoders` is now in the key (frozen to tuples there, because the tables
arrive as lists and a dict key may not hold one).

The cache is also bounded now, at `_MODEL_CACHE_LIMIT = 4`, oldest first.
Each entry owns a native model handle, and a long-lived worker predicting
with model after model held every one of them for the life of the process.
This supersedes the "The cache is unbounded and per process ... Whether a
real workload needs eviction is unknown" bullet in
`handoffs/task17_dask.md`, which is now stale.

### 1.4 Capabilities the adapter was not asking for

`require_capabilities` is the whole mechanism by which this module avoids
claiming something a backend cannot do, so a setting that needs distributed
support and is not in the `needed` set is exactly the failure the mechanism
exists to prevent. Three were missing:

- **`goss`** (new capability name). GOSS keeps the largest gradients and
  samples the rest, which is a decision over the global row set. A rank
  sampling its own rows is a different algorithm. Detected through the
  estimator's own `_resolve_boosting()`, so the `boosting` /
  `boosting_type` alias rule keeps one owner.
- **`custom_objective` for the classifier.** The regressor asked for it and
  the classifier did not, though `MojoBoostClassifier.objective` takes a
  callable too.
- **`class_weight`** (new capability name), asked for whenever
  `class_weight` is set. Separately, `class_weight="balanced"` is now
  **refused** by `DaskMojoBoostClassifier.fit`: it is one over the class
  frequency, the frequency is a global row count the client does not have,
  and a per-rank "balanced" would weight the same class differently on
  every rank. An explicit dict is a per-class multiplier a worker can apply
  locally, so it is accepted and asks for the capability.

`BACKEND_PROTOCOL_VERSION` is **not** bumped. It is 0, no backend exists,
and nothing has read version 0 yet. The first backend author reads the
current `CAPABILITIES` set, not a changelog.

### 1.5 Honesty fixes

- `PartitionMeta.label_values` said it was "used to agree on the global
  class list for a classifier". Nothing populates it and nothing reads it;
  the class list is the `classes` argument of `fit`, on purpose. The
  docstring now says so and says why a metadata scan cannot replace the
  argument.
- `predict_proba` on a Dask collection read `self.n_classes_` for the
  output width before `_distributed_predict` could call `_require_fitted`,
  so an unfitted classifier raised `AttributeError` instead of
  `NotFittedError`. It checks first now.
- `_query_groups` indexed `ids[0]` on a possibly empty sequence, raising
  `IndexError` from inside a validator whose whole job is to produce a
  `PartitionError` that says what to fix.
- `require_capabilities` was public, documented, used by the tests, and
  missing from `dask.__all__`. Added.

### 1.6 The circular dependency, narrowed and written down

`dask.py` imported four names from the package at module scope. Three of
them are base classes of the Dask estimators and cannot be lazy: a class
statement needs its bases. The fourth, `group_from_query_ids`, is needed
only when `partition_group` is called, and is now imported there.

So the constraint is now exactly as large as the class statements make it,
and it is stated in the module docstring rather than left to be discovered:
**`mojoboost/__init__.py` must not import `mojoboost.dask` from the top of
its own body**, because the estimators do not exist yet at that point. See
section 3.2 for what to do instead.

`cv.py` has no such constraint. It imports `.basic` and the leaf modules at
module scope and reaches back into the package only from inside function
bodies, which is what `basic.py` already does. Do not move those inner
imports out; that is what would create a cycle.

## 2. The proposed public symbol list

Machine-readable in `python/mojoboost/_public_api_plan.py`, which imports
nothing and executes nothing. Prose version:

### 2.1 Added to `mojoboost.__all__` by this proposal

| name | kind | from | why |
| --- | --- | --- | --- |
| `cv` | function | `mojoboost.cv` | LightGBM's spelling is `lgb.cv(params, train_set, ...)` |
| `CVBooster` | class | `mojoboost.cv` | what `cv(return_cvbooster=True)` returns |

That is the whole list. Two names.

### 2.2 Reachable as submodules, not exported

| module | surface | status |
| --- | --- | --- |
| `mojoboost.cv` | `cv`, `CVBooster` | both promoted above |
| `mojoboost.dask` | 31 names in its own `__all__` | contract only, nothing trains |

`mojoboost.dask` stays a submodule. Its three estimators raise
`DistributedNotAvailable` on every installation that exists, and a name
sitting in `mojoboost.__all__` next to `MojoBoostRegressor` reads as a
feature. Revisit when the checklist at the end of
`handoffs/task17_dask.md` is complete, not before.

### 2.3 Deliberately not exported

- `mojoboost.cv.FoldModel`. An internal adapter over one fold, named
  without an underscore only because the module docstring refers to it. It
  is not in `cv.__all__` and is not a supported extension point. `cv.py`
  now says so in as many words.
- The dask backend protocol (`register_backend`, `CAPABILITIES`,
  `TrainingJob`, `WorldPlan`, `PartitionMeta`, and the rest). It is a
  contract between mojoboost and a transport author, not something an
  application calls. It stays in `dask.__all__`.

## 3. Central edits this lane needs, and cannot make

### 3.1 `python/mojoboost/__init__.py`, the cv export

After `from .basic import Booster, Dataset, train`:

```python
from .cv import CVBooster, cv
```

and `"CVBooster"` and `"cv"` into `__all__`. This is the same request
`handoffs/task15_cv.md` section 1 makes, and it is safe wherever it sits
after `basic` is imported.

**One correction to that handoff.** It says the resulting collision is
familiar because "LightGBM has exactly this shape". LightGBM does not.
`lightgbm.cv` is a function defined in `lightgbm/engine.py`; there is no
`lightgbm/cv.py` for it to collide with. mojoboost is making a choice
LightGBM never had to make, so here it is, made explicitly:

The package attribute `mojoboost.cv` will be **the function**. Consequences,
which belong in a line of the package docstring:

- `mojoboost.cv(params, train_set, ...)` works, which is the point.
- `from mojoboost.cv import cv, CVBooster` keeps working, because that form
  resolves the submodule through `sys.modules` rather than through the
  package attribute.
- `import mojoboost.cv as m` binds `m` to the **function**, not the module,
  and `m.CVBooster` then raises `AttributeError` on a line that looks like
  a module import. Write `from mojoboost import cv, CVBooster` instead.

The alternative that removes the collision entirely is renaming the module
to `mojoboost/engine.py`, which is where LightGBM keeps both `train` and
`cv`. It needs a lane that owns `cv.py` **and**
`python/tests/parallel/test_cv.py`, which imports `mojoboost.cv` by name
(including `_fold_dataset`, `_generated_splits`, and `_query_of_row`). Worth
doing as an end state. Out of scope here, and not worth blocking the export
on.

### 3.2 `python/mojoboost/__init__.py`, reaching `mojoboost.dask`

`mojoboost.dask` has to be reachable after a plain `import mojoboost`, must
not import dask to be reachable, and must not be imported from the top of
`__init__.py` (section 1.6). A module-level `__getattr__` (PEP 562) answers
all three. Proposed, at the **end** of `__init__.py`, verbatim in
`_public_api_plan.LAZY_SUBMODULE_SNIPPET`:

```python
_LAZY_SUBMODULES = ("dask",)


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # answered directly from here on
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
```

Notes for whoever lands it:

- `handoffs/task17_dask.md` section on `__init__.py` says "no edit is
  required" because `import mojoboost.dask` already works. That is true and
  it is not the same statement: without this, `import mojoboost` followed
  by `mojoboost.dask.register_backend(...)` raises `AttributeError`. Either
  behavior is defensible; this proposal picks the one that matches
  `lightgbm.dask`.
- Do **not** replace it with `from . import dask` at the top of the file.
  That is the ordering hazard in section 1.6.
- Task 19 (`inspection`) and task 20 (`device_selection`) may want rows in
  `_LAZY_SUBMODULES` too. Merge from their handoffs; the tuple is the only
  thing that changes.

### 3.3 Packaging

Nothing. `python/pyproject.toml` already carries
`dask = ["dask[distributed]"]` and folds it into the `all` extra. Checked,
not edited.

### 3.4 Documentation

Not this lane's files, so listed rather than written:

- The package docstring in `__init__.py` needs the `mojoboost.cv`
  collision line from 3.1.
- `docs/LIGHTGBM_PARITY.md` should carry the table in section 4 below.
  `tools/check_parity.py` runs as its own CI job, so an entry that names a
  symbol should be checked against what the parity tool expects to find.
- `README.md` should not list `mojoboost.dask` among the things that work.

## 4. Intentional differences from LightGBM

Also in `_public_api_plan.LIGHTGBM_DIFFERENCES`, one row each.

**Cross-validation**

| LightGBM | mojoboost | why |
| --- | --- | --- |
| `results` holds metric lists only | plus `results["iterations"]` | a ranking cv reports one round, so a list's length is not the round count |
| every round of every objective is scored | a ranking cv reports the final round only, and refuses `callbacks` and `early_stopping_rounds` | `Booster.update()` does not cover LambdaRank, so a ranking fold is trained once at the full round count. Refused rather than silently ignored |
| callbacks get a 5-tuple with the stdev | callbacks get `(name, metric, mean, higher)` | `mojoboost.callback` unpacks four. The deviation is in the returned history, so `log_evaluation(show_stdv=True)` has nothing to show |
| `reset_parameter()` schedules a change | `CVBooster.reset_parameter` raises | `Booster` has no way to be told, so the schedule would look like it ran |
| `init_model` continues the folds | `init_model` is refused | continued training compares the binning, and each fold bins itself over its own rows, so every fold would be rejected by the trainer |

**Dask**

| LightGBM | mojoboost | why |
| --- | --- | --- |
| `lightgbm.dask` trains across a cluster | `mojoboost.dask` cannot train; `fit` raises `DistributedNotAvailable` | no transport, no backend |
| the classifier infers classes from `y` | `fit(X, y, classes)` requires the global list | a scan that missed a rare class gives two ranks different encodings and no error |
| the ranker takes `group` | the ranker takes `query_ids`, one sequence per partition | a group array cannot say whether the query at a partition edge continues into the next partition |
| `predict` accepts `pred_leaf` / `pred_contrib` | both refused on a collection | output width depends on the model, and collection metadata is written first |
| class weighting is available | `class_weight="balanced"` is refused, a dict is accepted | "balanced" needs a global row count per class (section 1.4) |

One more that is not a LightGBM difference but will look like a bug to
someone: after a distributed fit, `feature_importances_` with
`importance_type="gain"` returns zeros and warns. Split gains are not in
the serialized model format, and a distributed fit's model arrives as
bytes. This is the same behavior `MojoBoostRegressor.load()` has and is
already documented on `feature_importances_`. Split-count importance is
correct.

## 5. Native and binding work this surface is waiting on

None of it is required for the exports above. Each item removes a
documented refusal.

### 5.1 Would let `cv()` drop a refusal

1. **`Booster.update()` for LambdaRank.** `Booster._require_trainable`
   raises for `_eval.RANKING`, which is the whole reason a ranking cv
   reports one round and refuses callbacks and early stopping. Detailed in
   `handoffs/task15_cv.md` section 2.1. Highest value item on this list.
2. **Trainer state carried across `update()` calls.** `_IncrementalFold`
   costs one full rescoring pass per round per fold, so an R-round cv is
   O(R) rescorings rather than O(1). Correct, and quadratic.
3. **A way to change a hyperparameter between rounds.** Would turn
   `CVBooster.reset_parameter` from a raise into a forward.
4. **A binning-identity escape for `init_model`.** Continued training
   compares the dataset's binning against the model's; a fold cannot
   satisfy it by construction.

### 5.2 Would let `mojoboost.dask` stop being a contract

All of it is `src/mojoboost/distributed_transport.mojo` and the checklist
at the end of `handoffs/task17_dask.md`, which this lane does not duplicate.
Two items this lane adds to it:

5. **A global GOSS draw**, if the `goss` capability is ever to be
   declared by a backend. Per-rank sampling is a different algorithm.
6. **Global class counts**, if `class_weight="balanced"` is ever to work
   distributed. Today it is refused on the client.

### 5.3 A binding that would simplify this lane's own code

7. **Model serialization to and from bytes, without a file.**
   `_Base._model_bytes()` writes the model to a temporary file and reads it
   back, because the extension module exposes `save(handle, path)` and
   `load(path)` and nothing else. Every pickle of any estimator pays for
   it, and `_model_blob()` now pays for it too whenever the cache is cold
   (an unpickled distributed estimator predicting on a collection). A
   `save_to_string` / `load_from_string` pair on the binding would remove
   the temporary file from both paths. Small, self-contained, and it
   touches no algorithm.

## 6. Risks

1. **The `mojoboost.cv` collision is silent.** `import mojoboost.cv as m`
   will bind the function, and the error surfaces later, on `m.CVBooster`,
   as `AttributeError: 'function' object has no attribute 'CVBooster'`.
   Anyone reading that traceback will not think "shadowed submodule". The
   mitigation is the docstring line in 3.1, and the fix is 3.1's
   `engine.py` alternative.
2. **`_MODEL_CACHE_LIMIT = 4` is a guess.** It is chosen for the pattern
   the cache exists for (one model, many partitions of one collection,
   then the next model). A workload that interleaves predictions from five
   or more models on one worker will thrash and reparse. No workload has
   ever run, so there is nothing to measure it against. `clear_model_cache`
   still empties it, and the constant is one line.
3. **The `class_weight="balanced"` refusal is new behavior.** It fires at
   `fit` on a Dask classifier, before the cluster is touched. Anyone who
   wrote that call could not have gotten a trained model out of it (no
   backend exists), so nothing working breaks, but it is a behavior change
   and it is not covered by a test.
4. **`goss`, `class_weight`, and the classifier's `custom_objective` are
   new capability requests.** A backend that would have been handed such a
   fit and quietly trained something wrong will now be refused, which is
   the intent. Since no backend exists, nothing observable changes today.
5. **`_resolve_boosting()` is now called during capability negotiation.**
   It raises `ValueError` for an unknown or contradictory `boosting` /
   `boosting_type` pair, so that error now surfaces from `fit` slightly
   earlier than it would have. Same error, same message, same owner.
6. **Dropping `_model_blob_cache` from the pickle changes what a pickled
   distributed estimator contains.** Smaller, and it regenerates. The
   regenerated bytes come from re-serializing the Booster, so they are
   semantically the model but not necessarily byte-identical to what the
   backend returned. Nothing compares them; if a future backend wants the
   original bytes preserved verbatim across a pickle, that is a deliberate
   second field and should be argued for on its own.
7. **Everything about `DaskRuntime` is still unexercised.** Unchanged by
   this lane and repeated here so it is not lost: it has never run against
   a cluster.
8. **No test was written or run in this lane.** The changes in section 1
   are reasoned from the code. Section 7 is how to find out.

## 7. Exact validation commands for a later lane

One focused command per module, in order. Nothing here is a full suite.

Build the extension once (the Python tests need `_mojoboost.so`):

```
pixi run build-python
```

Then, per module:

```
pixi run -e pytest pytest -q python/tests/parallel/test_cv.py
pixi run -e pytest pytest -q python/tests/parallel/test_dask_contract.py
```

Expected: both green with no new skips. `test_cv.py` exercises the
callback path that section 1.1 rewired (`test_callbacks_see_the_across_fold_means`,
`test_a_before_iteration_callback_runs_before_the_round`,
`test_a_callback_that_stops_truncates_the_history`) and the CVBooster block
at the end of the file. `test_dask_contract.py` exercises the pickle,
`to_local`, prediction, and model-cache paths that sections 1.2 and 1.3
touched, in particular
`test_the_worker_model_cache_parses_a_model_once` (its two predictors share
a key, so one entry against a limit of four and nothing is evicted) and
`test_a_fitted_estimator_pickles_without_its_cluster`.

Import safety, which no test covers and which is the claim in
`dask.py`'s docstring. Run in an environment where dask is installed, since
passing with dask absent proves nothing:

```
python -c "import sys, mojoboost.dask; assert 'dask' not in sys.modules, sorted(m for m in sys.modules if m.startswith('dask'))"
python -c "import sys, mojoboost; assert 'mojoboost.dask' not in sys.modules"
```

The plan module is data. Importing it must not drag in dask or bind any
new package attribute:

```
python -c "import sys, mojoboost._public_api_plan as p; assert not [m for m in sys.modules if m.startswith('dask')]; print(p.PLAN_VERSION, len(p.PROPOSED_ADDITIONS))"
```

After section 3.1 and 3.2 land in `__init__.py`, these are the three that
prove the surface is what this file proposed:

```
python -c "import mojoboost as mb; mb.cv, mb.CVBooster; from mojoboost.cv import cv, CVBooster"
python -c "import mojoboost as mb; mb.dask.CAPABILITIES; print('dask reachable, dask not imported')"
python -c "import mojoboost as mb, mojoboost._public_api_plan as p; assert set(mb.__all__) == set(p.CURRENT_TOP_LEVEL) | {e['name'] for e in p.PROPOSED_ADDITIONS}"
```

The third one is the drift check. If it fails, either `__all__` grew
somewhere else (likely, other lanes are adding to it) or this plan went
stale; reconcile, do not silence.

## 8. What this lane did not do

- Did not export anything. `__init__.py` is untouched, so
  `mojoboost.cv(...)` does not work yet and `mojoboost.dask` is not an
  attribute of the package yet.
- Did not rename `cv.py`, which is the real fix for section 6 risk 1.
- Did not write or run a test, or run anything else.
- Did not touch `inspection.py` or `device_selection.py`, whose entries in
  `_public_api_plan.PROPOSED_LAZY_SUBMODULES` are placeholders carrying
  their owning task number and nothing else.
- Did not bump `BACKEND_PROTOCOL_VERSION`, and did not implement a backend.
- Did not touch `docs/`. Section 3.4 lists what wants updating.
