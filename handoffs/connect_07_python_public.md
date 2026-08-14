# Connect 07: estimator features and the public Python surface

Files this lane touched, and the only ones:

- `python/mojoboost/__init__.py`
- `python/mojoboost/basic.py`
- `python/mojoboost/_eval.py`
- `python/mojoboost/_public_api_plan.py`
- `handoffs/connect_07_python_public.md` (this file)

`python/mojoboost/callback.py` is owned by this lane and was **not**
changed; section 8 says why. `python/mojoboost/_public_api.py` does not
exist and was not created: `_public_api_plan.py` is the record and
`__init__.py` is the mechanism, and a third module between them would be
the duplicate glue this round exists to remove.

Nothing was committed or staged by this lane. No Mojo, pixi, pytest,
build, wheel, benchmark, or network command was run. One Python
invocation was made and it was `python3 -c "ast.parse(open(...).read())"`
against `_eval.py`, a syntax check; every other claim below is from
reading the code, and every command in section 12 is UNRUN.

**A commit landed mid-lane.** `dc21f03 "Connect accelerator and public API
foundations"` was made by another party while this lane was editing, and
it swept in the `basic.py` and `_eval.py` edits and part of the
`__init__.py` edits described here. Nothing was reverted and nothing of
another lane's was touched; the working tree still holds the rest.

---

## 1. Implementations found

Inventory before editing, by capability.

| Capability | Implementations found | Chosen as authoritative |
|---|---|---|
| Cross-validation | `python/mojoboost/cv.py` (1175 lines, real: builds folds, drives `train`/`Booster.update`, runs callbacks, early stopping, returns `CVBooster`) | `cv.py`. Nothing else does this. |
| Structured model inspection | `python/mojoboost/inspection.py` (facade + text-parser fallback), `src/mojoboost/model_dump.mojo`, `bindings/inspection_bindings.mojo` (new, unregistered) | `inspection.py` as the one Python reader; it already prefers the native hooks when the build has them. |
| Device decision | `src/mojoboost/device.mojo` / `device_policy.mojo` (native), `python/mojoboost/device_selection.py` (formatter over it, `_FullNativePolicy` + `_NarrowNativePolicy`), and `_Base._resolve_device` calling `_mojoboost.resolve_device` directly | native, reached through `device_selection`. |
| Prediction device | `bindings/_mojoboost.mojo` `predict_batch` / `predict_proba_batch` / `predict_leaf_batch` / `predict_leaf_multiclass_batch` / `gpu_predict_capability` (registered, new this round), `src/mojoboost/gpu_predict.mojo`, and the older `predict_range` / `predict_proba_range` / `predict_leaf*` | the batch entry points, with the older ones as the CPU-only fallback. |
| Metric names / aliases / directions / defaults | `src/mojoboost/objective_registry.mojo`, `bindings/objective_bindings.mojo` (new, unregistered), the mirrored dicts in `python/mojoboost/_eval.py` behind `_TABLE` | the registry, through `_eval._NativeTable`, with the mirror as the fallback. |
| Startup / install description | `python/mojoboost/diagnostics.py` | it; `build_info()` already read it. |
| Distributed | `python/mojoboost/dask.py` | it, and it cannot train. Kept lazy and unexported. |

Two things that looked like duplicates and are not:

- `device_selection._NarrowNativePolicy` calls `_mojoboost.resolve_device`,
  which is the same engine `_Base._resolve_device` called directly. It is a
  narrower view of one policy, not a second one.
- `inspection.py`'s text parser below its DELETION POINT banner is a second
  implementation of the schema, in Python. It is the other lane's to
  delete and is not touched here.

---

## 2. Call path, before and after

### 2.1 Public surface

Before:

```
import mojoboost
mojoboost.cv                 -> AttributeError
mojoboost.inspection         -> AttributeError
mojoboost.device_selection   -> AttributeError
mojoboost.dask               -> AttributeError
mojoboost.diagnostics        -> AttributeError
```

Every one of those modules existed, was documented, and was reachable only
by `import mojoboost.<name>` first. `__all__` held 19 names and none of
them was `cv`.

After: `__all__` holds 26. `cv` and `CVBooster` are imported eagerly at the
end of `__init__.py` (they must be: `cv.py` reaches back into this module).
The four submodules and five of the names resolve through a module-level
`__getattr__` (PEP 562) and cache into `globals()` on first access, so
`import mojoboost` still costs the extension and nothing else.

### 2.2 Prediction device

Before: `predict` took no `device`. Every dense call went to
`_mojoboost.predict_range` / `predict_proba_range` / `predict_leaf*`, which
have no device parameter, so prediction was CPU-only with no way to say
otherwise.

After:

```
MojoBoostRegressor.predict(X, device="gpu")
  -> _Base._batch_params(device, start, stop, raw_score)   # requested name
  -> _Base._predict_batch("predict_batch", "predict_range", ...)
  -> _mojoboost.predict_batch(model, x, n_rows, n_features, params, out)
       -> _predict_device(params, ...)            # bindings
            -> gpu_predict_support(...).raise_if_blocked()   # explicit gpu
            -> mojo_resolve_device(...)                      # device.mojo
       -> Model.predict_batch(..., device)
  -> returns the backend that ran
```

The same for `predict_proba` (`predict_proba_batch`), the ranker's
`predict`, and `pred_leaf` (`predict_leaf_batch` /
`predict_leaf_multiclass_batch`). Python names the device it *wants*; Mojo
decides and answers. Nothing in `__init__.py` thresholds, estimates, or
infers a backend.

### 2.3 Fit device

Before: `_Base._resolve_device` did its own `_DEVICES` spelling check, then
called `_mojoboost.resolve_device` directly and wrapped the failure in
`RuntimeError`.

After: it builds a `device_selection.Workload` and calls
`device_selection.select_device`, which is the one Python door to the
native policy. The refusal is now `DeviceUnavailableError`, a
`RuntimeError` subclass carrying the identical native message plus the
report. The direct `resolve_device` call is kept as the fallback for a
build whose `device_selection` cannot be imported at all.

The workload is no longer shape-only. `decide_device` is bound (it landed
in `bindings/_mojoboost.mojo` while this lane was open), so the extra
fields now decide rather than sit inert, and all four fit paths declare
them:

| Call site | Declares |
|---|---|
| `MojoBoostRegressor.fit`, dense | `objective_code`, `categorical`, `has_eval_set` |
| `MojoBoostClassifier.fit`, dense | `objective_code` (binary only, see below), `categorical`, `has_eval_set` |
| `MojoBoostRanker.fit` | `objective_code=_LAMBDARANK`, `categorical`, `has_eval_set` |
| `_Base._sparse_fit_params`, both estimators | `sparse=True`, `objective_code`, `categorical` |

`max_bin` and `has_missing` are read off the estimator inside
`_resolve_device`, so every call carries them. `MojoBoostClassifier`
gained `_objective_code(n_classes=None)`, which answers `_BINARY_LOGISTIC`
for two classes and `None` (undeclared) for softmax, because softmax is
not in the device vocabulary at all: it is its own trainer and what the
native policy gates on there is `n_outputs`. `MojoBoostRanker` gained
`_objective_code()` returning `_LAMBDARANK`. Undeclared is a reported gap
(`WARN_INCOMPLETE_REQUEST`), never an assumed value; that is
`OBJECTIVE_UNSPECIFIED` in `src/mojoboost/device_policy.mojo`.

The four Python CPU-only refusals (eval set, sparse, custom objective,
lambdarank) each have an exact native counterpart, and the native one now
fires first. They are kept as backstops behind one helper,
`_Base._gpu_unsupported`, whose docstring names the block each mirrors:

| Python backstop | Native block that now owns it |
|---|---|
| `_fit_with_metrics` | `BLOCK_VALIDATION_SET` |
| `_sparse_fit_params` | `BLOCK_SPARSE_INPUT` |
| `_fit_custom` | `BLOCK_CUSTOM_OBJECTIVE` |
| `MojoBoostRanker.fit` | `BLOCK_RANKING_OBJECTIVE` |

They are not dead code and must not be deleted with the native capability
work. A build that exposes `resolve_device` but not `decide_device` is the
`CONTRACT_NARROW` case `device_selection` still supports, and its answer
is shape-only, so an explicit `device="gpu"` would otherwise reach a
trainer with no kernel for the request. Against a full-contract build they
never fire, because `device` arrived already refused or already resolved
to `"cpu"`.

### 2.4 Metric registry

Before: `_eval._TABLE = _CompatTable()`, always, answering from mirrored
dicts.

After: `_eval._TABLE = _selected()`, which takes one snapshot of
`registry_metrics` / `registry_metric_aliases` / `registry_objectives` /
`registry_objective_aliases` and uses `_NativeTable` when all four are
present and consistent, and `_CompatTable` otherwise. Today all four are
absent (section 5), so the answer is byte-identical to before.
`_eval.registry_source()` reports which one is live.

### 2.5 Model inspection from the model object

Before: `Booster` had no `dump_model`, and `basic.py`'s docstring said
structured inspection did not exist. `_Base` had no `objective_`,
`feature_name_`, or `n_features_`.

After: `Booster.dump_model` / `trees_to_dataframe` / `trees_to_records` /
`get_split_value_histogram` delegate to `mojoboost.inspection` (import
inside the method, so `basic` and `inspection` are not a cycle), and
`_Base.objective_` / `feature_name_` / `n_features_` are properties reading
the same module. This is `handoffs/migration_19_model_inspection.md` §11,
applied.

---

## 3. Connections completed

1. **`cv` and `CVBooster` exported.** `from .cv import CVBooster, cv` at the
   end of `__init__.py`, both in `__all__`.
2. **`__getattr__` / `__dir__`** for `dask`, `device_selection`,
   `diagnostics`, `inspection`, plus `explain_device_choice`, `dump_model`,
   `trees_to_dataframe`, `trees_to_records`,
   `get_split_value_histogram`.
3. **`predict(device=...)`, `predict_proba(device=...)`** on all three
   estimators, routed to the batch entry points, including the
   `pred_leaf` path.
4. **Sparse prediction carries the device** in its params dict, so the
   native `_refuse_gpu_sparse` produces the refusal.
5. **`_resolve_device` goes through `device_selection`.**
6. **`_eval` reads the native registry** when the build exposes it.
7. **`Booster` gained the four inspection methods**; `basic.py`'s "No
   `dump_model` / `trees_to_dataframe`" bullet is replaced by what is
   actually true, including the two columns that cannot be filled.
8. **`_Base` gained `objective_`, `feature_name_`, `n_features_`.**
9. **`train()` refuses `feval` and `callbacks` by name** instead of as an
   unexpected keyword, and points at `fit(eval_set=...)` and `cv()`.
10. **`_public_api_plan.py` bumped to version 2**, with
    `CURRENT_TOP_LEVEL` matching the real `__all__` and the
    `inspection` / `device_selection` placeholders replaced.
11. **The fit workloads are complete**, not shape-only: objective code,
    sparsity, categorical presence, bin count, missing-value handling, and
    the presence of an eval set all reach
    `src/mojoboost/device_policy.mojo` now that `decide_device` is bound
    (section 2.3).
12. **`MojoBoostClassifier._objective_code` and
    `MojoBoostRanker._objective_code`** added, so all three estimators can
    name their objective to the policy. The regressor already had one.
13. **The four CPU-only guards became documented backstops** behind
    `_Base._gpu_unsupported`, each naming the native block that now owns
    the decision.

---

## 4. Duplicates fused or quarantined

| Duplicate | Disposition |
|---|---|
| `_Base._resolve_device`'s own `_DEVICES` spelling check + direct native call | Fused: `device_selection` does both. The direct call is kept only as the import-failure fallback and is documented as such. |
| A Python-side device *resolution* for prediction (an earlier draft of this lane resolved through `_resolve_device` before calling) | Removed before it landed. Prediction sends the requested name and Mojo resolves; two resolvers would have double-gated `auto`. |
| `_eval`'s mirrored `_METRICS` / `_ALIASES` / `_DEFAULTS` / `_TASK_DEFAULTS` | Quarantined, not deleted. They stay behind the marked block and `_CompatTable`, which is the fallback while the registry is unbound. `handoffs/migration_21_objective_metric_registry.md` §5 lists them as deletable *after* wiring; this lane wired the reader, not the deletion. |
| `__init__.py`'s `_SQUARED_ERROR.._CROSS_ENTROPY`, `_UNIMPLEMENTED_OBJECTIVES`, `_OBJECTIVES`, `_OBJECTIVE_PARAM` | Left alone. Same handoff lists them as deletable after `registry_objectives` / `registry_objective_unimplemented` are bound; they are load-bearing today and deleting them now would break every fit. Section 6 carries the request. |
| Four separately worded `if device != "cpu": raise RuntimeError(...)` blocks in `__init__.py` | Fused into `_Base._gpu_unsupported`, and demoted: each is now a backstop for the narrow contract, with the native block code that decides written next to it. Not deleted, for the reason in section 2.3. |
| The sparse fit's `if self.device == "gpu"` check, which read the raw attribute and so missed the `device_type` alias | Fused: `_sparse_fit_params` asks `_resolve_device(..., sparse=True)`, which resolves the alias and lets `BLOCK_SPARSE_INPUT` write the refusal. |
| `inspection.py`'s text parser | Not this lane's file. Untouched. |

---

## 5. Remaining disconnections

Three items that stood here when this handoff was first written are now
resolved and are recorded as such rather than deleted, because other
handoffs point at them. The registry bindings, the inspection bindings,
and `decide_device` were all registered in `bindings/_mojoboost.mojo` by
the bindings lane while this lane was open (section 6a). So
`_eval._native_snapshot()` now finds all four hooks, `inspection.objective_of`
now takes the `objective_code` hook instead of a `model_to_string()` round
trip, and the fit workloads are complete (section 2.3). None of that is
asserted as working: it is what the code paths select, unbuilt and untested
here.

What remains:

1. **The built extension may predate these entry points.**
   `python/mojoboost/_mojoboost.so` on disk was not rebuilt by this lane,
   so on that artifact `predict_batch` and the registry hooks may be
   absent and every `device=` other than `"cpu"` raises
   `_NO_DEVICE_PREDICT` while `_eval` runs the mirror. That is the
   fallback behaving correctly, not a defect, and it clears on a rebuild
   (`bindings/build.sh`, UNRUN, section 13).
2. **Validation and early stopping still *train* CPU-only, and now the
   native policy is what says so.** `BLOCK_VALIDATION_SET` in
   `src/mojoboost/device_policy.mojo` refuses `device="gpu"` with an eval
   set; `_fit_with_metrics` keeps the same refusal as a narrow-contract
   backstop. This is a missing capability, not a missing connection. The
   `gpu_validation_open` / `_reset` / `_accumulate` / `_metric` / `_raw`
   entry points exist and are registered, but they are a scoped handle for
   use *inside* a training loop, and the loop is
   `src/mojoboost/custom_metric.mojo`, which has no device parameter and
   is driven end to end by one `fit_with_metrics` call. Python cannot
   interleave from outside. Section 6c carries the request, and lifting it
   is a native change plus the deletion of one Python backstop, in that
   order.
3. **`pred_contrib` has no device path.** There is no
   `predict_contrib_batch`. An explicit `device="gpu"` is refused by
   `_refuse_device`; `"auto"` runs on the CPU, which is where it would
   resolve anyway.
4. **The sparse fit still passes `"cpu"` into `_params`** rather than the
   resolved name. That is not a gap: the resolution above it can only
   return `"cpu"` or raise, and the literal is what the CSC trainers read.
   It is listed so that whoever adds a sparse GPU kernel knows the string
   is there.
5. **`mojoboost.cv` shadows the module.** By decision, recorded in
   `_public_api_plan.NAME_COLLISIONS`. `import mojoboost.cv as m` binds the
   function. The clean fix is renaming `cv.py` to `engine.py`, which needs
   a lane owning `cv.py` and `python/tests/parallel/test_cv.py`.

---

## 6. Cross-lane patch requests, exact

### 6a. To the bindings lane. **LANDED, no action left**

This request was written when `bindings/_mojoboost.mojo` imported none of
these modules. It now imports `dataset_bindings`, `basic_bindings`,
`distributed_bindings`, `inspection_bindings`, and `objective_bindings`,
and registers every entry point below plus `decide_device`,
`objective_code_of_name`, and the `dataset_*` family. The patch is kept
verbatim only as the record of what was asked for; do not apply it again.
What it says about *behavior* is still the thing to read, and it is at the
bottom of this subsection.

The original request follows. In the import block:

```mojo
from objective_bindings import (
    metric_code_of_name,
    objective_code_of_name,
    objective_name_status_of,
    registry_metric_aliases,
    registry_metrics,
    registry_objective_aliases,
    registry_objective_unimplemented,
    registry_objectives,
    registry_vocabulary,
)
from inspection_bindings import (
    dump_model,
    dump_model_multiclass,
    dump_leaf_index,
    dump_leaf_index_multiclass,
    dump_raw_scores,
    dump_raw_scores_multiclass,
    objective_code,
    split_values,
    split_values_multiclass,
)
```

and in the `def_function` block, next to `resolve_device`:

```mojo
m.def_function[registry_objectives]("registry_objectives")
m.def_function[registry_objective_aliases]("registry_objective_aliases")
m.def_function[registry_objective_unimplemented](
    "registry_objective_unimplemented"
)
m.def_function[registry_metrics]("registry_metrics")
m.def_function[registry_metric_aliases]("registry_metric_aliases")
m.def_function[registry_vocabulary]("registry_vocabulary")
m.def_function[objective_code]("objective_code")
m.def_function[dump_model]("dump_model")
m.def_function[dump_model_multiclass]("dump_model_multiclass")
m.def_function[split_values]("split_values")
m.def_function[split_values_multiclass]("split_values_multiclass")
m.def_function[dump_raw_scores]("dump_raw_scores")
m.def_function[dump_raw_scores_multiclass]("dump_raw_scores_multiclass")
m.def_function[dump_leaf_index]("dump_leaf_index")
m.def_function[dump_leaf_index_multiclass]("dump_leaf_index_multiclass")
```

`bindings/dataset_bindings.mojo` was unregistered too (`dataset_create_csc`,
`dataset_subset`, `dataset_metadata`, `dataset_field`,
`dataset_bin_upper_bounds`, and the rest); it is registered now. Nothing in
this lane's files reads it, so it was noted rather than requested;
whichever lane owns `Dataset`'s Python surface should say what it needs.

Nothing in Python changes when this lands. `_eval._selected()` starts
returning `_NativeTable`, `inspection.objective_of` starts using the
`objective_code` hook, and `inspection.dump_model` starts using the native
dump. The one visible behavior change is stated in `_eval.py`'s docstring:
`_DEFAULTS` is keyed on the objective *name the user typed* and so has an
alias hole (`objective="regression_l2"` has no default metric); the
registry keys it on the code and closes it. Assert
`mojoboost._eval.registry_source()` in whichever test cares.

### 6b. To the objectives/metrics lane (`src/mojoboost/objective_registry.mojo`)

Nothing required. Two field contracts that `_eval._NativeTable` reads, so
that changing them is a knowing change and not a silent one:

- `registry_metrics()` record is `(code, canonical_name, task_name,
  higher_is_better, needs, transform)`; positions 0, 1, 2, 3 are read.
- `registry_objectives()` record is 13 fields; positions 0 (`code`), 2
  (`task_name`), and 12 (`default_metric_code`) are read, and `-1` means
  "no default".
- `task_name` must keep spelling the four tasks `regression`, `binary`,
  `multiclass`, `ranking`, which is what `_eval.REGRESSION` and friends
  hold.
- `_NativeTable` derives the per-task default metric as "the one
  `default_metric_code` every non-regression objective of that task
  agrees on". If two objectives of one task ever carry different defaults,
  that task silently loses its task-level default and falls through to the
  per-objective lookup. If that is ever intended, say so and this becomes
  an explicit registry query instead.

### 6c. To the validation lane (`src/mojoboost/custom_metric.mojo` + bindings)

Still open, and now purely a capability request. The *decision* moved
native when `decide_device` was bound: `BLOCK_VALIDATION_SET` is what
refuses `device="gpu"` with an eval set, because
`_resolve_device(..., has_eval_set=True)` declares it. What is missing is
the ability to score validation metrics on a device at all, which lives in
the file below and cannot be reached from Python.

To lift it, in this order:

1. `train_with_callbacks` (and the multiclass and ranker forms) take a
   device code and, for a device run, replace `_update_valid_raw` with the
   `GpuValidation` handle already bound: `gpu_validation_open`,
   `gpu_validation_reset`, `gpu_validation_accumulate`,
   `gpu_validation_metric` / `gpu_validation_raw`.
2. `fit_with_metrics` / `fit_multiclass_with_metrics` /
   `fit_ranker_with_metrics` read `params["device"]`, which the estimator
   already puts there (`_Base._params` sets `"device"`).
3. Remove the `BLOCK_VALIDATION_SET` gate in
   `src/mojoboost/device_policy.mojo`, which is the refusal users
   actually meet.
4. Then, and only then, in `python/mojoboost/__init__.py`, delete exactly
   this, in `_fit_with_metrics`:

```python
        # Backstop; BLOCK_VALIDATION_SET is what refuses this on a build
        # whose native policy can be asked. See `_gpu_unsupported`.
        self._gpu_unsupported(
            device, "validation metrics are scored on the CPU"
        )
```

   and the "Validation is scored on the CPU, so `device="gpu"` with an
   `eval_set` raises rather than falling back" sentence in the module
   docstring. Leave `_gpu_unsupported` itself and its three other callers
   alone. Do not delete it before a test asserts the device and host
   metric histories agree to the documented tolerance: the two differ at
   Float32 precision and early stopping compares a metric against its own
   running best, so a mixed history is not merely imprecise.

### 6d. To the device policy lane. **Python side done**

`_resolve_device` now sends the full workload from all four fit paths
(section 2.3), which is `handoffs/migration_20_device_policy.md` §2b
applied. Two things to know, neither a request:

- The classifier declares an objective code for binary only. Softmax
  passes `objective_code=None`, which crosses as `OBJECTIVE_UNSPECIFIED`
  and carries `WARN_INCOMPLETE_REQUEST`. If a multiclass objective code
  is ever added to the device vocabulary, `MojoBoostClassifier._objective_code`
  is the one place to say so.
- `_NarrowNativePolicy` is still reachable and still needed. Deleting it,
  or the `_gpu_unsupported` backstops that cover it, is a decision to stop
  supporting a build that has `resolve_device` without `decide_device`.
  Make it deliberately, in one change, in both files.

### 6e. To the documentation lane

- `docs/LIGHTGBM_PARITY.md` gains rows for `predict(device=)`,
  `Booster.dump_model`, `Booster.trees_to_dataframe`, `objective_`,
  `feature_name_`, `n_features_`, `mojoboost.cv`, and the
  `mojoboost.cv` module/function collision. `pixi run check-parity` is its
  own CI job, so edit it in the same commit as this surface.
- `README.md` still describes the package as reached by `import
  mojoboost.cv`; the top-level `cv` is now the spelling.

---

## 7. Fallbacks preserved

Every connection is a strategy switch with the established path underneath.
None of them is a heuristic; each is a presence check on a named entry
point, and each failure is loud.

| Connection | Fallback | What decides |
|---|---|---|
| `_predict_batch` | `predict_range` / `predict_proba_range` / `predict_leaf*` for `device="cpu"`; `RuntimeError` for anything else | `getattr(_mojoboost, entry, None)` |
| `_sparse_predict_params` | no `device` key at all for `"cpu"`; `RuntimeError` for anything else on a build without `predict_batch` | `getattr(_mojoboost, "predict_batch", None)` |
| `_resolve_device` | direct `_mojoboost.resolve_device` | `from . import device_selection` raising |
| the full workload | `device_selection`'s own `_NarrowNativePolicy`, which drops to `resolve_device` and answers on shape alone | `native_contract()`, that is, whether `decide_device` is bound |
| `_gpu_unsupported`, four call sites | nothing below it; it is itself the floor under the narrow contract | `device != "cpu"` after the policy has already answered |
| `_eval._TABLE` | `_CompatTable` and the mirrored dicts | all four registry hooks present *and* the snapshot internally consistent |
| `objective_` etc. | `inspection`'s own fallbacks | `inspection.py`'s existing `_hook` checks |

The `_eval` fallback is deliberately all-or-nothing: a snapshot that is
missing one hook, has an alias for a metric the registry did not list, or
cannot be built at all falls back whole, rather than answering some
queries natively and some from the mirror. Mixing the two is the drift
that module exists to prevent.

---

## 8. `callback.py`: examined, unchanged

Read for disconnections and none found. `RESETTABLE` is indexed
slot-for-slot against `RESET_SLOTS` in `bindings/_mojoboost.mojo` and both
sides say so; `CallbackRunner`, `_EarlyStopping`, and
`resolve_early_stopping` are all reached from `_fit_with_metrics` and from
`cv.py`; the four factories are re-exported at the top level and were
already in `__all__`. `TrainingHandle` and `RESETTABLE` are in
`callback.__all__` and are deliberately not re-exported at the top level:
they are the bridge contract, not application API. Adding an export for
its own sake would have been the "an import is not integration" mistake.

---

## 9. Argument coverage: reaches an implementation, or raises

Checked one by one, because "accepted and ignored" is the failure this
round is about. Every row is either wired or raises with a message naming
the alternative.

| Feature | Status |
|---|---|
| regression / binary / multiclass / ranking | wired, four trainers |
| dense | wired |
| sparse fit | wired (CSC); `device="gpu"` raises |
| sparse predict | wired (CSR); `device="gpu"` raises, natively when the build can |
| sparse + `eval_set` / `pred_leaf` / `pred_contrib` / iteration slicing / ranking / custom objective | each raises, naming `.toarray()` |
| categorical | wired, with the fit-time encoders reused at predict |
| `sample_weight` | wired, validated finite/nonnegative/not-all-zero |
| `class_weight` | wired (`_class_weight_rows`); `"balanced"` computed from the training rows |
| `eval_set` / `eval_names` / `eval_sample_weight` / `eval_group` | wired; each raises without an `eval_set` |
| `eval_metric` names | resolved through `_eval` (registry or mirror); a metric from another task raises and names what this one takes |
| `eval_metric` callables | wired through the metric bridge; combined with `eval_sample_weight` raises |
| `early_stopping_rounds`, `primary_metric`, `min_delta` | wired; without an `eval_set` raises |
| `callbacks` on `fit` | wired; without an `eval_set` raises; softmax and ranking trainers refuse rather than ignore |
| `callbacks` / `feval` on `train()` | **now raise by name**, pointing at `fit` and `cv` |
| `init_model` on `train()` | wired (`_continue`, binning checked) |
| `init_model` on `cv()` | refused, with the reason |
| custom objective | wired; `device != "cpu"` raises; with an `eval_set` raises |
| `raw_score` / `pred_leaf` / `pred_contrib` | wired; more than one at a time raises |
| `start_iteration` / `num_iteration` | wired, LightGBM's clamping |
| `validate_features` | wired |
| `device` on `fit` | wired through the native policy |
| `device` on `predict` / `predict_proba` | **new**, wired; unsupported combinations raise |
| every alias pair (`device`/`device_type`, `boosting`/`boosting_type`, `min_data_in_leaf`/`min_child_samples`, `lambda_l1`/`reg_alpha`, `lambda_l2`/`reg_lambda`, `min_child_hess`/`min_child_weight`, `bagging_fraction`/`subsample`, `bagging_freq`/`subsample_freq`, `categorical_feature`/`categorical_features`) | resolved once at fit time; two different non-default values raise rather than warn |
| GOSS + row bagging | raises (LightGBM silently disables bagging) |
| unimplemented LightGBM objectives | raise by name with the reason |

---

## 10. Fitted attributes

| Attribute | Source | Present when |
|---|---|---|
| `n_features_in_`, `feature_names_in_` | the fit | always / when `X` carried names |
| `classes_`, `n_classes_` | the fit (classifier) | classifier |
| `device_` | the native fit decision | always |
| `best_iteration_`, `n_iter_` | `_record_fit`, from the metric history when there was one | always |
| `feature_importances_` | cached native `feature_importance` | always |
| `booster_` | the one `Booster` around the handle | always |
| `objective_` | **new**, `inspection.objective_of` -> the model's objective code | always |
| `feature_name_` | **new**, `inspection.feature_name_of` -> `Booster.feature_name()` | always |
| `n_features_` | **new**, `inspection.n_features_of` -> `Booster.num_feature()` | always |
| `evals_result_`, `best_score_`, `stopped_early_` | the validation run | only with an `eval_set` |
| `categorical_feature_` | the fit | when categoricals were declared or inferred |

`best_score_` is deliberately absent without an `eval_set`, and
`inspection.best_score_of` raises with that sentence rather than inventing
a number. The three new ones are properties and are **not** in
`_FITTED_ATTRS`: their source is the model, so `load()` and unpickling
answer them without anything having to be kept in step. On an unfitted
estimator they raise `NotFittedError`, which subclasses `AttributeError`,
so `hasattr` is `False` as scikit-learn expects.

---

## 11. Serialization and public-API effects

**Serialization: none.** No model state was added, moved, or renamed. The
three new attributes read the existing serialized model; `__getstate__` /
`__setstate__` and the model file format are untouched. A model saved
before this lane loads after it and answers the new attributes.

**Public API, additive:**

- 7 new names in `__all__` (`cv`, `CVBooster`, `explain_device_choice`,
  `dump_model`, `trees_to_dataframe`, `trees_to_records`,
  `get_split_value_histogram`).
- 4 submodules reachable as attributes.
- `device=None` keyword on four predict methods; the default is exactly the
  established behavior.
- 4 new `Booster` methods, 3 new estimator properties.
- `_eval.registry_source()`.

**Public API, restrictive, and the two places to look if something breaks:**

- `train(feval=...)` and `train(callbacks=...)` now raise `ValueError`
  where they raised `TypeError`. Both were always errors.
- `mojoboost.cv` is now the function. Code doing `import mojoboost.cv as m`
  and then `m.cv(...)` breaks. `from mojoboost.cv import cv` does not, and
  that is the form `python/tests/parallel/test_cv.py` already uses, so no
  test in the tree is expected to move. `cv.py`'s own module docstring
  already stated this outcome before the export existed.
- A fit that asks for an unavailable GPU now raises
  `DeviceUnavailableError` instead of a bare `RuntimeError`. It subclasses
  `RuntimeError` and carries the same message text, so `except
  RuntimeError` and message assertions still pass; `type(exc) is
  RuntimeError` does not.

---

## 12. Risks

1. **`from mojoboost import *` now imports `inspection` and
   `device_selection`,** because their re-exported names are in `__all__`
   and `__getattr__` resolves them. Neither imports anything optional at
   module scope (`inspection` imports `struct` and `_arrays`;
   `device_selection` imports `json`), so this is a cost and not a
   dependency. `import mojoboost` alone imports neither.
2. **The `mojoboost.cv` collision is real and silent in one direction.**
   Documented in three places (module docstring, the import comment,
   `_public_api_plan.NAME_COLLISIONS`) and it still deserves the rename.
3. **`objective_` is not cached.** `objective_code` is registered now
   (`bindings/_mojoboost.mojo` line 348) and `inspection.objective_of`
   prefers it, so the `model_to_string()` round trip only remains against
   an `.so` built before that registration, which is what is on disk here.
   Either way reading it per row would be slow. The docstring says so;
   nothing in the package reads it in a loop.
4. **`_predict_batch` trusts the batch entry point's return value** for the
   backend that ran and falls back to the requested name when it returns
   `None`. If a future entry point returns something other than a device
   name, the value is only used as a return value here and is not consumed
   by any caller in this file yet.
5. **`_NativeTable` has never executed.** The four `registry_*` hooks are
   registered now (`bindings/_mojoboost.mojo` lines 352 to 361), so the
   snapshot path is selected the first time the extension is rebuilt, and
   it has still never run. That is the risk, because it goes from
   unreachable to live on a rebuild rather than on a code change anyone
   reviews. It is guarded by
   try/except at every step and falls back whole, so the failure mode is a
   silent return to the mirror rather than a break. Step 9 of section 13 is
   the check that it took.
6. **The `_gpu_unsupported` backstops assume `_resolve_device` ran first.**
   Each of the four call sites refuses only what its own `device` argument
   still says, and that argument is the resolved answer, not the raw
   parameter. A future fit path that calls a guard before resolving, or
   that resolves a workload missing the flag the guard covers, would see
   `"gpu"` for a request the native policy would have refused for a
   different reason and report the wrong one. The guards are ordered
   directly after their `_resolve_device` call today in all four places.
7. **Concurrent-lane churn.** `cv.py`, `dask.py`, `_arrays.py`,
   `inspection.py`, and `bindings/_mojoboost.mojo` all changed while this
   lane read them, and `python/mojoboost/__init__.py` itself took writes
   from the forced-splits lane (`import json as _json`,
   `_forced_splits_text`, a `contri_addr` params field, and
   `min_gain_to_split` / `monotone_penalty` alias resolution) while this
   lane was editing it. Nothing of that was reverted. The reads that matter
   were repeated at the end (`cv.py`'s module-scope imports,
   `_arrays.SparseBuffers.params()`'s keys, the batch entry point
   signatures, the native `_iteration_slice` semantics, and the
   `def_function` block) and still hold.
8. **This lane's work was committed by another lane, not by this one.**
   This lane ran no `git add` and no `git commit`. A concurrent
   tree-wide snapshot swept the working tree anyway, so
   `python/mojoboost/__init__.py` went in with `f6ae025` and part of this
   handoff with `9c1e771`. `git status` therefore shows those files clean,
   which reads as "nothing changed here" and is wrong. The changes are in
   history, mixed into commits whose messages describe other lanes, so a
   later reviewer looking for this lane by commit will not find it. Read
   the diff of those two commits against `e28a24d` for the Python surface
   rather than trusting the commit subjects.

---

## 13. Focused validation, in order. **Every command below is UNRUN.**

Run one at a time and stop at the first failure.

1. Rebuild the extension. The one on disk predates the batch entry points,
   the four `registry_*` hooks, `objective_code`, and `decide_device`, so
   until it is rebuilt `_eval` runs the mirror, `objective_` pays the
   `model_to_string()` round trip, and `device_selection` reports
   `CONTRACT_NARROW`. Three of the changes in this lane only take effect
   after this step:
   `pixi run build-python`
2. The cheapest end-to-end check that the surface imports and stays cheap:
   `pixi run -e pytest python -c "import mojoboost, sys; print(sorted(mojoboost.__all__)); print('dask' in sys.modules, 'pandas' in sys.modules, 'sklearn' in sys.modules)"`
   Expect the 26 names, and `False False False`.
3. The file most exposed to the `cv` collision:
   `pixi run -e pytest pytest -q python/tests/parallel/test_cv.py`
   It reaches the module as `from mojoboost.cv import CVBooster, cv`,
   which is the form the shadowing leaves working, so it is expected to
   pass unchanged. It is first because it is the one that would prove
   otherwise.
4. The fitted attributes, which gained three:
   `pixi run -e pytest pytest -q python/tests/test_attributes.py`
5. The prediction paths, which gained a keyword:
   `pixi run -e pytest pytest -q python/tests/test_predict.py`
6. `pixi run -e pytest pytest -q python/tests/parallel/test_inspection.py`
7. `pixi run -e pytest pytest -q python/tests/test_sklearn_integration.py`
8. The no-numpy path, because the list branches of the changed predict
   methods are only exercised there:
   `pixi run -e pytest pytest -q python/tests/test_no_numpy.py`
9. The two things step 1 switches on, which nothing before this asserts.
   6a landed while this lane was open, so both are expected to have
   changed:
   `pixi run -e pytest python -c "import mojoboost._eval as e; print(e.registry_source())"`
   Expect `native`, not `mirror`.
   `pixi run -e pytest python -c "import mojoboost.device_selection as d; print(d.native_contract())"`
   Expect `full`, not `narrow`. That is the one that proves the fit
   workloads reach the native policy rather than
   `_NarrowNativePolicy`'s shape-only answer.
   Then re-run 3 through 8, because step 1 changed which code path they
   take.

Not a validation step, but the check that costs nothing:
`git diff --check -- python/mojoboost/__init__.py python/mojoboost/basic.py python/mojoboost/_eval.py python/mojoboost/_public_api_plan.py handoffs/connect_07_python_public.md`
was run and is clean, and no line in the four Python files exceeds 79
columns.
