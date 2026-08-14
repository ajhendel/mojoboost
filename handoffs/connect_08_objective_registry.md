# Handoff, connect task 08: the native objective and metric registry, everywhere

Lane 08. Owned files, and the only files edited:

- `src/mojoboost/objective_registry.mojo`
- `src/mojoboost/objective.mojo`
- `src/mojoboost/metrics.mojo`
- `src/mojoboost/custom_metric.mojo`
- this handoff

Nothing was committed by this lane. Note for whoever reads `git log`:
concurrent lanes committed the shared worktree three times mid-session
(`dc21f03`, `860b1cf`, `e6f3959`) and swept this lane's in-progress edits
into those commits, so the four owned source files show as clean rather than
modified. The content is intact and was verified after each sweep; the commit
authorship is not this lane's doing and no commit was made from here. This
handoff is the only file left untracked.

Baseline for every line count below is `20e0fcc`, the last commit that
predates this lane's first edit.

| File | Before | After |
| --- | --- | --- |
| `objective_registry.mojo` | 1165 | 1698 |
| `metrics.mojo` | 600 | 751 |
| `custom_metric.mojo` | 1574 | 2283 |
| `objective.mojo` | 321 | 498 |

`git diff --check` is clean.

**Nothing was run.** No `mojo`, no `pixi`, no build, no test, no formatter.
Every claim below is from reading. Section 10 lists the smallest commands
that would check them, all marked UNRUN.

---

## 1. Implementations found

### 1.1 The registry itself

`src/mojoboost/objective_registry.mojo` (lane 21's output,
`handoffs/migration_21_objective_metric_registry.md`) was complete,
carefully documented, and **entirely dead**. Nothing imported it:

```
rg -n "objective_registry" --glob '!handoffs/**' --glob '!*.txt' .
```

returned six hits, all of them prose: one comment in `device_policy.mojo`
and five comments in `python/mojoboost/_eval.py`. It is not in
`src/mojoboost/__init__.mojo` either, so it was not even reachable as
`mojoboost.objective_registry`. Not one of its ninety-odd query functions
had a caller.

### 1.2 The facts it duplicated, by where they actually lived

| Fact | Registry | Live copies found |
| --- | --- | --- |
| inverse link per objective | `objective_link` | `Booster.response` (boosting.mojo), `response_scale` (custom_metric.mojo), `response_for_objective` (gpu_predict.mojo) |
| metric codes 0..20 | `comptime METRIC_*` | `comptime _METRIC_*` in `bindings/_mojoboost.mojo`, `_METRICS` in `python/mojoboost/_eval.py` |
| metric code -> computation | none | `eval_metric`'s 21-branch chain in `bindings/_mojoboost.mojo` |
| metric direction | `metric_higher_is_better` | `_METRICS` in `_eval.py`; nothing native |
| metric transform | `metric_transform` | hand-written branch in `eval_metric` |
| objective name/alias | `objective_code_from_name` | `objective_from_name` in params.mojo, `_OBJECTIVES` in `python/mojoboost/__init__.py` |
| parameter default | `objective_default_param` | `objective_default_alpha` in params.mojo, `_OBJECTIVE_PARAM` in Python |
| parameter domain | none | `_check_objective` in boosting.mojo only |
| GPU eligibility | `objective_backends`, `objective_gradients_on_device` | `supports_device_objective` (gpu_objectives_native.mojo), `gpu_trains_objective` (device_policy.mojo), `GPU_OBJECTIVES` (device_selection.py) |
| grad/hess source | none | implicit in which trainer you call |
| initialization rule | none | `_base_score` (boosting.mojo), `_multiclass_base_scores` (custom_metric.mojo), literal `0.0` (ranking.mojo) |
| class-weight applicability | none | nothing; LightGBM's silent-ignore behavior was inherited |
| objective/metric compatibility | none | `_eval.resolve` in Python only |

The four rows with an empty Registry column are what this lane added. The
rest were already stated in the registry and stated again elsewhere.

### 1.3 The metric implementations

All twenty-one built-in metric codes name a function that genuinely exists,
so nothing registered here is a placeholder:

- nineteen in `metrics.mojo`: `l2`, `rmse`, `l1`, `quantile_loss`,
  `huber_loss`, `mape`, `fair_loss`, `poisson_loss`, `gamma_loss`,
  `gamma_deviance`, `tweedie_loss`, `cross_entropy_loss`,
  `kullback_leibler`, `binary_log_loss`, `binary_error`, `binary_auc`,
  `average_precision`, `multiclass_log_loss`, `multiclass_error`;
- two in `ranking.mojo`: `ndcg`, `mean_average_precision`.

All fourteen objective codes name a real trainer path: eleven built-in
single-output through `fill_grad_hess`, `MULTICLASS` through
`train_multiclass`, `LAMBDARANK` through `train_ranker`, `CUSTOM` through
`train_custom`. Nothing was registered that is not implemented, and nothing
implemented was left unregistered.

---

## 2. Call path, before and after

### 2.1 Built-in metric, native caller

Before:

```
caller writes a MetricSetFn closure by hand
  -> knows the metric's direction by memory      (CustomMetric(name, higher_is_better))
  -> knows the objective's link by memory        (response_scale, or forgets)
  -> calls metrics.l2 / ranking.ndcg directly
  -> train_with_metrics
```

Nothing checked that the direction matched the metric, that the metric
suited the objective's task, or that the transform was applied. A wrong
direction produces a run that finishes and truncates to the *worst* round.

After:

```
resolve_builtin_metrics(names, objective)      -> registry: alias table + task compatibility
BuiltinMetricContext.single_output(objective)  -> carries objective, alpha, weights, groups, n_classes
ctx.check(codes, n_valid)                      -> registry: metric_is_builtin, check_objective_metric,
                                                  metric_scoring_param, metric_needs
train_with_builtin_metrics(...)
  -> builtin_metric_metadata(codes)            -> registry: metric_canonical_name, metric_higher_is_better
  -> MetricSuite(metadata, dispatch, primary)
  -> train_with_metrics  (unchanged loop)
       -> eval_builtin_metric(code, ctx, v, raw, y)
            -> registry: metric_transform
            -> response_scale -> registry: objective_link
            -> metrics.eval_metric_by_code | metrics.multiclass_* | ranking.ndcg / map
```

The registry's answer now reaches early stopping (`higher_is_better` decides
the comparison in `_StopState.observe`), reaches the transform, and reaches
refusal (an incompatible pair raises before the first tree).

### 2.2 Metric code to computation

Before: one hand-written 21-branch chain, in `bindings/_mojoboost.mojo`, in
the Python-binding layer, unreachable from Mojo.

After: `metrics.eval_metric_by_code` for the nineteen single-output codes,
and `custom_metric.eval_builtin_metric` above it for the transform plus the
two multiclass and two ranking shapes. Both are native. The binding's chain
is now a duplicate of a native call path; the exact patch that deletes it is
section 6.4.

### 2.3 Inverse link

Before: `objective_link` (dead) plus three live copies that agreed by
inspection.

After: `custom_metric.response_scale` reads `objective_link`, so one of the
three copies is gone. `Booster.response` and
`gpu_predict.response_for_objective` are still their own copies; the exact
patches are section 6.5, and they belong to lanes that own those files.

### 2.4 Custom objective

Before: `objective.mojo` stated the `CUSTOM` contract in prose and nothing
checked it against the registry, which stated the same contract in data.

After: `_check_custom_contract()` runs six registry queries at the top of
`train_custom` and `train_custom_with_valid` and raises if any disagrees with
what the file does. `matching_base_score` reads the other direction: it uses
the registry to refuse the three objectives with no single base score and
then delegates to `boosting._base_score`.

---

## 3. Connections completed

Each of these is real: registry state reaches the implementation, the
registry's output changes behavior, and an unsupported case raises.

1. **`metrics.eval_metric_by_code`** (new). The code-to-call map for the
   nineteen single-output metrics, native. Raises by name for the four codes
   it does not compute, naming the entry point that does.
2. **`custom_metric.eval_builtin_metric`** (new). Routes on
   `metric_transform`, applies `objective_link` through `response_scale`,
   validates the parameter through `metric_scoring_param`, and calls the one
   function that computes the metric. Three shapes, no code list maintained
   locally.
3. **`custom_metric.BuiltinMetricContext`** (new). Carries exactly what
   `metric_needs` says a metric needs: objective + alpha, class count,
   per-validation-set query groups + cutoff, per-validation-set weights.
   `check` validates the whole request against the registry before training.
4. **`custom_metric.builtin_metric_metadata`** (new). Registry name and
   registry direction into `CustomMetric`. This is the connection with the
   largest behavioral consequence: `higher_is_better` decides which way
   `_StopState.observe` compares and therefore which round the ensemble is
   truncated to.
5. **`custom_metric.resolve_builtin_metrics`** (new). Names to codes through
   `metric_code_for_objective`, so every LightGBM alias resolves and a metric
   for the wrong task is refused by name rather than scored.
6. **`custom_metric.default_metric_codes`** (new). LightGBM's "score the
   objective's own loss" rule, via `objective_default_metric`. Raises for
   `CUSTOM`, which has no default.
7. **Four `*_with_builtin_metrics` entry points** (new):
   `train_with_builtin_metrics`, `train_custom_with_builtin_metrics`,
   `train_multiclass_with_builtin_metrics`,
   `train_ranker_with_builtin_metrics`. Each takes metric *codes* plus a
   context and builds the dispatching closure itself. The objective is
   **not** a separate argument on the single-output and custom paths: it
   comes from the context, so the metric provably scores the objective the
   model trained with, at the parameter it trained at. The multiclass and
   ranker paths check the context's objective and (multiclass) its class
   count against the run and refuse a mismatch.
8. **`custom_metric.response_scale` reads `objective_link`.** One of the
   three link copies fused. It now raises for `LINK_SOFTMAX` instead of
   silently returning raw scores; see section 9.2 for why that is a
   behavior change and why it is the right one.
9. **`objective._check_custom_contract`** (new). Six registry facts asserted
   once per custom-objective training run, each paired with the line of
   `objective.mojo` that depends on it.
10. **`objective.matching_base_score`** (new). Registry decides which
    objectives have a single base score to match; `boosting._base_score`
    computes it. No second implementation of any initialization rule.
11. **`objective.check_custom_base_score`** (new), wired into both custom
    trainers. `INIT_CALLER` means the framework picks nothing, so a
    non-finite `base_score` is the one thing worth checking, and checking it
    stops the trainer from handing a NaN to a callback and then blaming the
    callback.
12. **`objective.custom_objective_backends`** (new). `objective_backends`
    rather than a local claim.
13. **`objective.apply_sample_weight`** (new public name for the existing
    `_apply_sample_weight`). Documents the single-application rule; see
    section 5.

New registry facts, all consumed by at least one of the above:

- `objective_grad_hess_source` (`GRAD_BUILTIN` / `GRAD_SOFTMAX` /
  `GRAD_LAMBDARANK` / `GRAD_CALLBACK`) — consumed by
  `_check_custom_contract`, and the query lane 03 needs (section 6.2).
- `objective_init_kind` (`INIT_LINK_MEAN` / `INIT_LABEL_PERCENTILE` /
  `INIT_LOG_CLASS_PRIOR` / `INIT_ZERO` / `INIT_CALLER`) — consumed by
  `_check_custom_contract` and `matching_base_score`. Composed from
  `objective_renews_leaves` and `objective_task`, not restated.
- `objective_class_weight_kind`, `objective_supports_scale_pos_weight`,
  `check_objective_class_weight` — consumed by `_check_custom_contract`;
  offered to lane 07 (section 6.3).
- `ParamDomain`, `objective_param_domain` — consumed by
  `check_objective_param` and `metric_scoring_param`.
- `objective_supports_metric`, `check_objective_metric`,
  `metric_code_for_objective`, `metric_scoring_param`,
  `objective_has_default_metric` — consumed by `ctx.check`,
  `resolve_builtin_metrics`, `eval_builtin_metric`, `objective_spec`.
- `check_objective_backend`, `backend_name` — offered to lanes 03 and 05.
- `ObjectiveSpec` gained `grad_source`, `init_kind`, `class_weight_kind`,
  `default_metric` (-1 for `CUSTOM`), which is what makes the one-record
  binding marshalling of section 6.4 possible without five extra calls.

---

## 4. Duplicates fused or quarantined

### 4.1 Fused (deleted, inside owned files)

| What | Where | Replaced by |
| --- | --- | --- |
| the objective-code link table in `response_scale` | `custom_metric.mojo` | `objective_link` |
| `BINARY_LOGISTIC`, `CROSS_ENTROPY`, `GAMMA`, `POISSON`, `TWEEDIE` imports | `custom_metric.mojo` | nothing — they existed only for that table |
| `comptime METRIC_L2 .. METRIC_MAP`, `N_BUILTIN_METRICS` | `objective_registry.mojo` | imported from `metrics.mojo` and re-exported |

The third row is a *move*, not a deletion, and it deserves the argument.
A metric code names a function, and nineteen of the twenty-one functions are
in `metrics.mojo`, so the numbering belongs with them. The registry imports
and re-exports them, so `objective_registry.METRIC_L2` still resolves and
every value is unchanged — lane 21's export patch (its section 3.1) still
works verbatim.

The move was forced as well as right. `metrics.mojo` **cannot** import
`objective_registry`: `boosting.mojo` imports `_argsort` from `metrics.mojo`
and the registry imports `boosting.mojo`, so the three would form a cycle.
Defining the codes in the leaf and importing upward is the only direction
that works. The dependency edges now read:

```
metrics  <-  boosting  <-  objective_registry  <-  objective  <-  custom_metric
   ^-------------------------|                        ^              ^
   (registry imports metrics directly too)            |              |
                                            custom_metric imports all of them
```

No cycle. This is stated in both module docstrings so the next person does
not try to "fix" it.

### 4.2 Quarantined (documented, not deleted, because the file is another lane's)

| Duplicate | File | Patch |
| --- | --- | --- |
| `comptime _METRIC_*` (21 lines) | `bindings/_mojoboost.mojo` | 6.4 |
| `eval_metric`'s 21-branch dispatch + hand-written transform | `bindings/_mojoboost.mojo` | 6.4 |
| `Booster.response`'s link chain | `boosting.mojo` | 6.5 |
| `response_for_objective`'s link chain | `gpu_predict.mojo` | 6.5 |
| `supports_device_objective` | `gpu_objectives_native.mojo` | 6.2 |
| `objective_from_name`, `objective_display_name`, `objective_default_alpha`, `_alpha_key_for`, `_raise_if_unimplemented_objective`, `comptime MULTICLASS` | `params.mojo` | lane 21 section 3.2, unchanged |
| `comptime LAMBDARANK` | `ranking.mojo` | lane 21 section 3.3, unchanged |
| `_METRICS`, `_ALIASES`, `_DEFAULTS`, `_TASK_DEFAULTS`, `_CompatTable` | `python/mojoboost/_eval.py` | 6.3 |
| `_OBJECTIVES`, `_OBJECTIVE_PARAM`, `_UNIMPLEMENTED_OBJECTIVES` | `python/mojoboost/__init__.py` | 6.3 |
| `GPU_OBJECTIVES`, `CPU_ONLY_OBJECTIVES` | `python/mojoboost/device_selection.py` | 6.3 |

### 4.3 Mirrors still inside the registry

Unchanged from lane 21's handoff, and still labelled in the module
docstring: `comptime MULTICLASS`, `comptime LAMBDARANK`, and
`objective_gradients_on_device`. Each is a duplicate of a file this lane does
not own and each is removed by a patch listed above. One mirror was added:
`objective_param_domain` restates the four intervals `_check_objective`
enforces. That one is **loud rather than silent** — `check_objective_param`
runs both and raises a "registry drift" error naming both sides if they ever
disagree. It is one comparison, once per training run.

---

## 5. Weighting: how the pieces compose, and why nothing is applied twice

The task asked that sample weights, class weights, `scale_pos_weight`,
`is_unbalance`, per-class weights, custom objectives, custom metrics, and
multiple metrics compose without duplicate application. What is true today,
by reading:

**Training weights.** Every class-level policy collapses to one
`sample_weight` vector before any trainer is called. `class_weight.mojo` is
the only expander: `class_weight_rows` multiplies the caller's row weight by
the class's weight (`combined = w * class_weights[label[r]]`),
`balanced_sample_weight` and `unbalanced_sample_weight` are thin wrappers
over it, and `scale_pos_weight_rows` is `class_weight_rows` with a two-entry
weight vector. Nothing downstream can tell the product from a weight passed
by hand, which is what makes double application impossible rather than
merely discouraged. `check_class_balance_params` already rejects
`is_unbalance` together with `scale_pos_weight`.

The gap this lane closed is the *other* direction: nothing refused a class
weighting argument on an objective with no classes. LightGBM silently
ignores `scale_pos_weight=4.0` on a poisson model.
`check_objective_class_weight` in the registry names it instead, keyed on
`objective_class_weight_kind`, which is keyed on the task. It is offered to
lane 07 (section 6.3); no native caller has one yet, because no native entry
point takes a `class_weight` argument — the expansion is the caller's, by
design.

**Custom-objective weights.** `objective.apply_sample_weight` (the public
name of `_apply_sample_weight`) is the one and only application point, called
once per round by `train_custom`, `train_custom_with_valid`,
`train_custom_with_metrics`, and `train_custom_gpu`. The callback never sees
the weights, so it cannot apply them a second time. That contract is now
written on the function rather than only in the module docstring.

**Validation-metric weights.** Deliberately *not* the training weights.
`BuiltinMetricContext.weights` holds one vector per validation set, supplied
explicitly, and the training `sample_weight` is never reused — which matches
`train_with_valid`'s existing contract and means a class weight folded into
the training weights cannot leak into a validation score. The two ranking
metrics take no weights at all, matching `ranking.mojo`'s signatures and
LightGBM. `metrics.check_metric_weight` remains the single validator.

**Multiple metrics.** Each metric is scored once per validation set per
round by `_eval_round`, and `ctx.check` now rejects the same code named
twice, which would otherwise put two identical columns in the history and
two identical entries in the early-stopping state.

---

## 6. Cross-lane patch requests, exact

### 6.1 Task 01 (`src/mojoboost/__init__.mojo`) — blocks everything else

Nothing in this lane is reachable as `mojoboost.*` until this lands. Add,
after the `from .metrics import (...)` block:

```mojo
from .objective_registry import (
    CLASS_WEIGHT_BINARY,
    CLASS_WEIGHT_MULTICLASS,
    CLASS_WEIGHT_NONE,
    GRAD_BUILTIN,
    GRAD_CALLBACK,
    GRAD_LAMBDARANK,
    GRAD_SOFTMAX,
    INIT_CALLER,
    INIT_LABEL_PERCENTILE,
    INIT_LINK_MEAN,
    INIT_LOG_CLASS_PRIOR,
    INIT_ZERO,
    LINK_EXP,
    LINK_IDENTITY,
    LINK_SIGMOID,
    LINK_SOFTMAX,
    NAME_SUPPORTED,
    NAME_UNIMPLEMENTED,
    NAME_UNKNOWN,
    NEEDS_CUTOFF,
    NEEDS_GROUPS,
    NEEDS_NOTHING,
    NEEDS_N_CLASSES,
    NEEDS_PARAM,
    PARAM_ALPHA,
    PARAM_FAIR_C,
    PARAM_NONE,
    PARAM_TWEEDIE_VARIANCE_POWER,
    SUPPORTS_CPU,
    SUPPORTS_GPU,
    TASK_BINARY,
    TASK_MULTICLASS,
    TASK_RANKING,
    TASK_REGRESSION,
    TRANSFORM_OBJECTIVE_LINK,
    TRANSFORM_RAW,
    TRANSFORM_SOFTMAX,
    MetricSpec,
    ObjectiveSpec,
    ParamDomain,
    all_objective_codes,
    backend_name,
    check_objective_backend,
    check_objective_class_weight,
    check_objective_metric,
    check_objective_param,
    metric_alias_names,
    metric_canonical_name,
    metric_code_for_objective,
    metric_code_from_name,
    metric_codes_for_task,
    metric_higher_is_better,
    metric_is_builtin,
    metric_names_for_task,
    metric_needs,
    metric_scoring_param,
    metric_spec,
    metric_task,
    metric_transform,
    objective_alias_names,
    objective_backends,
    objective_canonical_name,
    objective_class_weight_kind,
    objective_code_from_name,
    objective_default_metric,
    objective_default_param,
    objective_grad_hess_source,
    objective_gradients_on_device,
    objective_has_default_metric,
    objective_init_kind,
    objective_is_builtin,
    objective_is_known,
    objective_is_multi_output,
    objective_link,
    objective_name_status,
    objective_needs_groups,
    objective_param,
    objective_param_domain,
    objective_param_name,
    objective_spec,
    objective_supports_metric,
    objective_supports_scale_pos_weight,
    objective_task,
    objective_unimplemented_canonical,
    objective_unimplemented_reason,
    task_from_name,
    task_name,
    unimplemented_objective_alias_names,
)
```

Three deliberate omissions, each of which is a **duplicate-symbol error** if
added today. The registry re-exports all three, so importing them from here
*and* from their defining module puts the same name in the namespace twice:

- `MULTICLASS` and `LAMBDARANK` — defined twice until lane 21's sections 3.2
  and 3.3 land (`params.mojo`, `ranking.mojo`).
- `objective_renews_leaves` — already exported from `.boosting`.
- the `METRIC_*` codes and `N_BUILTIN_METRICS` — see the next block.

Extend the existing `from .metrics import (...)` block with:

```mojo
    DEFAULT_BINARY_THRESHOLD,
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
    eval_metric_by_code,
```

Import the metric codes from `.metrics` and **not** from
`.objective_registry`: they are the same constants (the registry
re-exports them) and importing both spellings into one namespace is a
duplicate.

Extend `from .objective import (...)` with:

```mojo
    apply_sample_weight,
    check_custom_base_score,
    custom_objective_backends,
    matching_base_score,
```

Extend `from .custom_metric import (...)` with:

```mojo
    BuiltinMetricContext,
    builtin_metric_metadata,
    default_metric_codes,
    eval_builtin_metric,
    resolve_builtin_metrics,
    train_custom_with_builtin_metrics,
    train_multiclass_with_builtin_metrics,
    train_ranker_with_builtin_metrics,
    train_with_builtin_metrics,
```

### 6.2 Task 03 (device objectives, `gpu_objectives_native.mojo`)

Three requests, in order of value.

**(a) Delete `supports_device_objective`'s chain and delegate.** It mirrors
`objective_gradients_on_device` value for value. Replace the body with:

```mojo
from .objective_registry import objective_gradients_on_device

def supports_device_objective(objective: Int) -> Bool:
    return objective_gradients_on_device(objective)
```

The import direction is safe: `objective_registry.mojo` imports only
`boosting.mojo` and `metrics.mojo`, neither of which touches `max.gpu.*`, so
this does not drag the GPU stack into the registry's consumers. It is the
direction lane 21's docstring said the dependency would run after wiring.

**(b) Use `objective_grad_hess_source`, not `objective_is_builtin`, for
round eligibility.** A fused device round needs to know *which* of the four
gradient sources it is facing, not merely whether `fill_grad_hess` handles
it:

| source | fused device round |
| --- | --- |
| `GRAD_BUILTIN` | eligible, gradients device-resident |
| `GRAD_SOFTMAX` | eligible only through the multiclass entry point (`_fill_softmax_grad_hess` has its own kernel) |
| `GRAD_LAMBDARANK` | never; pairwise within a query, no closed form per row |
| `GRAD_CALLBACK` | host gradients, device tree growth only |

`objective_is_builtin` collapses the last three into one "no", which is
why `train_custom_gpu` and `train_multiclass_gpu` had to exist outside the
predicate.

**(c) Refuse rather than fall back silently where a backend genuinely
cannot serve.** `check_objective_backend(objective, SUPPORTS_GPU)` raises a
sentence naming the objective. The one unsupported combination today is
`LAMBDARANK` on the GPU. Note the three predicates are deliberately
different and must not be merged: `objective_backends` (does a trainer
exist), `objective_gradients_on_device` (are the derivatives computed on the
device), `gpu_trains_objective` in `device_policy.mojo` (does `train_gpu`
itself accept the code). `objective_backends`'s docstring says so.

### 6.3 Task 07 (Python public API, `_eval.py`, `__init__.py`)

**(a)** Lane 21's section 3.7 stands unchanged: replace `_CompatTable` with a
`_NativeTable` reading the snapshots of section 6.4, then delete `_METRICS`,
`_ALIASES`, `_DEFAULTS`, `_TASK_DEFAULTS`, `_CompatTable`.

**(b)** `python/mojoboost/__init__.py`: delete `_SQUARED_ERROR.._CROSS_ENTROPY`,
`_OBJECTIVES`, `_OBJECTIVE_PARAM`, `_UNIMPLEMENTED_OBJECTIVES`,
`_unimplemented_objective_note`. All five are in the snapshots.

**(c)** `python/mojoboost/device_selection.py`: delete `GPU_OBJECTIVES` and
`CPU_ONLY_OBJECTIVES`; they are `backends & SUPPORTS_GPU` on the objective
record, and lane 21 recorded them as *wrong* today (its D4).

**(d) New**: the estimators can now refuse a class-weighting argument the
objective has no classes for, which LightGBM ignores silently. The registry
call is `check_objective_class_weight(objective, has_class_weight,
is_unbalance, scale_pos_weight)`, exposed through the section-6.4 binding as
a snapshot field (`class_weight_kind` on the objective record) so Python can
decide without a call per fit. `CLASS_WEIGHT_NONE` = reject `class_weight`,
`is_unbalance`, and any `scale_pos_weight != 1.0`; `CLASS_WEIGHT_BINARY` =
all three allowed; `CLASS_WEIGHT_MULTICLASS` = `class_weight` and
`balanced` allowed, `scale_pos_weight` and `is_unbalance` rejected.

**(e) New**: `parameter domains`. `objective_param_domain` gives
`(applies, lower, lower_open, has_upper, upper, upper_open, default)`, so an
estimator can validate `alpha` / `fair_c` / `tweedie_variance_power` before
the call rather than surfacing a trainer error. Add it to the objective
record in 6.4 rather than restating the four intervals in Python.

### 6.4 Task 14 (binding modules) — the largest single win

**(a) Delete `eval_metric`'s dispatch, do not port it.**
`bindings/_mojoboost.mojo` currently owns the only metric code-to-call map
and the only hand-written transform. Both are now native. The whole function
body collapses to argument marshalling plus one call:

```mojo
from mojoboost.custom_metric import BuiltinMetricContext, eval_builtin_metric

def eval_metric(code: PythonObject, params: PythonObject) raises -> PythonObject:
    var kind = Int(py=code)
    var nr = Int(py=params["n_rows"])
    var objective = Int(py=params["objective"])
    var ctx = BuiltinMetricContext(
        objective,
        Float64(py=params["alpha"]),
        Int(py=params["n_classes"]),
        _groups_list(params),          # List[RankGroups], one entry, or empty
        Int(py=params["ndcg_at"]),
        _weights_list(params, nr),     # List[List[Float64]], one entry, or empty
    )
    var pred = _f64_list(Int(py=params["pred_addr"]), _pred_len(kind, nr, ctx))
    var target = _f64_list(Int(py=params["y_addr"]), nr)
    return PythonObject(eval_builtin_metric(kind, ctx, 0, pred, target))
```

That deletes the twenty-one `comptime _METRIC_*` lines, the twenty-one
dispatch branches, the by-hand `response_scale` call, and the by-hand softmax
loop. `_pred_len` is `nr * ctx.n_classes` when
`metric_transform(kind) == TRANSFORM_SOFTMAX` and `nr` otherwise — one
registry call, not a code list. **Note the behavior change**: the native path
enforces `check_objective_metric`, so a metric that does not score the
model's task now raises instead of returning a number. That is section 9.1.

Because lane 06 owns `bindings/_mojoboost.mojo`, this is a patch for lane 06
to apply, and lane 14 should produce it as part of its import-and-export
patch rather than reimplementing dispatch in a new binding module.

**(b) `bindings/objective_bindings.mojo`, the five snapshot functions.**
Lane 21's section 4 specified these; the record is now wider. Exact tuples:

```
registry_objectives() -> tuple of, per code in all_objective_codes():
    (code, canonical_name, task_name, link, param_name, default_param,
     param_applies, param_lower, param_lower_open, param_has_upper,
     param_upper, param_upper_open,
     renews_leaves, multi_output, needs_groups, gradients_on_device,
     backends, builtin, grad_source, init_kind, class_weight_kind,
     default_metric)          # default_metric is -1 for CUSTOM

registry_objective_aliases()       -> tuple of (alias, code) over objective_alias_names()
registry_objective_unimplemented() -> tuple of (alias, canonical, reason)
registry_metrics()                 -> tuple of, per code 0..N_BUILTIN_METRICS-1:
    (code, canonical_name, task_name, higher_is_better, needs, transform)
registry_metric_aliases()          -> tuple of (alias, code) over metric_alias_names()
```

Everything after `builtin` on the objective record is `objective_spec`'s new
fields plus `objective_param_domain`'s; nothing needs a sixth function.
Conventions from lane 21 still hold: strings cross as `str`, and an unknown
name is *absent* from the alias tuple rather than a sentinel code.

**(c)** Do not add a metric-computation binding beyond (a). There is exactly
one native code-to-call map and it should stay that way.

### 6.5 The remaining two link copies (lanes owning `boosting.mojo` and `gpu_predict.mojo`)

Not this lane's files, and not lane 03's or 07's either. Recording them here
because `custom_metric.response_scale` has now moved and leaving the other
two behind is the whole risk of a partial migration.

```mojo
# boosting.mojo, Booster.response
var link = objective_link(self.objective)
if link == LINK_SIGMOID:
    return _sigmoid(raw)
if link == LINK_EXP:
    return exp(raw)
return raw
```

**This creates a cycle and must not be applied as written.**
`objective_registry.mojo` imports `boosting.mojo`, so `boosting.mojo` cannot
import the registry. Two ways out, for whoever owns the sequencing (lane 22
is the natural home for the decision):

1. Move the objective code constants (`SQUARED_ERROR .. CROSS_ENTROPY`,
   `DEFAULT_FAIR_C`, `DEFAULT_TWEEDIE_VARIANCE_POWER`) and
   `objective_renews_leaves` out of `boosting.mojo` into
   `objective_registry.mojo`, reversing that one edge. The registry would
   then import nothing from `boosting.mojo` except `_check_objective`, which
   is the only remaining reason for the edge, and `check_objective_param`
   could enforce `objective_param_domain` directly instead of delegating.
   This is the clean end state and it deletes the drift check in
   `check_objective_param`.
2. Leave `Booster.response` as it is and document it as the third copy.

`gpu_predict.response_for_objective` has no such problem — `gpu_predict.mojo`
does not sit under the registry — and becomes a two-line map from `LINK_*` to
its `RESPONSE_*` codes. Land it with
`tests/parallel/test_gpu_predict.mojo` and
`tests/test_backend_equivalence.mojo` green: it changes no value, it moves a
branch.

---

## 7. Remaining disconnections

Honest list of what is still not wired, and why.

1. **The registry is still not exported.** Until section 6.1 lands, none of
   this is reachable as `mojoboost.*`, and the only live consumers are the
   three owned modules that import it directly. This is the single blocking
   item.
2. **`metrics.mojo` does not import the registry and never will.** The cycle
   through `boosting._argsort` forbids it. This is stated as a fact rather
   than a gap: `metrics.mojo` is arithmetic and the registry connection lives
   one level up, in `custom_metric.eval_builtin_metric`. If a future change
   genuinely needs registry facts inside `metrics.mojo`, the patch is to move
   `_argsort` into a leaf module (`sampling.mojo` or a new `sortutil.mojo`)
   and repoint `boosting.mojo`'s import. Not done here: `boosting.mojo` is
   not this lane's file and the move buys nothing today.
3. **No native caller of `check_objective_class_weight`.** No native entry
   point takes a `class_weight` argument — expansion is the caller's job, by
   `class_weight.mojo`'s design — so the check has nowhere native to sit. It
   is for lane 07 (6.3d).
4. **No native caller of `check_objective_backend`.** For lanes 03 and 05.
5. **`objective_param_domain` is a second statement of `_check_objective`'s
   ranges.** Loud (the drift check) but still second. 6.5 option 1 removes
   it.
6. **The three registry mirrors from lane 21 are untouched**: `MULTICLASS`,
   `LAMBDARANK`, `objective_gradients_on_device`. Their deletions are lane
   21's sections 3.2, 3.3, 3.4 and are not this lane's to apply.
7. **`train_with_valid` and friends still infer no metric.** The built-in
   metric path is a new set of entry points beside them, not a replacement;
   see section 8.

---

## 8. Fallbacks preserved

Nothing was replaced. Every established path is untouched and still the
default:

- `train_with_metrics`, `train_custom_with_metrics`,
  `train_multiclass_with_metrics`, `train_ranker_with_metrics`,
  `train_with_metric`, `train_with_callbacks`, `fit_with_metrics`,
  `fit_with_callbacks`, `fit_multiclass_with_metrics`,
  `fit_ranker_with_metrics` — all unchanged, all still take a caller-supplied
  `MetricSuite`. The four `*_with_builtin_metrics` entry points are wrappers
  that build a suite and call them.
- `train_with_valid` / `train_custom_with_valid` — unchanged loop; the latter
  gained two validations at the top (contract assertion, finite base score).
- `train_custom` / `train_custom_with_valid` still accept the same
  arguments; `base_score` still defaults to 0.0.
- `metrics.mojo`'s twenty-three metric functions are byte-for-byte
  unchanged. `eval_metric_by_code` is a new function beside them.
- The binding's `eval_metric` is untouched by this lane and still works.
- A caller who wants a built-in metric *and* a hand-written one in one run
  writes a `MetricSuite` whose evaluator calls `eval_builtin_metric` for some
  indices and their own code for others. The two paths are not exclusive and
  that is documented in the module docstring.

The one place a strategy switch would have been the wrong tool is
`response_scale`: keeping a "use the registry or the old table" flag there
would have preserved exactly the drift the change exists to remove.

---

## 9. Risks and behavior changes

### 9.1 `check_objective_metric` refuses pairs that used to be scored

The native path now enforces "the metric's task must equal the objective's
task", which is `_eval.resolve`'s rule in Python and is stricter than
LightGBM. Nothing enforced it in Mojo before. Consequences:

- No effect on any existing native caller: `MetricSuite` metrics are
  caller-supplied and are not checked (they cannot be — a caller-supplied
  metric has no task).
- If the 6.4a binding patch lands, `eval_metric` starts refusing e.g.
  `binary_logloss` on a poisson model. Python already refuses that in
  `_eval.resolve`, so no supported Python path reaches it; a direct
  `_mojoboost.eval_metric` call would.

Rated low, but it is a real narrowing and should not land silently.

### 9.2 `response_scale` now raises for a multiclass objective

Before, `response_scale(MULTICLASS, raw)` fell through to identity. Now it
raises, because a softmax is not a per-row map and returning raw scores under
the name "response scale" is a wrong answer rather than a missing one.

Who could hit it:

- `bindings/_mojoboost.mojo:eval_metric` calls
  `response_scale(objective, raw)` on the single-output path only; the two
  multiclass metrics return before reaching it. A caller who asked for a
  *single-output* metric on a *multiclass* model would have got identity and
  now gets a raise — which is the same pair section 9.1 refuses anyway.
- `tests/test_custom_metrics.mojo` uses `response_scale` with
  `BINARY_LOGISTIC` and `SQUARED_ERROR` only (lines 444, 447). Unaffected.
- No other caller found (`rg "response_scale"`).

`response_scale`'s signature also gained an explicit `raises`. Mojo `def` is
implicitly raising, so this is documentation rather than a signature change,
but it is worth knowing if a caller is a `fn`.

### 9.3 The metric codes moved file

`objective_registry.METRIC_L2` now resolves through a re-export rather than a
local `comptime`. The values are identical and lane 21's export patch still
works, but **whether Mojo re-exports an imported `comptime` as a module
member is the one thing here that reading cannot settle.** The registry's
own docstring already assumed it for `objective_renews_leaves`, so the
assumption predates this lane. If it turns out not to hold, the fix is one
line per code in `objective_registry.mojo`
(`comptime METRIC_L2 = metrics.METRIC_L2`) and nothing else changes. This is
the first thing section 10's compile check would catch.

### 9.4 Cost

Every registry call added is once per training run or once per (metric,
validation set, round). Nothing was added to a per-row or per-node loop.
`eval_builtin_metric` allocates one transformed prediction vector per call,
which is what the binding's `response_scale` already did; the multiclass path
allocates a second (the softmax copy) so the caller's raw scores stay raw.
`_check_custom_contract` is six integer compares.

### 9.5 Unverified

Nothing here was compiled or run. In particular: the closure capture syntax
in the four `*_with_builtin_metrics` entry points
(`raises {imm codes, imm context} -> Float64`) follows
`custom_metric.train_with_metric` and `bindings/_mojoboost.mojo:469`, but a
capture of a `struct` value rather than a `List` is new in this file. The
`@staticmethod def name() -> StructName` form follows `BaggingParams.disabled`
and `GossParams.disabled`. `mut self` follows `_StopState.observe`. Bitwise
`&` against `==` is parenthesized everywhere rather than relying on Python
precedence.

---

## 10. Validation, smallest first — ALL UNRUN

Nothing below was executed. Run them in order; each is cheap and each
isolates one class of failure.

**UNRUN 1 — does it compile at all.** The only check that settles 9.3.

```
pixi run mojo build src/mojoboost/objective_registry.mojo -o /dev/null
```

**UNRUN 2 — does the package compile with the new import edges.** Catches a
cycle, which is the one structural risk introduced.

```
pixi run mojo build src/mojoboost/custom_metric.mojo -o /dev/null
```

**UNRUN 3 — one focused existing test, not the suite.** `custom_metric.mojo`
changed most and `response_scale` changed semantics:

```
pixi run mojo test tests/test_custom_metrics.mojo
```

**UNRUN 4 — the custom-objective contract assertions.**

```
pixi run mojo test tests/test_custom_objective.mojo
```

**UNRUN 5 — a registry self-check probe** (write it, do not add it to the
suite yet). Three properties, in this order:

1. round trip: every name in `objective_alias_names()` resolves, and every
   code in `all_objective_codes()` names itself;
2. domain agreement: for each of `QUANTILE`, `HUBER`, `FAIR`, `TWEEDIE`,
   sample the boundary and a value just inside and just outside, and confirm
   `check_objective_param` and `objective_param_domain.contains` agree — this
   is the drift check firing on purpose;
3. metric coverage: for every code `0 .. N_BUILTIN_METRICS - 1`,
   `eval_builtin_metric` returns a finite number on a two-row toy input with
   a compatible objective, and raises on an incompatible one.

**UNRUN 6 — differential, after 6.4a lands.** For each metric code and a
fixed toy model, the binding's old `eval_metric` and the new one must return
bit-identical values for every *compatible* pair. The incompatible pairs are
where they are allowed to differ (9.1), and that difference should be
enumerated rather than diffed away.

Do **not** run the full suite, a build loop, or a benchmark as part of
checking this lane.
