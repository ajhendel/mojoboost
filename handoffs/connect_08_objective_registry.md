# Handoff, connect task 08: the native objective and metric registry, everywhere

Lane 08. Owned files:

- `src/mojoboost/objective_registry.mojo`
- `src/mojoboost/objective.mojo`
- `src/mojoboost/metrics.mojo`
- `src/mojoboost/custom_metric.mojo`
- this handoff

**Three files outside that list were also edited, in a second pass, at the
repository owner's explicit direction** after the first pass filed them as
cross-lane patch requests. They are `src/mojoboost/__init__.mojo` (6.1),
`src/mojoboost/boosting.mojo` and `src/mojoboost/gpu_predict.mojo` (6.5), and
nothing else. Each edit is exactly the patch this handoff already specified,
applied rather than requested; every section below that described one as
pending has been rewritten to say what landed. No other lane's file was
touched, nothing was reverted, reformatted, or cleaned, and no commit was made
from here.

Note for whoever reads `git log`: concurrent lanes committed the shared
worktree several times mid-session (`dc21f03`, `860b1cf`, `e6f3959` among
them) and swept this lane's in-progress edits into those commits, so most of
these files show as clean rather than modified. The content is intact and was
verified after each sweep; the commit authorship is not this lane's doing.

Baseline for every line count below is `20e0fcc`, the last commit that
predates this lane's first edit.

| File | Before | After | Owned |
| --- | --- | --- | --- |
| `objective_registry.mojo` | 1165 | 1811 | yes |
| `metrics.mojo` | 600 | 751 | yes |
| `custom_metric.mojo` | 1574 | 2283 | yes |
| `objective.mojo` | 321 | 498 | yes |
| `__init__.mojo` | 331 | +140 lines, all additions | no — 6.1 |
| `boosting.mojo` | 1871 | see 6.5 | no — 6.5 |
| `gpu_predict.mojo` | 1341 | see 6.5 | no — 6.5 |

The last two carry concurrent edits from other lanes, so a line delta would
not be this lane's; 6.5 lists the changes by name instead.

`git diff --check` is clean.

**Nothing was run.** No `mojo`, no `pixi`, no build, no test, no formatter.
Every claim below is from reading. Section 10 lists the smallest commands
that would check them, all marked UNRUN.

Method note, so nobody has to guess how the counting claims were reached: the
symbol inventories in this handoff (which names exist in which module, the
454-export duplicate check in 6.1, the collision check against the pre-existing
exports) came from throwaway `python3` scripts that regex-parse the source
text — the same job as `rg`, over the same bytes. They never imported the
package, executed project code, invoked `mojo`, or built anything. If that
still reads as "ran Python" to you, treat every count in this document as
unverified and re-derive it with `rg`; nothing in the source tree depends on
those scripts, and none of them were kept.

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
| `comptime METRIC_L2 .. METRIC_MAP`, `N_BUILTIN_METRICS` | `objective_registry.mojo` | defined in `metrics.mojo`, imported and re-declared here |
| the objective-code link table in `Booster.response` | `boosting.mojo` | `objective_link` (6.5) |
| the objective-code link table in `response_for_objective` | `gpu_predict.mojo` | `objective_link` (6.5) |
| `_check_objective`'s four parameter-range branches | `boosting.mojo` | `check_objective_param` (6.5) |
| `objective_renews_leaves`'s body | `boosting.mojo` | delegates to the registry (6.5) |

The third row is a *move*, not a deletion, and it deserves the argument.
A metric code names a function, and nineteen of the twenty-one functions are
in `metrics.mojo`, so the numbering belongs with them. The registry imports
them under `_`-prefixed names and re-declares each as
`comptime METRIC_L2 = _METRIC_L2`, so `objective_registry.METRIC_L2` still
resolves, is defined in the module that exports it, and carries an unchanged
value — lane 21's export patch (its section 3.1) still works verbatim. That
re-declaration is `device.mojo`'s pattern over `device_policy.mojo`, adopted
for the reason `device.mojo` gives: the symbols a module has always exported
should be defined in it, whatever an importer's view of a re-exported name
turns out to be. It also retires the open question this handoff used to
carry as 9.3.

The move was forced as well as right. `metrics.mojo` **cannot** import
`objective_registry`: `boosting.mojo` imports `_argsort` from `metrics.mojo`,
so anything `boosting.mojo` sits above must stay above `metrics.mojo` too.

The last four rows are the second pass. The objective codes moved the other
way — *up*, out of `boosting.mojo` into the registry, re-declared back into
`boosting.mojo` by the same pattern — which reverses the one edge that used to
force the registry to import `boosting.mojo`. With that edge gone the whole
graph is a line:

```
metrics  <-  objective_registry  <-  boosting  <-  objective  <-  custom_metric
                    ^                    ^
                    |                    +-- gpu_predict, and everything else
                    +-- imported directly by all four
```

No cycle, and no module below `boosting.mojo` needs to know it exists. This
is stated in the module docstrings so the next person does not try to "fix"
it.

### 4.2 Quarantined (documented, not deleted, because the file is another lane's)

| Duplicate | File | Patch |
| --- | --- | --- |
| `comptime _METRIC_*` (21 lines) | `bindings/_mojoboost.mojo` | 6.4 |
| `eval_metric`'s 21-branch dispatch + hand-written transform | `bindings/_mojoboost.mojo` | 6.4 |
| `supports_device_objective` | `gpu_objectives_native.mojo` | 6.2 |
| `objective_from_name`, `objective_display_name`, `objective_default_alpha`, `_alpha_key_for`, `_raise_if_unimplemented_objective`, `comptime MULTICLASS` | `params.mojo` | lane 21 section 3.2, unchanged |
| `comptime LAMBDARANK` | `ranking.mojo` | lane 21 section 3.3, unchanged |
| `_METRICS`, `_ALIASES`, `_DEFAULTS`, `_TASK_DEFAULTS`, `_CompatTable` | `python/mojoboost/_eval.py` | 6.3 |
| `_OBJECTIVES`, `_OBJECTIVE_PARAM`, `_UNIMPLEMENTED_OBJECTIVES` | `python/mojoboost/__init__.py` | 6.3 |
| `GPU_OBJECTIVES`, `CPU_ONLY_OBJECTIVES` | `python/mojoboost/device_selection.py` | 6.3 |

### 4.3 Mirrors still inside the registry

Unchanged from lane 21's handoff, and still labelled in the module
docstring: `comptime MULTICLASS` (mirrored in `params.mojo`) and
`comptime LAMBDARANK` (mirrored in `ranking.mojo`). Both are removed by lane
21's sections 3.2 and 3.3, in files this lane does not own.
`objective_gradients_on_device` is the third and is lane 03's (6.2a).

`objective_param_domain` was a fourth mirror in the first pass, restating the
four intervals `_check_objective` enforced, kept honest by a drift check that
ran both. The second pass deleted the mirror instead: `_check_objective` now
calls `check_objective_param`, the domain is stated once, and the drift check
is gone because there is no longer a second side to drift from. The four
error sentences a user sees are unchanged, verbatim — see 6.5.

`comptime SQUARED_ERROR .. CROSS_ENTROPY`, `CUSTOM`, `DEFAULT_FAIR_C`,
`DEFAULT_TWEEDIE_VARIANCE_POWER`, and `objective_renews_leaves` are now
*defined* here and re-declared in `boosting.mojo`, so they are no longer
mirrors in either direction: one definition, one binding, one value.

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

### 6.1 Task 01 (`src/mojoboost/__init__.mojo`) — APPLIED

Nothing in this lane was reachable as `mojoboost.*` without it, so it was
applied here rather than requested (see the note at the top of this handoff).
`+140` lines, every one an addition; no existing export line was changed,
moved, or removed. Task 01 should read this as a done deal rather than a
request, and re-check it only if it is reorganizing the file.

The registry block was inserted after `from .metrics import (...)`, with a
comment above it saying what the surface is and which names are deliberately
absent:

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

Three deliberate omissions, each of which would be a **duplicate-symbol
error**. Every one of these names does reach `mojoboost.*` — just from the
module that defines it, not from the registry as well:

- the objective codes `SQUARED_ERROR .. CROSS_ENTROPY`, `CUSTOM`,
  `DEFAULT_FAIR_C`, `DEFAULT_TWEEDIE_VARIANCE_POWER`, and
  `objective_renews_leaves` — already exported from `.boosting`, which
  re-declares them from the registry (4.1). `MULTICLASS` arrives from
  `.params` and `LAMBDARANK` from `.ranking`, and stays that way until lane
  21's sections 3.2 and 3.3 land.
- the `METRIC_*` codes, `N_BUILTIN_METRICS`, and `DEFAULT_BINARY_THRESHOLD` —
  from `.metrics`; see the next block.
- the alias-name tables `OBJECTIVE_ALIAS_NAMES`, `METRIC_ALIAS_NAMES`,
  `KNOWN_OBJECTIVE_NAMES`, `UNIMPLEMENTED_OBJECTIVE_ALIAS_NAMES`, and the four
  `*_METRIC_NAMES` strings. Not a collision — these are the joined strings the
  `*_names` query functions return, and the functions are the surface. Export
  them only if a caller needs the table itself.

A duplicate check over the finished file (all `from .x import (...)` blocks,
454 names) reports no name twice.

Extend the existing `from .metrics import (...)` block with (applied; the
constants sort above `average_precision`, `eval_metric_by_code` between
`cross_entropy_loss` and `fair_loss`):

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
`.objective_registry`: they are the same constants (the registry re-declares
them) and importing both spellings into one namespace is a duplicate.

One consequence worth knowing before the compatibility snapshot lands:
`tools/api_snapshot.py`'s `mojo_exports_by_module` parses exactly these
blocks, so the new names become part of `compatibility/api_snapshot.json` the
first time it is written. That file does not exist in the tree yet, so
`--check` cannot fail on this today. `tests/parallel/api_snapshot_manifest.json`
does list the old `custom_metric` export set and is now stale, but it says of
itself that "nothing re-runs it", so it is a record rather than a gate.

Extend `from .objective import (...)` with (applied):

```mojo
    apply_sample_weight,
    check_custom_base_score,
    custom_objective_backends,
    matching_base_score,
```

Extend `from .custom_metric import (...)` with (applied; `BuiltinMetricContext`
sorts with the other structs, the rest alphabetically among the functions):

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

### 6.5 The other two link copies (`boosting.mojo`, `gpu_predict.mojo`) — APPLIED

The first pass filed these as requests and flagged the `boosting.mojo` one as
a trap: the registry imported `boosting.mojo`, so `boosting.mojo` could not
import the registry, and the patch as written would have made a cycle. Two
ways out were recorded. **Option 1, the clean one, was taken**, so this
section now describes what is in the tree.

**`objective_registry.mojo` — the edge reversal.** The registry no longer has
a `from .boosting import (...)` line at all; its only import is
`from .metrics import (...)`. What moved into it:

- `comptime SQUARED_ERROR = 0` .. `CROSS_ENTROPY = 12`, plus `CUSTOM = 6`,
  `LAMBDARANK = 7`, `MULTICLASS = -1`, `DEFAULT_FAIR_C = 1.0`, and
  `DEFAULT_TWEEDIE_VARIANCE_POWER = 1.5`. Same values.
- `objective_renews_leaves`, three comparisons, moved body and all.
- `check_objective_param` became self-contained: it asks
  `objective_param_domain` and raises. The four LightGBM-parity sentences
  (`"huber requires alpha > 0"`, `"quantile requires 0 < alpha < 1"`,
  `"fair requires alpha (fair_c) > 0"`,
  `"tweedie requires 1 < alpha (tweedie_variance_power) < 2"`) are reproduced
  verbatim, and a fifth generic branch built from `ParamDomain.describe()`
  covers anything added later. `_POISSON_MAX_DELTA_STEP` stayed in
  `boosting.mojo`: it is a term in a hessian, not a fact about an objective.

**`boosting.mojo` — four changes, all local.**

1. A `from .objective_registry import (...)` block importing the codes under
   `_`-prefixed names plus `LINK_EXP`, `LINK_SIGMOID`, `objective_link`,
   `check_objective_param`, and `objective_renews_leaves as
   _objective_renews_leaves`.
2. Each constant re-declared under the name this module has always exported:
   `comptime SQUARED_ERROR = _SQUARED_ERROR`, and so on for the other
   thirteen. Every existing `from .boosting import SQUARED_ERROR` in the tree
   keeps working and keeps reading the same value; nothing outside this file
   had to change.
3. `objective_renews_leaves` is now `return _objective_renews_leaves(objective)`.
4. `_check_objective`'s four `if objective == X and not (bounds)` branches
   replaced by one `check_objective_param(objective, alpha)`. Same messages,
   same order, one statement of the intervals.

And `Booster.response` is the patch as originally written:

```mojo
var link = objective_link(self.objective)
if link == LINK_SIGMOID:
    return _sigmoid(raw)
if link == LINK_EXP:
    return exp(raw)
return raw
```

`exp` and `_sigmoid` are still imported and still used elsewhere in the file
(the poisson, gamma, tweedie, and logistic gradient fills); nothing became
dead.

**`gpu_predict.mojo` — the easy one.** `response_for_objective` now maps
`LINK_*` to its own `RESPONSE_*` codes and the five objective-code imports it
carried from `.boosting` are gone:

```mojo
var link = objective_link(objective)
if link == LINK_SIGMOID:
    return RESPONSE_SIGMOID
if link == LINK_EXP:
    return RESPONSE_EXP
return RESPONSE_IDENTITY
```

It changes no value; it moves a branch. Its import line is now
`from .objective_registry import (LINK_EXP, LINK_SIGMOID,
metric_canonical_name, objective_link)`.

Note the deliberate asymmetry with `response_scale`: here `LINK_SOFTMAX`
falls through to `RESPONSE_IDENTITY`, because the multiclass GPU path takes
the softmax itself over a row block and asks this function only for the
per-element stage. `response_scale` has no such caller above it and raises
instead (9.2). Both are documented at their definitions.

All four copies of the inverse link — `Booster.response`, `response_scale`,
`response_for_objective`, and the binding's transform (6.4, still
outstanding) — are down to three read from one table and one request.

---

## 7. Remaining disconnections

Honest list of what is still not wired, and why.

The first pass listed seven. Three of them (the missing export, the
`Booster.response` copy, the duplicated parameter domain) were closed by the
second pass and are described in 6.1 and 6.5. What is left:

1. **`metrics.mojo` does not import the registry.** It is the leaf; the
   registry imports *it*. This is a fact rather than a gap: `metrics.mojo` is
   arithmetic, and the registry connection lives one level up in
   `custom_metric.eval_builtin_metric`. `boosting.mojo` also takes `_argsort`
   from it, so the direction is forced as well as chosen. If a future change
   genuinely needs registry facts inside `metrics.mojo`, the patch is to move
   `_argsort` into a leaf module (`sampling.mojo` or a new `sortutil.mojo`)
   and repoint `boosting.mojo`'s import.
2. **No native caller of `check_objective_class_weight`.** No native entry
   point takes a `class_weight` argument — expansion is the caller's job, by
   `class_weight.mojo`'s design — so the check has nowhere native to sit. It
   is for lane 07 (6.3d).
3. **No native caller of `check_objective_backend`.** For lanes 03 and 05.
4. **Two registry mirrors from lane 21 are untouched**: `MULTICLASS` in
   `params.mojo` and `LAMBDARANK` in `ranking.mojo`, both still reaching
   `mojoboost.*` from those modules rather than from the registry. Their
   deletions are lane 21's sections 3.2 and 3.3.
   `objective_gradients_on_device` is the third and is 6.2a.
5. **The binding still carries its own metric codes and its own transform.**
   6.4, the largest single remaining win, and the last copy of the inverse
   link.
6. **The Python layer still carries `_eval.py`'s tables.** 6.3.
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
task", which is stricter than LightGBM. Nothing enforced it in Mojo before.
Audited after the fact, by tracing every caller:

- **Its only caller is `BuiltinMetricContext.check`**, which this lane wrote.
  `rg` finds no other. So the rule narrows nothing that predates the lane; it
  is a new rule on a new surface, not a change to an old one.
- **`MetricSuite` metrics are untouched.** They are caller-supplied and are
  not checked — they cannot be, a caller-supplied metric has no task. Section
  8's fallbacks are all on that path.
- **It agrees with the Python layer name for name.** `_eval.py`'s `_METRICS`
  table assigns each of the twenty-one metrics a task, and the registry's
  `metric_task` assigns the same one in every case, including the two that
  could plausibly have gone the other way: `cross_entropy` and
  `kullback_leibler` are `REGRESSION` in both. Objective side likewise —
  `objective_task` sends `CROSS_ENTROPY` to `TASK_REGRESSION`, matching
  `_eval`'s `xentropy`. So a Python caller sees no change in which pairs are
  accepted, and if 6.4a lands the binding gains the check Python already
  applies above it.

Rated: real narrowing, no reachable regression. It should still not land
silently, which is what this section is for.

### 9.2 `response_scale` now raises for a multiclass objective

Before, `response_scale(MULTICLASS, raw)` fell through to identity. Now it
raises, because a softmax is not a per-row map and returning raw scores under
the name "response scale" is a wrong answer rather than a missing one.

Every caller traced, and the answer is that **no reachable path can reach the
raise**:

- `bindings/_mojoboost.mojo:eval_metric` calls `response_scale(objective,
  raw)` on the single-output path only; the `_METRIC_MULTI_LOGLOSS` /
  `_METRIC_MULTI_ERROR` branch returns above it. And the `objective` it
  passes cannot be `MULTICLASS`: `python/mojoboost/basic.py:Booster.eval`
  builds that params dict and already substitutes `_SQUARED_ERROR` whenever
  `task == _eval.MULTICLASS` or the Booster has no config, precisely because
  a multiclass model's link is not a per-row map. The substitution predates
  this lane. A caller reaching `_mojoboost.eval_metric` directly, bypassing
  `basic.py`, could pass `MULTICLASS` and would now get a raise instead of
  silent identity — which is the same pair 9.1 refuses anyway.
- `custom_metric.eval_builtin_metric` handles `TRANSFORM_SOFTMAX` and
  `TRANSFORM_RAW` above its `response_scale` call, so the multiclass and
  ranking metrics never reach it.
- `gpu_predict.response_for_objective` deliberately does *not* raise on
  `LINK_SOFTMAX`; see 6.5.
- `tests/test_custom_metrics.mojo` uses `response_scale` with
  `BINARY_LOGISTIC` and `SQUARED_ERROR` only (lines 444, 447). Unaffected.
- No other caller found (`rg "response_scale"` over the whole tree).

`response_scale`'s signature also gained an explicit `raises`. Mojo `def` is
implicitly raising, so this is documentation rather than a signature change,
but it is worth knowing if a caller is a `fn`.

### 9.3 The metric codes moved file — RESOLVED

The first pass left one open question here: `objective_registry.METRIC_L2`
resolved through a re-export rather than a local `comptime`, and **whether
Mojo re-exports an imported `comptime` as a module member is not something
reading a Mojo program can settle.**

It no longer has to be settled, because the registry does not rely on it.
Two pieces of evidence decided the approach:

- `split.soft_threshold_l1` is defined in `gain.mojo`, imported into
  `split.mojo`, and imported *from* `split.mojo` by `__init__.mojo`,
  `tree.mojo`, and the distributed path. So a `def` does re-export.
- `device.mojo` re-declares `device_policy.mojo`'s constants explicitly
  (`comptime CPU_DEVICE = _CPU_DEVICE`) rather than relying on it, and says
  in its own comment why: the symbols a module has always exported should be
  defined in it.

The second is the stronger signal, and it is what the registry now does: the
metric codes are imported under `_`-prefixed names and re-declared, twenty-two
lines of `comptime METRIC_X = _METRIC_X`. `boosting.mojo` does the same for
the fourteen objective constants coming the other way (6.5). Whatever the
answer to the original question, both modules define what they export.

### 9.4 Cost

Every registry call added is once per training run or once per (metric,
validation set, round). Nothing was added to a per-row or per-node loop.
`eval_builtin_metric` allocates one transformed prediction vector per call,
which is what the binding's `response_scale` already did; the multiclass path
allocates a second (the softmax copy) so the caller's raw scores stay raw.
`_check_custom_contract` is six integer compares.

### 9.5 Unverified

Nothing here was compiled or run, in either pass. Every claim below is a
reading of source text against in-repo precedent, and each is a place where a
compiler could disagree.

From the first pass, still unverified:

- the closure capture syntax in the four `*_with_builtin_metrics` entry
  points (`raises {imm codes, imm context} -> Float64`) follows
  `custom_metric.train_with_metric` and `bindings/_mojoboost.mojo:469`, but a
  capture of a `struct` value rather than a `List` is new in this file;
- the `@staticmethod def name() -> StructName` form follows
  `BaggingParams.disabled` and `GossParams.disabled`;
- `mut self` follows `_StopState.observe`;
- bitwise `&` against `==` is parenthesized everywhere rather than relying on
  Python precedence.

From the second pass (the four fixes), newly unverified:

- **the import graph is a line, by reading.** `metrics` imports only
  `std.math`; `objective_registry` imports only `.metrics`; `boosting` imports
  `.objective_registry`; `objective` and `gpu_predict` import `.boosting` and
  `.objective_registry`; `custom_metric` sits above all of them. No file was
  found importing a file below it in that order, but the check was `rg` over
  `from \.` lines, not a compile.
- **the re-declare pattern is assumed to be free.** `comptime X = _X` at
  module scope in `objective_registry.mojo` (22 metric codes) and in
  `boosting.mojo` (14 objective codes plus two link codes) follows
  `device.mojo` exactly. If Mojo rejects aliasing an imported `comptime` under
  a new name at module scope, both blocks fail together and the fix is to
  import the codes unaliased and delete the binding block.
- **454 exports, zero duplicates, by script.** The count and the collision
  check came from a regex parse of `__init__.mojo`'s import blocks (see the
  method note in the header), not from importing the package. A name that
  exists in its defining module but is not actually exportable — a `comptime`
  inside a struct body, say — would pass that check and fail the compile.
  Every added name was grepped to its definition site; none were inside a
  struct.
- **the three deliberate omissions in 6.1** rest on the claim that a name
  reaching `__init__.mojo` from two modules is a duplicate-symbol error. The
  objective codes and the `METRIC_*` codes are each now visible from two
  places (their definition site and the module that re-declares them), so
  each is imported from one only — the module it was always imported from, so
  no public name moved. If Mojo tolerates the duplicate instead of rejecting
  it, the omissions are harmless rather than necessary.
- **`Booster.response` was rewritten to branch on `objective_link`** rather
  than on objective codes. The mapping was checked row by row against the old
  branch set, but it is a behavioral rewrite of a public method and nothing
  executed it.
- **`_check_objective`'s four range branches were replaced by one
  `check_objective_param` call.** The messages the two produce are not
  byte-identical; if any test asserts on the old wording it will fail on the
  string, not on the logic.

---

## 10. Validation, smallest first — ALL UNRUN

Nothing below was executed. Run them in order; each is cheap and each
isolates one class of failure. The order matters more after the second pass
than it did after the first: the import graph was rearranged, so the early
checks are structural and the later ones are behavioral. Stop at the first
failure rather than running the rest.

**UNRUN 1 — the bottom of the graph.** `metrics.mojo` imports only
`std.math`, so this compiles or the metric codes themselves are wrong.

```
pixi run mojo build src/mojoboost/metrics.mojo -o /dev/null
```

**UNRUN 2 — the registry, one edge up.** Its only import is `.metrics`. This
is also the check that settles the re-declare pattern (9.5): 22 lines of
`comptime METRIC_X = _METRIC_X` at module scope either bind or they do not.

```
pixi run mojo build src/mojoboost/objective_registry.mojo -o /dev/null
```

**UNRUN 3 — the reversed edge.** `boosting.mojo` now imports *from* the
registry rather than the registry importing from it, and re-declares the
fourteen objective constants the same way. This is where a cycle would show
up, and it is the one structural risk the second pass introduced.

```
pixi run mojo build src/mojoboost/boosting.mojo -o /dev/null
```

**UNRUN 4 — the two consumers of `objective_link`.** `Booster.response` and
`response_for_objective` were both rewritten to branch on the link rather
than on objective codes (6.5).

```
pixi run mojo build src/mojoboost/gpu_predict.mojo -o /dev/null
```

**UNRUN 5 — the top of the graph.** `custom_metric.mojo` changed most and
sits above everything else.

```
pixi run mojo build src/mojoboost/custom_metric.mojo -o /dev/null
```

**UNRUN 6 — the package surface.** `+140` export lines landed in
`__init__.mojo` (6.1). This catches a name that exists in its defining module
but is not exportable, and any duplicate the regex check in 6.1 missed. It is
last among the compile checks because it is the only one that pulls in every
other lane's files too, so a failure here may not be this lane's.

```
pixi run mojo build src/mojoboost/__init__.mojo -o /dev/null
```

**UNRUN 7 — one focused existing test, not the suite.** `custom_metric.mojo`
changed most and `response_scale` changed semantics:

```
pixi run mojo test tests/test_custom_metrics.mojo
```

**UNRUN 8 — the custom-objective contract assertions.**

```
pixi run mojo test tests/test_custom_objective.mojo
```

**UNRUN 9 — the objective parameter checks.** `_check_objective`'s four range
branches became one `check_objective_param` call (6.5), and the error strings
are not byte-identical to the old ones (9.5).
`test_new_objectives_validate_alpha` is the test that exercises them; reading
it, it asserts only that *something* raised, never on the wording, so the
rewrite should pass it unchanged — which makes this the check that the
registry's domains match the ranges they replaced.

```
pixi run mojo test tests/test_objectives.mojo
```

**UNRUN 10 — a registry self-check probe** (write it, do not add it to the
suite yet). Three properties, in this order:

1. round trip: every name in `objective_alias_names()` resolves, and every
   code in `all_objective_codes()` names itself;
2. domain agreement: for each of `QUANTILE`, `HUBER`, `FAIR`, `TWEEDIE`,
   sample the boundary and a value just inside and just outside, and confirm
   `check_objective_param` and `objective_param_domain.contains` agree. The
   old cross-module drift check is gone (4.3) because there is nothing left to
   drift from; this is the intra-registry version, that the raising path and
   the declarative path describe the same interval;
3. metric coverage: for every code `0 .. N_BUILTIN_METRICS - 1`,
   `eval_builtin_metric` returns a finite number on a two-row toy input with
   a compatible objective, and raises on an incompatible one.

**UNRUN 11 — differential, after 6.4a lands.** For each metric code and a
fixed toy model, the binding's old `eval_metric` and the new one must return
bit-identical values for every *compatible* pair. The incompatible pairs are
where they are allowed to differ (9.1), and that difference should be
enumerated rather than diffed away.

Do **not** run the full suite, a build loop, or a benchmark as part of
checking this lane.
