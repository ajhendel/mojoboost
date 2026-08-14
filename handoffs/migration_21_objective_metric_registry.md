# Handoff, task 21, one native objective and metric registry

Lane 21 of the parallel round. Three files were written and nothing else in
the repository was touched. This lane staged and committed nothing; other
sessions sharing the checkout did commit, which is what the note below is
about.

**Departure from the round's no-execution rule, deliberate and limited.**
Two throwaway probes were compiled and run against the working tree, and
one Python script was run, because handing over an uncompiled Mojo module
and an unverified rewrite of a live Python module is worse than the rule
they break. No repository test was written, run, or added to any pixi task;
no benchmark, no CI, no build of the Python extension. What was run, and
what it proved, is section 7; the sources are the appendix.

| Path | What it is |
| --- | --- |
| `src/mojoboost/objective_registry.mojo` | New. The registry: every non-computational fact about an objective or a built-in metric, as pure query functions over `Int` codes. ~1150 lines, mostly docstrings. |
| `python/mojoboost/_eval.py` | Rewritten as a facade. Every public name, signature, return value, and error string is unchanged. The three tables are unchanged **value for value** and now live in one marked block behind a single indirection, `_TABLE`. |
| `handoffs/migration_21_objective_metric_registry.md` | This file. |

Nothing imports `objective_registry.mojo` yet. It is not exported from
`src/mojoboost/__init__.mojo`, which this lane does not own, so
`tools/check_parity.py` cannot see it and no parity row may cite it until
the export lands (section 3.1).

**Both files now compile and answer correctly.** The registry compiles as
part of the package and passes a full round-trip self-check; every fact it
mirrors was compared against its source and agrees; and the `_eval.py`
rewrite was compared answer for answer, message for message, against the
version it replaced. Section 7 has the results. What is still unverified:
the binding (it does not exist), the wiring edits in section 3 (not
applied), and the estimator test suite (needs a build of the extension).

**Two things about the shared checkout.** First, this lane committed
nothing, but two other sessions ran repository-wide commits while it was
writing: `9a9c8d1` swept an **intermediate** copy of
`src/mojoboost/objective_registry.mojo` (no enumeration section) into
history, and `b04b5f0` then swept the finished files. Read the tree or
`b04b5f0`, never `9a9c8d1`, and expect this file itself to be a commit or
two behind whatever swept it. Second,
findings D1 and D4 below were re-checked against the tree *after* lanes 09
and 20 landed their changes, and both moved; section 6.0 records what
changed and section 9 records the collision with lane 20 that has to be
resolved before either lane's wiring can land.

---

## 1. What the registry holds, and what it deliberately does not

`objective_registry.mojo` answers, for an objective code: its task, its
canonical name and every alias, its inverse link, which scalar parameter it
reads and that parameter's default, whether it renews leaf values, whether
it needs query groups, whether one iteration grows more than one tree,
whether its gradients have a device kernel, which backends can train it, and
which metric is its default loss. For a metric code: its canonical name and
aliases, its task, its direction, what it needs beyond predictions and
labels, and which transform predictions must have been through.

It holds no arithmetic. Gradients stay in `boosting.mojo`, ranking lambdas
in `ranking.mojo`, metric values in `metrics.mojo`, device kernels in
`gpu_objectives_native.mojo`. Two facts are *imported* rather than restated,
so they keep exactly one definition:

- `objective_renews_leaves` is re-exported from `boosting.mojo`. The
  boosting loop reads it; the registry hands back the same function.
- `check_objective_param` calls `boosting._check_objective(objective, [],
  value)`. With no labels that runs exactly the parameter range checks and
  skips every label check, so the ranges and their messages are not copied.

Custom objectives and custom metrics are outside the registry by
construction. `objective_is_builtin` and `metric_is_builtin` are the
boundary; everything they reject is a callable that carries its own name and
direction (`GradHessFn` in `objective.mojo`, `CustomMetric` in
`custom_metric.mojo`). `objective_default_metric(CUSTOM)` raises rather than
guessing, which is the rule `_eval.default_metric` already enforces.

## 2. Compile-time versus runtime representation

**Every query is a plain `def` over `comptime` constants.** No trait, no
vtable, no `Dict`, no stored table, no allocation except in the four
functions that return a `String` or a `List`.

That was chosen over the two obvious alternatives:

- *A trait with one implementation per objective.* Puts an indirect call
  where an integer compare is. The training hot path must never pay that.
- *A `List[ObjectiveSpec]` built once and indexed by code.* Puts a load and
  a bounds check where a compare is, needs global initialization order, and
  the codes are not contiguous (`MULTICLASS` is -1, 7 is `LAMBDARANK`).

With a compile-time-known code, `objective_link(SQUARED_ERROR)` folds to a
constant. With a runtime code it is a few compares, and it is called once
per training run, never per row or per round. `ObjectiveSpec` and
`MetricSpec` exist for callers that want the whole record at once (the
binding, marshalling one dict); they hold scalars only, so building one
allocates nothing and copying one is a memcpy.

The one duplication *inside* the file is deliberate and marked: the
`if` chains resolve names without allocating, and
`OBJECTIVE_ALIAS_NAMES` / `METRIC_ALIAS_NAMES` /
`UNIMPLEMENTED_OBJECTIVE_ALIAS_NAMES` enumerate the same spellings for the
binding to walk once at start-up. Without the enumeration the binding would
have to spell the names a third time. The round trip between the two is the
first check in section 6.

## 3. Required edits outside this lane's ownership

Ordered. Each step leaves the tree working.

### 3.1 Export the module (blocks everything else)

`src/mojoboost/__init__.mojo`, after the `from .metrics import (...)` block:

```mojo
from .objective_registry import (
    LINK_EXP,
    LINK_IDENTITY,
    LINK_SIGMOID,
    LINK_SOFTMAX,
    METRIC_AUC,
    ...  # the METRIC_* codes
    NEEDS_CUTOFF,
    NEEDS_GROUPS,
    NEEDS_N_CLASSES,
    NEEDS_PARAM,
    SUPPORTS_CPU,
    SUPPORTS_GPU,
    TASK_BINARY,
    TASK_MULTICLASS,
    TASK_RANKING,
    TASK_REGRESSION,
    MetricSpec,
    ObjectiveSpec,
    all_objective_codes,
    check_objective_param,
    metric_alias_names,
    metric_canonical_name,
    metric_code_from_name,
    metric_codes_for_task,
    metric_higher_is_better,
    metric_is_builtin,
    metric_names_for_task,
    metric_needs,
    metric_spec,
    metric_task,
    metric_transform,
    objective_alias_names,
    objective_backends,
    objective_canonical_name,
    objective_code_from_name,
    objective_default_metric,
    objective_default_param,
    objective_gradients_on_device,
    objective_is_builtin,
    objective_is_known,
    objective_link,
    objective_name_status,
    objective_needs_groups,
    objective_param,
    objective_param_name,
    objective_spec,
    task_from_name,
    task_name,
    unimplemented_objective_alias_names,
)
```

`MULTICLASS` and `LAMBDARANK` are **not** in that list until 3.2 and 3.3
land: they are defined twice today (see section 5), and importing both
copies into `__init__.mojo` is a duplicate-symbol error.

`objective_renews_leaves` is already exported from `.boosting`; do not
import it from both.

### 3.2 `src/mojoboost/params.mojo` becomes delegation

Delete `objective_from_name`'s alias chain, `_raise_if_unimplemented_objective`
entirely, `objective_display_name`'s chain, `objective_default_alpha`'s body,
`_alpha_key_for` entirely, and the `comptime MULTICLASS = -1` definition.
Replace with:

```mojo
from .objective_registry import (
    LAMBDARANK,
    MULTICLASS,
    NAME_UNIMPLEMENTED,
    NAME_UNKNOWN,
    objective_canonical_name,
    objective_code_from_name,
    objective_default_param,
    objective_name_status,
    objective_param_name,
    objective_unimplemented_canonical,
    objective_unimplemented_reason,
)

comptime PARAM_STRING_OBJECTIVES = String(
    "regression, binary, multiclass, poisson, huber, quantile, mae, gamma,"
    " tweedie, mape, fair, or cross_entropy"
)


def objective_from_name(name: String) raises -> Int:
    var status = objective_name_status(name)
    if status == NAME_UNIMPLEMENTED:
        raise Error(
            "objective '",
            objective_unimplemented_canonical(name),
            "' is not implemented; ",
            objective_unimplemented_reason(name),
        )
    if status == NAME_UNKNOWN:
        raise Error(
            "unknown objective '", name, "'; expected ",
            PARAM_STRING_OBJECTIVES,
        )
    var code = objective_code_from_name(name)
    if code == LAMBDARANK:
        raise Error(
            "objective 'lambdarank' needs query groups, which a parameter"
            " string cannot carry; use train_ranker in the Mojo API"
        )
    if code == CUSTOM:
        raise Error(
            "objective 'custom' needs a gradient callback; use fit_custom in"
            " the Mojo API"
        )
    return code


def objective_display_name(objective: Int) raises -> String:
    return objective_canonical_name(objective)


def objective_default_alpha(objective: Int) -> Float64:
    return objective_default_param(objective)
```

and `_check_alpha_key` reads `objective_param_name(config.objective)` in
place of `_alpha_key_for`.

**Message-preserving.** The unknown-objective sentence keeps the parameter
string's own shorter list (a parameter string cannot carry `lambdarank` or
`custom`), the unimplemented sentences are rebuilt from the same clauses
verbatim, and the two refusals keep their exact text. `objective_display_name`
gains `lambdarank`, `custom`, and `multiclass` instead of raising for the
first two, which no current caller reaches.

**Import direction.** The registry does not import `params.mojo`, precisely
so that this edit does not create a cycle.

### 3.3 `src/mojoboost/ranking.mojo`

Replace `comptime LAMBDARANK = 7` with
`from .objective_registry import LAMBDARANK`. Nothing else changes; the
value is identical. `ranking.mojo` reaches `model.mojo`, which reaches
`train_gpu.mojo`, which is why the registry does not import *from* ranking.

### 3.4 `src/mojoboost/gpu_objectives_native.mojo`

**Conflicts with lane 20's handoff; read section 9 first.** Both lanes
specify the same one-line replacement of `supports_device_objective`'s
chain, pointing at two different new functions. Under section 9's proposed
resolution it becomes:

```mojo
from .objective_registry import objective_gradients_on_device


def supports_device_objective(objective: Int) -> Bool:
    return objective_gradients_on_device(objective)
```

Value-for-value identical to the chain it replaces (verified by reading:
both are true for exactly the eleven built-in single-output codes), and
identical to lane 20's `gpu_objective_is_device_resident` except for its
`OBJECTIVE_UNSPECIFIED` sentinel, which is device-policy vocabulary and
does not belong in the objective registry. The registry does not import
this module, because it pulls `max.gpu.*` and `params.mojo` and the CLI
must not acquire that dependency.

### 3.5 `src/mojoboost/boosting.mojo`, `custom_metric.mojo`, `gpu_predict.mojo`

The link is decided independently in three places today
(`Booster.response`, `response_scale`, `response_for_objective`). They
agree. Point all three at `objective_link` so they cannot stop agreeing:

```mojo
# boosting.mojo, Booster.response
var link = objective_link(self.objective)
if link == LINK_SIGMOID:
    return _sigmoid(raw)
if link == LINK_EXP:
    return exp(raw)
return raw
```

`custom_metric.response_scale` takes the same shape. `gpu_predict.response_for_objective`
becomes a two-line map from `LINK_*` to `RESPONSE_*`. **Careful**: this is
the only edit in this handoff that touches a prediction path. It changes no
value; it moves a branch. Land it with `tests/test_backend_equivalence.mojo`
and `tests/parallel/test_gpu_predict.mojo` green.

### 3.6 `bindings/_mojoboost.mojo`

Delete the twenty-one `comptime _METRIC_* = ...` lines and import the
`METRIC_*` codes from the registry. Then add the five registry snapshot
functions of section 4 and register them in the `def_function` block.

`eval_metric`'s dispatch chain stays as it is: it maps a code to a *call*,
which is the one thing the registry does not do.

### 3.7 `python/mojoboost/_eval.py` (owned by this lane, wired later)

Replace `_TABLE = _CompatTable()` with `_TABLE = _NativeTable()`, add
`_NativeTable` reading the snapshots from section 4, and delete the marked
compatibility block: the `_METRICS`, `_ALIASES`, `_DEFAULTS`, and
`_TASK_DEFAULTS` dicts and `_CompatTable`. Nothing below the facade banner
changes. The module-level metric-code constants (`L2` ... `MAP`) stay, built
from the snapshot, because `basic.py` reads `_eval.NDCG` and `_eval.MAP`.

### 3.8 `python/mojoboost/__init__.py`

Delete `_SQUARED_ERROR` ... `_CROSS_ENTROPY` (lines 311-323),
`_UNIMPLEMENTED_OBJECTIVES` and `_unimplemented_objective_note` (325-367),
`MojoBoostRegressor._OBJECTIVES` (2257-2277), and
`MojoBoostRegressor._OBJECTIVE_PARAM` (2282-2287). `_objective_code` becomes
a call to the native resolver plus the estimator's own message; the four
range checks in `_objective_param` (2352-2361) become one
`check_objective_param` call across the boundary, or stay as a fast local
pre-check with the native call as the authority. `_metric_objective`
(1325-1340) keeps its shape but reads `objective_link` instead of assuming.

### 3.9 `src/mojoboost/device_policy.mojo` (**lane 20's file, section 9**)

Superseded while this lane was writing. `GPU_OBJECTIVES` and
`CPU_ONLY_OBJECTIVES` in `python/mojoboost/device_selection.py` are already
gone; lane 20 moved the gate into `device_policy.mojo`, keyed on the
objective code. Nothing is left to do in `device_selection.py`.

What remains is native: `device_policy.is_builtin_objective`,
`gpu_objective_is_device_resident`, and the `comptime LAMBDARANK = 7`
mirror at `device_policy.mojo:147` restate what the registry holds. See
section 9.

### 3.10 Documentation (**coordinate with lane 09**)

`docs/LIGHTGBM_PARITY.md:445-448` names `python/mojoboost/_eval.py` as "the
table of names, aliases, directions, and tasks". After wiring that sentence
must name `src/mojoboost/objective_registry.mojo`. Findings D1, D2, D4, and
D6 are documentation defects that exist *today*, independent of this lane.

## 4. The minimal binding API

Five functions, all pure data, all called **once** at `_eval` import and
cached in a module-level dict. Not one call per lookup: `resolve()` runs
inside `fit`, and a cached snapshot also makes the differential check in
section 6 a one-liner.

A cached snapshot is not a second table. It is derived at import from the
registry, never edited, and its contents are unreachable from any code that
could disagree with it.

```mojo
def registry_objectives() raises -> PythonObject
```
A tuple of one tuple per code in `all_objective_codes()`:
`(code, canonical_name, task_name, link, param_name, default_param,
renews_leaves, multi_output, needs_groups, gradients_on_device, backends,
builtin, default_metric_code)`. `default_metric_code` is `-1` for `CUSTOM`,
which is the one objective with no default; every other field comes from
`objective_spec` plus `objective_canonical_name` and `task_name`.

```mojo
def registry_objective_aliases() raises -> PythonObject
```
A tuple of `(alias, code)` over `objective_alias_names()`. Replaces
`MojoBoostRegressor._OBJECTIVES`.

```mojo
def registry_objective_unimplemented() raises -> PythonObject
```
A tuple of `(alias, canonical, reason)` over
`unimplemented_objective_alias_names()`. Replaces
`_UNIMPLEMENTED_OBJECTIVES`. Python keeps its own *sentence* ("use
MojoBoostClassifier, which derives the task from y") and takes only the
*reason* from here, because which estimator to redirect to is a Python fact
and the reason is not.

```mojo
def registry_metrics() raises -> PythonObject
```
A tuple of one tuple per code `0 .. N_BUILTIN_METRICS - 1`:
`(code, canonical_name, task_name, higher_is_better, needs, transform)`.
Replaces `_METRICS`.

```mojo
def registry_metric_aliases() raises -> PythonObject
```
A tuple of `(alias, code)` over `metric_alias_names()`. Replaces
`_ALIASES`.

No sixth function is needed. `_DEFAULTS` is `default_metric_code` on the
objective record; `task_metrics(task)` is a filter over the metric records;
`_OBJECTIVE_PARAM` is `(param_name, default_param)` on the objective record;
`GPU_OBJECTIVES` is `backends & SUPPORTS_GPU`.

Two conventions to fix before writing them: strings cross as `str`, not
bytes; and an unknown name is *absent from the alias tuple*, never a
sentinel code, so Python's "unknown objective" message stays Python's.

## 5. Code deletable after wiring

| File | What goes | Why it exists now |
| --- | --- | --- |
| `python/mojoboost/_eval.py` | `_METRICS`, `_ALIASES`, `_DEFAULTS`, `_TASK_DEFAULTS`, `_CompatTable` (~155 lines, one contiguous block) | No binding yet. Already isolated behind `_TABLE`. |
| `python/mojoboost/__init__.py` | `_SQUARED_ERROR`..`_CROSS_ENTROPY`, `_UNIMPLEMENTED_OBJECTIVES`, `_unimplemented_objective_note`, `_OBJECTIVES`, `_OBJECTIVE_PARAM` (~70 lines) | Second copy in a second language. |
| `python/mojoboost/device_selection.py` | `GPU_OBJECTIVES`, `CPU_ONLY_OBJECTIVES` (~18 lines) | Third copy, and wrong (D4). |
| `src/mojoboost/params.mojo` | `objective_from_name`'s chain, `_raise_if_unimplemented_objective`, `objective_display_name`'s chain, `objective_default_alpha`'s body, `_alpha_key_for`, `comptime MULTICLASS` (~110 lines) | Was the only home before this module. |
| `src/mojoboost/ranking.mojo` | `comptime LAMBDARANK = 7` (1 line) | Same. |
| `src/mojoboost/gpu_objectives_native.mojo` | `supports_device_objective`'s chain (~14 lines) | Same. |
| `bindings/_mojoboost.mojo` | `comptime _METRIC_*` (21 lines) | Same. |
| `src/mojoboost/objective_registry.mojo` | nothing | It is the destination. |

Three mirrors live in the registry *until* 3.2, 3.3, and 3.4 land, and are
labelled as such in the module docstring: `MULTICLASS`, `LAMBDARANK`, and
`objective_gradients_on_device`. Until then the mirrored files are
authoritative in fact and the registry is authoritative by intent. That is
the whole risk this lane carries: a change to `params.mojo`'s alias table
between now and 3.2 would be silently lost.

## 6. Semantic disagreements found

Found by reading, not by running. Each one names the sources and which is
right. **None was resolved silently.** D3 and D8 would change behavior to
fix and D7 would change messages, so those three need a decision rather
than a patch; D4 turned into a collision with lane 20 (section 9).

### 6.0 Two findings moved under concurrent lanes

The tree changed while this lane was reading it. Both findings were
re-checked against the current working tree:

- **D1 is fixed already.** Lane 09 rewrote the `fit(eval_metric=)` row
  (now `docs/LIGHTGBM_PARITY.md:221`) to `supported`, naming all
  twenty-one metrics. Kept below for the record and because the *residual*
  half of it, line 512, is still live.
- **D4 changed shape entirely.** Lane 20 deleted
  `device_selection.GPU_OBJECTIVES` and moved the gate into
  `src/mojoboost/device_policy.mojo`. The Python-versus-Mojo disagreement
  is gone; a Mojo-versus-Mojo one took its place. Rewritten below, and
  section 9 is the coordination it needs.

Line numbers elsewhere in this section were read before those landings and
may have shifted by a few lines in the files lanes 09 and 20 touched
(`docs/LIGHTGBM_PARITY.md`, `python/mojoboost/device_selection.py`,
`src/mojoboost/params.mojo`, `src/mojoboost/boosting.mojo`). The quoted
text is what to search for.

### D1. The parity contract contradicted itself about metric names (fixed)

Was: `docs/LIGHTGBM_PARITY.md:166` (`fit(eval_metric=)`) said "LightGBM's
built-in metric **names** ... are not accepted as strings yet ... a v1
gap", while section 9 said "Every name above is selectable from Python as
`eval_metric="auc"`." The code agreed with section 9: `_metric_spec`
(`__init__.py:431`) calls `_eval.resolve` for any `str`.

Lane 09 has since rewritten that row. **What is still live**:
`docs/LIGHTGBM_PARITY.md:221` and `:512` both name
`python/mojoboost/_eval.py` as the place the names, aliases, directions,
and tasks live. After wiring, that is
`src/mojoboost/objective_registry.mojo`, and `_eval.py` is where a Python
callable becomes a metric spec. **Fix: lines 221 and 512, with 3.10.**

### D2. The README lists eleven of the twenty-one metric names

`README.md:1519-1523` names `l2, rmse, l1, quantile, huber, binary_logloss,
binary_error, auc, multi_logloss, multi_error, ndcg`. Missing: `mape`,
`fair`, `poisson`, `gamma`, `gamma_deviance`, `tweedie`, `cross_entropy`,
`kullback_leibler`, `average_precision`, `map`, all of which `_eval`
accepts and `metrics.mojo` computes. `README.md:143-148` lists them
correctly, so the README also disagrees with itself. **Fix: the 1519 list.**

### D3. Seven objective aliases train but have no default metric

`MojoBoostRegressor._OBJECTIVES` accepts nineteen spellings;
`_eval._DEFAULTS` keys twelve. `MojoBoostRegressor(objective="mse")` trains,
but `fit(eval_set=...)` without an explicit `eval_metric` raises "no default
eval_metric for objective 'mse'". The seven: `regression_l2`, `l2`,
`mean_squared_error`, `mse`, `l1`, `mean_absolute_error`,
`mean_absolute_percentage_error`.

Cause: `_DEFAULTS` is keyed on **the name the user typed**;
`objective_default_metric` in the registry is keyed on **the code**, so it
has no hole. Wiring closes it and turns a `ValueError` into a working fit.
**That is a behavior change and needs sign-off.** This lane changed nothing
(the prompt forbids changing defaults here); the hole is documented in
`_eval.py`'s `_DEFAULTS` comment and in the registry's
`objective_default_metric` docstring.

### D4. Four native copies of "which objectives can the GPU train"

Found first as a Python-versus-Mojo defect:
`device_selection.GPU_OBJECTIVES` was
`{regression, mae, regression_l1, huber, quantile, poisson, binary,
multiclass}`, five objectives short of what
`supports_device_objective` covers (`gamma`, `tweedie`, `mape`, `fair`,
`cross_entropy`), and keyed on estimator spellings without aliases, so
`objective="mse"` fell outside it while `objective="regression"` (the same objective) did not. Both made `device="auto"` quietly pick the CPU for a
workload the GPU covers. **Lane 20 has deleted that set**, and the
alias-blindness with it, by moving the gate native and keying it on the
objective code.

What is left is a native duplication, now fourfold. The membership test for
"is this one of the eleven built-in single-output objectives" is written
out, identically, in:

1. `boosting._check_objective` (the trainer's own check)
2. `gpu_objectives_native.supports_device_objective`
3. `device_policy.is_builtin_objective` (lane 20, new)
4. `objective_registry.objective_is_builtin` (this lane, new)

and `LAMBDARANK = 7` is now declared three times (`ranking.mojo`,
`device_policy.mojo:147`, `objective_registry.mojo`). Two lanes solved the
same problem in the same round without seeing each other. **Fix: section 9.**

### D5. The inverse link is decided in three places (agreement verified)

`Booster.response` (`boosting.mojo:825`), `response_scale`
(`custom_metric.mojo:376`), `response_for_objective`
(`gpu_predict.mojo:438`). All three test the objective code themselves.
They agree, and that is now measured rather than assumed: appendix B checks
all three against `objective_link` for every single-output code. Nothing
enforces it going forward: an objective added with a
link would have to be added to three chains, and a model whose prediction
path and metric path disagreed about the link would report a loss on the
wrong scale without any error. **Fix: 3.5.**

### D6. Two wrong statements about which objectives have a link

`python/mojoboost/__init__.py:126-128`: "`raw_score` ... changes nothing for
the regressor's objectives or for the ranker, which have no link." Poisson,
gamma, and tweedie have the `exp` link and cross entropy the logistic, and
all four are regressor objectives; `predict` on a poisson model returns
`exp(raw)` and `raw_score=True` does not.

`docs/LIGHTGBM_PARITY.md:172`: "The objectives without a link (squared
error, huber, quantile, L1) predict raw either way" is correct in direction but
incomplete: MAPE and FAIR are also identity-linked.

`objective_link` is the answer to both. **Fix: both sentences.**

### D7. Two canonical names for the L1 objective

`params.objective_display_name(L1)` returns `mae`;
`lgbm_model_io.lgbm_objective_name(L1)` writes `regression_l1`, which is
LightGBM's own canonical spelling and what a LightGBM reader expects. Both
spellings resolve back to `L1`, so nothing round-trips wrong, and error
messages say `mae`. The registry **preserves `mae`** so that 3.2 changes no
message, and says so in `objective_canonical_name`'s docstring. Choosing
`regression_l1` instead is a one-line change plus every message that quotes
it. **Decision deferred, deliberately.**

### D8. A file-loaded `Booster` scores `fair` and `tweedie` at alpha 0.9

`basic.py:974`: `"alpha": float(0.9 if self._config is None else
self._config.alpha)`. A `Booster` read from a model file has no config, so
`booster.eval(data, "valid", metric="fair")` scores the fair loss at
`fair_c=0.9` (the default is 1.0) and `metric="tweedie"` raises
"tweedie variance_power must be in (1, 2)" from inside the metric.

`objective_default_param` (and `params.objective_default_alpha`, which it
mirrors) gives the per-objective default. The metric-side fallback in Python
does not use it. A file-loaded booster does not know its objective either,
which is the deeper reason; at minimum the fallback should be
`objective_default_param(objective_code)` once an objective is known, and
the two-metrics-that-cannot-work case should say so rather than scoring a
number nobody asked for. **Fix: `basic.py`, after wiring.**

### D9. Python conflates "not implemented" with "wrong estimator"

`_UNIMPLEMENTED_OBJECTIVES` (`__init__.py:329`) holds both
`cross_entropy_lambda` / `multiclassova` / `rank_xendcg` (genuinely not
implemented, and matching `params.mojo`) and `binary` / `multiclass` /
`softmax` / `lambdarank` (implemented, reached through a different
estimator). Both come out as "X is not available here". Two of the shared
entries also word the reason differently from `params.mojo`: Python's
`ova`/`ovr` says "use MojoBoostClassifier, which trains a shared softmax
model", Mojo's says one-vs-rest needs an independent binary model per class.

The registry separates the two: `objective_name_status` says whether a name
is implemented, `objective_task` says which estimator owns it, and
`objective_unimplemented_reason` gives one reason text for both languages.
**Fix: 3.8.**

### D10. Two spellings of "the multiclass objective" reach the binding

`params.MULTICLASS` is `-1`, outside `boosting.mojo`'s code space. But
`_metric_objective` (`__init__.py:1333`) and `basic.py:983` both pass
`_SQUARED_ERROR` as the objective for a multiclass model's metric, because
the binding softmaxes the raw rows itself and the objective code is unread
on that path. It works, and it reads as if a multiclass model were trained
with squared error. `objective_link(MULTICLASS) == LINK_SOFTMAX` plus
`metric_transform(MULTI_LOGLOSS) == TRANSFORM_SOFTMAX` says the true thing,
and makes the stand-in unnecessary. **Low priority; no defect today.**

### D11. Three name functions, three different coverages

`params.objective_display_name` raises for `LAMBDARANK` and `CUSTOM`;
`lgbm_model_io.lgbm_objective_name` covers both (`CUSTOM` with its own
message) but has no `MULTICLASS`; the registry's
`objective_canonical_name` covers all fourteen codes. Not a defect, but it
is why 3.2's delegation *widens* what `objective_display_name` accepts.

### D12. Not a disagreement, worth stating

`regression` is the l2 **metric** in `metric_code_from_name` and the
squared-error **objective** in `objective_code_from_name`; likewise
`binary`, `multiclass`, `softmax`, and `lambdarank` name both an objective
and a metric, with different referents. That is LightGBM's own overloading,
it is preserved exactly, and it is the reason the registry keeps two
resolvers rather than one.

## 7. Validation, in the order it should run

7.1, 7.2, and the differential in 7.3 **have been run and pass**; their
sources are in the appendix, ready to be promoted into repository tests.
Everything else, the pytest suites in 7.3 and all of 7.4 and 7.5, needs a
build of the Python extension, which this lane did not do. It was left
undone on purpose: twenty other lanes are mutating this checkout, so a
failure in the estimator suite today would say more about them than about
this change, and the differential below already pins `_eval` exactly.

### 7.1 Does it compile (RUN, passes)

```
pixi run mojo run -I src <appendix A>
    registry probe ok: 24 objective names, 39 metric names, 14 codes
```

Compiling `mojoboost.objective_registry` compiles `src/mojoboost/__init__.mojo`
with it, so this also shows the new module introduces no package-level name
collision even though it declares `MULTICLASS` and `LAMBDARANK`, which the
package already exports from `params.mojo` and `ranking.mojo`.

Five constructs were uncertain when the file was written and are now
settled, since they compiled: the bare `try` / `except` inside a
non-`raises` `def`; returning a `comptime String` through `.copy()`;
`String(token)` over `names.split()` in a non-`raises` `def`; `[]`
inferring `List[Float64]` for `_check_objective`'s `target`; and
`@always_inline` on a `def` returning `Bool`.

### 7.2 Focused Mojo test (RUN as two probes, passes)

Appendix A and appendix B are the two probes, and every assertion below
holds. Promote them verbatim into
`tests/parallel/test_objective_registry.mojo`, wired into the `test` task
in `pixi.toml`, converting the `raise Error` checks into
`std.testing.assert_*`:

1. **Round trip.** For every `n` in `objective_alias_names()`,
   `objective_code_from_name(n)` succeeds; for every code in
   `all_objective_codes()`,
   `objective_code_from_name(objective_canonical_name(code)) == code`. Same
   two for metrics over `metric_alias_names()` and
   `0 .. N_BUILTIN_METRICS - 1`.
2. **Mirrors agree, which is what authorizes the deletions in section 3.**
   Verified over all fourteen codes:
   `objective_gradients_on_device == supports_device_objective` (3.4),
   `objective_default_param == params.objective_default_alpha` (3.2),
   `objective_canonical_name == params.objective_display_name` for every
   code params handles (3.2), and, over all twenty-two shared names,
   `objective_code_from_name == params.objective_from_name`, with
   `lambdarank` and `custom` still refused by params. `MULTICLASS ==
   params.MULTICLASS` and `LAMBDARANK == ranking.LAMBDARANK` are proven at
   compile time: the compiler reports both comparison branches as
   unreachable. **This probe must run green before 3.2 through 3.4 land,
   and those edits delete it.**
3. **Lane 20's predicates agree too**, over all fourteen codes:
   `objective_is_builtin == device_policy.is_builtin_objective`,
   `objective_gradients_on_device ==
   device_policy.gpu_objective_is_device_resident`, and
   `LAMBDARANK == device_policy.LAMBDARANK` (again compile-time proven).
   This is what makes the section 9 merge mechanical rather than a
   negotiation.
4. **Links agree** (finding D5, now verified rather than asserted). For
   every single-output code at a fixed raw score, the link implied by
   `objective_link` reproduces `Booster.response(raw)` exactly, equals
   `response_scale(code, [raw])[0]` exactly, and maps to the same
   `RESPONSE_*` code `gpu_predict.response_for_objective` returns.
5. **Unimplemented names are preserved.** Each of the eight names in
   `unimplemented_objective_alias_names()` reports `NAME_UNIMPLEMENTED`;
   `"nonsense"` reports `NAME_UNKNOWN`; `"regression"` reports
   `NAME_SUPPORTED`.
6. **`metric_names_for_task` matches `metric_codes_for_task`**: joining the
   canonical names of the codes reproduces the string exactly, for all four
   tasks. This is the check that caught nothing but could have: those four
   strings were sorted by hand.
7. Defaults and parameter ranges: `objective_default_metric(SQUARED_ERROR)`
   is `l2`, `(LAMBDARANK)` is `ndcg`, `(CUSTOM)` raises;
   `check_objective_param(TWEEDIE, 2.5)` raises, `(TWEEDIE, 1.5)` passes,
   and `(MULTICLASS, 0.9)` is accepted unexamined.

Not yet asserted anywhere, and worth adding when this becomes a test: that
no name in `objective_alias_names()` or `metric_alias_names()` appears
twice. A duplicate would be harmless today (the resolver would return the
same code) but would silently break a Python dict built from the alias
snapshot.

### 7.3 Python facade, unchanged behavior (differential RUN, passes)

The rewrite must be behavior-identical, and it is. Appendix C loads the
pre-lane `_eval.py` (`git show ab25ad1:python/mojoboost/_eval.py`) and the
rewritten one side by side, by path, with no package import and no
extension module, and compares **408 cases**: every module constant; every
public name the old module exported; `task_metrics` over four tasks and a
junk task; `resolve` over all thirty-nine names plus whitespace, mixed
case, an integer, `None`, and nonsense, against all five tasks; and
`default_metric` over four tasks times twenty-six objective spellings plus
a callable and the no-objective call. Each case compares the return value,
or the exception type and its exact message. Result: no difference.

That covers `_eval` in isolation. It does not cover the estimators calling
it, which is what the suite below is for:

```
pixi run -e pytest pytest -q python/tests/test_eval_set.py
pixi run -e pytest pytest -q python/tests/test_basic.py
pixi run -e pytest pytest -q python/tests/parallel/test_cv.py
```

`test_eval_set.py:181-187` and `test_basic.py:393` assert on the exact
strings `unknown eval_metric` and `scores binary models`, which the facade
reproduces character for character. Then the whole suite:

```
pixi run -e pytest test-estimators
```

### 7.4 Differential: Python table against the native registry

The check that makes wiring safe. To be written as
`tools/check_registry_parity.py`, standard library plus the built extension,
run in CI next to `check-parity`:

```
pixi run build-python && python3 tools/check_registry_parity.py
```

It compares, and fails on any difference:

- `_eval._METRICS` against `registry_metrics()`: same twenty-one canonical
  names, same codes, same directions, same tasks.
- `_eval._ALIASES` composed with `_METRICS` against
  `registry_metric_aliases()`: same name-to-code map, both directions, no
  extra key on either side.
- `_eval._DEFAULTS` against `registry_objectives()`'s `default_metric_code`:
  every key that exists on both sides agrees. The seven keys of finding D3
  are expected to be **absent from Python and present natively**; the script
  must list them explicitly as a known, signed-off difference or fail.
- `MojoBoostRegressor._OBJECTIVES` against
  `registry_objective_aliases()` restricted to `TASK_REGRESSION` plus
  `CUSTOM`.
- `MojoBoostRegressor._OBJECTIVE_PARAM` against `(param_name,
  default_param)`.
There is no `GPU_OBJECTIVES` row: lane 20 removed that set. Its replacement
check is native and belongs in 7.2, item 3: `objective_is_builtin(c) ==
device_policy.is_builtin_objective(c)` for every code, plus
`LAMBDARANK == device_policy.LAMBDARANK`. Section 9 deletes it.

Written before any deletion, this script is what proves the deletion is
safe. Written after, it proves nothing.

### 7.5 Differential: nothing about training changed

The registry is metadata, so a correct wiring changes no number. Bit-exact
comparison before and after, on the same seed:

```
pixi run test
pixi run check-parity
pixi run -e bench compare-ranking
```

and, on a machine with an accelerator, because 3.4 and 3.5 touch the GPU
paths:

```
pixi run test-gpu
```

`docs/LIGHTGBM_PARITY.md` will need its section 9 sentence updated (3.10)
before `check-parity` is meaningful again.

## 8. What this lane did not do

- No repository test was written, and none was run. Three throwaway probes
  outside the repository were compiled and run (appendices A, B, C); the
  Python extension was not built, no benchmark ran, and no pixi task was
  added or changed. Sections 7.4 and 7.5 remain unrun.
- No public alias, default, direction, or task was changed. `_eval.py`'s
  three dicts are byte-identical in content to what they were.
- No file outside the three owned paths was edited, including
  `src/mojoboost/__init__.mojo`, so the registry is unreachable and
  unexported until 3.1.
- Findings D3, D4, D7, and D8 were documented, not fixed. Three of them
  change behavior and the fourth changes messages; that is an integration
  decision, not a lane decision.
- Lane 20's `src/mojoboost/device_policy.mojo` was **not** edited, even
  though it now duplicates part of the registry. See section 9.

## 9. Collision with lane 20, and the resolution to apply

Lane 20 (native device policy) and this lane both landed native objective
predicates in the same round, neither seeing the other. The overlap:

| Fact | Lane 20, `device_policy.mojo` | Lane 21, `objective_registry.mojo` |
| --- | --- | --- |
| the eleven built-in codes | `is_builtin_objective` | `objective_is_builtin` |
| gradients have a device kernel | `gpu_objective_is_device_resident` | `objective_gradients_on_device` |
| `train_gpu` accepts it | `gpu_trains_objective` | none (`objective_backends` answers a different question) |
| `LAMBDARANK = 7` | mirrored at line 147 | mirrored |

Both handoffs then specify **the same one-line edit** to
`gpu_objectives_native.supports_device_objective`, pointing at two
different functions. Applying both breaks; applying either alone leaves the
other's duplicate.

**The two lanes agree on every value.** Appendix B checks
`objective_is_builtin` against `device_policy.is_builtin_objective` and
`objective_gradients_on_device` against
`device_policy.gpu_objective_is_device_resident` over all fourteen codes,
and the two `LAMBDARANK` declarations against each other; all pass, the
constants at compile time. So this is a merge, not a reconciliation: no
behavior changes whichever way it is done.

**Proposed resolution.** The registry owns objective *identity and
capability*; `device_policy` owns *policy*: gates, reason codes, warnings,
memory estimates, and the `OBJECTIVE_UNSPECIFIED` sentinel. Concretely:

1. `device_policy.mojo` gains
   `from .objective_registry import LAMBDARANK, objective_gradients_on_device,
   objective_is_builtin` and deletes its `LAMBDARANK` mirror and the body
   of `is_builtin_objective`.
2. `gpu_trains_objective` and `gpu_objective_is_device_resident` **stay**,
   keeping their `OBJECTIVE_UNSPECIFIED` handling, with their bodies
   reduced to the imported predicate. They are policy questions with a
   policy sentinel; they are not the registry's to answer.
3. `supports_device_objective` delegates to the registry, not to
   `device_policy`, so that `gpu_objectives_native.mojo` does not acquire
   a dependency on the device policy layer.

This costs `device_policy` nothing in dependencies: it already imports
`.boosting`, and `objective_registry` imports `.boosting` and nothing else,
so no cycle and no new module is pulled into the CPU-compilable layer that
lane 20's mirror comment is protecting. Lane 20's mirror-pinning test can
then drop its objective rows: an import cannot drift.

**One real disagreement to settle, not just a duplicate.**
`gpu_trains_objective(CUSTOM)` is False and `objective_backends(CUSTOM)`
has `SUPPORTS_GPU`. Both are correct about different questions:
`train_gpu` does refuse `CUSTOM`, and `train_custom_gpu` does exist and
does grow trees on the device with the callback on the host
(`train_gpu.mojo`, and `docs/LIGHTGBM_PARITY.md` says so in section 8).
`MULTICLASS` is the same story with `train_multiclass_gpu`. The registry's
docstring now names the difference explicitly so nobody treats the two as
interchangeable, but the vocabulary should be settled at integration:
either rename `objective_backends` to say "a trainer exists for this
objective on this backend", or split it into per-entry-point answers. Do
not resolve it by making one of them agree with the other: they are
answering different questions and a caller needs both.

---

## Appendix. The probes that were run

Three files, none of them in the repository. They were run from the
repository root and are reproduced whole because they are the evidence
behind section 7 and because promoting A and B into
`tests/parallel/test_objective_registry.mojo` is most of that test already
written. Rewrite the `raise Error` checks as `std.testing.assert_*` and add
the duplicate-name check named at the end of 7.2.

### Appendix A. Registry self-check

```
pixi run mojo run -I src probe_registry.mojo
registry probe ok: 24 objective names, 39 metric names, 14 codes
```

```mojo
from mojoboost.objective_registry import (
    CUSTOM,
    LAMBDARANK,
    METRIC_L2,
    METRIC_NDCG,
    MULTICLASS,
    N_BUILTIN_METRICS,
    SQUARED_ERROR,
    TWEEDIE,
    all_objective_codes,
    check_objective_param,
    metric_alias_names,
    metric_canonical_name,
    metric_code_from_name,
    metric_codes_for_task,
    metric_names_for_task,
    metric_spec,
    objective_alias_names,
    objective_canonical_name,
    objective_code_from_name,
    objective_default_metric,
    objective_name_status,
    objective_spec,
    task_name,
    unimplemented_objective_alias_names,
)


def main() raises:
    # Every objective name resolves, and every code names itself back.
    var onames = objective_alias_names()
    for i in range(len(onames)):
        _ = objective_code_from_name(onames[i])
    var codes = all_objective_codes()
    for i in range(len(codes)):
        var c = codes[i]
        if objective_code_from_name(objective_canonical_name(c)) != c:
            raise Error("objective round trip failed for ", c)
        var spec = objective_spec(c)
        if spec.code != c:
            raise Error("spec code mismatch for ", c)

    # Every metric name resolves, and every code names itself back.
    var mnames = metric_alias_names()
    for i in range(len(mnames)):
        _ = metric_code_from_name(mnames[i])
    for c in range(N_BUILTIN_METRICS):
        if metric_code_from_name(metric_canonical_name(c)) != c:
            raise Error("metric round trip failed for ", c)
        var mspec = metric_spec(c)
        if mspec.code != c:
            raise Error("metric spec code mismatch for ", c)

    # Unimplemented names are still reported as unimplemented.
    var unimpl = unimplemented_objective_alias_names()
    for i in range(len(unimpl)):
        if objective_name_status(unimpl[i]) != 1:
            raise Error("expected unimplemented status")
    if objective_name_status("nonsense") != 2:
        raise Error("expected unknown status")
    if objective_name_status("regression") != 0:
        raise Error("expected supported status")

    # Task names, per-task metric lists, defaults, and parameter checks.
    for t in range(4):
        var names = metric_names_for_task(t)
        var per_task = metric_codes_for_task(t)
        var joined = String("")
        for i in range(len(per_task)):
            if i > 0:
                joined += ", "
            joined += metric_canonical_name(per_task[i])
        if joined != names:
            raise Error("task ", t, ": '", joined, "' != '", names, "'")
        _ = task_name(t)

    if objective_default_metric(SQUARED_ERROR) != METRIC_L2:
        raise Error("wrong default for squared error")
    if objective_default_metric(LAMBDARANK) != METRIC_NDCG:
        raise Error("wrong default for lambdarank")
    var custom_raised = False
    try:
        _ = objective_default_metric(CUSTOM)
    except:
        custom_raised = True
    if not custom_raised:
        raise Error("custom should have no default metric")

    var tweedie_raised = False
    try:
        check_objective_param(TWEEDIE, 2.5)
    except:
        tweedie_raised = True
    if not tweedie_raised:
        raise Error("tweedie 2.5 should be rejected")
    check_objective_param(TWEEDIE, 1.5)
    check_objective_param(MULTICLASS, 0.9)

    print("registry probe ok:", len(onames), "objective names,",
          len(mnames), "metric names,", len(codes), "codes")
```

### Appendix B. Mirrors against their sources

This is the one that authorizes the deletions in section 3 and the merge in
section 9. The three constant comparisons are reported by the compiler as
unreachable branches, which is a stronger result than passing: they cannot
differ.

```
pixi run mojo run -I src probe_mirrors.mojo
warning: 'if' condition always evaluates to 'False' (x3, the constant mirrors)
mirror probe ok: 14 codes, 24 names
```

```mojo
"""Does every fact the registry mirrors still equal its source?

This is the check that authorizes the deletions in the lane 21 handoff:
if any line here fails, a delegation would change behavior.
"""

from std.math import exp

from mojoboost.boosting import Booster, MonotoneConstraints
from mojoboost.custom_metric import response_scale
from mojoboost.device_policy import (
    gpu_objective_is_device_resident,
    is_builtin_objective,
)
from mojoboost.device_policy import LAMBDARANK as POLICY_LAMBDARANK
from mojoboost.gpu_objectives_native import supports_device_objective
from mojoboost.gpu_predict import (
    RESPONSE_EXP,
    RESPONSE_IDENTITY,
    RESPONSE_SIGMOID,
    response_for_objective,
)
from mojoboost.objective_registry import (
    CUSTOM,
    LAMBDARANK,
    LINK_EXP,
    LINK_IDENTITY,
    LINK_SIGMOID,
    LINK_SOFTMAX,
    MULTICLASS,
    all_objective_codes,
    objective_alias_names,
    objective_canonical_name,
    objective_code_from_name,
    objective_default_param,
    objective_gradients_on_device,
    objective_is_builtin,
    objective_link,
)
from mojoboost.params import MULTICLASS as PARAMS_MULTICLASS
from mojoboost.params import (
    objective_default_alpha,
    objective_display_name,
    objective_from_name,
)
from mojoboost.ranking import LAMBDARANK as RANKING_LAMBDARANK
from mojoboost.tree import Tree


def main() raises:
    # 1. Mirrored constants.
    if MULTICLASS != PARAMS_MULTICLASS:
        raise Error("MULTICLASS mirror differs")
    if LAMBDARANK != RANKING_LAMBDARANK:
        raise Error("LAMBDARANK mirror differs from ranking.mojo")
    if LAMBDARANK != POLICY_LAMBDARANK:
        raise Error("LAMBDARANK mirror differs from device_policy.mojo")

    var codes = all_objective_codes()
    for i in range(len(codes)):
        var c = codes[i]

        # 2. Device-kernel membership, three ways.
        if objective_gradients_on_device(c) != supports_device_objective(c):
            raise Error("gradients_on_device differs at code ", c)
        if objective_is_builtin(c) != is_builtin_objective(c):
            raise Error("is_builtin differs from device_policy at ", c)
        if objective_gradients_on_device(c) != (
            gpu_objective_is_device_resident(c)
        ):
            raise Error("device residency differs from policy at ", c)

        # 3. The scalar parameter default.
        if objective_default_param(c) != objective_default_alpha(c):
            raise Error("default param differs at code ", c)

        # 4. The link, against both places that decide it independently.
        var link = objective_link(c)
        if c != MULTICLASS:
            var booster = Booster(
                List[Tree](), 0.0, 0.1, c, MonotoneConstraints()
            )
            var raw = 0.75
            var expected: Float64
            if link == LINK_SIGMOID:
                expected = 1.0 / (1.0 + exp(-raw))
            elif link == LINK_EXP:
                expected = exp(raw)
            else:
                expected = raw
            if booster.response(raw) != expected:
                raise Error("Booster.response disagrees at code ", c)
            var scaled = response_scale(c, [raw])
            if scaled[0] != expected:
                raise Error("response_scale disagrees at code ", c)

            var device_code = response_for_objective(c)
            var expected_device: Int
            if link == LINK_SIGMOID:
                expected_device = RESPONSE_SIGMOID
            elif link == LINK_EXP:
                expected_device = RESPONSE_EXP
            else:
                expected_device = RESPONSE_IDENTITY
            if device_code != expected_device:
                raise Error("response_for_objective disagrees at ", c)

        # 5. The canonical name, against params' display name.
        if c != LAMBDARANK and c != CUSTOM:
            if objective_canonical_name(c) != objective_display_name(c):
                raise Error("canonical name differs at code ", c)

    # 6. Name resolution against params, for the names params accepts.
    var names = objective_alias_names()
    for i in range(len(names)):
        var name = names[i]
        if name == "lambdarank" or name == "custom":
            continue
        if objective_code_from_name(name) != objective_from_name(name):
            raise Error("name resolution differs for '", name, "'")

    # 7. The two names params refuses on purpose still raise there.
    for i in range(len(names)):
        var name = names[i]
        if name != "lambdarank" and name != "custom":
            continue
        var raised = False
        try:
            _ = objective_from_name(name)
        except:
            raised = True
        if not raised:
            raise Error("params should still refuse '", name, "'")

    print("mirror probe ok:", len(codes), "codes,", len(names), "names")
```

### Appendix C. `_eval.py` before against after

Needs no build and no extension module: it loads both versions by path. The
two path constants at the top were absolute to the session that ran it and
need adjusting before a rerun.

```
git show ab25ad1:python/mojoboost/_eval.py > eval_before.py
python3 diff_eval.py
compared 408 cases
PASS: no behavioral difference
```

```python
"""Behavioral diff of _eval.py before and after the lane 21 rewrite.

Loads both modules by path (no package import, no extension module) and
compares every public answer, exception type, and exception message.
"""

import importlib.util
import sys

SCRATCH = (
    "/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects/"
    "b1d30541-647a-4a5f-b760-48b9e5e30c1e/scratchpad"
)
AFTER_PATH = (
    "/Users/andrewhendel/CascadeProjects/mojoboost/python/mojoboost/_eval.py"
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


before = load("eval_before", SCRATCH + "/eval_before.py")
after = load("eval_after", AFTER_PATH)

failures = []


def cmp(label, fn_before, fn_after):
    try:
        a = ("ok", fn_before())
    except Exception as exc:
        a = (type(exc).__name__, str(exc))
    try:
        b = ("ok", fn_after())
    except Exception as exc:
        b = (type(exc).__name__, str(exc))
    if a != b:
        failures.append(f"{label}\n    before: {a!r}\n    after:  {b!r}")


# 1. module constants
CONSTS = [
    "L2", "RMSE", "L1", "QUANTILE", "HUBER", "BINARY_LOGLOSS",
    "BINARY_ERROR", "AUC", "MULTI_LOGLOSS", "MULTI_ERROR", "NDCG", "MAPE",
    "FAIR", "POISSON", "GAMMA", "GAMMA_DEVIANCE", "TWEEDIE",
    "CROSS_ENTROPY", "KLDIV", "AVERAGE_PRECISION", "MAP",
    "REGRESSION", "BINARY", "MULTICLASS", "RANKING",
]
for name in CONSTS:
    cmp(f"const {name}",
        lambda n=name: getattr(before, n),
        lambda n=name: getattr(after, n))

# every public name the old module exported must still exist
missing = [
    n for n in dir(before)
    if not n.startswith("__") and not hasattr(after, n)
]
if missing:
    failures.append(f"public names dropped: {missing}")

TASKS = [before.REGRESSION, before.BINARY, before.MULTICLASS, before.RANKING,
         "not_a_task"]

# 2. task_metrics
for task in TASKS:
    cmp(f"task_metrics({task!r})",
        lambda t=task: before.task_metrics(t),
        lambda t=task: after.task_metrics(t))

# 3. resolve, over every canonical name, every alias, and junk, x every task
NAMES = sorted(set(before._METRICS) | set(before._ALIASES)) + [
    "AUC", "  rmse  ", "L2_Root", "nonsense", "", "ndcg ", 7, None,
]
for name in NAMES:
    for task in TASKS:
        cmp(f"resolve({name!r}, {task!r})",
            lambda n=name, t=task: before.resolve(n, t),
            lambda n=name, t=task: after.resolve(n, t))

# 4. default_metric, over every objective spelling the regressor accepts
OBJECTIVES = [
    "regression", "regression_l2", "l2", "mean_squared_error", "mse",
    "huber", "quantile", "mae", "regression_l1", "l1",
    "mean_absolute_error", "poisson", "gamma", "tweedie", "mape",
    "mean_absolute_percentage_error", "fair", "cross_entropy", "xentropy",
    "lambdarank", "binary", "multiclass", "nonsense", "", None, 0,
]


def a_callable(raw, y):
    return raw, y


for task in TASKS:
    for obj in OBJECTIVES + [a_callable]:
        cmp(f"default_metric({task!r}, {obj!r})",
            lambda t=task, o=obj: before.default_metric(t, o),
            lambda t=task, o=obj: after.default_metric(t, o))
    cmp(f"default_metric({task!r}) no objective",
        lambda t=task: before.default_metric(t),
        lambda t=task: after.default_metric(t))

# 5. the mirrored tables themselves must be identical
for table in ("_METRICS", "_ALIASES", "_DEFAULTS"):
    cmp(f"table {table}",
        lambda t=table: getattr(before, t),
        lambda t=table: getattr(after, t))

total = len(CONSTS) + len(TASKS) + len(NAMES) * len(TASKS) + len(TASKS) * (
    len(OBJECTIVES) + 2
) + 3
print(f"compared {total} cases")
if failures:
    print(f"FAIL: {len(failures)} difference(s)")
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("PASS: no behavioral difference")
```
