# Task 08 handoff: booster refit, rollback, editing, and bounds

Everything below is static reasoning. **No build, test, formatter, linter,
Python process, or benchmark was run.** The code this task produced is
**uncompiled**. This task staged and committed nothing, and touched no file
outside its ownership.

Ownership exercised: `src/mojoboost/model_editing.mojo` (new),
`docs/MODEL_EDITING.md` (new), and this file (new).

**The three files are nonetheless committed, at `63aad82` ("Document
remaining integration and compatibility work"), not by this task.** This is
a shared checkout and a concurrent lane's `git commit` picked them up. Two
consequences for whoever reads this next: uncompiled code is already in
history, so the first build that runs is a build of committed code and any
syntax fix is a follow-up commit rather than an amend; and the same broad
commit is why `git log` attributes these files to another lane's message.
The content at `63aad82` is this task's content unmodified.

## 0. Conflict with CONNECT_EVERYTHING Task 11

The prompt names `CONNECT_EVERYTHING TASK 11` as the conflict. That task
owns `src/mojoboost/serialize.mojo`, `src/mojoboost/inspection.mojo`,
`src/mojoboost/importance.mojo`, and `python/mojoboost/inspection.py`, and
its prompt says: *"Model editing methods should be exposed only if
invariants, serialization, and prediction remain coherent; otherwise return
an explicit unsupported status."*

That lane was active in the worktree while this task ran, and it had
already answered that question **no**:

- `src/mojoboost/inspection.mojo` gained `MODEL_EDITING_SUPPORTED = False`
  and a hardcoded `model_editing_status_json()` naming three invariants
  plus serialization as the reasons.
- `python/mojoboost/inspection.py` mirrors it as
  `MODEL_EDITING_SUPPORTED = False` and has a "Not offered" section in its
  module docstring. It has since gained definitions for `leaf_outputs` and
  `model_editing_support`, the latter a hand-written refusal restating the
  same three invariants; §4.3 is written against that later state.

**This task supersedes that answer, and §1.1 says why the three named
invariants do not in fact block leaf editing.** The patches in §4 are the
mechanical form of the supersession. They must be applied by the Task 11
owner (or by whoever holds those four files next), not here.

If the two are read side by side before §4 is applied, the repository
states two contradictory things about editing. `docs/MODEL_EDITING.md` and
`src/mojoboost/model_editing.mojo` are the later decision.

## 1. Implementations found, and what was fused

| Capability | What existed before | What this task did |
| --- | --- | --- |
| `rollback_one_iter` | nothing, in any layer | implemented in `model_editing.mojo`, with an explicit boosting mode |
| `lower_bound` / `upper_bound` | nothing | implemented, raw and response scale, per class for softmax |
| `get_leaf_output` | `inspection.py.leaf_outputs`, a Python walk of the dump (Task 11 defined it while this task ran; it was a bare `__all__` entry when the survey was taken) | implemented natively as `leaf_outputs`, `leaf_outputs_shrunk`, `get_leaf_output`, `get_leaf_output_shrunk`; the Python one is kept as the dump-side reader, see §4.3 |
| `set_leaf_output` | refused by `inspection.mojo.model_editing_status_json` | implemented, with a clamp-or-reject policy and a whole-tree monotone re-verification |
| `shuffle_models` | nothing | implemented as `permute_iterations` (explicit) plus a seeded `shuffle_iterations` |
| `refit` | nothing for `Tree`; `linear_tree.refit_linear_tree` exists for the linear-leaf model of another lane and is a different operation | implemented, reusing the grower's leaf formula and the trainer's `_renew_leaf_values` |
| leaf recomputation from data | `boosting._renew_leaf_values` (quantile / L1 / MAPE renewal after each grown tree) | **called**, not reimplemented: the renewing objectives' refit path is that function plus a decay blend |
| leaf value formula | `tree._leaf_value` (takes a `Histogram`) and `gain.soft_threshold_l1` + `tree_parameters_extra.finish_leaf_output` | the latter two are called directly; `_leaf_value` was not reused because it needs a per node `Histogram` that a refit does not build |
| monotone interval chain | `tree.node_bounds` | called; `monotone_claim_holds` is the missing verifier built on it |
| forest shrinkage | `boosting_rf._forest_rate`, `boosting_rf.is_forest` | `_forest_rate` restated locally (four lines, a private symbol of an untracked file); `is_forest` deliberately **not** called, see §1.2 |
| capability status | `inspection.mojo.model_editing_status_json` (hardcoded "false") | replaced by `editing_capabilities()` as data plus one renderer; §4.2 deletes the old one |

No duplicate registry, policy engine, trainer, or model representation was
introduced. `model_editing.mojo` holds no model state of its own: every
function takes a `Booster`, a `MulticlassBooster`, a `Model`, a
`MulticlassModel`, or a `Tree` and mutates it in place.

### 1.1 Why the three recorded objections do not block leaf editing

`inspection.mojo.model_editing_status_json` gives three invariants plus
serialization as the reason editing is unsupported. Taken one at a time:

1. *"node covers are the training rows that reached a node, and exact
   feature contributions condition on them."* True, and unaffected. A leaf
   **value** write does not change routing, so the set of rows that reached
   each node is exactly what it was and every cover stays exactly as true
   as it was. `contrib.mojo` reads leaf values and covers and nothing else
   (`tree_expected_value` sums `value[i] * count[i]` over leaves;
   `_tree_shap` walks the same two), so contributions after an edit
   correctly describe the edited model.
2. *"an internal node's value is the value it held when it was created, not
   a function of its children."* True, and preserved: `set_leaf_output`
   refuses a non-leaf node outright, and `refit` writes leaves only.
3. *"a split gain was computed from the gradient sums a leaf held at growth
   time, which the tree no longer holds."* True, and it is provenance
   rather than inconsistency: the gain remains a true statement about the
   fit that grew the tree. LightGBM's `refit` behaves the same way.
   `clear_split_gains` is the explicit retraction, which flips
   `has_split_gains` to false so a consumer reads an absence rather than a
   zero.

The serialization objection (*"a v4 file carries node covers and split
gains, so an edited leaf would be saved alongside the sums and counts that
contradict it"*) is the one that needed work rather than argument. The
answer has three parts: covers do not contradict a leaf edit (point 1);
gains are provenance, retractable (point 3); and refit recomputes covers
**all-or-nothing per tree**, so a model never carries covers describing two
datasets. What genuinely cannot be recorded today is that an edit happened
at all -- see §5.2 for the proposed optional `edited` section.

### 1.2 Why the boosting mode is asked for and never inferred

`boosting_rf.is_forest(booster)` tests `learning_rate == 1 / len(trees)`.
Its own docstring calls this a structural test rather than a label. An
ordinary GBDT model with ten trees at `learning_rate=0.1` passes it, so
inferring "forest" from it would rescale a model that was never averaged;
and `boosting_dart` bakes its per round normalization into leaf values via
`scale_tree`, leaving a DART ensemble structurally identical to a GBDT one.

So `rollback` takes an `EDIT_MODE_*` code:

- `EDIT_MODE_GBDT` (default): drop trees, keep the rate.
- `EDIT_MODE_RF`: drop trees and rescale the rate to `1 / (K - dropped)`.
  Refused when the ensemble does not actually carry the `1 / K` rate, so
  the caller's claim is checked rather than trusted.
- `EDIT_MODE_DART`: refused, as LightGBM refuses it.
- `EDIT_MODE_UNKNOWN`: refused, with the reason. A model file, a pickle,
  and an estimator handle all leave a caller here.

## 2. Call path, before and after

**Before.** No layer had any of these operations. `python/mojoboost/basic.py`
`Booster` exposes `current_iteration`, `num_model_per_iteration`,
`num_trees`, `update`, `predict`, `feature_importance`, `dump_model`,
`save_model`; nothing edits. `python/mojoboost/inspection.py` names
`leaf_outputs` and `model_editing_support` in `__all__` and reports
`MODEL_EDITING_SUPPORTED = False`.

**After, inside ownership.** `src/mojoboost/model_editing.mojo` is a
complete native core with two live production seams that need no other
lane:

- `refit_dataset(mut model: Model, dataset: Dataset, ...)` and
  `refit_dataset_multiclass(...)` take the same `Dataset` the trainers and
  `update_dataset` take, enforce the same binning-match and
  no-sparse rules, and read label / weight / init score off the dataset.
- every bounds, rollback, and range function is expressed against the real
  `IterationRange` prediction-slicing contract, and `clamp_range` is the
  post-edit counterpart of `IterationRange.slice`'s silent clamping.

**After, still blocked.** Nothing imports `model_editing.mojo` yet:
`src/mojoboost/__init__.mojo` is not owned here, so the module has no
export, no binding, and no Python surface. §4 is the full set of patches
that closes that, in dependency order.

## 3. What the module contains

Public surface, grouped:

- **Admission**: `check_finite`, `check_tree_structure`,
  `check_tree_serializable`, `check_booster_serializable`,
  `check_multiclass_serializable`.
- **Monotone claim**: `monotone_claim_holds`, `check_monotone_claim`,
  `check_monotone_claim_multiclass`.
- **Leaf reads**: `leaf_node_index`, `leaf_outputs`, `leaf_outputs_shrunk`,
  `get_leaf_output`, `get_leaf_output_shrunk`, `multiclass_tree_index`,
  `get_leaf_output_multiclass`.
- **Leaf writes**: `set_leaf_output`, `set_leaf_output_multiclass`,
  `set_leaf_outputs`, `clear_split_gains`, `clear_split_gains_booster`,
  `clear_split_gains_multiclass`.
- **Rollback**: `rollback`, `rollback_one_iter`, `rollback_to`,
  `rollback_multiclass`, `rollback_one_iter_multiclass`,
  `rollback_to_multiclass`, `clamp_range`.
- **Bounds**: `ScoreBounds`, `tree_leaf_bounds`, `raw_score_bounds`,
  `raw_score_bounds_range`, `response_bounds`,
  `raw_score_bounds_multiclass`, `raw_score_bounds_multiclass_range`,
  `probability_bounds_multiclass`.
- **Order**: `permute_iterations`, `shuffle_iterations`,
  `permute_iterations_multiclass`, `shuffle_iterations_multiclass`.
- **Refit**: `RefitParams`, `RefitReport`, `refit`, `refit_multiclass`,
  `refit_dataset`, `refit_dataset_multiclass`.
- **Capability**: `MODEL_EDITING_SUPPORTED`, `EditingCapability`,
  `editing_capabilities`, `editing_capability`, `check_editing_supported`,
  `model_editing_status_json`, and the `EDIT_MODE_*` / `LEAF_EDIT_*`
  constants.

`docs/MODEL_EDITING.md` is the normative statement of all of it. The
invariants the prompt asked to be defined are stated there in one table
each: model claims, per-operation behavior, refused operations,
serialization effect, interaction with early stopping / continued training
/ prediction slicing / importance / inspection / GPU, and determinism.

### 3.1 Decisions worth re-reading before trusting the module

- **Leaf ordinals, not node ids, not LightGBM leaf ids.** Every entry point
  addresses a leaf by `Tree.leaf_ordinals` rank. Documented at three
  places; it will surprise anyone porting LightGBM code.
- **Leaf values are unshrunk.** `get_leaf_output` returns the stored value;
  LightGBM's is already multiplied by the learning rate. The `_shrunk`
  variants exist for that reason.
- **Bounds are sound, not tight**, and per class for softmax. The softmax
  probability bounds are the box corners of the per-class raw bounds, which
  is exact for the box and loose for the model.
- **The monotone re-verification is whole-tree, not per-leaf.** Changing
  one leaf moves the midpoint its parent divides at, which moves the
  *sibling* subtree's intervals. A per-leaf clamp alone is not sufficient,
  which is why `_write_leaf` clamps, writes, re-derives, verifies, and
  reverts on failure.
- **Refit recomputes covers all-or-nothing per tree.** A single zero cover
  produces a file the reader rejects (§5.1), and mixed covers describe a
  background dataset that never existed.
- **Refit is refused for CUSTOM and LAMBDARANK**, whose gradients the
  fitted ensemble does not carry.

### 3.2 A latent inconsistency in `_renew_leaf_values`, not fixed here

`boosting._renew_leaf_values` clamps each renewed leaf into
`node_bounds(tree, monotone)` computed **once**, then writes the leaves in
node order. By the argument above, writing leaf `i` can move leaf `j`'s
interval for `j > i`, so the clamp is against a stale interval for every
leaf after the first write in a constrained tree. The residual percentile
is very unlikely to land in the gap, and no failure has been observed
(nothing was run), but the guarantee is weaker than the docstring claims.

`monotone_claim_holds` in `model_editing.mojo` is the checker that would
detect it. This task did not touch `boosting.mojo`. See §4.9.

## 4. READY-TO-APPLY INTEGRATION PATCHES

Dependency order: 4.1 first (nothing else compiles without it), then 4.2 /
4.3 (the supersession), then 4.4 (bindings), then 4.5 (Python), then the
rest independently.

---

### 4.1 Export the module

- **Target file**: `src/mojoboost/__init__.mojo`
- **Target symbol**: the import block, appended after the existing
  `from .serialize import (...)` block at the end of the file.
- **Ownership**: public exports. CONNECT Task 06 / Task 14 per those
  prompts; whoever holds `__init__.mojo`.
- **Patch**: insert

```mojo
from .model_editing import (
    EDIT_MODE_DART,
    EDIT_MODE_GBDT,
    EDIT_MODE_RF,
    EDIT_MODE_UNKNOWN,
    LEAF_EDIT_CLAMP,
    LEAF_EDIT_REJECT,
    MODEL_EDITING_SUPPORTED,
    EditingCapability,
    RefitParams,
    RefitReport,
    ScoreBounds,
    check_booster_serializable,
    check_editing_supported,
    check_monotone_claim,
    check_monotone_claim_multiclass,
    check_multiclass_serializable,
    check_tree_serializable,
    check_tree_structure,
    clamp_range,
    clear_split_gains,
    clear_split_gains_booster,
    clear_split_gains_multiclass,
    editing_capabilities,
    editing_capability,
    get_leaf_output,
    get_leaf_output_multiclass,
    get_leaf_output_shrunk,
    leaf_node_index,
    leaf_outputs,
    leaf_outputs_shrunk,
    model_editing_status_json,
    monotone_claim_holds,
    multiclass_tree_index,
    permute_iterations,
    permute_iterations_multiclass,
    probability_bounds_multiclass,
    raw_score_bounds,
    raw_score_bounds_multiclass,
    raw_score_bounds_multiclass_range,
    raw_score_bounds_range,
    refit,
    refit_dataset,
    refit_dataset_multiclass,
    refit_multiclass,
    response_bounds,
    rollback,
    rollback_multiclass,
    rollback_one_iter,
    rollback_one_iter_multiclass,
    rollback_to,
    rollback_to_multiclass,
    set_leaf_output,
    set_leaf_output_multiclass,
    set_leaf_outputs,
    shuffle_iterations,
    shuffle_iterations_multiclass,
    tree_leaf_bounds,
)
```

- **State flow**: none; an import list.
- **Errors**: none introduced.
- **Dependency**: `model_editing.mojo` imports `.binning`, `.boosting`,
  `.categorical`, `.gain`, `.model`, `.monotone`, `.ranking`, `.sampling`,
  `.tree`, `.tree_parameters_extra`, `.trainset`. All of those are already
  imported by `__init__.mojo` transitively, and none of them imports
  `model_editing`, so no cycle is created. Place the block **after**
  `from .trainset import (...)` so a reader sees the dependency order.
- **Fallback**: if the export is not applied, the module is dead code and
  nothing else in §4 can be applied either.
- **Serialization effect**: none.
- **Public API effect**: adds the Mojo-level editing surface.
- **Validation (UNRUN)**: `pixi run mojo build src/mojoboost/__init__.mojo`
  or the repo's existing package build step.

---

### 4.2 Retire the "editing is unsupported" status in `inspection.mojo`

- **Target file**: `src/mojoboost/inspection.mojo`
- **Target symbols**: `comptime MODEL_EDITING_SUPPORTED = False` (currently
  around line 318) and `def model_editing_status_json() -> String` (the
  whole function, including its four-paragraph docstring and the comment
  block above the constant).
- **Ownership**: CONNECT Task 11.
- **Patch**: delete the comment block, the constant, and the whole
  function. Add to the import block at the top of the file:

```mojo
from .model_editing import (
    MODEL_EDITING_SUPPORTED,
    model_editing_status_json,
)
```

  The module docstring's closing paragraph should gain one sentence:
  *"Model editing is `model_editing.mojo`'s subject; `MODEL_EDITING_SUPPORTED`
  and `model_editing_status_json` are re-exported here so a caller that
  reaches for `inspection` finds them where the schema doc says they are."*
  That matches the existing rationale for re-exporting the `model_dump`
  primitives.
- **State flow**: `editing_capabilities()` -> `model_editing_status_json()`
  -> the binding in §4.4 -> `inspection.model_editing_support()` in §4.3.
  One table, three consumers.
- **Errors**: `model_editing_status_json` does not raise. The re-exported
  name has the same signature (`() -> String`), so no call site changes.
- **Dependency**: §4.1 is not required (a direct `from .model_editing
  import` works), but `inspection.mojo` importing `model_editing.mojo`
  which imports `.model` is a heavier import graph than `inspection.mojo`
  has today. `inspection.mojo` already imports `.model`, so this adds
  `.trainset` and `.ranking` to its transitive set. If that is unwanted,
  the alternative is to leave `inspection.mojo` alone and let §4.4 bind
  `model_editing.model_editing_status_json` directly; then §4.2 reduces to
  deleting the stale constant and function so the repository does not state
  two answers.
- **Fallback**: without this patch, `inspection.mojo` reports
  `supported: false` while `model_editing.mojo` reports `true`.
- **Serialization effect**: none.
- **Public API effect**: the JSON grows an `operations` array, a
  `leaf_index` key (`"ordinal"`), and a `leaf_value_is_shrunk` key
  (`false`), and `supported` flips to `true`. The old keys `operation`,
  `reason`, `invariants`, `serialized_state`, and `model_format_version`
  are **removed**; `docs/MODEL_INSPECTION_SCHEMA.md` does not describe this
  object, so no schema version is implicated, but a consumer written
  against the old shape breaks. If that matters, keep `model_format_version`
  by adding it back in `model_editing.model_editing_status_json` (it would
  need a `MODEL_FORMAT_VERSION` import from `model_dump.mojo`, which
  `model_editing.mojo` deliberately does not have today).
- **Validation (UNRUN)**: one focused `mojo` compile of `inspection.mojo`.

---

### 4.3 Turn on editing in the Python inspection facade

- **Target file**: `python/mojoboost/inspection.py`
- **Target symbols**: the module docstring's "Not offered" section (the
  "Leaf editing" paragraph, ~line 50), `MODEL_EDITING_SUPPORTED` (~line
  113), `leaf_outputs` (~line 615), and `model_editing_support` (~line
  653).
- **Ownership**: CONNECT Task 11.
- **Restated against the file as it stands.** When this patch was first
  written both functions were names in `__all__` with no definition and the
  patch supplied them. Task 11 has since defined both, so the patch is now
  smaller and the sketches it used to carry are gone: `leaf_outputs()` as
  written is correct and is kept, including its `dump=` parameter and its
  `tree_index` range check, which the sketch did not have. What remains is
  the status, which is still the pre-editing refusal.
- **Patch**:
  1. Replace the docstring's "Leaf editing" paragraph with a pointer:

```
Editing
-------
`docs/MODEL_EDITING.md` is the contract and
`src/mojoboost/model_editing.mojo` is the implementation.
`model_editing_support()` returns the native capability table, operation by
operation; `leaf_outputs()` reads leaf values and
`mojoboost.Booster.set_leaf_output()` writes one.
```

  2. Replace the constant with a native read plus a conservative default,
     and `import json as _json` at the top if it is not already imported:

```python
def _editing_supported():
    hook = getattr(_mojoboost, "model_editing_status", None)
    if hook is None:
        return False
    return bool(_json.loads(hook())["supported"])


#: Whether this build can edit a fitted model in place. Read from
#: `src/mojoboost/model_editing.mojo`, which is where the decision is made;
#: False when the extension predates the binding.
MODEL_EDITING_SUPPORTED = _editing_supported()
```

  3. Replace the body of `model_editing_support()` with the native table.
     Its present body is a hand-written refusal built from the three
     objections §1.1 answers, so it cannot be kept alongside a build that
     edits; the reasons it lists are now wrong, not merely stale:

```python
def model_editing_support():
    """The native capability table: which editing operations this build
    performs, and the reason for each one it does not.

    `src/mojoboost/model_editing.mojo` is where the table is built, in
    `editing_capabilities`; `docs/MODEL_EDITING.md` is the contract it
    reports on. Refusals carry their reason, so a caller asking "can I set
    a split threshold?" gets an answer to branch on rather than an
    exception.
    """
    hook = getattr(_mojoboost, "model_editing_status", None)
    if hook is None:
        return {"supported": False, "operations": [],
                "reason": "this extension predates model editing"}
    return _json.loads(hook())
```

  4. In `leaf_outputs()`, change the closing docstring line
     `"""Read only. This is the half of LightGBM's leaf-output pair
     mojoboost offers; `model_editing_support()` says why there is no
     setter."""` to name the setter that now exists:
     "`mojoboost.Booster.set_leaf_output()` is the writing half."
     The function body is unchanged.
- **State flow**: `_mojoboost.model_editing_status()` (§4.4) ->
  `model_editing_status_json()` -> `editing_capabilities()`.
- **Errors**: none raised; a missing hook degrades to "not supported",
  which is the pre-existing behavior.
- **Dependency**: §4.4 for the hook. Applying §4.3 alone is safe and
  changes nothing observable (the hook is absent, so the constant stays
  `False` and `model_editing_support()` returns the same "not supported"),
  but it does drop the three stated reasons, which is the point: they are
  the claims §1.1 refutes and they should not outlive this patch.
- **Fallback**: the `getattr` guard is the fallback and must stay: the
  wheel and the source tree can disagree about which bindings exist.
- **Serialization effect**: none.
- **Public API effect**: `MODEL_EDITING_SUPPORTED` becomes build-dependent
  rather than a literal. `model_editing_support()` keeps its name and its
  "answer to branch on" contract but changes shape: the old keys
  (`operation`, `reason`, `invariants`, `serialized_state`,
  `model_format_version`, `read_only_alternative`) are replaced by
  `operations`, a list of per-operation records. Any consumer reading the
  old keys breaks; the schema is documented in `docs/MODEL_EDITING.md`.
- **Validation (UNRUN)**: `python -c "import mojoboost.inspection"`.

---

### 4.4 Bindings

- **Target file**: new `bindings/model_editing_bindings.mojo`, registered
  from `bindings/_mojoboost.mojo`.
- **Ownership**: bindings (CONNECT Task 06 / the bindings lane).
- **Why a new file**: `_mojoboost.mojo` already delegates per subject
  (`dataset_bindings`, `inspection_bindings`, `objective_bindings`,
  `distributed_bindings`, `basic_bindings`) and registers them in the one
  `PythonModuleBuilder`. This follows that convention exactly.
- **Patch, part 1**: create `bindings/model_editing_bindings.mojo` with

```mojo
from std.python import Python, PythonObject

from mojoboost.boosting import IterationRange
from mojoboost.model import Model, MulticlassModel
from mojoboost.model_editing import (
    EDIT_MODE_GBDT,
    LEAF_EDIT_CLAMP,
    RefitParams,
    ScoreBounds,
    get_leaf_output as mojo_get_leaf_output,
    get_leaf_output_multiclass as mojo_get_leaf_output_multiclass,
    model_editing_status_json,
    raw_score_bounds,
    raw_score_bounds_multiclass,
    refit_dataset as mojo_refit_dataset,
    refit_dataset_multiclass as mojo_refit_dataset_multiclass,
    response_bounds,
    rollback_one_iter as mojo_rollback_one_iter,
    rollback_one_iter_multiclass as mojo_rollback_one_iter_multiclass,
    set_leaf_output as mojo_set_leaf_output,
    set_leaf_output_multiclass as mojo_set_leaf_output_multiclass,
    shuffle_iterations,
    shuffle_iterations_multiclass,
)
from mojoboost.trainset import Dataset


def model_editing_status() raises -> PythonObject:
    """The capability table as JSON. Never raises: a consumer asks this
    before it tries anything."""
    return PythonObject(model_editing_status_json())


def rollback_one_iter(
    model: PythonObject, mode: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(
        mojo_rollback_one_iter(m[].booster, Int(py=mode))
    )


def rollback_one_iter_multiclass(
    model: PythonObject, mode: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(
        mojo_rollback_one_iter_multiclass(m[].booster, Int(py=mode))
    )


def score_bounds(
    model: PythonObject, response: PythonObject
) raises -> PythonObject:
    """`(lower, upper)` on the raw scale, or the response scale when
    `response` is true."""
    var m = model.downcast_value_ptr[Model]()
    var b: ScoreBounds
    if Bool(py=response):
        b = response_bounds(m[].booster)
    else:
        b = raw_score_bounds(m[].booster)
    var out = Python.list()
    out.append(PythonObject(b.lower))
    out.append(PythonObject(b.upper))
    return out^


def score_bounds_multiclass(model: PythonObject) raises -> PythonObject:
    """One `(lower, upper)` pair per class, raw scale. There is no single
    bound for a softmax model; see docs/MODEL_EDITING.md."""
    var m = model.downcast_value_ptr[MulticlassModel]()
    var bounds = raw_score_bounds_multiclass(m[].booster)
    var out = Python.list()
    for k in range(len(bounds)):
        var pair = Python.list()
        pair.append(PythonObject(bounds[k].lower))
        pair.append(PythonObject(bounds[k].upper))
        out.append(pair^)
    return out^


def get_leaf_output(
    model: PythonObject, tree_index: PythonObject, leaf_ordinal: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(
        mojo_get_leaf_output(
            m[].booster, Int(py=tree_index), Int(py=leaf_ordinal)
        )
    )


def set_leaf_output(
    model: PythonObject,
    tree_index: PythonObject,
    leaf_ordinal: PythonObject,
    value: PythonObject,
    policy: PythonObject,
) raises -> PythonObject:
    """Returns the value actually stored, which differs from `value` when a
    monotone clamp applied."""
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(
        mojo_set_leaf_output(
            m[].booster,
            Int(py=tree_index),
            Int(py=leaf_ordinal),
            Float64(py=value),
            Int(py=policy),
        )
    )


def shuffle_models(
    model: PythonObject,
    seed: PythonObject,
    start: PythonObject,
    stop: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    shuffle_iterations(
        m[].booster,
        Int(py=seed),
        IterationRange.slice(
            m[].booster.n_iterations(), Int(py=start), Int(py=stop)
        ),
    )
    return PythonObject(None)


def refit_dataset(
    model: PythonObject, dataset: PythonObject, params: PythonObject
) raises -> PythonObject:
    """Refit in place and return the report as a dict."""
    var m = model.downcast_value_ptr[Model]()
    var d = dataset.downcast_value_ptr[Dataset]()
    var p = RefitParams.default()
    p.decay_rate = Float64(py=params["decay_rate"])
    p.min_leaf_rows = Int(py=params["min_leaf_rows"])
    p.recount = Bool(py=params["recount"])
    var report = mojo_refit_dataset(
        m[], d[], p, Float64(py=params["alpha"])
    )
    var out = Python.dict()
    out["n_trees"] = PythonObject(report.n_trees)
    out["n_leaves_updated"] = PythonObject(report.n_leaves_updated)
    out["n_leaves_kept"] = PythonObject(report.n_leaves_kept)
    out["n_leaves_clamped"] = PythonObject(report.n_leaves_clamped)
    out["n_trees_recounted"] = PythonObject(report.n_trees_recounted)
    return out^
```

  plus `_multiclass` siblings for `get_leaf_output`, `set_leaf_output`,
  `shuffle_models`, and `refit_dataset`, following the suffix convention
  `inspection_bindings.mojo` documents.

  Two open items the bindings owner must settle, both noted rather than
  guessed here: whether `RefitParams` should carry the caller's
  `lambda_l1` / `lambda_l2` / `max_delta_step` / `path_smooth` across the
  boundary (it should -- add them to the `params` dict and set
  `p.tree.lambda_l1`, `p.tree.lambda_reg`, `p.tree.extra.max_delta_step`,
  `p.tree.extra.path_smooth`), and whether the monotone vector needs to be
  passed too (it does **not**: `refit` reads the ensemble's own
  `booster.monotone` and only compares it to `params.tree.monotone`, so the
  binding must copy `booster.monotone` into `p.tree.monotone` or the
  `_same_signs` check rejects every constrained model. **This is the one
  place the binding can get it wrong silently**; the simplest fix is to
  change `refit`'s check to skip when `params.tree.monotone.is_empty()`,
  which is a change inside this task's own file and is recorded as §6
  follow-up rather than applied, because it weakens a check).

- **Patch, part 2**: in `bindings/_mojoboost.mojo`, add the import block
  next to the other `*_bindings` imports and one `m.def_function[...]` line
  per entry point inside `PyInit__mojoboost`, alongside the existing
  `m.def_function[feature_importance]("feature_importance")` group.
- **State flow**: Python `Booster._handle` (an opaque `Model` /
  `MulticlassModel`) is mutated in place, so every subsequent `predict`,
  `dump_model`, `save_model`, and `feature_importance` sees the edit. The
  handle identity does not change, so `_copy_handle` and `__getstate__`
  keep working.
- **Errors**: every Mojo `raise` surfaces as a Python exception, which is
  how the rest of this extension behaves.
- **Dependency**: §4.1 is not required (these import `mojoboost.model_editing`
  directly, as `inspection_bindings` imports `mojoboost.model_dump`).
- **Fallback**: none needed; new names only.
- **Serialization effect**: none directly. Edits change what a later
  `save_model` writes (`value`, `count`, `split_gain`, tree count, tree
  order, and `learning_rate` for a forest rollback), all within format v4.
- **Public API effect**: new `_mojoboost.*` names only.
- **Validation (UNRUN)**: `bash bindings/build.sh`, then
  `python -c "import mojoboost._mojoboost as m; print(m.model_editing_status())"`.

---

### 4.5 `Booster` methods

- **Target file**: `python/mojoboost/basic.py`
- **Target symbol**: `class Booster`, a new `# -- editing ----` section
  between `# -- importance ----` and `# -- structured inspection ----`.
- **Ownership**: the Python API lane. Explicitly **not** this task
  ("Do not edit ... Booster Python").
- **Patch**:

```python
    # -- editing ----------------------------------------------------------
    #
    # docs/MODEL_EDITING.md is the contract. Two things every method here
    # must do: drop `_importance_cache`, because split-count importance
    # changes with a rollback and gain importance goes stale with an edit,
    # and drop the early-stopping metadata the edit invalidates.

    def _invalidate(self, *, iterations_changed):
        self._importance_cache = None
        if iterations_changed:
            for attr in ("_metric_best_iteration", "best_iteration"):
                if hasattr(self, attr):
                    setattr(self, attr, None)

    def rollback_one_iter(self, boosting="gbdt"):
        """Drop the last boosting iteration. LightGBM's
        `rollback_one_iter`.

        `boosting` states the mode, which a fitted ensemble does not
        record: "gbdt" keeps the learning rate, "rf" rescales it to
        1/(K-1), and "dart" is refused. Defaults to this Booster's own
        `boosting` parameter when it has one.
        """
        mode = _EDIT_MODES[boosting]
        call = (
            _mojoboost.rollback_one_iter_multiclass
            if self._n_classes
            else _mojoboost.rollback_one_iter
        )
        remaining = int(call(self._handle, mode))
        self._invalidate(iterations_changed=True)
        return remaining

    def lower_bound(self):
        """The smallest raw score this model can produce. Sound, not
        attained; per class for a softmax model."""
        return self._bounds()[0]

    def upper_bound(self):
        """The largest raw score this model can produce."""
        return self._bounds()[1]

    def _bounds(self):
        if self._n_classes:
            pairs = _mojoboost.score_bounds_multiclass(self._handle)
            return [p[0] for p in pairs], [p[1] for p in pairs]
        lo, hi = _mojoboost.score_bounds(self._handle, False)
        return lo, hi

    def get_leaf_output(self, tree_id, leaf_id):
        """One leaf's stored value. `leaf_id` is mojoboost's leaf ordinal
        and the value is unshrunk; see docs/MODEL_EDITING.md."""
        ...

    def set_leaf_output(self, tree_id, leaf_id, value, on_monotone="clamp"):
        """Set one leaf's stored value; returns what was stored, which
        differs when a monotone clamp applied."""
        ...

    def shuffle_models(self, start_iteration=0, end_iteration=-1, seed=0):
        """Shuffle iterations, LightGBM's `shuffle_models`. Seeded, unlike
        LightGBM's; whole per-class blocks move for a softmax model."""
        ...
        self._invalidate(iterations_changed=True)

    def refit(self, data, label=None, decay_rate=0.9, **kwargs):
        """Rebuild leaf values from new data, keeping tree shapes.

        `data` must be a mojoboost.Dataset binned by this model's own
        mapper (build it with `reference=` the training set), because a bin
        index has to mean the same thing to the trees as to the rows.
        Returns the refit report as a dict.
        """
        ...
        self._invalidate(iterations_changed=False)
```

  with `_EDIT_MODES = {"gbdt": 0, "rf": 1, "dart": 2, None: -1}` and
  `_LEAF_EDIT = {"clamp": 0, "reject": 1}` as module constants mirroring
  the `EDIT_MODE_*` / `LEAF_EDIT_*` comptime values.

- **State flow**: the methods mutate `self._handle` through §4.4 and then
  clear the two caches. `self._names`, `self._config`, `self._train_set`,
  and `self._valid_sets` are untouched and stay correct: an edit changes
  neither the feature count nor the binning.
- **Errors**: `KeyError` from `_EDIT_MODES` on an unknown mode should be
  turned into a `ValueError` naming the accepted spellings. Everything else
  propagates from Mojo.
- **Dependency**: §4.4.
- **Fallback**: guard each method with
  `if not hasattr(_mojoboost, "rollback_one_iter"): raise
  NotImplementedError(...)` so a stale extension fails with a sentence
  rather than an `AttributeError`, matching the `_hook` pattern in
  `inspection.py`.
- **Serialization effect**: none new. `save_model` after an edit writes the
  edited model.
- **Public API effect**: seven new `Booster` methods matching LightGBM's
  names, with the three documented deviations (leaf ordinal, unshrunk
  value, explicit boosting mode).
- **Validation (UNRUN)**: one focused
  `pixi run python -m pytest python/tests/test_basic.py -k booster -x`.

---

### 4.6 Estimator-level invalidation

- **Target file**: `python/mojoboost/__init__.py`
- **Target symbol**: the `_FIT_ATTRIBUTES` tuple near line 1004 ("Public
  attributes that `fit` sets and a refit clears") and `_reset_fit_state`
  near line 2085.
- **Ownership**: the Python API lane.
- **Patch**: if the estimators ever expose the `Booster` for editing, they
  must clear `best_iteration_`, `n_iter_`, `_metric_best_iteration`, and
  the cached model bytes `dask.py` keeps (see the comment at
  `python/mojoboost/dask.py:1438`, "a refit drops the stale bytes here").
  Nothing in the estimators changes today, because editing is a `Booster`
  operation and `_from_estimator` copies the handle.
- **Everything else**: no state flow, no errors, no serialization or public
  API effect. Recorded so it is not discovered later.
- **Validation (UNRUN)**: none until an estimator exposes editing.

---

### 4.7 `Tree` conveniences

- **Target file**: `src/mojoboost/tree.mojo`
- **Target symbols**: two new methods on `struct Tree`.
- **Ownership**: the Tree lane. Explicitly not this task.
- **Patch**:

```mojo
    def leaf_node_index(self, leaf_ordinal: Int) raises -> Int:
        """The node id of the leaf with this ordinal, the inverse of
        `leaf_ordinals`."""

    def check_structure(self) raises:
        """Raise unless the parallel arrays, the child links, the leaf
        count, and the category pool are self-consistent."""
```

  Both bodies exist verbatim in `model_editing.leaf_node_index` and
  `model_editing.check_tree_structure`. If they move,
  `model_editing.mojo` should keep the free functions as one-line
  forwarders so its own call sites and docs do not churn.
- **Why**: `leaf_node_index` is the inverse of a `Tree` method and belongs
  next to it; `check_structure` is a statement about `Tree`'s own
  invariants that `serialize.mojo` (§4.8), `tree_sparse.mojo`, and
  `distributed.mojo` could all use.
- **State flow / errors / serialization / public API**: none beyond the two
  new methods.
- **Dependency**: none. Deliberately deferred: `Tree` is heavily
  contended right now.
- **Validation (UNRUN)**: one focused `mojo` compile of `tree.mojo`.

---

### 4.8 Serialization: close the cover flag asymmetry

- **Target file**: `src/mojoboost/serialize.mojo`
- **Target symbols**: `_write_trees` (the `var covers = tree.has_node_counts()`
  line) and `_read_trees` (the `if not c > 0.0: raise` loop).
- **Ownership**: CONNECT Task 11.
- **The defect** (independent of editing, found while writing
  `check_tree_serializable`): the writer decides whether to write covers
  from `Tree.has_node_counts()`, whose docstring says "A grown tree always
  has a positive root cover, so the root alone settles it" -- it tests
  `count[0] > 0.0` only. The reader requires **every** cover to be
  positive. A tree with a positive root cover and a zero anywhere else is
  therefore written and then rejected by this build's own reader.
- **Patch**: change the writer's predicate to the full test:

```mojo
        var covers = _all_positive_counts(tree)
```

  with

```mojo
def _all_positive_counts(tree: Tree) -> Bool:
    """Whether every node carries a positive cover, which is what the
    reader requires. `Tree.has_node_counts` tests the root alone, which is
    the right question for "did this tree ever record covers" and the wrong
    one for "can these covers be written"."""
    if len(tree.count) != len(tree.feature):
        return False
    for i in range(len(tree.count)):
        if not tree.count[i] > 0.0:
            return False
    return True
```

  Alternatively, and equivalently, call
  `model_editing.check_tree_serializable` from `save_model` /
  `save_multiclass_model` before writing, which makes the writer refuse
  rather than silently drop covers. The first option is the conservative
  one and is what is recommended.
- **State flow**: none; a predicate.
- **Errors**: with the first option, such a tree writes `counts 0` and
  loads with no covers, so `predict_contrib` raises the existing "tree
  carries no node counts" error instead of the file failing to load. That
  is a strictly better failure.
- **Dependency**: none.
- **Fallback**: not needed.
- **Serialization effect**: a file that this build could write and not read
  becomes a file it can write and read. No version change: v4 already has
  the presence flag.
- **Public API effect**: none.
- **Validation (UNRUN)**: one focused save/load round trip.

- **Second observation, not a patch**: as of the working tree this task
  read, `serialize.mojo`'s **writer** is at v4 (it writes
  `feature_names`, the `counts` presence flag, and the `gains` section)
  while `_read_version` accepts only `v1`, `v2`, `v3` and `_read_trees`
  neither reads those flags nor reads gains (it fills zeros). Anything this
  build saves is therefore unreadable by this build. That lane is mid-edit
  and this is almost certainly work in progress; it is recorded here
  because `check_tree_serializable` was written against the reader's rules
  and will need re-reading once v4's reader lands.

---

### 4.9 Provenance: an optional `edited` section

- **Target file**: `src/mojoboost/serialize.mojo` (and
  `src/mojoboost/model_dump.mojo` / `docs/MODEL_INSPECTION_SCHEMA.md` to
  report it).
- **Ownership**: CONNECT Task 11.
- **The gap**: a saved model does not record that it was edited. A consumer
  reading a refit model's split gains has no way to learn they describe an
  earlier fit; a consumer reading a shuffled model has no way to learn that
  iteration `k` is not round `k`.
- **Patch sketch** (a v5 addition, optional so v4 files stay readable and
  an unedited model's bytes stay identical):

```
edited 3
refit 2
shuffle 1
leaf_write 7
```

  written only when at least one counter is nonzero; `Booster` would gain
  an `edits: EditLog` field carrying the four counters, `model_editing`
  would bump them, and the dump would report them under an `edits` object
  alongside `has_split_gain` / `has_node_count`.
- **State flow**: `model_editing.*` -> `Booster.edits` -> writer -> reader
  -> `ModelDump` -> `dump_model()["edits"]`.
- **Errors**: none; a counter.
- **Dependency**: a new field on `Booster` (the boosting lane) and a format
  version bump. This is the largest of the requests here and the least
  urgent; nothing today is wrong without it, only silent.
- **Fallback**: a v4 file loads with all counters zero, which reads as "not
  known to be edited" rather than "not edited". The dump key must be
  documented that way.
- **Serialization effect**: new optional section, new format version.
- **Public API effect**: one new dump key.
- **Validation (UNRUN)**: v4-file-loads-as-v5 round trip.

---

### 4.10 Also worth reporting to the renewal lane

- **Target file**: `src/mojoboost/boosting.mojo`, `_renew_leaf_values`.
- **Ownership**: the boosting lane.
- **Issue**: §3.2. The renewal clamps every leaf against intervals computed
  once, before any leaf is written, but writing a leaf moves its sibling
  subtree's intervals. `model_editing.monotone_claim_holds(tree, monotone)`
  is a four-line check that would confirm or refute it after the renewal
  loop.
- **Everything else**: no state flow or API change; the fix, if the check
  ever fails, is either to iterate the clamp to a fixed point or to
  recompute `bounds` inside the loop.
- **Validation (UNRUN)**: a constrained quantile fit followed by
  `monotone_claim_holds`.

---

### 4.11 C ABI

- **Target file**: `capi/mojoboost.h`, `capi/mojoboost_capi.mojo`
- **Ownership**: the capi lane.
- **Patch**: `MojoBoostRollbackOneIter`, `MojoBoostGetLowerBoundValue`,
  `MojoBoostGetUpperBoundValue`, `MojoBoostGetLeafValue`,
  `MojoBoostSetLeafValue`, `MojoBoostShuffleModels`, `MojoBoostRefit`,
  matching LightGBM's C API names so a C consumer ports directly. Each
  returns the library's existing status convention rather than raising.
- **Dependency**: §4.1 or a direct import.
- **Everything else**: mirrors §4.4 exactly; no new state, no new
  serialization effect.
- **Validation (UNRUN)**: the existing capi header/impl consistency check.

## 5. Risks

1. **Uncompiled.** Nothing here has been through the Mojo compiler. The
   most likely failure modes, in order: a `raises` annotation missing on a
   function that calls a raising one; `List` literal defaults; and the
   mutable-subscript argument pattern `_refit_tree_newton(booster.trees[t],
   ...)`, which follows the existing precedent at
   `src/mojoboost/custom_metric.mojo:1095`
   (`scale_tree_values(trees[t], lr0)` into a `mut tree: Tree` parameter)
   but is worth checking first.
2. **Private-symbol dependencies.** `model_editing.mojo` imports six
   underscore-prefixed names from other lanes' files:
   `_check_sample_weight`, `_fill_softmax_grad_hess`, `_renew_leaf_values`,
   `_same_signs`, `_softmax_inplace` from `boosting.mojo`; `_splitmix64`
   and `_uniform` from `sampling.mojo`; `_int_labels` from `trainset.mojo`.
   The precedent is established (`ranking.mojo` imports
   `boosting._check_sample_weight`; `bindings/_mojoboost.mojo` imports
   `boosting._softmax_inplace`), but `trainset.mojo` was being modified
   concurrently. If `_int_labels` moves, only
   `refit_dataset_multiclass` breaks.
3. **The refit monotone check is too strict for a binding.** `refit`
   compares `params.tree.monotone` to `booster.monotone` with
   `_same_signs`, mirroring `train_more`. A caller that builds
   `RefitParams.default()` and does not copy the ensemble's constraint
   vector into it will be rejected on every constrained model. §4.4 flags
   this; the alternative is to relax the check to "empty means inherit",
   which is a change to this task's own file and was not made because it
   weakens a guard rather than strengthening one.
4. **The status JSON shape changes.** §4.2's public API note. A consumer of
   the old `model_editing_status_json()` object breaks.
5. **`_forest_rate` is restated.** Four lines duplicated from
   `boosting_rf._forest_rate` rather than imported, because `boosting_rf.mojo`
   is another lane's new file and the symbol is private. If the two ever
   disagree, forest rollback silently produces a differently scaled model.
   The `EDIT_MODE_RF` guard (`learning_rate != _forest_rate(held)` raises)
   makes a disagreement loud rather than silent, but importing the real one
   is better once that lane settles.
6. **Refit cost.** `refit` walks every row through every tree twice per
   tree (once to accumulate, once to advance the raw scores), so it is
   `O(n_rows * n_trees * depth)` with no sampling and no parallelism. The
   trainer's own loop has the same shape. `parallel.dispatch_rows` is the
   obvious later optimization and was not attempted, because a parallel
   accumulation would change the summation order and therefore the leaf
   values.
7. **Nothing invalidates a Python-side cache today**, because nothing calls
   into this module today. The moment §4.4 lands without §4.5's
   `_invalidate`, `feature_importance()` will return pre-rollback numbers
   from `_importance_cache`. Apply §4.5 with §4.4, not after.

## 6. Deliberately not done

- No tests, no build, no benchmark, no formatter, no commit.
- No edit to `Tree`, `Booster`, serialization, trainers, bindings, Python,
  or tests: §4 is the whole set of requests instead.
- No `set_base_score` / `set_learning_rate` / `set_objective`. Each is
  refused by name in `editing_capabilities()` with the reason.
- No structural editing (`add_tree`, `set_split_feature`, `set_threshold`,
  `set_node_count`, `set_split_gain`). Same.
- No `Booster`-level `edits` counter, because it needs a field on `Booster`
  and a format version. §4.9.
- No parallelism in the refit walk. Risk 6.
- No LightGBM-file (`lgbm_model_io.mojo`) interaction. Editing produces a
  mojoboost model; whether it can still be written as a LightGBM file is
  that module's question and it does not depend on anything here, because
  no field it reads changes type or range.

## 7. Smallest later validation, all UNRUN

In order. Each is one command; none of them was run.

1. `pixi run mojo build src/mojoboost/model_editing.mojo` -- does the
   module compile at all. Expect the `raises` and mutable-subscript issues
   in Risk 1 first.
2. After §4.1: the repo's existing package build step, to confirm the
   export block introduces no import cycle.
3. One focused Mojo test (per `handoffs/` convention, **one** test, not the
   suite) asserting: `raw_score_bounds` on a two-tree fitted model equals
   `base_score + lr * (min+min)` and `base_score + lr * (max+max)`;
   `rollback_one_iter` then `predict` equals `predict_range` over
   `[0, n-1)`; `set_leaf_output` on a monotone-constrained model returns a
   clamped value and `check_monotone_claim` still passes; `refit` with
   `decay_rate=1.0` leaves every leaf value bit-identical.
4. After §4.4: `bash bindings/build.sh` then
   `python -c "import mojoboost._mojoboost as m; print(m.model_editing_status())"`.
5. After §4.5: `pixi run python -m pytest python/tests/test_basic.py -x -k
   booster`.

The fourth item is the one that matters most: `decay_rate=1.0` must be the
identity on leaf values, and if it is not, the blend or the sequencing is
wrong.
