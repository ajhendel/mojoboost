# Task 11 handoff: serialization, split gains, inspection, model editing

Static work only. **Nothing was compiled, built, or tested.** No Mojo,
pixi, pytest, build, benchmark, formatter, or network command was run. One
exception to disclose: a single `python3 -c "ast.parse(open(...).read())"`
was run against `python/mojoboost/inspection.py` to check that the file
still parses. That imports nothing and executes none of the module, but it
is a Python process, so it is named here rather than left out.

This task committed nothing and staged nothing. Two repository-wide
commits from other lanes swept the tree while this one was working
(`dc21f03 Connect accelerator and public API foundations` and `860b1cf
Integrate training and interoperability subsystems`), so most of
`src/mojoboost/serialize.mojo` and all of `src/mojoboost/importance.mojo`
and `src/mojoboost/inspection.mojo` are already in history; the rest is
unstaged in the working tree. That was not this task's doing, and nothing
was reverted to undo it. Every change described below is present in the
working tree, whichever side of those commits it landed on.

Owned paths, and this task's whole footprint:

| File | What changed |
| --- | --- |
| `src/mojoboost/serialize.mojo` | model format **v4**: split gains, a presence flag on covers, an optional feature-names section, and `load_feature_names` |
| `src/mojoboost/importance.mojo` | docstrings only: gain importance now survives a save, and where the absence still shows |
| `src/mojoboost/inspection.mojo` | `MODEL_EDITING_SUPPORTED` and `model_editing_status_json`; gain docstrings tracking v4 |
| `python/mojoboost/inspection.py` | reads v4 (gains, cover flag, names); `feature_importance`, `leaf_outputs`, `model_editing_support`; name precedence |
| `handoffs/connect_11_serialization_inspection.md` | this file |

---

## 1. Implementations found, before anything was edited

| Capability | Implementations in the tree | Authoritative one |
| --- | --- | --- |
| The dump schema, as data | `src/mojoboost/model_dump.mojo` (`ModelDump`, `build_dump`) | **model_dump.mojo**, untouched here |
| The dump schema, as JSON | `src/mojoboost/inspection.mojo` | inspection.mojo, a renderer over `ModelDump` |
| The dump schema, rebuilt in Python | `python/mojoboost/inspection.py`, below its `COMPATIBILITY` banner | the fallback, and today the **only** path with a caller |
| Feature importance | `src/mojoboost/importance.mojo` (native, bound, reached by `Booster.feature_importance`) | **importance.mojo** |
| Objective code to name | `python/mojoboost/inspection.OBJECTIVE_NAMES` and `lgbm_model_io.lgbm_objective_name` | deliberately two; see task 14 §8, unchanged |
| Model save/load | `src/mojoboost/serialize.mojo` | serialize.mojo |
| A second model-text parser | `python/tests/test_contrib.py::_ReferenceModel` | a test's reference implementation, deliberately independent |

Task 14 built the Python schema by parsing the model text; task 19 moved
the schema into Mojo and left the parser as a fallback. Both handoffs
(`handoffs/task14_inspection.md`, `handoffs/migration_19_model_inspection.md`)
were read first and their decisions are preserved except where this task
was told otherwise, which is exactly one place, §3 below.

## 2. Call path, before and after

**Before.** `Booster.dump_model` does not exist. `mojoboost.inspection.
dump_model(model)` looks for a `dump_model` hook on the extension module,
finds none (the bindings expose no dump), and falls back to parsing
`Booster.model_to_string()`. Gains are absent from that text, so every
dump reported `has_split_gain: False` and every node's `split_gain` was
`None`. `Booster.feature_importance("gain")` reached `importance.mojo`
natively but summed zeros for any model that had been through a file or a
pickle, because the format dropped the gains. `src/mojoboost/inspection.mojo`
and `src/mojoboost/model_dump.mojo` had no caller anywhere: not exported
from `src/mojoboost/__init__.mojo`, not bound, not imported by any module
in `src/`.

**After.** The same entry points, with the facts now present in the file:

- `save_model` writes gains, so the text the Python facade parses carries
  them. `dump_model(...)["has_split_gain"]` is `True` for a model written
  by a current build, and every internal node has a real gain, on the path
  that actually runs today.
- `load_model` reads them back into `Tree.split_gain`, so
  `Booster.feature_importance("gain")` on a model read from a file or
  unpickled reports the gains it was trained with instead of zeros. That
  is a public-API behavior change reached through the existing binding,
  with no binding edit.
- `model_dump.has_split_gains` (native) answers `True` for a loaded model
  for the same reason, so the native dump reports what the text dump
  reports the moment the bindings expose it.
- `mojoboost.inspection` now also answers `feature_importance`,
  `leaf_outputs`, and `model_editing_support`, so the schema surface
  covers everything this task was asked to route through it.

**Still not connected**, and it needs files this task did not own:
`src/mojoboost/inspection.mojo` and `src/mojoboost/model_dump.mojo` still
have no caller. The exports live in `src/mojoboost/__init__.mojo` and the
hooks in `bindings/_mojoboost.mojo`, both out of scope here. §6 restates
the exact patches, which are `migration_19`'s wave 1 unchanged.

## 3. The one decision reversed from an earlier handoff

`task14_inspection.md` §6 and `migration_19_model_inspection.md` §15 both
recommended **not** serializing gains: a property of how a tree was grown,
one float per node, read only by inspection. This task was instructed to
make serialization preserve split gains, and it does. The instruction is
the later decision; the earlier reasoning is not wrong, it was outweighed.

What it costs, stated plainly: one float per node per tree in every saved
model, roughly a 12% larger file for a numerical model (one more
bit-pattern array alongside seven), and a format version bump with the
migration work §6 and §7 list. What it buys: gain importance and gain-based
inspection survive a save, a pickle, and a `MojoBoostRegressor.load`, which
they did not before, and the two `split_gains` hooks earlier handoffs
designed to work around the absence are no longer needed.

## 4. What v4 is

`_VERSION = "v4"`, `CURRENT_FORMAT_VERSION = 4`. v1, v2, and v3 still load
and predict exactly as they did; every addition is either flagged or
optional.

| Section | Where | Shape |
| --- | --- | --- |
| `feature_names` | optional, immediately after `mojoboost v4` and before the kind token | `feature_names <n>` then `n` escaped tokens |
| cover presence | per tree, replacing v3's unconditional block | `counts <0\|1>`, then the covers only when 1 |
| split gains | per tree, after the covers | `gains <0\|1>`, then one bit pattern per node when 1 |

Three things worth knowing about the choices:

1. **The cover flag fixes a round trip v3 could not make.** A model loaded
   from a v1 or v2 file has no covers; v3 wrote its zeros as if they were
   covers, and v3's own reader rejects a nonpositive cover. Saving such a
   model produced a file that would not load. v4 records the absence.
   The reader also now accepts a v3 tree whose covers are *all* zero and
   reads it as the absence it is, so files already written by that bug
   load. A tree with some covers and not others is still refused: nothing
   can use a partial set (`Tree.check_node_counts` requires every node's),
   so accepting it would only move the failure.
2. **Gains are flagged, not validated for sign.** A leaf's gain is 0.0,
   and a model loaded from a pre-v4 file has all zeros, which is why
   `_has_split_gains` (one positive gain settles it, the per-tree form of
   `model_dump.has_split_gains`) decides whether to write the array at
   all. NaN is refused on read, because one NaN poisons every gain
   importance sum taken over the ensemble.
3. **Feature names are a writer's parameter, not a model field.** `Model`
   carries no names, so `save_model(model, path, feature_names=[])` and
   `save_multiclass_model(...)` take them and `load_feature_names(path)`
   hands them back. Every existing call site passes two arguments and is
   unaffected (`bindings/_mojoboost.mojo:2250`, `capi/mojoboost_capi.mojo`,
   `cli/mojoboost_cli.mojo`, two Mojo tests). A wrong-length list is
   refused at save and at load rather than dropped, because a wrong name
   is worse than no name.

Names are stored one whitespace-free token each. The file is a
whitespace-separated token stream, so `_escape_name` escapes space, tab,
newline, carriage return, and the backslash, encodes an empty name as
`\e`, and **refuses** any other control byte with a named error rather
than inventing an encoding for it. `_unescape_name` is its exact inverse,
and `python/mojoboost/inspection.py` carries the same codec for the
fallback parser.

Everything the task named is now in the file and round-trips:
split gains, covers, missing directions (`default_left`, `missing_bin`,
and the mapper's per-feature missing bin), category sets (the `categorical`
section and each tree's `cat` section), leaf values, objective code or
class count, learning rate, base score or per-class base scores, monotone
constraints, and feature names. Iteration grouping is recoverable rather
than stored: trees are round-major and `n_classes` is in the header, which
is what `num_tree_per_iteration` and `iteration` in the dump are computed
from.

## 5. Duplicates: fused, kept, and quarantined

- **Fused.** Nothing recomputes feature importance. `inspection.
  feature_importance` delegates to `Booster.feature_importance`, which is
  `importance.mojo` through the existing binding; the dump's per-feature
  gain sums equal it by construction because both read the same
  `Tree.split_gain`.
- **Kept deliberately.** `python/mojoboost/inspection.py`'s parser, below
  its `COMPATIBILITY` banner. It is the only working path until the dump
  hooks exist, and deleting it now would leave `dump_model` with nothing.
  `migration_19` §8 is still the deletion list and is still correct; this
  task added three functions to it (`_parse_feature_names`,
  `_unescape_name`, `_text_gains`) and one signature (`_resolve_names`
  gained a `from_file` argument).
- **Kept deliberately.** `_has_split_gains` in `serialize.mojo` restates
  `model_dump.has_split_gains` per tree. Importing `model_dump` into
  `serialize.mojo` would put an uncompiled module on the critical build
  path of every save; six lines that depend on nothing but `Tree` are the
  cheaper risk. Both docstrings name each other.
- **Kept deliberately.** `_native_split_gains` and the `split_gains` hook
  it looks for. v4 makes them unnecessary for a model written by a current
  build, and they remain the way a *pre-v4* text gets its gains back from
  a live handle. The text is asked first now; the hook is the fallback.
- **Quarantined, not removed.** `python/tests/test_contrib.py` carries its
  own model-text parser. It is a test's independent reference and should
  stay independent, but it hard-asserts `v3` and will fail on the first
  v4 file. §7 has the exact patch.

## 6. Cross-lane patch requests

Nothing below was applied. Each names the file, the owner as this round's
task list has it, and the exact change.

### 6.1 `src/mojoboost/model_dump.mojo` (inspection lane, task 19)

Line 68, and it is the one patch that must land with this change or the
native dump misreports:

```mojo
comptime MODEL_FORMAT_VERSION = 4
```

and in `has_split_gains`'s docstring, replace "are not serialized" with
"are serialized from model format v4 on, so a model read from a v1, v2, or
v3 file carries zeros". The comment above `MODEL_FORMAT_VERSION` should
point at `CURRENT_FORMAT_VERSION` in `serialize.mojo`, which is the number
it has to track and which now exists for that purpose.

### 6.2 `src/mojoboost/__init__.mojo` (public exports, tasks 06 / 14)

`migration_19` §3's export block, unchanged, plus one line on the existing
`from .serialize import ...` if there is one (there is not today; serialize
is not re-exported from the package `__init__`, so `load_feature_names`
needs no export until it is).

### 6.3 `bindings/_mojoboost.mojo` (task 06)

1. **Wave 1 of `migration_19` §4, verbatim**: `dump_model`,
   `dump_model_multiclass`, and their registration. `python/mojoboost/
   inspection.py` picks them up by name with no Python change; `_hook`
   already looks for exactly these names and appends `_multiclass` for a
   softmax handle. Waves 2 and 3 are still separable and still optional.
2. **Feature names through save and load.** `save` / `save_multiclass`
   gain the `feature_names` / `n_names` pair the dataset constructor
   already uses (`python/mojoboost/basic.py:562`), and pass a
   `List[String]` to `save_model`:

```mojo
def save(
    model: PythonObject,
    path: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    save_model(m[], String(py=path), _dump_names(feature_names, n_names))
    return PythonObject(None)
```

   and one new reader, so a loaded `Booster` can recover them:

```mojo
def model_feature_names(path: PythonObject) raises -> PythonObject:
    """The names a saved model carries, empty for a file that has none."""
    var out = Python.list()
    var names = load_feature_names(String(py=path))
    for i in range(len(names)):
        out.append(PythonObject(names[i]))
    return out^
```

   with `load_feature_names` added to the `from mojoboost.serialize import`
   list and `m.def_function[model_feature_names]("model_feature_names")`
   next to `load`.

### 6.4 `python/mojoboost/basic.py` (task 06)

1. `Booster.save_model`: pass `self._names` (or `[]`) and its length to the
   binding, so names written by Python survive the file.
2. `Booster._load_path`: after loading, `self._names = _mojoboost.
   model_feature_names(path) or None`. This is what makes
   `Booster.feature_name()`, the native dump, and the text dump all report
   the same names for a model read from disk; without it the file's names
   are visible only to the fallback parser.
3. `Booster.feature_importance` docstring: "Gains are not part of the
   serialized model, so a Booster read back from a file or a pickle reports
   zero gain importance" is now wrong. It reports the trained gains; only a
   model read from a **pre-v4** file reports zeros.
4. `Booster.__getstate__` docstring: "and the split gains do not" pickle is
   now wrong for the same reason. A pickle is `model_to_string`, which
   carries gains from v4 on.
5. Still outstanding from `task14_inspection.md` §3, unchanged and not done
   by anyone yet: `Booster.dump_model`, `Booster.trees_to_dataframe`,
   `Booster.get_split_value_histogram` as delegating methods, and the
   module docstring's "No `dump_model` / `trees_to_dataframe`" bullet.

### 6.5 `python/mojoboost/__init__.py` (task 06)

Unchanged from `task14_inspection.md` §4: re-export the `inspection`
submodule, add `"inspection"` to `__all__`, and add `feature_name_`,
`n_features_`, `objective_` to `_Base` as delegating properties (not in
`_FITTED_ATTRS`).

## 7. Tests that must change, and why each is a change in what is true

None of these are editable here. The first two **will fail** once the
extension is rebuilt with this change; they pass against the prebuilt
`_mojoboost.so` in the tree, which still writes v3.

| File | Change |
| --- | --- |
| `python/tests/test_contrib.py` | `_ReferenceModel.__init__` asserts `version == "v3"`; accept `"v4"` and read the two new per-tree blocks in `_read_trees`: after `missing_bin`, expect the token `counts`, read a flag, read `n_nodes` covers only when it is 1 (else zeros), then expect `gains`, read a flag, read `n_nodes` gains when it is 1. Also skip an optional `feature_names <n>` + `n` tokens right after the version, before the kind token. `_downgrade_to_v2` rewrites `mojoboost v3` and drops one line per tree; for v4 it rewrites `mojoboost v4` and drops the `counts` flag line, the covers line if present, the `gains` flag line, and the gains line if present |
| `python/tests/parallel/test_inspection.py` | `test_dump_reports_its_own_version_and_source`: `has_split_gain` is now `True` (`source` still starts with `model_to_string`, so that half stands). `test_nodes_carry_the_documented_keys` line 115: an internal node's `split_gain` is a finite number, not `None`. `test_trees_to_records_has_lightgbm_columns` line 311: internal-node rows carry a gain; leaf rows still carry `None`. Worth adding in the same commit: the per-feature sums of the dump's `split_gain` equal `booster_.feature_importance("gain")`, and a saved-and-reloaded model reports `has_split_gain: True` where it used to report `False` |
| `tests/parallel/api_snapshot_manifest.json` | `versions.model_format` and `model_format.version` become `"v4"`; `readable_versions` gains `"v4"`; `model_format.sections` gains a `"v4"` entry (`per-node split gains, flagged`, `per-node covers behind a presence flag`, `optional feature names`); `absent_by_design` **drops** `"split gains"` and `"feature names"`; `mojo.public_api.serialize` gains `"load_feature_names"`; `inspection.all` gains `"MODEL_EDITING_SUPPORTED"`, `"feature_importance"`, `"leaf_outputs"`, `"model_editing_support"` |
| `tests/test_serialize.mojo` | worth adding, and the smallest real check of this change: fit a small model, save, load, and assert every node's `split_gain` matches bit for bit; then save a model built from a tree with no covers and assert it loads. A names round trip is a third case |

## 8. Documentation that is now stale

| File | Line | What |
| --- | --- | --- |
| `docs/COMPATIBILITY_POLICY.md` | 71 | model format version `v3` → `v4` |
| | 408 | "`v3`" in the format description |
| | 428 | the version table gains a v4 row |
| `docs/MODEL_INSPECTION_SCHEMA.md` | — | `has_split_gain` is false when the model carries no gains, which is now only a pre-v4 file (`migration_19` §7 already asked for this clause); the "Native hooks" section; feature names can come from the model file |
| `docs/LIGHTGBM_PARITY.md` | 280 | "now at v3" |
| | the five rows `task14_inspection.md` §5 lists | still pointing at task 14; `pixi run check-parity` is its own CI job, so edit them in the same commit as the Python surface |
| `README.md` | 582 | says gains cannot be recovered because they are not in the format |
| `docs/tutorials/feature_complete_walkthrough.md` | 305, 400 | "currently v3", "retrain and save in v3" |

## 9. Fallbacks preserved

- The Python text parser is untouched as a path: every entry point still
  falls back to it when a native hook is missing, and `dump_model(text)`
  still works on a bare model string.
- Gains: text first (v4), then the `split_gains` hook (pre-v4 text with a
  live handle), then `has_split_gain: False`. No path asserts gains exist.
- Names: caller's override, then the `Booster`'s own names, then the
  file's, then `Column_i`. The middle one is read from `booster._names`
  rather than `feature_name()`, because `feature_name()` invents
  `Column_i` and invented names must not outrank a file's real ones.
- Covers: v4 flag, v3 block, all-zero v3 block read as absent, v1/v2
  absent. `contrib.mojo` still refuses a model without them, with its own
  message.
- Editing: refused, and now refused *explicitly*.
  `inspection.model_editing_support()` and
  `inspection.model_editing_status_json` (Mojo) return the same status,
  the same three invariants, and name `count` and `split_gain` as the
  serialized state an edit would contradict. `set_leaf_output`,
  `set_leaf_value`, and `edit_leaf` remain absent, so
  `test_leaf_editing_is_not_offered` still passes unchanged.

## 10. Public API and serialization effects

- **New in `mojoboost.inspection`**: `feature_importance`, `leaf_outputs`,
  `model_editing_support`, `MODEL_EDITING_SUPPORTED`. All four are in
  `__all__`.
- **Changed behavior**, all through existing entry points:
  `Booster.feature_importance("gain")` after a save/load or a pickle;
  `dump_model(...)["has_split_gain"]` and every node's `split_gain`;
  `trees_to_dataframe`'s `split_gain` column; `feature_importances_` on an
  estimator restored from a file, which reaches the same native code.
- **Format**: v4 files are not readable by an older build. v1 through v4
  are readable by this one. `forward_compatible` was already `false` in the
  manifest and stays false.
- **File size**: one more float array per tree, and one flag line each for
  covers and gains.
- **Signatures**: `save_model` and `save_multiclass_model` gained a
  defaulted third parameter; no existing call site changes.
  `load_feature_names` is new.

## 11. Risks, in the order to check them

1. `src/mojoboost/serialize.mojo` **is uncompiled**, and it is on the
   critical path of every build. Two constructs carry the risk, both with
   in-repo precedent but neither previously used in this file:
   `String(name[byte=a:b])` slicing (precedent:
   `lgbm_model_io.mojo:578`), and indexing a `List[UInt8]` materialized
   from `String.as_bytes()` (`_name_bytes`, which exists precisely so the
   codec never indexes a `Span`). If either misbehaves, the whole names
   feature is removable on its own: delete `_escape_name`,
   `_unescape_name`, `_name_bytes`, the two section functions,
   `_check_feature_names`, `load_feature_names`, and the two defaulted
   parameters. Gains and the cover flag do not depend on any of it.
2. `_read_trees` now reassigns a `var List` (`split_gain = List[Float64]
   (capacity=n_nodes)` after a conditional read). If move-assignment onto
   a declared `var` is rejected, build the list in one branch instead.
3. `if g != g` as the NaN test on a `Float64`. Same shape as the existing
   `if not c > 0.0`; if it does not compile, `from std.math import isnan`
   is what `inspection.mojo` uses.
4. `src/mojoboost/inspection.mojo` and `src/mojoboost/model_dump.mojo` are
   still uncompiled, unchanged from `migration_19` §14. The
   `_json_string` / `codepoint_slices` risk that handoff names is still
   the first thing to check there; this task did not touch it and did not
   introduce a second use of it.
5. `python/mojoboost/inspection.py` was not executed. Its v4 branches
   cannot run until the extension is rebuilt, and its name codec has never
   been round-tripped against the Mojo one. The two were written from the
   same table; a focused test that saves a model with a name containing a
   space and reads it back is what would prove it.
6. `getattr(booster, "_names", None)` reaches a private attribute of
   `Booster` from a sibling module. It is the same package and the
   attribute is set in four places in `basic.py`, but it is coupling, and
   the honest fix is a public accessor that distinguishes "no names" from
   `Column_i` (`Booster.feature_name()` cannot).
7. Anything that reads a mojoboost model file outside this repository, or
   any stored `.mbst` fixture, is now written in a format it does not
   know. No fixture was found in the tree; the CLI, the C ABI, and the
   Python layer all read through `serialize.mojo`.

## 12. Smallest focused validation, in order. **ALL UNRUN.**

Run one at a time and stop at the first failure.

1. `pixi run mojo package src/mojoboost -o /tmp/mojoboost.mojopkg`
   — type-checks `serialize.mojo`, `importance.mojo`, `inspection.mojo`,
   and `model_dump.mojo` in one pass without adding a test file. Expect to
   fix §11's items before anything else here is worth trying.
2. `pixi run mojo run -I src tests/test_serialize.mojo`
   — the existing save/load suite, which is the one file that covers this
   change directly. It should pass unchanged: v4 round-trips everything v3
   did.
3. `pixi run build-python`
   — only after 1 and 2. Everything below needs the rebuilt extension.
4. `pixi run -e pytest pytest -q python/tests/test_contrib.py`
   — the first thing v4 breaks, and the patch in §7 is what fixes it.
5. `pixi run -e pytest pytest -q python/tests/parallel/test_inspection.py`
   — with §7's three edits applied. Without them, three cases fail for the
   documented reason, which is not a signal to revert.
6. `pixi run check-parity`
   — only if `docs/LIGHTGBM_PARITY.md` changed.

Do not run the full `pixi run test` chain for this change.
