# Task 11 handoff: serialization, split gains, inspection, model editing

Static work only. **Nothing was compiled, built, or tested.** No Mojo,
pixi, pytest, build, benchmark, formatter, or network command was run. One
exception to disclose: a single `python3 -c "ast.parse(open(...).read())"`
was run against `python/mojoboost/inspection.py` to check that the file
still parses. That imports nothing and executes none of the module, but it
is a Python process, so it is named here rather than left out.

This task committed nothing and staged nothing. Repository-wide commits
from other lanes swept the tree repeatedly while this one was working,
beginning with `dc21f03 Connect accelerator and public API foundations`
and `860b1cf Integrate training and interoperability subsystems` and
ending with `f6ae025 Snapshot parallel integration work` and `9c1e771
Refine integration handoffs and sparse training`. By the last of them
every file this task touched was already in history, so there is no
unstaged remainder to describe. That was not this task's doing, and
nothing was reverted to undo it. Every change described below is present
in the tree, whichever lane's commit carried it there.

The work ran in two phases. Phase 1 stayed inside the four owned paths
below the first rule. Phase 2 was explicitly authorized afterward and
applied the cross-lane patches phase 1 had only been able to request; §6
is written as applied rather than requested for that reason.

| File | What changed |
| --- | --- |
| **owned** | |
| `src/mojoboost/serialize.mojo` | model format **v4**: split gains, a presence flag on covers, an optional feature-names section, and `load_feature_names` |
| `src/mojoboost/importance.mojo` | docstrings only: gain importance now survives a save, and where the absence still shows |
| `src/mojoboost/inspection.mojo` | `MODEL_EDITING_SUPPORTED` and `model_editing_status_json`; gain docstrings tracking v4 |
| `python/mojoboost/inspection.py` | reads v4 (gains, cover flag, names); `feature_importance`, `leaf_outputs`, `model_editing_support`; name precedence |
| `handoffs/connect_11_serialization_inspection.md` | this file |
| **phase 2** | |
| `src/mojoboost/model_dump.mojo` | `MODEL_FORMAT_VERSION = 4` and the gain docstring behind it |
| `bindings/_mojoboost.mojo` | names through `save` / `save_multiclass`; `model_feature_names` |
| `python/mojoboost/basic.py` | names through `save_model` and `_load_path`; five stale docstrings |
| `python/tests/test_contrib.py` | its independent parser taught v4 |
| `python/tests/parallel/test_inspection.py` | v4 assertions, and four new tests |
| `tests/parallel/api_snapshot_manifest.json` | v4, and the `inspection` surface resynced |
| `docs/` ×5, `README.md` | §8 |

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
  for the same reason, so the native dump and the text dump report the
  same capability flags for the same model.
- `mojoboost.inspection` now also answers `feature_importance`,
  `leaf_outputs`, and `model_editing_support`, so the schema surface
  covers everything this task was asked to route through it.
- Feature names survive a save and a load, which is what makes the two
  implementations agree on `feature_names` for a model read from disk
  instead of one of them inventing `Column_i`.

**Now connected.** `src/mojoboost/model_dump.mojo` and
`src/mojoboost/inspection.mojo` had no caller anywhere when this task
began. `bindings/inspection_bindings.mojo` gives them one: the concurrent
inspection lane wrote and registered the dump hooks (§6.3), so
`mojoboost.inspection.dump_model` now finds a native `dump_model` and the
text parser is the fallback rather than the only path.
`model_editing_status_json` is the one function in `inspection.mojo` still
without a caller; it is a constant status with no handle argument, and
binding it is a one-line addition whenever a consumer wants it natively.

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
  its `COMPATIBILITY` banner. It was the only working path when this task
  began; the dump hooks are registered now (§6.3), so it is the fallback
  rather than the path. It is not dead code and should not be deleted with
  the hooks in place: a caller can hand `dump_model` a model *string*, for
  which there is no handle to call a hook on, and an older `_mojoboost.so`
  in the tree still lands here. `migration_19` §8 is still the deletion
  list and is still correct as an eventual plan; this task added three
  functions to it (`_parse_feature_names`, `_unescape_name`, `_text_gains`)
  and one signature (`_resolve_names` gained a `from_file` argument).
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
  stay independent; it hard-asserted `v3` and would have failed on the
  first v4 file, so it was taught v4 in place rather than pointed at the
  library parser it exists to disagree with. §7.

## 6. Cross-lane patches: applied

These began as requests, because the task that opened this handoff owned
only the four files in the table above. That boundary was then lifted
explicitly, so everything below is **applied** in the working tree, not
asked for. One item was deliberately left to its own lane, and it is
called out as such in §6.3.

Still nothing was compiled, built, or tested. "Applied" means the edit is
in the file, and nothing more than that.

### 6.1 `src/mojoboost/model_dump.mojo` — applied

`comptime MODEL_FORMAT_VERSION = 4`, the one patch that had to land with
this change or the native dump would misreport its own vintage. The
comment above it now points at `CURRENT_FORMAT_VERSION` in
`serialize.mojo`, which is the number it has to track and which exists for
that purpose. `has_split_gains`'s docstring no longer says gains "are not
serialized"; it says they are, from v4 on, and that a model read from a
v1, v2, or v3 file carries zeros.

### 6.2 `src/mojoboost/__init__.mojo` — nothing needed

`serialize` is not re-exported from the package `__init__`, so
`load_feature_names` needs no export line until it is. The file was left
alone.

### 6.3 `bindings/_mojoboost.mojo` — applied, with one deliberate exception

Applied here:

- `load_feature_names` imported from `mojoboost.serialize`.
- `_saved_names(feature_names, n_names)`, the sequence-plus-length decoder
  every string sequence at this boundary already uses.
- `save` and `save_multiclass` gained the `feature_names` / `n_names`
  pair and pass a `List[String]` through to `save_model`.
- `model_feature_names(path)`, which reads a file's names without loading
  the model, and returns an empty list for a pre-v4 file. Registered as
  `m.def_function[model_feature_names]("model_feature_names")`.

**Deliberately not applied here:** registering `bindings/
inspection_bindings.mojo`'s dump hooks. That module was authored by the
concurrent inspection lane, which was mid-flight on adding `-I bindings`
to `bindings/build.sh` while this task was working; duplicating the
registration would have clobbered it. That lane has since landed it —
`dump_model`, `dump_model_multiclass`, `dump_model_json`,
`dump_model_json_multiclass`, `split_values`, `dump_raw_scores`,
`dump_leaf_index`, `objective_code`, `model_file_kind`, and
`model_format_versions` are all registered as of `63aad82`. Nothing on the
Python side had to change to pick them up: `_hook` looks for exactly those
names and appends `_multiclass` for a softmax handle.

The `split_gains` / `split_gains_multiclass` hooks are **still not bound**,
and that is now a smaller gap than it was: v4 carries the gains in the
text, so the hook is only what gives a *pre-v4* file's gains back from a
live handle. `python/mojoboost/inspection.py` asks the text first and the
hook second, so binding it later changes nothing that works today.

### 6.4 `python/mojoboost/basic.py` — applied

1. `Booster.save_model` passes `names, len(names)` to both `save` and
   `save_multiclass`, so names written by Python survive the file.
2. `Booster._load_path` restores them with
   `names = list(_mojoboost.model_feature_names(path))` and assigns only
   `if names:`. The guard is not cosmetic: a pre-v4 blob returns an empty
   list, and an unconditional assignment would erase names an older pickle
   had carried in its own state.
3. `Booster.feature_importance`'s docstring no longer claims gains are not
   serialized. It reports the trained gains; only a model read from a
   pre-v4 file reports zeros.
4. `Booster.__getstate__`'s docstring, the `Booster` class docstring,
   `_continue`, `dump_model`, and the module docstring bullet were
   corrected the same way. A pickle is `model_to_string`, which carries
   gains from v4 on.
5. `Booster.dump_model`, `trees_to_dataframe`, and
   `get_split_value_histogram` exist as delegating methods. Another lane
   added them; they are noted here because §6 previously listed them as
   outstanding.

### 6.5 `python/mojoboost/__init__.py` — done by another lane

`mojoboost.inspection` is re-exported and `feature_name_`, `n_features_`,
`objective_` are delegating properties on `_Base`. Verified present, not
edited here.

## 7. Tests changed, and why each is a change in what is true

**All of these are edited and none are run.** Every one encodes a fact
that v4 changed, so leaving them alone would have left the suite asserting
the old format. They are written against a rebuilt extension: against the
prebuilt `_mojoboost.so` in the tree, which still writes v3, the v4
assertions do not hold. That is the expected state, not a defect, and it
is the first thing to check after a rebuild.

| File | Change |
| --- | --- |
| `python/tests/test_contrib.py` | `_ReferenceModel` asserts `"v4"` and keeps the version on `self`. `_read_feature_names` skips the optional `feature_names <n>` section between the version and the kind token; the count is read into a named local first, because a list comprehension over `self._next()` would leave the token order to evaluation order. `_read_trees` reads the `counts` flag and then the covers only when it is 1, then the `gains` flag and the gains the same way. `_Tree.__slots__` gained `"split_gain"`, without which the assignment raises. `_downgrade_to_v2` was rewritten to drop the four v3/v4 lines conditionally rather than one line per tree |
| `python/tests/parallel/test_inspection.py` | `model_format_version == 4`; `source in ("native", "model_to_string")`, because the native hooks are registered now and which one answers depends on the build; `has_split_gain is True`; internal nodes assert a finite `split_gain > 0.0`; `trees_to_records` asserts gain-present-iff-internal. `test_dump_from_the_model_text_alone` compares through a new `_without_source` helper, since a text dump and a native dump agree on everything except which of them built it. Four tests added: the editing refusal is reported rather than discovered; the dump's per-feature gain sums equal `feature_importance("gain")`; a saved and reloaded model keeps its gains; `leaf_outputs` are the leaf values by ordinal |
| `tests/parallel/api_snapshot_manifest.json` | `versions.model_format` and `model_format.version` are `"v4"`; `readable_versions` gained `"v4"`; `model_format.sections` gained a `v4` entry; `absent_by_design` **dropped** "split gains" and "feature names", which is the whole point of the change; `mojo.public_api.serialize` gained `load_feature_names`; `inspection.all` was resynced with `inspection.__all__` |
| `tests/test_serialize.mojo` | **not written.** Still the smallest real check of this change, and still worth adding: fit a small model, save, load, assert every node's `split_gain` matches bit for bit; save a model whose trees have no covers and assert it loads; round trip a name holding a space and a backslash |

## 8. Documentation updated

All applied. Listed so a reviewer can check the claims rather than
rediscover which files talk about the format.

| File | What changed |
| --- | --- |
| `docs/COMPATIBILITY_POLICY.md` | version table and §7.1 say v4; §7.2 gained a v4 row and two paragraphs, on reporting the absence of covers and gains rather than writing zeros, and on what a lossless re-save does; §5.3's persistence table and §7.5's "what the file deliberately does not hold" both corrected |
| `docs/MODEL_INSPECTION_SCHEMA.md` | `has_split_gain` is false only for a pre-v4 model, and both flags are properties of the model rather than of the version alone; a new "Where the names come from" section giving the four-source precedence; the editing status is now a documented payload; `leaf_outputs` and `feature_importance` added to the derived shapes; "Native hooks" rewritten for hooks that exist |
| `docs/LIGHTGBM_PARITY.md` | "Split gains in a dump" deferred → supported; `Booster.dump_model / trees_to_dataframe` partial → supported; the leaf-output row deferred → partial, since the getter half now exists; `Booster(model_file=)`; "now at v4". One citation was reworded so `tools/check_parity.py`'s `PATH_RE` can actually verify it: that regex only matches a backticked path, so `inspection.mojo::model_editing_status_json` was split into a path and a symbol |
| `README.md` | the covers-and-gains paragraph, which said gains cannot be recovered because the format does not hold them, and the pickle-versus-save paragraph |
| `docs/tutorials/feature_complete_walkthrough.md` | the persistence table; "currently v4"; what `has_split_gain` means; "re-save from a current build" replacing "retrain" |

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
  `inspection.model_editing_status_json` (Mojo) return the same seven
  keys with the same values: `supported`, `operation`, `reason`, the same
  three `invariants`, `serialized_state` naming `count` and `split_gain`,
  `model_format_version` (4 from either side), and
  `read_only_alternative`. The last two were each present on one side
  only until this task added the missing one to the other, so the seven-row
  table in `docs/MODEL_INSPECTION_SCHEMA.md` describes both payloads and
  not just the one it was written from. `set_leaf_output`,
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
   still uncompiled. The `_json_string` / `codepoint_slices` risk named in
   `migration_19` §14 is still the first thing to check there; this task
   introduced no second use of it. What this task did add to that file is
   `MODEL_EDITING_SUPPORTED` and `model_editing_status_json`, whose only
   construct worth a second look is `String(MODEL_FORMAT_VERSION)`. The
   file converts `Int` to `String` in dozens of places already
   (`String(rec.depth)` and its neighbors), but always from a struct
   field; this one is from the `comptime` alias in `model_dump.mojo`. If
   that distinction matters to the compiler, a local `var v: Int =
   MODEL_FORMAT_VERSION` before the conversion is the whole fix. Either
   way it carries less risk than the codec in item 1, not more.
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
   — the first thing v4 breaks. Its parser is already patched for v4, so
   this is a check of that patch and not of the library's.
5. `pixi run -e pytest pytest -q python/tests/parallel/test_inspection.py`
   — the four new cases in §7 are the ones that have never run: the
   editing status, the gain-sum identity, gains surviving a save, and
   `leaf_outputs`. Note that both files' v4 assertions **fail against the
   prebuilt `.so`**, so step 3 is not optional before either of them.
6. `pixi run check-parity`
   — `docs/LIGHTGBM_PARITY.md` changed, so this one is required rather
   than conditional. It is its own CI job. Three status transitions and
   one reworded path citation are what it will read.

Do not run the full `pixi run test` chain for this change.
