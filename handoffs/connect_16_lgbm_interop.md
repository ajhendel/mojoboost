# Connect 16: LightGBM model interop, without weakening the native format

Files this lane owns and touched:

- `src/mojoboost/lgbm_model_io.mojo` (rewritten in parts; was 1616 lines, now ~2870)
- `python/mojoboost/lgbm_model_io.py` (new)
- `handoffs/connect_16_lgbm_interop.md` (this file)

Nothing else was edited, staged, or committed. Everything below that needs
another lane's file is an exact patch request, not a change.

> **This lane committed nothing, but its files are committed.** A concurrent
> session swept the shared worktree into `860b1cf "Integrate training and
> interoperability subsystems"` and the commits around it while this lane was
> mid-edit, exactly as happened to `handoffs/task22_lgbm_io.md`'s lane in
> `b5a9afc`. Unlike that time the snapshot appears to be the finished state:
> `git status` reports both owned source files clean against HEAD and the
> markers of every change below are present. Verify before trusting that:
> `rg -c "import_lgbm_file|LGBM_INTEROP_STATUS" src/mojoboost/lgbm_model_io.mojo`
> should report 9, and `python/mojoboost/lgbm_model_io.py` should exist. If a
> later commit reverted either, the changes are described in enough detail
> below to be reapplied.

Read alongside `handoffs/task22_lgbm_io.md`, which is the original lane's
handoff for this module. Where the two disagree, this one is later: the
biggest disagreement is categorical splits, which that handoff called "the
largest gap" and which this lane found to be exactly representable. Its
reasoning is corrected in "Duplicates fused" below.

---

## 1. Implementations found

Inventory before writing anything. There was exactly one LightGBM
model-format implementation, and it had no caller.

| Where | What it is | Verdict |
| --- | --- | --- |
| `src/mojoboost/lgbm_model_io.mojo` | The only LightGBM text-format parser and writer in the repo | **Authoritative.** Extended and connected, not replaced |
| `python/mojoboost/inspection.py`, "COMPATIBILITY" section | A parser, but of `serialize.mojo`'s *own* format (`Booster.model_to_string()`), not LightGBM's | Not a duplicate of this capability. Untouched (Task 11 owns it) |
| `bench/compare_categorical_lightgbm.py`, `compare_missing_lightgbm.py`, `compare_ranking.py`, `tools/check_parity.py` | Call the real `lightgbm` package for numerical comparison | Not format code. Untouched |
| `src/mojoboost/objective_registry.mojo` | `objective_code_from_name`, `objective_canonical_name`, `objective_param_name` | **Authoritative for objective metadata.** This module now delegates to it |
| `src/mojoboost/params.mojo` `objective_from_name` | Narrowing wrapper over the registry | Dependency dropped; this module goes to the registry directly |
| `python/mojoboost/inspection.py` `OBJECTIVE_NAMES` | Python objective-code table | Another lane's duplicate. Not touched, flagged in section 6 |
| `docs/LIGHTGBM_PARITY.md:126, :290, :617` | Records this module as deferred, "No caller, no export", "connected to nothing" | Now stale. Patch request in section 6 |

`rg` for `lgbm_model_io|load_lgbm|save_lgbm|parse_lgbm|dump_lgbm|from_lightgbm|to_lightgbm`
over the whole tree found no importer, no binding, and no Python caller.
`src/mojoboost/__init__.mojo`, `bindings/_mojoboost.mojo`, `capi/`, `cli/`,
and `python/mojoboost/` all had zero references.

---

## 2. Call path, before and after

### Before

```
LightGBM model file  ->  (nothing)
```

The module compiled and had a test. No production or public path reached it.
`pixi run test` ran `tests/parallel/test_lgbm_model_io.mojo`, which was its
only caller of any kind.

### After (Mojo, live today)

```
import_lgbm_file(lgbm_path, model_path)
  -> open + _parse_lgbm_text                    header + Tree= blocks
  -> _check_version
  -> lgbm_objective_code                        -> objective_registry
  -> _synthesize_mapping
       -> _check_feature_indices
       -> _collect_categories                   -> CategoricalSpec
       -> _collect_edges                        numerical nodes only
       -> _collect_missing_bins                 numerical nodes only
       -> _bins_needed                          numerical + categorical width
       -> BinMapper
  -> _convert_trees -> _convert_tree -> _emit_node
       -> _edge_index          (numerical)
       -> _cat_bin_of          (categorical, through CategoricalSpec.bin_of)
       -> _NodeArrays.set_categorical  (same layout as Tree._set_split)
  -> Booster / MulticlassBooster
  -> serialize.save_model(model, model_path, report.feature_names)   <-- product

export_lgbm_file(model_path, lgbm_path)
  -> serialize.load_feature_names
  -> serialize.model_file_kind
  -> serialize.load_model / load_multiclass_model
  -> export_lgbm_model -> _tree_body -> _emit_lgbm_node
       -> _node_category_codes  (bins -> raw codes, via cat_pool_contains)
       -> _LgbmOut.add_category_set
  -> _assemble  (feature_names, feature_infos, tree_sizes)
  -> write
```

The product of an import is **a mojoboost model file**, never a live handle
kept in LightGBM's shape. That is the design decision this whole lane hangs
on; section 4 says why.

### After (Python, once the four bindings land)

```
mojoboost.lgbm_model_io.load_lightgbm_model(path)
  -> _mojoboost.lgbm_import_file(path, tmp.mbst)
  -> mojoboost.Booster(model_file=tmp.mbst)      existing native loader

mojoboost.lgbm_model_io.save_lightgbm_model(booster, path)
  -> booster.save_model(tmp.mbst)                existing native writer
  -> _mojoboost.lgbm_export_file(tmp.mbst, path)
```

No new handle marshalling, no new model type crossing the boundary, and no
Python-side parsing. The Python module is 250 lines of which none parse or
interpret a model.

---

## 3. Connections completed

**1. Categorical splits, both directions, exactly.** Previously every
categorical construct raised. Now `decision_type` bit 0, `num_cat`,
`cat_boundaries`, and `cat_threshold` are read, validated, and converted, and
mojoboost categorical nodes are written back out.

The argument that this is exact rather than approximate, in full, because the
prior handoff concluded the opposite:

- LightGBM's `cat_threshold` is a bitset over **raw category codes**, not over
  its internal bins. `Tree::CategoricalDecision` tests
  `static_cast<int>(fval)` against it directly, and `SerialTreeLearner::Split`
  converts bins to values through `BinToValue` before storing them. So the
  fitted category-to-bin table, which the file does not carry, is not needed
  to route.
- LightGBM sends a code left exactly when its bit is set. Everything else
  (unset bit, negative, NaN, past the end of the bitset) goes right, and
  `default_left` and `missing_type` are not consulted at all on a categorical
  node.
- mojoboost sends a bin left exactly when it is in the node's 256-bit set, and
  `CategoricalSpec.bin_of` maps missing, unseen, and untabulated values to bin
  0, which `Tree.goes_left` guarantees is never a set member.
- Therefore: take the union of every code any categorical split on a feature
  mentions, sort it, make it that feature's table, and map each split's codes
  to bins. Codes in the table route as they did by construction. A code
  outside the table has its bit unset in every split (that is what being
  outside the union means), so LightGBM sent it right, and it maps to bin 0,
  so mojoboost sends it right. The two agree on **every possible input**, not
  only on the training data.

The union is then widened with whatever `feature_infos` lists for the feature
(LightGBM writes the fitted category list there, `Common::Join(bin_2_categorical_, ":")`),
when the header carries one and the widened table still fits. Widening cannot
change any routing decision, because the codes it adds are exactly the ones no
split mentions; it just makes the imported table the one LightGBM fit rather
than a projection of it. `LgbmImportReport.n_widened_tables` says how many
features got the wider table.

**2. Objective metadata fused onto the registry.** `lgbm_objective_code` now
calls `objective_registry.objective_code_from_name` and adds only what is
specific to a model *file*. `lgbm_objective_name` now calls
`objective_canonical_name` and overrides it in exactly two documented places.
A third function, `lgbm_dropped_objective_param`, reports the objective
setting a fitted model cannot carry, and it is `objective_param_name`
verbatim.

**3. Feature names wired to the native v4 header.** Task 11's
`save_model(..., feature_names)` and `load_feature_names` landed mid-lane and
close a gap the prior handoff listed as open. LightGBM's `feature_names=`
header now crosses into the native file on import, and the native file's names
cross into `feature_names=` on export. A count that does not match the feature
count is dropped whole rather than trusted in part, and a name containing
whitespace is refused on write, because the line is space-separated and a
shifted name is worse than no name.

**4. The native format is the product of an import.** `import_lgbm_file`
writes `serialize.save_model`. That is not decoration: it is what proves every
piece of state the conversion produced (synthesized edges, category tables,
per-feature missing reservations, node covers, monotone constraints) is state
the native format actually carries. A conversion whose output could not
serialize would fail there rather than silently produce a process-lifetime-only
model.

**5. Conversion reports.** `LgbmImportReport` and `LgbmExportReport` carry the
facts the finished model no longer records: that the mapper is not the
training binning (`mapper_is_training_binning()`, which returns False and is a
method so a caller can assert on it), how many features turned out
categorical, how many tables were widened, whether node covers survived (so
`predict_contrib` will work), which objective setting was dropped, and whether
the exported file is bit-exact or ULP-accurate
(`predictions_bit_exact()`).

**6. A non-raising capability query.** `lgbm_unsupported_reason(text)` and
`lgbm_file_unsupported_reason(path)` return the refusal sentence, or `""` when
the file converts. They are answered by *running* the conversion, so there is
no second list of supported constructs to drift from the converter. This is
what makes "can I load this?" answerable without exception handling, which is
what the Python and C facades need.

**7. The experiment is labelled in one place.** `LGBM_INTEROP_STATUS` /
`lgbm_interop_status()` is the single sentence saying how far this has been
checked. The Python facade surfaces it verbatim and warns once per process on
first conversion.

### New reader rejections (all raise, all named)

- A feature split numerically in one node and categorically in another
  (mojoboost decides that per feature).
- More than `CAT_MAX_BINS - 1` = 255 categories on one feature.
- A category set wider than `_MAX_CAT_WORDS` 32-bit words (2,097,152 codes),
  read and written alike. Without this a file could declare a bitset billions
  of bits wide and the decoder would loop over all of it.
- An empty category set (routes every row right; not a split).
- A categorical `threshold` that is not a whole number in `[0, num_cat)`.
- `cat_boundaries` that does not open at 0, does not ascend, or disagrees with
  `len(cat_threshold)`.
- A `cat_threshold` word outside 32 bits.
- A node marked categorical in a tree declaring `num_cat=0`.
- `objective=custom` (previously it would have resolved through the registry
  to `CUSTOM`, whose link mojoboost cannot know).

`missing_type` is now *accepted and ignored* on categorical nodes, including
`Zero`, because LightGBM's own predictor ignores it there. It is still
rejected on numerical nodes.

### New writer rejections

- A categorical node whose feature has no category table in the model's
  mapper (its bins cannot be turned back into codes). This replaces the blanket
  "categorical models cannot be written yet".
- A feature name that is empty or contains whitespace.
- `feature_names` of the wrong length.
- A category set containing a code too large for the word cap.

---

## 4. Why file-to-file, and why the native format stays authoritative

The four public entry points (`import_lgbm_file`, `export_lgbm_file`,
`lgbm_file_unsupported_reason`, `lgbm_interop_status`) are all file-to-file or
file-to-string. Three reasons, in the order they mattered:

1. **The native format must not be weakened.** `serialize.mojo` stores floats
   as raw bit patterns, so a native round trip is bit-exact. LightGBM's is
   decimal text and structurally holds less. Making an import *land* in the
   native format makes the asymmetry the API's shape rather than a caveat in a
   docstring, and it means nothing in the codebase ever has a reason to keep a
   model in LightGBM's shape.
2. **The Python and C facades stay thin.** No model handle crosses the
   boundary, so `load_lightgbm_model` is `convert; Booster(model_file=...)`
   over machinery that already exists and is already tested.
3. **Serialization is exercised on the conversion path**, not separately.

The lower-level pairs (`parse_lgbm_model` / `dump_lgbm_model`,
`import_lgbm_model` / `export_lgbm_model`, and the `load_lgbm_*` /
`save_lgbm_*` file wrappers) are all still present and unchanged in behavior;
they are what the four entry points are built from and what a Mojo caller
holding a live `Model` still wants.

---

## 5. Duplicates fused or quarantined

**Fused (deleted from this module, now sourced from the registry):**

- The `multiclassova` / `multiclass_ova` / `ova` / `ovr` rejection and its
  hand-written message. `objective_registry.objective_unimplemented_reason`
  says it now, and says it for the other unimplemented objectives too, which
  this module never covered.
- The `lambdarank` special case. `objective_code_from_name` resolves it, so
  the "the registry refuses this name" workaround is gone.
- The write-direction objective name table (twelve branches). Now
  `objective_canonical_name` plus two documented overrides:
  `L1` writes `regression_l1` rather than the registry's `mae` (both resolve
  back to `L1`, but `regression_l1` is what LightGBM's own writer emits), and
  `BINARY_LOGISTIC` writes `binary sigmoid:1` so the file reads back without
  depending on LightGBM's default.
- The `.params.objective_from_name` dependency. This module now imports only
  `MULTICLASS` from `params.mojo`.
- Nine unused objective-code imports from `.boosting` and the whole
  `.ranking` import, which existed only to feed the deleted table.

**Not fused, deliberately:**

- `_MISSING_NONE` / `_MISSING_ZERO` / `_MISSING_NAN` and the `decision_type`
  bit masks stay local. They are LightGBM's file encoding, not mojoboost
  vocabulary, and nothing else in the repo should know them.
- `_LgbmTree` / `_LgbmOut` are LightGBM's node numbering, which is not
  mojoboost's `Tree` and must not become a second model representation. They
  are private, they exist only inside one conversion, and neither is returned
  from any public function.

**Corrected, not quarantined:** the prior handoff's reason for rejecting
categorical ("the code-to-bin table is not in the file... it can be
reconstructed, but only up to categories no split mentions, which changes
which unseen codes route where"). The second half is the error: a category no
split mentions routes right in LightGBM (its bit is unset everywhere) and
routes right in mojoboost (it maps to bin 0, never a set member). The two
agree, so nothing changes and nothing needed guessing.

---

## 6. Cross-lane patch requests

All exact. None applied.

### 6.1 Task 01 owns `src/mojoboost/__init__.mojo`

Add, keeping the file's alphabetical-by-module grouping (it currently sits
between `.interaction` and `.model` in import order; any position works, the
module is a leaf and imports nothing that imports it):

```mojo
from .lgbm_model_io import (
    LGBM_INTEROP_STATUS,
    LgbmExport,
    LgbmExportReport,
    LgbmImport,
    LgbmImportReport,
    LgbmMulticlassImport,
    dump_lgbm_model,
    dump_lgbm_multiclass_model,
    export_lgbm_file,
    export_lgbm_model,
    export_lgbm_multiclass_model,
    import_lgbm_file,
    import_lgbm_model,
    import_lgbm_multiclass_model,
    lgbm_dropped_objective_param,
    lgbm_file_unsupported_reason,
    lgbm_interop_status,
    lgbm_model_file_kind,
    lgbm_objective_code,
    lgbm_objective_name,
    lgbm_text_kind,
    lgbm_unsupported_reason,
    load_lgbm_model,
    load_lgbm_multiclass_model,
    parse_lgbm_model,
    parse_lgbm_multiclass_model,
    save_lgbm_model,
    save_lgbm_multiclass_model,
)
```

If a smaller surface is wanted, the four that matter are `import_lgbm_file`,
`export_lgbm_file`, `lgbm_file_unsupported_reason`, `lgbm_interop_status`.

### 6.2 Task 06 owns `bindings/_mojoboost.mojo`, Task 14 owns the narrow binding modules

Four functions. They take and return strings and a dict; no handle crosses.
Put them in a `bindings/model_io_bindings.mojo` (Task 14) and register them in
`_mojoboost.mojo` (Task 06), or put them straight in `_mojoboost.mojo`.

```mojo
from mojoboost.lgbm_model_io import (
    export_lgbm_file,
    import_lgbm_file,
    lgbm_file_unsupported_reason,
    lgbm_interop_status,
)


def lgbm_interop_status_binding() raises -> PythonObject:
    return PythonObject(lgbm_interop_status())


def lgbm_file_unsupported_reason_binding(
    path: PythonObject,
) raises -> PythonObject:
    return PythonObject(lgbm_file_unsupported_reason(String(py=path)))


def lgbm_import_file(
    lgbm_path: PythonObject, model_path: PythonObject
) raises -> PythonObject:
    var report = import_lgbm_file(
        String(py=lgbm_path), String(py=model_path)
    )
    var out = Python.dict()
    out["n_features"] = report.n_features
    out["n_trees"] = report.n_trees
    out["n_classes"] = report.n_classes
    out["objective"] = report.objective
    out["objective_line"] = report.objective_line
    out["format_version"] = report.format_version
    out["n_categorical_features"] = report.n_categorical_features
    out["n_widened_tables"] = report.n_widened_tables
    out["n_edges"] = report.n_edges
    out["n_missing_reservations"] = report.n_missing_reservations
    out["has_node_counts"] = report.has_node_counts
    out["mapper_is_training_binning"] = report.mapper_is_training_binning()
    var names = Python.list()
    for i in range(len(report.feature_names)):
        names.append(report.feature_names[i])
    out["feature_names"] = names
    return out


def lgbm_export_file(
    model_path: PythonObject, lgbm_path: PythonObject
) raises -> PythonObject:
    var report = export_lgbm_file(
        String(py=model_path), String(py=lgbm_path)
    )
    var out = Python.dict()
    out["n_trees"] = report.n_trees
    out["n_classes"] = report.n_classes
    out["objective_line"] = report.objective_line
    out["dropped_objective_param"] = report.dropped_objective_param
    out["n_categorical_features"] = report.n_categorical_features
    out["shrinkage_folded"] = report.shrinkage_folded
    out["base_score_folded"] = report.base_score_folded
    out["n_feature_names"] = report.n_feature_names
    out["predictions_bit_exact"] = report.predictions_bit_exact()
    return out
```

Registered as, alongside the existing `save` / `load` pair at
`bindings/_mojoboost.mojo:172-175`:

```mojo
m.def_function[lgbm_interop_status_binding]("lgbm_interop_status")
m.def_function[lgbm_file_unsupported_reason_binding](
    "lgbm_file_unsupported_reason"
)
m.def_function[lgbm_import_file]("lgbm_import_file")
m.def_function[lgbm_export_file]("lgbm_export_file")
```

**The Python names matter**: `python/mojoboost/lgbm_model_io.py` looks for
exactly `lgbm_interop_status`, `lgbm_file_unsupported_reason`,
`lgbm_import_file`, `lgbm_export_file`, and names any missing one in its
error. A build without them raises `LightGBMInteropUnavailable`; it does not
degrade to a Python conversion, by design.

The report dict is best-effort on the binding side: the Python glue accepts a
mapping and falls back to a one-key `{"summary": ...}` for anything else, so a
binding that cannot build a dict yet is still usable.

### 6.3 Task 07 owns `python/mojoboost/__init__.py`

Lazy, experimental, no import cost:

```python
def __getattr__(name):
    if name == "lgbm_model_io":
        from . import lgbm_model_io
        return lgbm_model_io
    raise AttributeError(name)
```

Do **not** re-export `load_lightgbm_model` / `save_lightgbm_model` at the top
level yet, and do not add them to `Booster`. Both would read as a supported
save path, which is exactly the confusion this lane is trying to prevent:
`Booster.save_model` is mojoboost's format and is bit-exact, LightGBM's is an
interchange. `mojoboost.lgbm_model_io.<fn>` keeps the distinction visible at
the call site. Revisit after the differential fixtures in section 9 pass.

### 6.4 Task 19 owns `docs/LIGHTGBM_PARITY.md`

Three places are now stale (line numbers as of this writing; the file is
being edited concurrently).

Line 126 currently reads, in part: `deferred | ... | No caller, no export, and
no test reads a file LightGBM actually wrote`. Replace the trailing note with:

> `src/mojoboost/lgbm_model_io.mojo` with `tests/parallel/test_lgbm_model_io.mojo`
> in `pixi run test`. Experimental and quarantined behind
> `mojoboost.lgbm_model_io`; the native format (`serialize.mojo`) is unaffected
> and remains the only persistence format. Numerical splits, categorical sets,
> default directions, missing-value routing, leaf values, tree and class order,
> base scores, feature names, and supported objectives convert; linear trees,
> `missing_type=Zero` on a numerical split, `average_output`, unimplemented
> objectives, non-unit `binary sigmoid`, custom objectives, and any feature
> needing more than 256 bins or 255 categories raise by name. No test reads a
> file LightGBM actually wrote.

Line 290 currently says lgbm_model_io "is an unintegrated experiment in reading
LightGBM's format". Replace with "is an experimental import and export path for
LightGBM's format (`mojoboost.lgbm_model_io`), which converts *into* and *out
of* the mojoboost format rather than replacing it". That row also says
serialize.mojo is "now at v3"; it is at v4 since Task 11 added feature names.

Line 617 lists `src/mojoboost/lgbm_model_io.mojo` among modules "connected to
nothing". It stays on that list until the export in 6.1 and the bindings in
6.2 land, and comes off it then, not now.

Do not claim arbitrary LightGBM model compatibility anywhere. The rejection
list above is the honest scope.

### 6.5 Nobody owns `tests/parallel/test_lgbm_model_io.mojo`

It is already in the `pixi run test` chain (`pixi.toml:9`), so no `pixi.toml`
change is needed.

Static review says the existing 29 tests stay source-compatible and should
still pass; this was not run (see section 9). Every public symbol it imports
still exists with a compatible signature (the three new parameters are all
defaulted), and each of the four rejection tests that touch changed code still
raises, for a better-stated reason:

| Test | Now raises because |
| --- | --- |
| `test_rejects_categorical_decision_type` | node marked categorical, tree declares `num_cat=0` |
| `test_rejects_num_cat` | `num_cat=1` with no `cat_boundaries` |
| `test_rejects_writing_a_categorical_node` | the model's mapper has no category table for that feature |
| `test_objective_mapping_and_its_rejections` | the registry's messages, same four refusals |

No test asserts on error text (`assert_raises` is used without `contains=`),
which is what makes the message changes safe.

Whoever owns tests next should add, in one file: a categorical read with
hand-computed routing including an unseen code; a categorical write and
reparse; the mixed numerical/categorical rejection; the >255-category
rejection; feature names surviving both directions; and the `n_widened_tables`
difference between a file with and without a categorical `feature_infos`.

---

## 7. Fallbacks preserved

- **Every previously working path is unchanged.** `parse_lgbm_model`,
  `parse_lgbm_multiclass_model`, `load_lgbm_model`, `load_lgbm_multiclass_model`,
  `dump_lgbm_model`, `dump_lgbm_multiclass_model`, `save_lgbm_model`,
  `save_lgbm_multiclass_model`, `lgbm_text_kind`, `lgbm_model_file_kind`,
  `lgbm_objective_code`, `lgbm_objective_name` all keep their signatures and
  behavior. The three new parameters are defaulted; the new functions are
  additions.
- **The category table has a conservative fallback.** `feature_infos` widening
  is attempted, and the split-code union is used whenever the header carries no
  usable list, the count does not match the feature count, an entry does not
  parse as codes, or the widened table would not fit. Both tables are exact;
  the fallback only gives up faithfulness to LightGBM's fitted table, never
  correctness.
- **Feature names have a conservative fallback.** Wrong count means no names,
  which means `Column_i` on write and an unnamed native file on import.
- **The Python facade has no fallback, on purpose.** A build without the
  bindings raises `LightGBMInteropUnavailable` naming the missing entry point.
  Converting in Python would be the duplicate implementation this lane exists
  to avoid.
- **`lgbm_unsupported_reason` never raises**, so a caller can probe a file
  without exception handling even when the answer is "no".

---

## 8. Serialization and public-API effects

- **`serialize.mojo` is untouched and unaffected.** No format version change,
  no new field, no new section. This module reads its public functions and
  writes nothing into it.
- **An imported model serializes through the existing v4 path**, categorical
  spec and per-node bitsets included (`serialize.mojo` has written both since
  v2/v3). `import_lgbm_file` is the proof: it fails at `save_model` if
  anything the conversion produced cannot be written.
- **Native public API: unchanged.** No existing symbol changed meaning.
- **New Mojo public symbols** (all in `mojoboost.lgbm_model_io`, none exported
  from the package yet): `LGBM_INTEROP_STATUS`, `LgbmImportReport`,
  `LgbmExportReport`, `LgbmImport`, `LgbmMulticlassImport`, `LgbmExport`,
  `import_lgbm_model`, `import_lgbm_multiclass_model`, `export_lgbm_model`,
  `export_lgbm_multiclass_model`, `import_lgbm_file`, `export_lgbm_file`,
  `lgbm_unsupported_reason`, `lgbm_file_unsupported_reason`,
  `lgbm_interop_status`, `lgbm_dropped_objective_param`.
- **New Python module** `mojoboost.lgbm_model_io`, importable today and
  functional once the bindings land. Importing it costs one `import os`, one
  `import tempfile`, one `import warnings`; the extension is imported inside
  the calls.
- **`EXPERIMENTAL = True`** is a module attribute so a caller can gate on it,
  and the first conversion in a process emits one `UserWarning` carrying
  `interop_status()`.

---

## 9. Risks

Ordered by how much they should worry the next person.

1. **Nothing was run.** No Mojo, no Python, no build, no test, per the lane's
   instructions. Every claim here is from static reading. The module has not
   been compiled since the changes, and Mojo signature drift in the
   concurrently-edited files it imports would surface as a compile error, not
   as anything subtler.
2. **Still no differential validation against real LightGBM.** This was the
   top risk in the prior handoff and it is unchanged. Every fixture is
   hand-written from the format's specification. The categorical reasoning in
   section 3 is from LightGBM's source semantics, carefully, and has never been
   checked against a file LightGBM wrote. Categorical is the newest and least
   checked part, so check it first.
3. **Concurrent-lane dependencies.** This module now imports
   `objective_registry.{objective_code_from_name, objective_canonical_name,
   objective_param_name}` (Task 08 is editing that file) and
   `serialize.{load_feature_names, load_model, load_multiclass_model,
   model_file_kind, save_model, save_multiclass_model}` (Task 11 is editing
   that one). All six were verified present at the time of writing.
   `load_feature_names` and the `feature_names` parameter on `save_model` are
   hours old; if Task 11 renames either, this module and only this module
   breaks, in `import_lgbm_file` / `export_lgbm_file` / the import block.
4. **`Common::ConstructBitset` word width is assumed to be 32.** It is
   `uint32_t` in LightGBM. If a future format version widens it, the reader
   would decode the wrong codes rather than fail. There is no version marker
   for it in the file, so this cannot be checked defensively; it is asserted by
   `_CAT_WORD_BITS` and stated here.
5. **Objective scalar parameters still do not survive an export.** `huber` and
   `quantile`'s `alpha`, `fair`'s `fair_c`, `tweedie`'s
   `tweedie_variance_power`: a fitted `Booster` does not keep them, so the
   written `objective=` line omits them and a LightGBM reader supplies its own
   default. Predictions are unaffected (no objective's inverse link reads its
   scalar). `LgbmExportReport.dropped_objective_param` names the one that went
   missing instead of writing a number the model was probably not trained with.
   If a later lane adds the scalar to `Booster`, wire it into
   `lgbm_objective_name` and drop the field.
6. **`leaf_weight` and `internal_weight` are still written as zeros.**
   mojoboost keeps node covers, not hessian sums. LightGBM treats both as
   optional on load; a consumer that reads them sees zeros.
7. **`tree_sizes` is computed but unverified.** It is the exact byte length of
   each emitted block, which is what LightGBM's fast loader wants, and no
   LightGBM has ever read it.
8. **Tree conversion recurses once per node**, both directions. Depth is
   bounded by tree depth; a pathological single-spine tree with thousands of
   leaves could reach a stack limit. Real leaf-wise models do not. Unchanged
   from before.
9. **`import_lgbm_file` parses the LightGBM text once and
   `export_lgbm_file` reads the native file three times** (names, kind, model).
   Correct, and wasteful on very large models. A one-shot conversion, so this
   was left alone rather than restructured across a lane boundary.
10. **`lgbm_unsupported_reason` performs the whole conversion** to answer.
    Calling it and then converting does the work twice. Documented on the
    function; the alternative is a second list of supported constructs, which
    is worse.

---

## 10. Remaining disconnections

- Not exported from `src/mojoboost/__init__.mojo` (6.1).
- No bindings, so `python/mojoboost/lgbm_model_io.py` raises
  `LightGBMInteropUnavailable` on every call in a current build (6.2).
- Not reachable from `mojoboost/__init__.py` (6.3).
- No C API and no CLI subcommand. Deliberate: Task 21 owns `capi/` and `cli/`,
  and an experimental path should not gain a stable C ABI before it has been
  validated. When it does, `import_lgbm_file` / `export_lgbm_file` /
  `lgbm_file_unsupported_reason` are the three to bind, and they need no
  handle type.
- `docs/LIGHTGBM_PARITY.md` still describes the module as unintegrated (6.4).
- No test covers any of the new behavior (6.5).
- `python/mojoboost/inspection.py`'s `OBJECTIVE_NAMES` remains a Python copy of
  the objective-code table. Not this lane's file, and now one more consumer
  (`lgbm_import_file`'s report carries an objective code) that would prefer the
  registry to reach Python. Task 08's registry binding is the fix.

---

## 11. Smallest later focused commands, all UNRUN

Nothing in this section was executed. One at a time, in this order.

```
# 1. Does it compile and does the existing suite still pass?
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_lgbm_model_io.mojo

# 2. Python module imports cleanly and reports the missing binding by name.
pixi run python -c "from mojoboost import lgbm_model_io as m; \
print(m.EXPERIMENTAL); \
import contextlib; \
exec('try:\n m.interop_status()\nexcept m.LightGBMInteropUnavailable as e:\n print(e)')"

# 3. Only after the bindings in 6.2 land: a real LightGBM file, both ways.
#    This is the differential check risk 2 is about. It needs a lightgbm
#    install, which this lane did not have.
pixi run python - <<'PY'
import lightgbm, numpy as np, mojoboost
from mojoboost import lgbm_model_io
X = np.random.rand(400, 4); y = (X[:, 0] > 0.5).astype(float)
X[:, 3] = np.random.randint(0, 6, 400)          # a categorical column
d = lightgbm.Dataset(X, y, categorical_feature=[3])
b = lightgbm.train({"objective": "binary", "num_leaves": 7}, d, 10)
b.save_model("lgbm.txt")
print(lgbm_model_io.unsupported_reason("lgbm.txt") or "convertible")
print(lgbm_model_io.convert_to_mojoboost("lgbm.txt", "converted.mbst"))
ours = mojoboost.Booster(model_file="converted.mbst")
print(np.abs(np.asarray(b.predict(X)) - np.asarray(ours.predict(X))).max())
PY
```

Step 3 is the one that turns `LGBM_INTEROP_STATUS` from "hand-written fixtures
only" into something else. Until it runs, keep the word experimental on every
surface that mentions this module.

`git diff --check --` over the two owned source paths reports no whitespace
errors. Nothing was committed.
