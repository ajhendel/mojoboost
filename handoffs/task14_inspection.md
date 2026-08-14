# Task 14 handoff: model inspection and the stable dump schema

Everything below is work that has to happen in files this task was not
allowed to touch. Nothing here has been applied; nothing has been staged
or committed.

## What this task added

| File | What it is |
| --- | --- |
| `docs/MODEL_INSPECTION_SCHEMA.md` | the normative schema; both implementations answer to it |
| `python/mojoboost/inspection.py` | the working consumer: builds the schema by parsing `Booster.model_to_string()` |
| `src/mojoboost/inspection.mojo` | the native producer: builds the same schema from `Model` / `MulticlassModel` |
| `python/tests/parallel/test_inspection.py` | 30 tests over the schema, the derived shapes, and the attribute wiring |

## What was run, and what was not

`pixi run -e pytest pytest -q python/tests/parallel/test_inspection.py`
was run against the `_mojoboost.so` already in the working tree: **30
passed**. It ran twice rather than once, because three lines were
reformatted to the repository's 79 column limit after the first pass and
reporting a result for code that had since changed would have been a false
claim; the second run confirms the code as it stands. No build was
triggered: the task depends on `build-python`, so the task was bypassed
and `pytest` invoked directly in the `pytest` environment.

The five assigned paths carry no trailing whitespace, no tabs, and no line
over 79 columns. `git diff --check` reports nothing on them, though that
is vacuous here since all five files are new and untracked.

`src/mojoboost/inspection.mojo` **has not been compiled**. This task's
budget was one focused test, and the assigned test is the Python one; no
Mojo test file was in scope to add or run. Treat that module as unbuilt
code: compile it before wiring anything to it. Nothing else in the tree
imports it, so it cannot break an existing build until it is exported.

The Python module's `split_gains` branch is likewise unexercised, because
the hook it looks for does not exist yet. Everything else in
`inspection.py` is covered by the test file.

## Why the Python side parses text instead of calling into Mojo

The bindings expose no model-structure entry point today, and
`bindings/_mojoboost.mojo` was out of scope. `Booster.model_to_string()`
already returns the whole model in mojoboost's versioned save format, with
floats as IEEE-754 bit patterns, so parsing it is exact and needs no new
native surface. That is the entire reason for the parser in
`inspection.py`, and it is why inspection works today rather than after
the bindings land.

It has exactly one gap, and it is the save format's: **split gains are not
serialized** (see `src/mojoboost/serialize.mojo`), so every node's
`split_gain` is `None` and the dump reports `has_split_gain: False`. Two
ways to close it are below; the hook is the cheaper one.

## 1. Mojo exports

`src/mojoboost/__init__.mojo`, alongside the existing
`from .importance import gain_importance, split_importance` line:

```mojo
from .inspection import (
    DUMP_FORMAT_VERSION,
    MODEL_FORMAT_VERSION,
    category_bins,
    category_codes,
    dump_model,
    dump_multiclass_model,
    has_threshold,
    node_depths,
    split_gains,
    split_ordinals,
    threshold_value,
)
```

Name collision to check first: `dump_model` is a new top-level name and
`split_gains` is close to `split_importance`; neither collides with an
existing export as of this writing, but the package `__init__` is edited
by several lanes this round, so re-check before adding.

`src/mojoboost/inspection.mojo` imports only `BinMapper`, `Model`,
`MulticlassModel`, `MonotoneConstraints`, `Tree`, `CAT_MAX_BINS`,
`cat_pool_contains`, and `std.math.isnan`. It adds no dependency to
anything and nothing imports it.

## 2. Binding functions

`bindings/_mojoboost.mojo`. Both pairs follow the existing
one-entry-point-per-model-kind convention (`n_features` /
`n_features_multiclass`, `feature_importance` /
`feature_importance_multiclass`), because a handle is a `Model` or a
`MulticlassModel` and the Python layer already knows which it holds.

The small hook, and the only one that adds information the model text
lacks:

```mojo
def split_gains(model: PythonObject) raises -> PythonObject:
    """Per node split gains, one list per tree. Gains are recorded during
    growth and are not serialized, so this is the only way a consumer can
    have them."""
    var m = model.downcast_value_ptr[Model]()
    return _gain_lists(m[].booster.trees)


def split_gains_multiclass(model: PythonObject) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return _gain_lists(m[].booster.trees)


def _gain_lists(trees: List[Tree]) raises -> PythonObject:
    var out = Python.list()
    for t in range(len(trees)):
        var row = Python.list()
        for i in range(len(trees[t].split_gain)):
            row.append(PythonObject(trees[t].split_gain[i]))
        out.append(row^)
    return out^
```

The whole-schema hook, for consumers outside Python:

```mojo
def dump_model(
    model: PythonObject, feature_names: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(inspection_dump_model(m[], _names(feature_names)))


def dump_multiclass_model(
    model: PythonObject, feature_names: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(
        inspection_dump_multiclass_model(m[], _names(feature_names))
    )
```

with `_names` converting a Python sequence of strings to `List[String]`
and returning an empty list for `None` (the module falls back to
`Column_0`, `Column_1`, ... on any length mismatch, so `_names` does not
have to validate). Import the Mojo functions under aliases, since the
binding functions share their names:

```mojo
from mojoboost.inspection import (
    dump_model as inspection_dump_model,
    dump_multiclass_model as inspection_dump_multiclass_model,
    split_gains as inspection_split_gains,
)
```

Registration lines in `PyInit__mojoboost`, next to the existing
`feature_importance` pair:

```mojo
m.def_function[split_gains]("split_gains")
m.def_function[split_gains_multiclass]("split_gains_multiclass")
m.def_function[dump_model]("dump_model")
m.def_function[dump_multiclass_model]("dump_multiclass_model")
```

`python/mojoboost/inspection.py` picks up `split_gains` /
`split_gains_multiclass` the moment they exist, with no change on the
Python side: `_native_split_gains` looks them up by name, `source` becomes
`"model_to_string+split_gains"`, and `has_split_gain` becomes `True`. It
does **not** call `dump_model`; that hook is for the C ABI and the CLI.

Contract the hook must satisfy, since nothing checks it at build time:
one list per tree, in tree order, each as long as that tree's node array,
a leaf's entry `0.0`. `dump_model` raises if the tree count is wrong; a
wrong node count would surface as an `IndexError` on that tree.

## 3. Python package surface

### `python/mojoboost/basic.py`

`Booster` is where LightGBM puts all three of these. Each is a delegation,
so there is one implementation and not two:

```python
    def dump_model(self):
        """The model as the inspection schema (see
        docs/MODEL_INSPECTION_SCHEMA.md). LightGBM's `dump_model`, with
        mojoboost's own schema rather than LightGBM's."""
        from .inspection import dump_model

        return dump_model(self)

    def trees_to_dataframe(self):
        from .inspection import trees_to_dataframe

        return trees_to_dataframe(self)

    def get_split_value_histogram(
        self, feature, bins=None, as_frame=False
    ):
        from .inspection import get_split_value_histogram

        return get_split_value_histogram(
            self, feature, bins=bins, as_frame=as_frame
        )
```

Import inside the method, the way `_objectives()` already defers its
import, so `basic` and `inspection` do not form a cycle.

The module docstring's "Differences from LightGBM" list has a bullet that
is now wrong and should go:

```
- **No `dump_model` / `trees_to_dataframe`.** Structured model inspection is
  its own piece of work; `model_to_string()` gives the whole model in
  mojoboost's versioned text format meanwhile.
```

Replace it with a bullet naming the two real deviations, both stated in
the schema doc: the split value histogram's return type is chosen by
`as_frame` rather than by whether pandas can be imported, and leaf editing
is not offered.

### `python/mojoboost/__init__.py`

Re-export the module so `mojoboost.inspection` resolves without an
explicit submodule import, and add the three attributes that do not exist
yet. Add `"inspection"` to `__all__`.

## 4. The six attributes

`inspection.py` implements each as a free function so that wiring it on is
a delegating property and not a second source of truth.

| Attribute | State today | Wiring |
| --- | --- | --- |
| `booster_` | **exists** on `_Base` | no change. `inspection.booster_of` only normalizes an estimator and a `Booster` to the same thing |
| `n_iter_` | **exists**, set by `_record_fit` | no change. `inspection.n_iter_of` reads the recorded value and falls back to `current_iteration()` for a bare `Booster` |
| `best_score_` | **exists**, set by `_fit_with_metrics` | no change. `inspection.best_score_of` reads it and raises `AttributeError` naming `eval_set` when the fit had no validation set |
| `feature_name_` | missing | add to `_Base`: `@property def feature_name_(self): return inspection.feature_name_of(self)`. Reads `booster_.feature_name()`, so it is `Column_0`, ... when the model carries no names |
| `n_features_` | missing | add to `_Base`: `@property def n_features_(self): return inspection.n_features_of(self)`. Reads the model, not `n_features_in_` |
| `objective_` | missing | add to `_Base`: `@property def objective_(self): return inspection.objective_of(self)`. Resolved from the objective code the model carries, so `objective="mae"` reports `"regression_l1"` and a `Booster` read from a file answers too |

All three new ones are read-only properties, so they must **not** go in
`_Base._FITTED_ATTRS`, which lists attributes `fit` assigns and a refit
deletes. They raise `NotFittedError` before a fit, through `booster_`,
which is the behavior the class already documents.

`objective_of` costs one `model_to_string()` round trip per call. If that
matters, cache it on the estimator at `_record_fit` time; the test suite
does not, so it is not cached today.

## 5. `docs/LIGHTGBM_PARITY.md`

Five rows currently point at task 14 and are now answerable. The file is
checked by `tools/check_parity.py` (`pixi run check-parity`, its own CI
job), so edit it in the same change as the wiring, not before.

| Line | Row | Now |
| --- | --- | --- |
| 96 | `Booster` | drop "`dump_model` is the gap, task 14"; name `dump_model`, `trees_to_dataframe`, and `get_split_value_histogram` as supported |
| 110 | `plot_split_value_histogram` | still `deferred` for the plot itself, but the data is `get_split_value_histogram`; point at `docs/MODEL_INSPECTION_SCHEMA.md`, not at task 14 |
| 192 | `objective_` | `deferred` becomes `supported`, resolved from the model's objective code |
| 194 | `feature_name_` | the row says a second attribute alongside `feature_names_in_` is not added. That decision is now reversed: `feature_name_` exists and reports LightGBM's `Column_i` fallback, which `feature_names_in_` does not. Rewrite the row rather than leaving it contradicting the code |
| 222, 228 | `Booster.dump_model` / `trees_to_dataframe`, `get_split_value_histogram` | `deferred` becomes `supported`, with the two deviations named: no `weight` column, and `as_frame` instead of a pandas-presence switch. `lower_bound` / `upper_bound` on line 228 are **not** done and stay deferred |

Line 68's "model inspection, dumping, and editing" in the not-yet-covered
list should keep "editing" and drop the rest.

## 6. Serialization wiring

**Recommended: no format change.** Ship the `split_gains` hook. Gains are
a property of how a tree was grown rather than of what it computes, they
are not needed to predict, and adding them to the file would grow every
saved model by one float per node for a fact only inspection reads.
`docs/MODEL_INSPECTION_SCHEMA.md` already gives consumers
`has_split_gain` to branch on, and `importance.mojo` has documented the
same absence since it was written.

If gains are serialized anyway, the change is a v4 in
`src/mojoboost/serialize.mojo` and it must follow the pattern v3 set:

- bump `_VERSION` to `"v4"` and add the history entry.
- write `tree.split_gain` in `_write_trees`, after the v3 `count` block,
  as bit patterns like every other float.
- read it in `_read_trees` under `if version >= 4`, leaving the list empty
  for older files so a v1, v2, or v3 model still loads and predicts
  exactly as it does now.
- `Tree.__init__` already fills a wrong-length optional array with its
  default, so an empty `split_gain` needs no new branch. Note that unlike
  `count` it must **not** be validated as positive: a leaf's gain is 0.0.
- `python/mojoboost/inspection.py` needs `4` added to
  `SUPPORTED_MODEL_FORMAT_VERSIONS` and one `if version >= 4` block in
  `_parse_trees`; the schema itself does not change, only which source can
  fill `split_gain`.
- `gain_importance` in `src/mojoboost/importance.mojo` would start
  reporting nonzero gains for a loaded model, and its docstring and
  `Booster.feature_importance`'s both say the opposite today. Both need
  updating, and `python/tests/test_attributes.py` should be checked for a
  test that asserts the current zero.

## 7. Test wiring

`python/tests/parallel/test_inspection.py` sits under `python/tests`, so
`pixi run -e pytest test-estimators` and the CI job that runs it already
collect it. `python/tests/conftest.py` supplies its `regression` and
`multiclass` fixtures from the parent directory. Nothing to add.

The tests skip `trees_to_dataframe` and the `as_frame` histogram when
pandas is absent, so they pass in an environment without it.

If the `split_gains` hook lands, add to that file: `has_split_gain` is
True, `source` ends in `+split_gains`, every internal node's `split_gain`
is finite and nonnegative, every leaf's is `0.0`, and the per-feature sums
equal `booster_.feature_importance("gain")`. That last one ties the dump
to an existing, independently computed number and is the check worth
having.

## 8. Overlap with the concurrent LightGBM model IO lane

`src/mojoboost/lgbm_model_io.mojo` landed in the same parallel round and
carries `lgbm_objective_name(objective: Int) raises -> String`, which maps
the same objective codes this task's `OBJECTIVE_NAMES` table maps. The two
were checked against each other and agree on every code except two, in
both cases **deliberately**. Do not collapse them into one table without
reading this:

| Code | `lgbm_model_io.lgbm_objective_name` | `inspection.OBJECTIVE_NAMES` | Why they differ |
| --- | --- | --- | --- |
| 1 | `"binary sigmoid:1"` | `"binary"` | theirs is a LightGBM `objective=` **file line**, parameters included; mine is the value of the `objective_` **attribute**, which LightGBM spells `"binary"` |
| 6 | raises | `"custom"` | a custom objective has nowhere to go in a LightGBM file, so raising is right there; `objective_` still has to answer, and a model's own text records the code |

Every other code matches exactly: `regression`, `poisson`, `huber`,
`quantile`, `regression_l1`, `gamma`, `tweedie`, `mape`, `fair`,
`cross_entropy`, `lambdarank`.

If the two are ever unified, the shared thing is the code to canonical
name map, and each caller keeps its own decoration: the file writer
appends `sigmoid:1` and raises on custom, the attribute does neither.
`test_objective_names_track_the_estimator_table` already pins the Python
side against `MojoBoostRegressor._OBJECTIVES`, so a drift there fails
rather than passing quietly.

One naming collision to avoid: that lane's `_WRITE_VERSION = "v4"` is the
**LightGBM** format version and has nothing to do with the mojoboost save
format v4 discussed in section 6. Two different v4s.

## 9. The one unverified Mojo construct

`src/mojoboost/inspection.mojo` is uncompiled, and its APIs were audited
against confirmed usage elsewhere in the tree. Everything has in-repo
precedent (`List.copy()`, `List[T](capacity=)`, `.resize(n, default)`,
`var stack: List[Int] = [0]`, `.pop()`, `ref tree = trees[t]`,
`mut out: String`, `isnan`, `cat_pool_contains`, bare `def` recursion)
**except one**:

- `_json_string` iterates with `for cp in s.codepoint_slices()` and
  materializes each with `String(cp)`. That is the documented current
  string-iteration API, but nothing else in this repository uses it, so it
  is the first place to look if the module does not compile. The fallback
  if it misbehaves is to escape by byte, the way
  `serialize.mojo::_parse_u64` already walks `token.as_bytes()`.

Everything else was brought into line with the surrounding files rather
than left as a note: the three ternaries inside `out += ...` are now
parenthesized the way `serialize.mojo` writes them (`out += ("1 " if ...
else "0 ")`).

## 10. Deliberate omissions

- **Leaf editing.** Not offered, and `docs/MODEL_INSPECTION_SCHEMA.md`
  states the three invariants that make an arbitrary edit undetectable:
  node covers, internal node values, and the sums gains were computed
  from. `test_leaf_editing_is_not_offered` pins the absence so adding it
  becomes a deliberate act with a place to state its invariants.
- **Plotting.** No dependency was added and none is needed.
  `get_split_value_histogram` returns the data; the plot is the caller's.
- **`Booster.lower_bound` / `upper_bound`.** Not part of this task and
  still deferred in the parity doc. They are cheap once the dump exists
  (the min and max leaf value over the ensemble), but they are a separate
  claim and were not tested here.
