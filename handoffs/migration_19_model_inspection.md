# Task 19 handoff: model inspection and tree traversal, moved into Mojo

Everything below is work that has to happen in files this task was not
allowed to touch. None of it has been applied. No build, test, formatter,
or Python process was run: this task is static reasoning only, and the code
it produced is **uncompiled**.

This task committed nothing. A repository-wide commit from another lane
(`9a9c8d1`) swept the tree while this one was working, so
`src/mojoboost/model_dump.mojo` and `src/mojoboost/inspection.mojo` are
already in history; `python/mojoboost/inspection.py` is modified and
uncommitted, and this file is untracked. Nothing in the tree was staged or
committed on purpose here, and no other lane's work was touched.

It supersedes sections 1, 2, and 6 of `handoffs/task14_inspection.md` and
leaves that file's sections 3, 4, 5, 7, 8, and 10 standing. Read this one
first; where the two disagree, this one is the later decision and says why.

## 1. What this task changed

| File | Before | Now |
| --- | --- | --- |
| `src/mojoboost/model_dump.mojo` | did not exist | the schema as Mojo data: `ModelDump`, `DumpTree`, `DumpNode`, `DumpFeature`, the two builders, the tree primitives, and the dump interpreter |
| `src/mojoboost/inspection.mojo` | built the schema *and* wrote JSON | writes JSON, and nothing else. Every fact it prints comes from `model_dump.mojo` |
| `python/mojoboost/inspection.py` | parsed `model_to_string()` and rebuilt the schema in Python | a facade: it calls the native dump, nests the flat node tables, and derives records / DataFrame / histogram. The parser survives below one banner, marked for deletion |
| `handoffs/migration_19_model_inspection.md` | did not exist | this file |

Nothing else was touched. The tree carries many other lanes' changes at the
same time; these four paths are this task's whole footprint.

## 2. Where each fact is computed now

The rule this task applied: anything that requires knowing how a mojoboost
model is laid out is computed in Mojo, once. Python converts.

| Fact | Where it is computed | Notes |
| --- | --- | --- |
| node depths, parents | `model_dump.node_depths`, `node_parents` | depth in edges from the root, root 0 |
| leaf ordinals, split ordinals | `Tree.leaf_ordinals`, `model_dump.split_ordinals` | the ordinals `predict(pred_leaf=True)` reports |
| numerical threshold from a split bin | `model_dump.threshold_value` | the bin's upper edge, the exact routing boundary |
| whether a split bin has a threshold at all | `model_dump.has_threshold` | false for categorical, and for a bin with no upper edge |
| category bins and their raw codes | `model_dump.category_bins`, `category_codes` | bin `i + 1` is `categories[i]` |
| per feature binning, missing bin, monotone sign | `model_dump._build_features` | one `DumpFeature` per feature |
| `has_split_gain`, `has_node_count` | `model_dump.has_split_gains`, `has_node_counts` | read off the trees, not assumed |
| the whole schema | `model_dump.build_dump`, `build_multiclass_dump` | one description, two entry points |
| routing a raw row through the dump | `model_dump.dump_bin_value`, `dump_goes_left`, `dump_leaf_node` | see §2.1 |
| raw scores from the dump | `model_dump.dump_raw_scores` | `base_score[k] + sum(shrinkage * leaf_value)` |
| split value collection | `model_dump.dump_split_values` | depth first per tree, root first |
| JSON rendering | `inspection.dump_json` | nested `tree_structure`, as the schema doc documents |
| objective **code** to LightGBM **name** | `python/mojoboost/inspection.OBJECTIVE_NAMES` | deliberate, see §12 |
| nested dicts, records, DataFrame, histogram buckets | `python/mojoboost/inspection.py` | presentation only |

### 2.1 The dump interpreter is deliberately a second implementation

`dump_raw_scores` and `dump_leaf_index` bin with the dump's own
`bin_upper_bounds` and route with the dump's own node records. They never
call `BinMapper.bin_value` or `Tree.goes_left`.

That is the point. The claim worth testing is "walking the dump lands where
the model lands", and a wrapper around the model's own prediction path
cannot fail that claim, so it would not be worth testing. A dump whose
thresholds were converted wrongly scores differently from
`Model.predict_raw`, and this is what makes that visible.

The duplication is ~40 lines of Mojo, in one file, next to what it mirrors.
It replaces the same duplication in Python, where it was three files away
from the code it had to agree with.

## 3. Mojo exports

`src/mojoboost/__init__.mojo`, alongside the existing
`from .importance import gain_importance, split_importance` line. Note the
primitives now come from `.model_dump`, not from `.inspection`:

```mojo
from .model_dump import (
    DUMP_FORMAT_VERSION,
    MODEL_FORMAT_VERSION,
    DumpFeature,
    DumpNode,
    DumpTree,
    ModelDump,
    build_dump,
    build_multiclass_dump,
    category_bins,
    category_codes,
    dump_bin_value,
    dump_goes_left,
    dump_leaf_index,
    dump_leaf_node,
    dump_raw_scores,
    dump_split_values,
    has_node_counts,
    has_split_gains,
    has_threshold,
    node_depths,
    node_parents,
    split_ordinals,
    threshold_value,
)
from .inspection import (
    dump_json,
    dump_model,
    dump_multiclass_model,
    split_gains,
)
```

Name collisions to check before adding, since several lanes edit this file
this round:

- `split_gains` (inspection) against `split_importance` (importance): close,
  not colliding.
- `has_node_counts` is a free function here and a **method** on `Tree`;
  those do not collide, but do not confuse them when reading a diff.
- `node_depths` / `split_ordinals` were named in the task 14 handoff as
  `.inspection` exports. They moved. `inspection.mojo` re-imports them, so
  `from .inspection import node_depths` still resolves; import from
  `.model_dump` anyway, so the package `__init__` names their real home.
- `dump_model` is a top-level name in three places now, all deliberate: the
  Mojo JSON function, the binding function, and the Python facade function.

`model_dump.mojo` imports `BinMapper`, `Model`, `MulticlassModel`,
`MonotoneConstraints`, `Tree`, `CAT_MAX_BINS`, `UNKNOWN_BIN`,
`cat_pool_contains`, and `std.math.isnan`. `inspection.mojo` imports
`model_dump`, `Model`, `MulticlassModel`, `Tree`, and `isnan`. Neither adds
a dependency to anything that did not already have one, and nothing else in
`src/` imports either module.

## 4. Binding functions

All in `bindings/_mojoboost.mojo`. Every pair follows the existing
one-entry-point-per-model-kind convention (`n_features` /
`n_features_multiclass`), because a handle is a `Model` or a
`MulticlassModel` and the Python layer already knows which it holds. Note
the suffix order: this task uses `dump_model_multiclass`, not task 14's
`dump_multiclass_model`, so it matches `feature_importance_multiclass` and
`predict_leaf_multiclass`.

Wave 1 is what `python/mojoboost/inspection.py` needs to stop parsing text
at all. Waves 2 and 3 are separable and can land later; the facade falls
back cleanly without them.

### Imports

```mojo
from mojoboost.model_dump import (
    DumpFeature,
    DumpNode,
    DumpTree,
    ModelDump,
    build_dump as inspection_build_dump,
    build_multiclass_dump as inspection_build_multiclass_dump,
    dump_leaf_index as inspection_dump_leaf_index,
    dump_raw_scores as inspection_dump_raw_scores,
    dump_split_values as inspection_dump_split_values,
)
from mojoboost.inspection import (
    dump_json as inspection_dump_json,
    split_gains as inspection_split_gains,
)
```

### Shared helpers

```mojo
def _py_dict() raises -> PythonObject:
    """An empty Python dict.

    Built through `builtins` rather than a `Python.dict()` constructor: the
    module builder already imports Python objects this way everywhere else,
    and `builtins.dict` exists in every CPython this can be built against.
    If the pinned Mojo exposes `Python.dict()`, use that instead; nothing
    else here changes.
    """
    return Python.import_module("builtins").dict()


def _dump_names(
    feature_names: PythonObject, n_names: PythonObject
) raises -> List[String]:
    """Feature names from a Python sequence, with its length passed
    alongside as every other sequence at this boundary is (see
    `_parse_valid_sets`). An empty list is what makes the builder fall back
    to `Column_0`, `Column_1`, ...; the Python layer has already rejected a
    wrong-length list by then, so this does not validate."""
    var out = List[String]()
    for i in range(Int(py=n_names)):
        out.append(String(py=feature_names[i]))
    return out^


def _py_int_list(values: List[Int]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):
        out.append(PythonObject(values[i]))
    return out^


def _py_f64_list(values: List[Float64]) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):
        out.append(PythonObject(values[i]))
    return out^
```

### Wave 1: the dump itself

```mojo
def _py_feature(info: DumpFeature) raises -> PythonObject:
    """One feature record, already in the schema's shape."""
    var out = _py_dict()
    out["index"] = PythonObject(info.index)
    out["name"] = PythonObject(info.name)
    out["missing_bin"] = PythonObject(info.missing_bin)
    out["missing_type"] = PythonObject(
        "NaN" if info.has_missing_bin() else "None"
    )
    out["monotone"] = PythonObject(info.monotone)
    out["num_bin"] = PythonObject(info.num_bin)
    if info.is_categorical:
        out["type"] = PythonObject("categorical")
        out["categories"] = _py_int_list(info.categories)
        out["bin_upper_bounds"] = PythonObject(None)
    else:
        out["type"] = PythonObject("numerical")
        out["categories"] = PythonObject(None)
        out["bin_upper_bounds"] = _py_f64_list(info.bin_upper_bounds)
    return out^


def _py_node(
    dump: ModelDump, node: DumpNode, has_gain: Bool
) raises -> PythonObject:
    """One node of the flat node table. `left` and `right` stay indices
    into that table; the Python layer nests them."""
    var out = _py_dict()
    out["node_index"] = PythonObject(node.node_index)
    out["depth"] = PythonObject(node.depth)
    out["parent"] = PythonObject(node.parent)
    out["left"] = PythonObject(node.left)
    out["right"] = PythonObject(node.right)
    out["is_leaf"] = PythonObject(node.is_leaf)
    out["leaf_index"] = PythonObject(node.leaf_index)
    out["split_index"] = PythonObject(node.split_index)
    out["value"] = PythonObject(node.value)
    out["count"] = PythonObject(node.count)
    if node.is_leaf:
        return out^
    out["split_feature"] = PythonObject(node.split_feature)
    out["split_feature_name"] = PythonObject(
        dump.features[node.split_feature].name
    )
    out["decision_type"] = PythonObject(
        "==" if node.is_categorical else "<="
    )
    out["threshold_bin"] = PythonObject(node.threshold_bin)
    if node.has_threshold:
        out["threshold"] = PythonObject(node.threshold)
    else:
        out["threshold"] = PythonObject(None)
    if node.is_categorical:
        out["categories"] = _py_int_list(node.categories)
        out["category_bins"] = _py_int_list(node.category_bins)
    else:
        out["categories"] = PythonObject(None)
        out["category_bins"] = PythonObject(None)
    out["default_left"] = PythonObject(node.default_left)
    out["missing_bin"] = PythonObject(node.missing_bin)
    out["missing_type"] = PythonObject(
        "NaN" if node.missing_bin >= 0 else "None"
    )
    # Null and not a zero a consumer could mistake for a measured gain: a
    # model read back from a file has no gains at all.
    if has_gain:
        out["split_gain"] = PythonObject(node.split_gain)
    else:
        out["split_gain"] = PythonObject(None)
    return out^


def _py_tree(dump: ModelDump, tree: DumpTree) raises -> PythonObject:
    var out = _py_dict()
    out["tree_index"] = PythonObject(tree.tree_index)
    out["iteration"] = PythonObject(tree.iteration)
    out["class_id"] = PythonObject(tree.class_id)
    out["num_leaves"] = PythonObject(tree.num_leaves)
    out["num_nodes"] = PythonObject(tree.num_nodes())
    out["num_cat"] = PythonObject(tree.num_cat)
    out["max_depth"] = PythonObject(tree.max_depth)
    out["shrinkage"] = PythonObject(tree.shrinkage)
    var nodes = Python.list()
    for i in range(len(tree.nodes)):
        nodes.append(_py_node(dump, tree.nodes[i], dump.has_split_gain))
    out["nodes"] = nodes^
    return out^


def _py_dump(dump: ModelDump) raises -> PythonObject:
    """A built dump as plain Python objects. Floats cross as C doubles, so
    the payload holds the model's own bits and not a decimal of them."""
    var out = _py_dict()
    out["dump_format_version"] = PythonObject(dump.dump_format_version)
    out["model_format_version"] = PythonObject(dump.model_format_version)
    if dump.has_objective_code:
        out["objective_code"] = PythonObject(dump.objective_code)
    else:
        out["objective_code"] = PythonObject(None)
    out["num_class"] = PythonObject(dump.num_class)
    out["num_tree_per_iteration"] = PythonObject(
        dump.num_tree_per_iteration
    )
    out["num_iteration"] = PythonObject(dump.num_iteration)
    out["learning_rate"] = PythonObject(dump.learning_rate)
    out["base_score"] = _py_f64_list(dump.base_score)
    out["num_feature"] = PythonObject(dump.num_feature)
    out["num_bin"] = PythonObject(dump.num_bin)
    if dump.is_constrained():
        out["monotone_constraints"] = _py_int_list(dump.monotone_constraints)
    else:
        out["monotone_constraints"] = PythonObject(None)
    out["has_split_gain"] = PythonObject(dump.has_split_gain)
    out["has_node_count"] = PythonObject(dump.has_node_count)
    var infos = Python.list()
    for f in range(len(dump.features)):
        infos.append(_py_feature(dump.features[f]))
    out["feature_infos"] = infos^
    var trees = Python.list()
    for t in range(len(dump.trees)):
        trees.append(_py_tree(dump, dump.trees[t]))
    out["trees"] = trees^
    return out^


def dump_model(
    model: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    """The fitted model as the inspection schema, as plain Python objects.

    See docs/MODEL_INSPECTION_SCHEMA.md. Trees arrive as flat node tables
    with child links; `python/mojoboost/inspection.py` nests them.
    """
    var m = model.downcast_value_ptr[Model]()
    return _py_dump(
        inspection_build_dump(m[], _dump_names(feature_names, n_names))
    )


def dump_model_multiclass(
    model: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return _py_dump(
        inspection_build_multiclass_dump(
            m[], _dump_names(feature_names, n_names)
        )
    )
```

### Wave 2: the derived queries

```mojo
def split_values(
    model: PythonObject, feature: PythonObject
) raises -> PythonObject:
    """Every threshold the ensemble splits `feature` at, depth first per
    tree. Raises for a categorical feature, whose splits are category sets
    and have no value to bin."""
    var m = model.downcast_value_ptr[Model]()
    return _py_f64_list(
        inspection_dump_split_values(
            inspection_build_dump(m[]), Int(py=feature)
        )
    )


def split_values_multiclass(
    model: PythonObject, feature: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return _py_f64_list(
        inspection_dump_split_values(
            inspection_build_multiclass_dump(m[]), Int(py=feature)
        )
    )


def dump_raw_scores(
    model: PythonObject, x_addr: PythonObject, n_features: PythonObject
) raises -> PythonObject:
    """Raw scores for one raw example, routed through the dump's own bin
    edges and node records rather than through the model's prediction path.
    One entry for a single-output model, one per class for a softmax one."""
    var m = model.downcast_value_ptr[Model]()
    var nf = Int(py=n_features)
    return _py_f64_list(
        inspection_dump_raw_scores(
            inspection_build_dump(m[]), _f64_list(Int(py=x_addr), nf)
        )
    )


def dump_raw_scores_multiclass(
    model: PythonObject, x_addr: PythonObject, n_features: PythonObject
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nf = Int(py=n_features)
    return _py_f64_list(
        inspection_dump_raw_scores(
            inspection_build_multiclass_dump(m[]),
            _f64_list(Int(py=x_addr), nf),
        )
    )


def dump_leaf_index(
    model: PythonObject,
    tree_index: PythonObject,
    x_addr: PythonObject,
    n_features: PythonObject,
) raises -> PythonObject:
    """The leaf ordinal one raw example reaches in one tree, routed through
    the dump. The same numbering `predict_leaf` reports."""
    var m = model.downcast_value_ptr[Model]()
    var nf = Int(py=n_features)
    return PythonObject(
        inspection_dump_leaf_index(
            inspection_build_dump(m[]),
            Int(py=tree_index),
            _f64_list(Int(py=x_addr), nf),
        )
    )


def dump_leaf_index_multiclass(
    model: PythonObject,
    tree_index: PythonObject,
    x_addr: PythonObject,
    n_features: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    var nf = Int(py=n_features)
    return PythonObject(
        inspection_dump_leaf_index(
            inspection_build_multiclass_dump(m[]),
            Int(py=tree_index),
            _f64_list(Int(py=x_addr), nf),
        )
    )


def objective_code(model: PythonObject) raises -> PythonObject:
    """The trainer's objective code the model carries. A softmax model has
    no single code, which is why there is no multiclass counterpart: the
    Python layer answers `"multiclass"` from the handle's kind."""
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(m[].booster.objective)
```

Wave 2's five dump-consuming functions each **rebuild the dump per call**.
That is fine for `objective_code` (which does not) and for a one-row check,
and wrong for a loop over rows. If a caller ever needs per-row routing at
scale, the fix is a handle that owns a built `ModelDump` (`m.add_type`
alongside `Model` and `Dataset`), not a faster rebuild. Nothing in the
Python layer loops today: `raw_scores` and `leaf_index_of` are one-row
conformance checks.

### Wave 3: JSON, for consumers outside Python

```mojo
def dump_model_json(
    model: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    """The whole schema as a JSON string, for the C ABI and the CLI. The
    Python layer does not use this: JSON floats are decimal, and the object
    payload above carries the model's exact bits."""
    var m = model.downcast_value_ptr[Model]()
    return PythonObject(
        inspection_dump_json(
            inspection_build_dump(m[], _dump_names(feature_names, n_names))
        )
    )


def dump_model_json_multiclass(
    model: PythonObject,
    feature_names: PythonObject,
    n_names: PythonObject,
) raises -> PythonObject:
    var m = model.downcast_value_ptr[MulticlassModel]()
    return PythonObject(
        inspection_dump_json(
            inspection_build_multiclass_dump(
                m[], _dump_names(feature_names, n_names)
            )
        )
    )
```

### Registration

In `PyInit__mojoboost`, next to the existing `feature_importance` pair:

```mojo
        m.def_function[dump_model]("dump_model")
        m.def_function[dump_model_multiclass]("dump_model_multiclass")
        m.def_function[split_values]("split_values")
        m.def_function[split_values_multiclass]("split_values_multiclass")
        m.def_function[dump_raw_scores]("dump_raw_scores")
        m.def_function[dump_raw_scores_multiclass](
            "dump_raw_scores_multiclass"
        )
        m.def_function[dump_leaf_index]("dump_leaf_index")
        m.def_function[dump_leaf_index_multiclass](
            "dump_leaf_index_multiclass"
        )
        m.def_function[objective_code]("objective_code")
        m.def_function[dump_model_json]("dump_model_json")
        m.def_function[dump_model_json_multiclass](
            "dump_model_json_multiclass"
        )
```

`python/mojoboost/inspection.py` picks each of these up by name the moment
it exists, through `_hook`, and needs no change on the Python side.

### The task 14 `split_gains` hook is superseded

`handoffs/task14_inspection.md` §2 specified `split_gains` /
`split_gains_multiclass` returning `list[list[float]]`, to give a
text-sourced dump its gains back. Wave 1 makes that unnecessary: the native
dump carries gains already. `inspection.py` still looks for the hook (in
`_native_split_gains`, inside the compatibility section) so that shipping it
alone remains a working half step, but do not build it *and* wave 1. If
wave 1 lands first, delete that hook from the plan.

## 5. Ownership and lifetime of the returned data

- **Everything the bindings return is a fresh Python object.** `_py_dump`
  copies every scalar and builds every list; the payload shares no storage
  with the model handle. A caller may keep it after the handle is
  collected, and mutating it cannot corrupt a model.
- **Nothing retains a Python buffer.** `dump_raw_scores` and
  `dump_leaf_index` copy the row out of the caller's buffer with
  `_f64_list` before doing anything, exactly as `predict` does. The Python
  side keeps the buffer referenced across the call anyway (`_row_buffer`
  returns it for that reason), so neither side depends on the other's
  lifetime.
- **`ModelDump` owns its own storage.** It holds `List`s of Mojo values, not
  pointers into `Model` or `BinMapper`. A dump built from a model outlives
  that model, and `build_dump` is the only thing that reads the model at
  all.
- **Cost.** Building a dump copies every node and every bin edge, so it is
  O(nodes + edges) in time and memory. For the models the Python layer
  dumps that is nothing; for a 10k-tree ensemble it is a real allocation,
  which is the other reason a per-row loop wants a dump handle rather than
  a rebuild.

## 6. Schema versioning

Two numbers, and they are not the same number:

- `dump_format_version` is the schema's, and it is **1**. It is declared in
  `src/mojoboost/model_dump.mojo` (`DUMP_FORMAT_VERSION`) and mirrored by
  `python/mojoboost/inspection.py` (`DUMP_FORMAT_VERSION`). Bump it only
  for a change a consumer written against the previous version could not
  survive: a key removed, a key's type changed, or a key's meaning changed.
  Adding an optional key does not bump it, and a consumer must ignore keys
  it does not know. **Both constants move together**; the Python one exists
  so a caller can branch without loading the extension.
- `model_format_version` is the save format's, and it is **3**. The native
  dump reports the version a model written today serializes to
  (`MODEL_FORMAT_VERSION` in `model_dump.mojo`, which must track `_VERSION`
  in `serialize.mojo`); the compatibility parser reports the version of the
  text it actually read. This is the number that tells a consumer which
  optional facts a model of that vintage can carry.

If `serialize.mojo` gains a v4, `MODEL_FORMAT_VERSION` here changes with
it, and `SUPPORTED_MODEL_FORMAT_VERSIONS` in `inspection.py` gains a `4`
for as long as the parser survives. Neither is a schema bump.

## 7. Compatibility with old model formats

The dump is built from an in-memory model, so "old format" means "a model
loaded from an old file", and it shows up as capability flags rather than
as a parse failure:

| Model came from | `has_node_count` | `has_split_gain` | Notes |
| --- | --- | --- | --- |
| training, this build | true | true | the full dump |
| a v3 file | true | **false** | gains are not serialized; every `split_gain` is null |
| a v2 file | false | false | v2 predates node covers; every count is 0.0 |
| a v1 file | false | false | v1 also predates missing-value routing, so every node's `missing_bin` is -1 and `missing_type` is `"None"` |

Two behavior changes from what `src/mojoboost/inspection.mojo` did before
this task, both deliberate:

1. It hardcoded `has_split_gain: true` for any native dump. That was wrong
   for a loaded model, whose gains are all zero. `has_split_gains(trees)`
   now decides, by looking for one positive gain: a split is only ever taken
   for a positive gain, so one settles it, and an ensemble that never split
   has no gain to report either way.
2. With `has_split_gain` false, every node's `split_gain` is rendered
   `null` rather than `0.0`, so a consumer cannot mistake an absent gain for
   a measured one. This matches what the schema doc already promises for a
   text-sourced dump.

`docs/MODEL_INSPECTION_SCHEMA.md` says `has_split_gain` is "False when the
dump was built by parsing a model file". That wording needs one clause
added: it is false whenever the model carries no gains, which includes a
natively dumped model that was loaded from a file. The rule the doc states
(branch on the flag, never on the source) is unchanged, and no consumer
written against the doc breaks.

## 8. Python code that can be deleted after wiring

Everything below the banner in `python/mojoboost/inspection.py`:

```
# ========================================================================
# COMPATIBILITY: the schema, rebuilt in Python from the model text
# ========================================================================
```

That is one contiguous block, from `_UNKNOWN_BIN` to the end of the file:
`_f64_from_bits`, `_Reader`, `parse_model_string`, `_parse_mapper`,
`_parse_categorical`, `_parse_monotone`, `_parse_trees`, `_model_text`,
`_native_split_gains`, `_dump_from_text`, `_resolve_names`,
`_feature_infos`, `_node_depths`, `_split_ordinals`, `_category_bins`,
`_threshold_value`, `_tree_info`, `_bin_value`, `_goes_left`, `_walk`,
`_leaf_index_from_dump`, `_raw_scores_from_dump`, `_split_values_from_dump`
— lines 700 to 1301 as the file stands, a little under 600 lines, or 46% of
it.

Delete with it, above the banner:

| Where | What |
| --- | --- |
| line 1 area | `import struct as _struct` |
| `__all__` | `"parse_model_string"` |
| `dump_model` | the final `return _dump_from_text(model, feature_names)`, and with it the `isinstance(model, str)` branch: `dump_model` then takes a fitted model or a `Booster` and nothing else |
| `split_values` | the `if not isinstance(model, str)` guard around the hook, and the trailing `return _split_values_from_dump(dump, index)` |
| `leaf_index_of` | the `source = dump_model(source)` fallback and the trailing `_leaf_index_from_dump` call |
| `raw_scores` | the same two lines |
| `objective_of` | the two-line `parse_model_string` fallback |
| module docstring | the "Until the native binding lands" section |
| `SUPPORTED_MODEL_FORMAT_VERSIONS` | the constant, unless a caller is found to depend on it (it is public today) |

After that deletion, `leaf_index_of` and `raw_scores` take a model and not
a dump, which is the signature change §9 covers.

**Do not delete any of it before wave 1 ships.** Every entry point above the
banner falls back to this block today, and `dump_model(text)` is the only
way to inspect a model file without loading it.

## 9. Tests that change in the wiring commit

`python/tests/parallel/test_inspection.py` was not editable by this task and
is correct as it stands today: it describes a build with no native hooks.
Wave 1 changes what the module reports, so these cases move with it. Each is
a change in what is true, not a regression.

| Test | Why it changes | What it should assert |
| --- | --- | --- |
| `test_dump_reports_its_own_version_and_source` | `source` becomes `"native"` and `has_split_gain` becomes True | `dump["source"] == "native"`, `has_split_gain is True` |
| `test_nodes_carry_the_documented_keys` | internal nodes now carry a real gain | `node["split_gain"] >= 0.0` and finite, instead of `is None` |
| `test_dump_from_the_model_text_alone` | the model dumps natively and the text dumps through the parser, so `source`, `has_split_gain`, and every `split_gain` differ | compare the two with `source`, `has_split_gain`, and the per node `split_gain` stripped; that is the real claim, that the two sources describe the same trees |
| `test_a_booster_read_back_dumps_the_same_trees` | a reloaded model has no gains, because gains are not serialized | the same stripped comparison |
| `test_trees_to_records_has_lightgbm_columns` | `assert all(row["split_gain"] is None ...)` | gains are present now; assert leaves have no gain row value and internal nodes do |

Worth adding in the same commit, since they are what wave 1 buys:

- the per-feature sums of `split_gain` over the dump equal
  `booster_.feature_importance("gain")`. That ties the dump to an
  independently computed number and is the check worth having.
- `raw_scores(model, row)` (native) equals `raw_scores(dump, row)`
  (interpreted) equals `booster_.predict([row], raw_score=True)`, for a
  model with NaN and with a categorical feature. That is the three-way
  agreement the whole design rests on, and it is checkable while both paths
  exist.
- a model saved and reloaded reports `has_split_gain False` and
  `has_node_count True`.

A focused Mojo test is also missing and should land with wave 1, as
`tests/parallel/test_model_dump.mojo`: build a dump from a small fitted
`Model`, then assert leaf ordinals are `0..n_leaves-1`, an internal node's
`count` equals its children's, `dump_raw_scores` matches
`Model.predict_raw` to the last bit on a handful of rows (including a NaN
row), `dump_leaf_index` matches `Booster.leaf_indices`, and a categorical
model's `categories` are a subset of the feature's. One file, one `mojo
run`; do not add it to the `test` task's chain without checking the CI
budget.

## 10. Documentation

| File | Change |
| --- | --- |
| `docs/MODEL_INSPECTION_SCHEMA.md` | the `has_split_gain` clause in §7 above; the implementations table (`inspection.py` is a facade over the native dump, with the parser as fallback); the "Native hooks" section, which names task 14's two hooks and should name wave 1's instead |
| `docs/LIGHTGBM_PARITY.md` | unchanged by this task, but still carries the five rows task 14 §5 listed. `pixi run check-parity` is its own CI job, so edit it in the same commit as the Python surface, not before |
| `python/mojoboost/basic.py` | the "No `dump_model` / `trees_to_dataframe`" bullet in the module docstring is still there and still wrong once §11 lands |

## 11. Python package surface

Unchanged from `handoffs/task14_inspection.md` §3 and §4, and still not
applied. In short:

- `Booster.dump_model`, `Booster.trees_to_dataframe`, and
  `Booster.get_split_value_histogram` delegate to this module, with the
  import inside the method so `basic` and `inspection` do not form a cycle.
- `python/mojoboost/__init__.py` re-exports the submodule and adds
  `"inspection"` to `__all__`.
- `_Base` gains `feature_name_`, `n_features_`, and `objective_` as
  delegating properties, and none of them goes in `_FITTED_ATTRS`.

One addition from this task: `objective_of` now prefers the
`objective_code` hook (wave 2) and only falls back to a `model_to_string()`
round trip. Once the hook exists the attribute is a single call and there is
no reason to cache it.

## 12. Intentional LightGBM differences

Carried forward, all still true, plus one new:

1. **No `weight` column** in `trees_to_dataframe`. LightGBM's node weight is
   a sum of hessians; mojoboost records the training row cover, in `count`.
   A weight is not something this schema can report, rather than something
   it declines to.
2. **`split_gain` is null on a model read from a file.** Gains are recorded
   during growth and are deliberately not serialized. LightGBM writes them
   into its model file and so always has them.
3. **`get_split_value_histogram` returns what you asked for.** LightGBM
   switches its return type on whether pandas can be imported and takes
   `xlabel` / `ylabel` arguments only a plot uses; here `as_frame` decides
   and the plotting arguments do not exist.
4. **Thresholds are exact.** A mojoboost threshold is a bin's upper edge,
   so `bin <= threshold_bin` holds exactly when `value <= threshold`.
   LightGBM's thresholds are midpoints between observed values and do not
   have that property. This is why a consumer can route raw values through
   the dump at all.
5. **`leaf_index` is mojoboost's leaf ordinal**, leaves ranked in node-array
   order. It is what `predict(pred_leaf=True)` reports. It is not
   LightGBM's leaf id, and the two agree only by coincidence.
6. **Leaf values are unshrunk.** `leaf_value_is_shrunk: false`, and the
   arithmetic is `base_score[k] + sum(shrinkage * leaf_value)`. LightGBM
   folds the shrinkage into the leaf.
7. **No leaf editing.** `set_leaf_output` has no counterpart, for the three
   invariants stated in the schema doc and in both modules' headers.
8. **The objective name is resolved in Python, not in Mojo** (new). The
   model holds a trainer code; `OBJECTIVE_NAMES` spells it the way
   LightGBM's `objective_` attribute does. `src/mojoboost/lgbm_model_io.mojo`
   spells the same codes for a LightGBM *file*, where code 1 is
   `"binary sigmoid:1"` and code 6 (custom) raises. Task 14 §8 has the
   table; the two are deliberately not one function, and
   `test_objective_names_track_the_estimator_table` pins the Python side
   against `MojoBoostRegressor._OBJECTIVES` so it cannot drift.

## 13. Focused validation, in order

Nothing below was run. Run them one at a time, and stop at the first
failure rather than batching.

1. **Type-check the two Mojo modules.** They are uncompiled:
   `pixi run mojo package src/mojoboost -o /tmp/mojoboost.mojopkg`
   This compiles every module in the package, which is the cheapest way to
   put `model_dump.mojo` and `inspection.mojo` in front of the compiler
   without adding a test file. Expect to fix syntax before anything else in
   this handoff is worth attempting.
2. **The focused Mojo test**, once written:
   `pixi run mojo run -I src tests/parallel/test_model_dump.mojo`
3. **Build the extension** with the binding functions added:
   `pixi run build-python`
4. **The one Python test file this touches:**
   `pixi run -e pytest pytest -q python/tests/parallel/test_inspection.py`
   (with §9's edits applied; without them the five cases named there fail
   for the documented reason, which is not a signal to revert the wiring)
5. **The parity check**, if `docs/LIGHTGBM_PARITY.md` changed:
   `pixi run check-parity`

Do not run the full `pixi run test` chain for this change: it builds and
runs 47 Mojo test files and none of them touches inspection.

## 14. Risk list: what is unverified

`src/mojoboost/model_dump.mojo` and the rewritten
`src/mojoboost/inspection.mojo` **have not been compiled**. Their APIs were
audited against confirmed usage elsewhere in the tree; everything has
in-repo precedent except the items below, which are where to look first if
the package does not build.

1. `_json_string` iterates with `for cp in s.codepoint_slices()` and
   materializes each with `String(cp)`. Inherited from the previous
   `inspection.mojo`, still the only use of that API in this repository.
   Fallback: escape by byte, the way `serialize.mojo::_parse_u64` walks
   `token.as_bytes()`.
2. Structs with an explicit `__init__` that leaves most fields at a default
   (`DumpNode`, `DumpFeature`, `ModelDump`) and are then filled by
   assignment. `Tree.__init__` sets the same precedent, but `Tree` is
   filled by its constructor rather than by its caller.
3. `ref tree = dump.trees[t]` held across a `stack.append(...)` in
   `dump_split_values`, and `ref rec = tree.nodes[node]` held across the
   recursive `_write_node` call. Both are immutable borrows of one object
   while a different object is mutated, which the borrow checker should
   allow; if it does not, index instead of binding a `ref`.
4. `Error("...", value, "...")` with a `String` among the arguments, in
   `ModelDump.feature_index` and `dump_split_values`. Existing raises in
   `inspection.mojo` interleave `Int`s only.
5. `var` declared inside both arms of an `if` / `else` in `_build_features`
   (`begin`, `end`). The previous `_write_feature_infos` did exactly this,
   so this one is precedent-backed and listed only because it looks like a
   redefinition.
6. In the bindings, `_py_dict()` and `PythonObject.__setitem__`. If
   assigning into a `PythonObject` dict is not available, build each record
   as a `Python.list()` of `(key, value)` pairs and call `builtins.dict` on
   it, or hand the pairs to Python and let `_schema_from_native` build the
   dict; the Python side changes in one function either way.

`python/mojoboost/inspection.py` was not executed either. Its native path
cannot run until wave 1 exists, and its compatibility path is the code that
passed 30 tests under task 14, moved below a banner with three call sites
renamed (`_dump_from_text`, `_leaf_index_from_dump`, `_raw_scores_from_dump`,
`_split_values_from_dump`) and `_resolve_names` delegating to
`_resolve_override` so the length check has one implementation.

## 15. What this task deliberately did not do

- **Did not delete the text parser.** The bindings expose no dump today, so
  deleting it would break `dump_model` for every caller, including the
  tests. §8 is the deletion, and it is one commit after wave 1.
- **Did not change the JSON shape.** `docs/MODEL_INSPECTION_SCHEMA.md` is
  normative and was not editable here, so `dump_json` still nests
  `tree_structure` exactly as documented, even though `ModelDump` holds the
  nodes as a flat table with child links. The flat table is what the
  bindings hand to Python; the nesting is a rendering of it.
- **Did not add a `ModelDump` handle type.** Wave 2's per-call rebuild is
  the honest simple thing for one-row checks. §5 says what to do instead if
  a per-row loop ever appears.
- **Did not invent metadata.** No hessian sum, no LightGBM leaf id, no
  `lower_bound` / `upper_bound`, and no gain for a model that does not carry
  one. Where a fact is absent, the schema says so with a flag and a null.
- **Did not touch `serialize.mojo`.** The recommendation from task 14 §6
  stands: do not serialize gains. They are a property of how a tree was
  grown rather than of what it computes, `has_split_gain` already lets a
  consumer branch, and a v4 would grow every saved model by one float per
  node for a fact only inspection reads.
