# Model inspection schema

This is the normative statement of what a mojotrees model dump contains.
Two implementations produce it, and this document is what holds them to
each other:

| Implementation | Source | Carries gains |
| --- | --- | --- |
| `src/mojotrees/model_dump.mojo` | the in-memory `Model` / `MulticlassModel` | yes |
| `src/mojotrees/inspection.mojo` | the same dump, rendered as JSON | yes |
| `python/mojotrees/inspection.py` | the native dump, or `Booster.model_to_string()` parsed | yes from a v4 model either way; a pre-v4 model needs the `split_gains` hook |

A consumer reads the schema, not either implementation. Everything below
is part of the contract; anything not below is not.

## Versioning

`dump_format_version` is this schema's version. It is bumped only for a
change that a consumer written against the previous version could not
survive: a key removed, a key's type changed, or a key's meaning changed.
Adding an optional key does not bump it, so a consumer must ignore keys it
does not know.

`model_format_version` is a different number: the version of mojotrees's
save format (`src/mojotrees/serialize.mojo`) that the model was read from
or would serialize to. It is what tells a consumer which optional facts a
model of that vintage can carry at all. v1 and v2 models predate node
covers, so a dump built from one reports `has_node_count: false`; v1
through v3 predate split gains, so a dump built from one reports
`has_split_gain: false` unless a hook supplies them.

The current values are `dump_format_version: 1` and
`model_format_version: 4`.

## Capability flags

Two keys say what this particular dump can answer, and a consumer that
uses gains or covers must branch on them rather than assume:

- `has_split_gain` (bool). False for a model read from a v1, v2, or v3
  file, whose nodes predate serialized gains: gains are recorded during
  growth and cannot be recovered from a fitted tree, so a dump built by
  parsing such a file can only report them if the `split_gains` hook
  supplies them. A v4 file carries them, and a dump built from one
  reports true without any hook. Every node's `split_gain` is `null` when
  this is false.
- `has_node_count` (bool). False for a model read from a v1 or v2 file,
  whose nodes predate covers, and false for a v4 model whose trees were
  grown without them or were loaded from such a file. Every node's
  `internal_count` / `leaf_count` is `0.0` when this is false.

Both flags are properties of the model in front of you and not of the
format version alone: v4 records for each tree whether it has covers and
whether it has gains, so a model that never had them re-saves as a v4
file that says so, rather than as one that claims zeros are real.

## Top level

| Key | Type | Meaning |
| --- | --- | --- |
| `dump_format_version` | int | this schema's version |
| `producer` | str | always `"mojotrees"` |
| `model_format_version` | int | mojotrees save format version, 1 through 4 |
| `source` | str | where the dump was built from: `"model_to_string"`, `"model_to_string+split_gains"`, or `"native"` |
| `objective` | str or null | resolved objective name, LightGBM's spelling; `"multiclass"` for a softmax model. The native dump leaves this null and reports the code alone |
| `objective_code` | int or null | the trainer's objective code; null for a multiclass model, which has no single-output code |
| `num_class` | int | 1 for a single-output model, the class count for a softmax one |
| `num_tree_per_iteration` | int | trees one boosting iteration grows: 1, or `num_class` |
| `num_iteration` | int | boosting iterations the ensemble holds |
| `learning_rate` | float | the shrinkage applied at prediction time |
| `base_score` | list[float] | the ensemble's starting raw score, one entry per class |
| `leaf_value_is_shrunk` | bool | always false; see "Leaf values" below |
| `num_feature` | int | features the model was fitted on |
| `max_feature_idx` | int | `num_feature - 1`, as LightGBM reports it |
| `num_bin` | int | the binning's bin budget (LightGBM's `max_bin`) |
| `feature_names` | list[str] | one per feature; `Column_0`, `Column_1`, ... when the model carries none. See "Where the names come from" below |
| `feature_infos` | list[object] | one record per feature, below |
| `monotone_constraints` | list[int] or null | one sign per feature, or null when the model was grown unconstrained |
| `has_split_gain` | bool | above |
| `has_node_count` | bool | above |
| `tree_info` | list[object] | one record per tree, below |

### Leaf values

mojotrees stores unshrunk leaf values and multiplies by the shrinkage when
it predicts. LightGBM folds the shrinkage into the leaf value instead, so
a reader porting code from LightGBM has to know which convention is in
front of it. That is what `leaf_value_is_shrunk: false` says, and the
arithmetic it implies is

```
raw_score[k] = base_score[k] + sum over trees of (shrinkage * leaf_value)
```

with each tree's `shrinkage` on its own record, equal to the top level
`learning_rate`. `mojotrees.inspection.raw_scores(dump, row)` is this sum,
and `python/tests/parallel/test_inspection.py` checks it against the
model's own `predict(raw_score=True)` to the last bit.

### Where the names come from

A model file carries feature names from v4 on, as an optional section, so
a model read back from disk reports the names it was fitted with rather
than inventing them. Four sources are consulted, most authoritative
first, and the first one that supplies exactly `num_feature` names wins:

1. the caller's `feature_names=` override, which is the only way to
   rename features for a dump;
2. the names the live `Booster` carries, which are the training frame's;
3. the names the file carries, which is all a model read back from a v4
   file has;
4. `Column_0`, `Column_1`, ..., which is a placeholder and not a name the
   model claims.

A pre-v4 file carries no names, so a model loaded from one falls to 4
unless the caller supplies them. Names are escaped in the file, so a name
holding a space, a tab, or a backslash survives the round trip exactly;
a name holding any other control byte is refused at save time rather
than written as something that will not read back.

## `feature_infos[f]`

| Key | Type | Meaning |
| --- | --- | --- |
| `index` | int | the feature's column index |
| `name` | str | its name, the same string `feature_names[index]` holds |
| `type` | str | `"numerical"` or `"categorical"` |
| `bin_upper_bounds` | list[float] or null | ascending bin edges; null for a categorical feature |
| `categories` | list[int] or null | ascending kept category codes; null for a numerical feature. Code `categories[i]` is bin `i + 1` |
| `num_bin` | int | bins this feature uses, the missing bin included |
| `missing_bin` | int | the bin reserved for missing values, or -1 |
| `missing_type` | str | `"NaN"` when a bin is reserved, `"None"` when not |
| `monotone` | int | the constraint sign the feature was grown under: -1, 0, or 1 |

A numerical feature with `k` edges uses bins `0..k`, plus a missing bin at
`k + 1` when one is reserved. A value maps to the first bin whose upper
edge it does not exceed. A categorical feature uses bin 0 for missing,
unseen, and dropped codes, and bin `i + 1` for `categories[i]`.

## `tree_info[t]`

| Key | Type | Meaning |
| --- | --- | --- |
| `tree_index` | int | position in the ensemble, 0-based |
| `iteration` | int | `tree_index // num_tree_per_iteration` |
| `class_id` | int | `tree_index % num_tree_per_iteration`; 0 for a single-output model |
| `num_leaves` | int | leaves in this tree |
| `num_nodes` | int | nodes in this tree, leaves included |
| `num_cat` | int | nodes that split by category set |
| `max_depth` | int | the deepest leaf's depth, in edges from the root |
| `shrinkage` | float | the multiplier applied to this tree's leaf values |
| `tree_structure` | object | the root node |

Trees are round-major for a multiclass model: the tree for round `i` and
class `k` is at `i * num_class + k`, which is what `iteration` and
`class_id` restate rather than making a reader recompute.

## Nodes

A node is a leaf or an internal node, and the two are told apart by which
keys they carry. `leaf_index` is present exactly on leaves and
`split_index` exactly on internal nodes. Branch on a key's presence; there
is no type tag.

### Leaf

| Key | Type | Meaning |
| --- | --- | --- |
| `node_index` | int | the node's index in the tree's flat arrays |
| `leaf_index` | int | the leaf's ordinal, in `[0, num_leaves)` |
| `leaf_value` | float | the unshrunk output |
| `leaf_count` | float | training rows that reached this leaf; 0.0 when `has_node_count` is false |
| `depth` | int | edges from the root; the root is 0 |

`leaf_index` is mojotrees's own leaf ordinal: leaves ranked in node-array
order. It is exactly what `predict(pred_leaf=True)` reports, it is fixed
once a tree is grown, and a saved and reloaded model assigns the same
ordinals. It is not LightGBM's leaf id, and the two agree only by
coincidence.

`node_index` is the flat-array index. It is stable for a given model, and
it is what makes a node addressable, but it is an implementation detail of
how the tree is laid out: node ids number internal nodes and leaves
together. Prefer `leaf_index` when you mean a leaf.

### Internal node

| Key | Type | Meaning |
| --- | --- | --- |
| `node_index` | int | index in the tree's flat arrays |
| `split_index` | int | the node's ordinal among this tree's internal nodes, in node order |
| `split_feature` | int | the feature this node tests |
| `split_feature_name` | str | that feature's name |
| `decision_type` | str | `"<="` for a numerical split, `"=="` for a categorical one |
| `threshold` | float or null | the largest value routed left; null on a categorical node, and on a numerical node whose split bin has no upper edge |
| `threshold_bin` | int | the bin id the split compares against |
| `categories` | list[int] or null | the raw category codes routed left; null on a numerical node |
| `category_bins` | list[int] or null | the same set as bin ids; null on a numerical node |
| `default_left` | bool | where a missing value goes |
| `missing_bin` | int | the split feature's missing bin, or -1 |
| `missing_type` | str | `"NaN"` or `"None"`, as on the feature |
| `split_gain` | float or null | the gain this split earned; null when `has_split_gain` is false |
| `internal_value` | float | the value this node held when it was created |
| `internal_count` | float | training rows that reached this node |
| `depth` | int | edges from the root |
| `left_child` | object | the left subtree |
| `right_child` | object | the right subtree |

### Routing

A row goes left exactly when, in this order:

1. the node is categorical (`category_bins` is not null) and the row's bin
   is in `category_bins`; bin 0 is never a member, so missing, unseen, and
   dropped categories always go right; otherwise
2. the row's bin equals `missing_bin`, in which case `default_left`
   decides; otherwise
3. the row's bin is `<= threshold_bin`.

`threshold` is the bin's upper edge, which makes it the exact real-valued
boundary and not an approximation of one: `bin <= threshold_bin` holds
exactly when `value <= threshold`. A consumer can therefore route raw
values through the dump, and land where the model lands, without binning
anything. LightGBM's thresholds are midpoints between observed values and
do not have this property.

`internal_count` is a node's cover, the training rows that reached it. It
is exactly the sum of its two children's counts. Under bagging or GOSS it
counts the sampled rows, which are the rows the node's value was fitted
from.

There is no `weight` key. LightGBM's node weight is a sum of hessians;
mojotrees records the row cover and not the hessian sum, so a weight is
not something this schema can report rather than something it declines to.

## Derived shapes

`python/mojotrees/inspection.py` builds five things from the dump, and
none of them adds a dependency:

- `trees_to_records(model)` returns one dict per node, in
  `TREE_FRAME_COLUMNS` order, depth first per tree. Needs nothing
  installed.
- `trees_to_dataframe(model)` is the same rows as a pandas DataFrame, with
  LightGBM's column names. Raises `ImportError` naming
  `trees_to_records` when pandas is absent, since pandas is not a
  mojotrees dependency.
- `get_split_value_histogram(model, feature, bins=None, as_frame=False)`
  returns `(counts, bin_edges)`, the data behind LightGBM's
  `plot_split_value_histogram`. No plotting dependency is introduced, and
  none is needed.
- `leaf_outputs(model, tree_index=None)` returns the leaf values by the
  ordinal `predict(pred_leaf=True)` reports, one list per tree, or one
  tree's list. Unshrunk, as they are stored.
- `feature_importance(model, importance_type="split")` delegates to
  `Booster.feature_importance`, which is
  `src/mojotrees/importance.mojo`. It is here so that the inspection
  surface answers the question without becoming a second answer to it:
  the dump's per-feature `split_gain` sums equal the `"gain"` importance
  by construction, and that identity is the check worth writing.

Two deliberate deviations from LightGBM in the frame: `split_gain` is
`None` unless the dump carries gains, and `weight` is always `None` for
the reason above.

One deliberate deviation in the histogram: LightGBM switches its return
type on whether pandas can be imported, and takes `xlabel` and `ylabel`
arguments that only a plot uses. Here the return type is the one you
asked for, through `as_frame`, and does not depend on what happens to be
installed. `bins` keeps LightGBM's meaning: `None`, or a number above the
count of distinct split values, gives one bin per distinct value.

## Leaf editing is not offered

LightGBM's `Booster.set_leaf_output` has no counterpart here, and this is
a decision rather than a gap. It is also a reported decision rather than
a discovered one: `mojotrees.inspection.model_editing_support()` returns
the status, and `src/mojotrees/inspection.mojo` states the same one
natively, in `model_editing_status_json`. A consumer branches on it
instead of catching an `AttributeError`.

| Key | Type | Meaning |
| --- | --- | --- |
| `supported` | bool | false in this build |
| `operation` | str | `"set_leaf_output"`, the operation being refused |
| `reason` | str | one sentence: a fitted tree records facts about the fit that produced it |
| `invariants` | list[str] | the three below, one string each |
| `serialized_state` | list[str] | `["count", "split_gain"]`, the invariants a file carries |
| `model_format_version` | int | the format that carries them, 4 |
| `read_only_alternative` | str | `"leaf_outputs"` |

A grown mojotrees tree satisfies invariants that an arbitrary leaf edit
falsifies with nothing left to detect it:

- `internal_count` is the training rows that reached a node, and exact
  feature contributions (`src/mojotrees/contrib.mojo`) condition on it as
  the background weighting.
- `internal_value` is the value a node held when it was created, which is
  what recovers the monotonic output intervals of a constrained model
  (`src/mojotrees/monotone.mojo`).
- `split_gain` was computed from the gradient and hessian sums the leaf
  held at growth time.

Editing a leaf value leaves all three describing a model that no longer
exists, and no check could tell an intentional edit from a corrupt one.
Two of the three are now serialized as well, model format v4 carrying
node covers and split gains, so the contradiction would outlive the
session that made it: the file would hold an edited leaf next to the
counts and gain sums that deny it. Until each invariant can be restated
after an edit and tested, inspection stays read only, and
`leaf_outputs` is the half of LightGBM's leaf-output pair that is
offered.

## Native hooks

The Python implementation is a facade over the extension module. It asks
for a native dump first and falls back to parsing
`Booster.model_to_string()`, and `source` in the dump says which of the
two answered. The fallback is not a lesser answer for a current model:
from format v4 on, the text carries the gains, the covers, and the
feature names, so a text-built dump reports the same capability flags a
native one does.

- `_mojotrees.dump_model(handle, names, n_names)` and
  `_mojotrees.dump_model_multiclass(...)` build the schema natively, from
  `src/mojotrees/model_dump.mojo`, and are what `source: "native"` means.
- `_mojotrees.dump_model_json(...)` and
  `_mojotrees.dump_model_json_multiclass(...)` return the same schema as a
  JSON string from `src/mojotrees/inspection.mojo`, for a consumer outside
  Python that would rather not parse a token stream.
- `_mojotrees.split_gains(handle)` and
  `_mojotrees.split_gains_multiclass(handle)`, each returning
  `list[list[float]]`, are optional and are asked only when the text is
  older than v4: a live handle still holds the gains that an older file
  dropped, which is the one case that flips `has_split_gain` to true
  without the text. `source` is then
  `"model_to_string+split_gains"`.
- `_mojotrees.model_feature_names(path)` reads a file's names without
  loading it, and returns an empty sequence for a pre-v4 file.
