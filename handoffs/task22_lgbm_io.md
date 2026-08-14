# Task 22 handoff: LightGBM model file interop

Lane files (the only ones this lane touched):

- `src/mojoboost/lgbm_model_io.mojo` (new)
- `tests/parallel/test_lgbm_model_io.mojo` (new)
- `handoffs/task22_lgbm_io.md` (this file)

`model.mojo`, `serialize.mojo`, and `tree.mojo` were read and not edited.
Nothing was committed or staged by this lane.

> **Read this before integrating.** A concurrent session committed a snapshot
> of this lane's two files mid-task, in `b5a9afc "Add parallel accelerator and
> compatibility foundations"`, together with many other lanes' work. That
> snapshot predates the fixes below and **does not compile**: it transfers
> `_NodeArrays` fields out one at a time, which the ownership checker rejects
> with "field 'arrays.feature' destroyed out of the middle of a value". The
> working tree holds the corrected, tested version as an uncommitted delta
> over `b5a9afc` — three changes: the `_NodeArrays.into_tree` swap-based
> handoff (the compile fix), the corrected round-trip exactness claim in the
> module docstring, and the matching test. Commit the working tree, not
> `b5a9afc`'s version of these two files. Nothing from any other lane was
> overwritten; the only files this lane wrote are the three listed above.

## What landed

An isolated reader and writer for LightGBM's text model format, expressed
entirely in terms of the public `BinMapper`, `Tree`, `Booster`, and
`MulticlassBooster` structures.

Public API in `mojoboost.lgbm_model_io`:

| Symbol | Signature |
| --- | --- |
| `parse_lgbm_model` | `(text: String) raises -> Model` |
| `parse_lgbm_multiclass_model` | `(text: String) raises -> MulticlassModel` |
| `load_lgbm_model` | `(path: String) raises -> Model` |
| `load_lgbm_multiclass_model` | `(path: String) raises -> MulticlassModel` |
| `dump_lgbm_model` | `(model: Model) raises -> String` |
| `dump_lgbm_multiclass_model` | `(model: MulticlassModel) raises -> String` |
| `save_lgbm_model` | `(model: Model, path: String) raises` |
| `save_lgbm_multiclass_model` | `(model: MulticlassModel, path: String) raises` |
| `lgbm_text_kind` | `(text: String) raises -> String` (`"objective"` / `"multiclass"`) |
| `lgbm_model_file_kind` | `(path: String) raises -> String` |
| `lgbm_objective_code` | `(objective_line: String) raises -> Int` |
| `lgbm_objective_name` | `(objective: Int) raises -> String` |

`lgbm_text_kind` / `lgbm_model_file_kind` are deliberately shaped like
`serialize.model_file_kind`, so a caller holding a path can dispatch between
the single-output and multiclass loaders the same way for both formats.

### Header and per-tree fields covered

Header: `version`, `num_class`, `num_tree_per_iteration`, `max_feature_idx`,
`objective` (name plus `key:value` settings), `label_index`, `feature_names`,
`feature_infos`, `tree_sizes`, and the bare `average_output` flag. Trees:
`num_leaves`, `num_cat`, `split_feature`, `split_gain`, `threshold`,
`decision_type` (categorical bit, `default_left` bit, `missing_type` field),
`left_child`, `right_child`, `leaf_value`, `leaf_count`, `internal_value`,
`internal_count`, `is_linear`, `shrinkage`. Everything from `end of trees`
onward (feature importances, the parameter dump, `pandas_categorical`)
describes how the model was trained rather than what it computes and is read
past.

### Thresholds and bins

mojoboost routes on `Tree.threshold_bin`, LightGBM on a real threshold. The
two are reconciled exactly, both directions, by

```
bin(v) <= t   iff   v <= edges[edge_offsets[f] + t]
```

which is `BinMapper.bin_value`'s own definition. Reading collects the
distinct thresholds LightGBM used on each feature, sorts them, and makes
exactly those the feature's edges. Writing emits
`edges[edge_offsets[f] + threshold_bin]`. Nothing is approximated.

**Consequence worth flagging to the integrator:** a model read from a
LightGBM file carries only the edges its trees split on, not the binning
LightGBM fit. It predicts identically and it is all the file holds, but it is
not the training-time binning, so such a mapper must not feed
`boosting.train_more` or anything else that assumes the mapper describes the
training data. `BinMapper.matches` will (correctly) refuse to pair it with a
freshly fit mapper.

### Base score, learning rate, and how exact the round trip is

LightGBM folds both the shrinkage and the boost-from-average init score into
its stored leaf values. So reading gives `learning_rate = 1.0`,
`base_score = 0.0`, leaf values verbatim; writing multiplies every leaf value
by `learning_rate` and adds `base_score` into the iteration-0 trees the way
`Tree::AddBias` does (recording `shrinkage=1` on those, as LightGBM does).

Measured, not assumed:

- A model whose `learning_rate` is already 1.0 and `base_score` 0.0 — which
  is exactly what reading a LightGBM file produces — round-trips
  **bit-exactly** in predictions, and `dump(parse(dump(parse(t))))` equals
  `dump(parse(t))` byte for byte.
- A model with `learning_rate = 0.1` round-trips to within a few ULP, not
  exactly. The test caught this at 1 ULP and the module docstring now states
  it. Cause: prediction accumulates `score += learning_rate * leaf_value`,
  which fuses into a single rounded multiply-add, whereas the file forces
  `learning_rate * leaf_value` to be rounded on its own before it can be
  stored. No text format can avoid it; the leaf value is the only place the
  shrinkage can live. Do not advertise bit-exact LightGBM export for a shrunk
  model.

Every float the writer emits is checked: it is rendered, parsed back, and the
model is refused if the two differ. So a precision regression in Mojo's float
formatting surfaces as a loud error, never as a silently perturbed threshold.

### Explicit rejections (all raise, all named)

Reading: categorical splits (`decision_type` bit 0, `num_cat > 0`,
`cat_threshold`, `cat_boundaries`); linear trees (`is_linear=1`,
`leaf_const`, `leaf_coeff`); `missing_type=Zero`; a feature whose nodes
disagree on `missing_type`; `average_output` (random-forest boosting);
`multiclassova` and its aliases; `binary` with a non-unit `sigmoid`; an empty
`objective=` line (custom objective); unknown or unimplemented objectives
(delegated to `params.objective_from_name`, so its wording stays the single
source of truth); unsupported format versions (v2/v3/v4 are accepted); a
feature index past `max_feature_idx`; malformed child links (a node or leaf
reached twice, or never); a model needing more than 256 bins for one feature
(mojoboost bins are `UInt8`); non-finite thresholds.

Writing: categorical nodes; the `CUSTOM` objective; a `threshold_bin` with no
upper edge in the mapper; a non-finite or non-round-tripping value; a model
with no trees but a nonzero base score.

### Objective mapping

Read: `regression`/`regression_l2`/`l2`/`mse`/`mean_squared_error`,
`binary`, `poisson`, `huber`, `quantile`,
`mae`/`regression_l1`/`l1`/`mean_absolute_error`, `gamma`, `tweedie`,
`mape`, `fair`, `cross_entropy`/`xentropy`, `multiclass`/`softmax`,
`lambdarank`. All but `lambdarank` and `binary` go through
`params.objective_from_name`; `lambdarank` is handled here because
`objective_from_name` refuses it (there the name means "train this", which
needs query groups, whereas a fitted ranking model is just trees whose
prediction is the raw score).

Write: the inverse. **Known gap:** the scalar parameter of `huber`,
`quantile`, `fair`, and `tweedie` is not kept on a fitted `Booster` (it
shaped the gradients, not the trees), so the written `objective=` line omits
it and a LightGBM reader will report its own default. Predictions are
unaffected. If a later lane adds `alpha` to `Booster`, wire it into
`lgbm_objective_name`.

## Shared node vocabulary (coordination with task 14's dump schema)

Both lanes should name a node the same way. This module's docstring carries
the identical list; the two must be reconciled once, not translated
repeatedly. Canonical form is mojoboost's, with LightGBM's difference noted.

| Name | Meaning | LightGBM difference |
| --- | --- | --- |
| `node_index` | Position in a tree's flat node arrays. Parents always precede children, so an ascending scan is a valid top-down pass. | LightGBM numbers internal nodes and leaves in two separate spaces. |
| `is_leaf` | `feature[node] < 0`. | Implied by the sign of the child link. |
| `split_feature` | Column index at an internal node, -1 at a leaf. | Same, in `split_feature`. |
| `threshold` | Real-valued upper bound of the left branch; `<=` goes left. | Same. |
| `threshold_bin` | The same split in bin space. `threshold == edges[edge_offsets[split_feature] + threshold_bin]`. | Not present; derived. |
| `decision_type` | Numerical `<=` versus categorical set membership. | A bitfield that also carries `default_left` and `missing_type`. |
| `default_left` | Where a missing value goes. | `decision_type` bit 1. |
| `missing_bin` | The feature's reserved missing bin, -1 for none. | Replaced by `missing_type` in bits 2-3 (0 None, 1 Zero, 2 NaN). |
| `left_child` / `right_child` | `node_index` values. | A negative number is the leaf `~child` (-1 is leaf 0); a non-negative one is an internal-node index. |
| `leaf_ordinal` | A leaf's rank among its tree's leaves in `node_index` order (`Tree.leaf_ordinals`). | LightGBM's leaf index is a different numbering. This writer emits LightGBM leaves in preorder, which coincides with `leaf_ordinal` only by accident — **neither schema should present them as the same field.** |
| `value` | Node output: the leaf value at a leaf; at an internal node, the value it carried when created. | `leaf_value` / `internal_value`, both already shrunk. |
| `count` | Training rows reaching the node (the cover exact TreeSHAP conditions on). | `leaf_count` / `internal_count`. |
| `split_gain` | Gain recorded when the node was split. | Same. |

Two vocabulary points task 14 should mirror verbatim:

1. `threshold` and `threshold_bin` are two views of one split, related by the
   identity above. A dump schema that publishes only one of them forces every
   consumer to re-derive the other; publishing both with that identity stated
   is cheaper and unambiguous.
2. `leaf_ordinal` is mojoboost's own numbering and must never be labelled a
   LightGBM leaf id.

## Central integration required (not done by this lane)

None of this was edited; it is the exact work the integrator needs to do.

1. **`src/mojoboost/__init__.mojo`** — export the module so
   `from mojoboost import lgbm_model_io` works, alongside the existing
   `serialize` entry. Nothing else in the package imports it, so the addition
   is purely additive.

2. **`bindings/_mojoboost.mojo`** — add Python bindings mirroring the
   existing `save_model` / `load_model` pair:

   ```
   load_lgbm_model(path: str) -> Model
   load_lgbm_multiclass_model(path: str) -> MulticlassModel
   save_lgbm_model(model, path: str) -> None
   save_lgbm_multiclass_model(model, path: str) -> None
   lgbm_model_file_kind(path: str) -> str
   ```

   The string-in / string-out pair (`parse_lgbm_model`, `dump_lgbm_model`) is
   the one LightGBM's own Python API exposes as
   `Booster(model_str=...)` / `Booster.model_to_string()`, so bind those too
   if the wrapper aims at drop-in use.

3. **`python/mojoboost/__init__.py`** — surface them as
   `mojoboost.Booster.from_lightgbm_file(path)` /
   `.to_lightgbm_file(path)`, or as module-level
   `load_lightgbm_model` / `save_lightgbm_model`. Whichever shape is chosen,
   the docstring must repeat the two caveats above: the synthesized mapper is
   not the training binning, and export of a shrunk model is ULP-accurate
   rather than bit-exact.

4. **`capi/`** — no change needed unless the C ABI should gain
   `mojoboost_load_lgbm_model`. If it does, it needs the same
   `objective`/`multiclass` dispatch the existing loader has, via
   `lgbm_model_file_kind`.

5. **`docs/LIGHTGBM_PARITY.md`** — add a row for model-file interop stating
   what converts and what raises. The rejection list above is the content; do
   not claim arbitrary LightGBM models load, because categorical and linear
   models do not.

6. **`pixi.toml`** — add
   `mojo run -I src tests/parallel/test_lgbm_model_io.mojo` to the `test`
   task. Not done here: `pixi.toml` is a shared hotspot.

7. **`serialize.mojo`** — no change required. The two formats are
   independent; this module does not touch the `mojoboost` v3 format and
   nothing in `serialize.mojo` needs to know it exists.

## Focused test

Command (CPU throttle, single worker, run once after each fix):

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_lgbm_model_io.mojo
```

Result: **29 tests run, 29 passed, 0 failed, 0 skipped** (2.38 s).

`mojo` is not on `PATH` in this checkout, so the command goes through
`pixi run`; the throttle wrapper and `MOJOBOOST_NUM_WORKERS=1` are unchanged.

The test ran three times in total: once failing to launch (`mojo` not found),
once surfacing an ownership error in `_NodeArrays` plus the 1-ULP round-trip
finding, and once green. No other test, suite, build, benchmark, or parity
script was run.

`git diff --check --` over the three assigned paths reports no whitespace
errors.

Coverage: reading a two-tree regression model with hand-computed predictions;
the synthesized mapper's exact edges and offsets; node covers and preorder
node ordering; a single-leaf tree; all three missing-value behaviours
(`NaN` + default-left, `NaN` + default-right, `None` binning NaN as 0.0);
multiclass reading, softmax, and argmax; kind dispatch and both cross-loader
errors; a save/load file round trip; and thirteen rejection paths. On the
writing side: text stability under reparsing (single-output and multiclass),
prediction equality after a round trip, both writer rejections built from
hand-assembled `Tree` values, and a trained-model round trip on 240 rows with
missing values, checking the ULP bound on the first hop and bit-exactness on
the second.

## Risks and known gaps

- **Not validated against real LightGBM.** No LightGBM install was available
  and downloads were out of scope, so every fixture is hand-written from the
  format's specification. The reader is exercised only against text this
  lane composed, and the writer's output has never been fed to LightGBM. The
  first integration step should be a one-off manual check that
  `lightgbm.Booster(model_file=...)` accepts a file from
  `save_lgbm_model`, and that `parse_lgbm_model` reads a file LightGBM wrote.
  Treat the format claims here as careful and unverified against the real
  implementation.
- `leaf_weight` and `internal_weight` are written as zeros. mojoboost keeps
  node covers, not hessian sums, and LightGBM treats both fields as optional
  on load — but a consumer reading them will see zeros, not real weights.
- `feature_names` are written as LightGBM's own `Column_i` defaults, because
  a `Model` carries no feature names. If task 14 adds a `feature_name_`
  accessor, thread it into `_feature_names` and `_assemble`.
- `tree_sizes` is computed as the exact byte length of each emitted tree
  block, which is what LightGBM's fast loader expects, but that path is
  unverified for the reason above.
- Tree conversion recurses once per node in both directions. Depth is bounded
  by the tree's depth, so a pathologically deep chain (thousands of leaves in
  a single spine) could reach a stack limit. Real leaf-wise models do not.
- Categorical support is the largest gap. Reading it needs the inverse of
  `CategoricalSpec`: LightGBM's `cat_threshold` holds raw category codes,
  mojoboost's bitsets hold bins, and the code-to-bin table is not in the
  file. It can be reconstructed (the codes present in `cat_threshold` are a
  subset of the fitted categories), but only up to categories no split
  mentions, which changes which unseen codes route where. That is why it
  raises rather than guesses.
