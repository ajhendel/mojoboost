# Task 13 handoff: exclusive feature bundling (EFB) core

Status: the core landed, nothing is wired up, nothing is enabled.

Delivered in this lane, and nothing else:

- `src/mojoboost/efb.mojo` — conflict measurement, deterministic bundle
  construction, offset encoding, decoding, the bundled matrix, histogram
  reconstruction, memory accounting, and the fallback verdict.
- `tests/parallel/test_efb.mojo` — 21 tests, all passing under
  `mojo run -I src tests/parallel/test_efb.mojo`. That is the only command
  this lane ran.
- this file.

Everything below is work for other lanes or a later change. None of it was
touched here, because every item is a central or shared file.

---

## 1. What the core actually guarantees

Read the module docstring of `efb.mojo` first; it is the specification. The
four facts the rest of this document leans on:

1. **A feature's "zero" is its default bin** — the bin holding 0.0, i.e.
   `sparse.default_bins(mapper)[f]`, which is already what `SparseBinnedMatrix`
   carries in `default_bin`. A row is *non-default* for f when its bin differs
   from that. Absent entries and explicitly stored entries that bin to the
   default are indistinguishable, by construction and by test.
2. **Singleton bundles are identity encoded.** A bundle with one member has
   `slot_offset = 0`, keeps the member's default bin as the column default,
   and keeps its category table and reserved missing bin unchanged. So a plan
   whose bundles are all singletons reproduces the source matrix exactly (up
   to dropping stored default-bin entries), and a categorical or
   missing-reserving feature that is left unbundled costs nothing.
3. **Multi-member bundles reserve bundle bin 0** for "all members default"
   and give member k the range `[slot_offset[k], slot_offset[k] + bins[k]-1)`.
   Total bins `1 + sum(bins - 1)`, capped at 256 by the `UInt8` bin storage.
4. **Splits stay per-original-feature.** A bundle is a storage layout, never a
   change of hypothesis space. `unbundle_histogram` is the bridge: it returns
   a member's local histogram, with the member's default bin recovered by
   subtracting the member's own range from the bundle block total.

Two deliberate restrictions, both defaults, both reversible by a parameter or
by the follow-up work named below:

- categorical features are never bundled with anything (`efb.mojo` decides
  this unconditionally);
- a feature reserving a missing bin is not bundled unless
  `EfbParams.bundle_missing` is set, because a bundled column carries one
  `missing_bin` and a node learns one `default_left`.

---

## 2. Central imports and exports (not done here)

`src/mojoboost/__init__.mojo` must re-export the public surface, in the
alphabetical position the file already uses (between `.categorical` and
`.gain` reads naturally):

```mojo
from .efb import (
    EFB_MAX_BINS,
    EFB_NONE,
    EFB_SHARED_BIN,
    EfbParams,
    FeatureBundling,
    LocalHistogram,
    bundle_csc,
    conflict_count,
    feature_bin_count,
    fit_bundles,
    nondefault_rows,
    pairwise_conflict,
    unbundle_histogram,
)
```

`efb.mojo` imports `.binning` (`BinMapper`), `.categorical`
(`CategoricalSpec`), `.metrics` (`_argsort`), and `.sparse`
(`SparseBinnedMatrix`). It is a leaf: nothing in the package imports it, so
adding the export cannot create a cycle.

`pixi.toml` needs the test appended to the `test` task, after
`tests/test_sparse.mojo` (it depends on the same modules):

```
&& mojo run -I src tests/parallel/test_efb.mojo
```

CI runs `pixi run test`, so no separate workflow change is needed.
`tests/parallel/` is this round's shared convention — several lanes' tests
already live there alongside `api_snapshot_manifest.json` — so the path is
stable and needs no consolidation.

**`tests/parallel/api_snapshot_manifest.json` must be updated in the same
change as the `__init__.mojo` export.** The manifest's `mojo.exports_by_module`
block is a dict keyed by module name (33 modules today), sourced from
`__init__.mojo`, and its own `verification` block records that it checks "the
module set and every module's name set". Adding the `efb` export without
adding an `exports_by_module.efb` entry listing exactly the thirteen names
above will show up as API drift. That file belongs to another lane in this
round; do not edit it from here, but do not land the export without it
either.

---

## 3. Parameters

`enable_bundle` already has a row in `docs/LIGHTGBM_PARITY.md` (line ~337)
reading `deferred | Exclusive Feature Bundling. v1 work, task 13`. It stays
`deferred` until the wiring below lands; do not upgrade it on the strength of
this core alone, because nothing reaches it from a public entry point yet.

When wiring does land, `params.mojo` needs these keys in `TrainConfig` and
`parse_params`, LightGBM's names and defaults:

| key | type | default | meaning |
|---|---|---|---|
| `enable_bundle` | Bool | `true` in LightGBM | master switch; recommend defaulting to `false` in mojoboost until a benchmark justifies otherwise |
| `max_conflict_rate` | Float64 | `0.0` | `EfbParams.max_conflict_rate` |

`EfbParams` also carries `max_bundle_bins`, `max_bundle_size`,
`max_nondefault_rate`, `min_reduction`, and `bundle_missing`, which have no
LightGBM equivalent. Recommendation: do **not** expose them as string params
in the first pass. `max_bundle_bins` is forced by the storage type,
`max_nondefault_rate` mirrors LightGBM's internal 0.95 filter, and
`min_reduction`/`bundle_missing` are policy knobs whose right values are a
benchmark result, not a user decision. Keep them Mojo-API only
(`params._is_mojo_api_only`) if they must be reachable at all.

`TrainConfig` validation should reject `enable_bundle=true` together with a
dense-matrix entry point (see §4): bundling is a sparse-path feature.

---

## 4. Sparse integration (CSC/CSR binning)

This is the load-bearing part. The order matters.

**Fit time (CSC).** In `model_sparse.fit_csc` / `fit_multiclass_csc`, after
`fit_bins_csc` and `transform_csc` produce the `BinMapper` and the
`SparseBinnedMatrix`:

```mojo
var plan = fit_bundles(mapper, data, efb_params)
var train_data = bundle_csc(data, plan) if plan.use_bundling else data^
```

Do not consult anything but `plan.use_bundling` for the decision; it already
folds in the column-count reduction and the rectangular-histogram widening
(`histogram_slots_unbundled` vs `histogram_slots_bundled`). If a caller wants
to force bundling, it should pass `min_reduction = 0.0` and read
`n_bundles() < n_features`, not second-guess the verdict.

**The plan must travel with the model.** It is a function of the mapper and
the training matrix, and prediction cannot re-derive it: a scoring matrix has
different sparsity. So `Model` and `MulticlassModel` gain an optional
`FeatureBundling` alongside their `BinMapper`. Suggested shape, which keeps
the no-bundling path byte-identical:

```mojo
struct Model:
    var mapper: BinMapper
    var bundling: FeatureBundling   # a plan with use_bundling == False means "none"
    var booster: Booster
```

An all-singleton plan with `use_bundling = False` is the natural "no
bundling" value and needs no `Optional`.

**Dense path.** `binning.fit_bins` / `BinMapper.transform` produce a
`BinnedMatrix` with no `default_bin` table, because a dense matrix has no
notion of absence. EFB on the dense path would first have to define the
default bin per feature (`mapper.bin_value(f, 0.0)`, exactly what
`sparse.default_bins` computes) and then measure conflicts by a full column
scan. That is possible and cheap, but it buys much less: the win is
proportional to how many columns are almost always default. **Leave the dense
path alone** in the first pass and raise from `params` if `enable_bundle` is
combined with `fit` / `train`.

**CSR.** `CsrMatrix` is prediction-side only and is transposed to CSC by
`to_csc` before any binning, so nothing on the CSR ingest path changes.
Prediction is covered in §6.

**Histograms.** `histogram_sparse.build_histogram_sparse*` are
column-oriented and read `default_bin`, `col_offsets`, `row_index`, and
`bin`. A bundled `SparseBinnedMatrix` satisfies that contract unchanged, so
they need **no** modification: they accumulate over bundle bins, which is
exactly the intended speed-up (fewer columns, one pass each). The bundled
matrix's `n_bins` is `plan.max_bundle_bins()`, so the histogram is
`n_bundles * max_bundle_bins` wide instead of `n_features * n_bins`.

**Distributed / GPU.** `collective.mojo` and `histogram_gpu.mojo` /
`train_gpu.mojo` reduce and lay out histograms by `(feature, bin)`. A bundled
matrix is still just a matrix with fewer, wider columns, so the reductions
are shape-agnostic and correct. But `train_gpu`'s kernels are written against
the dense `BinnedMatrix`, so bundling and GPU training do not meet yet;
`resolve_device` should keep sending bundled training to the CPU trainer
until the sparse GPU path exists.

---

## 5. Split recovery

`split.find_best_split` scans a feature's bins as an ordinal range and
`categorical.find_best_categorical_split` searches sets. Neither is valid
across a multi-member bundle: a threshold spanning two members compares bin
ids of unrelated features.

The required change is confined to the scan loop, and it is the reason
`unbundle_histogram` exists.

1. For a **singleton** bundle, nothing changes. The column is the feature,
   identity encoded, category table and missing bin intact. This is why
   singleton identity encoding is worth its extra branch.
2. For a **multi-member** bundle, iterate members. For member rank k, call

   ```mojo
   var local = unbundle_histogram(
       plan, bundle, k, hist.grad, hist.hess, hist.count,
       bundle * hist.n_bins,
   )
   ```

   and run the existing numerical scan over `local.grad/hess/count`, which is
   indexed by the member's own local bins. The winning `SplitInfo` must then
   record the **original** feature index, `plan.member_at(bundle, k)`, and the
   **original** local bin threshold. Nothing downstream ever sees a bundle id.

That last sentence is the whole design contract: bundles exist between
`bundle_csc` and the split scan, and nowhere else. `Tree.feature` keeps
holding original feature indices, so §7, §8, §9, and §10 below need no
translation layer at all — only the row-routing in §6 does.

Consequences to check when implementing:

- `unbundle_histogram` costs O(bundle_bins) per member because of the block
  total. Hoist the total out of the member loop when doing this for real; the
  core keeps it inside for clarity, and a lane that hoists it must keep the
  subtraction order identical or the last bits of the default bin will move.
- The reconstruction is exact only when `plan.is_lossless()`. With
  `max_conflict_rate > 0`, a collision the member lost is folded into that
  member's default bin. That is the documented approximation; a training path
  that wants exactness should assert `plan.is_lossless()`.
- Interaction constraints (`interaction.mojo`) and feature sampling
  (`sampling.mojo`) mask by **original** feature index. With bundling, the
  masks have to be applied per member inside the bundle loop, not per column,
  or a masked feature will be scanned because a bundle-mate was allowed. This
  is the single easiest thing to get wrong in the whole integration.
- Monotone constraints (`monotone.mojo`) are per original feature and reach
  the scan through the same per-member path; no change beyond indexing by
  `plan.member_at(...)` instead of by the column.
- `min_data_in_leaf` and the Hessian floors read counts out of the local
  histogram, which `unbundle_histogram` fills exactly. No change.

---

## 6. Prediction

Training-time routing (`tree_sparse.grow_tree_sparse`) partitions rows by the
column it split on. Since splits record original features, the partition step
must map feature → bundle → bundle bin → decoded local bin, or, more
cheaply, precompute per-node the set of bundle bins that route left. The
second is preferable: `plan.encode` per row per node is a binary-search-free
but branchy inner loop, whereas a 256-bit routing mask per node is one test.

Prediction-time routing:

- `tree_sparse.predict_row_sparse` / `predict_row_sparse_csc` and
  `sparse.SparseBinnedRows.bins_of_row` return one bin per **column**. With
  bundling they return one bin per **bundle**, so the tree walk, which asks
  for `feature`, must go through the plan:

  ```mojo
  var b = bins[plan.bundle_of[f]]
  var local = plan.decode_bin(plan.bundle_of[f], b)
  if plan.decode_feature(plan.bundle_of[f], b) != f:
      local = plan.slot_default[plan.slot_of[f]]   # this feature is at its default
  ```

  The `decode_feature != f` branch is not an edge case, it is the common
  case: in a bundle of ten features, nine of them are at their default in any
  given row. Implement it as one lookup, not two.
- `model_sparse.predict_*_csr` binds raw CSR rows through `_row_bin`, which
  calls `mapper.bin_value` per stored entry. That produces **local** bins, so
  it must then be bundled: `plan.encode(f, local)`, with the bundle's default
  for every feature the row does not store. Unlike the training side there is
  no collision to resolve, because a prediction row can legitimately be
  non-default on two bundle-mates — and when it is, **one of them must be
  dropped, exactly as in training**. Use the same rule (earliest member in
  bundle order wins) or predictions will disagree with the trees. Put that
  rule in one shared helper rather than duplicating it in `bundle_csc` and
  in the predict path.
- `Model.predict` / `predict_raw` / `predict_range` / `leaf_indices` take
  dense rows through `mapper.bin_row`. Same treatment: bin to local, then
  encode through the plan.

---

## 7. Feature importance

`importance.split_importance` and `importance.gain_importance` read
`Tree.feature` and bound it against `n_features`. Because splits record
original feature indices (§5), **both work unchanged and report per-original
feature**. That is the correct answer and the reason for the contract.

The only thing to add is a regression test: fit the same data with and
without bundling and assert the importances match. If they do not, a bundle
id leaked into `Tree.feature`.

---

## 8. TreeSHAP

`contrib.mojo` is routing-based: the recursion only asks `Tree.goes_left`,
and it attributes to `Tree.feature`. So exact contributions come out
per-original-feature with **no change to contrib.mojo at all**, provided
`goes_left` receives the row's bins in the same representation the tree was
grown with. Concretely:

- the caller that builds the per-row bin vector for `contrib` must apply the
  same bundle encode/decode as §6, or
- better, decode once into original local bins before calling contrib, so
  contrib keeps seeing `n_features` bins per row.

The second is strongly preferred: contrib is O(L·D²) per tree and not on the
hot path, so paying one decode per row is free, and it keeps the exactness
argument in the contrib docstring (efficiency: contributions plus expected
value sum to the raw score) valid without re-deriving it for bundles.

One genuine caveat to write into `contrib.mojo`'s docstring when this lands:
with `max_conflict_rate > 0` the model was trained on a matrix where some
values were dropped, so contributions explain *that* model faithfully — the
Shapley properties still hold exactly — but the model is an approximation of
the unbundled one. Do not let anyone read the collision loss as a contrib
bug.

---

## 9. Model dumping

Whatever dumps trees (the C API's text dump, the CLI's model output, and
`Model.write_to`) prints original feature indices already, per §5, so the
dump format does not change. Two additions are worth making:

- print the bundling summary alongside the mapper: `n_bundles`,
  `max_bundle_bins`, `total_collisions`, and whether `use_bundling` was
  honored. A model whose predictions carry a lossy bundling should say so in
  its own dump.
- `FeatureBundling` has no `Writable` conformance. Add `write_to` /
  `write_repr_to` when the dump needs it; do not add it speculatively.

---

## 10. Serialization

`serialize.mojo` is at `_VERSION = "v3"`. Bundling needs **v4**, and it must
follow the established pattern exactly: a new section that is written only
when there is something to write, so a model with no bundling serializes to
the bytes it does today and v1/v2/v3 files keep loading.

Before touching `_VERSION`, read the `inspection.py` bullet in §11. The
Python side re-implements this reader and hard-refuses unknown versions, so
the v4 bump is not a Mojo-only change and cannot land on its own.

Both `save_model` and `save_multiclass_model` emit sections in a fixed order:
`_write_mapper`, `_write_categorical`, `_write_monotone`, `_write_trees`. The
new section goes between `_write_categorical` and `_write_monotone`, i.e. a
`_write_bundling` call in both writers and a `_read_bundling` peeking for the
`bundling` tag the way `_read_monotone` peeks for `monotone`:

```
bundling <n_bundles> <n_features> <n_rows> <n_bins> <source_entries> <bundled_entries> <use_bundling>
<bundle_start ...>            # n_bundles + 1 ints
<members ...>                 # per member slot
<slot_offset ...>             # per member slot
<slot_bins ...>               # per member slot
<slot_default ...>            # per member slot
<slot_missing ...>            # per member slot
<bundle_bins ...>             # per bundle
<collisions ...>              # per bundle
```

Notes for whoever writes it:

- `bundle_of` and `slot_of` are derivable from `members` + `bundle_start`;
  rebuild them on load rather than storing them, the way the loader already
  rebuilds `_sized_missing_bins`.
- Everything here is an integer, so no `_f64_to_token` bit-pattern encoding
  is needed. `use_bundling` is a Bool; follow whatever the monotone section
  does for flags.
- Call `plan.validate()` immediately after loading. It catches every
  structural corruption the format can express — a feature in two bundles,
  ranges that do not tile the bundle's bins, an offset that disagrees with
  the member widths — before a single bin is decoded. A hand-edited or
  truncated file must raise, not mis-route.
- A model that trained with a lossy bundling
  (`total_collisions() > 0`) is still fully described by this section; the
  loss is in the trees, not in the plan.
- `BinMapper.matches` is what lets `train_more` add trees to a separately
  binned dataset. Bundling needs the same: a `FeatureBundling.matches` that
  compares members, offsets, widths, and defaults, and `boosting.train_more`
  must refuse to continue when the plan differs, or bin ids will silently
  mean two different features.

The `Model`/`MulticlassModel` field added in §4 is what the writer and reader
serialize; nothing else in the format moves.

---

## 11. Bindings

None of these were touched.

- `bindings/_mojoboost.mojo` and `python/mojoboost/basic.py`: expose
  `enable_bundle` and `max_conflict_rate` as ordinary params through the
  existing param-string plumbing (`params.parse_params`), so the Python side
  needs no new call. If the plan summary is worth surfacing, add a read-only
  accessor returning `(n_bundles, max_bundle_bins, total_collisions,
  use_bundling)` rather than exporting the struct.
- `python/mojoboost/_sklearn.py`: the estimator documents passing shared
  parameters through `**kwargs` rather than naming each one, so
  `enable_bundle=False` should reach the param string with no signature
  change. Confirm against the manifest before assuming it: the snapshot
  checks `python.shared_estimator_parameters` names *and* every default
  value, so a new named parameter is a snapshot change and a bare `**kwargs`
  passthrough is not.
- `capi/` and `cli/`: both go through `params.parse_params`, so both get the
  keys for free. `cli` should print the bundling summary in its model dump
  (§9). The C header needs no new symbol.
- `python/mojoboost/inspection.py`: **this is the one that breaks.** It is not
  a thin wrapper — it is an independent Python re-implementation of the model
  text reader (`parse_model_string`, `_parse_mapper`, `_parse_categorical`,
  `_parse_monotone`, `_parse_trees`), and it gates on

  ```python
  SUPPORTED_MODEL_FORMAT_VERSIONS = (1, 2, 3)
  ```

  raising `ValueError("model format v4 is newer than this build reads")` for
  anything else. So the moment `_VERSION` goes to v4 (§10), **every** v4 model
  becomes unreadable to `dump_model`, `trees_to_records`,
  `trees_to_dataframe`, and `split_values`, whether or not it has any
  bundling. The serialization change and this file must land together:
  extend the version tuple and add a `_parse_bundling` that mirrors
  `_parse_categorical`/`_parse_monotone`, including the optional-section peek
  so v1–v3 files keep parsing.

  Once that is done the *reported* surface really is unchanged, because
  splits carry original feature indices (§5, §7, §8) — `_threshold_value`
  and `_bin_value` reconstruct raw thresholds from a bin id through the
  mapper, which is only correct while `Tree.feature` and the stored threshold
  stay per-original-feature. Add a test asserting the dump matches between a
  bundled and an unbundled fit of the same data; that assertion is what
  catches a bundle id leaking into a tree.

---

## 12. Docs

- `docs/LIGHTGBM_PARITY.md`: the `enable_bundle` row exists and stays
  `deferred` until §4–§6 land. When they do, the row moves to `partial`
  (sparse path only, CPU only) with a note pointing at `src/mojoboost/efb.mojo`
  and `tests/parallel/test_efb.mojo`, and `max_conflict_rate` gets its own
  row. Run `python3 tools/check_parity.py` in the same change; it verifies
  the cited files exist and the named public symbols are still there.
- `README.md`: a short subsection under the sparse documentation, stating
  what bundling does, that it is off by default, and the two restrictions
  (categorical never bundled, missing values opt-in).
- `docs/COMPATIBILITY_POLICY.md`: §1.4 "Version numbers that move
  independently" carries the row `Model format version | _VERSION in
  src/mojoboost/serialize.mojo | v3 | The file format gains or changes a
  section`. Bump the `Current` cell to v4 in the same change as §10. §3.1
  also records that a model format section is never removed, which is the
  rule the optional-section design already satisfies.

---

## 13. Open questions this lane deliberately did not answer

1. **Should `max_conflict_rate > 0` be reachable from the param string at
   all?** It trades exactness for columns, and the loss is invisible in the
   metrics. The core supports it, counts it exactly (`collisions`,
   `is_lossless`), and defaults it off. Someone should benchmark before
   exposing it.
2. **`bundle_missing`.** Making it usable means generalizing the split
   search from one `missing_bin` and one `default_left` per node to a set of
   missing bins per column. `plan.missing_bins(bundle)` already returns that
   set. Until then, features with missing values simply do not bundle, which
   costs bundling opportunities on exactly the sparse data EFB is for.
3. **Dense-path EFB.** Cheap to add (§4), unclear payoff. Measure first.
4. **GPU.** Bundling and `train_gpu` do not meet; the bundled matrix is
   sparse and the GPU trainer is dense. Not a blocker, but state it in the
   parity row so `partial` is honest.
5. **Where the ordering heuristic should live.** The greedy visits features
   by descending non-default count. LightGBM sorts by conflict-graph degree
   for its non-approximate path. If a benchmark shows the degree order packs
   better, it is a localized change to the `_argsort` key in `fit_bundles`
   and nothing else — but it would change every existing plan, so it needs
   the serialization `matches` check from §10 in place first.
