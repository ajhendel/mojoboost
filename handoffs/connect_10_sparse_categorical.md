# Connect 10: sparse and categorical training across CPU and GPU

Lane ownership: `src/mojoboost/sparse.mojo`, `histogram_sparse.mojo`,
`tree_sparse.mojo`, `boosting_sparse.mojo`, `model_sparse.mojo`,
`gpu_sparse.mojo`, `gpu_sparse_layout.mojo`, `gpu_categorical.mojo`, and this
file. Nothing outside that list was edited. No tests were written or run, no
build was invoked, nothing was committed by this lane.

**Nothing here is a correctness, performance, parity, or hardware claim.**
Every statement below is from static reading. Every command in the last
section is UNRUN.

## 1. What already existed

### CPU sparse, complete and connected
- `sparse.mojo` CSC/CSR storage and validation, sparse quantile binning
  (`fit_bins_csc`), sparse categorical spec fitting
  (`fit_categorical_spec_csc`), sparse binning (`transform_csc`),
  `SparseBinnedMatrix` and its row view.
- `histogram_sparse.mojo` node/subset/full accumulators, the shared entry
  permutation (`SparseEntryOrder`), per-node entry windows
  (`SparseNodeEntries`), the stable segmented partition.
- `tree_sparse.mojo` leaf-wise growth, sharing `tree._search`,
  `tree._leaf_value`, monotone bounds, interaction constraints, feature
  subsampling, and histogram subtraction verbatim with the dense grower.
- `boosting_sparse.mojo` the boosting loop, leaf renewal, early stopping,
  multiclass.
- `model_sparse.mojo` `fit_csc` / `fit_multiclass_csc` / `predict_*_csr`.

This path was already reachable and already produced ordinary `Model` /
`Booster` values, so nothing about serialization changes in this lane.

### GPU sparse and categorical, real but entirely unreferenced
- `gpu_sparse_layout.mojo` device layout accounting, hard support codes, the
  per-node cost model, `SparseRangeTable`, EFB bundle compatibility.
- `gpu_sparse.mojo` seven kernels, `GpuSparseHistogramBuilder` (device
  resident matrix, per-node histogram, row + entry partition at a split),
  host reference models.
- `gpu_categorical.mojo` `CatSetPool`, `_cat_pool_side_kernel`, the category
  statistics kernels, `enqueue_category_stats`, `category_stats_host`.

A repo-wide search found **no non-test, non-doc caller of any of these three
modules**. They were not exported from `src/mojoboost/__init__.mojo` either.

### The disconnections that mattered
1. `boosting_sparse.mojo`'s docstring claimed categorical features were "not
   available on the sparse path (rejected when the data is binned)". They
   are available and were already fully wired: `transform_csc` bins them and
   `tree_sparse` passes `cats=data.cats` into `_search`. The documentation
   was the only thing rejecting them.
2. The sparse categorical rules had exactly one statement, and it lived in
   `gpu_categorical.mojo` behind `max.gpu` imports, so no CPU caller could
   reach it and nothing called it.
3. `SparseBinnedMatrix` had no validator. The GPU builder checked a subset
   inline; the CPU trainers checked nothing.
4. `train_multiclass_sparse` dropped `params.tree.monotone` when building its
   `MulticlassBooster`, so a sparse-trained multiclass model serialized as
   an unconstrained one.
5. `train_sparse` / `train_sparse_with_valid` had no `class_bagging`;
   `train_multiclass_sparse` had no `goss`, and open-coded the softmax
   gradient fill that `boosting._fill_softmax_grad_hess` already owns.
6. `CatSetPool` had no consumer and `_cat_pool_side_kernel` had no host
   launcher: a pool could be staged and uploaded and nothing would ever
   route by it.
7. `enqueue_category_stats` had no caller, because nobody owned the output
   plane or filled the node totals it subtracts against.
8. Nothing anywhere said, as a value, "the sparse GPU primitives exist and
   the sparse GPU training path does not". `device_policy.mojo` said the
   opposite ("there is no sparse GPU histogram kernel"), which stopped being
   true when this lane's primitives landed.
9. `efb.bundle_csc` produced a `SparseBinnedMatrix` and no sparse trainer
   could consume one correctly, yet `train_sparse` accepted it without
   complaint — an unguarded trap rather than a missing feature (section 3.8).
   `efb.check_bundling_honored`, which exists to turn exactly that silence
   into an error, had no caller anywhere in the repository.

## 2. Call path, before and after

**Before**

```
CSC in -> fit_bins_csc -> transform_csc -> train_sparse
       -> grow_tree_sparse -> build_histogram_sparse_node -> Model
(no validation of the binned matrix anywhere)

gpu_sparse_layout / gpu_sparse / gpu_categorical : no callers at all
```

**After**

```
CSC in -> fit_bins_csc -> transform_csc
       -> prepare_bundling_csc(mapper, binned, params.bundling)   <- new
             |  enable_bundle off (default) -> matrix untouched,
             |                                 inactive view
             '- enable_bundle on -> efb.fit_bundles -> efb.bundle_csc
                                 -> bundled matrix + SparseBundling view
       -> SparseBinnedMatrix.validate()          <- new, once per fit
       -> train_sparse | train_sparse_with_valid | train_multiclass_sparse
             (each takes the SparseBundling and passes it down)
       -> grow_tree_sparse
             -> _node_histogram = build_histogram_sparse_node (per bundle
                column) + efb.expand_bundled_histogram (back to per feature)
             -> tree._search / tree._leaf_value, both in original space
       -> Model, in the original feature space, plan dropped

sparse.default_category_bin / absent_is_unknown /
check_sparse_categorical_semantics                <- moved here from
                                                     gpu_categorical
        consumed by SparseBinnedMatrix.validate, by
        gpu_sparse_layout.sparse_gpu_capability, and by gpu_categorical

DeviceCaps + SparseBinnedMatrix
       -> gpu_sparse_layout.sparse_gpu_capability -> SparseGpuCapability
       -> check_sparse_gpu_histograms  (raises unless primitives can run)
       -> check_sparse_gpu_training    (ALWAYS raises today)
                 ^
                 |
       GpuSparseHistogramBuilder.__init__ now goes through it and keeps the
       record in `self.capability`

GpuSparseHistogramBuilder.apply_split ---.
                                          >-- enqueue_default_side
gpu_categorical.apply_categorical_split_pooled  -- + side kernel
                                          '-- finish_split (one partition,
                                              one range bookkeeping)

GpuSparseHistogramBuilder.enqueue_node_totals <- shared by enqueue_leaf and
                                                 by GpuCategoryStats.compute
```

## 3. Connections completed

### 3.1 One authoritative validator for the binned sparse form
`SparseBinnedMatrix.validate()` (new, `sparse.mojo`) checks dimensions, bin
count, offset shape and monotonicity, `default_bin` / `missing_bin` table
sizes and ranges, `missing_bin != default_bin` per feature, row indices in
range and strictly ascending per column, stored bins in range, and the
categorical rules. Called once per fit by `train_sparse`,
`train_sparse_with_valid` (both matrices), and `train_multiclass_sparse`, and
by `sparse_gpu_capability` before the device builder uploads anything. It is
O(nnz), so it is deliberately per-fit and not per-tree.

The GPU builder's inline structural checks were **removed** in favour of it;
what remains there is only the device-specific half (Int32 index widths, bin
ceiling, shared memory, 256-bit category set), and that now runs through the
capability record.

### 3.2 Categorical semantics moved to the layer that owns them
`default_category_bin`, `absent_is_unknown`, and
`check_sparse_categorical_semantics` moved from `gpu_categorical.mojo` to
`sparse.mojo`, unchanged in behaviour. They are facts about the
representation, not about a device, and keeping them behind `max.gpu` imports
is what made them unreachable. `gpu_categorical.mojo` imports
`default_category_bin` from `sparse.mojo` and derives an absent row's bin
from the category table rather than trusting the uploaded column.

**Import path change:** these three names are no longer defined in
`gpu_categorical`. Import them from `.sparse`. No test, binding, Python
module, or other source file referenced them, so nothing downstream breaks;
the export block in section 5.1 already reflects the new home.

### 3.3 CPU sparse trainer reaches the samplers the dense one reaches
- `train_sparse` and `train_sparse_with_valid` gained
  `class_bagging: ClassBaggingParams`, validated through
  `boosting._check_class_bagging` (which is what makes it exclusive with
  `bagging` and `goss` and restricts it to binary classification), drawn
  through `sampling.refresh_class_bag`, gated by `has_positive_rows` exactly
  as `boosting._boost_rounds` gates it, and folded into the
  round-skip condition (`... or balanced`).
- `train_multiclass_sparse` gained `goss: GossParams`, using
  `boosting._multiclass_goss_select` + `goss.apply_goss_scaling` + the
  shared `GossSelection`, so the sample, the scaling, and the round-skipping
  rule are the dense implementation rather than a second copy. It also now
  calls `_check_goss` and `params.tree.monotone.check_features`.
- `train_multiclass_sparse`'s open-coded softmax gradient/hessian loop was
  replaced by `boosting._fill_softmax_grad_hess`. Static reading says the
  two were arithmetically identical (`w * (p - y)`, `w * max(2p(1-p),
  1e-16)`), so this is a fuse and not a behaviour change; that equivalence
  is on the UNRUN check list.
- `model_sparse.fit_csc` gained `init_score` and `class_bagging` pass-through;
  `fit_multiclass_csc` gained `goss`.

### 3.4 A sparse-trained multiclass model keeps its constraints
`train_multiclass_sparse` now passes `params.tree.monotone.copy()` to
`MulticlassBooster`, as the dense path does. Before this, the ensemble
recorded no constraints, which (a) serialized a constrained model as an
unconstrained one and (b) would let `train_multiclass_more` accept a
different constraint vector for continued rounds. **This changes serialized
output** for sparse multiclass fits with monotone constraints: the monotone
block now carries the signs instead of the empty default. Non-monotone fits
are unaffected.

### 3.5 One capability record, honest about training
New in `gpu_sparse_layout.mojo` (pure host arithmetic, no device needed):

- `categorical_support(cats, n_features, n_bins) -> Int`, the non-raising
  form of the existing `check_categorical_support`, plus the new code
  `SPARSE_CATEGORY_SET_OVERFLOW = 9` so the 256-bit-set limit has a name.
  `check_categorical_support` is now a thin raising wrapper over it.
- `sparse_gpu_training_is_wired() -> Bool`, constant `False`. Deliberately a
  function: it is the single place a future trainer flips, and forgetting to
  flip it fails closed.
- `SparseGpuCapability {support, categorical, histograms, training,
  unknown_absent}` with `blocked_reason()` and `explain()`.
  `histograms` and `training` are separate fields precisely so neither can
  be read as the other.
- `sparse_gpu_capability(caps, data)`, `check_sparse_gpu_histograms(...)`,
  `check_sparse_gpu_training(...)`.

`check_sparse_gpu_training` **always raises** today: an unsupported shape
raises with its own reason, and a fully supported one raises saying the
primitives exist but nothing drives them. There is no path through this lane
that reports GPU and runs on the CPU.

`GpuSparseHistogramBuilder.__init__` now constructs through
`check_sparse_gpu_histograms` and stores the record in `self.capability`, so
holding a builder is still not a claim that training is wired.

### 3.6 The categorical pool and the category statistics now have callers
- `GpuSparseHistogramBuilder` gained `check_split_ids`,
  `enqueue_default_side(parent, goes_left)`, and
  `finish_split(parent, left, right, expected_left)`, all factored **out of**
  `apply_split` with no behaviour change: `apply_split` now composes them.
- `gpu_categorical.apply_categorical_split_pooled(builder, pool, set_offset,
  feature, parent, left, right, expected_left)` is new. It launches
  `_cat_pool_side_kernel` (which had no launcher) over the split feature's
  entry window, decides the absent rows' side with `CatSetPool.contains` on
  the feature's default bin, and then calls `builder.finish_split`. The
  pooled and four-argument forms therefore share one partition, one range
  table, and one set of reserved-id rules; only the mask computation differs.
  It validates the set through `check_cat_bitset` and refuses an offset
  outside the pool's uploaded region.
- `gpu_sparse` gained `enqueue_node_totals(node)`, factored out of
  `enqueue_leaf` (which now calls it). `gpu_categorical.GpuCategoryStats`
  owns the `3 * n_bins` output plane, drives that same totals reduction, calls
  the previously caller-less `enqueue_category_stats`, and converts the
  download through `category_stats_from_fixed`. It is constructed **from the
  builder**, so its buffer cannot land on a different `DeviceContext` than
  the kernels that fill it.

### 3.7 Documentation that was actively wrong
`boosting_sparse.mojo`'s "not available" list now states what is actually
supported (categorical, class bagging, GOSS, the whole `extra` bundle) and
what is not (custom objectives, the GPU backend), and points at the
capability record rather than asserting the absence of a kernel.
`gpu_sparse.mojo` and `gpu_categorical.mojo` docstrings were updated to match
what they now do.

### 3.8 EFB connected to sparse growth

This was deferred in the first pass and then explicitly requested. It is now
wired, and the trap the deferral was protecting against is closed by
construction rather than by documentation.

**The trap.** `efb.bundle_csc` produces a `SparseBinnedMatrix` that
`train_sparse` accepted as-is. Fed one, `grow_tree_sparse` would have split on
*bundle* columns, so `Tree.feature` would have held bundle ids that
importance, inspection, model dump, and dense prediction all read as original
feature ids. Nothing would have raised.

**The design, and why it differs from the dense one.** The dense path
(`efb.BundledMatrix`, `tree._hist_full` / `_hist_subset`) keeps **both**
matrices: it accumulates from the bundled one and partitions rows on the
original one. Keeping the original CSC alongside the bundled one would give
back exactly the memory sparse bundling exists to save, so the sparse path
keeps **only** the bundled matrix and decodes stored entries through a
resolved view. Everything else follows the dense path exactly.

New in `tree_sparse.mojo`:

- **`SparseBundling`** — a `FeatureBundling` resolved against the bundled
  matrix it produced, plus the original-space metadata a grower reads.
  `SparseBundling.of(plan, data)` rebuilds `missing_bin`, `default_bin`, and
  the `CategoricalSpec` in the original feature space, and precomputes flat
  `(column, column bin) -> (owning slot, that member's local bin)` tables.
  `of` refuses a plan paired with the matrix it was *fitted* on rather than
  the one it *produced* (three shape checks), because that pairing would
  decode every bin against the wrong column widths, silently.
  - `column(feature)` — which matrix column a feature's entries live in.
  - `local_bin(feature, column_bin)` — that feature's own bin. A bin owned by
    another member, the shared bin, and a bin past the column's width all
    resolve to the feature's **default** bin, which is what each of them
    means: the row stores nothing for this feature.
  - `local_table(feature, n_bins)` — the above as a table, built once per
    split, so routing a bundled split costs one lookup per entry, the same as
    the direct bin read it replaces.
  - `columns_for(features)` — a column is accumulated when *any* member was
    picked by feature subsampling.
  - `source_bins(data)` — the bin count the *original* features were binned
    into (`plan.n_bins`). A bundled matrix is only as wide as its widest
    bundle, so its own `n_bins` is not what a validation matrix binned by the
    same mapper compares against.
  - `none()` is the inactive, unresolved value; both mappings are the
    identity on it. This is what makes the bundled and unbundled paths one
    path rather than two.
- **`_node_histogram(...)`** — accumulate the node over bundle columns
  (`build_histogram_sparse_node`, O(nnz_in_node) over as many columns as
  there are bundles), then `efb.expand_bundled_histogram` straight back into
  per-feature shape. **This is where a bundle stops existing.**
  `tree._search`, `tree._leaf_value`, and `subtract_histogram` below it read
  the shape they have always read, and neither `_search` nor
  `split.find_best_split` gained a parameter. Sibling subtraction survives
  because the expansion is linear: expanding parent and child and subtracting
  gives what subtracting first and expanding the difference would.
- **`grow_tree_sparse(..., bundling=SparseBundling.none())`** now keeps two
  feature counts apart deliberately: `n_features` (original — every
  parameter, constraint, split, and histogram) and `n_columns` (the matrix's
  own — only entry bookkeeping and accumulation). Split *application* is the
  one place that mixes them, and it does so explicitly: entries are read from
  `bundles.column(split.feature)` and each stored bin is put through
  `local_of`, so a bin belonging to another member of the column becomes
  "this feature is at its default" — the same fold `_node_histogram` made
  when the histogram this split was chosen from was expanded. An unresolved
  view is resolved here as an inactive one.
- **`predict_row_sparse` / `predict_row_sparse_csc`** take the view too, for
  the one caller that predicts the *bundled* matrix (the boosting loop's own
  score update). Predicting an unbundled matrix leaves it at `none()`.

New in `boosting_sparse.mojo`:

- **`SparseBundledData`** — the bundled matrix and its view together. Unlike
  `efb.BundledMatrix` it holds one matrix, not two.
- **`prepare_bundling_csc(mapper, data, settings)`** — the single entry
  point. Off (the default), it returns the matrix untouched with an inactive
  view. On, it runs `settings.check()`, `efb.fit_bundles`, and
  `efb.bundle_csc`, validates the result, and resolves the view. A plan that
  declines to bundle (`use_bundling == False`, which `fit_bundles` decides
  from its own histogram-slot cost model) returns the untouched matrix, so
  the fallback is inside the plan rather than bolted on.
- **`_resolve_bundling(...)`** — called by all three trainers right after
  `validate()`. A resolved view is checked against the matrix; an unresolved
  one goes through **`efb.check_bundling_honored(settings, trainer)`**, which
  had no caller anywhere in the repository before this. This is the second
  half of closing the trap: a caller who sets `enable_bundle` and then hands
  the matrix to a trainer that cannot honour it now gets a clear error
  instead of a silent drop.
- `train_sparse`, `train_sparse_with_valid`, and `train_multiclass_sparse`
  each take a trailing `bundling` and thread it into `grow_tree_sparse` and
  `_add_tree_scores`. Monotone constraints are checked against
  `bundles.n_features`, and `train_sparse_with_valid` compares the validation
  matrix's `n_bins` against `bundles.source_bins(data)` rather than the
  training matrix's own.

`model_sparse.fit_csc` / `fit_multiclass_csc` fit the plan from
`params.bundling` and **drop it** before returning: `mapper` and the trees are
in the original feature space, so `save_model` is byte-identical to an
unbundled fit's and no model carries a plan. Multiclass shares one plan across
every class's tree in every round.

**Bounded by construction.** Only lossless plans are usable
(`EfbSettings.check` refuses `max_conflict_rate > 0.0`), so routing-by-decode
and accumulation-by-expansion agree bin for bin, and both agree with an
unbundled fit. A categorical feature is always a singleton bundle under
identity encoding (`efb.mojo` guarantees this unconditionally), so a bundled
column is never categorical and `SparseBundling.of` rebuilds the original
`CategoricalSpec` exactly. **None of this is verified; see section 11.**

## 4. Extra tree parameters on the sparse path

Nothing needed connecting; this is a **verification result**, recorded because
the task asked for it. `grow_tree_sparse` already:

- calls `params.extra.check(n_features, num_leaves, max_depth,
  min_data_in_leaf)` before the first histogram;
- passes `grower_applies_extra=True` to every `_search` call, which is what
  `tree._search` requires before it will honour a bundle where
  `needs_grower_support()` holds;
- applies `max_delta_step` and `path_smooth` through `_leaf_value` at the
  root and both children, in the cap-then-smooth-then-monotone-clamp order
  the dense grower uses, with `parent_output` threaded correctly;
- passes `node=` and `tree_index=` so `extra_trees`' threshold draw is keyed
  by (seed, tree, node, feature) rather than collapsing to one stream;
- leaves `min_gain_to_split`, `monotone_penalty`, `feature_contri`, and the
  per-split CEGB cost to `_search`, where they are live for every backend.

`boosting_sparse._renew_leaf_values_sparse` already applies
`finish_leaf_output` with the same ordering, so the three renewing objectives
do not escape the cap. `ForcedSplits` and the coupled CEGB cost are refused
globally in `ExtraTreeParams.check_scalars`, so no sparse-specific handling
applies.

## 5. Exact cross-lane patch requests

### 5.1 Task 01 — `src/mojoboost/__init__.mojo` (and `train_gpu.mojo`)

The export block in `handoffs/performance_16_sparse_categorical_gpu.md`
section 4.1 is now **stale**: three names moved to `sparse.mojo` and several
are new. Use this instead, added next to the existing sparse block:

```mojo
from .sparse import (
    absent_is_unknown,
    check_sparse_categorical_semantics,
    default_category_bin,
)
from .gpu_sparse_layout import (
    MeasuredCosts,
    SparseDeviceLayout,
    SparseGpuCapability,
    SparseRangeTable,
    SparseVerdict,
    categorical_support,
    check_bundle_compatibility,
    check_sparse_gpu_histograms,
    check_sparse_gpu_training,
    decide_sparse,
    sparse_gpu_capability,
    sparse_gpu_training_is_wired,
    sparse_support,
)
from .gpu_sparse import (
    GpuSparseHistogramBuilder,
    build_histogram_gpu_sparse,
    check_entry_row_consistency,
    partition_entries_host,
    side_mask_host,
)
from .gpu_categorical import (
    CatSetPool,
    CategoryStats,
    GpuCategoryStats,
    apply_categorical_split_pooled,
    cat_bitset_from_codes,
    category_stats_host,
    check_cat_bitset,
    codes_from_cat_bitset,
)
```

Three more names go into the **existing** sparse blocks at
`src/mojoboost/__init__.mojo:471` and `:476` — the blocks are already there,
only the names are new. Without them a caller cannot fit a sparse bundling
plan through the public surface at all (section 3.8):

```mojo
from .tree_sparse import (
    SparseBundling,          # <- new
    SparseTreeResult,
    grow_tree_sparse,
    predict_row_sparse,
)
from .boosting_sparse import (
    SparseBundledData,       # <- new
    prepare_bundling_csc,    # <- new
    train_multiclass_sparse,
    train_sparse,
    train_sparse_with_valid,
)
```

`tests/parallel/api_snapshot_manifest.json` needs the same names in the same
change or the snapshot test fails. That file is outside this lane.

Also for Task 01: **do not** add a sparse entry point to `train_gpu.mojo` in
this round. If one is added later, it must call
`gpu_sparse_layout.check_sparse_gpu_training(caps, data)` first and must flip
`sparse_gpu_training_is_wired()` in the same change; that function exists so
the claim and the implementation cannot diverge. `train_gpu.mojo` needs no
edit for this lane today.

### 5.2 Task 05 — `src/mojoboost/device_policy.mojo` (not in my required list, but the message is now false)

`BLOCK_SPARSE_INPUT`'s message asserts a fact that no longer holds. Keep the
block, replace the text:

```mojo
        blocks.add(
            BLOCK_SPARSE_INPUT,
            String(
                "sparse input trains on the CPU; the sparse GPU primitives"
                " exist but no training path is wired to them"
            ),
        )
```

`docs/DEVICE_SELECTION.md:73` carries the same claim and cites
`python/mojoboost/__init__.py::_fit_sparse` as the enforcer; `device_policy`
is the authority and the Python path is downstream of it.

### 5.3 Task 06 — GPU prediction bindings

There is no sparse GPU prediction kernel and none is planned in this lane.
Requests:

1. Any binding entry point that takes a sparse matrix **and** an explicit
   GPU device must raise rather than fall back. Route it through the record:

```mojo
from .gpu_sparse_layout import check_sparse_gpu_training
# ... explicit-GPU sparse request:
_ = check_sparse_gpu_training(caps, binned_sparse)   # always raises today
```

   If you would rather not construct a `SparseBinnedMatrix` at the boundary,
   raise with the same wording so the two agree:
   `"sparse input trains and predicts on the CPU; the sparse GPU primitives
   exist but no path is wired to them"`.
2. CPU sparse prediction is `model_sparse.predict_csr` /
   `predict_proba_csr` / `predict_class_csr` / `predict_raw_csr`; those take
   a `CsrMatrix` and are unchanged by this lane.

### 5.4 Task 07 — Python public API

1. `_fit_sparse` (or its successor) should pass through the two arguments
   that now exist natively: `class_bagging` for binary/regression sparse
   fits (`train_sparse`, `fit_csc`) and `goss` for sparse multiclass
   (`train_multiclass_sparse`, `fit_multiclass_csc`). Before this round the
   sparse path silently ignored both.
2. Sparse + `device="gpu"` must raise, not warn-and-fall-back. The policy
   block already exists; keep it, and use the corrected message from 5.2.
3. Optional but recommended for the categorical trap: when a sparse fit
   declares categorical features, `sparse.check_sparse_categorical_semantics`
   returns the feature ids whose absent rows route to the unknown bin. That
   is a modelling fact worth surfacing (a warning or a fitted attribute), not
   an error. It is also already enforced structurally by `validate()`, so
   Python does not need to re-check anything.
4. Sparse multiclass models now record monotone constraints; if any Python
   attribute reads `booster.monotone` for multiclass, it will start returning
   the real signs.

### 5.5 Task 09 — EFB and the dense algorithm lane

1. **EFB is now wired to sparse growth** (section 3.8), through
   `boosting_sparse.prepare_bundling_csc` and `tree_sparse.SparseBundling`.
   This changes two things you own. **Neither is a code request against
   `efb.mojo`'s algorithm** — the plan, the encoding, and
   `expand_bundled_histogram` are used exactly as written.

   1a. **`efb.check_bundling_honored`'s wording is now wrong.** It had no
   caller anywhere in the repository; `boosting_sparse._resolve_bundling` is
   now its first one. Its docstring lists "the sparse, GPU, distributed,
   ranking, custom-objective, and custom-metric trainers" as paths that do
   not apply bundling, and its message says the named trainer "builds its
   histograms another way and would ignore it". The sparse trainers now do
   apply it — they just need a plan fitted first, which is a different
   failure and deserves a different sentence. Two small edits, both yours:

   - Drop "sparse" from the docstring's list of trainers that do not honour
     `BoosterParams.bundling`, and say instead that the sparse trainers
     honour it via a fitted plan passed in, refusing the switch alone.
   - Give the message a second half for that case. The function only has
     `trainer: String` to go on, so the simplest correct change is to append
     one clause to the existing text rather than branch:

```mojo
        " If this trainer takes a fitted plan (the sparse trainers do),"
        " fit one with boosting_sparse.prepare_bundling_csc and pass it in"
        " instead of setting enable_bundle alone"
```

   I did not edit `efb.mojo`, so the misleading text is live today. Until it
   changes, a sparse caller who sets `enable_bundle` without calling
   `prepare_bundling_csc` gets a correct refusal with a wrong explanation.

   1b. **`expand_bundled_histogram`'s contract is now load-bearing for two
   growers, not one.** `tree_sparse._node_histogram` depends on exactly the
   three properties its docstring already states: the output is laid out
   `[feature * out_n_bins + b]` in original feature space; a non-empty
   `features` rewrites only those slices and leaves the rest untouched (the
   sparse caller zeroes its buffer per node, so unselected features are
   zero); and the expansion is linear, which is what lets the sparse grower
   keep histogram subtraction. If any of those three changes, tell me — the
   sparse path breaks silently, not loudly. Same request for
   `FeatureBundling.n_bins` meaning "the bin count of the matrix the plan was
   fitted on"; `SparseBundling.source_bins` reads it to size the expanded
   histogram and to check a validation matrix.
2. `check_bundle_compatibility(caps, plan, source_cats)` in
   `gpu_sparse_layout.mojo` is the device-side EFB gate and already refuses
   a categorical feature inside a multi-member bundle
   (`SPARSE_BUNDLED_CATEGORICAL`) and reports `recovery_is_exact`. It needs
   no change from you.
3. **Please keep these `boosting.mojo` names importable**, since
   `boosting_sparse.mojo` now imports them:
   `_check_class_bagging`, `_fill_softmax_grad_hess`,
   `_multiclass_goss_select`, alongside the ones it already imported. If you
   rename or inline any of them, say so and I will follow.
4. If `ExtraTreeParams` is folded into `TreeParams`, `tree_sparse.mojo` reads
   `params.extra.max_delta_step`, `params.extra.path_smooth`, and calls
   `params.extra.check(...)`; `boosting_sparse.mojo` passes
   `params.tree.extra` to `_renew_leaf_values_sparse`. Keep an accessor or
   give me the new field names.
5. `sampling.ClassBaggingParams`, `has_positive_rows`, and
   `refresh_class_bag` are now imported by `boosting_sparse.mojo` and
   `model_sparse.mojo`. Same request: tell me before renaming.

### 5.6 Task 14 — bindings

Narrow adapter for the capability record, no algorithm duplicated at the
boundary. Suggested signature for a `dataset_bindings.mojo` or a sparse
binding module:

```mojo
# Returns (support_code, categorical_code, histograms, training,
#          unknown_absent_feature_ids). Never raises for an unsupported
# shape -- the codes are the answer; it raises only for a malformed matrix.
def sparse_gpu_capability_record(
    data: SparseBinnedMatrix, caps: DeviceCaps
) raises -> PythonObject
```

backed by `gpu_sparse_layout.sparse_gpu_capability`, with
`gpu_sparse_layout.sparse_support_name(code)` for the human-readable reason.
Do not reimplement the codes on the Python side; they are stable and
appended to, never renumbered.

Two more that are ready to expose and need no new native work:
`sparse.check_sparse_categorical_semantics(data) -> List[Int]` (the
unknown-absent feature ids) and
`gpu_categorical.codes_from_cat_bitset(cats, feature, bitset) -> List[Int]`
(a split's raw category codes, which is the form a model dump wants; bin ids
are an artifact of the fitted table).

Do not expose `CatSetPool`, `GpuCategoryStats`, `DeviceBuffer`, or any
device pointer to Python.

## 6. Duplicates fused or quarantined

| Duplicate | Disposition |
| --- | --- |
| Categorical semantics stated only in `gpu_categorical` | **Fused** into `sparse.mojo`; the GPU module imports from there |
| `GpuSparseHistogramBuilder` inline structural checks | **Fused** into `SparseBinnedMatrix.validate()` |
| Raising-only categorical support check | **Fused**: `categorical_support` is the primitive, `check_categorical_support` wraps it |
| Open-coded softmax grad/hess in `train_multiclass_sparse` | **Fused** into `boosting._fill_softmax_grad_hess` |
| Row/entry partition + range bookkeeping, about to be copied for the pooled categorical arm | **Prevented**: factored into `finish_split`, both arms call it |
| Node-total reduction, needed by both histogram and category stats | **Fused** into `enqueue_node_totals` |
| `MeasuredCosts` / `decide_sparse` cost model | **Quarantined as-is.** Still returns `SPARSE_UNDECIDED` with no measurements, still has no caller, and deliberately keeps no default thresholds. Nothing in this round consumes it, and nothing should until the benchmark in `docs/GPU_SPARSE_CATEGORICAL_DESIGN.md` has run. |
| `SparseRangeTable.bound_split` / `defer_ranges` | **Preserved, off by default.** The exact-midpoint path is the established one. |
| A second sparse grower for bundled matrices | **Prevented.** `SparseBundling.none()` is the identity on both mappings, so `grow_tree_sparse` has one code path, not a bundled one and an unbundled one. |
| Per-node member recovery (`efb.unbundle_histogram`) vs. whole-histogram expansion | **Fused onto the dense answer.** The sparse grower calls `efb.expand_bundled_histogram`, the same function `tree._expand_bundled` calls, rather than recovering member by member inside the split search. No new EFB code was written. |
| A sparse copy of `efb.prepare_bundling` | **Prevented.** `prepare_bundling_csc` is the sparse sibling of the dense entry point and reuses `fit_bundles` / `bundle_csc` verbatim; it differs only in returning one matrix instead of two. |

## 7. Fallbacks preserved

- The CPU sparse trainer is untouched as the only sparse training path. No
  strategy switch, no automatic device selection, no density heuristic was
  added anywhere.
- `defer_ranges` stays `False` by default; a split still downloads exact
  per-feature midpoints unless the caller opts out.
- `STRATEGY_ATOMIC` remains the only sparse accumulation strategy.
- `apply_split`'s four-argument categorical form is unchanged and remains the
  default; the pooled form is additive.
- `sparse_gpu_training_is_wired()` returning `False` is the conservative
  gate: everything that could claim sparse GPU training reads it.
- **Sparse EFB is off unless asked for.** `params.bundling.enabled` defaults
  to `False`, so `prepare_bundling_csc` returns the matrix untouched and
  every trainer resolves an inactive view. The unbundled path is unchanged
  and remains the default. `fit_bundles` is a second gate: a plan whose
  histogram-slot cost does not beat the unbundled one sets
  `use_bundling = False` and `prepare_bundling_csc` returns the matrix
  untouched, so the fallback lives inside the plan rather than beside it.
  Only lossless plans are reachable (`EfbSettings.check`).

## 8. Remaining disconnections

1. **No sparse GPU trainer.** The primitives are driveable
   (`upload_gradients` / `begin_tree` / `build_leaf` / `apply_split`), and
   `gpu_split_search.mojo` consumes the resulting `Histogram` unchanged, but
   nothing drives them. Device-side leaf values would need
   `gpu_objectives_native.update_raw_ranges`, which works off leaf **row**
   ranges that the sparse path also maintains — that should carry over, and
   it is unverified.
2. **No sparse GPU prediction.** `gpu_predict.mojo` is dense-only.
3. **Sparse EFB is unexported and unverified.** It is wired (section 3.8),
   but `SparseBundling`, `SparseBundledData`, and `prepare_bundling_csc` are
   not in `src/mojoboost/__init__.mojo` (Task 01 owns it; block in 5.1) and
   nothing reaches it from Python. Nothing has compiled or run, and the
   bundled-equals-unbundled claim is a reading, not a result (section 11).
   The **GPU** sparse path still has no bundling support beyond the existing
   `check_bundle_compatibility` gate, and `SparseGpuCapability` does not yet
   carry a bundling verdict — an explicit-GPU request for a bundled sparse
   matrix is refused today only because sparse GPU training as a whole is.
4. **`decide_sparse` has no measurements**, so no policy may consume it.
5. **Custom objectives** remain unavailable on the sparse path
   (`objective.mojo` takes a `BinnedMatrix`). Unchanged by this round.
6. **The GPU modules are still not exported** from `src/mojoboost/__init__.mojo`
   (Task 01 owns it; block supplied in 5.1).
7. **No tiled sparse accumulation strategy**; a measurement away, not a
   design change.

## 9. Serialization and public API effects

- **Serialization format: unchanged.** No new fields, no new records.
- **Serialized content: one change.** Sparse multiclass fits with monotone
  constraints now record those constraints on the `MulticlassBooster`
  (section 3.4). This makes sparse multiclass match dense multiclass; a model
  saved before this change and loaded after it is unaffected.
- **Native API: additive.** New public names are
  `SparseBinnedMatrix.validate`, `sparse.default_category_bin`,
  `sparse.absent_is_unknown`, `sparse.check_sparse_categorical_semantics`,
  `gpu_sparse_layout.{categorical_support, SPARSE_CATEGORY_SET_OVERFLOW,
  sparse_gpu_training_is_wired, SparseGpuCapability, sparse_gpu_capability,
  check_sparse_gpu_histograms, check_sparse_gpu_training}`,
  `GpuSparseHistogramBuilder.{capability, check_split_ids,
  enqueue_default_side, finish_split, enqueue_node_totals}`,
  `gpu_categorical.{apply_categorical_split_pooled, GpuCategoryStats}`.
- **Serialization is unchanged by EFB, deliberately.** The plan is
  training-time scaffolding: `fit_csc` / `fit_multiclass_csc` drop it before
  returning, the trees name original features and original bins, and a model
  fitted with `enable_bundle` on is the same structure as one fitted with it
  off. No model carries a `FeatureBundling`, and `save_model` /
  `load_model` / inspection / importance need no knowledge of bundling.
- **Native API: additive (EFB).** `tree_sparse.SparseBundling`,
  `boosting_sparse.{SparseBundledData, prepare_bundling_csc}`, and
  `tree_sparse._node_histogram` (private).
- **Signature changes, all trailing optional arguments with defaults**, so
  existing positional calls keep working: `train_sparse(+class_bagging,
  +bundling)`, `train_sparse_with_valid(+class_bagging, +bundling)`,
  `train_multiclass_sparse(+goss, +bundling)`, `grow_tree_sparse(+bundling)`,
  `predict_row_sparse(+bundling)`, `predict_row_sparse_csc(+bundling)`,
  `fit_csc(+init_score, +class_bagging)`, `fit_multiclass_csc(+goss)`. The
  four external sparse call sites I found (`trainset.mojo`,
  `external_memory.mojo`) therefore need no edit.
- **One behaviour change on an existing signature.** All three sparse
  trainers now call `efb.check_bundling_honored`, so a caller who sets
  `params.bundling.enabled` and passes an unbundled matrix gets an error
  where they previously got a silent drop. Nothing in the repository does
  this today (no sparse call site sets `params.bundling`), but a downstream
  caller that did would newly raise. This is intentional: silently ignoring
  `enable_bundle` is what made the bundle-id trap reachable.
- **One check tightened.** `train_sparse_with_valid` now compares the
  validation matrix's `n_bins` against `bundles.source_bins(data)` (the
  original binning width) rather than the training matrix's `n_bins`. With
  bundling off these are the same number, so unbundled callers see no
  change.
- **Moved names:** `default_category_bin`, `absent_is_unknown`,
  `check_sparse_categorical_semantics` are no longer defined in
  `gpu_categorical.mojo`. Nothing in the repository imported them.
- **Error-message change:** the categorical support failures now read
  `"sparse GPU path does not support this dataset: <reason>"` rather than the
  bare reason, so they match the shape failures. No test asserts on them.

## 10. Risks

1. **Nothing was compiled.** Every edit is static. The likeliest failure mode
   is a Mojo signature or borrow detail: `GpuCategoryStats.compute` takes
   `mut builder` and reads `builder.ctx` while taking pointers to other
   builder fields (the same pattern the builder's own methods use, but from
   outside the struct), and `apply_categorical_split_pooled` does the same.
2. **`validate()` is newly strict on the CPU path.** It now rejects a
   `SparseBinnedMatrix` with non-ascending or duplicated row indices within a
   column. That has always been the documented canonical form and both
   producers (`transform_csc`, `efb.bundle_csc`) satisfy it by construction,
   but any hand-built matrix in a test that did not would now fail at
   training time rather than silently mis-binary-search.
3. **`validate()` costs O(nnz) per fit.** Negligible against training, but it
   is new work on a path that previously did none.
4. **The multiclass gradient fuse** assumes `_fill_softmax_grad_hess` is
   arithmetically identical to the loop it replaced. Static reading says yes;
   it is the first thing to check (section 11).
5. **A concurrent lane committed this lane's in-progress edits.** Commit
   `860b1cf` ("Integrate training and interoperability subsystems", 12:43)
   swept `sparse.mojo`, `boosting_sparse.mojo`, `model_sparse.mojo`,
   `gpu_sparse.mojo`, and `gpu_sparse_layout.mojo` into a commit this lane
   did not make and did not ask for. The file contents are intact and
   correct; only the "do not commit" instruction was violated, and not by
   this session. `gpu_categorical.mojo` and this handoff remain uncommitted.
6. **`sparse_gpu_training_is_wired()` is a discipline, not an enforcement.** A
   future trainer could ignore it. The handoff is the only thing binding it.
7. **Sparse EFB is the largest unverified change in this lane, by a wide
   margin.** Three specific places it could be wrong, in the order I would
   check them:
   - *Routing versus accumulation could disagree.* Split application decodes
     a stored bin through `SparseBundling.local_bin`; the histogram the split
     was chosen from folded foreign bins into the member's default via
     `efb._recover_member_into`. These are two independent statements of the
     same rule, written in two files, and only a bundled-equals-unbundled
     test will show they agree. Nothing enforces it structurally.
   - *`source_bins` sizes the expanded histogram.* If `plan.n_bins` ever
     meant anything other than the source matrix's `n_bins`,
     `expand_bundled_histogram` would raise "member needs more bins than the
     output histogram has per feature" — loudly, at least, not silently.
   - *`_node_histogram` allocates a fresh `Histogram` per node* under
     bundling, where the dense path reuses a scratch buffer. That is an
     allocation cost, not a correctness one, and the sparse accumulator
     already returned a fresh histogram per node before this change.
8. **`missing_bin` under bundling is read from `plan.slot_missing`, not from
   the bundled matrix.** `SparseBundling.of` rebuilds the original-space
   table, and `bundle_csc` keeps a column-level `missing_bin` only when its
   bundle has exactly one, writing `EFB_NONE` otherwise. The grower reads the
   original-space table and never the column's, so the two are allowed to
   differ — but that also means `plan.slot_missing` is the only statement of
   a bundled feature's missing bin, and this lane relies on `efb.mojo`'s
   accounting for it rather than re-deriving it.

## 11. Smallest later focused checks — ALL UNRUN

Run one at a time; never the full suite.

```
# 1. Sparse multiclass must be unchanged by the gradient fuse and the
#    monotone recording (the highest-value check in this list).
pixi run mojo run tests/test_sparse.mojo          # UNRUN

# 2. Categorical on the sparse path, including the absent-is-unknown case.
pixi run mojo run tests/test_categorical.mojo     # UNRUN

# 3. EFB still round-trips: it builds SparseBinnedMatrix values by hand and
#    is the most likely place `validate()`'s strictness would bite.
pixi run mojo run tests/parallel/test_efb.mojo    # UNRUN

# 4. Monotone constraints, which sparse multiclass now records.
pixi run mojo run tests/test_monotone.mojo        # UNRUN
```

Check 3 is now the second-highest value in the list, not the third: it is the
only existing test that exercises `efb.bundle_csc` output, and section 3.8
put a grower behind it. Run 1 and 3 first.

New tests worth writing later, in the order they buy the most:

0. **Bundled equals unbundled on the sparse path.** Fit twice on the same
   CSC, once with `enable_bundle` off and once on, and require the two
   `Model`s to be tree-for-tree identical — same `feature`, same `threshold`,
   same `left`/`right`, same `value` to floating-point tolerance. Only
   lossless plans are reachable, so equality is the contract, not an
   approximation. This is the single check that would catch every failure
   mode in risk 7, and it needs a dataset EFB will actually bundle (sparse,
   mutually exclusive columns) or it silently tests nothing — assert
   `prepare_bundling_csc(...).active()` first.
1. `SparseBundling.local_bin` against `efb.expand_bundled_histogram`
   directly: for every feature and every column bin, the bin routing sends a
   row to must be the bin expansion credits that row to. This is risk 7's
   first bullet stated as a unit test, without a trainer in the way.
2. `train_sparse` raises when `params.bundling.enabled` is set and no plan
   was fitted (the `check_bundling_honored` path), and does **not** raise
   when `prepare_bundling_csc` was used.
3. A bundled sparse fit round-trips through `save_model` / `load_model` and
   predicts identically on dense rows, confirming no plan leaked into the
   model.
4. Categorical plus bundling on one matrix: a categorical feature must stay a
   singleton bundle, and `SparseBundling.of` must rebuild its `CategoricalSpec`
   entry verbatim.
5. `train_sparse_with_valid` accepts a validation matrix binned by the same
   mapper while training on a bundled matrix (the `source_bins` check), and
   rejects one binned to a different width.
6. `SparseBinnedMatrix.validate()` rejects each malformation it names, and
   accepts every matrix `transform_csc` and `efb.bundle_csc` produce.
2. `train_multiclass_sparse` with and without `goss` matches
   `train_multiclass` on the densified matrix, tree for tree.
3. `train_sparse` with `class_bagging` matches `train` on the densified
   matrix.
4. A sparse multiclass fit with monotone constraints round-trips through
   `save_model` / `load_model` with its constraints intact.
5. `check_sparse_gpu_training` raises for a perfectly supported sparse
   dataset, with the "no training path is wired" wording. Host-only, no GPU
   needed.
6. `sparse_gpu_capability` on a shape past each limit returns the right code
   and `histograms == False`. Host-only, `DeviceCaps` can be injected.
7. Device-required: `apply_categorical_split_pooled` and the four-argument
   `apply_split` produce identical row ranges and entry windows for the same
   categorical set.
8. Device-required: `GpuCategoryStats.compute` matches `category_stats_host`
   for the same node and feature.
