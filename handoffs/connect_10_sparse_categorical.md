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
       -> SparseBinnedMatrix.validate()          <- new, once per fit
       -> train_sparse | train_sparse_with_valid | train_multiclass_sparse
       -> grow_tree_sparse -> build_histogram_sparse_node -> Model

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

1. **Do not enable EFB on the sparse training path.** `efb.bundle_csc`
   already produces a `SparseBinnedMatrix` that `train_sparse` would accept
   as-is, and that is exactly the trap: `grow_tree_sparse` would then split
   on *bundle* columns, so the fitted `Tree.feature` ids would be bundle ids
   and every downstream consumer (importance, inspection, model dump,
   prediction against an unbundled matrix) would silently read them as
   original feature ids. Sparse EFB needs `efb.unbundle_histogram` applied
   per member per node *before* split search, plus split recovery to the
   original feature identity. That is split-search and grower work, not
   representation work. If you want it, tell me the contract and I will wire
   `tree_sparse` in a later round.
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

## 8. Remaining disconnections

1. **No sparse GPU trainer.** The primitives are driveable
   (`upload_gradients` / `begin_tree` / `build_leaf` / `apply_split`), and
   `gpu_split_search.mojo` consumes the resulting `Histogram` unchanged, but
   nothing drives them. Device-side leaf values would need
   `gpu_objectives_native.update_raw_ranges`, which works off leaf **row**
   ranges that the sparse path also maintains — that should carry over, and
   it is unverified.
2. **No sparse GPU prediction.** `gpu_predict.mojo` is dense-only.
3. **EFB is not wired to sparse growth** and must not be until split recovery
   to original feature identity exists (5.5.1).
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
- **Signature changes, all trailing optional arguments with defaults**, so
  existing positional calls keep working: `train_sparse(+class_bagging)`,
  `train_sparse_with_valid(+class_bagging)`,
  `train_multiclass_sparse(+goss)`, `fit_csc(+init_score, +class_bagging)`,
  `fit_multiclass_csc(+goss)`.
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

New tests worth writing later, in the order they buy the most:

1. `SparseBinnedMatrix.validate()` rejects each malformation it names, and
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
