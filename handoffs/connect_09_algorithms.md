# Connect 09: EFB and extra tree parameters wired into real tree growth

Scope: `src/mojoboost/efb.mojo`, `tree_parameters_extra.mojo`, `sampling.mojo`,
`params.mojo`, `binning.mojo`, `tree.mojo`, `split.mojo`, `categorical.mojo`,
`boosting.mojo`, and this file. Nothing outside that set was edited.

No tests, builds, benchmarks, or tools were run. Every claim below is from
static reading of the tree. Nothing here is a correctness, parity, or
performance claim.

## 1. Implementations found

**Exclusive feature bundling.** One implementation, `efb.mojo`, complete for
the sparse CSC path: conflict measurement (`pairwise_conflict`,
`conflict_count`), a greedy bundler over sparse descriptors (`fit_bundles`),
offset bin encoding with a shared bin 0, a `use_bundling` verdict, a
`bundle_csc` encoder, and `unbundle_histogram` for split recovery. No second
bundler exists anywhere in the repo. It was reachable from no trainer:
`params._validate` refused `enable_bundle=true` outright, and no grower took a
plan. `handoffs/task13_efb.md` recorded the refusal and gave the reason as a
belief that a bundle plan has to travel with the model (v4 serialization, a
`Model` field, an `inspection.py` reader). That belief is wrong, and section 6
says why.

**Extra tree parameters.** One implementation, `tree_parameters_extra.mojo`,
already largely connected by the lane behind `handoffs/task12_tree_parameters.md`
and `handoffs/integration_08_algorithms.md`. Already live in the dense CPU
path before this lane: `min_gain_to_split`, `min_sum_hessian_in_leaf`,
`monotone_penalty`, `feature_contri`, cegb split cost (all inside
`split._feature_gain`), and `extra_trees` / `extra_seed` / `max_delta_step` /
`path_smooth` behind `grower_applies_extra`. Row bagging variants,
per-tree/per-level/per-node feature sampling, and the counter-based
splitmix64 seeds were already connected through `sampling.mojo`. Categorical
thresholds and regularization were already fused into
`tree_parameters_extra` helpers. What remained disconnected was
`forcedsplits_filename`: `ForcedSplits` held raw `Float64` thresholds, the
grower has no bin edges, and `check_scalars` refused any non-empty forced
tree by name.

No duplicate registry, alternate policy engine, alternate trainer, or parallel
model representation was created. No new module was added.

## 2. Call path, before

```
params.parse -> _validate -> refuse enable_bundle by name
                          -> refuse non-empty forced splits by name
boosting._boost_rounds -> tree.grow_tree(data, grad, hess, params.tree, bag, i)
                       -> _HistPool over data.n_features x data.n_bins
                       -> build_histogram_into(hist, data, ...)
                       -> split.find_best_split(hist, ...)
efb.fit_bundles / bundle_csc / unbundle_histogram   [unreachable]
tree_parameters_extra.ForcedSplits                  [unreachable]
```

## 3. Call path, after

```
params.parse -> _validate -> efb.check_bundling_supported(enabled, device==CPU)
                          -> config.booster.bundling.check()
                          -> enable_bundle      -> booster.bundling.enabled
                          -> max_conflict_rate  -> booster.bundling.params

boosting._boost_rounds / train_with_valid
  / _boost_rounds_multiclass / train_multiclass_with_valid
    -> var bundling = efb.prepare_bundling(data, params.bundling)   [once per call]
    -> tree.grow_tree(data, grad, hess, params.tree, bag, i, bundling)

tree.grow_tree
    if bundling.active:
        tree_columns = efb.columns_for_features(bundling.plan, tree_features)
        _hist_full/_hist_subset -> build_histogram_into(scratch, bundling.data,
                                                        grad, hess, tree_columns)
                                -> efb.expand_bundled_histogram(...)  [per-feature shape]
    else:
        _hist_full/_hist_subset -> build_histogram_into(hist, data, grad, hess,
                                                        tree_features)
    -> split.find_best_split(hist, ...)          [identical either way]
    -> row partitioning reads the ORIGINAL matrix, never the bundle

binning.map_forced_splits(mapper, forced) -> ForcedSplits with .bins
tree.grow_tree -> forced node selected ahead of any gain-chosen leaf,
                  split applied through the same loop body
```

`prepare_bundling` is the single fallback point. It returns
`BundledMatrix.none()` (`active = False`, grow on the original matrix) when
bundling is disabled, when the matrix is degenerate, or when the greedy's own
`use_bundling` verdict is negative. Default is disabled
(`DEFAULT_ENABLE_BUNDLE = False`), so default-model behavior is unchanged.

## 4. Connections completed

### EFB, dense CPU, end to end

- `EfbSettings` (enabled flag + `EfbParams`) added to `efb.mojo` and carried on
  `BoosterParams.bundling`. `BoosterParams` moved off `@fieldwise_init` to an
  explicit `__init__` with `bundling` last and defaulted, so every existing
  positional caller (`bindings/_mojoboost.mojo:313`, the CLI, the C API) is
  unaffected.
- `params.mojo` now stores `enable_bundle` and `max_conflict_rate` instead of
  refusing them. `check_bundling_supported` accepts `false` always, accepts
  `true` on CPU, and refuses by name off-CPU. `check_bundling_params` still
  refuses `max_conflict_rate > 0.0`: lossy bundling is withheld pending a
  benchmark, so the knob is range-checked but only its exact-bundling value is
  accepted. No inert parameter is exposed.
- Dense conflict measurement: `dense_bin_counts` (per-column bin width off the
  training matrix), `dense_default_bins` (most frequent bin per column, ties to
  the smaller id), `nondefault_rows_dense`, `count_nondefault_dense`. A dense
  matrix has no notion of absence, so the "default bin" is what a sparse
  column's implicit zero would have been; on the data EFB exists for it is the
  bin containing 0.0.
- `_fit_bundles_core` is now the single greedy. `fit_bundles` (sparse) and
  `fit_bundles_dense` both build descriptors and delegate. The core takes an
  explicit `counts` list rather than reading `len(rows[f])`, so a dense caller
  can skip materializing ineligible columns.
- `bundle_dense` produces the encoded `BinnedMatrix`; singletons keep identity
  encoding, bundle members are offset onto a shared bin 0.
- `BundledMatrix` (plan + encoded matrix + `active`) is the one value that
  crosses into the grower. `prepare_bundling` is the only constructor callers
  use.
- Histogram use: `expand_bundled_histogram` turns a per-bundle histogram back
  into a per-feature one before any search. It is linear in the accumulated
  counts, so the sibling-subtraction trick stays valid on the expanded shape.
  Cost is O(#features x #bins) per node.
- Split recovery to original feature identity is structural, not a fix-up:
  because expansion happens before `find_best_split`, the split that comes back
  already names the original feature and the original local bin. `split.mojo`
  needed no functional change at all (see section 5).
- Missing/zero semantics on the dense path: `fit_bundles_dense` sets
  `slot_missing = EFB_NONE` and ignores `bundle_missing`, because missing values
  are routed by the original matrix during partitioning, never by the bundle.
- Feature names and per-tree/per-level feature sampling: `columns_for_features`
  maps sampled original features to the bundle columns that must be scanned. A
  bundle is scanned if any member is sampled; unsampled members ride along in
  the same column and are simply never selected, since expansion emits a slice
  per original feature and only sampled features are searched.
- Unbundled fallback: three paths in `prepare_bundling`, plus `bundling.active`
  guards at every use site in `grow_tree`. With bundling off, `grow_tree` is
  byte-for-byte the previous code path.
- `check_bundling_honored(settings, trainer)` is a one-line refusal that any
  trainer which does not apply bundling can call, so an ignored `enable_bundle`
  is reported rather than silently dropped. It is not yet called anywhere; the
  call sites are outside this lane's ownership and are listed in section 8.

### Forced splits

- `ForcedSplits` gained `var bins: List[Int]`, `is_mapped()`, and
  `bin_at(i)`. Its `__init__` is explicit with `bins` defaulted, so
  one-argument callers still compile.
- `binning.map_forced_splits(mapper, forced)` maps raw thresholds to bin ids
  through `mapper.bin_value(f, v)`, refusing categorical features and NaN
  thresholds by name. This is the only place bin edges and forced splits meet.
- `check_scalars` now refuses only *unmapped* non-empty forced trees, and names
  `binning.map_forced_splits` as the fix.
- `tree._LeafState` gained `var forced: Int` (-1 when nothing is owed).
  `grow_tree` selects the lowest pending forced node ahead of any gain-chosen
  leaf, applies it through the existing loop body as
  `SplitInfo(fn.feature, params.extra.forced.bin_at(node), 0.0, True, False)`,
  and propagates `fn.left` / `fn.right` into the children. A forced split that
  leaves a child empty raises, naming the node: a gain-chosen split can never do
  this, but a forced one can, because it is applied whatever the data says.
  Leaf-wise growth resumes from the frontier the forced tree left behind.

### Parameters confirmed reaching the dense path

`min_gain_to_split`, `min_sum_hessian_in_leaf`, `max_delta_step`,
`path_smooth`, `extra_trees` / `extra_seed`, `feature_contri`,
`cegb_tradeoff` / `cegb_penalty_split`, `monotone_penalty`,
`monotone_constraints_method`, categorical thresholds and regularization,
forced splits, row bagging variants, per-tree / per-level / per-node feature
sampling, deterministic counter-based seeds.

## 5. Duplicates fused or quarantined

- **Two copies of the bundle recovery arithmetic.** `unbundle_histogram_into`
  and `unbundle_histogram` briefly held the same loop. Factored into
  `_recover_member_into`; both public forms now delegate. `expand_bundled_histogram`
  uses it too, so there is exactly one copy of the offset arithmetic.
- **Two greedy bundlers.** The dense entry point initially had its own loop.
  Fused into `_fit_bundles_core`; sparse and dense differ only in how they
  build descriptors.
- **Inline bag range check in `grow_tree`.** Replaced by
  `sampling.check_row_set(bag, data.n_rows)`, which is the one place that
  property is enforced. This strengthens the check from in-range to strictly
  ascending.
- **Nothing quarantined.** No dead duplicate remains in the owned files.

### A design that was tried and reverted

The first EFB integration searched the bundled histogram in place, rebinding
`grad_p` / `hess_p` / `count_p` inside `find_best_split` between
`hist.grad.unsafe_ptr()` (an immutable argument origin) and a local buffer's
pointer (a mutable local origin). `histogram.build_histogram_into` documents
that a pointer taken from a struct field carries that field's origin, so that
rebinding is at best origin-fragile, and builds are forbidden here so I could
not check it. It was replaced by expansion in `grow_tree`, which needs zero
functional change to `split.mojo`, is exact rather than approximate, and keeps
sibling subtraction valid by linearity. All `split.mojo` edits from the first
design were reverted with targeted edits (not `git checkout`, to avoid touching
concurrent work). `split.mojo`'s only surviving change is one docstring
paragraph in `find_best_split` recording that a bundled histogram is expanded
before the search runs, so nothing in that function ever sees a bundle.

## 6. Serialization and public API effects

**No serialization change, and no model-format version bump, is needed for any
of this work.** A bundle is a training-time histogram layout only:

- splits recorded on the tree carry the original feature id and the original
  local bin, because expansion happens before the search;
- row partitioning reads the original matrix, not the bundle;
- prediction never sees a bundle, since the plan is dropped when `grow_tree`
  returns.

So `Model`, `serialize.mojo` (no v4), `python/mojoboost/inspection.py`,
`importance.mojo`, and `contrib.mojo` are all unaffected. This retires the
blocker recorded in `handoffs/task13_efb.md`.

Forced splits likewise produce an ordinary tree; nothing new is serialized.

Public API effect: `enable_bundle` and `max_conflict_rate` are now accepted by
the parameter parser on CPU instead of refused. `BoosterParams` has a new
trailing defaulted field. The new EFB and forced-split symbols are not yet
exported from `__init__.mojo` (section 8).

## 7. Remaining disconnections

- `enable_bundle` is applied only by the four dense CPU trainers in
  `boosting.mojo`. Every other trainer that calls `grow_tree` or a sibling
  grower ignores it silently: `objective.mojo`, `ranking.mojo`,
  `ranking_advanced.mojo`, `custom_metric.mojo`, `alternate_boosting.mojo`,
  `boosting_rf.mojo`, plus `boosting_sparse.mojo` (`grow_tree_sparse`),
  `train_gpu.mojo` (`grow_tree_gpu`), and `distributed.mojo`
  (`_grow_tree_distributed`). `check_bundling_honored` exists for exactly this
  and is not yet called.
- `max_conflict_rate > 0` (lossy bundling) is still refused. The greedy
  supports it; it is withheld pending a benchmark.
- Automatic EFB is not enabled and must not be until benchmarks exist.
- `cegb_penalty_feature_coupled` and `cegb_penalty_feature_lazy` remain refused
  by name. They need a per-model feature-use ledger threaded through six
  trainers, which is a larger change than this lane owns.
- `extra_trees` for categorical features remains refused by name.
- `feature_pre_filter` has no implementation anywhere in the repo. It is now
  refused by name in `check_extra_option_supported`, with the message
  explaining that mojoboost's scan already produces the
  `feature_pre_filter=false` model and only the speed-up is missing. I did not
  add an isolated module for it.
- `linear_tree` / `linear_lambda` unchanged by this lane.
- Forced splits are reachable from the Mojo API only: a parameter string cannot
  carry a filename that this lane may parse, and the JSON reader would live
  outside owned files.

## 8. Exact cross-lane patch requests

None of these files were touched.

**`src/mojoboost/__init__.mojo` (Task 01)** — add to the existing export block:

```mojo
from .efb import (
    EfbSettings,
    BundledMatrix,
    prepare_bundling,
    columns_for_features,
    expand_bundled_histogram,
    unbundle_histogram_into,
    check_bundling_honored,
    fit_bundles_dense,
    bundle_dense,
    dense_bin_counts,
    dense_default_bins,
    nondefault_rows_dense,
    count_nondefault_dense,
)
```

and add `map_forced_splits` to the existing `from .binning import ...` block.

**`tests/parallel/api_snapshot_manifest.json` (Task 01)** — add the same names
to `exports_by_module` for `efb` and `binning`.

**`bindings/_mojoboost.mojo` (Task 06) and `python/mojoboost/__init__.py`
(Task 07)** — surface `enable_bundle` and `max_conflict_rate`, and note in the
docstring that they apply on the dense CPU path only. The still-Mojo-only extra
parameters (forced splits) need no binding until a JSON reader exists.

**One-line honesty guard, per trainer.** Add on entry, before any tree is
grown, so an ignored `enable_bundle` is reported rather than dropped:

```mojo
check_bundling_honored(params.bundling, "<trainer name>")
```

- `src/mojoboost/tree_sparse.mojo`, `boosting_sparse.mojo` (Task 10)
- `src/mojoboost/train_gpu.mojo` (Task 01)
- `src/mojoboost/distributed.mojo` (Task 13)
- `src/mojoboost/ranking.mojo`, `ranking_advanced.mojo`, `custom_metric.mojo`,
  `objective.mojo`, `alternate_boosting.mojo`, `boosting_rf.mojo` (Task 08 /
  Task 17)

**`docs/LIGHTGBM_PARITY.md` and `tools/check_parity.py` (Task 19)** —
`enable_bundle` from `deferred` to `partial` (dense CPU trainers only);
`forcedsplits_filename` to `partial` (Mojo API only, no file reader); add a
`max_conflict_rate` row (accepted at 0.0, refused above); add a
`feature_pre_filter` row (refused by name, model already matches `false`).

## 9. Fallbacks preserved

- `DEFAULT_ENABLE_BUNDLE = False`. Default-model behavior is unchanged.
- `prepare_bundling` falls back to `BundledMatrix.none()` on three conditions,
  and `none()` means "grow on the original matrix" everywhere.
- Every bundling use site in `grow_tree` is under `if bundling.active`. With it
  false the grower runs its previous code exactly.
- Leaf-wise (best-first) growth is preserved. Forced splits do not change the
  growth discipline; they only override *which* frontier leaf splits next and
  on what, and leaf-wise selection resumes once the forced tree is exhausted.
- `max_conflict_rate > 0` stays refused, so no lossy path can be reached by
  accident.
- Off-CPU `enable_bundle=true` is refused by name rather than ignored.

## 10. Risks

- **Unbuilt.** No file here has been compiled. Mojo argument conventions,
  origins, and default-argument ordering in the touched signatures are
  unverified. The `BoosterParams` conversion from `@fieldwise_init` to an
  explicit `__init__` is the highest-risk edit for positional callers, even
  though `bundling` was placed last with a default.
- **`expand_bundled_histogram` cost.** O(#features x #bins) per node on top of
  the build. Whether bundling is a net win is a benchmark question and is
  entirely unmeasured. This is why automatic EFB stays off.
- **Dense default bin.** Defining it as the most frequent bin is a judgment
  call. It coincides with the bin of 0.0 on the sparse-ish data EFB targets,
  but on dense data with no dominant value the greedy will find few bundles
  and `use_bundling` should decline. Unverified.
- **Bin counts from the matrix.** `dense_bin_counts` reads observed widths off
  the training matrix rather than a `BinMapper`. Sound here only because the
  dense plan never encodes a row it did not see. Any future reuse of a dense
  plan across matrices would break this.
- **Forced splits and empty children.** The new error is the intended behavior
  but has never been triggered.
- **Concurrent-lane interaction.** See section 11.

## 11. Repository state notes

- **This lane committed nothing.** However, concurrent lanes swept this lane's
  working-tree edits into their own commits twice: first `e6f3959` ("Connect
  remaining parity and project stewardship work") picked up the then-current
  `efb` / `split` / `params` / `binning` / `boosting` /
  `tree_parameters_extra` edits, and a later commit in the range
  `5085097..63aad82` picked up `tree.mojo`. Nothing was lost, nothing was
  reverted, and no commit was amended. All work described here is present in
  `HEAD` and was re-verified there. Flagging it because the ownership rule
  assumed this lane's edits would still be unstaged at handoff time.
- **A concurrent lane edited an owned file.** `comptime MAX_BINS = 256` was
  added to `binning.mojo` by another lane. It was preserved, not reverted.
- `git diff --check` is clean.

## 12. Smallest later focused commands — UNRUN

None of these have been run. Run one at a time, never the whole suite.

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_efb.mojo

MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_tree_parameters_extra.mojo

MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/test_regularization.mojo

MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/test_bagging.mojo

MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/test_sparse.mojo

MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/test_capi.mojo
```

Order suggestion: `test_efb` first (it exercises the shared greedy that both
the sparse and dense entry points now delegate to, so a break there explains
everything downstream), then `test_tree_parameters_extra`, then
`test_regularization` and `test_bagging` as the default-behavior regression
check, then `test_sparse` and `test_capi` for the `BoosterParams` signature
change.

The first thing to confirm once a build is possible is the negative result:
with `enable_bundle` unset, the dense CPU trainers must produce the same model
they produced before this lane. Everything else is secondary to that.
