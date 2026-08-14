# Integration 08 handoff, EFB, extra tree parameters, expanded sampling

What this session did, in the nine files it owns, and nothing else.

| File | Change |
|---|---|
| `src/mojoboost/tree_parameters_extra.mojo` | `check_scalars`, three "who has to honor this" predicates, two new named refusals, docstring recording where each rule is live |
| `src/mojoboost/split.mojo` | `ExtraTreeParams` reaches the scan; per-feature cost placement; the `extra_trees` draw; candidate scoring at finished outputs |
| `src/mojoboost/tree.mojo` | `TreeParams.extra` and `TreeParams.feature_fraction_bylevel`; leaf values through `finish_leaf_output`; per-level feature draw; the grower-tier gate in `_search` |
| `src/mojoboost/categorical.mojo` | inline arithmetic replaced by the shared helpers, no parameter moved |
| `src/mojoboost/boosting.mojo` | class-conditional bagging in the three single-output trainers; leaf renewal honors the cap and the smoothing |
| `src/mojoboost/params.mojo` | thirteen new keys, named refusals before the unknown-key branch, scalar validation |
| `src/mojoboost/sampling.mojo` | `check_feature_fractions` validates the third fraction |
| `src/mojoboost/efb.mojo` | `check_bundling_supported`, `check_bundling_params`, docstring stating why bundling is still not reachable |
| `handoffs/integration_08_algorithms.md` | this file |

No test was written or run, nothing was executed, nothing was committed by this
session. Static reasoning only, as the brief required, so every claim below is
a reading of the code and not a measurement.

**State of the shared checkout.** A concurrent lane committed while this work
was in progress (`39fab1a`), and that commit swept in the then current contents
of `split.mojo` and `tree_parameters_extra.mojo`. The edits are intact on disk;
they are simply split across HEAD and the working tree. Nothing here depends on
that, but a reviewer diffing against `ab25ad1` rather than the working tree will
see two of the nine files in the wrong place.

---

## 1. What became reachable

### 1.1 Live on every grower that calls `tree._search`

That is `tree.grow_tree` (dense CPU), `tree_sparse.grow_tree_sparse`, and
`train_gpu.grow_tree_gpu` on its host split scan, which is the default. Each of
these rules is a function of the histogram, the node's row count, and the
node's depth, all of which every caller already passes, so they needed no
change on the caller's side.

| LightGBM name | Aliases accepted | Where it acts |
|---|---|---|
| `min_gain_to_split` | `min_split_gain` | floor on a feature's best candidate, `split._feature_gain` |
| `monotone_penalty` | `monotone_splits_penalty`, `ms_penalty`, `mc_penalty` | discount on a constrained split's gain, by depth |
| `monotone_constraints_method` | `monotone_constraining_method`, `mc_method` | validated, `basic` only, the other two refused by name |
| `feature_contri` | Mojo API only | per-feature gain multiplier |
| `cegb_tradeoff`, `cegb_penalty_split` | none | absolute cost subtracted per split, scaled by the leaf's rows |

The costs and the floor are charged once per feature, after that feature's own
scan and before it is compared against the running best. That is LightGBM's
placement and the only one where a per-feature cost is charged once rather than
once per candidate. A categorical feature's winning partition goes through the
same acceptance, so a category split and a threshold split still compete on
equal footing.

### 1.2 Live in `tree.grow_tree` alone

| LightGBM name | Aliases accepted | Where it acts |
|---|---|---|
| `extra_trees` / `extra_seed` | `extra_tree` | one drawn threshold per feature per node, `split.find_best_split` |
| `max_delta_step` | `max_tree_output`, `max_leaf_output` | `tree._leaf_value`, `split._split_gain`, `boosting._renew_leaf_values` |
| `path_smooth` | none | same three places |

`extra_trees` needs the node id and the tree index, and the two leaf-output
rules need a leaf's row count and its parent's finished output. No other grower
passes those. Rather than let a default 0 stand in for a node id, or let a
backend emit uncapped leaves for a caller who asked for a cap, `tree._search`
refuses an active value from a caller that has not set
`grower_applies_extra`. `tree.grow_tree` sets it; nothing else does. The gate
is `ExtraTreeParams.needs_grower_support()`.

Three details worth keeping.

- A leaf is now valued **before** its own split search runs, because its value
  is the `parent_output` its candidates' children smooth toward. That is a
  reordering of two statements at the root of `grow_tree` and nothing else.
- Candidates are scored at the finished outputs, not at the free Newton step,
  whenever either rule is on. LightGBM does the same, and a gain scored at the
  free step is not the gain the tree realizes. The monotone clamp is applied
  on top of the finished output, so an active constraint still bounds what a
  leaf emits.
- `boosting._renew_leaf_values` (quantile, L1, MAPE) applies the cap and the
  smoothing to the renewed percentile. Without that, those three objectives
  would be the only ones to escape both.

### 1.3 Sampling

| Name | Aliases accepted | Where it acts |
|---|---|---|
| `feature_fraction_bylevel` | `colsample_bylevel` | third draw between the tree's set and the node's, `tree.grow_tree` |
| `pos_bagging_fraction` / `neg_bagging_fraction` | Mojo API only | `boosting.train`, `train_more`, `train_with_valid` |

`feature_fraction_bylevel` is XGBoost's parameter. LightGBM has no per-level
fraction, so it belongs in the extensions section of the parity document rather
than in the LightGBM matrix.

Class-conditional bagging composes as the task 21 handoff predicted, and
nothing downstream of the draw changed.

- **With uniform bagging and with GOSS.** Exclusive, and refused rather than
  silently ignored, by `boosting._check_class_bagging`. LightGBM turns the
  loser off quietly; this repository already raises for GOSS against bagging
  and now does the same for the third sampler.
- **Objective.** Binary logistic only, as in LightGBM. The multiclass trainers
  do not accept the parameter at all, so the combination cannot be expressed.
- **The enable gate.** `class_bagging.enabled() and has_positive_rows(target)`,
  hoisted above the round loop, one label pass per run. With no positive row
  the fall back is plain `bagging_fraction`, which is LightGBM's own behavior
  and not a dead branch.
- **Counter-based streams.** `refresh_class_bag` draws bag `iteration // freq`
  from `splitmix64` keyed by (seed, bag index, row). No draw depends on how
  many draws came before it, so a continued run picks up exactly the bags an
  uninterrupted one would have, which is the property `_boost_rounds` already
  relies on through `round_offset`.
- **Ranker query sampling.** Untouched. `ranking.train_ranker` builds
  `row_bag` from whole query groups and passes it to `grow_tree` as an
  ascending row list; class bagging is refused for that objective, so the two
  cannot meet.
- **GPU row lists.** A class bag is an ascending, duplicate-free `List[Int]`,
  the same shape a uniform bag and a GOSS sample have, so
  `GpuActiveRows.begin_tree(bag)` takes it with no conversion. No GPU file was
  touched and none needs to be for this.
- **The degenerate-tree guard.** Now accepts a balanced round, or a stump
  drawn from an unlucky class bag would be read as convergence and end
  training.

### 1.4 Parameter strings

`params.parse_params` accepts `min_gain_to_split`, `max_delta_step`,
`path_smooth`, `extra_trees`, `extra_seed`, `monotone_penalty`,
`monotone_constraints_method`, `cegb_tradeoff`, `cegb_penalty_split`,
`feature_fraction_bylevel`, and their aliases, plus three keys it accepts only
at their defaults (`enable_bundle=false`, `max_conflict_rate=0.0`,
`data_sample_strategy=bagging`). `_validate` runs
`ExtraTreeParams.check_scalars`, so a bad value is rejected before any data is
read; the per-feature vectors are checked against the dataset in `grow_tree`,
because a parameter string cannot carry one.

Note that these keys reach `TreeParams` but **not** the Python layer, which
builds its own params dict for `bindings/_mojoboost.mojo` and never goes
through `parse_params`. Section 3 is the work that closes that.

---

## 2. What remains inert, and why

### 2.1 EFB, entirely

`efb.mojo` is unchanged apart from two guard functions and a docstring section.
`enable_bundle=true` raises, naming what it would take.

The reason is that bundling cannot be finished inside the files this session
owns. A plan is a function of the mapper and the training matrix, and
prediction cannot re-derive it, because a scoring matrix has a different
sparsity pattern. So a bundled model is only a model once the plan travels with
it, and that means `Model` and `MulticlassModel` (model.mojo, model_sparse.mojo),
a v4 section in `serialize.mojo`, the matching reader in
`python/mojoboost/inspection.py`, and the plan applied at
`model_sparse.fit_csc` and undone in every prediction path. None of those is
owned here.

The brief allowed EFB behind a disabled-by-default option unless the current
representation could recover original feature splits and the missing and zero
semantics without a serialization change. It cannot, so the option is refused
rather than offered. A flag that trains a model nobody can save or score is
worse than no flag, and a silently unbundled fit is the least visible failure
in the whole set, since nothing in the metrics would show it.

`max_conflict_rate` is refused above 0.0 for a second reason on top of that
one. It trades exactness for columns, and the loss lands in the trees rather
than anywhere a metric reports.

### 2.2 Forced splits

`parse_forced_splits` and `ForcedSplits` are unchanged and still validate a
document. `ExtraTreeParams.check_scalars` refuses a non-empty forced tree.

Applying a forced node means mapping its raw threshold to a bin, and `grow_tree`
is handed a `BinnedMatrix`, which carries `bins`, `n_rows`, `n_features`,
`n_bins`, `cats`, and `missing_bin`, and no bin edges. The mapper lives in
`Model`, one layer up. Applying forced splits is therefore a `grow_tree`
signature change plus every caller of it, which is six files outside this
ownership. The brief said not to create placeholders for forced split files, so
there is none.

### 2.3 The coupled CEGB penalty

`cegb_penalty_feature_coupled` is charged the first time a feature is split on
anywhere in the ensemble. That is per-model state, so it needs a feature-use
ledger threaded through every trainer and read by every grower, and there are
six trainers. `penalized_gain` is called with `feature_already_used = True`,
which is the "do not charge it" value, and `check_scalars` refuses a non-zero
vector so the parameter cannot be set and quietly ignored.

`cegb_penalty_split` and `cegb_tradeoff` are live, so the parity row for CEGB
is `partial` with two of four sub-parameters implemented and two refused by
name.

### 2.4 Linear trees

Unchanged from the task 12 handoff. `linear_tree` and `linear_lambda` are
refused by name. A linear tree changes what a leaf is, not how a tree is
controlled.

### 2.5 `extra_trees` with categorical features

Refused. LightGBM randomizes the categorical set search too, but over partition
positions rather than over thresholds, which is a different draw. Scoring
categoricals exhaustively while every numerical feature gets a single draw
would be neither LightGBM's behavior nor an honest approximation of it, so the
combination raises in `find_best_split`.

### 2.6 Two growers honor none of this

- **`distributed.grow_tree_distributed`** keeps private copies of `_search`
  (distributed.mojo:271) and `_leaf_value` (distributed.mojo:258) and calls
  `find_best_split` directly. It therefore ignores the whole bundle,
  including the tier that every other backend gets for free, and it cannot be
  gated from the files owned here because it never calls anything that sees
  both the parameters and the backend. This is the one place where an active
  setting is silently dropped, and closing it is two forwarded arguments
  (section 3.6).
- **`train_gpu._grow_tree_gpu_device_search`** scores candidates in a kernel
  and never calls `tree._search`. It is opt-in (`SPLIT_SEARCH_DEVICE`, or
  `MOJOBOOST_GPU_SPLIT_STRATEGY=device`) and already documented as differing
  from the host scan on near-ties, but it now also ignores the bundle and the
  per-level feature draw.

### 2.7 Still without a caller

From `sampling.mojo`, `expand_row_scale`, `contiguous_ranges`,
`ranges_row_count`, `row_mask`, `check_row_set`, `sampling_param_names`,
`canonical_sampling_param`, and `is_sampling_param` remain uncalled inside
`src/`. `canonical_data_sample_strategy` gained one caller, `parse_params`.
Wiring `canonical_sampling_param` into `cli/` and `capi/` is the cleanup the
task 21 handoff described and is not a blocker. `expand_row_scale` still waits
on a device-side objective that keeps gradients on the GPU.

---

## 3. Required edits outside this ownership

Ordered by what unblocks the most.

### 3.1 `src/mojoboost/__init__.mojo`, and the API snapshot with it

Nothing new is exported, so none of this is reachable as `mojoboost.X` from a
Mojo user's import. Add, in the alphabetical position the file already uses:

```mojo
from .efb import (
    EFB_MAX_BINS, EFB_NONE, EFB_SHARED_BIN, EfbParams, FeatureBundling,
    LocalHistogram, bundle_csc, check_bundling_params,
    check_bundling_supported, conflict_count, feature_bin_count, fit_bundles,
    nondefault_rows, pairwise_conflict, unbundle_histogram,
)
from .sampling import (
    ClassBaggingParams, DEFAULT_FEATURE_FRACTION_BYLEVEL,
    DEFAULT_NEG_BAGGING_FRACTION, DEFAULT_POS_BAGGING_FRACTION,
    canonical_data_sample_strategy, canonical_sampling_param,
    check_row_set, contiguous_ranges, expand_row_scale, has_positive_rows,
    is_sampling_param, ranges_row_count, refresh_class_bag, row_mask,
    sample_rows_by_class, sampling_param_names, select_level_features,
    select_split_features,
)
from .tree_parameters_extra import (
    DEFAULT_EXTRA_SEED, MONOTONE_BASIC, ExtraTreeParams, FeaturePenalties,
    ForcedSplitNode, ForcedSplits, cap_leaf_output,
    check_extra_option_supported, finish_leaf_output, parse_forced_splits,
    parse_monotone_method, passes_min_gain, smooth_leaf_output,
)
```

`tests/parallel/api_snapshot_manifest.json` keys `mojo.exports_by_module` by
module name and its own verification block records that it checks the module
set and every module's name set, so the three `exports_by_module` entries have
to land in the same change or the addition reads as API drift.

`pixi.toml` already runs `tests/parallel/test_efb.mojo`,
`tests/parallel/test_sampling.mojo`, and
`tests/parallel/test_tree_parameters_extra.mojo` in the `test` task, so that
part of the task 12, 13, and 21 handoffs is done and needs nothing further.

### 3.2 `bindings/_mojoboost.mojo`

`_parse_params` (around line 294) builds `TreeParams` by keyword, so the two
new fields are additive.

- Add `feature_fraction_bylevel=Float64(py=params["feature_fraction_bylevel"])`
  to the `TreeParams(...)` call.
- Add a `_parse_extra(params, n_features)` in the style of
  `_parse_cat_params`, returning an `ExtraTreeParams`, and pass it as
  `extra=`. `feature_contri` is a per-feature vector of doubles, so it arrives
  as an address and goes through `_f64_list` (around line 183) the way
  `_parse_monotone` reads its vector. Do **not** read
  `cegb_penalty_feature_coupled`; it is refused, and reading it would surface
  the refusal as a training error rather than a parameter error.
- Add `_parse_class_bagging(params)` in the style of `_parse_bagging`, reading
  `pos_bagging_fraction`, `neg_bagging_fraction`, `bagging_freq`, and
  `bagging_seed`, and pass it at the `train` / `train_more` /
  `train_with_valid` call sites. The multiclass trainers do not take it.
- Only if `min_gain_to_split`, `max_delta_step`, or `path_smooth` are to be
  schedulable by a callback: `RESET_SLOTS`, `_write_reset`, and `_read_reset`
  gain slots, and `python/mojoboost/callback.py` `RESETTABLE` gains the same
  names in the same commit. The two index one buffer and a one-sided change
  silently reassigns hyperparameters.

Every key read here must exist in the Python params dict, so 3.2 and 3.3 land
together.

### 3.3 `python/mojoboost/__init__.py`

Constructor keywords, the validation block, and **both** params dicts. Use
LightGBM's spellings and resolve the scikit-learn ones through
`_resolve_alias`, which already raises when two spellings disagree.

New keywords: `min_gain_to_split` (alias `min_split_gain`), `max_delta_step`,
`path_smooth`, `extra_trees`, `extra_seed`, `monotone_penalty`,
`monotone_constraints_method`, `cegb_tradeoff`, `cegb_penalty_split`,
`feature_contri`, `feature_fraction_bylevel` (alias `colsample_bylevel`),
`pos_bagging_fraction`, `neg_bagging_fraction`.

Do not add `subsample`-style aliases twice; `bagging_fraction`/`subsample` and
`bagging_freq`/`subsample_freq` are already resolved. Validate all three
feature fractions and both class fractions in (0, 1] with the `ValueError`
shape the file already uses for `bagging_fraction`.

### 3.4 `python/mojoboost/_sklearn.py`

The estimator passes shared parameters through `**kwargs`, so a bare
passthrough needs no signature change. Confirm against the snapshot before
assuming it: `python.shared_estimator_parameters` checks names and every
default value, so a newly named parameter is a snapshot change and a
passthrough is not.

### 3.5 Serialization

**No change and no version bump for anything in section 1.** `serialize.mojo`
writes the bin mapper and the tree arrays and no hyperparameters. Every control
integrated here changes either a training-time decision or a leaf's `value`,
and leaf values are already serialized. A model trained with any of them loads
in an unmodified reader, and a v3 file written before them still loads.

EFB is the exception and is not integrated. Its v4 section, and
`SUPPORTED_MODEL_FORMAT_VERSIONS` in `python/mojoboost/inspection.py`, are
described in `handoffs/task13_efb.md` sections 10 and 11 and are unchanged by
this session.

### 3.6 The other growers

Each is a small, mechanical opt-in.

- `src/mojoboost/distributed.mojo`. Its `_search` (line 271) should forward
  `params.extra`, `n_rows`, and `depth` to `find_best_split`, and its
  `_leaf_value` (line 258) should take the same four arguments
  `tree._leaf_value` now takes. Until then the distributed grower ignores
  every control in section 1 without saying so, which is the one silent gap
  this integration could not close from inside its own files.
- `src/mojoboost/tree_sparse.mojo` and `src/mojoboost/train_gpu.mojo`. Both
  already get section 1.1 for free. To get section 1.2 they need to pass
  `node=`, `tree_index=`, `parent_output=`, and `grower_applies_extra=True` to
  `_search`, and the four leaf-output arguments to `_leaf_value`, exactly as
  `tree.grow_tree` does. Both should also switch their three
  `select_node_features` calls to `select_split_features` so
  `feature_fraction_bylevel` is not silently dropped; at 1.0 the two are the
  same call.
- `src/mojoboost/train_gpu.mojo`, device split search. `_search_leaf_device`
  and `_grow_tree_gpu_device_search` should either honor the bundle or refuse
  an active one. Refusing is the right first move, since the kernel scores in
  Float32 and the penalties would need to move into it.

### 3.7 Leaf renewal on the other trainers

`boosting._renew_leaf_values` gained a trailing `extra` argument that defaults
to inactive, so these compile and behave as before, and they drop the cap and
the smoothing if a caller sets one.

- `src/mojoboost/custom_metric.mojo:667`, pass `current.tree.extra`.
- `src/mojoboost/train_gpu.mojo`, both `_renew_leaf_values` call sites, pass
  `params.tree.extra`.
- `src/mojoboost/boosting_sparse.mojo` has its own
  `_renew_leaf_values_sparse` (line 62, called at 190 and 279) and needs the
  same treatment written into it.

### 3.8 Docs and the parity checker

Not owned here, so nothing was flipped. What is now true:

- `min_split_gain`, `min_gain_to_split`, `max_delta_step`, `path_smooth`,
  `extra_trees`/`extra_seed`, `monotone_penalty`, and `feature_contri` are
  implemented, with the backend caveat in section 2.6.
- `monotone_constraints_method` stays `different` and can now cite an
  explicit rejection rather than silence.
- `forcedsplits_filename` moves to `partial`; the document is parsed and
  validated, applying it is refused.
- CEGB is `partial`, two of four.
- `linear_tree`, `linear_lambda` stay `unsupported`.
- `enable_bundle` stays `deferred`. `max_conflict_rate` deserves a row saying
  the same thing.
- `pos_bagging_fraction`/`neg_bagging_fraction` move off `deferred` for the
  Mojo API and stay unreachable from Python until 3.3.
- Known gap 3 (`colsample_bytree`, `colsample_bynode`) is closed at the
  parameter-string level; `tools/check_parity.py` keeps an exception list that
  must match the Known gaps section exactly, so closing it means editing the
  script in the same change.
- `feature_fraction_bylevel` belongs in the extensions section, not the
  LightGBM matrix.
- Known gap 1 says the sparse modules are unexported and unreachable. They are
  exported today (`__init__.mojo` lines 242 to 277). That row is stale
  independently of this work.

---

## 4. Default-behavior invariants

Every one of these is the reason a default configuration must produce the
model it produced before. They are stated so a reviewer can check them by
reading, since none of them was measured.

1. **`ExtraTreeParams()` is inactive and tested to be.** `is_active()` is
   False for the defaults, and `find_best_split` tests it once per node rather
   than multiplying by 1.0 per candidate.
2. **The per-feature restructure selects the same split.** The scan now keeps
   a feature-local best starting at 0.0 and compares it against the running
   best with a strict `>`, where it used to compare each candidate against the
   running best inline. Both accept a candidate exactly when it strictly beats
   everything before it, in the same feature order and the same bin order, and
   neither prunes, so the chosen `(feature, bin, default_left)` is unchanged.
3. **`_split_gain` is untouched on the old path.** With neither a constraint
   nor a finished output it returns the same expression it always returned.
   With a constraint and no finished output, `bounds.clamp(-g / (h + l2))` is
   computed as before, only spelled in two statements.
4. **`_leaf_value` returns early.** With both the cap and the smoothing off it
   returns the Newton step without calling `finish_leaf_output`, so there is no
   multiply-by-1 rounding difference.
5. **`feature_fraction_bylevel = 1.0` is the identity.** `select_level_features`
   returns the tree's set unchanged at 1.0, so `select_split_features` reduces
   to the `select_node_features` call it replaced. The task 21 suite asserts
   this directly.
6. **Class bagging is off unless asked for twice.** `enabled()` needs
   `freq > 0` and a fraction below 1.0, and the trainer additionally needs a
   positive row in the labels. Otherwise `refresh_bag` runs exactly as before.
7. **Leaf renewal allocates nothing new.** The parent-output table is built
   only when the smoothing is active; with an inactive bundle the function
   walks the same path and writes the same values.
8. **The categorical refactor is arithmetic-identical.** `cat_side_cap` is the
   same `min(max_cat_threshold, (used + 1) // 2)`, `cat_partition_gain` with
   `cat_l2 = 0.0` adds exactly 0.0 to `lambda_reg`, and `cat_enters_search`
   negates the same comparison.
9. **`TreeParams` grew by appending.** `feature_fraction_bylevel` and `extra`
   are the last two constructor arguments, both defaulted, so every positional
   caller in `src/`, `tests/`, `bench/`, and `bindings/` is unaffected.
   `serialize.mojo` does not reference `TreeParams` at all, so nothing depends
   on its arity.
10. **No serialized bytes move, no format version changes, no export changes.**
    The Python API, the C ABI, and the CLI therefore behave exactly as they did,
    because none of them can yet set any of the new fields.
11. **Determinism.** The `extra_trees` draw is counter-based splitmix64 keyed by
    (seed, tree index, node id, feature) and the per-level draw by
    (seed, tree index, depth), matching what `sampling.mojo` already does and
    already documents as an intentional difference from LightGBM's advancing
    stream. No draw depends on scan order, thread count, or training history,
    so a CPU tree and a GPU tree drawn from the same parameters agree.

---

## 5. Focused validation to run next

One test per change, never the whole suite, and each under the machine-wide
build lock. Run them in this order and stop at the first failure.

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_tree_parameters_extra.mojo
```
The module's own 47 tests. `check_scalars` now refuses a non-empty forced tree
and an active coupled vector, so
`test_bundle_validation_rejects_rather_than_clamps` still raises where it
expects to, but from a different line. This is the one suite whose expectations
this session's edits touch directly.

```
... pixi run mojo run -I src tests/test_regularization.mojo
```
Leaf values and the default gain path through the restructured scan. This is
the cheapest test that would catch invariant 2 or 4 being wrong.

```
... pixi run mojo run -I src tests/test_missing.mojo
```
The missing-value scan, whose two `continue` guards became one negated `if`.
Behavior-identical by inspection, and this suite is where it shows if not.

```
... pixi run mojo run -I src tests/test_categorical.mojo
```
The helper substitution, invariant 8.

```
... pixi run mojo run -I src tests/test_feature_sampling.mojo
```
The `select_split_features` swap at all three call sites, invariant 5.

```
... pixi run mojo run -I src tests/parallel/test_sampling.mojo
```
The 21 sampling tests, including the drop-in equality this integration relies
on and the `check_feature_fractions` callers that still pass two arguments.

```
... pixi run mojo run -I src tests/test_monotone.mojo
```
The constrained scoring path, which `_split_gain` was restructured around.

```
... pixi run mojo run -I src tests/test_bagging.mojo
... pixi run mojo run -I src tests/test_goss.mojo
```
The two samplers class bagging is now exclusive with.

```
... pixi run mojo run -I src -I capi tests/test_capi.mojo
```
`parse_params` and `params_names_mojo_api_only`, which gained a
`data_sample_strategy=goss` case.

```
... pixi run mojo run -I src tests/test_sparse.mojo
... pixi run mojo run -I src tests/test_distributed.mojo
```
The two growers that call into `tree.mojo` without opting in. Both should be
unchanged; they are here because a default-argument mistake in `_search` or
`_leaf_value` would show up first as a sparse or distributed tree that no
longer matches the dense one.

```
python3 tools/check_parity.py
```
Last, and only after section 3.8 is written, since it fails on any cited path
that does not exist and on any exception list that does not match.

Not run here, and therefore not claimed: any of the above, `pixi run test`,
`pixi run test-python`, the GPU suites, and any benchmark. Nothing in this
handoff is a performance claim.
