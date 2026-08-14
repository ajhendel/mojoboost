# Task 12 handoff: remaining tree-parameter primitives

## What this lane shipped

| File | What it is |
|---|---|
| `src/mojoboost/tree_parameters_extra.mojo` | New module. Pure decision rules, one validated parameter bundle (`ExtraTreeParams`), one parsed data structure (`ForcedSplits`). Touches no grower, no histogram, no shared struct. |
| `tests/parallel/test_tree_parameters_extra.mojo` | 47 analytical tests. Run once: `pixi run mojo run -I src tests/parallel/test_tree_parameters_extra.mojo` — **47 passed, 0 failed, 0 skipped**. |

Nothing else in the repository was edited, nothing was staged or committed.
No other suite was run, so nothing here is a claim about the rest of the tree.

The module is inert until integrated: no existing call site references it, so
merging it alone changes no fit.

## Audit result: what already existed

These are implemented and owned elsewhere, and this lane added **no second
copy** of any of them: `num_leaves`, `max_depth`, `min_data_in_leaf`,
`min_sum_hessian_in_leaf`, `lambda_l1`, `lambda_l2`, `feature_fraction`,
`feature_fraction_bynode`, `feature_fraction_seed`, `monotone_constraints`,
`interaction_constraints`, `bagging_*`, `top_rate`/`other_rate`, and every
categorical hyperparameter (`cat_smooth`, `cat_l2`, `max_cat_threshold`,
`max_cat_to_onehot`, `min_data_per_group`) in
`categorical.CategoricalParams`.

The categorical helpers in the new module add no parameters. They are the
arithmetic `categorical.mojo` performs inline today (lines 389, 392, 402,
404-406, 437-441), factored out so the new rules can be applied to category
partitions and ordinal thresholds through one set of formulas. Integration
replaces the inline arithmetic with these calls, so no duplicate survives.

## Parameter map

Defaults are LightGBM's unless the row says otherwise.

| LightGBM name | Aliases | Default | Primitive | CPU | GPU |
|---|---|---|---|---|---|
| `min_gain_to_split` | `min_split_gain` | 0.0 | `passes_min_gain` | yes | yes |
| `max_delta_step` | `max_tree_output`, `max_leaf_output` | 0.0 | `cap_leaf_output`, `finish_leaf_output`, `split_gain_from_outputs` | yes | yes |
| `path_smooth` | — | 0.0 | `smooth_leaf_output`, `finish_leaf_output` | yes | yes |
| `feature_contri` | `feature_contrib`, `fc`, `fp`, `feature_penalty` | empty (all 1.0) | `FeaturePenalties.contri` | yes | yes |
| `cegb_tradeoff` | — | 1.0 | `FeaturePenalties.cegb_tradeoff` | yes | yes |
| `cegb_penalty_split` | — | 0.0 | `FeaturePenalties.cegb_penalty_split` | yes | yes |
| `cegb_penalty_feature_coupled` | — | empty | `FeaturePenalties.cegb_penalty_feature_coupled` | yes | yes |
| `cegb_penalty_feature_lazy` | — | empty | **explicit rejection** (`check_extra_option_supported`) | rejected | rejected |
| `extra_trees` | `extra_tree` | false | `extra_threshold_index` | yes | see note |
| `extra_seed` | — | 6 (`DEFAULT_EXTRA_SEED`) | `extra_split_stream` | yes | see note |
| `monotone_penalty` | `monotone_splits_penalty`, `ms_penalty`, `mc_penalty` | 0.0 | `monotone_penalty_factor`, `apply_monotone_penalty` | yes | yes |
| `monotone_constraints_method` | `monotone_constraining_method`, `mc_method` | `basic` | `parse_monotone_method` (accepts `basic`, rejects the other two by name) | yes | yes |
| `forcedsplits_filename` | `fs`, `forced_splits_filename`, `forced_splits` | empty | `parse_forced_splits` → `ForcedSplits` | yes | yes, with the row-partition note below |
| `linear_tree` | `linear_trees` | false | **explicit rejection**; deferred subsystem, see below | rejected | rejected |
| `linear_lambda` | `linear_l2` | 0.0 | **explicit rejection** | rejected | rejected |

CPU/GPU columns mean: after integration at the call sites below, the option is
live on that backend. The reason nearly every row is "yes" on both is that
`train_gpu.grow_tree_gpu`, `tree_sparse.grow_tree_sparse`, and
`tree.grow_tree` all call the *same* `tree._search` and `tree._leaf_value`
(`src/mojoboost/train_gpu.mojo:79`, `src/mojoboost/tree_sparse.mojo:50`); the
GPU differs only in how histograms are built and rows are partitioned, not in
how candidates are scored or leaves valued.

**`extra_trees` GPU note.** It changes only which candidate indices are
scored, and candidate scoring already happens on the host in `_search` for
every backend, so the GPU path needs no kernel change. If a future GPU-side
split scan lands, the rule to preserve is that the chosen index comes from
`extra_split_stream(seed, tree_index, node, feature)` and from nothing else,
so a GPU tree and a CPU tree remain identical.

**Forced-splits GPU note.** Applying a forced split needs a row partition on a
threshold that was not chosen by gain. On CPU that is `partition_rows_into`
(`src/mojoboost/tree.mojo:488`); on GPU it is the device partition in
`train_gpu.grow_tree_gpu`. Both take a `SplitInfo`, so a forced node is
applied by synthesizing a `SplitInfo` with the mapped bin and calling the
backend's existing partitioner. No new device code.

## Exact integration points

### 1. `src/mojoboost/tree.mojo`

- **`TreeParams` (line 76)**: add the fields of `ExtraTreeParams` — either the
  bundle as one field `var extra: ExtraTreeParams` (smallest diff, keeps
  `TreeParams.__init__`'s positional argument order intact) or flattened.
  Recommend the bundle: `TreeParams.default()` (line 134) then stays a
  one-liner and every existing positional caller is unchanged. Add
  `var extra: ExtraTreeParams = ExtraTreeParams()` as the last keyword
  argument of `__init__` (line 106).
- **`_leaf_value` (line 611)**: this is where `max_delta_step` and
  `path_smooth` land. It needs two new arguments, `n_data: Int` and
  `parent_output: Float64`, and must end with
  `finish_leaf_output(value, max_delta_step, path_smooth, n_data, parent_output)`.
  Call sites to update, all of which already have the row count in hand:
  - `src/mojoboost/tree.mojo:767` (root; `parent_output = 0.0`), `:847`,
    `:852` (children; `parent_output` is the parent leaf's finished value)
  - `src/mojoboost/train_gpu.mojo:201`, `:283`, `:288`
  - `src/mojoboost/tree_sparse.mojo:232`, `:339`, `:344`
  - `src/mojoboost/distributed.mojo:258` is a separate private `_leaf_value`
    with the same body; give it the same treatment, call sites `:480`, `:556`,
    `:560`.
  The parent's finished output must be carried on `_LeafState`
  (`src/mojoboost/tree.mojo:422`) as one more field, next to `bounds`.
- **`_search` (line 630)**: takes the node's `depth` already, which is what
  `monotone_penalty` needs, and takes `params`, which carries the bundle.
  Forward `params.extra` into `find_best_split`, plus the node's row count
  (`n_rows`, already an argument) and the node id (new argument, needed for
  the `extra_trees` stream; both growers already have it).
- **Frontier selection (line 785)**: unchanged. `min_gain_to_split` is
  enforced inside the scan, so a leaf whose best candidate fails the floor
  reports `found = False` and is simply never selected — the same mechanism
  the depth limit uses. Leaf-wise order is untouched.
- **Forced splits**: apply in `grow_tree` before the leaf-wise loop. Walk
  `ForcedSplits` breadth-first; for each forced node, map its raw
  `threshold` through the bin mapper to a bin id, build a `SplitInfo`,
  partition, and add the two children exactly as the main loop does, then
  let leaf-wise growth resume from the resulting frontier. `check_budget`
  has already guaranteed the forced tree fits inside `num_leaves` and
  `max_depth`, so the loop cannot run out of budget mid-way.

### 2. `src/mojoboost/split.mojo`

- **`find_best_split` (line 202)**: add one argument, the bundle (or the four
  values it needs). Then:
  - Replace the two acceptance tests `if gain > best.gain` (lines 381, 409)
    and the categorical one (line 322) with
    `if passes_min_gain(gain, min_gain_to_split) and gain > best.gain`. With
    the default floor of 0.0 this is exactly today's test, so an untouched
    configuration stays bit-identical.
  - Apply per-feature costs **once per feature**, after that feature's best
    candidate is known and before it is compared against `best` — that is
    LightGBM's placement, and it is also the only placement where
    `cegb_penalty_feature_coupled` can be charged once rather than per
    candidate. Use `FeaturePenalties.penalized_gain(gain, f, n_rows, used)`.
    This needs the node's row count passed in (it is `total_c` for a feature
    with no missing rows; pass the node's count explicitly rather than
    inferring it).
  - Apply `apply_monotone_penalty(gain, sign, depth, monotone_penalty)` on
    the same line, using the `sign` already computed at line 273 and the
    node depth passed down from `_search`.
  - `extra_trees`: inside the per-feature loop, when the flag is set, compute
    `n_candidates = n_scan - 1` (or `n_scan` when the feature reserves a
    missing bin, since the top threshold is then a real candidate) and
    `var pick = extra_threshold_index(n_candidates, extra_seed, tree_index, node, f)`,
    then evaluate only `b == pick`. `tree_index` must reach
    `find_best_split`; it is already threaded to `grow_tree` and to
    `grow_tree_gpu`. A feature whose single draw fails `min_data_in_leaf` or
    `min_child_hess` yields no split for that feature, which is LightGBM's
    behavior — no fallback scan.
  - `max_delta_step` / `path_smooth` change what a child emits, so when
    either is active the candidate must be scored with
    `split_gain_from_outputs` at the finished outputs rather than with the
    free-Newton formula at line 165. The monotone path at line 191 already
    does exactly this shape; extend that branch's condition from
    `constrained` to `constrained or capped or smoothed`.

### 3. `src/mojoboost/categorical.mojo`

- Replace the inline arithmetic with the shared helpers, so the two searches
  cannot drift: line 389 → `cat_enters_search`, line 392 → `cat_sort_key`,
  line 402 → `cat_effective_l2`, lines 404-406 → `cat_side_cap`, lines
  437-441 → `cat_partition_gain`.
- Apply the same gain floor and per-feature penalties to `CatSplit.gain`
  before it is returned, so a categorical candidate and a numerical one are
  still compared on equal footing.

### 4. `src/mojoboost/boosting.mojo`

- `_renew_leaf_values` (line 584, called at `:888` and `:1142`) replaces a
  leaf's Newton value with a residual percentile for quantile and L1. That
  value must go through `cap_leaf_output` and `smooth_leaf_output` too, or
  those two objectives silently escape `max_delta_step` and `path_smooth`.
  It already clamps into the monotone interval, so this is one more step in
  the same place.

### 5. `src/mojoboost/params.mojo`

- Add to `SUPPORTED_KEYS` (line 52) and to the parse chain (line 413):
  `min_gain_to_split` (+ alias `min_split_gain`), `max_delta_step`,
  `path_smooth`, `extra_trees`, `extra_seed`, `monotone_penalty`,
  `monotone_constraints_method`, `cegb_tradeoff`, `cegb_penalty_split`.
- Remove `monotone_constraints_method` and the cegb names from
  `_MOJO_API_ONLY` (line 62) only as each becomes reachable from a parameter
  string. `feature_contri`, `cegb_penalty_feature_coupled`, and
  `forcedsplits_filename` are vectors or documents, so keep them Mojo-API
  only (a parameter string cannot carry a per-feature vector any more than it
  can carry `monotone_constraints` today).
- Call `check_extra_option_supported(key)` in the parse loop *before* the
  unknown-key branch (line 511), so `linear_tree`, `linear_lambda`, and
  `cegb_penalty_feature_lazy` produce their named errors rather than
  "unknown parameter".
- `_validate` (line 348): call `config.booster.tree.extra.check(...)`. Note it
  needs `min_data_in_leaf`, because `path_smooth` requires at least 2.

### 6. `src/mojoboost/__init__.mojo`

Export from `.tree_parameters_extra`: `DEFAULT_EXTRA_SEED`,
`MONOTONE_BASIC`, `ExtraTreeParams`, `FeaturePenalties`, `ForcedSplits`,
`ForcedSplitNode`, `parse_forced_splits`, `parse_monotone_method`, and the
scalar rules a caller might want to reproduce (`passes_min_gain`,
`cap_leaf_output`, `smooth_leaf_output`). `tools/check_parity.py:636` parses
this file for the names the contract promises, so anything cited in
`docs/LIGHTGBM_PARITY.md` must appear here.

### 7. Bindings and Python

- `bindings/_mojoboost.mojo:294` `_parse_params`: read the new keys off the
  params dict, in the same style as `_parse_cat_params`. `feature_contri` and
  `cegb_penalty_feature_coupled` are per-feature vectors of doubles, so they
  arrive as addresses and go through `_f64_list`
  (`bindings/_mojoboost.mojo:183`), the way `_parse_monotone`
  (`bindings/_mojoboost.mojo:278`) reads its vector.
- `bindings/_mojoboost.mojo:609` `RESET_SLOTS`, `:615` `_write_reset`, `:631`
  `_read_reset`: add `min_gain_to_split`, `max_delta_step`, and `path_smooth`
  as new slots if they are to be schedulable; if they are added, bump
  `RESET_SLOTS` and update `python/mojoboost/callback.py:105 RESETTABLE` in
  the same commit — the two index one buffer and a one-sided change silently
  reassigns hyperparameters.
- `python/mojoboost/__init__.py`: constructor kwargs, the validation block ending at
  line 1225, and **both** params dicts (`:1226` and `:1428`). Use
  LightGBM's spellings, and accept its aliases the way `_resolve_alias`
  already does for `min_child_samples`/`reg_alpha`.

### 8. Serialization

**No change, and no version bump.** `serialize.mojo` writes the bin mapper and
the tree arrays; it does not write hyperparameters. Every option here changes
either a training-time decision or a leaf's `value`, and leaf values are
already serialized. Forced splits produce ordinary nodes. A model trained with
any of these options loads in an unmodified reader, and a v3 file written
before them still loads.

### 9. Docs and the parity checker

Rows in `docs/LIGHTGBM_PARITY.md` to flip once integrated: line 129
(`min_split_gain`), 297 (`extra_trees`/`extra_seed`), 301 (`max_delta_step`,
from `partial`), 304 (`linear_lambda`, keep `unsupported`), 305
(`min_gain_to_split`), 316 (`monotone_penalty`), 317 (`feature_contri`), 318
(`forcedsplits_filename`, from `unsupported` to `partial` — the document is
parsed, reading the file stays a CLI job), 320 (cegb: `partial`, since
`cegb_penalty_feature_lazy` is rejected), 321 (`path_smooth`). Line 315
(`monotone_constraints_method`) stays `different` and can now cite the
explicit rejection.

`tools/check_parity.py` will fail the build unless: every cited path exists,
and any cited `tests/*.mojo` suite is run by a pixi task
(`unwired_tests`, line 656). So citing
`tests/parallel/test_tree_parameters_extra.mojo` in the contract requires
adding it to the `test` task in `pixi.toml` in the same commit, or listing it
in `KNOWN_UNWIRED_TESTS`. Adding it to `test` is the right move.

## Semantics preserved

- **Leaf-wise growth.** Nothing here changes which leaf is split next. The
  gain floor, the penalties, and the monotone penalty change what a candidate
  scores; `extra_trees` and forced splits change which candidates exist. A
  leaf with no acceptable candidate reports `found = False` and is skipped,
  exactly as `max_depth` already makes it do.
- **Determinism.** `extra_trees` draws from a counter-based splitmix64 stream
  keyed by (seed, tree index, node id, feature) and by nothing else, matching
  `sampling.mojo`. Feature scan order, thread count, and training history
  cannot move a draw. This keeps `sampling.mojo`'s documented intentional
  difference from LightGBM (whose stream advances as training proceeds) and
  extends it to one more parameter.
- **Bit-identical default path.** `ExtraTreeParams.is_active()` is False for
  the defaults and is tested. Every integration point above should be written
  so that an inactive bundle takes today's code path, not a multiply-by-1.0
  version of it.

## Deferred: linear trees

`linear_tree` and `linear_lambda` are deliberately absent — no flag, no
placeholder, no field. A linear tree is not a tree control; it changes what a
leaf *is*. Each leaf holds a ridge regression over the numerical features on
its branch instead of a constant, which requires:

1. raw (unbinned) feature values retained alongside the binned matrix, per
   leaf, during growth;
2. a per-leaf normal-equation solve, regularized by `linear_lambda`, at every
   split evaluation and again at every leaf;
3. coefficient arrays on `Tree` and in the serialized format — a real version
   bump, unlike everything else in this handoff;
4. a different prediction path in every predictor, including the GPU one and
   the C ABI;
5. its own answers for missing values, categorical features, feature
   importance, and TreeSHAP (whose exact attribution assumes constant
   leaves).

That is a task of its own. Until it exists,
`check_extra_option_supported` refuses both names with a message saying what
they would take, which is the repository's rule for a real LightGBM feature
that is not implemented: say so, never ignore it.

## Verify before flipping the parity rows to `supported`

Three behaviors in this module are stated from LightGBM's documented rules
rather than from a line-by-line reading of its source. They are correct as
implemented and tested, but the parity contract should not claim
LightGBM-identical until they are checked against a LightGBM run
(`bench/compare_*.py` is the existing pattern):

1. **`path_smooth` at the root.** This module uses `parent_output = 0.0` for
   the root, which shrinks the root's own output toward zero. Confirm
   LightGBM does the same rather than skipping smoothing at the root.
2. **CEGB coupled-penalty ledger scope.** `penalized_gain` takes
   `feature_already_used` from the caller, so the caller decides whether the
   ledger is per-tree or per-model. LightGBM's is per-model (charged on a
   feature's first use anywhere in the ensemble); integration should use that
   and say so at the call site.
3. **`feature_contri` at zero.** Here a zero multiplier makes the feature
   unable to clear any floor, which is equivalent to exclusion. Confirm
   LightGBM does not additionally drop such a feature from the sampler.

Negative `feature_contri` is a deliberate difference, not an open question:
LightGBM accepts it, this module rejects it, because a negative multiplier
inverts a gain and breaks the invariant that a chosen split has positive gain.

## Not done here

- No grower, split search, binder, binding, or doc was edited — those are all
  the integration work above.
- Only the focused suite was run. No other Mojo suite, no pytest, no
  benchmark, no build. Nothing in this handoff is a performance claim.
