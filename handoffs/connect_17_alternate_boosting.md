# connect_17: alternate boosting modes

Scope: connect DART, random-forest boosting, linear trees, and CEGB **only
where an implementation actually exists**. Owned files:
`src/mojoboost/boosting_dart.mojo`, `src/mojoboost/boosting_rf.mojo`,
`src/mojoboost/linear_tree.mojo`, `src/mojoboost/cegb.mojo`,
`src/mojoboost/alternate_boosting.mojo`, and this handoff.

**Nothing was run.** No Mojo, no build, no test, no benchmark, no formatter,
no network. Every claim below is from reading the source. Nothing here is a
correctness, performance, parity, packaging, or hardware claim. Nothing was
committed by this lane.

---

## 1. Inventory: what already existed

| Capability | Implementation found | State on arrival |
|---|---|---|
| DART | `src/mojoboost/boosting_dart.mojo` (landed concurrently, during this lane's work) | Complete algorithm core over `(trees, weights, raw)`. Grows no tree, owns no loop, reachable from nothing. |
| Random forest | none | No file, no rule, no parameter. |
| Linear trees | none. Refused by name in `tree_parameters_extra.check_extra_option_supported` (`linear_tree`, `linear_lambda`) and in `lgbm_model_io` (`is_linear=1`, `leaf_const`, `leaf_coeff`) | Deliberately absent. `tree_parameters_extra`'s docstring already argues it is a subsystem, not a parameter. |
| CEGB | `tree_parameters_extra.FeaturePenalties` (`cegb_tradeoff`, `cegb_penalty_split`, `cegb_penalty_feature_coupled`) | The two computable split penalties are **already connected** to production: `TreeParams.extra.penalties` -> `tree._search` -> `split._feature_gain` -> `FeaturePenalties.penalized_gain`. `params.parse_params` accepts `cegb_tradeoff` and `cegb_penalty_split`. The coupled ledger and the lazy penalty are parsed and refused. |

Also relevant on arrival:

- `boosting.Booster` / `MulticlassBooster` carry **one** shrinkage scalar for
  the whole ensemble; `Booster.predict_raw_row` sums
  `learning_rate * tree.predict_row(...)`.
- `serialize.mojo` is at v3 and writes `objective`, `learning_rate`,
  `base_score`, an optional monotone section, optional categorical sections,
  and per-node covers. No per-tree anything.
- `params._MOJO_API_ONLY` lists `boosting` and `boosting_type`, so a
  parameter string reports them as Mojo-API-only.
- `python/mojoboost/__init__.py` line 381: `_BOOSTING_TYPES = ("gbdt", "goss")`.
- `docs/LIGHTGBM_PARITY.md` calls `dart` and `rf` deferred (section 7), and
  is **stale** in the same table on `min_gain_to_split`, `extra_trees`,
  `feature_contri`, `monotone_penalty`, `path_smooth`, `max_delta_step`, and
  the CEGB row: it says the rules exist "unintegrated" and that "no grower
  consults it", but `split.find_best_split` consults all of them today. Not
  this lane's file; see the patch requests.
- `tools/connectivity_audit.py` already classifies `alternate_boosting`,
  `boosting_dart`, and `boosting_rf` as `PENDING` / `connect_17`.

### Concurrency note

`boosting_dart.mojo` is nominally this lane's file, but a concurrent lane
wrote a full DART algorithm core into it while this lane was working. **That
file was kept as authoritative and was not edited, reverted, or reformatted.**
The DART trainer this lane had drafted into the same path was discarded
rather than merged, because the concurrent core is strictly better: it is
multiclass-shaped (`n_classes` runs through every entry point), it carries the
early-stopping snapshot DART actually needs, and it keeps a slow reference
(`dart_recompute_raw`) for the cache.

Separately, `src/mojoboost/boosting_rf.mojo` (written by this lane) was swept
into another lane's commit `dc21f03` before this lane finished. This lane did
not stage or commit anything. The worktree copy and `HEAD` agree.

---

## 2. Call path, before and after

### Before

```
model.fit / trainset / bindings / capi / cli
  -> boosting.train            (gbdt, and goss via GossParams)
     -> boosting._boost_rounds -> tree.grow_tree
boosting_dart.*                (no caller anywhere)
```

`boosting_dart` was an orphan. There was no random-forest code at all.

### After

```
alternate_boosting.fit_boosting            (raw features -> Model)
alternate_boosting.train_boosting          (binned matrix -> Booster)
alternate_boosting.train_boosting_more     (continue)
  |
  +-- gbdt / goss -> boosting.train / boosting.train_more   (UNCHANGED)
  |
  +-- dart -> alternate_boosting.train_dart / train_dart_more
  |            -> boosting_dart.select_drop
  |            -> boosting_dart.dart_begin_round
  |            -> boosting_dart.check_dart_supported
  |            -> boosting._fill_grad_hess
  |            -> bagging.refresh_bag
  |            -> tree.grow_tree
  |            -> boosting._renew_leaf_values
  |            -> boosting_dart.dart_normalization
  |            -> boosting_dart.dart_commit_round
  |            -> alternate_boosting.fold_weights_into_trees
  |            -> boosting.Booster (learning_rate = 1.0)
  |
  +-- rf   -> boosting_rf.train_rf / train_rf_more
               -> boosting._base_score, _fill_grad_hess, _renew_leaf_values
               -> bagging.refresh_bag
               -> tree.grow_tree
               -> boosting.Booster (learning_rate = 1 / K)
```

`boosting.mojo`, `tree.mojo`, `split.mojo`, `params.mojo`, `serialize.mojo`,
`model.mojo`, the bindings, the Python package, the tests, and every GPU file
are untouched by this lane.

**The chain is still not reachable from a user.** `src/mojoboost/__init__.mojo`
does not re-export `alternate_boosting`, so the connectivity audit will still
call all three modules orphans. That single export is patch request P1 below
and is the whole remaining gap for the Mojo API.

---

## 3. Connections completed

### 3.1 DART is now driven, not merely defined

`alternate_boosting._dart_rounds` is the loop the concurrent core was written
for and did not have. Per round it does exactly what the core's docstrings
specify:

1. `select_drop(dart, len(trees), round)` with the **absolute** round index.
2. `dart_begin_round(...)` subtracts the dropped iterations from the cached
   `raw` and keeps what it removed in a reused `contribution` buffer.
3. `refresh_bag`, then `_fill_grad_hess` **against the dropped-out `raw`**.
   This is the mechanism: the new tree fits the residual of an ensemble
   missing k of its trees.
4. `tree.grow_tree` on the bag.
5. `_renew_leaf_values` **against the dropped-out `raw`**, which is the
   requirement `check_dart_supported`'s docstring names for the leaf-renewing
   objectives (`mae`, `mape`, `quantile`). Renewing against the full
   ensemble's score would fit a residual the new tree is not responsible for.
6. `dart_normalization(k, learning_rate, xgboost_dart_mode)`.
7. `dart_commit_round(...)` scales the dropped weights, puts the rescaled
   share of `contribution` back into `raw`, appends the new tree at its
   weight, and advances `raw` by it.

Real state reaching the implementation: the binned matrix, the labels, the
sample weights, the objective and its `alpha`, the full `TreeParams` bundle
(including `extra`, monotone, interaction, categorical, feature sampling), the
bag, and the absolute round index. Output affecting behavior: the drop set
changes which gradients a round sees, and the normalization changes what every
affected tree contributes to every prediction.

### 3.2 Per-tree weights reach a single-scalar `Booster` exactly

The concurrent core's docstring states that a DART model "is not
representable by today's `Booster`, cannot be written by today's
`serialize.mojo`, and must not be reachable from `fit` until both carry a
weight vector."

**That premise is resolved rather than worked around**, and no weight vector
is needed in the model. `alternate_boosting.fold_weights_into_trees`
multiplies each tree's node values by that tree's weight and lets the
ensemble's factor be 1.0. This is what LightGBM itself does: `Tree::Shrinkage`
scales `leaf_value_` and `internal_value_` in place, and a LightGBM model file
stores already-shrunk leaf values with a per-tree `shrinkage` field that
records what was applied rather than a factor still to apply.
`lgbm_model_io.mojo` already documents mojoboost's writer doing the same
folding at export.

The equality is exact, not approximate. `Booster.predict_raw_row` computes
`s += learning_rate * leaf`; with `learning_rate = 1.0` and `w` already inside
`leaf`, it evaluates `fl(w * v)`, the same single rounded product the weighted
form would evaluate. The only residual difference is whether a compiler fuses
the multiply-add, which is the caveat `lgbm_model_io.mojo` already documents
for its own round trip.

Folding also covers **internal** node values, not just leaves, because
`contrib.mojo` conditions on node values and `lgbm_model_io` writes them as
`internal_value`. Split gains are **not** folded, matching LightGBM: they stay
the gains that were actually measured. A weight of exactly `1.0` is skipped, so
a tree no round ever rescaled comes back bit-identical to the grower's output,
and folding is idempotent.

### 3.3 Random forest

`boosting_rf.mojo` is new and is the whole of that mode: gradients computed
**once** at the constant offset, one tree per round on its own bag, `raw`
never updated, and the ensemble stored as `learning_rate = 1 / K` over the
trees actually kept. `boosting.Booster`'s single factor is exactly an average
when it is `1 / K`, so `average_output` needs no model field.

### 3.4 One vocabulary for `boosting`

`alternate_boosting.parse_boosting` / `boosting_name` /
`AlternateBoostingParams` are the only place the four LightGBM names resolve.
LightGBM's aliases (`gbrt`, `tree`, `gradient_boosting_decision_tree`,
`random_forest`) are accepted. No second registry: `gbdt` and `goss` route
into the untouched production `boosting.train`.

---

## 4. Duplicates fused or quarantined

| Duplicate | Disposition |
|---|---|
| This lane's own draft DART trainer in `boosting_dart.mojo` | **Deleted, not merged.** The concurrent core is the single authoritative DART implementation. |
| This lane's draft `DartParams`, `select_dropped`, `normalization`, `scale_tree`, `_dart_stream` | **Deleted.** Superseded by `boosting_dart.DartParams`, `select_drop`, `dart_normalization`, and the fold. |
| splitmix64 / uniform draw | **Not duplicated by this lane.** `boosting_dart` keeps its own copy of `_splitmix64`/`_uniform` (as `tree_parameters_extra` also does, with the same "one shared copy at integration" note). This lane adds no third copy. Consolidation is patch request P7. |
| `Tree` leaf/internal scaling | **Quarantined, documented.** `alternate_boosting.fold_weights_into_trees` is the one piece of `Tree` behavior implemented outside `tree.mojo`. It should become `Tree.shrinkage(rate)`; patch request P4. |
| `model.fit` | **Quarantined wrapper.** `alternate_boosting.fit_boosting` re-does only the bind-transform-wrap glue, calling the same `fit_bins`, the same mapper, and returning the same `Model`. It exists because `model.fit` has no mode argument and is scheduled for deletion by patch request P3. |
| CEGB | **No duplicate created.** `tree_parameters_extra.FeaturePenalties` stays the single implementation. See section 7. |

---

## 5. Definitions this lane is committing to

**Invariants.** `raw[r]` equals `base_score + sum_j weights[j] * trees[j](r)`
at the top and bottom of every DART round, and equals the dropped-out score in
between. For RF, `raw` is the constant offset and is never written. After
folding, `Booster.predict_raw_row` reproduces the trainer's final `raw` for
every training row, to floating-point association.

**Seeds.** All counter-based, no draw depending on how many draws preceded it.
DART: `drop_seed` (LightGBM default 4), stream keyed by `(seed, round)`, offset
0 the skip draw and offset `1 + i` iteration `i`'s draw. Bagging keeps
`bagging_seed` (3) and feature sampling `feature_fraction_seed`; the three
never share a stream. Every alternate mode reads the **absolute** round index,
so continuation follows the schedule an uninterrupted run would.

**Normalization.** `dart_normalization` only. With `k` dropped and `v` the
learning rate: `k = 0` gives `(new = v, dropped_scale = 1)`; LightGBM's own
mode gives `(v / (k + 1), k / (k + 1))`; `xgboost_dart_mode` gives
`(v / (k + v), k / (k + v))`. **These constants are NOT verified against
LightGBM's `dart.hpp`.** The specific open question, stated by the core's own
docstring, is whether LightGBM's factor is `v / (k + 1)` or `1 / (k + 1)`,
that is, whether normalization multiplies the shrinkage or replaces it. Do not
claim DART parity until that is read off LightGBM's source. It is isolated to
one function.

**Penalties.** Unchanged and untouched. `TreeParams.extra` flows into every
alternate mode exactly as it flows into `boosting.train`, so
`min_gain_to_split`, `max_delta_step`, `path_smooth`, `monotone_penalty`,
`feature_contri`, and the two computable CEGB split penalties apply inside
DART and RF trees with no extra wiring. `max_delta_step` and `path_smooth`
apply to a DART tree **before** its DART weight is folded in, which is the
right order: they cap and smooth the Newton output, and the weight is an
ensemble-level factor.

**Model representation.** No new representation. `Tree` unchanged, `Booster`
unchanged, `MulticlassBooster` unchanged. DART is `learning_rate = 1.0` with
folded values; RF is `learning_rate = 1 / K`.

**Objective compatibility.** Everything `boosting._check_objective` admits,
which is every built-in single-output objective. `CUSTOM` is refused by that
same check. The leaf-renewing objectives (`mae`, `mape`, `quantile`) are
supported and renew against the correct score in both modes. `lambdarank` is
refused by name in both.

**Multiclass / ranking.** Refused, single-output only. Both refusals name what
they would take (section 8). `boosting_dart` is already written for the
round-major multiclass layout, so multiclass DART is a loop, not a redesign.

**Constraints.** Monotone constraints survive DART: every weight
`dart_normalization` produces is positive, and a positive combination of trees
each monotone in a feature is monotone in it, so `Booster.monotone` keeps
meaning what it claims. Folding by a positive scalar preserves that.
Interaction constraints, categorical splits, feature subsampling, and row
bagging read no tree weight at all and are unaffected.

**Continued training.** `train_boosting_more` dispatches.
- RF is exact and cheap: gradients come from the stored `base_score`, so no
  pass over existing trees is needed, existing trees are untouched, and only
  the shared factor moves from `1 / K` to `1 / (K + added)`.
- DART rebuilds `raw` with `dart_recompute_raw`, lifts the folded ensemble
  back to weights with `dart_uniform_weights(n_trees, 1.0)`, adds rounds, and
  folds again. It is **not** bit-identical to one longer call: the single-call
  path carries `raw` forward incrementally through rescalings. The trees
  chosen agree; the last bits of the scores need not.
- DART continuation **rewrites** existing trees, because a round that drops a
  tree rescales it. The ensemble handed back is not the one handed in plus new
  trees. This is the one place in the library where that is true.
- The rate a DART ensemble was trained with is **lost** by folding, so
  `train_dart_more` takes it from its own `params` and cannot check it against
  the model. This is the single cost of the folded representation.

**Serialization requirements.** None. Both modes serialize through v3 today.

---

## 6. Fallbacks preserved

- `gbdt` and `goss` route into `boosting.train` / `boosting.train_more`
  verbatim, with the same positional arguments. No behavior change is possible
  for an existing caller, because no existing caller reaches this module.
- `AlternateBoostingParams()` defaults to `BOOSTING_GBDT` with
  `DartParams.disabled()`, so a caller that constructs the bundle and sets
  nothing gets today's trainer.
- `boosting.train` remains the production entry point and is unmodified. This
  module is the strategy switch, and it is a separate door precisely so the
  established path cannot regress before anything has been run.

Failures are explicit, never downgrades:

| Rejected | Where |
|---|---|
| `dart` or `rf` with GOSS enabled | `AlternateBoostingParams.validate`, `check_dart_supported` |
| `goss` mode with a disabled `GossParams`, and the converse | `AlternateBoostingParams.validate` |
| `dart` mode with a disabled `DartParams`, and an enabled one under another mode | `AlternateBoostingParams.validate` |
| `dart` on a non-CPU device | `check_dart_supported` |
| `dart` with `lambdarank` | `check_dart_supported` |
| `dart` with `uniform_drop=False` | `DartParams.validate` |
| `dart` with `drop_rate=0, skip_drop=1` (gbdt wearing dart's names) | `DartParams.validate` |
| `rf` with `learning_rate != 1.0` | `boosting_rf.check_rf_params` |
| `rf` without row bagging | `boosting_rf.check_rf_params` |
| `rf` with `lambdarank` | `boosting_rf._check_rf_objective` |
| continuing a gbdt/rf ensemble as dart | `train_dart_more` (unit-factor check) |
| continuing a gbdt/dart ensemble as rf | `boosting_rf.is_forest` |

`rf` refusing a learning rate rather than silently overriding it (LightGBM
sets `shrinkage_rate_ = 1`) follows the rule `params._check_alpha_key` already
states: a number the caller set that the run does not use hides a real
mistake. `is_forest` and the DART unit-factor check are **structural**: they
can catch a mismatch but cannot certify a match, since a `Booster` records no
mode. That is stated in both docstrings.

---

## 7. CEGB: no file created, and why

`src/mojoboost/cegb.mojo` was **not** created. `FeaturePenalties` in
`tree_parameters_extra.mojo` is the single authoritative CEGB implementation
and it is **already connected** to the production split search. Creating a
second CEGB module would be exactly the duplicate policy engine this task
forbids.

What is genuinely missing is one thing: the **coupled first-use ledger**.
`cegb_penalty_feature_coupled[f]` is charged the first time feature `f` is
split on anywhere in the ensemble. The hook is already in place and already
wired shut: `split._feature_gain` calls
`penalized_gain(g, feature, node_rows, True)` with `feature_already_used`
hard-coded to `True`, so the coupled term is never subtracted, and
`ExtraTreeParams.check_scalars` refuses a non-zero coupled vector up front.

This lane deliberately did **not** approximate it from an owned file. A
ledger maintained between trees by an alternate trainer would charge the cost
for every split on `f` in the first tree that uses `f`, rather than for the
first split alone. That is a different penalty from LightGBM's, and charging
it would be worse than refusing it. The exact edit is patch request P5.

`cegb_penalty_feature_lazy` stays refused: it charges for the rows that have
not yet read a feature, which is per-row state carried across the whole
ensemble. No change here.

---

## 8. Linear trees: no file created, and the required model-format change

`src/mojoboost/linear_tree.mojo` was **not** created. There is no linear-tree
implementation anywhere in the repository, and creating a placeholder because
the parity table lists the feature is exactly what this task forbids.
`tree.Tree` cannot represent a linear leaf, so **linear trees stay
disconnected.**

`Tree` holds one `Float64` per node in `value`. A linear leaf holds an
intercept plus one coefficient per numerical feature on its branch. That is
not a wider field, it is a different node kind, and it changes six things:

1. **`Tree` (model format).** New per-leaf arrays: a coefficient pool, a
   per-leaf `(offset, count)` into it, the feature ids the coefficients belong
   to, and a per-leaf intercept. Empty on a constant-leaf tree, so an existing
   model keeps its exact layout.
2. **Serialization.** A v4 bump. The section must be written only when a tree
   actually has linear leaves, the way v2's monotone and categorical sections
   are, so a constant-leaf model still serializes to the bytes it does today
   and a v3 file still loads.
3. **Growth inputs.** Raw, unbinned feature values must reach the grower
   alongside the `BinnedMatrix`. `grow_tree` currently receives bins only, and
   a ridge fit on bin ids is not the model LightGBM fits.
4. **Growth.** A per-leaf regularized normal-equation solve (`linear_lambda`),
   plus a rule for a rank-deficient system, which is the common case in a
   small leaf.
5. **Prediction.** A different leaf evaluation in `Booster.predict_raw_row`,
   `predict_row_sparse`, `gpu_predict.flatten_trees` and its kernel, and
   `lgbm_model_io` on both sides (`is_linear`, `leaf_const`, `leaf_coeff`,
   which it rejects today).
6. **Its own answers** for missing values inside a leaf's regression, for
   categorical features (LightGBM excludes them from the leaf model), and for
   TreeSHAP, which currently attributes a constant per leaf.

Until all six exist, `check_extra_option_supported` refusing `linear_tree` and
`linear_lambda` by name, and `lgbm_model_io` refusing `is_linear=1`, is the
correct behavior and this lane changed neither.

---

## 9. Remaining disconnections

1. **No public entry point reaches any of this.**
   `src/mojoboost/__init__.mojo` does not re-export `alternate_boosting`, so
   the connectivity audit will still report all three modules as orphans.
   Patch request P1. This is the single highest-value follow-up.
2. **No parameter string, Python, C ABI, or CLI route.** P2, P3, P6.
3. **DART validation-set early stopping is not connected.**
   `boosting_dart` provides `DartBestState`, `dart_record_best`, and
   `dart_restore_best` and none of them is called. There is deliberately no
   `train_dart_with_valid` rather than one that truncates and silently gets
   the weights wrong: a round after the best one may have rescaled a tree the
   best round contained, so popping trees recovers the right tree set and the
   wrong model. Shape of the missing function, for whoever adds it: mirror
   `boosting.train_with_valid`, score with `boosting._mean_loss`, call
   `dart_record_best(best, weights, loss, min_delta)` after each round,
   `dart_restore_best(best, trees, weights)` at the end, and fold **after**
   restoring.
4. **Multiclass DART and multiclass RF.** `boosting_dart` already threads
   `n_classes` through `select_drop` (iterations, not trees, so a round's
   whole class group drops together), `dart_begin_round`,
   `dart_commit_round`, and `dart_recompute_raw`. What is missing is the loop
   and a `MulticlassBooster`-shaped fold. RF multiclass is the same `1 / K`
   trick on `MulticlassBooster`, which also carries a single factor.
5. **Sparse and GPU trainers.** `boosting_sparse` and `train_gpu` own their
   own round loops and offer no mode. `check_dart_supported` already refuses
   DART on a non-CPU device with the reason (device-resident raw scores
   advance by one factor per round and there is no path for undoing a dropped
   tree's contribution on the device). RF on the GPU is easier: gradients are
   computed once and never updated.
6. **`dart_recompute_raw` is never checked against the incremental cache.**
   The core wrote it partly as that reference. Nothing compares them.
7. **`docs/LIGHTGBM_PARITY.md` is stale** on the whole `tree_parameters_extra`
   family and on `boosting`. P8.

---

## 10. Exact cross-lane patch requests

### P1 - `src/mojoboost/__init__.mojo` (owner: public Mojo API)

Insert after the `from .boosting_sparse import (...)` block:

```mojo
from .boosting_dart import (
    DartParams,
    check_dart_supported,
    dart_normalization,
    select_drop,
)
from .boosting_rf import check_rf_params, is_forest, train_rf, train_rf_more
from .alternate_boosting import (
    BOOSTING_DART,
    BOOSTING_GBDT,
    BOOSTING_GOSS,
    BOOSTING_RF,
    AlternateBoostingParams,
    boosting_name,
    fit_boosting,
    fold_weights_into_trees,
    parse_boosting,
    train_boosting,
    train_boosting_more,
    train_dart,
    train_dart_more,
)
```

This alone moves all three modules from orphan to connected in
`tools/connectivity_audit.py`. That table's `alternate_boosting`,
`boosting_dart`, and `boosting_rf` rows should then be deleted, not flipped to
`CONNECTED` (its own editing rules say so).

### P2 - `src/mojoboost/params.mojo` (owner: parameter parsing)

1. Remove `boosting` and `boosting_type` from `_MOJO_API_ONLY`.
2. Add a `var boosting: AlternateBoostingParams` field to `TrainConfig`.
3. Parse `boosting` / `boosting_type` through
   `alternate_boosting.parse_boosting`, and parse `drop_rate`, `max_drop`,
   `skip_drop`, `uniform_drop`, `xgboost_dart_mode`, `drop_seed` into
   `config.boosting.dart`. All six are scalars, so unlike
   `monotone_constraints` a parameter string **can** carry them; DART is
   fully expressible from a parameter string.
4. Add all seven names to `SUPPORTED_KEYS`.
5. In `_validate`, call `config.boosting.validate(goss)` once the GOSS
   selection is known.
6. **`boosting=rf` must still be refused from a parameter string**, because it
   requires `bagging_fraction` / `bagging_freq`, which are in `_MOJO_API_ONLY`.
   Refuse it with that reason by name rather than letting it fail later inside
   `check_rf_params`. Lifting the bagging keys into parameter strings is a
   separate decision, not this one.
7. Delete the `drop_rate / max_drop / skip_drop / xgboost_dart_mode /
   uniform_drop / drop_seed` "deferred" row from the parity table (P8).

### P3 - `src/mojoboost/model.mojo` (owner: end-to-end fit)

Add `boosting: AlternateBoostingParams = AlternateBoostingParams()` to
`model.fit` (after `params`, before `max_bins`, so no positional caller
breaks), and route the CPU branch through
`alternate_boosting.train_boosting` instead of `boosting.train`. Keep the GPU
branch on `train_gpu` and raise for a non-gbdt mode there, matching
`check_dart_supported`. Then **delete `alternate_boosting.fit_boosting`**; it
exists only because this argument does not.

### P4 - `src/mojoboost/tree.mojo` (owner: `Tree`)

Add LightGBM's own method:

```mojo
    def shrinkage(mut self, rate: Float64):
        """Multiply every node value by `rate`, LightGBM's `Tree::Shrinkage`.
        Internal node values as well as leaves: `contrib.mojo` conditions on
        them and `lgbm_model_io` writes them as `internal_value`. Split gains
        are not scaled."""
        for i in range(len(self.value)):
            self.value[i] = self.value[i] * rate
```

Then reduce `alternate_boosting.fold_weights_into_trees` to a loop over it,
keeping the `rate == 1.0` skip.

### P5 - `src/mojoboost/split.mojo` + `src/mojoboost/tree.mojo` (CEGB coupled ledger)

1. `split.find_best_split`: add `feature_used: List[Bool] = []` (empty means
   "treat every feature as already used", which is today's behavior exactly).
2. `split._feature_gain`: replace the hard-coded `True` with
   `len(feature_used) == 0 or feature_used[feature]`.
3. `tree.grow_tree`: add `mut feature_used: List[Bool]` (or a small ledger
   struct), thread it into `tree._search`, and set `feature_used[f] = True`
   when a split on `f` is committed, so the charge lands on the first split
   only and not on every split in the first tree that uses the feature.
4. `boosting._boost_rounds` (and the RF and DART loops here): own the ledger
   across the ensemble and hand the same one to every round. It is per-model,
   which is why no split search can own it.
5. `tree_parameters_extra`: drop the
   `check_extra_option_supported("cegb_penalty_feature_coupled")` call from
   `ExtraTreeParams.check_scalars`, and rewrite the `_feature_gain` docstring
   paragraph that explains why `True` is passed.

`cegb_penalty_feature_lazy` stays refused. Do not fold it into this patch.

### P6 - `python/mojoboost/__init__.py`, `bindings/`, `capi/`, `cli/`

- `_BOOSTING_TYPES` (line 381) becomes `("gbdt", "goss", "dart", "rf")`.
- Add `drop_rate`, `max_drop`, `skip_drop`, `uniform_drop`,
  `xgboost_dart_mode`, `drop_seed` as estimator arguments, defaulted to
  LightGBM's values, and pass them only when `boosting == "dart"`.
- **`boosting="rf"` must set `learning_rate=1.0` explicitly**, and must raise
  rather than silently override a rate the user set, matching
  `check_rf_params`. It must also require `subsample`/`subsample_freq`.
- Note for whoever writes this: sklearn users pass `learning_rate` by habit,
  so the `rf` error message needs to be good.

### P7 - splitmix64 consolidation (owner: `sampling.mojo`)

Three copies of the same mixing function now exist: `bagging._splitmix64`,
`tree_parameters_extra._mix64`, and `boosting_dart._splitmix64`. Each carries
a comment saying one shared copy belongs somewhere at integration. Pick one
home and delete the other two. This lane added none of them and none of the
three has a caller outside its own module.

### P8 - `docs/LIGHTGBM_PARITY.md` (owner: parity)

Section 7 currently says `min_gain_to_split` is `unsupported` and that "no
grower consults it", and says the same of `extra_trees`, `feature_contri`,
`monotone_penalty`, `path_smooth`, `max_delta_step`, and the CEGB split
penalties. `split.find_best_split` consults all of them today via
`TreeParams.extra`. Those rows are stale independently of this lane and
`tools/check_parity.py` should be re-run against them. Do not move the
`boosting` row to `supported` until P1 and P6 land and DART's normalization
constants have been checked (section 5).

---

## 11. Serialization and public-API effects

**Serialization: none.** No new field, no version bump, no reader or writer
change. DART lands as `learning_rate = 1.0` with folded values and RF as
`learning_rate = 1 / K`, both of which v3 already writes. A DART or RF model
round-trips today, and `contrib`, `importance`, `inspection`, and
`gpu_predict` read them with no change.

`lgbm_model_io` export is consistent for both, though unverified: it folds
`learning_rate` into the leaf values it writes and emits `shrinkage=<rate>`
per tree, so a DART model exports at `shrinkage=1` with values that already
carry every rescaling (which is what LightGBM's own DART files look like), and
an RF model exports as an equivalent summed GBDT with `shrinkage=1/K` rather
than as an `average_output` model. Its **importer** still refuses
`average_output` by name, which stays correct: reading such a file would need
the `1 / K` conversion applied on the way in, and that is a separate change.

**Public API: no change yet, by construction.** Nothing outside these three
files imports them. `boosting.train`, `model.fit`, `params.parse_params`, the
bindings, the C ABI, the CLI, and the Python package all behave exactly as
before. P1 through P3 and P6 are what make the modes reachable.

---

## 12. Risks

1. **The DART normalization constants are unverified** against LightGBM's
   `dart.hpp` (section 5). This is the top risk and it is a parity risk, not a
   crash risk: the mode trains either way.
2. **None of this has been compiled.** Highest-risk constructs, in order:
   `ref tree = trees[i]` in `fold_weights_into_trees` (precedent:
   `boosting.mojo:1700`); importing private names `_base_score`,
   `_check_objective`, `_check_sample_weight`, `_fill_grad_hess`,
   `_renew_leaf_values` from `boosting.mojo` (precedent: `objective.mojo:64`
   imports `_check_sample_weight`); passing `booster.trees` as a `mut`
   argument; `grown^` into `dart_commit_round`'s `var new_trees`.
3. **`train_dart_more` cannot verify the rate.** Folding loses it. A caller
   who continues with a different `learning_rate` gets a model no error
   describes. `boosting.train_more` refuses that mismatch; DART structurally
   cannot.
4. **`is_forest` and the DART unit-factor check are heuristics.** A GBDT model
   whose rate happens to be `1 / K`, or exactly `1.0`, passes. Both docstrings
   say so. The real fix is recording the mode on the model, which is a
   serialization change this lane deliberately did not make.
5. **DART memory and time.** A dropout round costs one pass over the dropped
   trees over all rows, plus two full-length buffers (`contribution` and the
   `raw` it writes). With `max_drop=50` that is up to 50 tree evaluations per
   row per round. Unmeasured.
6. **RF averaging denominator.** `1 / K` counts trees **kept**, not rounds
   requested; a degenerate round is skipped. Averaging over requested rounds
   would shrink the model toward the base score, but a caller who compares
   `n_estimators` against `len(trees)` will see them differ.
7. **`fit_boosting` and `model.fit` can drift** while both exist. P3 deletes
   one.
8. **Shared checkout.** `boosting_dart.mojo` changed underneath this lane once
   already, and `boosting_rf.mojo` was committed by another lane's `git add`.
   Re-read all three before building on this handoff.

---

## 13. Smallest later focused commands - ALL UNRUN

None of the following was executed. Run **one** of them per change, never the
full suite and never a build or benchmark loop.

```
# 1. Does the new code compile at all, and is the DART loop wired correctly?
#    Write tests/parallel/test_alternate_boosting.mojo first; see below.
mojo run -I src tests/parallel/test_alternate_boosting.mojo

# 2. After P1 only: are the three modules still orphans?
python3 tools/connectivity_audit.py

# 3. After P8 only: does the parity table match the repository?
pixi run check-parity
```

Smallest useful contents for `tests/parallel/test_alternate_boosting.mojo`,
in the order that isolates a failure fastest. This lane wrote no tests.

1. `parse_boosting` / `boosting_name` round-trip over the four names and
   LightGBM's aliases, and one unknown name raising.
2. `AlternateBoostingParams.validate` raising on each of the six refused
   mode/sampler combinations in section 6.
3. **The cheapest correctness check there is**, named by
   `dart_normalization`'s own docstring: a DART run with `skip_drop = 1.0` and
   `drop_rate = 0.0` never drops, so it must produce exactly the GBDT
   ensemble. Note this is currently **refused** by `DartParams.validate` as a
   configuration mistake, so the test needs `skip_drop = 1.0` with a non-zero
   `drop_rate` instead, which also never drops. Compare
   `train_dart(...).predict_raw_bins(...)` against
   `boosting.train(...).predict_raw_bins(...)` and expect exact equality after
   accounting for the fold (every weight is `learning_rate`, so folded values
   are `learning_rate * v` and the booster factor is 1.0, versus unfolded `v`
   at factor `learning_rate`; the two products are the same rounded value).
4. `train_rf` with `learning_rate != 1.0` raises; without bagging raises; and
   with both correct, `booster.learning_rate == 1.0 / len(booster.trees)`.
5. `train_rf` then `train_rf_more`: the first `K` trees are untouched
   (compare a copy) and the factor became `1 / (K + added)`.
6. `save_model` / `load_model` round-trip on a DART model and an RF model,
   then compare predictions. This is the claim in section 11 that nothing
   about serialization had to change.
7. `train_dart_more` raising on a booster whose factor is not 1.0, and
   `train_rf_more` raising via `is_forest` on a GBDT booster.
