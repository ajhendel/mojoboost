# connect_17: alternate boosting modes

Scope: connect DART, random-forest boosting, linear trees, and CEGB **only
where an implementation actually exists**. Owned files:
`src/mojoboost/boosting_dart.mojo`, `src/mojoboost/boosting_rf.mojo`,
`src/mojoboost/linear_tree.mojo`, `src/mojoboost/cegb.mojo`,
`src/mojoboost/alternate_boosting.mojo`, and this handoff.

**Nothing was run.** No Mojo, no build, no test, no benchmark, no formatter,
no network. Every claim below is from reading the source. Nothing here is a
correctness, performance, parity, packaging, or hardware claim. This lane
staged and committed nothing; other lanes committed some of these files
underneath it (section 1).

---

## 1. Inventory, and what the shared checkout did to it

Both alternate-mode algorithm cores landed from **concurrent lanes while this
lane was working**, into paths nominally owned by this lane. Both were kept
and neither was edited, reverted, or reformatted. Two drafts this lane had
written into the same paths were discarded rather than merged, because both
concurrent cores are strictly better.

| Capability | Authoritative implementation | State on arrival |
|---|---|---|
| DART | `src/mojoboost/boosting_dart.mojo` (concurrent) | Complete algorithm core over `(trees, weights, raw)`. Grows no tree, owns no loop, reachable from nothing. Multiclass-shaped: `n_classes` runs through every entry point. Carries the early-stopping snapshot DART needs and a slow reference (`dart_recompute_raw`). |
| Random forest | `src/mojoboost/boosting_rf.mojo` (concurrent) | Complete: single-output and multiclass, `train_rf` / `train_rf_more` / `train_rf_with_valid` and the three multiclass counterparts, GOSS and class bagging, LightGBM's `AddBias` bias folding, `check_rf_*` validation. Keeps its **own** `RfBooster` / `RfMulticlassBooster` and bridges to `Booster` / `MulticlassBooster` with `to_booster` / `to_multiclass_booster`. Reachable from nothing. |
| Linear trees | none | Refused by name in `tree_parameters_extra.check_extra_option_supported` (`linear_tree`, `linear_lambda`) and in `lgbm_model_io` (`is_linear=1`, `leaf_const`, `leaf_coeff`). Deliberately absent. |
| CEGB | `tree_parameters_extra.FeaturePenalties` | The two computable split penalties are **already connected** to production: `TreeParams.extra.penalties` -> `tree._search` -> `split._feature_gain` -> `penalized_gain`. `params.parse_params` accepts `cegb_tradeoff` and `cegb_penalty_split`. The coupled ledger and the lazy penalty are parsed and refused. |

Drafts this lane discarded, so that no duplicate survives: a DART trainer
plus `DartParams` / `select_dropped` / `normalization` / `scale_tree` /
`_dart_stream` in `boosting_dart.mojo`, and a whole single-output-only RF
module in `boosting_rf.mojo`. Nothing of either remains.

Also relevant on arrival:

- `boosting.Booster` / `MulticlassBooster` carry **one** shrinkage scalar for
  the whole ensemble; `Booster.predict_raw_row` sums
  `learning_rate * tree.predict_row(...)`. There is no `average_output` flag.
- `serialize.mojo` is at v3: `objective`, `learning_rate`, `base_score`, an
  optional monotone section, optional categorical sections, per-node covers.
  No per-tree anything, and no mode.
- `params._MOJO_API_ONLY` lists `boosting` and `boosting_type`.
- `python/mojoboost/__init__.py` line 381:
  `_BOOSTING_TYPES = ("gbdt", "goss")`.
- `tools/connectivity_audit.py` already classifies `alternate_boosting`,
  `boosting_dart`, and `boosting_rf` as `PENDING` / `connect_17`.
- `docs/LIGHTGBM_PARITY.md` section 7 is **stale** independently of this
  lane: it says `min_gain_to_split`, `extra_trees`, `feature_contri`,
  `monotone_penalty`, `path_smooth`, `max_delta_step`, and the CEGB split
  penalties are unintegrated and that "no grower consults it", but
  `split.find_best_split` consults all of them today.
- `boosting_rf.mojo` names `handoffs/remaining_02_rf.md` and `boosting_dart.mojo`
  names `handoffs/remaining_01_dart.md` for their own remaining patches.
  Read those alongside this one; they are the cores' views, this is the
  connector's.

**Concurrency warning.** `boosting_dart.mojo`, `boosting_rf.mojo`, and an
earlier revision of this handoff were all committed by other lanes' `git add`
sweeps (`dc21f03`, `e6f3959`). Re-read all three source files before building
on anything below.

---

## 2. Call path, before and after

### Before

```
model.fit / trainset / bindings / capi / cli
  -> boosting.train            (gbdt, and goss via GossParams)
     -> boosting._boost_rounds -> tree.grow_tree
boosting_dart.*                (no caller anywhere)
boosting_rf.*                  (no caller anywhere)
```

### After

```
alternate_boosting.fit_boosting            (raw features -> Model)
alternate_boosting.train_boosting          (binned matrix -> Booster)
alternate_boosting.train_boosting_more     (continue)
alternate_boosting.train_boosting_with_valid          (early stopping)
alternate_boosting.train_boosting_multiclass          (-> MulticlassBooster)
alternate_boosting.train_boosting_multiclass_more
alternate_boosting.train_boosting_multiclass_with_valid
  |
  +-- gbdt / goss -> boosting.train / train_more / train_with_valid /
  |                  train_multiclass / train_multiclass_more /
  |                  train_multiclass_with_valid            (UNCHANGED)
  |
  +-- dart -> alternate_boosting.train_dart / train_dart_more
  |            -> boosting_dart.check_dart_supported
  |            -> boosting_dart.select_drop
  |            -> boosting_dart.dart_begin_round
  |            -> boosting._fill_grad_hess     (at the DROPPED-OUT score)
  |            -> bagging.refresh_bag
  |            -> tree.grow_tree
  |            -> boosting._renew_leaf_values  (at the DROPPED-OUT score)
  |            -> boosting_dart.dart_normalization
  |            -> boosting_dart.dart_commit_round
  |            -> alternate_boosting.fold_weights_into_trees
  |            -> boosting.Booster (learning_rate = 1.0)
  |
  +-- dart, the other three entry points -> REFUSED by name
  |     (multiclass: no round loop yet; with_valid: truncation is wrong)
  |
  +-- rf   -> alternate_boosting._check_rf_uniform_args
               -> boosting_rf.train_rf / train_rf_more
                    -> RfParams.from_booster_params
                    -> boosting_rf.check_rf_learning_rate / check_rf_params
                    -> boosting_rf.train_forest -> RfBooster
                    -> RfBooster.to_booster()  (base 0, rate 1 / T)
               -> boosting_rf.train_forest_with_valid
               -> boosting_rf.train_forest_multiclass
               -> boosting_rf.train_forest_multiclass_with_valid
                    -> RfMulticlassBooster.to_multiclass_booster()
               (multiclass continuation REFUSED; see 3.5)
```

`boosting.mojo`, `tree.mojo`, `split.mojo`, `params.mojo`, `serialize.mojo`,
`model.mojo`, the bindings, the Python package, the tests, and every GPU file
are untouched by this lane.

**The chain is still not reachable from a user.** `src/mojoboost/__init__.mojo`
does not re-export `alternate_boosting`, so the connectivity audit will still
call all three modules orphans. That single export is patch request P1 and is
the whole remaining gap for the Mojo API.

---

## 3. Connections completed

### 3.1 DART is now driven, not merely defined

`alternate_boosting._dart_rounds` is the loop the concurrent core was written
for and did not have. Per round:

1. `select_drop(dart, len(trees), round)` with the **absolute** round index.
2. `dart_begin_round(...)` subtracts the dropped iterations from the cached
   `raw` and keeps what it removed in a reused `contribution` buffer.
3. `refresh_bag`, then `_fill_grad_hess` **against the dropped-out `raw`**.
   This is the mechanism: the new tree fits the residual of an ensemble
   missing k of its trees.
4. `tree.grow_tree` on the bag.
5. `_renew_leaf_values` **against the dropped-out `raw`**, which is exactly
   the requirement `check_dart_supported`'s docstring states for the
   leaf-renewing objectives (`mae`, `mape`, `quantile`). Renewing against the
   full ensemble's score would fit a residual the new tree is not
   responsible for.
6. `dart_normalization(k, learning_rate, xgboost_dart_mode)`.
7. `dart_commit_round(...)` scales the dropped weights, returns the rescaled
   share of `contribution` to `raw`, appends the new tree at its weight, and
   advances `raw` by it.

Real state reaching the implementation: the binned matrix, the labels, the
sample weights, the objective and its `alpha`, the full `TreeParams` bundle
(including `extra`, monotone, interaction, categorical, feature sampling),
the bag, and the absolute round index. Output affecting behavior: the drop
set changes which gradients a round sees, and the normalization changes what
every affected tree contributes to every prediction.

### 3.2 DART's per-tree weights reach a single-scalar Booster exactly

The DART core's docstring states a DART model "is not representable by
today's `Booster`, cannot be written by today's `serialize.mojo`, and must
not be reachable from `fit` until both carry a weight vector."

**That premise is resolved rather than worked around, and no weight vector is
needed in the model.** `alternate_boosting.fold_weights_into_trees`
multiplies each tree's node values by that tree's weight and lets the
ensemble's factor be 1.0. This is what LightGBM itself does: `Tree::Shrinkage`
scales `leaf_value_` and `internal_value_` in place, and a LightGBM model
file stores already-shrunk leaf values with a per-tree `shrinkage` field that
records what was applied rather than a factor still to apply.
`lgbm_model_io.mojo` already documents mojoboost's writer folding the same
way at export.

The equality is exact. `Booster.predict_raw_row` computes
`s += learning_rate * leaf`; with `learning_rate = 1.0` and `w` already
inside `leaf` it evaluates `fl(w * v)`, the same single rounded product the
weighted form would evaluate. The only residual difference is whether a
compiler fuses the multiply-add, which is the caveat `lgbm_model_io.mojo`
already documents for its own round trip.

Folding covers **internal** node values as well as leaves, because
`contrib.mojo` conditions on node values and `lgbm_model_io` writes them as
`internal_value`. Split gains are **not** folded, matching LightGBM: they
stay the gains that were actually measured. A weight of exactly `1.0` is
skipped, so a tree no round ever rescaled comes back bit-identical to the
grower's output, and folding is idempotent, which is what makes
`train_dart_more` work.

### 3.3 Random forest is dispatched, and its adapter is not duplicated

This connector originally carried its own RF adapter: an `rf_params_of` that
built `RfParams` from the uniform argument set, a `train_forest` that kept
the `RfBooster`, a `to_booster()` bridge in `train_boosting`, and a refusal
of RF in `train_boosting_more`. **`boosting_rf` then landed all of that
itself**, in a later revision: `RfParams.from_booster_params`, `is_forest`,
and `train_rf` / `train_rf_more` taking exactly the uniform argument set and
returning an ordinary `Booster`, with `train_forest` / `train_forest_more` /
`train_forest_with_valid` as the `RfBooster`-native names.

**All four of this lane's RF pieces were deleted rather than kept.** The
local `train_forest` would also have collided with `boosting_rf.train_forest`
at the `__init__.mojo` re-export in P1. What remains here is dispatch plus
one refusal: `_check_rf_uniform_args`, because `train_rf` takes no
`ClassBaggingParams` and a forest randomized only by the class-conditional
fractions is a configuration `boosting_rf.rf_randomizer_name` explicitly
accepts, so it is named with the entry point that takes it rather than
dropped. GOSS under `rf` is refused the same way, in
`AlternateBoostingParams.validate`, which is the layer that sees it.

### 3.4 One vocabulary for `boosting`

`parse_boosting` / `boosting_name` / `AlternateBoostingParams` are the only
place the four LightGBM names resolve. The `rf` spellings are resolved by
calling `boosting_rf.is_rf_boosting` rather than by a second list, so the
names and the trainer that implements them cannot drift apart. `gbdt` and
`goss` route into the untouched production `boosting.train`.

### 3.5 The rest of the RF surface is connected, on request

`handoffs/remaining_02_rf.md` section 4.8 asked this lane for the multiclass
and validation-set dispatchers, and stated that `boosting_rf.mojo` needs
nothing new for them. Three entry points were added here accordingly:
`train_boosting_with_valid`, `train_boosting_multiclass`, and
`train_boosting_multiclass_with_valid`, each dispatching `rf` to the
already-written `train_forest_with_valid`, `train_forest_multiclass`, and
`train_forest_multiclass_with_valid`, and `gbdt`/`goss` to the untouched
`boosting.train_with_valid`, `train_multiclass`, and
`train_multiclass_with_valid`. A fourth, `train_boosting_multiclass_more`,
was added for `gbdt`/`goss` continuation.

This does not reintroduce the adapter deleted in section 4. The translation
is one call to `boosting_rf`'s own public `RfParams.from_booster_params` plus
its own `to_booster` / `to_multiclass_booster`; no `RfParams` is assembled
field by field here, and no check is restated. Only
`check_rf_learning_rate` is called directly, because these three
`train_forest_*` functions are the raw entry points and do not call it
themselves the way `train_rf` does.

Two refusals are new and deliberate:

- **`dart` on all three**, and on the multiclass continuation. Multiclass
  DART is a missing loop (section 9.4). Validation-set DART is refused
  because truncation is wrong for it, which is section 5's "continued
  training" argument applied to early stopping; the same function dispatches
  `rf` to a truncating implementation, because a forest's trees are
  independent and truncation there is exact.
- **`rf` on `train_boosting_multiclass_more`.** The single-output bridge
  recovers the constant by recomputing it from `target`; the multiclass
  constant is the vector of class log priors, and `boosting_rf` exposes no
  bridged multiclass continuation. Recomputing the priors here would be a
  second copy of `boosting_rf._class_log_priors`, so the error names
  `train_forest_multiclass_more` instead. Removing this refusal is a
  `boosting_rf` change (P10), not one for this file.

---

## 4. Duplicates fused or quarantined

| Duplicate | Disposition |
|---|---|
| This lane's draft DART trainer, `DartParams`, `select_dropped`, `normalization`, `scale_tree`, `_dart_stream` | **Deleted, not merged.** `boosting_dart` is the single DART implementation. |
| This lane's draft RF module (`train_rf`, `_grow_forest`, `is_forest`, `_forest_rate`, `check_rf_params`) | **Deleted, not merged.** `boosting_rf` is the single RF implementation and is multiclass-capable, which the draft was not. |
| This lane's RF adapter (`rf_params_of`, `train_forest`, the `to_booster` call, the RF refusal in `train_boosting_more`) | **Deleted after `boosting_rf` landed the same adapter** (`RfParams.from_booster_params`, `is_forest`, `train_rf`, `train_rf_more`). The local `train_forest` would additionally have collided with `boosting_rf.train_forest` at the P1 re-export. |
| Second `boosting` name list | **Avoided.** `parse_boosting` delegates the `rf` spellings to `is_rf_boosting`. |
| splitmix64 / uniform draw | **Not duplicated by this lane.** Three copies exist (`bagging._splitmix64`, `tree_parameters_extra._mix64`, `boosting_dart._splitmix64`), each with its own "one shared copy at integration" note. This lane added no fourth. Consolidation is P7. |
| `Tree` value arithmetic outside `tree.mojo` | **Quarantined, documented.** Two functions now do it: `alternate_boosting.fold_weights_into_trees` (multiply) and `boosting_rf._add_bias` (add). Both should become `Tree.shrinkage` and `Tree.add_bias`, LightGBM's own names; P4. |
| `model.fit` | **Quarantined wrapper.** `fit_boosting` re-does only the bin-transform-wrap glue, calling the same `fit_bins`, the same mapper, and returning the same `Model`. Scheduled for deletion by P3. |
| CEGB | **No duplicate created.** See section 7. |
| `RfBooster` / `RfMulticlassBooster` | **Not fused, and this is the one genuine parallel model representation in the tree.** This lane did not create it and cannot remove it: a forest averages and `Booster` has no `average_output` flag. It is bridged, not duplicated further. Collapsing it is P9, and it is the most architecturally significant follow-up here. |

---

## 5. Definitions this lane is committing to

**Invariants.** DART: `raw[r]` equals `base_score + sum_j weights[j] *
trees[j](r)` at the top and bottom of every round, and equals the dropped-out
score in between. After folding, `Booster.predict_raw_row` reproduces the
trainer's final `raw` for every training row, to floating-point association.
RF: `raw` is the constant offset and is never written; the bias is folded
into each tree by `_add_bias`, so a forest's tree predicts on the raw scale by
itself and the mean of the trees is the whole prediction.

**Seeds.** All counter-based; no draw depends on how many draws preceded it.
DART uses `drop_seed` (LightGBM default 4) with the stream keyed by
`(seed, round)`, offset 0 the skip draw and offset `1 + i` iteration `i`'s
draw. Bagging keeps `bagging_seed` (3), feature sampling
`feature_fraction_seed`, GOSS its own. No two share a stream. Every alternate
mode reads the **absolute** round index, so continuation follows the schedule
an uninterrupted run would.

**Normalization.** `dart_normalization` only. With `k` dropped and `v` the
learning rate: `k = 0` gives `(new = v, dropped_scale = 1)`; LightGBM's own
mode gives `(v / (k + 1), k / (k + 1))`; `xgboost_dart_mode` gives
`(v / (k + v), k / (k + v))`. **These constants are NOT verified against
LightGBM's `dart.hpp`.** The specific open question, stated by the core's own
docstring, is whether LightGBM's factor is `v / (k + 1)` or `1 / (k + 1)`,
that is, whether normalization multiplies the shrinkage or replaces it. Do
not claim DART parity until that is read off LightGBM's source. It is
isolated to one function.

RF's normalization is `1 / T` and is not in question, but `T` counts the
trees in the forest, so **adding trees rescales the whole model** and
prediction from a partially grown forest is not a prefix of prediction from
the finished one.

**Penalties.** Unchanged and untouched. `TreeParams.extra` flows into both
alternate modes exactly as it flows into `boosting.train`, so
`min_gain_to_split`, `max_delta_step`, `path_smooth`, `monotone_penalty`,
`feature_contri`, and the two computable CEGB split penalties apply inside
DART and RF trees with no extra wiring. Under DART, `max_delta_step` and
`path_smooth` apply to a tree **before** its DART weight is folded in, which
is the right order: they cap and smooth the Newton output, and the weight is
an ensemble-level factor.

**Model representation.** `Tree`, `Booster`, and `MulticlassBooster` are
unchanged. DART is `learning_rate = 1.0` with folded values. RF is
`RfBooster` natively, bridged to `Booster` at base score 0 and rate `1 / T`.

**Objective compatibility.** DART: everything `boosting._check_objective`
admits (every built-in single-output objective); `CUSTOM` refused by that
check, `lambdarank` refused by `check_dart_supported`. RF: the same, minus
`CUSTOM` and `lambdarank`, both refused by name in `check_rf_objective` with
LightGBM's own reasons. The leaf-renewing objectives (`mae`, `mape`,
`quantile`) are supported in both and renew against the correct score.

**Multiclass / ranking.** RF implements multiclass
(`boosting_rf.train_rf_multiclass` and friends), which this connector does
**not** yet expose: `train_boosting` is single-output. DART is single-output
throughout, though `boosting_dart` is already written for the round-major
multiclass layout (it drops whole iterations, not individual trees, so a
round's class group drops together), so multiclass DART is a loop rather than
a redesign. Ranking is refused by both modes by name.

**Constraints.** Monotone constraints survive DART: every weight
`dart_normalization` produces is positive, and a positive combination of trees
each monotone in a feature is monotone in it, so `Booster.monotone` keeps
meaning what it claims; folding by a positive scalar preserves that. RF
carries the same argument through its average and checks the constraint
vector on continuation. Interaction constraints, categorical splits, feature
subsampling, and row bagging read no tree weight and are unaffected by either
mode.

**Continued training.**
- DART: `train_dart_more` rebuilds `raw` with `dart_recompute_raw`, lifts the
  folded ensemble back to weights with `dart_uniform_weights(n_trees, 1.0)`,
  adds rounds, and folds again. It is **not** bit-identical to one longer
  call, because the single-call path carries `raw` forward incrementally
  through the rescalings. The trees chosen agree; the last bits of the scores
  need not. It **rewrites** existing trees, since a round that drops a tree
  rescales it, so the ensemble handed back is not the one handed in plus new
  trees. This is the only place in the library where that is true. The rate
  the ensemble was trained with is **lost** by folding, so it is taken from
  `params` and cannot be checked; that is the single cost of the fold.
- RF: dispatched from `train_boosting_more` to `boosting_rf.train_rf_more`,
  which gates on `boosting_rf.is_forest` (base score 0, rate 1 / n_trees)
  before it touches anything, since rewriting `learning_rate` on a boosted
  ensemble would silently rescale every tree in it. There are no raw scores
  to replay. The constant the new trees are fitted at is **not recoverable**
  from the model, because `to_booster` folded the averaging weight into the
  leaves and a `Booster` has nowhere to record a base score; it is
  **recomputed** from `target` and `sample_weight`, so the trees added match
  a single call exactly only when this is the data the forest was trained on
  — the precondition `boosting.train_more` already states. A forest kept as
  an `RfBooster` carries its own base score and needs no such precondition;
  that is what `boosting_rf.train_forest_more` is for. See risk 5b.

**Serialization requirements.** None for either mode as bridged. Both write
through v3 today. What a v4 would buy is in P9.

---

## 6. Fallbacks preserved

- `gbdt` and `goss` route into `boosting.train` / `boosting.train_more`
  verbatim. No behavior change is possible for an existing caller, because no
  existing caller reaches this module.
- `AlternateBoostingParams()` defaults to `BOOSTING_GBDT` with
  `DartParams.disabled()`, so a caller that constructs the bundle and sets
  nothing gets today's trainer.
- `boosting.train` remains the production entry point, unmodified. This module
  is a separate door precisely so the established path cannot regress before
  anything has been run.

Failures are explicit, never downgrades:

| Rejected | Where |
|---|---|
| `dart` with GOSS | `AlternateBoostingParams.validate`, `check_dart_supported` |
| `dart` on a non-CPU device | `check_dart_supported` |
| `dart` with `lambdarank` | `check_dart_supported` |
| `dart` with `uniform_drop=False` | `DartParams.validate` |
| `dart` with `drop_rate=0, skip_drop=1` (gbdt wearing dart's names) | `DartParams.validate` |
| `dart` with balanced (class-conditional) bagging | `train_boosting` |
| `dart` continued from a non-unit-factor ensemble | `train_dart_more` |
| `goss` mode with a disabled `GossParams` | `AlternateBoostingParams.validate` |
| an enabled `GossParams` under `gbdt` or `dart` | `AlternateBoostingParams.validate` |
| an enabled `DartParams` under any non-`dart` mode | `AlternateBoostingParams.validate` |
| `rf` with a learning rate other than 1.0 | `boosting_rf.train_rf` -> `check_rf_learning_rate` |
| `rf` with `init_score` | `check_rf_init_score` |
| `rf` with a custom objective or `lambdarank` | `check_rf_objective` |
| `rf` with no source of per-tree randomness | `check_rf_params` |
| `rf` with GOSS, through this entry point | `AlternateBoostingParams.validate` |
| `rf` with balanced bagging, through this entry point | `_check_rf_uniform_args` |
| `rf` continued from an ensemble that is not shaped like a forest | `boosting_rf.is_forest`, via `train_rf_more` |

Note what the two `rf` sampler refusals are and are not. A forest randomized
by GOSS, or by the class-conditional bagging fractions, is a **legal**
configuration that `boosting_rf.rf_randomizer_name` explicitly accepts. It is
refused *through this entry point only*, because `boosting_rf.train_rf` takes
neither a `GossParams` nor a `ClassBaggingParams`, and both errors name
`boosting_rf.train_forest` with an `RfParams` as the way to get it. That is a
narrowness of the uniform argument set, not a capability gap, and it is the
one place this connector is less capable than the core it dispatches to.

`rf` refusing a learning rate rather than silently overriding it (LightGBM
sets `shrinkage_rate_ = 1`) follows the rule `params._check_alpha_key` already
states: a number the caller set that the run does not use hides a real
mistake. The DART unit-factor check is **structural**: it can catch a mismatch
but cannot certify a match, since a `Booster` records no mode. Its docstring
says so.

---

## 7. CEGB decision resolved: `cegb.mojo` is authoritative

This handoff originally declined to create `src/mojoboost/cegb.mojo` and
proposed P5, a smaller `List[Bool]` first-use ledger. The later CEGB lane
implemented the complete state model and identified a substantive flaw in P5
under leaf-wise growth: cached best-split candidates retain a coupled
first-use charge after another leaf commits the same feature. Without
refunding that stale charge, the frontier ordering and therefore the tree do
not represent the stated CEGB formula.

The project decision is now explicit:

- `src/mojoboost/cegb.mojo` is the sole authoritative CEGB implementation.
- Its `CegbLedger`, per-row lazy ledger, EFB feature recovery, backend checks,
  and `cegb_stale_cached_gain` refund are the integration contract.
- P5 is withdrawn and must not be applied, partially or otherwise.
- The pre-existing CEGB fields and arithmetic in `FeaturePenalties` are a
  migration fragment, not a second authority. PATCH 2 in
  `handoffs/remaining_04_cegb.md` removes them during integration.
- Until that integration is complete, coupled and lazy CEGB remain refused;
  no backend may approximate them with the withdrawn ledger.

This resolution preserves the one-policy-engine rule while choosing the
implementation that correctly accounts for cached leaf-wise candidates.

---

## 8. Linear trees: no file created, and the required model-format change

`src/mojoboost/linear_tree.mojo` was **not** created. There is no linear-tree
implementation anywhere in the repository, and creating a placeholder because
the parity table lists the feature is what this task forbids. `tree.Tree`
cannot represent a linear leaf, so **linear trees stay disconnected.**

`Tree` holds one `Float64` per node in `value`. A linear leaf holds an
intercept plus one coefficient per numerical feature on its branch. That is
not a wider field, it is a different node kind, and it changes six things:

1. **`Tree` (model format).** New per-leaf arrays: a coefficient pool, a
   per-leaf `(offset, count)` into it, the feature ids those coefficients
   belong to, and a per-leaf intercept. Empty on a constant-leaf tree, so an
   existing model keeps its exact layout.
2. **Serialization.** A v4 bump. The section must be written only when a tree
   actually has linear leaves, the way v2's monotone and categorical sections
   are, so a constant-leaf model still serializes to the bytes it does today
   and a v3 file still loads.
3. **Growth inputs.** Raw, unbinned feature values must reach the grower
   alongside the `BinnedMatrix`. `grow_tree` receives bins only, and a ridge
   fit on bin ids is not the model LightGBM fits.
4. **Growth.** A per-leaf regularized normal-equation solve
   (`linear_lambda`), plus a rule for a rank-deficient system, which is the
   common case in a small leaf.
5. **Prediction.** A different leaf evaluation in `Booster.predict_raw_row`,
   `predict_row_sparse`, `gpu_predict.flatten_trees` and its kernel, and
   `lgbm_model_io` on both sides (`is_linear`, `leaf_const`, `leaf_coeff`,
   which it rejects today).
6. **Its own answers** for missing values inside a leaf's regression, for
   categorical features (LightGBM excludes them from the leaf model), and for
   TreeSHAP, which currently attributes a constant per leaf.

Until all six exist, `check_extra_option_supported` refusing `linear_tree` and
`linear_lambda` by name, and `lgbm_model_io` refusing `is_linear=1`, is the
correct behavior. This lane changed neither.

---

## 9. Remaining disconnections

1. **No public entry point reaches any of this.**
   `src/mojoboost/__init__.mojo` does not re-export `alternate_boosting`, so
   the connectivity audit will still report all three modules as orphans.
   P1. Highest-value follow-up.
2. **No parameter string, Python, C ABI, or CLI route.** P2, P3, P6.
3. **Multiclass RF continuation has no bridged route.** Resolved for
   training and early stopping in section 3.5, which connects
   `train_forest_multiclass` and `train_forest_multiclass_with_valid`.
   `train_forest_multiclass_more` remains reachable only as an
   `RfMulticlassBooster`, because the class log priors are folded away by
   `to_multiclass_booster` and re-deriving them here would duplicate
   `boosting_rf._class_log_priors`. P10, or P9, which removes the fold
   entirely.
4. **Multiclass DART does not exist.** `boosting_dart` is already shaped for
   it (`n_classes` in `select_drop`, `dart_begin_round`, `dart_commit_round`,
   `dart_recompute_raw`; it drops whole iterations so a round's class group
   goes together). What is missing is the loop and a
   `MulticlassBooster`-shaped fold.
5. **DART validation-set early stopping is not connected.**
   `DartBestState`, `dart_record_best`, and `dart_restore_best` exist and
   nothing calls them. There is deliberately no `train_dart_with_valid`
   rather than one that truncates and silently gets the weights wrong: a
   round after the best one may have rescaled a tree the best round
   contained, so popping trees recovers the right tree set with the wrong
   weights. Shape for whoever adds it: mirror `boosting.train_with_valid`,
   score with `boosting._mean_loss`, call
   `dart_record_best(best, weights, loss, min_delta)` after each round,
   `dart_restore_best(best, trees, weights)` at the end, and fold **after**
   restoring. It would then replace the `dart` refusal in
   `train_boosting_with_valid`, which is where a caller meets this today.
   Forests are already connected there (section 3.5), since truncation is
   exact for independent trees.
6. **A bridged forest's iteration ranges are still wrong.**
   `Booster.predict_raw_bins_range` divides by the whole tree count whatever
   the range, so `Model.predict_range` on an `rf` model is not the forest's
   own prediction. Documented in `RfBooster.to_booster`, in this connector,
   and here; enforced nowhere. P9 is the fix.
7. **Sparse and GPU trainers.** `boosting_sparse` and `train_gpu` own their
   own round loops and offer no mode. `check_dart_supported` already refuses
   DART on a non-CPU device with the reason (device-resident raw scores
   advance by one factor per round and there is no path for undoing a dropped
   tree's contribution on the device). RF on the GPU is easier: gradients are
   computed once and never updated.
8. **`dart_recompute_raw` is never checked against the incremental cache.**
   The core wrote it partly as that reference. Nothing compares them.
9. **`docs/LIGHTGBM_PARITY.md` is stale** on the whole
   `tree_parameters_extra` family and on `boosting`. P8.

---

## 10. Exact cross-lane patch requests

### P1 - `src/mojoboost/__init__.mojo` (owner: public Mojo API)

Insert after the `from .boosting_sparse import (...)` block:

```mojo
from .boosting_dart import (
    DEFAULT_DROP_RATE,
    DEFAULT_DROP_SEED,
    DEFAULT_MAX_DROP,
    DEFAULT_SKIP_DROP,
    DartBestState,
    DartDrop,
    DartNormalization,
    DartParams,
    check_dart_supported,
    dart_begin_round,
    dart_commit_round,
    dart_normalization,
    dart_recompute_raw,
    dart_record_best,
    dart_restore_best,
    dart_uniform_weights,
    dart_weights_are_uniform,
    select_drop,
)
from .boosting_rf import (
    RF_SHRINKAGE,
    RfBooster,
    RfMulticlassBooster,
    RfParams,
    check_rf_init_score,
    check_rf_learning_rate,
    check_rf_objective,
    check_rf_params,
    is_forest,
    is_rf_boosting,
    rf_randomizer_name,
    train_forest,
    train_forest_more,
    train_forest_multiclass,
    train_forest_multiclass_more,
    train_forest_multiclass_with_valid,
    train_forest_with_valid,
    train_rf,
    train_rf_more,
)
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
    train_boosting_multiclass,
    train_boosting_multiclass_more,
    train_boosting_multiclass_with_valid,
    train_boosting_with_valid,
    train_dart,
    train_dart_more,
)
```

`handoffs/remaining_02_rf.md` section 4.1 asks for the same `boosting_rf`
export block. The two requests are the same edit; land one of them, with the
union of the two symbol lists.

This alone moves all three modules from orphan to connected in
`tools/connectivity_audit.py`. That table's `alternate_boosting`,
`boosting_dart`, and `boosting_rf` rows should then be **deleted**, not
flipped to `CONNECTED`; its own editing rules say so.

### P2 - `src/mojoboost/params.mojo` (owner: parameter parsing)

1. Remove `boosting` and `boosting_type` from `_MOJO_API_ONLY`.
2. Add a `var boosting: AlternateBoostingParams` field to `TrainConfig`.
3. Parse `boosting` / `boosting_type` through
   `alternate_boosting.parse_boosting`, and parse `drop_rate`, `max_drop`,
   `skip_drop`, `uniform_drop`, `xgboost_dart_mode`, `drop_seed` into
   `config.boosting.dart`. All six are scalars, so unlike
   `monotone_constraints` a parameter string **can** carry them: DART is
   fully expressible from a parameter string.
4. Add all seven names to `SUPPORTED_KEYS`.
5. In `_validate`, call `config.boosting.validate(goss)` once the GOSS
   selection is known.
6. **`boosting=rf` must still be refused from a parameter string**, because
   it requires `bagging_fraction` / `bagging_freq` (or `feature_fraction`, or
   `data_sample_strategy=goss`), and the bagging keys are in
   `_MOJO_API_ONLY`. Refuse it with that reason by name rather than letting
   it fail later inside `check_rf_params`. Lifting the bagging keys into
   parameter strings is a separate decision.
7. `boosting=rf` must also reject an explicitly set `learning_rate` via
   `boosting_rf.check_rf_learning_rate`, and must **not** call it on a
   defaulted rate, which the function's own docstring warns about.

### P3 - `src/mojoboost/model.mojo` (owner: end-to-end fit)

Add `boosting: AlternateBoostingParams = AlternateBoostingParams()` to
`model.fit` (after `params`, before `max_bins`, so no positional caller
breaks) and route the CPU branch through
`alternate_boosting.train_boosting`. Keep the GPU branch on `train_gpu` and
raise for a non-gbdt mode there, matching `check_dart_supported`. Then
**delete `alternate_boosting.fit_boosting`**; it exists only because this
argument does not.

### P4 - `src/mojoboost/tree.mojo` (owner: `Tree`)

Add LightGBM's two node-value operations, which two modules now implement
outside `tree.mojo`:

```mojo
    def shrinkage(mut self, rate: Float64):
        """Multiply every node value by `rate`, LightGBM's `Tree::Shrinkage`.
        Internal node values as well as leaves: `contrib.mojo` conditions on
        them and `lgbm_model_io` writes them as `internal_value`. Split gains
        are not scaled."""
        for i in range(len(self.value)):
            self.value[i] = self.value[i] * rate

    def add_bias(mut self, bias: Float64):
        """Shift every node value by `bias`, LightGBM's `Tree::AddBias`."""
        for i in range(len(self.value)):
            self.value[i] = self.value[i] + bias
```

Then reduce `alternate_boosting.fold_weights_into_trees` to a loop over
`shrinkage` (keeping the `rate == 1.0` skip) and `boosting_rf._add_bias` to a
call to `add_bias`.

### P5 - WITHDRAWN: do not implement the plain coupled ledger

The former P5 proposed threading `List[Bool] feature_used` through split,
tree, and boosting. It is withdrawn because it did not refund coupled cost
from cached leaf candidates when another leaf first committed their feature.
That changes leaf-wise frontier ordering and can produce a tree inconsistent
with the CEGB formula.

Do not apply any part of the former P5. Use PATCH 1 through PATCH 4 in
`handoffs/remaining_04_cegb.md`, centered on `CegbConfig`, `CegbLedger`,
`prepare_cegb_node`, `cegb_commit_split`, and
`cegb_stale_cached_gain`. Those patches also fuse CEGB out of
`FeaturePenalties`, implement the lazy ledger and EFB feature recovery, and
retain explicit backend refusals.

### P6 - `python/mojoboost/__init__.py`, `bindings/`, `capi/`, `cli/`

- `_BOOSTING_TYPES` (line 381) becomes `("gbdt", "goss", "dart", "rf")`.
- Add `drop_rate`, `max_drop`, `skip_drop`, `uniform_drop`,
  `xgboost_dart_mode`, `drop_seed` as estimator arguments, defaulted to
  LightGBM's values, and pass them only when `boosting == "dart"`.
- `boosting="rf"` must require `learning_rate == 1.0` and raise rather than
  silently override a rate the user set, and must require one of
  `subsample`/`subsample_freq`, `feature_fraction`, or GOSS. sklearn users
  pass `learning_rate` by habit, so that error message needs to be good.
- `boosting="rf"` must NOT reject GOSS; it is a legal randomizer there.

### P7 - splitmix64 consolidation (owner: `sampling.mojo`)

Three copies of the same mixing function exist: `bagging._splitmix64`,
`tree_parameters_extra._mix64`, `boosting_dart._splitmix64`. Each carries a
comment saying one shared copy belongs somewhere at integration. Pick one
home and delete the other two. This lane added none of them and none has a
caller outside its own module.

### P8 - `docs/LIGHTGBM_PARITY.md` (owner: parity)

Section 7 says `min_gain_to_split` is `unsupported` and that "no grower
consults it", and says the same of `extra_trees`, `feature_contri`,
`monotone_penalty`, `path_smooth`, `max_delta_step`, and the CEGB split
penalties. `split.find_best_split` consults all of them today via
`TreeParams.extra`. Those rows are stale independently of this lane, and
`tools/check_parity.py` should be re-run against them. Do **not** move the
`boosting` row off `deferred` until P1 and P6 land and DART's normalization
constants have been checked (section 5).

### P9 - `average_output` on the model (owner: `Booster` + `serialize.mojo`)

The architectural follow-up, and the one that removes the parallel
representation. `boosting_rf` keeps `RfBooster` and `RfMulticlassBooster`
only because `Booster` cannot say "average these trees". Adding a
`var average_output: Bool` to `Booster` and `MulticlassBooster`, plus an
optional v4 section written only when it is true (the way v2's monotone and
categorical sections are, so an existing model serializes to the bytes it
does today), would let:

- `RfBooster` collapse into `Booster`, deleting `to_booster`, the split
  between `train_forest`/`train_rf` and `train_forest_more`/`train_rf_more`,
  and both sharp edges the fold creates: iteration ranges, and
  `train_rf_more` recomputing a base score it cannot read back
  (section 5, risk 5b);
- `Booster.predict_raw_bins_range` divide by the range's tree count rather
  than the whole ensemble's, which is what makes a forest's iteration ranges
  meaningful;
- `lgbm_model_io` stop refusing `average_output` on import, which it does
  today with a message that says mojoboost sums its trees.

Do this before, not after, P6: a Python `boosting="rf"` whose
`predict(num_iteration=k)` silently means something else is worse than no
Python route at all.

### P10 - `train_rf_multiclass_more` (owner: `boosting_rf.mojo`)

Small, and only worth doing if P9 is not. `train_boosting_multiclass_more`
refuses `rf` because a bridged `MulticlassBooster` has lost the class log
priors its trees were fitted at (section 3.5). The single-output path solves
this by recomputing the constant inside `boosting_rf.train_rf_more`; the
multiclass equivalent is the same three steps next to it:

- a structural check, the multiclass counterpart of `is_forest`
  (`base_scores` all zero, `learning_rate == 1 / n_iterations()`);
- `_class_log_priors(labels, n_classes, sample_weight)` to recompute the
  base scores, carrying the same precondition `train_rf_more` documents
  (exact only when this is the data the forest was trained on);
- `_rf_rounds_multiclass` for the new rounds, then rescale to the new size.

Then `train_boosting_multiclass_more` replaces its `rf` refusal with a call,
matching the single-output path exactly. P9 makes this unnecessary by
removing the fold, so prefer P9 if both are on the table.

---

## 11. Serialization and public-API effects

**Serialization: none.** No new field, no version bump, no reader or writer
change. DART lands as `learning_rate = 1.0` with folded values; a bridged RF
model lands as base score 0 with `learning_rate = 1 / T`. v3 already writes
both. Each round-trips today, and `contrib`, `importance`, `inspection`, and
`gpu_predict` read them with no change.

`lgbm_model_io` export is consistent for both, though unverified: it folds
`learning_rate` into the leaf values it writes and emits `shrinkage=<rate>`
per tree, so a DART model exports at `shrinkage=1` with values that already
carry every rescaling (which is what LightGBM's own DART files look like), and
a bridged RF model exports as an equivalent summed GBDT with `shrinkage=1/T`
rather than as an `average_output` model. Its **importer** still refuses
`average_output` by name, which stays correct until P9: reading such a file
would need the `1 / T` conversion applied on the way in.

**Public API: no change yet, by construction.** Nothing outside these three
files imports them. `boosting.train`, `model.fit`, `params.parse_params`, the
bindings, the C ABI, the CLI, and the Python package all behave exactly as
before. P1 through P3 and P6 are what make the modes reachable.

The seven functions P1 would expose (`train_boosting`,
`train_boosting_more`, `train_boosting_with_valid`,
`train_boosting_multiclass`, `train_boosting_multiclass_more`,
`train_boosting_multiclass_with_valid`, `fit_boosting`) each take every
argument its `boosting.mojo` counterpart takes, in the same positional
order, with `boosting: AlternateBoostingParams` inserted as the first
defaulted argument. A caller that passes `AlternateBoostingParams()` gets
`gbdt` and the production function verbatim, so the mode surface is a
superset of the existing one at every entry point and P3 can route
`model.fit` through it without changing any existing call.

---

## 12. Risks

1. **The DART normalization constants are unverified** against LightGBM's
   `dart.hpp` (section 5). Top risk, and a parity risk rather than a crash
   risk: the mode trains either way.
2. **None of this has been compiled.** Highest-risk constructs in the file
   this lane wrote, in order: `ref tree = trees[i]` in
   `fold_weights_into_trees` (precedent: `boosting.mojo:1700`,
   `lgbm_model_io.mojo:1016`); importing private names `_base_score`,
   `_check_objective`, `_check_sample_weight`, `_fill_grad_hess`,
   `_renew_leaf_values` from `boosting.mojo` (precedent: `objective.mojo:64`,
   and `boosting_rf.mojo` imports fourteen of them); passing `booster.trees`
   as a `mut` argument; `grown^` into `dart_commit_round`'s `var new_trees`;
   the ten-positional-argument calls into `boosting.train` /
   `boosting.train_more`.
3. **`train_dart_more` cannot verify the rate.** Folding loses it. A caller
   who continues with a different `learning_rate` gets a model no error
   describes. `boosting.train_more` refuses that mismatch; DART structurally
   cannot until the mode is recorded on the model (P9's neighborhood).
4. **The DART unit-factor check is a heuristic.** A GBDT model whose rate
   happens to be exactly `1.0` passes. Its docstring says so.
5. **A bridged forest's iteration ranges are wrong and nothing stops a
   caller reading them.** `Model.predict_range` on a `Model` built by
   `fit_boosting` from an `rf` run divides by the whole tree count whatever
   the range. Documented in three places, enforced in none. P9 is the fix;
   until then this is the sharpest edge in the connector.
5b. **`boosting_rf.train_rf_more` recomputes the base score** from `target`
   and `sample_weight` rather than recovering it, because `to_booster` folds
   it into the leaves. Its own docstring says so. Continuing a bridged forest
   on anything but the data it was trained on is therefore wrong in a way no
   error catches; `train_forest_more` on an `RfBooster` has no such
   precondition.
6. **DART memory and time.** A dropout round costs one pass over the dropped
   trees over all rows, plus two full-length buffers (`contribution` and the
   `raw` it writes). At `max_drop=50` that is up to 50 tree evaluations per
   row per round. Unmeasured.
7. **`fit_boosting` and `model.fit` can drift** while both exist. P3 deletes
   one.
8. **Shared checkout.** `boosting_dart.mojo` and `boosting_rf.mojo` were both
   replaced underneath this lane, and both plus an earlier revision of this
   handoff were committed by other lanes. Re-read all three before building
   on this document.

---

## 13. Smallest later focused commands - ALL UNRUN

None of the following was executed. Run **one** of them per change, never the
full suite and never a build or benchmark loop.

```
# 1. Does the new code compile, and is the DART loop wired correctly?
#    Write tests/parallel/test_alternate_boosting.mojo first; see below.
mojo run -I src tests/parallel/test_alternate_boosting.mojo

# 2. After P1 only: are the three modules still orphans?
python3 tools/connectivity_audit.py

# 3. After P8 only: does the parity table match the repository?
pixi run check-parity
```

Smallest useful contents for `tests/parallel/test_alternate_boosting.mojo`,
ordered so a failure isolates fastest. This lane wrote no tests.

1. `parse_boosting` / `boosting_name` round-trip over the four names and
   LightGBM's aliases, including `random_forest` (which routes through
   `is_rf_boosting`), and one unknown name raising.
2. `AlternateBoostingParams.validate` raising on each refused mode/sampler
   combination in section 6, and **not** raising for `rf` with GOSS enabled,
   which is the one asymmetry worth pinning.
3. **The cheapest DART correctness check there is**, named by
   `dart_normalization`'s own docstring: a run that never drops must produce
   exactly the GBDT ensemble. Use `skip_drop = 1.0` with a non-zero
   `drop_rate` (`drop_rate = 0` with `skip_drop = 1` is refused by
   `DartParams.validate` as a configuration mistake). Compare
   `train_dart(...)` against `boosting.train(...)` on the same data and
   expect equal predictions: every weight is `learning_rate`, so the folded
   values are `learning_rate * v` at factor 1.0 versus `v` at factor
   `learning_rate`, the same rounded product.
4. `train_boosting` with `AlternateBoostingParams.rf()` raising on a
   learning rate other than 1.0, and otherwise returning a `Booster` that
   satisfies `boosting_rf.is_forest`
   (`learning_rate == 1.0 / len(trees)`, `base_score == 0.0`).
5. `train_boosting` with `rf` raising for an enabled `GossParams` and for an
   enabled `ClassBaggingParams`, each with the message naming
   `boosting_rf.train_forest`. Then `train_boosting_more` with `rf` growing
   the forest and rescaling `learning_rate` to `1 / (T + added)`, and raising
   via `boosting_rf.is_forest` when handed a GBDT ensemble.
6. `save_model` / `load_model` round-trip on a DART model and on a bridged RF
   model, then compare predictions. This is the section 11 claim that nothing
   about serialization had to change.
7. `train_dart` then `train_dart_more`, checking only that the tree count grew
   and that a second `train_dart_more` on the result still runs (folding is
   idempotent). Do **not** assert bit equality against a single longer call;
   section 5 says it does not hold.
8. The three entry points added in section 3.5, each in one line:
   `train_boosting_multiclass` with `rf` returning a `MulticlassBooster`
   whose `learning_rate` is `1 / rounds` and whose `base_scores` are all
   zero, and with `gbdt` matching `boosting.train_multiclass` exactly;
   `train_boosting_with_valid` with `rf` returning at most
   `params.n_estimators` trees; and all four multiclass/valid entry points
   raising for `dart`. Nothing about the algorithms is being tested here,
   only that the dispatch reaches them, since `boosting_rf` owns their
   correctness.
9. `train_boosting_multiclass_more` raising for `rf` with the message naming
   `boosting_rf.train_forest_multiclass_more`, and adding rounds for `gbdt`.
   Delete this case when P10 or P9 lands.
