# Random-forest mode in mojoboost

Random-forest mode is LightGBM's `boosting='rf'`. This document specifies
what mojoboost implements, what it deliberately refuses, what it deliberately
does differently, and what has not been verified. It describes
`src/mojoboost/boosting_rf.mojo`, which is the algorithm core and is not a
public API.

Status, in one line: the core is written and its `boosting='rf'` surface is
imported by `src/mojoboost/alternate_boosting.mojo`; no random-forest code
has been compiled or run, and `boosting='rf'` is still rejected by the Python
layer and by `parse_params`. See `handoffs/remaining_02_rf.md`.

## 1. What random-forest mode is, and what it is not

It is not row bagging. Ordinary GBDT with `bagging_fraction` still boosts:
every round recomputes gradients at the current raw scores, so tree `i + 1`
corrects what trees `0..i` left behind, and the model is a sum. Row bagging
only changes which rows each correction is measured on.

Random-forest mode drops the correction. Gradients and hessians are computed
once, at the constant raw score `base_score`, and every tree in the run fits
those same numbers. The trees differ only through the randomization the
sampler and the feature draw supply. The model is an average, and there is no
shrinkage.

| | GBDT with bagging | `boosting='rf'` |
|---|---|---|
| gradients | recomputed every round, at the running raw score | computed once, at `base_score` |
| tree `i` depends on tree `i - 1` | yes | no |
| aggregation | sum | mean |
| shrinkage | `learning_rate` | forced to 1, `learning_rate` refused |
| a dropped tree | changes later trees | changes only the denominator |
| iteration ranges | additive: `[0, k)` plus `[k, n)` is the model | not additive: each range is its own mean |
| more rounds | adds bias correction | removes variance |

LightGBM implements this in `src/boosting/rf.hpp` as a `GBDT` subclass that
sets `average_output_ = true`, forces `shrinkage_rate_ = 1.0`, calls
`Boosting()` once from `Init` rather than once per iteration, and keeps the
training score as a running mean.

## 2. The round

For a single-output objective, once per run:

1. `base_score` is the objective's own base score, `boosting._base_score`,
   the same value `train` starts a boosted run from. LightGBM's
   `BoostFromAverage`.
2. `grad0`, `hess0` are `boosting.fill_grad_hess` at the constant raw-score
   vector `base_score`, once. LightGBM's `RF::Boosting()`, called from
   `Init`.

Then, for tree `i` of `n_estimators`:

3. Draw the row sample for round `i` (section 4).
4. Grow the tree with `tree.grow_tree(data, grad0, hess0, params.tree, bag,
   i)`. The grower is unmodified: every tree control applies to a forest as
   it does to a boosted ensemble.
5. For `quantile`, `mae`, and `mape`, replace the leaf values with a
   percentile of the residuals (section 3).
6. Add `base_score` to every node value, so the tree predicts on the raw
   scale by itself. LightGBM's `Tree::AddBias`.

No round is ever skipped and no run stops early: exactly `n_estimators` trees
are grown. The tree count is the denominator of the average, so dropping a
tree rescales every prediction the model makes, and with the gradients fixed
a degenerate tree says nothing about convergence.

## 3. Leaf renewal

The three objectives whose Newton step carries no curvature, `quantile`,
`mae`, and `mape` (`boosting.objective_renews_leaves`), replace each leaf
value with a percentile of the residuals. In GBDT the residual is `label`
minus the current score. In random-forest mode LightGBM's `residual_getter`
is `label` minus `init_score`, a constant.

`boosting._renew_leaf_values` already takes the raw-score vector as an
argument, so the constant vector is passed to it and there is no second
renewal path. Renewal runs before the bias is added, as in LightGBM. For a
constant offset `c`, `percentile(y - c) + c` is `percentile(y)`, so an L1
forest's leaves end up at the median of the labels in the leaf, which is what
a random forest is supposed to produce.

## 4. Required randomization

Because every tree fits identical gradients, an unrandomized run grows
`n_estimators` copies of one tree and averages them back to that tree.
LightGBM refuses such a configuration in `RF::Init`:

```cpp
if (config->data_sample_strategy == std::string("bagging")) {
  CHECK((config->bagging_freq > 0 && config->bagging_fraction < 1.0f &&
         config->bagging_fraction > 0.0f) ||
        (config->feature_fraction < 1.0f && config->feature_fraction > 0.0f));
} else {
  CHECK_EQ(config->data_sample_strategy, std::string("goss"));
}
```

`check_rf_params` is that check. `rf_randomizer_name` reports which source
satisfies it, testing in LightGBM's own order:

| Source | Parameter | In LightGBM's `CHECK` |
|---|---|---|
| uniform row bagging | `bagging_fraction < 1` with `bagging_freq > 0` | yes |
| gradient sampling | `data_sample_strategy=goss` | yes |
| per-tree feature sampling | `feature_fraction < 1` | yes |
| class-conditional row bagging | `pos_bagging_fraction` / `neg_bagging_fraction` | no, see section 8 |

`feature_fraction_bynode`, `feature_fraction_bylevel`, and `extra_trees` are
not accepted as the only source, matching LightGBM. They do decorrelate
trees, but accepting a configuration LightGBM aborts on would be a silent
difference in the accepting direction, and the error names the three
parameters that do work.

`RfParams.default()` is 100 trees on 80% row bags redrawn every round.
LightGBM has no default that satisfies its own check: `bagging_fraction`
defaults to 1.0 and `bagging_freq` to 0, which is exactly the combination
`RF::Init` aborts on.

## 5. Determinism

Every per-tree decision reads the absolute round index, exactly as
`boosting._boost_rounds` does.

| Decision | Keyed on | Module |
|---|---|---|
| row bag | `(bagging_seed, round / bagging_freq)` | `bagging.mojo` |
| balanced row bag | `(bagging_seed, round / bagging_freq)` | `sampling.mojo` |
| GOSS sample | `(goss_seed, round)` | `goss.mojo` |
| per-tree feature set | `(feature_fraction_seed, tree_index)` | `sampling.mojo` |
| per-node feature set | `(feature_fraction_seed, tree_index, node)` | `sampling.mojo` |

`tree_index` is `round` for a single-output forest and
`round * n_classes + k` for a multiclass one, so no two trees in a run share
a feature draw. None of the draws carries state between trees, so tree `i` is
the same tree whether it was grown in one call of 100 rounds or in the second
of two calls of 50.

`bagging_freq > 1` reuses one bag for that many consecutive trees, as in
LightGBM. Under constant gradients that means those trees differ only in
their feature draw, which is a weaker forest than `bagging_freq = 1`; it is
not an error and mojoboost does not warn about it.

## 6. Aggregation, and the two model representations

`RfBooster` is the model: `trees`, `base_score`, `objective`, `monotone`.
Prediction is the mean of the trees. The base score is inside the trees
(section 2, step 6), so nothing is added afterwards; `RfBooster.base_score`
is kept for continued training and for a reader who needs to interpret a
single tree's values.

`RfBooster.to_booster()` returns an ordinary `boosting.Booster` with a base
score of 0 and a `learning_rate` of `1 / T`. A `Booster` computes
`base_score + sum_i(learning_rate * tree_i)`, which is then the forest's own
mean. That is what makes an rf model reach the rest of the library today:
`serialize.save_model` writes it with no format change, and `contrib`,
`importance`, `inspection`, `gpu_predict`, and `model_dump` see a plain
ensemble.

Two things do not survive the bridge.

**Iteration ranges.** LightGBM divides a sliced prediction by the number of
iterations in the slice (`GBDT::Predict` divides by
`num_iteration_for_pred_`), so `[0, k)` is a full prediction from its own `k`
trees. `Booster.predict_raw_bins_range` divides by `T` whatever the range and
adds a base score of 0 to a range starting at 0. Only the full range is the
forest's own prediction. `RfBooster.predict_raw_bins_range` has LightGBM's
semantics.

**Continued training.** A bridged `Booster` reports a `learning_rate` of
`1 / T` and a base score of 0, so `boosting.train_more` would shrink new GBDT
trees by the averaging weight of the old ones and boost them from zero.
`train_rf_more` is the continuation entry point for a bridged forest, and it
checks the ensemble's shape with `is_forest` before touching it.

The mean through the bridge is accumulated pre-scaled, `sum(tree_i / T)`,
where `RfBooster.predict_raw_bins` accumulates and then divides,
`sum(tree_i) / T`. The two are the same arithmetic in a different
association. They agree exactly whenever `T` is a power of two, since scaling
by an exact power of two commutes with rounding; otherwise they can differ in
the last place.

## 7. Multiclass

LightGBM's `RF::Init` asserts `num_tree_per_iteration_ == num_class_`, so
random-forest mode covers softmax multiclass. One tree per class per round,
round-major as in `boosting.MulticlassBooster`, so the tree for round `i` and
class `k` is `trees[i * n_classes + k]`.

The base scores are the log class priors, weighted when `sample_weight` is
given, the same quantity `train_multiclass` computes. The softmax
probabilities of those constant scores are the same for every row, so the
per-class gradients are one pass for the whole run rather than one per round.
One bag serves a whole round, so the per-class trees of a round stay
comparable; each class's tree still draws its own feature set.

`RfMulticlassBooster.to_multiclass_booster()` bridges on the same terms as
the single-output case.

Multiclass is reachable from `train_forest_multiclass` only.
`alternate_boosting.mojo` refuses multiclass for both alternate modes, so
`boosting='rf'` with `objective='multiclass'` is not trainable through the
mode surface yet. That is a wiring gap, not an algorithm gap; the handoff
records it.

## 8. What is refused

| Refused | Where | Why |
|---|---|---|
| custom objectives | `check_rf_objective` | LightGBM: "RF mode do not support custom objective function". A gradient callback would be evaluated once, at a constant score |
| `lambdarank` | `check_rf_objective` | lambda gradients at a constant score carry no ranking, so every tree fits the same rank-free target |
| `init_score` | `check_rf_init_score` | a per-row offset moves the point each row's gradient is taken at, and a forest takes every gradient at one shared constant. LightGBM `CHECK_EQ(train_data->metadata().init_score(), nullptr)` |
| `learning_rate` other than 1.0 | `check_rf_learning_rate` | LightGBM accepts it and overrides it, so the number the user set is not the number the model trains with |
| no randomization | `check_rf_params` | section 4 |
| two row samplers at once | `boosting._check_goss`, `boosting._check_class_bagging` | both own the row list a tree is grown on. Unchanged from GBDT |
| balanced bagging outside binary | `boosting._check_class_bagging` | LightGBM reads `pos_bagging_fraction` for binary classification only |

## 9. Intentional differences from LightGBM

**Continued training keeps the base score.** LightGBM recomputes
`init_scores_` in `Boosting()`, and `BoostFromAverage` returns 0 once
`models_` is non-empty, so trees added to an existing rf model are fitted at
a raw score of 0 rather than at the base score the first trees used. That
makes 50 plus 50 rounds a different model from 100. `RfBooster` stores the
base score and `train_forest_more` reuses it, so the two agree here. A forest
held as a bridged `Booster` has nowhere to record it, so `train_rf_more`
recomputes it from the training data it is handed, which is exact under the
precondition `boosting.train_more` already states.

**Degenerate trees keep their value.** When no split passes the tree
constraints, LightGBM discards the grown stump and appends a tree whose
single leaf is 0, which pulls the average toward zero rather than toward the
objective's own constant. mojoboost renews, biases, and keeps the grown stump
like any other tree, so a forest that can find no split predicts the base
score rather than nothing.

**GOSS does not corrupt the shared gradients.** LightGBM's GOSS scales the
sampled rows' gradients in place, and since rf never recomputes them the
multipliers compound across rounds. Each round in mojoboost scales a copy.

**The GOSS warmup is keyed to the rf shrinkage.** LightGBM skips sampling for
`int(1 / config.learning_rate)` rounds, reading the configured
`learning_rate` even though rf has forced its shrinkage to 1. At the default
0.1 that is ten full-data rounds, which under constant gradients are ten
identical trees. mojoboost passes the rf shrinkage of 1.0, so the warmup is
one round.

**Balanced bagging counts as randomization.** LightGBM's `CHECK` reads
`bagging_fraction` alone and would abort on a run randomized by
`pos_bagging_fraction` / `neg_bagging_fraction`, although that is the same
per-round row draw. Section 4.

**The draws are counter-based splitmix64**, not LightGBM's per-block LCG, so
the individual rows and features drawn differ at equal seeds. The schedules,
the counts, and the distributions match. This is the trade `bagging.mojo` and
`goss.mojo` already make, for the reason they give.

## 10. What has not been verified

Nothing here has been compiled or run. No test file references
`boosting_rf.mojo`, no benchmark trains a forest, and no comparison against
LightGBM's rf output has been made. Every claim above is a reading of
`src/boosting/rf.hpp` and of the mojoboost modules named, and the parity
claims are claims about intent rather than measurements.

The first checks worth running are listed in section 5 of
`handoffs/remaining_02_rf.md`. The cheapest of them is also the sharpest: a
forest of `T` identical trees is what an unrandomized run would produce, so
`check_rf_params` refusing that configuration is the one invariant the whole
mode rests on.
