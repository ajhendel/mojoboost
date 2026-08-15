# DART in mojotrees

DART is LightGBM's `boosting='dart'`: Dropouts meet Multiple Additive
Regression Trees (Rashmi and Gilad-Bachrach, 2015). This document specifies
what mojotrees implements, what it deliberately refuses, and what has not
been verified. It describes `src/mojotrees/boosting_dart.mojo`, which is the
algorithm core and is not a public API.

Status, in one line (2026-08-15): `boosting='dart'` is reachable from the
estimators, from `_mojotrees.fit`, and from `alternate_boosting.fit_boosting`
/ `fit_boosting_multiclass` on the Mojo API; both drop rules are read off
LightGBM's `dart.hpp`; `tests/test_alternate_boosting.mojo` and
`python/tests/test_params.py` run it. Not done: `eval_set` early stopping
under dart from any entry point, and any comparison against LightGBM's
output. Section 11 records what was and was not verified.

## 1. The problem DART solves

In GBDT, tree `i + 1` is fitted to the residual left by trees `0..i`. Later
trees therefore specialize on whatever the early trees failed to fit, and
their contribution shrinks toward nothing. The ensemble ends up dominated by
its first few trees, which is over-specialization: adding rounds stops buying
accuracy long before the round budget is spent.

DART borrows dropout from neural networks. Each round hides a random subset
of the trees already grown, fits the new tree to the residual of what
remains, and then puts the hidden trees back at a reduced weight. A tree
grown this way cannot be a footnote to the trees before it, because it never
saw all of them.

## 2. The round

Let `W` be the per-tree weight vector and `raw` the cached raw score of every
training row. One round, with the module function that performs each step:

1. **Select the drop set** (`select_drop`). A skip draw fires with
   probability `skip_drop`; if it does, the round drops nothing and is an
   ordinary GBDT round. Otherwise each already-grown iteration is drawn at a
   probability that `max_drop` has already capped, and selection stops at
   `max_drop` entries (section 4.1). The set may legitimately come out
   empty, which normalizes to an ordinary GBDT round.
2. **Uncache the dropped trees** (`dart_begin_round`). Subtract each dropped
   iteration's weighted contribution from `raw`, keeping what was subtracted
   so the round does not have to walk those trees twice.
3. **Grow the new tree.** This is `tree.grow_tree`, unchanged, against
   gradients taken at the dropped-out `raw`. `boosting_dart` grows nothing;
   the caller owns the loop.
4. **Normalize and commit** (`dart_normalization`, `dart_commit_round`).
   Give the new tree its weight, scale the dropped iterations down, put the
   scaled contribution back into `raw`, and append.

`dart_recompute_raw` sums the same quantity directly from the model. The
cache maintained by steps 2 and 4 is supposed to equal it, and that identity
is the cheapest test of the whole module.

## 3. Normalization

With `k` iterations dropped and a learning rate of `lr`, the new tree enters
at `new_weight` and every dropped iteration is multiplied by
`dropped_scale`:

| Mode | `new_weight` | `dropped_scale` |
|---|---|---|
| default (DART paper, LightGBM's default) | `lr / (k + 1)` | `k / (k + 1)` |
| `xgboost_dart_mode=true` | `lr / (k + lr)` | `k / (k + lr)` |
| `k = 0` (skipped or empty ensemble) | `lr` | `1.0` |

The point is that the `k` dropped trees plus the newcomer weigh about what
the `k` dropped trees weighed alone, so the ensemble does not inflate in
scale every time a round drops.

Two properties fall out and are worth stating because other subsystems lean
on them:

- **Every weight is positive.** So a positive combination of trees each
  monotone in a feature stays monotone in that feature, and the claim
  `Booster.monotone` records survives DART. Monotone constraints are allowed
  under DART for exactly this reason.
- **`k = 0` reproduces GBDT bit for bit.** A DART run whose every round skips
  produces the GBDT ensemble, weight for weight. That makes `skip_drop = 1`,
  and equally `drop_rate = 0`, degenerate configurations, which
  `DartParams.validate` rejects rather than silently training plain GBDT
  under DART's name.

> **VERIFIED** against LightGBM `src/boosting/dart.hpp` (master, read
> 2026-08-15). The question an earlier revision of this section left open,
> whether normalization multiplies the shrinkage or replaces it, is settled:
> it multiplies. `DART::DroppingTrees` ends with `shrinkage_rate_ =
> learning_rate / (1.0f + k)`, or `learning_rate / (learning_rate + k)` in
> XGBoost mode with an explicit `k == 0` branch back to `learning_rate`, and
> `DART::Normalize` leaves each dropped iteration at `k / (k + 1)` (or
> `k / (k + learning_rate)`) of what it weighed. The table above is those
> factors.

## 4. Determinism

The drop draw follows the rule the rest of the library follows (`bagging.mojo`,
`goss.mojo`, `sampling.mojo`): splitmix64 over a counter seeded by
`(drop_seed, round)`. A draw depends on its seed and its absolute round index
and never on history, so continued training that resumes at round 40 draws
what an uninterrupted 100-round run would have drawn at round 40. That is the
property `train_more` already promises for bagging and GOSS.

Within a round the stream is laid out so nothing collides: offset 0 is the
skip decision, offset `1 + i` is iteration `i`'s draw. LightGBM draws the
same two things in the same order from one sequential generator.

### 4.1 The selection rule

`select_drop` is `DART::DroppingTrees`, rule for rule. After the skip draw:

1. `max_drop`, when positive, caps the rate before anything is drawn, so a
   round is unlikely to reach the cap rather than merely being truncated at
   it.
2. Iteration `i` is dropped when its draw falls below its probability.
3. Selection stops at `max_drop` entries, in ascending iteration order.

The probability, and the rate cap, depend on `uniform_drop`:

| | `uniform_drop=true` | `uniform_drop=false` (LightGBM's default) |
|---|---|---|
| probability of dropping iteration `i` | `drop_rate` | `drop_rate * w_i / mean(w)` |
| rate cap under `max_drop` | `min(drop_rate, max_drop / n_iterations)` | `min(drop_rate, max_drop * inv_average_weight / sum_weight)` |

`w_i` is what iteration `i` currently weighs, which is the vector the
trainer already carries, so the non-uniform rule costs one argument to
`select_drop` and no new bookkeeping. With every weight equal the two rules
coincide exactly. The non-uniform cap is LightGBM's expression verbatim; it
does not reduce to a probability by dimensions and with realistic weights it
comes out far above 1, so under that rule the hard stop at `max_drop` is
effectively all `max_drop` does. It is reproduced rather than corrected
because matching LightGBM's drop-set sizes is the point.

Three differences from LightGBM, none of which changes what a rule means:

- **The bits.** The draws come from counter-based splitmix64 rather than
  LightGBM's sequential generator, so equal seeds select different sets. This
  is the trade `bagging.mojo` and `goss.mojo` already make, and it is what
  makes offset `1 + i` independent of whether the loop stopped early.
- **`max_drop <= 0` is uncapped.** LightGBM's break tests
  `size() >= size_t(max_drop)`, so its `max_drop = 0` stops after one drop
  and its negative values wrap to no cap at all.
- **No forced drop.** An earlier revision of this module forced an unskipped
  round to drop at least one iteration. LightGBM
  does not, and does not need to: `k = 0` normalizes to exactly
  `learning_rate`, which is an ordinary GBDT round. Forcing raised the
  effective drop rate well above the configured one early in a run, when
  candidates are few.

## 5. Multiclass

`MulticlassBooster` stores trees round-major: `trees[i * n_classes + k]`.
DART drops whole **iterations**, never a single class's tree, because
dropping one class's tree alone would tilt the softmax toward the classes
whose trees survived. So `DartDrop.iterations` holds iteration indices, and
every entry point takes `n_classes` and walks the round-major layout.
`n_classes = 1` is the single-output case, where an iteration is a tree.
LightGBM drops the same way: `DroppingTrees` selects an iteration index and
then loops `cur_tree_id` over `num_tree_per_iteration_`.

`alternate_boosting._dart_rounds_multiclass` is the loop that drives it, and
`train_dart_multiclass`, `train_dart_multiclass_more`, and
`train_dart_multiclass_with_valid` are its three entry points. The round is
the single-output round with the softmax probabilities recomputed from the
dropped-out scores, one bag for the whole round, and one feature draw per
class per round (`round * n_classes + k`). Softmax is not a leaf-renewing
objective, so a multiclass round has no renewal step.

## 6. Early stopping

**A DART ensemble cannot be truncated to its best round.** This is the one
place where DART breaks an assumption the existing trainer is built on, and
it is worth being precise about.

`train_with_valid` finds the best round, then pops trees off the end. For
GBDT that recovers exactly the model that scored best, because the trees it
keeps were never touched by the rounds it discarded. Under DART they were: a
round after the best one may have dropped and rescaled trees that the best
round contained. Popping trees recovers the right tree set and the wrong
weights.

So the weights are snapshotted whenever the validation loss improves
(`DartBestState`, `dart_record_best`), and restoring truncates and then
overwrites the weights (`dart_restore_best`). The cost is one Float64 per
tree per improvement, against the alternative of keeping every round's
vector, which is quadratic in the round count. Restoring is exact because a
tree's node values are not touched until `fold_weights_into_trees` runs, once,
at the end of the fit (section 10): until then the trees are the grower's
output and the round's rescalings live entirely in the weight vector.

`alternate_boosting.train_dart_with_valid` and
`train_dart_multiclass_with_valid` are that loop. Both maintain a second
score cache over the validation rows through the same two functions the
training cache goes through (`dart_begin_round`, then `dart_advance_scores`),
which is what makes the two caches provably the same arithmetic on different
rows.

LightGBM has no counterpart: `DART::EvalAndCheckEarlyStopping` returns false
unconditionally, so `early_stopping_round` is silently inert under
`boosting='dart'` there. These functions train the rounds LightGBM would and
additionally stop on them.

## 7. Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `drop_rate` | 0.1 | Per-iteration drop probability in a dropping round |
| `max_drop` | 50 | Cap on one round's drop set; `<= 0` is uncapped |
| `skip_drop` | 0.5 | Probability a round drops nothing |
| `uniform_drop` | false | Draw every iteration at the same rate rather than in proportion to its weight (section 4.1) |
| `xgboost_dart_mode` | false | Use XGBoost's normalization constant |
| `drop_seed` | 4 | Counter-stream seed, distinct from `bagging_seed` (3) |

## 8. What is refused, and why

Each of these is refused by name, with what it would take, rather than being
downgraded to GBDT or trained as something the parameter names do not
describe. That is the rule `params.mojo` and
`tree_parameters_extra.check_extra_option_supported` already follow.

- **GPU.** `train_gpu` advances device-resident raw scores by one shrinkage
  factor per round and has no path for undoing a dropped tree's contribution
  on the device.
- **Ranking (`lambdarank`).** The pairwise lambdas are computed from each
  query's current score ordering. A dropout round reorders queries by trees
  the round then puts back, so the gradients would not describe the model
  being built.
- **GOSS.** GOSS ranks rows by gradient magnitude, which under DART is the
  gradient of the dropped-out ensemble rather than of the model, so the
  sampled rows would track the dropout draw instead of the residual.

## 9. What is explicitly allowed

Stated so the list above is not read as "DART is refused":

- **Monotone constraints**, for the positive-weight reason in section 3.
- **Interaction constraints, categorical splits, feature subsampling, row
  bagging.** None of them read a tree's weight. They shape how a tree is
  grown; DART only changes what a grown tree weighs.
- **Leaf-renewing objectives** (`mae`, `mape`, `quantile`, `huber`). Renewal
  fits leaf values to residuals of the current raw scores, and under DART the
  current raw scores are the dropped-out ones, which is the residual the new
  tree is actually responsible for. This is a requirement on the caller: it
  must renew against the `raw` that `dart_begin_round` hands back, not
  against the full ensemble's. It is not a reason to refuse.

## 10. Model state

A DART model needs one Float64 weight per tree beyond what a GBDT model
needs. `Booster` carries a single `learning_rate` and
`Booster.predict_raw_row` sums `learning_rate * trees[i].predict_row(...)`,
so a DART ensemble is not representable by it directly.

`alternate_boosting.mojo` reconciles this by folding each tree's weight into
that tree's node values at the end of training and leaving the ensemble's
factor at 1.0, which is what LightGBM's `Tree::Shrinkage` does. That avoids a
serialization format change.

The fold gives up exactly one thing: a tree's weight cannot be read back off
a saved model. That costs continued training two things, and nothing else
anything:

- the rate the earlier rounds used, which `train_dart_more` and
  `train_dart_multiclass_more` therefore take from the caller, checking only
  what a folded ensemble can prove about itself, that its stored factor is
  1.0;
- the ability of `uniform_drop=false` to tell the pre-existing iterations
  apart, since lifting the ensemble back gives every one of them a weight of
  1.0. LightGBM sidesteps this by never dropping an iteration from before the
  continuation: its `tree_weight_` holds only the current session's
  iterations.

Everything else, prediction included, is exact:
`Booster.predict_raw_row` computes `learning_rate * leaf` with a
`learning_rate` of 1.0 and the weight already inside `leaf`, which is the
same product the weighted form would have rounded once.

## 11. Verification status

What has been done: LightGBM's `src/boosting/dart.hpp` was read at the source
level (master, 2026-08-15) and every rule in sections 3 and 4 is a reading of
it, quoted where it matters. Every function named in this document has been
compiled, by a throwaway driver that calls each entry point so that `mojo
build` elaborates its body rather than only its signature.

What has since been run (integration round, 2026-08-15):
`tests/test_alternate_boosting.mojo` trains dart single-output and
multiclass models through `alternate_boosting`, checks them finite and
distinct from gbdt, and round-trips them through the model file;
`python/tests/test_params.py` fits `boosting="dart"` through
`MojoTreesRegressor`. What has not: no benchmark trains a DART model and no
output has been compared against LightGBM's, so the parity row says
`partial` and claims no numeric equality.

The cheapest checks, in the order they are worth writing:

1. `dart_recompute_raw` equals the incrementally maintained `raw` after every
   round. That one identity covers `dart_begin_round`,
   `dart_advance_scores`, `dart_commit_round`, and the weight vector at once.
2. A DART run whose every round skips (`skip_drop` just under 1, checked by
   the drop counts rather than by the parameter) is the GBDT ensemble, tree
   for tree and weight for weight.
3. `select_drop` drop-set sizes against the rate: at `uniform_drop=true` the
   count is binomial in `(n_iterations, drop_rate)` until `max_drop` binds,
   and at `uniform_drop=false` an ensemble with equal weights reproduces the
   uniform counts exactly.
4. `train_dart_with_valid` restores a model whose validation loss is the
   best one it reported, which truncation alone would not.
