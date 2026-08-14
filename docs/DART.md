# DART in mojoboost

DART is LightGBM's `boosting='dart'`: Dropouts meet Multiple Additive
Regression Trees (Rashmi and Gilad-Bachrach, 2015). This document specifies
what mojoboost implements, what it deliberately refuses, and what has not
been verified. It describes `src/mojoboost/boosting_dart.mojo`, which is the
algorithm core and is not a public API.

Status, in one line: the core is written and is imported by
`src/mojoboost/alternate_boosting.mojo`; no DART code has been compiled or
run, and `boosting='dart'` is still rejected by the Python layer and by
`parse_params`. See `handoffs/remaining_01_dart.md`.

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
   ordinary GBDT round. Otherwise each already-grown iteration is a candidate
   with probability `drop_rate`, the set is capped at `max_drop`, and it is
   forced non-empty.
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
  produces the GBDT ensemble, weight for weight. That makes `skip_drop=1`
  with `drop_rate=0` a degenerate configuration, which `DartParams.validate`
  rejects rather than silently training plain GBDT under DART's name.

> **UNVERIFIED.** Whether LightGBM's factor is `lr / (k + 1)` or `1 / (k + 1)`
> (that is, whether normalization multiplies the shrinkage or replaces it) has
> not been checked against LightGBM's `dart.hpp`. The table is what the DART
> paper implies once shrinkage is applied. It is isolated in
> `dart_normalization` so that settling the question is a one-line change. Do
> not claim LightGBM parity for DART on the strength of this table.

## 4. Determinism

The drop draw follows the rule the rest of the library follows (`bagging.mojo`,
`goss.mojo`, `sampling.mojo`): splitmix64 over a counter seeded by
`(drop_seed, round)`. A draw depends on its seed and its absolute round index
and never on history, so continued training that resumes at round 40 draws
what an uninterrupted 100-round run would have drawn at round 40. That is the
property `train_more` already promises for bagging and GOSS.

Within a round the stream is laid out so nothing collides: offset 0 is the
skip decision, offset `1 + i` is iteration `i`'s draw.

Two rules are mojoboost's own, not reproductions of LightGBM's order, and are
called out as differences rather than parity:

- **Cap by smallest draw.** When more iterations are selected than `max_drop`
  allows, the `max_drop` with the smallest draws are kept. LightGBM truncates
  a shuffled list. Ranking by the same draws that made the selection needs no
  second random stream and is reproducible from the seed alone.
- **Forced non-empty.** An unskipped round with no candidate drops the single
  iteration with the smallest draw. A dropout round that drops nothing is
  indistinguishable from a skipped one, which would make `skip_drop` mean two
  different things at once.

## 5. Multiclass

`MulticlassBooster` stores trees round-major: `trees[i * n_classes + k]`.
DART drops whole **iterations**, never a single class's tree, because
dropping one class's tree alone would tilt the softmax toward the classes
whose trees survived. So `DartDrop.iterations` holds iteration indices, and
every entry point takes `n_classes` and walks the round-major layout.
`n_classes = 1` is the single-output case, where an iteration is a tree.

The core is written for multiclass throughout. No multiclass DART loop is
connected.

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
vector, which is quadratic in the round count.

No `train_dart_with_valid` exists. The pieces are provided; the loop is not
connected.

## 7. Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `drop_rate` | 0.1 | Per-iteration drop probability in a dropping round |
| `max_drop` | 50 | Cap on one round's drop set; `<= 0` is uncapped |
| `skip_drop` | 0.5 | Probability a round drops nothing |
| `uniform_drop` | see below | Select the drop set by an independent draw per iteration |
| `xgboost_dart_mode` | false | Use XGBoost's normalization constant |
| `drop_seed` | 4 | Counter-stream seed, distinct from `bagging_seed` (3) |

## 8. What is refused, and why

Each of these is refused by name, with what it would take, rather than being
downgraded to GBDT or trained as something the parameter names do not
describe. That is the rule `params.mojo` and
`tree_parameters_extra.check_extra_option_supported` already follow.

- **`uniform_drop=false`**, which is LightGBM's default. LightGBM selects the
  drop set by a non-uniform rule this module has not reproduced, and
  approximating it with the uniform draw would train a different model than
  the parameter names promise. `DartParams.disabled()` therefore defaults
  `uniform_drop` to true, which is a deliberate inversion of LightGBM's
  default and is the one parameter default that does not match LightGBM.
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
serialization format change. Its consequences, including the one place the
weight has to be readable back off a model, are recorded in
`handoffs/remaining_01_dart.md` section 4.

## 11. Verification status

Nothing in this document has been executed. No Mojo was compiled, no test was
written or run, no benchmark was taken, and LightGBM was not consulted at the
source level. Every claim here is a claim about code as written, not about
code as observed. The checks that would change that are listed, marked
`UNRUN`, in `handoffs/remaining_01_dart.md` section 7.
