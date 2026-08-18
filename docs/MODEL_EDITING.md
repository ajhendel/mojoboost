# Model editing

This is the normative statement of what a fitted mojotrees model may be
changed into, and what it may not. `src/mojotrees/model_editing.mojo` is
the only implementation; everything below is part of the contract, and
anything not below is not offered.

Editing exists because LightGBM offers it: `rollback_one_iter`,
`lower_bound`, `upper_bound`, `get_leaf_output`, `set_leaf_output`,
`shuffle_models`, and `refit`. mojotrees offers the same operations under
one added rule, which is the reason this document exists:

> **An edited model must still load, and must still be a true description
> of itself.** An operation that could leave the model unreadable, or
> leave a claim on the model that has become false, is refused by name
> rather than offered with a warning.

Nothing in this document has been executed. The module is uncompiled and
untested; see `handoffs/remaining_08_model_editing.md (deleted, recover with git log --all --diff-filter=D -- handoffs/remaining_08_model_editing.md)`.

## What a fitted model claims

A `Booster` is not just an array of trees. It carries six claims, and every
operation below is judged by whether it leaves each one true.

| Claim | Where it lives | What would falsify it |
| --- | --- | --- |
| Routing | `feature`, `threshold_bin`, `left`, `right`, `default_left`, `missing_bin`, `cat_offset`, `cat_bitset` | changing any of them |
| Covers | `Tree.count[i]`: training rows that reached node `i` | changing routing, or recomputing covers from different data than the leaf values were fitted from |
| Internal values | `Tree.value[i]` for `feature[i] >= 0`: the value the node held when it was created | writing one |
| Monotone claim | `Booster.monotone`: predictions are monotone in the constrained features | a leaf value outside its node's monotone interval |
| Gains | `Tree.split_gain[i]`: what the split earned from the gradient sums the node held at growth time | writing one; refit and leaf edits make it *stale in meaning*, not false |
| Loadability | `_read_trees` in `src/mojotrees/serialize.mojo` | a zero cover on a tree that reports covers, a feature index past the model's feature count, a missing bin outside the bin range, a category offset that does not start a set |

`check_tree_structure`, `check_tree_serializable`, `monotone_claim_holds`,
and `check_monotone_claim` are the runnable form of that table. Every
mutating entry point checks what it could have broken.

## Operations

### `rollback_one_iter`, `rollback`, `rollback_to`

Drops whole iterations off the end of the ensemble. A single-output model
holds one tree per iteration; a softmax model holds `n_classes` per
iteration and rollback removes whole blocks, because dropping any other
number would shift which class every remaining tree scores.

The **base score is untouched**. It belongs to iteration 0 (see
`IterationRange` in `src/mojotrees/boosting.mojo`), so an ensemble rolled
back to nothing predicts the base score alone, which is exactly what a
zero-iteration ensemble has always predicted. LightGBM reaches the same
place by a different route: it folds its base score into the first
iteration's leaf values rather than storing it apart.

Rollback needs the **boosting mode**, which a fitted ensemble does not
record:

| Mode | Constant | Behavior |
| --- | --- | --- |
| GBDT | `EDIT_MODE_GBDT` (default) | drop the trees, keep the learning rate. The remaining trees contribute exactly what they did |
| Random forest | `EDIT_MODE_RF` | drop the trees **and** rescale the rate to `1 / (K - dropped)`. A forest is an average, so this is the correct forest of the remaining trees, and it changes every remaining tree's contribution |
| DART | `EDIT_MODE_DART` | **refused**. Dropped iterations were rescaled in place when later rounds added trees, so the first `n-1` trees are not the model DART would have produced at round `n-1`. LightGBM refuses this too |
| unknown | `EDIT_MODE_UNKNOWN` | **refused**, with the reason. A model file, a pickle, and an estimator handle all leave a caller here |

The mode is never guessed. `boosting_rf.is_forest` is a structural test
(`learning_rate == 1 / K`), and an ordinary ten-tree model trained at
`learning_rate=0.1` passes it, so nothing reads it as a label. Asking for
`EDIT_MODE_RF` on an ensemble that does not carry the `1 / K` rate is an
error rather than a rescale.

Rolling back an ensemble that holds no iterations is an error, not a no-op:
a caller unwinding a loop is asking for a specific model, and handing back
the base-score model would hide the miscount.

### `lower_bound` and `upper_bound`

`raw_score_bounds` returns both at once as a `ScoreBounds`:

```
lower = base_score + learning_rate * sum over trees of (smallest leaf value)
upper = base_score + learning_rate * sum over trees of (largest  leaf value)
```

**Sound, not tight.** Each tree's extreme leaf is taken independently, and
no row need reach the extreme leaf of every tree at once, so the endpoints
are attainable only by coincidence. That is what LightGBM reports, and it
is the only bound obtainable without searching the joint feasible set of
leaf combinations.

- The base score is inside the interval. LightGBM's is too, by being folded
  into the first iteration's leaves.
- An ensemble with no trees has `lower == upper == base_score`.
- `raw_score_bounds_range` bounds an `IterationRange`, adding the base score
  only when the range starts at 0, exactly as `predict_raw_bins_range` does.
- `response_bounds` maps the endpoints through `Booster.response`. Every
  inverse link in the objective registry is nondecreasing (sigmoid for
  logistic and cross entropy, `exp` for poisson, gamma, and tweedie, the
  identity otherwise), so the interval maps to the interval. A CUSTOM model
  returns its raw bounds, because the framework does not know the caller's
  link.

For a **softmax** model there is no single bound.
`raw_score_bounds_multiclass` returns one `ScoreBounds` per class, because
class `k`'s score is the sum over rounds of `trees[i * n_classes + k]` and
nothing else. Summing every tree's extremes across all classes, which is
what a literal reading of LightGBM's single-value bound would do, adds up
numbers that never enter the same score and bounds nothing.

`probability_bounds_multiclass` is offered as well, and is derived rather
than assumed:

```
p_k = exp(z_k) / sum_j exp(z_j)
```

is increasing in `z_k` and decreasing in every other `z_j`, so over the box
the per-class raw bounds describe it is largest at `z_k = hi_k` with every
other class at its `lo`, and smallest at the opposite corner. Both are
computed there, with the exponentials shifted by their maximum. The result
is loose twice over (the raw bounds are not attained, and the box is larger
than the set of score vectors one row can produce), and it is a guarantee
about what the model cannot output rather than a description of what it
does.

### `get_leaf_output` and `set_leaf_output`

Two differences from LightGBM that a caller has to know:

1. **Leaf addressing.** Leaves are named by mojotrees's *leaf ordinal*: a
   leaf's rank among the tree's leaves in node order, which is what
   `predict(pred_leaf=True)` reports and what `Tree.leaf_ordinals`
   documents. It is not LightGBM's leaf id, and the two agree only by
   coincidence.
2. **Value scale.** mojotrees stores the value a leaf was fitted at and
   multiplies by `learning_rate` when it predicts; the dump reports the rate
   as `shrinkage` and states `leaf_value_is_shrunk: false`. LightGBM folds
   shrinkage into the stored value. So a number read here is the LightGBM
   number divided by the learning rate. `get_leaf_output_shrunk` and
   `leaf_outputs_shrunk` give the contribution scale.

A leaf write is safe because of what it does **not** touch. Routing is
unchanged, so every row still reaches the leaf it reached; covers therefore
remain exactly as true as they were, and exact feature contributions
(`contrib.mojo`, which reads leaf values and covers and nothing else) stay
correct for the edited model. Internal node values are not writable at all:
`set_leaf_output` refuses a non-leaf node, since an internal value is what
the monotone interval chain and the dump's `internal_value` are derived
from.

Two things a write does change:

- **Split gains go stale in meaning.** A gain still truthfully describes
  what the split earned when the tree was grown; it no longer describes the
  model in hand. That is LightGBM's behavior as well.
  `clear_split_gains_booster` is the explicit retraction: it zeroes them,
  which makes `has_split_gains` report `false`, which is the schema's way
  of saying "no gains to report" rather than reporting a zero a consumer
  could read as a measurement. Prediction is unaffected either way.
- **Feature importance.** Split-count importance is structural and does not
  move. Gain importance does not move arithmetically, and goes stale with
  the gains. Any cached importance at the Python layer must be dropped.

**Monotonic constraints.** If the ensemble records constraints, an
arbitrary leaf value can falsify them, and a model that claims a
monotonicity it does not have is exactly the inconsistency this module
exists to prevent. Every write therefore:

1. clamps the value into the leaf's interval, derived by `node_bounds` from
   the internal node values *before* the write, or refuses it outright
   under `LEAF_EDIT_REJECT`;
2. writes;
3. re-derives the whole interval chain and verifies **every** leaf, because
   a leaf's value sets the midpoint its parent divides at, so moving one
   leaf moves its sibling subtree's intervals;
4. restores the previous value and raises if step 3 fails.

The return value is what was actually stored, which is the only way a
caller learns that a clamp happened. `set_leaf_outputs` rewrites a whole
tree in ordinal order, which makes the sequence of interval shifts explicit
and reproducible instead of a property of the caller's loop.

Non-finite values are refused. A NaN or infinite leaf value round-trips
through the model file (floats travel as raw bit patterns), poisons every
prediction that reaches the leaf, and can only be reported as `null` by the
JSON dump.

### `shuffle_models`

`shuffle_iterations` permutes whole iterations in an `IterationRange`;
`permute_iterations` takes the permutation explicitly and is what the
shuffle draws for. The draw is the counter-based splitmix64 stream the row
and feature samplers use, in its own domain, so the same seed and range
give the same order on any platform and a shuffle seed cannot reproduce a
bagging draw.

For a **softmax** model, `shuffle_iterations_multiclass` permutes whole
per-class blocks and never reorders within one. This is the operation, not
an optimization of it: `trees[i * n_classes + k]` is what makes a tree
class `k`'s, so a permutation that moved individual trees would leave a
structurally valid ensemble in which every class scores some other class's
trees, with nothing downstream able to detect it.

What shuffling costs:

- **Floating point.** Prediction sums the trees, and a sum does not care
  about order in exact arithmetic. Floating-point addition is not
  associative, so a shuffled model's predictions can differ from the
  original's in the last ulp.
- **Iteration meaning.** An iteration index no longer names the round that
  grew that tree, so `best_iteration` and any per-iteration history are
  stale, and `predict(..., num_iteration=k)` no longer means "the first `k`
  rounds".

Neither is a model inconsistency, which is why the operation is offered;
both are caller state, which is why the Python layer must invalidate its
metadata (see the handoff).

### `refit`

Rebuilds every leaf value from new data while keeping every tree's shape.

The ensemble is walked in training order. At tree `t` the gradients are
taken at the raw scores the already-refit trees `0..t-1` produce on the
refit data, tree `t`'s leaves are rebuilt from the rows that reach them,
and the raw scores are advanced by the new tree. That is the sequence
training itself follows, which is what makes a refit at `decay_rate = 0`
the model this structure would have been fitted with. A softmax refit takes
the class probabilities once per round from the current raw scores and then
advances each class in turn, which is what `_boost_rounds_multiclass` does.

The leaf rule is the grower's own, not a second formula: the
L1-soft-thresholded Newton step `-T(G) / (H + lambda_l2)`, then
`max_delta_step` and `path_smooth` through `finish_leaf_output`. For
QUANTILE, L1, and MAPE, whose leaves are renewed rather than fitted by a
Newton step, refit calls the trainer's own `_renew_leaf_values` instead;
refitting those by the Newton rule would fit a different model from the one
training produced.

`decay_rate` is LightGBM's `refit_decay_rate` (default 0.9):

```
new = decay_rate * old + (1 - decay_rate) * fresh
```

Both values are on the unshrunk scale. Shrinkage is a single positive
factor shared by every tree, so blending unshrunk values is the same blend
LightGBM performs on shrunk ones.

| Refit changes | Refit does not change |
| --- | --- |
| leaf values | routing |
| node covers, all-or-nothing per tree | internal node values |
| | split gains |
| | base score, learning rate, objective |
| | the monotone claim |
| | the iteration count, or the class-to-tree mapping |

**Covers are recomputed all-or-nothing**, per tree. Covers from two
datasets in one tree would make exact feature contributions condition on a
background that never existed, and one zero cover produces a model the
reader refuses to load. A tree in which some node draws no refit row
therefore keeps every one of its original covers, and the returned
`RefitReport` says how many trees were recounted.

**Empty leaves keep their old value.** `RefitParams.min_leaf_rows` defaults
to 1, which only expresses that a leaf no row reaches has nothing to be
refit from; LightGBM applies no floor at all. Raising it trades
faithfulness to the new data for stability.

**Monotone constraints survive without a second clamp.** The intervals are
derived once, before any leaf moves, from internal node values a refit
never writes. The old value already lies in its interval (refit refuses to
operate on a model that does not satisfy its own claim), and the fresh
value is clamped into it, so every convex combination of the two lies in it
as well.

`refit` is **refused** for two objectives:

- **CUSTOM.** The gradients come from a caller-supplied callable that the
  fitted ensemble does not carry.
- **LAMBDARANK.** The gradients are computed within a query group, which a
  `(data, target)` pair does not describe.

Both would have to guess, and a guessed gradient produces leaf values that
look ordinary and are wrong.

`refit_dataset` and `refit_dataset_multiclass` are the production entry
points. They take a `Dataset`, so the label, the weights, and the init
score come off it rather than being passed again, and they enforce the same
two rules `update_dataset` does: the dataset must be binned by the mapper
the model was trained under (or a bin index means one thing to the trees
and another to the rows), and a sparse dataset is refused rather than
densified.

`init_score` is the offset the ensemble was trained under, if any. It is
training state that the model does not carry (a booster trained from an
init score records a base score of 0), so a refit has to be handed it again
for the same reason `train_more` does. It has no multiclass form: one
offset per row cannot say what each class starts from.

## Operations that are refused

`editing_capabilities()` is the single table, and
`model_editing_status_json()` renders it. Each refusal names the invariant
that makes it a refusal rather than an omission.

| Operation | Why not |
| --- | --- |
| `set_internal_value` | an internal node's value is the value it held when it was created; the monotone interval chain and the dump's `internal_value` are derived from it, and nothing could tell an edit from a corruption |
| `set_split_feature` | changing routing falsifies every node cover below the node, which exact feature contributions condition on, and no recomputation is possible without the training rows |
| `set_threshold` | same as `set_split_feature` |
| `set_split_gain` | a gain is what a split earned from gradient sums the tree no longer holds; it can be retracted with `clear_split_gains`, not rewritten |
| `set_node_count` | a cover is a training row count, not a knob; `refit` recomputes covers from data, all-or-nothing per tree |
| `add_tree` | an appended tree has no cover, no gain, and no fitted leaf values; `train_more` is the supported way to add iterations |
| `set_learning_rate` | one rate shrinks every tree, so changing it rescales the whole ensemble at once. The only rate change offered is the `1 / K` rescale a forest rollback performs |
| `set_objective` | the objective selects the inverse link and the gradient family the leaf values were fitted under; changing it leaves every leaf value meaningless |
| `set_base_score` | the base score is the objective's own starting point for the fit the trees corrected; shifting it afterwards moves every prediction by a constant the trees never saw |

## Serialization

Every operation here writes only fields the model format already carries,
so no format version is required for editing itself:

| Field | Written by | Format version |
| --- | --- | --- |
| `Tree.value` | leaf edits, refit | v1 |
| `Tree.count` | refit, when recounted | v3, behind a presence flag as of v4 |
| `Tree.split_gain` | `clear_split_gains` | v4, behind a presence flag |
| tree count | rollback | v1 |
| tree order | shuffling | v1 |
| `learning_rate` | forest rollback only | v1 |

What the format cannot carry is **provenance**: a saved model does not say
that it was edited, refit, rolled back, or shuffled. A consumer reading a
refit model's split gains has no way to learn they describe an earlier fit.
The handoff proposes an optional `edited` section for a later format
version; until then, provenance is the caller's to record.

`check_tree_serializable` restates the reader's own admission rules and is
checked by the mutating entry points, so an edit cannot produce a model
this build can hold but not read back. It also catches a case the writer
and reader disagree on today, independent of editing: the writer decides
whether to write covers from `Tree.has_node_counts`, which looks at the
root cover alone, while the reader requires *every* cover to be positive.

## Interaction with other model state

| State | Effect |
| --- | --- |
| Early stopping (`best_iteration_`, the metric history) | rollback and shuffling invalidate it outright; refit leaves the iteration count alone but makes the recorded best *score* stale. It lives in the Python layer and must be cleared there |
| Continued training (`train_more`, `update_dataset`) | stays valid after every operation here: it recomputes the raw scores from whatever trees the ensemble holds, so it resumes from the edited model. What is no longer true is the claim that "40 rounds then 60 more equals 100 in one call", which is about an unedited ensemble |
| Prediction slicing (`IterationRange`) | a range built before a rollback names iterations that no longer exist. `IterationRange.slice` clamps silently, which is right at the prediction boundary and wrong after an edit, so `clamp_range` raises instead |
| Feature importance | split-count importance is structural and survives everything but rollback; gain importance survives arithmetically and goes stale in meaning after a refit or a leaf edit |
| Model dump / inspection | every fact the dump reports is read from the model, so a dump taken after an edit describes the edited model. `has_split_gain` and `has_node_count` remain the two capability flags a consumer must branch on |
| GPU-trained models | unaffected. Device is not a property of a fitted model, and every operation here is host-side arithmetic on the tree arrays |

## Determinism

- Shuffling is seeded and reproducible across platforms.
- Refit is deterministic: no sampler runs, and every sum is accumulated in
  row order.
- Leaf edits and rollback perform no arithmetic that could vary.
