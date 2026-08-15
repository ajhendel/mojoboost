# Cost-effective gradient boosting (CEGB)

Status: **implemented and reachable.** `src/mojotrees/cegb.mojo` is the
authoritative implementation of all four of LightGBM's `cegb_*` controls, and
it is the only one: `tree_parameters_extra.FeaturePenalties` holds a
`CegbConfig` and applies the `feature_contri` multiplier alone. The dense CPU
grower (`tree.grow_tree_with_cegb`) carries the ledger, `boosting.fit` and
`boosting.fit_multiclass` own it for the whole ensemble, and every other
grower charges the split cost and *refuses* the two penalties that need the
ledger. `cegb_tradeoff` and `cegb_penalty_split` are reachable from a
parameter string; the two per-feature vectors are Mojo-API-only, for the same
reason `monotone_constraints` is. Section 12 is the precise reachability
table.

Everything below is derived by reading the code, and the LightGBM claims from
reading LightGBM master. Nothing here has been run.

## 1. What CEGB is

Ordinary gradient boosting picks the split with the highest gain. CEGB picks
the split with the highest gain **net of what that split will cost to
evaluate**. The cost is a property of the data access a split forces, and it
is charged in the same units as the gain, so the two are directly comparable.

The point is a model that is cheap to serve. A feature that costs a lot to
compute and buys a little accuracy stops winning splits, so it never enters
the model and never has to be computed at prediction time. LightGBM's paper
name for this is cost-effective gradient boosting; the four parameters are
`cegb_tradeoff`, `cegb_penalty_split`, `cegb_penalty_feature_coupled`, and
`cegb_penalty_feature_lazy`.

## 2. The four parameters

| Parameter | Default | Charged | Needs |
| --- | --- | --- | --- |
| `cegb_tradeoff` | 1.0 | multiplies every term below | nothing |
| `cegb_penalty_split` | 0.0 | per split, per active row in the node | the node's row count |
| `cegb_penalty_feature_coupled` | empty | once, at a feature's first split anywhere in the ensemble | a per-feature ledger, ensemble lifetime |
| `cegb_penalty_feature_lazy` | empty | per active row in the node that has not yet read this feature | a per-(feature, row) ledger, ensemble lifetime |

`cegb_tradeoff = 0.0` disables the whole mechanism whatever the other three
say. An empty or all-zero vector charges nothing however long it is, which is
why `CegbConfig.coupled_active` and `lazy_active` test the values and not the
length.

Every value must be finite and nonnegative. LightGBM does not check; a
negative cost is a bonus for reading data, which inverts the mechanism and
can make a chosen split's adjusted gain exceed its raw gain. `CegbConfig.check`
rejects it. That is an intentional difference and it is listed again in
section 9.

## 3. The gain adjustment

For a node `N` whose active rows are `R(N)`, and a dataset feature `f`:

```
gain_cegb(f, N) = contri[f] * gain_raw(f, N) - delta(f, N)

delta(f, N) = tradeoff * ( penalty_split * |R(N)|
                         + coupled[f] * [f has never been split on]
                         + lazy[f]    * unread(f, R(N)) )

unread(f, R) = |{ r in R : feature f has never been read for row r }|
```

`gain_raw` is the ordinary second-order split gain, `contri[f]` is LightGBM's
`feature_contri` multiplier, and `[P]` is 1 when `P` holds and 0 otherwise.

Three facts about that formula carry the whole design.

**The multiplier scales, the costs subtract.** `contri[f]` multiplies the raw
gain; the three CEGB terms are then subtracted as absolute amounts of gain.
They are not rescaled by the multiplier. A caller who sets both gets
LightGBM's composition rather than a product of the two mechanisms.

**`contri` is applied exactly once, and not by cegb.mojo.**
`tree_parameters_extra.FeaturePenalties.contri_of` is the multiplier's only
owner and `penalized_gain` multiplies without subtracting anything.
`cegb_adjusted_gain` and `CegbNodeCosts.adjusted_gain` take a gain that has
already been multiplied and only subtract. Double-applying `feature_contri`,
or the split cost, is the one way these two mechanisms can silently corrupt
each other, which is why the two live in separate calls one line apart in
`split._feature_gain` rather than in one function that does both.

**Only the split term is a pure function of the node.** The coupled term reads
a flag that lives for the whole ensemble, and the lazy term reads a
per-(feature, row) flag that does too. That division is why one of the four
parameters is live in the split search today and three are not.

## 4. Where each term is charged

Once per feature, after that feature's scan and before its best candidate is
compared against the running best. `split._feature_gain` is already that
point, for `feature_contri`, the monotone penalty, and `min_gain_to_split`.

That placement is not cosmetic. The split cost is a property of the node, and
the coupled and lazy costs are properties of the feature, so charging them per
*candidate* would multiply a per-feature cost by however many thresholds the
feature happens to offer. A feature with 255 bins would be charged 255 times
for a cost it incurs once.

`CegbNodeCosts` exists so that placement stays cheap. It costs one node's
features once, before the scan, and the scan then charges a candidate with one
lookup and one subtraction. Without it, the lazy term would walk the node's
rows once per candidate and split search would become quadratic in the leaf
size.

## 5. The ledger

`CegbLedger` holds the two flags the node does not carry:

- `feature_used[f]`: feature `f` has been split on somewhere.
- `row_read`, a bitset over (feature, global row id): row `r` has had feature
  `f` read for it, because `r` passed through a node that split on `f`.

Both live for the **whole ensemble**, not for one tree. That is CEGB's
premise: the model pays for a feature once, so a second tree reusing a feature
the first tree already needed gets it free. A ledger reset per tree would
charge the first-use cost once per tree, which is a much harsher and quite
different regularizer.

`cegb_commit_split` is the ledger's only writer, and it runs after a split has
been **chosen**, never during a scan. Two consequences: a parallel scan can
never observe a half-updated ledger, and re-scoring a node gives the same
answer it gave the first time.

### 5.1 The cached-candidate refund

Leaf-wise growth scores each leaf's best split once and keeps it in the
frontier (`tree._LeafState.split`) until that leaf is chosen. A cached
candidate on feature `f` that was charged `f`'s first-use cost goes stale the
moment some *other* leaf's split uses `f` first: the model now computes `f`
anyway, so the cached candidate's real gain is higher by
`tradeoff * coupled[f]`.

`cegb_stale_cached_gain` returns that amount. Adding it to the cached gain is
correct and re-running the scan is not necessary, because the split term and
the lazy term are properties of the cached candidate's own node and are
unchanged by a split somewhere else.

**Checked against LightGBM master, and one difference remains.**
`CostEfficientGradientBoosting::UpdateLeafBestSplits` walks every leaf but the
one just split, adds `cegb_tradeoff * cegb_penalty_feature_coupled[f]` to that
leaf's cached candidate *on the newly used feature*, and installs it as the
leaf's best split when it now beats it. The amount and the "only candidates on
`f`" condition are what `cegb_stale_cached_gain` implements.

What LightGBM can do and mojotrees cannot: it keeps a candidate per (leaf,
feature) in `splits_per_leaf_`, sized `num_leaves * num_features` and cleared
per tree, so a leaf whose best split is on some other feature can still have
its runner-up on `f` promoted once `f` is free. mojotrees's frontier caches one
candidate per leaf, so a refund can improve a cached best but cannot resurrect
a candidate that was not it. On the leaves where the two can differ, mojotrees
keeps a split whose adjusted gain is no lower than the one LightGBM would have
promoted only when that promotion would not have won; where it would have won,
the two trees diverge. Closing it means carrying the per-(leaf, feature) table,
which is `num_leaves * num_features` split records against the one this grower
keeps. Recorded in section 10 as a difference, not an open question.

### 5.2 Continued training

The ledger is training state, not model state. It is not in the serialized
model, and it should not be: a saved model records the trees the ledger
produced, not the ledger. Scoring, `save_model`, and `load_model` are all
unaffected.

`boosting.fit_more` after a round trip through disk is affected. A resumed run
would start from an empty ledger, recharge every first-use cost, treat every
row as never having read anything, and grow a different ensemble from the one
an uninterrupted run would have grown. `check_cegb_continued_training` refuses
that combination rather than letting it diverge silently. The split cost alone
survives a round trip untouched, since it reads no ledger.

The handoff carries the serialization request that would lift the refusal: a
format version bump adding `feature_used` (`n_features` bits) and, when the
lazy penalty is configured, the row bitset (`n_features * n_rows` bits, which
is 12.5 MB for 100 features over 10 million rows and is why it is opt-in).

## 6. Which feature a split is charged to

CEGB's vectors and its ledger are indexed by **dataset feature id**. The split
search does not always see that id.

- **No bundling, and every categorical feature.** Categorical splits are
  searched as category partitions over one column, so the search space is the
  dataset and the mapping is the identity.
- **Exclusive feature bundling on.** A scanned "feature" is a bundle, and a
  threshold bin belongs to one member of it. `efb.decode_feature` recovers the
  member, and the member is what actually gets read at prediction time, so the
  member is what CEGB charges. Charging the bundle would make one sparse
  feature's first use pay for every feature bundled with it, which would
  couple penalties across features that have nothing to do with each other.

`efb.FeatureBundling.charged_feature` performs that recovery. It lives on the
bundling plan rather than in cegb.mojo because it is a pure bundling query and
because cegb.mojo imports nothing from the package: `binning` and `categorical`
both import `tree_parameters_extra`, which now imports `cegb`, so an import in
the other direction would close a cycle. A multi-member bundle's shared bin
belongs to every member at once and cannot be attributed to one feature, so it
is refused rather than charged to an arbitrary member. A validated bundling
plan never puts a threshold there, since the shared bin is where every member's
default value lands, so this is a guard on an inconsistent plan and not a case
a caller has to handle.

No grower calls it today, and none has to: `tree.grow_tree` expands a bundled
histogram back to one slice per original feature before searching it
(`_hist_full`) and partitions rows by the original matrix, so the feature a
split names there is already a dataset feature. The function is what a grower
that searched bundles directly would need.

## 7. Active rows: bagging and GOSS

`|R(N)|` is the node's **active** row count, which under row bagging or GOSS
is the sampled count and not the dataset's. That matches LightGBM, whose data
partition is built over the bagged index set. Two consequences a caller should
know before tuning:

- A given `cegb_penalty_split` buys a different amount of regularization at
  `bagging_fraction = 0.5` than at 1.0. The charge halves; the gains, computed
  from the same sampled gradients, do not. CEGB's costs are absolute
  quantities of gain and do not renormalize themselves. cegb.mojo does not
  rescale them either. Silently dividing by the bagging fraction would be a
  different penalty from LightGBM's, and adding a knob to correct another knob
  is worse than saying so.
- Under GOSS the `other` rows' gradients are already multiplied by
  `GossSelection.multiplier`, so gains are on a full-data scale while `|R(N)|`
  is on a sampled scale. That mismatch is GOSS's, not CEGB's, and it is
  recorded rather than corrected.

The lazy ledger is indexed by **global** row id for the same reason. A row
that sits out one round's bag keeps whatever read-state it had, so the next
round in which it is sampled charges for it correctly. Every row list the
module accepts already holds global ids: `grow_tree`'s `bag`, and the child
lists `partition_rows_into` derives from it.

## 8. Backend support

| Path | split | coupled | lazy | Where |
| --- | --- | --- | --- | --- |
| Dense CPU grower (`tree.grow_tree_with_cegb`) | applied | applied | applied | `tree._search` -> `split.find_best_split` |
| Dense CPU grower (`tree.grow_tree`) | applied | refused | refused | inert ledger, `check_cegb_grower_support` |
| Sparse grower (`tree_sparse.grow_tree_sparse`) | applied | refused | refused | same `_search` |
| GPU, host split search | applied | refused | refused | same `_search` |
| GPU, device split search | refused | refused | refused | `check_cegb_device_split_search` |
| Distributed (`distributed.grow_tree_distributed`) | applied | refused | refused | `find_best_split`'s guard |

`grow_tree` and `grow_tree_with_cegb` are the same grower; the first calls the
second with `CegbLedger.none()`. A `mut` argument cannot be defaulted, which is
why the ledger-carrying form is a second entry point rather than a defaulted
parameter, and the split is useful in its own right: the sixteen `grow_tree`
call sites across the trainers are unchanged and opt in one at a time.

Which trainers opt in today: `boosting.train`/`train_with_valid` (through
`_boost_rounds`) and `boosting.train_multiclass`/`train_multiclass_with_valid`
(through `_boost_rounds_multiclass`). The ranking, random-forest, DART,
custom-objective, and sparse trainers keep the split cost and refuse the other
two, which is the honest state rather than a silent zero.

Three refusals, each for a specific reason:

- **A grower with no ledger.** The coupled cost needs a per-feature flag
  threaded through every tree of the ensemble, and the lazy cost needs the
  node's row ids as well. Both are refused rather than charged as zero:
  `CegbLedger`'s inert form answers "already paid" to every question, which is
  the right answer when nothing is configured and the wrong one when something
  is, so `prepare_cegb_node` checks each half of the ledger against the half of
  the configuration that reads it.
- **GPU device split search.** Candidates are ranked inside the kernel and the
  host downloads one record per node (`train_gpu._search_leaf_device`). A
  host-side cost applied after the winner is chosen ranks nothing. The lazy
  penalty is further out of reach still, since its bitset would have to be
  device-resident and updated from the row-compaction pass.
- **Distributed, lazy specifically.** `unread(f, R(N))` counts rows, and the
  rows are sharded. The true count is a sum over ranks, which needs one integer
  allreduce per (node, feature) scanned. That message does not exist in
  `distributed_transport`, and a rank counting only its own shard could rank
  features differently from its peers, which is a split disagreement and not
  merely a wrong penalty. The split cost is safe there (every rank scores from
  the reduced histogram and the exact global row count), and the coupled cost
  would be safe too once that grower carried a ledger, because every rank
  commits the same chosen split and so flips the same flag at the same moment
  with no message. `check_cegb_distributed` states that division.
- **Continued training from a loaded model.** Section 5.2.

`check_cegb_grower_support(config, carries_ledger, carries_rows)` is the
general form, and it mirrors `tree._search`'s existing `grower_applies_extra`
refusal. The repository's rule is that a backend which cannot apply a setting
says so rather than ignoring it. It is asked in three places: once per tree in
`grow_tree_with_cegb`, so a trainer that threaded no ledger is told before the
first histogram; in `tree._search`, which is what refuses the sparse and GPU
host callers; and in `split.find_best_split`, which catches every remaining
direct caller including the distributed grower.

## 9. Determinism

Every quantity is either an integer count or one multiplication of three
Float64 values in a fixed order, from state written only at commit time.
Nothing depends on scan order, thread count, or how many nodes were scored
first.

One intentional numerical difference. `unread` is counted as an integer and
multiplied once: `lazy[f] * count`. LightGBM accumulates `lazy[f]` once per
unread row, which for a large leaf is a different floating-point number.
mojotrees's is the exact one. This must be confirmed before the parity table
calls the lazy penalty bit-comparable with LightGBM.

## 10. Intentional differences from LightGBM

1. **Negative costs are rejected rather than accepted** (section 2). LightGBM
   has no check; a negative cost is a bonus for reading data, which inverts
   the mechanism.
2. **The lazy term is `lazy[f] * count`**, one multiplication, rather than a
   `count`-long accumulation (section 9). LightGBM's `CalculateOndemandCosts`
   adds `penalty` once per unread row inside its loop, which for a large leaf
   is a different floating-point number. mojotrees's is the exact one, so the
   lazy penalty is not bit-comparable with LightGBM's and the parity table
   must not claim it is.
3. **The cached-candidate refund cannot promote a runner-up** (section 5.1).
   LightGBM keeps a candidate per (leaf, feature); mojotrees keeps one per
   leaf. The refund amount and the condition match; what is missing is the
   promotion of a candidate that was not the leaf's best.
4. **The coupled and lazy penalties are refused per grower, not per name.**
   `check_extra_option_supported` no longer lists them, because they are
   implemented; `check_cegb_grower_support` refuses them for a backend that
   cannot carry the ledger, which is the repository's rule for a setting a
   backend cannot apply.
5. **One ledger across all classes in a multiclass fit**, not one per class
   (section 8). A feature computed for class 0's tree is computed for the row,
   so charging it again per class would make one feature cost `n_classes`
   times what deploying it costs. This follows from the cost being a property
   of the served model; it has **not** been compared against LightGBM's
   multiclass path.
6. **Resumed training is refused** rather than silently recharging
   (section 5.2).

## 11. What is left

1. **The per-(leaf, feature) candidate table**, which is what closes
   difference 3. It is `num_leaves * num_features` split records carried
   through the frontier, and it changes which tree comes out on the leaves
   where a runner-up would have been promoted. Worth measuring before
   building: the promotion can only fire on a leaf whose cached best is on a
   different feature from the one just committed *and* whose runner-up on the
   committed feature beats it once refunded.
2. **The Python estimator parameters.** `cegb_penalty_feature_coupled` and
   `cegb_penalty_feature_lazy` are per-feature vectors and the sklearn
   estimator has no parameter carrying one for them; the binding reads both
   buffer addresses already (`_penalties` in `bindings/basic_bindings.mojo`)
   and `sklearn.py` sends 0 for each. Adding them is two constructor
   parameters, two validated buffers on the pattern of
   `_feature_contri_buffer`, and a regenerated `compatibility/api_snapshot.json`.
3. **Serialization, only if resumed training must work.** `feature_used` is
   `n_features` bits and is cheap; the lazy read bitset is
   `n_features * n_rows` bits (12.5 MB for 100 features over 10 million rows)
   and is meaningless against a different dataset, so it must be written only
   when the lazy penalty is configured and must refuse to load against a
   different row count. That is training state in a model file, which is a
   design question rather than a mechanical addition; a separate resume file
   keeps the model format clean and is the recommendation. Either way
   `check_cegb_continued_training` should be relaxed to accept a restored
   ledger, not deleted.
4. **The ledger on the remaining growers.** `tree_sparse.grow_tree_sparse`
   materializes node row lists exactly as the dense grower does, so threading
   a ledger there is the same three edits. The ranking, random-forest, DART,
   and custom-objective trainers each need only to own a ledger and call
   `grow_tree_with_cegb`.
5. **A differential test against LightGBM**, which is the only thing that can
   settle differences 3 and 5.

## 12. Reachability, precisely

| Claim | True today? |
| --- | --- |
| Implemented | yes, `src/mojotrees/cegb.mojo`, the only implementation |
| Imported by a grower | yes, `tree.mojo` and `split.mojo` |
| Called on a real split decision | yes, `split._feature_gain` on every scanned feature |
| Ledger carried across an ensemble | yes, `boosting._boost_rounds` and `_boost_rounds_multiclass` |
| Publicly reachable | `cegb_tradeoff` and `cegb_penalty_split` from a parameter string, the C ABI, and the Python estimator; both vectors from the Mojo API and the binding |
| Focused-tested | `tests/test_cegb.mojo`, wired into `pixi run test` and `test-cpu`. **Written, not run** |
| Differential-tested against LightGBM | no |
| Hardware-validated | not applicable; CPU arithmetic only |

`docs/LIGHTGBM_PARITY.md`'s four `cegb_*` rows are updated with this file as
their reference.
