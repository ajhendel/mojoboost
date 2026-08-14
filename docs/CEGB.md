# Cost-effective gradient boosting (CEGB)

Status: **implemented, not yet reachable.** `src/mojoboost/cegb.mojo` is the
authoritative implementation of all four of LightGBM's `cegb_*` controls. No
grower calls it yet, and no public parameter reaches it, so setting a CEGB
option through `params.parse_params` today still does what it did before this
file existed. Section 8 lists exactly what has to land before each parameter
becomes reachable, and section 9 records what has and has not been checked.

Everything below is derived by reading the code. Nothing here has been run.

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
owner. `cegb_adjusted_gain` and `CegbNodeCosts.adjusted_gain` take a gain that
has already been multiplied and only subtract. Double-applying
`feature_contri` is the one way these two mechanisms can silently corrupt each
other, and it is a live risk right now, because
`FeaturePenalties.penalized_gain` still applies both the multiplier and the
CEGB split cost in one call. Until the patch in
`handoffs/remaining_04_cegb.md` splits it, `penalized_gain` and the cegb.mojo
entry points must not both be called on one gain.

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

This is an open parity question, and it is flagged in the code as well as
here. LightGBM refreshes its cached per-leaf best splits at the same point.
Whether it refunds every cached leaf or only the leaves whose cached split is
on the newly used feature decides which tree comes out. cegb.mojo implements
the arithmetically consistent rule, which is to refund what was charged. That
choice has not been compared against LightGBM 4.7.0's source.

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

`cegb_dataset_feature` performs that recovery. A multi-member bundle's shared
bin belongs to every member at once and cannot be attributed to one feature,
so it is refused rather than charged to an arbitrary member. A validated
bundling plan never puts a threshold there, since the shared bin is where
every member's default value lands, so this is a guard on an inconsistent plan
and not a case a caller has to handle.

EFB itself is disabled by default and unconnected (`efb.check_bundling_supported`
still accepts only LightGBM's "off"), so this path is written and unexercised.

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
| Dense CPU grower (`tree.grow_tree`) | reachable now | needs the ledger threaded | needs the ledger plus node row ids | `tree._search` -> `split.find_best_split` |
| Sparse grower (`tree_sparse.grow_tree_sparse`) | reachable now | needs the ledger threaded | needs the ledger plus node row ids | same `_search` |
| GPU, host split search | reachable now | needs the ledger threaded | needs a device-resident bitset | same `_search` |
| GPU, device split search | refused | refused | refused | `check_cegb_device_split_search` |
| Distributed (`distributed.grow_tree_distributed`) | reachable now | rank-consistent, needs the ledger | refused | `check_cegb_distributed` |

"Reachable now" means the term is a function of inputs those callers already
pass. It does **not** mean a public parameter reaches it; see the status line
at the top.

Three refusals, each for a specific reason:

- **GPU device split search.** Candidates are ranked inside the kernel and the
  host downloads one record per node (`train_gpu._search_leaf_device`). A
  host-side cost applied after the winner is chosen ranks nothing. The lazy
  penalty is further out of reach still, since its bitset would have to be
  device-resident and updated from the row-compaction pass.
- **Distributed, lazy only.** `unread(f, R(N))` counts rows, and the rows are
  sharded. The true count is a sum over ranks, which needs one integer
  allreduce per (node, feature) scanned. That message does not exist in
  `distributed_transport`, and a rank counting only its own shard could rank
  features differently from its peers, which is a split disagreement and not
  merely a wrong penalty. The split cost is safe there (every rank scores from
  the reduced histogram and the exact global row count), and the coupled cost
  is safe too once a ledger exists, because every rank commits the same chosen
  split and so flips the same flag at the same moment with no message.
- **Continued training from a loaded model.** Section 5.2.

`check_cegb_grower_support(config, carries_ledger, carries_rows)` is the
general form, and it mirrors `tree._search`'s existing `grower_applies_extra`
refusal. The repository's rule is that a backend which cannot apply a setting
says so rather than ignoring it.

## 9. Determinism

Every quantity is either an integer count or one multiplication of three
Float64 values in a fixed order, from state written only at commit time.
Nothing depends on scan order, thread count, or how many nodes were scored
first.

One intentional numerical difference. `unread` is counted as an integer and
multiplied once: `lazy[f] * count`. LightGBM accumulates `lazy[f]` once per
unread row, which for a large leaf is a different floating-point number.
mojoboost's is the exact one. This must be confirmed before the parity table
calls the lazy penalty bit-comparable with LightGBM.

## 10. Intentional differences from LightGBM

1. Negative costs are rejected rather than accepted (section 2).
2. The lazy term is `lazy[f] * count` rather than a `count`-long accumulation
   (section 9).
3. The cached-candidate refund is restricted to candidates on the newly used
   feature (section 5.1). **Open**: LightGBM's rule has not been read.
4. `cegb_penalty_feature_lazy` and `cegb_penalty_feature_coupled` are refused
   by `check_extra_option_supported` today rather than accepted and ignored.
   That refusal is correct while nothing consumes them, and it is what has to
   be removed last, not first.

## 11. What has to land before each parameter is exposed

In order. Each step is written out as a patch in
`handoffs/remaining_04_cegb.md`, with the owning file named, because
`src/mojoboost/cegb.mojo`, this document, and that handoff are the only files
this lane owns.

1. **Split cost, one home.** `FeaturePenalties` sheds `cegb_tradeoff`,
   `cegb_penalty_split`, and `cegb_penalty_feature_coupled` and holds a
   `CegbConfig`; `penalized_gain` keeps only the `contri` multiplier;
   `split._feature_gain` subtracts through `CegbNodeCosts`. No behavior change
   and no new parameter, but after it there is exactly one CEGB
   implementation.
2. **Ledger threading.** `tree.grow_tree` builds a `CegbLedger`, passes the
   prepared node costs into `_search`, and calls `cegb_commit_split` plus the
   frontier refund when it commits a split. This is what makes
   `cegb_penalty_feature_coupled` correct rather than parsed.
3. **Ensemble lifetime.** `boosting.fit`/`fit_multiclass` own the ledger and
   hand the same one to every `grow_tree` call, and refuse a resumed
   `fit_more`.
4. **Row ids.** The same threading carries each node's row list, which
   `grow_tree` already materializes, making `cegb_penalty_feature_lazy`
   correct.
5. **Only then** remove the two names from `check_extra_option_supported`, add
   them to `params` (Mojo API only, since both are per-feature vectors that a
   whitespace-separated parameter string cannot carry any more than
   `monotone_constraints` can), export from `src/mojoboost/__init__.mojo`, and
   move the parity row off `deferred`.

Steps 1 through 4 change no public surface. Step 5 is the only one that does.

The earlier, smaller P5 proposal in
`handoffs/connect_17_alternate_boosting.md` has been withdrawn. Its plain
`List[Bool]` first-use ledger did not refund coupled cost from cached leaf
candidates after another leaf committed the feature, so it could change the
leaf-wise frontier order relative to the stated formula. The authoritative
integration is PATCH 1 through PATCH 4 of
`handoffs/remaining_04_cegb.md`, including `cegb_stale_cached_gain`. P5 must
not be applied.

## 12. Reachability, precisely

| Claim | True today? |
| --- | --- |
| Implemented | yes, `src/mojoboost/cegb.mojo` |
| Imported by a grower | no |
| Called on a real split decision | no |
| Publicly reachable | no; `params` accepts `cegb_tradeoff` and `cegb_penalty_split` only, and both flow into `FeaturePenalties`, not into this module |
| Focused-tested | no. This lane does not write or run tests |
| Differential-tested against LightGBM | no |
| Hardware-validated | not applicable; CPU arithmetic only |

`docs/LIGHTGBM_PARITY.md` line 380 still says `deferred` for the four
parameters, and that row is correct until step 5 above. Changing it belongs to
the lane that owns that file, on evidence, not to this one.
