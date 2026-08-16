# grow_policy = oblivious, a design

> **Design draft from source reading. Nothing here is measured.** This is
> Part B of the CatBoost catalog; Part A is `CATBOOST_CATALOG.md`. Items
> marked **verify** have NOT been checked against CatBoost source and must be
> before any lane builds them.

> **CPU half BUILT, 2026-08-16, `lane/oblivious-cpu`.** The CPU items of B2
> are implemented and tested (`growth_policy.GROW_OBLIVIOUS`,
> `split.find_best_split_shared`, `tree._grow_oblivious_levels`,
> `tests/test_oblivious.mojo`). CatBoost source was read first and returned
> **four corrections** to this draft, recorded under A8 in
> `CATBOOST_CATALOG.md`: the zero-contribution rule for an illegal leaf has
> no CatBoost precedent and is **ours, explicitly NOT verified from source**
> (CatBoost's `SymmetricTree` has neither `min_data_in_leaf` nor
> `min_sum_hessian_in_leaf`, so it never had to define the case; we match the
> GPU device, because host and device must grow the same tree); the
> cross-leaf aggregate is a sum of child terms with the parent term removed
> once per level, and CatBoost's CPU *default* score function is `Cosine`, a
> ratio, not a sum; `num_leaves` is not "ignored" by CatBoost but forced to
> `1 << depth` with an error on a conflicting value; and the tie rule is
> randomized by default. **B5's first falsifier did NOT fire on the CPU**:
> the cross-leaf reduction is fused into the single feature-parallel dispatch
> the search already makes, so a level costs one dispatch and not one per leaf
> plus a reduce. Nothing below is measured.

> **GPU half BUILT, 2026-08-16, `lane/oblivious-wire`.** The level schedule,
> the level commit rule and the batched whole-level child build are wired and
> reachable: `gpu_resident_round.grow_tree_device_oblivious`,
> `gpu_tree_tables._commit_level_kernel`,
> `histogram_gpu.enqueue_desc_level_children`, and the ordinary `grow_policy`
> dispatch in `train_gpu`. All three of B5's standing blockers are closed and
> `oblivious_open_blockers()` is empty. **B5's first kill criterion is
> answered**: `oblivious_schedule_launches(6)` is **56** command buffers at
> `max_items >= 64`, against the registered census's 62 and a queue that is 64
> deep; the six-launch gap is one record-filing phase per level that a level
> does not need, and the census is deliberately left unedited. **The
> `max_items >= 64` precondition is enforced rather than documented** -- a
> builder holding the leaf-wise default of 32 refuses by name
> (`OBLIVIOUS_LEVEL_HISTOGRAM`), and depth 5 cannot see the difference because
> a depth-5 level tops out at 32 children. **The accuracy gate is met
> structurally and not bit-for-bit**, and the distinction is stated where it is
> asserted (`tests/test_gpu_oblivious_device.mojo::_assert_same_shape`): the
> device tree is node-for-node identical to `tree._grow_oblivious_levels` in
> feature, threshold, missing direction, child ids, node ids and node row
> counts at depths 3 and 6, with leaf values agreeing to 1e-4 relative. Bit
> identity across backends is not reachable and is not claimed -- the host
> Newton step is Float64 over Float64 sums and the device's is Float32 over
> fixed-point Int32 sums. B4's "device vs host trees node-identical" should be
> read that way. Nothing here is measured.

## Part B. `grow_policy = oblivious`

### B1. What it is

At depth d there are 2^d leaves. One (feature, threshold[, categorical set,
missing direction]) is chosen per level and applied to every leaf. Leaf index
of a row is the bit pattern of its d outcomes. Depth default 6 in CatBoost.

Why it fits our GPU: a tree is d decisions instead of num_leaves-1, every
level is one histogram pass over all leaves plus one search, and the
per-decision host cost that this campaign has spent a week on is paid 6
times, not 31. Prediction is d comparisons and a table lookup. It is a
different model class: weaker per tree, more trees, competitive accuracy on
many tabular sets (CatBoost's default), and its comparator is CatBoost.

### B2. What changes, per backend

CPU (`tree.mojo`, `split.mojo`, `growth_policy.mojo`, `model.mojo` params):
- Schedule: reuse the depth-wise schedule (`GrowthSchedule`) with one
  addition: after all leaves of a level have histograms, run ONE search and
  apply its split to every leaf of the level.
- Search: for every candidate (feature f, bin b), gain summed over the leaves
  of the level: `sum_l gain_l(f,b)` using each leaf's own left/right sums.
  Gain is not additive across histograms, so per-leaf histograms are needed
  (2^d of them; at d=6, 50 features, 255 bins, 16-byte cells that is ~13 MB,
  fine). Ties: ascending feature then bin, as today. `min_data_in_leaf`,
  `min_child_hess`: a candidate is legal for a leaf if it passes there;
  CatBoost scores leaves that fail as zero contribution (verify); we do the
  same and record it.
- Histograms: build the smaller child of each pair, subtract the larger,
  exactly today's loop; the pool holds a level, not a frontier.
- Partition: every leaf partitioned by the same split; the arena partition
  (L1) does it leaf by leaf; row order preserved.
- Tree representation: an ordinary binary tree with the same split repeated
  at a level. Predict, dump, LightGBM model I/O, monotone, interaction, and
  forced splits keep working unchanged. Optional later: a compact
  table-lookup predictor.
- Leaf values: Newton step per leaf as today; `leaf_estimation_iterations`
  later.
- Estimated size: one lane for schedule + search + partition wiring, one lane
  for tests (bit identity across workers, agreement with a brute-force
  reference on small data) and the accuracy run. Bits move only in the new
  mode.

**As built (CPU), and where the fused reduction landed.** The requirement
that the cross-leaf reduction never get its own launch is met on the CPU by
construction, and the reason generalizes: a candidate (f, b) reads feature
f's histogram slice and nothing else, in every leaf, so the whole reduction
for feature f fits inside feature f's own task of the one
`dispatch_features_with` the search already makes. The leaf loop is the OUTER
loop inside that task, each leaf's slice walked once ascending by bin, folded
into a per-bin accumulator the task owns.

**Leaf numbering and node-id order are a cross-backend contract and the CPU
implements the device's.** Leaf index is the bit pattern of a row's outcomes
with the FIRST level's outcome as the LEAST significant bit (CatBoost's, from
`index_calcer.cpp`: `splitWeight = 1 << splitParams.Depth`), so a left child
keeps its parent's index and a right child at level d adds `1 << d`. Node ids
are assigned level by level, over the level's leaves in ascending LEAF INDEX,
left child before right. That is NOT ascending node id: at level 2 the leaves
in node-id order carry indices 0, 2, 1, 3, so the two orders first diverge
when level 2's children are created and a depth-3 tree already tells them
apart. Both halves are asserted in `tests/test_oblivious.mojo`
(`test_leaf_numbering_is_first_level_lowest_bit`,
`test_node_ids_follow_ascending_leaf_index_left_before_right`).

**Derived bound, arithmetic only,
no measurement:** search dispatches per tree fall from `num_leaves - 1` to
`max_depth`, i.e. 63 to 6 at depth 6 with 64 leaves. Cells read are unchanged
(L slices of `n_active * n_bins` either way), so this removes fan-out and
barrier cost and no arithmetic. Refused rather than half-applied under the
mode: categorical features, forced splits, `extra_trees`, and the CEGB
penalties that read the ensemble ledger. `max_depth` is required and capped
at 16; `num_leaves` does not bind. Empty leaves are real leaves emitting 0.0
and make `Tree.check_node_counts` (and so exact feature contributions) refuse
such a tree, which is the one consumer the ordinary representation does not
carry through unchanged.

GPU (resident plane, `gpu_resident_round.mojo`, `gpu_split_search.mojo`,
`gpu_active_rows.mojo`):
- Level histogram pass: the frontier IS the level; the existing range
  histogram family builds all leaves' histograms per step already.
- Search: a cross-leaf reduction, per (f,b) sum gains over leaves, then the
  same argmax the device search does now. REQUIREMENT, not a suggestion: the
  reduction is FUSED into an existing launch (the per-level search or
  commit), never its own command buffer. Static census off the measured
  per-step shape (GPU orchestrator, 2026-08-16): d=6 fused = 62 buffers per
  tree (under the measured 64 knee by 2); d=6 as its own launch = 68 (over
  by 4); d=5 is under either way (53/58); d=7 is over either way (71/78).
  A lane told "one launch per level" will build the 68 version and it will
  look correct. Two caveats: the 9-per-step figure is leaf-wise and an
  oblivious level may cost more; and 64 is a knee (per-launch enqueue cost
  roughly doubles past it), not a wall, so the argument weakens rather than
  vanishes on the far side.
- Partition: every leaf by the same split; the device partition already
  handles a frontier; A1 (physical compaction) is where CatBoost's segmented
  sort would come in and is optional.
- Trips: with the resident plane already at 1 host trip per tree, oblivious
  changes launches per tree, not trips. Measured per-tree launch shape after
  the launch-fusion lane (GPU orchestrator, 2026-08-16): 7 fixed + 9 per
  growth step + 1, i.e. 278 per tree at 30 steps (308 before fusion). At
  depth 6 that is 7 + 6 x 9 + 1 = 62, a derived bound of ~4.5x fewer
  launches, and it lands under the measured queue-depth knee of 64 command
  buffers, which is the real prize. (An earlier draft said ~6x from an
  assumed 31 x 4 shape; superseded by the measured shape.)
- Estimated size: two lanes (cross-leaf search; level schedule + partition),
  one lane to add the mode to the phase profile and the harness arms.

### B3. Parameters (LightGBM/CatBoost naming)

`grow_policy = "leafwise" | "depthwise" | "oblivious"` (default leafwise);
`max_depth` binds for oblivious (CatBoost `depth`, default 6 there; we do not
change our default); `random_strength` (A3) applies to all modes;
`num_leaves` is ignored under oblivious and says so.

### B4. Validation

- Real-data run: oblivious vs CatBoost defaults on the same Mac (CPU-only for
  CatBoost) AND vs LightGBM `stock+det`, thresholds relative to each,
  reported side by side. Speed end-to-end, both backends.
- Bit identity across `MOJOTREES_NUM_WORKERS` and across task counts, and
  device vs host trees node-identical (the resident test extended with the
  new policy).
- Golden: a seventh-style fixture for the new mode, regenerated once with
  the reason.

### B5. What would kill it, registered before any number

1. Launch-count precondition, checkable statically in an hour with the
   launch-fusion census: if the level schedule does not bring the per-tree
   command-buffer count under 64 (the measured queue-depth knee), the
   queue-depth argument evaporates and only the accuracy question remains.
   Settle this before any timing.
2. Accuracy: if, at equal end-to-end time on the M-series GPU, oblivious does
   not reach LightGBM leaf-wise accuracy within thresholds on the real-data
   set (and is not within thresholds of CatBoost defaults), it stays opt-in
   for prediction-latency users and is not a headline.

## Part C. Comparator definition for CatBoost (for the harness lane, later)

CatBoost at its defaults (`depth=6`, symmetric trees, `learning_rate` auto,
`iterations` matched to ours by tree count, `thread_count` = the same 10,
CPU-only on macOS), Dataset construction included in end-to-end. Its own
determinism setting where offered. Reported as a third column, never instead
of LightGBM `stock+det`.

## Part D. The CPU level engine, 2026-08-16

### D1. The defect, stated before the fix

The CPU grower that shipped grew an oblivious tree with the **leaf-wise
engine**. `_grow_oblivious_levels` searched the level once, as B2 says it
should, and then produced the level's statistics leaf by leaf: for each of
the level's `L` leaves it called the leaf-wise subset builder over that
leaf's row-id list, and that builder walks every drawn feature's bin column.
Feature-major bins are `n_rows` bytes a column, so a level of `L` leaves
streamed the whole `n_features * n_rows` bin matrix `L` times. At depth 6 the
levels are 1, 2, 4, 8, 16 and 32 leaves wide, so one tree read the bin matrix
**63 times where it needs to read it 6**.

That is arithmetic over the loop nest, not a measurement, and it is the
reason the CatBoost-shape arm measured 14.4 s against CatBoost's 3.8 s on the
same tree shape in `bench/results/COMPARISON_RUN_2026-08-16.md` Block A --
slower than our own leaf-wise arm at 10.1 s, for a tree that needs strictly
less work than a leaf-wise one of the same leaf count.

### D2. What CatBoost does, read from source

Verified in the clone, not relayed:

- **Fan-out is over candidate FEATURES.** `CalcBestScore`
  (`catboost/private/libs/algo/greedy_tensor_search.cpp`) hands one task per
  feature to `ExecRange`. Each task owns a private
  `stats[leaf * bucketCount + bin]` array. No atomics, no shared histogram,
  no cross-thread reduction.
- **One pass over the documents per feature covers every leaf.**
  `TStatsIndexer::GetIndex` (`scoring.cpp`) is
  `BucketCount * LeafIndices[obj] + quantizedValue`, and `UpdateWeighted`
  loops a document range adding into it.
- **Level-wide sibling subtraction.**
  `TCalcScoreFold::SetSmallestSideControl` (`calc_score_cache.cpp`) counts
  the documents whose new bit is set and keeps the **minority side for the
  whole level** -- one boolean, not one per leaf -- then
  `SelectSmallestSplitSide` compacts those documents and sets their index to
  `srcIndices[i] | (1 << (curDepth - 1))`, which lands every kept document in
  the second half of the stats array. `CalcStatsKernel` zeroes only that
  second half, so the first half still holds the previous level's stats, and
  `FixUpStats` derives the other side by `stats[i].Remove(stats[i + half])`,
  swapping when the kept side was the left one.
- The compacted fold is rebuilt from the **full sampled fold** each level
  (`SelectSmallestSplitSide(curDepth + 1, ctx->SampledDocs, ...)`), not from
  the previous level's fold, so it is always about half the documents and
  always in the full fold's ascending order.

### D3. What is built here

`src/mojotrees/oblivious_level.mojo` plus `tree._oblivious_level_fold`.
Per level:

1. `_oblivious_level_fold` makes one blocked pass over the tree's document
   set, in **ascending row order**, counting the left side; picks the side to
   build by CatBoost's rule (keep the minority, right on a tie); and compacts
   that side into `(doc, slot, gradient, hessian)` arrays, still ascending.
   It also advances each document's slot to the next level's, which is `k`
   for a left child and `k + L` for a right one -- exactly the next level's
   ascending-leaf-index order, because a left child keeps its parent's leaf
   index and a right child adds this level's power of two.
2. `accumulate_level_stats` fans out over the drawn columns. Each task zeroes
   its own column's stripe in every slot and then makes **one pass over the
   fold**, adding into a flat `[slot][feature][bin]` buffer. Per task the
   working set is `L * n_bins` cells, 196 KB at 32 leaves and 256 bins.
3. Each leaf takes its slot out of the flat buffer (`copy_level_slot`, one
   contiguous run when no feature draw is active) and derives its sibling
   with the same `subtract_histogram_into` the leaf-wise path uses.

The search, the leaf values, the node ids, the leaf numbering and the
frontier are untouched. The defect was never in any of them.

**Reachability.** `MOJOTREES_OBLIVIOUS_LEVEL_ENGINE` defaults to ON, so every
`grow_policy = oblivious` CPU fit takes it without opting in; `0` restores
the leaf-by-leaf builder so an A/B runs in one process. It is refused, by
name, for an EFB-bundled matrix, whose bin ids do not live in `data.bins`;
that case keeps the old builder.

### D4. The divergence, declared rather than discovered

Trees are **not** bit-identical to the ones the leaf-by-leaf builder grew,
and there are exactly two reasons:

1. **Addend order.** The leaf-wise subset builder folds per-row-block partial
   sums, so a cell's addends arrive in a block order. The level engine adds
   strictly in ascending document order inside one feature's task. Same
   addends, different association, so cells differ in the last bits.
2. **Which side is derived.** The old path built each leaf's smaller child
   and subtracted for the larger. The level engine builds the level's
   minority side and subtracts for the other, so for a leaf whose own smaller
   child is on the level's majority side the pair (built, derived) swaps and
   the derived cells round the other way.

Both are ulp-level. Neither changes which candidate wins except on an exact
tie between two candidates' summed gains, where the loser of the tie can
change. Determinism is unaffected: every cell is written by one task, in an
order fixed by the arguments, so the result is identical at every
`MOJOTREES_NUM_WORKERS` and on every machine.
