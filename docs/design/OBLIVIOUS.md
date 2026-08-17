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
> deep. (**Since 2026-08-17 the shipped default enqueues 55**, not 56:
> `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` became the default and
> `oblivious_schedule_launches(6, skip_last_build=True)` returns 55. Both counts
> are under the 64-deep queue, so the kill criterion is answered either way, and
> 56 stays quoted here as the all-off figure this paragraph was written about.) the six-launch gap is one record-filing phase per level that a level
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

## Part D. The CPU level engine: built, measured, REVERTED, 2026-08-16

**Do not build this again without reading D5 first.** It was built in full,
it is bit-identical to the grower it replaced, and it is **1.48x slower** on
the decision row. The code is recoverable with `git show ac52d7b`.

### D1. The hypothesis it was built to test

The CPU grower produces an oblivious level's statistics **leaf by leaf**:
`_grow_oblivious_levels` searches the level once and then, for each of the
level's `L` leaves, calls the leaf-wise subset builder, which walks every
drawn feature's bin column for that one leaf. Feature-major bins are `n_rows`
bytes a column, so a level of `L` leaves touches the whole
`n_features * n_rows` bin matrix `L` times. At depth 6 the levels are 1, 2,
4, 8, 16 and 32 wide, so a tree touches it 63 times where it appears to need
6. That reading is why the CatBoost-shape arm measured 14.4 s against
CatBoost's 3.8 s on the same tree shape
(`bench/results/COMPARISON_RUN_2026-08-16.md` Block A), losing even to our own
leaf-wise arm at 10.1 s.

### D2. What CatBoost does, read from source rather than relayed

- **Fan-out is over candidate FEATURES.** `CalcBestScore`
  (`catboost/private/libs/algo/greedy_tensor_search.cpp`) hands one task per
  feature to `ExecRange`; each owns a private
  `stats[leaf * bucketCount + bin]`. No atomics, no shared histogram, no
  cross-thread reduction.
- **One pass over the documents per feature covers every leaf.**
  `TStatsIndexer::GetIndex` (`scoring.cpp`) is
  `BucketCount * LeafIndices[obj] + quantizedValue`; `UpdateWeighted` loops a
  document range adding into it.
- **Level-wide sibling subtraction.** `SetSmallestSideControl`
  (`calc_score_cache.cpp`) counts the documents whose new bit is set and keeps
  the **minority side for the whole level** -- one boolean, not one per leaf.
  `SelectSmallestSplitSide` compacts them and sets the index to
  `srcIndices[i] | (1 << (curDepth - 1))`, landing every kept document in the
  second half of the stats array; `CalcStatsKernel` zeroes only that half, so
  the first half still holds the previous level's stats, and `FixUpStats`
  derives the other side by `stats[i].Remove(stats[i + half])`, swapping when
  the kept side was the left one.
- The compacted fold is rebuilt from the **full sampled fold** each level
  (`SelectSmallestSplitSide(curDepth + 1, ctx->SampledDocs, ...)`), so it is
  always about half the documents and always in ascending order.

All four were verified in the clone. The relay was accurate.

### D3. What was built

`src/mojotrees/oblivious_level.mojo` plus `tree._oblivious_level_fold`, at
`ac52d7b`. Per level: one blocked ascending pass compacted the level-minority
side and advanced every document's slot; one feature-parallel dispatch folded
every leaf into a flat `[slot][feature][bin]` buffer (196 KB of working set
per task at 32 leaves and 256 bins); each leaf copied its slot out and derived
its sibling with the existing `subtract_histogram_into`. Search, leaf values,
node ids, leaf numbering and frontier untouched.

**It is correct.** Every node value of every tree is bit-identical to the
leaf-by-leaf grower's at 200 x 4 depth 1, 400 x 6 depth 4 and 20,000 x 20
depth 6, and deterministic at `MOJOTREES_NUM_WORKERS` 1, 3 and 8.

One thing was NOT a rounding difference and is worth carrying forward
whatever else is: **a per-row derivative is a Float32 quantity in a Float64
word by default** (`histogram.DERIVATIVE_PRECISION_FLOAT32`, LightGBM's
`score_t`), and the leaf-wise builder narrows every row as it gathers.
Gathering the level fold raw moved every leaf value in the **eighth**
significant figure -- a different model, not a rounding difference -- and a
1e-9 relative tolerance in the test passed it. Any future gather of
derivatives outside `histogram.mojo` has to take `const_h_env.narrow`.

### D4. The measurement

Apple M4, 10 threads, `799,110 x 100`, depth 6, 20 trees, arms interleaved
inside one process, twelve repeats, under `/tmp/mojotrees-bench.lock`
`mode: timing`, box verified quiet by `ps -Ao comm | grep "mojo$"` returning
nothing immediately before the window.

| arm | median s | full range | plateau (repeats 3-11) |
| --- | --- | --- | --- |
| level engine | 2.983 | [2.767, 4.080] | [2.767, 3.232] |
| leaf-by-leaf (shipped) | 2.034 | [1.871, 3.318] | [1.871, 2.196] |

**RESOLVED on the plateau and the level engine LOSES: 1.48x slower**, ranges
disjoint (2.767 against 2.196). Repeats 0 to 2 are the warm-up and inflate
both ranges enough that a naive whole-range test calls it indistinguishable;
the plateau is the comparison, as `PROFILE_PROTOCOL.md` requires. The sign
reproduced in five separate sightings between 0.66x and 0.88x, at two shapes
and under both forced bin layouts. Canary: CPU 226.578 ms before, 216.569 ms
after, a 4.4 percent move inside the 5 percent bar, and both arms are CPU and
adjacent, so the 47 percent margin is not a canary artifact.

### D5. Why it lost, which is the part worth keeping

Phase profile (`MOJOTREES_PHASE_PROFILE=async`), same shape, 10 trees:

| phase | leaf-by-leaf | level engine |
| --- | --- | --- |
| histogram | 815.5 ms, **86.9%** | 1248.4 ms, **85.4%** |
| partition | 72.5 ms, 7.7% | 133.2 ms, 9.1% |
| split search | 26.6 ms, 2.8% | 24.7 ms, 1.7% |
| subtract | 15.3 ms, 1.6% | 15.4 ms, 1.1% |
| histogram buffers | 0.6 ms, 0.07% | 33.5 ms, 2.3% |
| gradient fill + score update | 7.4 ms, 0.8% | 7.1 ms, 0.5% |
| unattributed | 4.9 ms, 0.52% | 3.4 ms, 0.24% |
| **wall** | **942.8 ms** | **1465.9 ms** |

Three facts kill the hypothesis:

1. **The two arms build the SAME number of node-rows: 23.20M against
   23.50M** over ten trees, or 2.32M per tree against a theoretical
   `6 * N/2 = 2.40M`. Sibling subtraction had already reduced the level to
   half the documents; per-leaf smaller-child selection and CatBoost's
   level-minority selection pick essentially the same half. **There was never
   any row work to save**, so the best a level engine can do is match.
2. **The re-read the hypothesis was about is served from cache, not DRAM.**
   The leaf-wise builder processes a feature GROUP at a time (four features,
   3.2 MB at this shape), so the 32 leaves of a level re-read a
   cache-resident slice, not an 80 MB matrix. Forcing the layout confirms the
   traffic model is not the lever: feature-major 0.792 s, row-major 1.073 s,
   auto 0.944 s -- and the level engine lost under BOTH (0.66x and 0.88x).
3. **The replacement kernel is 1.53x worse per row** (53.1 against 35.2 ns
   per thousand rows). The shipped builder is a tuned SIMD accumulate with
   blocked private cells that amortizes the row-id list across a feature
   group; the level engine's inner loop is scalar and re-reads its document
   and slot arrays once per feature. Closing that gap means rebuilding that
   kernel for the level shape, for a best case of parity by fact 1.

The level engine also adds a redundant pass: it keeps the per-leaf partition
for the frontier AND makes its own compaction pass, which is the whole of the
partition column's 72.5 -> 133.2 ms. Removing it would recover about 60 ms of
a 520 ms deficit.

### D6. Where the 14.4 seconds actually is, and whose it is

**87 percent of an oblivious fit is the general CPU histogram build**, and it
is the same build every leaf-wise fit pays. The oblivious control plane --
one search per level, `L` builds of the smaller child totalling `N/2` rows,
`L` subtractions, `L` partitions -- costs 13 percent in total, of which the
symmetric-specific parts (search 2.8 percent, subtract 1.6 percent, buffers
0.07 percent) are under 5 percent.

Concretely, **one level of `L` leaves performs exactly ONE split search**
(`split_search`: 70 calls over 10 trees = 1 root + 6 levels per tree, in both
arms) **and `L` histogram builds totalling about `N/2` rows regardless of
`L`.** The per-leaf structure costs `L` call overheads and `L` full-width
subtractions and nothing else. The brief's hypothesis -- per-leaf histogram
work plus a per-leaf search and a reconcile -- was already false before any of
this was written.

So the oblivious arm is not slow because it is oblivious. It is slow because
the CPU histogram build is slow, plus it grows a 64-leaf tree where the plain
arm grows a 31-leaf one. **That is the general CPU histogram lane's, not
this one's**, and the 25x feature-group re-walk and the `_cache_group` width
clamp that lane found are the live leads. One incidental reading for whoever
owns the bin layout: the timed `auto` choice picked row-major and paid 19
percent for it at this shape (0.944 s against 0.792 s forced feature-major).
