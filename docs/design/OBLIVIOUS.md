# grow_policy = oblivious, a design

> **Design draft from source reading. Nothing here is measured.** This is
> Part B of the CatBoost catalog; Part A is `CATBOOST_CATALOG.md`. Items
> marked **verify** have NOT been checked against CatBoost source and must be
> before any lane builds them.

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

GPU (resident plane, `gpu_resident_round.mojo`, `gpu_split_search.mojo`,
`gpu_active_rows.mojo`):
- Level histogram pass: the frontier IS the level; the existing range
  histogram family builds all leaves' histograms per step already.
- Search: a cross-leaf reduction kernel: per (f,b) sum gains over leaves,
  then the same argmax the device search does now. One launch per level.
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
