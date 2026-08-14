# Task 24 handoff, experimental GPU-friendly level-wise growth

Lane files added

- `src/mojoboost/levelwise_policy.mojo` (new)
- `src/mojoboost/gpu_levelwise.mojo` (new)
- `docs/design/GPU_LEVELWISE.md` (new)
- `handoffs/algorithm_24_levelwise.md` (this file)

Nothing central was edited. `tree.mojo`, `boosting.mojo`, `train_gpu.mojo`,
`histogram_gpu.mojo`, `gpu_active_rows.mojo`, `gpu_split_search.mojo`,
`split.mojo`, `sampling.mojo`, `src/mojoboost/__init__.mojo`, the Python
layer, the bindings, the tests, the README, packaging, and the workflows are
all untouched. Neither new module is imported by anything, so this lane
cannot change a single fitted model.

No tests were written and nothing was built, run, or benchmarked, by
instruction. **The two modules have not been compiled.** That is the first
thing the integration session should do and it is called out again in
section 7.

Line references below are against `b04b5f0` ("Integrate parallel release and
accelerator work"). The checkout is shared and moved once while this lane was
reading, folding `ExtraTreeParams` into `TreeParams` as `params.extra` and
adding the `max_delta_step` / `path_smooth` tail to `tree._leaf_value`. Both
new modules were written against the post-move state and
`gpu_levelwise.child_leaf_value` mirrors the new `_leaf_value` including that
tail. Re-check the references if HEAD has moved again.

## 1. What the lane found before writing anything

Read in full: `tree.mojo` (growth, `_LeafState`, `partition_rows_into`,
`_HistPool`, `_search`, `grow_tree`), `train_gpu.mojo` (`grow_tree_gpu`,
`_grow_tree_gpu_device_search`, `_count_left`, `_GpuLeafState`),
`gpu_active_rows.mojo` (`LeafRange`, `LeafRangeTable`, `RowRouting`,
`enqueue_partition`, `enqueue_range_histogram`), `histogram_gpu.mojo`
(builder API), `split.mojo` (`SplitInfo`), `monotone.mojo`,
`interaction.mojo`, `sampling.mojo`, `tree_parameters_extra.mojo`.

Findings that shaped the design:

| Finding | Where | Consequence |
| --- | --- | --- |
| Live leaves own contiguous, non-overlapping row ranges that tile `[0, n_active)` | `gpu_active_rows.mojo`, `LeafRangeTable.check_invariants` | One pass over the active buffer serves a whole level. This is the entire mechanical case for the mode and it is already true. |
| Both growers pick the best leaf by scanning the frontier list with strict `>` from a `best_gain` of `0.0` | `tree.mojo:908`, `train_gpu.mojo:461` | Ties resolve by frontier slot order, which is a list-maintenance artifact and does not survive ranking a level. The mode needs its own stated tie rule. |
| The depth limit and the row minimum are checked at the top of `_search`, before any bin is read | `tree.mojo:732` | Both can be evaluated before a level's launch, which is what lets an ineligible node be dropped from the batch. Mirrored, not called, and flagged below as the one duplication. |
| `_count_left` routes every bin through `SplitInfo.goes_left` with a missing-bin override, and node covers already come from those exact integer counts | `train_gpu.mojo:163` | Generalizing it to the gradient and hessian sums (`child_sums`) is what lets a level commit before its children's histograms exist, and it keeps covers exact so `has_node_counts` and TreeSHAP work unchanged. |
| `_leaf_value` now applies `max_delta_step` and `path_smooth`, taking the leaf's row count and the parent's emitted value | `tree.mojo:655`, `tree.mojo:974` | Both inputs are available to a level-wise grower before the child histogram exists (the count from `child_sums`, the parent value from `LevelNode.value`), so the leaf-side controls are honored rather than refused. |
| Per-node feature draws are keyed on node id | `sampling.select_node_features` | Breadth-first ids differ from best-first ids, so the two modes cannot be compared seed for seed. |
| `select_level_features` and `select_split_features` already exist, keyed on depth | `sampling.mojo:247`, `sampling.mojo:270` | The per-level draw needs no new sampling work. `select_split_features` is the call a level-wise grower should make. |
| `GpuHistogramBuilder.out_dev` is a single `n_features * n_bins * 3` Int32 buffer | `histogram_gpu.mojo:170` | Batching a level needs a multi-slot output buffer. This is the largest single piece of device work the mode needs. |
| `GpuActiveRows.enqueue_partition` scans one window into shared `offsets_dev` / `block_sums_dev` / `total_dev` | `gpu_active_rows.mojo:983` | Batching needs a segmented scan and per-node totals. |
| `_HistPool` sizes its free list at `num_leaves + 1` | `tree.mojo:608` | Leaf-wise growth already holds a histogram per live leaf, which bounds the memory objection to batching at 1.5x. |
| `ExtraTreeParams` is now folded into `TreeParams` as `params.extra` | `tree.mojo:132` | It was held apart while its own lane ran, which is the convention `LevelwiseParams` follows now. The same fold is the eventual home for the level-wise bundle, and section 6 puts it last on purpose. |

## 2. New API in `src/mojoboost/levelwise_policy.mojo`

Host-only policy. No histograms, no device, no dataset.

Constants

- `BUDGET_RANK`, `BUDGET_WHOLE_LEVEL`
- `STOP_RUNNING`, `STOP_LEAF_BUDGET`, `STOP_MAX_DEPTH`, `STOP_DRY_LEVEL`,
  `STOP_LEVEL_CAP`
- `UNLIMITED_LEVEL_NODES`

Types

- `struct LevelwiseParams(Copyable, Movable)` with `enabled`, `budget_mode`,
  `max_level_nodes`, plus `off()`, `on()`, `is_active()`, `check()`
- `struct LevelCandidate(Copyable, Movable, Writable)` with `node`, `gain`,
  `eligible`, plus `terminal()`

Functions

- `count_eligible(candidates) -> Int`
- `rank_level(candidates) -> List[Int]`
- `leaf_budget(n_leaves, num_leaves) -> Int`
- `admit_level(candidates, n_leaves, num_leaves, budget_mode) raises -> List[Bool]`
- `count_admitted(admitted) -> Int`
- `leaves_after_level(n_leaves, n_admitted) -> Int`
- `level_stop_reason(depth, max_depth, n_leaves_after, num_leaves, n_eligible, n_admitted) -> Int`
- `stop_reason_name(reason) -> String`
- `depth_permits_split(depth, max_depth) -> Bool`
- `rows_permit_split(n_rows, min_data_in_leaf) -> Bool`
- `prefilter_level(nodes, row_counts, depth, max_depth, min_data_in_leaf) raises -> List[Int]`
- `level_capacity(depth, num_leaves) -> Int`
- `full_level_depth(num_leaves) -> Int`
- `effective_max_depth(num_leaves, max_depth) -> Int`
- `struct LaunchProfile(Copyable, Movable, Writable)`, `leafwise_profile(num_leaves)`,
  `levelwise_profile(num_leaves, max_depth, max_level_nodes)`

## 3. New API in `src/mojoboost/gpu_levelwise.mojo`

Host-side frontier and commitment. Imports only `gain`, `histogram`,
`interaction`, `levelwise_policy`, `monotone`, and `split`, so it compiles on
a machine with no accelerator. The `gpu_` prefix names its target, not its
dependencies.

Constants

- `HOST_HIST_BYTES_PER_CELL = 24`, `DEVICE_HIST_BYTES_PER_CELL = 12`

Types

- `struct LevelNode(Copyable, Movable)`: `node`, `depth`, `n_rows`, `branch`,
  `bounds`, `value`
- `struct LevelFrontier(Copyable, Movable)`: `nodes`, `depth`, plus `root()`,
  `n_nodes()`, `is_empty()`, `total_rows()`, `node_ids()`, `row_counts()`,
  `check_sorted()`
- `struct ChildSums(Copyable, Movable, Writable)`
- `struct CommittedSplit(Copyable, Movable)`
- `struct LevelCommit(Copyable, Movable)`: `splits`, `terminal`, `depth`,
  `n_leaves_after`, `stop_reason`, plus `n_splits()`, `is_done()`,
  `next_frontier()`

Functions

- `child_sums(hist, split, missing_bin) raises -> ChildSums`
- `child_leaf_value(grad_sum, hess_sum, lambda_reg, lambda_l1, n_data, parent_output, max_delta_step, path_smooth) -> Float64`,
  the mirror of `tree._leaf_value` including its `max_delta_step` and
  `path_smooth` tail
- `level_candidates(frontier, splits, max_depth, min_data_in_leaf) raises -> List[LevelCandidate]`
- `plan_level(...) raises -> LevelCommit`
- `decide_level(...) raises -> LevelCommit`, the whole per-level host decision
- `check_child_ids(commit, next_node_id) raises`
- `level_histogram_cells`, `level_histogram_bytes`, `max_level_nodes_for_bytes`,
  `peak_resident_nodes`, `peak_resident_bytes`

## 4. Decisions this lane made, and why

These are the calls a reviewer should confirm rather than assume, because
each of them could reasonably have gone the other way.

1. **`num_leaves` stays a hard bound.** A level would otherwise overshoot it.
   `BUDGET_RANK` admits the highest-gain prefix that fits, so the last level
   is partial. The alternative, letting a level overshoot, was rejected
   because model size, serialization, and every existing statement about
   `num_leaves` depend on the bound holding.
2. **Tie-breaking is gain descending, then node id ascending**, stated rather
   than inherited. The leaf-wise rule is frontier slot order, which is not a
   rule that survives ranking a level.
3. **Node ids are assigned in ascending parent order, never in gain order.**
   The ranking decides membership only. This keeps tree layout a function of
   the frontier and the mask, and it preserves the `left == right - 1`
   invariant both shipped growers produce.
4. **The two shape rules are duplicated, not called.** `depth_permits_split`
   and `rows_permit_split` mirror the guards at the top of `tree._search`.
   The duplication is deliberate: `_search` can only answer after a histogram
   exists, and the batch has to be narrowed before it is built. They are each
   one expression, and the docstrings on both sides name each other. **If
   `_search`'s guards ever change, these must change with them.** That is the
   single maintenance debt this lane creates.
5. **Child values come from `child_sums`, not from child histograms.** This
   is what decouples commitment from the next level's build. It costs a
   floating-point difference in the last bits (exact in real arithmetic,
   different summation order in practice), which is documented in three
   places and must not be papered over in a test.
6. **No batched kernels were written.** They belong in three files this lane
   does not own, and writing speculative copies of them here would have
   created a second source of truth for the routing and scan rules that the
   existing modules are explicit about wanting to own.
7. **Nothing is registered in `__init__.mojo` and no parameter is exposed.**
   Per the brief, and because the mode should not be reachable by accident
   before it has been measured.

## 5. Device work the mode needs, in the files that own it

Specified here so the integration session does not have to rederive it. None
of this was written.

**`histogram_gpu.mojo`**

- `out_dev` becomes `n_slots * n_features * n_bins * 3` Int32, with a slot
  index per node in the level. `n_slots` is bounded by
  `LevelwiseParams.max_level_nodes` or by the level width, whichever is
  smaller.
- `enqueue_level(nodes: List[Int])`: one launch accumulating every listed
  node's range into its own slot. Each row belongs to exactly one node, so
  the slot for a row is a lookup on the range table.
- `download_level()` and a `histogram_from_host` variant taking a slot index.
- Sibling subtraction survives: enqueue only the smaller sibling of each pair
  and subtract for the larger. See the first open question in the design doc,
  which is whether that is worth the gather.

**`gpu_active_rows.mojo`**

- A segmented variant of `enqueue_partition`: per-node routing descriptors
  (`RowRouting` already carries exactly the right fields, so upload an array
  of them rather than inventing a struct), per-node block-sum segments, and
  per-node totals.
- `LeafRangeTable.split_many(parents, lefts, rights, n_lefts)`, applying a
  whole level's range splits in one host pass. The existing `split` already
  validates everything a batched form needs; keep its checks per entry.
- Each node's `expected_left` comes from `CommittedSplit.n_left`, so a level
  partitions with no readback, exactly as a single split does today.

**`gpu_split_search.mojo`**

- Per-node allow masks, candidate feature sets, and monotone bounds become
  arrays rather than restaged scalars; the staging contract comment on
  `GpuSplitSearcher` is the thing to re-read before changing it.
- One record per node downloaded per level (136 bytes each), not one per
  node per launch.

**`train_gpu.mojo`**

- `_grow_tree_gpu_levelwise`, behind a resolver in the shape of
  `resolve_split_search`, defaulting off. The loop is: `LevelFrontier.root`,
  then per level `prefilter_level`, `enqueue_level`, batched search,
  `decide_level`, apply to the `Tree` in `commit.splits` order (two
  `_add_node` calls per entry, then `_set_split`), `check_child_ids`,
  segmented partition, `commit.next_frontier()`, until `commit.is_done()`.
- Use `sampling.select_split_features` for the per-node candidate set so the
  per-level draw composes.

## 6. Order to integrate in

1. Compile both new modules on their own. Nothing imports them, so this is a
   syntax and type check with no behavioral risk.
2. Write host-only tests for `levelwise_policy` (ranking, ties, both budget
   modes, stop reasons, the capacity and depth arithmetic) and for
   `gpu_levelwise` (`child_sums` against a hand-built histogram, `plan_level`
   id assignment, `check_child_ids`, `next_frontier` ordering, monotone
   clamping and midpoint collapse). All of it runs without a GPU.
3. Add a **host** level-wise grower over `BinnedMatrix` first, as the
   reference model. It has no device dependency and it is what a device
   version gets checked against, the same way `partition_range_host` backs
   the device partition today.
4. Then the device work in section 5, one file at a time, each checked
   against the host reference.
5. Only then the `train_gpu.mojo` entry point, defaulting off.
6. Only then a benchmark, built the way section 10 of the design doc
   specifies. Do not report matched-parameter timings as a quality result.
7. Exposing a parameter is the last step and a separate decision. Nothing in
   this lane should be user reachable until the benchmark exists.

## 7. What a reviewer should check first

- **Neither module has been compiled.** Mojo dialect risk is concentrated in
  `LevelFrontier` (a `Copyable, Movable` struct holding `List[LevelNode]`),
  the `var`-argument constructors on `LevelNode` and `CommittedSplit`, and
  the `Some[Writer]` implementations. `rank_level` was deliberately written
  as a selection sort using only `append`, `resize`, and indexing, because
  `List.insert` is used nowhere else in this repository and could not be
  verified.
- **`plan_level` transfers `split^` into `CommittedSplit`.** The branch set
  is computed into a local before that transfer for exactly this reason. If
  the constructor call is reordered during review, check that nothing reads
  `split` after the transfer.
- **`child_sums` must stay in step with `_count_left`.** They walk the same
  bins by the same rule and would drift silently.
- **The 1.5x memory bound in section 6 of the design doc** rests on a level's
  node count being the tree's whole leaf count at that moment. Confirm that
  reading before quoting the bound.
- **No claim of LightGBM equivalence appears anywhere in this lane**, and
  none should be added. LightGBM grows leaf-wise; this is a different
  algorithm and the design doc says so in its first paragraph.

## 8. Not done

- No kernels, no grower, no trainer entry point.
- No tests, no builds, no benchmarks, by instruction.
- No parameter, no `__init__.mojo` registration, no parity table row, no
  README text. `docs/LIGHTGBM_PARITY.md` was not touched and should not gain
  a row for this: level-wise growth is not a LightGBM feature and it is not a
  parity gap.
- No commit.
