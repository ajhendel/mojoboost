# Where a GPU tree's time goes, leaf-wise against depth-wise

Written 2026-08-17. This is a reading lane. Nothing here was built, nothing was
run, nothing was committed. Every number below is either read off a stored
benchmark artifact in this repository, read off the source, or marked as an
estimate with its arithmetic shown.

## 0. What is verified and what is inferred, stated once

Three categories are used throughout and the words are not interchangeable.

**Measured** means a number that exists in a stored artifact in this
repository, with a path. The two artifacts this document leans on are
`bench/real_data/results/20260817T110847Z-dense1mfixed/` (the run that produced
the wall clocks and, through `backend_proof.py`, the wait and dispatch counts)
and `docs/METAL_TIMELINE.md` with `bench/results/metal_timeline_2026-08-15/`
(the Metal System Trace that supplies the per-dispatch and per-readback unit
prices).

**Read off the source** means a launch count, a grid shape, or a control path
derived by following the code. It is not a measurement. It can be wrong if the
code moves, so every citation quotes enough surrounding text to survive a line
shift.

**Estimated** means arithmetic over the first two categories. Estimates are
labeled inline, every time, and none of them is quoted anywhere as a result.

One further caution, which applies to the whole document. The 76.5 percent GPU
idle figure and the 22.9 percent compute figure that motivated this lane were
measured at 200,000 rows by 50 features against the **host-stepped** leaf-wise
control plane, in `docs/METAL_TIMELINE.md` section 4. Neither the shape nor the
control plane survives to the arm measured here. Section 4 below shows that at
799,110 rows by 100 features the shipped leaf-wise arm makes two blocking
readbacks per tree rather than thirty-two, and that its host-side floor is
about one seventh of its wall clock. **The idle figure is stale for this arm at
this shape and must not be used to price a control-plane change here.**
`docs/design/REDTEAM_SPEED_PLAN.md` section 2.3 already made this point and it
is repeated because the brief for this lane still carried the old figure.

## 1. Which code each arm runs, and the proof that it does

This is the first thing to settle, because the three GPU arms take three
different growers and no summary table says so.

### 1.1 The run

`bench/real_data/results/20260817T110847Z-dense1mfixed/`, 799,110 rows by 100
features, 100 trees, 10 threads, five repeats per arm, `dense_regression` at
tier `large`, comparator `stock+det@v2`. Medians and spreads, computed from
`records.csv`.

| arm | device | median train_s | min | max | spread | rmse (identical across all 5 repeats) |
|---|---|---|---|---|---|---|
| mojotrees | gpu | 3.6589 | 3.6114 | 3.7119 | 0.1005 (2.75%) | 0.30778238 |
| mojotrees_depthwise | gpu | 3.2449 | 3.1830 | 3.3322 | 0.1492 (4.60%) | 0.30736693 |
| mojotrees_catboost_mode | gpu | 17.0719 | 17.0322 | 17.8772 | 0.8451 (4.95%) | 0.30327112 |
| catboost | cpu | 3.2692 | 3.1955 | 4.9191 | 1.7236 (52.7%) | 0.30346763 |
| lightgbm | cpu | 4.6341 | 4.4820 | 5.8745 | 1.3924 (30.1%) | 0.31102765 |
| mojotrees | cpu | 7.4787 | 7.3984 | 9.3334 | 1.9351 (25.9%) | 0.30778238 |
| mojotrees_depthwise | cpu | 7.0853 | 6.8507 | 8.5192 | 1.6685 (23.6%) | 0.30740265 |

These are the brief's figures, so the run is identified correctly.

Under M0 the depth-wise against leaf-wise GPU comparison is **resolved**. The
medians differ by 0.4140 and the wider arm's own spread is 0.1492. It is
stronger than that. The leaf-wise minimum, 3.6114, is above the depth-wise
maximum, 3.3322, so no ordering artifact inside this window can flip the sign.
The arms were not interleaved, which is a real weakness under
`bench/results/PROFILE_PROTOCOL.md`, and the non-overlap is what makes the
result usable anyway.

The rmse column is bit-stable across all five repeats of every arm. The
accuracy difference between the two policies is therefore not noise. It also
reproduces on the host backend, where depth-wise reads 0.30740265 against
leaf-wise 0.30778238.

### 1.2 The parameters, which are NOT what the checked-out harness says

The record at
`records/037-dense_regression.mojotrees_depthwise.gpu.t10.r0.json` carries the
resolved parameters for the depth-wise arm.

```
num_leaves 31, max_depth -1, learning_rate 0.1, n_estimators 100,
min_data_in_leaf 20, min_child_hess 0.001, lambda_l1 0.0,
max_bin 255, grow_policy "depthwise", random_state 190019
```

Every one of those, apart from `grow_policy`, is identical to the leaf-wise
arm's. **The measurement is a pure growth-order isolation.**

The working tree disagrees. `bench/real_data/scenarios.py` now holds a
seven-key `MOJOTREES_DEPTHWISE` dict adding `max_depth 6`, `num_leaves 64`,
`learning_rate 0.3`, `lambda_l2 1.0`, `min_child_hess 1.0` and
`min_data_in_leaf 1`, and `git diff` shows the whole dict is an uncommitted
addition. It postdates the run. Anyone who reads the arm's definition out of
the checked-out file and applies it to these numbers will conclude that the
accuracy difference is a learning-rate difference. It is not. The record is the
authority and it says one key changed.

### 1.3 The route, measured

`bench/real_data/backend_proof.py` arms `MOJOTREES_PHASE_PROFILE=async` for
every recorded run (`enable()`, `ENV_VAR = "MOJOTREES_PHASE_PROFILE"`,
`MODE = "async"`) and `run.py` parses the trainer's own `phase_profile` block
out of the worker's stdout. The parsed block is stored in every record. So the
run left behind a real reading of the instrument, and it pins the route.

From `records.json`, repeat 0 of each GPU arm, per tree.

| arm | label | trees | nodes | transfer calls | syncs | charged dispatches |
|---|---|---|---|---|---|---|
| mojotrees | train_gpu | 100 | 0 | 100 | 200 | 0 |
| mojotrees_depthwise | train_gpu | 100 | 6100 | 600 | 700 | 16300 |
| mojotrees_catboost_mode | train_gpu | 100 | 12600 | 200 | 100 | 5600 |

Each row identifies exactly one grower.

**Leaf-wise takes the device-owned plane, `gpu_resident_round.grow_tree_device_resident`.**
`nodes = 0` and `dispatches = 0` with `transfer calls = 1` per tree is the
signature of exactly one bracket in the package, the one in `train_gpu.mojo`
that reads

```
profile.charge(PROF_TRANSFER, n_root, 0, syncs=1)
```

followed by a `PROF_DEVICE_PLANE` charge with no `dispatches` argument, under
the comment "NO LAUNCH COUNT IS CHARGED HERE, AND THE ASYMMETRY WITH THE
OBLIVIOUS BRACKET IS DELIBERATE". The plane calls neither `note_node` nor any
dispatch charge, so zeros there mean "not instrumented" and the single transfer
means the plane's one `download_desc_tables`. The second sync per tree is
the round's fixed-point scale readback, accounted for in section 2.1.

**Depth-wise takes the host-stepped batched loop, `train_gpu._device_search_resident`.**
It cannot take the plane. `gpu_tree_tables.tree_resident_supported` returns
`TREE_RESIDENT_DEPTHWISE` on `if params.grow_policy != GROW_LEAFWISE`, which
`gpu_resident_round.resident_round_supported` turns into `RESIDENT_TABLES`, and
`train_gpu.mojo` then falls through to `_device_search_resident`. The counts
confirm it arithmetically. `_enqueue_resident_split` calls `note_node` twice per
split, once for `PROF_HISTOGRAM` and once for `PROF_SUBTRACT`, and the root adds
one, so `nodes` per tree is `1 + 2 * splits`. The measured 61 gives 30 splits,
which is 31 leaves, which is `num_leaves`. The transfer count of 6 is the root
`download_frontier` plus one per batch, so five batches, which is five committed
levels, which under `BUDGET_RANK` is 1 + 2 + 4 + 8 + 15 = 30 splits. The two
derivations agree exactly.

**The CatBoost-mode arm takes the oblivious level schedule, `grow_tree_device_oblivious`.**
`nodes = 126` per tree is `2^(max_depth+1) - 2` at depth 6, which is what the
bracket derives from tree geometry, and `dispatches = 56` per tree is
`oblivious_schedule_launches(6, OBLIVIOUS_MAX_ITEMS)` called rather than copied.

So the brief's "roughly 5 waits per tree instead of about 30" is a quotation
from `_device_search_resident`'s own docstring describing depth-wise against
leaf-wise **inside that loop**. The shipped leaf-wise arm is not in that loop.
The true shipped comparison is 2 blocking readbacks per tree against 7, and the
arm with 7 is the faster one. That inversion is the whole result of this
document.

## 2. One leaf-wise tree, stage by stage

Grower `gpu_resident_round.grow_tree_device_resident`, entered from
`train_gpu._grow_tree_gpu_device_search` at the branch guarded by
`if resident_round_enabled():` and `if why == RESIDENT_OK:`. 31 leaves, 30
growth steps, 100 active features, 255 bins, squared error, no bagging, device
gradients.

Everything in this section apart from the two unit prices is **read off the
source**. Launch counts are the count of `enqueue_function` calls reached, not a
constant.

### 2.1 Round-level stages, outside the tree

These are in `train_gpu._train_gpu_rounds`, device-gradient arm. The arm is
identified by the presence of `_device_round_random_score_scale` and
`fill_gradients_device` in the branch, and by the profile's `PROF_GRAD_FILL`
charge carrying `syncs=1 if scale_synced else 0`.

| stage | launches | host syncs | code |
|---|---|---|---|
| gradient and hessian fill | 1 | 0 | `GpuObjectiveState.fill_grad_hess`, one `ctx.enqueue_function[_grad_hess_kernel]` |
| gradient quantization | 1 | 0 | `GpuActiveRows._ensure_quantized`, on by default (`MOJOTREES_GPU_QUANTIZED_GRADS` defaults to 1) |
| fixed-point scale reduction | 1 | **1** | `_refresh_scales` at `scale_refresh == 0` calls `state.magnitude_sums`, which is "same launch, same copy, same synchronize" |
| root row seeding | 1 | 0 | `GpuActiveRows.begin_tree`, `enqueue_function[_iota_kernel]` on the unbagged path |
| prediction score update | 1 | 0 | `GpuObjectiveState.update_raw_ranges`, one `enqueue_copy` of the segment table plus one `enqueue_function[_range_table_add_raw_kernel]` |
| **round-level total** | **5** | **1** | |

The score update deserves a note because it is the CPU backend's worst stage
and it is not a problem here. On the host the update walks the finished tree for
every training row serially. On the device it is one kernel over the row-range
table the grower left behind, with no wait, and the profile's
`PROF_SCORE_UPDATE` line is not even charged on this arm.

The scale reduction is the one round trip. `fill_gradients_device`'s docstring
states it plainly, that "nothing can be enqueued until the answer is home, so
this is a **round trip**". It is the second sync per tree in the measured table
of section 1.3.

### 2.2 Tree-level stages

| stage | launches per tree | host syncs | code |
|---|---|---|---|
| root histogram | 2 | 0 | `builder.enqueue_leaf(0, resident_slot=0)`, which reaches `enqueue_range_histogram` and issues `_zero_int32_kernel` then the atomic family |
| tree table reset | 1 | 0 | `enqueue_desc_begin_tree` to `DeviceTreeTables.begin_tree(n_active, root_slot=0, wait=False)`, one `_reset_tables_kernel` |
| root search | 2 | 0 | `searcher.enqueue_frontier(root_batch, ...)`, a scan launch and a reduce launch, plus one upload copy of the packed table parent |
| root value seed | 1 | 0 | `enqueue_desc_seed_root`, one `_seed_root_value_kernel` at `grid_dim=1, block_dim=1` |
| 30 growth steps | 270 | 0 | the `for step in range(params.num_leaves - 1):` loop, broken out below |
| terminating step | 1 | 0 | the extra `enqueue_desc_step` below the loop, "the step that ends growth rather than performing it" |
| table pack and download | 1 | **1** | `download_desc_tables`, one `_pack_tables_kernel`, one copy, one `synchronize` |
| **tree-level total** | **278** | **1** | |

Per growth step, in the loop's own order, nine launches and no wait.

| step stage | launches | grid | code |
|---|---|---|---|
| pick, commit, write the tree, move the slot pool, publish the descriptor | 1 | `grid_dim=1, block_dim=PICK_THREADS` (64) | `enqueue_desc_step` to `DeviceTreeTables.enqueue_step`, `_pick_and_commit_kernel` |
| point the two scratch records at the children's slots | 1 | `grid_dim=1, block_dim=1` | `enqueue_desc_stage_search` to `enqueue_stage_child_search`, `_stage_child_search_kernel` |
| partition the parent's rows | 2 | `(241, 13)` at this shape, see 2.3 | `enqueue_desc_partition(row_bound)` to `GpuActiveRows.enqueue_partition_desc`, `_flag_scan_kernel` and `_scatter_kernel`; the third launch is deferred under the default fusion |
| build the smaller child, subtract the larger in the same kernel | 2 | zeroing at `ceil(76500/256)` capped, atomic at `(50, 1)` | `enqueue_desc_child(row_bound)` to `GpuActiveRows.enqueue_desc_histogram`, `_copy_back_zero_slot_kernel` (the fused copy-back plus zeroing) then the atomic family |
| search both children | 2 | scan `(100, 2)` at `block_dim=64`, reduce `2` at `block_dim=64` | `_launch_child_search` to `gpu_split_search._launch_search`, `_scan_slot_wide_kernel` and `_reduce_slots_block_kernel` |
| file the two records in the frontier slots that own them | 1 | `grid_dim=1, block_dim=PICK_THREADS` | `enqueue_desc_copy_records` to `enqueue_copy_records`, `_copy_records_kernel` |
| **per step** | **9** | | |

**Per tree, leaf-wise, the total is 283 kernel launches, 2 uploads, 2 downloads,
and 2 blocking `synchronize` calls**, counting the round-level stages of 2.1
alongside the tree. The two syncs are the plane's `download_desc_tables` and the
round's scale readback, which is exactly the measured 200 syncs over 100 trees
in 1.3. Four of the nine per-step launches run at
`grid_dim=1`, so 93 of the 283 launches per tree occupy one threadgroup of a
ten-core GPU. That figure matters in section 4 and it is the plane's price for
the wait it removes.

### 2.3 The two grids the plane cannot size, and what that actually costs

This is the hypothesis a reader forms first on seeing `row_bound = n_root` and
it is worth killing carefully, because at this shape it is nearly free and at
other shapes it is not.

The plane cannot know a node's row count host-side. That is the point of it. So
`enqueue_desc_partition(row_bound)` and `enqueue_desc_child(row_bound)` are both
handed `n_root`, which at this shape is 799,110, for every one of the 30 steps.

**The histogram grid does not change.** `enqueue_desc_histogram` calls
`derive_tiling(caps, max_rows, n_slots, n_bins, STRATEGY_ATOMIC, ...)` with
`block_shared_bytes` left at its default of zero, so
`target_blocks_for(caps, 0)` is `caps.sm_count * TARGET_BLOCKS_PER_SM`, which on
the 10-core M4 recorded in `docs/METAL_TIMELINE.md` section 1.3 is 80. Then
`row_tile_floor(80, 100, 0)` is `ceil(80 / 100)`, which is **1**, and
`resolve_tiling` clamps `wanted` down by `tiles_by_rows` but never up, so
`n_tiles = 1` for every node of every tree regardless of `max_rows`. The grid is
`(ceil(n_slots / GROUP), n_tiles)`. `GROUP` is 2, because
`GpuHistogramBuilder` calls `free_feature_group(self.rows.bin_cap, baseline)`
with `baseline = 2` on Metal and `free_feature_group` returns the baseline
unchanged at a 256-bin capacity. So the grid is **`(50, 1)`, always**, and the
oversized `max_rows` costs the histogram nothing at this shape.

There is independent confirmation that `n_tiles` is 1. The depth-wise arm's
charged dispatch count of 163 per tree decomposes exactly as
`1 + 30 * (PARTITION_LAUNCHES + LAUNCHES_ATOMIC) + 6 * SPLIT_SEARCH_DEVICE_LAUNCHES`
with `PARTITION_LAUNCHES = 4`, `LAUNCHES_ATOMIC = 1` and
`SPLIT_SEARCH_DEVICE_LAUNCHES = 2`. `LAUNCHES_ATOMIC` is charged only when
`resolve_tiling` resolved `STRATEGY_ATOMIC`, and under `STRATEGY_AUTO` that
happens only at `n_tiles == 1`. So the measured profile says every histogram in
that fit, root included, ran at one row tile.

The same instrument confirms it in the other direction at a different feature
count, which is the check that makes it an arithmetic rule rather than a
coincidence. `bench/results/profile_2026-08-15/phase_gpu_1m.txt`, at
1,000,000 rows by **50** features, reads
`phase histogram all calls=3100 dispatches=6200`, which is **two** launches per
build, which is `LAUNCHES_TILED`. At 50 features `row_tile_floor(80, 50)` is 2,
`n_tiles` is 2, and `STRATEGY_AUTO` resolves to `STRATEGY_TILED`. So the tile
count is 2 at 50 features and 1 at 100, both predicted by
`ceil(target_blocks / n_slots)` and both confirmed by a stored dispatch count.
**Crossing 80 active features on this machine silently changes the histogram
strategy**, and nothing in the parameter surface says so.

**The partition grid does change, and the cost is small and boundable.**
`_partition_grid(bound, threads, cap)` with `bound = 799110`, `threads = 256`
(`TARGET_BLOCK_THREADS`) and `cap = block_threads = 256` gives
`tiles_total = 3122`, `blocks = 256`, `tiles = 13`, `blocks = 241`. So every
descriptor partition launches `(241 blocks, 13 tiles)`. The host-stepped arm
calls `enqueue_partition(bins, window, routing)` with the parent's real window,
so a 25,000-row parent gets 98 blocks and 1 tile.

**Estimated bound on the surplus.** 241 blocks times 13 tiles is 3,133
block-tile visits per launch, two launches per step, thirty steps, so 188,000
block-tile visits per tree of which a few hundred do work. Each empty visit is a
bounds compare and a loop increment. At 241 times 256, which is 61,700
concurrent lanes, and taking four cycles per empty tile at the M4's clock, the
whole surplus is on the order of **10 to 50 microseconds per tree, under 0.2
percent**. This is an estimate with its arithmetic shown and it is a bound, not
a measurement. The conclusion is that the plane's fixed grids are not the
problem at this shape and a lane spent on them would be wasted.

The conclusion changes at fewer features. At 50 features `row_tile_floor(80, 50)`
is 2, and the plane would then hand a 300-row node a two-tile grid where the
host arm gives it one. That is still small, but it is the reason the bound above
is written with its inputs visible.

## 3. One depth-wise tree, stage by stage

Grower `train_gpu._device_search_resident`, reached because
`tree_resident_supported` refuses `grow_policy != GROW_LEAFWISE`. 31 leaves, 30
splits, five committed levels, five batches plus the root search.

Round-level stages are identical to section 2.1, five launches and one round
trip. What differs is the tree.

| stage | launches per tree | host syncs | code |
|---|---|---|---|
| root histogram | 2 | 0 | `builder.enqueue_resident_leaf(root, root_slot)` to `enqueue_leaf(node, resident_slot=slot)` |
| root search | 2 | 0 | `searcher.enqueue_frontier(builder.batcher[0].out_dev, root_batch, ...)`, plus one upload copy |
| root record download | 0 | **1** | `searcher.download_frontier(1)`, charged as `profile.charge(PROF_TRANSFER, n_root, root_dl_started, syncs=1)` |
| 30 splits, 5 per split | 150 | 0 | `_enqueue_resident_split`, broken out below |
| 5 batch searches | 10 | 0 | `searcher.enqueue_frontier(... batch ...)` per batch, plus one upload copy each |
| 5 batch downloads | 0 | **5** | `searcher.download_frontier(len(batch))` per batch |
| **tree-level total** | **164** | **6** | |

Per split, five launches and no wait.

| split stage | launches | grid | code |
|---|---|---|---|
| partition the parent's rows | 3 | sized to the parent's own window | `builder.apply_split(...)` to `GpuActiveRows.enqueue_partition`, whose docstring says "Three launches, whatever the range length", `_flag_scan_kernel`, `_scatter_kernel`, `_copy_back_kernel`. The partition-tail fusion is consulted only in `enqueue_partition_desc`, so the host arm always pays three |
| build the smaller child, subtract the larger in the same kernel | 2 | zeroing plus atomic `(50, 1)` | `builder.enqueue_resident_leaf_subtracting(built_node, built_slot, parent_slot)` to `enqueue_leaf(node, resident_slot, subtract_from_slot)` |
| **per split** | **5** | | |

Two more things about this loop are worth pinning down because they are the
reason it is comparable to the plane at all.

**It does the same histogram arithmetic.** `subtraction_builds_left(n_left, n_right)`
picks the smaller child, the smaller child is accumulated, and the larger is
derived by subtracting inside the same kernel. `PROF_SUBTRACT` is charged with
zero time and zero launches for exactly that reason, under the comment "The
sibling subtraction rides inside that same kernel here rather than costing a
launch of its own". So neither arm reads the derived child's rows.

**Batching does not change a decision.** `GrowthSchedule.plan_level` "calls
`next_leaf` and returns what `next_leaf` returned", so the batch is the same
leaves in the same sequence a one-at-a-time loop would have split. Leaf-wise
growth gets a one-element list from the same function, which is why the
leaf-wise arm inside this loop makes one wait per split.

**Per tree, depth-wise, the total is 169 kernel launches, 7 uploads, 7
downloads, and 7 blocking `synchronize` calls**, counting the round-level
stages of 2.1 alongside the tree. Six of the uploads and six of the downloads
are the searcher's tables and its records; the seventh of each is the score
update's segment table and the scale reduction's readback.

### 3.1 Cross-check against the instrument

The profile's charged 163 dispatches per tree is a **model** rather than a
count. The reading above says the truth is 164. The discrepancy is one launch at
the root and two stale constants that happen to cancel.

- `phase_profile.PARTITION_LAUNCHES = 4` is **stale**. Its docstring lists "a
  flag-and-scan pass, a block-sum scan, a scatter, and a copy back", but
  `GpuActiveRows.enqueue_partition` says "Three launches, whatever the range
  length ... It was four until this lane folded the block-sums scan into the
  scatter". The constant over-charges each partition by one.
- `gpu_tiling.LAUNCHES_ATOMIC = 1` **under-counts**. The atomic path in
  `enqueue_range_histogram` issues a `_zero_int32_kernel` before the atomic
  family whenever `tiling.strategy != STRATEGY_TILED`, which is always here, so
  a build is two launches. The constant under-charges each build by one.

Four plus one is five and three plus two is five, so the per-split figure is
right by accident and the root is short by one. This is precisely the failure
`phase_profile.mojo`'s own module docstring warns about, that these constants
"can go stale silently if the function it counts is restructured", and it has
now happened to both of them. Correcting them is a two-line edit in a file this
lane may not touch and it is recorded here as a finding rather than a fix.

`_device_search_resident`'s docstring carries the same drift. It prices a split
at "eight launches and one wait" where the reading gives five plus two per
batch, and a depth-wise level at "`6L + 2` launches" where the reading gives
`5L + 2`. Its "5 waits for a tree instead of 30" omits the root download, and
the measured figure is 6 tree waits plus the round's scale readback, which is 7.

## 4. The two arms side by side, priced against the measured device constants

The unit prices below are **measured**, in `docs/METAL_TIMELINE.md` section 6,
from a Metal System Trace of 22,107 command buffers. They are device properties
rather than workload properties, which that section establishes by showing the
median enqueue moved from 12.67 to 12.62 microseconds between a 50,000-row and a
200,000-row capture while compute changed by 2.3x.

| quantity | measured median | source |
|---|---|---|
| the commit call on the CPU | 12.62 us | METAL_TIMELINE 6, `enqueue: the commit call on CPU` |
| commit N to commit N+1 when the host is not blocked | 14.58 us | same table |
| one blocking readback, commit to next commit | 606.1 us | METAL_TIMELINE 6.2, of which 3.7 us is the GPU moving bytes |
| gap between back-to-back GPU dispatches | 3.33 us | METAL_TIMELINE 4.1 |
| GPU time of a dispatch under 5 us | 3.2 us mean | METAL_TIMELINE 5.1, n = 7005 totalling 22.55 ms |

### 4.1 The host-side floor

**Estimated**, launches times the non-blocked commit interval plus readbacks
times the blocking readback cost.

| arm | launches | host enqueue | readbacks | readback cost | host floor |
|---|---|---|---|---|---|
| leaf-wise plane | 283 | 4.13 ms | 2 | 1.21 ms | **5.34 ms per tree** |
| depth-wise loop | 169 | 2.46 ms | 7 | 4.24 ms | **6.70 ms per tree** |

Measured wall clock is 36.59 ms per tree leaf-wise and 32.45 ms per tree
depth-wise. So the host floor is 15 percent and 21 percent of the respective
trees, the GPU is the critical path in both, and **the arm with the higher host
floor is the faster arm by 4.14 ms per tree.** Depth-wise pays an estimated 1.36
ms per tree more in fixed host cost and still wins.

That is the anomaly stated numerically. Whatever explains it is on the device
side, not the host side, and it is worth at least 4.14 plus 1.36, so an
**estimated 5.5 ms per tree of device time**.

### 4.2 What can be attributed on the device side

The plane issues 114 more launches per tree than the loop. The composition is

- 93 more control launches, all at `grid_dim=1`. Thirty each of
  `_pick_and_commit_kernel`, `_stage_child_search_kernel` and
  `_copy_records_kernel`, plus `_reset_tables_kernel`, `_seed_root_value_kernel`
  and the terminating `_pick_and_commit_kernel`.
- 50 more search launches. The plane searches two records thirty times; the loop
  searches a whole level five times. The total block work is nearly identical,
  `100 * 2 * 31 = 6200` scan blocks for the loop against `30 * 100 * 2 = 6000`
  for the plane, so this is the same work split twenty-five launches finer.
- 30 fewer partition launches, because the descriptor partition folds its
  copy-back into the next histogram's zeroing under the default
  `MOJOTREES_GPU_FUSE_PARTITION_TAIL`.

**Estimated** GPU-side cost of that difference, using 3.2 microseconds of GPU
time plus a 3.33 microsecond gap for a small dispatch.

| component | count | estimated per dispatch | estimated total |
|---|---|---|---|
| extra `grid_dim=1` control launches | 93 | 6.5 us | 0.60 ms |
| extra search launch overheads | 25 | 6.5 us | 0.16 ms |
| fewer partition launches | -30 | 6.5 us | -0.20 ms |
| plane's oversized partition grids | 30 steps | see 2.3 | 0.01 to 0.05 ms |
| **attributed** | | | **0.6 ms per tree** |

**So an estimated 0.6 ms of the 5.5 ms is attributable to launch shape, and
roughly 4.9 ms is not.** No fixed cost in either arm accounts for it. The only
remaining variable is how many rows each arm's histograms accumulate, and that
is a property of the tree shape rather than of the control plane.

### 4.3 The row-work residual, quantified from a stored profile

With sibling subtraction, a tree's accumulated rows are
`n_root + sum over internal nodes of min(rows(left), rows(right))`. That sum
equals the number of times a row takes the minority branch, summed over rows.
The partition, separately, walks `sum over internal nodes of rows(v)`, which is
`N` times the row-weighted mean leaf depth.

Neither quantity is derivable by reading, because both depend on how balanced
the splits are on this data. **One of them has already been measured**, and the
measurement is sitting in the repository unread.

`bench/results/profile_2026-08-15/phase_gpu_1m.txt` is a stored `async` phase
profile of a **leaf-wise** GPU fit at 1,000,000 rows by 50 features, 100 trees,
`nodes=6100`, `root_rows=1000000`, taken on the host-driven plane before the
device plane became the default. Its `rows` columns read

```
phase histogram all  calls=3100 dispatches=6200 syncs=0 rows=230601070 ...
phase subtract  all  calls=3000 dispatches=0    syncs=0 rows=508893793 ...
phase partition all  calls=3000 dispatches=12000 syncs=0 rows=639494863 ...
```

Per tree, at `N = 1,000,000`, that is

| quantity | per tree | as a multiple of N |
|---|---|---|
| rows accumulated, root included | 2,306,011 | 2.306 N |
| rows accumulated, splits only | 1,306,011 | 1.306 N |
| rows derived by subtraction | 5,088,938 | 5.089 N |
| rows walked by the partition | 6,394,949 | 6.395 N |

Two facts fall straight out. The partition figure equals built plus derived,
which is `sum over internal nodes of rows(v)`, so the **row-weighted mean leaf
depth of a 31-leaf leaf-wise tree on this generator is 6.395**. And the built
child is `1.306 / 6.395`, so **the average split sends 20.4 percent of the
parent's rows to the child that is actually accumulated**. Leaf-wise growth on
this data is markedly unbalanced.

Now do the same arithmetic for depth-wise. Its shape is not data dependent,
because `BUDGET_RANK` fills levels. Levels 0 through 3 split every leaf, so
their parents tile all `N` rows; level 4 admits 15 of its 16 leaves under the
budget, so its parents cover about `15/16 N`. Therefore

```
partition rows        = N (1 + 1 + 1 + 1 + 0.94)  =  4.94 N
```

against leaf-wise's measured 6.395 N, which is **23 percent fewer partitioned
rows**. And carrying the measured 20.4 percent split imbalance across,

```
accumulated rows      = N + 0.204 * 4.94 N        =  2.01 N
```

against leaf-wise's measured 2.306 N, which is **an estimated 13 percent fewer
accumulated rows**.

**Estimated, and the assumption is named.** The 20.4 percent imbalance is
measured on leaf-wise trees at 1,000,000 by 50 and is carried to depth-wise
trees at 799,110 by 100. It is a property of the data and the split rule rather
than of the growth order, and both shapes come from the same
`dense_regression` generator, so the carry is reasonable. It is not measured and
depth-wise's own imbalance could differ. The depth arithmetic, by contrast, is
exact given `BUDGET_RANK` and 31 leaves.

Does 13 percent cover the residual? The residual from 4.2 is an estimated 4.9 ms
per tree out of 36.59. Thirteen percent of the histogram would supply that only
if the histogram were the whole tree, which it is not. So the honest reading is
that **the row-work account supplies a large part of the residual and probably
not all of it**, with the partition's 23 percent saving contributing on top of
the histogram's 13 percent, and the two together plausibly covering 3 to 4 ms of
the 4.9. What is no longer in doubt is the direction. Depth-wise does strictly
less row work on this data, in both the histogram and the partition, and the
size is in the right range.

Section 7 item R0 converts the estimate into a measurement of both arms at the
same shape, and it needs no new code.

## 5. The instrumentation that exists, what it cannot see, and what is actually possible

### 5.1 What exists today

Six instruments, and the first of them is stronger than its own docstring
suggests.

1. **`PhaseProfile` (`src/mojotrees/phase_profile.mojo`).** Eleven phases by
   five node size classes, seven counter columns each, off by default, with
   `async` and `fenced` modes. The whole of section 1.3 and section 3.1 above
   was derived from a stored `async` reading of it. It pinned which grower each
   of three arms took, the wait count per tree, the split count per tree, and
   the resolved histogram strategy, without a line of new code. The instrument
   is not blind. It cannot break the device plane into sub-phases, which is a
   different and narrower limitation than the one its docstring emphasizes.
2. **`bench/real_data/backend_proof.py`.** Arms the profile at `async` for every
   recorded harness run and parses `trees`, `nodes`, `transfer_calls`,
   `convert_calls`, `host_sync_calls`, `syncs` and `dispatches` into the record.
   This is why the run in section 1 could be attributed retrospectively. It
   discards the rest of the report, including the `rows`, `slots` and `cells`
   columns, which is the one gap that matters for section 4.3.
   `bench/results/profile_2026-08-15/phase_gpu_1m.txt` is the one place a full
   report was kept as text rather than parsed and dropped, and section 4.3 is
   entirely a reading of it. **Keeping the text is worth more than parsing six
   fields out of it**, and item D of 5.3 is the four-line change that keeps the
   row columns permanently.
3. **`MOJOTREES_GPU_TREE_RESIDENT_TRACE` and `..._TRACE_STEPS`.** One record per
   tree, or per step, holding the plane's status, counters and whole frontier.
   The step form downloads inside the loop, so it is a debugging instrument and
   not a timing one.
4. **`MOJOTREES_GPU_PHASE_TRACE=1`.** Per-tree `hist_s`, `partition_s`,
   `search_s` and `other_s` on the host-stepped loop, with a drain after the
   partition and after the histogram.
5. **`probes/readback_cost.mojo`.** Eleven readback arms, including `bare_sync`,
   `kernel_sync`, `pinned_pair_nosync`, `host_direct`, `spin`, `cpu_callback`
   and `event`.
6. **`bench/apple/metal_capture.sh` and `bench/apple/metal_timeline.py`.** A
   full Metal System Trace, taken with `xctrace record --template 'Metal System
   Trace' --launch`, reduced to per-dispatch lifecycles by a standard-library
   Python reader.

### 5.2 The claim that per-kernel device time is unreachable, re-tested rather than inherited

The brief asked for this to be checked rather than accepted. It holds, and the
important part of the answer is that **the re-test already exists and costs one
run.**

`probes/readback_cost.mojo` contains `_report_event(ctx)`, which calls
`ctx.create_event()` inside a `try` and reports either the failure or, on
success, the detail string "an event was created; a narrower wait may exist".
So the claim is not an inherited assertion in a docstring, it is an executable
arm of a probe that ships in this repository. Running
`mojo run probes/readback_cost.mojo` re-tests it, and if a MAX release ever
lands Metal events, that arm is where it shows up. This lane may not run it, so
what follows is the recorded result rather than a fresh one.

`docs/GPU_PORTABILITY.md` section 6.5.2 records, verified by execution on an
M4,

| API | result |
|---|---|
| `DeviceContext.create_event()` | raises `eventCreate is not supported on this device` |
| `DeviceContext.create_stream()` | raises `createStream is not supported on this device`; `num_streams()` answers 1 |
| `DeviceContext.enqueue_cpu_function(f)` | raises `enqueue_cpu_function is only supported on CPU DeviceContexts` |
| `DeviceContext.handle` / `unsafe_ptr` / `native_handle` / `stream_handle` | do not exist; the compiler rejects each |
| `DeviceContext.query` / `is_idle` / `poll` / `is_complete` | do not exist; the compiler rejects each |
| `DeviceGraph.create` | raises `createGraphBuilder() not supported on this device context` |

and, independently, by disassembly of `libMGPRT.dylib`, that `newSharedEvent`,
`newSharedEventHandle`, `newSharedEventWithHandle:`, `newEvent`, `newFence`,
`encodeSignalEvent:value:`, `encodeWaitForEvent:value:`,
`notifyListener:atValue:block:` and `addCompletedHandler:` are each registered
in the runtime's metal-cpp selector table and each has **zero load sites**,
while the only synchronization selectors with load sites are `commit` with 8 and
`waitUntilCompleted` with 6.

Two further routes are closed and are worth naming because they are the ones a
reader proposes next.

- **Polling device memory from the host.** Not available. Section 6.5.3 records,
  measured by execution, that `DeviceBuffer.unsafe_ptr()` returns the buffer's
  `gpuAddress`, that an M4 returned `0x10000080000` for a fresh device buffer
  and **faulted** on the first host load, and that a `HostBuffer`'s pointer is a
  real host address which a kernel cannot use as a GPU address. Both names for
  the same shared memory exist inside the runtime and the Mojo API hands out
  exactly one per allocation kind.
- **A device timer.** `DeviceContext.execution_time` exists in the API.
  `docs/METAL_TIMELINE.md` section 1.1 records that "the only device timer
  implementations present in the shipped binaries are the CUDA and HIP ones".
  So this is an upstream gap in MAX rather than a hardware limit, and
  `docs/design/UPSTREAM_MAX_METAL_GAPS.md` is the report.

So `phase_profile.mojo`'s statement that "per-kernel device time is not
reachable on this machine by any route" is correct as of the recorded probes,
and the correct reading of it is narrower than "we cannot time the device". It
means there is no *in-process* per-kernel timer. There is an out-of-process one
and it works.

### 5.3 What instrumentation IS possible here, and what each costs

**A. A Metal System Trace against the device-owned plane. Cost, one capture.**
This is the single largest gap in the repository's instrumentation and it is not
a missing capability, it is a capture nobody has taken. Every trace in
`bench/results/metal_timeline_2026-08-15/` predates the resident plane becoming
the default and was taken on the host-stepped loop. `grow_tree_device_resident`'s
own docstring says "what this plane needs instead is the Metal timeline that
motivated it", and that timeline has never been pointed at it. The trace gives
per-dispatch GPU start, end and duration from the GPU's own clock, the commit
timestamp, the completion timestamp, and the hardware's own Active and Idle
record. It needs no event, no fence, and no library edit. The measured cost is
14 percent of wall clock (`gpu_train_s` 2.070 s untraced against 2.363 s traced)
which does not distort the GPU-clock timestamps, a 119 MB trace for 4.11 seconds
of process life, and root is not required. **This is what breaks the plane's
nine per-step launches into named phases, and it is the only thing that will.**

**B. Kernel labels. Cost, one string per launch site.** METAL_TIMELINE section 3
records that MAX sets no labels and pushes no debug groups, so all 22,109
dispatches in a capture are called `Compute Command 0` or `Blit Command 0` and
"which kernel ran is recoverable only by position and duration". For the plane
that is fatal, because nine kernels per step with four of them at `grid_dim=1`
are nine anonymous dispatches whose only distinguishing feature is duration, and
three of them will have nearly identical durations. A `pushDebugGroup` or an
encoder `label` at the launch site, or the equivalent upstream in MAX, makes
every future capture name its kernels. **Ranked against every other
instrumentation option available, this is the highest value per unit of effort
in the repository**, and it is a prerequisite for A being decisive rather than
suggestive on the plane.

**C. A `resident_schedule_launches` beside `oblivious_schedule_launches`. Cost,
one pure function.** `train_gpu.mojo` already names this as the follow-up, in
the comment "The follow-up that closes it is a `resident_schedule_launches`
beside the oblivious one, owned by whoever owns that file". Today the plane
charges `dispatches=0`, and section 2.2 of this document exists only because a
human read the loop. A pure function of `num_leaves` and the speculation flag,
called by the bracket the way the oblivious bracket calls its counterpart, turns
that reading into a report. It buys no time. It buys the ability to notice when
the plane's launch structure changes, which nothing today can.

**D. Widen `backend_proof.py`'s parse to keep the `rows` and `slots` columns.
Cost, four lines of Python.** The report already emits
`phase_profile phase histogram all <calls> <dispatches> <syncs> <rows> <slots> <cells> <nanos> ...`
and the parser reads only the `calls` field for three phase names. Keeping
`rows` per phase would put the row-work question of section 4.3 into every
harness record permanently, at no run-time cost, because the counters are
already being incremented.

**E. `MTL_CAPTURE_ENABLED=1` with `_start_metal_trace_capture`. Cost, a call
site, and it is the wrong tool.** The installed MAX toolchain ships
`max/mojo/max/gpu/host/_metal_capture.mojo` over
`_AsyncRT_DeviceContext_startMetalTraceCapture`. It writes a `.gputrace` that
only the Xcode Metal debugger opens, so nothing can be scripted off it. Right
for a question about what happens inside one kernel, wrong for a question about
what happens between kernels.

**F. `PROFILE_FENCED`, priced honestly.** The module docstring says `fenced`
"pays two extra host synchronizations per split" and calls that "a real
perturbation of the schedule". The number is worth attaching. On the
host-stepped loop at 30 splits that is 60 extra blocking readbacks per tree, at
a measured 606 microseconds each, which is **36 ms per tree, roughly doubling
the tree at this shape**. A `fenced` profile of this arm is not a perturbed
measurement of the shipped run, it is a measurement of a different workload. On
the device plane `fenced` does nothing at all, because the plane takes no
profile, which is a fact worth stating because the mode reads as if it applied
everywhere.

**G. What is not possible and should stop being proposed.** Any in-process
per-kernel timer. There is no event, no fence, no second stream, no completion
callback, no queue or command-buffer handle, no completion query, and no
pollable device memory. The only in-process instrument is host wall time closing
over `synchronize()`, and each use of it costs 606 microseconds.

## 6. The depth-wise result explained

### 6.1 The speed half

The mechanism is not the wait count, and the wait count is the reason the
result looked anomalous. Restating the measured facts of section 4.

- The shipped leaf-wise arm makes **2** blocking readbacks per tree, not 30. It
  is the device-owned plane, and it is on by default.
- The depth-wise arm makes **7**, five more, worth an estimated 3.03 ms per tree
  at the measured 606 microseconds each.
- The depth-wise arm also issues 114 fewer launches, worth an estimated 1.66 ms
  per tree of host enqueue at the measured 14.58 microseconds each, which is
  less than what the extra readbacks cost it. Net, the depth-wise arm's host
  floor is an estimated 1.36 ms per tree **worse**.
- It wins by 4.14 ms per tree anyway.

So depth-wise wins on the device, and the device account is

1. **An estimated 0.6 ms per tree** from the plane's 93 extra single-threadgroup
   control dispatches and its 25 extra search launch overheads. The plane moved
   the pick, the commit, the tree write and the record filing onto the GPU so
   the host would not have to wait for them, and every one of those is a
   dispatch that occupies one threadgroup of a ten-core device. Depth-wise gets
   the same decisions for free in host arithmetic, and pays for the privilege
   with one wait per level rather than one per split. **At five levels that
   trade is cheap. At thirty splits it would not be, which is exactly why the
   plane exists and exactly why leaf-wise cannot simply adopt the loop.**
2. **An estimated 4.9 ms per tree** unattributed by any fixed cost, which is row
   work, and section 4.3 quantifies it from a stored profile rather than
   guessing at it. A 31-leaf leaf-wise tree on this generator has a measured
   row-weighted mean leaf depth of **6.395** and accumulates a measured
   **2.306 N** rows per tree. A 31-leaf depth-wise tree has an exact mean depth
   of **4.94** and, carrying the measured 20.4 percent split imbalance across,
   an estimated **2.01 N**. So depth-wise walks an estimated **23 percent fewer
   rows in the partition** and accumulates an estimated **13 percent fewer rows
   in the histogram**, and it does so because filling levels is the shallowest
   way to spend a fixed leaf budget while leaf-wise growth on this data is
   markedly unbalanced. **Depth-wise is faster here because it grows a shallower
   tree, not because it waits less. It waits more.**

There is a second-order point worth recording. The plane's justification is
three measurements in `bench/results/session3_2026-08-16/RESULTS.md`, at 50,000,
250,000 and 1,000,000 rows, and all of them were taken at **50 features**. At 50
features `row_tile_floor(80, 50)` is 2 and the histogram runs at two row tiles,
so compute is a smaller share of the round and the control plane a larger one.
At 100 features the tile count collapses to 1, the histogram gets slower per
row, and compute takes over. **The plane's default was licensed by measurements
at a feature count where its mechanism pays most, and this run is the first
measurement at a feature count where it pays least.** That is not a defect in
the plane, it is a missing crossover rule, and section 7 R3 is where it belongs.

### 6.2 The accuracy half

The accuracy result is not an anomaly either. It sits on a monotone trend across
four growth policies measured in the same run, and the trend has a mechanism in
the generator.

Ordered by growth **breadth**, meaning how uniformly a tree spends its split
budget across the feature space rather than concentrating it.

| growth policy | arm | rmse | breadth |
|---|---|---|---|
| symmetric, one split per level applied to every leaf | mojotrees_catboost_mode gpu | 0.30327 | maximal |
| symmetric | catboost cpu | 0.30347 | maximal |
| depth-wise, every leaf of a level split | mojotrees_depthwise gpu | 0.30737 | high |
| depth-wise | mojotrees_depthwise cpu | 0.30740 | high |
| leaf-wise, best gain anywhere | mojotrees gpu and cpu | 0.30778 | low |
| leaf-wise | lightgbm cpu | 0.31103 | low |

**Accuracy improves monotonically with breadth on this dataset**, across two
engines, two backends and three growth policies, and every arm is bit-stable
across its five repeats. Depth-wise beating leaf-wise is one step of that
ladder rather than an isolated coincidence.

The mechanism is in the target. `bench/real_data/generators.py`,
`dense_regression`, builds

```
signal = 3.0 * x0
       + 2.0 * x1 * x2
       + 1.5 * sin(6.0 * x3)
       - 2.0 * (x4 > 0.7)
       + 0.8 * x5 ** 2
y = signal + 0.30 * normal
```

over 100 features, so 94 of them are pure noise, and the docstring says outright
that "the signal is deliberately not a sum of univariate terms. A model that
only ever finds axis-aligned marginal structure scores visibly worse here".

Four of the five signal terms reward breadth and only one rewards depth.

- `2.0 * x1 * x2` is a pure interaction. It can only be captured by a tree that
  splits `x2` beneath an `x1` split, in every `x1` cell. Depth-wise guarantees
  that, because every leaf of every level is split, so an `x1` split at one
  level has `x2` splits under both of its children at the next and the `(x1, x2)`
  grid forms. Leaf-wise compares gains globally, and if both children of an `x1`
  split offer more gain on `x0` than on `x2`, the grid never forms in that
  branch.
- `1.5 * sin(6.0 * x3)` covers roughly two periods over the feature's range, so
  it needs several thresholds along `x3` inside every region, which is breadth
  again.
- `0.8 * x5 ** 2` needs a few thresholds along `x5` everywhere.
- `3.0 * x0` is a strong marginal, so it dominates the gain ranking. Under
  leaf-wise growth it wins repeatedly, which is exactly how the budget gets
  spent on one feature.
- `-2.0 * (x4 > 0.7)` is the one term that rewards a single deep, precisely
  placed split, and it is the only one leaf-wise growth is better suited to.

So on this target leaf-wise growth's global gain ranking is a trap. It spends 31
leaves chasing the largest marginal, `x0`, and the step in `x4`, and leaves the
interaction and the two curvature terms underfit in whichever branches did not
win the ranking. With 100 trees at a learning rate of 0.1 the ensemble is not
converged, so coverage per tree matters and depth beats nothing.

**The difference is a property of the shape, and it is small.** 0.30737 against
0.30778 is 0.135 percent relative, and the accuracy budget in
`docs/design/ACCURACY_BUDGET.md` and the standing 1 percent directive both
place it far inside noise for a decision. What it is *not* is evidence that
depth-wise growth is generally more accurate. This is one synthetic dataset with
94 percent noise features and a deliberately interaction-heavy target chosen to
punish marginal-only models. The same run's real-data companion at 463,715 rows
by 90 features is not in this file and would have to be read before any such
claim.

One incidental check, because it looks like it should mean something and does
not. The GPU depth-wise arm reads 0.30736693 against the CPU depth-wise arm's
0.30740265, so the GPU is very slightly better. That is the Float32 device gain
comparison resolving a near-tie differently, as
`_grow_tree_gpu_device_search`'s docstring says it may. It is a coincidence, not
a finding.

## 7. The ranked plan

Ranked by expected value, which is the estimated payoff times the probability it
survives measurement, not by ease. Every payoff is an estimate and its
arithmetic is shown. Every item states its bit-identity risk.

Three implementation lanes currently own `histogram_gpu.mojo` and
`gpu_leaf_batching.mojo`, `gpu_resident_round.mojo` and `train_gpu.mojo`, and
`gpu_split_search.mojo` and `gpu_active_rows.mojo`. This plan is advisory and is
written so each item names the file it lands in.

### R0. Close the row-work residual. No code, no build, one interleaved run.

**Payoff in milliseconds, zero. Payoff in attribution, the whole of section
4.3, and a registered prediction to test it against.**

Section 4.3 already half-answers this from a stored profile, and it registers
its prediction here before the run rather than after. **At 799,110 by 100, with
both arms on `_device_search_resident`, the depth-wise arm's `histogram` row
count should be about 13 percent below the leaf-wise arm's and its `partition`
row count about 23 percent below.** If the run says otherwise, section 4.3 is
wrong and the residual of 4.2 is something nobody has found yet.

`bench/bench_train_gpu.mojo` already parses `gpu` and `gpu-depth` as arms
(`ARM_DEPTHWISE`, and `_arm_name` appends `-depth`), interleaves them in one
process, and prints a phase profile per arm per repeat. Run

```
MOJOTREES_GPU_TREE_RESIDENT=0 MOJOTREES_PHASE_PROFILE=async \
  ./build/bench_train_gpu 799110 100 reg 5 gpu,gpu-depth
```

`MOJOTREES_GPU_TREE_RESIDENT=0` forces the leaf-wise arm onto
`_device_search_resident`, which is where depth-wise already is, so both arms
are on the same grower, both are fully instrumented, and the only difference
between them is the growth order and the batch width. Then read

```
phase_profile phase histogram all <calls> <dispatches> <syncs> <rows> <slots> <cells> ...
```

The `rows` field is exactly `n_root + sum of built_rows`, charged at
`_enqueue_resident_split`'s `profile.charge(PROF_HISTOGRAM, built_rows, ...)`.
Comparing the two arms' `rows` answers, with no inference at all, whether
depth-wise accumulates fewer rows and by how much. The same run's wall clocks
answer, with both arms on one grower, what the batch width alone is worth.

Two cautions. The `rows` field is in stdout, and `run.py` parses the profile and
discards the text, so this must be the Mojo bench binary and a grep rather than
a harness run. And the env var is read per tree, so it cannot be an interleaved
arm; it applies to both arms of the process equally, which is what makes this
comparison valid.

Bit-identity risk, none. `MOJOTREES_GPU_TREE_RESIDENT=0` is a documented A/B
handle and `tests/test_gpu_tree_resident.mojo` asserts the plane and the loop
produce node-identical trees over six configurations with no tolerance
anywhere.

**This is item zero because every estimate in R1 through R3 is priced against a
residual that R0 converts into a measurement.**

### R1. Attack the redundant row-index and gradient re-read in the histogram kernel.

**Estimated payoff 0 to 7 ms per tree, which is 0 to 19 percent, the largest
number on this list and the widest range, for a reason the device cannot
resolve.**

Mechanism, read off the source. The atomic histogram grid is
`(ceil(n_active / GROUP), n_tiles)`. At this shape `GROUP` is 2 and `n_tiles` is
1, so 50 threadgroups each own two features and each walks every row of the
window. Per row, per block, the kernel reads a 4-byte row index and an 8-byte
interleaved fixed-point gradient and hessian pair, plus one 1-byte bin per
feature it owns. So per row the pass loads `50 * 12 = 600` bytes of *re-read*
against 100 bytes of bins that are read once. **Eighty-six percent of the load
traffic in the kernel that dominates the run is the same twelve bytes fetched
fifty times.**

Arithmetic. A root pass at 799,110 rows loads `799110 * 700` bytes, which is 560
MB, against a unique footprint of `799110 * 112`, which is 89.5 MB, far past any
cache on this part. Section 4.3 measures a leaf-wise tree at **2.306 N**
accumulated rows, so a tree loads `2.306 * 799110 * 700` bytes, which is **1.29
GB**. Raising `GROUP` from 2 to 8 cuts the per-row figure from 700 to
`100 + 12 * 12.5 = 250` bytes, a 2.8x cut, so an estimated **0.83 GB less
traffic per tree**. At the machine's rated 120 GB/s that is **about 7 ms per
tree if the kernel is bandwidth-bound and near zero if it is not**, and
`docs/METAL_TIMELINE.md` section 5.3 establishes that which of those is true
cannot be measured on this device, because the M4 exposes exactly one
Instruments GPU counter and it is `RT Unit Active`.

This is the same conclusion `docs/design/REDTEAM_SPEED_PLAN.md` section 2.4
reached at 50 features, where the re-read factor is 25x. At 100 features it is
50x and the case is stronger, not weaker.

The complication, stated rather than hidden. `histogram_shared_bytes(bin_cap, group)`
is `group * bin_cap * 12`, so `GROUP = 8` at a 256-bin capacity costs 24 KiB of
threadgroup memory, which fits a 32 KiB budget but drops residency from an
estimated 5 blocks per core to 1, and the block count from 50 to 13 on a 10-core
device. `free_feature_group`'s docstring says exactly this, that anything wider
than the rule returns "is not free" and "trades resident blocks for row-side
traffic", a trade it marks UNMEASURED. The clean version narrows the cell first.
Under a
declared `constant_hessian`, which is true for squared error and already a flag
that `enqueue_range_histogram` uses to compute `hist_planes = 2 if
self.constant_hessian else 3`, the threadgroup footprint could be two planes
instead of three. That alone raises the free rung at 256 bins only from 2 to 3,
which is not a ladder rung, so a real gain needs the count plane packed as well.

Bit-identity risk, **none for the group width**. Widening `GROUP` changes which
thread accumulates which cell and nothing else; accumulation is fixed-point
Int32 and integer addition is associative and commutative, which is the argument
`_range_hist_atomic_kernel` already makes for its forty instantiations agreeing
with each other. The ladder is already instantiated at 1, 2, 4, 8 and 16.
Narrowing the cell under a declared constant hessian also moves no bit, but the
declaration is load bearing and `set_const_hessian`'s docstring already warns
that "declaring it on a round where it is not true produces a wrong hessian
plane, silently".

**The first step costs nothing to build.** `MOJOTREES_GPU_FEATURE_GROUP` and
`set_feature_group` are both reachable, so an interleaved A/B of group 2 against
4 and 8 at 799,110 by 100 in one process is available today. Lands in
`gpu_active_rows.mojo` and `gpu_tiling.mojo`.

### R2. Raise the row-tile count while staying on the atomic strategy.

**Estimated payoff 0 to 4 ms per tree, with a real chance of a regression, and
its true value is that it is the missing counter.**

Mechanism, read off the source. `row_tile_floor(target_blocks, n_slots)` returns
`ceil(target_blocks / n_slots)`, which at `target_blocks = 80` and
`n_slots = 100` is **1**. So at 100 active features every histogram in the fit,
including the 799,110-row root, runs at one row tile and each block's 256 lanes
walk 3,122 rows serially. That is the corner `row_tile_floor`'s own docstring
names, that "at any feature count at or above 80 the term collapsed to 1 and a
node got a single tile whatever its row count, so one threadgroup per feature
scanned the whole node".

**The known refutation of the tile floor is confounded and does not close this.**
`MOJOTREES_GPU_MIN_TILES=device` raises `n_tiles` to `target_blocks`, and at
`n_tiles > 1` `STRATEGY_AUTO` flips to `STRATEGY_TILED`, which allocates
`n_tiles * n_features * n_bins` partial cells and adds a reduce kernel. The
measured loss is therefore a measurement of the tiled strategy and its partial
traffic, not of the tile count. `docs/design/REDTEAM_SPEED_PLAN.md` and
`bench/results/gpu_window_2026-08-16` both record the loss without separating
the two. The device plane forces `STRATEGY_ATOMIC` explicitly in
`enqueue_desc_histogram`, and `resolve_tiling` skips the `tiles_by_memory` clamp
for `STRATEGY_ATOMIC` (`if strategy != STRATEGY_ATOMIC and n_tiles > tiles_by_memory`),
so **more tiles under atomics has never been measured.**

Why it is ranked here rather than lower. More tiles does not reduce traffic, it
raises parallelism, so it pays only if the kernel is latency-bound, while R1
pays only if it is bandwidth-bound. **R1 and R2 together are the counter this
device does not have.** If R2 helps and R1 does not, the kernel is latency-bound
and the layout work in `CLEANSHEET_GPU.md` section 3 is not the prize. If R1
helps and R2 does not, the kernel is at the memory wall and only a layout change
helps. Either answer retires a question this repository has carried since
2026-08-15. Running them as one arm would answer neither, and this repository
has already produced two correct-but-incomplete results by moving two things
between two timings.

Bit-identity risk, none. `resolve_tiling`'s docstring carries the argument, that
"two geometries over the same rows sum the same bins in a different order to the
same value", and the strategy tests assert it bit-exactly.

Cost, zero code. `MOJOTREES_GPU_MIN_TILES=device` with
`MOJOTREES_GPU_HIST_STRATEGY=atomic`, interleaved, or the existing
`row-tiles-N` arms in `bench_train_gpu.mojo` which pass a tile *length* rather
than a floor and are a separate rule. Lands in `gpu_tiling.mojo` if a default
moves.

### R3. Give the leaf-wise arm a feature-count crossover between the plane and the loop.

**Estimated payoff up to 4.1 ms per tree, which is 11 percent, bounded above by
the measured gap, and the cheapest form of it is a default rather than a build.**

Mechanism, measured. Section 4 shows the depth-wise host-stepped loop beating
the leaf-wise device plane by 4.14 ms per tree while paying an estimated 1.36 ms
per tree more in fixed host cost. Section 6.1 shows why the plane's three
licensing measurements do not cover this shape, all having been taken at 50
features where the histogram runs at two row tiles.

The claim to test is narrow and it is not "the plane is wrong". It is that **the
plane's advantage is a function of the feature count**, because the feature count
determines the tile count, which determines compute's share of the round, which
determines whether removing 5 readbacks per tree is worth 93 extra
single-threadgroup dispatches per tree. The measurement is R0's second half,
leaf-wise on the plane against leaf-wise on the loop, interleaved, at 100
features and at 50.

If the loop wins at 100 features, the answer is a shape rule in
`device_policy.crossover_rules()`, which is where this repository puts numbers a
measurement has to license, and not a flat flip. A flat flip would regress
50,000 rows, 250,000 rows and 1,000,000 rows at 50 features, all three of which
are resolved measurements in `session3_2026-08-16/RESULTS.md`.

Two cheaper variants of the same idea exist and should be measured in the same
window because they are already built. `MOJOTREES_GPU_SPECULATION=1` arms the
K=1 prebuild in the plane, which the census measured at a **66.8 percent** hit
rate at 1,000,000 rows against **964 wasted builds per fit**, and it has never
been measured end to end; `SPECULATION_BUILD_VAR`'s docstring registers the
condition it must meet. And `MOJOTREES_GPU_FUSE_PARTITION_TAIL=0` is the control
for the one launch the plane already saves.

Bit-identity risk, **none, and it is asserted rather than argued**.
`tests/test_gpu_tree_resident.mojo` compares the plane's trees against
`_device_search_resident`'s node for node with no tolerance, `value` as bit
patterns, over six configurations, and asserts from the plane's own trace that
the plane produced them. So the two growers are interchangeable by test and a
crossover rule moves no bit. `tests/test_gpu_speculation_build.mojo` does the
same for the speculation over eight rounds.

Lands in `device_policy.mojo` for the rule and `train_gpu.mojo` for the gate.

### R4. Kernel labels, so a capture can name what it timed.

**Payoff in milliseconds, zero. Payoff as a multiplier on R1, R2 and R3,
large.**

See section 5.3 item B. Nine anonymous dispatches per plane step, four of them at
`grid_dim=1` with near-identical durations, cannot be attributed from a trace by
position and duration. One label per launch site fixes it for every future
capture. Ranked above R5 and R6 because without it a Metal capture against the
plane, which is the only instrument that can decompose it, produces a ranked list
of durations with no names on it.

Bit-identity risk, none. A label is metadata.

### R5. Make the plane's descriptor grids descriptor-sized.

**Estimated payoff 0.01 to 0.05 ms per tree at this shape. Listed so that nobody
spends a lane on it.**

Section 2.3 bounds the surplus from the plane's fixed `(241, 13)` partition grid
at an estimated 10 to 50 microseconds per tree, and shows that the histogram
grid does not change at all because `n_tiles` is 1 either way. The intuition that
the plane's inability to size its grids is expensive is **wrong at this shape**,
and the arithmetic is in 2.3 so it does not have to be rediscovered. It becomes
worth something at fewer features, where `n_tiles` rises and the surplus is paid
on every small node, and even there it is a fraction of a percent.

Bit-identity risk, none. `enqueue_partition_desc`'s docstring carries the
argument, that `_scatter_kernel` derives each row's destination "from the routing
flags and from the row's position in the range, and from nothing else", so a
grid sized for a longer range produces the same permutation.

### R6. The 306 KB unconditional slot zeroing per build.

**Estimated payoff 0.05 to 0.08 ms per tree. Listed to be closed, not opened.**

Every histogram build zeroes `3 * n_features * n_bins` Int32, which is 76,500
cells or 306 KB, regardless of the node's row count. At 31 builds per tree that
is 9.5 MB written per tree and 950 MB per fit, which at 120 GB/s is an estimated
0.08 ms per tree. It also cannot simply be dropped, because the atomic strategy
folds with `Atomic.fetch_add` and needs a zero start, and the plane forces the
atomic strategy. Narrowing the cell under R1 cuts it by a third for free. There
is nothing else here.

### What is deliberately absent

**Sibling subtraction in the oblivious level batch.** A lane owns it. It bears on
neither the leaf-wise nor the depth-wise device path, because both already
subtract, `_enqueue_resident_split` through
`enqueue_resident_leaf_subtracting` and the plane through `enqueue_desc_child`.
It is accounted for here only to bound what it can be worth. The symmetric arm
builds 126 node histograms per tree with no subtraction, which is every row once
per level across 6 levels, an estimated `6N` accumulated rows against
depth-wise's estimated `3N`. Restoring subtraction should therefore roughly halve
the accumulate. The arm measures 17.07 s against depth-wise's 3.24 s, so halving
the accumulate takes it to an estimated 10 s at best and leaves it 3x behind.
**Subtraction alone does not explain the symmetric arm's 5.3x deficit and should
not be expected to close it.** The rest is presumably the level-batched
multi-leaf kernel and the 56 command buffers per tree, and it is not this
document's question.

**Anything justified by the 76.5 percent idle figure.** See section 0. At this
shape the shipped leaf-wise arm's host-side floor is an estimated 15 percent of
its wall clock. Control-plane work here is priced against 5.34 ms per tree, not
against 76.5 percent of 36.59.

## 8. Corrections this lane found in standing claims

Recorded because each is a claim a reader will otherwise inherit, and none is
fixed here.

1. **`bench/real_data/scenarios.py`'s `MOJOTREES_DEPTHWISE` dict does not
   describe the arm that produced the depth-wise numbers.** It is an
   uncommitted working-tree addition postdating the run. The record says one key
   changed. See 1.2.
2. **`phase_profile.PARTITION_LAUNCHES = 4` is stale.** The host arm issues
   three. See 3.1.
3. **`gpu_tiling.LAUNCHES_ATOMIC = 1` under-counts.** The atomic path issues a
   zeroing kernel and the accumulate, so two. See 3.1.
4. **`_device_search_resident`'s docstring prices a leaf-wise split at eight
   launches and a depth-wise level at `6L + 2`.** The reading gives five and
   `5L + 2`, both one lower per partition, and its "5 waits for a tree" omits
   the root download, so the measured figure is 6 tree waits plus the round's
   scale readback.
5. **The 76.5 percent GPU idle figure is stale for the shipped leaf-wise arm.**
   It was measured on the host-stepped control plane at 200,000 by 50.
6. **No Metal System Trace has ever been taken against the device-owned plane**,
   although the plane's own docstring names that as what it needs. Every capture
   in `bench/results/metal_timeline_2026-08-15/` predates it.
7. **Crossing 80 active features on a 10-core M4 silently changes the histogram
   strategy from tiled to atomic**, because `row_tile_floor` returns
   `ceil(sm_count * TARGET_BLOCKS_PER_SM / n_slots)` and `STRATEGY_AUTO`
   resolves on whether that is greater than 1. Confirmed in both directions by
   stored dispatch counts, two launches per build at 50 features and one at 100.
   No parameter, no docstring and no report says so, and it means a benchmark at
   50 features and one at 100 are measuring two different kernels. See 2.3.
8. **CLOSED 2026-08-17.** This item read
   "`gpu_tree_tables.tree_resident_requested` still spells the
   `MOJOTREES_GPU_TREE_RESIDENT` default as `== "1"`" while
   `gpu_resident_round.resident_round_enabled` spells it `!= "0"`, and warned
   that R0 and R3 both turn that variable so a caller reaching the wrong
   predicate would get the pre-flip answer. `gpu_tree_tables` now owns both
   questions over one `comptime TREE_RESIDENT_VAR`, `tree_resident_enabled`
   spelled `!= "0"` and `tree_resident_explicitly_requested` spelled `== "1"`,
   and `tree_resident_requested` is a one-line deprecated alias for the second
   that reads no variable of its own. R0 and R3 turn one gate with one default.
