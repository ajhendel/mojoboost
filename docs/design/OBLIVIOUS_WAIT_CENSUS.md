# The symmetric tree's launches and waits, per line

Counted by reading the source on 2026-08-17, at head, for one
`grow_policy = oblivious` tree of `max_depth = 6` grown by
`gpu_resident_round.grow_tree_device_oblivious`. Nothing here was measured on
a run. Every row names the line that issues the thing it counts, so a reader
can check any one of them without re-deriving the whole table.

**THE LINE NUMBERS ARE A SNAPSHOT AND HAVE ALREADY DRIFTED. Read the FUNCTION
names, not the numbers.** Added 2026-08-17, after checking. Several lanes edit
`gpu_resident_round.mojo`, `train_gpu.mojo` and `gpu_split_search.mojo` on the
same day, and every insertion above a cited line moves it. Two examples caught
in this file within hours of it being written: `GpuSplitSearcher._copy_noise`
was cited as `gpu_split_search.mojo:7985` and is at `:8179`, and the per-level
noise staging cited at `gpu_resident_round.mojo:3459` and `:3461` is at `:3624`
and `:3626`. The counts, the call order and the named entry points are what this
census asserts and they are unaffected; a number that does not resolve means the
citation drifted, not that the row is wrong. Only the `_copy_noise` citation has
been refreshed, in the finding below that turns on it. Re-numbering the tables
was declined because they would drift again by the next merge and because a
half-refreshed table is worse than a uniformly stale one.

This file exists because three counts were in circulation and they disagreed:
62 (`oblivious_launch_census(6)`, the registered prediction), 56
(`oblivious_schedule_launches(6)`, the built schedule) and 126 (a claim of
launches per tree that nothing in the source produces). The third is the
node-histogram count, `(1 << (max_depth + 1)) - 2`, charged at
`train_gpu.mojo:2058` and correct as a count of histograms; it is not a launch
count and never was. **56 is the launch count with every switch off and it is
confirmed below, launch by launch.**

**Since 2026-08-17 the shipped default is 55, not 56**, and both numbers are
live rather than one superseding the other.
`MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` became the default that day, and
`oblivious_schedule_launches(6, skip_last_build=True)` returns **55** where
`oblivious_schedule_launches(6)` returns 56. So: **56 is the all-off count, 55
is the count a symmetric depth-6 fit enqueues today.** The single launch is the
last level's two batch launches dropped and one paid back, because that
level's partition can no longer defer its copy-back into a batch that does not
run. Every table below is counted at all-off, which is why they still sum to
56, and the shipped-default deltas are stated where they arise.

## What a command buffer is counted as

One `DeviceContext.enqueue_function`. One `enqueue_copy` is counted
separately, in its own column, because on Metal a copy is a synchronous
full-queue drain in both directions (**measured** by disassembly,
`docs/GPU_PORTABILITY.md` section 6.1) and a launch is not. One
`DeviceContext.synchronize()` is counted in a third column.

The three columns are not addable and section 6.1.1 is why. A round trip, a
host wait for a device answer the host needs before it can decide what to
enqueue, predicts time. A drain is an ordering point and predicts time only
when the queue it drains holds work. That distinction is the whole content of
the noise finding below.

## Per tree, before the first level

| what | site | launches | copies | syncs |
| --- | --- | --- | --- | --- |
| tree feature set | `train_gpu.mojo:1865` `builder.set_features` | 0 | 0 | see note 1 |
| searcher reset | `train_gpu.mojo:1868` `cache.reset_for_tree` | 0 | 0 | 0 |
| root row permutation | `train_gpu.mojo:1939` `builder.begin_tree(bag)` | 1 unbagged | 0 unbagged, 1 bagged | 0 unbagged, **1 bagged** |
| level plan geometry | `gpu_resident_round.mojo:3363` `stage_desc_level_plan` | 0 | **2** | 0 |
| root histogram | `gpu_resident_round.mojo:3367` `enqueue_leaf(0, slot=0)` | 2 to 4, note 2 | 0 | 0 |
| tree tables reset | `gpu_resident_round.mojo:3371` `enqueue_desc_begin_tree` | 1, note 3 | 0 | 0 |
| per-record tables | `gpu_resident_round.mojo:3386` `searcher._copy_tables` | 0 | **1** | 0 |

Note 1. `GpuHistogramBuilder.set_features` (`histogram_gpu.mojo:1404`) returns
before it does anything when the feature set has not changed, which is every
tree of a fit that does not set `feature_fraction`. When it has changed it
ends in `feat_dev.map_to_host()`, and a mapping blocks until the device is
idle. So a column-subsampled symmetric fit pays a full device wait per tree
that an unsubsampled one does not. Not fixed here; recorded because it is
invisible at the call site.

Note 2. `enqueue_leaf` forwards to `GpuActiveRows.enqueue_range_histogram`
(`gpu_active_rows.mojo:8254`), which is one zeroing launch plus either the
partial family and its reduce, or the atomic family. It also runs
`_ensure_quantized` and `_ensure_compacted`, each one launch and each once per
tree because `GpuActiveRows.begin_tree` clears `quant_valid` and
`compact_valid`. `oblivious_schedule_launches` models the whole per-tree
prologue as 7 and that is the term those launches live in.

Note 3. The device-reset arm, `MOJOTREES_GPU_TABLE_RESET` unset, which is one
kernel and no transfer. `set_reset_on_device(False)` makes the same reset five
`enqueue_copy` calls, so that arm costs five more drains per tree.

## Per level, six times

| what | site | launches | copies | syncs |
| --- | --- | --- | --- | --- |
| point the leaf records at the level's slots | `gpu_resident_round.mojo:3422` `enqueue_desc_stage_level_search` | 1 | 0 | 0 |
| draw this level's noise plane | `gpu_resident_round.mojo:3459` `stage_random_score_level` | 0 | 0 | 0, host work |
| upload this level's noise plane | `gpu_resident_round.mojo:3461` `searcher._copy_noise` | 0 | **1 when `random_strength > 0`** | 0 |
| search the level | `gpu_resident_round.mojo:3464` `_launch_oblivious_search` | 2 | 0 | 0 |
| seed the root value, level 0 only | `gpu_resident_round.mojo:3507` `enqueue_desc_seed_root` | 1, once | 0 | 0 |
| commit the level | `gpu_resident_round.mojo:3512` `enqueue_desc_level` | 1 | 0 | 0 |
| partition the whole prefix | `gpu_resident_round.mojo:3542` `enqueue_desc_partition` | 2, note 4 | 0 | 0 |
| build every child of the level | `gpu_resident_round.mojo:3546` `enqueue_desc_level_children` | 2, note 5 | 0 | 0 |

Note 4. `GpuActiveRows.enqueue_partition_desc` (`gpu_active_rows.mojo:7616`)
issues a scan and a scatter, and issues `_copy_back_kernel` as a third only
when `fuse_partition_tail` is off. The oblivious loop turns the fusion on at
`gpu_resident_round.mojo:3348` and the level's batched build discharges the
debt, so it is 2.

Note 5. `GpuLeafBatcher.enqueue_device_plan_batch_fused`
(`gpu_leaf_batching.mojo:2441`) is a zeroing pass and an accumulation pass,
two launches whatever the level's width, and the zeroing pass carries the
partition's deferred copy-back.

**Note 5b, added 2026-08-17. The sixth level does not do this under the shipped
default.** `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` is on unless refused, so at
`max_depth = 6` the last level enqueues no batch at all: these two launches
become zero and note 4's partition becomes 3 rather than 2, because with no
following batch to discharge the debt the partition runs unfused and issues
`_copy_back_kernel` itself. Net one launch fewer per tree, 56 to 55. The rest of
this table is unaffected, and so is every level but the last.

## Per tree, after the last level

| what | site | launches | copies | syncs |
| --- | --- | --- | --- | --- |
| the commit that ends growth | `gpu_resident_round.mojo:3581` `enqueue_desc_level` | 1 | 0 | 0 |
| bring the tables home | `gpu_resident_round.mojo:3592` `download_desc_tables` | 1, the pack kernel | **1** | **1** |

`DeviceTreeTables.download` (`gpu_tree_tables.mojo:3909`) takes the packed arm
by default, which is `_fetch_packed` at `:3833`: one pack kernel, one copy of
the packed buffer into pinned host memory, one `synchronize()`. The
`synchronize` is load-bearing and the docstring there records the measurement
that proved it, a pinned copy read without one returning 64 of 64 stale words.

## The totals

    launches   7 + 6 * 6 + 1 + 2 * 6 = 56    matches oblivious_schedule_launches(6, 64)
                                             ALL SWITCHES OFF
               7 + 6 * 6 + 1 + 2 * 5 + 1 = 55
                                             SHIPPED DEFAULT since 2026-08-17,
                                             matches oblivious_schedule_launches(
                                                 6, skip_last_build=True)
    copies     4 with random_strength = 0
              10 with random_strength > 0
    syncs      1, plus 1 more when the round bags

Both launch counts are confirmed against the schedule function. **56 is the
all-off count and 55 is the shipped one**, the difference being the last
level's batched child build, which is two launches, minus the one launch its
partition then has to spend on its own copy-back. The intended one host
synchronization per tree is confirmed as the only `synchronize()` on the shipped
default arm, and no switch of the four adds or removes a `synchronize()`.

## The finding: six drains per tree that are not in anyone's count

`GpuSplitSearcher._copy_noise` (`gpu_split_search.mojo:8179`) is an
`enqueue_copy`, so on Metal it is a synchronous full-queue drain. The level
loop calls it once per level. It returns immediately when
`random_strength = 0`, which is why it was invisible: the leaf-wise plane's
census was taken on a fit that did not set it.

**The CatBoost-mode default set does set it, and the condition is worth
stating in full because this paragraph got it wrong twice on 2026-08-17 in
opposite directions.** The write is `params.mojo:1755-1756`, inside
`_apply_catboost_mode_defaults`, and it reads

    if not saw_random_strength and not config.is_multiclass():
        config.booster.tree.extra.random_strength = CATBOOST_RANDOM_STRENGTH

so a CatBoost-mode fit that names no `random_strength` and is not multiclass
gets 1.0 on **either** backend. `train_gpu._device_search_unsupported_reason`
delegates to `ExtraTreeParams.device_unsupported_reason`, whose
`random_strength` arm is now `has_categorical and random_strength > 0.0`, so
the device search does not refuse the value either. The shipped symmetric fit
therefore makes six of these drains per tree.

**What this sentence used to say, both times.** It first cited
`params.mojo:1703` and gave the condition as "whenever the user did not name
`random_strength`", dropping the multiclass half; 1703 was never the write in
any case, it is a line of the function's own docstring, and the number is
recorded here only so a reader who finds it quoted elsewhere knows what it was
pointing at. A correction was then drafted saying the opposite, that a
CatBoost-mode GPU fit keeps 0.0 because the write also required
`config.device == CPU_DEVICE`, and therefore that the shipped symmetric GPU fit
makes ZERO of these drains. That was true of the committed source when it was
written and is false at head. The device test was a defect, not a rule, and it
was removed the same day as a bug with no switch, because it made one parameter
string build two different models. `docs/design/RANDOM_STRENGTH_UNITS.md`
section 2 is the record of that removal. The drafted correction survives in
`docs/design/SWITCH_CONTRACT_REPAIRS.md` section 3, which is annotated as
superseded; do not apply it.

**The measured symmetric arms were never affected either way**, which is worth
knowing before anyone re-reads a result on the strength of this. `bench`'s
`MOJOTREES_CATBOOST_MODE` passes `random_strength: 1.0` **explicitly**, so
`saw_random_strength` is True, the mode default is skipped entirely, and every
symmetric arm in every recorded run had the noise on and made these drains.
The only fit the device test ever changed was one that named nothing.

Where they sit is what makes them different from the thirteen per-tree copies
section 6.1.1 measured as a null. Each one is issued after the previous
level's commit, partition and batched child build have been enqueued and
before this level's search, so the host blocks until every kernel of the
previous level has retired. It cannot run one command buffer ahead. The device
therefore idles from the moment the previous level finishes until the host
wakes and enqueues the next search, six times a tree, and the round profile
that measured the GPU idle 76.5 percent of the time is consistent with exactly
that shape. Section 6.1.1's null was measured on copies that drained a queue
holding nothing; these drain a queue holding a whole level.

`MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST=1` collapses the six to one, above the
loop, by giving each level its own search record so that every plane can be
resident at once. The bit-identity argument is at
`gpu_resident_round.OBLIVIOUS_NOISE_HOIST_VAR`.

**Off by default, and as of 2026-08-17 that is a MEASURED NULL rather than an
absent measurement.** This paragraph read "Off by default because the time is
unmeasured", and the docstring at `OBLIVIOUS_NOISE_HOIST_VAR` still says the
same thing, which is stale in that file too. The arm was run that day, on a fit
that does set `random_strength > 0`, and it came in **indistinguishable** in the
registered M0 sense (`bench/results/PROFILE_PROTOCOL.md` M0), which is why it is
the one of the four symmetric arms whose default did not flip while the other
three did. It is not filed under `bench/results/`, so the citation is the
2026-08-17 lane brief.

That null does not withdraw the count, and the distinction matters because the
count is what this section asserts. Six drains per tree, each sitting between a
level's build and the next level's search, is still what the source does. What
the null says is that collapsing them to one did not show up in wall time at the
shape measured, which is a fact about how much of the level's tail those drains
were actually costing and not a fact about how many there are.

## The second finding: one sixth of the histogram work WAS discarded, and is not any more since 2026-08-17

The last level's `enqueue_desc_level_children` builds `2^max_depth` child
histograms that nothing reads, because a leaf at `max_depth` is never split.
Both `grow_tree_device_oblivious`'s docstring and
`oblivious_schedule_launches` recorded this and both declined to skip it, on
the grounds that the batch is what pays the partition's deferred copy-back so
skipping it moves the cost rather than removing it.

**That is true of command buffers and false of work.** Reading the two
kernels:

- the batch is a zeroing pass over `2^max_depth` slots of
  `3 * n_features * n_bins` Int32 apiece, then an accumulation over every
  active row times every active feature;
- `_copy_back_kernel` is one grid-strided pass over `n_active` Int32.

At depth 6 the last level owns 64 of the 126 slots a tree zeroes, so it is
about half of all the zeroing traffic and one sixth of all the accumulation,
against a copy-back that is one pass over the row permutation. The trade the
note declined is one extra command buffer for that.

`MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` takes it. The level's partition
still runs, because `_publish_level_row_ranges` and `update_raw_device` both
read the final leaves' windows, and it runs unfused so that it pays its own
copy-back. The argument that it cannot change a bit is at
`gpu_resident_round.OBLIVIOUS_SKIP_LAST_BUILD_VAR`.

**ON BY DEFAULT SINCE 2026-08-17, and this paragraph said "Off by default
because the time is unmeasured".** Both halves of that sentence are gone. The
time was measured that day: 22.76 s to 18.06 s at 799,110 x 100 x 100 trees,
symmetric depth 6, three interleaved round-robin cycles on an M4, **1.26x**
alone and part of a 2.08x-to-2.20x combined arm with the sibling subtraction and
the wide scan, with rmse 2.439382420 unchanged to nine decimals in every cycle
of every arm. The default was then flipped in the same session under
LANE_RULES rule 5, so the variable is now an escape hatch that restores the
discarded build, spelled `!= "0"`, and it is scheduled for deletion after one
round. The counted prediction in this section and the measured result agree in
direction and roughly in size, which is the point worth keeping: the note this
section corrected had declined the trade on a command-buffer argument that was
true and irrelevant.

## How this composes with sibling subtraction

`MOJOTREES_GPU_OBLIVIOUS_SUBTRACT` (`gpu_leaf_batching.oblivious_subtract_requested`)
makes the level build accumulate only the smaller child of each pair and derive
the sibling by exact Int32 subtraction. **It adds no launch**, so every count in
this file is unchanged by it: the level is two launches either way and
`oblivious_launch_census(6)` is still 62.

**ON BY DEFAULT SINCE 2026-08-17.** This paragraph was written when the arm was
opt-in and spelled the switch `=1`; at head the predicate is `!= "0"` and the
subtraction is what a symmetric fit does unless refused. Measured that day at
799,110 x 100 x 100 trees on an M4: 1.78x, 21.97 s to 12.34 s, with a second
interleaved three-cycle reading of 22.76 s to 14.39 s alone and 10.36 s combined
with the wide scan and the skipped last build, 2.20x. rmse identical to nine
decimals in every arm of every cycle, as the exact-integer identity argument at
`_batch_hist_atomic_subtract_kernel` requires.

What it changes is how many children a row is read for, which is the axis
`SKIP_LAST_BUILD` does not act on, so the two compose exactly. Row builds per
tree, tabulated at `train_gpu.mojo`'s `node_hists` block, which is what
`PhaseProfile.note_nodes` is charged:

    subtract  skip_last   row builds        depth 6
    off       off         2^(d+1) - 2       126
    off       on          2^d - 2            62
    on        off         2^d - 1            63
    on        on          2^(d-1) - 1        31

`skip_last` truncates whichever series is running one level early; `subtract`
halves the width of every level that runs. **Both are the shipped default since
2026-08-17, so the bottom row, 31 row builds at depth 6, is what a symmetric fit
does today.** This sentence read "The two have never been measured on together";
they were measured together that day, in the same interleaved round robin, and
the combined arm with the wide scan came in at 2.20x against the all-off
baseline.

## Two arms that add a wait, recorded so nobody rediscovers them

`MOJOTREES_GPU_PACKED_GRADS=1`. `GpuActiveRows._ensure_quantized` then calls
`_check_stage16_bound` (`gpu_active_rows.mojo:9335`), which copies the
overflow counters home and `synchronize()`s to read them. That is a genuine
round trip, not a drain, and it happens once per tree because `begin_tree`
clears `quant_valid`. It is the second host wait per tree on that arm, and it
lands immediately after the root histogram is enqueued.

`feature_fraction < 1`. See note 1.

## What was checked and found not to be a wait

- `GpuHistogramBuilder.open_resident` and `open_resident_tables`
  (`histogram_gpu.mojo:2525` and `:2718`) reuse an already-open pool and an
  already-open table set. The historical "open_resident per-tree alloc and
  drain" bug, a hundred-round fit allocating a hundred slot pools, is fixed at
  head: the guard is at `histogram_gpu.mojo:2586` and the tables' at `:2752`.
- `GpuActiveRows.begin_tree` unbagged is one `_iota_kernel` and no transfer.
- `enqueue_leaf`, `enqueue_desc_*` and `_launch_oblivious_search` neither
  transfer nor synchronize.
- `_publish_level_row_ranges` is host bookkeeping over the downloaded
  snapshot. It enqueues nothing.
