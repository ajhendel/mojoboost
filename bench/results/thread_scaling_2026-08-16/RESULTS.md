# Thread scaling, and a correction about which LightGBM we are racing

Lane `lane/cpu-thread-scaling`, worktree `wt19`, branched from `fb356d9`.
Taken 2026-08-16 under `/tmp/mojotrees-bench.lock`, `mode: timing`, bracketed
by `bench/bench_canary.mojo`.

Raw transcripts, saved before any of this was read:

- `main.txt` -- the partition block, ten arms, twelve repeats, one process.
- `phase_profile.txt` -- `MOJOTREES_PHASE_PROFILE=async`, one fit at one
  worker and one at auto, immediately after the block and inside the lock.
- `canary_start.txt`, `canary_end.txt` -- the window's two readings.

---

## 0. What survives the window's thermal state, and what does not. Read first.

**This window heated badly and it ended nine percent slower than it started,
on the canary's minimum as well as its median.** That is not "noisier"; the
box itself was slower at its best. So every claim below is sorted here before
it is made, and the sort is by *what kind of ratio it is*, not by how much I
like it.

**SURVIVES -- ratios taken inside one process with the arms interleaved
repeat by repeat.** Drift is common to numerator and denominator.

- Every comparison between mojotrees arms, and between an arm and LightGBM,
  in block one. These are paired per-repeat with win counts.
- `gap_w1 = 1.048`, eight of eight repeats, spread 6.8 percent. The
  one-worker arms drifted 2.0 percent end to end; this cell is clean.
- **The per-node-class decomposition of the histogram**, which is section 7's
  central finding, because the classes are interleaved through the fit by
  construction: every tree visits a root, a large, a medium, a small and a
  tiny node, so a per-class *ratio within one fit* has the fit's drift in both
  halves.
- **The class-against-class scaling ratio**, e.g. "the root gets 2.36x and
  small nodes get 1.22x", to the extent it is quoted as the ratio **1.93**
  between them. That number is a ratio of ratios in which any per-fit
  multiplicative drift cancels *exactly*, and the arithmetic is shown in
  section 7.

**DOES NOT SURVIVE -- needs a fresh window on a cool box.**

- `gap_auto = 1.321`. It is **consistent, not resolved** under M0: eight of
  eight in direction, but one plateau repeat came in at 1.011 against a body
  between 1.29 and 1.42, and the arm's own spread is as large as the effect.
  **Do not quote 1.321 as a number.**
- The **ratio of ratios, 1.262**, inherits that and is quoted only as
  agreeing in magnitude with the 1.298 this lane was briefed with.
- Every **absolute second** in this file, most of all the auto ones. The auto
  arms nearly doubled end to end.
- The per-class **absolute** scaling figures (2.358, 1.224, ...) taken singly,
  because each divides one whole fit by another whole fit. `cpu_round1`
  established that such a ratio is worthless below about 1.2x and that rule
  is respected here.

**NOT MEASURED AT ALL.** Block two, the fan-out arms, never ran. Section 6.

---

## 1. The correction, which is the largest finding here and is not a number

**The comparator does not run the histogram builder this campaign has been
describing.** Measured directly, by training `stock+det` for two rounds at
`verbosity=1` on this exact dataset at both thread counts:

```
[LightGBM] [Info] Auto-choosing col-wise multi-threading, ...   (num_threads=1)
[LightGBM] [Info] Auto-choosing col-wise multi-threading, ...   (num_threads unset)
```

Both cells. Column-wise.

### What column-wise actually is, with citations

Read from LightGBM at `bdf3704` (`VERSION.txt` 4.7.0.99); the installed
comparator is the 4.7.0 release and the checkout is a depth-1 clone, so the
two could not be diffed. The multi-threading in these files has been stable
across the 4.x line.

`Dataset::ConstructHistogramsInner` (`src/io/dataset.cpp:1318`) branches on
`share_state->is_col_wise` at line 1323. On the column-wise side:

- **One `#pragma omp parallel for schedule(static)` over used dense feature
  groups** (`src/io/dataset.cpp:1385-1386`).
- Each iteration `memset`s **its own** destination slice at
  `hist_data + group_bin_boundaries_[group] * 2` and accumulates directly
  into it (`src/io/dataset.cpp:1439-1450`).
- **There are no private buffers and there is no merge.** The destinations
  are disjoint by construction, so there is nothing to reduce.
- `enable_bundle=false` is in `LIGHTGBM_ALIGNMENT`, so
  `Dataset::Construct` keeps `OneFeaturePerGroup(used_features)`
  (`src/io/dataset.cpp:355`) and never calls `FastFeatureBundling`
  (guarded at line 367). The run's own log confirms it: `Total Bins 25500`
  over 100 features at `max_bin=255`. **So every group is exactly one
  feature: 100 dispatch units, one feature over all of the node's rows.**
- The inner kernel is `DenseBin::ConstructHistogramInner`
  (`src/io/dense_bin.hpp:98-140`): `grad[ti] += ordered_gradients[i];`
  and `hess[ti] += ordered_hessians[i];` as **two scalar adds**, with
  `hist_t` a `double` (`include/LightGBM/bin.h:34`) and `ti = bin << 1`, so
  the cell is the same 16 bytes ours is. It is prefetched only on the
  indexed path, at `pf_offset = 64 / sizeof(VAL_T)`
  (`src/io/dense_bin.hpp:110-111`).

### What the private-buffers-and-merge description is actually about

It is the **row-wise** builder, `MultiValBinWrapper::ConstructHistograms`,
and it is correct about it:

- Row blocks with a private histogram each, into one flat `hist_buf` sized
  `n_data_block_ * num_bin_aligned_ * 2`
  (`src/io/train_share_states.cpp:211-222`). **Block 0 is not private**: it
  writes `origin_hist_data_` directly, and only blocks `1..n-1` get scratch
  (`include/LightGBM/train_share_states.h:183-191`).
- The block count is
  `Threading::BlockInfo(num_threads_, num_data, min_block_size_)`
  (`include/LightGBM/train_share_states.h:63`), which is
  `min(num_threads, ceil(cnt / min_cnt_per_block))`
  (`include/LightGBM/utils/threading.h:31-42`). **So it never exceeds the
  thread count, and it is a function of the thread count** -- which is
  precisely the property our `plan_row_block_count` refuses to have, because
  it would make the output depend on `MOJOTREES_NUM_WORKERS`.
- `min_block_size_` is `min(0.3 * num_bin / num_element_per_row + 1, 1024)`
  floored at 32 (`src/io/train_share_states.cpp:48-50`).
- **The merge is partitioned by bin range, and the brief's claim is
  correct.** `HistMerge` calls
  `Threading::BlockInfo(num_threads_, num_bin_, 512, ...)`
  (`src/io/train_share_states.cpp:120-123`) -- so the range boundaries come
  from the same `BlockInfo` helper, at a minimum of **512 bins** per block --
  and each bin block then loops `for (int tid = 1; tid < n_data_block_; ++tid)`
  accumulating into `dst` (lines 179-189). No two threads write one cell, and
  the fold order over blocks is fixed and ascending, exactly as ours is.

### On "how the partition varies with node size": it does not

Asked directly by the brief, and the answer is a null.

- The column-wise partition is `num_used_dense_group` units at **every** node
  size. Nothing in that loop consults the node's row count.
- The **only** node-size adaptivity on the column-wise path is the
  `if (num_data >= 1024)` guard on the ordered-gradient gather
  (`src/io/dataset.cpp:1358, 1368, 1376`).
- The column-wise versus row-wise decision is made **once**, at the root, in
  `Dataset::GetShareStates` (`src/io/dataset.cpp:655-728`) by timing one
  construction of each on the full dataset and keeping the winner
  (lines 694-705). It is never revisited per node.
- Row blocks do vary with node size, but only inside the row-wise builder,
  through `BlockInfo(num_threads, num_data, min_block_size_)`. **That builder
  never runs in this comparison.**

So there is no threshold to find. There is one root-time choice between two
programs, and at this shape it chooses the one with no row blocking at all.

### Why nobody in this repository knew

`bench/real_data/engines.py::_histogram_builder` says so in its own
docstring: since C9 the comparator sets neither force flag, LightGBM 4.7
exposes no getter for the resolved builder, "and the one report of it is a
log line this harness silences with verbosity -1. So the auto case records a
null." The recipe that recovers it is one Python run at `verbosity=1`, and it
is cheap: two rounds.

**A stale statement to fix, in a file this lane does not own.**
`bench/results/PROFILE_PROTOCOL.md` still opens its comparator rule with
"`scenarios.LIGHTGBM_ALIGNMENT` pins `force_row_wise = True`, and that pin
stays". That pin was dropped when the comparator became `stock+det`;
`scenarios.LIGHTGBM_ALIGNMENT` today is five keys and none of them is a
builder. The protocol's sentence is now false, and it is the sentence that
would have told a lane which builder to read.

---

## 2. The window, and what it can and cannot support

`799110 x 100`, regression, seed 0, 100 rounds, 31 leaves, 255 bins. Ten arms
in one process, one repeat of every arm before the next repeat of any of them,
twelve repeats. Machine line as the run printed it:

```
cores=10 (perf 4, eff 6, logical 10) pool=all tasks_per_core=4
max_auto_tasks=40 f64_lanes=2 neon=yes assumed_line=128 assumed_l1d=65536
```

**Box state.** No `mojo` process, no lock held, no suite running, verified
with `ps -Ao comm | grep "mojo$"` immediately before the lock. Load averages
1.55 / 5.23 / 8.15 at the start; the five- and fifteen-minute figures were
still decaying from the compile that built the harness. During the run the
only process above 10 percent was this benchmark.

**This window heated, hard, and the ten-thread arms are the ones that show
it.** Over the run the single-worker arms moved about three percent
end to end while the auto arms moved about fifty. Both engines moved
together, which is what makes the paired reduction the one to read; but a
median over a monotone ramp is a median of the ramp, so:

- the **one-worker** cell is tight enough to quote directly;
- the **auto** cell is quoted as a **paired per-repeat ratio with a win
  count**, and where the M0 rule is not met the word is `consistent` or
  `indistinguishable` and not a number.

**Canary, both readings, minimum to minimum and median to median.** See
section 5. Quoting only one of the two would answer a different question:
they can move in opposite directions, and on the serial lane's window they
did.

## 3. The arms

Nothing in `src/` was changed to take these. Every arm is an environment
setting the policy already reads, so the sweep is a measurement and not a
change with a benchmark attached.

| arm | `ROW_BLOCKS` | `FEATURE_GROUP` | root shape | shape at 12,486 rows |
|---|---|---|---|---|
| `base` | derived | derived | 27 blocks x 13 groups of 8 = **351 units** | 3 x 13 = 39 |
| `noblk` | 1 | derived (4) | 1 x 25 = **25 units** | 1 x 25 = 25 |
| `lgbmlike` | 1 | 1 | 1 x 100 = **100 units** | 1 x 100 = 100 |
| `g16` | derived | 16 | 27 x 7 = **189 units** | 3 x 7 = 21 |

`lgbmlike` is LightGBM's column-wise partition feature for feature: one unit
per feature, no private histogram beyond the output slice, no reduction.

**A registered prediction, written into the harness before the run and left
there.** `lgbmlike` re-walks the node's gradient stream once per feature --
100 passes where `base` takes 13 -- and LightGBM pays exactly those 100
passes. If gradient traffic were what binds, we would already be ahead of the
comparator on it by 7.7x and we are not. So the prediction was that
`lgbmlike` does not win on gradient traffic.

## 4. The determinism contract, checked inside the run that measured it

Two claims, and they are different:

- **Worker invariance.** For every knob setting, the digest at
  `MOJOTREES_NUM_WORKERS=1` and at auto must be **equal**, by `bitcast` to
  `uint64` and integer equality, never a tolerance. This is the property a
  bin-range-partitioned reduction would put at risk, and it is why our fold
  walks blocks in ascending order inside a task that owns a whole slot: no
  partition of the dispatch units can reach the order.
- **Schedule neutrality.** Every arm that changes only how the same units are
  handed out -- the interleave width, the oversubscription factor, the core
  pool -- must digest identically to `base`. Only the two blocking arms are
  exempt, and they are exempt for a stated reason: they change which rows are
  summed together before the fold, which is a different Float64 sequence by
  construction.

## 5. Block one: the partition. Every alternative lost.

Twelve repeats, first four dropped, eight in the plateau. Medians in seconds.

| arm | one worker | auto | scaling (paired median) |
|---|---|---|---|
| **`base`** | **14.778** | **8.627** | **1.704** |
| `noblk` | 16.525 | 9.970 | 1.653 |
| `lgbmlike` | 20.662 | 12.705 | 1.612 |
| `g16` | 15.443 | 8.999 | 1.734 |
| LightGBM | 14.074 | 6.567 | **2.133** |

### The four cells, as paired per-repeat ratios with win counts

Paired, because the window drifted and a ratio of medians would inherit the
drift. The count is repeats out of eight in which the numerator was faster.

| quantity | median | min | max | numerator faster |
|---|---|---|---|---|
| `gap_w1` = `base` / LightGBM at one worker | **1.048** | 1.019 | 1.087 | 0 of 8 |
| `gap_auto` = `base` / LightGBM at auto | **1.321** | 1.011 | 1.424 | 0 of 8 |
| **ratio of ratios** = `gap_auto / gap_w1` | **1.262** | 0.930 | 1.364 | -- |

- `gap_w1` **resolved**: eight of eight, spread 6.8 percent of the median,
  and it reproduces the serial lane's 1.0507 on a different day and a
  different window.
- `gap_auto` **consistent, not resolved** under M0: eight of eight in
  direction, but one plateau repeat came in at 1.011 against a body sitting
  between 1.29 and 1.42, and the arm's own spread is as large as the effect's
  distance from 1.
- The ratio of ratios, **1.262 median**, is the number this lane was given as
  1.298 and it agrees with it. The individual scaling figures do **not** agree
  with the brief's 1.856 and 2.409, and that is the window rather than a
  disagreement: this one throttled so hard that the auto arms nearly doubled
  end to end, which drags both engines' scaling down together and leaves
  their ratio where it was.

### Every partition arm lost, at both worker counts

| candidate | at one worker | at auto |
|---|---|---|
| `base` / `noblk` | 0.893 median, 8 of 8, **resolved** | 0.871 median, 8 of 8, **resolved** |
| `base` / `lgbmlike` | 0.714 median, 8 of 8, **resolved** | 0.683 median, 8 of 8, **resolved** |
| `base` / `g16` | 0.956 median, 8 of 8, **consistent** | 0.973 median, 6 of 8, **indistinguishable** |

Read that top to bottom. Turning row blocking off costs 12 percent serially
and 15 percent in parallel. Adopting **LightGBM's own partition, feature for
feature**, costs **40 percent serially and 46 percent in parallel**. Widening
the interleave past the L1 clamp is a small consistent loss at one worker and
nothing at all at auto.

**The registered prediction held.** `lgbmlike` was predicted not to win on
gradient traffic, and it did not: it is the worst arm in the run despite being
the comparator's own shape.

**So the partition is not the parallel gap, and it is not a gap at all.** Our
`(block, group)` decomposition beats the comparator's column-wise partition by
a wide margin on the same machine, in the same process, at both thread counts.
Whatever the remaining 1.26x is, it is not the shape of the work split.

**One arm to read narrowly.** `g16` is the only arm whose direction is not
consistent at auto, and `apple_cpu_policy`'s own comment invites exactly this
A/B and says the constant is a portable floor rather than this machine's
cache. The answer here is that raising it buys nothing at this shape: the
clamp is not costing us anything worth recovering.

### Determinism: every arm, both worker counts, one digest

```
222091343700048511
```

Ten arms, `bitcast` to `uint64`, integer equality, no tolerance anywhere.
`worker_invariance` identical for all four knob settings; `width_neutrality`
identical. The digest also equals the one
`bench/results/serial_kernel_2026-08-16/` recorded, which is the same branch's
default.

**One result stronger than the contract, and it should not be leaned on.**
`noblk` and `lgbmlike` change the *block count*, which changes which rows are
summed together before the fold, and they still produced the same bits. That
is an empirical fact about this dataset and this shape, not a property of the
code: the contract is worker invariance at a fixed block count, and it is the
only thing this run establishes. The mechanism was not chased.

### What the window did to itself

- `w1_base` over twelve repeats: 14.302 to 14.899, spread 2.0 percent of the
  plateau median.
- `wA_base` over twelve repeats: 5.953 rising monotonically to 9.611, spread
  **37 percent** of the plateau median.
- LightGBM's auto arm did the same thing by the same factor, 4.537 to 6.950.

Both engines throttled together, which is what makes the paired ratios usable
and the absolute auto numbers not comparable with any other window. Nothing
else was on the box: during the run the only process above ten percent was
this benchmark.

**Canary, both readings.**

| | start | end | change |
|---|---|---|---|
| CPU minimum | 198.320 ms | 215.951 ms | **+8.89%** |
| CPU median | 198.522 ms | 216.708 ms | **+9.16%** |
| GPU minimum | 232.684 ms | 226.990 ms | -2.45% |
| GPU median | 234.309 ms | 227.820 ms | -2.77% |

Both CPU readings moved by the same nine percent, which is a different fact
from the serial lane's window: theirs moved 0.006 percent on minima and 8.1
percent on medians, which reads as "best case unchanged, window noisier".
**This one reads as "the box itself was nine percent slower by the end, at its
best as well as on average".** The GPU probe, which this run never touched,
moved under three percent in the other direction, so the slowdown is the CPU
package rather than the whole machine.

## 6. Block two: the fan-out. NOT RUN.

**This block was built, compiled, smoke-tested and never measured.** The
session was terminated by an account spend limit during the cool-down between
the two blocks. On resuming, `tools/bench_lock.sh status` reported the box
**HELD** by `lane/test-sweep (wt21)` running the full 155-file correctness
suite, load 11.1, eight `mojo` processes, ETA about 40 minutes. A timing
window may not overlap that, so nothing was re-taken.

The arms exist and are selected by name (`two`), so the block is one
invocation away:

```
pixi run -e bench mojo run -I src bench/bench_thread_scaling.mojo 799110 100 12 two
```

which runs `w1_base, w1_tpc1, w1_lgbm, wA_base, wA_tpc1, wA_tpc16, wA_pool,
wA_lgbm` interleaved. It carries its own `base` and `lgbm` cells at both
worker counts, so it is self-contained and no number in it would ever be
compared with a number from block one.

**It is deliberately shorter than block one** -- eight arms rather than ten,
and no `lgbmlike` arm, which was the slowest in the run at 20.7 s a repeat.
Block one put roughly 110 seconds of dense compute into every repeat for
twenty-two minutes and that is what cooked the box; block two is about 72
seconds a repeat, which should hold the thermal ramp down. Whoever runs it
should still bracket it with the canary and quote both readings.

**A note on the lock tool, because it nearly cost a suite.** The first
`status` call reported `ABANDONED ... (pid 59096 is gone)` while eight `mojo`
processes were burning the box at load 11. Thirty-four seconds later the same
call reported `HELD ... (pid 59415 alive)`. The tool was right both times --
the suite lane had a failed acquisition at 18:24:51 and a real one at
18:25:25, and I read the file in between -- but the failure mode it exposes is
real: an `ABANDONED` verdict is not evidence that the box is free, and the
cheap confirmation is `ps -Ao pcpu,comm -r | head` before believing it.

## 7. The phase profile: the histogram is the round, and its scaling is the round's

`MOJOTREES_PHASE_PROFILE=async`, one fit at one worker and one at auto, at
this shape, immediately after block one and inside the same lock. The previous
per-phase table (`bench/results/cpu_round1_2026-08-16/RESULTS.md`) was taken at
`1,000,000 x 50` on an older tree and before the packed histogram cell became
the default, so it could not be quoted here and this replaces it for this
shape. 100 trees, 6,100 nodes, 21,500 dispatches, no syncs.

| phase | serial s | share of serial | auto s | scaling |
|---|---|---|---|---|
| **histogram** | **12.353** | **85.3%** | **6.307** | **1.959x** |
| partition | 1.402 | 9.7% | 0.762 | 1.840x |
| split_search | 0.512 | 3.5% | 0.333 | 1.536x |
| score_update | 0.075 | 0.52% | 0.047 | 1.594x |
| grad_fill | 0.069 | 0.48% | 0.043 | 1.615x |
| subtract | 0.062 | 0.43% | 0.093 | **0.671x** |
| hist_alloc | 0.0006 | 0.004% | 0.0006 | 1.0x |
| **attributed** | **14.474** | | **7.585** | **1.908x** |

Unattributed 0.07 percent serial and 0.17 percent at auto, so the instrument
accounts for the whole round in both.

Three things follow, and the first two are the lane's answer.

1. **The histogram is 85.3 percent of the serial round and scales at 1.96x
   against the whole round's 1.91x.** The round's scaling *is* the histogram's
   scaling. Nothing else is large enough to matter.

2. **Derived bound, and it is a ceiling rather than a prediction.** If every
   non-histogram phase in the auto round vanished entirely, the round would go
   from 7.585 s to 6.307 s, **16.8 percent**. Closing the measured `gap_auto`
   of 1.321x needs **24.3 percent** out of the auto round. So *the whole of
   the rest of the program is not enough*, and essentially all of the
   remaining gap has to come out of the parallel histogram. Restated as the
   target: the histogram's parallel scaling has to go from **1.96x to about
   2.6x**, or its serial cost has to fall.

3. **`subtract` measures 50 percent SLOWER in parallel**, 0.062 s to 0.093 s.
   `cpu_round1` measured the same sign at 0.80x and then retracted the
   interpretation, correctly, because the two arms were separate whole fits
   and a sub-1.2x ratio taken that way is machine state. **These two arms are
   also separate fits, so the same caveat applies and this is not upgraded.**
   What is new is that the effect is larger here and the direction agrees on a
   second, independent window. It is 0.43 percent of the round, so it is not a
   prize; it is an oddity for whoever owns `split.mojo`, and the natural thing
   to compare it against is that **LightGBM has no separate subtract phase at
   all** -- it folds the subtraction, the histogram fix and the split scan for
   both children into one `parallel for` over features
   (`src/treelearner/serial_tree_learner.cpp:517-519`), so its subtract costs
   no fan-out of its own.

The `histogram` phase runs **6,200 dispatches** across 3,100 calls, which is
two fan-outs per node histogram: the zero-and-accumulate over
`(block, group)` units and the fold over slots. Of the fit's 21,500
dispatches, 6,200 are the histogram's. That is what block two exists to
price.

### 7a. The per-node-class breakdown. This is the finding.

The same profile, split by node size class. `slots` is the phase's own work
unit, one (row, feature) accumulate, so **nanoseconds per slot is a rate and
the classes are directly comparable**. `root=all`, `large>1/8`,
`medium>1/64`, `small>1/512`, `tiny<=1/512` of the dataset.

| class | calls | slots | ns/slot at 1 worker | vs root | ns/slot at auto | vs root | share of auto histogram |
|---|---|---|---|---|---|---|---|
| root | 100 | 7.99e9 | **0.394** | 1.00 | **0.167** | 1.00 | 21.2% |
| large | 246 | 5.12e9 | 0.561 | 1.42 | 0.248 | 1.48 | 20.1% |
| medium | 1,349 | 4.63e9 | 0.977 | 2.48 | 0.492 | 2.94 | 36.1% |
| **small** | 1,098 | 0.67e9 | **2.385** | **6.05** | **1.948** | **11.65** | 20.7% |
| tiny | 307 | 0.026e9 | 8.308 | 21.07 | 4.813 | 28.79 | 2.0% |
| all | 3,100 | 18.43e9 | 0.670 | | 0.342 | | |

Two statements, and they are different in kind.

**The first is a decomposition and it survives the thermal problem.** Inside
one fit, a (row, feature) accumulate at a small node costs **6.05x** what it
costs at the root, and at a tiny node **21x**. Every class is visited by every
one of the 100 trees, so the classes are interleaved through the fit and a
per-class ratio inside one fit has the drift in both halves. **The histogram
is not one kernel with one rate; it is a good kernel at big nodes and a bad
one at small nodes, and that is true at one worker before any parallelism is
involved.**

**The second is drift-free by construction, and it is the answer to the
question this lane was asked.** The small-node penalty is **6.05x serially and
11.65x in parallel**, so it gets

    11.65 / 6.05 = 1.93x worse when the fit goes parallel

and that number is a ratio of ratios in which any per-fit multiplicative drift
cancels exactly. It is identical to `2.358 / 1.224`, the root's scaling over
the small class's scaling, which is the same quantity written the other way
round. **Small nodes lose almost half of their relative efficiency the moment
the fit is parallel, and the root loses none of it.**

That is where the parallel gap lives, and it is inside this lane's files.

**Derived bound, and it is a ceiling rather than a prediction.** If every node
class ran at the root's own auto rate of 0.167 ns per slot, the auto histogram
would be **3.082 s against the measured 6.307 s** -- a 3.2 s cut of a 7.585 s
round, **42.6 percent**. Closing `gap_auto` needs about 24 percent. So this
one item is, for the first time in this lane, a target large enough to contain
the whole remaining gap. It is a ceiling because a 4,000-row node cannot have
the root's locality; the useful reading is that the prize is *large*, not that
it is 42.6 percent.

### 7b. Where the small-node cost comes from: NOT established, and one model refuted

The obvious explanation is a fixed cost per node histogram build -- two
dispatches, a plan, a zeroed output -- which would be invisible at the root
and would dominate a 6,000-row node. **I fitted it and it does not hold.**

Charging every class the root's own serial rate of 0.3943 ns per slot and
attributing the remainder to a per-call cost:

| class | calls | avg rows | measured s | at root's rate | residual s | **residual per call** |
|---|---|---|---|---|---|---|
| root | 100 | 799,110 | 3.151 | 3.151 | 0.000 | 0.000 ms |
| large | 246 | 208,096 | 2.871 | 2.018 | 0.853 | **3.467 ms** |
| medium | 1,349 | 34,284 | 4.518 | 1.824 | 2.694 | **1.997 ms** |
| small | 1,098 | 6,102 | 1.598 | 0.264 | 1.334 | **1.215 ms** |
| tiny | 307 | 840 | 0.214 | 0.010 | 0.204 | **0.665 ms** |

A fixed per-call cost would put one number in that last column. It puts four,
spanning 5.2x, and they *fall* as nodes shrink -- the opposite of what makes a
fixed cost visible. So **the excess is not a per-build constant.** It falls
roughly as `rows^0.3`: each 6x drop in node size takes about 1.7x off the
residual.

I had written the fixed-cost reading into this file before checking it against
the three classes I had not used to fit it. It is kept here as a refuted model
rather than deleted, because the check is the point and because the next lane
will otherwise reach for the same explanation.

**What is actually established:** the per-slot rate rises monotonically and
steeply as nodes shrink, at both worker counts, and it degrades nearly twice
as fast in parallel as at one worker. **What is not:** why. The instrument
that would say is a breakdown *within* a histogram build -- gather, zero,
scatter, fold -- which `phase_profile` does not have and which is a
`histogram.mojo` change this lane can make without a timing window, since
counting bytes and dispatches is arithmetic rather than measurement.

## 7c. What I am unsure of, and it is load-bearing

**I cannot tell whether LightGBM's advantage at auto is in its histogram or
outside it.** Everything in section 7 decomposes *our* round. LightGBM's
whole-round scaling is 2.133x against our 1.704x, and there are two shapes
that produce that and this run cannot separate them:

- its **histogram** parallelizes better than ours does; or
- its **non-histogram work** is a larger share of its round and parallelizes
  better, so its whole-round figure is better without its histogram being
  better at all.

**And the same doubt applies to the small-node finding, which is the one I
most want to be true.** Section 7a establishes that *our* histogram degrades
6x per unit work from root to small nodes serially and 11.65x at auto. It does
**not** establish that LightGBM's does not. LightGBM builds a per-node
histogram too, zeroes a 25,500-bin destination per node
(`src/io/dataset.cpp:1440`), and gathers ordered gradients per node
(`src/io/dataset.cpp:1368`); it has its own small-node overhead and this run
measured none of it. What the finding licenses is "here is where our parallel
time goes"; it does not yet license "here is where we lose to them".

Ours is 85.3 percent histogram serially. If LightGBM's is materially less --
and it plausibly is, since it re-reads the ordered gradient array once per
feature and does its subtraction and split scan inside a single feature loop
-- then its round has more parallel-friendly work in it and its histogram may
be no better than ours. Settling this needs LightGBM's own
`Common::FunctionTimer` output, which is behind its `USE_TIMER` build and is
not in the installed wheel. **Until it is settled, "our parallel histogram is
worse than theirs" is an inference and not a measurement**, and the second
possibility would point the next lane at `partition` and `split_search`
instead of at the histogram.

## 8. What this lane could not do, and the exact changes it would have needed

Ownership for this lane is `src/mojotrees/histogram.mojo`,
`src/mojotrees/apple_cpu_policy.mojo`, and bench files it adds. Four things
the evidence points at are outside it.

1. **`pixi.toml` needs one task line.** Every other benchmark has one and
   `tools/check_pixi_tasks.py` prints a note for an entry point no task runs.
   The line, under `[tasks]` beside `bench-serial-kernel`:

   ```
   bench-thread-scaling = "mojo run -I src bench/bench_thread_scaling.mojo"
   ```

   Unlike its two neighbours this one takes **both** worker counts in a
   single invocation, because the number it reports is a ratio of ratios and
   two invocations put a thermal regime between the numerator and the
   denominator. It needs the bench environment for the LightGBM arm:
   `pixi run -e bench bench-thread-scaling 799110 100 12`.

2. **`bench/results/PROFILE_PROTOCOL.md` carries a false sentence.** Its
   comparator rule still says `scenarios.LIGHTGBM_ALIGNMENT` pins
   `force_row_wise = True` "and that pin stays". It does not pin it; the
   alignment table is five keys and none is a builder. A lane reading the
   protocol to find out which LightGBM it is racing is told the wrong one,
   which is how this campaign came to describe the row-wise builder as the
   comparator's.

3. **`bench/real_data/engines.py::_histogram_builder` could stop recording a
   null.** It says LightGBM exposes no getter and the only report is a log
   line the harness silences. That is true of a *training* run, but the
   builder can be recovered out of band for the price of two rounds at
   `verbosity=1`, which is what this lane did. A one-off probe stored beside
   the scenario would put the resolved builder back in every record.

4. **The phases that are not the histogram.** LightGBM fuses the sibling
   subtraction, the histogram fix and the split scan for **both** children
   into one `#pragma omp parallel for schedule(static)` over features
   (`src/treelearner/serial_tree_learner.cpp:517-519`), so its subtract costs
   no fan-out of its own and scales with the split search. Ours are separate
   phases with separate dispatches, and
   `bench/results/cpu_round1_2026-08-16/RESULTS.md` measured `subtract` at
   0.80x and `split_search` at 1.22x. Those live in `split.mojo` and
   `tree.mojo`.
