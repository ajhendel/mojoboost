# Stage-level profile, Apple M4, 2026-08-15

The first stage-level profile ever recorded in this repository. Taken on an
idle machine against the rules committed beforehand in
`bench/results/PROFILE_PROTOCOL.md`, and saved before being interpreted.

Toolchain Mojo 1.0.0 (ed45d567). Apple M4, 10 cores (4 performance, 6
efficiency), 16 GB. Branch `perf-round-2` at the `open_resident` fix. LightGBM
via `bench/bench_lightgbm.py`, same shapes, same 100 rounds, 31 leaves, 255
bins, squared error. Reduction is the median of three with the spread beside
it, per the protocol.

Everything below is seconds of TRAIN time, binning excluded, except where the
binning line says otherwise.

## The headline, 1,000,000 x 50

| arm | before round 2 | now (median) | spread |
|---|---|---|---|
| mojotrees CPU, auto | 11.36 | **6.98** | 10.8% |
| mojotrees GPU | 4.10 | **3.58** | 2.5% |
| LightGBM, 10 threads | 2.80 | 2.86 | n/a |

Round 2 is worth 1.63x on the CPU and 1.15x on the GPU, measured interleaved
in one window. The gap to LightGBM closes from 4.06x to **2.44x** on the CPU
and from 1.46x to **1.25x** on the GPU. The round's own pre-registered
estimates were 7 to 8 seconds and 3.6 to 3.8; both were accurate, which is
worth recording because the previous round's estimates were not.

Binning, not included above: **ours 0.377s against LightGBM's 1.207s**, so we
bin 3.2x faster. That is a column we already win and nobody had noticed.

## Scale, and where the GPU stops being the right answer

| shape | CPU | GPU | LightGBM 10t | GPU vs our CPU |
|---|---|---|---|---|
| 1,000,000 x 50 | 6.98 | **3.58** | 2.86 | 1.85x faster |
| 250,000 x 50 | **1.66** | 1.89 | 1.00 | 0.83x, GPU loses |
| 50,000 x 50 | **0.56** | 1.63 | 0.59 | 0.33x, GPU loses |

Two things fall out.

**Our CPU beats LightGBM at 50,000 rows**, 0.564 against 0.594. That is the
first shape on which this library is faster than LightGBM at anything, and it
is the shape a new user is most likely to try first.

**The GPU carries about 1.5 seconds of fixed cost per fit**, which is what
1.63 at 50k against 3.58 at 1M implies. Below roughly 1,000,000 rows the
device loses to our own CPU, and no kernel work changes that; only removing
fixed cost does.

## Phase attribution, 1,000,000 x 50, 100 trees

Two modes, because they answer different questions and the instrument's own
docstring warns that `async` must not be read as device time: the queue is
asynchronous, so the download is where everything drains and `transfer`
absorbs it.

| phase | async | fenced | fenced share |
|---|---|---|---|
| histogram | 0.062 | 2.417 | **49.3%** |
| transfer | 3.094 | 1.381 | **28.2%** |
| partition | 0.132 | 0.865 | **17.7%** |
| split_search | 0.107 | 0.117 | 2.4% |
| grad_fill | 0.116 | 0.117 | 2.4% |
| wall | 3.518 | 4.905 | |

Fencing costs 1.39 seconds of the 4.90, which is the price of draining the
queue between phases and is the reason the fenced total must not be quoted as
the round.

Counters, identical in both modes: **24,400 dispatches and 3,100 host
synchronizations across 100 trees**, which is 244 dispatches and **31 syncs
per tree**. Thirty-one is the leaf budget: one host round trip per split,
exactly as the design study predicted. Unattributed time is 0.19% in async and
0.16% in fenced, so the instrument accounts for essentially the whole round.

## What the rules said, and one rule that was wrong

**R1, CPU serial against LightGBM serial.** Ours 15.96 median at one worker,
LightGBM 8.82 at one thread, so **1.81x**. That lands in the 1.3x to 2.0x band,
whose pre-registered answer is "both the inner loop and parallel efficiency
contribute; take the interleaved histogram cells first because it is exact and
cheap, and re-measure before committing to row blocks." That is the rule and it
fired cleanly.

Worth noting alongside it: at one worker we are 15.96 and at ten we are 6.98,
so we get 2.29x from ten cores. LightGBM gets 8.82 to 2.86, which is 3.08x.
Both of us scale poorly on a 4-performance-core part; they scale better.

**R2, the histogram's share of the GPU round. The rule as written was
mis-specified and I am recording that rather than applying it mechanically.**
It said above 60% take the kernel, below 40% take the per-split sync, and
between the two take the search fold. The measured share is 49.3%, which
selects the search fold. But split search is **2.4%** of the round, so folding
its launches cannot win more than a rounding error. The rule's middle branch
was written on the assumption that a middling histogram share implied a
meaningful search cost, and the profile shows those are unrelated.

The honest reading of the same numbers: the round is 49% histogram and 46%
transfer plus partition, and there is no single dominant phase. Both the
device-owned tree, which attacks the 31 round trips inside that 46%, and the
kernel work, which attacks the 49%, are justified. Neither alone gets to
LightGBM: removing all fixed cost leaves the kernel, and fixing the kernel
leaves 1.5 seconds of fixed cost per fit that already loses at 250,000 rows.

**R5, the split gate.** Not measured this session; the 250k and 500k device
against host comparison is still owed.

## SUPERSEDED IN PART, the same day, by an actual Metal timeline

`docs/METAL_TIMELINE.md` and `bench/results/metal_timeline_2026-08-15/` record
the first GPU trace this project has ever taken, with `xctrace record` against
the `Metal System Trace` template. It contradicts the reading of the phase
shares above, and the contradiction is instructive rather than a defect in
either instrument.

**The GPU is idle 76.5% of the training span at 200,000 rows, and 87.5% at
50,000.** Busy time is 552 ms of a 2,347 ms span, confirmed to within 2.4% by
two independent tables in the trace. It is not a downclock; the device sat at
its Maximum performance state for 77.9% of the capture.

So the phase shares above are shares of *attributed device work*, not shares of
wall clock. Compute of every kind is **22.9% of a round**. Reading "histogram
49.3%" as "half the round is the histogram kernel" was wrong, and it was my
error rather than the instrument's: the phase profile measures where device
work is charged, and it cannot see time in which no phase is running at all.

The mechanism is measured, not inferred. The host never blocks on a compute
kernel, 2 times in 18,701, and blocks on **94.1% of blits**. That is 3,208
serialization points, **32.1 per round**, which independently reproduces the
3,100 synchronizations counted above by a completely different instrument. One
blocking readback costs **606 microseconds** at the median, of which **3.7
microseconds** is the GPU actually moving bytes. Thirty-two of those is 20.16
ms of a 23.50 ms round, or **85.8%**.

Which means the `transfer` phase above is not about data movement at all. The
GPU spends **0.65%** of the span moving bytes. `transfer` is the phase the wait
gets charged to, and the wait is the product.

**This reorders the plan.** The device-owned tree is not one of two justified
directions, it is the direction, and its ceiling is higher than the phase
shares implied. The histogram kernel work drops accordingly: at 22.9% compute,
a kernel twice as fast buys about a ninth of the round, against a wait that is
five sixths of it.

Two premises quoted in earlier reviews are refuted from source by the same
lane. The root grid is 25 feature-pair blocks by 2 row tiles, because Metal
defaults to feature group 2, not one block per feature. And it is **391 serial
rows per thread, not roughly 2,000**.

Whether the kernel is latency-bound or bandwidth-bound remains **unanswerable
on this machine**: Instruments exposes exactly one GPU counter on this M4, `RT
Unit Active`, which is raytracing. There is no occupancy, ALU, bandwidth or
cache counter to read.

One incidental finding worth more than it looks. Two of the four captures ran
**entirely at the Minimum GPU performance state** while the other two ran mostly
at Maximum, which is the most plausible mechanism anyone has yet identified for
the two-to-three-fold benchmark drift this repository fights on Apple silicon.
It is a clock state, not a workload difference, and it means an interleaved
comparison can still be invalid if the two arms straddle a clock transition.

## What this profile does not say

- The Metal timeline above answers what this section originally said was
  unanswered. What follows is what remains open after it. Every claim about bubbles between kernels,
  occupancy, or whether the histogram kernel is latency-bound rather than
  bandwidth-bound remains arithmetic. A capture is the next cheap thing.
- The 1.5 second per-fit fixed cost is now attributed: it is the per-split
  wait, roughly 3,100 round trips at about 445 microseconds, scaling with trees
  times leaves rather than with rows. It is not session setup and not kernel
  compilation, so session reuse and warmup do not touch it.
- The original text of this bullet, retained so the correction is legible: it
  was inferred from three shapes, not attributed. Whether it is session setup, kernel compilation, or per-tree
  work is unknown, and `bench/README.md` mentions a 1.6 to 1.9 second session
  setup that may or may not sit inside the timed region.
- Multiclass, bagging, GOSS, and the real datasets are not in this profile at
  all. The worst number on disk is covertype, where the GPU takes 44 to 56
  seconds against LightGBM's 4 to 7, and nothing here measures it.
- Fenced and async attribute differently by construction, and the true
  per-phase device time is bracketed by the two rather than given by either.

## Multiclass, measured for the first time, 465,000 x 54, 7 classes

The dispatch bug that made every previous multiclass GPU number a mislabelled
CPU fit is fixed, and `bench/bench_train_gpu.mojo` gained a `multi:K` arm, so
this is the first honest multiclass measurement in the project.

| arm | median | spread |
|---|---|---|
| mojotrees CPU | 25.47 | 7.7% |
| mojotrees GPU | **15.30** | 0.1% |
| mojotrees GPU, `MOJOTREES_GPU_CLASS_BATCH=7` | 15.45 | 0.8% |

**The GPU wins multiclass by 1.63x**, resolved well outside the noise floor.
That settles the route-then-measure question the right way: honoring an
explicit `device='gpu'` for multiclass is a win, not a silent regression, so
the dispatch fix stands without needing a fallback warning.

**Class batching is indistinguishable.** 15.45 against 15.30 is inside the
spread, so batching seven classes into one launch buys nothing at this shape.
That matters beyond multiclass: classes in a round are independent trees over
one shared uploaded matrix, which is structurally the same problem as
independent fits, so this is the first measured point on the multi-fit
batching idea.

**It is not a refutation, and an earlier version of this paragraph overstated
it as one.** Batching amortizes per-dispatch cost, and the Metal timeline has
since shown that per-dispatch cost is not what this round spends its time on,
so a null result here was predictable rather than informative. The condition
under which batching should help has not been created yet: once the per-split
host wait is gone, the residual cost on the small-leaf tail is a latency-bound
kernel that under-fills the GPU, and batching independent trees is exactly
what fills it. That cost is currently hidden behind the wait.

So the correct record is **retest after the device-owned tree lands, at
250,000 rows and below**, not "dead". Filing it as refuted would have thrown
away an idea on evidence that could not have supported it either way.

Against LightGBM, using its own recorded covertype figures of 6.7 to 6.9
seconds, we are roughly **2.2x behind** on multiclass rather than the 10x that
the mislabelled record implied. Not comparable arm for arm, because
`bench_lightgbm.py` has no multiclass mode and covertype is a real dataset
rather than this synthetic shape, so treat 2.2x as an order of magnitude and
not a measurement.
