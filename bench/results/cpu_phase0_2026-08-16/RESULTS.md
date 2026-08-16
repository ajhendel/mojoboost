# CPU round 1, Phase 0. Apple M4, 2026-08-16

The first CPU-side profile and baseline this repository has taken against its
own registered rules. Protocol in `../PROFILE_PROTOCOL.md`, sections C0 through
C8, all committed before any of this ran (commit 04ba2ea).

Toolchain Mojo 1.0.0. Apple M4, 10 cores (4 performance, 6 efficiency), 16 GB.
Branch `cpu-round-1` off `perf-round-2 @ 8038369`. 100 rounds, 31 leaves, 255
bins, squared error. LightGBM 4.7 reached through Python interop in the same
process, pinned to 10 threads via `MOJOTREES_LGBM_THREADS=10`.

**LightGBM's resolved builder is `force_row_wise`**, pinned by
`scenarios.LIGHTGBM_ALIGNMENT`. See C3 below; it matters more than it looks.

## Box state, stated rather than assumed

**The box was NOT verifiably quiet.** Load average went 2.47 at the first arm
to 4.88 at the last, which is the concurrent GPU campaign compiling. Under the
narrow timing-only lock agreed with that session, compiles are permitted during
a timing window, so this is the lock working as designed rather than a
violation.

What that costs, and what it does not:

- **The ratios are the trustworthy part.** Every comparison below is
  interleaved within one process, so a build landing mid-sequence hits both
  arms alike.
- **The absolute seconds are contaminated** and are labelled that way. Our
  arm's 21 percent spread at 1M is partly somebody else's compile.
- The headline will be re-taken in a genuinely quiet window before it goes in
  any summary table. The GPU session has offered a courtesy pause for it.

## A correction to the harness before any number is read

**`bench/bench_train_gpu.mojo` computes its own `resolved` /
`indistinguishable` verdict from the arms' MINIMA. The protocol requires the
MEDIAN.** `PROFILE_PROTOCOL.md` is explicit — "Report median of three as the
decision statistic", and "Median rather than min, and the two documents now
agree on it" — because the minimum is the luckiest sample and contention here
is the finding rather than the noise.

So the harness's printed `lightgbm_vs_cpu:` line is not the protocol's verdict.
Every verdict below is recomputed by hand under **M0 on medians**: resolved
when the medians differ by more than the wider arm's own min-to-max spread,
over at least five pairs.

This changes one result, and it is the one this project most wants to be true.
See the 50,000 row line.

## C1. The baseline, three shapes, interleaved in one process

Samples are given in the order they ran, per C8, so drift reads as a trend.

### 1,000,000 x 50

| arm | samples, in run order | median | min-max range | spread |
|---|---|---|---|---|
| ours, CPU | 5.458 5.781 6.000 6.606 6.453 | **6.000** | 1.148 | 21.0% |
| LightGBM, 10t | 2.725 2.898 3.005 2.945 2.949 | **2.945** | 0.280 | 10.3% |

Median delta 3.055 against a wider-arm range of 1.148. **RESOLVED.**
**We are 2.04x behind at one million rows.**

Our arm rises 18 percent across the five repeats; LightGBM's rises 10 percent.
Both rise, which per C8 is the machine, but they do not rise equally.

### 250,000 x 50

| arm | samples, in run order | median | min-max range | spread |
|---|---|---|---|---|
| ours, CPU | 1.945 1.649 1.661 1.666 1.693 | **1.666** | 0.295 | 17.9% |
| LightGBM, 10t | 1.072 1.087 1.098 1.050 1.059 | **1.072** | 0.048 | 4.6% |

Median delta 0.594 against a wider-arm range of 0.295. **RESOLVED.**
**We are 1.55x behind at 250,000 rows.**

### 50,000 x 50

| arm | samples, in run order | median | min-max range | spread |
|---|---|---|---|---|
| ours, CPU | 0.496 0.532 0.542 0.536 0.550 | **0.536** | 0.053 | 10.8% |
| LightGBM, 10t | 0.619 0.584 0.565 0.594 0.587 | **0.587** | 0.054 | 9.5% |

Median delta 0.0516 against a wider-arm range of 0.0537.

**CONSISTENT, NOT RESOLVED.** All five pairs favor us, without exception, and
the median margin is 9.6 percent in our favor. But the margin is smaller than
the wider arm's own spread, so under this project's own M0 rule the magnitude
is not established.

**The harness printed `resolved` for this shape. It is wrong, because it
compared minima.** On minima we lead by 13.9 percent against a 10.8 percent
floor and it clears; on medians we lead by 9.6 percent against a floor of
roughly 10 percent and it does not.

This is the only shape this library has ever beaten LightGBM on, it is the
shape a new user meets first, and it is the one place where using the luckiest
sample instead of the median flips the verdict. It should be re-taken in a
quiet window with more repeats before it is quoted as a win. **The direction is
not in doubt; the size is.**

### Binning, which is not in any of the times above

| shape | ours | LightGBM | ratio |
|---|---|---|---|
| 1,000,000 x 50 | 0.176 | 1.109 | **6.3x faster** |
| 250,000 x 50 | 0.042 | 0.242 | **5.8x faster** |
| 50,000 x 50 | 0.0487 | 0.0477 | **dead heat, 2% behind** |

The figure carried in this project's memory is "binning ~3x faster", as a flat
multiple. It is not flat. It is **6.3x at a million rows and gone entirely at
fifty thousand**, where we are fractionally behind. A single multiple
misdescribes this column at both ends.

## C2. R1, the serial discriminator. The rule fired, on a different number

Our CPU at `MOJOTREES_NUM_WORKERS=1` against LightGBM at `num_threads=1`,
1,000,000 x 50, interleaved in one process.

| arm | samples, in run order | median | min-max range | spread |
|---|---|---|---|---|
| ours, 1 worker | 12.357 12.428 14.104 12.338 12.273 | **12.357** | 1.831 | 14.9% |
| LightGBM, 1 thread | 9.013 9.606 9.098 9.070 9.084 | **9.084** | 0.593 | 6.6% |

Median delta 3.273 against a wider-arm range of 1.831. **RESOLVED.**
**Serial ratio 1.36x.**

The previously recorded figure was **1.81x**, taken 2026-08-15 in a window
whose state was not recorded. C2 required re-taking it rather than inheriting
it, precisely because this machine's regimes move a comparator by a factor of
two, and re-taking it moved the number by 25 percent. That is the rule paying
for itself.

**1.36x lands in the "between 1.3x and 2.0x" band**, whose registered answer is
"both contribute; take the interleaved histogram cells first because it is
exact and cheap, and re-measure before committing to row blocks."

It lands near the **bottom** of that band, and the honest note is that at 1.30x
the rule would have selected the parallel lane instead. It did not, and the
rule is followed as written rather than reread until it gives the answer the
next section makes tempting.

### The decomposition, which is the most useful thing in this document

Multicore scaling, from the two measurements above:

| engine | 1 thread | 10 threads | scaling |
|---|---|---|---|
| ours | 12.357 | 6.000 | **2.06x** |
| LightGBM | 9.084 | 2.945 | **3.08x** |

LightGBM's 3.08x independently reproduces the 3.08x recorded on 2026-08-15 from
a different window, which is a useful sign that the comparator's scaling is a
stable property even when its absolute times are not.

Our 2.04x deficit at one million rows factors exactly:

- **serial inner loop**: 12.357 / 9.084 = **1.360x behind**
- **parallel scaling**: 3.084 / 2.059 = **1.498x behind**
- product: 1.360 x 1.498 = **2.038**, against a measured end-to-end ratio of
  **2.037**

The two terms reproduce the measured ratio to three decimal places. That is
arithmetic rather than a coincidence — it is what the two measurements mean —
but it does establish that there is no third term hiding. All of the deficit is
in those two places.

**The parallel term is now the larger of the two**, 1.498 against 1.360. On
2026-08-15's numbers it was the smaller one. Nothing about the code changed;
the serial measurement did.

What each half is worth on its own, both **derived bounds** over the two
measurements:

- Fix parallel scaling to LightGBM's 3.08x, touch the inner loop not at all:
  12.357 / 3.084 = **4.01 seconds**, still 1.36x behind.
- Fix the serial inner loop to LightGBM's 9.084s, keep our 2.06x scaling:
  9.084 / 2.059 = **4.41 seconds**, still 1.50x behind.
- Both: **2.94 seconds**, which is parity.

**Neither half alone reaches the target.** C2 registered this before the data,
on the older numbers, and it survives the new ones. A lane that delivers its
whole share and leaves us at 4 seconds has not failed.

## C5. The CPU stage profile, and the finding that redirects the round

`bench/bench_profile.mojo`, 1,000,000 x 50, 3 reps, serial against auto.
Machine header as reported by the instrument: 10 physical / 10 logical cores, 4
performance; SIMD float64 width 2, kernel lanes 8; `tasks_per_core` 4,
`parallel_min_ops` 65536; auto-mode task counts 40 over features, 15 over rows.

| stage | serial_s | auto_s | speedup |
|---|---|---|---|
| bin_fit | 0.2009 | 0.0841 | 2.39x |
| bin_transform | 0.1837 | 0.0764 | 2.41x |
| grad_hess | 0.00109 | 0.00043 | 2.53x |
| **hist_full** | **0.0500** | **0.0154** | **3.25x** |
| **hist_subset50** | **0.0282** | **0.0083** | **3.40x** |
| **hist_subset10** | **0.00609** | **0.00186** | **3.28x** |
| hist_subtract | 0.0000169 | 0.0000152 | 1.11x |
| split_scan | 0.0000382 | 0.0000267 | 1.43x |
| partition | 0.00409 | 0.00109 | 3.74x |
| **grow_tree** | **0.2100** | **0.0850** | **2.47x** |
| predict | 0.0333 | 0.0323 | **1.03x** |

### The histogram stages dominate the serial column, as C5 predicted

Excluding binning (not in train time) and the composite `grow_tree`, the three
histogram stages are 0.0843 of roughly 0.089 seconds of measured serial
training work. C5's registered prediction was "at or above half of it". Held.

### But the parallel loss is NOT in the histogram kernel, and that is new

**Every individual kernel scales better than a whole tree does.** The histogram
builders reach 3.25x to 3.40x and the partition reaches 3.74x, against
`grow_tree` at **2.47x** for the same work composed into a tree.

The kernels are within reach of LightGBM's 3.08x whole-fit scaling. The tree
built out of them is not. So the 1.498x parallel deficit is **not** primarily
inside the histogram accumulation loop; it is in what happens between and
around the kernel calls — per-node allocation, the serial sections of tree
growth, and per-dispatch overhead.

**This is the single most consequential finding in Phase 0, and it points away
from the lane the round was expecting to need.** L3, row-block private
histograms with a fixed-order fold, exists to raise histogram parallelism above
the "tasks <= feature groups" cap. The histogram stages are already the
best-scaling part of the CPU path. L3's prize is therefore smaller than the
brief assumed, and it is the most expensive and least exact lane in the plan —
it is the one that changes bits and forces a golden re-baseline.

### The limit of that reading, stated because it is load-bearing

The profile measures histogram builds at 1,000,000, 500,000 and 100,000 rows.
**A 31-leaf tree spends most of its nodes far below 100,000 rows**, and the
profile says nothing about those. If small-node builds fall below
`parallel_min_ops` and run serial, that would also show up as `grow_tree`
scaling worse than its kernels, and it would have a different fix from
per-node overhead.

So the finding as stated is: *the parallel loss is not in large-node histogram
accumulation.* Whether it is per-node overhead or small-node serialization is
**not yet distinguished**, and Phase 0 does not get to claim it is. The next
measurement, before any lane commits to one of those, is the node-size-class
phase breakdown that `src/mojotrees/phase_profile.mojo` already produces for
the `cpu` arm.

### The `predict` row is an instrument defect, not a finding. Retracted before use.

`predict` reports 1.03x, serial 0.0333 against auto 0.0323, and the obvious
reading is that batch prediction does not parallelize — which would have
independently confirmed a line in `PROFILE_PROTOCOL.md` that has never been
measured on the CPU side ("our batch prediction runs at parallel efficiency
1.00 against LightGBM's 6.5 to 9.5 on the same ten threads").

**That reading is wrong and this paragraph is the retraction, written before
the number was used to justify a lane.**

`bench/bench_profile.mojo` implements its `predict` stage as its own serial
loop over rows in the benchmark body — `acc += tree.predict_row(data, r)` —
over a single tree. It never calls the shipped batch path. So the stage is
serial in both columns by construction and 1.03x is the measurement of a loop
the library does not use, not of the library.

The shipped path is `boosting.predict_batch_range`, and it **is** parallel:
`dispatch_rows` at `boosting.mojo:1106`. Its own docstring says batch
prediction "was" at parallel efficiency 1.00 "because this loop was serial",
in the past tense, and explicitly declines to claim what it became: "What that
becomes is not measured here and this docstring claims no number."

So the true state is: **the parallel batch prediction shipped in round 2 is
wired, and its speedup has never been measured by anyone.** That is a
measurement gap, not a defect, and it is the orchestrator's to close rather
than a lane's.

Two consequences, both recorded rather than acted on here:

1. `bench/bench_profile.mojo`'s `predict` stage should exercise
   `predict_batch_range` so the instrument measures the shipped path. Until it
   does, its `predict` row must not be quoted.
2. The parallel-efficiency-1.00 line in `PROFILE_PROTOCOL.md` describes the
   pre-round-2 state and should be dated, because it currently reads as
   present tense.

**This is the second instrument correction in this document**, after the
minima-versus-median verdict discrepancy. Both were found by checking what the
instrument actually does rather than by reading its output, and both would have
produced a confidently wrong lane decision. It is worth stating plainly that
Phase 0's most valuable output so far has been two corrections to the tools,
not two measurements.

## C3. The builder discriminator. L4 is dead, and we lose to the builder we have

LightGBM at 1,000,000 x 50, ten threads, both builders **pinned** so neither
pays the auto-selection cost, interleaved in one process, five repeats.

| arm | samples, in run order | median | min-max range | spread |
|---|---|---|---|---|
| `force_row_wise` (what we pin) | 2.680 2.708 2.856 2.983 3.054 | **2.856** | 0.375 | 14.0% |
| `force_col_wise` | 2.865 2.909 3.052 3.132 3.120 | **3.052** | 0.267 | 9.3% |

Median delta 0.196 against a wider-arm range of 0.375. **CONSISTENT, NOT
RESOLVED.** All five pairs favor row-wise; median margin 6.9 percent.

**C3's rule required "materially faster, resolved by M0" to authorize L4. Not
met. L4 does not start.** The registered prediction was that L4 would not
start, and it held — though the predicted *direction* was wrong. C3 expected
column-wise to be slightly favored at 50 dense features; row-wise is
consistently ahead instead.

### The first reading: the pin is conservative

The builder we pin is LightGBM's **faster** one at this shape, so pinning it
makes the comparison harder on us rather than easier. Every margin this project
has claimed against LightGBM was claimed against LightGBM's better
configuration. The error that was worried about — a win manufactured against a
handicapped comparator — did not occur.

It should still be stated wherever a margin is quoted, in one sentence, because
the alternative is an outside reader discovering the pin themselves.

### The second reading, which is the one that redirects the round

`force_col_wise` is the column-wise builder, structurally the same shape
mojotrees already has. It measures **3.052 seconds against our 6.000**.

**We lose to LightGBM by roughly 2x using the builder we already have.**

So the deficit is our *execution* of a builder shape we have, not the absence
of one we lack. The entire gap between LightGBM's two builders is 6.9 percent,
which could not close a 2x deficit even if L4 landed perfectly. That kills the
round's largest speculative lane on evidence, before it cost anything, and it
keeps the round pointed at the inner loop and the scheduling.

**Caveat on that specific ratio**: it rests on our contaminated 6.000 median.
The *conclusion* is robust, since no plausible clean number closes a 2x gap and
6.9 percent cannot. But the figure "1.97x" carries somebody else's compiles in
it and will be restated after the clean-window retake rather than carried
forward.

## C4. The core pool. Void as an absolute, decisive as a ratio

`MOJOTREES_CPU_CORE_POOL=performance` (4 performance cores only) against the
default (all 10), 1,000,000 x 50, three repeats per run, alternated across
processes A B A B A B because the harness has no env-var arm.

| pair | default | performance | performance is slower by |
|---|---|---|---|
| 1 | 6.115 | 7.167 | **17.2%** |
| 2 | 9.655 | 11.168 | **15.7%** |
| 3 | 13.047 | 14.688 | **12.6%** |

**The absolute numbers are void.** The same configuration measured 6.115,
9.655 and 13.047 seconds across twelve minutes, a **2.1x degradation**, with
load average climbing 4.6 to 6.3 as the concurrent GPU campaign's lanes
compiled. C8 says a comparison whose arms straddle a regime change is discarded
rather than corrected, and no correction factor survives a 2.1x drift.

**The ratio is not void, and this is the useful part.** Each pair was taken
adjacent in time, so the within-pair comparison brackets the drift. All three
pairs agree in direction and land within 4.6 percentage points of each other
across a 113 percent shift in level:

**Restricting to the four performance cores is consistently 12 to 17 percent
SLOWER at one million rows.**

That answers the C4 question in the direction that requires no action: the
default stays, and the six efficiency cores are contributing real work despite
being a late-finishing tail on every barrier. Removing them is not the fix for
the parallel deficit; scheduling around them might be.

It also demonstrates the measurement technique this round depends on, from the
failure side. **Adjacent pairing survived a regime change that destroyed every
absolute number in the same file.** That is the strongest available argument
for the in-process interleaving rule in C-ops, and it was obtained by
accident.

## Confirmed by reading, not measurement: round 2's policy cache was never wired

`DispatchSettings` and `ResolvedCpuPolicy` have **zero callers anywhere in
`src/`** outside the two modules that define them. The only references are in
`tests/test_cpu_dispatch.mojo`.

Round 2 shipped a "resolved CPU policy cached per fit" and the mechanism is
built, tested, and never called. Every histogram build still performs its own
`getenv` sweep and core detection per dispatch.

This is the third instance of one pattern in this repository — the GPU campaign
independently found `gpu_gradient_stream.HostGradientStage` unused and
duplicating `stage_gradients`. Built, tested, never wired, and the benefit
claimed in the commit message. The suite stays green because the tests call the
mechanism directly, so nothing ever notices the call sites do not exist.

**Adopted into this round's merge protocol:** for any lane claiming a per-fit
or per-round saving, grep for callers of the mechanism outside its defining
module before believing the claim.
