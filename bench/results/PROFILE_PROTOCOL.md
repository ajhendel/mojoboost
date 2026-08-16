# Profile session protocol, and the decisions it settles

Written **before** the session ran, deliberately. The point of committing this
first is that the next round's lanes are chosen by a rule agreed in advance
rather than by whichever number turns out to be the most interesting. If a
result below is ambiguous against its rule, the honest answer is that the rule
did not fire and the lane does not start, not that the rule should be reread
until it does.

Nothing in this file is a measurement. The companion results file records what
was measured; this one records what we said we would do about it.

## Why a protocol at all

Two rounds of optimization work have now been chosen from arithmetic over
source rather than from a profile, and the record is mixed. The GPU row-tile
floor was the change its author was most confident in, was structurally well
argued, raised occupancy exactly as intended, and measured 22 percent slower at
50 features and 36 percent slower at 100. It was reverted. In the same round,
pre-quantized gradients measured a real win and block primitives in the split
search measured indistinguishable, and neither outcome was predictable from the
source.

No stage-level profile of either backend has ever been recorded in this
repository.

## Session conditions

- Idle machine. No agent, no editor build, no background job. This machine's
  device timings drift two to three times across time windows and a loaded run
  measured 24.9 percent spread where a quiet one gives 0.3.
- Capture thermal state into the results header with
  `bench/apple/thermal_capture.sh`, before and after, so a throttled tail is
  visible rather than inferred.
- Every comparison interleaved within one process where the harness supports
  it (`bench-train-gpu rows feats reg N arm1,arm2`), and alternated across
  processes where it does not.
- Report **median** of three as the decision statistic, with the min and the
  spread beside it. A pair whose difference is inside the larger arm's spread
  is **indistinguishable**, and is recorded as such rather than as a small win.

  Median rather than min, and the two documents now agree on it. The minimum is
  the luckiest sample, and contention is not noise here, it is the finding:
  our batch prediction runs at parallel efficiency 1.00 against LightGBM's 6.5
  to 9.5 on the same ten threads, and a statistic that discards contention
  would hide exactly that. `bench/real_data/run.py` and `report.py` were
  reconciled onto the median in the same round for the same reason, and a
  protocol that named a different statistic than the harness stores would put
  two records in conflict the first time anyone compared them.
- Header records: toolchain version, machine, thermal state, LightGBM version,
  LightGBM's resolved builder (`force_row_wise` / `force_col_wise` / auto), and
  the thread count given to both sides.
- **Save the results file before interpreting any of it.**

## Shapes

Three, not one. This round changed dispatch thresholds, the feature-group
width, the parallel grain, and the GPU split gate, and every one of those bites
hardest at small and medium sizes. A profile taken only at the headline shape
can show a win while the shape a user actually hits first regresses.

- 1,000,000 x 50 (the headline, and the shape that sits exactly on the GPU
  split gate)
- 250,000 x 50
- 50,000 x 50

## The decision rules

### R1. Is the CPU problem the inner loop, or parallel efficiency?

Measure our CPU backend at `MOJOTREES_NUM_WORKERS=1` against LightGBM at
`num_threads=1`, same shape, same parameters.

- **Ratio within 1.3x**: our serial inner loop is competitive and the deficit
  is parallel efficiency. Next CPU lane is **row-block private histograms with
  a fixed-order fold**. That changes bits against the current serial summation
  order, so it is a deliberate golden-fixture update with a stated ulp
  movement and an opt-out, and it does not start before `docs/NUMERICS.md` is
  in place, which it now is.
- **Ratio above 2x**: the inner loop is the problem. Next CPU lanes are
  **interleaved histogram cells** (grad and hess adjacent, `count` narrowed to
  Int32) and the **Int32 arena in-place partition**.
- **Between 1.3x and 2x**: both contribute. Take the interleaved cells first,
  because it is exact and cheap, and re-measure before committing to row
  blocks.

### R2. How much of the GPU round is the histogram?

From the phase profile's `histogram` share of the attributed round.

- **Above 60 percent**: the kernel latency lane (unrolled row loop, hoisted
  loads, then re-measuring 4 to 8 tiles now that partial traffic is known to
  have confounded the earlier tile experiment) and the group-major bin layout
  move up.
- **Below 40 percent**: the per-split host synchronization is the target
  instead, and which lane takes it is decided by R3.
- **Between**: take the search fold, which is the cheaper of the two and
  touches one file.

### R3. Is the fully-enqueued tree worth its cost?

Depth-wise growth is already implemented and costs about 5 host waits per tree
against roughly 30 for leaf-wise. It is the cheap proxy.

- **Depth-wise materially faster**: the per-split wait is real, and the
  fully-enqueued tree lane is justified. It runs **alone** in its round, not
  beside anything else touching `train_gpu.mojo` or `gpu_active_rows.mojo`.
- **Depth-wise indistinguishable**: the wait is not the cost, the enqueued-tree
  lane does not start, and the GPU work goes to the histogram kernel instead.

### R4. Feature-group width, and the core pool

Both are unconfounded for the first time now that the task splitter dispatches
over groups and a one-feature task can no longer silently defeat interleaving.

- Ship whichever of width 2 and width 4 wins on **wall time**, and only if it
  also wins at 250,000 rows. The byte table alone does not decide this: width 4
  cuts gradient traffic to 208 MB per node from 400 but drops the dispatch to
  13 groups on 10 cores, which is 65 percent utilization against 83.
- Same rule for `MOJOTREES_CPU_CORE_POOL`. Do not change the default on a
  single-shape result.

### R5. The GPU split gate

`M4_MIN_NORMALIZED_WORK` is 50,000,000 while the policy's own comment records
the observed win point at 25,000,000. The gate is therefore set at twice the
evidence and declines the device across a decade of shapes, each fallback
costing a 153 KB download and a host synchronization per node.

Measure device against host split search, interleaved, at 250,000 x 50 and
500,000 x 50. **Set the threshold at the measured crossover**, not at a
multiple of it, and record the evidence in `CrossoverEvidence` with the
`POLICY_VERSION` bumped.

## What does not happen until this file has a companion of results

No optimization lane starts. Three no-measurement lanes are in flight as this
is written (the `auto` device crossover, the benchmark harness parity fixes,
and the GPU contraction consistency triage), and those are corrections rather
than optimizations.

`main` is not promoted either. `perf-round-2` is bit-exact against pre-merge
and green on the full suites, but "no regression at 50,000 rows" is not yet
known, and that is the shape a user meets first.

---

# Sweep II protocol, written before the sweep ran

Same discipline as above and for the same reason: the previous round produced
three results that would each have been read differently had the rule been
written afterwards. Committed before any arm is executed.

## Provenance vocabulary, used from here on

Every number in this project now carries one of four labels, stated each time:

- **measured**: read off an instrument on a stated machine in a stated window.
- **fitted**: derived from measured points by a model, with the points named.
  A two-point fit and a three-point fit over different row ranges are different
  numbers and must not be compared as though they were the same quantity.
- **derived bound**: arithmetic over bytes, launches or counts. A bound, never
  a prediction.
- **estimated**: a judgment. Says so.

This exists because two numbers were quoted all week as measurements when one
was a derived bound and the other was a two-point fit from a different era over
a different row range, and the confusion inverted a priority.

## The open question this sweep exists to answer

Two fits of our GPU wall clock disagree, and **both are real**:

- **2.24 microseconds per row**, fitted from the post-round-2 points at 50,000,
  250,000 and 1,000,000 rows, alongside a 1.33 second constant.
- **3.2 microseconds per row**, fitted from the pre-round-2 pair at 1,000,000
  and 5,000,000 rows.

They are not in conflict. They are different row ranges. Taken together they
say our cost per row may be **superlinear above one million rows**, which is
exactly where the 2.24 figure stops having evidence. The claim "our marginal
cost per row is about 10 percent better than LightGBM's" is supported **below**
one million rows and **unproven above it**.

So the sweep must include 2,000,000 rows, and 5,000,000 if the box allows,
because that is the only part of the curve where the two fits can be told
apart. A sweep that stops at 1,000,000 would confirm the number we already have
and leave the question that matters untouched.

## Arms

Interleaved in one process where the harness supports it, alternated across
processes where it does not. Five repeats, median as the decision statistic
with min and spread beside it.

- our CPU
- our GPU, host-driven plane (today's default)
- our GPU, `MOJOTREES_GPU_TREE_RESIDENT=1`
- our GPU, `grow_policy=depthwise`
- LightGBM at 10 threads

## Shapes

250,000 / 1,000,000 / 2,000,000 rows by 50 features, and 5,000,000 if memory
and time allow. 50,000 is included for the tree-resident arm alone, as a
regression check.

## Recorded for every arm, without exception

GPU performance state before and after, thermal state, sync count, and the
backend proof. The clock state is not optional: two of four captures in the
previous session ran entirely at Minimum GPU clock while two ran at Maximum,
which is the most plausible cause of this machine's two-to-threefold drift, and
an interleaved comparison straddling a clock transition is invalid even though
it is interleaved.

Enqueue time is recorded **separately from wall time**. The Metal command queue
is 64 buffers deep and the stall when it fills is invisible inside
`objc_msgSend`, so a device-resident plane emitting on the order of two hundred
launches per tree can backpressure. If that happens it must show up as host
time rather than as "the GPU got slower".

## Decision rules

### S1. Does the tree-resident plane become the default GPU plane?

All three must hold:

- trees are node-identical to the host plane. **Already satisfied**, by
  `tests/test_gpu_tree_resident.mojo`, which compares value bits with no
  tolerance across the leaf budget, early termination, missing values, bagging,
  and a refused configuration.
- it is faster at **both** 250,000 and 1,000,000 rows.
- it does not regress at 50,000.

Any one failing means it stays opt-in. Two of the three are speed; the first is
not negotiable and is the reason it is listed first.

### S2. What is depthwise allowed to be?

**A benchmark row and an opt-in. Never the parity default.** Depth-wise growth
grows a *different tree* than leaf-wise, so "it captures most of the 1.33
seconds for free" is only true for a user who did not ask for leaf-wise. Making
it the default would be answering a speed question by changing the model, which
this project does not do. Measure it, report it, and if it wins decisively that
is an argument for making the device-resident leaf-wise plane as good, not for
switching growth policies underneath somebody.

### S3. Where does the engineering go next?

If the sweep confirms roughly a 1.33 second constant and roughly 2.1 seconds of
compute at one million rows, then **after the resident plane lands the round is
compute-dominated and the histogram kernel is the main event**, not a phase two.
That inverts the ordering this project has assumed all week, so it should be
stated explicitly rather than absorbed.

The bound, labelled as a bound: an ideal-layout kernel is three to four times
below today's byte traffic. A realistic one-and-a-half to two times on the
histogram phase would put the fit near 1.4 to 1.7 seconds against LightGBM's
2.86. That is an **estimate over a derived bound**, not a projection, and it
should not be quoted as though the sweep produced it.

### S4. What would falsify the whole direction?

If the tree-resident plane measures **no faster** at 1,000,000 rows despite
removing thirty host round trips per tree, then the 1.33 second constant is not
the round trips and the model of this workload is wrong. That is the single
most informative possible outcome and it must be reported as loudly as a win.

---

# Session III protocol, registered 2026-08-15 evening, before any of it ran

The previous session ended with three numbers that had earned different amounts
of belief and were written down as though they had earned the same amount. This
section fixes the rules first so that cannot happen again, and it is committed
before the lanes it judges have finished.

## M0. The resolution rule, stated once and referred to by name

An A/B is **resolved** when the two arms' medians differ by more than the wider
arm's own min-to-max spread, over at least five alternating pairs. It is
**consistent** when the direction is the same in a majority of pairs but the
spread is as large as the effect. It is **indistinguishable** otherwise.

Only a resolved result goes in a summary table as a number. A consistent one may
be reported, and must be reported with the word "consistent" and the spread
beside it. This is the rule the resident-plane figure failed: three pairs, one
inverted, an ON arm ranging 3.136 to 3.612 against an effect of 0.57. It was
written as "15 percent". It was consistent, not resolved.

## M1. Quiet-box precondition, and it is a precondition rather than a preference

No measurement counts if any lane, build, or compile is running. This has cost
this project two numbers already, one of them taken at 18.6 percent spread with
four agents compiling. Lanes run between measurement sessions, never during. The
coordinator verifies the machine is idle before the first pair and after the
last, and records that it did.

## M2. What gets measured this session, in this order

1. **The upload collapse.** Resident plane with the begin-tree reset kernel and
   the hoisted search tables against the resident plane without them. This is
   the round's main claim and it goes first while the box is coldest.
2. **The resident plane itself, re-taken.** ON against OFF at five or more
   pairs, to settle M0 on a figure that is currently only consistent.
3. **The histogram row unroll.** `gpu-unroll` against `gpu-nounroll`,
   interleaved in one process, which is the whole reason `set_row_unroll` is a
   runtime argument rather than a comptime knob.
4. **LightGBM, interleaved, with its own repeat spread.** Never measured on this
   machine. Until it exists there is no noise floor to judge any margin against
   and every "we beat LightGBM" sentence is unearned.

## M3. The prediction, registered before the data

**Corrected before any data was taken, and the error is worth keeping visible.**
The brief for the tables lane said eleven waits removed. It is ten. Six
downloads becoming one saves five, not six, and five plus five is ten. The lane
caught it and refused to carry the estimate that depended on it. An off-by-one
in a prediction that is about to be compared against a measurement is exactly
the kind of error that gets absorbed as "close enough" afterwards, so it is
fixed here rather than in the write-up.

Removing **ten** of sixteen waits per tree in the tables path (five uploads in
`begin_tree`, five of six downloads in `download_desc_tables`), three of four in
the search path, and one of two in the raw-score update is **fourteen** fewer
waits per tree. Both lanes have now landed and the composed figure is sixteen
per tree down to **three**, plus the monotone map moving from per-tree to
per-fit, so fifteen rather than fourteen. The estimate below is not rescaled for
that one wait; it stays as registered. At the **measured** 458 microseconds per synchronization over
100 trees that is an **estimated** 0.64 seconds, taking a 3.17 second leaf-wise
fit to roughly 2.5. The total is unchanged because the two other lanes' shares
were counted separately; only its composition moved.

That estimate is recorded so the measurement can refute it rather than be fitted
to it. Its weakest assumption, stated so it can be blamed later: it prices every
copy at the 458 microseconds derived from the depthwise A/B, where the copies
were 153 KB histograms. Most of these are a few hundred bytes. If per-copy cost
turns out to depend on size after all, this estimate is too high and the
per-synchronization constant needs re-deriving rather than re-applying.

If the collapse lands and the fit does **not** move by at least 0.3 seconds,
then per-copy cost is not 458 microseconds in this position, and the per-sync
constant that three independent routes have agreed on is wrong somewhere. Say
so at least as loudly as a win, per S4.

## M4. What the LightGBM spread decides

If LightGBM's own repeat spread on this machine is wider than the margin being
claimed against it, then no margin either way is a result, and both the 6.5
percent depthwise win and any leaf-wise deficit revert to "indistinguishable"
until the comparison is repeated enough times to separate them. This rule is
registered before the spread is known, precisely because it could go against
the result this project wants.

## M5. The comparator changed, so figures across it do not compare

Putting LightGBM inside the interleaved loop also removed a serialize-and-reload
round trip from inside its timed call. That makes LightGBM **faster**, so it
moves the comparison against us, which is the correct direction for a
comparator to move and the reason it stays. But no figure taken before that
commit may be placed in a table beside one taken after it. Every LightGBM number
in this session is re-taken.

## M6. What is allowed to be called a win at the end of this session

Leaf-wise, at 1,000,000 x 50, resolved by M0, against a LightGBM arm measured in
the same process in the same window with its own spread reported. Nothing else.
Not depthwise, which grows a different tree and is an opt-in by S2. Not a
projection from a wait count, however well the wait count has behaved.
