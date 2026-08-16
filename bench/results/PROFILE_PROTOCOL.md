# HOW TO TAKE A NUMBER

This checklist is the protocol. Everything below it is the reasoning, the
history, and the decision rules, and it is worth reading once. This part is worth
reading every time, which is why it is ten lines and sits at the top: a protocol
that must be read in full before each use is one that gets skimmed.

1. **Take the lock** if you are timing: `/tmp/mojotrees-bench.lock`, `mode:
   timing` only, your addressable session name, what and an ETA. Compiles proceed
   freely; timing does not. See `MACHINE_LOCK.md`.
2. **Record the box** before the first arm and after the last: `uptime`, and
   `ps -Ao pcpu,comm | sort -rn | head`. Record it even when it looks fine. A
   quiet box can still be a slow one, and load alone will not tell you, so the
   regime label comes from the arms moving together (A3) and not from these two
   commands. They establish contention, which is a different question.
3. **Interleave arms in one process** where the harness allows. Alternating
   processes turns a neighbour's build into your result.
4. **Five pairs minimum**, and read the verdict off **M0** as amended by **A2**:
   resolved by spread, or resolved by the sign of the per-pair difference.
5. **Label every number** `measured` / `fitted` / `derived bound` / `estimated`,
   every time it appears.
6. **Label the regime.** Effect sizes move with machine state, not just levels.
   Where they differ, report both and refuse to pick one.
7. **A null in a dirty or slow window closes nothing.** It defers. Retake before
   using it to cancel a lane. (A3)
8. **Compare against the comparator at its best**, not at a configuration we
   pinned for our own convenience. See the comparator rule below.
9. **Register the prediction and its falsification before the data**, in this
   file, and do not edit either afterwards. A refuted registered prediction is
   the process working.
10. **Write the results file before interpreting any of it.**

## The comparator rule

`scenarios.LIGHTGBM_ALIGNMENT` pins `force_row_wise = True`, and that pin stays:
on auto, LightGBM spends its first iterations timing both builders and that
one-off lands inside the measured region, which would flatter us.

But a pin we chose is not a fair comparator. **The rule is LightGBM's better
pinned builder at each shape**, measured once per shape and recorded beside the
result. Then "ahead by X" means ahead of LightGBM at its best, which is the only
form of the claim that survives an outside reader running LightGBM themselves.

Every LightGBM figure this repository has recorded to date was taken row-wise,
including the 2.767 that framed a week of work. Until each shape has its builder
measured, every margin carries that caveat explicitly.

---

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

> **Annotation, 2026-08-16. This prediction was measured and refuted. It is
> left exactly as registered.** The measurement is 0.016 seconds against the
> 0.64 below, a null under M0 (`session3_2026-08-16/RESULTS.md`); the
> withdrawal of the model it rests on is `docs/GPU_PORTABILITY.md` section
> 6.1.1. The falsification condition M3 states in its last paragraph fired as
> written.
>
> Worth recording precisely, because M3 did name its own weakest assumption and
> was half right about it. It predicted the failure mode as *per-copy cost
> depending on size after all*. That is not what happened. Byte count is
> irrelevant, as this document assumed; what is wrong is applying a
> per-synchronization constant to a copy at all. The 458 microseconds is
> **derived** from the depthwise A/B, and what that A/B removed were per-level
> **round trips** — host code blocking on a device answer it needs before it
> can decide what to enqueue next. A copy that drains a queue holding nothing
> is not a round trip and costs nothing. The constant remains correct for round
> trips: removing about thirty per tree **measured** 0.75 seconds in the same
> session, resolved.
>
> The rule for the next registration: count round trips to predict time, count
> copies to predict portability risk and ordering hazards, and never quote a
> copy count times a per-synchronization constant as a predicted saving. Text
> below unedited.

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

---

# CPU round 1 protocol, registered 2026-08-16, before any of it ran

The CPU backend has never had a round of its own. Round 2 improved it 1.63x as
a side effect of work aimed elsewhere, and the stage profile that would say
where the remaining time goes has been taken for the GPU and not for the CPU.
This section registers the rules for the CPU round before its first
measurement, for the same reason M0 through M6 were registered: so the lanes
are chosen by a rule agreed in advance rather than by whichever number turns
out to be the most interesting.

It inherits **M0** (resolved / consistent / indistinguishable) and **M1**
(quiet box) unchanged and refers to them by name. It does not restate them.

## C0. What this round is for, and what it is not for

**Target: parity with LightGBM at ten threads at 1,000,000 x 50, with no
regression at 50,000.** Not a win. A CPU win over LightGBM is not expected and
will not be claimed if it happens once; LightGBM is a mature C++ implementation
of this exact algorithm and the honest goal is to stop losing to it.

The CPU path is not a consolation prize. It is the fallback for small data,
where the GPU's fixed cost loses outright; for every configuration the device
refuses; and for every machine without Metal. Its floor is a product floor.

**Provenance labels are mandatory** and use the Sweep II vocabulary above:
measured / fitted / derived bound / estimated. Lanes in this round cannot
measure anything, because the orchestrator takes every timing, so a lane's
numbers may only ever be labelled **derived bound** or **estimated**.

## C-ops. Operating conditions, which changed after this round was planned

Two conditions differ from every previous round and both weaken the isolation
this protocol used to be able to assume. They are recorded here rather than
discovered in a postmortem.

1. **One shared working tree, no per-lane worktrees and no branch checkouts.**
   Lanes edit `/Users/andrewhendel/CascadeProjects/mojotrees` directly, on
   `perf-round-2`, concurrently with each other and with the GPU campaign.
   File-disjointness is therefore the *only* isolation mechanism in this round.
   There is no branch to throw away if a lane goes wrong, so a lane that edits
   a file it does not own is not a merge conflict, it is damage to somebody
   else's in-flight work. The file ownership table is load-bearing.
2. **A second orchestrator is running the GPU campaign on the same box.**
   Agreed with it, in writing, before this round starts:
   - **There is a measurement lock, and this bullet used to deny it.** As
     originally written it read "There is no measurement lock", recording that
     one had been negotiated between the two orchestrators and then overruled,
     that `/tmp/mojotrees-bench.lock` was deleted, and that neither session
     would recreate it. That was true for one interval on the night of
     2026-08-16 and is not true now: the lock was reinstated in a narrower form
     the same night, `mode: timing` only, and `MACHINE_LOCK.md` is the
     authority on it. The instruction audit found the file present and held
     while three passages of this repository disagreed about whether it
     existed.

     The correction is kept visible rather than silently overwritten, because
     the justification this bullet gave for recording the absence — "a protocol
     that describes a control which is not in force is worse than one that
     admits it has none" — applies with the sign flipped to a protocol
     describing the absence of a control that *is* in force. Both are the same
     error. **Take the lock before timing; see `MACHINE_LOCK.md`.**
   - **What replaces it is measurement technique, not scheduling.** Every A/B
     in this round is interleaved *within one process* wherever the harness
     supports it, so both arms are sampled repeatedly inside the same window
     and a compile happening in the middle of it hits both arms alike. An
     alternating-*process* A/B is what breaks on a shared box: arm A runs
     during somebody's build and arm B runs after it, and the difference is
     the build.
   - The size of that effect is now measured rather than assumed. The GPU
     campaign's row-unroll A/B came out **indistinguishable** in a noisy
     window, 8.1 percent against a 14.1 percent floor, and **resolved** in a
     quiet one, 10.8 percent against a 2.1 percent floor. Same command, same
     shape, opposite verdict.
   - Every results header in this round states whether the box was
     **verifiably quiet**, with the evidence it is claimed from. Where it was
     not, the number is labelled and kept rather than presented as clean.
   - Phase 0 is taken **before** this round launches any lane, because the
     lanes this orchestrator controls are the largest noise source it can do
     anything about.
   - `boosting.round_has_constant_hessian`,
     `histogram.objective_has_constant_hessian` and
     `histogram.CONSTANT_HESSIAN` are GPU-visible contracts. No CPU lane
     changes their signature or semantics.
   - **Golden re-baselines are serialized and land alone.** The GPU campaign's
     re-baseline goes first. Ours follow one at a time, each with its ulp
     movement stated in its own commit. Two re-baselines landing together
     produce a fixture whose next failure nobody can attribute.

## C1. What Phase 0 measures, in this order, before any lane starts

No optimization lane starts until this exists, per the precedent set above.

1. **Baseline, interleaved.** Our CPU against LightGBM at ten threads, in one
   process, five repeats, at 1,000,000 / 250,000 / 50,000 x 50. The `cpu` and
   `lightgbm` arms of `bench/bench_train_gpu.mojo` already do exactly this and
   are used rather than rebuilt.
2. **R1 re-taken.** Our CPU at `MOJOTREES_NUM_WORKERS=1` against LightGBM at
   `num_threads=1`, same shape. Decides the lane order, per C2.
3. **The CPU stage profile.** `bench/bench_profile.mojo` at 1,000,000 x 50,
   serial against auto, every stage. Decides which stage the round attacks,
   per C5.
4. **The builder discriminator.** LightGBM `force_col_wise` against
   `force_row_wise` at ten threads. Decides L4, per C3.
5. **The core pool.** `MOJOTREES_CPU_CORE_POOL=performance` against the
   default. Per C4.

## C2. R1, restated, and what it is allowed to decide

R1 above already fired once, on 2026-08-15: ours 15.96 at one worker against
LightGBM's 8.82 at one thread, **1.81x**, which is the "between" branch. That
branch says both the inner loop and parallel efficiency contribute, take the
interleaved histogram cells first because it is exact and cheap, and
**re-measure before committing to row blocks**.

It is re-taken rather than inherited because Session III demonstrated that this
machine's regimes move a comparator by a factor of two, and 1.81x was taken in
a window whose state was not recorded.

- **Within 1.3x**: the serial inner loop is competitive, the deficit is
  parallel efficiency, and **L3 (row-block private histograms) leads**.
- **Above 2.0x**: the inner loop is the problem and **L1 and L2 lead**.
- **Between**: both contribute. **L2 first**, then re-measure before L3.

Registered before the data: the ratio is expected to reproduce between 1.6x and
2.0x, so the middle branch fires again and L2 leads. **estimated.** If it lands
outside 1.3x to 2.0x the plan changes and this line is what says so.

The second number R1 produces is the one worth more: our multicore scaling was
15.96 to 6.98, **2.29x on ten cores**, against LightGBM's 8.82 to 2.86,
**3.08x**. Both of us scale badly on a four-performance-core part. The gap
between 2.29 and 3.08 is what L3 exists to close, and it is a bounded prize:
closing it entirely, with the serial inner loop untouched, puts us at 5.18
seconds against LightGBM's 2.86. **derived bound.** So L3 alone cannot reach
parity, and neither can L1, L2 and L5 alone. This round needs both halves, and
saying so now prevents a later result being read as a failure when it was
always arithmetically insufficient on its own.

## C3. The builder discriminator, on a premise that had to be corrected first

LightGBM chooses between a column-wise and a row-wise histogram builder and
picks by a cost model at dataset construction. We have only the column-wise
shape. L4 would add a row-major `UInt8` copy of the matrix used for histogram
construction only, LightGBM's `MultiValDense` shape, chosen by a chooser, with
the column matrix retained for partition and predict.

**The premise this round was planned on was wrong, and it was corrected from
source before any data was taken rather than after.** The brief asked whether
LightGBM's `force_row_wise` is faster than its `force_col_wise` here, on the
assumption that our comparator runs LightGBM's default auto choice.

It does not. `scenarios.LIGHTGBM_ALIGNMENT` sets **`force_row_wise = True`**,
and `bench/bench_lightgbm.py` inherits it, so every LightGBM number this
project has ever recorded against its CPU arm was taken on the **row-wise
builder**, pinned. It is pinned for a methodological reason and a good one:
left on auto, LightGBM spends its first iterations timing both strategies and
that one-off lands inside the training measurement.

So the question is not "would row-wise help LightGBM here". It is **how much of
LightGBM's lead at 1,000,000 x 50 is the builder we do not have**, and that is
answerable directly:

Run the comparator at `force_row_wise=True` against `force_col_wise=True`,
interleaved, ten threads, at 1,000,000 x 50. Both arms are pinned, so neither
pays the auto-selection cost and the difference is the builder alone.

- **Row-wise materially faster than column-wise, resolved by M0**: a
  meaningful part of the gap we are chasing is a builder shape we lack rather
  than an inefficiency in the shape we have. **L4 is authorized**, and the
  measured delta is the ceiling on what it can return.
- **Indistinguishable, or column-wise faster**: LightGBM beats us with the
  same builder shape we already have, so the deficit is in our execution of it
  and not in its structure. **L4 does not start**, and L1, L2, L3 and L5 are
  the whole round.

Registered before the data: at 50 dense features and 255 bins the two are
expected to be close, with column-wise slightly favored, because row-wise wins
on sparse and high-feature-count data and this shape is neither. **estimated.**
The prediction is that **L4 does not start**. It is written down so that
declining to build it later reads as a rule firing rather than as a lane
quietly dropped.

The second reading of the same measurement is worth as much as the first and
costs nothing extra: if column-wise LightGBM is *also* far faster than us, then
we lose to the builder we already have, and no amount of L4 would have helped.

## C4. The core pool, and what a single-shape result may not do

`MOJOTREES_CPU_CORE_POOL=performance` restricts the pool to the four
performance cores. On a 4P + 6E part, six efficiency cores that finish late are
a tail on every barrier, and excluding them can win despite using fewer cores.

Same rule R4 already set for the feature-group width: **do not change a default
on a single-shape result.** A pool setting must win at 1,000,000 *and* not
regress at 250,000 and 50,000 before it becomes the default. If it wins at one
shape only it stays an environment variable and the shape it won at is recorded
beside it.

## C5. The stage profile, and what it is allowed to redirect

`bench/bench_profile.mojo` reports, per stage, a serial time and an auto time.
Two different questions come out of it and the round must not confuse them:

- **the serial column** is what L1, L2 and L5 move.
- **the ratio of the two** is what L3 and the core pool move.

The rule: **the round attacks stages in descending order of serial time, and
within a stage the choice between an inner-loop lane and a scheduling lane is
made by that stage's serial-to-auto ratio**, not by which lane is already
written. A stage below 5 percent of the serial total does not get a lane in
this round however unsatisfying its ratio is, because a lane costs a day and
cannot return more than the stage contains.

Registered before the data: `hist_full` plus the two `hist_subset` stages are
expected to dominate the serial column, at or above half of it, and
`partition` to be the next largest. **estimated.** If the histogram stages come
out below a third of the serial total, then L1, L2, L3 and L5 are all aimed at
the wrong place and this round is replanned rather than executed.

The instrument's own docstring warns that a single run of it is an estimate and
not a measurement. It is run at the same repeat count as everything else here,
and if a stage's repeats disagree by more than the effect being attributed to
it, that stage is reported as unresolved rather than ranked.

## C6. The exactness and determinism contract every lane inherits

Stated once, written into every lane brief verbatim.

- **Determinism across `MOJOTREES_NUM_WORKERS` (1, 3, 8) is required** and each
  lane writes a test that proves it for its own change. Determinism on a given
  toolchain, run to run and machine to machine, is not negotiable in this
  round. Bit-identity with *past* output is.
- **Exact comparisons only.** `to_bits()` or integer equality. No tolerances.
  A test that needed a tolerance is a test that did not establish what it
  claims.
- **A test for a gated path must prove the gate opened**: assert a counter, a
  trace line, or a path marker. Never assume it. This project has already
  shipped a test whose six fixtures all ran below the gate and verified
  nothing.
- **Any change that moves a multiply relative to an add is a numerics change.**
  FMA contraction has cost this project two results. If bits move, the lane
  says so and stops; it does not regenerate the golden fixture on its own
  initiative, because re-baselines are serialized under C-ops.

## C7. What is allowed to be called a result at the end of this round

Two things, and nothing else.

1. **A speed claim**: our CPU at 1,000,000 x 50, resolved by M0 over at least
   five interleaved repeats, against a LightGBM arm taken in the same process
   in the same window with its own spread reported, *and* the 50,000 and
   250,000 shapes reported beside it whether or not they moved in our favor. A
   headline that omits the shape that regressed is not a result.
2. **A null**: a lane that landed, is correct, is tested, and moved nothing.
   Reported as loudly as a win, per S4 and M3. Round 2's upload collapse is the
   precedent: thirteen copies per tree removed for sixteen milliseconds, and
   the honest report of that is worth more than the lane was.

Explicitly not a result: a per-stage improvement in `bench_profile.mojo` that
does not show up in the end-to-end interleaved comparison. The stage profile is
an instrument for choosing lanes, not for scoring them.

## C8. Thermal drift, which is this round's largest measurement hazard

Both arms of this round's headline comparison are the ten-core CPU arm. The GPU
campaign measured that arm drifting **2.80 to 3.50 seconds across five
back-to-back runs** as heat accumulated, while its own GPU arm held 2.7 percent
across the same sequence. This round does not get to sit behind a thermally
stable arm the way that comparison did.

Therefore, and registered before any data:

- Every CPU-versus-LightGBM comparison is **interleaved within one process**,
  never taken as sequential blocks per arm. Drift then hits both arms alike
  instead of accumulating into whichever ran last.
- **The repeats are reported in the order they ran**, so a monotone rise across
  a sequence is visible as a trend rather than absorbed into a spread. If both
  arms rise together across five repeats, that is the machine and not a result.
- Thermal state is captured before and after each session, per the session
  conditions above.
- A comparison whose two arms straddle a regime change is **discarded, not
  corrected**. Session III established that the regimes move a comparator by a
  factor of 2.2 and there is no correction factor that survives that.

---

# Amendments, 2026-08-16, written from what the protocol got wrong about itself

Session III is the first round in which these rules were applied end to end. They
worked: M3's registered falsification caught a prediction that was wrong by a
factor of forty, M0 twice stopped a number going into a table that did not
deserve to be there, and the gate-proof rule caught a test that verified nothing.
None of that is being weakened.

Three defects surfaced in use, and they are fixed here rather than worked around
a second time.

## A1. The thermal instruction was never followable. Replace it, do not keep it.

Session conditions above say to "capture thermal state into the results header
with `bench/apple/thermal_capture.sh`, before and after". That script measures
nothing: it is a plan printer, its own header says it "starts no sampler, runs
no privileged command, fits no model, and writes no record", and `--execute` is
parsed and deliberately refused with exit code 3 because the plan it prints
contains `sudo powermetrics`.

**A session did follow it, and that is worse than nobody having followed it.**
This paragraph originally read "No session has ever done this and none could
have", and the instruction audit refuted it from a committed artifact.
`bench/results/profile_2026-08-15/header.txt` opens a section headed
`=== thermal before ===` and pastes the script's output into it. What is pasted
is a plan, carrying `run id thermal-PENDING` and `would write
.../thermal-PENDING.json`. So the header of the first stage-level profile this
repository ever recorded carries a plan printer's plan under a heading that
reads as a capture, and has since `24e5330`. There is no matching
`=== thermal after ===`, so even the half that ran, ran once.

An instruction nobody can follow is bad because a protocol that lists it reads
as though the step is being taken. An instruction somebody *does* follow, which
returns a plausible block of text that is not a measurement, is worse: it leaves
a **record**, and the record is what the next reader trusts. That is exactly the
defect this protocol exists to prevent, sitting inside the protocol and inside
its own results.

**The instruction is now:** capture

    uptime
    ps -Ao pcpu,comm | sort -rn | head

before the first arm and after the last, into the results file, **including when
the box looks fine**. Both campaigns say they now do this; the audit found the
`uptime` half recorded in one results directory and the `ps` half in none, so
treat it as a rule that is not yet habit.

The script stays where it is, because the plan it prints is a real plan. **But
the pointer this paragraph used to give was itself dangling**, and the audit
caught it: `handoffs/performance_17_thermal_energy.md` was deleted on
2026-08-14 in `21ff9fa`, two days before this section was written to point at
it. `bench/apple/thermal_capture.sh --self-check` fails, exit code 4, on
exactly that missing file, and its `--execute` refusal message sends the reader
there too.

Until it is restored, the surviving listing of the privileged commands is
`docs/APPLE_THERMAL_ENERGY.md`, sections at lines 167-173 and 426-438, and the
full original is one command away:

    git show 21ff9fa^:handoffs/performance_17_thermal_energy.md

`bench/results/INSTRUCTION_AUDIT.md` section 10 reproduces the commands
verbatim. The script is simply not an instrument and the protocol must stop
implying it is.

**And the limitation gets stated wherever a regime is named.** `pmset -g therm`
returns nothing useful on Apple silicon. So "fast window" and "slow window" in
this repository are inferred **from effect** — both arms moving together — and
not read from any instrument. That inference has already been made and retracted
once: a slow pair was attributed to `mobileassetd` at 100 percent CPU, and after
waiting it out and confirming the box idle the numbers stayed high.

Note the trap that makes load average insufficient on its own: the slow regime
showed a **quiet** box and slow results at the same time. Load alone would have
said nothing was wrong.

## A2. M0 is too strict in the one case that matters most. Use paired differences.

As written, M0 resolves an A/B when the medians differ by more than the **wider
arm's own spread**. That is right for two arms of comparable stability and wrong
when one arm is an unstable comparator.

Session III hit exactly that. Five interleaved runs at 1,000,000 x 50 had our arm
at 2.7 percent spread and LightGBM drifting 24 percent upward from heat. Our arm
was faster in **five runs of five**, by 5.6 to 11.3 percent. Under M0 as written
that is unresolved forever, because no real effect of that size can ever exceed a
comparator's 24 percent drift. But five of five in one direction is a one-in-
thirty-two outcome under the null, and pretending it carries no information is
not conservatism, it is a different error.

**M0 is amended.** An A/B is **resolved** when either holds:

- **(a)** the medians differ by more than the wider arm's min-to-max spread, over
  at least five alternating pairs, as before; **or**
- **(b)** the arms are **paired** — interleaved in one process, or alternated
  process by process — and the **per-pair difference** has the same sign in at
  least five of five, or six of seven, pairs. The magnitude reported is then the
  median of the per-pair differences, with its own range, and never a difference
  of pooled medians.

Rule (b) is only available for paired designs, because it is drift that it
survives and pairing is what makes drift common-mode. It is not available for two
numbers taken in different windows, which remains forbidden.

Both figures get reported when they disagree, and which rule fired gets named.

## A3. Regime is part of a result, not noise to be averaged out

This machine drifts two- to threefold and **effect sizes move with it, not just
levels**. The resident plane measured 24 percent in a fast window and 8 percent
in a slow one, both resolved, both correct. The row unroll measured
*indistinguishable* at an 8.1 percent delta against a 14.1 percent floor in a slow
window and *resolved* at 10.8 against 2.1 in a quiet one — **same command, same
code, opposite verdict**.

So:

- A result carries its regime label. Where the effect size differs by regime,
  **report both numbers and refuse to pick one.**
- A null taken in a slow or dirty window **is not evidence of absence** and may
  not close a question. It may only defer it. Anything called indistinguishable
  outside a quiet window is retaken before it is used to cancel a lane.
- Prefer in-process interleaved arms over alternating processes wherever the
  harness supports both. Interleaving makes drift common-mode; alternating
  processes turns a neighbour's build into your result.

## A3b. Let the compiler enumerate, not the reader. Measured at 30x.

Not a measurement rule, but it belongs beside them because it is the same
discipline applied to code: **when a change has an unknown number of call sites,
make the old form fail to compile rather than grepping for it.**

The evidence, from the CPU campaign's `Histogram` field migration on 2026-08-16.
A careful hand survey by two orchestrators found "about eight couplings". Making
the fields private and letting the build fail found **233 direct reads across 35
files**, plus a ninth file-level coupling nobody had listed. **A factor of
thirty.**

Three details worth carrying, because each defeats a different hand method:

- **Most of the hand list was false positives.** Eight modules that looked like
  couplings read `Tree.count`, `QuantizedHistogram.grad`, `FeatureTotals.grad` or
  `PackedHistogram.count` -- different structs that share the field names and
  nothing else. Grepping a field name finds the wrong set in both directions.
- **Five test files read a plane without ever naming the type**, because the
  value comes back from a builder with an inferred type. `grep Histogram tests/`
  misses all five.
- **The compiler-driven pass still made ten errors**, in the direction reading
  cannot catch: it converted `local.grad[b]` where `local` was a different struct
  with plain fields. A false positive compiles until it does not, and the suite
  caught it.

The limit, stated so nobody over-trusts it: Mojo has no field privacy, so a
positional constructor still compiles and the single-chokepoint constructor is
convention rather than a compiler guarantee. That gap closes at the storage
change, which necessarily alters the signature.

## A5b. Audit a document for the condition that would falsify it, not for errors

The CPU campaign's diagnosis of its own `ACCURACY_BUDGET.md` defect, kept because
it names a class rather than an instance:

> The real defect was not the algebra. It was that a document about numerical
> accuracy discussed a gain form and **never mentioned L1 anywhere** -- so nothing
> prompted a reader to check the one condition that breaks it.

The algebra in that section was correct. It was correct *under an assumption the
document did not state and therefore did not invite anyone to test*: that
`GL + GR = G`. Soft thresholding is not additive, so at `lambda_l1 > 0` the
identity fails, and applying it anyway is a **systematic bias** of 1.6e-04
relative where the shipped form sits at 1.0e-05 -- median and p99 agreeing to two
figures, which is how you tell bias from rounding.

**The audit question is therefore not "is this wrong" but "what setting would
falsify this paragraph".** A reader checking for errors finds none; a reader
asking which regularizer, which objective, which sampler, which bin count would
break the claim finds the gap in one pass.

This is the same shape as amendment A1 and the instruction audit. There the
failure was an instruction nobody could execute; here it is an assumption nobody
was invited to check. Both are silent, both survive review, and both are found by
asking a document what it depends on rather than whether it is right.

Neither campaign has read its documents this way yet, and both should.

## A4. What this says about the protocol as a whole

It is working, and the evidence is that every one of these defects was found by
following it rather than by abandoning it. A protocol that never fires against
its author is decoration; this one refuted its author's main claim of the round
within an hour of the data arriving.

The failure mode to watch is not strictness, it is **inherited instructions that
nobody executes**. A1 was one. Anything in this file that has never actually been
run should be treated as suspect until someone runs it.

## A5. The regime now has an instrument, and it is not calibrated yet

A1 replaced an unfollowable instruction with a followable one and was honest
that the replacement is weak: `uptime` and the top processes did not, and could
not, catch the Session III slow window, which showed a **quiet box and slow
results at the same time**. A3 then made regime part of every result. Between
them the protocol requires a label it has no way to read, so every regime label
in this repository is inferred by hand from effect sizes -- an inference that
was made and retracted the same night.

`bench/canary.mojo` is the instrument. Two fixed probes, a single-threaded CPU
mixing chain and a saturating GPU kernel, neither of which touches
`src/mojotrees/`, a dataset, a thread count, or anything else a session varies.
`bench/bench_train_gpu.mojo` runs both **first and last** in every run and
reports `canary_cpu_ratio` and `canary_gpu_ratio` against a recorded baseline,
plus the raw milliseconds, plus a `canary` object on the `json_summary` record.
The two engines are reported separately and never averaged, because Session III
measured the CPU degrading by 2.2 in a window where the GPU degraded by 1.5.

**Three rules follow, and they take effect the moment a baseline exists.**

- **A results header states `canary_cpu_ratio` and `canary_gpu_ratio`, or
  states that no baseline was recorded.** A regime named without one is an
  inference from effect and is labelled as such, exactly as A1 requires today.
- **A run whose canary reports `SHIFTED-DURING-SESSION` is discarded, not
  corrected.** This is C8's existing rule with the straddle now detectable
  instead of noticed afterwards. The threshold is 5 percent of the smaller
  reading, which is **chosen and not measured**: the smallest effect this
  repository has ever resolved was 10.8 percent, so a machine that moved five
  percent between the first arm and the last was already moving by half the
  size of what is being hunted.
- **The canary does not replace the A1 capture.** `uptime` and the top
  processes say *what* was running; the canary says *whether it mattered*. Both
  go in the header. The canary's blind spot is the mirror of A1's: it reads the
  machine's delivery and cannot name the cause.

**The baselines have not been measured and the file ships with them null.**
Establish them on a verifiably quiet box with

    pixi run mojo run -I src bench/bench_canary.mojo 7

and paste the block it prints into `bench/canary_baseline.json`, filling in the
date, the toolchain, the machine, and the evidence the box was quiet. Until
then the mechanism reports raw milliseconds and says the ratio is unavailable,
which is correct behavior and not a gap to be closed by filling in a plausible
number. A fabricated baseline in a regime detector does not fail loudly; it
mislabels every session afterwards in a form that reads as data.

And per A4: **this has never been run in anger.** The probe constants were
sized from an estimate, no baseline exists, and no session has yet carried a
canary line. Treat it as suspect until someone does.

---

# The build/gate rule, 2026-08-16. Supersedes every size threshold in every brief.

This replaces the ~2 percent floor, every "small, not worth it" verdict, and
every estimate-based decision not to build. **Nothing is dropped for being
small.**

1. **Strictly-less-work-and-exact changes get BUILT with no measurement gate**,
   largest first. Fewer bytes, trips, launches, atomics, allocations; same
   result. **The wave window measures them; it does not gate them.**
2. **Trades ship behind a switch and are A/B'd before becoming default.** Task
   and tile counts, floors, group widths, layouts -- anything that *moves* work
   rather than removing it.
3. **Bit-moving changes take the real-data gate** against stock+det.
4. **Items close ONLY when proven zero or impossible**, with the evidence
   recorded. An estimate is not a proof and an author's own expectation is not a
   proof.

## What this reopens on the GPU side, and what stays shut

Reopened, with the category:

- **trip-count** (1) -- running now. Note the target rescaled: a trip is
  **measured** at 202 us, not the 458 us derived, so 200 trips to ~10 is worth
  about **0.038 seconds**, not 0.09.
- **atomic-halving** (1 where the per-node bound holds) -- after the
  decomposition probe, only because it shares kernel bodies with whatever that
  probe selects. **See the correction below: half of it already exists.**
- **ellpack-bins** (1) -- after the CPU per-feature-width spec, regardless of the
  gather fraction.
- **row-tiles DOWN, the 1-tile arm** (2) -- never tested in that direction; in
  the window.
- **feature_group 4** (2) -- group 2 over 1 measured at 1.17x, group 4 unmeasured;
  switch exists; in the window.
- **K1 item 1, index-width narrowing** (1) -- its author called it an expected
  null *on reasoning*, which rule 4 says is not proof. Already built as
  `set_narrow_index` and already an arm, so it is in the window with no new work.

Still closed **with evidence**: `DeviceGraph` on Metal; device row-major bins at
256 bins (reopens only if the decomposition shows the gather under ~10 percent);
K >= 2 speculation; feature-blocked layout at group 1; CPU fallback for wide
features; and the thirteen async pinned copies, which are **proven zero** -- they
never waited, so removing them could not have bought anything.

## Correction: half of atomic-halving is already built and shipping

`_hist_rows_step` carries a comptime `CELIDE` parameter, and under it **the
hessian atomic is already skipped**: the inner loop issues `sg` and `sc` and not
`sh`, and `hist_planes` is 2 rather than 3.

So on any round with a constant hessian -- which includes **unweighted non-GOSS
squared error, the objective every headline figure here uses** -- the kernel
already pays **two shared atomics per (row, feature), not three.**

Two consequences that must travel with the atomic-fraction number:

- **The measured 9.8 percent is the cost of TWO atomics, not three.** Every
  statement of the form "the three shared atomics" in this campaign's lane
  reports, including K3's and the bin-layout lane's, is describing the
  weighted/logistic path rather than the measured one.
- **What is left of atomic-halving is the packed grad+hess variant only**, taking
  two atomics to one. Against a measured 9.8 percent for two, the remaining prize
  is bounded near **5 percent** of the histogram phase, and it is bounded by
  whether the per-node bound holds often enough to select the narrow variant.

That does not close it -- rule 1 says build it -- but it prices it honestly
before anyone builds, which is the point of writing the bound down first.
