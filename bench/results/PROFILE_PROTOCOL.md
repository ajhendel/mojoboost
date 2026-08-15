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
