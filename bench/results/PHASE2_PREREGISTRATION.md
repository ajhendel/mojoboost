# Phase 2, the kernel batch: registered before any lane launches

Written while the box was held by the CPU campaign and this session was
compiling nothing. Nothing here has been measured. Every estimate below exists so
that the measurement can **refute** it rather than be fitted to it, which is the
one practice that has repeatedly worked this week: a registered prediction was
wrong by a factor of forty and said so, and a registered rule killed another
campaign's largest speculative lane against its author's own prediction.

Read `PROFILE_PROTOCOL.md` "HOW TO TAKE A NUMBER" first. Verdicts by **M0 as
amended by A2** (paired designs may resolve on the sign of the per-pair
difference). Every result carries a **canary line** and a regime label.

## Why the kernel, and why now

The control plane is finished and the file says so. Sixteen host waits per tree
became three; the three that remain are one round trip, roughly 100 per fit, an
**estimated** 0.05 seconds that M0 cannot resolve on this machine. **No further
control-plane lane is justified.** The measured evidence that the kernel is where
the remaining margin lives:

- the row unroll alone was **measured** at 10.8 percent of a 1M fit, resolved,
  and 2.14x on the histogram kernel in isolation
- the upload collapse, thirteen copies per tree, was **measured** at 0.016
  seconds: a null

One is a kernel change and one is a control-plane change, and they differ by
nearly two orders of magnitude in what they bought.

## The four lanes, file-disjoint

### K1, hist-latency. Owns kernel bodies and `gpu_tiling.mojo`.

Int32 index math, a vectorized `gq` load, then re-test 4-8 row tiles **now that
partial traffic is known to be the confound** that made the earlier tile
experiment a 22-36 percent regression.

- **Estimate:** 10-25 percent on the histogram phase.
- **Refutation threshold, with a regime condition that is part of the rule:**
  refuted if the whole fit does not move by at least **0.1 seconds at 1,000,000
  x 50**, measured with **interleaved arms in one process, resolved under M0,
  and the canary reporting an unshifted fast window**.

  **If the window is not fast, the lane is neither confirmed nor refuted and the
  measurement is re-taken, not filed.** 0.1 seconds is 4 percent of a 2.58
  second fit, against an arm spread of 2.7 percent in a fast window and 11 to 14
  percent in a slow one. Without the regime condition a real 4 percent win is
  filed as a null in any bad window -- which is exactly what happened to the row
  unroll, called indistinguishable at 8.1 percent against a 14.1 percent floor
  and resolved at 10.8 against 2.1 four hours later. Same code, opposite verdict.
- **End-to-end condition:** an isolated-kernel improvement that does not move the
  fit is not a win. Both numbers are reported and the fit is the one that
  decides. See K3, where the same rule is argued at length.
- The tile re-test is the part most likely to fail again. It failed once with a
  well-argued mechanism and a satisfied amortization bound. If it regresses a
  second time, the standing bound `MIN_ROWS_PER_TILE_BIN_FACTOR = 8` is itself
  wrong and that is the finding.

### K2, speculative K=1. Owns `gpu_resident_round.mojo`.

Prebuild the top runner-up's child histograms while the pick commits. Exact by
construction: a child histogram is valid whenever it is computed. A census found
the greedy pick is the top runner-up in **100 percent of 4,030 decisions**, so
exactly one speculative candidate suffices.

- **Estimate:** 5-15 percent at 1M, more at 50k where launch shape dominates.
- **Refutation threshold:** no measurable movement at **either** 50k or 1M.
- **The test must prove the speculative build was CONSUMED, not merely
  launched.** This project shipped a test whose six fixtures ran below the gate
  they were testing and verified nothing; a speculative path that always
  launches and never hits would pass any test that only asserts it ran.
- **Report a consumed fraction and a wasted-build fraction as numbers in the
  results, per shape.** The census that justifies K=1 -- the greedy pick being
  the top runner-up in 100 percent of 4,030 decisions -- was taken under
  conditions close enough to synthetic that it should not be trusted as the real
  hit rate. **The measured hit rate is what makes K=1 sufficient or not**, and if
  it comes in materially below 100 percent then K=1 is the wrong K and the lane
  reports that rather than shipping a speculation that misses. A wasted build is
  real device work spent on a child that is discarded.

### K3, feature-blocked layout. Owns `gpu_binned_layout.mojo`, `gpu_bin_packing.mojo`, and a NEW kernel variant file.

Device-side `[group][row][G bytes]` re-layout at upload, plus a reader kernel
behind a comptime flag. **Sequenced after K1 and measured after merge**, never in
parallel with it — two changes to the same phase measured together cannot be
attributed, which is the mistake two lanes made in the last round when each
correctly reported a wait count against its own baseline and neither figure was
the total.

- **Estimate:** 1.5-2.5x on the histogram phase. The largest single item left.
- **Refutation threshold, AMENDED before launch:** below 1.2x **measured in-run
  by node size class**, not on the isolated histogram benchmark.

  The amendment is owed to the CPU campaign, which retracted a Phase 0 reading
  built on exactly that instrument. Its isolated synthetic kernel calls said
  histogram accumulation scales 3.25-3.40x; measured inside real trees the same
  quantity is **2.13x overall and falls off a cliff as nodes shrink** -- root
  2.69x, large 2.47x, medium 1.92x, small 1.20x, tiny 1.39x. It had already
  demoted a lane on the wrong number.

  A GPU kernel change is exposed to the identical error, and more so: a layout
  change is evaluated on full-width root-sized histograms in isolation, while
  most nodes in a real tree are small. **An isolated 2x that is 1.1x on the
  small-node classes is not a 2x.** So K3 reports both, and the in-run figure by
  size class is the one that decides.

  This project has one case where the two instruments agreed -- the row unroll,
  2.14x isolated and 10.8 percent of the whole fit, both resolved -- and that
  agreement is why the unroll result stands. Agreement is the thing to check
  for, not to assume.
- **Refuted if EITHER the isolated figure is below 1.2x by size class OR the fit
  moves less than 0.15 seconds at 1,000,000 x 50.** Both, not either alone.
  The row-tile floor is the case in point: it raised occupancy exactly as
  designed, was the change its author was most confident in, and measured **22
  percent slower at 50 features and 36 percent slower at 100** in a real fit. An
  isolated win that vanishes end to end is not a win, and this repository has
  already paid for that lesson once.
- It gets a **new** kernel variant file rather than editing K1's body, so the two
  can be held as arms in one binary.

### K4, auto-reaches-gpu. Owns `device_policy.mojo`, `device.mojo`.

`DeviceCapabilities.detect()` cannot fire the M4 rule, so `device='auto'` never
selects the GPU. A shipped-default bug, not a performance lane, and it ships now.

- **No estimate and no threshold**, because it is a correctness fix. Its
  measurement is that `auto` selects the GPU above the measured crossover and
  does not below it.

## What this batch does not do, and why each is excluded

- **No further wait-count work.** Measured at 0.016 seconds for thirteen copies
  per tree. The chapter is closed and the docstrings say so.
- **No depthwise-specific work.** It is a benchmark row and an opt-in under S2,
  and it becomes the commit rule inside the one plane rather than a second path.
- **No int16 packing.** The repo's own bound `2^(W-1)-2` gives 32,766 safe rows
  against a 1,000,000-row target, and packing makes overflow carry into the
  neighbouring field, which destroys the modular-arithmetic argument that makes
  sibling subtraction exact.
- **No more hygiene lanes.** The last batch was six of them. Justified once,
  after the week this was; twice would be drift.
- **Nothing touching CPU-campaign files** without a stop-and-report.

## The shape of the claim this batch is trying to earn

At 1,000,000 x 50 in a fast window the current position is 2.58 seconds against
LightGBM's 2.66 to 2.95, consistently ahead by 5 to 11 percent and **not
resolved** because the comparator's own spread is wider than the margin.

If K1 and K3 both land at the low end of their estimates the fit goes near 2.0,
which would be a margin wide enough to resolve against a comparator that drifts.
**That sentence is an estimate over two unmeasured lanes and must not be quoted
as a projection.** It is written down so that failing to reach it is visible.


## Phase 1 sequencing, registered with the rest

- **The retake does not start when the lock clears. It starts when the 5-minute
  load average is under about 4.** The 1-minute figure recovers long before the
  machine does: at the moment the box was handed over it read 2.56 over one
  minute against 7.65 over five and 12.76 over fifteen. Starting on the 1-minute
  number would measure the tail of a six-lane batch and call it a fast window.
- **Push before promoting**, so the CPU-only CI matrix runs on the exact SHA that
  gets promoted rather than on an ancestor. The CPU-only build is the one an M4
  structurally cannot catch locally -- a GPU entry point missing its `comptime if
  not has_accelerator()` guard dies with "Unknown GPU architecture" only on the
  x86-64 half.
- **The deletion commit touches `train_gpu.mojo` only**, and the API snapshot is
  regenerated **once, after it**, since exports move. Not per lane: four lanes
  each regenerating one generated file is four conflicts on it, which is what
  happened last round and why it is written down here.


---

## K1's estimate, revised DOWN by its own author before any measurement

Registered above: 10-25 percent on the histogram phase. **K1's own report puts it
at 0-10 percent**, and the revision is recorded here, before the arms have been
run, so the eventual measurement is compared against the honest prediction rather
than the flattering one.

Its reasoning, and it is a structural claim rather than a hedge:

- **Item 2, the vectorized `gq` load, was already there.** It landed with the row
  unroll. What K1 found is that `unsafe_load[width=2]` with no explicit alignment
  emits `align 4`, and the explicit spelling emits `align 8` -- **observed from
  emitted LLVM IR**, by compiling both spellings and diffing, not measured on a
  device. An under-aligned vector load is one a backend may split back into the
  two scalar loads the width-2 spelling existed to replace. So this item is not a
  new optimization; it is a check on whether last round's change bought anything
  at all. Largest of the three if Metal was in fact splitting it, zero if not.
- **Item 1, Int32 index math, is registered as an expected null by its own
  author.** Of the two narrowable quantities, one is a loop-invariant base a
  compiler should already hoist, and the other is the induction variable where
  narrowing trades a 64-bit add for a 32-bit add plus a widening at every use.
  Against three shared atomics and a scattered gather per (row, feature), one
  index add either way is noise.
- **Item 3, row tiles, is expected to fail in the direction registered.** But it
  found something the earlier experiment missed: **the earlier run only ever
  tested tiles going UP** -- 80 tiles, a device-wide floor -- and lost 22 and 36
  percent. **Downward has never been tested.** At 50 features the default is only
  2 tiles, so if per-tile zero-and-flush traffic is what dominates, which is what
  the earlier loss argues, then **1 tile is the untested arm most likely to win**
  and 4 and 8 are most likely to lose again.

### The observation that matters more than the lane

> the inner loop's cost is the scattered `bins[feature * n_rows + row]` gather
> and the three shared atomics per (row, feature), not index arithmetic and not
> the gradient read

Index-width and load-width work **cannot reach either of those**. That is K3's
`[group][row][G bytes]` re-layout, and this lane's audit is indirect evidence
that K3 is where the remaining margin actually lives. K1 is a check on an
existing change and a first honest test of the tile direction nobody tried.

One consequence for the shipping configuration specifically: the gradient-pair
load is amortized over `GROUP` features, and the default `feature_group` is 1 at
256 bins, so **item 2's share is largest exactly at the default and shrinks if
anyone widens the group**.
