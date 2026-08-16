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
- **Refutation threshold:** if the whole fit does not move by at least **0.1
  seconds at 1,000,000 x 50**, the lane is refuted and does not get a second
  round. Registered because the unroll moved 0.284 seconds and set the scale.
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
