# The wave-4 CPU window, 2026-08-16

Branch `cpu-round-1` at `9cc45e7`. Comparator: **LightGBM stock defaults plus
`deterministic=true`**, label `stock+det`, registered in `PROFILE_PROTOCOL.md`
section C9. Headline is **end to end**: ingestion plus binning plus training.

Box: f9 confirmed drained at 09:06:44, zero mojo processes, verified with
`ps -Ao comm | grep "mojo$"` rather than `pgrep -fl mojo` (see the last section).
Lock held at `/tmp/mojotrees-bench.lock` throughout; f9 compiled nothing.

## The headline, 1M x 50 regression, 100 rounds

The one run whose canary said `stable` (CPU drift 0.3 percent, GPU 1.4):

| | binning | training (median of 5) | **end to end** |
|---|---|---|---|
| mojotrees CPU | 0.1206 | 6.4529 | **6.5735** |
| LightGBM stock+det | 0.3263 | 3.4305 | **3.7569** |

**LightGBM is 1.75x ahead end to end.** Verdict `resolved`: the training delta
is -46.8 percent against a noise floor of 7.7.

Two things this decomposition says that the training-only figure could not.

**Our binning is 2.7x faster than theirs**, 0.1206 against 0.3263. That is a
real win and it is invisible in every training-only number this project has
quoted. It is also the argument for the end-to-end headline, made by the first
run taken under it.

**And training carries the entire loss.** We are 1.88x behind on training and
2.7x ahead on binning, and binning is small enough that the end-to-end gap
(1.75x) is barely better than the training gap. There is no ingestion story to
tell here. The work is in the histogram and the split search.

Train loss agrees to three digits, 0.0035931 against 0.0034837, so this is a
speed gap and not two different models.

**One caveat on the binning column, which I did not notice until after I had
quoted it.** The harness times binning **once** per invocation, before the
repeat loop. Every training figure here is a median of five or twelve; every
binning figure is a single sample with no spread. Adding one to the other is
not wrong, but the end-to-end totals inherit an unrepeated term.

The 2.7x binning claim survives it only because three separate 1M invocations
give three samples per side: ours 0.1206, 0.1046, 0.1104, theirs 0.3263,
0.3120, 0.3156. Medians 0.1104 against 0.3156 is **2.86x**, tight on both
sides, and that is the number to quote rather than the single-run 2.7x.

The smaller sizes do **not** survive it and must not be read: our binning
single samples are 0.0481 at 50k and 0.0390 at 250k, which is a *smaller*
number at five times the rows. That is a first-touch or page-fault artifact,
not a measurement, and it is the clearest evidence that the binning term needs
repeats before any end-to-end total below 1M means anything.

## MOJOTREES_CPU_TASK_FLOOR: a win at 50k, nothing at 250k

**The floor is resolved at 50k and it is a win.** Twelve repeats, plateau
compared (last four), floor-off taken **first** on the cooler box so the
ordering worked against the winner:

| 50k x 50 | mojotrees plateau | LightGBM plateau | ratio | canary |
|---|---|---|---|---|
| floor on (default) | 0.5177 | 0.5001 | **1.0352** | stable, 0.2 pct |
| floor off | 0.5455 | 0.4918 | **1.1092** | cpu 0.4 pct (see note) |

**5.4 percent on raw training time, 6.7 percent on the anchor-normalized
ratio, and the two plateaus do not overlap**: floor-on's slowest plateau
sample is 0.5297 and floor-off's fastest is 0.5371. The LightGBM anchor moved
1.7 percent between the two invocations, so the box was comparable and the
effect is not the box.

This is what the floor's own docstring predicted and could not test. Its
argument is that a `sync_parallelize` pays its wake and barrier in full the
moment the loop is declared parallel, so a task count below the core count
leaves cores idle behind a cost already bought -- which bites where work sits
between one grain and the machine ceiling, meaning **small nodes**. 50k is
that regime. 250k is not, and 250k is where it measures as nothing.

Note on that canary: the floor-off run at 50k is marked
`SHIFTED-DURING-SESSION`, but its **CPU** drift is 0.4 percent and the shift is
entirely the GPU probe at 6.9. The regime verdict ORs the two engines, so a
CPU-only arm pair can be failed by a GPU probe that no CPU number depends on.
The run is kept on that basis and the reasoning is stated rather than the
verdict quietly ignored. **The verdict should be per-engine**; it is not.

### The same A/B at 250k: indistinguishable

The floor is **on by default** and its own docstring says it is unmeasured.
This A/B was to answer it. It does not.

Compared at the **plateau** (the last four of twelve repeats) and normalized by
the LightGBM arm measured in the same process and the same window, which is the
only way to compare across two invocations on this box:

| | mojotrees | LightGBM | ratio |
|---|---|---|---|
| floor on (default) | 1.7243 | 1.0443 | 1.6512 |
| floor off | 2.0717 | 1.2668 | 1.6354 |

**1.6512 against 1.6354 is a 1 percent difference and it is not resolvable.**

The raw times invite the opposite conclusion and it would have been wrong. On
raw medians at five repeats, floor-on looked 5.8 percent faster (1.5750 against
1.6668) and I would have called the floor a win. The LightGBM anchor shows why
not: it moved 20 percent between the two invocations (1.0443 to 1.2668) on an
arm neither setting can touch. **The whole of that 5.8 percent was the box.**

At this size the floor is neither vindicated nor refuted. Taken with the 50k
row above, the two together say something coherent and better than either
alone: **the floor helps where task counts fall below the core count and does
nothing where they do not.** It stays on by default, which is now a measured
setting at 50k and an unmeasured one at 250k and 1M.

### The scale picture, and where we actually stand

| rows | our training | LightGBM training | ratio | verdict |
|---|---|---|---|---|
| 50,000 | 0.5177 | 0.5001 | 1.04x behind | indistinguishable |
| 250,000 | 1.7243 | 1.0443 | 1.65x behind | resolved |
| 1,000,000 | 6.4529 | 3.4305 | 1.88x behind | resolved |

**We are level at 50k and the gap opens monotonically with scale.** That shape
matters more than any single number: a per-row cost we pay and LightGBM does
not would look like this, and a fixed overhead would look like the opposite.
Whatever is wrong is in the part that grows with rows -- the histogram build
and the row partition -- and not in per-tree or per-node overhead.

The "wins at 50k" claim carried in the standing scorecard **does not reproduce
under this comparator**: at 50k we are 1.04x behind, not ahead. That is
expected rather than alarming, because the comparator changed twice since that
claim (to stock defaults, then to `lambda_l2 = 0`) and the earlier figure was
against a pinned LightGBM. The scorecard line should be restated or dropped.

## What the window found about the box, which changes the protocol

**Repeats inside one process are not exchangeable. They warm and plateau.**
Twelve repeats at 250k, canary `stable` at 0.5 percent drift, so this is not
drift:

- mojotrees: 1.475 rising to about 1.72, **+17 percent**, flat from repeat 9
- LightGBM: 0.948 rising to about 1.05, **+11 percent**, flat from repeat 7

I first read the rise as ours alone, because at five repeats LightGBM's rise is
inside its own noise and looks flat. It is not flat. Twelve repeats shows both
arms doing the same thing. **There is no leak in our arm** and the claim that
there was is withdrawn here rather than carried.

Two consequences, both real:

**A five-repeat median measures the transient.** The median of five is the third
sample, taken while both engines are still climbing. Every number this project
has taken at five repeats sits between cold and steady state, at neither.

**We degrade more under sustained load than LightGBM does.** The ratio moves
from 1.565 cold (first two repeats) to 1.651 warm (last four). Our
disadvantage is 5 percent worse in steady state than a cold benchmark shows,
and steady state is the regime a user training all day is in.

**And the box throttles across runs, hard.** Three 1M runs back to back, no
competing process, load 5 to 6, no thermal warning recorded: LightGBM's own
median went 3.43, 3.96, 4.80, which is 40 percent on an arm nothing touched.
The canary caught both of the degraded runs at 65 and 70 percent CPU drift and
they are discarded, not footnoted. Only the first 1M run is quoted above.

**The working protocol this implies**, and it is not what section C9 currently
says: one arm pair per canary window, short runs, a real cooldown between them,
`SHIFTED-DURING-SESSION` treated as a discard, and cross-invocation comparisons
normalized by an in-process anchor rather than taken on raw times.

## Discarded, and why, so nobody re-quotes them

| run | canary | why discarded |
|---|---|---|
| 1M, floor off, first take | shifted 69.9 pct | box degraded mid-run, both arms |
| 1M, floor off, second take | shifted 65.1 pct | same |
| 250k, floor off, 5 repeats | shifted 5.3 pct | marginal, over the 5.0 threshold |
| 250k, floor off, 12 repeats | shifted 26.6 pct | box warming from my own load |

Note that at 250k the floor-off arm is discarded in **every** take, and in each
one it was the *second* run of a pair on a warming box. That is why the 50k
pair was taken with floor-off first, and why the 50k result is the one quoted:
there the ordering handicapped the setting that won.

## Not in this window

**Row-major is not an arm.** `lane/row-major-bins` was merged, found red
against the interleaved histogram cell, and reverted. It is with its lane.

**Head against `e8c0877` is not taken, and the instruction to caveat it is not
enough.** `lambda_l2` moved from 1.0 to 0.0 this round. That is not only a
model difference to be footnoted: `lambda` sits in the split gain, so the two
builds **grow different trees**, and different trees take different time to
grow. A head-versus-`e8c0877` wall clock would be a speed comparison and a
tree-shape comparison added together with no way to separate them, and no
caveat text makes such a number mean anything.

The fix is cheap and I am recommending it rather than taking a confounded
number: run **head with `lambda_l2 = 1.0`**, matching `e8c0877`'s default. Then
the trees are the same and the difference is speed alone. The bench harness has
no flag for `lambda_l2`, so this needs one line of glue before the run. Owed,
with that shape specified.

## Provenance

Everything in the headline table and the plateau table is **measured**, on this
box, on 2026-08-16, inside the lock, with the canary verdict recorded per run.
The steady-state ratio figures are **derived** from measured samples by taking
the last four. The reading of *why* the box throttles is **not established**:
no thermal warning was recorded and I did not instrument power or frequency.

The box-state check `pgrep -fl mojo` **reports a busy box when the box is
idle**: it matches every Chromium `mojom` helper in VS Code, Chrome and Docker.
The anchored form `ps -Ao comm | grep "mojo$"` is correct and is what both
sessions used here. This is the grep bug on the housekeeping list; it is real
and it was hit twice today.
