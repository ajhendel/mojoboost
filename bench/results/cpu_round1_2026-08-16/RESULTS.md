# CPU round 1, clean window and the in-run scaling profile. 2026-08-16

> ## SUPERSEDED COMPARATOR. Every speed ratio below is NOT PUBLISHED.
>
> **The reason changed the same day; the conclusion did not.** These figures
> were first marked superseded because the comparator was to become LightGBM
> plus `use_quantized_grad=true`, which our CPU float path could not match.
> That comparator is withdrawn. The comparator is now **LightGBM stock
> defaults plus `deterministic=true`**, labelled `stock+det`, and our float
> path **is** like-for-like against it.
>
> **These numbers are still superseded, for a different and simpler reason:
> they were taken against the OLD pinned configuration** — LightGBM forced to
> `force_row_wise`, to `min_data_in_bin=1`, and to bin every row via
> `bin_construct_sample_cnt = n_rows`. Every one of those pins is being
> dropped. A ratio against a comparator we required to do more work than it
> would choose is not a ratio against stock.
>
> **What changes going forward:** once the harness lane lands `stock+det`, CPU
> speed numbers are publishable again, taken fresh. The quantized-gradient
> lane no longer gates that.
>
> **The project now has exactly one comparator: LightGBM at stock defaults
> plus `use_quantized_grad=true`, with its own defaults for the
> sub-parameters.** Every mojotrees arm runs with quantized gradients on. The
> GPU already does, in fixed-point Int32. The CPU does not yet: it needs a
> `use_quantized_grad` option built to LightGBM's scheme, which is a lane that
> has not started.
>
> **Until that option exists, a CPU-versus-LightGBM speed number compares our
> float path against their quantized path.** That is not a like-for-like
> comparison and Andrew's ruling is that such numbers are **not published**
> rather than published with a caveat.
>
> So every ratio in this file — 2.01x behind at 1,000,000 rows, 1.49x at
> 250,000, 4.6 percent ahead at 50,000 — is **superseded and must not be
> quoted**, including by this campaign's own later write-ups. They are kept
> here because the file is a record of what was measured and when, and
> deleting a measurement because its comparator changed is how a project loses
> the ability to explain its own history.
>
> **What survives, and is still worth reading:** everything in this file that
> is a comparison of mojotrees against *itself* — the in-run scaling profile
> by node size class, the retractions, the instrument corrections, and the
> golden-fixture coverage finding. None of those involve LightGBM's
> configuration and none are affected.
>
> **The binning ratios were already withdrawn** for an unrelated and
> independent reason (the comparator was pinned to bin every row), and remain
> so. They now fail on two counts rather than one.

Taken under `mode: timing` with the GPU campaign holding every compile and
commit. Protocol in `../PROFILE_PROTOCOL.md` sections C0-C8. Saved before
interpretation.

Branch `cpu-round-1` at `8bd9f23`, which includes the dispatch-wiring lane and
its `tree.mojo` glue. 100 rounds, 31 leaves, 255 bins, squared error. LightGBM
4.7 in the same process, pinned to 10 threads, `force_row_wise` (see C3).

## Box state. This window did NOT meet the precondition, and is labelled accordingly

**This is a fast-regime retake, not a clean window, and nothing here upgrades a
verdict.**

The precondition in `PROFILE_PROTOCOL.md` is an idle machine. What this window
actually had:

- Load average **2.98 / 13.35 / 15.91** at the first arm. The one-minute figure
  was still decaying from the preceding compile storm; the five- and
  fifteen-minute figures show what the box had just been doing.
- No builds, verified: the other campaign held every compile and commit, and
  `pgrep` for `mojo run|mojo precompile|pixi run` was empty at the start of
  every arm.
- During CW1 my benchmark held 366 percent CPU, `Code Helper (Renderer)` 16
  percent, nothing else above 7. The other session observed a transient VS Code
  spike to **60 percent** mid-window that my captures did not catch.
- The regime canary, on two fixed probes with no training code in them, reports
  **`drift_cpu_pct: 22.5`, `drift_gpu_pct: 0.7`, `regime: shifted`** inside a
  single short run.

I began the arms on the judgment that instantaneous CPU was idle even though
the one-minute load average had not yet fallen below the threshold I had set
myself. **That judgment was wrong by the rule as written**, and the rule is the
one that matters, so this window is labelled rather than defended.

**Specifically: the 50,000-row question is NOT settled by this window and the
50k result is NOT upgraded.** It stays *consistent, not resolved*. This
project's only claimed win over LightGBM does not get promoted on a window that
failed its own precondition.

## The headline, and what moved

All verdicts computed by hand under **M0 on medians**, not from the harness's
printed line (which computed from minima; the other campaign has since fixed
that at `cf2eeb0`, landing after these runs).

| shape | ours | LightGBM | ratio | Phase 0 ratio | verdict |
|---|---|---|---|---|---|
| 1,000,000 x 50 | 5.866 | 2.915 | **2.01x behind** | 2.04x | resolved |
| 250,000 x 50 | 1.617 | 1.082 | **1.49x behind** | 1.55x | resolved |
| 50,000 x 50 | 0.550 | 0.576 | **4.6% ahead** | 9.6% ahead | **consistent** |

Samples in run order, per C8:

- 1M ours: 5.496 5.716 5.866 6.127 6.335 (spread 15.3%)
- 1M LightGBM: 2.705 2.809 2.915 2.944 3.047 (spread 12.6%)
- 250k ours: 1.515 1.538 1.781 1.617 1.637 (spread 17.5%)
- 250k LightGBM: 0.999 1.087 1.082 1.095 1.059 (spread 9.7%)
- 50k ours, 9 repeats: 0.493 0.628 0.568 0.552 0.570 0.542 0.545 0.550 0.547
- 50k LightGBM, 9 repeats: 0.586 0.705 0.576 0.565 0.596 0.586 0.576 0.570 0.565

### The dispatch-wiring lane is a NULL, and it is reported as one

The lane removed roughly 86,300 `getenv` calls and 18,500 core detections per
fit (**derived bound**, its own arithmetic, verified by a poison test proving
200 node builds read the environment exactly zero times).

The ratio at 1M went **2.037 to 2.012**. At 250k, 1.554 to 1.494. Both are
inside the run-to-run variation of the arms that produced them, and no
statistic here can separate a 1 percent effect from this machine's drift.

**So: correct, tested, proven wired, and worth nothing measurable.** Exactly
what the lane predicted for itself, in the section of its report where it
argued against its own value. It stays — it deletes real work, it makes the
policy a fit-scoped fact rather than a per-call re-derivation, and it costs
nothing — but it is not a speed result and will not be described as one.

### The 50,000 row win got weaker with more evidence, and it is still not resolved

Phase 0 had 5 pairs and a 9.6 percent median margin, verdict **consistent**.
Nine pairs gives a **4.6 percent** median margin, and under M0 it is **still
consistent, not resolved**: the median delta is 0.026 against a wider-arm
min-to-max range of 0.135.

**But the direction is now unanimous across fourteen independent pairs**, five
in Phase 0 and nine here, without a single inversion.

The honest statement, and it is the one that goes in the scorecard: *we are
ahead of LightGBM at 50,000 rows in every one of fourteen paired measurements,
by a median of 4.6 to 9.6 percent depending on the window, and the margin is
smaller than the measurement noise on this machine so its size is not
established.* Doubling the repeats halved the apparent margin, which is what
should be expected when the first estimate came from a lucky window.

### Binning. THE RATIO IS AGAINST A COMPARATOR WE REQUIRED TO DO MORE WORK

| shape | ours | LightGBM, pinned to bin every row | ratio |
|---|---|---|---|
| 1,000,000 | 0.199 | 1.006 | 5.1x faster |
| 250,000 | 0.048 | 0.239 | 5.0x faster |
| 50,000 | 0.048 | 0.046 | 4.6% behind |

**Read the column heading. This is not LightGBM's binning cost; it is
LightGBM's binning cost under a pin we imposed.**

`bench/bench_lightgbm.py` sets `params["bin_construct_sample_cnt"] =
int(n_rows)`. LightGBM's default is a **fixed 200,000-row subsample** whatever
the dataset size, so its stock binning cost stops growing above 200,000 rows
while ours grows with the data. The pin exists for a good reason, stated in
`bench/real_data/scenarios.py`: unpinned, the two engines fit edges from
different data and that is the largest single source of split divergence above
200,000 rows. Its own rationale also says, in a clause nobody had carried
forward, that leaving it alone "makes LightGBM's binning time look better than
a like-for-like measurement would" — which is true in the direction that
flatters LightGBM on *edges* and, as it turns out, flatters **us** on *time*.

**Derived bound, and it is enough to withdraw the headline.** At 1,000,000 rows
the pin makes LightGBM bin five times more data than it would choose to. If its
binning is roughly linear in rows, stock LightGBM bins 200,000 rows in about
0.20 s, against our 0.199 s over the full million. **That is parity, not 5.1x**,
and the 5.1x is substantially a measurement of the extra work we required.

So: **the 5.1x and 6.3x figures are withdrawn as statements about LightGBM.**
What survives is narrower and still worth having: *given the same edges fitted
from the same rows, we fit them about five times faster.* That is a real
property of our binner and it is the right comparison for parity of the model.
It is not a claim about what a user waits for at LightGBM's defaults.

This must be re-measured against stock LightGBM before any version of it is
quoted again, and the project is about to move its own defaults to stock, which
changes which world the number lives in.

The credit for catching this belongs to the GPU campaign, which read the pin's
rationale rather than the pin. It is the fourth instance this week of a real
number being quoted for a question it could not answer, and the third of them
was mine.

Note our binning is 0.048 s at both 250,000 and 50,000 rows, which looks like a
fixed floor rather than row-proportional work below about a quarter million
rows. Not investigated, and it interacts with the above: a fixed floor is
exactly what a sampled edge fit produces.

## What a quiet box does not buy

**Both arms drifted monotonically upward through the clean window.** Ours rose
15.3 percent from first repeat to last, LightGBM's 12.6 percent, both without
an inversion, with nothing compiling and the other campaign fully held.

That is not contention. It is the CPU drift the GPU campaign's regime canary
measured directly on two fixed probes — `drift_cpu_pct: 22.5` against
`drift_gpu_pct: 0.7` inside a single short run, with no training code in either
probe.

**Conclusion, and it is a constraint on this whole campaign: on this machine a
CPU absolute time is a property of the window, not of the code. Only ratios
taken from arms interleaved inside one process are results.** Every seconds
figure in this document is context. No lane may be scored against one.

## The in-run scaling profile: the round's central question, answered

`MOJOTREES_PHASE_PROFILE=async` on the same 1,000,000 x 50 fit twice, at
`MOJOTREES_NUM_WORKERS=1` and at auto, dividing per-(phase, node size class)
nanoseconds. 100 trees, 6,100 nodes. Size classes: `root=all`, `large>1/8`,
`medium>1/64`, `small>1/512`, `tiny<=1/512` of the dataset.

Determinism across worker counts is a standing contract, so both runs grow
identically shaped trees and the classes line up node for node.

### Where the serial round actually goes

| phase | serial | share |
|---|---|---|
| **histogram** | **10.566 s** | **83.9%** |
| partition | 1.763 s | 14.0% |
| split_search | 0.226 s | 1.8% |
| subtract | 0.030 s | 0.24% |
| hist_alloc | 0.0006 s | 0.005% |
| total | 12.586 s | |

The total independently reproduces the measured 12.357 s single-worker fit, so
the instrument accounts for essentially the whole round.

### The scaling, by phase and by node size. This is the finding.

| phase | root | large | medium | small | tiny | **overall** |
|---|---|---|---|---|---|---|
| **histogram** | 2.69x | 2.47x | **1.92x** | **1.20x** | 1.39x | **2.13x** |
| partition | 1.67x | 2.29x | 1.55x | **0.78x** | **0.80x** | 1.99x |
| split_search | 1.04x | 1.22x | 1.21x | 1.23x | 1.36x | **1.22x** |
| subtract | — | **0.76x** | **0.82x** | **0.80x** | **0.76x** | **0.80x** |
| whole round | | | | | | **2.07x** |

### RETRACTION: Phase 0's central reading was wrong

Phase 0 concluded, from `bench_profile.mojo`'s isolated stage numbers, that
"every individual kernel scales better than a whole tree does — histograms
3.25x to 3.40x, partition 3.74x, against `grow_tree` at 2.47x" and therefore
that **the parallel loss was not in histogram accumulation**. On that basis it
demoted L3, the row-block private-histogram lane.

**That is retracted.** Measured inside real trees, histogram accumulation
scales at **2.13x**, not 3.3x, and it is the largest single parallel loss in
the round. The isolated-kernel figures were synthetic calls at 1,000,000,
500,000 and 100,000 rows with strided row lists and a fresh allocation per
call; they did not describe what the grower does.

This is the third instrument correction of this round and the only one that had
already changed a decision. The lesson is narrow and worth stating: **an
isolated kernel benchmark is not evidence about that kernel's behavior inside
the program.** The in-run instrument existed the whole time and cost two runs.

### What the profile says to do, in order

1. **The histogram at medium and small nodes is the prize.** Together they are
   2.36 s of the 6.08 s auto round and they scale at 1.92x and 1.20x against
   the root's 2.69x. **Derived bound:** bringing both to the root's 2.69x saves
   about **0.95 s of a 6.08 s round, 15.6 percent**. That does not reach parity
   on its own — 5.13 s against LightGBM's 2.92 — which is consistent with C2's
   registered arithmetic that neither half suffices alone.

2. **Three phases measure SLOWER in parallel than serial** — `subtract` at
   every size (0.76x to 0.82x) and `partition` at small and tiny (0.78x,
   0.80x).

   **RETRACTED, same day, before any lane acted on it.** The first version of
   this bullet read that as "dispatch overhead exceeding the work — the grain
   floor is letting through fan-outs that cannot pay for themselves", and
   proposed raising those crossovers.

   That is refuted by exact arithmetic on the planner, done by the grain-floor
   lane. `subtract_ops(50 * 255) = 38,250`, or 25,500 under constant hessian.
   The partition passes `3n`, and its `small` class tops out at 15,625 rows so
   `3n = 46,875`. **Both are below the 65,536 crossover, so `plan_tasks`
   returns 1 and both run serial — in the auto arm and in the one-worker arm
   alike.** Since the wiring glue they also go through `plan_tasks_with` on a
   resolved snapshot, so the two arms execute an identical instruction stream
   with no `getenv` and no core detection in either.

   **There is no fan-out under either phase to remove, so there is nothing for
   a grain-floor change to fix.** A ratio of 0.76x between two arms running
   identical code is not a property of the code. It is whole-machine state —
   clock and cache residency — misattributed to a phase, because the ratio was
   taken across two *whole fits* rather than with the arms interleaved. Given
   the regime canary measures 22.5 percent CPU drift inside a single short run,
   an 11 ms artifact across two fits is entirely unremarkable.

   **The instructive part is what this says about the method, not the phase.**
   A per-phase serial-versus-auto ratio taken from two separate whole-fit runs
   inherits the drift of the whole machine between them. It is trustworthy for
   the large effects in the table above, where the signal is 2x, and it is
   worthless below about 1.2x. Every ratio in the histogram row survives that;
   the subtract row does not, and neither does `split_search` at root (1.04x).

   The genuine planner defect runs in the **opposite** direction and is
   described below.

3. **`split_search` scales at 1.22x and nothing else, and this one is a real
   defect with a real cause.** The profile-fidelity lane derived it before it
   was measured: `split_scan_ops(50, 255, two_sided) = 216,750`, and
   `216,750 // 65,536 = 3`, so the scan fanned out **three ways regardless of
   node size**, leaving 7 of 10 cores idle on every one of 6,100 nodes.

   The grain-floor lane found the general form. `DEFAULT_MIN_TASK_OPS` was
   **aliased to `PARALLEL_MIN_OPS`**, so `by_grain = total_ops // 65,536`
   collapsed the task count for every caller whose total sits between 1 and 40
   grains — the entire small and medium tail, plus the split scan at every
   size. `MIN_TASKS_ABOVE_GRAIN = 2` was already an admission that the grain
   permits no legal split between one grain and two; it just capped the
   admission at two.

   **So the planner is wrong in one direction, not two: too few tasks, never
   too many.** The crossover that decides serial-versus-parallel is not
   implicated by anything in this profile, because the three phases it would
   have had to mis-admit were never admitted at all.

   Fixed on this branch: the task count now floors at `dispatch_cores` once the
   crossover is cleared, with `max_auto_tasks` still binding last. Split scan
   goes 3 to 10 tasks, gradient fill 3 to 10 at 1M rows, partition at medium
   nodes 2 to 10. **The root and medium histogram are unchanged, and not by
   luck** — both already had more whole grains than the 25 accumulation groups
   the dispatch has to hand out, so the floor has nothing to raise. That is
   asserted as literal integers in the test, so a change to the estimator
   breaks the test rather than silently invalidating the argument.

   **Derived bound, and it is a ceiling rather than a prediction:** the split
   scan cannot improve by more than 10/3 = 3.33x, it is 3.0 percent of the auto
   round, and it is followed by a serial ascending fold over all 50 features
   that does not parallelize at all. So the whole prize here is well under
   0.14 s. Small. It is in the round because it was free once the rule was
   understood, not because it was worth a lane on its own.

4. **`hist_alloc` is 0.005 percent of the serial round.** The booster-scoped
   `_HistPool` from round 2 works, and **histogram allocation is not the
   per-node overhead**. This partially undercuts L1's stated motivation: its
   remaining case is the partition's own row lists (inside `partition`'s 14
   percent) and halving row-id traffic with Int32, not histogram allocation.

## The golden fixture does not cover either bit-moving change in this round

Three lanes independently reported that `tests/test_golden_bits.mojo` passes
**untouched** through their work, and each was right for a different reason.
Verified directly rather than taken on report:

- **Five of the six fixtures never call `fit_bins` at all.** They use
  `bin_equal_width(x, n_rows, n_features, 32)`. The binning defaults change —
  `bin_construct_sample_cnt` to 200,000 and `min_data_in_bin` to 3 — cannot
  reach them by construction.
- **The sixth, `misscat`, does call `fit_bins`**, at `max_bins=32` with
  categorical features and missing values. It is 200 rows. The sample cap only
  engages above 200,000 rows, and `min_data_in_bin=3` merges only levels
  holding one or two rows, which continuous uniforms into 32 bins do not
  produce.
- **Row-block private histograms need 8,160 rows to engage** at 255 bins. The
  fixtures are 150 to 200 rows. Not one of them blocks.

So this round made **two deliberate bit-moving changes and the contract
fixture saw neither of them.**

**This is not a complaint about the lanes; it is a finding about the
fixture.** The golden test is the artifact this project treats as its
bit-identity contract, and the honest statement of what it currently protects
is: eight-tree boosters over 150 to 200 rows of uniform noise at 32 bins,
five of them on equal-width bins that no production path uses.

It is a real regression net for the tree grower, the objectives, the split
scan and the score update, all of which it exercises. It is **not** a net for
the binner, for anything that engages above a few hundred rows, or for any
decomposition whose threshold sits above its fixture sizes. A change can move
bits on every real dataset a user has and leave it green.

Two consequences, neither of them acted on in this round:

1. **The wave's single golden regeneration is expected to be a no-op**, and if
   it is, that fact should be recorded rather than read as "no bits moved".
   Bits moved. The fixture cannot see them.
2. **The fixture needs at least one fixture above the thresholds the code now
   has** — above 8,160 rows for the block path and above 200,000 for the
   sample cap — or those paths ship with no bit contract at all. That is a
   lane, and it belongs to whoever owns the fixture next.

This is the same species as the three built-tested-never-wired findings and
the two gate-blind tests: a check that passes for a reason unrelated to the
thing it is believed to check.

## Consequences for the round's plan

- **L3 (row-block private histograms) is rehabilitated and is the strongest
  remaining lane**, having been wrongly demoted on isolated-kernel numbers. It
  attacks the histogram at exactly the node sizes where the loss is. It remains
  the only lane that moves bits, so it still lands alone with a deliberate
  golden re-baseline.
- **L2 (interleaved cells) keeps its case**, which was always the 1.36x serial
  term rather than scaling, and the serial term is 83.9 percent histogram.
- **L1's case is weaker than written** and its brief must be corrected before
  it starts: allocation is measured as negligible, so it stands on row-id
  traffic and on the partition's own lists.
- **The grain-floor lane ran and inverted its own premise**, which is the right
  outcome for a lane launched off a misread. It was sent to stop `subtract` and
  small-node `partition` fanning out into a loss; it proved neither fans out at
  all, declined to change the crossover, and fixed the opposite defect instead.
  Its report is the reason the two retractions above exist.

- **The score-update lane's prize is unmeasured and unmeasurable from here.**
  It made `_add_by_leaf` cut inside leaves rather than between them, which
  removes an imbalance whose size depends on the largest leaf's row share `p`.
  At 31 leaves over 30 tasks the old floor was 1.94x even for perfectly even
  leaves. Nothing in this profile sizes the score update as a share of the
  round, so nothing here says what that is worth. `PROF_SCORE_UPDATE` is where
  it would be settled.
