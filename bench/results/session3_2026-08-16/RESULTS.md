# Session III results, 2026-08-16

Measured after all six lanes merged, the full suites passed (82 test files), the
parity contract passed, and the API gate went green. Protocol and decision rules
in `../PROFILE_PROTOCOL.md`, rules M0-M6, all registered before any of this ran.

Machine: Apple M4 MacBook. Every figure below is the median of five in-process
repeats. Every A/B is either interleaved in one process or alternated across
processes, never compared across sessions.

**The headline is that the round's main claim was refuted and a claim nobody was
testing was confirmed.**

---

## M2.1 The upload collapse. REFUTED.

1,000,000 x 50, resident plane on, alternating processes, five pairs.

| pair | collapse ON | collapse OFF |
|---|---|---|
| 1 | 2.517 | 2.500 |
| 2 | 2.484 | 2.504 |
| 3 | 2.500 | 2.520 |
| 4 | 2.478 | 2.511 |
| 5 | 2.489 | 2.505 |
| **median** | **2.489** | **2.505** |

**0.016 seconds. The registered prediction was 0.64.** Not resolved by M0: the
0.016 gap sits well inside the ON arm's own 0.039 spread. The direction favors
ON in four pairs of five, so this is not a regression; it is a null.

M3 registered, before the data, that a result under 0.3 seconds means the
per-synchronization constant is wrong and must be reported as loudly as a win.
It is. Thirteen copies per tree removed, roughly 1,300 per fit, for sixteen
milliseconds.

## M2.2 The resident plane. RESOLVED, and larger than before.

Same shape, alternating processes.

| pair | resident ON | resident OFF | delta |
|---|---|---|---|
| 1 | 2.482 | 3.244 | 0.762 |
| 2 | 2.482 | 3.228 | 0.746 |
| 3 | 2.477 | 3.230 | 0.753 |
| 4 | 2.494 | 3.234 | 0.740 |
| 5 | 4.089 | 4.416 | 0.326 |
| 6 | 4.057 | 4.426 | 0.368 |

Pairs 1-4 are **resolved** by M0 with room to spare: a 0.75 second effect against
arm spreads of 0.017 and 0.016, with the four deltas landing within 0.022 of one
another. Every pair of the six favors the plane.

**Pairs 5 and 6 are a different machine regime, and the honest reading is that
the effect size itself moved, not just the level.** Both arms rose about 65
percent and the delta fell to 0.35. I first attributed the shift to
`mobileassetd` at 100 percent CPU, which was real and was an M1 violation; but
after waiting it out and confirming the box idle, the numbers stayed high. So
the attribution was wrong and is retracted here. This is the two-to-threefold
drift this repository has documented, most plausibly the GPU clock state that
`PROFILE_PROTOCOL.md` already names as the likeliest cause.

What that means for how the number may be quoted: **the resident plane is worth
24 percent in the fast regime and 8 percent in the slow one.** A single figure
for it is wrong. Only the direction is regime-independent.

## Together, M2.1 and M2.2 refute the cost model and replace it

The rule written into two docstrings this round, "every copy counts, in both
directions", is correct about mechanism and wrong as a **cost** model. A copy may
well drain the queue. Draining a queue that holds nothing costs nothing.

What costs is the **round trip**: host code waiting on a device answer it needs
before it can decide what to enqueue next. The resident plane removes about
thirty of those per tree and is worth 0.75 seconds. The upload collapse removed
thirteen copies per tree that were never waiting on unfinished work, and is worth
0.016.

The earlier version of `grow_tree_device_resident`'s docstring had this right and
was overwritten during this round:

> the number of *round trips*, in the sense of host code that reads a device
> answer and then decides what to enqueue next

The 458 microsecond constant, derived from the depthwise A/B where what was
removed were per-level round trips, applies to round trips. It was extended to
copies by analogy, by me, and the extension is refuted.

### And the "three independent confirmations" was two errors agreeing

This project stated, repeatedly and including to Andrew, that the per-split wait
had been confirmed by three routes sharing no inputs: a curve fit giving a 1.42
second intercept, a counter times a per-event cost giving 3,100 x 458 us = 1.42
seconds, and the resident plane predicting 0.69 seconds and measuring 0.57.

The first two are sound and remain so. **The third was a coincidence.** The 0.57
seconds it measured came from removing about thirty *round trips* per tree. The
model that predicted 0.69 was counting fifteen *copies and syncs*. Both numbers
were near 0.6 and the agreement was read as confirmation of the model. It was
two different quantities landing close together, and it made a wrong cost model
look corroborated at exactly the moment it should have been questioned.

That is why the collapse prediction was then wrong by a factor of forty rather
than being caught early: the erroneous model had apparently just passed a test.
A prediction that agrees with a measurement for the wrong reason is worse than
one that fails, because it buys the model credit it has not earned.

**None of the collapse work is reverted.** It is correct, it is tested, it makes
staleness structurally impossible in the searcher's tables, and it removes real
if unmeasurable work. It is simply not a speed result, and the parts of it that
were argued as speed have been relabelled.

## M2.3 The histogram row unroll. INDISTINGUISHABLE.

1,000,000 x 50, interleaved in one process, five repeats each.

| arm | median | spread of median |
|---|---|---|
| unroll on (shipped default) | 4.084 | 11.2% |
| unroll off | 4.322 | 13.9% |

Delta 8.1 percent against a noise floor of 14.1 percent: **indistinguishable**.
The direction favors the shipped default, so the default stands and nothing
changes.

Worth recording separately: **both arms produced a training loss identical to the
last digit**, 0.0034559282961870628, at a million rows. The associativity
argument that says this knob cannot change a model is no longer only an argument.

Taken in the slow regime, so the spreads are wide. This is the item most worth
re-taking in a fast window.

### Re-taken in a fast window. RESOLVED, and the verdict flips.

Same command, same shape, quiet box:

| arm | median | spread of median |
|---|---|---|
| unroll on (shipped default) | **2.597** | 2.1% |
| unroll off | 2.881 | 1.7% |

**Resolved: 10.8 percent, against a 2.1 percent noise floor.** The unroll is
worth 0.284 seconds at 1,000,000 x 50.

**Same code, same command, opposite verdict, because of machine state alone.**
In the slow window this was 8.1 percent against a 14.1 percent floor and was
correctly called indistinguishable. That is the clearest demonstration this
project has of what a dirty or throttled box does to a real effect, and it is
worth more than the unroll result itself: a null taken in a bad window is not
evidence of absence.

Training loss identical to the last digit in both arms in both windows.

### The isolated benchmark agrees, and prices the kernel

`bench-hist 1000000 50 20`, twenty repeats per arm:

| arm | median per node | |
|---|---|---|
| unroll on | 1.488 ms | |
| unroll off | 3.178 ms | |

**2.14x on the histogram kernel alone**, resolved, band 3.0 percent.

The plan registered that a disagreement between the isolated and end-to-end
benchmarks would be the finding. **There is no disagreement.** Both resolve, both
favor the unroll. The isolated arm shows the kernel more than halving; the fit
moves 10.8 percent, which is the same win seen through everything else the round
does. That ratio is itself informative and is the first honest read this project
has on how much of a fit the histogram kernel is at this shape.

Also measured in the same run, and a null: feature group 2 against group 1 is
1.11x, **indistinguishable** at a band of 11.5 percent.

## M2.5 The parallel grain at the host-scan shape. NULL.

50,000 x 50 with the resident plane off so the host scan is genuinely taken,
three alternating pairs.

| pair | `PARALLEL_MIN_OPS=32768` | default 65,536 |
|---|---|---|
| 1 | 1.700 | 1.710 |
| 2 | 1.703 | 1.704 |
| 3 | 1.714 | 1.722 |

A 0.001 to 0.010 second difference on a 1.70 second fit. **Indistinguishable.**
The 38,250-op conversion sitting just under the default grain, on the knife edge
this project flagged, is not costing anything measurable. The grain stays where
it is and the docstring's refusal to move it without measurement is vindicated.

## M2.4 LightGBM, interleaved, with its own spread. THE COMPARATOR MOVED.

1,000,000 x 50, same process, same window, five repeats each, LightGBM at ten
threads to match our CPU arm's contract.

| arm | samples | median | spread of median |
|---|---|---|---|
| ours, GPU resident | 4.324 3.862 3.779 3.863 3.931 | **3.863** | 14.1% |
| LightGBM, 10 threads | 5.579 5.781 5.819 5.807 6.329 | **5.807** | 12.9% |

Verdict **resolved**: 47.6 percent against a 14.4 percent noise floor. In this
window we are **1.50x faster than LightGBM**.

An unthreaded run in the same window gave LightGBM 5.813 with a spread of only
3.7 percent, so the thread setting is not what places it near 5.8.

**LightGBM's repeat spread on this machine has now been measured for the first
time: 3.7 percent in a stable window, 12.9 percent in an unstable one.** Every
margin this project has ever claimed against LightGBM was claimed against an
unknown floor. It is 3.7 percent at best, which is smaller than the margins that
have been claimed, so the practice was luckier than it was sound.

### The 2.767 figure is not reproduced, and it framed the whole round

Every "we are 1.14x behind" and "we are 0.93x ahead" statement this week rests on
LightGBM at 2.767 seconds. Interleaved, in-process, at both thread settings, it
measures 5.81 here. That is 2.1x apart, and LightGBM is CPU-only so the GPU clock
regime cannot explain it.

Two candidates, and this session cannot separate them:

1. The machine is in a materially slower state tonight for CPU as well as GPU. Our
   own GPU arm moved 2.48 to 3.86 across the same hours, which is consistent.
2. The 2.767 figure was taken under a setup this harness does not reproduce. It
   predates M5, which recorded that the comparator changed when LightGBM went
   inside the interleaved loop.

**Until this is settled, no ratio against 2.767 may be quoted, including the ones
in this repository's own earlier write-ups.** The ratio measured inside one
window is 1.50x in our favor. The ratio against the remembered number was 1.14x
against us. They cannot both be reported and only one of them was measured the
way M6 requires.

---

## What M6 permits to be called a win

M6: leaf-wise, at 1,000,000 x 50, resolved by M0, against a LightGBM arm measured
in the same process in the same window with its own spread reported.

**That is exactly what M2.4 is, and it says 1.50x in our favor.** It is the first
comparison this project has taken that satisfies the rule it wrote for itself.

The honest qualifier is that it was taken in a window where both our arms and the
comparator are slower than they were four hours earlier, and the rule says
nothing about which window. Re-taking it in a fast window is the first item of
the next session, and the prediction registered here, before that data, is that
the ratio moves **further** in our favor, because our arm improves with clock
state and LightGBM does not use the GPU at all.

---

## S1, closed. The resident plane becomes the default GPU plane.

Taken in a fast window (load 2.15, nothing else running), alternating processes,
five pairs each. These are the two shapes S1 has been short of since the plane
landed.

**250,000 x 50**

| pair | resident ON | resident OFF |
|---|---|---|
| 1 | 1.095 | 1.963 |
| 2 | 1.113 | 2.006 |
| 3 | 1.126 | 2.010 |
| 4 | 1.150 | 2.061 |
| 5 | 1.158 | 2.016 |
| **median** | **1.126** | **2.010** |

**0.88 seconds, 44 percent.** Resolved by M0 by a wide margin: the delta is more
than ten times the wider arm's spread, and every pair agrees.

**50,000 x 50**

| pair | resident ON | resident OFF |
|---|---|---|
| 1 | 0.792 | 1.733 |
| 2 | 0.795 | 1.721 |
| 3 | 0.793 | 1.713 |
| 4 | 0.779 | 1.726 |
| 5 | 0.788 | 1.727 |

Median 0.789 against 1.724: **2.2x faster**, resolved, with arm spreads under
0.02. S1 asked only for "no regression at 50,000". It is the largest relative
win of the three shapes.

**The gate was proved open rather than assumed.** `MOJOTREES_GPU_TREE_RESIDENT_
TRACE=1` reports `plane=device-resident status=budget_spent commits=30 leaves=31
root_rows=50000` at 50k and `root_rows=250000` at 250k, so these compare the
plane against the shipping loop and not the fallback against itself. That check
exists because a test written earlier in this project ran six fixtures entirely
below the gate and verified nothing.

### S1's three conditions

| condition | verdict |
|---|---|
| trees node-identical to the host plane | satisfied, `tests/test_gpu_tree_resident.mojo`, no tolerance |
| faster at 250,000 **and** 1,000,000 | 44% and 24%, both resolved |
| no regression at 50,000 | 2.2x faster |

**All three hold. Under S1 the resident plane becomes the default GPU plane.**

### And it retires the crossover gate's premise

`M4_MIN_NORMALIZED_WORK = 50,000,000` exists to route small shapes to the host
scan, because the device search used to lose below it: this repository's own
record says the device path "beats host scan at 250k rows, loses at 50k". At
50,000 the normalized work is 2,500,000, a twentieth of the gate.

The resident plane wins there by 2.2x.

> **CORRECTION, same night. That last sentence did not follow from this data and
> is withdrawn.**
>
> The `gpu-device` bench arm passes `split_search=SPLIT_SEARCH_DEVICE`, which
> **forces** the device split search and bypasses `M4_MIN_NORMALIZED_WORK`
> entirely. So both arms of every S1 pair above had the device search forced:
> the comparison is the resident plane against the **shipping device loop**, and
> it is sound as such. It is not a comparison against the host scan.
>
> But the gate chooses host scan versus device search. **This session never
> measured the host scan at 50,000 or 250,000 at all**, so it cannot say the
> gate's premise is falsified. It can only say that when the device search is
> taken, the resident plane is much better at running it.
>
> The missing measurement, and it is now the highest-value one outstanding:
> **resident plane on the forced device arm against the automatic path (host
> scan) at 50,000 and 250,000.** That is what decides whether
> `M4_MIN_NORMALIZED_WORK` can be deleted. Until it exists, the gate stays and
> no claim about its premise may be made.
>
> This is the "measured the wrong arm" error, which this project has now made in
> three distinct forms in twenty-four hours: a vacuous test that ran below a
> gate, a comparator throttled harder than us, and now an inference about a gate
> drawn from data that bypassed it.

### What S1 does and does not deliver to a default user

The plane is only reachable when the device split search is chosen. On
`SPLIT_SEARCH_AUTO` that requires `normalized_split_work >= 50,000,000`, which
250,000 x 50 (12.5M) does not meet and 1,000,000 x 50 (exactly 50M) just does.

**So flipping the default delivers the 2.2x and 44 percent to almost nobody
yet.** They are real wins on the forced arm, and the flip is a **prerequisite**
for retiring the gate rather than the delivery of anything to default users.
Retiring the gate is what cashes this in, and retiring it needs the measurement
named above first.

---

## The comparator investigation. The 1.50x claim is WITHDRAWN.

Before quoting any margin, three hypotheses had to be separated: the machine was
slow, the pre-M5 figure came from a setup this harness does not reproduce, or
**LightGBM is hobbled inside a process that has already initialized Metal and
Mojo** (thread-pool sizing, affinity, memory pressure, `num_threads` resolution
after import). The third would make a 1.50x claim an error in our own favor, so
it was excluded first.

Bracketed in one fast window, standalone / in-process / standalone:

| run | LightGBM, 10 threads | spread |
|---|---|---|
| A, standalone, own process | 2.662 | 2.0% |
| B, in-process, interleaved | 2.802 | 9.6% |
| C, standalone, own process | 2.822 | 2.9% |

**In-process sits between the two standalone runs. The comparator is not
hobbled**, and the third hypothesis is refuted. Standalone also reproduces the
historical 2.767 figure almost exactly, so the first hypothesis is confirmed: the
5.807 reading earlier in this session was the machine's slow regime and nothing
else.

### What the slow regime was actually doing

Across the regime change, LightGBM went 2.66 to 5.81, a factor of **2.2**. Our
GPU arm went 2.59 to 3.86, a factor of **1.5**. Ten CPU cores throttle harder
than the GPU does.

So the 1.50x figure was a measurement of the comparator being thermally
throttled harder than we were. Reporting it would have been claiming a win from
the machine's power state, which is the mislabelled-covertype error with the
sign reversed, and the first outside reader would have found it.

### The margin, measured properly

Five interleaved in-process runs, fast window, ten threads each side:

| run | ours, resident | LightGBM | ours ahead by |
|---|---|---|---|
| 1 | 2.592 | 2.802 | 8.1% |
| 2 | 2.564 | 2.801 | 5.6% |
| 3 | 2.551 | 2.883 | 10.4% |
| 4 | 2.584 | 2.953 | 10.0% |
| 5 | 2.620 | 3.499 | 11.3% |
| **median** | **2.584** | **2.883** | **11.6%** |

**Verdict: consistent, not resolved.** Every one of five runs favors us and our
own arm is remarkably tight at 2.7 percent spread, but LightGBM's spread over
these runs is 24 percent and drifting upward run over run, which under M0 is
wider than the effect. The direction is unanimous; the magnitude is not
established.

Against LightGBM's own best standalone sample in this window, 2.640, our median
of 2.584 is still ahead, by 2 percent.

### The one asymmetry worth keeping

Our arm did not degrade across five back-to-back runs; LightGBM went 2.80 to
3.50 over the same sequence as load accumulated. The GPU arm is thermally stable
under repetition and the ten-core CPU arm is not.

**Read that as a property of the engine, not of the comparator**, and the
distinction is not pedantic. The CPU campaign measured the mirror image the same
night: at 1,000,000 x 50 *their* CPU arm rose 18 percent across five repeats
while LightGBM rose only 10 percent. So "LightGBM degrades harder under
repetition" is false as a general claim. It held here because the GPU is
thermally stable, not because LightGBM is fragile; against another ten-core CPU
arm, LightGBM is the *steadier* side.

Anyone quoting this asymmetry must say which engines were on which side. It is
why a head-to-head number depends on how long the machine has been working, and
it is why the canary reports the CPU and GPU probes separately and never
averages them.

### A caveat on every LightGBM number in this repository, found the same night

`scenarios.LIGHTGBM_ALIGNMENT` pins **`force_row_wise = True`**. Found by the CPU
campaign orchestrator reading the params line its own smoke test printed.

So every LightGBM figure this project has ever recorded — the 2.767 that framed a
week of work, the 2.662 standalone above, and the 2.883 in-process median used as
the comparator here — was taken against a builder **we forced**, not against
whatever LightGBM would have chosen.

The pin is methodologically right and should stay: on auto, LightGBM spends its
first iterations timing both strategies and that one-off lands inside the
measured region. But the consequence has to travel with the margin. "Ahead by 5
to 11 percent" is precisely "ahead of LightGBM with `force_row_wise` pinned", and
if the column-wise builder is materially faster at this shape then the margin is
against a handicapped configuration.

That is the same class of error as the throttled comparator withdrawn above,
reached from a different direction, and it would be found by the first outside
reader who ran LightGBM themselves.

### Resolved the same night: the pin is CONSERVATIVE, and the margins stand

The CPU campaign measured it at 1,000,000 x 50, ten threads, both builders
pinned, interleaved in one process, five repeats:

| LightGBM builder | median | spread |
|---|---|---|
| `force_row_wise`, what we pin | **2.856** | 14.0% |
| `force_col_wise` | 3.052 | 9.3% |

All five pairs favor row-wise, median margin **6.9 percent**; consistent, not
resolved, since the delta is smaller than the wider arm's own range.

**The builder we pin is LightGBM's faster one at this shape, so pinning it makes
the comparison harder on us, not easier.** The feared error — a margin won
against a handicapped comparator — did not occur, and it failed to occur in the
direction that supports rather than undermines what is published above. Nothing
in this file is retracted on this account.

**The caveat stays anyway**, in one sentence, because 6.9 percent is consistent
rather than resolved and an outside reader on defaults could land anywhere in
that band and would additionally pay the strategy-timing cost the pin removes.
The correct phrasing wherever a margin appears: *the comparator is pinned to
`force_row_wise`, which is LightGBM's faster builder at this shape by a
consistent 6.9 percent, so the pin is conservative with respect to our margin.*

### A statistic discrepancy in the harness, which may affect close verdicts

The CPU campaign found that `bench_train_gpu.mojo` computes its
`resolved` / `indistinguishable` verdict from the arms' **minima**, while this
protocol requires the **median** and says why: the minimum is the luckiest
sample, and contention here is the finding rather than noise.

At 50,000 rows it flips this project's one claimed win over LightGBM: resolved on
minima, consistent-not-resolved on medians. Every pair still favors us, so the
direction is not in doubt, but the verdict word is.

**Audited against this file: no verdict here changes.** The unroll is 10.8
percent against a 2.1 percent floor on either statistic; the collapse null and
the S1 figures were computed from medians by hand; the withdrawn slow-window
LightGBM comparison is withdrawn on other grounds. The harness line should still
be corrected to use the median, and that is queued behind the canary lane, which
currently owns that file.

### The three sentences that are now supportable

1. At 1,000,000 x 50 in a fast window, resident leaf-wise is **2.58 seconds**.
2. LightGBM at ten threads on the same data in the same window is **2.66 to 2.95
   seconds** depending on how it is run and how warm the machine is.
3. We are **consistently ahead by 5 to 11 percent, and that margin is not
   resolved** under this project's own rule, because the comparator's spread is
   wider than the margin -- and it is a margin against LightGBM with
   `force_row_wise` pinned, per the caveat above.

Not "1.14x behind". Not "1.50x ahead". Both of those were this project's
headline within the last twenty-four hours and neither survives.

## Still open
- The parallel-grain check at the host-scan shape (M2.5), untaken.
- M2.3 re-taken in a fast window.
- Whether tonight's machine state is thermal, clock, or something else. Nothing
  in this session establishes it and the results are labelled by regime rather
  than corrected for it.


---

## The canary's first reading, and it is the most consequential number of the session

Taken 2026-08-16 on the first run of `bench/canary.mojo` after it was wired in.
Three repeats, one short run, the CPU campaign active on the same box. Two fixed
probes: a serial single-threaded integer chain and a saturating GPU kernel.
**Neither probe touches a dataset, the training code, or any knob either campaign
varies**, so nothing here is a property of anything either of us wrote.

    drift_cpu_pct: 22.5      regime: shifted
    drift_gpu_pct:  0.7

The CPU probe measured 467 ms at the start of the run and 572 ms at the end. The
GPU probe held to 0.7 percent across the same interval.

**Measured.** The baselines are not yet recorded so the ratios are null; only the
start-to-end drift is meaningful, and it is enough.

### What it settles

Three separate findings this week were arguments from arm behavior, each open to
the objection that the arm's own code explained it:

- ours: the GPU arm held 2.7 percent spread across five runs while LightGBM
  drifted 2.80 to 3.50
- the CPU campaign's: their CPU arm rose 18 percent across five repeats while
  LightGBM rose 10
- the resident plane measuring 24 percent in one window and 8 percent in another

All three are now explained by one directly measured fact: **on this machine the
CPU throttles hard and fast and the GPU essentially does not.** It is not a
property of LightGBM, not of our arms, and not of either campaign's code.

### What it costs us

**A CPU-versus-CPU comparison on this box drifts materially inside a single
run.** Not between sessions, not between windows -- inside one three-repeat run
lasting well under a minute.

So absolute seconds from any CPU-heavy arm are not a property of the code, and
may be reported only as ratios from interleaved arms with the canary line beside
them. The CPU campaign has accepted this for its own headline and is rewriting
its results file to lead with the ratio. That is the correct response and it was
reached on evidence neither campaign could have produced from its own arms.

### And it is the argument for the instrument itself

`PROFILE_PROTOCOL.md` had asked for thermal state since the first session. The
instruction pointed at a script that measures nothing, and for two days at a
handoff file that had been deleted. Every regime label in this repository until
now was inferred from effect, by hand, and one such inference was made and
retracted the same night.

The canary answers a different question than the protocol asked -- not what state
the machine reports, but what the machine delivers -- and it answered it on the
first run, without privileges, in a way that changed what another campaign is
willing to publish.

---

## The speculation census: K=1's hit rate, measured before the speculation was built

Lane K2 was asked to build speculative child histograms. It did not, and the
reason is a better outcome than the build would have been.

### K>=2 is provably useless, so K=1 is the only K

Not a census result, a theorem, and it is worth stating because it closes a
question rather than estimating it. At step *k* the two children that step *k*
creates have no records until step *k*'s own search writes them, which is the
last device work of the step -- so their candidacy is established exactly when
speculating on them would stop being speculation. The candidate set is therefore
the pre-existing leaves, and over that set the ranking **cannot move**: a commit
writes only the split leaf's frontier row, the appended row at `n_live`, and the
two records those children own. Every other leaf keeps its slot, record, depth
and row count bit for bit, hence its gain and its admissibility, and the pick's
`block.max` then `block.min` resolves an unchanged set identically, ties
included.

So the best pre-existing leaf at step *k+1* **is** the top runner-up at step *k*.
A second speculative candidate is a leaf that provably cannot be picked next.

### And the census that justified K=1 was measuring the theorem, not the hit rate

"The greedy pick was the top runner-up in 100 percent of 4,030 decisions" is
exactly what the theorem predicts **for the decisions where the pick is a
pre-existing leaf**. It says nothing about how often it is one. The figure is a
tautology over a conditioned subset, and this project has been quoting it as
evidence for a hit rate it cannot speak to.

### The actual hit rate, measured

`MOJOTREES_GPU_SPECULATION_CENSUS=1`, one fit of 100 trees per shape, summed over
the fit rather than averaged per tree (a three-leaf tree must not weigh the same
as a thirty-leaf one):

| shape | trees | builds | consumed | wasted | hit rate |
|---|---|---|---|---|---|
| 1,000,000 x 50 | 100 | 2900 | 1936 | 964 | **66.8%** |
| 250,000 x 50 | 100 | 2900 | 1902 | 998 | **65.6%** |
| 50,000 x 50 | 100 | 2900 | 1917 | 983 | **66.1%** |

**Measured**, and note what kind of measurement it is: the census is a pure host
function over the commit log that already comes home in the plane's one download.
It launches nothing, transfers nothing, and derives its answer from the tree
rather than from a clock, so **it is valid on a contaminated box and identical in
a fast window and a slow one**. It was taken while a second campaign had four
lanes compiling, and that does not weaken it at all.

Two structural facts bound it below 1 before any run, both visible in the
numbers: step 1 can never consume, because step 0's candidate set is empty, and
the last step's build has no commit after it. So `consumed <= builds - 1` always.

### What it decides

The lane registered its own bar before seeing the number: a hit rate materially
below about 50 percent makes the lane a loss before the launch-shape benefit is
counted. **66.8 percent clears it**, and the stability across a twentyfold range
of row counts is itself evidence that the rate is a property of leaf-wise growth
rather than of a shape.

The honest cost, which is not zero: **964 wasted builds per fit at 1M**, each a
partition and a histogram over a child window, spent on a child that is
discarded. A consumed build is the same work moved earlier; a wasted one is work
that would not otherwise exist. So the trade is roughly a third more child
histogram builds in exchange for a better launch shape, and whether that wins is
a measurement nobody has taken.

### Why it was not built, and what it needs

The speculation cannot be expressed from `gpu_resident_round.mojo` alone. Every
descriptor-aware launch reads one fixed buffer, `GpuActiveRows.step_dev`, so a
speculative step needs a second descriptor and a kernel that publishes a
runner-up without committing. Two changes, in two files that lane did not own,
neither large:

- `gpu_active_rows.mojo` -- a descriptor-pointer parameter on
  `enqueue_partition_desc` and `enqueue_desc_histogram`, defaulting to `step_dev`
  so no call site moves, plus a second `STEP_WORDS` buffer. The kernels already
  take the descriptor as an argument and already carry an ignore-the-descriptor
  flag.
- `gpu_tree_tables.mojo` -- a `_pick_runner_up_kernel`: phase two of
  `_pick_and_commit_kernel` with the committing phase replaced by a descriptor
  write, excluding the slot this step's commit took.

**Sequenced behind K1**, which owns the histogram kernel bodies in
`gpu_active_rows.mojo`. Two lanes editing that file concurrently is the shape
this batch was designed to avoid.

### One design correction the lane found that would have been a silent bug

**A speculative build must not fold the sibling subtraction in.**
`enqueue_desc_child` derives the larger child in place from the parent's slot,
which on a miss would destroy the histogram of the leaf being speculated on. The
speculation must build the smaller child into a spare slot and leave the
subtraction to the consuming step, so `open_resident(num_leaves)` becomes
`num_leaves + 1` and `RESIDENT_SCRATCH_RECORDS` gains two.

### And a note on dead steps

They make this worse rather than neutral. A dead step is nearly free today
because every descriptor-aware kernel reads `STEP_LIVE` and returns. A
speculative build adds a partition, a histogram and a search pair to **every**
step including dead ones, where it cannot possibly be consumed. `dead=0` at every
shape above, so it does not bite here -- but a fit that stops early on
`min_gain_to_split` would pay it, which is why `dead` is reported beside `builds`
rather than folded into it.
