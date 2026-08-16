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

---

## Candidates 5 and 6: the lane found a bug in the specification it was implementing

### The retracted claim stays retracted, and a different change is the real win

This project once relayed, enthusiastically, that dropping `- parent_score` from
the gain was a free accuracy win. **That is still wrong.** Rounding is monotone,
so subtracting a common constant preserves order, and in the near-tie regime
Sterbenz applies (`P/2 <= left_score + right_score <= 2P`), making the
subtraction exactly representable. A null, and no rewrite of it shipped.

**The cross form is a different change and it is real.** The bits are lost one
step earlier, in forming `left_score + right_score`, whose rounding error is
`eps` of its own magnitude, roughly `eps * parent_score` -- an absolute floor
that does not shrink as the gain shrinks. The cross form never forms that sum.

**Derived bound:** shipped resolves to `eps*P`, cross to `eps*sqrt(P*gain)`, an
improvement factor of `sqrt(parent_score/gain)`. Modelled in standalone NumPy,
**not in mojotrees**, percent of near-tie pairs ranked correctly:

| parent/gain | shipped resolves | cross resolves | ratio | bound predicts |
|---|---|---|---|---|
| 1 | 1e-6 | 1e-6 | 1 | 1 |
| 30 | 1e-5 | 3e-6 | 3.3 | 5.5 |
| 300 | 1e-4 | 1e-5 | 10 | 17 |
| 2900 | 1e-3 | 1e-4 | 10 | 54 |

Where it lives: one-sided gradients, late logistic and softmax rounds, nearly
pure leaves. **Where it does not: below a parent/gain ratio of 1 it buys nothing
and is about 30 percent worse on the median.** Said at the site rather than
buried.

Incidental and worth keeping: a form that cannot resolve a gap does not
coin-flip. The shipped form *ties* the candidates and the scan keeps its
incumbent, which is why it scores 19 percent rather than 50.

### Section 8 of ACCURACY_BUDGET.md is invalid at `lambda_l1 > 0`

**The identity requires `GL + GR = G`. Under L1 the gain is built from `T(GL)`,
`T(GR)`, `T(G)`, and soft thresholding is not additive.** The document never
mentions L1 anywhere.

Applying it anyway measures a **systematic bias** of 1.6e-04 relative at a
parent/gain ratio of 293, where the shipped form sits at 1.0e-05 -- with median
and p99 agreeing to two figures, which rounding does not do. **Shipping section 8
as written would have degraded every L1 fit.** The arm now refuses itself
whenever `lambda_l1 != 0`.

Four further corrections the lane reported and the document should absorb: its
formula does not cover the categorical many-vs-many walk (children scored at
`lambda_l2 + cat_l2` against a parent at `lambda_l2`); "never worse" is too
strong; its three stated obligations dissolve because converting back to gain
units costs one node constant which cancels against `l2/(H+l2)*P` rather than
against `P`; and the effect is larger than its "up to 20x", reaching ~1000x on
the median over all candidates.

### Candidate 3 moved the ground under candidate 6 without removing it

The budget derived 6 from an inexact dequantization. **That reason is gone**, and
so is the next one: with a power-of-two scale the float subtraction is itself
exact by Sterbenz whenever the left child holds between half and twice the total.

What survives is stronger than what was written: the whole error is the two
Int32-to-Float32 casts, each rounding by `2^-24` **of the node total**, so the
derived right child inherits an error set by its parent's magnitude. Integer
subtraction casts once, afterwards, and that does not shrink with the scale's
shape. 6 still pays 2 to 2.5x on top of 5, and the "6 only with 5" rule holds --
power-of-two arguably sharpens it, since the anti-correlation is now exact rather
than approximate.

### Bits move, fixtures do not, and the two unrunnable files now ran

Bits move on **every** path through the device split search: every gain value,
the `min_child_hess` admission test, and the record's child stats. The golden
fixtures do not move -- `test_golden_bits` is cpu-safe, trains through the CPU
path, and never reaches this module.

The lane could not run `test_gpu_split_search` or `test_gpu_split_scan` under its
budget and, rather than guessing, asserted a bound: 600 histograms of exactly
their shape, both arms selecting the same split, differing by at most 1.96 units
of `eps32*(parent_score + gain)`, which is 6e-06 against their `atol=1e-4`.

**Both now run and both pass**, along with `test_gpu_gain_form` and
`test_gpu_tree_resident`. The bound held.

### The default, and what it rests on

`GAIN_FORM_CROSS` ships as the default, following candidate 3's precedent. Its
docstring states plainly that this rests on an identity, a derived bound and a
NumPy model, and on **no mojotrees measurement**.

That is the second numeric default this wave changed without a mojotrees
measurement, after `pair_alignment`. Unlike that one, **this changes bits**. It
is therefore gated on the real-data accuracy run against
`bench/real_data/thresholds.json`, which is also the GPU path's first accuracy
record of any kind. If that run does not clear the thresholds, the default flips
to opt-in and the arm stays for the L1-free case where the bound is largest.

---

## The GPU path's first accuracy record, and GAIN_FORM_CROSS clears

`bench/real_data/run.py --device gpu --tier smoke`, verified with `verify.py`:

**10 pass, 0 fail, 3 warn, 15 skip.** Baselines cleared on all three GPU-capable
scenarios: `dense_regression` RMSE 9.10211 against a 10.8525 baseline (improvement
0.1613, needs 0.15); `imbalanced_binary` AUC 0.950526 against a 0.65 floor;
`multiclass` multi_logloss 0.356234 against 1.20635 (improvement 0.7047, needs
0.1). Data pinning verified against `checksums.lock.json` on all three.

**This is the first accuracy record the GPU path has ever had.** Three of six
scenarios are covered: `categorical_missing`, `ranking` and `sparse_highdim`
declare no GPU support, and the LightGBM cells skip because LightGBM runs on the
CPU in this harness, so a cpu-vs-gpu row is labelled a mojotrees-internal
comparison rather than a differential.

### GAIN_FORM_CROSS is accuracy-neutral here, and the default stands

Cross against subtractive, device search forced, produced **bit-identical
training loss** (0.004108700157832688). That is the expected outcome, not a
plumbing failure: the lane's own test established that both arms **select the
same split** on ordinary data and differ by at most 1.96 units of
`eps32*(parent_score + gain)`. The two forms diverge only in the near-tie regime,
which ordinary data rarely enters.

So the change is an accuracy improvement **where near-ties occur** and a no-op
elsewhere, and it costs nothing measurable in either direction on real data. The
default stands, gated as registered.

### Two failed attempts at that A/B, and the second one is the fifth instance

Recorded because the failures are more instructive than the result.

**Attempt one had no environment variable set at all** -- I ran the same arm
twice under two labels and got identical numbers, which is precisely the
degenerate A/B this project has caught in three other places this week.

**Attempt two set the variable but ran below the split gate.** At 20,000 x 20 the
normalized split work is 400,000 against a 50,000,000 threshold, so the **host
scan** runs and the device gain form is never reached. The real-data smoke tier
is small for the same reason. So the A/B compared the host path against itself,
twice, and would have reported "no effect" about code that never executed.

**That is the fifth instance of the same failure in this repository**, after the
vacuous resident-plane test, the tile arms reaching one node in sixty-one, the
device-capability fixtures asserting a value no detection path could produce, and
the S1 inference drawn from an arm that bypassed the gate it was about. The
standing rule -- *a test for a gated path must prove the gate opened* -- applies
to **measurements** exactly as it applies to tests, and it is now written that
way rather than left to be inferred.

### One gap worth closing

The harness's `conditions` line reports `row_unroll`, `narrow_index`,
`pair_alignment`, `min_tiles` and `rows_per_tile`, but **not the gain form and
not the scale shape** -- the two arms in this wave that change numerics rather
than launch shape. A results file that names the launch shape and omits the
arithmetic is the wrong way round. Not fixed here; `bench/` has a live lane.

---

## The CPU/GPU prediction disagreement is not a tie-break bug, and probably not a bug

The lane briefed to fix it was told to establish that ties were the cause before
assuming it. **They are not**, and the refutation is thorough enough to close the
hypothesis.

### The host's rule, and one part of it nobody had written down

Read from `split.mojo` rather than from the brief: **highest gain; among equal
gains, the first in scan order**, where scan order is three levels -- bins
ascending within a feature, **missing-left before missing-right inside a bin**,
then feature slots ascending. The middle one is a real tie-break that the brief
did not mention and that nothing had documented.

**The tolerance the host applies to an equal gain is exactly zero** -- a bare
Float64 `>`. `SPLIT_TIE_RELATIVE` is not part of any decision on either backend;
it is only the width of a *report*.

### The device already matches, and this was tested rather than read

Every device arm encodes the same order, including the missing-direction rule as
an ascending candidate ordinal, and the default block reduction (`block.max` on
gain, then `block.min` on the slot) is sound because those primitives broadcast.

- **A 96-slot node with its winning gain tied at slots 6, 70 and 71.** At 64
  reduce threads that requires both the intra-thread ascending walk (6 beats 70)
  and the cross-thread `block.min` (6 beats 71), and it spans more than one warp.
  **No device test had ever fed the reduction a tie** -- the existing tie case
  exercised the host replica only, and every device case had a unique winner.
- **1,200 pseudo-random histograms**, 200 cases x 3 gradient regimes x 2 gain
  forms: given the *same* histogram, device replica and host scan chose the
  identical (feature, bin, direction) **1,200 times out of 1,200**.

### What it actually is

**The two backends do not read the same histogram.** The device reads fixed-point
sums of quantized gradients; the host reads Float64 sums. The 1,200-case sweep
agrees perfectly precisely because it fed both the same integers -- which
isolates the cause by construction.

Three pieces of corroboration that this is expected rather than broken:
`bench/real_data/verify.py` already downgrades this failure to a WARN citing the
documented near-tie divergence; multiclass predictions are sha256-**identical**
across backends while the two regression scenarios differ; and where they differ,
**metrics agree to 3.8e-05 and 2.2e-06 with GPU RMSE very slightly better**,
which a broken tie-break would not produce.

So the honest reframing: **row-level prediction parity is not achievable while
the backends deliberately read different histograms, and metric parity holds to
five decimal places.** The 0.115-0.169 figure has been treated as a defect all
week. It is the visible consequence of a design decision this project made on
purpose.

### What the lane did change, and a fifth dead mechanism

`SPLIT_TIE_RELATIVE` is a fraction of the **gain**, but this scan's resolution is
set by `parent_score / gain`, which the module's own docstring says and whose
measured table ranges to 293. At that ratio the indistinguishable width is more
than an order of magnitude past `SPLIT_TIE_RELATIVE * gain`, **so the near-tie
test was calling coin flips resolved exactly where the flips happen.**
`resolution_floor()` now takes the wider of the two widths, with the old
behaviour reachable and strictly narrower.

And the module's own remedy for this whole class, `host_rescan_recommended`, has
**no caller anywhere in `src/`**. Built, documented, dead. That is the fifth
instance -- after `HostGradientStage`, `DispatchSettings`, the stale
`tree_resident_requested` gate, and the hand-written scale mirror in
`distributed_gpu`.

**No bits move.** Nothing touches a scan, a reduction, or a gain expression; the
only behavioural change is which nodes a fallback that nothing calls would flag,
and it can only flag more.


---

## The speculation is built, and the instrument it shipped refuted my registered figure

### 27 wasted builds predicted, 0 issued

The registered economics said a speculative build adds a partition, a histogram
and a search pair to **every** step including dead ones. **That is not what
shipped and not what happens.** `_pick_runner_up_kernel` returns on `STEP_LIVE`
before it reads the frontier, so a dead step publishes nothing and costs three
near-empty launches. The speculation also runs **no search**, so there is no
speculative search pair at all.

Measured on the early-stopping fixture: the commit-log census says
`builds=29 wasted=27`; the device says `device_builds=2 device_consumed=2`.
**Twenty-seven wasted builds predicted, zero issued.**

**The census is an upper bound, not a count**, and it overcounts in four ways a
commit log structurally cannot see: a dead step, a budget spent after this
commit, no admissible pre-existing leaf, and no free slot. `SpeculationCensus.
builds` and `.wasted` are now documented as bounds and the census line carries
`device_builds` / `device_consumed` beside them.

**The concrete correction to my registered figure:** the budget check alone costs
the last growth step of every full-budget tree, so at 1M x 50 the honest issued
count is at most **2,800 not 2,900** and the waste at most **864 not 964**, about
10 percent less than registered. The instrument to measure it exactly now exists.

The 66.8 percent hit rate is unchanged and nothing in this lane bears on it; the
lane's own fixtures are a different shape and report 100 percent, which says the
rate is shape-dependent rather than that 66.8 is wrong.

### Two design corrections, both better than what I specified

**No extra pool slot and no extra scratch records.** I specified
`open_resident(num_leaves + 1)` and two more scratch records. The lane instead
builds into **the slot the next commit will acquire** -- lowest free by the commit
kernel's own upward scan, with nothing acquiring or releasing in between, so it
is a prediction with a proof. That matters practically: `open_resident`'s argument
lives in `train_gpu.mojo`, which the lane could not edit.

The parent's histogram survives a miss because the prebuild runs with `do_sub`
off -- my correction honored by a different and cheaper mechanism.

**A hit is decided by identity of the work, not of the leaf.** `_spec_consume_
kernel` compares ten descriptor fields -- window, routing rule, built-child
window, built slot, category set -- so it **does not rely on the K=1 theorem at
all**. It verifies directly that the launches already issued were handed this
step's arguments. Too strict costs a hit; too loose would be a wrong tree.

### The consumption counter proves consumption because it is the same branch

`SPEC_STAT_CONSUMED` increments in exactly one place: the branch that writes
`build_dev[STEP_LIVE] = 0`, which is the word that makes the real partition and
the real accumulation return at their first read. **The increment and the
suppression are the same branch**, so a speculation that launched everything and
hit nothing cannot move it. The test also asserts it **equal** to the census's
`consumed` per tree, and a two-leaf budget is a guaranteed zero pole asserting
zeros on both counters.

Bits do not move: forests compared node for node, `value` and `split_gain` as bit
patterns, over **eight** rounds -- one round would hide a scores-only divergence,
which is the bug that once cost this plane every tree after the first.

---

## The bin layout question is closed: feature-major stays

Two lanes reached the same verdict independently, by different routes, and the
second one found the argument that settles it.

**The row-side amortization the blocked layout was supposed to buy is already
bought by the shipping launch shape.** `_range_hist_atomic_kernel[GROUP, BIN_CAP]`
owns `GROUP` feature slots per threadgroup and reads the row side once for all of
them, on the feature-major buffer. What is left for a blocked layout is the bin
gather alone -- and **the gather is not broken**: consecutive threads take
consecutive `j`, node rows stay ascending, so at a fixed feature the warp's reads
are ascending and at the root exactly contiguous.

**A GPU coalesces across threads, not across one thread's own accesses.** So K1's
"no coalescing across the features a thread needs" is a CPU-shaped framing of a
pattern the device already handles. That sentence set this lane's registered
estimate at 1.5-2.5x, and it was the wrong mental model.

Three further findings, all **derived bounds**, nothing measured:

- **`GROUP = 1` is a regression, not a tie.** A launch owning one slot fetches a
  `G`-wide row, spends one byte, and the block is walked again per lane: `G`-fold
  streaming amplification. Pinned by test.
- **Feature subsampling inverts the sign.** Random `feature_fraction` leaves every
  storage block live while narrowing feature-major's active columns: 0.5 loses
  1.12x, 0.25 loses 1.77x.
- **The upload transform is NOT what makes it a bad trade** and should not be
  cited as though it were. It is tens of milliseconds against a seconds-long run.
  What makes it bad is that the win is 1.00x at the default bin count and that
  three kernel families index this buffer.

### The two lanes disagreed on a fact, and the fact is settled against K3

K3 stated that `free_feature_group` returns **1** at 256 bins, which is the basis
of its "provable no-op at the shipping configuration" argument. The layout lane
says **2 on Metal**.

**The layout lane is right.** `histogram_gpu.__init__`'s own docstring says
Metal's baseline is the pairing, measured at 1.39x end to end on an Apple M4 with
byte-identical predictions, and that "a 256-bin one gets the baseline back
unchanged". So shipping on this M4 is `feature_group = 2`, not 1.

**This does not change the verdict** -- G=2 is 1.00x to 1.12x, still inside the
refutation threshold -- but it invalidates K3's arm design, which was built on
`feature_group = 1` being the shipping baseline. Arm A is not the shipping
configuration on the device this round measures on.

### And a contract disagreement worth keeping

K3 states the contract as "histogram feature-blocked at `G = feature_group`"; the
layout lane states it as "**feature-major, full stop**". The difference is real:
K3 treats `G=1` blocked as identical to feature-major, which holds only if the
buffer is *built* at `G=1`, and a `G=8` buffer under a group of 1 is an 8x
regression. A blocked buffer would have to be rebuilt whenever `set_feature_group`
moves -- which is exactly what that setter exists to allow at run time.
Feature-major has no such coupling. **The layout lane's statement is the one to
keep.**

### Both lanes independently recommend the same next measurement

**The shared-atomic fraction of the histogram phase.** Three shared atomics per
(row, feature) are identical under both layouts, so they cancel in a comparison
but not in a speedup, and **their share is unmeasured**, which makes every
memory-side estimate in this batch an upper bound on wall-clock effect. Two lanes
that disagreed about the layout agree about this.

---

## The readback probe: the per-trip floor, the transport ranking, and a corrected constant

**Measured**, arms interleaved in one process, box at load 2.14 with two
campaigns idle-to-light. A ratio from adjacent arms survives a contended window;
the absolute microseconds are labelled accordingly.

| arm | cmd buffers/trip | per-trip | vs floor |
|---|---|---|---|
| `bare_sync` (empty queue, the floor) | 1 | **10.59 us** | 1.0x |
| `plain_one` (one packed copy, plain pointer) | 2 | **124.85 us** | 10.9x |
| `kernel_sync` | 2 | 149.29 us | 14.1x |
| `plain_pair` | 3 | 156.17 us | 13.6x |
| `pinned_one_sync` | 3 | 175.84 us | 16.6x |
| **`pinned_pair_sync` (what ships today)** | 4 | **202.14 us** | 19.1x |
| `map` | 3 | 349.47 us | 30.4x |

**The lane's structural prediction was right without a timing**: `plain_one` wins,
`map` is worst despite its docs wording. Shipping 202.14 to `plain_one` 124.85 is
**77 microseconds a trip, 38 percent**, from packing six words into one buffer and
copying into ordinary heap memory rather than two pinned ones.

### The 458 microsecond constant is superseded, and it was too high

This campaign has priced every host trip at ~458 microseconds, **derived** from
the depthwise A/B. A trip is now **measured directly at 202 microseconds** on the
shipping transport.

So the standing per-fit arithmetic changes: 200 trips at 202 microseconds is
**0.040 seconds**, not the 0.092 the derived constant gave. **The control plane
is worth less than half what the campaign has been assuming**, and everything
downstream of that constant -- including the estimated 0.05 second remainder that
closed the control-plane chapter -- was too pessimistic in the direction of
overvaluing trip removal.

That does not reopen the chapter; it closes it harder. It also means the
`trip-count` lane's target of ~10 trips per fit is worth about **0.038 seconds**,
not 0.09, and it should be judged against that.

### Queue depth: the docstring's claim is supported

Enqueue cost per launch, against stream length:

| launches | enqueue us | | launches | enqueue us |
|---|---|---|---|---|
| 1 | 7.00 | | 96 | 9.90 |
| 16 | 5.94 | | 128 | 10.81 |
| 32 | 6.97 | | 192 | 14.13 |
| 48 | 7.19 | | 256 | 14.75 |
| 64 | 6.59 | | 512 | 16.63 |

**Flat at 6-7 microseconds through 64, then rising to 14-17 beyond it.** The knee
sits between 64 and 96, which is exactly where a 64-deep queue would put it. So
`gpu_resident_round`'s claim -- that the plane emits on the order of 306 command
buffers between waits and therefore runs one-in-one-out -- **is supported**, and
the cost of being past the depth is roughly a doubling of per-launch enqueue time.

**And it cannot be raised.** No knob exists: zero load sites on both raising
selectors, no such environment variable, and the compiler lists four
`DeviceContext` constructor overloads, none taking one.

### Four transports do not exist, each with its reason

`direct` (a `DeviceBuffer` pointer is a GPU virtual address that faults on a host
load), `spin` (needs one allocation both kernel-writable and host-readable, which
MAX exposes on neither type), `cpu_callback` (`enqueue_cpu_function` is CPU
contexts only), and `event` (`eventCreate is not supported on this device`).

The probe prints each unavailable arm **with why**, so a future MAX that closes a
gap shows up as `unavailable -> ok` with no edit to the harness. That is the right
shape for a capability probe and it is worth copying.


---

## The clock-state probe: indistinguishable, and the question closes with a number

Bucket C. A fixed reference kernel -- `bench/canary.mojo`'s GPU probe, reused
rather than reinvented so the reading is comparable with the canary line already
in every window header -- timed under three conditions the probe controls, seven
repeats each, **interleaved rather than blocked** so a drift across the run
cannot land entirely on one condition.

| condition | median | spread |
|---|---|---|
| **cold** (after a 2-second idle gap) | 252.8 ms | 10.0% |
| **saturated** (immediately after a queue-filling burst) | 248.9 ms | 9.8% |
| **warm** (burst, drain, then measure) | 243.5 ms | 5.6% |

`saturated/cold` = **1.015**, `warm/cold` = **1.038**. Best gain over cold is
**3.7 percent against a 10.0 percent band**: **indistinguishable** under M0.

### What that closes

The Metal timeline found runs sitting at the **Minimum** GPU performance state
with the same kernel ~2.8x faster at Maximum, and nothing in this campaign could
control or record it. **At these workloads the clock is not moving**, so:

- **No lane follows.** The warm-up burst does not become a rule-(2) switch.
- **`launch-fusion`, `trip-count` and the speculative prebuild do NOT get a
  second payoff.** Each removes host stalls; if idle gaps had cost clock state,
  they would have been buying it back on top of the work they remove. They are
  worth exactly the work they remove and no more, which is the honest and less
  flattering reading.
- The 2.8x figure from the timeline is **not refuted** -- it was observed on a
  different workload shape and probably a different power regime. What is refuted
  is the idea that *this campaign's* idle gaps are causing it.

### The caveat that comes with it, stated because it is the way this is wrong

`IDLE_SECONDS = 2.0` is **chosen, not measured**. A machine that downclocks on a
shorter gap than two seconds would read as flat here, because every "cold" sample
would already be warm. The probe's own verdict text says so rather than leaving
it to a reader.

So the honest claim is narrow: **no clock effect at a two-second idle gap on this
reference kernel**. Not "the clock is fixed". If a future window sees an
unexplained regime shift, the first thing to vary is that constant, and the probe
is in the tree to make that a one-line change.

Taken at load 8.63 -- a busy box -- which the ratio design tolerates and the
absolute milliseconds do not. The three medians are not quotable as kernel
timings; only the ratios between them are.
