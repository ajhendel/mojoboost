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

The resident plane wins there by 2.2x. So the gate's premise is falsified at
precisely the shape it was built to protect, and re-deriving or deleting it is
now a live question rather than a tuning exercise. That is a **reduction** in the
number of paths, not an addition of a small-data path.

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

**Unresolved. The CPU campaign is measuring `force_row_wise` against
`force_col_wise` at 1,000,000 x 50, both pinned. Until that lands, every margin
in this file carries this caveat.**

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
