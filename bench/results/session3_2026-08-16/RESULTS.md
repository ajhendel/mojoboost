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

## Still open

- S1 is still two shapes short: the resident plane at 250,000 and at 50,000.
- The parallel-grain check at the host-scan shape (M2.5), untaken.
- M2.3 re-taken in a fast window.
- Whether tonight's machine state is thermal, clock, or something else. Nothing
  in this session establishes it and the results are labelled by regime rather
  than corrected for it.
