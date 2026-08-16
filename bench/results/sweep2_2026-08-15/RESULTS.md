# Sweep II, Apple M4, 2026-08-15

Run against the rules committed beforehand in `bench/results/PROFILE_PROTOCOL.md`,
on a quiet machine, at commit `7443673`. Mojo 1.0.0 (ed45d567). Apple M4, 10
cores (4 performance, 6 efficiency), 16 GB. No thermal warning recorded.

Five repeats per arm, median as the decision statistic with spread beside it.
LightGBM 4.7 at 10 threads via `bench/bench_lightgbm.py`, same shapes, 100
rounds, 31 leaves, 255 bins, squared error. All figures are seconds of train
time, binning excluded. All **measured** unless labelled otherwise.

## The table

| shape | our CPU | our GPU | our GPU depthwise | LightGBM 10t |
|---|---|---|---|---|
| 250,000 x 50 | 1.649 | 1.967 | 1.909 | **1.023** |
| 1,000,000 x 50 | 5.942 | 3.756 | **2.587** | 2.767 |
| 2,000,000 x 50 | 13.483 | 6.093 | **5.417** | 5.228 |

Spreads: our GPU 1.1 / 1.8 / 12.5 percent; depthwise 6.7 / 0.3 / 10.8; our CPU
9.3 / 13.5 / 35.6. The 2,000,000-row CPU arm at 35.6 percent is too noisy to
carry a verdict and is reported rather than relied on.

Ratios against LightGBM, where above one means behind:

| shape | CPU | GPU | GPU depthwise |
|---|---|---|---|
| 250,000 | 1.61x | 1.92x | 1.87x |
| 1,000,000 | 2.15x | 1.36x | **0.93x, ahead** |
| 2,000,000 | 2.58x | 1.17x | 1.04x |

## The answer to "where are we behind"

**Only on fixed cost. Our marginal cost per row now matches LightGBM's.**

Fitted from the measured points, three shapes each:

| | 250k to 1M | 1M to 2M |
|---|---|---|
| our GPU | 2.385 us/row | 2.337 us/row |
| LightGBM | 2.325 us/row | 2.461 us/row |

Two consequences, and both retire a claim this project has been carrying.

**The slope is flat, so the cost per row is not superlinear above one million
rows.** That was the open question this sweep existed to answer, and the answer
is no. The 3.2 microseconds per row fitted from the pre-round-2 one-to-five
million pair does not survive; whatever it was measuring, round two removed it.

**Our slope and LightGBM's are the same to within the spread.** The earlier
reading that we are about 10 percent better per row is not supported either.
Across four fitted segments the two libraries are 2.33 to 2.46 microseconds per
row and they interleave. On the marginal cost of a row, this library and
LightGBM are even.

So the entire deficit is the intercept. Extrapolating each fit back to zero
rows: ours is about **1.42 seconds**, LightGBM's about **0.44 seconds**. The gap
is roughly **1.0 second of fixed cost**, and it does not grow with the data.
That is the whole of what is wrong, and it is exactly what the Metal timeline
said in a different unit.

## Depthwise beats LightGBM at one million rows

2.587 against 2.767, at a 0.3 percent spread, which is the tightest arm in the
sweep. It is at parity at two million (5.417 against 5.228, 3.6 percent behind)
and behind at 250,000 where the fixed cost dominates everything.

**This does not become the default, per rule S2 written before the sweep ran.**
Depth-wise growth grows a *different tree* than leaf-wise, so the win belongs to
a user who asked for depth-wise, not to one who asked for leaf-wise and got it
substituted. Answering a speed question by changing the model is not something
this project does.

What it is instead is the strongest available evidence for the control-plane
thesis. Depth-wise batches a level into one host wait rather than one per split,
roughly 5 per tree against 30. It is the same fixed cost being removed by a
cruder mechanism, and removing it is worth 1.17 seconds at one million rows,
which is close to the 1.0 second intercept gap the fits identify. Two
independent routes to the same number.

## The tree-resident plane does not work, and its tests were vacuous

The device-resident control plane raises `the device-owned tree stopped
abnormally: running` **every time it actually executes.**

The failure was initially invisible because of an interaction nobody predicted.
The plane is entered from inside `_grow_tree_gpu_device_search`, so it runs only
when the automatic policy selects the **device** split search, which happens
when `normalized_split_work` reaches 50,000,000. Every shape below that gate
falls back to the host scan, and the fallback is the shipping path, so it
succeeds.

That is why it appeared to work at 850,000 x 50 (42.5 million, host scan) and at
1,000,000 x 20 (20 million, host scan) and failed at exactly 1,000,000 x 50,
which is 50,000,000 exactly and the only shape in the sweep on the device side of
the gate.

Forcing the gate settles it: at 20,000 rows with `MOJOTREES_GPU_SPLIT_STRATEGY=device`,
the plane fails with the gate on and the run succeeds with it off.

**Every equivalence test in `tests/test_gpu_tree_resident.mojo` ran at 4,000 to
6,000 rows, far below the gate, so every one of them compared the host-scan
fallback against itself and passed by vacancy.** The file asserted trees are
node-identical with no tolerance across six configurations, and none of those
six exercised a single line of the plane it was written to test. That is my
error: I wrote the file, ran it, and reported the plane as verified.

The lesson is narrow and worth keeping. A test for a gated path has to prove the
gate opened. This file now would need to force the split strategy and assert the
plane actually ran, rather than assuming a configuration reaches it.

## What this sweep does not say

- Nothing about GPU clock state, which the protocol asked for. `powermetrics`
  needs elevated privileges and was not run. The tight spreads on the GPU arms
  at 250,000 and 1,000,000 (1.1 and 1.8 percent) argue against a clock
  transition inside those windows, but that is an inference, not a measurement.
- Nothing about 5,000,000 rows. The sweep stopped at 2,000,000, which is enough
  to answer the superlinearity question but leaves the far tail unmeasured.
- Nothing about sync counts per arm, which the protocol asked for and which
  would have caught the tree-resident failure earlier had it been collected
  first.
- The two-million-row CPU figure, at 35.6 percent spread, is not reliable.
