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

## Addendum: the path taken, which the sweep above failed to record

The table above has no column for which split-search path each arm took, and it
should have. `normalized_split_work` is `n_rows * active_features * (n_bins/255)
* (num_leaves/31)` and the automatic policy sends anything below 50,000,000 to
the host scan. So 250,000 x 50 is 12.5 million and **both GPU arms took the host
scan there**, while 1,000,000 and 2,000,000 took the device search. The three
points of each GPU arm are not measurements of one configuration.

Forcing the device search at 250,000, five repeats each:

| arm at 250,000 x 50 | automatic (host scan) | device search forced |
|---|---|---|
| GPU leaf-wise | 1.967 | 2.268 |
| GPU depth-wise | 1.909 | **1.214** |

**The gate is right for leaf-wise and wrong for depth-wise.** Leaf-wise is 15
percent worse on the device search at this size, which is what the threshold
exists to prevent. Depth-wise is **37 percent better**, because it batches a
level into one search and the host-scan path cannot batch a 153 KB download and
a synchronization per node away.

So the gate costs depth-wise 0.70 seconds at 250,000 rows, and it costs it
because `normalized_split_work` does not know which growth policy it is deciding
for. The threshold was measured for leaf-wise and is applied to both.

That does not overturn the fixed-cost reading, but it does qualify it. Corrected
for path, the depth-wise points are 1.214 / 2.587 / 5.417, fitting 1.83 us per
row from 250,000 to 1,000,000 and 2.83 from 1,000,000 to 2,000,000. Still
nonlinear, so something other than the gate remains unexplained in that arm.

It also moves the practical answer. Depth-wise on the device search at 250,000
is 1.214 against LightGBM's 1.023, so **1.19x behind rather than 1.87x**. It
does not beat LightGBM there, but the claim "the GPU loses below a million rows"
was substantially the gate's doing rather than the hardware's.

**Every future sweep records the path taken per arm per shape.** The backend
proof and the sync count are already collected by the harness; not putting them
in the table is what let three points of two different configurations be fitted
as one line.

## The correction that matters most, from the same numbers

Our GPU's marginal cost per row equals LightGBM's **on ten CPU cores**. Taken
with the intercepts, the arithmetic is unforgiving: remove the whole one second
of excess fixed cost and 1,000,000 rows lands near 2.8 seconds, which is
**parity, not a win**.

Depth-wise beat that number, so it did not only remove fixed cost. Level-batched
histograms also fill the GPU better, which moves the **slope**. That reframes the
plan: the histogram kernel is not a phase two behind the control plane, it is
**the entire margin**. A GPU whose per-row cost equals a laptop CPU's is leaving
a large multiple on the table by the layout arithmetic, and closing the control
plane alone reaches parity and stops.

## Addendum 2: what the intercept is made of, measured rather than assumed

The fitted intercept above (about 1.42 seconds) was attributed to per-split host
round trips on the strength of the Metal timeline. That was an inference. It has
now been measured directly, with the phase profile's counters, which are exact
counts rather than timings and so are unaffected by machine load.

At 1,000,000 rows by 50 features, 100 trees:

| arm | syncs | syncs per tree | dispatches | wall |
|---|---|---|---|---|
| GPU leaf-wise | 3,100 | 31 | 24,400 | 3.726 s |
| GPU depth-wise | 600 | 6 | 19,400 | 2.581 s |

**Depth-wise removes 2,500 synchronizations and buys 1.145 seconds**, which is
**458 microseconds per synchronization**. That is independently close to the
606 microsecond median blocking readback the Metal System Trace measured by a
completely different method.

It also removes 5,000 dispatches, but at the measured 12.62 microseconds per
enqueue those are worth about 63 milliseconds, roughly five percent of the
saving. **The synchronizations are the cost; the dispatches are not.**

The arithmetic then closes from both ends. 3,100 synchronizations at 458
microseconds is **1.42 seconds**, which is the intercept fitted from three
wall-clock points in the table at the top of this file. A curve fit over
wall clocks and a counter multiplied by a per-event cost agree to two decimal
places, having shared no inputs.

### One consequence, and it is more optimistic than the earlier reading

An earlier analysis held that removing the excess fixed cost lands one million
rows at parity, on the reasoning that ours is 1.42 seconds against LightGBM's
0.44 and the difference is about one second. That treats LightGBM's intercept as
a floor for ours, and nothing makes it one.

A control plane at one synchronization per tree would pay 100 of them, or about
0.046 seconds, rather than 1.42. That is an **estimate** built on a measured
per-sync cost and a design's static count: 3.756 minus 1.374 is roughly **2.38
seconds against LightGBM's 2.767**, which is a win of about 1.16x rather than
parity.

Two things temper it, and both are real. `docs/GPU_PORTABILITY.md` section 6
establishes that `enqueue_copy` drains the queue in **both** directions on Metal,
so the per-round gradient upload is itself a synchronization that a
"one wait per tree" design does not count. And the resident plane does not
currently run at all. So the honest statement is that the target is a win rather
than parity, and that nobody has yet paid for it.

### What this does not change

The slope finding stands untouched, and it is still the larger problem. Our
marginal cost per row equals LightGBM's on ten CPU cores whatever happens to the
intercept. Removing every synchronization leaves that entirely intact, so the
histogram kernel remains where the multiple has to come from.

## Addendum 3: why the gate moved one arm and not the other

Asked as "are we fixated on depth-wise". The counters answer it without
argument. At 250,000 rows by 50 features with the policy-aware gate live:

| arm | syncs | per tree | wall |
|---|---|---|---|
| GPU leaf-wise | 3,100 | 31 | 1.956 s |
| GPU depth-wise | 600 | 6 | 1.207 s |

**The gate changes depth-wise's synchronization count and does not change
leaf-wise's at all**, which is why it moved one arm 0.70 seconds and the other
by nothing.

The mechanism is the growth policy, not the scan. Leaf-wise needs an answer
after every split before it can enqueue the next, so it makes 31 round trips per
tree whether the scan runs on the host or on the device. Moving the scan changes
only *what* comes back, a 136-byte record instead of a 153 KB histogram, and on
Metal that is irrelevant: a blocking readback costs about 450 microseconds
regardless of its size, of which under four microseconds is moving bytes.
Depth-wise commits a whole level per step, so a level's answers batch into one
wait and 31 becomes 6.

So the scan location was never a lever for leaf-wise, and the earlier finding
that the device search merely ties the host scan for leaf-wise at 250,000 was
the same fact seen from the other side.

**Depth-wise is a control arm, not a direction.** It is the only working way to
measure what removing per-split waits is worth, because the leaf-wise resident
plane crashes whenever it executes. Every depth-wise number is a preview of
where leaf-wise lands once that plane runs, and the plane removes *more* waits
than depth-wise does, one per tree against six, while growing the same tree the
project grows today.

The one thing depth-wise genuinely teaches leaf-wise is separate from the waits:
building a level's histograms in one launch fills the GPU better than building
one small leaf at a time. Leaf-wise can have that without changing its tree, by
building the frontier's pending histograms speculatively, since a child
histogram is valid whenever it is computed. That is an exact transformation, not
an approximation, and it is where this result should feed back in.
