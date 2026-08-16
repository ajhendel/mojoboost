# A clean sheet GPU trainer for Apple silicon

This document answers one question: if nothing existing had to be preserved,
what is the fastest gradient-boosted-trees trainer that can be built for an
Apple M4, subject to holding held-out metric parity with LightGBM?

It is a design, not a plan of record and not a review. It does not evaluate the
shipped implementation. It reads the shipped implementation only where the code
is the cheapest available description of what the hardware does, and it says so
each time it does that.

**Correction, 2026-08-15.** One measured input to the arithmetic below has
since been remeasured and did not hold. This document's per-row slopes come
from a three-point fit at 50,000, 250,000, and 1,000,000 rows and put our
marginal cost 14 percent below LightGBM's (2.05 against 2.39 microseconds per
row, section 3). A five-repeat sweep to 2,000,000 rows
(`bench/results/sweep2_2026-08-15/RESULTS.md`) fits four segments in which the
two libraries span 2.33 to 2.46 microseconds per row and **interleave**, so on
the marginal cost of a row this library and LightGBM are even and no per-row
advantage may be assumed. That changes the conclusion of the short version
below: removing the whole fixed-cost term is now **estimated** to reach parity
with LightGBM at 1,000,000 rows rather than to beat it, and the margin, if
there is one, has to come from the layout work in sections 4 onward. The
measured shipped datum that supports the layout half of the argument is the
depth-wise arm in that same sweep, which moved the slope and not only the
intercept. Every projection in this document that multiplies a per-row
advantage should be read against that.

Every number below is either **measured**, **derived** by arithmetic from
measured numbers, or **assumed**. Those three words are used literally
throughout. Where a quantity cannot be obtained on this device at all, the
section says so and names the experiment that would obtain it. There are eight
such places and they are collected in section 8. Section 10 does the same for
the platform capabilities the design depends on, separating the ones the shipped
code has already exercised on this M4 from the ones I could only read about.

The short version. The 3.58 seconds a 1,000,000 by 50 fit takes decomposes,
by two independent routes that agree to four percent, into roughly 1.5 seconds
of host serialization that no arithmetic requires and roughly 2.1 seconds of
device work that is within a couple of percent of what its own memory layout
makes inevitable. Neither term is a kernel-tuning problem. The first is a
control-plane shape and the second is a bin-layout shape. Fix the control plane
and the trainer beats LightGBM on this machine at this shape; fix the layout too
and it beats LightGBM by three to four times; and both fixes are model
preserving, so the accuracy constraint costs nothing to satisfy. Only the third
change, dropping the bin count, touches the model, and it is third for that
reason.

Growing the tree leaf-wise, as LightGBM does, turns out to be affordable rather
than the obstacle it looks like. The case for replacing it with level-wise or
oblivious growth is a case about synchronization cost, and this design's first
move is to take the synchronization cost to zero.

---

## 1. Where the 3.58 seconds goes

### 1.1 The measurements this rests on

All of these were taken on the machine in question and are quoted from
`docs/METAL_TIMELINE.md` and `bench/results/profile_2026-08-15/RESULTS.md`.

| quantity | value | shape |
| --- | --- | --- |
| our GPU train time | 3.58 s | 1,000,000 x 50, 100 rounds, 31 leaves, 255 bins |
| our GPU train time | 1.89 s | 250,000 x 50 |
| our GPU train time | 1.63 s | 50,000 x 50 |
| LightGBM, 10 threads | 2.86 / 1.00 / 0.59 s | same three shapes |
| GPU idle share of span | 76.5% | 200,000 rows, Maximum clock 77.9% of capture |
| GPU idle share of span | 87.5% | 50,000 rows, Maximum clock 70.7% of capture |
| compute GPU time | 536.96 ms | 200,000 rows, 100 rounds |
| compute GPU time | 231.20 ms | 50,000 rows, 100 rounds |
| blit GPU time | 15.33 ms | 200,000 rows, whole run |
| serialization points | 32.1 per round | measured twice, by two instruments |
| one blocking readback | 606 us median | of which 3.7 us is bytes moving |
| enqueue (commit call) | 12.62 us median | device property, stable across shapes |
| completion notification | 101.33 us median | device property |
| commit N to commit N+1, unblocked | 14.58 us median | |
| median gap between kernels | 3.33 us | |
| idle inside one queue wait | 162 us median | GPU has work submitted and has not begun it |

Two of these deserve a note before anything is built on them. The busy/idle
split is confirmed by two independent tables in the trace that agree to 2.4
percent, and it is not a downclock. And the per-dispatch costs are device
properties rather than workload properties: between the 50,000 and 200,000 row
captures the median enqueue moved by 0.05 microseconds and the completion
notification by 4.4, while compute between them changed by a factor of 2.3.

### 1.2 First reconstruction: decompose the readback

From the lifecycle decomposition of the 3,206 blocking readbacks at 200,000
rows, the terms that are *not* the GPU doing useful work are:

```
the commit call itself                        0.35 ms per round
GPU idle inside the queue wait                5.08 ms per round
GPU end -> completion signal                  3.57 ms per round
completion signal -> next commit              5.68 ms per round
                                             --------
non-overlapped host serialization            14.68 ms per round
```

The one term deliberately excluded is the other half of the queue wait, the
51.3 percent during which the GPU was executing previously submitted work.
Waiting for real work to drain is not overhead and is not counted here.

14.68 milliseconds per round is **derived**, by summing measured medians. Over
100 rounds it is 1.47 seconds.

### 1.3 Second reconstruction: the scaling line

Fit a straight line through the two extreme shapes, 50,000 rows at 1.63 s and
1,000,000 rows at 3.58 s:

```
slope     = (3.58 - 1.63) / 950,000  = 2.053e-6 s per row
intercept = 1.63 - 50,000 * 2.053e-6 = 1.527 s
```

The middle point checks out to within eight percent: the line predicts 2.04 s
at 250,000 rows against a measured 1.89. So the fit is a fair description and
not an artifact of the two points chosen.

**1.53 seconds of the 3.58 does not scale with rows.** Compare that with the
1.47 seconds the trace decomposition produced. Two instruments that know
nothing about each other, one a Metal system trace and one a wall clock across
three problem sizes, agree on the fixed cost to within four percent.

I regard this as the most solid quantitative fact in this document, and every
design decision in section 2 is downstream of it.

### 1.4 The same line, run against LightGBM

```
LightGBM slope     = (2.86 - 0.59) / 950,000 = 2.389e-6 s per row
LightGBM intercept = 0.59 - 50,000 * 2.389e-6 = 0.471 s
```

This is worth pausing on, because it inverts the usual reading of the
benchmark table. **Our marginal cost per row is already 14 percent lower than
LightGBM's** (2.05 against 2.39 microseconds per thousand rows). We lose the
1,000,000 row shape entirely on a fixed term: 1.53 seconds against their 0.47.

The brief quotes a per-row slope of 3.2 microseconds against LightGBM's 2.4.
That is the *average* cost per thousand rows at the 1,000,000 row shape
(3.58/1.0 and 2.86/1.0), which mixes in the fixed term. The *marginal* slopes,
which are what a scaling argument needs, are 2.05 and 2.39. Both readings are
correct about different quantities and the difference between them is exactly
the fixed cost this section is about. I use the marginal figures below.

The consequence is blunt. A trainer that removed its fixed cost entirely and
changed nothing else would land at 1,000,000 rows at about
`2.053e-6 * 1,000,000 = 2.05` seconds, against LightGBM's 2.86. That is a 1.4x
win, from a change that touches no kernel and no arithmetic.

### 1.5 The variable term, and a correction to a widely quoted number

`docs/METAL_TIMELINE.md` reports compute at 22.9 percent of a round. That is a
200,000 row figure and it does not carry to 1,000,000 rows, because compute
scales with rows and the serialization does not. Extrapolating compute the same
way:

```
compute per round, 50,000 rows  = 231.20/100 = 2.312 ms
compute per round, 200,000 rows = 536.96/100 = 5.370 ms
slope     = (5.370 - 2.312)/150,000 = 2.039e-5 ms per row
intercept = 2.312 - 50,000*2.039e-5 = 1.293 ms
compute per round at 1,000,000 rows = 1.293 + 20.39 = 21.7 ms   (derived)
```

Check the whole budget against the measurement:

```
21.7 ms compute + 14.68 ms serialization = 36.4 ms per round
36.4 ms * 100 rounds = 3.64 s      against a measured 3.58 s
```

The budget closes to within 1.7 percent. So at the shape everyone quotes:

> **At 1,000,000 by 50, the round is roughly 60 percent device compute and
> 40 percent host serialization.** The 22.9 percent compute figure is a
> 200,000 row figure and quoting it at 1,000,000 rows understates the kernel
> by nearly a factor of three.

Caveats, stated because this is a five-fold extrapolation off two points. The
two captures had different clock-state mixes (Maximum 70.7 and 77.9 percent),
which biases the compute slope by an unknown amount in an unknown direction.
Compute time also has a floor of 1.29 ms per round independent of rows, which
is the tail of small nodes that cannot fill the device, and a five-fold
extrapolation of a two-term model is not a strong instrument. What makes me
willing to use it is that the budget closes: two extrapolated terms and a
measured total agree to under two percent, and that is not something a badly
wrong slope usually does.

### 1.6 The target

The brief gives an ideal-layout roofline for the data plane of roughly 0.4
seconds per 100 rounds at this shape. Section 3.6 rebuilds that number from
first principles and gets 0.18 to 0.28 seconds under a fully efficient layout
at 63 bins, and 0.63 to 0.95 seconds at 255 bins. The stated 0.4 is inside that
range and I use it as the target without further argument.

So the design has to move 36.4 milliseconds per round to about 4, and the two
terms it has to move are of comparable size. Neither one alone is enough.

---

## 2. The control plane

### 2.1 What the round trip actually costs, and therefore what to attack

One blocking readback is 606 microseconds of wall clock. Of that, 3.7
microseconds is the GPU moving the bytes, 110 is the operating system
delivering a completion notification, 165 is host code between the notification
and the next commit, and 298 is the interval between the commit returning and
the GPU starting the blit, roughly half of which is genuine queue drain.

Only the 158 microseconds of idle-inside-queue-wait looks like something a
faster path could recover. So **removing one synchronization is worth 606
microseconds and making one faster is worth at most 158.** The target is the
count, and the count has to go to zero, not down.

That framing rules out a whole family of half-measures. Batching readbacks,
compressing what crosses the boundary, using unified memory more cleverly,
pinning staging buffers: all of these attack the 3.7 microseconds or the 158,
and the trace already says the GPU spends 0.65 percent of the span moving
bytes. There is nothing there.

### 2.2 The design: a fixed dispatch schedule over device-resident parameters

Three properties together get the host wait count to zero, and all three are
needed; any one missing puts it back.

**Property one: every kernel reads its own parameters from device memory.**
No kernel takes a node id, a row range, a threshold, a feature index, a bin
count, a leaf value or a scale as a launch argument. There is a single
device-resident `TreeState` struct that holds the frontier table, the split
records, the leaf values, the node row ranges, the fixed-point scales and the
loop counters, and every kernel binds it and reads what it needs. The host
never learns a split, so it never needs to be told one.

**Property two: the dispatch schedule is fixed at enqueue time and does not
depend on data.** There is no indirect dispatch, so grid sizes and the number of
dispatches must be host-side constants. The schedule is therefore always the
worst case: `num_leaves - 1` split iterations are always enqueued, whether or
not the tree finds that many splits, and every kernel begins by reading a
device-resident `active` flag and returning immediately if the iteration is
dead. Enqueue is a host-side loop over constants; nothing in it consults the
device.

This property is worth more than the wait-count argument alone, because it is
also the precondition for capturing the round as a replayable graph. Section 2.7
returns to that.

**Property three: work sizing is data-dependent through a device-resident work
queue, not through the grid.** Each of the heavy kernels launches a small fixed
grid (a persistent-threads grid, sized to fill the device once) and each
threadgroup pulls work items from a queue in device memory using a global
atomic counter. The previous kernel writes the queue. This is what indirect
dispatch would have bought, obtained without it, and it is the piece that makes
property two affordable: a dead iteration costs one small grid of threadgroups
each doing one atomic read and returning, not a grid sized for a million rows.

It is worth being precise about why the persistent-threads pattern is safe here
when a grid barrier is not, because "persistent kernel" is often used to mean
both and only one of them is safe on Metal.

A work-queue kernel never waits on another threadgroup. Each threadgroup loops
`item = atomic_fetch_add(queue_head, 1)` until the queue is exhausted, does the
item, and exits. A threadgroup that the scheduler never starts simply never
takes an item, and the ones that did run drain the queue between them. There is
no forward-progress requirement and no barrier. A grid barrier requires the
opposite: every participating threadgroup must be resident simultaneously, or a
threadgroup spinning at the barrier holds a core while waiting for one that can
no longer be scheduled. Section 2.5 works through why that cannot be made safe
here.

There is one determinism constraint on the queue and it must be written into
the kernel rather than discovered later: **a work item carries the output slot
it writes, it does not derive the slot from the threadgroup id.** Which
threadgroup takes which item varies between runs; which slot an item writes must
not. Section 4.3 depends on this.

### 2.3 Host wait counts, option by option

Counting a *wait* as any point where the host blocks until the device has
finished something. This is the table the brief asks for.

One clarification first, because the brief's phrasing of "one command buffer per
tree or per round" does not quite map onto what the runtime does. In the MAX
Metal path every `enqueue_function` becomes its own command buffer holding
exactly one encoder; there is no encoder batching anywhere. The trace shows this
directly: 22,107 command buffers carrying work, one encoder each. So "one
command buffer per round" is not something this API can express today. What it
*can* express is a round in which nothing blocks: the queue is in order and
asynchronous, so 187 command buffers submitted back to back with no
`synchronize()` between them behave, from the device's point of view, exactly
like one. The measured evidence that this works is the 3.33 microsecond median
inter-kernel gap and the 14.58 microsecond unblocked commit interval. The unit
that matters is therefore the **wait**, not the command buffer, and the table
counts waits.

| option | waits per tree | waits per round | notes |
| --- | --- | --- | --- |
| A. Host decides each split (today's shape) | 31 | ~32 | measured at 32.1; 606 us each |
| B. Device search, host commits | 30 | ~31 | the readback shrinks to 136 bytes and the cost does not change, because the cost is the round trip, not the bytes |
| C. Device search batched per level | 5 to 6 | 6 to 7 | at 31 leaves, a depth-5 tree |
| D. Fully device-resident tree, host waits at the end of the tree | **1** | 1 | this is the whole win; everything past it is small |
| E. Device-resident tree plus gradient and score update, host waits at the end of the round | **1** | 1 | same count as D |
| F. Pipelined rounds: submit round r+1 before waiting on r; check a device-computed early-stopping flag one round late | **0** | 0 in steady state | one wait at the end of the fit, plus one when the flag says stop |
| G. Whole fit submitted without any wait | 0.01 | 0.01 | one wait per fit; see the risks below |
| H. Persistent megakernel with a software grid barrier | 0.01 | 0.01 | unbounded deadlock risk, section 2.5 |

The design is **F**. D and E get the count to one, which already recovers
essentially all of the 1.47 seconds, since one wait per round is 606
microseconds against 32 of them. F is a small further step and mostly buys
robustness: it means a round's wall clock is never gated by a host that happened
to be descheduled.

Arithmetic for the recovery. Today the non-overlapped serialization is 14.68 ms
per round, spread across 32.1 waits, so 0.457 ms of it belongs to each wait.

Option E keeps one wait per round, so it keeps one wait's worth: **0.46 ms per
round**, or 0.046 s over 100 rounds, against 1.47 s today.

Option F keeps none in steady state, because the host commits round r+1 while r
is still running and reads the stop flag from a buffer that has already
completed. What is left is the commit calls themselves, which section 2.6 prices
separately and which are not a wait.

So the recovery is **1.47 s -> somewhere between 0.00 and 0.05 s**, and rounding
generously for everything this model does not see, call it 1.47 -> 0.10 s. A
saving of about 1.37 seconds, or 38 percent of the current total, before a
single kernel changes.

### 2.4 The dispatch schedule of one round

Per round, with leaf-wise growth at 31 leaves, the dispatch schedule is:

```
1   fused score update and gradient/hessian fill          (1 dispatch)
2   global |g| and |h| magnitude reduction -> scales      (2 dispatches: reduce, finalize)
3   quantize gradients to interleaved fixed point         (1 dispatch)
4   root histogram                                        (2 dispatches: partial, reduce)
5   for each of 30 split iterations:
      scan, argmax, commit, write child records           (1)
      partition the parent's row window                   (2: flag+scan, scatter)
      child histogram with fused sibling subtraction      (2: partial, reduce)
6   finalize leaf values                                  (1)
```

That is `1 + 2 + 1 + 2 + 30*5 + 1 = 157` dispatches per round, against a
measured 221 today. The dispatch count is not the headline; the wait count is,
and it goes from 32 to zero. But section 2.6 shows the dispatch count becomes
the binding constraint once the waits are gone, so it is worth spending a little
design effort on it now.

The scan, argmax and commit are one dispatch rather than three because **one
threadgroup can do the whole node.** The scan is a per-feature prefix sum over
the node's histogram, 50 features by 256 bins is 12,800 cells, which 256 threads
handle at 50 cells each. Cross-feature argmax then needs only a `barrier()` and
a block reduction, both of which are available inside a threadgroup, and the
commit is a handful of stores by thread zero. Splitting this across
threadgroups would require a device-wide barrier, which does not exist here;
keeping it inside one threadgroup is what makes it a single dispatch. It also
localizes every accuracy-critical decision, the gain formula and the guards and
the monotone clamp and the leaf value, into one kernel, which is where section 4
wants them.

Cost of the dispatches themselves, on the device: 157 gaps at the measured
3.33 microsecond median is 0.52 ms per round, 52 ms over 100 rounds.

Cost of dead iterations. A tree that finds only 24 splits still enqueues 30, so
6 iterations of 5 dispatches each run as no-ops. With persistent grids of a few
hundred threadgroups, a no-op dispatch is an atomic read and a return; I
**assume** 2 to 5 microseconds of GPU time each, which is consistent with the
measured distribution (half of all compute dispatches today are under 8.21
microseconds). 30 no-ops at 5 us is 0.15 ms per round, 15 ms over 100 rounds.
Acceptable. I do not know the true cost of a no-op dispatch on this device and
section 8 names the microbenchmark.

**Early stopping without a wait.** The round's validation metric is computed on
device at the end of the round and a single `should_stop` word is written. The
host reads that word from the *previous* round's already-completed command
buffer, so the read never blocks. Stopping is therefore one round late, and one
extra round of work is discarded. That is a real and acceptable cost: at a
learning rate of 0.1 an extra tree changes nothing, and the model is truncated
at the recorded round.

**Why not option G, the whole fit submitted with no wait at all.** It would work
and it is tempting. Two reasons not to make it the default. First, the model has
to be readable during the fit for callbacks, checkpointing and progress
reporting, and a fit with no wait in it defers all of that to the end. Second, I
do not know whether macOS applies a watchdog timeout to a long-running compute
submission on an integrated GPU with no display attached to it; on some
configurations Apple does terminate long GPU work. That is an honest unknown
(section 8) and it is not worth taking the risk for the difference between 0.01
and 0 waits per tree.

### 2.5 The megakernel, and exactly what breaks

The maximal version of this design is one dispatch per round: a persistent
kernel holding the whole tree loop, with a software grid barrier between the
histogram phase and the scan phase, spinning on an atomic counter in device
memory.

It would work if all participating threadgroups were guaranteed to be resident
at once. Metal gives no such guarantee. Threadgroups of a dispatch are
scheduled as resources free up, and a threadgroup that has reached the barrier
and is spinning is holding a core's resources while waiting for a threadgroup
that the scheduler has not started and now cannot start. That is a deadlock, and
on a compute-only submission it is a hang rather than an error.

The standard mitigation is to launch no more threadgroups than can be
simultaneously resident. That requires knowing the maximum resident threadgroup
count, and there is no way to get it here. The three attributes that answer
runtime queries on Metal are `MULTIPROCESSOR_COUNT`, `MAX_THREADS_PER_BLOCK` and
`MAX_SHARED_MEMORY_PER_BLOCK`; there is no occupancy or blocks-per-multiprocessor
attribute, and `WARP_SIZE` is rejected outright on this backend. Nor could the
residency be *measured*, because Instruments exposes exactly one GPU counter on
this M4 and it is `RT Unit Active`. A safety margin would have to be guessed and
it would be guessed against a hang.

The API surface says the same thing from the other direction. There is no
cooperative-launch argument on `enqueue_function`; the semaphore and named-
barrier primitives that would implement a grid barrier are documented as
NVIDIA-only; and the only grid-wide synchronization available on this backend is
the kernel boundary itself, which the in-order queue provides for free. The
platform is telling you to express phases as separate dispatches, and the design
in 2.2 does.

There is also a cost even when it works. A histogram kernel wants a large grid
sized by the data; a persistent kernel is capped at the residency limit, so the
histogram phase would run at whatever occupancy the barrier permits rather than
whatever occupancy the histogram wants, and the two are not the same number.

**Verdict: no.** The measured payoff is small anyway. A megakernel saves the
inter-kernel gaps, 187 at 3.33 microseconds, which is 0.62 ms per round out of
a target round of 4 ms. That is 15 percent, bought with an unbounded and
unmeasurable hang risk. The work-queue pattern of section 2.2 delivers the part
of the megakernel that matters, which is data-dependent sizing without host
involvement, and it is deadlock-free by construction.

### 2.6 The next bottleneck, which the design creates and must then answer

Removing the waits does not remove the enqueues. Every dispatch is a command
buffer and the measured unblocked commit interval, host side, is 14.58
microseconds. At 157 dispatches:

```
host enqueue cost per round = 157 * 14.58 us = 2.29 ms
over 100 rounds                              = 229 ms
```

That is fine against today's 35.8 millisecond round and fine against the
8 millisecond round the design reaches at 255 bins. It is **not** comfortably
fine against the 3.5 to 5.2 millisecond round of section 3.6 at 63 bins, where
it is 45 to 65 percent of the round. The host would become the pacer, the device
would idle behind it, and the trainer would be back to being limited by
something other than arithmetic.

I want to flag this clearly rather than let it be discovered at the end of stage
three, because it is a consequence of the design and not an accident. Three
answers, in the order I would try them:

1. **Fewer, larger dispatches.** The 30 split iterations are 150 of the 157. The
   histogram partial-plus-reduce pair collapses to one dispatch whenever a node
   needs only one tile, which is true for most of the tail of a leaf-wise tree;
   that removes roughly 15 dispatches per round for free. The
   partition's flag-scan and scatter can collapse to one dispatch if the
   permutation is double-buffered and the scan is done over a fixed small number
   of chunks whose sums one threadgroup reduces, at the cost of a serial pass
   over the chunk sums; that removes another 30. A round of about 110 dispatches
   is 1.60 ms of enqueue, which fits.
2. **Enqueue on a dedicated thread, running ahead.** The enqueue loop reads
   nothing from the device, so it can run arbitrarily far ahead of the GPU,
   bounded only by how many command buffers the driver will hold. Whether that
   bound is generous on this driver, I do not know.
3. **Capture and replay**, which is the next section.

### 2.7 Command graphs, which may or may not exist on this backend

> **Answered 2026-08-16 and the answer is no on Metal.** The ten-minute test
> below was run. `DeviceGraph.create` raises
> `createGraphBuilder() not supported on this device context` on an M4 under
> MAX 26.5.0, at builder creation, before a single node is added; the driver
> ships `CUDADeviceGraphBuilder.cpp` and `HIPDeviceGraphBuilder.cpp` and no
> Metal equivalent. `docs/GPU_PORTABILITY.md` section 6.5 carries the
> evidence and the recovered API surface, and
> `tools/probe_device_graph.mojo` reruns the check on any device. The
> section is kept as written because its reasoning about *why* a graph would
> have suited this design is still correct and is what a CUDA or HIP port
> should read first -- with one correction, which is that the frozen surface
> is larger than this section assumed: there is no node-update call, so
> `grid_dim` is frozen at capture too, and our frontier-sized grids are the
> part of the design that does not fit.

The brief states that there is no command-graph API in the portable surface. I
found one documented: `max.gpu.host.device_graph`, with `DeviceGraph.create`,
`DeviceGraphBuilder`, and `graph.replay()` described as "lower overhead than
re-enqueueing each operation individually". The builder offers
`recording_context()`, a `DeviceContext` view on which ordinary `enqueue_*`
calls record into the graph rather than execute, so existing launch code works
unmodified.

I am flagging the discrepancy rather than resolving it, because I could not
verify the claim that matters. The installed toolchain ships the standard
library as compiled packages with no readable source, so everything above is
from the published stable documentation and not from this build; and the API is
modeled on CUDA graphs, with nothing in the documentation stating that Metal
backs it. **Whether `DeviceGraph.create` succeeds on this M4 is unknown and it
is a ten-minute test.**

If it works it is close to ideal for this design, for a reason that is not
obvious. A graph is a fixed DAG captured once and replayed, which is exactly
what property two of section 2.2 constructs: a dispatch schedule that does not
depend on data. The design does not need graphs, but the design happens to be
the only shape a graph could capture, and the payoff is that the per-round host
cost of section 2.6 collapses to one replay call. One further detail is worth
noting as confirmation rather than obstacle: host-visible waits raise inside a
recording context. That is precisely the discipline the design imposes anyway.

If it does not work on Metal, nothing above changes; answers one and two of
section 2.6 stand on their own.

### 2.8 What this costs in accuracy

Nothing. Every decision this control plane moves to the device is a decision the
host makes today from data the device produced. The split argmax, the shape
rules, the monotone clamp, the leaf value, the growth-policy pick: all of them
are deterministic functions of the histogram and the parameters. Section 4
covers the one genuine numerical difference, which is that the argmax now runs
in Float32 on a device with no Float64, and gives the fix.

---

## 3. The data plane

### 3.1 The measurement, and what it already implies

At 200,000 rows the longest kernel in each round has a median duration of
574.8 microseconds. `docs/METAL_TIMELINE.md` prices the root histogram pass at
14 bytes of load-level traffic per row per threadgroup block, with 25 blocks
(50 features at a Metal feature-group baseline of 2), for 70 MB and an apparent
121.8 GB/s against a machine rated at about 120.

That reading was left open in the timeline document as "somewhere between 21.6
and roughly 120 GB/s". I think it can be closed, and closed in a way that names
the fix.

Decompose the 14 bytes: a 4-byte row index, an 8-byte interleaved fixed-point
gradient/hessian pair, and two 1-byte bins. Twelve of the fourteen bytes are
per-row quantities that do not depend on the feature, and they are re-read once
per feature-pair block, which is 25 times. The ideal traffic for a root pass
that reads each row's gradient once and each of its 50 bins once is
`4 + 8 + 50 = 62` bytes, or 58 if the row index is elided at the root where the
permutation is the identity.

```
actual load-level traffic per row-visit    25 * 14 = 350 bytes
ideal                                                 58 bytes
amplification                                        6.03x
```

Now the same ratio from the clock:

```
ideal root pass at 200,000 rows = 200,000 * 58 B = 11.6 MB
11.6 MB at 120 GB/s             = 96.7 us
measured                        = 574.8 us
ratio                           = 5.94x
```

Two routes, 6.03 and 5.94. **The root histogram is running at close to six
times its ideal traffic and it is achieving close to the machine's rated
bandwidth while doing so.** That is as strong a statement as this device
permits that the kernel is bandwidth-bound, and it says the bandwidth being
consumed is redundancy rather than data.

This is inference, not a counter reading, and it rests on the 120 GB/s figure
and on the source-derived grid. But the two routes are independent of each
other and they agree to 1.5 percent.

### 3.2 The gather, which is the larger half

The measurement above is the root, where all rows are active and the
permutation is the identity, so every read is sequential. Below the root it is
not, and this is where most of the 21.7 ms per round at 1,000,000 rows lives.

Bins are stored feature-major, one byte per cell, at `bins[f * n_rows + r]`.
A node holding a fraction `p` of the rows reads, per feature, one byte every
`1/p` bytes down that feature's column. Write `L` for the GPU cache line size.
Bytes actually fetched per useful bin byte are `min(1/p, L)`.

For a balanced 31-leaf tree with sibling subtraction, building only the smaller
child at each split, the per-level row-visit counts are about 1,000,000 at the
root and 500,000 at each of five levels below it. Per row-visit the traffic is
`25 * (4 + 8) = 300` bytes of index and gradient re-reads plus
`50 * min(1/p, L)` bytes of bins.

| level | p | bin bytes per feature | per row-visit | row-visits | traffic |
| --- | --- | --- | --- | --- | --- |
| root | 1 | 1 | 350 B | 1.0 M | 350 MB |
| 1 | 0.5 | 2 | 400 B | 0.5 M | 200 MB |
| 2 | 0.25 | 4 | 500 B | 0.5 M | 250 MB |
| 3 | 0.125 | 8 | 700 B | 0.5 M | 350 MB |
| 4 | 0.0625 | 16 | 1100 B | 0.5 M | 550 MB |
| 5 | 0.031 | 32 | 1900 B | 0.5 M | 950 MB |
| | | | | | **2650 MB per tree** |

At 120 GB/s that is **22.1 milliseconds per tree**.

Section 1.5 derived, from an entirely different route (extrapolating measured
compute across two shapes), **21.7 milliseconds per round** of GPU compute at
1,000,000 rows.

These agree to two percent. I did not expect them to and I want to be careful
about how much weight that carries, because the model has four assumptions in
it: a balanced tree, 120 GB/s achieved, `L` at least 64, and no cache reuse
across the strided reads. But note that `L` drops out entirely: every stride in
the table is 32 bytes or less, so `min(1/p, L)` equals `1/p` for any cache line
of 64 bytes or more, and the model is insensitive to a quantity I do not know.

The conclusion I draw, and I mark it as inference:

> **The histogram at 1,000,000 rows is bandwidth-bound, and roughly 87 percent
> of the bandwidth it consumes is layout redundancy: the feature-major gather
> below the root, and the per-feature-block re-read of the row index and the
> gradient pair. The ideal traffic for the same tree is about 3,000,000
> row-visits at 58 bytes, or 174 MB, against 2,650 MB.**

### 3.3 The layout I would build

**Blocked row-major bins.** Group features into stripes of `W` features and
store, for each stripe `b`, a contiguous array of `n_rows` records of `W` bytes:
`bins[b][r][0..W)`. Within a stripe a row's bins are contiguous. Across stripes
they are far apart.

`W` is not a free parameter. It is set by threadgroup memory, because the whole
point is that one pass over a row serves `W` features, which requires `W`
features' histograms to be resident in threadgroup memory at once.

```
W = floor(threadgroup_memory_bytes / (n_bins * bytes_per_cell))
```

The figures below use 32 KiB per threadgroup, which is the documented Apple
family figure. It should not be hardcoded: `MAX_SHARED_MEMORY_PER_BLOCK` is one
of the three device attributes that answer on Metal, and the repo already queries
it, so `W` should be computed at startup from the real number. If the real
number is larger, every `W` in this section grows proportionally and the 255-bin
case improves with no accuracy question attached.

**The accumulator cell is 8 bytes: one Int32 gradient plane and one Int32 count
plane.** Section 4.2 explains why 8 bytes is a floor and why 4-byte and 2-byte
cells are not reachable under the scale scheme that determinism requires.
Squared error has a constant hessian, so the hessian plane is the count plane
scaled by a constant and does not need its own accumulator. A non-constant
hessian costs a third plane and takes the cell to 12 bytes.

| bins | cell | W | feature blocks at 50 features |
| --- | --- | --- | --- |
| 256 | 8 B | 16 | 4 |
| 256 | 12 B (general hessian) | 10 | 5 |
| 64 | 8 B | 64 | 1 |
| 64 | 12 B | 42 | 2 |
| 16 | 8 B | 256 | 1, with room for 4 replicas |

The 255-bin case and the low-cardinality case are structurally different
problems and this table is why. At 64 bins a 50-feature dataset is a
**one-block** problem: the row's whole 50-byte bin stripe is read once, the
gradient is read once, and the amplification is 1.0. At 256 bins it is a
four-block problem and the amplification floor is
`(4 * (4 + 4) + 4 * 16) / 58 = 1.66`.

**Padding.** Pad the stripe record to a power of two so a gathered stripe read
never straddles two cache lines: 16 bytes at `W = 16`, 64 bytes at `W = 50`
(from a 64-feature stripe budget). Padding wastes DRAM capacity, which is free,
to save DRAM traffic, which is not.

**Do not physically reorder the bin matrix.** The obvious alternative to
gathering is to compact the bins so each node's rows are contiguous. At
1,000,000 by 50 the matrix is 50 MB, so one compaction is 100 MB of read plus
write, and keeping contiguity requires one compaction per level, five per tree,
for 500 MB. The gather it would eliminate costs, under the blocked row-major
layout of the next table, about 750 MB. Trading 750 for 500 plus a still-needed
288 MB of sequential reads is a loss. The ratio does not improve with more
features, because the matrix and the gather grow together. This is settled by
arithmetic and I would not spend a benchmark on it.

### 3.4 What the layout buys, level by level

Same tree, same row-visit counts. Under blocked row-major with `W = 16` at 256
bins, per row-visit the traffic is `4 * (4 + 4)` bytes of index and gradient
re-reads plus `4 * min(W/p, L)` bytes of stripe. Here `L` does *not* drop out,
because the stripe stride `W/p` exceeds 64 bytes from level 2 down. Both values
are given.

| level | p | stripe stride | per row-visit, L=64 | per row-visit, L=128 |
| --- | --- | --- | --- | --- |
| root | 1 | 16 B | 96 B | 96 B |
| 1 | 0.5 | 32 B | 160 B | 160 B |
| 2 | 0.25 | 64 B | 288 B | 288 B |
| 3 | 0.125 | 128 B | 288 B | 544 B |
| 4 | 0.0625 | 256 B | 288 B | 544 B |
| 5 | 0.031 | 512 B | 288 B | 544 B |

```
total, L=64   = 1.0M*96 + 0.5M*(160+288+288+288+288) =  752 MB per tree ->  6.3 ms
total, L=128  = 1.0M*96 + 0.5M*(160+288+544+544+544) = 1136 MB per tree ->  9.5 ms
today                                                = 2650 MB per tree -> 22.1 ms
```

**3.5x at a 64-byte line, 2.3x at 128.** The cache line size is the single
largest lever I cannot measure and section 8 names the experiment.

At 64 bins with `W = 50` in a 64-byte padded stripe, one block:

```
root  : 64 (stripe) + 4 (grad)                      =  68 B
levels: min(64/p, L) + 4  = 68 B (L=64) or 132 B (L=128)
total, L=64   = 1.0M*68 + 0.5M*4*68  = 204 MB -> 1.7 ms per tree
total, L=128  = 1.0M*68 + 0.5M*4*132 = 332 MB -> 2.8 ms per tree
```

**13x to 8x against today.** The 63-bin case is not a small optimization of
the 255-bin case; it is a different regime, because it is the regime in which
the whole feature set fits in one threadgroup's histogram and every read is
one cache line serving every feature. This is why it deserves its own place in
the sequence in section 7, and why it is third rather than first: it is the
only one of the three changes that can fail on accuracy.

### 3.5 The kernel shape

**One threadgroup owns (one row tile, one feature stripe, one node).** 256
threads. The tile length is chosen so that the per-tile Int32 accumulation
cannot overflow, which section 4.2 shows is not binding under the sum-based
scale, and so that the number of tiles is enough to fill 10 cores several times
over without producing so many partials that the reduce becomes expensive.

At 1,000,000 rows with 4 feature stripes, a tile of 32,768 rows gives 31 tiles
by 4 stripes, 124 threadgroups. On 10 cores that is 12 per core. At 63 bins with
one stripe it gives 31 threadgroups, which is 3 per core and probably too few;
there the tile should shrink to 8,192 rows for 122 threadgroups. The tile length
is a comptime-specialized parameter chosen from a small table keyed on
`(n_rows, n_stripes)`, and it is the one place I would put a tuning knob.

**Two kernels, partial and reduce, not one atomic kernel.** Each threadgroup
writes its tile's histogram to its own slot in device memory with plain stores
and no atomics; a reduce kernel sums the slots. The reason is not performance,
it is section 4.3: a plain-store partial plus a deterministic reduce is
order-independent by construction, and it also allows the cross-tile sum to be
carried in Int64 while the per-tile accumulation stays Int32 in threadgroup
memory, which buys precision headroom for free.

Partial slab size at 1,000,000 rows and 256 bins: 31 tiles by 50 features by
256 bins by 8 bytes = 3.2 MB for the root, less at every level below. Summed
over a tree, about 4.7 MB written and read, which is 0.08 ms. Negligible
against 750 MB.

**Comptime specialization over a small closed set, not runtime policy.** The
threadgroup histogram is allocated with `stack_allocation`, whose size is a
compile-time parameter, and dynamic threadgroup memory is not verified on this
backend. So `(BIN_CAP, W, plane count)` are comptime parameters and the kernel
is instantiated once per combination the library supports. The set is small:
`BIN_CAP` in {32, 64, 128, 256}, plane count in {2, 3}, and `W` determined by
the other two, so eight instantiations cover everything. `MAX_SHARED_MEMORY_PER_BLOCK`
answers on Metal, so the host reads the real budget at startup and picks the row
of the table rather than assuming 32 KiB. This is the whole of what replaces the
policy layer thrown away in section 6.

**Threadgroup-memory atomics for the accumulation, integer only.** Each thread
processes rows from its tile, and for each of the `W` features in its stripe
does one atomic add to the gradient plane and one to the count plane. That is
`2 * W` shared atomics per row. At `W = 16` and 3,000,000 row-visits per tree
across 4 stripes, that is 3,000,000 * 50 * 2 = 300 million shared atomic adds
per tree. Whether that rate is achievable on an Apple GPU I do not know
(section 8); it is 300M / 6.3ms = 48 G-atomics per second across 10 cores,
which is 4.8 G per core, which is roughly one per cycle per core at 4.8 GHz and
therefore certainly not achievable per *thread* but plausibly achievable across
256 threads per group hitting a banked 32 KiB scratchpad. If it is not, the
mitigation is bin-range replication of the shared histogram, which costs
threadgroup memory and therefore costs `W`, and the trade is measurable.

One small piece of good news here. The atomics API defaults `fetch_add` to
relaxed memory ordering on Apple GPU targets and sequential ordering elsewhere,
so the default is already the cheap one on this device and nothing needs to
weaken it by hand.

**Sibling subtraction stays.** It halves the row-visits below the root, which
is the single largest algorithmic saving available, and under integer
accumulation with a per-round scale it is exact: the parent's bins are the
exact integer sum of the two children's. Fuse the subtraction into the reduce
kernel's store, as an add of a negated slot, so it costs no dispatch.

**Elide the row index at the root.** At the root the permutation is the
identity, so `perm[i] == i` and the 4-byte index read is pure waste. A comptime
`IDENTITY_PERM` specialization of the histogram kernel removes it. That is 4 of
the 96 bytes at the root, or 4 MB per tree. Small, but it is free.

### 3.6 The rebuilt roofline

Per tree at 1,000,000 by 50, 31 leaves, blocked row-major, 63 bins:

```
histogram                    204 to 332 MB      1.7 to 2.8 ms
partition (see below)         60 to 130 MB      0.5 to 1.1 ms
gradient + score update        20 MB            0.17 ms
histogram partials + scan      12 MB            0.10 ms
dispatch gaps (157 * 3.33us)                    0.52 ms
small/no-op kernel floor                        0.40 ms
                                              --------------
                                              3.4 to 5.1 ms per round, device
                                              0.34 to 0.51 s per 100 rounds
```

That is the device side. Section 2.6 shows the host side at 157 dispatches is
2.29 ms per round, which at this point is 45 to 65 percent of the device round
and therefore uncomfortably close to becoming the pacer. The dispatch-count
reductions in 2.6 answer it, and a working `DeviceGraph` would answer it
completely. This is the one place where the control plane and the data plane are
coupled: making the device fast enough turns the enqueue rate into a real number.

The stated roofline is 0.4 seconds. My reconstruction is 0.35 to 0.52. I take
that as agreement and I would not defend either figure to better than a factor
of 1.5.

The partition term. A split rewrites only the parent's window of the row
permutation, so per tree the partition touches `sum over splits of parent rows`,
about 5,000,000 row slots for a balanced 31-leaf tree. Per slot: read the index
(4), read the split feature's bin (a single-feature gather, `min(1/p, L)`
bytes, averaging perhaps 10), write the destination (4). If the scatter writes
into a second permutation buffer and the range table is updated to point at it,
rather than scattering to scratch and copying back, the copy-back disappears
and the term is 5,000,000 * 18 = 90 MB. Double-buffering the permutation costs
4 MB of device memory and removes a third of the partition traffic and one
dispatch per split; that is an easy trade.

At 255 bins the histogram term becomes 6.3 to 9.5 ms and the round is 7.9 to
11.6 ms, or 0.79 to 1.16 seconds per 100 rounds. Against LightGBM's 2.86 that
is still 2.5x to 3.6x.

### 3.7 Row-major versus column-major, stated plainly

The brief asks the question directly, so here is the answer separated from the
arithmetic that produced it.

**Column-major (feature-major) is right if and only if every pass is
sequential.** It is the correct layout for binning, for a full-dataset scan, and
for the root histogram, and it is the layout that makes a single-feature
operation cheap. It is catastrophic for a gathered pass, because each feature is
its own stream and a row's 50 bins are 50 separate cache lines. The measured
consequence is the 32 bytes per useful byte at level 5 in section 3.2.

**Row-major is right if and only if the whole row is consumed.** A pure
row-major layout is not usable here, because consuming a whole row means holding
50 features' histograms in threadgroup memory, which is 100 KiB at 256 bins
against 32 KiB available.

**Blocked row-major is the layout that respects both constraints**, and the
block width is not a design choice but a quotient: threadgroup memory divided by
the per-feature histogram size. That is why the bin count is a first-class
performance parameter in this design rather than a quality parameter with a
performance side effect.

**Bin-blocking is worse than feature-blocking and I would not build it.** The
alternative to blocking by feature is blocking by bin range: hold all 50
features but only 16 bins each, read the row's whole 50-byte stripe, and skip
bins outside the range. Threadgroup memory is then 50 * 16 * 8 = 6.4 KiB, and
the row's stripe is one cache line. But it requires 256/16 = 16 passes over the
rows, so it multiplies the dominant term by 16 to divide a subordinate one. The
arithmetic is not close.

### 3.8 Packing gradient and hessian into one atomic

The brief raises this and the answer is no, for a reason that is arithmetic
rather than taste.

Squared error needs, per bin, a gradient sum and a count. Packing them into one
32-bit word means accumulating `w = (q << C) + 1` so that the low `C` bits
carry the count and the rest carry the gradient sum. The count field must not
overflow into the gradient field, so a tile of `T` rows needs `C >= log2(T)`.
With `T = 4096` that is `C = 12`, leaving 20 signed bits for a sum of up to
4096 quantized gradients, so each `|q| < 2^19/4096 = 128`, which is seven bits
of gradient resolution per row. Section 4.2 shows the shipped scheme uses 30
bits of headroom for exactly this reason. Seven bits will not hold parity and I
would not spend a benchmark finding out.

Two Int32 planes, 8 bytes per cell, is the floor. What *is* worth doing is the
thing the shipped code already does: elide the hessian plane whenever the
objective declares a constant hessian, and reconstruct it at flush time as
`h_const * count`. That takes the cell from 12 bytes to 8 and `W` from 10 to 16,
which is a 1.6x on the block count and therefore on the index and gradient
re-reads.

---

## 4. Numerics, which is the constraint the design has to survive

Accuracy is the hard constraint. The measured position, from
`bench/real_data/results/20260815T190351Z-h2h/`, is four to five parts in ten
thousand against LightGBM on four of six real datasets, three parts in a
thousand on the noisiest (average precision on a rare class), and six parts in a
million on rcv1. That is the bar. Nothing in this design may move it.

### 4.1 What must be preserved exactly

These are not negotiable and they are not where the performance is, so
preserving them is cheap:

- The gain formula, including LightGBM's `ThresholdL1` soft threshold, with
  `lambda_l2` in the denominator and no additional epsilon.
- The per-candidate guards in LightGBM's order and placement, and the
  per-feature costs applied once after that feature's best candidate is known.
- The leaf value `-T(G)/(H + lambda_l2)`, unshrunk in storage, with
  `max_delta_step` capping and `path_smooth` applied in LightGBM's order.
- The binning rule: quantile edges at midpoints of adjacent *distinct* values,
  one bin per distinct value when the distinct count fits the budget, one
  reserved missing bin above the ordinary bins for any feature with a training
  NaN.
- The default-direction rule: every threshold evaluated twice, missing-left
  scored first so exact ties keep `default_left`.
- The strictly-positive gain bar and the ascending-scan tie-break.

None of these interact with the layout or the control plane. They live in the
scan and commit kernels, which are 2.4 percent of the round.

### 4.2 The fixed-point scheme, which I would keep unchanged

The shipped scheme sets `scale = 2^30 / sum|v|` over the whole dataset, per
round, per plane. I would not change it, and it is worth writing down why,
because I started this design intending to replace it with a per-node scale and
that would have been a mistake.

- **The bound is structural, not statistical.** Because the scale is set from
  the *total* magnitude sum, any node's rows are a subset of the whole, so no
  partial sum can exceed `2^30` regardless of tree shape, node size, tile
  length or accumulation order. There is no overflow analysis to redo when the
  kernel shape changes. The only ceiling is that rounding adds at most half a
  unit per row, giving `2^30 + n/2`, which fits Int32 up to about 2.1 billion
  rows.
- **Per-round-per-dataset is what makes sibling subtraction exact.** Parent and
  both children share one scale, so the parent's bins are the exact integer sum
  of the children's and subtraction introduces no error at all. A per-node scale
  would break that, and sibling subtraction is the largest algorithmic saving in
  the data plane. This is the mistake I nearly made.
- **It is why the histogram is order-independent**, which section 4.3 needs.
- **Resolution is about `2^-30` of the round's total gradient magnitude**, which
  is finer than Float32 carries at these magnitudes. The scheme is not a
  compression; it is an exactness device that happens to cost nothing.

The one change I would make is in the *reduce*, not the accumulation: carry the
cross-tile sum in Int64 rather than Int32. It costs nothing (the partials are
already separate slots and the reduce is 0.08 ms per tree) and it removes the
`2^30 + n/2` ceiling from the design entirely, which matters at 100 million rows
and does not matter at one million. Per-tile accumulation stays Int32 in
threadgroup memory because Metal has no 64-bit threadgroup atomics.

### 4.3 Determinism as a design property, not a hope

The design is bit-reproducible run to run on fixed hardware, and this falls out
of choices already made rather than needing its own machinery:

- Histogram accumulation is exact integer arithmetic, so it is independent of
  atomic ordering, threadgroup scheduling, tile count and grid geometry.
- The partial-plus-reduce shape uses plain stores in the partial and a
  deterministic tree reduction across a fixed slot order, so even the reduce
  does not depend on completion order.
- The dispatch schedule is fixed and does not depend on data, so the geometry is
  a function of the shape alone.
- The work queue changes *which* threadgroup does *which* tile between runs.
  Under integer accumulation into a fixed slot indexed by tile id rather than by
  threadgroup id, that has no effect on any value. This is a real constraint on
  the work-queue design and it must be written into the kernel: **a work item
  carries its output slot index, it does not derive it from the threadgroup
  id.**
- Tie-breaks are total and structural: gain descending, then feature ascending,
  then bin ascending, then node id ascending. Not "first to finish".

### 4.4 The one genuine numerical risk: Float32 gains on a device with no Float64

Apple GPUs have no double precision. Moving the split argmax onto the device,
which section 2 requires, means computing gains in Float32 where LightGBM and
our own CPU backend compute them in Float64. Gains have order `1e-7` relative
error in Float32, and candidates whose true gains differ by less than that can
be ordered differently.

The metric consequence is probably nil, because two candidates with gains equal
to seven significant figures are two nearly equally good splits and picking
either is nearly equally good. But "probably nil" is not what a hard accuracy
constraint accepts, and there is a cheap way not to rely on it.

**Compute the gain in emulated double-single arithmetic.** Represent each
intermediate as an unevaluated sum of two Float32 values and use the standard
error-free transformations (`two_sum`, `two_product`) for the four operations
the gain needs. This gives roughly 48 bits of significand, more than enough to
reproduce a Float64 argmax on everything but exact ties, and exact ties are
resolved by the structural tie-break in 4.3 anyway. The cost is about 4x on a
kernel that scans 50 features by 256 bins, or 12,800 cells, and that kernel is
2.4 percent of the round today. Four times 2.4 percent of a round that is about
to get nine times shorter is still under a tenth of the new round. It is cheap
insurance and I would take it.

**This forces `--fp-mode contract=off` for the whole compilation.** Double-single
arithmetic depends on the compiler *not* fusing `a*b` and the subsequent
subtraction that recovers the rounding error; a contracted FMA silently
destroys the error term and the emulation returns garbage that looks plausible.
The flag is whole-compilation granularity with no per-function override, so this
is a global decision.

I would take it, and I would take it independently of the double-single
question, for three reasons. First, the data plane is integer arithmetic;
contraction has nothing to fuse in the histogram, so `contract=off` costs
approximately nothing where the time is. Second, `docs/NUMERICS.md` records
three separate incidents in which contraction moved the model, two of them found
only after the fact and one found by audit, and each time the fix was to
restructure a source line so the compiler would make a particular choice. That
is an unstable equilibrium and a clean sheet should not start on it. Third, the
mandate explicitly discards bit-identity with the existing output, which is the
only thing that made `contract=fast` load-bearing: the golden fixtures were
produced under it.

A practical note that would otherwise cost someone a day: `--fp-mode` is
accepted by `mojo build` and `mojo run` and not by `mojo package` or
`mojo precompile`, so a design that ships a `.mojopkg` needs the flag applied at
the site that compiles the kernels, and the build has to be structured so that
site exists.

### 4.5 What the design does *not* do to accuracy

To be explicit, since the point of the section is that the fast design is the
accurate one:

- Growth policy is unchanged. Leaf-wise, one split committed per iteration,
  same priority rule, same shape rules, same stopping. Section 5 argues this is
  also the right performance choice once the round trips are gone.
- Binning is unchanged, at whatever bin count is chosen. The bin count is the
  one parameter this design has an interest in and section 7 puts it last for
  exactly that reason.
- Histogram values are unchanged, to the bit, since the quantization and the
  scale are unchanged and integer accumulation is order-independent.
- Sibling subtraction is unchanged and remains exact.
- The only value that changes is the gain, and it changes toward LightGBM
  rather than away from it, because double-single is closer to Float64 than the
  Float32 the device split search uses today.

---

## 5. The algorithm: is leaf-wise right for a GPU?

### 5.1 The question restated

Leaf-wise growth is a dependent chain: you cannot know which leaf to split next
until you have the histograms of the leaves you just created. Thirty-one such
steps per tree. On a machine where each step costs a 606 microsecond host round
trip, that chain is the whole cost, and the pressure to replace it with
something wider is enormous. That pressure is why level-wise, oblivious trees
and speculative expansion are all on the table.

**Once the round trips are gone, the pressure mostly goes with them.** This is
the central algorithmic claim of the design and it is worth stating as a
conditional: the case for changing the growth policy is a case about
synchronization cost, and if the synchronization cost is zero the case has to be
remade on different grounds.

### 5.2 What leaf-wise costs when a step is free

The dependent chain does not disappear; it just gets cheap. Per split iteration
the chain is five dispatches (scan and commit, two partition, two histogram),
and the dependency is expressed by submission order on an in-order queue, which
costs the measured 3.33 microsecond median gap. Thirty splits at five dispatches
is 150 gaps, 0.5 ms per round.

Row traffic is *not* worse than level-wise. This surprises people and it is
worth the arithmetic. With sibling subtraction, leaf-wise builds one histogram
per committed split, over the smaller child's rows, so the total row-visits per
tree are `n_rows` at the root plus the sum over splits of `min(n_left,
n_right)`, which for a balanced tree is `n_rows * depth / 2`. A level-wise
grower building only smaller siblings has exactly the same sum. The two policies
read the same number of rows.

What leaf-wise costs is **shape**, in two ways:

1. **The small-node tail.** The last several splits of a 31-leaf tree operate on
   nodes of a few thousand rows or fewer, which cannot fill 10 GPU cores. The
   measured evidence for this is the 1.293 ms per round intercept in section
   1.5: compute has a floor that does not scale with rows, and that floor is the
   tail. At 1,000,000 rows it is 6 percent of compute and irrelevant; at 50,000
   rows it is 56 percent of compute and it is most of why the GPU loses that
   shape.
2. **The gather.** Leaf-wise addresses a node's rows through a permutation, so
   every read below the root is strided. Level-wise with a per-row node id can
   read every row sequentially. Section 5.4 prices this.

### 5.3 Level-wise, priced honestly

Level-wise growth allows a genuinely different data plane: keep rows in their
original order, maintain a `node_id[row]` array, and build every node's
histogram at a level in **one sequential pass over all rows**, with each row
atomically accumulating into its own node's slot. No permutation, no partition
kernel, no gather.

```
traffic per level = n_rows * (2 node_id + 4 gradient + 50 bins) = 56 B per row
5 levels at 1,000,000 rows                                      = 280 MB per tree
                                                                = 2.33 ms
```

Against blocked row-major leaf-wise at 6.3 to 9.5 ms per tree, that is 2.7x to
4.1x better, and it is 5 sequential passes against 3,000,000 strided row-visits.
It is the better data plane and I want to say so plainly rather than bury it.

There is a catch and it is large. Multi-node histograms cannot live in
threadgroup memory: a level of 32 nodes at 50 features by 256 bins by 8 bytes is
3.2 MB. So the accumulation must use **device-memory atomics**, and the traffic
that generates is `n_rows * n_features * 2` atomic operations per level, which
is 100 million atomics per level and 500 million per tree. If those are serviced
by the system level cache, where a 3.2 MB slab plausibly lives, the design is
excellent. If they reach DRAM they are `500e6 * 4 bytes = 2 GB` per tree and the
design is four times worse than leaf-wise, not three times better.

**I do not know the M4's system level cache size, nor whether Metal device
atomics are serviced there, and there is no counter on this machine that could
tell me.** Section 8 names the microbenchmark. This is the single largest
unresolved fork in the design.

### 5.4 What level-wise costs in accuracy, which is the reason it is not the default

A level-wise tree and a leaf-wise tree at the same `num_leaves` are different
models. LightGBM's entire premise is that spending a fixed leaf budget where the
gain is, rather than spreading it evenly, reduces training loss faster per leaf.
At matched `num_leaves`, level-wise should fit worse per tree. Matching quality
requires retuning depth, leaf count, learning rate and estimator count together,
and the honest comparison is a time-to-matched-quality curve, not a
seconds-per-100-trees number.

The accuracy constraint in this brief is parity with LightGBM. LightGBM grows
leaf-wise. Adopting level-wise as the default means the parity claim has to be
re-established from scratch on every dataset, under a retuned configuration, and
it will not be a parity claim at matched parameters any more. That is a large
bill for a data-plane improvement, and the data-plane improvement is contingent
on a cache question I cannot answer.

**Decision: leaf-wise is the default and level-wise is a measured mode.** Build
level-wise second, behind the same device-resident control plane, and let the
time-to-quality curves decide whether it earns a place. It is a genuinely
promising second product; it is not a way to make this one faster without
changing it.

### 5.5 Oblivious / symmetric trees

CatBoost's GPU speed comes substantially from oblivious trees: one split
condition per level, shared by every node at that level. The GPU consequences
are excellent. The node id becomes a `depth`-bit path index appended one bit per
level, so there is no partition at all and no permutation. There are exactly
`depth` histogram passes per tree. The split search is one argmax over
`features * bins` of the summed-over-nodes gain, which is a single small
reduction.

It is the fastest tree to grow on this hardware and I believe that without
needing a measurement. It is also the largest model change of the three: an
oblivious tree of depth 6 is a much weaker learner than a 31-leaf leaf-wise
tree, CatBoost compensates with a different regularization regime and a
different default estimator count, and no configuration of an oblivious learner
reproduces a LightGBM model.

**Not in this design.** It is a different library. Worth saying out loud that if
the mandate were "fastest trainer to a fixed held-out metric, model shape
unconstrained" rather than "fastest trainer at parity with LightGBM", oblivious
trees would deserve a serious look and this document would be shorter.

### 5.6 Speculative and batched leaf expansion

Both are answers to a problem this design has already removed, and I would build
neither.

**Batched expansion** builds the histograms of several frontier leaves in one
pass. Under leaf-wise growth there are only ever two leaves needing a histogram
at any moment, the children of the split just committed, and those two share the
parent's rows, so building them together is one pass over the parent's window
and is what sibling subtraction already does more cheaply. There is no batch to
form without changing which splits get taken.

**Speculative expansion** builds histograms for leaves that might be split next
and discards the ones that are not. It preserves the model exactly, since the
speculation only ever pre-computes something that would have been computed
later or not at all. Its payoff is amortizing launch overhead across the
speculation, and launch overhead after section 2 is 3.33 microseconds per
dispatch. Paying wasted bandwidth, which is the dominant cost, to save
dispatches, which are not, is the wrong trade. The measured evidence points the
same way: batching seven multiclass trees into one launch was indistinguishable
from not batching, 15.45 against 15.30 seconds, because per-dispatch cost is not
what the round spends its time on.

### 5.7 One algorithmic change I would make

Not a growth policy change, but worth naming because it addresses the small-node
tail directly and does not change the model.

**Switch the histogram kernel shape by node size, at a comptime-specialized
threshold.** Large nodes want many tiles and few features per threadgroup;
small nodes want one tile, all features, and as much replication as threadgroup
memory allows, because at 3,000 rows the problem is filling the machine and not
bandwidth. Below some row count, the entire node fits in a single threadgroup's
scratchpad with the histogram accumulated in registers per thread and reduced
once. The two shapes are the same arithmetic, so the model does not move, and
the crossover is a measurable constant on this device rather than a heuristic.

This matters most at exactly the shapes the GPU currently loses: at 50,000 rows
the small-node floor is 56 percent of compute.

---

## 6. What I would throw away

Structures, not files, and the reason in each case is that the structure exists
to manage a cost the clean design does not have.

**1. The host-side frontier.** Today the node-to-row-range map, the frontier
state, the split records and the tree itself are host data structures, and the
device is handed scalars. That inverts the ownership the machine wants. In the
clean design the tree is a device-resident struct and the host holds nothing
during a fit except the enqueue loop counter. The host-side copy exists only as
a download at the end. This is the single change everything else in section 2
follows from.

**2. Every restaging path.** Copying node tables, feature tables, allow masks
and float parameters to the device before each search launch is four host-to-
device copies per launch group whose entire purpose is to tell the device
something the device computed. If kernel parameters live in device memory, the
restaging disappears along with the pinned staging buffers, the epoch
invalidation, and the synchronize-before-overwrite that staging requires.

**3. The hybrid CPU/GPU leaf scheduler and its cost model.** It exists because a
device leaf costs a 606 microsecond round trip and a host leaf does not. Remove
the round trip and the premise is gone. It is also the most expensive structure
in the codebase to keep honest, because it requires a host replica of the
fixed-point histogram that is bit-identical to the device one, which is a
second implementation of the hottest code in the system whose correctness is a
standing obligation. A clean design has one histogram.

**4. The backend policy layer.** Device policy, vendor policy, unified-memory
policy, per-vendor histogram policy, portability shims, layout planners that
nothing calls: roughly nine thousand lines of runtime decision-making about
which kernel shape to run. A Metal-specific design replaces all of it with a
small closed set of comptime-specialized kernels and a table keyed on
`(n_bins, cell_bytes, node_size_class)`. The mandate explicitly permits a
Metal-specific design and this is the largest thing that permission buys.

**5. The dual histogram strategies, `atomic` and `tiled`, and the resolver that
chooses between them.** Two kernel families, a tiling cost model, and a
selection rule. The clean design has one family, partial-plus-reduce, because
the reduce is 0.08 ms per tree and buying determinism and Int64 headroom for
0.08 ms is not a decision that needs a resolver.

**6. Feature-major bins on the training path**, for the reasons in section 3.2
and 3.7. Keep feature-major for binning, which is a per-feature sequential
operation and is already 3.2x faster than LightGBM's. Convert once, at upload,
into the blocked row-major training layout. The conversion is one pass over
50 MB and costs about 1 ms per fit.

**7. The host-side split search on the GPU path.** It exists to make split
decisions match the CPU exactly, and it costs a 300 KB download per split. It
is the right call under today's numerics and the wrong structure: the correct
way to make the device argmax match is to make the device arithmetic good
enough, which section 4.4 does for about 4x on 2.4 percent of a round.

**8. The scatter-to-scratch-and-copy-back partition.** Double-buffer the
permutation instead. It removes a dispatch per split and a third of the
partition traffic, and the only thing it costs is 4 MB.

**9. Phase-fencing instrumentation as a first-class mode.** The fenced profile
costs 1.39 seconds of a 4.90 second run to measure a 3.52 second one, and its
central conclusion, that the histogram is 49.3 percent of the round, was
overturned by the trace the same day because a phase profile cannot see time in
which no phase is running. A design that never blocks has no phases to fence.
Replace it with dispatch-level labels, which is a `pushDebugGroup` at the launch
site, and read the timeline. The timeline document is right that this would make
every future capture dramatically more useful for a very small cost, and on a
device with one usable counter it is the only instrument there is.

**10. The 4-bin quantized gradient lattice, the sparse GPU layout planner, the
speculation ledger, the unused frontier batching structures.** Implemented and
unreachable. A clean sheet does not carry unreached code, and the specific harm
here is that each one is a plausible-looking answer to a question a future
reader will ask, which costs them a day to discover is not wired up.

---

## 7. If I could build only three things

In order. Each one is independently shippable and each one is measurable on its
own.

### First: the device-resident round

Zero blocking waits per round. Every kernel parameter in device memory. A fixed
dispatch schedule with no-op guards on dead iterations. Persistent-grid work
queues for data-dependent sizing. Pipelined submission so the host never blocks
in steady state, with early stopping read one round late from an
already-completed buffer.

**What it buys:** the 1.47 seconds of non-overlapped host serialization, less
the roughly 0.1 second a pipelined submit still costs. **3.58 s -> about
2.1 s**, which beats LightGBM's 2.86 on this machine at this shape by 1.4x.

**Why first:** it is the largest single term, it is the best-evidenced number in
the document (two independent instruments agreeing to four percent), it changes
no arithmetic and therefore cannot move the metric, and every subsequent
optimization is measured through a round that is no longer 40 percent wait. It
also fixes the shape where the GPU currently loses worst: the fixed cost is a
constant per round, so at 50,000 rows it is 88 percent of the run rather than 40
percent of it.

**How it can fail:** if the enqueue cost cannot be hidden. 157 dispatches at the
measured 14.58 microsecond unblocked commit interval is 2.29 ms of host CPU per
round, against a device round that after this change is about 22 ms and after
the second change about 8. It fits at both, with 9x and 3.5x margin. It stops
fitting at stage three, and section 2.6 is the plan for that. The right move is
to instrument the enqueue rate from the first day of this stage, so the margin
is a measured number and not an estimate by the time it matters.

### Second: the blocked row-major bin layout and the wide histogram kernel

Convert bins at upload into stripes of `W = floor(32768 / (n_bins * 8))`
features. One threadgroup per (row tile, feature stripe, node), 8-byte cells
(Int32 gradient, Int32 count), plain-store partials, Int64 reduce with fused
sibling subtraction, comptime identity-permutation specialization at the root,
and a small-node kernel shape below a measured crossover.

**What it buys:** the histogram's traffic amplification, from 2,650 MB per tree
to 752 or 1,136 MB depending on the cache line size. **About 2.1 s -> 0.8 to
1.2 s**, which is 2.4x to 3.6x LightGBM.

**Why second:** it is the larger of the two remaining terms and it is the one
whose analysis is best supported, because a first-principles traffic model built
from the layout lands within 2 percent of an independently extrapolated
measurement. It also changes no values: the quantization, the scale, the
subtraction and the accumulation are all unchanged, so the model is bit-
identical to the first stage's.

**How it can fail:** the cache line question. If the GPU line is 128 bytes
rather than 64, a 16-byte stripe wastes 8x on a gathered read below level 2 and
the win is 2.3x rather than 3.5x. It still wins; it wins less. If the line is
larger still, or if the strided prefetcher behaves worse than the model assumes,
the win could be smaller again. This is the thing in this document I am least
sure of.

### Third: the bin count, decided by measurement

Measure held-out metric at 255, 127, 63 and 31 bins on the six real datasets
already in the harness, at matched everything else, against LightGBM at 255.
Adopt the smallest bin count that holds the current parity band, and make it the
default only if it does.

**What it buys, if 63 holds:** `W` goes from 16 to 64, a 50-feature dataset
becomes a one-block problem, the amplification floor goes from 1.66 to 1.0, and
every gathered read becomes one cache line serving every feature. Histogram
traffic from 752 or 1,136 MB per tree to 204 or 332. **About 0.8 to 1.2 s ->
0.34 to 0.51 s**, at or slightly better than the stated roofline, and 5.6x to
8.4x LightGBM.

**Why third:** it is the only one of the three that can fail on the hard
constraint, and it is the only one whose payoff is contingent on something
outside the machine. It is also the only one that has to be measured rather than
derived, and putting it last means it is measured on a trainer where the effect
is visible rather than buried under a 40 percent wait.

**How it can fail, two ways.** It may simply not hold parity, in which case the
answer is 255 bins and 0.8 to 1.2 seconds, which is still a very good trainer.
There is a partial fallback if it fails globally but holds per dataset: the bin
count is already a per-fit parameter and the layout adapts to it automatically
through `W`, so a user who can afford 63 bins gets the fast regime without a
separate code path. Note also that the binning level rule already gives one bin
per distinct value under budget, so low-cardinality features are *already* in
the fast regime today, and the measured 9.3x on low-cardinality binning is the
same phenomenon showing up one stage earlier.

The second failure is not about accuracy at all: at a 3.4 to 5.1 millisecond
device round, the host's 2.29 milliseconds of enqueue per round stops being
hideable and the trainer becomes host-paced. Section 2.6 lists the three
answers. This is the reason the `DeviceGraph` test in section 8 should be run
first even though it belongs to stage one: if graphs work on Metal, stage three
lands where the arithmetic says; if they do not, stage three needs the dispatch
schedule trimmed to about 110 before it is worth attempting.

### Where that lands

| stage | 1,000,000 x 50, 100 rounds | vs LightGBM 2.86 s |
| --- | --- | --- |
| today | 3.58 s | 0.80x |
| after the device-resident round | ~2.1 s | 1.4x |
| after the layout, 255 bins | 0.8 to 1.2 s | 2.4x to 3.6x |
| after the bin count, if 63 holds | 0.34 to 0.51 s | 5.6x to 8.4x |

Every number in the right two columns is derived rather than measured and each
one carries the assumptions of the section that produced it. I would expect the
first row of the projection to hold within 10 percent, the second within 30, and
the third within a factor of 1.5.

---

## 8. What I do not know

Each of these is a real gap, each one is named because a confident guess would
cost more to disprove than it is worth, and each one has an experiment attached
that is cheaper than the decision it would inform.

**1. The GPU cache line size on the M4, and the efficiency of a strided
16-byte gather.** This sets whether the second stage buys 3.5x or 2.3x, and it
is the largest uncertainty in the document. *Experiment:* a synthetic histogram
kernel over a permutation with a tunable stride, run at fixed clock state,
reporting effective bytes per row-visit from wall clock. Half a day, no library
change.

**2. Whether Metal device-memory atomics into a few-megabyte slab are serviced
by the system level cache, and at what rate.** This decides the level-wise fork
in section 5.3 entirely: it is the difference between level-wise being 3x better
than leaf-wise on the data plane and 4x worse. I also do not know the M4's
system level cache size. *Experiment:* a kernel doing `n` atomic adds into a
slab of tunable size, sweeping the size across the plausible cache range and
looking for the knee.

**3. The rate of threadgroup-memory integer atomics on an Apple GPU core.** The
design needs about 48 billion shared atomic adds per second across 10 cores to
hit the projected histogram time. I believe that is plausible against a banked
32 KiB scratchpad with 256 threads per group, and I have no measurement.
*Experiment:* a kernel that does nothing but shared atomic adds into a
controlled bin distribution, sweeping the collision rate.

**4. The cost of a no-op dispatch with a small persistent grid.** The fixed
dispatch schedule depends on a dead iteration being nearly free. I assumed 2 to
5 microseconds from the shape of the measured duration distribution. *Experiment:*
enqueue 1,000 trivially-returning kernels with no wait between them and read the
timeline.

**5. Whether `DeviceGraph` is backed on Metal.** Section 2.7. If it is, the
enqueue-rate problem of section 2.6 disappears and stage three's roofline becomes
reachable without contorting the dispatch schedule. If it is not, nothing breaks
but the schedule has to be trimmed. *Experiment:* call `DeviceGraph.create` with
a two-kernel builder and see whether it raises. Ten minutes, and it should be
the very first thing anyone does.

**6. How far ahead of the device the driver will let the host enqueue.** This
decides whether answer two of section 2.6 works. *Experiment:* submit 10,000
trivial kernels without waiting and measure where the commit call starts
blocking.

**7. Whether macOS applies a watchdog timeout to a long compute submission on an
integrated GPU.** This is why the design does not submit a whole fit without a
wait. *Experiment:* submit a deliberately long compute stream and see whether it
is killed.

**8. Whether 63 bins holds held-out parity.** Not knowable without running it,
and section 7 makes it the third stage for that reason.

Three things I listed as unknowns while drafting and then found answers for, kept
here so nobody re-opens them. The threadgroup memory limit is queryable:
`MAX_SHARED_MEMORY_PER_BLOCK` is one of exactly three `DeviceAttribute` values
that answer on Metal, so `W` should be computed from the real number at startup
rather than from the documented 32 KiB. Float32 atomic add does not exist on
Metal, which is why the fixed-point scheme is not optional. And 64-bit atomics
are unavailable or unverified in both memory spaces, which the design does not
need: the Int64 cross-tile reduce of section 4.2 is a plain-store reduction over
distinct slots, not an atomic accumulation.

Two things I want to flag as *not* unknowns, because they have been treated as
open questions and I think section 3 closes them:

- Whether the histogram is latency-bound or bandwidth-bound. Two independent
  routes put the root pass at six times ideal traffic while achieving close to
  the machine's rated bandwidth, and a first-principles traffic model over the
  whole tree lands within 2 percent of independently extrapolated compute time.
  It is bandwidth-bound, and the bandwidth is redundancy.
- Whether occupancy is the problem. The current root grid is 50 threadgroups of
  256 threads at 6 KiB of threadgroup memory, which is 5 groups and 30 KiB per
  core. That is not obviously starved, and the traffic model explains the
  measured time without invoking occupancy at all. If the traffic model is right
  there is no occupancy deficit left to find.

One measurement discipline that the design has to carry rather than solve. The
device moved between Minimum and Maximum performance state across captures of
the same binary, which is the most plausible mechanism for the two-to-threefold
benchmark drift this machine shows, and it means an interleaved comparison can
still be invalid if the two arms straddle a clock transition. Every measurement
above and every measurement proposed here must print its performance state
breakdown beside its durations. There is a plausible and unproven second-order
effect worth watching: a trainer that keeps the GPU busy 90 percent of the time
rather than 24 may hold the clock at Maximum on its own, which would make the
design look better than its arithmetic predicts and would also make it easier to
measure.

---

## 9. What generalizes, and what does not

The design above is for dense numeric features, squared error, single output.
Here is what the rest costs.

**Non-constant hessian** (logistic, Poisson, Tweedie, and everything else). The
hessian needs its own Int32 plane, so the cell goes from 8 bytes to 12 and `W`
from 16 to 10 at 256 bins, or from 64 to 42 at 64 bins. At 256 bins that is 5
feature blocks instead of 4, so the index and gradient re-read term grows by
25 percent and the total by about 8 percent. At 64 bins with 50 features it is
still one block and costs nothing at all. Everything else is unchanged. The
gradient fill kernel is a per-objective specialization and nothing else in the
design knows which objective it is.

**Multiclass.** K independent trees per round over one shared bin matrix. This
gets *better* under the device-resident control plane rather than merely
surviving: the K trees have no dependency on each other within a round, so their
dispatches interleave in the submission stream and fill the device where a
single tree's small-node tail would not. The measured 1.63x GPU win on 7-class
covertype is against a control plane that pays 32 round trips per round per
class. The scales are already per class per round.

**Ranking.** Only the gradient computation changes, and it changes on the
device. LightGBM reads its pairwise sigmoid from a lookup table, which is a
device-resident constant array. No structural consequence.

**Categorical features.** The histogram is unchanged: a categorical feature is
still a bin index. What changes is the scan, which sorts bins by
`sum_g / sum_h` and searches contiguous prefixes rather than thresholds, and the
partition, which routes by 256-bit set membership rather than by comparison. The
scan is 2.4 percent of the round and the routing descriptor is 56 bytes per
node. Both fit in the device-resident record without changing its shape. This
generalizes cleanly.

**Missing values.** Already in the layout: a reserved bin above the ordinary
bins, evaluated twice per threshold. No consequence for anything in sections 2
or 3.

**Bagging, GOSS, feature sampling.** All are expressible as a per-row active
flag and a per-node feature mask, both device-resident. The grid stays fixed
because it is sized by the full row count and inactive rows exit early. This is
one of the places where the fixed-grid discipline pays for itself: a sampling
scheme that changed the grid would reintroduce a host dependency.

**Sparse features and exclusive feature bundling.** Out of scope and genuinely
different. EFB changes what a "feature" is at bin-layout time, which is upstream
of everything in section 3, and a sparse layout wants a different histogram
kernel entirely. The clean design should not pretend to cover it; it should have
a clean seam at the bin layout so a second layout can be added without touching
the control plane.

**Other GPU vendors.** The mandate permits a Metal-specific design and this one
takes that permission in three places: the 32 KiB threadgroup budget that sets
`W`, the absence of Float64 that forces double-single gains, and the absence of
64-bit threadgroup atomics that forces Int32 per-tile accumulation. All three are
comptime-parameterized quantities rather than hardcoded ones, so a CUDA port
would change three constants and gain a fourth option (warp-level primitives,
which Mojo does not expose here in any case). The control plane of section 2 is
not Metal-specific at all: it would be better on a platform with indirect
dispatch, because the fixed dispatch schedule and the no-op guards exist only to
work around its absence.

---

## 10. The platform facts this design rests on

Collected here because a design is only as good as the capability claims under
it, and because the verification status of these differs a lot. "Repo-verified"
means the shipped code exercises it on this M4 and it works. "Docs" means the
published stable documentation says so and I could not check it against this
build, because the installed toolchain ships the standard library as compiled
packages with no readable source.

| fact the design uses | status |
| --- | --- |
| the queue is in order and `enqueue_*` is asynchronous; N kernels, one `synchronize()` | repo-verified; the whole hazard model already rests on it |
| `grid_dim` and `block_dim` accept runtime host integers | docs; the repo passes computed dims |
| threadgroup int32 atomic add works in threadgroup and device memory | repo-verified, this is what ships today |
| there is no float atomic add on Metal | repo-verified, and the reason the fixed-point scheme exists |
| `barrier()` is a threadgroup execution barrier and memory fence | repo-verified |
| `stack_allocation` with a comptime size gives threadgroup memory | repo-verified |
| `MULTIPROCESSOR_COUNT`, `MAX_THREADS_PER_BLOCK`, `MAX_SHARED_MEMORY_PER_BLOCK` answer on Metal | repo-verified; `WARP_SIZE` is rejected |
| `block.prefix_sum` / `sum` / `max` / `min` / `broadcast` with a comptime `block_size` | repo-verified |
| Apple GPUs have no Float64 | repo-verified, and the reason for section 4.4 |
| no indirect dispatch | docs, by exhaustive absence rather than by a statement |
| no cooperative launch, no device-wide barrier; semaphores and named barriers are NVIDIA-only | docs |
| `--fp-mode contract` is whole-compilation, on `build`/`run` only | repo-verified in `docs/NUMERICS.md` |
| `max_single_alloc_size()` is `maxBufferLength` on Metal | docs, and it is the one explicitly Metal-aware method |
| `DeviceGraph` exists and Metal backs it | **unverified, and section 8 item 5** |
| 64-bit atomics on Metal | unverified; the design does not need them |
| dynamic threadgroup memory (`shared_mem_bytes`) on Metal | unverified; the design uses comptime sizes instead |
| events, streams, `select_stream` on Metal | unverified; the repo uses none of them, and `select_stream` is documented to return a view equivalent to the context on backends with no multi-stream model |

Two refinements to the platform facts as the brief states them, offered as
corrections to make rather than as disagreements.

**Warp primitives do exist, just not where the brief looked.** There is no
`max.gpu.primitives.warp`, which is what the brief says. There is
`std.gpu.primitives.warp`, with `sum`, `reduce`, `prefix_sum`, `broadcast`, the
four shuffles, `vote` and the lane-group variants, and the portability guide
names Apple explicitly: `syncwarp()` on Apple silicon provides execution
synchronization within a SIMD group with no memory fence, and lane masks are
ignored. The block collectives are built on it. This does not change the design,
because the design's hot loop is atomics into threadgroup memory rather than
cross-lane reduction, but it does mean the split-scan kernel of section 2.4 has
a shuffle-based reduction available if the block collective turns out slow. The
caveat is that the runtime `WARP_SIZE` query is rejected on Metal, so a
simdgroup width of 32 would have to be a compile-time assumption rather than a
queried fact.

**Command graphs may exist.** Section 2.7. I have not verified it and the design
does not require it.

## 11. Summary of the claims, by confidence

**Highest confidence.** Roughly 1.5 seconds of the 3.58, or 14.7 milliseconds
per round, is non-overlapped host serialization that no arithmetic requires.
Two independent instruments agree on it to four percent. Removing it is a
control-plane change that touches no arithmetic and cannot move the metric, and
it alone puts this trainer ahead of LightGBM at 1,000,000 by 50 on this machine.

**High confidence.** The histogram is bandwidth-bound and roughly six-sevenths
of the bandwidth it consumes is layout redundancy. A first-principles traffic
model over the whole tree lands within 2 percent of an independently
extrapolated measurement, and the root-pass amplification is confirmed twice, at
6.03x from the byte accounting and 5.94x from the clock.

**Moderate confidence.** Blocked row-major bins with a threadgroup-memory-sized
feature stripe reduce that traffic by 3.5x at a 64-byte cache line and 2.3x at
128. The mechanism is certain; the magnitude depends on a cache line size I
cannot measure.

**A consequence of the design that the design then has to answer.** With the
waits gone, the host's enqueue rate becomes the next thing in the way: 157
command buffers per round at the measured 14.58 microsecond unblocked commit
interval is 2.29 milliseconds of host CPU per round, which is comfortable
against an 8 millisecond round and marginal against a 3.5 millisecond one. This
is not a reason not to do it; it is a reason to instrument the enqueue rate from
day one and to test whether `DeviceGraph` works on Metal before stage three
rather than after.

**Deliberate judgment rather than measurement.** Leaf-wise growth stays,
because the case for replacing it was a case about synchronization cost and the
synchronization cost is about to be zero, and because every alternative
(level-wise, oblivious) changes the model and therefore reopens the accuracy
claim that is the hard constraint here.

**Lowest confidence, and the thing I would want measured before anyone commits
to the second stage.** That a gathered 16-byte stripe read costs one cache line
rather than several, and that the strided access pattern behaves the way the
model assumes. If it does not, the second stage buys 1.2x instead of 3.5x, and
the whole projection past stage one collapses to something like 1.7 seconds
rather than 0.8. The experiment in section 8 item 1 is half a day and it should
be run before the layout work starts, not after.
