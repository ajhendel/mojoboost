# The Metal timeline

This is the first GPU timeline this project has ever had. Everything the
repository has said about GPU bubbles, occupancy, and whether the histogram
kernel is latency-bound or bandwidth-bound has until now been arithmetic over
wall-clock totals. The stage-level profile recorded in
`bench/results/profile_2026-08-15/` was a real advance over that, but it still
measures phases by bracketing them on the host. It can say that the histogram
phase is 49.3 percent of a round. It cannot say whether the GPU was executing
anything during that phase, and it cannot see a single gap.

A Metal System Trace can. This document records how one was taken, what it
says against the three questions the round was blocked on, and, at least as
importantly, what it cannot say and why.

The short version. The GPU is idle for 76 percent of a training run at 200,000
rows and for 88 percent at 50,000. Almost all of that idleness is the host
blocking on readbacks: 32 of them per boosting round, each costing about 0.6
milliseconds of wall clock, of which about 4 microseconds is the GPU actually
moving bytes. The round is bound by round-trip latency on a synchronization
pattern, and the stage profile's "transfer is 28.2 percent of the round" is not
a statement about data movement at all, because the GPU spends 0.65 percent of
the span moving bytes.

Whether the histogram kernel itself is latency-bound or bandwidth-bound is
still not known, and section 5 explains why it cannot be known on this device:
the Apple M4 exposes Instruments exactly one GPU counter and it is about the
raytracing unit. What can be said is that the question matters less than it
appeared to, because compute of every kind is 22.9 percent of a round.

## 1. Method

### 1.1 What was tried, in order

The obvious path on macOS is Instruments, and the obvious problem is that Mojo
is not an Xcode project and produces no signed application bundle. That turned
out not to matter. `xctrace record` will launch and trace an arbitrary
executable, and on this machine it did so without a privacy prompt, without
root, and without any entitlement on the target.

Four things were tried. Only the first was needed.

1. `xctrace record --template 'Metal System Trace' --launch`. This works. It
   is the whole of the method and everything in this document comes from it.

2. `--instrument 'Metal GPU Counters'` added to the same command. This is
   worse than useless and is documented here so nobody repeats it. The run
   completes but Instruments reports `GPU Service reported error: Selected
   counter profile is not supported on target device`, and the resulting trace
   contains zero rows in every counter table, where the plain template had
   contained 730,967. Section 3 explains what the default template does and
   does not give you.

3. `MTL_CAPTURE_ENABLED=1` with a programmatic capture scope. This exists and
   is reachable. The installed MAX toolchain ships
   `max/mojo/max/gpu/host/_metal_capture.mojo` with
   `_start_metal_trace_capture(ctx, path)` and `_end_metal_trace_capture(ctx)`
   over the exported C symbols `_AsyncRT_DeviceContext_startMetalTraceCapture`
   and `_AsyncRT_DeviceContext_stopMetalTraceCapture`, and the runtime carries
   the string `Metal GPU trace capture is unavailable; set MTL_CAPTURE_ENABLED=1
   before launching the process`. It was not used, for two reasons. It writes a
   `.gputrace` that only the Xcode Metal debugger opens, so nothing can be
   extracted from it by script, and it would require adding a call site to the
   library, which this lane is not permitted to do. It is the right tool for
   a future question about what happens inside one kernel, and the wrong tool
   for a question about what happens between kernels.

4. `powermetrics` sampling GPU residency. Not run. It needs root, this
   repository's `bench/apple/thermal_capture.sh` deliberately refuses to issue
   privileged commands on a reader's behalf, and the question it would have
   answered coarsely ("is the GPU busy or idle") is answered exactly by the
   trace. It remains available as an independent check if anyone doubts
   section 4.

The toolchain search also turned up `MODULAR_ENABLE_PROFILING`,
`MODULAR_ENABLE_GPU_PROFILING`, and `MODULAR_PROFILE_FILENAME`. The first two
are compile-time defines rather than environment variables for Mojo, and the
GPU one drives an NVTX and ROCTx bridge with no Metal backend, so neither
would have produced a Metal timeline. `MODULAR_PROFILER_PLUGIN` wants a
plugin dylib that this environment does not ship. `DeviceContext.execution_time`
exists but the only device timer implementations present in the shipped
binaries are the CUDA and HIP ones. None of this is a path to a Metal
timeline today.

### 1.2 The exact commands

The binary is built ahead of the capture rather than traced through
`mojo run`, because `mojo run` compiles in the same process and would put
several seconds of a Mojo compile at the head of the trace.

```sh
pixi run --manifest-path pixi.toml \
    mojo precompile -I ./src ./src/mojotrees -o ./build/mojotrees.mojopkg

pixi run --manifest-path pixi.toml \
    mojo build -I ./build bench/bench_train_gpu.mojo -o ./build/bench_train_gpu

xcrun xctrace record \
    --template 'Metal System Trace' \
    --output ./metal.trace \
    --no-prompt --target-stdout - \
    --launch -- ./build/bench_train_gpu 200000 50 reg 1 gpu

python3 bench/apple/metal_timeline.py ./metal.trace \
    --process bench_train_gpu --rounds 100
```

`bench/apple/metal_capture.sh` is those four commands with argument checking,
a `--self-check`, and a `--print-plan` that prints them without running
anything. The reader, `bench/apple/metal_timeline.py`, pulls the tables out
with `xctrace export --xpath` and reduces them. It imports the standard
library only, so a trace can be read on a machine with no pixi environment.

### 1.3 Conditions

Machine: MacBook Air, Apple M4, 10 GPU cores, 11.84 GiB recommended maximum
working set. macOS 26.5.2 (25F84). Xcode 26.6 (17F113), Instruments 16.0,
AGX tracecode version 3.44.6.

Workload: `bench_train_gpu 200000 50 reg 1 gpu`, which is 200,000 rows by 50
features, squared error, one repeat, the `SPLIT_SEARCH_AUTO` GPU arm, 100
boosting rounds at the library defaults. The trace is 119 MB and covers 4.11
seconds of process life, of which the training span is 2.35 seconds.

Four captures were taken in all, at 50,000, 100,000, 200,000, and 400,000
rows. This document is written from the 200,000 row one. Section 5.4 explains
which of the others may be compared to it and which may not, and the reduced
output for all four is in `bench/results/metal_timeline_2026-08-15/`.

Not an idle machine by the standard of
`docs/APPLE_GPU_BENCHMARK_PROTOCOL.md`. Chrome, Code, Discord, Docker, and
Zoom were resident. That matters much less here than it would for a
throughput number, because everything below is a within-run structural
measurement rather than a comparison of two wall clocks, and because the
trace records what every other process did on the GPU and it is almost
nothing: 125 dispatches from the window server and a browser helper against
22,109 from the benchmark.

### 1.4 What tracing costs

`gpu_train_s` was 2.070 seconds untraced and 2.363 seconds under the trace, so
Instruments adds about 14 percent. The GPU-side timestamps come from the GPU's
own clock and are not distorted by that. The CPU-side ones are measured on an
instrumented process and should be read as upper bounds. Every conclusion
below is stated so that the 14 percent cannot reverse it, and where it could
have, that is said.

## 2. What the trace contains

The template records 80-odd tables. Five carry the argument.

`metal-gpu-intervals` has one row per encoder that ran on the GPU, with the
GPU's own start timestamp and duration. This is the only table that says what
the hardware was doing. `metal-application-command-buffer-submissions` has one
row per `commit` on the CPU, with the duration of the commit call.
`metal-command-buffer-completed` has one row per completion notification
delivered back to the process. Joining the three on the command buffer id
gives a complete lifecycle for 22,107 command buffers: when the host started
enqueuing, when it finished, when the GPU began, when the GPU ended, and when
the process was told. `metal-gpu-state-intervals` is the GPU hardware's own
Active and Idle record for the whole machine, which is an independent
cross-check on the per-process numbers. `gpu-performance-state-intervals` is
the clock state, which turns out to be the reason two captures of the same
binary disagree.

Two format details are load-bearing and undocumented by Apple. The XML export
interns repeated values, so an element carries `id="N"` the first time a value
appears anywhere in the document and `ref="N"` on every later appearance, and
a reader that does not resolve those gets empty strings for most of the file.
And a row is a flat sequence of direct child elements, one per schema column,
in the order the `<schema>` block declares them, with no names on the row
itself.

The run submitted 22,107 command buffers carrying work plus 3,225 carrying
none, and every one of them holds exactly one encoder. There is no batching of
encoders into command buffers anywhere in the GPU path.

## 3. What the trace cannot contain

**GPU counters.** This is the single largest limitation and it is a property
of the device, not of the method. The Apple M4 in this machine offers
Instruments exactly one GPU counter:

```
GPU counters available on this device
  RT Unit Active
```

That is the percentage of GPU cores on which the raytracing unit is active. It
is worthless to a histogram kernel. There is no occupancy counter, no ALU
utilization counter, no memory bandwidth counter, no cache hit rate, and no
register pressure. Asking for the `Metal GPU Counters` instrument by name does
not unlock others, it removes the one that was there. Consequently no kernel
in this repository can today be called latency-bound or bandwidth-bound on the
basis of a measurement. Section 5 says what can be inferred instead, and marks
it as inference.

**Kernel names.** MAX sets no labels and pushes no debug groups on its Metal
encoders, so every compute dispatch in the trace is called `Compute Command 0`
and every copy `Blit Command 0`. There are exactly two distinct labels across
22,109 dispatches. Which kernel ran is recoverable only by position and
duration. Every attribution of a dispatch to a named kernel below is therefore
inference, and is marked. If this becomes load-bearing, the fix is upstream in
MAX or a `pushDebugGroup` at the launch site, and it would make every future
capture dramatically more useful for a very small cost.

**Shader Timeline**, which would give per-line cost inside a kernel, is a
recording setting reachable from the Instruments UI and not from `xctrace`.
It is off in every trace taken here.

## 4. Question 1: are there bubbles, and how large

Answered. Yes, and they are much larger than the estimate.

The estimate under review was roughly 20 milliseconds of pipeline gap per 41
millisecond round, which is about 49 percent. The measurement, at 200,000
rows, is 76 percent.

```
dispatches                    22109  (18703 compute, 3406 blit)
span, first GPU start to last GPU end      2346.74 ms
GPU busy (union of intervals)               552.28 ms  (23.53% of span)
GPU idle inside the span                   1794.45 ms  (76.47% of span)
compute GPU time                            536.96 ms  (22.88% of span)
blit GPU time (bytes actually moving)        15.33 ms  ( 0.65% of span)
```

Busy is the union of the intervals, not the sum, because two encoders can
overlap on the GPU and summing would claim more busy time than the clock
allows. In this run they never do overlap, so the two agree, but the reader
computes the union anyway.

Two independent checks say this is real rather than an artifact of how the
per-process rows were filtered.

The GPU hardware's own Active and Idle record, which is not filtered by
process and therefore counts the window server too, reports 565.48 milliseconds
Active and 2793.48 milliseconds Idle across the whole 4.11 second trace. The
per-process union over the shorter training span was 552.28 milliseconds. The
13 millisecond difference is the other processes on the machine. Two different
instruments, recorded through different mechanisms, agree on the busy time to
2.4 percent.

The GPU was not merely downclocked. The performance state record says the
device sat at Maximum for 1862.41 milliseconds of the 2389.38 milliseconds of
recorded state time, which is 77.9 percent, with Minimum for 20.2 percent and
Medium for 1.9 percent. The idleness is genuine absence of work, not slow work.

### 4.1 The shape of the gaps

The gaps are not uniform. They are bimodal, and only one of the two modes
matters.

```
gaps between busy runs   n=22108  total=1794.45 ms  mean=81.17 us  med=3.33 us  p90=413.83 us
  <1us      n= 7451  total=    2.83 ms  ( 0.2% of all idle)
  1-5us     n= 6889  total=   22.67 ms  ( 1.3% of all idle)
  5-20us    n= 1573  total=    9.82 ms  ( 0.5% of all idle)
  20-100us  n= 1211  total=   96.77 ms  ( 5.4% of all idle)
  0.1-1ms   n= 4980  total= 1655.12 ms  (92.2% of all idle)
  >1ms      n=    4  total=    7.23 ms  ( 0.4% of all idle)
```

The median gap is 3.33 microseconds, which is nothing. Two thirds of all gaps
are under 5 microseconds and together they account for 1.5 percent of the idle
time. Back-to-back dispatch on this device is efficient and there is no case
for optimizing it.

Then there are 4,980 gaps between a tenth of a millisecond and a millisecond,
and they carry 92.2 percent of all the idle time. Section 6 identifies exactly
what they are.

### 4.2 Per round

Cutting the dispatch stream at each of the 100 longest kernels gives a clean
round structure. The cut points are spaced a median of 22.36 milliseconds
apart, and there are exactly as many of them as there are boosting rounds,
which is the evidence that the cut is in the right place.

```
wall per round        median    22.36 ms
GPU busy per round    median     4.92 ms
dispatches per round  median      221
GPU busy fraction     median     22.9%  (min 15.0%, max 93.6%)
serialization points per round    32.1
```

221 dispatches per round is consistent with the accounting written down in
`src/mojotrees/train_gpu.mojo`, which prices leaf-wise growth at eight
launches and one wait per split, four for the row partition, two or three for
the histogram, and two for the split search, plus four per round outside the
tree for the gradient computation, the fixed-point scales, the quantization,
and the raw score update. Thirty splits at eight launches is 240, so 221
implies a tree that stopped a few leaves short of the 31-leaf budget, which
is ordinary.

### 4.3 The idleness gets worse as the shape gets smaller

The 50,000 row capture sat at Maximum performance state for 70.7 percent of
its recorded state time against the 200,000 row capture's 77.9 percent, which
is close enough that the two may be compared. They say this:

```
                    50,000 rows        200,000 rows
GPU busy               12.47%              23.53%
GPU idle               87.53%              76.47%
compute GPU time      231.20 ms           536.96 ms
blit GPU time          16.39 ms            15.33 ms
dispatches              22107               22107
```

Quartering the rows cut the compute by 2.3x and changed nothing else. The
dispatch count is identical to the last unit, the bytes moved are within 7
percent, and the fixed per-dispatch costs of section 6 are the same to within
a microsecond. The GPU is idle 87.5 percent of the time at 50,000 rows because
the round trips did not get any cheaper when the work did.

That the compute fell by 2.3x rather than by 4x is worth noting separately.
Row work scales with rows, but most of the 18,701 compute dispatches are at
nodes far too small to fill the device, and those cost what they cost whatever
the row count. The compute stream has a fixed floor of its own, underneath the
much larger fixed floor of the synchronizations.

This is the mechanism behind the observation already in the round's commit
history, that the GPU carries about 1.5 seconds of fixed cost per fit and
therefore loses to this library's own CPU below roughly a million rows. The
fixed cost is 32 blocking readbacks per round, and it is a constant.

### 4.4 The synchronization count

The 32.1 serialization points per round is the number to hold on to. The
stage-level profile counted 3,100 host synchronizations over 100 trees and
called it 31 per tree. This trace, through a completely different instrument
that knows nothing about the library's own counters, counts 3,208 and calls it
32.1 per tree. Those are the same number. The two measurements confirm each
other.

## 5. Question 2: is the histogram kernel latency-bound or bandwidth-bound

**Not answered by measurement, and it cannot be on this device.** There is no
occupancy counter and no bandwidth counter to read. What follows is what the
timeline plus the source can support, clearly separated into the two.

### 5.1 What was measured

The longest kernel in each round has a median duration of 574.8 microseconds,
with a minimum of 549.0 and a maximum of 1884.0 across the 100 rounds, and the
100 of them are spaced a median of 22.36 milliseconds apart, which is the round
period.

Compute time is concentrated in a small number of large dispatches:

```
where compute GPU time goes, by kernel duration class
  <2us       n=     1  total=   0.00 ms  (  0.0%)
  2-5us      n=  7005  total=  22.55 ms  (  4.2%)
  5-10us     n=  3201  total=  22.52 ms  (  4.2%)
  10-25us    n=  4975  total=  74.75 ms  ( 13.9%)
  25-50us    n=   592  total=  21.32 ms  (  4.0%)
  50-100us   n=  1392  total=  96.15 ms  ( 17.9%)
  100-250us  n=  1334  total= 185.82 ms  ( 34.6%)
  250-500us  n=    73  total=  22.23 ms  (  4.1%)
  0.5-1ms    n=   115  total=  66.74 ms  ( 12.4%)
  >1ms       n=    15  total=  24.87 ms  (  4.6%)
```

Half the compute dispatches are shorter than 8.21 microseconds and together
those contribute under 9 percent of compute time. Whatever is worth optimizing
in a kernel is in the top decile.

### 5.2 What was inferred, from the source rather than the trace

The claim under review was that at the root the histogram launches about 50
threadgroups with about 2,000 serial rows per thread. Reading the launch site
rather than the trace, because the trace cannot name a kernel:

The grid is `(blocks, n_tiles)` where `blocks = ceil(n_slots / GROUP)`, set in
`src/mojotrees/gpu_active_rows.mojo`. On Metal the feature group baseline is 2
rather than the 1 used on CUDA and HIP, so with 50 features `blocks` is 25.
The tiling resolves to 2 row tiles at this shape. The grid is therefore
(25, 2), which is 50 threadgroups of 256 threads, or 12,800 lanes.

So the count of 50 is right and the structure behind it is not. It is 25
feature-pair blocks by 2 row tiles, not one block per feature. On CUDA and HIP
the same expression does give one block per feature.

The serial row count is wrong by a factor of five. The row loop strides by
`block_dim.x` over the block's own tile, and `gpu_tiling.rows_per_thread` is
`ceil(rows_per_tile / block_threads)`, which is `ceil(100000 / 256)`, or **391
rows per thread**, not 2,000.

The histogram is a two-kernel operation on this path, a partial kernel
followed by a reduce, with a conditional third zeroing kernel. Threadgroup
memory is `group * bin_cap * 12` bytes, which is 6 KiB per threadgroup here.

### 5.3 What can and cannot be concluded from those two together

On occupancy. 50 threadgroups over 10 GPU cores is 5 threadgroups per core,
or 1,280 threads per core, at 6 KiB of threadgroup memory each for 30 KiB per
core. Neither figure looks starved for an Apple GPU core. This is arithmetic
over a published core count and a grid read out of the source, not an
occupancy measurement, and it should not be quoted as one. It is enough to say
that the "only 50 threadgroups, therefore latency-starved" story is not
obviously true and should not be acted on without evidence.

On bandwidth, the arithmetic is more interesting and still cannot settle it.
Each of the 50 blocks walks 100,000 rows and reads, per row, a 4-byte row
index, an 8-byte interleaved fixed-point gradient and hessian pair, and one
1-byte bin for each of its 2 features. That is 14 bytes per row per block, so
70 MB of load-level traffic for a root pass. Against the measured 574.8
microseconds that is 121.8 GB/s, and this machine's rated memory bandwidth is
about 120 GB/s.

That number cannot be what it appears to be, because a kernel cannot pull 122
GB/s out of a 120 GB/s memory system. What it means is that a real fraction of
the row index and gradient re-reads are being served from cache, since the
unique footprint is only about 12.4 MB, which at 574.8 microseconds would be
21.6 GB/s. The true DRAM rate is somewhere between 21.6 and roughly 120 GB/s
and the trace cannot narrow it further. If the reuse is poor, the kernel is at
the memory wall and only a layout change helps. If the reuse is good, there is
headroom and unrolling or more tiles would help. **This is exactly the question
the counters would have answered and they are not available.**

### 5.4 The scaling experiment, and why it failed

The natural way around a missing counter is to vary the problem size and look
at the slope. Two more captures were taken, at 100,000 and 400,000 rows, and
the result is not usable. The reason is worth recording because it also
explains something this repository has been fighting for weeks.

```
capture           GPU performance state, share of recorded state time
 50,000 rows      Maximum 70.7%,  Minimum 27.0%,  Medium 2.3%
200,000 rows      Maximum 77.9%,  Minimum 20.2%,  Medium 1.9%
100,000 rows      Minimum 100%
400,000 rows      Minimum 100%
```

The 100,000 and 400,000 row captures ran entirely at the device's Minimum GPU
performance state. The 50,000 and 200,000 row captures ran mostly at Maximum,
which is why section 4.3 is allowed to compare those two and this section is
not allowed to use the other pair for a slope. Kernel
durations across that pair differ by the clock and not by the workload, and no
scaling claim can be made from them. The visible symptom is that the 100,000
row run had a longer root kernel, 766.6 microseconds, than the 200,000 row
run's 574.8, and a longer wall clock, 2.92 seconds against 2.36, which is
impossible on the workload and ordinary on the clock.

This is a mechanism for the drift recorded in
`docs/APPLE_GPU_BENCHMARK_PROTOCOL.md` and repeated across this repository's
benchmark results, where the same binary at the same shape reads two or three
times apart in different time windows. The trace names the cause. Every
capture should have its performance state printed before its durations are
compared to anything, and the reader does print it.

The busy and idle conclusion in section 4 survives this, and in the direction
that strengthens it. A lower clock lengthens kernels while leaving the host
round trips alone, so a Minimum-state capture overstates the busy fraction.
The 400,000 row capture reads 46.5 percent busy at Minimum, and at Maximum it
would read lower.

## 6. Question 3: what one enqueue costs and what one host wait costs

Answered, and the three figures that appeared to disagree by 3x turn out to be
measuring three different things, none of them wrong.

```
enqueue: the commit call on CPU   n=22107  mean=  14.86 us  med=  12.62 us  p90=  17.83 us
commit end -> GPU start           n=22107  mean= 216.48 us  med= 203.54 us  p90= 321.29 us
GPU busy per command buffer       n=22107  mean=  24.99 us  med=   6.25 us  p90=  65.33 us
GPU end -> completion signal      n=22107  mean= 110.24 us  med= 101.33 us  p90= 148.79 us
commit N -> commit N+1            n=22106  mean= 106.30 us  med=  14.58 us  p90= 532.38 us
```

These are device properties and not workload properties, which the captures
confirm rather than assume. Between the 50,000 and 200,000 row runs the median
enqueue moved from 12.67 to 12.62 microseconds and the median completion
notification from 96.92 to 101.33, while the compute time between them changed
by a factor of 2.3. `bench/README.md` was right to describe them as a property
of the device.

`bench/README.md` says "roughly 20us to submit a launch and 126us for the
wait". Both are right. The submit is measured here at 12.62 microseconds
median on an instrumented process, so 20 is a sound upper bound. The 126
microsecond wait is the completion notification latency, measured here at
101.33 microseconds median. The README's derived "about 280us a split" is also
right: the median serialized turnaround from the GPU finishing a readback to
the host committing the next command buffer is 285 microseconds.

`HybridCosts.launch_nanos`, at 62,128 nanoseconds, matches none of the
intervals in the table because it is not an interval. It is a fitted
coefficient from `bench_hybrid_costs`, documented in
`src/mojotrees/hybrid_leaf_scheduler.mojo` as the fixed cost of "enqueuing and
running one histogram kernel, exclusive of the row work", which is a
regression intercept over a two-kernel operation and not something the timeline
contains a matching pair of timestamps for. There was never a contradiction to
resolve. There was an undocumented difference in what the two numbers denote,
and it should be said in the docstring that this is a fitted intercept rather
than a measured launch.

The interesting number in that table is the last one. When the host is not
blocked it commits a new command buffer every 14.58 microseconds, which is the
enqueue cost and nothing else. The mean of 106.30 microseconds is seven times
the median because a small minority of the intervals contain a full blocking
round trip.

### 6.1 Which command buffers the host actually blocks on

This is the mechanistic result of the lane.

A command buffer is a serialization point if its successor was committed only
after the GPU had already finished it. Anything else was in flight, so the
host pipelined past it and paid no wait.

```
serialization points   3208 of 22107 command buffers (14.5%)
host time stalled      924.55 ms (39.4% of the span)
Blit       3406 total,   3206 of them serialization points (94.1%)
Compute   18701 total,      2 of them serialization points ( 0.0%)
```

The host never waits on a compute kernel. Two out of 18,701, which is noise.
It waits on 94 percent of all the blits. Every one of the 32 synchronizations
per round is a blocking readback, and the 4,980 large gaps of section 4.1 are
those readbacks and their consequences.

### 6.2 What a blocking readback actually costs

Decomposing the lifecycle of the 3,206 blocking readbacks, against a round
that is 23.50 milliseconds of wall clock:

```
stage                                  median us    mean us  per round ms  % of round
the commit call itself                      10.0       10.8          0.35        1.5%
commit end -> GPU start                    298.1      325.0         10.42       44.3%
GPU moving bytes                             3.7        4.6          0.15        0.6%
GPU end -> completion signal               109.5      111.2          3.57       15.2%
completion signal -> next commit           164.8      177.1          5.68       24.2%
TOTAL, commit -> next commit               606.1      628.7         20.16       85.8%
```

One blocking readback costs 606 microseconds of wall clock at the median. Of
that, 3.7 microseconds is the GPU moving the bytes. The data movement is
0.74 percent of the cost of the operation whose entire purpose is data
movement.

Thirty-two of these per round at 606 microseconds is 20.16 milliseconds of a
23.50 millisecond round, which is 85.8 percent. This is the whole shape of the
problem in one line.

### 6.3 How much of the queue wait is real

The largest single term above is the 298 microseconds between the host
finishing its commit call and the GPU starting the blit. Some of that is
legitimate: the readback is queued behind compute the host had already
submitted, and waiting for real work to drain is not overhead. So the trace
was asked how much of each of those windows the GPU spent executing anything
at all.

```
total queue wait            1041.81 ms
  GPU executing other work   534.20 ms  (51.3%)
  GPU idle                   507.61 ms  (48.7%)
idle time inside one queue wait: median 162.0 us, mean 158.3 us
```

Half of it is the GPU draining prior compute, which is the pipeline working as
intended. The other half, 158 microseconds per readback and 5.08 milliseconds
per round, is the GPU sitting idle with a submitted command buffer it has not
started. That is submission path latency and it is pure loss.

## 7. What this changes

**The transfer phase is not a transfer cost.** The stage-level profile
attributes 28.2 percent of the round to transfer. The GPU spends 15.33
milliseconds of a 2346.74 millisecond span moving bytes, which is 0.65 percent.
The phase is expensive because the host blocks on it, not because bytes are
slow. This is consistent with, and is the mechanism behind, the finding already
recorded in `docs/APPLE_UNIFIED_MEMORY.md` that copies run at 75 to 85 GB/s and
transfer is not the cost. Any lane proposing to make transfers faster, to use
unified memory more cleverly, or to compress what crosses the boundary, is
optimizing 0.65 percent of the round.

**The bubble estimate justified the large lane, and understates it.** The
question was whether roughly 20 milliseconds of gap per 41 millisecond round
was real, because if it were it would justify moving the whole tree onto the
device. The gaps are 76 percent of the span rather than 49 percent, they are
concentrated in one identified mechanism, and the mechanism is exactly the
host round trip that a device-resident tree removes. The lane is justified,
and its ceiling is higher than the estimate implied.

**The right target is the count of synchronizations, not the cost of one.**
Each is already close to the floor for a blocking round trip on this device.
The 3.7 microseconds of byte movement cannot be reduced, the 110 microsecond
completion notification is the operating system's, and only the 158
microseconds of idle queue wait looks addressable without changing the
pattern. Removing a synchronization is worth 606 microseconds. Making one
faster is worth at most 158.

**Kernel work is not where the round is.** Compute occupies 22.9 percent of a
round. Even an infinitely fast histogram leaves 77 percent of the round in
place. This does not mean kernel work is worthless, because the device-resident
tree will expose it once the round trips are gone, but it does mean a kernel
lane cannot be justified by its effect on the current round.

## 8. What remains unanswered

Whether the histogram kernel is latency-bound or bandwidth-bound. Section 5
narrows it to somewhere between 21.6 and roughly 120 GB/s of DRAM traffic and
cannot do better. Settling it needs either a device that exposes real GPU
counters to Instruments, or a microbenchmark that varies only the reuse
distance while holding the clock state fixed, or an Xcode GUI capture with
Shader Timeline enabled on a single dispatch. The GUI path is the cheapest of
the three and this lane did not attempt it.

Occupancy, as measured rather than computed from a grid.

What is inside the 164.8 microseconds between the completion notification and
the next commit. That is host code, so a CPU time profile would answer it, and
Instruments records one in this same template. It was not analyzed here.

Whether the 158 microseconds of idle queue wait is reducible at all, or is the
floor for a command buffer submission on this driver. A synthetic two-command-buffer
microbenchmark under the same capture would answer it directly.

Why the device chooses Minimum performance state for some runs and Maximum for
others. The narrative field says "due to active device conditions" in every
case, which says nothing. This is worth pursuing because it is the largest
known source of noise in every benchmark this repository publishes on Apple
silicon.

## 9. Reproducing this

```sh
bash bench/apple/metal_capture.sh --self-check
bash bench/apple/metal_capture.sh --print-plan
bash bench/apple/metal_capture.sh --rows 200000 --features 50 --arm gpu
```

The script builds the binary, records the trace, and reduces it. A trace is
about 120 MB per four seconds of traced run and is kept, not deleted, along
with a cached directory of the exported XML tables beside it so that
re-reading is fast. An existing trace can be re-read without recapturing:

```sh
bash bench/apple/metal_capture.sh --analyze-only path/to/some.trace
python3 bench/apple/metal_timeline.py path/to/some.trace --rounds 100
open path/to/some.trace     # the Instruments UI, for the visual timeline
```

Before comparing any two captures, check that their GPU performance state
breakdowns match. If they do not, their kernel durations are not comparable
and section 5.4 is the reason.

Full reduced output for the capture this document is written from is in
`bench/results/metal_timeline_2026-08-15/`.
