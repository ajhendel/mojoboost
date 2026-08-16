# Apple Silicon CPU and GPU timings

Every cell below is empty. Cells here get filled in by an actual run on
actual hardware, by the person who ran it, and by nothing else. No estimate,
no extrapolation from another chip, no number carried over from a different
shape or a different build.

If you are reading this file to find out whether the GPU is faster on your
Apple part, the answer is that we do not know, because the only Apple part
anyone has run this on is one M4. On that one M4 the GPU is faster than our
own CPU at a million rows and slower below it. See
[What we already know](#what-we-already-know).

## The table

| Chip | Rows | Features | Rounds | Objective | CPU seconds | GPU seconds | GPU / CPU | macOS | MAX / Mojo | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1 |  |  |  |  |  |  |  |  |  |  |
| M1 Pro / Max |  |  |  |  |  |  |  |  |  |  |
| M2 |  |  |  |  |  |  |  |  |  |  |
| M2 Pro / Max |  |  |  |  |  |  |  |  |  |  |
| M3 |  |  |  |  |  |  |  |  |  |  |
| M3 Pro / Max |  |  |  |  |  |  |  |  |  |  |
| M4 |  |  |  |  |  |  |  |  |  |  |
| M4 Pro / Max |  |  |  |  |  |  |  |  |  |  |

Add a row per shape rather than overwriting one. A chip with one shape
measured is a chip we know one thing about.

## What we already know

`bench/bench_train_gpu.mojo` has been run end to end on an Apple M4 against
the multicore CPU trainer, several times, and the answer has changed since
this section was first written. It used to read "The GPU trainer was slower
at every shape that was tried", and that is no longer true. In seconds of
training with binning excluded, 100 rounds, 31 leaves, 255 bins, squared
error, median of three interleaved arms
(`bench/results/profile_2026-08-15/RESULTS.md`):

| shape | CPU | GPU |
| --- | --- | --- |
| 1,000,000 x 50 | 6.98 | **3.58** |
| 250,000 x 50 | **1.66** | 1.89 |
| 50,000 x 50 | **0.564** | 1.63 |

Multiclass, 465,000 x 54 over 7 classes: CPU 25.47, GPU **15.30**.

A second sweep on the same machine, five repeats per arm, repeated the first
two shapes (5.942 CPU against 3.756 GPU, and 1.649 against 1.967), added
2,000,000 x 50 at 13.483 against 6.093, and added a `grow_policy="depthwise"`
GPU arm that is the fastest thing this project has measured: 1.909 / 2.587 /
5.417 across the three shapes. At 1,000,000 x 50 that arm is faster than
LightGBM on ten CPU cores, 2.587 against 2.767, which is the first time this
library has been ahead of LightGBM on training time at a large shape
(`bench/results/sweep2_2026-08-15/RESULTS.md`). It grows a different tree than
LightGBM does and its accuracy has not been measured, so read the caveats in
the main README before quoting it.

The structural reason for the small-shape losses is also better understood
than "each per leaf histogram build scans all rows", which order-preserving
row compaction has since addressed. The first Metal timeline
(`docs/METAL_TIMELINE.md`) finds the GPU idle for 87.5% of a training span at
50,000 rows, at the device's Maximum clock, because the host blocks on 32
readbacks per round at about 606 microseconds each. Those round trips cost
what they cost whatever the row count, which is the 1.5 seconds of fixed
per-fit cost the table above implies.

One crossover rule now ships from these records, scoped to Metal on an M4 at
the largest shape. `device="auto"` still resolves to the CPU because that
rule cannot match a device profile that names no hardware, which is a wiring
gap rather than a missing measurement.

This table still starts empty, and filling it is still worth doing, because
every number above is one machine. A second data point on different silicon
is the cheapest way to learn whether the M4 result is about the chip or about
the kernel.

## How to produce a row

1. Build from source in the checkout you are measuring.

   ```sh
   pixi install
   pixi run build-python
   ```

2. Quiet the machine. Plugged into power, no thermal throttling from a prior
   run, no other Mojo compile or benchmark in flight, nothing else heavy
   running. A shared checkout with several builds going is not a measurable
   machine.

3. Run one shape, both backends, from the same process and the same data.

   ```sh
   PYTHONPATH=python python examples/apple_silicon/five_minute_tour.py \
       --time --rows 100000 --features 100 --time-rounds 100
   ```

   The script prints a prefilled markdown row for this table. Fill in the
   chip, the OS and toolchain versions, and anything the notes column should
   carry.

4. Record what the run actually did, including a GPU request that raised.
   A refusal is a result. Put the exception in the notes column.

5. Repeat the run at least three times and record the best of them, or record
   all of them. Say which in the notes. A single timing on a laptop is noise.

`pixi run bench-train-gpu [rows feats reg|binary]` is the Mojo side
equivalent and avoids the Python boundary entirely. Prefer it when the
question is about the trainer rather than about the estimator API.

## Rules for this file

- A number goes in only when the person adding it ran the thing.
- Both backends in a row come from the same machine, the same data, and the
  same build. A CPU number from one commit next to a GPU number from another
  is not a comparison.
- The CPU column is the multicore CPU trainer at its default worker count
  unless the notes say otherwise. If `MOJOTREES_NUM_WORKERS` was set, say so.
- A ratio below 1.0 means the GPU took less time. Write the ratio, not an
  adjective.
- An empty cell stays empty. Nothing is more useful than a fabricated
  baseline is harmful.

A fuller measurement protocol, covering warmup and compilation time,
transfers, peak memory, energy, and comparison against LightGBM and XGBoost,
is being written separately as `docs/APPLE_GPU_BENCHMARK_PROTOCOL.md`. Where
that document exists and disagrees with the four steps above, it wins.
