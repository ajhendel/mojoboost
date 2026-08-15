# Apple Silicon CPU and GPU timings

Every cell below is empty. Cells here get filled in by an actual run on
actual hardware, by the person who ran it, and by nothing else. No estimate,
no extrapolation from another chip, no number carried over from a different
shape or a different build.

If you are reading this file to find out whether the GPU is faster, the
answer today is that we do not know, and the one measurement that exists says
no. See [What we already know](#what-we-already-know).

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
the multicore CPU trainer. The GPU trainer was slower at every shape that was
tried, and the structural reason is understood, which is that each per leaf
histogram build scans all rows to filter by leaf id. Device side row
compaction is the work that would change that.

That result is why `device="auto"` resolves to the CPU, why no crossover
threshold ships, and why this table starts empty rather than starting with a
headline. It is also why filling this table is worth doing. A second data
point on different silicon is the cheapest way to learn whether the M4 result
is about the chip or about the kernel.

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
