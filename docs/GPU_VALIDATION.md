# Cross-vendor GPU validation

mojotrees has one GPU source. `histogram_gpu.mojo` holds the histogram and
partition kernels, `train_gpu.mojo` drives device-resident tree growth, and
`gpu_tiling.mojo` derives launch geometry from device attributes at runtime.
There is no CUDA file, no HIP file, and no Metal file. That is a design
commitment, and it is worth exactly as much as the evidence that the one
source is correct and reasonable on every backend it claims to target.

This document is that evidence, plus the instructions for producing the parts
of it that do not exist yet.

## Status

| Backend | Device | Correctness | Determinism | Phase timings | Profiler trace |
|---|---|---|---|---|---|
| Metal | Apple M4 (10 core) | pass | pass | partial | not run |
| CUDA | NVIDIA RTX 5090 (Blackwell, sm_120) | **partial** | **not run** | **partial** | **not run** |
| HIP | AMD Instinct MI300X (gfx942, SR-IOV) | **pass** | **partial** | **not run** | **not run** |

NVIDIA hardware executed this code for the first time on 2026-08-18, on a
leased RunPod RTX 5090. **It builds, it runs, and one real cross-vendor
defect was found and fixed here.** It is not yet validated.

Correctness reads **partial**. One real defect was found, fixed and
re-verified on this device: NVPTX contracted a multiply and an add into an
FMA in `_range_add_raw_kernel` where Metal's compiler did not, breaking
bit-identity between two score-update arms on 1841 of 3000 rows. Moving the
multiply to the host closed it, and the test now passes on the same card.

It is **not** `pass`, and the reason narrowed considerably on 2026-08-18.
**66 GPU assertions pass on this card**, across five suites, each in under
thirty seconds, and they include the test that caught the FMA defect. What
does not pass is one class of work: a full training fit never returns. That
is a hang rather than slowness, it is localized to Modular's kernel-compiler
runtime against the NVIDIA driver, and it is written up as an upstream report
in [`docs/UPSTREAM_MAX_CUDA_HANG.md`](UPSTREAM_MAX_CUDA_HANG.md). Determinism
has still never been measured here, because the harness that would measure it
is the thing that hangs.

An earlier revision of this section said "36 GPU tests do not finish inside a
300s cap and the reason is not yet established". That count came from a run
whose pool over-subscribed the GPU by two orders of magnitude and whose cap
was set below the compile time, and it is superseded by the run recorded in
section 2f. It is quoted rather than deleted because it was cited elsewhere
while it stood.

No row above may be read as a support claim. The full record, including the
elimination argument that identifies the kernel, is in
[NVIDIA RTX 5090, CUDA, 2026-08-18](#nvidia-rtx-5090-cuda-2026-08-18).

**AMD hardware executed this code for the first time on 2026-08-19**, on a
leased MI300X, and it did something no other backend has managed this month:
it ran a full training fit. 22 assertions pass, including the 14 in
`test_gpu_training`, the suite that has never returned on CUDA. See
[AMD Instinct MI300X, HIP, 2026-08-19](#amd-instinct-mi300x-hip-2026-08-19).

That result is why the CUDA row's open question is now much narrower. The same
source, the same commit, the same MAX version and the same script complete on
HIP in 63 seconds and hang forever on CUDA. Whatever the cause is, we do it
identically on both.

The sentence this section used to carry, that no AMD hardware had ever
executed this code, is retired. It stood from 2026-08-14 to 2026-08-19.

Nothing in this repository should be read as a claim about NVIDIA or AMD
behavior or performance. The Metal "partial" for phase timings means the
harness runs and prints, not that a recorded sweep exists. In particular, the
argument that a bandwidth-hungry histogram kernel would go faster on a card
with its own high-bandwidth memory is a **prediction**, not a result, and one
recorded execution of the procedure below on such a card is what would turn it
into one.

The Metal row is also one machine and not a chip family. Every Metal figure
came off a single Apple M4 laptop, 10 CPU cores, 10 GPU multiprocessors, 16
GB, macOS 26.5.2, whose CPU and GPU share one memory bus rated at about 120
GB/s. Any comparison run on it puts our GPU arm and a CPU comparator on the
same bandwidth, which handicaps every arm identically on this machine and does
not represent a machine where the accelerator owns its memory and the CPU
comparator keeps the whole host bus. No other Apple chip has run this code
either.

## What validation means here

Per device, six things, in this order. Correctness gates everything after it,
so a failure at step 2 or 3 means no timing is recorded at all.

1. **Build.** `has_accelerator()` is a compile-time query, so the GPU path is
   compiled in only on a machine that has a device. A clean build on the
   target is itself a result.
2. **Correctness.** GPU histograms and GPU-trained models agree with the CPU
   reference within the documented Float32 tolerance, and integer counts
   agree exactly.
3. **Determinism.** Repeat runs are bit-identical. Fixed-point Int32
   accumulation is what buys this, and it is the property most likely to
   break on a backend whose atomics or scheduling differ.
4. **Shapes.** Several dataset shapes, not one. Wide, tall, square, and small
   enough to be launch-overhead bound, because they stress different ratios
   of `grid.x` to `grid.y`.
5. **Phases.** Setup, host-to-device transfers, kernels, and total training
   measured separately, so a slow total can be attributed rather than
   guessed at.
6. **Profiler.** Occupancy, shared-memory footprint, and atomic contention
   read off a real trace and compared against the harness predictions.

## Hardware

Any device MAX supports is acceptable. The kernels use nothing exotic: 2D
grids, threadgroup shared memory, integer atomics, and no Float64 on device.

Practical options, cheapest first:

- **NVIDIA, cloud.** AWS `g5.xlarge` (A10G), AWS `g6.xlarge` (L4), or GCP
  `g2-standard-4` (L4). Any of these is a few dollars for a validation run.
  A Deep Learning AMI or base Ubuntu image with the proprietary driver both
  work; the CUDA toolkit is not required, only the driver.
- **AMD, cloud.** Harder to rent by the hour than NVIDIA. AWS does not offer
  a general-purpose AMD GPU instance, so the realistic paths are Azure
  `Standard_NG*` / MI300X capacity, an Oracle Cloud MI300X shape, or a
  bare-metal provider that leases MI210 or MI300X. ROCm must be installed and
  `rocm-smi` must report the device.
- **AMD, desktop.** An RX 7900 XT/XTX or RX 9070 on a Linux box with ROCm is
  a legitimate HIP validation target and much cheaper to keep around. Record
  the exact card; consumer RDNA and datacenter CDNA differ enough that a
  result from one does not transfer to the other.

Check the device against Modular's current GPU compatibility list before
paying for anything. If MAX does not support the card, the build fails and
that is a result worth recording too, in the failures section below.

## Procedure

Identical on both vendors. Only the driver-reporting command differs, because
`nvidia-smi` and `rocm-smi` are different programs.

```sh
git clone https://github.com/mojotrees/mojotrees && cd mojotrees
curl -fsSL https://pixi.sh/install.sh | sh
pixi install
```

### 1. Record the environment

```sh
uname -a
lscpu | grep -iE 'model name|architecture|^cpu\(s\)'
pixi run mojo --version
pixi list --environment default | grep -Ei '^(mojo|max)'

# NVIDIA
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

# AMD
rocm-smi
rocminfo | grep -iE 'name|gfx|compute unit'
cat /opt/rocm/.info/version
```

### 2. Correctness and determinism

```sh
pixi run test       # full suite, includes the GPU equivalence tests
pixi run test-gpu   # GPU suites on their own, for a shorter loop
```

The GPU tests print `skipped: no accelerator` and pass on a CPU-only machine.
**A pass that says `skipped` is not a validation.** Read the output. If the
GPU tests skipped on a machine with a GPU, the build did not see the device
and nothing has been tested.

Then confirm the pin-to-CPU path works on a machine that does have a device,
which is the only place that branch is reachable:

```sh
MOJOTREES_DISABLE_GPU=1 pixi run test
```

### 3. Shapes and phases

```sh
pixi run gpu-validate                       # built-in four-shape sweep
pixi run gpu-validate 20 250000 200         # explicit rounds and shape
pixi run gpu-validate 20 10000 20 100000 100 50000 400 1000000 20
```

`bench/bench_gpu_validation.mojo` prints, per device, the identity and
capability attributes, and per shape the launch geometry and the phase
breakdown. Pin CPU threading first so the CPU comparison is reproducible:

```sh
export MOJOTREES_NUM_WORKERS=8
export MOJOTREES_PARALLEL_MIN_OPS=65536
```

Phases the harness reports, and what each covers:

| Phase | Covers |
|---|---|
| `binning_s` | host-side quantile binning, no device involvement |
| `setup_s` | context, capability query, allocation, binned-matrix upload |
| `stage_s` | Float64 to Float32 gradient conversion, host only |
| `grad_h2d_s` | staged gradients and hessians copied to the device |
| `hist_kernel_root_s` | `enqueue_leaf` plus synchronize, kernels only |
| `hist_d2h_s` | fixed-point histogram copied back to the host |
| `hist_decode_s` | fixed-point to Float64 conversion, host only |
| `partition_kernel_s` | one `apply_split` plus synchronize, kernel only |
| `hist_kernel_child_s` | the same histogram kernels after a real split |
| `gpu_train_s` | complete `train_gpu`, uninstrumented |
| `cpu_train_s` | complete `train`, same shape and parameters |

Every phase is measured directly, through the builder's own phase methods
(`stage_gradients`, `upload_staged`, `enqueue_leaf`, `download_raw`,
`histogram_from_host`). Nothing is derived by subtraction. What the harness
cannot see is where time goes inside a kernel, which is what step 4 is for.

Both trainers report training MSE. Record it. A throughput number without a
loss number next to it is not a result.

### 4. Profiling

The harness prints predictions from host-side arithmetic. The profiler says
whether they are true. Confirm metric names against `ncu --query-metrics` or
`rocprofv3 --list-avail` on the installed version rather than trusting the
names below, which drift between releases.

Build a binary once so the profiler has something to attach to:

```sh
pixi run mojo build -I src -o /tmp/gpu-validate bench/bench_gpu_validation.mojo
```

**NVIDIA.** Nsight Compute for per-kernel detail, Nsight Systems for the
timeline that shows transfer and kernel overlap:

```sh
ncu --set full --kernel-name-base function \
    --launch-count 20 -o gpu-validate /tmp/gpu-validate 5 100000 100
nsys profile -o gpu-validate /tmp/gpu-validate 5 100000 100
```

**AMD.** `rocprofv3` for kernel and copy durations, ROCm Compute Profiler
(formerly Omniperf) for the occupancy and LDS detail:

```sh
rocprofv3 --kernel-trace --memory-copy-trace --stats -- /tmp/gpu-validate 5 100000 100
rocprof-compute profile -n gpu-validate -- /tmp/gpu-validate 5 100000 100
```

What to look at, and which harness line it tests:

| Question | Harness prediction | NVIDIA | AMD |
|---|---|---|---|
| Achieved occupancy | `blocks_per_sm_at_launch` | `sm__warps_active.avg.pct_of_peak_sustained_active` | wave occupancy in the ROCm Compute Profiler occupancy panel |
| What limits occupancy | `shared_memory_resident_block_limit` | `launch__occupancy_limit_shared_mem` versus the register and block limiters | LDS allocation granularity versus VGPR count |
| Static shared memory | `shared_bytes_reserved_per_block` | `launch__shared_mem_per_block_static` | LDS allocated per workgroup |
| Wasted shared memory | `shared_bytes_used_per_block` versus reserved | same static figure, compared against bins actually used | same |
| Shared atomic contention | `threads_per_bin_contention_proxy` | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum` | LDS bank-conflict counter (`SQ_LDS_BANK_CONFLICT` on most versions) |
| Global atomic traffic | `global_atomics_per_launch_upper` | `lts__t_sectors_op_atom.sum` | L2 or TCC atomic request counters |
| Kernel versus transfer | `hist_kernel_*` against `hist_d2h_s` | Nsight Systems timeline | `rocprofv3` kernel and copy rows |

Run both accumulation strategies. They are the same per-threadgroup work and
differ only in how partials combine, so the pair isolates atomic cost from
everything else:

```sh
MOJOTREES_GPU_HIST_STRATEGY=atomic pixi run gpu-validate 20 100000 100
MOJOTREES_GPU_HIST_STRATEGY=tiled  pixi run gpu-validate 20 100000 100
```

`tiled` issues no global atomics at all, so the gap between the two is the
global-atomic cost on that device, measured rather than argued. Both must
produce bit-identical histograms; `tests/test_gpu_strategies.mojo` asserts
that on whatever device runs it, and a divergence on real hardware is a
correctness bug, not a tuning result.

Three things the code already predicts, all worth confirming rather than
assuming:

- The kernels reserve `3 * GROUP * BIN_CAP * 4` bytes of shared memory, where
  `BIN_CAP` is the dataset's bin count rounded up the ladder 32, 64, 128,
  256. They used to reserve `3 * MAX_BINS * 4` at every bin count, so a
  64-bin dataset reserved the same 3072 bytes as a 256-bin one; sizing the
  reservation to the bin count was predicted here as the first thing to try
  if the profiler showed shared memory limiting occupancy, and it has since
  been done. It is what makes a feature group wider than 2 affordable at low
  bin counts, since the freed memory buys group width at the same footprint.
- Every in-range row issues three shared-memory atomics into a bin chosen by
  its feature value. With 256 threads and 64 bins, four threads per bin
  collide on average. High bank-conflict counts point at per-warp
  privatization or a wider shared layout, again portable.
- Every node's histogram scans all `n_rows` and filters on the leaf id, so a
  tree costs `num_leaves * n_rows * n_features` bin reads. Deep in a tree
  most threads find no matching row. Low achieved occupancy on the child
  histograms is expected and is what order-preserving row compaction is
  meant to fix.

## Apple M4, 2026-08-15: the five-lane GPU performance round

Recorded here because this file is where device measurements live, and
because three of the five changes in that round are defaults now and the
fourth was reverted on the strength of these numbers.

Mojo 1.0.0 (ed45d567). All figures are seconds of GPU training at 100
boosting rounds on a regression target, reported as the minimum of the
repeats, taken with `bench/bench_train_gpu.mojo`. Arms inside one process
are interleaved; arms across processes were alternated and each figure below
was reproduced at least twice in the same window, because this machine's
device timings drift two to three times across windows and only back-to-back
runs compare.

Headline, 1,000,000 rows by 50 features, five repeats:

| arm | seconds | spread |
|---|---|---|
| CPU backend | 11.36 | 3.8 percent |
| GPU, every change on | 4.10 | 5.8 percent |
| GPU, every escape hatch set to the old path | 4.28 | 0.7 percent |

So the round is worth about 3 to 4 percent end to end at this shape, and the
GPU backend is about 2.8 times the CPU one. The 0.56x figure that circulated
before this round is stale by a wide margin and should not be quoted.

**The 2.8x in the paragraph above is itself now stale, and in our own
disfavor.** A second round landed on 2026-08-15 and moved the CPU further
than it moved the GPU: 11.36 to **6.98** on the CPU and 4.10 to **3.58** on
the GPU at the same shape, so the GPU is **1.85x** the CPU rather than 2.8x.
Neither figure was wrong when it was taken; the ratio fell because the CPU
got 1.63x faster while the GPU got 1.15x faster. The record is
`bench/results/profile_2026-08-15/RESULTS.md`, taken under the rules
committed beforehand in `bench/results/PROFILE_PROTOCOL.md`. Quote 1.85x, or
better, quote the two seconds figures and let a reader form the ratio.

Per change, isolated by setting one escape hatch at a time from the
all-on configuration:

| change | reverting it costs | verdict |
|---|---|---|
| pre-quantized gradients (`MOJOTREES_GPU_QUANTIZED_GRADS`) | 5.59 against 5.03 | a win, on by default |
| block-primitive partition scan (`MOJOTREES_GPU_SCAN_PRIMITIVES`) | 5.12 against 5.03 | a small win, on by default |
| block-primitive split search (`MOJOTREES_GPU_SPLIT_PRIMITIVES`) | 5.07 against 5.03 | indistinguishable |
| row-tile floor (`MOJOTREES_GPU_MIN_TILES`) | 4.11 against 5.03 | a LOSS, reverted to opt-in |

The split-search primitives were also measured on the shape they were
written for, 50,000 rows by 50 features, where the device search loses to
the host scan. Three alternating pairs gave 2.473, 2.629, 2.685 with the
primitives and 2.649, 2.476, 2.688 without: indistinguishable. That gap has
another cause. Note for whoever picks it up that the launch count is not it,
since the search already runs one grid over every (feature slot, node) task
and one over every node, which is the shape LightGBM's CUDA best-split
finder uses.

The row-tile floor is written up in `gpu_tiling.row_tile_floor`, including
the 100-feature case where it loses by 36 percent, the strategy-forced arms
showing the loss is not the reduction kernel, and the observation that the
whole loss happened inside the region `MIN_ROWS_PER_TILE_BIN_FACTOR` calls
safe.

Three claims that appear elsewhere in this repository are refuted by the two
tables above and should not be repeated. The row-tile floor does not help; it
measured 4.11 against 5.03 and is opt-in for that reason. Block primitives in
the split search did not help; 5.07 against 5.03 is inside the spread, which
the protocol requires be recorded as indistinguishable rather than as a small
win. And bit-packed bins cannot cut memory traffic 3 to 8x by the mechanism
usually named for it: four-bit bins against one-byte bins is 2.00x and no
arithmetic makes it more. Whether a four-byte packed load beats four one-byte
loads is a separate and still unmeasured question on every backend
(`docs/NVIDIA_GPU.md`, `docs/AMD_GPU.md`).

## Apple M4, 2026-08-15: the second performance round, and the first GPU timeline

Two things landed the same day and they have to be read together, because
the second one changes how the first one should be interpreted.

The full record is `bench/results/profile_2026-08-15/RESULTS.md`, taken under
the pre-registered rules in `bench/results/PROFILE_PROTOCOL.md`, and the
timeline is written up in `docs/METAL_TIMELINE.md` with the reduced captures
in `bench/results/metal_timeline_2026-08-15/`. Everything below is seconds of
training time with binning excluded, 100 rounds, 31 leaves, 255 bins, squared
error, median of three with the arms interleaved.

| shape | our CPU | our GPU | LightGBM, 10 threads |
|---|---|---|---|
| 1,000,000 x 50 | 6.98 | **3.58** | **2.86** |
| 250,000 x 50 | **1.66** | 1.89 | **1.00** |
| 50,000 x 50 | **0.564** | 1.63 | 0.594 |

The GPU is 1.85x our own CPU at a million rows, 0.83x at 250,000, and 0.33x
at 50,000. The device carries about 1.5 seconds of fixed cost per fit, which
is what those three numbers imply and which the timeline below then
attributes.

A second sweep on the same machine at five repeats
(`bench/results/sweep2_2026-08-15/RESULTS.md`) repeats the top two shapes,
adds 2,000,000 x 50, and adds a `grow_policy="depthwise"` GPU arm:

| shape | our CPU | our GPU, leaf-wise | our GPU, depth-wise | LightGBM, 10 threads |
|---|---|---|---|---|
| 250,000 x 50 | 1.649 | 1.967 | 1.909 | **1.023** |
| 1,000,000 x 50 | 5.942 | 3.756 | **2.587** | 2.767 |
| 2,000,000 x 50 | 13.483 | 6.093 | 5.417 | **5.228** |

Two things in that table are new and neither is a general performance claim.
**Fitted** across the two segments, our marginal cost per row (2.385 then
2.337 microseconds) and LightGBM's (2.325 then 2.461) interleave, so the two
are even per row and the whole deficit is an intercept of roughly one second
(**derived** from the same fits). And the depth-wise arm is ahead of LightGBM
at 1,000,000 x 50 by 6.5 percent at a 0.3 percent spread, which is the first
measured win this project has on training time at a large shape. It compares
our depth-wise trees against LightGBM's leaf-wise trees, which are different
models, and no training loss was recorded for any arm of the sweep, so this
is a timing result and not an accuracy one.

Binning, excluded from the table: ours 0.377s against LightGBM's 1.207s, so
we bin 3.2x faster.

Multiclass, 465,000 rows by 54 features over 7 classes, measured honestly for
the first time now that `train_dataset_multiclass` no longer discards the
device it resolved:

| arm | median | spread |
|---|---|---|
| our CPU | 25.47 | 7.7 percent |
| our GPU | **15.30** | 0.1 percent |
| our GPU, `MOJOTREES_GPU_CLASS_BATCH=7` | 15.45 | 0.8 percent |

**The GPU wins multiclass by 1.63x**, resolved well outside the noise floor,
and class batching at seven is indistinguishable from the sequential
schedule.

### What the timeline says, and why it does not contradict the phase profile

The stage-level phase profile attributes **49.3 percent of device work** to
the histogram at 1,000,000 x 50. The Metal timeline says **compute of every
kind is 22.9 percent of a round** at 200,000 rows. Both are correct and they
are not the same statement, because they have different denominators. The
phase profile measures where attributed device work is charged and cannot see
time in which no phase is running at all. The timeline measures the whole
wall-clock span, idle included. So the histogram is about half of the work
the device does, and the device does work for less than a quarter of the
round.

That is the correction, and it matters because "the histogram kernel is the
dominant cost of a GPU round" was inferred from the first number and is
false. An infinitely fast histogram kernel leaves roughly four fifths of the
round untouched.

What the round is actually spending is host round trips:

```
GPU idle inside the training span   76.5% at 200,000 rows, 87.5% at 50,000
GPU performance state               Maximum for 77.9% of the capture
host blocks on blits                3,206 of 3,406  (94.1%)
host blocks on compute kernels      2 of 18,701     ( 0.0%)
serialization points per round      32.1
one blocking readback, median       606.1 us, of which 3.7 us is bytes moving
enqueue (the commit call), median    12.62 us
completion notification, median     101.33 us
serialized turnaround, median       285 us
```

Thirty-two blocking readbacks at 606 microseconds is 20.16 milliseconds of a
23.50 millisecond round, or 85.8 percent. The idleness is not a downclock:
the device sat at its Maximum performance state for 77.9 percent of the
capture.

Two consequences worth stating plainly. The `transfer` phase is not a
transfer cost, because the GPU spends 0.65 percent of the span moving bytes;
it is the phase the wait gets charged to. And the right target is the
**count** of synchronizations rather than the cost of one, since removing a
synchronization is worth 606 microseconds while making one faster is worth at
most the 158 microseconds of idle queue wait inside it.

One limitation, and it is a property of the device rather than the method.
This M4 exposes Instruments exactly one GPU counter, `RT Unit Active`, which
is about the raytracing unit. There is no occupancy, ALU, bandwidth, or cache
counter to read, so **no kernel in this repository can today be called
latency-bound or bandwidth-bound on the basis of a measurement**. Section 5
of `docs/METAL_TIMELINE.md` narrows the histogram kernel's DRAM traffic to
somewhere between 21.6 and roughly 120 GB/s and cannot do better.

One incidental finding that bears on every benchmark in this file. Two of the
four captures ran entirely at the device's Minimum GPU performance state
while the other two ran mostly at Maximum. That is the most plausible
mechanism yet identified for the two-to-three-fold drift this repository
fights on Apple silicon, and it means an interleaved comparison can still be
invalid if the two arms straddle a clock transition. Print the performance
state before comparing two captures.

## Recording a result

Append a block per device, verbatim from the terminal. Do not summarize, do
not round, do not paste a number you did not watch print.

```text
### <vendor> <device>, <date>

Driver:        <nvidia-smi / rocm-smi output>
Mojo:          <mojo --version>
MAX:           <pixi list max>
Host CPU:      <lscpu model name>, <cores>
OS:            <uname -a>

pixi run test        <pass | fail, with the failing assertion>
pixi run test-gpu    <pass | fail>
MOJOTREES_DISABLE_GPU=1 pixi run test   <pass | fail>

<paste the full gpu-validate output>

Profiler:      <occupancy, shared memory limiter, bank conflicts, atomics>
Failures:      <anything that did not build, run, or agree>
Unsupported:   <capabilities the backend does not implement>
```

### Apple M4, Metal, 2026-08-14

The only device this code has run on, kept here as the format worked example
and as the baseline the CUDA and HIP records get compared against. The device
header is what `gpu-validate` printed.

```text
Mojo:          Mojo 1.0.0 (ed45d567)
MAX:           26.5.0
Host:          Apple M4, 10 GPU cores, macOS 25.5.0 arm64

name: Apple M4
api: metal
arch_name: 4-metal4
compute_capability: 4
multiprocessor_count: 10
warp_size: unavailable
max_threads_per_block: 1024
max_threads_per_multiprocessor: unavailable
max_blocks_per_multiprocessor: unavailable
max_shared_memory_per_block: 32768
max_registers_per_block: unavailable
max_grid_dim_x: 2147483647
max_grid_dim_y: 2147483647
clock_rate_khz: unavailable
```

Correctness and determinism suites pass. Phase timings run and print; no
recorded sweep across the four default shapes exists yet, and no profiler
trace, which is why the status table says `partial` and `not run`.

Notes worth carrying into the CUDA and HIP runs:

- Metal answers five of the eleven attributes and refuses six, including
  `WARP_SIZE`. The tiling policy's fallbacks are load bearing here, not
  decorative.
- Metal reports `max_grid_dim_y` as the full 2^31 - 1, so CUDA's 65535 cap is
  genuinely the binding one and the clamp exists solely for CUDA. Confirm
  what each of the other backends reports.
- At 255 bins the shared reservation is 3072 bytes against 3060 used, so the
  reserve-by-`MAX_BINS` gap is invisible at the default bin count. It only
  opens up at smaller `max_bin`, which is where to look for it.

### AMD Instinct MI300X, HIP, 2026-08-19

The first AMD execution in this project's history, and the first backend other
than Metal to complete a training fit. Leased RunPod pod, script run as the
container's main process, output read from the pod log.

```text
Host        RunPod secure cloud, EU-RO-1, Ubuntu 24.04.2, glibc 2.39
CPU         AMD EPYC 9474F 48-core, NPROC reports 192
GPU         AMD Instinct MI300X, gfx942, SKU MI3SRIOV (SR-IOV virtualized)
ROCm        6.4.1-83
Toolchain   Mojo 1.0.0 (ed45d567), pixi 0.77.0
Commit      1b57950
```

#### The result

```text
build                          rc=0   27 s   mojopkg 7,225,057 bytes
probe alloc/launch/subbuffer/parallel  all REPRO*_COMPLETED_NO_DEADLOCK
variety, 128 distinct kernels  cold 40.10 ms      warm 40.82 ms
test_gpu_scan_primitives       rc=0   10 s   6 tests run: 6 passed
test_gpu_raw_update_packing    rc=0   12 s   2 tests run: 2 passed
test_gpu_training              rc=0   63 s   14 tests run: 14 passed
```

**22 assertions, all passing, on hardware that had never run a line of this
code.** `test_gpu_raw_update_packing` is the FMA test, so the host-prescale
fix holds on a third backend. `test_gpu_training` is the suite that hangs
forever on CUDA.

#### Why the FIRST AMD attempt reported no accelerator, and what it teaches

The first pod, on `rocm/dev-ubuntu-24.04:latest`, built the package cleanly in
27 seconds and then every probe printed:

```text
no accelerator present; nothing to reproduce
```

`rocm-smi` and `rocminfo` both saw gfx942. `/dev/kfd` and
`/dev/dri/renderD176` were both present. The difference was one number:

```text
/opt/rocm/lib/libamdhip64.so.7        <- ROCm 7, MAX reports no accelerator
/opt/rocm/.info/version  6.4.1-83     <- ROCm 6.4.1, everything works
```

`pixi.toml` pins `max >=26.5,<27`, which targets ROCm 6. On a ROCm 7 image MAX
finds no HIP runtime it recognizes and `has_accelerator()` is false at
COMPTIME, so the entire GPU half of this package compiles out and the build
still succeeds. **A green build on AMD is not evidence the GPU path exists.**
The two builds produced byte-identical package sizes, which is the cheapest
tell that nothing GPU-shaped changed.

Anyone validating on AMD should check `/opt/rocm/.info/version` before
trusting a single result, and `docs/AMD_GPU.md` now says so.

#### The variety numbers, beside the other two backends

```text
128 distinct kernels, one process     cold        warm      ratio
  Metal, Apple M4                    1370 ms      74 ms     18.0x
  HIP, AMD Instinct MI300X             40.1 ms    40.8 ms    1.0x
```

This reframes the cache question recorded in the M4 control. The M4's 18x
cold-to-warm gap looked like the important variable; against HIP it is a
symptom. Per kernel, cold, Metal costs 8 to 13 ms and HIP costs 0.30 ms, a
gap of roughly thirty times. **The MI300X compiles 128 distinct kernels in
less time than the M4 takes to compile four**, and needs no cross-process
cache because there is nothing worth amortizing.

So the honest statement is that per-kernel JIT cost differs enormously between
backends and caching is how the expensive one copes. Whether CUDA is expensive
like Metal or cheap like HIP is the number still missing.

#### What this does NOT establish

- **Determinism is `partial`, not `pass`.**
  `test_gpu_subsampled_training_is_deterministic` passed, which is
  within-run agreement. Repeat-run bit-identity across separate processes,
  which is what the Metal row means by `pass`, was not attempted.
- **No timing claim.** No phase sweep, no profiler, and nothing here may be
  placed beside an M4 or a LightGBM number. The 63 seconds is a test suite
  including compilation, not a fit time.
- **One card, virtualized.** The SKU reads `MI3SRIOV`, so this is an SR-IOV
  partition rather than a bare-metal MI300X, and CDNA results do not transfer
  to RDNA parts in any case (`docs/AMD_GPU.md`).
- **Nothing about MI300X performance.** The device plane stays gated on the
  validated table in `gpu_split_policy`, because correctness is not a
  crossover and no host-against-device sweep was run here.

### Apple M4, Metal, 2026-08-19: the kernel-variety control

Run on the development M4 as the CONTROL for the NVIDIA hang, not as a
performance record. `pixi run probe-cuda-variety` compiles and launches 128
distinct kernel instantiations in one process, importing nothing from this
package. Two consecutive runs, same binary, same machine.

```text
device: Apple M4        api: metal      n_kernels: 128

run 1, cold                          run 2, warm
  KERNEL   0   8.541 ms                KERNEL   0   8.541 ms
  KERNEL  76   8.550 ms                KERNEL   4   0.617 ms
  KERNEL 127   9.955 ms                KERNEL  64   0.347 ms
                                       KERNEL 127   0.378 ms
  PHASE_G total  1370.67 ms            PHASE_G total    74.39 ms
  ROUND 1..4  6.0 4.4 4.1 4.2 ms       ROUND 1..4  53.6 6.7 5.2 14.9 ms
  VARIETY_COMPLETED_NO_DEADLOCK        VARIETY_COMPLETED_NO_DEADLOCK
```

**Three things, and the third was not what the probe was written to ask.**

1. **The M4 does not hang.** 128 distinct kernels compile and launch in one
   process and the program completes. The premise that the RTX 5090 hang is
   specific to that backend survives its control, which it might not have.

2. **Per-kernel cost is flat.** Cold, every kernel from 0 to 127 costs
   roughly 8 to 13 ms with no trend. That kills a specific reading of the
   NVIDIA hang -- that per-kernel JIT cost accumulates in-process and the
   hang is simply the far end of a curve that never turns over. On Metal
   there is no curve. Whatever NVIDIA is doing, it is not this.

3. **Something amortizes kernel compilation across processes here, and it is
   probably not MAX.** The same binary on the same machine took 1370 ms and
   then 74 ms, an 18x drop, with the first kernel unchanged at 8.5 ms and
   every subsequent one falling under 1.4 ms. Nothing in the program changed
   between the runs, so whatever is saving the work is outside it.

**A correction, made the same day and before this left the repository.** The
first version of this section said "Metal caches compiled kernels on disk,
across processes" as though the mechanism were established. The 18x is
measured and stands. The mechanism was inferred, and `docs/STARTUP_LATENCY.md`
had already audited exactly this question against the MAX documentation and
concluded the opposite: MAX's documented cache is per-`DeviceContext`,
in-memory, in-process, and dies with the context, and **no on-disk cache for
Mojo GPU kernels is documented anywhere**. That audit even anticipated this
measurement -- "if it turns out to be [cached across processes],
`kernel_create` will simply be small, and that is a measurement, not an
assumption to build on."

So the honest reading is that macOS caches compiled Metal shaders at the
operating-system level, below and outside MAX entirely, which is a thing
Apple's stack is known to do and a thing Linux with NVIDIA has no equivalent
of by default. That is still a guess. What is not a guess is that the same
work costs 18x less the second time on this machine.

Either way it is the asymmetry to carry to the card, because
`~/.nv/ComputeCache` holds **zero files** after a CUDA run. If the effect is
real and absent there, then the backend that works pays for kernel
compilation once and the backend that hangs pays every process. The cheap
test is to run this probe twice on the RTX 5090 and see whether run 2 is 18x
faster or identical to run 1. **Run that before filing anything upstream:** a
report that names the wrong mechanism costs the reader more than one that
names none.

**What this does NOT show.** Nothing about speed. These are microseconds of
trivial arithmetic wrapped around a compile, and no number here belongs
beside any training timing. The probe measures compilation and dispatch and
was written to answer a yes-or-no question about hanging.

### NVIDIA RTX 5090, CUDA, 2026-08-18

The first non-Metal execution in this project's history. Recorded from a
leased RunPod pod, driven over SSH. **This is a partial record and is not a
support claim**: see "What is not established" below before citing anything
here.

```text
Host        RunPod secure cloud, EU-CZ-1, Ubuntu 24.04.3, glibc 2.39
CPU         256 vCPU (x86-64), AVX2 present, so the x86-64-v3 floor is met
GPU         NVIDIA GeForce RTX 5090, Blackwell, compute capability 12.0
VRAM        32607 MiB
Driver      580.159.03
Toolchain   Mojo 1.0.0 (ed45d567), pixi 0.76.2, from this repo's pixi.lock
Commit      7aefb7f
```

Two facts about that hardware belong in the record rather than in a reader's
head:

- **The driver clears Mojo's floor by one patch release.** MAX requires
  NVIDIA driver 580 or later; this host reports 580.159.03. RunPod assigns
  both old and new hosts for the same GPU model, and two earlier pods in this
  same session landed on CUDA 12.4 machines where the container refused to
  start at all (`nvidia-container-cli: requirement error: unsatisfied
  condition: cuda>=12.8`). A slightly older host would have produced no
  record, not a worse one. The `cudaVersion` field is visible at pod creation
  and is the cheap way to check before paying.
- **This part is "known compatible", not "continuously tested".** In
  Modular's compatibility table the RTX 50XX series is listed under *Known
  compatible for development*; B200 is the only NVIDIA part under *Tested for
  serving*. That does not weaken what passed, but it shifts the prior on what
  a failure means: a lowering or codegen failure here is more likely to be
  upstream than in this repository. See `docs/GPU_PORTABILITY.md` section 1.1
  for the frame, and `docs/HARDWARE_CONTRIBUTORS.md` for the tier table.

#### 1. Build

**Pass.** This is a real result rather than a formality, because
`has_accelerator()` is a compile-time query: the CUDA path had never been
compiled in anywhere, by anyone, before this run.

```text
pixi run build-pkg    exit 0, 28 s, build/mojotrees.mojopkg = 7,124,190 bytes
```

All 110 modules in `src/mojotrees` elaborated. Warnings only, no errors. Two
warnings are worth a follow-up rather than a shrug, because both name a
branch that compiles out on this backend and neither is obviously intended
to:

```text
src/mojotrees/distributed_gpu.mojo:315   'if HAS_DEVICE_COLLECTIVE:' is unreachable
src/mojotrees/gpu_objectives_native.mojo:2927
        'if (1 << MVS_BLOCK_SHIFT) != MVS_BLOCK_SIZE:' is unreachable
```

#### 2. Correctness

**Partial.** GPU tests execute on the device and the majority pass, including
`test_gpu_scan_primitives`, `test_gpu_partition_launches`, `test_gpu_tiling`,
`test_gpu_level_batcher`, `test_gpu_portability` and `test_gpu_vendor_policy`.

> **FIXED AND RE-VERIFIED, 2026-08-18.** The defect described below was
> real. It was fixed by moving the multiply to the host so the kernel has a
> plain add and nothing left to fuse, and the fix was re-run on the same RTX
> 5090 that exhibited the bug:
>
> ```text
> before:  FAIL test_gpu_raw_update_packing  1841 of 3000 rows disagree
> after:   ok   test_gpu_raw_update_packing  (3s wall, 863.348ms in tests)
> ```
>
> Host-prescale was chosen over writing an explicit `fma`, and the reason
> matters: an explicit `fma` would make CUDA and Metal agree by making
> **Metal fuse too**, which moves every M4 bit, every recorded model digest
> and every accuracy anchor in one commit. Prescaling leaves Metal
> bit-identical to what it already produced. Two portable fixes existed and
> only one of them was free.
>
> The account below is kept because the diagnosis method is the reusable
> part.

One failure was **real, reproducible, and explained**:
`test_gpu_raw_update_packing`, together with `test_gpu_fma_consistency`.
**The device compiler contracts a multiply and an add into an FMA on CUDA
where Metal's compiler did not, and it moved the bits of every
device-resident model.**

The evidence, from the test logs themselves:

```text
test_gpu_raw_update_packing
  PASS  test_step_bits_survive_the_descriptor_round_trip     (host only)
  FAIL  test_packed_table_arm_matches_the_per_leaf_arm
        packed and per-leaf arms disagree at row 2:
            packed 0xbfa2bec6   per-leaf 0xbfa2bec5
        assert left == right:  left 1841, right 0

test_gpu_fma_consistency
  raw update arms disagree at row 0 node 5:
      per-thread 0xbe04c2f0  per-leaf 0xbe04c2f1  table 0xbe04c2f0
      ulps(per-thread, table) 0
  rows disagreeing: 2225 of 3000, worst 0 ulp;
  per-thread matched the unfused host answer on 3000 rows and the fused
  one on 0
```

Read those together and the mechanism is settled by elimination rather than
inferred. Three arms compute the same score update:

| arm | multiply location | result |
|---|---|---|
| per-thread (`_update_raw_kernel`) | host | matches unfused, 3000/3000 |
| table (`_range_table_add_raw_kernel`) | host, packed into `SEG_STEP` | matches per-thread, 0 ulp |
| per-leaf (`_range_add_raw_kernel`) | **in the kernel** | differs by 1 ulp |

`gpu_objectives_native.mojo:774-777` is
`raw[i] = raw[i] + learning_rate * values[node]`, the only surviving
multiply adjacent to an add among the three arms. It is the only arm that
differs. The two arms that multiply on the host and hand the kernel a
finished Float32 agree with each other and with the unfused host answer
exactly.

The cause is a build default this repository already documents. `pixi.toml`
records that Mojo's `--fp-mode` defaults to `contract=fast`, which "fuses
`a + b*c` into an FMA across statements". Metal's compiler declined that
fusion; NVPTX takes it. The code's own justification for believing the
per-leaf arm was safe is at `gpu_objectives_native.mojo:720-724` and
`:862-878`, and it is explicitly an empirical observation about **one
device compiler**: that the product is "uniform and gets hoisted and rounded
on its own". Thread-uniformity is not a reason an LLVM-based NVPTX backend
would decline to emit `fma.rn.f32`, and it did not decline.

Three things follow, and the third is the one that matters.

1. This is **not** contention damage. The producing run was the invalid
   254-job one, but the failure is not an artifact of it: the disagreement
   is a deterministic 1-ulp arithmetic difference on 1841 and 2225 rows
   respectively, the host-only half of the same file passed, and a starved
   process does not produce self-consistent ULP accounting.
2. The magnitude is 1 ulp, so no result here is "wrong" in any absolute
   sense. Both roundings are defensible IEEE outcomes.
3. **But bit-identity is the property this project's determinism and
   host-replica guarantees are built on**, and the arms are asserted equal
   precisely because a model must not depend on which arm ran. On CUDA they
   are not equal, so that guarantee does not currently hold on NVIDIA.

The fix that the codebase has already applied twice elsewhere is to move the
multiply to the host so the kernel has nothing to fuse
(`_update_raw_kernel` had exactly this done to it), or to write the
contraction explicitly as `fma`, as `gpu_split_search.mojo:851-874` does.
Both are portable changes rather than vendor branches. Neither is applied
here, and applying one is not this document's call.

#### 2b. The open question: 36 tests do not finish

> **SUPERSEDED 2026-08-19 by section 2f. Kept for the method, not the
> counts.** This section's numbers came from a suite run whose pool
> over-subscribed the GPU and whose per-test cap sat below the compile time,
> so its `TIMEOUT` rows conflate three different things. A later run with
> both faults fixed put 66 assertions in the pass column and left exactly one
> class of test hanging. Read 2f for the counts and this section for how the
> two obvious explanations were killed.

With a working per-test timeout, the GPU suite at `MOJOTREES_TEST_JOBS=12`
reported:

```text
13 ok      (including real device work: 1892ms, 1505ms, 961ms, 867ms)
 2+ FAIL   test_gpu_objectives, test_gpu_tree_resident,
           test_host_replica, test_gpu_tree_tables
36 TIMEOUT killed at 300s, no result reported
```

**Neither of the two obvious explanations survived contact with evidence, and
both are recorded here so nobody re-runs them.**

*Rejected: CPU oversubscription from a mis-read core count.* The container
advertises 256 CPUs (`nproc`, `cpuset.cpus.effective = 0-255`) while its
cgroup quota is `cpu.max = 2720000 100000`, i.e. **27.2 CPUs** — a 9.4x gap
that would plausibly wedge a barrier-synchronised pool. It does not explain
this: pinning `MOJOTREES_NUM_WORKERS=8` changed nothing, both arms timed out
identically.

*Rejected, 2026-08-18: slow CUDA codegen rather than a hang.* The reasoning
below was sound when written and the experiment that settled it is in section
2d: `mojo build` compiles a hanging test in 31 seconds and the resulting
binary, with no compiler anywhere in the process, hangs. It is not
compilation. The original text follows because the reasoning was reasonable
and the shape of the mistake is worth keeping.

Both arms above were
still emitting **compiler diagnostics** when the cap fired, not runtime
output. `test_gpu_auto_reaches_gpu` spends 56s wall for 867ms of test time,
so ~55s is compilation, and a heavier GPU test plausibly needs more than
300s of NVPTX codegen on a host throttled to 27 effective cores. If that is
what this is, these are not failures at all — they are a cap set below the
compile time. A single test run under a 900s cap would settle it and has not
completed yet.

Until one of those is closed, **no count above should be quoted as a CUDA
correctness result.** A `TIMEOUT` here means "we do not know", not "it is
broken".

Both suite runs were also terminated by an external 60-minute kill
(`RC=137`), so the counts are lower bounds on a truncated run, not totals.

#### 2c. What the deadlock is NOT, established by experiment

Six explanations were tested on the RTX 5090 and six were eliminated. They are
recorded so the next lane spends its money on something else.

| explanation | how it died |
|---|---|
| Job contention | 32 timeouts at `jobs=4` against 26 at `jobs=12`. Fewer jobs made it worse. |
| CPU oversubscription | The container advertises 256 CPUs against a `cpu.max` quota of 27.2. Real, and fixed, and **not this**: the suite is unchanged with the cgroup clamp in place. |
| Slow NVPTX codegen | The heartbeat shows processes burning 18-30% CPU and then dropping to **zero ticks** with threads in `futex_wait_queue`. A compiler burns CPU until it finishes. |
| Stale readback | The unsynchronized copy was a real defect and is fixed. The deadlock is unchanged. |
| Resident grower queue depth | `MOJOTREES_GPU_TREE_RESIDENT=0` timed out identically to the default on **two** tests (`test_gpu_tree_resident`, `test_gpu_training`), both `rc=124` at ~400s. |
| **MAX's allocator, launches, or queue depth** | **Two MAX-only probes pass.** See below. |

The last one is the load-bearing result. `probes/cuda_deadlock_alloc.mojo` and
`probes/cuda_deadlock_launch.mojo` import **nothing** from this package, and
between them they perform 8,000 device allocations, pinned host allocations,
copies with no drain between, 3,000 kernel launches on a reused buffer, 3,000
allocate-then-launch cycles, and 3,000 allocations plus launches with **no
`synchronize()` at all** until the end. Every phase completes:

```text
PHASE_A_OK  PHASE_B_OK  PHASE_C_OK  REPRO_COMPLETED_NO_DEADLOCK
PHASE_D_OK  PHASE_E_OK  PHASE_F_OK  REPRO2_COMPLETED_NO_DEADLOCK
```

So the runtime primitives are sound on this device and the deadlock requires
something this repository does that they do not cover. It is ours.

#### 2d. The trigger is Modular's kernel JIT, and it is not histogram kernel count

**Located, 2026-08-18.** `mojo build` compiles a hanging test in **31 seconds**
and the resulting **binary** then hangs, so this is not the Mojo compiler and
not the toolchain. A backtrace of that binary, with no compiler in the
process, names the component:

```text
#5-7  libKGENCompilerRTShared.so     <- Modular's kernel-compiler RUNTIME
#1-3  libcuda.so.1                   <- the NVIDIA driver
#0    __futex_abstimed_wait_common64 <- blocked
```

14 threads on futexes, 3 in `poll`, GPU idle, zero CPU. `~/.nv/ComputeCache`
holds **zero files** after a run, so nothing is cached and every process
recompiles from scratch.

**Histogram kernel count is not the trigger.** *(This paragraph read "Kernel
count is NOT the trigger" until 2026-08-19. The experiment is unchanged and
the conclusion drawn from it was too broad: one family was reduced, not the
process's whole kernel population. See section 2g, candidate 2.)*
`MOJOTREES_KERNEL_MATRIX_FULL = False`
collapses the two histogram families from 40 instantiations to 12. Rebuilt at
that setting, `test_gpu_training` and `test_device` both still hang at a 600s
cap (rc=124, 600s and 601s). The matrix reduction was built precisely so this
could be falsified rather than assumed, and it was falsified.

**A methodological correction that invalidates earlier stacks.** The binary
forks: the parent sits in `sigsuspend` and the child wedges on a futex. Every
backtrace taken before 2026-08-18 23:00 was of the PARENT, which is merely
waiting and is not stuck. Any conclusion drawn from those -- including the
earlier reading that the deadlock was in `enqueueCreateBuffer` -- is
unreliable. The child stack, captured with `follow-fork-mode child`, is what
the frames above show.

#### 2e. What is left, and it is upstream

Eleven explanations are eliminated. Four MAX-only probes covering allocation,
pinned buffers, copies, launches, queue depth, sub-buffers and concurrent
device access from eight workers all pass. Reducing the kernel matrix changes
nothing. What remains is a hang inside Modular's kernel-compiler runtime
against the CUDA driver, and this repository cannot reach inside either.

The next action is a bug report, not another experiment here. It has what one
needs: a compiled binary that reproduces, a stack naming the component, exact
versions (Mojo 1.0.0, driver 580.159.03, RTX 5090 sm_120), and four probes
proving the primitives are individually sound.

**That report is written**, at
[`docs/UPSTREAM_MAX_CUDA_HANG.md`](UPSTREAM_MAX_CUDA_HANG.md), and it has not
been filed anywhere. It states its own confidence separately from its
evidence, and it is honest about the two things this record does not have: no
minimal reproducer, and two untested candidates on our own side.

#### 2f. The definitive run: what works and what blocks

**2026-08-18, commit 542962c, RTX 5090.** Every earlier count in this record
came from a suite run with one or both of two harness faults in it: a pool
that over-subscribed the GPU by two orders of magnitude, and a per-test cap
set below the NVPTX compile time. Both were fixed. This run has neither, uses
a 240 s cap per test, and runs the suites one at a time.

```text
--- WHAT WORKS: known-good tests, 240s cap each ---
test_gpu_scan_primitives     rc=0 secs=8   6 tests run: 6 passed
test_gpu_partition_launches  rc=0 secs=10  8 tests run: 8 passed
test_gpu_runtime             rc=0 secs=28  40 tests run: 40 passed
test_gpu_raw_update_packing  rc=0 secs=10  2 tests run: 2 passed
test_gpu_tiling              rc=0 secs=4   10 tests run: 10 passed
--- WHAT BLOCKS ---
test_gpu_training            rc=124 secs=241
```

**66 assertions pass on NVIDIA hardware.** That is the number to quote, and
it is not a formality: `test_gpu_raw_update_packing` is the test that caught
the FMA contraction defect the same morning, so the set has teeth.

One class of test wedges. It is the class that drives a full training fit,
which is also the class that pushes the most distinct kernel instantiations
through Modular's JIT in a single process.

#### 2g. The two candidates that remain, and both are cheap

Every candidate the earlier revision of this section listed has since been
probed and cleared. `create_sub_buffer` is covered by
`probes/cuda_deadlock_subbuffer.mojo`; device access from inside
`sync_parallelize` is covered by `probes/cuda_deadlock_parallel.mojo`, which
shares one `DeviceContext` across eight workers doing allocation, launch and
windowed writes. Both pass. Two candidates survive and neither has been run.

1. **The 48 KiB shared-memory path.** This device reports
   `max_shared_memory_per_block = 49152` against the M4's 32768, so feature
   group 16 at 256 bins **raises on every Mac and is accepted here**. That is
   code no machine in this project's history has executed, and it targets the
   histogram arm the hanging tests drive. Against it: an over-large shared
   memory request normally fails a launch rather than parking a thread on a
   futex. It is still the cheapest thing left.

2. **Kernel variety, as opposed to kernel count in one family.** Section 2d
   records that collapsing the histogram families from 40 instantiations to
   12 changed nothing, and that was read at the time as killing the volume
   hypothesis. It is narrower than that: one family was reduced, and a
   process that JITs hundreds would not be expected to notice 28 fewer. With
   `~/.nv/ComputeCache` empty after every run, every instantiation is being
   compiled from scratch. A probe that JITs N distinct kernels in one process
   and sweeps N would settle it, and no existing probe varies the kernel at
   all.

Each is a small addition to an existing probe and each returns a yes or a no
in minutes on leased hardware.

#### 3. Determinism

**Not run.** Repeat-run bit-identity was never attempted on this device. Every
attempt was consumed by the unresolved item above: the harness that would
measure it is one of the things that does not finish. This remains the single
biggest hole in this record.

#### 4. Phase timings

**Partial.** One shape completed.

```text
== shape: 10000 rows x 20 features ==   n_bins 255, rounds 20
  binning_s              0.009621961
  setup_s                0.671638688
  strategy               tiled
  grid_dim               20 x 5          block_dim 256
  rows_per_tile          2000            partial_cells 25500
  stage_s                0.000030966
  grad_h2d_s             0.000010296
  hist_kernel_root_s     0.000152939
  hist_d2h_s             0.000007812
  hist_decode_s          0.000059369
  partition_kernel_s     0.000112909
  hist_kernel_child_s    0.000025298
  hist_cells 5100        hist_child_rows 5019
```

Device capabilities as reported, which are the numbers the tiling policy
keys on:

```text
  multiprocessor_count             170
  max_threads_per_block           1024
  max_threads_per_multiprocessor  1536
  max_blocks_per_multiprocessor     24
  max_shared_memory_per_block    49152
  max_registers_per_block        65536
  max_grid_dim_x            2147483647
  max_grid_dim_y                 65535
  clock_rate_khz               2407000
```

#### The result that matters most

```text
  blocks_per_sm_at_launch          0.5882352941176471
  shared_memory_resident_block_limit   16
  global_atomics_per_launch_upper       0
```

**The launch fills less than six tenths of one block per SM on a 170-SM
device.** The dataset is too small for the part. This is NOT a geometry
defect, and an earlier revision of this record said it was. The correction
is kept in full below because the mistake is more instructive than the
number.

**What actually happened, arithmetically.** The policy did scale with the
device. `target_blocks_for` (`gpu_tiling.mojo:851`) is
`sm_count * blocks_per_sm_for(...)`, which on this part is `170 * 8 = 1360`,
seventeen times the M4's 80. It asked for 1360 blocks and got 100. The
binding term was `tiles_by_rows` (`gpu_tiling.mojo:582`):

```text
min_rows_per_tile = max(8 * 255 bins, 4 * 256 threads) = 2040
tiles_by_rows     = ceil(10000 / 2040)                 = 5
by_occupancy      = ceil(1360 / 20 features)           = 68
n_tiles           = min(68, 5)                         = 5
grid_dim          = 20 x 5 = 100 blocks / 170 SMs      = 0.588
```

The device asked for 68 tiles per feature and the row supply granted 5. The
term that clamped it, `MIN_ROWS_PER_TILE_BIN_FACTOR = 8`
(`gpu_tiling.mojo:147`) times the dataset's own 255 bins, **contains no
device input whatsoever**. Multiprocessor count, shared memory, the partial
budget and `MAX_GRID_DIM_Y` all bound nothing here.

**The shape was labeled as this case before the run.** The default sweep in
`bench_gpu_validation.mojo:394-403` describes its four shapes as "Small
(launch-overhead bound), square, wide, and tall". 10,000 x 20 is the one
designated launch-overhead bound. It is the corner built to starve, and it
is the only shape that completed.

The same arithmetic on the other three, on these caps, predicts they fill
the device with no code change: 100,000 x 100 gives 8.24 blocks/SM,
50,000 x 400 gives 9.41, and 1,000,000 x 20 gives 8.00. Those are
predictions from the shipped rule and are **not yet measured**; measuring
them is the cheapest way to close this question.

Note also that 8 blocks/SM is arithmetically incoherent at 10,000 rows: it
would need 68 tiles, so 148 rows per tile against a 256-wide block, leaving
40% of every block's lanes with no row at all. Under the current rules the
device-wide target first becomes reachable at roughly **137k rows** at 20
features, one block per SM at about **16.3k**.

For contrast, this identical shape fills the M4 exactly: `ceil(80/20) = 4`
tiles, 80 blocks on 10 cores, 8.0 blocks per core. The device grew 17x and
the dataset did not.

#### The finding that survived, and it points the other way

```text
  max_threads_per_multiprocessor  1536
  max_blocks_per_multiprocessor     24
  block_dim                        256
```

At 256 threads this device admits `1536 / 256 = 6` co-resident blocks per
SM. `TARGET_BLOCKS_PER_SM = 8` (`gpu_tiling.mojo:130`) therefore asks for
**33% more blocks than one wave can hold**. The correct rule already exists
in this repository, at `gpu_vendor_policy.resident_blocks_from_reported`
(`gpu_vendor_policy.mojo:460-524`), which would return `min(24, 6) = 6` and
a target of 1020. It is gated off behind `SUPPORT_PORTABLE`
(`gpu_backend_policy.mojo:95-109`) and did not run.

This is the safe direction of change: it makes the target *smaller*, it
reads reported device attributes rather than branching on a vendor, and it
is the portable-change-first path this document already prefers over a
branch.

#### A reporting defect in this harness, not in the library

```text
  shared_memory_resident_block_limit   16      <- overstates by 2.7x
```

`bench_gpu_validation.mojo:222-223` computes that as
`49152 / 3072`, from shared memory alone. The device also reported a
thread-slot limiter of 6, which the bench ignored. The real residency bound
at this block width is 6, not 16. Any reader of this record would otherwise
conclude there was 16-block headroom that does not exist.

Similarly, `global_atomics_per_launch_upper: 0` is a **definition, not a
measurement**. `bench_gpu_validation.mojo:236-242` sets it to zero whenever
the tiled strategy is selected. It records which branch was taken, not what
the device did, and no profiler counter was collected to confirm it.

#### What this does not justify

No tuning change. The one lever that would raise occupancy here is lowering
`MIN_ROWS_PER_TILE_BIN_FACTOR`, and deleting it entirely would buy 200
blocks instead of 100, taking 0.59 to 1.18 blocks/SM on a 170-SM device
while doubling how often each tile pays a 255-bin zero and flush. The only
measurement this repository holds on that exact trade
(`gpu_tiling.mojo:743-769`) says raising the tile count lost 22 to 36
percent and concludes the factor is if anything too *low*. The available
lever has been measured pointing the wrong way.

#### 5. Profiler

**Not run.**

#### What is not established

State plainly, because a partial record is easy to over-read:

- Determinism, on any shape. Not attempted.
- Correctness, while `test_gpu_raw_update_packing` is open.
- Any comparison against the Apple M4 numbers elsewhere in this repository.
  Different vendor, different architecture, different host CPU (x86-64 with
  256 threads against 10-core ARM). Nothing here is a like-for-like speed
  claim and no number here should be placed beside an M4 number.
- Anything at all about AMD.

#### Two defects this run found in the harness, not in the library

Both were found by running on a machine unlike the development laptop, which
is the point of doing this.

1. **`pixi run build-pkg` failed on a fresh clone.** `build/` is gitignored
   and `mojo precompile` will not create its output directory, so the task
   only ever worked on a machine where an earlier run had left the directory
   behind. Every development machine here had one. Fixed in `pixi.toml` by
   `mkdir -p build &&`. `tools/run_tests.sh` was never affected; it already
   did this at its own precompile site.

2. **The test pool over-subscribed the GPU by two orders of magnitude.**
   `tools/run_tests.sh` sized its pool as `NCPU - 2`, which is 8 on the M4
   and **254** on this host, so 254 processes each opened a device context
   against one card. Measured consequence: GPU utilization 0% with 7.1 GB of
   VRAM consumed by contexts alone, one test killed at 503 s by MAX's own
   watchdog reporting a bare `Alarm clock`, and no forward progress. This is
   what invalidated the first correctness run. Fixed by clamping the default
   in `gpu` mode via `GPU_MAX_JOBS`, and by giving `run_one` an actual
   wall-clock timeout, which it never had.

### Failures and unsupported capabilities

Record these even when everything else passes. A backend that answers no
capability query still works, and knowing which queries it refuses is how
`gpu_tiling.mojo` gets its fallbacks right.

Known already, from reading rather than running:

- **Metal refuses six of the eleven device attributes.** Measured, not
  assumed: `WARP_SIZE`, `MAX_THREADS_PER_MULTIPROCESSOR`,
  `MAX_BLOCKS_PER_MULTIPROCESSOR`, `MAX_REGISTERS_PER_BLOCK`, and
  `CLOCK_RATE` all raise, and `SHARED_MEMORY_PER_MULTIPROCESSOR` is not a
  member of `DeviceAttribute` at all. `gpu_tiling.mojo` falls back to
  portable constants when a query raises, and `bench_gpu_validation.mojo`
  prints `unavailable` rather than failing. Run every backend through the
  report to find its own refusals; the fallbacks are only correct as long as
  we know which ones fire.
- **No Float64 on device.** Gradients and hessians are carried as Float32
  because Apple GPUs have no Float64. NVIDIA and AMD both have it, so this is
  a portability floor, not a hardware limit, and it is the reason CPU and GPU
  models agree to Float32 tolerance rather than bit-exactly. Raising it for
  CUDA and HIP only would be a vendor branch and needs the justification
  below.
- **Shared memory is reserved and budgeted by the same bin capacity.** This
  entry used to record a gap: the kernels reserved three 256-long Int32
  planes whatever the dataset's bin count while `gpu_tiling.shared_bytes_for`
  budgeted `n_bins * 12`, so on a device whose per-threadgroup shared memory
  fell between the two the policy would accept a shape the launch could not
  satisfy. The gap is closed. Both are now
  `histogram_bin_capacity(n_bins) * 12` per feature slot. A new and narrower
  gap replaces it: both price one slot, and a threadgroup owning a group of G
  occupies G times as much, so the model under-reports at any group past 1. No supported backend is anywhere near that range (the smallest
  guaranteed allocation is 16 KiB), so this is latent rather than live, and
  `tests/test_gpu_portability.mojo` checks the reservation against the
  portable floor so it stays that way. Sizing the reservation to `n_bins`
  would close it and buy occupancy at the same time.
- **`grid.y` and CUDA's 65535 cap.** Row tiles land in `grid.y`, which CUDA
  caps at 65535 while HIP and Metal allow more. `gpu_tiling.derive_tiling`
  clamps to the CUDA number on every backend, so no row count the builder
  accepts can exceed it. `tests/test_gpu_portability.mojo` checks that up to
  the largest row count the kernels can index. Recorded here because the
  clamp is the only thing preventing it, and a tiling change could remove it.
- **Fixed-point row ceiling.** Scaled magnitudes target 2^30 and rounding
  adds up to half a unit per row, so the worst-case accumulator stays inside
  Int32 up to roughly 2.1 billion rows. Determinism rests on this: once
  accumulation wraps, integer addition stops being order independent. Also
  pinned by that test file.

## Before adding device-specific tuning

The rule for this repository: **no vendor branch without a benchmark and a
profiler trace from the device it targets.**

A vendor branch is any `comptime` switch on architecture, any per-chip
constant, any second kernel kept for one backend. `gpu_tiling.mojo` already
shows the preferred alternative, which is reading a device attribute at
runtime and deriving the geometry from it. A runtime policy that happens to
choose different numbers on different hardware is not a vendor branch. It is
the whole point.

Before a branch is justified, all four:

1. A phase breakdown from `pixi run gpu-validate` on the target device
   showing which phase is the problem.
2. A profiler trace showing the mechanism, not just the symptom. "Slow" is
   not a mechanism. "Occupancy capped at two blocks per SM by static shared
   memory" is.
3. A portable change tried first and measured. Most findings here have a
   portable fix, and a portable fix that helps one device and does not hurt
   the others is strictly better than a branch.
4. Numbers from at least two devices, so the constant being special-cased is
   demonstrably device-specific rather than a property of the workload that
   happened to be measured once.

The `MOJOTREES_GPU_HIST_STRATEGY`, `MOJOTREES_GPU_ROW_TILE`, and
`MOJOTREES_GPU_BLOCK_THREADS` environment overrides exist so a sweep can be
run without editing or rebuilding anything. Use them to find the crossover
before hard-coding one.

## CI

`.github/workflows/gpu-validation.yml` runs everything above on
`workflow_dispatch`, one job per vendor, and uploads the report as an
artifact. It targets self-hosted runners:

```text
[self-hosted, linux, gpu, cuda]
[self-hosted, linux, gpu, rocm]
```

Neither is registered, so neither job has ever run. To register one, on a
machine with a working driver:

```sh
# From the repository's Settings, Actions, Runners, New self-hosted runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner.tar.gz -L <url from that page>
tar xzf actions-runner.tar.gz
./config.sh --url https://github.com/mojotrees/mojotrees \
            --token <token from that page> \
            --labels self-hosted,linux,gpu,cuda   # or ...,gpu,rocm
./run.sh
```

Then dispatch **GPU validation** from the Actions tab and paste the artifact
into the record section above.

A leased cloud instance does not need to become a registered runner. Running
the procedure by hand and pasting the output is worth the same, and for a
one-off validation it is less work.

The regular `ci.yml` workflow stays CPU-only and does not depend on any of
this. `tests/test_gpu_portability.mojo` runs there, on every push, without a
GPU, because the launch limits and numeric bounds it checks are host-side
arithmetic. It is what catches a tuning change that would break CUDA or HIP
while still passing on Metal.
