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
| CUDA | none available | **not run** | **not run** | **not run** | **not run** |
| HIP | none available | **not run** | **not run** | **not run** | **not run** |

No NVIDIA or AMD hardware has executed this code. Not once, not on a laptop,
not in CI. The development machine is an Apple M4, GitHub-hosted runners have
no GPU, and no self-hosted runner is registered. Every CUDA and HIP row above
stays **not run** until someone executes the procedure below and pastes real
output into the record section.

Nothing in this repository should be read as a claim about NVIDIA or AMD
behavior or performance. The Metal "partial" for phase timings means the
harness runs and prints, not that a recorded sweep exists.

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
