# Accelerator record template: Apple silicon, Metal

Copy this file, fill every field from output you watched print, and append the
result to the record section of `docs/GPU_VALIDATION.md`. Then, and only then,
change the row in `packaging/matrix/accelerators/index.toml`.

Rules that make a record worth having:

- Paste verbatim. Do not summarize, do not round, do not retype a number from
  memory. A field you did not run says `not run`, never a guess.
- Correctness gates timings. If step 3 fails, stop and record the failure; no
  throughput number is recorded from a device that computes the wrong answer.
- A GPU suite that prints `skipped: no accelerator` is not a pass. On a machine
  with a GPU it means the build did not see the device, which is a finding in
  its own right and belongs in **Failures**.
- `unavailable` is a result. Metal refuses six of the eleven device attributes
  on the recorded M4; which six it refuses on your chip is exactly what the
  tiling fallbacks depend on.

## Before you run anything: the Metal toolchain

Apple silicon reporting a GPU is not the same as the machine being able to
build GPU code. The compile-time `has_accelerator()` query sees the device; the
Metal compiler is a separate installation, and without it the GPU-touching
targets fail to build. This is why `.github/workflows/ci.yml` has no macOS
runner: a GitHub-hosted Apple silicon runner reports an accelerator and has no
Metal toolchain, so the GPU equivalence test cannot build there.

Record what the machine actually has, before the results:

```text
xcode-select -p                   <path, or the error>
xcrun --find metal                <path, or "not found">
xcrun metal --version             <version, or the error, verbatim>
xcodebuild -version               <version and build, or "not installed">
```

If `xcrun --find metal` fails, the Metal compiler component is missing.
Installing it is an Xcode operation, not a mojotrees one, and the exact
component name has moved between Xcode releases, so confirm it against the
Xcode version on the box rather than against this file. Record the fix you
used, so the next person on the same macOS version does not rediscover it.

A missing Metal toolchain is a legitimate record. Fill in the environment
block, write `build failed, Metal toolchain absent` under **Failures**, and
stop. That result is worth publishing: it is the difference between "the GPU
path is broken on this chip" and "this machine could never have built it".

## The record

```text
### Apple <chip>, Metal, <YYYY-MM-DD>

Chip:          <e.g. Apple M4 Pro>
GPU cores:     <count as Apple states it for this configuration>
Machine:       <model identifier>
Memory:        <unified memory size>
macOS:         <sw_vers -productVersion>, <uname -m>
Mojo:          <pixi run mojo --version>
MAX:           <pixi list --environment default | grep -Ei '^(mojo|max)'>
Xcode:         <xcodebuild -version>
metal:         <xcrun metal --version>

pixi run test                            <pass | fail, with the failing assertion>
pixi run test-gpu                        <pass | fail>
MOJOTREES_DISABLE_GPU=1 pixi run test    <pass | fail>

Device header, verbatim from `pixi run gpu-validate`:

name:
api:
arch_name:
compute_capability:
multiprocessor_count:
warp_size:
max_threads_per_block:
max_threads_per_multiprocessor:
max_blocks_per_multiprocessor:
max_shared_memory_per_block:
max_registers_per_block:
max_grid_dim_x:
max_grid_dim_y:
clock_rate_khz:

Attributes this backend refused:
<list every query that raised, by name>

<paste the full gpu-validate sweep, every shape, with training MSE next to
every timing>

Strategy pair (the two must produce bit-identical histograms):
MOJOTREES_GPU_HIST_STRATEGY=atomic   <output>
MOJOTREES_GPU_HIST_STRATEGY=tiled    <output>

Profiler:      <not run, or the Metal capture: occupancy, threadgroup memory
                limiter, and where kernel time went>
Failures:      <anything that did not build, run, or agree>
Unsupported:   <capabilities the backend does not implement>
```

## Apple-specific things to check while you are there

Each of these is a prediction the code makes. The point of the run is to find
out whether it is true on this chip.

- **Shared memory per threadgroup.** The kernels reserve `3 * MAX_BINS * 4` =
  3072 bytes per launch whatever the dataset's bin count. The recorded M4
  reports 32768 available. Record what this chip reports; the portable floor
  the code assumes is 16 KiB, and `tests/test_gpu_portability.mojo` pins the
  reservation against it.
- **`max_grid_dim_y`.** The recorded M4 reports the full 2^31 - 1, which is why
  the CUDA 65535 clamp is known to be CUDA's constraint alone. Confirm this
  chip agrees before anyone reasons about the clamp.
- **No Float64 on device.** Gradients and hessians are Float32 because Apple
  GPUs have no Float64. This is why CPU and GPU models agree to a Float32
  tolerance rather than bit-exactly, and it is not a bug to be recorded as one.
- **`warp_size` unavailable.** The tiling policy's fallback fires here on the
  recorded M4. If this chip answers the query, say so: it changes which code
  path derived the launch geometry, and therefore what the timings mean.
- **The wheel angle.** If you are on the machine that builds release wheels,
  record `otool -l python/mojotrees/_mojotrees.so | grep -A 4 LC_BUILD_VERSION`
  in the same session. The build's `minos` is what the wheel's platform tag has
  to match, and it is the single number the macOS install story turns on.
