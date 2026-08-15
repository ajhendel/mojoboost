# Accelerator record template: AMD, HIP

Copy this file, fill every field from output you watched print, and append the
result to the record section of `docs/GPU_VALIDATION.md`. Then, and only then,
change the row in `packaging/matrix/accelerators/index.toml`.

No AMD device has ever executed this code.

Rules that make a record worth having:

- Paste verbatim. A field you did not run says `not run`.
- Correctness gates timings.
- A GPU suite that prints `skipped: no accelerator` is not a pass. On a machine
  with a GPU it means the build did not see the device, which on ROCm usually
  means the runtime is installed but the user is not in the render group, or
  the card is not in the supported-target list of the installed ROCm.
- Record the architecture, not just the board name. `gfx1100` and `gfx942` are
  different machines wearing the same vendor logo, and a result from consumer
  RDNA transfers to datacenter CDNA about as well as a result from Metal does.

## The record

```text
### AMD <board> (<gfx target>), HIP, <YYYY-MM-DD>

Board:         <rocm-smi --showproductname, or the rocminfo name line>
gfx target:    <rocminfo | grep -i gfx>
Compute units: <rocminfo | grep -i "compute unit">
Memory:        <rocm-smi --showmeminfo vram>
ROCm:          <cat /opt/rocm/.info/version>
Driver:        <rocm-smi --showdriverversion, or the amdgpu module version>
hipcc:         <hipcc --version, or "not installed (not required)">
Instance:      <cloud instance type, or "bare metal">
Host CPU:      <lscpu model name>, <core count>
OS:            <uname -a>
Mojo:          <pixi run mojo --version>
MAX:           <pixi list --environment default | grep -Ei '^(mojo|max)'>

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
<list every query that raised, by name, or "none">

CPU threading pinned for the comparison:
MOJOTREES_NUM_WORKERS=<value>
MOJOTREES_PARALLEL_MIN_OPS=<value>

<paste the full gpu-validate sweep, every shape, with training MSE next to
every timing>

Strategy pair (the two must produce bit-identical histograms):
MOJOTREES_GPU_HIST_STRATEGY=atomic   <output>
MOJOTREES_GPU_HIST_STRATEGY=tiled    <output>

Profiler (rocprofv3 / ROCm Compute Profiler):
  Wave occupancy              <occupancy panel>
  Occupancy limiter           <LDS allocation granularity vs VGPR count>
  LDS per workgroup           <allocated bytes>
  LDS bank conflicts          <SQ_LDS_BANK_CONFLICT, or the current name>
  L2/TCC atomic requests      <counter and value>
  Kernel vs copy              <rocprofv3 kernel and memory-copy rows>

Failures:      <anything that did not build, run, or agree>
Unsupported:   <capabilities the backend does not implement>
```

Confirm the counter names against `rocprofv3 --list-avail` on the installed
version before trusting the ones above.

## AMD-specific things to check while you are there

- **Wavefront width.** 64 on CDNA, 32 on RDNA. The tiling policy rounds threads
  per group to a warp, so a device that answers `warp_size` differently from
  the recorded Metal fallback derives a different launch geometry. Record what
  it answered and whether the fallback fired.
- **LDS size and the 3072-byte reservation.** The kernels reserve three 256-long
  Int32 planes per launch regardless of the dataset's bin count. Record
  `max_shared_memory_per_block` and whether LDS is what caps occupancy; if it
  is, sizing the reservation to the actual bin count is the portable fix and it
  helps every backend, which is the kind of change this project prefers.
- **Determinism across a different atomic implementation.** Same reasoning as
  the NVIDIA template, more so: LDS integer atomics on RDNA and CDNA are the
  most plausible place for bit-identical repeats to stop being bit-identical.
- **Support-list refusal is a result.** If the installed ROCm does not list the
  card as a supported target, the build fails and that belongs in **Failures**
  with the exact error, plus the ROCm version that refused it. Do not work
  around it with `HSA_OVERRIDE_GFX_VERSION` and then record the run as a
  validation; an overridden target is a different device from the one the
  record names, and if you do run one anyway, say so on its own line.
- **No wheel angle.** There is no Linux wheel and no HIP-specific artifact.
