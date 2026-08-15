# Accelerator record template: NVIDIA, CUDA

Copy this file, fill every field from output you watched print, and append the
result to the record section of `docs/GPU_VALIDATION.md`. Then, and only then,
change the row in `packaging/matrix/accelerators/index.toml`.

No NVIDIA device has ever executed this code. The first record filed against
this template is the first NVIDIA evidence that exists, so it is worth taking
the extra hour to fill in the profiler section rather than leaving the repo
with a correctness-only claim.

Rules that make a record worth having:

- Paste verbatim. A field you did not run says `not run`.
- Correctness gates timings. A failure in the equivalence or determinism suite
  stops the run; nothing downstream is recorded.
- A GPU suite that prints `skipped: no accelerator` is not a pass. On a machine
  with a GPU it means the build did not see the device.
- The CUDA **driver** is required. The CUDA toolkit is not, except for the
  profilers in step 4.

## The record

```text
### NVIDIA <board>, CUDA, <YYYY-MM-DD>

Board:         <nvidia-smi --query-gpu=name --format=csv,noheader>
Driver:        <nvidia-smi --query-gpu=driver_version --format=csv,noheader>
Memory:        <nvidia-smi --query-gpu=memory.total --format=csv,noheader>
Compute cap:   <nvidia-smi --query-gpu=compute_cap --format=csv,noheader>
nvcc:          <nvcc --version, or "not installed (not required)">
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

Profiler (ncu / nsys):
  Achieved occupancy          <sm__warps_active.avg.pct_of_peak_sustained_active>
  Occupancy limiter           <launch__occupancy_limit_shared_mem vs register and block limiters>
  Static shared memory        <launch__shared_mem_per_block_static>
  Shared atomic conflicts     <l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom.sum>
  Global atomic traffic       <lts__t_sectors_op_atom.sum>
  Kernel vs transfer          <what the nsys timeline shows>

Failures:      <anything that did not build, run, or agree>
Unsupported:   <capabilities the backend does not implement>
```

Confirm the metric names against `ncu --query-metrics` on the installed
version before trusting the ones above; they drift between releases.

## NVIDIA-specific things to check while you are there

- **`max_grid_dim_y` and the 65535 clamp.** Row tiles land in `grid.y`, and
  CUDA is the backend the 65535 cap comes from. `gpu_tiling.derive_tiling`
  clamps on every backend because of it. Record what this device reports and
  confirm the clamp is doing what the portability test says it does.
- **Determinism is the interesting result.** Fixed-point Int32 accumulation is
  what buys bit-identical repeats, and a backend whose atomics or scheduling
  differ from Metal's is exactly where that property would break. This is the
  single most valuable line in an NVIDIA record.
- **Float64 exists here and does not on Metal.** Resist using it. Raising
  gradients to Float64 for CUDA only is a vendor branch and needs the four
  pieces of evidence listed under "Before adding device-specific tuning" in
  `docs/GPU_VALIDATION.md`.
- **Attribute answers.** Metal refuses six of eleven queries. CUDA is expected
  to answer all of them, which means the tiling policy takes a different path
  here than in every measurement taken so far. Say so explicitly in the record;
  a timing comparison against the M4 numbers is comparing two different derived
  geometries, not two devices running the same launch.
- **No wheel angle.** There is no Linux wheel and no CUDA-specific artifact.
  A device record changes `packaging/matrix/accelerators/index.toml` and
  nothing in the target table.
