# NVIDIA GPUs

Written: 2026-08-14

**No NVIDIA device has ever executed a line of this code.** Not in CI, not on
a workstation, not once. `docs/GPU_VALIDATION.md` is the record and every
CUDA row in it reads "not run". Nothing in this file is a claim about how
mojoboost behaves or performs on NVIDIA hardware.

What this file is: how the portable contract in `docs/GPU_PORTABILITY.md`
reads on CUDA, what to look at first when someone finally runs it, and which
specialization points would apply here and why none is taken. The procedure
for producing the missing evidence is in `docs/GPU_VALIDATION.md` and is not
repeated.

## Support level

`portable-untested`. `gpu_backend_policy.backend_support(API_CUDA)` returns
`SUPPORT_PORTABLE`: the shared source targets CUDA and the contract covers
it, and this repository has run nothing on it.

That level does not block training. The portable kernels compile for
whatever device MAX opens, and a CUDA device will run them. What it blocks
is specialization, through
`gpu_portability.require_specializations_allowed`, and the refusal names
`MOJOBOOST_GPU_BACKEND_UNVALIDATED=1` as the acknowledged override.

## What the contract says here

| Contract field | Value on CUDA | Why |
|---|---|---|
| `subgroup_width` | 0, unknown | Nobody here has read `WARP_SIZE` on a CUDA device. The number everyone knows is 32; it is not written down as a fact this project established, and nothing divides by it |
| `launch_granularity` | 64 | `gpu_tiling.WARP_GRANULARITY`, a multiple of the 32-wide warp. A rounding rule, not a width claim |
| `max_grid_dim_y` | 65535 | CUDA's cap, and the value the whole package is written to, since it is the tightest of the three backends |
| `grid_axes` | 2 | `grid.x` features or leaf slots, `grid.y` row tiles or classes. No `grid.z` anywhere |
| `device_float64_permitted` | false | Apple silicon has no Float64 and one source takes the weakest backend's floor. See below |
| `unified_memory` | false from the API name alone | Must come from a report on this backend. A discrete card and a Jetson cannot be told apart by the string "cuda" |

CUDA is the backend whose limits the portable bounds were taken from, so it
is the one where a portable bound and the device bound should coincide
exactly. If `grid.y` is ever the binding constraint on a workload, it is
binding for a real reason here and not because of a conservative floor.

## The things to look at first

In the order they are most likely to matter, and each is a question for a
profiler rather than a prediction.

1. **Static threadgroup memory and occupancy.** The shipping histogram
   kernels allocate three `MAX_BINS`-wide Int32 planes, 3072 bytes, at every
   bin count. CUDA's static shared-memory limit per block is 48 KiB without
   an opt-in the shared source does not take, so 3 KiB is not close to a
   limit. It is still the first thing to read off a trace, because
   `gpu_tiling.TARGET_BLOCKS_PER_SM` aims for eight resident blocks per
   multiprocessor as a fixed target rather than deriving it from what the
   device reports, and eight blocks times 3 KiB is a number worth checking
   against reality on a part with a large multiprocessor count. If occupancy
   is capped by static shared memory rather than by anything else, the
   bin-capacity specialization below is the portable fix.

2. **Device-memory atomic contention.** `STRATEGY_ATOMIC` folds every
   partial histogram into the output with `Atomic.fetch_add` on Int32.
   `derive_tiling` already prefers `STRATEGY_TILED` whenever there is more
   than one tile to reduce, which is the contention-avoiding path, so the
   atomic strategy is reached mainly on small nodes. Confirm that is what
   happens rather than assuming it: `MOJOBOOST_GPU_HIST_STRATEGY=atomic` and
   `=tiled` force either path for a sweep, and both must produce the
   identical integers.

3. **Determinism.** Fixed-point Int32 accumulation is what makes repeat runs
   bit-identical, and it is the property most likely to break on a backend
   whose atomics or scheduling differ from the one it was developed on. It
   is step 3 of the validation procedure and it gates every timing after it.

4. **Multiprocessor count.** `gpu_tiling.FALLBACK_SM_COUNT` is 16 for a
   device that does not answer the query. A datacenter part reports an order
   of magnitude more, and the tile count derives from it, so an unanswered
   query would silently underfill the device. Check that
   `MULTIPROCESSOR_COUNT` is actually answered before reading any timing.

## Float64, and why it is off

CUDA devices have Float64. The shared source does not use it, anywhere, on
any backend, because Apple silicon has none and there is one source. So
gradients cross to the device as Float32 and histogram accumulation is
fixed-point Int32.

The cost is real and is not hidden: agreement with the CPU builder is to
Float32 precision, not bit-exact. The benefit is that a histogram is exact
and associative and therefore bit-identical run to run, strategy to
strategy, and backend to backend, which is worth more than the precision on
an accumulator that sums into a leaf value anyway.

A Float64 variant for CUDA is a specialization like any other.
`gpu_portability.require_device_float64` is the one place it would be gated
from, and it would have to reproduce the fixed-point integers exactly, which
is a harder requirement than it sounds and is the reason nobody should reach
for it as a first optimization.

## Specialization points that would apply here

All gated, none taken, and the gate is
`gpu_portability.require_specializations_allowed`.

- **Bin-capacity kernels.** The largest and most portable win available, and
  it is not CUDA-specific. A 32-bin histogram currently occupies the same
  3 KiB as a 256-bin one. `gpu_histogram_specializations.bin_capacity_for`
  and `kernel_shared_bytes` already compute what a capacity-parameterized
  kernel would occupy; the kernels do not yet take the parameter.
- **Packed four-byte bin loads.** `plan_packed_window` already computes the
  aligned span of a contiguous run of row ids, and `pack4_bins` is the
  portable implementation. Whether a 32-bit load beats four byte loads is a
  measurement, and none exists on any backend.
- **Subgroup reductions.** Nothing exists. It would need a width read from
  the device, which CUDA does answer, plus a portable implementation
  producing identical integers beside it.

Before any of these becomes a vendor branch, the four-part rule in
`docs/GPU_VALIDATION.md` under "Before adding device-specific tuning"
applies: a phase breakdown, a profiler trace showing a mechanism, a portable
change tried first, and numbers from two devices.

## If it does not build or does not run

That is a result and it belongs in the failures section of
`docs/GPU_VALIDATION.md`, with the driver version, the MAX version, and the
exact error. A build failure on a card MAX does not support is the expected
outcome for that card and is worth recording once so the next person does
not pay for the same instance.

`has_accelerator()` is resolved at compile time, so the GPU path is compiled
in only on a machine that has a device. A wheel built elsewhere will not
grow a CUDA path by being installed on a CUDA box.
