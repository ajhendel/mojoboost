# AMD GPUs

Written: 2026-08-14

**No AMD device has ever executed a line of this code.** Not in CI, not on a
workstation, not once. `docs/GPU_VALIDATION.md` is the record and every HIP
row in it reads "not run". Nothing in this file is a claim about how
mojotrees behaves or performs on AMD hardware.

What this file is: how the portable contract in `docs/GPU_PORTABILITY.md`
reads on HIP, what to look at first when someone finally runs it, and which
specialization points would apply here. The procedure for producing the
missing evidence is in `docs/GPU_VALIDATION.md` and is not repeated.

## Support level

`portable-untested`. `gpu_backend_policy.backend_support(API_HIP)` returns
`SUPPORT_PORTABLE`: the shared source targets HIP and the contract covers
it, and this repository has run nothing on it.

Training is not blocked by that level. Specialization is, through
`gpu_portability.require_specializations_allowed`, whose refusal names
`MOJOTREES_GPU_BACKEND_UNVALIDATED=1` as the acknowledged override.

`parse_api` recognizes both spellings, `hip` and `rocm`, so
`MOJOTREES_GPU_BACKEND=rocm` and `=hip` both name this backend.

## CDNA and RDNA are two targets, not one

The single most important thing on this page. A result from an MI210 or
MI300X does not transfer to an RX 7900 XTX and the reverse is equally
false. They differ in wavefront width, in cache hierarchy, and in what
`MAX_SHARED_MEMORY_PER_BLOCK` reports. `docs/GPU_VALIDATION.md` already says
to record the exact card for this reason; treat "AMD" in any status table as
shorthand for "the one card that was measured".

The wavefront is 64 lanes on CDNA and 32 on RDNA. That is why no subgroup
width is assumed anywhere in this package and why
`gpu_tiling.WARP_GRANULARITY` is 64: it is a multiple of both, so a
threadgroup width rounded to it is launchable on either without knowing
which one is in front of us.

## What the contract says here

| Contract field | Value on HIP | Why |
|---|---|---|
| `subgroup_width` | 0, unknown | 64 on CDNA, 32 on RDNA, and this repository has read the attribute on neither. Nothing divides by it |
| `launch_granularity` | 64 | A multiple of both wavefront widths. A rounding rule, not a width claim |
| `max_grid_dim_y` | 65535 | CUDA's cap. HIP allows more; the tightest of the three backends applies everywhere because one source targets all of them |
| `grid_axes` | 2 | `grid.x` features or leaf slots, `grid.y` row tiles or classes. No `grid.z` anywhere |
| `device_float64_permitted` | false | Apple silicon has no Float64 and one source takes the weakest backend's floor |
| `unified_memory` | false from the API name alone | Must come from a report. An APU with a shared address space and a discrete MI300X cannot be told apart by the string "hip" |

The `grid.y` bound is the one place HIP is left money on the table by the
portable floor. If a workload is ever clamped to 65535 tiles on an AMD
device, that clamp is CUDA's limit being applied to a device that does not
have it. It is a deliberate consequence of one source and would need a
per-backend bound, read from the device rather than assumed, to lift.

## The things to look at first

1. **Threadgroup memory, called LDS here.** The shipping histogram kernels
   allocate three `MAX_BINS`-wide Int32 planes, 3072 bytes, at every bin
   count. AMD parts typically report 64 KiB of LDS per workgroup, so 3 KiB
   is not near a limit, but LDS allocation granularity and the fixed
   eight-blocks-per-multiprocessor target in
   `gpu_tiling.TARGET_BLOCKS_PER_SM` interact in ways worth reading off a
   profiler rather than assuming. If occupancy is capped by LDS, the
   bin-capacity specialization is the portable fix.

2. **Device-memory atomic throughput.** `STRATEGY_ATOMIC` folds partials
   into the output with Int32 atomics; `STRATEGY_TILED` avoids them entirely
   by writing each partial to a slot nothing else writes.
   `derive_tiling` prefers tiled whenever there is more than one tile, so
   the atomic path is reached mainly on small nodes.
   `MOJOTREES_GPU_HIST_STRATEGY=atomic` and `=tiled` force either for a
   sweep, and both must produce the identical integers.

3. **Determinism.** Fixed-point Int32 accumulation is what makes repeat runs
   bit-identical, and it is the property most likely to break on a backend
   whose atomics or scheduling differ from Metal. Step 3 of the validation
   procedure, and it gates every timing after it.

4. **Whether the attribute queries are answered.** `query_device_caps` reads
   `MULTIPROCESSOR_COUNT`, `MAX_THREADS_PER_BLOCK`, and
   `MAX_SHARED_MEMORY_PER_BLOCK`, each falling back to a conservative
   constant when a backend does not implement the query. The fallback
   multiprocessor count is 16. An unanswered query on a 300-compute-unit
   part would silently underfill it, and the symptom is a disappointing
   timing rather than an error, so check the answers before reading any
   number.

5. **ROCm version skew.** `rocm-smi` reporting the device is necessary and
   not sufficient. Record `/opt/rocm/.info/version` alongside the MAX
   version; a mismatch between what MAX was built against and what is
   installed is the most likely cause of a failure that looks like a code
   bug.

## Specialization points that would apply here

All gated, none taken.

- **Bin-capacity kernels.** The largest portable win available and not
  AMD-specific. A 32-bin histogram occupies the same 3 KiB of LDS as a
  256-bin one today.
- **Packed four-byte bin loads.** `plan_packed_window` and `pack4_bins`
  exist as the portable implementation. Whether a 32-bit load beats four
  byte loads on any AMD part is unmeasured.
- **A per-backend `grid.y` bound.** The one specialization on this page that
  is specifically about AMD, and it is a policy change rather than a kernel
  change: read the device's real bound instead of applying CUDA's. It needs
  a device that answers the query and a reason to care, which means a
  workload that actually clamps.
- **Wavefront-width reductions.** Nothing exists, and this is the backend
  where assuming a width is most obviously wrong, given CDNA and RDNA
  disagree.

The four-part rule in `docs/GPU_VALIDATION.md` under "Before adding
device-specific tuning" applies to all of them: a phase breakdown, a
profiler trace showing a mechanism, a portable change tried first, and
numbers from two devices. On AMD, "two devices" should mean one CDNA part
and one RDNA part wherever the finding could plausibly depend on wavefront
width.

## If it does not build or does not run

That is a result and it belongs in the failures section of
`docs/GPU_VALIDATION.md`, with the ROCm version, the MAX version, the exact
card, and the exact error. AMD hardware is harder to rent by the hour than
NVIDIA, so a recorded failure saves the next person more here than anywhere
else.

`has_accelerator()` is resolved at compile time, so the GPU path is compiled
in only on a machine that has a device. A wheel built elsewhere will not
grow a HIP path by being installed on a ROCm box.
