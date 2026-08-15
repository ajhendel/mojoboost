# GPU portability contract

Written: 2026-08-14

mojotrees has one GPU source. `docs/GPU_VALIDATION.md` states that
commitment and holds the record of what has actually run on hardware. This
document is the other half: **what the one source requires of a backend**,
which of those requirements every supported API specifies, which assumptions
in the tree are Apple-shaped rather than portable, and where a backend may
specialize without forking the source.

Nothing here is a claim about NVIDIA or AMD behavior. Only Metal has
executed a kernel from this repository, on one Apple M4. Everything below
about CUDA and HIP is a statement about what those APIs specify, or about
what the code in this tree does, not about anything anyone observed.

The Mojo modules that carry this contract are:

| Module | Holds |
|---|---|
| `src/mojotrees/gpu_backend_policy.mojo` | Which backend is in front of us, and how far this repository has taken it |
| `src/mojotrees/gpu_portability.mojo` | The primitive contract, the launch gate, and the specialization gate |
| `src/mojotrees/gpu_tiling.mojo` | The launch geometry itself, and the three device attributes it is derived from |
| `src/mojotrees/gpu_histogram_specializations.mojo` | The specialization primitives and the real threadgroup footprints |
| `src/mojotrees/device_policy.mojo` | Whether training runs on a device at all |

## 1. The primitive contract

The shared kernels reach for a small, fixed set of device primitives. This
set was read off the imports of every `src/mojotrees/gpu_*.mojo` and
`histogram_gpu.mojo`, not assumed. There is nothing else: no vendor
intrinsic, no subgroup operation, no cooperative group, no dynamic
threadgroup memory, no device-side allocation, no float atomic.

| Requirement | Spelled | Used by | Metal | CUDA | HIP |
|---|---|---|---|---|---|
| `threadgroup-barrier` | `max.gpu.sync.barrier()` | every histogram, scan, and partition kernel | specified | specified | specified |
| `static-threadgroup-allocation` | `stack_allocation[..., AddressSpace.SHARED]` | partial histograms, prefix-sum scratch | specified | specified | specified |
| `threadgroup-int32-atomic-add` | `Atomic.fetch_add` on a shared `Int32` | per-block partial histogram | specified | specified | specified |
| `device-int32-atomic-add` | `Atomic.fetch_add` on a device `Int32` | `STRATEGY_ATOMIC` flush only | specified | specified | specified |
| `two-dimensional-grid` | `grid_dim=(x, y)` | features or leaf slots on `x`, row tiles or classes on `y` | specified | specified | specified |
| `in-order-queue` | `DeviceContext` enqueue order | the hazard model in `gpu_runtime.mojo` | specified | specified | specified |

"specified" means the API documents the primitive. It does not mean this
project ran it there. The one column that has been exercised is Metal, and
`gpu_backend_policy.backend_support` is where that difference is recorded so
a reader cannot confuse the two.

`gpu_portability.required_primitives(strategy)` returns this list per
resolved histogram strategy. The two strategies differ by exactly one entry:
`STRATEGY_TILED` writes each partial to a slot nothing else writes and
reduces in a second kernel, so it needs no device-memory atomic;
`STRATEGY_ATOMIC` folds its partials into the output and does.

## 2. Backend support levels

`gpu_backend_policy.mojo` carries three levels, and the distinction between
the last two is the whole point of the module.

| Level | Meaning | Backends |
|---|---|---|
| `unsupported` | No contract exists. An API code outside the covered set, which is a caller bug and is refused | none in normal operation |
| `portable-untested` | The shared source targets it and the contract covers it. This repository has never executed a kernel on it | CUDA, HIP, and a device that did not name its API |
| `exercised` | Kernels from this repository have run on it and the run is recorded in `docs/GPU_VALIDATION.md` | Metal, on one Apple M4 |

`portable-untested` does not block training. Blocking it would refuse the
design commitment: one source, compiled for whatever device MAX opens. What
it blocks is **specialization**, through
`gpu_portability.require_specializations_allowed`, because a kernel variant
chosen for an architecture nobody has run on rests on a measurement that
does not exist.

An unidentified device is `portable-untested` rather than `unsupported` on
purpose. Nothing in the shipping code reads an API name before it launches,
so `API_UNKNOWN` is the value every launch carries today, and every bound in
its contract is the tightest of the three backends.

## 3. Apple-shaped assumptions in the shared code

Found by inspection, listed whether or not they are defects. Several are
deliberate and correct; naming them is what keeps them deliberate.

### 3.1 Unified memory is inferred from the API name

`apple_gpu_policy.mojo:327` sets `GpuProfile.unified_memory` to
`api == API_METAL`. That is right for Metal and wrong as a general rule: an
integrated AMD part, an NVIDIA Jetson, CUDA managed memory, and an
XNACK-enabled HIP device are all unified in ways an API name cannot
distinguish from a discrete card. Because `unified_memory` feeds
`apple_gpu_policy.partial_budget_bytes` (which divides a reported memory
budget by 32 instead of 16) and `unified_memory_policy.plan_session_routes`,
a unified non-Apple device currently gets the discrete partial-buffer
fraction.

The consequence today is bounded: the reported memory budget is 0 in every
profile this tree constructs, so `partial_budget_bytes` returns the portable
ceiling either way. `gpu_portability.contract_from_profile` takes
`unified_memory` from the profile rather than from the API, so the value has
one place to be corrected. The correction itself belongs to the module that
builds profiles, and is filed as a patch request in
`handoffs/connect_20_gpu_portability.md`.

### 3.2 Float32 gradients and no Float64 on device

`histogram_gpu.mojo:58` and `gpu_predict.mojo:37` both record it: Apple GPUs
have no Float64, so leaf values, base scores, gradients, and hessians are
Float32 on the device and histogram accumulation is fixed-point Int32.

This is Apple's floor imposed on every backend, and it is the right call for
one source, but it is a real cost on CUDA and HIP, which have Float64. It is
recorded as `BackendContract.device_float64_permitted = False` on every
backend rather than left as a comment, and
`gpu_portability.require_device_float64` is the one place a future Float64
variant would be gated from. Such a variant would have to reproduce the
fixed-point integers exactly, because those integers are what make a
histogram bit-identical run to run.

### 3.3 The specialization planner is Apple-named

`apple_histogram_policy.mojo` is the only module that turns device facts and
a node's shape into a specialized launch plan, and until now the only
constructor anything called for its capability input was
`DeviceHistogramCapabilities.portable()`, which answers "nothing reported"
for every device. So the specialization layer was reachable in practice only
through an Apple-named module, and every non-Apple device planned as if it
had reported nothing.

`gpu_portability.histogram_capabilities(contract)` is the connection: it
builds that same existing record from a backend contract, so a CUDA or AMD
device reaches the existing planner with its own capabilities instead of the
portable blank. Nothing in the planner is Apple-specific; the file name is.
Renaming it is filed as a patch request.

### 3.4 Subgroup width is never known and never assumed

Metal rejects the `WARP_SIZE` attribute query, and this repository has never
run the query on CUDA or HIP. `BackendContract.subgroup_width` is therefore
0, meaning unknown, on every backend in the table, and nothing in the
package divides by it.

`WARP_GRANULARITY = 64` in `gpu_tiling.mojo` is a different kind of value
and must not be read as a width claim. It is the multiple a threadgroup
width is rounded to, chosen because 64 is AMD's CDNA wavefront and a
multiple of the 32-wide warp everywhere else, so one constant is launchable
on all three backends. `BackendContract.launch_granularity` carries it under
a name that says so.

### 3.5 Three device attributes, and the shared-memory guard is optimistic

`gpu_tiling.query_device_caps` reads exactly three attributes:
`MULTIPROCESSOR_COUNT`, `MAX_THREADS_PER_BLOCK`, and
`MAX_SHARED_MEMORY_PER_BLOCK`, each falling back to a conservative constant
when a backend does not implement the query. That is the entire capability
surface every shipping GPU module sees.

`gpu_tiling.shared_bytes_for(n_bins)` models a block's threadgroup footprint
as `n_bins * 12`, which its own docstring records as the footprint of a
kernel sized to its bin count. The kernels that ship allocate three
`MAX_BINS`-wide Int32 planes whatever `n_bins` is, which is 3072 bytes. The
two agree only at 256 bins, so a device advertising less than 3 KiB would
pass the guard in `derive_tiling` and then fail to launch.

`gpu_portability.kernel_shared_request(n_bins, features)` returns the
footprint the compiled kernel really has, choosing between the two according
to whether the build has bin-capacity specialized kernels, and
`require_launch_geometry` checks that against the reported device.
`require_device_can_host_kernels` asks the same question once at session
open, against `MIN_SHARED_MEMORY_PER_BLOCK`.

No supported backend advertises less than 3 KiB. The checks exist so that
stays a fact rather than an assumption on hardware nobody here has opened.

### 3.6 Duplicate portable constants

`MAX_GRID_DIM_Y = 65535` is CUDA's `grid.y` cap and the bound the whole
package is written to, since Metal and HIP allow more and one source targets
all three. It is defined in `gpu_tiling.mojo:115` and mirrored in
`apple_gpu_policy.mojo:96`. `gpu_leaf_batching.mojo` used to carry a third
copy and now imports it. The remaining mirror is filed as a patch request.

`gpu_portability.mojo` imports the `gpu_tiling.mojo` definition, so the
portability contract and the geometry it gates cannot drift into disagreeing
about the bound.

### 3.7 One unchecked `grid.y`

`gpu_multiclass_batch.mojo:1041` launches `grid_dim=(row_blocks, k_count)`,
putting the class count on the `grid.y` axis rather than a tile count.
`derive_tiling` clamps its tile count to `MAX_GRID_DIM_Y`; nothing clamps
`k_count`. No realistic class count approaches 65535, so this is a missing
check rather than a live defect, and
`gpu_portability.require_launch_geometry` is the check to add. Filed as a
patch request.

## 4. Launch geometry contract

| Axis | Carries | Bound checked against |
|---|---|---|
| `grid.x` | features, active features under subsampling, or leaf slots | positive only. No portable upper bound is established here, and CUDA's is over two billion |
| `grid.y` | row tiles, leaf-batch tiles, or classes | `BackendContract.max_grid_dim_y`, which is 65535 |
| `grid.z` | nothing | not used. No portable bound for it has ever been established in this repository, and an earlier batched draft that wanted one was removed rather than given a guess |
| threadgroup width | lanes per block | the device's reported `MAX_THREADS_PER_BLOCK`. Rounding to `launch_granularity` is `gpu_tiling.clamp_block_threads`' job, not the gate's |
| threadgroup memory | the partial histogram | the device's reported `MAX_SHARED_MEMORY_PER_BLOCK`, against the real compiled footprint |

## 5. Synchronization and allocation contract

**Queue ordering.** A `DeviceContext` queue is in order, so device work never
needs a host synchronization to observe earlier device work. `HazardTracker`
in `gpu_runtime.mojo` is the model built on that, and it says a host
synchronization is required in exactly two cases: the host is about to read
memory the device has an unretired write to, or the host is about to write
memory the device has an unretired read or write to. That model is
backend-independent by construction, because it rests on queue ordering
rather than on any device's memory model.

**Allocation.** Every buffer is an explicit copy today, on every platform, in
every build. `unified_memory_policy.mojo` is the layer that could return
anything else and cannot currently do so without both an environment request
and an explicit acknowledgment that the request is unproven. A backend with
a unified address space does not change that default, and a route that skips
a copy changes when the host may next write the buffer, which is a
synchronization obligation and is returned with every decision there.

**Numerics.** Gradients and hessians cross to the device as Float32.
Accumulation is fixed-point Int32 with a scale chosen on the host, so
histogram addition is exact and associative and the result is bit-identical
run to run and strategy to strategy. Counts are exact. Agreement with the
CPU builder is to Float32 precision, not bit-exact. This is the property
most likely to differ first on a backend whose atomics or scheduling differ,
which is why `docs/GPU_VALIDATION.md` gates every timing behind a
determinism check.

## 6. Gates and environment variables

| Gate | Refuses |
|---|---|
| `require_backend_covered(api)` | an API code outside the covered set |
| `require_device_can_host_kernels(contract, caps)` | a device that cannot host the kernels at all, checked once at session open |
| `require_bins_supported(n_bins)` | a bin count outside `[1, MAX_BINS]` |
| `require_primitives(contract, strategy)` | a strategy whose primitives the backend does not promise |
| `require_specializations_allowed(contract, selected)` | a kernel variant a plan *selected* on a named backend nobody has run |
| `require_launch_geometry(...)` | a launch the reported device cannot run |
| `require_histogram_launchable(...)` | all of the above for one resolved histogram launch |
| `require_device_float64(contract)` | Float64 device arithmetic, on every backend |

Two details of the specialization gate that are easy to get backwards and
that change whether it is correct:

**Selected, not compiled-in.** `histogram_gpu.build_kernel_features()`
reports `batched_leaf_kernel = True`, because linking that module
instantiates the batched kernels from `gpu_leaf_batching.mojo`. That is a
fact about compilation. Selecting them is a separate decision that
`apple_histogram_policy` makes only when `SPEC_LEVEL_BATCHED` was asked for
by name and never from `auto`. The gate takes what the plan chose, so it
fires on the choice. Handing it the compiled set instead would refuse every
run on every backend. `require_histogram_launchable` takes both values
separately for this reason: the compiled set decides the threadgroup
footprint, the selected set faces the validation gate.

**The gate is only as strong as the backend identification.** An
unidentified backend (`API_UNKNOWN`) is not refused. It is the value every
launch carries today, because nothing in the shipping code reads an API name
before it launches and `MOJOTREES_GPU_BACKEND` is unset on an ordinary run,
so an unidentified device is indistinguishable here from the Apple part this
repository was developed on. Refusing it would refuse every run that ships,
including every Metal one. A CUDA device therefore escapes the gate until
the reported API name reaches the policy layer, which is filed as a patch
request in `handoffs/connect_20_gpu_portability.md` and is the single change
that turns this gate from a declaration into an enforcement.

| Variable | Effect |
|---|---|
| `MOJOTREES_GPU_BACKEND` | Names the API for reporting. Read by `device_policy.env_declared_api`, never a capability number. A declaration, not a detection |
| `MOJOTREES_GPU_BACKEND_UNVALIDATED=1` | Acknowledges selecting a specialization on a backend with no validation record, and runs it anyway. Any number measured under it must be reported with it |

The second follows `MOJOTREES_GPU_TRANSFER_UNPROVEN` in
`unified_memory_policy.mojo` deliberately: the evidence that would lift the
gate can only be produced by running the thing the gate blocks, so without
an override the gate is unsatisfiable by construction.

## 7. Specialization points, and why none is taken

Each of these is a place a backend could diverge without forking the source.
None is enabled anywhere, on any backend, because none has been measured on
any backend.

| Point | Where | What would have to be true first |
|---|---|---|
| Bin-capacity kernels | `gpu_histogram_specializations.bin_capacity_for`, `KernelFeatures.specialized_bin_kernels` | The kernels take a capacity parameter. Then a 32-bin histogram stops occupying 3 KiB of threadgroup memory, which raises residency on any device where threadgroup memory is the binding constraint |
| Packed four-byte bin loads | `plan_packed_window`, `pack4_bins`, `KernelFeatures.packed_bin_loads` | A measurement that a four-byte aligned load beats four one-byte loads on the device in hand. The portable implementation beside it produces the identical integers and is the definition |
| Batched multi-leaf launches | `gpu_leaf_batching.mojo`, `KernelFeatures.batched_leaf_kernel` | The batched kernels are compiled in and validated. Tiles pack onto `grid.y`, whose portable bound is known, rather than onto a `grid.z` whose bound is not |
| Float64 accumulation | `require_device_float64` | A variant that reproduces the fixed-point Int32 integers exactly on a backend that has Float64 |
| Subgroup reductions | nothing exists | A width read from the device, on a backend that answers the query, plus a portable implementation producing identical integers |

The rule the specialization modules enforce structurally, and that this
document restates so it is not lost: a non-portable operation may be reached
only through a flag on a capability record, and only where a portable
implementation of the same arithmetic sits next to it producing the
identical integers. The portable implementation is the definition. The fast
path is an optimization that has to reproduce it.

## 8. Per-backend notes

`docs/NVIDIA_GPU.md` and `docs/AMD_GPU.md` carry what to expect and what to
check first on each. `docs/GPU_VALIDATION.md` carries the procedure that
turns either from `portable-untested` into `exercised`, and the empty record
sections that procedure fills in.
