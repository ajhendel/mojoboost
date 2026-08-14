# Connect 20: portable CUDA/AMD specialization points

Status: the portability contract layer is landed and consumes the real
production types. **Nothing calls it yet**, because every call site lives in
another lane's files. The patch requests below are exact and each one is a
few lines.

Nothing was committed, staged, or formatted. No Mojo, Python, pixi, build,
test, benchmark, linter, formatter, or network command was run. Nothing in
this handoff is a claim about correctness, performance, parity, packaging,
or hardware validation.

Files created (all inside this lane's exclusive ownership):

- `src/mojoboost/gpu_backend_policy.mojo` (new, 223 lines)
- `src/mojoboost/gpu_portability.mojo` (new, 746 lines)
- `docs/GPU_PORTABILITY.md` (new)
- `docs/NVIDIA_GPU.md` (new)
- `docs/AMD_GPU.md` (new)
- `handoffs/connect_20_gpu_portability.md` (this file)

Files edited: none. Every other file named below is another lane's and is
quoted, not changed.

Line numbers below were read while several lanes were mid-edit
(`git status` showed `device_policy.mojo`, `gpu_tiling.mojo`,
`gpu_histogram_specializations.mojo`, `histogram_gpu.mojo`,
`initialization.mojo`, and `unified_memory_policy.mojo` modified). Anchor on
the symbol names, which are stable; treat the numbers as approximate.

## 1. Implementations found

Inventory before writing anything. Six places already held part of this
capability, and none of them held the contract itself.

| Where | Holds | Reachable from production? |
|---|---|---|
| `gpu_tiling.DeviceCaps` + `query_device_caps` | The entire capability surface every shipping GPU module sees: three device attributes, each with a conservative fallback | Yes. `histogram_gpu`, `gpu_runtime`, `gpu_predict`, `gpu_sparse`, `gpu_multiclass_batch`, `gpu_objectives_native`, `gpu_fused_round`, `gpu_gradient_stream` all call it |
| `apple_gpu_policy.GpuProfile` | A richer profile: API code, Apple generation, cores, threads, threadgroup memory, memory budget, unified flag, synthetic flag. Already API-generic despite the file name | Only through `device_policy`, which never has an open context, so it is always `generic()` or `PROFILE_DECLARED` |
| `apple_gpu_policy.parse_api` / `API_METAL` / `API_CUDA` / `API_HIP` | The API vocabulary, and the only place any of the three backends is named | `device_policy.env_declared_api` reads `MOJOBOOST_GPU_BACKEND` into it. Nothing else |
| `gpu_histogram_specializations.DeviceHistogramCapabilities` / `KernelFeatures` | The specialization capability boundary, the real threadgroup footprints, and the rule that a non-portable operation is reachable only behind a flag | Partly. `histogram_gpu.histogram_plan` calls the planner, but hands it `DeviceHistogramCapabilities.portable()` unconditionally |
| `apple_histogram_policy.derive_histogram_plan` | The one specialization planner, plus `kernel_block_bytes`, which is the compiled-footprint selector | Yes, from `histogram_gpu.histogram_plan` |
| `device_policy.DeviceCapabilities` | Whether training runs on a device at all, with blocks, warnings, evidence, and a memory estimate | Yes, the public device decision |

What was missing, and is what this lane added: **nothing anywhere named the
primitives the kernels actually require, and nothing checked a launch
against the device that reported its own limits.** The only backend-shaped
check in the tree is `derive_tiling`'s shared-memory guard, which its own
docstring records as optimistic.

## 2. Call path, before

    train_gpu / histogram_gpu / gpu_predict
      -> gpu_tiling.query_device_caps(ctx)        # 3 attributes
      -> gpu_tiling.derive_tiling(caps, ...)      # geometry, guard on n_bins*12
      -> ctx.enqueue_function[...](grid_dim=..., block_dim=...)

    histogram_gpu.histogram_plan
      -> apple_histogram_policy.derive_histogram_plan(
             profile_from_caps(caps),
             DeviceHistogramCapabilities.portable(),   # <- always the blank
             build_kernel_features(), ...)

    device_policy.decide_device
      -> GpuProfile.generic() or PROFILE_DECLARED from MOJOBOOST_GPU_BACKEND

No backend identity reaches a launch. No primitive is named. No specialized
variant is gated on where it has run. The real compiled threadgroup
footprint is never checked against the device.

## 3. Call path, after (once the patch requests below land)

    GpuSession.__init__ / GpuHistogramBuilder.__init__
      -> contract = gpu_portability.contract_for(api)          # or from a profile
      -> gpu_portability.require_device_can_host_kernels(contract, caps)

    per launch
      -> gpu_portability.require_histogram_launchable(
             contract, caps, tiling, grid_x, n_bins, compiled, selected)
      -> ctx.enqueue_function[...]

    histogram_gpu.histogram_plan
      -> derive_histogram_plan(
             profile, gpu_portability.histogram_capabilities(contract), ...)

    device_policy.decide_device
      -> gpu_backend_policy.backend_support(api) -> WARN_UNVALIDATED_BACKEND

Today only the left column of that exists. The right column is unreferenced
code until another lane applies section 6.

## 4. What was built

### `gpu_backend_policy.mojo`, the leaf

Answers two questions that were being collapsed into one word:

- `backend_is_covered(api)`: does the shared source target this backend?
- `backend_is_exercised(api)`: has this repository run a kernel on it?

Three levels: `SUPPORT_UNSUPPORTED` (an API code outside the covered set, a
caller bug, refused), `SUPPORT_PORTABLE` (CUDA, HIP, and an unidentified
device), `SUPPORT_EXERCISED` (Metal, on one M4, per `docs/GPU_VALIDATION.md`).

It imports `std.os.getenv` and the API vocabulary from `apple_gpu_policy`
and nothing else. It does not detect the backend: the API code is an
argument, because `device_policy.env_declared_api()` and
`GpuProfile.from_reported` already answer that and neither should be
duplicated. This is what makes the module importable from `device_policy`,
which must stay compilable and testable on a machine with no accelerator and
therefore cannot import anything that pulls in `max.gpu.host`.

New environment variable: `MOJOBOOST_GPU_BACKEND_UNVALIDATED=1`, shaped
exactly like `MOJOBOOST_GPU_TRANSFER_UNPROVEN` in `unified_memory_policy`.

### `gpu_portability.mojo`, the contract

`BackendContract` names the six primitives the shipping kernels actually
use, read off the imports of every `gpu_*.mojo` and `histogram_gpu.mojo`,
not assumed:

    threadgroup-barrier             max.gpu.sync.barrier()
    static-threadgroup-allocation   stack_allocation[..., AddressSpace.SHARED]
    threadgroup-int32-atomic-add    Atomic.fetch_add on a shared Int32
    device-int32-atomic-add         Atomic.fetch_add on a device Int32
    two-dimensional-grid            grid_dim=(x, y), no grid.z anywhere
    in-order-queue                  the hazard model in gpu_runtime.mojo

Plus `subgroup_width` (0, unknown, on every backend), `launch_granularity`
(`gpu_tiling.WARP_GRANULARITY`, a rounding rule and not a width claim),
`max_grid_dim_y` (`gpu_tiling.MAX_GRID_DIM_Y`), `grid_axes`,
`device_float64_permitted` (false everywhere), and `unified_memory`.

Gates, all raising with a message that names the value, the bound, and the
backend:

    require_backend_covered(api)
    require_device_can_host_kernels(contract, caps)     # once, at session open
    require_bins_supported(n_bins)
    require_primitives(contract, strategy)
    require_specializations_allowed(contract, selected)
    require_launch_geometry(contract, caps, gx, gy, threads, shared_bytes)
    require_histogram_launchable(...)                   # all of the above
    require_device_float64(contract)                    # refuses, everywhere

Two connections into existing types rather than new ones:

- `histogram_capabilities(contract)` builds the existing
  `DeviceHistogramCapabilities` from a backend contract. It is the only way
  a non-Metal device reaches the existing specialization planner with
  anything other than the blank.
- `kernel_shared_request(n_bins, compiled)` returns the footprint the
  compiled kernel really has, composed from
  `gpu_histogram_specializations.kernel_shared_bytes`,
  `bin_capacity_for`, and `unspecialized_kernel_shared_bytes`. It closes the
  gap `gpu_tiling.shared_bytes_for` documents: the model is `n_bins * 12`,
  the shipping kernels allocate 3072 bytes at every bin count, and the two
  agree only at 256 bins.

### Two files, not one, and why

`gpu_portability.mojo` imports `gpu_tiling`, which imports `max.gpu.host`.
`device_policy.mojo` deliberately mirrors constants rather than importing
`gpu_tiling` for exactly that reason. Splitting backend identity (leaf, no
GPU imports) from the launch contract (above `gpu_tiling`) is what lets
`device_policy` consume the first without inheriting the second. It is one
policy in two layers, not two policy engines: `gpu_portability` imports
`gpu_backend_policy` and restates none of it.

## 5. Duplicates fused or quarantined

Fused, by importing rather than restating:

- `MAX_GRID_DIM_Y`, `WARP_GRANULARITY`, `STRATEGY_*`, `strategy_name`,
  `DeviceCaps`, `HistogramTiling` from `gpu_tiling`.
- `MAX_BINS`, `PLANES_PER_HISTOGRAM`, `BYTES_PER_PLANE_CELL`,
  `kernel_shared_bytes`, `bin_capacity_for`,
  `unspecialized_kernel_shared_bytes`, `DeviceHistogramCapabilities`,
  `KernelFeatures` from `gpu_histogram_specializations`.
- `API_*`, `api_name`, `GpuProfile` from `apple_gpu_policy`.

`MIN_SHARED_MEMORY_PER_BLOCK` is derived from
`PLANES_PER_HISTOGRAM * BYTES_PER_PLANE_CELL * MAX_BINS` rather than written
as 3072, so it cannot drift from `kernel_shared_bytes(MAX_BINS)`.

Duplicates found and **not** fused, with the reason:

- `MAX_GRID_DIM_Y = 65535` exists twice: `gpu_tiling.mojo:115` and the
  mirror at `apple_gpu_policy.mojo:96`. This one is structural and should
  stay. `apple_gpu_policy` is deliberately a leaf so the whole policy stack
  is exercisable without an accelerator, and `gpu_tiling` imports
  `max.gpu.host`, so the leaf cannot import the definition. There is no
  module below both to move it to. Request in 6.4 is to pin it, not to
  collapse it. (`gpu_leaf_batching` used to hold a third copy and now
  imports `gpu_tiling`'s.)
- `apple_histogram_policy.kernel_block_bytes(features, capacity)` is the
  same two-branch selector as `kernel_shared_request`. Not imported here,
  because this module has to stay *below* `apple_histogram_policy` so that
  module can consume `histogram_capabilities` without closing a cycle.
  Request 6.3 is to collapse it in that direction.
- `_bool_text` is a three-line formatter that `device_policy` and
  `apple_histogram_policy` each already have. A local copy was written
  rather than importing either, for the same direction-of-dependency
  reason. Documented at its definition.

Nothing was quarantined and nothing was deleted.

## 6. Exact cross-lane patch requests

Each is small. None changes behavior on the shipping path except where
noted, and every one of them is inert until a backend is identified.

### 6.1 Task 01 (`gpu_runtime.mojo`, `histogram_gpu.mojo`, `__init__.mojo`)

**(a) `GpuSession`, one device-level check at open.** At
`gpu_runtime.mojo:1272`, after `self.caps = query_device_caps(self.ctx)`:

```mojo
from .apple_gpu_policy import API_UNKNOWN
from .gpu_portability import BackendContract, contract_for
from .gpu_portability import require_device_can_host_kernels

# in the struct
var contract: BackendContract

# in __init__, taking `api: Int = API_UNKNOWN` as a new trailing parameter
self.contract = contract_for(api)
require_device_can_host_kernels(self.contract, self.caps)
```

The default `API_UNKNOWN` keeps every existing caller unchanged. A caller
that holds a `DeviceDecision` should pass
`decision.capabilities.profile.api` so the contract is the device's rather
than the floor.

**(b) `GpuHistogramBuilder`, one gate per launch.** After each
`self.tiling = derive_tiling(...)` (`histogram_gpu.mojo:424` and `:535`):

```mojo
require_histogram_launchable(
    self.contract,        # or contract_for(API_UNKNOWN) until (a) lands
    self.caps,
    self.tiling,
    len(self.active),     # grid.x is the active feature count
    self.n_bins,
    build_kernel_features(),      # compiled
    selected_kernel_features(),   # what the plan chose; see below
)
```

`selected` must be what the resolved `HistogramPlan` chose, **not**
`build_kernel_features()`. `build_kernel_features()` returns
`KernelFeatures(False, False, True)` because linking the module instantiates
the batched kernels, and passing that as `selected` would refuse every run
on a named non-Metal backend. If the plan does not yet expose what it
selected, pass `KernelFeatures.none()` and file the follow-up: the gate then
checks geometry and primitives, which is still worth having.

**(c) `histogram_plan` should stop handing the planner a blank.**
`histogram_gpu.mojo:829` currently passes
`DeviceHistogramCapabilities.portable()`. Replace with:

```mojo
from .gpu_portability import contract_for, histogram_capabilities
...
histogram_capabilities(self.contract),
```

The docstring above that call is correct today and will need one sentence
changed: the capabilities are no longer unconditionally blank, they are the
backend's, and `subgroup_width` is still 0 because nothing reads it.

**(d) `__init__.mojo` exports.** Beside the existing
`from .apple_histogram_policy import (...)` block at line 210:

```mojo
from .gpu_backend_policy import (
    SUPPORT_EXERCISED,
    SUPPORT_PORTABLE,
    SUPPORT_UNSUPPORTED,
    backend_is_exercised,
    backend_support,
    describe_backend,
    support_name,
)
from .gpu_portability import (
    BackendContract,
    contract_for,
    contract_from_profile,
    describe_contract,
    histogram_capabilities,
    kernel_shared_request,
    porting_note,
    require_device_can_host_kernels,
    require_histogram_launchable,
)
```

### 6.2 Task 02 (`gpu_active_rows.mojo`, `gpu_leaf_batching.mojo`)

The two range-histogram launches at `gpu_active_rows.mojo:1375` and `:1407`
pass `grid_dim=(n_slots, tiling.n_tiles)`. `derive_tiling` clamps
`n_tiles` to `MAX_GRID_DIM_Y`, so those are already within bound, but the
threadgroup footprint is not checked against the device anywhere. One call
to `require_launch_geometry(contract, caps, n_slots, tiling.n_tiles,
tiling.block_threads, kernel_shared_request(n_bins, compiled))` before the
pair covers both.

`gpu_leaf_batching` packs several leaves' tiles onto one `grid.y`
(`plan.total_tiles`) and clamps to `MAX_GRID_DIM_Y` itself in two places.
That clamp is the right one; passing the same value through
`require_launch_geometry` makes it a checked bound rather than a clamp that
could be edited away silently.

### 6.3 Task 04 (`apple_histogram_policy.mojo`, `gpu_tiling.mojo`, `gpu_multiclass_batch.mojo`)

**(a) Consume the contract instead of restating the footprint.** Replace the
body of `apple_histogram_policy.kernel_block_bytes` with a call to
`gpu_portability.kernel_shared_request`, or delete it and have its callers
use that directly. The import direction is safe:
`gpu_portability` imports `gpu_tiling` and
`gpu_histogram_specializations` only, so `apple_histogram_policy` may import
`gpu_portability` without a cycle. **The reverse must never happen:**
`gpu_tiling` and `gpu_histogram_specializations` must not import
`gpu_portability`, or the layering closes.

**(b) Rename the module.** Nothing in `apple_histogram_policy.mojo` is
Apple-specific: it is the specialization planner for every backend, and the
file name is why a CUDA device looked like it had no planner. Renaming it to
`gpu_histogram_policy.mojo` is a mechanical change to the imports in
`histogram_gpu.mojo`, `gpu_multiclass_batch.mojo`, `__init__.mojo`, and the
tests. The same argument applies more weakly to `apple_gpu_policy.mojo`,
which holds the portable `GpuProfile` and the API vocabulary for all three
backends; that one is Task 05's file.

**(c) One unchecked `grid.y`.** `gpu_multiclass_batch.mojo:1059` launches
`grid_dim=(self._row_blocks(), k_count)`, putting the class count on
`grid.y` with nothing clamping or checking it. No realistic class count
approaches 65535, so this is a missing check rather than a live defect.
`require_launch_geometry(contract, caps, self._row_blocks(), k_count,
block_threads, shared_bytes)` before the launch is the fix.

### 6.4 Task 05 (`device_policy.mojo`, `apple_gpu_policy.mojo`)

**(a) Report the backend support level in the device decision.**
`device_policy` may import `gpu_backend_policy` safely: it is a leaf whose
only imports are `std.os.getenv` and `apple_gpu_policy`, so it pulls in no
GPU stack and the module stays compilable on a machine with no accelerator.
It must **not** import `gpu_portability`, which imports `gpu_tiling` and
therefore `max.gpu.host`.

```mojo
from .gpu_backend_policy import (
    SUPPORT_EXERCISED, backend_support, describe_backend, support_name,
)

comptime WARN_UNVALIDATED_BACKEND = 13   # next free code

# in warning_name
if code == WARN_UNVALIDATED_BACKEND:
    return String("unvalidated-backend")

# in _collect_warnings, wherever the profile is in hand
if backend_support(caps.profile.api) != SUPPORT_EXERCISED:
    warnings.append(WARN_UNVALIDATED_BACKEND)

# in describe_decision
out += String(describe_backend(decision.capabilities.profile.api), "\n")
```

This is a warning and not a block, deliberately. Blocking would refuse the
design commitment of one portable source.

**(b) Stop inferring unified memory from the API name.**
`apple_gpu_policy.mojo:327` sets `unified_memory` to `api == API_METAL` in
`GpuProfile.from_reported`. That is right for Metal and wrong as a rule: an
AMD APU, an NVIDIA Jetson, CUDA managed memory, and an XNACK-enabled HIP
device are all unified in ways an API name cannot distinguish from a
discrete card. Suggested change that keeps every existing caller identical:

```mojo
@staticmethod
def from_reported(
    reported_api: String,
    reported_arch: String,
    core_count: Int,
    max_threads_per_block: Int,
    max_shared_memory_per_block: Int,
    memory_budget_bytes: Int = 0,
    unified_memory: Int = -1,     # -1 = infer from the API, as today
) -> GpuProfile:
```

with `-1` reproducing today's `api == API_METAL` and `0`/`1` letting a
caller that read the device report it. The consequence today is bounded:
every profile this tree constructs has a memory budget of 0, so
`partial_budget_bytes` returns the portable ceiling whichever divisor it
would have chosen. `gpu_portability.contract_from_profile` already takes
`unified_memory` from the profile rather than from the API, so there is one
place for the corrected value to flow from.

**(c) Plumb the reported API name.** This is the single change that turns
the specialization gate from a declaration into an enforcement. Today
`API_UNKNOWN` is the value every launch carries, because nothing reads an
API name before launching, and `require_backend_exercised` therefore cannot
refuse an unidentified device without refusing every Metal run too. A caller
that opens a `DeviceContext` and can obtain the API name should build the
profile through `GpuProfile.from_reported` and carry
`profile.api` into the session. Until then the gate is only as strong as
`MOJOBOOST_GPU_BACKEND`.

**(d) Pin the `MAX_GRID_DIM_Y` mirror.** `apple_gpu_policy.mojo:96` mirrors
`gpu_tiling.mojo:115`. Keep the mirror (the leaf cannot import a module that
pulls in `max.gpu.host`) and add it to the existing mirror-pinning test the
module docstring already refers to, if it is not there.

### 6.5 Task 06 (`gpu_predict.mojo`, `bindings/_mojoboost.mojo`)

**(a)** `gpu_predict.mojo:955` reads `query_device_caps(ctx)` and derives a
block width. One `require_device_can_host_kernels(contract, caps)` there,
plus `require_launch_geometry` before the prediction launches with whatever
threadgroup memory those kernels allocate, gives prediction the same gate
training gets. Predict launches are one-dimensional today, so pass 1 for
`grid_y`.

**(b)** The capability record exposed to Python should carry the backend
support level and `describe_contract(contract)`. Python must format it and
must not re-derive it: the backend is a Mojo-side fact. The strings are
`gpu_backend_policy.support_name` and `gpu_portability.describe_contract`,
and neither allocates a device.

**(c)** An explicit GPU prediction request on a named, unexercised backend
should surface the support level in its error text when it fails for another
reason, so a user on CUDA reading an error knows the backend has never been
validated here. It must not fail *because* of the support level.

### 6.6 Task 18 (`docs/PLATFORM_MATRIX.md`, `packaging/matrix/platform_matrix.toml`)

**(a)** Add a GPU-backend section to `docs/PLATFORM_MATRIX.md` using that
file's existing four-word vocabulary, and keep the TOML in step so
`validate_matrix.py` stays the authority:

| Backend | Status | Evidence |
|---|---|---|
| Metal (Apple silicon) | `validated` | `docs/GPU_VALIDATION.md`, Apple M4 record |
| CUDA | `designed` | none. No NVIDIA device has run this code |
| HIP / ROCm | `designed` | none. No AMD device has run this code |

`validated` requires an evidence file in that document's own rule, and only
Metal has one. Do not promote a row without one.

**(b)** Cross-link `docs/GPU_PORTABILITY.md`, `docs/NVIDIA_GPU.md`, and
`docs/AMD_GPU.md` from `docs/PLATFORM_MATRIX.md` and
`docs/INSTALLATION.md`, and include them in whatever doc set ships in the
sdist, so a user who installs on a CUDA box can find out what is and is not
known before they file a performance bug.

**(c)** A wheel built on a machine with an accelerator carries a GPU path
compiled in; one built without does not, because `has_accelerator()` is a
comptime query. Both new backend docs state this. If the packaging lane
records a per-artifact "GPU path compiled in" flag, that is the truthful
place for it.

## 7. Fallbacks preserved

- Every primitive Bool in `contract_for` is True on every covered backend,
  including `API_UNKNOWN`. The gates therefore pass exactly where today's
  code already launches, and the established path is unchanged.
- `API_UNKNOWN` is covered rather than refused, so a device that never named
  its API keeps running.
- The specialization gate does not refuse an unidentified backend, so no
  Metal run is affected before 6.4(c) lands.
- `require_device_float64` refuses on every backend, which is what the
  source already does implicitly. It changes nothing and gives a future
  variant one place to be gated from.
- Nothing removes, relaxes, or replaces `derive_tiling`'s existing guard.
  `kernel_shared_request` is a second, tighter check beside it; tightening
  the first would change which shapes `derive_tiling` accepts, which is
  another lane's decision.

## 8. Serialization and public API effects

None yet. No model state, no serialized field, no Python-visible name
changes, because nothing outside this lane was edited.

Once 6.1(d) lands, `mojoboost` gains the Mojo-level exports listed there.
Once 6.4(a) lands, `describe_decision` gains one line and the decision gains
one warning code, both of which are diagnostic text rather than model state.
Nothing here needs to serialize: a backend contract is a property of the
machine a model is trained on, not of the model, and writing it into a model
file would make the file machine-specific for no benefit.

## 9. Risks

1. **Unreferenced code until section 6 is applied.** The honest status: this
   lane could not edit a single call site, so what landed is a contract with
   no callers. If no other lane applies the patch requests, this is
   documentation with a type checker.
2. **The specialization gate is weak until the API name is plumbed.** 6.4(c)
   is the change that matters. Without it a CUDA device is indistinguishable
   from the M4 and the gate never fires. This is stated in the module
   docstring, in `docs/GPU_PORTABILITY.md`, and here, rather than papered
   over.
3. **`compiled` versus `selected` is a live footgun.** Passing
   `build_kernel_features()` for `selected` in 6.1(b) would refuse every run
   on a named non-Metal backend, because that function reports
   `batched_leaf_kernel = True` for a linkage reason. The parameter names
   and both docstrings say so; a reviewer should still check that call site
   specifically.
4. **Concurrent edits.** Six files were being edited by other lanes while
   this was written. The imports used are long-standing names
   (`DeviceCaps`, `HistogramTiling`, `MAX_BINS`, `KernelFeatures`,
   `GpuProfile`, `MAX_GRID_DIM_Y`, `WARP_GRANULARITY`, `STRATEGY_*`), but a
   rename in `gpu_tiling.mojo` or `gpu_histogram_specializations.mojo` would
   break the two new files at the import line and nowhere else.
5. **Nothing here compiled.** No build was run, by instruction. The Mojo in
   both new files is unverified: it follows the syntax and the conventions
   of the modules it sits beside, and that is all that can be said.
6. **The primitive table is a reading of API specifications**, not of
   hardware, and it says so in every place it appears. If a backend does not
   honor one of the six, the failure will look like wrong numbers rather
   than a refusal, which is why the determinism step in
   `docs/GPU_VALIDATION.md` gates every timing.

## 10. Remaining disconnections

- No call site calls any gate. Section 6.
- No backend identity reaches a launch. Section 6.4(c).
- `apple_histogram_policy` is still the only planner and is still
  Apple-named. Section 6.3(b).
- `HistogramPlan` does not report which variants it selected, which is what
  6.1(b) needs to pass the right `selected` value. If it cannot, the caller
  passes `KernelFeatures.none()` and the specialization gate stays inert
  while the geometry gate works.
- `subgroup_width` is 0 on every backend and nothing queries `WARP_SIZE`
  anywhere. That is correct today (Metal refuses the query) and stays
  correct until someone runs the query on a backend that answers it.
- The memory budget is 0 in every profile this tree constructs, so
  `partial_budget_bytes` is the portable ceiling everywhere and the unified
  versus discrete divisor has never had an effect.

## 11. Smallest later focused commands, all UNRUN

Nothing below was executed. They are recorded so the next lane runs the
smallest thing that could fail, one at a time.

```sh
# UNRUN. Type-check the two new modules through a module that imports them,
# after at least one export from 6.1(d) exists.
pixi run mojo build -I src src/mojoboost/gpu_portability.mojo

# UNRUN. The one focused test to write for this lane, when tests are allowed
# again: pure host arithmetic, no accelerator needed.
pixi run mojo run -I src tests/parallel/test_gpu_portability.mojo
```

What that test should cover, since it is cheap and none of it needs a
device: `contract_for` raises outside the covered set; `API_UNKNOWN` yields
the portable floor with every primitive true; `require_backend_exercised`
refuses CUDA and HIP, allows Metal, and allows `API_UNKNOWN`;
`MOJOBOOST_GPU_BACKEND_UNVALIDATED=1` lifts the first;
`kernel_shared_request` returns 3072 without the specialized kernels and
`kernel_shared_bytes(bin_capacity_for(n))` with them;
`require_launch_geometry` refuses a `grid.y` of 65536, a block wider than
the device maximum, and a shared request above the device maximum;
`require_histogram_launchable` passes for the shapes `derive_tiling`
produces from `DeviceCaps.fallback()`; and `MIN_SHARED_MEMORY_PER_BLOCK`
equals `kernel_shared_bytes(MAX_BINS)`.
