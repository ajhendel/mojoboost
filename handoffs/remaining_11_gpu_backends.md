# Task 11: portable GPU kernel specializations for NVIDIA and AMD

Status: two new policy modules and one new document landed inside this
lane's exclusive ownership. Nothing outside those four files was touched.
Nothing was staged, committed, built, run, or tested. No test file was
written, per the lane's instructions.

Files owned and written:

- `src/mojoboost/gpu_cuda_policy.mojo` (new)
- `src/mojoboost/gpu_amd_policy.mojo` (new)
- `docs/GPU_BACKEND_SPECIALIZATIONS.md` (new)
- `handoffs/remaining_11_gpu_backends.md` (this file)

**No hardware validation is claimed anywhere.** This repository has run a
GPU kernel on one Apple M4 and on nothing else. Every CUDA and HIP row in
`docs/GPU_VALIDATION.md` reads "not run".

## Coordination with CONNECT_EVERYTHING Task 20

That lane is active in this worktree. At the time of writing it owns
`src/mojoboost/gpu_portability.mojo` (untracked, growing during this task:
26894 bytes at 12:41, 30659 bytes at 12:51) and
`src/mojoboost/gpu_backend_policy.mojo` (committed), and it has already
wired `require_device_can_host_kernels` and two
`require_histogram_launchable` calls into `src/mojoboost/histogram_gpu.mojo`.

This lane built **on** that work rather than beside it. Neither new module
defines a backend contract, a primitive set, a grid bound, a support level,
or a launch gate: all five are imported from Task 20's files and called.
The symbols depended on, all confirmed present in
`gpu_portability.mojo` as of the 12:51 revision:

`REQ_IN_ORDER_QUEUE`, `BackendContract`, `contract_from_profile`,
`describe_contract`, `histogram_capabilities`, `kernel_shared_request`,
`require_histogram_launchable`, `require_specializations_allowed`.

Two contract changes in that file were caught during this task and adopted:
`require_specializations_allowed` now gates on the *selected* variants
rather than the compiled-in ones, and `require_histogram_launchable` now
takes `compiled` plus an optional `selected`. Both new modules follow that
split: `derive_plan_for` derives a `selected` set and reports it on
`BackendLaunchPlan.selected`, and the backend launch gates pass it through.

**Risk to Task 20:** if that lane renames any of the eight symbols above
before it lands, both new modules stop compiling. The fix is an import
rename in two files and nothing else; no logic depends on the spelling.

## Implementations found before writing anything

Inventoried so that nothing here duplicates them.

| Capability | Existing owner | What this lane did |
|---|---|---|
| Tile count, row tile, block-width rounding, partial budget, launch counts | `gpu_tiling.mojo` (`resolve_tiling`, `derive_tiling`, `clamp_block_threads`, `partial_cell_limit_for`, `launches_for_strategy`) | called, never restated |
| Bin capacities, kernel shared footprints, packed windows, storage descriptor, capability record | `gpu_histogram_specializations.mojo` | called, never restated |
| Which backends are covered and which have been run, the unvalidated override | `gpu_backend_policy.mojo` | called |
| Per-backend contract, primitive set, launch gates, Float64 refusal | `gpu_portability.mojo` | called |
| Specialization ladder, per-node level and reason, batching decision | `apple_histogram_policy.mojo` | not touched, not duplicated; the new plan carries a `baseline` the same way `HistogramPlan` does |
| Partial-buffer budget fraction | `apple_gpu_policy.partial_budget_bytes` | called (it is not Apple-specific despite the module name) |
| Transfer routes per buffer role | `unified_memory_policy.plan_session_routes` | called |
| Host/device synchronization model | `gpu_runtime.HazardTracker` | referenced, not duplicated; the new modules only check the in-order-queue promise the model rests on |
| CPU/GPU device selection and the crossover evidence table | `device_policy.mojo` | untouched, deliberately: selection happens before a backend is known |

**No duplicate was created and none was found to fuse.** The one
duplication risk this lane could have introduced, a second copy of the
occupancy rule and the plan assembly in the AMD module, was avoided by
importing them from the CUDA module. See "Patch 1" for the relocation that
removes that awkward edge.

## What the two modules add

Only bounds. Eleven specialization points, each grounded in a reported
`DeviceAttribute` this repository already queries in
`bench/bench_gpu_validation.mojo`, or in a documented property of the
backend's programming model that the shared source is already subject to.
`docs/GPU_BACKEND_SPECIALIZATIONS.md` is the long form. The short form:

1. **Subgroup width** from `WARP_SIZE`, 0 when unreported, never
   substituted from an architectural constant. Plausibility refusals: 32 on
   CUDA, 32 or 64 on HIP.
2. **Thread-block geometry** rounds to the reported width when there is
   one, and delegates to `gpu_tiling.clamp_block_threads` when there is not.
3. **Shared memory** checked against two ceilings: the model's static
   declaration limit (48 KiB CUDA, 64 KiB HIP LDS) and the device's
   reported per-block limit. Inert today at 3 KiB.
4. **Atomics** unchanged; `atomic_conflict_degree` reports the structural
   contention of each flush.
5. **Occupancy** from `MAX_THREADS_PER_MULTIPROCESSOR` and
   `MAX_BLOCKS_PER_MULTIPROCESSOR`, falling back to
   `gpu_tiling.TARGET_BLOCKS_PER_SM`. Two bounds recorded as knowingly
   absent (`shared_bound_known`, `registers_bound_known`).
6. **Partial histogram strategy**: no preference on either backend.
   `StrategyInputs` reports what a rule would key on and carries
   `TILED_PREFERENCE_UNMEASURED`.
7. **Packed-bin alignment**: the shared planner is called; each backend
   adds only its transaction unit (32-byte sector, 64-byte line) as a
   reported diagnostic.
8. **Allocation** delegates to `plan_session_routes`; unified memory is
   reported-only on both backends and defaults false.
9. **Streams**: one in-order queue, `require_concurrent_queues` is a
   capability error.
10. **Synchronization**: `require_in_order_queue` against the contract.
11. **Capability errors**: seven new refusals, three delegated. Both
    `require_cuda` and `require_amd` refuse `API_UNKNOWN` on purpose.

## Call path, before and after

**Before.** `GpuHistogramBuilder.__init__` reads three attributes through
`gpu_tiling.query_device_caps`, builds an `API_UNKNOWN` profile through
`apple_histogram_policy.profile_from_caps`, derives a contract, and gates
each launch with `require_histogram_launchable`. Every device on every
backend plans from the same three numbers and takes the portable
granularity, the fixed residency target, and the flat partial budget.

**After (this lane, unwired).** Unchanged. Nothing imports either new
module. `derive_cuda_plan` and `derive_amd_plan` exist, are pure, and have
no caller. This is `deferred` under `docs/CAPABILITY_LEVELS.md` and the
document says so.

**After (with the patches below applied by their owners).** The builder
reads ten attributes instead of three, holds a `DeviceReport` beside its
`DeviceCaps`, and dispatches its per-launch gate on the profile's API code:
Metal keeps `apple_histogram_policy` exactly as it is, CUDA and HIP reach
`require_cuda_launchable` and `require_amd_launchable`, and an
unidentified device keeps today's portable path. A device that answers the
two per-multiprocessor attributes gets a residency derived from them
instead of the fixed 8.

## READY-TO-APPLY INTEGRATION PATCHES

Not applied. Each is another lane's file. Each is stated so it can be
applied mechanically.

---

### Patch 1: relocate the backend-neutral half into the portability layer

- **Target file:** `src/mojoboost/gpu_portability.mojo`
- **Target symbols (moved verbatim from `gpu_cuda_policy.mojo`):**
  `ATTR_UNREPORTED`, `_reported_or`, `_ceil_div`, `DeviceReport`,
  `RESIDENCY_PORTABLE_TARGET`, `RESIDENCY_BY_BLOCKS`,
  `RESIDENCY_BY_THREADS`, `RESIDENCY_BY_BOTH`, `residency_source_name`,
  `Occupancy`, `resident_blocks_from_reported`,
  `TILED_PREFERENCE_UNMEASURED`, `atomic_conflict_degree`,
  `StrategyInputs`, `strategy_inputs`, `preferred_strategy`,
  `BackendLaunchPlan`, `BackendSpecialization`, `describe_specialization`,
  `describe_plan`, `derive_block_threads_for`,
  `require_shared_within_ceiling`, `require_shared_reported_fits`,
  `derive_plan_for`. That is everything above the
  `# End backend-neutral section` marker in `gpu_cuda_policy.mojo` and
  nothing below it.
- **Signatures:** unchanged. `gpu_portability.mojo` already imports
  `DeviceCaps`, `HistogramTiling`, `KernelFeatures`, `GpuProfile`, and
  `strategy_name`; it additionally needs `APPLE_GEN_UNKNOWN`,
  `partial_budget_bytes`, `BYTES_PER_PARTIAL_CELL`,
  `FALLBACK_MAX_THREADS_PER_BLOCK`, `FALLBACK_SHARED_MEMORY_PER_BLOCK`,
  `FALLBACK_SM_COUNT`, `TARGET_BLOCK_THREADS`, `TARGET_BLOCKS_PER_SM`,
  `WARP_GRANULARITY`, `STRATEGY_ATOMIC`, `STRATEGY_TILED`,
  `clamp_block_threads`, `derive_tiling`, `resolve_tiling`, and
  `bin_capacity_for`. It has `_bool_text` already; drop the copy that moves
  with the block.
- **Call sites to re-point:** the import block at the top of
  `gpu_cuda_policy.mojo` and the `from .gpu_cuda_policy import (...)` block
  at the top of `gpu_amd_policy.mojo` both change to
  `from .gpu_portability import (...)`. The AMD module's import of
  `preferred_strategy` moves with it.
- **State flow:** none. Every moved symbol is pure arithmetic over Ints,
  Bools, and the structs named above.
- **Errors:** unchanged. `require_shared_within_ceiling` and
  `require_shared_reported_fits` keep raising the same messages.
- **Ownership:** `gpu_portability.mojo` belongs to CONNECT_EVERYTHING
  Task 20. This lane cannot apply it.
- **Fallback if not applied:** none needed. The modules compile and behave
  identically with the neutral half living in the CUDA file; the only cost
  is the `gpu_amd_policy -> gpu_cuda_policy` import edge, which reads
  oddly.
- **Serialization effect:** none. Nothing moved is serialized.
- **Public API effect:** none. Neither module is exported.
- **Dependency:** must land after Task 20's file is committed, or the two
  edits collide.
- **Later validation (UNRUN):**
  `pixi run mojo build -I src src/mojoboost/gpu_amd_policy.mojo` compiles.

---

### Patch 2: read the full device report

- **Target file:** `src/mojoboost/gpu_tiling.mojo`
- **Target symbol:** new `query_full_device_report`, beside the existing
  `query_device_caps`.
- **Signature:**
  `def query_full_device_report(ctx: DeviceContext) -> DeviceReport:`
- **Body:** one `_attribute_or(ctx, DeviceAttribute.X, 0)` per attribute,
  passing `0` as the default so an unanswered attribute stays unreported
  rather than becoming a fallback constant, then
  `DeviceReport.queried(...)`. The ten attributes are exactly the ones
  `bench/bench_gpu_validation._report_device` prints:
  `MULTIPROCESSOR_COUNT`, `WARP_SIZE`, `MAX_THREADS_PER_BLOCK`,
  `MAX_THREADS_PER_MULTIPROCESSOR`, `MAX_BLOCKS_PER_MULTIPROCESSOR`,
  `MAX_SHARED_MEMORY_PER_BLOCK`, `MAX_REGISTERS_PER_BLOCK`,
  `MAX_GRID_DIM_X`, `MAX_GRID_DIM_Y`, `CLOCK_RATE`. There is no memory
  budget attribute; leave `memory_budget_bytes` at 0 and
  `unified_memory` at False until an accessor exists, which is the same
  gap `apple_histogram_policy.profile_from_caps` already documents.
- **Call site:** `histogram_gpu.GpuHistogramBuilder.__init__` (the
  private-context overload, `src/mojoboost/histogram_gpu.mojo:329`), beside
  the existing `var caps = query_device_caps(ctx)`.
- **State flow:** the report flows into the builder field added by Patch 3
  and nowhere else.
- **Errors:** none. `_attribute_or` swallows a refused query, which is the
  established behavior and the reason Metal's six refusals are not fatal
  today.
- **Ownership:** `gpu_tiling.mojo` is a shared kernel-lane file.
- **Fallback if not applied:** every caller uses
  `DeviceReport.unreported()`, under which both new modules produce
  `matches_baseline() == True` plans. Nothing regresses.
- **Serialization effect:** none.
- **Public API effect:** none unless exported.
- **Dependency:** needs `DeviceReport` importable, so it lands after
  Patch 1 (or imports from `gpu_cuda_policy.mojo`, which would invert the
  layering and should not be done).
- **Later validation (UNRUN):** `pixi run mojo run -I src` on a one-file
  driver that opens a context and prints
  `report.answered()`; on the project M4 it should print `4` or `5`, which
  is what `docs/GPU_VALIDATION.md` already records Metal answering.

---

### Patch 3: hold the report and dispatch the launch gate by backend

- **Target file:** `src/mojoboost/histogram_gpu.mojo`
- **Target symbol:** `GpuHistogramBuilder`
- **New field**, beside `var caps: DeviceCaps` (currently
  `src/mojoboost/histogram_gpu.mojo:295`):
  `var report: DeviceReport` with the comment "the full attribute set, for
  the backend policy; `unreported()` on any path that did not query it".
- **Assignment:** in the four-argument `__init__`, beside
  `self.caps = caps.copy()`, add `self.report = report.copy()`; add a
  `report: DeviceReport = DeviceReport.unreported()` parameter with that
  default so no existing call site changes, and pass
  `query_full_device_report(ctx)` from the private-context overload.
- **New method:**
  ```
  def _require_launchable_backend(
      self, grid_x: Int, selected: KernelFeatures
  ) raises:
  ```
  which switches on `self.contract.api`: `API_CUDA` calls
  `gpu_cuda_policy.require_cuda_launchable`, `API_HIP` calls
  `gpu_amd_policy.require_amd_launchable`, and everything else calls
  today's `require_histogram_launchable` unchanged.
- **Call sites:** the two existing `require_histogram_launchable` calls at
  `src/mojoboost/histogram_gpu.mojo:466` (whole-matrix launch in the
  constructor) and `:594` (narrowed grid in `set_features`).
- **State flow:** `self.report` in, a refusal or nothing out. The gate
  changes no geometry: `self.tiling` is still
  `gpu_tiling.derive_tiling`'s. Making the *plan* backend-derived as well
  is Patch 4 and is deliberately separate, because a gate that only refuses
  cannot change a result and a plan that changes geometry can.
- **Errors:** the backend gates add the in-order-queue check and the static
  shared-memory ceiling. Neither can fire on today's path: the contract
  promises the queue on every covered backend, and the kernels declare
  3 KiB against a 48 KiB floor.
- **Ownership:** `histogram_gpu.mojo` is the trainer/kernel lane's.
- **Fallback preserved:** the `else` branch is byte-for-byte today's call.
  An `API_UNKNOWN` device, which is every device on today's path because
  nothing reads `ctx.api()` into the profile, takes it.
- **Serialization effect:** none. `GpuHistogramBuilder` is session state
  and is not serialized; no model field changes.
- **Public API effect:** none.
- **Dependency:** Patches 1 and 2.
- **Later validation (UNRUN):**
  `pixi run mojo run -I src tests/parallel/test_histogram_gpu.mojo` if such
  a file exists on the machine holding a device; on a CPU-only machine, a
  compile of `histogram_gpu.mojo` is the whole check.

---

### Patch 4: let the backend supply the shape-level bounds

- **Target file:** `src/mojoboost/apple_histogram_policy.mojo`
- **Target symbol:** `derive_histogram_plan`
- **Signature change (append two defaulted parameters, so no existing call
  site changes):**
  ```
  requested_level: Int = SPEC_LEVEL_UNSET,
  max_partial_cells: Int = 0,
  resident_override: Int = 0,
  granularity_override: Int = 0,
  ```
- **Body change:** inside the `if applied >= SPEC_LEVEL_SHAPE:` block, use
  `resident_override` in place of
  `resident_blocks_per_core(profile, features, capacity)` when it is
  positive, and pass `granularity_override` to `clamp_block_threads` in
  `_shape_block_threads` when it is positive.
- **Call site:** `histogram_gpu.GpuHistogramBuilder.histogram_plan`
  (`src/mojoboost/histogram_gpu.mojo:808`), which would pass
  `plan.occupancy.blocks_per_multiprocessor` and
  `plan.block_width_granularity` from a `derive_cuda_plan` or
  `derive_amd_plan` result on a CUDA or HIP device, and `0`/`0` elsewhere.
- **State flow:** two Ints. No new struct crosses the boundary.
- **Errors:** unchanged. A nonpositive override means "not supplied", not
  an error, matching how `max_partial_cells` already behaves.
- **Ownership:** `apple_histogram_policy.mojo` is the Apple-policy lane's,
  which this lane was told not to edit.
- **Fallback preserved:** both overrides default to 0 and the Metal path is
  untouched.
- **Serialization effect:** none.
- **Public API effect:** none; the module is not exported.
- **Dependency:** Patch 3 for the values to come from.
- **Why it matters:** it collapses three per-backend shape derivations into
  one planner. Without it, `derive_cuda_plan` and
  `derive_histogram_plan` are two entry points that both call
  `resolve_tiling` with different bounds, which is correct but is two
  places to keep honest.
- **Later validation (UNRUN):** a focused assertion that
  `derive_histogram_plan(..., resident_override=0, granularity_override=0)`
  is field-for-field the plan it produces today.

---

### Patch 5: record the extra attributes on a device decision

- **Target file:** `src/mojoboost/device_policy.mojo`
- **Target symbol:** `capabilities_from_reported`
  (`src/mojoboost/device_policy.mojo:1931`)
- **Signature change (append, all defaulted to 0):**
  `warp_size: Int = 0, max_threads_per_multiprocessor: Int = 0,
  max_blocks_per_multiprocessor: Int = 0, max_registers_per_block: Int = 0,
  max_grid_dim_x: Int = 0, max_grid_dim_y: Int = 0`
- **Body change:** carry them on `DeviceCapabilities` as a
  `DeviceReport` field and emit them in `DeviceDecision.serialize` as
  `attr_warp_size=`, `attr_threads_per_sm=`, and so on, plus
  `attributes_answered=N/10`.
- **Call site:** `decide_device_report_reported`
  (`src/mojoboost/device_policy.mojo:2013`), whose flat-scalar signature
  grows by six Ints. That function is the seam the bindings, the C API, and
  the CLI bind, so the six additions are the whole boundary change.
- **State flow:** attributes in, `key=value` lines out. **No gate consults
  any of them.** A decision must not start depending on a subgroup width.
- **Errors:** none added.
- **Ownership:** `device_policy.mojo` is the device-policy lane's.
- **Fallback preserved:** every new parameter defaults to unreported, and
  a decision built without them serializes exactly as it does today except
  for one extra `attributes_answered=0/10` line.
- **Serialization effect:** **yes, this one has one.**
  `DeviceDecision.serialize` gains up to seven lines. Any consumer that
  parses that format strictly (the bindings, the C API, the CLI, and
  whatever `tools/` reads it) must tolerate unknown keys before this lands.
  If they do not, split it: land the fields without the serialization first.
- **Public API effect:** `decide_device_report_reported` is re-exported
  through `src/mojoboost/device.mojo`, so its signature is public. Appending
  defaulted parameters is source-compatible for keyword and short
  positional callers.
- **Dependency:** Patch 1, for `DeviceReport` to live in a module
  `device_policy.mojo` can import without closing a cycle. Note that
  `device_policy.mojo` deliberately does not import the GPU kernel stack;
  `gpu_portability.mojo` imports `gpu_tiling.mojo`, which imports
  `max.gpu.host`, so **this patch as written would pull a device-side
  import into a module that must stay CPU-testable.** Resolve that first:
  either put `DeviceReport` in a leaf module with no `max.gpu.host` import
  (`apple_gpu_policy.mojo` is such a leaf), or pass the six Ints and keep
  the struct out of `device_policy.mojo`. The second is smaller and is what
  this handoff recommends.
- **Later validation (UNRUN):** a focused assertion that a decision built
  with no reported attributes serializes to the same string as today plus
  the one `attributes_answered` line.

---

### Patch 6: packaging manifest

- **Target file:** `packaging/` module manifest (whichever file enumerates
  `src/mojoboost/*.mojo` for the wheel; this lane did not open
  `packaging/`).
- **Change:** add `gpu_cuda_policy.mojo` and `gpu_amd_policy.mojo`.
- **State flow, errors, serialization:** none.
- **Public API effect:** none.
- **Fallback:** if the manifest is a glob, no change is needed and this
  patch is a no-op; confirm before applying.
- **Dependency:** none.
- **Later validation (UNRUN):** the existing artifact validator, whatever
  `packaging/matrix/smoke/` runs.

---

### Patch 7: documentation and parity rows

- **Target files:** `docs/LIGHTGBM_PARITY.md`, `docs/GPU_VALIDATION.md`,
  `tools/check_parity.py`.
- **`docs/LIGHTGBM_PARITY.md`**, two rows in the same shape as the existing
  "Apple GPU tuning policy" row, which is the precedent for an implemented
  and unintegrated policy module:

  ```
  | NVIDIA GPU backend policy | deferred | yes | no | no | no | n/a | no | n/a | `src/mojoboost/gpu_cuda_policy.mojo`. Nothing reads it; `src/mojoboost/gpu_tiling.mojo` is still the geometry in force and no NVIDIA device has run this project's kernels |
  | AMD GPU backend policy | deferred | yes | no | no | no | n/a | no | n/a | `src/mojoboost/gpu_amd_policy.mojo`. Same status; imports the backend-neutral half from the NVIDIA module until it relocates to `gpu_portability.mojo` |
  ```

  Check the column count against the live header before pasting; this lane
  read the file's row shape but did not re-verify the header width.
- **`docs/GPU_VALIDATION.md`:** add the per-backend attribute capture table
  described in `docs/GPU_BACKEND_SPECIALIZATIONS.md`, seeded with the
  measured Metal column already in that file.
- **`tools/check_parity.py`:** watch `derive_cuda_plan` and
  `derive_amd_plan` as the public symbols behind the two new rows, so the
  rows stop reading `deferred` when a caller appears. Static inspection
  only; do not import the package.
- **Ownership:** documentation and tooling lanes.
- **Later validation (UNRUN):** `pixi run check-parity`.

---

## Fallbacks preserved

- **Every plan carries its baseline.** `BackendLaunchPlan.baseline` is what
  `gpu_tiling.derive_tiling` would have launched, obtained by calling it.
  `matches_baseline()` compares strategy, block width, tile count, and rows
  per tile. A caller that distrusts a backend plan launches
  `plan.baseline` with no re-derivation.
- **An unreported device plans portably.** `DeviceReport.unreported()`
  takes the portable granularity (`clamp_block_threads`), the portable
  residency target (`TARGET_BLOCKS_PER_SM`), and the portable partial
  budget, so it matches the baseline by construction.
- **Every specialization stays off.** `KernelFeatures.none()` is what every
  build reports for the variants that matter here, and
  `require_specializations_allowed` refuses the rest on CUDA and HIP
  without `MOJOBOOST_GPU_BACKEND_UNVALIDATED=1`.
- **No new environment variable.** The three existing overrides
  (`MOJOBOOST_GPU_ROW_TILE`, `MOJOBOOST_GPU_BLOCK_THREADS`,
  `MOJOBOOST_GPU_HIST_STRATEGY`) reach every plan because the arithmetic
  that honors them is the shared `resolve_tiling`.

## Remaining disconnections

Stated rather than papered over.

1. **Nothing imports either module.** Both are `deferred`. Patches 2 and 3
   are what change that.
2. **No API name reaches the policy layer.**
   `apple_histogram_policy.profile_from_caps` hardcodes `API_UNKNOWN`, and
   `histogram_gpu.mojo` builds its contract from it, so
   `require_cuda_launchable` and `require_amd_launchable` would never
   dispatch even after Patch 3. Reading `ctx.api()` and `ctx.arch_name()`
   into `GpuProfile.from_reported` at the builder is the missing half;
   `bench/bench_gpu_validation.mojo:133` shows both accessors exist. This
   belongs in Patch 3's lane and is called out here because Patch 3 alone
   is inert without it.
3. **No memory budget accessor anywhere.** `GpuProfile.memory_budget_bytes`
   is 0 on every path in this repository, so `partial_budget_for` always
   returns the portable ceiling and the unified/discrete fraction never
   fires. Same gap `apple_gpu_policy.mojo` and
   `apple_histogram_policy.profile_from_caps` already document.
4. **`unified_memory` is unreachable on CUDA and HIP.**
   `GpuProfile.from_reported` sets it `api == API_METAL`, so a HIP APU
   cannot report itself unified today even though the policy is written to
   accept it. Fixing that is a change to `apple_gpu_policy.mojo`, which
   this lane does not own.
5. **No test file.** This lane was instructed not to write or run one. The
   focused tests that should exist are listed below, all UNRUN.
6. **The `gpu_amd_policy -> gpu_cuda_policy` import edge.** Correct but
   awkward; Patch 1 removes it.

## Risks

- **Compile risk against a moving file.** `gpu_portability.mojo` was being
  written during this task. Eight imported symbols were verified present at
  12:51; a rename in that lane breaks both new modules until the import
  block is updated. No logic depends on the spelling.
- **The occupancy change is the one behavior change.** On a device that
  reports the two per-multiprocessor attributes, residency stops being 8
  and becomes a derived number, which changes the tile count and therefore
  which strategy `STRATEGY_AUTO` resolves to. That is the intended effect
  and it is unmeasured on any NVIDIA or AMD part. It cannot reach today's
  path (nothing calls the modules) and, once wired, cannot reach Metal
  (which refuses both attributes).
- **`require_cuda` and `require_amd` refuse `API_UNKNOWN`.** If a future
  caller dispatches on a guess rather than on a declared or reported API
  name, it gets a raise. That is deliberate; the message names the
  `MOJOBOOST_GPU_BACKEND` declaration that fixes it.
- **The CUDA 32-lane rounding can produce a narrower block than the
  portable rule.** A 40-row node on a CUDA device reporting `WARP_SIZE=32`
  gets a 32-thread block where `clamp_block_threads` floors at 64. Legal on
  the device, unmeasured, and reachable only when the width was actually
  reported.
- **Patch 5 has a real serialization effect** and a real layering hazard
  (pulling `max.gpu.host` into a CPU-testable module). Both are called out
  in the patch. Do not apply it as literally written without resolving the
  import direction.

## Smallest later focused commands, all UNRUN

Nothing below was executed. Each is one command, one file.

```
# Compile the two new modules (CPU-only machine is enough; both are pure).
pixi run mojo build -I src src/mojoboost/gpu_cuda_policy.mojo   # UNRUN
pixi run mojo build -I src src/mojoboost/gpu_amd_policy.mojo    # UNRUN

# The focused test this lane was forbidden to write, once it exists.
pixi run mojo run -I src tests/parallel/test_gpu_backend_policies.mojo  # UNRUN

# Parity rows, after Patch 7.
pixi run check-parity                                            # UNRUN
```

The focused test, when someone writes it, should assert exactly these and
nothing that needs a device:

1. `DeviceReport.unreported()` produces a plan with
   `matches_baseline() == True` across the four default bench shapes.
2. `resident_blocks_from_reported(0, 0, 256) == TARGET_BLOCKS_PER_SM` and
   `source == RESIDENCY_PORTABLE_TARGET`.
3. `resident_blocks_from_reported(2048, 32, 256)` is 8 with
   `source == RESIDENCY_BY_THREADS`, and swapping the arguments to
   `(2048, 4, 256)` gives 4 with `source == RESIDENCY_BY_BLOCKS`.
4. `require_subgroup_width_plausible` raises for a CUDA report of 64 and
   passes for 32; the AMD one passes for both and raises for 16.
5. `require_cuda(API_UNKNOWN)` and `require_amd(API_UNKNOWN)` both raise.
6. `require_shared_within_ceiling(API_CUDA, 49152 + 1, 49152)` raises and
   `(API_HIP, 49152 + 1, 65536)` does not.
7. `atomic_conflict_degree(STRATEGY_ATOMIC, 7) == 7`,
   `atomic_conflict_degree(STRATEGY_TILED, 7) == 1`, and
   `STRATEGY_AUTO` raises.
8. `preferred_strategy` returns `STRATEGY_AUTO` for every constructible
   `StrategyInputs`, which is the assertion that keeps an unmeasured rule
   from being installed quietly.
9. `packed_body_transactions` on the same window is exactly twice the AMD
   value on CUDA, whenever the body is a multiple of 64 bytes.
10. Every `BackendSpecialization` from either module reports
    `device_float64_permitted == False` and `concurrent_queues == False`.
