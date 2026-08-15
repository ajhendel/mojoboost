# GPU backend specializations: NVIDIA and AMD

Written: 2026-08-14
Policy source: `src/mojotrees/gpu_vendor_policy.mojo` (the
`gpu_cuda_policy.mojo` and `gpu_amd_policy.mojo` twins this document was
written against were merged into it on 2026-08-15, f23bd1b; the names below
that still say `gpu_cuda_policy` / `gpu_amd_policy` describe that history
and now resolve to `gpu_vendor_policy`)

## Status, stated first

Nothing in this document is a hardware validation claim.

This repository has executed a GPU kernel on exactly one device class,
Apple silicon, on one Apple M4. `docs/GPU_VALIDATION.md` is the record and
every CUDA row and every HIP row in it reads "not run".
`gpu_backend_policy.backend_support` returns `SUPPORT_PORTABLE` for both
backends, which means the shared source targets them and nobody here has
seen it run on them.

Against `docs/CAPABILITY_LEVELS.md`, the two modules described here are:

| Level | NVIDIA policy | AMD policy | Evidence |
|---|---|---|---|
| implemented | yes | yes | `src/mojotrees/gpu_vendor_policy.mojo` (formerly `gpu_cuda_policy.mojo` + `gpu_amd_policy.mojo`) |
| integrated | no | no | no shipping code path imports either module; see "Exact integration requests" |
| publicly reachable | no | no | not exported from `src/mojotrees/__init__.mojo` |
| focused-tested | no | no | no test file exists; this lane was forbidden to write or run one |
| differential-tested | n/a | n/a | LightGBM has no counterpart to a per-backend launch policy |
| hardware-validated | no | no | no NVIDIA or AMD device has ever run this project's kernels |
| release-packaged | no | no | nothing in `packaging/` references either module |

`docs/LIGHTGBM_PARITY.md` should carry these two rows as `deferred`, which
is the status that file defines for an implemented and unintegrated module.
The exact row text is in `handoffs/remaining_11_gpu_backends.md`.

## Why a per-backend policy exists at all

The design commitment is one source. There is no CUDA file, no HIP file,
and no Metal file: `histogram_gpu.mojo`, `gpu_active_rows.mojo`,
`train_gpu.mojo`, and `gpu_predict.mojo` are compiled for whichever device
MAX opens. `gpu_portability.mojo` writes down what that source is allowed
to require of any device, and `gpu_tiling.mojo` writes down how a launch
geometry is derived from bounds.

What was missing between them is the fact that the three backends do not
answer the same questions. Metal refuses six of the ten device attributes
this package knows how to ask for. CUDA and HIP answer more of them, and
two of the ones Metal refuses are exactly the two that bound occupancy.
A policy layer that plans from three attributes on every backend is
throwing away the extra answers on the two backends that give them.

These two modules are that layer, and they are only that layer. They add
bounds. They do not add kernels, they do not fork the algorithm, and they
do not restate any arithmetic that already exists lower down.

## Layering

```
gpu_tiling.mojo                  tile arithmetic, block-width rounding,
                                 partial budget, launch counts
    ^
gpu_histogram_specializations    bin capacities, shared footprints,
                                 packed windows, storage descriptor
    ^
gpu_backend_policy.mojo          which backends are covered, which have
                                 ever been run
    ^
gpu_portability.mojo             the primitive set, the launch gates, the
                                 per-backend contract
    ^                    ^                          ^
apple_histogram_policy   gpu_cuda_policy.mojo  ->  gpu_amd_policy.mojo
(Metal + the ladder)     (NVIDIA)                  (AMD, importing the
                                                    neutral half)
```

The arrow between the two new modules is a real import and it is
deliberate. `DeviceReport`, `Occupancy`, `BackendLaunchPlan`,
`BackendSpecialization`, `StrategyInputs`, and the derivations over them
are backend-neutral, and a second copy of the occupancy rule or the plan
assembly is precisely the duplicate policy engine this work exists to
avoid. Their home is `gpu_portability.mojo`; that file belongs to another
lane, so the relocation is a ready-to-apply patch in the handoff rather
than an edit.

## The eleven specialization points

Each row states what is encoded, where the number comes from, and what
happens when the device does not report it. "Reported" always means an
attribute `DeviceContext.get_attribute` exposes and this repository
already queries in `bench/bench_gpu_validation.mojo`.

### 1. Subgroup width

| | Metal | CUDA | HIP |
|---|---|---|---|
| `WARP_SIZE` answered | no (measured on M4) | expected yes | expected yes |
| Value used | 0, unknown | reported, else 0 | reported, else 0 |
| Plausible values | n/a | 32 | 32 or 64 |

Neither module substitutes an architectural constant for a missing report.
`CUDA_WARP_LANES` and the two `AMD_WAVEFRONT_*` constants exist only so
`require_subgroup_width_plausible` can refuse a positive report that
cannot be that backend's width, which would mean the attribute was misread
or the API code is wrong. `require_subgroup_width_known` is the gate a
future cross-lane specialization passes; nothing calls it, because no
kernel here performs a shuffle, a ballot, or a warp-level reduction.

AMD is the reason this is done by report rather than by constant at all.
CDNA executes 64 lanes and RDNA executes 32, so a guess on that backend is
as likely to be wrong as right.

### 2. Thread-block geometry

`derive_block_threads_for` takes the portable `TARGET_BLOCK_THREADS`,
bounds it by the rows available, clamps it to the reported device maximum,
and rounds down to `block_width_granularity`.

The granularity is the whole backend difference:

- Width not reported: `gpu_tiling.WARP_GRANULARITY` (64), and the
  derivation delegates to `gpu_tiling.clamp_block_threads`, so the width is
  the shipping one.
- CUDA reporting 32: a narrow node may take a 32-thread block where the
  portable rule floors at 64. This is the one place either module can
  produce a launch the portable rule would not have.
- HIP reporting 64: identical to the portable rule, because 64 already is
  the CDNA wavefront.
- HIP reporting 32: refines to 32 for an RDNA part.

`wavefront_matches_granularity` in the AMD module is the single line a
capture on new AMD hardware should be read against, because it is where
the portable constant and the device stop agreeing.

### 3. Shared memory

Two ceilings, checked separately because they are different kinds of fact.

| Ceiling | CUDA | HIP | Source |
|---|---|---|---|
| Static declaration allowed by the programming model | 48 KiB | 64 KiB | model, not device |
| Per-block limit the device advertises | reported | reported | `MAX_SHARED_MEMORY_PER_BLOCK` |

The shared source declares its histogram planes with
`stack_allocation[..., AddressSpace.SHARED]`, which is a static
declaration, and takes no dynamic-shared-memory opt-in, so the model
ceiling binds it. What is checked against both is
`gpu_portability.kernel_shared_request`, the footprint the compiled kernel
really has, not the `n_bins * 12` model whose own docstring records that it
is optimistic below 256 bins.

Both checks are inert today. The shipping kernels request 3 KiB. They are
written down so that stays a fact rather than an assumption.

### 4. Atomics

No specialization, and this is the point. Accumulation is fixed-point Int32
in threadgroup memory and, under the atomic strategy, Int32 in device
memory. Integer addition is associative, so strategy choice never changes a
histogram, and there is no float atomic anywhere in this package. Neither
module relaxes that on a backend that has faster float atomics, because
doing so would make a histogram depend on scheduling.

What both modules add is `atomic_conflict_degree`, which reports how many
threadgroups fold into the same output cell under a resolved strategy: the
tile count under `STRATEGY_ATOMIC`, and one under `STRATEGY_TILED`, where
each tile owns its own slot. It is a structural property of the two
flushes and a reported quantity, not a decision.

The primitive requirements themselves stay in
`gpu_portability.required_primitives`, which both modules reach through
`require_histogram_launchable`.

### 5. Occupancy bounds

This is the largest genuine difference between the backends.

| Attribute | Metal (M4, measured) | CUDA | HIP |
|---|---|---|---|
| `MAX_THREADS_PER_MULTIPROCESSOR` | refused | expected | expected |
| `MAX_BLOCKS_PER_MULTIPROCESSOR` | refused | expected | expected |
| shared memory per multiprocessor | not a `DeviceAttribute` member at all | same | same |

`resident_blocks_from_reported` takes the tightest of the two reported
bounds, floors at one, and falls back to `gpu_tiling.TARGET_BLOCKS_PER_SM`
when neither was reported. `Occupancy.source` says which bound produced
the number, so a residency figure can be told from the fixed target.

It deliberately does not reproduce
`apple_gpu_policy.resident_blocks_per_core`, which divides the per-*block*
shared-memory limit by a per-block footprint. On a backend that answers the
two per-multiprocessor attributes, that estimate answers a different
question. On one that does not, this falls back to the same fixed target
`derive_tiling` already uses rather than to an estimate built from a limit
that was never a pool.

Two bounds are recorded as knowingly absent rather than silently omitted:
`Occupancy.shared_bound_known` is always false because the per-SM shared
pool is not queryable, and `Occupancy.registers_bound_known` is always
false because `MAX_REGISTERS_PER_BLOCK` is reported but registers per
thread for the compiled kernel are not.

### 6. Partial histogram strategy

No backend preference. `preferred_strategy` returns `STRATEGY_AUTO` on
every shape on both backends, which means the portable rule in
`gpu_tiling.resolve_tiling` decides: tiled when there is more than one tile
to reduce, atomic when there is not.

It is widely held that NVIDIA device-memory atomics are cheap at low
contention and that a tiled reduction wins at high contention. That is
exactly the kind of claim this repository has no measurement for on any
NVIDIA or AMD part, and installing it from reasoning is what
`device_policy.crossover_rules` refuses to do for the CPU/GPU threshold on
any device that has not run the sweep (its two rules are Apple M4 rules
from a recorded sweep, and nothing else is installed) for the same reason.

`StrategyInputs` reports what a measured rule would key on (tile count,
slot count, bin count, block width, residency, conflict degree, partial
cells, partial bytes) and carries `TILED_PREFERENCE_UNMEASURED` where the
threshold would go. Fill that in from a recorded sweep, cite the record,
and bump `device_policy.POLICY_VERSION`.

### 7. Packed-bin alignment

The window arithmetic is not re-derived. Both modules call
`gpu_histogram_specializations.plan_packed_window_for`, which takes a
`BinStorageDescriptor` and refuses every layout the four-lane arithmetic is
false for. Re-deriving a byte offset in a backend module is precisely how a
second copy comes to read the wrong bytes.

What each module adds is the transaction unit the window is measured in.

| | CUDA | HIP |
|---|---|---|
| Coalescing unit | 32-byte sector | 64-byte cache line |
| Widest single load the backend offers | 16 bytes (`int4`) | 16 bytes (`dwordx4`) |
| What the shared source emits | 4 bytes (`pack4_bins`) | 4 bytes (`pack4_bins`) |

`packed_body_transactions` reports how many transactions a window's packed
body costs. It is a diagnostic. Nothing selects the packed path on it, and
the packed path itself is gated twice over: the kernel variant must be
compiled in (`KernelFeatures.packed_bin_loads`, false in every build today)
and the backend must have been exercised
(`gpu_portability.require_specializations_allowed`, which refuses CUDA and
HIP unless `MOJOTREES_GPU_BACKEND_UNVALIDATED=1`).

### 8. Allocation

`allocation_plan` calls `unified_memory_policy.plan_session_routes` with
the reported unified-memory flag and returns its `SessionMemoryPlan`. That
module owns the route vocabulary, the evidence ladder, and the refusals; a
second route table in a backend module that disagreed with it by one role
would be worse than none.

The backend fact is that unified memory does not follow from the API name
outside Metal:

| Backend | Unified inferable from the API name | Why |
|---|---|---|
| Metal | yes | every Apple silicon GPU is unified and no discrete one is |
| CUDA | no | a discrete card, an integrated part, and managed memory all report "cuda" |
| HIP | no | a discrete card, an XNACK-enabled part, and an APU all report "hip" |

`DeviceReport.unified_memory` therefore has to be reported and defaults
false, which routes every buffer through the staged copy
(`ROUTE_COPY_STAGED`) that `GpuHistogramBuilder` already requires. Nothing
about a CUDA or HIP session's allocation changes; the modules only say why.

`partial_budget_for` reuses `apple_gpu_policy.partial_budget_bytes`, which
is not Apple-specific despite where it lives: a fraction of the reported
budget, the tighter fraction when memory is unified, capped by the portable
ceiling, and the whole portable ceiling when no budget was reported.

### 9. Streams

`concurrent_queues_available` returns false on both backends, and
`require_concurrent_queues` is a capability error rather than a silent
serialization.

This is a statement about this repository, not about CUDA or HIP. Both APIs
have streams. This package holds one `DeviceContext` per session
(`gpu_runtime.GpuSession` owns exactly one) and reaches it through
`enqueue_create_buffer`, `enqueue_create_host_buffer`, `enqueue_copy`,
`enqueue_memset`, and `synchronize`. No abstraction here reaches a second
queue, and claiming overlap the code does not implement would make
`gpu_runtime.PhaseCounters` attribute time to a concurrency that never
happened.

### 10. Synchronization

`require_in_order_queue` checks the backend contract's
`REQ_IN_ORDER_QUEUE`. That is the property `gpu_runtime.HazardTracker`
rests on: device work never needs a host synchronization to observe earlier
device work, so the only required synchronizations are the two host-side
hazards it tracks (host reads memory the device has an unretired write to,
host writes memory the device has an unretired read or write to).

Both CUDA and HIP promise it for a single stream, which is the only queue
model this package uses. Nothing else about synchronization is
backend-specific, and neither module tracks a hazard: that model has one
implementation and it is `gpu_runtime.mojo`.

### 11. Capability errors

Every refusal either delegates to an existing gate or is new because the
fact is not portable.

| Error | Module | Raised when |
|---|---|---|
| `require_cuda` / `require_amd` | new | the API code is not this module's, including `API_UNKNOWN` |
| `require_subgroup_width_plausible` | new | a positive reported width cannot be this backend's |
| `require_subgroup_width_known` | new | a specialization needs a width the device did not report |
| `require_shared_within_ceiling` | new | a static threadgroup declaration exceeds the model ceiling |
| `require_shared_reported_fits` | new | it exceeds what the device advertises per block |
| `require_in_order_queue` | new | the contract does not promise an in-order queue |
| `require_concurrent_queues` | new | a caller needs two queues to overlap |
| `require_backend_covered` | delegated | the API code has no contract at all |
| `require_specializations_allowed` | delegated | a compiled-in kernel variant is selected on an unexercised backend |
| `require_histogram_launchable` | delegated | bins, primitives, or geometry are outside what the device can run |

`require_cuda` and `require_amd` refuse `API_UNKNOWN` on purpose. A device
that did not name its API might be NVIDIA and might not, and planning a
48 KiB static ceiling and a 32-lane rounding granularity for an
unidentified device is a guess. The portable path already covers it; an
operator who knows better declares `MOJOTREES_GPU_BACKEND`.

## The default-off guarantee

`BackendLaunchPlan` carries `baseline`, which is what
`gpu_tiling.derive_tiling` would have launched for the same node on the
same device, obtained by calling it. `matches_baseline` compares the
strategy, the block width, the tile count, and the rows per tile.

A device that reports only the three attributes `DeviceCaps` already
carries takes the portable granularity, the portable residency target, and
the portable partial budget, so its plan matches the baseline. Everything
either module can do differently rests on an attribute the device answered.
A caller that does not trust a plan can launch `plan.baseline` instead
without re-deriving anything.

The environment overrides reach every plan, because the arithmetic that
honors them is the shared `gpu_tiling.resolve_tiling`:
`MOJOTREES_GPU_ROW_TILE`, `MOJOTREES_GPU_BLOCK_THREADS`, and
`MOJOTREES_GPU_HIST_STRATEGY`. Neither module introduces an environment
variable of its own.

## Exact integration requests

None of these were applied. Each one is another lane's file, and the
mechanically applicable form of each, with signatures and call sites, is in
`handoffs/remaining_11_gpu_backends.md`.

### Shared kernels and the GPU dataflow lane

1. `src/mojotrees/gpu_portability.mojo`: accept the relocation of the
   backend-neutral half of `gpu_cuda_policy.mojo` (`DeviceReport`,
   `Occupancy`, `BackendLaunchPlan`, `BackendSpecialization`,
   `StrategyInputs`, `resident_blocks_from_reported`,
   `derive_block_threads_for`, `require_shared_within_ceiling`,
   `require_shared_reported_fits`, `derive_plan_for`, and the two
   `describe_*`), and re-point both backend modules' imports at it. This
   removes the `gpu_amd_policy -> gpu_cuda_policy` edge.
2. `src/mojotrees/gpu_tiling.mojo`: add `query_full_device_report(ctx)`
   beside `query_device_caps`, reading the ten attributes
   `bench/bench_gpu_validation._report_device` already reads and returning
   a `DeviceReport`. It is the only new device-touching code any of this
   needs, and it is four lines per attribute over the existing
   `_attribute_or`.
3. `src/mojotrees/apple_histogram_policy.mojo`: give
   `derive_histogram_plan` optional `resident_override` and
   `granularity_override` parameters so `SPEC_LEVEL_SHAPE` can be handed a
   residency derived from the per-multiprocessor attributes and a rounding
   granularity derived from a reported subgroup width. That collapses the
   three per-backend shape derivations into one planner.

### Trainer

4. `src/mojotrees/histogram_gpu.mojo`: hold a `DeviceReport` beside
   `DeviceCaps` on `GpuHistogramBuilder`, and call the backend gate
   (`require_cuda_launchable` or `require_amd_launchable`, selected by the
   profile's API code, with `apple_histogram_policy` unchanged for Metal)
   once per launch. On today's path the report is `unreported()`, the
   contract is the `API_UNKNOWN` one, and every gate passes, so this is
   inert until a report is filled in.
5. `src/mojotrees/train_gpu.mojo`: no change requested. Device selection
   happens before a backend is known and belongs to `device_policy.mojo`.

### Device policy

6. `src/mojotrees/device_policy.mojo`: extend
   `capabilities_from_reported` to take the six additional attributes and
   record them, so `decide_device_report_reported` can report a backend's
   answered-attribute count. A decision does not gate on any of them; the
   value is that a support ticket carries the device's own answers.
7. `src/mojotrees/device_policy.mojo`: when a crossover rule is eventually
   installed for CUDA or HIP, `CrossoverEvidence.api` already scopes it and
   `POLICY_VERSION` already versions it. No change needed now, and none
   should be made from reasoning.

### Packaging

8. `packaging/`: no artifact change. Both modules are pure host arithmetic
   with no new dependency, so they are included by whatever includes
   `src/mojotrees/*.mojo`. The request is only that the manifest check that
   lists modules gains these two, so a wheel that silently drops them is
   caught.
9. `packaging/`: `MOJOTREES_GPU_BACKEND_UNVALIDATED` is an existing knob
   owned by `gpu_backend_policy.mojo`; document it in the installation
   notes as "acknowledges running an unvalidated specialization", not as a
   performance switch.

### Validation

10. `docs/GPU_VALIDATION.md`: add a per-backend attribute capture table
    with a row per attribute and a column per backend, seeded with the
    measured Metal column already in that file (five answered of ten). The
    first NVIDIA or AMD run fills a column, and
    `DeviceReport.answered()` is the number it reports.
11. `docs/GPU_VALIDATION.md`: the first CUDA capture should be checked
    against `require_subgroup_width_plausible` (expect 32) and the first
    HIP capture against the same (expect 32 or 64), and any refusal
    recorded rather than worked around.
12. `docs/LIGHTGBM_PARITY.md`: two `deferred` rows, one per module, with
    the level table above.
13. `tools/check_parity.py`: watch the public symbols behind those rows
    (`derive_cuda_plan`, `derive_amd_plan`) for the drift that file already
    watches for, so the rows stop reading `deferred` when a caller appears.

## What would falsify the design

Stated so a first run on real hardware has something to disagree with.

- If CUDA or HIP refuses `MAX_THREADS_PER_MULTIPROCESSOR` and
  `MAX_BLOCKS_PER_MULTIPROCESSOR` the way Metal does, the occupancy
  derivation never fires and `Occupancy.source` reads
  `portable_target` everywhere. The modules then add nothing but the
  ceilings and the refusals, and the residency section above should be
  rewritten rather than kept as an aspiration.
- If a CUDA device reports a warp size other than 32, or an AMD device
  reports anything but 32 or 64, `require_subgroup_width_plausible` raises.
  That is the intended behavior and it means one of the attribute, the
  report, or the API code is wrong. It is not a reason to widen the check.
- If the reported per-block shared memory is ever below the 3 KiB the
  shipping kernels declare, `require_shared_reported_fits` raises where
  `gpu_tiling.derive_tiling` would have passed. That gap is the one
  `gpu_portability.kernel_shared_request` exists to close, and a device in
  it would prove the optimistic model wrong rather than the gate.
