# Handoff: device policy authority (migration 20, continued)

The original migration 20 note, which made `device_policy.mojo` the one
implementation of the device decision and `device.mojo` a facade over it,
was removed with the other closed handoffs in 21ff9fa. It is recoverable
with `git show 21ff9fa^:handoffs/migration_20_device_policy.md`. Its intent
was coherent and is honored below rather than reversed.

## consolidation K2 (2026-08-15)

Scope: the competing device-selection and GPU-policy tables. Commits, all
by explicit path: d104dfc (A), ff3717a and 923c812 (B), d9759ec (D),
f23bd1b (C). Audit ran before and after; every duplicate name in K2's
section is gone (device vs device_policy: 15 names; apple_gpu_policy vs
gpu_tiling: 12 names; gpu_amd_policy vs gpu_cuda_policy: 17 names;
HISTOGRAM_BYTES_PER_CELL: 1; describe_plan and resident_blocks_per_core:
2).

### A. device.mojo vs device_policy.mojo

Decision: migration 20's direction stands. `device_policy.mojo` is the
implementation; `device.mojo` is now a **pure re-export** (one
`from .device_policy import (...)` block, no `def`, no `comptime`), so
`AUTO_DEVICE`, `CPU_DEVICE`, `GPU_DEVICE`, `NO_DEVICE`, `AUTO_MIN_CELLS`,
`BINS_UNSPECIFIED`, `OBJECTIVE_UNSPECIFIED`, `parse_device`, `device_name`,
`gpu_available`, `env_auto_min_cells`, `gpu_supports_outputs`,
`resolve_device`, `resolve_device_full`, `decide_device_report`, and
`decide_device_report_reported` are each defined exactly once. Every
importer of `mojotrees.device` (model, params, trainset, alternate_boosting,
boosting_dart, external_memory, bindings/_mojotrees, bindings/basic_bindings,
capi, cli, tests) keeps compiling unchanged; Mojo re-exports imported names
from a module the same way `__init__.mojo` does.

Deleted: `device.gpu_supports` (the pre-migration alias of
`gpu_supports_outputs`). Zero callers anywhere in the repository (the only
textual hit was a regex in tools/connectivity_audit.py). Not exported from
`__init__.mojo`.

Verified: a scratch test importing every re-exported name through
`mojotrees.device` and comparing against `mojotrees.device_policy` passed.
`tests/test_device.mojo` itself could not be run: it imports `model.mojo`,
which imports `train_gpu.mojo`, and a peer lane's uncommitted edit to
`train_gpu.mojo` (line 2515, `data=data`) does not parse at the moment. Run
it once that lane lands; nothing in device.mojo depends on it.

Not a duplicate, left alone: `EVIDENCE_NONE`, `block_reason_name`, and
`describe_decision` existed in both `device_policy` and
`unified_memory_policy` but name different facts (device_policy's is
`describe_device_decision` since the 2026-08-15 cleanup pass; the transfer
one keeps the short name for its `histogram_gpu` caller) (device-decision evidence
identifiers and block codes vs transfer-route evidence levels and block
codes); device_policy line 166 already says so and imports the transfer
names under `transfer_*` aliases. connect_05 owns unified_memory_policy.

Deferred: device_policy's three "mirrors" of histogram_gpu constants
(`MAX_GPU_ROWS`, `MIN_GPU_BINS`, `MAX_GPU_BINS`) are differently named and
still unpinned by any test, as its own comment says. Importing them would
pull histogram_gpu's kernel stack into params.mojo's import graph. Left as
is; a `test_device_policy.mojo` that pins them is still owed (connect_05's
spec).

Audit note for C0: with device.mojo a pure facade, the audit now reports
"device.mojo imports ... and uses none of them" against device_policy, the
same shape it already reports for `__init__.mojo`'s re-export block. That
is the facade working, not a dead edge. A peer commit (13d87ef) taught the
audit not to count re-exported constants as duplicate definitions; the
unused-import pass may want the same facade awareness. I did not edit the
audit.

### B. apple_gpu_policy.mojo vs gpu_tiling.mojo

`gpu_tiling` is the authority (20 importers). apple_gpu_policy now imports
and re-exports `STRATEGY_AUTO/ATOMIC/TILED`, `TARGET_BLOCK_THREADS`,
`WARP_GRANULARITY`, `BYTES_PER_PARTIAL_CELL`, `MAX_GRID_DIM_Y`,
`FALLBACK_MAX_THREADS_PER_BLOCK`, `FALLBACK_SHARED_MEMORY_PER_BLOCK`,
`MIN_ROWS_PER_TILE_BIN_FACTOR`, `MIN_ROWS_PER_TILE_THREAD_FACTOR`, and
`strategy_name` from gpu_tiling. Its own three names for the same facts are
now aliases: `PARTIAL_BUDGET_CEILING_BYTES = PARTIAL_BUDGET_BYTES`,
`FALLBACK_CORE_COUNT = FALLBACK_SM_COUNT`,
`MAX_RESIDENT_BLOCKS_PER_CORE = TARGET_BLOCKS_PER_SM`. The mirror block and
its "collapse at integration" note are gone; test_apple_gpu_policy's
mirror test now asserts the re-export path (kept, docstring updated).

`apple_gpu_policy.derive_block_threads(profile, n_rows)` is renamed
`shape_block_threads` and written over `gpu_tiling.clamp_block_threads`
(same arithmetic: min of target, rows, device max, rounded down to a warp,
floored at one warp; the test that pins 256/64/64/192 still passes), so
`gpu_tiling.derive_block_threads(caps)` is the only function of that name.
Callers updated: apple_histogram_policy._shape_block_threads and the test.

Also (923c812): `resident_blocks_per_core` existed in apple_gpu_policy
(over `n_bins`) and apple_histogram_policy (over the compiled kernel
footprint) with the same fit/cap/floor body. The body is now
`apple_gpu_policy.resident_blocks_for_bytes`; the bins form keeps its name
and message; the kernel form is `apple_histogram_policy.resident_blocks_for_kernel`
(one internal caller, updated).

The layering claim "apple_gpu_policy imports nothing" is now "imports only
gpu_tiling". gpu_tiling imports `max.gpu.host` for `query_device_caps`,
which nothing in apple_gpu_policy calls; model.mojo already imports the
whole GPU stack, and test_gpu_tiling.mojo runs in the CPU `test` task, so
this adds no dependency the package did not already have.

### C. gpu_amd_policy.mojo vs gpu_cuda_policy.mojo

Traced first. Role: per-vendor backend specialization policy (occupancy from
`MAX_THREADS/BLOCKS_PER_MULTIPROCESSOR`, reported-warp granularity, static
shared-memory ceilings, transaction-size diagnostics). gpu_tiling and
query_device_caps do NOT cover it: DeviceCaps carries three attributes and
a fixed residency target. So this is an unreached but coherent feature, not
a superseded one, and it was not deleted. It was merged: the two files
differed only in constants plugged into the same fifteen functions.

`src/mojotrees/gpu_vendor_policy.mojo` (git mv from gpu_cuda_policy, so
history follows) keeps the backend-neutral section verbatim and replaces
both vendor sections with one, driven by a `VendorTraits` struct
(`cuda_traits()`, `hip_traits()`, `traits_for(api)`): `subgroup_width`,
`require_subgroup_width_plausible/known`, `block_width_granularity`,
`subgroup_matches_granularity` (was AMD's `wavefront_matches_granularity`),
`static_shared_ceiling`, `require_shared_memory_supported`,
`require_in_order_queue`, `concurrent_queues_available`,
`require_concurrent_queues`, `unified_memory_inferable`, `allocation_plan`,
`partial_budget_for`, `pack_alignment_bytes`, `native_wide_load_bytes`,
`coalescing_bytes`, `packed_body_transactions`, and the entry points
`vendor_contract`, `vendor_specialization`, `vendor_histogram_capabilities`,
`derive_vendor_plan`, `vendor_strategy_inputs`, `vendor_preferred_strategy`,
`require_vendor_launchable`, `describe_vendor`. Functions identical on both
vendors take no traits. Deleted outright: the `plan_packed_window`
passthrough (call `gpu_histogram_specializations.plan_packed_window_for`).
Renamed: neutral `describe_plan` is `describe_backend_plan` so
apple_histogram_policy's is the only `describe_plan`.
`src/mojotrees/gpu_amd_policy.mojo` is removed.

`tests/parallel/test_gpu_vendor_policy.mojo` (new, host arithmetic only) is
the first test to reach this code: traits, unreported device equals the
portable baseline on both vendors, admissibility (CUDA 32; HIP 64 and 32;
both refuse outsiders and pass unreported), granularity refinement,
transaction counts, no strategy preference. Passes. It is registered in
pixi.toml's `test` chain; note that edit was swept into a peer's commit
bceec09 ("Add grow_policy") before I committed it, so it is in history under
that message.

Docs C0 owns that name the old files (K2 does not edit docs):
`docs/GPU_BACKEND_SPECIALIZATIONS.md` lines 4, 5, 22, 69, 353, 359 and
`docs/INTEGRATION_INVENTORY.md` lines 63, 64, 81, 82. The audit's
"named in docs; the file is not there" findings for the two old paths will
clear when those are repointed at `gpu_vendor_policy.mojo`. The audit's
CLASSIFICATION table has no row for gpu_vendor_policy; it shows as an
orphan reached only from tests, which is the same honest status the twins
had, and C0 may want an EXPERIMENTAL row saying so.

### D. HISTOGRAM_BYTES_PER_CELL

Definition site: `apple_cpu_policy.mojo` (the CPU histogram policy leaf,
imported by histogram.mojo and parallel.mojo; imports only std).
`histogram_cache_policy.mojo` imports it. Cycle-free: histogram_cache_policy
already imported parallel, which imports apple_cpu_policy.

### Not touched

`unified_memory_policy.mojo` (connect_05), `python/mojotrees/device_selection.py`
(K7, wave 2), `src/mojotrees/__init__.mojo`, docs, README.
`describe_policy` in apple_cpu_policy vs apple_gpu_policy remains: two
one-line describers over different profile structs, outside K2's list.

### Export changes for C0 (`src/mojotrees/__init__.mojo`)

None required. The `from .device import (...)` block there still resolves
every name it lists. Nothing new needs exporting; gpu_vendor_policy is
policy-internal like the twins were.
