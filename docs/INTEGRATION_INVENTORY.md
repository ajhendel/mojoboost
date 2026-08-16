# Integration inventory

Snapshot: 2026-08-14, tree at commit `29d76e4`. The import closure was
re-derived at `63aad82` and re-checked at `9c1e771` and `29d76e4`; it is
unchanged across all three, so every table below is a reading of that
closure.
Checked by: `python3 tools/audit_integration.py`

What is written in this repository but not reachable from any entry point,
and what Python asks the native layer for that the native layer does not
answer. It is the evidence behind every `no` in the `integrated` and
`publicly reachable` columns of `docs/LIGHTGBM_PARITY.md`, and it is the
list a connecting change works through.

**This file is a snapshot, and the tool is the authority.** Lanes land work
continuously, and a module that was an orphan an hour ago may not be one
now. Run `python3 tools/audit_integration.py` for today's answer; it reads
the tree, compares it against the tables below, and prints a corrected
table when they disagree. Neither it nor `tools/connectivity_audit.py`
imports the package or builds anything.

## How a module gets counted

`tools/connectivity_audit.py` computes the import closure of
`src/mojotrees/*.mojo` from four shipping roots:

| Root | Label |
|---|---|
| `src/mojotrees/__init__.mojo` | `mojo-api` |
| `bindings/_mojotrees.mojo` | `bindings` |
| `capi/mojotrees_capi.mojo` | `c-abi` |
| `cli/mojotrees_cli.mojo` | `cli` |

A module no root reaches is an **orphan**. `tests/` and `bench/` are
followed too, but only to annotate: a module reached from a test and from
nothing else is exercised, not shipped, and that is a materially different
state from dead code.

Reachability is the weakest of the three things a capability needs. A
module can be imported and never called, and an import that nothing uses is
the cheapest possible fake connection. `connectivity_audit.py` reports
those separately, and the levels in `docs/CAPABILITY_LEVELS.md` keep
`implemented`, `integrated`, and `publicly reachable` apart for exactly
this reason.

## Orphan native modules

No entry point reaches these. `Kind` and `Owner` are read from the
`CLASSIFICATION` table in `tools/connectivity_audit.py`, which is the one
place a judgment about an orphan is recorded; this table is a rendering of
it, not a second opinion.

| Module | Kind | Owner | Why it is not reached |
|---|---|---|---|
| `backend` | EXPERIMENTAL | connect_01 | A one-function dispatch shim kept as the reference the CPU/GPU equivalence test compares against. Test-only by design |
| `gpu_vendor_policy` | EXPERIMENTAL | consolidation_K2 | CUDA and HIP occupancy policy, merged from the gpu_cuda_policy / gpu_amd_policy twins (f23bd1b). Reached only from its test until a discrete-GPU trainer consults it; that is the same status the twins had. handoffs/migration_20_device_policy.md |

Three shapes recur and are worth naming, because they change what a fix
costs:

- **Chains.** One connecting edge at the head of a chain reaches all of
  it, so the count of orphans overstates the count of decisions.
  `gpu_categorical` -> `gpu_sparse` -> `gpu_sparse_layout` was the last such
  chain: `train_gpu_sparse.mojo` (the sparse GPU trainer, reached from
  `model_sparse.fit_csc` on `device='gpu'` and from the package root)
  imports the head and the whole chain is now shipped. (`gpu_amd_policy`
  and `gpu_cuda_policy` were one until the consolidation round merged them
  into `gpu_vendor_policy`, f23bd1b; `sequence` and `external_memory` were
  one until the integration round exported both and gave `Dataset` a chunk
  binding.)
- **Test-only modules.** `backend` and `gpu_vendor_policy` are imported by
  their own suites and by nothing else. Their tests pass, which is why the
  parity contract can say `focused-tested: yes` and `integrated: no` in the
  same row without contradicting itself. (`lgbm_model_io` was one until
  `bindings/lgbm_bindings.mojo` bound its four file-level entry points and
  `mojotrees.lgbm_model_io` became a lazy submodule; its status text still
  calls the converter experimental, and that is a claim about validation,
  not reach.)
- **Named but not imported.** `linear_tree` and `cegb` have their parameter
  names parsed in `src/mojotrees/params.mojo`, so a user can set them and
  nothing happens. A parameter that parses is not a capability that runs,
  and this is the failure mode the `integrated` column exists to catch.

Between commit `860b1cf` and this snapshot a connecting lane reached seven
modules that were orphans in the previous revision: `gpu_binned_layout` and
`gpu_bin_packing` (now imported by `src/mojotrees/histogram_gpu.mojo` and
re-exported from `src/mojotrees/__init__.mojo`), `gpu_portability` and
`gpu_backend_policy`, `gpu_multiclass_batch`, and `hybrid_leaf_scheduler`
with `histogram_cache_policy` (now imported by
`src/mojotrees/gpu_runtime.mojo`). Reached is not called, so each was read
for a call site rather than promoted as a group, and they did not land in
the same place. `gpu_portability` is called: `histogram_gpu` opens its
builder through `require_bins_supported`,
`require_device_can_host_kernels`, and `require_histogram_launchable`, which
replaced hand-rolled checks, and `gpu_backend_policy` answers underneath it.
`gpu_multiclass_batch` is called, behind `MOJOTREES_GPU_CLASS_BATCH`, and
its parity row moved from `deferred` to `partial` for that reason. The other
four are reached and not exercised: `gpu_binned_layout` contributes one
overflow guard and no packed plan, `gpu_bin_packing` nothing at all,
`hybrid_leaf_scheduler` a report that moves no histogram, and
`histogram_cache_policy` only what that report reads. Their rows stay at
`deferred` with the reach written into the evidence.
`CLASSIFICATION` in `tools/connectivity_audit.py` still carries
entries for all seven; they are now judgments about modules that are no
longer findings, and removing them is a patch for that file's owner.

## Binding modules the extension does not register

`_mojotrees.mojo` builds the only `PythonModuleBuilder` in the repository,
so a module it does not import defines nothing a user can call.

| File | What it defines | What stays blocked |
|---|---|---|

None. The table is kept empty so `tools/audit_integration.py` keeps
checking it: a binding module the extension stops importing lands here as
a GAP. `bindings/binding_support.mojo` (marshalling helpers: `py_dict`,
`f64_buffer`, `csc_from_params`, and the rest) used to be listed as the
one module the extension did not need; `_mojotrees.mojo` now imports it
too, since its private copies of the buffer readers (`_f64_list`,
`_int_list`, `_csc`, `_csr`) were retired in favor of the shared ones,
whose bulk copy is the single implementation.

This section was the largest disconnection in the previous revision, and it
is closed. At commit `860b1cf` the extension imported none of the auxiliary
modules; at `63aad82` it imports `basic_bindings`, `dataset_bindings`,
`distributed_bindings`, `inspection_bindings`, and `objective_bindings`, and
its `def_function` table registers `dump_model`, `dump_model_multiclass`,
`split_values`, `dump_leaf_index`, `dump_raw_scores`, `objective_code`,
`decide_device`, and the six `registry_*` entries. What was three Python
fallbacks is now three thin formatters over native answers, with the
fallbacks kept behind `getattr` for an extension built before the change.

One name did not survive that pass. `split_gains` still appears nowhere in
`bindings/`, so the gain hook `python/mojotrees/inspection.py` probes for
has no implementation on either side of the seam; it is a missing
implementation rather than an unregistered one, and the row below is what is
left of this section.

## Native names Python reaches for that no binding registers

Every one of these is a *degraded path*, not a failure: Python probes with
`getattr(_mojotrees, name, None)`, gets `None`, and takes a documented
slower route. The fallback is the honest way to ship an unfinished seam.
It is also the mechanism by which the same question ends up with two
answers, so each row is a disconnection to close rather than a design to
keep.

| Native name | Python caller | What happens without it |
|---|---|---|
| `split_gains`, `split_gains_multiclass` | `python/mojotrees/inspection.py` | every dumped node carries `split_gain: None` and `has_split_gain: False`, because gains are recorded during growth and never serialized. `_native_split_gains` probes for the hook, finds nothing on either side of the seam, and the dump reports its source as `model_to_string` rather than `model_to_string+split_gains` |

The rest of this table is gone, and the reason is worth recording rather
than deleting. At `860b1cf` it held eight rows: `decide_device`,
`dump_model`, `split_values`, `dump_leaf_index`, `dump_raw_scores`,
`objective_code`, and the four `registry_*` names. All of them are now in
the `def_function` table of `bindings/_mojotrees.mojo`. The `getattr` probes
in `python/mojotrees/inspection.py`, `device_selection.py`, and `_eval.py`
remain, which is correct: they are the compatibility path for an extension
built from an older tree, and `_eval.registry_source()` and
`report.contract` still say which implementation answered. A probe that
finds its hook is not a fallback in force.

## Policy that exists twice

The cases where one question has two implementations. In each, the native
one is meant to win, and the Python one exists because the seam is not
finished. Listed here so that a reader of either side finds the other.

| Question | Native, authoritative | Python, in force today |
|---|---|---|
| Which backend runs this job | `src/mojotrees/device_policy.mojo` | `explain_device_choice` reaches the full native report now that `decide_device` is bound, but `_Base._resolve_device` still calls `_mojotrees.resolve_device` directly, so what a `fit` actually does is decided by the narrower of the two entry points |
| What an objective or metric is called | `src/mojotrees/objective_registry.mojo` | the mirror tables in `python/mojotrees/_eval.py` are still present and still the code path when the hooks are absent; `registry_source()` returns `"native"` or `"compat"` and is the only honest way to say which one answered |
| What a model dump contains | `src/mojotrees/inspection.mojo`, `src/mojotrees/model_dump.mojo` | `python/mojotrees/inspection.py` keeps its `model_to_string` parser as the fallback, and uses it unconditionally for split gains, which no native hook supplies |
| How class weights become row weights | `src/mojotrees/class_weight.mojo` | `_Base._class_weight_rows` in `python/mojotrees/__init__.py` computes them in Python. This one is not a fallback: no binding is probed, and the Mojo module has no caller anywhere in `src/` |

`class_weight` is the sharpest of the four, and the only one where no seam
is even open. `src/mojotrees/class_weight.mojo` is re-exported from
`src/mojotrees/__init__.mojo`, so it counts as reachable, and nothing in
`src/` imports it: a Mojo caller can use it, a `MojoTreesClassifier` never
does, and the two implementations can disagree without any test noticing.
`README.md` says which one runs.

There is also a name collision worth knowing about:
`derive_block_threads` is defined twice, in `src/mojotrees/gpu_tiling.mojo`
(taking `DeviceCaps`) and in `src/mojotrees/apple_gpu_policy.mojo` (taking
a `GpuProfile` and a row count). Both are live. `gpu_tiling`'s is the
policy in force for every default launch; `apple_gpu_policy`'s is consulted
only by `apple_histogram_policy`, which only departs from the baseline when
`MOJOTREES_GPU_HIST_SPECIALIZATION` asks it to.

## Reachable, but with no default effect

Modules that are imported by a shipping path and whose output changes
nothing unless an environment variable asks for it. These are integrated in
the sense `docs/CAPABILITY_LEVELS.md` defines, and it would still be false
to describe them as behavior a user gets.

| Module | Reached from | Default outcome | What turns it on |
|---|---|---|---|
| `apple_histogram_policy` | `histogram_gpu` | `SPEC_LEVEL_BASELINE`, which is `derive_tiling` verbatim | `MOJOTREES_GPU_HIST_SPECIALIZATION` set to `shape`, `packed`, or `batched` |
| `apple_gpu_policy` | `device_policy`, `apple_histogram_policy` | supplies the device profile and the memory estimate; its tuning derivations are consulted only through the line above | as above |
| `gpu_split_search` | `train_gpu` | the host scan, because Float32 device gains can flip near-tie decisions and the measured difference (a few percent either way, `docs/LIGHTGBM_PARITY.md`) does not pay for that | `MOJOTREES_GPU_SPLIT_STRATEGY=device` |
| `unified_memory_policy` | `device_policy`, `histogram_gpu` | one live route; the others it scores are not implemented in any trainer | `MOJOTREES_GPU_TRANSFER` |
| `device_policy` crossover table | `device`, and through it every `fit` | one rule, scoped to Metal on an M4 for squared error at 1,000,000 x 50 and above, which cannot fire through `resolve_device` because `DeviceCapabilities.detect()` opens no device and a hardware-scoped rule cannot match an unidentified profile. `auto` therefore still resolves to the CPU everywhere, with a warning that says which half of "does not cover" applied | `MOJOTREES_AUTO_MIN_CELLS`, or a caller passing a profile read from an open `DeviceContext` to `decide_device_report_reported` |
| `gpu_multiclass_batch` | `train_gpu`, `histogram_gpu` | a sequential schedule, so multiclass GPU training stays one tree per class per round and `_train_multiclass_gpu_batched` is not entered. Now measured rather than merely unmeasured: batching seven classes gives 15.45s against the sequential schedule's 15.30s at 465,000 x 54, which is inside the spread and therefore indistinguishable, so the default is right and there is nothing here to turn on | `MOJOTREES_GPU_CLASS_BATCH` above one, or a caller passing its own batch |
| `hybrid_leaf_scheduler` | `gpu_runtime` | a report and nothing else. `GpuSession.note_hybrid` records what was asked for and why it was declined; no histogram changes device | nothing. `MOJOTREES_HYBRID_LEAVES` and `MOJOTREES_HYBRID_TRACE` change what is reported, not where a histogram is built |

Two of those rows are held off by default because they were measured and did
not pay, not because the connecting work is outstanding, and the difference
matters to anyone reading this table for a list of available wins.

`gpu_split_search` is the clearer case. Its device scan is finished, tested,
and slower. On an M4 it trains about 24% behind the host scan at 50000 rows
by 100 features over three repeats per arm, and at 250000 by 100 the two are
indistinguishable, so promoting it would buy split decisions that can differ
from the CPU's in exchange for nothing measurable. Two theory-driven attempts
at its remaining cost have both come back inside noise, and the module's own
docstring now records that the scan kernel's shape is not where that cost
lives, so the next attempt on it should start from a profile.

The `batched` specialization behind `apple_histogram_policy` is the subtler
one. It batches histogram builds across leaves, which is worth doing under
level-wise growth where a whole level is built at once. The shipping grower
is leaf-wise and now derives a sibling by subtraction, so a split builds one
histogram and computes the other, and there is at most one build per split
left to batch. That specialization is waiting on a level-batched
histogram phase under `grow_policy=depthwise` (`growth_policy.mojo` orders
the level; nothing batches its builds yet), not on a switch, and reading it
as a free win off this table is a mistake made at least once.

## Python package modules

| Module | In `mojotrees.__all__` | Notes |
|---|---|---|
| `basic` | `Booster`, `Dataset`, `train` | the only submodule `docs/COMPATIBILITY_POLICY.md` allows importing by path |
| `callback` | six callback names | re-exported at the top level as LightGBM does |
| `cv` | `cv`, `CVBooster` | orchestration over `Dataset` and `Booster`, so a fold model is what `train()` would have built |
| `inspection` | `dump_model`, `trees_to_dataframe`, `trees_to_records`, `get_split_value_histogram` | resolved lazily on first attribute access |
| `device_selection` | `explain_device_choice` | resolved lazily; the rest of the module is not public |
| `dask` | nothing | no transport ships, so every `fit` raises `DistributedNotAvailable` |
| `lgbm_model_io` | nothing | resolved lazily; LightGBM model-file import and export through the native converter, experimental by its own `interop_status()` |
| `diagnostics` | nothing | formats phase durations something else measured; nothing imports it and no suite covers it |
| `_public_api_plan` | nothing | a plan expressed as data. Its own docstring says nothing in the package imports it, and importing it would be the bug |
| `_compat` | nothing | holds the pre-import interpreter guard, and nothing calls it, so on an interpreter below the extension floor the process aborts on a missing symbol instead of raising this module's message |

## What this inventory does not tell you

It is a statement about imports and symbol tables. It cannot tell you that
a connected path is correct, that a test is meaningful, or that a
benchmark was run. `docs/LIGHTGBM_PARITY.md` carries the claims,
`docs/CAPABILITY_LEVELS.md` defines what each claim requires, and
`docs/GPU_VALIDATION.md` carries the record of what hardware has actually
executed.
