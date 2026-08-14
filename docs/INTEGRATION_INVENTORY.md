# Integration inventory

Snapshot: 2026-08-14, tree at commit `860b1cf`.
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
`src/mojoboost/*.mojo` from four shipping roots:

| Root | Label |
|---|---|
| `src/mojoboost/__init__.mojo` | `mojo-api` |
| `bindings/_mojoboost.mojo` | `bindings` |
| `capi/mojoboost_capi.mojo` | `c-abi` |
| `cli/mojoboost_cli.mojo` | `cli` |

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
| `alternate_boosting` | PENDING | connect_17 | DART and random-forest dispatch. Nothing imports it, so neither mode has a route; `boosting.mojo` still owns the only round loop |
| `backend` | EXPERIMENTAL | connect_01 | A one-function dispatch shim kept as the reference `tests/test_backend_equivalence.mojo` compares against. Test-only by design |
| `boosting_dart` | PENDING | connect_17 | Reached only from `alternate_boosting`, itself unreachable |
| `boosting_rf` | PENDING | connect_17 | Reached only from `alternate_boosting`, itself unreachable |
| `cegb` | PENDING | unassigned | LightGBM's four `cegb_*` controls as a gain adjustment. No grower charges a split, and no parameter turns it on |
| `gpu_backend_policy` | PENDING | connect_20 | Reached only from `gpu_portability`, itself unreachable |
| `gpu_bin_packing` | PENDING | connect_02 | Reached only from `gpu_binned_layout`, itself unreachable |
| `gpu_binned_layout` | PENDING | connect_02 | Packed-bin layout planner; `train_gpu` never asks for a plan |
| `gpu_categorical` | PENDING | connect_10 | GPU category statistics; the GPU trainer refuses categoricals |
| `gpu_levelwise` | PENDING | connect_02 | Level-wise GPU growth; no trainer offers a level-wise mode |
| `gpu_multiclass_batch` | PENDING | connect_04 | Class-batched GPU rounds; multiclass GPU training is per class |
| `gpu_portability` | PENDING | connect_20 | Portable specialization points. Its own test imports `gpu_tiling` and `histogram_gpu` rather than this module, so nothing reaches it |
| `gpu_sparse` | PENDING | connect_10 | Reached only from `gpu_categorical`, itself unreachable |
| `gpu_sparse_layout` | PENDING | connect_10 | Reached only from `gpu_sparse`, itself unreachable |
| `histogram_cache_policy` | PENDING | connect_04 | Reached only from `hybrid_leaf_scheduler`, itself unreachable |
| `hybrid_leaf_scheduler` | PENDING | connect_04 | CPU/GPU per-leaf placement; no trainer consults it |
| `levelwise_policy` | PENDING | connect_02 | Reached only from `gpu_levelwise`, itself unreachable |
| `lgbm_model_io` | PENDING | connect_16 | LightGBM text model reader and writer; no entry point offers it. Reached from `tests/parallel/test_lgbm_model_io.mojo` only |

Two shapes recur and are worth naming, because they change what a fix
costs:

- **Chains.** `gpu_bin_packing` is unreachable only because
  `gpu_binned_layout` is, and `gpu_sparse_layout` only because `gpu_sparse`
  is. One connecting edge at the head of a chain reaches all of it, so the
  count of orphans overstates the count of decisions.
- **Test-only modules.** `backend` and `lgbm_model_io` are imported by
  their own suites and by nothing else. Their tests pass, which is why the
  parity contract can say `focused-tested: yes` and `integrated: no` in the
  same row without contradicting itself.

## Binding modules the extension does not register

`bindings/` holds five auxiliary modules beside `_mojoboost.mojo`.
`_mojoboost.mojo` builds the only `PythonModuleBuilder` in the repository
and imports none of them, so nothing they define is callable from Python.

| File | What it defines | What stays blocked |
|---|---|---|
| `bindings/binding_support.mojo` | fifteen marshalling helpers (`py_dict`, `f64_buffer`, `csc_from_params`, and the rest) | nothing directly; it is the substrate the other four are written against |
| `bindings/dataset_bindings.mojo` | `dataset_metadata`, `dataset_feature_names`, `dataset_categorical_features`, `dataset_field`, `dataset_bin_upper_bounds`, `dataset_missing_bins`, and three more | `Dataset` accessors that today read cached Python state instead of asking the native dataset |
| `bindings/distributed_bindings.mojo` | `distributed_capability`, `distributed_check_machine_list`, `distributed_status_message`, `transport_status_message` | distributed training has no Python route at all |
| `bindings/inspection_bindings.mojo` | `dump_model`, `dump_model_multiclass`, `split_values`, `dump_raw_scores`, `dump_leaf_index`, their multiclass twins, `objective_code`, `dump_model_json`, `model_file_kind`, `model_format_versions` | `mojoboost.inspection` rebuilds the dump by parsing `Booster.model_to_string()` |
| `bindings/objective_bindings.mojo` | `registry_objectives`, `registry_objective_aliases`, `registry_metrics`, `registry_metric_aliases`, `registry_vocabulary`, `objective_code_of_name`, `metric_code_of_name`, and three more | `python/mojoboost/_eval.py` reads mirror tables instead of the native registry |

Registering these is one edit in one file, and it is the highest leverage
connecting change available: it converts three Python fallbacks into thin
formatters over native answers.

Note what is *not* in that list. `split_gains` appears nowhere in
`bindings/`, so the gain hook `python/mojoboost/inspection.py` probes for
has no implementation on either side of the seam, and `decide_device` has
no wrapper in any binding module either. Those two are missing
implementations rather than unregistered ones.

## Native names Python reaches for that no binding registers

Every one of these is a *degraded path*, not a failure: Python probes with
`getattr(_mojoboost, name, None)`, gets `None`, and takes a documented
slower route. The fallback is the honest way to ship an unfinished seam.
It is also the mechanism by which the same question ends up with two
answers, so each row is a disconnection to close rather than a design to
keep.

| Native name | Python caller | What happens without it |
|---|---|---|
| `decide_device` | `python/mojoboost/device_selection.py` | the report runs in its `"narrow"` contract: the backend is still the native answer through `resolve_device`, but the blocking reasons, warnings, memory estimate, policy version, and evidence identifier do not cross |
| `dump_model`, `dump_model_multiclass` | `python/mojoboost/inspection.py` | the dump is rebuilt by parsing `Booster.model_to_string()` |
| `split_gains`, `split_gains_multiclass` | `python/mojoboost/inspection.py` | every dumped node carries `split_gain: None` and `has_split_gain: False`, because gains are recorded during growth and never serialized |
| `split_values`, `split_values_multiclass` | `python/mojoboost/inspection.py` | derived from the parsed model text |
| `dump_leaf_index`, `dump_leaf_index_multiclass` | `python/mojoboost/inspection.py` | derived from the parsed model text |
| `dump_raw_scores`, `dump_raw_scores_multiclass` | `python/mojoboost/inspection.py` | derived from the parsed model text |
| `objective_code` | `python/mojoboost/inspection.py`, `python/mojoboost/device_selection.py` | the objective is resolved from a Python table instead of from the native registry |
| `registry_metrics`, `registry_metric_aliases`, `registry_objectives`, `registry_objective_aliases` | `python/mojoboost/_eval.py` | the metric and objective vocabulary comes from mirror dicts in Python rather than from `src/mojoboost/objective_registry.mojo` |

## Policy that exists twice

The cases where one question has two implementations. In each, the native
one is meant to win, and the Python one exists because the seam is not
finished. Listed here so that a reader of either side finds the other.

| Question | Native, authoritative | Python, in force today |
|---|---|---|
| Which backend runs this job | `src/mojoboost/device_policy.mojo` | `device_selection.py` formats; the estimators bypass it entirely and call `_mojoboost.resolve_device` from `_Base._resolve_device` |
| What an objective or metric is called | `src/mojoboost/objective_registry.mojo` | mirror tables in `python/mojoboost/_eval.py` |
| What a model dump contains | `src/mojoboost/inspection.mojo`, `src/mojoboost/model_dump.mojo` | `python/mojoboost/inspection.py` parses the model text |
| How class weights become row weights | `src/mojoboost/class_weight.mojo` | `_Base._class_weight_rows` in `python/mojoboost/__init__.py` computes them in Python. This one is not a fallback: no binding is probed, and the Mojo module has no caller anywhere in `src/` |

`class_weight` is the sharpest of the four, and the README claim that there
is "one weighting mechanism rather than two" describes the intent rather
than the tree: the Mojo module is publicly reachable from
`src/mojoboost/__init__.mojo` and reached by nothing, while the Python
estimators use their own arithmetic.

There is also a name collision worth knowing about:
`derive_block_threads` is defined twice, in `src/mojoboost/gpu_tiling.mojo`
(taking `DeviceCaps`) and in `src/mojoboost/apple_gpu_policy.mojo` (taking
a `GpuProfile` and a row count). Both are live. `gpu_tiling`'s is the
policy in force for every default launch; `apple_gpu_policy`'s is consulted
only by `apple_histogram_policy`, which only departs from the baseline when
`MOJOBOOST_GPU_HIST_SPECIALIZATION` asks it to.

## Reachable, but with no default effect

Modules that are imported by a shipping path and whose output changes
nothing unless an environment variable asks for it. These are integrated in
the sense `docs/CAPABILITY_LEVELS.md` defines, and it would still be false
to describe them as behavior a user gets.

| Module | Reached from | Default outcome | What turns it on |
|---|---|---|---|
| `apple_histogram_policy` | `histogram_gpu` | `SPEC_LEVEL_BASELINE`, which is `derive_tiling` verbatim | `MOJOBOOST_GPU_HIST_SPECIALIZATION` set to `shape`, `packed`, or `batched` |
| `apple_gpu_policy` | `device_policy`, `apple_histogram_policy` | supplies the device profile and the memory estimate; its tuning derivations are consulted only through the line above | as above |
| `gpu_split_search` | `train_gpu` | the host scan, because Float32 device gains can flip near-tie decisions | `MOJOBOOST_GPU_SPLIT_STRATEGY=device` |
| `unified_memory_policy` | `device_policy`, `histogram_gpu` | one live route; the others it scores are not implemented in any trainer | `MOJOBOOST_GPU_TRANSFER` |
| `device_policy` crossover table | `device`, and through it every `fit` | empty, so `auto` resolves to the CPU on every machine and every workload | `MOJOBOOST_AUTO_MIN_CELLS` |

## Python package modules

| Module | In `mojoboost.__all__` | Notes |
|---|---|---|
| `basic` | `Booster`, `Dataset`, `train` | the only submodule `docs/COMPATIBILITY_POLICY.md` allows importing by path |
| `callback` | six callback names | re-exported at the top level as LightGBM does |
| `cv` | `cv`, `CVBooster` | orchestration over `Dataset` and `Booster`, so a fold model is what `train()` would have built |
| `inspection` | `dump_model`, `trees_to_dataframe`, `trees_to_records`, `get_split_value_histogram` | resolved lazily on first attribute access |
| `device_selection` | `explain_device_choice` | resolved lazily; the rest of the module is not public |
| `dask` | nothing | no transport ships, so every `fit` raises `DistributedNotAvailable` |
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
