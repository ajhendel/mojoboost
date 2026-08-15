# consolidation K6 - bindings boundary

Lane K6 of the consolidation round (CLAUDE_CODE_CONSOLIDATION_PROMPTS.txt).
Files owned: bindings/**. Nothing pushed; explicit-path commits only.

## Commits

- e98d4eb  bindings/_mojotrees.mojo: removed the whole-model `predict`,
  `predict_raw`, `predict_proba` bindings (three `def_function` rows and
  three defs). Verified by building the extension once in a scratch
  worktree at HEAD with this file copied in (the shared tree's
  train_gpu.mojo is mid-edit by connect_04 and does not compile right
  now), then a MojoTreesRegressor fit/predict smoke check. Audit
  "Binding functions no Python module calls": 59 -> 56.

## 1. bindings/binding_support.mojo

The audit finding is a FALSE POSITIVE. `audit_binding_modules` checks only
whether bindings/_mojotrees.mojo imports a sibling directly. binding_support
is imported by all five capability modules the entry point does import
(objective_bindings:35, distributed_bindings:34, dataset_bindings:33,
basic_bindings:32, inspection_bindings:43), and bindings/build.sh passes
`-I bindings`, so every one of its 15 helpers is compiled and reached.
Nothing to wire, nothing to delete.

For C0 (tools/ owner): make the check transitive over `from <sibling>
import` inside bindings/, or drop the finding for a module some imported
sibling imports. binding_support is not a "surface" module (it exports no
Python-visible function); it is the boundary helper library.

Duplicate noted, NOT resolved: bindings/_mojotrees.mojo keeps its own
`_f64_list` / `_int_list` (bulk `unsafe_memcpy`, "invalid buffer" errors)
beside binding_support's `f64_buffer` / `int_buffer` (element loop,
"null buffer address" / "buffer length must not be negative" errors), and
its own `_csc` / `_csr` beside `csc_from_params` / `csr_from_params`.
binding_support's own docstring says `_mojotrees.mojo`'s copy is the one
to retire. Deferred: retiring either copy changes the error text a
Python test could match, and the memcpy form is the one worth keeping, so
this is a two-file edit (adopt memcpy inside binding_support, then point
`_mojotrees.mojo` at it) that needs an extension build; the shared tree
cannot build until connect_04 lands. One commit for the next bindings
owner; no behavior question left open.

## 2. Exported binding functions with no Python caller (59 flagged)

The audit's caller scan is `_mojotrees.<name>` and
`getattr(_mojotrees, "<name>")` only. Python also reaches the extension
through string dispatch: `_eval._REGISTRY_HOOKS`, `inspection._hook(booster,
name)` (appends `_multiclass`), `sklearn._predict_batch(entry, legacy)`,
`_dask_runtime` provider probes, and `device_selection` `getattr(ext, ...)`.
That makes 19 of the 59 false negatives.

| name | disposition |
|---|---|
| registry_metric_aliases, registry_objectives, registry_objective_aliases | REACHED (`_eval._REGISTRY_HOOKS`); audit false negative; keep |
| dump_model_multiclass, split_values, split_values_multiclass, dump_leaf_index, dump_leaf_index_multiclass, dump_raw_scores, dump_raw_scores_multiclass | REACHED (`inspection._hook`); audit false negative; keep |
| predict_leaf, predict_leaf_batch, predict_leaf_multiclass, predict_leaf_multiclass_batch, predict_proba_batch | REACHED (`sklearn._predict_batch` entry/legacy pairs); audit false negative; keep |
| distributed_capability | REACHED (`_dask_runtime.CAPABILITY_ENTRY_POINTS`, provider probe); keep |
| objective_code_of_name | REACHED (`device_selection.py:475`, `getattr(ext, ...)`); keep |
| predict, predict_raw, predict_proba | DELETED e98d4eb: full-range twins of predict_range / predict_proba_range that Python outgrew (docs/CONNECTION_AUDIT.md section 3); no test, no C ABI route |
| dataset_num_data, dataset_num_feature, num_trees, gpu_predict_capability, gpu_validation_open, gpu_validation_open_multiclass, gpu_validation_accumulate, gpu_validation_accumulate_multiclass, gpu_validation_metric, gpu_validation_raw, gpu_validation_reset, gpu_validation_shape | owned-by-lane connect_07 (already classified in the audit); keep |
| efb_check, efb_defaults, extra_option_supported, extra_params_check, forced_splits_check | owned-by-lane connect_22 (already EXPERIMENTAL in the audit); keep |
| dataset_bin_upper_bounds, dataset_copy_field, dataset_feature_num_bin, dataset_field_length, dataset_missing_bins | keep-forward-surface: LightGBM Dataset field readers (`get_field`, `feature_num_bin`, bin edges, missing bins) that basic.py's Dataset does not yet expose |
| check_objective_param, metric_code_of_name, objective_name_status, registry_objective_unimplemented, registry_vocabulary | keep-forward-surface: registry queries beside the four `_REGISTRY_HOOKS` Python does read; `_eval` still carries a Python mirror it prefers when a hook is missing |
| distributed_check_machine_list, distributed_status_message, transport_status_message | keep-forward-surface: distributed_bindings companions of the reached distributed_capability; a real transport's Python side reads them |
| dump_model_json, dump_model_json_multiclass, model_file_kind, model_format_versions | keep-forward-surface (inspection_bindings): JSON dump for the C ABI and CLI, loader kind, schema versions; `model_file_kind` is documented as distinct from `file_kind` (refuses dataset files) |
| file_kind | keep-forward-surface: header sniff answering objective/multiclass/dataset; sibling of model_file_kind by design |
| gpu_validation_metric_matches_host | keep-forward-surface: companion of the connect_07 gpu_validation surface |
| native_clock_ns, startup_environment, startup_phase_contract | keep-forward-surface (basic_bindings): startup/report contract the Python side does not read yet |

Nothing else met the deletion bar: no other export is an exact duplicate
of one Python calls.

## 3. Native names Python reaches for that do not exist

`mojo` is the audit reading `_mojotrees.mojo` (the file name) as an
attribute in a docstring; false positive, C0's regex. Not in my files.

Also seen (not flagged by the audit, string dispatch): `split_gains` /
`split_gains_multiclass` (inspection.py:1181, graceful None fallback; no
binding exists, and no wrapper in bindings/inspection_bindings.mojo either,
so adding one is connect_11's plumbing, not this lane's), and the four
`lgbm_*` names in python/mojotrees/lgbm_model_io.py, which is an
UNTRACKED file of a live lane (connect_16); its own error text names the
missing binding contract. Recorded, not touched.

## CLASSIFICATION block for tools/connectivity_audit.py (coordinator applies)

```python
    # -- binding table entries with no Python caller (consolidation K6) ------
    "binding_support": (
        EXPERIMENTAL,
        "consolidation_K6",
        "Boundary helper library imported by all five capability modules the "
        "entry point imports (build.sh passes -I bindings); it exports no "
        "Python-visible function. The direct-import check is the false positive.",
    ),
    "registry_metric_aliases": (EXPERIMENTAL, "consolidation_K6", "Reached by _eval._REGISTRY_HOOKS through getattr by string; caller scan misses it."),
    "registry_objectives": (EXPERIMENTAL, "consolidation_K6", "Reached by _eval._REGISTRY_HOOKS through getattr by string; caller scan misses it."),
    "registry_objective_aliases": (EXPERIMENTAL, "consolidation_K6", "Reached by _eval._REGISTRY_HOOKS through getattr by string; caller scan misses it."),
    "dump_model_multiclass": (EXPERIMENTAL, "consolidation_K6", "Reached by inspection._hook, which appends _multiclass to a string name."),
    "split_values": (EXPERIMENTAL, "consolidation_K6", "Reached by inspection._hook by string name."),
    "split_values_multiclass": (EXPERIMENTAL, "consolidation_K6", "Reached by inspection._hook, which appends _multiclass to a string name."),
    "dump_leaf_index": (EXPERIMENTAL, "consolidation_K6", "Reached by inspection._hook by string name."),
    "dump_leaf_index_multiclass": (EXPERIMENTAL, "consolidation_K6", "Reached by inspection._hook, which appends _multiclass to a string name."),
    "dump_raw_scores": (EXPERIMENTAL, "consolidation_K6", "Reached by inspection._hook by string name."),
    "dump_raw_scores_multiclass": (EXPERIMENTAL, "consolidation_K6", "Reached by inspection._hook, which appends _multiclass to a string name."),
    "predict_leaf": (EXPERIMENTAL, "consolidation_K6", "Legacy half of a sklearn._predict_batch (entry, legacy) pair, reached by string."),
    "predict_leaf_multiclass": (EXPERIMENTAL, "consolidation_K6", "Legacy half of a sklearn._predict_batch (entry, legacy) pair, reached by string."),
    "predict_proba_batch": (EXPERIMENTAL, "consolidation_K6", "Entry half of a sklearn._predict_batch pair, reached by string."),
    "distributed_capability": (EXPERIMENTAL, "consolidation_K6", "Probed by _dask_runtime.CAPABILITY_ENTRY_POINTS through the provider by string."),
    "objective_code_of_name": (EXPERIMENTAL, "consolidation_K6", "Reached by device_selection.py through getattr(ext, ...) by string."),
    "dataset_bin_upper_bounds": (PENDING, "consolidation_K6", "LightGBM Dataset field reader; basic.py's Dataset does not expose it yet."),
    "dataset_copy_field": (PENDING, "consolidation_K6", "LightGBM Dataset.get_field; basic.py's Dataset does not expose it yet."),
    "dataset_feature_num_bin": (PENDING, "consolidation_K6", "LightGBM Dataset.feature_num_bin; basic.py's Dataset does not expose it yet."),
    "dataset_field_length": (PENDING, "consolidation_K6", "Length query paired with dataset_copy_field; connects with it."),
    "dataset_missing_bins": (PENDING, "consolidation_K6", "LightGBM Dataset field reader; basic.py's Dataset does not expose it yet."),
    "check_objective_param": (PENDING, "consolidation_K6", "Registry query beside the four hooks _eval reads; _eval prefers its Python mirror when a hook is missing."),
    "metric_code_of_name": (PENDING, "consolidation_K6", "Registry query; Python resolves metric names through _eval's mirror today."),
    "objective_name_status": (PENDING, "consolidation_K6", "Registry query; Python resolves objective names through _eval's mirror today."),
    "registry_objective_unimplemented": (PENDING, "consolidation_K6", "Registry query; _fit_args restates the unimplemented-objective notes today."),
    "registry_vocabulary": (PENDING, "consolidation_K6", "Registry vocabularies for a reader of the records; no Python reader yet."),
    "distributed_check_machine_list": (PENDING, "consolidation_K6", "Companion of distributed_capability; a real transport's Python side reads it."),
    "distributed_status_message": (PENDING, "consolidation_K6", "Companion of distributed_capability; a real transport's Python side reads it."),
    "transport_status_message": (PENDING, "consolidation_K6", "Companion of distributed_capability; a real transport's Python side reads it."),
    "dump_model_json": (PENDING, "consolidation_K6", "JSON dump for the C ABI and CLI; Python builds its dict from dump_model instead."),
    "dump_model_json_multiclass": (PENDING, "consolidation_K6", "JSON dump for the C ABI and CLI; Python builds its dict from dump_model instead."),
    "model_file_kind": (PENDING, "consolidation_K6", "Loader kind for a saved model, distinct from file_kind by design; Booster._load_path still tries a loader and reads the exception."),
    "model_format_versions": (PENDING, "consolidation_K6", "Schema versions a consumer branches on; no Python reader yet."),
    "file_kind": (PENDING, "consolidation_K6", "Header sniff answering objective / multiclass / dataset; no Python reader yet."),
    "gpu_validation_metric_matches_host": (PENDING, "consolidation_K6", "Companion of the connect_07 gpu_validation surface, reached only through a session Python never opens."),
    "native_clock_ns": (PENDING, "consolidation_K6", "Startup/report contract of basic_bindings; no Python reader yet."),
    "startup_environment": (PENDING, "consolidation_K6", "Startup/report contract of basic_bindings; no Python reader yet."),
    "startup_phase_contract": (PENDING, "consolidation_K6", "Startup/report contract of basic_bindings; no Python reader yet."),
```

Better than 19 EXPERIMENTAL rows for the false negatives: extend
`python_native_reads` in tools/connectivity_audit.py to also collect
double-quoted string literals that match an exported binding name (and the
`name + "_multiclass"` form inspection uses); then those rows disappear on
their own. C0's call.

## Deferred

- `_f64_list`/`_int_list`/`_csc`/`_csr` in _mojotrees.mojo vs
  binding_support (section 1): one build-gated commit, after connect_04.
- `split_gains` binding (section 3): connect_11's plumbing.
- `lgbm_*` bindings (section 3): connect_16, untracked file.
