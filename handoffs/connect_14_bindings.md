# Connect 14 handoff: bindings for the native capabilities except GPU prediction

Owned and edited by this lane, and nothing else:

- `bindings/binding_support.mojo` (new)
- `bindings/objective_bindings.mojo` (new)
- `bindings/dataset_bindings.mojo` (new)
- `bindings/inspection_bindings.mojo` (new)
- `bindings/distributed_bindings.mojo` (new)
- `bindings/basic_bindings.mojo` (new)
- `handoffs/connect_14_bindings.md` (this file)

Nothing outside that list was touched. `bindings/_mojoboost.mojo` belongs
to task 06 and is untouched here; every line it needs is in section 6.1,
ready to paste.

Nothing was committed by this lane. Note for a reviewer: another lane ran
a sweeping commit (860b1cf) partway through this work, so
`binding_support.mojo`, `objective_bindings.mojo`, `inspection_bindings.mojo`
and an earlier `dataset_bindings.mojo` are already inside that commit's
tree rather than sitting as untracked files. The later edits to
`dataset_bindings.mojo` and the two newer modules are in the working tree.

**Nothing here has been compiled.** No Mojo, no pixi, no build, no test.
Every claim below is from reading the sources named in it. Section 9 lists
the checks that would establish the rest, all of them UNRUN.

---

## 1. The state this lane found

Nine native capabilities were implemented and had no way to Python. Two of
them had Python code already written *against a binding that did not
exist*, which is the sharpest evidence that the binding was the missing
piece and not the design:

| Capability | Native home | Python already calling it |
| --- | --- | --- |
| Structured model dump | `model_dump.mojo`, `inspection.mojo` | `inspection.py` `_hook(booster, "dump_model")`, `"split_values"`, `"dump_leaf_index"`, `"dump_raw_scores"`, `getattr(_mojoboost, "objective_code")` |
| Device explanation | `device_policy.mojo`, `device.decide_device_report` | `device_selection.py` `getattr(ext, "decide_device")`, falling back to a `"narrow"` contract |
| Objective/metric registry | `objective_registry.mojo` | `_eval.py`, which documents the five `_mojoboost.objective_registry_*` calls that would delete its tables |
| Dataset beyond dense | `trainset.mojo` `from_csc`, `from_reference`, `subset` | none yet; `connect_12_dataset_cv.md` §6.3 has the Python half |
| Dataset prepared data | `trainset.mojo` accessors, `BinMapper`, `efb.feature_bin_count` | `basic.py` answers from its own Python copies |
| Extra tree parameters | `tree_parameters_extra.mojo` | none |
| EFB configuration | `efb.mojo` `check_bundling_supported` / `check_bundling_params` | none |
| Distributed runtime | `collective.mojo`, `distributed.mojo`, `distributed_transport.mojo` | `dask.py`, which needs a real refusal |
| Model format metadata | `serialize.model_file_kind`, `model_dump` versions | `basic.py` `_load_path` guesses the kind by catching an exception |
| Startup diagnostics | `initialization.mojo` | `diagnostics.py` `parse_trace`, with a hand-kept phase table |

GPU prediction and validation are task 06's and are deliberately absent
from every module here.

## 2. Call path, before and after

Before, for the dump (the largest case):

```
inspection.dump_model(model)
  -> _native_dump  -> _hook(booster, "dump_model") -> None      (unbound)
  -> _dump_from_text
       -> Booster.model_to_string()      the v3 save format, as text
       -> parse_model_string             a second implementation, in Python,
                                         of tree layout, bin edges, category
                                         decoding, and row routing
```

After, once section 6.1 is applied:

```
inspection.dump_model(model)
  -> _native_dump -> _mojoboost.dump_model(handle, names, n_names)
       -> inspection_bindings.dump_model
            -> model_dump.build_dump(model, names)      <- the one dump
            -> _py_dump                                  <- shape only
  -> _schema_from_native      nests the flat node tables, names the objective
```

The text parser stays in place and unreached; its deletion is section 8 of
`handoffs/migration_19_model_inspection.md` and belongs to the lane that
owns `inspection.py`, not to this one.

The same shape holds for the other capabilities: a Python facade that
already exists starts reaching a native implementation that already
exists, and no second implementation is created on either side.

## 3. What each module carries

**`binding_support.mojo`.** The boundary helpers, and nothing else. Python
value builders (`py_dict`, `py_int_list`, `py_f64_list`, `py_str_list`,
`py_pair`), buffer readers (`f64_buffer`, `int_buffer`,
`int_buffer_from_f64`, `str_sequence`), one writer into a caller-sized
buffer (`write_f64_buffer`), the two sparse matrix rebuilders
(`csc_from_params`, `csr_from_params`), and three validators (`flag`,
`nonnegative`, `index_within`). Every read refuses a null address, a
negative length, an out-of-range index, and a flag that is neither 0 nor
1. No address ever travels outward.

**`objective_bindings.mojo`.** The five registry snapshots specified in
`migration_21_objective_metric_registry.md` §4 (`registry_objectives`,
`registry_objective_aliases`, `registry_objective_unimplemented`,
`registry_metrics`, `registry_metric_aliases`), plus `registry_vocabulary`
for the scalar codes those records are written in, and three single
lookups (`objective_code_of_name`, `metric_code_of_name`,
`objective_name_status_of`) and `check_objective_param` for callers
holding one name and no snapshot. Records cross as lists, in the field
order each docstring states. An unknown name is absent from the alias list
and never a sentinel code.

**`dataset_bindings.mojo`.** Three constructors that `dataset_create` has
no room for (`dataset_create_csc`, `dataset_create_reference`,
`dataset_subset`) and the reads off a constructed dataset
(`dataset_metadata`, `dataset_feature_names`,
`dataset_categorical_features`, `dataset_field`, `dataset_field_length`,
`dataset_copy_field`, `dataset_feature_num_bin`,
`dataset_bin_upper_bounds`, `dataset_missing_bins`). Every constructor
forwards to the `Dataset` static of the same name; no binning rule, no
validation rule, and no fold policy is written here.

**`inspection_bindings.mojo`.** Waves 1, 2 and 3 of
`migration_19_model_inspection.md` §4: `dump_model(_multiclass)`,
`split_values(_multiclass)`, `dump_raw_scores(_multiclass)`,
`dump_leaf_index(_multiclass)`, `dump_model_json(_multiclass)`,
`objective_code`, and the model format metadata (`model_file_kind`,
`model_format_versions`). Trees cross as flat node tables;
`inspection._nested_node` nests them, as it already expects to.

**`distributed_bindings.mojo`.** `distributed_capability`,
`distributed_check_machine_list`, `distributed_status_message`,
`transport_status_message`. Fail closed: `multi_process` is False with the
reason, because the wire protocol has no socket endpoint to run over. See
section 5 for the one fact this module states rather than reads.

**`basic_bindings.mojo`.** `decide_device_workload` (see 6.1 for which
device entry point to register), `extra_params_check`,
`extra_option_supported`, `forced_splits_check`, `efb_check`,
`efb_defaults`, `startup_phase_contract`, `startup_environment`,
`native_clock_ns`.

## 4. Duplicates fused, and duplicates left alone

**Fused into `binding_support.mojo`.** `_f64_list`, `_int_list`,
`_int_list_from_f64`, `_csc` and `_csr` in `_mojoboost.mojo` are the same
five reads this lane needs. They could not be imported from there without
a cycle (`_mojoboost` imports these modules), so `binding_support` owns
them now and 6.1(f) is the optional patch that retires the copies in
`_mojoboost.mojo`. Until that patch lands there are two copies of five
short buffer reads. They are byte-for-byte equivalent by construction and
neither can drift silently: both are called by the same Python buffers.

**Not duplicated: the split gains hook.** `task14_inspection.md` §2 and
`migration_19` §4 both specify a `split_gains` hook that hands a
text-sourced dump its gains back. Wave 1 makes it unnecessary, because the
native dump carries gains, and `migration_19` says explicitly not to build
both. `inspection.py:1176` still looks for it, harmlessly, inside its
compatibility section. **Do not implement `split_gains`.**

**Not duplicated: device selection.** `decide_device_workload` runs no
policy: it folds two sentinels and forwards to `device.decide_device_report`,
which forwards to the one engine in `device_policy.mojo`. Task 06's
ten-argument version, if it is registered instead, does exactly the same.
Register one, never both (6.1(b)).

**Name collision found and resolved: `objective_code`.** Two earlier plans
both asked for a binding by that name, with different arguments:
`migration_19` §4 wants `objective_code(model_handle)` and `migration_20`
§1b wants `objective_code(objective_name)`. `inspection.py:803` calls the
handle form and `device_selection.py:388` calls the name form, so binding
either one silently disables the other (each fails soft, returning None or
falling back to text). Resolution: `inspection_bindings.objective_code`
takes the handle, matching the existing Python; the name resolver is
`objective_bindings.objective_code_of_name`, and 6.3 is the one-line
`device_selection.py` patch that finds it.

**Answered rather than duplicated: `connect_12_dataset_cv.md` §6.2(e).**
That section asks for `dataset_is_sparse`, `dataset_nnz`, and
`dataset_has_raw` as three entry points. They are three keys of
`dataset_metadata` instead, which the Python `Dataset` reads in one call
along with the other eleven. There is one way to ask, not two.

## 5. The one fact a binding states rather than reads

`distributed_bindings._HAS_WIRE_TRANSPORT = False`.

There is no native predicate for "this build has a wire transport" because
there is nothing yet to predicate on: a socket `ByteEndpoint` is an absent
implementation, not a disabled one (`distributed_transport.mojo` header,
"Not implemented"). The comptime is marked in place as the fact this
module states, and flipping it is explicitly **not** the way to enable
distributed training. The patch that removes it is 6.5.

Everything else in every module is read from native code.

## 6. Cross-lane patch requests

### 6.1 `bindings/_mojoboost.mojo` (owner: task 06) — required

Nothing existing changes. Two additive blocks.

**(a) Imports.** Add above the `mojoboost.*` import block:

```mojo
from basic_bindings import (
    decide_device_workload,
    efb_check,
    efb_defaults,
    extra_option_supported,
    extra_params_check,
    forced_splits_check,
    native_clock_ns,
    startup_environment,
    startup_phase_contract,
)
from dataset_bindings import (
    dataset_bin_upper_bounds,
    dataset_categorical_features,
    dataset_copy_field,
    dataset_create_csc,
    dataset_create_reference,
    dataset_feature_names,
    dataset_feature_num_bin,
    dataset_field,
    dataset_field_length,
    dataset_metadata,
    dataset_missing_bins,
    dataset_subset,
)
from distributed_bindings import (
    distributed_capability,
    distributed_check_machine_list,
    distributed_status_message,
    transport_status_message,
)
from inspection_bindings import (
    dump_leaf_index,
    dump_leaf_index_multiclass,
    dump_model,
    dump_model_json,
    dump_model_json_multiclass,
    dump_model_multiclass,
    dump_raw_scores,
    dump_raw_scores_multiclass,
    model_file_kind,
    model_format_versions,
    objective_code,
    split_values,
    split_values_multiclass,
)
from objective_bindings import (
    check_objective_param,
    metric_code_of_name,
    objective_code_of_name,
    objective_name_status_of,
    registry_metric_aliases,
    registry_metrics,
    registry_objective_aliases,
    registry_objective_unimplemented,
    registry_objectives,
    registry_vocabulary,
)
```

**(b) Registration.** In `PyInit__mojoboost`, after
`m.def_function[resolve_device]("resolve_device")`:

```mojo
        # -- inspection (handoffs/migration_19_model_inspection.md) ----
        m.def_function[dump_model]("dump_model")
        m.def_function[dump_model_multiclass]("dump_model_multiclass")
        m.def_function[split_values]("split_values")
        m.def_function[split_values_multiclass]("split_values_multiclass")
        m.def_function[dump_raw_scores]("dump_raw_scores")
        m.def_function[dump_raw_scores_multiclass](
            "dump_raw_scores_multiclass"
        )
        m.def_function[dump_leaf_index]("dump_leaf_index")
        m.def_function[dump_leaf_index_multiclass](
            "dump_leaf_index_multiclass"
        )
        m.def_function[dump_model_json]("dump_model_json")
        m.def_function[dump_model_json_multiclass](
            "dump_model_json_multiclass"
        )
        m.def_function[objective_code]("objective_code")
        m.def_function[model_file_kind]("model_file_kind")
        m.def_function[model_format_versions]("model_format_versions")
        # -- objective and metric registry -----------------------------
        m.def_function[registry_objectives]("registry_objectives")
        m.def_function[registry_objective_aliases](
            "registry_objective_aliases"
        )
        m.def_function[registry_objective_unimplemented](
            "registry_objective_unimplemented"
        )
        m.def_function[registry_metrics]("registry_metrics")
        m.def_function[registry_metric_aliases]("registry_metric_aliases")
        m.def_function[registry_vocabulary]("registry_vocabulary")
        m.def_function[objective_code_of_name]("objective_code_of_name")
        m.def_function[metric_code_of_name]("metric_code_of_name")
        m.def_function[objective_name_status_of]("objective_name_status")
        m.def_function[check_objective_param]("check_objective_param")
        # -- datasets ---------------------------------------------------
        m.def_function[dataset_create_csc]("dataset_create_csc")
        m.def_function[dataset_create_reference](
            "dataset_create_reference"
        )
        m.def_function[dataset_subset]("dataset_subset")
        m.def_function[dataset_metadata]("dataset_metadata")
        m.def_function[dataset_feature_names]("dataset_feature_names")
        m.def_function[dataset_categorical_features](
            "dataset_categorical_features"
        )
        m.def_function[dataset_field]("dataset_field")
        m.def_function[dataset_field_length]("dataset_field_length")
        m.def_function[dataset_copy_field]("dataset_copy_field")
        m.def_function[dataset_feature_num_bin]("dataset_feature_num_bin")
        m.def_function[dataset_bin_upper_bounds](
            "dataset_bin_upper_bounds"
        )
        m.def_function[dataset_missing_bins]("dataset_missing_bins")
        # -- run configuration ------------------------------------------
        m.def_function[extra_params_check]("extra_params_check")
        m.def_function[extra_option_supported]("extra_option_supported")
        m.def_function[forced_splits_check]("forced_splits_check")
        m.def_function[efb_check]("efb_check")
        m.def_function[efb_defaults]("efb_defaults")
        # -- distributed -------------------------------------------------
        m.def_function[distributed_capability]("distributed_capability")
        m.def_function[distributed_check_machine_list](
            "distributed_check_machine_list"
        )
        m.def_function[distributed_status_message](
            "distributed_status_message"
        )
        m.def_function[transport_status_message](
            "transport_status_message"
        )
        # -- startup diagnostics -----------------------------------------
        m.def_function[startup_phase_contract]("startup_phase_contract")
        m.def_function[startup_environment]("startup_environment")
        m.def_function[native_clock_ns]("native_clock_ns")
```

**(c) The device entry point: register exactly one.**
`handoffs/connect_05_device_policy.md` §5.1 asks for a ten-argument
`decide_device` written directly in `_mojoboost.mojo`. Prefer that: it is
the shape `_FullNativePolicy.decide` in `device_selection.py` already
sends, so it needs no Python patch. Then **do not** register
`decide_device_workload`.

If ten arguments turn out to exceed what `def_function` accepts (eight is
proven in tree by `predict_range`; ten is not, and
`handoffs/performance_15_startup.md` claims a cap of six, which the
existing eight-argument entry points contradict), register this instead:

```mojo
        m.def_function[decide_device_workload]("decide_device")
```

and apply 6.3(b) in the same commit, because the Python caller's argument
shape changes with it.

**(d) `dataset_create` gains `keep_raw`** (`connect_12_dataset_cv.md`
§6.2(a)): read `keep_raw` from `params` and pass it as the twelfth
positional argument to `Dataset`. Without it no constructed dataset can be
subset. `dataset_create_csc` and `dataset_create_reference` already read
that key, so the three constructors agree once this lands.

**(e) `add_type` is unchanged.** Every function above takes or returns
`Model`, `MulticlassModel`, or `Dataset`, all three already registered.
`dataset_create_csc`, `dataset_create_reference`, and `dataset_subset`
return a `Dataset` through `PythonObject(alloc=...)`, the same ownership
transfer `dataset_create` uses.

**(f) Optional, retires the duplicate reads.** Delete `_f64_list`,
`_int_list`, `_int_list_from_f64`, `_csc`, and `_csr` from
`_mojoboost.mojo` and add:

```mojo
from binding_support import (
    csc_from_params as _csc,
    csr_from_params as _csr,
    f64_buffer as _f64_list,
    int_buffer as _int_list,
    int_buffer_from_f64 as _int_list_from_f64,
)
```

The signatures are identical except that the two sparse builders take the
params mapping directly, which is what `_csc(params)` and `_csr(params)`
already do. `_sparse_shape` becomes unused and goes with them.

### 6.2 `bindings/build.sh` (owner: whoever touches it first; task 06 or 18) — required

The build compiles one file with `-I src`, so the sibling modules are not
on the import path:

```sh
pixi run mojo build --emit shared-lib -I src -I bindings \
    bindings/_mojoboost.mojo -o python/mojoboost/_mojoboost.so
```

If Mojo already puts the input file's own directory on the path, this is a
no-op and harmless. It has not been verified either way here, because
verifying it means running a build. Any packaging script that reproduces
this command needs the same flag.

### 6.3 `python/mojoboost/device_selection.py` (owner: task 05)

**(a) One line, always worth applying.** `_code_for_objective_name`
currently looks only for `objective_code`, which is now the *model handle*
accessor. Prefer the name resolver:

```python
    resolve = getattr(ext, "objective_code_of_name", None) or getattr(
        ext, "objective_code", None
    )
```

Without it a name-only caller silently reports its objective as
undeclared, which skips the objective gate. With it, `explain_device_choice(
X, y, objective="lambdarank")` gets `BLOCK_RANKING_OBJECTIVE` as intended.

**(b) Only if 6.1(c) took the fallback branch.** Replace the ten positional
arguments in `_FullNativePolicy.decide` with the mapping:

```python
        text = self._decide(
            requested,
            {
                "n_rows": int(workload.n_rows),
                "n_features": int(workload.n_features),
                "n_outputs": int(workload.n_outputs),
                "n_bins": _BINS_UNSPECIFIED
                if workload.max_bin is None
                else int(workload.max_bin),
                "objective": _OBJECTIVE_UNSPECIFIED
                if workload.objective_code is None
                else int(workload.objective_code),
                "sparse": 1 if workload.sparse else 0,
                "categorical": 1 if workload.categorical else 0,
                "has_missing": 1 if workload.has_missing else 0,
                "uses_validation": 1 if workload.has_eval_set else 0,
            },
        )
```

**(c) A latent mismatch worth fixing whichever branch is taken.**
`_BINS_UNSPECIFIED = -1` in Python, and `BINS_UNSPECIFIED = 0` natively.
Both bindings fold *any* negative `n_bins` to the native sentinel, so -1
works, but a caller that sends 0 meaning "undeclared" would have it read
as a declared bin count of zero. Either keep sending -1 (the bindings
handle it) or change the Python constant to 0; do not send both.

### 6.4 `python/mojoboost/_eval.py` and `__init__.py` (owner: task 07)

The five registry snapshots exist now, so `migration_21` §3.7 and §3.8 are
unblocked. Call each **once** at `_eval` import and cache it; not once per
lookup, because `resolve()` runs inside `fit`. Two contract notes:

- Records are **lists**, not tuples (`migration_21` §4 said tuples). A
  consumer indexes them the same way; build tuples on arrival if it wants
  immutability.
- `objective_default_metric` is field 12 of an objective record and is -1
  for the custom objective, the only one with no default.

### 6.5 `src/mojoboost/distributed_transport.mojo` (owner: task 13)

Add the predicate this lane had to state instead of read:

```mojo
def runtime_capability() raises -> String:
    """Whether this build can form a multi-process world, as
    `key=value` lines: `multi_process`, `max_world_size`,
    `protocol_version`, and `reason` (empty when it can).

    True requires a `ByteEndpoint` over a real connection. Nothing in this
    module is one; `MemoryEndpoint` is a fake and is named as one.
    """
```

`distributed_bindings.distributed_capability` then reads it and the
`_HAS_WIRE_TRANSPORT` comptime is deleted in the same commit.

### 6.6 `src/mojoboost/__init__.mojo` (owner: task 01) — only if needed

These modules import submodules that the package `__init__.mojo` does not
re-export: `model_dump`, `inspection`, `objective_registry`,
`device_policy`, `initialization`, `tree_parameters_extra`, `efb`,
`raw_data`, and `distributed_transport`. Direct submodule imports
(`from mojoboost.model_dump import ...`) should resolve under `-I src`
whether or not the package re-exports them, which is why they are written
that way. If the build says otherwise, the fix is to re-export those
modules' names from `__init__.mojo`; `migration_21` §3.1 already lists the
registry's names for exactly this reason.

### 6.7 `python/mojoboost/basic.py` (owner: task 07)

`Booster._load_path` picks its loader by calling `load` and catching the
exception, which cannot tell a wrong-loader error from a corrupt file.
With `model_file_kind` bound:

```python
    def _load_path(self, path):
        kind = str(_mojoboost.model_file_kind(path))
        if kind == "multiclass":
            self._handle = _mojoboost.load_multiclass(path)
            self._n_classes = int(_mojoboost.n_classes(self._handle))
            self._task = _eval.MULTICLASS
        else:
            self._handle = _mojoboost.load(path)
            self._n_classes = 0
```

`connect_12_dataset_cv.md` §6.3 is the rest of this file's work and is
unblocked by 6.1: `dataset_create_csc`, `dataset_create_reference`, and
`dataset_subset` are the three calls it asked for.

## 7. Fallbacks preserved

- **The text-sourced dump stays.** `inspection.py` keeps its compatibility
  section and reaches it whenever a hook is absent, so a build with none
  of section 6.1 applied behaves exactly as it does today.
- **The narrow device contract stays.** `resolve_device` and
  `gpu_available` are untouched, and `_NarrowNativePolicy` still answers
  when no decision entry point is bound.
- **Python's compatibility tables stay** until task 07 deletes them. The
  snapshots are additive; nothing breaks by binding them and not reading
  them.
- **`dataset_create` is untouched** and keeps working for dense input.
  The three new constructors are additive.
- **`distributed_capability` fails closed**, so a caller that starts
  consulting it refuses distributed work rather than starting it.

## 8. Effects on serialization and the public API

- **No serialized format changes.** No module here writes a model, and no
  model state is added. `model_format_versions` reports the two version
  numbers the format already carries; `model_file_kind` reads a header
  that `save_model` already writes.
- **The dump is read-only.** `split_gain` is null for a model with no
  gains, which is the state a model read back from a v1, v2, or v3 file is
  in; the serialization lane has since made v4 carry them, and
  `model_format_versions` reports whichever version the build writes,
  because it reads the constant rather than restating it.
- **The public Python API gains nothing by itself.** Every function here
  is reachable only once task 06 registers it and a Python caller looks
  for it. The names in 6.1(b) are the ones `inspection.py`,
  `device_selection.py`, and `_eval.py` already look for, so the surface
  they open is the one those modules already document.
- **Ownership.** Everything returned is a fresh Python object or a fresh
  handle. No pointer, no device buffer, and no borrowed view crosses
  outward. Addresses travel inward only, and the caller keeps its buffers
  alive for the call, which is the convention the existing bindings state.

## 9. Risks, and what is unverified

1. **Nothing compiles yet.** The single largest risk. Syntax, import
   resolution, `def_function` arity, and `PythonObject` conversions are
   all read, not run.
2. **Cross-module imports in `bindings/`** depend on 6.2. If Mojo does not
   put the input file's directory on the path and `-I bindings` is not
   added, `_mojoboost.mojo` will not resolve `from inspection_bindings
   import ...` and the build fails loudly. It cannot fail quietly.
3. **`def_function` arity above eight is unproven** (section 6.1(c)).
4. **Submodule imports of modules not re-exported by `__init__.mojo`**
   (section 6.6). Same character of failure: loud, at build time.
5. **`Python.import_module("builtins").dict()`** is used for every dict
   returned. If the pinned Mojo exposes a direct constructor, it is a
   mechanical substitution in `binding_support.py_dict` and nowhere else.
6. **Wave 2 rebuilds the dump per call.** Right for the one-row
   conformance checks `inspection.py` makes, wrong for a loop over rows.
   If a caller ever needs per-row routing at scale the fix is a handle
   that owns a built `ModelDump` (`m.add_type` alongside `Model`), not a
   faster rebuild.
7. **`dataset_bin_upper_bounds` reads the `BinMapper` layout directly**
   (`edges[edge_offsets[f] : edge_offsets[f + 1]]`). That layout is
   documented on the struct and is the same one `model_dump._build_features`
   reads. If it ever changes, this is a second reader to update; a
   `bin_upper_bounds(mapper, feature)` helper in `binning.mojo` would
   remove that, and would be the right home for it.
8. **`extra_params_check` reports what a fit would need but changes no
   fit.** The extra parameters still have to reach the growers, which is
   task 09's connection. A caller that gets `needs_grower_support: true`
   and trains anyway gets a model that ignored the parameters; the check
   makes that detectable, not impossible.
9. **`distributed_capability` states one fact** (section 5).
10. **No startup trace exists to report.** `StartupTrace` is deliberately
    not a singleton, so `startup_environment` reports the *request*
    (`MOJOBOOST_STARTUP_TRACE`, `MOJOBOOST_GPU_WARMUP`) and there is no
    honest `startup_report()` to bind. A trace owner has to exist first;
    `handoffs/performance_15_startup.md` §5 describes the owner.

## 10. Focused checks, all UNRUN

In order. Each is the smallest thing that establishes the next.

```sh
# 1. Does the extension build with the new modules on the path?
#    Requires 6.1 and 6.2 to be applied first.
bash bindings/build.sh                                          # UNRUN

# 2. Is every new name actually exported?
python -c "import mojoboost._mojoboost as m; print(sorted(n for n in dir(m) if not n.startswith('_')))"   # UNRUN

# 3. Did the device wiring take? Expects "full".
python -c "from mojoboost.device_selection import native_contract; print(native_contract())"              # UNRUN

# 4. Does the native dump agree with the text-sourced one, key for key?
#    The differential that retires the Python parser.
pixi run pytest python/tests/parallel/test_inspection.py -x -q  # UNRUN

# 5. Does the device wiring hold end to end?
pixi run pytest python/tests/parallel/test_device_selection.py -x -q  # UNRUN

# 6. One focused Mojo probe that the dataset constructors round trip:
#    build from CSC, subset, and check num_data and nnz.
pixi run mojo test tests/test_trainset.mojo                    # UNRUN
```

Check 4 is the one that matters most: it is the only check that can show
the native dump and the Python parser disagreeing, and a disagreement
there is a bug in one of them rather than in the binding.
