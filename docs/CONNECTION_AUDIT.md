# Connection audit

What in this repository is reachable from something a user can call, and
what is not.

mojotrees was built by many parallel lanes. A lane can finish a capability
completely - module written, tests passing, handoff filed - without the
capability ever becoming reachable, because the edit that would reach it
lives in a file the lane does not own. This document is the map of that
state at one moment, and `tools/connectivity_audit.py` is how you get the
map again at any other moment.

**Snapshot: commit `860b1cf`, 2026-08-14, mid-afternoon.** The tree was
moving while this was written: nine native modules that had no importer at
the top of the hour had one by the end of it, and `bindings/` went from one
Mojo file to seven. Treat every count below as a reading, not a constant.
Re-run the script before acting on any single row.

**Second pass, 2026-08-14, evening.** Four of the findings below were then
fixed, three of them in this lane, and each is marked *Closed* where it was
first stated rather than deleted, because a finding and its fix are more
useful together than either alone. Section 12 lists the four in one place.
Everything not marked *Closed* is still the mid-afternoon reading.

```
python3 tools/connectivity_audit.py                 # full report
python3 tools/connectivity_audit.py --section binding-modules
python3 tools/connectivity_audit.py --json          # machine readable
```

The script was authored in the same lane as this document and, per that
lane's terms, **has not been run**. Every number below was gathered by hand
with `rg` and `sed` against the working tree. The script exists so the next
person does not have to, and its first run may disagree with a row here;
where it does, the script is the one looking at today's tree.

---

## 1. The entry points

Five, and only five, things can start a call into mojotrees:

| Root | What it is | Reaches |
| --- | --- | --- |
| `python/mojotrees/__init__.py` | the public Python package | native code only through the extension |
| `bindings/_mojotrees.mojo` | the CPython extension | 63 `def_function` registrations |
| `src/mojotrees/__init__.mojo` | the public Mojo API | 32 re-exported modules |
| `capi/mojotrees_capi.mojo` | the C ABI | 14 `@export`ed functions |
| `cli/mojotrees_cli.mojo` | the `mojotrees` command | `train`, `predict`, `info`, `help`, `version` |

Everything else in the repository is reachable, or is not, through one of
these. A module reached only from `tests/` or `bench/` is exercised, not
shipped, and this document says so explicitly wherever it applies.

---

## 2. The finding that matters most: five binding modules that do not exist
at runtime

`bindings/` now holds seven Mojo files:

| File | Public functions | Lines |
| --- | ---: | ---: |
| `_mojotrees.mojo` | 64 registrations | 1,846 |
| `basic_bindings.mojo` | 9 | 357 |
| `binding_support.mojo` | 12 (helpers) | 184 |
| `dataset_bindings.mojo` | 9 | 208 |
| `distributed_bindings.mojo` | 4 | 152 |
| `inspection_bindings.mojo` | 13 | 397 |
| `objective_bindings.mojo` | 10 | 256 |

`_mojotrees.mojo` imports **none** of the other six. The
`PythonModuleBuilder` block inside it is the entire Python-facing surface of
the extension, so the 45 functions in the five `*_bindings.mojo` files are
attributes of nothing. `bindings/build.sh` compiles only

```sh
pixi run mojo build --emit shared-lib -I src \
    bindings/_mojotrees.mojo -o python/mojotrees/_mojotrees.so
```

which does not put `bindings/` on the include path either, so
`objective_bindings.mojo`'s `from binding_support import py_dict` has no way
to resolve even if the entry point did import it.

This is not a theoretical gap. Python is already reaching for these exact
names and already falling back:

| Python asks for | Where | Implemented in | Falls back to |
| --- | --- | --- | --- |
| `_mojotrees.dump_model` | `inspection.py` | `inspection_bindings.mojo` | re-parsing the saved text model in Python |
| `_mojotrees.objective_code` | `inspection.py:696` | `inspection_bindings.mojo` | `None`, then a Python-side guess |
| `_mojotrees.registry_metrics` | `_eval.py` | `objective_bindings.mojo` | a hand-maintained code table |
| `_mojotrees.decide_device` | `device_selection.py` | `basic_bindings.mojo` | the `"narrow"` mode, using `resolve_device` |

All four are now written. All four are unreachable for the same single
reason. `handoffs/connect_05_device_policy.md` §5.1 independently asked lane
06 for the `decide_device` registration and called it "the one that unblocks
the rest"; this audit agrees and extends it to the other four modules.

`basic_bindings.mojo` carries more than `decide_device`: `extra_params_check`,
`extra_option_supported`, `forced_splits_check`, `efb_check`, `efb_defaults`,
`startup_phase_contract`, `startup_environment`, `native_clock_ns`. That is
the Python route to the extra tree parameters of section 5 and to the startup
tracing of `initialization.mojo`, and it is behind the same one edit.

`python/mojotrees/inspection.py` carries a `# DELETION POINT` banner naming
the precise set - `dump_model`, `dump_model_multiclass`, `split_values*`,
`dump_raw_scores*`, `dump_leaf_index*` - and roughly 580 lines of Python
below it exist only until those names appear. They now exist in Mojo. They
are still not in the table.

One edit in one file closes most of this, and that file belongs to lane 06.
See the patch queue in `handoffs/connect_22_audit.md`.

> **Closed, by lane 06, the same evening.** `bindings/_mojotrees.mojo` now
> imports all five capability modules (`dataset_bindings`, `basic_bindings`,
> `distributed_bindings`, `inspection_bindings`, `objective_bindings`) and
> the registration table is at 116 entries, up from the 63 counted above.
> All four names in the table above are registered:
> `decide_device_workload` under the name `decide_device`, plus `dump_model`,
> `objective_code`, and `registry_metrics`. The `# DELETION POINT` block in
> `python/mojotrees/inspection.py` is now a fallback with a live native path
> in front of it rather than the only implementation; deleting it is lane
> 07's call and is not this audit's to make.

---

## 3. The mirror: a GPU surface with no Python caller

The extension exports functions no module under `python/mojotrees/` mentions:

```
gpu_predict_capability
gpu_validation_open           gpu_validation_open_multiclass
gpu_validation_accumulate     gpu_validation_accumulate_multiclass
gpu_validation_metric         gpu_validation_raw
gpu_validation_reset          gpu_validation_shape
predict_batch                 predict_proba_batch
predict_leaf_batch            predict_leaf_multiclass_batch
dataset_num_data              dataset_num_feature
num_trees                     predict                predict_proba
predict_raw
```

The first thirteen are a complete GPU prediction and GPU validation surface,
registered and unreachable. `MojoTreesRegressor.predict` calls
`predict_range`; nothing calls `gpu_predict_capability`, so a fitted model
never learns whether GPU prediction covers it and never takes that path. The
last six are older entries that Python outgrew - `predict` and
`predict_proba` were superseded by their `_range` forms without being
removed - which is a smaller problem of the same kind: the table says the
surface is larger than the reachable API.

Exporting is not connecting, in either direction.

---

## 4. Native modules no entry point reaches

Nine, in seven clusters. A cluster is one unreachable module plus whatever
is reachable only through it: fixing the head of the cluster fixes all of it,
which is why they are grouped.

| Cluster | Also unreachable through it | Lane | Reading |
| --- | --- | --- | --- |
| `alternate_boosting` | `boosting_dart`, `boosting_rf` | 17 | DART and random forest have no route; `boosting.mojo` still owns the only round loop |
| `gpu_binned_layout` | `gpu_bin_packing` | 02 | `train_gpu` never asks for a packed-bin plan |
| `gpu_levelwise` | `levelwise_policy` | 02 | resolved Aug 15 2026: `levelwise_policy` became `growth_policy` (the leaf pick every grower calls under both `grow_policy` values); `gpu_levelwise` was removed unused (handoffs/consolidation_K8.md) |
| `gpu_multiclass_batch` | (`gpu_output_planes` is also reached via `apple_histogram_policy`) | 04 | multiclass GPU training is still per-class |
| `hybrid_leaf_scheduler` | `histogram_cache_policy` | 04 | no trainer consults per-leaf CPU/GPU placement |
| `gpu_categorical` | `gpu_sparse`, `gpu_sparse_layout` | 10 | the GPU trainer refuses categoricals, so its category kernels are unused |
| `gpu_portability` | `gpu_backend_policy` | 20 | **Closed.** `tests/test_gpu_portability.mojo` imported `gpu_tiling` and `histogram_gpu`, not this module - the test named for it did not exercise it |

> **`gpu_portability` closed in the second pass.**
> `src/mojotrees/histogram_gpu.mojo` imports it, holds a `BackendContract`
> field, and calls `require_bins_supported` once before binning and
> `require_histogram_launchable` at each of the three launch sites. The
> contract is the thing being connected, not the import: the launch bound
> the kernel was written against is now checked against the device that will
> run it, rather than assumed. `gpu_backend_policy` is reached through it.
> `tests/test_gpu_portability.mojo` imports the module it is named for.
>
> The other six clusters are unchanged, so six clusters, not seven.

Two more are reachable from tests only, and one of them is fine that way:

- **`backend.mojo`** - a one-function dispatch shim reached only from
  `tests/test_backend_equivalence.mojo`, which is in the `test` pixi task.
  Test-only by design: it is the reference the equivalence test compares
  against. **EXPERIMENTAL**, leave it.
- **`lgbm_model_io.mojo`** - 1,400 lines of LightGBM text model reader and
  writer, reached only from `tests/parallel/test_lgbm_model_io.mojo`. No
  entry point offers LightGBM interop to anyone. **PENDING**, lane 16.

### What moved during the snapshot

At the top of the same afternoon, `initialization`, `inspection`,
`objective_registry`, `raw_data`, `unified_memory_policy`,
`distributed_transport`, `gpu_fused_round`, `gpu_leaf_batching`, and
`apple_histogram_policy` all had zero importers. All nine have one now. That
is the pace the tree is changing at, and it is the argument for the script
over this list.

It is also a warning. An import is the weakest evidence of connection there
is: `inspection.mojo` is now imported by `inspection_bindings.mojo`, which
the extension does not import, which means `inspection.mojo` gained an edge
and gained no reachability from Python. The script's `orphans` section uses
only `_mojotrees.mojo` as the bindings root for exactly this reason.

---

## 5. Two parameter surfaces, one of them unreachable from Python

`src/mojotrees/params.mojo` parses LightGBM's text parameter spec into a
`TrainConfig`. It is the only production writer of `ExtraTreeParams`:

```
min_gain_to_split   max_delta_step   path_smooth
extra_trees         extra_seed       monotone_penalty
monotone_method     penalties        forced (splits)
```

`parse_params` is imported by `capi/mojotrees_capi.mojo` and
`cli/mojotrees_cli.mojo`. It is **not** imported by
`bindings/_mojotrees.mojo`, which builds its argument list from explicit
positional parameters instead. So the C ABI and the CLI can set nine tree
parameters that the Python estimators cannot express at all - they are not
in `_Base.__init__`, and there is no `**params` passthrough that would reach
`parse_params`.

This is a real capability split along an interface boundary, not a naming
mismatch. Whichever way it is resolved - route the bindings through
`parse_params`, or add the parameters to the estimator signature and thread
them - the resolution is a cross-lane edit, because lane 09 owns
`params.mojo`, lane 06 owns the extension, and lane 07 owns the estimators.

> **Closed in the second pass**, the second way: the parameters are on the
> estimator and threaded through, and `params.mojo` stays the parameter-string
> parser it was rather than becoming the boundary's parser too.
>
> `_Base.__init__` takes `min_gain_to_split` (alias `min_split_gain`),
> `max_delta_step`, `path_smooth`, `extra_trees`, `extra_seed`,
> `monotone_penalty` (alias `monotone_constraints_penalty`),
> `monotone_constraints_method`, `feature_contri`, `cegb_tradeoff`,
> `cegb_penalty_split`, and `forced_splits`. `_params()` sends each of them
> on every fit, inactive defaults included, because the native parser
> subscripts the mapping rather than testing for a key.
>
> `bindings/basic_bindings.mojo` grew `extra_params_from_mapping`, and
> `_parse_params` in `bindings/_mojotrees.mojo` calls it and folds the result
> into `TreeParams.extra`. It is the same function `extra_params_check`
> validates with, which is the point: one parser over one mapping, so what a
> caller can ask about and what a fit is trained with cannot come apart.
>
> No range check was added in Python. `ExtraTreeParams.check` runs inside
> `tree.grow_tree` and is what the C ABI and the CLI already reach through
> `params.mojo`, so there is one authority for what these values may be
> rather than a Python copy of it to drift from. Python validates the one
> thing native cannot: the `feature_contri` buffer's length and dtype, where
> `n_features` is known.
>
> Two of the eleven still refuse rather than train, natively and by name, and
> the refusal is the same one the CLI gets: `forced_splits` needs its raw
> thresholds mapped onto a fitted binning, which `binning.map_forced_splits`
> can do and no entry point calls; and `cegb_penalty_feature_coupled` needs a
> per-model feature-use ledger no trainer keeps, so the key is sent as 0 and
> there is no estimator parameter that can set it.

### EFB is a guard with no engine behind it

`src/mojotrees/efb.mojo` is reachable, through `params.mojo` and
`dataset_bindings.mojo`. What is reachable is `check_bundling_supported`
and `check_bundling_params`, which raise when a caller asks for bundling.
The machinery - `fit_bundles`, `bundle_csc`, `unbundle_histogram`,
`FeatureBundling` - has no production caller. `boosting.mojo` and `tree.mojo`
mention a bundle only to say that it "defaults to inactive".

So `enable_bundle=true` is correctly refused rather than silently ignored,
which is the right failure. But the module reads as connected in an import
graph and is not connected in any behavioral sense. This is the pattern the
audit exists to separate, and EFB is its clearest instance.

> **Closed in the second pass, in two halves, and the first half was already
> done when the fix pass started.** The engine had gained production callers
> in the native tree between the snapshot and the fix: `prepare_bundling` in
> `boosting.mojo` fits a plan once per training call for each of the four
> dense trainers, and `prepare_bundling_csc` in `boosting_sparse.mojo` does
> the same for `fit_csc` and `fit_multiclass_csc`, calling `fit_bundles` and
> `bundle_csc`. So the sentence above was true when written and had stopped
> being true by the evening, which is the argument for the script over the
> document once more.
>
> The half that was still open was the Python route, and it was the same
> shape as the parameter split above: `_parse_params` built its
> `BoosterParams` with three arguments and never the fourth, so
> `BoosterParams.bundling` took its disabled default on every fit that came
> through Python. `enable_bundle` was reachable from `params.mojo` - the CLI
> and the C ABI - and from nowhere else.
>
> `bindings/basic_bindings.mojo` grew `efb_settings_from_mapping` on the
> `extra_params_from_mapping` pattern, `efb_check` was rewritten to use it,
> and `_parse_params` passes the result as `BoosterParams.bundling`.
> `_Base.__init__` takes `enable_bundle` and the six knobs it governs.
>
> Which trainers may honor the switch is decided at the boundary, because
> that is the last place that knows which trainer runs next. `_parse_params`
> takes two new arguments for it: `cpu`, which the dense entry points set
> from the device they resolved, and `unbundled`, which names the entry point
> when its trainer applies no plan at all. An entry point that names itself
> gets `efb.check_bundling_honored`, which refuses an active switch by name;
> the rest get `efb.check_bundling_supported`, which is the check
> `params.mojo` makes for a parameter string. So `enable_bundle=True` trains
> on a dense CPU fit, on a sparse fit, and on continued training, and raises
> with the trainer's name on `device="gpu"`, on a custom objective, on a
> custom metric or an eval set, and on the ranker. Nothing silently drops it.

---

## 6. Python package

`python/mojotrees/` holds fourteen modules. Two are not reached from
`__init__.py` by any import chain:

- **`_public_api_plan.py`** - a proposal expressed as data. Its own docstring
  says "nothing in the package imports it" and means it: importing it would
  be the bug. **EXPERIMENTAL**, leave it, and leave the docstring, because
  it is what stops the next audit from re-finding it.

- **`_compat.py`** - **PENDING, and it is a live defect.** The module holds
  `unsupported_interpreter()` and `import_extension()`, the checks that must
  run *before* `mojotrees._mojotrees` is imported. Its docstring explains
  why, and the explanation is measured: on CPython 3.9 the Mojo runtime
  resolves `Py_NewRef` out of libpython at load time, fails to find it, and
  **aborts the process**. An abort cannot be caught, so a `try` around the
  import cannot help; the only place a check can do any good is in front of
  the import. Nothing calls it. `python/mojotrees/__init__.py:260` does
  `from . import _arrays, _eval, _mojotrees, ...` with no guard ahead of it.
  A user on an unsupported interpreter gets `ABORT: symbol not found:
  Py_NewRef` instead of the message this module was written to print.

  **Closed in the second pass.** `python/mojotrees/__init__.py` now does
  `from . import _compat` and binds the extension with
  `_mojotrees = _compat.import_extension()`, which is the only import of it
  in the package, so the guard runs in front of the load it guards. The
  ordering is the whole fix and it is fragile by nature: any later edit that
  imports `_mojotrees` directly, above that line, restores the abort. The
  comment above the call says so, which is the cheapest defense there is.

A stale copy of the package from an earlier build lived under
`python/build/` when this was written (a `lib.macosx-*` directory holding
its own `mojotrees/__init__.py`); it has since been removed. It is not an orphan so much
as debris, and it will confuse any grep run over `python/` that does not
exclude it. The script skips `build/` by name.

---

## 7. Serialization

**This section was wrong when it was drafted and is kept as an example of
why the check belongs in a script.** Two hours before this snapshot,
`Tree.split_gain` was filled in by training (`tree.mojo:327`,
`distributed.mojo:670`), read by `importance.mojo` and `inspection.mojo`, and
written by neither `save_model` nor `_write_trees` - so `importance_type=
"gain"` returned zeros after any save/load round trip, silently. The reader
even documented it: "Split gains are a training artifact and are not
serialized."

At `dc21f03` the format is at `v4`, `_write_trees` emits per-node gains
behind a `_has_split_gains` guard, and `_read_trees` restores them. The
finding is closed.

What remains is the general check, which is what the script keeps: for each
field of `Tree`, `Booster`, and `BinMapper`, does `serialize.mojo` mention
it at all? A field that training fills and serialization drops is state a
saved model loses, and prediction is the wrong place to notice, because
prediction usually does not need it. Importance, inspection, and
contributions do.

The section-keyword check is the cheaper half: every literal `save` writes
(`objective`, `learning_rate`, `base_score`, `base_scores`, `multiclass`,
`mapper`, `categorical`, `monotone`, `trees`, `tree`, `cat`) against every
literal `load` looks for. A keyword on one side only is a format the two
halves disagree about.

---

## 8. Duplicate registries and policies

Two that were duplicates and are not any more, recorded because the shape
recurs:

- **Device vocabulary.** `device.mojo` defines `parse_device`, `device_name`,
  `gpu_available`, `resolve_device`, and `gpu_supports`, and
  `device_policy.mojo` defines all of them too. It is not a duplicate: the
  first is a documented facade that forwards every call to the second, and
  says so in its module docstring. Duplicate *names* with one implementation
  behind them is the resolved form of this problem, and the script's
  `duplicate-registries` section will keep flagging it - correctly, as
  something to look at, and the answer each time is "read the docstring".

- **Objective tables.** `objective_registry.mojo` is now imported by
  `objective.mojo`, `custom_metric.mojo`, `gpu_objectives_native.mojo`,
  `lgbm_model_io.mojo`, and `objective_bindings.mojo`. `params.mojo` still
  carries its own `objective_from_name` and `objective_display_name`, and
  `python/mojotrees/_eval.py` still carries a metric-code table its own
  comment describes as "the mirror of the table in bindings/_mojotrees.mojo".
  Three tables, one of them in another language, and the Python one is
  mirrored by hand because `registry_metrics` is unreachable (section 2).

The general rule the script encodes: a public name defined in two native
modules is either a facade with a docstring saying so, or a second table
that will drift. There is no third case.

---

## 9. Referenced paths that do not exist

Of 155 repository paths named by `README.md` and `docs/*.md`, two are not
there:

| Path | Named by | Reading |
| --- | --- | --- |
| a startup benchmark under `bench/` | `docs/STARTUP_LATENCY.md:265` | the doc says it "does not exist" in the same sentence, so this is honest prose, not a broken claim; the sentence no longer spells a path so the script does not read it as one |
| `tools/api_snapshot.py` | `docs/COMPATIBILITY_POLICY.md:770` | the doc calls itself the specification for a script that has not been written; `tests/parallel/api_snapshot_manifest.json` exists and nothing reads it |

Parity evidence is in better shape than that ratio suggests, and the reason
is that `tools/check_parity.py` already gates it. **This audit does not
re-check the parity contract** - not the level definitions, not the evidence
columns, not the claim schema. `check_parity.py` owns
`docs/LIGHTGBM_PARITY.md` and `docs/CAPABILITY_LEVELS.md`, it runs as its own
CI job, and a second parity checker would be precisely the duplication this
document spends section 8 complaining about. `connectivity_audit.py` asks one
question about parity - do the files named as evidence exist - and defers
everything about what the evidence means.

---

## 10. How to read a finding

Three kinds, and the distinction is the whole point of the exercise:

- **`DEAD`** - nothing reaches it and nothing is expected to. Remove it.
- **`EXPERIMENTAL`** - deliberately not wired, kept for a later decision.
  Leave it, and put the reason in the module's own docstring so the next
  audit does not re-find it. `backend.mojo` and `_public_api_plan.py` are
  the two that qualify today.
- **`PENDING`** - implemented, meant to be reachable, and blocked on a named
  edit in a file its lane does not own. Every `PENDING` finding belongs in
  the cross-lane patch queue with an owner.

Nothing in this repository is currently classified `DEAD`. That is a real
result and not a soft one: the unreachable code here is not abandoned, it is
unconnected, and the two call for opposite responses.

The second pass is the evidence for it. Every one of the four findings it
closed was closed by *connecting* something - an import, a parser, a
parameter list, a line ordering - and not one of them by deleting anything.
The modules were finished; what was missing was the edit in the file the
lane did not own. A `DEAD` finding would have called for the opposite
response, and there was nothing to make one about.

The classification lives in one table, `CLASSIFICATION` in
`tools/connectivity_audit.py`. Everything else in that script is mechanical.
An unclassified finding defaults to `PENDING`, which is the conservative
answer - it lands in the queue and a human decides.

---

## 11. What this audit cannot tell you

It reads text. It does not build, import, or run anything, and it does not
parse Mojo - it matches import statements and top-level declarations with
regular expressions tuned to this repository's style. A file written in some
other style is under-reported, never mis-reported: text the patterns do not
recognize contributes no edges.

So: **a connected path here is a path that exists, never a path that works.**
This audit cannot tell you that GPU training produces the same trees as CPU
training, that a registered binding has the right signature, that a
serialized field round-trips to the same value, or that any performance claim
holds. Those need the test suites, the parity checker, and hardware. Reaching
a module is the precondition for all of it, and the only thing measured here.

The cross-lane patch queue, with an owner for every `PENDING` finding above,
is in `handoffs/connect_22_audit.md`.

---

## 12. What the second pass closed

Four findings, in one place. Each is stated in full where it was first
reported; this is the index.

| Finding | Section | Closed by | Files |
| --- | --- | --- | --- |
| The five binding modules the extension did not import | 2 | lane 06 | `bindings/_mojotrees.mojo` |
| `gpu_portability` unreachable, and its own test aimed elsewhere | 4 | this lane | `src/mojotrees/histogram_gpu.mojo`, `tests/test_gpu_portability.mojo` |
| Nine `ExtraTreeParams` fields settable from C and the CLI and not from Python, and `enable_bundle` with them | 5 | this lane | `python/mojotrees/__init__.py`, `python/mojotrees/basic.py`, `bindings/basic_bindings.mojo`, `bindings/_mojotrees.mojo` |
| `_compat.py`'s interpreter guard never running | 6 | this lane | `python/mojotrees/__init__.py` |

Two things are worth carrying forward from how they were fixed.

**One parser, two callers.** Both parameter fixes took the same shape: a
`*_from_mapping` function in `bindings/basic_bindings.mojo`, called by the
checker a user can ask with *and* by the `_parse_params` a fit runs through.
The alternative - a parser at the boundary and a second one behind the
checker - is section 8's duplicate-registry problem in a new place, and it
would fail in the worst available way, by validating something other than
what gets trained.

**No range check crossed into Python.** Every value added to the estimator
signature is checked natively, by the same code the C ABI and the CLI reach.
A Python-side copy of a bound would be the thing the next audit finds.

**Not done here.** `docs/LIGHTGBM_PARITY.md` carries roughly a dozen rows
that say some variant of "no Python estimator parameter reaches it", and
those rows are now false. That file belongs to the parity lane and is gated
by `tools/check_parity.py`, so the exact replacement text is a patch request
in §7 of `handoffs/connect_22_audit.md` rather than an edit made here.
