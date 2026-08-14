# Task 09 handoff: the LightGBM parity contract, refreshed

Lane 09 of a parallel round. Files this lane owns and changed, and nothing
else:

| Path | State |
|---|---|
| `docs/LIGHTGBM_PARITY.md` | rewritten as contract version 2 |
| `docs/CAPABILITY_LEVELS.md` | new |
| `tools/check_parity.py` | extended |
| `handoffs/integration_09_parity.md` | this file |

This lane committed and staged nothing. It should be noted anyway that a
concurrent lane's whole-tree commits swept these files in while they were
being written: `docs/CAPABILITY_LEVELS.md` and `tools/check_parity.py`
landed in `9a9c8d1 "Prepare packaging and parallel optimization work"`, and
`handoffs/integration_09_parity.md` in `b04b5f0 "Integrate parallel release
and accelerator work"`. The content in `9a9c8d1` is final for both files.
`docs/LIGHTGBM_PARITY.md` and this handoff carry later edits in the working
tree, from the second packaging pass described in section 4. Whoever
assembles the round should know that this lane's work is spread across two
commits with other lanes' messages.

No implementation, test, README, packaging file, or workflow was touched.

## What was run, and what was not

**Nothing was executed.** No Python, no Mojo, no pixi, no pytest, no build,
no benchmark, and in particular **`tools/check_parity.py` was not run**, so
this lane cannot claim the script passes. It was verified statically
instead, with `grep`, `sed`, and `awk` over the two files it reads:

- every one of the 99 repository paths the contract cites resolves to a file
  that exists
- every status cell in the contract is one of the five vocabulary words
  (142 `supported`, 61 `deferred`, 45 `different`, 37 `partial`, 17
  `unsupported`)
- all 104 names in `REQUIRED_SUPPORTED` appear as a `supported` row
- all 36 keys in `STALE_DEFERRED_WATCHES` appear as a row, and every one of
  those rows is `deferred` or `unsupported`
- none of the watched public symbols resolves today, so the new check is
  armed and silent rather than failing on arrival
- every new name in `REQUIRED_MOJO_EXPORTS`, `REQUIRED_BASIC_METHODS`,
  `REQUIRED_FITTED_ATTRS`, `REQUIRED_FIT_ARGS`, and
  `REQUIRED_PREDICT_ARGS` exists in the file the script parses
- every cited `tests/*.mojo` suite is named in a `pixi.toml` task

The first thing the integration owner should do is run
`python3 tools/check_parity.py`.

## What the audit read

The statuses below were re-derived from the live entry points, not from
version 1's prose and not from any other lane's handoff:
`python/mojoboost/__init__.py`, `basic.py`, `_eval.py`, `_arrays.py`,
`cv.py`, `inspection.py`, `dask.py`, `device_selection.py`,
`bindings/_mojoboost.mojo`, `src/mojoboost/__init__.mojo`, `device.mojo`,
`model.mojo`, `serialize.mojo`, `sampling.mojo`, `capi/mojoboost.h`,
`capi/README.md`, `cli/README.md`, `pixi.toml`,
`.github/workflows/ci.yml`, `docs/COMPATIBILITY_POLICY.md`,
`docs/PLATFORM_MATRIX.md`, and `docs/GPU_VALIDATION.md`.

The LightGBM-side inventories in sections 7, 8, and 9 (the 141 parameter
names, the objective names, the metric names) were **not** re-enumerated
against a running LightGBM install. They are carried forward from the
version 1 audit against 4.7.0, and the contract now says so in its header.

## 1. Status changes, with evidence

Every row whose status word changed. Rows that only gained a note are in
section 2.

| Row | v1 | v2 | Evidence |
|---|---|---|---|
| `cv` (section 1) | deferred | partial | `python/mojoboost/cv.py` implements `cv()` with folds, `stratified`, `shuffle`, caller `folds` or a scikit-learn splitter, `metrics`, `feval`, `fpreproc`, `init_model`, `eval_train_metric`, `return_cvbooster`, early stopping. Tested by `python/tests/parallel/test_cv.py`. Not `supported`: it is not in `mojoboost.__all__`, and section 2 of `docs/COMPATIBILITY_POLICY.md` says importing a submodule other than `basic` is not public |
| `CVBooster` (section 1) | deferred | partial | Same file, same reachability limit |
| `plot_split_value_histogram` (section 1) | deferred | unsupported | The plot was never in scope (matplotlib is not a dependency mojoboost will take, which is what the other two plotting rows already said). The deferral was about the data behind it, and the data now exists as `mojoboost.inspection.get_split_value_histogram`. Judgment call, see section 4 |
| `plot_tree` / `create_tree_digraph` (section 1) | deferred | unsupported | Same, with graphviz. The structured tree exists as `mojoboost.inspection.dump_model` |
| `fit(eval_metric=)` (section 3) | partial | supported | `python/mojoboost/_eval.py` resolves 21 LightGBM metric names plus aliases to a code, a direction, and a task; `bindings/_mojoboost.mojo` exposes `eval_metric`; the value comes from `src/mojoboost/metrics.mojo`, the same code the Mojo API calls. Callables, tuples, and dicts still work |
| `fit(eval_sample_weight=)` (section 3) | different | supported | It is an argument of all three estimators' `fit` and weights the built-in metrics. The v1 claim that "validation rows are never weighted" is false as of this tree. Combining it with a caller-supplied callable raises, which is the remaining difference and is now in the note |
| `n_estimators_` / `n_iter_` (section 4) | different (one row) | two rows: `n_iter_` supported, `n_estimators_` different | `n_iter_` is in `_Base._FITTED_ATTRS` and is set on every estimator's `fit`. `n_estimators_` is not, and is not being added |
| `Booster.dump_model` / `trees_to_dataframe` (section 5) | deferred | partial | `mojoboost.inspection.dump_model` and `trees_to_dataframe` build the schema in `docs/MODEL_INSPECTION_SCHEMA.md` from `Booster.model_to_string()`. Not `supported`: they are module functions rather than `Booster` methods, split gains are absent, and the submodule import is not public |
| `Booster.get_split_value_histogram` (section 5) | deferred (bundled with lower/upper bound) | partial, and split into its own row | `mojoboost.inspection.get_split_value_histogram` and `split_values`. `Booster.lower_bound` / `upper_bound` stay `deferred` in a row of their own, because nothing computes them |
| `scipy.sparse` CSR (section 6) | deferred | partial | The chain is live end to end: `_arrays.check_X_sparse` to `fit_csc`/`predict_csr` in `bindings/_mojoboost.mojo` to `src/mojoboost/model_sparse.mojo`. The v1 claim that the sparse modules are "not exported from `src/mojoboost/__init__.mojo`, not reachable from Python, and have no tests" is false on all three counts. Not `supported`: no `eval_set`, no GPU, no ranker, no custom objective, no `pred_leaf`, no `pred_contrib`, no iteration slicing, and the Python-side tests skip in CI (section 3 below) |
| `scipy.sparse` CSC (section 6) | deferred | partial | Same |
| `is_enable_sparse` (section 7) | deferred | different | There is no toggle to defer. The input type decides which path runs, and neither path converts to the other |
| `metric` (section 7) | partial | supported | Section 9's names are reachable from `fit(eval_metric=)`, from `MetricSuite` in Mojo, and from `metrics=` in `mojoboost.cv` |
| `predict_contrib` (section 7) | partial | supported | `predict(pred_contrib=True)` is on all three estimators, not Mojo only. Dense input only, which is in the note |
| `GPU multiclass` (section 12) | partial | supported | `fit_multiclass` in `src/mojoboost/model.mojo` resolves the device and calls `train_multiclass_gpu`; `gpu_supports` in `src/mojoboost/device.mojo` returns True for every output count, with a docstring saying multiclass "is no longer the exception it was". `tests/test_device.mojo` covers the routing, `tests/test_gpu_objectives.mojo` compares GPU to CPU. See section 4: this row rests on a file that contradicts itself |
| `GPU prediction` (section 12) | supported | partial | Unchanged behavior, corrected status. The `predict` binding in `bindings/_mojoboost.mojo` takes no device, so the Python estimators cannot reach it; v1's own note said "Mojo API only" while the status said `supported` |
| `C API` (section 13) | deferred | partial | `capi/mojoboost.h` declares twelve functions with an ABI version query, `capi/mojoboost_capi.mojo` implements them, `tests/test_capi.mojo` runs inside `pixi run test` and therefore in CI. Not `supported`: it is not LightGBM's `c_api.h`, `pixi run test-c` is in no CI job, and no artifact ships the library |
| `macOS arm64 wheel` (section 13) | supported | partial | **The one downgrade.** `packaging/build_wheel.sh` and `packaging/test_wheel.sh` exist and are correct as far as reading them shows, and a release lane landed `.github/workflows/release-macos.yml` mid-round, which builds, verifies, clean-installs, and hashes the wheel on a tag or a manual dispatch. Nothing guards it on an ordinary change: `.github/workflows/ci.yml` has `test`, `python`, `parity`, and a manual `bench`, and no wheel job. `docs/PLATFORM_MATRIX.md` independently records the target as `designed` and says explicitly that a wheel found on a disk is not a record. Removed from `REQUIRED_SUPPORTED` in `tools/check_parity.py`, as that check's own failure message instructs. See section 4 |
| `macOS x86-64 wheel` (section 13) | deferred | unsupported | `docs/PLATFORM_MATRIX.md` lists `macos-x86_64` as `unsupported`, not as waiting for a machine |
| `manylinux wheel` (section 13) | deferred | split: new `Linux wheel, x86-64 and ARM64` row at partial, `manylinux wheel` stays deferred | `packaging/linux/build_wheel_linux.sh` and `.github/workflows/release-linux.yml` build both architectures with ELF inspection and a metadata check, and default to plain `linux_x86_64` / `linux_aarch64` tags. That workflow treats promotion to a manylinux tag as a deliberate input with a glibc floor you must have measured, so the wheel exists and the manylinux promise does not |

### Rows added

| Row | Status | Why it is new |
|---|---|---|
| Section 0, the whole capability-level table | 19 capabilities scored | The seven levels, per `docs/CAPABILITY_LEVELS.md` |
| `fit(eval_group=)` (section 3) | supported | The ranker takes per-validation-set query boundaries and nothing said so |
| `categorical_feature_` (section 4) | different | A fitted attribute that exists and was undocumented |
| `Booster.lower_bound` / `upper_bound` (section 5) | deferred | Split out of the `get_split_value_histogram` row |
| `n_estimators_` (section 4) | different | Split out of the `n_iter_` row |
| Persistent GPU session and scheduling (section 12) | partial | `src/mojoboost/gpu_runtime.mojo` is exported and partly used |
| Apple-specific tiling policy (section 12) | deferred | `src/mojoboost/apple_gpu_policy.mojo` exists and nothing reads it |
| Sparse input on the GPU (section 12) | unsupported | The sparse path raises for `device="gpu"`, which nothing recorded |
| command line tool (section 13) | different | `cli/mojoboost` exists and the contract said the CLI surface was "never" |
| callbacks in `cv` (section 10) | partial | `mojoboost.cv(callbacks=)` |

## 2. Notes corrected without a status change

Short list, because each one was a false or stale sentence:

1. `Booster` (section 1): "`dump_model` is the gap, task 14" is no longer
   the gap; the dump exists, the gains do not.
2. `fit(eval_set=)` and `fit(callbacks=)` (section 3): validation sets and
   early stopping now cover the softmax classifier and the ranker.
   Callbacks still do not, and both trainers refuse a callback list.
3. `objective` (section 2): the classifier rejects `objective=` with a
   message rather than a `TypeError`, and the regressor gained `base_score`
   for custom objectives.
4. `boost_from_average` (section 7) and custom objectives (section 8):
   `base_score` is the explicit starting raw score, a number or `"mean"`.
   The Python custom-objective path is CPU only, because the `fit_custom`
   binding takes no device.
5. `device_type` (section 7): `gpu` now covers multiclass.
6. `feature_fraction_bynode` (section 7): `src/mojoboost/train_gpu.mojo`
   calls the same `select_node_features` the CPU grower calls, so it is not
   a CPU-only parameter.
7. `top_rate` / `other_rate` (section 7): `goss_warmup_rounds` is exposed.
8. Every `tree_parameters_extra.mojo` parameter (`min_gain_to_split`,
   `max_delta_step`, `path_smooth`, `feature_contri`, `monotone_penalty`,
   `extra_trees`, the two computable CEGB penalties, `forcedsplits_filename`,
   `monotone_constraints_method`): the rule exists as a primitive and no
   grower calls it, so setting the parameter still does nothing. The status
   did not move for exactly that reason.
9. `enable_bundle`, the transport rows, and the Dask rows: same treatment
   for `efb.mojo`, `distributed_transport.mojo`, and
   `python/mojoboost/dask.py`.
10. Section 9's closing paragraph said "Every name above is selectable from
    Python", which included the two `deferred` metrics and the two Mojo-only
    accuracy spellings. It now says "Every metric marked `supported`".
11. `Dataset.subset` (section 5): no longer blocking cross validation, and
    the reason (`cv` re-bins per fold to avoid leaking quantiles) is worth
    keeping.
12. `two_round` / `header` / `label_column` and friends (section 7): the
    mojoboost CLI has `--header`, `--label`, and `--weight` flags, which are
    not these parameters and are not accepted in a parameter string. Said so
    rather than leaving a reader to infer a contradiction.

The "Known gaps" section was rewritten from five items to eight. Items 2
and 4 of the old list were already marked "Closed" and are dropped; the
sparse item and the TreeSHAP item are dropped because both are now false.

## 3. Requests on files this lane may not edit

None of this is applied. Each is a fact the audit found, in a file lane 09
does not own.

1. **`pixi.toml`: scipy is missing from `[feature.pytest.dependencies]`.**
   `python/tests/test_validation.py` and `python/tests/test_contrib.py`
   guard their sparse cases with `pytest.importorskip("scipy.sparse")`, so
   `pixi run -e pytest test-estimators` skips them, and that is the only
   Python test run CI performs. Adding `scipy = "*"` next to `pandas` is
   what would let section 6's rows move from `partial` to `supported`.
2. **`python/mojoboost/_sklearn.py`: `input_tags.sparse` is False** while
   `fit` and `predict` accept sparse input, and
   `python/tests/test_sklearn_integration.py` asserts the False. A
   scikit-learn utility that respects the tag will densify before calling
   mojoboost, which defeats the sparse path. Whoever owns the sparse lane
   should decide whether the tag or the assertion moves first.
3. **`src/mojoboost/device.mojo`: the module docstring contradicts
   `gpu_supports` in the same file.** The docstring says "Multiclass grows
   one tree per class per round on the CPU only, so `gpu` raises for it and
   `auto` chooses the CPU"; `gpu_supports` returns True for every output
   count and its own docstring says multiclass is no longer the exception.
   The contract believes the code. If the docstring is right and the guard
   is the bug, section 12's `GPU multiclass` row has to go back to `partial`.
4. **`python/mojoboost/__init__.py`: the module docstring contradicts
   itself about built-in metrics.** The "validation sets and early stopping"
   section documents `eval_metric="l2"` and the whole name table; the
   "prediction options" section, twelve lines later, says "There are no
   built-in validation metrics in the Python API yet". The second is stale.
5. **`python/mojoboost/__init__.py`: `cv` and `CVBooster` are not
   exported.** `handoffs/task15_cv.md` asks for this. Until it happens, or
   until `docs/COMPATIBILITY_POLICY.md` grows an exception for these
   submodules, sections 0 and 1 keep them at `partial`. The same sentence
   applies to `mojoboost.inspection`.
6. **`bindings/_mojoboost.mojo`: no `split_gains` function.**
   `python/mojoboost/inspection.py` picks gains up automatically the moment
   one exists, and `src/mojoboost/inspection.mojo` already computes them
   (`def split_gains(trees)`). Until then every dump reports
   `has_split_gain: False`.
7. **`.github/workflows/ci.yml`: `pixi run test-c` and the wheel tasks are
   in no job.** Those two absences are the whole reason the `C API` row is
   `partial` and the `macOS arm64 wheel` row was downgraded.

## 4. Uncertain rows, for the integration owner to decide

Ranked by how much a wrong call costs.

1. **The three wheel rows, and how fast they are moving.** Section 13 was
   audited twice, because a release lane landed `packaging/macos/`,
   `packaging/linux/`, `.github/workflows/release-macos.yml`,
   `.github/workflows/release-linux.yml`, and
   `.github/workflows/release-provenance.yml` while this lane was writing.
   The second pass read those files in the tree, which is evidence, and
   left `macOS arm64 wheel` at `partial` on one narrow ground: a release
   workflow that fires on a tag is not a per-change guard, and
   `docs/PLATFORM_MATRIX.md` still says `designed`. **If that lane adds a
   per-change wheel job, or records a clean-install run in the matrix,
   restore the row to `supported` and put `"macOS arm64 wheel"` back in
   `REQUIRED_SUPPORTED`.** Do it on the evidence, not on their handoff. The
   same test decides the new `Linux wheel, x86-64 and ARM64` row. Expect
   section 13 to be the first part of this contract to go stale.
2. **`GPU multiclass`, upgraded to `supported`.** It rests on
   `gpu_supports` admitting every output count, in a file whose module
   docstring says the opposite. Someone who knows which of the two is
   intended should confirm before this row is quoted anywhere.
3. **`GPU prediction`, downgraded to `partial`.** Nothing about the
   capability changed; this is a consistency fix between the status word
   and v1's own note. If the project's reading of `supported` includes
   "reachable from the Mojo API alone", revert it, and say so in the status
   vocabulary so the next auditor does not undo it again.
4. **`plot_split_value_histogram` and `plot_tree`, moved from `deferred` to
   `unsupported`.** The reasoning is that these two rows were deferred only
   because their underlying data was missing, and the plotting itself was
   always out of scope for the same reason `plot_importance` is. If
   plotting is genuinely wanted later, these belong back at `deferred`.
5. **`cv`, `CVBooster`, and the inspection rows at `partial`.** The whole
   argument is section 2 of `docs/COMPATIBILITY_POLICY.md`. If the policy's
   author intended that clause to cover only the private helpers, these
   rows are `supported` today and should be moved.
6. **Sparse at `partial` rather than `supported`.** Two independent
   reasons, either of which alone would keep it there: the feature gaps
   listed in the row, and the untested-in-CI problem from section 3. The
   first is enough on its own, so this one is the least likely to be wrong.
7. **`fit(eval_sample_weight=)` at `supported`.** LightGBM weights
   validation metrics; mojoboost does too, now, but refuses to combine
   weights with a caller's callable. That is a narrowing, and someone may
   prefer `partial` for it.
8. **`is_enable_sparse` moved from `deferred` to `different`.** Arguable:
   LightGBM's parameter also controls sparse optimization of dense input,
   which mojoboost has no equivalent of at all.

## 5. What changed in `tools/check_parity.py`

Everything version 1 checked, it still checks. Three things are new.

**A stale-deferred watch.** `STALE_DEFERRED_WATCHES` maps 36 rows to
public-symbol probes. When a row says `deferred` or `unsupported` and every
probe behind it resolves, the script fails and asks for a re-audit of that
one row. It never decides what the row should say.

The probe grammar is deliberately narrow and deliberately symbol only:

    pyall:NAME              NAME in mojoboost.__all__
    pysub:MODULE:NAME       NAME in python/mojoboost/MODULE.py's __all__
    pymethod:CLASS.METHOD   METHOD defined on CLASS
    pyarg:CLASS.METHOD:ARG  ARG is a parameter of CLASS.METHOD
    pyattr:NAME             NAME in _Base._FITTED_ATTRS
    mojo:NAME               NAME re-exported from src/mojoboost/__init__.mojo

**A file path is never a probe**, and the script says why in its docstring.
Six modules in this tree are fully implemented, individually tested, and
callable by nobody; a check that upgraded a row because a file appeared
would have manufactured six false claims in this round alone. The six are
watched by the name an integrator would have to export to reach them
(`fit_bundles`, `passes_min_gain`, `RankAddress`, `parse_lgbm_model`,
`derive_block_threads`, `select_level_features`, `split_gains`).

A watch whose row name is not in the contract is itself a failure, so a row
rename cannot silently disarm the watch on it.

**Level-table checks.** The script now reads `docs/CAPABILITY_LEVELS.md`,
requires it to define exactly the seven levels in order, requires section
0's table to use those seven as its columns, requires each cell to be
`yes`, `no`, or `n/a` with a non-empty evidence cell, and enforces the two
implications the levels carry by definition (integrated implies
implemented, publicly reachable implies integrated) plus the one that
matters for drift: a capability that is publicly reachable cannot also be
`deferred` or `unsupported`.

**A wider symbol inventory.** `REQUIRED_MOJO_EXPORTS` gained the sparse
chain and the contribution entry points; `REQUIRED_FITTED_ATTRS` gained
`stopped_early_`, `n_iter_`, and `categorical_feature_`;
`REQUIRED_FIT_ARGS` gained `eval_sample_weight` and `callbacks` and a real
entry for the ranker; `REQUIRED_PREDICT_ARGS` gained `pred_contrib` and the
ranker; `REQUIRED_BASIC_METHODS` gained `Dataset.feature_name` and
`Dataset.categorical_feature`; and `basic.train` is now checked for
`valid_sets`, `valid_names`, and `init_model`.

## 6. What this lane did not do

- It did not run anything, so no claim here is a test result.
- It did not upgrade any row on the strength of a parallel handoff. The
  handoffs in `handoffs/` were read for context only; every status word
  above cites a file in the tree.
- It did not touch the LightGBM-side inventories, which still need a pass
  against a running 4.7.0 install to confirm no parameter, objective, or
  metric name has been missed.
- It did not resolve the seven items in section 3, all of which live in
  files this lane was not allowed to edit.
