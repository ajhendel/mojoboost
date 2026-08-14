# LightGBM parity contract

Contract version: 3
Audited against: **LightGBM 4.7.0** (the version pinned in the `bench` pixi
environment; `lightgbm.__version__` reported `4.7.0` when version 1 of this
file was written)
Audited: 2026-08-14, mojoboost at commit `29d76e4` plus the working tree of
that day. The import closure was re-derived at `63aad82` and re-checked at
`9c1e771` and `29d76e4`; it is unchanged across all three, because every
edge added in between either sits inside a module that was already
reachable or points into one that was already an orphan

What version 3 re-derived: reachability. Every `integrated` and `publicly
reachable` cell was recomputed by tracing imports from the four shipping
roots (`bindings/_mojoboost.mojo`, `src/mojoboost/__init__.mojo`,
`capi/mojoboost_capi.mojo`, `cli/mojoboost_cli.mojo`) and by reading
`mojoboost.__all__` and the `def_function` table, rather than by trusting
version 2's prose. The module-level result is in
`docs/INTEGRATION_INVENTORY.md` and the map is `docs/ARCHITECTURE.md`. What
it did not re-derive: LightGBM's own inventories in sections 7, 8, and 9,
which are carried forward from the version 1 audit against 4.7.0 and have
not been re-enumerated against a running install.

Other lanes were editing the tree while this was written, so treat the
statuses as a snapshot: run `python3 tools/check_parity.py` and re-read the
rows you care about before relying on one. The check is what keeps the
snapshot from rotting silently. It can now also tell you that a `deferred`
row looks stale, by watching the public symbols behind it, but deciding
what a row should say instead is still a judgment for whoever moved the
code.

This file is the authoritative statement of what mojoboost does and does
not do relative to LightGBM's public interface. It is a contract, not a
wish list. A row saying `supported` means the named behavior exists, is
reachable from a public entry point, and has a test that would fail if it
stopped working.

## How to use this document

- **Before implementing** a LightGBM feature, find its row. If it already
  says `supported` or `partial`, strengthen what is there instead of adding
  a second way to do the same thing.
- **When a public contract changes**, update the row in the same change that
  updates the code, the bindings, the docs, and the serialization format.
- **When you change a row**, run `python3 tools/check_parity.py`. It fails if
  a row that must stay `supported` was deleted or downgraded, if a cited file
  no longer exists, if a named public symbol disappeared, or if a `deferred`
  row's public symbols have quietly appeared.

## Status vocabulary

| Status | Meaning |
|---|---|
| `supported` | Reachable from a public entry point, tested, and semantically equivalent to LightGBM except where the notes say otherwise |
| `partial` | Exists but does not cover LightGBM's full surface (fewer inputs, one backend, Mojo only, reachable but not public, and so on). The notes say what is missing |
| `different` | Deliberately not LightGBM's design. The notes say why, and what mojoboost does instead |
| `deferred` | Agreed to be worth doing, out of scope for v1. The notes say why, not merely that it is absent |
| `unsupported` | Not planned. The notes say why not |

`deferred` and `unsupported` are not interchangeable. Nothing difficult and
high value is marked `unsupported` to keep the matrix clean; the expensive
items (a real distributed transport, LightGBM model file interop, non-Apple
GPU validation) are `deferred` with a named reason and, where one exists, a
task number in the maintainer's local planning notes.

**A status word is a summary, and a summary of seven separate facts.**
`docs/CAPABILITY_LEVELS.md` defines them: implemented, integrated, publicly
reachable, focused-tested, differential-tested, hardware-validated, and
release-packaged. Section 0 below scores every capability where those facts
currently disagree with each other, which is where a one-word status is
most likely to mislead. Read that section before quoting a row from
sections 1 to 13 in anything a user will see.

## v1 scope

**In v1.** A user should be able to migrate a LightGBM tabular workload that
uses the scikit-learn estimators, without dropping to a fork of their code:

- the three scikit-learn style estimators, LightGBM's parameter names, and
  the aliases LightGBM's own estimators use
- the objectives and metrics ordinary tabular work uses (regression family,
  binary, multiclass, ranking, Poisson, quantile, MAE, Huber, custom)
- validation sets, evaluation metrics, and early stopping
- all four prediction modes: response, raw score, leaf index, and feature
  contributions
- continued training from a fitted model, and initial scores
- native categorical features and missing values
- CSR/CSC input without densification
- Dataset-level and Booster-level APIs for users who do not want an
  estimator
- cross validation
- model inspection, dumping, and editing
- CPU parity on every objective, GPU coverage for the objectives that share
  the per-row gradient/Hessian interface
- Linux and macOS wheels

**Out of v1, deferred.** Real distributed transports and the Dask
integration built on them, unbiased LambdaRank, DART and random-forest
boosting, linear trees, and CEGB. Each is a subsystem, and each is listed
with its reason below rather than dropped.

**Never.** LightGBM's file-based configuration surface (`config`, `data`,
`valid`, `output_model`, `task`, `convert_model`, and the parameters that
only feed it), its deprecated aliases, and any parameter whose only purpose
is to select between implementations mojoboost does not have. mojoboost has
its own command line tool, which is not that surface; see section 13.

---

## 0. Capability levels for the contested capabilities

The capabilities where "does mojoboost have this" has more than one honest
answer, scored against `docs/CAPABILITY_LEVELS.md`. Cells are `yes`, `no`,
or `n/a`. A capability not listed here is one whose levels agree with its
status word, and the row in the sections below is the whole story.

| Capability | Status | implemented | integrated | publicly reachable | focused-tested | differential-tested | hardware-validated | release-packaged | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| SciPy sparse Python input | partial | yes | yes | yes | no | no | n/a | yes | Chain is live: `python/mojoboost/_arrays.py` to `bindings/_mojoboost.mojo` (`fit_csc`, `predict_csr`) to `src/mojoboost/model_sparse.mojo`. The Mojo layer is covered by `tests/test_sparse.mojo` in `pixi run test`; the Python entry point has two tests that call `pytest.importorskip("scipy.sparse")` and `pixi.toml` does not put scipy in the `pytest` environment, so they skip in the only run CI performs |
| Exact TreeSHAP contributions | supported | yes | yes | yes | yes | yes | n/a | yes | `src/mojoboost/contrib.mojo`, reached by `predict(pred_contrib=True)` on all three estimators. `tests/test_contrib.mojo` checks against Shapley values enumerated over all 2^M subsets, which is an independent implementation, and runs in `pixi run test` |
| Cross-validation (`mojoboost.cv`) | partial | yes | yes | yes | yes | no | n/a | yes | `python/mojoboost/cv.py`, orchestration over `Dataset`, `Booster`, and `Booster.eval`, so a fold model is what `train()` would have built. `python/tests/parallel/test_cv.py` runs under `pixi run -e pytest test-estimators`. `cv` and `CVBooster` are in `mojoboost.__all__`, so both are public under section 2 of `docs/COMPATIBILITY_POLICY.md`; `partial` is about the surface rather than the reach, since LightGBM's `cv` returns a plain dict keyed the same way but has been exercised against far more of its own parameter space than this one has |
| Model inspection and dump (`mojoboost.inspection`) | partial | yes | yes | yes | yes | no | n/a | yes | `dump_model`, `trees_to_dataframe`, `trees_to_records`, and `get_split_value_histogram` are in `mojoboost.__all__` and resolve the submodule lazily, so all four are public. `partial` is now about the *producer*, not the reach: `python/mojoboost/inspection.py` rebuilds the schema by parsing `Booster.model_to_string()`, while `src/mojoboost/inspection.mojo` and `src/mojoboost/model_dump.mojo` are the native producers with no caller. `bindings/_mojoboost.mojo` now imports `bindings/inspection_bindings.mojo` and registers `dump_model`, `dump_model_multiclass`, `split_values`, `dump_leaf_index`, `dump_raw_scores`, and `objective_code`, so an extension built from this tree takes the native route and the parser is the fallback rather than the path. What keeps this `partial` is `split_gains`, which exists on neither side of the seam: every dumped node reports `split_gain: None`. `python/tests/parallel/test_inspection.py`, and `docs/INTEGRATION_INVENTORY.md` |
| Split gains in a dump | supported | yes | yes | yes | yes | no | n/a | yes | `src/mojoboost/serialize.mojo` writes per-node split gains as of model format v4, so a dump reports a measured gain per internal node and `has_split_gain: True`, whichever path built it, and `Booster.feature_importance("gain")` survives a save, a pickle, and a `load`. A model read from a v1, v2, or v3 file carries none and says so through the flag; that is the only remaining `None`. `python/mojoboost/inspection.py` still looks for a `_mojoboost.split_gains` hook, which nothing defines, as the way an older file's gains could come from a live handle |
| Dask adapter (`mojoboost.dask`) | deferred | yes | no | no | yes | n/a | n/a | yes | `python/mojoboost/dask.py` validates partition metadata, plans ranks, negotiates capabilities, and predicts from model bytes, all against a fake backend in `python/tests/parallel/test_dask_contract.py`. No backend is registered by default, so `fit` raises `DistributedNotAvailable` before touching a cluster. `DaskRuntime` has never run against a live cluster |
| Explainable device selection (`mojoboost.device_selection`) | partial | yes | yes | yes | yes | n/a | n/a | yes | `explain_device_choice` is in `mojoboost.__all__`, so a user can ask "what would `device='gpu'` do here" and get a report rather than an exception. `python/mojoboost/device_selection.py` with 54 tests in `python/tests/parallel/test_device_selection.py`. Two things keep it `partial`. The estimators do not route through it: `_Base._resolve_device` calls `_mojoboost.resolve_device` directly, so a report and a `fit` are two calls into the same native engine rather than one. The `"narrow"` contract is no longer forced: `bindings/_mojoboost.mojo` registers `decide_device`, so an extension built from this tree carries the blocking reasons, warnings, memory estimate, and evidence identifier, and `report.contract` says which of the two answered. `docs/INTEGRATION_INVENTORY.md` |
| Startup diagnostics (`mojoboost.diagnostics`) | deferred | yes | no | no | no | n/a | n/a | yes | `python/mojoboost/diagnostics.py` formats phase durations that something else measured, over the ten-phase contract in `src/mojoboost/initialization.mojo`. Not a LightGBM capability, so it has no parity row of its own. The native half is wired now (`src/mojoboost/device_policy.mojo` and `src/mojoboost/gpu_runtime.mojo` both import `initialization`), and the Python half is not: nothing imports `python/mojoboost/diagnostics.py`, no name from it is in `mojoboost.__all__`, and neither `python/tests/` nor `tests/` has a suite for it |
| Exclusive feature bundling | partial | yes | yes | yes | yes | n/a | n/a | n/a | `src/mojoboost/efb.mojo` with `tests/parallel/test_efb.mojo` in `pixi run test`. Wired end to end as of this revision: `BoosterParams.bundling` carries `EfbSettings`, `src/mojoboost/boosting.mojo` fits the plan once with `prepare_bundling` and hands the bundled matrix to every tree, `src/mojoboost/split.mojo` unbundles a bundled histogram, and `src/mojoboost/params.mojo` parses `enable_bundle` and `max_conflict_rate`. Reachable through the parameter string, which section 2 of `docs/COMPATIBILITY_POLICY.md` makes public, and through `BoosterParams` from Mojo. `partial` because no Python estimator parameter reaches it, so a `MojoBoostRegressor` user still cannot ask for it |
| Distributed transport | partial | yes | yes | no | yes | n/a | n/a | n/a | `src/mojoboost/distributed_transport.mojo` with `tests/parallel/test_distributed_transport.mojo` in `pixi run test`. `src/mojoboost/distributed.mojo` now imports it and calls `require_transport`, `open_local_collective`, and `histogram_plan`, so the local runtime goes through the transport contract. Only the local runtime does: `RankAddress` is not exported, `run_distributed` raises for any other runtime spec, and no process has connected to another |
| LightGBM model file interop | deferred | yes | no | no | yes | no | n/a | n/a | `src/mojoboost/lgbm_model_io.mojo` with `tests/parallel/test_lgbm_model_io.mojo` in `pixi run test`. No caller, no export, and no test reads a file LightGBM actually wrote |
| Remaining tree-parameter rules | partial | yes | yes | yes | yes | n/a | n/a | n/a | `src/mojoboost/tree_parameters_extra.mojo`, with `tests/parallel/test_tree_parameters_extra.mojo`. Most of the bundle is wired as of this revision. `src/mojoboost/split.mojo` imports and calls `passes_min_gain`, `apply_monotone_penalty`, `extra_threshold_index`, and `finish_leaf_output`, which covers `min_gain_to_split`, `monotone_penalty`, `extra_trees` with `extra_seed`, `max_delta_step`, and `path_smooth`; forced splits reach the grower through `binning.map_forced_splits`; and `src/mojoboost/params.mojo` parses each of those keys, so they are reachable through the parameter string. Two rules are still inert: `feature_contri` and the CEGB penalties are named in the comments around the per-feature scan and no function computing either is imported anywhere. No Python estimator parameter reaches any of them |
| Apple GPU tuning policy | partial | yes | yes | no | yes | n/a | yes | n/a | `src/mojoboost/apple_gpu_policy.mojo` with `tests/parallel/test_apple_gpu_policy.mojo`. It is read now, in two places: `src/mojoboost/device_policy.mojo` takes `GpuProfile` and `partial_budget_bytes` from it for the device profile and the session memory estimate, and `src/mojoboost/apple_histogram_policy.mojo` takes its `derive_block_threads`. What it still does not do is decide any default launch: `apple_histogram_policy` resolves to `SPEC_LEVEL_BASELINE` unless `MOJOBOOST_GPU_HIST_SPECIALIZATION` asks otherwise, so `src/mojoboost/gpu_tiling.mojo` remains the tiling policy in force. Note that `derive_block_threads` now names two different functions, one per module, with different signatures |
| Per-level feature sampling | partial | yes | yes | yes | yes | n/a | n/a | n/a | `select_split_features` in `src/mojoboost/sampling.mojo`, with `tests/parallel/test_sampling.mojo`. `TreeParams.feature_fraction_bylevel` is carried through `src/mojoboost/tree.mojo`, `src/mojoboost/tree_sparse.mojo`, and `src/mojoboost/train_gpu.mojo`, `src/mojoboost/params.mojo` parses it under LightGBM's spelling and XGBoost's `colsample_bylevel`, and the distributed and device split-search paths refuse it by name rather than ignoring it. `select_level_features` itself still has no caller. It is XGBoost's parameter, so it is an extension rather than a parity row, and no Python estimator parameter reaches it |
| GPU packed-bin layout | deferred | yes | no | no | no | n/a | n/a | n/a | `src/mojoboost/gpu_binned_layout.mojo` plans a packed layout over the bit primitives in `src/mojoboost/gpu_bin_packing.mojo`. The module is reached as of this revision and the layout is still not planned: `src/mojoboost/histogram_gpu.mojo` imports `check_layout_support` and calls it when the builder opens, which refuses a feature count or an `n_rows * n_features` cell count that would wrap an Int32 index. That is one guard, not a packed plan: no caller builds a `BinLayoutPlan` (`plan_feature_major`, `plan_row_major`, `plan_feature_blocked` are named only in prose elsewhere), nothing calls `pack_binned_matrix` or a `gpu_bin_packing` primitive, and no suite imports either module. `docs/INTEGRATION_INVENTORY.md` |
| Level-wise GPU growth | deferred | yes | no | no | no | n/a | n/a | n/a | `src/mojoboost/gpu_levelwise.mojo` with its rule table in `src/mojoboost/levelwise_policy.mojo`. No trainer offers a level-wise mode, so nothing reaches either, and growth stays leaf-wise everywhere. No suite imports them |
| Class-batched GPU multiclass rounds | partial | yes | yes | yes | no | n/a | no | n/a | `src/mojoboost/gpu_multiclass_batch.mojo` over the interleaved planes in `src/mojoboost/gpu_output_planes.mojo`. Wired as of this revision: `src/mojoboost/train_gpu.mojo` asks `class_schedule` for a plan and, when that plan is not sequential, dispatches to `_train_multiclass_gpu_batched`, which builds a `GpuClassBatch.for_plan` and runs the round under `MulticlassRoundGuard`. The default is still one tree per class per round, which is what the supported row below describes; the batch widens only when `MOJOBOOST_GPU_CLASS_BATCH` asks, and that variable is public under section 2 of `docs/COMPATIBILITY_POLICY.md`, which is the whole of the reach. No suite imports the module, nothing has compared the batched round against the sequential one, and no accelerator has run it |
| Hybrid CPU and GPU leaf placement | deferred | yes | no | no | no | n/a | n/a | n/a | `src/mojoboost/hybrid_leaf_scheduler.mojo` decides per leaf which device should build its histogram, over the cache model in `src/mojoboost/histogram_cache_policy.mojo`. Both modules are reached as of this revision and neither places a leaf: `src/mojoboost/gpu_runtime.mojo` imports the scheduler, and `GpuSession.note_hybrid` builds a `HybridContext.from_env` so a run can report what `MOJOBOOST_HYBRID_LEAVES` and `MOJOBOOST_HYBRID_TRACE` asked for and why it was declined. The file says it plainly: "Nothing here moves a histogram." Every histogram is still built where it would have been built. No suite imports either module |
| GPU sparse and categorical kernels | deferred | yes | no | no | no | n/a | n/a | n/a | `src/mojoboost/gpu_categorical.mojo`, `src/mojoboost/gpu_sparse.mojo`, and `src/mojoboost/gpu_sparse_layout.mojo` exist as a chain that no root reaches. The GPU trainer still refuses both sparse input and categorical features by name, which section 12 records, so the refusal a user meets is accurate; what is not accurate is calling the kernels absent. No suite imports any of the three |
| DART and random forest boosting | deferred | yes | no | no | no | n/a | n/a | n/a | `src/mojoboost/alternate_boosting.mojo` dispatches to `src/mojoboost/boosting_dart.mojo` and `src/mojoboost/boosting_rf.mojo`. Nothing imports the dispatcher, so `src/mojoboost/boosting.mojo` still owns the only round loop and `boosting="dart"` and `"rf"` remain unrecognized. No suite imports any of the three |
| Cost-effective gradient boosting | deferred | yes | no | no | no | n/a | n/a | n/a | `src/mojoboost/cegb.mojo` implements LightGBM's four `cegb_*` controls as a gain adjustment, and `src/mojoboost/params.mojo` parses `cegb_tradeoff` and `cegb_penalty_split`. No grower charges a candidate, so the parsed values reach a config field nothing reads. No root reaches the module and no suite imports it |
| GPU multiclass training | supported | yes | yes | yes | yes | yes | yes | yes | `fit_multiclass` resolves the device and calls `train_multiclass_gpu` (`src/mojoboost/model.mojo`), and `gpu_supports` in `src/mojoboost/device.mojo` now admits every output count, so `MojoBoostClassifier(device="gpu")` trains on the accelerator. `tests/test_gpu_objectives.mojo` checks GPU against CPU, `tests/test_device.mojo` checks the routing. Apple M4 only (`docs/GPU_VALIDATION.md`) |
| GPU split selection on the device | supported | yes | yes | yes | yes | yes | yes | yes | `src/mojoboost/gpu_split_search.mojo`, reached from `grow_tree_gpu` and from `MOJOBOOST_GPU_SPLIT_STRATEGY=device`. Off by default: `SPLIT_SEARCH_AUTO` resolves to the host scan, because Float32 device gains can flip near-tie decisions and the measured speed difference does not pay for that on its own. It grows over a device-resident frontier, so a split builds one histogram and subtracts for the sibling, as the host-search grower does; see the measurements under "GPU split selection" below |
| GPU prediction | partial | yes | yes | yes | yes | yes | yes | yes | `src/mojoboost/gpu_predict.mojo`, reached from `Model.predict_batch(device=)` in the Mojo API. The Python `predict` binding takes no device (`bindings/_mojoboost.mojo`), so the estimators always predict on the CPU |
| C ABI | partial | yes | yes | yes | yes | n/a | n/a | no | `capi/mojoboost.h` declares twelve functions with a version query; `capi/mojoboost_capi.mojo` implements them. `tests/test_capi.mojo` runs in `pixi run test`, hence in CI. `capi/run_c_tests.sh` (`pixi run test-c`) compiles the C caller and is not in any CI job, and no wheel or release artifact carries the shared library |
| Command line tool | different | yes | yes | yes | yes | n/a | n/a | no | `cli/mojoboost_cli.mojo` with `tests/test_cli.mojo` in `pixi run test`. It is mojoboost's own CSV tool over the Mojo API, not LightGBM's config-file surface, which stays `unsupported` in section 7 |
| macOS arm64 wheel artifact | partial | yes | yes | yes | no | n/a | n/a | yes | `packaging/build_wheel.sh` bundles the four MAX runtime dylibs with an `@loader_path` rpath and re-signs, `packaging/test_wheel.sh` installs and smoke-tests it, and `packaging/macos/build_release_wheel.sh` is driven by a release workflow that builds, verifies, clean-installs, and hashes on a tag or a manual dispatch. `focused-tested` is `no` because no per-change job guards any of that: the everyday CI workflow has no wheel job, and `docs/PLATFORM_MATRIX.md` still records the target as `designed` |

---

## 1. Top-level Python package symbols

Every name in `lightgbm.__all__` for 4.7.0, verified by importing the pinned
package during the version 1 audit.

| LightGBM symbol | Status | Notes | mojoboost |
|---|---|---|---|
| `LGBMRegressor` | supported | `MojoBoostRegressor`. Named for the library, not for LightGBM; behavior and parameter names match where documented | `python/mojoboost/__init__.py` |
| `LGBMClassifier` | supported | `MojoBoostClassifier`. Binary and softmax multiclass, chosen from the label count | `python/mojoboost/__init__.py` |
| `LGBMRanker` | supported | `MojoBoostRanker`, LambdaRank | `python/mojoboost/__init__.py` |
| `LGBMModel` | different | No shared public base class. `_Base` holds the shared hyperparameters but is private, because a bare `LGBMModel` with `objective=` selecting the task is a second way to spell what the three estimators already do | `python/mojoboost/__init__.py` |
| `Booster` | supported | `mojoboost.Booster`: prediction, evaluation, feature importance, model IO, iteration counts, and continued training with `update()`. The estimators hold one on `booster_`, so there is a single model object. `dump_model` is not a method; `mojoboost.inspection.dump_model(model)` is the dump, section 0 | `python/mojoboost/basic.py`, `python/tests/test_basic.py` |
| `Dataset` | supported | `mojoboost.Dataset`, over the Mojo `Dataset` in `src/mojoboost/trainset.mojo`: data, label, weight, group, init score, feature names, categorical declaration, and binning metadata, binned once and reused. Immutable once constructed; see section 5 for the mutators mojoboost does not have | `python/mojoboost/basic.py`, `src/mojoboost/trainset.mojo`, `tests/test_trainset.mojo` |
| `train` | supported | `mojoboost.train(params, train_set, num_boost_round, valid_sets, valid_names, init_model)`. Trains the same trees the estimators train, which `python/tests/test_basic.py` asserts bit for bit. `init_model` continues from a fitted booster. No per-round history or early stopping here; those are on the estimators' `fit` | `python/mojoboost/basic.py` |
| `cv` | partial | `python/mojoboost/cv.py` implements it: folds, `stratified`, `shuffle`, caller-supplied `folds` or a scikit-learn splitter, `metrics`, `feval`, `fpreproc`, `init_model`, `eval_train_metric`, `return_cvbooster`, and early stopping, returning LightGBM's `{metric-mean, metric-stdv}` history. It re-bins per fold rather than slicing a constructed `Dataset`, so fold binning cannot leak. Reachable only as `from mojoboost.cv import cv`: it is not in `mojoboost.__all__`, which section 2 of `docs/COMPATIBILITY_POLICY.md` makes the definition of public. Section 0 | `python/mojoboost/cv.py`, `python/tests/parallel/test_cv.py` |
| `CVBooster` | partial | With `cv`, and reachable the same narrow way | `python/mojoboost/cv.py` |
| `early_stopping` | supported | `fit(callbacks=[early_stopping(rounds, first_metric_only=, verbose=, min_delta=)])`, and the `fit(early_stopping_rounds=, min_delta=)` spelling. The callback configures the trainer's own stopper rather than reimplementing the rule; passing both spellings raises. Differs in which round survives: the primary metric's best on the first validation set, not the pair that ran out of patience first | `python/mojoboost/callback.py`, `src/mojoboost/callback.mojo`, `src/mojoboost/custom_metric.mojo` |
| `log_evaluation` | supported | `log_evaluation(period=, show_stdv=)`. `period<=0` is silent. `show_stdv` is accepted and inert on `fit`: it formats a cross-validation fold's spread, and `fit` has no folds | `python/mojoboost/callback.py`, `src/mojoboost/callback.mojo` |
| `record_evaluation` | supported | `record_evaluation(dict)` fills the dict in place. `evals_result_` is still populated directly too; it starts one round earlier, at the base-score-only model | `python/mojoboost/callback.py`, `src/mojoboost/callback.mojo`, `python/mojoboost/__init__.py` |
| `reset_parameter` | partial | `reset_parameter(**schedules)` with lists or callables, for the nine hyperparameters the loop re-reads each round (`callback.RESETTABLE`). A key outside that set raises rather than being ignored, which LightGBM does not do. A learning-rate schedule bakes shrinkage into the leaf values | `python/mojoboost/callback.py`, `src/mojoboost/callback.mojo` |
| `EarlyStopException` | supported | Raised by a callback to stop the run; rolls the ensemble back to the best round as LightGBM does | `python/mojoboost/callback.py`, `src/mojoboost/callback.mojo` |
| `EvalResult` | different | The 4-tuple `(data_name, metric_name, value, is_higher_better)` is what `env.evaluation_result_list` holds, matching LightGBM's shape; there is no named type for it | `python/mojoboost/callback.py`, `src/mojoboost/callback.mojo` |
| `register_logger` | unsupported | mojoboost has no logging layer to redirect. Training is silent by design; adding a logger to redirect is not a goal | none |
| `plot_importance` | unsupported | Plotting belongs in the caller's plotting library. `feature_importances_` is the data; matplotlib is not a dependency mojoboost will take | none |
| `plot_metric` | unsupported | Same reason. `evals_result_` is the data | none |
| `plot_split_value_histogram` | unsupported | The plot is out of scope for the same reason as the other two. The data it renders now exists: `mojoboost.inspection.get_split_value_histogram(model, feature)` returns it, section 5 | `python/mojoboost/inspection.py` |
| `plot_tree` / `create_tree_digraph` | unsupported | The graphviz rendering is out of scope; graphviz is not a dependency mojoboost will take. The structured tree it renders now exists as `mojoboost.inspection.dump_model`, section 5 | `python/mojoboost/inspection.py` |
| `Sequence` | deferred | The incremental-data protocol is part of the `Dataset` work, tasks 7 and 10. Nothing in the tree implements it | none |
| `DaskLGBMRegressor` / `DaskLGBMClassifier` / `DaskLGBMRanker` | deferred | `python/mojoboost/dask.py` has the three classes and the client-side contract, and they raise `DistributedNotAvailable` on `fit` because no backend is registered and no transport is wired up. Building them on the in-process prototype would be a distribution claim with nothing behind it. Section 0, and task 17 after task 16 | `python/mojoboost/dask.py`, `docs/distributed.md` |

## 2. Estimator constructor parameters

Every parameter of `lightgbm.LGBMModel.__init__` in 4.7.0.

| LightGBM parameter | Status | Notes |
|---|---|---|
| `boosting_type` | partial | Accepted as `boosting` (LightGBM's native name) with `boosting_type` as an alias. `gbdt` and `goss` only; `dart` and `rf` are deferred, see section 7 |
| `num_leaves` | supported | Same default (31) |
| `max_depth` | supported | Same default (-1). Leaf-wise growth is preserved under the limit |
| `learning_rate` | supported | Same default (0.1) |
| `n_estimators` | supported | Same default (100) |
| `subsample_for_bin` | deferred | mojoboost bins from every row rather than a sample. Correct but slower on very large data; the sampled binner is a performance item, not a semantic one |
| `objective` | partial | Regressor: `regression`, `huber`, `quantile`, `mae`/`regression_l1`, `poisson`, `gamma`, `tweedie`, `mape`, `fair`, `cross_entropy`, or a callable. Classifier: rejected with a message rather than a bare `TypeError`, because the task comes from the labels and a custom objective is single-output only. The objectives that are not implemented are listed in section 8, and each is reported by name rather than as an unknown one |
| `class_weight` | supported | `MojoBoostClassifier(class_weight=...)`: `"balanced"` or a `{label: weight}` dict, folded into the row weights before training, so there is one weighting mechanism rather than two. scikit-learn's rule for `balanced` (row counts, not weighted counts). `src/mojoboost/class_weight.mojo` |
| `min_split_gain` | unsupported | LightGBM's `min_gain_to_split`. Not implemented in any grower; see section 7 |
| `min_child_weight` | supported | Alias for `min_child_hess` (LightGBM's `min_sum_hessian_in_leaf`), default 1e-3 |
| `min_child_samples` | supported | Alias for `min_data_in_leaf`, default 20 |
| `subsample` | supported | Alias for `bagging_fraction` |
| `subsample_freq` | supported | Alias for `bagging_freq` |
| `colsample_bytree` | partial | mojoboost's name is `feature_fraction`; `colsample_bytree` itself is **not** accepted as an alias, and neither is `colsample_bynode`, though `feature_fraction_bynode` is spelled natively. Confirmed against `_Base.__init__` in this audit. Adding the two `colsample_*` spellings is a v1 gap |
| `reg_alpha` | supported | Alias for `lambda_l1` |
| `reg_lambda` | supported | Alias for `lambda_l2`. Note the default differs: mojoboost defaults to 1.0, LightGBM to 0.0. Documented in the README defaults table |
| `random_state` | different | mojoboost has no single global seed. Each sampler takes its own (`bagging_seed`, `feature_fraction_seed`, `goss_seed`), and every stream is counter-based, so a draw depends only on its seed and index and never on history. One global seed would reintroduce the ordering dependence that design removes |
| `n_jobs` | different | Thread count is `MOJOBOOST_NUM_WORKERS`, an environment variable, not a model parameter, because it changes how a model is computed and never what it equals. Same rule as the GPU tiling knobs |
| `importance_type` | supported | `split` (default) and `gain`, LightGBM's two types |
| `**kwargs` (arbitrary core parameters) | different | Unknown keyword arguments raise. LightGBM forwards them to the C++ config, which silently accepts typos of parameters it does not know |

Additional mojoboost constructor parameters that LightGBM spells only as
core parameters (`use_missing`, `categorical_feature` and its
`categorical_features` alias, `max_cat_to_onehot`, `max_cat_threshold`,
`cat_smooth`, `cat_l2`, `min_data_per_group`, `interaction_constraints`,
`monotone_constraints`, `bagging_seed`, `top_rate`, `other_rate`,
`goss_seed`, `goss_warmup_rounds`, `feature_fraction`,
`feature_fraction_bynode`, `feature_fraction_seed`, `max_bin`, `device`)
are covered in section 7. The ranker adds `lambdarank_truncation_level`,
`sigmoid`, `lambdarank_norm`, and `ndcg_eval_at`; the regressor adds
`alpha`, `fair_c`, `tweedie_variance_power`, and `base_score`, which is the
starting raw score for a custom objective only (a number, or `"mean"` for
the weighted label mean).

**Alias rule, intentionally different.** Setting a parameter and its alias
to different non-default values raises. LightGBM warns and keeps one. A typo
that silently trains a different model is worse than a failed call.

## 3. Estimator `fit` and `predict`

| LightGBM argument | Status | Notes |
|---|---|---|
| `fit(X, y)` | supported | numpy, pandas, plain sequences, or a SciPy sparse matrix (section 6) |
| `fit(sample_weight=)` | supported | Weighted gradients, Hessians, and base scores; zero-weight rows drop out |
| `fit(init_score=)` | deferred | Starting from an existing score vector is v1 work, task 6. `Dataset(init_score=)` has it on the functional API (section 5); the estimators do not take the argument |
| `fit(group=)` | supported | Ranker only, LightGBM's `group` array. `group_from_query_ids` builds it from a query id column |
| `fit(eval_set=)` | supported | List of `(X, y)` pairs, or a single pair, or the `eval_X=`/`eval_y=` spelling. Every estimator takes it, including the softmax classifier and the ranker. A classifier's validation labels go through the encoding `classes_` records, so a label absent from training is an error rather than a silent miscount. Not available for sparse input |
| `fit(eval_names=)` | supported | Names for the validation sets, used as `evals_result_` keys; `valid_0`, `valid_1`, ... by default |
| `fit(eval_sample_weight=)` | supported | One weight vector per validation set, applied to the built-in metrics by the same Mojo code that scores them. Combining it with a caller-supplied callable raises, because a callable is handed unweighted predictions and dropping the weights quietly would be worse than refusing |
| `fit(eval_init_score=)` | deferred | With `init_score`, task 6 |
| `fit(eval_metric=)` | supported | LightGBM's built-in metric names (`l2`, `rmse`, `l1`, `quantile`, `huber`, `mape`, `fair`, `poisson`, `gamma`, `gamma_deviance`, `tweedie`, `cross_entropy`, `kullback_leibler`, `binary_logloss`, `binary_error`, `auc`, `average_precision`, `multi_logloss`, `multi_error`, `ndcg`, `map`, and their aliases), callables, tuples, dicts, or a mixture, defaulting to the objective's own loss. A name is resolved by `python/mojoboost/_eval.py` and computed by `src/mojoboost/metrics.mojo`, the same code the Mojo API calls, so the two answers cannot drift. Two deliberate differences: a metric that cannot mean anything for the model being fitted is rejected rather than scored, and a callable's direction is declared up front with `("name", f, True)` rather than returned per call, because early stopping needs it before the first evaluation |
| `fit(feature_name=)` | partial | Feature names are read from a pandas frame's columns into `feature_names_in_` and checked at predict time, and `Dataset(feature_name=)` takes them on the functional API. An explicit `feature_name=` argument on the estimators' `fit`, and carrying names into the model file, are not there |
| `fit(categorical_feature=)` | supported | Accepted as `categorical_feature` on the constructor (LightGBM's name) rather than on `fit`, because scikit-learn's clone contract keeps hyperparameters on the estimator. Indices, column names, or `"auto"` (the default, meaning every pandas `category` column), and reported back on the fitted `categorical_feature_`. One difference: a `category` column left out of an explicit list raises, where LightGBM quietly feeds its codes to the numerical scan |
| `fit(callbacks=)` | partial | Supported for the regressor and the binary classifier, which train through `train_with_callbacks`. The softmax and LambdaRank loops score validation metrics and stop early but have no per-round hook, so they refuse a callback list rather than ignoring it. Needs an `eval_set` |
| `fit(init_model=)` | deferred | Continued training from the estimators is task 6. `train(init_model=)` has it on the functional API (section 5) |
| `predict(X)` | supported | Response scale, matching LightGBM's default |
| `predict(raw_score=)` | supported | Scores on the link scale. The objectives without a link (squared error, huber, quantile, L1) and the ranker predict raw either way |
| `predict(start_iteration=)` / `predict(num_iteration=)` | supported | A slice of the ensemble, LightGBM's pair, with LightGBM's clamping. `num_iteration=None` means every iteration the model kept. Not available for sparse input |
| `predict(pred_leaf=)` | supported | Leaf index per tree. Combining it with `raw_score` raises, where LightGBM silently lets one win. Not available for sparse input |
| `predict(pred_contrib=)` | supported | Exact TreeSHAP (path-dependent), LightGBM's shapes: `n_features + 1` columns with the expected value last, and `n_classes * (n_features + 1)` in class-major blocks for multiclass. Every row sums to its raw score. Combining it with `raw_score` or `pred_leaf` raises. Needs node covers, so a model saved in format v1 or v2 raises rather than guessing (`src/mojoboost/contrib.mojo`). Not available for sparse input, which refuses rather than densifying |
| `predict(validate_features=)` | supported | Same flag. A name mismatch raises either way, as scikit-learn already refuses to predict through one; the flag turns the one-sided cases from warnings into errors |
| `score(X, y)` | supported | R^2 for the regressor, accuracy for the classifier, mean NDCG for the ranker |
| `fit(eval_group=)` (ranker) | supported | Not a LightGBM `fit` argument name for the sklearn wrapper's ranker in this shape, but the same idea: each validation set's own query boundaries, required when a ranker is given an `eval_set` |

## 4. Fitted attributes

| LightGBM attribute | Status | Notes |
|---|---|---|
| `n_features_in_` | supported | |
| `feature_names_in_` | supported | Set when `X` carried string column names |
| `n_features_` | different | LightGBM's pre-scikit-learn spelling of `n_features_in_`. Not duplicated |
| `classes_` / `n_classes_` | supported | Classifier. Labels of any single comparable type, sorted |
| `feature_importances_` | supported | Both kinds computed at fit time. A model read back with `load()` reports zero gain importance and warns, because `src/mojoboost/serialize.mojo` does not write gains |
| `best_iteration_` | different | Always set, to the boosting iteration the model is used at. LightGBM sets it only when early stopping ran |
| `best_score_` | partial | A single float, the primary metric's best value on the first validation set. LightGBM's is a nested dict over sets and metrics; the rest of that grid is in `evals_result_` |
| `evals_result_` | supported | `{valid_name: {metric_name: [values]}}`. Index 0 is the base-score-only model, so entry `i` is the score after `i` trees; LightGBM starts at the first iteration |
| `booster_` | supported | The `Booster` holding the fitted model, and the only place the handle lives. It cannot `update()`: an estimator bins its own matrix and keeps no `Dataset` to grow on |
| `objective_` | deferred | The resolved objective name is not exposed; `objective` is echoed back as given. `mojoboost.inspection.objective_of(model)` reads it out of the model text meanwhile |
| `n_iter_` | supported | The number of boosting iterations that were trained. It differs from `best_iteration_` only when a validation metric peaked before the last round with early stopping off |
| `n_estimators_` | different | LightGBM's second name for the same number. `n_iter_` reports it; a third spelling is not added |
| `feature_name_` | different | `booster_.feature_name()` reports the training feature names, or LightGBM's `Column_0`, `Column_1`, ... when there were none. A second estimator attribute alongside `feature_names_in_` is not added |
| `categorical_feature_` (mojoboost) | different | Not a LightGBM attribute. The resolved categorical column indices, so `"auto"` can be inspected after the fact |
| `device_` (mojoboost) | different | Not a LightGBM attribute. Records which backend actually ran, because `device="auto"` makes that a runtime outcome |
| `stopped_early_` (mojoboost) | different | Not a LightGBM attribute. True when early stopping fired, which `best_iteration_ < n_estimators` alone does not distinguish from objective convergence |

## 5. Booster and Dataset APIs

`mojoboost.Booster` and `mojoboost.Dataset` are the functional API in
`python/mojoboost/basic.py`, over the Mojo `Dataset` and its trainers in
`src/mojoboost/trainset.mojo`. The estimators hold the same `Booster` on
`booster_`, so there is one model object in the package rather than one per
API, and `python/tests/test_basic.py` asserts that a model trained through
`train()` and the same model trained through an estimator predict
identically, value for value.

The rows that say `different` are where a LightGBM method conflicts with
owning data safely in Mojo rather than with a pointer into the caller's
memory. LightGBM's post-construction mutators are the main one: bin edges
are fitted from the data and from the categorical declaration, so changing
either afterwards would leave the binned matrix describing data the dataset
no longer holds. Every field is a constructor argument instead.

| LightGBM API group | Status | Notes |
|---|---|---|
| `Booster(params, train_set)`, `update`, `current_iteration` | supported | A booster starts at zero iterations and `update(n)` grows it, returning LightGBM's is-finished flag. 40 rounds then 60 more are the 100-round model, bit for bit (`tests/test_trainset.mojo`, `python/tests/test_basic.py`). `train(init_model=)` is the other door onto the same continuation. Ranking is the exception: LambdaRank gradients need per-query state the fitted ensemble does not carry, so a ranking booster raises rather than appending trees that would be wrong |
| `Booster(model_file=)` / `Booster(model_str=)` | supported | Reads back a single-output or softmax model; the file says which. A v4 file carries the split gains and the feature names, so a booster read back reports both. What a file still does not carry is the training parameters, so it asks for an explicit `eval` metric. One written before v4 reports `Column_i` names and zero gain importance |
| `Booster.rollback_one_iter` / `reset_parameter` | deferred | Truncating an ensemble and re-parameterizing a run in flight are both reachable in principle (the ensemble is a tree list); neither is implemented. With the callback work, task 3 |
| `Booster.predict` with all prediction modes | partial | `Booster.predict` covers response, raw score, and iteration ranges, with LightGBM's clamping rules. Leaf indices and feature contributions are on the estimators' `predict` (section 3) and still not on the Booster |
| `Booster.save_model` / `model_to_string` / `model_from_string` | different | Present under LightGBM's names, but the format is mojoboost's own versioned text (`src/mojoboost/serialize.mojo`, now at v4), which stores floats as raw bit patterns so a round trip predicts bit-exactly. It is not LightGBM-readable; `src/mojoboost/lgbm_model_io.mojo` is an unintegrated experiment in reading LightGBM's format, section 0. `save_model` takes no `num_iteration` or `start_iteration` |
| `Booster.dump_model` / `trees_to_dataframe` | supported | `Booster.dump_model()` and `Booster.trees_to_dataframe()` are methods, and both delegate to `mojoboost.inspection`, which builds a documented, versioned schema (`docs/MODEL_INSPECTION_SCHEMA.md`). Split gains are in the model text as of format v4, so every internal node reports a measured `split_gain` and the dump reports `has_split_gain: True`; a model read from an older file reports `None` and says so through the flag. `weight` stays None: mojoboost records a node's row cover, not a hessian sum. Section 0 |
| `Booster.feature_importance` | supported | Both `split` and `gain`, on the Booster and as `feature_importances_`. Gains are not in the model file, so a booster read back or unpickled reports zero gain importance |
| `Booster.num_feature` / `num_trees` / `num_model_per_iteration` | supported | Public methods on the Booster; `num_model_per_iteration` is the class count for a softmax model and 1 otherwise |
| `Booster.eval` / `eval_train` / `eval_valid` / `add_valid` | supported | Returns LightGBM's `(name, metric, value, is_higher_better)` tuples, weighted by the dataset's own weights. The metric defaults to the objective's own loss and the value comes from `src/mojoboost/metrics.mojo`, the same code `fit(eval_set=)` scores with, so the two APIs cannot drift |
| `Booster.feature_name` | supported | The training set's names, or LightGBM's `Column_0`, `Column_1`, ... when it had none |
| `Booster.get_split_value_histogram` | partial | `mojoboost.inspection.get_split_value_histogram(model, feature, bins=, as_frame=)` returns the histogram, and `split_values(model, feature)` the raw thresholds. Module functions, not `Booster` methods, and reachable only by importing the submodule. Section 0 |
| `Booster.lower_bound` / `upper_bound` | deferred | The ensemble's extreme leaf values. Model inspection, task 14, and not in the dump schema today |
| `Booster.get_leaf_output` / `set_leaf_output` / `shuffle_models` / `refit` | partial | Reading is supported and writing is refused explicitly. `mojoboost.inspection.leaf_outputs(model)` gives every leaf's value by the ordinal `predict(pred_leaf=True)` reports, and `model_editing_support()` reports the refusal with its reasons rather than leaving it to be found as a missing attribute (`MODEL_EDITING_SUPPORTED` is False, and `src/mojoboost/inspection.mojo` says the same natively, in `model_editing_status_json`). The reason: a fitted tree carries node covers, internal node values, and split gains computed at growth time; an arbitrary leaf edit falsifies its ancestors while leaving them in place, no check could tell an intentional edit from a corrupt one, and format v4 serializes both covers and gains, so the contradiction would outlive the session |
| `Booster.free_dataset` / `set_train_data_name` | different | A booster holds a reference to its `Dataset`, which is what keeps continued training possible; dropping it is the caller's to do by dropping the booster. The training set is named `training` in `eval_train`, as LightGBM names it, and the name is not settable |
| `Booster.set_network` / `free_network` | deferred | Distributed training, task 16. Section 0 |
| `Dataset` construction and `construct` | supported | Binning happens on `construct()` or on the first `train()` that uses the dataset, and every later run reuses it. `free_raw_data` defaults to False here, not True: evaluation predicts through the model rather than reading an internal score buffer, so `eval_train()` needs the raw matrix |
| `Dataset.create_valid` / `set_reference` | different | A validation set is an ordinary `Dataset`; mojoboost predicts it through the model's own mapper, so it does not need the training set's bin mappers. `reference=` is accepted and its binning parameters are checked, so a mismatched reference is reported rather than ignored |
| `Dataset.subset` | deferred | Row subsets of a constructed dataset. No longer blocking cross validation: `mojoboost.cv` slices the raw matrix and re-bins per fold on purpose, because slicing a constructed dataset would carry every fold's quantiles into every other fold |
| `Dataset.get_field` and the typed accessors (`label`, `weight`, `group`, `init_score`) | supported | All four are constructor arguments and all four read back, alongside `feature_name()` and `categorical_feature()`. `init_score` is training state, not model state: boosting starts from it and the fitted model predicts the trees alone (`tests/test_trainset.mojo`) |
| `Dataset.position` | deferred | Needs unbiased LambdaRank, task 12 |
| `Dataset.set_field` / `set_label` / `set_weight` / `set_group` / `set_init_score` / `set_categorical_feature` / `set_feature_name` | different | Not offered. A dataset is immutable once constructed, because its bin edges were fitted from the data and the categorical declaration it was built with; construct another dataset to change a field |
| `Dataset.get_data` | supported | Returns the matrix as it was passed in, or None once `free_raw_data` dropped it |
| `Dataset.num_data` / `num_feature` / `feature_num_bin` | partial | `num_data`, `num_feature`, and `num_bin` (the binning's own bin count) are there. LightGBM's per-feature `feature_num_bin` is not |
| `Dataset.save_binary` / `add_features_from` | deferred | A binary dataset format and column-wise dataset merging; neither is needed to train and both are real work |

## 6. Data inputs

| LightGBM input | Status | Notes |
|---|---|---|
| 2-D numpy array | supported | Converted to column-major float64 |
| pandas `DataFrame` | supported | Column names recorded as `feature_names_in_` |
| Python lists / sequences | supported | Works with numpy absent, which is also how the wheel is smoke-tested |
| `scipy.sparse` CSR | partial | `fit` and `predict` take any SciPy sparse matrix or array and keep it sparse: CSC to fit because histogram accumulation is feature-oriented, CSR to predict because prediction is row-oriented, converted without densifying and without mutating the caller's matrix. An implicit zero is the numerical value 0.0, matching LightGBM's default `zero_as_missing=false`, so a sparse fit equals the dense fit of the same matrix with the gaps filled with zeros. What raises rather than densifying behind your back: `device="gpu"`, a Python objective callable, `eval_set` and early stopping, ranking, `pred_leaf`, `pred_contrib`, and iteration slicing. Two loose ends recorded in "Known gaps": the scikit-learn tag still declares `input_tags.sparse` False, and scipy is not in the `pytest` pixi environment, so the Python-side sparse tests skip in CI. Section 0 |
| `scipy.sparse` CSC | partial | Same path, same limits |
| `Dataset` from a file path | unsupported | Part of the file-based configuration surface mojoboost does not implement. `cli/mojoboost` reads CSV, which is a different thing (section 13) |
| `Sequence` / batched construction | deferred | Task 10, with the `Dataset` object |
| pyarrow tables and arrays | deferred | Task 10. The `pytest` environment installs pyarrow and polars for tests that skip themselves today |
| polars frames | deferred | Task 10 |
| `datatable` frames | unsupported | LightGBM's own support is legacy; not worth matching |
| NaN as missing | supported | `use_missing`, LightGBM's parameter, on by default. A reserved bin, a per-node default direction, and the same direction at predict time. NaN is the missing marker in sparse input too, wherever it is stored |
| `zero_as_missing` | deferred | The parameter is not offered. Sparse input already fixes the semantics at LightGBM's default (an implicit zero is the number zero), so what is missing is the ability to ask for the other behavior |
| Infinities in `X` | different | Rejected. LightGBM's own scikit-learn wrapper validates with `force_all_finite="allow-nan"`, which permits infinities into the C++ binner; mojoboost refuses them rather than binning them as extreme finite values by accident |

## 7. Core parameters

All 141 canonical parameter names in LightGBM 4.7.0, from
`lightgbm.basic._ConfigAliases` as enumerated in the version 1 audit.
Aliases are omitted; section 2 lists the aliases mojoboost accepts.

### Core and objective

| Parameter | Status | Notes |
|---|---|---|
| `objective` | partial | Section 8 |
| `boosting` | partial | `gbdt`, `goss`. `dart` and `rf` deferred |
| `data_sample_strategy` | partial | `bagging` and `goss` are both implemented, selected through `boosting="goss"` (LightGBM 3.x spelling) rather than through this newer parameter |
| `num_iterations` | supported | Spelled `n_estimators` in Python, `BoosterParams.n_rounds` in Mojo |
| `learning_rate` | supported | |
| `num_leaves` | supported | |
| `tree_learner` | different | mojoboost has one grower. The distributed prototype is selected by calling `train_distributed`, not by a string |
| `num_threads` | different | `MOJOBOOST_NUM_WORKERS`, see section 2 |
| `device_type` | supported | Accepted as `device` with `device_type` as an alias. Values differ: mojoboost has `cpu`, `gpu`, `auto`; LightGBM has `cpu`, `gpu`, `cuda`. One portable GPU backend, so no vendor split, and `auto` is an addition. `gpu` now covers multiclass as well as single-output training (`gpu_supports` in `src/mojoboost/device.mojo` admits every output count); the module docstring above it still says multiclass is CPU only and is stale, which is recorded in "Known gaps" |
| `seed` | different | Per-sampler seeds, section 2 |
| `deterministic` | different | Always on. Determinism is a property of the implementation (counter-based sampling, fixed-point GPU histogram reduction), not a toggle |
| `num_class` | supported | Inferred from the labels in Python; an explicit argument to `train_multiclass`/`fit_multiclass` in Mojo |

### Learning control

| Parameter | Status | Notes |
|---|---|---|
| `max_depth` | supported | |
| `min_data_in_leaf` | supported | |
| `min_sum_hessian_in_leaf` | supported | Spelled `min_child_hess` |
| `bagging_fraction` | supported | |
| `pos_bagging_fraction` / `neg_bagging_fraction` | deferred | Class-conditional bagging for unbalanced binary data. `class_weight` covers the common case; sampling by class does not |
| `bagging_freq` | supported | |
| `bagging_seed` | supported | |
| `bagging_by_query` | different | Always on for the ranker. A half-sampled query would be normalized against a maxDCG no served ranking ever had, so mojoboost does not offer the row-sampling variant |
| `feature_fraction` | supported | |
| `feature_fraction_bynode` | supported | Applied on both backends: `src/mojoboost/train_gpu.mojo` calls the same `select_node_features` the CPU grower calls |
| `feature_fraction_seed` | supported | |
| `extra_trees` / `extra_seed` | partial | Randomized split thresholds: one drawn threshold per feature, keyed by (`extra_seed`, tree index, node, feature). `src/mojoboost/split.mojo` calls `extra_threshold_index` and `src/mojoboost/params.mojo` parses both keys, so the parameter string reaches it. Numerical thresholds only, and it raises for a categorical feature rather than ignoring it. No Python estimator parameter. Section 0 |
| `early_stopping_round` | supported | `fit(early_stopping_rounds=)`, and `train_with_valid` / `train_with_metrics` in Mojo |
| `early_stopping_min_delta` | supported | `fit(min_delta=)`. Same strict-improvement rule as LightGBM |
| `first_metric_only` | different | Every metric flagged for early stopping is watched, and the ensemble is truncated to the best round of the **primary** metric on the **first** validation set. LightGBM truncates to the best iteration of whichever pair ran out of patience first, which makes the kept model depend on scheduling. `mojoboost.cv` does take a `first_metric_only` argument, because a fold history has no primary-metric truncation to fall back on |
| `max_delta_step` | partial | Fixed at LightGBM's Poisson value (`poisson_max_delta_step`, 0.7) inside the Poisson objective. The general rule is live for every objective now: `finish_leaf_output` caps leaf outputs in `src/mojoboost/split.mojo` and `src/mojoboost/tree.mojo`, and `src/mojoboost/params.mojo` parses the key. `partial` because no Python estimator parameter reaches it. Section 0 |
| `lambda_l1` | supported | LightGBM's `ThresholdL1` soft thresholding, applied to split gains and leaf values |
| `lambda_l2` | supported | Default differs (1.0 here, 0.0 in LightGBM); documented in the README |
| `linear_lambda` | unsupported | Only meaningful with `linear_tree` |
| `min_gain_to_split` | partial | The gain floor is live: `src/mojoboost/split.mojo` rejects a candidate through `passes_min_gain` before it can win, and `src/mojoboost/params.mojo` parses both `min_gain_to_split` and LightGBM's `min_split_gain` alias. `partial` rather than `supported` because no Python estimator parameter reaches it, so only the Mojo API, the C ABI, and the CLI can set it. Section 0 |
| `drop_rate` / `max_drop` / `skip_drop` / `xgboost_dart_mode` / `uniform_drop` / `drop_seed` | deferred | DART parameters. DART itself is deferred, see `boosting` |
| `top_rate` / `other_rate` | supported | GOSS. Same defaults, same `\|grad * hess\|` importance, same warmup rule, with `goss_warmup_rounds` exposing the warmup count |
| `min_data_per_group` | supported | Categorical |
| `max_cat_threshold` | supported | Categorical |
| `cat_l2` | supported | Categorical |
| `cat_smooth` | supported | Categorical |
| `max_cat_to_onehot` | supported | Categorical |
| `top_k` | deferred | Voting-parallel only; with task 16 |
| `monotone_constraints` | supported | Per-feature -1/0/1. The guarantee holds at any feature value, not only on the training data |
| `monotone_constraints_method` | different | One method. LightGBM's `basic`/`intermediate`/`advanced` choice is an artifact of three implementations; mojoboost's bounds propagation is the exact one. `src/mojoboost/tree_parameters_extra.mojo` parses the parameter name for a future caller and nothing reads it |
| `monotone_penalty` | partial | Depth-scaled penalty on constrained splits, live through `apply_monotone_penalty` in `src/mojoboost/split.mojo` and parsed by `src/mojoboost/params.mojo`. No Python estimator parameter. Section 0 |
| `feature_contri` | deferred | Per-feature gain multipliers. The rule exists in `src/mojoboost/tree_parameters_extra.mojo` and no scan applies it; the comments around the per-feature scan in `src/mojoboost/split.mojo` name where it would be charged, which is not the same as charging it. Section 0 |
| `forcedsplits_filename` | unsupported | File-based configuration surface. `src/mojoboost/tree_parameters_extra.mojo` can parse the file's contents into a validated forced-split tree, but reading a file is not something mojoboost does |
| `refit_decay_rate` | deferred | With `refit`, task 14 |
| `cegb_tradeoff` / `cegb_penalty_split` / `cegb_penalty_feature_lazy` / `cegb_penalty_feature_coupled` | deferred | Cost-effective gradient boosting. All four now have an implementation in `src/mojoboost/cegb.mojo`, including the lazy per-feature penalty and the per-row read state it needs, and no root reaches that module. `src/mojoboost/params.mojo` parses `cegb_tradeoff` and `cegb_penalty_split` into a config field no grower reads, so setting either still does nothing. Section 0 |
| `path_smooth` | partial | Leaf-value smoothing toward the parent, live through `finish_leaf_output` in `src/mojoboost/split.mojo` and `src/mojoboost/tree.mojo`, and parsed by `src/mojoboost/params.mojo`. No Python estimator parameter. Section 0 |
| `interaction_constraints` | supported | LightGBM's per-branch allowed-feature rule, including the sharp edge that a feature in no group is never split on |
| `verbosity` | different | Training is silent. There is no logging layer to turn up |
| `input_model` / `output_model` / `saved_feature_importance_type` / `snapshot_freq` | unsupported | File-based configuration surface. `save()`/`load()` cover the model itself, and `cli/mojoboost` has its own `--model` flag |
| `use_quantized_grad` / `num_grad_quant_bins` / `quant_train_renew_leaf` / `stochastic_rounding` | deferred | Quantized-gradient training. Interesting for the GPU path; nothing depends on it today |

### Dataset and IO

| Parameter | Status | Notes |
|---|---|---|
| `max_bin` | supported | Default 255 |
| `max_bin_by_feature` | deferred | Per-feature bin counts. Straightforward once the binner takes a vector; low demand |
| `min_data_in_bin` | partial | Enforced for categorical features (`min_data_per_group` governs the sorted search); the numerical binner has no minimum-population rule |
| `data_random_seed` | different | Binning is deterministic and reads every row, so there is no sampling seed to set |
| `bin_construct_sample_cnt` | deferred | See `subsample_for_bin` in section 2 |
| `is_enable_sparse` | different | There is no toggle: the input type decides. A SciPy sparse matrix takes the sparse path (section 6) and a dense matrix takes the dense one, and neither converts to the other |
| `enable_bundle` | partial | Exclusive Feature Bundling. `src/mojoboost/boosting.mojo` fits one plan per training run with `prepare_bundling` and hands the bundled matrix to every tree, and `src/mojoboost/params.mojo` parses `enable_bundle` and `max_conflict_rate`. No Python estimator parameter reaches it. Section 0 |
| `use_missing` | supported | |
| `zero_as_missing` | deferred | Section 6 |
| `feature_pre_filter` | different | mojoboost does not drop features during binning, so there is nothing to disable. A constant feature simply never yields a positive-gain split |
| `pre_partition` | deferred | Distributed, task 16 |
| `two_round` / `header` / `label_column` / `weight_column` / `group_column` / `ignore_column` / `parser_config_file` / `precise_float_parser` / `forcedbins_filename` / `save_binary` | unsupported | LightGBM's text-parsing parameters. mojoboost's own CLI has `--header`, `--label`, and `--weight` flags over its own CSV reader (`cli/README.md`); they are not these parameters and are not accepted in a parameter string |
| `categorical_feature` | supported | Indices, names, or `"auto"`. pandas `category` columns are label-encoded by the estimator, and the encoding is kept for prediction; the model file carries the category tables but not the labels, so a model read back from disk takes integer codes |
| `data` / `valid` / `config` / `task` / `convert_model` / `convert_model_language` / `output_result` | unsupported | File-based configuration surface |
| `histogram_pool_size` | different | Histogram memory is pooled per grower without a user cap |

### Objective-specific

| Parameter | Status | Notes |
|---|---|---|
| `boost_from_average` | different | Always on for the built-in objectives, always off for custom ones (the framework does not know the link). LightGBM makes it a toggle. A custom objective sets its starting raw score explicitly with `base_score=`, a number or `"mean"` |
| `is_unbalance` / `scale_pos_weight` | partial | Both are in `src/mojoboost/class_weight.mojo` under their LightGBM names (`unbalanced_sample_weight`, `scale_pos_weight_rows`). Not constructor parameters on the Python classifier, where `class_weight={1: w}` is `scale_pos_weight` and `class_weight="balanced"` is `is_unbalance` up to a constant factor |
| `sigmoid` | partial | Supported for LambdaRank, as a `MojoBoostRanker` constructor parameter. Not exposed for the binary objective, which uses the standard logistic |
| `alpha` | supported | Huber transition point and quantile level, LightGBM's meanings |
| `fair_c` | supported | The fair loss's `c`, default 1.0. One trainer slot holds whichever scalar parameter the objective reads, and naming one that belongs to a different objective is an error rather than a silently ignored value |
| `poisson_max_delta_step` | different | Fixed at LightGBM's default 0.7 rather than exposed |
| `tweedie_variance_power` | supported | Tweedie's rho, in (1, 2), default 1.5. Outside that range it would no longer be the compound Poisson-gamma the objective assumes, so it is rejected rather than clamped |
| `lambdarank_truncation_level` | supported | Default 30, and also the maxDCG cutoff, as in LightGBM |
| `lambdarank_norm` | supported | Default on |
| `label_gain` | different | Fixed at LightGBM's default `2^i - 1` for labels 0..30, which is also why labels outside that range are rejected. A user-supplied gain vector is deferred |
| `lambdarank_position_bias_regularization` | deferred | Part of unbiased LambdaRank, which is out of v1 |
| `objective_seed` | different | No objective in mojoboost draws random numbers |
| `reg_sqrt` | deferred | `sqrt`-transformed regression. Rare |
| `multi_error_top_k` / `auc_mu_weights` | deferred | With top-k multiclass error and `auc_mu`, neither of which is implemented |

### Metric

| Parameter | Status | Notes |
|---|---|---|
| `metric` | supported | Section 9. `fit(eval_metric=)` in Python, `MetricSuite` in Mojo, `metrics=` in `mojoboost.cv` |
| `metric_freq` / `is_provide_training_metric` | different | Metrics are evaluated every round on the validation sets only; the training set is not scored automatically. `mojoboost.cv(eval_train_metric=True)` is the one place a training score is offered |
| `eval_at` | partial | `ndcg_eval_at`, a single cutoff for the ranker's `score` and for its `ndcg`/`map` eval metrics. LightGBM takes a list; `ndcg_score` takes any cutoff, and `ndcg_at_cutoffs` in Mojo takes several at once |

### Network, GPU, and prediction

| Parameter | Status | Notes |
|---|---|---|
| `num_machines` / `local_listen_port` / `time_out` / `machine_list_filename` / `machines` | deferred | No transport is wired up. `src/mojoboost/collective.mojo` defines the contract a transport would implement, `src/mojoboost/distributed_transport.mojo` is an unintegrated implementation of one (section 0), and `docs/distributed.md` states what is and is not built. Task 16 |
| `gpu_platform_id` / `gpu_device_id` / `gpu_device_id_list` / `gpu_use_dp` / `num_gpu` | different | One portable backend, one device, Float64 host arithmetic with a fixed-point device reduction. There is no OpenCL platform to select, and no double-precision toggle because the reduction is integer |
| `force_col_wise` / `force_row_wise` | different | mojoboost builds histograms feature-major, always. The choice exists in LightGBM to trade off multi-threading strategies; here the parallel dispatch is governed by `MOJOBOOST_PARALLEL_MIN_OPS` |
| `predict_raw_score` | supported | Section 3 |
| `predict_leaf_index` | supported | Section 3 |
| `predict_contrib` | supported | Section 3. Dense input only |
| `num_iteration_predict` / `start_iteration_predict` | supported | Section 3 |
| `pred_early_stop` / `pred_early_stop_freq` / `pred_early_stop_margin` | unsupported | Early-exit prediction trades accuracy for latency with no error bound. Not planned |
| `predict_disable_shape_check` | different | Shape and feature-name checks are always on. Disabling them turns a caught error into a silently wrong prediction |

## 8. Objectives

Every objective name accepted by LightGBM 4.7.0, verified in the version 1
audit by training a 1-round model with each name against the pinned
install.

| LightGBM objective | Status | mojoboost | Evidence |
|---|---|---|---|
| `regression` (l2) | supported | `SQUARED_ERROR`, Python `objective="regression"` | `src/mojoboost/boosting.mojo`, `tests/test_mojoboost.mojo` |
| `regression_l1` / `mae` | supported | `L1`, with LightGBM's `RenewTreeOutput` leaf-value replacement | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `huber` | supported | `HUBER`, `alpha` is the transition point. No leaf renewal, as in LightGBM | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `quantile` | supported | `QUANTILE`, `alpha` is the level, with weighted-percentile leaf renewal | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `poisson` | supported | `POISSON`, exp link, log-mean base score, `poisson_max_delta_step` in the Hessian. Python `objective="poisson"` | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `binary` | supported | `BINARY_LOGISTIC` | `src/mojoboost/boosting.mojo`, `tests/test_mojoboost.mojo` |
| `multiclass` (softmax) | supported | `train_multiclass` / `fit_multiclass`, one tree per class per round, on either backend | `src/mojoboost/boosting.mojo`, `tests/test_multiclass_model.mojo` |
| `lambdarank` | supported | `train_ranker` / `fit_ranker` / `MojoBoostRanker` | `src/mojoboost/ranking.mojo`, `tests/test_ranking.mojo` |
| custom (callable) | different | Single output only, weights applied by the framework, gradients validated every round, base score explicit through `base_score=`. CPU only from Python: the `fit_custom` binding takes no device, and `train_custom_gpu` is the Mojo entry point for a pre-binned matrix. See the README section on custom objectives for each difference and why | `src/mojoboost/objective.mojo`, `tests/test_custom_objective.mojo` |
| `mape` | supported | `MAPE`, gradient scaled by LightGBM's `1 / max(1, \|y\|)` label weight, median leaf renewal under those same weights. Python `objective="mape"` | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `fair` | supported | `FAIR`, with `fair_c` (the trainer's `alpha` slot). Python `objective="fair", fair_c=...` | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `gamma` | supported | `GAMMA`, exp link, log-mean base score, strictly positive labels required | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `tweedie` | supported | `TWEEDIE`, exp link, with `tweedie_variance_power` in (1, 2) (the trainer's `alpha` slot). Python `objective="tweedie", tweedie_variance_power=...` | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `cross_entropy` | supported | `CROSS_ENTROPY` (alias `xentropy`), logistic link with labels anywhere in [0, 1]. On the regressor, since its labels are soft targets rather than classes | `src/mojoboost/boosting.mojo`, `tests/test_objectives.mojo` |
| `cross_entropy_lambda` | different | Not implemented, and reported by name rather than as an unknown objective. It parameterizes the rate through `log1p(exp(raw))`, a different link, so it is not an alias of `cross_entropy` and cannot be reached by setting one | `src/mojoboost/params.mojo`, `python/mojoboost/__init__.py` |
| `multiclassova` | different | Not implemented, reported by name. One-vs-rest needs an independent binary model per class, which is a different trainer from the shared-softmax `multiclass` | `src/mojoboost/params.mojo`, `python/mojoboost/__init__.py` |
| `rank_xendcg` | different | Not implemented, reported by name. Out of v1 with the rest of the unbiased/alternative ranking objectives; `lambdarank` is the ranking objective provided | `src/mojoboost/params.mojo`, `python/mojoboost/__init__.py` |

GPU coverage: `train_gpu` covers every single-output built-in objective
above that shares the per-row gradient/Hessian interface,
`train_multiclass_gpu` covers softmax and is reached by `fit_multiclass`
and therefore by `MojoBoostClassifier(device="gpu")`, and
`train_custom_gpu` covers custom objectives from Mojo only. LambdaRank is
CPU only. Every GPU claim here has run on one device class, Apple M4; see
`docs/GPU_VALIDATION.md` and section 12.

## 9. Metrics

Every metric name accepted by LightGBM 4.7.0, verified the same way in the
version 1 audit.

| LightGBM metric | Status | Notes |
|---|---|---|
| `l2` | supported | Mojo `rmse` reports the root; the squared value is the training loss used for early stopping |
| `rmse` | supported | `src/mojoboost/metrics.mojo`, `tests/test_metrics.mojo` |
| `l1` | supported | Same file |
| `quantile` / `huber` | supported | Read the estimator's `alpha`, so they score the loss the objective trained on |
| `mape` | supported | LightGBM's `\|y - p\| / max(1, \|y\|)`, the same label weight the MAPE objective trains against |
| `fair` | supported | Reads `fair_c` |
| `poisson` | supported | `mu - y log mu` on the response scale |
| `gamma` / `gamma_deviance` | supported | The gamma likelihood and its deviance, which is 0 at a perfect prediction |
| `tweedie` | supported | Reads `tweedie_variance_power`; scoring at a different rho scores a different loss |
| `cross_entropy` / `kullback_leibler` | supported | Continuous labels in [0, 1]. KL is the cross entropy minus the labels' own entropy, so a perfect prediction scores 0 |
| `binary_logloss` | supported | `src/mojoboost/metrics.mojo` |
| `multi_logloss` | supported | Same file |
| `binary_error` / `multi_error` | supported | Under LightGBM's spellings, with `binary_accuracy` and `multiclass_accuracy` as their Mojo-only complements |
| `auc` | supported | Rank-based, with scikit-learn's tie handling |
| `average_precision` | supported | Step-wise precision-recall area, scikit-learn's rule for ties (no trapezoid interpolation) |
| `ndcg` | supported | Any cutoff, per query, averaged; `ndcg_score` from Python. An all-zero-label query counts as 1.0, as in LightGBM |
| `map` | supported | Binary relevance (any label above 0), AP@k divided by `min(k, relevant)`, and a query with nothing relevant counts as 1.0, matching this module's NDCG convention. `src/mojoboost/ranking.mojo` |
| `cross_entropy_lambda` | deferred | With the objective of that name, which is not implemented |
| `auc_mu` | deferred | The multiclass AUC generalization; needs the class-pair projection LightGBM builds, and no multiclass ranking metric is provided yet |
| custom metrics (`feval`) | supported | Mojo: `MetricSuite`, several metrics with a declared direction and early-stopping flag. Python: `fit(eval_metric=...)` with callables, and `feval=` in `mojoboost.cv`. Differences from `feval` are listed at the top of `src/mojoboost/custom_metric.mojo` |

Every metric marked `supported` above is selectable from Python as
`eval_metric="auc"` and scored by the same Mojo functions the Mojo API
exposes; the table of names, aliases, directions, and tasks is
`python/mojoboost/_eval.py`, mirrored by the metric codes in
`bindings/_mojoboost.mojo`. Two differences from LightGBM are deliberate: a
metric that cannot mean anything for the model being fitted is rejected
rather than scored (the regressor takes the regression metrics, the
classifier the binary or multiclass ones, the ranker `ndcg` and `map`), and
predictions are transformed by the *objective's* inverse link exactly once
before any metric sees them.

## 10. Callbacks

| LightGBM callback | Status | Notes |
|---|---|---|
| callback protocol (`CallbackEnv`, ordering, `before_iteration`) | supported | Same namedtuple fields, same `order`/`before_iteration` attributes, same two-phase split. One boundary crossing per phase per round, benchmarked in `bench/bench_callbacks.py`. Regressor and binary classifier only: the softmax and LambdaRank trainers refuse a callback list rather than ignoring it |
| `early_stopping` | supported | As a callback and as `fit` arguments, section 1 |
| `log_evaluation` | supported | `period<=0` silences it |
| `record_evaluation` | supported | Also `evals_result_` without a callback |
| `reset_parameter` | partial | The nine hyperparameters the loop re-reads each round; see section 1 |
| `EarlyStopException` | supported | Stops the run and rolls back to the best round |
| callbacks in `cv` | partial | `mojoboost.cv(callbacks=)` runs them per fold, with the same reachability caveat as `cv` itself (section 0) |

## 11. Distributed modes

| LightGBM mode | Status | Notes |
|---|---|---|
| data parallel | partial | Designed and prototyped: row partitioning, local histograms, all-reduce, globally consistent splits, identical trees on every rank, deterministic failure agreement. Every rank runs in one process (`LocalCollective`), so nothing has crossed a network. `src/mojoboost/distributed.mojo`, `src/mojoboost/collective.mojo`, `tests/test_distributed.mojo`, `docs/distributed.md` |
| feature parallel | deferred | Section 2 of `docs/distributed.md` explains why data parallel comes first |
| voting parallel | deferred | Same |
| a real transport (MPI, sockets, gRPC) | deferred | `src/mojoboost/distributed_transport.mojo` implements a session state machine and its own suite, and nothing imports it: `src/mojoboost/distributed.mojo` still takes a `Collective`, and `src/mojoboost/__init__.mojo` does not export the transport. Section 0, task 16. **No distributed performance or scaling claim is made anywhere** |
| Dask integration | deferred | `python/mojoboost/dask.py` is the client-side contract and cannot train; section 0, task 17 |
| distributed GPU | deferred | Out of v1 |

## 12. GPU and accelerators

Every row here has run on exactly one device class, Apple M4 with Metal.
`docs/GPU_VALIDATION.md` holds the record and the procedure for CUDA and
HIP, and no NVIDIA or AMD device has run this code.

| Capability | Status | Notes |
|---|---|---|
| GPU histogram construction | supported | Portable kernel, 2D grid, fixed-point Int32 reduction, two combine strategies that are bit-identical to each other. `src/mojoboost/histogram_gpu.mojo`, `tests/test_gpu_strategies.mojo` |
| End-to-end GPU training | supported | `train_gpu`, device-resident binned matrix and compacted per-leaf row ranges. `src/mojoboost/train_gpu.mojo`, `src/mojoboost/gpu_active_rows.mojo`, `tests/test_gpu_training.mojo` |
| Device-resident objectives | supported | Built-in objectives without row sampling generate gradients and advance raw scores on the device, so nothing per-row crosses the host boundary in a plain round; bagging and GOSS keep the host path because their samples are ranked host-side. `src/mojoboost/gpu_objectives_native.mojo`, `tests/test_gpu_objectives.mojo` |
| GPU split selection | supported | Off by default. `MOJOBOOST_GPU_SPLIT_STRATEGY=device` (or `grow_tree_gpu(split_search=SPLIT_SEARCH_DEVICE)`) searches each node's histogram on the device and downloads one 136-byte record instead of the histogram. It keeps a histogram slot per live leaf, so a split builds only its smaller child and derives the sibling with a device subtraction, and searches both children in one launch pair: one histogram build, one wait, and 272 bytes per split. Measured on an M4 with `bench-train-gpu`, 100 trees, four CPU workers: at 250000 x 100 it trains in 3.15 s against the host scan's 3.22 s, and at 50000 x 100 in 2.85 s against 2.43 s, so it wins where the accumulation dominates and loses where the fixed per-node search cost does. Float32 gains can flip near-tie split decisions versus the host scan, and a few percent does not pay for that, so `SPLIT_SEARCH_AUTO` still resolves to the host scan. `MOJOBOOST_GPU_SPLIT_RESIDENT=0` forces the older loop, which builds both children and waits twice per split (5.40 s on the same run). `src/mojoboost/gpu_split_search.mojo` |
| GPU multiclass | supported | `fit_multiclass` resolves the device and calls `train_multiclass_gpu`, and `gpu_supports` admits every output count, so `MojoBoostClassifier(device="gpu")` trains one tree per class per round on the accelerator. `src/mojoboost/model.mojo`, `tests/test_device.mojo`, `tests/test_gpu_objectives.mojo` |
| GPU prediction | partial | `Model.predict_batch` and `MulticlassModel.predict_batch` take a `device` and walk the trees on the accelerator; binning stays host-side, so both devices route every row to the same leaf. Mojo API only: the `predict` binding takes no device, so the Python estimators always predict on the CPU. Not a LightGBM capability (`device_type` covers training only), so this is a mojoboost addition. `src/mojoboost/gpu_predict.mojo`, `tests/parallel/test_gpu_predict.mojo` |
| Persistent GPU session and scheduling | partial | `src/mojoboost/gpu_runtime.mojo` is exported and `src/mojoboost/histogram_gpu.mojo` borrows a `GpuSession`'s device context. The residency ledger, staging ring, and phase counters around it are exercised by `tests/parallel/test_gpu_runtime.mojo` rather than by a trainer |
| Apple-specific tiling policy | partial | `src/mojoboost/apple_gpu_policy.mojo` is read by `src/mojoboost/device_policy.mojo` (device profile, session memory estimate) and by `src/mojoboost/apple_histogram_policy.mojo` (block threads). It still decides no default launch geometry: the specialization level resolves to `SPEC_LEVEL_BASELINE` unless `MOJOBOOST_GPU_HIST_SPECIALIZATION` asks otherwise, so `src/mojoboost/gpu_tiling.mojo` remains the policy in force. Section 0 |
| Sparse input on the GPU | deferred | `_sparse_fit_params` raises for `device="gpu"` rather than densifying, so the refusal a user meets is accurate. Sparse GPU kernels do exist as of this revision, in `src/mojoboost/gpu_sparse.mojo` and `src/mojoboost/gpu_sparse_layout.mojo`, and no entry point reaches either. Downgraded from `unsupported` for that reason: the reason not to build it no longer holds, because it has been built. Section 0 |
| CUDA (NVIDIA) validation | deferred | The source targets it and `tests/test_gpu_portability.mojo` pins the launch limits CUDA imposes, but **no NVIDIA device has run this code**. `.github/workflows/gpu-validation.yml` is the manual job that would produce a record |
| HIP (AMD) validation | deferred | Same |
| GPU speed vs CPU | different | `auto` ships with its size heuristic disabled and always chooses the CPU, because no benchmark on any device has found a crossover. `MOJOBOOST_AUTO_MIN_CELLS` is the knob for running that benchmark. Shipping a threshold first would be a performance claim with nothing behind it |

## 13. Packaging and distribution

| LightGBM property | Status | Notes |
|---|---|---|
| PyPI wheels | deferred | Nothing has been uploaded to PyPI. The release workflows can publish to TestPyPI, behind a manual dispatch input and a repository variable, which is a rehearsal rather than a release |
| macOS arm64 wheel | partial | `packaging/build_wheel.sh` bundles the four MAX runtime dylibs with an `@loader_path` rpath and re-signs them, `packaging/test_wheel.sh` installs and smoke-tests the result, and `packaging/macos/build_release_wheel.sh` runs under a tag-triggered release workflow that also clean-installs and hashes the artifact. Nothing guards any of it on an ordinary change: the everyday CI workflow has no wheel job, and `docs/PLATFORM_MATRIX.md` records the target as `designed`, with an explicit note that a wheel found on a disk is not a record. Downgraded from `supported` in contract version 2 for that reason; restoring it needs a per-change job or a recorded clean-install run cited from the matrix, not a rewording |
| macOS x86-64 wheel | unsupported | `docs/PLATFORM_MATRIX.md` lists `macos-x86_64` as out of scope |
| Linux wheel, x86-64 and ARM64 | partial | `packaging/linux/build_wheel_linux.sh` builds both, with ELF inspection and a wheel-metadata check, under a release workflow with a runner per architecture. The default tag policy is plain `linux_x86_64` / `linux_aarch64`. Same gap as the macOS row: no per-change job, and `docs/PLATFORM_MATRIX.md` still says `designed` |
| manylinux wheel | deferred | The manylinux tag is a glibc promise, and the Linux release workflow makes promoting to it a deliberate input with a floor you have to have measured, rather than the default. Nothing has measured it |
| Windows wheel | deferred | Task 18 |
| Python version range | different | `requires-python = ">=3.14"`, one interpreter, because MAX 26.5.0 ships as a 3.14 build and pins `python 3.14.*`. LightGBM ships 3.9 through 3.13 |
| conda package | deferred | Task 18 |
| R package | deferred | Not started in this repository |
| C API | partial | `capi/mojoboost.h` is a stable, versioned C ABI over dense training, prediction, save, load, and the three model accessors, with an error object and an ownership table (`capi/README.md`). It is mojoboost's own interface and deliberately narrower than LightGBM's `c_api.h`, which it does not implement and is not source compatible with. `tests/test_capi.mojo` runs in CI; the C-side test (`pixi run test-c`) does not, and no release artifact carries the library |
| command line tool | different | `cli/mojoboost` trains and scores CSV files over the Mojo API, with the same parameter string the C ABI takes. It is not LightGBM's config-file surface (`config`, `task`, `data`, `valid`), which stays `unsupported` in section 7, and it reads no configuration file. `cli/README.md`, `tests/test_cli.mojo` |
| source build from a clean checkout | supported | `pixi install && pixi run test`, run by `.github/workflows/ci.yml` on x86-64 and ARM64 Linux |

---

## Known gaps in this contract

Findings from the audit that are not LightGBM parity items but that make the
matrix less trustworthy than it looks. They are recorded here rather than
quietly fixed, because each is somebody's in-flight work:

1. **The scikit-learn sparse tag contradicts the sparse path.**
   `python/mojoboost/_sklearn.py` reports `input_tags.sparse` False, and
   `python/tests/test_sklearn_integration.py` asserts that it does, while
   `fit` and `predict` accept SciPy sparse matrices. A scikit-learn utility
   that respects the tag will densify before calling mojoboost, which is
   the thing the sparse path exists to avoid.
2. **The Python sparse tests cannot run in the environment CI uses.**
   `python/tests/test_validation.py` and `python/tests/test_contrib.py`
   guard their sparse cases with `pytest.importorskip("scipy.sparse")`, and
   `pixi.toml` does not list scipy under `[feature.pytest.dependencies]`.
   Both cases skip in `pixi run -e pytest test-estimators`, which is the
   only Python test run CI performs, so section 6's rows are `partial`
   rather than `supported`.
3. **`colsample_bytree` and `colsample_bynode` are not accepted** as aliases
   even though the rest of the scikit-learn spellings are. Re-confirmed
   against `_Base.__init__` in this audit.
4. **Six modules are implemented, individually tested, and wired to
   nothing.** `src/mojoboost/efb.mojo`, `src/mojoboost/inspection.mojo`, `src/mojoboost/lgbm_model_io.mojo`,
   `src/mojoboost/distributed_transport.mojo`, `src/mojoboost/apple_gpu_policy.mojo`, and
   `src/mojoboost/tree_parameters_extra.mojo` have no importer in `src/mojoboost/` and no
   export from `src/mojoboost/__init__.mojo`. Their suites run in
   `pixi run test`, which makes them look supported from the test output
   alone. Section 0 scores each one.
5. **Five Python modules are reachable only by an import the compatibility
   policy calls private.** `python/mojoboost/cv.py`,
   `python/mojoboost/inspection.py`, `python/mojoboost/dask.py`,
   `python/mojoboost/device_selection.py`, and
   `python/mojoboost/diagnostics.py` work when imported by path, and
   section 2 of `docs/COMPATIBILITY_POLICY.md` says importing a `mojoboost`
   submodule other than `basic` is not public. Either the names move into
   `mojoboost.__all__` or the policy grows an exception; until one of those
   happens their rows stay `partial` or `deferred`.
   `python/mojoboost/_public_api_plan.py` is a lane's written proposal for
   that export block. It is data, nothing imports it, and no name has moved
   yet, which is why the rows above still read as they do.
6. **Two docstrings outrank their code and are wrong.** The module
   docstring of `src/mojoboost/device.mojo` says multiclass is CPU only,
   which `gpu_supports` in the same file contradicts. The
   `prediction options` section of `python/mojoboost/__init__.py` says
   "There are no built-in validation metrics in the Python API yet", which
   the `validation sets and early stopping` section of the same docstring
   contradicts and `python/mojoboost/_eval.py` refutes. Both are in files
   this audit does not own.
7. **Split gains are absent from every dump**, because
   `src/mojoboost/serialize.mojo` does not write them and
   `bindings/_mojoboost.mojo` exposes no `split_gains` hook. Anything built
   on the dump inherits that hole, including `trees_to_dataframe`.
8. **The comparisons against LightGBM print, they do not assert.**
   `bench/compare_missing_lightgbm.py`, `bench/compare_categorical_lightgbm.py`,
   and `bench/compare_ranking.py` put the two libraries side by side for a
   human to read, in the `bench` pixi environment, and no CI job runs them.
   Where this file says `differential-tested`, the evidence is an in-repo
   reference implementation (subset enumeration for contributions, a serial
   reference for histograms), not a LightGBM run.

## Enforcement

`tools/check_parity.py` (stdlib only, no build required) checks:

- every status cell in this file uses the vocabulary above
- every repository path cited in this file exists
- a fixed inventory of rows still says `supported`, so a supported row
  cannot be deleted or downgraded without the check failing
- the public Python symbols those rows depend on still exist, by importing
  `mojoboost` when the extension is built and by parsing
  `python/mojoboost/__init__.py` otherwise
- the public Mojo symbols those rows depend on are still exported from
  `src/mojoboost/__init__.mojo`
- **that a `deferred` or `unsupported` row has not gone stale**, by
  resolving the public symbols behind a watched row. A watch is a list of
  public names (a `__all__` entry, a class method, an argument, a Mojo
  export), never a file path, because a file existing is not support. When
  every name behind a watched row resolves, the check fails and asks for a
  re-audit of that row
- the section 0 level table uses exactly the seven level names defined in
  `docs/CAPABILITY_LEVELS.md`, that its cells are `yes`, `no`, or `n/a`,
  and that no capability marked publicly reachable is also `deferred` or
  `unsupported`
- **that section 0's `publicly reachable` cells are true, in both
  directions**, for the rows one public Python name decides. The watch
  above only fires on `deferred` and `unsupported` rows, so a `partial` row
  could keep saying a capability was out of reach long after its name
  landed in `mojoboost.__all__`, which is exactly what happened to `cv`,
  `inspection`, and `device_selection`. Reachability is not a judgment, so
  this check decides rather than asking
- every Mojo test suite cited here is run by a pixi task. A suite that is
  cited as evidence but never executed is not evidence; the exception list
  in the script must match the "Known gaps" section exactly, and it is
  currently empty

Run it with `python3 tools/check_parity.py` or `pixi run check-parity`. It
runs in CI on every push and pull request.

`tools/check_parity.py` checks *claims*. It does not compute reachability, and it
cannot tell you that a module in `src/mojoboost/` is connected to anything.
Two other scripts do, and the `integrated` column of section 0 rests on
them:

- `tools/connectivity_audit.py` builds the import graph from the four
  shipping roots and reports orphan modules, unused imports, duplicate
  registries, public parameters with no consumer, and native names Python
  reaches for that no binding exports.
- `tools/audit_integration.py` checks `docs/INTEGRATION_INVENTORY.md`
  against that graph, so the written inventory of disconnections cannot
  drift away from the tree.

Neither imports the package or builds anything, and neither re-checks this
contract. `docs/ARCHITECTURE.md` is the map the three of them describe from
different angles.

## Changelog

- **v3 (2026-08-14)**: re-audited reachability against the tree at
  `860b1cf`, by tracing imports from the four shipping roots rather than by
  reading prose. Three rows had rotted in the direction that matters least
  to a maintainer and most to a user, claiming a capability was out of
  reach after its name landed in `mojoboost.__all__`: `cv`,
  `inspection`, and `device_selection` are all publicly reachable and all
  three tables said they were not. Upgraded on new call sites found in the
  tree: exclusive feature bundling, distributed transport, the
  tree-parameter rules (`min_gain_to_split`, `max_delta_step`,
  `path_smooth`, `extra_trees`, `monotone_penalty`, forced splits),
  per-level feature sampling, and the Apple GPU tuning policy. Downgraded
  from `unsupported` to `deferred`: sparse input on the GPU, because
  kernels for it now exist and are unreachable, which is a different
  statement from absent. Added seven section 0 rows for capability families
  that landed disconnected (packed-bin layout, level-wise GPU growth,
  class-batched multiclass, hybrid leaf placement, GPU sparse and
  categorical kernels, DART and random forest, CEGB). Added
  `docs/ARCHITECTURE.md`, `docs/INTEGRATION_INVENTORY.md`, and
  `tools/audit_integration.py`, and a check that section 0's `publicly
  reachable` cells cannot rot the way those three did. Re-derived a second
  time against `63aad82` and its successors before publishing, because
  seventy-odd files landed while version 3 was being written, and the
  closure had moved
  underneath four of the rows the same pass had just added. Model
  inspection and explainable device selection stopped being blocked by
  unregistered bindings; `bindings/_mojoboost.mojo` now registers the
  inspection, objective-registry, dataset, distributed, and `decide_device`
  entry points, so the Python fallbacks are compatibility paths for an
  older extension rather than the path. Class-batched GPU multiclass moved
  from `deferred` to `partial` on a real call site behind
  `MOJOBOOST_GPU_CLASS_BATCH`. Packed-bin layout and hybrid leaf placement
  stayed `deferred` and gained evidence saying what of them is now reached,
  because an import is not a call and a report is not a placement.
- **v2 (2026-08-14)**: re-audited mojoboost's side against the tree at
  `ab25ad1`. Added section 0 and `docs/CAPABILITY_LEVELS.md`, which split
  "supported" into seven separate facts. Upgraded: sparse CSR/CSC input,
  `fit(eval_metric=)`, `fit(eval_sample_weight=)`, `cv`/`CVBooster`,
  `dump_model`/`trees_to_dataframe`/`get_split_value_histogram`, `n_iter_`,
  GPU multiclass, and the C API. Downgraded: the macOS arm64 wheel, on the
  evidence of `docs/PLATFORM_MATRIX.md` and the absence of a per-change
  wheel job, and GPU prediction, which was already documented as Mojo only.
  Added rows for the command line tool, the Linux wheel, sparse-on-GPU, the
  persistent GPU session, and the Apple tiling policy. Recorded six modules
  that are implemented and unintegrated, and two docstrings that contradict
  their own code.
- **v1 (2026-08-14)**: first audit, against LightGBM 4.7.0 and mojoboost at
  `6190f88`.
