# LightGBM parity contract

Contract version: 3
Audited against: **LightGBM 4.7.0** (the version pinned in the `bench` pixi
environment; `lightgbm.__version__` reported `4.7.0` when version 1 of this
file was written)
Audited: 2026-08-14, mojotrees at commit `29d76e4` plus the working tree of
that day. The import closure was re-derived at `63aad82` and re-checked at
`9c1e771` and `29d76e4`; it is unchanged across all three, because every
edge added in between either sits inside a module that was already
reachable or points into one that was already an orphan

What version 3 re-derived: reachability. Every `integrated` and `publicly
reachable` cell was recomputed by tracing imports from the four shipping
roots (`bindings/_mojotrees.mojo`, `src/mojotrees/__init__.mojo`,
`capi/mojotrees_capi.mojo`, `cli/mojotrees_cli.mojo`) and by reading
`mojotrees.__all__` and the `def_function` table, rather than by trusting
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

This file is the authoritative statement of what mojotrees does and does
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
| `different` | Deliberately not LightGBM's design. The notes say why, and what mojotrees does instead |
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
integration built on them. Unbiased LambdaRank, DART and random-forest
boosting, linear trees, and CEGB were on this list until the integration
round of 2026-08-15 wired each of them through to the estimators; their
rows below say what each still lacks (mostly a LightGBM differential).

**Never.** LightGBM's file-based configuration surface (`config`, `data`,
`valid`, `output_model`, `task`, `convert_model`, and the parameters that
only feed it), its deprecated aliases, and any parameter whose only purpose
is to select between implementations mojotrees does not have. mojotrees has
its own command line tool, which is not that surface; see section 13.

---

## 0. Capability levels for the contested capabilities

The capabilities where "does mojotrees have this" has more than one honest
answer, scored against `docs/CAPABILITY_LEVELS.md`. Cells are `yes`, `no`,
or `n/a`. A capability not listed here is one whose levels agree with its
status word, and the row in the sections below is the whole story.

| Capability | Status | implemented | integrated | publicly reachable | focused-tested | differential-tested | hardware-validated | release-packaged | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| SciPy sparse Python input | partial | yes | yes | yes | no | no | n/a | yes | Chain is live: `python/mojotrees/_arrays.py` to `bindings/_mojotrees.mojo` (`fit_csc`, `predict_csr`) to `src/mojotrees/model_sparse.mojo`. The Mojo layer is covered by `tests/test_sparse.mojo` in `pixi run test`; the Python entry point has two tests that call `pytest.importorskip("scipy.sparse")` and `pixi.toml` does not put scipy in the `pytest` environment, so they skip in the only run CI performs |
| Exact TreeSHAP contributions | supported | yes | yes | yes | yes | yes | n/a | yes | `src/mojotrees/contrib.mojo`, reached by `predict(pred_contrib=True)` on all three estimators. `tests/test_contrib.mojo` checks against Shapley values enumerated over all 2^M subsets, which is an independent implementation, and runs in `pixi run test` |
| Cross-validation (`mojotrees.cv`) | partial | yes | yes | yes | yes | no | n/a | yes | `python/mojotrees/cv.py`, orchestration over `Dataset`, `Booster`, and `Booster.eval`, so a fold model is what `train()` would have built. `python/tests/parallel/test_cv.py` runs under `pixi run -e pytest test-estimators`. `cv` and `CVBooster` are in `mojotrees.__all__`, so both are public under section 2 of `docs/COMPATIBILITY_POLICY.md`; `partial` is about the surface rather than the reach, since LightGBM's `cv` returns a plain dict keyed the same way but has been exercised against far more of its own parameter space than this one has |
| Model inspection and dump (`mojotrees.inspection`) | partial | yes | yes | yes | yes | no | n/a | yes | `dump_model`, `trees_to_dataframe`, `trees_to_records`, and `get_split_value_histogram` are in `mojotrees.__all__` and resolve the submodule lazily, so all four are public. `partial` is now about the *producer*, not the reach: `python/mojotrees/inspection.py` rebuilds the schema by parsing `Booster.model_to_string()`, while `src/mojotrees/inspection.mojo` and `src/mojotrees/model_dump.mojo` are the native producers with no caller. `bindings/_mojotrees.mojo` now imports `bindings/inspection_bindings.mojo` and registers `dump_model`, `dump_model_multiclass`, `split_values`, `dump_leaf_index`, `dump_raw_scores`, and `objective_code`, so an extension built from this tree takes the native route and the parser is the fallback rather than the path. What keeps this `partial` is `split_gains`, which exists on neither side of the seam: every dumped node reports `split_gain: None`. `python/tests/parallel/test_inspection.py`, and `docs/INTEGRATION_INVENTORY.md` |
| Split gains in a dump | supported | yes | yes | yes | yes | no | n/a | yes | `src/mojotrees/serialize.mojo` writes per-node split gains as of model format v4, so a dump reports a measured gain per internal node and `has_split_gain: True`, whichever path built it, and `Booster.feature_importance("gain")` survives a save, a pickle, and a `load`. A model read from a v1, v2, or v3 file carries none and says so through the flag; that is the only remaining `None`. `python/mojotrees/inspection.py` still looks for a `_mojotrees.split_gains` hook, which nothing defines, as the way an older file's gains could come from a live handle |
| Dask adapter (`mojotrees.dask`) | deferred | yes | no | no | yes | n/a | n/a | yes | `python/mojotrees/dask.py` validates partition metadata, plans ranks, negotiates capabilities, and predicts from model bytes, all against a fake backend in `python/tests/parallel/test_dask_contract.py`. No backend is registered by default, so `fit` raises `DistributedNotAvailable` before touching a cluster. `DaskRuntime` has never run against a live cluster |
| Explainable device selection (`mojotrees.device_selection`) | partial | yes | yes | yes | yes | n/a | n/a | yes | `explain_device_choice` is in `mojotrees.__all__`, so a user can ask "what would `device='gpu'` do here" and get a report rather than an exception. `python/mojotrees/device_selection.py` with 54 tests in `python/tests/parallel/test_device_selection.py`. Two things keep it `partial`. The estimators do not route through it: `_Base._resolve_device` calls `_mojotrees.resolve_device` directly, so a report and a `fit` are two calls into the same native engine rather than one. The `"narrow"` contract is no longer forced: `bindings/_mojotrees.mojo` registers `decide_device`, so an extension built from this tree carries the blocking reasons, warnings, memory estimate, and evidence identifier, and `report.contract` says which of the two answered. `docs/INTEGRATION_INVENTORY.md` |
| Startup diagnostics (`mojotrees.diagnostics`) | deferred | yes | no | no | no | n/a | n/a | yes | `python/mojotrees/diagnostics.py` formats phase durations that something else measured, over the ten-phase contract in `src/mojotrees/initialization.mojo`. Not a LightGBM capability, so it has no parity row of its own. The native half is wired now (`src/mojotrees/device_policy.mojo` and `src/mojotrees/gpu_runtime.mojo` both import `initialization`), and the Python half is not: nothing imports `python/mojotrees/diagnostics.py`, no name from it is in `mojotrees.__all__`, and neither `python/tests/` nor `tests/` has a suite for it |
| Exclusive feature bundling | partial | yes | yes | yes | yes | n/a | n/a | n/a | `src/mojotrees/efb.mojo` with `tests/test_efb.mojo` in `pixi run test`. Wired end to end as of this revision: `BoosterParams.bundling` carries `EfbSettings`, `src/mojotrees/boosting.mojo` fits the plan once with `prepare_bundling` and hands the bundled matrix to every tree, `src/mojotrees/split.mojo` unbundles a bundled histogram, and `src/mojotrees/params.mojo` parses `enable_bundle` and `max_conflict_rate`. Reachable through the parameter string, which section 2 of `docs/COMPATIBILITY_POLICY.md` makes public, and through `BoosterParams` from Mojo. `partial` because no Python estimator parameter reaches it, so a `MojoTreesRegressor` user still cannot ask for it |
| Distributed transport | partial | yes | yes | no | yes | n/a | n/a | n/a | `src/mojotrees/distributed_transport.mojo` with `tests/test_distributed_transport.mojo` in `pixi run test`. `src/mojotrees/distributed.mojo` now imports it and calls `require_transport`, `open_local_collective`, and `histogram_plan`, so the local runtime goes through the transport contract. Only the local runtime does: `RankAddress` is not exported, `run_distributed` raises for any other runtime spec, and no process has connected to another |
| LightGBM model file interop | partial | yes | yes | yes | yes | no | n/a | n/a | `src/mojotrees/lgbm_model_io.mojo` is the converter; `bindings/lgbm_bindings.mojo` binds its four file-level entry points (`lgbm_interop_status`, `lgbm_file_unsupported_reason`, `lgbm_import_file`, `lgbm_export_file`) and `python/mojotrees/lgbm_model_io.py` (a lazy submodule, `mojotrees.lgbm_model_io`) wraps them as `interop_status`, `unsupported_reason`, `convert_to_mojotrees`, `convert_from_mojotrees`, `load_lightgbm_model`, `save_lightgbm_model`; every conversion goes through a mojotrees model file. `tests/test_lgbm_model_io.mojo` in `pixi run test`; `python/tests/test_lgbm_interop.py` round-trips a fitted booster. `partial` because the status text says what it says: no file a real LightGBM build wrote has been read and no file written here has been read back by LightGBM (the first use warns once), monotone constraints are not read out of the file (the two booster constructions in this file stay exempted from check 10 in `tools/check_parity.py`), and `is_linear=1` is refused |
| Remaining tree-parameter rules | partial | yes | yes | yes | yes | n/a | n/a | n/a | `src/mojotrees/tree_parameters_extra.mojo`, with `tests/test_tree_parameters_extra.mojo`. Most of the bundle is wired as of this revision. `src/mojotrees/split.mojo` imports and calls `passes_min_gain`, `apply_monotone_penalty`, `extra_threshold_index`, and `finish_leaf_output`, which covers `min_gain_to_split`, `monotone_penalty`, `extra_trees` with `extra_seed`, `max_delta_step`, and `path_smooth`; forced splits reach the grower through `binning.map_forced_splits`; and `src/mojotrees/params.mojo` parses each of those keys, so they are reachable through the parameter string. `feature_contri` and the CEGB penalties are live too as of this revision: `split._feature_gain` applies `FeaturePenalties.penalized_gain` for the multiplier and `cegb.CegbNodeCosts.adjusted_gain_at` for every CEGB term, one line apart. `feature_contri` is reachable from the Python estimator; the two CEGB vectors are Mojo-API-only (see the CEGB rows) |
| Apple GPU tuning policy | partial | yes | yes | no | yes | n/a | yes | n/a | `src/mojotrees/apple_gpu_policy.mojo` with `tests/test_apple_gpu_policy.mojo`. It is read now, in two places: `src/mojotrees/device_policy.mojo` takes `GpuProfile` and `partial_budget_bytes` from it for the device profile and the session memory estimate, and `src/mojotrees/apple_histogram_policy.mojo` takes its `derive_block_threads`. What it still does not do is decide any default launch: `apple_histogram_policy` resolves to `SPEC_LEVEL_BASELINE` unless `MOJOTREES_GPU_HIST_SPECIALIZATION` asks otherwise, so `src/mojotrees/gpu_tiling.mojo` remains the tiling policy in force. Note that `derive_block_threads` now names two different functions, one per module, with different signatures |
| NVIDIA and AMD GPU backend policy | deferred | yes | no | no | yes | n/a | no | n/a | `src/mojotrees/gpu_vendor_policy.mojo`, the merge of the two former per-vendor twins (f23bd1b), with `tests/test_gpu_vendor_policy.mojo` in `pixi run test`. Nothing reads it. `src/mojotrees/gpu_tiling.mojo` is still the geometry in force, and no NVIDIA or AMD device has ever run this project's kernels, so every function that would select a specialized kernel variant refuses unless `MOJOTREES_GPU_BACKEND_UNVALIDATED=1` acknowledges that. The test is host arithmetic over reported numbers and opens no device, which is the only way any of it can be exercised here. `docs/GPU_BACKEND_SPECIALIZATIONS.md`, `docs/GPU_VALIDATION.md` |
| Per-level feature sampling | partial | yes | yes | yes | yes | n/a | n/a | n/a | `select_split_features` in `src/mojotrees/sampling.mojo`, with `tests/test_sampling.mojo`. `TreeParams.feature_fraction_bylevel` is carried through `src/mojotrees/tree.mojo`, `src/mojotrees/tree_sparse.mojo`, and `src/mojotrees/train_gpu.mojo`, `src/mojotrees/params.mojo` parses it under LightGBM's spelling and XGBoost's `colsample_bylevel`, and the distributed and device split-search paths refuse it by name rather than ignoring it. `select_level_features` itself still has no caller. It is XGBoost's parameter, so it is an extension rather than a parity row, and no Python estimator parameter reaches it |
| GPU packed-bin layout | deferred | yes | no | no | no | n/a | n/a | n/a | `src/mojotrees/gpu_binned_layout.mojo` plans a packed layout over the bit primitives in `src/mojotrees/gpu_bin_packing.mojo`. The module is reached as of this revision and the layout is still not planned: `src/mojotrees/histogram_gpu.mojo` imports `check_layout_support` and calls it when the builder opens, which refuses a feature count or an `n_rows * n_features` cell count that would wrap an Int32 index. That is one guard, not a packed plan: no caller builds a `BinLayoutPlan` (`plan_feature_major`, `plan_row_major`, `plan_feature_blocked` are named only in prose elsewhere), nothing calls `pack_binned_matrix` or a `gpu_bin_packing` primitive, and no suite imports either module. `docs/INTEGRATION_INVENTORY.md` |
| Depth-wise (level-wise) growth | different | yes | yes | yes | yes | n/a | n/a | n/a | `TreeParams.grow_policy` / Python `grow_policy=` / parameter string `grow_policy` (`leafwise` default, `depthwise`; XGBoost's names, LightGBM has no such switch). Order decided by `src/mojotrees/growth_policy.mojo` `GrowthSchedule` (the one pick both policies go through), followed by `tree.grow_tree` (`src/mojotrees/tree.mojo`), `tree_sparse.grow_tree_sparse` (`src/mojotrees/tree_sparse.mojo`), and all three loops in `src/mojotrees/train_gpu.mojo`; the distributed prototype rejects it. `tests/test_grow_policy.mojo`. Half of the batched per-level device launch the design doc argues for (`docs/design/GPU_LEVELWISE.md`) now ships, in the device-resident split search and there only: a level costs one host wait and one search launch pair instead of one per split, about 5 waits per tree rather than 30, while the row partition and the histogram build stay per node. Measured on an Apple M4 (`bench/results/sweep2_2026-08-15/RESULTS.md`), that makes depth-wise the fastest GPU arm this project has: 2.587s at 1,000,000 x 50 against leaf-wise's 3.756s and LightGBM's 2.767s at 10 threads, the only shape and arm on which this library is ahead of LightGBM on training time. Two qualifications belong to that number wherever it is quoted. It is our depth-wise trees against LightGBM's leaf-wise trees, which are different models, so it is not a parity result and does not make depth-wise a default. And **no accuracy measurement of any kind has been taken on a depth-wise arm**: the sweep record kept timings only, and the real-data parity evidence in `bench/real_data/results/` is entirely leaf-wise. The 0.3 percent spread on that arm is ours; `bench/bench_lightgbm.py` trains once in a separate process with no repeat loop, so the comparator is a single sample with an unknown noise floor |
| Class-batched GPU multiclass rounds | partial | yes | yes | yes | no | n/a | no | n/a | `src/mojotrees/gpu_multiclass_batch.mojo` over the interleaved planes in `src/mojotrees/gpu_output_planes.mojo`. Wired as of this revision: `src/mojotrees/train_gpu.mojo` asks `class_schedule` for a plan and, when that plan is not sequential, dispatches to `_train_multiclass_gpu_batched`, which builds a `GpuClassBatch.for_plan` and runs the round under `MulticlassRoundGuard`. The default is still one tree per class per round, which is what the supported row below describes; the batch widens only when `MOJOTREES_GPU_CLASS_BATCH` asks, and that variable is public under section 2 of `docs/COMPATIBILITY_POLICY.md`, which is the whole of the reach. No suite imports the module, nothing has compared the batched round against the sequential one, and no accelerator has run it |
| Hybrid CPU and GPU leaf placement | unsupported | no | no | no | no | n/a | n/a | n/a | Removed on 2026-08-16 with the `hybrid_leaf_scheduler` module. It decided per leaf which device should build that leaf's histogram, behind a double opt-in (`MOJOTREES_HYBRID_LEAVES` plus `MOJOTREES_HYBRID_COSTS=apple-m4`), and it only ever reached host-gradient runs. Its premise was that the host could usefully take the leaves the device was slow at; that premise is gone, because the device-resident tree plane beats the host path at every shape measured, all resolved under rule M0 (2.2x at 50,000 rows, 44 percent at 250,000, 24 percent at 1,000,000: `bench/results/session3_2026-08-16/RESULTS.md`). LightGBM has no equivalent, so this row is not a parity gap in either direction. The design record is `docs/design/HYBRID_TRAINING.md`, the calibration is `bench/results/apple_m4_hybrid_costs_2026-08-15.md`, and the host replica builder the scheduler called survives as the CPU/GPU oracle (`histogram.build_histogram_subset_replica_into`, `tests/test_host_replica.mojo`). its cache-policy companion, which only that module imported, was deleted the same day |
| Sparse GPU training | partial | yes | yes | yes | yes | no | yes | no | `src/mojotrees/train_gpu_sparse.mojo` (`train_gpu_sparse`, `train_gpu_sparse_with_valid`, `train_multiclass_gpu_sparse`, exported from the package root) drives `src/mojotrees/gpu_sparse.mojo` and routes categorical splits through `src/mojotrees/gpu_categorical.mojo`; `model_sparse.fit_csc` / `fit_multiclass_csc` dispatch to it on `device='gpu'`, and the estimators reach that through `bindings/_mojotrees.mojo` `fit_csc` on a SciPy matrix with `device="gpu"`. `tests/test_gpu_sparse.mojo` (13 checks, M4: every row's device leaf equals the host walk of the grown tree, categorical and missing routing included; fits agree with `train_sparse` at the dense GPU trainer's tolerance) and `python/tests/test_sparse_gpu.py`. Not differential-tested against LightGBM's GPU, not release-packaged beyond the extension build, and its crossover is unmeasured, so `auto` keeps the CPU. Refuses exclusive feature bundling and, as on the CPU, custom objectives and eval sets |
| DART and random forest boosting | partial | yes | yes | yes | yes | no | n/a | n/a | `src/mojotrees/alternate_boosting.mojo` (exported from the package root) dispatches `boosting="dart"` / `"rf"` to `src/mojotrees/boosting_dart.mojo` and `src/mojotrees/boosting_rf.mojo`; `_mojotrees.fit` reads the `boosting` key and routes both there while gbdt and goss make the call they made before, and `MojoTreesRegressor` / `MojoTreesClassifier` take `boosting=` (alias `boosting_type=`) with `drop_rate`, `max_drop`, `skip_drop`, `xgboost_dart_mode`, `uniform_drop`, `drop_seed`; both `uniform_drop` rules follow LightGBM's `src/boosting/dart.hpp` (`docs/DART.md` section 4.1) and rf trains at learning rate 1.0 as LightGBM's does. Multiclass dart and rf exist on the Mojo API as `fit_boosting_multiclass`. `tests/test_alternate_boosting.mojo` in `pixi run test`; `python/tests/test_params.py` fits both through the estimator. `partial`: dense CPU single-output only from Python (sparse input, `eval_set`, a callable objective, the multiclass classifier, and the ranker refuse both modes by name); no `eval_set` early stopping under dart from any entry point; no LightGBM differential, so the fitted models are not claimed numerically equal to LightGBM's |
| Cost-effective gradient boosting | partial | yes | yes | yes | yes | no | n/a | n/a | `src/mojotrees/cegb.mojo` implements LightGBM's four `cegb_*` controls and is the only implementation: `FeaturePenalties` holds a `CegbConfig` and `penalized_gain` applies the `feature_contri` multiplier alone. Charged at `split._feature_gain` through `CegbNodeCosts`, which `tree.grow_tree_with_cegb` prepares per node and `boosting._boost_rounds` / `_boost_rounds_multiclass` back with one `CegbLedger` per ensemble. `src/mojotrees/params.mojo` parses `cegb_tradeoff` and `cegb_penalty_split`; both per-feature vectors are Mojo-API-only, as `monotone_constraints` is, and reach the binding through `_penalties`. `tests/test_cegb.mojo` in `pixi run test`, written and not yet run. `partial` for three reasons, all in `docs/CEGB.md` section 10: the cached-candidate refund cannot promote a runner-up where LightGBM's per-(leaf, feature) table can, the lazy term multiplies by a count where LightGBM accumulates per row, and no Python estimator parameter carries either vector. Growers that carry no ledger charge the split cost and refuse the other two by name rather than ignoring them |
| GPU multiclass training | supported | yes | yes | yes | yes | yes | yes | yes | `fit_multiclass` resolves the device and calls `train_multiclass_gpu` (`src/mojotrees/model.mojo`), and `gpu_supports` in `src/mojotrees/device.mojo` now admits every output count, so `MojoTreesClassifier(device="gpu")` trains on the accelerator. `tests/test_gpu_objectives.mojo` checks GPU against CPU, `tests/test_device.mojo` checks the routing. Apple M4 only (`docs/GPU_VALIDATION.md`) |
| GPU split selection on the device | supported | yes | yes | yes | yes | yes | yes | yes | `src/mojotrees/gpu_split_search.mojo`, reached from `grow_tree_gpu` and from `MOJOTREES_GPU_SPLIT_STRATEGY=device`. Off by default: `SPLIT_SEARCH_AUTO` resolves to the host scan, because Float32 device gains can flip near-tie decisions and the measured speed difference does not pay for that on its own. It grows over a device-resident frontier, so a split builds one histogram and subtracts for the sibling, as the host-search grower does; see the measurements under "GPU split selection" below |
| GPU prediction | partial | yes | yes | yes | yes | yes | yes | yes | `src/mojotrees/gpu_predict.mojo`, reached from `Model.predict_batch(device=)` in the Mojo API and, since the device-aware entry points landed in `bindings/_mojotrees.mojo` (`predict_batch`, `predict_proba_batch`, `_predict_device`, `gpu_predict_capability`), from Python as well: `predict`, `predict_proba`, and the ranker's `predict` all take `device=`. The older claim here, that the binding takes no device and the estimators always predict on the CPU, is withdrawn. `partial` now for a narrower reason: contributions (`pred_contrib`) and sparse input have no device path and refuse an explicit `gpu` |
| C ABI | partial | yes | yes | yes | yes | n/a | n/a | no | `capi/mojotrees.h` declares twelve functions with a version query; `capi/mojotrees_capi.mojo` implements them. `tests/test_capi.mojo` runs in `pixi run test`, hence in CI. `capi/run_c_tests.sh` (`pixi run test-c`) compiles the C caller and is not in any CI job, and no wheel or release artifact carries the shared library |
| Command line tool | different | yes | yes | yes | yes | n/a | n/a | no | `cli/mojotrees_cli.mojo` with `tests/test_cli.mojo` in `pixi run test`. It is mojotrees's own CSV tool over the Mojo API, not LightGBM's config-file surface, which stays `unsupported` in section 7 |
| macOS arm64 wheel artifact | partial | yes | yes | yes | no | n/a | n/a | yes | `packaging/build_wheel.sh` bundles the four MAX runtime dylibs with an `@loader_path` rpath and re-signs, `packaging/test_wheel.sh` installs and smoke-tests it, and `packaging/macos/build_release_wheel.sh` is driven by a release workflow that builds, verifies, clean-installs, and hashes on a tag or a manual dispatch. `focused-tested` is `no` because no per-change job guards any of that: the everyday CI workflow has no wheel job, and `docs/PLATFORM_MATRIX.md` still records the target as `designed` |

---

## 1. Top-level Python package symbols

Every name in `lightgbm.__all__` for 4.7.0, verified by importing the pinned
package during the version 1 audit.

| LightGBM symbol | Status | Notes | mojotrees |
|---|---|---|---|
| `LGBMRegressor` | supported | `MojoTreesRegressor`. Named for the library, not for LightGBM; behavior and parameter names match where documented | `python/mojotrees/__init__.py` |
| `LGBMClassifier` | supported | `MojoTreesClassifier`. Binary and softmax multiclass, chosen from the label count | `python/mojotrees/__init__.py` |
| `LGBMRanker` | supported | `MojoTreesRanker`, LambdaRank | `python/mojotrees/__init__.py` |
| `LGBMModel` | different | No shared public base class. `_Base` holds the shared hyperparameters but is private, because a bare `LGBMModel` with `objective=` selecting the task is a second way to spell what the three estimators already do | `python/mojotrees/__init__.py` |
| `Booster` | supported | `mojotrees.Booster`: prediction, evaluation, feature importance, model IO, iteration counts, and continued training with `update()`. The estimators hold one on `booster_`, so there is a single model object. `dump_model` is not a method; `mojotrees.inspection.dump_model(model)` is the dump, section 0 | `python/mojotrees/basic.py`, `python/tests/test_basic.py` |
| `Dataset` | supported | `mojotrees.Dataset`, over the Mojo `Dataset` in `src/mojotrees/trainset.mojo`: data, label, weight, group, init score, feature names, categorical declaration, and binning metadata, binned once and reused. Immutable once constructed; see section 5 for the mutators mojotrees does not have | `python/mojotrees/basic.py`, `src/mojotrees/trainset.mojo`, `tests/test_trainset.mojo` |
| `train` | supported | `mojotrees.train(params, train_set, num_boost_round, valid_sets, valid_names, init_model)`. Trains the same trees the estimators train, which `python/tests/test_basic.py` asserts bit for bit. `init_model` continues from a fitted booster. No per-round history or early stopping here; those are on the estimators' `fit` | `python/mojotrees/basic.py` |
| `cv` | partial | `python/mojotrees/cv.py` implements it: folds, `stratified`, `shuffle`, caller-supplied `folds` or a scikit-learn splitter, `metrics`, `feval`, `fpreproc`, `init_model`, `eval_train_metric`, `return_cvbooster`, and early stopping, returning LightGBM's `{metric-mean, metric-stdv}` history. It re-bins per fold rather than slicing a constructed `Dataset`, so fold binning cannot leak. `mojotrees.cv(params, train_set, ...)` is the call: the name is in `mojotrees.__all__`, which section 2 of `docs/COMPATIBILITY_POLICY.md` makes the definition of public, and the attribute resolves to the function rather than to the submodule of the same name (`_public_api_plan.NAME_COLLISIONS`). `partial` is about the parameter space exercised, not the reach. Section 0 | `python/mojotrees/cv.py`, `python/tests/parallel/test_cv.py` |
| `CVBooster` | partial | With `cv`, and public the same way: `mojotrees.CVBooster` is in `__all__`, so an isinstance check on what `cv(return_cvbooster=True)` returns needs no submodule import | `python/mojotrees/cv.py` |
| `early_stopping` | supported | `fit(callbacks=[early_stopping(rounds, first_metric_only=, verbose=, min_delta=)])`, and the `fit(early_stopping_rounds=, min_delta=)` spelling. The callback configures the trainer's own stopper rather than reimplementing the rule; passing both spellings raises. Differs in which round survives: the primary metric's best on the first validation set, not the pair that ran out of patience first | `python/mojotrees/callback.py`, `src/mojotrees/callback.mojo`, `src/mojotrees/custom_metric.mojo` |
| `log_evaluation` | supported | `log_evaluation(period=, show_stdv=)`. `period<=0` is silent. `show_stdv` is accepted and inert on `fit`: it formats a cross-validation fold's spread, and `fit` has no folds | `python/mojotrees/callback.py`, `src/mojotrees/callback.mojo` |
| `record_evaluation` | supported | `record_evaluation(dict)` fills the dict in place. `evals_result_` is still populated directly too; it starts one round earlier, at the base-score-only model | `python/mojotrees/callback.py`, `src/mojotrees/callback.mojo`, `python/mojotrees/__init__.py` |
| `reset_parameter` | partial | `reset_parameter(**schedules)` with lists or callables, for the nine hyperparameters the loop re-reads each round (`callback.RESETTABLE`). A key outside that set raises rather than being ignored, which LightGBM does not do. A learning-rate schedule bakes shrinkage into the leaf values | `python/mojotrees/callback.py`, `src/mojotrees/callback.mojo` |
| `EarlyStopException` | supported | Raised by a callback to stop the run; rolls the ensemble back to the best round as LightGBM does | `python/mojotrees/callback.py`, `src/mojotrees/callback.mojo` |
| `EvalResult` | different | The 4-tuple `(data_name, metric_name, value, is_higher_better)` is what `env.evaluation_result_list` holds, matching LightGBM's shape; there is no named type for it | `python/mojotrees/callback.py`, `src/mojotrees/callback.mojo` |
| `register_logger` | unsupported | mojotrees has no logging layer to redirect. Training is silent by design; adding a logger to redirect is not a goal | none |
| `plot_importance` | unsupported | Plotting belongs in the caller's plotting library. `feature_importances_` is the data; matplotlib is not a dependency mojotrees will take | none |
| `plot_metric` | unsupported | Same reason. `evals_result_` is the data | none |
| `plot_split_value_histogram` | unsupported | The plot is out of scope for the same reason as the other two. The data it renders now exists: `mojotrees.inspection.get_split_value_histogram(model, feature)` returns it, section 5 | `python/mojotrees/inspection.py` |
| `plot_tree` / `create_tree_digraph` | unsupported | The graphviz rendering is out of scope; graphviz is not a dependency mojotrees will take. The structured tree it renders now exists as `mojotrees.inspection.dump_model`, section 5 | `python/mojotrees/inspection.py` |
| `Sequence` | partial | An object implementing LightGBM's `lgb.Sequence` protocol (`__len__`, `__getitem__`, `batch_size`) is accepted everywhere a feature matrix is (`Dataset`, the estimators), read in `batch_size` slices as LightGBM reads it and streamed into the native `Dataset` one converted batch at a time through `dataset_chunks_begin` / `push` / `finish`; the class itself is not re-exported, and `mojotrees._sequence.Batches` is the in-house spelling. Native chunk protocol and out-of-core binner: `src/mojotrees/sequence.mojo`, `src/mojotrees/external_memory.mojo`, `tests/test_external_memory.mojo` | `python/mojotrees/_sequence.py`, `bindings/sequence_bindings.mojo` |
| `DaskLGBMRegressor` / `DaskLGBMClassifier` / `DaskLGBMRanker` | deferred | `python/mojotrees/dask.py` has the three classes and the client-side contract, and they raise `DistributedNotAvailable` on `fit` because no backend is registered and no transport is wired up. Building them on the in-process prototype would be a distribution claim with nothing behind it. Section 0, and task 17 after task 16 | `python/mojotrees/dask.py`, `docs/distributed.md` |

## 2. Estimator constructor parameters

Every parameter of `lightgbm.LGBMModel.__init__` in 4.7.0.

| LightGBM parameter | Status | Notes |
|---|---|---|
| `boosting_type` | supported | Accepted as `boosting` (LightGBM's native name) with `boosting_type` as an alias. `gbdt`, `goss`, `dart`, and `rf`; the paths that dart and rf do not cover refuse them by name, see section 7 |
| `num_leaves` | supported | Same default (31) |
| `max_depth` | supported | Same default (-1). Leaf-wise growth is preserved under the limit; under `grow_policy=depthwise` it becomes the primary control |
| `learning_rate` | supported | Same default (0.1) |
| `n_estimators` | supported | Same default (100) |
| `subsample_for_bin` | deferred | The sampled edge fit is `fit_bins(..., bin_construct_sample_cnt=...)` in `src/mojotrees/binning.mojo` (section 5) and its default is now LightGBM's 200,000, so the scikit-learn API already fits edges from a sample. What is deferred is *control* of it: `MojoTreesRegressor(subsample_for_bin=...)` is not accepted, so an estimator caller cannot change the count or ask for every row |
| `objective` | partial | Regressor: `regression`, `huber`, `quantile`, `mae`/`regression_l1`, `poisson`, `gamma`, `tweedie`, `mape`, `fair`, `cross_entropy`, or a callable. Classifier: rejected with a message rather than a bare `TypeError`, because the task comes from the labels and a custom objective is single-output only. The objectives that are not implemented are listed in section 8, and each is reported by name rather than as an unknown one |
| `class_weight` | supported | `MojoTreesClassifier(class_weight=...)`: `"balanced"` or a `{label: weight}` dict, folded into the row weights before training, so there is one weighting mechanism rather than two. scikit-learn's rule for `balanced` (row counts, not weighted counts). `src/mojotrees/class_weight.mojo` |
| `min_split_gain` | supported | Accepted as `min_gain_to_split` (LightGBM's native name) with `min_split_gain` as an alias, same default (0.0). The floor is applied in `src/mojotrees/split.mojo` through `passes_min_gain`, so a candidate below it cannot win; see section 7 |
| `min_child_weight` | supported | Alias for `min_child_hess` (LightGBM's `min_sum_hessian_in_leaf`), default 1e-3 |
| `min_child_samples` | supported | Alias for `min_data_in_leaf`, default 20 |
| `subsample` | supported | Alias for `bagging_fraction` |
| `subsample_freq` | supported | Alias for `bagging_freq` |
| `colsample_bytree` | supported | mojotrees's native name is `feature_fraction`, with `colsample_bytree` accepted as an alias; `colsample_bynode` is likewise an alias of `feature_fraction_bynode`. Both resolve through `_resolve_alias`, so setting a name and its alias to different values raises rather than silently picking one |
| `reg_alpha` | supported | Alias for `lambda_l1` |
| `reg_lambda` | supported | Alias for `lambda_l2`. The default no longer differs: mojotrees defaulted `lambda_l2` to 1.0 against LightGBM's 0.0 until 2026-08-16, and it is 0.0 on both sides now (`tools/check_parity.py` asserts it) |
| `random_state` | different | mojotrees has no single global seed. Each sampler takes its own (`bagging_seed`, `feature_fraction_seed`, `goss_seed`), and every stream is counter-based, so a draw depends only on its seed and index and never on history. One global seed would reintroduce the ordering dependence that design removes |
| `n_jobs` | different | Thread count is `MOJOTREES_NUM_WORKERS`, an environment variable, not a model parameter, because it changes how a model is computed and never what it equals. Same rule as the GPU tiling knobs |
| `importance_type` | supported | `split` (default) and `gain`, LightGBM's two types |
| `**kwargs` (arbitrary core parameters) | different | Unknown keyword arguments raise. LightGBM forwards them to the C++ config, which silently accepts typos of parameters it does not know |

Additional mojotrees constructor parameters that LightGBM spells only as
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
| `fit(eval_metric=)` | supported | LightGBM's built-in metric names (`l2`, `rmse`, `l1`, `quantile`, `huber`, `mape`, `fair`, `poisson`, `gamma`, `gamma_deviance`, `tweedie`, `cross_entropy`, `kullback_leibler`, `binary_logloss`, `binary_error`, `auc`, `average_precision`, `multi_logloss`, `multi_error`, `ndcg`, `map`, and their aliases), callables, tuples, dicts, or a mixture, defaulting to the objective's own loss. A name is resolved by `python/mojotrees/_eval.py` and computed by `src/mojotrees/metrics.mojo`, the same code the Mojo API calls, so the two answers cannot drift. Two deliberate differences: a metric that cannot mean anything for the model being fitted is rejected rather than scored, and a callable's direction is declared up front with `("name", f, True)` rather than returned per call, because early stopping needs it before the first evaluation |
| `fit(feature_name=)` | partial | Feature names are read from a pandas frame's columns into `feature_names_in_` and checked at predict time, and `Dataset(feature_name=)` takes them on the functional API. An explicit `feature_name=` argument on the estimators' `fit`, and carrying names into the model file, are not there |
| `fit(categorical_feature=)` | supported | Accepted as `categorical_feature` on the constructor (LightGBM's name) rather than on `fit`, because scikit-learn's clone contract keeps hyperparameters on the estimator. Indices, column names, or `"auto"` (the default, meaning every pandas `category` column), and reported back on the fitted `categorical_feature_`. One difference: a `category` column left out of an explicit list raises, where LightGBM quietly feeds its codes to the numerical scan |
| `fit(callbacks=)` | partial | Supported for the regressor and the binary classifier, which train through `train_with_callbacks`. The softmax and LambdaRank loops score validation metrics and stop early but have no per-round hook, so they refuse a callback list rather than ignoring it. Needs an `eval_set` |
| `fit(init_model=)` | deferred | Continued training from the estimators is task 6. `train(init_model=)` has it on the functional API (section 5) |
| `predict(X)` | supported | Response scale, matching LightGBM's default |
| `predict(raw_score=)` | supported | Scores on the link scale. The objectives without a link (squared error, huber, quantile, L1) and the ranker predict raw either way |
| `predict(start_iteration=)` / `predict(num_iteration=)` | supported | A slice of the ensemble, LightGBM's pair, with LightGBM's clamping. `num_iteration=None` means every iteration the model kept. Not available for sparse input |
| `predict(pred_leaf=)` | supported | Leaf index per tree. Combining it with `raw_score` raises, where LightGBM silently lets one win. Not available for sparse input |
| `predict(pred_contrib=)` | supported | Exact TreeSHAP (path-dependent), LightGBM's shapes: `n_features + 1` columns with the expected value last, and `n_classes * (n_features + 1)` in class-major blocks for multiclass. Every row sums to its raw score. Combining it with `raw_score` or `pred_leaf` raises. Needs node covers, so a model saved in format v1 or v2 raises rather than guessing (`src/mojotrees/contrib.mojo`). Not available for sparse input, which refuses rather than densifying |
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
| `feature_importances_` | supported | Both kinds computed at fit time. A model read back with `load()` reports zero gain importance and warns, because `src/mojotrees/serialize.mojo` does not write gains |
| `best_iteration_` | different | Always set, to the boosting iteration the model is used at. LightGBM sets it only when early stopping ran |
| `best_score_` | partial | A single float, the primary metric's best value on the first validation set. LightGBM's is a nested dict over sets and metrics; the rest of that grid is in `evals_result_` |
| `evals_result_` | supported | `{valid_name: {metric_name: [values]}}`. Index 0 is the base-score-only model, so entry `i` is the score after `i` trees; LightGBM starts at the first iteration |
| `booster_` | supported | The `Booster` holding the fitted model, and the only place the handle lives. It cannot `update()`: an estimator bins its own matrix and keeps no `Dataset` to grow on |
| `objective_` | deferred | The resolved objective name is not exposed; `objective` is echoed back as given. `mojotrees.inspection.objective_of(model)` reads it out of the model text meanwhile |
| `n_iter_` | supported | The number of boosting iterations that were trained. It differs from `best_iteration_` only when a validation metric peaked before the last round with early stopping off |
| `n_estimators_` | different | LightGBM's second name for the same number. `n_iter_` reports it; a third spelling is not added |
| `feature_name_` | different | `booster_.feature_name()` reports the training feature names, or LightGBM's `Column_0`, `Column_1`, ... when there were none. A second estimator attribute alongside `feature_names_in_` is not added |
| `categorical_feature_` (mojotrees) | different | Not a LightGBM attribute. The resolved categorical column indices, so `"auto"` can be inspected after the fact |
| `device_` (mojotrees) | different | Not a LightGBM attribute. Records which backend actually ran, because `device="auto"` makes that a runtime outcome |
| `stopped_early_` (mojotrees) | different | Not a LightGBM attribute. True when early stopping fired, which `best_iteration_ < n_estimators` alone does not distinguish from objective convergence |

## 5. Booster and Dataset APIs

`mojotrees.Booster` and `mojotrees.Dataset` are the functional API in
`python/mojotrees/basic.py`, over the Mojo `Dataset` and its trainers in
`src/mojotrees/trainset.mojo`. The estimators hold the same `Booster` on
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
| `Booster.rollback_one_iter` / `reset_parameter` | partial | `Booster.rollback_one_iter()` and `rollback_to(n)` truncate the ensemble natively (`src/mojotrees/model_editing.mojo`, `bindings/model_editing_bindings.mojo`); `tests/test_model_editing.mojo` checks the tree count drops by one and predictions match a fresh fit of one fewer round. `reset_parameter` stays as section 1 describes it (the nine schedulable hyperparameters). Not compared against LightGBM numerically |
| `Booster.predict` with all prediction modes | partial | `Booster.predict` covers response, raw score, and iteration ranges, with LightGBM's clamping rules. Leaf indices and feature contributions are on the estimators' `predict` (section 3) and still not on the Booster |
| `Booster.save_model` / `model_to_string` / `model_from_string` | different | Present under LightGBM's names, but the format is mojotrees's own versioned text (`src/mojotrees/serialize.mojo`, now at v4), which stores floats as raw bit patterns so a round trip predicts bit-exactly. It is not LightGBM-readable; `src/mojotrees/lgbm_model_io.mojo` is an unintegrated experiment in reading LightGBM's format, section 0. `save_model` takes no `num_iteration` or `start_iteration` |
| `Booster.dump_model` / `trees_to_dataframe` | supported | `Booster.dump_model()` and `Booster.trees_to_dataframe()` are methods, and both delegate to `mojotrees.inspection`, which builds a documented, versioned schema (`docs/MODEL_INSPECTION_SCHEMA.md`). Split gains are in the model text as of format v4, so every internal node reports a measured `split_gain` and the dump reports `has_split_gain: True`; a model read from an older file reports `None` and says so through the flag. `weight` stays None: mojotrees records a node's row cover, not a hessian sum. Section 0 |
| `Booster.feature_importance` | supported | Both `split` and `gain`, on the Booster and as `feature_importances_`. Gains are not in the model file, so a booster read back or unpickled reports zero gain importance |
| `Booster.num_feature` / `num_trees` / `num_model_per_iteration` | supported | Public methods on the Booster; `num_model_per_iteration` is the class count for a softmax model and 1 otherwise |
| `Booster.eval` / `eval_train` / `eval_valid` / `add_valid` | supported | Returns LightGBM's `(name, metric, value, is_higher_better)` tuples, weighted by the dataset's own weights. The metric defaults to the objective's own loss and the value comes from `src/mojotrees/metrics.mojo`, the same code `fit(eval_set=)` scores with, so the two APIs cannot drift |
| `Booster.feature_name` | supported | The training set's names, or LightGBM's `Column_0`, `Column_1`, ... when it had none |
| `Booster.get_split_value_histogram` | partial | `mojotrees.inspection.get_split_value_histogram(model, feature, bins=, as_frame=)` returns the histogram, and `split_values(model, feature)` the raw thresholds. Module functions, not `Booster` methods, and reachable only by importing the submodule. Section 0 |
| `Booster.lower_bound` / `upper_bound` | supported | The ensemble's extreme raw scores, computed natively over the leaf values (`src/mojotrees/model_editing.mojo`, reached as `Booster.lower_bound()` / `upper_bound()`); `tests/test_model_editing.mojo` checks every prediction lies inside them. Not in the dump schema, same as LightGBM |
| `Booster.get_leaf_output` / `set_leaf_output` / `shuffle_models` / `refit` | partial | All four are reachable: `Booster.get_leaf_output(tree, leaf)`, `set_leaf_output(tree, leaf, value)`, `shuffle_models(start, end)`, `refit(data, label, ...)` route through `bindings/model_editing_bindings.mojo` to `src/mojotrees/model_editing.mojo`; `MODEL_EDITING_SUPPORTED` is True and `model_editing_support()` reads the native status. `tests/test_model_editing.mojo`: a leaf write moves only that leaf's rows by exactly the delta, shuffle keeps the score sum, refit reports the leaves it re-estimated. A leaf edit does not recompute ancestor covers or gains, so `dump_model` reports the growth-time values for those; the file format writes the edited leaves. Not compared against LightGBM numerically |
| `Booster.free_dataset` / `set_train_data_name` | different | A booster holds a reference to its `Dataset`, which is what keeps continued training possible; dropping it is the caller's to do by dropping the booster. The training set is named `training` in `eval_train`, as LightGBM names it, and the name is not settable |
| `Booster.set_network` / `free_network` | deferred | Distributed training, task 16. Section 0 |
| `Dataset` construction and `construct` | supported | Binning happens on `construct()` or on the first `train()` that uses the dataset, and every later run reuses it. `free_raw_data` defaults to False here, not True: evaluation predicts through the model rather than reading an internal score buffer, so `eval_train()` needs the raw matrix |
| `Dataset.create_valid` / `set_reference` | different | A validation set is an ordinary `Dataset`; mojotrees predicts it through the model's own mapper, so it does not need the training set's bin mappers. `reference=` is accepted and its binning parameters are checked, so a mismatched reference is reported rather than ignored |
| `Dataset.subset` | partial | `Dataset.subset(rows, shared_binning=False)` is reachable, over `dataset_subset` in `bindings/dataset_bindings.mojo` and `Dataset.subset` / `Dataset.subset_shared_binning` in `src/mojotrees/trainset.mojo`. Rows must be strictly ascending and in range, the source must have kept its raw matrix, since bins cannot be refitted from bins, and a subset of a ranking dataset takes whole queries or is refused. The default is deliberately not LightGBM's. It fits bins over the subset's own rows, so the rows left out had no say in the edges, which is what a fold or a held-out split needs; `shared_binning=True` is LightGBM's semantics, binning the part as the whole was so a model trained on the whole can score it. What holds this at `partial` is coverage above the native layer. `tests/test_trainset.mojo` now pins both constructors, the refusals, and the whole-query rule, but nothing exercises the binding or the row-slicing branch in `python/mojotrees/basic.py`. `mojotrees.cv` neither uses nor needs it, since it slices the raw matrix and re-bins per fold on purpose |
| `Dataset.get_field` and the typed accessors (`label`, `weight`, `group`, `init_score`) | supported | All four are constructor arguments and all four read back, alongside `feature_name()` and `categorical_feature()`. `init_score` is training state, not model state: boosting starts from it and the fitted model predicts the trees alone (`tests/test_trainset.mojo`) |
| `Dataset.position` | supported | `Dataset(position=)` carries the per-row position column and `MojoTreesRanker.fit(position=)` takes it; with `lambdarank_position_bias_regularization` it turns on unbiased LambdaRank (`src/mojotrees/ranking_advanced.mojo`, `tests/test_ranking_advanced.mojo`) |
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
| `scipy.sparse` CSR | partial | `fit` and `predict` take any SciPy sparse matrix or array and keep it sparse: CSC to fit because histogram accumulation is feature-oriented, CSR to predict because prediction is row-oriented, converted without densifying and without mutating the caller's matrix. An implicit zero is the numerical value 0.0, matching LightGBM's default `zero_as_missing=false`, so a sparse fit equals the dense fit of the same matrix with the gaps filled with zeros. `device="gpu"` trains on the accelerator through the sparse GPU trainer (section 12) and `device="auto"` keeps the CPU. What raises rather than densifying behind your back: a Python objective callable, `eval_set` and early stopping, ranking, GPU prediction, `pred_leaf`, `pred_contrib`, and iteration slicing. The two loose ends recorded in "Known gaps" are both closed: the scikit-learn tag declares `input_tags.sparse` True and a meta-estimator test runs a sparse matrix through `cross_val_score`, and scipy is declared in the `pytest` pixi environment rather than inherited from scikit-learn, so the Python-side sparse cases run in CI. What holds the row at `partial` is the list above, the entry points that raise instead of taking sparse. Section 0 |
| `scipy.sparse` CSC | partial | Same path, same limits |
| `Dataset` from a file path | unsupported | Part of the file-based configuration surface mojotrees does not implement. `cli/mojotrees` reads CSV, which is a different thing (section 13) |
| `Sequence` / batched construction | supported | An `lgb.Sequence`, a `mojotrees._sequence.Batches` of arrays or frames, or a list of them, streamed a batch at a time into the same native `Dataset` a dense matrix builds; `python/tests/test_preflight_readers.py` checks the chunk count and the fit |
| pyarrow tables and arrays | supported | Tables, record batches, and arrays (features, labels, weights, a dictionary column as categories) through `python/mojotrees/_arrow.py`, dispatched from `_arrays`; predicts identically to the same data as a numpy array (`python/tests/test_wired_features.py`, which skips when pyarrow is absent; the `pytest` environment installs it) |
| polars frames | supported | Frames and series through `python/mojotrees/_polars.py`, the same way; the same test |
| `datatable` frames | unsupported | LightGBM's own support is legacy; not worth matching |
| NaN as missing | supported | `use_missing`, LightGBM's parameter, on by default. A reserved bin, a per-node default direction, and the same direction at predict time. NaN is the missing marker in sparse input too, wherever it is stored |
| `zero_as_missing` | deferred | The parameter is not offered. Sparse input already fixes the semantics at LightGBM's default (an implicit zero is the number zero), so what is missing is the ability to ask for the other behavior |
| Infinities in `X` | different | Rejected. LightGBM's own scikit-learn wrapper validates with `force_all_finite="allow-nan"`, which permits infinities into the C++ binner; mojotrees refuses them rather than binning them as extreme finite values by accident |

## 7. Core parameters

All 141 canonical parameter names in LightGBM 4.7.0, from
`lightgbm.basic._ConfigAliases` as enumerated in the version 1 audit.
Aliases are omitted; section 2 lists the aliases mojotrees accepts.

### Core and objective

| Parameter | Status | Notes |
|---|---|---|
| `objective` | partial | Section 8 |
| `boosting` | supported | `gbdt`, `goss`, `dart`, `rf`. Dart and rf train dense single-output models on the CPU; sparse input, `eval_set`, a callable objective, multiclass, and ranking refuse them by name rather than fitting gbdt silently. Section 0 |
| `data_sample_strategy` | partial | `bagging` and `goss` are both implemented, selected through `boosting="goss"` (LightGBM 3.x spelling) rather than through this newer parameter |
| `num_iterations` | supported | Spelled `n_estimators` in Python, `BoosterParams.n_rounds` in Mojo |
| `learning_rate` | supported | |
| `num_leaves` | supported | |
| `tree_learner` | partial | `serial`, `data`, `feature`, `voting` (and the `_parallel` spellings), with `num_machines` as the world size; every estimator takes them and `_mojotrees.distributed_train_local` trains the world in this process over `LocalCollective`. Feature parallel equals serial training bit for bit; voting parallel is deterministic and inexact by design. `partial` because the world never leaves the process: no transport ships, so `num_machines` is a thread-local rank count and no scaling claim is made (`docs/DISTRIBUTED_STRATEGIES.md`, `tests/test_distributed_strategies.mojo`, `python/tests/test_wired_features.py`) |
| `num_threads` | different | `MOJOTREES_NUM_WORKERS`, see section 2 |
| `device_type` | supported | Accepted as `device` with `device_type` as an alias. Values differ: mojotrees has `cpu`, `gpu`, `auto`; LightGBM has `cpu`, `gpu`, `cuda`. One portable GPU backend, so no vendor split, and `auto` is an addition. `gpu` now covers multiclass as well as single-output training (`gpu_supports_outputs` in `src/mojotrees/device.mojo` admits every output count), and on an M4 the GPU wins multiclass by 1.63x at 465,000 x 54 over 7 classes (`bench/results/profile_2026-08-15/RESULTS.md`). The module docstring that used to say multiclass was CPU only no longer does; that gap is closed |
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
| `bagging_by_query` | different | Always on for the ranker. A half-sampled query would be normalized against a maxDCG no served ranking ever had, so mojotrees does not offer the row-sampling variant |
| `feature_fraction` | supported | |
| `feature_fraction_bynode` | supported | Applied on both backends: `src/mojotrees/train_gpu.mojo` calls the same `select_node_features` the CPU grower calls |
| `feature_fraction_seed` | supported | |
| `extra_trees` / `extra_seed` | partial | Randomized split thresholds: one drawn threshold per feature, keyed by (`extra_seed`, tree index, node, feature). `src/mojotrees/split.mojo` calls `extra_threshold_index` and `src/mojotrees/params.mojo` parses both keys, so the parameter string reaches it. Numerical thresholds only, and it raises for a categorical feature rather than ignoring it. No Python estimator parameter. Section 0 |
| `early_stopping_round` | supported | `fit(early_stopping_rounds=)`, and `train_with_valid` / `train_with_metrics` in Mojo |
| `early_stopping_min_delta` | supported | `fit(min_delta=)`. Same strict-improvement rule as LightGBM |
| `first_metric_only` | different | Every metric flagged for early stopping is watched, and the ensemble is truncated to the best round of the **primary** metric on the **first** validation set. LightGBM truncates to the best iteration of whichever pair ran out of patience first, which makes the kept model depend on scheduling. `mojotrees.cv` does take a `first_metric_only` argument, because a fold history has no primary-metric truncation to fall back on |
| `max_delta_step` | partial | Fixed at LightGBM's Poisson value (`poisson_max_delta_step`, 0.7) inside the Poisson objective. The general rule is live for every objective now: `finish_leaf_output` caps leaf outputs in `src/mojotrees/split.mojo` and `src/mojotrees/tree.mojo`, and `src/mojotrees/params.mojo` parses the key. `partial` because no Python estimator parameter reaches it. Section 0 |
| `lambda_l1` | supported | LightGBM's `ThresholdL1` soft thresholding, applied to split gains and leaf values |
| `lambda_l2` | supported | Same default as LightGBM, 0.0, as of 2026-08-16; it was 1.0 here before that, which was the last tree default in this library that was not LightGBM's. `tools/check_parity.py` asserts every stock default now, on the Mojo and the Python surface, so this cannot drift again without failing a gate. Fits that did not set it explicitly moved: `lambda_l2` is the denominator of the Newton step |
| `linear_tree` | partial | Linear leaves: `BoosterParams.linear` (`src/mojotrees/linear_tree.mojo`) from the Mojo API, the parameter string, and `linear_tree=` on the estimators; each grown tree's leaves are refit to an affine function of the numerical features on the branch after growth, the sidecar is scaled with the learning rate, truncated by early stopping, evaluated on the raw row at prediction, and written as format v5 only when active (`tests/test_linear_tree.mojo`, `python/tests/test_wired_features.py`). `partial`: the binned-only trainers, `predict_contrib`, GPU prediction, continued training on a linear booster, LambdaRank, and DART with linear leaves refuse by name; `dump_model` shows the constant fallback rather than `leaf_const` / `leaf_coeff`; `lgbm_model_io` refuses `is_linear=1`; not compared against LightGBM numerically |
| `linear_lambda` | supported | With `linear_tree`; the ridge penalty on the leaf coefficients, nonnegative |
| `min_gain_to_split` | supported | The gain floor is live: `src/mojotrees/split.mojo` rejects a candidate through `passes_min_gain` before it can win, and `src/mojotrees/params.mojo` parses both `min_gain_to_split` and LightGBM's `min_split_gain` alias. Every caller reaches it: the Mojo API, the C ABI, the CLI, and the Python estimator, which takes both spellings and sends the resolved value on every fit (`python/mojotrees/sklearn.py`). Section 0 |
| `drop_rate` / `max_drop` / `skip_drop` / `xgboost_dart_mode` / `uniform_drop` / `drop_seed` | supported | DART parameters, on `boosting_dart.DartParams` at LightGBM's semantics (`docs/DART.md`) and reachable from the estimators and the parameter string. `uniform_drop` defaults to True here where LightGBM defaults to False; both rules are implemented |
| `top_rate` / `other_rate` | supported | GOSS. Same defaults, same `\|grad * hess\|` importance, same warmup rule, with `goss_warmup_rounds` exposing the warmup count |
| `min_data_per_group` | supported | Categorical |
| `max_cat_threshold` | supported | Categorical |
| `cat_l2` | supported | Categorical |
| `cat_smooth` | supported | Categorical |
| `max_cat_to_onehot` | supported | Categorical. **Was off by one against LightGBM; corrected 2026-08-16 by the `wide-categorical-bins` lane.** LightGBM's test is `num_bin <= max_cat_to_onehot` (`src/treelearner/feature_histogram.cpp:183`), where `num_bin` is `kept_categories + 1` because `src/io/bin.cpp:456-460` pushes a dummy bin at index 0 and sets `num_bin_ = 1` before admitting any category, so LightGBM one-hots `max_cat_to_onehot - 1` real categories. `src/mojotrees/categorical.mojo` compared `n_categories` and so one-hotted one more; it now compares `n_categories + 1` and the two agree at every value. The default 4 did not move. See `docs/design/CATBOOST_CATALOG.md` A16, which also records why CatBoost's `one_hot_max_size` is a **different** boundary (`<=` on real categories, no `+1`) and must not be collapsed onto this one |
| `top_k` | supported | Voting parallel's per-rank vote count, with `tree_learner="voting"`; section 11 |
| `monotone_constraints` | supported | Per-feature -1/0/1. The guarantee holds at any feature value, not only on the training data. Every trainer records the constraints it fit under on the booster it returns, which is what makes the guarantee survive a round trip and what continued training compares against to refuse a mid-flight change. That passthrough is a defaulted argument, so omitting it compiles and yields a model that silently reports no constraints; check 10 in `tools/check_parity.py` fails the build on any construction that omits it, because the omission has already happened across several trainers once and there is nothing at a call site to notice an absence |
| `monotone_constraints_method` | different | One method. LightGBM's `basic`/`intermediate`/`advanced` choice is an artifact of three implementations; mojotrees's bounds propagation is the exact one. `src/mojotrees/tree_parameters_extra.mojo` parses the parameter name for a future caller and nothing reads it |
| `monotone_penalty` | partial | Depth-scaled penalty on constrained splits, live through `apply_monotone_penalty` in `src/mojotrees/split.mojo` and parsed by `src/mojotrees/params.mojo`. No Python estimator parameter. Section 0 |
| `feature_contri` | deferred | Per-feature gain multipliers. The rule exists in `src/mojotrees/tree_parameters_extra.mojo` and no scan applies it; the comments around the per-feature scan in `src/mojotrees/split.mojo` name where it would be charged, which is not the same as charging it. Section 0 |
| `forcedsplits_filename` | unsupported | File-based configuration surface. `src/mojotrees/tree_parameters_extra.mojo` can parse the file's contents into a validated forced-split tree, but reading a file is not something mojotrees does |
| `refit_decay_rate` | deferred | With `refit`, task 14 |
| `cegb_tradeoff` / `cegb_penalty_split` | supported | Cost-effective gradient boosting, the two scalars. Parsed by `src/mojotrees/params.mojo`, carried on `TreeParams.extra.penalties.cegb`, and charged at `split._feature_gain` through `cegb.CegbNodeCosts`. The split cost needs only the node's row count, so every grower that routes through `tree._search` applies it. `docs/CEGB.md` |
| `cegb_penalty_feature_coupled` / `cegb_penalty_feature_lazy` | partial | The two per-feature vectors, implemented in `src/mojotrees/cegb.mojo` and charged against the per-ensemble `CegbLedger` that `boosting._boost_rounds` owns and `tree.grow_tree_with_cegb` threads. Mojo-API-only, because a whitespace-separated parameter string cannot carry a per-feature vector any more than it can carry `monotone_constraints`; no Python estimator parameter reaches either. A grower with no ledger (sparse, distributed, GPU, and the ranking/RF/DART trainers) refuses them by name through `cegb.check_cegb_grower_support` rather than ignoring them, and a resumed `train_more` refuses them too, since the ledger is not in the model file. `docs/CEGB.md` sections 8 and 10 |
| `path_smooth` | partial | Leaf-value smoothing toward the parent, live through `finish_leaf_output` in `src/mojotrees/split.mojo` and `src/mojotrees/tree.mojo`, and parsed by `src/mojotrees/params.mojo`. No Python estimator parameter. Section 0 |
| `interaction_constraints` | supported | LightGBM's per-branch allowed-feature rule, including the sharp edge that a feature in no group is never split on |
| `verbosity` | different | Training is silent. There is no logging layer to turn up |
| `input_model` / `output_model` / `saved_feature_importance_type` / `snapshot_freq` | unsupported | File-based configuration surface. `save()`/`load()` cover the model itself, and `cli/mojotrees` has its own `--model` flag |
| `use_quantized_grad` / `num_grad_quant_bins` / `quant_train_renew_leaf` / `stochastic_rounding` | deferred | Quantized-gradient training. Interesting for the GPU path; nothing depends on it today |

### Dataset and IO

| Parameter | Status | Notes |
|---|---|---|
| `max_bin` | supported | Default 255 |
| `max_bin_by_feature` | deferred | Per-feature bin counts. Straightforward once the binner takes a vector; low demand |
| `min_data_in_bin` | partial | A `fit_bins` argument (`src/mojotrees/binning.mojo`) **defaulting to LightGBM's 3**, honored in both branches of LightGBM's `GreedyFindBin`, verified against `src/io/bin.cpp` on LightGBM master: with no more distinct values than ordinary bins the levels are merged in order until each bin holds the minimum, and past the bin budget the budget itself shrinks to `max(1, min(max_bin, n / min_data_in_bin))`. Also enforced for categorical features (`min_data_per_group` governs the sorted search). `partial` for two reasons: the per-level greedy LightGBM runs *inside* the shrunken budget (its `is_big_count_value` rule and its recomputed target bin size) is still not implemented, so the two diverge on a column with more distinct values than bins; and the sparse binner (`sparse.fit_bins_csc`) does not take the argument, so **a sparse matrix and its dense form no longer bin identically at the default** -- the dense side merges rare levels and the sparse side does not. That is a live gap and the sparse fit is the side that is wrong. `min_data_in_bin=1` is still reachable at the Mojo API and is still edge for edge the binning that predates the option (`tests/test_binning.mojo`). No parameter-string key and no Python estimator parameter reaches it |
| `data_random_seed` | partial | A `fit_bins` argument seeding the `bin_construct_sample_cnt` sample below, defaulting to LightGBM's 1. It is read at the default now, because the sample is drawn at the default; it is a fixed constant and never a clock or a global, so the sampled fit is the same fit on every machine and at every `MOJOTREES_NUM_WORKERS`. The stream is mojotrees's own counter-based splitmix64, not LightGBM's 32-bit LCG, so the same seed does not select the same rows |
| `bin_construct_sample_cnt` | partial | A `fit_bins` argument (`src/mojotrees/binning.mojo`) **defaulting to LightGBM's 200,000**: the bin edges are fit from a sample of the rows drawn once per fit and shared by every feature, as LightGBM's are. The sample is exactly `bin_construct_sample_cnt` ascending row indices from a counter-based splitmix64 selection sample, identical at every `MOJOTREES_NUM_WORKERS`. A matrix of fewer rows than the count is not sampled at all. `partial` because the sample is not LightGBM's sample for a given seed, so the two libraries fit their edges from different rows above 200,000 and their splits diverge there for that reason alone; and because no parameter-string key, no Python estimator parameter, and not the sparse binner reaches it -- `sparse.fit_bins_csc` still fits from every row, which is the second half of the dense/sparse gap named in the `min_data_in_bin` row. See `subsample_for_bin` in section 2 |
| `is_enable_sparse` | different | There is no toggle: the input type decides. A SciPy sparse matrix takes the sparse path (section 6) and a dense matrix takes the dense one, and neither converts to the other |
| `enable_bundle` | partial | Exclusive Feature Bundling. `src/mojotrees/boosting.mojo` fits one plan per training run with `prepare_bundling` and hands the bundled matrix to every tree, and `src/mojotrees/params.mojo` parses `enable_bundle` and `max_conflict_rate`. No Python estimator parameter reaches it. Section 0 |
| `use_missing` | supported | |
| `zero_as_missing` | deferred | Section 6 |
| `feature_pre_filter` | partial | **Implemented, and reachable from the Mojo API only.** LightGBM's `true` is a Dataset construction step: `BinMapper::FindBin` marks a feature trivial when `NeedFilter(cnt_in_bin, total_sample_cnt, min_split_data, bin_type)` finds no prefix of its bins leaving at least `min_split_data` rows on both sides, and `Dataset::Construct` then leaves every trivial feature out of `used_features` (`src/io/bin.cpp` and `src/io/dataset.cpp`, both read). `min_split_data` is `filter_cnt`, and `src/io/dataset_loader.cpp` (now read, in both `ConstructBinMappersFromTextData` and `CostructFromSampleData`) computes it as `static_cast<data_size_t>(static_cast<double>(min_data_in_leaf * total_sample_size) / num_dist_data)` -- `min_data_in_leaf` scaled to the bin-construction sample and truncated toward zero, which at the stock 20, 200,000 and 1,000,000 is **4**, not 20. mojotrees has all of it: `binning.filter_count`, `binning.need_filter` (both LightGBM branches, categorical included), `fit_bins(feature_pre_filter=..., min_data_in_leaf=...)`, a `usable` list on `BinMapper` that `transform` carries onto the `BinnedMatrix`, and `sampling.select_tree_features(..., usable)`, which draws from the survivors and sizes the draw by their count -- LightGBM's `ColSampler` does exactly that, `GetCnt(valid_feature_indices_.size(), fraction)` over `Dataset::ValidFeatureIndices()` (`src/treelearner/col_sampler.hpp`). Features are **not renumbered**: a dropped feature keeps its id, its column and its importance slot, which is LightGBM's behavior too, since `GBDT::FeatureImportance` sizes its vector by `max_feature_idx_ + 1 = num_total_features` and a dropped feature simply reads 0 (`src/boosting/gbdt_model_text.cpp`, `src/boosting/gbdt.cpp`). Three gaps remain and none is a difference in the rule: (a) the four growers still call `select_tree_features` without the pool, so a prefiltered fit narrows nothing until that one argument is added in `src/mojotrees/tree.mojo`, `src/mojotrees/tree_sparse.mojo`, `src/mojotrees/train_gpu.mojo` and `src/mojotrees/train_gpu_sparse.mojo`; (b) `params.TrainConfig` carries neither this flag nor `min_data_in_leaf` into binning, so `tree_parameters_extra.check_feature_pre_filter` still refuses `true` **from a parameter string**, for the same reason `forcedsplits_filename` is refused there while being fully implemented; (c) `fit_bins` defaults it to `False` rather than to LightGBM's `true`, because `False` is the fit that preceded the option and mojotrees at `False` also keeps one-bin features that LightGBM drops whatever the flag says. `tools/check_parity.py` now fails if any link of that chain is removed, rather than failing if the flag is accepted. A differential against LightGBM must still pin `feature_pre_filter=False` on the LightGBM side |
| `pre_partition` | deferred | Distributed, task 16 |
| `two_round` / `header` / `label_column` / `weight_column` / `group_column` / `ignore_column` / `parser_config_file` / `precise_float_parser` / `forcedbins_filename` / `save_binary` | unsupported | LightGBM's text-parsing parameters. mojotrees's own CLI has `--header`, `--label`, and `--weight` flags over its own CSV reader (`cli/README.md`); they are not these parameters and are not accepted in a parameter string |
| `categorical_feature` | supported | Indices, names, or `"auto"`. pandas `category` columns are label-encoded by the estimator, and the encoding is kept for prediction; the model file carries the category tables but not the labels, so a model read back from disk takes integer codes |
| `data` / `valid` / `config` / `task` / `convert_model` / `convert_model_language` / `output_result` | unsupported | File-based configuration surface |
| `histogram_pool_size` | different | Histogram memory is pooled per grower without a user cap |

### Objective-specific

| Parameter | Status | Notes |
|---|---|---|
| `boost_from_average` | different | A toggle since 2026-08-16, as in LightGBM, and `True` is the default under `lossguide` exactly as it is in LightGBM (`config.h:948`). It names behavior that was always there rather than adding any: `boosting._base_score` has computed the objective's optimal constant on every fit since the beginning, so leaving it alone is bit-identical to every fit made before the parameter existed. `False` starts every row at 0.0 and is honored by the dense single-output CPU and GPU round loops; every other trainer refuses it by name rather than starting from the label mean under it. Under `grow_policy='symmetrictree'` an unset value resolves **per objective as CatBoost does** (`options_helper.cpp:353-374`): on for RMSE, MAE, Quantile and MAPE, off for Logloss, CrossEntropy and MultiClass. Still always off for custom objectives (the framework does not know the link, which is LightGBM's rule too at `gbdt.cpp:331`); a custom objective sets its starting raw score explicitly with `base_score=`, a number or `"mean"` |
| `is_unbalance` / `scale_pos_weight` | partial | Both are in `src/mojotrees/class_weight.mojo` under their LightGBM names (`unbalanced_sample_weight`, `scale_pos_weight_rows`). Not constructor parameters on the Python classifier, where `class_weight={1: w}` is `scale_pos_weight` and `class_weight="balanced"` is `is_unbalance` up to a constant factor |
| `sigmoid` | partial | Supported for LambdaRank, as a `MojoTreesRanker` constructor parameter. Not exposed for the binary objective, which uses the standard logistic |
| `alpha` | supported | Huber transition point and quantile level, LightGBM's meanings |
| `fair_c` | supported | The fair loss's `c`, default 1.0. One trainer slot holds whichever scalar parameter the objective reads, and naming one that belongs to a different objective is an error rather than a silently ignored value |
| `poisson_max_delta_step` | different | Fixed at LightGBM's default 0.7 rather than exposed |
| `tweedie_variance_power` | supported | Tweedie's rho, in (1, 2), default 1.5. Outside that range it would no longer be the compound Poisson-gamma the objective assumes, so it is rejected rather than clamped |
| `lambdarank_truncation_level` | supported | Default 30, and also the maxDCG cutoff, as in LightGBM |
| `lambdarank_norm` | supported | Default on |
| `label_gain` | supported | LightGBM's default `2^i - 1` for labels 0..30 when unset; a user vector on `MojoTreesRanker(label_gain=)` routes the fit through `src/mojotrees/ranking_advanced.mojo`. Labels above 30 stay refused even with a longer vector; no LightGBM differential covers a custom gain vector |
| `lambdarank_position_bias_regularization` | partial | Unbiased LambdaRank with a `position` column (`Dataset(position=)`, `MojoTreesRanker.fit(position=)`), `src/mojotrees/ranking_advanced.mojo`. `partial`: refused together with `eval_set`, and no LightGBM differential exists for it, so numeric parity is not claimed |
| `objective_seed` | different | No objective in mojotrees draws random numbers |
| `reg_sqrt` | deferred | `sqrt`-transformed regression. Rare |
| `multi_error_top_k` / `auc_mu_weights` | deferred | With top-k multiclass error and `auc_mu`, neither of which is implemented |

### Metric

| Parameter | Status | Notes |
|---|---|---|
| `metric` | supported | Section 9. `fit(eval_metric=)` in Python, `MetricSuite` in Mojo, `metrics=` in `mojotrees.cv` |
| `metric_freq` / `is_provide_training_metric` | different | Metrics are evaluated every round on the validation sets only; the training set is not scored automatically. `mojotrees.cv(eval_train_metric=True)` is the one place a training score is offered |
| `eval_at` | partial | `ndcg_eval_at`, a single cutoff for the ranker's `score` and for its `ndcg`/`map` eval metrics. LightGBM takes a list; `ndcg_score` takes any cutoff, and `ndcg_at_cutoffs` in Mojo takes several at once |

### Network, GPU, and prediction

| Parameter | Status | Notes |
|---|---|---|
| `num_machines` / `local_listen_port` / `time_out` / `machine_list_filename` / `machines` | deferred | No transport is wired up. `src/mojotrees/collective.mojo` defines the contract a transport would implement, `src/mojotrees/distributed_transport.mojo` is an unintegrated implementation of one (section 0), and `docs/distributed.md` states what is and is not built. Task 16 |
| `gpu_platform_id` / `gpu_device_id` / `gpu_device_id_list` / `gpu_use_dp` / `num_gpu` | different | One portable backend, one device, Float64 host arithmetic with a fixed-point device reduction. There is no OpenCL platform to select, and no double-precision toggle because the reduction is integer |
| `force_col_wise` / `force_row_wise` | different | Both layouts exist. `MOJOTREES_CPU_BIN_LAYOUT=feature` is `force_col_wise` and `=row` is `force_row_wise`; unset is LightGBM's `auto`, one timed root build per fit with each builder, keeping the faster (`histogram.choose_bin_layout_timed`). Three differences from LightGBM. (1) Their two layouts accumulate in different orders and so can move the model at the ulp level; ours are asserted bit-identical at tree level, so the timed choice cannot change a fit's output, only its speed. (2) Their `auto` allocates both layouts unconditionally; ours refuses the row-major copy above a memory budget and degrades to feature-major (`binning.ROW_MAJOR_DEFAULT_BUDGET_MB`, 1 GiB, `MOJOTREES_CPU_ROW_MAJOR_MAX_MB` to move it, `MOJOTREES_CPU_ROW_MAJOR` to force either way). (3) Their `MultiValBin` widens to 2 or 4 bytes per feature per row when a feature exceeds 256 bins and never packs below 1; ours is capped at 1 byte and packs a feature of at most 16 realized bins into a nibble. Parallel dispatch is governed by `MOJOTREES_PARALLEL_MIN_OPS` in either layout |
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

| LightGBM objective | Status | mojotrees | Evidence |
|---|---|---|---|
| `regression` (l2) | supported | `SQUARED_ERROR`, Python `objective="regression"` | `src/mojotrees/boosting.mojo`, `tests/test_mojotrees.mojo` |
| `regression_l1` / `mae` | supported | `L1`, with LightGBM's `RenewTreeOutput` leaf-value replacement | `src/mojotrees/boosting.mojo`, `tests/test_objectives.mojo` |
| `huber` | supported | `HUBER`, `alpha` is the transition point. No leaf renewal, as in LightGBM | `src/mojotrees/boosting.mojo`, `tests/test_objectives.mojo` |
| `quantile` | supported | `QUANTILE`, `alpha` is the level, with weighted-percentile leaf renewal | `src/mojotrees/boosting.mojo`, `tests/test_objectives.mojo` |
| `poisson` | supported | `POISSON`, exp link, log-mean base score, `poisson_max_delta_step` in the Hessian. Python `objective="poisson"` | `src/mojotrees/boosting.mojo`, `tests/test_objectives.mojo` |
| `binary` | supported | `BINARY_LOGISTIC` | `src/mojotrees/boosting.mojo`, `tests/test_mojotrees.mojo` |
| `multiclass` (softmax) | supported | `train_multiclass` / `fit_multiclass`, one tree per class per round, on either backend | `src/mojotrees/boosting.mojo`, `tests/test_multiclass_model.mojo` |
| `lambdarank` | supported | `train_ranker` / `fit_ranker` / `MojoTreesRanker` | `src/mojotrees/ranking.mojo`, `tests/test_ranking.mojo` |
| custom (callable) | different | Single output only, weights applied by the framework, gradients validated every round, base score explicit through `base_score=`. CPU only from Python: the `fit_custom` binding takes no device, and `train_custom_gpu` is the Mojo entry point for a pre-binned matrix. See the README section on custom objectives for each difference and why | `src/mojotrees/objective.mojo`, `tests/test_custom_objective.mojo` |
| `mape` | supported | `MAPE`, gradient scaled by LightGBM's `1 / max(1, \|y\|)` label weight, median leaf renewal under those same weights. Python `objective="mape"` | `src/mojotrees/boosting.mojo`, `tests/test_objectives.mojo` |
| `fair` | supported | `FAIR`, with `fair_c` (the trainer's `alpha` slot). Python `objective="fair", fair_c=...` | `src/mojotrees/boosting.mojo`, `tests/test_objectives.mojo` |
| `gamma` | supported | `GAMMA`, exp link, log-mean base score, strictly positive labels required | `src/mojotrees/boosting.mojo`, `tests/test_objectives.mojo` |
| `tweedie` | supported | `TWEEDIE`, exp link, with `tweedie_variance_power` in (1, 2) (the trainer's `alpha` slot). Python `objective="tweedie", tweedie_variance_power=...` | `src/mojotrees/boosting.mojo`, `tests/test_objectives.mojo` |
| `cross_entropy` | supported | `CROSS_ENTROPY` (alias `xentropy`), logistic link with labels anywhere in [0, 1]. On the regressor, since its labels are soft targets rather than classes | `src/mojotrees/boosting.mojo`, `tests/test_objectives.mojo` |
| `cross_entropy_lambda` | different | Not implemented, and reported by name rather than as an unknown objective. It parameterizes the rate through `log1p(exp(raw))`, a different link, so it is not an alias of `cross_entropy` and cannot be reached by setting one | `src/mojotrees/params.mojo`, `python/mojotrees/__init__.py` |
| `multiclassova` | different | Not implemented, reported by name. One-vs-rest needs an independent binary model per class, which is a different trainer from the shared-softmax `multiclass` | `src/mojotrees/params.mojo`, `python/mojotrees/__init__.py` |
| `rank_xendcg` | different | Not implemented, reported by name. Out of v1 with the rest of the unbiased/alternative ranking objectives; `lambdarank` is the ranking objective provided | `src/mojotrees/params.mojo`, `python/mojotrees/__init__.py` |

GPU coverage: `train_gpu` covers every single-output built-in objective
above that shares the per-row gradient/Hessian interface,
`train_multiclass_gpu` covers softmax and is reached by `fit_multiclass`
and therefore by `MojoTreesClassifier(device="gpu")`, and
`train_custom_gpu` covers custom objectives from Mojo only. LambdaRank is
CPU only. Every GPU claim here has run on one device class, Apple M4; see
`docs/GPU_VALIDATION.md` and section 12.

## 9. Metrics

Every metric name accepted by LightGBM 4.7.0, verified the same way in the
version 1 audit.

| LightGBM metric | Status | Notes |
|---|---|---|
| `l2` | supported | Mojo `rmse` reports the root; the squared value is the training loss used for early stopping |
| `rmse` | supported | `src/mojotrees/metrics.mojo`, `tests/test_metrics.mojo` |
| `l1` | supported | Same file |
| `quantile` / `huber` | supported | Read the estimator's `alpha`, so they score the loss the objective trained on |
| `mape` | supported | LightGBM's `\|y - p\| / max(1, \|y\|)`, the same label weight the MAPE objective trains against |
| `fair` | supported | Reads `fair_c` |
| `poisson` | supported | `mu - y log mu` on the response scale |
| `gamma` / `gamma_deviance` | supported | The gamma likelihood and its deviance, which is 0 at a perfect prediction |
| `tweedie` | supported | Reads `tweedie_variance_power`; scoring at a different rho scores a different loss |
| `cross_entropy` / `kullback_leibler` | supported | Continuous labels in [0, 1]. KL is the cross entropy minus the labels' own entropy, so a perfect prediction scores 0 |
| `binary_logloss` | supported | `src/mojotrees/metrics.mojo` |
| `multi_logloss` | supported | Same file |
| `binary_error` / `multi_error` | supported | Under LightGBM's spellings, with `binary_accuracy` and `multiclass_accuracy` as their Mojo-only complements |
| `auc` | supported | Rank-based, with scikit-learn's tie handling |
| `average_precision` | supported | Step-wise precision-recall area, scikit-learn's rule for ties (no trapezoid interpolation) |
| `ndcg` | supported | Any cutoff, per query, averaged; `ndcg_score` from Python. An all-zero-label query counts as 1.0, as in LightGBM |
| `map` | supported | Binary relevance (any label above 0), AP@k divided by `min(k, relevant)`, and a query with nothing relevant counts as 1.0, matching this module's NDCG convention. `src/mojotrees/ranking.mojo` |
| `cross_entropy_lambda` | deferred | With the objective of that name, which is not implemented |
| `auc_mu` | deferred | The multiclass AUC generalization; needs the class-pair projection LightGBM builds, and no multiclass ranking metric is provided yet |
| custom metrics (`feval`) | supported | Mojo: `MetricSuite`, several metrics with a declared direction and early-stopping flag. Python: `fit(eval_metric=...)` with callables, and `feval=` in `mojotrees.cv`. Differences from `feval` are listed at the top of `src/mojotrees/custom_metric.mojo` |

Every metric marked `supported` above is selectable from Python as
`eval_metric="auc"` and scored by the same Mojo functions the Mojo API
exposes; the table of names, aliases, directions, and tasks is
`python/mojotrees/_eval.py`, mirrored by the metric codes in
`bindings/_mojotrees.mojo`. Two differences from LightGBM are deliberate: a
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
| callbacks in `cv` | partial | `mojotrees.cv(callbacks=)` runs them per fold, with the same reachability caveat as `cv` itself (section 0) |

## 11. Distributed modes

| LightGBM mode | Status | Notes |
|---|---|---|
| data parallel | partial | Designed and prototyped: row partitioning, local histograms, all-reduce, globally consistent splits, identical trees on every rank, deterministic failure agreement. Every rank runs in one process (`LocalCollective`), so nothing has crossed a network. `src/mojotrees/distributed.mojo`, `src/mojotrees/collective.mojo`, `tests/test_distributed.mojo`, `docs/distributed.md` |
| feature parallel | partial | Implemented over the in-process world (`tree_learner="feature"`): every rank holds every row and its own feature block, candidates are all-gathered per node, and the result equals single-node training bit for bit. Same in-process caveat as data parallel |
| voting parallel | partial | Implemented over the in-process world (`tree_learner="voting"`, `top_k`): local votes, one vote reduction, packed reduction over the elected features. Deterministic and inexact by design, as LightGBM's is. Same in-process caveat |
| a real transport (MPI, sockets, gRPC) | deferred | `src/mojotrees/distributed_transport.mojo` implements a session state machine and its own suite, and nothing imports it: `src/mojotrees/distributed.mojo` still takes a `Collective`, and `src/mojotrees/__init__.mojo` does not export the transport. Section 0, task 16. **No distributed performance or scaling claim is made anywhere** |
| Dask integration | deferred | `python/mojotrees/dask.py` is the client-side contract and cannot train; section 0, task 17 |
| distributed GPU | deferred | Out of v1 |

## 12. GPU and accelerators

Every row here has run on exactly one device class, Apple M4 with Metal.
`docs/GPU_VALIDATION.md` holds the record and the procedure for CUDA and
HIP, and no NVIDIA or AMD device has run this code.

| Capability | Status | Notes |
|---|---|---|
| GPU histogram construction | supported | Portable kernel, 2D grid, fixed-point Int32 reduction, two combine strategies that are bit-identical to each other. `src/mojotrees/histogram_gpu.mojo`, `tests/test_gpu_strategies.mojo` |
| End-to-end GPU training | supported | `train_gpu`, device-resident binned matrix and compacted per-leaf row ranges. `src/mojotrees/train_gpu.mojo`, `src/mojotrees/gpu_active_rows.mojo`, `tests/test_gpu_training.mojo` |
| Device-resident objectives | supported | Built-in objectives without row sampling generate gradients and advance raw scores on the device, so nothing per-row crosses the host boundary in a plain round; bagging and GOSS keep the host path because their samples are ranked host-side. `src/mojotrees/gpu_objectives_native.mojo`, `tests/test_gpu_objectives.mojo` |
| GPU split selection | supported | Off by default. `MOJOTREES_GPU_SPLIT_STRATEGY=device` (or `grow_tree_gpu(split_search=SPLIT_SEARCH_DEVICE)`) searches each node's histogram on the device and downloads one 136-byte record instead of the histogram. It keeps a histogram slot per live leaf, so a split builds only its smaller child and derives the sibling with a device subtraction, and searches both children in one launch pair: one histogram build, one wait, and 272 bytes per split. Measured on an M4 with `bench-train-gpu`, 100 trees, four CPU workers. At 50000 x 100 the device scan trains in 3.03 to 3.06 s over three runs against the host scan's 2.43 to 2.49 s, so it is about 24% behind at that size, which is the expected direction: the device scan trades a per-node histogram download for a per-node kernel launch, and that only pays once the accumulation is large enough to dominate the launch. At 250000 x 100 a single run put the device scan at 3.15 s against 3.22 s, which is one run per arm and 2% apart and so says the two are indistinguishable at that size rather than that the device scan is ahead; that pair has not been repeated. Take one run per arm as unusable here in general: a device arm whose three repeats spanned 3.03 to 3.06 s also produced a 6.12 s transient on an unrelated run. Reproduce or refute any of these with `pixi run bench-train-gpu 50000 100 reg 5 gpu-host,gpu-device`, which alternates the arms inside one process and prints each arm's spread beside the delta, and which reports a gap narrower than that spread as indistinguishable rather than as a number. Float32 gains can flip near-tie split decisions versus the host scan, and a few percent does not pay for that, so `SPLIT_SEARCH_AUTO` still resolves to the host scan. `MOJOTREES_GPU_SPLIT_RESIDENT=0` forces the older loop, which builds both children and waits twice per split (5.40 s on the same run). `src/mojotrees/gpu_split_search.mojo` |
| GPU multiclass | supported | `fit_multiclass` resolves the device and calls `train_multiclass_gpu`, and `gpu_supports` admits every output count, so `MojoTreesClassifier(device="gpu")` trains one tree per class per round on the accelerator. `src/mojotrees/model.mojo`, `tests/test_device.mojo`, `tests/test_gpu_objectives.mojo` |
| GPU prediction | partial | `Model.predict_batch` and `MulticlassModel.predict_batch` take a `device` and walk the trees on the accelerator; binning stays host-side, so both devices route every row to the same leaf. No longer Mojo-API only: `bindings/_mojotrees.mojo` registers `predict_batch`, `predict_proba_batch` and `gpu_predict_capability`, and `predict`, `predict_proba` and the ranker's `predict` take `device=` with `None` meaning the CPU. `partial` because contributions (`pred_contrib`) and sparse input have no device path and refuse an explicit `gpu` rather than running on the CPU and reporting otherwise. Not a LightGBM capability (`device_type` covers training only), so this is a mojotrees addition. `src/mojotrees/gpu_predict.mojo`, `tests/test_gpu_predict.mojo` |
| Persistent GPU session and scheduling | partial | `src/mojotrees/gpu_runtime.mojo` is exported and `src/mojotrees/histogram_gpu.mojo` borrows a `GpuSession`'s device context. The residency ledger, staging ring, and phase counters around it are exercised by `tests/test_gpu_runtime.mojo` rather than by a trainer |
| Apple-specific tiling policy | partial | `src/mojotrees/apple_gpu_policy.mojo` is read by `src/mojotrees/device_policy.mojo` (device profile, session memory estimate) and by `src/mojotrees/apple_histogram_policy.mojo` (block threads). It still decides no default launch geometry: the specialization level resolves to `SPEC_LEVEL_BASELINE` unless `MOJOTREES_GPU_HIST_SPECIALIZATION` asks otherwise, so `src/mojotrees/gpu_tiling.mojo` remains the policy in force. Section 0 |
| Sparse input on the GPU | partial | An explicit `device="gpu"` on sparse input trains on the accelerator through `train_gpu_sparse` (`src/mojotrees/train_gpu_sparse.mojo`): the CSC binned matrix stays device-resident and compressed, every node's histogram is accumulated from its stored entries with the implicit zeros recovered by subtraction, and the row and entry partitions run on the device. `auto` keeps the CPU because the path's crossover against the CPU sparse trainer is unmeasured. Not on it: exclusive feature bundling (refused by name), custom objectives, eval sets, and GPU prediction of sparse rows (`_refuse_gpu_sparse` still holds for prediction). Section 0 |
| CUDA (NVIDIA) validation | deferred | The source targets it and `tests/test_gpu_portability.mojo` pins the launch limits CUDA imposes, but **no NVIDIA device has run this code**. `.github/workflows/gpu-validation.yml` is the manual job that would produce a record |
| HIP (AMD) validation | deferred | Same |
| GPU speed vs CPU | different | Measured, and the answer depends on the shape. On an Apple M4 at 100 rounds, 31 leaves, 255 bins, squared error, training time with binning excluded: 1,000,000 x 50 is CPU 6.98s against GPU 3.58s, so the GPU is 1.85x; 250,000 x 50 is CPU 1.66s against GPU 1.89s; 50,000 x 50 is CPU 0.564s against GPU 1.63s. The second sweep repeats the first two at five repeats and agrees (5.942 against 3.756, and 1.649 against 1.967) and adds 2,000,000 x 50 at 13.483 against 6.093. Against LightGBM at 10 threads, our leaf-wise GPU arm is behind at every shape measured (1.92x, 1.36x, 1.17x) and our depth-wise GPU arm is ahead at 1,000,000 x 50 only, growing a different tree, with its accuracy unmeasured. The device carries roughly 1.5s of fixed cost per fit, so it loses to our own CPU below about a million rows; fitting the second sweep puts our intercept at about 1.42s against LightGBM's 0.44s, with the two libraries' marginal cost per row even at 2.33 to 2.46 microseconds. Multiclass is the other way round: 465,000 x 54 over 7 classes is CPU 25.47s against GPU 15.30s, so the GPU wins by 1.63x. `crossover_rules()` holds two rules from those records. The single-output rule is scoped to Metal on an M4 for unweighted squared error, single output, dense input, at 50 or more features. The multiclass rule, installed 2026-08-16 from the 465,000 x 54 x 7 record above, is scoped to Metal on an M4 for dense input at 54 or more features and two or more trees per round; it is scoped by trees per round rather than by objective code because that is what the trainers branch on and what every multiclass entry point declares, and its class count is bounded below at two and not above. Its row floor is `AUTO_GPU_MIN_ROWS`, a plain provisional constant at 250,000 rows, deliberately set below the 1,000,000-row shape the records cover: end to end against LightGBM stock+det the GPU arm is 1.18x ahead there while the CPU arm is 1.75x behind, so an `auto` that under-reaches costs more than the 14 percent the GPU loses by at 250,000, and the real crossover is scheduled to be measured at the end of the current feature work. Since 2026-08-16 it fires: `DeviceCapabilities.detect()` names the hardware from the accelerator this binary was compiled for (`profile_source=build-target`, warned as `build-target-hardware` because it is a build property and not a reading), and `parse_apple_generation` can parse `4-metal4`, the architecture string a Metal device actually reports, which it could not before. Every rule is scoped to an objective and `resolve_device` declares none unless one is passed, so a four-argument call still gets the CPU; the six trainer entry points that hold an objective now pass it, and `resolve_device_full`, `decide_device`, `decide_device_report` and the Python `device_selection.select_device` all do too. `MOJOTREES_DERIVATIVE_PRECISION=float64` is a separate and unconditional refusal: the GPU gradient upload narrows every per-row derivative to Float32, so `auto` routes to the CPU at every shape and an explicit `device='gpu'` raises. `MOJOTREES_AUTO_MIN_CELLS` remains the escape hatch for hardware no rule covers and reports `EVIDENCE_ENV` rather than a validated choice. `bench/results/sweep2_2026-08-15/RESULTS.md`, `bench/results/profile_2026-08-15/RESULTS.md`, `docs/GPU_VALIDATION.md` |
| Where a GPU round's time goes | different | Located and not yet fixed. The first Metal timeline (`docs/METAL_TIMELINE.md`) says the GPU is idle for 76.5% of a training span at 200,000 rows and 87.5% at 50,000, at the device's Maximum clock. The host blocks on 94.1% of blits and on 2 of 18,701 compute kernels, which is 32.1 serialization points per round; one blocking readback costs 606us of wall clock, of which 3.7us is the GPU moving bytes. Compute of every kind is 22.9% of a round. The stage profile's "histogram is 49.3%" is a share of attributed *device work*, not of the round, and the two numbers are both correct because their denominators differ |

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
| C API | partial | `capi/mojotrees.h` is a stable, versioned C ABI over dense training, prediction, save, load, and the three model accessors, with an error object and an ownership table (`capi/README.md`). It is mojotrees's own interface and deliberately narrower than LightGBM's `c_api.h`, which it does not implement and is not source compatible with. `tests/test_capi.mojo` runs in CI; the C-side test (`pixi run test-c`) does not, and no release artifact carries the library |
| command line tool | different | `cli/mojotrees` trains and scores CSV files over the Mojo API, with the same parameter string the C ABI takes. It is not LightGBM's config-file surface (`config`, `task`, `data`, `valid`), which stays `unsupported` in section 7, and it reads no configuration file. `cli/README.md`, `tests/test_cli.mojo` |
| source build from a clean checkout | supported | `pixi install && pixi run test`, run by `.github/workflows/ci.yml` on x86-64 and ARM64 Linux |

---

## Known gaps in this contract

Findings from the audit that are not LightGBM parity items but that make the
matrix less trustworthy than it looks. They are recorded here rather than
quietly fixed, because each is somebody's in-flight work:

1. ~~**The scikit-learn sparse tag contradicts the sparse path.**
   `python/mojotrees/_sklearn.py` reports `input_tags.sparse` False, and
   `python/tests/test_sklearn_integration.py` asserts that it does, while
   `fit` and `predict` accept SciPy sparse matrices. A scikit-learn utility
   that respects the tag will densify before calling mojotrees, which is
   the thing the sparse path exists to avoid.~~ Closed in `d03a75c`;
   `python/mojotrees/_sklearn.py` sets `tags.input_tags.sparse = True` and
   the test asserts True, with a second test that runs a sparse matrix
   through `cross_val_score` so the tag is checked against the path it
   promises rather than on its own.
2. ~~**The Python sparse tests cannot run in the environment CI uses.**
   `python/tests/test_validation.py` and `python/tests/test_contrib.py`
   guard their sparse cases with `pytest.importorskip("scipy.sparse")`, and
   `pixi.toml` does not list scipy under `[feature.pytest.dependencies]`.
   Both cases skip in `pixi run -e pytest test-estimators`, which is the
   only Python test run CI performs, so section 6's rows are `partial`
   rather than `supported`.~~ Withdrawn in this revision, because it was
   not true. scipy was already in that environment transitively, since
   scikit-learn requires it, and `pixi.lock` resolves scipy 1.18.0 under
   the `pytest` environment on all three platforms. The guarded cases have
   been running in CI all along. What was real is narrower and is now
   fixed rather than recorded: that coverage rested on somebody else's
   dependency, so a scikit-learn release that dropped scipy would have
   turned those cases back into silent skips with CI still green.
   `pixi.toml` declares `scipy` under `[feature.pytest.dependencies]`
   directly, which makes it a contract instead of an accident.
3. ~~**`colsample_bytree` and `colsample_bynode` are not accepted** as
   aliases even though the rest of the scikit-learn spellings are.~~ Closed
   in this revision; `_Base.__init__` takes both and section 2 says so.
4. **Two native modules are implemented, individually tested, and wired
   to nothing.** `src/mojotrees/backend.mojo` and
   `src/mojotrees/gpu_vendor_policy.mojo` have no importer outside
   `tests/` and no export from `src/mojotrees/__init__.mojo`. Their suites
   run in `pixi run test`, which makes them look supported from the test
   output alone. Section 0 scores each one.

   This item named six modules before 2026-08-15.
   `src/mojotrees/efb.mojo`, `src/mojotrees/inspection.mojo`,
   `src/mojotrees/lgbm_model_io.mojo`,
   `src/mojotrees/distributed_transport.mojo`,
   `src/mojotrees/apple_gpu_policy.mojo`, and
   `src/mojotrees/tree_parameters_extra.mojo` were
   wired by the integration round and each now has importers in
   `src/mojotrees/` or `bindings/`.
   `python3 tools/connectivity_audit.py` is the check, and its first
   section is the live list; this item names only what that list has held
   long enough to be a contract question rather than a lane in flight.
5. **The names moved; the policy sentence did not.** `cv` and `CVBooster`
   are in `mojotrees.__all__`, and so are `explain_device_choice`,
   `dump_model`, `trees_to_dataframe`, `trees_to_records`, and
   `get_split_value_histogram`, which `__getattr__` resolves out of
   `device_selection` and `inspection` on first access. That closes this
   item for the two modules it was written about.

   What is left is three submodules that `_LAZY_SUBMODULES` in
   `python/mojotrees/__init__.py` makes reachable as `mojotrees.dask`,
   `mojotrees.diagnostics`, and `mojotrees.lgbm_model_io` after a plain
   `import mojotrees`, and that export nothing at the top level on
   purpose: dask's estimators raise `DistributedNotAvailable`, and the
   LightGBM converter warns that no file a real LightGBM build wrote has
   been read here. Section 6.1 of `docs/COMPATIBILITY_POLICY.md` still
   reads "No other submodule is a supported import path", which is now a
   sentence about three modules the package deliberately answers for.
   Either it names them as supported-but-experimental or it says the
   attribute is not a promise; their rows stay `partial` or `deferred`
   either way, for reasons that are about what they can do, not about how
   they are reached.

   `python/mojotrees/_public_api_plan.py` is the record of all of it as
   data, with the decision behind each name. Nothing imports it, which is
   deliberate, and `python/tests/test_public_api_plan.py` compares every
   factual table in it against the package so the record cannot drift
   from the code without failing.
6. ~~**Two docstrings outrank their code and are wrong.** The module
   docstring of `src/mojotrees/device.mojo` says multiclass is CPU only,
   and the `prediction options` section of
   `python/mojotrees/__init__.py` says "There are no built-in validation
   metrics in the Python API yet".~~ Closed in this revision; neither
   sentence survives. `src/mojotrees/device.mojo` is now a re-export
   facade over `src/mojotrees/device_policy.mojo` and describes
   `gpu_supports_outputs` as covering "the class count for multiclass",
   and the claim about validation metrics is gone from
   `python/mojotrees/__init__.py`, which `python/mojotrees/_eval.py`
   always refuted.
7. **Split gains are absent from every dump**, because
   `src/mojotrees/serialize.mojo` does not write them and
   `bindings/_mojotrees.mojo` exposes no `split_gains` hook. Anything built
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
  `mojotrees` when the extension is built and by parsing
  `python/mojotrees/__init__.py` otherwise
- the public Mojo symbols those rows depend on are still exported from
  `src/mojotrees/__init__.mojo`
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
  landed in `mojotrees.__all__`, which is exactly what happened to `cv`,
  `inspection`, and `device_selection`. Reachability is not a judgment, so
  this check decides rather than asking
- every Mojo test suite cited here is run by a pixi task. A suite that is
  cited as evidence but never executed is not evidence; the exception list
  in the script must match the "Known gaps" section exactly, and it is
  currently empty

Run it with `python3 tools/check_parity.py` or `pixi run check-parity`. It
runs in CI on every push and pull request.

`tools/check_parity.py` checks *claims*. It does not compute reachability, and it
cannot tell you that a module in `src/mojotrees/` is connected to anything.
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
  reach after its name landed in `mojotrees.__all__`: `cv`,
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
  unregistered bindings; `bindings/_mojotrees.mojo` now registers the
  inspection, objective-registry, dataset, distributed, and `decide_device`
  entry points, so the Python fallbacks are compatibility paths for an
  older extension rather than the path. Class-batched GPU multiclass moved
  from `deferred` to `partial` on a real call site behind
  `MOJOTREES_GPU_CLASS_BATCH`. Packed-bin layout and hybrid leaf placement
  stayed `deferred` and gained evidence saying what of them is now reached,
  because an import is not a call and a report is not a placement.
- **v2 (2026-08-14)**: re-audited mojotrees's side against the tree at
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
- **v1 (2026-08-14)**: first audit, against LightGBM 4.7.0 and mojotrees at
  `6190f88`.
