# Consolidation round (started 2026-08-14, coordinator C0)

No feature work. One authority per fact, orphans dispositioned, behavior
frozen. Lanes K1..K12; this file is the one record for the round. K1 and K5
wrote to handoffs/connect_22_audit.md and K2 to
handoffs/migration_20_device_policy.md; every other lane's notes were folded
into the sections below and their files removed (see handoffs/INDEX.md for
the rule going forward: decisions live in tools/connectivity_audit.py's
CLASSIFICATION table, not in per-lane prose).

## K11 objective and metric code authority

Already achieved by the objective-registry lane before this round: the
objective codes are defined once in objective_registry.mojo and boosting.mojo
binds them back under the names its callers import; the METRIC_* codes are
defined once in metrics.mojo and the registry binds them (the import cycle
registry -> boosting -> metrics forbids the other direction). Both tables
were compared value by value; no value disagrees.

Done this round:
- tools/connectivity_audit.py no longer counts `comptime NAME = _NAME` over
  an imported `_NAME` as a second definition (13d87ef). Duplicate public
  names 90 -> 39; every survivor has two independent bodies.
- ranking.mojo binds LAMBDARANK from the registry (7bb92e6, value 7).

Deferred, with reasons:
- params.mojo `MULTICLASS = -1` duplicates the registry's; params.mojo was
  held by a live lane while this ran. Same one-line rebind as ranking.
- gpu_predict.mojo `METRIC_L2 = 0`, `METRIC_L1 = 1`, ... are the device
  kernel's own reduction codes with different values from the host codes and
  an explicit host->device mapping (`device_metric_for`). Not a duplicate
  table. A rename to DEVICE_METRIC_* would clear the audit collision with no
  value change, but train_gpu.mojo imports the names and was held by a live
  lane. Do the rename when that file is clean.
- boosting.objective_renews_leaves is a documented wrapper of the registry's
  function under the historical name; nine modules import it from boosting
  (four of them held by live lanes). Leave it; the rule has one body.
- linear_tree.mojo `_objective_renews_leaves` mirror: K10 (orphan-module
  mop-up).

## K9 distributed stack layering

    collective.mojo            Collective trait, zeros_f64 (exported)
        ^          ^
        |          |
    distributed_transport.mojo  schema digest, cost contract, checkpoint
        ^      ^   ^            record, RuntimeSpec, transport_available()
        |      |   |            (== False in every build)
        |      |   +---- distributed.mojo   data-parallel trainer (EXPORTED:
        |      |                             train_distributed, grow_tree_
        |      |                             distributed). The production path.
        |      +--- distributed_strategies.mojo  feature-/voting-parallel
        |               ^                         cores; require_strategy_
        |               |                         operational refuses both.
        +---- distributed_gpu.mojo  fixed-point exchange contract; imports
                                    FIXED_ONE/magnitude_sum from
                                    quantized_gradient (8e7fc07); require_
                                    distributed_gpu refuses it.

    python: dask.py, _dask_runtime.py are intentionally lazy (explicit
    submodule import only) and drive distributed.mojo through the bindings.

Answer to the K9 question: (a). distributed_strategies and distributed_gpu
are layers a landed transport would consume, not parallel designs and not
superseded by distributed.mojo, which implements only data parallel. Both
are self-gated and say so in their module docstrings. Disposition: park,
recorded in the audit CLASSIFICATION table as EXPERIMENTAL with the gates
named. The FIXED_ONE / magnitude_sum duplicate is resolved by import.
Nothing was wired (no feature work). `strategy_name` in
distributed_strategies vs gpu_tiling is a name collision between unrelated
vocabularies (parallel strategy vs histogram tiling strategy); left as is.

## K4 sequence (recorded by K4 in connect_21; audit table updated here)

sequence.mojo and python _sequence.py parked, connect later; both now carry
an owner and reason in the audit table instead of "unassigned".

## K4 sequence: native `sequence.mojo` vs python `_sequence.py`

Audit only, nothing edited. They are two layers of one design, not
competitors: `_sequence.py` is the Python recognition of `lgb.Sequence`
plus the Arrow/polars dispatcher `_arrow.py` / `_polars.py` are written
against (materialize-then-train, not bounded memory); `sequence.mojo` is
the native chunk protocol `external_memory.mojo` builds the two-pass binner
on (LightGBM's push-rows analogue). No function exists in both. Neither
meets the deletion bar in either direction. Both PARKED: connect
`sequence` via export + `external_memory` validation, `_sequence.py` via
connect_07's `_arrays.check_X` front and a chunk binding. Recorded, not
touched: `CancelToken` is defined in `sequence.mojo` (`cancelled`, `polls`,
`none()`) and `validation.mojo` (`cancelled`, `reason`, `live()`); merge
into one token carrying both when either module connects.

## K6 bindings boundary

- Deleted (e98d4eb): whole-model `predict`, `predict_raw`, `predict_proba`
  bindings; full-range twins of `predict_range` / `predict_proba_range`,
  which are what Python calls. No test or C ABI route. Verified with one
  extension build in a HEAD worktree plus a fit/predict smoke check.
- `binding_support.mojo` is not dead: every capability module the entry
  point imports imports it (`-I bindings`). Audit made transitive (428bea8).
- 19 exports are reached by string dispatch (`_eval._REGISTRY_HOOKS`,
  `inspection._hook` + `_multiclass`, `sklearn._predict_batch` entry/legacy
  pairs, `_dask_runtime` provider probes, `device_selection getattr`). Audit
  now counts quoted export names as reads (888c3f9).
- 22 exports remain uncalled and are kept as forward surface, one PENDING
  row each in `CLASSIFICATION` (LightGBM Dataset field readers, registry
  queries beside the hooks, distributed/transport status companions,
  inspection JSON/kind/version queries, basic_bindings startup contract).
  12 more are connect_07's, 5 connect_22's.
- Only exact-duplicate binding: `_f64_list` / `_int_list` / `_csc` / `_csr`
  in `_mojotrees.mojo` vs `binding_support`'s helpers (different error text,
  memcpy vs loop). Build-gated two-file edit; deferred (see Close-out).
- `split_gains` / `split_gains_multiclass` are named by inspection.py with a
  graceful None fallback and have no binding at all: connect_11's plumbing.

## K7 Python package split

Snapshot first (c66b997, `tools/api_snapshot.py --write` once on the
untouched tree; never regenerated). Then, one commit per extraction with
`--check` showing zero rows after each: `_environment.py` (087f1a7:
`gpu_available`, `build_info`, `show_versions`, provenance helpers),
`_fit_args.py` (ccfe358: the `_IMPORTANCE_TYPES` / `_DEVICES` /
`_BOOSTING_TYPES` vocabularies, `_as_iteration`, `_metric_specs`,
`_check_eval_arguments`, ...), `_ranking.py` (d7044e3:
`group_from_query_ids`, `ndcg_score`, ...). Every moved name is bound back
by an explicit `from ._x import (...)` so `mojotrees.<name>` and in-package
`from . import _x` are unchanged. Phase 2 needed a tools/ change (8e214a2:
the snapshot resolves `_Base` and each estimator against `sklearn.py`, and
`MOJOTREES_ABI_VERSION`, a C macro, stops counting as an env var), then
`_Base` + `MojoTreesRegressor` / `Classifier` / `Ranker` + the objective
code literals moved to `mojotrees/sklearn.py` in one commit (2e1b26a).
`__init__.py` 4,615 -> 545 lines, snapshot diff zero. pytest is not in the
default pixi env, so no python test file was run; import smoke and direct
calls of moved functions were. `_public_api_plan.py` kept (see Close-out).

## K8 GPU round and scheduler authority

Decision (47e2173, read-only): `train_gpu.mojo`'s leaf-wise grower is the
one growth architecture; `gpu_frontier`, `gpu_fused_round`,
`gpu_multiclass_batch`, `hybrid_leaf_scheduler`, `gpu_leaf_batching`
(through `histogram_gpu`) and the grow-policy module are consulted
planners, kept. The one competing grower, `gpu_levelwise`, was parked
"superseded only if connect_02 records per-level batched launches are not
wanted as a replacement grower"; connect_02 recorded exactly that the same
day and deleted it (f4651d1), renaming `levelwise_policy` to
`growth_policy` with `GrowthSchedule(policy)` making the leaf pick for every
grower. `docs/design/GPU_LEVELWISE.md` stays as the design record for the
batching; its `gpu_levelwise.*` names describe the removed prototype.
Step 3 (edits, all gated on connect_04's files, see Close-out): `MAX_ROWS`
single site across `gpu_active_rows` / `gpu_multiclass_batch` /
`gpu_predict` / `histogram_gpu`; trim `gpu_frontier`'s uncalled
`SpeculationLedger` / `speculative_order` / `verify_speculation` /
`leaves_per_launch` after confirming connect_04's pending edits do not pick
them up. No edit to `train_gpu.mojo`; the CPU/GPU differential tests stay.

## K10 orphan feature dispositions and duplicate rewires

Parked with a named unblocker, nothing deleted: `cegb` (trainer must accept
cegb params), `linear_tree` (Booster holds a LinearEnsemble; codes now from
`objective_registry`), `model_editing` (its `MODEL_EDITING_SUPPORTED=True`
is the claim, inspection's `False` ships; on connect, inspection re-exports
it), `ranking_advanced` (`fit_ranker` grows the params), `lgbm_model_io`
(binding + `LGBM_INTEROP_STATUS` flips), `gpu_categorical` ->
`gpu_sparse` -> `gpu_sparse_layout` chain (train_gpu accepts categorical
specs), `external_memory` (with `sequence`). `validation` is
remaining_12's, `alternate_boosting` / `boosting_dart` / `boosting_rf`
connect_17's. Bit-exact rewires so orphans import the authority instead of
mirroring it: `boosting_dart` (c6a39ae), `model_editing` (96184c5),
`ranking_advanced` (747b2fc) take `splitmix64` / `uniform` / `GOLDEN` from
`rng.mojo` (the latter two imported from `sampling`, which K1 removed, so
they did not compile before); `linear_tree` takes its objective codes and
renewal rule from `objective_registry` and its `LeafStats` is renamed
`LinearLeafStats` because it is not gpu_frontier's record (e3ed5ab);
`boosting_rf` takes `LAMBDARANK` from `objective_registry` (26f04f3,
8b8cc53); `model_dump` takes `_MAX_CATEGORY` from `categorical` (5f21ac8);
`model_editing` takes `_F64_MAX` from `inspection`. One scratchpad
compile-and-value check covered all seven modules; no repository test
reaches any of them. Deferred on purpose: `MODEL_EDITING_SUPPORTED` (the two
values disagree by design), `check_relevance_labels` (name collision, not a
duplicate; python `_validation.py:592` names validation's by string, so
rename on that side), `CancelToken` (K4), the byte-identical
`_stream(seed, index)` in bagging / goss / boosting_dart (needs `rng.mojo`).

## Close-out (2026-08-15)

Audit counts, `python3 tools/connectivity_audit.py`, same tool each time
except where a row says the tool changed:

| Snapshot | Findings | Native orphans | Duplicate public names |
|---|---|---|---|
| pre-round e3d2de6 | 292 | 21 | 119 |
| round start (K1, K4, K5 landed) | 265 | 20 | 90 |
| after K10/K11 audit rewires (13d87ef re-export rule) | 204 | 18 | 28 |
| close-out (transitive binding reach, root re-exports, string-dispatch reads: 428bea8, 888c3f9) | 138 | 18 | 28 |

The last two rows are mostly the audit getting more accurate rather than the
tree changing: `binding_support` was reached all along, the package root's
imports are its export surface, and 19 binding exports are called by string.
Every remaining orphan and uncalled export has an owner and a reason in
`CLASSIFICATION`; `docs/INTEGRATION_INVENTORY.md` renders the same table and
`tools/audit_integration.py` reports zero errors and zero gaps against it.
Nothing was deleted under the deletion bar except K6's three whole-model
predict bindings (e98d4eb) and K2's cuda/amd twins (f23bd1b); everything
else that is unreachable is parked with a named unblocker.

Decisions taken by C0:

- `python/mojotrees/_public_api_plan.py` stays. Four docs cite it and its
  own docstring says nothing should import it; it is documentation as data,
  classified EXPERIMENTAL under connect_07.
- The 28 remaining duplicate public names are recorded, not rewired: each
  pair sits in at least one parked module (`CancelToken` in sequence /
  validation, `MAX_ROWS` across four GPU modules held by connect_04,
  `LeafCandidate` in gpu_frontier / growth_policy, and so on). They are the
  next round's first list.

Gated, not done, with the exact unblocker:

- K8 step 3 (`MAX_ROWS` single site across gpu_active_rows /
  gpu_multiclass_batch / gpu_predict / histogram_gpu; gpu_frontier
  `SpeculationLedger` trim) and K11's deferred `gpu_predict`
  `DEVICE_METRIC_*` rename: `train_gpu.mojo`, `hybrid_leaf_scheduler.mojo`,
  `gpu_active_rows.mojo`, `histogram_gpu.mojo` are dirty in the shared
  checkout (connect_04 / hybrid costs, `train_gpu.mojo` mid-edit and not
  parsing at close-out). Runnable the moment `git status` shows them clean.
- K6's `_f64_list` / `_int_list` / `_csc` / `_csr` helpers in
  `_mojotrees.mojo` duplicating `binding_support`: same gate (needs an
  extension build).
- Stale doc pointers to `gpu_levelwise.mojo` / `levelwise_policy.mojo` in
  `docs/design/GPU_LEVELWISE.md` are the growth_policy lane's (f4651d1)
  historical text; `docs/CONNECTION_AUDIT.md`, `docs/C_API.md`,
  `docs/RANDOM_FOREST_MODE.md`, `docs/DISTRIBUTED_STRATEGIES.md` name files
  that were never created (unrun-test names and an old build path); left
  as they read.

## Integration round (2026-08-15)

Wiring lanes; one paragraph per lane.

**W1, boosting_type dart and rf (f41221e, cef4667, 80297ef).**
`alternate_boosting`, `boosting_dart`, and `boosting_rf` are exported from
the package root; `_mojotrees.fit` reads a `boosting` key and routes dart
and rf to `alternate_boosting.fit_boosting` (same `fit_bins`, same `Model`,
CPU only, a GPU device is refused), leaving gbdt and goss on the exact call
they made before; `MojoTreesRegressor` / `Classifier` take
`boosting='dart' | 'rf'` plus `drop_rate`, `max_drop`, `skip_drop`,
`xgboost_dart_mode`, `uniform_drop` (True, the non-uniform variant is
refused natively), `drop_seed`; rf ignores `learning_rate` and trains at
1.0 as LightGBM's RF does. Fit paths that bypass that binding (sparse,
`eval_set`, callable objective, multiclass, ranker) refuse the two modes by
name. `tests/test_alternate_boosting.mojo` (6 tests, run: gbdt through the
dispatcher bit-exact with `model.fit`, dart and rf finite and distinct from
gbdt, save/load round trip, rf refusals); extension built in a HEAD
worktree and exercised from Python. Not reached: dart/rf with `eval_set`
(dart early stopping is unsettled by design in `boosting_dart`; rf's
`train_forest_with_valid` exists natively but no binding), multiclass rf
(`train_forest_multiclass` exported, no binding), the sparse trainers, and
continued training (`train_boosting_more` exported, no binding).
`params.mojo` still lists `boosting` as Mojo-API-only for parameter strings.

W7 validation layer: `validation.mojo` is now the check every trainer runs.
`trainset._check_columns` / `_int_labels` / `_relevance_labels`,
`sparse._check_compressed`, `params._validate`, `callback.check_resettable`,
and `boosting._check_sample_weight` call `check_shape`, `check_column_length`,
`check_group_counts`, `check_categorical_features`, `check_class_codes`,
`check_relevance_labels`, `check_compressed`, `check_booster_ranges`,
`check_max_bin`, and `check_weights` instead of their local copies (same
conditions, plus finiteness and the size ceilings; the messages keep their
old prefixes and now name the row and value; `boosting` keeps its own
length message because `tests/test_custom_objective.mojo` matches it). The
package root exports validation's entry points. Python:
`sklearn._Base._check_fit_structure` runs `_validation.check_shape`,
`check_length(y)`, `check_optional_length(sample_weight)` and the group
structure rules in all three `fit` paths, before the buffer conversions,
which would refuse the same inputs later with a vaguer message; the pandas
index-alignment check is deliberately not run (positional `y` stays
accepted). Left as they are: `boosting.mojo`'s multiclass / target-length
checks around lines 1112, 1197, 1286, 1682 (`_check_labels`-style, messages
tests match on) and `serialize`'s load path (`check_loaded_tree` is exported
but the loader still bounds nodes its own way; a serialize lane's edit).
`CancelToken` was left to W5. Tests: `tests/test_validation.mojo` (8, new,
reach every routed check through `Dataset`, `train_dataset*`,
`parse_params`), `test_callbacks`, `test_sparse`, `test_trainset` green in a
HEAD worktree; Python smoke via the in-tree package. Commits 116604c
18a6323 bff8d7c + audit rows.

W8 (Python readers for the uncalled bindings): 2e10584 deleted the metric
registry mirror in `_eval.py` (identical to the registry, compared spelling
for spelling; `resolve()` now asks `metric_code_of_name`, the `L2..MAP`
constants come off the registry snapshot, and `tools/api_snapshot.py`
derives `python.eval_metric_names` from `objective_registry.mojo`);
8b0280c routes regressor objective resolution and the alpha / fair_c /
tweedie_variance_power range checks through `objective_code_of_name`,
`objective_name_status`, `check_objective_param`, and reads
`registry_objective_unimplemented` / `registry_vocabulary`
(`MojoTreesRegressor._OBJECTIVES` stays as the frozen contract the snapshot
reads, checked against the registry by `python/tests/test_registry_readers.py`);
925fd91 `build_info()` gains `model_format_versions` and `startup`
(`startup_phase_contract`, `startup_environment`, `native_clock_ns`) and
`diagnostics.check_phase_contract()`; 5706537 `mojotrees.dask.check_machine_list`
(`distributed_check_machine_list`, also replacing the Python duplicate
address check) and `_dask_runtime.status_message` (`distributed_status_message`,
`transport_status_message`); 94f0e35 `Dataset.get_field` through
`dataset_field_length` + `dataset_copy_field`, `Dataset.feature_num_bin` /
`bin_upper_bounds` / `missing_bins`, `dataset_num_data` / `dataset_num_feature`
in `_from_handle`, `Booster.num_trees` native, `Booster.model_to_json`,
`Booster.file_kind` / `model_file_kind` (and `_load_path` sniffing the header).
Left uncalled: `gpu_validation_metric_matches_host` (the connect_07
gpu_validation surface has no Python reader at all) and the connect_07 /
connect_22 rows (`gpu_predict_capability`, `gpu_validation_*`, `efb_*`,
`extra_*`, `forced_splits_check`), which are other lanes'.

**W5, ingestion stack.** `sequence.mojo` and `external_memory.mojo` are
exported and run: `sequence.mojo` did not compile (a `to_bits` cast, a
`len(String)`, two partial field moves) and now does; `CancelToken` is one
struct in `sequence.mojo` with both spellings and `validation.mojo` imports
it; `tests/test_external_memory.mojo` (in `test` and `test-cpu`) runs the
checks `docs/EXTERNAL_MEMORY.md` section 14 called unrun: streamed edges
and bins equal `fit_bins` / `transform` bit for bit, chunks equal their
slices, the cache reopens through its checksums and refuses the wrong
mapper, `train_external` predicts identically to `train_dataset`, row
coverage rejects gaps and overlaps, a cancelled token stops a build.
`bindings/sequence_bindings.mojo` adds `dataset_chunks_begin` / `push` /
`num_data` / `finish` (a `ChunkAccumulator` type; column-major batches
appended and binned once into the same `Dataset` `dataset_create`
returns). Python: `_arrays` asks `_sequence`'s dispatchers first at
`feature_names`, `frame_categories`, `check_X`, `f64_vector`, and
`encode_labels` (lazy import; numpy/pandas/sequence/SciPy inputs take the
exact old path), so Arrow tables and record batches, polars frames and
series, `lgb.Sequence` objects, and `_sequence.Batches` are accepted by
`Dataset` and the estimators; `Dataset(batches)` streams through
`_sequence.stream_dataset` into the chunk binding one converted batch at a
time (`reference=` materializes instead); `lgb.Sequence` is read in
`batch_size` slices as LightGBM reads it. Smoke after `build-python` in a
HEAD worktree: numpy blocks, an `lgb.Sequence`, a pyarrow table, pyarrow
record batches, a polars frame, Arrow/polars label columns, and the
estimator on batches and on polars all predict identically to the
concatenated array. Not done: `compatibility/api_snapshot.json` needs
`--write` for the additive `sequence` / `external_memory` export rows
(entangled with other lanes' additive rows, so left to the coordinator);
`validation.mojo`'s `CancelToken` import is a re-export the audit flags as
unused because `__init__.mojo` still imports the name from `.validation`
(swap it to `.sequence` when `__init__.mojo` is free); pyarrow and polars
are not in the pixi env, so the Arrow/polars smoke ran against a scratch
install and no CI test covers them.

W6 (distributed strategies): `DistributedRunOptions.tree_learner` (serial |
data | feature | voting, LightGBM's names as codes) and `top_k` select the
grower; data parallel is untouched and default. New growers in
`distributed.mojo` over the `distributed_strategies.mojo` cores: feature
parallel (every rank holds every row, own feature block plus feature 0,
candidate all-gather per node, base score and loss read from one copy)
equals single-node training bit for bit; voting parallel (row partition,
top_k local votes, one vote reduction, packed reduction over the elected
features plus feature 0, siblings by local subtraction) is deterministic and
inexact by design. `require_strategy` no longer refuses the two modes;
`search_owned_features` refuses only extra_trees. Bindings:
`distributed_capability` reads `transport_available()`/`transport_validated()`
and lists `tree_learners`; `distributed_gpu_status()` reports
`distributed_gpu.mojo`'s closed gates; `distributed_train_local` trains a
`LocalCollective` world from the fit buffers, reached from every estimator
by `tree_learner=..., num_machines=N, top_k=K` (three new estimator
parameters, recorded in the API snapshot). `distributed_strategies` and
`distributed_gpu` are exported from `__init__.mojo` and left the orphan list.
Test: `tests/parallel/test_distributed_strategies.mojo` (4 pass);
`tests/test_distributed.mojo` (21) unchanged; Python smoke through the
estimator (feature == serial exactly at num_machines 2 and 3). Still gated:
distributed GPU's device path (needs `train_gpu.mojo`), any multi-process
run (`transport_validated()` stays False), the `tree_learner` and `top_k`
rows in `docs/LIGHTGBM_PARITY.md` (file held by another lane; both are now
implemented for the in-process world). No new handoff file.

**W2, linear trees.** `linear_tree.mojo` is reachable: `BoosterParams.linear`
(`LinearParams`, LightGBM's `linear_tree` / `linear_lambda`) switches it on
from the Mojo API, the parameter string (`params.mojo`), and the estimators
(`linear_tree=`, `linear_lambda=`, sent through the params mapping and read by
`_parse_params`). `custom_metric.train_with_callbacks` and
`train_multiclass_with_metrics` fit each grown tree's leaves after growth
from the round's gradients, hessians, and rows (`refit_linear_tree`), update
training and validation scores through the leaves (`add_tree_scores`;
`ValidSet` keeps its raw matrix), scale the sidecar under a learning-rate
schedule, and truncate it with early stopping; the raw matrix is copied only
when the switch is on, so constant-leaf runs are byte-identical to before.
`Booster` / `MulticlassBooster` carry `linear: LinearEnsemble`;
`Model` / `MulticlassModel` `predict*` evaluate it on the raw row;
`serialize` writes v5 with the `linear` section only when active (constant
models still write v4) and reads it back. Refused by name, not dropped: the
binned-only trainers (`boosting.train*`, `model.fit*`), `predict_contrib`,
GPU prediction, `train_more` on a linear booster (continued training needs
the raw matrix through the resume pass; `linear_tree.resume_raw_scores`
exists for it), and LAMBDARANK (`check_objective_compatible`). Not done:
`model_dump` does not emit `leaf_const` / `leaf_coeff` (a dump of a linear
model shows the constant fallback), `alternate_boosting.fold_weights_into_trees`
does not yet call `LinearEnsemble.scale_all` (DART + linear_tree is a W1
follow-up), and `lgbm_model_io` still refuses `is_linear=1`. Test:
`tests/test_linear_tree.mojo` (3 passed). Commits b353bfd 4ab237d 7d7985e and
the params.mojo / tree_parameters_extra.mojo follow-up. CEGB (Task A of the
lane) was found mid-integration by another session (0ef2115) and left to it.

W3, model editing and advanced ranking. `model_editing.mojo` is wired:
`bindings/model_editing_bindings.mojo` (registered from `_mojotrees.mojo`)
exposes rollback_one_iter / rollback_to, get_leaf_output / set_leaf_output,
shuffle_models, refit (parameters as a mapping, since def_function is proven
to eight arguments), and score_bounds, each with a `_multiclass` twin; leaf
values cross on LightGBM's shrunk scale. `Booster` in basic.py gains
LightGBM's rollback_one_iter, rollback_to, get_leaf_output, set_leaf_output,
shuffle_models, refit, lower_bound, upper_bound; inspection.py's
MODEL_EDITING_SUPPORTED is True and model_editing_support() returns the
native status. The purposeful duplicate is gone: inspection.mojo re-exports
model_editing's MODEL_EDITING_SUPPORTED and model_editing_status_json.
Test: tests/test_model_editing.mojo (5 tests) plus a Python smoke run.
`ranking_advanced.mojo` is wired through `advanced_ranking_requested`:
`trainset.train_dataset_ranker_advanced` and the fit_ranker /
train_dataset_ranker bindings route to train_ranker_advanced only when
label_gain, lambdarank_position_bias_regularization (with a position
column), pair_sampling_rate, or max_dcg_cutoff is set; a default run is the
old call byte for byte. MojoTreesRanker exposes the five parameters and
fit(position=...); Dataset(position=...) carries the column for the Booster
path. ranking_advanced's check_relevance_labels is renamed
check_labels_within_gain (validation.mojo's is the one python
_validation.py names). Test: tests/test_ranking_advanced.mojo (4 tests) plus
a Python smoke run. Left: fit with eval_set plus advanced ranking params is
refused with a message (custom_metric.train_ranker_with_metrics computes the
baseline lambdas; that file was held by W2 during this lane); labels above
30 stay refused by trainset._relevance_labels even with a longer label_gain;
no LightGBM differential for unbiased LambdaRank exists yet, so the parity
row must not claim numeric parity. The audit's unused-import check now
honors `X as _Y` aliases (it flagged aliased binding imports as unused).
compatibility/api_snapshot.json needs one coordinator --write for the new
additive Booster methods and Ranker parameters.

**D1, duplicates and the remaining binding readers.** The audit's "public
names defined by more than one native module" section went from 24 names to
the 6 that live in files another session holds. Same fact, now imported:
`gpu_sparse_layout.DEFAULT_MAX_NODES` from `gpu_objectives_native`;
`MAX_BINS` from `binning` in `gpu_gradient_stream`,
`gpu_histogram_specializations` (re-exported to the GPU lane) and
`gpu_output_planes` (`gpu_sparse` uses `SPARSE_MAX_BINS` directly);
`MAX_ROWS` from `gpu_active_rows` in `gpu_multiclass_batch` and
`gpu_predict`; `boosting.objective_renews_leaves` is `objective_registry`'s
re-exported; `inspection._F64_MAX` is `model_dump`'s. Different facts that
shared a name, renamed on the specific side: `unified_memory_policy`'s
transfer vocabulary (`TRANSFER_STATUS_*`, `N_TRANSFER_ROLES`,
`transfer_role_name`, `transfer_block_name`, `TRANSFER_EVIDENCE_*`),
`quantized_gradient` (`QUANT_REASON_*`, `describe_quantization_decision`),
`describe_cpu_policy` / `describe_gpu_policy`, `stream_layout_name`,
`startup_origin_name`, `session_phase_name` / `session_role_name` /
`N_SESSION_ROLES`, `sparse_verdict_name`, `HistogramWirePlan`,
`tree_learner_name` (the package root had exported two `strategy_name`s
since W6), `FrontierCandidate`, `backend.build_histogram_on`, and
`gpu_predict`'s device metric codes as `DEVICE_METRIC_*`. Commits 2142587,
73268e7 (hub, temp index), plus the inspection import. Verified by a
compile-and-value check over every touched module in a HEAD worktree,
`tests/test_backend_equivalence.mojo` (4 passed), and an extension build.
Python readers (commit below): `mojotrees.preflight` (`native_preflight`
runs `extra_params_check` + `efb_check` in every estimator fit before data
is converted; `check_forced_splits`; `unimplemented_option_message`, which
`basic.train` uses so a known LightGBM option gets the native "not
implemented, here is what it would take" text; `bundling_defaults` /
`bundling_knobs` so a knob set to `None` takes `efb_defaults`);
`device_selection.explain_predict_device` (`gpu_predict_capability`);
`dask.distributed_gpu_status` and `_dask_runtime.gpu_exchange_status`
(`distributed_gpu_status`, also in `describe_runtime`);
`_sequence.stream_dataset` checks `dataset_chunks_num_data` against the
batches' count; `mojotrees.gpu_validation.GpuValidation` wraps the
`gpu_validation_*` surface and `Booster.eval(..., device="gpu")` scores a
dense set on the accelerator (device kernel when it matches the host
definition, else the host suite). `python/tests/test_preflight_readers.py`
(8 checks, run against a fresh extension built at HEAD; the device eval ran
on the M4). Left for the GPU lane, all in files another session holds:
`MAX_BINS` and `MAX_ROWS` in `histogram_gpu.mojo`;
`unified_memory_policy.describe_decision` (imported by `histogram_gpu.mojo`;
rename to `describe_transfer_decision` there and in device_policy's
namesake); `tree.partition_rows` vs `distributed.partition_rows`
(`tests/parallel/test_gpu_active_rows.mojo` imports tree's; rename tree's
to `partition_split_rows` with that test); `train_gpu.mojo` importing
`METRIC_L1` / `METRIC_L2` from `gpu_predict` (switch to `DEVICE_METRIC_*`
and drop the two aliases). Not done: the "imports unused" rows the lanes
left behind (each lane's own file), and `validation.MAX_ROWS` (a different
ceiling, `1 << 44`, not flagged, but the same name; a rename to
`MAX_INPUT_ROWS` is a one-file edit for whoever next touches validation).

### Integration round close-out (2026-08-15, C0)

Nine lanes (W1 dart/rf, W2 linear trees, W3 model editing + advanced
ranking, W5 ingestion, W6 distributed strategies, W7 validation, W8 binding
readers, D1 duplicates + remaining readers, D2 follow-ups), plus cegb from a
peer session (0ef2115). All on origin/main; top of round 6b8dca9.

| Audit | pre-round e3d2de6 | after consolidation | after integration |
|---|---|---|---|
| findings | 292 | 136 | 45 |
| native orphans | 21 | 18 | 6 |
| python modules unreached | 8 | 8 | 4 |
| binding exports uncalled | 59 | 22 | 0 |
| duplicate public names | 119 | 28 | 6 |

Reachable now from the estimators or `Booster` / `Dataset`: `boosting_type`
dart and rf (binary and multiclass), `cegb_*`, `linear_tree` /
`linear_lambda` (v5 file section), `rollback_one_iter` / `rollback_to` /
`get_leaf_output` / `set_leaf_output` / `shuffle_models` / `refit` /
`lower_bound` / `upper_bound`, `label_gain` / position bias / pair sampling
/ `fit(position=)`, Arrow / polars / `lgb.Sequence` / batch inputs through
a chunk binding, `tree_learner` feature and voting over a local collective,
`Dataset.get_field` / `feature_num_bin` / bin edges, `Booster.model_to_json`
/ `file_kind`, `Booster.eval(device="gpu")`, native pre-flight checks and
the compiled objective and metric registries (Python mirrors deleted), the
validation layer under every trainer, dataset build, and estimator fit.
`pixi run check-parity`, `tools/api_snapshot.py --check`,
`tools/audit_integration.py` all green; the new tests are in the `test` and
`test-cpu` pixi suites.

What remains, and why (revised 2026-08-15 afternoon, after the GPU session
committed at aa49cb9 and the second cleanup pass below took the rows that
had been gated on its files):

- Three native orphans, one chain: `gpu_categorical` -> `gpu_sparse` ->
  `gpu_sparse_layout`. The GPU trainer refuses categorical specs, so the
  category-statistics kernels have no caller until `train_gpu` accepts
  them; that is a feature, not a wiring edit. `gpu_vendor_policy` (no
  CUDA/HIP device to consult it) and `backend` (the CPU/GPU equivalence
  reference) stay by design.
- The audit reports 9 findings and 0 PENDING outside that chain: no
  duplicate public names, no unused imports, no uncalled bindings.
- Still the GPU session's to decide, none of it a wiring gap: class-batched
  multiclass by default, hybrid leaves reach without the environment
  switches, the distributed GPU device path, the `gpu_frontier`
  speculation trim.
- No LightGBM numeric differential ran for the newly reached features; the
  parity rows say so.

Second cleanup pass (2026-08-15 afternoon, C0), once `git status` was clean:

- `histogram_gpu.mojo` imports `MAX_ROWS` from `gpu_active_rows` and drops
  its own `MAX_BINS` (only comments read it; `tests/test_gpu_portability.mojo`
  and `bench/bench_gpu_validation.mojo` now import `binning.MAX_BINS`) and
  the unused `STRATEGY_ATOMIC` / `STRATEGY_TILED`; `train_gpu.mojo` imports
  `DEVICE_METRIC_L1` / `DEVICE_METRIC_L2` and the `METRIC_*` aliases in
  `gpu_predict` are gone; `tree.partition_rows` is `partition_split_rows`
  (callers: `tests/parallel/test_gpu_active_rows.mojo`,
  `tests/test_cpu_parallel.mojo`, `bench/bench_profile.mojo`).
  `tests/parallel/test_gpu_active_rows.mojo` 28/28 on the M4.
- `_mojotrees.mojo`'s private buffer readers (`_f64_list`, `_int_list`,
  `_int_list_from_f64`, `_sparse_shape`, `_csc`, `_csr`) are retired for
  `binding_support`'s `f64_buffer`, `int_buffer`, `int_buffer_from_f64`,
  `csc_from_params`, `csr_from_params`; the bulk `unsafe_memcpy` read moved
  into `binding_support`, so it is the one implementation and the other
  binding modules get it too. The extension was rebuilt at this tree and
  `python/tests` (630) plus `python/test_python_api.py` pass against it.
- Stale tests that the fresh build exposed, all asserting a state a lane
  had since changed: `test_validation.test_y_length_must_match_x` matched
  the message `bff8d7c` replaced; `test_inspection` asserted model editing
  is not offered (W3 wired it; the two tests now check the `Booster`
  methods and the per-operation status, and `inspection.py`'s "Not
  offered" section is rewritten); `test_python_api.test_goss` and
  `test_lgbm_interop` (a `verbose` param, which `train` refuses by
  documented decision).
- `docs/INTEGRATION_INVENTORY.md`: the unregistered-binding-module table
  is empty (kept, so the audit keeps checking it).

Cleanup pass (2026-08-15 evening, C0), after the close-out above:

- LightGBM model-file interop wired: `bindings/lgbm_bindings.mojo` binds
  `lgbm_interop_status`, `lgbm_file_unsupported_reason`, `lgbm_import_file`,
  `lgbm_export_file` (report structs cross as dicts); `mojotrees.lgbm_model_io`
  is a lazy submodule; four partial field moves in `lgbm_model_io.mojo`'s
  file-level entry points (`return x.report^`) became `.copy()`, which is why
  those entry points had never elaborated before; `tests/parallel/test_lgbm_model_io.mojo`
  29/29 and `python/tests/test_lgbm_interop.py` (round trip through a
  fitted booster) added. Audit: the bindings root now seeds from the
  capability modules `_mojotrees.mojo` imports (`sibling_modules`), which is
  how a module reached only through `bindings/x_bindings.mojo` counts as
  reached.
- Dead imports stripped from 15 files (audit rows 22 -> 1, the survivor is
  in the held `histogram_gpu.mojo`); `device.mojo`, `inspection.mojo`, and
  `apple_gpu_policy.mojo` are exempted as documented re-export facades
  (`REEXPORT_FACADES`). After this pass the audit reports 15 findings:
  5 native orphans, 4 Python modules unreached (all lazy or by design),
  0 uncalled bindings, 5 duplicate names, 1 unused import.
- `device_policy.describe_decision` -> `describe_device_decision` (nothing
  called it; `unified_memory_policy`'s namesake keeps its `histogram_gpu`
  caller in the held file).
- Stale-path rows: `REPO_PATH` no longer reads `.h` off `.hpp` or a longer
  path's tail as one of ours; `docs/design/GPU_LEVELWISE.md` section 11,
  `docs/STARTUP_LATENCY.md`, and `docs/CONNECTION_AUDIT.md` stopped naming
  files that were removed.
- Parity rows rewritten from the tree: LightGBM model file interop, DART
  and random forest (section 0), `boosting` / `boosting_type`, the six DART
  parameters, `linear_tree` (new row) / `linear_lambda`, `tree_learner` /
  `top_k`, feature and voting parallel, `Sequence` (twice), pyarrow, polars,
  `Dataset.position`, `label_gain`,
  `lambdarank_position_bias_regularization`; the v1 scope paragraph. Status
  headers of `docs/LINEAR_TREES.md`, `docs/DART.md`,
  `docs/RANDOM_FOREST_MODE.md` updated. `pixi run check-parity` ok.
- Python-side reach tests for what only had native suites:
  `python/tests/test_wired_features.py` (linear leaves incl. v5 round trip,
  `tree_learner` data/feature/voting over `num_machines=2`, `label_gain` and
  `position`, Arrow and polars inputs; 8 pass against the 11:27 extension
  build) and dart/rf through `MojoTreesRegressor` in
  `python/tests/test_params.py` (whose stale "dart raises" row was removed).
