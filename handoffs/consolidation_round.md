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
