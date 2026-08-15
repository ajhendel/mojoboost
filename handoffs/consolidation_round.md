# Consolidation round (started 2026-08-14, coordinator C0)

No feature work. One authority per fact, orphans dispositioned, behavior
frozen. Lanes K1..K12; this file is the coordinator's record for the lanes
whose original handoff files no longer exist (21ff9fa removed 84 handoffs on
2026-08-14). K1 and K5 wrote to handoffs/connect_22_audit.md; K4 wrote to
handoffs/connect_21_native_interfaces.md.

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
