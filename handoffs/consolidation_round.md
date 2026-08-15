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
