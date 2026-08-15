# Consolidation K8, step 1: GPU round and scheduler authority (decision doc)

Read-only analysis, 2026-08-15. Facts from python3 tools/connectivity_audit.py
and the tree as it sits on disk (train_gpu.mojo, hybrid_leaf_scheduler.mojo,
gpu_active_rows.mojo, histogram_gpu.mojo are dirty under the connect_04 lane;
line numbers below are from that on-disk state and may shift when it lands).
No code was changed. No performance claims are made.

## Recommended authority

train_gpu.mojo's leaf-wise grower (with the depth-wise ordering that
bceec09 added through levelwise_policy.LevelSchedule) is the one growth
architecture. The planner modules it already consults stay: gpu_frontier
(child-by-subtraction rule, frontier records), gpu_fused_round (round
eligibility and the device-side tree router), gpu_multiclass_batch
(class-batched rounds and their guard), hybrid_leaf_scheduler (host/device
leaf placement), levelwise_policy (grow_policy vocabulary and level
schedule), gpu_leaf_batching (packed launch planning, consulted through
histogram_gpu). Exactly one module is a competing grower: gpu_levelwise.

## Module by module

| module | (a) design proposed | (b) what train_gpu uses today | (c) role | verdict |
|---|---|---|---|---|
| gpu_frontier (1204 lines) | Leaf-wise frontier records (LeafStats, LeafCandidate, FrontierLeaf, LeafWorkItem, CommitPlan, LeafFrontier), the smaller-child subtraction rule, and a speculative-order ledger for order-free searches | train_gpu.mojo:128 imports subtraction_builds_left; called at :1201 and :1734 (and mirrored by plan_split's tie rule at :1645). histogram_gpu.mojo:139 takes LeafWorkItem; gpu_active_rows.mojo:155 takes CommitPlan and LeafFrontier; gpu_leaf_batching.mojo:161 takes NO_SLOT, LeafFrontier, LeafStats, LeafWorkItem | consulted planner and shared record types | keep-as-consulted-planner. Sub-note: SpeculationLedger / speculative_order / verify_speculation / leaves_per_launch have no caller outside the module; candidates for a later trim, not this round. LeafStats also defined in linear_tree (K10 rewires linear_tree to import). |
| gpu_leaf_batching (2155 lines) | Packs several leaves' histogram builds into one launch: BatchPlan, GpuLeafBatcher, plan_batch, slots_for_budget | Not imported by train_gpu directly. histogram_gpu.mojo:146 imports GpuLeafBatcher, plan_batch, slots_for_budget; batcher list at :329/:484, slots_for_budget at :1260. train_gpu reaches it through GpuHistogramBuilder (train_gpu.mojo:155) | consulted planner (one level down) | keep-as-consulted-planner. verdict_name collides with gpu_sparse_layout by name only. |
| gpu_fused_round (1256 lines) | Device-resident round: RoundScales, MagnitudeReader, gradient/magnitude enqueue, GpuTreeRouter that advances every row's raw score on device, round_eligibility as the one answer to "can this round stay on the device" | train_gpu.mojo:129-134 imports ROUND_OK, GpuTreeRouter, round_eligibility, round_eligibility_reason; round_eligibility at :476-488, GpuTreeRouter list at :2002-2005, described at :75, :91, :452, :1962, :2155 | consulted planner plus a device helper the grower drives | keep-as-consulted-planner |
| gpu_multiclass_batch (1658 lines) | Class-batched softmax rounds: batched gradient, magnitude, zero, scatter, and shared/ranged histogram kernels; GpuClassBatch and MulticlassRoundGuard | train_gpu.mojo:135 imports GpuClassBatch, MulticlassRoundGuard; GpuClassBatch.for_plan at :2480, guards at :2487, :2624, :2683; histogram_gpu.mojo:140 imports GpuClassBatch (:811, :832, :844) | consulted planner plus kernels the grower drives | keep-as-consulted-planner. MAX_ROWS is defined in four modules (gpu_active_rows, gpu_multiclass_batch, gpu_predict, histogram_gpu); pick one site when connect_04's files are clean. |
| hybrid_leaf_scheduler (1697 lines, dirty) | Per-leaf host/device placement with cost model and replica verification: HybridCosts, HybridContext, LeafWork, plan_split, place_leaf, mode and decline vocabularies | train_gpu.mojo:156-171 imports the mode constants, HybridContext, HybridCosts, env_hybrid_costs, env_hybrid_mode, mode_name, plan_split; env_hybrid_mode/costs at :1450-1454, HybridContext at :1634, plan_split at :1648. gpu_runtime.mojo:82 imports HybridContext, LeafWork, decline_name, decline_reason, describe_context | consulted planner (the grower elects, the scheduler advises) | keep-as-consulted-planner; owned by connect_04 until its files are clean, do not touch |
| levelwise_policy (710 lines) | Grow-policy vocabulary (GROW_LEAFWISE, GROW_DEPTHWISE, parse/name/check), level admission math, LevelCandidate, LevelSchedule, LaunchProfile | Now REACHABLE (bceec09 today): train_gpu.mojo:193-198 imports GROW_DEPTHWISE, LevelCandidate, LevelSchedule, check_grow_policy; check_grow_policy at :751 and :1374, LevelSchedule at :861/:1121/:1543, LevelCandidate at :869/:1129/:1550. Also imported by tree.mojo:100, tree_sparse.mojo:86, params.mojo:56, distributed.mojo:124, bindings/_mojotrees.mojo:211, __init__.mojo:56 | consulted planner, shared by CPU and GPU growers | keep-as-consulted-planner. The audit's CLASSIFICATION entry ("Reached only from gpu_levelwise, itself unreachable", owner connect_02) is now false and should be dropped by C0 (tools/connectivity_audit.py). __init__.mojo imports GROW_DEPTHWISE, GROW_LEAFWISE, grow_policy_name, parse_grow_policy as re-exports; the "imports and uses none" row is the usual re-export false positive. |
| gpu_levelwise (779 lines) | A depth-batched grower: LevelNode, LevelFrontier, plan_level, decide_level, LevelCommit, child_sums, child_leaf_value; one launch per level, one host decision per level (docs/design/GPU_LEVELWISE.md) | Nothing imports it (audit: unreachable, owner connect_02). Its own docstring: "An experimental growth mode, not an optimization of the shipped one. The leaf-wise growers in tree.mojo and train_gpu.mojo are untouched by this lane and remain the only growth mojotrees trains with." | competing (replacement grower) | park-with-disposition. It is not superseded-delete-candidate yet: bceec09 delivers depth-wise ORDERING inside the leaf-wise frontier growers (one leaf commit at a time under LevelSchedule), while gpu_levelwise proposes level-batched launches and level commits (plan_level/LevelCommit); the shipped path does not demonstrably cover that, and no test reaches gpu_levelwise, so bars (b) and (c) are unmet. Disposition line: "parked; superseded only if connect_02 records that per-level batched launches are not wanted, at which point delete gpu_levelwise.mojo and docs/design/GPU_LEVELWISE.md together". |

Other train_gpu imports that touch growth structure but are not competing
designs: gpu_active_rows (device row ranges the frontier commits into;
dirty, connect_04), gpu_split_policy.decide_split_search and
gpu_split_search (search strategy, not growth), gpu_output_planes
.BatchEligibility, gpu_runtime (session and lifecycle), apple_histogram_policy
.ClassSchedule. All consulted planners; none proposes a second grower.

## Tests that reach each module

No test names gpu_frontier, gpu_leaf_batching, gpu_fused_round, or
gpu_multiclass_batch directly; they are exercised only through train_gpu /
histogram_gpu, i.e. tests/test_gpu_training.mojo, tests/test_backend_equivalence.mojo,
tests/test_gpu_objectives.mojo, tests/test_gpu_portability.mojo,
tests/test_gpu_strategies.mojo, tests/test_categorical.mojo, tests/test_missing.mojo,
tests/test_interaction.mojo, tests/test_custom_objective.mojo,
tests/parallel/test_gpu_active_rows.mojo, tests/parallel/test_gpu_runtime.mojo.
levelwise_policy: tests/test_grow_policy.mojo (12 tests, in `test` and
`test-cpu`). hybrid_leaf_scheduler: tests/parallel/test_hybrid_replica.mojo
(dirty, connect_04). gpu_levelwise: none.

## What a step-3 execution would edit (gate on connect_04 landing)

- tools/connectivity_audit.py CLASSIFICATION: drop the stale levelwise_policy
  entry; add gpu_levelwise as EXPERIMENTAL owner connect_02 with the parked
  disposition above. C0-owned; can be done now, does not wait on connect_04.
- MAX_ROWS single site: gpu_active_rows.mojo, gpu_multiclass_batch.mojo,
  gpu_predict.mojo, histogram_gpu.mojo (two of these are dirty). Wait.
- gpu_frontier trim of the unused speculation API: gpu_frontier.mojo only,
  but confirm gpu_leaf_batching / gpu_active_rows (dirty) do not pick it up
  in connect_04's pending edits. Wait.
- gpu_levelwise deletion, if and only if connect_02 records supersession:
  src/mojotrees/gpu_levelwise.mojo, docs/design/GPU_LEVELWISE.md, plus the
  audit entry. Not this round unless that note appears.
- No edit to train_gpu.mojo. The CPU/GPU differential tests stay untouched.

Sign-off requested from C0 on: (1) levelwise_policy is a consulted planner
now, not an orphan chain; (2) gpu_levelwise is parked, not deleted; (3) all
other modules keep-as-consulted-planner with no code change this round.

## connect_02 note (Aug 15 2026, later the same day): supersession recorded

Per-level batched launches, if ever built, are wanted as a histogram-phase
service underneath the shipped frontier growers (feed
`gpu_leaf_batching.mojo`'s multi-leaf kernels with a level's builds under
`grow_policy=depthwise`, or with both children under leaf-wise device
search), not as a replacement grower with its own commit path. That is the
supersession the K8 disposition line asked for, so:

- `src/mojotrees/gpu_levelwise.mojo` deleted (unused, untested; `git log`
  keeps it). `docs/design/GPU_LEVELWISE.md` kept, header rewritten: it is
  the design record for the batching, and its `gpu_levelwise.*` names now
  describe the removed prototype.
- `src/mojotrees/levelwise_policy.mojo` renamed `growth_policy.mojo` and
  trimmed to what the growers call. `LevelSchedule` became
  `GrowthSchedule(policy)`, which now makes the leaf-wise pick too (strict
  `>` best gain, ties to the lower slot, unchanged), so every grower
  (`tree.grow_tree`, `tree_sparse.grow_tree_sparse`, the three
  `train_gpu.mojo` loops) builds one `LeafCandidate` list and calls
  `next_leaf` once, with no policy branch of its own. Dropped as dead once
  `gpu_levelwise` went: `LevelwiseParams`, `LaunchProfile`,
  `leafwise_profile`, `levelwise_profile`, `prefilter_level`,
  `depth_permits_split`, `rows_permit_split`, `level_capacity`,
  `full_level_depth`, `effective_max_depth`, `UNLIMITED_LEVEL_NODES`.
- `tools/connectivity_audit.py` CLASSIFICATION: `gpu_levelwise` entry
  removed with the module. `docs/INTEGRATION_INVENTORY.md`,
  `docs/CONNECTION_AUDIT.md`, `docs/LIGHTGBM_PARITY.md`, README roadmap
  updated.
- `train_gpu.mojo` was touched (the three pick blocks and the import line
  only), against K8's "no edit" line; the hunks are disjoint from
  connect_04's hybrid edits in the same file and were committed by hunk.
