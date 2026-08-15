# Handoffs index

Four files. The rule from the consolidation round on: a lane does not get
its own handoff file. The durable state is the `CLASSIFICATION` table in
`tools/connectivity_audit.py` (one row per unreached module or uncalled
export: kind, owner, why, and what unblocks it), rendered into
`docs/INTEGRATION_INVENTORY.md` and gated by `tools/audit_integration.py`.
If a decision about connectivity is not a row there, it is not recorded.
Prose goes in the one round record below only when a table row cannot hold
it (a layering diagram, a deletion-bar argument).

| File | What it holds |
|---|---|
| `consolidation_round.md` | The consolidation round (2026-08-14/15): per-lane sections K4, K6, K7, K8, K9, K10, K11, close-out counts, decisions, and the exact edits still gated on connect_04 |
| `connect_22_audit.md` | Static tooling triage (K1, K5): which `tools/` scripts are wired into CI, which are one-shot, which were removed |
| `migration_20_device_policy.md` | `device_policy.mojo` as the one device authority; `gpu_cuda_policy` + `gpu_amd_policy` merged into `gpu_vendor_policy` (K2) |
| `remaining_14_validation_plan.md` | The focused validation planner; `tools/validation_plan.py` names its patches by this file |

Removed here, content folded into `consolidation_round.md`: the per-lane
`consolidation_K6/K7/K8/K10.md`, `connect_21_native_interfaces.md`, and the
closed `rename_15_swept_work.md`. Earlier, 21ff9fa removed 84 handoffs
whose work had landed. `git log` keeps all of them.

Related: `compatibility/api_snapshot.json` (frozen Python API,
`tools/api_snapshot.py --check`), `docs/LIGHTGBM_PARITY.md`
(`pixi run check-parity`).
