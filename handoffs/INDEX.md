# Handoffs index

One line per file. A handoff records what a lane found, decided, changed,
and left for a named owner. 21ff9fa (2026-08-14) removed 84 older handoffs
whose work had landed; `handoffs/consolidation_round.md` carries the
coordinator's record for the lanes that had nowhere else to write.

| File | Lane | What it holds |
|---|---|---|
| `consolidation_round.md` | C0 | Round record: K11 objective/metric authority, K9 distributed layering, K4 note, close-out counts, and what stays gated |
| `connect_22_audit.md` | K1, K5 | Static tooling triage: which `tools/` scripts are wired into CI, which are one-shot, which were removed |
| `connect_21_native_interfaces.md` | K4 | `sequence.mojo` vs python `_sequence.py` ownership audit; both parked, `CancelToken` duplication recorded |
| `migration_20_device_policy.md` | K2 | `device_policy.mojo` as the one device authority; `gpu_cuda_policy` + `gpu_amd_policy` merged into `gpu_vendor_policy` |
| `consolidation_K6.md` | K6 | Bindings boundary: exports dispositioned, whole-model predict twins removed, string-dispatch reads the audit missed, forward surface kept |
| `consolidation_K7.md` | K7 | Python package split: `__init__.py` to `sklearn.py` / `_environment.py` / `_fit_args.py` / `_ranking.py` with the API snapshot unchanged |
| `consolidation_K8.md` | K8 | GPU round and scheduler authority decision: `train_gpu` leaf-wise grower + consulted planners; step 3 (MAX_ROWS single site, frontier trim) gated on connect_04 |
| `consolidation_K10.md` | K10 | Orphan feature dispositions and duplicate rewires (cegb, linear_tree, model_editing, ranking_advanced, validation, lgbm_model_io, gpu_categorical chain, external_memory) |
| `remaining_14_validation_plan.md` | remaining-14 | The focused validation planner (`tools/validation_plan.py`) |
| `rename_15_swept_work.md` | rename-15 | Corrective record of work swept into 142e32f by the rename session; closed |

Related, outside this directory: `docs/INTEGRATION_INVENTORY.md` (rendered
from `tools/connectivity_audit.py`'s `CLASSIFICATION` table and gated by
`tools/audit_integration.py`), `compatibility/api_snapshot.json` (frozen
Python API, `tools/api_snapshot.py --check`), `docs/LIGHTGBM_PARITY.md`
(`pixi run check-parity`).
