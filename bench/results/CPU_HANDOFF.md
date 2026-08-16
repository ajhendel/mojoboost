# CPU orchestrator handoff

Written 2026-08-16, updated after the window. Branch `cpu-round-2`, based on
`perf-round-2` at `901e31d` where wave 4 is merged. Suite was green at 72/72
on the wave-4 base; it has NOT been re-run since the merge into
`perf-round-2`, which brought in the GPU campaign's work. Read `LANE_RULES.md` and `PROFILE_PROTOCOL.md` (sections C0 through
C9) first; this file is only state, not policy.

## Merged into cpu-round-1

Wave 4 landed in this order, each as its own `--no-ff` merge:

- `lane/lambda-l2-stock` — `lambda_l2` 1.0 to 0.0 in three literals. The one
  that mattered is `python/mojotrees/sklearn.py:48`, which is what
  `bench/real_data` actually fits; `TreeParams.default()` was cosmetic by
  comparison.
- `lane/leaf-estimation` — CatBoost Newton iterations, default 1.
- `lane/catboost-harness` — CatBoost as a peer column. Headline untouched.
- `lane/float32-switch` — `derivative_precision`, and the accuracy gate that
  could not fail (absolute 0.02 bounding a metric worth 0.0136).
- `lane/oblivious-cpu` — symmetric trees, cross-leaf reduction fused into the
  existing feature dispatch, ordinary binary representation.

Earlier in the round: dispatch snapshot per fit, alloc churn, field rename,
const-hessian, interleaved cells, stock defaults, harness comparator.

`lane/row-major-bins` was merged, found red against the interleaved cell,
and **reverted**. It is back with its lane with a four-cause diagnosis. It is
not in the tree.

## Done since this file was first written

- Float32 re-taken under lambda = 0. **The decision closes and the default
  stands**; see `bench/results/cpu_float32_lambda0_2026-08-16/RESULTS.md`.
- The window. End-to-end headline, task floor resolved at 50k, and four
  discarded runs listed by name; see `bench/results/cpu_window_2026-08-16/`.
- CatBoost reachable through `run.py`, `sparse_highdim` capped at smoke.
- Both catalog corrections were already in the file when I checked: multiclass
  `leaf_estimation_iterations` is 1, and the min-child rule is marked ours.
- `cpu-round-1` merged into `perf-round-2` at `901e31d`.
- `_multiclass_rf_gradients` needed nothing: it is a `def`, which raises in
  Mojo 1.0. That glue item was a non-issue and is struck.

## Owed, in order

0. **Head against `e8c0877` needs a lambda-matched form, not a caveat.**
   `lambda_l2` moved 1.0 to 0.0 this round and lambda sits in the split gain,
   so the two builds grow different trees and a wall clock would confound
   speed with tree shape. Run **head at `lambda_l2 = 1.0`** instead. The bench
   harness has no flag for it, so this needs one line of glue first.

1. **Float32 under lambda = 0. DONE, see above. Kept for the reasoning:** The 39/44 accuracy run that the Float32
   default rests on was taken under `lambda_l2 = 1`, and `H + lambda` is
   exactly the damping term that hid the Float32 loss. Re-take
   `imbalanced_binary` and `multiclass` under the now-stock lambda = 0 before
   anyone treats that decision as closed. Under lambda = 1 the CPU AP fell
   9.4 percent and AUC 0.006 on `imbalanced_binary`.
2. **The window.** End to end, ingestion plus binning plus training, 1M x 50
   against LightGBM stock + `deterministic=true` (label `stock+det`), five
   repeats, medians, verdict on medians. Arms in this order:
   `MOJOTREES_CPU_TASK_FLOOR` on/off FIRST, then row-major on/off, then head
   against `e8c0877`. 250k and 50k are reported rows, not decision rows. The
   lambda change is a caveat on any comparison against `e8c0877` and must be
   stated in the results file. Quiet box only: hold all compiles, both
   sessions.
3. **CatBoost harness glue.** Reachable through `bench/real_data/run.py`
   (widen the engine choices). Cap the `sparse_highdim` cell, smoke tier or
   bounded iterations, so a timeout cannot take the run's exit code. Add a
   CatBoost Lossguide row, their leaf-wise, `max_leaves` 31, beside
   SymmetricTree. State iterations and learning rate on every CatBoost row,
   since their auto lr depends on the budget.
4. **Catalog corrections** in `docs/design/CATBOOST_CATALOG.md`: multiclass
   `leaf_estimation_iterations` default is 1, not 10 (10 is Logloss); and the
   zero-contribution min-child rule is OURS, not CatBoost's, and is already
   marked not-verified-from-source in the oblivious entry.
5. **Merge `cpu-round-1` into `perf-round-2`** so the GPU session can rebase.
   Send them the SHA, plus the 12 moved `fit_bins_csc` sites in
   `tests/test_gpu_sparse.mojo` for their Metal check.
6. `lane/ordered-ts` merges when it reports.
7. Launch `wide-categorical-bins`, but only after the `max_cat_to_onehot`
   off-by-one lands as its own commit. Launch `score-function` now that
   `oblivious-cpu` has released `split.mojo`.

## Glue the lanes cannot reach

- `tree.mojo` around line 2098 and `boosting.mojo:1593` — put
  `params.extra.derivative_precision` onto the resolved `ConstHessianSettings`.
- `boosting_rf.mojo:1448` — `_multiclass_rf_gradients` needs `raises`.
- `src/mojotrees/__init__.mojo` — `GROW_OBLIVIOUS` export. DONE, in this
  commit.

## Standing traps, learned the hard way this round

- Never quote a SHA you have not run `git cat-file -t` on. This was violated
  once, in `bcbaf46`, one message after flagging the other session for it.
- `objective_registry.MULTICLASS` is -1 and `OBJECTIVE_UNSPECIFIED` is -2.
  Passing the former to `resolve_device` raises on every multiclass fit. It
  was fixed in `model.mojo` first, the gate reproduced the identical failure,
  because the harness calls `trainset.mojo` and `external_memory.mojo`. Fix
  all three.
- Do not rename a field by name across the tree. `grad`/`hess`/`count` are
  field names on more than one struct; a blanket pass converted
  `LocalHistogram` and `Tree.count` as collateral, twice.
- Lanes report merges as done that are not committed on their branch. Check
  `git status` in the worktree before believing "Already up to date".
