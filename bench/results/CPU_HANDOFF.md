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

1. **Re-run the suite on the merged base.** It was green at 72/72 on the
   wave-4 base, before the merge into `perf-round-2` brought the GPU
   campaign's work in. Nobody has run it since.

2. **A CatBoost Lossguide row** (their leaf-wise, `max_leaves` 31) beside
   SymmetricTree, in `bench/real_data`. This is the one part of the CatBoost
   harness item still outstanding; reachability and the `sparse_highdim` cap
   landed in `9cc45e7`. Iterations and learning rate were already structural
   through `CATBOOST_MATCHED` and already on every row, so that half needed
   nothing.

3. **`lane/ordered-ts` merges when it reports.** As of the merge its branch
   head was still an old base commit with none of its own work on it, so it
   has not reported yet. CTR feature combinations under `max_ctr_complexity`
   go to the same owner once it lands.

4. **`wide-categorical-bins`**, but only after the `max_cat_to_onehot`
   off-by-one lands as its own commit.

5. **Queued and not launched**: Langevin plus `model_shrink_rate`. Ordered
   boosting stays a design note, since CatBoost itself defaults to Plain
   above about 50k rows.

## Out with lanes right now

`canonical-naming`, `score-function`, `derivative-precision-wiring`,
`mvs-bootstrap`. All four branched off `perf-round-2` at `901e31d`.

**`src/mojotrees/sampling.mojo` has two lanes in it and the split is a
ruling, not a convention.** `canonical-naming` holds
`canonical_bootstrap_type`, `canonical_sampling_param` and
`sampling_param_names` exclusively; `mvs-bootstrap` holds everything else in
the file and hands me its three table lines as glue once naming lands. That
makes naming a dependency of MVS rather than a parallel lane. MVS was told
not to open `src/mojotrees/mvs.mojo`: a new file importing sampling's
constants is the shape that produced the CEGB import cycle.

## The naming spec

`docs/PARAMETER_NAMING.md`, committed `3e9f0fa`. One canonical name per
parameter, always an existing vendor name, every other spelling an accepted
alias; LightGBM's names stay on the wire for model I/O and `check_parity`.
Two corrections from Andrew are already folded in and both arrived after the
lane's brief, so **the file wins over any brief**: `grow_policy` takes
`lossguide | depthwise | symmetrictree` with `leafwise`/`oblivious`/
`symmetric` as aliases and the alias words used in prose, and there is no
`boosting_scheme` key -- one `boosting_type` carries
`gbdt | dart | goss | rf | plain | ordered`, `plain` aliasing `gbdt` and
`ordered` parsing then raising not-implemented.

It also carries one behavior fix that is easy to read past: `subsample < 1`
implies `subsample_freq = 1` unless set explicitly. LightGBM's silent no-op
there is a defect we are not copying.

## Glue the lanes cannot reach

- `src/mojotrees/__init__.mojo` — `GROW_OBLIVIOUS` export. **DONE.**
- `boosting_rf.mojo:1448` — **struck, non-issue.** `_multiclass_rf_gradients`
  is a `def`, which raises in Mojo 1.0.
- `derivative_precision` onto the resolved `ConstHessianSettings` — **this is
  not glue and I did not write it.** `ConstHessianSettings.resolve()` has
  exactly two call sites in the package, `boosting.mojo:1985` and
  `tree.mojo:2676`, and every other signature takes the `unresolved()`
  sentinel. Wiring those two and lifting the parameter's refusal would make it
  work there and **silently ignore it on any fit path that does not resolve a
  snapshot**, which is the identical silent-downgrade defect the refusal
  exists to prevent, merely relocated. It needs an audit of every fit path
  first, so it went to `lane/derivative-precision-wiring` with that analysis.

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
  `git status` in the worktree before believing "Already up to date". This
  happened twice in wave 4, on `oblivious-cpu` and on the categorical lane.
- **Never regenerate a generate-from-current artifact while a file still holds
  conflict markers.** `tools/api_snapshot.py --write` run mid-merge could not
  parse its own previous value, so it silently DROPPED `numerical_contracts`
  and `platforms.tiers` and reported it only as a note reading "no previous
  value to carry". It does not fail, it narrows. Resolve first, regenerate
  second, then diff the result against both parents.
- `pgrep -fl mojo` **reports a busy box when the box is idle**: it matches
  every Chromium `mojom` helper in VS Code, Chrome and Docker. Use the
  anchored form `ps -Ao comm | grep "mojo$"`. Do not use `-x` on the
  basename; the pixi env path means the binary is not always plain `mojo`.
- A five-repeat median measures the warm-up transient, not steady state. Both
  engines climb about 8 repeats before plateauing. Use twelve and compare the
  plateau, and normalize cross-invocation comparisons by the LightGBM arm
  measured in the same process.
