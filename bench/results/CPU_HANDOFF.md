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

## Two findings to route

### `MOJOTREES_DERIVATIVE_PRECISION=float64` is largely a no-op on sparse fits

Found by `lane/derivative-precision-wiring` while auditing every fit path,
and it **predates that lane** — it is live in the shipping env entry, not
in anything we added. Under `float64` the objective stops narrowing, but
`tree_sparse.grow_tree_sparse` still passes `narrow=True` at every
`histogram_sparse` call site (`tree_sparse.mojo:494, 620, 624, 784, 797`),
so the cells and the node totals silently re-narrow. GOSS and weighted
rounds then accumulate `Float32(w*g)` where the arm the user asked for
accumulates `w*g`. **Nothing says so anywhere.** `distributed.mojo` has the
identical shape at nine `build_histogram`/`build_histogram_subset` sites.

A user who sets the documented flag on a sparse or distributed fit gets the
slow arm's cost and the fast arm's precision. That is a wrong answer with no
signal, which is the same class as the CatBoost `mvs_reg = 0` empty tree.
Needs a lane, and the fix shape is known: one `ConstHessianSettings` or
`narrow: Bool` parameter on `grow_tree_sparse`, forwarded.

### The objective half is 25 call sites and there is no chokepoint

To make `derivative_precision` mean anything the parameter has to reach
`fill_grad_hess` (`boosting.mojo:452`), `_fill_grad_hess` (`:668`) and
`_fill_softmax_grad_hess` (`:2490`), then be forwarded from about 25 call
sites across 8 files. The cheap alternative was checked and does not work:
folding it onto `DispatchSettings` looks like 6 sites, but every trainer
outside `boosting.mojo` passes the sentinel, so it reproduces the same
silent-downgrade defect one type over, and `_fill_softmax_grad_hess` takes
no `settings` at all. `test_the_objective_half_is_still_missing` is the
tripwire and its docstring carries the fit-level test to write in its place.

### oblivious shipped and is unreachable

`GROW_OBLIVIOUS` merged in wave 4 and I exported it from
`src/mojotrees/__init__.mojo` myself. It cannot be asked for.
`python/mojotrees/sklearn.py` validates `grow_policy` against
`_GROW_POLICIES` (around line 985), which carries `leafwise` (alias
`lossguide`) and `depthwise` and nothing else, and **every arm in
`bench/real_data` and every sklearn user goes through that surface**. This
is the second feature this round that was built, tested, merged and
unreachable; CatBoost was the first.

It is with `lane/canonical-naming`, which owns that validator and whose spec
already carries the value. Wiring it turns the largest caveat on the CatBoost
comparison from a permanent unmatchable into a matched parameter. Check the
whole path when it reports: accepted-and-dropped is worse than rejected, and
`bench/real_data` has already shipped exactly that shape once.

## Owed, in order

0. **Head against `e8c0877` needs a lambda-matched form, not a caveat.**
   `lambda_l2` moved 1.0 to 0.0 this round and lambda sits in the split gain,
   so the two builds grow different trees and a wall clock would confound
   speed with tree shape. **The glue is written and not yet committed**:
   `_bench_params()` in `bench/bench_train_gpu.mojo` reads
   `MOJOTREES_BENCH_LAMBDA_L2`, raises on an unparseable value, prints on
   the `arm_conditions:` line, and is the ONE construction point both arms
   are built from -- an override reaching our arm and not LightGBM's would
   silently fit two different models, which is worse than the confounded
   number it was meant to fix. It needs a compile before it commits, and
   then the run is `MOJOTREES_BENCH_LAMBDA_L2=1.0` on head.

1. **Re-run the suite on the merged base.** IN FLIGHT as of this writing,
   on `perf-round-2` at `c8ed5b2` -- f9's head rather than `901e31d`, which
   verifies the merge and everything f9 has landed since. **119 test files,
   not 72**: the merge brought the GPU campaign's tests in. f9 handed the
   box over by message; its lock text is preserved at
   `/tmp/mojotrees-bench.lock.f9-handover`.

2. ~~A CatBoost Lossguide row.~~ **DONE, `ca33ebc`.** `catboost_lossguide`
   passes exactly two parameters, `grow_policy=Lossguide` and
   `max_leaves=31`, merged through `extra` so they meet
   `CATBOOST_REFUSED_PARAMS` like any scenario value. Verified by a
   3-iteration fit, not assumed: CatBoost accepts the pair, and it also
   keeps `depth=6` as an active cap under Lossguide where LightGBM's
   `max_depth` default is unlimited -- so the row **narrows** `tree_shape`
   rather than closing it, and both the docstring and the record say so.
   `CATBOOST_ENGINES`/`PEER_ENGINES` now replace the same tuple that was
   hand-written in three files. This closes the CatBoost harness item.

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
- `derivative_precision` onto the resolved `ConstHessianSettings` — **DONE,
  merged at `fc223da`, and the analysis that sent it to a lane was right for
  a reason I had not guessed.** The snapshot hop is closed at
  `tree.mojo:2676`, the single point every dense CPU grower passes through,
  so seven trainers picked it up with no edit each. **The refusal stays**:
  the missing half is the OBJECTIVE, not the snapshot. `fill_grad_hess` and
  `_fill_softmax_grad_hess` pick their row loop from a live `getenv`, so a
  parameter-only float64 fit stores `Float64(Float32(g))` and reads it
  un-re-narrowed — not float32, because accumulation order moves, and not
  float64, because the low 29 significand bits are already gone. A third
  numerical configuration, which is worse than either arm. No bits moved:
  `widened(False)` is asserted identity on all four fields.

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
