# CPU orchestrator handoff

Written 2026-08-16, updated after the window and again after the directive
below. Branch `cpu-round-2`, based on `perf-round-2` at `901e31d`. Read
`LANE_RULES.md` and `PROFILE_PROTOCOL.md` (C0 through C9) first; this file is
only state, not policy.

## THE DIRECTIVE THAT SUPERSEDES THE MEASUREMENT SECTIONS BELOW

Andrew, 2026-08-16, after the window: **no more speed tests until the feature
work is done.** Head against `e8c0877` and the phase profile are **cancelled**
(the `MOJOTREES_BENCH_LAMBDA_L2` glue written for the first was reverted
unbuilt rather than shipped unverified). **No full suite runs**: local tests
on touched files, merge as they pass, this is alpha. Report merged/blocked
only. **One comparison run happens at the end**, both backends, LightGBM
stock+det and CatBoost side by side.

The box is released and the lock is deleted. Everything in this file about
canaries, plateaus and anchor normalization is still true and is still how the
final run must be taken; it is simply not what anyone should be doing now.

## Suite state, and it is the last suite run

`perf-round-2` at `c8ed5b2`, **119 test files, 118 pass**. The merge brought
the GPU campaign's tests in, which is why it is 119 and not 72.

**`test_gpu_objectives` FAILS.** Relayed to f9 with the list of what I merged;
`test_gpu_objectives_native` passes beside it, as do every other GPU file
including `test_gpu_training`, `test_gpu_tree_resident` and `test_host_replica`,
so it is one file rather than the device path. Log at
`scratchpad/suite-c8ed5b2.log`. Not mine and not touched.

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

## Three findings routed this round -- ALL CLOSED

Kept because each was a *class* of defect, not a one-off, and the class is
still live somewhere in this package.

**1. `MOJOTREES_DERIVATIVE_PRECISION=float64` was largely a no-op on sparse
and distributed fits.** The objective stopped narrowing and the grower
re-narrowed at five `histogram_sparse` sites and nine `build_histogram`
sites, so a user paid the slow arm's cost for the fast arm's precision with
no signal at all. Closed in `7a01acd`. Two details worth keeping:
`build_histogram_sparse_node` derives the default-bin leftover as `totals`
minus the stored entries, so `totals` must be summed at the SAME setting as
the accumulation or the whole node's rounding difference lands in one bin;
and the distributed schema digest did not cover the setting, so a rank that
had the variable exported and a rank that did not would all-reduce histograms
taken at two precisions with nothing detecting it.

**2. A GPU fit at `float64` had ALWAYS silently returned the Float32
answer.** `gpu_gradient_stream.stage_gradients` narrows on upload;
`extra.is_active()` was believed to guard it and does not -- it gates the
device split search and the resident tree, not the histogram. Now **refused**
at both GPU growers from either entry, not forwarded: forwarding would make
the host compute Float64 gradients the device then narrows, which is the
third-configuration failure that kept the parameter refused for two rounds.
A `device='auto'` fit at float64 now raises; f9 has a lane routing `auto` to
CPU instead, on the rule that an explicit `gpu` must still raise.

**3. Oblivious shipped and was unreachable.** `GROW_OBLIVIOUS` merged in wave
4 and could not be asked for, because the estimator validated `grow_policy`
against a table carrying `leafwise` and `depthwise` only, and every benchmark
arm and every sklearn user goes through that file. Closed in `05d498d`.

**The class these three share is the important part**, and it is now four
occurrences counting CatBoost-was-built-and-unreachable: *we ship settings
that are accepted and then quietly ignored.* Every one was invisible until
somebody audited a path end to end rather than at its entry. CatBoost has the
same disease -- `mvs_reg = 0` drives every row weight to zero through a NaN
that its own comparison eats, training a tree on nothing with no error.
**Audit reachability, not just correctness**, and prefer a refusal to a
silent downgrade every time.

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

`ordered-ts-2`, `row-major-auto`, `interleave-finish`, `objective-marshaller`.
All four branched off **`cpu-round-2` at `bfd6187`**, not `perf-round-2`,
because that head is far ahead now.

**`lane/ordered-ts` produced nothing at all** -- empty branch, empty worktree,
no report, across two sessions. Retired as `lane/ordered-ts-abandoned` and
relaunched clean as `ordered-ts-2`. A lane that reports a refusal is useful; a
lane that reports nothing is not, and the two look identical from here until
you check the worktree. Check for emptiness rather than waiting.

**Two lanes are sequenced behind `interleave-finish`** and must not start
early: retyping the eight row-id readers to `Int32`, then wiring
`_LeafState.rows` to a `LeafSpan`. Both were declined this round for
colliding with the histogram cell that lane is rewriting.

**Catalog numbering collides on almost every merge.** Three lanes took
A11/A12 and one took A9 against an existing A9; I renumbered each on merge.
The catalog now runs to **A16**. Tell the next lane to read the file first and
take A17 upward.

## The CatBoost set, against Andrew's list

DONE and merged, the whole list except CTR combinations: oblivious CPU (**and
reachable now**, `05d498d`), Bayesian bootstrap, `leaf_estimation_iterations`,
Cosine score (`9b1f4da`, a **provable no-op at stock**, see below), MVS
(`9e2d73e`), auto learning rate (`2174831`), Langevin plus `model_shrink_rate`
(`5c13a7e`), border quantization plus `one_hot_max_size` (`b637ebf`), the
naming spec wired (`05d498d`), and the harness (defaults row,
matched-trees-and-lr row, Lossguide row `ca33ebc`, iterations and lr
structural on every row through `CATBOOST_MATCHED`).

OUTSTANDING: ordered target statistics, then CTR feature combinations under
`max_ctr_complexity`, which start the hour ordered-ts lands. **Ordered-ts is
the merge point**: when it lands, `cpu-round-2` goes into `perf-round-2` and
f9 gets told.

### Earlier this round, for the record
`catboost-quantization` (border quantization plus `one_hot_max_size`).

NOT REPORTED: `lane/ordered-ts`. CTR feature combinations under
`max_ctr_complexity` follow it to the same owner.

### Cosine is zero at our settings, by derivation

Worth stating plainly so nobody spends a run on it. Cosine's numerator IS the
L2 sum, `sum G^2/(H+lambda)`; its denominator is
`sum G^2*H/(H+lambda)^2`. At `lambda = 0` those are the **same expression**,
so Cosine is `sqrt(L2)`, `sqrt` is strictly increasing, and the argmax cannot
move. `lambda_l2` is 0 stock as of this round. The admission test is
unaffected too, since `num > parent` iff `sqrt(num) > sqrt(parent)`.

The one arm where it bites at stock is **leaf-wise** growth, where the queue
compares gains from different parents and `sqrt(a)-sqrt(p)` genuinely reorders
against `a-p`. That arm needs the cross-lane wiring in
`tree_parameters_extra.mojo` and `tree.mojo` and is untested. If anyone ever
gates Cosine on real data, that is the arm to run, not depth-wise.

Also settled: the widely-repeated "Lossguide defaults to NewtonL2" sits inside
`if (TaskType == GPU)`. On CPU it is unreachable and NewtonL2 would be refused
by `CB_ENSURE` anyway. Cosine is the CPU default for **every** grow policy.

## Glue held, waiting on `canonical-naming` -- ALL THREE LANDED

`canonical-naming` merged at `05d498d` and all three unblocked:

1. **MVS reachability** -- the `mvs` arm and the `mvs_reg` key are in.
2. **`symmetrictree` in the grow-policy validator** -- so the oblivious
   policy merged in wave 4 is finally callable. The lane verified the whole
   path rather than the validator: `bindings/_mojotrees.mojo:680` is the
   single shared `TreeParams` builder every fit entry point goes through, and
   each grower either branches on `GROW_OBLIVIOUS` or constructs a
   `GrowthSchedule` whose `__init__` **raises** on the code. No third shape.
3. **`subsample` is ONE key** -- shared by every sampler that selects rows,
   which is CatBoost's own shape. Bernoulli, MVS and Poisson read it;
   Bayesian refuses it because it weights every row rather than selecting any.

Three test files asserted the old behavior and were corrected on merge:
`test_sampling`, `test_bayesian_bootstrap`, `test_derivative_precision`.

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

## THE STANDING RULE OF THIS ROUND: BUILT IS NOT REACHED

Measured on `perf-round-2`, 2026-08-16, by counting importers in `src/` and
`bindings/` (not by reading reports):

| module | importers |
|---|---|
| `ctr`, `ctr_combinations`, `langevin`, `embedding`, `text_features`, `catboost_ranking`, `survival`, `multi_target`, `onnx_export` | **0** |
| `ordered_boosting`, `auto_learning_rate` | 2 |

**Eight of ten modules this round built are imported by nothing outside their
own tests.** Every one of them has passing tests, a source-verified catalog
note, and a determinism argument. None of them can be asked for.

And above them sits a second layer of the same defect: the knobs that ARE in
the Mojo package are **refused at the Python surface**, and `bench/real_data`
reaches only what `python/mojotrees/sklearn.py` accepts. So the "us in
CatBoost mode" arm is depthwise plus `lambda_l2 = 3` and nothing else.

This is now **five separate occurrences in one round**: CatBoost built and
unreachable, oblivious trees built and unreachable, `float64` accepted and
ignored on sparse and distributed, `float64` accepted and ignored on the GPU,
and the whole CatBoost knob set refused at the Python layer. Plus CatBoost's
own instance of it, `mvs_reg = 0`, which drives every row weight to zero
through a NaN its comparison eats.

### The second category, which no walk counts: a gate that is reachable and blind

Found by f9 reading rather than running, and **confirmed here by grep before
being written down**: `boosting_type` appears **zero times** across
`device_policy.mojo`, `train_gpu.mojo` and `gpu_split_search.mojo`.
`resolve_device_full` takes thirteen parameters and none of them is the
boosting type.

So now that `boosting_type='ordered'` reaches a fit -- which this round's
reachability lane is what made true -- **`device='auto'` at or above the
250,000-row crossover resolves to GPU, and the GPU trainer has no ordered
code.** Plain boosting under an ordered label, or a failure deeper in. The
five refusals ordered boosting carries are about samplers, objectives and
continuation; none is about the device.

**This is the same defect wearing a different hat, and the count above does
not catch it.** The 13-of-107 walk finds modules nothing imports. This is a
gate that IS imported, IS called on every fit, and is blind to a parameter
that has just started mattering. Both were invisible to every lane that owned
the feature -- because a lane tests the thing it built, and neither failure is
in the thing it built.

So rule 3 needs a second half: audit not only "is there code that reads this
setting" but **"is there a gate that should refuse this combination, and can
it see the parameter it would refuse on"**. A wired feature can create the
exposure for a gate somebody else owns, which is exactly what happened here.

It is f9's to fix and it is queued rather than urgent: the comparison run does
not trigger it, because `MOJOTREES_CATBOOST_MODE` sets seven tree-shape and
regularization keys and `boosting_type` is not among them. **Deliberately not
patched here** -- the comparison head must not take unverified code under a
no-test order to close a hole nothing in the run reaches.

**The rule, for every lane from here:**

1. A lane is not done when its tests pass. It is done when something outside
   its own test imports it, or when it has said in one line why nothing does.
2. **Test that a setting CHANGES A FIT**, not that it validates. Every one of
   these five passed validation. Accepted-and-silently-ignored is the failure
   mode this project produces, and it is invisible to the tests we write.
3. Audit **reachability**, not correctness: not "is this validated" but "if I
   set this and run a fit, is there code that reads it, and does the answer
   differ".
4. A feature the Python layer refuses is as absent as one never written, and
   it passes every Mojo test.

`tools/connectivity_audit.py` classifies orphans and `docs/INTEGRATION_INVENTORY.md`
renders it; several lanes owe entries there. That gate is the mechanical form
of rule 1 and it is currently non-strict.

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
