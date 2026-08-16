# The single comparison run: registered before the window

Written while the box is quiet and this session is compiling nothing. **Nothing
here has been measured.** It exists so the run can be executed without
improvising inside the window, and so that what each row may claim is fixed
before any number exists to be flattered by it.

Read `PROFILE_PROTOCOL.md` "HOW TO TAKE A NUMBER" first. Verdicts by **M0 as
amended by A2**. Every block carries a **canary line** and a regime label.

## The head, and the honest state of it

**Updated 2026-08-16. The head has moved twice and is not yet final.**
`a3cac1f` (below) was superseded by `174fd68`, which **did not compile** -- two
`raises`-context errors from the CTR lane, merged unrun -- and then by
`b47b939`, which does. `b47b939` is still not the run head: two more lanes (MVS
0.8 forwarding into the CatBoost-mode arm, and CTR model serialization) are
landing, and the run holds for the SHA that carries them. The sequence on that
SHA is fixed: `build-python`, `--dry-run`, one smoke-tier pass at 5k x 20 so an
arm that dies at fit dies there rather than in the window, canary, then the run.

**Read `built_extension.stale_sources` in the record before reading any number
in the table.** It was added at `b47b939` after this session found
`python/mojotrees/_mojotrees.so` twenty-two hours stale with 76 `.mojo` files
newer than it, which would have run both mojotrees arms against the previous
day's binary with nothing in the record saying so. Above zero means the run did
not execute the code its commit says it did. The block also carries the
binary's sha256, so two records can be shown to have run the same *build* and
not merely the same source commit.

**Two harness defects were found in the hold, by reading rather than by
measuring, and both are fixed.** `4eec7a9`: `build_matrix` nested repeat inside
engine, so the runner executed every repeat of one arm and then every repeat of
the next, putting the five arms in five different thermal windows on a machine
measured drifting two to three times across windows that size -- with
`mojotrees.gpu`, the headline row, sorting last. It would not have looked
wrong, because each arm's spread stays tight inside its own window. Execution is
now round-interleaved and the manifest records `arm_order`. `7e41e16`: the
CatBoost-mode GPU skip led with a reason that a queued lane is about to delete;
it now leads with the permanent one.

---

**`a3cac1f`** on `perf-round-2`, verified by `git cat-file -t`. Clean but for an
uncommitted `pixi.lock`, which is the CatBoost dependency solving against a
`pixi.toml` that already carries `catboost = ">=1.2"` at the committed head.
**Record that with the run**: the environment is not the one the SHA alone
describes, and a later reader reconstructing this from the commit will not get
the same solve.

**No test has run on this tree since roughly 12:20 EDT.** Everything merged after
that is unrun, including two test files that have never executed and one that
has never compiled, and including the whole CatBoost arm and the new three-way
categorical path. **The comparison run is the first execution of that code.**
This is not a reason to delay it -- it is a reason to order it (below) so that a
first-execution failure costs the cheapest row rather than the decision row.

## What the run may not be

- **No pre-runs and no re-runs to confirm.** A row that fails is a result and
  gets reported as one.
- **Twelve repeats only if the canary holds; otherwise five, said out loud.**
  **Status, measured 2026-08-16 on `b47b939`: the canary refused the window and
  no baseline was recorded.** `pixi run bench-canary 7` on a box with no mojo,
  no pixi and no build anywhere, load 1.8, returned `cpu_spread_pct 15.9` and
  `gpu_spread_pct 7.7` against a 3 percent calibration bar, and printed
  `calibration_warning: ... Do not record these numbers`. They were not
  recorded; `bench/canary_baseline.json` is still null, which is the correct
  output and not a gap. The A1 capture says why and it is not a lane:
  WindowServer at 14.6 percent, Chrome, VS Code, three `claude` processes.
  **That is the ambient desktop, and waiting does not remove it.** So a canary
  *ratio* is unavailable unless those applications are closed for the
  calibration and kept closed through the run. Absent that, the run is five
  repeats and the header states that no baseline could be established and why.
  **Decided by Andrew, 2026-08-16: spend the one retake adjacent to the real
  run, and do not wait on him to close anything.** If the applications happen
  to be shut by then and the 3 percent bar holds, the baseline is recorded and
  the run takes twelve repeats. If not, five repeats, and the header says the
  baseline could not be established and gives this reason. The retake is not
  spent on another calibration attempt in the meantime.
- **If the canary refuses the window, wait for a cool box and take it once
  more, and that is the last time.**
- **Accuracy beside every speed number in the same table. Both or nothing.**
  A speed row without its accuracy column is not published, not even as
  provisional.
- No width sweeps, no 2x2, no thermal interpretation.

## Decisions taken 2026-08-16 evening, and one consequence of them

Dense decision row at **1,000,000 x 100** in `real_data`, all arms in one
process; **the 50k row is dropped**; `--dry-run` first is permitted and is not
a test; dense first, `high_cardinality_categorical` last; canary recorded,
interleaved, twelve repeats if the canary holds else five and say so.

**A consequence was recorded here from `2de4102`, and it is WITHDRAWN. The
block it describes is a defect, and the defect is this session's.** What was
written, and believed, is kept below because the point of registering a plan
before the data is that a later reader can see what was believed when:

> The CatBoost-mode arm is being set to CatBoost's real CPU defaults, and
> those include `score_function=Cosine`. The device split search computes
> `G^2/(H+lambda)` only, and rather than implement a second gain functional in
> a kernel that cannot be tested under the no-test order, this session refused
> it: `BLOCK_SCORE_FUNCTION`, with `device='auto'` routing to the CPU and an
> explicit `device='gpu'` raising. So `mojotrees_catboost_mode` cannot run on
> the GPU.

The first sentence is true. **The conclusion does not follow from it, and it
was not checked before the block was written.** A GPU fit does not have to use
the device split search:

- `resolve_split_search` in `train_gpu.mojo` defaults to `SPLIT_SEARCH_HOST`.
  The host scan **is** the default, and every refusal string in
  `gpu_split_search.mojo` says so in as many words.
- `train_gpu._device_search_semantics_supported` is a deliberately
  non-raising AUTO gate. Its docstring: AUTO needs a non-raising eligibility
  answer "so it can retain the fully featured host scan instead of failing a
  fit."
- `ExtraTreeParams.is_active()` names `score_function`, and its own docstring
  states the consequence: the device-scan grower raises on it, while
  `gpu_tree_tables` and `gpu_resident_round` decline the resident and
  oblivious device trees and fall back to the host scan, "which reads the
  field".
- `_check_device_search_supported`, the raising check, is called at the head
  of the device-scan grower, not at the head of the GPU trainer.

So before the block, `device='gpu'` with `score_function=Cosine` **worked at
defaults**: GPU histograms, GPU everything else, host split scan, Cosine
honored correctly. The block replaced a correct GPU fit with a refusal, to
prevent a wrong answer the existing gates were already preventing.

**The arm set does not change, and neither does any number in this table.**
`mojotrees_catboost_mode` is still a CPU row with an empty GPU cell, because
CatBoost is CPU-only in this harness and that row has no counterpart to be read
against on any device -- which is the ground `bench/real_data/run.py` now
states first, ahead of the block, precisely so that removing the block does not
silently add a column. The removal is queued for after the run under the same
rule this file applies to the ordered-boosting hole below: the comparison head
does not take unverified code, under a no-test order, to close a hole the run
does not touch. That rule applies to this session's own mistakes or it is not a
rule.

**Decided by Andrew, 2026-08-16: the removal goes after the run.** Both
sessions recommended it independently and for the same reason, which is the
rule above rather than convenience. So the block ships in the comparison head,
knowingly wrong, and this paragraph is the record that it was known. What a
reader should take from the resulting table: at this head, `device='auto'` on a
Cosine fit resolved to the CPU. No row in the table asks that of it.

**Also decided: nothing merges from `lane/ordered-device`, including
`rung_of_position`.** That lane stopped correctly, refusing to build against a
`2.64n` plane budget that is the value at `n = 1e6` rather than a bound --
verified from source arithmetic at `n = 204,801`, where the exact count is
`614,201 = 2.9990n` against a true bound of `3n`. The design goes back for
another round after the run. Holding `rung_of_position` out is right on its own
terms as well: it is exported, it moved the api snapshot 629 to 630, and
nothing calls it, which is this repository's most-repeated defect.

**And that costs almost nothing, which was checked rather than assumed, after
this session first wrote down that it cost a lot.** `CatBoostEngine.load` in
`bench/real_data/engines.py` raises `EngineError` for any device but CPU, on
the stated grounds that CatBoost's GPU training is a different quantization --
`border_count` capped at 255 on GPU against 65535 on CPU -- and so is not the
same measurement. That covers `catboost_lossguide` too, through
`CATBOOST_ENGINES`. `verify.py` then groups by `(scenario, engine, device,
threads)`, with device in the key.

**CatBoost has no GPU row and never has.** A GPU `mojotrees_catboost_mode` row
is the mojotrees half of a peer pair whose other half is CPU-only by
construction, so it would have had no counterpart to be read against on that
device. The refusal removes a cell that was already comparable to nothing.

The lever this file previously offered -- run the arm at `score_function=L2`
to buy the GPU cell back -- is therefore **withdrawn**, and it was the wrong
trade in the first place: it would have given up a true CatBoost default to
obtain a number nothing pairs with. `score_function=Cosine` stays.

## The blocks

The six arms do not live in one harness. The CatBoost arms exist only in the
Python harness `bench/real_data/`; the Mojo harness `bench/bench_train_gpu.mojo`
has `gpu-device` and `lightgbm` and no CatBoost arm at all. `run.py` has `--tier`
as a three-way choice with **no rows/features override**, and `dense_regression`
resolves to 200,000 x 50 at standard and 1,000,000 x 100 at large. So **1M x 50
and 50k are not expressible in the harness that has all six arms.**

What "one process, interleaved arms" protects is arms compared *against each
other* inside one thermal window. That protection is a property of a block, not
of the whole table. Two blocks with different arm sets preserves it; forcing one
process by moving the decision shape to 100 features does not, because it
changes the decision.

### Block 1 -- the dense rows. Runs first.

Contingent on Andrew's answer, one of:

- **(b), recommended.** Mojo harness at the stated shapes, four arms:
  ```
  MOJOTREES_LGBM_THREADS=10 MOJOTREES_BENCH_JSON=bench/results/compare_1m.json \
    pixi run -e bench bench-train-gpu 1000000 50 reg <N> gpu-device,gpu-cpu,lightgbm
  MOJOTREES_LGBM_THREADS=10 \
    pixi run -e bench bench-train-gpu 50000 50 reg <N> gpu-device,gpu-cpu,lightgbm
  ```
  The decision row keeps its shape. CatBoost is absent from it and the table
  says so rather than implying it was beaten.

- **(a).** Python harness at `--tier large`, all six arms, shape **1,000,000 x
  100**. One process, but the decision row is no longer the decision row and
  every prior 1M x 50 number stops being comparable to it.

### Block 2 -- the categorical row. Runs LAST, and this ordering is the plan.

```
pixi run -e bench python bench/real_data/run.py \
  --scenario high_cardinality_categorical --tier standard \
  --engine mojotrees --engine lightgbm --engine catboost \
  --engine mojotrees_catboost_mode \
  --device cpu --device gpu --repeats <N>
```

`--tier standard` **is** 1,000,000 rows for this scenario (10 numeric plus the
cardinality ladder), so the stated shape needs no override and no decision.

It runs last because it is simultaneously the newest code, the never-executed
path, the least characterized cost, and the most brittle guard:

- `CATBOOST_SCENARIO_SUPPORT['high_cardinality_categorical']` became `None` only
  at `da7eb96`; before that it was a refusal.
- The scenario's own note calls it the newest CatBoost cell and the least
  characterized, with three costs and **not one of them measured** -- the
  `encode` phase is a full extra copy of the matrix plus a transient second for
  the reconstruction hash, roughly 120 MB each at this tier, a bound **derived
  from the shape and not a reading**.
- There is **one canonical digest** and the mixed-dtype frame must hash back to
  the canonical float64 form. If the reconstruction disagrees, `verify.py`
  **fails the scenario** rather than reporting a comparison. That is the right
  design and it means a first-execution problem here surfaces as a failed row,
  not a wrong number.

A surprise in this block therefore costs this row and leaves the dense decision
row already taken.

## What each row may claim, fixed in advance

**mojotrees GPU.** `device='auto'` must be *shown* to have reached the GPU at
1M, not assumed. No extra run is needed: `real_data` records
`device_requested` and `device_used` per row from `booster.device_`, and the
Mojo harness records resolved per-arm conditions. **Read the record. A GPU row
whose record says cpu is a CPU row.** This campaign has five instances of a
measurement that never executed the code it was about.

**CatBoost defaults.** `max_ctr_complexity` resolves to **1** at CatBoost's own
CPU defaults, against a documented default of 4 -- read out of the source by one
lane and back off fitted models by another, on three shapes and two losses. So
the CatBoost arm **builds no CTR feature combinations**, and our not having them
costs us nothing in the categorical row. That row is a fairer fight than the
ORDERED_TS gap suggested. Equally: **nothing in this run says anything about
CatBoost's Ordered mode**, because CatBoost on CPU never selects it and no arm
passes `boosting_type=Ordered`. No row may be read as covering it.

**Confirmed from CatBoost's source rather than assumed, 2026-08-16, and the
usual reading of the threshold is backwards.** `boosting_options.cpp:16`
constructs the default as Plain. `catboost_options.cpp:806` is the only
place Ordered is installed as a default, and its condition includes
`TaskType == GPU`; every arm here is CPU. `defaults_helper.h::UpdateBoostingTypeOption`
hard-sets Plain when the option is unset and
`(learnSampleCount >= 50000 || IterationCount < 500)`. **The 50,000 constant
turns Ordered off, not on**, it is a disjunction, and the half that gets
dropped in conversation is `IterationCount < 500`: at `BASE_PARAMS`'
100 estimators that clause fires at **every** tier, so shrinking a scenario
would not buy an Ordered row either. A future explicit Ordered arm is a third
data point, not a correction to a wrong one.

**Verified directly at v1.2.10 on 2026-08-16 evening, and one citation
corrected.** The three sites are `boosting_options.cpp:16` (the constructed
default is Plain), `catboost_options.cpp:806` (the only site that installs
Ordered as a default, guarded by `TaskType == ETaskType::GPU`), and
`defaults_helper.h:33-42` (`UpdateBoostingTypeOption`, which only ever *sets*
Plain and leaves the option NotSet otherwise). The line above previously read
`785-790`, taken from a relay and never opened -- **a right claim with a wrong
citation, which is as wrong as the reverse and harder to notice.**

**The consequence is stronger than the original sentence.** Because the Ordered
site is GPU-gated and the constructed default is Plain, **NotSet resolves to
Plain on CPU**, so the tree count cannot switch the mode however the
`(rows >= 50000 || iterations < 500)` disjunction lands. At 500 trees and under
50,000 rows the hard-set does not fire and the answer is still Plain. Every
CatBoost row this project runs on CPU is Plain, at any tree count, and that is
now a property rather than an observation.

Two consequences worth recording while the source is open. CatBoost **refuses**
Ordered together with a nonsymmetric grow policy rather than silently resolving
to Plain (`catboost_options.cpp:1046-1050`). And on our side,
`boosting.mojo:2481-2485` raises on ordered together with any bootstrap, so
once MVS lands in `MOJOTREES_CATBOOST_MODE` **that arm can never also carry
ordered** -- a mojotrees ordered row has to be its own arm. Clean refusal, but
it forecloses a combination someone will otherwise try to build later.

**"us in CatBoost mode" is not CatBoost mode.** `MOJOTREES_CATBOOST_MODE` sets
seven keys and all seven are tree shape and regularization: `grow_policy`
depthwise, `max_depth` 6, `num_leaves` 64, `min_data_in_leaf` 1,
`min_child_hess` 0.0, `lambda_l1` 0.0, `lambda_l2` 3.0. It sets **no**
`boosting_type`, no sampler, no `score_function`. Four CatBoost knobs remain
unreachable, each for one named missing edge: `score_function` has no field
carrying the choice, `random_strength`'s scale function has zero callers, the MVS
and Bayesian samplers have zero callers because `train`'s sampler is bagging and
GOSS and nothing else, and no binner constructs a CTR column. **Head this row
"us with CatBoost's tree shape and regularization", not "us in CatBoost mode".**
The harness is honest about this internally -- `mojotrees_catboost_mode_params`
says it is not a claim mojotrees can be made into CatBoost, `CATBOOST_UNMATCHABLE`
carries both known gaps, and a selfcheck already caught one version of the
function producing CatBoost's depth with mojotrees's growth. The risk is the
heading in a table, not the code.

**`categorical_missing` has no CatBoost row and the reason is permanent**, not a
harness limit: a sentinel level would get an ordinary target statistic where both
other engines make a missing category structurally unsplittable, and that
generator drops values as a function of the target's tail, so the sentinel would
be a target-correlated feature available to one arm only.

## Registered failure modes

Written down now so that none of them becomes a reason to take another run.

1. **An arm's record shows it did not reach what it names** (a GPU row resolving
   to cpu, a refused CatBoost cell). Report the row as not taken and say which
   edge was missing. Do not substitute a different shape.
2. **The categorical digest fails to reconstruct.** `verify.py` fails the
   scenario. That is the guard working; report the failure, do not disable the
   digest to obtain a comparison.
3. **The canary refuses the window.** One retake on a cool box, then stop. Do
   not record a baseline the instrument declines to give.
4. **A never-executed path crashes on first execution.** Report it as the
   first-execution result it is. It is evidence about `a3cac1f`, which is what
   the run was for.

## Four more device-policy defects, found by a tool, all deferred past the run

`tools/refusal_consistency.py` (06, `71454cb`) compares what each of the four
refusal layers *claims* for a parameter -- `device_policy`, the two GPU
trainers, the `*_ok` flags in `bindings/_mojotrees.mojo`, and `sklearn.py` --
and reports disagreements. It found five. One is this session's
`score_function` block, found independently by a tool that did not know the
report existed. **The other four are in device policy, which is this session's
file, and all four were verified here rather than adopted.**

- **`random_strength` is an unrouted refusal, and it is a live user-visible
  wrong behavior.** `bindings/_mojotrees.mojo:991` refuses it on the GPU;
  `device_policy` has no block for it; `Workload` has no field for it. So
  `device='auto'` above the crossover routes a `random_strength` fit to the
  GPU and then raises, when the entire purpose of `auto` is to choose a
  backend that can run the fit. Explicit `gpu` should raise; `auto` should
  take the CPU.
- **`enable_bundle`, `linear_tree` and `forced_splits` are three gates that
  cannot fire.** `device_policy.mojo:2269`, `:2280` and `:2290` test
  `request.bundling`, `request.linear_tree` and `request.forced_splits`, and
  the only production `Workload(...)`, at `sklearn.py:2068`, sets **none of
  them**. Three POLICY_VERSION 5 blocks, written, versioned, and dead on the
  only surface every benchmark and every pip user reaches. The trainers refuse
  all three by name, so nothing returns a wrong answer; what happens is that
  `auto` errors where it was designed to fall back.

The comment at `sklearn.py:2104` is worth keeping as the record of how this
survives review: it says the narrow fallback contract "cannot see
`boosting_type` or `score_function` any more than it sees `enable_bundle` or
`linear_tree`". This session wrote that sentence, naming these exact
parameters, while editing the full path -- and did not check whether the full
path could see them either. **Naming a blindness in one branch is not the same
as testing for it in the branch you are editing.**

**None of the four can reach this run**, and that was checked rather than
assumed: the harness passes an explicit `--device`, and no arm sets
`random_strength`, `enable_bundle`, `linear_tree` or a forced-split document.
They are deferred past the run under the rule Andrew ratified for
`BLOCK_SCORE_FUNCTION`, and for the same reason.

## One defect found before the window, and why it is not patched

`resolve_device_full` takes thirteen parameters and **none of them is
`boosting_type`**; `device_policy.mojo` carries no ordered-boosting refusal; and
`boosting_type` appears zero times in `train_gpu.mojo` and
`gpu_split_search.mojo`. Now that `boosting_type='ordered'` reaches a fit,
`device='auto'` at or above the 250,000-row crossover resolves to GPU, where no
ordered code exists: plain boosting under an ordered label, or a failure deeper
in. Ordered boosting's five refusals are about samplers, objectives and
continuation; none is about the device.

**This run does not reach it** -- `MOJOTREES_CATBOOST_MODE` sets no
`boosting_type` -- and the comparison head must not take unverified code under a
no-test order to close a hole the run does not touch. `a3cac1f` stood unchanged
at the time this was written; the evidence is in `CPU_HANDOFF.md`.

**Updated 2026-08-16: Andrew assigned it, and it was closed in `2de4102`**, as
`BLOCK_ORDERED_BOOSTING` in `device_policy.mojo` plus trainer-side raises in
`train_gpu.mojo` and `train_gpu_sparse.mojo`, wired through the bindings and
`sklearn.py` so it is not another built-but-unreachable gate. That block is
correct and stays. The `score_function` block added in the same commit is not,
and is withdrawn above -- **one commit, two blocks, one right and one wrong**,
which is worth recording as its own lesson: the two were written together,
justified in the same paragraph, and reviewed as a pair, and being right about
the first is exactly what made the second easy to believe.

It is worth naming as a second category beside "modules nothing imports". A
transitive reachability walk finds an unreachable module. **No walk finds a gate
that is imported, called on every fit, and blind to the parameter it would
refuse on.** Both were invisible for the same reason: a lane tests the thing it
built, and neither failure is in the thing it built. So the audit needs both
halves -- is there code that reads this setting, *and* is there a gate that
should refuse this combination, and can it see the parameter it would refuse on.
