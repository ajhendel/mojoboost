# The single comparison run: registered before the window

Written while the box is quiet and this session is compiling nothing. **Nothing
here has been measured.** It exists so the run can be executed without
improvising inside the window, and so that what each row may claim is fixed
before any number exists to be flattered by it.

Read `PROFILE_PROTOCOL.md` "HOW TO TAKE A NUMBER" first. Verdicts by **M0 as
amended by A2**. Every block carries a **canary line** and a regime label.

## The head, and the honest state of it

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

**The consequence, which follows from `2de4102` and needs to be read before
the arm list is trusted.** The CatBoost-mode arm is being set to CatBoost's
real CPU defaults, and those include `score_function=Cosine`. The device split
search computes `G^2/(H+lambda)` only, and rather than implement a second gain
functional in a kernel that cannot be tested under the no-test order, this
session refused it: `BLOCK_SCORE_FUNCTION`, with `device='auto'` routing to
the CPU and an explicit `device='gpu'` raising.

So **`mojotrees_catboost_mode` cannot run on the GPU.** Under `auto` it
resolves to the CPU and the record will say `device_used=cpu`; under an
explicit `gpu` it raises. "Us in CatBoost mode on both backends" is therefore
a CPU row and an empty GPU cell.

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
no-test order to close a hole the run does not touch. `a3cac1f` stands
unchanged. Queued as a refusal block for when building resumes; the evidence is
in `CPU_HANDOFF.md`.

It is worth naming as a second category beside "modules nothing imports". A
transitive reachability walk finds an unreachable module. **No walk finds a gate
that is imported, called on every fit, and blind to the parameter it would
refuse on.** Both were invisible for the same reason: a lane tests the thing it
built, and neither failure is in the thing it built. So the audit needs both
halves -- is there code that reads this setting, *and* is there a gate that
should refuse this combination, and can it see the parameter it would refuse on.
