# CI failure triage, first run of branch perf-round-2

Nine Mojo test files failed the first CI run this branch has ever had. The
branch accumulated 646 commits behind a workflow that triggers only on `push`
to `main` and on `pull_request` (`.github/workflows/ci.yml` lines 3 to 6), was
pushed directly, and was never opened as a pull request.

Everything below was produced by READING source at `HEAD` (`0b7ad5b`) and by
`git log -S` over the relevant symbols and literals. Nothing was built, run, or
timed. Where a conclusion is inferred rather than read, it says so in a line
beginning **Inferred**.

## Working tree warning, read this before editing anything

`git status` shows `tests/test_tree_parameters_extra.mojo` already MODIFIED and
uncommitted, by a peer lane in this shared checkout. The working tree copy
already carries the lambda_l2 sweep described in section 9 and no longer
matches the CI log. Every line number in this document is the number at `HEAD`,
not the number in the working tree, and the two differ by seventeen lines in
that one file. Also uncommitted are `bench/results/ANCHOR_COVERAGE.md`,
`compatibility/api_snapshot.json` and `python/mojotrees/sklearn.py`.

## Verdict counts

| Verdict | Count | Sections |
| --- | --- | --- |
| REGRESSION, product is wrong | 2 | 3, 6 |
| EXPECTED, test is stale | 6 | 1, 2, 5, 7, 8, 9 |
| UNDETERMINED | 1 | 4 |

## The hypothesis in the brief, tested

The lambda_l2 flip from 0.0 to 1.0 is real, is declared, and IS the cause of
exactly one of the nine failures, section 9. It is not the cause of the other
eight. The brief's instruction to verify rather than assume was the right one.
The flip landed in `fa18810` (2026-08-17 12:10) which touched
`src/mojotrees/tree.mojo`, `python/mojotrees/sklearn.py`,
`python/mojotrees/basic.py`, `tools/check_parity.py`,
`bench/real_data/scenarios.py` and two design documents, and no test file at
all. `bd46e62` (2026-08-17 12:56) swept prose in five more source files and
still reached no test. So rule 5a was violated, but its blast radius in the
Mojo suite is one file and two assertions.

The larger pattern, which the brief did not predict, is that FIVE of the nine
failures trace to two commits made on 2026-08-16 in PARALLEL LANES that never
saw each other. `e3cfb47` (18:49) changed what a parameter string accepts and
what CatBoost mode supplies. `f76ef20` (18:45), four minutes earlier on a
different lane, repaired the very test cases that `e3cfb47` was about to
invalidate, and repaired them to assert the pre-`e3cfb47` behavior. Neither is
an ancestor of the other. Both are ancestors of `HEAD`.

---

## 1. tests/test_auto_learning_rate.mojo, test_catboost_mode_default_is_silent_on_a_closed_gate, line 439

**What fails.** `assert_true(close(leaves.resolved_learning_rate(100000), 0.1))`
where `leaves = parse_params("grow_policy=oblivious leaf_estimation_iterations=1")`.
The claim is that naming `leaf_estimation_iterations` closes CatBoost's
automatic learning rate gate, so the run trains at the flat rate instead of a
derived one. The preceding line 438, which asserts the gate is closed, PASSED.
Only the value assertion failed.

**Cause.** The gate is closed and the product is right about that. What moved
is the rate the closed gate falls back to. Under `grow_policy=oblivious`,
`_apply_catboost_mode_defaults` (`src/mojotrees/params.mojo:1745-1746`) supplies
`config.booster.learning_rate = CATBOOST_CONSTANT_LEARNING_RATE`, which is
CatBoost's 0.03, whenever the string did not name `learning_rate`. This string
does not name it. `resolved_learning_rate` returns `booster.learning_rate`
untouched when the gate is closed (`params.mojo:274-291` into
`auto_learning_rate.mojo:690-691`), so the resolved rate is 0.03 and the test
demands 0.1.

Read as evidence that the two lines above it pass under the same reading. Line
415 names `learning_rate=0.03` and passes, line 423 names `learning_rate=0.1`
and passes. Only the unnamed case moved.

**Commit.** `e3cfb47` "feat: a default CatBoost supplied and a default the user
typed are not the same act", 2026-08-16 18:49. `git log -S
CATBOOST_CONSTANT_LEARNING_RATE -- src/mojotrees/params.mojo` returns that
commit and no other. The test line was written four minutes earlier in
`f76ef20`, whose own message records that this case previously used `=2`, raised
inside `parse_params`, and never reached its assertion at all. So the assertion
at line 439 has never once executed successfully anywhere.

**Verdict.** EXPECTED, test is stale.

**Edit.** Replace the literal `0.1` at line 439 with
`CATBOOST_CONSTANT_LEARNING_RATE` imported from the module that defines it, and
say in the comment that CatBoost mode supplies its own constant as the base the
derivation would have replaced. To keep teeth, the assertion must ALSO
distinguish the closed gate from an open one, because 0.03 is now the value both
a closed gate and an unreachable table row produce. Add a second assertion that
`leaves.auto_lr_note` is `auto_lr_skipped:leaf_estimation_iterations`, which
`_default_auto_learning_rate` writes at `params.mojo:1893-1897` and which no
other path can produce. That names the key CatBoost's short circuit stopped at
and cannot be satisfied by an accident.

**Second, latent failure in the same test.** Line 445 asserts that
`parse_params("grow_policy=oblivious leaf_estimation_iterations=2")` raises. It
does not, for the reason in section 7. This assertion is currently unreached
because line 439 aborts the test first, and it will fail the moment line 439 is
fixed. Fix both in one edit or the file fails twice.

---

## 2. tests/test_canonical_names.mojo, two tests

### 2a. test_catboost_only_names_state_or_refuse, "Didn't raise" at line 505

**What fails.** `with assert_raises(contains="leaf_estimation_iterations"): _ =
parse_params(String("leaf_estimation_iterations=5"))`. The claim is that the
parameter string surface refuses a leaf estimation count above 1 by name, rather
than accepting it and dropping it in a trainer that does not implement it.

**Cause.** The blanket refusal was deliberately replaced by a routing check.
`params.mojo:1372-1379` now parses the key and stores it with no refusal, and
`_check_leaf_estimation_routing` (`params.mojo:1789-1841`) raises only when
`config.is_multiclass()` is true. This string names no objective, so the config
is single output, so nothing raises. The source comment at
`params.mojo:1355-1367` states the change and its reason in full, that CatBoost
mode resolves this parameter per objective and a string that could not express
10 could not express the default it is asked to port.

**Commit.** `e3cfb47`, same as section 1. `git log -S
_check_leaf_estimation_routing -- src/mojotrees/params.mojo` returns only that
commit.

**Verdict.** EXPECTED, test is stale.

**Edit.** The old claim is dead and the new claim is narrower, so assert the
narrower one rather than deleting the case. Keep the accepted arm
`_ = parse_params(String("leaf_estimation_iterations=1"))` at line 504, change
line 505 and 506 to route through multiclass,
`with assert_raises(contains="leaf_estimation_iterations"): _ =
parse_params(String("objective=multiclass num_class=3 leaf_estimation_iterations=5"))`,
and add a THIRD arm that pins the acceptance which replaced the refusal,
`_ = parse_params(String("objective=regression leaf_estimation_iterations=5"))`.
The three arms together say what the surface now guarantees, which the single
refusal no longer does. Note the refusal message names
`model.fit_multiclass`, so `contains="leaf_estimation_iterations"` still
matches.

### 2b. test_a_vendor_alias_of_a_mojo_api_parameter_says_so

**What fails.** The loop at lines 513 to 529 asserts, for twelve spellings, both
`assert_raises(contains="Mojo API only")` and
`params_names_mojo_api_only(spelling)`. Exactly one spelling fails, and it fails
the FIRST of the two assertions only.

**Cause.** `one_hot_max_size=5` at line 517. `params.mojo:1283-1292` gives that
key a dedicated refusal whose text is "one_hot_max_size is supported by the Mojo
API and by the scikit-learn estimator only", which does not contain the literal
substring "Mojo API only". That branch sits at line 1283, ahead of the generic
`elif _is_mojo_api_only(key)` fall-through at `params.mojo:1572-1578` whose
message does contain it. The key is still in `_MOJO_API_ONLY`
(`params.mojo:178`), so the second assertion at line 529 is still true. I
checked all twelve spellings against the elif chain; `one_hot_max_size` is the
only one of them with a dedicated branch.

**Commit.** `e3cfb47` again. `git log -S 'one_hot_max_size is supported by the
Mojo API and by the' -- src/mojotrees/params.mojo` returns that commit.

**Verdict.** EXPECTED, test is stale. This is a prose match against a reworded
message while the refusal itself stands, which is precisely the failure mode the
comment at `tests/test_canonical_names.mojo:486-497` describes for
`max_ctr_complexity` and calls "the same shape as fb356d9". The file diagnosed
this class of staleness and then reintroduced it.

**Edit.** Do not widen the message to satisfy the test, because the dedicated
message is better than the generic one and exists on purpose. Match on the KEY
instead, which is the rule the same file already adopted, and assert the
capability separately. Change the loop body to
`with assert_raises(contains=_key_of(spelling)):` where `_key_of` returns the
text before `=`, and keep `assert_true(params_names_mojo_api_only(spelling))`
unchanged. That keeps teeth because the pairing of "refuses by name" with
"reports itself as Mojo API only" is exactly what rules out the unknown
parameter fall-through, which would also contain the key.

---

## 3. tests/test_const_hessian.mojo, COMPILE FAILURE

**What fails.** Not an assertion. The file does not compile on a CPU only
target. The log shows `constraint failed: Unknown GPU architecture detected`
from the standard library's `gpu/host/info.mojo`, `failed to run the pass
manager`, and `call expansion failed` / `function instantiation failed` at
`src/mojotrees/gpu_active_rows.mojo` around lines 8641, 9043, 9162, 9184 and
9412, one of them with parameter value `("GROUP": 1)`.

**The chain, read end to end.** I mapped each reported line to its enclosing
definition. All five are inside `struct GpuActiveRows`
(`gpu_active_rows.mojo:5274`), in `enqueue_range_histogram` (`:8457`),
`_enqueue_partial_at` (`:8804`), `_enqueue_atomic_family` (`:9043`) and
`_enqueue_atomic_at` (`:9184`). `_enqueue_atomic_family` is the parametric
dispatch the `("GROUP": 1)` value belongs to.

`enqueue_range_histogram` carries NO accelerator guard. Its body from line 8521
onward calls `self.ctx.enqueue_function[...]` directly. The file does contain
correctly guarded entry points, `comptime if not has_accelerator(): raise` with
an `else:` around the whole body, at `:7698`, `:7731` and `:7766`, so the
pattern is known and applied elsewhere in the same struct. `git log -S 'def
enqueue_range_histogram' -- src/mojotrees/gpu_active_rows.mojo` returns only
`cc4d847`, the project rename, so this entry point has never carried the guard.

The whole caller chain above it is unguarded too.
`histogram_gpu.mojo:2245 build_leaf` calls `:2004 enqueue_leaf` calls
`enqueue_range_histogram` at `:2071` and `:2087`, and neither intermediate has
a guard.

**Why this file and not its three siblings.** `test_backend_equivalence.mojo`,
`test_host_replica.mojo` and `test_missing.mojo` all import
`mojotrees.histogram_gpu` and all sit in the CPU set, and all three compile. In
every one of them, every function that touches the builder carries the
`comptime if not has_accelerator():` guard INSIDE ITSELF.
`test_const_hessian.mojo` is the only one with unguarded module level HELPERS
that reach the device, `_gpu_leaf_matches` at line 440 and
`_gpu_feature_group_matches` at line 513. Their callers are guarded
(`:489`, `:504`, `:539`, `:606`), but the helpers themselves are ordinary module
functions and are elaborated regardless.

**Commit.** The helpers landed with the file in `e108139`, 2026-08-15. This file
has therefore never compiled on a CPU only runner since the day it was written,
and nothing said so for two days because the branch had no CI.

**Verdict.** REGRESSION, product is wrong. The test is right to call a public
method of a public struct from a helper. The product is wrong to make that
uncompilable on a supported platform. The repository's own rule for GPU entry
points is stated in the guards at `gpu_active_rows.mojo:7698` and is not applied
here, and the consequence is not confined to tests. Any user on CPU only
x86-64 or ARM64 who writes Mojo that calls `GpuHistogramBuilder.build_leaf`
outside a `comptime` guard gets `Unknown GPU architecture detected` from inside
the standard library instead of the intended runtime refusal.

**File and line to fix.** `src/mojotrees/gpu_active_rows.mojo:8479`, the end of
`enqueue_range_histogram`'s signature. Wrap the entire body, lines 8521 to 8664,
in `comptime if not has_accelerator(): raise Error(...)` with an `else:`,
matching `:7698`. That one guard is sufficient to fix the compile, because it is
the only unguarded thing in the chain that instantiates a kernel. `build_leaf`
and `enqueue_leaf` then compile on CPU only and raise at run time, and
`_gpu_leaf_matches` is never CALLED on such a build, so no test behavior
changes.

**Inferred, not read.** That `_enqueue_partial_at`, `_enqueue_atomic_family` and
`_enqueue_atomic_at` need no guard of their own once the public entry point has
one. They are private and reached only through it. If the build still fails
after the one guard, guard `histogram_gpu.mojo:2004 enqueue_leaf` as well and
report which.

---

## 4. tests/test_cpu_parallel.mojo, test_the_crossover_is_tested_against_the_union, line 599

**What fails.** `assert_true(n_calls > len(sizes))` where `sizes` is `[50, 50]`
and `n_calls` counts how many times the dispatched body was invoked. The claim
is that the fused dispatch produced more chunks than there are regions, that is,
that it actually fanned out rather than handing each region to one task.

**Mechanism, derived by reading.** `PARALLEL_MIN_OPS` is `1 << 16`, so
`ops_each` is 49,152 and `2 * ops_each` is 98,304. `dispatch_regions`
(`parallel.mojo:878`) calls `plan_tasks(region_units(sizes), total_ops)`, which
is `plan_tasks(100, 98304)`. In auto mode that reaches `_cap_tasks`
(`parallel.mojo:557-595`), where `by_grain = 98304 // 65536 = 1`, then
`floor = dispatch_cores` raised to at least `MIN_TASKS_ABOVE_GRAIN = 2`. So the
task count IS the machine's physical core count, with a floor of 2.

`_run_regions` (`parallel.mojo:893-942`) gives task `w` the flat range
`[w*100/T, (w+1)*100/T)` and cuts it at the region boundary at 50. At `T = 2`
the boundary is exactly a task boundary, so there are precisely 2 calls and
`2 > 2` is false. At `T >= 3` there are at least 3 calls and the assertion
holds. Note that line 574, `assert_true(plan_tasks(100, 2 * ops_each) > 1)`,
passes at `T = 2` as well, which is why the test gets that far.

So the assertion is satisfied if and only if the runner reports three or more
physical cores. `dispatch_cores` is `profile.physical_cores`
(`apple_cpu_policy.mojo:1009-1013`), and `env_core_floor` defaults ON
(`parallel.mojo:383-420`), so nothing in the test's `_auto()` and env clearing
changes this.

**Commit.** The test landed in `7e7c4d4`, 2026-08-16, "parallel: measure what a
fan-out costs, and add dispatch_regions", written on a ten physical core Apple
M4 where `T` is 10.

**Verdict.** UNDETERMINED.

The product is very likely right and the test very likely encodes the author's
core count, but I will not call it without one number. Two things I cannot
settle by reading. First, whether the runner's physical core count is really
below 3, which is plausible on GitHub's x86-64 image (4 vCPU with SMT reports 2
physical) and implausible on `ubuntu-24.04-arm` (4 vCPU, no SMT). Second,
whether the failure appeared on ONE runner or BOTH. If it failed on the ARM
runner as well, my derivation is wrong and something else is capping the fan-out,
and the verdict changes.

**What settles it.** Two facts, either of which is one line on the runner.
(a) Which of the two matrix legs reported this failure, readable from the CI
log I do not have. (b) The value of `plan_tasks(100, 98304)` and of
`CpuProfile.detect().describe()` on each runner, which is a two line addition to
the top of this test or a standalone probe. If `plan_tasks` is 2 and
`describe()` reports `cores=2`, this is EXPECTED, test is stale, and the edit is
below. If `plan_tasks` is 4 or more and the assertion still fails, it is a
REGRESSION in `_run_regions` and I will trace it.

**Edit, conditional on that answer being "cores below 3".** Do not pin
`MOJOTREES_NUM_WORKERS`, because an explicit worker count bypasses the crossover
and the crossover is the whole subject of the test. Derive the expectation
instead. Bind `var tasks = plan_tasks(region_units(sizes), 2 * ops_each)` before
the dispatch, assert `tasks > 1` as line 574 already does, then replace line 599
with `assert_equal(n_calls, tasks)` for the even split plus a separate arm that
runs `dispatch_regions(body, sizes, ops_each)` and asserts exactly 2 calls. That
second arm is the real claim, that the union clears the crossover and each half
alone does not, and it is machine independent.

---

## 5. tests/test_device_auto_crossover.mojo, test_an_open_context_still_outranks_the_build_target, line 351

**What fails.** `assert_equal(decision.selected_device, GPU_DEVICE)` after
building capabilities from a hand written M4 attribute set through
`capabilities_from_reported`. The claim is that a reported device profile
outranks the build target.

**Cause.** It outranks the build target for the PROFILE, and does not and must
not outrank it for AVAILABILITY. `capabilities_from_reported`
(`device_policy.mojo:3607-3644`) forwards to `DeviceCapabilities.from_profile`,
which at `device_policy.mojo:1456-1460` sets `gpu_available` to
`build_has_accelerator() and not disabled`. `build_has_accelerator`
(`:891-900`) is `comptime if has_accelerator()`, a property of the binary. On
the CPU only runner it is False, so `gpu_available` is False, so
`_collect_blocks` takes its very first branch at `device_policy.mojo:2182` and
returns immediately with a no accelerator block, so `decide_device` lands on
`DECISION_AUTO_CPU_BLOCKED` at `:3329` and selects `CPU_DEVICE`.

The assertions before it, `profile_source == PROFILE_REPORTED` at line 346 and
`apple_generation == APPLE_GEN_M4` at line 347, are true on every runner and
passed. `decide_device` itself is pure and reads no environment, so this failure
is deterministic on any accelerator free build and is not flaky.

**Verdict.** EXPECTED, test is stale, with one nuance worth stating. The
asserted fact was never true on a CPU only build rather than having been
superseded. The product is right, and it would be a serious defect if a machine
with no accelerator selected the GPU because a caller handed it a string saying
"metal".

**Edit.** The same file already carries the exact remedy and uses it four lines
below. `_on_the_measured_build()` at line 179 and its use at line 382, `if not
_on_the_measured_build(): return`. Insert that guard between line 347 and line
348, so the profile half of the claim, which is what the test is NAMED for,
keeps running everywhere, and only the device selection half is skipped.

To keep teeth on a CPU only runner rather than skipping into vacuity, add one
unconditional assertion above the guard that the reported profile REACHES the
rule, using the pure matcher directly,
`assert_true(crossover_rules()[0].matches(caps, _auto(AUTO_GPU_MIN_ROWS,
M4_TRAINING_MIN_FEATURES)))`. `CrossoverEvidence.matches`
(`device_policy.mojo:1855-1877`) consults only the profile and the request and
never `gpu_available`, so that assertion runs and bites on every runner, and it
is the claim the test title actually makes.

---

## 6. tests/test_feature_sampling.mojo, test_a_complete_feature_list_excludes_nothing_however_it_is_ordered, line 720

**What fails.** `assert_equal(_bits(full.grad_at(b)),
_bits(by_repeated.grad_at(b)))` for each bin `b` of feature 0, where
`by_repeated = build_histogram(data, grad, hess, repeated)` and `repeated` is
eight copies of the id 0 over an eight feature matrix. The claim is that a
repeated feature id accumulates that feature once.

**Cause, read from the accumulation kernel.** `_accumulate_full`
(`histogram.mojo:1487`) sets `n_active = len(features) = 8` and dispatches the
interleave ladder at `:1561-1586`, choosing `_accumulate_full_at[GROUP, NARROW]`
for GROUP in 1, 2, 4, 8, 16 from `plan.group_width`. Inside
`accumulate_groups` (`histogram.mojo:1751-1830`), each group first ZEROES every
slice its lanes name (`:1773-1795`) and then walks every row adding into each
lane's slice (`:1813-1830`). With eight identical ids the arithmetic is exactly
this.

- At `GROUP == 1` there are eight groups, each zeroes feature 0 and refills it,
  and the last one wins. The result equals one accumulation and the test passes.
- At `GROUP == 4` there are two groups. Each zeroes feature 0 four times, which
  is once, then adds every row FOUR times, once per lane. The second group wipes
  the first. The result is exactly 4x the correct gradient.
- At `GROUP == 8` the result is 8x.

So the histogram a duplicate id produces is `group_width` times the right
answer. `group_width` is a scheduling decision derived from
`profile.dispatch_cores()` (`apple_cpu_policy.plan_feature_group`), so the
NUMBER depends on the machine.

There is a second and worse consequence. Groups are dispatched across tasks
(`accumulate_groups(g_start, g_end)` is the parallel body). Two groups that name
the same feature write the same output slice from two tasks with no
synchronization, which is a data race, not merely a wrong constant.

**The product contradicts its own documented invariant, in two places.**
`_check_features` (`histogram.mojo:1211-1222`) states "a repeated id is accepted
here ... Nothing downstream is unsound for it, the accumulation would simply do
feature 0 three times". It does not do it three times, it does it `group_width`
times and then partly wipes it. And `plan_feature_group`'s docstring
(`apple_cpu_policy.mojo`, the line reading "Whatever it returns, the result is
bit-identical") claims width invariance, which is false for any list with a
duplicate.

**Commit.** `6c839a0`, 2026-08-15, "Widen the CPU accumulation interleave into a
ladder, and let the schedule deliver it", is where `_accumulate_full_at` gained
its `GROUP` parameter. `git log -S _accumulate_full_at -- src/mojotrees/histogram.mojo`
returns `6c839a0`, then `e108139`, then `abbbf98`.

**Verdict.** REGRESSION, product is wrong. The test is right that the behavior
changed, and the change is a silent wrong answer plus a data race, under a
docstring that promises neither can happen.

**File and line to fix.** `src/mojotrees/histogram.mojo:1211`, `_check_features`.
Refuse duplicate ids by name rather than range only, and rewrite the docstring,
whose current soundness claim is the thing that is false. Refusing is the right
resolution rather than restoring idempotence, because no caller in the tree
wants a duplicate, because idempotence under duplicates would still leave two
tasks writing one slice, and because a silent multiplier is exactly what the
refuse rather than ignore rule elsewhere in this package exists to prevent. Also
correct the width invariance sentence in `plan_feature_group` so it is scoped to
lists without duplicates.

**Then update the test.** With the refusal in place, lines 715 to 723 assert a
behavior that no longer exists, so replace them with
`with assert_raises(contains="repeated"): _ = build_histogram(data, grad, hess,
repeated)`. That keeps the point the block was written for, which the comment at
lines 712 to 714 states, that `len(features) == n_features` does not establish
covering and the short circuit therefore has to test ascendingness. The refusal
proves the same thing and cannot be satisfied by an accident.

---

## 7. tests/test_leaf_estimation.mojo, test_the_parameter_string_carries_the_name, line 646

**What fails.** `with assert_raises(): _ = parse_params("objective=regression
leaf_estimation_iterations=2")`.

**Cause.** Identical to section 2a and from the same commit. The objective is
declared as regression, so the config is single output, so
`_check_leaf_estimation_routing` (`params.mojo:1827-1830`) returns without
raising. The range check `ExtraTreeParams.check_leaf_estimation`
(`tree_parameters_extra.mojo:2343-2349`) rejects values below 1, so line 649's
`=0` case still raises correctly. Only the `=2` case moved.

**Commit.** `e3cfb47`.

**Verdict.** EXPECTED, test is stale.

**Edit.** The test's own docstring at lines 634 to 636 states the superseded
rule and has to move with the assertion. Rewrite it to say that a value above 1
is refused on the trainer that drops it, which is `model.fit_multiclass`, and is
honored on `model.fit`. Replace line 646 and 647 with a multiclass string, and
ADD an assertion that the single output case now resolves and stores the value,
`assert_equal(parse_params("objective=regression
leaf_estimation_iterations=2").booster.tree.extra.leaf_estimation_iterations, 2)`
together with `assert_true(...leaf_estimation_active())`. That is a stronger
test than the old one, because the old one could not tell a refusal apart from
the key not existing.

---

## 8. tests/test_objective_marshalling.mojo, three python boundary tests

**What fails.** `test_python_boundary_preserves_the_multiclass_marker`,
`test_python_boundary_keeps_undeclared_distinguishable` and
`test_python_boundary_carries_real_codes_unchanged`, all three, and all three at
the CALL rather than at any assertion.

**Cause.** The `_workload` fixture (`tests/test_objective_marshalling.mojo:234-264`)
builds a real CPython dict and sends eleven keys. `decide_device_workload`
(`bindings/basic_bindings.mojo:139-163`) now reads THIRTEEN, and reads every one
with no default and deliberately so. The two the fixture does not send are
`workload["random_strength"]` (`basic_bindings.mojo:161`) and
`workload["derivative_precision"]` (`:162`). Each missing key raises a CPython
`KeyError` inside the call, so all three tests die before any assertion runs.

This is the SAME defect the fixture's own docstring records at lines 241 to 250,
recurring against the next pair of required keys, exactly as that docstring
predicted it would.

**Commits.** `73d808a`, 2026-08-16, "device_policy: BLOCK_RANDOM_STRENGTH, so
retiring a block cannot make a cliff" added `random_strength`. `9d51e29`,
2026-08-16, "device_policy: route derivative_precision, so both entries answer
the same way" added `derivative_precision`.

**The shipping Python surface is NOT affected.**
`python/mojotrees/device_selection.py:706-709` sends all four of
`ordered_boosting`, `score_function`, `random_strength` and
`derivative_precision`, and the `Workload` constructor defaults all four
(`device_selection.py:296-299`). So the real sender is current and only the test
fixture is stale. No user is affected.

**Verdict.** EXPECTED, test is stale. What it costs is coverage rather than
correctness. These three tests are the only ones pinning that `-1` (multiclass)
and `-2` (undeclared) stay distinguishable across the Python boundary, which is
the defect this file is named for, and they have executed no assertion since
2026-08-16.

**Edit.** Add `w["random_strength"] = PythonObject(0.0)` and
`w["derivative_precision"] = PythonObject(0)` to `_workload`, and, following the
pattern `f76ef20` established for the previous pair, add echo assertions in
`test_python_boundary_carries_real_codes_unchanged` beside the two already at
lines 342 and 343, `assert_equal(_field(report, String("random_strength")),
String("0.0"))` and `assert_equal(_field(report,
String("derivative_precision")), String("0"))` with the serialized spellings
checked against `DeviceDecision.serialize`. That turns the next such drift into
one named failure instead of three anonymous ones.

**Inferred, not read.** The exact serialized text of those two fields. Confirm
against `serialize` in `src/mojotrees/device_policy.mojo` before writing the
literals, or the fix trades a KeyError for a string mismatch.

---

## 9. tests/test_tree_parameters_extra.mojo, two tests, lines 743 and 786 at HEAD

**What fails.** At `HEAD`, `test_defaults_are_lightgbms_stock_values` line 743
asserts `tree.lambda_reg.to_bits() == Float64(0.0).to_bits()`, and
`test_default_lambda_l2_reaches_the_leaf_formula` line 786 feeds the default
through `raw_leaf_output(-4.0, 4.0, lambda_l1, lambda_reg)` and asserts the bits
of 1.0.

**Cause.** `TreeParams.default()` returns `TreeParams(31, 20, 1.0, 1e-3, 0.0)`
at `src/mojotrees/tree.mojo:284`. The third positional is `lambda_reg`. So the
field is 1.0 and the Newton step is `-(-4) / (4 + 1) = 0.8`, not 1.0. Both
assertions fail, and they are the ONLY two failures in the nine that the
lambda_l2 flip causes.

**Commit.** `fa18810`, 2026-08-17 12:10, which moved the literal in
`tree.mojo` alongside `sklearn.py::_LAMBDA_L2` (now 1.0 at
`python/mojotrees/sklearn.py:72`) and `basic.py`'s refit fallback, declared the
divergence in `tools/check_parity.py::STOCK_DIVERGENCES` at line 1497 with its
reason and its exit condition, and touched no test.
`bd46e62`, 2026-08-17 12:56, is the sweep commit and reached
`bench/real_data/*`, `docs/design/*`, `sklearn.py` and five source files, and
still no test.

**Verdict.** EXPECTED, test is stale. The product is right, the divergence is
declared and priced, and this is rule 5a applied one file short.

**Edit.** Already applied in the working tree by a peer lane, uncommitted, as
warned at the top of this document. The working tree version asserts
`Float64(1.0).to_bits()` at what is now line 760, rewrites the docstring to say
the test name is half a misnomer and that the assertion is "the value we
declared" rather than "LightGBM's value", and inverts the arithmetic comment so
0.8 is the shipped answer and 1.0 is the unregularized one. That is the right
sweep and it keeps teeth, because the second test still feeds the default
through the leaf formula rather than only reading the field.

One thing the peer edit does not do, and should. Nothing in the Mojo suite pins
the OTHER half of the rule, that under `grow_policy='symmetrictree'` an unset
`lambda_l2` resolves to CatBoost's 3.0 through `CATBOOST_L2_LEAF_REG`
(`src/mojotrees/tree_parameters_extra.mojo:688`, supplied at
`src/mojotrees/params.mojo:1747-1748`). Add one assertion that
`parse_params("grow_policy=oblivious").booster.tree.lambda_reg` is 3.0 and that
`parse_params("").booster.tree.lambda_reg` is 1.0. Two defaults for one
parameter, keyed on grow policy, is exactly the shape that goes stale silently,
and it is currently pinned nowhere in this suite.

---

## Ranking, most blocking for a PyPI release first

A test that asserts a superseded default blocks LESS than a compile failure on a
supported platform, and much less than a silent wrong number. Sections 1, 2, 7
and 9 are mechanical test edits with no user visible consequence. Sections 3 and
6 are the only two that a user can hit.

| Rank | Section | One line reason |
| --- | --- | --- |
| 1 | 3, const_hessian compile failure | A public method of a public struct cannot be compiled on CPU only x86-64 or ARM64, so a supported platform is broken, not merely untested. |
| 2 | 6, feature_sampling duplicate ids | Silent wrong histogram scaled by the interleave width, plus two tasks racing on one output slice, under docstrings that promise neither. |
| 3 | 8, objective_marshalling | Product surface is correct, but the only tests pinning the multiclass marker across the Python boundary have executed nothing since 2026-08-16. |
| 4 | 5, device_auto_crossover | Product is correct and conservative, but the reported profile path is now unverified on every runner the project actually has. |
| 5 | 4, cpu_parallel | Verdict is UNDETERMINED, so it blocks until one printed number says whether the fan-out rule or only the test's core count assumption is wrong. |
| 6 | 2, canonical_names | Two stale claims, one a prose match against a reworded message, one a refusal deliberately narrowed to multiclass. No user impact. |
| 7 | 7, leaf_estimation | Same narrowed refusal as section 2a, same commit, same absence of user impact. |
| 8 | 1, auto_learning_rate | Asserts 0.1 where CatBoost mode supplies its own 0.03, and hides a second failure on the next line. No user impact. |
| 9 | 9, tree_parameters_extra | The declared lambda_l2 divergence, already swept in the working tree, blocking nothing. |

## Standing observation

Five of the nine trace to two commits made four minutes apart on 2026-08-16 in
lanes that could not see each other, `f76ef20` repairing tests to assert what
`e3cfb47` was simultaneously changing. Neither is an ancestor of the other.
A pull request would have run CI on the merge and caught all five at once. The
workflow trigger at `.github/workflows/ci.yml` lines 3 to 6 is the single
change that prevents the next recurrence, and it is a smaller change than any
of the nine fixes below it.
