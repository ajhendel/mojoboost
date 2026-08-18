# Device selection

`device="auto"` has to answer one question, "CPU or GPU for this run", and
it has to be able to say why. This document describes the policy, the
evidence rule behind it, and the report it produces.

The policy lives in `python/mojotrees/device_selection.py`. The device
vocabulary it implements is the one in `src/mojotrees/device.mojo`, which
is the authority; this layer adds explanation, Python-level feature gates,
a memory estimate, and a versioned table of measured crossovers.

Every crossover figure quoted below was measured on one Apple M4 laptop, one
machine and not a chip family, on a part whose CPU and GPU share a single
memory bus. That shared bus is why a crossover measured here need not sit
where a crossover on a machine with a discrete accelerator would sit, and it
is one reason each rule is scoped to the hardware it was measured on rather
than generalized. No NVIDIA or AMD device has ever run this code, so no rule
covers one. The full scoping is under
[How it performs](../README.md#how-it-performs) in the README.

## The three values

| Value | Behavior |
|---|---|
| `"cpu"` | The default and the dependable path. Always resolves to itself. Float64 throughout, every objective, every input. |
| `"gpu"` | An explicit request. It runs on the accelerator or it raises. There is no fallback. |
| `"auto"` | The policy below. It picks the GPU only when the GPU path covers the workload and a validated crossover rule says the GPU is faster for that shape on that backend. Otherwise the CPU. |

Names are case insensitive, as LightGBM treats `device_type`. Anything
outside the three raises `ValueError`.

Why `"gpu"` never falls back: a silent fallback turns "my GPU run" into "a
CPU run that took the same wall clock and I never knew". A refusal is
information; a fallback destroys it.

## What `auto` currently does, and why

**`auto` reaches the GPU.** On Metal on an Apple M4, dense input, at 250,000
or more rows: for squared error at 50 or more features on a single output,
and for multiclass at 54 or more features on two or more classes. Everywhere
else it keeps the CPU and says which half of "no rule covered this" applied.

`crossover_rules()` in `src/mojotrees/device_policy.mojo` holds two rules,
one per output regime. They are disjoint, so no request matches both:

```text
apple-m4-metal-dense-regression
    profile.api              == metal
    profile.apple_generation == m4
    request.objective        == squared error
    request.n_outputs        <= 1
    request.n_features       >= 50
    request.n_rows           >= AUTO_GPU_MIN_ROWS   (250,000)

apple-m4-metal-dense-multiclass
    profile.api              == metal
    profile.apple_generation == m4
    request.n_outputs        >= 2
    request.n_features       >= 54
    request.n_rows           >= AUTO_GPU_MIN_ROWS   (250,000)
```

There is no cell-count term in either. `min_cells` was 50,000,000, the
product of the old 1,000,000-row floor and the 50-feature scope, and gating
on rows and on their product meant the shipped threshold was something a
reader had to derive by division. It is 0 now, which does not constrain, and
`AUTO_GPU_MIN_ROWS` is the one number for both rules.

**The multiclass rule is scoped by trees per round and not by objective
code, deliberately.** `n_outputs >= 2` is the exact statement of "this is a
softmax fit": it is what `train_multiclass_gpu` against `train_multiclass`
branches on, no single-output objective can present it, and it is declared
whether or not the caller named an objective. That last part is what makes
the rule reachable at all -- `model.fit_multiclass`,
`trainset.train_dataset_multiclass` and `external_memory.train_external_
multiclass` each pass `n_classes` and no objective, so a rule scoped
`objective == multiclass` would have been unreachable from every Mojo entry
point while looking correct in the table. Python's `binding_params` does
declare the objective, and reaches the same rule.

Its evidence is one record: 465,000 rows by 54 features over 7 classes,
GPU **15.30 s** against CPU **25.47 s**, medians of three at 0.1 and 7.7
percent spread (`bench/results/profile_2026-08-15/RESULTS.md`,
`docs/GPU_VALIDATION.md`). The GPU wins multiclass by 1.63x, and the two
spreads are nowhere near touching. Before 2026-08-16 there was no rule here,
so `auto` sent every softmax fit to the CPU at every size, which on this
hardware is the arm that loses. The class count is bounded below at two and
**not bounded above**: a class is one more independent tree over the same
already-uploaded matrix, so a larger class count is more of the work that was
measured. Capping it at the measured seven would send a ten-class fit to the
CPU while a seven-class fit of the same shape went to the device.

**The row floor is a provisional constant, deliberately set below its own
evidence, and that is the one thing to know before changing it.** The
hardware, objective and output scopes rest on two interleaved CPU-against-GPU
records at 1,000,000 x 50 (`bench/results/apple_m4_large_scaling_2026-08-14.md`
and the 2026-08-15 section of `docs/GPU_VALIDATION.md`, GPU 2.6x to 2.8x the
CPU). The floor does not. It was lowered from 1,000,000 to 250,000 on
2026-08-16 as a stated trade:

- end to end against LightGBM stock+det at 1,000,000 x 50, our GPU arm is
  1.18x ahead and our CPU arm is 1.75x behind, so which backend `auto`
  reaches is a 2.1x swing decided by dispatch, and an `auto` that
  under-reaches hands a user the losing arm silently and forever;
- at 250,000 x 50 the GPU measured 1.89 s against our own CPU's 1.66 s
  (`bench/results/profile_2026-08-15/RESULTS.md`), a 14 percent loss on a
  machine the same protocol records drifting by a factor of two between
  windows;
- at 50,000 x 50 the loss is 2.9x, which is why the floor is at 250,000 and
  not lower.

The crossover point itself is unmeasured and lies somewhere in (50,000,
1,000,000] rows at 50 features. Measuring it is scheduled for the end of the
current feature work, and it comes in as an edit to `AUTO_GPU_MIN_ROWS` plus
a `POLICY_VERSION` bump.

**`MOJOTREES_DERIVATIVE_PRECISION=float64` sends `auto` to the CPU at every
shape, and no threshold is involved.** The GPU gradient upload narrows every
per-row derivative to Float32 (`gpu_gradient_stream.stage_gradients`), so the
accelerator cannot produce the Float64 answer at any size. Precision is a
capability, so it enters as `BLOCK_DERIVATIVE_PRECISION` and `decide_device`
reads the blocks before it reads the crossover table: the shape is never
compared. `device='gpu'` under the same setting raises instead, and the
asymmetry is deliberate. A caller who named a backend is told it cannot
honor the request; a caller who asked us to pick is given the backend that
can.

**Why `auto` could not pick the GPU at all until 2026-08-16, on the very
machine the rule was measured on.** Three things stopped it, each sufficient
on its own, and the transcript below is what the second-to-last state of it
looked like.

1. A hardware-scoped rule can match only a device profile that names the
   hardware, and `DeviceCapabilities.detect()` opens no device, so it
   returned the portable fallback with `api = unknown`.
   `CrossoverEvidence.matches` tests the API first, so the rule was declined
   before the shape was ever compared. Identity now comes from the accelerator
   this binary was *compiled for*, through the comptime
   `_accelerator_arch()`, reported as `profile_source=build-target`. It costs
   nothing and opens nothing. It is a property of the build rather than of the
   machine, exactly as `has_accelerator()` already is, so a decision that
   selects a backend on it carries `WARN_BUILD_TARGET_HARDWARE` saying so.
2. `parse_apple_generation` could not parse `4-metal4`, which is the string a
   Metal device actually reports through `DeviceContext.arch_name()`:
   "metal4" ends in "l4", not "m4". So `GpuProfile.from_reported` turned a
   genuine M4 reading into `apple_generation = unknown` and the generation
   scope declined the rule a second time, for `capabilities_from_reported` and
   `decide_device_report_reported` too, which are the entry points built for a
   caller holding an open `DeviceContext`. Nothing caught it because
   `apple_m4_observed()` hands `APPLE_GEN_M4` in through the fieldwise
   constructor, so the tests asserting the rule fires were asserting it
   against a value no detection path could produce.
3. Every crossover rule is scoped to the objective it was measured on, and
   `resolve_device` declares no objective unless one is passed. This one is
   deliberate and is not being closed by weakening the scope: a caller that
   did not say what it is training has not earned a claim measured on
   something else. `resolve_device` takes an optional `objective`, and the
   six trainer entry points that hold one now pass it (`model.fit`,
   `trainset.train_dataset`, `external_memory.train_external`, and their
   multiclass counterparts, which pass "unspecified" because a softmax fit's
   per-class objective is not one they see). A caller that leaves it
   defaulted still gets the CPU at every shape *for a single output*, by
   design. Multiclass is the exception and it is not one of these three
   defects: `apple-m4-metal-dense-multiclass` is scoped by trees per round
   rather than by objective code, precisely so that the three multiclass
   entry points reach it while declaring no objective. Trees-per-round is a
   stronger declaration than the objective code, not a weaker one, so this is
   not the scope being loosened.

The transcript below is the state before (1) and (2) were fixed. It is kept
because "the table is empty", "the table has a rule that cannot be reached",
and "the rule fired" are three different states, the report distinguishes
them, and only the first is fixed by taking a measurement.

No NVIDIA or AMD device has ever executed this code at all (see
`docs/GPU_VALIDATION.md`, where every CUDA and HIP row still reads
**not run**), so no rule can exist for either.

```text
# Historical: policy version 2, before 2026-08-16. The same request today
# reports profile_source 'build-target', api metal, apple m4, and resolves
# to GPU with decision 'auto-gpu-evidence'.
device='auto' resolved to CPU.

Device      accelerator available, metal, Apple M4
Workload    1,000,000 rows x 100 features, objective 'regression', 1 output(s) per round, max_bin=255, dense
Memory      107.1 MiB device, 171.1 MiB including the tiled partial-histogram budget, 7.9 MiB pinned host (estimate)
Budget      16.0 GiB
Rules       version 2, 1 rule(s), none matched

Why
  [auto-cpu-below-evidence] none of the 1 crossover rule(s) in policy
  version 2 covers this device and workload, so auto keeps the CPU. No
  device attributes were read (profile source 'fallback'), so every
  hardware-scoped rule was out of reach whatever the shape; a caller
  holding an open DeviceContext should read its attributes and go through
  decide_device_report_reported
```

Note which half of "does not cover" fired. The shape above is exactly the
shape the one rule was measured at, and the rule still did not match,
because the profile named no hardware. That the report said which half is
what made the gap findable at all: it never implied the shape was too small.
Under policy version 3 the same request names the hardware and matches, and
a report that still says "does not cover" now means the shape. Under version
4 the shape it means is `AUTO_GPU_MIN_ROWS`, 250,000 rows, rather than the
1,000,000 above.

## Hard blocks and soft uncertainty

Two different things keep a workload off the GPU, and conflating them
would either refuse runs that work or promise runs that do not.

A **hard block** is something that will actually fail. Explicit `"gpu"`
raises on it and `"auto"` takes the CPU:

| Block | Reason code | Enforced by |
|---|---|---|
| No accelerator for this build | `no-accelerator` | `gpu_available` in `src/mojotrees/device.mojo` |
| `MOJOTREES_DISABLE_GPU=1` | `gpu-disabled-env` | the same function |
| Sparse input | `unsupported-feature` | `_fit_sparse` in `python/mojotrees/__init__.py` |
| A custom objective callable | `unsupported-feature` | `_fit_custom` in the same file |
| An `eval_set` (validation is scored on the CPU) | `unsupported-feature` | `_fit_with_metrics` in the same file |
| `lambdarank` | `unsupported-feature` | the ranker's `fit` |
| Multiclass on a build whose GPU path lacks it | `unsupported-feature` | `gpu_supports` in `device.mojo` |
| More rows than the kernels can index | `workload-limit` | `MAX_ROWS` in `src/mojotrees/histogram_gpu.mojo` |
| `max_bin` outside [2, 256] | `workload-limit` | `MAX_BINS` there and the binner |
| The memory estimate does not fit the budget | `insufficient-memory` | the estimate below |
| `MOJOTREES_DERIVATIVE_PRECISION=float64` | `derivative-precision` | `stage_gradients` in `src/mojotrees/gpu_gradient_stream.mojo` narrows every derivative to Float32 |
| `enable_bundle` | `feature-bundling` | only the dense CPU trainers in `boosting.mojo` build a bundled matrix; `_check_gpu_booster_params` in `src/mojotrees/train_gpu.mojo` |
| `linear_tree` | `linear-tree` | linear leaves need the raw feature matrix and every GPU trainer takes a binned one; same function |
| A forced-split document | `forced-splits` | applied by `tree.grow_tree` and nothing else; `_check_gpu_forced_splits` in `src/mojotrees/train_gpu.mojo` |
| `MOJOTREES_CONST_HESSIAN_VERIFY=1` | `const-hessian-verify` | the audit walks the host hessian array, which only the CPU builders do; `_check_gpu_const_hessian_verify` there |
| `boosting_type='ordered'` | `ordered-boosting` | the per-permutation rung planes are built in `src/mojotrees/ordered_boosting.mojo` and advanced by `boosting.train`; no device trainer builds them. Trainer halves: `_check_gpu_booster_params` and `_refuse_unhonored` in `train_gpu_sparse.mojo` |
| `score_function` neither `L2` nor `Cosine` | `score-function` | an unknown selector: the device scan kernels test for Cosine and treat every other code as L2, so an accelerator would answer under the wrong functional's name |
| `score_function=Cosine` **beside a categorical column** | `score-function` | the category partition search scores with the L2 gain, so the pair puts two functionals inside one argmax. `device='cpu'` does not lift it either; `split.find_best_split` raises on the same pair |
| `random_strength` above zero **beside a categorical column** | `random-strength` | only the partition search's single winner would be noised while every numerical feature had every candidate noised. `device='cpu'` does not lift it either |

**Both of the last two were WIDE refusals and were narrowed on 2026-08-17,
and the two rows above are what remains.** They used to read "`score_function`
other than `L2`" and "`random_strength` above zero", because no accelerator
path evaluated the Cosine ratio and no accelerator round loop computed the
noise's per-tree scale. Both capabilities landed that day: the device scan
kernels evaluate Cosine (`820c06b`), the oblivious level launch stages and
reads the noise plane (`c775959`), and both arms of
`train_gpu._train_gpu_rounds` compute `random_score_scale` per tree. The
CatBoost-mode default set is `score_function=cosine` **and**
`random_strength=1`, so before the narrowing the shipped symmetric default
could not use the accelerator at all. Narrowed against the standing rule in
`device_policy.mojo` rather than against the capability alone: a block may be
retired only when no downstream refusal would still fire for the same fit.

The four before those arrived together on 2026-08-16 from the refusal
sweep, and each of them was, until that day, accepted and silently not
applied. The enumeration that found them is the next section.

**The last two arrived the same evening, by a different route, and the route
is the point.** Neither was found by a sweep of the parameter surface, because
on the morning of that day neither parameter could reach a fit: `params.mojo`
refused `score_function` other than L2 by name, and `boosting_type` reached no
trainer. The CatBoost reachability work removed both refusals at the Python
surface, and **wiring a feature opened a hole in a gate that a different lane
owned.** No reachability walk finds that: this gate is imported, is called on
every fit, and was merely *blind* to the parameter it should have refused on.
So the audit has two halves, not one -- is there code that reads this setting,
**and** is there a gate that should refuse this combination, and can it see the
parameter it would refuse on.

Note on `score_function`: the surviving unknown-selector arm is written
`!= SCORE_L2 and != SCORE_COSINE` rather than as a whitelist of one, so a
third selector added later refuses the device instead of silently receiving
an L2 answer. Cosine itself is no longer refused, and the reason the WIDE
refusal was not waived on "Cosine and L2 provably agree" is still worth
carrying, because the same argument will be offered again about some other
pair: they have the same argmax at `lambda_l2 = 0`, but that identity is per
node and per parent, and it breaks under a positive `lambda_l2` (the
CatBoost-mode arm uses 3), under a leaf-wise queue comparing gains across
parents, and beside `random_strength`, whose units Cosine changes. That last
one is not symmetric and is now user-visible: see
`docs/design/RANDOM_STRENGTH_UNITS.md`. **A `random_strength` value does not
transfer between `score_function=L2` and `score_function=Cosine`.**

**The middle one is met on a default fit**, which is the part that keeps
getting lost: `lossguide` is the stock `grow_policy` and `lossguide` is the
leaf-wise queue, so `sqrt(a) - sqrt(p)` versus `a - p` across different
parents is the ordinary case rather than an exotic one. Cosine is not inert at
our defaults. Two sessions read `split.mojo` as saying it was on 2026-08-16;
the sentence there has been corrected at the source.

**Soft uncertainty** is a workload nobody has measured or documented as
covered, most often an objective outside the set `device.mojo` names
(squared error, binary logistic, poisson, huber, quantile, L1). It never
blocks an explicit `"gpu"` request, because refusing a run the native
layer would have accepted is its own kind of lie. It does keep `"auto"` on
the CPU, since choosing the GPU on an uncharacterized path is exactly the
guess `auto` exists to not make. An unidentifiable backend is soft for the
same reason: no crossover rule can be scoped to a device nobody can name.

## What a GPU fit honours, refuses, and used to ignore

**This table is the deliverable of the 2026-08-16 refusal sweep and is
meant to be re-run rather than trusted.** It exists because
`leaf_estimation_iterations`, `MOJOTREES_DERIVATIVE_PRECISION=float64`, and
`MOJOTREES_GPU_VERIFY_ROWS` were each found by accident, in the same week,
to be accepted by a GPU entry point and silently not applied. Three
instances of one failure is a method, not a coincidence, and the method is
this: **check each knob against the code that would have to read it, never
against an aggregate predicate that appears to cover it.**

Every one of the three, and every one found since, was hidden by an
aggregate that a reader assumed was the guard. `ExtraTreeParams.is_active()`
names `forced`, and refuses it on exactly one non-default path.
`ExtraTreeParams.is_active()` gates the device split search and the resident
tree and does *not* gate the histogram, which is how a Float64 derivative
request reached a Float32 kernel. `tree._search` refuses
`needs_grower_support()`, which is strictly smaller than `is_active()`, and
the difference between the two sets is where things live that nobody
refuses. Do not read any of those three as an answer.

Verdicts are for the dense GPU trainers (`train_gpu`,
`train_gpu_with_valid`, `train_custom_gpu`, `train_multiclass_gpu`) unless a
row says otherwise. "Ignored (was)" means the sweep found it silently
ignored and this release refuses it.

### Ensemble parameters (`BoosterParams`)

| Knob | Verdict | Where |
|---|---|---|
| `n_estimators`, `learning_rate` | honoured | the round loop in `train_gpu.mojo` |
| `enable_bundle` (`bundling`) | **ignored (was)** → refused | `_check_gpu_booster_params`; `train_gpu_sparse._refuse_bundling` already refused it, which is what exposed the dense gap |
| `linear_tree` (`linear`) | **ignored (was)** → refused | `_check_gpu_booster_params` and `train_gpu_sparse._refuse_unhonored`; `boosting.train` had always refused it |

### Tree parameters (`TreeParams`)

| Knob | Verdict | Where |
|---|---|---|
| `num_leaves`, `min_data_in_leaf`, `max_depth` | honoured | the growth loop and `tree._search` |
| `lambda_l1`, `lambda_l2`, `min_sum_hessian_in_leaf` | honoured | `tree._search` on the host arm, `GpuSplitParams` on the device arm |
| `interaction_constraints` | honoured | `constraints.allowed_features` per node |
| `feature_fraction`, `feature_fraction_bynode`, `feature_fraction_seed` | honoured | `sampling.select_tree_features` / `select_split_features` |
| `feature_fraction_bylevel` | honoured on the host scan, refused on the device scan | `_check_device_search_supported` |
| `monotone_constraints` | honoured | `active_signs`, `node_bounds`, `child_bounds` |
| the categorical family (`max_cat_threshold`, `cat_smooth`, `cat_l2`, `min_data_per_group`, `max_cat_to_onehot`) | honoured | `find_best_split(cat_params=)` on the host arm, `GpuSplitParams` on the device arm |
| `grow_policy` | honoured (`oblivious` routes to the device plane, which is its only GPU grower) | `check_grow_policy`, `GrowthSchedule` |

### Tree extras (`ExtraTreeParams`)

| Knob | Verdict | Where |
|---|---|---|
| `min_gain_to_split`, `monotone_penalty`, `feature_contri` | honoured | `split.find_best_split`, which reads `extra` directly |
| `max_delta_step`, `path_smooth` | honoured | `_leaf_value` in the GPU grower; `grower_applies_extra=True` |
| `extra_trees`, `extra_seed` | honoured | keyed by node id and tree index, both passed |
| `random_strength`, `random_score_scale`, `random_strength_seed` | honoured on both backends, refused beside a categorical column | `split.mojo`'s per-candidate noise on the host; the host-drawn Float32 plane the device scan kernels add on the device (`gpu_split_search.random_score_plane`, `oblivious_score_plane`). The per-tree scale is computed by `boosting._round_random_score_scale` and by `train_gpu._device_round_random_score_scale`, so **both devices resolve the same value from the same parameters**. `params.mojo` supplied CatBoost mode's `random_strength=1.0` on `device=cpu` only until 2026-08-17, which made one parameter string build two different models |
| `monotone_method` (non-`basic`) | refused | `ExtraTreeParams.check_scalars` |
| `cegb_penalty_split` / `cegb_tradeoff` | honoured | reconstructed in `find_best_split` when the grower carries no ledger |
| `cegb_penalty_feature_coupled` / `_lazy` | refused | `cegb.check_cegb_grower_support` |
| **`forced_splits`** | **ignored (was)** → refused | `_check_gpu_forced_splits`; applied only by `tree.grow_tree`, and AUTO steered *into* the gap because `is_active()` being true is what declines the device arm |
| `use_quantized_grad` | refused (both backends) | `ExtraTreeParams.check_quantized_grad` |
| `derivative_precision=float64` (the parameter) | refused (both backends) | `ExtraTreeParams.check_derivative_precision` |
| `leaf_estimation_iterations > 1` | honoured in `train_gpu` / `train_gpu_with_valid`; refused by entry point in `train_custom_gpu` and `train_multiclass_gpu` | `_check_leaf_estimation_config`, `_refuse_leaf_estimation` |
| `leaf_estimation_iterations > 1`, sparse GPU | **ignored (was)** → refused | `train_gpu_sparse._refuse_unhonored` |

### Side bundles

| Knob | Verdict | Where |
|---|---|---|
| `BaggingParams`, `GossParams` | honoured, identical draws on both backends | `check_bagging`, `_check_goss`, `refresh_bag`, `goss_round` |
| `ClassBaggingParams` | honoured (sparse GPU only; no dense GPU entry point takes it) | `train_gpu_sparse` |
| `DartParams` | refused | `boosting_dart` refuses a non-CPU device |
| `RfParams`, `AlternateBoostingParams`, `RankerParams`, `BayesianBootstrapParams`, `QuantGradParams` | cannot reach a GPU fit | no GPU trainer takes them |

### Environment knobs a GPU fit can be handed

Knobs prefixed `MOJOTREES_GPU_*` are device-plane tuning and are honoured by
definition; knobs prefixed `MOJOTREES_CPU_*` declare their own scope in
their name and a GPU run ignoring them is correct. What follows is
everything else, plus the two GPU knobs the sweep found inert.

| Knob | Verdict on a GPU fit | Where |
|---|---|---|
| `MOJOTREES_DISABLE_GPU`, `MOJOTREES_AUTO_MIN_CELLS`, `MOJOTREES_GPU_BACKEND` | honoured | `device_policy.mojo` |
| `MOJOTREES_DERIVATIVE_PRECISION=float64` | refused (fixed 2026-08-16) | `BLOCK_DERIVATIVE_PRECISION` |
| `MOJOTREES_CONST_HESSIAN` | honoured, through the device's own read | `GpuActiveRows.const_hessian_allowed` |
| **`MOJOTREES_CONST_HESSIAN_VERIFY`** | **ignored (was)** → refused | the audit is a host walk; `_check_gpu_const_hessian_verify` |
| **`MOJOTREES_GPU_VERIFY_ROWS`** | honoured on the incremental loop; **inert on the device-owned plane (was)** → refused there | `_check_verify_rows_reachable`, at the one place the plane is elected |
| `MOJOTREES_BINNING_SELECT_MIN_ROWS` | honoured (binning precedes both backends and changes the bins the device is fed) | `binning.fit_bins` |
| `MOJOTREES_NUM_WORKERS`, `MOJOTREES_PARALLEL_MIN_OPS`, `MOJOTREES_PARALLEL_MIN_TASK_OPS`, `MOJOTREES_CPU_TASK_FLOOR`, `MOJOTREES_CPU_CORE_POOL`, `MOJOTREES_CPU_TASKS_PER_CORE` | partly honoured: they govern host dispatch, which a GPU fit still uses for its host-side work | `parallel.mojo`, `apple_cpu_policy.mojo` |
| `MOJOTREES_PHASE_PROFILE`, `MOJOTREES_STARTUP_TRACE`, `MOJOTREES_GPU_WARMUP` | honoured | `phase_profile.mojo`, `initialization.mojo` |
| `MOJOTREES_LEAF_SCORE_UPDATE` | ignored, and **not** refused | `boosting.mojo` only. It selects between two ways of advancing raw scores that produce the same numbers, so ignoring it costs a GPU fit nothing observable, and refusing it would break an interleaved CPU/GPU benchmark that sets it once for the process. Classified, deliberately not refused. |
| `MOJOTREES_OBLIVIOUS_TRACE` | ignored | `tree._grow_oblivious_levels` only; the device oblivious plane has its own trace under `MOJOTREES_GPU_TREE_RESIDENT_TRACE` |
| `MOJOTREES_CPU_*` (bin layout, feature group, row blocks, row-major, compaction floor, quant grad and scale) | ignored, correctly | the prefix declares the scope |
| `MOJOTREES_DISTRIBUTED_*`, `MOJOTREES_DASK_BACKEND` | not on any training path here | `python/mojotrees/_dask_runtime.py` |

`MOJOTREES_DIST_*` used to head that last row and no longer exists. The seven
names were deleted from `distributed_transport.mojo` on 2026-08-17 along with
their only reader, `runtime_from_env`, because nothing called it and the
multi-process feature they configured does not ship. Setting one now does
nothing at all, which is what it did before, with the difference that the
repository no longer advertises them. See
[docs/design/DEAD_SWITCH_RESOLUTION.md](design/DEAD_SWITCH_RESOLUTION.md).

Two knobs in that last group are inert on **both** backends, which is a
different defect and belongs to whoever owns those modules rather than to
the device policy: `MOJOTREES_CPU_BIN_LAYOUT`'s reader chain ends at
`histogram.choose_bin_layout_timed`, which nothing calls, and
`MOJOTREES_CPU_QUANT_GRAD` / `MOJOTREES_CPU_QUANT_SCALE` end at
`quantized_gradient.decide_cpu_histogram` / `cpu_quant_params`, whose only
caller is a test. Likewise `quant_train_renew_leaf` and
`stochastic_rounding` on `ExtraTreeParams` have no reader in the package at
all; they are moot only because `use_quantized_grad` is refused ahead of
them.

A third used to be listed here, `MOJOTREES_GPU_GRAD_LAYOUT`. It was deleted
from `gpu_gradient_stream.mojo` on 2026-08-17 with its reader
`env_grad_layout`, for the same reason. No caller consumed the layout
constant it returned, so the variable could not move a run onto the
interleaved derivative plane. The interleaved implementation is untouched and
is selected by calling `enqueue_leaf_interleaved`, which is a call-site
decision rather than a configuration one, and which nothing calls yet.

## Environment variables

| Variable | Effect |
|---|---|
| `MOJOTREES_DISABLE_GPU=1` | Reports no accelerator. `"gpu"` raises, `"auto"` takes the CPU on a machine that does have one. |
| `MOJOTREES_AUTO_MIN_CELLS` | Cells (`n_rows * n_features`) at or above which `auto` selects the GPU. `0` means "whenever the GPU path covers the workload". Unset, negative, or unparsable means the heuristic is off, which is the default. |
| `MOJOTREES_GPU_BACKEND` | Names the backend when detection cannot. Only scopes crossover rules; it never enables or disables anything. |

`MOJOTREES_AUTO_MIN_CELLS` is parsed here exactly as `env_auto_min_cells`
parses it in `device.mojo`, so the two layers cannot disagree about what a
given environment does. It is the knob for running the crossover benchmark
that would justify a default. A GPU choice reached through it is reported
with `validated == False` and a warning saying the choice rests on no
measurement.

## The crossover rule table

**The table is native, not Python.** It lives in
`crossover_rules()` in `src/mojotrees/device_policy.mojo`, carries
`POLICY_VERSION = 8`, and holds two rules. Versions 7 and 8 added blocks and
moved no rule, so the table is unchanged from version 6; the bumps exist
because a request that sets `boosting_type='ordered'`, a non-L2
`score_function`, or `random_strength` above zero gets a different answer than
it did before.

**Version 8 is also the first block added to preserve a FALLBACK rather than to
add a refusal, and the distinction is the rule to carry forward.** The trainers
already refused `random_strength` on every accelerator path. What they did not
do was refuse it *in time*: they raise from inside the grower, after `auto` has
already chosen the accelerator on shape, so the fit died where it should have
been routed. A block does two jobs -- it refuses a configuration, and it is
often the only thing making `auto` route instead of fail. **So a block may be
retired only when no downstream refusal would still fire for the same fit.**
Retiring `BLOCK_SCORE_FUNCTION` when the device learns Cosine, with nothing
here for `random_strength`, would have turned the shipped default from a
working CPU fit into a raising one -- not by anybody introducing a bug, but as
a consequence of doing the scheduled thing. The Python module no longer
defines `RULES_VERSION`, `CROSSOVER_RULES`, or a `CrossoverRule` type at
all: it formats the native decision and adds nothing to it, so a rule that
existed only in Python would be a rule the Mojo API, the CLI, and the C API
did not have. The field table below describes the native
`CrossoverEvidence`, whose Mojo field names are given beside the older
Python ones where the two differ.

A `CrossoverEvidence` is a claim about measured performance, so it carries
the measurement with it. `evidence_id` is required and the constructor
refuses a rule without it.

| Field | Meaning |
|---|---|
| `name` | Short identifier, shown in the report. |
| `evidence_id` | Where the numbers live: a document section, a benchmark file, a commit. Required. |
| `measured_on` | The device the numbers came from. |
| `api`, `apple_generation` | Scope. Unset means the rule is not limited that way. |
| `objective` | The objective that was benchmarked. Unset means the rule is not scoped that way, which is how the multiclass rule is written: `n_outputs >= 2` states "softmax fit" exactly and the objective code adds nothing to it. |
| `min_rows`, `min_features`, `min_cells` | The thresholds themselves. |
| `max_outputs` | Upper bound on trees per round the measurement covered. Zero does not constrain. |
| `min_outputs` | Lower bound on trees per round. Zero does not constrain. `max_outputs = 1` and `min_outputs = 2` are what make the two shipped rules disjoint. |

A rule matches only when every field that is set matches, so widening a
rule to hardware nobody measured takes a deliberate edit rather than an
oversight.

### Adding a rule

Adding a rule is a benchmarking result, not a code change:

1. Run the sweep. `pixi run gpu-validate` for the phase breakdown and
   `bench/bench_train_gpu.mojo` for end-to-end training, on the device the
   rule will claim, following the procedure in `docs/GPU_VALIDATION.md`.
   Pin CPU threading (`MOJOTREES_NUM_WORKERS`, `MOJOTREES_PARALLEL_MIN_OPS`)
   so the CPU side of the comparison is reproducible.
2. Find the crossover with `MOJOTREES_AUTO_MIN_CELLS`, which exists so the
   sweep needs no rebuild.
3. Record the output in the record section of `docs/GPU_VALIDATION.md`,
   loss numbers next to throughput numbers.
4. Add the rule scoped to what was measured, cite that record in
   `evidence_id`, name the device in `measured_on`, and bump
   `POLICY_VERSION`.

Do not add a rule from reasoning, from another project's numbers, or from
a single shape. `tests/test_device.mojo` checks that every shipped rule
carries its evidence, which is what makes the citation a requirement rather
than a convention.

## The memory estimate

Every report carries an estimate of what one GPU training session would
allocate, derived term by term from the buffers
`GpuHistogramBuilder.__init__` creates in `src/mojotrees/histogram_gpu.mojo`:

| Term | Bytes | Buffer |
|---|---|---|
| `binned_matrix` | `n_rows * n_features` | `bins_dev`, uint8 |
| `leaf_ids` | `n_rows * 4` | `leaf_dev`, int32 |
| `gradients` | `n_rows * 4 * n_outputs` | `grad_dev`, float32 |
| `hessians` | `n_rows * 4 * n_outputs` | `hess_dev`, float32 |
| `histograms` | `n_features * n_bins * 12` | `out_dev`, three int32 planes |
| `feature_ids` | `n_features * 4` | `feat_dev`, int32 |

Plus, pinned on the host, two float32 staging planes of `n_rows` and one
copy of the histogram buffer.

The tiled accumulation strategy also allocates a partial-histogram buffer
whose size comes from device attributes read at runtime, so it cannot be
computed host-side without a device. `PARTIAL_BUDGET_BYTES` in
`src/mojotrees/gpu_tiling.mojo` caps it at 64 MiB, and that cap is what
`upper_bound_bytes` adds.

It is an estimate and it is labeled one everywhere it appears. It counts
training buffers, not allocator overhead, and the `n_outputs` factor on
the gradient planes is an upper bound that assumes every class plane is
resident at once. It blocks a run only when a device memory budget is
known and the estimate exceeds it. When the budget is unknown the report
says so and memory is not a factor. On a unified memory backend the budget
shown is installed host RAM, which the report also says.

## The report

`select_device(device, workload)` returns a
`DeviceReport` and raises `DeviceUnavailableError` (a `RuntimeError`
subclass) when an explicit `"gpu"` cannot run. It takes no `capabilities`
or `rules` argument: both were removed when the decision moved wholly
behind the native seam, so there is no Python-side way to substitute a
different machine or a different table.

`explain_device_choice(X, y=None, device="auto", **workload_kwargs)` is
the same policy in a form that never raises: a request that would fail
comes back with `would_raise=True` and the message in `error`, so "what
would `device='gpu'` do here" is answerable without try/except.
`report.raise_if_unsupported()` turns it back into the raise.

```python
from mojotrees.device_selection import explain_device_choice

report = explain_device_choice(X, y, device="auto")
print(report)                            # the prose explanation
report.to_dict()["resolved"]             # "cpu" or "gpu"
[r.code for r in report.reasons]         # the stable reason codes
report.memory.upper_bound_bytes          # what a GPU run would allocate
report.validated                         # a rule backed this GPU choice
```

Report fields:

| Field | Meaning |
|---|---|
| `requested` | The normalized request, `"cpu"`, `"gpu"`, or `"auto"`. |
| `resolved` | `"cpu"`, `"gpu"`, or None when the request cannot run. |
| `reasons` | Ordered `Reason(code, message)` records, the decision itself. |
| `capabilities`, `workload`, `memory` | Everything the decision rested on. |
| `rules_version`, `rules_considered`, `matched_rule` | Which table was consulted and what it said. |
| `validated` | True only for a GPU chosen by a rule with evidence. |
| `would_raise`, `error` | Set instead of raising, in `explain_device_choice`. |
| `warnings` | Choices that are legitimate but unbacked, such as an explicit GPU request. |

`to_dict()` and `to_json()` are JSON-serializable, which is what a support
ticket or a CI log wants. `explanation` renders the same content as prose
and `str(report)` is that explanation.

Reason codes are stable strings, safe to match on: `explicit-cpu`,
`explicit-gpu`, `no-accelerator`, `gpu-disabled-env`,
`unsupported-feature`, `unsupported-objective`, `workload-limit`,
`insufficient-memory`, `unvalidated-path`, `no-validated-rule`,
`rule-matched`, `below-rule-threshold`, `env-threshold`,
`below-env-threshold`.

## Injecting capabilities

**This section describes an API that no longer exists in Python, and it is
kept only so that a reader who remembers it knows where the behavior went.**
`Capabilities`, `detect_capabilities()`, and the `capabilities=` and `rules=`
arguments to `select_device` were removed when the decision moved wholly
behind the native seam. `python/mojotrees/device_selection.py` exports
`Workload`, `DeviceReport`, `Reason`, `TransferRoute`, `PredictSupport`,
`select_device`, `explain_device_choice`, `explain_predict_device`, and
`native_contract`, and nothing else.

Describing a machine you do not have is now a native-side operation:
`decide_device_report_reported` in `src/mojotrees/device_policy.mojo` takes a
profile a caller supplies, which is the same idea one layer down, and
`tests/test_device.mojo` is where devices nobody here owns are covered.

Note that accelerator availability is a property of the build, not of the
running machine: Mojo resolves `has_accelerator()` at compile time. A
redistributed wheel built where a device was present reports one as
available, so a `"gpu"` request there fails when the device is opened
rather than when it is resolved. `build_has_accelerator` records what the
build claimed, and `MOJOTREES_DISABLE_GPU=1` is the way to pin such a
build to the CPU.

## Status

This module is the policy and report layer. It is not yet wired into the
estimators; `MojoTreesRegressor(device="auto")` still resolves through
`_resolve_device`, which calls the native `resolve_device` directly. The
exact wiring, including why the estimators must keep passing a resolved
concrete device name to the native layer rather than `"auto"`, is in
`handoffs/apple_a9_device_selection.md`.
