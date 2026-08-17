# Constraint and split-veto reach

Registered 2026-08-17. This is the reach audit for the constraints and
split-veto family, taken after two live wrong answers of the same shape were
found on one day, where the library accepted a parameter, did not honor it on one
code path, said nothing, and every gate stayed green.

Method, so a reader knows what a cell is worth
----------------------------------------------

Every cell below was established by reading the consuming site, not by
running a fit. **Nothing here was measured and no fit was trained**, per
`bench/results/LANE_RULES.md`. A cell reads HONORED only when the consuming
code was read; a cell reads REFUSED only when the refusal's message and its
caller were both read. Where a message names an escape hatch, the hatch was
followed to the grower it names and checked to exist.

The four verdicts:

- **HONORED.** The consuming code was read.
- **REFUSED BY NAME.** Acceptable. A named refusal degrades honestly.
- **SILENTLY IGNORED.** A wrong answer. One was found; see below.
- **RAISES** where it should work or route cleanly. One was found.

The distinction the sweep was asked to carry forward, and it held
------------------------------------------------------------------

A parameter **named in a declaration** that reroutes to a supported path is
materially different from a parameter nothing mentions. **The reroute was
verified rather than trusted, and one of the two lists turned out to have no
consulting code at all.**

`gpu_split_search.device_search_eligibility` and its companion
`device_search_reason`, which carry `SEARCH_EXTRA_PARAMS`, `SEARCH_OK`,
`SEARCH_FEATURE_BYLEVEL` and `SEARCH_TOO_MANY_BINS`, **have no caller
anywhere in the package.** Grepped over `src`, `tests`, `bench` and `docs`:
the only hits are the definitions and prose references to them. The
docstring says so in the future tense, at `gpu_split_search.mojo:5706`,
"`train_gpu._check_device_search_supported` is the caller that should consume
it; the handoff carries that patch". That patch never landed.

The names in that list are nonetheless honored or refused, because a
**different** and live gate does the work: `train_gpu._check_device_search_
supported` (train_gpu.mojo:1357) delegates to
`_device_search_unsupported_reason` (:470), which delegates to
`ExtraTreeParams.device_unsupported_reason`
(tree_parameters_extra.mojo:1959), which asks one question per parameter and
returns the parameter's own name. So the effect is right and the dead list is
a trap for the next reader rather than a live defect. It is filed as a defect
anyway, under rule 4, because an unreachable eligibility function that duplicates a
live one is the exact artifact that goes stale and then gets believed.

**The reroute itself is real.** Verified end to end, not assumed:

1. `_check_device_search_supported` is called at train_gpu.mojo:2027, the
   first line of `_grow_tree_gpu_device_search`.
2. Every device grower in the package is reached only from inside that
   function: `grow_tree_device_oblivious` (:2206),
   `grow_tree_device_resident` (:2388), `_device_search_resident` (:2449),
   `_device_search_incremental` (:2466). Grepped for other call sites; there
   are none.
3. Under AUTO the question form `_device_search_semantics_supported` (:705)
   reads the same predicate and `split_search_decision_for` (:745) routes the
   fit onto the host scan instead of raising.
4. The host scan honors every member of the bundle, through
   `tree._search` (tree.mojo:1867) into `split._feature_gain`
   (split.mojo:574-578).

There is one exception to step 3 and it is the RAISES finding below.

The matrix
----------

Policies are leaf-wise (`GROW_LEAFWISE`), depth-wise (`GROW_DEPTHWISE`) and
symmetric (`GROW_OBLIVIOUS`). GPU is split into its two arms because they are
different code and the AUTO decision picks between them per fit: **host
scan** is `_grow_tree_gpu`'s downloaded-histogram loop, **device scan** is
`_grow_tree_gpu_device_search`.

### monotone_constraints, the sign vector

| Path | Verdict | Citation |
|---|---|---|
| CPU leaf-wise | HONORED | `tree.grow_tree` resolves `active_signs()` at tree.mojo:3208, threads it to `_search` (:3482, :3961) and clamps child values at :3836-3878 |
| CPU depth-wise | HONORED | same loop; the policy only changes which frontier leaf `GrowthSchedule` picks, and the clamp at :3836-3878 runs for whichever leaf it is |
| CPU symmetric | HONORED | `_grow_oblivious_levels` passes `monotone=signs` and one `OutputBounds` per leaf to `find_best_split_shared` (tree.mojo:2490-2491), and clamps both child values at :2705-2743 |
| GPU host scan, all policies | HONORED | `_grow_tree_gpu` at train_gpu.mojo:3539, `monotone=signs` at :3612, clamp at :3777-3878 |
| GPU device scan, leaf-wise | HONORED | in-kernel rejection from `mono_dev` (`gpu_monotone_violates`, gpu_split_search.mojo:520), host clamp in `_commit_device_split` (train_gpu.mojo:1792-1807) |
| GPU device scan, depth-wise | HONORED | same two sites; `_enqueue_resident_split` passes `frontier[index].bounds` at :3167 |
| GPU device-owned leaf-wise plane | REFUSED BY NAME, reroutes | `gpu_tree_tables.tree_resident_supported` :528 returns `TREE_RESIDENT_MONOTONE`; train_gpu.mojo:2449 falls through to `_device_search_resident`, which honors it |
| GPU symmetric | REFUSED BY NAME, raises | `oblivious_device_supported` :1919 returns `OBLIVIOUS_TABLES`; train_gpu.mojo:2187 raises and names `device='cpu'`, correctly, because the CPU symmetric grower does honor it |
| CPU sparse, leaf-wise and depth-wise | HONORED | tree_sparse.mojo:610, :714, :850-966 |
| GPU sparse | HONORED | train_gpu_sparse.mojo:357, :411, :500-531 |
| linear leaves | REFUSED BY NAME | linear_tree.mojo:932-940 |
| distributed | REFUSED BY NAME | distributed.mojo:673 |

The wiring the sweep was asked to check first is **intact**. `set_monotone`
has a production caller at train_gpu.mojo:1927, inside
`GpuSplitSearcherCache.reset_for_tree`, three lines above the
`set_score_function` call that had none. The chain was followed to the kernel
rather than stopping at the call: `set_monotone` writes `mono_dev` and sets
`self.constrained` (gpu_split_search.mojo:7930); `self.constrained` has
exactly two writers, the constructor (:6814) and that method, verified by
grep; the enqueue sites read it at :8092, :8168, :8364, :8579; the scan
kernels take it as `constrained` and gate the rejection on it at :2109,
:2790, :3191, :4028, :4783. The per-fit upload skip in `reset_for_tree`
(`_same_signs`) is sound because a searcher rebuild sets
`signs_staged = False` at :1925, so a reshape cannot leave the device holding
a stale vector.

**A CPU/GPU gate for this exists, contrary to the brief's premise, and this
correction matters because the premise was the reason monotone led the
sweep.** `tests/test_gpu_training.mojo:586`,
`test_gpu_monotone_constraints_match_cpu`, fits the same constrained
`BoosterParams` on `train` and `train_gpu`, walks feature 0's bins upward and
feature 1's downward, and asserts the step direction **exactly** per backend
(:638-639, :646-647). A silently dropped constraint on the arm that test
reaches would fail it.

What that gate does NOT establish, and this is the residual risk:

1. **It does not prove which arm ran.** No assertion of the split-search
   decision, which `bench/results/LANE_RULES.md` requires of a test for a
   conditional path. As of the 2026-08-17 threshold withdrawal in
   `gpu_split_policy.decide_split_search` the device scan is taken for every
   eligible shape at every size on observed hardware, and monotone is
   eligible, so it very likely reaches the device arm on an M4 and takes the
   host arm on anything else. Which one it took is not recorded.
2. **Its docstring is now stale.** It reads "Monotonic constraints are
   enforced host-side, on downloaded histograms", which describes the host
   arm only. On the device arm the candidate rejection is in the kernel.
   Replacement text is in the handoff below.
3. **It covers leaf-wise only.** No gate fits a constrained model under
   depth-wise or symmetric growth on either backend.

The separate reroute claim is gated and that gate is a good one:
`tests/test_gpu_tree_resident.mojo:517-523` sets a constraint, asserts the
device-owned plane refused, and asserts the two forests are the same; :562-591
asserts the refusal reason string is `monotone constraints` and not some other
refusal standing in for it.

### interaction_constraints

| Path | Verdict | Citation |
|---|---|---|
| CPU leaf-wise | HONORED | `allowed_features` at tree.mojo:3478 for the root and :3929 for both children, `extend_branch` at :3928 |
| CPU depth-wise | HONORED | same sites |
| CPU symmetric | HONORED, REDEFINED | tree.mojo:2450-2464. The level's mask is the INTERSECTION over its leaves. Documented in `interaction.mojo` as of this sweep; it was not documented before |
| GPU host scan | HONORED | train_gpu.mojo:3600, :3823 |
| GPU device scan, leaf-wise | HONORED | :2517, :2589-2601 into `set_allowed` (:1334) |
| GPU device scan, depth-wise | HONORED | :3173-3174, staged per record |
| GPU device-owned leaf-wise plane | REFUSED BY NAME, reroutes | `tree_resident_supported` :532, `TREE_RESIDENT_INTERACTION`; falls through to the honoring loop |
| GPU symmetric | REFUSED BY NAME, raises | `oblivious_device_supported` :1921 |
| CPU sparse | HONORED | tree_sparse.mojo:714 site and :914-966 |
| GPU sparse | HONORED | train_gpu_sparse.mojo:399, :539-570 |

### forced_splits

| Path | Verdict | Citation |
|---|---|---|
| CPU dense leaf-wise | HONORED | tree.mojo:3514 seeds the root, :3579-3621 applies forced nodes in forced-node order |
| CPU dense depth-wise | HONORED, REDEFINED | same sites. Forced nodes are drained FIRST and the `GrowthSchedule` pick only runs when nothing is forced (:3586-3589), so the forced prefix is applied out of level order. Not documented anywhere; see the handoff |
| CPU dense symmetric | REFUSED BY NAME | `tree._check_oblivious` tree.mojo:2243-2248 |
| GPU, every arm and every policy | REFUSED BY NAME | `_check_gpu_forced_splits` train_gpu.mojo:1464-1506, called at :3490 **before the two arms split**, so a strategy switch cannot change the answer |
| GPU sparse | REFUSED BY NAME, but the message offers a hatch that does not exist | `_check_gpu_forced_splits_sparse` train_gpu_sparse.mojo:262-284 |
| **CPU sparse, leaf-wise and depth-wise** | **SILENTLY IGNORED** | `tree_sparse.grow_tree_sparse` contains no reference to `forced` at all; grepped the whole file |
| distributed | REFUSED BY NAME | distributed.mojo:673, :732 |
| parameter string | REFUSED BY NAME | `forcedsplits_filename` is named as unsupported |

**This is the sweep's one silent wrong answer and it is worse than a plain
gap, because two other layers actively steer a fit into it.**

The mechanism, each link read:

1. `grow_tree_sparse` calls `params.extra.check(...)`
   (tree_sparse.mojo:604), which calls `check_scalars`,
   `forced.check_features(n_features)` and
   `forced.check_budget(num_leaves, max_depth)`
   (tree_parameters_extra.mojo:2449-2452). So the grower **validates the
   document, confirms it fits the feature count and the leaf and depth
   budget, and then never reads it.** A document that fails validation is
   reported; a correct one is dropped. That is the inverse of the useful
   behavior.
2. `check_scalars`' forced-split refusal fires only for an UNMAPPED document
   (:2227-2233). A document mapped through `binning.map_forced_splits`, which
   is the only kind a caller who followed the instructions has, passes.
3. `grow_tree_sparse` passes `grower_applies_extra=True` to `tree._search`
   (tree_sparse.mojo:719, :920, :944), so `_search`'s
   `needs_grower_support()` guard (tree.mojo:1917) does not fire. That guard
   covers `max_delta_step`, `path_smooth`, `extra_trees` and
   `random_strength`, all of which this grower genuinely does apply. It says
   nothing about `forced`. `train_gpu_sparse` states this exact reasoning at
   :268-274 for its own path and then closes it with a refusal; the CPU
   sparse grower has the same hole and no refusal.
4. `boosting_sparse.mojo:26-32` claims the opposite in prose. It says the
   sparse path supports "every `TreeParams` field including the whole `extra`
   bundle" and then enumerates six members of that bundle, omitting `forced`,
   which reads as though the enumeration were the whole bundle. Correction
   text is in the handoff.
5. `device_policy.mojo:2390-2398`, `BLOCK_FORCED_SPLITS`, steers a
   `device='auto'` forced-split fit onto the CPU with
   `DECISION_AUTO_CPU_BLOCKED`, on the stated ground that "a user who asked
   us to pick asked for the backend that *can* honor it" (:2411-2414). The
   block's own sentence, "forced splits are applied by tree.grow_tree and by
   no other grower", is literally true and is exactly why the conclusion
   drawn from it is wrong, because the routing picks a **device**, and the CPU device
   has two growers. On sparse input it lands on the one that drops the
   document.
6. `train_gpu_sparse._check_gpu_forced_splits_sparse` closes the sparse GPU
   arm with the message "Train on the CPU, or leave forced_splits unset".
   For sparse input, training on the CPU means `grow_tree_sparse`. **The
   refusal offers the exit that drops the document.**

Reachable from Python, not only from the Mojo API. `python/mojotrees/
sklearn.py` accepts `forced_splits` (:1115, :3327) and `fit` accepts SciPy
sparse input without densifying (`python/mojotrees/__init__.py:51-61`).
`MojoTreesRegressor(forced_splits=...).fit(csc_matrix, y)` on the CPU is the
reproducer.

### feature_contri

| Path | Verdict | Citation |
|---|---|---|
| CPU leaf-wise and depth-wise | HONORED | `split._feature_gain` :574, gated by `penalize = extra_active and extra.penalties.contri_active()` (:744); `contri_active()` is inside `is_active()` through `FeaturePenalties.is_active()` (tree_parameters_extra.mojo:456), so `extra_active` cannot be False while a multiplier is set |
| CPU symmetric | HONORED | `find_best_split_shared` computes the same two flags at split.mojo:1497-1498 and calls the same `_feature_gain` at :1951 |
| GPU host scan, all policies | HONORED | routes through `tree._search` into the same `_feature_gain` |
| GPU device scan, all policies | REFUSED BY NAME | `device_unsupported_reason` :2009-2010 returns "feature_contri or the CEGB costs" |
| GPU device-owned planes | REFUSED BY NAME | `tree_resident_supported` :548-552 and `oblivious_device_supported` :1922 both read `device_unsupported_reason` |
| CPU sparse | HONORED | `grower_applies_extra=True` into the same `_search` |
| parameter string | REFUSED BY NAME | `_MOJO_API_ONLY` params.mojo:183-184, with a sentence saying a whitespace-separated string cannot carry a per-feature vector |

### min_gain_to_split

| Path | Verdict | Citation |
|---|---|---|
| CPU all three policies | HONORED | `passes_min_gain` split.mojo:577-578, reached from both `find_best_split` and `find_best_split_shared`. A failing feature returns gain 0.0 and the fold compares strictly greater against a best starting at 0.0, so it is a veto and not a discount |
| GPU host scan, all policies | HONORED | same site |
| GPU device scan and device-owned planes | REFUSED BY NAME | `device_unsupported_reason` :1999 |

### monotone_penalty

| Path | Verdict | Citation |
|---|---|---|
| CPU all three policies | HONORED | `apply_monotone_penalty` split.mojo:576, and tree_parameters_extra.mojo:1085-1091 confirms it applies only to a feature that carries a constraint, so a model with no constraints is unaffected by any value |
| GPU host scan, all policies | HONORED | same site |
| GPU device scan and device-owned planes | REFUSED BY NAME | `device_unsupported_reason` :2007 |

### max_delta_step, and the output-bound surface

`max_delta_step` IS the output-bound parameter on this surface.
`params.mojo:1063-1068` maps `max_delta_step`, `max_tree_output` and
`max_leaf_output` onto the one field. There is no separate `output_min` or
`output_max`. The internal `monotone.OutputBounds` is not user-settable and
is covered by the monotone row.

| Path | Verdict | Citation |
|---|---|---|
| CPU leaf-wise and depth-wise | HONORED | `tree.grow_tree` opts in with `grower_applies_extra=True` (tree.mojo:3487) and applies the cap in `_leaf_value` |
| CPU symmetric | HONORED | `_grow_oblivious_levels` applies it at tree.mojo:2719, :2733, and passes `finish` into the candidate scoring through `find_best_split_shared` (split.mojo:1500) |
| GPU host scan, all policies | HONORED | train_gpu.mojo:3537, :3590, :3617 |
| GPU device scan and device-owned planes | REFUSED BY NAME | `device_unsupported_reason` :2001 |
| any other caller of `_search` | REFUSED BY NAME | tree.mojo:1917-1923, the `needs_grower_support` guard |

### path_smooth

Same shape as `max_delta_step` in every cell, and the two share
`needs_leaf_finish()`. Refused on the device at
`device_unsupported_reason` :2003. `check_scalars` additionally refuses
`path_smooth > 0` beside `min_data_in_leaf < 2`
(tree_parameters_extra.mojo:2218-2222), which is LightGBM's silent bump
turned into a refusal.

### min_data_in_leaf and min_child_hess, as split vetoes

| Path | Verdict | Citation |
|---|---|---|
| CPU leaf-wise and depth-wise | HONORED | parent-level veto at tree.mojo:1936, per-child inside `find_best_split` |
| CPU symmetric | HONORED, REDEFINED | split.mojo:1335-1344. A leaf that cannot satisfy either floor at a candidate does NOT veto the candidate; it contributes 0.0 and is split anyway, possibly into an empty child. The level-wide precheck is at tree.mojo:2426 |
| GPU host scan | HONORED | same `_search` path |
| GPU device scan, leaf-wise and depth-wise | HONORED | both floors are in `GpuSplitParams` (gpu_split_search.mojo:5756-5757) and applied per child in every scan kernel (:2165-2174, :2894-2897, :3293-3296, :4202-4205); the parent-level rules are applied host-side in `_apply_shape_rules` (train_gpu.mojo:1285-1298), written once so the two device loops cannot cut growth at different leaves |
| GPU symmetric device plane | HONORED, REDEFINED identically to the CPU | gpu_split_search.mojo:3676-3760 states the same per-leaf zero-contribution rule and :4202-4205 implements it |
| GPU device-owned leaf-wise plane | HONORED | passed to the device commit kernel at gpu_resident_round.mojo:2759-2760, :2838-2839, :2902-2903; `min_data_in_leaf < 1` is separately refused as `RESIDENT_MIN_DATA` (:1532) because the resident subtraction needs the built child nonempty |

The oblivious redefinition is the same on both backends and is stated in both
places, so it is a documented policy and not a divergence. It is worth
carrying into any harness arm: `min_data_in_leaf` under symmetric growth is
not the veto it is under the other two policies.

### min_data_per_group

Applies to categorical partition search only.

| Path | Verdict | Citation |
|---|---|---|
| CPU leaf-wise and depth-wise | HONORED | categorical.mojo:596, :601 |
| GPU device scan, leaf-wise and depth-wise | HONORED | staged at gpu_split_search.mojo:8171, :8582 and applied in the kernel at :9043 (read), :9066 (right-child floor) and :9073 (accepted-step floor), the same two placements and the same order as categorical.mojo:596 and :601 |
| CPU symmetric | N/A, refused upstream | `tree._check_oblivious` tree.mojo:2227-2242 refuses a searchable categorical column, so no categorical partition search happens |
| GPU symmetric | N/A, refused upstream | `oblivious_device_supported` :1916 returns `OBLIVIOUS_CATEGORICAL` |
| parameter string | REFUSED BY NAME | `_MOJO_API_ONLY` params.mojo:178 |

### monotone_constraints_method

REFUSED BY NAME on every path, twice. `parse_monotone_method`
(tree_parameters_extra.mojo:1094-1122) refuses `intermediate` and `advanced`
at parse time by name, and `check_scalars` (:2208-2213) refuses any non-basic
code even if it were set through the Mojo API. No cell can silently take
`basic` under another method's label.

The RAISES finding
------------------

`device='auto'` plus `grow_policy=oblivious` plus any member of this family
**raises instead of routing to the CPU**, which is the one place the AUTO
reroute described above does not hold.

The mechanism. `_grow_tree_gpu` routes `GROW_OBLIVIOUS` to
`_grow_tree_gpu_device_search` unconditionally at train_gpu.mojo:3494-3509,
without consulting `decision.uses_device()`, because there is no host-scan
symmetric grower on that backend to weigh against. So the AUTO reroute never
gets a chance, and `_check_device_search_supported` at :2027 (or the
`OBLIVIOUS_TABLES` raise at :2187) fires. `device_policy.mojo:2643-2656` says
this outright and is honest about it, saying the fields are not on a
`DeviceRequest`, so no block can see them, and "a fit that combines oblivious
growth with a monotone constraint still does" cliff.

Severity is low and it is deliberately named rather than left to be found.
The message points at `device='cpu'`, and the CPU symmetric grower genuinely
honors monotone constraints, interaction constraints, `min_gain_to_split`,
`monotone_penalty`, `feature_contri`, `max_delta_step` and `path_smooth`, so
the exit exists. What is wrong is only that `device='auto'` means "pick the
backend that can do this" everywhere else in the codebase and does not here.

The fix is to carry the fields on `DeviceRequest` and add blocks, which is
exactly what device_policy.mojo:2654-2656 already prescribes. It is a
different lane's file and a larger change than a message repair, so it is
handed off rather than specified line by line.

Classification of every change
------------------------------

Per the brief's rule, a fix to a silently-ignored parameter is a **bug fix**
and ships on with no switch; a change to a parameter that is already honored
**moves bits** under LANE_RULES rule 3 and needs `getenv(NAME) == "1"`.

| Change | Class | Switch |
|---|---|---|
| Refuse `forced_splits` in `grow_tree_sparse` | BUG FIX | none. A fit that reaches the new refusal was returning an unforced tree under a forced label and now fails loudly. **No existing correct fit changes**, because no fit was getting forced splits on this path |
| Repair `_check_gpu_forced_splits_sparse`'s message | BUG FIX, message only | none |
| Repair `BLOCK_GROW_POLICY` and `BLOCK_MAX_DEPTH` messages | BUG FIX, message only | none |
| Repair `boosting_sparse`'s bundle claim | BUG FIX, prose only | none |
| Repair `test_gpu_monotone_constraints_match_cpu`'s docstring | BUG FIX, prose only | none |
| Delete or wire `device_search_eligibility` | neither. Deleting dead code moves no bit; wiring it would be a refactor of a live gate and should not be done for tidiness | none |
| Reach sections added to `monotone.mojo` and `interaction.mojo` | neither. Docstrings | none |
| Move the CPU constrained-categorical check from per-node to per-fit | MOVES BITS in the failure direction only. It changes which fits raise, never a value | a switch is the wrong instrument here; it should land as a plain widening of an existing refusal, with the sentence stated |

**Nothing in this sweep changes a split-veto threshold.** Every veto in the
family was found either applied or refused by name. The one repair that
changes a tree, the sparse `forced_splits` refusal, does not apply a veto
that was previously unapplied. It stops a fit that was silently growing the
wrong shape. That is the distinction the brief asked to keep separate and it
falls entirely on the "this was not applied" side, with no threshold
arithmetic touched.

What could not be verified without a compiler
---------------------------------------------

1. **Whether `test_gpu_monotone_constraints_match_cpu` reaches the device
   scan on this machine.** The reasoning above says it should, on an M4,
   after the 2026-08-17 threshold withdrawal. It is an inference from reading
   `decide_split_search`, not an observation. One traced run with
   `MOJOTREES_GPU_SPLIT_TRACE` would settle it, and the test should assert it
   rather than leave it inferred.
2. **Whether `grow_tree_sparse` with a mapped forced-split document actually
   returns an unforced tree**, as against failing somewhere else first. The
   grep is conclusive that nothing reads the document, and `extra.check`
   passes a mapped one, but the end-to-end behavior was not observed.
3. **Whether the depth-wise forced-split prefix leaves `GrowthSchedule` in a
   consistent state.** Forced nodes are drained before the schedule is
   consulted at all, so the schedule sees a frontier whose leaves may sit at
   several depths on its first call. `GrowthSchedule` was not read closely
   enough to say whether it tolerates that, admits the right prefix, or
   quietly ranks across levels. It is a composition nothing tests and nothing
   documents.
4. **Whether the Float32 midpoint concern behind `TREE_RESIDENT_MONOTONE` is
   still the binding one.** The refusal's stated reason
   (gpu_tree_tables.mojo:474-478) is that `child_bounds` computes a Float64
   division by two that a Float32 midpoint would not match. Since
   `_commit_device_split` already keeps the whole clamp and midpoint chain on
   the host in Float64 for the device-scan loops, the same arrangement may be
   available to the device-owned plane. Not priced, and pricing it needs the
   plane's table layout, which is another lane's file.
5. **Every line number in this document.** They were read at the head of
   2026-08-17 in a shared checkout with nine live lanes. Nine of the files
   cited are owned by other lanes and may have moved by the time this is
   read; the symbol names are the durable part.

Handoff, for files this lane does not own
-----------------------------------------

Every item below is a CORRECTION or a BUG FIX and not a suggestion. Apply
verbatim. Re-grep the anchor before applying; peers are editing these files.

### H1. BUG FIX. `src/mojotrees/tree_sparse.mojo`, refuse forced splits

The one silent wrong answer. Add the guard function above
`grow_tree_sparse`, then call it. Wording is `train_gpu_sparse.
_check_gpu_forced_splits_sparse`'s, with the exit corrected, because that
message's own exit is wrong (see H2).

Insert before `def grow_tree_sparse(`:

```mojo
def _check_sparse_forced_splits(params: TreeParams) raises:
    """Refuse a forced-split document on a grower that does not apply one.

    `tree.grow_tree` is the only grower in this package that reads
    `params.extra.forced`. This one does not, and never has. Grep the file,
    the identifier does not appear.

    WHY THREE GUARDS THAT LOOK LIKE THEY COVER IT DO NOT, because this is the
    third time this exact gap has been found and each time it looked covered.

    1. `params.extra.check` above VALIDATES the document -- `check_features`
       against this dataset and `check_budget` against `num_leaves` and
       `max_depth` -- and then nothing reads it. A document that fails
       validation is reported and a correct one is dropped, which is the
       inverse of useful.
    2. `check_scalars`' forced-split refusal fires only for an UNMAPPED
       document. A document mapped through `binning.map_forced_splits`, which
       is the only kind a caller who followed the instructions holds, passes.
    3. This grower passes `grower_applies_extra=True` to `tree._search`, so
       the `needs_grower_support()` guard does not fire. That flag covers
       `max_delta_step`, `path_smooth`, `extra_trees` and `random_strength`,
       every one of which this grower genuinely does apply. It says nothing
       about `forced`.

    WORSE THAN A GAP, because two layers steer a fit into it.
    `device_policy.BLOCK_FORCED_SPLITS` sends a `device='auto'` forced-split
    fit to the CPU on the ground that the CPU is "the backend that can honor
    it", and `train_gpu_sparse._check_gpu_forced_splits_sparse` tells a
    sparse GPU caller to "train on the CPU". Both pick a DEVICE, and the CPU
    device has two growers; on sparse input they land on this one.
    """
    if params.extra.forced.is_empty():
        return
    raise Error(
        "forced splits are applied by tree.grow_tree, the DENSE CPU grower,"
        " and by no other grower in this package; grow_tree_sparse never"
        " reads the document and would return an unforced tree. There is no"
        " sparse grower on any backend that applies one, so device='cpu'"
        " does not lift this. Densify the matrix and train with"
        " tree.grow_tree, or leave forced_splits unset",
    )
```

Then, immediately after the existing `params.extra.check(...)` call, whose
current text is

```mojo
    # This grower applies the whole `extra` bundle, so it validates it the way
    # the dense grower does: against this dataset, before the first histogram.
    params.extra.check(
        n_features,
        params.num_leaves,
        params.max_depth,
        params.min_data_in_leaf,
    )
```

replace that block with

```mojo
    # This grower applies MOST of the `extra` bundle, so it validates it the
    # way the dense grower does, against this dataset, before the first
    # histogram. It said "the whole" bundle until 2026-08-17, and `forced` is
    # the member that was never true of; see `_check_sparse_forced_splits`.
    params.extra.check(
        n_features,
        params.num_leaves,
        params.max_depth,
        params.min_data_in_leaf,
    )
    # Refused here rather than validated and dropped. `extra.check` above has
    # just confirmed the document fits this dataset and this budget, which
    # made the drop read as support.
    _check_sparse_forced_splits(params)
```

The docstring of `grow_tree_sparse` says "Arguments and semantics match
`grow_tree`". Append to that sentence:

```
    ...fixes which features the tree and its nodes may split on.

    One exception, and it is the only one: `params.extra.forced` is REFUSED
    rather than applied (`_check_sparse_forced_splits`). Every other member
    of the bundle is honored here.
```

### H2. CORRECTION. `src/mojotrees/train_gpu_sparse.mojo:283`, the exit does not exist

The refusal offers the grower that drops the document. Current text:

```mojo
        " never reads the document and would return an unforced tree. Train"
        " on the CPU, or leave forced_splits unset",
```

Replace with:

```mojo
        " never reads the document and would return an unforced tree."
        " Training on the CPU does NOT lift this and used to be offered here"
        " in error, corrected 2026-08-17. tree_sparse.grow_tree_sparse does"
        " not read the document either. Densify the matrix and train with"
        " tree.grow_tree, which is the only grower that applies one, or"
        " leave forced_splits unset",
```

### H3. CORRECTION. `src/mojotrees/device_policy.mojo:2639-2640`, `BLOCK_GROW_POLICY`

The message offers an exit that `tree._check_oblivious` raises on. Current
text:

```mojo
                " level search evaluates ordinal thresholds only; the CPU"
                " grower grows the same symmetric tree with categorical"
                " splits",
```

Replace with:

```mojo
                " level search evaluates ordinal thresholds only. The CPU"
                " grower does NOT grow the same symmetric tree with"
                " categorical splits and this block said it did until"
                " 2026-08-17. tree._check_oblivious raises on exactly this"
                " pair, for the same reason, because a level shares one split"
                " and a category partition is ordered by one node's own"
                " statistics. The exit that exists is CatBoost-mode CTRs"
                " (SimpleCtrConfig.catboost_defaults), which replace every"
                " column above one_hot_max_size and leave nothing"
                " categorical for either backend to search",
```

### H4. CORRECTION. `src/mojotrees/device_policy.mojo:2625`, `BLOCK_MAX_DEPTH`

Same file, same block set, and the same defect shape. `growth_policy.
OBLIVIOUS_MAX_DEPTH` is 16 and `tree._check_oblivious` raises above it, so
"at any depth" is false for `max_depth > 16`. Current text:

```mojo
                "; the CPU grower grows the same symmetric tree at any depth",
```

Replace with:

```mojo
                "; the CPU grower grows the same symmetric tree up to"
                " growth_policy.OBLIVIOUS_MAX_DEPTH, which is 16 and above"
                " which tree._check_oblivious raises as well. This block said"
                " 'at any depth' until 2026-08-17",
```

### H5. CORRECTION. `src/mojotrees/device_policy.mojo:2649`, stale line reference

`train_gpu:1780` is now inside `_commit_device_split`'s signature. The raise
this sentence means is at train_gpu.mojo:2187, in
`_grow_tree_gpu_device_search`'s oblivious branch. Current text:

```mojo
    # them and `train_gpu:1780` still RAISES for them under `auto`.
```

Replace with:

```mojo
    # them and `train_gpu._grow_tree_gpu_device_search`'s oblivious branch
    # (train_gpu.mojo:2187 at the head of 2026-08-17) still RAISES for them
    # under `auto`, because `_grow_tree_gpu` routes GROW_OBLIVIOUS to that
    # function without consulting `decision.uses_device()` at all.
```

### H6. CORRECTION. `src/mojotrees/boosting_sparse.mojo:26-27`

The prose claims the whole bundle reaches the sparse grower. It does not;
`forced` is the member it never reached. Current text:

```
and interaction constraints, feature subsampling, and every `TreeParams`
field including the whole `extra` bundle -- `min_gain_to_split`,
```

Replace with:

```
and interaction constraints, feature subsampling, and every `TreeParams`
field including all of the `extra` bundle except `forced` -- `min_gain_to_split`,
```

And after "renewal included.", append a paragraph:

```
Not available and refused by name: `forced_splits`. It is the one member of
the `extra` bundle `grow_tree_sparse` does not apply, and this paragraph
claimed the whole bundle until 2026-08-17, when the reach sweep found the
document accepted, validated against the dataset and the budget, and then
dropped. `tree_sparse._check_sparse_forced_splits` refuses it now. There is
no sparse grower on any backend that applies one, so this is not a backend
choice; densify and use `tree.grow_tree`.
```

### H7. CORRECTION. `src/mojotrees/gpu_split_search.mojo:7881`, stale line reference

`train_gpu.mojo` line 1034 is not `set_monotone`'s caller. Current text:

```mojo
        Called once per tree by the trainer (`train_gpu.mojo` line 1034) and
```

Replace with:

```mojo
        Called once per tree by the trainer, from
        `train_gpu.GpuSplitSearcherCache.reset_for_tree` (train_gpu.mojo:1927
        at the head of 2026-08-17; this docstring said line 1034 until then,
        which is not a call site), and
```

### H8. CORRECTION. `tests/test_gpu_training.mojo:587-589`, stale docstring

The device-scan arm rejects candidates in the kernel, not host-side. Current
text:

```mojo
    """Monotonic constraints are enforced host-side, on downloaded histograms,
    so the GPU trainer must produce a monotone model too, and the same one the
    CPU trainer does to Float32 tolerance."""
```

Replace with:

```mojo
    """The GPU trainer must produce a monotone model, and the same one the CPU
    trainer does to Float32 tolerance.

    This said constraints "are enforced host-side, on downloaded histograms"
    until 2026-08-17. That is true of ONE of the two arms. On the host-scan
    arm the whole rule runs in `tree._search`. On the device-scan arm the
    candidate rejection is in the scan kernel, gated by
    `GpuSplitSearcher.constrained` and reading `mono_dev`, and only the
    child-value clamp and the midpoint collapse stay on the host, in
    `train_gpu._commit_device_split`, in Float64.

    WHICH ARM THIS TEST TAKES IS NOT ASSERTED, and it should be. Monotone
    constraints are not in `ExtraTreeParams.device_unsupported_reason`, so
    this configuration is eligible for the device scan, and since the
    2026-08-17 threshold withdrawal in `gpu_split_policy.decide_split_search`
    an eligible shape takes the device scan at every size on observed
    hardware. So this most likely exercises the device arm on an M4 and the
    host arm elsewhere, and nothing here records which. Under
    `bench/results/LANE_RULES.md` a test for a conditional path must prove the
    gate opened; this one does not."""
```

### H9. DEFECT, no correction text because the answer is a decision

`gpu_split_search.device_search_eligibility`, `device_search_reason`, and the
four `SEARCH_*` codes have no caller anywhere. The docstring at :5706 states
the intended caller in the future tense. The live gate is
`ExtraTreeParams.device_unsupported_reason`, which asks one question per
parameter and is strictly better than this list, which is a single bundled
`SEARCH_EXTRA_PARAMS` code.

Two acceptable resolutions and one that is not. Delete them, which moves no
bit; or keep them and mark them plainly as an unreached alternative with the
live gate named, so the next reader does not repair the wrong one. What is
not acceptable is leaving :5706's future tense standing, because it reads as
a pending patch rather than as a superseded design, and this sweep spent real
time confirming the reroute happens through a different function than the one
the list advertises.

### H10. Harness arms, ranked, at most three

Another lane owns `bench/real_data/scenarios.py`. **Zero of this family is
set to a non-default value by any arm today**; `min_gain_to_split` appears at
line 366 set to `0.0`, which is its default.

1. **`mojotrees monotone leafwise`**, monotone constraints on a real
   scenario, run on both `cpu` and `gpu` so `verify.check_device_agreement`
   compares them row by row. This is the highest-value arm in the family by a
   distance, because it is the only parameter here that is HONORED on the device
   rather than refused, so it is the only one where a device regression can
   produce a wrong answer rather than a refusal, and a violated monotone
   constraint is a broken promise about model behavior rather than a metric
   difference. It is also the arm that would have caught the device-scan
   Cosine bug's sibling if one existed, because it is the same launch path.
   Two notes for whoever writes it. The constraint must be on a feature the
   model actually uses or the arm proves nothing, so pick the sign from the
   scenario's own importance ranking rather than hardcoding feature 0. And
   the accuracy column will be WORSE than the unconstrained arm by
   construction, because `basic` gives up accuracy for the guarantee, so this
   arm needs its own anchor and must not be read against the plain one.

2. **`mojotrees extra-bundle refusal`**, a NEGATIVE arm. Set
   `min_gain_to_split` to something positive with `device='gpu'` and
   `split_search=device` and assert the refusal fires with
   `min_gain_to_split` in the message. Six parameters in this family share
   one refusal predicate and a single silent retirement of it would drop all
   six at once, which is exactly what happened to `score_function` and
   `random_strength` on 2026-08-17. A negative arm is cheap and it is the
   only thing that notices a refusal going away.

3. **`mojotrees monotone symmetric`**, CPU only, since the GPU symmetric
   route refuses. It closes the one honored cell in the family with no gate
   at all. The CPU symmetric grower applies the constraint through
   `find_best_split_shared` with one interval per leaf of the level, which is
   a different mechanism from the leaf-wise clamp and is tested by nothing.

Deliberately NOT proposed, an arm for `forced_splits`. The defect found here
is fixed by a refusal, and a benchmark arm is the wrong instrument for a
refusal on a path that will now raise. A one-line test asserting the raise
belongs in the test suite instead, and it is not this lane's file.
