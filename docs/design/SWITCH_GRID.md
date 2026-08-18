# The switch grid

Every `MOJOTREES_*` environment switch under `src/` and `bindings/`, what
reads it, what it defaults to, which growth policy and which backend it
actually reaches, and whether any number in this repository stands behind it.

Built on 2026-08-17 by reading source at head. **Nothing here was measured by
this lane and nothing was built or run.** Where a number appears it is quoted
from a results file, a docstring that records a measurement, or from the
briefing that opened this lane, and the source of it is named in the row.

Files move under this document, so every citation quotes the text it points
at rather than resting on a line number.

> **Section 5 has been actioned and this grid is now a snapshot of the state
> it audited, not of head.** All nine dead switches were resolved on
> 2026-08-17, the same day, by a follow-on lane. Eight were deleted and one
> was already only a docstring line. The switches named in section 5, in
> section 3K's table, and in the `MOJOTREES_GPU_GRAD_LAYOUT` row of section 3
> **no longer exist in the source**, so read those rows as the finding that
> justified the deletion rather than as a description of code you can grep
> for. The verdicts, the evidence, and the tombstones are in
> [DEAD_SWITCH_RESOLUTION.md](DEAD_SWITCH_RESOLUTION.md). Every other row in
> this grid is untouched.

> **Two CPU layout rows were also actioned on 2026-08-17, later the same
> day.** `MOJOTREES_CPU_LAYOUT_BY_NODE` and `MOJOTREES_CPU_BIN_LAYOUT_PROBE`
> were both unreachable from the symmetric CPU grower. Both are now wired
> into it rather than deleted, because the rule they carry is a property of
> node size and a symmetric tree has small nodes like any other. Their rows
> in section 3I, their entry in section 4, item 3 of section 6 and item J of
> section 7 are edited in place and say what changed. The measurement
> consequence is stated in section 6, item 3, and it matters more than the
> wiring. A symmetric reading of the by-node switch recorded as "neutral" is
> a null by construction and not a result.

> **FOUR GPU PERFORMANCE SWITCHES WERE MEASURED AND THEIR DEFAULTS FLIPPED ON
> 2026-08-17, later the same day, and this grid described none of the flips
> until now.** `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT`,
> `MOJOTREES_GPU_OBLIVIOUS_WIDE`, `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` and
> `MOJOTREES_GPU_SPLIT_WIDE` are all spelled `!= "0"` at head and are therefore
> **ON unless refused**. Every one of their four rows previously stated the
> opposite predicate (`== "1"`), stated that unset behaves as off, and carried a
> verdict of SHOULD BE THE DEFAULT or NEEDS MEASURING. All of that was true when
> the grid was built in the morning and false by the evening. The rows in
> sections 3A and 3B, the polarity note in section 2, the counts and ranks 1
> through 4 in section 4, and interaction A in section 7 are edited in place and
> say what they used to say. `docs/design/GROWTH_POLICY_REACH.md` records the
> same four flips and agrees with this grid at head.
>
> The fifth arm of that group, `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST`, did NOT
> flip, and the reason is that it was measured and came in indistinguishable.
> Its predicate is still `== "1"` and its default is still off. A measured null
> is not an unmeasured switch and its row now says which it is.

> **UPDATED 2026-08-18.** Four changes. **(1)** Two switches were added to the
> tree and have rows now, `MOJOTREES_PREDICT_TILE`, default **ON** and measured
> (section 3L, a section this grid did not have), and
> `MOJOTREES_BINNING_SORTED_TIE_REPAIR`, default off and unmeasured
> (section 3I). **(2)** The `MOJOTREES_GPU_ROW_COMPACTION` row in section 3D
> read `ALL GPU` and `CORRECTLY OFF, on a prediction rather than a
> measurement`; both are wrong at head. It is now REFUSED under
> `grow_policy=oblivious`, and the consumer it was waiting for was built and
> measured a loss. The row says what it used to say. **(3)** Section 1's
> reconciliation is recomputed, 90 to 108 raw hits and 85 to 86 live readers,
> by the grep the section itself states. **(4)** Section 8 gains item 6, six
> live switches that have no row here at all.
>
> **Two switches that existed earlier on 2026-08-18 are deliberately absent.**
> `MOJOTREES_CPU_OVERSUBSCRIBE` was deleted because its fix became
> unconditional, and `MOJOTREES_NOISE_DRAW_WEIGHT` was measured a null and
> deleted (`DECLINED_OPTIMIZATIONS.md` E15). Neither gets a row. A grid row is
> a promise that a reader can set the variable.

---

## 1. The population, and how it reconciles

**RECOMPUTED 2026-08-18, by the method stated on the first line, which is the
only reason the number could be moved at all.** The as-audited arithmetic on
2026-08-17 is kept below it, because a reconciliation whose old value is
discarded cannot be checked.

    grep -rho 'MOJOTREES_[A-Z0-9_]*' src/ bindings/ | sort -u    -> 108 hits
    names some code reads, as a quoted literal                   ->  86
    prefix and docstring line-wrap fragments, not names          ->   4
    a build-script shell variable, not a runtime switch          ->   1
    tombstones: named in prose, no reader anywhere               ->  17
                                                                     ---
                                                                     108

The 22 that are not real switches, by kind. A name counts as read when it
appears somewhere under `src/` or `bindings/` as a double-quoted literal, which
is what `getenv`, `_env_int` and a `comptime` alias all require and what prose
never uses, since prose spells these in backticks.

- **Four fragments.** `MOJOTREES_` the bare prefix, plus three line-wraps,
  `MOJOTREES_GPU_SPLIT_`, `MOJOTREES_GPU_NOISE_STAGE_`, and
  `MOJOTREES_DIST_`, the last of these from prose writing `MOJOTREES_DIST_*`.
- **One shell variable.** `MOJOTREES_TARGET_FLAGS`.
- **Seven distributed tombstones.** The `MOJOTREES_DIST_*` set of section 3K,
  deleted 2026-08-17 and still named in `distributed_transport.mojo` prose that
  records what was removed.
- **Nine arm tombstones.** `MOJOTREES_GPU_GRAD_LAYOUT` and
  `MOJOTREES_STARTUP_REPORT_FD` from the 2026-08-17 sweep;
  `MOJOTREES_GPU_HIST_LEAN`, `MOJOTREES_GPU_HIST_PAIR_GRID` and
  `MOJOTREES_GPU_HIST_PRIVATE`, geometry knobs on two accumulation kernels that
  were built, measured bit-identical and null or worse on 2026-08-17, and
  removed with their kernels (`DECLINED_OPTIMIZATIONS.md` rows E11 to E13);
  `MOJOTREES_GPU_BATCH_QUANT` (E10, 1.018x null) and
  `MOJOTREES_GPU_BATCH_CONST_HESS` (E14, 1.016x null); and, on 2026-08-18,
  `MOJOTREES_GPU_OBLIVIOUS_COMPACT_BINS` (C1, 0.757x loss) and
  `MOJOTREES_CPU_OVERSUBSCRIBE` (the fix became unconditional). **None of these
  gets a row, and neither does `MOJOTREES_NOISE_DRAW_WEIGHT`**, which was
  measured a null the same day (`DECLINED_OPTIMIZATIONS.md` E15) and deleted so
  completely that it does not even survive in prose; it appears now only in
  frozen `bench/real_data/` result artifacts. **A grid row for a name no code
  reads is how a reader gets sent looking for a switch that is not there.**
- **One foreign name.** `MOJOTREES_CATBOOST_MODE` is `bench`'s harness
  variable, named once in `gpu_resident_round.mojo` prose. It configures a
  benchmark arm, not a fit, and it is out of this grid's boundary for the same
  reason `MOJOTREES_TARGET_FLAGS` is.

**The drift is 18 raw hits and most of it predates 2026-08-18.** Nine names of
it are the tombstone prose left behind by the 2026-08-17 deletions, which the
old arithmetic counted as five specials and no tombstone class at all. The
population of names code actually reads moved 85 to 86, which is the honest
headline, two added this session (`MOJOTREES_PREDICT_TILE`,
`MOJOTREES_BINNING_SORTED_TIE_REPAIR`), two deleted
(`MOJOTREES_CPU_OVERSUBSCRIBE`, `MOJOTREES_NOISE_DRAW_WEIGHT`), one deleted the
day before that the old count had already added
(`MOJOTREES_GPU_OBLIVIOUS_COMPACT_BINS`), and `MOJOTREES_GPU_SKIP_TERMINAL_CHILDREN`
added on 2026-08-17. **A grep count is not a switch count**, and treating the
two as one number is what let 90 stand while the tree moved under it.

The 2026-08-17 arithmetic, as audited, is preserved below.

    grep -rho 'MOJOTREES_[A-Z0-9_]*' src/ bindings/ | sort -u    ->  90 hits
    real switches that code reads                                ->  85
    documented but read by nothing                               ->   1
    a build-script shell variable, not a runtime switch          ->   1
    docstring line-wrap fragments, not names                     ->   3
                                                                     ---
                                                                      90

The five that were not real switches then.

| raw hit | what it actually is |
|---|---|
| `MOJOTREES_` | the bare prefix, written in prose in ten docstrings, for example `gpu_runtime.mojo` "Environment contract, matching the `MOJOTREES_` convention in parallel.mojo" |
| `MOJOTREES_GPU_SPLIT_` | a line-wrap. `gpu_split_search.gain_form_requested` writes "the same posture `MOJOTREES_GPU_SPLIT_" and continues "PRIMITIVES` and `histogram_gpu.set_scale_shape` take" on the next line |
| `MOJOTREES_GPU_NOISE_STAGE_` | a line-wrap of `MOJOTREES_GPU_NOISE_STAGE_PARALLEL`, in a comment inside `gpu_split_search.stage_random_score_level` |
| `MOJOTREES_STARTUP_REPORT_FD` **(DELETED 2026-08-17)** | named once, in `initialization.mojo` module prose, as "reserved, unread here". No `getenv` anywhere in the tree read it, and the prose line is now gone too. See section 5 |
| `MOJOTREES_TARGET_FLAGS` | a shell variable in `bindings/build.sh`, "`$MOJOTREES_TARGET_FLAGS` is deliberately unquoted", expanded into a `mojo build` command line. It is not read by Mojo code and configures the compiler, not a fit |

The 85 remaining names all appear as a double-quoted literal in a `getenv`
call, in an `_env_int(name, default)` call, or as a `comptime` alias that a
`getenv` then takes. Every one of the 85 was traced to the function that
reads it.

Two of those five rows need a footnote at head.
`MOJOTREES_STARTUP_REPORT_FD`'s row says "the prose line is now gone too"; the
deleted line is gone, and `initialization.mojo` now carries a tombstone line
that names the variable while recording its removal, which is why the grep
still returns it. It is a tombstone rather than a reader, and it is counted as
one in the recomputed arithmetic above. And the fragment class has gained a
fourth member since, `MOJOTREES_DIST_`, from prose writing `MOJOTREES_DIST_*`
in `distributed_transport.mojo`.

**The population moved twice on 2026-08-17 and twice more on 2026-08-18, and
the preserved arithmetic is the as-audited one.** On 2026-08-17 nine names were
deleted (section 5) and one was added, `MOJOTREES_GPU_SKIP_TERMINAL_CHILDREN`,
read by `train_gpu.skip_terminal_children_enabled`, which has a row in
section 3B marked as postdating the grid. On 2026-08-18 two more were added,
`MOJOTREES_PREDICT_TILE` (section 3L) and
`MOJOTREES_BINNING_SORTED_TIE_REPAIR` (section 3I), and two were deleted after
measurement, `MOJOTREES_CPU_OVERSUBSCRIBE` and `MOJOTREES_NOISE_DRAW_WEIGHT`.
**The counts have been re-derived from the grep**, which is the recomputation
at the top of this section; it is safe to move a number here only because the
method that produces it is written down.

An earlier audit, `bench/results/INSTRUCTION_AUDIT.md` section 9, works from a
different population of 68. That list is `compatibility/api_snapshot.json`'s
`environment.observed`, which is a literal scan over a wider tree, so it
contains names this grid does not (`MOJOTREES_HYBRID_LEAVES`,
`MOJOTREES_HYBRID_COSTS`, `MOJOTREES_HYBRID_TRACE`,
`MOJOTREES_HYBRID_GUARD_TRANSFER`, `MOJOTREES_BUILD_LOCK`,
`MOJOTREES_UM_LADDER_PCT`, `MOJOTREES_PIXI_MANIFEST`, and the
`MOJOTREES_DISTRIBUTED_*` trio in `python/mojotrees/_dask_runtime.py`) and
omits names this grid has. The two counts are not in conflict; they scan
different directories with different rules. This grid's boundary is exactly
`src/` and `bindings/`, as assigned.

---

## 2. How to read the columns

**Default.** Three polarities exist in this tree and the spelling carries
meaning, stated by `gpu_resident_round.speculative_build_enabled`. An
inequality against `"0"` means the arm is **ON** unless refused and is
reserved for a default the authors did not think needed arguing. An equality
against `"1"` means the arm is **OFF** unless asked for and is how an unproven
arm is spelled. A parsed word or integer means the switch **selects** among
named arms and unset picks a stated one.

Since 2026-08-17 the inequality also carries a second meaning that the sentence
above did not cover, and four rows of this grid now use it. Under LANE_RULES
rule 5 a proven arm's default flips in the session that measures it, and the
variable then survives one round spelled `!= "0"` as an escape hatch that turns
the new behavior OFF before it is deleted. So an inequality is no longer only "a
default the authors did not think needed arguing"; it is also "an arm that was
argued, measured, and shipped", and the two are told apart by the flip comment
beside the `getenv`.

**Kind.** PERFORMANCE means same answer, different speed. BEHAVIOR means the
model or the numbers move. DIAGNOSTIC means trace, census, or profile output.
CONFIGURATION means it picks a backend, a worker count, a role, or a geometry.

**Reaches.** LEAF (lossguide), DEPTH (depthwise), SYM (oblivious, the CatBoost
shape), and CPU or GPU. `ALL` means all three policies on that backend.

**Verdict.** The five assigned values, plus one convention the assignment did
not cover. Fourteen switches were already default ON when this grid was built,
so "should it be the default" is answered by the shipped state. Those carry
**SHOULD BE THE DEFAULT (already is)** and are excluded from the ranked
candidate list in section 4, because there is nothing to flip. Where such a
switch's off arm has never been priced that is said in the Measured column,
not laundered into the verdict.

**Four more joined them later on 2026-08-17**, by being measured and flipped
rather than by having shipped that way, so the shipped population of default-ON
switches is eighteen and not fourteen. Those four carry
**SHOULD BE THE DEFAULT, AND IS SINCE 2026-08-17** so that the verdict records
the movement instead of reading as though it always said this, and they stay
listed at ranks 1 through 4 of section 4 as closed items rather than being
deleted from it.

---

## 3. The grid

### 3A. The symmetric (oblivious) GPU plane

This is where the four switches measured on 2026-08-17 live. All four are read
in, or dispatched from, code that only `grow_tree_device_oblivious` reaches.

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT` | `gpu_leaf_batching.oblivious_subtract_requested`, `return getenv("MOJOTREES_GPU_OBLIVIOUS_SUBTRACT") != "0"`. Dispatched at `histogram_gpu.enqueue_desc_level_children`, `if oblivious_subtract_requested():` | **unset behaves as ON since 2026-08-17.** This cell read "unset behaves as off" against a predicate spelled `== "1"`, which was the state the grid audited and is not the state at head | PERFORMANCE | SYM GPU only. The branch is placed in `histogram_gpu` and not inside the batcher "so that a two-item leaf-wise plan cannot reach the subtracting arm by having the environment set" | **MEASURED**, 1.78x, 21.97 s to 12.34 s at 799,110 x 100 x 100 trees, rmse identical to nine decimals (2026-08-17 lane brief). The flip comment at `oblivious_subtract_requested` records a second interleaved reading the same day, three round-robin cycles at the same shape, 22.76 s to 14.39 s alone and 10.36 s with the wide scan and the skipped last build, **2.20x combined**, rmse 2.439382420 in every arm of every cycle. Bit-identity is an exact integer argument at `_batch_hist_atomic_subtract_kernel`. `docs/design/OBLIVIOUS_WAIT_CENSUS.md` tabulates the row-build counts, 126 to 63 at depth 6 | **SHOULD BE THE DEFAULT, AND IS SINCE 2026-08-17.** Flipped in the session that measured it, under LANE_RULES rule 5. The variable survives one round as an escape hatch turning the subtraction OFF and is then deleted |
| `MOJOTREES_GPU_OBLIVIOUS_WIDE` | `gpu_split_search.oblivious_wide_scan_requested`, `return getenv("MOJOTREES_GPU_OBLIVIOUS_WIDE") != "0"`. Dispatched in the oblivious launch, `if oblivious_wide_scan_requested():`, which then refuses a bin count above `OBLIVIOUS_WIDE_MAX_BINS_PER_THREAD * OBLIVIOUS_WIDE_THREADS` | **unset behaves as ON since 2026-08-17.** This cell read "unset behaves as off" against a predicate spelled `== "1"`, which was the state the grid audited and is not the state at head | PERFORMANCE | SYM GPU only | **MEASURED**, 4.5 percent, resolved, bit-identical (2026-08-17 lane brief). The flip comment at `oblivious_wide_scan_requested` holds both readings of that day, 20.40 s narrow against 19.49 s wide with disjoint ranges on a quiet box, and 22.76 s against 21.28 s, 1.07x, in a loaded three-cycle round robin that also produced the 2.20x combined arm. rmse 2.439382420 throughout. No results file under `bench/results/` names this variable, so the measurement is still filed only in source and in the brief | **SHOULD BE THE DEFAULT, AND IS SINCE 2026-08-17.** Flipped in the session that measured it, under LANE_RULES rule 5, and the variable now restores the narrow `block_dim=1` scan |
| `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` | `gpu_resident_round.oblivious_skip_last_build_requested`, `return getenv(OBLIVIOUS_SKIP_LAST_BUILD_VAR) != "0"`. Consumed at `var skip_last_build = oblivious_skip_last_build_requested()` above the level loop, and again in `train_gpu` to size the profile's launch count | **unset behaves as ON since 2026-08-17.** This cell read "unset behaves as off" against a predicate spelled `== "1"`, which was the state the grid audited and is not the state at head | PERFORMANCE | SYM GPU only | **MEASURED**, **1.26x** alone, 22.76 s to 18.06 s at 799,110 x 100 x 100 trees over three interleaved round-robin cycles, and part of the 2.08x-to-2.20x combined arm; rmse 2.439382420 unchanged in every cycle (flip comment at `oblivious_skip_last_build_requested`, and the same figure at `oblivious_schedule_launches`'s neighbouring note). An earlier reading of "about 1.20x, a lower bound because the box drifted" is superseded by it. The launch count moves 56 to **55** at depth 6, which `oblivious_schedule_launches(6, skip_last_build=True)` returns and which is deliberately the least interesting part of the change | **SHOULD BE THE DEFAULT, AND IS SINCE 2026-08-17.** Flipped in the session that measured it, under LANE_RULES rule 5. Nothing reads the histograms it skips, so building them was a defect rather than a trade |
| `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST` | `gpu_resident_round.oblivious_noise_hoist_requested`, `return getenv(OBLIVIOUS_NOISE_HOIST_VAR) == "1"`. Also read in `train_gpu._search_record_slots`, `if oblivious_noise_hoist_requested(): want = oblivious_leaf_budget(params) + params.max_depth` | unset behaves as off | PERFORMANCE | SYM GPU only, **and only when `random_strength > 0`**. `_copy_noise` returns immediately at zero. The CatBoost-mode default set does set it, `params.CATBOOST_RANDOM_STRENGTH` is 1.0 | **MEASURED, AND THE RESULT IS A NULL.** It was run on 2026-08-17 and came in **indistinguishable** in the registered M0 sense (`PROFILE_PROTOCOL.md` M0), which is why it is the one arm of the four that did not flip (2026-08-17 lane brief). This cell previously read "UNRESOLVED, measured slower in a run confounded by drift, so neither result stands"; the standing reading is a measured null rather than a withdrawn one. The count behind it is still read from source, six `enqueue_copy` drains per depth-6 tree, each between a level's child build and the next level's search. Not yet filed under `bench/results/`, and the docstring at `gpu_resident_round.OBLIVIOUS_NOISE_HOIST_VAR` still says the time is unmeasured, which is stale | **CORRECTLY OFF on a measured null**, not on an absent measurement. Re-measuring is optional rather than owed, and only on a fit that sets `random_strength > 0` |

### 3B. GPU split search, shared and leaf-wise

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_GPU_SPLIT_WIDE` | `gpu_split_search.wide_scan_requested`, `return getenv("MOJOTREES_GPU_SPLIT_WIDE") != "0"`, reached through `wide_scan_for(has_categorical)`, which ANDs it with "no categorical feature", and stored once at construction as `self.wide_scan = wide_scan_for(any_cat)` | **unset behaves as ON since 2026-08-17.** This cell read "unset behaves as off" against a predicate spelled `== "1"`, which was the state the grid audited and is not the state at head | PERFORMANCE | **LEAF and DEPTH GPU (device split search), corrected 2026-08-17.** This cell read "Not DEPTH in practice, because the device-resident plane refuses non-leaf-wise growth", and that inference does not hold: the refusal is about the device-owned growth plane, not about the searcher. `self.wide_scan` is set once at the single `GpuSplitSearcher` construction in `train_gpu` and both entry points pass it to `_launch` (`enqueue` for one record, `enqueue_frontier` for a batch), so a depth-wise fit scans wide too. This is what `docs/design/GROWTH_POLICY_REACH.md` already recorded, and section 8 item 1 asked for exactly this trace. Not SYM, which runs the separate oblivious kernel | **MEASURED**, and this cell used to read ASSERTED and NEEDS MEASURING. The interleaved pair the docstring asked for was run the same day, three round-robin cycles at 799,110 x 100 continuous features, leaf-wise at the shared defaults, 100 trees, M4: narrow 3.922 / 3.874 / 3.932 s against wide 3.220 / 3.196 / 3.231 s, disjoint ranges so M0 **resolved**, rmse 6.116601511 in all six runs. **1.21x**, four times what the oblivious arm got, on a plane where the scan is a larger share because it scans two children rather than a whole level. `wide_scan_for` ANDs the request with "no categorical", so a categorical dataset measures the narrow kernel on both arms and reports a null | **SHOULD BE THE DEFAULT, AND IS SINCE 2026-08-17.** Flipped in the session that measured it, under LANE_RULES rule 5, and the variable now restores the narrow kernel |
| `MOJOTREES_GPU_SKIP_TERMINAL_CHILDREN` **(ADDED TO THE TREE AFTER THIS GRID WAS BUILT, 2026-08-17)** | `train_gpu.skip_terminal_children_enabled`, `return getenv(SKIP_TERMINAL_CHILDREN_VAR) == "1"`, over `comptime SKIP_TERMINAL_CHILDREN_VAR`, read once per tree | unset behaves as off, and the docstring gives the reason as "nothing has measured it" | PERFORMANCE | **LEAF and DEPTH GPU, in `_device_search_resident` only**, so leaf-wise reaches it only when the fit falls back off the device-owned plane. Not SYM: `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` is its level-shaped twin and is the shipped default | **ASSERTED**, with a counted case and no run. `docs/design/GROWTH_POLICY_REACH.md` records the arithmetic, roughly a 14 percent cut in histogram row traffic plus one round trip under depth-wise growth at 31 leaves with a binding depth, and the three legs of the bit-identity argument are written at `SKIP_TERMINAL_CHILDREN_VAR` | **NEEDS MEASURING.** Its symmetric twin was measured at 1.26x and this one removes strictly more work, since it drops the search as well as the build |
| `MOJOTREES_GPU_SPLIT_PRIMITIVES` | `gpu_split_search.split_primitives_requested`, `return getenv("MOJOTREES_GPU_SPLIT_PRIMITIVES") != "0"`, stored as `self.use_primitives` | unset behaves as **ON** | PERFORMANCE | LEAF and SYM GPU, one searcher serves both | **ASSERTED** for speed. Bit-equality is asserted field for field by `tests/test_gpu_split_scan.mojo` | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_GPU_SPLIT_TABLE_PACK` | `gpu_split_search.table_upload_hoisting_requested`, `return getenv("MOJOTREES_GPU_SPLIT_TABLE_PACK") != "0"`, stored as `self.hoist_tables` | unset behaves as **ON** | PERFORMANCE | LEAF and SYM GPU | **ASSERTED.** "the packed arm writes the device exactly the bytes the four-copy arm writes ... and differs only in how many times the host blocks" | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_GPU_NOISE_STAGE_PARALLEL` | `gpu_split_search.noise_stage_parallel_requested`, `return getenv("MOJOTREES_GPU_NOISE_STAGE_PARALLEL") != "0"`, consumed as `if not noise_stage_parallel_requested():` inside `stage_random_score_level` | unset behaves as **ON**. This is the switch a grid that assumed off would get wrong | PERFORMANCE | LEAF and SYM GPU, only where a noise plane is staged, so only at `random_strength > 0`. Under the CatBoost-mode default that is every symmetric fit | **ASSERTED** in seconds, argued in work. The docstring counts 15.3M serial draws with a `log` and a `sqrt` each at the shipped symmetric default, moved across workers. Exactness is argued from `host_random_score_noise` being a pure function of six arguments | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_GPU_SPLIT_GAIN_FORM` | `gpu_split_search.gain_form_requested`, `GAIN_FORM_SUBTRACTIVE if getenv("MOJOTREES_GPU_SPLIT_GAIN_FORM") == "subtractive" else DEFAULT_GAIN_FORM`, and `DEFAULT_GAIN_FORM = GAIN_FORM_CROSS` | unset selects `cross`, the cancellation-free form. The word `subtractive` selects back to the older gain. Any other value silently leaves the default | BEHAVIOR. The two forms are different arithmetic over the same histogram, so a near-tie can resolve differently | LEAF and SYM GPU. `gpu_resolve_gain_form` overrides to subtractive whenever `lambda_l1 != 0`, because the cross identity is invalid under soft thresholding | **ASSERTED**. No results file names it | CORRECTLY OFF (it is the escape hatch back, not a candidate) |
| `MOJOTREES_GPU_SPLIT_RESIDENT` | `train_gpu.resident_frontier_disabled`, `return getenv("MOJOTREES_GPU_SPLIT_RESIDENT") == "0"` | unset behaves as **ON**, the resident frontier runs wherever it fits | CONFIGURATION | LEAF, DEPTH and SYM GPU. On SYM it gates `opened` at the oblivious route and a `0` therefore makes a symmetric GPU fit **raise**, not fall back. See section 7 | **MEASURED**, memory records 5.40 s to 3.15 s at 250k and a loss at 50k; `docs/LIGHTGBM_PARITY.md` names the variable | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_GPU_SPLIT_STRATEGY` | `train_gpu.env_split_search`, `var s = getenv("MOJOTREES_GPU_SPLIT_STRATEGY")` mapping `device` and `host` to constants | unset selects AUTO | CONFIGURATION | ALL GPU | **MEASURED**, `bench/results/sweep2_2026-08-15/RESULTS.md`, the forced-gate A/B at 20,000 rows with `=device` | DIAGNOSTIC in the candidate sense; it selects a backend rather than proposing one |
| `MOJOTREES_GPU_SPLIT_TRACE` | `train_gpu.split_trace_enabled`, `getenv("MOJOTREES_GPU_SPLIT_TRACE") == "1" or getenv("MOJOTREES_GPU_PHASE_TRACE") == "1"` | unset behaves as off | DIAGNOSTIC | ALL GPU | n/a | DIAGNOSTIC |
| `MOJOTREES_GPU_PHASE_TRACE` | same function, plus two `var phase_trace = getenv("MOJOTREES_GPU_PHASE_TRACE") == "1"` sites in the round loops | unset behaves as off | DIAGNOSTIC | ALL GPU | n/a | DIAGNOSTIC |
| `MOJOTREES_GPU_READBACK` | `gpu_runtime.env_readback_transport`, `var raw = getenv("MOJOTREES_GPU_READBACK")`, matched against `readback_transport_name`, raising on an unknown word. Stored as `self.readback` on the searcher | unset selects `READBACK_DEFAULT` | CONFIGURATION | LEAF and SYM GPU | **MEASURED** per transport in the table the function guards, for example `READBACK_MAP` at 349.47 us a trip against `plain_one`'s 124.85. Three of seven rows are refused outright | CORRECTLY OFF |

### 3C. The device-resident growth plane

`gpu_tree_tables.tree_resident_supported` contains `if params.grow_policy != GROW_LEAFWISE: return TREE_RESIDENT_DEPTHWISE`, so this whole plane is **leaf-wise only**. Depth-wise GPU growth falls through to `_device_search_incremental`.

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_GPU_TREE_RESIDENT` | three readers over one name. The gate is `gpu_resident_round.resident_round_enabled`, `return getenv("MOJOTREES_GPU_TREE_RESIDENT") != "0"`. A diagnostic is `resident_round_explicitly_requested`, `== "1"`. A third, `gpu_tree_tables.tree_resident_requested`, also `== "1"`, is a stale duplicate | unset behaves as **ON** at the gate, off at the two `== "1"` readers | CONFIGURATION | LEAF GPU only, per the refusal above | **MEASURED**, `bench/results/session3_2026-08-16/RESULTS.md`; the docstring says the plane "has three measured results behind it" | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_GPU_SPECULATION` | `gpu_resident_round.speculative_build_enabled`, `return getenv(SPECULATION_BUILD_VAR) == "1"`, three consumption sites in the resident loop | unset behaves as off | PERFORMANCE | LEAF GPU only. On SYM it is not merely inert, it **refuses the plane**, `if speculative_build_enabled(): return OBLIVIOUS_SPECULATION` | **PARTLY MEASURED.** `session3_2026-08-16/RESULTS.md` registers a 66.8 percent census hit rate at 1,000,000 rows and 964 wasted builds per fit. Neither arm has been timed end to end | **NEEDS MEASURING** |
| `MOJOTREES_GPU_SPECULATION_CENSUS` | `gpu_resident_round.speculation_census_sink`, `return getenv(SPECULATION_CENSUS_VAR)`. Empty is off; `1`, `stdout` or `-` print; anything else is an appended file path | unset behaves as off | DIAGNOSTIC | LEAF GPU | **MEASURED** as an instrument, `session3_2026-08-16/RESULTS.md` | DIAGNOSTIC |
| `MOJOTREES_GPU_FUSE_PARTITION_TAIL` | `gpu_resident_round.partition_fusion_enabled`, `return getenv(PARTITION_FUSION_VAR) != "0"`, one consumer, `builder.rows.set_partition_fusion(partition_fusion_enabled())` | unset behaves as **ON** | PERFORMANCE | LEAF GPU only. The oblivious loop sets fusion unconditionally and says so, "Not optional here, where it is merely the default on the leaf-wise plane" | **ASSERTED**, with a no-lose argument, "It issues one command buffer where the step used to issue two, storing the same values to the same addresses under the same guard" | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_GPU_TREE_RESIDENT_TRACE` | `gpu_resident_round.resident_trace_sink`, `return getenv(RESIDENT_TRACE_VAR)`. Same sink contract as the census | unset behaves as off | DIAGNOSTIC | LEAF and SYM GPU; the oblivious loop reads it too | **MEASURED** as an instrument, `session3_2026-08-16/RESULTS.md` quotes its output to prove a gate was open | DIAGNOSTIC |
| `MOJOTREES_GPU_TREE_RESIDENT_TRACE_STEPS` | `gpu_resident_round`, `return getenv(RESIDENT_TRACE_STEPS_VAR) == "1"`, ANDed with a live trace sink | unset behaves as off | DIAGNOSTIC | LEAF and SYM GPU | n/a. The docstring warns it "reinstates exactly the per-split synchronization the plane exists to remove", so no timing may be taken with it on | DIAGNOSTIC |
| `MOJOTREES_GPU_TABLE_RESET` | `gpu_tree_tables.DeviceTreeTables.__init__`, `self.reset_on_device = getenv("MOJOTREES_GPU_TABLE_RESET") != "0"` | unset behaves as **ON**, the device-kernel reset | PERFORMANCE | LEAF and SYM GPU | **ASSERTED** in seconds, counted in drains. `OBLIVIOUS_WAIT_CENSUS.md` note 3, the off arm "makes the same reset five `enqueue_copy` calls, so that arm costs five more drains per tree" | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_GPU_PACKED_DOWNLOAD` | same constructor, `self.packed_download = getenv("MOJOTREES_GPU_PACKED_DOWNLOAD") != "0"` | unset behaves as **ON** | PERFORMANCE | LEAF and SYM GPU | **ASSERTED**. The packed arm is one pack kernel, one pinned copy and one `synchronize()`; `OBLIVIOUS_WAIT_CENSUS.md` records the synchronize as load-bearing, "a pinned copy read without one returning 64 of 64 stale words" | SHOULD BE THE DEFAULT (already is) |

### 3D. GPU active rows, gradients and compaction

All of these are read once in `GpuActiveRows.__init__`, so they reach every
GPU growth policy that constructs a builder, unless a row says otherwise.

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_GPU_ROW_COMPACTION` | `GpuActiveRows.__init__`, `if _env_int("MOJOTREES_GPU_ROW_COMPACTION", 0) != 0: self.set_row_compaction(True)`. Also honored from `train_gpu`, `builder.rows.set_row_compaction(row_compaction or builder.rows.row_compaction_requested())` | unset behaves as off | PERFORMANCE | **LEAF and DEPTH GPU. REFUSED on SYM since 2026-08-18**, and this cell read `ALL GPU`. `histogram_gpu.GpuHistogramBuilder.stage_desc_level_plan` raises on `self.rows.row_compaction_requested()`, which is the field BOTH the variable and the `row_compaction: Bool` parameter set, so neither route reaches an oblivious fit. The reason is not that it was inert there, because the level build consumes no compacted plane, so the arm paid one rebuild plus two launches a level, 13 command buffers on a depth-6 tree, collecting only the root build out of 63 histograms, and 55 + 13 = 68 overruns a 64-deep Metal queue that does not raise when overrun. See `docs/design/GROWTH_POLICY_REACH.md` and `docs/design/PATH_COVERAGE.md` 3A.1 | **MEASURED, INDIRECTLY, AND NEGATIVE.** This cell read "**ASSERTED.** ... Never measured" beside an arithmetic prediction of 1.7x to 2.5x underwater at 1M x 50. The compaction itself is still unmeasured end to end, but the consumer it was waiting for was built as `MOJOTREES_GPU_OBLIVIOUS_COMPACT_BINS` and run on real data, 463,715 x 90, 100 trees, symmetric depth 6, both arms interleaved in ONE run, **0.757x on the device-MVS arm and 0.949x on the host-MVS one**, bit-identical models on both pairs. `docs/design/DECLINED_OPTIMIZATIONS.md` row C1 reads MEASURED NEGATIVE. The prediction was right in sign | **CORRECTLY OFF, and now on a measurement.** Not "CORRECTLY OFF" in the ordinary sense, because on the symmetric plane it is not off, it is refused, and on the two leaf-shaped planes it stays off with the gather-is-the-cost argument for it now refuted by C1 |
| `MOJOTREES_GPU_COMPACT_FLAG_READ` | same constructor, `self.compact_flag_read = _env_int("MOJOTREES_GPU_COMPACT_FLAG_READ", 0) != 0` | unset behaves as off | PERFORMANCE | ALL GPU, but **inert alone**. The comment states it, "inert on its own: it changes nothing at all unless the compaction arm above is also on" | **ASSERTED** | CORRECTLY OFF, and see section 7 |
| `MOJOTREES_GPU_COMPACTION_TRACE` | `gpu_active_rows._compact_trace_sink`, `return getenv(COMPACTION_TRACE_VAR)` | unset behaves as off | DIAGNOSTIC | ALL GPU | n/a | DIAGNOSTIC |
| `MOJOTREES_GPU_QUANTIZED_GRADS` | `GpuActiveRows.__init__`, `self.quantized_gradients = _env_int("MOJOTREES_GPU_QUANTIZED_GRADS", 1) != 0` | unset behaves as **ON** | PERFORMANCE | ALL GPU | **ASSERTED**, with an argument from the expression rather than a measurement. "Cannot change a histogram, only what the kernel gathers per row" | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_GPU_PACKED_GRADS` | same constructor, `self.packed_gradients = _env_int("MOJOTREES_GPU_PACKED_GRADS", 0) != 0` | unset behaves as off | PERFORMANCE | ALL GPU | **ASSERTED**, and counted against. `OBLIVIOUS_WAIT_CENSUS.md` records that it adds a genuine round trip per tree via `_check_stage16_bound` | CORRECTLY OFF |
| `MOJOTREES_GPU_SCAN_PRIMITIVES` | same constructor, `self.scan_primitives = _env_int("MOJOTREES_GPU_SCAN_PRIMITIVES", 1) != 0 and _scan_primitive_width_supported(threads)` | unset behaves as **ON**, subject to the width having an instantiation | PERFORMANCE | ALL GPU | **ASSERTED** with a strictly-less-work argument, "it computes the same permutation with strictly less work, so it does not need a measurement to justify being on" | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_GPU_FEATURE_GROUP` | two readers. `GpuActiveRows.__init__`, `var group = _env_int("MOJOTREES_GPU_FEATURE_GROUP", 1)`, rounded down to a rung. `GpuHistogramBuilder`, `if getenv("MOJOTREES_GPU_FEATURE_GROUP") == "":` then widens by `free_feature_group` | **unset selects auto.** Unset lets the builder widen to the free-footprint rule (baseline 2 on Metal). Set pins the width in both directions | CONFIGURATION | ALL GPU | **ASSERTED** for the widening, which is argued as an occupancy no-op. Anything wider is stated as unmeasured | CORRECTLY OFF |
| `MOJOTREES_GPU_VERIFY_ROWS` | two readers, deliberately. `GpuActiveRows.__init__`, `self.verify_counts = _env_int("MOJOTREES_GPU_VERIFY_ROWS", 0) != 0`, and `train_gpu.verify_rows_requested`, `== "1"` | unset behaves as off | DIAGNOSTIC | Effectively **refused** on both default GPU planes. `_check_verify_rows_reachable` raises on the resident plane and on the oblivious plane, naming `MOJOTREES_GPU_TREE_RESIDENT=0` as the way to reach it | n/a | DIAGNOSTIC, and see section 7 |
| `MOJOTREES_CONST_HESSIAN` | two readers. `histogram.const_hessian_allowed`, `return _env_int("MOJOTREES_CONST_HESSIAN", 1) != 0`, and the same expression in `GpuActiveRows.__init__` as `self.const_hessian_allowed` | unset behaves as **ON**, meaning the specialization is permitted. It still does nothing until a caller declares an objective that guarantees it | CONFIGURATION (a permission, not an election) | ALL, CPU and GPU | **ASSERTED** | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_CONST_HESSIAN_VERIFY` | `histogram.const_hessian_verify`, `return _env_int("MOJOTREES_CONST_HESSIAN_VERIFY", 0) != 0` | unset behaves as off | DIAGNOSTIC | ALL CPU. It cannot run on GPU; `device_policy` routes `auto` around it because "the audit is a host walk over the host hessian array ... no GPU builder can" | n/a. It is one extra pass over `n_rows` per build | DIAGNOSTIC |

### 3E. GPU histogram geometry and specialization

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_GPU_HIST_STRATEGY` | `gpu_tiling.env_strategy`, `var s = getenv("MOJOTREES_GPU_HIST_STRATEGY")`, mapping `atomic` and `tiled` | unset selects AUTO | CONFIGURATION | ALL GPU | **ASSERTED** in this tree. `INSTRUCTION_AUDIT.md` names it without a results row | CORRECTLY OFF |
| `MOJOTREES_GPU_ROW_TILE` | `gpu_tiling.resolve_tiling`, `forced_rows = _env_int("MOJOTREES_GPU_ROW_TILE", 0)`, and an explicit `rows_per_tile_request` argument outranks it | unset and `0` are the same, meaning no override | CONFIGURATION | ALL GPU | **ASSERTED** | CORRECTLY OFF |
| `MOJOTREES_GPU_MIN_TILES` | `gpu_tiling.env_min_tiles`, `if getenv("MOJOTREES_GPU_MIN_TILES") == "device": return -1`, else `_env_int(..., 0)` | unset and `0` mean no floor beyond the occupancy term; the word `device` asks for the device-wide floor | CONFIGURATION | ALL GPU | **MEASURED and negative.** "It is an opt-in rather than the default because it was measured slower at every shape tried". This is the tile-floor experiment the brief says not to re-litigate | CORRECTLY OFF |
| `MOJOTREES_GPU_BLOCK_THREADS` | two readers with identical text, `gpu_tiling.derive_block_threads` and `apple_histogram_policy._shape_block_threads`, both `var requested = _env_int("MOJOTREES_GPU_BLOCK_THREADS", 0)` | `0` or unset selects the derived target | CONFIGURATION | ALL GPU | **ASSERTED** | CORRECTLY OFF |
| `MOJOTREES_GPU_HIST_SPECIALIZATION` | `apple_histogram_policy.env_specialization_level`, matching `shape`, `packed`, `batched` | unset, empty and unrecognized all select `SPEC_LEVEL_BASELINE`. "There is no `auto`" | CONFIGURATION | ALL GPU | **ASSERTED**. `bench/results/PHASE2_PREREGISTRATION.md` registers it; no result file reports a number for it | NEEDS MEASURING, at low rank. The `batched` rung is the one the histogram file points at |
| `MOJOTREES_GPU_BATCH_SLOTS` | `histogram_gpu.env_batch_slots`, `var n = _env_int("MOJOTREES_GPU_BATCH_SLOTS", DEFAULT_BATCH_SLOTS)`, clamped to `[2, MAX_BATCH_SLOTS]`. One consumer, `var want = pool_slots if pool_slots > 0 else env_batch_slots()` | unset selects `DEFAULT_BATCH_SLOTS` | CONFIGURATION | ALL GPU, but "Only read when batching was requested at all, so the default path never consults it", which ties it to `HIST_SPECIALIZATION=batched` | **ASSERTED** | CORRECTLY OFF, and see section 7 |
| `MOJOTREES_GPU_CLASS_BATCH` | two readers, `gpu_output_planes.env_class_batch` and `apple_histogram_policy`, both `_env_int("MOJOTREES_GPU_CLASS_BATCH", 0)` | `0` or unset selects auto | CONFIGURATION | ALL GPU, multiclass only | **MEASURED**, `bench/results/profile_2026-08-15/RESULTS.md`, a results row reading `mojotrees GPU, MOJOTREES_GPU_CLASS_BATCH=7` at 15.45 s and 0.8 percent | CORRECTLY OFF |
| `MOJOTREES_GPU_CLASS_BATCH_BYTES` | `gpu_output_planes.env_class_batch_budget`, `_env_int("MOJOTREES_GPU_CLASS_BATCH_BYTES", CLASS_BATCH_BUDGET_BYTES)` | unset selects `CLASS_BATCH_BUDGET_BYTES` | CONFIGURATION | ALL GPU, multiclass only | **ASSERTED** | CORRECTLY OFF |
| `MOJOTREES_GPU_SPARSE_SKIP_FREQ` | `gpu_sparse.env_skip_freq_percent`, `var p = _env_int("MOJOTREES_GPU_SPARSE_SKIP_FREQ", DEFAULT_SKIP_FREQ_PERCENT)`, clamped to 100. Stored once per session as `self.skip_percent` | unset selects 50 percent | CONFIGURATION | ALL GPU, sparse matrices only | **ASSERTED, and the docstring says so plainly.** "Fifty is chosen for what it *proves* rather than for what it was measured to earn, because it was not measured to earn anything" | NEEDS MEASURING, at low rank; it reaches only sparse inputs |

### 3F. GPU runtime, memory route and backend identity

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_GPU_TRACE` | `gpu_runtime.PhaseCounters.from_env`, `return PhaseCounters(getenv("MOJOTREES_GPU_TRACE") == "1")` | unset behaves as off | DIAGNOSTIC | ALL GPU | n/a | DIAGNOSTIC |
| `MOJOTREES_GPU_STAGING_SLOTS` | `gpu_runtime.env_staging_slots`, `var n = _env_int("MOJOTREES_GPU_STAGING_SLOTS", DEFAULT_STAGING_SLOTS)`, clamped to `[1, MAX_STAGING_SLOTS]` | unset selects 2 | CONFIGURATION | ALL GPU, host-gradient arms | **ASSERTED** | NEEDS MEASURING, at low rank |
| `MOJOTREES_GPU_TRANSFER` | `unified_memory_policy.env_requested_route`, `var s = getenv("MOJOTREES_GPU_TRANSFER")`; empty returns `DEFAULT_ROUTE`, anything unparsable **raises** | unset selects the staged copy | CONFIGURATION | ALL GPU | **MEASURED and negative** for the alternatives. `docs/APPLE_UNIFIED_MEMORY.md` and the Aug 15 unified-memory run found `host_direct` wrong, `map_write` slower, and copy at 75 to 85 GB/s, so transfer is not the cost | CORRECTLY OFF |
| `MOJOTREES_GPU_TRANSFER_UNPROVEN` | `unified_memory_policy.env_ack_unproven`, `return getenv("MOJOTREES_GPU_TRANSFER_UNPROVEN") == "1"` | unset behaves as off | CONFIGURATION (an acknowledgment gate) | ALL GPU, and **inert alone**. It only changes an outcome when `MOJOTREES_GPU_TRANSFER` names a route whose evidence level is below `ENABLE_LEVEL` | n/a | CORRECTLY OFF, and see section 7 |
| `MOJOTREES_GPU_BACKEND` | `device_policy.env_declared_api`, `var s = getenv("MOJOTREES_GPU_BACKEND")`; empty is `API_UNKNOWN` | unset selects `API_UNKNOWN` | CONFIGURATION (a declaration, "it never supplies a capability number") | ALL GPU | n/a | CORRECTLY OFF |
| `MOJOTREES_GPU_BACKEND_UNVALIDATED` | `gpu_backend_policy.env_ack_unvalidated`, `return getenv("MOJOTREES_GPU_BACKEND_UNVALIDATED") == "1"` | unset behaves as off | CONFIGURATION (an acknowledgment gate) | ALL GPU, and **inert alone**; it only matters once `MOJOTREES_GPU_BACKEND` names an API with no validation record | n/a | CORRECTLY OFF, and see section 7 |
| `MOJOTREES_GPU_WARMUP` | `initialization.env_warmup_level`, matching `train` and `all`. Three consumers, including `bindings/basic_bindings.mojo`, `out["warmup_level"] = PythonObject(warmup_level_name(env_warmup_level()))` | unset or unrecognized selects `WARMUP_OFF` | CONFIGURATION | ALL GPU, startup only | **ASSERTED**. `docs/STARTUP_LATENCY.md` discusses it; the audit records that its documented invocation `pixi run bench-startup` is not a task | NEEDS MEASURING, at low rank; it moves startup, not steady-state training |
| `MOJOTREES_GPU_OBJECTIVE` | `train_gpu.env_objective_source`, matching `device` and `host` | unset selects AUTO, which then takes the device path wherever it is available | CONFIGURATION | ALL GPU | **ASSERTED** | CORRECTLY OFF |
| `MOJOTREES_GPU_VALID_SCORING` | `train_gpu.env_valid_scoring`, matching `device` and `host` | unset selects AUTO, and "AUTO resolves through `MOJOTREES_GPU_VALID_SCORING` and then to the host walk", so the effective default is HOST | CONFIGURATION | ALL GPU, validation scoring only | **ASSERTED.** The comment says the host walk stands "until a benchmark says otherwise" | NEEDS MEASURING, at low rank; it moves early-stopping overhead, not tree growth |
| `MOJOTREES_GPU_GRAD_LAYOUT` **(DELETED 2026-08-17)** | `gpu_gradient_stream.env_grad_layout`, `if getenv("MOJOTREES_GPU_GRAD_LAYOUT") == "interleaved": return LAYOUT_INTERLEAVED` | unset selects `LAYOUT_SPLIT` | PERFORMANCE by intent | **Nothing.** `env_grad_layout` has zero callers in `src/`, `bindings/` or `tests/`, and `LAYOUT_INTERLEAVED` is never selected outside its own module | n/a | **DEAD** |

### 3G. Device selection

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_DISABLE_GPU` | `device_policy.gpu_disabled_by_env`, `return getenv("MOJOTREES_DISABLE_GPU") == "1"` | unset behaves as off | CONFIGURATION | ALL, both backends | n/a | CORRECTLY OFF |
| `MOJOTREES_AUTO_MIN_CELLS` | `device_policy.env_auto_min_cells`, `var s = getenv("MOJOTREES_AUTO_MIN_CELLS")`; empty or unparsable returns `AUTO_MIN_CELLS` | unset selects `AUTO_MIN_CELLS` | CONFIGURATION | ALL, the `auto` device rule only | **ASSERTED** in this file. The crossover it encodes is measured elsewhere in the campaign, but no results file names this variable | CORRECTLY OFF |

### 3H. CPU parallel policy

Per the standing rule, the CPU path is the correctness oracle and is no longer
optimized, so everything in 3H and 3I ranks below any GPU row.

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_NUM_WORKERS` | `parallel.env_num_workers`, `return _env_int("MOJOTREES_NUM_WORKERS", 0)` | `0` is auto, `1` is serial, `N` forces N-way | CONFIGURATION | ALL CPU | **MEASURED** extensively; it labels arms in `cpu_round1_2026-08-16`, `thread_scaling_2026-08-16` and every `cpu_float32_lambda0` record | CORRECTLY OFF |
| `MOJOTREES_PARALLEL_MIN_OPS` | `parallel.env_parallel_min_ops`, `return _env_int("MOJOTREES_PARALLEL_MIN_OPS", PARALLEL_MIN_OPS)` | unset selects `PARALLEL_MIN_OPS` | CONFIGURATION | ALL CPU | **MEASURED** as a recorded arm in the same manifests | CORRECTLY OFF |
| `MOJOTREES_PARALLEL_MIN_TASK_OPS` | `parallel.env_parallel_min_task_ops`, `var n = _env_int("MOJOTREES_PARALLEL_MIN_TASK_OPS", DEFAULT_MIN_TASK_OPS)`, and non-positive falls back | unset selects `DEFAULT_MIN_TASK_OPS`, which equals the whole-loop crossover | CONFIGURATION | ALL CPU | **ASSERTED** | NEEDS MEASURING, at low rank |
| `MOJOTREES_CPU_TASK_FLOOR` | `parallel.env_core_floor`, `var s = getenv("MOJOTREES_CPU_TASK_FLOOR"); return s != "0"` | unset behaves as **ON** | PERFORMANCE | ALL CPU | **MEASURED**, `bench/results/cpu_window_2026-08-16/RESULTS.md`, "MOJOTREES_CPU_TASK_FLOOR: a win at 50k, nothing at 250k", twelve repeats, floor-on ratio 1.0352. The docstring's "This exists because the floor is unmeasured" is now **STALE** | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_CPU_TASKS_PER_CORE` | `apple_cpu_policy.env_tasks_per_core`, `_env_int("MOJOTREES_CPU_TASKS_PER_CORE", DEFAULT_TASKS_PER_CORE)` | `0` or unparsable selects `DEFAULT_TASKS_PER_CORE` | CONFIGURATION | ALL CPU | **ASSERTED** | NEEDS MEASURING, at low rank |
| `MOJOTREES_CPU_CORE_POOL` | `apple_cpu_policy.env_core_pool`, matching `performance`, `PERFORMANCE`, `p`; "Anything unrecognized means `all`" | unset selects `CORE_POOL_ALL` | CONFIGURATION | ALL CPU | **MEASURED** as an arm in `bench/results/cpu_phase0_2026-08-16/RESULTS.md` | CORRECTLY OFF |
| `MOJOTREES_CPU_FEATURE_GROUP` | `apple_cpu_policy.env_feature_group`, `var s = getenv("MOJOTREES_CPU_FEATURE_GROUP")`, empty returns 0, an off-ladder value **raises** | unset selects 0, meaning derive | CONFIGURATION | ALL CPU | **ASSERTED** | CORRECTLY OFF |
| `MOJOTREES_CPU_COMPACT_MIN_ROWS` | `apple_cpu_policy.env_compact_min_rows`, `return _env_int("MOJOTREES_CPU_COMPACT_MIN_ROWS", DEFAULT_COMPACT_MIN_ROWS)` | unset selects `DEFAULT_COMPACT_MIN_ROWS` | CONFIGURATION | ALL CPU | **ASSERTED** | NEEDS MEASURING, at low rank |

### 3I. CPU histogram layout, kernels and numerics

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_CPU_SERIAL_KERNEL` | `histogram.serial_kernel_arm`, `var s = getenv("MOJOTREES_CPU_SERIAL_KERNEL")`, matching `base`, `stride`, `packed`; unrecognized returns `SERIAL_KERNEL_FULL` | unset selects `full`, the shipped kernel | PERFORMANCE | **ALL CPU, and within a fit only the row-blocked feature-major builds.** Corrected 2026-08-17; this cell read "ALL CPU" unqualified. `_accumulate_blocked_at` is the sole caller, so a node below the amortization floor (under 8,160 rows at 255 bins and the shipped 8/1 ratio) runs the unblocked ladder and measures nothing, and neither row-major kernel consults it at all. The three growth policies are alike in this | **MEASURED**, `bench/results/serial_kernel_2026-08-16/`, twelve repeats at 799,110 x 100 at both worker counts. At auto, run 3 reads base 12.01, stride 11.93, packed 10.59, full 10.40, so the shipped default is the fastest arm | CORRECTLY OFF |
| `MOJOTREES_CPU_FLOAT64_GATHER` | `histogram.float64_gather_arm`, `return getenv("MOJOTREES_CPU_FLOAT64_GATHER") == "1"`; consumed only in the `else` arm of a comptime branch, `use_pairs = plan.compact_rows and float64_gather_arm()` | unset behaves as off | PERFORMANCE | ALL CPU, **but only under `derivative_precision=float64`**, and the shipped default is float32. "Under Float32 this function is not called at all ... so the shipped default pays not even the read". It cannot affect a default fit. Narrowed 2026-08-17. Within `float64` it reaches only the FEATURE-MAJOR subset builder. `_accumulate_subset_row_major` has no `else` arm on the `comptime if NARROW`, so pairing this with `MOJOTREES_CPU_BIN_LAYOUT=row` or with a non-blocking node under `MOJOTREES_CPU_LAYOUT_BY_NODE=1` measures nothing, and `_accumulate_full` never gathers on either precision | **ASSERTED** | NEEDS MEASURING, at the bottom of the list, because it cannot reach a shipped fit |
| `MOJOTREES_CPU_BIN_LAYOUT` | `apple_cpu_policy.env_bin_layout`, matching `auto`, `feature`/`col`/`0`, `row`/`1`; anything else **raises** | unset selects `auto`, and `resolve_bin_layout` degrades to feature-major when no row-major view exists | PERFORMANCE | ALL CPU. The fit-level layout is set in `GrowScratch`, which the symmetric grower shares | **MEASURED and negative for `row`.** The probe docstring records "Row-major measured 1.15x slower than feature-major at 799,110 x 100 in this lane's window and 1.35x slower in another lane's" | CORRECTLY OFF |
| `MOJOTREES_CPU_BIN_LAYOUT_PROBE` | `tree._env_bin_layout_probe`, `var s = getenv("MOJOTREES_CPU_BIN_LAYOUT_PROBE")`; empty is False, otherwise `s != "0"`. Consumed by the once-per-fit `choose_bin_layout_timed` | unset behaves as off | PERFORMANCE, in effect a measurement instrument | **ALL CPU, since 2026-08-17. This row previously read "ALL CPU, since the fit layout is shared with the symmetric grower", and the inference was wrong.** Sharing the field is not reaching the probe. `GrowScratch.resolve_layout_timed` was offered only from `grow_tree_leaves_profiled`'s per-split block, and the oblivious grower returns before that loop, so a symmetric fit left `layout_pending` set for its whole life and `choose_bin_layout_timed` never ran. The offer is now made from `_grow_oblivious_levels` too. A symmetric probe run taken before that date compared the shipped layout with itself | **MEASURED and negative**, by the same 1.15x and 1.35x above, both leaf-wise. The docstring is explicit, "the rule this probe implements is not one anybody should be running by default today" | CORRECTLY OFF |
| `MOJOTREES_CPU_LAYOUT_BY_NODE` | `tree._env_layout_by_node`, `var s = getenv("MOJOTREES_CPU_LAYOUT_BY_NODE")`; empty is False, otherwise `s != "0"`. Consumed at three `_node_bin_layout(...)` sites | unset behaves as off | PERFORMANCE | **ALL CPU, since 2026-08-17.** This row read "LEAF and DEPTH CPU, effectively dead on SYM CPU" and that was correct when the grid was built. `tree._grow_oblivious_levels` passed `scratch.bin_layout` straight to `_hist_subset` at both of its build sites; it now computes a per-child `built_layout` from `_node_bin_layout` exactly as the leaf-wise loop does. The one build that still reaches no layout argument on any policy is the unbagged root, `tree._hist_full`, because `histogram` has no whole-dataset row-major builder | **MEASURED LEAF-WISE ONLY**, in the docstring's own table dated 2026-08-16 at 799,110 x 100. Small nodes 2.716 to 2.061 ns per slot, 1.32x; tiny nodes 9.684 to 2.141, 4.52x; the three larger classes unmoved. Worth about 5 percent of the whole fit at one worker and "indistinguishable at auto". **UNMEASURED on the symmetric grower.** A symmetric M4 reading of "neutral" at 800,000 x 100 exists and must be discarded, because it was taken while the arm could not reach the code, so it measured the shipped layout against itself | NEEDS MEASURING, at low rank, and now measurable on all three policies |
| `MOJOTREES_CPU_ROW_MAJOR` | `binning.env_row_major_mode`, matching `auto`, `0`/`off`/`false`, `1`/`on`/`true`; anything else **raises** | **unset selects auto and auto is not off.** The view is built when it fits `row_major_budget_bytes()` | CONFIGURATION | ALL CPU | **ASSERTED** for the budget; the layout it feeds is measured negative above | CORRECTLY OFF |
| `MOJOTREES_CPU_ROW_MAJOR_MAX_MB` | `binning.row_major_budget_bytes`, `_env_int("MOJOTREES_CPU_ROW_MAJOR_MAX_MB", ROW_MAJOR_DEFAULT_BUDGET_MB) * 1024 * 1024` | unset selects `ROW_MAJOR_DEFAULT_BUDGET_MB`; `0` lifts the budget | CONFIGURATION | ALL CPU | **ASSERTED** | CORRECTLY OFF |
| `MOJOTREES_CPU_ROW_BLOCKS` | `apple_cpu_policy.env_row_blocks`, `return _env_int("MOJOTREES_CPU_ROW_BLOCKS", 0)` | `0` and unset both mean derive | **BEHAVIOR.** "This knob moves bits ... The block count is a summation order, so `1` and `4` produce two different Float64 histograms" | ALL CPU | **ASSERTED** | CORRECTLY OFF, it is a bisection handle |
| `MOJOTREES_CPU_ROW_BLOCK_AMORTIZE` | `apple_cpu_policy.env_row_block_amortize`, `return parse_row_block_amortize(getenv("MOJOTREES_CPU_ROW_BLOCK_AMORTIZE"))`; `checked_row_block_amortize` **raises** on a refused value | unset selects the shipped 8/1 | **BEHAVIOR**, by the same summation-order argument | ALL CPU | **ASSERTED** | CORRECTLY OFF |
| `MOJOTREES_DERIVATIVE_PRECISION` | two readers in `histogram.mojo`, the resolver `getenv("MOJOTREES_DERIVATIVE_PRECISION") != DERIVATIVE_PRECISION_FLOAT64` and the validator `check_derivative_precision`, which **raises** on a typo | unset selects float32 | **BEHAVIOR** | ALL CPU, and the GPU staging path narrows to Float32 regardless | **MEASURED**, `bench/results/cpu_float32_lambda0_2026-08-16/` holds paired f32 and f64 runs, and `docs/design/ACCURACY_BUDGET.md` prices it. The brief records float32 as measured neutral or better on the CPU symmetric path | CORRECTLY OFF |
| `MOJOTREES_CPU_QUANT_GRAD` | `quantized_gradient.cpu_quant_grad_allowed`, `return _env_int("MOJOTREES_CPU_QUANT_GRAD", 1) != 0`; one consumer, `if not cpu_quant_grad_allowed():` | unset behaves as **ON**, meaning permitted. "It still does nothing until a caller enables `use_quantized_grad`, which is off by default" | CONFIGURATION (a permission) | ALL CPU | **ASSERTED** | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_CPU_QUANT_SCALE` | `quantized_gradient.env_cpu_quant_scale_rule`, `if _env_int("MOJOTREES_CPU_QUANT_SCALE", 1) == 0: return SCALE_MAX_ABS`, else `SCALE_MAGNITUDE_SUM` | unset selects `SCALE_MAGNITUDE_SUM`, which matches the GPU lattice | **BEHAVIOR.** At the default "`num_grad_quant_bins` **does not affect the lattice**" | ALL CPU, quantized path only | **ASSERTED** | CORRECTLY OFF, it is a deliberate CPU/GPU-agreement tradeoff |
| `MOJOTREES_LEAF_SCORE_UPDATE` | `boosting._leaf_score_update_enabled`, `return getenv("MOJOTREES_LEAF_SCORE_UPDATE") != "0"`; four consumers, `var by_leaf = _leaf_score_update_enabled()` | unset behaves as **ON** | PERFORMANCE | ALL CPU, and the GPU host-gradient round arms | **ASSERTED**, and stated as not a tuning knob, "there is no workload on which the traversal is the better route, and none has been measured either way" | SHOULD BE THE DEFAULT (already is) |
| `MOJOTREES_BINNING_SELECT_MIN_ROWS` | `binning.env_select_min_rows`, `var n = _env_int("MOJOTREES_BINNING_SELECT_MIN_ROWS", SELECT_MIN_ROWS)`, non-positive falls back | unset selects `SELECT_MIN_ROWS` | CONFIGURATION | ALL, binning runs before growth | **ASSERTED**. "the two paths resolve the same order statistics, so this decides which one runs and nothing else" | NEEDS MEASURING, at low rank |
| `MOJOTREES_BINNING_SORTED_TIE_REPAIR` **(ADDED TO THE TREE AFTER THIS GRID WAS BUILT, 2026-08-18)** | `binning.env_sorted_tie_repair`, `return getenv("MOJOTREES_BINNING_SORTED_TIE_REPAIR") == "1"` | unset behaves as off, "so the fit is instruction-for-instruction the fit it was" | PERFORMANCE | ALL, both backends. Binning runs above the policy fork, and it reaches only the tie-repair pass, which fires on the 59 of 90 columns of `year_prediction_msd` that have a tied boundary. A column the ascending probe rejects falls back to `resolve_above_unsorted`, so the switch measures nothing on data that does not sort | **UNMEASURED.** The docstring carries arithmetic and a bit-identity proof, not a run. 94.4 million mispredicting compares to repair 170 numbers, against 254 * 18 = 4,572 plus a 200,000-compare ascending probe, "about 350x less work in the repair". Bit-identity is proved rather than hoped, including the sign of zero, because on an array the probe has accepted the equal minima are contiguous | **NEEDS MEASURING.** The switch names the run that deletes it, one interleaved A/B of the binning phase on the 463,715 x 90 real cell, edges compared byte for byte. A win with identical edges makes it the only behavior and the variable goes; a loss, or one moved edge, deletes the code instead |

### 3J. Diagnostics with no policy specificity

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_PHASE_PROFILE` | `phase_profile.env_profile_mode`, `var raw = getenv("MOJOTREES_PHASE_PROFILE")`; `""`, `0`, `off` are off, `1`/`async` and `fenced` are the two modes, **anything else raises** | unset selects off | DIAGNOSTIC | ALL, both backends. On the device-resident and oblivious planes it counts launches without timing phases, by design | **MEASURED** as an instrument, recorded in `thread_scaling_2026-08-16`, `cpu_round1_2026-08-16` and every `cpu_float32_lambda0` manifest | DIAGNOSTIC |
| `MOJOTREES_STARTUP_TRACE` | `initialization.StartupTrace.from_env`, `return StartupTrace(getenv("MOJOTREES_STARTUP_TRACE") == "1")` | unset behaves as off | DIAGNOSTIC | ALL | **ASSERTED as an instrument.** `INSTRUCTION_AUDIT.md` section 9b, "The variable works; every documented way to exercise it does not" | DIAGNOSTIC |
| `MOJOTREES_OBLIVIOUS_TRACE` | `growth_policy.ObliviousTrace.resolve`, `var s = getenv("MOJOTREES_OBLIVIOUS_TRACE"); return ObliviousTrace(s == "1" or s == "true" or s == "TRUE")`. One consumer, `var trace = ObliviousTrace.resolve()` in `tree._grow_oblivious_levels` | unset behaves as off | DIAGNOSTIC | **SYM CPU only.** The GPU symmetric plane traces through `MOJOTREES_GPU_TREE_RESIDENT_TRACE` instead | n/a | DIAGNOSTIC |

### 3K. Distributed

**DELETED 2026-08-17, all seven.** The table below is the finding, preserved.
`runtime_from_env` and its `_env_int` helper were removed from
`distributed_transport.mojo` and no code in the repository reads any
`MOJOTREES_DIST_*` name now. Reason and tombstone in
[DEAD_SWITCH_RESOLUTION.md](DEAD_SWITCH_RESOLUTION.md).

All seven were read in exactly one function, `distributed_transport.runtime_from_env`.

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_DIST_MODE` | `var mode_text = getenv("MOJOTREES_DIST_MODE")` | unset selects `RUNTIME_LOCAL` | CONFIGURATION | nothing | n/a | **DEAD** |
| `MOJOTREES_DIST_WORLD_SIZE` | `local_runtime(_env_int("MOJOTREES_DIST_WORLD_SIZE", 1), job_id)` | unset selects 1 | CONFIGURATION | nothing | n/a | **DEAD** |
| `MOJOTREES_DIST_RANK` | `_env_int("MOJOTREES_DIST_RANK", -1)` | unset selects -1, which `spec.validate()` then refuses | CONFIGURATION | nothing | n/a | **DEAD** |
| `MOJOTREES_DIST_MACHINES` | `var machines = getenv("MOJOTREES_DIST_MACHINES")`; empty raises in transport mode | unset is empty | CONFIGURATION | nothing | n/a | **DEAD** |
| `MOJOTREES_DIST_JOB_ID` | `_env_int("MOJOTREES_DIST_JOB_ID", 0)`; negative raises | unset selects 0 | CONFIGURATION | nothing | n/a | **DEAD** |
| `MOJOTREES_DIST_TIMEOUT_S` | `_env_int("MOJOTREES_DIST_TIMEOUT_S", 300)` | unset selects 300 | CONFIGURATION | nothing | n/a | **DEAD** |
| `MOJOTREES_DIST_RESTART_EPOCH` | `_env_int("MOJOTREES_DIST_RESTART_EPOCH", 0)` | unset selects 0 | CONFIGURATION | nothing | n/a | **DEAD** |

`runtime_from_env` has **no caller anywhere in the repository**. A
repository-wide grep for the name returns one hit, its own definition. The
Python distributed runtime uses a different, non-overlapping set of names
(`MOJOTREES_DISTRIBUTED_BASE_PORT`, `MOJOTREES_DISTRIBUTED_CONNECT_TIMEOUT`,
`MOJOTREES_DISTRIBUTED_PROVIDER`), so nothing bridges to these seven.

### 3L. Prediction

**A section this grid did not have.** `predict.mojo` reads three variables and
none of them had a row anywhere above, which is a boundary error rather than a
judgement. The grid's axes are growth policy and backend, and a batch walker
has neither, so the walkers fell out of every section. Only the row this lane
verified from source is filled in. `MOJOTREES_RAW_PREDICT` and
`MOJOTREES_PREDICT_TRACE` are named in section 8 as still owed.

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_PREDICT_TILE` **(ADDED TO THE TREE AFTER THIS GRID WAS BUILT, 2026-08-18)** | `predict.predict_tile_enabled`, `return getenv("MOJOTREES_PREDICT_TILE") != "0"`. Consumed in `predict.predict_raw_batch`, which holds the tiled closure and the `apply_row_major` one it replaced | **unset behaves as ON.** The `!= "0"` spelling this repository reserves for an arm that was argued, measured and shipped, and the variable is the escape hatch back to the row-outer nest | PERFORMANCE | **Scoring, not growth**, so all three policies and both backends reach it through the same walker. It is a loop interchange over the same trees, so it has no policy specificity at all. Not the GPU walker, which is `gpu_predict.GpuPredictor` and tiles by its own grid | **MEASURED, 2026-08-18**, and the docstring's own registered run is the one that was taken, one interleaved process on the real held-out split, 51,630 rows x 90 features, 100 trees, leaf-wise and depth-wise arms. **1.10x leaf-wise and 1.54x depth-wise, bit-identical** (`bench/results/RESUME_2026-08-18.md`). A loop interchange is a claim about the cache and nothing else, which is why an interleaved process rather than a before-build against an after-build | **SHOULD BE THE DEFAULT, AND IS SINCE 2026-08-18.** Flipped in the session that measured it, under LANE_RULES rule 5. The docstring at `predict_tile_enabled` still describes the measurement as owed rather than taken, which is **stale**; on the result it records, that function, the variable and the `apply_row_major` closure are all due for deletion |

---

## 4. Ranked candidates

Every switch whose verdict is SHOULD BE THE DEFAULT or NEEDS MEASURING,
highest expected value first, with the single measurement that would settle
it. The fourteen "already is" rows are excluded, because there is nothing to
flip.

**Ranks 1 through 4 are CLOSED as of 2026-08-17.** All four were measured that
day and their defaults flipped in the same session under LANE_RULES rule 5, so
the "one measurement that settles it" column on those four rows records a
measurement that has been taken rather than one that is owed. They are kept in
the list, annotated, because the ranking is the record of how they were judged
and a silently shortened list would not show that the top of it resolved. Rank 5
is closed too, on a null rather than on a win.

Counts over the 85. By kind, CONFIGURATION 43, PERFORMANCE 25, DIAGNOSTIC 12,
BEHAVIOR 5. By verdict, as the grid was built: CORRECTLY OFF 33, NEEDS
MEASURING 14, SHOULD BE THE DEFAULT (already is) 14, DIAGNOSTIC 13, DEAD 8,
SHOULD BE THE DEFAULT 3.

**The verdict counts moved again on 2026-08-18.** `MOJOTREES_PREDICT_TILE`
arrived already measured and already flipped, so it is a nineteenth default-ON
switch and is NOT ranked below, for the same reason the other "already is" rows
are not, which is that there is nothing to flip. `MOJOTREES_BINNING_SORTED_TIE_REPAIR`
arrived unmeasured and is ranked 18. `MOJOTREES_GPU_ROW_COMPACTION` stays
CORRECTLY OFF and unranked, but on a measurement now rather than on a
prediction.

**The verdict counts moved later on 2026-08-17 and the line above is the
as-audited tally, kept for that reason.** All three SHOULD BE THE DEFAULT rows
flipped (`OBLIVIOUS_SUBTRACT`, `OBLIVIOUS_WIDE`, `OBLIVIOUS_SKIP_LAST_BUILD`),
and one NEEDS MEASURING row was measured and flipped with them
(`SPLIT_WIDE`), so default-ON is 18 rather than 14, SHOULD BE THE DEFAULT is 0,
and NEEDS MEASURING is 12 once `OBLIVIOUS_NOISE_HOIST` is also removed from it
on a measured null. The by-kind counts are unaffected; nothing changed kind.

| rank | switch | why here | the one measurement that settles it |
|---|---|---|---|
| 1 | `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT` **CLOSED, DEFAULT ON 2026-08-17** | 1.78x measured on the shipped symmetric default, bit-identical by an exact integer argument. Largest single number on the board | Settled. The interleaved on/off pair was run, three round-robin cycles, and the combined arm with ranks 2 and 4 came in at 2.20x with rmse identical to nine decimals in every cycle. What is left is a **regression gate** at a second shape, ideally 250k x 50. This cell used to end "before the default flips"; it has flipped |
| 2 | `MOJOTREES_GPU_OBLIVIOUS_WIDE` **CLOSED, DEFAULT ON 2026-08-17** | 4.5 percent, resolved, bit-identical, on the same plane. It composes with rank 1 on a different axis, scan width against accumulation width | Settled. The pair was run with the subtracting arm on as well, 22.76 s to 21.28 s in the loaded round robin and 20.40 s to 19.49 s with disjoint ranges on the quiet box, so the win survives the halved accumulation. The smallness is the finding: the scan was never this plane's bottleneck |
| 3 | `MOJOTREES_GPU_SPLIT_WIDE` **CLOSED, DEFAULT ON 2026-08-17** | The same widening of the same `block_dim=1` scan on the leaf-wise plane. This cell read "still off, while its oblivious sibling measured 4.5 percent"; it was measured that evening and is on | Settled by exactly the run this cell asked for. Interleaved, non-categorical, 799,110 x 100 leaf-wise, three cycles: 3.922 / 3.874 / 3.932 s narrow against 3.220 / 3.196 / 3.231 s wide, disjoint, rmse 6.116601511 throughout. 1.21x, four times the oblivious arm rather than the small win the docstring predicted |
| 4 | `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` **CLOSED, DEFAULT ON 2026-08-17** | This cell read "about 1.20x, magnitude a lower bound". The repeat was taken and the figure is **1.26x**, 22.76 s to 18.06 s. Independent axis from rank 1, so the two compose to 31 row builds per depth-6 tree against 126 | Settled. The launch side moves 56 to 55 at depth 6 and is the least interesting part of it; the accumulation pass and the largest zeroing pass in the tree are what the 1.26x is |
| 5 | `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST` **CLOSED ON A NULL, STILL OFF** | Six full-queue drains per depth-6 tree, each sitting in the worst position a drain can occupy, and the CatBoost-mode default set puts them there. This cell read "one run said slower and was confounded"; the standing reading is that it was measured and came in **indistinguishable** | Taken, on a fit that does set `random_strength > 0`. An M0-indistinguishable result is a null and not a pending item, so this is the one arm of the five that did not flip and did not need to. Re-measuring is optional. A fit at `random_strength = 0` still measures nothing, which is how this became invisible in the first place |
| 6 | `MOJOTREES_GPU_SPECULATION` | 66.8 percent census hit rate at 1M rows against 964 wasted builds per fit. The condition to judge it is already registered in `session3_2026-08-16/RESULTS.md`, "the launch-shape gain has to beat the wasted work in a whole fit, not in a phase" | Two interleaved end-to-end pairs, at **50,000 and at 1,000,000 rows**, leaf-wise, as the registered protocol demands. Do not run it against a symmetric fit; it refuses the plane |
| 7 | `MOJOTREES_GPU_HIST_SPECIALIZATION=batched` | Preregistered in `PHASE2_PREREGISTRATION.md`, never reported. It is the only rung of that ladder the histogram file points at | One interleaved pair, `baseline` against `batched`, at the shape the preregistration names |
| 8 | `MOJOTREES_GPU_STAGING_SLOTS` | Default 2, unmeasured, and it is the host-gradient upload ring, which is on the critical path of every host-gradient arm | A sweep of 1, 2, 4 in one process on a host-gradient fit |
| 9 | `MOJOTREES_GPU_VALID_SCORING=device` | The host walk stands "until a benchmark says otherwise" and validation is per round | One paired fit with early stopping enabled, host against device |
| 10 | `MOJOTREES_GPU_SPARSE_SKIP_FREQ` | The default of 50 is admitted to be argued rather than measured | A crossover sweep on a sparse matrix, 0 / 25 / 50 / 75. Low rank because it reaches only sparse inputs |
| 11 | `MOJOTREES_CPU_LAYOUT_BY_NODE` | Measured 1.32x and 4.52x at the two smallest node classes, about 5 percent of a fit at one worker, indistinguishable at auto. All three figures are leaf-wise | An auto-mode interleaved pair at a shape with many small nodes. **Corrected 2026-08-17.** This cell used to warn that the switch "cannot reach the symmetric CPU grower at all". It could not, and now it can, so a symmetric arm is worth running and no earlier symmetric reading of it counts |
| 12 | `MOJOTREES_CPU_TASKS_PER_CORE` | Unmeasured fan-out multiplier, and `cpu_window` showed the neighbouring floor is worth a percent at 50k | A sweep of 1, 2, 4 at 50k, where the floor result says fan-out matters |
| 13 | `MOJOTREES_PARALLEL_MIN_TASK_OPS` | Unmeasured per-task floor sitting beside a measured whole-loop crossover | An A/B at the default against half and double, at 50k |
| 14 | `MOJOTREES_CPU_COMPACT_MIN_ROWS` | Unmeasured threshold on the CPU compaction path | A sweep at the crossover the constant names |
| 15 | `MOJOTREES_BINNING_SELECT_MIN_ROWS` | Unmeasured; it selects between two order-statistic paths that resolve the same answer | A binning-only timing at both paths on one matrix |
| 16 | `MOJOTREES_GPU_WARMUP` | Unmeasured, and the documented way to exercise it is a pixi task that does not exist | A startup-latency capture at off, train and all. Note the harness gap first |
| 17 | `MOJOTREES_CPU_FLOAT64_GATHER` | A real 12-passes-to-1 argument, and it **cannot affect a default fit**, because it is only reached under `derivative_precision=float64` and the shipped default is float32 | A float64 CPU fit, gather on against off. Last, because winning changes nothing that ships |
| 18 | `MOJOTREES_BINNING_SORTED_TIE_REPAIR` **(ADDED 2026-08-18)** | Appended rather than inserted, so that ranks 1 to 17 keep the numbers other sections cite; by expected value it belongs beside rank 15, which is the other binning-phase item. 94.4 million mispredicting compares to repair 170 numbers on the real 463,715 x 90 cell, against 4,572 plus a 200,000-compare probe. The reason it is not higher is share, not size, because binning runs once per fit | The run the switch names, one interleaved A/B of the binning phase on that cell, with the fitted edges compared byte for byte. It deletes itself either way, the variable on a win with identical edges and the code on a loss or a moved edge |

---

## 5. Dead switches

Nine names, in three kinds of dead.

**All nine were resolved on 2026-08-17, after this grid was written.** Eight
reads were deleted, one docstring promise was deleted, none was wired up, and
each site carries a tombstone naming what would have to exist for the switch
to return. The per-switch verdicts and evidence are in
[DEAD_SWITCH_RESOLUTION.md](DEAD_SWITCH_RESOLUTION.md). What follows is the
finding as it stood, kept because the deletions are only defensible against
it.

**Read by nothing (1).** `MOJOTREES_STARTUP_REPORT_FD`. Named once, in
`initialization.mojo` prose, as "reserved, unread here". No `getenv` in the
tree takes it. It is nevertheless carried in `compatibility/api_snapshot.json`
and `compatibility/DRIFT_REPORT.md`, and listed in
`python/mojotrees/diagnostics.py` inside a tuple whose comment reads "Listed,
never interpreted". `INSTRUCTION_AUDIT.md` already flagged it and it is still
here. It is not in this grid's 85 because it is not read.

**Read by a function nothing calls (8).** The seven `MOJOTREES_DIST_*` names
of section 3K, all reached only through `distributed_transport.runtime_from_env`,
which has no caller in the repository; and `MOJOTREES_GPU_GRAD_LAYOUT`,
reached only through `gpu_gradient_stream.env_grad_layout`, which has no
caller in `src/`, `bindings/` or `tests/`, and whose `LAYOUT_INTERLEAVED`
constant is never selected outside its own module.

**Shadowed (1, counted above).** `gpu_tree_tables.tree_resident_requested`
reads `MOJOTREES_GPU_TREE_RESIDENT` as `== "1"` while the live gate,
`gpu_resident_round.resident_round_enabled`, reads it as `!= "0"`. The two
disagree about the default. The name itself is live through the gate, so the
name is not dead; the second reader is. Its own docstring is unambiguous
about the hazard, "Two predicates over one variable now disagree about that
variable's default, and the next caller to reach for the one in
`gpu_tree_tables` gets the pre-flip answer with no warning."

**Reachable but refused on every default path (1, not counted as dead).**
`MOJOTREES_GPU_VERIFY_ROWS=1` raises on both the device-resident plane and
the oblivious plane, so on the shipped GPU defaults it cannot verify anything.
It is a genuine switch on the incremental loop.

---

## 6. Growth-policy asymmetries worth naming

These are the column-5 findings that a name alone would not surface.

1. **The device-resident growth plane is leaf-wise only.**
   `gpu_tree_tables.tree_resident_supported` contains
   `if params.grow_policy != GROW_LEAFWISE: return TREE_RESIDENT_DEPTHWISE`.
   So `MOJOTREES_GPU_TREE_RESIDENT`, `MOJOTREES_GPU_SPECULATION`,
   `MOJOTREES_GPU_SPECULATION_CENSUS` and `MOJOTREES_GPU_FUSE_PARTITION_TAIL`
   never reach a depth-wise GPU fit, which falls through to
   `_device_search_incremental`.

2. **`MOJOTREES_GPU_FUSE_PARTITION_TAIL` is inert on the symmetric plane.**
   The oblivious loop sets fusion unconditionally, "Not optional here, where
   it is merely the default on the leaf-wise plane", because the level's
   batched build is the only thing that can pay the deferred copy-back.

3. **`MOJOTREES_CPU_LAYOUT_BY_NODE` was effectively dead on the symmetric CPU
   grower. FIXED 2026-08-17, and the finding is kept because a number rests
   on it.** As audited, `tree._grow_oblivious_levels` passed
   `scratch.bin_layout` directly to both of its `_hist_subset` calls, so the
   only `_node_bin_layout` call a symmetric fit reached was the bagged root,
   where the node is the whole sample and the small-node rule cannot fire.
   The grower now derives a per-child `built_layout` at each of its two build
   sites, from the same `_node_bin_layout` predicate and the same hoisted
   active-feature count the leaf-wise loop uses.

   **The consequence for the record, which is the reason this item is
   long.** A symmetric CPU reading of this switch was taken on an Apple M4 at
   800,000 x 100 and recorded as "neutral", and that neutral was used to
   argue that per-node layout is not where the symmetric CPU cost lives. It
   supports no such conclusion. The arm never differed from its baseline, so
   the only thing the run established is that the two identical programs ran
   at the same speed. Per-node layout on the symmetric grower is UNMEASURED.

   The same defect covered `MOJOTREES_CPU_BIN_LAYOUT_PROBE`, which was also
   offered only from the leaf-wise loop and is now offered from both.
   `MOJOTREES_CPU_BIN_LAYOUT` itself was never affected, because it sets
   `GrowScratch.bin_layout`, which the symmetric grower did read and pass.

4. **`MOJOTREES_CPU_FLOAT64_GATHER` cannot affect a default fit.** Its read
   sits in the `else` arm of a comptime `NARROW` branch, and the shipped
   `derivative_precision` is float32, so the default path does not execute
   the `getenv` at all.

5. **Four symmetric-GPU switches have no leaf-wise twin.**
   `OBLIVIOUS_SUBTRACT`, `OBLIVIOUS_WIDE`, `OBLIVIOUS_SKIP_LAST_BUILD` and
   `OBLIVIOUS_NOISE_HOIST` are SYM only. `OBLIVIOUS_SUBTRACT`'s placement in
   `histogram_gpu` rather than in the batcher is deliberate, so that a
   two-item leaf-wise plan cannot reach the subtracting arm.

   **Corrected 2026-08-17 on two counts.** This item also said "one leaf-wise
   switch has no symmetric twin, `SPLIT_WIDE` is LEAF only". `SPLIT_WIDE`
   reaches DEPTH as well, through the shared searcher field (section 8 item 1),
   and it does have a symmetric twin, which is `OBLIVIOUS_WIDE` in the same
   file. And `OBLIVIOUS_SKIP_LAST_BUILD` now has a general twin,
   `MOJOTREES_GPU_SKIP_TERMINAL_CHILDREN`, built the same day for
   `_device_search_resident`.

6. **`MOJOTREES_OBLIVIOUS_TRACE` is the CPU symmetric grower's trace and
   nothing else.** The GPU symmetric plane traces through
   `MOJOTREES_GPU_TREE_RESIDENT_TRACE`.

7. **`feature_fraction_bynode` is a PARAMETER, not a switch, and it draws per
   LEVEL under `grow_policy=oblivious`.** Added 2026-08-17 by a sweep for
   accepted-then-not-honored inputs, which is the class the two GPU bugs of
   that day belonged to. `_grow_oblivious_levels` calls
   `select_split_features` with the level's depth and the level's lowest node
   id, so the whole level shares one draw. This is a redefinition and not a
   defect, and there is no alternative, because the leaves of a level must
   agree on one split and cannot agree on a candidate some were never
   offered. It is
   listed because `tree._check_oblivious` documents itself as refusing rather
   than half-applying every parameter whose meaning does not survive the
   mode, and this one is half-applied by necessity. At depth 6 the same
   fraction takes 63 draws per tree leaf-wise and 6 symmetric, so the two are
   not comparable across policies. Now stated in `_check_oblivious`'s own
   docstring as well.

---

## 7. Switches that interact

Pairs and groups where turning both on does something neither does alone.

**A. The four symmetric-GPU arms, which is the composition that had to be
worked out by hand today.** `docs/design/OBLIVIOUS_WAIT_CENSUS.md` has the
table and it is reproduced here because it is the interaction, not a
footnote. Row builds per tree at depth 6:

    SUBTRACT  SKIP_LAST   row builds        depth 6
    off       off         2^(d+1) - 2       126
    off       on          2^d - 2            62
    on        off         2^d - 1            63
    on        on          2^(d-1) - 1        31

`SKIP_LAST` truncates whichever series is running one level early; `SUBTRACT`
halves the width of every level that runs. **They compose exactly, and as of
2026-08-17 they have been measured on together**, which is the correction this
paragraph needed: it read "have never been measured on together", and the
combined arm of `SUBTRACT`, `SKIP_LAST` and `OBLIVIOUS_WIDE` came in at 22.76 s
to 10.36 s, 2.20x, with rmse 2.439382420 in every arm of every cycle. Three of
the four are the shipped default now, so the bottom row of that table, 31 row
builds at depth 6, is what a symmetric fit does today. `OBLIVIOUS_WIDE` acts on
a third axis, the scan's threads per (leaf, feature), and `NOISE_HOIST` on a
fourth, the number of drains per tree. None of the four adds a launch, so
`oblivious_launch_census(6)` is 62 in all sixteen combinations except that
`SKIP_LAST` removes the last level's build. On the schedule the code actually
enqueues, `oblivious_schedule_launches(6)` is 56 with everything off and **55
under the shipped default**, since `SKIP_LAST` drops the last level's two batch
launches and pays one back for the partition's own copy-back.

**B. `MOJOTREES_GPU_SPECULATION` plus `grow_policy=oblivious` is a refusal,
not a combination.** `_oblivious_route_reason` contains
`if speculative_build_enabled(): return OBLIVIOUS_SPECULATION`, and
`train_gpu` turns any non-OK reason into a raise, because "there is nothing on
this backend to fall back *to*". So a benchmark that exports
`MOJOTREES_GPU_SPECULATION=1` for a leaf-wise arm and then runs a symmetric
arm in the same shell **fails the symmetric arm**. The stated reason is
correctness, not policy; a speculative build "publishes a *live* leaf" whose
histogram a batched level plan would overwrite.

**C. `MOJOTREES_GPU_SPLIT_RESIDENT=0` breaks symmetric GPU fits.** It gates
`opened` on the oblivious route,
`not resident_frontier_disabled() and budget >= 2 and builder.open_resident(...)`,
and a not-OK route raises. On the leaf-wise plane the same `0` is a benign
fallback to the incremental loop. Same variable, two very different
consequences by policy.

**D. `MOJOTREES_GPU_VERIFY_ROWS` plus `MOJOTREES_GPU_TREE_RESIDENT`.**
`VERIFY_ROWS=1` raises on the resident and oblivious planes and the error text
names the other switch, "Set MOJOTREES_GPU_TREE_RESIDENT=0 to take the
incremental loop, which performs it". The pair is the only way to reach the
check.

**E. `MOJOTREES_GPU_ROW_COMPACTION` plus `MOJOTREES_GPU_COMPACT_FLAG_READ`.**
The flag-read arm is stated to be "inert on its own: it changes nothing at all
unless the compaction arm above is also on". `MOJOTREES_GPU_COMPACTION_TRACE`
is the third member, and exists precisely so that "a requested-but-never-engaged
arm is distinguishable from a working one". **Since 2026-08-18 the pair cannot
be set at all under `grow_policy=oblivious`**, because the compaction arm is
refused there and the flag-read arm is inert without it, so the whole
interaction is leaf-wise and depth-wise only.

**F. `MOJOTREES_GPU_TRANSFER` plus `MOJOTREES_GPU_TRANSFER_UNPROVEN`.** The
acknowledgment does nothing alone. It only changes an outcome when the route
request would otherwise be refused with `BLOCK_NO_EVIDENCE`, and the error
text says so, "set MOJOTREES_GPU_TRANSFER_UNPROVEN=1 to run it anyway and
report that flag with any number it produces".

**G. `MOJOTREES_GPU_BACKEND` plus `MOJOTREES_GPU_BACKEND_UNVALIDATED`.** Same
shape. The acknowledgment is inert until the declaration names an API with no
validation record.

**H. `MOJOTREES_GPU_HIST_SPECIALIZATION=batched` plus
`MOJOTREES_GPU_BATCH_SLOTS`.** The slot depth is "Only read when batching was
requested at all, so the default path never consults it".

**I. `MOJOTREES_GPU_MIN_TILES` plus `MOJOTREES_GPU_ROW_TILE`, and both against
their in-process arguments.** `resolve_tiling` takes `min_tiles_request` and
`rows_per_tile_request` arguments that outrank both variables, "so that a
benchmark holding tile geometries as arms is not silently overridden by a
variable some earlier session exported".

**J. `MOJOTREES_CPU_BIN_LAYOUT`, `MOJOTREES_CPU_BIN_LAYOUT_PROBE`,
`MOJOTREES_CPU_LAYOUT_BY_NODE` and `MOJOTREES_CPU_ROW_MAJOR`.** Four switches
over one decision. `CPU_ROW_MAJOR=0` builds no row-major view, and
`resolve_bin_layout` then degrades every request to feature-major, so it
silently disables the other three. The probe decides the fit layout; the
by-node rule then overrides it per node.

A fifth interaction, added 2026-08-17. **None of the three layout switches
reaches the root histogram of a tree grown without bagging.** That build is
`tree._hist_full`, which takes no `layout` argument because `histogram`
exposes no whole-dataset row-major builder; the only by-layout entry is
`build_histogram_subset_by_layout_into_scratch`. It is one build per tree
against thousands, and it is the node where the two layouts differ least,
since the root walks the identity row list. It is recorded so that a layout
A/B is not read as covering a build it cannot touch.

**K. `MOJOTREES_CPU_ROW_BLOCKS` and `MOJOTREES_CPU_ROW_BLOCK_AMORTIZE`.** Two
spellings of one knob. `ROW_BLOCKS` names a count directly and bypasses the
amortization floor and the byte budget; `AMORTIZE` names the rule that derives
one. An explicit count therefore makes the ratio irrelevant.

**L. `MOJOTREES_CONST_HESSIAN` and `MOJOTREES_CONST_HESSIAN_VERIFY`.** The
verify pass only runs on a builder that was told the hessians are constant,
which `CONST_HESSIAN=0` prevents. Separately, `CONST_HESSIAN_VERIFY=1` steers
`device='auto'` away from the GPU, because no GPU builder can perform the host
walk.

**M. `MOJOTREES_GPU_SPLIT_GAIN_FORM` and `lambda_l1`.**
`gpu_resolve_gain_form` forces the subtractive form whenever `lambda_l1 != 0`
regardless of the request, because the cross identity is invalid under soft
thresholding. So the variable has no effect on an L1 fit.

**N. `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST` and `random_strength`.** The hoist
collapses six drains to one, and at `random_strength = 0` there are zero
drains to collapse. It is also conditional on the searcher holding
`max_depth` records above the leaf budget, which `train_gpu._search_record_slots`
only asks for under the same switch, "A searcher that does not falls back to
the per-level path rather than indexing past its tables".

---

## 8. Cells this lane could not determine

Stated plainly, with what would have to be read to close each.

1. **`MOJOTREES_GPU_SPLIT_WIDE` on the depth-wise plane. CLOSED 2026-08-17:
   it reaches.** This item marked the row LEAF on the reasoning that depth-wise
   cannot take the resident plane, and asked whether the same
   `GpuSplitSearcher` is constructed and therefore honors `self.wide_scan`. It
   is. There is exactly one `GpuSplitSearcher(` construction in the tree, in
   `train_gpu`, `self.wide_scan = wide_scan_for(any_cat)` is set there once, and
   both search entry points pass it down to `_launch`, `enqueue` for a single
   record and `enqueue_frontier` for a batch. The row is now LEAF and DEPTH.
   Rank 3's measurement was taken leaf-wise, which is the default policy.

2. **Whether any of the four symmetric arms was filed to `bench/results/`.** A
   grep for `MOJOTREES_GPU_OBLIVIOUS_WIDE` and
   `MOJOTREES_GPU_NOISE_STAGE_PARALLEL` across `bench/results/` and `docs/`
   returns nothing, so the 1.78x and the 4.5 percent are cited to the lane
   brief and not to an artifact. Whoever holds those runs should file them;
   this grid will otherwise read as ASSERTED to the next reader.

3. **`MOJOTREES_GPU_SPARSE_SKIP_FREQ` kind.** Recorded as CONFIGURATION. It
   changes which bins an accumulation visits, and whether the skipped bin's
   counts are recovered by subtraction, and therefore whether the histogram is
   identical, was not traced. Reading `gpu_sparse._resolve_skip_bins` and the
   accumulation kernel that consumes `skip_bins` would decide between
   CONFIGURATION and BEHAVIOR.

4. **`MOJOTREES_GPU_OBJECTIVE` and `MOJOTREES_GPU_VALID_SCORING` policy
   reach.** Both are marked ALL GPU on the grounds that they are resolved
   above the grower. Whether the device-gradient arm or device validation
   scoring is refused under `grow_policy=oblivious` was not checked; reading
   `_train_gpu_rounds`'s two arms would settle it.

5. **Whether `MOJOTREES_GPU_TABLE_RESET=0` and `MOJOTREES_GPU_PACKED_DOWNLOAD=0`
   are reachable on the leaf-wise plane as well as the symmetric one.** Both
   are read in the shared `DeviceTreeTables` constructor, which both planes
   open, so the answer is almost certainly yes; it was inferred from the
   constructor's placement rather than traced to a leaf-wise call site.

6. **SIX LIVE SWITCHES HAVE NO ROW IN THIS GRID, found 2026-08-18 by the
   header's own claim to cover "every `MOJOTREES_*` environment switch under
   `src/` and `bindings/`".** They are listed and NOT given rows, because this
   lane verified only that each has a reader and did not trace its default,
   kind, reach or evidence, and a half-filled row is worse than a named gap.

   - `MOJOTREES_GPU_MVS_DEVICE`, `train_gpu.mojo`,
     `getenv("MOJOTREES_GPU_MVS_DEVICE") == "1"`. The one that matters most.
     The device MVS solve is measured at 2.23x oblivious, 3.23x leaf-wise and
     2.97x depth-wise (`bench/results/RESUME_2026-08-18.md`), it is the arm the
     C1 measurement was taken on both sides of, and it is spelled `== "1"`,
     so a switch with three measured multiples behind it is off by default and
     ungridded.
   - `MOJOTREES_GPU_SEARCH_RESTAGE_HOIST`, `train_gpu.mojo`, `== "1"`.
   - `MOJOTREES_GPU_FUSE_STEP_STAGE` and `MOJOTREES_GPU_FUSE_COPY_STEP`,
     `gpu_resident_round.mojo`, both `== "1"` through
     `comptime STEP_STAGE_FUSION_VAR` and `comptime COPY_STEP_FUSION_VAR`. The
     comptime alias is why a name-level scan of `getenv` arguments misses them.
   - `MOJOTREES_RAW_PREDICT` and `MOJOTREES_PREDICT_TRACE`, `predict.mojo`,
     `!= "0"` and a trace sink. Section 3L exists now and is where they go.

   **The shape of the miss is worth more than the list.** Four of the six are
   on planes the grid's axes do name, and they were missed one at a time; two
   are on the prediction walker, which no axis of this grid reaches, and they
   were missed as a class. A grid whose header claims completeness needs the
   grep in section 1 run against its own row set, not just against the tree.

7. **A conflict in `train_gpu.mojo`, RESOLVED 2026-08-17 in this grid's
   favor.** As audited, two comments in `_grow_tree_gpu_device_search` said the
   `random_strength` line is "NOT REACHED BY ANY FIT TODAY" because
   `_check_device_search_supported` refuses `params.extra.is_active()`, while
   `_device_search_unsupported_reason` in the same file said the opposite and
   `OBLIVIOUS_WAIT_CENSUS.md` built its whole six-drain finding on the blanket
   refusal having been retired. The grid followed the reason-function, because
   that is the one the check calls, and marked `NOISE_HOIST` reachable. **That
   was right.** At head there is one comment at that site and it begins
   "REACHED. The 'NOT REACHED BY ANY FIT TODAY' note that stood here was WRONG",
   with the call graph traced link by link;
   `docs/design/GROWTH_POLICY_REACH.md` carries the same trace. So rank 5 is
   reachable, it was measured, and it came in indistinguishable. Nothing here
   is owed to another lane any longer.
