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

---

## 1. The population, and how it reconciles

    grep -rho 'MOJOTREES_[A-Z0-9_]*' src/ bindings/ | sort -u    ->  90 hits
    real switches that code reads                                ->  85
    documented but read by nothing                               ->   1
    a build-script shell variable, not a runtime switch          ->   1
    docstring line-wrap fragments, not names                     ->   3
                                                                     ---
                                                                      90

The five that are not real switches.

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

**Kind.** PERFORMANCE means same answer, different speed. BEHAVIOR means the
model or the numbers move. DIAGNOSTIC means trace, census, or profile output.
CONFIGURATION means it picks a backend, a worker count, a role, or a geometry.

**Reaches.** LEAF (lossguide), DEPTH (depthwise), SYM (oblivious, the CatBoost
shape), and CPU or GPU. `ALL` means all three policies on that backend.

**Verdict.** The five assigned values, plus one convention the assignment did
not cover. Fourteen switches are already default ON, so "should it be the
default" is answered by the shipped state. Those carry
**SHOULD BE THE DEFAULT (already is)** and are excluded from the ranked
candidate list in section 4, because there is nothing to flip. Where such a
switch's off arm has never been priced that is said in the Measured column,
not laundered into the verdict.

---

## 3. The grid

### 3A. The symmetric (oblivious) GPU plane

This is where the four switches measured on 2026-08-17 live. All four are read
in, or dispatched from, code that only `grow_tree_device_oblivious` reaches.

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT` | `gpu_leaf_batching.oblivious_subtract_requested`, `return getenv("MOJOTREES_GPU_OBLIVIOUS_SUBTRACT") == "1"`. Dispatched at `histogram_gpu.enqueue_desc_level_children`, `if oblivious_subtract_requested():` | unset behaves as off | PERFORMANCE | SYM GPU only. The branch is placed in `histogram_gpu` and not inside the batcher "so that a two-item leaf-wise plan cannot reach the subtracting arm by having the environment set" | **MEASURED**, 1.78x, 21.97 s to 12.34 s at 799,110 x 100 x 100 trees, rmse identical to nine decimals (2026-08-17 lane brief). Bit-identity is an exact integer argument at `_batch_hist_atomic_subtract_kernel`. `docs/design/OBLIVIOUS_WAIT_CENSUS.md` tabulates the row-build counts, 126 to 63 at depth 6 | **SHOULD BE THE DEFAULT** |
| `MOJOTREES_GPU_OBLIVIOUS_WIDE` | `gpu_split_search.oblivious_wide_scan_requested`, `return getenv("MOJOTREES_GPU_OBLIVIOUS_WIDE") == "1"`. Dispatched in the oblivious launch, `if oblivious_wide_scan_requested():`, which then refuses a bin count above `OBLIVIOUS_WIDE_MAX_BINS_PER_THREAD * OBLIVIOUS_WIDE_THREADS` | unset behaves as off | PERFORMANCE | SYM GPU only | **MEASURED**, 4.5 percent, resolved, bit-identical (2026-08-17 lane brief). No results file under `bench/results/` names this variable, so the measurement is not yet filed | **SHOULD BE THE DEFAULT** |
| `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` | `gpu_resident_round.oblivious_skip_last_build_requested`, `return getenv(OBLIVIOUS_SKIP_LAST_BUILD_VAR) == "1"`. Consumed at `var skip_last_build = oblivious_skip_last_build_requested()` above the level loop, and again in `train_gpu` to size the profile's launch count | unset behaves as off | PERFORMANCE | SYM GPU only | **MEASURED**, about 1.20x, direction solid, magnitude a lower bound because the box drifted upward during the run (2026-08-17 lane brief). `OBLIVIOUS_WAIT_CENSUS.md` still reads "Off by default because the time is unmeasured", which is now stale | **SHOULD BE THE DEFAULT** |
| `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST` | `gpu_resident_round.oblivious_noise_hoist_requested`, `return getenv(OBLIVIOUS_NOISE_HOIST_VAR) == "1"`. Also read in `train_gpu._search_record_slots`, `if oblivious_noise_hoist_requested(): want = oblivious_leaf_budget(params) + params.max_depth` | unset behaves as off | PERFORMANCE | SYM GPU only, **and only when `random_strength > 0`**. `_copy_noise` returns immediately at zero. The CatBoost-mode default set does set it, `params.CATBOOST_RANDOM_STRENGTH` is 1.0 | **UNRESOLVED.** Measured slower on 2026-08-17 in a run confounded by drift, so neither result stands. The count behind it is read from source, six `enqueue_copy` drains per depth-6 tree, each between a level's child build and the next level's search | **NEEDS MEASURING** |

### 3B. GPU split search, shared and leaf-wise

| name | read at | default | kind | reaches | measured? | verdict |
|---|---|---|---|---|---|---|
| `MOJOTREES_GPU_SPLIT_WIDE` | `gpu_split_search.wide_scan_requested`, `return getenv("MOJOTREES_GPU_SPLIT_WIDE") == "1"`, reached through `wide_scan_for(has_categorical)`, which ANDs it with "no categorical feature", and stored once at construction as `self.wide_scan = wide_scan_for(any_cat)` | unset behaves as off | PERFORMANCE | LEAF GPU (device split search). Not SYM, which runs the separate oblivious kernel. Not DEPTH in practice, because the device-resident plane refuses non-leaf-wise growth (section 6) | **ASSERTED.** The docstring says so and names the sibling result, "the shipped leaf-wise default is now the only scan in this file still running one lane per (leaf, feature) while its own sibling has a measured win" | **NEEDS MEASURING** |
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
| `MOJOTREES_GPU_ROW_COMPACTION` | `GpuActiveRows.__init__`, `if _env_int("MOJOTREES_GPU_ROW_COMPACTION", 0) != 0: self.set_row_compaction(True)`. Also honored from `train_gpu`, `builder.rows.set_row_compaction(row_compaction or builder.rows.row_compaction_requested())` | unset behaves as off | PERFORMANCE | ALL GPU | **ASSERTED.** Order-preserving and bit-identical by construction. An arithmetic prediction in the lane brief puts it 1.7x to 2.5x underwater at 1M x 50, because partition touches 2.77 rows for every one the histogram touches. Never measured | CORRECTLY OFF, on a prediction rather than a measurement |
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

---

## 4. Ranked candidates

Every switch whose verdict is SHOULD BE THE DEFAULT or NEEDS MEASURING,
highest expected value first, with the single measurement that would settle
it. The fourteen "already is" rows are excluded, because there is nothing to
flip.

Counts over the 85. By kind, CONFIGURATION 43, PERFORMANCE 25, DIAGNOSTIC 12,
BEHAVIOR 5. By verdict, CORRECTLY OFF 33, NEEDS MEASURING 14, SHOULD BE THE
DEFAULT (already is) 14, DIAGNOSTIC 13, DEAD 8, SHOULD BE THE DEFAULT 3.

| rank | switch | why here | the one measurement that settles it |
|---|---|---|---|
| 1 | `MOJOTREES_GPU_OBLIVIOUS_SUBTRACT` | 1.78x measured on the shipped symmetric default, bit-identical by an exact integer argument. Largest single number on the board | None to settle the win. What is left is a **regression gate**, an interleaved on/off pair at a second shape, ideally 250k x 50, confirming direction and rmse identity before the default flips |
| 2 | `MOJOTREES_GPU_OBLIVIOUS_WIDE` | 4.5 percent, resolved, bit-identical, on the same plane. It composes with rank 1 on a different axis, scan width against accumulation width | An interleaved pair with **`SUBTRACT=1` already on**, to confirm the 4.5 percent survives the halved accumulation. Rank 1 changes what fraction of the level is scan |
| 3 | `MOJOTREES_GPU_SPLIT_WIDE` | The same widening of the same `block_dim=1` scan on the leaf-wise plane, still off, while its oblivious sibling measured 4.5 percent. The shipped leaf-wise search runs two hundred single-lane threadgroups | One interleaved `bench-train-gpu` pair, wide against narrow, on a **non-categorical** dataset, at 1M x 100 leaf-wise. `wide_scan_for` ANDs the request with "no categorical", so a categorical arm silently measures the narrow kernel twice |
| 4 | `MOJOTREES_GPU_OBLIVIOUS_SKIP_LAST_BUILD` | About 1.20x, direction solid, magnitude a lower bound. Independent axis from rank 1, so the two compose to 31 row builds per depth-6 tree against 126 | A **repeat** of the interleaved pair on a settled box, with `SUBTRACT=1` on, to recover the magnitude the drift ate. Direction does not need re-establishing |
| 5 | `MOJOTREES_GPU_OBLIVIOUS_NOISE_HOIST` | Six full-queue drains per depth-6 tree, each sitting in the worst position a drain can occupy, and the CatBoost-mode default set puts them there. One run said slower and was confounded | An interleaved pair on a fit that **actually sets `random_strength > 0`** and no categorical column, so `_copy_noise` is live. A fit at `random_strength = 0` measures nothing, which is how this became invisible in the first place |
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

5. **Four symmetric-GPU switches have no leaf-wise twin and one leaf-wise
   switch has no symmetric twin.** `OBLIVIOUS_SUBTRACT`, `OBLIVIOUS_WIDE`,
   `OBLIVIOUS_SKIP_LAST_BUILD` and `OBLIVIOUS_NOISE_HOIST` are SYM only;
   `SPLIT_WIDE` is LEAF only. `OBLIVIOUS_SUBTRACT`'s placement in
   `histogram_gpu` rather than in the batcher is deliberate, so that a
   two-item leaf-wise plan cannot reach the subtracting arm.

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
halves the width of every level that runs. **They compose exactly and have
never been measured on together.** `OBLIVIOUS_WIDE` acts on a third axis, the
scan's threads per (leaf, feature), and `NOISE_HOIST` on a fourth, the number
of drains per tree. None of the four adds a launch, so
`oblivious_launch_census(6)` is 62 in all sixteen combinations except that
`SKIP_LAST` removes the last level's build.

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
arm is distinguishable from a working one".

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

1. **`MOJOTREES_GPU_SPLIT_WIDE` on the depth-wise plane.** Marked LEAF, and
   the reasoning is that depth-wise cannot take the resident plane. What was
   not traced is whether `_device_search_incremental` constructs the same
   `GpuSplitSearcher` and therefore also honors `self.wide_scan`. Reading
   `train_gpu._device_search_incremental` end to end would settle it. If it
   does, the row becomes LEAF and DEPTH and rank 3's measurement should be
   taken leaf-wise anyway, since that is the default policy.

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

6. **A conflict in `train_gpu.mojo` that this lane could not resolve and did
   not try to.** Two comments in `_grow_tree_gpu_device_search` say the
   `random_strength` line is "NOT REACHED BY ANY FIT TODAY" because
   `_check_device_search_supported` refuses `params.extra.is_active()`.
   `_device_search_unsupported_reason`, in the same file, says the opposite,
   that the `random_strength` blanket refusal is retired and is now
   conditional on there being no categorical column, and
   `OBLIVIOUS_WAIT_CENSUS.md` builds its whole six-drain finding on the
   refusal having been retired. The reason-function is the one the check
   actually calls, so this grid follows it and marks `NOISE_HOIST` reachable.
   The two stale comments should be corrected by whoever owns that file. If
   they turn out to be right, rank 5 is unreachable and drops off the list.
