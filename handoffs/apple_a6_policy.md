# A6: Apple-silicon tuning policy

Status: landed as a standalone layer, not yet wired into any caller. Nothing
outside the three files below was touched, and nothing was committed or
staged.

Files:

- `src/mojoboost/apple_gpu_policy.mojo` (new)
- `tests/parallel/test_apple_gpu_policy.mojo` (new, `tests/parallel/` created)
- `handoffs/apple_a6_policy.md` (this file)

Test run: `pixi run mojo run -I src tests/parallel/test_apple_gpu_policy.mojo`,
18 tests, 18 passed. One docstring-capitalization warning was fixed after that
run; it is a docstring reword in the test file with no effect on any
assertion.

## What the layer does

Reported device facts and a dataset shape in, a launch plan out. It opens no
device, allocates nothing, and launches nothing, so all of it is testable on a
CPU-only machine and on a machine that has none of the Apple parts involved.

Inputs, all of them reported or passed in, none of them assumed: API name,
architecture string, GPU core count (`MULTIPROCESSOR_COUNT`), threads per
threadgroup, threadgroup memory per block, device memory budget, and the
`(n_rows, n_features, n_bins)` shape.

Outputs on `TuningPolicy`: `block_threads`, `n_tiles`, `rows_per_tile`,
`strategy`, `partial_cell_limit`, `partial_cells`,
`resident_blocks_per_core`, and a `CrossoverInputs` value.

## The four things it decides differently from `gpu_tiling.mojo`

Each of these is architectural, derived from a number the device reports.
None of them is a measurement, and none is a per-chip constant.

1. **Residency from reported threadgroup memory.** `gpu_tiling.mojo` targets a
   fixed 8 blocks per multiprocessor. This layer computes how many
   `n_bins`-wide partial histograms the advertised threadgroup memory
   actually holds, capped at 8 and floored at 1. A device whose shared memory
   fits 8 partials gets the same tile geometry it does today (see "where the
   two layers agree exactly" below); a 16 KiB device at 255 bins gets 5
   instead of 8.
2. **Partial budget from reported device memory.** A fraction of the reported
   budget rather than a flat 64 MiB: 1/16 on discrete memory, 1/32 when
   memory is unified, capped by the same 64 MiB ceiling. An unreported budget
   (0) falls back to the ceiling, so a device that says nothing about its
   memory gets the same tile geometry it does today. Unified memory gets the
   tighter
   fraction because on Apple that budget is system RAM, shared with the
   dataset and the binned matrix the host is holding.
3. **Threadgroup size bounded by the row count.** A 256-thread block scanning
   40 rows leaves seven eighths of its lanes idle, which is a large fraction
   of a device that reports 7 to 10 cores. Still a warp multiple, still at
   least one warp, still never above the device maximum.
4. **Crossover inputs reported, threshold withheld.** `CrossoverInputs`
   carries api, generation, core count, unified-memory flag, cells, bins, and
   `device_parallel_width` (`core_count * block_threads`). `min_cells` is
   `CROSSOVER_DISABLED` (-1) in every case this module can construct.

### Where the two layers agree exactly

Worth stating precisely, because "it plans the same" is easy to overclaim.
Hand `derive_policy` a profile whose threadgroup memory fits 8 partials at the
given bin count and whose memory budget is 0 (which is *every* profile today,
since no budget accessor exists), and it produces **the same `n_tiles` and the
same `rows_per_tile`** as `derive_tiling` on the equivalent `DeviceCaps`.
Divergences #1 and #2 are inert in that case by construction.

The memory bound is spelled differently in the two modules — `derive_tiling`
computes `budget // (hist_cells * 12)` while this layer computes
`(budget // 12) // hist_cells` — but those are equal for positive integers by
`floor(floor(x/n)/m) == floor(x/(n*m))`, so the difference is presentational.

`block_threads` is the one output that can still differ, and only when
`n_rows < 256`, where divergence #3 narrows the launch to the warps the rows
can actually feed. That cannot perturb the tile geometry: at fewer than 256
rows, `min_rows_per_tile` is at least 256 either way, so both layers land on a
single tile regardless of the block width they chose.

Deliberately **not** divergent: the strategy choice. Apple GPUs implement
global integer atomics, but nothing in this project has measured atomic
throughput against the tiled reduction on any Apple part, so the rule is the
one already shipping (tiled when there is more than one tile to reduce,
atomic when there is not).

## Fixtures, and what is and is not measured

- `apple_synthetic(generation)` for M1 through M5. Every one is flagged
  `synthetic`. Core counts are published base-configuration GPU core counts,
  taking the lowest published base configuration where a generation ships
  more than one (7, 8, 8, 10, 10). Threadgroup memory is the portable Apple
  floor (16 KiB), not what any particular chip advertises. Memory budget is 0,
  meaning unread. **None of these is a performance measurement, and no
  measurement of M1, M2, M3, or M5 exists in this repository.** Replace each
  with a reading from real hardware, clearing `synthetic`, as that hardware
  becomes available.
- `apple_m4_observed()` is the one Apple capability triple this repository has
  read off hardware: `(10, 1024, 32768)`, the same numbers
  `tests/test_gpu_tiling.mojo` already pins as "Apple M4 reports these
  exactly". It is `synthetic = False` and it is a **named fixture, not a
  fallback**: nothing inherits its numbers. A test asserts that the portable
  fallback and the synthetic M4 both differ from it, and that a device
  reporting other numbers gets a different plan.
- `GpuProfile.generic()` is the portable fallback for Metal, CUDA, and HIP
  alike: `API_UNKNOWN`, not Apple-shaped, not unified, and it mirrors
  `gpu_tiling.mojo`'s fallback constants.

## Exact integration

Nothing calls this yet. Three steps, in this order.

### 1. The call site that reads the device (in `gpu_tiling.mojo` or its caller)

`query_device_caps` already reads the three attributes. Add the API and
architecture strings and hand everything to `from_reported`:

```mojo
from .apple_gpu_policy import GpuProfile, derive_policy

def query_gpu_profile(ctx: DeviceContext) -> GpuProfile:
    return GpuProfile.from_reported(
        ctx.api(),
        _arch_or(ctx, ""),
        _attribute_or(ctx, DeviceAttribute.MULTIPROCESSOR_COUNT, 0),
        _attribute_or(ctx, DeviceAttribute.MAX_THREADS_PER_BLOCK, 0),
        _attribute_or(ctx, DeviceAttribute.MAX_SHARED_MEMORY_PER_BLOCK, 0),
        0,  # memory budget: see step 3
    )
```

`_attribute_or` is passed `0` here on purpose: `from_reported` applies the
same conservative fallbacks itself, so a backend that answers nothing
degrades the plan in one place rather than two. The architecture string has
no existing accessor in this repo; until one exists, pass `""`, which yields
`APPLE_GEN_UNKNOWN`. That costs nothing today because **no decision in
`derive_policy` branches on the generation** (it is metadata for the
crossover inputs and for `describe_policy` output), so wiring it can wait.

### 2. Consuming the plan

`TuningPolicy` and `HistogramTiling` carry the same four launch fields under
the same names, so the kernel launch code changes only in where it gets them:

```mojo
var profile = query_gpu_profile(ctx)
var policy = derive_policy(profile, n_rows, n_features, n_bins, requested)
# policy.block_threads, policy.n_tiles, policy.rows_per_tile,
# policy.strategy, policy.partial_cells  -- as today
# policy.partial_cell_limit              -- new: the budget ceiling
```

For the feature-subsampling path that re-derives against an already allocated
buffer, `derive_tiling`'s `max_partial_cells` argument has no counterpart
here; that path should keep calling `derive_tiling` until step 3 is done.

### 3. Two follow-ups this lane deliberately left open

- **Memory budget is passed as 0.** `DeviceContext` has no total-memory
  accessor used anywhere in this repo, so every profile currently reports an
  unknown budget and falls back to the 64 MiB ceiling, which is exactly
  today's behavior. Divergence #2 above only takes effect once a real budget
  is threaded in. Find the attribute (or query it per backend), pass it, and
  the unified-memory tightening starts applying. `partial_budget_bytes` is
  unit-tested for both branches already.
- **`max_partial_cells` equivalent.** Add an optional capacity argument to
  `derive_policy` mirroring `derive_tiling`'s, so feature subsampling can
  re-derive against an allocated buffer without reallocating.

### 4. Collapsing the mirrored constants

`apple_gpu_policy.mojo` copies 13 constants and one function from
`gpu_tiling.mojo` (marked by a `--- Mirrors ---` block) so this layer stands
alone while it lands alongside concurrent work on the tiling module.
`test_mirrored_constants_match_gpu_tiling` asserts every mirror still equals
its source, so the two layers cannot silently disagree about a warp, a
budget, or a strategy code.

At integration, delete the mirror block and import from `gpu_tiling` instead.
The values are identical today, so this is a pure deletion; drop that test
with the block. Note that `MAX_RESIDENT_BLOCKS_PER_CORE` mirrors
`TARGET_BLOCKS_PER_SM` and changes meaning from a fixed target to a ceiling
on the derived value, so keep it named separately rather than importing it.

### 5. Test wiring

`tests/parallel/` is new and not referenced by any pixi task. Add the file to
the `test` task list in `pixi.toml` when the lane merges:

```
&& mojo run -I src tests/parallel/test_apple_gpu_policy.mojo
```

It needs no accelerator and runs in 0.02s, so it belongs in the plain `test`
task rather than `test-gpu`. It does import `mojoboost.gpu_tiling` for the
mirror check, which pulls in `max.gpu.host`, the same import
`test_gpu_tiling.mojo` already makes on CPU-only runners.

## The one decision worth revisiting: generation drives nothing

The brief lists architecture/generation among the policy's inputs. It is
parsed (`parse_apple_generation`), carried on `GpuProfile`, reported in
`CrossoverInputs` and `describe_policy`, and tested. But **no branch in
`derive_policy` reads it.** That was deliberate: tuning on a part number is
exactly the per-chip special-casing this layer exists to avoid, and no
measurement distinguishes the generations, so a branch on `apple_generation`
would be a performance claim with nothing behind it.

There is one use of the generation that would *not* be a performance claim,
and an integrator may reasonably want it: using it to improve a **capability
fallback** when the device reports nothing. Today a Metal device that fails to
answer `MULTIPROCESSOR_COUNT` falls back to the portable `FALLBACK_CORE_COUNT`
of 16, which is too high for every Apple part in the M1-M5 range and would
overshoot the tile count. If the architecture string names a generation, the
published base-configuration count is a strictly better guess than 16:

```mojo
# in GpuProfile.from_reported, replacing the core-count line
var cores = core_count
if cores < 1 and generation != APPLE_GEN_UNKNOWN:
    cores = synthetic_apple_core_count(generation)   # raises: needs `raises`
if cores < 1:
    cores = FALLBACK_CORE_COUNT
```

This substitutes a published specification for a portable guess on a path
that is otherwise guessing anyway. It is still not a measurement. Two
consequences to weigh before applying it: `from_reported` would have to become
`raises`, and the "no chip's numbers reach another chip" property gets a
caveat, since a device reporting nothing would then inherit its generation's
spec sheet. It is left unapplied because this lane's budget allowed one test
run, which has been spent, and shipping an unexercised branch through
`from_reported` (the entry point every caller uses) is the worse trade.

## Relationship to the parallel lanes (checked read-only, nothing edited)

- **A9, `python/mojoboost/device_selection.py`.** That lane models the same
  CPU/GPU crossover on the Python side, and its `CROSSOVER_RULES` table ships
  **empty** for exactly the reason `CrossoverInputs.min_cells` here is
  `CROSSOVER_DISABLED`: the one end-to-end measurement (M4,
  `bench/bench_train_gpu.mojo`) came out slower than the CPU trainer. The two
  layers agree, independently. They are not redundant: A9 decides *whether to
  use the GPU at all* and explains that choice to a user; this layer decides
  *how to launch* once the GPU has been chosen. If a benchmark ever produces a
  threshold, it belongs in A9's versioned table with its `evidence` field, and
  `min_cells` here should be fed from that table rather than acquiring a second
  independent value.
- **A7, `docs/APPLE_UNIFIED_MEMORY.md` and `bench/apple/unified_memory.mojo`.**
  That lane records that no unified-memory route has been compiled or run, and
  that no zero-copy claim is licensed yet. Nothing in this layer depends on
  one. The unified-memory branch here is only the budget *fraction*
  (divergence #2), which rests on the uncontested hardware fact that the pool
  is shared, not on any claim about zero-copy routes or their performance.
- **A5, `src/mojoboost/gpu_runtime.mojo`.** Sync, staging, and phase counters;
  no capability queries. Re-checked after that lane landed: **no
  total-memory or memory-budget accessor exists anywhere in `src/` or
  `bench/`**, so the step-3 follow-up above (thread a real budget in) is still
  open and the budget is still passed as 0.

## Review notes on the delivered code

- `partial_budget_bytes` has a `if bytes < 0` guard that cannot fire: the
  `<= 0` early return above it means the dividend is positive and the divisors
  are compile-time positive constants. Harmless, and left in place rather than
  churning verified source for a dead branch. Drop it whenever the file is
  next touched.
- `derive_policy` honors an explicit `STRATEGY_TILED` request even when the
  resulting `partial_cells` exceeds `partial_cell_limit`, matching
  `derive_tiling`'s "an explicit request wins" behavior. Both numbers are on
  the returned plan so the caller can see it happened. Only the `AUTO` path is
  guaranteed to stay inside the limit, and that guarantee is what the test
  asserts across every profile and shape it sweeps.
- A dataset with fewer rows than one warp still launches one full warp;
  `WARP_GRANULARITY` is a hard floor, so `derive_block_threads` never returns
  a partially masked launch.
