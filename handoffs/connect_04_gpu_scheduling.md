# Connect 04: multiclass batching, hybrid scheduling, Apple histogram policy

Lane 04. Owned files: `src/mojoboost/gpu_multiclass_batch.mojo`,
`src/mojoboost/hybrid_leaf_scheduler.mojo`,
`src/mojoboost/apple_histogram_policy.mojo`, `src/mojoboost/gpu_tiling.mojo`,
and this handoff.

**Two files outside that list were edited, at the user's explicit
direction:** `src/mojoboost/histogram_gpu.mojo` and
`src/mojoboost/train_gpu.mojo`, both Task 01's. The user asked this lane to
implement §7.2 -- the blocking gap that kept batched gradients from reaching
the grower -- rather than leave it as a patch request. §7.2 records exactly
what was added, what mechanism was chosen instead of the one originally
requested, and why. Nothing else in those two files was changed; the existing
sequential multiclass loop is byte-identical and the new path is unreachable
without `MOJOBOOST_GPU_CLASS_BATCH`.

**Nothing here was run.** No Mojo, no Pixi, no build, no test, no benchmark.
Every statement below is from reading the source. No claim is made about
correctness, performance, parity, or hardware behavior; the arithmetic
identities argued in §4 are arguments, not results, and §11 lists the
commands that would check them.

**One note on the working tree.** Another lane committed while this lane was
mid-edit: `dc21f03` and `860b1cf` swept in the in-progress state of
`gpu_tiling.mojo`, `apple_histogram_policy.mojo`, and
`gpu_multiclass_batch.mojo`. This lane did not commit and did not ask for
that. No work was lost -- the working tree holds the finished state of all
four files, and the remaining `hybrid_leaf_scheduler.mojo` changes are
uncommitted. Anyone diffing this lane's work must diff the working tree, not
`HEAD`, and must not "restore" those three files from an earlier commit.

---

## 1. What existed before, and which implementation was made authoritative

Six modules already implemented parts of "how is a GPU histogram launched,
for how many classes, and on which device". They did not know about each
other, and three of the four owned ones had no importer at all.

| Capability | Implementations found | Authoritative one | Why |
| --- | --- | --- | --- |
| Launch geometry (tiles, block width, strategy, partial budget) | `gpu_tiling.derive_tiling`; a second copy of the same arithmetic inside `apple_histogram_policy.derive_histogram_plan` at `SPEC_LEVEL_SHAPE`; a third partial copy in `apple_gpu_policy.mojo`'s mirrored constants (Task 05) | `gpu_tiling.resolve_tiling` (new) | It is the portable layer every backend already goes through, and the one both callers can express their bounds against without either owning the other |
| Block-width clamp | `gpu_tiling.derive_block_threads`, `apple_histogram_policy._shape_block_threads`, `apple_gpu_policy.derive_block_threads` | `gpu_tiling.clamp_block_threads` (new) for the rule; each layer keeps its own choice of what to clamp | The rule (warp multiple, device max, one-warp floor) is not a policy question; what to clamp is |
| Partial-buffer budget in cells | `gpu_tiling.derive_tiling` inline, `apple_histogram_policy.baseline_partial_cell_limit` | `gpu_tiling.partial_cell_limit_for` (new) | Same number, two spellings; `baseline_partial_cell_limit` now delegates |
| Launch count per node | a defaulted literal `1` on `hybrid_leaf_scheduler.LeafWork.node_of` | `gpu_tiling.launches_for_strategy` (new) | The cost model must charge for the launches the geometry will actually issue |
| Class grouping for a softmax round | `gpu_output_planes.plan_class_batches` (not owned, already complete) | `gpu_output_planes.plan_class_batches`, consumed | A second grouping planner that disagreed by a class would be worse than none |
| Bin/row sharing across classes | `gpu_multiclass_batch`'s two kernels, dispatched by whichever method the caller picked | `GpuClassBatch.enqueue_level_histogram` (new) dispatching on `gpu_output_planes.BatchEligibility` | Sharing is licensed by what the classes actually share, not by the method name a caller reached for |
| Histogram reuse | `histogram_cache_policy.mojo` (not owned) decides what is reusable; `hybrid_leaf_scheduler` ignored it | `histogram_cache_policy`, consumed through `ReuseOffer` | The reuse argument is made once, there; the scheduler only carries its answer |

No new module, registry, policy engine, trainer, or model representation was
created. Everything added is either a factoring of code that was already in
these four files or a consumption of a module that already owned the
decision.

---

## 2. The layer, after

```
gpu_tiling.mojo                 tile arithmetic (resolve_tiling), block-width
                                clamp, partial-cell budget, launches per
                                strategy. Portable; no Apple, no device open.
     ^                    ^                         ^
apple_histogram_policy    gpu_multiclass_batch      hybrid_leaf_scheduler
 - specialization ladder   - GpuClassBatch buffers   - per-leaf placement
 - shape-derived geometry  - two batched kernels     - reuse / cpu / gpu
   via resolve_tiling      - plan_geometry() from    - LeafWork.for_tiling()
 - plan_class_schedule() ->  gpu_tiling's policy       reads the launch count
   plan + ClassBatchPlan   - enqueue_level_histogram   off a resolved tiling
 - HistogramPlan.tiling()    dispatches on
   projects back down       BatchEligibility
```

Dependency direction is strictly upward; there is no cycle.
`apple_histogram_policy` now also imports the pure planning half of
`gpu_output_planes` (`BatchEligibility`, `ClassBatchPlan`,
`plan_class_batches`, `env_class_batch`), which imports only
`parallel._env_int`. `hybrid_leaf_scheduler` gained one import,
`gpu_tiling.HistogramTiling`, and deliberately did **not** take a dependency
on `apple_histogram_policy`: a caller holding a `HistogramPlan` passes
`plan.tiling()`, so the scheduler stays a pure-host module with two small
dependencies.

---

## 3. Call path, before and after

### 3.1 Launch geometry

*Before.* `GpuHistogramBuilder.__init__` (histogram_gpu.mojo:427) and
`set_features` (histogram_gpu.mojo:538) call `derive_tiling` and store
`self.tiling`; every enqueue launches from that. Separately,
`GpuHistogramBuilder.histogram_plan` (histogram_gpu.mojo:815) calls
`derive_histogram_plan`, and its only consumer is the batching verdict at
histogram_gpu.mojo:1035. So the ladder was consulted for one yes/no question
and never for a launch.

*After.* Unchanged in behavior, and now adoptable in one line:
`HistogramPlan.tiling()` returns the plan's geometry as the exact
`HistogramTiling` every launch site already takes, and at
`SPEC_LEVEL_BASELINE` it equals `self.baseline` field for field. The
provider change that would make the plan the launch is §7.1; this lane did
not make it, because `histogram_gpu.mojo` is Task 01's file.

### 3.2 Multiclass

*Before.* `_train_multiclass_gpu_rounds` (train_gpu.mojo:1499) walks classes
one at a time in both branches. In the device-gradient branch
(train_gpu.mojo:1530-1560) each class calls
`builder.fill_softmax_gradients_device(state, k)`, grows a whole tree, then
`builder.update_raw_device(...)`. `gpu_multiclass_batch.mojo` had no importer
in `src/`; its kernels, its buffers, and its round guard were unreachable
from any call path.

*After.* The module is reachable, and reached:

- `apple_histogram_policy.plan_class_schedule(...)` returns a `ClassSchedule`
  holding both halves of the decision (the launch plan and the
  `ClassBatchPlan`), and its default is the sequential path.
- `GpuHistogramBuilder.class_schedule(n_classes, eligibility)` asks it from
  the builder's own profile and shape (§7.2).
- `_train_multiclass_gpu_rounds` calls that once per fit. If the schedule is
  sequential -- the shipped default -- the existing loop runs unchanged and
  nothing below happens. Otherwise it hands off to
  `_train_multiclass_gpu_batched`.
- That helper allocates `GpuClassBatch.for_plan(ctx, ..., schedule.batches)`,
  drives the round with `MulticlassRoundGuard.note_batch(plan, b)` (which
  refuses out-of-order batches), fills one batch's gradients at a time, and
  copies each slot into the builder with `scatter_slot` /
  `fill_batched_gradients` before `grow_tree_gpu`.

Trees are still appended in ascending `k` and tree `(i, k)` still lands at
`i * n_classes + k`.

Not yet reached: `GpuClassBatch.enqueue_level_histogram` and the two batched
histogram kernels. Those need a grower that consumes a whole frontier level's
histograms at once (§9 item 7). The batched path today shares the gradient
fill and the magnitude readback, not the bin read.

### 3.3 Hybrid placement

*Before.* `grow_tree_gpu` (train_gpu.mojo:679) builds every node on the
device: `builder.build_leaf(root)` at :771 and the two children at :875/:878.
`hybrid_leaf_scheduler.mojo` had no importer.

*After.* Still no importer, and still decides `PLACE_GPU` for every leaf
under the shipped configuration (§6). What changed is that a placement is now
a statement about a whole split rather than half of one, and that the launch
count it charges can come from the resolved geometry instead of a default.

---

## 4. Connections completed (in the owned files)

1. **`resolve_tiling` is the one tile rule.** `derive_tiling` is now a
   wrapper that supplies the portable bounds; `derive_histogram_plan`'s
   `SPEC_LEVEL_SHAPE` supplies the shape-derived ones. About 45 lines of
   restated arithmetic were deleted from `apple_histogram_policy.mojo`.

   *Numerically identical for `derive_tiling`.* The only rewritten term is
   the memory bound: it was
   `PARTIAL_BUDGET_BYTES // (hist_cells * BYTES_PER_PARTIAL_CELL)` and is now
   `partial_cell_limit_for(0) // hist_cells`, i.e.
   `(PARTIAL_BUDGET_BYTES // BYTES_PER_PARTIAL_CELL) // hist_cells`. For
   positive integers `floor(A / (b*c)) == floor(floor(A / c) / b)`, so the
   two agree at every shape. Both spellings were already present in the tree
   (the second is what `baseline_partial_cell_limit` computed), which is how
   the duplication was found.

   *One deliberate behavior change, at `SPEC_LEVEL_SHAPE` only.* The
   restated arithmetic ignored `MOJOBOOST_GPU_HIST_STRATEGY`; the shared rule
   honors it, because `resolve_tiling` resolves `STRATEGY_AUTO` through
   `env_strategy()` as `derive_tiling` always has. Visible only when that
   variable is set to `atomic` **and** a specialization level of `shape` or
   above was requested -- two off-by-default switches at once -- and it makes
   the level agree with the baseline it is compared against. Recorded here
   rather than hidden because it is the one place the fusion is not a
   refactor.

2. **`clamp_block_threads`** is the single block-width rule;
   `derive_block_threads` and `_shape_block_threads` both end in it. Same
   arithmetic both sides, so no shape changes width.

3. **`launches_for_strategy` / `HistogramTiling.launches()` /
   `HistogramPlan.gpu_launches()`.** One launch for the atomic strategy, two
   for the tiled one, written down once. `LeafWork.for_tiling` reads it off a
   resolved tiling, so the hybrid cost model charges what the device will
   issue rather than the default `1`. `STRATEGY_AUTO` has no launch count and
   raises, because an unresolved request is not a plan.

4. **`HistogramPlan.tiling()`** projects the ladder's output onto the
   descriptor the launch sites take. This is what makes the specialization
   ladder adoptable without teaching any call site about levels.

5. **`plan_class_schedule` + `ClassSchedule`.** The class grouping is run
   from a resolved plan: `plan.n_tiles` and
   `profile.max_shared_memory_per_block` are exactly the two inputs
   `plan_class_batches` needs and previously had to be guessed at by a
   caller. `ClassSchedule.shared_group_size()`, `shares_bin_reads()`,
   `is_sequential()`, and `bin_passes_per_round()` are the pure decisions
   Task 01 and Task 05 consume; `describe_schedule` is the trace line.

6. **`GpuClassBatch` takes its geometry from the policy.** The struct now
   holds the `DeviceCaps` it was built from and exposes
   `plan_geometry(node_rows)`, which is `derive_tiling` over the batch's own
   shape with the strategy pinned to `STRATEGY_ATOMIC` (both batched kernels
   accumulate with atomics and allocate no partial buffer, so a tiled budget
   would be fiction). `enqueue_level_histogram` uses it whenever the caller
   passes `rows_per_tile = 0`.

7. **Bin sharing is dispatched from `BatchEligibility`, not from the call
   site.** `enqueue_level_histogram` takes the eligibility record, the batch
   size, and the per-threadgroup class count, and picks the shared-row kernel
   or the ranged one. It returns whether reads were shared, so a caller
   reports what it got.

8. **A wide batch can still share reads.** `_batch_hist_shared_kernel` gained
   `slot_begin` and `write_counts`, and `enqueue_shared_groups` issues
   `ceil(k_count / per_block)` launches over contiguous ascending slot runs.
   This closes a real gap between the planner and the kernel:
   `ClassBatchPlan.per_block` is capped at `SHARED_CLASS_CAP` (4) while
   `batch_size` is capped by memory, so before this a batch of 8 with
   `per_block` 4 had no launch at all -- `enqueue_shared_histogram` refuses
   `k_count > SHARED_CLASS_CAP`. The output is zeroed once before the first
   group and the batch's single count plane is written by the first group
   only, since every group scans the same rows.

   *Why this changes no cell.* A class's gradient, its own fixed-point scale,
   and the bin a row lands in do not depend on which launch carried the
   class; accumulation is Int32 and integer addition is associative. Slot `s`
   therefore holds the same integers whether it shared a threadgroup with
   three other classes or with none. Argued, not measured -- see §11.

9. **The round guard drives from the plan.** `MulticlassRoundGuard.note_batch`
   derives the class range from `ClassBatchPlan.batch_begin`/`batch_count`
   and requires batches to arrive ascending and without gaps, which is what
   makes "collect by ascending slot" the same statement as "collect by
   ascending class". Out-of-order batches would still produce correct trees
   but would break that identity silently, so they are refused.

10. **Unsupported cases fail clearly.** `batched_shared_bytes()` reports the
    shared kernel's static threadgroup allocation (9216 B: `2 *
    SHARED_CLASS_CAP * MAX_BINS + MAX_BINS` Int32 cells). A batch allocated
    with `shared_counts = True` on a device advertising less is refused at
    construction; `enqueue_shared_groups` refuses it again at the launch, and
    `supports_shared_bin_reads()` lets a caller ask first.

11. **The hybrid scheduler has three answers, not two.** `PLACE_REUSE`,
    `ReuseOffer`, and `place_leaf_with_reuse` add the reuse arm the task
    calls for, consuming `histogram_cache_policy`'s `FRESH`,
    `origins_are_subtractable`, and origin constants rather than restating
    what is reusable. `SplitPlan` now carries a placement for **both**
    children: the sibling is `PLACE_REUSE` with `ORIGIN_SUBTRACTED` -- the
    subtraction both growers already perform, now stated instead of implied
    -- and falls back to an ordinary placement when the parent and the direct
    child came through different arithmetic (a dequantized device parent
    minus a `MODE_HOST_FLOAT64` child is not the sibling).
    `SplitPlan.builds()` reports how many accumulations a split costs.

    Split semantics are untouched: the direct/subtracted division, the
    `n_left <= n_right` tie-break, and the gain scan are all unchanged, and
    reuse hands back the histogram the grower would have produced.

---

## 5. Duplicates fused or quarantined

Fused (inside owned files, safe to remove):

- the second copy of the tile arithmetic in `derive_histogram_plan`
  (~45 lines) -> `resolve_tiling`.
- the second copy of the block-width clamp in `_shape_block_threads`
  -> `clamp_block_threads`.
- the second spelling of the partial-cell budget in
  `baseline_partial_cell_limit` -> `partial_cell_limit_for`.
- the implicit "one launch" assumption in `LeafWork.node_of`
  -> `launches_for_strategy` (the default is kept for callers with no
  resolved geometry, and its docstring now says not to rely on it).

Not fused, and why (all outside this lane's ownership -- see §7):

- `apple_gpu_policy.mojo` (Task 05) *mirrors* eleven `gpu_tiling` constants
  by copy, with `test_apple_gpu_policy.mojo` pinning the mirrors. The values
  still agree after this lane's edits; nothing here changed a constant.
- `MAX_RESIDENT_BLOCKS_PER_CORE` (apple_gpu_policy, 8) and
  `TARGET_BLOCKS_PER_SM` (gpu_tiling, 8) are two names for one threshold in
  two modules.
- `gpu_leaf_batching.mojo` (Task 02) keeps its own `MAX_BINS` and grid bound.

Quarantined: nothing. No dead code was left behind in the owned files.

---

## 6. Fallbacks preserved (what is on by default: nothing new)

| Decision | Default | What it takes to move it |
| --- | --- | --- |
| Specialization level | `SPEC_LEVEL_BASELINE` -> geometry is `derive_tiling`'s | explicit level or `MOJOBOOST_GPU_HIST_SPECIALIZATION` |
| Strategy | `STRATEGY_AUTO` -> tiled above one tile, atomic at one | `MOJOBOOST_GPU_HIST_STRATEGY` |
| Class batching | **sequential** (`requested_batch` resolves to 1) | explicit `requested_batch` or `MOJOBOOST_GPU_CLASS_BATCH` |
| Bin sharing | off, because `per_block` is 1 at batch size 1 | a batch above 1 *and* an eligible level |
| Hybrid placement | `PLACE_GPU` for every leaf, `DECLINE_COSTS_UNMEASURED` | a `HybridCosts` with a citation, and `MOJOBOOST_HYBRID_LEAVES` |
| Reuse | `ReuseOffer.none()` unless a caller offers one; the sibling of a split is offered `ORIGIN_SUBTRACTED`, which is what the grower already does | nothing -- it states existing behavior |
| Level-wise growth | untouched by this lane; not enabled anywhere here | -- |

`plan_class_schedule` deliberately does **not** ask
`plan_class_batches` for the widest batch the budget admits. Batching changes
no number a round produces, but it changes the memory a fit holds and the
order work reaches the queue in, and neither has been measured against the
sequential loop on any device. An explicitly requested batch that does not
fit still raises rather than shrinking, which is `plan_class_batches`'s own
contract and is left to it.

No speedup is claimed anywhere in the four files. Every docstring that
mentions a cost says what it costs, not what it saves.

---

## 7. Cross-lane patch requests (exact)

All still requests except §7.2, which the user directed this lane to
implement; it is kept in place, rewritten as a record of what was done.

### 7.1 Task 01 -- make the plan the launch (`histogram_gpu.mojo`)

The builder already computes a plan (`histogram_plan`, line 815) and already
stores a tiling (`self.tiling`, lines 427 and 538). They are the same numbers
at the default level and can diverge only when a level was requested. Adopt
the plan as the launch geometry, keeping the baseline as the fallback:

```mojo
# histogram_gpu.mojo, after `self.tiling = derive_tiling(...)` at line 427
# and again after the re-derivation in `set_features` at line 538:
if self.spec_level > SPEC_LEVEL_BASELINE:
    # `HistogramPlan.tiling()` is the plan's geometry as the descriptor the
    # enqueues already take. At SPEC_LEVEL_BASELINE it equals what
    # derive_tiling just returned, so the guard is only to keep the default
    # path bit-identical without relying on that equality.
    var planned = self.histogram_plan(self.n_rows).tiling()
    if planned.partial_cells <= self.part_capacity:
        self.tiling = planned
```

Conservative on purpose: the partial buffer is allocated once from
`self.tiling.partial_cells`, so a re-derived geometry may only stay within
the buffer it already has. The plan is given `self.part_capacity` as
`max_partial_cells` and so should already satisfy this; the check is there
because "should" is not "does".

### 7.2 Task 01 -- let a batched gradient plane feed the grower (IMPLEMENTED)

**This section was a request and is now a change.** The user directed this
lane to implement it, so `histogram_gpu.mojo` and `train_gpu.mojo` -- Task
01's files -- were edited. That is outside the ownership stated at the top of
this handoff and is recorded here so Task 01 sees it rather than discovers
it. Nothing else in either file was touched; in particular the existing
sequential multiclass loop is byte-identical.

**The mechanism is not the one this section originally proposed.** The
request was pointer adoption -- `adopt_gradients` taking raw
`MutPointer[Float32, ...]` at the batch's slot offset. It was not
implemented, for two reasons found while writing it:

1. No struct in `src/mojoboost` stores a pointer field. Holding one would
   need `Optional[MutPointer[...]]` with an erased origin, a first-of-its-kind
   construct in this package, and unverifiable without a build this lane may
   not run.
2. The builder owns its gradient buffers for the whole session and every
   enqueue below reads them. Adopting a pointer into someone else's
   allocation makes every later build depend on that allocation outliving the
   builder -- a lifetime obligation the type system would not be carrying.

What was implemented instead is a device-to-device slot copy, which mirrors
the existing `fill_softmax_gradients_device` exactly and adds no new lifetime
rule. Three additions:

```mojo
# gpu_multiclass_batch.mojo (owned by this lane)
def _scatter_slot_kernel(...)          # one row per thread, plane -> plane
def GpuClassBatch.scatter_slot(
    mut self, slot: Int,
    mut grad_dst: DeviceBuffer[DType.float32],
    mut hess_dst: DeviceBuffer[DType.float32],
) raises

# histogram_gpu.mojo (Task 01's file)
def class_schedule(self, n_classes: Int, eligibility: BatchEligibility,
                   requested_batch: Int = 0) raises -> ClassSchedule
def fill_batched_gradients(mut self, mut batch: GpuClassBatch,
                           slot: Int) raises
```

`fill_batched_gradients` is the same four statements as
`fill_softmax_gradients_device`: the class's planes arrive in the builder's
own buffers (here by copy rather than by a softmax kernel), both fixed-point
scales are set from `batch.scale_of(slot)` / `batch.hess_scale_of(slot)`,
`has_gradients` is set, and `round_epoch` advances so the histogram cache
treats it as a new round's gradients. It refuses a batch whose `n_rows`
disagrees with the builder's. `scatter_slot` refuses before `fill_gradients`
and before `refresh_scales`, because a plane without the scale it was reduced
at cannot be quantized.

The copy costs one `Float32[n_rows]` read and write per class per round, on
the device, on the queue both already share. That is the price of not
introducing a pointer field; it is a stated cost, not a claimed saving.

**The round loop.** `train_gpu.mojo` gained
`_train_multiclass_gpu_batched`, called from `_train_multiclass_gpu_rounds`
immediately after `state.init_raw(builder.ctx, base_scores)` and before the
sequential loop:

```mojo
var schedule = builder.class_schedule(
    n_classes, BatchEligibility.deeper_node()
)
if not schedule.is_sequential():
    return _train_multiclass_gpu_batched(
        builder, life, state, n_classes, params,
        base_scores^, schedule, split_search,
    )
```

`deeper_node()` and not `round_root()` because this path shares no bin read:
the grower builds each class's histogram through the builder, one class at a
time. What the batch buys here is the batched gradient fill and one magnitude
readback per batch instead of `n_classes` of each.

Since `plan_class_schedule`'s default is sequential (§6), the guard is false
under the shipped configuration and the existing loop runs unchanged. The
batched helper is reachable only with `MOJOBOOST_GPU_CLASS_BATCH` set.

Ordering is preserved by construction: batches are contiguous ascending runs
and slots within a batch are ascending, so `trees` is appended in ascending
`k` and tree `(i, k)` still lands at `i * n_classes + k`. The helper mirrors
the sequential loop's bookkeeping rather than inventing its own -- same
`life.begin_round()`/`begin_tree()`/`end_tree()`/`end_round()` calls, same
`MulticlassRoundGuard` sequence (Task 01 had already wired the guard into
both existing loops, including `close_round` rather than `abandon_round` on
the no-progress path), and the same no-progress truncation and `break`.

What this still does **not** get is the shared bin read. Driving
`GpuClassBatch.enqueue_level_histogram` from the grower needs a grower that
consumes `batch.histograms(k_count)` for a whole frontier level; that is a
larger change and is still not requested here (§9 item 7).

### 7.3 Task 01 -- exports (`__init__.mojo`)

The package already re-exports part of `apple_histogram_policy` at line 210.
Requested additions, all pure host types and functions:

```mojo
from .apple_histogram_policy import (
    SPEC_LEVEL_BASELINE,
    SPEC_LEVEL_BATCHED,
    ClassSchedule,          # new
    HistogramPlan,          # new
    HistogramWorkload,
    batching_declined_reason,
    derive_histogram_plan,
    describe_plan,          # new
    describe_schedule,      # new
    env_specialization_level,
    plan_class_schedule,    # new
    plan_from_caps,         # new
)
from .gpu_tiling import (   # new block
    HistogramTiling,
    describe_tiling,
    launches_for_strategy,
)
from .hybrid_leaf_scheduler import (   # new block
    PLACE_GPU,
    HybridContext,
    Placement,
    ReuseOffer,
    describe_placement,
    place_leaf,
    place_leaf_with_reuse,
    plan_split,
)
```

`gpu_multiclass_batch` exports are deliberately not requested: its types own
device buffers, and nothing should reach them except a trainer that has a
`DeviceContext`.

### 7.4 Task 05 -- collapse the mirrors (`apple_gpu_policy.mojo`)

Its docstring says the mirrored constants "should collapse into imports at
integration" and `handoffs/apple_a6_policy.md` records the same. The values
still agree; this lane changed none of them. Requested:

```mojo
from .gpu_tiling import (
    BYTES_PER_PARTIAL_CELL,
    FALLBACK_MAX_THREADS_PER_BLOCK,
    FALLBACK_SHARED_MEMORY_PER_BLOCK,
    MAX_GRID_DIM_Y,
    MIN_ROWS_PER_TILE_BIN_FACTOR,
    MIN_ROWS_PER_TILE_THREAD_FACTOR,
    STRATEGY_ATOMIC,
    STRATEGY_AUTO,
    STRATEGY_TILED,
    TARGET_BLOCK_THREADS,
    WARP_GRANULARITY,
    clamp_block_threads,
)
```

with `FALLBACK_CORE_COUNT = FALLBACK_SM_COUNT`,
`PARTIAL_BUDGET_CEILING_BYTES = PARTIAL_BUDGET_BYTES`, and
`derive_block_threads(profile, n_rows)` ending in `clamp_block_threads`
rather than repeating the rounding. Note that
`MAX_RESIDENT_BLOCKS_PER_CORE` and `gpu_tiling.TARGET_BLOCKS_PER_SM` are the
same number under two names; if Task 05 keeps both, the relationship deserves
a comment, because `resolve_tiling` now takes the product
`cores * resident` from whichever layer is planning.

This touches `tests/parallel/test_apple_gpu_policy.mojo`, which asserts the
mirrors equal their sources; those assertions become trivially true and can
stay.

### 7.5 Task 02 -- one geometry per launch (`gpu_active_rows.mojo`, `gpu_leaf_batching.mojo`)

`gpu_active_rows` already imports `HistogramTiling` and `derive_tiling`. It
should accept a caller-supplied `HistogramTiling` wherever it re-derives one,
so a planned geometry (`HistogramPlan.tiling()`) reaches the launch instead
of being re-derived from `DeviceCaps` at the enqueue. `gpu_leaf_batching`'s
local `MAX_BINS` and grid bound should come from the same source as
everyone else's.

### 7.6 Documentation lane -- one line in `docs/design/HYBRID_TRAINING.md`

That document enumerates the placement outcomes. It now has a third,
`PLACE_REUSE`, and `SplitPlan` carries a placement for both children. This
lane may not edit docs outside this handoff.

---

## 8. Serialization and public API

**No serialization effect.** Nothing in the four files touches `Tree`,
`Model`, `MulticlassBooster`, `serialize.mojo`, or any on-disk format. The
multiclass tree slot is still `round * n_classes + k`
(`gpu_output_planes.round_tree_slot`, unchanged and consumed); the round
guard exists to enforce exactly that.

**No public Python or C API effect.** None of these modules is exported to
Python today, and this lane requested no export (§7.3 is a request to Task
01, not an edit). The §7.2 implementation adds no Python-visible surface
either: `class_schedule` and `fill_batched_gradients` are Mojo methods on a
type that is not bound, and `_train_multiclass_gpu_batched` is private.

**Native API changes in Task 01's files (§7.2), all additive:**

- `GpuHistogramBuilder` gained two methods, `class_schedule` and
  `fill_batched_gradients`. No existing method changed signature or body.
- `train_gpu.mojo` gained one private function,
  `_train_multiclass_gpu_batched`, and one early-return guard in
  `_train_multiclass_gpu_rounds` that is false under the shipped defaults.
- Four imports were added to `train_gpu.mojo` (`ClassSchedule`,
  `GpuClassBatch`, `MulticlassRoundGuard`, `BatchEligibility`;
  `GpuObjectiveState` was made explicit) and four to `histogram_gpu.mojo`.
  This makes `gpu_multiclass_batch` and `hybrid_leaf_scheduler`'s sibling
  `apple_histogram_policy` importers of the trainer's dependency set for the
  first time; the direction is still strictly upward and there is no cycle.

**Native API changes in the owned files, all additive except two:**

- `_batch_hist_shared_kernel` gained two trailing parameters (`slot_begin`,
  `write_counts`). It is a private kernel with one caller, both in this file.
- `SplitPlan` gained a field (`subtracted_placement`) and `plan_split` gained
  a defaulted parameter (`parent_origin`). `SplitPlan` is constructed only by
  `plan_split`, and neither has any caller in the repository.
- `N_DECLINE_REASONS` went 11 -> 12 with the new `REUSED_FRESH_HISTOGRAM`.
- `MulticlassRoundGuard` gained a field (`next_batch`), reset by
  `open_round`.
- `GpuClassBatch` gained a field (`caps`), set by both constructors.

Everything else is new names. No existing signature changed, so no unowned
importer of `gpu_tiling` (`gpu_active_rows`, `gpu_predict`, `gpu_sparse`,
`gpu_leaf_batching`, `histogram_gpu`, `train_gpu`, and four test files) sees
a difference.

---

## 9. Remaining disconnections

1. **The trainer calls the multiclass half, not the hybrid half.**
   `train_gpu.mojo` now reaches `apple_histogram_policy` and
   `gpu_multiclass_batch` through §7.2. `hybrid_leaf_scheduler` still has no
   importer anywhere.
2. **The launch still comes from `self.tiling`, not from the plan.** The
   ladder decides batching yes/no (histogram_gpu.mojo:1035) and, now, the
   class schedule. It still does not decide a launch. §7.1 is unapplied.
3. **Batched gradients now reach the grower, by copy.** §7.2 is
   implemented. What remains is that the copy exists at all: a grower that
   accumulated directly from the batch's slot would not need it, and that is
   the same level-wide grower item 7 asks for.
4. **The hybrid scheduler has no cost model and cannot get one here.**
   `HybridCosts.unmeasured()` declines every leaf, by design, and a measured
   model needs a benchmark this lane may not write or run. The reuse arm is
   the only part of the scheduler that is useful without one, because reuse
   is decided before the cost comparison.
5. **`MODE_REPLICA` still has no replica kernel.** Unchanged: this module
   says what the kernel must be and declines until a caller declares it
   verified.
6. **`ROWS_DEVICE_COPY` is still not expressible.** `enqueue_copy` copies a
   whole buffer, so a per-range row readback needs an API this project does
   not have. The snapshot design exists so nothing depends on it.
7. **Shared bin reads at a round's root need a level-wide grower.** §7.2's
   integration gets the batched gradient fill and the single magnitude
   readback; the shared histogram launch needs a grower that consumes a whole
   frontier level's histograms at once. This is why the batched path asks for
   `BatchEligibility.deeper_node()` rather than `round_root()`: it would be
   claiming a sharing it does not perform. `enqueue_level_histogram`,
   `enqueue_shared_groups`, and both batched kernels remain unreached.
8. **No emitter for `bench/apple/histogram_plan.json`.** That file is
   hand-derived and says so. This lane's refactor is argued to be
   numerically identical for `derive_tiling` and for
   `derive_histogram_plan` with no environment overrides set, so the file
   should still match -- but "argued" is the operative word, and the emitter
   named in the file's own `verification` block is what would settle it.

---

## 10. Risks

- **The fusion argument is arithmetic, not a test run.** If
  `floor(A/(b*c)) == floor(floor(A/c)/b)` is applied where one of the terms
  can be zero or negative, the two spellings diverge. `resolve_tiling`
  rejects nonpositive bounds outright, and `derive_tiling` clamps
  `target_blocks` to at least 1 so a hand-built `DeviceCaps(0, ...)` behaves
  as it did before, but this has not been executed.
- **`slot_begin` addressing is new kernel code.** The gradient read, the
  scale read, and both output planes were re-indexed by the absolute slot
  while threadgroup memory stays indexed by the group-local class. An error
  there would silently write one class's histogram into another's plane.
  Nothing exercises it: §7.2 landed without reaching the batched kernels (§9
  item 7), so this stays unreached.
- **This lane edited two files it does not own.** `histogram_gpu.mojo` and
  `train_gpu.mojo` are Task 01's, and the edits were made at the user's
  direction, not by agreement with that lane. If Task 01 has the same files
  open this is a merge conflict waiting to happen, and the resolution should
  favor Task 01's version of everything outside the two new methods, the new
  private helper, and the imports listed in §8.
- **`_scatter_slot_kernel` is new device code with no test.** It is the
  simplest kernel in the file -- one row per thread, two loads, two stores,
  bounds-checked -- but "simplest" is not "checked". An off-by-`n_rows` in
  the slot base would feed the grower a neighboring class's gradients, and
  the result would be a silently worse model rather than a crash. §11 lists
  what would catch it.
- **The copy is a per-class cost the sequential path does not pay.** Batching
  trades `n_classes` softmax gradient fills and magnitude readbacks for one
  per batch, and pays back one `Float32[n_rows]` device copy per class.
  Whether that trade is positive is a measurement nobody has taken, which is
  one more reason the schedule defaults to sequential.
- **The count plane is written by the first group only.** If a caller ever
  issues the groups of one batch across two zeroing passes, counts would be
  lost rather than doubled. `enqueue_shared_groups` owns both the zeroing and
  the loop precisely so a caller cannot split them, but the invariant is
  structural rather than checked.
- **`MOJOBOOST_GPU_HIST_STRATEGY` now reaches `SPEC_LEVEL_SHAPE`.** Any
  measurement taken at `shape` with the strategy pinned to `atomic` before
  this change was, in effect, taken at a different geometry than it recorded.
  No such measurement exists in the repository.
- **A wider batch is more device memory held for a whole fit.** The schedule
  defaults to sequential for that reason among others; a caller that raises
  it should read `ClassBatchPlan.bytes_per_batch` first.
- **`ReuseOffer` trusts the caller.** It checks freshness and origin
  compatibility, but it cannot check that the offered buffer really describes
  this node. `histogram_cache_policy.HistogramKey` is what makes that
  checkable, and a caller that offers reuse without consulting it is offering
  a claim this module cannot verify.

---

## 11. Smallest later validation, all UNRUN

Not executed by this lane. Listed smallest-first; each is one focused command.

```
UNRUN  pixi run mojo run tests/test_gpu_tiling.mojo
       The refactor's only existing guard: derive_tiling's geometry,
       strategy resolution, block widths, and the env overrides. Must pass
       unchanged -- this lane claims the numbers did not move.

UNRUN  pixi run mojo run tests/test_gpu_portability.mojo
       Second consumer of derive_tiling, at both ends of the device range.

UNRUN  pixi run mojo run tests/parallel/test_apple_gpu_policy.mojo
       Pins apple_gpu_policy's mirrored constants against gpu_tiling. This
       lane changed no constant, so it should be unaffected; run it before
       Task 05 collapses the mirrors (§7.4), not after.

UNRUN  a new focused test, not written here (tests are out of this lane's
       ownership), asserting for a spread of (caps, rows, features, bins):
         derive_histogram_plan(..., SPEC_LEVEL_BASELINE).tiling()
             == derive_tiling(caps, ...)                    field for field
       and, with MOJOBOOST_GPU_HIST_SPECIALIZATION unset:
         plan_class_schedule(...).batches.is_sequential()    is True
       Both are properties this lane asserts in prose and nowhere else.

UNRUN  the emitter named in bench/apple/histogram_plan.json's `verification`
       block, diffed against the hand-derived table. It is the only thing
       that would confirm the shape level's numbers survived the fusion.

UNRUN  pixi run mojo run tests/test_train_gpu.mojo  (or whichever existing
       file covers the multiclass GPU path -- this lane did not open the
       tests directory)
       The shipped defaults must be untouched by §7.2: plan_class_schedule
       returns sequential, the guard in _train_multiclass_gpu_rounds is
       false, and the existing loop runs. This is the regression check for
       editing two files this lane does not own.

UNRUN  the same file with MOJOBOOST_GPU_CLASS_BATCH=2, on a device, on a
       small softmax fit, compared tree-for-tree against the same fit with
       the variable unset. Equality is the claim §7.2 makes and does not
       check: the batched softmax kernel was read to have the same arithmetic
       in the same order as the sequential one, and the magnitude reduction
       the same grid and the same ascending Float64 host fold, so the scales
       and therefore the fixed-point histograms should be identical. That
       argument is what would be under test, along with _scatter_slot_kernel's
       slot base.

UNRUN  a bit-comparison of a batched shared histogram against the sequential
       one, at batch sizes 2, 4, and 8 with per_block 4, on hardware. This is
       what would check the slot_begin addressing and the once-only count
       plane. It needs a level-wide grower (§9 item 7) first, which §7.2 did
       not build, and it needs a device.
```
