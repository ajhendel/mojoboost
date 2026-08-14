# Algorithm 22: batched multi-leaf histogram construction

Status: designed and implemented in isolation. Nothing central was edited.
Nothing was built, run, tested, benchmarked, or profiled, per the lane's
instructions, so every performance statement below is a prediction with a
named way to refute it and not a result.

Files added by this lane:

- `src/mojoboost/gpu_frontier.mojo` (host side, no device)
- `src/mojoboost/gpu_leaf_batching.mojo` (kernels, buffers, planning)
- `bench/apple/leaf_batching_plan.json`
- `handoffs/algorithm_22_leaf_batching.md` (this file)

One command was run in this lane, a JSON syntax check on the plan file. No
Mojo, pixi, test, build, benchmark, or profiler was executed.

## 1. The negative result, first

**On the trainer's default path, batching has nothing to batch.**

`grow_tree_gpu` grows best-first. Each iteration picks the single best-gain
leaf, splits it, builds the *smaller* child's histogram on the device, and
derives the sibling by host-side `subtract_histogram`. That is one histogram
per commit. A batching primitive handed one item per launch does exactly what
the current code does, plus a table upload and a binary search.

So this lane is not a kernel optimization that can be dropped in. It is half
of a change to the **grower**, and the half that can be built and reasoned
about first. `gpu_frontier.leaves_per_launch` computes the number for each
grower the project has or could have:

| Grower | Leaves offered per launch | Available today |
| --- | --- | --- |
| `grow_tree_gpu` host search (the default) | 1 | yes |
| `_grow_tree_gpu_device_search` | 2 (both children are built) | yes |
| Speculative frontier (section 4) | up to `2 * depth`, capped by the frontier and the slot pool | needs the loop in section 6 |
| Level-wise growth (lane 24) | the whole level | separate lane |

Anyone measuring these kernels should measure `c6_feeder_reality` in the
benchmark plan before anything else, because every other number is an answer
to a question that only exists once a grower can offer more than one leaf.

## 2. What the two modules provide

### `gpu_frontier.mojo`, host side only

| Symbol | Role |
| --- | --- |
| `LeafCandidate` | a leaf's best split, its exact child counts, and its raw child values, with an explicit state so "not searched yet" is never read as a gain of zero |
| `FrontierLeaf` | node id, row window, depth, interaction branch, monotone bounds, histogram slot, partition flag |
| `LeafWorkItem` | the whole contract with the device half: a row window, an output slot, a gradient plane |
| `LeafFrontier` | the live leaves in the trainer's slot order, with `select_best`, `pending`, `plan_commit`, `apply_commit`, `check_invariants` |
| `CommitPlan` | everything one commit implies, computed before anything is mutated: node ids, both child windows, the clamped child values, the subtraction choice |
| `speculative_order`, `verify_speculation`, `SpeculationLedger` | the speculation machinery and its miss accounting |
| `search_is_order_free` | whether candidates, not just histograms, are order independent |
| `leaves_per_launch` | the table above, as arithmetic |

No `DeviceContext`, no buffer, no kernel, no environment read, so the whole
frontier story is reasonable and later testable on a machine with no
accelerator.

### `gpu_leaf_batching.mojo`, device side

| Symbol | Role |
| --- | --- |
| `_batch_hist_partial_kernel` | one (feature slot, packed tile) partial, per item scales and features |
| `_batch_hist_atomic_kernel` | the same accumulation folded straight into the item's output slice |
| `_batch_reduce_kernel` | each item's own tiles summed into its own slice, ascending |
| `_batch_zero_kernel` | zeroes only the slices this batch writes |
| `_subtract_slice_kernel` | `dst = parent - child` in the fixed-point buffer, aliasing safe |
| `plan_batch`, `BatchPlan`, `BatchItemPlan` | tile distribution, strategy resolution, the partial-buffer bound |
| `batch_cost`, `serial_cost`, `BatchCost`, `subtraction_saves_reads` | the symbolic cost model |
| `HistogramSlotPool`, `slot_bytes`, `pool_bytes`, `slots_for_budget` | the bounded device histogram pool and its subtraction stamps |
| `GpuLeafBatcher` | the buffers and the two entry points, `enqueue_batch` and `enqueue_subtract` |

### The packed tile axis

A batch's launch is two dimensional. `grid.x` is the active feature slot, as
today. `grid.y` is a flat concatenation of every item's row tiles, and a
threadgroup finds its item by binary searching the items' prefix-summed tile
offsets, once, on thread 0, behind the barrier the kernel already pays to
zero its shared histogram.

The alternative, a leaf axis on `grid.z`, would have to give every item the
largest item's tile count, so a frontier of one million-row leaf and thirty
400-row leaves would launch thirty times more threadgroups than it needs and
have them exit immediately. Highly unbalanced frontiers are the normal shape
of leaf-wise growth, not a corner, so the packed axis is the default here and
`BatchCost.idle_blocks` is carried, and is zero by construction, so that a
plan which stops being packed cannot hide it. The packed axis also keeps the
launch inside two grid dimensions, whose portable limits this project has
already established (`MAX_GRID_DIM_Y = 65535`); the portable limit on a third
is not something this project has established on Metal, CUDA, and HIP alike.

## 3. Semantic invariants

Each of these is a property the implementation is structured to make true,
with the reason it holds, not a rule someone has to remember.

1. **Batch size is invisible in the result.** Every item writes only its own
   output slice, no accumulation crosses an item, accumulation inside an item
   is the same fixed-point Int32 the single-leaf kernels use, and the tiled
   reduction sums an item's own tiles in ascending order. A leaf's histogram
   is therefore bit-identical whether built alone or in a batch of thirty.
   Test `c1_identical_histograms` asserts it word for word, not to a
   tolerance, because both sides are exact integers.
2. **Both strategies agree.** `STRATEGY_ATOMIC` and `STRATEGY_TILED` differ
   only in how partials combine, and integer addition is associative, so they
   produce the identical histogram. Same argument the single-leaf path
   already relies on.
3. **Best-first order is preserved.** `LeafFrontier.select_best` is the
   trainer's rule byte for byte: ascending slot scan, strict `>`, initial
   best gain 0.0, so ties go to the lowest slot and a nonpositive gain never
   splits anything. `apply_commit` puts the left child in the parent's slot
   and appends the right, which is the trainer's convention and therefore the
   tie order every later selection depends on. Node ids come out in the same
   order two `Tree._add_node` calls would assign them.
4. **Order independence.** A frontier leaf's rows are decided by the path
   from the root to it and not by the order the other leaves are split in,
   because a split rewrites only its own leaf's range of the active-row
   permutation. So a leaf's histogram, and the best split that histogram
   admits under a fixed feature set, are invariant to every other leaf's
   commit order. This is what makes batching construction legitimate at all.
5. **What is *not* order independent**, and therefore stays serial: node ids,
   and through them the per-node feature draw when
   `feature_fraction_bynode < 1.0` (`search_is_order_free` is the predicate);
   and which leaves get split at all, since `num_leaves` is a budget. Depth,
   the interaction branch and its allow mask, and the monotone bounds all
   descend from the ancestor chain, which order cannot change.
6. **A speculative partition is harmless.** It leaves the leaf's row *set*
   unchanged and only reorders rows inside its own range, and histograms are
   sums over the range. Further, the stable partition is idempotent: a second
   pass over an already partitioned range recomputes the same flags and
   stability keeps both sides in the order they are already in, so re-running
   it is the identity. The `partitioned` flag therefore exists to avoid
   paying twice and not to keep anything correct.
7. **Subtraction is exact and guarded.** Both operands are fixed-point Int32
   under the same scales, so the difference is exact, where the host path
   subtracts two already dequantized Float64 values and takes one rounding.
   Validity needs the same fixed-point scales and the same active feature
   set; `HistogramSlotPool` carries a caller-supplied stamp encoding both and
   `enqueue_subtract` refuses mismatched stamps. Within a tree the scales are
   constant and the feature set is fixed by `set_features`, so the stamp is a
   guard against wiring mistakes and against a batch that spans rounds or
   classes, which is exactly where it would silently corrupt.
8. **Distinct output slots.** Two items writing one slice is the one way a
   batch could corrupt a histogram, so `plan_batch` refuses a duplicate and
   `LeafFrontier.check_invariants` refuses one in the frontier.
9. **Empty leaves are legal.** A leaf with no rows is planned at one row, as
   `GpuActiveRows.range_tiling` already does. Its row loop runs zero
   iterations and the zeroing pass has already written the zeros that are its
   correct histogram.
10. **Categorical splits are untouched.** A categorical split is a routing
    decision, which lives in the partition and in `Tree.goes_left`, not in
    accumulation. A batched histogram of a leaf under a categorical parent is
    the same histogram; the categorical *search* over it is the split
    searcher's, unchanged.

## 4. Speculation, and why it is free

Best-first growth produces at most two new leaves per commit, so a batch
wider than two needs leaves whose commit has not happened yet. Invariant 4 is
what makes that admissible: their histograms are already determined.

The loop is:

1. Rank the ready candidates by gain, ties to the lower slot
   (`speculative_order`). This predicts the next `k` commits.
2. Enqueue, in one batch, the histograms of the children those `k` commits
   would create, and the partitions those commits need.
3. Commit serially. Before each commit, call `verify_speculation`, which asks
   the frontier as it now stands which leaf best-first would take and
   compares. Stop the speculative run at the first disagreement.

A disagreement costs nothing but time. The partitions already enqueued are
correct whenever their leaves are eventually committed (invariant 6). The
histograms already built stay valid until their leaves are split, so they are
wasted only if `num_leaves` cuts growth before the leaf is reached, which is
what `SpeculationLedger.wasted_histograms` counts.

With `feature_fraction_bynode == 1.0`, which is the default, the searches
batch too, because the search does not read the node id. Below 1.0 the
histograms still batch and the searches follow the commits.

The correctness claim is `c7_speculation_is_free` in the benchmark plan: the
fitted model is byte identical at every speculation depth. If that ever
fails, speculation is wrong and no timing of it means anything.

## 5. Memory bounds

With `F = n_features`, `B = n_bins`, `S = n_slots` at most `F`, `K` items in
a batch, `P` pool slots, `T` total tiles in a batch:

| Buffer | Bytes | Grows with |
| --- | --- | --- |
| output pool | `P * 3 * F * B * 4` | the frontier |
| partials | `T * 3 * S * B * 4` (`T * S * B * 12`) | the batch width |
| item table | `K * 10 * 4` | the batch width |
| feature table | `K * F * 4` | the batch width |

Worked, at `B = 256`:

| `F` | one slot | 31 slots | 255 slots |
| --- | --- | --- | --- |
| 50 | 150 KiB | 4.6 MiB | 37 MiB |
| 200 | 600 KiB | 18 MiB | 149 MiB |
| 1000 | 2.9 MiB | 91 MiB | 747 MiB |

The output pool is the term that has to be bounded, and it is: `P` is fixed
at construction, `slots_for_budget` sizes it against a byte budget, and a
leaf whose slot is reclaimed can always be rebuilt from its row range, so
eviction costs a histogram and never a wrong answer. A 255-leaf frontier over
1000 features cannot hold its histograms on the device and has to be told so
rather than allowed to allocate 747 MiB.

The partial buffer is the term that can make batching *worse*. It grows with
the batch, and the tiled strategy is what a wide batch wants. `plan_batch`
handles the collision in the open: if the tiles the batch wanted do not fit,
shares are scaled down proportionally with a floor of one tile each; if not
even the floors fit, the plan resolves to the atomic strategy, which
allocates nothing. Both outcomes set `VERDICT_PARTIAL_BOUND`, which is the
signal that the batch is paying for its width. At `S = 200`, `B = 256` and
128 tiles the partial buffer is 75 MiB, already past the 64 MiB
`PARTIAL_BUDGET_BYTES` the single-leaf path uses, so this is not a
hypothetical bound.

`T` is also capped by `MAX_GRID_DIM_Y = 65535`, and a plan that exceeds it
raises rather than truncating.

## 6. Integration seams

Every edit below is to a file this lane may not touch. They are ordered so
the tree stays buildable after each one.

### Seam 1: construct the batcher on the builder's context (`histogram_gpu.mojo`)

```mojo
from .gpu_leaf_batching import GpuLeafBatcher, slots_for_budget
```

and one field on `GpuHistogramBuilder`, initialized after `self.caps` and
`self.part_capacity` are resolved:

```mojo
    var batcher: GpuLeafBatcher
...
        self.batcher = GpuLeafBatcher(
            self.ctx,
            self.caps,
            data.n_rows,
            data.n_features,
            data.n_bins,
            slots_for_budget(pool_budget_bytes, data.n_features, data.n_bins),
            self.part_capacity,
        )
```

It must be the builder's own `DeviceContext`. Sharing the context is what
puts the batch's kernels on the same in-order queue as the gradient upload,
the objective kernels, and the partition, which is what orders them without a
fence. Constructing it on a private context would be a correctness bug, not a
performance one; see risk R5.

Nothing else in `histogram_gpu.mojo` changes. The batcher takes the binned
matrix, the active-row buffer, and the gradient planes as pointers
(`self.bins_dev.unsafe_ptr()`, `self.rows.rows_dev.unsafe_ptr()`,
`self.grad_dev.unsafe_ptr()`, `self.hess_dev.unsafe_ptr()`), so nothing is
uploaded or copied twice.

### Seam 2: the split searcher needs a histogram offset (`gpu_split_search.mojo`)

This is the one seam that needs a signature change, and it is small.
`GpuSplitSearcher.enqueue` takes a whole `DeviceBuffer` and scans it from
zero. A batched output pool holds `P` histograms in one buffer, so the
searcher has to be told which one.

```mojo
# _scan_slot_kernel: add a parameter and use it as the base.
    hist_offset: Int32,
...
    var base = Int(hist_offset) + f * nb
    # and every `hs` reference becomes `Int(hist_offset) + hs`

# _launch_search: pass it through.
    hist_offset: Int = 0,

# GpuSplitSearcher.enqueue: one more defaulted argument.
    def enqueue(..., record: Int = 0, hist_offset: Int = 0) raises:
```

With the default of 0 every existing call site is unchanged. A batched caller
passes `slot * 3 * n_features * n_bins`.

The searcher already has the rest of what a batched frontier needs:
`max_records` holds one record per leaf, `enqueue_pick_best` reduces a set of
records to the best-gain leaf with ties to the lower slot, and
`download_words` brings back every record in one synchronization. The
docstring in that module already names this as the next stage ("the host
downloads one record per tree level"); this lane is the histogram half of
that stage.

Its staging contract also has to be widened. Today the allow mask and the
parameter block are single pinned buffers reused per node, so one node's
`enqueue` must be followed by its `download` before the next node stages. A
batch of `K` leaves searched in one go needs `K` parameter slots and `K` allow
masks, indexed by record. That is a buffer resize and one index, not a kernel
change, but it must land before any batched search, not after.

### Seam 3: the batched grower (`train_gpu.mojo`)

A new `_grow_tree_gpu_batched`, alongside the two growers already there, and
a `SPLIT_SEARCH_*`-style constant to select it. Shape:

```mojo
    var frontier = LeafFrontier()
    frontier.begin_tree(n_root)
    # root: one item, which is the same launch it is today
    while frontier.n_leaves < params.num_leaves:
        var order = speculative_order(frontier, depth)
        var slots = frontier.pending()
        if len(slots) > 0:
            var items = frontier.work_items(slots)   # after acquiring pool slots
            var plan = plan_batch(caps, items, scales, n_slots, n_bins, ...)
            batcher.enqueue_batch(plan, bins, rows, grad, hess)
            # one search launch per item, or one batched search after seam 2
            # one download for the whole batch
        var best = frontier.select_best()
        if best < 0:
            break
        var commit = frontier.plan_commit(best, signs)
        # device: partition the parent, then either build the smaller child
        # and enqueue_subtract the sibling, or batch both children
        frontier.apply_commit(commit)
```

The `Tree` is built from the `CommitPlan`s, which carry the node ids, both
child windows, the clamped values, and the split. Nothing in `tree.mojo`
changes.

### Seam 4: the range table and the frontier must not diverge

`LeafFrontier` and `GpuActiveRows.LeafRangeTable` hold the same windows,
indexed differently (frontier slot against node id). Holding both is
deliberate: it makes a disagreement detectable rather than silent, and
`LeafFrontier.check_invariants` is the host mirror of
`LeafRangeTable.check_invariants`. The integration should call both under
`MOJOBOOST_GPU_VERIFY_ROWS=1` and neither in a normal run.

### Seam 5: the mirrored constants collapse (`gpu_tiling.mojo`)

`gpu_leaf_batching.mojo` imports `DeviceCaps`, `derive_block_threads`, and
the `STRATEGY_*` constants, so a strategy int means one thing across the
backend. It mirrors five numeric constants (`MAX_BINS`, `MAX_GRID_DIM_Y`,
`BYTES_PER_PARTIAL_CELL`, `TARGET_BLOCKS_PER_SM`, `MIN_ROWS_PER_TILE_*`)
rather than importing them, following the pattern `apple_gpu_policy.mojo` and
`gpu_histogram_specializations.mojo` established for landing alongside
concurrent work on that file. They are marked in the source and should
collapse into imports at integration.

The per-item tile distribution is deliberately *not* `derive_tiling`. A batch
shares one tile budget across several leaves of very different sizes, which
is a shape `derive_tiling` has no way to express: it resolves one
(rows, features, bins) triple.

### Seam 6: overlap with `gpu_histogram_specializations.plan_leaf_batches`

Lane 14 owns a host-side greedy planner that groups *consecutive* leaves into
launches with a shared `grid.z` and reports the threadgroups that would exit
immediately (`LeafBatch.wasted_blocks`). This lane's `plan_batch` solves the
same planning problem with a packed tile axis, which has no wasted
threadgroups to report, and adds the strategy resolution, the partial-buffer
bound, and the per-item scales and feature sets the kernels here read.

They should not both ship. The recommendation is that `plan_batch` supersedes
`plan_leaf_batches`, and that `batching_pays` and `total_wasted_blocks` from
that module are kept as the comparison the `c3_packed_tile_axis` benchmark
needs: they model the padded alternative this lane rejected, so they are
exactly the baseline for measuring whether rejecting it was right. Whoever
integrates the two should read this paragraph and lane 14's handoff together.

### Seam 7: runtime bookkeeping (`gpu_runtime.mojo`)

Two new pool slots and one new resource for the hazard tracker:

```mojo
comptime SLOT_BATCH_OUT = 11
comptime SLOT_BATCH_ITEMS = 12
comptime RES_BATCH_OUT = 11
```

Bookkeeping only, exactly as the existing slots are.

## 7. Race and ordering risks

**R1. Staging buffer reuse.** The item table, the scales, and the feature
table are pinned host buffers reused by every batch, and the host write is
not ordered against a copy the device already holds. So a batch's copies must
complete before the next batch stages over them: the caller synchronizes, or
downloads a result, between two batches. Stated rather than enforced, for the
same reason `GpuSplitSearcher.enqueue` states it, that enforcing it with a
synchronization here would serialize the pipelining a batch exists to buy.
Two batches in flight need a staging ring; `gpu_runtime.StagingRing` already
models one.

**R2. Two items, one output slot.** Refused by `plan_batch` and by
`LeafFrontier.check_invariants`. It is the only way a batch could corrupt a
histogram, so it is checked in both places rather than either.

**R3. Subtraction aliasing.** `_subtract_slice_kernel` gives each thread one
cell of each of the three slices at the same offset, so `dst` may alias
either operand and deriving in place over the parent's slot is legal. What is
*not* legal is aliasing `dst` onto a slot some other live leaf owns;
`HistogramSlotPool.owner` is the guard, and a caller must `release` before it
reuses a slot.

**R4. Subtraction across stamps.** Fixed-point scales change per round and
per class, and the active feature set changes per tree. Subtracting across
either produces a silently wrong histogram, so the pool carries a stamp and
`check_subtractable` refuses a mismatch.

**R5. The batcher must share the builder's context.** The batch reads the
gradient planes the objective kernels write and the active-row buffer the
partition kernels write. Ordering between them comes entirely from the single
in-order queue. A batcher constructed on its own `DeviceContext` has no
ordering against either and would read half-written gradients. There is no
overload here that opens a private context, and there should never be one.

**R6. Partition against an in-flight histogram of the same range.** A
partition rewrites the row ids inside a leaf's own range. A histogram kernel
reading that same range concurrently could see one row twice and miss
another, which *would* change the histogram, so the harmlessness argument in
invariant 6 covers a partition that has completed and not one that is running.
Today the in-order queue serializes them. Any future work that puts
histograms and partitions on separate streams needs an explicit event between
a leaf's partition and any histogram over its range or its children's.

**R7. Eviction under flight.** Reclaiming a slot whose batch has not
completed lets the next batch overwrite a histogram that is still being read.
The pool is host-side bookkeeping and cannot see the queue, so eviction must
happen at a point the caller knows is drained, which in practice is after the
batch's records or histograms have been downloaded.

**R8. Determinism.** Nothing here uses a float atomic, a warp shuffle, a
subgroup width, or an allocation order that depends on timing. The slot
allocator hands out the lowest free slot rather than the most recently freed,
so the device buffer's contents are a function of the tree and not of the
allocator's history. Tile order in the reduction is ascending. The batched
path is therefore bit-deterministic run to run, which is the property the GPU
backend already guarantees.

## 8. Future focused cases

- **Per-node feature narrowing in accumulation.** `set_item_features` exists
  and is unused. Per-node sampling narrows the search today, not the
  accumulation, so nothing needs it yet; a grower that wanted to accumulate
  only a node's sampled features can take it without a kernel change, because
  the kernels already read the feature table per item. Every item in a batch
  must list the same *number* of features, which per-node sampling already
  guarantees since the count comes from the tree's set and the fraction.
- **Multiclass output planes.** Each item carries a gradient plane index and
  its own fixed-point scales, which is what a class needs, because
  `fill_softmax_gradients_device` sets a different scale per class. What a
  cross-class batch additionally needs is `C` gradient planes resident
  (`grad` sized `C * n_rows`) and `C` row permutations, since the per-class
  trees partition rows differently. The item's `row_begin` is absolute into
  whatever row buffer is bound, so several permutations in one buffer need no
  extra field. Scheduling classes into batches is lane 26's; the seam is
  these two fields and this paragraph.
- **Sparse and EFB layouts.** The row loop reads a dense `bins[f * n_rows + r]`
  matrix. A CSC or bundled layout changes that inner loop and nothing about
  the batch structure, the item table, the reduction, or the planner.
- **A device-resident frontier.** `enqueue_pick_best` already reduces a set of
  split records to the best-gain leaf on the device. Combined with a batched
  search over the pool, the host would download one record per *batch*
  instead of one per node, and eventually only the finished tree. The record
  layout needs no change for that, which is why it carries child statistics
  and leaf values.
- **Two batches in flight.** Needs the staging ring of R1 and a second item
  table, and would overlap one batch's kernels with the next batch's upload.
  Worth doing only if a profile shows the upload is visible, which at
  `K * 10 * 4` bytes plus `K * F * 4` bytes it very likely is not.
- **Bin-capacity specialization.** The shared histogram arrays here are
  `MAX_BINS` wide unconditionally, exactly like the shipping kernels, so lane
  14's comptime bin-capacity parameter applies to these kernels identically
  and independently.

## 9. Decisive benchmark shapes

`bench/apple/leaf_batching_plan.json` is the plan, with seven claims, the
counters each needs, four frontier profiles, and the shape axes. In short:

1. `c6_feeder_reality` first: leaves per launch under each grower. Everything
   else is conditional on this being greater than one.
2. `c1_identical_histograms` and `c7_speculation_is_free`: correctness, word
   for word and model byte for byte. If either fails, stop.
3. `c2_occupancy_gain`: batched against serial on the `p1_balanced`,
   `p2_one_huge_many_tiny`, and `p3_all_tiny` frontier profiles, with the
   planner's verdict recorded next to every timing. The prediction is that
   batching wins exactly where the verdict is `occupancy_gain` and not where
   it is `single_fills`. **No crossover threshold is asserted anywhere in
   this lane**, and none should be added until this row is filled in on real
   hardware.
4. `c3_packed_tile_axis`: the packed launch against a padded one whose
   `grid.y` is the largest item's tile count, on the unbalanced profile. This
   is the measurement that justifies the binary search.
5. `c5_partial_buffer_bound`: sweep batch width at fixed partial capacity and
   find the width where the verdict flips to `partial_bound`.
6. `c4_device_subtraction`: the subtract kernel against download plus host
   subtract, per commit.

The axes that matter most are `features` (it decides how much of the device
one leaf already fills, so a 1000-feature dataset may never need batching at
all) and `min_data_in_leaf` with `num_leaves` (they decide how small the tail
leaves get).

## 10. What would refute this lane

- Every grower that can offer more than one leaf per launch turns out to cost
  more elsewhere than batching saves. Then these kernels are correct and
  useless, and that is the result.
- The binary search or the per-item table lookups cost more than the launches
  they save on small frontiers. `BatchCost` predicts they do not; a profile
  decides.
- On Apple silicon the launch overhead this lane removes turns out to be
  small next to the row work, in which case batching helps only the
  `p3_all_tiny` tail and the honest recommendation is to batch only there.
- The partial buffer bound bites before the occupancy gain arrives, meaning
  every batch wide enough to fill the device is forced to the atomic
  strategy. That is measurable directly with `c5_partial_buffer_bound` and
  would mean the tiled strategy and batching do not compose.

## 11. Known limitations

- **Nothing is wired in.** No central file was edited, no test was written,
  and no code path in the shipping trainer reaches any of it.
- **Nothing was compiled.** These modules have not been through `mojo build`.
  Whoever integrates should expect to fix syntax before anything else, and
  should read section 6 before writing a test.
- **The batched search is not implemented here.** It belongs to
  `gpu_split_search.mojo`, which this lane may not edit; seam 2 is the exact
  change.
- **`max_items` is 32 by default.** That bounds the binary search and the
  staging cost. It is not a claim about where batching stops paying, and the
  benchmark plan sweeps past it only by raising the constant.
- **The verdict is not a policy.** `plan_batch` reports three structural facts
  and an honest unknown. Nothing in this lane chooses to batch or not to
  batch, and nothing should until section 9 has been run.
