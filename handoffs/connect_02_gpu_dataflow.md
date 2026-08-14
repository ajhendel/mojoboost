# Connect 02: the GPU dataflow contract

Owned files: `src/mojoboost/gpu_active_rows.mojo`,
`src/mojoboost/gpu_leaf_batching.mojo`, `src/mojoboost/gpu_frontier.mojo`,
`src/mojoboost/gpu_binned_layout.mojo`,
`src/mojoboost/gpu_histogram_specializations.mojo`, and this file.

Nothing outside that list was edited. Nothing was committed by this lane, and
nothing was run: no Mojo, pixi, build, test, benchmark, formatter, or network
command. Every claim below is from static reading. No correctness,
performance, parity, or hardware claim is made anywhere in it.

**Shared-checkout note.** Partway through this work another lane committed the
whole working tree as `860b1cf Integrate training and interoperability
subsystems`, which swept four of the five owned files into that commit while
they were mid-edit. The file contents are intact and complete; only
`gpu_leaf_batching.mojo` still shows as modified in `git status`. This lane did
not create that commit and did not stage anything.

---

## 1. Implementations found, before anything changed

| Capability | Implementations | State |
|---|---|---|
| Active rows / compact child ranges | `gpu_active_rows.GpuActiveRows` + `LeafRangeTable` + `RowRouting`; host reference `partition_range_host` | **Already connected.** `histogram_gpu.GpuHistogramBuilder` owns a `GpuActiveRows`, `begin_tree`/`apply_split`/`enqueue_leaf` all go through it, and `train_gpu` drives those. No duplicate row-assignment path survives. |
| Multi-leaf batching | `gpu_leaf_batching` (kernels, `plan_batch`, `HistogramSlotPool`, `GpuLeafBatcher`) | Kernels and planner complete; **no importer at all** at the start of this round. Task 01 has since begun wiring it into `histogram_gpu._build_leaves_batched` (see §6). |
| Frontier state | `gpu_frontier.LeafFrontier` (host-only) **vs** `train_gpu._GpuLeafState` **vs** `train_gpu._GpuRecordLeafState` | Three parallel frontier representations. `LeafFrontier` was imported by nothing but `gpu_leaf_batching` (for `LeafWorkItem`); the trainer runs on the two `List[_Gpu*LeafState]` forms. |
| Packed / blocked bin layouts | `gpu_binned_layout` (plans + cost model) over `gpu_bin_packing` (bit primitives) | **No importer at all.** Produced plans nothing consumed and nothing could refuse. |
| Histogram specializations | `gpu_histogram_specializations` | Imported only by `apple_histogram_policy`. Carried two hand-copied mirrors of `gpu_tiling` constants and a packed-window planner that silently assumed one byte per cell. |

Duplicated descriptors and utilities found across the five owned files:
`MAX_BINS` (three copies), `MAX_ROWS` (two), `BYTES_PER_PARTIAL_CELL` (three,
one under the name `BYTES_PER_HIST_SLOT`), `MAX_GRID_DIM_Y` (two),
`TARGET_BLOCKS_PER_SM` and `MIN_ROWS_PER_TILE_*` (two each), the three-plane
count (`N_PLANES` / `PLANES_PER_HISTOGRAM`), and `_ceil_div` (two, one of them
dead).

---

## 2. Call path, before

```
train_gpu.grow_tree_gpu
  -> GpuHistogramBuilder.begin_tree(bag)      -> GpuActiveRows.begin_tree
  -> GpuHistogramBuilder.build_leaf(node)     -> GpuActiveRows.enqueue_range_histogram
  -> host _search / GpuSplitSearcher
  -> GpuHistogramBuilder.apply_split(...)     -> GpuActiveRows.partition
  frontier kept as List[_GpuLeafState] inside the trainer

gpu_leaf_batching        : nothing calls it
gpu_frontier             : nothing but gpu_leaf_batching's LeafWorkItem import
gpu_binned_layout        : nothing calls it
gpu_histogram_specializations : apple_histogram_policy only
```

## 3. Call path, after this lane

The row path is unchanged and still authoritative. What is new is a seam the
central trainer can call, and a descriptor every other module now agrees on.

```
                     gpu_histogram_specializations        (leaf: no deps but gpu_tiling)
                        MAX_BINS, PLANES_PER_HISTOGRAM
                        BinStorageDescriptor              <-- the one bin-storage descriptor
                              ^                ^
                              |                |
        gpu_binned_layout ----+                +---- gpu_leaf_batching
        (produces descriptors)                       (admits batches on them)

        gpu_frontier  (host state: offsets, lengths, stats, completion, plane)
              ^                          ^
              |                          |
        gpu_active_rows ---------------- gpu_leaf_batching
        (device ranges + commit)          (device batching + slots)
```

Grower sequence, per tree:

```
n_active = rows.begin_tree_with(frontier, bag, params.num_leaves, class_index)
loop:
    admission = admit_frontier_batch(frontier, batcher.pool, storage, features)
    if admission.admitted():
        slots = assign_batch_slots(frontier, batcher.pool, admission, stamp)
        plan  = plan_frontier_batch(caps, frontier, admission, g32, h32,
                                    len(active), n_bins, strategy, part_cap)
        check_batch_covers_ranges(plan, frontier, admission)
        batcher.enqueue_frontier_batch(plan, bins_dev, rows, grad_dev, hess_dev)
        raw = batcher.download_slots(slots)          # one map, not one per leaf
        # per item: frontier.set_candidate(...), frontier.set_stats(...)
    else:
        # established path, unchanged:
        rows.enqueue_range_histogram(...) per node
    if frontier.is_complete(): break                 # FRONTIER_BUDGET_SPENT / _NO_CANDIDATE
    best = frontier.select_best()
    plan = frontier.plan_commit(best, signs, missing_bin[feature])
    _ = rows.apply_commit(frontier, bins_dev, plan)  # device partition + host move + recheck
```

---

## 4. Connections completed

**Active rows -> frontier (one state transition per split).**
`GpuActiveRows.begin_tree_with`, `.partition_commit`, `.apply_commit`, and
`.check_frontier` are new. `apply_commit(frontier, bins, plan)` runs the device
partition first, then moves the frontier, then re-checks the two tables against
each other, and returns the left count. `partition_commit` builds the
`RowRouting` itself from `plan.split` and `plan.missing_bin` through the
existing `RowRouting.from_split`, so the missing/categorical rule is not
restated a fourth time. `gpu_active_rows` now imports `CommitPlan` and
`LeafFrontier` (no cycle: `gpu_frontier` imports neither).

**Child histograms consume only the child's rows.** This was already true and
is unchanged: `enqueue_range_histogram` reads `rows[begin .. begin+count)` for
the node's own `LeafRange`. What is new is that the *batched* path is held to
the same thing: `GpuLeafBatcher.enqueue_frontier_batch` takes the
`GpuActiveRows` struct rather than a bare row pointer and refuses any item
window outside `[0, n_active)` — the bagged case, where a window inside the
buffer can still be outside the dataset the tree is being fit to.

**Frontier owns offsets, lengths, statistics, completion, and the class
index.** New on `gpu_frontier`:

- `LeafStats` (`sum_grad`, `sum_hess`, exact `count`) with `subtract`, plus
  `FrontierLeaf.stats` / `stats_known`, `LeafFrontier.set_stats`, `stats_of`,
  `stats_known`, and `set_sibling_stats` (the host half of the subtraction
  trick, next to `enqueue_subtract`'s device half).
- `LeafFrontier.max_leaves`, `budget_left()`, `status()`, `is_complete()`,
  and the `FRONTIER_*` codes. `status()` distinguishes *budget spent* from
  *nothing left to split* from *work still in flight*, which neither
  `grow_tree_gpu` nor `_grow_tree_gpu_device_search` can currently tell apart.
- `LeafFrontier.plane`, the multiclass index, set by `begin_tree` and stamped
  onto every `LeafWorkItem` by `work_items` (whose `plane` argument now
  defaults to -1, meaning "the frontier's own").
- `slot_of_node`, `batch_slots(max_items)`.
- `CommitPlan.missing_bin`, so the plan carries what the device partition needs.
  `plan_commit` takes a `missing_bin` argument (default -1) and stores -1 for a
  categorical split.
- `check_invariants` also holds every leaf's `stats.count` equal to its
  `row_count`.

**Leaf batching describes many jobs with no per-leaf allocation or
synchronization.** New on `gpu_leaf_batching`: `BATCH_*` admission codes,
`BatchAdmission`, `admit_frontier_batch`, `assign_batch_slots` (all-or-nothing,
with rollback), `release_batch_slots`, `uniform_scales`, `plan_frontier_batch`,
`check_batch_covers_ranges`, `batch_windows`, `batched_leaf_stats`,
`subtraction_stamp`, `HistogramSlotPool.stamp_of` / `slot_of_owner`,
`GpuLeafBatcher.enqueue_frontier_batch`, and `GpuLeafBatcher.download_slots`.
`download_slots` maps the output pool once for a whole batch; the existing
`download_slot` maps once per leaf, which gave back per-leaf synchronization on
the way home.

**Packed bins choose storage truthfully.** `BinStorageDescriptor` (declared in
`gpu_histogram_specializations`, produced only by
`BinLayoutPlan.storage_descriptor`) carries the storage class, the element bit
width, the block factor `G`, whether the buffer *is* `BinnedMatrix.bins`, and
two marker flags. Storage classes are `BIN_STORAGE_PACKED` (1-7 bits),
`BIN_STORAGE_U8`, `BIN_STORAGE_U16`, `BIN_STORAGE_WIDER`.

The truthful part is what it refuses:

- `storage_for_width` maps a width above 8 to `U16`/`WIDER` rather than
  rounding it down. Those classes are **unreachable from this repository's data
  representation** (`binning.fit_bins` caps `max_bins` at 256 and
  `BinnedMatrix.bins` is a `List[UInt8]`), and `check_shipping` says so in
  words instead of pretending otherwise.
- `check_shipping` refuses any storage the compiled kernels cannot read —
  every kernel indexes `bins[f * n_rows + r]` as one byte, so only
  `BIN_STORAGE_U8` with `block_features == 1` passes.
- Markers: `missing_markers_preserved`, `categorical_markers_preserved`, and
  `check_markers_preserved` refuse a width too narrow for a feature's missing
  bin or a categorical feature's highest category bin. `storage_descriptor`
  runs them before it reports anything, `candidate_plans` now runs both classes
  (it previously checked only the categorical one), and **marker loss is never
  falled back from** — `resolve_storage` falls back on an unserviceable storage
  class, never on a lost marker.

**Specializations consume the same descriptors.** New:
`plan_packed_window_for(storage, ...)`, which returns an unusable window with
`WINDOW_STORAGE_NOT_BYTES` for any layout the four-lane byte arithmetic does
not describe (that arithmetic was previously applied on the caller's word);
`features_admit(features, device, storage)`;
`BinStorageDescriptor.bin_capacity()` (bounded by the *storage* width, not only
by `n_bins`), `kernel_shared_bytes_per_block()`, and
`shipping_shared_bytes_per_block()`.

---

## 5. Duplicates fused or quarantined (owned files only)

Fused — one definition, imported everywhere else:

| Constant | Now defined in | Removed from |
|---|---|---|
| `MAX_BINS` | `gpu_histogram_specializations` | `gpu_active_rows`, `gpu_leaf_batching`; `gpu_binned_layout.LAYOUT_MAX_BINS` is now bound to it |
| `MAX_ROWS` | `gpu_active_rows` | `gpu_leaf_batching` |
| `BYTES_PER_PARTIAL_CELL` | `gpu_tiling` (pre-existing) | the mirror in `gpu_histogram_specializations`, the copy in `gpu_leaf_batching`; `gpu_binned_layout.BYTES_PER_HIST_SLOT` is now bound to it |
| `MAX_GRID_DIM_Y`, `TARGET_BLOCKS_PER_SM`, `MIN_ROWS_PER_TILE_BIN_FACTOR`, `MIN_ROWS_PER_TILE_THREAD_FACTOR` | `gpu_tiling` (pre-existing) | `gpu_leaf_batching` |
| three-plane count | `gpu_histogram_specializations.PLANES_PER_HISTOGRAM` | `gpu_leaf_batching.N_PLANES` is now bound to it |
| row window type | `gpu_active_rows.LeafRange` | `gpu_leaf_batching.batch_windows` returns it rather than a second begin/count pair |

Removed: the dead private `_ceil_div` in `gpu_histogram_specializations`
(unused, and `gpu_tiling`/`gpu_leaf_batching` each have one).

Not removed, deliberately: `gpu_leaf_batching._ceil_div` (used throughout its
planner), `gpu_binned_layout.check_categorical_widths` (public, still the
raising form callers may want on widths alone),
`gpu_active_rows._range_reduce_kernel` (a byte-for-byte copy of
`histogram_gpu._hist_reduce_kernel`, already documented in place as something
integration should delete; deleting it is a `histogram_gpu` change, so it stays
quarantined behind that comment).

Not fused, and why: `gpu_frontier` still declares nothing device-side and
imports no GPU module, so the frontier remains reasonable on a machine with no
accelerator. The dependency runs the other way — `gpu_active_rows` imports the
frontier.

---

## 6. Remaining disconnections

1. **`LeafFrontier` is still not the trainer's frontier.** `train_gpu` runs on
   `List[_GpuLeafState]` (host search) and `List[_GpuRecordLeafState]` (device
   search). Both reproduce, by hand, what `select_best`, `plan_commit`, and
   `apply_commit` do. Adopting the frontier is a `train_gpu.mojo` change and is
   Task 01's call; §7 gives the exact shape. Until then the completion state,
   the leaf statistics, and the multiclass plane exist but nothing reads them.
2. **`histogram_gpu._build_leaves_batched` still assembles its item list by
   hand**, from `LeafRange`s rather than from a `LeafFrontier`, because the
   trainer above it has no frontier to hand down (see item 1). It does now use
   `subtraction_stamp`, `uniform_scales`, `enqueue_frontier_batch`, and
   `download_slots`, so the launch and the download are both once-per-batch;
   what remains hand-rolled is only the slot acquisition, which
   `admit_frontier_batch`/`assign_batch_slots` would take over once a frontier
   exists.
3. **No layout other than the dense 8-bit one is uploadable.** The descriptor
   now says so clearly, but the packed and blocked kernels do not exist.
   `resolve_storage` falls back to the baseline, so this is a documented
   ceiling rather than a silent one.
4. **`gpu_binned_layout`'s cost model still returns `LAYOUT_UNDECIDED`**
   without measurements, by design. Nothing in this round added a threshold and
   nothing should until `bench/apple/bin_layout_plan.json` is filled in.
5. **Speculation is unused.** `speculative_order`, `verify_speculation`, and
   `SpeculationLedger` remain correct and uncalled. They are not needed for the
   batched path Task 01 is taking (a split's two children are already a batch of
   two) and are left alone rather than wired on speculation.

---

## 7. Exact cross-lane patch requests

All of these are in Task 01's files (`histogram_gpu.mojo`, `train_gpu.mojo`,
`gpu_split_search.mojo`). This lane did not make any of them.

**7.1 through 7.4 — already adopted by Task 01 during this round.** Recorded
here because they document what the seam is for, and because they are what a
reviewer should look for if any of it regresses. As of the last read of
`histogram_gpu.mojo`, `_build_leaves_batched` already:

- computes its slot stamp as `subtraction_stamp(self.round_epoch,
  self.feat_epoch)` rather than the undefined `self.batch_stamp` an earlier
  in-flight edit referenced (bounds: `feature_epoch < 2^24`,
  `round_index < 2^32`, both raising rather than aliasing);
- builds its per-item scales with `uniform_scales(take, g32, h32)`;
- launches through `enqueue_frontier_batch(plan, bins, self.rows, grad, hess)`,
  which validates every item window against the tree's live active prefix (the
  bagged case, where a window inside the buffer can still be outside the
  dataset the tree is fit to) and refuses a batcher built for another dataset;
- downloads a whole chunk with one `download_slots(slots)` mapping instead of
  one `download_slot` per leaf.

Nothing further is requested for those four.

**7.5 `histogram_gpu.mojo` — gate on the storage descriptor (outstanding).**
`GpuHistogramBuilder` uploads `data.bins` verbatim. To make that a checked
decision rather than an assumption, in `__init__` after the upload:

```mojo
        # The layout the kernels index, as a descriptor, checked once.
        self.storage = baseline_descriptor(data)      # gpu_binned_layout
        self.storage.check_shipping()
```

with `var storage: BinStorageDescriptor` on the struct. It is the dense-u8
descriptor, so it cannot fail today; it exists so a later packed or blocked
upload has to pass `check_shipping` to reach a kernel, and so
`admit_frontier_batch` has something real to gate on.

Once it is in place, `_build_leaves_batched`'s hand-rolled slot loop can become
`admit_frontier_batch` / `assign_batch_slots`, which is the last hand-rolled
part of that function — but only after 7.6, since admission takes a frontier.

**7.6 `train_gpu.mojo` — adopt the frontier (larger, outstanding).**
Replace `List[_GpuLeafState]` in `grow_tree_gpu` with a `LeafFrontier`:

- `builder.rows.begin_tree_with(frontier, bag, params.num_leaves, 0)` in place
  of `builder.begin_tree(bag)`;
- per node: `frontier.set_candidate(slot, LeafCandidate.ready(split, n_left,
  n_right, left_value, right_value, parent_value))` and
  `frontier.set_stats(slot, LeafStats(sum_g, sum_h, n_rows))`;
- loop condition `while not frontier.is_complete()` and report
  `frontier_status_name(frontier.status())` rather than falling out of a
  `while n_leaves < params.num_leaves`;
- commit: `var plan = frontier.plan_commit(best, signs, missing_bin)` then
  `_ = builder.rows.apply_commit(frontier, builder.bins_dev.unsafe_ptr(), plan)`
  in place of the hand-written clamp/child-bounds/`apply_split` block.

Node ids come out identical: `plan_commit` assigns `next_node` then
`next_node + 1`, and `apply_commit` overwrites the parent's slot with the left
child and appends the right, which is exactly `frontier[best_i] = ...;
frontier.append(...)`. `select_best` is the trainer's rule byte for byte
(strictly-greater, initial best gain 0.0, ascending scan). Tree layout is
therefore preserved by construction, not by inspection — but nothing has been
run to demonstrate it, so treat it as a claim to test, not a result.

**7.7 `train_gpu.mojo` / `gpu_split_search.mojo` — the device-search blocker.**
`_grow_tree_gpu_device_search` cannot batch because `GpuSplitSearcher.enqueue`
takes a whole `DeviceBuffer` with no word offset, so a pooled histogram slot
cannot be handed to it. The provider change is an `enqueue` overload taking
`(buffer, slot_index, slot_cells)` or a base offset. That is Task 03's file;
this lane records the need and does not touch it.

**7.8 `__init__.mojo` — optional surface, no change required.** Task 01 has
already added `BatchPlan`, `GpuLeafBatcher`, `plan_batch`, and
`slots_for_budget` from `gpu_leaf_batching` to the package surface; the other
four owned modules are not re-exported and nothing here needs them to be. If
the seam should be reachable from the package, the minimal further set is
`BinStorageDescriptor`, `baseline_descriptor`, `resolve_storage`,
`LeafFrontier`, `LeafStats`, and the `BATCH_*` codes. Adding an export is an
additive public-API change, so it is Task 01's call and not made here.

---

## 8. Fallbacks preserved

- The single-leaf launch path (`GpuActiveRows.enqueue_range_histogram`) is
  untouched and is what every `admit_frontier_batch` decline leaves the caller
  on. Five distinct decline reasons, all reported, none silent.
- `STRATEGY_ATOMIC` remains the strategy a batch resolves to when the partial
  buffer cannot hold the tiles it wanted (`VERDICT_PARTIAL_BOUND`), unchanged.
- `resolve_storage(..., allow_fallback=True)` returns the dense-u8 baseline for
  any plan the kernels cannot read. With `allow_fallback=False` it raises
  instead. Marker loss raises in both cases.
- `plan_packed_window` keeps its old signature and behaviour;
  `plan_packed_window_for` is the descriptor-gated form beside it, and
  `apple_histogram_policy`'s existing call is unaffected.
- `assign_batch_slots` is all-or-nothing: a pool that runs dry mid-batch gives
  back every slot the call took and restores every frontier assignment, so the
  caller drops to the single-leaf path with nothing leaked.
- `PackedLoadWindow`'s field list is unchanged, so
  `apple_histogram_policy`'s fieldwise construction
  (`PackedLoadWindow(False, rows, 0, 0, WINDOW_NOT_A_RUN)`) still compiles.
- `tests/parallel/test_gpu_active_rows.mojo` imports only `GpuActiveRows`,
  `LeafRange`, `LeafRangeTable`, `RowRouting`, `partition_range_host`; all five
  keep their signatures. `LeafFrontier.begin_tree(n)` keeps working — the two
  new arguments default.

## 9. Serialization and public API effects

**No serialization effect.** No model state was added, removed, or reshaped:
`Tree`, `SplitInfo`, `Histogram`, `BinnedMatrix`, and every serializer are
untouched. Bin ids are stored and never renumbered, so `missing_bin`,
`threshold_bin`, and the `CatBitset` masks keep their meanings and no
serialized model changes.

**Public API: additive only, and none of it added by this lane.** Task 01 has
re-exported `BatchPlan`, `GpuLeafBatcher`, `plan_batch`, and `slots_for_budget`
from `gpu_leaf_batching` through `mojoboost/__init__.mojo`; all four keep the
signatures they already had. Nothing else in this lane is reachable from the
package, the bindings, or Python, and no existing signature was changed in a
way an existing caller would notice: every new argument
(`begin_tree(n, max_leaves, plane)`, `plan_commit(slot, signs, missing_bin)`,
`work_items(slots, plane)`) has a default that reproduces the previous
behaviour.

The three struct field additions (`FrontierLeaf.stats`/`stats_known`,
`LeafFrontier.max_leaves`/`plane`, `CommitPlan.missing_bin`) are host-only
in-memory state with no on-disk form. `CommitPlan` is `@fieldwise_init`, so its
positional constructor gained a trailing argument; it is constructed in exactly
one place (`plan_commit`), which was updated.

## 10. Risks

1. **Unbuilt.** Nothing was compiled. The likeliest breakages are mechanical:
   `comptime X = <imported comptime>` bindings (`LAYOUT_MAX_BINS`,
   `BYTES_PER_HIST_SLOT`, `N_PLANES`) if Mojo requires a literal there, and
   `stack_allocation[MAX_BINS, ...]` in `gpu_active_rows` now that `MAX_BINS`
   is imported rather than local. Both revert to a local `comptime` in one line
   each if they fail.
2. **New import edges.** `gpu_active_rows -> gpu_frontier`,
   `gpu_active_rows -> gpu_histogram_specializations`,
   `gpu_binned_layout -> gpu_histogram_specializations`,
   `gpu_histogram_specializations -> gpu_tiling`,
   `gpu_leaf_batching -> gpu_active_rows`. Traced by hand as acyclic;
   `gpu_frontier` still imports nothing from this lane. A cycle would surface
   at the first build.
3. **`gpu_histogram_specializations` is no longer dependency-free.** It now
   imports `gpu_tiling`, which imports `max.gpu.host` and reads the
   environment. Every function in it is still pure Int/Bool arithmetic, and
   `apple_histogram_policy` (its only consumer) already imports `gpu_tiling`
   directly, so the transitive set is unchanged for that consumer.
4. **Concurrent lanes.** Task 01 is editing `histogram_gpu.mojo` and
   `train_gpu.mojo` right now, including its own batching glue. The §7 patches
   are written against what those files contained at the time of reading and
   may need rebasing.
5. **`check_frontier` costs a pass per commit.** Linear in the frontier (a few
   hundred entries) against a partition that is four kernel launches, so it
   should not matter, but it is a real cost and a caller that measures it as
   significant should call it per batch rather than per commit.
6. **`batched_leaf_stats` divides by the accumulation scales.** It is only
   correct for a slot accumulated under the scales passed to it, which is what
   the stamp is meant to guarantee. A caller that ignores the stamp gets wrong
   sums with no error.

## 11. Smallest later focused checks — ALL UNRUN

None of these were run. They are listed smallest first, and each is one command.

```
# 1. Does the lane still compile at all (cheapest possible signal)?
UNRUN: pixi run mojo build src/mojoboost/gpu_histogram_specializations.mojo

# 2. The one existing test that touches an owned file's public surface.
UNRUN: pixi run mojo test tests/parallel/test_gpu_active_rows.mojo

# 3. The policy module that consumes the specialization descriptors.
UNRUN: pixi run mojo test tests/parallel/test_apple_gpu_policy.mojo

# 4. The trainer path the active-row seam sits under.
UNRUN: pixi run mojo test tests/test_gpu_training.mojo
```

Later focused tests to write (none written this round, per the round's rule):

- `LeafRangeTable` and `LeafFrontier` agree after a scripted split sequence:
  `rows.apply_commit` then `rows.check_frontier(frontier)` over a handful of
  commits, including an empty child.
- `frontier.status()` returns `FRONTIER_BUDGET_SPENT` for a frontier at
  `max_leaves` with ready candidates, and `FRONTIER_NO_CANDIDATE` for one under
  `max_leaves` with none.
- `plan_commit(..., missing_bin=m)` puts `m` on the plan for a numerical split
  and -1 for a categorical one, and `RowRouting.from_split` on that plan routes
  the missing bin the way `Tree.goes_left` does.
- `storage_descriptor` raises on a width one bit too narrow for a feature's
  missing bin, and on one too narrow for a categorical feature's top category.
- `resolve_storage` returns the baseline for a packed plan and raises with
  `allow_fallback=False`.
- `plan_packed_window_for` returns `WINDOW_STORAGE_NOT_BYTES` for a 4-bit
  descriptor and matches `plan_packed_window` exactly for a dense-u8 one.
- `admit_frontier_batch` returns each of the five decline codes from a
  constructed frontier/pool/descriptor/features combination.
- `assign_batch_slots` on a pool one slot short leaves the pool's free count
  and every `hist_slot` exactly as it found them.
- `download_slots([a, b])` equals `download_slot(a)` followed by
  `download_slot(b)`, concatenated.
