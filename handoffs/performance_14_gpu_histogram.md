# Task 14: Apple-aware GPU histogram specialization and launch policy

Status. Written by static reasoning only. **Nothing in this lane was
compiled, run, tested, benchmarked, or profiled**, which the task forbade.
No performance claim is made anywhere below. Every specialization this lane
describes is off by default, and the reason it is off is that no measurement
supports turning it on.

Files this lane owns and changed.

- `src/mojoboost/gpu_histogram_specializations.mojo` (new)
- `src/mojoboost/apple_histogram_policy.mojo` (new)
- `bench/apple/histogram_plan.json` (new)
- `src/mojoboost/gpu_tiling.mojo` (additive only, see section 9)
- `handoffs/performance_14_gpu_histogram.md` (this file)

This lane committed nothing. A concurrent integration commit in the shared
checkout (`b04b5f0`, "Integrate parallel release and accelerator work") swept
these files up mid-edit, so most of them are already in history without this
lane having staged them; a few later cleanups to the two new modules are
still unstaged. Nothing in that commit came from this lane's intent, and
nothing here was reviewed as part of it.

Files this lane read and did not touch: `histogram_gpu.mojo`,
`gpu_active_rows.mojo`, `train_gpu.mojo`, `apple_gpu_policy.mojo`,
`device.mojo`, `parallel.mojo`, `binning.mojo`, all of `tests/`, all of
`python/`, `bindings/`, `docs/`, `packaging/`, and every existing benchmark
driver. Every edit those need is written out verbatim in sections 8 and 9.

---

## 1. The central claim, stated once

The shipping histogram launch is `gpu_tiling.derive_tiling`. This lane did
not change it, did not wrap it, and did not put anything in front of it. It
added a layer above it that can plan a specialized launch, whose default
answer on every device and every shape is to call `derive_tiling` and return
what `derive_tiling` returned.

That is not caution for its own sake. There is exactly one end to end GPU
training measurement in this repository (M4, `bench/bench_train_gpu.mojo`),
it is of the unspecialized path, and it is slower than the CPU trainer,
which is why `device.mojo` still resolves `auto` to the CPU. A specialized
histogram kernel that has never been compiled cannot be known to be faster
than one that has never been fast. So the ladder below ships switched off,
with the evidence each rung needs written down in section 7.

What the lane does deliver, in order of how much it is worth:

1. A named discrepancy that is a fact about the current code, not a
   hypothesis. The shipping kernels allocate 3072 bytes of threadgroup
   memory per block at every bin count, while the tiling model computes
   `n_bins * 12`. Any residency estimate derived from the model is
   optimistic by up to a factor of 8/5 on a 16 KiB device. Section 4.
2. A policy layer whose every input is a reported device property or a
   workload shape, with no branch on a model string, an architecture string,
   or an M-series generation anywhere in it.
3. Conservative primitives for three specializations, each behind an
   explicit capability boundary with a portable implementation beside it.
4. A plan table (`bench/apple/histogram_plan.json`) showing what the policy
   selects for fourteen (device, shape) pairs, hand derived and marked as
   such, with the exact emitter that must confirm it.

---

## 2. Policy inputs, every one of them

`derive_histogram_plan(profile, device, features, work, requested_strategy,
requested_level, max_partial_cells)`.

### 2.1 `profile: apple_gpu_policy.GpuProfile`

Reused rather than redefined. This lane does not own `apple_gpu_policy.mojo`
and did not edit it.

| field | source | how a missing value is handled |
| --- | --- | --- |
| `core_count` | `MULTIPROCESSOR_COUNT` | `FALLBACK_CORE_COUNT` (16), applied by `from_reported`, low rather than typical |
| `max_threads_per_block` | `MAX_THREADS_PER_BLOCK` | `FALLBACK_MAX_THREADS_PER_BLOCK` (1024) |
| `max_shared_memory_per_block` | `MAX_SHARED_MEMORY_PER_BLOCK` | `FALLBACK_SHARED_MEMORY_PER_BLOCK` (16384) |
| `memory_budget_bytes` | nothing queries it anywhere in this repo | 0, meaning unreported, which sends `partial_budget_bytes` to the flat 64 MiB ceiling |
| `unified_memory` | true exactly on Metal | false |
| `api`, `apple_generation`, `synthetic` | parsed or set by the constructor | **no decision reads any of the three** |

`profile_from_caps(caps)` builds one from a `DeviceCaps`, which is what every
call site in this repository actually holds. It is exact today: the two
fields it cannot fill (`memory_budget_bytes`, `unified_memory`) interact only
with each other, and a zero budget makes the unified fraction inert.

### 2.2 `device: gpu_histogram_specializations.DeviceHistogramCapabilities`

The capability boundary. `portable()` answers no to everything.

| field | meaning | portable value |
| --- | --- | --- |
| `subgroup_width` | lanes in lockstep | **0, meaning unknown**. Nothing divides by it. Metal rejects `WARP_SIZE`, and `WARP_GRANULARITY` in `gpu_tiling.mojo` is a launch rounding chosen to be a multiple of every backend's width, not a claim about any device's width |
| `max_grid_dim_z` | bound on a batched launch's third dimension | 65535, the smallest portable bound |
| `wide_byte_loads` | one aligned four byte bin load has been **shown** to beat four one byte loads | false |
| `unified_memory` | budget shared with the host | false |

### 2.3 `features: gpu_histogram_specializations.KernelFeatures`

Which specialized kernel variants are compiled in. Deliberately separate from
device capability, because a device can be perfectly able to run a batched
launch while the batched kernel does not exist. `none()` is the truth about
this build: all three false.

`specialized_bin_kernels`, `packed_bin_loads`, `batched_leaf_kernel`.

### 2.4 `work: HistogramWorkload`

| field | meaning |
| --- | --- |
| `dataset_rows` | rows in the binned matrix, which is also the column stride: the bin of row `r` of feature `f` is at `f * dataset_rows + r` |
| `node_rows` | rows this node owns. Zero is legal and plans at one row, matching `GpuActiveRows.range_tiling` |
| `n_slots` | active features, which is `grid.x`, narrowed by feature subsampling |
| `n_bins` | bins including the missing bin |
| `first_row` | row id at the start of the node's run, meaningful only with the flag below |
| `rows_are_contiguous_run` | **the caller's assertion**, never inferred |

The run flag deserves its own paragraph. The row loop reads
`bins[f * n_rows + r]` where `r` comes from the active row permutation, so it
is a gather. It is contiguous exactly when the node's slice of the
permutation holds consecutive ascending row ids, which is a property of the
permutation and cannot be derived from any shape. A false answer costs only
the packed path. A wrongly true answer would make the kernel read the wrong
rows and silently produce a wrong histogram, which is why nothing in this
lane guesses it and why `HistogramWorkload.node(...)` defaults it off.

### 2.5 `requested_strategy`, `requested_level`, `max_partial_cells`

- `requested_strategy`: `STRATEGY_AUTO`, `STRATEGY_ATOMIC`, `STRATEGY_TILED`,
  with the same meaning `derive_tiling` gives them.
- `requested_level`: `SPEC_LEVEL_UNSET` (consult the environment, then
  default to baseline), or an explicit rung. **An explicit level is a ceiling,
  not a floor.**
- `max_partial_cells`: an already allocated partial buffer. It narrows the
  memory bound rather than replacing it, so a plan can never exceed either
  the buffer or the budget. `derive_tiling` replaces; this narrows. The two
  agree whenever the buffer was sized under the budget, which is how it comes
  to exist.
- Environment: `MOJOBOOST_GPU_HIST_SPECIALIZATION` in
  `off | baseline | shape | packed | batched`. Unset, empty, and unrecognized
  all mean baseline. There is deliberately **no `auto`**: a level that turned
  itself on when it liked the shape would be exactly the unmeasured default
  this module exists to refuse. `MOJOBOOST_GPU_ROW_TILE` and
  `MOJOBOOST_GPU_BLOCK_THREADS` are honored at every level so a benchmark can
  pin a geometry and vary only the specialization.

---

## 3. Policy outputs, every one of them

`HistogramPlan`.

| field | meaning |
| --- | --- |
| `level_requested` | the rung asked for, after the environment and clamping |
| `level_applied` | the rung reached, a **contiguous** ladder position and a statement about **this node** |
| `reason` | `REASON_AS_REQUESTED`, or which check stopped it |
| `strategy` | `STRATEGY_ATOMIC` or `STRATEGY_TILED`, never `AUTO` on return |
| `block_threads` | threads per threadgroup |
| `n_tiles` | `grid.y` |
| `rows_per_tile` | rows one threadgroup scans |
| `rows_per_thread` | `ceil(rows_per_tile / block_threads)`, **derived and reported, not chosen** |
| `bin_capacity` | 16, 32, 64, 128, or 256 |
| `shared_bytes_per_block` | the footprint of the kernel that will actually run in this build |
| `resident_blocks_per_core` | blocks the plan expects resident |
| `partial_cell_limit` | ceiling on `partial_cells` from the budget or the allocated buffer |
| `partial_cells` | `n_tiles * n_slots * n_bins`, or 0 on the atomic strategy |
| `packed_loads` | whether the packed row loop was selected |
| `packed_window` | the head, body, and tail split, with its own reason code |
| `baseline` | the `HistogramTiling` `derive_tiling` would have produced, always |
| `matches_baseline()` | whether the four launch fields equal the baseline's |

`baseline` is carried on every plan on purpose. It makes "identical at
baseline" checkable rather than asserted, it lets a benchmark report both
without re-deriving, and it gives a call site somewhere to fall back to if a
specialization misbehaves in the field.

Batching outputs are separate, because batching is a decision across nodes
and a `HistogramPlan` is about one node: `plan_batched_leaves(...)` returns
`List[LeafBatch]`, empty meaning launch exactly as today, and
`batching_declined_reason(...)` says why. Each `LeafBatch` carries `first`,
`n_leaves`, `rows_per_tile`, `n_tiles`, `min_rows`, `max_rows`, `blocks`, and
`wasted_blocks`.

`wasted_blocks` is reported rather than hidden. Every leaf in a batch shares
one `grid.y`, sized for the largest, so a small batch mate contributes
threadgroups that exit immediately. That is the cost side of batching and it
is the number a benchmark has to weigh the saved launches against.

---

## 4. The threadgroup memory discrepancy

This is the one finding in the lane that is a fact about the current tree
rather than a proposal.

`gpu_active_rows._range_hist_atomic_kernel` and `_range_hist_partial_kernel`
each make three `stack_allocation[MAX_BINS, Scalar[DType.int32],
AddressSpace.SHARED]` calls. That is `3 * 256 * 4 = 3072` bytes per block, at
every bin count, because `MAX_BINS` is a compile time constant and `n_bins`
is a runtime argument.

`gpu_tiling.shared_bytes_for(n_bins)` returns `n_bins * 12`, which is the
footprint of a kernel sized to its bin count. The two agree only at 256 bins.

Consequences, all of them checkable by reading:

- **The `derive_tiling` guard is optimistic.** It asks whether
  `n_bins * 12` fits the advertised threadgroup memory. A device advertising
  less than 3072 bytes would pass the guard and then fail to launch. No
  supported backend advertises that little, so this is a latent trap rather
  than a live bug, and this lane documented it in `shared_bytes_for` instead
  of tightening the guard, because tightening it changes which shapes
  `derive_tiling` accepts and this lane does not change that.
- **Any residency estimate from the model is optimistic.**
  `apple_gpu_policy.resident_blocks_per_core` divides the advertised
  threadgroup memory by `n_bins * 12`. On a 16 KiB device at 63 bins that is
  `16384 / 756 = 21`, capped to 8. The truth with the shipping kernel is
  `16384 / 3072 = 5`. The policy in this lane uses `kernel_block_bytes`,
  which returns the real footprint of whichever kernel is compiled in, so it
  gets 5 and plans 3 tiles where the model would have planned 4.
- **Closing the gap needs no new device primitive.** A capacity parameter on
  the existing kernels is the whole change. It is the cheapest of the three
  specializations and the only one that touches no device capability.

`shared_bytes_unmodeled(n_bins)` names the gap: 2880 bytes at 16 bins, 2688
at 32, 2304 at 64, 1536 at 128, 0 at 256.

---

## 5. Memory bound calculations, worked

All arithmetic is Int floor division. `BYTES_PER_PARTIAL_CELL` is 12: an
Int32 gradient, an Int32 hessian, and an Int32 count per (tile, feature, bin)
cell.

### 5.1 Partial buffer

```
hist_cells        = n_slots * n_bins
partial_cells     = n_tiles * hist_cells          (tiled strategy only)
partial_bytes     = 12 * partial_cells
partial_cell_limit(baseline) = 67108864 / 12 = 5592405
partial_cell_limit(shape)    = partial_budget_bytes(profile) / 12
tiles_by_memory   = partial_cell_limit / hist_cells
```

`partial_budget_bytes` is the reported device budget divided by 32 when
memory is unified and by 16 when it is not, capped by the same 64 MiB
ceiling, and it returns the ceiling outright when the budget is unreported.

**The fraction is inert on any real machine, and this is worth knowing before
anyone builds on it.** The fraction only binds when
`budget / divisor < 64 MiB`, that is below 2 GiB of reported budget on
unified memory and below 1 GiB on discrete. No device this project targets is
that small, and no device reports a budget at all today, so both limits are
the flat 5592405 cells everywhere, including in the whole plan table.

**The memory bound almost never binds either.** For it to reduce the tile
count below the occupancy bound of 3 or 4, a shape needs
`hist_cells > 5592405 / 4`, that is more than 1.4 million (feature, bin)
cells, which is roughly 5400 features at 256 bins. Every shape in the plan
table has `tiles_by_memory` between 682 and 10922 against a `wanted` of 2 to
4. **Occupancy is what sets the tile count in practice, not memory.** Anyone
tuning the partial budget is tuning a constraint that is not currently
binding.

### 5.2 Threadgroup memory and residency

```
unspecialized block bytes = 3 * 4 * 256      = 3072
specialized   block bytes = 3 * 4 * capacity = 12 * capacity
resident_blocks_per_core  = clamp(max_shared_per_block / block_bytes, 1, 8)
target_blocks             = core_count * resident_blocks_per_core
tiles_by_occupancy        = ceil(target_blocks / n_slots)
```

At 16384 bytes advertised: unspecialized gives 5 at every bin count;
specialized gives 8, 8, 8, 8, 5 across the five classes. At 32768: 8
everywhere, both ways, because the cap of 8 binds first. **The bin capacity
specialization changes no launch at all on a 32 KiB device.** That is the
honest headline of the plan table, and it is why the evidence gate in
section 7.1 asks for a device where the cap does not bind.

### 5.3 Rows per tile

```
min_rows_per_tile = max(8 * capacity, 4 * block_threads)
tiles_by_rows     = ceil(node_rows / min_rows_per_tile)
n_tiles           = min(tiles_by_occupancy, tiles_by_rows, tiles_by_memory)
rows_per_tile     = ceil(node_rows / n_tiles)
n_tiles           = ceil(node_rows / rows_per_tile)      (re-derived)
```

The baseline uses `8 * n_bins` where the shape level uses `8 * capacity`.
The substitution is the same one the footprint makes, and it is nearly inert:
at 256 threads the thread bound is 1024, which dominates the bin bound for
every capacity up to 128. Only capacity 256 (2048) exceeds it. The two
differ at all only when `n_bins` is not a power of two, and then by less than
a factor of two.

The re-derivation at the end is what guarantees the last tile is never empty
and that `n_tiles * rows_per_tile >= node_rows` with
`(n_tiles - 1) * rows_per_tile < node_rows`, which is the invariant
`tests/test_gpu_tiling.mojo::_assert_covers_rows` already pins for the
baseline and which any new level must satisfy identically.

### 5.4 Batched launches

```
batch blocks       = n_leaves * n_slots * n_tiles
batch partial cells= n_leaves * n_tiles * hist_cells
max_batch_leaves   = min(32, max_grid_dim_z,
                         partial_cell_limit / partial_cells)   [tiled]
                   = min(32, max_grid_dim_z)                   [atomic]
```

The partial buffer bound is the one that is easy to miss. A batched tiled
launch needs one partial histogram per (leaf, tile), so the budget that
bounded `n_tiles` now has to cover `n_leaves * n_tiles` of them. At the plan
table's largest tiled shape (24576 partial cells) the limit of 5592405 allows
227 leaves, so 32 binds first, but a wider dataset flips that and
`batching_declined_reason` returns `REASON_PARTIAL_BUDGET` when fewer than
two leaves fit.

---

## 6. Fallback order, exactly

Applied in this order, and every step down is recorded in
`HistogramPlan.reason`.

1. A requested level above what `KernelFeatures` says is compiled in falls to
   the highest level that is. `REASON_KERNEL_ABSENT`.
2. `SPEC_LEVEL_PACKED` falls to `SPEC_LEVEL_SHAPE` when the device has not
   reported that wide loads pay (`REASON_DEVICE_UNPROVEN`), when the node's
   rows are not a contiguous run (`REASON_ROWS_NOT_A_RUN`), when the column
   stride is not a multiple of 4 (`REASON_WINDOW_UNUSABLE`, window reason
   `column_stride_unaligned`), or when the aligned body is shorter than 4
   packed loads (`REASON_WINDOW_UNUSABLE`, window reason `body_too_short`).
3. `SPEC_LEVEL_SHAPE` falls to `SPEC_LEVEL_BASELINE` when a block of the
   compiled kernel does not fit the advertised threadgroup memory.
   `REASON_SHARED_MEMORY`.
4. `SPEC_LEVEL_BASELINE` is `derive_tiling`, which brings its own fallbacks:
   the tiled strategy falls to the atomic one whenever more than one tile
   will not fit the partial budget, and every device attribute the query
   refuses falls to a conservative portable constant rather than failing.

Two consequences worth stating because they are easy to get wrong.

- `level_applied` is contiguous, so `SPEC_LEVEL_BATCHED` is only reported
  when the packed level was also reached **for that node**. Batching itself
  does not depend on that: `plan_batched_leaves` gates on `level_requested`
  and `KernelFeatures`, so a frontier does not lose its batching because one
  node's rows happened not to be contiguous.
- The stride check is a whole launch property, not a per feature one. Column
  `f` begins at `f * dataset_rows`, so an odd stride gives every feature a
  different alignment and one window cannot describe the launch. Per feature
  windows are possible and were deliberately not built, see section 10.

---

## 7. Evidence required before enabling each specialization

None of these gates has been met. Each is stated as what a later validation
pass must produce, not as what anyone expects it to find.

### 7.1 Bin capacity specialization (`specialized_bin_kernels`)

1. **Bit exactness.** The specialized kernel and the shipping kernel produce
   byte identical `out_dev` contents for all five classes, at `n_bins` both
   equal to and below the capacity (16 and 15, 32 and 31, and so on), on both
   strategies, for a node with rows and a node with none. This is the gate
   that catches the loop bound error described in section 8.1, and it must
   run before any timing.
2. **A device where the residency cap does not bind.** On a 32 KiB device the
   specialization changes no launch at all (section 5.2), so timing it there
   measures nothing. The measurement needs a device advertising 16 KiB, or a
   bin count and capacity where 8 blocks are not already reachable.
3. **A histogram build timing** on at least one Metal and one CUDA or HIP
   device, at every class, against the shipping kernel, with the run recorded
   under `bench/apple/` protocol rules.
4. **A regenerated `bench/apple/histogram_plan.json`** from a real emitter,
   diffed against the hand derived table in this lane. A difference is a bug
   in one of the two.

### 7.2 Packed bin loads (`packed_bin_loads` plus `wide_byte_loads`)

1. **Bit exactness** against the scalar loop, including the head and tail,
   at run lengths that are not multiples of 4 and at `first_row` values in
   every residue class mod 4.
2. **A measurement of how often the run flag is even true.** This is the gate
   that decides whether the specialization is worth building at all. The
   permutation is only known to be an ascending run at the root of an
   unbagged tree. After the first split the order preserving partition keeps
   each child's rows in the parent's order, which is ascending, but a child's
   rows are a subsequence of the parent's, not a contiguous run of row ids
   unless the split happened to be on a prefix. **Instrument a real training
   run and count the fraction of nodes whose rows are a run.** If it is only
   the root, the specialization is worthless and should be dropped rather
   than measured.
3. **A microbenchmark of the load itself** on the target device, comparing
   one aligned four byte load against four one byte loads, which is exactly
   what `wide_byte_loads` claims. Until that exists the flag stays false and
   `pack4_bins` runs the four scalar loads, so the packed path is not
   expected to be faster and must not be described as such.

### 7.3 Batched small leaves (`batched_leaf_kernel`)

1. **Bit exactness** between a batched frontier and the same frontier
   launched one leaf at a time, per leaf.
2. **A frontier size distribution from a real training run.** The whole
   payoff depends on how many nodes are too small to fill the device, and
   nobody has measured that. Record the row count of every node at every
   depth for a representative dataset, then compute how many launches the
   batch plan would save and how many wasted blocks it would create. Both
   numbers come out of `plan_leaf_batches` without any device.
3. **Launch overhead per node**, measured, so the saved launches can be
   priced. `bench/bench_histogram.mojo` already times `enqueue_leaf`
   separately, which is the right place.
4. **The partial buffer growth accounted for** (section 5.4), including the
   reallocation policy, since `GpuHistogramBuilder.part_capacity` is fixed at
   construction and a batched tiled launch wants `n_leaves` times more.

---

## 8. Narrow call site instructions for Task 07

Task 07 owns `histogram_gpu.mojo`, `gpu_active_rows.mojo`, `train_gpu.mojo`,
and `__init__.mojo`. **This lane edited none of them.** Everything below is a
proposal for that lane, ordered so each step is independently revertable and
none of them changes behavior until the last one is explicitly asked for.

### 8.0 Exports (`src/mojoboost/__init__.mojo`)

```mojo
from .gpu_histogram_specializations import (
    DeviceHistogramCapabilities,
    KernelFeatures,
    LeafBatch,
    PackedLoadWindow,
    bin_capacity_for,
    kernel_shared_bytes,
    plan_leaf_batches,
    plan_packed_window,
)
from .apple_histogram_policy import (
    SPEC_LEVEL_BASELINE,
    SPEC_LEVEL_BATCHED,
    SPEC_LEVEL_PACKED,
    SPEC_LEVEL_SHAPE,
    HistogramPlan,
    HistogramWorkload,
    derive_histogram_plan,
    describe_plan,
    plan_batched_leaves,
    profile_from_caps,
)
```

### 8.1 The kernel change the bin capacity specialization needs

In `gpu_active_rows.mojo`, both histogram kernels. **This is the entire
change**, and the danger in it is one line.

```mojo
# was, in both _range_hist_atomic_kernel and _range_hist_partial_kernel
var sg = stack_allocation[
    MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
]()

# becomes, with the kernel parameterized on capacity
def _range_hist_atomic_kernel[capacity: Int](...):
    var sg = stack_allocation[
        capacity, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
```

**The loop bounds must stay `nb` and must not become `capacity`.** The zero
loop, the accumulate loop, and the flush loop all run `while b < nb`. A flush
loop running to `capacity` would write `out_hist[f * nb + b]` for `b >= nb`,
which lands inside the next feature's slice and corrupts a histogram silently
rather than crashing. `capacity` bounds the allocation and nothing else.
Section 7.1's first gate exists to catch exactly this.

Dispatch is a five arm comptime ladder at the launch site, keyed on
`plan.bin_capacity`:

```mojo
if plan.bin_capacity == 16:
    self.ctx.enqueue_function[_range_hist_atomic_kernel[16]](...)
elif plan.bin_capacity == 32:
    ...
```

Five instantiations per kernel is the trade against 256 of them; see
`bin_capacity_for`.

### 8.2 Consuming a plan in `GpuHistogramBuilder.enqueue_leaf`

`enqueue_leaf` currently calls `self.rows.range_tiling(...)`. The narrow
change keeps that call as the baseline and adds the plan beside it:

```mojo
# in GpuHistogramBuilder, new fields
var profile: GpuProfile          # profile_from_caps(caps), set in __init__
var device_caps: DeviceHistogramCapabilities   # .portable()
var features: KernelFeatures                   # .none()
var spec_level: Int                            # SPEC_LEVEL_UNSET

# in enqueue_leaf, replacing the range_tiling call
var window = self.rows.range_of(leaf)
var work = HistogramWorkload.node(
    self.n_rows, window.count(), n_slots, self.n_bins
)
var plan = derive_histogram_plan(
    self.profile,
    self.device_caps,
    self.features,
    work,
    self.tiling.strategy,
    self.spec_level,
    self.part_capacity,
)
var tiling = HistogramTiling(
    plan.strategy,
    plan.block_threads,
    plan.n_tiles,
    plan.rows_per_tile,
    plan.partial_cells,
)
```

With `spec_level` at `SPEC_LEVEL_UNSET` and no environment variable set,
`plan.strategy`, `plan.block_threads`, `plan.n_tiles`, and
`plan.rows_per_tile` are `derive_tiling`'s, because that is where the
baseline branch gets them. `plan.matches_baseline()` is true, and it is worth
asserting that in the first integration commit so a later change to the
policy cannot silently move the default path.

Note the one difference from `range_tiling`: that method plans an empty node
at one row internally, and `derive_histogram_plan` does the same, so an empty
node produces the same launchable geometry either way.

### 8.3 Setting the run flag, if and only if 7.2's second gate passes

`HistogramWorkload.node(...)` sets `rows_are_contiguous_run` false. Only
`GpuActiveRows` can know better. The cheapest honest source is a per node
flag maintained by `begin_tree` and `partition`:

- `begin_tree` with no bag writes the identity permutation, so the root's
  range is a run with `first_row = 0`.
- `begin_tree` with a bag stages arbitrary row ids, so the root is **not** a
  run unless the bag is checked, which costs an `n_rows` host pass. Do not
  check it by default.
- `partition` is order preserving, so a child of a run is ascending but is a
  run of row ids only when its rows were a prefix or a suffix of the parent's.
  Deriving that on the device costs a comparison the partition already has
  the data for, but it is new device work and must not be added before 7.2's
  second gate says the flag is ever true often enough to matter.

Until then, pass `False` and let the packed level decline with
`REASON_ROWS_NOT_A_RUN`. That is the correct answer, not a limitation.

### 8.4 Batching, last

`plan_batched_leaves` takes the frontier's row counts in launch order and
returns the batches. The grower would call it once per depth, after the
splits at that depth are applied and before the next round of
`enqueue_leaf` calls. An empty list means launch as today. The kernel side is
a `grid.z` dimension and a per leaf `(begin, count, out_offset)` descriptor
buffer; nothing else in the accumulation changes, which is what keeps the
batched result bit identical.

---

## 9. What this lane changed in `gpu_tiling.mojo`, and why it is safe

Additive only. `tests/test_gpu_tiling.mojo` pins `derive_tiling`,
`derive_block_threads`, `env_strategy`, `strategy_name`, and
`shared_bytes_for` behavior, and none of those changed.

| change | kind |
| --- | --- |
| module docstring paragraph on where specialization lives | comment |
| `shared_bytes_for` docstring: the model versus the shipping kernels' real 3072 byte footprint, and the guard's optimism | comment |
| `rows_per_thread(tiling)` | new function |
| `describe_tiling(tiling)` | new function |

No constant changed value, no signature changed, no branch changed. The two
new functions are pure and have no callers inside the module.

---

## 10. Required future correctness cases

This lane wrote no tests, which the task forbade. These are the cases a later
validation pass must cover, and the first five are correctness rather than
performance.

1. **Loop bound versus allocation bound.** A specialized kernel at capacity
   128 with `n_bins = 100` must write exactly 100 cells per feature. Assert
   the histogram of a 100 bin dataset is byte identical to the shipping
   kernel's, and separately that cells 100 through 127 of each feature's
   slice are untouched. Section 8.1.
2. **Every level produces the same histogram.** For a fixed dataset and a
   fixed node, baseline, shape, packed, and batched must produce byte
   identical `out_dev` contents. Integer accumulation is associative and no
   level changes the quantization, so this is a hard equality, not a
   tolerance. Any level that fails it is wrong, not merely slower.
3. **Tile coverage at every level.** The invariant
   `_assert_covers_rows` already pins for `derive_tiling`
   (`n_tiles * rows_per_tile >= n_rows` and
   `(n_tiles - 1) * rows_per_tile < n_rows`) must hold for
   `HistogramPlan` at every level and every shape in a sweep.
4. **Empty and single row nodes.** `node_rows = 0` must plan at one row and
   produce a launchable geometry with `n_tiles >= 1`. `node_rows = 1` with a
   256 thread target must not produce `rows_per_thread = 0`.
5. **Packed window residues.** `plan_packed_window` at `first_row % 4` in
   {0, 1, 2, 3} and counts spanning the `MIN_PACKED_BODY_QUADS` boundary must
   satisfy `window.covered() == count` in every case, usable or not. This is
   the property that keeps a caller from dropping rows.
6. **Batch plan coverage.** `plan_leaf_batches` must cover every input leaf
   exactly once, in order, with `sum(n_leaves) == len(row_counts)`, for
   frontiers including zero counts, single leaves, and counts spanning more
   than `BATCH_ROW_SPREAD`.
7. **Batch bounds.** No batch may exceed `max_batch_leaves`, and on the tiled
   strategy `n_leaves * partial_cells` must not exceed `partial_cell_limit`.
8. **Default equals baseline.** With no environment variable and
   `SPEC_LEVEL_UNSET`, `derive_histogram_plan(...).matches_baseline()` must be
   true for every profile and shape in a sweep. This is the test that keeps
   the default path from drifting, and it is the most important one in the
   list.
9. **Environment parsing.** Each of `off`, `baseline`, `shape`, `packed`,
   `batched`, an unrecognized string, and unset, resolving as documented.
10. **A device too small for the kernel.** A profile advertising less than
    3072 bytes of threadgroup memory currently passes `derive_tiling`'s guard
    and would fail at launch. Decide whether that should raise, and where.
    This lane documented it rather than changing the guard.

---

## 11. Profiler questions

Every one of these is unanswered. They are written as questions rather than
as expectations because this lane ran no profiler.

On Metal (Xcode GPU capture, Instruments):

1. **What actually limits occupancy in the histogram kernel?** Threadgroup
   memory, registers, or neither? Section 4 assumes threadgroup memory, which
   is what makes the bin capacity specialization worth anything. If registers
   bind first, the specialization changes nothing and should be dropped.
2. **How expensive is the shared memory atomic on a hot bin?** The kernel
   does three `Atomic.fetch_add` into threadgroup memory per row. On a skewed
   feature most rows land in the same bin, so this is the serialization point
   that a privatized or per lane accumulator would address, and that is a
   different specialization from any of the three here.
3. **What fraction of the kernel is the gather?** `bins[col + r]` with `r`
   from a permutation. If the gather dominates, the packed load matters and
   so does the run flag frequency; if it does not, section 7.2 can be closed
   without building anything.
4. **What is the actual per launch overhead**, so the batching payoff in 7.3
   can be priced against a measured number rather than an assumed one.
5. **Does the zero and flush of the partial histogram show up at all** at 16
   bins, where it is `2 * 16 / block_threads` iterations against
   `rows_per_tile / block_threads`? The `MIN_ROWS_PER_TILE_BIN_FACTOR` of 8
   exists to bound that ratio and has never been checked against a profile.

On CUDA (Nsight Compute) and HIP (rocprof), the same five, plus:

6. **Does the tiled strategy actually beat the atomic one** on a device with
   fast global atomics? The AUTO rule prefers tiled at more than one tile on
   every backend, and that rule predates any measurement on any of them.

---

## 12. Review notes on the delivered code

- `gpu_histogram_specializations.mojo` mirrors two constants from
  `gpu_tiling.mojo` (`MAX_BINS`, `BYTES_PER_PARTIAL_CELL`) so the primitives
  layer stays free of the `max.gpu.host` import and can be reasoned about
  device side. This follows the pattern `apple_gpu_policy.mojo` established
  and should collapse into imports at integration, at which point a mirror
  equality test like that module's belongs with it. **This lane wrote no such
  test**, so the mirrors are currently unpinned.
- `apple_histogram_policy.mojo` imports `gpu_tiling`, which imports
  `max.gpu.host`. `apple_gpu_policy.mojo` deliberately avoids that so it can
  be exercised anywhere. This module cannot, because calling `derive_tiling`
  for the baseline is exactly how the "identical by construction" guarantee
  is obtained, and that guarantee is worth more than the import freedom. The
  same import is already made by `tests/test_gpu_tiling.mojo` on CPU only
  runners.
- The shape level reimplements `derive_tiling`'s tile arithmetic rather than
  calling it, because two of its four bounds change. The duplication is real
  and is the most likely place for the two to drift. Correctness case 8 is
  what would catch a drift in the default path; a drift in the shape path
  would only be caught by correctness case 3.
- `pack4_bins` is a portable four load pack. It is **not expected to be
  faster** than the scalar loop it replaces, and nothing in this lane claims
  it is. It exists so the packing and unpacking arithmetic can be validated
  for bit exactness before a wide load is introduced behind
  `wide_byte_loads`. Writing the wide load itself was deliberately declined:
  this lane could not compile or measure it, and an unexercised device side
  load in the histogram path is the worst possible thing to leave behind.
- Per feature packed windows were declined for the same reason. The stride
  alignment requirement rejects any dataset whose row count is not a multiple
  of 4, which is most of them, and lifting it means one window per feature
  and a per feature head and tail in the kernel. That is a larger change and
  it should not be built before 7.2's second gate.
- If rows are a contiguous run, the gradient and hessian loads `grad[r]` and
  `hess[r]` are contiguous too, so they could be vectorized alongside the
  bins. This lane did not build it. The quantization
  `Int32(round(g * g_scale))` must stay per row whatever happens, since that
  is what makes the accumulation reproducible.
- `bench/apple/histogram_plan.json` is hand derived and says so in its
  `provenance` block. It is 14 of the 20 shapes in the natural cross product;
  the 6 omitted are the middle bin classes at 4096 rows, which behave exactly
  like the 16 bin case at that row count. Confirm the whole table with a real
  emitter before building on any row of it.

---

## 13. Relationship to the parallel lanes, checked read only

- **Task 07 (`integration_07_apple_gpu.md`).** That lane did not change
  `histogram_gpu.mojo` or `gpu_active_rows.mojo`, so every call site in
  section 8 is still where this lane found it. Its table of "who decides
  what" lists histogram accumulation as decided by the device capability
  policy through `MOJOBOOST_GPU_HIST_STRATEGY`, which this lane leaves
  untouched: `MOJOBOOST_GPU_HIST_SPECIALIZATION` is a second, independent
  switch, and setting neither is the default both lanes ship.
- **A6 (`apple_gpu_policy.mojo`, frozen this round).** Consumed, not
  duplicated. `GpuProfile`, `partial_budget_bytes`, `derive_block_threads`,
  and `MAX_RESIDENT_BLOCKS_PER_CORE` are imported from it. Its own
  `resident_blocks_per_core` divides by the modeled footprint; this lane's
  divides by the real one, which is the only divergence between the two and
  section 4 is why. A6's step 3 follow up (thread a real memory budget in) is
  still open, and section 5.1 shows the budget fraction is inert until it
  lands and would remain inert on any machine with more than 2 GiB.
- **Task 20 (native device policy).** Chooses whether to use the GPU at all.
  This lane chooses how to launch once that choice is made, and it introduces
  no crossover threshold, no size heuristic, and no `auto` that resolves to a
  device. `apple_gpu_policy.CrossoverInputs.min_cells` stays
  `CROSSOVER_DISABLED` and nothing here reads it.
- **A8 (`bench/apple/suite.py`, `schema.json`).** `histogram_plan.json` is
  deliberately **not** a run record and must never be merged with one. It has
  no `measurements`, and its `measurements` key is explicitly null with a
  reason, following that schema's rule that the absence of a number is a fact
  the record states rather than a zero.
- **Task 16 (sparse and categorical GPU).** The categorical routing in
  `RowRouting` reads a bitset rather than a threshold, but the histogram
  kernels do not branch on it: they accumulate by bin either way. Nothing in
  this lane's specializations interacts with categorical splits. A sparse bin
  representation would change the gather and would invalidate the packed
  window arithmetic, which assumes one byte per (row, feature).

---

## 14. Validation commands for the coordinated pass

None of these was run. They are listed so the later pass does not have to
reconstruct them.

```
pixi run mojo build -I src src/mojoboost/gpu_histogram_specializations.mojo
pixi run mojo build -I src src/mojoboost/apple_histogram_policy.mojo
pixi run mojo build -I src src/mojoboost/gpu_tiling.mojo
pixi run mojo run -I src tests/test_gpu_tiling.mojo
pixi run mojo run -I src tests/test_gpu_strategies.mojo
pixi run mojo run -I src tests/parallel/test_apple_gpu_policy.mojo
```

The first three are the compile check this lane could not perform. The fourth
and fifth are the pinned baseline behavior, which must be unchanged. The
sixth is A6's suite, which imports `gpu_tiling` for its mirror check and so
is sensitive to the additive edits in section 9.

A new test file for this lane's two modules is required and was not written.
It needs no accelerator and belongs in the plain `test` task rather than
`test-gpu`, next to A6's, whose `tests/parallel/` directory is still not
referenced by any pixi task (A6 section 5).
