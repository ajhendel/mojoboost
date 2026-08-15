# Hybrid CPU/GPU leaf scheduling and histogram reuse

Design note for `src/mojotrees/hybrid_leaf_scheduler.mojo` and
`src/mojotrees/histogram_cache_policy.mojo`. Nothing described here is
enabled, and nothing described here has been measured. The modules are
policy and bookkeeping only: they accumulate no histogram, own no buffer,
open no device, and are not called from any trainer.

## 1. The observation

`GpuHistogramBuilder.build_leaf` costs, per node:

| term | scales with |
| --- | --- |
| kernel launch (one, or two on the tiled strategy) | nothing |
| accumulation over the node's compacted row range | `node_rows * n_slots` |
| `download_raw`: `12 * n_features * n_bins` bytes, plus one `synchronize()` | nothing |
| `histogram_from_host`: `n_features * n_bins` cell conversions | nothing |

Three of the four terms are fixed per node. At 100 features and 255 bins the
download alone is 306 KB and the conversion is 25,500 cells, and a leaf that
owns four rows pays all of it.

A host build of the same node costs:

| term | scales with |
| --- | --- |
| `Histogram.reset`: `n_features * n_bins` cells zeroed | nothing |
| `_accumulate_subset`: scattered adds over the node's rows | `node_rows * n_slots` |
| materializing the node's rows on the host's permutation copy | the *parent's* rows, and only once per split |
| the tree's active-row snapshot | `n_active`, **once per tree** (§3) |

The host has a fixed term too — the zeroing pass — so a tiny leaf is *not*
free on the host, and that is the first thing a reader should take from this
note. What the host does not pay is the launch, the 306 KB download, and the
conversion; what it pays instead is a slower per-row accumulation, a cheap
per-split partition mirror, and one whole-permutation readback per tree that
every host leaf in that tree shares. Whether that trade wins is an empirical
question about eight machine constants, and the answer is not in this
repository.

Deep in a leaf-wise tree most nodes are small. `num_leaves` defaults to 31,
and the leaves that survive to the frontier late in a tree are the ones that
have been split repeatedly. That is the population this design is aimed at.

## 2. What has to be true for a host build to be possible at all

Three facts about the current trainer make it possible, and two configurations
remove it.

**The binned matrix is already host-resident.** `train_gpu` holds the caller's
`BinnedMatrix` for the whole fit and walks it every round in
`tree.predict_row`. The device's copy is a second copy. A host build reads
bins at no transfer cost.

**The parent histogram is already host-resident** on the default split path.
`grow_tree_gpu` downloads every node's histogram and performs sibling
subtraction on the host (`subtract_histogram`). So a host-built child's
sibling comes out of arithmetic the grower already performs, with no extra
work at all.

Under `SPLIT_SEARCH_DEVICE` this is false: `_grow_tree_gpu_device_search`
builds and searches both children on the device and downloads a 136-byte
record instead of a histogram. There is no host parent, and there is nothing
for a host build to plug into. **Hybrid scheduling and device split search are
mutually exclusive**, and `DECLINE_NO_HOST_PARENT` says so.

**Only the row ids are device-owned**, on the host-gradient path. `grad` and
`hess` are Float64 host lists that `upload_gradients` copies to the device;
the host keeps them. What the host lacks is the node's slice of the
active-row permutation, which lives in `rows_dev`.

Under the device objective path (`fill_gradients_device`) this is false: the
gradients are generated on the device from device-resident labels and raw
scores and never exist on the host. Pulling them back would cost `8 * n_rows`
per round to save a per-node download, which inverts the whole point of that
path. **Hybrid scheduling and device-generated gradients are mutually
exclusive**, and `DECLINE_GRADIENTS_ON_DEVICE` says so.

That leaves the intersection where hybrid scheduling is even a candidate:
host gradients (which is every custom objective, every bagged or GOSS run,
and every early-stopping run) with host split search (the default).

## 3. Ownership

Stated once, because a hybrid grower is exactly where an unstated ownership
rule becomes a bug.

| state | owner | who may read it | who may write it |
| --- | --- | --- | --- |
| binned matrix, host copy | the caller's `BinnedMatrix` | host accumulation, prediction | nobody during a fit |
| binned matrix, device copy | `GpuHistogramBuilder.bins_dev` | device kernels | uploaded once at construction |
| gradients/hessians, host | the trainer's `grad`/`hess` lists | host accumulation, GOSS ranking | `_fill_grad_hess`, once per round |
| gradients/hessians, device | `grad_dev`/`hess_dev` | device kernels | `upload_staged`, once per round |
| fixed-point scales | `builder.g_scale`/`h_scale` | both producers | once per round, with the gradients |
| active-row permutation | `GpuActiveRows.rows_dev` | device kernels | the partition, one leaf at a time |
| leaf ranges | `LeafRangeTable` (host mirror of the above) | both | `begin_tree`, `split` |
| a node's row ids, host side | the electing build, for the duration of that build | that build | nothing else |
| histograms | the frontier leaf that holds them | the grower | the producer, once |

Two invariants follow, and both are checkable:

1. **The device always partitions.** A host-built node does not stop the
   device from splitting its range. Keeping the partition unconditional is
   what keeps `LeafRangeTable`'s tiling invariant true, keeps
   `update_raw_device` able to read leaf ranges, and keeps the host row
   mirror comparable against `download_range` in a verification mode. The
   cost is a kernel launch over a small range, which is the cheapest thing in
   this system.
2. **A node's histogram has exactly one producer.** `MODE_MIRROR` looks like
   an exception and is not: the host build's output is compared and
   discarded, and the device's histogram is the one admitted to the cache and
   consumed by the grower.

### Where a host build gets its rows

> **Superseded (2026-08-15).** The premise of this subsection was wrong:
> `DeviceBuffer.create_sub_buffer(offset, size)` views a window of the row
> buffer, and `enqueue_copy` from that view moves exactly `4 * node_rows`
> bytes with no allocation and no kernel. `GpuHistogramBuilder.readback_range`
> is that call, and the shipped grower plans every leaf as
> `ROWS_DEVICE_COPY`: each elected leaf pays only for its own rows, nothing
> is amortized, and the election no longer depends on the dataset's size.
> The snapshot machinery below is kept in the vocabulary and the cost model
> (`snapshot_nanos`, `ROWS_HOST_SNAPSHOT`, `partition_range_host`) but the
> GPU grower does not use it. See §10.

The obvious mechanism was believed not to be available, and this was the
design's sharpest constraint.

**A per-range readback is not expressible** (wrong, see above). `DeviceContext.enqueue_copy`
copies the whole source buffer — confirmed by `download_grad_hess` in
`gpu_objectives_native.mojo`, which copies an `n_rows` buffer into an
`n_rows * n_classes` host buffer and gets `n_rows` elements. So
`download_rows` moves `4 * n_rows` bytes whatever the node's size, and a
"read back just this node's rows" call needs either a device buffer allocated
per call (an allocation per elected leaf) or a sub-buffer / offset-and-count
API this project does not have. A per-node readback of `4 * n_rows` would
cost *more* than the `12 * F * B` download it was meant to avoid on any
dataset with more rows than `3 * F * B`.

**So the design does not use one.** The host takes one whole-permutation
snapshot per tree and maintains it itself:

- `snapshot_nanos(n_active, costs)` — one `download_rows` and one sync,
  `4 * n_active` bytes, **once per tree**.
- Every node alive at snapshot time reads `snapshot[begin : end]` for free;
  the ranges are already host-tracked in `LeafRangeTable`.
- A split re-partitions `rows_dev[parent.begin : parent.end]` and provably
  touches nothing outside it (`gpu_active_rows.mojo` states that invariant
  explicitly). The host keeps the snapshot valid by mirroring that one
  partition on its own copy with `partition_range_host` — the same stable,
  buffer-order rule, agreeing with the device index for index. Cost is one
  partition over the *parent's* rows (`host_materialize_nanos`), and it
  materializes **both** children at once, so the sibling never pays again.
- The root needs no snapshot at all: `ROWS_HOST_IDENTITY` unbagged (the host
  reproduces `0..n_active` without reading anything), `ROWS_HOST_BAG`
  otherwise (it is the host's own list, in the order it handed to
  `begin_tree`).

The snapshot is charged **in full to the one leaf that first elects to use
it** (`DECLINE_SNAPSHOT_NOT_PAID` when that leaf's own saving cannot cover
it); every later host leaf in the tree reads it for nothing. Deliberately
conservative: charging the first leaf the whole price cannot make a tree
slower than the pure-device path, whereas amortizing over an *expected* number
of host leaves would be a guess about a distribution nobody has measured. The
cost of that conservatism is that a tree whose savings are spread thinly
across many leaves never snapshots at all — E4 and E6 in §9 are what would say
whether that matters.

`ROWS_DEVICE_COPY` stays in the vocabulary, and `row_transfer_bytes` still
prices it, so that if a per-range copy ever becomes expressible the model can
compare the two. Nothing in the design depends on it.

A *full* host mirror — mirroring every split from the root down, with no
snapshot — was considered and rejected: it costs a host partition per split
over the parent's rows, which is the dominant cost of CPU tree growth and
would be paid on exactly the large leaves the GPU is there to handle. The
mirroring here is lazy: only along the paths to elected leaves, and only after
the snapshot has made the ancestors cheap.

## 4. Substitutability: the arithmetic that has to match

This is the crux of the correctness story, and it is the reason
`MODE_HOST_FLOAT64` is named separately from `MODE_REPLICA`.

The device accumulates in fixed point (`histogram_gpu.mojo`):

```
gq = round(Float32(grad[r]) * g_scale)      # Int32
```

accumulates `gq` with integer atomics, and the host dequantizes on download by
`Float64(sum) * (1 / g_scale)`. The CPU builder (`histogram.mojo`) sums
Float64 gradients directly. **The two are not the same number.** They agree to
Float32 precision, which is enough to compare a benchmark and not enough to
substitute one for the other: a tree grown from a mixture of the two is a
third tree, different from both homogeneous ones, and its splits differ
wherever two candidate gains were within Float32 noise.

So a substitutable host build must reproduce the device pipeline exactly:

1. Convert `grad[r]` to Float32 — bit for bit what `stage_gradients` does.
2. Multiply by the round's Float32 `g_scale` and `round()` to Int32.
3. Accumulate in Int32. Integer addition is associative and commutative, so
   the accumulation order (per-feature tasks, SIMD, anything) cannot change
   the sum. This is the same property the two GPU strategies rely on to be
   bit-identical to each other.
4. Dequantize with the same `Float64(sum) * (1 / scale)`.

If steps 1–4 hold, a host-built histogram is **bit-identical** to the device's,
sibling subtraction across the two is exact, and hybrid growth produces the
same tree as pure-GPU growth. That is a strong claim and it rests on one
unverified assumption: that a Float32 multiply-and-round on the host produces
the same Int32 as the same operation in a Metal / CUDA / HIP kernel. Compilers
may contract the multiply into an FMA, denormal handling may flush, and the
rounding mode of `round()` on either side must be the same tie rule. None of
this has been checked on any device.

That check is what `MODE_MIRROR` is for, and
`HybridContext.quantized_replica_verified` is the flag it sets. `MODE_REPLICA`
is declined with `DECLINE_REPLICA_UNVERIFIED` until then.

The replica kernel itself is deliberately **not** in this lane: it would be an
edit to `histogram.mojo`, which this lane does not own. Section 8 specifies
it.

## 5. Reuse, and the reuse that is not there

`histogram_cache_policy.mojo` enumerates this in its docstring; the short
version, because the honest answer is mostly negative:

- **Across rounds: none.** A histogram is a sum of that round's gradients. The
  fixed-point scales change too.
- **Across trees: none.** `begin_tree` reseeds the permutation and node ids
  restart at 0, so a node id from the previous tree names different rows.
- **Across feature sets: none, and dangerously so.** Every histogram has the
  dataset's full `n_features * n_bins` shape with inactive features' slices
  left at zero. A histogram built under feature set A reads under feature set
  B as "those features had no rows" — wrong, and silently so. This is the one
  invalidation that a single-device grower gets for free (it never has a
  stale histogram to hit) and a cache must enforce explicitly.
- **Across per-node feature draws: total reuse, already.**
  `feature_fraction_bynode` narrows the *search*, not the accumulation. One
  histogram serves every draw, and re-searching it under different monotone
  bounds is free. Neither event invalidates anything.
- **Parent and sibling: the whole of the available reuse**, and both growers
  already take it.

So the cache is not a hit-rate optimization. Its job is to make the existing
lifetime rule enforceable when two producers exist:
`HistogramCache.check_subtraction` refuses to subtract a Float64 host
histogram from a dequantized device parent, which is precisely the mistake a
hybrid grower can make without noticing.

### Invalidation

Three counters plus a dataset identity, in `CacheEpochs`:

| counter | bumped by | invalidates |
| --- | --- | --- |
| `round_epoch` | `upload_gradients` / `fill_gradients_device` / each class's softmax fill | everything |
| `tree_epoch` | `begin_tree` | everything from the previous tree |
| `feature_epoch` | a `set_features` that actually changed the set | every histogram built under the old set |

`set_features` already returns early when the set is unchanged, and
`CacheEpochs.set_features(changed)` takes that same boolean, so an unchanged
feature set does not invalidate the root histogram that was built under it.

Invalidation is not an operation. A key either matches the current epochs or
it does not; `staleness` says which counter disagreed, and `drop_stale`
sweeps.

### Memory bound

`HISTOGRAM_BYTES_PER_CELL` is 24 (Float64 grad + Float64 hess + `Int` count).
The live set is bounded at `num_leaves + 1` histograms: one per frontier leaf,
plus the parent that is still live while its subtraction is being read.

```
capacity_bound_bytes = 24 * n_features * n_bins * (num_leaves + 1)
```

At 100 features, 255 bins, 31 leaves: 19.6 MB. A hybrid grower does not raise
this. It changes which device accumulated a buffer, not how many are live.

## 6. Determinism contract

Four clauses. The first two are enforced by the module's structure; the third
and fourth are obligations on the integration.

1. **Placement is a pure function of its arguments.** `place_leaf` reads node
   row counts, dataset shape, active feature count, launch count, run
   configuration, the snapshot flag, and cost coefficients. It reads no
   clock, no queue depth, and no completion signal. Two runs on one machine
   place every leaf identically, and a run where the host build happens to
   overlap the device places them the same as one where it does not.

   The snapshot flag is the one argument that is tree *state* rather than
   tree shape, which is why it is passed in (`HybridContext.with_snapshot`)
   rather than remembered inside the scheduler. It advances exactly once per
   tree, at the grower's own deterministic node order, and nothing about when
   a build finished can reach it. Placement is thus a pure function of
   (shape, configuration, snapshot state), and the snapshot state is itself a
   deterministic function of the tree traversal.

2. **Completion order is not consumption order.** `WaveBarrier` holds the
   announcement order (the order the grower will consume the results) and the
   completion order (whatever the hardware delivered) separately. `announced()`
   is the only one a caller may use; `completion_order()` exists so a trace
   can show the two differed, which is the evidence the barrier is
   load-bearing. `require_ready` raises if a split is selected while any
   announced build is outstanding.

3. **The frontier scan is unchanged.** The best-gain loop in `grow_tree_gpu`
   scans the frontier by index with a strict `>`, so ties go to the lower
   frontier index. Hybrid scheduling must not reorder the frontier: the left
   child replaces the parent's slot and the right child is appended, whichever
   device built which. A hybrid grower that appended in completion order would
   change the tie-break and therefore the tree.

4. **Row counts come from the parent histogram, not from the device.** The
   grower already knows `n_left` exactly from the parent's integer counts
   (`_count_left`), before the partition runs. `plan_split` takes those counts
   and plans without waiting for any device work — which is also what lets a
   host build of the small child overlap the device partition instead of
   following it.

Under `MODE_REPLICA` with a verified replica, hybrid growth is bit-identical
to pure-GPU growth. Under `MODE_MIRROR` it is bit-identical by construction
(the device's histogram is consumed). Under `MODE_HOST_FLOAT64` it is not, and
that mode exists to be benchmarked as a different algorithm, never to be
enabled as an optimization of the shipped one.

## 7. Failure and fallback

A host build can fail: a shape mismatch, a row readback error, a node whose
rows turn out not to match the range. The fallback is to build that node on
the device instead.

**That fallback is only safe under bitwise equivalence.** If the host build
substitutes and is *not* bit-identical, then falling back changes which
histogram the tree was grown from, and two runs that failed at different
nodes produce different models. So:

- Under `MODE_MIRROR`, a host-build failure is a diagnostic. The device's
  histogram was going to be used anyway; log and continue.
- Under `MODE_REPLICA` with `quantized_replica_verified`, a fallback is
  invisible: the two paths produce the same bits, so the tree is the same
  whether the fallback fired or not. This is the only mode where per-node
  fallback is a legitimate resilience mechanism.
- Under `MODE_HOST_FLOAT64`, a per-node fallback is **not** permitted. The two
  paths disagree, so a mid-tree fallback silently produces a third model. The
  correct failure behavior is to abandon the tree and regrow it on one device.

The cost model has no failure mode of its own: `HybridCosts.unmeasured()`
declines every leaf, which is the shipped behavior, so a cost model that is
missing, wrong, or partially filled cannot do anything worse than the current
trainer does.

## 8. Integration seam

Nothing below is implemented in this lane. Each item names the file it would
touch and what it needs.

### 8.1 `gpu_active_rows.mojo` — nothing required

`download_rows` as it exists today is the snapshot. The integration was to
call it once per tree, at the first leaf that can pay for it, and maintain the
result with `partition_range_host`, which also already exists and is already
the tested reference model for the device partition.

A genuine per-range readback lets a single elected leaf skip the whole-buffer
cost. Three ways were listed:

1. a `node_rows`-sized `DeviceBuffer` allocated per call, then a compaction
   kernel in the shape of the existing `_copy_back_kernel`;
2. `enqueue_copy` gaining an element-count or offset parameter; or
3. a `create_sub_buffer`-style view on `DeviceBuffer`.

**Option 3 already exists** (`DeviceBuffer.create_sub_buffer[dtype](offset,
size)`), and it is what shipped: `GpuHistogramBuilder.readback_range(begin,
count, out)` in `histogram_gpu.mojo`. No allocation, no kernel, one copy of
`4 * count` bytes and one synchronize. With it the snapshot, the mirror, and
`DECLINE_SNAPSHOT_NOT_PAID` are all unnecessary in the GPU grower, and the
per-leaf transfer term (`host_transfer_nanos` under `ROWS_DEVICE_COPY`) is the
one the scheduler actually charges.

### 8.2 `histogram.mojo` — the fixed-point replica builder

```
def build_histogram_subset_fixed_into(
    mut out: Histogram,
    data: BinnedMatrix,
    grad: List[Float64], hess: List[Float64],
    rows: List[Int], row_start: Int, row_count: Int,
    g_scale: Float32, h_scale: Float32,
    features: List[Int] = [],
) raises
```

Same loop as `_accumulate_subset`, with Int32 fixed-point accumulators and the
Float32 quantization of §4, dequantized on the way out. It must reuse
`dispatch_features` unchanged: per-feature tasks write disjoint output slices,
and integer accumulation is order-independent, so the task count cannot change
the result — the same argument `histogram.mojo` already makes for the Float64
path.

### 8.3 `train_gpu.mojo` — the call sites

Five, in `grow_tree_gpu`'s host-search path only:

1. After `builder.set_features(tree_features)`, construct the
   `HybridContext` once per tree (with `n_active_rows` and
   `snapshot_taken=False`) and the `CacheEpochs` once per session.
2. At the root, `place_leaf` with `tree_row_source(is_root=True, bagged)`.
   The root is the one node whose rows are free on the host, and also the
   largest, so it will be declined on cost — but it is where a mirror mode
   gets its first comparison for free.
3. Replacing the `if n_left <= n_right` block: call `plan_split`, then either
   `builder.build_leaf(direct_node)` or the host builder, then
   `subtract_histogram` exactly as today.
4. When a placement comes back with `takes_snapshot`, call
   `builder.rows.download_rows()` once, keep the result as the tree's host
   permutation, and replace the context with `ctx.with_snapshot(True)` for
   the rest of the tree. Every later split mirrors its own partition into
   that array with `partition_range_host`, which keeps it valid without
   re-reading it. `begin_tree` drops it.
5. Around the frontier scan: `barrier.open()` / `expect` / `complete` /
   `require_ready()`.

The `_GpuLeafState` frontier is untouched, the subtraction is untouched, the
best-gain scan is untouched, and `apply_split` is untouched.

### 8.4 `gpu_runtime.mojo` — accounting

A host build removes one `SYNC_HOST_READ` on `RES_OUT` and adds one on the row
buffer. `HazardTracker` has no resource id for the active-row buffer today
(`RES_LEAF` is the old per-row leaf-id array). A hybrid integration wants one,
so `audit_round` keeps counting the syncs that actually happen.

### 8.5 Python

None. Placement is per leaf and stays in Mojo; the only surface a binding
could ever want is the mode switch, and that is already an environment
variable.

## 9. Experiments that would settle it

In order. Each one can kill the design, and the first two do not require any
of the integration above.

**E1 — Calibrate the eight coefficients.** A microbenchmark over
`node_rows ∈ {4, 16, 64, 256, 1k, 4k, 16k, 64k}` × `n_features ∈ {10, 100,
1000}` × `n_bins ∈ {16, 64, 255}`, timing `enqueue_leaf`, `download_raw`,
`histogram_from_host`, `build_histogram_subset_into`, `Histogram.reset`,
`partition_range_host`, and `download_rows` separately. Fills `HybridCosts`
and produces the `evidence_id` its constructor demands. Until this runs,
every placement is
`DECLINE_COSTS_UNMEASURED`.

**E2 — The bitwise replica claim.** Build the same node's histogram both ways
(device fixed-point, host replica of §4) over adversarial gradients: values
near the Float32 ulp, values at the rounding tie, denormals, and a scale near
the `_fixed_scale` floor. Compare Int32 planes, not dequantized Float64. If
they differ on any device, `MODE_REPLICA` is dead on that device and the
design reduces to `MODE_MIRROR` (diagnostic) plus `MODE_HOST_FLOAT64` (a
different algorithm).

**E3 — Where the crossover actually is.** With E1's coefficients, sweep
`node_rows` at fixed `(n_features, n_bins)` and find the row count where the
modelled costs cross. Then measure it directly and check that the model
predicted it. A model that does not predict its own crossover within a factor
of two is not a model.

**E4 — Does it matter on a real tree?** Instrument a tree's node-size
distribution at `num_leaves ∈ {31, 127, 1023}` on datasets of 10k, 100k, and
1M rows, and compute what fraction of nodes fall below E3's crossover and what
fraction of *total node-build time* they represent. A design that fires on 60%
of nodes worth 3% of the time is not worth integrating; that outcome is a
legitimate result and should be recorded as one.

**E5 — Overlap.** The model in §1 assumes the host build and the device work
are serial. They need not be: the host can accumulate the small child while
the device partitions and while the *other* subtree's kernels run. Measure the
achieved overlap before claiming it; the `synchronize()` inside the snapshot
serializes more than it looks, though it is paid once per tree rather than per
node.

A constraint on E5 that the substitution design does not have, stated here
so the experiment is read correctly. On unified memory the CPU and the GPU
draw on one memory bus, so overlap adds throughput only while the device
work in flight is *not* bandwidth-bound. A histogram accumulation over a
large node is bandwidth-bound by construction (every row's `n_slots` bins
are gathered once); a host build racing it competes for the same bytes per
second and can slow the device kernel by as much as it gains. What overlap
can add is the launch-bound tail: small leaves, partitions, the per-node
fixed costs the substitution scheduler already targets. So the shipped
1.20x on a bagged 20k-row fit
(`bench/results/apple_m4_hybrid_costs_2026-08-14.md`) came from *removing*
fixed device costs on small leaves, not from adding CPU throughput to a
bandwidth-bound phase, and it is the first thing E5 must reproduce before
overlap is credited with anything more. The measurement E5 owes is
therefore per phase, not per fit: device kernel time with and without a
concurrent host build over a node large enough to saturate the bus, and
again over one small enough not to. A design that shows overlap paying on
the second and costing on the first has learned the same lesson the
substitution rule already encodes, and should keep the rule.

**E6 — Is the snapshot the limit?** Two questions, and they decide where the
next effort goes. First: how often does the first candidate leaf fail
`DECLINE_SNAPSHOT_NOT_PAID` — that is, how often does a tree have savings
worth having but no single leaf big enough to buy the snapshot? Second: once a
snapshot is taken, what fraction of host-build time goes to
`partition_range_host` materialization versus accumulation? If the first
number is large, the per-range readback of §8.1 option 1 becomes worth
building; if the second is large, the lazy mirroring is the wrong shape and
the snapshot should be retaken instead of maintained.

## 10. Status

Wired, default off, and off unless *two* switches opt in.

- **E1 ran** (`pixi run bench-hybrid-costs`, `bench/bench_hybrid_costs.mojo`),
  and `HybridCosts.apple_m4()` cites its output
  (`bench/results/apple_m4_hybrid_costs_2026-08-15.md`; the 2026-08-14 run
  is kept, but it lacked the scattered-leaf rate below and coefficients
  from two time windows are not mixed). The costs are selected only by an
  explicit `MOJOTREES_HYBRID_COSTS=apple-m4`, never inferred from the
  hardware; without it every leaf still declines with
  `DECLINE_COSTS_UNMEASURED`.
- **The host rate depends on row density.** Bins are feature-major, so a
  small leaf of a large dataset reads one cache line per bin — measured
  ~3x the contiguous root's per-slot rate (`host_scatter_nanos_per_krow_slot`
  vs `host_nanos_per_krow_slot`). Priced with the contiguous rate alone the
  scheduler elected leaves at 500k–1M rows that lost; `host_slot_nanos_per_k`
  now interpolates by the node's mean row gap (`LeafWork.dataset_rows`,
  carried for exactly this), which is what keeps the never-slower guarantee
  at large row counts.
- **The §8.2 replica builder exists**
  (`histogram.build_histogram_subset_replica_into`, reached through
  `GpuHistogramBuilder.build_leaf_host_replica`), with the quantization
  hoisted to once per row — the per-(row, feature) form measured twenty
  times slower than the Float64 builder, the hoisted form within 1.7x.
- **E2 runs in-process**: a `replica` run's first accepted leaf executes as
  a mirror, and the bitwise comparison sets
  `GpuHistogramBuilder.replica_state`. Verified, later accepted leaves
  substitute; refuted, hybrid scheduling retires for the fit and the
  pure-device path continues. `tests/parallel/test_hybrid_replica.mojo`
  makes the same comparison over adversarial gradients and holds a hybrid
  fit bit-identical to the pure-device fit.
- **The §8.3 integration uses the per-range readback, not the snapshot.**
  Step 4's maintained snapshot (mirror every later split with
  `partition_range_host`) prices at `host_partition_nanos_per_krow` =
  15,485 on the calibration machine — tens of milliseconds per tree at
  500k rows, more than the builds it enables — and the first shipped
  integration retook a whole-permutation snapshot per elected leaf instead,
  which meant no leaf could pay above ≈200k active rows. Both are moot:
  §8.1's option 3 exists (`DeviceBuffer.create_sub_buffer`), so
  `grow_tree_gpu` plans every leaf as `ROWS_DEVICE_COPY` and
  `GpuHistogramBuilder.readback_range` moves exactly that leaf's window
  before the host replica runs. Each leaf is charged its own transfer and
  nothing is amortized, so a host placement can never make a tree slower
  than the pure-device path, and the election depends only on the leaf's
  size, not the dataset's. `guard_transfer_dominates` is off in the grower
  (`MOJOTREES_HYBRID_GUARD_TRANSFER=1` restores it), because under a
  per-range readback the "transfer" is the fixed synchronize the device
  path also pays; measurements in
  `bench/results/apple_m4_hybrid_costs_2026-08-14.md`.
- **`MODE_HOST_FLOAT64` stays unwired** in the grower, exactly because §7
  allows it no per-node fallback: it is a different algorithm, and nothing
  in the trainer should reach one through an environment variable that
  looks like an optimization switch.

E3 through E5 remain open. E6 is answered by construction: there is no
snapshot to go unpaid.
