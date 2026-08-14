# Hybrid CPU/GPU leaf scheduling and histogram reuse

Design note for `src/mojoboost/hybrid_leaf_scheduler.mojo` and
`src/mojoboost/histogram_cache_policy.mojo`. Nothing described here is
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
| row-id readback: `4 * node_rows` bytes, plus one `synchronize()` | `node_rows` |

The host has a fixed term too — the zeroing pass — so a tiny leaf is *not*
free on the host, and that is the first thing a reader should take from this
note. What the host does not pay is the launch, the 306 KB download, and the
conversion; what it pays instead is a readback proportional to the node and a
slower per-row accumulation. Whether that trade wins is an empirical question
about four machine constants, and the answer is not in this repository.

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

`row_source_is_host` names three cases where the host already has them and
one where it does not:

- `ROWS_HOST_IDENTITY` — the unbagged root. `begin_tree` seeds the identity
  permutation with a kernel; the host can reproduce `0..n_active` without
  reading anything.
- `ROWS_HOST_BAG` — the bagged root. The bag is the host's own list, in the
  order it handed to `begin_tree`.
- `ROWS_HOST_MIRROR` — any descendant of a node whose rows the host already
  holds. The host re-runs the same stable partition on its own list;
  `partition_range_host` in `gpu_active_rows.mojo` is that reference model,
  and it agrees with the device index for index by construction (both are
  stable in *buffer* order, not row-id order).
- `ROWS_DEVICE_COPY` — everything else. `4 * node_rows` bytes and one
  synchronization.

`child_row_source` encodes the consequence: **a host-owned subtree pays its
readback once, at the subtree's root.** That matters more than it looks. A
leaf elected for a host build is small by hypothesis, so its whole subtree is
small, and the readback that admitted it also admits every node beneath it.
The transfer is amortized over the subtree, not over one node.

A full host mirror of the permutation was considered and rejected: maintaining
it for every node costs a host partition per split over the parent's rows,
which is the dominant cost of CPU tree growth and would be paid on the large
leaves the GPU is there to handle. The mirror is lazy, per subtree, and only
where a placement was accepted.

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

1. **Placement is a pure function of shape.** `place_leaf` reads node row
   counts, dataset shape, active feature count, launch count, run
   configuration, and cost coefficients. It reads no clock, no queue depth,
   and no completion signal. Two runs on one machine place every leaf
   identically, and a run where the host build happens to overlap the device
   places them the same as one where it does not.

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

### 8.1 `gpu_active_rows.mojo` — a per-range row download

Today `download_rows` copies the whole `n_rows` buffer and synchronizes, and
`download_range` calls it and slices. A host build of a small leaf needs
`4 * node_rows` bytes, not `4 * n_rows`:

```
def download_range_only(mut self, node: Int) raises -> List[Int]:
    # enqueue_copy of rows_dev[window.begin : window.end] into a host buffer
    # sized by the window, then synchronize.
```

Without it, the readback term in the cost model is `4 * n_rows` for every
node and the model will decline almost everything. This is the single change
that decides whether hybrid scheduling is worth integrating.

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

Four, in `grow_tree_gpu`'s host-search path only:

1. After `builder.set_features(tree_features)`, construct the
   `HybridContext` once per tree and the `CacheEpochs` once per session.
2. At the root, `place_leaf` with `ROWS_HOST_IDENTITY` or `ROWS_HOST_BAG`.
   The root is the one node whose rows are free on the host, and also the
   largest, so it will be declined on cost — but it is where a mirror mode
   gets its first comparison for free.
3. Replacing the `if n_left <= n_right` block: call `plan_split`, then either
   `builder.build_leaf(direct_node)` or the host builder, then
   `subtract_histogram` exactly as today.
4. Around the frontier scan: `barrier.open()` / `expect` / `complete` /
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

**E1 — Calibrate the seven coefficients.** A microbenchmark over
`node_rows ∈ {4, 16, 64, 256, 1k, 4k, 16k, 64k}` × `n_features ∈ {10, 100,
1000}` × `n_bins ∈ {16, 64, 255}`, timing `enqueue_leaf`, `download_raw`,
`histogram_from_host`, `build_histogram_subset_into`, `Histogram.reset`, and a
range readback separately. Fills `HybridCosts` and produces the `evidence_id`
its constructor demands. Until this runs, every placement is
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
achieved overlap before claiming it; a `synchronize()` in the row readback
serializes more than it looks.

**E6 — The subtree amortization.** Measure how much of the win comes from
`ROWS_HOST_MIRROR` (one readback per host-owned subtree) versus per-node
readbacks. If the subtree rule carries the design, the row-source bookkeeping
is the important part of this lane and the cost model is secondary.

## 10. Status

Not enabled. Not called. Not measured.

`MOJOBOOST_HYBRID_LEAVES` selects a mode and every mode is still declined for
want of a measured cost model — deliberately, so that the decline reason is
observable through `describe_placement` before any numbers exist.
