# Algorithm 27: hybrid CPU/GPU leaf scheduling and histogram reuse

Status: two standalone policy modules plus a design note. **Not wired into any
caller, not enabled, not measured, not tested, not committed.** No existing
file was edited.

Files delivered (all four are this lane's exclusive ownership):

- `src/mojoboost/hybrid_leaf_scheduler.mojo` (new)
- `src/mojoboost/histogram_cache_policy.mojo` (new)
- `docs/design/HYBRID_TRAINING.md` (new; `docs/design/` created)
- `handoffs/algorithm_27_hybrid_scheduler.md` (this file)

Nothing was built, run, benchmarked, or profiled, per the lane's constraints.
The modules have not been compiled. Section 10 lists what a first compile is
most likely to catch.

Provenance note: this lane committed nothing. A concurrent session in the
shared checkout swept the two Mojo modules into `b04b5f0 "Integrate parallel
release and accelerator work"` while this lane was still writing the two
documents, so the modules are already tracked; the working tree carries three
later `raises` annotations on top of that commit (see §10.7). The two
documents are untracked.

Read `docs/design/HYBRID_TRAINING.md` first; it is the argument. This file is
the integration record.

---

## 1. The result, stated up front

The design is **feasible and narrowly scoped**, and the honest summary is
three sentences.

The GPU pays three per-node costs that do not scale with the node — a kernel
launch, a `12 * n_features * n_bins` byte download with a host
synchronization, and a `n_features * n_bins` cell conversion — so a four-row
leaf costs almost as much as a four-thousand-row leaf. The host pays a fixed
zeroing pass of the same cell count plus a row-proportional readback, so the
host is *not* automatically cheaper for a tiny leaf and the crossover is a
real empirical question about seven machine constants, none of which this
repository has measured. Hybrid scheduling is available only on the
intersection of host-resident gradients and host split search, which excludes
`SPLIT_SEARCH_DEVICE` and the device objective path entirely.

The reuse half of the task returns a **negative result**: beyond the
parent/sibling subtraction both growers already exploit, there is no
histogram reuse available in this trainer. Gradients change every round, node
ids restart every tree, and a histogram built under one feature set is
silently wrong under another. What the cache layer is actually for is making
that last invalidation *detectable*, and making a cross-device subtraction
refuse to mix two arithmetics.

---

## 2. State machine

### 2.1 Per-node placement

```
                        ┌──────────────┐
                        │  ANNOUNCED   │  grower calls barrier.expect(node)
                        └──────┬───────┘
                               │ place_leaf(ctx, work)   [pure]
                 ┌─────────────┴─────────────┐
       accepted  │                           │  declined (any reason)
                 v                           v
        ┌────────────────┐          ┌──────────────────┐
        │ HOST_ELECTED   │          │  DEVICE_ELECTED  │
        └───────┬────────┘          └────────┬─────────┘
                │                            │ enqueue_leaf
     rows host-side?                         │ download_raw  (sync)
      no │      │ yes                        │ histogram_from_host
         v      │                            v
  ┌────────────┐│                   ┌──────────────────┐
  │ ROWS_READY │◄┘                  │ DEVICE_BUILT     │
  │  (readback,│                    └────────┬─────────┘
  │   sync)    │                             │
  └─────┬──────┘                             │
        │ host accumulate                    │
        v                                    │
  ┌────────────┐   MODE_MIRROR: compare      │
  │ HOST_BUILT ├────────────────────────────►│  device histogram wins
  └─────┬──────┘                             │
        │ MODE_REPLICA / MODE_HOST_FLOAT64   │
        v                                    v
        └──────────────► ┌──────────────┐ ◄──┘
                         │  COMPLETED   │  barrier.complete(node)
                         └──────┬───────┘
                                │ all announced nodes complete
                                v
                         ┌──────────────┐
                         │ COMMIT_READY │  barrier.require_ready()
                         └──────────────┘
```

A host build that raises re-enters at `DEVICE_ELECTED`. §6 states when that
fallback is legitimate.

### 2.2 Per-session epochs

`CacheEpochs` moves on exactly four events and never moves backward:

```
start(dataset_id)
   │
   ├── begin_round()      round_epoch++      every histogram dies
   ├── begin_tree()       tree_epoch++       every histogram from the last tree dies
   ├── set_features(true) feature_epoch++    every histogram under the old set dies
   └── set_features(false)  (no-op)          the root histogram survives
```

`set_features(changed)` takes the boolean `histogram_gpu.set_features` already
computes for its own early return. Passing `True` unconditionally would drop
every tree's root histogram; passing the real answer keeps it.

### 2.3 Per-histogram lifetime

```
   produced ──admit(key, origin)──► LIVE ──release(key)──► gone
                                     │
                                     └──drop_stale(now)──► gone (epoch moved)
```

`admit` refuses `ORIGIN_UNKNOWN`, refuses a key that is already stale, refuses
a duplicate key, and refuses to exceed the byte ceiling. There is no eviction:
a ceiling can only refuse, because nothing here owns a buffer it could free.

---

## 3. Cost-model inputs

Seven coefficients on `HybridCosts`, all integers in nanoseconds per unit so
the comparison is reproducible bit for bit on any machine reading the same
numbers:

| coefficient | unit | measures |
| --- | --- | --- |
| `launch_nanos` | ns / launch | enqueue + run of one histogram kernel, row work excluded |
| `sync_nanos` | ns | one `ctx.synchronize()` host round trip |
| `transfer_nanos_per_kib` | ns / KiB | device→host copy through pinned staging |
| `device_nanos_per_krow_slot` | ns / 1000 (row,feature) | device accumulation |
| `host_nanos_per_krow_slot` | ns / 1000 (row,feature) | `_accumulate_subset`'s scattered adds |
| `host_zero_nanos_per_kcell` | ns / 1000 cells | `Histogram.reset` |
| `convert_nanos_per_kcell` | ns / 1000 cells | `histogram_from_host` |

Eight workload inputs on `LeafWork`, every one a count the grower already
holds:

`node`, `node_rows` (exact, from the parent histogram's integer counts before
the partition runs), `n_slots` (active features — what is accumulated),
`n_features` (full count — what is *downloaded*, because `out_dev` is copied
at the dataset's full shape whatever the active set is), `n_bins`,
`dataset_rows` (column stride; carried for a fitted model, no term uses it
yet), `row_source`, `gpu_launches` (1 atomic / 2 tiled, taken from the
resolved tiling rather than re-derived here).

Six run inputs on `HybridContext`: `mode`, `device_split_search`,
`gradients_host_resident`, `bins_host_resident`,
`quantized_replica_verified`, `guard_transfer_dominates`.

The model:

```
device = launch*launches + krow_slots*device_rate
       + ceil(12*F*B / 1024)*transfer + sync + kcells*convert

host   = [ ceil(4*rows / 1024)*transfer + sync ]   (zero if rows are host-side)
       + kcells*zero + krow_slots*host_rate
```

**No threshold is shipped.** `HybridCosts.unmeasured()` is the only cost model
this repository can construct — the argument-taking constructor raises without
an `evidence_id`, mirroring `CrossoverEvidence` in `device_policy.mojo` — and
it makes every placement `PLACE_GPU / DECLINE_COSTS_UNMEASURED`. The
distinction the module rests on: a *threshold* says "nodes under N rows go to
the host" and would be invented; a *cost model* says how long each path takes
and the comparison follows. Both are inert without measurements, but only the
second has somewhere to put an answer.

Two deliberate conservatisms:

- A modelled tie stays on the device (`>=`, not `>`). A model that cannot
  separate two paths must not move work.
- `guard_transfer_dominates` (default on) refuses a host build whose readback
  costs more than the accumulation it enables, **even when the head-to-head
  still favors the host**. This directly implements the lane's "avoid copying
  a small leaf to the CPU when the required state transfer costs more than its
  histogram", and it is separable precisely because the head-to-head can
  legitimately disagree — the device path also transfers. A benchmark that
  disagrees clears the flag rather than editing the model.

---

## 4. Determinism contract

Four clauses. Clauses 1 and 2 are enforced by the delivered code; 3 and 4 are
obligations the integration must keep.

1. **Placement is a pure function of shape.** `place_leaf` reads counts,
   shapes, run configuration, and coefficients. No clock, no queue depth, no
   completion signal, no memory of an earlier placement. Replaying a tree's
   nodes in any order reproduces every placement.

2. **Completion order is not consumption order.** `WaveBarrier` keeps
   `announced()` (the order the grower will consume) and `completion_order()`
   (whatever the hardware delivered) as separate lists. Only `announced()` may
   be consumed; `completion_order()` and `order_was_permuted()` exist so a
   trace can *show* the two differed, which is the evidence the barrier is
   load-bearing. `require_ready()` raises if a split is selected while any
   announced build is outstanding, so no gain comparison can see a partly-
   filled frontier.

3. **The frontier scan is unchanged.** `grow_tree_gpu` scans by index with a
   strict `>`, so ties go to the lower frontier index; the left child replaces
   the parent's slot and the right child is appended. A hybrid grower must
   insert in that same order whichever device built which. Appending in
   completion order would change the tie-break and therefore the tree.

4. **Row counts come from the parent histogram.** `n_left` is exact from
   `_count_left` before the partition runs, so `plan_split` never waits on the
   device to decide anything — which is also what lets a host build overlap the
   device partition rather than follow it.

Equivalence by mode:

| mode | substitutes | same tree as pure GPU |
| --- | --- | --- |
| `MODE_OFF` (default) | no | yes, trivially |
| `MODE_MIRROR` | no — builds both, uses the device's | yes, by construction |
| `MODE_REPLICA` | yes | yes **iff** the bitwise claim holds on that device |
| `MODE_HOST_FLOAT64` | yes | **no** — a deliberately different algorithm |

The bitwise claim (§4 of the design note) is that a host fixed-point replica —
Float32 conversion, Float32 scale multiply, `round()` to Int32, Int32
accumulation, identical dequantization — reproduces the device's Int32 planes
exactly. Integer addition is associative, so accumulation *order* provably
cannot matter; what is unverified is whether a Float32 multiply-and-round on
the host matches the same operation in a Metal/CUDA/HIP kernel (FMA
contraction, denormal flushing, rounding tie rule). `MODE_REPLICA` is declined
with `DECLINE_REPLICA_UNVERIFIED` until `quantized_replica_verified` is set,
and only `MODE_MIRROR` can set it.

---

## 5. Cache-memory bounds

- `HISTOGRAM_BYTES_PER_CELL = 24` (Float64 grad + Float64 hess + `Int` count).
- `FIXED_BYTES_PER_CELL = 12` (the device's three Int32 planes).
- `capacity_bound_bytes(num_leaves, F, B) = 24 * F * B * (num_leaves + 1)`.

The `+ 1` is the split peak: leaf-wise growth holds one histogram per frontier
leaf, and during a split the parent is still live (the subtraction is reading
it) while both children exist. One split is in flight at a time in both
growers, so the excess is one histogram, not one per level.

At F=100, B=255, `num_leaves`=31: 19.6 MB. Hybrid scheduling does not raise
this bound — it changes which device accumulated a buffer, not how many are
live, and the subtracted sibling is a buffer the single-device grower also
held.

`HistogramCache.within_bound(num_leaves)` returning False is a **lifetime bug
in the caller**, not a memory-pressure signal: leaf-wise growth cannot hold
more than `num_leaves + 1` histograms unless a parent was never released.
`MOJOBOOST_HIST_CACHE_BYTES` turns the bound into a refusal rather than a swap
storm; it tunes nothing, because nothing here evicts.

---

## 6. Failure fallback

A host build can fail (shape mismatch, readback error, a row window that does
not match). The fallback is to build that node on the device.

**The fallback is only safe under bitwise equivalence**, and this is the
subtlest part of the design:

- `MODE_MIRROR` — a host-build failure is a diagnostic. The device histogram
  was going to be used anyway. Log and continue.
- `MODE_REPLICA` with a verified replica — a fallback is invisible: both paths
  produce the same bits, so the tree is identical whether it fired or not.
  This is the only mode where per-node fallback is a legitimate resilience
  mechanism.
- `MODE_HOST_FLOAT64` — per-node fallback is **forbidden**. The two paths
  disagree, so a mid-tree fallback silently yields a third model, and two runs
  that failed at different nodes disagree with each other. Correct behavior is
  to abandon the tree and regrow it on one device.

The policy layer itself has no failure mode that can make things worse:
unmeasured costs decline every leaf, which is exactly today's behavior, so a
missing, wrong, or half-filled cost model degrades to the shipped trainer.

---

## 7. Integration seam

Five items, in dependency order. Item 7.1 decides whether the rest is worth
doing.

### 7.1 `gpu_active_rows.mojo` — a per-range row download (**blocking**)

`download_rows` copies the whole `n_rows` buffer and synchronizes;
`download_range` calls it and slices. A host build of a small leaf needs
`4 * node_rows` bytes, not `4 * n_rows`. Needed:

```mojo
def download_range_only(mut self, node: Int) raises -> List[Int]:
    # enqueue_copy of rows_dev[window.begin : window.end] into a host buffer
    # sized by the window, then synchronize. Same ordering guarantees as
    # download_range; only the byte count differs.
```

Without it the readback term is `4 * n_rows` for every node and the model will
decline essentially everything. **This is the single change that decides
whether hybrid scheduling is worth integrating at all.** It is not in this
lane's ownership.

### 7.2 `histogram.mojo` — the fixed-point replica builder

```mojo
def build_histogram_subset_fixed_into(
    mut out: Histogram,
    data: BinnedMatrix,
    grad: List[Float64], hess: List[Float64],
    rows: List[Int], row_start: Int, row_count: Int,
    g_scale: Float32, h_scale: Float32,
    features: List[Int] = [],
) raises
```

`_accumulate_subset`'s loop with Int32 fixed-point accumulators, the Float32
quantization of the design note §4, and the same `1/scale` dequantization on
the way out. It reuses `dispatch_features` unchanged: per-feature tasks write
disjoint slices and integer accumulation is order-independent, so the task
count cannot change the result — the same argument `histogram.mojo` already
makes for its Float64 path and the same one the two GPU strategies rely on.

Not in this lane's ownership; specified here so whoever owns
`histogram.mojo` can land it independently and `MODE_MIRROR` can then verify
it.

### 7.3 `train_gpu.mojo` — four call sites, host-search path only

1. After `builder.set_features(tree_features)`: build the `HybridContext` once
   per tree (`HybridContext.from_env(device_split_search, gradients_host,
   bins_host)`), and hold one `CacheEpochs` for the session.
2. Root: `place_leaf` with `ROWS_HOST_IDENTITY` (unbagged) or `ROWS_HOST_BAG`.
   The root is the one node whose rows are free on the host and also the
   largest, so it will be declined on cost — but under `MODE_MIRROR` it is
   where the first bitwise comparison comes for free.
3. Replacing the `if n_left <= n_right:` block: `plan_split(...)`, then either
   `builder.build_leaf(plan.direct_node)` or the host builder, then
   `subtract_histogram` exactly as today.
4. Around the frontier scan: `barrier.open()` / `expect` per announced child /
   `complete` per finished build / `require_ready()` before the best-gain loop.

Untouched: `_GpuLeafState`, the subtraction, the best-gain scan, `_search`,
`apply_split`, and every device-search path.

### 7.4 `gpu_runtime.mojo` — sync accounting

A host build removes one `SYNC_HOST_READ` on `RES_OUT` and adds one on the
active-row buffer, which has no resource id today (`RES_LEAF` is the retired
per-row leaf-id array). A hybrid integration wants one so `audit_round` keeps
counting the syncs that actually happen.

### 7.5 Python / bindings

**None, by design.** Placement is per leaf and stays in Mojo, per the lane's
Mojo-first rule. The only surface a binding could want is the mode switch, and
that is already `MOJOBOOST_HYBRID_LEAVES`.

---

## 8. What was discovered while reading (no edits made)

Findings for other lanes and owners. None of these is a hybrid-scheduling
problem; all were found while establishing the cost model.

1. **`download_raw` ignores `set_features`.** `out_dev` is allocated and
   copied at `3 * n_features * n_bins` regardless of the active set, and
   `histogram_from_host` converts all of it. Under `feature_fraction=0.1` the
   GPU downloads and converts 90% zeros, on every node. Independent of this
   lane; owner is `histogram_gpu.mojo`.
2. **`Histogram.reset` has the symmetric issue.** It zeroes the whole buffer
   even when only the active feature slices will be written. Both fixed costs
   in the model are larger than they need to be, on both sides, and they
   partly cancel — which is why neither is a hybrid-scheduling argument.
3. **`SPLIT_SEARCH_DEVICE` builds both children.** `_grow_tree_gpu_device_search`
   builds and searches each child on the device rather than subtracting, which
   its own docstring flags as an untested tradeoff. It is also what makes
   hybrid scheduling structurally impossible on that path: no host parent
   exists to subtract from.
4. **The two growers' `_LeafState` and `_GpuLeafState` differ only in row
   representation** (`List[Int]` versus a device range) and are otherwise
   field-for-field identical, including the branch/depth/bounds bookkeeping.
   A hybrid grower is closer to a merge of the two than to a third thing.
5. **Per-node feature subsampling is consistent across backends and does not
   invalidate a histogram.** `feature_fraction_bynode` narrows the *search* on
   both paths, never the accumulation. This is load-bearing for the cache:
   were it otherwise, every node would need its own histogram key epoch.
6. **`partition_range_host` is already the tested host mirror of the device
   partition**, stable in buffer order. It is what makes `ROWS_HOST_MIRROR`
   sound, and it means the host-side row derivation needs no new algorithm.
7. **No semantic disagreement was found** between the CPU and GPU growers in
   split selection, routing, constraint handling, or row ordering. The only
   difference is arithmetic precision (Float64 versus dequantized Float32),
   which is documented in both modules and is exactly what §4's replica
   requirement is about.

---

## 9. What this lane deliberately did not build

- **No replica accumulation kernel.** It belongs in `histogram.mojo`, which
  this lane does not own. Specified in 7.2.
- **No per-range row download.** Belongs in `gpu_active_rows.mojo`. Specified
  in 7.1.
- **No thresholds, no default coefficients, no synthetic cost fixtures.** A
  fabricated coefficient set would make the module *look* live while producing
  placements nobody measured. `HybridCosts.unmeasured()` is the only fixture.
- **No `HybridCosts` for Apple M4 or any other part.** The M4 is the one
  device this project has read from hardware, and it has no histogram timing
  breakdown recorded anywhere in the repository.
- **No scheduler-owned histogram storage.** The frontier states already own
  the buffers; moving them into a cache would be an edit to two growers this
  lane must not touch. The cache is a ledger, in the same shape as
  `PoolLedger` and `ResidencyLedger` in `gpu_runtime.mojo`.
- **No overlap machinery.** `WaveBarrier` makes overlap *safe*; it does not
  make it happen. Whether to run the host build concurrently with device work
  is an integration decision that should follow experiment E5.

---

## 10. Review notes on the delivered code

Neither module has been compiled (the lane forbids running Mojo). Highest-risk
items for a first compile, in order:

1. `HybridCosts` has two `__init__` overloads — a no-argument non-raising one
   (unmeasured) and an argument-taking raising one (measured). If overload
   resolution on `__init__` is unhappy, split the second into a
   `@staticmethod measured(...)` and keep only the no-arg constructor.
2. `String` fields in `Copyable, Movable` structs with an explicit raising
   `__init__` — the pattern is copied from `CrossoverEvidence` in
   `device_policy.mojo`, which compiles today.
3. `List[CacheEntry]` mutation during the `drop_stale` sweep uses
   swap-with-last plus `pop()`, the same shape as `release`. Index handling was
   written to re-test the swapped-in element (the `i` counter does not advance
   on a drop).
4. `HistogramCache.within_bound` reads `self.entries[0].key` for the shape; it
   returns True on an empty cache before that read.
5. `_yes_no` is defined before its first use, and `_ceil_div` is local to the
   scheduler module (both modules deliberately avoid importing each other's
   helpers; the only cross-module import is the scheduler taking origin
   constants, `fixed_download_bytes`, `histogram_cells`, and `origin_name`
   from the cache-policy module).
6. `HistogramCache.drop_stale`, `HistogramCache.clear`, and `WaveBarrier.open`
   are marked `raises` defensively, because they call `List.clear()` /
   `List.pop()` and the repository has no non-raising precedent for either. If
   a compile shows those are non-raising, dropping the annotations is safe and
   makes the three callable from non-raising code.
7. Neither module imports anything from the GPU layer, so both are exercisable
   on a CPU-only machine and in CPU-only CI. `histogram_cache_policy.mojo`
   imports only `_env_int` from `parallel.mojo`; `hybrid_leaf_scheduler.mojo`
   adds `std.os.getenv`.

---

## 11. Future validation (no test files were written)

When a later lane is permitted to write tests, these are the cases with the
most information per assertion:

- `place_leaf` returns `PLACE_GPU` for every mode and every shape while
  `HybridCosts.unmeasured()` is in force, with `DECLINE_COSTS_UNMEASURED`
  once the run-level gates pass and the specific gate's reason before that.
- The decline ladder reports the *outermost* missing thing: device split
  search outranks device gradients outranks unverified replica outranks
  unmeasured costs.
- `HybridCosts(...)` raises without an `evidence_id`.
- `place_leaf` is order-independent: the same `LeafWork` list placed forward
  and backward gives identical placements.
- `WaveBarrier` raises on `require_ready` with one build outstanding, accepts
  completions in any permutation, and returns `announced()` unchanged by that
  permutation.
- `staleness` returns the right code for each of the four epoch counters moved
  singly, and `FRESH` for `set_features(False)`.
- `check_subtraction` refuses `ORIGIN_CPU_FLOAT64` against `ORIGIN_GPU_FIXED`
  and accepts `ORIGIN_CPU_REPLICA` against it.
- `sibling_key` produces the complement window on both sides (child at the
  parent's begin, and child at the parent's end).
- `capacity_bound_bytes` against a simulated 31-leaf growth trace that admits
  and releases in the grower's order and never exceeds the bound.

Benchmark experiments E1–E6 are specified in `docs/design/HYBRID_TRAINING.md`
§9, with E1 (coefficient calibration) and E2 (the bitwise replica claim) as
the two that gate everything else.
