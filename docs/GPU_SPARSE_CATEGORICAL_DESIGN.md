# Sparse and categorical GPU primitives

This document specifies the device-side sparse and categorical primitives in
`src/mojoboost/gpu_sparse_layout.mojo`, `src/mojoboost/gpu_sparse.mojo`, and
`src/mojoboost/gpu_categorical.mojo`.

**Status: primitives only.** Nothing here is wired into training. There is no
`device="gpu"` path for sparse input, no automatic switch away from the dense
GPU builder, no threshold on density, and no densification anywhere. A
`SparseBinnedMatrix` goes to the device compressed and comes back as an
ordinary `Histogram`. The integration a training path would need is in
`handoffs/performance_16_sparse_categorical_gpu.md`.

The ordinary `Model` representation and CPU prediction are untouched. No
serialization format changes, and none is proposed below without saying so
explicitly.

## 1. Why a sparse device path at all

The dense GPU builder uploads the binned matrix as `n_rows * n_features`
bytes and every node's histogram reads `node_rows * n_active` bins. On the
data that motivates sparse storage (bag of words, one hot categoricals,
click logs) most of those reads return the same bin: the one holding 0.0.

The sparse path reads only the stored entries, and recovers the implicit
zeros by subtraction. Per node it costs `entries_in_node` bin reads plus one
sequential pass over `node_rows` gradients, against `node_rows * n_active`
random bin reads. Whether that trade wins is a measurement, and section 10
says which measurement.

## 2. Device layout

`gpu_sparse_layout.SparseDeviceLayout` is the accounting. The buffers are the
`SparseBinnedMatrix` of `sparse.mojo` with its index widths narrowed to what
the kernels index with, plus three index structures.

| Buffer | Type | Elements | Purpose |
| --- | --- | --- | --- |
| `row_index` | Int32 | nnz | stored entry to row |
| `bin` | UInt8 | nnz | stored entry to bin |
| `col_offsets` | Int32 | n_features + 1 | feature to entry range |
| `default_bin` | UInt8 | n_features | bin holding the implicit zero |
| `order` | Int32 | nnz | entry permutation, grouped by node |
| `scratch` | Int32 | nnz | scatter destination |
| `ranges` | Int32 | 2 * max_nodes * n_features | per (node, feature) window |
| `side` | UInt8 | n_rows | per-row split side |

A stored entry therefore costs 9 bytes here (4 + 1 + 4 for its slot in
`order`, and `scratch` shadows `order`) against 1 byte in the dense matrix.
`SparseDeviceLayout.bytes_ratio()` reports the crossover; it is reached well
below full density, which is one of the several reasons no automatic
threshold is derived from density alone.

`ranges` is sized by `max_nodes` at construction, so growing a tree allocates
nothing on the device.

### Two indexings, both device resident

A node owns

* a **row range** `rows[begin, end)` in the `GpuActiveRows` permutation the
  dense path already maintains, and
* an **entry window** `order[start_f, end_f)` per feature.

The two are partitioned by the same per-row mask at every split, so an entry
is in a node's window exactly when its row is in that node's range. That
invariant is what makes the subtraction in section 4 sound;
`gpu_sparse.check_entry_row_consistency` is its host-side statement, and it
is the first thing to check when a sparse histogram disagrees with a dense
one.

Both partitions are stable, so a node's rows come out in the order the CPU
grower's row list holds them, and a feature's entries stay in ascending row
order inside every window. The second property is what makes a child's window
a sub-window of its parent's.

## 3. Zero, stored zero, and missing

Unchanged from `sparse.mojo`, because the device holds the same numbers.

* An **absent** entry is the value `0.0`. It bins to `default_bin[f]`, it
  participates in quantile edges like any other value, and it routes by the
  threshold like any other value. It is never visited by a kernel; it arrives
  in the histogram through the leftover.
* An **explicitly stored zero**, and any stored value that happens to bin to
  `default_bin[f]`, is accumulated as a stored entry into that same bin and
  then excluded from the leftover. Storing it or dropping it produces the
  identical histogram, bit for bit. `efb.bundle_csc` drops such entries and
  is therefore free to.
* A stored **NaN** is a missing value. It bins to the feature's reserved
  missing bin, which is never a default bin, and routes by the node's learned
  `default_left`. Absence is not missingness and there is no way to express
  missingness by leaving an entry out.

`zero_as_missing` is not implemented in mojoboost and no argument accepts it.

## 4. The histogram

Three kernels per node, one download.

1. `_sparse_totals_kernel` sums `round(g * scale)`, `round(h * scale)`, and
   `1` over the node's row range into three Int32 cells.
2. `_sparse_hist_kernel` accumulates the node's stored entries into their
   bins. `grid.x` is the active feature slot and `grid.y` a tile of that
   feature's entry window; each threadgroup keeps a shared-memory Int32
   partial and folds it into the output with global integer atomics.
3. `_sparse_default_fill_kernel` gives each active feature's default bin the
   leftover, `total - sum(stored)`, with a read-modify-write because the
   default bin may already hold stored entries that binned there.

### Bit identity with the dense GPU histogram

Stronger than the CPU sparse path can claim. `histogram_sparse.mojo` notes
that its default bin, derived by subtraction over Float64, agrees with the
dense histogram only to rounding. Here every accumulated quantity is an exact
fixed-point Int32, and the same per-row quantization feeds the totals kernel
and the accumulation kernel, so for feature `f`

```
default_cell = total - sum over stored entries of f in this node
             = sum over rows of this node with no stored entry for f
```

exactly. Counts are exact integers on both sides, and integer addition is
associative, so neither the atomics nor the tile decomposition can change a
result. A sparse GPU histogram is therefore **bit-identical** to the dense
GPU histogram of the same data with the same gradients and the same scale.

That equality is the sharpest available correctness test and it is what the
validation plan in the handoff leans on.

Overflow: the fixed-point scale bounds the magnitude sum of all scaled values
by `2^30`, so `total` and `sum(stored)` are each within that, and their
difference is a sum over a subset of the same rows and so also within `2^30`.
No intermediate leaves Int32.

### Determinism

Every accumulated value is an integer, every partition destination is a pure
function of position and exact integer prefix sums, and no float is ever
atomically accumulated. Run to run, worker count to worker count, and tile
count to tile count, the results are identical.

### One accumulation strategy

Only `STRATEGY_ATOMIC` is implemented. The dense path's tiled strategy exists
to stop many row tiles contending on the same output bins; the sparse path's
tiles are entry tiles, which are far fewer per feature, and the hottest bin
by far (the default one) is never touched by the accumulation at all. It is
filled once, in a separate reducing kernel. A partial-buffer variant is
a measurement away, not a design change.

### Launch geometry constraints

Three constraints the kernels impose on their launchers, all enforced by
`GpuSparseHistogramBuilder` and all easy to get wrong from a second caller.

* The two reducing kernels (node totals, default-bin completion) keep three
  Int32 planes per thread in shared memory and are sized for
  `REDUCE_MAX_THREADS = 256`. At the 1024 the partition uses, the request
  would be 12 KB per threadgroup, which is most of what a conservatively
  reported device offers, so both are clamped.
* A device is cleared against `sparse_kernel_shared_bytes`, the largest
  request any of the four kernels makes, **not** against
  `gpu_tiling.shared_bytes_for(n_bins)`. The latter is the wrong figure twice
  over: a `stack_allocation` takes a compile-time extent, so the accumulation
  kernel asks for `3 * 256` Int32 whatever `n_bins` is (192 bytes claimed
  against 3072 actual at `n_bins = 16`), and the entry partition asks for
  more than any of them with a size that does not depend on `n_bins` at all.
  Both bounds are defined in `gpu_sparse_layout.mojo` and aliased into
  `gpu_sparse.mojo`, so the check and the allocations cannot read different
  numbers.
* The totals kernel writes three cells and nothing else, so its block count
  *is* the atomic contention on those cells. It is capped at a few waves
  (`TOTALS_BLOCKS_PER_SM`) and each thread walks the row range with a grid
  stride, which is exact whatever order it covers rows in. The kernel takes
  the block count as an argument for that reason and must be launched with
  exactly that many blocks.
* The default-bin completion runs one threadgroup per active slot, not one
  thread. A thread-per-feature version issues every one of its
  `n_slots * 3 * n_bins` loads from a single thread at a stride of `n_bins`,
  entirely uncoalesced, and on the wide data this path exists for that can
  cost more than the accumulation it completes.

### Known weakness: uneven columns

`grid.y` is uniform across `grid.x`, so the tile count is derived from the
largest active feature's entry window
(`SparseRangeTable.max_entries`). A feature with far fewer entries leaves its
tail tiles with nothing to do. The waste is proportional to how uneven the
column occupancies are and it is the first thing to profile on real data.

The entry partition has the matching weakness: it launches one threadgroup
per feature at every split, including the many features with no entries in
that node at all, which read two Int32 and write five. Both are the price of
a uniform per-feature grid and both are fixable by the same change, an
entry-major grid with a feature lookup, which is not measured and so is not
made.

## 5. Applying a split

A sparse split routes rows through the split feature's stored entries alone:
every row of the node takes the side of `default_bin[f]` unless it has a
stored entry for `f`. So the side is materialized first, one byte per row.

1. `_sparse_side_default_kernel` writes the default side over the node's row
   range.
2. `_sparse_side_entries_kernel` overwrites the rows that do have a stored
   entry for the split feature, applying `RowRouting`'s rule (set membership
   for a categorical node, the learned default direction for the missing bin,
   the inclusive threshold otherwise).
3. The **row** range is partitioned by `GpuActiveRows.enqueue_partition`,
   unchanged and unforked. The side buffer is handed to it as a synthetic
   one-column binned matrix: feature 0, inclusive threshold bin 0, no missing
   bin, so the rule `bin <= 0` reads exactly as "side is left". The sparse
   path inherits the dense path's stable, deterministic, already validated
   partition instead of a second copy of it.
4. The **entry** windows are partitioned by `_entry_partition_kernel`, one
   threadgroup per feature, two chunked block scans and a fold-back.

The kernel publishes each feature's midpoint into the device range table for
both children, and into a `mids` buffer.

### The one extra host round trip

The host mirrors the device windows so it can size `grid.y`. By default a
split downloads `n_features` Int32 and synchronizes. That is the single cost
the sparse path pays that the dense path does not.

`GpuSparseHistogramBuilder.defer_ranges = True` removes it: both children
inherit the parent's window as an upper bound, the device windows stay exact,
the kernels read the device windows, and histograms are unchanged. Only the
launch geometry is over-provisioned, and the over-provisioning compounds with
depth. Which of the two wins is, again, a measurement.

## 6. Bagging and GOSS

A bagged root gets its row range from the bag, but its entry windows would
otherwise start as whole columns, including entries of rows the bag left out.
Accumulating those while the totals cover the bag alone makes the leftover
wrong, possibly negative.

`begin_tree` therefore runs one extra partition at the root, by bag
membership, and keeps the in-bag half. Cost is one pass over `nnz` per tree.
The out-of-bag entries are parked in a reserved node id (`max_nodes - 1`) and
are unreachable from every live node afterwards. A split may not use that id.

This covers GOSS as well, which arrives as the same kind of row subset, in
the caller's order.

## 7. Categorical features

### Absent means category code 0

For a categorical feature `sparse.default_bins` gives the bin of the value
`0.0`, which `CategoricalSpec.bin_of` resolves two ways.

* **Code 0 was kept.** The default bin is that category's own bin. Every row
  without a stored entry is a row of category 0, which is what it is, and
  those rows join whichever side of a split category 0 lands on.
* **Code 0 was not kept**, because the column never held it or it was dropped
  as too rare for `max_bins - 1`. The default bin is `UNKNOWN_BIN`, bin 0.
  Bin 0 is never a member of a split's category set, so every absent row
  routes right at every categorical node of that feature, along with the
  missing, unseen, and dropped rows.

Both are correct. The second is a trap, because a one-hot-ish column whose
zeros mean "not this category" behaves completely differently depending on
whether 0 survived the table, and no number says so.
`gpu_categorical.absent_is_unknown` answers it per feature and
`check_sparse_categorical_semantics` answers it for a matrix, returning the
flagged feature ids so a caller can report it rather than discover it in a
fitted model.

`check_sparse_categorical_semantics` also enforces two structural rules: a
categorical column's `default_bin` must equal the bin of code 0, and a
categorical column must reserve no missing bin (it routes missing to bin 0 by
construction, and a reserved bin would give it a second, contradictory
route).

### Bin 0 must never enter a category set

The device routing rule tests the bitset bit for a row's bin with **no**
special case for bin 0. A set with bit 0 raised sends every missing, unseen,
and dropped row left, silently reversing the documented default. The searches
never produce such a set, but a set can also arrive from a caller, a
deserialized model, or a language binding, so `cat_bitset_from_codes` refuses
to build one (it raises on a code the fitted table does not hold, rather than
mapping it to bin 0) and `check_cat_bitset` refuses to accept one.

`codes_from_cat_bitset` is the inverse, and is the form a binding or a model
dump wants: bin ids are an artifact of the fitted table, codes are what the
caller supplied.

### Device-side category sets

`CatSetPool` holds many 256-bit sets on the device in the same layout
`gpu_predict.mojo` already uses for a model's sets: `CAT_BITSET_WORDS` words
per set, and a slot's absolute word offset is what a kernel indexes with. A
grower can then route a categorical split through `_cat_pool_side_kernel`,
which reads the set from the pool instead of carrying four `UInt64` through
the launch arity. Same rule, same result as the categorical arm of
`_sparse_side_entries_kernel`; the two are interchangeable.

Sets are built on the host, where the searches produce them, and uploaded
whole. They are never edited in place on the device.

### Device-side category statistics

`enqueue_category_stats` produces one categorical feature's per-bin
`[grad | hess | count]` at one node, straight from the compressed column,
with the implicit zeros folded into the default bin by the same subtraction.
It takes raw pointers rather than a builder, so it composes with whatever
owns the buffers, and it allocates nothing.

The output layout is exactly what `categorical.find_best_categorical_split`
reads as the `base = 0` slice of a histogram, and it is exactly one column of
the histogram `_sparse_hist_kernel` builds, so the two agree bit for bit.

`category_stats_host` is the Float64 reference and is usable on its own.

### The split search is not here

`gpu_split_search.mojo` already runs LightGBM's one-vs-rest and sorted
category searches on the device, over a histogram, and neither knows nor
cares whether that histogram came from a dense matrix or a compressed one. A
sparse histogram feeds it unchanged. Nothing in this lane duplicates it.

## 8. EFB compatibility

`efb.bundle_csc` returns an ordinary `SparseBinnedMatrix`, so a bundled
matrix trains through these primitives with no change at all: the kernels are
bin-agnostic and a bundle's shared bin 0 is just that column's `default_bin`.
The compatibility question is entirely about what happens after the histogram
comes back.

* **Recovery stays host-side.** `efb.unbundle_histogram` reconstructs a
  member's local histogram from the bundle's block, recovering the member's
  default bin by subtracting the member's own range from the block total.
  That is the same trade the sparse accumulator already makes, applied twice.
* **Recovery is exact iff the plan is lossless.** A collision the member lost
  is folded into that member's default bin. No device change undoes it;
  `BundleCompatibility.recovery_is_exact` reports it.
* **Categorical features are never bundled.** `efb.mojo` guarantees this
  unconditionally, and the guarantee is what keeps a singleton's category
  table and its reserved unknown bin valid under identity encoding.
  `check_bundle_compatibility` checks it rather than assuming it, because a
  bundled categorical column would be silently wrong: its bins would be
  offset into a shared range, bin 0 would stop being the unknown bin, and
  every unseen category would start routing left.
* **A multi-member bundle's dropped default-bin entries cost nothing.**
  `bundle_csc` drops entries sitting at a member's default bin, and the
  sparse accumulator recovers them through the bundle's default bin, so the
  bundled matrix and the unbundled one give the same numbers for the same
  rows.

## 9. Complexity

Per node, with `E` stored entries in the node, `R` rows in the node, `A`
active features, `B` bins.

| Quantity | Dense GPU | Sparse GPU |
| --- | --- | --- |
| bin reads | `R * A` | `E` |
| gradient reads | `R * A` | `R + E` |
| kernel launches | 1 (2 tiled) | 3 |
| host syncs per node | 1 | 1 |
| device work for the default bin | none | `A * B` adds, one threadgroup per feature |

Per split, with `F` features, `N` the parent's row count, `P` the parent's
entry count.

| Quantity | Dense GPU | Sparse GPU |
| --- | --- | --- |
| kernel launches | 4 | 6 |
| element passes | `N` | `N + P` (two passes over `P`) |
| host syncs | 0 with `expected_left` | 1, or 0 with `defer_ranges` |

Per tree: one `nnz` pass to reset the entry permutation, one `n_features`
pass to seed the root windows, and, when bagged, one extra `nnz` partition.

Memory: section 2. The whole structure is allocated once per session.

## 10. What must be measured before a policy exists

`gpu_sparse_layout.decide_sparse` returns `SPARSE_UNDECIDED` unless the
caller hands it per-unit costs it actually measured, and it stays undecided.
There is no default threshold, no density heuristic, and no automatic
fallback on this module's authority, because the units differ by orders of
magnitude across devices and a constant baked in here would become exactly
the unmeasured threshold this design refuses to ship.

The measurements a policy needs, in order:

1. **Per-node accumulation.** Sparse against dense `build_leaf` on one
   dataset at several densities, at several node sizes, holding features and
   bins fixed. Gives `ns_per_bin_read` and `ns_per_row_read`.
2. **Launch and synchronization cost** on the target device, so the three
   launches and the one range download can be weighed against the dense
   path's one launch. Gives `ns_per_launch` and `ns_per_host_sync`.
3. **`defer_ranges` on against off**, over trees of realistic depth, since
   the trade is a host round trip against compounding grid over-provisioning.
4. **Column-occupancy skew**, since `grid.y` is uniform across features.
   Datasets with a few dense columns among many sparse ones are the adverse
   case.
5. **Whole-training wall clock**, not per-node, because the once-per-session
   upload and the per-split partitions are not in the per-node number.
6. **The geometry constants this lane picked without measuring.**
   `TOTALS_BLOCKS_PER_SM` (the node-total reduction's block cap) and
   `REDUCE_MAX_THREADS` (the reducing kernels' block size) were chosen to
   match the conventions already in `gpu_tiling.mojo`, not from a
   measurement. Neither can change a result, so neither is urgent, but both
   are knobs and should be swept once there is a harness to sweep them with.
   The rest of the launch geometry is not in this category: coalescing the
   default-bin read and capping the totals grid have no crossover to find,
   which is why they were changed without a measurement.

Only after 1 through 6 does a threshold belong anywhere, and it belongs in a
policy module with the measurements cited, not in these primitives.

## 11. Unsupported combinations, explicitly

Every one of these raises. None of them degrades silently, densifies, or
drops a feature.

| Combination | Where it is rejected |
| --- | --- |
| more than 256 bins | `sparse_support` |
| more rows, features, or entries than an Int32 index holds | `sparse_support` |
| device shared memory too small for this module's widest kernel | `sparse_support` |
| a categorical feature with at least `n_bins` categories | `check_categorical_support` |
| a categorical feature with more than 255 categories | `check_categorical_support` |
| a categorical feature reserving a missing bin | `check_sparse_categorical_semantics` |
| a categorical column whose default bin is not code 0's bin | `check_sparse_categorical_semantics` |
| a stored bin past the fitted category table | `check_sparse_categorical_semantics` |
| a category set containing bin 0 | `check_cat_bitset` |
| a category code not in the fitted table, in a split set | `cat_bitset_from_codes` |
| a categorical feature inside a multi-member EFB bundle | `check_bundle_compatibility` |
| a stored entry naming a row or bin outside the matrix | builder construction |
| a node id at or past the reserved id | `apply_split` |
| `max_nodes < 2` | builder construction |

Not supported and not rejected because it cannot arise here: `device="gpu"`
for sparse input is not exposed at all.

## 12. Relationship to the CPU sparse path

`histogram_sparse.mojo` and `tree_sparse.mojo` are unchanged and remain the
only sparse training path. The device primitives mirror their structure
deliberately, entry permutation for entry permutation and window for window,
so the two can be compared directly during integration:

* `SparseEntryOrder` maps to `order` plus `scratch`;
* `SparseNodeEntries` maps to one node's slice of `ranges`, and the host
  mirror in `SparseRangeTable` reuses the same struct;
* `build_histogram_sparse_node` maps to the three kernels of section 4;
* `_partition_ranges` maps to `_entry_partition_kernel`, and
  `partition_entries_host` is the serial reference both are compared against.

The CPU path remains the fallback for every case section 11 rejects, and for
every device the primitives have not been validated on.
