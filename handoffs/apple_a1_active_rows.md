# Apple A1: device-side active-row compaction

Status: implemented and tested in isolation. Nothing central was edited.

Files added by this lane:

- `src/mojoboost/gpu_active_rows.mojo`
- `tests/parallel/test_gpu_active_rows.mojo`
- `handoffs/apple_a1_active_rows.md` (this file)

Focused test run (Apple M4, accelerator present):

```
MOJOBOOST_NUM_WORKERS=1 nice -n 19 tools/with_build_lock.sh \
  pixi run mojo run -I src tests/parallel/test_gpu_active_rows.mojo
16 tests run: 16 passed, 0 failed, 0 skipped
```

Six of those sixteen compile and launch the new kernels on the GPU; the other
ten are host-side and run on any machine. No other test, suite, benchmark, or
build was run.

## What the module provides

`gpu_active_rows.mojo` holds one device-resident permutation of the row
indices in which every live leaf owns a contiguous half-open range.

| Symbol | Role |
| --- | --- |
| `LeafRange` | `[begin, end)` slice of the active-row buffer |
| `LeafRangeTable` | node id -> range, plus `check_invariants()` |
| `RowRouting` | the `Tree.goes_left` rule, host and device |
| `partition_range_host` | serial reference model for the device partition |
| `GpuActiveRows` | the device buffers, kernels, and range bookkeeping |

Device kernels: `_iota_kernel`, `_flag_scan_kernel`,
`_scan_block_sums_kernel`, `_scatter_kernel`, `_copy_back_kernel`,
`_zero_int32_kernel`, `_range_hist_atomic_kernel`,
`_range_hist_partial_kernel`, `_range_reduce_kernel`.

A split is four launches: flag + block scan, block-sum scan, scatter,
copy-back. Every destination index is a pure function of an element's
position and exact Int32 prefix sums, so the partition is stable and
bit-deterministic; no atomic decides where a row lands.

## Replacing the all-row leaf-id filtering path, step by step

Every step below is a change to a central file, for the single integration
owner of the Mojo training path. They are ordered so the tree stays buildable
after each one.

### Step 1 — construct the row buffer alongside the histogram buffers

`src/mojoboost/histogram_gpu.mojo`:

```mojo
from .gpu_active_rows import GpuActiveRows, RowRouting
```

Add one field to `GpuHistogramBuilder`:

```mojo
    var rows: GpuActiveRows
```

and initialize it in `__init__`, after `self.caps` and `self.tiling` are
resolved and after `self.part_capacity` is set:

```mojo
        self.rows = GpuActiveRows(
            self.ctx, data.n_rows, data.n_features, data.n_bins, self.caps
        )
```

`GpuActiveRows` takes the builder's own `DeviceContext`, so nothing is
uploaded twice and no second device is opened.

### Step 2 — seed the root from the compacted buffer

Replace the body of `GpuHistogramBuilder.begin_tree` with:

```mojo
    def begin_tree(mut self, bag: List[Int] = []) raises:
        self.rows.begin_tree(bag)
```

`GpuActiveRows.begin_tree` validates the bag exactly as the current code
does, writes the identity permutation with a kernel when the bag is empty
(so an unbagged tree costs no host-to-device row transfer at all), and stages
the bag in the caller's order when it is not.

The `OUT_OF_BAG` sentinel becomes dead: an out-of-bag row is simply not
inside the root range, so no kernel iterates it. Delete `comptime OUT_OF_BAG`
and the `leaf_dev` field, its `enqueue_create_buffer`, and its
`enqueue_memset`.

### Step 3 — partition through the range table

Replace the launch inside `GpuHistogramBuilder.apply_split` (keep every
existing validation) and add one argument:

```mojo
    def apply_split(
        mut self,
        feature: Int,
        threshold_bin: Int,
        parent: Int,
        left: Int,
        right: Int,
        missing_bin: Int = -1,
        default_left: Bool = False,
        is_categorical: Bool = False,
        cat_bitset: CatBitset = cat_empty(),
        expected_left: Int = -1,
    ) raises:
        ...  # existing range checks, unchanged
        var routing: RowRouting
        if is_categorical:
            routing = RowRouting.categorical(feature, cat_bitset)
        else:
            routing = RowRouting.numerical(
                feature, threshold_bin, missing_bin, default_left
            )
        _ = self.rows.partition(
            self.bins_dev.unsafe_ptr(),
            parent,
            left,
            right,
            routing,
            expected_left,
        )
```

`expected_left = -1` (the default, and what any caller that has no count
passes) downloads the device count and synchronizes. Passing the count keeps
the split fully enqueued, which is what the grower should do; see step 5.

### Step 4 — build histograms over the node's range only

Replace the body of `GpuHistogramBuilder.enqueue_leaf` with a per-node
tiling and one call:

```mojo
    def enqueue_leaf(mut self, leaf: Int) raises:
        if not self.has_gradients:
            raise Error("call upload_gradients before build_leaf")
        var n_slots = len(self.active)
        var tiling = self.rows.range_tiling(
            self.caps,
            leaf,
            n_slots,
            self.tiling.strategy,
            self.part_capacity,
        )
        self.rows.enqueue_range_histogram(
            tiling,
            leaf,
            self.bins_dev.unsafe_ptr(),
            self.grad_dev.unsafe_ptr(),
            self.hess_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            self.out_dev.unsafe_ptr(),
            self.part_dev.unsafe_ptr(),
            n_slots,
            Float32(self.g_scale),
            Float32(self.h_scale),
        )
```

`range_tiling` derives the launch geometry from the rows the node actually
owns rather than from `n_rows`, which is the second half of the win: a small
node gets a small grid instead of a dataset-sized one. It re-derives through
the same `derive_tiling` policy and is capped by `self.part_capacity`, so the
partial buffer allocated at construction is never exceeded and never
reallocated.

`download_raw` and `histogram_from_host` are unchanged: the output layout,
the fixed-point scale, and the `[grad | hess | count]` planes are all the
same.

### Step 5 — hand the grower's exact left count to the partition

`src/mojoboost/train_gpu.mojo`, in `grow_tree_gpu`, the `builder.apply_split`
call already sits after `_count_left`:

```mojo
        builder.apply_split(
            split.feature,
            split.bin,
            parent_node,
            left_node,
            right_node,
            split_missing_bin,
            split.default_left,
            split.is_categorical,
            split.cat_bitset,
            expected_left=n_left,
        )
```

That is the only change to `train_gpu.mojo`. `_count_left` is exact (it sums
the parent histogram's integer counts under the same routing rule), so no
download is needed to know where the children's ranges start, and a split
adds no host synchronization.

### Step 6 — delete the superseded kernels

From `histogram_gpu.mojo`, delete `_hist_leaf_kernel`,
`_hist_partial_kernel`, `_partition_kernel`, and `_hist_reduce_kernel`.

`_hist_reduce_kernel` has a byte-identical copy in `gpu_active_rows.mojo` as
`_range_reduce_kernel`, because the reduction depends on the partial layout
only, not on which rows produced it. It has to live in `gpu_active_rows`
rather than be imported from `histogram_gpu`: after step 1 `histogram_gpu`
imports `gpu_active_rows`, so the reverse import would be a cycle. If a
future refactor wants one copy, move it to `gpu_tiling.mojo`, which both
already import.

### Step 7 — documentation that goes stale on this change

These are wrong the moment step 4 lands and are outside this lane's paths:

- `histogram_gpu.mojo` module docstring: the paragraph beginning "Every
  node's histogram is built by scanning all n_rows and filtering on the leaf
  id" and the bagging paragraph that describes the `OUT_OF_BAG` parking.
- `train_gpu.mojo` module docstring: "a per-row leaf-assignment array (row ->
  current leaf node id)" and the `begin_tree` bagging note.
- `docs/LIGHTGBM_PARITY.md`: any row citing the all-row scan as the GPU
  histogram cost model.
- `src/mojoboost/parallel.mojo`'s `MOJOBOOST_` environment contract: add
  `MOJOBOOST_GPU_VERIFY_ROWS` (see below).
- `README.md` GPU section, if it quotes the per-tree bin-read cost.

## New environment variable

`MOJOBOOST_GPU_VERIFY_ROWS=1` sets `GpuActiveRows.verify_counts` at
construction. When on, a `partition` given an `expected_left` downloads the
device's own count and raises if the two disagree, which is the check that
the routing rule the host counted with is the routing rule the device
applied. It costs one host synchronization per split, so it is off by
default; turn it on in CI and during the first integration. The field is also
settable directly (`builder.rows.verify_counts = True`).

## Invariants the integration must not break

1. **Stability.** A compacted range is the CPU grower's row list for the same
   node, index for index. `tests/parallel/test_gpu_active_rows.mojo` asserts
   this against `tree.partition_rows` for numerical splits, both missing-bin
   directions, and categorical set splits. Any change to the scan or scatter
   has to keep it.
2. **Tiling.** Live ranges tile `[0, n_active)` with no gap or overlap;
   `LeafRangeTable.check_invariants()` states it and the tests call it after
   every split.
3. **One owner per range.** `LeafRangeTable.split` refuses to hand a range to
   a leaf id that already owns one, so a wiring mistake in node numbering
   raises instead of silently orphaning rows.
4. **Node ids are dense and ascending.** The table is a `List` indexed by node
   id. `grow_tree_gpu` already allocates ids that way through
   `tree._add_node`; a grower that reuses or sparsely allocates ids would
   waste table entries (correctness is unaffected).
5. **Subtraction trick unchanged.** The grower still builds the smaller child
   and subtracts for the sibling. Under compaction that is strictly better
   than before: the direct build now costs the smaller child's rows rather
   than the whole dataset.

## Benchmark counters to add (not run by this lane)

No benchmark, profiler, or timing run was performed here, and no speedup is
claimed. The counters below are what a later, explicitly requested run on an
idle machine should report, in `bench/bench_train_gpu.mojo` and
`bench/bench_histogram.mojo`:

Work actually done

1. `rows_scanned_per_tree`: sum over built nodes of that node's range count.
   The current path's value is `nodes_built * n_rows`; the ratio of the two
   is the compaction win and should be reported directly.
2. `bin_reads_per_tree`: `rows_scanned_per_tree * n_active_features`.
3. `mean_node_fraction`: mean of `node_count / n_active`, which predicts the
   win before it is measured (deep leaf-wise trees make it small).
4. `nodes_built_directly` vs `nodes_subtracted`, so the interaction with the
   subtraction trick is visible.

Time

5. `partition_ms_per_split`, split into the four launches (flag+scan,
   block-sum scan, scatter, copy-back), so a scan that dominates a small node
   is visible.
6. `hist_enqueue_ms_per_node` against node row count, which under the current
   path is flat in node size and under compaction should be linear.
7. `host_syncs_per_tree`. Expect one per node histogram download and zero
   extra from splits when `expected_left` is supplied; with
   `MOJOBOOST_GPU_VERIFY_ROWS=1` expect one more per split.
8. `begin_tree_ms`, bagged and unbagged, since the unbagged path became a
   kernel and the bagged path is one `n_rows` Int32 copy.
9. End-to-end `train_gpu` wall clock against the CPU trainer at 100k, 1M, and
   10M rows on the M4, against the recorded 0.56x-at-1M baseline.

Memory

10. `device_bytes_added`: `3 * n_rows * 4` (rows, scratch, offsets) plus
    `(max_blocks + 1) * 4`, minus the `n_rows * 4` the deleted `leaf_dev`
    used. Net is about `+2 * n_rows * 4` bytes, 8 MB at 1M rows.
11. `host_pinned_bytes_added`: `2 * n_rows * 4` for the staging and download
    buffers, both of which exist for tests and bagging rather than for the
    hot path.

Sanity

12. Trained-model agreement between the compacted and filtering paths on the
    same seed. Histograms are the same integers, so predictions should match
    bit for bit, not merely to a tolerance. Worth running once during
    integration before the old kernels are deleted.

## Known risks

- **Scan block size coupling.** `_scatter_kernel` reads its block offset by
  `block_idx.x`, so it must be launched with the block size
  `_flag_scan_kernel` used. `enqueue_partition` uses one `self.block_threads`
  for both; keep it that way if the launches are ever refactored.
- **Single-block second scan.** `_scan_block_sums_kernel` runs one
  threadgroup that loops over the block sums in chunks. At 1M rows and 256
  threads that is 16 iterations, which is negligible, but a device with a
  much smaller block size or a much larger dataset would want a second level.
  It is correct at any size; only the constant factor changes.
- **Per-node tiling.** `range_tiling` re-derives the geometry for every node.
  That is pure host arithmetic (`derive_tiling`), a few dozen operations, but
  it now happens once per node instead of once per tree.
- **`build`/`build_histogram_gpu` one-shot path.** Both go through
  `begin_tree()` then `build_leaf(0)`, so they keep working with the root
  range covering every row. Their tests are `test_gpu_strategies.mojo` and
  `test_histogram_reference.mojo`; run those first during integration.
- **Not integrated, not benchmarked.** Nothing in `train_gpu.mojo` or
  `histogram_gpu.mojo` calls this module yet, so no end-to-end GPU training
  has run on the compacted path and no performance claim can be made from
  this lane.
