# Task 16 handoff: sparse and categorical GPU primitives

Status: the primitives landed, nothing is wired up, nothing is enabled.
`device="gpu"` still refuses sparse input everywhere it refused it before,
and no file outside this lane changed.

Delivered in this lane, and nothing else:

- `src/mojoboost/gpu_sparse_layout.mojo` (861 lines) device layout, hard
  support limits, per-node cost model, node entry-window bookkeeping, entry
  tiling, and the EFB compatibility check. Pure host arithmetic, no
  `DeviceContext`, no allocation, no launch.
- `src/mojoboost/gpu_sparse.mojo` (1540 lines) seven kernels, the
  device-resident `GpuSparseHistogramBuilder`, and the host reference models
  the kernels are meant to be compared against.
- `src/mojoboost/gpu_categorical.mojo` (711 lines) the sparse categorical
  semantics and their checker, the raw-code to bin-set conversions, the
  device category-set pool and its routing kernel, and device-side per
  category statistics with a host reference.
- `docs/GPU_SPARSE_CATEGORICAL_DESIGN.md` the specification. Read it before
  this file; this file is the integration, that one is the contract.
- this file.

No tests were written and no command was run. This lane was static reasoning
only, by instruction, so **nothing below has been compiled or executed**.
Section 6 is the test suite the next lane owes, and section 8 puts compiling
these three files first for that reason.

Every cross-file claim below was checked against the working tree rather than
recalled. Line numbers are from the tree as of this lane.

One checkout note. This round shares a checkout across lanes, and a
concurrent integration commit (`b04b5f0`) picked up
`gpu_sparse_layout.mojo`, `gpu_sparse.mojo`, `gpu_categorical.mojo`, and
`docs/GPU_SPARSE_CATEGORICAL_DESIGN.md` at an intermediate state. The final
state of this lane is the working tree, which carries a small uncommitted
diff on the first two files (a borrow-safety refactor of the zeroing helper
and of two `self.method(self.field)` call sites). Read the working tree, not
`b04b5f0`.

---

## 1. What the primitives guarantee

Four facts the rest of this document leans on. All four are specified in the
design doc and in the module docstrings.

1. **A sparse GPU histogram is bit-identical to the dense GPU histogram** of
   the same data, gradients, and scale. Not close, identical. Everything
   accumulated is exact fixed-point Int32, and the default bin is
   `total - sum(stored)` computed from the same per-row quantization as the
   accumulation. This is stronger than the CPU sparse path, whose Float64
   subtraction agrees with the dense CPU histogram only to rounding
   (`histogram_sparse.mojo` says so in its own docstring).
2. **A node owns both a row range and a per-feature entry window**, and the
   two are partitioned by the same per-row side mask at every split. The
   subtraction in (1) is only sound while that invariant holds;
   `gpu_sparse.check_entry_row_consistency` is its statement.
3. **The row partition is the dense path's**, unforked.
   `GpuActiveRows.enqueue_partition` is called with the side buffer as a
   synthetic one-column binned matrix (feature 0, threshold bin 0, no missing
   bin), so `bin <= 0` reads as "side is left". Row order therefore matches
   the dense GPU path and the CPU grower's lists index for index.
4. **Absent means the value 0.0, and for a categorical feature that means
   category code 0.** When code 0 is not in the fitted table, absent rows
   land in the unknown bin and route right at every categorical node of that
   feature. `gpu_categorical.absent_is_unknown` reports it per feature.

Two deliberate restrictions:

- only `STRATEGY_ATOMIC` accumulation is implemented (design doc section 4);
- the last node id (`max_nodes - 1`) is reserved for a bagged root's
  out-of-bag entries and a split may not use it.

## 2. Memory complexity

Per session, allocated once at construction, never per node.
`SparseDeviceLayout` computes all of it.

| Buffer | Bytes |
| --- | --- |
| `row_index` | `4 * nnz` |
| `bin` | `1 * nnz` |
| `order` | `4 * nnz` |
| `scratch` | `4 * nnz` |
| `col_offsets` | `4 * (n_features + 1)` |
| `default_bin` | `1 * n_features` |
| `ranges` | `8 * max_nodes * n_features` |
| `side` | `1 * n_rows` |
| **total structure** | `13 * nnz + 5 * n_features + 8 * max_nodes * n_features + n_rows + 4` |

Against the dense builder's `1 * n_rows * n_features` for its bin matrix.
A stored entry costs 13 bytes here against 1 byte dense, so
`bytes_ratio() < 1` needs density below roughly `1/13`, before the `ranges`
term. `SparseDeviceLayout.bytes_ratio()` is the number to report, and it is
one of several reasons no threshold is derived from density alone.

The `ranges` term is the one that surprises: at `max_nodes = 2048` and
100k features it is 1.6 GB. **A wide sparse dataset must lower `max_nodes` to
its actual leaf budget.** That is a real limit of this design and section 9
records the alternative.

Shared with the dense path unchanged, and not counted above: gradients,
hessians, the `n_features * n_bins * 3` Int32 histogram, the active-row
permutation and its scan buffers, and the pinned staging buffers.

## 3. Work complexity

Per node, with `E` entries in the node, `R` rows, `A` active features, `B`
bins.

| | Dense GPU | Sparse GPU |
| --- | --- | --- |
| bin reads | `R * A` | `E` |
| gradient reads | `R * A` | `R + E` |
| launches | 1, or 2 tiled | 3 |
| host syncs | 1 | 1 |
| default-bin work | none | `A * B` adds, one threadgroup per feature |

Per split, with `N` the parent's rows and `P` its entries.

| | Dense GPU | Sparse GPU |
| --- | --- | --- |
| launches | 4 | 6 |
| element passes | `N` | `N + 2P` |
| host syncs | 0 with `expected_left` | 1, or 0 with `defer_ranges` |

Per tree: one `nnz` pass to reset the permutation, one `n_features` pass to
seed the root windows, plus one extra `nnz` partition when bagged.

Two costs have no dense counterpart and both are in the table above: the
`R`-length totals reduction per node, and the `n_features` Int32 range
download per split.

## 4. Required integration edits

None of these files were touched. Every one is central or shared, and each
edit belongs to whoever owns that file.

### 4.1 Module exports

`src/mojoboost/__init__.mojo` re-exports the GPU modules around lines 166 to
213 and the sparse modules around 254 to 285. Three blocks are missing:

```mojo
from .gpu_sparse_layout import (
    MeasuredCosts, SparseDeviceLayout, SparseRangeTable, SparseVerdict,
    check_bundle_compatibility, decide_sparse, sparse_support,
)
from .gpu_sparse import (
    GpuSparseHistogramBuilder, build_histogram_gpu_sparse,
    check_entry_row_consistency, partition_entries_host, side_mask_host,
)
from .gpu_categorical import (
    CatSetPool, CategoryStats, absent_is_unknown, cat_bitset_from_codes,
    category_stats_host, check_cat_bitset,
    check_sparse_categorical_semantics, codes_from_cat_bitset,
)
```

Adding them makes the primitives importable as `mojoboost.X`. It also puts
them in the API snapshot, so `tests/parallel/api_snapshot_manifest.json`
needs the same names. Do these together or the snapshot test fails.

### 4.2 The device policy still says the kernel does not exist

`src/mojoboost/device_policy.mojo:1034` blocks sparse input with
`BLOCK_SPARSE_INPUT` and the message

> sparse input trains on the CPU; there is no sparse GPU histogram kernel

The block must stay, because nothing is wired up, but the message became
false in this lane. The accurate replacement, which keeps the block and
stops promising a fact that no longer holds:

> sparse input trains on the CPU; the sparse GPU primitives exist but no
> training path is wired to them

`docs/DEVICE_SELECTION.md:73` carries the same block in its table and cites
`_fit_sparse` in `python/mojoboost/__init__.py` as the enforcer, which is
only half true: `device_policy.mojo` is the authority and the Python path is
downstream of it. Fix the citation while you are there.

### 4.3 A sparse GPU trainer

`src/mojoboost/train_gpu.mojo` mirrors the boosting loop for the dense GPU
path, exactly as `boosting_sparse.mojo` mirrors it for the CPU sparse path. A
sparse GPU trainer is the fourth mirror and it does **not** belong in either
existing file. The call sequence the builder already supports:

```
builder = GpuSparseHistogramBuilder(binned_csc, max_nodes)
per round:   builder.upload_gradients(g, h)
per tree:    builder.begin_tree(bag)
per node:    hist = builder.build_leaf(node)
per split:   builder.apply_split(feature, bin, parent, left, right,
                                 missing_bin, default_left,
                                 is_categorical, cat_bitset, expected_left)
```

`gpu_split_search.mojo` consumes the resulting `Histogram` unchanged; it does
not know how a histogram was built.

Not provided by this lane, and needed by that trainer: device-side leaf value
computation for sparse (the dense path's `update_raw_ranges` in
`gpu_objectives_native.mojo` works off leaf **row** ranges, which the sparse
path also maintains, so it should carry over unchanged; verify that before
assuming it).

### 4.4 Binning

`sparse.fit_bins_csc` and `sparse.transform_csc` already produce the
`SparseBinnedMatrix` the builder takes, on the CPU, in parallel across
features. No change is needed for the primitives. A GPU binner is a separate
lane and is not implied by anything here.

### 4.5 EFB

`efb.bundle_csc` output feeds the builder unchanged. What a caller must add:

- call `check_bundle_compatibility(caps, plan, source_cats)` before training,
  and refuse on `SPARSE_BUNDLED_CATEGORICAL`;
- report `recovery_is_exact` to the user, since a lossy plan makes
  `unbundle_histogram` approximate;
- run `efb.unbundle_histogram` per member per node before split search, on
  the host, since split search must stay per original feature.

`efb.mojo` itself needs no change.

## 5. Serialization and binding changes

**Model format: none, and none proposed.** These primitives produce a
`Histogram` and mutate index structures. They do not touch `Tree`, `Model`,
`MulticlassModel`, `serialize.mojo`, or `lgbm_model_io.mojo`, and a model
trained through them would be byte-for-byte the model the CPU sparse path
produces, exactly as `model_sparse.fit_csc` already promises. CPU prediction
is unchanged. If a later lane finds itself proposing a format change to make
sparse GPU work, that is a signal something has drifted from this design, not
a step in it.

**Bindings: none required, three additions if the path is ever exposed.**

1. `bindings/_mojoboost.mojo` already accepts CSC and CSR through the
   `sparse_data_addr` / `sparse_indices_addr` / `sparse_indptr_addr` /
   `sparse_nnz` key convention (`_csc` at line 223, `_sparse_shape` at 214).
   A sparse GPU fit needs no new marshalling at all: the same `CscMatrix`
   reaches the trainer, only the trainer differs. What it needs is a
   `device` argument on `fit_csc` / `fit_multiclass_csc`, which those
   bindings do not currently carry.
2. `python/mojoboost/device_selection.py` already sends a `sparse` flag
   across the boundary (line 496) and `_NarrowNativePolicy` already parses
   the refusal. When the block in 4.2 is lifted, the only change on the
   Python side is that `sparse-input` stops appearing in `_blocks`. No new
   field, no new contract version.
3. `python/mojoboost/__init__.py::_fit_sparse` (line 2902) is where an
   explicit `device="gpu"` for sparse input would stop being refused. It
   should keep refusing until the validation in section 8 has passed on the
   target device, and its refusal message should cite the same reason string
   the native policy uses so the two cannot drift.

**C API: none.** `capi/mojoboost_capi.mojo` exposes no sparse entry point
today, so there is nothing to update and nothing to break.

## 6. Adversarial correctness cases

The test suite this lane owes and did not write. Each one is a case where a
plausible implementation is silently wrong rather than loudly wrong.

**Sparse structure and semantics**

1. **Explicitly stored zeros.** Two matrices differing only in whether zeros
   are stored must give bit-identical histograms and bit-identical trees.
   This is the case that catches an accumulator that forgets to exclude
   stored default-bin entries from the leftover.
2. **A stored value that bins to the default bin but is not zero.** Same
   requirement, and it is not the same test: the first can pass by special
   casing the value 0.0.
3. **Stored NaN.** Must land in the reserved missing bin, must not be
   confused with an absent entry, and must follow `default_left` rather than
   the threshold. Assert the missing bin's count directly, not just the tree.
4. **A feature with no stored entries at all.** Every row is an implicit
   zero; the default bin must receive the whole node total and every other
   bin must be zero. The accumulation kernel is not launched for it, so this
   is the test that the default-fill kernel runs unconditionally.
5. **A feature stored for every row.** The leftover must be exactly zero, in
   all three planes. A sign error in the subtraction shows up here and
   nowhere else.
6. **An empty node.** Zero rows, zero entries; the histogram must be all
   zeros, including the default bins.
7. **A node whose rows all take the default side of a split.** One child gets
   everything, the other gets an empty range and an empty window; the empty
   child's next histogram must still be all zeros rather than stale.

**The two indexings**

8. **Entry and row windows must agree after every split.** Run
   `check_entry_row_consistency` after each split of a full tree. This is the
   single highest-value test in the list: every subtraction result depends on
   it and a violation produces plausible-looking wrong numbers.
9. **Stability.** After a split, download the row permutation and the entry
   permutation and compare against `partition_range_host` and
   `partition_entries_host` index for index, not as sets.
10. **Bagged root compaction.** With a bag, assert no entry of an out-of-bag
    row is in node 0's windows, and that the histogram equals the histogram
    of the bag-only submatrix. Without the compaction the default-bin count
    goes **negative**, which is the loud symptom to assert for.
11. **A bag in non-ascending order** (what GOSS produces). Row order must
    track the CPU grower's list, which is seeded from the same sample in the
    same order.

**Bit identity**

12. **Sparse against dense, same gradients, same scale.** Build the same
    dataset both ways and compare the raw fixed-point planes, not the
    Float64 histogram. Equality must be exact. Do it at the root and at
    several interior nodes of a grown tree.
13. **Feature subsampling.** With `set_features` narrowed, inactive features'
    slices must be exactly zero, default bins included, so sibling
    subtraction stays exact.
14. **`defer_ranges` on against off.** The histograms must be bit-identical;
    only the launch geometry differs. A mismatch means a kernel is trusting
    the host's window instead of the device's.

**Categorical**

15. **Category code 0 kept against dropped.** The same column, binned with a
    `max_bins` that keeps code 0 and one that drops it, must give different
    and individually correct routings. `absent_is_unknown` must report the
    difference.
16. **A category set containing bin 0.** Construct one by hand and assert
    `check_cat_bitset` raises. Then assert what the device would have done
    with it (every unknown row routes left), so the hazard is documented by a
    test rather than by a comment.
17. **A category code not in the fitted table**, passed to
    `cat_bitset_from_codes`. Must raise, not silently select bin 0.
18. **Round trip.** `codes_from_cat_bitset(cat_bitset_from_codes(codes))`
    must return `codes` sorted and deduplicated.
19. **Category statistics against the histogram.** For a categorical feature,
    `enqueue_category_stats` and the corresponding column of
    `_sparse_hist_kernel` must agree bit for bit.
20. **A categorical column reserving a missing bin.** Must be rejected by
    `check_sparse_categorical_semantics`. Construct it by hand; the binner
    does not produce it.

**EFB**

21. **A bundled matrix against its unbundled source.** Per member per node,
    `unbundle_histogram` of the bundled histogram must equal the unbundled
    histogram, exactly when the plan is lossless.
22. **A plan that bundles a categorical feature.** Construct it by hand and
    assert `check_bundle_compatibility` returns
    `SPARSE_BUNDLED_CATEGORICAL`.
23. **A lossy plan.** `recovery_is_exact` must be false and the recovered
    member histogram must differ from the unbundled one by exactly the
    collisions, in the member's default bin.

**Limits**

24. Every row of the design doc's section 11 table, asserted to raise.

## 7. Expected CPU fallback behavior

Unchanged in every case, and that is deliberate.

- `device="gpu"` with sparse input **still raises** with
  `BLOCK_SPARSE_INPUT`. Nothing in this lane relaxes it and section 4.2 keeps
  the block while fixing only its wording.
- `device="auto"` with sparse input **still takes the CPU**, for the same
  reason and by the same code path.
- The CPU sparse path (`model_sparse.fit_csc`, `boosting_sparse`,
  `tree_sparse`, `histogram_sparse`) is untouched and remains the only
  sparse training path.
- Once a sparse GPU trainer exists, the fallback contract should match the
  dense one, which `device.mojo` states plainly: explicit `"gpu"` raises
  rather than silently falling back, `"auto"` takes the CPU. Do not
  introduce a silent sparse-to-dense fallback either. Densifying behind a
  caller's back is the failure mode this whole lane is built to avoid, and a
  dataset that needs the sparse path is usually a dataset that will not fit
  dense.
- Every limit in design doc section 11 raises rather than degrading. A
  caller that wants to continue catches and chooses the CPU explicitly.

## 8. Staged integration and validation plan

Each stage is independently useful and independently revertable. Do not start
a stage before the previous one is green.

**Stage 0, compile.** These three files have never been through the
compiler. Build them and fix whatever the compiler says before reading
anything else in this plan. Expect the usual suspects: origin parameters on
the pointer-taking host helpers, the two-argument `__init__` overloads on
`GpuSparseHistogramBuilder`, and shared-memory `stack_allocation` sizes.

**Stage 1, host-only tests.** Everything in `gpu_sparse_layout.mojo` and the
host reference models in the other two files runs without a device:
`SparseRangeTable` splitting and its invariants, `partition_entries_host`,
`side_mask_host`, `check_entry_row_consistency`, `category_stats_host`, the
bitset conversions, and `check_bundle_compatibility`. Cases 1 through 7, 15
through 18, and 21 through 24 are reachable here. This is most of section 6
and it needs no GPU.

**Stage 2, device against host.** On an actual device, compare each kernel
against its host reference: the entry partition against
`partition_entries_host`, the side mask against `side_mask_host`, the
category statistics against `category_stats_host`. Cases 8 through 11 and 19.

**Stage 3, bit identity.** Case 12, the sharpest test available, plus 13 and
14. If stage 3 passes, the accumulation is correct.

**Stage 4, measure.** Design doc section 10, all five measurements. Publish
them in this handoff before anything consumes `decide_sparse`. Until then
`decide_sparse` returns `SPARSE_UNDECIDED` and every caller must handle it.

**Stage 5, a trainer.** Section 4.3, behind an explicit opt-in with no
default change. Compare a whole fitted model against the CPU sparse path's
on the same data.

**Stage 6, exposure.** Only now: the export block (4.1), the policy message
(4.2), the `device` argument on the sparse bindings (5.1), and finally
lifting the refusal in `_fit_sparse`. `device="auto"` should keep choosing
the CPU until stage 4's measurements say otherwise, matching how
`AUTO_MIN_CELLS` is already handled for the dense path.

## 9. Open questions and known weaknesses

1. **The `ranges` table scales with `max_nodes * n_features`** (section 2),
   which is the wrong shape for a wide sparse dataset. The alternative is a
   per-node table allocated on demand, or a compressed representation keyed
   by the live leaves rather than by node id. Neither is designed here.
   Interim mitigation: pass a `max_nodes` matching the actual leaf budget.
2. **`grid.y` is uniform across features**, so a node's tile count comes from
   its widest active column and narrower columns idle their tail tiles
   (design doc section 4). Datasets with a few dense columns among many
   sparse ones are the adverse case. A per-feature grid, or a flattened
   entry-major grid with a feature lookup, would fix it; both change the
   kernel's indexing and neither is measured.
3. **One host round trip per split** unless `defer_ranges` is set, which
   trades it for compounding grid over-provisioning. Which wins is stage 4's
   question and neither is the obvious answer.
4. **`STRATEGY_TILED` is not implemented** for the sparse accumulation. The
   partial buffer would be indexed by (entry tile, active slot, bin) exactly
   as the dense one is by (row tile, active slot, bin), and
   `gpu_active_rows._range_reduce_kernel` would combine it unchanged, since
   that reduction depends only on the partial layout. That is the shape; the
   justification for building it is a measurement, not this note.
5. **The bagged-root compaction reuses `_entry_partition_kernel` with
   `left == parent`.** The aliasing is safe because every thread reads the
   parent's window before the first barrier and only thread 0 writes, after
   the last. It is the one place in these modules where a correctness
   argument rests on barrier ordering rather than on disjointness, so it is
   the one place to re-read if the kernel is ever restructured.
6. **Nothing here has run.** Treat every performance statement in the design
   doc as a cost model and every correctness statement as a claim with a
   named test that does not exist yet.
