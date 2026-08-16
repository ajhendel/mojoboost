# The blocked bin layout: what the GPU histogram reader wants, and what it is worth

Lane K3, Phase 2. Nothing in this document has been measured. Every number
carries its provenance, and the estimates exist so a measurement can refute
them rather than be fitted to them.

The code is `src/mojotrees/gpu_blocked_bins.mojo` (geometry, host reference,
device transform), the reader arm in `gpu_active_rows._hist_rows_step` and
`_hist_accumulate_rows`, and `tests/test_gpu_blocked_bins.mojo`.

---

## 1. The contract, stated from the reader's side

This is the half a layout lane implements against. The histogram kernel reads
a bin at

```
    offset(f, r) = (f / G) * block_stride + (f mod G) + r * G
    block_stride = align_up(n_rows * G, 16)
```

over a feature axis padded up to a multiple of `G`, one `UInt8` per cell, and
the buffer's size is `(n_blocks - 1) * block_stride + n_rows * G`. That is
`gpu_binned_layout.BinLayoutPlan` built over the padded axis at width 8 with
`target_block = G` and promotion on, and `check_blocked_matches_plan` proves
the closed form and the plan agree cell for cell, which is the only claim in
this lane that could be wrong silently.

Six requirements, in the order they bind:

1. **`G` must equal `GpuActiveRows.feature_group`.** Not "should". The
   mechanism is that a threadgroup consumes every one of the `G` adjacent
   bytes it pulls for one row, and a threadgroup consumes `feature_group`
   feature slots. Both directions of mismatch produce a *correct* histogram
   at a cost nobody chose, so neither would be found by any test or any
   measurement. `check_blocked_group_matches` refuses it.
2. **`G` is a rung of the feature-group ladder** (1, 2, 4, 8, 16), because
   each rung is a separate kernel instantiation and a `G` off the ladder
   could be laid out and never consumed.
3. **The feature axis is padded to a multiple of `G`.** A short final block
   would need `lane_of` and `block_of` as tables instead of a division and a
   remainder, and the kernel cannot walk a table per row. The pad columns are
   written as zero and never read: a feature id reaching the row loop comes
   from `feat_ids` and is below `n_features`.
4. **Blocks are 16-byte aligned**, which is `BinLayoutPlan`'s rule and not
   this lane's. It is why `block_stride` is not simply `n_rows * G`, and it
   is the term a reimplementation is most likely to drop, since it is
   invisible on any shape where `n_rows * G` happens to be a multiple of 16.
5. **Bin ids are stored, never renumbered.** Width 8 throughout, so
   `missing_bin`, the categorical bitsets, and every `threshold_bin` keep
   their meaning without a check. `gpu_binned_layout.check_markers_preserved`
   is the check that would be needed if a narrower width were ever used, and
   this layout deliberately uses none.
6. **The feature-major buffer stays.** This layout is an *addition*, not a
   replacement. `_flag_scan_kernel` and `_scatter_kernel` in
   `gpu_active_rows.mojo`, `gpu_predict`, `gpu_sparse`, `gpu_categorical`,
   `gpu_leaf_batching` and the host binning path all index
   `bins[f * n_rows + r]`, and a half-converted matrix is a much worse state
   than either layout on its own: the two agree until a lane forgets one of
   them, and then a histogram is wrong on some shapes and not others.

The transform that produces this buffer is one enqueued device pass per fit
(`enqueue_blocked_relayout`), reading the feature-major buffer already on the
device. It is deferred to the first histogram launch so that requesting a
layout and growing no tree costs an allocation and no device work.

---

## 2. Why this layout and not the two obvious alternatives

### Not compression

Per `(row, feature)` the row loop moves one bin byte, one Int32 row index and
an eight-byte quantized gradient pair. The bin is one byte of thirteen.
Halving it to four bits removes under four percent of the traffic and adds a
shift and a mask per visit. `gpu_binned_layout` reaches the same conclusion
from the same arithmetic and says so.

### Not full row-major, and this is now a cross-campaign question

`binning.BinnedMatrix` is gaining a row-major multi-value view alongside the
feature-major one, owned by the CPU campaign. **The GPU histogram builder
cannot use it, and the reason is threadgroup memory rather than taste.**

A threadgroup accumulating `k` features needs `3 * k * bin_cap * 4` bytes of
shared memory. At 256 bins that is 3 KiB per feature, so an Apple M4's 32 KiB
budget holds at most 10 feature slots and the ladder caps at 8. A row-major
record of a 50-feature matrix is 50 bytes wide, of which a threadgroup can
consume at most 8: it would pull the whole record and discard 84 percent of
it, and it would do so once per feature block, re-reading the same rows
`ceil(50 / 8)` times. That is strictly worse than feature-major, which at
least reads only the columns it wants.
`gpu_binned_layout.max_block_for_shared` is the same bound, and
`candidate_plans` already declines to emit a row-major plan unless one block
of every feature fits shared memory.

So the GPU's answer to "which layout does your builder read" is:

- **the histogram builder reads feature-*blocked* at `G = feature_group`**,
  which at the shipping default (`feature_group = 1`) **is** feature-major,
  byte for byte;
- **the predictor wants row-major** and always did. `gpu_predict` walks many
  features of one row and is the one GPU consumer a row-major view helps.
  `gpu_binned_layout`'s opening paragraphs name this split: training reads
  one feature across many rows, prediction reads many features across one
  row, and the two want opposite layouts.

A row-major view built by the CPU campaign is therefore useful to this
backend at prediction time and not at histogram time, and nothing about it
should be routed into the GPU histogram path on the grounds that it exists.

---

## 3. Where the win is, and where it provably is not

Write `S` for the device's memory sector and `count` for a node's rows. A
node's rows stay ascending (`gpu_active_rows` partitions stably), so a node of
`count` rows out of `n_rows` reads a subset with average stride
`n_rows / count`.

**At the root and in the large classes the layout is neutral, not a small
win.** The active rows are the identity permutation or close to it, so a
feature-major column is read contiguously by consecutive threads and touches
exactly its own bytes. The blocked buffer holds the same bytes in a different
arrangement and streams exactly as many. There is nothing to save.

**Below `count < n_rows / S` every read is its own sector whatever it wanted
from it**, and that is where the arrangement pays. Per `(row, feature)`, in
that regime, at `GROUP = G = 4`, counting sectors (**derived**, from the model
in `gpu_binned_layout.sectors_touched`, not measured):

| | bin sectors | row index | gradient pair | total |
|---|---|---|---|---|
| feature-major, group 4 | 1 | 0.25 | 0.25 | 1.5 |
| blocked, `G` = group = 4 | 0.25 | 0.25 | 0.25 | 0.75 |

**2.0x on memory sectors, in the scattered regime only.**

The crossover lands in a specific size class, and this is the sharpest
falsifiable prediction the lane makes. At 1,000,000 rows:

| sector | crossover count | fraction of root | class |
|---|---|---|---|
| 32 B | 31,250 | 1/32 | inside `medium` |
| 64 B | 15,625 | 1/64 | the `medium`/`small` boundary |
| 128 B | 7,812 | 1/128 | inside `small` |

**So the win must begin at the medium/small boundary, give or take one class
depending on the cache line, which nobody here has measured**
(`docs/design/CLEANSHEET_GPU.md` section 8 lists it as unknown). If a
measurement shows the win concentrated in `root` or `large`, the mechanism
above is not what produced it and the reading should be distrusted rather
than banked.

Two things push the built-node mix toward the classes where this pays, and
one pushes the other way:

- **Sibling subtraction builds the smaller child.** The histograms actually
  built are systematically the small side of every split, so the built-work
  mix is shifted toward `small` and `tiny` relative to the node census.
- **Leafwise growth** puts most nodes deep.
- **Feature subsampling pushes back.** A threadgroup's slots are
  `feat_ids[slot0 .. slot0 + G)`, and under `colsample` those need not be the
  `G` lanes of one block: active features 0, 5, 9, 12 at `G = 4` sit in four
  blocks at four lanes, so the block issues four scattered loads exactly as
  feature-major would while occupying four times the bytes. Correctness is
  unaffected -- an address is computed from a feature id -- but the win is.
  `blocked_alignment_fraction` reports the share of an active set that lands
  in whole blocks, so a run can say which regime it was in instead of
  assuming.

---

## 4. What it costs

**The transform.** One kernel launch per fit. Reads `n_rows * n_features`
bytes, writes `n_rows * padded_features`. At 1,000,000 x 50 with `G = 4`:
50 MB read, 52 MB written. Against the 75-85 GB/s device copy rate this
project **measured** on an Apple M4 on 2026-08-15, that is a **derived bound**
of about 1.3 ms, or under 0.06 percent of a 2.58 second fit. It amortizes over
roughly 6,100 node histograms in a 100-round fit, so its per-node share is not
worth carrying in a cost model.

**Residency, which is the cost that matters.** The feature-major buffer stays
(requirement 6), so the device holds `n_rows * (n_features + padded_features)`
bytes of bins instead of `n_rows * n_features`: **102 MB instead of 50 MB** at
the reference shape. On a device where the binned matrix is a large share of
the working set, that is the term that decides, and it is why the buffer is
allocated on request rather than at construction.

**Per visit, in the row loop.** One integer multiply (`r * rstride`) that the
feature-major arm does not perform, because its stride is one. Per owned slot
and never per row: one integer division and one remainder to form the column
base. If the layout ever loses, that multiply is the first thing to suspect,
and `rstride` being a power of two is what should make it a shift.

---

## 5. Exactness

**This cannot change a histogram, by construction.** The transform writes
`blocked[offset(f, r)] = bins[f * n_rows + r]` for every cell, and the row
loop reads `offset(f, r)` where it read `f * n_rows + r`. The same byte is
fetched from a different address; the same bin then selects the same shared
cell, the same three quantized values are added to it, and the set of
`(row, feature)` visits, the tiling, the slot assignment, the flush and the
sibling subtraction are all untouched.

This is a **stronger** claim than the row unroll's or `GROUP`'s. Those reorder
integer adds and rely on addition being associative and commutative. This arm
does not reorder them at all. There is no floating-point arithmetic anywhere
on the path -- the quantized arm has none in the row loop by construction, and
the Float32 arm's `Int32(round(x * scale))` is untouched.

**Bits do not move.** `tests/test_golden_bits.mojo` needs no regeneration for
this lane. The arm is off by default; even on, it produces the identical Int32
histograms.

The two executable forms of the argument are
`gpu_blocked_bins.blocked_roundtrips` (the blocked buffer decodes to the dense
matrix, cell for cell) and `check_blocked_matches_plan` (the kernel's address
is the priced plan's address). The failure mode they guard is the one nothing
else would catch: a wrong offset reads a **legal bin id belonging to a
different row**, so the histogram is well formed, the split is plausible, no
bound is violated and no tolerance is exceeded.

---

## 6. How to measure it, and the arm that is easy to get wrong

The layout is selectable at **run time in one binary**
(`GpuActiveRows.set_blocked_layout`), in the style of `set_row_unroll`, so the
arms interleave in one process, which is the only protocol that compares
anything on this machine.

**Three arms, and the middle one is not optional:**

| arm | `set_feature_group` | `set_blocked_layout` |
|---|---|---|
| A: shipping default | 1 (whatever `free_feature_group` gives) | off |
| B: group only | 4 | off |
| C: group + layout | 4 | 4 |

**C against B is the layout.** C against A is the layout plus the group
change, and the group change is separately worth 1.17x (group 2 over 1,
atomic, `bench-hist 100000 100 20`) and 1.39x (group 2 over 1 at 5M x 50),
both **measured** on an Apple M4 and both recorded in
`_range_hist_atomic_kernel`'s docstring. **Reporting C against A and calling
it the layout would attribute a measured group effect to an unmeasured layout
effect**, which is this repository's standing failure mode wearing a fifth
costume.

Report **by node size class** (`phase_profile.classify_node`), not in
aggregate, and report the isolated histogram figure alongside the in-run one.
The registered rule is that the in-run figure decides. Note that here the
isolated instrument is biased *low*, not high: a full-width root-sized
histogram benchmark is precisely the shape section 3 says the layout is
neutral on. That is not a reason to trust the isolated number in the other
direction either; it is a reason to expect the two instruments to disagree and
to say which is which.

Also report `blocked_alignment_fraction` for the active feature set, so a
subsampled run is labelled rather than assumed.

---

## 7. The estimate, revised DOWN by its own author before any measurement

Registered in `bench/results/PHASE2_PREREGISTRATION.md`: **1.5-2.5x on the
histogram phase**, refuted below 1.2x by size class or if the fit moves less
than 0.15 s at 1,000,000 x 50.

**Revised: 1.0x at the shipping default by construction, and 1.10-1.20x on
the histogram phase at `feature_group = 4`.** That is at or below the
refutation threshold, so **this lane predicts its own refutation under the
registered rule**, and the revision is recorded before the arms are run so
that the eventual measurement is compared against the honest prediction.

Three reasons, in descending order of how much they move the number.

**1. At the configuration this project benchmarks, the layout is a provable
no-op.** `gpu_tiling.free_feature_group` returns 1 at `bin_cap = 256`, because
three Int32 planes of 256 cells is 3 KiB per slot and a second slot doubles
the threadgroup footprint. Every headline shape here (1,000,000 x 50 at 255
bins) runs at `feature_group = 1`, where `G = 1` **is** the feature-major
layout. Not "probably small" -- identical. Reaching the win requires widening
the group first, which is a change to modules this lane does not own and a
residency trade `free_feature_group` states in as many words is unmeasured on
every device this project runs on.

**2. The three shared atomics are untouched, and they bound the win from
above.** K1's audit named two costs in the inner loop, the scattered gather
*and* the three shared atomic `fetch_add`s per `(row, feature)`. This lane
addresses exactly one of them. If atomics are a fraction `a` of the phase, the
layout's ceiling is `1 / (a + (1 - a) / 2)` even where the sector model gives
its full 2.0x. At `a = 0.4` that is 1.43x, in the scattered classes only.
**Nobody here has measured `a`**, and it is the single largest reason the
registered 1.5-2.5x is likely too high. It is also the more promising target:
shared-atomic conflict is attacked by privatization or by
`std.gpu.primitives.warp` aggregation, neither of which is a layout change and
neither of which this batch contains.

**3. The neutral classes carry real weight.** Root and large are 1.0x by the
argument in section 3, so the phase-level figure is a mix. Taking 1.43x on the
scattered classes and 1.0x elsewhere, with sibling subtraction shifting maybe
45 percent of built row-work into `small` and `tiny`, gives
`1 / (0.55 + 0.45 / 1.43) = 1.16x` -- **an estimate built on two unmeasured
inputs**, the atomic fraction and the built-work mix, and quoted only so that
missing it is visible.

### Is it a null?

**At the shipping default: yes, structurally, and that is a complete result.**
It is not a prediction that could be wrong; `G = feature_group = 1` makes the
blocked reader compute the feature-major address.

**Conditional on `feature_group >= 2`: no, but the honest expectation is
1.1-1.2x on the phase, concentrated entirely in `small` and `tiny`.** The
mechanism is real, the sector arithmetic is sound, and the change is exact and
cheap. It is simply not the largest thing left, and the registered estimate
said it was.

**What this lane would do instead, if the sequence were free**: measure the
shared-atomic fraction of the histogram phase first, because it decides both
the ceiling on this lane and whether the next lane should be attacking
atomics rather than addresses. That measurement is small -- an arm that
accumulates into shared memory without the atomic (producing a wrong
histogram, run once, never shipped, and gated so it cannot reach a model) --
and it would have re-priced this whole lane before it was written.

---

## 8. What was deliberately not built

- **A comptime kernel variant.** The reader is a block-uniform runtime branch
  on one Int32 launch argument, which is what makes the arms interleavable in
  one process. A comptime variant would have doubled a forty-instantiation
  family and made the only available protocol on this machine impossible.
- **A `narrow` twin of the blocked address arm.** `narrow` is registered by
  its own author as an expected null against three shared atomics and a
  scattered gather. A fourth address arm to measure a null inside an arm is
  not worth the code.
- **A host packing pass on the training path.**
  `gpu_binned_layout.pack_binned_matrix` already produces this buffer, and
  using it would charge `n_rows * n_features` host writes and a second upload
  of the whole matrix. The device pass charges neither. The host version stays
  as `blocked_relayout_host`, which is the reference a test compares against
  and the half of CI an Apple M4 structurally cannot run.
- **A shared-memory tiled transpose in the relayout.** The pass is once per
  fit and already under a millisecond at the reference shape by the bound in
  section 4. A two-times improvement on it is invisible next to the arm spread
  on this machine, and it would be a second piece of untested address
  arithmetic guarding the same invariant.
- **Any change to the `feature_group` default.** That is `gpu_tiling.mojo` and
  `histogram_gpu.mojo`, and it is an unmeasured residency trade. This lane
  reports that it is the binding precondition; it does not take the decision.
- **Any change to the packed (sub-byte) family.** `gpu_bin_packing.mojo` and
  the packed corners of `gpu_binned_layout.mojo` are untouched. Compression is
  a poor bet on its own by section 2, and its real argument -- that a narrower
  width lets a block be wider under the same shared-memory budget -- only
  becomes interesting after the group question in reason 1 above is settled.

---

## 9. The packed-pair accumulator: the row bound is not what blocks it

Added after the registration's "no int16 packing" exclusion was withdrawn.
The withdrawal is correct as far as it goes and the correction it makes is
worth keeping: **the `2^(W-1) - 2` ceiling of 32,766 is a per-node bound, and
applying it to a 1,000,000-row *fit* was a category error.** Most nodes in a
real tree are far below it, sibling subtraction builds the smaller child, and
a per-node bound is a *selection condition* rather than a refutation. The
overflow-carry objection dissolves with it: a carry into the neighbouring
field matters only if the field overflows, and under a respected bound it does
not, so exactness would need no modular-arithmetic argument at all. (That
argument stays load-bearing for the unpacked Int32 case, which has no per-node
guard.)

**A second objection survives the correction, it is dispositive, and this
repository has already measured it.** The row bound is necessary and not
sufficient, because a bin's sum is bounded by the *gradient mass* in the bin
and not by the row count, and our scale is per **tree**, not per node.

`docs/design/ACCURACY_BUDGET.md` states the shipped rule and the measurement.
`SCALE_MAGNITUDE_SUM` sets `units = 2^30 / sum_dataset|g|`, so any partial sum
over the whole dataset is bounded by `2^30` and a single bin overflows int16
once it holds `32767 / 2^30 = 3.05e-5` of the total gradient magnitude --
three thousandths of one percent. That is not a theoretical bound.
**Experiment D, in that document: accumulating a 1,000,000-row node's 255-bin
histogram at the shipped global scale overflows int16 in 172 of 255 bins, and
at 100,000 rows in 228 of 255.** One hundred thousand rows is already three
times *below* the 32,766-row ceiling and it overflows worse, because a smaller
node holds fewer rows but not proportionally less gradient mass -- boosting
concentrates residual magnitude into exactly the nodes a tree keeps splitting.
ACCURACY_BUDGET's own summary is that "there is no node shape on which the
current scale and an int16 accumulator coexist".

So the blocking constraint is **the scale, not the row count**, and no
per-node row test can rescue it. The two rules that could:

| family | rule | int16-safe node rows |
|---|---|---|
| magnitude-sum, rescaled per node | `units = 2^k / sum_node\|g\|`, k = 14 | 32,766 |
| per-row clamp (LightGBM's) | clamp each row to `[-B/2, B/2]`, B = 4 | 16,383 |
| per-row clamp, B = 16 | | 4,095 |

Both make int16 a **node-size-dependent representation**, which is precisely
why LightGBM promotes bit width per leaf
(`cuda_single_gpu_tree_learner.cpp` calls `GetHistBitsInLeaf` for every
histogram it builds, and its mixed-width sibling subtraction exists for the
same reason).

Neither is available to this lane:

- **Rescaling per node** means requantizing the node's rows before each
  histogram, which is a pass over the node's rows -- the same order of work as
  the histogram itself. It would also change every quantized value, so it is
  an accuracy change and not a launch-shape change; under the speed mandate
  bit-identity is negotiable and accuracy is not.
- **Adopting the per-row clamp** is a change to `quantized_gradient.mojo`'s
  scale policy, which this lane does not own, and it points against the
  numerics settlement of 2026-08-16 that put the CPU default on LightGBM's
  Float32/Float64 precision and made Int32 fixed-point cells opt-in and
  deferred.

**Verdict: not built, and the reason is recorded rather than the option
dropped.** What would unlock it is a per-row-clamp quantization rule, and if
that ever lands, the design is already determined by three facts in the
existing code:

1. **The selection must be device-side, not host-side.** Under the default
   resident plane, `enqueue_desc_histogram` builds every non-root histogram
   and the host does not know the node's row count -- that is why that path is
   pinned to the atomic strategy. The kernel already reads the count as
   `desc[STEP_BUILT_COUNT]`, so a block-uniform branch on it against the bound
   is available and costs one scalar test per threadgroup. A host-side
   selection would silently reach the root and nothing else, which is the
   defect the row-tile arms were found in.
2. **The packing must stay inside the threadgroup accumulator.** The flush
   unpacks and writes the same three Int32 planes, so `gpu_split_search`, the
   reduction, the sibling subtraction and the download are all untouched, and
   a mixed-width tree needs no mixed-width subtraction. That containment is
   what makes the variant a one-kernel change instead of a plane-layout
   change.
3. **Its reach is `small` and `tiny`, the same classes as the layout.** At
   1,000,000 rows a 16,383-row bound is below a sixty-fourth of the root, so
   the variant fires exactly where section 3 says the blocked layout fires and
   nowhere else. Two optimizations aimed at the same classes do not compose
   into a phase-level 2x; they compete for the same fraction of the work.

The end-to-end rule stands whatever the arithmetic says: a phase win the fit
does not show is the row-tile floor again, and this variant would have to
clear a whole fit at 1,000,000 x 50 to ship.

---

## 10. The tile-floor loss is not a serial reduction: checked, and refuted

The proposed mechanism was that our per-tile flush might be a serial reduction
where LightGBM's is a parallel range reduction, which would mean the earlier
row-tile experiment confounded the tile direction with the reduction shape.
**It does not hold. Every reduction on this path is already parallel over bin
ranges**, in the four places it could have been otherwise:

| site | shape |
|---|---|
| `_range_hist_atomic_kernel`, shared zeroing | `b = tid; while b < span; b += block_dim.x` |
| `_range_hist_atomic_kernel`, flush | `c = tid; while c < nb; c += block_dim.x`, one thread per bin cell |
| `_range_hist_partial_kernel`, partial write | same, one thread per bin cell |
| `_range_reduce_kernel` | `i = global_idx.x`, **one thread per output cell**, each summing `n_tiles` partials |

The last is the one that matters and it is LightGBM's shape exactly: the grid
is `n_planes * n_slots * n_bins`, so bins are spread across threads and the
per-thread loop is over tiles, not over bins. There is no thread anywhere on
this path that owns a whole histogram.

**Consequence: the row-tile arms are testing what they are thought to be
testing**, and the tile-floor loss needs its already-registered explanation
rather than a new one. That explanation is partial *traffic*, which the
preregistration names as the known confound and which the code supports: the
partial buffer is `n_tiles * n_planes * n_slots * n_bins` Int32, so a
device-wide floor of 80 tiles at 50 features and 256 bins writes and reads
back **12.3 MB per node histogram** (**derived**, from the allocation
formula), against a node's own bin traffic of at most `count * n_slots` bytes.
At 50 features that is the term that grew 22 percent slower and at 100
features 36 percent, and it grows linearly in the tile count, which is what a
floor raises. Reduction *shape* was never in it.

One thing the reduce kernel does carry that is worth a look if the tile
question is reopened: each thread's `n_tiles` reads are strided by
`n_planes * n_slots * n_bins` words, so a tile-major partial layout gives the
reduction a stride of tens of kilobytes. Transposing the partial buffer to
bin-major would make each thread's tile walk contiguous. That is a change to
the partial layout shared by `_range_hist_partial_kernel` and
`_range_reduce_kernel` and by nothing else, so it is contained -- but it is a
tiling lane's change, not this one's, and it is unmeasured.
