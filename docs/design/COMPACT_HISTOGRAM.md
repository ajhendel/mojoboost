# The compact histogram accumulator

`MOJOTREES_CPU_PACKED_HIST=1`. Built 2026-08-18. **Measured a LOSS, and NOT
bit-identical, both against the prediction written before the run.** Kept
off, and kept in the tree as the thing the next attempt starts from. Read
section 5 before section 1.

## 1. What is wrong with the rectangular histogram

A `Histogram` is `n_features * n_bins` cells whatever the data does with them,
flattened as `f * n_bins + b`. Every consumer indexes it that way, so the
shape is load-bearing for the split scan, the sibling subtraction, the GPU
download and about thirty test helpers.

The cells any row can actually reach are far fewer. `feature_bins[f]` is the
highest bin id observed for feature f plus one, so a binary column reaches two
cells and reserves 255. On covertype, 464,809 x 54 with 44 binary columns:

    rectangular   54 x 255 = 13,770 cells x 24 bytes = 322.7 KB
    compact       sum_f feature_bins[f] ~ 2,400 cells =  56.2 KB

The M4 has 64 KB of L1 data cache per core. The rectangular histogram misses
it by 5.0x and the compact one fits.

Two things follow, and the second is the one that surprised us. Packing does
NOT change which logical cells a row touches, so the reuse per touched cell is
identical; what changes is their ADDRESSES. Rectangular, consecutive features
are 4,080 bytes apart and every one of a row's 54 updates lands on its own
cache line. Compact, forty-four binary columns sit four to a 64-byte line, so
one line serves several features and the whole working set stays resident
across rows instead of being evicted between them.

## 2. Why it should matter at small nodes and not at large ones

The accumulate is partitioned by FEATURE. At a large node the pool's tasks
each walk a slice of the columns, and a slice fits L1 whether or not it is
packed. One task walks all 54 columns only when the node is too small to be
split across the pool, which is exactly where the 2026-08-18 phase profile put
the cost: the per-slot rate degrades **19.4x** from root nodes to tiny ones,
and 62.0 percent of the covertype CPU round is in this accumulate.

`plan_row_block_count` stops row-blocking below 8,160 rows, which at 464,809
rows and a complete depth-8 tree falls between depth 5 and depth 6. So 192 of
every 255 splits land in the classes this is aimed at.

## 3. Why the packing itself is bit-identical, and why the switch is not

The same Float64 additions happen in the same order, at different addresses.
`_expand_packed_histogram` then copies each feature's `feature_bins[f]` cells
from `bin_offset[f]` to `f * n_bins` and zeroes the bins beyond that width.

Those zeroed bins are ones no row can reach, since `feature_bins[f]` is the
observed maximum plus one. They are still written rather than left alone,
because the split scan reads all `n_bins` of them and a stale value there
would be a phantom bin.

The expansion is one sequential write-only pass per node, replacing a
scattered read-modify-write over five times the footprint.

## 4. What it is exclusive with, and why

**The blocked arm, and this is what made the switch move bits.**
`MOJOTREES_CPU_ROW_BLOCKS` is documented in `apple_cpu_policy` as one that
"moves bits", because a block count is a summation order. Forcing blocking off
therefore changes the model on every node the planner would have blocked, and
that is a property of this switch and not of the packing.

`_accumulate_blocked_at` keeps private partial histograms
in the TAIL of the caller's scratch at `part_off`, sized from
`n_features * n_bins` and indexed rectangularly. None of that holds for a
compact buffer, so `_accumulate_subset` forces the flat ladder when packed.
Combining them means packing the partials too, which is the row-major blocked
kernel's existing trick and a separate change.

**`_zero_excluded`.** Skipped when packed, equivalently rather than as an
omission: the compact buffer is freshly zeroed per node, so an excluded
feature is already zero and the expansion copies that zero out. Running it
would also be a wild write, since it indexes `f * n_bins` into a buffer with
no such cell. **This was the segfault the first version produced**, and it is
recorded because the failure mode of a layout change is a wild write and not a
wrong number.

## 5. What was measured

`bench/real_data/arms_packed_hist.py`, interleaved arms with a LightGBM drift
canary in every cell block, three whole-process repeats, real covertype and
real year at two shapes.

**The prediction was written before the run.** Covertype should win because 44
of its 54 columns are binary; year should NOT, because its 90 dense continuous
columns each use most of their 255 bins, so its compact footprint is close to
its rectangular one and the switch only adds the expansion. If year wins too,
the footprint mechanism is wrong.

RESULTS, run `20260818T185452Z-packed`, median of three, end to end
(binning plus training), spreads in the run record and LARGE:

    scenario            shape     baseline   packed   ratio
    covertype           deep        32.157   44.855   0.717x
    covertype           31 leaves   18.848   22.860   0.824x
    year                deep         9.836   10.029   0.981x
    year                31 leaves    4.875    5.054   0.965x

**A loss everywhere, and biggest exactly where the prediction said it would
win.** Prediction digests: identical on year at both shapes, DIFFERENT on
covertype at both.

Those two facts are the same fact. Covertype is the shape the planner row
blocks; year at these shapes is not. So on covertype the packed arm was not
packing-versus-baseline, it was packing-AND-no-blocking versus baseline, and
it paid for losing the blocking. Year, where nothing was blocked either way,
isolates the packing plus the expansion and reads 0.965x to 0.981x, which is
about the cost of the expansion pass and the per-node allocation.

**So this run does not test the hypothesis.** It prices the confound. What it
does establish is that the expansion and the allocation together cost roughly
2 to 4 percent, which is the floor any real win has to clear.

The next measurement adds `MOJOTREES_CPU_ROW_BLOCKS=1` as its own arm, with no
packing. Then packed-versus-blocks=1 isolates the packing and should be
bit-identical, and blocks=1-versus-baseline prices the blocking separately.

## 6. Known cost not yet removed

The compact buffer is allocated per node rather than carried on the caller's
scratch the way `pairs` is. If the switch wins it should win by more once that
allocation is hoisted; if it loses, the allocation is the first thing to rule
out before the mechanism is.
