"""Device layouts for binned data, and the cost model that compares them.

Pure host arithmetic. Nothing here opens a `DeviceContext`, allocates a
device buffer, or launches a kernel, so every layout decision is reasonable
about without a GPU, the way `gpu_tiling.mojo` and `gpu_sparse_layout.mojo`
are. This module consumes `gpu_bin_packing.mojo` (bit primitives), reads
`binning.mojo` and `categorical.mojo`, and produces the
`BinStorageDescriptor` declared in `gpu_histogram_specializations.mojo`; none
of them consume it, and it edits nothing they own.

What is on the device today
---------------------------
`GpuHistogramBuilder` uploads `BinnedMatrix.bins` verbatim: one `UInt8` per
cell, feature-major, `bins[f * n_rows + r]`. Every kernel that reads a bin
reads it that way:

    histogram   col = f * n_rows;  bin = bins[col + rows[begin + j]]
    partition   bin = bins[f * n_rows + row]
    predict     bin = bins[node_feature * n_rows + row]

The first two are the training path and read *one feature across many rows*.
The third is the prediction path and reads *many features across one row*.
Those two access patterns want opposite layouts, which is the first thing a
single layout cannot win.

The layout family
-----------------
Every layout in this module is the same construction with different
parameters: features are partitioned into **blocks**, each block is stored
**row-major inside the block**, and blocks are laid out one after another.
Inside a block every feature is packed at one common bit width, so a block is
a single packed stream and one formula addresses the whole family:

    index(f, r) = r * G[b] + lane(f)          b = block_of[f]
    base(f)     = block_offset[b]
    width(f)    = block_width[b]

with `gpu_bin_packing`'s byte and shift formulas doing the rest. The named
layouts are the corners of that space:

    FEATURE_MAJOR_U8      G = 1 for every block, width 8 for every block
      block 0 = feature 0        block 1 = feature 1
      +---------------------+---------------------+
      | r0 r1 r2 ... rN-1   | r0 r1 r2 ... rN-1   |   1 byte per cell
      +---------------------+---------------------+
      This is exactly today's buffer: width 8 needs no pad and this plan
      takes no alignment, so `block_offset[f]` is `f * n_rows` and the
      "packed" buffer is `BinnedMatrix.bins` itself. `is_passthrough`.

    ROW_MAJOR             one block, G = n_features
      row 0                 row 1
      +--------------------+--------------------+
      | f0 f1 f2 ... fF-1  | f0 f1 f2 ... fF-1  |
      +--------------------+--------------------+

    FEATURE_BLOCKED       G features per block, blocks in feature order
      block 0 (f0..f3)                        block 1 (f4..f7)
      +----------------+----------------+---   +----------------+
      | r0: f0 f1 f2 f3| r1: f0 f1 f2 f3| ...  | r0: f4 f5 f6 f7|
      +----------------+----------------+---   +----------------+

    ...PACKED             the same three with width < 8 where the feature's
                          bin count allows it. A 4-bit block puts two
                          features of one row in one byte; a 3-bit block puts
                          two features and two thirds of a third.

The row-side argument this module used to make, and why it is dead
------------------------------------------------------------------
Per (row, feature) the histogram kernel moves, in the worst case,

    1 byte   the bin
    4 bytes  the row index `rows[begin + j]`
    8 bytes  the quantized gradient pair `gq[2 * r]`, `gq[2 * r + 1]`

This module used to argue from those numbers that blocking is the whole
game: `grid.x` was the feature, so the twelve row-side bytes were re-read
once per feature, and a block of `G` features would read them once and spend
them `G` times, taking the per-(row, feature) cost from thirteen bytes to
`w/8 + 12/G`.

That argument is dead, and the module must not be read as still making it.
`gpu_active_rows._range_hist_atomic_kernel` is now parameterized on `GROUP`:
one threadgroup owns `GROUP` feature slots, loads `rows[begin + j]` and the
gradient pair once, and spends them `GROUP` times. **The row-side
amortization a blocked layout was going to buy is already bought, by a launch
shape, on the layout that ships.** `histogram_gpu` takes the rung from
`gpu_tiling.free_feature_group`, which is 8 at 64 bins and 2 on Metal at 256,
and `GpuActiveRows.set_feature_group` moves it at run time. A blocked storage
layout adds nothing to that term. What is left for it is the bin gather
alone, which is the smaller half of a smaller number than the old reading
implied.

What the bin gather actually costs
----------------------------------
The gather is `bins[f * n_rows + r]` with `r = rows[begin + j]` and
`j = tile_begin + thread_idx.x`. Consecutive threads take consecutive `j`,
and a node's rows stay ascending (`gpu_active_rows.mojo` partitions stably),
so **at a fixed feature the warp's reads are ascending, and at the root
exactly contiguous**: 32 threads read 32 consecutive bytes. Feature-major is
already coalesced across threads, which is the axis a GPU coalesces on. The
intuition that one thread's `G` accesses are `n_rows` apart and therefore
uncoalesced is a CPU-shaped intuition; each of those `G` accesses is
separately warp-contiguous, and the device never sees them as one thread's
working set.

Write `S` for the device's memory sector, `rho = count / n_rows` for a node's
row fraction, `w` for the storage width, and `G` for the block width. A
sector of a feature-major column holds `k = S / (w/8)` rows; a sector of a
`G`-wide block holds `k / G` of them. Per feature per node the two layouts
touch (DERIVED BOUND, from `sectors_touched`):

    feature-major   min(rho, 1/S) * n_rows
    blocked         min(rho, G/S) * n_rows / G       (for G * w/8 <= S)

so the blocked layout's advantage on the bin term is exactly

    clamp(G / (rho * S), 1, G)

and it is **1.0 for every node with rho >= G/S**. Feature-blocking is a
deep-node change and nothing else. At the root the two layouts touch the
identical number of sectors because they move the identical bytes, and a
`G`-fold advantage only appears once a node's rows are sparse enough that a
feature-major sector carries one wanted row instead of many.

What the sum over a tree says, and it is smaller than it looks
--------------------------------------------------------------
A node comparison is the wrong denominator, because a level of `2^d` nodes
covers all `n_rows` between them however deep it is. `layout_tree_cost` sums
`layout_node_cost` over a balanced tree of depth `D` and is what any layout
argument here has to be built on. Two sector models are priced, because the
two-regime `min` is a bound and an occupancy model
`E[distinct] = M * (1 - (1 - rho)^k)` over `M = n_rows / k` sectors is an
equally defensible reading of the same gather. They turn out to agree within
about 15%, and in both directions, so the model choice is not what the answer
hangs on.

`S` is. At `G = GROUP = 8`, 100 features all active, width 8, 1M rows, bin
plus row-side sectors (DERIVED BOUND under each model, both reproducible from
`layout_tree_cost`):

    S      D     min-model    occupancy model
    32     5       2.03x          1.88x
    64     5       1.47x          1.59x
    128    5       1.15x          1.31x
    128    6       1.47x          1.60x

`S` for an Apple M4 is **UNMEASURED**, this module refuses to default it, and
it moves the claim by 1.8x across the plausible range.

Then the coupling that closes it. The advantage needs a wide `G`; a wide `G`
is only useful up to the launch group `GROUP` that consumes it, because a
launch block spanning several storage blocks gets locality `min(G, GROUP)`
and one sitting inside a wider block gets `GROUP` and no more; and `GROUP` is
bounded by threadgroup memory at `GROUP * BIN_CAP * 12` bytes.
`gpu_tiling.free_feature_group` therefore returns 8 at 64 bins and **2 on
Metal at 256, 1 everywhere else**. At the bin count this project defaults to,
the whole table above collapses to (DERIVED BOUND, same computation):

    G = GROUP = 2     S = 32    D = 5     1.12x / 1.11x
                      S = 64    D = 5     1.00x / 1.06x
                      S = 128   D = 5     1.00x / 1.02x
                      (min-model / occupancy model, as above)
    GROUP = 1         any S,    any D     1.00x at G = 1, and worse
                                          than 1.00x at every wider G

That last row is not an approximation and it is the sharpest thing here. At
`GROUP = 1` a launch owns one feature, so it fetches a `G`-wide row and uses
one byte of it; the block is then walked again for each of the other lanes,
and in the streaming regime it pays for the whole block every time. A blocked
layout under a group of 1 is a `G`-fold *regression*, not a tie. Every
backend that is not Metal takes `GROUP = 1` at the default bin count.

Two more subtractions, both against. Feature subsampling inverts the sign: a
random `feature_fraction` leaves every storage block live while narrowing
feature-major's active columns, so at 0.5 the blocked layout **loses** 1.12x
and at 0.25 it loses 1.77x (occupancy model, `S = 128`, `D = 5`,
`G = GROUP = 8`, DERIVED BOUND; the min-model says 1.27x and 2.00x).
And nothing in either model touches the three shared atomics per
(row, feature), which are identical under both layouts and dilute whatever
the memory win is by an unmeasured fraction.

The verdict: the GPU histogram path reads feature-major
-------------------------------------------------------
At the bin count this project defaults to, the modeled advantage is 1.00x on
every backend but Metal and 1.00x to 1.12x on Metal, under both sector
models, before the atomics dilute it and before feature subsampling inverts
it. The configuration where it is worth 1.5x to 2x needs 64 bins or fewer,
`GROUP` at 8, no feature subsampling, and a sector size nobody has measured.
That is not a paper case that holds; it is a paper case with a live condition
attached to it. So this module answers `LAYOUT_UNDECIDED`, as it was built
to, and the device histogram path keeps `bins[f * n_rows + r]`.

The contract the histogram kernel is handed
-------------------------------------------
Written down because it is now a decision and not an accident, and because a
reader of the kernel should not have to re-derive it:

    buffer        one `UInt8` per cell, `n_rows * n_features` of them
    address       bins[f * n_rows + r], f the feature id, r the row id
    stride        1 byte between adjacent rows of one feature
                  n_rows bytes between adjacent features of one row
    order         `rows[begin ..< begin + count]` ascending, so consecutive
                  threads read ascending and, at the root, adjacent bytes
    ownership     `histogram_gpu.GpuHistogramBuilder` uploads it once with
                  `enqueue_copy` and never rewrites it; it is exactly
                  `BinnedMatrix.bins`, byte for byte, with no transform
    who else      the partition kernel and the prediction kernel index the
                  same buffer the same way, so this contract is not the
                  histogram path's alone to change

What a change would have cost, for the record: a relayout is a host pass of
`n_rows * n_features` byte reads and the same number of strided writes, once
per session, plus the same upload. At 1M by 100 that is 100 MB each way,
ESTIMATED at tens of milliseconds for a cache-hostile transpose, against a
GPU training run of seconds. The transform is not what made this a bad trade
and it should not be cited as though it were; what made it a bad trade is
that the win is 1.00x at the default bin count and that three kernel families
index this buffer, not one.

This is the GPU's answer to the layout question, and it is deliberately not
the CPU's. `binning.BinnedMatrix` is gaining a row-major bin view for the CPU
builder, which walks rows and holds one row's features in cache, so locality
*within a walker* is what the CPU pays for. A device kernel is
thread-per-row: locality within a thread is worth nothing to it, and
coalescing *across threads at a fixed feature* is worth everything. The two
campaigns should not be expected to converge, and the GPU histogram path
reading feature-major while the CPU histogram path reads row-major is the
correct outcome rather than an inconsistency to reconcile.

A full row-major view is refused on a second and independent ground, which
belongs to the histogram kernel rather than to this module: a threadgroup
needs `3 * k * BIN_CAP * 4` bytes of shared memory for `k` feature slots,
which is 3 KiB per feature at 256 bins, so a 32 KiB threadgroup holds at most
eight slots however the buffer is laid out. A row-major record wider than
that is fetched and mostly discarded, and the record is re-read
`ceil(n_features / 8)` times. Row-major on the device is not a layout the
histogram kernel could exploit even if the sector arithmetic favored it.

What would decide it, cheaply
-----------------------------
The whole disagreement reduces to one measurable: how many sectors a
scattered ascending gather of `rho * n_rows` one-byte reads actually touches.
That is measurable **with the kernel that already ships and no new layout**,
by timing a one-feature histogram at fixed `count` while varying `n_rows`,
hence `rho`. The `min` model predicts a hard knee at `rho = 1/S`; the
occupancy model predicts a smooth curve with no knee. Whichever curve comes
back also hands over `S`, and `layout_tree_cost` then answers the layout
question outright instead of bracketing it. Until that exists, see
`bench/apple/bin_layout_plan.json`.

What this module refuses to decide
----------------------------------
`layout_node_cost` counts work in units a profiler can weigh (sectors
touched, decode ALU ops, launches, upload bytes, pack writes) and
`decide_layout` combines them *only* when the caller supplies per-unit costs
it actually measured on the device in question. With no measurements the
verdict is `LAYOUT_UNDECIDED`, and it stays undecided: there is no default
threshold, no width heuristic that fires on its own, and no automatic
fallback that would pick a layout on this module's authority. Lower transfer
and cache traffic can and does lose to bit extraction, and which way it goes
on an M4's unified memory is not derivable from anything written here.

Compression, separately
-----------------------
The bin is one byte of thirteen. Halving it to four bits removes 3.8% of the
per-(row, feature) traffic in the best case, which is why compression on its
own is a poor bet. Its real argument is *shared memory*: a feature of width
`w` needs `2^w` histogram slots, not `n_bins`, so a block of 4-bit features
needs 16 slots per feature where 8-bit features need 256, and a narrow block
can be sixteen times wider under the same threadgroup budget. That is an
occupancy argument, not a bandwidth one, and it is orthogonal to the blocking
question this module just answered.

Semantics this layout does not touch
------------------------------------
Bin ids are stored, never renumbered (see `gpu_bin_packing`). So:

- `BinnedMatrix.missing_bin[f]` keeps its value and a missing row is still
  found by the same equality test;
- a categorical feature's bins keep their positions in the 256-bit
  `CatBitset`, so `RowRouting.categorical` is unchanged;
- a split's `threshold_bin` keeps its meaning, so trees, model dumps, and
  LightGBM parity are all unaffected;
- `n_bins` still sizes every histogram. Width is a *storage* property of a
  feature, not a change to how many bins it has.

Preserving an id is not the same as being able to *store* it, and that is the
one way a width choice could change what a model means. A width narrower than
a feature's missing bin, or than a categorical feature's highest category bin,
would truncate the marker: missing rows would route by the threshold instead
of by `default_left`, and a category would fall out of its split set, both
without any bounds violation. `check_markers_preserved` is the refusal, it
runs inside every descriptor this module produces, and it is never falled back
from.

Host binning is untouched: `fit_bins` and `BinMapper.transform` produce the
same `BinnedMatrix` they always did, and packing runs after them as a pure
reformatting of bytes that already exist.

What the rest of the lane consumes
----------------------------------
A `BinLayoutPlan` is the layout *decision*; `BinStorageDescriptor` (defined in
`gpu_histogram_specializations.mojo`) is the small thing the histogram
kernels, the specialization planner, and the trainer read. This module is the
only one that produces one, through `BinLayoutPlan.storage_descriptor`, so
nothing downstream re-derives an element width or a block factor from a bin
count. Three entry points matter to a caller:

    baseline_descriptor(data)          the UInt8 matrix uploaded today
    plan.storage_descriptor(mb, cats)  what a candidate plan really is
    resolve_storage(data, plan)        the descriptor to actually train on

`resolve_storage` is the conservative switch: a plan the shipping kernels
cannot read falls back to the baseline rather than being uploaded, because
every kernel in this repository indexes `bins[f * n_rows + r]` as one byte.
Storage classes wider than a byte (`BIN_STORAGE_U16`, `BIN_STORAGE_WIDER`) are
named so a descriptor can report them truthfully, and are unreachable from
this repository's data representation: `binning.fit_bins` caps `max_bins` at
256 and `BinnedMatrix.bins` is a `List[UInt8]`, so no plan resolves above 8
bits and `BinStorageDescriptor.check_shipping` says so in as many words rather
than pretending otherwise.
"""

from .binning import BinMapper, BinnedMatrix
from .categorical import CAT_MAX_BINS, CategoricalSpec
from .gpu_bin_packing import (
    BIN_WIDTH_FALLBACK,
    BIN_WIDTH_MIN,
    bins_representable,
    check_bin_width,
    decode_cost,
    element_bit_shift,
    element_byte_offset,
    pack_value,
    packed_data_bytes,
    packed_stream_bytes,
    streams_disjoint,
    unpack_value,
    width_for_bins,
    writable_footprint_end,
)
from .gpu_histogram_specializations import (
    MAX_BINS,
    BinStorageDescriptor,
    bin_storage_name,
    storage_for_width,
    storage_is_shipping,
)
from .gpu_tiling import BYTES_PER_PARTIAL_CELL, DeviceCaps

# Byte offsets into the packed buffer cross into kernels as Int32, the same
# bound `histogram_gpu.MAX_ROWS` already imposes on row ids.
comptime LAYOUT_MAX_INDEX = Int(Int32.MAX)

# The widest histogram the GPU backend accepts. One definition for the whole
# GPU dataflow lane, in `gpu_histogram_specializations.mojo`; the name below
# is kept because this module's support codes read in its terms.
comptime LAYOUT_MAX_BINS = MAX_BINS

# Every block starts at a multiple of this. Sixteen bytes is the coarsest
# natural alignment on the three backends and is a multiple of every load
# width a kernel might use, so a block's base address never splits a vector
# load and never depends on the width of the block before it.
comptime BLOCK_ALIGN_BYTES = 16

# Int32 gradient + Int32 hessian + Int32 count per shared histogram slot.
# Imported rather than restated: it is the same cell `gpu_tiling` and the
# batched kernels size their partial buffers with.
comptime BYTES_PER_HIST_SLOT = BYTES_PER_PARTIAL_CELL

# The *row side* of a histogram build. Two shapes, because the shipping
# kernel has two row loops:
#
#   quantized (the default)  the Int32 active-row index, read contiguously
#                            out of the compacted range, plus the Int32
#                            gradient/hessian pair, read as one eight-byte
#                            scattered access at `gq[2 * r]`;
#   float                    the Int32 active-row index plus two separate
#                            four-byte scattered planes, `grad[r]`, `hess[r]`.
#
# Either way the row side is read once per *launch block* of `GROUP` feature
# slots, not once per feature and not once per storage block. `GROUP` is a
# launch shape (`GpuActiveRows.set_feature_group`) and is independent of any
# storage layout, which is why this term is no longer an argument for
# blocking; see the module docstring.
comptime ROW_SIDE_FLOAT_ARRAYS = 2
comptime BYTES_ROW_SIDE_ELEM = 4
comptime BYTES_GRAD_PAIR = 8

# A caller that has not established its device's memory sector passes this,
# and every sector-denominated cost comes back unmeasured rather than
# computed against a guess. Common real values are 32 (NVIDIA sector), 64,
# and 128 (cache line); none of them is defaulted to here.
comptime SECTOR_BYTES_UNKNOWN = 0


# --- Layout kinds ---------------------------------------------------------

comptime LAYOUT_FEATURE_MAJOR_U8 = 0
comptime LAYOUT_FEATURE_MAJOR_PACKED = 1
comptime LAYOUT_ROW_MAJOR_U8 = 2
comptime LAYOUT_ROW_MAJOR_PACKED = 3
comptime LAYOUT_FEATURE_BLOCKED_U8 = 4
comptime LAYOUT_FEATURE_BLOCKED_PACKED = 5


def layout_name(kind: Int) -> String:
    if kind == LAYOUT_FEATURE_MAJOR_U8:
        return String("feature-major-u8")
    if kind == LAYOUT_FEATURE_MAJOR_PACKED:
        return String("feature-major-packed")
    if kind == LAYOUT_ROW_MAJOR_U8:
        return String("row-major-u8")
    if kind == LAYOUT_ROW_MAJOR_PACKED:
        return String("row-major-packed")
    if kind == LAYOUT_FEATURE_BLOCKED_U8:
        return String("feature-blocked-u8")
    if kind == LAYOUT_FEATURE_BLOCKED_PACKED:
        return String("feature-blocked-packed")
    return String("unknown")


def layout_is_packed(kind: Int) -> Bool:
    """Whether a kind may use a width below 8. A packed *kind* whose plan
    happens to resolve every block to width 8 is still byte-identical to the
    matching unpacked kind; the flag describes the policy, not the bytes."""
    return (
        kind == LAYOUT_FEATURE_MAJOR_PACKED
        or kind == LAYOUT_ROW_MAJOR_PACKED
        or kind == LAYOUT_FEATURE_BLOCKED_PACKED
    )


def layout_is_fallback(kind: Int) -> Bool:
    """Whether a kind is the ordinary `UInt8` feature-major buffer the GPU
    backend uploads today. Exactly one kind is."""
    return kind == LAYOUT_FEATURE_MAJOR_U8


# --- Support ---------------------------------------------------------------
#
# The shapes no layout in this module can express, as opposed to the ones it
# might merely lose on. Every one of them is a hard refusal: silently
# widening a feature past 8 bits or silently truncating a bin id is exactly
# the failure this module exists to make impossible.

comptime LAYOUT_OK = 0
comptime LAYOUT_EMPTY = 1
comptime LAYOUT_TOO_MANY_BINS = 2
comptime LAYOUT_TOO_MANY_ROWS = 3
comptime LAYOUT_TOO_MANY_FEATURES = 4
comptime LAYOUT_BYTES_OVERFLOW = 5
comptime LAYOUT_SHARED_MEMORY = 6
comptime LAYOUT_WIDTH_TOO_NARROW = 7
comptime LAYOUT_CATEGORIES_EXCEED_BINS = 8


def layout_support_name(reason: Int) -> String:
    if reason == LAYOUT_OK:
        return String("supported")
    if reason == LAYOUT_EMPTY:
        return String("empty matrix")
    if reason == LAYOUT_TOO_MANY_BINS:
        return String("more than 256 bins")
    if reason == LAYOUT_TOO_MANY_ROWS:
        return String("more rows than an Int32 index holds")
    if reason == LAYOUT_TOO_MANY_FEATURES:
        return String("more features than an Int32 index holds")
    if reason == LAYOUT_BYTES_OVERFLOW:
        return String("packed buffer larger than an Int32 byte offset holds")
    if reason == LAYOUT_SHARED_MEMORY:
        return String("block needs more shared memory than the device has")
    if reason == LAYOUT_WIDTH_TOO_NARROW:
        return String("a bin id does not fit its feature's width")
    if reason == LAYOUT_CATEGORIES_EXCEED_BINS:
        return String("categorical feature has more categories than bins")
    return String("unknown")


def layout_support(n_rows: Int, n_features: Int, n_bins: Int) -> Int:
    """Whether a dataset shape can be laid out at all, as a code. Does not
    raise, so a caller can report the reason."""
    if n_rows < 1 or n_features < 1 or n_bins < 1:
        return LAYOUT_EMPTY
    if n_bins > LAYOUT_MAX_BINS:
        return LAYOUT_TOO_MANY_BINS
    if n_rows > LAYOUT_MAX_INDEX:
        return LAYOUT_TOO_MANY_ROWS
    if n_features > LAYOUT_MAX_INDEX:
        return LAYOUT_TOO_MANY_FEATURES
    # The uncompressed size bounds every packed size, so checking it here
    # covers all widths at once.
    if n_rows > LAYOUT_MAX_INDEX // n_features:
        return LAYOUT_BYTES_OVERFLOW
    return LAYOUT_OK


def check_layout_support(n_rows: Int, n_features: Int, n_bins: Int) raises:
    """Raising form of `layout_support`."""
    var reason = layout_support(n_rows, n_features, n_bins)
    if reason != LAYOUT_OK:
        raise Error(
            "binned layout does not support this dataset: "
            + layout_support_name(reason)
        )


# --- Per-feature bin extents ----------------------------------------------
#
# A width is only as safe as the bin count it was derived from, and there are
# two different bin counts a caller might have. They are not
# interchangeable, so both are provided and both say which they are.


def declared_bins_from_mapper(mapper: BinMapper) raises -> List[Int]:
    """Per-feature bin *capacity*, from the fitted mapper.

    This is the count that is safe for any matrix the mapper produces,
    including matrices binned later at prediction time, because it is derived
    from what `BinMapper.bin_value` can return rather than from what one
    matrix happens to contain:

    - a numerical feature with `k` edges returns bins `0 .. k`, so it needs
      `k + 1`, and one more when it reserves a missing bin at `k + 1`;
    - a categorical feature returns `0 .. n_categories`, bin 0 being the
      unknown/missing bin, so it needs `n_categories + 1`.

    Prefer this over `observed_bins_from_matrix` whenever the mapper is in
    hand. A plan built from observed extents is valid for the matrix it was
    observed on and for nothing else.
    """
    var n_features = mapper.n_features
    if len(mapper.edge_offsets) != n_features + 1:
        raise Error("mapper edge offsets must have n_features + 1 entries")
    if len(mapper.missing_bin) != n_features:
        raise Error("mapper missing-bin table must be one entry per feature")
    var out = List[Int](capacity=n_features)
    for f in range(n_features):
        var used: Int
        if mapper.cats.is_cat(f):
            used = mapper.cats.n_categories(f) + 1
        else:
            var k = mapper.edge_offsets[f + 1] - mapper.edge_offsets[f]
            used = k + 1
            var mb = mapper.missing_bin[f]
            if mb >= 0 and mb + 1 > used:
                used = mb + 1
        if used < 1:
            used = 1
        if used > mapper.n_bins:
            raise Error(
                "feature uses more bins than the mapper's n_bins allows"
            )
        out.append(used)
    return out^


def observed_bins_from_matrix(data: BinnedMatrix) raises -> List[Int]:
    """Per-feature bin count observed by scanning `data`: one past the
    largest bin id actually present in each column.

    A full `n_rows * n_features` host pass, and valid *for this matrix
    only*. A feature whose rarest bin is absent from the training rows gets a
    narrower width here than `declared_bins_from_mapper` would give it, and
    packing a different matrix against that plan would overflow the width.
    `check_plan_covers_matrix` is what catches that; deriving from the mapper
    is what avoids it.
    """
    if len(data.bins) != data.n_rows * data.n_features:
        raise Error("binned matrix size must equal n_rows * n_features")
    var out = List[Int](capacity=data.n_features)
    var src = data.bins.unsafe_ptr()
    for f in range(data.n_features):
        var col = f * data.n_rows
        var top = 0
        for r in range(data.n_rows):
            var v = Int(src.unsafe_load(col + r))
            if v > top:
                top = v
        var used = top + 1
        # A feature that reserves a missing bin keeps room for it even if no
        # row of this matrix is missing, so a later matrix from the same
        # mapper does not overflow the width on its first NaN.
        var mb = data.missing_bin[f]
        if mb >= 0 and mb + 1 > used:
            used = mb + 1
        out.append(used)
    return out^


def widths_from_bin_counts(
    bin_counts: List[Int], packed: Bool
) raises -> List[Int]:
    """Per-feature storage widths from per-feature bin counts.

    With `packed` false every width is 8, which is the ordinary `UInt8`
    layout expressed in this module's terms rather than as a separate case.
    """
    var out = List[Int](capacity=len(bin_counts))
    for f in range(len(bin_counts)):
        if bin_counts[f] < 1:
            raise Error("a feature must use at least one bin")
        if not packed:
            out.append(BIN_WIDTH_FALLBACK)
            continue
        var w = width_for_bins(bin_counts[f])
        if w == 0:
            raise Error("a feature with more than 256 bins cannot be packed")
        out.append(w)
    return out^


def check_categorical_widths(
    cats: CategoricalSpec, widths: List[Int], n_bins: Int
) raises:
    """A categorical feature's width has to cover every bin its split set can
    name.

    The set is a 256-bit `CatBitset` indexed by bin id, and the ids are
    preserved by packing, so the only thing that can go wrong is a width too
    narrow for the feature's highest category bin. Checked here so a plan
    fails at layout time rather than at split time.
    """
    for f in range(len(widths)):
        if not cats.is_cat(f):
            continue
        var n_cat = cats.n_categories(f)
        if n_cat >= n_bins:
            raise Error("categorical feature has more categories than bins")
        if n_cat + 1 > CAT_MAX_BINS:
            raise Error(
                "categorical feature has more categories than the 256-bit"
                " split set holds"
            )
        if n_cat + 1 > bins_representable(widths[f]):
            raise Error(
                "categorical feature's width cannot hold its category bins"
            )


def missing_markers_preserved(
    widths: List[Int], missing_bin: List[Int]
) raises -> Bool:
    """Whether every feature's missing bin id still fits the width it is
    stored at.

    Packing never renumbers a bin, so a missing row is still found by the same
    equality test `BinnedMatrix.missing_bin[f]` drives, *provided the id is
    representable at all*. A width narrower than the missing bin would store a
    truncated id, and rows that were missing would silently route by the
    threshold instead of by `default_left`. That is a wrong answer with no
    bounds violation anywhere, so it is a check rather than a comment.

    A feature with no missing bin passes -1, which nothing can fail.
    """
    if len(widths) != len(missing_bin):
        raise Error("width and missing-bin tables must be the same length")
    for f in range(len(widths)):
        var mb = missing_bin[f]
        if mb < 0:
            continue
        if mb >= bins_representable(widths[f]):
            return False
    return True


def categorical_markers_preserved(
    cats: CategoricalSpec, widths: List[Int], n_bins: Int
) -> Bool:
    """Non-raising form of `check_categorical_widths`.

    Same question, as a Bool, because `BinStorageDescriptor` carries the
    answer as a flag and a descriptor producer should not have to catch to
    fill it in.
    """
    for f in range(len(widths)):
        if not cats.is_cat(f):
            continue
        var n_cat = cats.n_categories(f)
        if n_cat >= n_bins:
            return False
        if n_cat + 1 > CAT_MAX_BINS:
            return False
        if n_cat + 1 > bins_representable(widths[f]):
            return False
    return True


def check_markers_preserved(
    widths: List[Int],
    missing_bin: List[Int],
    cats: CategoricalSpec,
    n_bins: Int,
) raises:
    """Raising form of both marker checks, named for what it protects.

    The one condition under which a width choice changes what a model means:
    the missing sentinel and the categorical set membership are both bin ids,
    and both are lost by a width too narrow to hold them. Every descriptor
    producer below runs this before it reports a storage class.
    """
    if not missing_markers_preserved(widths, missing_bin):
        raise Error(
            "binned layout does not support this dataset: "
            + layout_support_name(LAYOUT_WIDTH_TOO_NARROW)
            + " (a feature's missing bin does not fit its storage width)"
        )
    if not categorical_markers_preserved(cats, widths, n_bins):
        raise Error(
            "binned layout does not support this dataset: "
            + layout_support_name(LAYOUT_CATEGORIES_EXCEED_BINS)
            + " (a categorical feature's highest category bin does not fit"
            " its storage width)"
        )


# --- The plan -------------------------------------------------------------


def _align_up(value: Int, alignment: Int) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


struct BinLayoutPlan(Copyable, Movable):
    """A resolved device layout for one dataset shape.

    Features are partitioned into blocks in feature order. Block `b` holds
    `block_size[b]` consecutive features starting at `block_first[b]`, all
    stored at `block_width[b]` bits, row-major within the block, as one
    packed stream beginning at byte `block_offset[b]`. Feature `f` occupies
    lane `f - block_first[block_of[f]]` of its block.

    Byte formulas, all of them:

        G(b)       = block_size[b]
        elems(b)   = n_rows * G(b)
        stream(b)  = packed_stream_bytes(elems(b), block_width[b])
        offset(0)  = 0
        offset(b+1)= align_up(offset(b) + stream(b), BLOCK_ALIGN_BYTES)
        total      = offset(last) + stream(last)

        index(f,r) = r * G(block_of[f]) + lane(f)
        byte(f,r)  = block_offset[block_of[f]] + (index(f,r) * w) / 8
        shift(f,r) = (index(f,r) * w) mod 8

    `packed_stream_bytes` carries each block's own tail pad below width 8,
    not just the last block's, and that is not decoration. A packed block's
    last element reads *and writes* the byte one past its data, so without a
    pad between blocks the read-modify-write in `gpu_bin_packing.pack_value`
    would make adjacent blocks share a byte and block-parallel packing would
    lose updates. `check_blocks_disjoint` verifies the result.

    The passthrough case
    --------------------
    A plan of one feature per block, every block at width 8, is the layout
    the GPU backend uses today. It gets `BLOCK_ALIGN_BYTES = 1` and no pads,
    because width 8 needs neither, so its offsets come out as `f * n_rows`
    and its buffer is `BinnedMatrix.bins` itself. `is_passthrough` reports
    it, and it matters for more than tidiness: a passthrough plan uploads the
    dense buffer with no packing pass at all, so the baseline in any
    comparison is charged exactly what the current code path costs and not a
    byte more.
    """

    var kind: Int
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var width: List[Int]
    """Per-feature storage width in bits. Equal for every feature of a
    block."""
    var block_of: List[Int]
    var lane_of: List[Int]
    var block_first: List[Int]
    var block_size: List[Int]
    var block_width: List[Int]
    var block_offset: List[Int]
    var total_bytes: Int
    var passthrough: Bool
    """Whether this plan *is* `BinnedMatrix.bins`: one feature per block,
    width 8 throughout, so `block_offset[f] == f * n_rows` and
    `total_bytes == n_rows * n_features`."""

    def __init__(
        out self,
        kind: Int,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        var width: List[Int],
        var block_first: List[Int],
        var block_size: List[Int],
    ) raises:
        """Build the derived tables from a block partition and per-feature
        widths. Raises if the partition does not tile the features exactly or
        if a block is not width-homogeneous."""
        check_layout_support(n_rows, n_features, n_bins)
        if len(width) != n_features:
            raise Error("width table must be one entry per feature")
        if len(block_first) != len(block_size):
            raise Error("block tables must have the same length")
        if len(block_first) < 1:
            raise Error("a layout must have at least one block")

        self.kind = kind
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.width = width^
        self.block_first = block_first^
        self.block_size = block_size^

        var n_blocks = len(self.block_first)
        self.block_of = List[Int](capacity=n_features)
        self.block_of.resize(n_features, -1)
        self.lane_of = List[Int](capacity=n_features)
        self.lane_of.resize(n_features, -1)
        self.block_width = List[Int](capacity=n_blocks)
        self.block_offset = List[Int](capacity=n_blocks)

        # The layout the GPU backend already uses: one feature per block at
        # width 8. It needs no alignment padding and no tail pads, so its
        # offsets fall out as `f * n_rows` and its buffer is `bins` itself.
        self.passthrough = n_blocks == n_features
        if self.passthrough:
            for b in range(n_blocks):
                var first = self.block_first[b]
                if self.block_size[b] != 1:
                    self.passthrough = False
                elif first < 0 or first >= n_features:
                    # Malformed; the validation loop below raises on it.
                    self.passthrough = False
                elif self.width[first] != BIN_WIDTH_FALLBACK:
                    self.passthrough = False
        var align = BLOCK_ALIGN_BYTES
        if self.passthrough:
            align = 1

        var expected = 0
        var offset = 0
        for b in range(n_blocks):
            if self.block_first[b] != expected:
                raise Error("blocks must tile the features in feature order")
            if self.block_size[b] < 1:
                raise Error("a block must hold at least one feature")
            var first = self.block_first[b]
            var size = self.block_size[b]
            if first + size > n_features:
                raise Error("block runs past the last feature")
            var w = self.width[first]
            check_bin_width(w)
            for i in range(size):
                var f = first + i
                if self.width[f] != w:
                    raise Error("a block must be width-homogeneous")
                self.block_of[f] = b
                self.lane_of[f] = i
            self.block_width.append(w)
            self.block_offset.append(offset)
            # The stream size carries the block's own tail pad below width 8,
            # which is what keeps its writable footprint disjoint from the
            # next block's.
            var stream = packed_stream_bytes(n_rows * size, w)
            if stream > LAYOUT_MAX_INDEX - offset:
                raise Error(
                    "packed buffer larger than an Int32 byte offset holds"
                )
            offset = _align_up(offset + stream, align)
            expected = first + size

        if expected != n_features:
            raise Error("blocks must cover every feature")

        # `offset` is now the aligned start the next block would take; the
        # buffer only needs up to the end of the last block's stream.
        var last = n_blocks - 1
        var last_stream = packed_stream_bytes(
            n_rows * self.block_size[last], self.block_width[last]
        )
        self.total_bytes = self.block_offset[last] + last_stream
        if self.total_bytes > LAYOUT_MAX_INDEX:
            raise Error(
                "packed buffer larger than an Int32 byte offset holds"
            )

    def n_blocks(self) -> Int:
        return len(self.block_first)

    def block_features(self, block: Int) raises -> Int:
        """`G(block)`, how many features share the block's rows."""
        if block < 0 or block >= self.n_blocks():
            raise Error("block index out of range")
        return self.block_size[block]

    def block_row_bits(self, block: Int) raises -> Int:
        """Bits one row of one block occupies: `G(block) * width(block)`.

        The quantity that decides whether a block's row lands inside one
        memory sector, which is what a blocked layout is for.
        """
        return self.block_size[block] * self.block_width[block]

    def block_data_bytes(self, block: Int) raises -> Int:
        if block < 0 or block >= self.n_blocks():
            raise Error("block index out of range")
        return packed_data_bytes(
            self.n_rows * self.block_size[block], self.block_width[block]
        )

    def element_index(self, feature: Int, row: Int) raises -> Int:
        """`index(f, r)`, the element's position inside its block's stream."""
        if feature < 0 or feature >= self.n_features:
            raise Error("feature index out of range")
        if row < 0 or row >= self.n_rows:
            raise Error("row index out of range")
        var b = self.block_of[feature]
        return row * self.block_size[b] + self.lane_of[feature]

    def byte_of(self, feature: Int, row: Int) raises -> Int:
        """`byte(f, r)`: the byte the decode window starts at, which is the
        only byte it reads at width 8 and the first of two below it. The
        formula a kernel evaluates, checked."""
        var idx = self.element_index(feature, row)
        var b = self.block_of[feature]
        return self.block_offset[b] + element_byte_offset(
            idx, self.block_width[b]
        )

    def shift_of(self, feature: Int, row: Int) raises -> Int:
        """`shift(f, r)`, in `[0, 7]`."""
        var idx = self.element_index(feature, row)
        var b = self.block_of[feature]
        return element_bit_shift(idx, self.block_width[b])

    def bytes(self) -> Int:
        """Device bytes the packed matrix occupies, per-block tail pads and
        alignment included. Exactly `dense_bytes()` for a passthrough
        plan, which carries neither."""
        return self.total_bytes

    def dense_bytes(self) -> Int:
        """What the same matrix costs as today's `UInt8` buffer, which is
        what `GpuHistogramBuilder` uploads."""
        return self.n_rows * self.n_features

    def bytes_ratio(self) -> Float64:
        """Packed bytes over dense bytes. Above 1.0 means the layout is
        *larger* than the buffer it replaces, which a width-8 blocked plan
        with many small blocks genuinely can be: every block below width 8
        carries a pad and every block but the passthrough case's is aligned
        to `BLOCK_ALIGN_BYTES`. Reported, never thresholded on."""
        return Float64(self.bytes()) / Float64(self.dense_bytes())

    def is_passthrough(self) -> Bool:
        """Whether this plan's buffer is `BinnedMatrix.bins` itself.

        True means the packed reader computes `f * n_rows + r` for every
        cell, exactly what `bins[f * n_rows + r]` computes today, and that
        no decode instruction runs and no packing pass is needed: the
        existing upload is already this layout. It is the strongest form of
        "fallback", and `check_passthrough_offsets` proves the addresses
        rather than trusting the flag.
        """
        return self.passthrough

    def check_passthrough_offsets(self) raises:
        """A passthrough plan really does address cells as
        `f * n_rows + r`.

        Worth checking rather than asserting, because the claim is what
        licenses uploading `data.bins` unchanged; if the offsets ever drifted
        the kernel would read a shifted matrix and train on nonsense without
        any bounds violation.
        """
        if not self.passthrough:
            return
        if self.total_bytes != self.n_rows * self.n_features:
            raise Error("passthrough plan is not the dense buffer's size")
        for f in range(self.n_features):
            if self.block_offset[self.block_of[f]] != f * self.n_rows:
                raise Error("passthrough plan's offsets are not f * n_rows")
            if self.byte_of(f, 0) != f * self.n_rows:
                raise Error("passthrough plan does not address bins[f*N+r]")

    def shared_bytes_for_block(self, block: Int) raises -> Int:
        """Threadgroup memory a kernel needs to hold one partial histogram
        per feature of `block`.

        A feature of width `w` can only produce bins `0 .. 2^w - 1`, so its
        partial needs `2^w` slots and not `n_bins`. That is the bound that
        makes narrow blocks wide: at `n_bins = 256` an 8-bit block costs
        3072 bytes per feature, a 4-bit block 192.
        """
        if block < 0 or block >= self.n_blocks():
            raise Error("block index out of range")
        var slots = bins_representable(self.block_width[block])
        if slots > self.n_bins:
            slots = self.n_bins
        return self.block_size[block] * slots * BYTES_PER_HIST_SLOT

    def max_shared_bytes(self) raises -> Int:
        """The widest block's shared-memory need, which is what has to fit."""
        var top = 0
        for b in range(self.n_blocks()):
            var need = self.shared_bytes_for_block(b)
            if need > top:
                top = need
        return top

    def fits_shared_memory(self, caps: DeviceCaps) raises -> Bool:
        return self.max_shared_bytes() <= caps.max_shared_memory_per_block

    def check_fits_shared_memory(self, caps: DeviceCaps) raises:
        if not self.fits_shared_memory(caps):
            raise Error(
                "binned layout does not support this dataset: "
                + layout_support_name(LAYOUT_SHARED_MEMORY)
            )

    def blocks_touched(self, active: List[Int]) raises -> Int:
        """How many blocks hold at least one of `active`'s features.

        The launch count of a blocked histogram, and the number of times the
        row index, gradient, and hessian are re-read per row.
        """
        var seen = List[Bool](capacity=self.n_blocks())
        seen.resize(self.n_blocks(), False)
        var count = 0
        for i in range(len(active)):
            var f = active[i]
            if f < 0 or f >= self.n_features:
                raise Error("active feature index out of range")
            var b = self.block_of[f]
            if not seen[b]:
                seen[b] = True
                count += 1
        return count

    def subsample_waste(self, active: List[Int]) raises -> Float64:
        """Fraction of the bin bytes a blocked read pulls in and discards
        under this active feature set.

        Zero when every touched block is fully active (and always zero for a
        one-feature-per-block plan, which is why feature-major is the layout
        that loses nothing to `colsample`). Approaches one as a wide block
        contributes a single active feature.
        """
        var live = List[Int](capacity=self.n_blocks())
        live.resize(self.n_blocks(), 0)
        for i in range(len(active)):
            var f = active[i]
            if f < 0 or f >= self.n_features:
                raise Error("active feature index out of range")
            live[self.block_of[f]] += 1
        var pulled = 0
        var used = 0
        for b in range(self.n_blocks()):
            if live[b] == 0:
                continue
            pulled += self.block_size[b] * self.block_width[b]
            used += live[b] * self.block_width[b]
        if pulled == 0:
            return 0.0
        return Float64(pulled - used) / Float64(pulled)

    def check_blocks_disjoint(self) raises:
        """Every block's writable footprint is disjoint from every other's,
        and the last one's stays inside the buffer.

        The invariant that makes block-parallel packing sound and that makes
        the last element's decode window in-bounds. Quadratic in the block
        count and meant for verification rather than for the hot path, but
        the block count is at most `n_features` and this runs once.
        """
        var n = self.n_blocks()
        for a in range(n):
            var end = self.block_offset[a] + writable_footprint_end(
                self.n_rows * self.block_size[a], self.block_width[a]
            )
            if end >= self.total_bytes:
                raise Error(
                    "block's decode window runs past the packed buffer"
                )
            for b in range(a + 1, n):
                if not streams_disjoint(
                    self.block_offset[a],
                    self.n_rows * self.block_size[a],
                    self.block_width[a],
                    self.block_offset[b],
                    self.n_rows * self.block_size[b],
                    self.block_width[b],
                ):
                    raise Error("two blocks share a writable byte")

    # --- The storage descriptor -------------------------------------------
    #
    # What this plan looks like to a histogram kernel. The plan is where the
    # storage *choice* is made; `BinStorageDescriptor` is the small thing the
    # kernels and the specialization module read, so nothing downstream
    # re-derives a width or a block factor from a bin count.

    def uniform_width(self) -> Int:
        """The width every block shares, or -1 when they differ.

        A mixed-width plan is legal (a `plan_feature_blocked` with `promote`
        false cuts blocks at width changes), and it is exactly the plan that
        has no single element width to report, so the descriptor refuses it
        rather than reporting one block's width as the matrix's.
        """
        if self.n_blocks() < 1:
            return -1
        var w = self.block_width[0]
        for b in range(1, self.n_blocks()):
            if self.block_width[b] != w:
                return -1
        return w

    def uniform_block_size(self) -> Int:
        """`G`, when every block holds the same number of features, else -1.

        The last block of a blocked plan is short whenever `target_block` does
        not divide `n_features`, so this is -1 more often than not; that is
        the honest answer, because a kernel addressing `r * G + lane` needs one
        `G` for the whole matrix.
        """
        if self.n_blocks() < 1:
            return -1
        var g = self.block_size[0]
        for b in range(1, self.n_blocks()):
            if self.block_size[b] != g:
                return -1
        return g

    def storage_class(self) raises -> Int:
        """The `BIN_STORAGE_*` class this plan resolves to.

        Derived from the element width and nothing else, so a plan that packs
        at 4 bits reports packed storage even though its buffer is still a
        `List[UInt8]`: the class describes what a *cell* costs, not what the
        allocation's element type happens to be. A mixed-width plan has no
        class and raises.
        """
        var w = self.uniform_width()
        if w < 0:
            raise Error(
                "a mixed-width layout has no single storage class; promote"
                " the widths first (see promote_widths_uniform) or plan"
                " feature-major"
            )
        return storage_for_width(w)

    def storage_descriptor(
        self, missing_bin: List[Int], cats: CategoricalSpec
    ) raises -> BinStorageDescriptor:
        """This plan as the descriptor the histogram lane consumes.

        Refuses before it reports, in this order: a mixed-width or mixed-block
        plan has nothing truthful to say; a width that cannot hold a feature's
        missing bin or a categorical feature's highest category bin would lose
        the marker, which `check_markers_preserved` turns into an error rather
        than into a flag a caller might ignore. What comes back is therefore
        either a descriptor whose marker flags are both true, or an error
        naming the feature class that could not be represented.

        Reporting a class is not the same as claiming a kernel can read it.
        `BinStorageDescriptor.check_shipping` is that question, and only the
        8-bit one-feature-per-block plan answers it yes today.
        """
        var width = self.uniform_width()
        if width < 0:
            raise Error(
                "a mixed-width layout has no single storage descriptor;"
                " promote the widths first (see promote_widths_uniform)"
            )
        var block = self.uniform_block_size()
        if block < 0:
            raise Error(
                "a layout whose blocks hold different feature counts has no"
                " single storage descriptor; a kernel addressing"
                " r * G + lane needs one G for the whole matrix"
            )
        if len(missing_bin) != self.n_features:
            raise Error("missing-bin table must be one entry per feature")
        check_markers_preserved(
            self.width, missing_bin, cats, self.n_bins
        )
        var desc = BinStorageDescriptor(
            storage_for_width(width),
            width,
            self.n_rows,
            self.n_features,
            self.n_bins,
            block,
            self.is_passthrough(),
            True,
            True,
        )
        desc.check()
        return desc^

    def check_storage_shipping(
        self, missing_bin: List[Int], cats: CategoricalSpec
    ) raises:
        """Refuse this plan unless the kernels compiled into this build can
        read it.

        The gate the trainer wants before it uploads anything: a packed or
        blocked plan is a legal *description* and an illegal *upload*, because
        every shipping kernel indexes `bins[f * n_rows + r]` as one byte.
        Failing here names the storage class; failing later would mean
        training on bytes read as the wrong cells.
        """
        var desc = self.storage_descriptor(missing_bin, cats)
        desc.check_shipping()

    def storage_name(self) raises -> String:
        return bin_storage_name(self.storage_class())


# --- Plan builders --------------------------------------------------------


def plan_feature_major(
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    var width: List[Int],
) raises -> BinLayoutPlan:
    """One feature per block: today's layout, at whatever widths are given.

    With every width 8 this is byte for byte the current buffer, with no
    alignment padding and no tail pads, and `is_passthrough` returns true.
    """
    var first = List[Int](capacity=n_features)
    var size = List[Int](capacity=n_features)
    for f in range(n_features):
        first.append(f)
        size.append(1)
    var packed = False
    for f in range(len(width)):
        if width[f] != BIN_WIDTH_FALLBACK:
            packed = True
    var kind = LAYOUT_FEATURE_MAJOR_U8
    if packed:
        kind = LAYOUT_FEATURE_MAJOR_PACKED
    return BinLayoutPlan(
        kind, n_rows, n_features, n_bins, width^, first^, size^
    )


def plan_row_major(
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    var width: List[Int],
) raises -> BinLayoutPlan:
    """One block holding every feature: a row's bins are contiguous.

    Requires a single width across the whole matrix, because a block is
    width-homogeneous. `promote_widths_uniform` is how a mixed-width dataset
    gets one: it widens every feature to the maximum, which never changes a
    bin id and costs only bits.
    """
    if len(width) != n_features:
        raise Error("width table must be one entry per feature")
    for f in range(1, n_features):
        if width[f] != width[0]:
            raise Error(
                "row-major needs one width for every feature; promote first"
            )
    var kind = LAYOUT_ROW_MAJOR_PACKED
    if width[0] == BIN_WIDTH_FALLBACK:
        kind = LAYOUT_ROW_MAJOR_U8
    return BinLayoutPlan(
        kind, n_rows, n_features, n_bins, width^, [0], [n_features]
    )


def plan_feature_blocked(
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    var width: List[Int],
    target_block: Int,
    promote: Bool,
) raises -> BinLayoutPlan:
    """Blocks of at most `target_block` consecutive features.

    Blocks are cut so that they stay width-homogeneous. With `promote` false
    a run of equal widths is what forms a block, so a dataset whose widths
    alternate degenerates gracefully to feature-major. With `promote` true a
    block takes the next `target_block` features whatever their widths and
    stores them all at the widest one, trading bits for locality; promotion
    is safe because a wider width holds every id the narrow one held.
    """
    if len(width) != n_features:
        raise Error("width table must be one entry per feature")
    if target_block < 1:
        raise Error("target block width must be positive")

    var first = List[Int]()
    var size = List[Int]()
    var f = 0
    while f < n_features:
        var take = target_block
        if f + take > n_features:
            take = n_features - f
        if promote:
            var top = width[f]
            for i in range(1, take):
                if width[f + i] > top:
                    top = width[f + i]
            for i in range(take):
                width[f + i] = top
        else:
            # Cut the run where the width changes.
            var run = 1
            while run < take and width[f + run] == width[f]:
                run += 1
            take = run
        first.append(f)
        size.append(take)
        f += take

    var packed = False
    for i in range(len(width)):
        if width[i] != BIN_WIDTH_FALLBACK:
            packed = True
    var kind = LAYOUT_FEATURE_BLOCKED_U8
    if packed:
        kind = LAYOUT_FEATURE_BLOCKED_PACKED
    return BinLayoutPlan(
        kind, n_rows, n_features, n_bins, width^, first^, size^
    )


def promote_widths_uniform(var width: List[Int]) raises -> List[Int]:
    """Every feature widened to the widest, so a single-block layout is
    expressible. Never narrows anything, so no bin id can stop fitting."""
    if len(width) == 0:
        raise Error("width table must not be empty")
    var top = width[0]
    for f in range(1, len(width)):
        if width[f] > top:
            top = width[f]
    check_bin_width(top)
    for f in range(len(width)):
        width[f] = top
    return width^


def max_block_for_shared(
    width: Int, n_bins: Int, caps: DeviceCaps
) raises -> Int:
    """The widest width-homogeneous block whose per-feature partial
    histograms still fit one threadgroup's shared memory.

    This is the ceiling `plan_feature_blocked`'s `target_block` has to
    respect, and it is where compression pays its clearest dividend: the
    ceiling scales as `1 / min(2^width, n_bins)`.
    """
    check_bin_width(width)
    var slots = bins_representable(width)
    if slots > n_bins:
        slots = n_bins
    var per_feature = slots * BYTES_PER_HIST_SLOT
    var fit = caps.max_shared_memory_per_block // per_feature
    if fit < 1:
        return 0
    return fit


# --- Packing a matrix into a plan ----------------------------------------


struct PackedBinMatrix(Copyable, Movable):
    """A `BinnedMatrix` re-laid-out under a `BinLayoutPlan`.

    Host memory, ready to upload as one `UInt8` device buffer. `bin_at`
    mirrors `BinnedMatrix.bin_at` exactly, so the two are directly
    comparable and `matches_dense` can assert cell-for-cell equality.
    """

    var bytes: List[UInt8]
    var plan: BinLayoutPlan

    def __init__(out self, var bytes: List[UInt8], var plan: BinLayoutPlan):
        self.bytes = bytes^
        self.plan = plan^

    def bin_at(self, row: Int, feature: Int) raises -> Int:
        """The bin of (row, feature), decoded exactly as a kernel would."""
        # `element_index` is what bounds-checks both indices, so it runs
        # before anything indexes a plan table with them.
        var idx = self.plan.element_index(feature, row)
        var b = self.plan.block_of[feature]
        return unpack_value(
            self.bytes,
            self.plan.block_offset[b],
            idx,
            self.plan.block_width[b],
        )

    def bytes_len(self) -> Int:
        return len(self.bytes)


def pack_binned_matrix(
    data: BinnedMatrix, plan: BinLayoutPlan
) raises -> PackedBinMatrix:
    """Re-lay-out `data` under `plan`.

    Host work only: reads the dense column-major matrix and writes the
    packed buffer. `BinLayoutPlan` gives every block a tail pad, so blocks'
    writable footprints are disjoint and this loop is safe to parallelize
    across blocks. Within a block the writes overlap (adjacent elements
    share a byte whenever the width is not 8) and must stay ordered, which
    is why the inner loops are ascending and single-writer.

    Every bin id is checked against its width before it is stored, so a plan
    built from one matrix's observed extents fails loudly here when applied
    to a different matrix rather than silently truncating an id.
    """
    if data.n_rows != plan.n_rows or data.n_features != plan.n_features:
        raise Error("plan shape does not match the matrix")
    if len(data.bins) != data.n_rows * data.n_features:
        raise Error("binned matrix size must equal n_rows * n_features")

    var buf = List[UInt8](capacity=plan.total_bytes)
    buf.resize(plan.total_bytes, 0)
    var src = data.bins.unsafe_ptr()

    for b in range(plan.n_blocks()):
        var first = plan.block_first[b]
        var size = plan.block_size[b]
        var w = plan.block_width[b]
        var base = plan.block_offset[b]
        var limit = bins_representable(w)
        for r in range(plan.n_rows):
            var idx = r * size
            for i in range(size):
                var f = first + i
                var value = Int(src.unsafe_load(f * plan.n_rows + r))
                if value >= limit:
                    raise Error(
                        "bin id does not fit the width it is being packed"
                        " at"
                    )
                pack_value(buf, base, idx + i, w, value)

    return PackedBinMatrix(buf^, plan.copy())


def check_plan_covers_matrix(
    data: BinnedMatrix, plan: BinLayoutPlan
) raises:
    """Whether every cell of `data` fits the width `plan` gives its feature.

    The check `pack_binned_matrix` performs, without producing a buffer, so
    a caller can validate a plan against a second matrix (a validation set,
    or a prediction batch) before committing to it.
    """
    if data.n_rows != plan.n_rows or data.n_features != plan.n_features:
        raise Error("plan shape does not match the matrix")
    var src = data.bins.unsafe_ptr()
    for f in range(plan.n_features):
        var limit = bins_representable(plan.width[f])
        var col = f * plan.n_rows
        for r in range(plan.n_rows):
            if Int(src.unsafe_load(col + r)) >= limit:
                raise Error(
                    "binned layout does not support this dataset: "
                    + layout_support_name(LAYOUT_WIDTH_TOO_NARROW)
                )


def matches_dense(packed: PackedBinMatrix, data: BinnedMatrix) raises -> Bool:
    """Whether the packed matrix decodes to the dense one, cell for cell.

    The round-trip invariant the whole module rests on, as a function rather
    than as a claim: if this is ever false, a histogram built from the packed
    buffer is a histogram of different data.
    """
    if data.n_rows != packed.plan.n_rows:
        return False
    if data.n_features != packed.plan.n_features:
        return False
    for f in range(data.n_features):
        for r in range(data.n_rows):
            if packed.bin_at(r, f) != data.bin_at(r, f):
                return False
    return True


# --- Cost model -----------------------------------------------------------
#
# What one node's histogram costs under a layout, in units a profiler can
# weigh. Kept as separate counts rather than collapsed into a number,
# because their per-unit costs differ by orders of magnitude and only a
# measurement knows the ratios on a given device.


def sectors_touched(
    count: Int, span_bytes: Int, bytes_per_access: Int, sector_bytes: Int
) -> Int:
    """Memory sectors an ascending gather of `count` accesses touches.

    `span_bytes` is the extent the accesses are spread over, and
    `bytes_per_access` is how many contiguous bytes each one wants. The
    model is the two regimes and nothing between them:

        touched = min(count * ceil(bytes_per_access / sector),
                      ceil(span_bytes / sector))

    The left term is the scattered regime, where every access pays for its
    own sector and narrowing the data buys nothing. The right term is the
    streaming regime, where the whole extent is walked and narrowing the data
    pays in full. Node rows stay ascending (`gpu_active_rows.mojo` partitions
    stably), so a node is somewhere on this curve rather than randomly
    placed on it, and the crossover is at `count = span_bytes / sector`.

    Returns 0 when `sector_bytes` is `SECTOR_BYTES_UNKNOWN`, which is how an
    unmeasured device propagates into an undecided verdict instead of into a
    number.
    """
    if sector_bytes <= 0:
        return 0
    if count <= 0 or span_bytes <= 0:
        return 0
    var per_access = (bytes_per_access + sector_bytes - 1) // sector_bytes
    if per_access < 1:
        per_access = 1
    var scattered = count * per_access
    var streamed = (span_bytes + sector_bytes - 1) // sector_bytes
    if scattered < streamed:
        return scattered
    return streamed


def sectors_touched_occupancy(
    count: Int, span_bytes: Int, bytes_per_access: Int, sector_bytes: Int
) -> Int:
    """The same gather under an occupancy model instead of the two-regime
    bound, and it is here because the two disagree about the layout question.

    `sectors_touched` returns `min(scattered, streamed)`, which is a bound
    with a hard knee at `count = span / sector`. The smooth answer is the
    expected number of distinct sectors a `count`-element subset of the
    `n = span / bytes_per_access` slots touches, when a sector holds
    `k = sector / bytes_per_access` of them:

        E[distinct] = M * (1 - (1 - rho)^k)      M = n / k, rho = count / n

    The bound is tight for a sector carrying many wanted rows, which is almost
    surely touched, and loose for one carrying few, which is often empty. Both
    layouts have sectors of both kinds at different depths, so the correction
    does **not** run consistently in either layout's favor: summed over a
    tree it moves the feature-major-versus-blocked ratio by up to about 15%
    and in both directions, depending on the sector size (see the module
    docstring's table). That is small enough that the model choice is not what
    the layout decision turns on, and it is only knowable because both are
    computed. Reporting one and not the other would have been picking a
    layout by picking an approximation.

    Neither model is measured. Which one describes a device is decidable, and
    the module docstring says with what: hold `count` fixed, vary `n_rows`,
    and look for the knee. Returns 0 for `SECTOR_BYTES_UNKNOWN`, the same way
    `sectors_touched` does, so an unmeasured device stays unmeasured.
    """
    if sector_bytes <= 0:
        return 0
    if count <= 0 or span_bytes <= 0 or bytes_per_access <= 0:
        return 0
    var per_access = (bytes_per_access + sector_bytes - 1) // sector_bytes
    if per_access > 1:
        # One access spans several sectors and no two accesses share one, so
        # the occupancy question does not arise.
        return count * per_access
    var slots = span_bytes // bytes_per_access
    if slots < 1:
        slots = 1
    if count >= slots:
        return (span_bytes + sector_bytes - 1) // sector_bytes
    var k = sector_bytes // bytes_per_access
    if k < 1:
        k = 1
    var m = (slots + k - 1) // k
    var rho = Float64(count) / Float64(slots)
    # (1 - rho)^k without pow(), so this stays exact at the ends: rho = 0
    # gives 0 touched and rho = 1 gives every sector touched.
    var empty = 1.0
    var base = 1.0 - rho
    var e = k
    var acc = base
    while e > 0:
        if e & 1 == 1:
            empty *= acc
        acc *= acc
        e >>= 1
    var touched = Float64(m) * (1.0 - empty)
    var out = Int(touched + 0.5)
    if out < 1:
        out = 1
    if out > m:
        out = m
    return out


# Which of the two gather models a cost is priced under. There is no default
# in `decide_layout`: a caller states the model the way it states the sector
# size, because both are claims about a device.
comptime SECTOR_MODEL_BOUND = 0
comptime SECTOR_MODEL_OCCUPANCY = 1


def sector_model_name(model: Int) -> String:
    if model == SECTOR_MODEL_BOUND:
        return String("bound")
    if model == SECTOR_MODEL_OCCUPANCY:
        return String("occupancy")
    return String("unknown")


def sectors_under(
    model: Int,
    count: Int,
    span_bytes: Int,
    bytes_per_access: Int,
    sector_bytes: Int,
) raises -> Int:
    """`sectors_touched` or `sectors_touched_occupancy`, by name."""
    if model == SECTOR_MODEL_BOUND:
        return sectors_touched(
            count, span_bytes, bytes_per_access, sector_bytes
        )
    if model == SECTOR_MODEL_OCCUPANCY:
        return sectors_touched_occupancy(
            count, span_bytes, bytes_per_access, sector_bytes
        )
    raise Error("unknown sector model")


@fieldwise_init
struct LayoutNodeCost(Copyable, Movable):
    """The work one node's histogram costs under one layout.

    `bin_sectors` and `row_sectors` are memory sectors, not bytes: what a
    device pays for a scattered read is the sector, however few of its bytes
    are wanted. `decode_ops` is the ALU work `gpu_bin_packing.decode_cost`
    charges per element, zero at width 8. `launches` is one per *launch
    block*, which is `ceil(active / feature_group)` and has nothing to do with
    how many storage blocks the layout has. `shared_bytes` is the widest
    touched block's threadgroup allocation, which is an occupancy input rather
    than a time, so it is reported and never folded into a duration.

    Not modelled here at all: the three shared atomic adds per (row, feature).
    They are identical under every layout in this module, so they cancel in a
    comparison, but they do *not* cancel in a speedup: a layout that halves
    this cost moves the run time by less than half, by an unmeasured amount.
    """

    var bin_sectors: Int
    var row_sectors: Int
    var decode_ops: Int
    var launches: Int
    var shared_bytes: Int

    def is_priced(self) -> Bool:
        """Whether the sector counts came from a known sector size. A cost
        with zero sectors and nonzero decode work was built against
        `SECTOR_BYTES_UNKNOWN` and must not be compared with another."""
        return self.bin_sectors > 0


def layout_node_cost(
    plan: BinLayoutPlan,
    node_rows: Int,
    active: List[Int],
    sector_bytes: Int,
    feature_group: Int = 1,
    model: Int = SECTOR_MODEL_BOUND,
    quantized: Bool = True,
) raises -> LayoutNodeCost:
    """Price one node's histogram under `plan` for one active feature set.

    The three terms:

    - **bins**, per *pass* over a touched storage block, and the pass count is
      where the launch group enters. A launch owning `feature_group` slots
      spends `min(G, feature_group)` contiguous bytes of each row it fetches
      and leaves the rest of the sector on the floor, so a block of `G`
      features under a narrower group is walked `ceil(live / min(G, group))`
      times and pays for its sectors on each walk. This is the whole reason
      `G` and `GROUP` want to be the same number: a `G` wider than the group
      is a `G`-fold streaming regression, not a saving, and a `G` narrower
      than the group leaves the group's locality unused.
    - **row side**, per *launch block*, not per storage block. This is the
      correction that killed the argument this module used to make for
      blocking. `gpu_active_rows._range_hist_atomic_kernel[GROUP, BIN_CAP]`
      owns `GROUP` feature slots per threadgroup and reads `rows[begin + j]`
      and the gradient pair once for all of them, so the row side is divided
      by `feature_group` on *every* layout including the feature-major one
      that ships. Charging it once per storage block, as this function used
      to, handed a blocked layout a saving the launch shape had already
      taken, and tilted every comparison toward the candidate.
      `feature_group` defaults to 1 so an unstated group is the pessimistic
      answer rather than a flattering one; the shipping default comes from
      `gpu_tiling.free_feature_group`.
    - **decode.** `decode_cost(w).alu_ops` per decoded element, over the
      active features only.

    `quantized` selects between the kernel's two row loops: the default reads
    the interleaved Int32 pair `gq[2 * r]` as one eight-byte scattered access,
    and `quantized=False` reads `grad[r]` and `hess[r]` as two separate
    four-byte ones. The active-row index itself is a contiguous walk of the
    compacted range under both, not a scatter, and is charged as such.

    What is deliberately not modelled: L2 residency across blocks (a small
    node's row side may be cached after the first block, which would make
    the row-side term an overestimate), the shared atomics, the partition
    kernel's own bin reads, and the once-per-session pack and upload.
    """
    if node_rows < 0 or node_rows > plan.n_rows:
        raise Error("node row count out of range")
    if len(active) == 0:
        raise Error("active feature set must not be empty")
    if feature_group < 1:
        raise Error("feature group must be at least one slot")

    var live = List[Int](capacity=plan.n_blocks())
    live.resize(plan.n_blocks(), 0)
    for i in range(len(active)):
        var f = active[i]
        if f < 0 or f >= plan.n_features:
            raise Error("active feature index out of range")
        live[plan.block_of[f]] += 1

    var bin_sectors = 0
    var decode_ops = 0
    var shared = 0

    for b in range(plan.n_blocks()):
        if live[b] == 0:
            continue
        # The lanes one launch consumes in one pass, and how many passes the
        # block's live lanes therefore take. A launch narrower than the block
        # fetches the block's whole row and uses part of it, so the block is
        # walked again for the rest.
        var lanes = plan.block_size[b]
        if feature_group < lanes:
            lanes = feature_group
        var passes = (live[b] + lanes - 1) // lanes
        var pass_bits = lanes * plan.block_width[b]
        var pass_bytes = (pass_bits + 7) // 8
        bin_sectors += passes * sectors_under(
            model,
            node_rows,
            plan.block_data_bytes(b),
            pass_bytes,
            sector_bytes,
        )
        var per_element = decode_cost(plan.block_width[b]).alu_ops
        decode_ops += node_rows * live[b] * per_element
        var need = plan.shared_bytes_for_block(b)
        if need > shared:
            shared = need

    # One launch per `feature_group` active slots, and the row side once per
    # launch. `grid.x` is the slot group in the shipping kernel, so this is
    # the launch count as well as the row-side multiplier.
    var launches = (len(active) + feature_group - 1) // feature_group

    # The compacted row range is contiguous, so its own extent is what the
    # index walk spans, not the whole `n_rows` array.
    var index_span = node_rows * BYTES_ROW_SIDE_ELEM
    var row_once = sectors_under(
        model, node_rows, index_span, BYTES_ROW_SIDE_ELEM, sector_bytes
    )
    if quantized:
        row_once += sectors_under(
            model,
            node_rows,
            plan.n_rows * BYTES_GRAD_PAIR,
            BYTES_GRAD_PAIR,
            sector_bytes,
        )
    else:
        row_once += ROW_SIDE_FLOAT_ARRAYS * sectors_under(
            model,
            node_rows,
            plan.n_rows * BYTES_ROW_SIDE_ELEM,
            BYTES_ROW_SIDE_ELEM,
            sector_bytes,
        )
    var row_sectors = launches * row_once

    return LayoutNodeCost(
        bin_sectors, row_sectors, decode_ops, launches, shared
    )


@fieldwise_init
struct LayoutTreeCost(Copyable, Movable):
    """The same units as `LayoutNodeCost`, summed over a whole tree.

    A node comparison is the wrong denominator for this decision and
    `decide_layout` says so in its own docstring. A level of `2^d` nodes
    covers all `n_rows` between them however deep it is, so the layouts'
    relative cost is set by how the row mass is spread across depths and not
    by any one node. This is the sum that answers it.
    """

    var bin_sectors: Int
    var row_sectors: Int
    var decode_ops: Int
    var launches: Int

    def total_sectors(self) -> Int:
        return self.bin_sectors + self.row_sectors


def layout_tree_cost(
    plan: BinLayoutPlan,
    depth: Int,
    active: List[Int],
    sector_bytes: Int,
    feature_group: Int = 1,
    model: Int = SECTOR_MODEL_BOUND,
    quantized: Bool = True,
) raises -> LayoutTreeCost:
    """Sum `layout_node_cost` over a balanced tree of `depth` levels.

    Level `d` holds `2^d` nodes of `n_rows / 2^d` rows each, root included, so
    the walk covers every depth the histogram builder visits and weights each
    by how many nodes are there. Balanced is an idealization in two directions
    that partly cancel: leaf-wise growth makes some nodes much smaller than
    `2^-d` (which favors a blocked layout, since its advantage is
    `clamp(G / (rho * S), 1, G)` and rises as `rho` falls), and sibling
    subtraction builds only the smaller child of a pair (which favors it
    again, for the same reason). Both errors point the same way, so this sum
    is a **lower** bound on a blocked layout's advantage under a fixed sector
    model, and not a central estimate.

    What it is not is a lower bound on the advantage full stop, because the
    choice of sector model moves the answer further than either idealization
    does. Compute it under both `SECTOR_MODEL_BOUND` and
    `SECTOR_MODEL_OCCUPANCY` and read the pair, not either one.
    """
    if depth < 0:
        raise Error("tree depth must not be negative")

    var bin_sectors = 0
    var row_sectors = 0
    var decode_ops = 0
    var launches = 0
    var nodes = 1
    for _ in range(depth + 1):
        var node_rows = plan.n_rows // nodes
        if node_rows < 1:
            break
        var cost = layout_node_cost(
            plan, node_rows, active, sector_bytes, feature_group, model,
            quantized,
        )
        bin_sectors += nodes * cost.bin_sectors
        row_sectors += nodes * cost.row_sectors
        decode_ops += nodes * cost.decode_ops
        launches += nodes * cost.launches
        nodes *= 2
    return LayoutTreeCost(bin_sectors, row_sectors, decode_ops, launches)


@fieldwise_init
struct LayoutBuildCost(Copyable, Movable):
    """The once-per-session cost of putting a layout on the device.

    Separate from the per-node cost because it amortizes over a whole
    training run and therefore has to be weighed against a different
    denominator. A one-shot `build_histogram_gpu` pays it against a single
    histogram; a thousand-tree run pays it against tens of thousands.
    """

    var pack_writes: Int
    """Element writes the host packing pass performs: one per cell. Zero for
    a plan that is byte-identical to the dense buffer, which uploads
    `data.bins` directly."""

    var upload_bytes: Int
    """Bytes crossing to the device."""

    var resident_bytes: Int
    """Bytes the layout occupies on the device for the whole session."""


def layout_build_cost(plan: BinLayoutPlan) -> LayoutBuildCost:
    """Price building and uploading `plan` once.

    A passthrough plan costs no pack writes at all, because its buffer is
    the `BinnedMatrix.bins` the caller already holds. Charging the baseline
    a packing pass it does not perform would tilt every comparison toward
    the candidate, so the flag has to be exact and
    `check_passthrough_offsets` is what makes it so.
    """
    var writes = 0
    if not plan.is_passthrough():
        writes = plan.n_rows * plan.n_features
    return LayoutBuildCost(writes, plan.bytes(), plan.bytes())


@fieldwise_init
struct MeasuredLayoutCosts(Copyable, Movable):
    """Per-unit costs, in nanoseconds, that somebody measured on the device
    the decision is being made for.

    Every field defaults to zero, which means *unmeasured*, and an unmeasured
    model returns `LAYOUT_UNDECIDED` rather than a guess. There is
    deliberately no default set of numbers and no default sector size: a
    sector costs an order of magnitude more on a discrete GPU's HBM than on
    an M4's unified memory, and a constant baked in here would become the
    automatic threshold this module is not allowed to build.
    """

    var sector_bytes: Int
    var ns_per_sector: Float64
    var ns_per_decode_op: Float64
    var ns_per_launch: Float64
    var ns_per_upload_byte: Float64
    var ns_per_pack_write: Float64

    @staticmethod
    def unmeasured() -> MeasuredLayoutCosts:
        return MeasuredLayoutCosts(
            SECTOR_BYTES_UNKNOWN, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    def is_measured(self) -> Bool:
        """Whether every unit the model uses has a positive cost.

        A partly filled set counts as unmeasured. Mixing a measured sector
        cost with a guessed decode cost is precisely how a threshold gets
        built by accident, and the decode cost is the one a reader is most
        tempted to assume is negligible.
        """
        return (
            self.sector_bytes > 0
            and self.ns_per_sector > 0.0
            and self.ns_per_decode_op > 0.0
            and self.ns_per_launch > 0.0
            and self.ns_per_upload_byte > 0.0
            and self.ns_per_pack_write > 0.0
        )

    def evaluate_node(self, cost: LayoutNodeCost) -> Float64:
        """Nanoseconds this model predicts for one node."""
        return (
            Float64(cost.bin_sectors) * self.ns_per_sector
            + Float64(cost.row_sectors) * self.ns_per_sector
            + Float64(cost.decode_ops) * self.ns_per_decode_op
            + Float64(cost.launches) * self.ns_per_launch
        )

    def evaluate_build(self, cost: LayoutBuildCost) -> Float64:
        """Nanoseconds this model predicts for the once-per-session build."""
        return (
            Float64(cost.pack_writes) * self.ns_per_pack_write
            + Float64(cost.upload_bytes) * self.ns_per_upload_byte
        )


comptime LAYOUT_UNDECIDED = 0
comptime LAYOUT_PREFER_CANDIDATE = 1
comptime LAYOUT_PREFER_BASELINE = 2
comptime LAYOUT_UNSUPPORTED = 3


def layout_verdict_name(verdict: Int) -> String:
    if verdict == LAYOUT_UNDECIDED:
        return String("undecided")
    if verdict == LAYOUT_PREFER_CANDIDATE:
        return String("candidate")
    if verdict == LAYOUT_PREFER_BASELINE:
        return String("baseline")
    return String("unsupported")


@fieldwise_init
struct LayoutVerdict(Copyable, Movable):
    """The outcome of a comparison, with the numbers it was made from.

    `verdict` is `LAYOUT_UNDECIDED` unless the caller supplied a complete set
    of measurements, and `LAYOUT_UNSUPPORTED` when the candidate cannot run
    on the device at all. Both nanosecond fields are zero in those cases:
    an undecided verdict carries no number a caller could be tempted to
    compare anyway.
    """

    var verdict: Int
    var support: Int
    var candidate_ns: Float64
    var baseline_ns: Float64

    def is_decided(self) -> Bool:
        return (
            self.verdict == LAYOUT_PREFER_CANDIDATE
            or self.verdict == LAYOUT_PREFER_BASELINE
        )


def decide_layout(
    caps: DeviceCaps,
    candidate: BinLayoutPlan,
    baseline: BinLayoutPlan,
    node_rows: Int,
    active: List[Int],
    amortize_nodes: Int,
    costs: MeasuredLayoutCosts,
    feature_group: Int = 1,
    model: Int = SECTOR_MODEL_BOUND,
) raises -> LayoutVerdict:
    """Compare two plans over one representative node, with the build cost
    amortized across `amortize_nodes` nodes.

    Returns `LAYOUT_UNDECIDED` whenever `costs` is unmeasured, whatever the
    shapes are. That is the whole point of the signature: a caller who wants
    a decision has to produce the measurements, and a caller with none gets
    no decision to lean on.

    Even fully measured this is a *node* comparison with a linear build
    amortization, and a node is the wrong denominator: shallow nodes stream,
    deep nodes gather, and the two regimes prefer different layouts, so which
    layout wins a tree is set by how the row mass spreads across depths.
    `layout_tree_cost` is the sum that answers that and is what a layout
    argument should be built on. Still outside both: the partition kernel's
    own bin reads, the prediction path, the shared atomics, and the choice
    between the two sector models, which moves the answer across the decision
    line on its own. A policy that consumed this as a training-run verdict
    would be over-reading it; the staged benchmark in
    `bench/apple/bin_layout_plan.json` is what closes those gaps.
    """
    if amortize_nodes < 1:
        raise Error("amortization must cover at least one node")

    var support = layout_support(
        candidate.n_rows, candidate.n_features, candidate.n_bins
    )
    if support == LAYOUT_OK and not candidate.fits_shared_memory(caps):
        support = LAYOUT_SHARED_MEMORY
    if support != LAYOUT_OK:
        return LayoutVerdict(LAYOUT_UNSUPPORTED, support, 0.0, 0.0)

    if not costs.is_measured():
        return LayoutVerdict(LAYOUT_UNDECIDED, LAYOUT_OK, 0.0, 0.0)

    var cand_node = layout_node_cost(
        candidate, node_rows, active, costs.sector_bytes, feature_group, model
    )
    var base_node = layout_node_cost(
        baseline, node_rows, active, costs.sector_bytes, feature_group, model
    )
    var share = Float64(amortize_nodes)
    var cand_build = costs.evaluate_build(layout_build_cost(candidate))
    var base_build = costs.evaluate_build(layout_build_cost(baseline))
    var cand_ns = costs.evaluate_node(cand_node) + cand_build / share
    var base_ns = costs.evaluate_node(base_node) + base_build / share

    var verdict = LAYOUT_PREFER_BASELINE
    if cand_ns < base_ns:
        verdict = LAYOUT_PREFER_CANDIDATE
    return LayoutVerdict(verdict, LAYOUT_OK, cand_ns, base_ns)


# --- Convenience: the candidate set --------------------------------------


def baseline_plan(data: BinnedMatrix) raises -> BinLayoutPlan:
    """The layout the GPU backend uses today, as a plan.

    Every comparison is against this, and `is_passthrough` is true of it,
    so a caller that resolves to the baseline uploads `data.bins` unchanged
    and executes no decode instruction.
    """
    var widths = List[Int](capacity=data.n_features)
    widths.resize(data.n_features, BIN_WIDTH_FALLBACK)
    return plan_feature_major(
        data.n_rows, data.n_features, data.n_bins, widths^
    )


def baseline_descriptor(data: BinnedMatrix) raises -> BinStorageDescriptor:
    """The storage the GPU backend uploads today, as a descriptor.

    The conservative answer, and the one a caller that has made no layout
    decision should pass: one `UInt8` per cell, one feature per block,
    passthrough, markers intact at width 8 by construction. It goes through
    the same `storage_descriptor` path every candidate does, so the baseline
    is described by the same code that describes the candidates rather than by
    a hand-written constant that could drift from them.
    """
    return baseline_plan(data).storage_descriptor(
        data.missing_bin, data.cats
    )


def resolve_storage(
    data: BinnedMatrix,
    var preferred: BinLayoutPlan,
    allow_fallback: Bool = True,
) raises -> BinStorageDescriptor:
    """The descriptor to actually train on, given a preferred plan.

    The internal strategy switch this lane offers the trainer. A preferred
    plan the shipping kernels can read is returned as itself. One they cannot
    is *not* silently uploaded: with `allow_fallback` true (the default) the
    dense 8-bit baseline is returned instead, which is the established path
    and is byte-identical to what `GpuHistogramBuilder` already uploads; with
    it false the storage class that could not be served is raised.

    Marker loss is never falled back from. A plan whose widths would drop a
    missing bin or a category bin raises out of `storage_descriptor` before
    this function can choose anything, because "quietly train on a different
    dataset" is not a fallback, it is a wrong answer.
    """
    var desc = preferred.storage_descriptor(data.missing_bin, data.cats)
    if storage_is_shipping(desc.storage) and desc.block_features == 1:
        return desc^
    if not allow_fallback:
        desc.check_shipping()
        return desc^
    return baseline_descriptor(data)


def candidate_plans(
    data: BinnedMatrix,
    bin_counts: List[Int],
    caps: DeviceCaps,
    target_block: Int,
) raises -> List[BinLayoutPlan]:
    """The layout family for one dataset, in a fixed order, for a benchmark
    to walk.

    Order follows `bench/apple/bin_layout_plan.json`'s `variants` grouping:
    feature-major u8 (the baseline, and a passthrough plan), feature-major
    packed, feature-blocked u8, feature-blocked packed, row-major u8,
    row-major packed. One blocked plan per kind, at this call's
    `target_block`; the plan file sweeps several block widths, which is
    several calls rather than a longer list from one.

    Blocks are capped by `max_block_for_shared` as well as by
    `target_block`, so a returned plan always fits the device's threadgroup
    memory, and the two row-major plans appear only when a single block of
    every feature fits it (usually it does not, which is itself the answer
    for the training path). No plan is marked preferred and none is filtered
    out on a guess: the point of the list is that a benchmark measures all
    of them.
    """
    if len(bin_counts) != data.n_features:
        raise Error("bin count table must be one entry per feature")

    var packed_widths = widths_from_bin_counts(bin_counts, True)
    var flat_widths = widths_from_bin_counts(bin_counts, False)
    # Both marker classes, not just the categorical one: a packed width that
    # cannot hold a feature's missing bin is exactly as wrong as one that
    # cannot hold its category bins, and neither may reach a benchmark.
    check_markers_preserved(
        packed_widths, data.missing_bin, data.cats, data.n_bins
    )

    var out = List[BinLayoutPlan]()

    out.append(
        plan_feature_major(
            data.n_rows, data.n_features, data.n_bins, flat_widths.copy()
        )
    )
    out.append(
        plan_feature_major(
            data.n_rows, data.n_features, data.n_bins, packed_widths.copy()
        )
    )

    var flat_cap = max_block_for_shared(
        BIN_WIDTH_FALLBACK, data.n_bins, caps
    )
    var flat_block = target_block
    if flat_cap < flat_block:
        flat_block = flat_cap
    if flat_block >= 1:
        out.append(
            plan_feature_blocked(
                data.n_rows,
                data.n_features,
                data.n_bins,
                flat_widths.copy(),
                flat_block,
                True,
            )
        )

    # A packed blocked plan is capped by the *widest* width it may promote
    # to, which is the widest width present.
    var top_width = BIN_WIDTH_MIN
    for f in range(len(packed_widths)):
        if packed_widths[f] > top_width:
            top_width = packed_widths[f]
    var packed_cap = max_block_for_shared(top_width, data.n_bins, caps)
    var packed_block = target_block
    if packed_cap < packed_block:
        packed_block = packed_cap
    if packed_block >= 1:
        out.append(
            plan_feature_blocked(
                data.n_rows,
                data.n_features,
                data.n_bins,
                packed_widths.copy(),
                packed_block,
                True,
            )
        )

    # Row-major is one block of every feature, so it only exists when that
    # block fits shared memory. It is included for the prediction path and
    # for the deep-node regime even when training would not launch it.
    if flat_cap >= data.n_features:
        out.append(
            plan_row_major(
                data.n_rows,
                data.n_features,
                data.n_bins,
                promote_widths_uniform(flat_widths.copy()),
            )
        )
    if packed_cap >= data.n_features:
        out.append(
            plan_row_major(
                data.n_rows,
                data.n_features,
                data.n_bins,
                promote_widths_uniform(packed_widths.copy()),
            )
        )

    return out^
