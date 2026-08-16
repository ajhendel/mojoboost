"""Device-side active-row compaction.

The GPU trainer currently finds a node's rows by scanning all `n_rows` and
filtering on a per-row leaf id (see histogram_gpu.mojo): every histogram
costs a full pass over the dataset whatever the node's size, so a tree costs
`num_leaves * n_rows * n_features` bin reads where the CPU builder's
row-index lists cost `n_rows * n_features` per level. This module holds the
replacement: one device-resident permutation of the row indices in which
every live leaf owns a *contiguous* half-open range, so a node's histogram
reads exactly its own rows and nothing else.

The layout
----------
`rows[0 : n_active)` is a permutation of the rows this tree grows on (the
bag, or every row when unbagged). Leaf `node` owns `rows[begin : end)`, its
`LeafRange`. Splitting a leaf stably partitions its own range in place: the
rows going left come first, in their original relative order, then the rows
going right, also in their original relative order. The children's ranges
are therefore `[begin, begin + n_left)` and `[begin + n_left, end)` and the
live leaves keep tiling `[0, n_active)` exactly, with no gaps and no
overlap, for the whole tree. Nothing outside the parent's range is ever read
or written by its split, so sibling leaves cannot be disturbed.

Why stability matters
---------------------
The CPU grower's `partition_rows_into` (tree.mojo) is a block-counted,
prefix-summed, stable partition, and its comment records the property this
module has to keep: both sides come out in ascending row order whatever the
block count, so the parallel result is identical to the serial one, index
for index. Compaction here reproduces that exactly, which is what lets the
two backends be compared row list against row range. Since the partition is
stable in *buffer* order rather than in row-id order, a root seeded from a
GOSS sample that is not ascending still tracks the CPU grower's list, which
is seeded from the same sample in the same order.

Determinism
-----------
The device partition is a flag pass, an exclusive prefix sum, and a scatter.
Every destination index is a pure function of the element's position and the
integer prefix sums, so no atomic decides where a row lands and no run-to-run
scheduling difference can reorder the output. The prefix sums are exact
Int32 sums over 0/1 flags, so they cannot round.

Say it once more precisely, because a later reader will want to change the
launch shape again and this is the invariant that constrains what may be
changed. Write `flag(i)` for whether element `i` of the range goes left,
`P(j)` for the number of `i < j` with `flag(i)`, and `T` for `P(count)`. The
partition writes element `j` to `P(j)` when it goes left and to `T + (j -
P(j))` when it goes right. `P` and `T` are properties of the flags and of the
positions and of nothing else, so *no* choice of how the range is cut into
threadgroups, how many threadgroups there are, how many tiles each one walks,
or which kernel arm computes the prefix can move a single row. That is why
this module treats the block cap, the tile count, and the scan arm as
launch-shape knobs and says so at every one of them.

Collective primitives, and a rule this module repeals
-----------------------------------------------------
The scan inside a threadgroup is `max.gpu.primitives.block.prefix_sum`, warp
shuffles plus a single shared round, in place of the hand-rolled
Hillis-Steele scan that cost two barriers per doubling step. That repeals the
standing project rule against warp-level primitives, on the grounds that the
block and warp modules are portable across NVIDIA, AMD, and Metal and are
Modular's to tune, so calling them inherits their improvements instead of
maintaining a scan here forever. The two rules that stay are untouched: there
is still one portable source with no per-backend branch, and the scan is
still integer-only, so the determinism above is unaffected, `prefix_sum` over
Int32 being exact in whatever order it sums. The hand-rolled kernels are kept
as a fallback arm, selected by `MOJOTREES_GPU_SCAN_PRIMITIVES=0` or
`set_scan_primitives(False)`, so a backend where the primitive misbehaves has
somewhere to go and a benchmark can hold both arms in one process. The two
arms produce the identical permutation, asserted element for element in
tests/test_gpu_scan_primitives.mojo. Neither arm has been timed against the
other on any device, so nothing here claims one is faster.

What the partition costs, and what does not help
------------------------------------------------
Measured on an M4 with an interleaved A/B against the same buffers: a
leaf-wise chain of seven partitions from 5M rows down to 78k takes about
4.6 ms of device time in total, so a 100-tree, 31-leaf fit spends roughly a
second partitioning against ten-plus seconds of histograms; the phase trace
in train_gpu.mojo charges the phase more than that only because its sync
after `apply_split` pays queue latency the untraced fit does not. Carrying
the flag in the offset word so the scatter skips its bin gather measured
1.05x on that chain, resolved but small, because the flag pass has just
pulled the same lines into cache. A single-block in-place kernel for ranges
under one threadgroup (flag, scan, scatter in one launch, no scratch, no
copy back) measured 1.00x on 256 partitions of 600 rows, 78 us each either
way: the per-partition cost there is enqueue overhead, not launch count, so
it was not kept. Do not retry either without a new reason.

The launch count, and an open tension
-------------------------------------
A partition is three launches: flag and scan, scatter, copy back. It was four
until the block-sums scan was folded into the head of the scatter; see
`_scatter_kernel` for how, `_partition_grid` for the block cap that keeps the
folded scan's redundant reads bounded, and `_copy_back_kernel` for the design
that would remove the third launch and for the readers that stopped it being
built here.

The reason to want fewer launches is a stage-level profile of a 1M by 50
round of 100 trees on an M4, which charged partition 17.7 percent against
histogram's 49.3 and transfer's 28.2, and a follow-up audit of it which
believed roughly half of that 17.7 was fence attribution and put partition's
real floor at a four-launch minimum near 237 microseconds on a machine where
an enqueue costs about 20 microseconds and a host wait about 126. On that
reading a launch is worth removing on its own.

That reading is in tension with the paragraph above, which records that
collapsing a small range's four launches into one measured 1.00x. Both cannot
be the whole story, and this module does not know which is right: neither
number was re-measured by the change that removed the launch, and nothing
here claims the removal made anything faster. What it claims is that three
launches compute what four computed, element for element, which is a
statement about the permutation and not about time.

Bagging
-------
Compaction subsumes the OUT_OF_BAG sentinel: the bag is written into the
first `len(bag)` slots and the root range covers only those, so an unbagged
row is simply not in the range any kernel iterates. Nothing has to mark it,
match it, or skip it, and a bagged tree reads `len(bag)` rows per node
instead of `n_rows`.

Missing values and categoricals
-------------------------------
`RowRouting.goes_left` is the same rule as `Tree.goes_left`,
`SplitInfo.goes_left`, and the existing `_partition_kernel`: a categorical
node routes by 256-bit set membership (bin 0, the missing/unseen/dropped
bin, is never a member, so those rows go right), a numerical node sends the
feature's missing bin whichever way `default_left` says, and every other row
goes by the inclusive threshold. The host reference model and the device
kernels call the same rule with the same arguments, so they cannot drift.

Bounds safety
-------------
Ranges are validated on the host before any launch: a range must sit inside
`[0, n_active)`, `n_left` must lie in `[0, count]`, and child ids must be
new and distinct from the parent. Every kernel guards its own element index
against the range length, so the tail block of a partial threadgroup writes
nothing. Row ids are Int32 and are checked against `n_rows` when the root is
seeded, so the indirect bin loads `bins[feature * n_rows + row]` stay in the
matrix.

Allocation
----------
Every device buffer is allocated once, at construction, sized by `n_rows`
(plus one Int32 per scan block and one for the total). Splitting a leaf
allocates nothing on the device and, on the host, at most appends to the
range table. The scatter writes into a second full-size buffer and a copy
kernel folds just the parent's range back, so the ranges other leaves hold
stay valid: ping-ponging whole buffers would invalidate them.

**Ownership and growth.** `GpuActiveRows` owns exactly the index machinery:
`rows`, `scratch`, `offsets` (one Int32 per row each), the per-scan-block sums,
the one-element total, and the pinned staging buffers. Nothing here ever
reallocates: the buffers are sized for the *dataset*, and every tree, node,
and split works inside that fixed footprint, so there is no capacity growth
policy because there is no capacity that grows. What does grow is the host
`LeafRangeTable`, by at most two `LeafRange` appends per split, and it is
cleared at every `reset_root`. A dataset larger than the buffers is not a
resize, it is a different `GpuActiveRows`.

Empty leaves
------------
An empty range is a first-class state, not an error. `partition` of an empty
window enqueues nothing and reports zero rows left, and its two children are
recorded as empty ranges at the parent's own offset, so the live ranges keep
tiling `[0, n_active)`. `enqueue_range_histogram` for an empty node zeroes the
output and returns without launching an accumulation, which is the histogram
that node has. A leaf that holds no rows therefore costs one zeroing pass and
no row work, and never a special case in the caller.

Multiclass, bagging, and GOSS
-----------------------------
One `GpuActiveRows` holds one permutation, and a K-class round grows K trees
over it, one per class, in sequence: `begin_tree` reseeds the root and clears
the range table, so class `k + 1` starts from a clean permutation. The class
index does not reach this module at all, because which rows exist does not
depend on the class; it reaches the gradient plane instead, through
`LeafFrontier.plane` and `gpu_leaf_batching.ITEM_PLANE`.

Bagging and GOSS reach it in exactly one way: as the `bag` list handed to
`begin_tree`, which becomes the live prefix. There are no per-row weights on
the device and none are wanted. GOSS's amplification of the small-gradient
sample is applied to the gradients before they are uploaded (see
`goss.apply_goss_scaling`), so a row's weight is already inside `grad[r]` and
`hess[r]` by the time any kernel here reads it. A second weight vector would
be a second place to scale, and two places to scale is one too many.

What is not here
----------------
This module owns no dataset, gradients, or histogram output. Those live in
`GpuHistogramBuilder`, and the range-histogram entry points below take them
as pointers so the builder can call in without a second copy of anything.
See `handoffs/apple_a1_active_rows.md` for the step-by-step replacement of
the leaf-id filtering path and `handoffs/connect_02_gpu_dataflow.md` for the
frontier seam (`begin_tree_with`, `apply_commit`, `check_frontier`) this
module offers a grower.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, global_idx, grid_dim, thread_idx
from std.math import round
from std.memory import stack_allocation
from std.sys import has_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import broadcast as block_broadcast
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier

from .binning import BinnedMatrix
from .categorical import CatBitset, cat_empty
from .gpu_frontier import CommitPlan, LeafFrontier
from .gpu_histogram_specializations import MAX_BINS
from .gpu_tiling import (
    HIST_FEATURE_GROUP_LADDER,
    HIST_FEATURE_GROUP_MAX,
    STRATEGY_ATOMIC,
    STRATEGY_TILED,
    DeviceCaps,
    HistogramTiling,
    derive_block_threads,
    derive_tiling,
    free_feature_group,
    histogram_bin_capacity,
    histogram_shared_bytes,
    is_feature_group_width,
)
from .gpu_split_search import CAT_WORDS, CAT_WORD_BITS
from .parallel import _env_int
from .split import SplitInfo

# Row indices and leaf ids cross into the kernels as Int32.
comptime MAX_ROWS = Int(Int32.MAX)


# --- The step descriptor --------------------------------------------------
#
# One committed growth step, written by the device and read by the device.
#
# Why this table exists at all. Every kernel below takes the split it is
# routing by, and the window it is routing, as *launch arguments*: the host
# knows both because the host chose the split. A device-owned tree
# (`gpu_tree_tables.mojo`) moves that choice onto the device, and then the
# host does not know either one and cannot pass them without reading them
# back first. A blocking readback on an Apple M4 was measured at 606
# microseconds against 3.7 microseconds of actual byte movement
# (`docs/METAL_TIMELINE.md`), and there are 31 of them in a default 31-leaf
# tree, so the readback is the cost and removing it is the point.
#
# The descriptor is therefore the seam. The commit kernel in
# `gpu_tree_tables.mojo` writes it as the last act of a commit, and the
# partition and histogram
# kernels here read the same words instead of taking them as arguments. It
# is a flat Int32 row, in the same style as the split records in
# `gpu_split_search.mojo`, because a flat row is what a kernel can index
# without knowing anything about the layout of the tables it came from.
#
# The layout lives *here*, in the consumer, and not in the producer, purely
# to keep the import graph a tree: `gpu_tree_tables` already imports this
# module's neighbours, and nothing in this file may import
# `gpu_tree_tables` without creating a cycle through
# `histogram_gpu -> gpu_active_rows`.
#
# **Everything is written on every commit and nothing is left over from the
# step before.** A step that commits nothing writes `STEP_LIVE = 0` and
# leaves the rest at whatever it held, so `STEP_LIVE` is the only word a
# reader may consult before it has checked `STEP_LIVE`. Every kernel here
# that takes a descriptor returns immediately when it is zero, which is what
# makes a whole tree's worth of steps safe to enqueue up front: the steps
# past the end of growth run, read one word, and do nothing.

comptime STEP_LIVE = 0
"""1 when the step this descriptor describes committed a split, 0 otherwise.
The guard every descriptor-driven kernel checks first."""

comptime STEP_ROW_BEGIN = 1
"""Start of the split leaf's window into the active-row permutation. The
partition rewrites exactly `[STEP_ROW_BEGIN, STEP_ROW_BEGIN + STEP_ROW_COUNT)`
and nothing else, which is what leaves every other leaf's window intact."""

comptime STEP_ROW_COUNT = 2
"""Rows in that window."""

comptime STEP_FEATURE = 3
comptime STEP_THRESHOLD = 4
comptime STEP_MISSING_BIN = 5
comptime STEP_DEFAULT_LEFT = 6
comptime STEP_IS_CAT = 7
"""The five words of `RowRouting`, in the same meanings: the split feature,
the inclusive threshold bin (-1 on a categorical split), the feature's
missing bin (-1 when it reserves none), the direction missing rows take, and
whether the node routes by category set instead of by threshold."""

comptime STEP_BUILT_BEGIN = 8
comptime STEP_BUILT_COUNT = 9
"""The window of the child whose histogram is *accumulated* rather than
derived, which is the smaller of the two under
`gpu_frontier.subtraction_builds_left`. Written by the commit because it is
`begin`/`begin + n_left` arithmetic the commit has already done, and reading
it here saves the histogram kernel from having to know which child won that
comparison."""

comptime STEP_BUILT_SLOT = 10
"""The resident pool slot the accumulated child's histogram is written
into."""

comptime STEP_SUB_SLOT = 11
"""The pool slot the accumulation is also subtracted from, which is the
parent's slot and therefore the derived sibling's slot after the commit
reassigns it. The device mirror of `enqueue_resident_leaf_subtracting`'s
`parent_slot`."""

comptime STEP_LEFT_SLOT = 12
comptime STEP_RIGHT_SLOT = 13
"""The pool slots the two children's histograms end up in, whichever of them
was built and whichever was derived. What the next search reads."""

comptime STEP_LEFT_REC = 14
comptime STEP_RIGHT_REC = 15
"""The split-record slots the two children's records must end up in, which
are the frontier slots' own record indices. The search itself writes a fixed
pair of scratch records, because a launch's record range is a host-side
constant; a device kernel then copies those two into these two. See
`gpu_tree_tables._copy_records_kernel`."""

comptime STEP_CAT0 = 16
"""First of `CAT_WORDS` category-set words, in the split record's own 16-bit
packing. Copied out of the record without reinterpretation, and unpacked
into the four UInt64 the routing rule wants by `_step_cat_word` below."""

comptime STEP_WORDS = STEP_CAT0 + CAT_WORDS


@always_inline
def _step_cat_word(
    desc: MutPointer[Int32, MutAnyOrigin], w: Int
) -> UInt64:
    """UInt64 word `w` of the descriptor's category set.

    `_row_goes_left` wants the set as four UInt64; a split record carries it
    as sixteen Int32 holding sixteen bits each, and the tree table stores the
    record's words verbatim so that no bit is reinterpreted between the
    search that chose the set and the partition that routes by it. This is
    the one place the two spellings meet, and it is a reassembly rather than
    a reinterpretation: bit `b` of the set is bit `b % CAT_WORD_BITS` of Int32
    word `b // CAT_WORD_BITS`, and bit `b` of UInt64 word `w` is bit `b - 64 *
    w`, so four consecutive Int32 words make one UInt64 word in ascending
    order.

    The mask against 0xFFFF is not decoration. A 16-bit field held in an
    Int32 is sign-extended by the widening conversion whenever bit 15 is set,
    which would set every bit above it in the UInt64 and put categories into
    the set that the search never chose. That is a fault a small fixture
    would miss, because bit 15 of word 0 is category 15 and most category
    sets are sparse and low.
    """
    var out = UInt64(0)
    comptime WORDS_PER_U64 = 64 // CAT_WORD_BITS
    for k in range(WORDS_PER_U64):
        var idx = w * WORDS_PER_U64 + k
        if idx < CAT_WORDS:
            var v = UInt64(
                UInt32(Int32(desc[unsafe_offset = STEP_CAT0 + idx][0]))
            ) & UInt64(0xFFFF)
            out |= v << UInt64(k * CAT_WORD_BITS)
    return out

# The scan kernels keep one Int32 per thread in shared memory, so the block
# size they can be launched with is bounded by this allocation rather than by
# the device maximum. 1024 Int32 is 4 KB, well inside every backend's
# per-threadgroup budget, and 1024 is also the largest block any supported
# device accepts.
comptime SCAN_MAX_THREADS = 1024


def _scan_primitive_width_supported(threads: Int) -> Bool:
    """Whether the primitive scan arm has a kernel instantiated at this
    threadgroup width.

    `block.prefix_sum` takes its block size as a *compile-time* parameter, so
    the primitive kernels cannot read `block_dim.x` the way the hand-rolled
    ones do: each width is a separate instantiation and the host has to pick
    one. Two constraints bound the menu. The primitive itself carries
    `constrained` "Block size must be a multiple of warp size", which
    `gpu_tiling.clamp_block_threads` already satisfies unconditionally by
    rounding every width down to a multiple of `WARP_GRANULARITY` (64, which
    is a multiple of the 32-wide warp on NVIDIA and Metal and is AMD's
    wavefront exactly). And every instantiation is compile time spent, so the
    menu is the four powers of two from 128 up, which is every width
    `derive_block_threads` produces on a supported device: the target is 256
    and the clamp only lowers it on a device whose maximum threadgroup is
    smaller than that, which none of the three backends has.

    A width outside the menu is reachable only through an explicit
    `MOJOTREES_GPU_BLOCK_THREADS` override (192 and 320 are legal widths that
    are not on it). That is not an error and is not silently rounded, because
    rounding the width would change the block count and the block count is
    what `block_sums_dev` was sized for: such a width simply runs the
    hand-rolled arm, which reads its width from `block_dim.x` and serves any
    of them. Both arms produce the same permutation, so this changes nothing
    a caller can observe.
    """
    return threads == 128 or threads == 256 or threads == 512 or (
        threads == 1024
    )


def _partition_grid(n: Int, threads: Int, cap: Int) raises -> Tuple[Int, Int]:
    """The partition's launch geometry: how many threadgroups, and how many
    `threads`-wide tiles each of them walks.

    Before this lane the answer was always "one tile per threadgroup, as many
    threadgroups as it takes", and the per-block left counts that shape
    produced were turned into global offsets by a *separate* launch. The
    scatter now does that scan itself, at its own head, which is what removes
    the launch; the price is that every threadgroup of the scatter re-reads
    the whole block-sums array, so the block count has to be bounded or the
    re-read grows quadratically in the range length. Bounding it is what this
    function is for.

    The rule, given a range of `n` elements:

        tiles_total = ceil(n / threads)      tiles the range needs at all
        blocks      = min(tiles_total, cap)  bounded threadgroup count
        tiles       = ceil(tiles_total / blocks)   tiles each one walks
        blocks      = ceil(tiles_total / tiles)    trim the empty tail

    The second assignment to `blocks` only ever lowers it, and it exists so
    that a range like `cap + 1` tiles does not launch `cap` threadgroups of
    which half own nothing: with two tiles each, `ceil((cap + 1) / 2)` of them
    is enough to cover the range. It cannot raise `blocks` above `cap`,
    because `tiles >= tiles_total / cap` gives `blocks <= cap`.

    Two properties the kernels depend on, both of which hold by construction.
    First, `blocks * tiles * threads >= n`, so every element of the range is
    covered by exactly one (block, tile, lane) triple. Second, the blocks
    partition the range into *contiguous* chunks in ascending order, block `b`
    owning `[b * tiles * threads, (b + 1) * tiles * threads)` clipped to the
    range, which is what keeps the scatter's destination arithmetic a stable
    partition.

    Note what this does *not* change: while `tiles_total <= cap` the geometry
    is identical to what the module launched before this lane, one tile per
    threadgroup. With the default cap (`block_threads`, see
    `GpuActiveRows.partition_block_cap`) that covers every range shorter than
    `threads * threads` elements, which at the usual 256-wide threadgroup is
    every range under 65,536 rows. Only ranges larger than that see the
    multi-tile loop at all, so on a small dataset, and on every node below the
    first level or two of a large one, the flag pass runs exactly the code
    path it ran before.
    """
    if n < 1:
        raise Error("partition grid needs at least one element")
    if threads < 1:
        raise Error("partition grid needs a positive threadgroup width")
    if cap < 1:
        raise Error("partition grid needs a positive block cap")
    var tiles_total = (n + threads - 1) // threads
    var blocks = tiles_total
    if blocks > cap:
        blocks = cap
    var tiles = (tiles_total + blocks - 1) // blocks
    blocks = (tiles_total + tiles - 1) // tiles
    return (blocks, tiles)


# Features one histogram threadgroup may accumulate at once, and the top of
# the ladder the kernel family is instantiated over. One threadgroup's shared
# histogram is three Int32 planes of `GROUP * BIN_CAP` cells, where `BIN_CAP`
# is the dataset's bin count rounded up the capacity ladder
# (`gpu_tiling.histogram_bin_capacity`) rather than fixed at `MAX_BINS`, so a
# group of G on a 64-bin dataset costs `3 * G * 64 * 4` bytes and not
# `3 * G * 256 * 4`. That is what makes widths past the pairing affordable at
# all: group 8 at 64 bins occupies the 6 KiB group 2 used to occupy at every
# bin count. `set_feature_group` still refuses a width that is not a rung,
# 3 among them, because a width with no instantiation cannot launch.
comptime FEATURE_GROUP_MAX = HIST_FEATURE_GROUP_MAX
comptime FEATURE_GROUP_LADDER = HIST_FEATURE_GROUP_LADDER


@always_inline
def _row_goes_left(
    bin: Int32,
    threshold_bin: Int32,
    missing_bin: Int32,
    default_left: Int32,
    is_categorical: Int32,
    cat0: UInt64,
    cat1: UInt64,
    cat2: UInt64,
    cat3: UInt64,
) -> Bool:
    """The routing rule, device side. Identical to `Tree.goes_left` and to
    the existing `_partition_kernel`: set membership for a categorical node,
    the default direction for the missing bin, the inclusive threshold
    otherwise. A feature with no missing bin passes -1, which no bin id can
    equal, so those rows fall through to the threshold."""
    if is_categorical != 0:
        var word: UInt64
        var w = Int(bin) >> 6
        if w == 0:
            word = cat0
        elif w == 1:
            word = cat1
        elif w == 2:
            word = cat2
        else:
            word = cat3
        return ((word >> UInt64(Int(bin) & 63)) & 1) != 0
    if bin == missing_bin:
        return default_left != 0
    return bin <= threshold_bin


def _iota_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
):
    """Seed the active-row buffer with the identity permutation. The unbagged
    root is every row in ascending order, which is what the CPU grower builds
    its root list as."""
    var r = global_idx.x
    if r < Int(n_rows):
        rows[unsafe_offset=r] = Int32(r)


def _zero_int32_kernel(
    buf: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
):
    """Zero an Int32 device range. Used instead of `enqueue_memset` where the
    caller handed in a pointer rather than the buffer."""
    var i = global_idx.x
    if i < Int(n):
        buf[unsafe_offset=i] = Int32(0)


def _zero_slot_desc_kernel(
    pool: MutPointer[Int32, MutAnyOrigin],
    cells: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
):
    """Zero the resident pool slot the step descriptor names, and only that
    one.

    The atomic histogram strategy folds its threadgroup partials into the
    output with `Atomic.fetch_add`, so it accumulates onto whatever the
    destination already held and the destination has to start at zero. On the
    host-driven path `enqueue_range_histogram` zeroes it with
    `_zero_int32_kernel` over a pointer the host has already offset to the
    right slot. Here the slot is a device value, so the offset is computed
    inside the kernel from `STEP_BUILT_SLOT` instead.

    Guarded on `STEP_LIVE` for a reason that is worth spelling out, because
    getting it wrong would be silent and destructive rather than merely
    wrong: on a step that committed nothing the descriptor's slot word is
    whatever it was left at, and zeroing an arbitrary slot would erase a live
    leaf's histogram, which would then be searched and would produce a
    plausible split with no rows behind it. A dead step must touch nothing at
    all, and this is the kernel where that mattered most.

    Grid-strided, so the caller sizes the grid to the pool's slot width once
    and never to a per-step quantity.
    """
    if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
        return
    var base = Int(desc[unsafe_offset=STEP_BUILT_SLOT][0]) * Int(cells)
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var i = Int(global_idx.x)
    while i < Int(cells):
        pool[unsafe_offset = base + i] = Int32(0)
        i += stride


def _flag_scan_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    begin: Int32,
    count: Int32,
    feature: Int32,
    threshold_bin: Int32,
    missing_bin: Int32,
    default_left: Int32,
    is_categorical: Int32,
    cat0: UInt64,
    cat1: UInt64,
    cat2: UInt64,
    cat3: UInt64,
    tiles: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Stage one of the stable partition: flag each row of the range and scan
    the flags within each threadgroup.

    Block `b` owns the contiguous chunk of `tiles * block_dim.x` elements
    starting at `b * tiles * block_dim.x`, clipped to the range, and walks it
    one `block_dim.x`-wide tile at a time. `offsets[j]` becomes the number of
    left-going rows before `j` *inside j's own chunk*, shifted left one bit,
    with j's own flag in the low bit; `block_sums[b]` is the number of
    left-going rows in the whole chunk. The running count across the chunk's
    tiles is carried in a register that every thread of the block computes
    identically, exactly the way the block-sums scan this lane deleted carried
    its own running total.

    Before this lane `tiles` was always 1 and the chunk was always one tile,
    so this loop had no iterations to run and the code below reduces to what
    was here before whenever `_partition_grid` returns `tiles == 1`, which it
    does for every range shorter than `block_threads * block_threads`
    elements. What changed is only that a longer range is now covered by a
    bounded number of threadgroups walking several tiles each rather than by
    an unbounded number walking one, which is what lets the scatter scan the
    block sums itself instead of a separate launch doing it. The permutation
    cannot notice: a row's destination is a function of the routing flags and
    the row's position in the range, and neither depends on how the range was
    cut into blocks. See `_scatter_kernel` for that argument written out.

    Carrying the flag in the offset word is what lets the scatter route the
    row without gathering its bin a second time: the bin loads are the one
    random-access read of the partition (`bins[feature * n_rows + row]`, one
    byte from a fresh cache line per row once the leaf's rows are sparse in
    the dataset), so reading each once instead of twice halves the
    partition's gathered traffic, and the packed word costs nothing the plain
    prefix did not. The prefix is bounded by the chunk length, which is
    bounded by the range length, which is bounded by `n_rows` and therefore
    by `Int32.MAX / 2` for any dataset this module accepts, so the shift
    cannot overflow. Threads past the end of the range contribute a zero
    flag, so a partial tail tile still sums correctly.

    The scan is the plain Hillis-Steele shared-memory scan: two barriers per
    doubling step, with the read separated from the write so no thread reads a
    slot another has already advanced. This is the *fallback* arm.
    `_flag_scan_prim_kernel` is the default and scans with `block.prefix_sum`
    instead; see the module docstring for which rule that repeals and why.
    This kernel is kept rather than deleted because it reads its width from
    `block_dim.x` and so serves any threadgroup width, including the ones no
    primitive instantiation exists for, and because a benchmark wanting both
    arms in one process needs both to be present. It produces the identical
    offsets and block sums.

    The descriptor arm
    ------------------
    With `use_desc`, the window and the routing rule are read out of `desc`
    (see "The step descriptor" at the head of this module) instead of out of
    the launch arguments, and the launch arguments for those eight values are
    ignored. That is the whole of what a device-owned tree needs from this
    kernel: the split it routes by was chosen by a kernel rather than by the
    host, so it lives in device memory rather than in a register the host
    filled.

    Nothing else changes, and in particular the *geometry* does not. A
    descriptor-driven launch cannot size its grid to a range length it does
    not know, so it is launched at a grid that covers the whole active
    prefix, which is an upper bound on any window. That is safe because the
    permutation this partition computes is a function of the routing flags
    and of each row's position in the range and of nothing else; the argument
    is written out in `_scatter_kernel`, which is where the destination
    arithmetic lives. Cutting a short range into more chunks than it needs
    therefore produces the same permutation as cutting it into few. What an
    over-provisioned grid costs is threadgroups that own nothing, and the
    early return below is what keeps that cost to a launch slot.

    Whether the extra threadgroups are cheap enough to pay for the readback
    they remove is UNMEASURED: no benchmark of this arm has been run.
    """
    var tid = thread_idx.x
    var nthreads = block_dim.x
    var n = Int(count)
    var row_begin = Int(begin)
    var feat = feature
    var thr = threshold_bin
    var miss = missing_bin
    var dleft = default_left
    var iscat = is_categorical
    var c0 = cat0
    var c1 = cat1
    var c2 = cat2
    var c3 = cat3
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        row_begin = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
        feat = desc[unsafe_offset=STEP_FEATURE][0]
        thr = desc[unsafe_offset=STEP_THRESHOLD][0]
        miss = desc[unsafe_offset=STEP_MISSING_BIN][0]
        dleft = desc[unsafe_offset=STEP_DEFAULT_LEFT][0]
        iscat = desc[unsafe_offset=STEP_IS_CAT][0]
        c0 = _step_cat_word(desc, 0)
        c1 = _step_cat_word(desc, 1)
        c2 = _step_cat_word(desc, 2)
        c3 = _step_cat_word(desc, 3)
    var col = Int(feat) * Int(n_rows)
    var chunk = Int(tiles) * nthreads
    var base = Int(block_idx.x) * chunk

    # A block whose whole chunk lies past the end of the range. `base`, `n`
    # and `tiles` are block-uniform, so the entire threadgroup returns
    # together and no thread is left waiting at a barrier the others skipped.
    # The zero it writes is exactly what the tile loop below would have
    # accumulated, so this is an early exit and not a behavior change; the
    # host-argument arm reaches it only on a grid that was already
    # over-provisioned, which `_partition_grid` does not produce.
    if base >= n:
        if tid == 0:
            block_sums[unsafe_offset = Int(block_idx.x)] = Int32(0)
        return

    var s = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var carry = Int32(0)
    for t in range(Int(tiles)):
        var j = base + t * nthreads + tid

        var flag = Int32(0)
        if j < n:
            var row = rows[unsafe_offset = row_begin + j][0]
            var bin = Int32(bins[unsafe_offset = col + Int(row)])
            if _row_goes_left(
                bin, thr, miss, dleft, iscat, c0, c1, c2, c3
            ):
                flag = Int32(1)
        s[unsafe_offset=tid] = flag
        barrier()

        # Every thread runs the same number of steps (the bound is the
        # uniform block size), so the barriers are reached by the whole
        # threadgroup. `tiles` is uniform too, so the tile loop itself
        # cannot leave a thread behind at a barrier.
        var offset = 1
        while offset < nthreads:
            var carried = Int32(0)
            if tid >= offset:
                carried = s[unsafe_offset = tid - offset][0]
            barrier()
            if tid >= offset:
                s[unsafe_offset=tid] = s[unsafe_offset=tid][0] + carried
            barrier()
            offset += offset

        if j < n:
            # Inclusive minus own flag is the exclusive prefix within the
            # tile; plus the chunk's running count it is the exclusive
            # prefix within the chunk. The flag rides in the low bit for
            # the scatter.
            offsets[unsafe_offset=j] = (
                (carry + s[unsafe_offset=tid][0] - flag) * Int32(2)
            ) + flag
        carry += s[unsafe_offset = nthreads - 1][0]
        # The tile total has been read by every thread; the next iteration
        # may now overwrite the shared buffer.
        barrier()

    if tid == 0:
        block_sums[unsafe_offset = Int(block_idx.x)] = carry


def _flag_scan_prim_kernel[block_size: Int](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    begin: Int32,
    count: Int32,
    feature: Int32,
    threshold_bin: Int32,
    missing_bin: Int32,
    default_left: Int32,
    is_categorical: Int32,
    cat0: UInt64,
    cat1: UInt64,
    cat2: UInt64,
    cat3: UInt64,
    tiles: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Stage one, scanned with `block.prefix_sum` instead of by hand.

    Everything a later stage reads is byte for byte what `_flag_scan_kernel`
    writes: `offsets[j]` is still `(exclusive_prefix_within_j's_chunk << 1) |
    flag`, and `block_sums[b]` is still block b's left count over the same
    contiguous chunk, so the two arms remain interchangeable stage by stage.
    The difference is only how the intra-tile scan is computed.
    `block.prefix_sum` is warp shuffles plus one shared round rather than a
    doubling loop with two barriers per step, and it sums exact Int32 flags,
    so the permutation cannot differ; the test file asserts that against the
    hand-rolled arm rather than asserting it here in a comment.

    Signature, checked against the installed toolchain rather than
    remembered: `block.prefix_sum[dtype: DType, //, *, block_size: Int,
    exclusive: Bool = False](val: Scalar[dtype]) -> Scalar[dtype]`. The
    inclusive form is asked for because both things this kernel needs come
    out of it: the exclusive prefix is the inclusive one minus the thread's
    own flag, exactly as the hand-rolled arm derives it, and the last
    thread's inclusive value is the block total. Asking for the exclusive
    form instead would need a second collective to recover the total.

    `block_size` is a compile-time parameter of the primitive, not a
    `block_dim.x` read, which is why this kernel is parametric and why the
    launch has to pass the matching `block_dim`; see
    `_scan_primitive_width_supported` for the menu of widths and why a width
    outside it falls back rather than being rounded. The primitive is a
    collective, so every thread of the threadgroup reaches it: the flag is
    computed as zero for threads past the end of the range and the call sits
    outside the `j < n` guard, which is also what keeps a partial tail tile
    summing correctly.

    The tile loop and the register-carried running count are the same
    construction `_flag_scan_kernel` uses and the same one the deleted
    block-sums kernel used to carry its total across chunks: the chunk total
    reaches every thread through `block.broadcast[block_size=block_size](
    inclusive, src_thread=block_size - 1)` rather than through a shared slot
    every thread re-reads. Signature, checked against the toolchain:
    `block.broadcast[dtype: DType, width: SIMDLength, //, *, block_size:
    Int](val: SIMD[dtype, width], src_thread: Int = 0) -> SIMD[dtype,
    width]`. Broadcasting from the last thread is what keeps `carry`
    bit-identical on every thread of the block, which is what makes the
    tile-to-tile carry deterministic; summing exact Int32 in a fixed tile
    order, it cannot depend on scheduling.

    The trailing `barrier()` is deliberate and is not the primitive's job to
    supply: the next iteration calls the same collectives again, and this
    kernel does not get to assume anything about whether a primitive fences
    its own scratch on exit. `tiles` is a uniform argument, so every thread
    of the block runs the same number of iterations and reaches every
    collective.

    `desc` and `use_desc` are the descriptor arm, identical in meaning to
    `_flag_scan_kernel`'s: read that kernel's docstring for what the arm is
    for and why an over-provisioned grid computes the same permutation. The
    early return matters more here than there, because this arm's collectives
    are called once per tile whether or not the tile owns any element, so an
    empty block that ran the loop would pay `tiles` prefix sums and `tiles`
    broadcasts to sum a column of zeros.
    """
    var tid = thread_idx.x
    var n = Int(count)
    var row_begin = Int(begin)
    var feat = feature
    var thr = threshold_bin
    var miss = missing_bin
    var dleft = default_left
    var iscat = is_categorical
    var c0 = cat0
    var c1 = cat1
    var c2 = cat2
    var c3 = cat3
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        row_begin = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
        feat = desc[unsafe_offset=STEP_FEATURE][0]
        thr = desc[unsafe_offset=STEP_THRESHOLD][0]
        miss = desc[unsafe_offset=STEP_MISSING_BIN][0]
        dleft = desc[unsafe_offset=STEP_DEFAULT_LEFT][0]
        iscat = desc[unsafe_offset=STEP_IS_CAT][0]
        c0 = _step_cat_word(desc, 0)
        c1 = _step_cat_word(desc, 1)
        c2 = _step_cat_word(desc, 2)
        c3 = _step_cat_word(desc, 3)
    var col = Int(feat) * Int(n_rows)
    var chunk = Int(tiles) * block_size
    var base = Int(block_idx.x) * chunk

    # Block-uniform, so the whole threadgroup leaves together and no
    # collective is reached by part of a block. See `_flag_scan_kernel`.
    if base >= n:
        if tid == 0:
            block_sums[unsafe_offset = Int(block_idx.x)] = Int32(0)
        return

    var carry = Int32(0)
    for t in range(Int(tiles)):
        var j = base + t * block_size + tid

        var flag = Int32(0)
        if j < n:
            var row = rows[unsafe_offset = row_begin + j][0]
            var bin = Int32(bins[unsafe_offset = col + Int(row)])
            if _row_goes_left(
                bin, thr, miss, dleft, iscat, c0, c1, c2, c3
            ):
                flag = Int32(1)

        var inclusive = block_prefix_sum[
            block_size=block_size, exclusive=False
        ](flag)
        var tile_total = block_broadcast[block_size=block_size](
            inclusive, src_thread = block_size - 1
        )

        if j < n:
            offsets[unsafe_offset=j] = (
                (carry + inclusive - flag) * Int32(2)
            ) + flag
        carry += tile_total
        barrier()

    if tid == 0:
        block_sums[unsafe_offset = Int(block_idx.x)] = carry


def _scatter_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    total: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    tiles: Int32,
    n_blocks: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Stage two: scan the block sums at the head of the block, then write
    each row of this block's chunk to its compacted slot.

    Before this lane the block-sums scan was a launch of its own, one
    threadgroup that turned the per-block left counts into per-block starting
    offsets in place and wrote the range's total. It is now the first thing
    every scatter block does, over the same block sums, with the same chunked
    single-threadgroup scan the deleted kernel used; the block sums are read
    and not modified, so every block computing the scan redundantly is safe
    as well as necessary. That is the launch this lane removes: four launches
    per partition become three.

    What the scan has to yield here is two numbers, and only two. `mine` is
    the number of left-going rows in every chunk before this block's, which is
    the exclusive prefix of `block_sums` at `block_idx.x`; `grand` is the
    range's total left count. Both are block-uniform, so the loop that
    produces them broadcasts rather than leaving them in one lane. The
    exclusive prefix at the block's own index is recovered as the inclusive
    prefix there minus that index's own block sum, which any thread can read
    straight out of `block_sums` because nothing writes it.

    The cost of doing it here rather than in a launch of its own is that each
    of the `n_blocks` blocks re-reads all `n_blocks` block sums, so the extra
    traffic is quadratic in the block count. That is why the block count is
    capped (`_partition_grid`, `GpuActiveRows.partition_block_cap`): with the
    default cap of one threadgroup width, `n_blocks <= threads` and the worst
    case, a range of exactly `threads * threads` elements, re-reads
    `threads * threads` Int32 against `3 * threads * threads` Int32 of useful
    streamed traffic, so it is bounded by a third of the row work and is far
    below that for any longer range. The block sums are a few kilobytes and
    are read by every block at once, so they are cache-resident; none of this
    has been measured, and the bound is an argument about counts, not a
    timing.

    The scatter itself. A left-going row at local index `j` lands at its
    global left rank `p`; a right-going one lands after every left-going row,
    at `grand + (j - p)`, where `j - p` is its rank among the right-going
    rows. Both ranks are monotone in `j`, so both sides keep their relative
    order and the partition is stable. `p` is `mine` plus the exclusive prefix
    within the chunk that stage one packed into `offsets[j]`, which is exactly
    the number of left-going rows at positions before `j` in the range. Note
    that `p` is therefore a function of the routing flags and of `j` alone: it
    does not depend on how the range was cut into chunks, on how many chunks
    there were, or on how many tiles each walked. That is the whole argument
    that this lane leaves the permutation element for element unchanged.

    The direction comes out of the low bit of the packed offset stage one
    wrote, so this stage touches no bin and needs no routing rule: it is a
    pure permutation of the range from three streamed Int32 reads per row.

    Launched with the same width and the same `tiles` stage one used, because
    `block_idx.x` and `tiles` are what select this element's chunk, and the
    packed offsets are chunk-relative.

    This is the fallback arm; `_scatter_prim_kernel` is the default and
    differs only in scanning with `block.prefix_sum` and `block.broadcast`
    instead of in shared memory by hand.

    `desc` and `use_desc` are the descriptor arm: only the window moves,
    since this stage reads the routing decision out of the packed offsets and
    never touches a bin. A dead step leaves the permutation untouched, which
    is what makes a tree's worth of steps safe to enqueue before growth has
    decided how many of them there will be.
    """
    var tid = thread_idx.x
    var nthreads = block_dim.x
    var nb = Int(n_blocks)
    var me = Int(block_idx.x)
    var n = Int(count)
    var b = Int(begin)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])

    var s = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var carry = Int32(0)
    var mine = Int32(0)
    var base = 0
    while base < nb:
        var idx = base + tid
        var own = Int32(0)
        if idx < nb:
            own = block_sums[unsafe_offset=idx][0]
        s[unsafe_offset=tid] = own
        barrier()

        var offset = 1
        while offset < nthreads:
            var carried = Int32(0)
            if tid >= offset:
                carried = s[unsafe_offset = tid - offset][0]
            barrier()
            if tid >= offset:
                s[unsafe_offset=tid] = s[unsafe_offset=tid][0] + carried
            barrier()
            offset += offset

        # `me` is block-uniform, so this branch is taken by the whole
        # threadgroup or by none of it and no barrier is skipped.
        if me >= base and me < base + nthreads:
            mine = (
                carry
                + s[unsafe_offset = me - base][0]
                - block_sums[unsafe_offset=me][0]
            )
        carry += s[unsafe_offset = nthreads - 1][0]
        # The chunk total has been read by every thread; the next iteration
        # may now overwrite the shared buffer.
        barrier()
        base += nthreads

    if me == 0 and tid == 0:
        total[unsafe_offset=0] = carry

    var chunk = Int(tiles) * nthreads
    var first = me * chunk
    for t in range(Int(tiles)):
        var j = first + t * nthreads + tid
        if j < n:
            var row = rows[unsafe_offset = b + j][0]
            var packed = offsets[unsafe_offset=j][0]
            var p = Int(mine) + (Int(packed) >> 1)
            var dst: Int
            if (packed & Int32(1)) != 0:
                dst = p
            else:
                dst = Int(carry) + (j - p)
            scratch[unsafe_offset = b + dst] = row


def _scatter_prim_kernel[block_size: Int](
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    total: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    tiles: Int32,
    n_blocks: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Stage two on the primitive arm: the same head scan and the same
    scatter, with `block.prefix_sum` and `block.broadcast` in place of the
    hand-rolled shared-memory scan.

    Byte for byte the same writes as `_scatter_kernel` makes, for the same
    reason the two stage-one arms agree: the scan sums exact Int32 over a
    fixed order, so the prefix it produces is a value and not an
    approximation, and every destination index is computed from that prefix
    by the same integer arithmetic. `tests/test_gpu_partition_launches.mojo`
    asserts that against the hand-rolled arm element for element rather than
    leaving it as a comment here.

    Two broadcasts per chunk rather than one. `block_size - 1` gives the
    chunk total, as on stage one; `me - base` gives the inclusive prefix at
    this block's own index, from which its own block sum is subtracted to
    make the exclusive prefix. `src_thread` is a runtime argument of
    `block.broadcast`, and `me - base` is block-uniform inside the guarded
    branch, so the collective is reached by the whole threadgroup with the
    same source lane.

    `desc` and `use_desc` are the descriptor arm, as in `_scatter_kernel`:
    only the window moves, and a dead step returns before any collective, so
    the whole threadgroup leaves together.
    """
    var tid = thread_idx.x
    var nb = Int(n_blocks)
    var me = Int(block_idx.x)
    var n = Int(count)
    var b = Int(begin)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])

    var carry = Int32(0)
    var mine = Int32(0)
    var base = 0
    while base < nb:
        var idx = base + tid
        var own = Int32(0)
        if idx < nb:
            own = block_sums[unsafe_offset=idx][0]

        var inclusive = block_prefix_sum[
            block_size=block_size, exclusive=False
        ](own)
        var chunk_total = block_broadcast[block_size=block_size](
            inclusive, src_thread = block_size - 1
        )
        if me >= base and me < base + block_size:
            var at_me = block_broadcast[block_size=block_size](
                inclusive - own, src_thread = me - base
            )
            mine = carry + at_me
        carry += chunk_total
        barrier()
        base += block_size

    if me == 0 and tid == 0:
        total[unsafe_offset=0] = carry

    var chunk = Int(tiles) * block_size
    var first = me * chunk
    for t in range(Int(tiles)):
        var j = first + t * block_size + tid
        if j < n:
            var row = rows[unsafe_offset = b + j][0]
            var packed = offsets[unsafe_offset=j][0]
            var p = Int(mine) + (Int(packed) >> 1)
            var dst: Int
            if (packed & Int32(1)) != 0:
                dst = p
            else:
                dst = Int(carry) + (j - p)
            scratch[unsafe_offset = b + dst] = row


def _copy_back_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
):
    """Stage three: fold the compacted range back over the parent's slots.

    Only the
    parent's own range is touched, so every other leaf's range survives the
    split untouched; that is why this is a copy and not a buffer swap.

    Launched over its own grid, `ceil(count / threads)` blocks of one tile
    each, and not over the capped grid the flag pass and the scatter share.
    It has no cross-block dependency of any kind, so there is nothing for a
    bounded block count to buy it, and giving it the uncapped grid keeps its
    shape exactly what it was before this lane.

    The loop is grid-strided rather than a single guarded store. On the
    host-argument arm that is the same thing: the grid is `ceil(count /
    threads)` blocks, so the stride is at least `count` and every thread runs
    exactly one iteration, which is the store this kernel has always made.
    What it buys is the descriptor arm, which cannot size a grid to a count
    it does not know and would otherwise have to launch
    `ceil(n_active / threads)` blocks to cover the largest window a step
    might pick. With a stride it can launch the capped grid the other two
    stages use and let each thread walk. No element is visited twice either
    way, because the destinations are distinct positions of one range.

    `desc` and `use_desc` are the descriptor arm; see `_flag_scan_kernel`.

    Why this launch is still here
    -----------------------------
    Removing it is a real saving and was scoped for this lane: it is one of
    three launches, and it moves 8 bytes per row of the range (a 4-byte read
    from `scratch` and a 4-byte write to `rows`) that exist only to undo the
    fact that the scatter cannot write in place. The design that removes it
    is per-range ping-pong: `scratch` becomes a co-equal second row buffer,
    each `LeafRange` carries a bit saying which of the two currently holds
    its rows, the scatter reads from the parent's live buffer and writes to
    the other one, and the two children are recorded as living in the buffer
    the scatter just wrote. No copy, and the ranges other leaves hold stay
    valid because nothing outside the parent's window is touched either way.

    It is not done here because the bit has to be consulted by *every* reader
    of the permutation, and the readers are not all in this file. The audit,
    at the commit that added this note:

    - `enqueue_range_histogram` in this module, which hands the row pointer
      to `_range_hist_atomic_kernel` and `_range_hist_partial_kernel`. It
      knows the node, so it could pick the pointer on the host with no kernel
      change at all, which is the easy half.
    - `GpuHistogramBuilder.readback_range` (histogram_gpu.mojo), which builds
      a sub-buffer view of `rows_dev[begin : begin + count]` and hands it to a
      host histogram build. Same fix, different file.
    - `GpuHistogramBuilder.snapshot_rows` (histogram_gpu.mojo), which copies
      the *whole* permutation in one transfer. This one is not a pointer
      swap: under per-range ping-pong there is no single buffer that holds
      the whole permutation, so a snapshot becomes a gather across both
      buffers driven by the range table, or a merge pass, and the hybrid
      replica path that consumes it is exactly the path whose bit-for-bit
      agreement with the device is load-bearing.
    - `download_rows` and `download_range` in this module, which have the
      same problem as `snapshot_rows` for the same reason.
    - the two launches in `gpu_objectives_native.mojo` that take
      `rows.rows_dev.unsafe_ptr()` over the whole active prefix, which have
      it again.
    - `_quantize_grad_hess_kernel` here, which reads no permutation at all
      (it is indexed by row id) and is therefore *not* affected; recorded so
      that a later reader does not have to re-derive that.

    Three of those live in files this lane may not edit and one of them,
    `snapshot_rows`, is not a one-line change anywhere. A half-applied
    ping-pong leaves one reader looking at the stale buffer for one range,
    which produces a wrong histogram for one node on some tree shapes and
    not others, which is precisely the kind of fault no fixture catches. So
    the copy stays, and the design is written down here rather than
    half-built.
    """
    var n = Int(count)
    var b = Int(begin)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        b = Int(desc[unsafe_offset=STEP_ROW_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_ROW_COUNT][0])
    var stride = Int(block_dim.x) * Int(grid_dim.x)
    var j = Int(global_idx.x)
    while j < n:
        var i = b + j
        rows[unsafe_offset=i] = scratch[unsafe_offset=i][0]
        j += stride


def _quantize_grad_hess_kernel(
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """Quantize this round's gradients once, into one interleaved buffer.

    `gq[2r]` and `gq[2r + 1]` are exactly the `Int32(round(grad[r] *
    g_scale))` and `Int32(round(hess[r] * h_scale))` a histogram kernel would
    otherwise compute per (row, feature) visit: same Float32 product, same
    rounding, so a histogram accumulated from this buffer is bit-identical to
    one accumulated from the Float32 planes. That is an argument from the
    expression, not a measurement: the two sites evaluate the same Float32
    multiply and the same `round`, and hoisting an expression out of a loop
    whose other operands it does not depend on cannot change its value.

    What changes is the traffic. The histogram kernels gather `grad[r]` and
    `hess[r]` by row id, so at a leaf whose rows are sparse in the dataset
    each is a separate cache line per row per threadgroup, and a 50-feature
    histogram at the pairing gathers both 25 times per row. Interleaving the
    two quantized words puts them in one line, halving those gathers, and
    moves the two multiplies and rounds off the inner loop entirely. One
    streaming pass per tree over `n_rows` (16 bytes each), against
    `n_features` gathered passes per node.

    Because the result is provably identical and the work is strictly less,
    this is the default source (`MOJOTREES_GPU_QUANTIZED_GRADS`, on unless set
    to 0). The Float32 arms of the kernels below are kept so a benchmark can
    hold both, not because either is more correct than the other.
    """
    var r = global_idx.x
    if r < Int(n_rows):
        gq[unsafe_offset = 2 * r] = Int32(
            round(grad[unsafe_offset=r][0] * g_scale)
        )
        gq[unsafe_offset = 2 * r + 1] = Int32(
            round(hess[unsafe_offset=r][0] * h_scale)
        )


# --- The row walk, shared by both accumulation strategies ----------------
#
# Rows one thread keeps in flight at once inside the histogram row loop.
#
# What this constant is for. The row loop's per-row work is a chain of
# dependent loads: read the row index out of the permutation, use it to
# gather the quantized gradient pair, use it again to gather this slot's bin
# byte, and only then issue the shared atomics. Written one row at a time,
# each of those loads has to complete before the next address can be formed,
# so a thread has at most one memory request outstanding at any moment and
# the loop runs at the latency of the gather rather than at the bandwidth of
# it. Walking `HIST_ROW_UNROLL` rows per iteration, with every load of a
# stage issued before any load of the next stage is consumed, puts that many
# requests in flight per thread instead of one.
#
# Why 4 and not 2 or 8. This is a guess and it should be read as one. Four
# rows costs four Int registers for the row indices plus three Int32 arrays
# of four, which is small against what a threadgroup already holds, and four
# is the smallest depth that covers a three-stage chain with one to spare.
# Nothing on this machine measured the right value, and nothing could
# without a device counter this backend does not expose. Changing it is a
# one-line edit and a rebuild, and the result is bit-identical at every
# value (see `_hist_rows_step`), so an A/B costs two builds and no
# correctness argument.
#
# `GpuActiveRows.set_row_unroll(False)` puts the loop back to one row per
# iteration at run time without a second instantiation, which is the arm a
# benchmark holds against this one; see `_hist_accumulate_rows`.
comptime HIST_ROW_UNROLL = 4


# --- The blocked row-major stripe, and why it is not here ----------------
#
# `docs/design/CLEANSHEET_GPU.md` section 3.3 proposes replacing the
# feature-major bin matrix with stripes of `W` features stored row-major,
# `W = floor(threadgroup_memory / (n_bins * cell_bytes))`, so that one
# gathered read of a padded `W`-byte record serves `W` features instead of
# one. Its traffic model puts a whole tree at 752 MB against today's 2650 MB
# at a 64-byte cache line, and 1136 MB at 128. That is the largest single
# lever anyone has costed on this backend and it is not built here. The
# reasons are worth writing down, because the next lane to reach for it
# should reach for the measurement first and not for the layout.
#
# **It is not a change to this file.** The row loop above reads `bins` as
# `feature * n_rows + row`. Every other reader of that buffer would have to
# change with it or a second copy would have to be maintained, and the
# readers are not in one place: `_partition_kernel` and `_flag_scan_kernel`
# in this module, the whole of `gpu_binned_layout.mojo`, the leaf-id scan in
# `histogram_gpu.mojo`, `gpu_split_search.mojo`, `gpu_sparse.mojo`,
# `gpu_categorical.mojo`, `gpu_leaf_batching.mojo`, `gpu_multiclass_batch.mojo`,
# and the CPU builder in `histogram.mojo` that the hybrid replica has to stay
# bit-identical to. A half-applied stripe layout, where some kernels read the
# striped copy and others read the column-major one, is a much worse state
# than none: the two agree until a lane forgets to convert one of them, and
# then one node's histogram is wrong on some shapes and not others.
#
# **The arithmetic that motivates it turns on a number nobody here has
# measured.** The whole 3.5x-versus-2.3x spread in that section is the cache
# line size, and section 8 of the same document lists it as unknown. There is
# no counter on an Apple M4 that reports it and no documented figure this
# project trusts. So the honest order of work is: measure the line first,
# then decide whether the layout is worth its blast radius.
#
# **The measurement, which is small and is not this lane's file.** A
# standalone kernel that gathers one 4-byte word per thread from a buffer far
# larger than any cache, at a stride `S` swept over 4, 8, 16, 32, 64, 128,
# and 256 bytes, and reports achieved bytes per second. Useful traffic is
# constant across the sweep and real traffic is `max(S, L)` per gather, so
# the achieved rate falls in proportion to `S` above the line size and is
# flat at or below it. The knee is `L`. Two further points are worth taking
# in the same harness, because they are the assumptions the stripe rests on
# and not merely the line size: whether a 16-byte gathered read costs the
# same as a 4-byte one at the same stride, which is the specific claim
# section 3.3 flags as its least certain, and whether padding a record to a
# power of two changes the answer. That harness belongs in `bench/`, which
# this lane does not own, and it should be written before a single byte of a
# second bin layout is.
#
# Nothing above is an argument that the stripe is wrong. It is an argument
# that it is a project and not a patch, and that a cheap experiment stands
# between here and knowing which of the two numbers in the model applies.


# The three shared accumulation planes, as `stack_allocation` hands them
# back. Named here because the row walk takes them as arguments rather than
# allocating them, and a threadgroup pointer's type has to be written out to
# cross a function boundary.
comptime _SharedI32 = Pointer[
    Int32, MutUntrackedOrigin, address_space = AddressSpace.SHARED
]

# Per-thread scratch: the column bases, the rows in flight, and the values
# gathered for them. Every index into these arrays is a compile-time
# constant, so they exist to be promoted into registers and never to be
# addressed.
comptime _LocalInt = Pointer[
    Int, MutUntrackedOrigin, address_space = AddressSpace.GENERIC
]
comptime _LocalI32 = Pointer[
    Int32, MutUntrackedOrigin, address_space = AddressSpace.GENERIC
]


@always_inline
def _hist_rows_step[
    GROUP: Int, U: Int, CELIDE: Bool, QUANT: Bool
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    sg: _SharedI32,
    sh: _SharedI32,
    sc: _SharedI32,
    col: _LocalInt,
    rr: _LocalInt,
    gv: _LocalI32,
    hv: _LocalI32,
    bv: _LocalI32,
    owned: Int,
    nb: Int,
    row_begin: Int,
    j: Int,
    stride: Int,
    g_scale: Float32,
    h_scale: Float32,
):
    """Accumulate `U` rows of one thread's stride into the shared planes.

    The rows are `j, j + stride, ..., j + (U - 1) * stride` of the node's
    range, and the caller guarantees every one of them is inside the tile.
    Written in four stages, each of which issues all `U` of its loads before
    the next stage consumes any of them:

    1. `U` row indices out of the permutation.
    2. `U` quantized gradient pairs, gathered by those indices.
    3. per owned slot, `U` bin bytes out of that slot's column.
    4. per owned slot, `U` shared atomic triples.

    Stages 2 and 3 both depend on stage 1 and stage 4 depends on stage 3, so
    the chain is three deep however large `U` is; what `U` buys is `U`
    independent requests at each level of it rather than one.

    **Why this cannot change a histogram.** Every instantiation performs the
    same set of adds. For each (row, feature) in the tile it adds the same
    three quantized values -- the row's gradient, its hessian, and one -- to
    the same three shared cells, chosen by the same bin byte read from the
    same address. What `U` changes is only the order in which those adds are
    issued and the order in which the loads that feed them are issued.
    Accumulation is Int32 throughout and integer addition is associative and
    commutative, so a reordering of the adds cannot move a bit. It is the
    identical argument `_range_hist_atomic_kernel` already makes for `GROUP`
    and `BIN_CAP`, applied to a third launch-shape parameter, and it is
    structural rather than a tolerance.

    The one thing that would break it is arithmetic that is not integer, and
    there is exactly one such site: the Float32 arm's
    `Int32(round(x * scale))`. That expression is reproduced here character
    for character from the loop it replaces, on purpose. `docs/NUMERICS.md`
    section 5.6 records why it is structurally immune to contraction (the
    multiply is consumed by `round`, so there is no add for it to fuse
    into), and that argument survives unrolling only because the expression
    survives unrolling. A future edit that hoists the multiply, or that
    lets an add reach it, breaks a property this file depends on. The
    quantized arm, which is the default, has no floating-point arithmetic in
    it at all.

    **The gradient pair is one load.** `_quantize_grad_hess_kernel` writes
    the two quantized words interleaved, so `gq[2r]` and `gq[2r + 1]` are
    adjacent and, since the pair starts at byte `8r` of a device allocation,
    8-byte aligned. A width-2 load therefore reads both in one instruction
    where the loop it replaces issued two. It reads the identical eight bytes
    and assigns them to the identical two variables, so it cannot change a
    value; it is one fewer load and one fewer address computation per row.
    The elided-hessian arm reads only the gradient word and keeps the scalar
    load, because the second word is not wanted there.

    **What stays 64-bit, and why.** The shared-plane index `s` is computed in
    Int32 because it is bounded by `GROUP * BIN_CAP`, at most 16 * 256 =
    4096, by construction rather than by data. The other three index
    expressions in this loop are deliberately left in Int, and a later reader
    should not narrow them:

    - `col[k] + r` is `feature * n_rows + row`, which reaches
      `n_features * n_rows` and overflows Int32 on any dataset past about
      2.1 billion cells.
    - `2 * r` reaches `2 * n_rows`, and `MAX_ROWS` in this module is
      `Int32.MAX`, so it overflows for any fit past 2^30 rows.
    - `row_begin + j + u * stride` is bounded by `n_rows`, so it would fit,
      but it is formed once per row against three formed per (row, feature)
      and is not worth a special case.

    Whether narrowing `s` removes any instruction at all depends on whether
    the Metal compiler was already narrowing it, which nothing on this
    machine can show. It is written narrow because the bound is a
    compile-time constant and writing it wide asserted something the code
    did not need.
    """
    # Stage one: the row indices.
    comptime for u in range(U):
        rr[unsafe_offset=u] = Int(
            rows[unsafe_offset = row_begin + j + u * stride][0]
        )

    # Stage two: the quantized gradient (and hessian) for each of them.
    comptime if QUANT:
        comptime for u in range(U):
            comptime if CELIDE:
                gv[unsafe_offset=u] = gq[
                    unsafe_offset = 2 * rr[unsafe_offset=u]
                ][0]
            else:
                var pair = gq.unsafe_load[width=2](2 * rr[unsafe_offset=u])
                gv[unsafe_offset=u] = pair[0]
                hv[unsafe_offset=u] = pair[1]
    else:
        comptime for u in range(U):
            gv[unsafe_offset=u] = Int32(
                round(
                    grad[unsafe_offset = rr[unsafe_offset=u]][0] * g_scale
                )
            )
            comptime if not CELIDE:
                hv[unsafe_offset=u] = Int32(
                    round(
                        hess[unsafe_offset = rr[unsafe_offset=u]][0] * h_scale
                    )
                )

    # Stages three and four, one owned slot at a time.
    comptime for k in range(GROUP):
        if k < owned:
            var base = col[unsafe_offset=k]
            var lift = Int32(k * nb)
            comptime for u in range(U):
                bv[unsafe_offset=u] = Int32(
                    Int(bins[unsafe_offset = base + rr[unsafe_offset=u]])
                )
            comptime for u in range(U):
                var s = Int(lift + bv[unsafe_offset=u][0])
                _ = Atomic.fetch_add(
                    sg.unsafe_offset(s), gv[unsafe_offset=u][0]
                )
                comptime if not CELIDE:
                    _ = Atomic.fetch_add(
                        sh.unsafe_offset(s), hv[unsafe_offset=u][0]
                    )
                _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))


@always_inline
def _hist_accumulate_rows[
    GROUP: Int, UNROLL: Int, CELIDE: Bool, QUANT: Bool
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    sg: _SharedI32,
    sh: _SharedI32,
    sc: _SharedI32,
    slot0: Int,
    owned: Int,
    nb: Int,
    nr: Int,
    row_begin: Int,
    tile_begin: Int,
    tile_end: Int,
    g_scale: Float32,
    h_scale: Float32,
    unrolled: Bool,
):
    """One thread's whole walk over its share of one tile.

    This is the row loop both `_range_hist_atomic_kernel` and
    `_range_hist_partial_kernel` run, factored out so there is one copy of it
    rather than eight. The two kernels differ in what they do with the shared
    planes afterwards, not in how they fill them, and the four arms each of
    them used to write out longhand (`const_hess` by `use_quant`) are the two
    comptime flags here. Selecting them at compile time rather than
    branching per row is what the two longhand loops bought and it is kept:
    the caller picks one of four instantiations from two block-uniform
    runtime flags, once, outside the loop.

    **The walk.** Thread `tid` owns rows `tile_begin + tid`,
    `+ block_dim.x`, `+ 2 * block_dim.x`, ... of the range, up to
    `tile_end`. That set is unchanged from the one-row-per-iteration loop
    this replaces. The main loop takes `UNROLL` of them per iteration while
    a full group remains, and the tail loop takes the rest one at a time, so
    the union of the two is exactly the arithmetic progression above with no
    row visited twice and none dropped. The tail is the same
    `_hist_rows_step` at `U = 1`, so the two cannot drift apart under a later
    edit.

    **`unrolled`.** Setting it false collapses the main loop's bound to the
    thread's first row, so the loop body never runs and the tail loop does
    the whole walk one row at a time. That is precisely the shape this
    module shipped before this lane, reachable at run time and therefore
    interleavable inside one process, which is the only protocol that
    compares anything on a machine whose device timings drift several-fold
    between time windows. It costs one block-uniform branch outside the loop
    and no second kernel instantiation, which matters: this family is already
    forty instantiations and the const-hessian flag was left a runtime flag
    for exactly that reason.

    Both arms produce the identical histogram, for the reason `_hist_rows_step`
    argues in full, so this is a launch-shape knob and never a numeric one.

    **What is claimed, and what is not.** Claimed, by counting: the main loop
    executes one loop test and one induction add per `UNROLL` rows instead of
    per row, and the quantized non-elided arm issues one 8-byte load per row
    where the loop it replaces issued two 4-byte loads. Not claimed: any
    speedup. The unroll raises the number of live registers in the row loop
    by roughly `UNROLL` times the values a single row needs, and threadgroup
    residency on this backend is bounded by things this project cannot query.
    If the unroll loses, that is how it loses, and `set_row_unroll(False)` is
    the arm that shows it.
    """
    var tid = Int(thread_idx.x)
    var stride = Int(block_dim.x)

    # One column base per owned slot, read once and spent for every row.
    # The unowned tail entries are zeroed rather than left undefined so that
    # a comptime-unrolled body which computes an address it never uses --
    # which is what `if k < owned` leaves the compiler free to do -- cannot
    # form one out of stack garbage.
    var col = stack_allocation[GROUP, Int]()
    comptime for k in range(GROUP):
        col[unsafe_offset=k] = 0
        if k < owned:
            col[unsafe_offset=k] = (
                Int(feat_ids[unsafe_offset = slot0 + k][0]) * nr
            )

    # The rows in flight and what was gathered for them. Allocated once,
    # outside both loops, so there is one alloca per thread and not one per
    # iteration.
    var rr = stack_allocation[UNROLL, Int]()
    var gv = stack_allocation[UNROLL, Scalar[DType.int32]]()
    var hv = stack_allocation[UNROLL, Scalar[DType.int32]]()
    var bv = stack_allocation[UNROLL, Scalar[DType.int32]]()

    var j = tile_begin + tid
    # `limit` is where the last full group of `UNROLL` rows can start. Left
    # at `j` when the unroll is off or unavailable, which skips the main
    # loop entirely without a second code path.
    var limit = j
    comptime if UNROLL > 1:
        if unrolled:
            limit = tile_end - (UNROLL - 1) * stride
    while j < limit:
        _hist_rows_step[GROUP, UNROLL, CELIDE, QUANT](
            bins,
            rows,
            grad,
            hess,
            gq,
            sg,
            sh,
            sc,
            col,
            rr,
            gv,
            hv,
            bv,
            owned,
            nb,
            row_begin,
            j,
            stride,
            g_scale,
            h_scale,
        )
        j += UNROLL * stride
    while j < tile_end:
        _hist_rows_step[GROUP, 1, CELIDE, QUANT](
            bins,
            rows,
            grad,
            hess,
            gq,
            sg,
            sh,
            sc,
            col,
            rr,
            gv,
            hv,
            bv,
            owned,
            nb,
            row_begin,
            j,
            stride,
            g_scale,
            h_scale,
        )
        j += stride


@always_inline
def _hist_accumulate_dispatch[
    GROUP: Int, UNROLL: Int
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    sg: _SharedI32,
    sh: _SharedI32,
    sc: _SharedI32,
    slot0: Int,
    owned: Int,
    nb: Int,
    nr: Int,
    row_begin: Int,
    tile_begin: Int,
    tile_end: Int,
    g_scale: Float32,
    h_scale: Float32,
    celide: Bool,
    quant: Bool,
    unrolled: Bool,
):
    """Resolve the two runtime flags into one of four comptime arms, once,
    above the row loop.

    Both flags are launch arguments and therefore block-uniform, so this is a
    scalar branch taken once per threadgroup and not a divergence. It exists
    so that the four arms are written once here instead of once in each of
    the two kernels."""
    if celide:
        if quant:
            _hist_accumulate_rows[GROUP, UNROLL, True, True](
                bins,
                rows,
                grad,
                hess,
                gq,
                feat_ids,
                sg,
                sh,
                sc,
                slot0,
                owned,
                nb,
                nr,
                row_begin,
                tile_begin,
                tile_end,
                g_scale,
                h_scale,
                unrolled,
            )
        else:
            _hist_accumulate_rows[GROUP, UNROLL, True, False](
                bins,
                rows,
                grad,
                hess,
                gq,
                feat_ids,
                sg,
                sh,
                sc,
                slot0,
                owned,
                nb,
                nr,
                row_begin,
                tile_begin,
                tile_end,
                g_scale,
                h_scale,
                unrolled,
            )
    elif quant:
        _hist_accumulate_rows[GROUP, UNROLL, False, True](
            bins,
            rows,
            grad,
            hess,
            gq,
            feat_ids,
            sg,
            sh,
            sc,
            slot0,
            owned,
            nb,
            nr,
            row_begin,
            tile_begin,
            tile_end,
            g_scale,
            h_scale,
            unrolled,
        )
    else:
        _hist_accumulate_rows[GROUP, UNROLL, False, False](
            bins,
            rows,
            grad,
            hess,
            gq,
            feat_ids,
            sg,
            sh,
            sc,
            slot0,
            owned,
            nb,
            nr,
            row_begin,
            tile_begin,
            tile_end,
            g_scale,
            h_scale,
            unrolled,
        )


def _range_hist_atomic_kernel[
    GROUP: Int, BIN_CAP: Int
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    rows_per_tile: Int32,
    begin: Int32,
    count: Int32,
    g_scale: Float32,
    h_scale: Float32,
    sub_offset: Int32,
    do_sub: Int32,
    use_quant: Int32,
    const_hess: Int32,
    desc: MutPointer[Int32, MutAnyOrigin],
    use_desc: Int32,
    row_unroll: Int32,
):
    """One node's histogram over a compacted row range, accumulated in
    threadgroup memory and folded into the output with global atomics.

    One kernel for the whole family. The hand-written one-feature and
    two-feature atomic kernels this replaces differed in exactly two things,
    how many feature slots a block owned and how wide its shared planes were,
    and both are now comptime parameters: `GROUP` walks the ladder 1, 2, 4, 8,
    16 and `BIN_CAP` walks 32, 64, 128, 256. `enqueue_range_histogram` picks
    the pair, taking `BIN_CAP` from `gpu_tiling.histogram_bin_capacity` and
    the group from the free-footprint rule in
    `gpu_tiling.free_feature_group`.

    **What the capacity parameter fixes.** The old kernels allocated three
    `MAX_BINS`-wide Int32 planes whatever the dataset's bin count was, so a
    64-bin dataset paid four times the threadgroup memory it needed and a
    32-bin one eight times. The allocation here is `GROUP * BIN_CAP` per
    plane, so it is the memory the histogram occupies and nothing more.
    Threadgroup memory is what bounds resident blocks per core, so this is a
    residency change and not a bandwidth one, which is why the effect it has
    on wall clock is a per-device measurement rather than an argument. It is
    UNMEASURED at every capacity below 256 on every device this project runs
    on. What would measure it is an interleaved A/B in one process of one
    shape binned at 64 against the same shape binned at 256 with the same row
    count, which is not the comparison `bench_histogram.mojo` makes today.

    **What the group parameter buys.** With one feature per block,
    `rows[begin + j]` and the row's quantized gradient pair are read once per
    feature. A block owning `GROUP` slots reads them once and spends them
    `GROUP` times, which is the `pair_features` walk `histogram.mojo` already
    does on the CPU, lifted to a threadgroup and widened. Measured for the
    tiled twin below on an Apple M4 at 1.17x for group 2 over group 1 on the
    atomic path (`pixi run bench-hist 100000 100 20`, arms interleaved in one
    process because this machine's device timings drift several-fold between
    time windows). Groups 8 and 16 have never been launched on any device, so
    nothing is claimed for them.

    **Exactness.** Every instantiation produces the identical integer
    histogram, and this is structural rather than a tolerance. Accumulation is
    fixed-point Int32 throughout, integer addition is associative and
    commutative, and every atomic adds the same per-(row, feature) quantized
    value into the same bin whatever the block shape, so only the order of the
    adds differs between instantiations and the order of integer adds cannot
    change a sum. `GROUP` and `BIN_CAP` change where a partial lives, never
    what goes into it.

    **Tail blocks.** A block whose `slot0 + GROUP` overruns `n_slots` owns
    only the slots that remain. It zeroes only its own span, accumulates only
    into its own slices, and flushes only those, so a feature is never counted
    twice and never dropped. The owned slices are stacked at `k * n_bins`
    inside the allocation, so the stacking is bounded by
    `GROUP * n_bins <= GROUP * BIN_CAP` and the zeroing walk covers exactly
    the cells the flush will read.

    **The row loop.** It lives in `_hist_accumulate_rows`, shared with
    `_range_hist_partial_kernel`, because the two kernels differ in what they
    do with the shared planes after the barrier and not at all in how they
    fill them. `use_quant` selects between the pre-quantized interleaved
    buffer `_quantize_grad_hess_kernel` writes and the two Float32 planes,
    and it does so by picking one of four comptime arms above the loop rather
    than by branching per row, so the default path spends no floating-point
    arithmetic in the row loop at all. It is a runtime flag rather than a
    third comptime parameter because doubling forty instantiations to eighty
    is compile time every build on every backend pays. The two produce
    bit-identical histograms by construction; see
    `_quantize_grad_hess_kernel`.

    `row_unroll` is a second such flag, selecting how many rows one thread
    keeps in flight; both settings visit the same rows and add the same
    integers in a different order, which cannot change a sum. See
    `_hist_accumulate_rows` and `_hist_rows_step` for the argument in full.

    **Fused subtraction.** With `do_sub`, the flush also subtracts what it
    added from the slot `sub_offset` words away, which is the sibling
    subtraction folded into the build. The destination is a signed offset
    rather than a second pointer because both slots live in one pool buffer,
    and two pointers into one buffer is an aliasing the compiler is right to
    refuse. It is the same arithmetic the standalone
    `gpu_leaf_batching._subtract_slice_kernel` performs and exact for the same
    reason, both slots being fixed-point Int32
    under one scale, so a parent's bins are the exact integer sum of its
    children's. Bins this node never touched are left alone, which is what
    subtracting zero from them would have done anyway.

    **The elided hessian plane.** With `const_hess` the caller has declared
    that every row's hessian is exactly `histogram.CONSTANT_HESSIAN`, which
    four of the built-in objectives guarantee when the fit carries no sample
    weights. The row loop then performs two shared-memory atomics per (row,
    feature) instead of three, the shared zeroing covers two planes instead
    of three, and the flush writes the global hessian plane as
    `hq_const * vc` instead of reading a third shared plane.

    That reconstruction is the exact integer the three-plane path would have
    accumulated, and the argument is worth spelling out because everything
    rests on it. Every row contributes
    `Int32(round(hess[r] * h_scale))` to its bin. With `hess[r]` equal to
    1.0f for every row, and `1.0f * x` exactly `x` for every finite Float32
    because multiplying by one cannot round, every one of those contributions
    is the single value `Int32(round(h_scale))`, which is what `hq_const`
    below computes -- in this kernel, on this device, from the same launch
    argument, so no host-versus-device rounding claim is made or needed. A
    bin that received `vc` rows therefore holds `vc` copies of `hq_const`
    added together, and `hq_const * vc` is that sum. Int32 addition and
    multiplication agree modulo 2^32, so the two are the same bits even in
    the overflow regime neither of them ever reaches: under
    `SCALE_MAGNITUDE_SUM` the hessian scale is `2^30 / sum|h|`, so the whole
    dataset's hessian plane sums to about 2^30 and one bin is a subset of
    that.

    `const_hess` is a runtime flag and not a third comptime parameter, for
    the reason the two row loops give below: forty instantiations are already
    what every build on every backend pays for, and eighty is a compile-time
    cost this specialization has not earned. The consequence is honest and
    worth stating rather than eliding: `sh` below is a comptime
    `stack_allocation` and is still allocated on the elided path, so the
    threadgroup footprint does not shrink and **no occupancy improvement
    follows from this flag**. What follows is one third fewer shared atomics
    in the row loop and one third less shared zeroing. Making the plane count
    comptime, and getting the residency with it, is a separate change behind
    a measurement of the compile-time trade.

    The global flush still writes three planes, because the device-resident
    split search in `gpu_split_search.mojo` scans this buffer in place and
    reads all three. Eliding the plane from the device *layout* would be a
    change to that module's contract and is not this lane's; the download in
    `histogram_gpu.histogram_from_host` therefore still moves three planes as
    well.

    The descriptor arm
    ------------------
    With `use_desc`, four values move from launch arguments into device
    memory: the row window this build reads (`STEP_BUILT_BEGIN`,
    `STEP_BUILT_COUNT`), the resident pool slot it writes
    (`STEP_BUILT_SLOT`), and the slot it subtracts itself from
    (`STEP_SUB_SLOT`). `out_hist` is then the *base* of the pool rather than a
    pointer the host has already offset, and `sub_offset` is derived here as
    `(sub_slot - built_slot) * 3 * hist_size` instead of being passed in.
    Which of the two children was chosen to be built and which derived was
    decided by the commit kernel and is already baked into those four words.

    Nothing about the accumulation moves. The same rows are gathered, the
    same fixed-point products are formed, the same shared atomics run and the
    same global atomics flush. The only floating-point operations in this
    kernel are the two per-row quantizing products, which are untouched, so a
    histogram built on this arm is the one the host-argument arm would have
    built, bin for bin. That is a structural statement and not a tolerance.

    The empty-tile early return below is what makes an over-provisioned grid
    affordable. A descriptor-driven launch does not know the row count, so it
    is launched with `grid.y` sized for the whole active prefix; a block whose
    tile begins past the end of the range would otherwise zero its shared
    planes, gather nothing, and then walk every bin of the flush finding
    nothing to add. `block_idx.y` and the count are block-uniform, so the
    whole threadgroup leaves together, and the result is unchanged on both
    arms because such a block contributes nothing either way.
    """
    var slot0 = GROUP * Int(block_idx.x)
    var owned = Int(n_slots) - slot0
    if owned > GROUP:
        owned = GROUP
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)
    var n = Int(count)
    var row_begin = Int(begin)
    # Where this build's own slot starts inside `out_hist`, in Int32 words.
    # Zero on the host-argument arm, because there the host offset the
    # pointer before the launch.
    var out_base = 0
    var sub_off = Int(sub_offset)
    if Int(use_desc) != 0:
        if desc[unsafe_offset=STEP_LIVE][0] == Int32(0):
            return
        row_begin = Int(desc[unsafe_offset=STEP_BUILT_BEGIN][0])
        n = Int(desc[unsafe_offset=STEP_BUILT_COUNT][0])
        var cells = 3 * hs
        var built_slot = Int(desc[unsafe_offset=STEP_BUILT_SLOT][0])
        var from_slot = Int(desc[unsafe_offset=STEP_SUB_SLOT][0])
        out_base = built_slot * cells
        sub_off = (from_slot - built_slot) * cells

    # Empty tiles cost a launch slot and nothing else. See the docstring.
    var tile_begin = Int(block_idx.y) * Int(rows_per_tile)
    if tile_begin >= n:
        return
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    var sg = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sc = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    # The elided plane's one quantized value, and the flag that selects it.
    # `Float32(1.0) * h_scale` is exactly `h_scale`, so this is
    # `Int32(round(h_scale))`, computed here rather than passed in so that it
    # is the same expression evaluated by the same device compiler as the
    # per-row quantization it stands in for.
    var celide = Int(const_hess) != 0
    var hq_const = Int32(round(Float32(1.0) * h_scale))

    var span = owned * nb
    var b = tid
    while b < span:
        sg[unsafe_offset=b] = 0
        if not celide:
            sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    # One feature id per owned slot, read once and used by the flush. The
    # loop is unrolled so the array is indexed by a compile-time constant and
    # stays in registers rather than spilling to local memory. The column
    # bases the row loop needs are read the same way inside
    # `_hist_accumulate_rows`, which is the only place they are wanted.
    var fid = stack_allocation[GROUP, Int]()
    comptime for k in range(GROUP):
        fid[unsafe_offset=k] = 0
        if k < owned:
            fid[unsafe_offset=k] = Int(feat_ids[unsafe_offset = slot0 + k][0])

    # `celide` takes two atomics per (row, feature) instead of three. The
    # hessian is neither read nor accumulated: it is the same Int32 for every
    # row, so the count plane already carries everything the hessian plane
    # would have.
    _hist_accumulate_dispatch[GROUP, HIST_ROW_UNROLL](
        bins,
        rows,
        grad,
        hess,
        gq,
        feat_ids,
        sg,
        sh,
        sc,
        slot0,
        owned,
        nb,
        nr,
        row_begin,
        tile_begin,
        tile_end,
        g_scale,
        h_scale,
        celide,
        Int(use_quant) != 0,
        Int(row_unroll) != 0,
    )
    barrier()

    var sub = Int(do_sub) != 0
    comptime for k in range(GROUP):
        if k < owned:
            var base = fid[unsafe_offset=k] * nb
            var lift = k * nb
            var c = tid
            while c < nb:
                var s = lift + c
                if sc[unsafe_offset=s][0] != 0:
                    var vg = sg[unsafe_offset=s][0]
                    var vc = sc[unsafe_offset=s][0]
                    # The refill. `hq_const * vc` is the sum of `vc` copies of
                    # `hq_const`, which is precisely what the three-plane
                    # accumulation would have left in `sh`.
                    var vh = (
                        hq_const * vc if celide else sh[unsafe_offset=s][0]
                    )
                    var go = out_base + base + c
                    _ = Atomic.fetch_add(out_hist.unsafe_offset(go), vg)
                    _ = Atomic.fetch_add(
                        out_hist.unsafe_offset(hs + go), vh
                    )
                    _ = Atomic.fetch_add(
                        out_hist.unsafe_offset(2 * hs + go), vc
                    )
                    if sub:
                        var so = out_base + sub_off + base + c
                        _ = Atomic.fetch_add(out_hist.unsafe_offset(so), -vg)
                        _ = Atomic.fetch_add(
                            out_hist.unsafe_offset(hs + so), -vh
                        )
                        _ = Atomic.fetch_add(
                            out_hist.unsafe_offset(2 * hs + so), -vc
                        )
                c += block_dim.x


def _range_hist_partial_kernel[
    GROUP: Int, BIN_CAP: Int
](
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    gq: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    partials: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_slots: Int32,
    n_bins: Int32,
    rows_per_tile: Int32,
    begin: Int32,
    count: Int32,
    g_scale: Float32,
    h_scale: Float32,
    use_quant: Int32,
    const_hess: Int32,
    row_unroll: Int32,
):
    """The tiled twin of `_range_hist_atomic_kernel`: the same threadgroup
    accumulation, written to a per-(tile, slot) partial slot instead of folded
    into the output with global atomics.

    Same two comptime parameters and the same rules for them, so everything
    the atomic kernel's docstring argues about capacity, group width,
    exactness, tail blocks, and the row loop holds here unchanged; the row
    loop is not merely equivalent but literally the same code, since both
    kernels call `_hist_accumulate_dispatch`. `row_unroll` reaches it from
    here for the same reason and with the same guarantee. What
    is particular to this kernel is the partial layout, and it is untouched:
    a block owning several slots writes the same per-slot
    `[grad | hess | count]` slices that as many one-slot blocks would have
    written, each slice still written by exactly one block and by no atomic at
    all, so `_range_reduce_kernel` and the fused sibling subtraction it
    carries combine these partials unchanged whatever `GROUP` and `BIN_CAP`
    are.

    This is the kernel a large fit actually runs: `resolve_tiling` picks the
    tiled strategy whenever occupancy wants more than one tile per feature,
    and at 5M x 50 the histogram build is measured at about 79% of GPU
    training wall clock. Two measurements from the hand-written variants this
    replaces carry over, both on an Apple M4 with three interleaved repeats
    per arm in one process, same binned data, predictions byte-identical: a
    full 100-round `train_gpu` at 5M x 50 ran 15.96s at group 1 against 11.49s
    at group 2 (1.39x, the group-2 arm's own spread 0.03%), and 12.09s at
    group 2 against 11.70s at group 4 (1.034x, both spreads under 1%). Both
    were taken with three `MAX_BINS`-wide planes allocated at every bin count,
    which is what made group 4 nearly a wash: the traffic halving was given
    back by the occupancy halving. Whether sizing the planes to the real bin
    count changes that verdict is exactly the thing this parameterization
    makes measurable and it has NOT been measured. The measurement is an
    interleaved A/B of group 2 against group 4 at a bin count of 64 or below,
    in one process, the protocol `bench_histogram.mojo` already implements.

    **The elided hessian plane.** `const_hess` carries the same declaration
    the atomic kernel's docstring argues in full, and has one extra
    consequence here: the partial buffer holds two planes per tile rather
    than three, laid out as `[grad | count]`, so the write this kernel does
    and the read `_range_reduce_kernel` does are both a third smaller. That
    layout change is contained: the partial buffer is written by this kernel
    and read by that one and by nothing else in the package, and the
    allocation is unchanged, so a two-plane build simply uses two thirds of a
    buffer sized for three. The two kernels are handed the same flag from the
    same launch site, which is what keeps them agreeing about which layout
    they are looking at.

    The reconstruction happens in the reduction rather than here, because a
    partial tile's count is not the bin's count. Multiplying each tile's
    count by `hq_const` and summing would give the same integer -- integer
    multiplication distributes over addition exactly, at every width, and
    modulo 2^32 as well -- but it would spend one multiply per tile per cell
    instead of one per cell, so the reduction is where it belongs.
    """
    var slot0 = GROUP * Int(block_idx.x)
    var owned = Int(n_slots) - slot0
    if owned > GROUP:
        owned = GROUP
    var t = block_idx.y
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var n = Int(count)
    var plane = Int(n_slots) * nb
    var celide = Int(const_hess) != 0
    # Planes per tile in the partial buffer: `[grad | hess | count]`, or
    # `[grad | count]` when the hessian plane is elided.
    var n_planes = 2 if celide else 3

    var sg = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sc = stack_allocation[
        GROUP * BIN_CAP,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var span = owned * nb
    var b = tid
    while b < span:
        sg[unsafe_offset=b] = 0
        if not celide:
            sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var tile_begin = t * Int(rows_per_tile)
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    _hist_accumulate_dispatch[GROUP, HIST_ROW_UNROLL](
        bins,
        rows,
        grad,
        hess,
        gq,
        feat_ids,
        sg,
        sh,
        sc,
        slot0,
        owned,
        nb,
        nr,
        Int(begin),
        tile_begin,
        tile_end,
        g_scale,
        h_scale,
        celide,
        Int(use_quant) != 0,
        Int(row_unroll) != 0,
    )
    barrier()

    # The count plane is last in both layouts, so its index is
    # `n_planes - 1`: plane 2 of three, plane 1 of two.
    var count_plane = (n_planes - 1) * plane
    comptime for k in range(GROUP):
        if k < owned:
            var base = t * n_planes * plane + (slot0 + k) * nb
            var lift = k * nb
            var c = tid
            while c < nb:
                var s = lift + c
                partials[unsafe_offset = base + c] = sg[unsafe_offset=s][0]
                if not celide:
                    partials[unsafe_offset = base + plane + c] = sh[
                        unsafe_offset=s
                    ][0]
                partials[unsafe_offset = base + count_plane + c] = sc[
                    unsafe_offset=s
                ][0]
                c += block_dim.x


def _range_reduce_kernel(
    partials: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    n_tiles: Int32,
    sub_offset: Int32,
    do_sub: Int32,
    h_scale: Float32,
    const_hess: Int32,
):
    """Element-wise reduction of the tiled partials, in ascending tile order.

    Byte for byte what `histogram_gpu._hist_reduce_kernel` does: the
    reduction depends on the partial layout only, not on which rows produced
    it, so integration should delete this copy and call the existing kernel.
    It lives here so the range path can be exercised end to end from this
    module alone.

    With `do_sub`, the same thread that writes a cell also subtracts it from
    the slot `sub_offset` words away, which is the sibling subtraction folded
    into the build. The destination is a signed offset rather than a second
    pointer because both slots live in one pool buffer. Each
    thread owns one cell of each slot and no thread reads a cell another
    writes, so the in-place update is race free even though the destination
    is a live histogram. The cells this reduction does not cover are the
    inactive features', which are zero in every live slot -- the build zeroes
    a whole slot whenever the feature set is narrowed -- so subtracting them
    would have been a no-op.

    **The elided hessian plane.** With `const_hess` the partials hold two
    planes per tile, `[grad | count]`, so the grid covers two thirds of the
    cells and each thread reads two thirds of the words. The thread that owns
    a count cell writes two output cells instead of one: the count itself,
    and the hessian plane as `hq_const * acc`, where `hq_const` is
    `Int32(round(Float32(1.0) * h_scale))` and `acc` is the bin's total row
    count. That is the exact integer the three-plane path would have reduced
    to, because on the three-plane path every row contributed the same
    `hq_const` to the hessian plane and `acc` rows contributed in total; the
    argument is written out in full in `_range_hist_atomic_kernel`.

    The fused subtraction stays exact for the same reason and needs no
    special case beyond doing it twice: the sibling's hessian cell is reduced
    by `hq_const * acc`, which is what the three-plane path would have
    subtracted from it.

    No thread races another over the extra write. The hessian output cell is
    reached only from the count plane's thread for the same (feature, bin),
    and on the elided path no thread is assigned the hessian plane at all,
    since the grid is sized by the partial layout rather than by the output
    layout.
    """
    var i = global_idx.x
    var nb = Int(n_bins)
    var plane = Int(n_slots) * nb
    var celide = Int(const_hess) != 0
    var n_planes = 2 if celide else 3
    var n = n_planes * plane
    if i < n:
        var acc = Int32(0)
        var off = i
        for _ in range(Int(n_tiles)):
            acc += partials[unsafe_offset=off][0]
            off += n
        var p = i // plane
        var rem = i - p * plane
        var slot = rem // nb
        var b = rem - slot * nb
        var f = Int(feat_ids[unsafe_offset=slot][0])
        # Plane `p` of the partial layout maps to plane `p` of the output,
        # except that the elided layout's second plane is the count, which is
        # the output's third.
        var out_plane = 2 if (celide and p == 1) else p
        var cell = out_plane * Int(hist_size) + f * nb + b
        out_hist[unsafe_offset=cell] = acc
        if Int(do_sub) != 0:
            var sc = Int(sub_offset) + cell
            out_hist[unsafe_offset=sc] = out_hist[unsafe_offset=sc][0] - acc
        if celide and p == 1:
            var hq_const = Int32(round(Float32(1.0) * h_scale))
            var hv = hq_const * acc
            var hcell = Int(hist_size) + f * nb + b
            out_hist[unsafe_offset=hcell] = hv
            if Int(do_sub) != 0:
                var hs = Int(sub_offset) + hcell
                out_hist[unsafe_offset=hs] = (
                    out_hist[unsafe_offset=hs][0] - hv
                )


@fieldwise_init
struct LeafRange(Copyable, Movable):
    """The half-open window `[begin, end)` of the active-row buffer owned by
    one leaf. An empty range (`begin == end`) means the leaf holds no rows,
    which is what a split leaf becomes: its rows now belong to its
    children."""

    var begin: Int
    var end: Int

    @staticmethod
    def empty() -> LeafRange:
        return LeafRange(0, 0)

    def count(self) -> Int:
        return self.end - self.begin

    def is_empty(self) -> Bool:
        return self.end <= self.begin

    def contains(self, index: Int) -> Bool:
        return index >= self.begin and index < self.end

    def overlaps(self, other: LeafRange) -> Bool:
        if self.is_empty() or other.is_empty():
            return False
        return self.begin < other.end and other.begin < self.end


struct LeafRangeTable(Copyable, Movable):
    """Node id -> `LeafRange`, for one tree.

    Node ids double as leaf ids in the GPU grower and are handed out in
    ascending order, so the table is a `List` indexed by node id that grows
    by appending. Splitting a leaf costs at most two appends and no device
    work: the children's ranges are the two halves of the parent's, and the
    parent's becomes empty so the live ranges keep tiling `[0, n_active)`.
    """

    var ranges: List[LeafRange]
    var n_rows: Int
    var n_active: Int

    def __init__(out self, n_rows: Int):
        self.ranges = List[LeafRange]()
        self.n_rows = n_rows
        self.n_active = 0

    def reset_root(mut self, n_active: Int) raises:
        """Start a new tree with `n_active` rows at node 0."""
        if n_active < 0 or n_active > self.n_rows:
            raise Error("active row count out of range")
        self.ranges.clear()
        self.ranges.append(LeafRange(0, n_active))
        self.n_active = n_active

    def n_nodes(self) -> Int:
        return len(self.ranges)

    def get(self, node: Int) raises -> LeafRange:
        if node < 0 or node >= len(self.ranges):
            raise Error("leaf id has no active-row range")
        return self.ranges[node].copy()

    def _grow_to(mut self, node: Int):
        while len(self.ranges) <= node:
            self.ranges.append(LeafRange.empty())

    def split(
        mut self, parent: Int, left: Int, right: Int, n_left: Int
    ) raises -> LeafRange:
        """Hand the parent's range to its two children and return the left
        child's. `n_left` is the number of rows the partition sent left; the
        left child takes the parent's first `n_left` slots and the right
        child the rest, which is exactly what the stable partition wrote."""
        var r = self.get(parent)
        if n_left < 0 or n_left > r.count():
            raise Error("left row count is outside the parent's range")
        if left < 0 or right < 0:
            raise Error("leaf ids must be nonnegative")
        if left > MAX_ROWS or right > MAX_ROWS or parent > MAX_ROWS:
            raise Error("leaf ids must fit in Int32")
        if left == parent or right == parent or left == right:
            raise Error(
                "child leaf ids must differ from the parent and each other"
            )
        var mid = r.begin + n_left
        self._grow_to(left)
        self._grow_to(right)
        # A leaf that already owns rows would be overwritten, which would
        # silently orphan them; ids come from the tree's node counter, so
        # this can only fire on a wiring mistake.
        if not self.ranges[left].is_empty():
            raise Error("left child already owns an active-row range")
        if not self.ranges[right].is_empty():
            raise Error("right child already owns an active-row range")
        self.ranges[left] = LeafRange(r.begin, mid)
        self.ranges[right] = LeafRange(mid, r.end)
        self.ranges[parent] = LeafRange.empty()
        return LeafRange(r.begin, mid)

    def total_active(self) -> Int:
        var total = 0
        for i in range(len(self.ranges)):
            total += self.ranges[i].count()
        return total

    def check_invariants(self) raises:
        """The live ranges must tile `[0, n_active)`: inside the buffer,
        pairwise disjoint, and covering every active slot exactly once.

        Disjointness plus a total of `n_active` is coverage, since the
        ranges all sit inside a buffer of that length. Tree growth here is a
        few hundred leaves at most, so the quadratic check is cheaper than
        sorting and is what a test wants to read.
        """
        for i in range(len(self.ranges)):
            var a = self.ranges[i].copy()
            if a.is_empty():
                continue
            if a.begin < 0 or a.end > self.n_active:
                raise Error("active-row range escapes the active prefix")
            for k in range(i + 1, len(self.ranges)):
                if a.overlaps(self.ranges[k]):
                    raise Error("active-row ranges overlap")
        if self.total_active() != self.n_active:
            raise Error("active-row ranges do not cover the active prefix")


@fieldwise_init
struct RowRouting(Copyable, Movable):
    """Everything a partition needs to route one leaf's rows.

    The same three fields the host grower, `Tree.goes_left`, and the device
    kernels already agree on, gathered so the host reference model and the
    kernels take their arguments from one place and cannot drift apart.
    """

    var feature: Int
    var threshold_bin: Int
    var missing_bin: Int
    var default_left: Bool
    var is_categorical: Bool
    var cat_bitset: CatBitset

    @staticmethod
    def numerical(
        feature: Int,
        threshold_bin: Int,
        missing_bin: Int = -1,
        default_left: Bool = False,
    ) -> RowRouting:
        return RowRouting(
            feature,
            threshold_bin,
            missing_bin,
            default_left,
            False,
            cat_empty(),
        )

    @staticmethod
    def categorical(feature: Int, bitset: CatBitset) -> RowRouting:
        """Bin 0 is never in the set, so missing, unseen, and dropped
        categories route right, as they do everywhere else."""
        return RowRouting(feature, -1, -1, False, True, bitset)

    @staticmethod
    def from_split(split: SplitInfo, missing_bin: Int) -> RowRouting:
        """The routing a chosen split implies. `missing_bin` is the split
        feature's missing bin, or -1; a categorical split ignores it, exactly
        as the grower does when it passes -1 for one."""
        if split.is_categorical:
            return RowRouting.categorical(split.feature, split.cat_bitset)
        return RowRouting.numerical(
            split.feature, split.bin, missing_bin, split.default_left
        )

    def goes_left(self, bin: Int) -> Bool:
        """Host mirror of `_row_goes_left`."""
        return _row_goes_left(
            Int32(bin),
            Int32(self.threshold_bin),
            Int32(self.missing_bin),
            Int32(1) if self.default_left else Int32(0),
            Int32(1) if self.is_categorical else Int32(0),
            self.cat_bitset[0],
            self.cat_bitset[1],
            self.cat_bitset[2],
            self.cat_bitset[3],
        )

    def check(self, n_features: Int, n_bins: Int) raises:
        if self.feature < 0 or self.feature >= n_features:
            raise Error("split feature out of range")
        if not self.is_categorical and (
            self.threshold_bin < 0 or self.threshold_bin >= n_bins
        ):
            raise Error("split threshold bin out of range")
        if self.missing_bin >= n_bins:
            raise Error("split missing bin out of range")


def partition_range_host(
    mut rows: List[Int32],
    mut scratch: List[Int32],
    data: BinnedMatrix,
    window: LeafRange,
    routing: RowRouting,
) raises -> Int:
    """Stable-partition `rows[window]` in place and return the left count.

    The reference model for the device kernels: same routing rule, same
    stable order, same in-place contiguous result, computed serially on the
    host. Tests hold the device output to it, and it documents the contract
    in code that can be read without a GPU. `scratch` is caller-owned and
    only its `window` window is used, so repeated calls allocate nothing.
    """
    if len(rows) != len(scratch):
        raise Error("scratch must be the same length as the row buffer")
    if window.begin < 0 or window.end > len(rows) or window.count() < 0:
        raise Error("range escapes the row buffer")
    routing.check(data.n_features, data.n_bins)

    var n_left = 0
    for j in range(window.begin, window.end):
        var row = Int(rows[j])
        if row < 0 or row >= data.n_rows:
            raise Error("active row index out of range")
        if routing.goes_left(data.bin_at(row, routing.feature)):
            n_left += 1

    var li = window.begin
    var ri = window.begin + n_left
    for j in range(window.begin, window.end):
        var row = rows[j]
        if routing.goes_left(data.bin_at(Int(row), routing.feature)):
            scratch[li] = row
            li += 1
        else:
            scratch[ri] = row
            ri += 1
    for j in range(window.begin, window.end):
        rows[j] = scratch[j]
    return n_left


struct GpuActiveRows(Movable):
    """The device-resident active-row permutation and its leaf ranges.

    Construct once per training session against the histogram builder's
    `DeviceContext`, call `begin_tree` per tree, `partition` per split, and
    `enqueue_range_histogram` per node. It owns only the index machinery: the
    binned matrix, the gradients, and the histogram output stay with
    `GpuHistogramBuilder` and arrive as pointers.
    """

    var ctx: DeviceContext
    # The active-row permutation; `rows[0 : n_active)` is live.
    var rows_dev: DeviceBuffer[DType.int32]
    # Scatter destination. Only the partitioned range is ever read back out
    # of it, so the two buffers never have to agree outside that window.
    var scratch_dev: DeviceBuffer[DType.int32]
    # Chunk-local exclusive left-prefix, one per element of the range, with
    # the element's own routing flag packed into the low bit. A chunk is one
    # threadgroup's share of the range (`_partition_grid`).
    var offsets_dev: DeviceBuffer[DType.int32]
    # Per-chunk left counts. Written by the flag pass and read, never
    # rewritten, by every block of the scatter, which scans them at its own
    # head; they were turned into block offsets in place by a launch of their
    # own until that fold removed it.
    var block_sums_dev: DeviceBuffer[DType.int32]
    # One Int32: the range's total left count, written by the scatter.
    var total_dev: DeviceBuffer[DType.int32]
    # The step descriptor (see "The step descriptor"), `STEP_WORDS` Int32.
    #
    # It lives here rather than with the tree tables that write it for two
    # reasons. The kernels that read it are all in this file, so the buffer
    # sits with its readers; and every host-driven launch has to pass
    # *something* for the descriptor argument, which must be a real pointer
    # and must not be any other buffer that launch already passes, because a
    # kernel launch may not be handed the same buffer twice mutably. Reusing
    # one of the partition's own buffers as the inert stand-in is what the
    # first draft did and the compiler correctly refused it.
    #
    # Nothing here writes it. `gpu_tree_tables.DeviceTreeTables.enqueue_step`
    # takes this pointer and its commit kernel fills it, which is the whole
    # of the coupling between that module and this one.
    var step_dev: DeviceBuffer[DType.int32]
    var host_total: HostBuffer[DType.int32]
    var host_rows: HostBuffer[DType.int32]
    var stage_rows: HostBuffer[DType.int32]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var block_threads: Int
    var ranges: LeafRangeTable
    var verify_counts: Bool
    # Feature slots one histogram threadgroup accumulates: a rung of the
    # ladder 1, 2, 4, 8, 16, each of which has a kernel instantiation. Never
    # read by anything but `enqueue_range_histogram`, and it changes no
    # histogram it produces.
    var feature_group: Int
    # The interleaved pre-quantized gradient buffer (`_quantize_grad_hess_kernel`),
    # 2 * n_rows Int32, and the state that says whether it describes the
    # gradients the next histogram will be asked to read. It is invalidated by
    # `begin_tree`, which every gradient refill in train_gpu.mojo is followed
    # by, and by a change of scale, which any refill that changed a value
    # changes with overwhelming likelihood; the two together are what make it
    # safe to key on rather than on a round counter this module cannot see.
    var gq_dev: DeviceBuffer[DType.int32]
    var quantized_gradients: Bool
    var quant_valid: Bool
    var quant_g_scale: Float32
    var quant_h_scale: Float32
    # scan-primitives lane: whether the two scan stages run on
    # `block.prefix_sum` (the default) or on the hand-rolled Hillis-Steele
    # kernels. A launch-shape knob and nothing else, because the two arms
    # produce the identical permutation and the identical left count.
    var scan_primitives: Bool
    # The most threadgroups one partition launches, whatever the range
    # length; a longer range is covered by giving each of them more tiles to
    # walk rather than by launching more of them (`_partition_grid`). It is a
    # cap and not a count: a range short enough to fit in fewer blocks
    # launches fewer. Bounded because the scatter scans the block sums at its
    # own head, once per block, so the redundant reads grow as the square of
    # the block count; see `_scatter_kernel` for the bound the default
    # (`block_threads`) buys. A launch-shape knob and nothing else, because
    # the permutation is a function of the routing flags and each row's
    # position in the range and so cannot see the tiling at all.
    var partition_block_cap: Int
    # --- hist-kernel-family lane ---
    # The shared-plane width every histogram launch from here instantiates its
    # kernel at: `histogram_bin_capacity(n_bins)`, one of 32, 64, 128, 256.
    # Held rather than recomputed per node because it is a property of the
    # dataset, and read by `set_feature_group` so a width whose footprint the
    # device cannot hold is refused where it is asked for rather than at the
    # launch that would fail.
    var bin_cap: Int
    # `caps.max_shared_memory_per_block` as reported when this was
    # constructed. The only device fact this struct keeps, and it keeps it for
    # one reason: `3 * group * bin_cap * 4` has to be checked against
    # something before a group is accepted.
    var max_shared_bytes: Int
    # --- const-hessian lane ---
    # Whether the caller has declared that this round's objective guarantees a
    # per-row hessian of exactly `histogram.CONSTANT_HESSIAN`, and whether the
    # environment permits acting on such a declaration at all. The effective
    # answer is the conjunction, kept in `constant_hessian` so the launch site
    # reads one field. Declared, never inferred: see
    # `histogram.objective_has_constant_hessian` for why a round whose
    # hessians happen to be equal is not the same thing.
    var constant_hessian: Bool
    var const_hessian_allowed: Bool
    # --- hist-kernel-margin lane ---
    # Whether the histogram row loop walks `HIST_ROW_UNROLL` rows per
    # iteration with the loads of each stage issued together, or one row per
    # iteration as it did before that lane. A launch-shape knob and nothing
    # else: both arms visit the same rows and add the same integers, only in
    # a different order, and integer addition does not care
    # (`_hist_rows_step`).
    #
    # On by default, on the same footing the primitive scan arm is on by
    # default: it is strictly fewer instructions and, on the quantized arm,
    # strictly fewer loads for the identical result, so it does not need a
    # measurement to justify being the default. What it is not is free of
    # risk -- it raises the live register count in the row loop, and
    # threadgroup residency on this backend is bounded by things this project
    # cannot query -- which is why `set_row_unroll` exists.
    #
    # No environment variable, for the reason `partition_block_cap` gives:
    # the arm belongs in the call, because an A/B that reads its arm from the
    # environment has already once in this repository run one arm under the
    # other's label.
    var row_unroll: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        caps: DeviceCaps,
    ) raises:
        """Allocate every buffer the whole session will use.

        Sized by `n_rows`, never by a node: a split allocates nothing. The
        scan block size is the tiling policy's block size, clamped to what
        the shared-memory scan buffer holds.
        """
        if n_rows < 1:
            raise Error("GPU backend requires at least one row")
        if n_features < 1:
            raise Error("GPU backend requires at least one feature")
        if n_bins < 1:
            raise Error("GPU backend requires at least one bin")
        if n_bins > MAX_BINS:
            raise Error("GPU backend supports at most 256 bins")
        if n_rows > MAX_ROWS:
            raise Error("GPU backend supports at most 2^31 - 1 rows")

        self.ctx = ctx
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins

        var threads = derive_block_threads(caps)
        if threads > SCAN_MAX_THREADS:
            threads = SCAN_MAX_THREADS
        self.block_threads = threads
        var max_blocks = (n_rows + threads - 1) // threads

        self.rows_dev = self.ctx.enqueue_create_buffer[DType.int32](n_rows)
        self.scratch_dev = self.ctx.enqueue_create_buffer[DType.int32](n_rows)
        self.offsets_dev = self.ctx.enqueue_create_buffer[DType.int32](n_rows)
        self.block_sums_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_blocks
        )
        self.total_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.step_dev = self.ctx.enqueue_create_buffer[DType.int32](STEP_WORDS)
        self.host_total = self.ctx.enqueue_create_host_buffer[DType.int32](1)
        self.host_rows = self.ctx.enqueue_create_host_buffer[DType.int32](
            n_rows
        )
        self.stage_rows = self.ctx.enqueue_create_host_buffer[DType.int32](
            n_rows
        )
        self.ranges = LeafRangeTable(n_rows)
        # Off by default: the grower already knows the exact left count from
        # the parent's integer histogram counts, so the download is a check,
        # not a dependency. `MOJOTREES_GPU_VERIFY_ROWS=1` turns it on, which
        # is what tests and a first integration want.
        self.verify_counts = _env_int("MOJOTREES_GPU_VERIFY_ROWS", 0) != 0
        # The shared-plane width every kernel launched from here is
        # instantiated at, fixed by the dataset's bin count.
        self.bin_cap = histogram_bin_capacity(n_bins)
        self.max_shared_bytes = caps.max_shared_memory_per_block
        # One feature slot per threadgroup unless asked otherwise. Every rung
        # of the ladder produces the identical histogram, so this is a
        # launch-shape knob and nothing else. This constructor cannot see
        # which API the context speaks, so the per-device default lives in the
        # caller: GpuHistogramBuilder widens it by the free-footprint rule in
        # `gpu_tiling.free_feature_group` when the environment does not
        # choose. A request that is not a rung, or one whose footprint this
        # device cannot hold, is rounded down here rather than raised, because
        # a constructor is not where an environment variable should fail a
        # fit; `set_feature_group` raises, because there the width came from a
        # caller who can be told.
        var group = _env_int("MOJOTREES_GPU_FEATURE_GROUP", 1)
        if group < 1:
            group = 1
        if group > FEATURE_GROUP_MAX:
            group = FEATURE_GROUP_MAX
        var rung = 1
        var chosen = 1
        for _ in range(FEATURE_GROUP_LADDER):
            if rung <= group and (
                histogram_shared_bytes(self.bin_cap, rung)
                <= self.max_shared_bytes
            ):
                chosen = rung
            rung *= 2
        self.feature_group = chosen
        # Pre-quantized interleaved gradients, the default source for every
        # histogram kernel. `MOJOTREES_GPU_QUANTIZED_GRADS=0` or
        # `set_quantized_gradients(False)` forces the Float32 planes back, so
        # a benchmark can hold both arms. Cannot change a histogram, only what
        # the kernel gathers per row; see `_quantize_grad_hess_kernel` for why
        # that is an argument from the expression rather than a measurement.
        self.gq_dev = self.ctx.enqueue_create_buffer[DType.int32](2 * n_rows)
        self.quantized_gradients = (
            _env_int("MOJOTREES_GPU_QUANTIZED_GRADS", 1) != 0
        )
        self.quant_valid = False
        self.quant_g_scale = 0.0
        self.quant_h_scale = 0.0
        # The primitive scan arm is the default: it computes the same
        # permutation with strictly less work, so it does not need a
        # measurement to justify being on. `MOJOTREES_GPU_SCAN_PRIMITIVES=0`
        # takes it off, and a width the primitive has no instantiation for
        # takes it off here rather than at every launch.
        self.scan_primitives = _env_int(
            "MOJOTREES_GPU_SCAN_PRIMITIVES", 1
        ) != 0 and _scan_primitive_width_supported(threads)
        # One threadgroup width's worth of scan blocks. Chosen by the
        # counting argument in `_scatter_kernel` and not by a measurement:
        # at this cap the scatter's redundant re-read of the block sums is
        # bounded by a third of the range's own streamed traffic in the worst
        # case, and every range shorter than `threads * threads` gets exactly
        # the one-tile-per-block geometry the module launched before the head
        # scan was folded in. `set_partition_block_cap` moves it, which is
        # how a benchmark holds two caps in one process and how the test file
        # forces the multi-tile path on a range small enough to check by
        # hand. No environment variable: the arm belongs in the call, because
        # an A/B that reads its arm from the environment has already once in
        # this repository run one arm under the other's label.
        self.partition_block_cap = threads
        # The constant-hessian specialization is available but not declared.
        # A builder that says nothing gets the three-plane path that shipped,
        # which is the only default a specialization keyed on the objective
        # can have: this constructor cannot see an objective.
        # `MOJOTREES_CONST_HESSIAN=0` withdraws the permission, so a later
        # `set_constant_hessian(True)` is refused rather than silently
        # honored, which is the off switch a bisection wants.
        self.const_hessian_allowed = _env_int("MOJOTREES_CONST_HESSIAN", 1) != 0
        self.constant_hessian = False
        # The unrolled row walk is the default; see the field.
        self.row_unroll = True

        # A bagged tree stages only its bag's slots, and the copy that
        # follows takes the whole buffer, so the tail is zeroed once here
        # rather than left as whatever the allocation held. No kernel reads
        # past the root range, so this is hygiene, not correctness.
        var stage = self.stage_rows.unsafe_ptr()
        for r in range(n_rows):
            stage.unsafe_store(r, Int32(0))

    def synchronize(mut self) raises:
        self.ctx.synchronize()

    def n_active(self) -> Int:
        """Rows this tree grows on: the bag, or every row when unbagged."""
        return self.ranges.n_active

    def range_of(self, node: Int) raises -> LeafRange:
        return self.ranges.get(node)

    def set_feature_group(mut self, group: Int) raises:
        """How many feature slots one histogram threadgroup accumulates.

        An argument rather than only an environment variable, because an A/B
        that reads its arm from the environment can silently run one arm
        under the other's label; that happened once in this repository with
        the split-search strategy and the benchmark harness now passes the
        arm in. Takes effect on the next `enqueue_range_histogram`, and
        cannot change a histogram, only the launch that builds it.

        Two refusals, and they are different refusals. A width that is not a
        rung of the ladder (3, or anything above 16) has no kernel
        instantiation and could not launch. A width that is a rung but whose
        three `group * bin_cap` Int32 planes exceed what this device reported
        for one threadgroup would compile and then fail at the launch, which
        is a worse place to find out; at 256 bins that is every group past 2
        on a 32 KiB budget, and at 32 bins it is nothing on the ladder.
        """
        if not is_feature_group_width(group):
            raise Error(
                "feature group must be 1, 2, 4, 8, or 16: those are the"
                " widths a kernel is instantiated at"
            )
        var need = histogram_shared_bytes(self.bin_cap, group)
        if need > self.max_shared_bytes:
            raise Error(
                "a feature group of ",
                group,
                " at ",
                self.bin_cap,
                " bins needs ",
                need,
                " bytes of threadgroup memory and this device reported ",
                self.max_shared_bytes,
            )
        self.feature_group = group

    def set_quantized_gradients(mut self, on: Bool):
        """Whether every histogram kernel reads the pre-quantized interleaved
        gradient buffer instead of the two Float32 planes.

        On by default, unlike when this was wired into one kernel only. The
        reason it can be a default is that it is exact by construction rather
        than by measurement: `gq[2r]` is the same Float32 multiply and the
        same `round` the row loop would have evaluated, hoisted out of a loop
        it does not depend on, so the histogram is bit-identical and the row
        loop does strictly less work. See `_quantize_grad_hess_kernel`.

        It stays an argument, and `MOJOTREES_GPU_QUANTIZED_GRADS=0` stays a
        way to force the Float32 planes, for the same reason
        `set_feature_group` is an argument: an A/B holds both arms in one
        process rather than reading its arm from the environment. Takes
        effect on the next `enqueue_range_histogram` and cannot change a
        histogram.
        """
        self.quantized_gradients = on
        self.quant_valid = False

    def set_row_unroll(mut self, on: Bool):
        """Whether the histogram row loop keeps `HIST_ROW_UNROLL` rows in
        flight or walks one row per iteration.

        Off is the shape this module shipped before the hist-kernel-margin
        lane, and it is reachable at run time rather than through a second
        kernel instantiation, so a benchmark can hold both arms in one
        process. That matters more here than it usually would: this machine's
        device timings drift several-fold between time windows, so only
        interleaved arms compare, and a comptime-only knob would have forced
        a two-build comparison instead.

        It cannot change a histogram. Both arms visit the same rows of the
        same range and add the same fixed-point integers into the same bins;
        only the order of the adds and of the loads that feed them differs,
        and integer addition is associative and commutative. See
        `_hist_rows_step` for the argument written out, and note that the one
        floating-point expression involved, the Float32 arm's
        `Int32(round(x * scale))`, is reproduced unchanged in both arms
        because `docs/NUMERICS.md` section 5.6 depends on its exact form.

        Takes effect on the next `enqueue_range_histogram`.
        """
        self.row_unroll = on

    def set_constant_hessian(mut self, on: Bool):
        """Declare that every row's hessian this round is exactly
        `histogram.CONSTANT_HESSIAN`, so the histogram kernels may stop
        accumulating the hessian plane and reconstruct it from the count.

        **This is a declaration about the objective, and the caller owns
        it.** `histogram.objective_has_constant_hessian` is the predicate
        that answers it correctly, and it is false for every weighted fit and
        for every GOSS round regardless of the objective code, because both
        put more than one value into `hess`. Declaring it on a round where it
        is not true produces a wrong hessian plane, silently, so it belongs
        next to where the trainer decides whether to pass weights and not
        anywhere further down.

        An argument as well as an environment variable, for the reason
        `set_feature_group` gives: an A/B that reads its arm from the
        environment can run one arm under the other's label, which happened
        once in this repository, so a benchmark holding both arms in one
        process passes the arm in rather than re-execing.
        `MOJOTREES_CONST_HESSIAN=0` wins over an argument in the one
        direction that is always safe, off, and this method reports what it
        actually did through `constant_hessian`.

        Takes effect on the next `enqueue_range_histogram`. When the
        declaration is true it cannot change a histogram: the plane it stops
        accumulating is reconstructed as the identical Int32, argued in
        `_range_hist_atomic_kernel`.
        """
        self.constant_hessian = on and self.const_hessian_allowed

    def set_scan_primitives(mut self, on: Bool) raises:
        """Whether the partition's two scan stages run on `block.prefix_sum`
        or on the hand-rolled Hillis-Steele kernels.

        An argument as well as an environment variable for the reason
        `set_feature_group` gives: an A/B that reads its arm from the
        environment can run one arm under the other's label, which has
        happened in this repository once, so a benchmark holding both arms in
        one process passes the arm in instead of re-execing. Takes effect on
        the next `enqueue_partition`. It cannot change the permutation or the
        left count, only which kernel computes the prefix sums, and the two
        being identical is what tests/test_gpu_scan_primitives.mojo asserts.

        Turning it on at a threadgroup width no primitive instantiation
        exists for raises rather than silently leaving the fallback arm
        selected, because a benchmark told to run the primitive arm must not
        quietly measure the other one.
        """
        if on and not _scan_primitive_width_supported(self.block_threads):
            raise Error(
                "the primitive scan arm has no kernel at this threadgroup"
                " width: 128, 256, 512, and 1024 are the widths instantiated"
            )
        self.scan_primitives = on

    def set_partition_block_cap(mut self, cap: Int) raises:
        """The most threadgroups one partition may launch.

        A launch-shape knob in exactly the sense `set_feature_group` and
        `set_scan_primitives` are: it changes how the range is cut into
        chunks and therefore how much redundant block-sums scanning the
        scatter does, and it cannot change one element of the permutation or
        one unit of the left count, because a row's destination is a function
        of the routing flags and of the row's position in the range and of
        nothing else. `tests/test_gpu_partition_launches.mojo` asserts that
        across caps rather than leaving it as a claim here.

        Two reasons it is settable. A test needs to reach the multi-tile path
        without allocating a range of `block_threads * block_threads` rows, so
        it lowers the cap instead and gets the same code path on a range small
        enough to check against the serial reference model. And a benchmark
        that wants to know whether the default (`block_threads`, chosen by a
        counting argument in `_scatter_kernel` and by no measurement at all)
        is the right cap on a given device has to be able to hold two caps in
        one process.

        Takes effect on the next `enqueue_partition`. There is no upper bound
        beyond the one the block-sums buffer already implies: the scatter's
        head scan walks the sums in chunks of a threadgroup width, so it
        serves any block count, and `_partition_grid` never returns more
        blocks than the range has tiles.
        """
        if cap < 1:
            raise Error("partition block cap must be at least one block")
        self.partition_block_cap = cap

    def begin_tree(mut self, bag: List[Int] = []) raises:
        """Seed the root's rows and make node 0 own all of them.

        Unbagged, the root is the identity permutation, written by a kernel
        so a tree costs no host-to-device row transfer at all. Bagged, the
        bag's rows are staged and copied in the caller's order, which is the
        order the CPU grower's root list has, and the rows left out are
        simply not inside the root range: no sentinel leaf id, no filtering,
        and no cost per node for a row this tree ignores.
        """
        if len(bag) > self.n_rows:
            raise Error("bag is larger than the dataset")
        # A new tree may carry a new round's gradients; the quantized copy
        # is rebuilt on the first histogram that asks for it.
        self.quant_valid = False
        if len(bag) == 0:
            var blocks = (
                self.n_rows + self.block_threads - 1
            ) // self.block_threads
            self.ctx.enqueue_function[_iota_kernel](
                self.rows_dev.unsafe_ptr(),
                Int32(self.n_rows),
                grid_dim=blocks,
                block_dim=self.block_threads,
            )
            self.ranges.reset_root(self.n_rows)
            return

        for i in range(len(bag)):
            if bag[i] < 0 or bag[i] >= self.n_rows:
                raise Error("bag row index out of range")
        # Any copy still reading the staging buffer has to finish before it
        # is overwritten.
        self.ctx.synchronize()
        var dst = self.stage_rows.unsafe_ptr()
        for i in range(len(bag)):
            dst.unsafe_store(i, Int32(bag[i]))
        self.ctx.enqueue_copy(dst_buf=self.rows_dev, src_ptr=dst)
        self.ranges.reset_root(len(bag))

    def enqueue_partition[
        bins_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        window: LeafRange,
        routing: RowRouting,
    ) raises:
        """Enqueue the stable partition of `window`. Does not transfer or
        synchronize; the left count lands in `total_dev`.

        Three launches, whatever the range length: the flag-and-scan pass,
        the scatter (which scans the block sums at its own head), and the
        copy back. It was four until this lane folded the block-sums scan
        into the scatter. The saving is a launch, not bytes: nothing about
        what is read or written per row changed, and the copy back still
        moves 8 bytes per row of the range that per-range ping-pong would
        remove, which `_copy_back_kernel` writes up as a design and explains
        why this lane did not build.

        No timing is claimed. The stage profile that motivated the lane
        attributes 17.7 percent of a 1M by 50 GPU round to partition on an
        M4, of which an audit believes about half is fence attribution, and
        puts partition's floor at a four-launch minimum of roughly 237
        microseconds against a per-enqueue cost of roughly 20; that arithmetic
        says a launch is worth removing, and it is also in tension with the
        earlier finding recorded in the module docstring, that collapsing a
        small range's four launches into one measured 1.00x. Both of those
        are measurements this lane could not repeat and did not repeat. What
        is asserted here is only that three launches now do what four did,
        element for element.
        """
        routing.check(self.n_features, self.n_bins)
        if window.begin < 0 or window.end > self.n_rows:
            raise Error("range escapes the row buffer")
        var n = window.count()
        if n <= 0:
            return

        var threads = self.block_threads
        var grid = _partition_grid(n, threads, self.partition_block_cap)
        var blocks = grid[0]
        var tiles = grid[1]
        # The copy back has no cross-block dependency, so it keeps the plain
        # one-tile-per-block grid rather than the capped one the scan and the
        # scatter share.
        var copy_blocks = (n + threads - 1) // threads
        var cat = routing.cat_bitset
        var default_left = Int32(1) if routing.default_left else Int32(0)
        var is_cat = Int32(1) if routing.is_categorical else Int32(0)

        # The flag pass and the scatter, on whichever arm is selected. The
        # primitive kernels are instantiated per width, so the dispatch is an
        # if-chain over the menu rather than a runtime block size; the whole
        # chain sits inside a `comptime if has_accelerator()` because a build
        # with no accelerator target has no warp size for `block.prefix_sum`
        # to constrain against, and pruning the branch is what keeps those
        # instantiations out of a CPU-only extension build entirely.
        var scanned = False
        comptime if has_accelerator():
            if self.scan_primitives:
                scanned = True
                if threads == 128:
                    self._enqueue_scan_primitives[128](
                        bins, window, routing, n, blocks, tiles
                    )
                elif threads == 256:
                    self._enqueue_scan_primitives[256](
                        bins, window, routing, n, blocks, tiles
                    )
                elif threads == 512:
                    self._enqueue_scan_primitives[512](
                        bins, window, routing, n, blocks, tiles
                    )
                elif threads == 1024:
                    self._enqueue_scan_primitives[1024](
                        bins, window, routing, n, blocks, tiles
                    )
                else:
                    # `__init__` and `set_scan_primitives` both refuse to
                    # select the arm at an uninstantiated width, so this is
                    # unreachable; falling back rather than raising keeps a
                    # width that slipped through producing correct rows.
                    scanned = False

        if not scanned:
            self.ctx.enqueue_function[_flag_scan_kernel](
                bins,
                self.rows_dev.unsafe_ptr(),
                self.offsets_dev.unsafe_ptr(),
                self.block_sums_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(window.begin),
                Int32(n),
                Int32(routing.feature),
                Int32(routing.threshold_bin),
                Int32(routing.missing_bin),
                default_left,
                is_cat,
                cat[0],
                cat[1],
                cat[2],
                cat[3],
                Int32(tiles),
                # Inert descriptor. The host chose this split and passed
                # it as arguments, so `use_desc` is zero and the kernel
                # never dereferences this pointer; a real buffer rather
                # than a null keeps the argument well typed everywhere,
                # and it must be a buffer this launch does not already
                # pass, which is why it is `step_dev` and not one of the
                # partition's own.
                self.step_dev.unsafe_ptr(),
                Int32(0),
                grid_dim=blocks,
                block_dim=threads,
            )
            # Same width and the same tiling as the flag pass: the scatter
            # looks its chunk up by `block_idx.x` and `tiles`, and the packed
            # offsets it reads are chunk-relative.
            self.ctx.enqueue_function[_scatter_kernel](
                self.rows_dev.unsafe_ptr(),
                self.scratch_dev.unsafe_ptr(),
                self.offsets_dev.unsafe_ptr(),
                self.block_sums_dev.unsafe_ptr(),
                self.total_dev.unsafe_ptr(),
                Int32(window.begin),
                Int32(n),
                Int32(tiles),
                Int32(blocks),
                # Inert descriptor. The host chose this split and passed
                # it as arguments, so `use_desc` is zero and the kernel
                # never dereferences this pointer; a real buffer rather
                # than a null keeps the argument well typed everywhere,
                # and it must be a buffer this launch does not already
                # pass, which is why it is `step_dev` and not one of the
                # partition's own.
                self.step_dev.unsafe_ptr(),
                Int32(0),
                grid_dim=blocks,
                block_dim=threads,
            )
        self.ctx.enqueue_function[_copy_back_kernel](
            self.rows_dev.unsafe_ptr(),
            self.scratch_dev.unsafe_ptr(),
            Int32(window.begin),
            Int32(n),
            # Inert descriptor; see the flag pass above.
            self.step_dev.unsafe_ptr(),
            Int32(0),
            grid_dim=copy_blocks,
            block_dim=threads,
        )

    def _enqueue_scan_primitives[
        width: Int, bins_origin: MutOrigin
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        window: LeafRange,
        routing: RowRouting,
        n: Int,
        blocks: Int,
        tiles: Int,
        use_desc: Int32 = Int32(0),
    ) raises:
        """The flag pass and the scatter, on the primitive arm, at one
        compile-time threadgroup width.

        Split out only so `enqueue_partition` can name the width once per
        menu entry instead of repeating seventeen kernel arguments per entry.
        `block_dim` is `width` and not `self.block_threads`: the two are equal
        wherever this is reached, and writing the compile-time one is what
        makes a mismatch between the parameter and the launch impossible to
        introduce later. `grid_dim` and `tiles` are the caller's, computed
        once by `_partition_grid` from `self.block_threads`, so the geometry
        and therefore the `block_sums_dev` footprint are exactly what the
        fallback arm uses. Passing the same pair to both kernels rather than
        recomputing it in each is deliberate: the scatter's chunk arithmetic
        has to be the flag pass's chunk arithmetic exactly, or a packed offset
        would be read against the wrong chunk's block sum.
        """
        var cat = routing.cat_bitset
        var default_left = Int32(1) if routing.default_left else Int32(0)
        var is_cat = Int32(1) if routing.is_categorical else Int32(0)
        self.ctx.enqueue_function[_flag_scan_prim_kernel[width]](
            bins,
            self.rows_dev.unsafe_ptr(),
            self.offsets_dev.unsafe_ptr(),
            self.block_sums_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(window.begin),
            Int32(n),
            Int32(routing.feature),
            Int32(routing.threshold_bin),
            Int32(routing.missing_bin),
            default_left,
            is_cat,
            cat[0],
            cat[1],
            cat[2],
            cat[3],
            Int32(tiles),
            self.step_dev.unsafe_ptr(),
            use_desc,
            grid_dim=blocks,
            block_dim=width,
        )
        self.ctx.enqueue_function[_scatter_prim_kernel[width]](
            self.rows_dev.unsafe_ptr(),
            self.scratch_dev.unsafe_ptr(),
            self.offsets_dev.unsafe_ptr(),
            self.block_sums_dev.unsafe_ptr(),
            self.total_dev.unsafe_ptr(),
            Int32(window.begin),
            Int32(n),
            Int32(tiles),
            Int32(blocks),
            self.step_dev.unsafe_ptr(),
            use_desc,
            grid_dim=blocks,
            block_dim=width,
        )

    def enqueue_partition_desc[
        bins_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        max_count: Int,
    ) raises:
        """Enqueue the partition of whatever window the step descriptor names.

        The device-owned counterpart of `enqueue_partition`. It launches the
        same three kernels in the same order on the same arm; what it does not
        do is know which leaf is being split, which split is being applied, or
        how many rows are involved, because on this path a kernel chose all
        three and the host has not read them back. `max_count` is the only
        number the host still supplies and it is an upper bound rather than a
        length: the active row count, which no window can exceed because the
        live windows tile `[0, n_active)`.

        Why an upper bound is enough. The geometry a partition is launched at
        never reaches the permutation. `_scatter_kernel` derives each row's
        destination from the routing flags and from the row's position in the
        range, and from nothing else; the chunking appears only in how the
        prefix is accumulated, and an integer prefix is the same number
        however it is summed. So a grid sized for a longer range than the one
        it finds produces exactly the permutation the exactly-sized grid would
        have produced, and the surplus threadgroups return at their first
        bounds check. That argument is written out at greater length in
        `_flag_scan_kernel` and `_scatter_kernel`.

        What the surplus costs is UNMEASURED. It is bounded above by the
        capped block count times the tile count that covers `max_count`, and
        every one of those blocks that owns nothing exits before it reads a
        row; whether that is cheaper than the host round trip it replaces is
        exactly the thing a benchmark would have to answer and no benchmark
        has been run.

        Nothing is downloaded and nothing synchronizes, which is the point.
        `total_dev` is still written by the scatter and is still the range's
        left count, but the device-owned path does not read it: the commit
        kernel already took both children's counts off the parent histogram's
        integer count plane, exactly as the host path takes them off the
        record. It stays written so that `MOJOTREES_GPU_VERIFY_ROWS` remains
        meaningful for anyone who wants to check the two against each other.
        """
        if max_count < 1:
            raise Error("a descriptor partition needs a positive row bound")
        if max_count > self.n_rows:
            raise Error("the row bound exceeds the row buffer")
        var threads = self.block_threads
        var grid = _partition_grid(
            max_count, threads, self.partition_block_cap
        )
        var blocks = grid[0]
        var tiles = grid[1]

        # The routing arguments are placeholders: `use_desc` is 1, so every
        # kernel below reads the split out of `step_dev` and ignores them. A
        # window of `[0, max_count)` is passed for the same reason, and is
        # deliberately the widest legal one rather than an empty one, so that
        # a wiring mistake that left `use_desc` at zero would partition a real
        # range and be caught by a row check rather than silently do nothing.
        var window = LeafRange(0, max_count)
        var routing = RowRouting.numerical(0, 0, -1, False)

        var scanned = False
        comptime if has_accelerator():
            if self.scan_primitives:
                scanned = True
                if threads == 128:
                    self._enqueue_scan_primitives[128](
                        bins, window, routing, max_count, blocks, tiles,
                        Int32(1),
                    )
                elif threads == 256:
                    self._enqueue_scan_primitives[256](
                        bins, window, routing, max_count, blocks, tiles,
                        Int32(1),
                    )
                elif threads == 512:
                    self._enqueue_scan_primitives[512](
                        bins, window, routing, max_count, blocks, tiles,
                        Int32(1),
                    )
                elif threads == 1024:
                    self._enqueue_scan_primitives[1024](
                        bins, window, routing, max_count, blocks, tiles,
                        Int32(1),
                    )
                else:
                    scanned = False

        if not scanned:
            self.ctx.enqueue_function[_flag_scan_kernel](
                bins,
                self.rows_dev.unsafe_ptr(),
                self.offsets_dev.unsafe_ptr(),
                self.block_sums_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(0),
                Int32(max_count),
                Int32(0),
                Int32(0),
                Int32(-1),
                Int32(0),
                Int32(0),
                UInt64(0),
                UInt64(0),
                UInt64(0),
                UInt64(0),
                Int32(tiles),
                self.step_dev.unsafe_ptr(),
                Int32(1),
                grid_dim=blocks,
                block_dim=threads,
            )
            self.ctx.enqueue_function[_scatter_kernel](
                self.rows_dev.unsafe_ptr(),
                self.scratch_dev.unsafe_ptr(),
                self.offsets_dev.unsafe_ptr(),
                self.block_sums_dev.unsafe_ptr(),
                self.total_dev.unsafe_ptr(),
                Int32(0),
                Int32(max_count),
                Int32(tiles),
                Int32(blocks),
                self.step_dev.unsafe_ptr(),
                Int32(1),
                grid_dim=blocks,
                block_dim=threads,
            )
        # The copy back runs on the capped grid here, not on the uncapped
        # `ceil(count / threads)` one the host arm gives it, because a count
        # this launch does not know cannot size a grid. It is grid-strided, so
        # the same threads walk the range instead of one thread owning one
        # element; see `_copy_back_kernel`.
        self.ctx.enqueue_function[_copy_back_kernel](
            self.rows_dev.unsafe_ptr(),
            self.scratch_dev.unsafe_ptr(),
            Int32(0),
            Int32(max_count),
            self.step_dev.unsafe_ptr(),
            Int32(1),
            grid_dim=blocks,
            block_dim=threads,
        )

    def enqueue_desc_histogram[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        pool_origin: MutOrigin, //
    ](
        mut self,
        pool_slots: Int,
        max_rows: Int,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        pool: MutPointer[Int32, pool_origin],
        n_slots: Int,
        g_scale: Float32,
        h_scale: Float32,
        caps: DeviceCaps,
    ) raises:
        """Accumulate the built child's histogram into the pool slot the step
        descriptor names, subtracting it from the sibling's slot as it goes.

        The device-owned counterpart of `enqueue_range_histogram`, narrowed to
        exactly the one shape a device-owned tree needs: one node, into a
        resident pool slot, with the sibling subtraction folded in.

        **The atomic strategy, always, and this is a decision rather than an
        oversight.** The tiled strategy's partial buffer and its reduction are
        both sized by `n_tiles`, and `n_tiles` is derived from the node's row
        count, which on this path the host does not know. Deriving it from the
        active row count instead would give a shallow node a reduction over
        thousands of empty tiles, which is a cost that grows exactly where the
        tree spends most of its splits. The atomic strategy has no such term:
        its only row-derived quantity is the grid's second dimension, and a
        block whose tile is empty returns at its first comparison
        (`_range_hist_atomic_kernel`).

        The two strategies produce the same histogram, which is what makes
        this a launch decision and not a numeric one. Accumulation is
        fixed-point Int32 in both, integer addition is associative and
        commutative, and each strategy adds the same per-(row, feature)
        quantized value into the same bin; only the order of the additions
        differs, and the order of integer additions cannot change a sum. That
        is the same argument `_range_hist_atomic_kernel` makes for its own
        forty instantiations agreeing with each other.

        **What is not claimed.** That the atomic strategy is as fast here as
        the strategy the host arm would have chosen. It may well not be on a
        large first split, where the tiled path exists precisely because
        atomics contend. No benchmark of this arm has been run and none is
        reported.

        `max_rows` is an upper bound on the built child's row count, and the
        active row count is the bound a caller has. The built child is the
        smaller of the two by `subtraction_builds_left`, so `n_active / 2`
        would also bound it; the looser bound is used because a bound that
        depends on which child won a comparison is a bound that has to be
        re-derived if that comparison ever changes.
        """
        if n_slots < 1 or n_slots > self.n_features:
            raise Error("active feature count out of range")
        if pool_slots < 1:
            raise Error("the resident pool must hold at least one slot")
        if max_rows < 1:
            raise Error("a descriptor histogram needs a positive row bound")

        var hist_size = self.n_features * self.n_bins
        var cells = 3 * hist_size
        var tiling = derive_tiling(
            caps,
            max_rows,
            n_slots,
            self.n_bins,
            STRATEGY_ATOMIC,
            self.part_capacity_unused(),
        )
        var threads = tiling.block_threads

        # The atomic flush is `Atomic.fetch_add` onto whatever the slot
        # already holds, so the slot has to start at zero. Only the one slot
        # the descriptor names is cleared, and only when the step is live;
        # clearing a slot on a dead step would erase a live leaf's histogram.
        var zero_blocks = (cells + threads - 1) // threads
        if zero_blocks > pool_slots * 4:
            zero_blocks = pool_slots * 4
        if zero_blocks < 1:
            zero_blocks = 1
        self.ctx.enqueue_function[_zero_slot_desc_kernel](
            pool,
            Int32(cells),
            self.step_dev.unsafe_ptr(),
            grid_dim=zero_blocks,
            block_dim=threads,
        )

        var use_quant = Int32(0)
        if self.quantized_gradients:
            self._ensure_quantized(grad, hess, g_scale, h_scale)
            use_quant = Int32(1)
        var const_hess = Int32(1) if self.constant_hessian else Int32(0)

        self._enqueue_atomic_family(
            bins,
            grad,
            hess,
            feat_ids,
            pool,
            n_slots,
            hist_size,
            tiling.rows_per_tile,
            # Window, destination slot and subtraction offset all come out of
            # the descriptor, so these three are placeholders the kernel
            # overwrites. The subtraction is always on: a device-owned split
            # always derives one child from the parent's slot.
            0,
            max_rows,
            g_scale,
            h_scale,
            0,
            Int32(1),
            use_quant,
            const_hess,
            tiling.n_tiles,
            threads,
            Int32(1),
        )

    def part_capacity_unused(self) -> Int:
        """Zero, the "no preallocated partial buffer" value `derive_tiling`
        takes.

        The atomic strategy allocates no partials, so the capacity a caller
        would pass to cap the tiled strategy's buffer has nothing to cap. It
        is a named method rather than a bare literal so that the reason is
        attached to the argument at the one call site that passes it.
        """
        return 0

    def download_left_count(mut self) raises -> Int:
        """The left count of the last enqueued partition. Synchronizes."""
        self.ctx.enqueue_copy(
            dst_ptr=self.host_total.unsafe_ptr(), src_buf=self.total_dev
        )
        self.ctx.synchronize()
        return Int(self.host_total.unsafe_ptr().unsafe_load(0))

    def partition[
        bins_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        parent: Int,
        left: Int,
        right: Int,
        routing: RowRouting,
        expected_left: Int = -1,
    ) raises -> Int:
        """Split `parent`'s range into its children's and return the left
        count.

        With `expected_left` given (the grower has it exactly, from the
        parent histogram's integer counts) nothing is downloaded and nothing
        synchronizes, so a split stays fully enqueued. `verify_counts`
        downloads the device count anyway and raises if the two disagree,
        which is the check that the routing rule the host counted with is
        the routing rule the device applied.
        """
        var window = self.ranges.get(parent)
        if expected_left > window.count():
            raise Error("left row count is outside the parent's range")
        self.enqueue_partition(bins, window, routing)

        var n_left: Int
        if window.count() == 0:
            n_left = 0
        elif expected_left < 0:
            n_left = self.download_left_count()
        elif self.verify_counts:
            var device_left = self.download_left_count()
            if device_left != expected_left:
                raise Error(
                    "device left count disagrees with the histogram count"
                )
            n_left = expected_left
        else:
            n_left = expected_left

        _ = self.ranges.split(parent, left, right, n_left)
        return n_left

    # --- The frontier seam -------------------------------------------------
    #
    # Three calls a grower makes instead of maintaining the row ranges and the
    # host-side leaf list separately. `LeafFrontier` holds the same windows
    # this module's `LeafRangeTable` holds, and holding them twice is only
    # safe if something keeps them equal; `check_frontier` is that something,
    # and `apply_commit` is the one state transition that moves both.

    def begin_tree_with(
        mut self,
        mut frontier: LeafFrontier,
        bag: List[Int] = [],
        max_leaves: Int = 0,
        plane: Int = 0,
    ) raises -> Int:
        """Start a tree on the device and on the host frontier at once, and
        return the number of rows it grows on.

        One call rather than two, because the two agree on exactly one number
        and letting a caller pass it twice is letting a caller pass it wrong:
        `n_active` is `len(bag)` when bagging or GOSS selected rows and
        `n_rows` when they did not, and both the root `LeafRange` and the root
        `FrontierLeaf` have to be that window.

        `max_leaves` is the `num_leaves` budget the frontier reports
        completion against and `plane` the class index of a multiclass round;
        neither reaches the device, because neither changes which rows exist.
        A multiclass round grows one tree per class over this same
        permutation, sequentially, so calling this once per class is correct
        and costs one root seed each.
        """
        self.begin_tree(bag)
        var n_active = self.ranges.n_active
        frontier.begin_tree(n_active, max_leaves, plane)
        return n_active

    def check_frontier(self, frontier: LeafFrontier) raises:
        """Every live frontier leaf's window is the range its node owns here.

        The check that makes a disagreement between the two tables loud
        instead of silent. A frontier leaf whose `row_begin`/`row_count` had
        drifted from its `LeafRange` would build a histogram of the wrong
        rows, and nothing about the launch would be out of bounds, so nothing
        else would notice. Linear in the frontier, which is a few hundred
        entries, and meant to be called after a commit or before a batch
        rather than inside a row loop.
        """
        if frontier.n_active != self.ranges.n_active:
            raise Error(
                "frontier and active-row table disagree on the active row"
                " count"
            )
        for i in range(frontier.size()):
            var leaf = frontier.leaf(i)
            var window = self.ranges.get(leaf.node)
            if window.begin != leaf.row_begin:
                raise Error(
                    "frontier leaf's row window does not start where its"
                    " active-row range does"
                )
            if window.count() != leaf.row_count:
                raise Error(
                    "frontier leaf's row count does not match its active-row"
                    " range"
                )

    def partition_commit[
        bins_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        plan: CommitPlan,
    ) raises -> Int:
        """Split the committed leaf's range into its children's, from the
        plan alone.

        The routing rule is built here, from `plan.split` and
        `plan.missing_bin`, through the same `RowRouting.from_split` the host
        reference model uses, so a caller never restates the
        missing/categorical rule and cannot state it differently from the
        kernels. The left count is the plan's `left_count`, which the search
        already counted exactly off the parent histogram's integer count
        plane, so nothing is downloaded and nothing synchronizes unless
        `MOJOTREES_GPU_VERIFY_ROWS=1` asks for the cross-check.

        An empty parent partitions to two empty children: the enqueue returns
        without a launch and the range table records `[b, b)` twice, which is
        what keeps the live ranges tiling `[0, n_active)` whatever the tree
        does.
        """
        var routing = RowRouting.from_split(plan.split, plan.missing_bin)
        return self.partition(
            bins,
            plan.parent_node,
            plan.left_node,
            plan.right_node,
            routing,
            plan.left_count,
        )

    def apply_commit[
        bins_origin: MutOrigin, //
    ](
        mut self,
        mut frontier: LeafFrontier,
        bins: MutPointer[UInt8, bins_origin],
        plan: CommitPlan,
    ) raises -> Int:
        """The whole of one commit: partition on the device, move the
        frontier on the host, and return the left row count.

        This is the single state transition a grower needs per split. It runs
        the device half first so a failure to route leaves the frontier
        untouched rather than half advanced, and it re-checks the two tables
        against each other afterwards, which is cheap next to a partition and
        is what turns a wiring mistake into an error at the split that caused
        it instead of at some later histogram.
        """
        var n_left = self.partition_commit(bins, plan)
        frontier.apply_commit(plan)
        self.check_frontier(frontier)
        return n_left

    def range_tiling(
        self,
        caps: DeviceCaps,
        node: Int,
        n_slots: Int,
        strategy: Int,
        max_partial_cells: Int,
    ) raises -> HistogramTiling:
        """Launch geometry for one node's histogram, derived from the rows
        that node actually owns rather than from the dataset.

        This is the second half of the win: an empty or tiny node gets a
        grid sized for its own rows instead of for `n_rows`. A node with no
        rows still needs a launchable geometry, so it is derived at one row.
        """
        var n = self.ranges.get(node).count()
        if n < 1:
            n = 1
        return derive_tiling(
            caps, n, n_slots, self.n_bins, strategy, max_partial_cells
        )

    def enqueue_range_histogram[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        out_origin: MutOrigin,
        part_origin: MutOrigin, //
    ](
        mut self,
        tiling: HistogramTiling,
        node: Int,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        out_hist: MutPointer[Int32, out_origin],
        partials: MutPointer[Int32, part_origin],
        n_slots: Int,
        g_scale: Float32,
        h_scale: Float32,
        sub_offset: Int = 0,
        subtract: Bool = False,
    ) raises:
        """Enqueue the histogram of `node`'s rows, reading only those rows.

        Everything but the row loop matches `GpuHistogramBuilder.enqueue_leaf`
        (same output layout, same fixed-point accumulation, same two
        strategies), so the histogram is the one that path would have built
        by scanning the whole dataset. The output is zeroed here with a
        kernel rather than `enqueue_memset` because the caller handed in a
        pointer; the zeroing rule is unchanged, only the atomic path and a
        narrowed feature set need it.

        Under `subtract`, the histogram `sub_offset` Int32 words from
        `out_hist` is a second one this build is taken out of as it goes:
        `sub -= out`, folded into whichever kernel writes the result. That is
        the sibling subtraction of a resident frontier done without the
        launch and without the pass over two whole slots that a separate
        `_subtract_slice_kernel` costs, and it is the same exact fixed-point
        arithmetic. A nonzero offset is required, since the two histograms
        are different slots of one pool; `sub_offset` is ignored without
        `subtract`. An empty node subtracts nothing, which is right: a child
        with no rows leaves its sibling holding the parent's histogram
        unchanged.

        Which kernel instantiation runs is resolved here and nowhere else:
        `bin_cap` from the dataset's bin count, the group from
        `feature_group` through `_launch_group`, and the pair through the two
        static trees below. Every instantiation produces the same integers, so
        this resolution is a launch shape and never a numeric decision. The
        pre-quantized gradient buffer is rebuilt here too, once per tree per
        scale, above the strategy branch because every strategy and every
        width now reads it.

        The constant-hessian flag is resolved here for the same reason, from
        `constant_hessian`, and handed to whichever kernels this launch uses.
        On the tiled path it also changes the reduction's grid, because the
        partial layout it selects has two planes per tile rather than three;
        the two kernels get the same flag from this one place, which is what
        keeps them agreeing about the layout. It does not change the zeroing
        rule: the reduction still writes every active feature's slice of all
        three output planes, so the same three conditions that force a
        zeroing pass force it here.
        """
        if n_slots < 1 or n_slots > self.n_features:
            raise Error("active feature count out of range")
        if subtract and sub_offset == 0:
            raise Error(
                "a fused subtraction must name a slot other than the build's"
            )
        var window = self.ranges.get(node)
        var hist_size = self.n_features * self.n_bins
        var threads = tiling.block_threads

        # The reduction of the tiled path writes every active feature's
        # slice, so that path only needs zeroing when some feature is
        # inactive; the atomic path always does, and so does a node with no
        # rows, whose histogram is entirely zeros.
        if (
            tiling.strategy != STRATEGY_TILED
            or n_slots < self.n_features
            or window.count() <= 0
        ):
            var cells = 3 * hist_size
            var zero_blocks = (cells + threads - 1) // threads
            self.ctx.enqueue_function[_zero_int32_kernel](
                out_hist,
                Int32(cells),
                grid_dim=zero_blocks,
                block_dim=threads,
            )
        if window.count() <= 0:
            return

        var do_sub = Int32(1) if subtract else Int32(0)

        # One quantizing pass per tree per scale, ordered before the histogram
        # that reads it by the queue. Every strategy and every group width
        # reads the same buffer now, so this sits above the dispatch rather
        # than inside one arm of it.
        var use_quant = Int32(0)
        if self.quantized_gradients:
            self._ensure_quantized(grad, hess, g_scale, h_scale)
            use_quant = Int32(1)

        var const_hess = Int32(1) if self.constant_hessian else Int32(0)
        var hist_planes = 2 if self.constant_hessian else 3

        if tiling.strategy == STRATEGY_TILED:
            self._enqueue_partial_family(
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                tiling.rows_per_tile,
                window.begin,
                window.count(),
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                tiling.n_tiles,
                threads,
            )
            var n_cells = hist_planes * n_slots * self.n_bins
            var blocks = (n_cells + threads - 1) // threads
            self.ctx.enqueue_function[_range_reduce_kernel](
                partials,
                feat_ids,
                out_hist,
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(tiling.n_tiles),
                Int32(sub_offset),
                do_sub,
                h_scale,
                const_hess,
                grid_dim=blocks,
                block_dim=threads,
            )
        else:
            self._enqueue_atomic_family(
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                tiling.rows_per_tile,
                window.begin,
                window.count(),
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                tiling.n_tiles,
                threads,
                # No descriptor: this arm's window and destination came
                # from the host, so the kernel is told to ignore the
                # descriptor buffer it is handed. See `step_dev`.
                Int32(0),
            )

    def _launch_group(self, n_slots: Int) -> Int:
        """The group width this launch runs at.

        `feature_group`, except that a single active feature slot runs at 1
        whatever was asked for. That is the `n_slots >= 2` guard the
        hand-written variants carried, kept verbatim so the grids this
        dispatch produces at widths 1, 2, and 4 are the grids the launch site
        produced before the family was parameterized. Nothing else is
        narrowed: a request stays the request.
        """
        if n_slots < 2:
            return 1
        return self.feature_group

    def _enqueue_partial_family[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        part_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        partials: MutPointer[Int32, part_origin],
        n_slots: Int,
        rows_per_tile: Int,
        begin: Int,
        count: Int,
        g_scale: Float32,
        h_scale: Float32,
        use_quant: Int32,
        const_hess: Int32,
        n_tiles: Int,
        threads: Int,
    ) raises:
        """Pick the `GROUP` rung, then hand off to the rung's own dispatch.

        The (GROUP, BIN_CAP) matrix is resolved as a static tree in two
        halves, five ways here and four ways in `_enqueue_partial_at`, rather
        than as twenty branches written out. Both halves are exhaustive over
        their ladder, so the resolution cannot fall through to a width the
        launch was not sized for.
        """
        var group = self._launch_group(n_slots)
        if group >= 16:
            self._enqueue_partial_at[16](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )
        elif group >= 8:
            self._enqueue_partial_at[8](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )
        elif group >= 4:
            self._enqueue_partial_at[4](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )
        elif group >= 2:
            self._enqueue_partial_at[2](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )
        else:
            self._enqueue_partial_at[1](
                bins,
                grad,
                hess,
                feat_ids,
                partials,
                n_slots,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                n_tiles,
                threads,
            )

    def _enqueue_partial_at[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        part_origin: MutOrigin, //,
        GROUP: Int,
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        partials: MutPointer[Int32, part_origin],
        n_slots: Int,
        rows_per_tile: Int,
        begin: Int,
        count: Int,
        g_scale: Float32,
        h_scale: Float32,
        use_quant: Int32,
        const_hess: Int32,
        n_tiles: Int,
        threads: Int,
    ) raises:
        """The tiled launch at one group width, over the four bin capacities.

        `blocks` is `ceil(n_slots / GROUP)` at every rung, which reproduces
        the grids the hand-written variants launched exactly: `n_slots` at
        one, `(n_slots + 1) // 2` at the pairing, `(n_slots + 3) // 4` at the
        quad. A tail block owns the slots that remain and the kernel handles
        it, so nothing here rounds the slot count up.

        `unroll` is the row-walk arm (`set_row_unroll`), read from the field
        here rather than threaded through the family dispatch above, because
        it changes no grid, no threadgroup footprint, and no histogram.
        """
        var blocks = (n_slots + GROUP - 1) // GROUP
        var unroll = Int32(1) if self.row_unroll else Int32(0)
        if self.bin_cap <= 32:
            self.ctx.enqueue_function[_range_hist_partial_kernel[GROUP, 32]](
                bins,
                self.rows_dev.unsafe_ptr(),
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                feat_ids,
                partials,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                unroll,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        elif self.bin_cap <= 64:
            self.ctx.enqueue_function[_range_hist_partial_kernel[GROUP, 64]](
                bins,
                self.rows_dev.unsafe_ptr(),
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                feat_ids,
                partials,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                unroll,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        elif self.bin_cap <= 128:
            self.ctx.enqueue_function[_range_hist_partial_kernel[GROUP, 128]](
                bins,
                self.rows_dev.unsafe_ptr(),
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                feat_ids,
                partials,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                unroll,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        else:
            self.ctx.enqueue_function[_range_hist_partial_kernel[GROUP, 256]](
                bins,
                self.rows_dev.unsafe_ptr(),
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                feat_ids,
                partials,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                use_quant,
                const_hess,
                unroll,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )

    def _enqueue_atomic_family[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        out_origin: MutOrigin, //
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        out_hist: MutPointer[Int32, out_origin],
        n_slots: Int,
        hist_size: Int,
        rows_per_tile: Int,
        begin: Int,
        count: Int,
        g_scale: Float32,
        h_scale: Float32,
        sub_offset: Int,
        do_sub: Int32,
        use_quant: Int32,
        const_hess: Int32,
        n_tiles: Int,
        threads: Int,
        use_desc: Int32,
    ) raises:
        """The atomic twin of `_enqueue_partial_family`, same static tree."""
        var group = self._launch_group(n_slots)
        if group >= 16:
            self._enqueue_atomic_at[16](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )
        elif group >= 8:
            self._enqueue_atomic_at[8](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )
        elif group >= 4:
            self._enqueue_atomic_at[4](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )
        elif group >= 2:
            self._enqueue_atomic_at[2](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )
        else:
            self._enqueue_atomic_at[1](
                bins,
                grad,
                hess,
                feat_ids,
                out_hist,
                n_slots,
                hist_size,
                rows_per_tile,
                begin,
                count,
                g_scale,
                h_scale,
                sub_offset,
                do_sub,
                use_quant,
                const_hess,
                n_tiles,
                threads,
                use_desc,
            )

    def _enqueue_atomic_at[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin,
        feat_origin: MutOrigin,
        out_origin: MutOrigin, //,
        GROUP: Int,
    ](
        mut self,
        bins: MutPointer[UInt8, bins_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        feat_ids: MutPointer[Int32, feat_origin],
        out_hist: MutPointer[Int32, out_origin],
        n_slots: Int,
        hist_size: Int,
        rows_per_tile: Int,
        begin: Int,
        count: Int,
        g_scale: Float32,
        h_scale: Float32,
        sub_offset: Int,
        do_sub: Int32,
        use_quant: Int32,
        const_hess: Int32,
        n_tiles: Int,
        threads: Int,
        use_desc: Int32,
    ) raises:
        """The atomic launch at one group width, over the four bin
        capacities.

        `unroll` is the row-walk arm (`set_row_unroll`), read from the field
        here rather than threaded through the family dispatch above, because
        it changes no grid, no threadgroup footprint, and no histogram."""
        var blocks = (n_slots + GROUP - 1) // GROUP
        var unroll = Int32(1) if self.row_unroll else Int32(0)
        if self.bin_cap <= 32:
            self.ctx.enqueue_function[_range_hist_atomic_kernel[GROUP, 32]](
                bins,
                self.rows_dev.unsafe_ptr(),
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                feat_ids,
                out_hist,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                Int32(sub_offset),
                do_sub,
                use_quant,
                const_hess,
                self.step_dev.unsafe_ptr(),
                use_desc,
                unroll,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        elif self.bin_cap <= 64:
            self.ctx.enqueue_function[_range_hist_atomic_kernel[GROUP, 64]](
                bins,
                self.rows_dev.unsafe_ptr(),
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                feat_ids,
                out_hist,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                Int32(sub_offset),
                do_sub,
                use_quant,
                const_hess,
                self.step_dev.unsafe_ptr(),
                use_desc,
                unroll,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        elif self.bin_cap <= 128:
            self.ctx.enqueue_function[_range_hist_atomic_kernel[GROUP, 128]](
                bins,
                self.rows_dev.unsafe_ptr(),
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                feat_ids,
                out_hist,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                Int32(sub_offset),
                do_sub,
                use_quant,
                const_hess,
                self.step_dev.unsafe_ptr(),
                use_desc,
                unroll,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )
        else:
            self.ctx.enqueue_function[_range_hist_atomic_kernel[GROUP, 256]](
                bins,
                self.rows_dev.unsafe_ptr(),
                grad,
                hess,
                self.gq_dev.unsafe_ptr(),
                feat_ids,
                out_hist,
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(rows_per_tile),
                Int32(begin),
                Int32(count),
                g_scale,
                h_scale,
                Int32(sub_offset),
                do_sub,
                use_quant,
                const_hess,
                self.step_dev.unsafe_ptr(),
                use_desc,
                unroll,
                grid_dim=(blocks, n_tiles),
                block_dim=threads,
            )

    def _ensure_quantized[
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
        g_scale: Float32,
        h_scale: Float32,
    ) raises:
        """Rebuild the interleaved quantized gradient buffer unless it was
        built for this tree at these scales already. One streaming launch
        over `n_rows`, ordered before the histogram that reads it by the
        queue."""
        if (
            self.quant_valid
            and self.quant_g_scale == g_scale
            and self.quant_h_scale == h_scale
        ):
            return
        var threads = self.block_threads
        var blocks = (self.n_rows + threads - 1) // threads
        self.ctx.enqueue_function[_quantize_grad_hess_kernel](
            grad,
            hess,
            self.gq_dev.unsafe_ptr(),
            Int32(self.n_rows),
            g_scale,
            h_scale,
            grid_dim=blocks,
            block_dim=threads,
        )
        self.quant_valid = True
        self.quant_g_scale = g_scale
        self.quant_h_scale = h_scale

    def download_rows(mut self) raises -> List[Int32]:
        """The whole active-row buffer, host side. Synchronizes, and is for
        tests and debugging: training never needs the permutation on the
        host."""
        self.ctx.enqueue_copy(
            dst_ptr=self.host_rows.unsafe_ptr(), src_buf=self.rows_dev
        )
        self.ctx.synchronize()
        var out = List[Int32](capacity=self.n_rows)
        var src = self.host_rows.unsafe_ptr()
        for r in range(self.n_rows):
            out.append(src.unsafe_load(r))
        return out^

    def download_range(mut self, node: Int) raises -> List[Int]:
        """One node's rows, in compacted order. Tests compare this against
        the CPU grower's row list for the same node."""
        var window = self.ranges.get(node)
        var all_rows = self.download_rows()
        var out = List[Int](capacity=window.count())
        for j in range(window.begin, window.end):
            out.append(Int(all_rows[j]))
        return out^
