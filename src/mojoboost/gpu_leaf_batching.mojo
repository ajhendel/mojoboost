"""Several leaves' histograms in one GPU launch.

`GpuActiveRows.enqueue_range_histogram` builds one node's histogram per
launch, over a grid of `(n_active_features, n_tiles)` threadgroups sized from
that node's own row count. That is right for the root and for the top of a
tree and wrong for the bottom of one: a leaf holding 400 rows out of a
million produces one row tile, so its launch creates `n_slots` threadgroups
against a device that wants hundreds, and a whole boosting round's tail is
made of such leaves. This module builds several leaves in one launch instead,
so the grid is filled by the batch rather than by any single leaf in it.

The packed grid
---------------
A batch is a list of `LeafWorkItem`s (see gpu_frontier.mojo). Each item
carries its own row window, its own rows-per-tile, and therefore its own tile
count, and the tiles of all the items are concatenated into one flat tile
axis. The launch is two dimensional:

    grid.x = n_slots            the active feature slot, as today
    grid.y = total_tiles        sum over the batch of each item's tiles

and a threadgroup finds which item it belongs to by binary searching the
items' tile offsets, which the host has already prefix summed. Two properties
follow, and both are the point:

- **No wasted threadgroups.** A grid that carried the leaf on a third axis
  would have to give every leaf the largest leaf's tile count, and the
  threadgroups past a small leaf's end would launch only to exit. A packed
  tile axis gives each leaf exactly the tiles it asked for, so a frontier of
  one huge leaf and thirty tiny ones costs exactly the tiles those thirty one
  leaves need. Highly unbalanced frontiers are the normal case in leaf-wise
  growth, so this is not a corner.
- **Two grid dimensions, not three.** The row-tile axis is already `grid.y`
  in the shipping kernels and the portable ceiling on it is known
  (`MAX_GRID_DIM_Y`). Nothing here needs a `grid.z` whose portable limit
  across Metal, CUDA, and HIP this project has not established.

The cost is a binary search per threadgroup over at most `max_items` entries,
run once by thread 0 into threadgroup memory, behind the barrier the kernel
already pays to zero its shared histogram. At 32 items that is five Int32
loads per threadgroup against a row loop of hundreds.

What batching does not change
-----------------------------
Every item writes to its own output slice, and no accumulation ever crosses
an item. Within an item, accumulation is the same fixed-point Int32 the
single-leaf kernels use, the tiled reduction sums an item's own tiles in
ascending order, and the atomic strategy folds shared partials into the
item's own slice with integer atomics. So a leaf's histogram is bit-identical
whether it was built alone or in a batch of thirty, and the batch size is a
launch decision that no result can observe. That is the invariant the whole
lane rests on.

Where the leaves come from
--------------------------
The primitives here are worth exactly as much as the grower above them can
feed them, and on the trainer's default path today that is one leaf per
commit: `grow_tree_gpu` builds the smaller child and derives the sibling by
subtraction, so a batch of one is all there is and this module is a no-op.
`gpu_frontier.leaves_per_launch` computes the number for each of the four
growers, and `handoffs/algorithm_22_leaf_batching.md` states the consequence
plainly rather than burying it: batching is a change to the *grower*, and
these kernels are the half of it that can be built and reasoned about first.
The three growers that can offer more than one leaf are the device-search
path (two children per commit), a speculative frontier (up to two per
speculated commit, and speculation is semantically free, see
gpu_frontier.mojo), and level-wise growth (a whole level, a separate lane).

Histogram subtraction, on the device
------------------------------------
`enqueue_subtract` derives a sibling's histogram from the parent's and the
built child's, in the fixed-point buffer where both already live. Integer
subtraction is exact, so the derived histogram is exactly the one the built
histogram would have been, and the parent no longer has to be downloaded for
a host-side `subtract_histogram`. Two conditions make it valid, and both are
checked rather than assumed: the two histograms must have been accumulated
under the same fixed-point scales (the scales are fixed per round and per
class, so this holds within a tree and not across rounds), and under the same
active feature set (a feature absent from one and present in the other would
leave a nonzero slice being subtracted from a zero one). `HistogramSlotPool`
carries the stamp that encodes both.

Memory
------
The three buffers this module sizes, symbolically, with `F = n_features`,
`B = n_bins`, `S = n_slots <= F`, `K` items in a batch, `P` pool slots:

    output      P * 3 * F * B * 4 bytes      one full-width histogram per slot
    partials    T * 3 * S * B * 4 bytes      T = total tiles in the batch
    item tables K * (ITEM_WORDS + 2) * 4 + K * F * 4 bytes

The output pool is what grows with the frontier, and it is why the pool is
bounded and evictable: a leaf whose slot was taken back can always be rebuilt
from its row range, so eviction costs a rebuild and never a wrong answer. The
partial buffer is what grows with the batch, and it is the term that can make
batching worse rather than better: a batch wide enough to fill the device can
ask for more tiles than the partial budget holds, at which point the planner
either gives the batch fewer tiles than it wanted or resolves it to the
atomic strategy, which needs no partial buffer at all. `plan_batch` reports
which of those happened; it does not silently do either.

The frontier seam
-----------------
A grower does not assemble a batch by hand. Four calls do it, and each one may
decline rather than force:

    admit_frontier_batch(frontier, pool, storage, features)
    assign_batch_slots(frontier, pool, admission, stamp)
    plan_frontier_batch(caps, frontier, admission, g_scale, h_scale, ...)
    batcher.enqueue_frontier_batch(plan, bins, rows, grad, hess)

`admit_frontier_batch` is the conservative switch and it declines for five
distinct reasons (`BATCH_*`), every one of which leaves the caller on the
established one-leaf-per-launch path that
`GpuActiveRows.enqueue_range_histogram` already implements. Two of them are
worth naming here: a build whose batched kernels are not compiled in and
validated (`KernelFeatures.batched_leaf_kernel`), and a bin matrix that is not
the one-`UInt8`-per-cell layout these kernels index
(`BinStorageDescriptor.is_dense_feature_major_u8`). Neither is a performance
judgement; both are facts about what can be run at all.

`enqueue_frontier_batch` takes the `GpuActiveRows` itself rather than a row
pointer, so a batch reads the permutation the tree is actually growing on and
every item's window is checked against that tree's live prefix before a kernel
starts. Under bagging the prefix is the bag, and a window outside it would be
inside the buffer and outside the dataset the tree is being fit to, which
nothing downstream could detect.

Nothing in the sequence allocates or synchronizes per leaf: slots come from a
pool sized once, the item table is two staged copies per batch, and the launch
is at most three kernels whatever the batch's width. `download_slots` closes
the loop on the way home, mapping the output pool once for a whole batch
rather than once per leaf.

Scope and coupling
------------------
This module owns kernels, buffers, and the launch planning for batched
histogram construction. It does not own the frontier (gpu_frontier.mojo), the
row permutation (`GpuActiveRows`), the gradients, or the split search: the
binned matrix and the gradient planes arrive as pointers and the row
permutation arrives as the struct that owns it, exactly as
`enqueue_range_histogram` takes them, so nothing is uploaded or copied twice.
`DeviceCaps`, `derive_block_threads`, the `STRATEGY_*` constants, the grid and
tile bounds, and `BYTES_PER_PARTIAL_CELL` are imported from `gpu_tiling.mojo`
rather than restated, and `MAX_BINS`/`PLANES_PER_HISTOGRAM` from
`gpu_histogram_specializations.mojo`, so every one of those numbers means one
thing across the backend. The per-item tile distribution is local, because a
shared tile budget across several leaves is a shape `derive_tiling` has no way
to express.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import round
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .gpu_active_rows import MAX_ROWS, GpuActiveRows, LeafRange
from .gpu_frontier import NO_SLOT, LeafFrontier, LeafStats, LeafWorkItem
from .gpu_histogram_specializations import (
    MAX_BINS,
    PLANES_PER_HISTOGRAM,
    BinStorageDescriptor,
    KernelFeatures,
)
from .gpu_tiling import (
    BYTES_PER_PARTIAL_CELL,
    MAX_GRID_DIM_Y,
    MIN_ROWS_PER_TILE_BIN_FACTOR,
    MIN_ROWS_PER_TILE_THREAD_FACTOR,
    STRATEGY_ATOMIC,
    STRATEGY_AUTO,
    STRATEGY_TILED,
    TARGET_BLOCKS_PER_SM,
    DeviceCaps,
    derive_block_threads,
)

# Gradient, hessian, and count planes, in that order, in every histogram.
# Imported rather than restated: it is the same three planes
# `gpu_histogram_specializations` sizes a shared histogram with and
# `histogram_gpu` lays its output buffer out as.
comptime N_PLANES = PLANES_PER_HISTOGRAM

# Items one launch may cover. The binary search is logarithmic in this and
# the item table is staged in full on every launch, so the bound keeps both
# costs flat; it is not a claim about where batching stops paying.
comptime DEFAULT_MAX_ITEMS = 32

# --- Item table layout ---------------------------------------------------
#
# One row of Int32 per item, staged as a single copy per launch. The kernels
# read it and nothing else about the batch's shape, so adding an item is a
# host-side table write and never a new argument.

comptime ITEM_BEGIN = 0
"""First slot of this item's window in the active-row buffer. Absolute, so a
caller holding several row permutations at once (one per class, say) puts the
permutation's base into this offset and needs no extra word."""

comptime ITEM_COUNT = 1
comptime ITEM_ROWS_PER_TILE = 2
comptime ITEM_TILE_BEGIN = 3
"""This item's first tile on the packed tile axis: the exclusive prefix sum
of the batch's tile counts, and the key the kernels binary search."""

comptime ITEM_TILES = 4
comptime ITEM_OUT = 5
"""Histogram slot this item's result is written to."""

comptime ITEM_PLANE = 6
"""Gradient plane this item reads, as a multiple of `n_rows`."""

comptime ITEM_WORDS = 8
"""Padded to eight so an item row is a round number of words."""

# Per-item fixed-point scales, Float32, in the kernels' own precision. Per
# item rather than per launch because a multiclass round holds one scale pair
# per class, and a batch may span classes.
comptime SCALE_G = 0
comptime SCALE_H = 1
comptime SCALE_WORDS = 2

# --- Planner verdicts ----------------------------------------------------

comptime VERDICT_UNKNOWN = 0
"""The plan is legal and neither structural fact below applies. Whether it is
faster is a measurement, and this module does not guess at one."""

comptime VERDICT_SINGLE_FILLS = 1
"""Some item on its own already reaches the block target, so the device was
not underfilled and batching it with others buys no occupancy."""

comptime VERDICT_OCCUPANCY_GAIN = 2
"""Every item alone leaves the device underfilled and the batch does not.
This is the case the module exists for."""

comptime VERDICT_PARTIAL_BOUND = 3
"""The tiled partial buffer could not hold the tiles the batch wanted. The
plan either kept fewer tiles or fell back to the atomic strategy, and either
way the batch is paying for its width."""


def verdict_name(verdict: Int) -> String:
    if verdict == VERDICT_SINGLE_FILLS:
        return String("single_fills")
    if verdict == VERDICT_OCCUPANCY_GAIN:
        return String("occupancy_gain")
    if verdict == VERDICT_PARTIAL_BOUND:
        return String("partial_bound")
    return String("unknown")


def _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b


# --- Kernels --------------------------------------------------------------


def _batch_zero_kernel(
    out_hist: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    n_items: Int32,
    hist_size: Int32,
):
    """Zero every output slice this batch writes.

    Needed for the same reasons the single-leaf path needs it: the atomic
    strategy accumulates into whatever is there, a narrowed feature set
    leaves slices the reduction never writes, and an empty leaf's histogram
    is all zeros with no kernel to produce them. Indexed through the item
    table so slices belonging to other leaves are untouched.
    """
    var cells = N_PLANES * Int(hist_size)
    var i = global_idx.x
    if i < Int(n_items) * cells:
        var k = i // cells
        var r = i - k * cells
        var slot = Int(items[unsafe_offset = k * ITEM_WORDS + ITEM_OUT][0])
        out_hist[unsafe_offset = slot * cells + r] = Int32(0)


@always_inline
def _item_for_tile(
    items: MutPointer[Int32, MutAnyOrigin], n_items: Int, tile: Int
) -> Int:
    """The item owning global tile `tile`: the largest `k` whose
    `ITEM_TILE_BEGIN` is at most `tile`.

    The offsets are ascending by construction (they are an exclusive prefix
    sum of positive tile counts), so this is an ordinary upper-bound search
    and it terminates in `ceil(log2(n_items))` steps. Every thread of a
    threadgroup shares `block_idx.y`, so every thread that runs it computes
    the same answer; the kernels below run it on one thread anyway and
    publish the result through threadgroup memory.
    """
    var lo = 0
    var hi = n_items - 1
    while lo < hi:
        var mid = (lo + hi + 1) >> 1
        var begin = Int(
            items[unsafe_offset = mid * ITEM_WORDS + ITEM_TILE_BEGIN][0]
        )
        if begin <= tile:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _batch_hist_partial_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    scales: MutPointer[Float32, MutAnyOrigin],
    partials: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_items: Int32,
    n_slots: Int32,
    n_bins: Int32,
    feat_stride: Int32,
):
    """One (feature slot, packed tile) partial histogram, written to its own
    slot of the global partial buffer.

    The row loop is `_range_hist_partial_kernel`'s, unchanged: gather the
    row through the active-row permutation, read its bin, quantize gradient
    and hessian into Int32, accumulate in threadgroup memory, flush once.
    What is new is that the tile index selects the leaf as well as the rows,
    and that the feature and the scales are read per item, so a batch may
    span leaves with different feature sets and different fixed-point scales
    without the kernel branching on either.

    No atomics on global memory and one write per cell, so the partial buffer
    is written exactly once per (tile, slot, bin) and the reduction below is
    the only reader.
    """
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var ns = Int(n_slots)
    var slot = Int(block_idx.x)
    var g_tile = Int(block_idx.y)

    var meta = stack_allocation[
        8, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    # One thread resolves the tile to its item, behind the barrier the shared
    # histogram already needs.
    if tid == 0:
        var k = _item_for_tile(items, Int(n_items), g_tile)
        var base = k * ITEM_WORDS
        meta[unsafe_offset=0] = Int32(k)
        meta[unsafe_offset=1] = items[unsafe_offset = base + ITEM_BEGIN][0]
        meta[unsafe_offset=2] = items[unsafe_offset = base + ITEM_COUNT][0]
        meta[unsafe_offset=3] = items[
            unsafe_offset = base + ITEM_ROWS_PER_TILE
        ][0]
        meta[unsafe_offset=4] = Int32(
            g_tile - Int(items[unsafe_offset = base + ITEM_TILE_BEGIN][0])
        )
        meta[unsafe_offset=5] = items[unsafe_offset = base + ITEM_PLANE][0]

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var k = Int(meta[unsafe_offset=0][0])
    var begin = Int(meta[unsafe_offset=1][0])
    var count = Int(meta[unsafe_offset=2][0])
    var rows_per_tile = Int(meta[unsafe_offset=3][0])
    var t = Int(meta[unsafe_offset=4][0])
    var plane_base = Int(meta[unsafe_offset=5][0]) * nr

    var f = Int(feat_ids[unsafe_offset = k * Int(feat_stride) + slot][0])
    var g_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_G][0]
    var h_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_H][0]

    var tile_begin = t * rows_per_tile
    var tile_end = tile_begin + rows_per_tile
    if tile_end > count:
        tile_end = count

    var col = f * nr
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = begin + j][0])
        var bin = Int(bins[unsafe_offset = col + r])
        var gq = Int32(round(grad[unsafe_offset = plane_base + r][0] * g_scale))
        var hq = Int32(round(hess[unsafe_offset = plane_base + r][0] * h_scale))
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gq)
        _ = Atomic.fetch_add(sh.unsafe_offset(bin), hq)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        j += block_dim.x
    barrier()

    var plane = ns * nb
    var out_base = g_tile * N_PLANES * plane + slot * nb
    b = tid
    while b < nb:
        partials[unsafe_offset = out_base + b] = sg[unsafe_offset=b][0]
        partials[unsafe_offset = out_base + plane + b] = sh[unsafe_offset=b][0]
        partials[unsafe_offset = out_base + 2 * plane + b] = sc[
            unsafe_offset=b
        ][0]
        b += block_dim.x


def _batch_hist_atomic_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    scales: MutPointer[Float32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_items: Int32,
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    feat_stride: Int32,
):
    """The same batched accumulation, folding each partial straight into its
    item's output slice with global integer atomics.

    The preserved fallback, for the same reason `STRATEGY_ATOMIC` is
    preserved in the single-leaf path: it needs no partial buffer, so it is
    the strategy a batch resolves to when the partial budget cannot hold the
    tiles the batch wants. Contention is no worse than the single-leaf
    kernel's, because tiles of different items fold into different slices;
    within an item it is exactly the same. Integer atomics make the result
    order-independent, so this path is bit-identical to the tiled one.
    """
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)
    var slot = Int(block_idx.x)
    var g_tile = Int(block_idx.y)

    var meta = stack_allocation[
        8, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    if tid == 0:
        var k = _item_for_tile(items, Int(n_items), g_tile)
        var base = k * ITEM_WORDS
        meta[unsafe_offset=0] = Int32(k)
        meta[unsafe_offset=1] = items[unsafe_offset = base + ITEM_BEGIN][0]
        meta[unsafe_offset=2] = items[unsafe_offset = base + ITEM_COUNT][0]
        meta[unsafe_offset=3] = items[
            unsafe_offset = base + ITEM_ROWS_PER_TILE
        ][0]
        meta[unsafe_offset=4] = Int32(
            g_tile - Int(items[unsafe_offset = base + ITEM_TILE_BEGIN][0])
        )
        meta[unsafe_offset=5] = items[unsafe_offset = base + ITEM_PLANE][0]
        meta[unsafe_offset=6] = items[unsafe_offset = base + ITEM_OUT][0]

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var k = Int(meta[unsafe_offset=0][0])
    var begin = Int(meta[unsafe_offset=1][0])
    var count = Int(meta[unsafe_offset=2][0])
    var rows_per_tile = Int(meta[unsafe_offset=3][0])
    var t = Int(meta[unsafe_offset=4][0])
    var plane_base = Int(meta[unsafe_offset=5][0]) * nr
    var out_slot = Int(meta[unsafe_offset=6][0])

    var f = Int(feat_ids[unsafe_offset = k * Int(feat_stride) + slot][0])
    var g_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_G][0]
    var h_scale = scales[unsafe_offset = k * SCALE_WORDS + SCALE_H][0]

    var tile_begin = t * rows_per_tile
    var tile_end = tile_begin + rows_per_tile
    if tile_end > count:
        tile_end = count

    var col = f * nr
    var j = tile_begin + tid
    while j < tile_end:
        var r = Int(rows[unsafe_offset = begin + j][0])
        var bin = Int(bins[unsafe_offset = col + r])
        var gq = Int32(round(grad[unsafe_offset = plane_base + r][0] * g_scale))
        var hq = Int32(round(hess[unsafe_offset = plane_base + r][0] * h_scale))
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gq)
        _ = Atomic.fetch_add(sh.unsafe_offset(bin), hq)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        j += block_dim.x
    barrier()

    var slice_base = out_slot * N_PLANES * hs + f * nb
    b = tid
    while b < nb:
        if sc[unsafe_offset=b][0] != 0:
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + b), sg[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + hs + b),
                sh[unsafe_offset=b][0],
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(slice_base + 2 * hs + b),
                sc[unsafe_offset=b][0],
            )
        b += block_dim.x


def _batch_reduce_kernel(
    partials: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    items: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_items: Int32,
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    feat_stride: Int32,
):
    """Sum each item's own tiles into its own output slice.

    One thread per output cell of the batch, decomposed into
    `(item, plane, slot, bin)`. An item's tiles are contiguous on the packed
    axis, so its reduction is a walk of `ITEM_TILES` strided reads starting
    at `ITEM_TILE_BEGIN`, in ascending tile order. The order is fixed and the
    values are exact integers, so the sum is reproducible and identical to
    the single-leaf reduction's.
    """
    var ns = Int(n_slots)
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var plane = ns * nb
    var per_item = N_PLANES * plane
    var i = global_idx.x
    if i >= Int(n_items) * per_item:
        return

    var k = i // per_item
    var rem = i - k * per_item
    var p = rem // plane
    var rem2 = rem - p * plane
    var slot = rem2 // nb
    var b = rem2 - slot * nb

    var base = k * ITEM_WORDS
    var tile_begin = Int(items[unsafe_offset = base + ITEM_TILE_BEGIN][0])
    var n_tiles = Int(items[unsafe_offset = base + ITEM_TILES][0])
    var out_slot = Int(items[unsafe_offset = base + ITEM_OUT][0])
    var f = Int(feat_ids[unsafe_offset = k * Int(feat_stride) + slot][0])

    var acc = Int32(0)
    var off = tile_begin * per_item + p * plane + slot * nb + b
    for _ in range(n_tiles):
        acc += partials[unsafe_offset=off][0]
        off += per_item
    out_hist[
        unsafe_offset = out_slot * N_PLANES * hs + p * hs + f * nb + b
    ] = acc


def _subtract_slice_kernel(
    out_hist: MutPointer[Int32, MutAnyOrigin],
    parent_slot: Int32,
    child_slot: Int32,
    dst_slot: Int32,
    cells: Int32,
):
    """`dst = parent - child`, elementwise over one histogram slot.

    The sibling subtraction trick, done where the histograms already live.
    Both operands are fixed-point Int32 accumulated under the same scales, so
    the difference is exact rather than exact-to-Float32, and the parent
    never has to cross to the host to be subtracted there.

    Each thread reads one cell of each of the three slices at the same
    offset and writes one cell, so `dst_slot` may alias either operand: the
    read of an aliased slice happens in the same thread that overwrites it,
    and no thread reads a cell another thread writes. Deriving in place over
    the parent's slot is therefore legal, and is what lets a frontier hold
    one slot per live leaf rather than one per node ever created.
    """
    var i = global_idx.x
    if i < Int(cells):
        var n = Int(cells)
        var a = out_hist[unsafe_offset = Int(parent_slot) * n + i][0]
        var b = out_hist[unsafe_offset = Int(child_slot) * n + i][0]
        out_hist[unsafe_offset = Int(dst_slot) * n + i] = a - b


# --- Batch planning -------------------------------------------------------


@fieldwise_init
struct BatchItemPlan(Copyable, Movable):
    """One item's resolved launch geometry."""

    var row_begin: Int
    var row_count: Int
    var out_slot: Int
    var plane: Int
    var rows_per_tile: Int
    var n_tiles: Int
    var tile_begin: Int
    var g_scale: Float32
    var h_scale: Float32


struct BatchPlan(Movable):
    """A resolved batched launch.

    Holds everything a launch needs and nothing about the frontier it came
    from, so a plan can be costed, compared against the serial plan, and
    checked for its memory bounds without a device present.
    """

    var items: List[BatchItemPlan]
    var strategy: Int
    var block_threads: Int
    var n_slots: Int
    var n_bins: Int
    var total_tiles: Int
    var partial_cells: Int
    """`total_tiles * n_slots * n_bins`, one cell carrying all three planes,
    so the buffer holds `3 * partial_cells` Int32. Zero under the atomic
    strategy."""

    var verdict: Int
    var target_blocks: Int

    def __init__(
        out self,
        var items: List[BatchItemPlan],
        strategy: Int,
        block_threads: Int,
        n_slots: Int,
        n_bins: Int,
        total_tiles: Int,
        partial_cells: Int,
        verdict: Int,
        target_blocks: Int,
    ):
        self.items = items^
        self.strategy = strategy
        self.block_threads = block_threads
        self.n_slots = n_slots
        self.n_bins = n_bins
        self.total_tiles = total_tiles
        self.partial_cells = partial_cells
        self.verdict = verdict
        self.target_blocks = target_blocks

    def n_items(self) -> Int:
        return len(self.items)

    def blocks(self) -> Int:
        """Threadgroups the histogram launch creates."""
        return self.n_slots * self.total_tiles

    def fills_device(self) -> Bool:
        return self.blocks() >= self.target_blocks


def _min_rows_per_tile(n_bins: Int, block_threads: Int) -> Int:
    var floor = MIN_ROWS_PER_TILE_BIN_FACTOR * n_bins
    var by_threads = MIN_ROWS_PER_TILE_THREAD_FACTOR * block_threads
    if by_threads > floor:
        floor = by_threads
    return floor


def plan_batch(
    caps: DeviceCaps,
    items: List[LeafWorkItem],
    scales: List[Float32],
    n_slots: Int,
    n_bins: Int,
    strategy: Int = STRATEGY_AUTO,
    max_partial_cells: Int = 0,
    max_items: Int = DEFAULT_MAX_ITEMS,
) raises -> BatchPlan:
    """Resolve one batched launch. Pure host arithmetic, no device needed.

    Tiles are handed out in three passes, all in exact integers so the plan
    is reproducible.

    1. **Want.** The batch as a whole aims at `sm_count * TARGET_BLOCKS_PER_SM`
       threadgroups, which is `ceil(target / n_slots)` tiles to share out.
       Each item's share is proportional to its row count, at least one tile,
       and never more than its rows can fill at the amortization floor, so a
       leaf of 300 rows does not get eight tiles of 40 rows each.
    2. **Fit.** The tiled strategy needs `total_tiles * n_slots * n_bins`
       partial cells. Where the budget is smaller, shares are scaled down
       proportionally, still with one tile per item as the floor. If the
       floors alone do not fit, the tiled strategy cannot serve this batch at
       all and the plan resolves to the atomic strategy, which needs no
       partial buffer. Both outcomes are reported through `verdict`.
    3. **Square up.** Rows per tile are re-derived from the final tile count
       and the tile count from that, so the last tile of every item is
       nonempty and `grid.y` covers exactly the rows the items hold.

    `max_partial_cells` is the partial buffer the caller already holds, in
    cells. Zero means unbounded, which is what a planner running without a
    device wants; `GpuLeafBatcher.enqueue_batch` refuses a plan that does not
    fit its own buffer, so an unbounded plan is checked before it can launch.
    An explicit `STRATEGY_TILED` that the buffer cannot serve is downgraded
    to the atomic strategy rather than refused, because the atomic strategy
    produces the identical histogram and refusing would leave the caller with
    no way to build the batch at all. `verdict` reports the downgrade.

    `scales` is `2 * len(items)` Float32, the fixed-point gradient and
    hessian scales the corresponding item's histogram must be accumulated
    with. They are per item because a batch may span classes, whose scales
    differ; a single-class batch passes the same pair repeatedly.
    """
    if len(items) < 1:
        raise Error("a batch needs at least one item")
    if len(items) > max_items:
        raise Error("batch holds more items than the launch bound allows")
    if len(scales) != SCALE_WORDS * len(items):
        raise Error("a batch needs one gradient and hessian scale per item")
    if n_slots < 1:
        raise Error("a batch needs at least one active feature")
    if n_bins < 1 or n_bins > MAX_BINS:
        raise Error("the GPU backend supports 1 to 256 bins")

    var block_threads = derive_block_threads(caps)
    var floor_rows = _min_rows_per_tile(n_bins, block_threads)
    var target_blocks = caps.sm_count * TARGET_BLOCKS_PER_SM
    if target_blocks < 1:
        target_blocks = 1

    var n = len(items)
    var counts = List[Int](capacity=n)
    var tile_caps = List[Int](capacity=n)
    var total_rows = 0
    for i in range(n):
        if items[i].row_begin < 0 or items[i].row_count < 0:
            raise Error("a batch item holds a negative row window")
        if items[i].out_slot < 0:
            raise Error("a batch item has no output slot")
        if items[i].plane < 0:
            raise Error("a batch item holds a negative gradient plane")
        for k in range(i):
            if items[k].out_slot == items[i].out_slot:
                raise Error("two batch items share an output slot")
        # An empty leaf still needs a launchable geometry, and one row is
        # what `GpuActiveRows.range_tiling` derives it at. Its row loop runs
        # zero iterations, so the histogram it produces is the zeros the
        # zeroing pass already wrote.
        var c = items[i].row_count
        if c < 1:
            c = 1
        counts.append(c)
        tile_caps.append(_ceil_div(c, floor_rows))
        total_rows += c

    # Pass 1: proportional shares of the tile target.
    var tiles_target = _ceil_div(target_blocks, n_slots)
    if tiles_target < n:
        tiles_target = n
    var tiles = List[Int](capacity=n)
    var total_tiles = 0
    for i in range(n):
        var want = _ceil_div(tiles_target * counts[i], total_rows)
        if want < 1:
            want = 1
        if want > tile_caps[i]:
            want = tile_caps[i]
        tiles.append(want)
        total_tiles += want

    # Pass 2: the partial-buffer bound.
    var resolved = strategy
    var verdict = VERDICT_UNKNOWN
    var hist_cells = n_slots * n_bins
    var tiles_budget: Int
    if max_partial_cells > 0:
        tiles_budget = max_partial_cells // hist_cells
    else:
        tiles_budget = MAX_GRID_DIM_Y
    if tiles_budget > MAX_GRID_DIM_Y:
        tiles_budget = MAX_GRID_DIM_Y

    if total_tiles > tiles_budget:
        if n > tiles_budget:
            # Not even one tile per item fits, so no tiled plan exists for
            # this batch. The atomic strategy allocates nothing and can.
            resolved = STRATEGY_ATOMIC
            verdict = VERDICT_PARTIAL_BOUND
            if total_tiles > MAX_GRID_DIM_Y:
                raise Error(
                    "batched histogram needs more row tiles than the portable"
                    " grid limit allows; batch fewer leaves"
                )
        else:
            var scaled = 0
            for i in range(n):
                var want = (tiles[i] * tiles_budget) // total_tiles
                if want < 1:
                    want = 1
                tiles[i] = want
                scaled += want
            # Integer flooring can leave the total above the budget by at
            # most one tile per item; give the surplus back in ascending item
            # order, never below the floor of one. Ascending rather than
            # largest-first because the order has to be a function of the
            # batch and not of a sort's tie-breaking, and every order that is
            # gives the same total.
            var over = scaled - tiles_budget
            var idx = 0
            while over > 0 and idx < n:
                if tiles[idx] > 1:
                    var give = tiles[idx] - 1
                    if give > over:
                        give = over
                    tiles[idx] -= give
                    over -= give
                idx += 1
            total_tiles = 0
            for k in range(n):
                total_tiles += tiles[k]
            verdict = VERDICT_PARTIAL_BOUND

    # Pass 3: square the geometry up so no tile is empty.
    var plans = List[BatchItemPlan](capacity=n)
    var running = 0
    for i in range(n):
        var rows_per_tile = _ceil_div(counts[i], tiles[i])
        if rows_per_tile < 1:
            rows_per_tile = 1
        var n_tiles = _ceil_div(counts[i], rows_per_tile)
        plans.append(
            BatchItemPlan(
                items[i].row_begin,
                items[i].row_count,
                items[i].out_slot,
                items[i].plane,
                rows_per_tile,
                n_tiles,
                running,
                scales[SCALE_WORDS * i + SCALE_G],
                scales[SCALE_WORDS * i + SCALE_H],
            )
        )
        running += n_tiles
    total_tiles = running
    if total_tiles > MAX_GRID_DIM_Y:
        raise Error(
            "batched histogram needs more row tiles than the portable grid"
            " limit allows; batch fewer leaves"
        )

    if resolved == STRATEGY_AUTO:
        # Same rule the single-leaf path uses. More than one tile per feature
        # is what the tiled path exists for; at one tile the partial buffer
        # is the same size as the output and the second launch buys nothing.
        if total_tiles > n:
            resolved = STRATEGY_TILED
        else:
            resolved = STRATEGY_ATOMIC

    var partial_cells = 0
    if resolved == STRATEGY_TILED:
        partial_cells = total_tiles * hist_cells
        if max_partial_cells > 0 and partial_cells > max_partial_cells:
            raise Error(
                "batch plan exceeds the partial buffer it was given"
            )

    if verdict == VERDICT_UNKNOWN:
        verdict = _occupancy_verdict(
            tile_caps, n_slots, target_blocks, total_tiles
        )

    return BatchPlan(
        plans^,
        resolved,
        block_threads,
        n_slots,
        n_bins,
        total_tiles,
        partial_cells,
        verdict,
        target_blocks,
    )


def _occupancy_verdict(
    tile_caps: List[Int],
    n_slots: Int,
    target_blocks: Int,
    total_tiles: Int,
) -> Int:
    """The structural occupancy fact about this plan, and only that.

    Three states are decidable without measuring anything. Some item would
    already reach the block target launched on its own, so batching cannot
    improve its occupancy. Or no item would and the batch does, which is the
    case batching exists for. Or neither, which is an honest unknown.

    The first test is against what an item would get *alone*, not against the
    tiles it got inside this batch, since inside a batch it shares the target
    with everyone else. Alone, an item gets the whole tile target, capped by
    the tiles its own rows can fill at the amortization floor, which is what
    `tile_caps` holds.

    No crossover threshold is implied by any of the three, and none is
    invented here. Which of them is worth batching on a given device is what
    `bench/apple/leaf_batching_plan.json` is for.
    """
    var alone_target = _ceil_div(target_blocks, n_slots)
    if alone_target < 1:
        alone_target = 1
    for i in range(len(tile_caps)):
        var alone = tile_caps[i]
        if alone > alone_target:
            alone = alone_target
        if alone < 1:
            alone = 1
        if alone * n_slots >= target_blocks:
            return VERDICT_SINGLE_FILLS
    if total_tiles * n_slots >= target_blocks:
        return VERDICT_OCCUPANCY_GAIN
    return VERDICT_UNKNOWN


# --- Symbolic cost --------------------------------------------------------


@fieldwise_init
struct BatchCost(Copyable, Movable):
    """What a plan costs, in countable units rather than in seconds.

    Every field is a number a profiler can be held to, which is the point:
    the comparison between batched and serial launches should be an
    arithmetic prediction that a measurement confirms or refutes, not an
    argument.
    """

    var launches: Int
    """Kernel launches, counting the zeroing pass and the reduction."""

    var blocks: Int
    """Threadgroups the histogram launch (or launches) creates."""

    var idle_blocks: Int
    """Threadgroups that exit without reading a row. Zero for a packed tile
    axis by construction, and carried anyway so a plan that stops being
    packed cannot hide it."""

    var bin_reads: Int
    """`sum over items of row_count * n_slots`: the gathered bin loads, the
    dominant term and the one batching does not change."""

    var partial_bytes: Int
    var output_bytes: Int
    var reduce_reads: Int
    """Int32 loads the reduction performs, `3 * n_slots * n_bins` per tile."""

    var atomic_folds: Int
    """Global integer atomics the atomic strategy issues, three per
    (tile, slot, nonempty bin) in the worst case."""


def batch_cost(plan: BatchPlan, n_features: Int) raises -> BatchCost:
    """The cost of running `plan` as one batched launch."""
    if n_features < 1:
        raise Error("cost model needs at least one feature")
    var bin_reads = 0
    var tiles = 0
    for i in range(plan.n_items()):
        bin_reads += plan.items[i].row_count * plan.n_slots
        tiles += plan.items[i].n_tiles
    var hist_cells = plan.n_slots * plan.n_bins
    var out_bytes = plan.n_items() * N_PLANES * n_features * plan.n_bins * 4
    var launches: Int
    var partial_bytes = 0
    var reduce_reads = 0
    var atomic_folds = 0
    if plan.strategy == STRATEGY_TILED:
        # Zero pass plus histogram plus reduction. The zero pass is still
        # needed whenever the batch does not write every feature slice.
        launches = 3
        partial_bytes = tiles * hist_cells * BYTES_PER_PARTIAL_CELL
        reduce_reads = tiles * N_PLANES * hist_cells
    else:
        launches = 2
        atomic_folds = tiles * N_PLANES * hist_cells
    return BatchCost(
        launches,
        plan.n_slots * tiles,
        0,
        bin_reads,
        partial_bytes,
        out_bytes,
        reduce_reads,
        atomic_folds,
    )


def serial_cost(plan: BatchPlan, n_features: Int) raises -> BatchCost:
    """The cost of the same items launched one leaf at a time, which is what
    the trainer does today.

    The row work is identical, because the same rows are read either way.
    What differs is the launch count, which scales with the item count
    instead of being constant, and the occupancy each launch reaches, which
    `BatchPlan.fills_device` reports and this model deliberately does not
    turn into a time.
    """
    var costed = batch_cost(plan, n_features)
    var per_item_launches = 3 if plan.strategy == STRATEGY_TILED else 2
    return BatchCost(
        per_item_launches * plan.n_items(),
        costed.blocks,
        costed.idle_blocks,
        costed.bin_reads,
        costed.partial_bytes,
        costed.output_bytes,
        costed.reduce_reads,
        costed.atomic_folds,
    )


def subtraction_saves_reads(n_left: Int, n_right: Int, n_slots: Int) -> Int:
    """Gathered bin loads the subtraction trick avoids on one commit.

    Building both children costs `(n_left + n_right) * n_slots`; building the
    smaller and subtracting costs `min(n_left, n_right) * n_slots` plus a
    launch over `3 * n_features * n_bins` cells that does not touch a row.
    The difference is the larger child's rows, which is what this returns,
    and it is why a batch of two children is not automatically better than a
    batch of one child plus a subtraction.
    """
    var larger = n_left if n_left > n_right else n_right
    return larger * n_slots


# --- Histogram slots ------------------------------------------------------


struct HistogramSlotPool(Movable):
    """Which device histogram slot belongs to which leaf.

    A slot is one full-width `3 * n_features * n_bins` Int32 histogram, the
    same shape and layout `GpuHistogramBuilder.out_dev` holds, so a slot can
    be downloaded and decoded by the existing `histogram_from_host` and read
    by the existing split-search kernels without a translation step.

    The pool is bounded and every slot is reclaimable, because the output
    buffer is the term that grows with the frontier: `P * 3 * F * B * 4`
    bytes. A leaf whose slot is taken back can be rebuilt from its row range
    at the cost of one more histogram, so the bound is a performance policy,
    never a correctness one.

    `stamp` is what makes `enqueue_subtract` safe. Two histograms may only be
    subtracted if they were accumulated under the same fixed-point scales and
    the same active feature set, and the stamp is the caller's encoding of
    both (a round counter and a feature-set counter folded together is
    enough). Slots with different stamps are refused rather than silently
    producing a histogram whose zero slices are not zero.
    """

    var capacity: Int
    var owner: List[Int]
    var stamp: List[Int]

    def __init__(out self, capacity: Int) raises:
        if capacity < 1:
            raise Error("a histogram slot pool needs at least one slot")
        self.capacity = capacity
        self.owner = List[Int](capacity=capacity)
        self.stamp = List[Int](capacity=capacity)
        for _ in range(capacity):
            self.owner.append(-1)
            self.stamp.append(-1)

    def free_slots(self) -> Int:
        var n = 0
        for i in range(self.capacity):
            if self.owner[i] < 0:
                n += 1
        return n

    def acquire(mut self, owner: Int, stamp: Int) raises -> Int:
        """The lowest free slot, or -1 when the pool is full.

        Lowest rather than most recently freed, so a sequence of acquires and
        releases produces the same slot assignment every run, which keeps the
        device buffer's contents a function of the tree and not of the
        allocator's history.
        """
        if owner < 0:
            raise Error("a histogram slot needs a nonnegative owner")
        for i in range(self.capacity):
            if self.owner[i] < 0:
                self.owner[i] = owner
                self.stamp[i] = stamp
                return i
        return -1

    def release(mut self, slot: Int) raises:
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        self.owner[slot] = -1
        self.stamp[slot] = -1

    def release_all(mut self):
        for i in range(self.capacity):
            self.owner[i] = -1
            self.stamp[i] = -1

    def owner_of(self, slot: Int) raises -> Int:
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        return self.owner[slot]

    def stamp_of(self, slot: Int) raises -> Int:
        """The compatibility stamp a slot was accumulated under, or -1 for a
        free slot. Two slots may only be subtracted when these agree; see
        `check_subtractable` and `subtraction_stamp`."""
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        return self.stamp[slot]

    def reassign(mut self, slot: Int, owner: Int) raises:
        """Hand a live slot to a different leaf without freeing it.

        The one move `acquire`/`release` cannot express, and the subtraction
        trick needs it: `enqueue_subtract(parent, child, dst=parent)` leaves
        the parent's slot holding the *larger child's* histogram, so the slot
        outlives its owner by one generation and the leaf that reads it next
        is not the leaf that filled it. Releasing and reacquiring would be
        wrong twice over, since it could hand the slot to another leaf in
        between and it would reset the stamp the derived histogram was
        actually accumulated under.

        The stamp is deliberately left alone for that reason: it describes
        the scales and feature set the words in the slot were accumulated
        with, which a change of owner does not move.
        """
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        if owner < 0:
            raise Error("a histogram slot needs a nonnegative owner")
        if self.owner[slot] < 0:
            raise Error("a free histogram slot has no owner to reassign")
        self.owner[slot] = owner

    def slot_of_owner(self, owner: Int) -> Int:
        """The live slot `owner` holds, or -1. Owners are leaf node ids and a
        node holds at most one slot, so the answer is unique."""
        for i in range(self.capacity):
            if self.owner[i] == owner:
                return i
        return -1

    def check_live(self, slot: Int) raises:
        if slot < 0 or slot >= self.capacity:
            raise Error("histogram slot out of range")
        if self.owner[slot] < 0:
            raise Error("histogram slot holds no leaf")

    def check_subtractable(self, a: Int, b: Int) raises:
        """Both slots live, and both accumulated under the same conditions."""
        self.check_live(a)
        self.check_live(b)
        if self.stamp[a] != self.stamp[b]:
            raise Error(
                "histograms accumulated under different scales or feature"
                " sets cannot be subtracted"
            )


def slot_bytes(n_features: Int, n_bins: Int) -> Int:
    """Device bytes one histogram slot occupies."""
    return N_PLANES * n_features * n_bins * 4


def pool_bytes(capacity: Int, n_features: Int, n_bins: Int) -> Int:
    return capacity * slot_bytes(n_features, n_bins)


def slots_for_budget(
    budget_bytes: Int, n_features: Int, n_bins: Int
) raises -> Int:
    """How many slots a byte budget buys, at least one.

    The number a caller needs to see before deciding to hold a frontier's
    histograms on the device at all: at 1000 features and 256 bins a slot is
    3 MB, so a 255-leaf frontier would ask for 750 MB and has to be told no.
    """
    if n_features < 1 or n_bins < 1:
        raise Error("slot sizing needs positive features and bins")
    var one = slot_bytes(n_features, n_bins)
    var n = budget_bytes // one
    if n < 1:
        return 1
    return n


def subtraction_stamp(round_index: Int, feature_epoch: Int) raises -> Int:
    """The compatibility stamp two histograms must share to be subtracted.

    `HistogramSlotPool` refuses a subtraction across differing stamps, and
    this is the encoding that makes the refusal mean the right thing. The two
    conditions are exactly the ones `enqueue_subtract`'s docstring names:

    - **Same fixed-point scales.** The scales are re-derived once per round
      and per class from the gradient magnitude sums, so they are constant
      within a tree and generally differ between trees. `round_index` is the
      caller's counter over (boosting round, class) pairs, which is the
      granularity a scale actually changes at.
    - **Same active feature set.** A feature present in one histogram and
      absent from the other would leave a nonzero slice being subtracted from
      a zero one. `feature_epoch` is the caller's counter, bumped whenever
      `set_features`/`set_shared_features` narrows or widens the set, which
      the trainer does once per tree.

    Folded rather than paired because the pool stores one Int and only ever
    compares stamps for equality; folding two nonnegative counters into
    `round * 2^24 + epoch` keeps that comparison exact for any run inside the
    two bounds below, and refuses outside them rather than aliasing two
    incompatible histograms onto one stamp. The bounds are far above any real
    fit (16.7 million feature epochs, and rounds counted over (iteration,
    class) pairs into the billions) and are checked anyway, because an
    aliased stamp is a wrong histogram and not a lost slot.
    """
    if round_index < 0 or feature_epoch < 0:
        raise Error("stamp counters must be nonnegative")
    if feature_epoch >= (1 << 24):
        raise Error("feature epoch is too large for a subtraction stamp")
    if round_index >= (1 << 32):
        raise Error("round index is too large for a subtraction stamp")
    return (round_index << 24) + feature_epoch


# --- The frontier seam ----------------------------------------------------
#
# What a grower calls to turn `LeafFrontier` state into one launch, without
# knowing anything below about tiles, item tables, or the packed grid axis.
# Four steps, in order, and each one may be declined rather than forced:
#
#     admit_frontier_batch   is a batch legal and worth assembling at all
#     assign_batch_slots     give each covered leaf an output histogram slot
#     plan_frontier_batch    resolve the launch geometry
#     GpuLeafBatcher.enqueue_frontier_batch   launch it over the active rows
#
# The admission step is the conservative switch. Every reason it can decline
# leaves the caller on the established one-leaf-per-launch path, which is what
# `GpuActiveRows.enqueue_range_histogram` already does correctly, so a grower
# can adopt the batched path without a second correctness story.

comptime BATCH_OK = 0

comptime BATCH_NO_PENDING = 1
"""No leaf needs work, so there is nothing to launch. Not a failure."""

comptime BATCH_SINGLE_ITEM = 2
"""Only one leaf needs work. A batch of one is the single-leaf launch with an
item table in front of it, so the established path is preferred outright."""

comptime BATCH_NO_SLOTS = 3
"""The histogram slot pool cannot give every covered leaf its own output
slice. Slots are reclaimable and a leaf can always be rebuilt from its row
range, so this is a memory bound, never a correctness one."""

comptime BATCH_KERNEL_ABSENT = 4
"""`KernelFeatures.batched_leaf_kernel` is false: the batched kernels are not
compiled in and validated for this build."""

comptime BATCH_STORAGE_UNSUPPORTED = 5
"""The bin matrix is not the one-`UInt8`-per-cell, one-feature-per-block
matrix these kernels index. A packed or blocked layout needs a decoding
kernel that does not exist; see `BinStorageDescriptor.check_shipping`."""


def batch_admission_name(reason: Int) -> String:
    if reason == BATCH_OK:
        return String("ok")
    if reason == BATCH_NO_PENDING:
        return String("no_pending_leaves")
    if reason == BATCH_SINGLE_ITEM:
        return String("single_item")
    if reason == BATCH_NO_SLOTS:
        return String("no_free_histogram_slots")
    if reason == BATCH_KERNEL_ABSENT:
        return String("batched_kernel_absent")
    if reason == BATCH_STORAGE_UNSUPPORTED:
        return String("bin_storage_unsupported")
    return String("unknown")


struct BatchAdmission(Copyable, Movable):
    """Whether a batch may be assembled from this frontier, and over which
    slots.

    Carries the numbers the decision was made from, so a caller that declines
    can report why rather than only that. `slots` is empty unless
    `reason == BATCH_OK`, which keeps "declined" and "admitted over nothing"
    from being the same value.
    """

    var reason: Int
    var slots: List[Int]
    """Frontier slots the batch would cover, ascending."""

    var pending: Int
    """Leaves needing work, before the `max_items` cap."""

    var free_slots: Int

    def __init__(
        out self,
        reason: Int,
        var slots: List[Int],
        pending: Int,
        free_slots: Int,
    ):
        self.reason = reason
        self.slots = slots^
        self.pending = pending
        self.free_slots = free_slots

    def admitted(self) -> Bool:
        return self.reason == BATCH_OK

    def n_items(self) -> Int:
        return len(self.slots)

    def describe(self) -> String:
        return (
            String("batch ")
            + batch_admission_name(self.reason)
            + " items "
            + String(len(self.slots))
            + " pending "
            + String(self.pending)
            + " free_slots "
            + String(self.free_slots)
        )


def _declined(reason: Int, pending: Int, free_slots: Int) -> BatchAdmission:
    """A declined admission, spelled once so every reason below produces the
    same shape: no slots, and the two numbers the decision was made from."""
    return BatchAdmission(reason, List[Int](), pending, free_slots)


def admit_frontier_batch(
    frontier: LeafFrontier,
    pool: HistogramSlotPool,
    storage: BinStorageDescriptor,
    features: KernelFeatures,
    max_items: Int = DEFAULT_MAX_ITEMS,
    min_items: Int = 2,
) raises -> BatchAdmission:
    """Decide whether this frontier can be batched, and over which slots.

    Pure host arithmetic against state that already exists: no device, no
    allocation, and nothing mutated, so a caller may ask before committing to
    anything. The order of the tests is the order a caller would want them
    reported in, cheapest structural fact first.

    A leaf that already holds a live slot is counted as needing no new one, so
    re-batching a frontier whose slots survived does not spuriously decline
    for want of capacity.

    `min_items` is the width below which the batched path is not worth taking;
    two is the smallest batch that is not a single-leaf launch in disguise,
    and it is the default because that is a structural fact rather than a
    measured threshold. A benchmark that wants a higher bar passes one.
    """
    if max_items < 1:
        raise Error("a batch holds at least one item")
    if min_items < 1:
        raise Error("a batch holds at least one item")
    storage.check()

    var pending = len(frontier.pending())
    var free = pool.free_slots()
    if not features.batched_leaf_kernel:
        return _declined(BATCH_KERNEL_ABSENT, pending, free)
    if not storage.is_dense_feature_major_u8():
        return _declined(BATCH_STORAGE_UNSUPPORTED, pending, free)
    if pending < 1:
        return _declined(BATCH_NO_PENDING, pending, free)
    if pending < min_items:
        return _declined(BATCH_SINGLE_ITEM, pending, free)

    var slots = frontier.batch_slots(max_items)
    if len(slots) < min_items:
        return _declined(BATCH_SINGLE_ITEM, pending, free)
    var needed = 0
    for i in range(len(slots)):
        var leaf = frontier.leaf(slots[i])
        if leaf.hist_slot == NO_SLOT:
            needed += 1
        elif pool.owner_of(leaf.hist_slot) != leaf.node:
            # The slot the leaf remembers was taken back by the pool, so it
            # needs a fresh one. Eviction costs a rebuild, never an answer.
            needed += 1
    if needed > free:
        return _declined(BATCH_NO_SLOTS, pending, free)
    return BatchAdmission(BATCH_OK, slots^, pending, free)


def assign_batch_slots(
    mut frontier: LeafFrontier,
    mut pool: HistogramSlotPool,
    admission: BatchAdmission,
    stamp: Int,
) raises -> List[Int]:
    """Give every leaf the batch covers its own output histogram slot.

    Returns the pool slots, in the admission's order, and writes each one back
    into the frontier so `work_items` picks it up. A leaf already holding a
    live slot under the same stamp keeps it, which is what makes re-batching
    cheap; a slot under a *different* stamp is released and reacquired,
    because a histogram accumulated under other scales or another feature set
    is not a histogram this batch may subtract from or add to.

    All-or-nothing. If the pool runs dry part way through (which
    `admit_frontier_batch` is meant to prevent, but which a caller that
    acquired slots in between can still cause), every slot this call took is
    given back and every frontier assignment it made is undone, so the
    frontier is exactly as it was and the caller can fall back to the
    single-leaf path without a leaked slot.
    """
    if not admission.admitted():
        raise Error(
            "only an admitted batch may be given slots: "
            + batch_admission_name(admission.reason)
        )
    var taken = List[Int]()
    var taken_at = List[Int]()
    var out = List[Int](capacity=len(admission.slots))
    for i in range(len(admission.slots)):
        var s = admission.slots[i]
        var leaf = frontier.leaf(s)
        var held = leaf.hist_slot
        if held != NO_SLOT:
            if (
                pool.owner_of(held) == leaf.node
                and pool.stamp_of(held) == stamp
            ):
                out.append(held)
                continue
            if pool.owner_of(held) == leaf.node:
                pool.release(held)
            frontier.assign_slot(s, NO_SLOT)
        var got = pool.acquire(leaf.node, stamp)
        if got < 0:
            for k in range(len(taken)):
                pool.release(taken[k])
                frontier.assign_slot(taken_at[k], NO_SLOT)
            raise Error(
                "the histogram slot pool ran out mid-batch; grow the pool"
                " (slots_for_budget) or batch fewer leaves"
            )
        taken.append(got)
        taken_at.append(s)
        frontier.assign_slot(s, got)
        out.append(got)
    return out^


def release_batch_slots(
    mut frontier: LeafFrontier, mut pool: HistogramSlotPool, slots: List[Int]
) raises:
    """Give a batch's output slots back and clear the frontier's memory of
    them. For a caller abandoning a planned batch, and for the end of a tree,
    where every slot is dead at once and `HistogramSlotPool.release_all` is
    the cheaper call."""
    for i in range(len(slots)):
        var s = slots[i]
        if s < 0 or s >= frontier.size():
            raise Error("frontier slot out of range")
        var held = frontier.leaf(s).hist_slot
        if held != NO_SLOT:
            pool.release(held)
            frontier.assign_slot(s, NO_SLOT)


def uniform_scales(
    n_items: Int, g_scale: Float32, h_scale: Float32
) raises -> List[Float32]:
    """One gradient/hessian scale pair repeated for every item.

    The single-class case, which is every batch the trainer assembles today:
    a round has one pair of fixed-point scales and every leaf of every tree in
    it accumulates under them. A multiclass batch that spanned classes would
    build the list itself, pair by pair, which is why `plan_batch` takes a
    list rather than a pair.
    """
    if n_items < 1:
        raise Error("a batch needs at least one item")
    if g_scale <= 0.0 or h_scale <= 0.0:
        raise Error("fixed-point scales must be positive")
    var out = List[Float32](capacity=SCALE_WORDS * n_items)
    for _ in range(n_items):
        out.append(g_scale)
        out.append(h_scale)
    return out^


def plan_frontier_batch(
    caps: DeviceCaps,
    frontier: LeafFrontier,
    admission: BatchAdmission,
    g_scale: Float32,
    h_scale: Float32,
    n_slots: Int,
    n_bins: Int,
    strategy: Int = STRATEGY_AUTO,
    max_partial_cells: Int = 0,
    max_items: Int = DEFAULT_MAX_ITEMS,
) raises -> BatchPlan:
    """Resolve the launch for an admitted, slotted batch.

    Thin on purpose: it turns frontier slots into `LeafWorkItem`s (which is
    where the row windows, the output slots, and the gradient plane come
    from), repeats the round's scales once per item, and hands both to
    `plan_batch`, which is the only tile arithmetic in this lane. Call
    `assign_batch_slots` first; an item whose leaf still holds `NO_SLOT` is
    refused here with a message that says so, rather than reaching
    `plan_batch` as a negative index.
    """
    if not admission.admitted():
        raise Error(
            "only an admitted batch may be planned: "
            + batch_admission_name(admission.reason)
        )
    var items = frontier.work_items(admission.slots)
    for i in range(len(items)):
        if items[i].out_slot == NO_SLOT:
            raise Error(
                "a batched leaf holds no output histogram slot; call"
                " assign_batch_slots before plan_frontier_batch"
            )
    var scales = uniform_scales(len(items), g_scale, h_scale)
    return plan_batch(
        caps,
        items,
        scales,
        n_slots,
        n_bins,
        strategy,
        max_partial_cells,
        max_items,
    )


def check_batch_covers_ranges(
    plan: BatchPlan, frontier: LeafFrontier, admission: BatchAdmission
) raises:
    """Every item's window is the window its frontier leaf holds.

    Cheap, and worth running before a launch that reads through the active-row
    permutation: an item whose `row_begin`/`row_count` had drifted from its
    leaf would accumulate a histogram of some other leaf's rows entirely
    inside the buffer, so no bound would be violated and no later check would
    notice. `GpuActiveRows.check_frontier` is the other half, holding the
    frontier equal to the device's own range table.
    """
    if plan.n_items() != len(admission.slots):
        raise Error("plan and admission cover different item counts")
    for i in range(plan.n_items()):
        var leaf = frontier.leaf(admission.slots[i])
        if plan.items[i].row_begin != leaf.row_begin:
            raise Error("a batch item does not start at its leaf's rows")
        if plan.items[i].row_count != leaf.row_count:
            raise Error("a batch item does not cover its leaf's rows")


def batch_windows(plan: BatchPlan) raises -> List[LeafRange]:
    """Each item's row window, as the lane's one window type.

    `LeafRange` is `gpu_active_rows`' half-open `[begin, end)`, and a batch
    item carries the same window as a begin and a count. Returning the shared
    type rather than a second pair is what keeps a caller from inventing a
    third spelling of a leaf's rows; `LeafRange.overlaps` is then the ready
    answer to "do two items of this batch read the same rows", which they
    never should, because live leaves tile the active prefix.
    """
    var out = List[LeafRange](capacity=plan.n_items())
    for i in range(plan.n_items()):
        var begin = plan.items[i].row_begin
        out.append(LeafRange(begin, begin + plan.items[i].row_count))
    return out^


def batched_leaf_stats(
    raw: List[Int32],
    n_features: Int,
    n_bins: Int,
    feature: Int,
    g_scale: Float64,
    h_scale: Float64,
) raises -> LeafStats:
    """One downloaded slot's gradient sum, hessian sum, and row count.

    A histogram's count plane sums to the leaf's rows over any one feature,
    and its gradient and hessian planes to the leaf's sums, so a single
    feature's slice is enough and the whole slot does not have to be scanned.
    The scales are the ones the slot was accumulated under, which is the same
    pair `HistogramSlotPool`'s stamp is meant to keep constant, so a caller
    that respects the stamp cannot convert with the wrong divisor.

    This is the host-side bridge from a batched result back into
    `LeafFrontier.set_stats`, and it is deliberately the only one: nothing
    here re-derives a leaf's rows from anything but its own counts.
    """
    var hist_size = n_features * n_bins
    if len(raw) != N_PLANES * hist_size:
        raise Error("raw histogram is not one full-width slot")
    if feature < 0 or feature >= n_features:
        raise Error("feature index out of range")
    if g_scale == 0.0 or h_scale == 0.0:
        raise Error("fixed-point scales must not be zero")
    var base = feature * n_bins
    var g = 0.0
    var h = 0.0
    var c = 0
    for b in range(n_bins):
        g += Float64(raw[base + b])
        h += Float64(raw[hist_size + base + b])
        c += Int(raw[2 * hist_size + base + b])
    return LeafStats(g / g_scale, h / h_scale, c)


# --- The batcher ----------------------------------------------------------


struct GpuLeafBatcher(Movable):
    """Device buffers and launches for batched multi-leaf histograms.

    Construct once per training session on the histogram builder's own
    `DeviceContext`, so the batch's kernels queue behind that builder's
    gradient upload and partition kernels with no fence. The binned matrix,
    the active-row permutation, and the gradient planes stay where they are
    and arrive as pointers; this struct owns only the item tables, the
    partial buffer, and the histogram slot pool's backing store.
    """

    var ctx: DeviceContext
    var n_features: Int
    var n_bins: Int
    var n_rows: Int
    var max_items: Int
    var n_planes: Int
    # One Int32 row per item: see the ITEM_* layout above.
    var items_dev: DeviceBuffer[DType.int32]
    # Per-item fixed-point scales, gradient then hessian.
    var scales_dev: DeviceBuffer[DType.float32]
    # Global feature ids per (item, slot), strided by `n_features` rather
    # than by the batch's `n_slots`, so narrowing the feature set never moves
    # an item's row. The kernels are told the stride. Replicated across items
    # when the batch shares one feature set, which is the trainer's case.
    var feat_dev: DeviceBuffer[DType.int32]
    # `pool.capacity` full-width histograms, in the builder's layout.
    var out_dev: DeviceBuffer[DType.int32]
    # `3 * partial_capacity` Int32, or one element when the atomic strategy
    # is all this batcher will ever run.
    var part_dev: DeviceBuffer[DType.int32]
    var partial_capacity: Int
    var block_threads: Int
    var stage_items: HostBuffer[DType.int32]
    var stage_scales: HostBuffer[DType.float32]
    var stage_feat: HostBuffer[DType.int32]
    var pool: HistogramSlotPool

    def __init__(
        out self,
        ctx: DeviceContext,
        caps: DeviceCaps,
        n_rows: Int,
        n_features: Int,
        n_bins: Int,
        pool_capacity: Int,
        partial_capacity: Int,
        max_items: Int = DEFAULT_MAX_ITEMS,
        n_planes: Int = 1,
    ) raises:
        """Allocate everything a session will use. Nothing below is sized by
        a leaf, so a batch allocates nothing.

        `partial_capacity` is in cells, `3 * partial_capacity` Int32 words,
        and passing zero means this batcher will only ever run the atomic
        strategy. `pool_capacity` slots at `3 * F * B` Int32 each is the term
        to watch; `slots_for_budget` sizes it against a byte budget.
        """
        if n_rows < 1:
            raise Error("the GPU backend requires at least one row")
        if n_rows > MAX_ROWS:
            raise Error("the GPU backend supports at most 2^31 - 1 rows")
        if n_features < 1:
            raise Error("the GPU backend requires at least one feature")
        if n_bins < 1 or n_bins > MAX_BINS:
            raise Error("the GPU backend supports 1 to 256 bins")
        if max_items < 1:
            raise Error("a batch holds at least one item")
        if n_planes < 1:
            raise Error("at least one gradient plane must be resident")
        if partial_capacity < 0:
            raise Error("partial capacity must be nonnegative")

        self.ctx = ctx
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_bins = n_bins
        self.max_items = max_items
        self.n_planes = n_planes
        self.partial_capacity = partial_capacity
        self.block_threads = derive_block_threads(caps)
        self.pool = HistogramSlotPool(pool_capacity)

        var hist_size = n_features * n_bins
        self.items_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_items * ITEM_WORDS
        )
        self.scales_dev = self.ctx.enqueue_create_buffer[DType.float32](
            max_items * SCALE_WORDS
        )
        self.feat_dev = self.ctx.enqueue_create_buffer[DType.int32](
            max_items * n_features
        )
        self.out_dev = self.ctx.enqueue_create_buffer[DType.int32](
            pool_capacity * N_PLANES * hist_size
        )
        var part_size = N_PLANES * partial_capacity
        if part_size < 1:
            part_size = 1
        self.part_dev = self.ctx.enqueue_create_buffer[DType.int32](part_size)
        self.stage_items = self.ctx.enqueue_create_host_buffer[DType.int32](
            max_items * ITEM_WORDS
        )
        self.stage_scales = self.ctx.enqueue_create_host_buffer[
            DType.float32
        ](max_items * SCALE_WORDS)
        self.stage_feat = self.ctx.enqueue_create_host_buffer[DType.int32](
            max_items * n_features
        )

        # Every item's feature table starts as the identity, so a caller that
        # never narrows the feature set can launch without staging one.
        var dst = self.stage_feat.unsafe_ptr()
        for k in range(max_items):
            for f in range(n_features):
                dst.unsafe_store(k * n_features + f, Int32(f))
        self.ctx.enqueue_copy(dst_buf=self.feat_dev, src_ptr=dst)
        self.ctx.synchronize()

    def synchronize(self) raises:
        self.ctx.synchronize()

    def hist_size(self) -> Int:
        return self.n_features * self.n_bins

    def slot_cells(self) -> Int:
        return N_PLANES * self.n_features * self.n_bins

    def max_tiles(self, n_slots: Int) raises -> Int:
        """Row tiles the partial buffer holds at `n_slots` active features.

        The capacity is in cells, and one tile costs `n_slots * n_bins` of
        them, so the tile budget moves with the feature set. This is the
        number to hand `plan_batch` as `max_partial_cells / hist_cells`
        reasoning, and the one that decides whether a wide batch has to fall
        back to the atomic strategy.
        """
        if n_slots < 1:
            raise Error("tile budget needs at least one active feature")
        return self.partial_capacity // (n_slots * self.n_bins)

    def set_shared_features(mut self, features: List[Int]) raises:
        """One feature set for every item in the next batch.

        The trainer's case: `GpuHistogramBuilder.set_features` narrows the
        grid to the tree's sampled features once per tree, and every node of
        that tree accumulates the same set. Replicating it per item costs
        `max_items * n_slots` Int32 of staging and lets the kernels index one
        table without branching on whether the batch shares a set.

        Shares the staging contract on `_stage_plan`: this writes the pinned
        feature table, so it runs once per tree, before that tree's first
        batch, and never between a batch's launch and its completion.
        """
        if len(features) == 0:
            raise Error("an active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
        var dst = self.stage_feat.unsafe_ptr()
        for k in range(self.max_items):
            for i in range(len(features)):
                dst.unsafe_store(k * self.n_features + i, Int32(features[i]))
        self.ctx.enqueue_copy(dst_buf=self.feat_dev, src_ptr=dst)

    def set_item_features(
        mut self, item: Int, features: List[Int]
    ) raises:
        """One item's own feature set, for a caller that narrows per node.

        Per-node feature sampling narrows the *search* today and not the
        accumulation, so nothing in the trainer needs this yet. It exists
        because the batched kernels read the feature table per item anyway,
        so supporting a per-leaf set costs one index and no branch, and a
        grower that wanted to accumulate only a node's sampled features could
        take it without a kernel change. Every item in a batch must still
        list the same *number* of features, which per-node sampling already
        guarantees since the count comes from the tree's set and the
        fraction.
        """
        if item < 0 or item >= self.max_items:
            raise Error("batch item index out of range")
        if len(features) == 0:
            raise Error("an active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        var dst = self.stage_feat.unsafe_ptr()
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
            dst.unsafe_store(item * self.n_features + i, Int32(features[i]))
        self.ctx.enqueue_copy(dst_buf=self.feat_dev, src_ptr=dst)

    def _stage_plan(mut self, plan: BatchPlan) raises:
        """Write the item table and the scales into pinned memory and upload
        both.

        **Ordering contract.** The three staging buffers are pinned host
        memory reused by every batch, and the host write below is not ordered
        against a copy the device has already been handed. So a batch's copies
        must have completed before the next batch stages over them: the
        caller synchronizes, or downloads a result, between two batches. This
        is the same contract `GpuSplitSearcher.enqueue` states for its allow
        mask and parameter block, and it is stated rather than enforced for
        the same reason, that enforcing it with a synchronization here would
        serialize exactly the pipelining a batch exists to buy. A caller that
        wants two batches in flight needs a staging ring, not a change to the
        kernels.
        """
        if plan.n_items() > self.max_items:
            raise Error("batch holds more items than this batcher allows")
        var dst = self.stage_items.unsafe_ptr()
        var sdst = self.stage_scales.unsafe_ptr()
        for i in range(plan.n_items()):
            var it = plan.items[i].copy()
            if it.row_begin < 0 or it.row_begin + it.row_count > self.n_rows:
                raise Error("a batch item's rows escape the row buffer")
            if it.out_slot < 0 or it.out_slot >= self.pool.capacity:
                raise Error("a batch item's output slot is out of range")
            if it.plane < 0 or it.plane >= self.n_planes:
                raise Error("a batch item's gradient plane is out of range")
            if it.g_scale <= 0.0 or it.h_scale <= 0.0:
                raise Error("fixed-point scales must be positive")
            var base = i * ITEM_WORDS
            dst.unsafe_store(base + ITEM_BEGIN, Int32(it.row_begin))
            dst.unsafe_store(base + ITEM_COUNT, Int32(it.row_count))
            dst.unsafe_store(
                base + ITEM_ROWS_PER_TILE, Int32(it.rows_per_tile)
            )
            dst.unsafe_store(base + ITEM_TILE_BEGIN, Int32(it.tile_begin))
            dst.unsafe_store(base + ITEM_TILES, Int32(it.n_tiles))
            dst.unsafe_store(base + ITEM_OUT, Int32(it.out_slot))
            dst.unsafe_store(base + ITEM_PLANE, Int32(it.plane))
            dst.unsafe_store(base + 7, Int32(0))
            sdst.unsafe_store(SCALE_WORDS * i + SCALE_G, it.g_scale)
            sdst.unsafe_store(SCALE_WORDS * i + SCALE_H, it.h_scale)
        self.ctx.enqueue_copy(dst_buf=self.items_dev, src_ptr=dst)
        self.ctx.enqueue_copy(dst_buf=self.scales_dev, src_ptr=sdst)

    def enqueue_batch[
        bins_origin: MutOrigin,
        rows_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        plan: BatchPlan,
        bins: MutPointer[UInt8, bins_origin],
        rows: MutPointer[Int32, rows_origin],
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
    ) raises:
        """Enqueue one batched histogram build. Does not transfer or
        synchronize.

        `bins` is the device-resident binned matrix, `rows` the active-row
        permutation `GpuActiveRows` maintains, and `grad`/`hess` the round's
        gradient planes; every item's window indexes into `rows` absolutely,
        so nothing about which permutation it came from reaches the kernels.

        The zeroing pass runs whenever the batch does not write every feature
        slice of every output, which is whenever the feature set is narrowed,
        whenever the atomic strategy is in use, or whenever an item holds no
        rows. It is cheaper to state that as "always but the widest tiled
        case" than to make the caller reason about it, so that is what this
        does.
        """
        if plan.n_items() < 1:
            raise Error("a batch needs at least one item")
        if plan.n_bins != self.n_bins:
            raise Error("plan and batcher disagree on the bin count")
        if plan.n_slots > self.n_features:
            raise Error("plan holds more feature slots than features")
        if plan.strategy == STRATEGY_TILED:
            if plan.partial_cells > self.partial_capacity:
                raise Error(
                    "batch plan needs more partial cells than this batcher"
                    " allocated"
                )
        self._stage_plan(plan)

        var threads = plan.block_threads
        var hs = self.hist_size()
        var n_items = plan.n_items()

        var needs_zero = (
            plan.strategy != STRATEGY_TILED or plan.n_slots < self.n_features
        )
        if not needs_zero:
            for i in range(n_items):
                if plan.items[i].row_count <= 0:
                    needs_zero = True
                    break
        if needs_zero:
            var cells = n_items * N_PLANES * hs
            self.ctx.enqueue_function[_batch_zero_kernel](
                self.out_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                Int32(n_items),
                Int32(hs),
                grid_dim=_ceil_div(cells, threads),
                block_dim=threads,
            )

        if plan.strategy == STRATEGY_TILED:
            self.ctx.enqueue_function[_batch_hist_partial_kernel](
                bins,
                rows,
                grad,
                hess,
                self.feat_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                self.scales_dev.unsafe_ptr(),
                self.part_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(n_items),
                Int32(plan.n_slots),
                Int32(self.n_bins),
                Int32(self.n_features),
                grid_dim=(plan.n_slots, plan.total_tiles),
                block_dim=threads,
            )
            var cells = n_items * N_PLANES * plan.n_slots * self.n_bins
            self.ctx.enqueue_function[_batch_reduce_kernel](
                self.part_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                self.out_dev.unsafe_ptr(),
                Int32(n_items),
                Int32(plan.n_slots),
                Int32(self.n_bins),
                Int32(hs),
                Int32(self.n_features),
                grid_dim=_ceil_div(cells, threads),
                block_dim=threads,
            )
        else:
            self.ctx.enqueue_function[_batch_hist_atomic_kernel](
                bins,
                rows,
                grad,
                hess,
                self.feat_dev.unsafe_ptr(),
                self.items_dev.unsafe_ptr(),
                self.scales_dev.unsafe_ptr(),
                self.out_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(n_items),
                Int32(plan.n_slots),
                Int32(self.n_bins),
                Int32(hs),
                Int32(self.n_features),
                grid_dim=(plan.n_slots, plan.total_tiles),
                block_dim=threads,
            )

    def enqueue_frontier_batch[
        bins_origin: MutOrigin,
        grad_origin: MutOrigin,
        hess_origin: MutOrigin, //
    ](
        mut self,
        plan: BatchPlan,
        bins: MutPointer[UInt8, bins_origin],
        mut rows: GpuActiveRows,
        grad: MutPointer[Float32, grad_origin],
        hess: MutPointer[Float32, hess_origin],
    ) raises:
        """`enqueue_batch` against the active-row permutation itself.

        The connection that makes a batch read *only* the rows its leaves own:
        the row pointer is `GpuActiveRows`' own buffer rather than something
        the caller found, and every item's window is checked against that
        permutation's live prefix before anything launches. A window outside
        `[0, n_active)` is not a bounds violation on the buffer (the buffer is
        `n_rows` long and the prefix is at most that), so nothing downstream
        would catch it: it would silently accumulate over rows this tree does
        not grow on, which under bagging is a different dataset.

        Enqueues only. No transfer, no synchronization, and no allocation, so
        a batch of thirty leaves costs the same host round trips as a batch of
        one: `_stage_plan`'s two copies and at most three launches.
        """
        if rows.n_rows != self.n_rows:
            raise Error(
                "the active-row permutation and this batcher were built for"
                " different datasets"
            )
        var live = rows.n_active()
        for i in range(plan.n_items()):
            var begin = plan.items[i].row_begin
            var count = plan.items[i].row_count
            if begin < 0 or count < 0 or begin + count > live:
                raise Error(
                    "a batch item's rows escape this tree's active row"
                    " prefix"
                )
        self.enqueue_batch(
            plan, bins, rows.rows_dev.unsafe_ptr(), grad, hess
        )

    def enqueue_subtract(
        mut self, parent_slot: Int, child_slot: Int, dst_slot: Int
    ) raises:
        """`dst = parent - child`, on the device, over one histogram slot.

        Refuses operands whose stamps differ, which is the check that the two
        histograms were accumulated under the same fixed-point scales and the
        same active feature set. `dst_slot` may be `parent_slot`, which
        derives the sibling in place and is how a frontier keeps one slot per
        live leaf.
        """
        self.pool.check_subtractable(parent_slot, child_slot)
        if dst_slot < 0 or dst_slot >= self.pool.capacity:
            raise Error("histogram slot out of range")
        var cells = self.slot_cells()
        var threads = self.block_threads
        self.ctx.enqueue_function[_subtract_slice_kernel](
            self.out_dev.unsafe_ptr(),
            Int32(parent_slot),
            Int32(child_slot),
            Int32(dst_slot),
            Int32(cells),
            grid_dim=_ceil_div(cells, threads),
            block_dim=threads,
        )

    def download_slot(mut self, slot: Int) raises -> List[Int32]:
        """One slot's fixed-point histogram, host side. Synchronizes.

        The words come back in `GpuHistogramBuilder`'s own
        `[grad | hess | count]` layout, so a caller converts them with the
        same arithmetic `histogram_from_host` uses and needs no second
        decoder.

        This is the path the batching exists to avoid, so it is deliberately
        the unoptimized one: a mapping of the whole output pool rather than a
        pinned one-way copy of one slice. It is here so a batched result can
        be read at all, by a caller mixing batched construction with a
        host-side split search and by anything checking a batched histogram
        against a single-leaf one. A caller doing that per node should move
        the search to the device instead of making this fast.
        """
        self.pool.check_live(slot)
        var cells = self.slot_cells()
        var out = List[Int32](capacity=cells)
        with self.out_dev.map_to_host() as host:
            var p = host.unsafe_ptr()
            for i in range(cells):
                out.append(p.unsafe_load(slot * cells + i))
        return out^

    def download_slots(mut self, slots: List[Int]) raises -> List[Int32]:
        """Several slots' fixed-point histograms in one mapping.

        The whole point of batching is that a launch is not paid per leaf, and
        `download_slot` gives that back on the way home: one `map_to_host` per
        leaf is one synchronization per leaf. This maps once and copies every
        requested slot out of that mapping, so a batch of thirty costs one.

        The result is the slots concatenated in the order given, `slot_cells()`
        words each, so slot `k` of the request starts at
        `k * slot_cells()`. Each one is in `GpuHistogramBuilder`'s
        `[grad | hess | count]` layout, exactly as `download_slot` returns it.
        """
        if len(slots) < 1:
            raise Error("a download needs at least one slot")
        var cells = self.slot_cells()
        for i in range(len(slots)):
            self.pool.check_live(slots[i])
            for k in range(i):
                if slots[k] == slots[i]:
                    raise Error("a download may not list a slot twice")
        var out = List[Int32](capacity=len(slots) * cells)
        with self.out_dev.map_to_host() as host:
            var p = host.unsafe_ptr()
            for i in range(len(slots)):
                var base = slots[i] * cells
                for c in range(cells):
                    out.append(p.unsafe_load(base + c))
        return out^
