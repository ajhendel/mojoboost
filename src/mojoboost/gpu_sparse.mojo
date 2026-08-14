"""Device-side sparse histogram accumulation and entry partitioning.

The sparse counterpart of `histogram_gpu.mojo` + `gpu_active_rows.mojo`: a
CSC binned matrix stays device-resident for a whole training session, every
live node owns a contiguous *entry* window per feature as well as a
contiguous *row* range, and a node's histogram reads only its own stored
entries plus one sequential pass over its own rows.

Nothing here is wired into training and nothing enables it. There is no
`device="gpu"` path for sparse input, no automatic switch from the dense
builder, and no densification anywhere: a `SparseBinnedMatrix` goes to the
device as a compressed structure and comes back as a `Histogram`. See
`docs/GPU_SPARSE_CATEGORICAL_DESIGN.md` for the design and
`handoffs/performance_16_sparse_categorical_gpu.md` for the integration a
training path would need.

Two indexings, both device-resident
-----------------------------------
A node needs both, and they carry different information:

- its **row range** `rows[begin : end)`, the same `GpuActiveRows`
  permutation the dense path already maintains. It answers "which rows are
  in this node", which is what the node totals need.
- its **entry windows** `order[start_f : end_f)`, one per feature, over a
  permutation of the stored entries. It answers "which stored entries of
  feature f belong to this node", which is what accumulation needs.

The two are partitioned by the same per-row side mask at every split, so they
never disagree: an entry is in a node's window exactly when its row is in
that node's range. That invariant is what makes the subtraction below sound,
and `check_entry_row_consistency` is the host-side statement of it.

How a histogram gets built
--------------------------
Three kernels per node, one download:

1. `_sparse_totals_kernel` sums `round(g * scale)`, `round(h * scale)`, and
   1 over the node's row range, into three Int32 cells.
2. `_sparse_hist_kernel` accumulates the node's stored entries into their
   bins, `grid.x` an active feature and `grid.y` a tile of that feature's
   entry window, with a shared-memory Int32 partial per threadgroup folded
   into the output by global integer atomics.
3. `_sparse_default_fill_kernel` gives each active feature's default bin the
   leftover: the node's total minus everything the stored entries accounted
   for. That is the implicit zeros, in one subtraction, without ever
   visiting them.

Bit-identical to the dense GPU histogram
----------------------------------------
Stronger than what the CPU sparse path can claim. `histogram_sparse.mojo`
notes that its default bin, being derived by subtraction over Float64, is
only equal to the dense histogram's up to rounding. Here every accumulated
quantity is an exact fixed-point Int32 (see `histogram_gpu.mojo`), and the
same per-row quantization `round(g * scale)` feeds both the totals kernel and
the accumulation kernel, so

    default_bin_cell = total - sum(stored entries)
                     = sum over rows with no stored entry

exactly, with no rounding anywhere. Integer addition is associative, so the
atomics cannot reorder a result either. A sparse GPU histogram is therefore
**bit-identical** to the dense GPU histogram of the same data with the same
gradients and the same scale, not merely close to it, and that equality is
the sharpest correctness test this module has.

The one bound to check: `total` and `sum(stored)` are each within +/- 2^30 by
the scale's construction, and their difference is a sum over a subset of the
same rows, so it is within +/- 2^30 too. No intermediate leaves Int32.

Zeros, stored zeros, and missing
--------------------------------
Unchanged from `sparse.mojo`, because the structure holds the same numbers:

- an **absent** entry is the value 0.0 and lands in `default_bin[f]`, via the
  leftover, never by being visited;
- an **explicitly stored zero**, and any stored value that happens to bin to
  `default_bin[f]`, is accumulated as a stored entry into that same bin and
  then *excluded* from the leftover, so storing it or dropping it gives the
  identical histogram, bit for bit;
- a stored **NaN** binned into the feature's reserved missing bin is a real,
  non-default bin: it is accumulated like any other entry and routed by the
  node's learned `default_left`, never confused with an absent entry.

`zero_as_missing` is not implemented here, in `sparse.mojo`, or anywhere
else, and no argument accepts it.

How a split is applied
----------------------
A sparse split routes rows through the split feature's stored entries alone:
every row of the node takes the side of `default_bin[f]` unless it has a
stored entry for f. So the side is materialized first, one byte per row, and
then *both* index structures are partitioned by it:

1. `_sparse_side_default_kernel` writes the default side over the node's row
   range;
2. `_sparse_side_entries_kernel` overwrites the rows that do have a stored
   entry for the split feature, using `RowRouting`'s rule, categorical set
   membership included;
3. the row range is partitioned by `GpuActiveRows.enqueue_partition`,
   **unchanged and unforked**, by handing it the side buffer as a synthetic
   one-column binned matrix (feature 0, threshold bin 0, no missing bin: the
   rule `bin <= 0` then reads exactly as "side == left"). The sparse path
   therefore inherits the dense path's stable, deterministic, already
   validated partition rather than a second copy of it;
4. `_entry_partition_kernel` stably partitions every feature's entry window
   by the same mask, one threadgroup per feature.

Both partitions are stable, so a node's rows stay in the order the CPU
grower's row list holds them and a feature's entries stay in ascending row
order inside every window. The second property is what keeps the windows
mergeable and makes a child's window a sub-window of its parent's.

Bagging
-------
A bagged root gets its row range from the bag, but its entry windows start as
whole columns, and those include entries of rows the bag left out. Feeding
those to the accumulation while the totals cover the bag alone would make the
leftover wrong (it can go negative). `begin_tree` therefore runs one extra
partition at the root, by bag membership, and keeps the in-bag half. Cost is
one pass over `nnz`, once per tree, and the out-of-bag entries are then
unreachable from any live node.

What is deliberately not here
-----------------------------
- **No tiled reduction strategy.** Only `STRATEGY_ATOMIC` accumulation is
  implemented. The tiled strategy exists on the dense path to keep many row
  tiles from contending on the same output bins; the sparse path's tiles are
  entry tiles, which are far fewer per feature, and the hottest bin by far
  (the default one) is never touched by the accumulation at all -- it is
  filled once, by one thread, in a separate kernel. The partial-buffer
  variant is a measurement away, not a design change; the handoff says what
  it would look like.
- **No device split search.** `gpu_split_search.mojo` already scans a
  histogram on the device and does not care how the histogram was built, so
  a sparse histogram feeds it unchanged.
- **No `device="gpu"` for sparse input.** Not exposed, not routed to, not
  defaulted to.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import isfinite, round
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .categorical import CatBitset, CategoricalSpec, cat_empty
from .gpu_active_rows import GpuActiveRows, LeafRange, RowRouting
from .gpu_objectives_native import device_fixed_scale
from .gpu_sparse_layout import (
    DEFAULT_MAX_NODES,
    SPARSE_MAX_BINS,
    SparseDeviceLayout,
    SparseRangeTable,
    check_categorical_support,
    check_sparse_support,
    derive_entry_tiling,
)
from .gpu_tiling import (
    STRATEGY_ATOMIC,
    DeviceCaps,
    derive_block_threads,
    query_device_caps,
)
from .histogram import Histogram, _zeroed_f64, _zeroed_int
from .histogram_sparse import SparseNodeEntries
from .sparse import SparseBinnedMatrix

comptime MAX_BINS = SPARSE_MAX_BINS

# The scan buffers hold one Int32 per thread, so the block size the
# segmented partition can be launched with is bounded by the allocation
# rather than by the device maximum, exactly as in `gpu_active_rows.mojo`.
comptime SCAN_MAX_THREADS = 1024

# Side codes. 0 is left so the synthetic routing `bin <= 0` that drives the
# row partition reads as "side == left" (see the module docstring).
comptime SIDE_LEFT = UInt8(0)
comptime SIDE_RIGHT = UInt8(1)

# The three totals planes, in the order the kernels write them.
comptime TOT_GRAD = 0
comptime TOT_HESS = 1
comptime TOT_COUNT = 2


# --- Kernels --------------------------------------------------------------


def _zero_i32_kernel(
    buf: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
):
    """Zero an Int32 device range."""
    var i = global_idx.x
    if i < Int(n):
        buf[unsafe_offset=i] = Int32(0)


def _fill_u8_kernel(
    buf: MutPointer[UInt8, MutAnyOrigin],
    n: Int32,
    value: UInt8,
):
    """Fill a UInt8 device range with one value."""
    var i = global_idx.x
    if i < Int(n):
        buf[unsafe_offset=i] = value


def _iota_i32_kernel(
    buf: MutPointer[Int32, MutAnyOrigin],
    n: Int32,
):
    """Seed the entry permutation with the identity, which already groups
    entries by feature: that is what the CSC layout is."""
    var i = global_idx.x
    if i < Int(n):
        buf[unsafe_offset=i] = Int32(i)


def _seed_root_ranges_kernel(
    ranges: MutPointer[Int32, MutAnyOrigin],
    col_offsets: MutPointer[Int32, MutAnyOrigin],
    n_features: Int32,
):
    """Give node 0 every stored entry, one window per feature.

    The root's window for feature f is that feature's whole CSC column, so
    the seeding is a copy of the column offsets into the node-0 slice of the
    range table. No other node is touched: a node id is always written by the
    split that creates it before anything reads it.
    """
    var f = global_idx.x
    if f < Int(n_features):
        ranges[unsafe_offset = 2 * f] = col_offsets[unsafe_offset=f][0]
        ranges[unsafe_offset = 2 * f + 1] = col_offsets[
            unsafe_offset = f + 1
        ][0]


def _sparse_totals_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    totals: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """The node's fixed-point gradient, hessian, and row-count totals.

    One element per thread over the node's compacted row range, summed within
    the threadgroup by the same Hillis-Steele shared scan the partition uses
    (a scan rather than a tree reduction so the block size need not be a
    power of two), then one global atomic per plane per block.

    The quantization is `round(g * scale)`, the *same expression* the
    accumulation kernel applies to the same row, which is what makes the
    leftover exact rather than approximately exact.
    """
    var tid = thread_idx.x
    var nthreads = block_dim.x
    var j = block_idx.x * nthreads + tid
    var n = Int(count)

    var sg = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sh = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var sc = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var gq = Int32(0)
    var hq = Int32(0)
    var cq = Int32(0)
    if j < n:
        var r = Int(rows[unsafe_offset = Int(begin) + j][0])
        gq = Int32(round(grad[unsafe_offset=r][0] * g_scale))
        hq = Int32(round(hess[unsafe_offset=r][0] * h_scale))
        cq = Int32(1)
    sg[unsafe_offset=tid] = gq
    sh[unsafe_offset=tid] = hq
    sc[unsafe_offset=tid] = cq
    barrier()

    var offset = 1
    while offset < nthreads:
        var cg = Int32(0)
        var ch = Int32(0)
        var cc = Int32(0)
        if tid >= offset:
            cg = sg[unsafe_offset = tid - offset][0]
            ch = sh[unsafe_offset = tid - offset][0]
            cc = sc[unsafe_offset = tid - offset][0]
        barrier()
        if tid >= offset:
            sg[unsafe_offset=tid] = sg[unsafe_offset=tid][0] + cg
            sh[unsafe_offset=tid] = sh[unsafe_offset=tid][0] + ch
            sc[unsafe_offset=tid] = sc[unsafe_offset=tid][0] + cc
        barrier()
        offset += offset

    if tid == nthreads - 1:
        _ = Atomic.fetch_add(
            totals.unsafe_offset(TOT_GRAD), sg[unsafe_offset=tid][0]
        )
        _ = Atomic.fetch_add(
            totals.unsafe_offset(TOT_HESS), sh[unsafe_offset=tid][0]
        )
        _ = Atomic.fetch_add(
            totals.unsafe_offset(TOT_COUNT), sc[unsafe_offset=tid][0]
        )


def _sparse_hist_kernel(
    order: MutPointer[Int32, MutAnyOrigin],
    entry_row: MutPointer[Int32, MutAnyOrigin],
    entry_bin: MutPointer[UInt8, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    ranges: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    node: Int32,
    n_features: Int32,
    n_bins: Int32,
    hist_size: Int32,
    entries_per_tile: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """Accumulate one node's stored entries into their bins.

    `grid.x` is the active feature slot and `grid.y` a tile of that feature's
    entry window, so a device gets `n_slots * n_tiles` threadgroups. The
    window is read from the device range table rather than passed in, which
    is what lets the host launch from an upper bound on the window (see
    `SparseRangeTable.bound_split`) without changing a single accumulated
    value: a tile past the window's end simply finds nothing to do.

    Implicit zeros are not visited here at all. `_sparse_default_fill_kernel`
    puts them in afterwards, in one subtraction per feature.
    """
    var slot = Int(block_idx.x)
    var f = Int(feat_ids[unsafe_offset=slot][0])
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var hs = Int(hist_size)

    var rbase = 2 * (Int(node) * Int(n_features) + f)
    var lo = Int(ranges[unsafe_offset=rbase][0])
    var hi = Int(ranges[unsafe_offset = rbase + 1][0])

    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    var b = tid
    while b < nb:
        sg[unsafe_offset=b] = 0
        sh[unsafe_offset=b] = 0
        sc[unsafe_offset=b] = 0
        b += block_dim.x
    barrier()

    var tile_begin = lo + block_idx.y * Int(entries_per_tile)
    var tile_end = tile_begin + Int(entries_per_tile)
    if tile_end > hi:
        tile_end = hi

    var i = tile_begin + tid
    while i < tile_end:
        var e = Int(order[unsafe_offset=i][0])
        var r = Int(entry_row[unsafe_offset=e][0])
        var bin = Int(entry_bin[unsafe_offset=e])
        var gq = Int32(round(grad[unsafe_offset=r][0] * g_scale))
        var hq = Int32(round(hess[unsafe_offset=r][0] * h_scale))
        _ = Atomic.fetch_add(sg.unsafe_offset(bin), gq)
        _ = Atomic.fetch_add(sh.unsafe_offset(bin), hq)
        _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        i += block_dim.x
    barrier()

    var base = f * nb
    b = tid
    while b < nb:
        if sc[unsafe_offset=b][0] != 0:
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(base + b), sg[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(hs + base + b), sh[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_hist.unsafe_offset(2 * hs + base + b),
                sc[unsafe_offset=b][0],
            )
        b += block_dim.x


def _sparse_default_fill_kernel(
    out_hist: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    default_bin: MutPointer[UInt8, MutAnyOrigin],
    totals: MutPointer[Int32, MutAnyOrigin],
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
):
    """Give each active feature's default bin the node's implicit zeros.

    One thread per active feature slot: sum that feature's whole bin row in
    each plane, subtract it from the node total, and add the difference to
    the default bin's cell. The read-modify-write is deliberate -- the
    default bin may already hold stored entries that happened to bin there,
    and those must not be counted twice.

    Only active slots are touched, so a feature the caller excluded keeps a
    zero slice and sibling subtraction stays exact, exactly as on the dense
    and CPU sparse paths.
    """
    var slot = global_idx.x
    if slot >= Int(n_slots):
        return
    var f = Int(feat_ids[unsafe_offset=slot][0])
    var nb = Int(n_bins)
    var hs = Int(hist_size)
    var base = f * nb

    var sum_g = Int32(0)
    var sum_h = Int32(0)
    var sum_c = Int32(0)
    for b in range(nb):
        sum_g += out_hist[unsafe_offset = base + b][0]
        sum_h += out_hist[unsafe_offset = hs + base + b][0]
        sum_c += out_hist[unsafe_offset = 2 * hs + base + b][0]

    var db = base + Int(default_bin[unsafe_offset=f])
    out_hist[unsafe_offset=db] = (
        out_hist[unsafe_offset=db][0]
        + (totals[unsafe_offset=TOT_GRAD][0] - sum_g)
    )
    out_hist[unsafe_offset = hs + db] = (
        out_hist[unsafe_offset = hs + db][0]
        + (totals[unsafe_offset=TOT_HESS][0] - sum_h)
    )
    out_hist[unsafe_offset = 2 * hs + db] = (
        out_hist[unsafe_offset = 2 * hs + db][0]
        + (totals[unsafe_offset=TOT_COUNT][0] - sum_c)
    )


def _sparse_side_default_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    side: MutPointer[UInt8, MutAnyOrigin],
    begin: Int32,
    count: Int32,
    value: UInt8,
):
    """Write one side over every row of a node's range.

    Step one of applying a sparse split: every row takes the side of the
    split feature's default bin, and only the rows with a stored entry for
    that feature are corrected afterwards.
    """
    var j = global_idx.x
    if j < Int(count):
        var r = Int(rows[unsafe_offset = Int(begin) + j][0])
        side[unsafe_offset=r] = value


def _sparse_side_entries_kernel(
    order: MutPointer[Int32, MutAnyOrigin],
    entry_row: MutPointer[Int32, MutAnyOrigin],
    entry_bin: MutPointer[UInt8, MutAnyOrigin],
    ranges: MutPointer[Int32, MutAnyOrigin],
    side: MutPointer[UInt8, MutAnyOrigin],
    node: Int32,
    n_features: Int32,
    feature: Int32,
    threshold_bin: Int32,
    missing_bin: Int32,
    default_left: Int32,
    is_categorical: Int32,
    cat0: UInt64,
    cat1: UInt64,
    cat2: UInt64,
    cat3: UInt64,
):
    """Correct the side of every row that has a stored entry for the split
    feature in this node.

    The routing rule is `RowRouting.goes_left` spelled out: set membership
    for a categorical node (bin 0, the missing/unseen/dropped bin, is never
    a member, so those rows go right), the learned default direction for the
    reserved missing bin, and the inclusive threshold otherwise. Absent
    entries never reach this kernel, which is precisely why they keep the
    default side written by the previous one.
    """
    var f = Int(feature)
    var rbase = 2 * (Int(node) * Int(n_features) + f)
    var lo = Int(ranges[unsafe_offset=rbase][0])
    var hi = Int(ranges[unsafe_offset = rbase + 1][0])
    var i = lo + global_idx.x
    if i >= hi:
        return

    var e = Int(order[unsafe_offset=i][0])
    var r = Int(entry_row[unsafe_offset=e][0])
    var bin = Int32(entry_bin[unsafe_offset=e])

    var goes_left: Bool
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
        goes_left = ((word >> UInt64(Int(bin) & 63)) & 1) != 0
    elif bin == missing_bin:
        goes_left = default_left != 0
    else:
        goes_left = bin <= threshold_bin

    side[unsafe_offset=r] = SIDE_LEFT if goes_left else SIDE_RIGHT


def _entry_partition_kernel(
    order: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    entry_row: MutPointer[Int32, MutAnyOrigin],
    side: MutPointer[UInt8, MutAnyOrigin],
    ranges: MutPointer[Int32, MutAnyOrigin],
    mids: MutPointer[Int32, MutAnyOrigin],
    parent: Int32,
    left: Int32,
    right: Int32,
    n_features: Int32,
):
    """Stably partition every feature's entry window of `parent` by `side`.

    One threadgroup per feature, which is the same parallel dimension the
    whole feature-oriented design uses. Within a block:

    - pass one counts the left-going entries, chunk by chunk, carrying the
      running count in a register every thread computes identically;
    - pass two rewrites the window into `scratch`, a left-going entry at its
      global left rank and a right-going one after every left-going entry at
      its own rank. Both ranks are monotone in position, so both sides keep
      their relative order: the partition is stable, entries stay in
      ascending row order inside each child window, and the result cannot
      depend on scheduling;
    - pass three folds the rewritten window back over the parent's slots, so
      the windows of every other live node survive untouched.

    Thread 0 then publishes the midpoint into the range table for both
    children and into `mids`, which the host reads back to mirror the split.

    `left == parent` is allowed and is how the root's bag compaction reuses
    this kernel: the parent's window is read by every thread before the first
    barrier and rewritten only after the last, so the aliasing is safe. The
    host is what forbids it for an ordinary split.
    """
    var f = Int(block_idx.x)
    var tid = thread_idx.x
    var nthreads = block_dim.x
    var nf = Int(n_features)
    var pbase = 2 * (Int(parent) * nf + f)
    var lo = Int(ranges[unsafe_offset=pbase][0])
    var hi = Int(ranges[unsafe_offset = pbase + 1][0])
    var n = hi - lo

    var s = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    # --- pass one: how many go left ------------------------------------
    var n_left = Int32(0)
    var base = 0
    while base < n:
        var idx = base + tid
        var flag = Int32(0)
        if idx < n:
            var e = Int(order[unsafe_offset = lo + idx][0])
            var r = Int(entry_row[unsafe_offset=e][0])
            if side[unsafe_offset=r] == SIDE_LEFT:
                flag = Int32(1)
        s[unsafe_offset=tid] = flag
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
        n_left += s[unsafe_offset = nthreads - 1][0]
        # Every thread has read the chunk total; the next chunk may now
        # overwrite the shared buffer.
        barrier()
        base += nthreads

    # --- pass two: scatter into the scratch window ----------------------
    var l_written = 0
    var r_written = 0
    base = 0
    while base < n:
        var idx = base + tid
        var flag = Int32(0)
        var entry = Int32(0)
        if idx < n:
            entry = order[unsafe_offset = lo + idx][0]
            var r = Int(entry_row[unsafe_offset = Int(entry)][0])
            if side[unsafe_offset=r] == SIDE_LEFT:
                flag = Int32(1)
        s[unsafe_offset=tid] = flag
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

        # Inclusive minus own flag is this element's rank among the chunk's
        # left-going entries; `tid - p` is its rank among the right-going
        # ones, since padding threads sit above every valid tid.
        var p = Int(s[unsafe_offset=tid][0] - flag)
        var chunk_left = Int(s[unsafe_offset = nthreads - 1][0])
        var chunk_n = nthreads if base + nthreads <= n else n - base
        if idx < n:
            var dst: Int
            if flag != 0:
                dst = lo + l_written + p
            else:
                dst = lo + Int(n_left) + r_written + (tid - p)
            scratch[unsafe_offset=dst] = entry
        l_written += chunk_left
        r_written += chunk_n - chunk_left
        barrier()
        base += nthreads

    # --- pass three: fold the window back -------------------------------
    var i = tid
    while i < n:
        order[unsafe_offset = lo + i] = scratch[unsafe_offset = lo + i][0]
        i += nthreads
    barrier()

    if tid == 0:
        var mid = lo + Int(n_left)
        mids[unsafe_offset=f] = Int32(mid)
        var lbase = 2 * (Int(left) * nf + f)
        var rbase = 2 * (Int(right) * nf + f)
        ranges[unsafe_offset=lbase] = Int32(lo)
        ranges[unsafe_offset = lbase + 1] = Int32(mid)
        ranges[unsafe_offset=rbase] = Int32(mid)
        ranges[unsafe_offset = rbase + 1] = Int32(hi)


# --- Host reference model -------------------------------------------------


def side_mask_host(
    data: SparseBinnedMatrix,
    order: List[Int],
    node: SparseNodeEntries,
    rows: List[Int],
    routing: RowRouting,
) raises -> List[UInt8]:
    """The per-row side mask the two side kernels produce, on the host.

    Same rule, same arguments, one place: every row of the node takes the
    side of `default_bin[feature]`, then the rows with a stored entry for
    that feature in this node take the side of their own bin. Rows outside
    the node keep `SIDE_RIGHT`, which no partition ever reads.
    """
    routing.check(data.n_features, data.n_bins)
    var side = List[UInt8](capacity=data.n_rows)
    side.resize(data.n_rows, SIDE_RIGHT)
    var f = routing.feature
    var default_left = routing.goes_left(Int(data.default_bin[f]))
    var default_side = SIDE_LEFT if default_left else SIDE_RIGHT
    for i in range(len(rows)):
        var r = rows[i]
        if r < 0 or r >= data.n_rows:
            raise Error("row index out of range")
        side[r] = default_side
    for i in range(node.starts[f], node.ends[f]):
        var e = order[i]
        if e < 0 or e >= data.nnz():
            raise Error("entry index out of range")
        var goes_left = routing.goes_left(Int(data.bin[e]))
        side[data.row_index[e]] = SIDE_LEFT if goes_left else SIDE_RIGHT
    return side^


def partition_entries_host(
    mut order: List[Int],
    data: SparseBinnedMatrix,
    node: SparseNodeEntries,
    side: List[UInt8],
    mut left: SparseNodeEntries,
    mut right: SparseNodeEntries,
) raises:
    """`_entry_partition_kernel` on the host, for the same inputs.

    The reference the device partition is compared against. Serial and
    obviously stable, which is the point: the device version is a chunked
    block scan of the same permutation and has to agree with it index for
    index.
    """
    if len(side) != data.n_rows:
        raise Error("side mask length must equal n_rows")
    var scratch = List[Int](capacity=len(order))
    scratch.resize(len(order), 0)
    for f in range(data.n_features):
        var lo = node.starts[f]
        var hi = node.ends[f]
        var w = lo
        for i in range(lo, hi):
            var e = order[i]
            if side[data.row_index[e]] == SIDE_LEFT:
                scratch[w] = e
                w += 1
        var mid = w
        for i in range(lo, hi):
            var e = order[i]
            if side[data.row_index[e]] != SIDE_LEFT:
                scratch[w] = e
                w += 1
        for i in range(lo, hi):
            order[i] = scratch[i]
        left.starts[f] = lo
        left.ends[f] = mid
        right.starts[f] = mid
        right.ends[f] = hi


def check_entry_row_consistency(
    data: SparseBinnedMatrix,
    order: List[Int],
    node: SparseNodeEntries,
    rows: List[Int],
) raises:
    """The invariant the whole default-bin subtraction rests on.

    Every entry in the node's windows must belong to a row in the node's row
    range, and no row may own two entries for one feature. The first fails if
    the two index structures were partitioned by different masks; the second
    fails only on a malformed matrix, which `CscMatrix.validate` already
    rejects, and is re-checked here because a violation would silently
    inflate a bin's count and deflate the default bin's.
    """
    var member = List[UInt8](capacity=data.n_rows)
    member.resize(data.n_rows, 0)
    for i in range(len(rows)):
        if rows[i] < 0 or rows[i] >= data.n_rows:
            raise Error("row index out of range")
        if member[rows[i]] != 0:
            raise Error("duplicate row index in the node's row range")
        member[rows[i]] = 1
    # One stamp array reused across features, so the check costs
    # O(n_rows + nnz_in_node) rather than O(n_features * n_rows).
    var seen = List[Int](capacity=data.n_rows)
    seen.resize(data.n_rows, -1)
    for f in range(data.n_features):
        for i in range(node.starts[f], node.ends[f]):
            var e = order[i]
            if e < 0 or e >= data.nnz():
                raise Error("entry index out of range")
            var r = data.row_index[e]
            if member[r] == 0:
                raise Error(
                    "node holds a stored entry for a row outside its range"
                )
            if seen[r] == f:
                raise Error("two stored entries for one (row, feature)")
            seen[r] = f


# --- Device-resident sparse builder --------------------------------------


struct GpuSparseHistogramBuilder(Movable):
    """Device-resident sparse histogram builder and node partitioner.

    Construct once per training session, `upload_gradients` once per boosting
    round, `begin_tree` + `build_leaf`/`apply_split` per tree. The lifecycle
    is deliberately the same as `GpuHistogramBuilder`'s, so a grower that
    already drives the dense builder drives this one with the same call
    sequence and only the construction differs.

    The last node id (`max_nodes - 1`) is reserved: `begin_tree` parks a
    bagged root's out-of-bag entries there. A split may not use it.
    """

    var ctx: DeviceContext
    # The row-side index, shared with the dense path unchanged.
    var rows: GpuActiveRows
    # The compressed structure: entry -> (row, bin), feature -> entry range.
    var entry_row_dev: DeviceBuffer[DType.int32]
    var entry_bin_dev: DeviceBuffer[DType.uint8]
    var col_dev: DeviceBuffer[DType.int32]
    var default_dev: DeviceBuffer[DType.uint8]
    # The entry permutation, its scatter destination, and the per-(node,
    # feature) windows into it.
    var order_dev: DeviceBuffer[DType.int32]
    var scratch_dev: DeviceBuffer[DType.int32]
    var ranges_dev: DeviceBuffer[DType.int32]
    var mids_dev: DeviceBuffer[DType.int32]
    # One byte per row: the side the current split routes it to.
    var side_dev: DeviceBuffer[DType.uint8]
    var grad_dev: DeviceBuffer[DType.float32]
    var hess_dev: DeviceBuffer[DType.float32]
    var feat_dev: DeviceBuffer[DType.int32]
    # [grad | hess | count], each n_features * n_bins entries.
    var out_dev: DeviceBuffer[DType.int32]
    # The node's fixed-point totals, one cell per plane.
    var totals_dev: DeviceBuffer[DType.int32]
    var stage_g: HostBuffer[DType.float32]
    var stage_h: HostBuffer[DType.float32]
    var host_out: HostBuffer[DType.int32]
    var host_mids: HostBuffer[DType.int32]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var nnz: Int
    var max_nodes: Int
    var col_offsets: List[Int]
    var default_bin: List[Int]
    var missing_bin: List[Int]
    var cats: CategoricalSpec
    var active: List[Int]
    var caps: DeviceCaps
    var block_threads: Int
    var layout: SparseDeviceLayout
    var windows: SparseRangeTable
    var g_scale: Float64
    var h_scale: Float64
    var has_gradients: Bool
    var defer_ranges: Bool
    """When set, a split does not download the per-feature midpoints and the
    host keeps the parent's windows as upper bounds for both children. The
    device windows stay exact either way, so histograms are unchanged; only
    the launch geometry is over-provisioned. Off by default, because the
    over-provisioning compounds with depth."""

    def __init__(
        out self,
        data: SparseBinnedMatrix,
        max_nodes: Int = DEFAULT_MAX_NODES,
    ) raises:
        """Upload `data` and size every buffer the session will use.

        Opens a private `DeviceContext`; the overload below builds on a
        caller's context so this can share a queue with other device work.
        """
        var ctx = DeviceContext()
        var caps = query_device_caps(ctx)
        self = Self(ctx, caps, data, max_nodes)

    def __init__(
        out self,
        ctx: DeviceContext,
        caps: DeviceCaps,
        data: SparseBinnedMatrix,
        max_nodes: Int = DEFAULT_MAX_NODES,
    ) raises:
        """Build on a caller-supplied context and its queried capabilities."""
        if max_nodes < 2:
            raise Error(
                "max_nodes must be at least 2; the last id is reserved"
            )
        var nnz = data.nnz()
        check_sparse_support(
            caps, data.n_rows, data.n_features, data.n_bins, nnz
        )
        check_categorical_support(data.cats, data.n_features, data.n_bins)
        if len(data.row_index) != nnz:
            raise Error("row_index and bin must have equal length")
        if len(data.col_offsets) != data.n_features + 1:
            raise Error("col_offsets must have length n_features + 1")
        if data.col_offsets[0] != 0:
            raise Error("col_offsets must start at 0")
        if data.col_offsets[data.n_features] != nnz:
            raise Error("col_offsets must end at nnz")
        if len(data.default_bin) != data.n_features:
            raise Error("default_bin must have one entry per feature")

        self.ctx = ctx
        self.n_rows = data.n_rows
        self.n_features = data.n_features
        self.n_bins = data.n_bins
        self.nnz = nnz
        self.max_nodes = max_nodes
        self.caps = caps.copy()
        self.cats = data.cats.copy()
        self.missing_bin = data.missing_bin.copy()
        self.layout = SparseDeviceLayout.of(data, max_nodes)
        self.windows = SparseRangeTable(data.n_features, nnz)
        self.g_scale = 1.0
        self.h_scale = 1.0
        self.has_gradients = False
        self.defer_ranges = False

        var threads = derive_block_threads(caps)
        if threads > SCAN_MAX_THREADS:
            threads = SCAN_MAX_THREADS
        self.block_threads = threads

        self.active = List[Int](capacity=data.n_features)
        for f in range(data.n_features):
            self.active.append(f)
        self.col_offsets = data.col_offsets.copy()
        self.default_bin = List[Int](capacity=data.n_features)
        for f in range(data.n_features):
            var d = Int(data.default_bin[f])
            if d < 0 or d >= data.n_bins:
                raise Error("default bin out of range")
            self.default_bin.append(d)

        # Every index the kernels dereference is checked once, here, rather
        # than guarded per access: a stored entry naming a row or a bin
        # outside the matrix would read out of bounds on the device, where
        # there is nothing to catch it.
        var widened_rows = List[Int32](capacity=nnz)
        for i in range(nnz):
            var r = data.row_index[i]
            if r < 0 or r >= data.n_rows:
                raise Error("stored entry row index out of range")
            if Int(data.bin[i]) >= data.n_bins:
                raise Error("stored entry bin out of range")
            widened_rows.append(Int32(r))
        var widened_cols = List[Int32](capacity=data.n_features + 1)
        for f in range(data.n_features + 1):
            widened_cols.append(Int32(data.col_offsets[f]))

        var hist_size = data.n_features * data.n_bins
        var range_cells = 2 * max_nodes * data.n_features
        # A zero-length device buffer is not portable, so an empty matrix
        # still gets a one-element placeholder for the entry buffers.
        var entry_cells = nnz if nnz > 0 else 1

        self.rows = GpuActiveRows(
            ctx, data.n_rows, data.n_features, data.n_bins, self.caps
        )
        self.entry_row_dev = ctx.enqueue_create_buffer[DType.int32](
            entry_cells
        )
        self.entry_bin_dev = ctx.enqueue_create_buffer[DType.uint8](
            entry_cells
        )
        self.col_dev = ctx.enqueue_create_buffer[DType.int32](
            data.n_features + 1
        )
        self.default_dev = ctx.enqueue_create_buffer[DType.uint8](
            data.n_features
        )
        self.order_dev = ctx.enqueue_create_buffer[DType.int32](entry_cells)
        self.scratch_dev = ctx.enqueue_create_buffer[DType.int32](entry_cells)
        self.ranges_dev = ctx.enqueue_create_buffer[DType.int32](range_cells)
        self.mids_dev = ctx.enqueue_create_buffer[DType.int32](
            data.n_features
        )
        self.side_dev = ctx.enqueue_create_buffer[DType.uint8](data.n_rows)
        self.grad_dev = ctx.enqueue_create_buffer[DType.float32](data.n_rows)
        self.hess_dev = ctx.enqueue_create_buffer[DType.float32](data.n_rows)
        self.feat_dev = ctx.enqueue_create_buffer[DType.int32](
            data.n_features
        )
        self.out_dev = ctx.enqueue_create_buffer[DType.int32](3 * hist_size)
        self.totals_dev = ctx.enqueue_create_buffer[DType.int32](3)
        self.stage_g = ctx.enqueue_create_host_buffer[DType.float32](
            data.n_rows
        )
        self.stage_h = ctx.enqueue_create_host_buffer[DType.float32](
            data.n_rows
        )
        self.host_out = ctx.enqueue_create_host_buffer[DType.int32](
            3 * hist_size
        )
        self.host_mids = ctx.enqueue_create_host_buffer[DType.int32](
            data.n_features
        )

        if nnz > 0:
            self.ctx.enqueue_copy(
                dst_buf=self.entry_row_dev,
                src_ptr=widened_rows.unsafe_ptr(),
            )
            self.ctx.enqueue_copy(
                dst_buf=self.entry_bin_dev, src_ptr=data.bin.unsafe_ptr()
            )
        self.ctx.enqueue_copy(
            dst_buf=self.col_dev, src_ptr=widened_cols.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.default_dev, src_ptr=data.default_bin.unsafe_ptr()
        )
        # The copies read host memory owned by locals and by the caller's
        # `data`, so they have to complete before the constructor returns.
        self.ctx.synchronize()

        # Stale windows from a previous tree are never read (a node id is
        # written by the split that creates it before anything reads it), but
        # the table is zeroed once so a wiring mistake reads an empty window
        # rather than another tree's.
        self._enqueue_zero_i32(self.ranges_dev.unsafe_ptr(), range_cells)

        # Every feature is active until `set_features` narrows it.
        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(data.n_features):
                dst.unsafe_store(f, Int32(f))

    # --- small launch helpers --------------------------------------------

    def _blocks(self, n: Int) -> Int:
        """Threadgroups covering `n` elements at this builder's block size,
        never zero: a launch of no blocks is not portable."""
        var b = (n + self.block_threads - 1) // self.block_threads
        return b if b > 0 else 1

    def _enqueue_zero_i32[
        buf_origin: MutOrigin, //
    ](mut self, buf: MutPointer[Int32, buf_origin], n: Int) raises:
        if n <= 0:
            return
        self.ctx.enqueue_function[_zero_i32_kernel](
            buf,
            Int32(n),
            grid_dim=self._blocks(n),
            block_dim=self.block_threads,
        )

    def strategy(self) -> Int:
        """The accumulation strategy. Only the atomic one is implemented on
        the sparse path; see the module docstring for why, and the handoff
        for what a tiled variant would take."""
        return STRATEGY_ATOMIC

    def synchronize(mut self) raises:
        self.ctx.synchronize()

    def entry_window(self, node: Int) raises -> SparseNodeEntries:
        """The host's belief about `node`'s per-feature entry windows. Exact
        unless `defer_ranges` was set when its parent split."""
        return self.windows.get(node)

    def row_range(self, node: Int) raises -> LeafRange:
        return self.rows.range_of(node)

    # --- feature subsampling ---------------------------------------------

    def set_features(mut self, features: List[Int]) raises:
        """Restrict later builds to `features` (global feature ids).

        Only the launch grid narrows; the dataset stays whole and
        device-resident. Slices of features not listed stay zero in every
        histogram built afterwards -- including their default bins, since the
        completion kernel walks active slots only -- which is what keeps
        sibling subtraction exact as long as one tree keeps one feature set.

        The *entry partition* is deliberately not narrowed: every feature's
        windows are maintained at every split, whether or not that feature is
        active, because a later node may activate it and a window that
        skipped a split would no longer be a sub-window of its parent's.
        """
        if len(features) == 0:
            raise Error("active feature set must not be empty")
        if len(features) > self.n_features:
            raise Error("active feature set is larger than n_features")
        var changed = len(features) != len(self.active)
        for i in range(len(features)):
            if features[i] < 0 or features[i] >= self.n_features:
                raise Error("feature index out of range")
            if not changed and self.active[i] != features[i]:
                changed = True
        if not changed:
            return
        self.active = features.copy()
        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(features)):
                dst.unsafe_store(i, Int32(features[i]))

    # --- gradients --------------------------------------------------------

    def stage_gradients(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises:
        """Convert this round's gradients and hessians into the device's
        Float32 in pinned host memory, and derive the fixed-point scales.

        The scales come from `gpu_objectives_native.device_fixed_scale`, the
        same function the dense builder's device-side objective path uses, so
        a sparse and a dense builder fed the same gradients quantize
        identically. That is what the bit-identity claim in the module
        docstring depends on.
        """
        if len(grad) != self.n_rows or len(hess) != self.n_rows:
            raise Error("gradient/hessian length must equal n_rows")
        self.has_gradients = False

        var g_scale = device_fixed_scale(_magnitude_sum(grad))
        var h_scale = device_fixed_scale(_magnitude_sum(hess))
        self.g_scale = Float64(g_scale)
        self.h_scale = Float64(h_scale)

        # Any copy still reading the staging buffers has to finish before
        # they are overwritten.
        self.ctx.synchronize()
        var dst_g = self.stage_g.unsafe_ptr()
        var dst_h = self.stage_h.unsafe_ptr()
        var src_g = grad.unsafe_ptr()
        var src_h = hess.unsafe_ptr()
        for r in range(self.n_rows):
            dst_g.unsafe_store(r, Float32(src_g.unsafe_load(r)))
            dst_h.unsafe_store(r, Float32(src_h.unsafe_load(r)))

    def upload_staged(mut self) raises:
        self.ctx.enqueue_copy(
            dst_buf=self.grad_dev, src_ptr=self.stage_g.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.hess_dev, src_ptr=self.stage_h.unsafe_ptr()
        )
        self.has_gradients = True

    def upload_gradients(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises:
        """Upload this round's per-row gradients and hessians (once per
        boosting round, not per node)."""
        self.stage_gradients(grad, hess)
        self.upload_staged()

    # --- per-tree setup ---------------------------------------------------

    def begin_tree(mut self, bag: List[Int] = []) raises:
        """Seed the tree's active rows and entry windows at node 0.

        Unbagged, the entry permutation is the identity (which already groups
        entries by feature, since that is the CSC layout) and the root's
        window for feature f is that feature's whole column.

        Bagged, the root's rows are the bag, and the root's windows are then
        compacted to the entries whose rows are in the bag. Without that
        compaction the accumulation would count out-of-bag entries that the
        node totals do not cover, and the default bin, being the leftover,
        would come out short by exactly that much -- possibly negative. The
        compaction is one extra pass over `nnz` per tree; the out-of-bag
        entries are parked in the reserved node id and are unreachable from
        every live node afterwards.
        """
        self.rows.begin_tree(bag)
        if self.nnz > 0:
            self.ctx.enqueue_function[_iota_i32_kernel](
                self.order_dev.unsafe_ptr(),
                Int32(self.nnz),
                grid_dim=self._blocks(self.nnz),
                block_dim=self.block_threads,
            )
        self.ctx.enqueue_function[_seed_root_ranges_kernel](
            self.ranges_dev.unsafe_ptr(),
            self.col_dev.unsafe_ptr(),
            Int32(self.n_features),
            grid_dim=self._blocks(self.n_features),
            block_dim=self.block_threads,
        )
        self.windows.reset_root_offsets(self.col_offsets)
        if len(bag) > 0:
            self._compact_root_to_bag(len(bag))

    def _compact_root_to_bag(mut self, n_bag: Int) raises:
        """Drop the root's out-of-bag entries by partitioning its windows on
        bag membership and keeping the in-bag half."""
        self.ctx.enqueue_function[_fill_u8_kernel](
            self.side_dev.unsafe_ptr(),
            Int32(self.n_rows),
            SIDE_RIGHT,
            grid_dim=self._blocks(self.n_rows),
            block_dim=self.block_threads,
        )
        self.ctx.enqueue_function[_sparse_side_default_kernel](
            self.rows.rows_dev.unsafe_ptr(),
            self.side_dev.unsafe_ptr(),
            Int32(0),
            Int32(n_bag),
            SIDE_LEFT,
            grid_dim=self._blocks(n_bag),
            block_dim=self.block_threads,
        )
        self._enqueue_entry_partition(0, 0, self.max_nodes - 1)
        self.windows.compact_root(self._download_mids())

    # --- histograms -------------------------------------------------------

    def enqueue_leaf(mut self, leaf: Int) raises:
        """Enqueue the three kernels building `leaf`'s histogram. Does not
        transfer or synchronize."""
        if not self.has_gradients:
            raise Error("call upload_gradients before build_leaf")
        if leaf < 0 or leaf >= self.max_nodes:
            raise Error("node id out of range")

        var hist_size = self.n_features * self.n_bins
        self._enqueue_zero_i32(self.out_dev.unsafe_ptr(), 3 * hist_size)
        self._enqueue_zero_i32(self.totals_dev.unsafe_ptr(), 3)

        var window = self.rows.range_of(leaf)
        if window.count() > 0:
            self.ctx.enqueue_function[_sparse_totals_kernel](
                self.rows.rows_dev.unsafe_ptr(),
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
                self.totals_dev.unsafe_ptr(),
                Int32(window.begin),
                Int32(window.count()),
                Float32(self.g_scale),
                Float32(self.h_scale),
                grid_dim=self._blocks(window.count()),
                block_dim=self.block_threads,
            )

        var n_slots = len(self.active)
        var max_entries = self.windows.max_entries(leaf, self.active)
        if max_entries > 0:
            var tiling = derive_entry_tiling(
                self.caps,
                max_entries,
                n_slots,
                self.n_bins,
                STRATEGY_ATOMIC,
            )
            self.ctx.enqueue_function[_sparse_hist_kernel](
                self.order_dev.unsafe_ptr(),
                self.entry_row_dev.unsafe_ptr(),
                self.entry_bin_dev.unsafe_ptr(),
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                self.ranges_dev.unsafe_ptr(),
                self.out_dev.unsafe_ptr(),
                Int32(leaf),
                Int32(self.n_features),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(tiling.rows_per_tile),
                Float32(self.g_scale),
                Float32(self.h_scale),
                grid_dim=(n_slots, tiling.n_tiles),
                block_dim=tiling.block_threads,
            )

        # Runs whatever the entry count was: a node with no stored entries at
        # all is a node whose every row is an implicit zero, and the default
        # bins still have to receive them.
        self.ctx.enqueue_function[_sparse_default_fill_kernel](
            self.out_dev.unsafe_ptr(),
            self.feat_dev.unsafe_ptr(),
            self.default_dev.unsafe_ptr(),
            self.totals_dev.unsafe_ptr(),
            Int32(n_slots),
            Int32(self.n_bins),
            Int32(hist_size),
            grid_dim=self._blocks(n_slots),
            block_dim=self.block_threads,
        )

    def download_raw(mut self) raises:
        """Copy the fixed-point histogram into pinned host memory and wait.
        One host synchronization per node, not one per plane."""
        self.ctx.enqueue_copy(
            dst_ptr=self.host_out.unsafe_ptr(), src_buf=self.out_dev
        )
        self.ctx.synchronize()

    def histogram_from_host(self) raises -> Histogram:
        """Convert the downloaded fixed-point planes into the Float64
        `Histogram`. Host work only; call after `download_raw`."""
        var hist_size = self.n_features * self.n_bins
        var g = _zeroed_f64(hist_size)
        var h = _zeroed_f64(hist_size)
        var c = _zeroed_int(hist_size)
        var g_inv = 1.0 / self.g_scale
        var h_inv = 1.0 / self.h_scale
        var gp = g.unsafe_ptr()
        var hp = h.unsafe_ptr()
        var cp = c.unsafe_ptr()
        var src = self.host_out.unsafe_ptr()
        for i in range(hist_size):
            gp.unsafe_store(i, Float64(src.unsafe_load(i)) * g_inv)
            hp.unsafe_store(i, Float64(src.unsafe_load(hist_size + i)) * h_inv)
            cp.unsafe_store(i, Int(src.unsafe_load(2 * hist_size + i)))
        return Histogram(g^, h^, c^, self.n_features, self.n_bins)

    def build_leaf(mut self, leaf: Int) raises -> Histogram:
        """Build the histogram of the rows currently assigned to `leaf`, over
        the currently active feature set. The returned histogram always has
        the dataset's full `n_features * n_bins` shape; inactive features'
        slices are zero."""
        self.enqueue_leaf(leaf)
        self.download_raw()
        return self.histogram_from_host()

    def build(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises -> Histogram:
        """Full-dataset sparse histogram on the GPU."""
        self.upload_gradients(grad, hess)
        self.begin_tree()
        return self.build_leaf(0)

    # --- splits -----------------------------------------------------------

    def _enqueue_entry_partition(
        mut self, parent: Int, left: Int, right: Int
    ) raises:
        self.ctx.enqueue_function[_entry_partition_kernel](
            self.order_dev.unsafe_ptr(),
            self.scratch_dev.unsafe_ptr(),
            self.entry_row_dev.unsafe_ptr(),
            self.side_dev.unsafe_ptr(),
            self.ranges_dev.unsafe_ptr(),
            self.mids_dev.unsafe_ptr(),
            Int32(parent),
            Int32(left),
            Int32(right),
            Int32(self.n_features),
            grid_dim=self.n_features,
            block_dim=self.block_threads,
        )

    def _download_mids(mut self) raises -> List[Int]:
        """The per-feature midpoints of the last enqueued entry partition.
        Synchronizes: this is the one host round trip a sparse split costs
        that a dense split does not, and `defer_ranges` is what trades it for
        over-provisioned launch geometry."""
        self.ctx.enqueue_copy(
            dst_ptr=self.host_mids.unsafe_ptr(), src_buf=self.mids_dev
        )
        self.ctx.synchronize()
        var src = self.host_mids.unsafe_ptr()
        var out = List[Int](capacity=self.n_features)
        for f in range(self.n_features):
            out.append(Int(src.unsafe_load(f)))
        return out^

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
        """Reassign `parent`'s rows *and* stored entries to `left`/`right`.

        The row half is `GpuActiveRows`' own partition, driven by the side
        mask through the synthetic one-column routing described in the module
        docstring, so it is the same stable partition the dense path uses and
        it produces the same row order. The entry half is the segmented
        partition, by the same mask.

        `expected_left` is the left row count the caller already knows from
        the parent histogram's integer counts; -1 downloads the device's own,
        which synchronizes.
        """
        if feature < 0 or feature >= self.n_features:
            raise Error("split feature out of range")
        if parent < 0 or left < 0 or right < 0:
            raise Error("node ids must be nonnegative")
        if (
            left >= self.max_nodes - 1
            or right >= self.max_nodes - 1
            or parent >= self.max_nodes - 1
        ):
            raise Error(
                "node id is at or past the reserved id; raise max_nodes"
            )
        if left == parent or right == parent or left == right:
            raise Error(
                "child node ids must differ from the parent and each other"
            )

        var routing: RowRouting
        if is_categorical:
            routing = RowRouting.categorical(feature, cat_bitset)
        else:
            routing = RowRouting.numerical(
                feature, threshold_bin, missing_bin, default_left
            )
        routing.check(self.n_features, self.n_bins)

        var rng = self.rows.range_of(parent)
        var window = self.windows.get(parent)

        # Every row of the node takes the side of the split feature's default
        # bin -- that is what an absent entry routes as, because an absent
        # entry *is* the value 0.0 -- and only the rows with a stored entry
        # for this feature are corrected below.
        var goes_left = routing.goes_left(self.default_bin[feature])
        if rng.count() > 0:
            self.ctx.enqueue_function[_sparse_side_default_kernel](
                self.rows.rows_dev.unsafe_ptr(),
                self.side_dev.unsafe_ptr(),
                Int32(rng.begin),
                Int32(rng.count()),
                SIDE_LEFT if goes_left else SIDE_RIGHT,
                grid_dim=self._blocks(rng.count()),
                block_dim=self.block_threads,
            )
        var n_entries = window.ends[feature] - window.starts[feature]
        if n_entries > 0:
            var cat = routing.cat_bitset
            self.ctx.enqueue_function[_sparse_side_entries_kernel](
                self.order_dev.unsafe_ptr(),
                self.entry_row_dev.unsafe_ptr(),
                self.entry_bin_dev.unsafe_ptr(),
                self.ranges_dev.unsafe_ptr(),
                self.side_dev.unsafe_ptr(),
                Int32(parent),
                Int32(self.n_features),
                Int32(feature),
                Int32(routing.threshold_bin),
                Int32(routing.missing_bin),
                Int32(1) if routing.default_left else Int32(0),
                Int32(1) if routing.is_categorical else Int32(0),
                cat[0],
                cat[1],
                cat[2],
                cat[3],
                grid_dim=self._blocks(n_entries),
                block_dim=self.block_threads,
            )

        # The side buffer is a synthetic one-column binned matrix: feature 0,
        # inclusive threshold bin 0, no missing bin, so `bin <= 0` reads as
        # "side == SIDE_LEFT". That lets the row partition be the dense
        # path's, unforked.
        _ = self.rows.partition(
            self.side_dev.unsafe_ptr(),
            parent,
            left,
            right,
            RowRouting.numerical(0, 0, -1, False),
            expected_left,
        )

        self._enqueue_entry_partition(parent, left, right)
        if self.defer_ranges:
            self.windows.bound_split(parent, left, right)
        else:
            self.windows.split(parent, left, right, self._download_mids())


def _magnitude_sum(values: List[Float64]) raises -> Float64:
    """Sum of magnitudes, which bounds every partial sum of the scaled
    values and is therefore what the fixed-point scale is derived from."""
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    if not isfinite(total):
        raise Error("gradients and hessians must be finite")
    return total


def build_histogram_gpu_sparse(
    data: SparseBinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
) raises -> Histogram:
    """One-shot sparse GPU histogram (uploads the matrix every call; use
    `GpuSparseHistogramBuilder` for repeated builds on one dataset)."""
    var builder = GpuSparseHistogramBuilder(data)
    return builder.build(grad, hess)
