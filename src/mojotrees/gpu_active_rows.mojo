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
from std.gpu import block_dim, block_idx, global_idx, thread_idx
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
from .parallel import _env_int
from .split import SplitInfo

# Row indices and leaf ids cross into the kernels as Int32.
comptime MAX_ROWS = Int(Int32.MAX)

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
):
    """Stage one of the stable partition: flag each row of the range and scan
    the flags within each threadgroup.

    `offsets[j]` becomes the number of left-going rows before `j` *inside
    j's own block*, shifted left one bit, with j's own flag in the low bit;
    `block_sums[b]` is the number of left-going rows in block b. Carrying the
    flag in the offset word is what lets the scatter route the row without
    gathering its bin a second time: the bin loads are the one random-access
    read of the partition (`bins[feature * n_rows + row]`, one byte from a
    fresh cache line per row once the leaf's rows are sparse in the
    dataset), so reading each once instead of twice halves the partition's
    gathered traffic, and the packed word costs nothing the plain prefix did
    not. The prefix is bounded by the block size, so the shift cannot
    overflow. Threads past the end of the range contribute a zero flag, so a
    partial tail block still sums correctly. The scan is the plain
    Hillis-Steele shared-memory scan: two barriers per doubling step, with
    the read separated from the write so no thread reads a slot another has
    already advanced.

    This is the *fallback* arm. `_flag_scan_prim_kernel` is the default and
    scans with `block.prefix_sum` instead; see the module docstring for which
    rule that repeals and why. This kernel is kept rather than deleted
    because it reads its width from `block_dim.x` and so serves any
    threadgroup width, including the ones no primitive instantiation exists
    for, and because a benchmark wanting both arms in one process needs both
    to be present. It produces the identical offsets and block sums.
    """
    var tid = thread_idx.x
    var nthreads = block_dim.x
    var j = block_idx.x * nthreads + tid
    var n = Int(count)
    var col = Int(feature) * Int(n_rows)

    var s = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var flag = Int32(0)
    if j < n:
        var row = rows[unsafe_offset = Int(begin) + j][0]
        var bin = Int32(bins[unsafe_offset = col + Int(row)])
        if _row_goes_left(
            bin,
            threshold_bin,
            missing_bin,
            default_left,
            is_categorical,
            cat0,
            cat1,
            cat2,
            cat3,
        ):
            flag = Int32(1)
    s[unsafe_offset=tid] = flag
    barrier()

    # Every thread runs the same number of steps (the bound is the uniform
    # block size), so the barriers are reached by the whole threadgroup.
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
        # Inclusive minus own flag is the exclusive prefix; the flag rides
        # in the low bit for the scatter.
        offsets[unsafe_offset=j] = (
            (s[unsafe_offset=tid][0] - flag) * Int32(2)
        ) + flag
    if tid == nthreads - 1:
        block_sums[unsafe_offset = block_idx.x] = s[unsafe_offset=tid][0]


def _scan_block_sums_kernel(
    block_sums: MutPointer[Int32, MutAnyOrigin],
    total: MutPointer[Int32, MutAnyOrigin],
    n_blocks: Int32,
):
    """Stage two: turn the per-block left counts into per-block starting
    offsets, and write the range's total left count.

    One threadgroup walks the block sums in chunks of its own size, scanning
    each chunk in shared memory and carrying the running total across chunks
    in a register every thread computes identically. Sequential over exact
    integers in a fixed order, so the result cannot depend on scheduling, and
    it needs no second level of launches however many blocks stage one used.
    """
    var tid = thread_idx.x
    var nthreads = block_dim.x
    var nb = Int(n_blocks)

    var s = stack_allocation[
        SCAN_MAX_THREADS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var carry = Int32(0)
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

        if idx < nb:
            block_sums[unsafe_offset=idx] = (
                carry + s[unsafe_offset=tid][0] - own
            )
        carry += s[unsafe_offset = nthreads - 1][0]
        # The chunk total has been read by every thread; the next iteration
        # may now overwrite the shared buffer.
        barrier()
        base += nthreads

    if tid == 0:
        total[unsafe_offset=0] = carry


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
):
    """Stage one, scanned with `block.prefix_sum` instead of by hand.

    Everything a later stage reads is byte for byte what `_flag_scan_kernel`
    writes: `offsets[j]` is still `(exclusive_prefix_within_j's_block << 1) |
    flag`, and `block_sums[b]` is still block b's left count, so
    `_scatter_kernel` is shared between the two arms unchanged. The
    difference is only how the intra-block scan is computed.
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
    outside the `j < n` guard, which is also what keeps a partial tail block
    summing correctly.
    """
    var tid = thread_idx.x
    var j = block_idx.x * block_size + tid
    var n = Int(count)
    var col = Int(feature) * Int(n_rows)

    var flag = Int32(0)
    if j < n:
        var row = rows[unsafe_offset = Int(begin) + j][0]
        var bin = Int32(bins[unsafe_offset = col + Int(row)])
        if _row_goes_left(
            bin,
            threshold_bin,
            missing_bin,
            default_left,
            is_categorical,
            cat0,
            cat1,
            cat2,
            cat3,
        ):
            flag = Int32(1)

    var inclusive = block_prefix_sum[block_size=block_size, exclusive=False](
        flag
    )

    if j < n:
        offsets[unsafe_offset=j] = ((inclusive - flag) * Int32(2)) + flag
    if tid == block_size - 1:
        block_sums[unsafe_offset = block_idx.x] = inclusive


def _scan_block_sums_prim_kernel[block_size: Int](
    block_sums: MutPointer[Int32, MutAnyOrigin],
    total: MutPointer[Int32, MutAnyOrigin],
    n_blocks: Int32,
):
    """Stage two on the primitive arm. Same output as
    `_scan_block_sums_kernel`: the block sums become block starting offsets
    in place and the range's total left count lands in `total`.

    One threadgroup still walks the block sums in chunks of its own width and
    still carries the running total across chunks in a register, but the
    per-chunk scan is `block.prefix_sum` and the chunk total reaches every
    thread through `block.broadcast[block_size=block_size](inclusive,
    src_thread=block_size - 1)` rather than through a shared slot every
    thread re-reads. Signature, again checked against the toolchain:
    `block.broadcast[dtype: DType, width: SIMDLength, //, *, block_size:
    Int](val: SIMD[dtype, width], src_thread: Int = 0) -> SIMD[dtype,
    width]`. Broadcasting from the last thread is what keeps `carry`
    bit-identical on every thread, which is what makes the chunk-to-chunk
    carry deterministic; summing over exact Int32 in a fixed chunk order, it
    cannot depend on scheduling.

    The trailing `barrier()` is deliberate and is not the primitive's job to
    supply: the next iteration calls the same collectives again, and this
    kernel does not get to assume anything about whether a primitive fences
    its own scratch on exit. One barrier per chunk is what the hand-rolled
    arm already paid, and a chunk is a whole threadgroup's worth of block
    sums, so the count of them is tiny.
    """
    var tid = thread_idx.x
    var nb = Int(n_blocks)

    var carry = Int32(0)
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

        if idx < nb:
            block_sums[unsafe_offset=idx] = carry + inclusive - own
        carry += chunk_total
        barrier()
        base += block_size

    if tid == 0:
        total[unsafe_offset=0] = carry


def _scatter_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    total: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
):
    """Stage three: write each row to its compacted slot.

    A left-going row at local index `j` lands at its global left rank `p`; a
    right-going one lands after every left-going row, at `n_left + (j - p)`,
    where `j - p` is its rank among the right-going rows. Both ranks are
    monotone in `j`, so both sides keep their relative order and the
    partition is stable. `n_left` is read from the device, so no host round
    trip sits between the scan and the scatter.

    The direction comes out of the low bit of the packed offset stage one
    wrote, so this stage touches no bin and needs no routing rule: it is a
    pure permutation of the range from three streamed Int32 reads per row.

    Launched with the block size stage one used, because `block_idx.x` is
    what selects this element's block offset.
    """
    var j = block_idx.x * block_dim.x + thread_idx.x
    var n = Int(count)
    if j < n:
        var b = Int(begin)
        var row = rows[unsafe_offset = b + j][0]
        var packed = offsets[unsafe_offset=j][0]
        var p = Int(block_sums[unsafe_offset = block_idx.x][0]) + (
            Int(packed) >> 1
        )
        var dst: Int
        if (packed & Int32(1)) != 0:
            dst = p
        else:
            dst = Int(total[unsafe_offset=0][0]) + (j - p)
        scratch[unsafe_offset = b + dst] = row


def _copy_back_kernel(
    rows: MutPointer[Int32, MutAnyOrigin],
    scratch: MutPointer[Int32, MutAnyOrigin],
    begin: Int32,
    count: Int32,
):
    """Fold the compacted range back over the parent's slots. Only the
    parent's own range is touched, so every other leaf's range survives the
    split untouched; that is why this is a copy and not a buffer swap."""
    var j = global_idx.x
    if j < Int(count):
        var i = Int(begin) + j
        rows[unsafe_offset=i] = scratch[unsafe_offset=i][0]


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

    **The two row loops.** `use_quant` selects between the pre-quantized
    interleaved buffer `_quantize_grad_hess_kernel` writes and the two Float32
    planes. They are two loops rather than one loop with a branch in it so the
    default path spends no floating-point arithmetic in the row loop at all,
    and one runtime flag rather than a third comptime parameter because
    doubling forty instantiations to eighty is compile time every build on
    every backend pays. The two produce bit-identical histograms by
    construction; see `_quantize_grad_hess_kernel`.

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

    # One feature id and one column base per owned slot, read once. The loop
    # is unrolled so both arrays are indexed by a compile-time constant and
    # stay in registers rather than spilling to local memory.
    var fid = stack_allocation[GROUP, Int]()
    var col = stack_allocation[GROUP, Int]()
    comptime for k in range(GROUP):
        fid[unsafe_offset=k] = 0
        col[unsafe_offset=k] = 0
        if k < owned:
            var f = Int(feat_ids[unsafe_offset = slot0 + k][0])
            fid[unsafe_offset=k] = f
            col[unsafe_offset=k] = f * nr

    var tile_begin = block_idx.y * Int(rows_per_tile)
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    if celide:
        # Two atomics per (row, feature). The hessian is neither read nor
        # accumulated: it is the same Int32 for every row, so the count plane
        # already carries everything the hessian plane would have.
        if Int(use_quant) != 0:
            var j = tile_begin + tid
            while j < tile_end:
                var r = Int(rows[unsafe_offset = Int(begin) + j][0])
                var gqv = gq[unsafe_offset = 2 * r][0]
                comptime for k in range(GROUP):
                    if k < owned:
                        var s = k * nb + Int(
                            bins[unsafe_offset = col[unsafe_offset=k] + r]
                        )
                        _ = Atomic.fetch_add(sg.unsafe_offset(s), gqv)
                        _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))
                j += block_dim.x
        else:
            var j = tile_begin + tid
            while j < tile_end:
                var r = Int(rows[unsafe_offset = Int(begin) + j][0])
                var gqv = Int32(round(grad[unsafe_offset=r][0] * g_scale))
                comptime for k in range(GROUP):
                    if k < owned:
                        var s = k * nb + Int(
                            bins[unsafe_offset = col[unsafe_offset=k] + r]
                        )
                        _ = Atomic.fetch_add(sg.unsafe_offset(s), gqv)
                        _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))
                j += block_dim.x
    elif Int(use_quant) != 0:
        var j = tile_begin + tid
        while j < tile_end:
            var r = Int(rows[unsafe_offset = Int(begin) + j][0])
            var gqv = gq[unsafe_offset = 2 * r][0]
            var hqv = gq[unsafe_offset = 2 * r + 1][0]
            comptime for k in range(GROUP):
                if k < owned:
                    var s = k * nb + Int(
                        bins[unsafe_offset = col[unsafe_offset=k] + r]
                    )
                    _ = Atomic.fetch_add(sg.unsafe_offset(s), gqv)
                    _ = Atomic.fetch_add(sh.unsafe_offset(s), hqv)
                    _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))
            j += block_dim.x
    else:
        var j = tile_begin + tid
        while j < tile_end:
            var r = Int(rows[unsafe_offset = Int(begin) + j][0])
            var gqv = Int32(round(grad[unsafe_offset=r][0] * g_scale))
            var hqv = Int32(round(hess[unsafe_offset=r][0] * h_scale))
            comptime for k in range(GROUP):
                if k < owned:
                    var s = k * nb + Int(
                        bins[unsafe_offset = col[unsafe_offset=k] + r]
                    )
                    _ = Atomic.fetch_add(sg.unsafe_offset(s), gqv)
                    _ = Atomic.fetch_add(sh.unsafe_offset(s), hqv)
                    _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))
            j += block_dim.x
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
                    _ = Atomic.fetch_add(out_hist.unsafe_offset(base + c), vg)
                    _ = Atomic.fetch_add(
                        out_hist.unsafe_offset(hs + base + c), vh
                    )
                    _ = Atomic.fetch_add(
                        out_hist.unsafe_offset(2 * hs + base + c), vc
                    )
                    if sub:
                        var so = Int(sub_offset) + base + c
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
):
    """The tiled twin of `_range_hist_atomic_kernel`: the same threadgroup
    accumulation, written to a per-(tile, slot) partial slot instead of folded
    into the output with global atomics.

    Same two comptime parameters and the same rules for them, so everything
    the atomic kernel's docstring argues about capacity, group width,
    exactness, tail blocks, and the two row loops holds here unchanged. What
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

    var col = stack_allocation[GROUP, Int]()
    comptime for k in range(GROUP):
        col[unsafe_offset=k] = 0
        if k < owned:
            col[unsafe_offset=k] = (
                Int(feat_ids[unsafe_offset = slot0 + k][0]) * nr
            )

    var tile_begin = t * Int(rows_per_tile)
    var tile_end = tile_begin + Int(rows_per_tile)
    if tile_end > n:
        tile_end = n

    if celide:
        if Int(use_quant) != 0:
            var j = tile_begin + tid
            while j < tile_end:
                var r = Int(rows[unsafe_offset = Int(begin) + j][0])
                var gqv = gq[unsafe_offset = 2 * r][0]
                comptime for k in range(GROUP):
                    if k < owned:
                        var s = k * nb + Int(
                            bins[unsafe_offset = col[unsafe_offset=k] + r]
                        )
                        _ = Atomic.fetch_add(sg.unsafe_offset(s), gqv)
                        _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))
                j += block_dim.x
        else:
            var j = tile_begin + tid
            while j < tile_end:
                var r = Int(rows[unsafe_offset = Int(begin) + j][0])
                var gqv = Int32(round(grad[unsafe_offset=r][0] * g_scale))
                comptime for k in range(GROUP):
                    if k < owned:
                        var s = k * nb + Int(
                            bins[unsafe_offset = col[unsafe_offset=k] + r]
                        )
                        _ = Atomic.fetch_add(sg.unsafe_offset(s), gqv)
                        _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))
                j += block_dim.x
    elif Int(use_quant) != 0:
        var j = tile_begin + tid
        while j < tile_end:
            var r = Int(rows[unsafe_offset = Int(begin) + j][0])
            var gqv = gq[unsafe_offset = 2 * r][0]
            var hqv = gq[unsafe_offset = 2 * r + 1][0]
            comptime for k in range(GROUP):
                if k < owned:
                    var s = k * nb + Int(
                        bins[unsafe_offset = col[unsafe_offset=k] + r]
                    )
                    _ = Atomic.fetch_add(sg.unsafe_offset(s), gqv)
                    _ = Atomic.fetch_add(sh.unsafe_offset(s), hqv)
                    _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))
            j += block_dim.x
    else:
        var j = tile_begin + tid
        while j < tile_end:
            var r = Int(rows[unsafe_offset = Int(begin) + j][0])
            var gqv = Int32(round(grad[unsafe_offset=r][0] * g_scale))
            var hqv = Int32(round(hess[unsafe_offset=r][0] * h_scale))
            comptime for k in range(GROUP):
                if k < owned:
                    var s = k * nb + Int(
                        bins[unsafe_offset = col[unsafe_offset=k] + r]
                    )
                    _ = Atomic.fetch_add(sg.unsafe_offset(s), gqv)
                    _ = Atomic.fetch_add(sh.unsafe_offset(s), hqv)
                    _ = Atomic.fetch_add(sc.unsafe_offset(s), Int32(1))
            j += block_dim.x
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
    # Block-local exclusive left-prefix, one per element of the range.
    var offsets_dev: DeviceBuffer[DType.int32]
    # Per-scan-block left counts, turned into block offsets in place.
    var block_sums_dev: DeviceBuffer[DType.int32]
    # One Int32: the range's total left count.
    var total_dev: DeviceBuffer[DType.int32]
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
        # The constant-hessian specialization is available but not declared.
        # A builder that says nothing gets the three-plane path that shipped,
        # which is the only default a specialization keyed on the objective
        # can have: this constructor cannot see an objective.
        # `MOJOTREES_CONST_HESSIAN=0` withdraws the permission, so a later
        # `set_constant_hessian(True)` is refused rather than silently
        # honored, which is the off switch a bisection wants.
        self.const_hessian_allowed = _env_int("MOJOTREES_CONST_HESSIAN", 1) != 0
        self.constant_hessian = False

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
        synchronize; the left count lands in `total_dev`."""
        routing.check(self.n_features, self.n_bins)
        if window.begin < 0 or window.end > self.n_rows:
            raise Error("range escapes the row buffer")
        var n = window.count()
        if n <= 0:
            return

        var threads = self.block_threads
        var blocks = (n + threads - 1) // threads
        var cat = routing.cat_bitset
        var default_left = Int32(1) if routing.default_left else Int32(0)
        var is_cat = Int32(1) if routing.is_categorical else Int32(0)

        # The two scan stages, on whichever arm is selected. The primitive
        # kernels are instantiated per width, so the dispatch is an if-chain
        # over the menu rather than a runtime block size; the whole chain
        # sits inside a `comptime if has_accelerator()` because a build with
        # no accelerator target has no warp size for `block.prefix_sum` to
        # constrain against, and pruning the branch is what keeps those
        # instantiations out of a CPU-only extension build entirely.
        var scanned = False
        comptime if has_accelerator():
            if self.scan_primitives:
                scanned = True
                if threads == 128:
                    self._enqueue_scan_primitives[128](
                        bins, window, routing, n, blocks
                    )
                elif threads == 256:
                    self._enqueue_scan_primitives[256](
                        bins, window, routing, n, blocks
                    )
                elif threads == 512:
                    self._enqueue_scan_primitives[512](
                        bins, window, routing, n, blocks
                    )
                elif threads == 1024:
                    self._enqueue_scan_primitives[1024](
                        bins, window, routing, n, blocks
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
                grid_dim=blocks,
                block_dim=threads,
            )
            self.ctx.enqueue_function[_scan_block_sums_kernel](
                self.block_sums_dev.unsafe_ptr(),
                self.total_dev.unsafe_ptr(),
                Int32(blocks),
                grid_dim=1,
                block_dim=threads,
            )
        # Same block size as the flag pass: the scatter looks its block
        # offset up by `block_idx.x`.
        self.ctx.enqueue_function[_scatter_kernel](
            self.rows_dev.unsafe_ptr(),
            self.scratch_dev.unsafe_ptr(),
            self.offsets_dev.unsafe_ptr(),
            self.block_sums_dev.unsafe_ptr(),
            self.total_dev.unsafe_ptr(),
            Int32(window.begin),
            Int32(n),
            grid_dim=blocks,
            block_dim=threads,
        )
        self.ctx.enqueue_function[_copy_back_kernel](
            self.rows_dev.unsafe_ptr(),
            self.scratch_dev.unsafe_ptr(),
            Int32(window.begin),
            Int32(n),
            grid_dim=blocks,
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
    ) raises:
        """The flag pass and the block-sum scan, on the primitive arm, at one
        compile-time threadgroup width.

        Split out only so `enqueue_partition` can name the width once per
        menu entry instead of repeating sixteen kernel arguments per entry.
        `block_dim` is `width` and not `self.block_threads`: the two are equal
        wherever this is reached, and writing the compile-time one is what
        makes a mismatch between the parameter and the launch impossible to
        introduce later. `grid_dim` is the caller's block count, computed
        from `self.block_threads`, so the block count and therefore the
        `block_sums_dev` footprint are exactly what the fallback arm uses.
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
            grid_dim=blocks,
            block_dim=width,
        )
        self.ctx.enqueue_function[_scan_block_sums_prim_kernel[width]](
            self.block_sums_dev.unsafe_ptr(),
            self.total_dev.unsafe_ptr(),
            Int32(blocks),
            grid_dim=1,
            block_dim=width,
        )

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
        """
        var blocks = (n_slots + GROUP - 1) // GROUP
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
    ) raises:
        """The atomic launch at one group width, over the four bin
        capacities."""
        var blocks = (n_slots + GROUP - 1) // GROUP
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
