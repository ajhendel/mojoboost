"""GPU histogram accumulation and row partitioning.

Same conceptual inputs and outputs as the CPU builders in `histogram.mojo`,
different algorithm for the hardware. The dataset stays device-resident for
a whole training session: the binned matrix is uploaded once at construction,
gradients and hessians once per boosting round, and a per-row leaf-assignment
array (row -> current leaf node id) lives on the device so tree growth never
copies row-index lists across the PCIe/unified-memory boundary.

Histograms build with a 2D grid: `grid.x` is the active feature, `grid.y` a
tile of rows, so a device gets `n_active * n_tiles` threadgroups of parallel
work instead of just `n_active`. Every threadgroup accumulates a partial
histogram for its (feature, row-tile) in shared memory, filtering rows by the
target leaf id. The launch geometry (threads per group, tiles per feature) is
derived from device capabilities at runtime rather than fixed at compile
time; see `gpu_tiling.mojo`.

Two strategies combine those partials, both selectable and both tested:

- `STRATEGY_TILED` writes each partial to its own slot of a global buffer
  and sums the slots in a second kernel, in ascending tile order. No global
  atomics and no contention on hot bins, at the cost of one extra kernel
  launch and a partial buffer bounded by a memory budget.
- `STRATEGY_ATOMIC` folds each partial into the output with global integer
  atomics. This is the original implementation, preserved as the fallback
  for hardware where the tiled path is not yet validated, and chosen
  automatically when the partial buffer would not fit the budget. The
  kernel is unchanged; only its launch geometry is now device-derived
  instead of a fixed constant, which cannot affect an atomic accumulation.

The two return bit-identical histograms, which the tests assert directly:
accumulation is exact fixed-point Int32 throughout and integer addition is
associative, so how the partials combine cannot change the result.

Gradient, hessian, and count planes share one Int32 device buffer laid out as
`[grad | hess | count]`, so a whole node's histogram costs one kernel launch
and one device-to-host copy instead of three of each. Per-node launch and
synchronization overhead dominates on small nodes, so this is the difference
between three host synchronizations per tree node and one. The partial buffer
repeats that layout once per row tile.

Within and across blocks, gradients and hessians accumulate as fixed-point
Int32 via integer atomics rather than float atomics. Metal has no float
atomic add, and integer accumulation is portable (CUDA/ROCm/Metal) and
order-independent, so GPU histograms are bit-deterministic run to run. The
fixed-point scale is chosen on the host from the global gradient/hessian
magnitude sums, which bound every partial sum (any leaf's rows are a subset
of all rows), so scaled accumulation cannot overflow.

The kernels use only primitives all three backends provide: shared memory,
`barrier()`, integer atomics on shared memory, and plain global loads and
stores. No warp shuffles, no float atomics, no vendor intrinsics, and no
per-architecture code paths. That portable baseline is the same source on
Metal, CUDA, and HIP; only the tiling numbers differ per device.

Gradients are carried as Float32 on the device: Apple GPUs have no Float64.
Results convert back to the Float64 `Histogram` on download; agreement with
the CPU builder is to Float32 precision, not bit-exact. Counts are exact.

Row bagging rides on the same leaf-assignment array: `begin_tree` parks
excluded rows at the OUT_OF_BAG leaf id instead of the root, and since node
ids are nonnegative, those rows match no histogram filter and no partition
for the rest of the tree. Nothing is compacted or reuploaded, so a bagged
tree costs the same device memory as an unbagged one.

Every node's histogram is built by scanning all n_rows and filtering on the
leaf id, so a tree costs `num_leaves * n_rows * n_features` bin reads where
the CPU builder's row-index lists cost `n_rows * n_features` per level. That
is the deliberate portable baseline: order-preserving row compaction is the
next step, and it changes the kernel, not this module's contract.

Transfers stage through pinned host buffers and one-way copies rather than
`map_to_host`, which copies in both directions on every use. That also makes
the phases separately timeable, which is what `bench/bench_histogram.mojo`
reports: `stage_gradients` (Float64 to Float32 conversion), `upload_staged`
(host to device), `enqueue_leaf` (kernels), `download_raw` (device to host),
and `histogram_from_host` (fixed-point to Float64 conversion).
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import isfinite, round
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .binning import BinnedMatrix
from .categorical import CatBitset, CategoricalSpec, cat_empty
from .gpu_tiling import (
    STRATEGY_ATOMIC,
    STRATEGY_AUTO,
    STRATEGY_TILED,
    DeviceCaps,
    HistogramTiling,
    derive_tiling,
    query_device_caps,
)
from .histogram import Histogram, _zeroed_f64, _zeroed_int

comptime MAX_BINS = 256

# Leaf id of a row excluded by row bagging. Node ids are nonnegative, so a
# row parked here is matched by no histogram build and no split.
comptime OUT_OF_BAG = Int32(-1)

# Row indices and leaf ids cross into the kernels as Int32.
comptime MAX_ROWS = Int(Int32.MAX)

comptime _FIXED_ONE = Float64(1 << 30)


def _hist_leaf_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    leaf_ids: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_bins: Int32,
    hist_size: Int32,
    rows_per_chunk: Int32,
    target_leaf: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """Atomic strategy: one shared-memory partial per (feature, row-tile),
    folded into the output with global integer atomics."""
    # grid.x indexes the active feature set, not the dataset's features, so
    # under feature subsampling excluded features get no threadgroups at all
    # and their output slices keep the zeros the memset left. Writes still go
    # to the global feature's slice, so the histogram layout never changes.
    var f = Int(feat_ids[unsafe_offset = Int(block_idx.x)][0])
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var hs = Int(hist_size)

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

    var row_begin = block_idx.y * Int(rows_per_chunk)
    var row_end = row_begin + Int(rows_per_chunk)
    if row_end > nr:
        row_end = nr

    var col = f * nr
    var r = row_begin + tid
    while r < row_end:
        if leaf_ids[unsafe_offset=r][0] == target_leaf:
            var bin = Int(bins[unsafe_offset=col + r])
            var gq = Int32(round(grad[unsafe_offset=r][0] * g_scale))
            var hq = Int32(round(hess[unsafe_offset=r][0] * h_scale))
            _ = Atomic.fetch_add(sg.unsafe_offset(bin), gq)
            _ = Atomic.fetch_add(sh.unsafe_offset(bin), hq)
            _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        r += block_dim.x
    barrier()

    # One flush into the shared [grad | hess | count] output buffer.
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


def _hist_partial_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    leaf_ids: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    partials: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_slots: Int32,
    n_bins: Int32,
    rows_per_tile: Int32,
    target_leaf: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    """Tiled strategy, stage one: one shared-memory partial per (feature,
    row-tile), written to that tile's own slice of the partial buffer.

    The partial buffer is indexed by active-feature slot rather than by
    global feature id, so under feature subsampling it stays proportional to
    the work actually launched. Tile `t` owns
    `partials[t * 3 * n_slots * n_bins ...]`, laid out `[grad | hess |
    count]` like the output. Every element is written exactly once by
    exactly one threadgroup, so the buffer needs no zeroing and the writes
    need no atomics.
    """
    var slot = Int(block_idx.x)
    var f = Int(feat_ids[unsafe_offset=slot][0])
    var t = block_idx.y
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)
    var plane = Int(n_slots) * nb

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

    var row_begin = t * Int(rows_per_tile)
    var row_end = row_begin + Int(rows_per_tile)
    if row_end > nr:
        row_end = nr

    var col = f * nr
    var r = row_begin + tid
    while r < row_end:
        if leaf_ids[unsafe_offset=r][0] == target_leaf:
            var bin = Int(bins[unsafe_offset=col + r])
            var gq = Int32(round(grad[unsafe_offset=r][0] * g_scale))
            var hq = Int32(round(hess[unsafe_offset=r][0] * h_scale))
            _ = Atomic.fetch_add(sg.unsafe_offset(bin), gq)
            _ = Atomic.fetch_add(sh.unsafe_offset(bin), hq)
            _ = Atomic.fetch_add(sc.unsafe_offset(bin), Int32(1))
        r += block_dim.x
    barrier()

    var base = t * 3 * plane + slot * nb
    b = tid
    while b < nb:
        partials[unsafe_offset = base + b] = sg[unsafe_offset=b][0]
        partials[unsafe_offset = base + plane + b] = sh[unsafe_offset=b][0]
        partials[unsafe_offset = base + 2 * plane + b] = sc[
            unsafe_offset=b
        ][0]
        b += block_dim.x


def _hist_reduce_kernel(
    partials: MutPointer[Int32, MutAnyOrigin],
    feat_ids: MutPointer[Int32, MutAnyOrigin],
    out_hist: MutPointer[Int32, MutAnyOrigin],
    n_slots: Int32,
    n_bins: Int32,
    hist_size: Int32,
    n_tiles: Int32,
):
    """Tiled strategy, stage two: one thread per partial element sums that
    element across tiles, in ascending tile order, and scatters the total to
    the global feature's slice of the output.

    All three planes reduce in this one loop. Sequential summation in a
    fixed order over exact integers, so the result cannot depend on
    scheduling, and every active output element is written by exactly one
    thread.
    """
    var i = global_idx.x
    var nb = Int(n_bins)
    var plane = Int(n_slots) * nb
    var n = 3 * plane
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
        out_hist[unsafe_offset = p * Int(hist_size) + f * nb + b] = acc


def _partition_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    leaf_ids: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    feature: Int32,
    threshold_bin: Int32,
    parent: Int32,
    left: Int32,
    right: Int32,
    missing_bin: Int32,
    default_left: Int32,
    is_categorical: Int32,
    cat0: UInt64,
    cat1: UInt64,
    cat2: UInt64,
    cat3: UInt64,
):
    var r = global_idx.x
    var nr = Int(n_rows)
    if r < nr:
        if leaf_ids[unsafe_offset=r][0] == parent:
            var bin = Int32(bins[unsafe_offset = Int(feature) * nr + r])
            var go_left: Bool
            if is_categorical != 0:
                # Categorical split: the node's 256-bit category set arrives
                # as four scalar words, so the test is the same membership
                # `Tree.goes_left` runs on the host. Bin 0 (missing, unseen,
                # dropped) is never a member, so those rows go right.
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
                go_left = ((word >> UInt64(Int(bin) & 63)) & 1) != 0
            # Rows in the missing bin follow the split's default direction,
            # the same rule `Tree.goes_left` applies on the host. A feature
            # with no missing bin passes -1, which no bin id can equal.
            elif bin == missing_bin:
                go_left = default_left != 0
            else:
                go_left = bin <= threshold_bin
            if go_left:
                leaf_ids[unsafe_offset=r] = left
            else:
                leaf_ids[unsafe_offset=r] = right


def _fixed_scale(values: List[Float64]) raises -> Float32:
    """Fixed-point scale from the magnitude sum: every partial sum of scaled
    values stays within +/- 2^30, half the Int32 range.

    Returned as Float32 because that is the precision the kernel multiplies
    by, so the host-side inverse matches the device quantization exactly.
    Magnitude sums below the floor are numerically zero against any
    regularization; the floor keeps the scale finite instead of dividing by
    (near) zero."""
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    if not isfinite(total):
        raise Error("gradients and hessians must be finite")
    if total < 1e-12:
        total = 1e-12
    var scale = Float32(_FIXED_ONE / total)
    if not isfinite(scale) or scale <= 0.0:
        raise Error(
            "gradient/hessian magnitudes are out of range for the GPU"
            " fixed-point histogram"
        )
    return scale


struct GpuHistogramBuilder(Movable):
    """Device-resident histogram builder and row partitioner for one binned
    dataset. Construct once per training session, `upload_gradients` once per
    boosting round, `begin_tree` + `build_leaf`/`apply_split` per tree."""

    var ctx: DeviceContext
    var bins_dev: DeviceBuffer[DType.uint8]
    var leaf_dev: DeviceBuffer[DType.int32]
    var grad_dev: DeviceBuffer[DType.float32]
    var hess_dev: DeviceBuffer[DType.float32]
    # The feature ids builds accumulate, device side; the first `n_active`
    # entries are live (see `set_features`).
    var feat_dev: DeviceBuffer[DType.int32]
    # [grad | hess | count], each n_features * n_bins entries.
    var out_dev: DeviceBuffer[DType.int32]
    # The same three planes once per row tile, indexed by active-feature
    # slot. One element when the resolved strategy needs no partial buffer.
    var part_dev: DeviceBuffer[DType.int32]
    var part_capacity: Int
    var stage_g: HostBuffer[DType.float32]
    var stage_h: HostBuffer[DType.float32]
    var host_out: HostBuffer[DType.int32]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    # The dataset's per-feature missing bins, so the grower can route missing
    # rows without holding on to the BinnedMatrix.
    var missing_bin: List[Int]
    var cats: CategoricalSpec
    var active: List[Int]
    var caps: DeviceCaps
    var tiling: HistogramTiling
    var g_scale: Float64
    var h_scale: Float64
    var has_gradients: Bool

    def __init__(
        out self, data: BinnedMatrix, strategy: Int = STRATEGY_AUTO
    ) raises:
        """Upload `data` and resolve the launch geometry for its shape.

        `strategy` forces `STRATEGY_ATOMIC` or `STRATEGY_TILED`; the default
        `STRATEGY_AUTO` lets `MOJOBOOST_GPU_HIST_STRATEGY` and then the
        device-capability policy in `gpu_tiling.mojo` decide.
        """
        if data.n_rows < 1:
            raise Error("GPU backend requires at least one row")
        if data.n_features < 1:
            raise Error("GPU backend requires at least one feature")
        if data.n_bins < 1:
            raise Error("GPU backend requires at least one bin")
        if data.n_bins > MAX_BINS:
            raise Error("GPU backend supports at most 256 bins")
        if data.n_rows > MAX_ROWS:
            raise Error("GPU backend supports at most 2^31 - 1 rows")
        if len(data.bins) != data.n_rows * data.n_features:
            raise Error("binned matrix size must equal n_rows * n_features")

        self.ctx = DeviceContext()
        self.n_rows = data.n_rows
        self.n_features = data.n_features
        self.n_bins = data.n_bins
        self.missing_bin = data.missing_bin.copy()
        self.cats = data.cats.copy()
        self.caps = query_device_caps(self.ctx)
        self.tiling = derive_tiling(
            self.caps, data.n_rows, data.n_features, data.n_bins, strategy
        )
        self.part_capacity = self.tiling.partial_cells
        self.g_scale = 1.0
        self.h_scale = 1.0
        self.has_gradients = False
        self.active = List[Int](capacity=data.n_features)
        for f in range(data.n_features):
            self.active.append(f)

        var n_cells = data.n_rows * data.n_features
        var hist_size = data.n_features * data.n_bins
        self.bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](n_cells)
        self.leaf_dev = self.ctx.enqueue_create_buffer[DType.int32](
            data.n_rows
        )
        self.grad_dev = self.ctx.enqueue_create_buffer[DType.float32](
            data.n_rows
        )
        self.hess_dev = self.ctx.enqueue_create_buffer[DType.float32](
            data.n_rows
        )
        self.out_dev = self.ctx.enqueue_create_buffer[DType.int32](
            3 * hist_size
        )
        self.feat_dev = self.ctx.enqueue_create_buffer[DType.int32](
            data.n_features
        )

        # Zero-length device buffers are not portable, so the atomic
        # strategy still allocates a one-element placeholder.
        var part_size = 3 * self.part_capacity
        if part_size < 1:
            part_size = 1
        self.part_dev = self.ctx.enqueue_create_buffer[DType.int32](part_size)

        self.stage_g = self.ctx.enqueue_create_host_buffer[DType.float32](
            data.n_rows
        )
        self.stage_h = self.ctx.enqueue_create_host_buffer[DType.float32](
            data.n_rows
        )
        self.host_out = self.ctx.enqueue_create_host_buffer[DType.int32](
            3 * hist_size
        )

        self.ctx.enqueue_memset(self.leaf_dev, 0)

        # Upload the binned matrix once; it is reused every call. The copy
        # reads host memory owned by the caller's `data`, so it has to
        # complete before the constructor returns.
        self.ctx.enqueue_copy(
            dst_buf=self.bins_dev, src_ptr=data.bins.unsafe_ptr()
        )
        self.ctx.synchronize()

        # Every feature is active until `set_features` narrows it.
        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for f in range(data.n_features):
                dst.unsafe_store(f, Int32(f))

    def strategy(self) -> Int:
        """The accumulation strategy this builder resolved to."""
        return self.tiling.strategy

    def synchronize(self) raises:
        """Block until every enqueued device operation has completed."""
        self.ctx.synchronize()

    def set_features(mut self, features: List[Int]) raises:
        """Restrict later `build_leaf` calls to `features` (global feature
        ids, one entry each). This is how the GPU grower consumes the same
        subsampled feature set as the CPU grower: the dataset stays whole and
        device-resident, only the launch grid narrows. Slices of features not
        listed here stay zero in every histogram built afterwards, which is
        what keeps sibling subtraction exact as long as one tree keeps one
        feature set."""
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
        # Fewer features in grid.x means fewer threadgroups, so the row
        # tiling is re-derived for the narrowed grid. The partial buffer is
        # never reallocated: its construction-time capacity is the cap.
        self.tiling = derive_tiling(
            self.caps,
            self.n_rows,
            len(self.active),
            self.n_bins,
            self.tiling.strategy,
            self.part_capacity,
        )
        with self.feat_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for i in range(len(features)):
                dst.unsafe_store(i, Int32(features[i]))

    def stage_gradients(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises:
        """Convert this round's gradients and hessians into the device's
        Float32 in pinned host memory. Host work only, no transfer."""
        if len(grad) != self.n_rows or len(hess) != self.n_rows:
            raise Error("gradient/hessian length must equal n_rows")

        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)
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
        """Copy the staged gradients and hessians to the device."""
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

    def begin_tree(mut self, bag: List[Int] = []) raises:
        """Reset every row's leaf assignment to the root node (id 0).

        A non-empty `bag` of row indices puts only those rows at the root
        and marks the rest OUT_OF_BAG. No node id is ever negative, so an
        out-of-bag row matches no `build_leaf` target and no `apply_split`
        parent: it is invisible to every histogram and every partition for
        the rest of this tree, without the gradients or the binned matrix
        being touched. The bag costs one n_rows Int32 write, which is what
        the unbagged reset costs as a memset.
        """
        if len(bag) == 0:
            self.ctx.enqueue_memset(self.leaf_dev, 0)
            return
        for i in range(len(bag)):
            if bag[i] < 0 or bag[i] >= self.n_rows:
                raise Error("bag row index out of range")
        with self.leaf_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            for r in range(self.n_rows):
                dst.unsafe_store(r, OUT_OF_BAG)
            for i in range(len(bag)):
                dst.unsafe_store(bag[i], Int32(0))

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
    ) raises:
        """Reassign rows of `parent` to `left`/`right` by the chosen split,
        entirely on the device. Rows in `missing_bin` follow `default_left`
        instead of the threshold; -1 (the default) means the feature has no
        missing bin and every row goes by the threshold.

        With `is_categorical`, `cat_bitset` is the node's category set and
        `threshold_bin`/`missing_bin`/`default_left` are ignored: a row goes
        left exactly when its bin is in the set, which is what
        `Tree.goes_left` does on the host."""
        if feature < 0 or feature >= self.n_features:
            raise Error("split feature out of range")
        if not is_categorical and (
            threshold_bin < 0 or threshold_bin >= self.n_bins
        ):
            raise Error("split threshold bin out of range")
        if missing_bin >= self.n_bins:
            raise Error("split missing bin out of range")
        if parent < 0 or left < 0 or right < 0:
            raise Error("leaf ids must be nonnegative")
        if left > MAX_ROWS or right > MAX_ROWS or parent > MAX_ROWS:
            raise Error("leaf ids must fit in Int32")
        if left == parent or right == parent or left == right:
            raise Error(
                "child leaf ids must differ from the parent and each other"
            )
        var threads = self.tiling.block_threads
        var blocks = (self.n_rows + threads - 1) // threads
        self.ctx.enqueue_function[_partition_kernel](
            self.bins_dev.unsafe_ptr(),
            self.leaf_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(feature),
            Int32(threshold_bin),
            Int32(parent),
            Int32(left),
            Int32(right),
            Int32(missing_bin),
            Int32(1) if default_left else Int32(0),
            Int32(1) if is_categorical else Int32(0),
            cat_bitset[0],
            cat_bitset[1],
            cat_bitset[2],
            cat_bitset[3],
            grid_dim=blocks,
            block_dim=threads,
        )

    def enqueue_leaf(mut self, leaf: Int) raises:
        """Enqueue the kernels building the histogram of the rows currently
        assigned to `leaf`. Does not transfer or synchronize."""
        if not self.has_gradients:
            raise Error("call upload_gradients before build_leaf")
        if leaf < 0 or leaf > MAX_ROWS:
            raise Error("leaf id must be nonnegative and fit in Int32")

        var hist_size = self.n_features * self.n_bins
        var n_slots = len(self.active)
        var threads = self.tiling.block_threads
        if self.tiling.strategy == STRATEGY_TILED:
            # The reduction writes every active feature's slice, so the
            # output only needs zeroing when some feature is inactive.
            if n_slots < self.n_features:
                self.ctx.enqueue_memset(self.out_dev, 0)
            self.ctx.enqueue_function[_hist_partial_kernel](
                self.bins_dev.unsafe_ptr(),
                self.leaf_dev.unsafe_ptr(),
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                self.part_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(self.tiling.rows_per_tile),
                Int32(leaf),
                Float32(self.g_scale),
                Float32(self.h_scale),
                grid_dim=(n_slots, self.tiling.n_tiles),
                block_dim=threads,
            )
            var n_cells = 3 * n_slots * self.n_bins
            var blocks = (n_cells + threads - 1) // threads
            self.ctx.enqueue_function[_hist_reduce_kernel](
                self.part_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                self.out_dev.unsafe_ptr(),
                Int32(n_slots),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(self.tiling.n_tiles),
                grid_dim=blocks,
                block_dim=threads,
            )
        else:
            self.ctx.enqueue_memset(self.out_dev, 0)
            self.ctx.enqueue_function[_hist_leaf_kernel](
                self.bins_dev.unsafe_ptr(),
                self.leaf_dev.unsafe_ptr(),
                self.grad_dev.unsafe_ptr(),
                self.hess_dev.unsafe_ptr(),
                self.feat_dev.unsafe_ptr(),
                self.out_dev.unsafe_ptr(),
                Int32(self.n_rows),
                Int32(self.n_bins),
                Int32(hist_size),
                Int32(self.tiling.rows_per_tile),
                Int32(leaf),
                Float32(self.g_scale),
                Float32(self.h_scale),
                grid_dim=(n_slots, self.tiling.n_tiles),
                block_dim=threads,
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
        the currently active feature set (every feature unless `set_features`
        narrowed it). The returned histogram always has the dataset's full
        `n_features * n_bins` shape; inactive features' slices are zero."""
        self.enqueue_leaf(leaf)
        self.download_raw()
        return self.histogram_from_host()

    def build(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises -> Histogram:
        """Build a full-dataset histogram on the GPU (uploads gradients and
        resets leaf assignments; use the finer-grained methods when training
        whole trees)."""
        self.upload_gradients(grad, hess)
        self.begin_tree()
        return self.build_leaf(0)


def build_histogram_gpu(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    strategy: Int = STRATEGY_AUTO,
) raises -> Histogram:
    """One-shot GPU histogram build (uploads the binned matrix every call;
    use `GpuHistogramBuilder` for repeated builds on one dataset)."""
    var builder = GpuHistogramBuilder(data, strategy)
    return builder.build(grad, hess)
