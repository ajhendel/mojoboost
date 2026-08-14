"""GPU histogram accumulation and row partitioning.

Same conceptual inputs and outputs as the CPU builders in `histogram.mojo`,
different algorithm for the hardware. The dataset stays device-resident for
a whole training session: the binned matrix is uploaded once at construction,
gradients and hessians once per boosting round, and a device-resident
permutation of the row indices (see gpu_active_rows.mojo) gives every live
leaf a contiguous half-open range of rows, so tree growth never copies
row-index lists across the PCIe/unified-memory boundary.

Histograms build with a 2D grid: `grid.x` is the active feature, `grid.y` a
tile of rows, so a device gets `n_active * n_tiles` threadgroups of parallel
work instead of just `n_active`. Every threadgroup accumulates a partial
histogram for its (feature, row-tile) in shared memory, reading only the
node's own compacted row range. The launch geometry (threads per group,
tiles per feature) is derived per node from device capabilities and the
node's own row count rather than fixed at compile time; see
`gpu_tiling.mojo` and `gpu_active_rows.mojo`.

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

Row bagging rides on the same permutation: `begin_tree` seeds the root
range with the bag's rows in the caller's order, and a row outside the bag
is simply not inside any range, so no kernel ever iterates it. A bagged
tree costs one n_rows Int32 staged copy; an unbagged one seeds the
identity permutation with a kernel and costs no transfer at all.

Every node's histogram is built by scanning exactly that node's rows, so a
tree costs on the order of `sum over built nodes of node_rows * n_features`
bin reads, the same asymptotic cost as the CPU builder's row-index lists,
instead of the `nodes_built * n_rows * n_features` the pre-compaction
filtering kernels paid. The partition that maintains the ranges is a
stable four-launch scan/scatter that is bit-deterministic and keeps each
node's rows in the exact order the CPU grower's row lists hold them.

Transfers stage through pinned host buffers and one-way copies rather than
`map_to_host`, which copies in both directions on every use. That also makes
the phases separately timeable, which is what `bench/bench_histogram.mojo`
reports: `stage_gradients` (Float64 to Float32 conversion), `upload_staged`
(host to device), `enqueue_leaf` (kernels), `download_raw` (device to host),
and `histogram_from_host` (fixed-point to Float64 conversion).
"""

from std.math import isfinite
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from .binning import BinnedMatrix
from .categorical import CatBitset, CategoricalSpec, cat_empty
from .gpu_active_rows import GpuActiveRows, RowRouting
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

# Row indices and leaf ids cross into the kernels as Int32.
comptime MAX_ROWS = Int(Int32.MAX)

comptime _FIXED_ONE = Float64(1 << 30)


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
    # The device-resident active-row permutation and its per-leaf ranges;
    # every histogram and every partition works through it.
    var rows: GpuActiveRows
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
        self.rows = GpuActiveRows(
            self.ctx, data.n_rows, data.n_features, data.n_bins, self.caps
        )
        self.g_scale = 1.0
        self.h_scale = 1.0
        self.has_gradients = False
        self.active = List[Int](capacity=data.n_features)
        for f in range(data.n_features):
            self.active.append(f)

        var n_cells = data.n_rows * data.n_features
        var hist_size = data.n_features * data.n_bins
        self.bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](n_cells)
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
        Float32 in pinned host memory. Host work only, no transfer.

        The device's copy is stale from here until `upload_staged`, and the
        scales below already describe the new values, so builds are refused
        in between rather than mixing one round's scale with another's
        data."""
        if len(grad) != self.n_rows or len(hess) != self.n_rows:
            raise Error("gradient/hessian length must equal n_rows")
        self.has_gradients = False

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
        """Seed the tree's active rows and make the root (node 0) own all of
        them.

        Unbagged, the root range is the identity permutation, written by a
        kernel so a tree costs no host-to-device row transfer at all. With a
        non-empty `bag`, the bag's rows are staged in the caller's order and
        the rows left out are simply not inside the root range: no sentinel
        leaf id, no per-node filtering, and no cost for a row this tree
        ignores. See gpu_active_rows.mojo.
        """
        self.rows.begin_tree(bag)

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
        """Reassign rows of `parent` to `left`/`right` by the chosen split,
        entirely on the device: the parent's contiguous row range is stably
        partitioned into the two children's. Rows in `missing_bin` follow
        `default_left` instead of the threshold; -1 (the default) means the
        feature has no missing bin and every row goes by the threshold.

        With `is_categorical`, `cat_bitset` is the node's category set and
        `threshold_bin`/`missing_bin`/`default_left` are ignored: a row goes
        left exactly when its bin is in the set, which is what
        `Tree.goes_left` does on the host.

        `expected_left` is the left row count the caller already knows
        exactly (the grower has it from the parent histogram's integer
        counts). Passing it keeps the split fully enqueued; the default -1
        downloads the device's own count, which synchronizes. With
        `MOJOBOOST_GPU_VERIFY_ROWS=1` a supplied count is checked against
        the device's anyway."""
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

    def enqueue_leaf(mut self, leaf: Int) raises:
        """Enqueue the kernels building the histogram of the rows `leaf`
        currently owns, reading only that node's compacted row range. Does
        not transfer or synchronize. A small node gets a grid sized for its
        own rows rather than for the dataset; see gpu_active_rows.mojo."""
        if not self.has_gradients:
            raise Error("call upload_gradients before build_leaf")
        if leaf < 0 or leaf > MAX_ROWS:
            raise Error("leaf id must be nonnegative and fit in Int32")

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
