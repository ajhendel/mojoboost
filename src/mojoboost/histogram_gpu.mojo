"""GPU histogram accumulation and row partitioning.

Same conceptual inputs and outputs as the CPU builders in `histogram.mojo`,
different algorithm for the hardware. The dataset stays device-resident for
a whole training session: the binned matrix is uploaded once at construction,
gradients and hessians once per boosting round, and a per-row leaf-assignment
array (row -> current leaf node id) lives on the device so tree growth never
copies row-index lists across the PCIe/unified-memory boundary.

Histograms build with a 2D grid: `grid.x` is the feature, `grid.y` a chunk of
rows, so large GPUs get `n_features * n_chunks` threadgroups of parallel work
instead of just `n_features`. Each block accumulates a partial histogram for
its (feature, row-chunk) in shared memory, filtering rows by the target leaf
id, then flushes non-empty bins into the global histogram with integer
atomics.

Within and across blocks, gradients and hessians accumulate as fixed-point
Int32 via integer atomics rather than float atomics. Metal has no float
atomic add, and integer accumulation is portable (CUDA/ROCm/Metal) and
order-independent, so GPU histograms are bit-deterministic run to run. The
fixed-point scale is chosen on the host from the global gradient/hessian
magnitude sums, which bound every partial sum (any leaf's rows are a subset
of all rows), so scaled accumulation cannot overflow.

Gradients are carried as Float32 on the device: Apple GPUs have no Float64.
Results convert back to the Float64 `Histogram` on download; agreement with
the CPU builder is to Float32 precision, not bit-exact. Counts are exact.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import round
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .binning import BinnedMatrix
from .histogram import Histogram, _zeroed_f64, _zeroed_int

comptime MAX_BINS = 256
comptime BLOCK_THREADS = 256

# Rows per grid.y chunk. Small enough that big datasets expose thousands of
# threadgroups to large discrete GPUs, large enough that the per-block
# shared->global flush (up to 3 * n_bins atomics) stays negligible.
comptime ROWS_PER_CHUNK = 32768

comptime _FIXED_ONE = Float64(1 << 30)


def _hist_leaf_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    leaf_ids: MutPointer[Int32, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    out_g: MutPointer[Int32, MutAnyOrigin],
    out_h: MutPointer[Int32, MutAnyOrigin],
    out_c: MutPointer[UInt32, MutAnyOrigin],
    n_rows: Int32,
    n_bins: Int32,
    rows_per_chunk: Int32,
    target_leaf: Int32,
    g_scale: Float32,
    h_scale: Float32,
):
    var f = block_idx.x
    var tid = thread_idx.x
    var nb = Int(n_bins)
    var nr = Int(n_rows)

    var sg = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sh = stack_allocation[
        MAX_BINS, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var sc = stack_allocation[
        MAX_BINS, Scalar[DType.uint32], address_space = AddressSpace.SHARED
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
            _ = Atomic.fetch_add(sc.unsafe_offset(bin), UInt32(1))
        r += block_dim.x
    barrier()

    var base = f * nb
    b = tid
    while b < nb:
        if sc[unsafe_offset=b][0] != 0:
            _ = Atomic.fetch_add(
                out_g.unsafe_offset(base + b), sg[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_h.unsafe_offset(base + b), sh[unsafe_offset=b][0]
            )
            _ = Atomic.fetch_add(
                out_c.unsafe_offset(base + b), sc[unsafe_offset=b][0]
            )
        b += block_dim.x


def _partition_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    leaf_ids: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    feature: Int32,
    threshold_bin: Int32,
    parent: Int32,
    left: Int32,
    right: Int32,
):
    var r = global_idx.x
    var nr = Int(n_rows)
    if r < nr:
        if leaf_ids[unsafe_offset=r][0] == parent:
            var bin = Int32(bins[unsafe_offset = Int(feature) * nr + r])
            if bin <= threshold_bin:
                leaf_ids[unsafe_offset=r] = left
            else:
                leaf_ids[unsafe_offset=r] = right


def _fixed_scale(values: List[Float64]) -> Float64:
    """Fixed-point scale from the magnitude sum: every partial sum of scaled
    values stays within +/- 2^30, half the Int32 range."""
    var total = 0.0
    for i in range(len(values)):
        total += abs(values[i])
    if total < 1e-12:
        total = 1e-12
    return _FIXED_ONE / total


struct GpuHistogramBuilder(Movable):
    """Device-resident histogram builder and row partitioner for one binned
    dataset. Construct once per training session, `upload_gradients` once per
    boosting round, `begin_tree` + `build_leaf`/`apply_split` per tree."""

    var ctx: DeviceContext
    var bins_dev: DeviceBuffer[DType.uint8]
    var leaf_dev: DeviceBuffer[DType.int32]
    var grad_dev: DeviceBuffer[DType.float32]
    var hess_dev: DeviceBuffer[DType.float32]
    var out_g_dev: DeviceBuffer[DType.int32]
    var out_h_dev: DeviceBuffer[DType.int32]
    var out_c_dev: DeviceBuffer[DType.uint32]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var n_chunks: Int
    var g_scale: Float64
    var h_scale: Float64
    var has_gradients: Bool

    def __init__(out self, data: BinnedMatrix) raises:
        if data.n_bins > MAX_BINS:
            raise Error("GPU backend supports at most 256 bins")
        self.ctx = DeviceContext()
        self.n_rows = data.n_rows
        self.n_features = data.n_features
        self.n_bins = data.n_bins
        self.n_chunks = (data.n_rows + ROWS_PER_CHUNK - 1) // ROWS_PER_CHUNK
        if self.n_chunks < 1:
            self.n_chunks = 1
        self.g_scale = 1.0
        self.h_scale = 1.0
        self.has_gradients = False

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
        self.out_g_dev = self.ctx.enqueue_create_buffer[DType.int32](
            hist_size
        )
        self.out_h_dev = self.ctx.enqueue_create_buffer[DType.int32](
            hist_size
        )
        self.out_c_dev = self.ctx.enqueue_create_buffer[DType.uint32](
            hist_size
        )
        self.ctx.enqueue_memset(self.leaf_dev, 0)

        # Upload the binned matrix once; it is reused every call.
        with self.bins_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            var src = data.bins.unsafe_ptr()
            for i in range(n_cells):
                dst.unsafe_store(i, src.unsafe_load(i))

    def upload_gradients(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises:
        """Upload this round's per-row gradients and hessians (once per
        boosting round, not per node)."""
        if len(grad) != self.n_rows or len(hess) != self.n_rows:
            raise Error("gradient/hessian length must equal n_rows")

        # Store the Float32-rounded scales the kernel will actually use, so
        # the host-side inverse matches the device quantization exactly.
        self.g_scale = Float64(Float32(_fixed_scale(grad)))
        self.h_scale = Float64(Float32(_fixed_scale(hess)))
        self.has_gradients = True

        with self.grad_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            var src = grad.unsafe_ptr()
            for r in range(self.n_rows):
                dst.unsafe_store(r, Float32(src.unsafe_load(r)))
        with self.hess_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            var src = hess.unsafe_ptr()
            for r in range(self.n_rows):
                dst.unsafe_store(r, Float32(src.unsafe_load(r)))

    def begin_tree(mut self) raises:
        """Reset every row's leaf assignment to the root node (id 0)."""
        self.ctx.enqueue_memset(self.leaf_dev, 0)

    def apply_split(
        mut self,
        feature: Int,
        threshold_bin: Int,
        parent: Int,
        left: Int,
        right: Int,
    ) raises:
        """Reassign rows of `parent` to `left`/`right` by the chosen split,
        entirely on the device."""
        var blocks = (self.n_rows + BLOCK_THREADS - 1) // BLOCK_THREADS
        self.ctx.enqueue_function[_partition_kernel](
            self.bins_dev.unsafe_ptr(),
            self.leaf_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(feature),
            Int32(threshold_bin),
            Int32(parent),
            Int32(left),
            Int32(right),
            grid_dim=blocks,
            block_dim=BLOCK_THREADS,
        )

    def build_leaf(mut self, leaf: Int) raises -> Histogram:
        """Build the histogram of the rows currently assigned to `leaf`."""
        if not self.has_gradients:
            raise Error("call upload_gradients before build_leaf")
        self.ctx.enqueue_memset(self.out_g_dev, 0)
        self.ctx.enqueue_memset(self.out_h_dev, 0)
        self.ctx.enqueue_memset(self.out_c_dev, 0)

        self.ctx.enqueue_function[_hist_leaf_kernel](
            self.bins_dev.unsafe_ptr(),
            self.leaf_dev.unsafe_ptr(),
            self.grad_dev.unsafe_ptr(),
            self.hess_dev.unsafe_ptr(),
            self.out_g_dev.unsafe_ptr(),
            self.out_h_dev.unsafe_ptr(),
            self.out_c_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_bins),
            Int32(ROWS_PER_CHUNK),
            Int32(leaf),
            Float32(self.g_scale),
            Float32(self.h_scale),
            grid_dim=(self.n_features, self.n_chunks),
            block_dim=BLOCK_THREADS,
        )

        var hist_size = self.n_features * self.n_bins
        var g = _zeroed_f64(hist_size)
        var h = _zeroed_f64(hist_size)
        var c = _zeroed_int(hist_size)
        var g_inv = 1.0 / self.g_scale
        var h_inv = 1.0 / self.h_scale
        with self.out_g_dev.map_to_host() as host:
            var src = host.unsafe_ptr()
            var dst = g.unsafe_ptr()
            for i in range(hist_size):
                dst.unsafe_store(i, Float64(src.unsafe_load(i)) * g_inv)
        with self.out_h_dev.map_to_host() as host:
            var src = host.unsafe_ptr()
            var dst = h.unsafe_ptr()
            for i in range(hist_size):
                dst.unsafe_store(i, Float64(src.unsafe_load(i)) * h_inv)
        with self.out_c_dev.map_to_host() as host:
            var src = host.unsafe_ptr()
            var dst = c.unsafe_ptr()
            for i in range(hist_size):
                dst.unsafe_store(i, Int(src.unsafe_load(i)))
        return Histogram(g^, h^, c^, self.n_features, self.n_bins)

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
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64]
) raises -> Histogram:
    """One-shot GPU histogram build (uploads the binned matrix every call;
    use `GpuHistogramBuilder` for repeated builds on one dataset)."""
    var builder = GpuHistogramBuilder(data)
    return builder.build(grad, hess)
