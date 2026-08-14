"""GPU histogram accumulation.

Same conceptual inputs and outputs as the CPU builders in `histogram.mojo`,
different algorithm for the hardware: one threadgroup per feature accumulates
that feature's histogram in shared memory, then writes it out with plain
stores. There are no cross-block collisions, so no global atomics.

Within a block, gradients and hessians accumulate as fixed-point Int32 via
integer atomics rather than float atomics. Metal has no float atomic add, and
integer accumulation is portable (CUDA/ROCm/Metal) and order-independent, so
GPU histograms are bit-deterministic run to run. The fixed-point scale is
chosen on the host from the global gradient/hessian magnitude sums, which
bound every partial sum, so scaled accumulation cannot overflow.

Gradients are carried as Float32 on the device: Apple GPUs have no Float64.
Results are converted back to the Float64 `Histogram` on download; agreement
with the CPU builder is to Float32 precision, not bit-exact.

`GpuHistogramBuilder` keeps the binned matrix and scratch buffers
device-resident across calls, so per-tree rebuilds only upload the gradient
and hessian vectors.

This first portable kernel deliberately uses one threadgroup per feature. It
is a correctness baseline, not the final throughput design: large discrete
GPUs will need multiple threadgroups per feature followed by a partial-
histogram reduction to expose enough parallel work.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.math import round
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .binning import BinnedMatrix
from .histogram import Histogram, _zeroed_f64, _zeroed_int

comptime MAX_BINS = 256
comptime BLOCK_THREADS = 256

comptime _FIXED_ONE = Float64(1 << 30)


def _hist_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    grad: MutPointer[Float32, MutAnyOrigin],
    hess: MutPointer[Float32, MutAnyOrigin],
    out_g: MutPointer[Float32, MutAnyOrigin],
    out_h: MutPointer[Float32, MutAnyOrigin],
    out_c: MutPointer[UInt32, MutAnyOrigin],
    n_rows: Int32,
    n_bins: Int32,
    g_scale: Float32,
    h_scale: Float32,
    g_inv_scale: Float32,
    h_inv_scale: Float32,
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

    var col = f * nr
    var r = tid
    while r < nr:
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
        out_g[unsafe_offset=base + b] = (
            Float32(sg[unsafe_offset=b][0]) * g_inv_scale
        )
        out_h[unsafe_offset=base + b] = (
            Float32(sh[unsafe_offset=b][0]) * h_inv_scale
        )
        out_c[unsafe_offset=base + b] = sc[unsafe_offset=b][0]
        b += block_dim.x


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
    """Device-resident histogram builder for one binned dataset."""

    var ctx: DeviceContext
    var bins_dev: DeviceBuffer[DType.uint8]
    var grad_dev: DeviceBuffer[DType.float32]
    var hess_dev: DeviceBuffer[DType.float32]
    var out_g_dev: DeviceBuffer[DType.float32]
    var out_h_dev: DeviceBuffer[DType.float32]
    var out_c_dev: DeviceBuffer[DType.uint32]
    var n_rows: Int
    var n_features: Int
    var n_bins: Int

    def __init__(out self, data: BinnedMatrix) raises:
        if data.n_bins > MAX_BINS:
            raise Error("GPU backend supports at most 256 bins")
        self.ctx = DeviceContext()
        self.n_rows = data.n_rows
        self.n_features = data.n_features
        self.n_bins = data.n_bins

        var n_cells = data.n_rows * data.n_features
        var hist_size = data.n_features * data.n_bins
        self.bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](n_cells)
        self.grad_dev = self.ctx.enqueue_create_buffer[DType.float32](
            data.n_rows
        )
        self.hess_dev = self.ctx.enqueue_create_buffer[DType.float32](
            data.n_rows
        )
        self.out_g_dev = self.ctx.enqueue_create_buffer[DType.float32](
            hist_size
        )
        self.out_h_dev = self.ctx.enqueue_create_buffer[DType.float32](
            hist_size
        )
        self.out_c_dev = self.ctx.enqueue_create_buffer[DType.uint32](
            hist_size
        )

        # Upload the binned matrix once; it is reused every call.
        with self.bins_dev.map_to_host() as host:
            var dst = host.unsafe_ptr()
            var src = data.bins.unsafe_ptr()
            for i in range(n_cells):
                dst.unsafe_store(i, src.unsafe_load(i))

    def build(
        mut self, grad: List[Float64], hess: List[Float64]
    ) raises -> Histogram:
        """Build a full-dataset histogram on the GPU."""
        if len(grad) != self.n_rows or len(hess) != self.n_rows:
            raise Error("gradient/hessian length must equal n_rows")

        var g_scale = _fixed_scale(grad)
        var h_scale = _fixed_scale(hess)

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

        self.ctx.enqueue_function[_hist_kernel](
            self.bins_dev.unsafe_ptr(),
            self.grad_dev.unsafe_ptr(),
            self.hess_dev.unsafe_ptr(),
            self.out_g_dev.unsafe_ptr(),
            self.out_h_dev.unsafe_ptr(),
            self.out_c_dev.unsafe_ptr(),
            Int32(self.n_rows),
            Int32(self.n_bins),
            Float32(g_scale),
            Float32(h_scale),
            Float32(1.0 / g_scale),
            Float32(1.0 / h_scale),
            grid_dim=self.n_features,
            block_dim=BLOCK_THREADS,
        )

        var hist_size = self.n_features * self.n_bins
        var g = _zeroed_f64(hist_size)
        var h = _zeroed_f64(hist_size)
        var c = _zeroed_int(hist_size)
        with self.out_g_dev.map_to_host() as host:
            var src = host.unsafe_ptr()
            var dst = g.unsafe_ptr()
            for i in range(hist_size):
                dst.unsafe_store(i, Float64(src.unsafe_load(i)))
        with self.out_h_dev.map_to_host() as host:
            var src = host.unsafe_ptr()
            var dst = h.unsafe_ptr()
            for i in range(hist_size):
                dst.unsafe_store(i, Float64(src.unsafe_load(i)))
        with self.out_c_dev.map_to_host() as host:
            var src = host.unsafe_ptr()
            var dst = c.unsafe_ptr()
            for i in range(hist_size):
                dst.unsafe_store(i, Int(src.unsafe_load(i)))
        return Histogram(g^, h^, c^, self.n_features, self.n_bins)


def build_histogram_gpu(
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64]
) raises -> Histogram:
    """One-shot GPU histogram build (uploads the binned matrix every call;
    use `GpuHistogramBuilder` for repeated builds on one dataset)."""
    var builder = GpuHistogramBuilder(data)
    return builder.build(grad, hess)
