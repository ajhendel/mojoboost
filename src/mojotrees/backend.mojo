"""Histogram backend selection.

The trainer shares one implementation of objectives, boosting, tree growth,
gain calculation, and binning; only the expensive histogram accumulation
kernel is backend-specific. This module is that boundary: `CPU` dispatches to
the SIMD + multicore builders in `histogram.mojo`, `GPU` to the
shared-memory kernel in `histogram_gpu.mojo`.

The GPU backend requires an accelerator at runtime and carries Float32
precision (see `histogram_gpu.mojo`); the CPU backend is Float64.

This module dispatches single full-dataset histogram builds. For complete
GPU-resident training (per-leaf histograms, device-side row partitioning,
one persistent builder across all boosting rounds) use `train_gpu` in
`train_gpu.mojo`.
"""

from std.sys import has_accelerator

from .binning import BinnedMatrix
from .histogram import Histogram
from .histogram import build_histogram as _build_histogram_cpu
from .histogram_gpu import build_histogram_gpu as _build_histogram_gpu

comptime CPU = 0
comptime GPU = 1


def build_histogram_on[
    backend: Int = CPU
](
    data: BinnedMatrix, grad: List[Float64], hess: List[Float64]
) raises -> Histogram:
    """Build a full-dataset histogram on the selected backend."""
    comptime assert backend == CPU or backend == GPU, "unknown backend"
    comptime if backend == CPU:
        return _build_histogram_cpu(data, grad, hess)
    else:
        comptime if not has_accelerator():
            raise Error("GPU backend requested but no accelerator present")
        else:
            return _build_histogram_gpu(data, grad, hess)
