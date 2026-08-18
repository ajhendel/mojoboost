"""MAX-only reproduction, phase 2: does allocation + KERNEL LAUNCH deadlock?

NO mojotrees code is imported. This is the discriminating experiment for a
hang observed on an NVIDIA RTX 5090 (driver 580.159.03, Mojo 1.0.0) where
roughly 60% of the GPU suite parks with this stack:

    #3  ___pthread_mutex_lock
    #4-10 libAsyncRTMojoBindings.so
    #11 M::Driver::DeviceContext::enqueueCreateBuffer(unsigned long)

Phase 1 of this repro (allocations + pinned host buffers + copies, 8000
iterations, no kernel launches) COMPLETED CLEAN. So MAX's allocator is not
deadlocking on its own. The one operation phase 1 never performed, and that
every wedged test performs, is a kernel launch.

Three phases, each printing progress, so a hang localizes itself:
  D  launch a kernel repeatedly, reusing one buffer (no allocation churn)
  E  allocate a fresh buffer per iteration, then launch on it
  F  E, with no synchronize at all until the very end -- the deep-queue case

If any phase hangs, the reproduction is upstream and this file is the bug
report. If all three complete, the cause needs something else mojotrees does
and the next variable to add is sub-buffers or multi-stream use.
"""

from std.sys import has_accelerator
from std.gpu import global_idx
from max.gpu.host import DeviceContext


def _bump_kernel(
    xs: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
):
    """Trivial elementwise kernel. The point is the launch, not the work."""
    var i = global_idx.x
    if i >= Int(n):
        return
    xs[unsafe_offset=i] = xs[unsafe_offset=i][0] + 1.0


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; nothing to reproduce")
    else:
        var ctx = DeviceContext()
        print("device:", ctx.name())
        print("api:", ctx.api())
        comptime N = 4096
        comptime BLOCK = 256
        comptime GRID = 16

        print("PHASE_D_START launch only, one reused buffer")
        var d = ctx.enqueue_create_buffer[DType.float32](N)
        for i in range(3000):
            ctx.enqueue_function[_bump_kernel](
                d.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
            )
            if i % 500 == 0:
                ctx.synchronize()
                print("  D", i)
        ctx.synchronize()
        print("PHASE_D_OK")

        print("PHASE_E_START allocate then launch, every iteration")
        for i in range(3000):
            var e = ctx.enqueue_create_buffer[DType.float32](N)
            ctx.enqueue_function[_bump_kernel](
                e.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
            )
            if i % 500 == 0:
                ctx.synchronize()
                print("  E", i)
        ctx.synchronize()
        print("PHASE_E_OK")

        print("PHASE_F_START allocate + launch, NO synchronize until the end")
        for i in range(3000):
            var f = ctx.enqueue_create_buffer[DType.float32](N)
            ctx.enqueue_function[_bump_kernel](
                f.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
            )
            if i % 500 == 0:
                print("  F", i)
        ctx.synchronize()
        print("PHASE_F_OK")
        print("REPRO2_COMPLETED_NO_DEADLOCK")
