"""Minimal MAX-only reproduction: does device buffer allocation deadlock?

NO mojotrees code is imported. If this hangs, the defect is in MAX's CUDA
backend and belongs upstream. If it completes, the deadlock needs our usage
pattern to appear and the defect is ours.

Captured stack from a wedged mojotrees GPU test, NVIDIA RTX 5090, driver
580.159.03, Mojo 1.0.0, 2026-08-18:

    #3  ___pthread_mutex_lock
    #4-10 libAsyncRTMojoBindings.so
    #11 M::Driver::DeviceContext::enqueueCreateBuffer(unsigned long)
    #12 AsyncRT_DeviceContext_createBuffer_async

Three phases, each printing before and after, so a hang is localized:
  A  many device allocations, nothing else
  B  device + pinned host allocations interleaved
  C  allocate, copy, allocate again without draining between
"""

from std.sys import has_accelerator
from max.gpu.host import DeviceContext


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; nothing to reproduce")
    else:
        var ctx = DeviceContext()
        print("device:", ctx.name())
        print("api:", ctx.api())

        print("PHASE_A_START device allocations")
        for i in range(4000):
            var b = ctx.enqueue_create_buffer[DType.float32](4096)
            _ = b
            if i % 500 == 0:
                print("  A", i)
        print("PHASE_A_OK")

        print("PHASE_B_START device + pinned host allocations")
        for i in range(2000):
            var d = ctx.enqueue_create_buffer[DType.float32](4096)
            var h = ctx.enqueue_create_host_buffer[DType.float32](4096)
            _ = d
            _ = h
            if i % 250 == 0:
                print("  B", i)
        print("PHASE_B_OK")

        print("PHASE_C_START allocate, copy, allocate, no drain between")
        for i in range(2000):
            var d = ctx.enqueue_create_buffer[DType.float32](4096)
            var h = ctx.enqueue_create_host_buffer[DType.float32](4096)
            ctx.enqueue_copy(dst_buf=d, src_buf=h)
            var d2 = ctx.enqueue_create_buffer[DType.float32](4096)
            _ = d2
            if i % 250 == 0:
                print("  C", i)
        ctx.synchronize()
        print("PHASE_C_OK")
        print("REPRO_COMPLETED_NO_DEADLOCK")
