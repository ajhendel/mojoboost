"""MAX-only reproduction, phase 4: does device access from PARALLEL WORKERS
deadlock?

NO mojotrees code is imported. This is the last cheap candidate for the hang
on an NVIDIA RTX 5090 where roughly 60% of the GPU suite parks in
`M::Driver::DeviceContext::enqueueCreateBuffer` with the GPU idle.

Seven explanations have been eliminated, three of them by probes 1-3, which
between them ran 8,000 device allocations, pinned host buffers, copies with no
drain, 3,000 kernel launches, allocate-then-launch cycles, unbounded queue
depth, and sub-buffers in the shipped one-parent-two-windows shape. All
passed.

Every one of those probes was SINGLE-THREADED. mojotrees is not: fits reach
the device from inside `sync_parallelize`. A lock held across a driver call is
exactly the shape that is harmless serially and deadlocks under a race, which
would explain why no amount of serial volume reproduces it.

  K  allocate from parallel workers, no launches
  L  allocate and launch from parallel workers
  M  L with a shared parent buffer and sub-buffer windows per worker
"""

from std.sys import has_accelerator
from std.gpu import global_idx
from max.algorithm import sync_parallelize
from max.gpu.host import DeviceContext


def _bump_kernel(
    xs: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
):
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
        comptime WORKERS = 8

        print("PHASE_K_START allocate from parallel workers")
        for round in range(200):
            @parameter
            def alloc_only(w: Int) capturing:
                try:
                    var b = ctx.enqueue_create_buffer[DType.float32](N)
                    _ = b
                except:
                    pass
            sync_parallelize[alloc_only](WORKERS)
            if round % 40 == 0:
                print("  K", round)
        ctx.synchronize()
        print("PHASE_K_OK")

        print("PHASE_L_START allocate AND launch from parallel workers")
        for round in range(200):
            @parameter
            def alloc_launch(w: Int) capturing:
                try:
                    var b = ctx.enqueue_create_buffer[DType.float32](N)
                    ctx.enqueue_function[_bump_kernel](
                        b.unsafe_ptr(), Int32(N),
                        grid_dim=GRID, block_dim=BLOCK
                    )
                except:
                    pass
            sync_parallelize[alloc_launch](WORKERS)
            if round % 40 == 0:
                print("  L", round)
        ctx.synchronize()
        print("PHASE_L_OK")

        print("PHASE_M_START shared parent, per-worker windows, launches")
        for round in range(200):
            var parent = ctx.enqueue_create_buffer[DType.float32](N * WORKERS)
            @parameter
            def win_launch(w: Int) capturing:
                try:
                    var win = parent.create_sub_buffer[DType.float32](
                        w * N, N
                    )
                    ctx.enqueue_function[_bump_kernel](
                        win.unsafe_ptr(), Int32(N),
                        grid_dim=GRID, block_dim=BLOCK
                    )
                except:
                    pass
            sync_parallelize[win_launch](WORKERS)
            if round % 40 == 0:
                print("  M", round)
        ctx.synchronize()
        print("PHASE_M_OK")
        print("REPRO4_COMPLETED_NO_DEADLOCK")
