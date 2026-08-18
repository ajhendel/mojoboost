"""MAX-only reproduction, phase 3: does create_sub_buffer deadlock?

NO mojotrees code is imported. This is the third and most targeted probe for
the hang observed on an NVIDIA RTX 5090 where roughly 60% of the GPU suite
parks in `M::Driver::DeviceContext::enqueueCreateBuffer` with the GPU idle.

Probes 1 and 2 both PASSED: 8,000 device allocations, pinned host buffers,
copies with no drain, 3,000 kernel launches, 3,000 allocate-then-launch
cycles, and 3,000 allocations plus launches with no synchronize until the end.
So allocation, copy, launch and queue depth are all sound on this device.

`create_sub_buffer` is the operation neither probe exercised, and it sits on
the exact path that hangs. From gpu_split_search.mojo:385, `records_dev` owns
one allocation and `rec_i_dev` / `rec_f_dev` are windows onto two regions of
it, so every split record read goes through a sub-buffer.

Four phases:
  G  make sub-buffers and drop them, no launches
  H  make sub-buffers and launch a kernel on the window
  I  the shipped shape: one parent, two disjoint windows, launch on both
  J  I with no synchronize until the very end
"""

from std.sys import has_accelerator
from std.gpu import global_idx
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

        print("PHASE_G_START sub-buffers, no launches")
        for i in range(3000):
            var parent = ctx.enqueue_create_buffer[DType.float32](N * 2)
            var w = parent.create_sub_buffer[DType.float32](0, N)
            _ = w
            if i % 500 == 0:
                print("  G", i)
        ctx.synchronize()
        print("PHASE_G_OK")

        print("PHASE_H_START sub-buffer then launch on the window")
        for i in range(3000):
            var parent = ctx.enqueue_create_buffer[DType.float32](N * 2)
            var w = parent.create_sub_buffer[DType.float32](0, N)
            ctx.enqueue_function[_bump_kernel](
                w.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
            )
            if i % 500 == 0:
                ctx.synchronize()
                print("  H", i)
        ctx.synchronize()
        print("PHASE_H_OK")

        print("PHASE_I_START one parent, two disjoint windows, launch on both")
        for i in range(3000):
            var parent = ctx.enqueue_create_buffer[DType.float32](N * 2)
            var wi = parent.create_sub_buffer[DType.float32](0, N)
            var wf = parent.create_sub_buffer[DType.float32](N, N)
            ctx.enqueue_function[_bump_kernel](
                wi.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
            )
            ctx.enqueue_function[_bump_kernel](
                wf.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
            )
            if i % 500 == 0:
                ctx.synchronize()
                print("  I", i)
        ctx.synchronize()
        print("PHASE_I_OK")

        print("PHASE_J_START same as I, NO synchronize until the end")
        for i in range(3000):
            var parent = ctx.enqueue_create_buffer[DType.float32](N * 2)
            var wi = parent.create_sub_buffer[DType.float32](0, N)
            var wf = parent.create_sub_buffer[DType.float32](N, N)
            ctx.enqueue_function[_bump_kernel](
                wi.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
            )
            ctx.enqueue_function[_bump_kernel](
                wf.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
            )
            if i % 500 == 0:
                print("  J", i)
        ctx.synchronize()
        print("PHASE_J_OK")
        print("REPRO3_COMPLETED_NO_DEADLOCK")
