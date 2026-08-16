"""Does `max.gpu.host.device_graph` capture and replay on *this* device?

Run it, do not read about it:

```
pixi run mojo run tools/probe_device_graph.mojo
```

Why this file exists at all. `docs/GPU_PORTABILITY.md` section 6.5 records
that MAX 26.5.0 has no Metal implementation of the device-graph builder, and
it records that as **verified by execution** rather than derived from a
symbol table, because this repository has twice believed a documented claim
about a GPU API that was false. The finding has a shelf life: MAX ships a
`MetalDeviceContext.cpp` and two graph builders beside it, and the day a
third appears the answer flips. This is the thirty-line check that tells you
which world you are in, on whatever device you are holding, without trusting
this file's own prose about it.

What it establishes, in order, and it stops at the first thing that fails:

1. Does `DeviceGraph.create` return a graph on this device? On Apple silicon
   under MAX 26.5.0 it does not -- it raises, and the raise is the answer.
2. If it does, does a replay of a two-node graph with one explicit
   dependency edge produce the right answer? A graph API that captured
   nothing and silently no-opped would pass step 1 and fail here.
3. Does replaying the same instantiated graph again keep producing the right
   answer? A capture that is correct once and stale afterwards would pass
   step 2 and fail here.

The sequence captured is deliberately the smallest thing shaped like a tree
round: a producer node, then a consumer node that must observe the
producer's writes, ordered by an explicit `dependencies=` edge rather than
by enqueue order. Every launch argument is fixed at capture and the only
thing the second replay changes is device memory, which is the constraint a
captured per-tree graph would have to live inside.

This is a probe, not a test. It is referenced by no pixi task and no CI job
on purpose: it needs a device, it asserts a property of the *toolchain*
rather than of this repository, and its expected verdict differs by backend,
so a suite that ran it would be asserting which machine it was on.
"""

from std.gpu import block_idx, thread_idx
from std.sys import has_accelerator
from max.gpu.host import DeviceContext, DeviceGraph, DeviceGraphBuilder

# One threadgroup width, fixed at capture like every other launch argument.
comptime BLOCK: Int = 64
comptime N: Int = 256
comptime GRID: Int = N // BLOCK

comptime SEED: Int32 = 7
comptime DELTA: Int32 = 1


def _fill_kernel(dst: MutPointer[Int32, MutAnyOrigin], n: Int32, val: Int32):
    """Producer: writes `val` to every element."""
    var i = Int32(block_idx.x) * Int32(BLOCK) + Int32(thread_idx.x)
    if i < n:
        dst[unsafe_offset = Int(i)] = val


def _bump_kernel(dst: MutPointer[Int32, MutAnyOrigin], n: Int32, delta: Int32):
    """Consumer: reads what the producer wrote and adds `delta`.

    This is the node that makes the probe mean something. It is only correct
    if the graph honored the `dependencies=[producer]` edge, and its result
    is only correct *once* if replay re-runs the producer rather than
    resuming from whatever the device memory happened to hold.
    """
    var i = Int32(block_idx.x) * Int32(BLOCK) + Int32(thread_idx.x)
    if i < n:
        dst[unsafe_offset = Int(i)] = dst[unsafe_offset = Int(i)] + delta


def main() raises:
    comptime if not has_accelerator():
        # The whole body is inside the `else`, because a CPU-only build that
        # elaborates a GPU entry point dies at compile time on "Unknown GPU
        # architecture" and an Apple machine never reproduces it locally.
        raise Error("probe_device_graph: no accelerator on this build")
    else:
        with DeviceContext() as ctx:
            print("device:", ctx.name())

            var dev = ctx.enqueue_create_buffer[DType.int32](N)
            var host = ctx.enqueue_create_host_buffer[DType.int32](N)
            ctx.synchronize()
            var dst = dev.unsafe_ptr()

            def build(mut builder: DeviceGraphBuilder) raises {imm}:
                var producer = builder.add_function[_fill_kernel](
                    dst,
                    Int32(N),
                    SEED,
                    grid_dim=GRID,
                    block_dim=BLOCK,
                    dependencies=[],
                )
                _ = builder.add_function[_bump_kernel](
                    dst,
                    Int32(N),
                    DELTA,
                    grid_dim=GRID,
                    block_dim=BLOCK,
                    dependencies=[producer],
                )

            # Step 1. Construction. Everything below is unreachable on a
            # backend whose driver has no graph builder.
            var graph: DeviceGraph
            try:
                graph = DeviceGraph.create(ctx, build)
            except e:
                print("VERDICT: NOT SUPPORTED on this device")
                print("  DeviceGraph.create raised:", e)
                print(
                    "  See docs/GPU_PORTABILITY.md section 6.5. Under MAX"
                    " 26.5.0 this is what Metal does."
                )
                return
            print("DeviceGraph.create: ok")

            var expected = SEED + DELTA

            # Step 2. One replay.
            graph.replay()
            ctx.synchronize()
            ctx.enqueue_copy(host, dev)
            ctx.synchronize()
            var ok_once = True
            for i in range(N):
                if host[i] != expected:
                    ok_once = False
            print("replay x1 correct:", ok_once, "(host[0] =", host[0], ")")

            # Step 3. Two further replays of the same instantiated graph. The
            # expected value does not accumulate: the producer node overwrites
            # the buffer at the top of every replay, so a run that reports
            # SEED + 3 * DELTA has a graph that dropped the producer.
            graph.replay()
            graph.replay()
            ctx.synchronize()
            ctx.enqueue_copy(host, dev)
            ctx.synchronize()
            var ok_thrice = True
            for i in range(N):
                if host[i] != expected:
                    ok_thrice = False
            print("replay x3 correct:", ok_thrice, "(host[0] =", host[0], ")")

            if ok_once and ok_thrice:
                print("VERDICT: SUPPORTED and correct on this device")
            else:
                print(
                    "VERDICT: constructs but REPLAYS WRONG -- this is the"
                    " silent-no-op case and it is the worst one"
                )
