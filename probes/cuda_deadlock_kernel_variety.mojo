"""MAX-only reproduction, phase 5: does JITTING MANY DISTINCT KERNELS hang?

NO mojotrees code is imported. This is the decisive experiment for the
question the other four probes could not reach, and the question is not
"is our code wrong" but "is the toolchain sound at the scale we use it".

WHY THIS PROBE EXISTS, AND WHY IT IS THE IMPORTANT ONE

A hang was observed on an NVIDIA RTX 5090 (driver 580.159.03, Mojo 1.0.0)
in which a full training fit never returns: 14 threads on futexes, the GPU
idle, zero CPU ticks, and the blocked frame inside
`libKGENCompilerRTShared.so` calling into `libcuda.so.1`. Meanwhile 66 of
this project's GPU assertions PASS on the same card in under thirty seconds
each. So the backend works, and something about the failing shape does not.

Four probes cleared every primitive the failing fit uses: allocation, pinned
host buffers, copies with no drain, 3000 launches, deep queues, sub-buffer
windows, and one shared `DeviceContext` driven from eight `sync_parallelize`
workers. All pass. Every one of them launches ONE kernel.

The variable none of them move is how many DISTINCT kernels a single process
compiles at run time. A training fit drives many instantiations through the
JIT, and `~/.nv/ComputeCache` holds ZERO files after a run, so every one of
them is compiled from scratch, every time, in-process.

An earlier attempt to test this from the other end reduced this project's
histogram families from 40 kernel instantiations to 12 and the hang did not
move, which was read at the time as "kernel count is not the trigger". That
reading was too broad: one family was reduced, not the process's whole kernel
population, and a process compiling hundreds would not be expected to notice
28 fewer. This probe moves the variable directly instead, over two orders of
magnitude, with nothing else in the picture.

WHAT EACH OUTCOME MEANS, DECIDED BEFORE THE RUN

  - It HANGS at some N. Then the reproduction is upstream, it is minimal, it
    is a hundred lines that import only MAX, and the bug report stops needing
    "clone our repository and run our test suite". This is the good outcome
    for everyone except Modular.

  - It COMPLETES at every N, including N well above what a fit uses. Then
    kernel variety is not the trigger either, the elimination list grows by
    one, and the search moves back onto this project's side of the line --
    where the remaining named candidate is the 48 KiB shared-memory path that
    raises on every Mac and is accepted on this card.

Either answer is worth the run. There is no outcome where this teaches
nothing, which is the property a probe should have before it is worth paying
for hardware to run it.

TWO PHASES

  G  Compile and launch N distinct kernels once each, printing the wall time
     of every one. The shape of that curve is data on its own: if per-kernel
     JIT cost is flat, volume is cheap and a hang is a lock; if it climbs,
     there is an in-process accumulation and the hang may just be the far end
     of a curve that never turns over.

  H  Launch the same N kernels again, four more rounds. Round 1 pays the JIT.
     If rounds 2 through 5 are fast, MAX caches compiled kernels within the
     process and only the first touch is expensive. If they are just as slow,
     nothing is cached anywhere -- not on disk, per the empty ComputeCache,
     and not in memory either -- and that finding matters more than this
     probe's headline question.

RUNNING IT

    pixi run probe-cuda-variety

The pixi task wraps it in `tools/with_timeout.sh auto`, so a hang returns 124
rather than sitting on a leased machine until someone notices. Raise
MOJOTREES_BENCH_TIMEOUT if phase G is merely slow; the per-kernel print tells
you which it is, because a compiler that is working prints and a wedged one
does not.

It runs on Metal too, and should. The M4 is the control: this project's whole
GPU suite passes there, so if phase G hangs on the M4 as well then the
premise that this is an NVIDIA-side problem was wrong from the start.
"""

from std.sys import has_accelerator
from std.gpu import global_idx
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext


# How many distinct kernels to compile. Each TAG below is a separate comptime
# instantiation and therefore a separate symbol for the JIT to compile, which
# is the entire point of the parameter: it does nothing to the arithmetic.
#
# 128 rather than 1024 because the HOST binary carries one call site per
# instantiation and `mojo build` has to elaborate all of them, so an
# over-large sweep spends the budget on the wrong compiler. 128 is above what
# a single fit in this project instantiates and is still a binary that builds
# in a sensible time. Raise it if 128 completes cleanly and the question is
# still open.
comptime N_KERNELS = 128

comptime N = 4096
comptime BLOCK = 256
comptime GRID = 16

# Rounds after the first. Phase H repeats the whole set to separate the cost
# of compiling a kernel from the cost of launching one.
comptime EXTRA_ROUNDS = 4


def _variety_kernel[TAG: Int](
    xs: MutPointer[Float32, MutAnyOrigin],
    n: Int32,
):
    """One kernel per TAG, all doing the same trivial work.

    The addend is derived from TAG so the parameter is genuinely used and
    cannot be optimized into a single shared instantiation. The work is
    deliberately trivial: this probe measures compilation and dispatch, and
    any real arithmetic in here would only add noise to that.
    """
    var i = global_idx.x
    if i >= Int(n):
        return
    xs[unsafe_offset=i] = xs[unsafe_offset=i][0] + Float32(TAG + 1)


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; nothing to reproduce")
    else:
        var ctx = DeviceContext()
        print("device:", ctx.name())
        print("api:", ctx.api())
        print("n_kernels:", N_KERNELS)

        var d = ctx.enqueue_create_buffer[DType.float32](N)
        ctx.synchronize()

        print("PHASE_G_START compile and launch", N_KERNELS, "distinct kernels")
        var g0 = perf_counter_ns()
        comptime for k in range(N_KERNELS):
            var t0 = perf_counter_ns()
            ctx.enqueue_function[_variety_kernel[k]](
                d.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
            )
            ctx.synchronize()
            var ms = Float64(perf_counter_ns() - t0) / 1.0e6
            # Printed for EVERY kernel, not every tenth. The whole diagnostic
            # value of this probe when it hangs is knowing the exact index it
            # stopped at, and a sampled print loses that.
            print("  KERNEL", k, "ms", ms)
        var g_ms = Float64(perf_counter_ns() - g0) / 1.0e6
        print("PHASE_G_OK total_ms", g_ms)

        print("PHASE_H_START relaunch the same set,", EXTRA_ROUNDS, "rounds")
        for r in range(EXTRA_ROUNDS):
            var r0 = perf_counter_ns()
            comptime for k in range(N_KERNELS):
                ctx.enqueue_function[_variety_kernel[k]](
                    d.unsafe_ptr(), Int32(N), grid_dim=GRID, block_dim=BLOCK
                )
            ctx.synchronize()
            var r_ms = Float64(perf_counter_ns() - r0) / 1.0e6
            print("  ROUND", r + 1, "ms", r_ms)
        print("PHASE_H_OK")

        print("VARIETY_COMPLETED_NO_DEADLOCK")
