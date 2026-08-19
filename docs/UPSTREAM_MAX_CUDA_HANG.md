# Upstream report: MAX kernel JIT hangs on an RTX 5090

Written: 2026-08-19

**This is a bug report against Modular's toolchain, drafted here and kept
here.** It is not a record of a mojotrees defect. The three defects this
project's first NVIDIA run did find were ours, were fixed, and are recorded
in `docs/GPU_VALIDATION.md`; this file is the one thing that run turned up
which this repository cannot fix, because it happens below any line of code
this repository owns.

It exists as a file rather than only as an issue for two reasons. The
evidence took a day and about eleven dollars of leased hardware to assemble
and should not live only in a web form. And the next person here who sees a
GPU fit stop returning needs to find this before they spend the day again.

## How confident this is

Stated before the evidence rather than after it, because the difference
between "we measured this" and "we concluded this" is the whole value of the
report.

**Measured, and not in doubt.** Mojo compiles this source for sm_120 and the
card runs it correctly for 66 assertions. Two full-fit tests never return.
The blocked thread's stack names `libKGENCompilerRTShared` and `libcuda`, the
GPU is idle, and no CPU is being burned. Four MAX-only probes covering every
primitive we use all pass.

**Inferred, and could be wrong.** That the fault is upstream rather than
ours. A deadlock inside a library you call can still be caused by how you
call it, and eleven eliminated explanations is not a proof, it is eleven
eliminated explanations. What is true is that everything testable on our side
came back clean and the two remaining candidates on our side are named below
as untested rather than dismissed.

## The one-paragraph version

We write no CUDA. mojotrees has no CUDA file, no HIP file and no Metal file,
by design: one Mojo source, and MAX lowers it per backend. On an NVIDIA RTX
5090 that lowering **works** -- 66 of our GPU assertions pass on the card,
including histogram kernels, partition kernels, scan primitives and score
updates. But one class of workload never returns. The process parks with 14
threads on futexes, the GPU at 0 percent and zero CPU ticks, and the stack
puts the block inside `libKGENCompilerRTShared.so` calling into
`libcuda.so.1`. Both are upstream of us. Four probes that import nothing from
our package exercise every primitive we use and all four pass, so it is not a
primitive we are misusing, and collapsing our kernel matrix from 40
instantiations to 12 does not move it, so it is not sheer volume.

## Environment

```text
Host        RunPod secure cloud, EU-CZ-1, Ubuntu 24.04.3, glibc 2.39
CPU         256 vCPU advertised (x86-64), cgroup cpu.max quota 27.2
GPU         NVIDIA GeForce RTX 5090, Blackwell, compute capability 12.0, sm_120
VRAM        32607 MiB
Driver      580.159.03
Toolchain   Mojo 1.0.0 (ed45d567), max >=26.5.0, pixi 0.76.2
Repo        github.com/mojotrees/mojotrees, commit 542962c
```

The driver clears MAX's stated floor of 580 by one patch release. Two
earlier pods in the same session landed on CUDA 12.4 hosts where the
container refused to start at all, so anyone reproducing this should check
`cudaVersion` before paying for the instance.

The RTX 50XX series is listed under *Known compatible for development* in
Modular's compatibility table rather than *Tested for serving*. That is
stated here as context, not as an excuse for the report: the same source
compiles and runs on the card, so the backend is close enough to working
that the remaining failure is worth a look.

## What works, which is most of it

Stated first, because a report that opens with a hang reads as "the backend
is broken" and that is not what we found.

| suite | result | wall |
|---|---|---|
| `test_gpu_scan_primitives` | 6 of 6 pass | 8 s |
| `test_gpu_partition_launches` | 8 of 8 pass | 10 s |
| `test_gpu_runtime` | 40 of 40 pass | 28 s |
| `test_gpu_raw_update_packing` | 2 of 2 pass | 10 s |
| `test_gpu_tiling` | 10 of 10 pass | 4 s |

66 assertions, all of them real device work, all under 30 seconds each.
`test_gpu_raw_update_packing` is worth naming twice: it is the test that
caught a genuine FMA contraction difference between NVPTX and Metal earlier
the same day, so it is a test with teeth and it passes.

## What hangs

`test_gpu_training` and `test_device` never return. Killed at a 240 s cap,
then at 600 s, then at 601 s. Both are full training fits, which is the one
shape that drives many distinct kernel instantiations through the runtime in
a single process.

The distinction that matters is that this is not slowness:

```text
CPU ticks over a 5 s window   0
GPU utilization               0 %
threads in futex_wait_queue   14
threads in poll                3
```

A slow compiler burns CPU until it finishes. This burns 18 to 30 percent for
a while and then goes to zero and stays there.

## The stack, from the child process

```text
#0    __futex_abstimed_wait_common64   <- blocked
#1-3  libcuda.so.1                     <- the NVIDIA driver
#5-7  libKGENCompilerRTShared.so       <- MAX's kernel-compiler runtime
```

Two facts about how this was captured, because both cost hours and both
invalidate the obvious approach.

**The binary forks.** The parent sits in `sigsuspend`, correctly waiting, and
the child is the one that wedges. Every backtrace we took before we noticed
was of the parent, and every conclusion drawn from those was wrong. `gdb`
needs `follow-fork-mode child`.

**`ptrace_scope=1` blocks attaching**, so `gdb` has to be the parent of the
process rather than attach to a running one.

**`~/.nv/ComputeCache` holds zero files after a run**, so nothing is being
cached and every process is recompiling kernels from scratch. That is the
observation that points at the JIT rather than at anything we hand it.

## What has been eliminated

Eleven explanations were tested on the card and eleven died. The ones worth
carrying into an upstream conversation:

| explanation | how it died |
|---|---|
| Slow NVPTX codegen, not a hang | `mojo build` compiles a hanging test in **31 s** and the resulting **binary** then hangs, with no compiler in the process. |
| Our kernel count, in the histogram families | `MOJOTREES_KERNEL_MATRIX_FULL=False` collapses those families from 40 instantiations to 12. Both tests still hang, at 600 s and 601 s. **This is narrower than "not volume", see below.** |
| MAX's allocator, launches or queue depth | Four MAX-only probes pass. See below. |
| CPU oversubscription | The container advertises 256 CPUs against a 27.2 CPU quota. Real, and separately fixed, and not this: pinning workers to 8 changed nothing. |
| Job contention | 32 timeouts at 4 concurrent jobs against 26 at 12. Fewer jobs made it worse. |
| Our own unsynchronized readback | A real defect, found and fixed. The hang is unchanged. |
| Resident grower queue depth | `MOJOTREES_GPU_TREE_RESIDENT=0` hangs identically on two tests. |

### The probes are the load-bearing part

`probes/cuda_deadlock_alloc.mojo`, `cuda_deadlock_launch.mojo`,
`cuda_deadlock_subbuffer.mojo` and `cuda_deadlock_parallel.mojo` import
**nothing** from this package. They are MAX and nothing else. Between them
they perform 8,000 device allocations, pinned host allocations, copies with
no drain between them, 3,000 kernel launches on a reused buffer, 3,000
allocate-then-launch cycles, 3,000 allocations plus launches with no
`synchronize()` at all until the end, sub-buffer windows onto a parent
allocation, and device access from eight concurrent workers.

```text
PHASE_A_OK  PHASE_B_OK  PHASE_C_OK  REPRO_COMPLETED_NO_DEADLOCK
PHASE_D_OK  PHASE_E_OK  PHASE_F_OK  REPRO2_COMPLETED_NO_DEADLOCK
```

Every primitive we use is individually sound on this device. Whatever the
trigger is, it is a combination or a scale that these do not reach.

`cuda_deadlock_parallel.mojo` is the one to read first, because it is the
closest to how a real fit drives the device. It shares **one**
`DeviceContext` across eight `sync_parallelize` workers and has them
allocate, allocate-and-launch, and launch against per-worker windows onto a
shared parent buffer. That is our access pattern, not a simplified one, and
it completes.

What no probe reaches is kernel VARIETY. All of them launch one kernel
repeatedly. A real fit drives many distinct instantiations through the JIT in
one process, and with `~/.nv/ComputeCache` empty every one of them is
compiled from scratch.

## What we have NOT done, stated plainly

**There is no minimal reproducer.** The smallest thing we can hand over is
"build this repository and run one of two tests", which is a large ask of
whoever picks the issue up. We got as far as proving the individual
primitives are fine and could not close the gap between that and the failing
fit before the hardware budget ran out. If Modular would rather have a
minimal case than a large one, say so and we will spend the next lease
bisecting toward it rather than measuring.

**The volume hypothesis is not actually dead.** The table above records that
collapsing the histogram families from 40 kernel instantiations to 12 changed
nothing, and that was the best-supported explanation we had. But it is one
family. If the trigger is sheer count across a process that JITs hundreds,
cutting 28 from one family would not be expected to move it, so what was
falsified is narrower than the headline claim. It is written here rather than
quietly dropped because an earlier revision of this file overstated it.

**One candidate on OUR side is named and untested.** This device reports
`max_shared_memory_per_block = 49152` against the M4's 32768, so our feature
group 16 at 256 bins **raises on every Mac and is accepted here**. That is a
code path no machine in this project's history has executed, and it is
specific to the histogram arm that the hanging tests drive. It is the single
cheapest next experiment and it has not been run. It could equally be
nothing: an over-large shared memory request would normally fail a launch
rather than park a thread on a futex.

## The one asymmetry we can see between the working backend and the hanging one

Added 2026-08-19, from the Metal control run.

We ran the kernel-variety probe on the Apple M4, where this project's entire
GPU suite passes, to check that 128 distinct kernels in one process is not
simply too much to ask of anybody. It is not: the M4 completes it, and
per-kernel cost is flat from kernel 0 to kernel 127 with no trend.

But two consecutive runs of the same binary on the same machine gave this:

```text
run 1, cold    128 kernels    1370 ms    steady-state 8-13 ms each
run 2, warm    128 kernels      74 ms    steady-state 0.4-1.4 ms each
```

Nothing in the program changed. The first kernel costs 8.5 ms in both runs
and every later one collapses, so **something is saving the compilation work
between processes on macOS.**

We are deliberately not naming the mechanism. Our first draft said MAX was
caching to disk; our own earlier audit of the MAX documentation
(`docs/STARTUP_LATENCY.md`) says the documented cache is per-`DeviceContext`,
in-memory, in-process, and that no on-disk kernel cache is documented at all.
So the likelier explanation is that macOS caches compiled Metal shaders below
MAX entirely. If either reading is wrong, please correct it; we would rather
be told than guess in public.

On the RTX 5090, `~/.nv/ComputeCache` holds **zero files** after a full run.
If the amortization above is real and has no equivalent there, then the
backend that works pays for kernel compilation once and the backend that
hangs pays it in every process, and a fit that instantiates many kernels is
carrying a cost on CUDA that it never carries on Metal.

We are not claiming that is the cause of the hang. We are saying it is the
one asymmetry visible from outside, it is consistent with where the stack
points, and it is cheap to check.

## Reproduction

```bash
git clone https://github.com/mojotrees/mojotrees && cd mojotrees
git checkout 542962c
pixi run build-pkg                       # ~30 s, exit 0, warnings only
bash tools/with_timeout.sh 600 pixi run mojo run -I src -I tests \
    tests/test_gpu_training.mojo         # returns 124
```

The four probes, which should all pass and which establish that the machine
and the toolchain are basically working:

```bash
pixi run probe-cuda-alloc
pixi run probe-cuda-launch
pixi run probe-cuda-subbuffer
pixi run probe-cuda-parallel
```

And the variety probe, which should be run **twice**, because the difference
between the two runs is the measurement:

```bash
pixi run probe-cuda-variety      # cold
pixi run probe-cuda-variety      # warm, and on Metal this is 18x faster
```

To see the stack rather than just the timeout, `gdb` must be the parent:

```bash
gdb -q -ex 'set follow-fork-mode child' -ex run \
    -ex 'thread apply all bt' --args ./test_gpu_training
```

## What we are asking for

1. Whether a hang inside `libKGENCompilerRTShared` against `libcuda` on
   sm_120 is known, and whether there is a workaround or a version that does
   not have it.
2. Whether MAX's kernel JIT is expected to write to `~/.nv/ComputeCache`, and
   whether an empty cache after a full run indicates a misconfiguration on
   our side. **This is now our leading question, see below.**
3. Whether a single process compiling many distinct kernel instantiations at
   run time is a supported shape, or whether we should be pre-compiling.

## Status

| step | state |
|---|---|
| Evidence assembled | done |
| Report drafted | done, this file |
| Filed upstream | **not yet.** Needs a decision on where (Modular forum against GitHub issue) and Andrew's go |
| Minimal reproducer | not attempted, see above |
| Shared memory probe | not written, see above |
| Kernel-variety probe | **written and run on the M4 control.** 128 distinct kernels complete, cost is flat, and the cold-to-warm gap exposed the cache asymmetry above. Not yet run on NVIDIA |

Nothing here has been sent anywhere. Filing it is an outward action and this
project does not take those without an explicit go.
