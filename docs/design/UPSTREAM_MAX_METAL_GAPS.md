# Upstream report to Modular: expose the host-visible pointer of a shared-storage DeviceBuffer on Metal

**The headline ask, and everything else in this report is evidence under it:**

> On Metal, every MAX buffer is already `MTLStorageModeShared` -- one
> allocation the CPU and GPU both address. **Expose both addresses for it.**
> Today `DeviceBuffer.unsafe_ptr()` returns the `gpuAddress`, which faults on a
> host load, and `HostBuffer.unsafe_ptr()` returns `contents`, which a kernel
> cannot address. The API hands out one name per allocation and never both.

That single API unblocks three things at once, and they are the reason this is
the headline rather than one gap among several:

- **zero-copy ingest** -- a NumPy array written straight into device memory
  rather than staged and copied;
- **zero-copy readback** -- split records and predictions read where they were
  written, instead of a blit whose measured cost is 202 microseconds for 136
  bytes of which under 4 microseconds is the transfer;
- **one histogram buffer shared between a CPU and a GPU backend**, which is a
  combined-engine design that is simply unavailable while the two cannot name
  the same allocation.

The hardware premise already holds: verified by execution, MAX's Metal
allocator has one `newBufferWithLength:options:` site with `options == 0`, so
every buffer is shared storage. **Nothing needs to change about how memory is
allocated. What is missing is a name.**

**Two further asks, smaller and independent:**

1. **Expose `maxCommandBufferCount` on the Metal queue.** MAX creates its queue
   with a bare `newCommandQueue` and never sets it, leaving the default depth of
   64. Verified three ways: one load site on `newCommandQueue`, zero load sites
   on both raising selectors, no environment variable, and four
   `DeviceContext` constructor overloads none of which takes one. **Measured
   consequence:** per-launch enqueue cost is flat at 6-7 microseconds through 64
   launches and rises to 14-17 beyond, a knee exactly where a 64-deep queue
   predicts. A leaf-wise tree in this library issues **278 command buffers
   between waits**, so it spends most of every tree past that knee paying
   roughly double.
2. **Expose batching of multiple launches into one command buffer.** On Metal
   every `enqueue_function` becomes its own single-encoder command buffer, which
   is what produces the 278 and what makes the queue depth bind at all. A
   library that knows its next ten launches have no host decision between them
   has no way to say so.

---

# Supporting detail: three Metal gaps in one subsystem

Status: **drafted, not filed.** Filing needs a human with a Modular account.
Everything below is a reproduction someone else can run, and every number is
labeled. Written 2026-08-16 by the `cheap-sync` lane.

Why it is drafted rather than worked around: a vendor path adopted without
filing the gap is a vendor path we keep forever. This is the report that has
to go out before anything in this repository reaches around MAX.

Environment for every reproduction below: Mojo 1.0.0 (ed45d567), MAX 1.0.0
from the `conda.modular.com/max` channel, Apple M4, macOS 15 (Darwin 25.5.0),
`ctx.api()` reports `metal`, `ctx.name()` reports `Apple M4`. The
disassembly items are against the `libMGPRT.dylib` shipped in that build,
38,658,720 bytes.

---

## The cost this is about

A Metal System Trace of one boosting round (`docs/METAL_TIMELINE.md`)
decomposed a single blocking readback:

```
stage                                  median us   % of round
the commit call itself                      10.0        1.5%
commit end -> GPU start                    298.1       44.3%
GPU moving bytes                             3.7        0.6%
GPU end -> completion signal               109.5       15.2%
completion signal -> next commit           164.8       24.2%
TOTAL, commit -> next commit               606.1       85.8%
```

**Measured.** 3.7 microseconds of 606 is the GPU moving bytes. The operation
whose entire purpose is data movement spends 0.6 percent of its time moving
data. Thirty-two of these per round is 85.8 percent of a 23.5 millisecond
round.

Every item below is a way that cost cannot currently be avoided on Metal,
where it can be on CUDA.

---

## Gap 1: no `MetalDeviceGraphBuilder`

`max.gpu.host.device_graph.DeviceGraph` exists as a type.
`DeviceGraph.create` raises:

```
createGraphBuilder() not supported on this device context
```

`CUDADeviceGraphBuilder` and `HIPDeviceGraphBuilder` are both present in the
same shipped runtime; the Metal equivalent is absent from a driver source set
that contains the other two. So a per-tree command sequence that is identical
in shape every tree cannot be captured once and replayed, and every launch
crosses the host.

Established by execution (the raise) and by disassembly (the two builders
that ship and the one that does not). This one was verified by the
`device-graph` lane in the same wave; it is included here because it is the
same subsystem and the same backend as the two below, and three gaps in one
report are one conversation rather than three.

## Gap 2: no event, fence, stream, host callback, or completion query

There is no way to wait for less than the whole queue, and no way to be told
that work finished without blocking a thread on it.

| API | result on a Metal `DeviceContext` |
|---|---|
| `create_event()` | raises `eventCreate is not supported on this device` |
| `create_stream()` | raises `createStream is not supported on this device` |
| `num_streams()` | returns 1 |
| `enqueue_cpu_function(f)` | raises `enqueue_cpu_function is only supported on CPU DeviceContexts` |
| `handle()`, `unsafe_ptr()`, `native_handle()`, `stream_handle()` | do not exist on `DeviceContext` |
| `query()`, `is_idle()`, `poll()`, `is_complete()` | do not exist on `DeviceContext` |

The runtime is not merely not exposing these; it is not using them.
**Measured by disassembly**, by locating each selector's metal-cpp
registration in the global selector table and then counting loads of the
resulting `SEL` global:

```
newSharedEvent                 0 call sites
newSharedEventHandle           0 call sites
newSharedEventWithHandle:      0 call sites
newEvent                       0 call sites
newFence                       0 call sites
encodeSignalEvent:value:       0 call sites
encodeWaitForEvent:value:      0 call sites
notifyListener:atValue:block:  0 call sites
addCompletedHandler:           0 call sites
commit                         8 call sites
waitUntilCompleted             6 call sites
```

`MTLSharedEvent` with `notifyListener:atValue:block:`, or simply
`addCompletedHandler:` setting a flag a caller can poll, would turn the
109.5-microsecond completion-signal term and the 164.8-microsecond
signal-to-next-commit term into something a spinning host can observe
directly. Neither is reachable today, and neither can be added from outside,
because no handle to the queue or to a command buffer leaves the API. A
downstream vendor shim is therefore not merely discouraged here, it is not
constructible.

**What would close it, smallest first:** any non-blocking completion query on
`DeviceContext` (`is_idle()`, or a token from `enqueue_function` with a
`ready()`); or `create_event()` implemented over `MTLSharedEvent`; or a
documented accessor returning the underlying `MTLCommandQueue`.

## Gap 3: no allocation is addressable from both sides

Apple silicon is unified memory and MAX already allocates accordingly.
**Measured by disassembly**: every Metal allocation goes through one
`newBufferWithLength:options:` call site, and the caller reached from the
device-context allocator passes `options == 0`, which is
`MTLStorageModeShared`. The memory is host-addressable.

The API hands out exactly one name for it per allocation kind, and never
both:

| Mojo object | pointer returned | usable by the host | usable by a kernel |
|---|---|---|---|
| `DeviceBuffer` | `gpuAddress` | **no**, faults | yes |
| `HostBuffer` | `contents` | yes | **no**, reads and writes elsewhere |

**Measured by execution.** A fresh `DeviceBuffer` on an M4 reported
0x10000080000, with consecutive allocations 256 bytes apart, and the first
host load faulted; host allocations in the same process sat near
0x100b4c000. A kernel handed a `HostBuffer`'s pointer as a destination left
all 34 words holding their pre-fill sentinel after a full `synchronize()`; a
kernel handed it as a source read zeros. Both directions, and both before and
after that buffer had been used as an `enqueue_copy` destination, so it is
not a registration-on-first-use effect. This is consistent with
`enqueue_function` binding arguments bindlessly by GPU address:
`gpuAddress` has one load site, `setBuffer:offset:atIndex:` has none, and
`useResource:usage:` has one.

The consequence is that reading a 136-byte result a kernel just wrote costs a
command buffer, on hardware where it could cost a load. It also means the
obvious workaround, allocating the `MTLBuffer` outside MAX and passing its
`gpuAddress` in, fails on residency, since `useResource:usage:` is called for
MAX's own allocations only.

**What would close it, smallest first:** `DeviceBuffer.host_ptr()` returning
`contents` where the storage mode allows it, with the documented ordering
requirement that the host must have observed a completion first; or making a
`HostBuffer` usable as a kernel argument by binding its `gpuAddress`.

---

## What we are not asking for

Not an asynchronous copy. **We found one.** `enqueue_copy` into a pinned
`HostBuffer` is already asynchronous on Metal in both directions, and only
the arbitrary-host-pointer overload takes the synchronous
commit-wait-memcpy path. That is a good default and we are not asking for it
to change. It is documented in this repository at
`docs/GPU_PORTABILITY.md` 6.5.1, where it also corrects an earlier claim of
ours that both overloads drained.

One request that follows from it, though, and it is cheap: **the asynchrony
is not documented, and the two overloads look identical at the call site.**
`enqueue_copy(dst_ptr=<HostBuffer pointer>, src_buf=...)` returns before the
copy has run and `enqueue_copy(dst_ptr=<heap pointer>, src_buf=...)` does
not, and nothing in the API reference distinguishes them. We found this by
reading stale bytes, and only under a kernel slow enough to lose the race;
under a fast kernel the incorrect code passes every time. A sentence in the
`enqueue_copy` documentation saying which destinations are asynchronous would
have saved this repository a wrong docstring that stood for a day and could
easily have become a silently wrong model.

## Reproductions

`probes/readback_cost.mojo` in this repository runs every claim above as a
gated arm and prints, per arm, whether it delivered the right bytes and what
it cost.

    pixi run probe-readback

Arms that are unavailable report why rather than being omitted, so a MAX
version that closes any of these gaps shows up as an arm changing from
`unavailable` to `ok` without anyone editing the harness.
