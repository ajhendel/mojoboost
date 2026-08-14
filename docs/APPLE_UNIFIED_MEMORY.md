# Apple unified memory: what is settled, what is not, and how to find out

Apple silicon puts the CPU and the GPU on one physical memory pool. That is a
hardware fact, it is not in dispute, and it is also not, by itself, a reason
to change a single line of `histogram_gpu.mojo`.

What would be a reason is a different statement: that a route MAX and Mojo
actually expose lets mojoboost hand the GPU a buffer the CPU already filled,
with no second copy and no second resident allocation, and get the right
answer back. That statement is unproven here. This document is the experiment
that would settle it, the rules for running it honestly, and an explicit
account of which claims each possible outcome would and would not license.

## Status

| Route | Implemented in the driver | Compiled | Executed | Result |
|---|---|---|---|---|
| `copy_staged` (baseline) | yes | **no** | **no** | **none** |
| `copy_direct` | yes | **no** | **no** | **none** |
| `map_write` | yes | **no** | **no** | **none** |
| `host_direct` | yes | **no** | **no** | **none** |
| `wrapped_host_buffer` | no | **no** | **no** | **none** |

`bench/apple/unified_memory.mojo` has never been compiled and never been run.
No number it would print exists. Nothing in this repository, this document
included, is a measurement of unified-memory behavior, a zero-copy claim, or
a performance claim. Every cell above stays **none** until someone runs the
protocol below and pastes real output into the record section at the end.

## Two claims, only one of which is free

These get conflated constantly, so they are separated here and kept separate
everywhere else in the repository.

**Claim 1, unified physical memory.** The CPU and GPU address the same DRAM.
True on Apple silicon by construction. It costs nothing to say and it implies
nothing about any particular API.

**Claim 2, zero copy for our data.** The specific buffer mojoboost fills on
the host is the same physical bytes the kernel reads, with no staging
allocation, no blit, and no page migration on first device touch.

Claim 1 makes Claim 2 *possible*. It does not make it true. A runtime is free
to allocate device-private storage and blit into it on a unified-memory part,
and several do, for entirely reasonable reasons (allocation placement, cache
mode, alignment, coherency granularity, driver-managed residency). Claim 2 is
about what the software stack in front of the hardware actually does, and the
only way to know is to measure it.

The failure mode this document exists to prevent is writing "Apple has unified
memory, so our transfers are free" in a benchmark table or a README. That
sentence asserts Claim 2 from Claim 1 and it is not supported.

## What the driver does

`bench/apple/unified_memory.mojo` runs the same trivially bandwidth-bound
checksum kernel over the same payload through every host-to-device route this
Mojo version is known to provide, and reports where the time went and whether
the device saw the right bytes.

The kernel is deliberately dumb. It sums every payload byte with a per-index
weight into one Int32 accumulator, in a grid-strided loop, with one global
atomic per thread. It is not a histogram kernel, it is not tuned, and it is
not a proposal for anything. It exists so that every route is compared under
an identical consumer, and so that a route that failed to publish its bytes is
caught rather than rewarded.

### The routes

| Route | Host side | Publish step | Device side |
|---|---|---|---|
| `copy_staged` | write into a pinned `HostBuffer` | `enqueue_copy` to a `DeviceBuffer` | reads the device buffer |
| `copy_direct` | write into a plain heap `List` | `enqueue_copy` to a `DeviceBuffer` | reads the device buffer |
| `map_write` | write through `DeviceBuffer.map_to_host()` | leaving the `with` block | reads the device buffer |
| `host_direct` | write into a pinned `HostBuffer` | none at all | reads the host buffer pointer |
| `wrapped_host_buffer` | write into host memory | none at all | reads a non-owning `DeviceBuffer` over that pointer |

`copy_staged` is the baseline because it is exactly what
`GpuHistogramBuilder.stage_gradients` plus `upload_staged` do today. Every
other route is measured against it.

`copy_direct` exists to price *pinning* on its own, and it is built so that
pinning is the only thing that varies. Both copy routes write the payload into
a host buffer of the same size and enqueue the same copy from it; only the
buffer differs. Skipping the write on the unpinned route would be the obvious
shortcut and it would be wrong twice over, once because it varies two things at
a time, and once because it would not model the trainer, whose
`stage_gradients` has to write its Float64-to-Float32 conversion somewhere no
matter which buffer receives it. If pinning turns out to be the whole story
then unified memory is not what to go looking at first.

One caveat, on their round-0 numbers only. The unpinned `List` is faulted in by
its own construction inside the allocation window, while whether the pinned
buffer is resident before its first write is the runtime's business. Compare
these two routes on the steady state and treat their round-0 difference as
suspect. Round 0 remains meaningful within each route and across the mapped
routes, which is where it matters.

`map_write` is the one mapped route the current API definitely provides. Its
cost is split deliberately: `write_ns` covers establishing the mapping and the
stores through it, `publish_ns` covers leaving the block, which is where the
runtime does whatever it does to make those writes visible. On a discrete GPU
that block exit is a real upload. On Apple it may be nothing. The driver
reports the split so the two cases are distinguishable; it does not and cannot
decide between them from timings alone.

`host_direct` is the only implemented route that could be zero copy. It
allocates no device payload buffer, issues no copy, and passes the pinned host
buffer's pointer straight into the kernel as the payload argument.

`wrapped_host_buffer` is Route 5 below. It is **not implemented**, and the
driver reports it as `not_probed` rather than guessing.

### Route 5, and why it is not in the driver

The candidate is a non-owning `DeviceBuffer` constructed over a host pointer:

```mojo
var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
# ... fill host ...
var view = DeviceBuffer[DType.uint8](ctx, host.unsafe_ptr(), n, owning=False)
ctx.enqueue_function[_checksum_kernel](
    view.unsafe_ptr(), out.unsafe_ptr(), Int32(n), stride,
    grid_dim=blocks, block_dim=BLOCK_THREADS,
)
```

That constructor is documented, but its behavior when handed a *host*
allocation rather than a device one is not, and it has not been compiled here.
Writing it into the driver and reporting whatever came out would be reporting
an unverified route as an available one.

To promote it: paste the snippet into a fifth route runner shaped like
`_run_host_direct`, compile, and record one of three outcomes.

- It does not compile. That is the answer. Record "API not available in this
  Mojo version" with the compiler message and the version.
- It compiles and raises. Record `unsupported` with the runtime message.
- It compiles, runs, and the checksum matches. It is now a candidate, subject
  to exactly the same external evidence `host_direct` needs before anyone says
  "zero copy".

This three-outcome shape is general. In Mojo a missing method is a compile
error, not a catchable one, so `try`/`except` can only report "this API exists
and refused", never "this API does not exist". The driver therefore
distinguishes `unsupported` (the route raised at runtime) from `not_probed`
(the route was never compiled). Neither is ever reported as `ok`.

### What is measured, per route

This is the complete set of keys emitted per route, in emission order, and it
is the contract lane A8's schema should be written against.

| Metric | Meaning |
|---|---|
| `status` | `ok`, `unsupported`, `wrong`, or `not_probed` |
| `detail` | emitted only when non-empty: the raise message, or why the checksum failed |
| `comparable` | `1` only for `ok`; `0` means the timings must not be compared |
| `payload_bytes` | the payload size this route was run at |
| `alloc_ns` | creating every buffer the route needs, plus one synchronize |
| `round0_write_ns` | first host write of the payload, carrying first-touch page faults |
| `round0_publish_ns` | first publish, carrying any first-use migration or residency work |
| `round0_total_ns` | the whole first round |
| `write_ns` | steady-state host write, mean over rounds 1..N-1 |
| `publish_ns` | steady-state publish, mean over rounds 1..N-1 |
| `kernel_ns` | memset plus kernel enqueue, mean |
| `sync_ns` | the synchronize that follows, mean |
| `readback_ns` | reading the checksum back, mean |
| `contend_ns` | host work inside the launch window, when contention mode is on; disjoint from `sync_ns` and part of the round |
| `round_mean_ns`, `round_min_ns` | steady-state round cost |
| `host_alloc_bytes`, `device_alloc_bytes`, `allocated_bytes` | what the route allocated on each side, and their sum |
| `checksum`, `expected` | the correctness gate |
| `round0_over_steady` | first round divided by the steady-state mean; omitted when there is no steady state |
| `steady_ns_per_byte` | steady-state round cost per payload byte; omitted when there is no steady state |

Two of these are derived rather than measured and are printed only so nobody
recomputes them by hand. `round0_over_steady` is a ratio of measured times, not
evidence about page migration on its own, and `steady_ns_per_byte` is what the
size ladder's regression test keys on.

A route reported `unsupported` prints `status` and `detail` and nothing else,
because it produced no measurements. A route reported `wrong` prints everything,
flagged `comparable: 0`, because how a broken route behaved is worth knowing
and comparing it against a working one is not.

Round 0 is always reported separately from the steady state. A mean that folds
first touch into the recurring cost hides the single number this whole
experiment is about. Page migration, if it happens, happens once.

`contend_ns` and `sync_ns` are disjoint, and both are round time. The
contention work runs on the same thread that later waits, so it is serialized
before the wait rather than overlapped with it; timing the two together would
book host time as device wait and manufacture a contention effect that is not
there. What overlaps is the device work, which has a consequence worth knowing
before reading a contended run: if the host work outlasts the kernel, `sync_ns`
collapses toward zero and the contention shows up in `contend_ns` and in the
round total instead. A near-zero `sync_ns` under contention is therefore not
evidence that the device was unaffected.

Because of that, the contention question can only be answered by comparing two
whole runs, one with `MOJOBOOST_UM_CONTEND=1` and one without, on the same
payload size. Read `round_mean_ns` and `kernel_ns + sync_ns` across the pair.
Never read contention out of a single run.

### The correctness gate

Every route is scored against a CPU reference checksum over the same bytes. A
route whose device checksum disagrees is reported `wrong`, is flagged
`comparable: 0`, and its timings must not be compared with anything. This is
not pedantry. A route that silently skipped the transfer would post the best
numbers in the run and be useless.

Byte 0 of the payload carries the round number and the expected checksum is
adjusted by the known delta, so a route that publishes round 0's bytes forever
fails from round 1 rather than passing on stale data. The checksum is a
wrapping Int32 sum; integer addition wraps deterministically and is
order-independent, so the grid-strided device accumulation and the sequential
host reference agree exactly rather than approximately.

## What the driver cannot measure, and what has to run alongside it

Peak resident memory, memory compression, and swap are not readable from
inside the process by any means this driver is willing to claim. It prints
`um.marker.begin_ns` and `um.marker.end_ns` so an external capture can be
lined up with the phases, and that external capture is **mandatory**, not
optional. The bytes a route reports allocating are the bytes it asked for, not
the bytes the system committed, and the whole zero-copy question is precisely
the gap between those two numbers.

Run, on macOS:

```sh
vm_stat > /tmp/um_vmstat_before.txt
/usr/bin/time -l mojo run -I src bench/apple/unified_memory.mojo 256 8 \
    > /tmp/um_run.txt 2> /tmp/um_time.txt
vm_stat > /tmp/um_vmstat_after.txt
```

From `/tmp/um_time.txt`, record `maximum resident set size`. From the two
`vm_stat` captures, record the deltas in `Pages occupied by compressor`,
`Swapins`, and `Swapouts`. A run with a nonzero swap delta is void: rerun it on
a quieter machine.

For the residency question specifically, `footprint -p <pid>` during the run
distinguishes dirty from compressed from swapped pages, and a Metal System
Trace in Instruments shows whether a blit encoder ran between the host write
and the kernel. The blit trace is the single most direct piece of evidence
about Claim 2 available, and it is the one a timing number cannot substitute
for.

## Protocol

The measurement is worth nothing if the machine was busy, so the machine
state is part of the procedure, not a footnote.

1. **Idle machine.** Nothing else running. No other Mojo, pixi, or pytest
   process, which matters more than usual in this repository because parallel
   lanes share one machine. No browser, no indexing, no Time Machine. Power
   adapter connected.
2. **Cool start.** At least two minutes idle before the first run so the run
   does not begin on a thermally loaded part.
3. **Record the machine.** Chip, core counts, total RAM, free RAM at start,
   macOS version, Mojo version, MAX version. The driver records the device
   attributes it can read; it cannot read any of these.
4. **Payload sizes.** Run at least three: one comfortably in cache-adjacent
   territory, one at a realistic dataset size, one large enough to be a
   meaningful fraction of RAM. `256`, `1024`, and `4096` MiB is a reasonable
   default set on a 16 GB machine. Note that a payload is allocated three
   times over in a full run (the source list, plus whatever each route
   allocates), so size the largest run accordingly.
5. **Rounds.** At least 8, so the steady state has 7 samples and round 0 is
   clearly separable.
6. **Repeat each configuration three times**, as three separate process
   launches, and report the spread. A single run of a memory experiment on a
   general-purpose OS is an anecdote.
7. **Contention.** Run the full set twice, once with `MOJOBOOST_UM_CONTEND=1`.
   Report the two side by side. Do not mix them.
8. **Ladder last, and deliberately.** `MOJOBOOST_UM_LADDER=1` doubles the
   payload until a route fails or per-byte round time regresses past
   `MOJOBOOST_UM_LADDER_PCT` (default 200) percent of the smallest size's.
   This is the mode that can push a machine into the compressor and into swap.
   Run it alone, on an idle machine, with the `vm_stat` capture bracketing it,
   and treat its answer as a property of that machine in that memory state.
   It is not a hardware limit and must never be quoted as one.

## Reading the results

The point of the table below is that most outcomes do not license the
interesting claim, and it is worth knowing that before the run rather than
after.

| Observation | What it licenses | What it does not |
|---|---|---|
| `host_direct` status `unsupported` | "This Mojo version does not accept a host-buffer pointer as a kernel argument." | Anything about Apple hardware. |
| `host_direct` status `wrong` | "The pointer is accepted but the device does not see host writes through it." Record the message; this is a correctness trap worth documenting loudly. | Nothing else. Do not report its timings. |
| `host_direct` status `ok`, `publish_ns` zero | "The device read the correct bytes with no copy issued by us." | **Not** "zero copy". The runtime may have migrated pages or blitted behind the enqueue. |
| The above, plus peak RSS showing one payload-sized allocation, not two | Strong evidence for Claim 2. | Still not proof on its own; pair it with the trace. |
| The above, plus a Metal System Trace with no blit encoder between write and kernel | This is the evidence. Now the word "zero copy" is earned, for this Mojo version, this OS, and this chip. | Any other chip, OS, or Mojo version. Say which one it was measured on. |
| `map_write` `publish_ns` near zero and `sync_ns` unchanged from baseline | The block exit is cheap here. | That no copy happened. Cheap is not free and free is not absent. |
| `round0_over_steady` near 1.0 | First touch cost nothing measurable at this size. | That there is no page migration; it may simply be small relative to the write. |
| `round0_over_steady` large | Something one-time and expensive happens on first touch. Worth chasing, since mojoboost uploads the binned matrix exactly once per session. | Which of allocation, fault, or migration it was. That needs the trace. |
| Any route faster than `copy_staged` | That route was faster **in this driver, on this payload, on this machine**. | That mojoboost would get faster. The trainer's transfer is one part of a round; see below. |

That last row is the one most likely to be over-read, so it gets stated
plainly. This driver measures a transfer under a synthetic consumer. mojoboost
training moves a binned matrix once per session and gradients once per round,
and the current end-to-end GPU measurement on an M4 is *slower* than the CPU
trainer, with per-node kernel launches and full-dataset scans dominating.
Making the transfer free would not by itself change that verdict. A route win
here is a reason to run `bench/bench_train_gpu.mojo`, not a substitute for it.

## Output grammar

Every line is `um.<scope>.<key>: <value>`, one metric per line, so a run can
be parsed with a line filter and no state. Scopes are `um.<route>` for the
four implemented routes, `um.wrapped_host_buffer` for the unprobed one,
`um.device` for device attributes, `um.marker` for the external-capture
brackets, `um.ladder` for the size ladder, and bare `um.` for run-level
fields. Status values are exactly `ok`, `unsupported`, `wrong`, `not_probed`.
Times are integer nanoseconds, sizes are integer bytes, ratios are Float64.

This grammar is stable and is what the Apple benchmark schema
(`bench/apple/schema.json`, lane A8) should consume. Any change to it needs
that lane to change with it.

## Invocation

```sh
mojo run -I src bench/apple/unified_memory.mojo [payload_mib] [rounds]
```

Defaults are 256 MiB and 8 rounds.

| Variable | Effect |
|---|---|
| `MOJOBOOST_UM_CONTEND=1` | host mutates an unrelated payload-sized buffer inside the launch window |
| `MOJOBOOST_UM_LADDER=1` | run the size ladder after the fixed-size run |
| `MOJOBOOST_UM_LADDER_MAX_MIB` | ladder ceiling, default 8192 |
| `MOJOBOOST_UM_LADDER_PCT` | per-byte regression cutoff in percent, default 200 |

The driver has no build-lock wrapper of its own. On a machine running parallel
lanes it must be serialized like every other heavy job, and its results are
void if it was not.

## If a route wins

Nothing in `histogram_gpu.mojo` changes on the strength of this driver alone.
The proposed API changes that a win would justify, and the exact call sites
they would touch, are written up in
`handoffs/apple_a7_unified_memory.md`. That file is a proposal contingent on
evidence that does not exist yet, and it says so.

## Record

Empty. No run has been performed.

When the first run happens, append to this section: date, chip, core counts,
RAM, free RAM at start, macOS version, Mojo version, MAX version, the full
`um.*` output, the `/usr/bin/time -l` peak resident set size, the `vm_stat`
deltas, and whether an Instruments trace was taken. Three repeats per
configuration or it does not go in.
