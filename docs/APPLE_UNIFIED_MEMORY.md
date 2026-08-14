# Apple unified memory: what is settled, what is not, and how to find out

Apple silicon puts the CPU and the GPU on one physical memory pool. That is a
hardware fact, it is not in dispute, and it is also not, by itself, a reason
to change a single line of `histogram_gpu.mojo`.

What would be a reason is a different statement: that a route MAX and Mojo
actually expose lets mojoboost hand the GPU a buffer the CPU already filled,
with no second copy and no second resident allocation, and get the right
answer back. That statement is unproven here. This document is the experiment
that would settle it, the rules for running it honestly, the policy seam that
would act on the answer, and an explicit account of which claims each possible
outcome licenses.

## Status

| Route | Direction | Implemented in the driver | Compiled | Executed | Result |
|---|---|---|---|---|---|
| `copy_staged` (baseline) | host to device | yes | **no** | **no** | **none** |
| `copy_direct` | host to device | yes | **no** | **no** | **none** |
| `map_write` | host to device | yes | **no** | **no** | **none** |
| `host_direct` | host to device | yes | **no** | **no** | **none** |
| `wrapped_host_buffer` | host to device | no | **no** | **no** | **none** |
| `out_copy` (baseline) | device to host | yes | **no** | **no** | **none** |
| `out_host_direct` | device to host | yes | **no** | **no** | **none** |
| `out_wrapped_host_buffer` | device to host | no | **no** | **no** | **none** |

`bench/apple/unified_memory.mojo` has never been compiled and never been run.
No number it would print exists. Nothing in this repository, this document
included, is a measurement of unified-memory behavior or a performance claim
of any kind. Every cell above stays **none** until someone runs the protocol
below and pastes real output into the record section at the end.

`src/mojoboost/unified_memory_policy.mojo` encodes the same emptiness in code:
`EvidenceLedger.installed()` reports `none` for every route, and the shipped
default is the staged copy for every buffer in the system.

## Two claims, only one of which is free

These get conflated constantly, so they are separated here and kept separate
everywhere else in the repository.

**Claim 1, unified physical memory.** The CPU and GPU address the same DRAM.
True on Apple silicon by construction. It costs nothing to say and it implies
nothing about any particular API.

**Claim 2, no duplication for our data.** The specific buffer mojoboost fills
on the host is the same physical bytes the kernel reads, with no staging
allocation, no blit, and no page migration on first device touch.

Claim 1 makes Claim 2 *possible*. It does not make it true. A runtime is free
to allocate device-private storage and blit into it on a unified-memory part,
and several do, for entirely reasonable reasons (allocation placement, cache
mode, alignment, coherency granularity, driver-managed residency). Claim 2 is
about what the software stack in front of the hardware actually does, and the
only way to know is to measure it.

There is a third statement that sits between them and is the one this
experiment can actually produce on its own:

**Claim 1.5, mojoboost issued no copy.** No `enqueue_copy` was called for this
payload. This is a fact about our source code. It is reported as
`copy_bytes_issued_total: 0` and it is *not* evidence for Claim 2: the runtime
may still migrate pages, blit behind an enqueue, or hold two physical copies,
and none of that is visible from inside the process.

The failure mode this document exists to prevent is writing "Apple has unified
memory, so our transfers are free" in a benchmark table or a README. That
sentence asserts Claim 2 from Claim 1 and it is not supported.

## What the driver does

`bench/apple/unified_memory.mojo` runs the same trivially bandwidth-bound
checksum kernel over the same payload through every delivery route this Mojo
version is known to provide, and reports where the time went, how many queue
drains the route owed, how many bytes it asked this library to copy, and
whether the device saw the right bytes.

The kernel is deliberately dumb. It sums every payload byte with a per-index
weight into one Int32 accumulator, in a grid-strided loop, with one global
atomic per thread. It is not a histogram kernel, it is not tuned, and it is
not a proposal for anything. It exists so that every route is compared under
an identical consumer, and so that a route that failed to publish its bytes is
caught rather than rewarded.

Both of its pointer arguments are `MutPointer[..., MutAnyOrigin]`, which is
the structural reason one kernel can serve every route: the payload pointer
may address a device buffer or a pinned host buffer, and so may the
accumulator. It is also the reason an eventual integration needs no kernel
signature change anywhere in `histogram_gpu.mojo`, only a change in which
pointer the launch site passes.

### The input routes: how the payload reaches the kernel

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

`host_direct` is the only implemented input route on which this library issues
no copy. It allocates no device payload buffer, issues no copy, and passes the
pinned host buffer's pointer straight into the kernel as the payload argument.

`wrapped_host_buffer` is Route 5 below. It is **not implemented**, and the
driver reports it as `not_probed` rather than guessing.

### The output route: how the result comes back

The routes above all answer "how do the bytes get to the kernel", and that is
only half of what a boosting round does. `download_raw` copies the histogram
back and drains the queue *once per tree node*, which makes it the most
frequent transfer in a fit by a wide margin: a 200-round fit growing 31 leaves
per tree moves the histogram about six thousand times and the binned matrix
once. So the driver runs the opposite direction too.

| Route | Accumulator | Return step | Drains per round |
|---|---|---|---|
| `out_copy` | `DeviceBuffer`, zeroed by an enqueued memset | `enqueue_copy` to pinned host memory, then a second drain | 2 |
| `out_host_direct` | pinned `HostBuffer`, zeroed by a host store | none: the host reads it after the kernel's drain | 1 |

`out_host_direct` is reported under its own scope with the input route fixed
at `copy_staged`, so the only variable against the `um.copy_staged` baseline is
where the kernel put its answer.

This direction has a failure mode the input direction does not. The kernel
writes its accumulator with a global integer atomic, and whether such an atomic
is coherent against host-visible memory is unverified on Metal, CUDA, and HIP
alike here. If it is not, the symptom is a wrong histogram rather than a raise.

That matters for the real histogram buffer under one of the two shipped
accumulation strategies: `STRATEGY_ATOMIC` folds every threadgroup's partial
into the output with `Atomic.fetch_add`, while `STRATEGY_TILED` writes partials
without atomics and reduces them with plain stores. Which one runs is decided
per node from that node's row count, so the buffer cannot be given a
once-per-session route that is only safe under one of them. That is why the
policy module refuses this route for `ROLE_HIST_OUT` structurally
(`BLOCK_DEVICE_WRITTEN_ATOMICS`) until the driver answers it, and why the
checksum gate is load-bearing on this route in a way it is not elsewhere.

### Modes: which of mojoboost's two transfer shapes is being modeled

`MOJOBOOST_UM_MODE` picks one. They ask different questions and both are worth
asking; neither is a variation on the other.

- `rewrite` (default) rewrites the whole payload every round. This is the
  per-round gradient upload, and it measures recurring transfer cost.
- `resident` writes the payload once and then launches over it repeatedly.
  This is the binned matrix: uploaded once per session, read by every node of
  every tree. Steady-state rounds have no host write and no publish at all, so
  what they measure is whether a route pays anything *per launch* for memory
  the device already holds. Halfway through the run the host retouches one
  byte, and that round is reported separately as `retouch_*`. That is the
  CPU-writes-after-GPU-reads transition, which is where a runtime that migrates
  pages toward whichever processor last touched them would charge for moving
  them back, and it is invisible in `rewrite` mode where every round pays it.

The staleness protection differs between the modes, and the difference is
stated rather than papered over. In `rewrite` mode byte 0 carries the round
number and the expected checksum is adjusted by the known delta, so a route
that publishes round 0's bytes forever fails from round 1. In `resident` mode
nothing is rewritten by design, so that check cannot run every round: before
the retouch round a `resident` run only proves the route published once, and
the retouch round is what catches a route whose later launches read a stale
copy. **Run `rewrite` first.** A `resident` run alone is not a substitute.

### Holding a second resident buffer

`MOJOBOOST_UM_HOLD_MIB` allocates a device-resident buffer of that size,
writes it once so it is genuinely committed rather than merely requested, and
holds it for the whole run. It models a validation matrix a session keeps
resident while it trains, which is a real configuration (`ROLE_VALID_BINS`) and
one that changes the memory state every other number in the run is measured
under. It matters most in combination with the ladder, where it lowers the
ceiling the ladder reports, which is the honest answer for a machine that is
also holding a validation set.

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
  anything about duplication.

The device-to-host counterpart, a kernel accumulating through a wrapped host
pointer, is `not_probed` for the same reason and promotes the same way.

This three-outcome shape is general. In Mojo a missing method is a compile
error, not a catchable one, so `try`/`except` can only report "this API exists
and refused", never "this API does not exist". The driver therefore
distinguishes `unsupported` (the route raised at runtime) from `not_probed`
(the route was never compiled). Neither is ever reported as `ok`.

### What is measured, per route

This is the complete set of keys emitted per route, in emission order, and it
is the contract the Apple benchmark schema should be written against.

| Metric | Meaning |
|---|---|
| `status` | `ok`, `unsupported`, `wrong`, or `not_probed` |
| `detail` | emitted only when non-empty: the raise message, or why the checksum failed |
| `comparable` | `1` only for `ok`; `0` means the timings must not be compared |
| `input_route` | which payload delivery route this row used |
| `out_route` | `copy` or `host_direct`: which result delivery route this row used |
| `mode` | `rewrite` or `resident` |
| `payload_bytes` | the payload size this route was run at |
| `alloc_ns` | creating every buffer the route needs, plus one synchronize |
| `round0_write_ns` | first host write of the payload, carrying first-touch page faults |
| `round0_publish_ns` | first publish, carrying any first-use migration or residency work |
| `round0_total_ns` | the whole first round |
| `write_ns` | steady-state host write, mean over the counted rounds |
| `publish_ns` | steady-state publish, mean |
| `kernel_ns` | accumulator zeroing plus kernel enqueue, mean |
| `sync_ns` | the synchronize that follows, mean |
| `readback_ns` | obtaining the checksum: a copy plus a second drain on `out_copy`, a load on `out_host_direct` |
| `contend_ns` | host work inside the launch window, when contention mode is on; disjoint from `sync_ns` and part of the round |
| `round_mean_ns`, `round_min_ns` | steady-state round cost |
| `host_alloc_bytes`, `device_alloc_bytes`, `allocated_bytes` | what the route asked for on each side, and their sum |
| `copy_bytes_issued_total` | bytes this driver handed to `enqueue_copy` over the whole run |
| `issues_copy` | `1` when that total is nonzero. Claim 1.5, not Claim 2 |
| `drains_per_round` | queue drains the route owed each round |
| `retouch_round`, `retouch_write_ns`, `retouch_publish_ns`, `retouch_total_ns`, `retouch_over_steady` | `resident` mode only: the CPU-writes-after-GPU-reads round, alone |
| `checksum`, `expected` | the correctness gate |
| `round0_over_steady` | first round divided by the steady-state mean; omitted when there is no steady state |
| `steady_ns_per_byte` | steady-state round cost per payload byte; omitted when there is no steady state |

Three of these are derived rather than measured and are printed only so nobody
recomputes them by hand. `round0_over_steady` and `retouch_over_steady` are
ratios of measured times, not evidence about page migration on their own, and
`steady_ns_per_byte` is what the size ladder's regression test keys on.

A route reported `unsupported` or `not_probed` prints `status` and `detail` and
nothing else, because it produced no measurements. A route reported `wrong`
prints everything, flagged `comparable: 0`, because how a broken route behaved
is worth knowing and comparing it against a working one is not.

Round 0 is always reported separately from the steady state, and so is the
retouch round. A mean that folds either into the recurring cost hides the
single number that round exists to produce. Page migration, if it happens,
happens at transitions.

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
payload size and in the same mode. Read `round_mean_ns` and `kernel_ns +
sync_ns` across the pair. Never read contention out of a single run.

### Synchronization ownership

A timing is comparable only if the route paid for the same guarantees, so the
driver reports `drains_per_round` and never nets it out of a total. `out_copy`
owes two drains a round (wait for the kernel, then wait for the copy back) and
`out_host_direct` owes one. That difference is part of what the route would
buy, not an artifact to correct away, and a comparison that silently absorbed
it would be comparing two different amounts of waiting.

Inside the driver every host write happens after the previous round's drain,
so no route here writes memory the device might still be reading. **A trainer
integration does not get that for free.** On a shared route the kernel itself
reads the host buffer, so the host may not refill it until every kernel that
read it has retired, which is later than "until the copy retired" and is the
whole tree rather than the start of the round. That obligation is written down
in `src/mojoboost/unified_memory_policy.mojo` as `SyncContract`, it is returned
with every route decision so the two cannot be separated, and the consequences
for `StagingRing` are in the handoff.

### The correctness gate

Every route is scored against a CPU reference checksum over the same bytes. A
route whose device checksum disagrees is reported `wrong`, is flagged
`comparable: 0`, and its timings must not be compared with anything. This is
not pedantry. A route that silently skipped the transfer would post the best
numbers in the run and be useless.

The checksum is a wrapping Int32 sum; integer addition wraps deterministically
and is order-independent, so the grid-strided device accumulation and the
sequential host reference agree exactly rather than approximately.

## What the driver cannot measure, and what has to run alongside it

Peak resident memory, memory compression, and swap are not readable from
inside the process by any means this driver is willing to claim. It prints
`um.marker.begin_ns` and `um.marker.end_ns` so an external capture can be
lined up with the phases, and that external capture is **mandatory**, not
optional. The bytes a route reports allocating are the bytes it asked for, not
the bytes the system committed, and the whole duplication question is precisely
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
   default set on a 16 GB machine. Note that a payload is allocated several
   times over in a full run (the source list, plus whatever each route
   allocates), so size the largest run accordingly.
5. **Rounds.** At least 8, so the steady state has samples and round 0 is
   clearly separable. `resident` mode needs at least 4 and wants more, so the
   retouch round has steady-state rounds on both sides of it.
6. **Both modes, `rewrite` first.** `rewrite` answers the gradient question and
   carries the strong staleness check; `resident` answers the binned-matrix
   question. Do not report a `resident` run without the `rewrite` run beside
   it.
7. **Repeat each configuration three times**, as three separate process
   launches, and report the spread. A single run of a memory experiment on a
   general-purpose OS is an anecdote.
8. **Contention.** Run the full set twice, once with `MOJOBOOST_UM_CONTEND=1`.
   Report the two side by side. Do not mix them.
9. **Resident hold, once.** One set with `MOJOBOOST_UM_HOLD_MIB` at a realistic
   validation-matrix size, to see what a resident second matrix does to the
   rest. Report it as its own configuration, never merged into the others.
10. **Ladder last, and deliberately.** `MOJOBOOST_UM_LADDER=1` doubles the
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
| `host_direct` status `ok`, `copy_bytes_issued_total` zero | Claim 1.5: "The device read the correct bytes with no copy issued by mojoboost." | **Not** Claim 2. The runtime may have migrated pages or blitted behind the enqueue. |
| The above, plus peak RSS showing one payload-sized allocation, not two, with no compressor or swap movement | Strong evidence for Claim 2. | Still not proof on its own; pair it with the trace. |
| The above, plus a Metal System Trace with no blit encoder between write and kernel | This is the evidence. Claim 2 is now earned, for this Mojo version, this OS, and this chip. | Any other chip, OS, or Mojo version. Say which one it was measured on. |
| `map_write` `publish_ns` near zero and `sync_ns` unchanged from baseline | The block exit is cheap here. | That no copy happened. Cheap is not free and free is not absent. |
| `out_host_direct` status `wrong` | "A global integer atomic against host-visible memory does not produce the right answer here." This is the most valuable negative result available and it closes the histogram-output question outright. | Anything about the input direction. |
| `out_host_direct` status `ok` and a lower `round_mean_ns` than `copy_staged` | That skipping the readback copy and one of the two drains was cheaper *in this driver*. | That the histogram download is safe to move. The driver's accumulator is four bytes; a real histogram is `3 * n_features * n_bins`, contended by many more threads. |
| `round0_over_steady` near 1.0 | First touch cost nothing measurable at this size. | That there is no page migration; it may simply be small relative to the write. |
| `round0_over_steady` large | Something one-time and expensive happens on first touch. Worth chasing, since mojoboost uploads the binned matrix exactly once per session. | Which of allocation, fault, or migration it was. That needs the trace. |
| `retouch_over_steady` near 1.0 in `resident` mode | A host write to settled memory costs nothing measurable on this route. | That nothing moved; a small payload write can hide a page migration. |
| `retouch_over_steady` large in `resident` mode | The CPU-writes-after-GPU-reads transition is expensive on this route, which matters directly for gradients, which pay it every round. | Which mechanism it was. Trace it. |
| Any route faster than `copy_staged` | That route was faster **in this driver, on this payload, on this machine, at this drain count**. | That mojoboost would get faster. See below. |

That last row is the one most likely to be over-read, so it gets stated
plainly. This driver measures a transfer under a synthetic consumer. mojoboost
training moves a binned matrix once per session, gradients once per round, and
a histogram once per node, and the current end-to-end GPU measurement on an M4
is *slower* than the CPU trainer, with per-node kernel launches and
full-dataset scans dominating. Making every transfer free would not by itself
change that verdict. A route win here is a reason to run
`bench/bench_train_gpu.mojo`, not a substitute for it, which is exactly why
`unified_memory_policy.ENABLE_LEVEL` is the trainer rung and not the driver
rung.

## The policy seam

`src/mojoboost/unified_memory_policy.mojo` is where a result would eventually
be acted on. It is a pure policy layer, in the shape of `apple_gpu_policy.mojo`:
no device, no allocation, no pointer, so all of it is testable on a machine
with no accelerator. It answers one question per buffer role, and it answers
`copy_staged` today for every one of them.

### Roles, and what is structurally possible for each

The route question is asked per role, because the roles differ in direction,
lifetime, and who owns the host memory. `structural_support` encodes what is
possible before any measurement is considered, and four of its answers are
findings in their own right that no benchmark can overturn.

| Role | Buffer | Direction | Shared route structurally possible? |
|---|---|---|---|
| `bins` | `GpuHistogramBuilder.bins_dev` | host to device | **No** for `host_direct`: `BLOCK_SOURCE_NOT_DEVICE_VISIBLE` |
| `grad`, `hess` | `grad_dev` / `hess_dev` via `stage_g` / `stage_h` | host to device | Yes, subject to evidence |
| `hist_out` | `out_dev` | device to host | **No**: `BLOCK_DEVICE_WRITTEN_ATOMICS` until the driver answers |
| `row_seed` | `GpuActiveRows.stage_rows` | host to device | Yes, subject to evidence |
| `valid_bins` | `gpu_predict.mojo`'s `valid_bins_dev` | host to device | **No** for `host_direct`, same as `bins` |
| `predict_out` | `gpu_predict.mojo`'s `host_out` | device to host | Yes, subject to evidence |
| `batch_bins` | `gpu_predict.mojo`'s `bins_dev` via `stage_bins` | host to device | Yes, and it is the only bins-shaped role where that is true |

The `bins` answer is the one that bounds the whole experiment's value, so it is
worth spelling out. `BinnedMatrix.bins` is a plain heap `List[UInt8]` built by
`binning.mojo` and owned by the caller; the builder copies it to the device and
keeps no reference to it. To hand a kernel that host pointer, the bytes would
first have to be copied into a runtime allocation, which is the copy the route
was supposed to remove: the result is two host-side copies instead of one host
copy and one device buffer, and on a unified pool that is not an improvement.
**Removing the duplication for the largest buffer in the system requires
`binning.mojo` to bin directly into a device-visible allocation.** That is a
change to another module's data ownership, not a flag, and it is in the handoff
as a required external edit rather than something this lane can do.

The `batch_bins` answer is the interesting exception. `GpuPredictor.upload_bins`
already stages the batch into a pinned `stage_bins` that the predictor owns, so
the pointer a shared route would need already exists, and a shared route there
would drop both the device buffer and the copy. That staging copy is not an
oversight to be deleted: `bins_dev` is sized to the high-water batch rather than
to this batch, and `enqueue_copy` moves a whole buffer, so copying from the
caller's exactly-sized list would read past its end. Removing it means a
sub-range copy (no such API is verified here), an exact-size reallocation per
batch, or the shared route.

The three duplication shapes are recorded in the policy module so nobody has to
re-derive them: `training_bins_duplication()` and `validation_bins_duplication()`
are two copies each (caller list plus device buffer, no staging), and
`batch_scoring_bins_duplication()` is three. A fit that also scores a held-out
set holds the first two at once, which is what `MOJOBOOST_UM_HOLD_MIB` models.

### The evidence ladder

A route's level is the highest rung with every rung below it satisfied, so
evidence about a route that has not been shown correct counts for nothing.

| Rung | What it means |
|---|---|
| `none` | the state of every route in this repository |
| `compiled` | the route compiles in this Mojo version |
| `checksum` | the device read or wrote the right bytes, round after round |
| `no_copy_issued` | mojoboost enqueued no copy. Claim 1.5 |
| `no_second_allocation` | the external capture shows one payload-sized resident allocation, no compressor or swap movement |
| `no_blit` | a Metal System Trace shows no blit encoder between the host write and the kernel. Claim 2 |
| `trainer` | `bench/bench_train_gpu.mojo` on the route beats the same benchmark on `copy_staged`, repeated, with identical models out |

`ENABLE_LEVEL` is `trainer`. A route below it is refused unless
`MOJOBOOST_GPU_TRANSFER_UNPROVEN=1` is set, and a decision taken that way
carries `ack_unproven`, which any number measured under it must be reported
with. The override exists because the top rung cannot be climbed without
running the trainer on the route, so without it the gate would be
unsatisfiable by construction. It is deliberately loud and deliberately not the
default.

### Environment contract

| Variable | Effect |
|---|---|
| `MOJOBOOST_GPU_TRANSFER` | `staged` (default), `direct`, `mapped`, `host_direct`, `wrapped`. An unparsable value raises rather than falling back, as `device.mojo` does for an impossible `gpu` request |
| `MOJOBOOST_GPU_TRANSFER_UNPROVEN=1` | run a route that has not earned `ENABLE_LEVEL`, and mark every decision it produces |

Nothing reads these yet. They are the contract an integration would implement,
and the handoff names the exact call sites.

## Output grammar

Every line is `um.<scope>.<key>: <value>`, one metric per line, so a run can
be parsed with a line filter and no state. Scopes are `um.<input_route>` for
the four implemented input routes, `um.out_host_direct` for the output route,
`um.wrapped_host_buffer` and `um.out_wrapped_host_buffer` for the unprobed
ones, `um.policy` for the shipped default and the installed evidence,
`um.device` for device attributes, `um.marker` for the external-capture
brackets, `um.ladder` for the size ladder, and bare `um.` for run-level fields.
Status values are exactly `ok`, `unsupported`, `wrong`, `not_probed`. Times are
integer nanoseconds, sizes are integer bytes, ratios are Float64.

The additions relative to the first version of this document are `um.mode`,
`um.hold_bytes`, the whole `um.policy.*` scope, the `um.out_host_direct` and
`um.out_wrapped_host_buffer` scopes, and the per-route keys `input_route`,
`out_route`, `mode`, `copy_bytes_issued_total`, `issues_copy`,
`drains_per_round`, and the five `retouch_*` keys. Every previously emitted key
is still emitted with the same name and meaning, so a parser written against
the old grammar still works and sees fewer keys than exist. Whoever owns the
Apple benchmark schema should extend it for the additions; this document is the
source of truth for the names.

## Invocation

```sh
mojo run -I src bench/apple/unified_memory.mojo [payload_mib] [rounds]
```

Defaults are 256 MiB and 8 rounds.

| Variable | Effect |
|---|---|
| `MOJOBOOST_UM_MODE` | `rewrite` (default) or `resident` |
| `MOJOBOOST_UM_CONTEND=1` | host mutates an unrelated payload-sized buffer inside the launch window |
| `MOJOBOOST_UM_HOLD_MIB` | hold a device-resident buffer of this size for the whole run; default 0 |
| `MOJOBOOST_UM_LADDER=1` | run the size ladder after the fixed-size run |
| `MOJOBOOST_UM_LADDER_MAX_MIB` | ladder ceiling, default 8192 |
| `MOJOBOOST_UM_LADDER_PCT` | per-byte regression cutoff in percent, default 200 |

The driver has no build-lock wrapper of its own. On a machine running parallel
lanes it must be serialized like every other heavy job, and its results are
void if it was not.

## If a route wins

Nothing in `histogram_gpu.mojo` changes on the strength of this driver alone.
The integration seam, the pointer and buffer lifetimes it would need, the
invalidation rules, the measurable hypotheses, and the required external edits
are written up in `handoffs/performance_18_unified_memory.md`. That file is a
proposal contingent on evidence that does not exist yet, and it says so.

## Record

Empty. No run has been performed.

When the first run happens, append to this section: date, chip, core counts,
RAM, free RAM at start, macOS version, Mojo version, MAX version, the mode, the
full `um.*` output, the `/usr/bin/time -l` peak resident set size, the `vm_stat`
deltas, and whether an Instruments trace was taken. Three repeats per
configuration or it does not go in. Give each accepted run an identifier and
put that identifier in `RouteEvidence.record` when a rung is set, so a flag in
the policy module can always be traced back to the run that earned it.
