"""Unified-memory experiment driver: explicit copying versus mapped, shared,
and device-written routes.

This file is the experiment, not a result. Its first execution is
UM-2026-08-15-M4-01 (`bench/results/apple_m4_unified_memory_2026-08-15.md`),
and what that run did and did not establish is in the record section of
`docs/APPLE_UNIFIED_MEMORY.md`. Do not cite this driver as evidence that a
transfer was elided, that a route is free, or that anything got faster; the
record is the citation, and it says which claims it licenses. See the doc
for the methodology, the measurement protocol that must accompany a run,
and what each result would and would not license.

What it asks
------------

Apple silicon has one physical memory pool shared by the CPU and the GPU.
That is a hardware fact and it is not in question. What is in question is
whether the routes MAX/Mojo actually exposes let mojotrees *use* that pool
without paying for a second copy of the data. Those are different claims
(`docs/APPLE_UNIFIED_MEMORY.md`, "Two claims, only one of which is free"),
and only the second one would change how `histogram_gpu.mojo` moves the
binned matrix, the per-round gradients, and the per-node histogram.

So the driver runs the same trivially bandwidth-bound checksum kernel over
the same payload through every delivery route this Mojo version is known to
provide, and reports, per route, where the time went, how many queue drains
the route owed, how many bytes it asked this library to copy, and whether the
device saw the right bytes. The kernel is deliberately dumb: it exists only
so the routes are compared under an identical consumer.

The route vocabulary (`ROUTE_*`, `STATUS_*`, and their names) is imported
from `mojotrees.unified_memory_policy` rather than defined here, so a route
cannot be called one thing in the experiment and another in the policy that
would eventually act on the experiment's answer. That module is also what
prints the shipped default and the evidence a route would have to earn before
anything in the trainer changes.

Input routes: how the payload reaches the kernel
------------------------------------------------

- `copy_staged`   host `List` -> pinned `HostBuffer` -> `enqueue_copy` ->
                  `DeviceBuffer`. This is exactly what
                  `GpuHistogramBuilder.stage_gradients` / `upload_staged`
                  do today, and it is the baseline every other route is
                  measured against.
- `copy_direct`   the same shape with a plain heap `List` in place of the
                  pinned staging buffer. Same-size host write, same copy,
                  so the single variable against `copy_staged` is pinning.
- `map_write`     `DeviceBuffer.map_to_host()`, payload written through the
                  mapped pointer, block exited, kernel reads the device
                  buffer. No second host allocation of our own. On a
                  discrete GPU this is a hidden round trip; on Apple it may
                  be an address remap. The driver cannot tell those apart
                  on its own, which is the entire reason the doc requires
                  an external memory/trace capture alongside it.
- `host_direct`   pinned `HostBuffer` written by the host and its pointer
                  passed *straight into the kernel* as the payload
                  argument, with no device buffer and no copy at all. This
                  is the only implemented input route on which this library
                  issues no copy. It is also the route most likely to
                  silently read garbage, which is what the checksum gate is
                  for.

A fifth input route, wrapping a host allocation in a non-owning
`DeviceBuffer`, is reported as `not_probed` rather than implemented. Its
constructor is documented but its behavior on a host pointer is unverified,
and a route that has not been compiled must not be reported as available. The
candidate code and how to promote it are in `docs/APPLE_UNIFIED_MEMORY.md`,
"Route 5".

The output route: how the result comes back
-------------------------------------------

The four routes above all answer "how do the bytes get to the kernel", and
that is only half of what a boosting round does. `download_raw` copies the
histogram back and drains the queue *once per tree node*, which is the most
frequent transfer in a fit by a wide margin. So the driver also runs the
opposite direction:

- `out_copy`        the kernel accumulates into a `DeviceBuffer` and the
                    result is copied to pinned host memory, which is what
                    `download_raw` does today. Every input route above uses
                    this, so they stay comparable with each other.
- `out_host_direct` the kernel's accumulator argument *is* a pinned
                    `HostBuffer` pointer, so there is no result buffer on the
                    device and no copy back. Reported under the
                    `um.out_host_direct` scope, with its input route fixed to
                    `copy_staged` so the only variable against the
                    `um.copy_staged` baseline is the output route.

This direction has a failure mode the input direction does not: the kernel
writes its accumulator with a global integer atomic, as the atomic-strategy
histogram kernel does with the real histogram buffer, and whether a global
atomic is coherent against host-visible memory is unverified on every backend
here. A wrong answer would be a silently wrong histogram rather than a raise,
so the checksum gate is load-bearing on this route in a way it is not
elsewhere.

Modes: what the payload's lifetime looks like
---------------------------------------------

`MOJOTREES_UM_MODE` selects which of mojotrees's two transfer shapes a run
models. They are not variations on a theme; they ask different questions.

- `rewrite` (default) rewrites the whole payload every round, which is the
  per-round gradient upload. It measures recurring transfer cost.
- `resident` writes the payload once and then launches over it repeatedly,
  which is the binned matrix: uploaded once per session, read by every node
  of every tree. Steady-state rounds in this mode have no host write and no
  publish at all, so what they measure is whether a route pays anything *per
  launch* for memory the device already holds. Halfway through the run the
  host retouches one byte and the round is reported separately as
  `retouch_*`: that is the CPU-writes-after-GPU-reads transition, which is
  where a migrating runtime would charge for moving pages back.

The staleness protection differs between them, and the difference is stated
rather than papered over. In `rewrite` mode byte 0 carries the round number,
so a route that publishes round 0's bytes forever fails from round 1. In
`resident` mode nothing is rewritten by design, so that check cannot run
every round; the retouch round is what catches a route whose later launches
read a stale copy, and until that round a `resident` run only proves the
route published once. Run `rewrite` first.

Correctness gate
----------------

Every route is scored against a CPU reference checksum computed from the same
bytes. A route whose device checksum disagrees is reported `wrong`, never
`ok`, and its timings are printed but flagged uncomparable: a route that
skipped the transfer entirely would look fastest and be useless. The checksum
is a wrapping Int32 sum with per-index weights; integer addition wraps
deterministically and is order-independent, so the grid-strided device
accumulation and the sequential host reference agree exactly rather than
approximately.

Synchronization ownership
-------------------------

A timing is only comparable if the route paid for the same guarantees. Two
routes that differ in how many queue drains they owe are not two measurements
of the same thing, so the driver reports `drains_per_round` per route and
never nets it out of a total. The default output route owes two drains a
round (wait for the kernel, then wait for the copy back); `out_host_direct`
owes one, and that difference is part of what the route would buy rather than
an artifact to be corrected away. Every route's host writes happen after the
previous round's drain, so no route in this driver writes memory the device
might still be reading. A trainer integration does not get that for free, and
`src/mojotrees/unified_memory_policy.mojo`'s `SyncContract` is where that
obligation is written down.

Copies issued, versus copies that happened
------------------------------------------

`copy_bytes_issued_total` counts the bytes *this driver* handed to
`enqueue_copy`. Zero there means mojotrees issued no copy. It does not mean
no copy happened: the runtime remains free to migrate pages, blit behind the
enqueue, or hold a second physical copy, and none of that is visible from
inside the process. That is why the doc's external capture is mandatory and
why the phrase "zero copy" appears nowhere as a conclusion.

Phases reported, per route
--------------------------

`alloc_ns` (buffer creation), `round0_write_ns` and `round0_publish_ns`
(first touch and first publish, which carry page faults and any first-use
migration), `write_ns` / `publish_ns` / `kernel_ns` / `sync_ns` /
`readback_ns` (steady-state means over the rounds after the first, excluding
the retouch round), `round_mean_ns` and `round_min_ns`, plus the bytes the
route allocated on each side. Round 0 is always reported separately from the
steady state; a mean that folds first touch into the recurring cost hides one
of the two numbers this experiment is about.

Peak resident memory, compression, and swap are NOT measured in process. The
driver has no honest way to read them, so it prints a marker line at the start
and end of the run and the doc pins the external capture (`/usr/bin/time -l`,
`vm_stat`, `footprint`) that has to bracket it.

Usage
-----

    mojo run -I src bench/apple/unified_memory.mojo [payload_mib] [rounds]

Environment:

    MOJOTREES_UM_MODE=rewrite|resident
                                  payload lifetime to model, default rewrite
    MOJOTREES_UM_CONTEND=1        host mutates an unrelated buffer while
                                  device work is in flight, to estimate
                                  CPU/GPU contention on the shared pool
    MOJOTREES_UM_HOLD_MIB         hold an extra device-resident buffer of
                                  this size for the whole run, modeling a
                                  validation matrix kept resident during
                                  training; default 0, which allocates none
    MOJOTREES_UM_LADDER=1         after the fixed-size run, double the
                                  payload until a route fails or per-byte
                                  round time regresses, to find the maximum
                                  practical dataset size on this machine
    MOJOTREES_UM_LADDER_MAX_MIB   ladder ceiling, default 8192
    MOJOTREES_UM_LADDER_PCT       regression cutoff in percent of the
                                  smallest size's ns/byte, default 200

The ladder is the memory-pressure part of the experiment and is off by
default on purpose: it is the mode that can push a machine into the
compressor and swap. `MOJOTREES_UM_HOLD_MIB` interacts with it directly, by
design: a ladder run with a held buffer is the closest this driver gets to
the memory state of a fit that is also holding a resident validation matrix.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.os import getenv
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    HostBuffer,
)

from mojotrees.unified_memory_policy import (
    DEFAULT_ROUTE,
    ENABLE_LEVEL,
    N_ROLES,
    N_ROUTES,
    ROUTE_COPY_DIRECT,
    ROUTE_COPY_STAGED,
    ROUTE_HOST_DIRECT,
    ROUTE_MAP_WRITE,
    ROUTE_WRAPPED_HOST,
    STATUS_NOT_PROBED,
    STATUS_OK,
    STATUS_UNSUPPORTED,
    STATUS_WRONG,
    EvidenceLedger,
    block_reason_name,
    evidence_name,
    explain_route,
    role_name,
    route_name,
    status_name,
)


# Threads per block for the checksum kernel. Nothing here is tuned: the
# kernel is a constant across routes, so its geometry cancels out of every
# route-to-route comparison. It is not a proposal for histogram geometry;
# that lives in gpu_tiling.mojo.
comptime BLOCK_THREADS = 256

# Grid cap. The kernel is a grid-stride loop, so any grid is correct; this
# only keeps the launch from scaling with payload size and turning the
# comparison into a launch-overhead comparison.
comptime MAX_BLOCKS = 4096

comptime MIB = 1024 * 1024

# Payload lifetimes the driver can model. See the module docstring: these
# are the two shapes mojotrees actually has, not two settings of one knob.
comptime MODE_REWRITE = 0
comptime MODE_RESIDENT = 1


def _mode_name(mode: Int) -> String:
    if mode == MODE_RESIDENT:
        return String("resident")
    return String("rewrite")


# The byte 0 tag used from the retouch round onward in resident mode. Any
# value that differs from every round tag `rewrite` mode would produce is
# fine; this one is chosen to be obvious in a hex dump.
comptime RETOUCH_TAG = 0xA5


def _checksum_kernel(
    payload: MutPointer[UInt8, MutAnyOrigin],
    out_sum: MutPointer[Int32, MutAnyOrigin],
    n_bytes: Int32,
    stride: Int32,
):
    """Wrapping weighted sum of every payload byte, one global atomic per
    thread.

    Deliberately the simplest kernel that (a) touches every byte, so a
    route that failed to publish is caught, and (b) is identical for every
    route, so the comparison is between memory routes and nothing else. The
    grid is strided rather than sized to the payload, so the launch cost is
    the same at every payload size.

    Both pointer arguments are `MutPointer[..., MutAnyOrigin]`, which is why
    one kernel can serve every route: `payload` may address a device buffer
    or a pinned host buffer, and so may `out_sum`. The global atomic on
    `out_sum` is deliberate and mirrors the real histogram kernels; on the
    `out_host_direct` route it is an atomic against host-visible memory,
    which is exactly the property that route exists to test.
    """
    var n = Int(n_bytes)
    var step = Int(stride)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var acc = Int32(0)
    while i < n:
        acc += Int32(Int(payload[unsafe_offset=i]) * (1 + (i & 15)))
        i += step
    _ = Atomic.fetch_add(out_sum.unsafe_offset(0), acc)


def _cpu_checksum(payload: List[UInt8]) -> Int32:
    """The reference the device result must equal exactly. Same wrapping
    Int32 arithmetic and same weights as the kernel; integer addition is
    associative and commutative under wraparound, so the sequential order
    here and the grid-strided order there give the same value."""
    var acc = Int32(0)
    for i in range(len(payload)):
        acc += Int32(Int(payload[i]) * (1 + (i & 15)))
    return acc


def _expected_after_tag(base: Int32, orig0: UInt8, cur0: UInt8) -> Int32:
    """`base` is the checksum of the untouched payload. Byte 0 carries the
    tag and its weight is 1, so retagging shifts the checksum by exactly the
    byte delta."""
    return base + Int32(Int(cur0)) - Int32(Int(orig0))


def _blocks_for(n_bytes: Int) -> Int:
    var want = (n_bytes + BLOCK_THREADS - 1) // BLOCK_THREADS
    if want < 1:
        return 1
    if want > MAX_BLOCKS:
        return MAX_BLOCKS
    return want


def _env_int(name: String, default: Int) -> Int:
    var s = getenv(name)
    if s.byte_length() == 0:
        return default
    try:
        var v = Int(s)
        if v > 0:
            return v
        return default
    except:
        return default


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _make_payload(n_bytes: Int) -> List[UInt8]:
    """Deterministic pseudorandom bytes, same stream as the other benches
    so a payload of a given size is reproducible across machines and runs.
    Constant data would let a route pass the checksum while publishing the
    wrong region."""
    var out = List[UInt8](capacity=n_bytes)
    for i in range(n_bytes):
        out.append(UInt8(Int(_splitmix64(UInt64(i)) >> 56) & 0xFF))
    return out^


@fieldwise_init
struct RunPlan(Copyable, Movable):
    """What one route is asked to do: how many rounds, which payload
    lifetime, which output route, and whether the host competes for
    bandwidth.

    Carried as one value rather than five parameters so that every runner
    provably runs the same plan, and so that adding a sixth question later
    does not mean editing four signatures.
    """

    var rounds: Int
    var mode: Int
    var out_shared: Bool
    var contend: Bool

    def retouch_round(self) -> Int:
        """The round in which the host rewrites the tag byte, or -1 when the
        mode rewrites every round anyway.

        Placed at the midpoint so there are steady-state rounds on both
        sides of it: the rounds before it measure launches over memory the
        device has settled into, and the rounds after it measure whether
        anything about that changed once the host touched it again.
        """
        if self.mode != MODE_RESIDENT:
            return -1
        return self.rounds // 2

    def writes_payload(self, rnd: Int) -> Bool:
        """Whether this round writes the whole payload on the host."""
        if self.mode == MODE_REWRITE:
            return True
        return rnd == 0

    def is_retouch(self, rnd: Int) -> Bool:
        """Whether this round rewrites only the tag byte."""
        return self.mode == MODE_RESIDENT and rnd == self.retouch_round()

    def tag(self, rnd: Int) -> UInt8:
        """The value byte 0 carries during round `rnd`.

        In `rewrite` mode this is the round number, which is what makes a
        stale publish fail from round 1. In `resident` mode nothing is
        rewritten until the retouch round, so the tag is constant before it
        and constant afterwards at a different value.
        """
        if self.mode == MODE_REWRITE:
            return UInt8(rnd & 0xFF)
        if rnd >= self.retouch_round():
            return UInt8(RETOUCH_TAG)
        return UInt8(0)

    def counted_in_steady_state(self, rnd: Int) -> Bool:
        """Round 0 carries first touch and the retouch round carries a
        transition; neither belongs in a steady-state mean.

        The runners branch on `rnd == 0` and `is_retouch` directly rather
        than calling this, because they have to route those two rounds to
        different accumulators anyway. It is here as the single written
        statement of what the steady state excludes.
        """
        return rnd > 0 and not self.is_retouch(rnd)


@fieldwise_init
struct FirstTouch(Copyable, Movable):
    """One singled-out round: the first, or the retouch. Reported on its own
    because folding either into a mean hides the event it measures."""

    var write_ns: Int
    var publish_ns: Int
    var total_ns: Int

    @staticmethod
    def empty() -> FirstTouch:
        return FirstTouch(0, 0, 0)


@fieldwise_init
struct RouteResult(Copyable, Movable):
    """One route's outcome. Every timing is nanoseconds; every byte count is
    what this route allocated or asked to copy, not what the process resident
    set grew by, which the driver cannot see."""

    var scope: String
    """Report prefix, so the two output routes can share a payload route's
    runner without sharing its key namespace."""

    var route: Int
    var status: Int
    var detail: String
    var plan: RunPlan
    var alloc_ns: Int
    var first: FirstTouch
    var retouch: FirstTouch
    var write_ns: Int
    var publish_ns: Int
    var kernel_ns: Int
    var sync_ns: Int
    var readback_ns: Int
    var contend_ns: Int
    var round_mean_ns: Int
    var round_min_ns: Int
    var host_bytes: Int
    var device_bytes: Int
    var copy_bytes_total: Int
    var drains_per_round: Int
    var checksum: Int
    var expected: Int

    @staticmethod
    def failed(
        scope: String, route: Int, plan: RunPlan, status: Int, detail: String
    ) -> RouteResult:
        """A route that raised or was never probed. Every timing is zero
        because none was taken, which is why `_report` prints nothing but
        the status and the detail for these."""
        return RouteResult(
            scope,
            route,
            status,
            detail,
            plan.copy(),
            0,  # alloc_ns
            FirstTouch.empty(),
            FirstTouch.empty(),
            0,  # write_ns
            0,  # publish_ns
            0,  # kernel_ns
            0,  # sync_ns
            0,  # readback_ns
            0,  # contend_ns
            0,  # round_mean_ns
            0,  # round_min_ns
            0,  # host_bytes
            0,  # device_bytes
            0,  # copy_bytes_total
            0,  # drains_per_round
            0,  # checksum
            0,  # expected
        )


@fieldwise_init
struct Phase(Copyable, Movable):
    """Accumulator for the per-round phase timings of one route.

    The six phases are disjoint and sum to the round, `contend_ns`
    included: the contention work runs on this thread, so it is round time
    like any other. Leaving it out of the total would make a contended run
    look cheaper than an uncontended one, which is backwards."""

    var write_ns: Int
    var publish_ns: Int
    var kernel_ns: Int
    var sync_ns: Int
    var readback_ns: Int
    var contend_ns: Int
    var total_ns: Int
    var min_total_ns: Int
    var rounds: Int

    @staticmethod
    def empty() -> Phase:
        return Phase(0, 0, 0, 0, 0, 0, 0, 0, 0)

    def add(
        mut self,
        write_ns: Int,
        publish_ns: Int,
        kernel_ns: Int,
        sync_ns: Int,
        readback_ns: Int,
        contend_ns: Int,
    ):
        var total = (
            write_ns
            + publish_ns
            + kernel_ns
            + contend_ns
            + sync_ns
            + readback_ns
        )
        self.write_ns += write_ns
        self.publish_ns += publish_ns
        self.kernel_ns += kernel_ns
        self.sync_ns += sync_ns
        self.readback_ns += readback_ns
        self.contend_ns += contend_ns
        self.total_ns += total
        if self.rounds == 0 or total < self.min_total_ns:
            self.min_total_ns = total
        self.rounds += 1

    def mean(self, total: Int) -> Int:
        if self.rounds == 0:
            return 0
        return total // self.rounds


def _host_contention_work(mut scratch: List[UInt8]) -> Int:
    """Host-side memory traffic run while device work is already queued.

    Stands in for the host work a real boosting round does in that window
    (gradient/hessian evaluation over n_rows). It touches one byte per
    cache line so it is bandwidth-shaped rather than ALU-shaped, which is
    the resource a unified pool actually shares. Returns nanoseconds spent,
    which is the only contention number the driver can honestly produce:
    the device-side cost of the same contention shows up as a longer
    `sync_ns` in the same round."""
    var t0 = perf_counter_ns()
    var i = 0
    while i < len(scratch):
        scratch[i] = scratch[i] + 1
        i += 64
    return perf_counter_ns() - t0


@fieldwise_init
struct Tail(Copyable, Movable):
    """What the shared half of a round cost after the host write, and what
    the device computed.

    `contend_ns` and `sync_ns` are disjoint. The contention work runs on
    this thread, so it is serialized *before* the wait rather than overlapped
    with it, and timing the two together would book host time as device wait
    and make every contended round look like the device slowed down. What
    overlaps is the device work, which is why a contended round can show a
    near-zero `sync_ns`: the kernel finished while the host was busy. Both
    are real round time and both are in `device_ns`."""

    var kernel_ns: Int
    var contend_ns: Int
    var sync_ns: Int
    var readback_ns: Int
    var checksum: Int32

    def device_ns(self) -> Int:
        return (
            self.kernel_ns + self.contend_ns + self.sync_ns + self.readback_ns
        )


@fieldwise_init
struct Consumer(Copyable, Movable):
    """The identical device-side half of every route.

    Zero the accumulator, launch the checksum kernel over whatever payload
    pointer the route hands it, optionally run host work in the shadow of
    that launch, synchronize, obtain the checksum. Every route shares this
    object, which is what makes the routes comparable at all: by construction
    the only things that differ between them are how the payload got
    somewhere the kernel could read it and, on `out_shared`, where the kernel
    put its answer.

    `out_shared` is the output route. False is today's shape: the kernel
    accumulates into a device buffer and the result is copied back into
    pinned host memory, costing a second drain. True hands the kernel the
    pinned host buffer's pointer as its accumulator, so there is no device
    accumulator, no copy back, and one drain. The zeroing moves with it: a
    device accumulator is zeroed by an enqueued memset, a host-visible one by
    a host store, and that store is only safe because the previous round
    ended in a drain.
    """

    var out_dev: DeviceBuffer[DType.int32]
    var host_out: HostBuffer[DType.int32]
    var n_bytes: Int32
    var blocks: Int
    var stride: Int32
    var out_shared: Bool

    @staticmethod
    def make(
        mut ctx: DeviceContext, n_bytes: Int, out_shared: Bool
    ) raises -> Consumer:
        """Both buffers are created either way. The unused one is four bytes
        and keeping it unconditional keeps the two output routes' allocation
        phases comparable; the four bytes are reported in the byte counts
        rather than hidden."""
        var blocks = _blocks_for(n_bytes)
        return Consumer(
            ctx.enqueue_create_buffer[DType.int32](1),
            ctx.enqueue_create_host_buffer[DType.int32](1),
            Int32(n_bytes),
            blocks,
            Int32(blocks * BLOCK_THREADS),
            out_shared,
        )

    def drains_per_round(self) -> Int:
        """Queue drains this consumer owes each round.

        Reported rather than netted out. Two routes that owe different
        numbers of drains are not two measurements of the same guarantee,
        and the difference here is the point of the output route rather than
        an artifact of it.
        """
        return 1 if self.out_shared else 2

    def run(
        mut self,
        mut ctx: DeviceContext,
        payload_ptr: MutPointer[UInt8, MutAnyOrigin],
        mut scratch: List[UInt8],
        contend: Bool,
    ) raises -> Tail:
        var t0 = perf_counter_ns()
        if self.out_shared:
            # A host store into memory the device wrote last round. Safe
            # only because every round below ends in a drain; a trainer
            # doing this would owe the same wait, which is what
            # unified_memory_policy.SyncContract records.
            self.host_out.unsafe_ptr().unsafe_store(0, Int32(0))
            ctx.enqueue_function[_checksum_kernel](
                payload_ptr,
                self.host_out.unsafe_ptr(),
                self.n_bytes,
                self.stride,
                grid_dim=self.blocks,
                block_dim=BLOCK_THREADS,
            )
        else:
            ctx.enqueue_memset(self.out_dev, 0)
            ctx.enqueue_function[_checksum_kernel](
                payload_ptr,
                self.out_dev.unsafe_ptr(),
                self.n_bytes,
                self.stride,
                grid_dim=self.blocks,
                block_dim=BLOCK_THREADS,
            )
        var t1 = perf_counter_ns()

        # Host work issued after the launch and before the synchronize, so
        # it runs against queued device work rather than beside an idle
        # device. On one physical pool that is the window where the two
        # compete for bandwidth.
        var contend_ns = 0
        if contend:
            contend_ns = _host_contention_work(scratch)

        # Timed on its own, deliberately. Folding the host work above into
        # this span would book host time as device wait and manufacture a
        # contention effect that is not there.
        var t_wait = perf_counter_ns()
        ctx.synchronize()
        var t2 = perf_counter_ns()

        var checksum = Int32(0)
        if self.out_shared:
            # No copy and no second drain: the accumulator is already in
            # host-visible memory and the drain above retired the kernel
            # that wrote it.
            checksum = self.host_out.unsafe_ptr().unsafe_load(0)
        else:
            ctx.enqueue_copy(
                dst_ptr=self.host_out.unsafe_ptr(), src_buf=self.out_dev
            )
            ctx.synchronize()
            checksum = self.host_out.unsafe_ptr().unsafe_load(0)
        var t3 = perf_counter_ns()

        return Tail(t1 - t0, contend_ns, t2 - t_wait, t3 - t2, checksum)


def _finish(
    scope: String,
    route: Int,
    plan: RunPlan,
    phase: Phase,
    alloc_ns: Int,
    first: FirstTouch,
    retouch: FirstTouch,
    host_bytes: Int,
    device_bytes: Int,
    copy_bytes_total: Int,
    drains_per_round: Int,
    checksum: Int32,
    expected: Int32,
    wrong_detail: String,
) -> RouteResult:
    """Assemble one route's result. A route whose device checksum missed is
    `wrong`, never `ok`: a route that skipped publishing entirely would post
    the best timings in the run and be worthless."""
    var status = STATUS_OK if checksum == expected else STATUS_WRONG
    var detail = String("") if status == STATUS_OK else wrong_detail
    return RouteResult(
        scope,
        route,
        status,
        detail,
        plan.copy(),
        alloc_ns,
        first.copy(),
        retouch.copy(),
        phase.mean(phase.write_ns),
        phase.mean(phase.publish_ns),
        phase.mean(phase.kernel_ns),
        phase.mean(phase.sync_ns),
        phase.mean(phase.readback_ns),
        phase.mean(phase.contend_ns),
        phase.mean(phase.total_ns),
        phase.min_total_ns,
        host_bytes,
        device_bytes,
        copy_bytes_total,
        drains_per_round,
        Int(checksum),
        Int(expected),
    )


comptime WRONG_BYTES = "device checksum disagreed with the host reference"


def _run_copy_route(
    mut ctx: DeviceContext,
    scope: String,
    mut payload: List[UInt8],
    base_sum: Int32,
    orig0: UInt8,
    plan: RunPlan,
    staged: Bool,
    mut scratch: List[UInt8],
) raises -> RouteResult:
    """`copy_staged` (staged=True) and `copy_direct` (staged=False), and, with
    `plan.out_shared`, the `out_host_direct` configuration built on the
    staged input.

    Both copy routes write the payload into a host buffer of the same size
    and then enqueue the same copy from it. The single variable between them
    is whether that buffer is the runtime's pinned allocation or plain heap
    memory, so the difference in their `write_ns` + `publish_ns` + `sync_ns`
    is what pinning is worth and nothing else.

    Skipping the write entirely on the unpinned route would be the obvious
    shortcut and it would be wrong: it would vary two things at once, and it
    would not model the trainer either, since `stage_gradients` has to write
    its Float64-to-Float32 conversion somewhere regardless of which buffer
    receives it. `copy_staged` is what `GpuHistogramBuilder` does today and
    is the baseline for the whole experiment.

    One caveat on their round-0 comparison specifically: `unpinned` is
    faulted in by its own construction inside the allocation window, while
    whether `pinned` is resident before first write is the runtime's
    business. So compare these two routes on the steady state, and read
    their round-0 difference as suspect. Round 0 is still meaningful
    *within* each route and across the mapped routes.
    """
    var route = ROUTE_COPY_STAGED if staged else ROUTE_COPY_DIRECT
    var n = len(payload)

    var t_alloc0 = perf_counter_ns()
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    # The route uses one of these two and the other is a placeholder: a
    # zero-length host buffer is not portable, and a zero-length List would
    # not be an allocation at all.
    var pinned_len = n if staged else 1
    var unpinned_len = 1 if staged else n
    var pinned = ctx.enqueue_create_host_buffer[DType.uint8](pinned_len)
    var unpinned = List[UInt8](capacity=unpinned_len)
    for _ in range(unpinned_len):
        unpinned.append(0)
    var consumer = Consumer.make(ctx, n, plan.out_shared)
    ctx.synchronize()
    var alloc_ns = perf_counter_ns() - t_alloc0

    var phase = Phase.empty()
    var first = FirstTouch.empty()
    var retouch = FirstTouch.empty()
    var checksum = Int32(0)
    var expected = Int32(0)
    var copy_bytes = 0

    for rnd in range(plan.rounds):
        var full = plan.writes_payload(rnd)
        var touch = plan.is_retouch(rnd)
        payload[0] = plan.tag(rnd)
        expected = _expected_after_tag(base_sum, orig0, payload[0])

        var t0 = perf_counter_ns()
        var src = payload.unsafe_ptr()
        # The two buffers are written through their own pointers rather than
        # through one shared local: a pinned host buffer's pointer and a
        # heap list's pointer are different types, and the branch keeps that
        # explicit instead of relying on them coercing.
        if staged:
            var dst = pinned.unsafe_ptr()
            if full:
                for i in range(n):
                    dst.unsafe_store(i, src.unsafe_load(i))
            elif touch:
                # The whole point of the retouch round: one host byte
                # written into memory the device has been reading for
                # several rounds.
                dst.unsafe_store(0, src.unsafe_load(0))
        else:
            var dst = unpinned.unsafe_ptr()
            if full:
                for i in range(n):
                    dst.unsafe_store(i, src.unsafe_load(i))
            elif touch:
                dst.unsafe_store(0, src.unsafe_load(0))
        var t1 = perf_counter_ns()

        # A round that wrote nothing publishes nothing: on this route the
        # device already holds the bytes, which is the resident-matrix
        # shape. A retouch round republishes, because a copy route's device
        # copy is only as fresh as its last copy.
        if full or touch:
            if staged:
                ctx.enqueue_copy(dst_buf=dev, src_ptr=pinned.unsafe_ptr())
            else:
                ctx.enqueue_copy(dst_buf=dev, src_ptr=unpinned.unsafe_ptr())
            copy_bytes += n
        # The copy is enqueued here, not completed here. What guarantees the
        # device has the bytes is the synchronize inside `Consumer.run`, so
        # `publish_ns` on this route is enqueue cost and `sync_ns` carries
        # the transfer. Neither number alone is a transfer time.
        var t2 = perf_counter_ns()

        var tail = consumer.run(ctx, dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), scratch, plan.contend)
        checksum = tail.checksum

        if rnd == 0:
            first = FirstTouch(t1 - t0, t2 - t1, (t2 - t0) + tail.device_ns())
        elif touch:
            retouch = FirstTouch(
                t1 - t0, t2 - t1, (t2 - t0) + tail.device_ns()
            )
        else:
            phase.add(
                t1 - t0,
                t2 - t1,
                tail.kernel_ns,
                tail.sync_ns,
                tail.readback_ns,
                tail.contend_ns,
            )
        if checksum != expected:
            break

    # Host side: the n-byte staging buffer this route used, the one-byte
    # placeholder for the one it did not, and the four-byte accumulator.
    # Device side: the payload buffer and the four-byte accumulator. The
    # placeholder is counted rather than rounded away, because both copy
    # routes allocate one and a byte count that quietly drops an allocation
    # is the kind of number this driver exists to not produce.
    return _finish(
        scope,
        route,
        plan,
        phase,
        alloc_ns,
        first,
        retouch,
        n + 1 + 4,
        n + 4,
        copy_bytes,
        consumer.drains_per_round(),
        checksum,
        expected,
        WRONG_BYTES,
    )


def _run_map_write(
    mut ctx: DeviceContext,
    scope: String,
    mut payload: List[UInt8],
    base_sum: Int32,
    orig0: UInt8,
    plan: RunPlan,
    mut scratch: List[UInt8],
) raises -> RouteResult:
    """`map_write`: no host buffer of ours at all; the payload is written
    through the device buffer's own mapped pointer.

    `write_ns` covers establishing the mapping and the stores through it;
    `publish_ns` covers leaving the block, which is where the runtime does
    whatever it does to make those writes visible to the device. On a
    discrete GPU that exit is a real upload. On Apple it may be nothing at
    all. The split is reported so the two cases are distinguishable, not so
    that either can be asserted from timings alone.

    In `resident` mode the retouch round re-enters the mapping to write one
    byte. That is the most expensive shape of a small update on this route
    and it is deliberately not optimized: the question the round asks is what
    re-establishing a mapping over settled memory costs, not how cheaply one
    byte can be delivered."""
    var n = len(payload)

    var t_alloc0 = perf_counter_ns()
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    var consumer = Consumer.make(ctx, n, plan.out_shared)
    ctx.synchronize()
    var alloc_ns = perf_counter_ns() - t_alloc0

    var phase = Phase.empty()
    var first = FirstTouch.empty()
    var retouch = FirstTouch.empty()
    var checksum = Int32(0)
    var expected = Int32(0)

    for rnd in range(plan.rounds):
        var full = plan.writes_payload(rnd)
        var touch = plan.is_retouch(rnd)
        payload[0] = plan.tag(rnd)
        expected = _expected_after_tag(base_sum, orig0, payload[0])

        var t0 = perf_counter_ns()
        var t1 = t0
        if full or touch:
            with dev.map_to_host() as mapped:
                var dst = mapped.unsafe_ptr()
                var src = payload.unsafe_ptr()
                if full:
                    for i in range(n):
                        dst.unsafe_store(i, src.unsafe_load(i))
                else:
                    dst.unsafe_store(0, src.unsafe_load(0))
                t1 = perf_counter_ns()
        var t2 = perf_counter_ns()

        var tail = consumer.run(ctx, dev.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), scratch, plan.contend)
        checksum = tail.checksum

        if rnd == 0:
            first = FirstTouch(t1 - t0, t2 - t1, (t2 - t0) + tail.device_ns())
        elif touch:
            retouch = FirstTouch(
                t1 - t0, t2 - t1, (t2 - t0) + tail.device_ns()
            )
        else:
            phase.add(
                t1 - t0,
                t2 - t1,
                tail.kernel_ns,
                tail.sync_ns,
                tail.readback_ns,
                tail.contend_ns,
            )
        if checksum != expected:
            break

    return _finish(
        scope,
        ROUTE_MAP_WRITE,
        plan,
        phase,
        alloc_ns,
        first,
        retouch,
        4,
        n + 4,
        0,
        consumer.drains_per_round(),
        checksum,
        expected,
        WRONG_BYTES,
    )


def _run_host_direct(
    mut ctx: DeviceContext,
    scope: String,
    mut payload: List[UInt8],
    base_sum: Int32,
    orig0: UInt8,
    plan: RunPlan,
    mut scratch: List[UInt8],
) raises -> RouteResult:
    """`host_direct`: the kernel's payload argument is a pinned host
    buffer's pointer. No device payload buffer exists and no copy is issued.

    This is the only implemented input route on which this library issues no
    copy, and a passing checksum here is necessary but nowhere near
    sufficient to say the bytes were never duplicated: it shows the device
    read the right bytes, not that it read them in place. The doc's external
    capture (no second payload-sized resident allocation, no blit encoder in
    a Metal System Trace) is what would settle that. If the runtime rejects
    the pointer the route raises and is reported `unsupported`; if it accepts
    the pointer and the device reads something else, the checksum reports
    `wrong`. Both are results and neither is smoothed over."""
    var n = len(payload)

    var t_alloc0 = perf_counter_ns()
    var shared = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var consumer = Consumer.make(ctx, n, plan.out_shared)
    ctx.synchronize()
    var alloc_ns = perf_counter_ns() - t_alloc0

    var phase = Phase.empty()
    var first = FirstTouch.empty()
    var retouch = FirstTouch.empty()
    var checksum = Int32(0)
    var expected = Int32(0)

    for rnd in range(plan.rounds):
        var full = plan.writes_payload(rnd)
        var touch = plan.is_retouch(rnd)
        payload[0] = plan.tag(rnd)
        expected = _expected_after_tag(base_sum, orig0, payload[0])

        # Every write below goes into memory a kernel read last round, and it
        # is safe only because that round ended in a drain: the allocation
        # window for round 0, `Consumer.run`'s synchronize for the rest. This
        # is the whole `RETIRE_ON_KERNEL` obligation in miniature, and a
        # trainer on this route owes it for every node's kernel rather than
        # for one copy. See unified_memory_policy.SyncContract.
        var t0 = perf_counter_ns()
        var dst = shared.unsafe_ptr()
        var src = payload.unsafe_ptr()
        if full:
            for i in range(n):
                dst.unsafe_store(i, src.unsafe_load(i))
        elif touch:
            dst.unsafe_store(0, src.unsafe_load(0))
        var t1 = perf_counter_ns()
        # Nothing is published because nothing is copied. That is the
        # hypothesis under test, not a finding, so `publish_ns` is
        # structurally zero on this route and must never be read as "the
        # transfer was free". Read `sync_ns` and the external capture.
        var t2 = t1

        var tail = consumer.run(
            ctx, shared.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), scratch, plan.contend
        )
        checksum = tail.checksum

        if rnd == 0:
            first = FirstTouch(t1 - t0, t2 - t1, (t2 - t0) + tail.device_ns())
        elif touch:
            retouch = FirstTouch(
                t1 - t0, t2 - t1, (t2 - t0) + tail.device_ns()
            )
        else:
            phase.add(
                t1 - t0,
                t2 - t1,
                tail.kernel_ns,
                tail.sync_ns,
                tail.readback_ns,
                tail.contend_ns,
            )
        if checksum != expected:
            break

    return _finish(
        scope,
        ROUTE_HOST_DIRECT,
        plan,
        phase,
        alloc_ns,
        first,
        retouch,
        n + 4,
        4,
        0,
        consumer.drains_per_round(),
        checksum,
        expected,
        "kernel took a host-buffer pointer but read the wrong bytes",
    )


def _run_route(
    mut ctx: DeviceContext,
    scope: String,
    route: Int,
    mut payload: List[UInt8],
    base_sum: Int32,
    orig0: UInt8,
    plan: RunPlan,
    mut scratch: List[UInt8],
) -> RouteResult:
    """Run one route, converting a raise into `unsupported` rather than
    letting it end the run. A route that refuses is a result.

    `ROUTE_WRAPPED_HOST` is never dispatched here: it is not compiled, and a
    route that was never compiled reports `not_probed` rather than being
    given a runner that would have to lie about what it did."""
    try:
        if route == ROUTE_COPY_STAGED:
            return _run_copy_route(
                ctx, scope, payload, base_sum, orig0, plan, True, scratch
            )
        if route == ROUTE_COPY_DIRECT:
            return _run_copy_route(
                ctx, scope, payload, base_sum, orig0, plan, False, scratch
            )
        if route == ROUTE_MAP_WRITE:
            return _run_map_write(
                ctx, scope, payload, base_sum, orig0, plan, scratch
            )
        if route == ROUTE_HOST_DIRECT:
            return _run_host_direct(
                ctx, scope, payload, base_sum, orig0, plan, scratch
            )
        return RouteResult.failed(
            scope,
            route,
            plan,
            STATUS_NOT_PROBED,
            "no runner is compiled for this route",
        )
    except e:
        return RouteResult.failed(
            scope, route, plan, STATUS_UNSUPPORTED, String(e)
        )


def _report(result: RouteResult, n_bytes: Int):
    var p = result.scope + "."
    print(p + "status:", status_name(result.status))
    if result.detail.byte_length() != 0:
        print(p + "detail:", result.detail)
    var measured = result.status == STATUS_OK or result.status == STATUS_WRONG
    if not measured:
        # `unsupported` and `not_probed` produced no measurements, so they
        # print their status and their reason and nothing that could be
        # mistaken for a timing.
        return
    # A `wrong` route still prints its timings, because knowing how a
    # broken route behaved is useful, but it is flagged uncomparable so no
    # reader and no downstream schema treats those numbers as a route
    # comparison. A route that never published would win on every timing.
    print(p + "comparable:", 1 if result.status == STATUS_OK else 0)
    print(p + "input_route:", route_name(result.route))
    var out_route = String("copy")
    if result.plan.out_shared:
        out_route = String("host_direct")
    print(p + "out_route:", out_route)
    print(p + "mode:", _mode_name(result.plan.mode))
    print(p + "payload_bytes:", n_bytes)
    print(p + "alloc_ns:", result.alloc_ns)
    print(p + "round0_write_ns:", result.first.write_ns)
    print(p + "round0_publish_ns:", result.first.publish_ns)
    print(p + "round0_total_ns:", result.first.total_ns)
    print(p + "write_ns:", result.write_ns)
    print(p + "publish_ns:", result.publish_ns)
    print(p + "kernel_ns:", result.kernel_ns)
    print(p + "sync_ns:", result.sync_ns)
    print(p + "readback_ns:", result.readback_ns)
    print(p + "contend_ns:", result.contend_ns)
    print(p + "round_mean_ns:", result.round_mean_ns)
    print(p + "round_min_ns:", result.round_min_ns)
    print(p + "host_alloc_bytes:", result.host_bytes)
    print(p + "device_alloc_bytes:", result.device_bytes)
    print(p + "allocated_bytes:", result.host_bytes + result.device_bytes)
    # Bytes this driver handed to enqueue_copy. Zero means mojotrees issued
    # no copy; it does not mean no copy happened, and nothing in this file
    # can tell the difference. See the module docstring.
    print(p + "copy_bytes_issued_total:", result.copy_bytes_total)
    # Narrower than it looks, which is why it is not called `issues_copy`.
    # `map_write` reports 0 here because leaving a mapped block is not an
    # `enqueue_copy` call, and that block exit may still be a full upload.
    # The policy module's `publishes_by_copy` is the wider question and
    # answers True for `map_write`; the two are named apart on purpose.
    print(p + "enqueues_copy:", 1 if result.copy_bytes_total > 0 else 0)
    # Queue drains the route owed per round. Routes owing different numbers
    # bought different guarantees; the difference is reported, never netted
    # out of a comparison.
    print(p + "drains_per_round:", result.drains_per_round)
    if result.plan.mode == MODE_RESIDENT:
        # The CPU-writes-after-GPU-reads transition, reported alone. A
        # migrating runtime would charge for moving pages back here and
        # nowhere else in the run.
        print(p + "retouch_round:", result.plan.retouch_round())
        print(p + "retouch_write_ns:", result.retouch.write_ns)
        print(p + "retouch_publish_ns:", result.retouch.publish_ns)
        print(p + "retouch_total_ns:", result.retouch.total_ns)
        if result.round_mean_ns > 0:
            print(
                p + "retouch_over_steady:",
                Float64(result.retouch.total_ns)
                / Float64(result.round_mean_ns),
            )
    print(p + "checksum:", result.checksum)
    print(p + "expected:", result.expected)
    # First touch is what a unified pool is supposed to make cheap, so the
    # ratio is reported rather than left to be recomputed by hand. It is a
    # ratio of measured times, not evidence about page migration on its
    # own; pair it with the doc's external capture.
    if result.round_mean_ns > 0:
        print(
            p + "round0_over_steady:",
            Float64(result.first.total_ns) / Float64(result.round_mean_ns),
        )
    if result.round_mean_ns > 0 and n_bytes > 0:
        print(
            p + "steady_ns_per_byte:",
            Float64(result.round_mean_ns) / Float64(n_bytes),
        )


def _report_policy():
    """What the shipped policy would do with any of this, printed before the
    measurements so a reader sees the gate before the numbers.

    The route a run makes look good is not the route the library uses. That
    decision lives in `src/mojotrees/unified_memory_policy.mojo`, it defaults
    to the staged copy for every buffer role, and it needs evidence at
    `ENABLE_LEVEL` (an end-to-end training result) before it will select
    anything else without an explicit acknowledgment."""
    print("um.policy.default_route:", route_name(DEFAULT_ROUTE))
    print("um.policy.enable_level:", evidence_name(ENABLE_LEVEL))
    var ledger = EvidenceLedger.installed()
    for route in range(N_ROUTES):
        try:
            print(
                String("um.policy.evidence.") + route_name(route) + ":",
                evidence_name(ledger.level_of(route)),
            )
        except:
            print(
                String("um.policy.evidence.") + route_name(route) + ":",
                "unreadable",
            )
    # Per role, what would block the one route that issues no copy, asked
    # with unified memory assumed so the answer isolates data ownership and
    # evidence from the platform. This is the part of the experiment's
    # subject that no run can change, and printing it beside the timings
    # keeps a reader from concluding that a fast route is an available one.
    for role in range(N_ROLES):
        var key = (
            String("um.policy.role.") + role_name(role) + ".host_direct:"
        )
        try:
            var decision = explain_route(
                role, ROUTE_HOST_DIRECT, True, ledger
            )
            print(key, block_reason_name(decision.reason))
        except:
            print(key, "unreadable")


def _report_not_probed(scope: String, detail: String):
    print(scope + ".status:", status_name(STATUS_NOT_PROBED))
    print(scope + ".detail:", detail)


def _ladder(
    mut ctx: DeviceContext,
    start_bytes: Int,
    max_bytes: Int,
    pct: Int,
    rounds: Int,
    mode: Int,
) raises:
    """Find the largest payload this machine still handles at roughly flat
    per-byte cost.

    Doubles the payload until a route raises (out of memory, launch
    refused) or until per-byte steady-state round time exceeds `pct`
    percent of the smallest size's, which is the shape a machine falling
    into the compressor or swap produces. The last size that stayed inside
    the cutoff for every `ok` route is reported as the practical maximum
    *for this machine, this OS, and this free-memory state*, which is not a
    property of the hardware and must not be quoted as one.

    Only the three input routes are laddered, not the output route: doubling
    the payload does not change the four-byte accumulator, so an output route
    would contribute a flat line and cost machine pressure to produce it.

    The per-byte metric means different things in the two modes and the two
    ladders must not be compared with each other. In `rewrite` mode the round
    it divides is a host write plus a publish plus a kernel scan; in
    `resident` mode the steady rounds write and publish nothing, so it is a
    kernel scan alone. Both scale with the payload, which is what makes each
    a usable regression signal on its own, and `um.ladder.mode` says which one
    a given ladder produced.

    This is the memory-pressure mode. Run it on an idle machine with the
    doc's `vm_stat` capture bracketing it, or its answer is about whatever
    else was running. With `MOJOTREES_UM_HOLD_MIB` set, the held buffer is
    still resident throughout, which lowers the ceiling this reports and is
    the point of running the two together.
    """
    print("um.ladder.enabled: 1")
    print("um.ladder.rounds:", rounds)
    print("um.ladder.regression_pct:", pct)
    print("um.ladder.mode:", _mode_name(mode))
    var routes = [
        ROUTE_COPY_STAGED,
        ROUTE_MAP_WRITE,
        ROUTE_HOST_DIRECT,
    ]
    var baseline_per_byte = List[Float64]()
    for _ in range(len(routes)):
        baseline_per_byte.append(0.0)

    var plan = RunPlan(rounds, mode, False, False)
    var n = start_bytes
    var last_good = 0
    while n <= max_bytes:
        var payload = _make_payload(n)
        var base_sum = _cpu_checksum(payload)
        var orig0 = payload[0]
        var scratch = List[UInt8](capacity=1)
        scratch.append(0)
        var all_inside = True
        for k in range(len(routes)):
            var scope = String("um.ladder.") + route_name(routes[k])
            var r = _run_route(
                ctx,
                scope,
                routes[k],
                payload,
                base_sum,
                orig0,
                plan,
                scratch,
            )
            var pfx = scope + "." + String(n) + "."
            print(pfx + "status:", status_name(r.status))
            if r.status != STATUS_OK:
                all_inside = False
                continue
            var per_byte = Float64(r.round_mean_ns) / Float64(n)
            print(pfx + "steady_ns_per_byte:", per_byte)
            print(pfx + "round_mean_ns:", r.round_mean_ns)
            if baseline_per_byte[k] == 0.0:
                baseline_per_byte[k] = per_byte
            elif per_byte * 100.0 > baseline_per_byte[k] * Float64(pct):
                print(pfx + "regressed: 1")
                all_inside = False
        if not all_inside:
            print("um.ladder.stopped_at_bytes:", n)
            break
        last_good = n
        n *= 2
    print("um.ladder.largest_flat_bytes:", last_good)
    print(
        "um.ladder.note:",
        (
            "machine- and free-memory-dependent; not a hardware limit and"
            " not valid without the vm_stat capture from the doc"
        ),
    )


def main() raises:
    var payload_mib = 256
    var rounds = 8
    var args = argv()
    if len(args) > 1:
        payload_mib = Int(String(args[1]))
    if len(args) > 2:
        rounds = Int(String(args[2]))
    if payload_mib < 1:
        raise Error("payload_mib must be at least 1")
    if rounds < 2:
        raise Error("rounds must be at least 2 (round 0 is reported alone)")

    var n_bytes = payload_mib * MIB
    var contend = getenv("MOJOTREES_UM_CONTEND") == "1"
    var ladder = getenv("MOJOTREES_UM_LADDER") == "1"
    var mode = MODE_REWRITE
    var mode_env = getenv("MOJOTREES_UM_MODE")
    if mode_env == "resident":
        mode = MODE_RESIDENT
    elif mode_env.byte_length() != 0 and mode_env != "rewrite":
        raise Error(
            "MOJOTREES_UM_MODE must be 'rewrite' or 'resident', got '",
            mode_env,
            "'",
        )
    if mode == MODE_RESIDENT and rounds < 4:
        # Round 0 writes, the midpoint retouches, and a steady state is
        # wanted on both sides of it.
        raise Error("resident mode needs at least 4 rounds")
    var hold_bytes = _env_int("MOJOTREES_UM_HOLD_MIB", 0) * MIB

    print("um.driver: unified_memory")
    print("um.executed_before: no")
    print("um.payload_mib:", payload_mib)
    print("um.payload_bytes:", n_bytes)
    print("um.rounds:", rounds)
    print("um.mode:", _mode_name(mode))
    print("um.contention:", 1 if contend else 0)
    print("um.hold_bytes:", hold_bytes)
    _report_policy()
    # Bracket markers for the external memory capture. Peak resident
    # memory, compressed pages, and swap are not readable from here, so the
    # doc pins /usr/bin/time -l, vm_stat, and footprint around the whole
    # run and these lines are how a reader lines those up with the phases.
    print("um.marker.begin_ns:", perf_counter_ns())

    comptime if not has_accelerator():
        # Not a failure of the experiment: the build has no accelerator, so
        # every route is unavailable and says so rather than printing zeros
        # that could be mistaken for measurements.
        print("um.accelerator: absent")
        for route in range(ROUTE_COPY_STAGED, ROUTE_WRAPPED_HOST):
            print(
                String("um.") + route_name(route) + ".status:",
                status_name(STATUS_UNSUPPORTED),
            )
            print(
                String("um.") + route_name(route) + ".detail:",
                "no accelerator in this build",
            )
        print("um.out_host_direct.status:", status_name(STATUS_UNSUPPORTED))
        print("um.out_host_direct.detail:", "no accelerator in this build")
        # The unprobed routes stay unprobed here rather than becoming
        # unsupported: the build having no accelerator says nothing about
        # whether their API exists.
        _report_not_probed(
            String("um.") + route_name(ROUTE_WRAPPED_HOST),
            "not compiled in this driver",
        )
        _report_not_probed(
            String("um.out_wrapped_host_buffer"),
            "not compiled in this driver",
        )
        print("um.marker.end_ns:", perf_counter_ns())
    else:
        print("um.accelerator: present")
        var ctx = DeviceContext()
        # Recorded so a result file can be attributed to a device without
        # trusting the filename. Missing attributes report -1 rather than a
        # plausible default.
        try:
            print(
                "um.device.multiprocessors:",
                ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
            )
        except:
            print("um.device.multiprocessors: -1")
        try:
            print(
                "um.device.max_threads_per_block:",
                ctx.get_attribute(DeviceAttribute.MAX_THREADS_PER_BLOCK),
            )
        except:
            print("um.device.max_threads_per_block: -1")

        # A device-resident buffer held for the whole run, modeling a
        # validation matrix the session keeps resident while it trains. It
        # is written once so it is genuinely committed rather than merely
        # requested, and it is never freed before the last route finishes.
        var hold_len = hold_bytes if hold_bytes > 0 else 1
        var hold = ctx.enqueue_create_buffer[DType.uint8](hold_len)
        ctx.enqueue_memset(hold, UInt8(0))
        ctx.synchronize()

        var payload = _make_payload(n_bytes)
        var base_sum = _cpu_checksum(payload)
        var orig0 = payload[0]
        # The contention buffer is a second payload-sized allocation, and
        # allocating it unconditionally would change the memory footprint
        # of the default run, which is one of the things being measured.
        var scratch = List[UInt8](capacity=1)
        if contend:
            scratch = _make_payload(n_bytes)
        else:
            scratch.append(0)

        # The four input routes, all delivering their result the way
        # `download_raw` does today, so they are comparable with each other.
        var plan = RunPlan(rounds, mode, False, contend)
        for route in range(ROUTE_COPY_STAGED, ROUTE_WRAPPED_HOST):
            var r = _run_route(
                ctx,
                String("um.") + route_name(route),
                route,
                payload,
                base_sum,
                orig0,
                plan,
                scratch,
            )
            _report(r, n_bytes)

        # The output route, on the staged input, so the only variable
        # against `um.copy_staged` above is where the kernel put its answer.
        var out_plan = RunPlan(rounds, mode, True, contend)
        var out_result = _run_route(
            ctx,
            String("um.out_host_direct"),
            ROUTE_COPY_STAGED,
            payload,
            base_sum,
            orig0,
            out_plan,
            scratch,
        )
        _report(out_result, n_bytes)

        # Route 5 in the doc. Its constructor is documented but its
        # behavior on a host pointer has not been compiled or run here, and
        # an unprobed route is reported as unprobed.
        _report_not_probed(
            String("um.") + route_name(ROUTE_WRAPPED_HOST),
            (
                "non-owning DeviceBuffer over a host allocation is not"
                " compiled in this driver; see docs/APPLE_UNIFIED_MEMORY.md"
                " Route 5"
            ),
        )
        # The device-to-host counterpart of route 5, and unprobed for the
        # same reason: a kernel writing through a wrapped host pointer would
        # answer whether the atomic coherence question has a second answer,
        # and no code here asks it.
        _report_not_probed(
            String("um.out_wrapped_host_buffer"),
            (
                "kernel accumulating into a non-owning DeviceBuffer over a"
                " host allocation is not compiled in this driver"
            ),
        )

        if ladder:
            _ladder(
                ctx,
                MIB,
                _env_int("MOJOTREES_UM_LADDER_MAX_MIB", 8192) * MIB,
                _env_int("MOJOTREES_UM_LADDER_PCT", 200),
                4 if mode == MODE_RESIDENT else 3,
                mode,
            )
        else:
            print("um.ladder.enabled: 0")

        # A real use of the held buffer, after the last measurement, and it
        # is here for a reason rather than as a formality: Mojo destroys a
        # value at its last use, so without a use down here the held buffer
        # would be freed right after the memset that created it and the run's
        # later routes would see a different memory state than its earlier
        # ones, which is the opposite of what holding it is for.
        ctx.enqueue_memset(hold, UInt8(0))
        ctx.synchronize()
        print("um.marker.end_ns:", perf_counter_ns())
