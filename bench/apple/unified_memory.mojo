"""Unified-memory experiment driver: explicit copying vs mapped/shared routes.

UNEXECUTED. This file is the experiment, not a result. Nothing in this
repository has yet run it, so it has not been compiled, and no number it
would print exists anywhere. Do not cite it as evidence of zero-copy
behavior, of a transfer being elided, or of any speedup. See
`docs/APPLE_UNIFIED_MEMORY.md` for the methodology, the measurement
protocol that must accompany a run, and what each result would and would
not license us to claim.

What it asks
------------

Apple silicon has one physical memory pool shared by the CPU and the GPU.
That is a hardware fact and it is not in question. What is in question is
whether the routes MAX/Mojo actually exposes let mojoboost *use* that pool
without paying for a second copy of the data. Those are different claims
(`docs/APPLE_UNIFIED_MEMORY.md`, "Two claims, only one of which is free"),
and only the second one would change how `histogram_gpu.mojo` moves the
binned matrix and the per-round gradients.

So the driver runs the same trivially bandwidth-bound checksum kernel over
the same payload through every host-to-device route this Mojo version is
known to provide, and reports, per route, where the time went and whether
the device saw the right bytes. The kernel is deliberately dumb: it exists
only so the routes are compared under an identical consumer.

Routes
------

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
                  is the only implemented route that could be zero-copy.
                  It is also the route most likely to silently read
                  garbage, which is what the checksum gate is for.

A fifth route, wrapping a host allocation in a non-owning `DeviceBuffer`,
is reported as `not_probed` rather than implemented. Its constructor is
documented but its behavior on a host pointer is unverified, and a route
that has not been compiled must not be reported as available. The
candidate code and how to promote it are in
`docs/APPLE_UNIFIED_MEMORY.md`, "Route 5".

Correctness gate
----------------

Every route is scored against a CPU reference checksum computed from the
same bytes. A route whose device checksum disagrees is reported `wrong`,
never `ok`, and its timings are printed but must not be compared: a route
that skipped the transfer entirely would look fastest and be useless. To
keep a stale read from passing, byte 0 of the payload is rewritten with the
round number each round and the expected checksum is adjusted by the known
delta, so a route that publishes round 0's bytes forever fails from round 1.
The checksum is a wrapping Int32 sum with per-index weights; integer
addition wraps deterministically and is order-independent, so the
grid-strided device accumulation and the sequential host reference agree
exactly rather than approximately.

Unavailable APIs
----------------

In Mojo a missing method is a compile error, not a catchable one, so
`try`/`except` cannot report "this API does not exist" -- it can only
report "this API exists and refused". Both outcomes are recorded, and they
are recorded differently: `unsupported` means the route raised at runtime,
`not_probed` means the route was never compiled here. Neither is ever
smoothed into `ok`.

Phases reported, per route
--------------------------

`alloc_ns` (buffer creation), `round0_write_ns` and `round0_publish_ns`
(first touch and first publish, which carry page faults and any first-use
migration), `write_ns` / `publish_ns` / `kernel_ns` / `sync_ns` /
`readback_ns` (steady-state means over rounds 1..N-1), `round_mean_ns` and
`round_min_ns`, plus the bytes the route allocated on each side. Round 0 is
always reported separately from the steady state; a mean that folds first
touch into the recurring cost hides the one number this experiment is
about.

Peak resident memory, compression, and swap are NOT measured in process.
The driver has no honest way to read them, so it prints a marker line at
the start and end of the run and the doc pins the external capture
(`/usr/bin/time -l`, `vm_stat`, `footprint`) that has to bracket it.

Usage
-----

    mojo run -I src bench/apple/unified_memory.mojo [payload_mib] [rounds]

Environment:

    MOJOBOOST_UM_CONTEND=1        host mutates an unrelated buffer while
                                  device work is in flight, to estimate
                                  CPU/GPU contention on the shared pool
    MOJOBOOST_UM_LADDER=1         after the fixed-size run, double the
                                  payload until a route fails or per-byte
                                  round time regresses, to find the maximum
                                  practical dataset size on this machine
    MOJOBOOST_UM_LADDER_MAX_MIB   ladder ceiling, default 8192
    MOJOBOOST_UM_LADDER_PCT       regression cutoff in percent of the
                                  smallest size's ns/byte, default 200

The ladder is the memory-pressure part of the experiment and is off by
default on purpose: it is the mode that can push a machine into the
compressor and swap.
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


# Threads per block for the checksum kernel. Nothing here is tuned: the
# kernel is a constant across routes, so its geometry cancels out of every
# route-to-route comparison. It is not a proposal for histogram geometry;
# that lives in gpu_tiling.mojo.
comptime BLOCK_THREADS = 256

# Grid cap. The kernel is a grid-stride loop, so any grid is correct; this
# only keeps the launch from scaling with payload size and turning the
# comparison into a launch-overhead comparison.
comptime MAX_BLOCKS = 4096

comptime STATUS_OK = 0
comptime STATUS_UNSUPPORTED = 1
comptime STATUS_WRONG = 2
comptime STATUS_NOT_PROBED = 3

comptime ROUTE_COPY_STAGED = 0
comptime ROUTE_COPY_DIRECT = 1
comptime ROUTE_MAP_WRITE = 2
comptime ROUTE_HOST_DIRECT = 3

comptime MIB = 1024 * 1024


def _status_name(status: Int) -> String:
    if status == STATUS_OK:
        return "ok"
    if status == STATUS_UNSUPPORTED:
        return "unsupported"
    if status == STATUS_WRONG:
        return "wrong"
    return "not_probed"


def _route_name(route: Int) -> String:
    if route == ROUTE_COPY_STAGED:
        return "copy_staged"
    if route == ROUTE_COPY_DIRECT:
        return "copy_direct"
    if route == ROUTE_MAP_WRITE:
        return "map_write"
    return "host_direct"


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
    round tag and its weight is 1, so retagging shifts the checksum by
    exactly the byte delta."""
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
    return out


@fieldwise_init
struct RouteResult(Copyable, Movable):
    """One route's outcome. Every timing is nanoseconds; every byte count is
    what this route allocated, not what the process resident set grew by,
    which the driver cannot see."""

    var route: Int
    var status: Int
    var detail: String
    var alloc_ns: Int
    var round0_write_ns: Int
    var round0_publish_ns: Int
    var round0_total_ns: Int
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
    var checksum: Int
    var expected: Int

    @staticmethod
    def failed(route: Int, status: Int, detail: String) -> Self:
        """A route that raised or was never probed. Every timing is zero
        because none was taken, which is why `_report` prints nothing but
        the status and the detail for these."""
        return Self(
            route,
            status,
            detail,
            0,  # alloc_ns
            0,  # round0_write_ns
            0,  # round0_publish_ns
            0,  # round0_total_ns
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
            0,  # checksum
            0,  # expected
        )


@fieldwise_init
struct Phase(Copyable, Movable):
    """Accumulator for the per-round phase timings of one route."""

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
    def empty() -> Self:
        return Self(0, 0, 0, 0, 0, 0, 0, 0, 0)

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
            write_ns + publish_ns + kernel_ns + sync_ns + readback_ns
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
    """What the shared half of a round after the host write cost, and what
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
    that launch, synchronize, read the checksum back. Every route shares
    this object, which is what makes the routes comparable at all: by
    construction the only thing that differs between them is how the
    payload got somewhere the kernel could read it."""

    var out_dev: DeviceBuffer[DType.int32]
    var host_out: HostBuffer[DType.int32]
    var n_bytes: Int32
    var blocks: Int
    var stride: Int32

    @staticmethod
    def make(mut ctx: DeviceContext, n_bytes: Int) raises -> Self:
        var blocks = _blocks_for(n_bytes)
        return Self(
            ctx.enqueue_create_buffer[DType.int32](1),
            ctx.enqueue_create_host_buffer[DType.int32](1),
            Int32(n_bytes),
            blocks,
            Int32(blocks * BLOCK_THREADS),
        )

    def run(
        mut self,
        mut ctx: DeviceContext,
        payload_ptr: MutPointer[UInt8, MutAnyOrigin],
        mut scratch: List[UInt8],
        contend: Bool,
    ) raises -> Tail:
        var t0 = perf_counter_ns()
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

        ctx.enqueue_copy(
            dst_ptr=self.host_out.unsafe_ptr(), src_buf=self.out_dev
        )
        ctx.synchronize()
        var checksum = self.host_out.unsafe_ptr().unsafe_load(0)
        var t3 = perf_counter_ns()

        return Tail(t1 - t0, contend_ns, t2 - t_wait, t3 - t2, checksum)


def _finish(
    route: Int,
    phase: Phase,
    alloc_ns: Int,
    round0_write_ns: Int,
    round0_publish_ns: Int,
    round0_total_ns: Int,
    host_bytes: Int,
    device_bytes: Int,
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
        route,
        status,
        detail,
        alloc_ns,
        round0_write_ns,
        round0_publish_ns,
        round0_total_ns,
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
        Int(checksum),
        Int(expected),
    )


comptime WRONG_BYTES = "device checksum disagreed with the host reference"


def _run_copy_route(
    mut ctx: DeviceContext,
    mut payload: List[UInt8],
    base_sum: Int32,
    orig0: UInt8,
    rounds: Int,
    staged: Bool,
    mut scratch: List[UInt8],
    contend: Bool,
) raises -> RouteResult:
    """`copy_staged` (staged=True) and `copy_direct` (staged=False).

    Both write the payload into a host buffer of the same size and then
    enqueue the same copy from it. The single variable between them is
    whether that buffer is the runtime's pinned allocation or plain heap
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
    var consumer = Consumer.make(ctx, n)
    ctx.synchronize()
    var alloc_ns = perf_counter_ns() - t_alloc0

    var phase = Phase.empty()
    var round0_write = 0
    var round0_publish = 0
    var round0_total = 0
    var checksum = Int32(0)
    var expected = Int32(0)

    for rnd in range(rounds):
        payload[0] = UInt8(rnd & 0xFF)
        expected = _expected_after_tag(base_sum, orig0, payload[0])

        var t0 = perf_counter_ns()
        if staged:
            var dst = pinned.unsafe_ptr()
            var src = payload.unsafe_ptr()
            for i in range(n):
                dst.unsafe_store(i, src.unsafe_load(i))
        else:
            var dst = unpinned.unsafe_ptr()
            var src = payload.unsafe_ptr()
            for i in range(n):
                dst.unsafe_store(i, src.unsafe_load(i))
        var t1 = perf_counter_ns()

        if staged:
            ctx.enqueue_copy(dst_buf=dev, src_ptr=pinned.unsafe_ptr())
        else:
            ctx.enqueue_copy(dst_buf=dev, src_ptr=unpinned.unsafe_ptr())
        # The copy is enqueued here, not completed here. What guarantees the
        # device has the bytes is the synchronize inside `Consumer.run`, so
        # `publish_ns` on this route is enqueue cost and `sync_ns` carries
        # the transfer. Neither number alone is a transfer time.
        var t2 = perf_counter_ns()

        var tail = consumer.run(ctx, dev.unsafe_ptr(), scratch, contend)
        checksum = tail.checksum

        if rnd == 0:
            round0_write = t1 - t0
            round0_publish = t2 - t1
            round0_total = (t2 - t0) + tail.device_ns()
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
        route,
        phase,
        alloc_ns,
        round0_write,
        round0_publish,
        round0_total,
        n + 4,
        n + 4,
        checksum,
        expected,
        WRONG_BYTES,
    )


def _run_map_write(
    mut ctx: DeviceContext,
    mut payload: List[UInt8],
    base_sum: Int32,
    orig0: UInt8,
    rounds: Int,
    mut scratch: List[UInt8],
    contend: Bool,
) raises -> RouteResult:
    """`map_write`: no host buffer of ours at all; the payload is written
    through the device buffer's own mapped pointer.

    `write_ns` covers establishing the mapping and the stores through it;
    `publish_ns` covers leaving the block, which is where the runtime does
    whatever it does to make those writes visible to the device. On a
    discrete GPU that exit is a real upload. On Apple it may be nothing at
    all. The split is reported so the two cases are distinguishable, not so
    that either can be asserted from timings alone."""
    var n = len(payload)

    var t_alloc0 = perf_counter_ns()
    var dev = ctx.enqueue_create_buffer[DType.uint8](n)
    var consumer = Consumer.make(ctx, n)
    ctx.synchronize()
    var alloc_ns = perf_counter_ns() - t_alloc0

    var phase = Phase.empty()
    var round0_write = 0
    var round0_publish = 0
    var round0_total = 0
    var checksum = Int32(0)
    var expected = Int32(0)

    for rnd in range(rounds):
        payload[0] = UInt8(rnd & 0xFF)
        expected = _expected_after_tag(base_sum, orig0, payload[0])

        var t0 = perf_counter_ns()
        var t1 = t0
        with dev.map_to_host() as mapped:
            var dst = mapped.unsafe_ptr()
            var src = payload.unsafe_ptr()
            for i in range(n):
                dst.unsafe_store(i, src.unsafe_load(i))
            t1 = perf_counter_ns()
        var t2 = perf_counter_ns()

        var tail = consumer.run(ctx, dev.unsafe_ptr(), scratch, contend)
        checksum = tail.checksum

        if rnd == 0:
            round0_write = t1 - t0
            round0_publish = t2 - t1
            round0_total = (t2 - t0) + tail.device_ns()
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
        ROUTE_MAP_WRITE,
        phase,
        alloc_ns,
        round0_write,
        round0_publish,
        round0_total,
        4,
        n + 4,
        checksum,
        expected,
        WRONG_BYTES,
    )


def _run_host_direct(
    mut ctx: DeviceContext,
    mut payload: List[UInt8],
    base_sum: Int32,
    orig0: UInt8,
    rounds: Int,
    mut scratch: List[UInt8],
    contend: Bool,
) raises -> RouteResult:
    """`host_direct`: the kernel's payload argument is a pinned host
    buffer's pointer. No device payload buffer exists and no copy is issued.

    This is the only implemented route that could be zero-copy, and a
    passing checksum here is necessary but nowhere near sufficient to say
    that it is: it shows the device read the right bytes, not that it read
    them in place. The doc's external capture (no second n-byte resident
    allocation, no blit encoder in a Metal System Trace) is what would
    settle that. If the runtime rejects the pointer the route raises and is
    reported `unsupported`; if it accepts the pointer and the device reads
    something else, the checksum reports `wrong`. Both are results and
    neither is smoothed over."""
    var n = len(payload)

    var t_alloc0 = perf_counter_ns()
    var shared = ctx.enqueue_create_host_buffer[DType.uint8](n)
    var consumer = Consumer.make(ctx, n)
    ctx.synchronize()
    var alloc_ns = perf_counter_ns() - t_alloc0

    var phase = Phase.empty()
    var round0_write = 0
    var round0_publish = 0
    var round0_total = 0
    var checksum = Int32(0)
    var expected = Int32(0)

    for rnd in range(rounds):
        payload[0] = UInt8(rnd & 0xFF)
        expected = _expected_after_tag(base_sum, orig0, payload[0])

        var t0 = perf_counter_ns()
        var dst = shared.unsafe_ptr()
        var src = payload.unsafe_ptr()
        for i in range(n):
            dst.unsafe_store(i, src.unsafe_load(i))
        var t1 = perf_counter_ns()
        # Nothing is published because nothing is copied. That is the
        # hypothesis under test, not a finding, so `publish_ns` is
        # structurally zero on this route and must never be read as "the
        # transfer was free". Read `sync_ns` and the external capture.
        var t2 = t1

        var tail = consumer.run(ctx, shared.unsafe_ptr(), scratch, contend)
        checksum = tail.checksum

        if rnd == 0:
            round0_write = t1 - t0
            round0_publish = t2 - t1
            round0_total = (t2 - t0) + tail.device_ns()
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
        ROUTE_HOST_DIRECT,
        phase,
        alloc_ns,
        round0_write,
        round0_publish,
        round0_total,
        n + 4,
        4,
        checksum,
        expected,
        "kernel took a host-buffer pointer but read the wrong bytes",
    )


def _run_route(
    mut ctx: DeviceContext,
    route: Int,
    mut payload: List[UInt8],
    base_sum: Int32,
    orig0: UInt8,
    rounds: Int,
    mut scratch: List[UInt8],
    contend: Bool,
) -> RouteResult:
    """Run one route, converting a raise into `unsupported` rather than
    letting it end the run. A route that refuses is a result."""
    try:
        if route == ROUTE_COPY_STAGED:
            return _run_copy_route(
                ctx, payload, base_sum, orig0, rounds, True, scratch, contend
            )
        if route == ROUTE_COPY_DIRECT:
            return _run_copy_route(
                ctx, payload, base_sum, orig0, rounds, False, scratch, contend
            )
        if route == ROUTE_MAP_WRITE:
            return _run_map_write(
                ctx, payload, base_sum, orig0, rounds, scratch, contend
            )
        return _run_host_direct(
            ctx, payload, base_sum, orig0, rounds, scratch, contend
        )
    except e:
        return RouteResult.failed(route, STATUS_UNSUPPORTED, String(e))


def _report(result: RouteResult, n_bytes: Int):
    var p = String("um.") + _route_name(result.route) + "."
    print(p + "status:", _status_name(result.status))
    if result.detail.byte_length() != 0:
        print(p + "detail:", result.detail)
    if result.status == STATUS_UNSUPPORTED:
        return
    # A `wrong` route still prints its timings, because knowing how a
    # broken route behaved is useful, but it is flagged uncomparable so no
    # reader and no downstream schema treats those numbers as a route
    # comparison. A route that never published would win on every timing.
    print(p + "comparable:", 1 if result.status == STATUS_OK else 0)
    print(p + "payload_bytes:", n_bytes)
    print(p + "alloc_ns:", result.alloc_ns)
    print(p + "round0_write_ns:", result.round0_write_ns)
    print(p + "round0_publish_ns:", result.round0_publish_ns)
    print(p + "round0_total_ns:", result.round0_total_ns)
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
    print(p + "checksum:", result.checksum)
    print(p + "expected:", result.expected)
    # First touch is what a unified pool is supposed to make cheap, so the
    # ratio is reported rather than left to be recomputed by hand. It is a
    # ratio of measured times, not evidence about page migration on its
    # own; pair it with the doc's external capture.
    if result.round_mean_ns > 0:
        print(
            p + "round0_over_steady:",
            Float64(result.round0_total_ns) / Float64(result.round_mean_ns),
        )
    if result.round_mean_ns > 0 and n_bytes > 0:
        print(
            p + "steady_ns_per_byte:",
            Float64(result.round_mean_ns) / Float64(n_bytes),
        )


def _ladder(
    mut ctx: DeviceContext,
    start_bytes: Int,
    max_bytes: Int,
    pct: Int,
    rounds: Int,
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

    This is the memory-pressure mode. Run it on an idle machine with the
    doc's `vm_stat` capture bracketing it, or its answer is about whatever
    else was running.
    """
    print("um.ladder.enabled: 1")
    print("um.ladder.rounds:", rounds)
    print("um.ladder.regression_pct:", pct)
    var routes = [
        ROUTE_COPY_STAGED,
        ROUTE_MAP_WRITE,
        ROUTE_HOST_DIRECT,
    ]
    var baseline_per_byte = List[Float64]()
    for _ in range(len(routes)):
        baseline_per_byte.append(0.0)

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
            var r = _run_route(
                ctx,
                routes[k],
                payload,
                base_sum,
                orig0,
                rounds,
                scratch,
                False,
            )
            var pfx = (
                String("um.ladder.")
                + _route_name(routes[k])
                + "."
                + String(n)
                + "."
            )
            print(pfx + "status:", _status_name(r.status))
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
    var contend = getenv("MOJOBOOST_UM_CONTEND") == "1"
    var ladder = getenv("MOJOBOOST_UM_LADDER") == "1"

    print("um.driver: unified_memory")
    print("um.executed_before: no")
    print("um.payload_mib:", payload_mib)
    print("um.payload_bytes:", n_bytes)
    print("um.rounds:", rounds)
    print("um.contention:", 1 if contend else 0)
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
        for route in range(4):
            print(
                String("um.") + _route_name(route) + ".status:",
                _status_name(STATUS_UNSUPPORTED),
            )
            print(
                String("um.") + _route_name(route) + ".detail:",
                "no accelerator in this build",
            )
        print(
            "um.wrapped_host_buffer.status:",
            _status_name(STATUS_NOT_PROBED),
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

        for route in range(4):
            var r = _run_route(
                ctx,
                route,
                payload,
                base_sum,
                orig0,
                rounds,
                scratch,
                contend,
            )
            _report(r, n_bytes)

        # Route 5 in the doc. Its constructor is documented but its
        # behavior on a host pointer has not been compiled or run here, and
        # an unprobed route is reported as unprobed.
        print(
            "um.wrapped_host_buffer.status:",
            _status_name(STATUS_NOT_PROBED),
        )
        print(
            "um.wrapped_host_buffer.detail:",
            (
                "non-owning DeviceBuffer over a host allocation is not"
                " compiled in this driver; see docs/APPLE_UNIFIED_MEMORY.md"
                " Route 5"
            ),
        )

        if ladder:
            _ladder(
                ctx,
                MIB,
                _env_int("MOJOBOOST_UM_LADDER_MAX_MIB", 8192) * MIB,
                _env_int("MOJOBOOST_UM_LADDER_PCT", 200),
                3,
            )
        else:
            print("um.ladder.enabled: 0")

        print("um.marker.end_ns:", perf_counter_ns())
