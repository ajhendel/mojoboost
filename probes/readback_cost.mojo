"""What one host readback of a 136-byte split record costs, path by path.

Why this exists
---------------

`docs/METAL_TIMELINE.md` decomposed one blocking readback into 606 microseconds
of wall clock of which 3.7 was the GPU moving bytes. That decomposition says
what a readback costs *in situ*, standing behind a round's worth of queued
compute. It does not say what the readback machinery costs on its own, and the
two questions have different answers and different fixes. `bench_launch_cost`
prices a launch and a launch-plus-wait; nothing prices the *readback*, which is
the operation the device-resident control plane performs once per split.

This harness prices it, one transport at a time, against a floor, and it gates
every transport on delivering the right 136 bytes.

**This file measures. It draws no conclusion about the trainer.** Turning a
per-trip microsecond figure into a per-fit second figure needs a readback
count, and `docs/GPU_PORTABILITY.md` section 6.4 records the five-fold error
this project made the last time it derived one from source.

What this harness established before it could be written
--------------------------------------------------------

Writing the arms required knowing which of them are correct, and answering
that overturned the mechanism the arms were designed against.
`docs/GPU_PORTABILITY.md` section 6.1 says `enqueue_copy` on Metal is a
synchronous full-queue drain in both directions. **That is true of one of its
two implementations and false of the other, and which one you get is chosen by
the destination.** Section 6.5 carries the full statement; the part that
shapes this file:

- `enqueue_copy` device-to-host into a **pinned `HostBuffer`** is
  **asynchronous**. **Measured** here: a slow kernel followed by such a copy,
  read with no intervening `synchronize()`, returned 64 of 64 stale words on
  four attempts out of four, and the right values after the `synchronize()`.
- `enqueue_copy` device-to-host into an **arbitrary host pointer** is
  **synchronous**, exactly as section 6.1 disassembled. **Measured** the same
  way: 0 of 64 stale words with no `synchronize()` at all, on the same kernel.
- `map_to_host()` is synchronous, and is a fresh host allocation with a copy
  in on entry and a copy out on exit, not an address remap.

So a correct readback is either one asynchronous blit plus one drain, or one
synchronous copy and no drain. Both are here. So are the incorrect shapes,
because section 6.1 has been telling this repository they were safe.

The floor, and why it is the first arm
--------------------------------------

Every wait on this backend is a command buffer committed and waited on. The
arms differ in how many of those they cost and in what else rides along.
`bare_sync` measures one of them with nothing queued, so the rest have a
denominator: an arm at 3x the floor and an arm at 1x are comparable in a way
that two microsecond figures are not.

The arms
--------

Correct transports, all checksum-gated against a host reference:

  `bare_sync`         `synchronize()` with nothing queued. The floor. One
                      command buffer, no encoder, no bytes.
  `kernel_sync`       one trivial kernel launch then `synchronize()`. The
                      floor plus an encoder, so the difference is what the
                      submission path costs once the buffer carries work.
  `pinned_pair_sync`  today's shape, exactly what
                      `GpuSplitSearcher.download_words` does: an
                      `enqueue_copy` per plane into pinned host memory, then
                      `synchronize()`. Two asynchronous blits and one drain.
                      The baseline every other transport bids against.
  `pinned_one_sync`   the same with both planes in one device buffer. One
                      asynchronous blit and one drain.
  `plain_pair`        two `enqueue_copy` calls into ordinary heap memory and
                      no `synchronize()` at all. Two synchronous copies, each
                      of which drains inside itself.
  `plain_one`         one `enqueue_copy` into ordinary heap memory. One
                      synchronous copy, one command buffer for the whole
                      readback, no separate drain.
  `map`               `map_to_host()`, which allocates and copies twice.

Refutation arms, kept because section 6.1 said they were safe and they are
not. Each is expected to report `wrong`, and an `ok` here would be the news:

  `pinned_pair_nosync`  today's shape with the trailing `synchronize()`
                        removed -- the edit section 6.1 bullet 3 licenses.
  `pinned_one_nosync`   the packed shape with the same edit.

Arms reported as unavailable, because the unavailability is the finding:

  `host_direct`   a kernel writes a `HostBuffer` and the host reads it with no
                  copy. Executed; fails its checksum.
  `direct`        the host dereferences `DeviceBuffer.unsafe_ptr()`. Not
                  executed: it faults.
  `spin`          the kernel writes the record then a sentinel, and the host
                  polls the sentinel with no command buffer at all.
  `cpu_callback`  `enqueue_cpu_function`, which would let a host function run
                  at a queue position and set a flag the main thread polls.
  `event`         `create_event`, the only thing in the API that could be a
                  non-draining wait.

Why `direct` and `host_direct` cannot work, which is not an ordering bug
-----------------------------------------------------------------------

Recorded because this repository's previous attempt at the same idea was filed
as "host_direct WRONG", and a wrong-value result invites the reading that the
host raced the kernel. It did not, and the fix is not a fence.

**Measured**, by disassembling `libMGPRT.dylib` and by execution:

- Every Metal allocation MAX makes goes through one
  `newBufferWithLength:options:` call site with `options == 0`, which is
  `MTLStorageModeShared`. Every buffer really is host-addressable memory, so
  the idea is sound at the hardware level.
- A `DeviceBuffer`'s Mojo pointer is the buffer's **`gpuAddress`**.
  (`gpuAddress` has one load site, `setBuffer:offset:atIndex:` has none, and
  `useResource:usage:` has one, so kernel arguments are bound bindlessly by
  GPU address.) A GPU virtual address is not a host virtual address: an M4
  returned 0x10000080000 for a fresh device buffer and faulted on the first
  host load, while host allocations in the same process sat near 0x100b4c000.
- A `HostBuffer`'s Mojo pointer is the buffer's **`contents`**, a real host
  address. Handing it to `enqueue_function` hands a CPU address to a kernel
  that reads it as a GPU address. **Measured**: a kernel writing a
  `HostBuffer` left it untouched after a full drain, and a kernel reading a
  `HostBuffer` the host had filled read zeros -- in both directions, and both
  before and after the buffer had been used as an `enqueue_copy` destination,
  which rules out "it was never registered with the runtime".

Both names for the same shared memory exist inside the runtime. The Mojo API
hands out exactly one of them per allocation kind and never both. **The
ordering requirement for a direct read is satisfiable and already paid** --
one command buffer completion is Metal's visibility point for shared storage,
and every arm here pays one. What is missing is an address.

Usage
-----

    mojo run -I src probes/readback_cost.mojo [trips] [trials]
    pixi run probe-readback
    pixi run probe-readback 300 7

Defaults: 200 trips per sample, 5 trials. Arms are interleaved inside each
trial, because this machine's device timings drift several-fold between time
windows and only adjacent samples compare; `bench_train_gpu.mojo` and
`bench_launch_cost.mojo` follow the same rule. Each arm reports its own
minimum and its own spread, and a reader who ignores the spread is reading
noise.

    MOJOTREES_PROBE_DEPTH=0   skip the queue-depth ladder

Correctness is checked in its own untimed pass, not inside the timed loop
-------------------------------------------------------------------------

Each arm runs twice: a verification pass of a few trips with a tag that
changes every trip and a full word-by-word comparison, and then a timed pass
with the same per-trip tag and no comparison. Two passes rather than one
because a comparison inside the timed loop would be timed, and a constant tag
inside the timed loop would let a readback that lags by one trip pass its own
gate -- which is precisely the failure the refutation arms exist to expose,
and precisely the failure this repository has seen before, where tree 0 was
bit-identical and every tree after it diverged.

The queue-depth ladder
----------------------

`docs/GPU_PORTABILITY.md` section 6.2 establishes by disassembly that MAX
creates its Metal queue with a bare `[device newCommandQueue]`, so Apple's
default of 64 command buffers in flight applies by absence, and that there is
one command buffer per launch. From those two it *derives* that a launch
stream longer than 64 runs one-in-one-out with the host blocked inside
`objc_msgSend`, where no instrument in this repository can see it. That
derivation has never been tested, and `gpu_resident_round.mojo` reasons from
it: it puts on the order of 306 command buffers between waits and concludes it
is throttled for most of them.

**There is no knob to test it with. Verified three ways.** The two selectors
that would raise the limit, `newCommandQueueWithMaxCommandBufferCount:` and
`setMaxCommandBufferCount:`, are registered in the runtime's metal-cpp
selector table and have **zero** load sites in the 38.6 MB binary, against one
load site for `newCommandQueue` (inside the Metal device context's
initialization, immediately before its `MODULAR_DISABLE_METAL_GPU_PRINT`
lookup). MAX reads no environment variable resembling a queue depth. And
`DeviceContext.__init__` has exactly four overloads, none taking one; the
compiler lists all four when handed a bad keyword.

So the ladder tests the consequence instead. It enqueues N trivial kernels
with no wait, for N on both sides of 64, and times **only the enqueue loop**.
If the queue is 64 deep and blocks when full, per-launch enqueue cost is flat
while N <= 64 and rises toward the completion rate beyond it, and the knee is
the depth. If it is flat to 512, then either the depth is not 64 or filling it
does not block, and a docstring that has been shaping this campaign's
reasoning is wrong.

One caution, stated before the numbers rather than after: a flat curve is not
proof of no backpressure if the kernel completes faster than the host can
enqueue, because then the queue never fills. `tail_us` is the guard. It is the
wait after the enqueue loop, and it grows with N exactly when work really was
still outstanding when the loop ended.
"""

from std.gpu import global_idx
from std.os import getenv
from std.sys import argv, has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer


# The split record's shape, from gpu_split_search.mojo: 23 integer words
# (`SPLIT_IWORDS`: seven scalars plus a 256-bit category set held 16 bits to a
# word) and 11 float words (`SPLIT_FWORDS`). 34 words, 136 bytes. Spelled as
# literals rather than imported, so this file compiles against the device API
# alone and a change to what the record *contains* cannot silently change what
# is being timed here. Transport cost is a function of the byte count.
comptime REC_IWORDS = 23
comptime REC_FWORDS = 11
comptime REC_WORDS = REC_IWORDS + REC_FWORDS
comptime REC_BYTES = REC_WORDS * 4

comptime VERIFY_TRIPS = 4
"""Trips in the untimed verification pass. Small on purpose: a lagging
readback is wrong on every trip after the first, so four is three more than
the gate needs and cheap enough to run before every timed pass."""


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


comptime VERIFY_SPIN = Int32(200000)
"""Device-side delay the verification pass runs the writers under.

Zero in the timed pass, so nothing measured pays for it. Nonzero in
verification, and it is load-bearing rather than defensive. The pinned
`enqueue_copy` is asynchronous, so whether a readback that skipped its
`synchronize()` returns stale bytes is a race between the blit and the host's
next instruction, and a 34-word kernel finishes long before the host can look.
Under a fast kernel the unsafe arms pass their own gate every time; under a
slow one they fail it every time. **That the fast case usually passes is
exactly what makes the shape dangerous**, and it is why this file drives the
gate with a kernel slow enough for the answer to be deterministic instead of
reporting whichever answer a small record happened to give.
"""


def _write_pair(
    rec_i: MutPointer[Int32, MutAnyOrigin],
    rec_f: MutPointer[Float32, MutAnyOrigin],
    tag: Int32,
    spin: Int32,
):
    """Fill the two-plane record the way the split-search reduce kernel does:
    integers in one buffer, floats in another.

    Values are a function of `tag`, so a readback that lags by a trip is a
    wrong answer rather than a coincidentally right one, and they are small
    whole numbers so the Float32 half compares exactly rather than
    approximately. `spin` is `VERIFY_SPIN` above; it is discarded rather than
    accumulated into the record so the reference stays a pure function of
    `tag`, and the compiler cannot drop it because the store depends on it.
    """
    var i = Int(global_idx.x)
    var acc = Int32(0)
    for k in range(Int(spin)):
        acc = acc * Int32(1103515245) + Int32(k)
    var keep = Int32(1) if acc == Int32(123456789) else Int32(0)
    if i < REC_IWORDS:
        rec_i[unsafe_offset=i] = tag * Int32(100) + Int32(i) + keep
    if i < REC_FWORDS:
        rec_f[unsafe_offset=i] = Float32(
            Int(tag) * 100 + 1000 + i + Int(keep)
        )


def _write_packed(
    rec: MutPointer[Int32, MutAnyOrigin], tag: Int32, spin: Int32
):
    """The same 136 bytes in one buffer.

    The float half is carried as integer words rather than as bit-cast
    Float32. What this arm prices is moving 136 bytes in one command buffer
    instead of two; the byte count and the command buffer count are identical
    either way, and a shipping packed record would bit-cast, which changes the
    decode and not the transfer.
    """
    var i = Int(global_idx.x)
    var acc = Int32(0)
    for k in range(Int(spin)):
        acc = acc * Int32(1103515245) + Int32(k)
    var keep = Int32(1) if acc == Int32(123456789) else Int32(0)
    if i < REC_IWORDS:
        rec[unsafe_offset=i] = tag * Int32(100) + Int32(i) + keep
    elif i < REC_WORDS:
        rec[unsafe_offset=i] = (
            tag * Int32(100)
            + Int32(1000)
            + Int32(i - REC_IWORDS)
            + keep
        )


def _touch(out_buf: MutPointer[Int32, MutAnyOrigin], n: Int32):
    """As little work as a kernel can do and still carry an encoder. Copied in
    spirit from `bench_launch_cost._touch_kernel`, so the launch numbers the
    two files produce are comparable."""
    var i = global_idx.x
    if i < Int(n):
        out_buf[unsafe_offset=i] = Int32(i)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

comptime ARM_OK = 0
comptime ARM_WRONG = 1
comptime ARM_UNAVAILABLE = 2


def _status_name(status: Int) -> String:
    if status == ARM_OK:
        return String("ok")
    if status == ARM_WRONG:
        return String("wrong")
    return String("unavailable")


struct Arm(Copyable, Movable):
    """One transport, its samples, and whether its bytes were right.

    `buffers` is declared by the caller rather than counted, because nothing
    inside this process can observe a command buffer. It is the number this
    arm's own source commits, split into the ones it waits on and the ones it
    does not, and it is printed beside the timing so a reader can check one
    against the other instead of taking either on faith.
    """

    var name: String
    var buffers: Int
    var waits: Int
    var status: Int
    var detail: String
    var expect_wrong: Bool
    var samples: List[Float64]

    def __init__(
        out self,
        name: String,
        buffers: Int,
        waits: Int,
        expect_wrong: Bool = False,
    ):
        self.name = name
        self.buffers = buffers
        self.waits = waits
        self.status = ARM_OK
        self.detail = String("")
        self.expect_wrong = expect_wrong
        self.samples = List[Float64]()

    def unavailable(mut self, why: String):
        self.status = ARM_UNAVAILABLE
        self.detail = why

    def note_mismatch(mut self, bad: Int, trip: Int):
        if bad == 0 or self.status != ARM_OK:
            return
        self.status = ARM_WRONG
        self.detail = (
            String("trip ")
            + String(trip)
            + ": "
            + String(bad)
            + " of "
            + String(REC_WORDS)
            + " words wrong"
        )

    def add(mut self, us_per_trip: Float64):
        self.samples.append(us_per_trip)

    def min_us(self) -> Float64:
        if len(self.samples) == 0:
            return 0.0
        var m = self.samples[0]
        for i in range(1, len(self.samples)):
            if self.samples[i] < m:
                m = self.samples[i]
        return m

    def max_us(self) -> Float64:
        if len(self.samples) == 0:
            return 0.0
        var m = self.samples[0]
        for i in range(1, len(self.samples)):
            if self.samples[i] > m:
                m = self.samples[i]
        return m

    def spread_pct(self) -> Float64:
        var lo = self.min_us()
        if lo <= 0.0:
            return 0.0
        return _round1((self.max_us() - lo) / lo * 100.0)


def _round1(v: Float64) -> Float64:
    var scaled = v * 10.0
    if scaled >= 0.0:
        return Float64(Int(scaled + 0.5)) / 10.0
    return Float64(Int(scaled - 0.5)) / 10.0


def _round2(v: Float64) -> Float64:
    var scaled = v * 100.0
    if scaled >= 0.0:
        return Float64(Int(scaled + 0.5)) / 100.0
    return Float64(Int(scaled - 0.5)) / 100.0


def _report(arm: Arm, trips: Int, floor_us: Float64):
    var scope = String("readback.") + arm.name
    print(scope + ".status:", _status_name(arm.status))
    if arm.detail.byte_length() > 0:
        print(scope + ".detail:", arm.detail)
    if arm.expect_wrong:
        print(
            scope + ".expected:",
            "wrong -- this arm exists to show that the edit"
            " GPU_PORTABILITY 6.1 licenses is unsafe",
        )
        if arm.status == ARM_OK:
            print(
                scope + ".surprise:",
                "this arm delivered the record: the asynchronous pinned copy"
                " may have been changed, and section 6.5 needs rechecking",
            )
    if arm.status == ARM_UNAVAILABLE:
        print(scope + ".command_buffers_per_trip: 0")
        return
    print(scope + ".command_buffers_per_trip:", arm.buffers)
    print(scope + ".waits_per_trip:", arm.waits)
    print(scope + ".trips_per_sample:", trips)
    print(scope + ".samples:", len(arm.samples))
    print(scope + ".per_trip_us:", _round2(arm.min_us()))
    print(scope + ".spread_pct:", arm.spread_pct())
    if floor_us > 0.0:
        print(scope + ".floors:", _round2(arm.min_us() / floor_us))
    if arm.status == ARM_WRONG and not arm.expect_wrong:
        print(
            scope + ".note:",
            "timings printed but uncomparable: this arm did not deliver the"
            " record",
        )


# ---------------------------------------------------------------------------
# Correctness reference
# ---------------------------------------------------------------------------


def _bad_pair(
    host_i: HostBuffer[DType.int32],
    host_f: HostBuffer[DType.float32],
    tag: Int32,
) -> Int:
    var bad = 0
    var si = host_i.unsafe_ptr()
    var sf = host_f.unsafe_ptr()
    for i in range(REC_IWORDS):
        if si.unsafe_load(i) != tag * Int32(100) + Int32(i):
            bad += 1
    for i in range(REC_FWORDS):
        if sf.unsafe_load(i) != Float32(Int(tag) * 100 + 1000 + i):
            bad += 1
    return bad


def _bad_pair_list(
    wi: List[Int32], wf: List[Float32], tag: Int32
) -> Int:
    var bad = 0
    for i in range(REC_IWORDS):
        if wi[i] != tag * Int32(100) + Int32(i):
            bad += 1
    for i in range(REC_FWORDS):
        if wf[i] != Float32(Int(tag) * 100 + 1000 + i):
            bad += 1
    return bad


def _bad_packed_ptr(
    p: Pointer[Int32, MutUntrackedOrigin], tag: Int32
) -> Int:
    var bad = 0
    for i in range(REC_IWORDS):
        if p.unsafe_load(i) != tag * Int32(100) + Int32(i):
            bad += 1
    for i in range(REC_FWORDS):
        if p.unsafe_load(REC_IWORDS + i) != (
            tag * Int32(100) + Int32(1000) + Int32(i)
        ):
            bad += 1
    return bad


comptime CLOBBER = Int32(-424242)
"""What a destination is filled with before an arm that might not write it.

Chosen so that "the copy never landed" reads differently from "the copy
landed with the wrong value", which matters because the two have different
fixes and because zero is a value a kernel could legitimately produce."""


def _clobber_pair(
    mut host_i: HostBuffer[DType.int32], mut host_f: HostBuffer[DType.float32]
):
    var pi = host_i.unsafe_ptr()
    var pf = host_f.unsafe_ptr()
    for i in range(REC_IWORDS):
        pi.unsafe_store(i, CLOBBER)
    for i in range(REC_FWORDS):
        pf.unsafe_store(i, Float32(-424242.0))


def _clobber_packed(mut host_p: HostBuffer[DType.int32]):
    var p = host_p.unsafe_ptr()
    for i in range(REC_WORDS):
        p.unsafe_store(i, CLOBBER)


def _bad_packed_list(w: List[Int32], tag: Int32) -> Int:
    var bad = 0
    for i in range(REC_IWORDS):
        if w[i] != tag * Int32(100) + Int32(i):
            bad += 1
    for i in range(REC_FWORDS):
        if w[REC_IWORDS + i] != tag * Int32(100) + Int32(1000) + Int32(i):
            bad += 1
    return bad


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() raises:
    comptime if not has_accelerator():
        print("no accelerator present; readback cost probe skipped")
    else:
        var trips = 200
        var trials = 5
        var args = argv()
        if len(args) > 1:
            trips = Int(String(args[1]))
        if len(args) > 2:
            trials = Int(String(args[2]))
        if trips < 1 or trials < 1:
            raise Error("trips and trials must both be at least 1")

        var ctx = DeviceContext()
        print("mojotrees readback cost probe")
        print("readback.device:", ctx.name())
        print("readback.api:", ctx.api())
        print("readback.record_bytes:", REC_BYTES)
        print("readback.trips:", trips, "trials:", trials)

        var rec_i = ctx.enqueue_create_buffer[DType.int32](REC_IWORDS)
        var rec_f = ctx.enqueue_create_buffer[DType.float32](REC_FWORDS)
        var rec_p = ctx.enqueue_create_buffer[DType.int32](REC_WORDS)
        var host_i = ctx.enqueue_create_host_buffer[DType.int32](REC_IWORDS)
        var host_f = ctx.enqueue_create_host_buffer[DType.float32](REC_FWORDS)
        var host_p = ctx.enqueue_create_host_buffer[DType.int32](REC_WORDS)
        var scratch = ctx.enqueue_create_buffer[DType.int32](1024)
        ctx.synchronize()

        # Ordinary heap destinations, for the synchronous copy path.
        var plain_i = List[Int32]()
        var plain_f = List[Float32]()
        var plain_p = List[Int32]()
        for _ in range(REC_IWORDS):
            plain_i.append(Int32(0))
        for _ in range(REC_FWORDS):
            plain_f.append(Float32(0.0))
        for _ in range(REC_WORDS):
            plain_p.append(Int32(0))

        var bare = Arm(String("bare_sync"), 1, 1)
        var kern = Arm(String("kernel_sync"), 2, 1)
        var pin_pair = Arm(String("pinned_pair_sync"), 4, 1)
        var pin_one = Arm(String("pinned_one_sync"), 3, 1)
        var pin_pair_ns = Arm(
            String("pinned_pair_nosync"), 3, 0, expect_wrong=True
        )
        var pin_one_ns = Arm(
            String("pinned_one_nosync"), 2, 0, expect_wrong=True
        )
        var plain_pair = Arm(String("plain_pair"), 3, 2)
        var plain_one = Arm(String("plain_one"), 2, 1)
        var mapped = Arm(String("map"), 3, 2)

        _warm(
            ctx, rec_i, rec_f, rec_p, host_i, host_f, host_p, scratch
        )

        # -- verification pass, untimed ------------------------------------
        #
        # Every arm gets its **own** tag within a trip, and the seven of them
        # never repeat. That is not tidiness. An earlier version gave the
        # whole trip one tag, and the two unsafe arms passed their own gate
        # every time: an arm that ran second was reading a destination an
        # earlier arm had already filled correctly with the same tag, so
        # "stale" and "correct" were the same bytes. A gate that can be
        # satisfied by the previous arm's success is not a gate. This is the
        # same shape of mistake as the host-side table that made tree 0
        # bit-identical and every tree after it diverge.
        #
        # The destinations are also clobbered with a sentinel before each
        # unsafe arm, so "the copy had not landed" is distinguishable from
        # "the copy landed with the wrong value".
        for v in range(VERIFY_TRIPS):
            var t0 = Int32(v * 10 + 1)

            ctx.enqueue_function[_write_pair](
                rec_i.unsafe_ptr(), rec_f.unsafe_ptr(), t0, VERIFY_SPIN,
                grid_dim=1, block_dim=64,
            )
            ctx.enqueue_copy(dst_ptr=host_i.unsafe_ptr(), src_buf=rec_i)
            ctx.enqueue_copy(dst_ptr=host_f.unsafe_ptr(), src_buf=rec_f)
            ctx.synchronize()
            pin_pair.note_mismatch(_bad_pair(host_i, host_f, t0), v)

            var t1 = Int32(v * 10 + 2)
            ctx.enqueue_function[_write_packed](
                rec_p.unsafe_ptr(), t1, VERIFY_SPIN, grid_dim=1, block_dim=64
            )
            ctx.enqueue_copy(dst_ptr=host_p.unsafe_ptr(), src_buf=rec_p)
            ctx.synchronize()
            pin_one.note_mismatch(_bad_packed_ptr(host_p.unsafe_ptr(), t1), v)

            # The two arms section 6.1 bullet 3 licenses.
            var t2 = Int32(v * 10 + 3)
            _clobber_pair(host_i, host_f)
            ctx.enqueue_function[_write_pair](
                rec_i.unsafe_ptr(), rec_f.unsafe_ptr(), t2, VERIFY_SPIN,
                grid_dim=1, block_dim=64,
            )
            ctx.enqueue_copy(dst_ptr=host_i.unsafe_ptr(), src_buf=rec_i)
            ctx.enqueue_copy(dst_ptr=host_f.unsafe_ptr(), src_buf=rec_f)
            pin_pair_ns.note_mismatch(_bad_pair(host_i, host_f, t2), v)
            ctx.synchronize()

            var t3 = Int32(v * 10 + 4)
            _clobber_packed(host_p)
            ctx.enqueue_function[_write_packed](
                rec_p.unsafe_ptr(), t3, VERIFY_SPIN, grid_dim=1, block_dim=64
            )
            ctx.enqueue_copy(dst_ptr=host_p.unsafe_ptr(), src_buf=rec_p)
            pin_one_ns.note_mismatch(_bad_packed_ptr(host_p.unsafe_ptr(), t3), v)
            ctx.synchronize()

            # The synchronous destination kind, with no drain of its own.
            var t4 = Int32(v * 10 + 5)
            for i in range(REC_IWORDS):
                plain_i[i] = Int32(-424242)
            for i in range(REC_FWORDS):
                plain_f[i] = Float32(-424242.0)
            ctx.enqueue_function[_write_pair](
                rec_i.unsafe_ptr(), rec_f.unsafe_ptr(), t4, VERIFY_SPIN,
                grid_dim=1, block_dim=64,
            )
            ctx.enqueue_copy(dst_ptr=plain_i.unsafe_ptr(), src_buf=rec_i)
            ctx.enqueue_copy(dst_ptr=plain_f.unsafe_ptr(), src_buf=rec_f)
            plain_pair.note_mismatch(_bad_pair_list(plain_i, plain_f, t4), v)

            var t5 = Int32(v * 10 + 6)
            for i in range(REC_WORDS):
                plain_p[i] = Int32(-424242)
            ctx.enqueue_function[_write_packed](
                rec_p.unsafe_ptr(), t5, VERIFY_SPIN, grid_dim=1, block_dim=64
            )
            ctx.enqueue_copy(dst_ptr=plain_p.unsafe_ptr(), src_buf=rec_p)
            plain_one.note_mismatch(_bad_packed_list(plain_p, t5), v)

            var t6 = Int32(v * 10 + 7)
            for i in range(REC_WORDS):
                plain_p[i] = Int32(-424242)
            ctx.enqueue_function[_write_packed](
                rec_p.unsafe_ptr(), t6, VERIFY_SPIN, grid_dim=1, block_dim=64
            )
            with rec_p.map_to_host() as m:
                var mp = m.unsafe_ptr()
                for i in range(REC_WORDS):
                    plain_p[i] = mp.unsafe_load(i)
            mapped.note_mismatch(_bad_packed_list(plain_p, t6), v)
        ctx.synchronize()

        # -- timed pass, arms interleaved inside each trial ----------------
        for t in range(trials):
            var base = Int32(t * 1000 + 100)

            ctx.synchronize()
            var a0 = perf_counter_ns()
            for _ in range(trips):
                ctx.synchronize()
            var a1 = perf_counter_ns()
            bare.add(Float64(a1 - a0) / 1e3 / Float64(trips))

            var b0 = perf_counter_ns()
            for _ in range(trips):
                ctx.enqueue_function[_touch](
                    scratch.unsafe_ptr(), Int32(1024),
                    grid_dim=4, block_dim=256,
                )
                ctx.synchronize()
            var b1 = perf_counter_ns()
            kern.add(Float64(b1 - b0) / 1e3 / Float64(trips))

            var c0 = perf_counter_ns()
            for k in range(trips):
                ctx.enqueue_function[_write_pair](
                    rec_i.unsafe_ptr(), rec_f.unsafe_ptr(), base + Int32(k),
                    Int32(0), grid_dim=1, block_dim=64,
                )
                ctx.enqueue_copy(dst_ptr=host_i.unsafe_ptr(), src_buf=rec_i)
                ctx.enqueue_copy(dst_ptr=host_f.unsafe_ptr(), src_buf=rec_f)
                ctx.synchronize()
            var c1 = perf_counter_ns()
            pin_pair.add(Float64(c1 - c0) / 1e3 / Float64(trips))

            var d0 = perf_counter_ns()
            for k in range(trips):
                ctx.enqueue_function[_write_packed](
                    rec_p.unsafe_ptr(), base + Int32(k), Int32(0),
                    grid_dim=1, block_dim=64,
                )
                ctx.enqueue_copy(dst_ptr=host_p.unsafe_ptr(), src_buf=rec_p)
                ctx.synchronize()
            var d1 = perf_counter_ns()
            pin_one.add(Float64(d1 - d0) / 1e3 / Float64(trips))

            var e0 = perf_counter_ns()
            for k in range(trips):
                ctx.enqueue_function[_write_pair](
                    rec_i.unsafe_ptr(), rec_f.unsafe_ptr(), base + Int32(k),
                    Int32(0), grid_dim=1, block_dim=64,
                )
                ctx.enqueue_copy(dst_ptr=host_i.unsafe_ptr(), src_buf=rec_i)
                ctx.enqueue_copy(dst_ptr=host_f.unsafe_ptr(), src_buf=rec_f)
            var e1 = perf_counter_ns()
            pin_pair_ns.add(Float64(e1 - e0) / 1e3 / Float64(trips))
            ctx.synchronize()

            var f0 = perf_counter_ns()
            for k in range(trips):
                ctx.enqueue_function[_write_packed](
                    rec_p.unsafe_ptr(), base + Int32(k), Int32(0),
                    grid_dim=1, block_dim=64,
                )
                ctx.enqueue_copy(dst_ptr=host_p.unsafe_ptr(), src_buf=rec_p)
            var f1 = perf_counter_ns()
            pin_one_ns.add(Float64(f1 - f0) / 1e3 / Float64(trips))
            ctx.synchronize()

            var g0 = perf_counter_ns()
            for k in range(trips):
                ctx.enqueue_function[_write_pair](
                    rec_i.unsafe_ptr(), rec_f.unsafe_ptr(), base + Int32(k),
                    Int32(0), grid_dim=1, block_dim=64,
                )
                ctx.enqueue_copy(dst_ptr=plain_i.unsafe_ptr(), src_buf=rec_i)
                ctx.enqueue_copy(dst_ptr=plain_f.unsafe_ptr(), src_buf=rec_f)
            var g1 = perf_counter_ns()
            plain_pair.add(Float64(g1 - g0) / 1e3 / Float64(trips))

            var h0 = perf_counter_ns()
            for k in range(trips):
                ctx.enqueue_function[_write_packed](
                    rec_p.unsafe_ptr(), base + Int32(k), Int32(0),
                    grid_dim=1, block_dim=64,
                )
                ctx.enqueue_copy(dst_ptr=plain_p.unsafe_ptr(), src_buf=rec_p)
            var h1 = perf_counter_ns()
            plain_one.add(Float64(h1 - h0) / 1e3 / Float64(trips))

            var i0 = perf_counter_ns()
            for k in range(trips):
                ctx.enqueue_function[_write_packed](
                    rec_p.unsafe_ptr(), base + Int32(k), Int32(0),
                    grid_dim=1, block_dim=64,
                )
                with rec_p.map_to_host() as m:
                    var mp = m.unsafe_ptr()
                    for i in range(REC_WORDS):
                        plain_p[i] = mp.unsafe_load(i)
            var i1 = perf_counter_ns()
            mapped.add(Float64(i1 - i0) / 1e3 / Float64(trips))

        var floor = bare.min_us()
        _report(bare, trips, 0.0)
        _report(kern, trips, floor)
        _report(pin_pair, trips, floor)
        _report(pin_one, trips, floor)
        _report(plain_pair, trips, floor)
        _report(plain_one, trips, floor)
        _report(mapped, trips, floor)
        _report(pin_pair_ns, trips, floor)
        _report(pin_one_ns, trips, floor)

        _report_host_direct(ctx)
        _report_direct()
        _report_spin()
        _report_cpu_callback(ctx)
        _report_event(ctx)

        if getenv("MOJOTREES_PROBE_DEPTH") != "0":
            _ladder(ctx, scratch, trials)
        else:
            print("depth.status: skipped by MOJOTREES_PROBE_DEPTH=0")


def _warm(
    mut ctx: DeviceContext,
    mut rec_i: DeviceBuffer[DType.int32],
    mut rec_f: DeviceBuffer[DType.float32],
    mut rec_p: DeviceBuffer[DType.int32],
    mut host_i: HostBuffer[DType.int32],
    mut host_f: HostBuffer[DType.float32],
    mut host_p: HostBuffer[DType.int32],
    mut scratch: DeviceBuffer[DType.int32],
) raises:
    """Launch every kernel and run every transport shape once before anything
    is timed.

    Both shapes, not just the cheap one: `bench_launch_cost` records an
    unwarmed wait arm reading 235 microseconds against a settled 148, so
    warming only the launch would leave that difference in the first trial of
    whichever arm ran first.
    """
    for _ in range(20):
        ctx.enqueue_function[_write_pair](
            rec_i.unsafe_ptr(), rec_f.unsafe_ptr(), Int32(1), Int32(0),
            grid_dim=1, block_dim=64,
        )
        ctx.enqueue_function[_write_pair](
            rec_i.unsafe_ptr(), rec_f.unsafe_ptr(), Int32(1), VERIFY_SPIN,
            grid_dim=1, block_dim=64,
        )
        ctx.enqueue_function[_write_packed](
            rec_p.unsafe_ptr(), Int32(1), Int32(0), grid_dim=1, block_dim=64
        )
        ctx.enqueue_function[_write_packed](
            rec_p.unsafe_ptr(), Int32(1), VERIFY_SPIN, grid_dim=1, block_dim=64
        )
        ctx.enqueue_function[_touch](
            scratch.unsafe_ptr(), Int32(1024), grid_dim=4, block_dim=256
        )
        ctx.synchronize()
        ctx.enqueue_copy(dst_ptr=host_i.unsafe_ptr(), src_buf=rec_i)
        ctx.enqueue_copy(dst_ptr=host_f.unsafe_ptr(), src_buf=rec_f)
        ctx.enqueue_copy(dst_ptr=host_p.unsafe_ptr(), src_buf=rec_p)
        ctx.synchronize()
        with rec_p.map_to_host() as _m:
            pass


# ---------------------------------------------------------------------------
# The unavailable arms
# ---------------------------------------------------------------------------


def _report_host_direct(mut ctx: DeviceContext) raises:
    """A kernel writes a `HostBuffer` and the host reads it with no copy.

    Executed rather than declared, because executing it is safe: the kernel
    addresses something that is not this buffer, so the host reads back what
    it left there and nothing faults. The checksum is the finding. The buffer
    is pre-filled with a sentinel, so "the kernel did not write here" is
    distinguishable from "the kernel wrote zeros".
    """
    var arm = Arm(String("host_direct"), 0, 0)
    var hb = ctx.enqueue_create_host_buffer[DType.int32](REC_WORDS)
    ctx.synchronize()
    var p = hb.unsafe_ptr()
    for i in range(REC_WORDS):
        p.unsafe_store(i, Int32(-424242))
    ctx.enqueue_function[_write_packed](
        p, Int32(3), Int32(0), grid_dim=1, block_dim=64
    )
    ctx.synchronize()
    var bad = _bad_packed_ptr(p, Int32(3))
    var untouched = 0
    for i in range(REC_WORDS):
        if p.unsafe_load(i) == Int32(-424242):
            untouched += 1
    if bad == 0:
        # If this ever passes, MAX has started binding host allocations and
        # the shape of this whole lane changes. Say so loudly rather than
        # quietly reporting a fast arm.
        arm.detail = String(
            "the kernel wrote the host buffer: MAX now binds host"
            " allocations by an address the host also holds, and the direct"
            " readback path is open"
        )
        _report(arm, 0, 0.0)
        return
    arm.unavailable(
        String("the kernel did not write this memory (")
        + String(bad)
        + " of "
        + String(REC_WORDS)
        + " words wrong, "
        + String(untouched)
        + " still holding the pre-fill sentinel). A HostBuffer's Mojo pointer"
        " is its MTLBuffer `contents`, a CPU address, and enqueue_function"
        " binds kernel arguments by `gpuAddress`. Not an ordering race: this"
        " ran after a full drain."
    )
    _report(arm, 0, 0.0)


def _report_direct():
    """Reading a `DeviceBuffer`'s own pointer on the host.

    Not executed. `DeviceBuffer.unsafe_ptr()` is the buffer's `gpuAddress`,
    and dereferencing it on the host raises SIGSEGV, which would end the
    process and take every other arm's numbers with it. It was executed once,
    in isolation, outside this harness, which is where the addresses quoted in
    the module docstring come from.
    """
    var arm = Arm(String("direct"), 0, 0)
    arm.unavailable(
        String(
            "DeviceBuffer.unsafe_ptr() is a GPU virtual address; dereferencing"
            " it on the host faults. Not executed here, because the fault"
            " would take the other arms' numbers with it."
        )
    )
    _report(arm, 0, 0.0)


def _report_spin():
    var arm = Arm(String("spin"), 0, 0)
    arm.unavailable(
        String(
            "needs one allocation that is both kernel-writable and"
            " host-readable, which MAX exposes on neither DeviceBuffer nor"
            " HostBuffer on Metal. The sentinel could be ordered behind the"
            " record with a device memory fence; there is nowhere to put it."
        )
    )
    _report(arm, 0, 0.0)


def _report_cpu_callback(mut ctx: DeviceContext):
    """`enqueue_cpu_function` would be a fence: a host function that runs at a
    queue position and sets a flag the main thread polls, which is the usual
    way a completion handler becomes a cheap wait."""
    var arm = Arm(String("cpu_callback"), 0, 0)
    var ran = False

    def _cb() {imm}:
        pass

    try:
        ctx.enqueue_cpu_function(_cb)
        ctx.synchronize()
        ran = True
    except e:
        arm.unavailable(String(e))
    if ran:
        arm.detail = String(
            "a host function ran at a queue position: a non-draining wait can"
            " be built on it"
        )
    _report(arm, 0, 0.0)


def _report_event(mut ctx: DeviceContext):
    """`create_event` is the only thing in the MAX API that could be a
    non-draining wait on this backend.

    Independently, **measured** by disassembly: `newSharedEvent`, `newEvent`,
    `newFence`, `encodeSignalEvent:value:`, `encodeWaitForEvent:value:`,
    `notifyListener:atValue:block:` and `addCompletedHandler:` are each
    registered in the runtime's metal-cpp selector table and each has **zero**
    load sites. The only synchronization selectors with load sites are
    `commit` (8) and `waitUntilCompleted` (6).
    """
    var arm = Arm(String("event"), 0, 0)
    var ok = False
    try:
        var e = ctx.create_event()
        _ = e
        ok = True
    except e:
        arm.unavailable(String(e))
    if ok:
        arm.detail = String("an event was created; a narrower wait may exist")
    _report(arm, 0, 0.0)


# ---------------------------------------------------------------------------
# The queue-depth ladder
# ---------------------------------------------------------------------------


def _ladder(
    mut ctx: DeviceContext,
    mut scratch: DeviceBuffer[DType.int32],
    trials: Int,
) raises:
    """Per-launch *enqueue* cost against launch-stream length.

    See the module docstring for what this tests and why there is no knob to
    test it with. `enq_us` is the enqueue loop divided by N; `tail_us` is the
    wait after it. A depth limit that blocks shows up as `enq_us` rising past
    the knee while `tail_us` stops growing, because the work is being absorbed
    during the enqueue rather than after it.
    """
    # Points straddling the 64 that section 6.2 derives.
    var depths = List[Int]()
    for n in [1, 8, 16, 32, 48, 64, 80, 96, 128, 192, 256, 384, 512]:
        depths.append(n)
    print("depth.status: measured")
    print(
        "depth.knob:",
        "none exists; verified by zero load sites on both raising selectors,"
        " by no such environment variable, and by the four DeviceContext"
        " constructor overloads the compiler lists",
    )
    for d in range(len(depths)):
        var n = depths[d]
        var best_enq = 0.0
        var best_tail = 0.0
        for t in range(trials):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for _ in range(n):
                ctx.enqueue_function[_touch](
                    scratch.unsafe_ptr(), Int32(1024),
                    grid_dim=4, block_dim=256,
                )
            var t1 = perf_counter_ns()
            ctx.synchronize()
            var t2 = perf_counter_ns()
            var enq = Float64(t1 - t0) / 1e3 / Float64(n)
            var tail = Float64(t2 - t1) / 1e3
            if t == 0 or enq < best_enq:
                best_enq = enq
            if t == 0 or tail < best_tail:
                best_tail = tail
        print(
            "depth.point:",
            n,
            "enq_us:",
            _round2(best_enq),
            "tail_us:",
            _round2(best_tail),
        )
