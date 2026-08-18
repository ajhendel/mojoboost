"""Persistent GPU runtime: one session per estimator, and the dependency
model that says which host synchronizations are actually required.

A GPU entry point in train_gpu.mojo that is not handed a session constructs
a fresh `GpuHistogramBuilder`, and that constructor opens a `DeviceContext`,
allocates seven device buffers and three pinned host buffers, uploads the
binned matrix, and blocks on
`ctx.synchronize()`. An estimator that fits twice, that fits and then scores
a validation matrix, or that runs a small grid search pays all of that again
each time, on a dataset the device already holds. This module is the state
that outlives a single `fit`:

- `GpuSession` owns exactly one `DeviceContext` and the bookkeeping below.
- `PoolLedger` decides allocate / grow / reuse for each buffer role, so a
  second fit on the same or smaller data allocates nothing.
- `ResidencyLedger` records which logical matrices (training, validation)
  are already device-resident, keyed by shape *and* a content fingerprint,
  so a re-fit on identical data skips the upload and a re-fit on different
  data cannot silently reuse stale bins.
- `StagingRing` rotates pinned staging slots so the host can convert the
  next round's gradients while the previous round's copy is still in flight.
  On Metal no copy is ever still in flight; see the note on its struct.
- `HazardTracker` is the dependency model. It is the piece that matters.
- `PhaseCounters` attributes time to compile, allocation, transfer, kernel,
  synchronization, and cleanup phases.

`PhaseCounters` is not the profiler, and the distinction is worth keeping.
Its phases are a *session's* -- what one `DeviceContext` spent its life on --
and it has no notion of a node, of a tree, or of the host backend, so it
cannot say whether a run's kernel time went to four large nodes or to a
hundred small ones. `phase_profile.mojo` answers that second question, with
its own grower-shaped vocabulary (histogram, partition, split search,
transfer, conversion, and the round-level gradient fill and score update),
each broken down by node size class and carrying dispatch and synchronization
counts beside the time, and it covers the CPU backend under the same names.
Neither is derivable from the other and neither should grow into the other.
`MOJOTREES_GPU_TRACE=1` turns this one on; `MOJOTREES_PHASE_PROFILE` turns
that one on.

The dependency model
--------------------

A `DeviceContext` queue is in order, so device work never needs a host
synchronization to see earlier device work. A host synchronization is
required in exactly two situations:

- the host is about to *read* memory the device has an unretired write to
  (downloading a histogram into pinned memory, then converting it), or
- the host is about to *write* memory the device has an unretired read or
  write to (refilling a staging buffer whose copy may still be running,
  or rewriting leaf ids while a histogram kernel is still scanning them).

`HazardTracker` tracks those two flags per resource and answers the
question. `ctx.synchronize()` drains the whole queue, so a required
synchronization clears every resource at once; that is why the tracker
counts *elided* checks separately from *required* ones. An elided check is
a place where today's code synchronizes unconditionally and the model says
it did not have to.

One backend fact the model does not encode, and must not be read as denying.
`docs/GPU_PORTABILITY.md` section 6.1 records that on Metal `enqueue_copy` is
a synchronous full-queue drain in both directions, **measured** by
disassembling the shipped runtime. **Section 6.5 corrects that on
2026-08-16: it is true of one of `enqueue_copy`'s two implementations and
false of the other, and the destination chooses which one runs.** A copy into
an arbitrary host pointer is the synchronous drain the disassembly found. A
copy into a pinned `HostBuffer` -- which is what every transfer in this
library actually uses -- is **asynchronous**, and section 6.5 carries the
measurement.

Two things follow for anyone reading this model's output on that backend, and
they are the opposite of what stood here before. A pinned copy is *not* a
synchronization, so it does not belong in a hazard count merely for being a
copy; the queue ordering the model is written against is the whole story for
it. And the second hazard class above -- the host about to overwrite a staging
buffer a copy may still be reading -- **is real on Metal, not vacuous**, and
an elided check the tracker reports against `RES_STAGE` is a check that was
genuinely needed. That class was documented here as vacuous from 2026-08-15
until 2026-08-16, and code written against that claim has a data race in it.

The model stays written against queue ordering rather than against any
backend, which is what makes it portable and what makes it conservative in
the safe direction. That conservatism is now load-bearing rather than
decorative.

A third thing follows, and it is the one this file is most likely to be
misread on. **What this tracker counts is hazards, not time.** Section 6.1.1,
withdrawn 2026-08-16, records that draining a queue holding nothing costs
nothing, and that the count which predicts seconds is the count of *round
trips*: host code blocking on a device answer it needs before it can decide
what to enqueue next. Removing thirteen copies per tree from the
device-resident plane **measured** 0.016 seconds at 1,000,000 x 50 against a
registered prediction of 0.64, a null under M0
(`bench/results/session3_2026-08-16/RESULTS.md`). So a required check this
model reports is a real ordering constraint and an elided one is a real
redundancy, and neither number converts to a saving.

Section 6.5, later the same day, supplies the *mechanism* that null was
missing: a pinned copy is nearly free because it never waited in the first
place, not because the drain it performed was cheap. `StagingRing` therefore
does **not** inherit the correction that once stood here. It was written as
though a second slot could never avoid a wait on Metal, and that was wrong.

Nothing in this file removes a synchronization from histogram_gpu.mojo or
train_gpu.mojo, and nothing here has been measured. It is the model and the
instrumentation that a removal has to be argued from, plus
`audit_round`, which replays the current per-round operation sequence
through the model so the argument is executable rather than prose. The
handoff (`handoffs/apple_a5_runtime.md (deleted, recover with git log --all --diff-filter=D -- handoffs/apple_a5_runtime.md)`) lists which of today's
synchronizations the model marks removable and what has to be proven first.

Environment contract, matching the `MOJOTREES_` convention in parallel.mojo:

- `MOJOTREES_GPU_TRACE=1` turns on the phase counters. Off by default so an
  untraced session pays no clock reads. Not to be confused with
  `MOJOTREES_PHASE_PROFILE`, which is the per-node-size profiler in
  phase_profile.mojo and answers a different question; see above.
- `MOJOTREES_GPU_STAGING_SLOTS`: staging ring depth, default
  `DEFAULT_STAGING_SLOTS`, clamped to `[1, MAX_STAGING_SLOTS]`. `1`
  reproduces today's single-buffer behavior exactly.
- `MOJOTREES_STARTUP_TRACE=1` times the one-time phases (initialization.mojo)
  a session actually performs: context creation, device discovery, kernel
  creation, and first versus warm fit. Off by default, and the phase *counts*
  are kept either way, which is what `session_state()` answers from.
- `MOJOTREES_GPU_READBACK` names the transport a small device answer comes
  home on, from `readback_transport_name`. Default `plain_one`, which the
  probe measured at 124.85 us a trip against the `pinned_pair_sync` shape
  that shipped before it at 202.14; `require_readback_correct` refuses the
  two spellings this backend is measured to get wrong, and
  `GpuSplitSearcher.set_readback_transport` refuses those and `map`. See the
  readback section below and `probes/readback_cost.mojo`.
- `MOJOTREES_GPU_WARMUP` (`off`, `train`, `all`) names which kernels a
  session intends to front-load. Nothing is created up front yet; what the
  plan buys today is that the first launch of a named kernel is attributed to
  warm-up rather than to the first round.

The startup half of this is deliberately owned here rather than duplicated.
`initialization.mojo` defines the phases, the cold/warm split, and the
`SessionState` that `device_policy.decide_device` takes as data, and its
`SessionState` docstring names `GpuSession` as the object that can answer
truthfully whether a context is open. `session_state()` is that answer.
"""

from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from .gpu_tiling import DeviceCaps, query_device_caps
from .initialization import (
    PHASE_CONTEXT_CREATE,
    PHASE_DEVICE_DISCOVERY,
    PHASE_FIRST_ALLOC,
    PHASE_FIRST_TRANSFER,
    PHASE_KERNEL_CREATE,
    WARMUP_ALL,
    WARMUP_TRAIN,
    FitLatency,
    SessionState,
    StartupTrace,
    WarmupPlan,
    session_state_from_trace,
)
from .parallel import _env_int


# ---------------------------------------------------------------------------
# Instrumentation phases
# ---------------------------------------------------------------------------

comptime PHASE_COMPILE = 0
comptime PHASE_ALLOC = 1
comptime PHASE_TRANSFER = 2
comptime PHASE_KERNEL = 3
comptime PHASE_SYNC = 4
comptime PHASE_CLEANUP = 5
comptime N_PHASES = 6


def _elapsed_since(started: Int) -> Int:
    """Nanoseconds since `started`, or 0 when `started` came from a disabled
    clock or the reading went backwards. The one place this arithmetic is
    written outside a `record`."""
    if started <= 0:
        return 0
    var elapsed = Int(perf_counter_ns()) - started
    if elapsed < 0:
        return 0
    return elapsed


def session_phase_name(phase: Int) -> String:
    if phase == PHASE_COMPILE:
        return String("compile")
    if phase == PHASE_ALLOC:
        return String("alloc")
    if phase == PHASE_TRANSFER:
        return String("transfer")
    if phase == PHASE_KERNEL:
        return String("kernel")
    if phase == PHASE_SYNC:
        return String("sync")
    if phase == PHASE_CLEANUP:
        return String("cleanup")
    return String("unknown")


struct PhaseCounters(Copyable, Movable):
    """Per-phase call counts and elapsed nanoseconds.

    Disabled by default: `record` still counts calls, which costs an integer
    add, but a disabled session never reads the clock, so an untraced fit
    carries no timing overhead at all. Counting calls unconditionally is
    what makes the lifecycle tests able to assert how many transfers or
    synchronizations a sequence performed without turning tracing on.
    """

    var calls: List[Int]
    var nanos: List[Int]
    var enabled: Bool

    def __init__(out self, enabled: Bool = False):
        self.calls = List[Int](capacity=N_PHASES)
        self.nanos = List[Int](capacity=N_PHASES)
        for _ in range(N_PHASES):
            self.calls.append(0)
            self.nanos.append(0)
        self.enabled = enabled

    @staticmethod
    def from_env() -> PhaseCounters:
        """Counters configured by `MOJOTREES_GPU_TRACE`."""
        return PhaseCounters(getenv("MOJOTREES_GPU_TRACE") == "1")

    def clock(self) -> Int:
        """A start timestamp, or 0 when tracing is off."""
        if not self.enabled:
            return 0
        return Int(perf_counter_ns())

    def record(mut self, phase: Int, started: Int) raises:
        """Count one occurrence of `phase`, and its duration when `started`
        came from `clock()` on an enabled counter set."""
        if phase < 0 or phase >= N_PHASES:
            raise Error("unknown runtime phase ", phase)
        self.calls[phase] += 1
        if self.enabled and started > 0:
            var elapsed = Int(perf_counter_ns()) - started
            if elapsed > 0:
                self.nanos[phase] += elapsed

    def calls_of(self, phase: Int) -> Int:
        if phase < 0 or phase >= N_PHASES:
            return 0
        return self.calls[phase]

    def nanos_of(self, phase: Int) -> Int:
        if phase < 0 or phase >= N_PHASES:
            return 0
        return self.nanos[phase]

    def total_calls(self) -> Int:
        var total = 0
        for p in range(N_PHASES):
            total += self.calls[p]
        return total

    def total_nanos(self) -> Int:
        var total = 0
        for p in range(N_PHASES):
            total += self.nanos[p]
        return total

    def reset(mut self):
        for p in range(N_PHASES):
            self.calls[p] = 0
            self.nanos[p] = 0

    def report(self) -> String:
        """One line per phase: `name calls nanos`. Nanoseconds are 0 for
        every phase when tracing is off, and that is not a measurement."""
        var out = String("")
        for p in range(N_PHASES):
            out += session_phase_name(p) + " " + String(self.calls[p])
            out += " " + String(self.nanos[p]) + "\n"
        return out


# ---------------------------------------------------------------------------
# Device resources and the hazards between them
# ---------------------------------------------------------------------------

# The buffers histogram_gpu.mojo holds, named here so the dependency model
# can talk about them without importing the builder.
comptime RES_BINS = 0
comptime RES_LEAF = 1
comptime RES_GRAD = 2
comptime RES_HESS = 3
comptime RES_FEAT = 4
comptime RES_OUT = 5
comptime RES_PART = 6
comptime RES_STAGE = 7
comptime RES_HOST_OUT = 8
# Reserved for the validation matrix and its device-side scores, so a
# prediction path can share one tracker with training.
comptime RES_VALID_BINS = 9
comptime RES_VALID_SCORE = 10
comptime N_RESOURCES = 11


def resource_name(resource: Int) -> String:
    if resource == RES_BINS:
        return String("bins")
    if resource == RES_LEAF:
        return String("leaf")
    if resource == RES_GRAD:
        return String("grad")
    if resource == RES_HESS:
        return String("hess")
    if resource == RES_FEAT:
        return String("feat")
    if resource == RES_OUT:
        return String("out")
    if resource == RES_PART:
        return String("part")
    if resource == RES_STAGE:
        return String("stage")
    if resource == RES_HOST_OUT:
        return String("host_out")
    if resource == RES_VALID_BINS:
        return String("valid_bins")
    if resource == RES_VALID_SCORE:
        return String("valid_score")
    return String("unknown")


# Why a synchronization happened, so a trace can distinguish a download the
# host genuinely waits on from a teardown drain.
comptime SYNC_HOST_READ = 0
comptime SYNC_HOST_WRITE = 1
comptime SYNC_TEARDOWN = 2
comptime SYNC_EXPLICIT = 3
comptime N_SYNC_REASONS = 4


def sync_reason_name(reason: Int) -> String:
    if reason == SYNC_HOST_READ:
        return String("host_read")
    if reason == SYNC_HOST_WRITE:
        return String("host_write")
    if reason == SYNC_TEARDOWN:
        return String("teardown")
    if reason == SYNC_EXPLICIT:
        return String("explicit")
    return String("unknown")


struct HazardTracker(Copyable, Movable):
    """Which resources have device work the host has not waited for.

    Two flags per resource: an unretired device write (the host cannot read
    it yet) and an unretired device read (the host cannot overwrite it yet).
    Device-to-device ordering needs no flags at all, because the queue is in
    order, which is precisely the reason so many synchronizations can go.

    `sync` clears every resource, because `DeviceContext.synchronize()` is a
    whole-queue drain and there is no finer-grained wait in use. A model with
    per-event waits would clear less and elide more; this one is deliberately
    the conservative version of the argument.

    **There is no finer-grained wait to be had on Metal, and that is now
    established rather than assumed.** `DeviceContext.create_event()` compiles
    and raises `eventCreate is not supported on this device`;
    `create_stream()` raises `createStream is not supported on this device`
    and `num_streams()` answers 1; `enqueue_cpu_function` raises `only
    supported on CPU DeviceContexts`, so a completion callback cannot be
    turned into a flag either. Independently, in the shipped runtime,
    `newSharedEvent`, `newEvent`, `newFence`, `encodeSignalEvent:value:`,
    `encodeWaitForEvent:value:`, `notifyListener:atValue:block:` and
    `addCompletedHandler:` are each registered in the metal-cpp selector table
    and each has **zero** load sites; the only synchronization selectors with
    load sites are `commit` and `waitUntilCompleted`. So on this backend the
    conservative model is not merely a safe choice, it is the only expressible
    one, and an elided check is the only kind of saving available.
    `docs/GPU_PORTABILITY.md` section 6.5 carries all of it.
    """

    var n_resources: Int
    var pending_write: List[Bool]
    var pending_read: List[Bool]
    var by_reason: List[Int]
    var elided: Int

    def __init__(out self, n_resources: Int = N_RESOURCES) raises:
        if n_resources < 1:
            raise Error("hazard tracker needs at least one resource")
        self.n_resources = n_resources
        self.pending_write = List[Bool](capacity=n_resources)
        self.pending_read = List[Bool](capacity=n_resources)
        for _ in range(n_resources):
            self.pending_write.append(False)
            self.pending_read.append(False)
        self.by_reason = List[Int](capacity=N_SYNC_REASONS)
        for _ in range(N_SYNC_REASONS):
            self.by_reason.append(0)
        self.elided = 0

    def _check(self, resource: Int) raises:
        if resource < 0 or resource >= self.n_resources:
            raise Error("unknown device resource ", resource)

    def note_device_read(mut self, resource: Int) raises:
        """A queued kernel or copy reads `resource`."""
        self._check(resource)
        self.pending_read[resource] = True

    def note_device_write(mut self, resource: Int) raises:
        """A queued kernel, memset, or copy writes `resource`."""
        self._check(resource)
        self.pending_write[resource] = True

    def host_read_hazard(self, resource: Int) -> Bool:
        if resource < 0 or resource >= self.n_resources:
            return True
        return self.pending_write[resource]

    def host_write_hazard(self, resource: Int) -> Bool:
        if resource < 0 or resource >= self.n_resources:
            return True
        return self.pending_write[resource] or self.pending_read[resource]

    def any_pending(self) -> Bool:
        for r in range(self.n_resources):
            if self.pending_write[r] or self.pending_read[r]:
                return True
        return False

    def sync(mut self, reason: Int) raises:
        """Record a whole-queue drain: every resource becomes host-safe."""
        if reason < 0 or reason >= N_SYNC_REASONS:
            raise Error("unknown synchronization reason ", reason)
        self.by_reason[reason] += 1
        for r in range(self.n_resources):
            self.pending_write[r] = False
            self.pending_read[r] = False

    def sync_for_host_read(mut self, resource: Int) raises -> Bool:
        """Drain only if the host cannot read `resource` yet. True when it
        had to."""
        self._check(resource)
        if not self.host_read_hazard(resource):
            self.elided += 1
            return False
        self.sync(SYNC_HOST_READ)
        return True

    def sync_for_host_write(mut self, resource: Int) raises -> Bool:
        """Drain only if the host cannot overwrite `resource` yet. True when
        it had to."""
        self._check(resource)
        if not self.host_write_hazard(resource):
            self.elided += 1
            return False
        self.sync(SYNC_HOST_WRITE)
        return True

    def syncs_for(self, reason: Int) -> Int:
        if reason < 0 or reason >= N_SYNC_REASONS:
            return 0
        return self.by_reason[reason]

    def required(self) -> Int:
        """Total drains the model performed."""
        var total = 0
        for i in range(N_SYNC_REASONS):
            total += self.by_reason[i]
        return total

    def report(self) -> String:
        var out = String("")
        for i in range(N_SYNC_REASONS):
            out += "sync." + sync_reason_name(i)
            out += " " + String(self.by_reason[i]) + "\n"
        out += "sync.elided " + String(self.elided) + "\n"
        return out


# ---------------------------------------------------------------------------
# Staging ring
# ---------------------------------------------------------------------------

comptime DEFAULT_STAGING_SLOTS = 2
comptime MAX_STAGING_SLOTS = 8


def env_staging_slots() -> Int:
    """`MOJOTREES_GPU_STAGING_SLOTS`, clamped to a usable ring depth."""
    var n = _env_int("MOJOTREES_GPU_STAGING_SLOTS", DEFAULT_STAGING_SLOTS)
    if n < 1:
        return 1
    if n > MAX_STAGING_SLOTS:
        return MAX_STAGING_SLOTS
    return n


struct StagingRing(Copyable, Movable):
    """Rotation over pinned host staging slots.

    `stage_gradients` converts Float64 gradients into pinned Float32 memory
    and `upload_staged` copies that memory to the device. With one slot the
    host cannot start the next round's conversion until the previous copy has
    retired, which is the `ctx.synchronize()` at the top of
    `stage_gradients` today. With two, the conversion of round i+1 targets
    the slot round i is not using, so the wait only happens if two rounds are
    staged with no drain in between.

    The ring is bookkeeping only: it says which slot to fill and whether
    filling it needs a wait. Rotating slots changes no arithmetic, because
    the same Float64 values are converted to the same Float32 values whatever
    memory holds them.

    Retirement is derived, not tracked separately. A slot records the queue
    drain count at the moment its copy was enqueued, and it is free again as
    soon as that count has moved: `synchronize()` drains everything, so one
    drain from any cause retires every outstanding copy. Deriving it this
    way means the ring cannot disagree with the hazard tracker about what
    the device has finished, which a second set of flags would eventually
    manage to do.

    **The hazard this ring exists for is real on Metal. Corrected
    2026-08-16.** From 2026-08-15 until then this docstring said the
    opposite: that `enqueue_copy` there is a synchronous drain in both
    directions, so a copy is retired the instant it returns, `pending()` is
    False for every slot on every call, and `MOJOTREES_GPU_STAGING_SLOTS`
    above 1 changes nothing observable. Every clause of that is wrong for the
    copy this library actually issues.

    `docs/GPU_PORTABILITY.md` section 6.5 has the measurement.
    `enqueue_copy(dst_buf=..., src_ptr=<pinned HostBuffer>)` on Metal is
    **asynchronous**: it enqueues a blit and returns. The hazard was
    reproduced directly. A host buffer was filled with ones, uploaded, and
    then overwritten with sevens with no drain in between; the device summed
    what it received. Across four repetitions the device saw the sevens once
    and the ones three times -- **non-deterministically**, which is the
    signature of a live race rather than of either fixed ordering.

    Three consequences, all of which the old text denied:

    - `pending()` is a real question and its answer is sometimes True.
    - A second slot really can avoid a wait, so
      `MOJOTREES_GPU_STAGING_SLOTS` above 1 has a mechanism behind it. What
      it is worth is still unmeasured, and a number would need the
      interleaved harness.
    - Any code that refills a staging arena without first draining or
      rotating has a data race that will be silent at small sizes and appear
      under load, because whether the blit or the host store wins depends on
      how long the queue ahead of it is. `histogram_gpu.stage_gradients`
      drains, so the shipping path is safe; its docstring's *reason* for
      keeping the drain ("this synchronize has nothing of its own to wait
      for") is the claim that is wrong, and the drain must not be removed on
      the strength of it.
    """

    var n_slots: Int
    var cursor: Int
    var enqueued_at: List[Int]
    var waits: Int
    var acquisitions: Int

    def __init__(out self, n_slots: Int = DEFAULT_STAGING_SLOTS) raises:
        if n_slots < 1:
            raise Error("staging ring needs at least one slot")
        if n_slots > MAX_STAGING_SLOTS:
            raise Error("staging ring is limited to ", MAX_STAGING_SLOTS)
        self.n_slots = n_slots
        self.cursor = 0
        self.enqueued_at = List[Int](capacity=n_slots)
        for _ in range(n_slots):
            self.enqueued_at.append(-1)
        self.waits = 0
        self.acquisitions = 0

    def pending(self, drains: Int) -> Bool:
        """True when the next slot's copy was enqueued and no drain has
        happened since, so the host cannot refill it yet."""
        var at = self.enqueued_at[self.cursor]
        return at >= 0 and drains <= at

    def note_wait(mut self):
        """The caller had to drain to reuse the next slot."""
        self.waits += 1

    def acquire(mut self) -> Int:
        """Take the next slot for host writing and advance the cursor. The
        caller must have resolved `pending()` first."""
        var slot = self.cursor
        self.enqueued_at[slot] = -1
        self.cursor += 1
        if self.cursor >= self.n_slots:
            self.cursor = 0
        self.acquisitions += 1
        return slot

    def mark_in_flight(mut self, slot: Int, drains: Int) raises:
        """A copy reading `slot` has been enqueued, with the queue drained
        `drains` times so far."""
        if slot < 0 or slot >= self.n_slots:
            raise Error("staging slot out of range")
        self.enqueued_at[slot] = drains

    def in_flight(self, drains: Int) -> Int:
        """How many slots still have an unretired copy reading them."""
        var n = 0
        for i in range(self.n_slots):
            var at = self.enqueued_at[i]
            if at >= 0 and drains <= at:
                n += 1
        return n


# ---------------------------------------------------------------------------
# Readback transports
# ---------------------------------------------------------------------------
#
# How a small device answer gets to the host, as run-time-selectable arms with
# their costs and their correctness stated per backend. The 136-byte split
# record is the case this exists for: `_device_search_resident` reads one per
# split, and that read is the round trip that predicts the plane's time.
#
# This is policy, not mechanism. It holds no context and issues no copy, in
# the same way `PoolLedger` holds sizes and not buffers, so it is testable on
# a machine with no accelerator and so the module that owns the buffers stays
# the module that touches them. `GpuSplitSearcher.download_words` in
# gpu_split_search.mojo is the adoption site; the transports below are named
# from its point of view.
#
# Why this is a table and not a constant
# --------------------------------------
#
# Two of the five transports were believed correct and are not, and the
# believing was done from a docstring. `probes/readback_cost.mojo` executes
# every one of them against a per-arm tag under a kernel slow enough to make
# the answer deterministic, and reports which delivered the record. The
# `correct_on_metal` column below is that result and nothing else. Anything
# that reads this table gets the measured answer rather than the argued one.

comptime READBACK_PINNED_PAIR_SYNC = 0
comptime READBACK_PINNED_ONE_SYNC = 1
comptime READBACK_PLAIN_PAIR = 2
comptime READBACK_PLAIN_ONE = 3
comptime READBACK_MAP = 4
comptime READBACK_PINNED_PAIR_NOSYNC = 5
comptime READBACK_PINNED_ONE_NOSYNC = 6
comptime N_READBACK_TRANSPORTS = 7

comptime READBACK_DEFAULT = READBACK_PLAIN_ONE
"""What ships, and as of 2026-08-16 it is a measurement rather than a
placeholder.

The measurement is `pixi run probe-readback` on an Apple M4, every arm
interleaved inside one process so that the several-fold drift this machine
shows between time windows cannot separate them:

| arm | command buffers/trip | per-trip |
|---|---|---|
| `bare_sync` (the floor) | 1 | 10.59 us |
| `plain_one` | 2 | 124.85 us |
| `pinned_pair_sync` | 4 | 202.14 us |
| `map` | 3 | 349.47 us |

`plain_one` is 38 percent under `pinned_pair_sync`, which is what this
library shipped until the same date, for the same 34 words: one packed
device buffer instead of two, one ordinary heap destination instead of two
pinned ones, two command buffers instead of four. Nothing about a record's
value changes, which is why this is a default flip and not a gated arm.

This default was `READBACK_PINNED_PAIR_SYNC` when the table was written,
deliberately, because the lane that wrote it had produced no timings and
`docs/GPU_PORTABILITY.md` section 6.4 records what happens here when a
command-buffer count is turned into a prediction without one. The count and
the measurement agree in rank order; the flip rests on the measurement."""


def readback_transport_name(transport: Int) -> String:
    if transport == READBACK_PINNED_PAIR_SYNC:
        return String("pinned_pair_sync")
    if transport == READBACK_PINNED_ONE_SYNC:
        return String("pinned_one_sync")
    if transport == READBACK_PLAIN_PAIR:
        return String("plain_pair")
    if transport == READBACK_PLAIN_ONE:
        return String("plain_one")
    if transport == READBACK_MAP:
        return String("map")
    if transport == READBACK_PINNED_PAIR_NOSYNC:
        return String("pinned_pair_nosync")
    if transport == READBACK_PINNED_ONE_NOSYNC:
        return String("pinned_one_nosync")
    return String("unknown")


@fieldwise_init
struct ReadbackTransport(Copyable, Movable):
    """One way of moving a small record from device to host, priced and
    graded.

    `command_buffers` counts what the transport commits, including the kernel
    that produced the record, because that is the count a Metal System Trace
    reports and comparing against a different denominator is how the five-fold
    error in section 6.4 happened. `host_waits` counts the ones the host
    blocks on, which is the count that predicts time.

    `correct_on_metal` is a **measured** result from
    `probes/readback_cost.mojo`, not a judgement. `correct_elsewhere` is
    deliberately True for the two arms Metal rejects, because what makes them
    wrong here is Metal's asynchronous pinned copy, and a backend whose copy
    is synchronous would accept them. Neither CUDA nor HIP has been checked;
    see the note on the field.
    """

    var transport: Int
    var command_buffers: Int
    var host_waits: Int
    var pinned_destination: Bool
    """Whether the copy lands in memory from `enqueue_create_host_buffer`.

    **The correctness discriminator, not a description.** Section 6.5.1 of
    `docs/GPU_PORTABILITY.md` establishes by execution that `enqueue_copy`
    has two implementations on Metal and that this field picks which one
    runs: into a pinned `HostBuffer` it enqueues a blit and returns, so the
    host must wait before reading; into an arbitrary host pointer it commits,
    waits, and memcpys inside itself, so there is nothing left to wait for.

    An adoption site that drops the trailing `synchronize()` is therefore
    obliged to check this field rather than to reason about the destination
    it thinks it passed, because the failure is latency-dependent: the unsafe
    shape passes under a fast kernel and returns the previous record under a
    slow one. `GpuSplitSearcher.download_words` asserts on it."""

    var packed_source: Bool
    """Whether one copy moves the whole record, or one copy moves each plane.

    A packed transport requires the integer and float planes to be one
    device allocation, which they are: `GpuSplitSearcher.records_dev` owns
    them and `rec_i_dev` / `rec_f_dev` are `create_sub_buffer` windows onto
    it. Before that they were two allocations and only the pair transports
    were reachable, which is why the two packed rows below used to say the
    layout was a change someone still had to make."""

    var correct_on_metal: Bool
    var correct_elsewhere: Bool
    """True where the transport's correctness depends only on queue ordering,
    which every backend provides. **Unestablished** on CUDA and HIP as a
    statement about MAX: nothing in this repository has run either. It is the
    weaker claim that the transport does not additionally depend on the copy
    being synchronous."""

    var note: String


def readback_transport(transport: Int) raises -> ReadbackTransport:
    """The row for one transport.

    Command buffer counts include the record-producing kernel, so every row
    is on the same denominator and a reader can subtract it once rather than
    per row.
    """
    if transport == READBACK_PINNED_PAIR_SYNC:
        return ReadbackTransport(
            transport, 4, 1, True, False, True, True,
            String(
                "the shape this library shipped until 2026-08-16: one"
                " enqueue_copy per record plane into pinned host memory, then"
                " synchronize(). The two copies are asynchronous blits, so the"
                " round trip is the synchronize alone. Measured 202.14 us a"
                " trip, and still reachable as the A/B arm"
            ),
        )
    if transport == READBACK_PINNED_ONE_SYNC:
        return ReadbackTransport(
            transport, 3, 1, True, True, True, True,
            String(
                "the same with both planes in one device buffer: one blit"
                " instead of two, same single wait. Reachable since"
                " GpuSplitSearcher.records_dev made the record one allocation"
            ),
        )
    if transport == READBACK_PLAIN_PAIR:
        return ReadbackTransport(
            transport, 3, 2, False, False, True, False,
            String(
                "two enqueue_copy calls into ordinary heap memory and no"
                " synchronize at all. That destination kind takes MAX's"
                " synchronous path, so each copy drains inside itself: fewer"
                " command buffers than the pinned pair and one more wait"
            ),
        )
    if transport == READBACK_PLAIN_ONE:
        return ReadbackTransport(
            transport, 2, 1, False, True, True, False,
            String(
                "one enqueue_copy into ordinary heap memory. The whole"
                " readback is the kernel's command buffer and the copy's: the"
                " fewest of any correct arm, and no pinned staging buffer to"
                " keep alive. Measured 124.85 us a trip against the pinned"
                " pair's 202.14, and READBACK_DEFAULT since 2026-08-16"
            ),
        )
    if transport == READBACK_MAP:
        return ReadbackTransport(
            transport, 3, 2, False, False, True, False,
            String(
                "map_to_host(). Documented as a host-accessible view; on"
                " Metal it is a fresh host allocation with a copy in on entry"
                " and a copy out on exit, so it is the most expensive"
                " transport here and not the cheapest. Measured 349.47 us, and"
                " GpuSplitSearcher.download_words declines to implement it"
            ),
        )
    if transport == READBACK_PINNED_PAIR_NOSYNC:
        return ReadbackTransport(
            transport, 3, 0, True, False, False, False,
            String(
                "the pinned pair with the trailing synchronize removed, which"
                " docs/GPU_PORTABILITY.md 6.1 licensed. WRONG on Metal: the"
                " pinned copy is asynchronous, so the host reads the previous"
                " record. Measured 34 of 34 words wrong"
            ),
        )
    if transport == READBACK_PINNED_ONE_NOSYNC:
        return ReadbackTransport(
            transport, 2, 0, True, True, False, False,
            String(
                "the packed shape with the same edit, and wrong for the same"
                " reason: 34 of 34 words"
            ),
        )
    raise Error("unknown readback transport ", transport)


def default_readback_for(api_is_metal: Bool) -> Int:
    """The shipped transport for a backend, rather than for Metal only.

    `READBACK_DEFAULT` is `READBACK_PLAIN_ONE`, and that choice is a
    measurement: 124.85 us a trip against the pinned pair's 202.14, taken on
    an M4 with every arm interleaved in one process. It is the right default
    **there**, and the reason it is safe there is that MAX's Metal runtime
    lowers a copy into ordinary heap memory as commit-wait-memcpy, so the
    copy drains inside itself and supplies the ordering.

    That is a disassembled implementation detail of one vendor's runtime, not
    an API guarantee. On a backend where the copy is asynchronous the same
    arm reads the *previous* record: the split's feature, bin, gain,
    `default_left` and child row counts all come back stale, and every launch
    geometry and allocation downstream is sized from them.

    So the default is now a function of the backend. Metal keeps the fast arm
    it earned. Anything else gets `READBACK_PINNED_ONE_SYNC`, the cheapest
    arm whose correctness rests on queue ordering alone -- one wait it
    performs itself rather than one it inherits. It costs a wait per trip and
    it is correct everywhere.

    This is not a permanent verdict on other backends. It is what holds until
    someone runs `pixi run probe-readback` there and records the result, at
    which point that backend can earn the faster arm the same way Metal did.
    """
    if api_is_metal:
        return READBACK_DEFAULT
    return READBACK_PINNED_ONE_SYNC


def env_readback_transport() raises -> Int:
    """`MOJOTREES_GPU_READBACK`, or `READBACK_DEFAULT`.

    Named transports only. A numeric spelling is refused rather than accepted,
    because the identifiers above are an implementation detail and a benchmark
    row reading `MOJOTREES_GPU_READBACK=3` says nothing to the person reading
    it a month later.

    Prefer `env_readback_transport_for(api_is_metal)` at any site that knows
    which device it is talking to. This spelling keeps `READBACK_DEFAULT` and
    is retained for callers that have no context, but it hands back the Metal
    default on every backend, which is what shipped a stale-read hazard to
    CUDA in the first place.
    """
    var raw = getenv("MOJOTREES_GPU_READBACK")
    if raw == "":
        return READBACK_DEFAULT
    for t in range(N_READBACK_TRANSPORTS):
        if raw == readback_transport_name(t):
            return t
    raise Error(
        "MOJOTREES_GPU_READBACK must name a transport, not ",
        raw,
        "; see gpu_runtime.readback_transport_name",
    )


def env_readback_transport_for(api_is_metal: Bool) raises -> Int:
    """`MOJOTREES_GPU_READBACK`, or the backend's own default.

    An explicit environment request still wins, because a probe has to be
    able to run an arm the shipped default would not choose. What changes is
    the fallback: with nothing set, a non-Metal device gets an arm whose
    correctness does not rest on a Metal implementation detail.
    """
    var raw = getenv("MOJOTREES_GPU_READBACK")
    if raw == "":
        return default_readback_for(api_is_metal)
    return env_readback_transport()


def require_readback_correct(transport: Int, api_is_metal: Bool) raises:
    """Refuse a transport this backend is known to get wrong.

    The two unsafe arms exist in the table so that a probe can execute them
    and so that the refutation is checkable rather than asserted. They must
    not be reachable from a fit, and this is the gate that keeps them out. It
    fires on the measured column, so it cannot drift away from what the probe
    reports without someone editing the row the probe wrote.

    Why there are two columns and not one
    -------------------------------------
    `correct_on_metal` is a **measurement**: `probes/readback_cost.mojo` ran
    every arm on an M4 and recorded which ones delivered the record.
    `correct_elsewhere` is a **weaker structural claim**: that the arm's
    correctness rests only on queue ordering, which every backend provides,
    rather than additionally on the copy being synchronous inside itself.

    Until 2026-08-18 this function read only the first column, and only when
    the running backend was Metal. `correct_elsewhere` had **zero readers in
    the entire codebase**. The consequence, found the first time this code
    ever ran on NVIDIA: the shipped default is `READBACK_PLAIN_ONE`, whose
    safety argument is "that destination kind takes MAX's synchronous path,
    so each copy drains inside itself" -- a disassembled fact about Metal's
    runtime, not a portable guarantee -- and on CUDA this gate waved it
    through without a word.

    A guard that only checks the one backend that was already known to work
    is not a guard. So on any non-Metal backend the weaker column now has to
    hold, and an arm whose correctness needs the copy to be self-draining is
    refused rather than assumed.
    """
    var row = readback_transport(transport)
    if api_is_metal and not row.correct_on_metal:
        raise Error(
            "readback transport ",
            readback_transport_name(transport),
            " does not deliver the record on Metal: ",
            row.note,
        )
    if not api_is_metal and not row.correct_elsewhere:
        raise Error(
            "readback transport ",
            readback_transport_name(transport),
            " is only known correct on Metal, and this is not Metal. Its",
            " correctness depends on the copy draining inside itself, which",
            " was measured by disassembling Metal's runtime and is",
            " unestablished on this backend: ",
            row.note,
            ". Use a transport whose correctness rests on queue ordering",
            " alone, or establish this one here first with",
            " `pixi run probe-readback` and record the result.",
        )


def require_readback_table_consistent() raises:
    """The trap of section 6.5.1, written as a check over the whole table.

    A row is not free to say whatever it likes. The destination kind decides
    where the wait lives, and a row that claims a destination and a wait count
    that cannot both hold is a row that will license the wrong edit at an
    adoption site. Two rules, and the second is the one that cost a lane:

    - An **unpinned** destination takes MAX's synchronous path, so the copy
      drains inside itself. Such a row must carry at least one wait (the
      copy's own) and must be correct on Metal. There is no unpinned row that
      needs a trailing `synchronize()`, and none that is wrong here.
    - A **pinned** destination takes the asynchronous blit. Such a row is
      correct only if it carries a wait. A pinned row with zero waits is the
      edit `docs/GPU_PORTABILITY.md` 6.1 licensed and 6.5.1 refuted, and it
      must be marked wrong on Metal so `require_readback_correct` refuses it.
    - An **unpinned** row may not claim `correct_elsewhere`. Added 2026-08-18,
      after the first NVIDIA run: every unpinned row carried
      `correct_elsewhere = True` while its only wait was the Metal copy's own
      drain, and nothing in the codebase read the field, so the claim was both
      wrong and unchecked. Two defects that concealed each other.

    Called by `tests/test_gpu_readback_transport.mojo`, and cheap enough to
    call anywhere; it reads seven rows and allocates no device memory. It
    exists so that a later edit to a row's numbers has to keep the row's
    story straight, rather than being free to make an unsafe arm look safe.
    """
    for t in range(N_READBACK_TRANSPORTS):
        var row = readback_transport(t)
        if not row.pinned_destination:
            if row.host_waits < 1:
                raise Error(
                    "readback transport ",
                    readback_transport_name(t),
                    " has an unpinned destination but claims no wait; the"
                    " synchronous copy path drains inside itself and that"
                    " drain is a wait",
                )
            if not row.correct_on_metal:
                raise Error(
                    "readback transport ",
                    readback_transport_name(t),
                    " has an unpinned destination and is marked wrong on"
                    " Metal; section 6.5.1 measured 0 of 64 stale words on"
                    " that path",
                )
            if row.correct_elsewhere:
                raise Error(
                    "readback transport ",
                    readback_transport_name(t),
                    " has an unpinned destination and claims correctness off"
                    " Metal. It cannot. Its wait IS the copy's own drain, and"
                    " that drain is a disassembled property of MAX's Metal"
                    " runtime, not an API guarantee. An arm earns"
                    " correct_elsewhere by carrying a wait it performs"
                    " itself, or by someone running probes/readback_cost.mojo"
                    " on the other backend and recording the result",
                )
        elif row.host_waits < 1 and row.correct_on_metal:
            raise Error(
                "readback transport ",
                readback_transport_name(t),
                " reads a pinned destination with no wait and claims to be"
                " correct on Metal; that is the asynchronous blit and section"
                " 6.5.1 measured 34 of 34 stale words",
            )


def readback_report() raises -> String:
    """Every transport as `name buffers waits dst packing metal_ok` lines, for
    a probe or a debug print. Not for parsing by anything shipped."""
    var out = String("")
    for t in range(N_READBACK_TRANSPORTS):
        var row = readback_transport(t)
        out += "readback." + readback_transport_name(t)
        out += " " + String(row.command_buffers)
        out += " " + String(row.host_waits)
        out += " " + ("pinned" if row.pinned_destination else "plain")
        out += " " + ("packed" if row.packed_source else "pair")
        out += " " + ("ok" if row.correct_on_metal else "wrong") + "\n"
    return out


# ---------------------------------------------------------------------------
# Residency: which matrices the device already holds
# ---------------------------------------------------------------------------

comptime ROLE_TRAIN = 0
comptime ROLE_VALID = 1
comptime N_SESSION_ROLES = 2


def session_role_name(role: Int) -> String:
    if role == ROLE_TRAIN:
        return String("train")
    if role == ROLE_VALID:
        return String("valid")
    return String("unknown")


comptime _FNV_OFFSET = UInt64(0xCBF29CE484222325)
comptime _FNV_PRIME = UInt64(0x100000001B3)


def bins_fingerprint(
    bins: List[UInt8], n_rows: Int, n_features: Int, n_bins: Int
) -> UInt64:
    """Content fingerprint of a binned matrix (FNV-1a over every cell, with
    the shape mixed in).

    Every cell, not a sample: a sampled fingerprint would let a matrix that
    differs only outside the sample reuse another matrix's device copy, and
    the failure mode is a silently wrong model. The cost is one host pass
    over the same bytes an upload would move, so it is strictly cheaper than
    the upload it can skip, and it is only paid when a session is asked to
    reuse residency at all.
    """
    var h = _FNV_OFFSET
    h = (h ^ UInt64(n_rows)) * _FNV_PRIME
    h = (h ^ UInt64(n_features)) * _FNV_PRIME
    h = (h ^ UInt64(n_bins)) * _FNV_PRIME
    for i in range(len(bins)):
        h = (h ^ UInt64(Int(bins[i]))) * _FNV_PRIME
    return h


@fieldwise_init
struct MatrixIdentity(Copyable, Movable):
    """What has to match for a device-resident matrix to be reusable."""

    var n_rows: Int
    var n_features: Int
    var n_bins: Int
    var fingerprint: UInt64

    @staticmethod
    def empty() -> MatrixIdentity:
        return MatrixIdentity(0, 0, 0, UInt64(0))

    def matches(self, other: MatrixIdentity) -> Bool:
        if self.n_rows != other.n_rows:
            return False
        if self.n_features != other.n_features:
            return False
        if self.n_bins != other.n_bins:
            return False
        if self.fingerprint != other.fingerprint:
            return False
        return True

    def cells(self) -> Int:
        return self.n_rows * self.n_features


struct ResidencyLedger(Copyable, Movable):
    """Which logical matrices the device holds, per role.

    A role is a slot in the session (`ROLE_TRAIN`, `ROLE_VALID`), not a
    buffer: `admit` answers "must I upload this?" and the caller does the
    upload. Identity is shape plus content fingerprint, so two different
    matrices of the same shape never alias, which is the one way a residency
    cache can change a result.
    """

    var identity: List[MatrixIdentity]
    var resident: List[Bool]
    var uploads: Int
    var reuses: Int
    var evictions: Int

    def __init__(out self):
        self.identity = List[MatrixIdentity](capacity=N_SESSION_ROLES)
        self.resident = List[Bool](capacity=N_SESSION_ROLES)
        for _ in range(N_SESSION_ROLES):
            self.identity.append(MatrixIdentity.empty())
            self.resident.append(False)
        self.uploads = 0
        self.reuses = 0
        self.evictions = 0

    def _check(self, role: Int) raises:
        if role < 0 or role >= N_SESSION_ROLES:
            raise Error("unknown residency role ", role)

    def is_resident(self, role: Int, identity: MatrixIdentity) -> Bool:
        if role < 0 or role >= N_SESSION_ROLES:
            return False
        if not self.resident[role]:
            return False
        return self.identity[role].matches(identity)

    def admit(mut self, role: Int, identity: MatrixIdentity) raises -> Bool:
        """Claim `role` for `identity`. True when the caller must upload,
        False when the device copy is already the right one.

        A mismatched identity evicts the old occupant first, so a role never
        reports resident with someone else's bytes behind it.
        """
        self._check(role)
        if identity.n_rows < 1 or identity.n_features < 1:
            raise Error("resident matrix needs positive rows and features")
        if self.is_resident(role, identity):
            self.reuses += 1
            return False
        if self.resident[role]:
            self.evictions += 1
        self.identity[role] = identity.copy()
        self.resident[role] = True
        self.uploads += 1
        return True

    def evict(mut self, role: Int) raises:
        self._check(role)
        if self.resident[role]:
            self.evictions += 1
        self.resident[role] = False
        self.identity[role] = MatrixIdentity.empty()

    def clear(mut self) raises:
        for role in range(N_SESSION_ROLES):
            self.evict(role)

    def resident_cells(self) -> Int:
        var total = 0
        for role in range(N_SESSION_ROLES):
            if self.resident[role]:
                total += self.identity[role].cells()
        return total


# ---------------------------------------------------------------------------
# Buffer pool bookkeeping
# ---------------------------------------------------------------------------

comptime SLOT_BINS = 0
comptime SLOT_LEAF = 1
comptime SLOT_GRAD = 2
comptime SLOT_HESS = 3
comptime SLOT_FEAT = 4
comptime SLOT_OUT = 5
comptime SLOT_PART = 6
comptime SLOT_STAGE = 7
comptime SLOT_HOST_OUT = 8
comptime SLOT_VALID_BINS = 9
comptime SLOT_VALID_SCORE = 10
comptime N_POOL_SLOTS = 11

comptime POOL_ALLOCATE = 0
comptime POOL_GROW = 1
comptime POOL_REUSE = 2


def pool_action_name(action: Int) -> String:
    if action == POOL_ALLOCATE:
        return String("allocate")
    if action == POOL_GROW:
        return String("grow")
    if action == POOL_REUSE:
        return String("reuse")
    return String("unknown")


struct PoolLedger(Copyable, Movable):
    """Allocate / grow / reuse decisions for one session's device buffers.

    Buffers grow to exactly what is asked for and never shrink. No headroom
    factor: a GPU buffer here is the binned matrix or a per-row array, so
    doubling to avoid a future reallocation can cost hundreds of megabytes
    to save an allocation that a fixed dataset never triggers. Never
    shrinking is what makes a second fit on the same or smaller data
    allocation-free, which is the case a per-estimator session exists for.

    The ledger holds sizes, not buffers. The session that owns the
    `DeviceContext` acts on the decision; keeping the policy separate is
    what lets it be tested without a device.
    """

    var capacity: List[Int]
    var elem_bytes: List[Int]
    var allocations: Int
    var growths: Int
    var reuses: Int

    def __init__(out self):
        self.capacity = List[Int](capacity=N_POOL_SLOTS)
        self.elem_bytes = List[Int](capacity=N_POOL_SLOTS)
        for _ in range(N_POOL_SLOTS):
            self.capacity.append(0)
            self.elem_bytes.append(0)
        self.allocations = 0
        self.growths = 0
        self.reuses = 0

    def _check(self, slot: Int) raises:
        if slot < 0 or slot >= N_POOL_SLOTS:
            raise Error("unknown pool slot ", slot)

    def request(
        mut self, slot: Int, n_elems: Int, elem_bytes: Int
    ) raises -> Int:
        """Decide how `slot` should serve `n_elems` elements of
        `elem_bytes` each, and record the decision.

        Returns `POOL_REUSE`, `POOL_GROW`, or `POOL_ALLOCATE`. A slot that
        changes element width is reallocated rather than reinterpreted: a
        device buffer is typed, and reusing an Int32 buffer as Float32 by
        byte count is exactly the kind of aliasing this ledger exists to
        make explicit.
        """
        self._check(slot)
        if n_elems < 1:
            raise Error("pool request must be for at least one element")
        if elem_bytes < 1:
            raise Error("pool request needs a positive element width")
        if self.capacity[slot] == 0:
            self.capacity[slot] = n_elems
            self.elem_bytes[slot] = elem_bytes
            self.allocations += 1
            return POOL_ALLOCATE
        if self.elem_bytes[slot] != elem_bytes:
            self.capacity[slot] = n_elems
            self.elem_bytes[slot] = elem_bytes
            self.growths += 1
            return POOL_GROW
        if self.capacity[slot] >= n_elems:
            self.reuses += 1
            return POOL_REUSE
        self.capacity[slot] = n_elems
        self.growths += 1
        return POOL_GROW

    def capacity_of(self, slot: Int) -> Int:
        if slot < 0 or slot >= N_POOL_SLOTS:
            return 0
        return self.capacity[slot]

    def resident_bytes(self) -> Int:
        var total = 0
        for slot in range(N_POOL_SLOTS):
            total += self.capacity[slot] * self.elem_bytes[slot]
        return total

    def release_all(mut self):
        """Drop every slot. The session calls this during teardown, after
        the queue is drained, so no buffer is released while device work
        still references it."""
        for slot in range(N_POOL_SLOTS):
            self.capacity[slot] = 0
            self.elem_bytes[slot] = 0


# ---------------------------------------------------------------------------
# Kernel warm-up registry
# ---------------------------------------------------------------------------

comptime KERNEL_HIST_ATOMIC = 0
comptime KERNEL_HIST_PARTIAL = 1
comptime KERNEL_HIST_REDUCE = 2
comptime KERNEL_PARTITION = 3
comptime KERNEL_PREDICT = 4
comptime N_KERNELS = 5


def kernel_name(kernel: Int) -> String:
    if kernel == KERNEL_HIST_ATOMIC:
        return String("hist_atomic")
    if kernel == KERNEL_HIST_PARTIAL:
        return String("hist_partial")
    if kernel == KERNEL_HIST_REDUCE:
        return String("hist_reduce")
    if kernel == KERNEL_PARTITION:
        return String("partition")
    if kernel == KERNEL_PREDICT:
        return String("predict")
    return String("unknown")


struct KernelRegistry(Copyable, Movable):
    """Which kernels this session has already launched once.

    `enqueue_function` compiles (or fetches from the driver cache) on first
    use, so the first launch of each kernel carries a cost the rest do not.
    The registry does not hold device function handles: binding those needs
    an API this module has not verified against the toolchain in use, and
    guessing at one would be worse than counting. What it does give is a
    correct place to attribute that first launch to `PHASE_COMPILE`, and a
    place for a later change to hang real handles off.
    """

    var warmed: List[Bool]
    var warm_count: Int

    def __init__(out self):
        self.warmed = List[Bool](capacity=N_KERNELS)
        for _ in range(N_KERNELS):
            self.warmed.append(False)
        self.warm_count = 0

    def _check(self, kernel: Int) raises:
        if kernel < 0 or kernel >= N_KERNELS:
            raise Error("unknown kernel id ", kernel)

    def needs_warm(self, kernel: Int) -> Bool:
        if kernel < 0 or kernel >= N_KERNELS:
            return False
        return not self.warmed[kernel]

    def mark_warm(mut self, kernel: Int) raises -> Bool:
        """Record that `kernel` has been launched. True the first time."""
        self._check(kernel)
        if self.warmed[kernel]:
            return False
        self.warmed[kernel] = True
        self.warm_count += 1
        return True

    def clear(mut self):
        for k in range(N_KERNELS):
            self.warmed[k] = False
        self.warm_count = 0


# ---------------------------------------------------------------------------
# Session lifecycle state machine
# ---------------------------------------------------------------------------

comptime STATE_NEW = 0
comptime STATE_OPEN = 1
comptime STATE_ROUND = 2
comptime STATE_TREE = 3
comptime STATE_CLOSED = 4
comptime N_STATES = 5


def state_name(state: Int) -> String:
    if state == STATE_NEW:
        return String("new")
    if state == STATE_OPEN:
        return String("open")
    if state == STATE_ROUND:
        return String("round")
    if state == STATE_TREE:
        return String("tree")
    if state == STATE_CLOSED:
        return String("closed")
    return String("unknown")


def can_transition(frm: Int, to: Int) -> Bool:
    """The legal moves of the session state machine.

    ```
    new --open--> open --begin_round--> round --begin_tree--> tree
                    ^                     |  ^                  |
                    |                     |  +----begin_tree----+
                    +------end_round------+  +-----end_tree-----+
    ```

    `begin_round` from `round` starts the next boosting round without
    passing through `open`, and `begin_tree` from `tree` starts the next
    class's tree inside one multiclass round. Every live state may close,
    and closing twice is legal so teardown is idempotent.
    """
    if frm < 0 or frm >= N_STATES or to < 0 or to >= N_STATES:
        return False
    if to == STATE_CLOSED:
        return True
    if frm == STATE_CLOSED:
        return False
    if frm == STATE_NEW:
        return to == STATE_OPEN
    if frm == STATE_OPEN:
        return to == STATE_ROUND
    if frm == STATE_ROUND:
        return to == STATE_ROUND or to == STATE_TREE or to == STATE_OPEN
    if frm == STATE_TREE:
        return to == STATE_TREE or to == STATE_ROUND
    return False


struct SessionLifecycle(Copyable, Movable):
    """The session's state plus the counts a trace wants.

    Separate from `GpuSession` so the whole lifecycle, including every
    illegal transition, is testable on a machine with no accelerator.
    """

    var state: Int
    var rounds: Int
    var trees: Int
    var opens: Int

    def __init__(out self):
        self.state = STATE_NEW
        self.rounds = 0
        self.trees = 0
        self.opens = 0

    def _move_to(mut self, to: Int) raises:
        if not can_transition(self.state, to):
            raise Error(
                "illegal GPU session transition: ",
                state_name(self.state),
                " -> ",
                state_name(to),
            )
        self.state = to

    def open(mut self) raises:
        self._move_to(STATE_OPEN)
        self.opens += 1

    def begin_round(mut self) raises:
        self._move_to(STATE_ROUND)
        self.rounds += 1

    def begin_tree(mut self) raises:
        self._move_to(STATE_TREE)
        self.trees += 1

    def end_tree(mut self) raises:
        if self.state != STATE_TREE:
            raise Error(
                "end_tree outside a tree: state is ", state_name(self.state)
            )
        self._move_to(STATE_ROUND)

    def end_round(mut self) raises:
        if self.state != STATE_ROUND:
            raise Error(
                "end_round outside a round: state is ",
                state_name(self.state),
            )
        self._move_to(STATE_OPEN)

    def close(mut self) raises:
        self._move_to(STATE_CLOSED)

    def is_closed(self) -> Bool:
        return self.state == STATE_CLOSED

    def is_live(self) -> Bool:
        return self.state != STATE_NEW and self.state != STATE_CLOSED

    def require_live(self) raises:
        if not self.is_live():
            raise Error(
                "GPU session is not usable here: state is ",
                state_name(self.state),
            )

    def require(self, state: Int) raises:
        if self.state != state:
            raise Error(
                "GPU session must be ",
                state_name(state),
                " here, but is ",
                state_name(self.state),
            )


# ---------------------------------------------------------------------------
# The seam the trainers announce their boundaries through
# ---------------------------------------------------------------------------


trait RoundLifecycle:
    """The four boundaries a boosting loop crosses, as the trainers in
    train_gpu.mojo announce them.

    `GpuSession` implements it by moving its state machine and counting; the
    no-op `NoLifecycle` implements it by counting only. A trainer is generic
    over this trait so one loop body serves both, which is what keeps the
    session an opt-in wrapper rather than a second trainer: the session-free
    entry point passes `NoLifecycle` and executes exactly the sequence of
    device calls it executed before this seam existed.

    The contract is the state machine in `can_transition`: a round is opened
    once and closed once, each tree inside it is opened and closed, and a
    multiclass round opens one tree per class without an intervening round
    boundary.
    """

    def begin_round(mut self) raises:
        """One boosting round starts (one tree, or one tree per class)."""
        ...

    def begin_tree(mut self) raises:
        """One tree starts growing."""
        ...

    def end_tree(mut self) raises:
        """The tree that was growing is finished."""
        ...

    def end_round(mut self) raises:
        """The round is finished, including its raw-score update."""
        ...


struct NoLifecycle(RoundLifecycle, Copyable, Movable):
    """The escape hatch: a lifecycle that owns no device and asserts nothing.

    This is what the session-free trainers pass, so a trainer generic over
    `RoundLifecycle` costs them two integer increments per round and cannot
    raise. It deliberately does not validate the transition order: a caller
    that wants the state machine's checks takes a `GpuSession`.
    """

    var rounds: Int
    var trees: Int

    def __init__(out self):
        self.rounds = 0
        self.trees = 0

    def begin_round(mut self) raises:
        self.rounds += 1

    def begin_tree(mut self) raises:
        self.trees += 1

    def end_tree(mut self) raises:
        pass

    def end_round(mut self) raises:
        pass


# ---------------------------------------------------------------------------
# The model of what the current pipeline enqueues
# ---------------------------------------------------------------------------
#
# These functions describe, in the tracker's vocabulary, exactly what
# histogram_gpu.mojo enqueues today. They are the shared source of truth for
# the audit below and for the lifecycle tests, so a claim about which
# synchronizations are removable is checked against one description of the
# pipeline rather than against prose in two places.


def model_upload_gradients(
    mut hazards: HazardTracker, mut staging: StagingRing
) raises -> Bool:
    """`stage_gradients` + `upload_staged`. True when the host had to wait.

    The host writes a staging slot, so the wait is a host-write hazard on
    `RES_STAGE`; the copy then reads that slot and writes the device
    gradient and hessian buffers.
    """
    var waited = False
    if staging.pending(hazards.required()):
        # A slot that has not retired always has an outstanding device read
        # against it, so this drain is the same event the ring is waiting
        # for. Counting the wait off the drain rather than off `pending`
        # keeps the two from ever disagreeing.
        if hazards.sync_for_host_write(RES_STAGE):
            staging.note_wait()
            waited = True
    else:
        hazards.elided += 1
    var slot = staging.acquire()
    hazards.note_device_read(RES_STAGE)
    hazards.note_device_write(RES_GRAD)
    hazards.note_device_write(RES_HESS)
    staging.mark_in_flight(slot, hazards.required())
    return waited


def model_begin_tree(mut hazards: HazardTracker, bagged: Bool) raises -> Bool:
    """`begin_tree`. True when the host had to wait.

    Unbagged it is an enqueued memset, which is device work and waits for
    nothing. Bagged it writes the leaf array from the host, which cannot
    happen while a histogram kernel is still reading it. A host write leaves
    no device work behind, so nothing is queued in that branch.
    """
    if not bagged:
        hazards.note_device_write(RES_LEAF)
        return False
    return hazards.sync_for_host_write(RES_LEAF)


def model_set_features(mut hazards: HazardTracker) raises -> Bool:
    """`set_features`. A host write to the active-feature array, which the
    histogram kernels read, so it waits for those reads and queues nothing
    of its own."""
    return hazards.sync_for_host_write(RES_FEAT)


def model_build_leaf(mut hazards: HazardTracker, tiled: Bool) raises -> Bool:
    """`enqueue_leaf` + `download_raw` + `histogram_from_host`. True when the
    host had to wait, which for this operation is always: the host reads the
    downloaded histogram, and the download is device work."""
    hazards.note_device_read(RES_BINS)
    hazards.note_device_read(RES_LEAF)
    hazards.note_device_read(RES_GRAD)
    hazards.note_device_read(RES_HESS)
    hazards.note_device_read(RES_FEAT)
    if tiled:
        hazards.note_device_write(RES_PART)
        hazards.note_device_read(RES_PART)
    hazards.note_device_write(RES_OUT)
    # The download reads the output buffer and writes pinned host memory.
    hazards.note_device_read(RES_OUT)
    hazards.note_device_write(RES_HOST_OUT)
    return hazards.sync_for_host_read(RES_HOST_OUT)


def model_apply_split(mut hazards: HazardTracker) raises:
    """`apply_split`. Pure device work: the partition kernel reads the bins
    and rewrites the leaf ids, and nothing on the host looks at either."""
    hazards.note_device_read(RES_BINS)
    hazards.note_device_write(RES_LEAF)


@fieldwise_init
struct SyncAudit(Copyable, Movable):
    """What one modeled boosting round costs in host synchronizations."""

    var required: Int
    """Drains the dependency model says the round genuinely needs."""

    var elided: Int
    """Checks that found no hazard: places today's code drains anyway."""

    var staging_waits: Int
    """Drains spent waiting for a staging slot to retire."""

    var unconditional: Int
    """Drains the current code performs for the same round, counted from
    the `ctx.synchronize()` calls in histogram_gpu.mojo: one in
    `stage_gradients` and one in `download_raw` per histogram built."""


def audit_round(
    n_builds: Int,
    tiled: Bool = True,
    staging_slots: Int = DEFAULT_STAGING_SLOTS,
    bagged: Bool = False,
    n_trees: Int = 1,
) raises -> SyncAudit:
    """Replay one boosting round through the dependency model.

    `n_builds` is the number of `build_leaf` calls per tree, which for a
    leaf-wise tree with L leaves is L: the root plus the smaller child of
    each of the L-1 splits. `n_trees` is 1 for single-output training and
    the class count for multiclass, where one round grows one tree per class
    off one builder.

    The round is: stage and upload the gradients once, then per tree set the
    active features, reset the leaf assignments, and alternate builds and
    splits. This is `train_gpu` plus `grow_tree_gpu`, with the split search
    and the tree bookkeeping left out because they touch no device state.
    """
    if n_builds < 1:
        raise Error("a round builds at least the root histogram")
    if n_trees < 1:
        raise Error("a round grows at least one tree")

    var hazards = HazardTracker()
    var staging = StagingRing(staging_slots)
    var waits = 0

    for _ in range(n_trees):
        if model_upload_gradients(hazards, staging):
            waits += 1
        _ = model_set_features(hazards)
        _ = model_begin_tree(hazards, bagged)
        for b in range(n_builds):
            _ = model_build_leaf(hazards, tiled)
            if b + 1 < n_builds:
                model_apply_split(hazards)

    # What the code does today, for the same work: one drain per
    # `stage_gradients` and one per `download_raw`.
    var unconditional = n_trees * (1 + n_builds)

    return SyncAudit(hazards.required(), hazards.elided, waits, unconditional)


# ---------------------------------------------------------------------------
# The session itself
# ---------------------------------------------------------------------------


struct GpuSession(RoundLifecycle, Movable):
    """One `DeviceContext` and everything that should outlive a single fit.

    `GpuHistogramBuilder` and `GpuPredictor` both take a session and borrow
    its context instead of opening their own, and the trainers in
    train_gpu.mojo take one through their session overloads, which is where
    the round and tree boundaries below come from. What is still missing is
    the owner: an estimator that holds the session across fits, so the pool
    and residency ledgers can actually skip an upload rather than only
    record that they could have. That, and routing the builder's drains
    through `sync_for_host_read` / `sync_for_host_write` instead of
    `ctx.synchronize()`, are described in `handoffs/apple_a5_runtime.md (deleted, recover with git log --all --diff-filter=D -- handoffs/apple_a5_runtime.md)`.

    Teardown is explicit. `close()` drains the queue once, releases the
    pooled slots, clears residency, and moves the lifecycle to `closed`; it
    is idempotent, so a caller may close in an error path and again in a
    normal one. Dropping a session without closing it still releases the
    context, but the release order is then the compiler's business rather
    than this module's, and no counter records the teardown, so `close()` is
    the supported path.
    """

    var ctx: DeviceContext
    var caps: DeviceCaps
    var counters: PhaseCounters
    var life: SessionLifecycle
    var hazards: HazardTracker
    var staging: StagingRing
    var residency: ResidencyLedger
    var pool: PoolLedger
    var kernels: KernelRegistry
    # The one-time costs, from initialization.mojo. `PhaseCounters` above
    # times a *running* session's phases; this times the phases a session is
    # opened to pay only once, and a session is the object that can say
    # truthfully whether they have been paid. `MOJOTREES_STARTUP_TRACE=1`
    # turns the timing on; the call counts are kept either way, which is
    # what `session_state` reads.
    var startup: StartupTrace
    var fits: FitLatency
    var warmup: WarmupPlan

    def __init__(out self, staging_slots: Int = 0) raises:
        """Open a device context and take the session's bookkeeping with it.

        `staging_slots` of 0 (the default) reads
        `MOJOTREES_GPU_STAGING_SLOTS`; an explicit positive value outranks
        the environment, matching how `strategy` outranks
        `MOJOTREES_GPU_HIST_STRATEGY` in gpu_tiling.mojo.

        Context creation and device discovery are the two startup phases a
        session actually performs, so they are timed here rather than
        estimated: `MOJOTREES_STARTUP_TRACE=1` reports them through
        `trace()`, and `session_state()` reports that they were paid whether
        or not the timing was on.
        """
        var counters = PhaseCounters.from_env()
        var startup = StartupTrace.from_env()
        var started = counters.clock()
        var slots = staging_slots if staging_slots > 0 else (
            env_staging_slots()
        )

        var opened = startup.clock()
        self.ctx = DeviceContext()
        startup.record(PHASE_CONTEXT_CREATE, opened)
        var probed = startup.clock()
        self.caps = query_device_caps(self.ctx)
        startup.record(PHASE_DEVICE_DISCOVERY, probed)

        self.counters = counters^
        self.life = SessionLifecycle()
        self.hazards = HazardTracker()
        self.staging = StagingRing(slots)
        self.residency = ResidencyLedger()
        self.pool = PoolLedger()
        self.kernels = KernelRegistry()
        self.startup = startup^
        self.fits = FitLatency()
        # `MOJOTREES_GPU_WARMUP` names the intent; the kernel ids are this
        # module's, which is why the plan takes ids rather than defining its
        # own. Nothing is created here: what the plan buys today is that the
        # first launch of a planned kernel is attributed to warm-up instead
        # of disappearing into the first round. Front-loading the creation
        # itself needs typed `DeviceFunction` fields on whichever struct
        # owns the context; see handoffs/performance_15_startup.md (deleted, recover with git log --all --diff-filter=D -- handoffs/performance_15_startup.md).
        self.warmup = WarmupPlan.from_env()
        if self.warmup.level >= WARMUP_TRAIN:
            _ = self.warmup.include(KERNEL_HIST_ATOMIC)
            _ = self.warmup.include(KERNEL_HIST_PARTIAL)
            _ = self.warmup.include(KERNEL_HIST_REDUCE)
            _ = self.warmup.include(KERNEL_PARTITION)
        if self.warmup.level >= WARMUP_ALL:
            _ = self.warmup.include(KERNEL_PREDICT)
        self.counters.record(PHASE_ALLOC, started)
        self.life.open()

    # -- instrumentation --------------------------------------------------

    def clock(self) -> Int:
        """Start a phase measurement (0 when tracing is off)."""
        return self.counters.clock()

    def record(mut self, phase: Int, started: Int) raises:
        """Close a phase measurement started by `clock()`."""
        self.counters.record(phase, started)

    def note_kernel(mut self, kernel: Int, started: Int) raises:
        """One kernel launch. The first launch of each kernel is attributed
        to `PHASE_COMPILE` as well, since that is where the compile or
        driver-cache lookup happens, and to the startup trace's
        `kernel_create`, which is the one-time half of the same cost. A
        kernel this session's warm-up plan named also records what its
        creation took, so `trace()` can say which kernel dominates a cold
        start rather than only that one does."""
        if self.kernels.mark_warm(kernel):
            self.counters.record(PHASE_COMPILE, started)
            self.startup.record(PHASE_KERNEL_CREATE, started)
            if self.warmup.is_planned(kernel):
                self.warmup.note_created(kernel, _elapsed_since(started))
        self.counters.record(PHASE_KERNEL, started)

    def note_transfer(mut self, started: Int) raises:
        self.counters.record(PHASE_TRANSFER, started)
        self.startup.record(PHASE_FIRST_TRANSFER, started)

    def note_alloc(mut self, started: Int) raises:
        """One device allocation, counted against both the running phase
        counters and the startup trace's `first_alloc`. `StartupTrace` keeps
        the first occurrence of a phase separately, so a caller that
        allocates every round still gets the cold number out of it."""
        self.counters.record(PHASE_ALLOC, started)
        self.startup.record(PHASE_FIRST_ALLOC, started)

    # -- startup and fit latency ------------------------------------------

    def begin_fit(self) -> Int:
        """Start a fit measurement. 0 when startup tracing is off, which is
        what makes an untraced fit pay no clock read at either end."""
        return self.startup.clock()

    def end_fit(mut self, started: Int) raises:
        """Close the measurement `begin_fit` started, against `first_fit`
        for this session's first fit and `warm_fit` for every one after it.
        Positional, not temporal: the first fit is the one that pays the
        one-time costs, however long the process sat idle first."""
        self.fits.note_fit(self.startup, started)

    def session_state(self) -> SessionState:
        """What this session has already paid, for `decide_device`.

        The device policy is pure by contract and takes this as data. A live
        session answers it truthfully, which no free function can: the
        context is open because this object opened it, and the kernels are
        ready exactly when the registry says every one has launched once.
        """
        var state = session_state_from_trace(self.startup, self.warmup.level)
        state.context_open = True
        state.kernels_ready = self.kernels.warm_count >= N_KERNELS
        return state^

    # -- synchronization boundaries ---------------------------------------

    def sync(mut self, reason: Int = SYNC_EXPLICIT) raises:
        """Drain the queue unconditionally and clear every hazard."""
        var started = self.counters.clock()
        self.ctx.synchronize()
        self.counters.record(PHASE_SYNC, started)
        self.hazards.sync(reason)

    def sync_for_host_read(mut self, resource: Int) raises -> Bool:
        """Drain only if the host cannot read `resource` yet."""
        if not self.hazards.host_read_hazard(resource):
            self.hazards.elided += 1
            return False
        self.sync(SYNC_HOST_READ)
        return True

    def sync_for_host_write(mut self, resource: Int) raises -> Bool:
        """Drain only if the host cannot overwrite `resource` yet."""
        if not self.hazards.host_write_hazard(resource):
            self.hazards.elided += 1
            return False
        self.sync(SYNC_HOST_WRITE)
        return True

    def note_device_read(mut self, resource: Int) raises:
        self.hazards.note_device_read(resource)

    def note_device_write(mut self, resource: Int) raises:
        self.hazards.note_device_write(resource)

    def acquire_staging(mut self) raises -> Int:
        """The staging slot to fill next, waiting only if its copy has not
        retired. This is the overlap: the host converts round i+1's
        gradients into another slot while round i's copy is still queued."""
        if self.staging.pending(self.hazards.required()):
            if self.sync_for_host_write(RES_STAGE):
                self.staging.note_wait()
        else:
            self.hazards.elided += 1
        return self.staging.acquire()

    def staged(mut self, slot: Int) raises:
        """A copy out of `slot` has been enqueued."""
        self.staging.mark_in_flight(slot, self.hazards.required())
        self.hazards.note_device_read(RES_STAGE)

    # -- residency and pooling --------------------------------------------

    def admit_matrix(
        mut self, role: Int, identity: MatrixIdentity
    ) raises -> Bool:
        """True when `role`'s matrix has to be uploaded, False when the
        device already holds exactly it."""
        self.life.require_live()
        return self.residency.admit(role, identity)

    def request_buffer(
        mut self, slot: Int, n_elems: Int, elem_bytes: Int
    ) raises -> Int:
        """Pool decision for one buffer role. `POOL_ALLOCATE` and
        `POOL_GROW` mean the caller creates a buffer and brackets the
        creation with `clock()` / `record(PHASE_ALLOC, ...)`; `POOL_REUSE`
        means it keeps the one it has and there is nothing to time.

        A grow replaces a buffer the device may still be reading, so the
        caller must drain first: freeing is a host-side write to everything
        the old buffer aliased.
        """
        self.life.require_live()
        return self.pool.request(slot, n_elems, elem_bytes)

    # -- lifecycle --------------------------------------------------------

    def begin_round(mut self) raises:
        self.life.begin_round()

    def begin_tree(mut self) raises:
        self.life.begin_tree()

    def end_tree(mut self) raises:
        self.life.end_tree()

    def end_round(mut self) raises:
        self.life.end_round()

    def close(mut self) raises:
        """Deterministic teardown, idempotent.

        Order matters and is fixed here: drain the queue, then release the
        pool and residency, then move to `closed`. Releasing before draining
        would free memory the device is still reading.
        """
        if self.life.is_closed():
            return
        var started = self.counters.clock()
        if self.hazards.any_pending():
            self.ctx.synchronize()
            self.hazards.sync(SYNC_TEARDOWN)
        self.pool.release_all()
        self.residency.clear()
        self.kernels.clear()
        self.life.close()
        self.counters.record(PHASE_CLEANUP, started)

    def trace(self) -> String:
        """Everything the counters know, as `name value` lines. Intended for
        a benchmark or a debug print, not for parsing by anything shipped."""
        var out = self.counters.report()
        out += self.hazards.report()
        out += "staging.slots " + String(self.staging.n_slots) + "\n"
        out += "staging.waits " + String(self.staging.waits) + "\n"
        out += "pool.allocations " + String(self.pool.allocations) + "\n"
        out += "pool.growths " + String(self.pool.growths) + "\n"
        out += "pool.reuses " + String(self.pool.reuses) + "\n"
        out += "residency.uploads " + String(self.residency.uploads) + "\n"
        out += "residency.reuses " + String(self.residency.reuses) + "\n"
        out += "kernels.warmed " + String(self.kernels.warm_count) + "\n"
        out += "session.rounds " + String(self.life.rounds) + "\n"
        out += "session.trees " + String(self.life.trees) + "\n"
        out += "session.state " + state_name(self.life.state) + "\n"
        out += self.startup.report()
        out += self.warmup.report()
        out += "session.paid " + self.session_state().describe() + "\n"
        return out
