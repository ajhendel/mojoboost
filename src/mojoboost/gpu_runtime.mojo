"""Persistent GPU runtime: one session per estimator, and the dependency
model that says which host synchronizations are actually required.

Today every GPU entry point in train_gpu.mojo constructs a fresh
`GpuHistogramBuilder`, and that constructor opens a `DeviceContext`,
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
- `HazardTracker` is the dependency model. It is the piece that matters.
- `PhaseCounters` attributes time to compile, allocation, transfer, kernel,
  synchronization, and cleanup phases.

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

Nothing in this file removes a synchronization from histogram_gpu.mojo or
train_gpu.mojo, and nothing here has been measured. It is the model and the
instrumentation that a removal has to be argued from, plus
`audit_round`, which replays the current per-round operation sequence
through the model so the argument is executable rather than prose. The
handoff (`handoffs/apple_a5_runtime.md`) lists which of today's
synchronizations the model marks removable and what has to be proven first.

Environment contract, matching the `MOJOBOOST_` convention in parallel.mojo:

- `MOJOBOOST_GPU_TRACE=1` turns on the phase counters. Off by default so an
  untraced session pays no clock reads.
- `MOJOBOOST_GPU_STAGING_SLOTS`: staging ring depth, default
  `DEFAULT_STAGING_SLOTS`, clamped to `[1, MAX_STAGING_SLOTS]`. `1`
  reproduces today's single-buffer behavior exactly.
"""

from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from .gpu_tiling import DeviceCaps, query_device_caps
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


def phase_name(phase: Int) -> String:
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
        """Counters configured by `MOJOBOOST_GPU_TRACE`."""
        return PhaseCounters(getenv("MOJOBOOST_GPU_TRACE") == "1")

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
            out += phase_name(p) + " " + String(self.calls[p])
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
    whole-queue drain and there is no finer-grained wait in use. A model
    with per-event waits would clear less and elide more; this one is
    deliberately the conservative version of the argument.
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
    """`MOJOBOOST_GPU_STAGING_SLOTS`, clamped to a usable ring depth."""
    var n = _env_int("MOJOBOOST_GPU_STAGING_SLOTS", DEFAULT_STAGING_SLOTS)
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
# Residency: which matrices the device already holds
# ---------------------------------------------------------------------------

comptime ROLE_TRAIN = 0
comptime ROLE_VALID = 1
comptime N_ROLES = 2


def role_name(role: Int) -> String:
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
        self.identity = List[MatrixIdentity](capacity=N_ROLES)
        self.resident = List[Bool](capacity=N_ROLES)
        for _ in range(N_ROLES):
            self.identity.append(MatrixIdentity.empty())
            self.resident.append(False)
        self.uploads = 0
        self.reuses = 0
        self.evictions = 0

    def _check(self, role: Int) raises:
        if role < 0 or role >= N_ROLES:
            raise Error("unknown residency role ", role)

    def is_resident(self, role: Int, identity: MatrixIdentity) -> Bool:
        if role < 0 or role >= N_ROLES:
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
        self.identity[role] = identity
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
        for role in range(N_ROLES):
            self.evict(role)

    def resident_cells(self) -> Int:
        var total = 0
        for role in range(N_ROLES):
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


struct GpuSession(Movable):
    """One `DeviceContext` and everything that should outlive a single fit.

    Nothing else in the tree constructs this yet: wiring it into the GPU
    trainers is central integration, described in
    `handoffs/apple_a5_runtime.md`. The intended shape is that an estimator
    holds one session, `GpuHistogramBuilder` borrows the session's context
    and buffers instead of opening its own, and every drain in the builder
    goes through `sync_for_host_read` / `sync_for_host_write` here rather
    than calling `ctx.synchronize()` directly.

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

    def __init__(out self, staging_slots: Int = 0) raises:
        """Open a device context and take the session's bookkeeping with it.

        `staging_slots` of 0 (the default) reads
        `MOJOBOOST_GPU_STAGING_SLOTS`; an explicit positive value outranks
        the environment, matching how `strategy` outranks
        `MOJOBOOST_GPU_HIST_STRATEGY` in gpu_tiling.mojo.
        """
        var counters = PhaseCounters.from_env()
        var started = counters.clock()
        var slots = staging_slots if staging_slots > 0 else (
            env_staging_slots()
        )
        self.ctx = DeviceContext()
        self.caps = query_device_caps(self.ctx)
        self.counters = counters^
        self.life = SessionLifecycle()
        self.hazards = HazardTracker()
        self.staging = StagingRing(slots)
        self.residency = ResidencyLedger()
        self.pool = PoolLedger()
        self.kernels = KernelRegistry()
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
        driver-cache lookup happens."""
        if self.kernels.mark_warm(kernel):
            self.counters.record(PHASE_COMPILE, started)
        self.counters.record(PHASE_KERNEL, started)

    def note_transfer(mut self, started: Int) raises:
        self.counters.record(PHASE_TRANSFER, started)

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
        return out
