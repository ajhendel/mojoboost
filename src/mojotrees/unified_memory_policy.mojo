"""Which allocation route a device buffer uses, and what has to be true
before it may use anything other than an explicit copy.

A pure policy layer, in the shape of `apple_gpu_policy.mojo`: buffer role,
platform facts, and recorded evidence in, one resolved route plus its
synchronization obligations out. It opens no device, allocates nothing,
copies nothing, and holds no pointer, so every decision here is checkable on
a machine with no accelerator.

The default is `copy_staged` for every role, on every platform, in every
build. Nothing in this module can currently return anything else without an
explicit environment request *and* an explicit acknowledgment that the
request is unproven, because `EvidenceLedger.installed()` is empty: no route
in this repository has been compiled, run, or measured. See
`docs/APPLE_UNIFIED_MEMORY.md`.

Why this is a module and not three lines in the histogram builder
----------------------------------------------------------------

Three reasons, and the third is the one that motivated the file.

1. The decision is per *role*, not per session. The binned matrix, the
   per-round gradients, and the per-node histogram output have different
   directions, different lifetimes, and different owners, and a route that
   is structurally impossible for one is unremarkable for another. A single
   session-wide flag would have to be resolved against the strictest role
   and would then never turn on.

2. The decision carries obligations, not just a pointer. A route that
   issues no copy changes *when the host may next write the buffer*, from
   "when the copy retired" to "when every kernel that read it retired".
   That is a synchronization contract, and shipping the pointer choice
   without it is how a shared route becomes a silent correctness bug that
   only appears under load. `sync_contract` is that half of the answer and
   it is returned with every decision.

3. It has to be Mojo. Whether a transfer can be omitted is a statement about
   pointer lifetime and device queue ordering. Python cannot see either, so
   Python cannot be the layer that decides; the most a Python surface may
   ever do with this module is print the decision it was handed.

Two claims, only one of which is free
-------------------------------------

Kept separate here and everywhere else in the repository.

**Claim 1**, unified physical memory: the CPU and GPU address the same DRAM.
True on Apple silicon by construction, costs nothing to say, implies nothing
about any API.

**Claim 2**, no duplication for our data: the buffer mojotrees fills on the
host is the same physical bytes the kernel reads, with no staging
allocation, no blit, and no migration on first device touch.

Claim 1 makes Claim 2 possible and does not make it true. This module never
infers one from the other: `unified_memory` is a necessary condition for
considering a shared route and is never sufficient to select one.

What this module deliberately refuses to encode
-----------------------------------------------

- **No probing.** Whether the runtime accepts a host pointer as a kernel
  argument is a fact about a toolchain, discovered by compiling and running
  `bench/apple/unified_memory.mojo`, and recorded in `RouteEvidence`. A
  policy layer that tried to find out itself would need a device, and would
  turn a testable function into an untestable one.
- **No timing.** Nothing here reads a clock or holds a measurement. A route
  is enabled by recorded evidence at `TRANSFER_EVIDENCE_TRAINER`, which is an
  end-to-end training result, not by a microbenchmark this module could be
  handed.
- **No inference from `no_copy_issued`.** That flag means mojotrees enqueued
  no copy. It says nothing about runtime page migration, a driver-side blit,
  or physical duplication, and it sits three rungs below the top of the
  evidence ladder for exactly that reason.
- **No per-chip constants.** The platform input is two booleans and an API
  code, supplied by the caller from what the device reported.

Environment contract, matching the `MOJOTREES_` convention:

- `MOJOTREES_GPU_TRANSFER`: `staged` (the default), `direct`, `mapped`,
  `host_direct`, or `wrapped`. An unparsable value raises rather than
  silently falling back to the default, following `device.mojo`, where an
  explicit `gpu` request raises instead of quietly becoming a CPU run.
- `MOJOTREES_GPU_TRANSFER_UNPROVEN=1`: acknowledge that the requested route
  has no installed evidence and run it anyway. Required for anything but
  `staged` until the ledger fills, and deliberately loud. It exists because
  the top rung of the evidence ladder is an end-to-end training measurement,
  which cannot be taken without running the trainer on the route: without an
  override the gate would be unsatisfiable by construction. A decision made
  under it carries `ack_unproven`, and a benchmark that reports a number
  taken that way must report the flag beside it.
"""

from std.os import getenv


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# The routes `bench/apple/unified_memory.mojo` knows how to talk about. The
# driver imports these rather than defining its own, so a route cannot be
# named one thing in the experiment and another in the policy.
comptime ROUTE_COPY_STAGED = 0
comptime ROUTE_COPY_DIRECT = 1
comptime ROUTE_MAP_WRITE = 2
comptime ROUTE_HOST_DIRECT = 3
comptime ROUTE_WRAPPED_HOST = 4
comptime N_ROUTES = 5

# The dependable one. Every field of this module exists to make selecting
# anything else difficult, and this is what it falls back to.
comptime DEFAULT_ROUTE = ROUTE_COPY_STAGED


def route_name(route: Int) -> String:
    if route == ROUTE_COPY_STAGED:
        return String("copy_staged")
    if route == ROUTE_COPY_DIRECT:
        return String("copy_direct")
    if route == ROUTE_MAP_WRITE:
        return String("map_write")
    if route == ROUTE_HOST_DIRECT:
        return String("host_direct")
    if route == ROUTE_WRAPPED_HOST:
        return String("wrapped_host_buffer")
    return String("unknown")


def parse_route(name: String) raises -> Int:
    """Route code for a public route name.

    Names are canonical lowercase, as in `device.mojo`. An unknown name
    raises: a typo in a performance knob must not resolve to a different
    performance path.
    """
    if name == "staged":
        return ROUTE_COPY_STAGED
    if name == "direct":
        return ROUTE_COPY_DIRECT
    if name == "mapped":
        return ROUTE_MAP_WRITE
    if name == "host_direct":
        return ROUTE_HOST_DIRECT
    if name == "wrapped":
        return ROUTE_WRAPPED_HOST
    raise Error(
        "unknown transfer route '",
        name,
        "'; expected 'staged', 'direct', 'mapped', 'host_direct', or"
        " 'wrapped'",
    )


# Route statuses, shared with the driver so a run's output and this module's
# vocabulary cannot drift apart. `unsupported` means the route was compiled
# and raised at runtime; `not_probed` means it was never compiled. In Mojo a
# missing method is a compile error rather than a catchable one, so those two
# are genuinely different facts and neither is ever smoothed into `ok`.
comptime TRANSFER_STATUS_OK = 0
comptime TRANSFER_STATUS_UNSUPPORTED = 1
comptime TRANSFER_STATUS_WRONG = 2
comptime TRANSFER_STATUS_NOT_PROBED = 3


def status_name(status: Int) -> String:
    if status == TRANSFER_STATUS_OK:
        return String("ok")
    if status == TRANSFER_STATUS_UNSUPPORTED:
        return String("unsupported")
    if status == TRANSFER_STATUS_WRONG:
        return String("wrong")
    if status == TRANSFER_STATUS_NOT_PROBED:
        return String("not_probed")
    return String("unknown")


# ---------------------------------------------------------------------------
# Buffer roles
# ---------------------------------------------------------------------------

# The buffers a training or scoring session actually owns, named by what they
# carry rather than by which struct field holds them. The route question is
# asked once per role.
comptime ROLE_BINS = 0
"""The binned matrix, `histogram_gpu.mojo`'s `bins_dev`. Host to device,
uploaded once per session, read by every histogram and every partition."""

comptime ROLE_GRAD = 1
"""Per-round gradients, `grad_dev`, staged through `stage_g`."""

comptime ROLE_HESS = 2
"""Per-round hessians, `hess_dev`, staged through `stage_h`."""

comptime ROLE_HIST_OUT = 3
"""The fixed-point histogram planes, `out_dev`. Device to host, downloaded
once per node, which is the most frequent transfer in a round."""

comptime ROLE_ROW_SEED = 4
"""The bagged root row list, `GpuActiveRows.stage_rows`. Host to device, once
per tree, and only when bagging is on."""

comptime ROLE_VALID_BINS = 5
"""A validation matrix held device-resident for scoring,
`gpu_predict.mojo`'s `valid_bins_dev`.

Uploaded by `upload_validation` straight from the caller's `data.bins` into a
buffer sized exactly to that matrix, so it holds two copies and no pinned
staging one, exactly like `ROLE_BINS`. It is a separate role because it is
resident for a different span: the training matrix lives for a fit, this lives
for as long as the estimator scores against the same validation set, and both
are resident at once."""

comptime ROLE_PREDICT_OUT = 6
"""Prediction scores coming back, `gpu_predict.mojo`'s `host_out`. Written by
the predict kernel with plain stores, not atomics."""

comptime ROLE_BATCH_BINS = 7
"""A batch of rows being scored, `gpu_predict.mojo`'s `bins_dev`, staged
through the pinned `stage_bins`.

The only role whose bytes already sit in a runtime allocation the session owns
before they reach the device, which is what makes it the only bins-shaped role
a shared route is structurally able to serve. See `structural_support`."""

comptime N_TRANSFER_ROLES = 8


def transfer_role_name(role: Int) -> String:
    if role == ROLE_BINS:
        return String("bins")
    if role == ROLE_GRAD:
        return String("grad")
    if role == ROLE_HESS:
        return String("hess")
    if role == ROLE_HIST_OUT:
        return String("hist_out")
    if role == ROLE_ROW_SEED:
        return String("row_seed")
    if role == ROLE_VALID_BINS:
        return String("valid_bins")
    if role == ROLE_PREDICT_OUT:
        return String("predict_out")
    if role == ROLE_BATCH_BINS:
        return String("batch_bins")
    return String("unknown")


comptime DIR_HOST_TO_DEVICE = 0
comptime DIR_DEVICE_TO_HOST = 1


def role_direction(role: Int) raises -> Int:
    """Which way the bytes move for this role.

    Direction is not decoration. The four routes the driver implements are
    all host-to-device delivery: they answer "how does the payload get
    somewhere the kernel can read it". A device-to-host role asks the
    opposite question, and the only shared answer available for it is that
    the kernel writes its result straight into host-visible memory, which is
    a different mechanism with a different failure mode (see
    `BLOCK_DEVICE_WRITTEN_ATOMICS`).
    """
    if role == ROLE_HIST_OUT or role == ROLE_PREDICT_OUT:
        return DIR_DEVICE_TO_HOST
    if role >= 0 and role < N_TRANSFER_ROLES:
        return DIR_HOST_TO_DEVICE
    raise Error("unknown buffer role ", role)


# ---------------------------------------------------------------------------
# Structural eligibility
# ---------------------------------------------------------------------------

comptime ELIGIBLE = 0

comptime BLOCK_SOURCE_NOT_DEVICE_VISIBLE = 1
"""The host bytes live in an allocation the runtime did not make, so there is
no pointer to hand a kernel without first copying into one that is."""

comptime BLOCK_LIFETIME_NOT_OWNED = 2
"""The buffer this session would publish is owned by a caller whose lifetime
this session does not control."""

comptime BLOCK_DEVICE_WRITTEN_ATOMICS = 3
"""The device may write this buffer with global integer atomics, and whether
those are coherent against host-visible memory is unverified on every backend
here. A wrong answer is a silently wrong histogram rather than a raise.

"May" is exact and is why the block is unconditional. `STRATEGY_ATOMIC` folds
every threadgroup's partial into the histogram with `Atomic.fetch_add` on the
output buffer; `STRATEGY_TILED` writes partials without atomics and reduces
them with plain stores. Which one runs is decided per node from that node's
row count (`GpuActiveRows.range_tiling`), so a buffer that is safe under one
strategy and unsafe under the other cannot be given a route that is resolved
once per session."""

comptime BLOCK_NOT_UNIFIED_MEMORY = 4
"""A route that skips the copy on the assumption that both processors reach
the same DRAM, asked for on a device that reports separate memory."""

comptime BLOCK_ROUTE_NOT_PROBED = 5
"""The route has never been compiled here, so nothing is known about it."""

comptime BLOCK_NO_EVIDENCE = 6
"""Structurally possible, but the evidence ladder is not satisfied."""

comptime BLOCK_DIRECTION = 7
"""The route is defined for the other transfer direction."""


def transfer_block_name(reason: Int) -> String:
    if reason == ELIGIBLE:
        return String("eligible")
    if reason == BLOCK_SOURCE_NOT_DEVICE_VISIBLE:
        return String("source_not_device_visible")
    if reason == BLOCK_LIFETIME_NOT_OWNED:
        return String("lifetime_not_owned")
    if reason == BLOCK_DEVICE_WRITTEN_ATOMICS:
        return String("device_written_atomics")
    if reason == BLOCK_NOT_UNIFIED_MEMORY:
        return String("not_unified_memory")
    if reason == BLOCK_ROUTE_NOT_PROBED:
        return String("route_not_probed")
    if reason == BLOCK_NO_EVIDENCE:
        return String("no_evidence")
    if reason == BLOCK_DIRECTION:
        return String("wrong_direction")
    return String("unknown")


def publishes_by_copy(route: Int) raises -> Bool:
    """Whether this route has a publish step that moves the payload.

    Deliberately *not* the same question as the driver's
    `copy_bytes_issued_total`, and the two are named differently so they are
    never read as the same fact:

    - the driver counts bytes handed to `enqueue_copy`, so `map_write` counts
      zero there, because leaving a mapped block is not a call this library
      makes;
    - this function counts `map_write` as publishing by copy anyway, because
      that block exit may be a full upload and its cost is unknown. Assuming
      it is free is exactly the error this module exists to prevent.

    False here means the route has no publish step at all, which is true of
    `host_direct` and the unprobed wrapped route and of nothing else. Even
    then it is a statement about what *this library* does. The runtime remains
    free to migrate pages or run a blit of its own, which is why
    `TRANSFER_EVIDENCE_NO_COPY_ISSUED` is a low rung on the ladder rather than the
    answer.
    """
    if route == ROUTE_COPY_STAGED or route == ROUTE_COPY_DIRECT:
        return True
    if route == ROUTE_MAP_WRITE:
        return True
    if route == ROUTE_HOST_DIRECT or route == ROUTE_WRAPPED_HOST:
        return False
    raise Error("unknown transfer route ", route)


def structural_support(role: Int, route: Int) raises -> Int:
    """Whether `role` could take `route` at all, before any evidence or
    platform question is asked.

    This is the part that no measurement can change, because it is about how
    mojotrees's own data structures are laid out today. Four findings are
    encoded here and each is worth stating in full, because each one bounds
    what the whole experiment could be worth:

    - `ROLE_BINS` cannot take a shared route as the data is currently owned.
      `BinnedMatrix.bins` is a plain heap `List[UInt8]` built by
      `binning.mojo` and owned by the caller; `GpuHistogramBuilder` copies it
      to the device and keeps no reference to it. To hand a kernel that host
      pointer, the bytes would first have to be copied into a runtime
      allocation, which is the copy the route was supposed to remove, leaving
      two host-side copies instead of one host and one device. Removing the
      duplication for the largest buffer in the system therefore requires
      `binning.mojo` to bin *into* a device-visible allocation. That is a
      change to another module's data ownership, not a flag.

    - `ROLE_HIST_OUT` is written by global integer atomics under one of the
      two shipped accumulation strategies, and which strategy runs is a
      per-node decision. Whether such an atomic is coherent against
      host-visible memory is unverified on Metal, CUDA, and HIP alike here,
      and the failure mode is a wrong histogram rather than a raise, so the
      role is blocked until the driver's `out_host_direct` route answers it.

    - `ROLE_VALID_BINS` inherits `ROLE_BINS`'s ownership problem exactly:
      `upload_validation` copies straight out of the caller's `data.bins`
      into a buffer sized to that matrix, so it holds two copies and the
      shared route has the same nothing to hand a kernel. It is worth
      remembering that this one is resident *alongside* the training matrix
      rather than instead of it.

    - `ROLE_BATCH_BINS` is the exception, and it is the only role in the
      system where a shared route is structurally available today.
      `upload_bins` already writes the batch into a pinned `stage_bins` that
      the predictor owns, so the pointer a kernel would need already exists.
      The staging copy is there because `bins_dev` is sized to the
      high-water batch rather than to this batch and `enqueue_copy` moves a
      whole buffer, so copying from the caller's exactly-sized list would
      read past its end. A shared route would keep the staging write and drop
      both the device buffer and the copy, taking the batch path from three
      resident copies to two. It is still gated on evidence like everything
      else.
    """
    if role < 0 or role >= N_TRANSFER_ROLES:
        raise Error("unknown buffer role ", role)
    if route < 0 or route >= N_ROUTES:
        raise Error("unknown transfer route ", route)

    # The default is always structurally available: it is what ships.
    if route == ROUTE_COPY_STAGED or route == ROUTE_COPY_DIRECT:
        return ELIGIBLE

    if route == ROUTE_WRAPPED_HOST:
        return BLOCK_ROUTE_NOT_PROBED

    var direction = role_direction(role)
    if direction == DIR_DEVICE_TO_HOST:
        if route == ROUTE_MAP_WRITE:
            return BLOCK_DIRECTION
        # host_direct in this direction means the kernel writes its result
        # into host-visible memory instead of into a device buffer that is
        # then copied back.
        if role == ROLE_HIST_OUT:
            return BLOCK_DEVICE_WRITTEN_ATOMICS
        return ELIGIBLE

    # ROLE_BATCH_BINS is deliberately not in this test: its bytes already
    # live in a pinned buffer the predictor owns, so it is the one
    # bins-shaped role whose source is device-visible.
    if role == ROLE_BINS or role == ROLE_VALID_BINS:
        if route == ROUTE_HOST_DIRECT:
            return BLOCK_SOURCE_NOT_DEVICE_VISIBLE
        # `map_write` needs no host allocation of ours, so the ownership
        # problem does not apply: the bytes are written through the device
        # buffer's own mapping. It stays gated on evidence like everything
        # else, and note that today's constructor copies straight out of the
        # caller's list, so this route would move that write rather than
        # remove it.
        return ELIGIBLE

    return ELIGIBLE


# ---------------------------------------------------------------------------
# The evidence ladder
# ---------------------------------------------------------------------------

# Rungs, in the order they must be climbed. A route's level is the highest
# rung with every rung below it satisfied, so a run that produced a Metal
# trace but no checksum scores `TRANSFER_EVIDENCE_NONE`, which is correct: an
# unverified route's trace is not evidence about a route that works.
comptime TRANSFER_EVIDENCE_NONE = 0
comptime TRANSFER_EVIDENCE_COMPILED = 1
"""The route compiles in this Mojo version. Distinguishes a real route from
one that only exists in a document."""

comptime TRANSFER_EVIDENCE_CHECKSUM = 2
"""The device read or wrote the correct bytes, round after round, against a
host reference. Necessary for any timing to mean anything."""

comptime TRANSFER_EVIDENCE_NO_COPY_ISSUED = 3
"""mojotrees enqueued no copy. A fact about this library, not about the
hardware or the runtime."""

comptime TRANSFER_EVIDENCE_NO_SECOND_ALLOCATION = 4
"""An external capture shows one payload-sized resident allocation rather
than two, with no swap or compressor movement during the run."""

comptime TRANSFER_EVIDENCE_NO_BLIT = 5
"""A Metal System Trace shows no blit encoder between the host write and the
kernel. This is the rung that separates 'we issued no copy' from 'no copy
happened', and no timing number substitutes for it."""

comptime TRANSFER_EVIDENCE_TRAINER = 6
"""`bench/bench_train_gpu.mojo` on the route beats the same benchmark on
`copy_staged`, on the same machine, repeated, with identical models out."""

comptime ENABLE_LEVEL = TRANSFER_EVIDENCE_TRAINER
"""What a route needs before it may be selected without an explicit
acknowledgment.

Deliberately the top rung. The transfer is one part of a boosting round.
The end-to-end GPU training measurement in this repository (M4,
`bench/results/apple_m4_large_scaling_2026-08-14.md`) has the device
trainer 2.6x to 3.3x ahead of the CPU trainer and still dominated by
per-node launches and synchronizations rather than by transfers, and the
driver's first run (UM-2026-08-15-M4-01) put the staged copy at 75 to 85
GB/s, under a millisecond for a 1,000,000 x 50 binned matrix. A driver win
is therefore a reason to run the trainer, never a reason to change the
trainer.
"""


def evidence_name(level: Int) -> String:
    if level == TRANSFER_EVIDENCE_NONE:
        return String("none")
    if level == TRANSFER_EVIDENCE_COMPILED:
        return String("compiled")
    if level == TRANSFER_EVIDENCE_CHECKSUM:
        return String("checksum")
    if level == TRANSFER_EVIDENCE_NO_COPY_ISSUED:
        return String("no_copy_issued")
    if level == TRANSFER_EVIDENCE_NO_SECOND_ALLOCATION:
        return String("no_second_allocation")
    if level == TRANSFER_EVIDENCE_NO_BLIT:
        return String("no_blit")
    if level == TRANSFER_EVIDENCE_TRAINER:
        return String("trainer")
    return String("unknown")


@fieldwise_init
struct RouteEvidence(Copyable, Movable):
    """What has actually been established about one route.

    Every field is a recorded observation, and every one of them is False in
    `EvidenceLedger.installed()` today. Filling one in means pasting a run
    into `docs/APPLE_UNIFIED_MEMORY.md`'s record section and naming that
    record here; a flag set with no record behind it is the failure this
    struct exists to make visible.
    """

    var route: Int
    var compiled: Bool
    var checksum_ok: Bool
    var no_copy_issued: Bool
    var single_resident_allocation: Bool
    var no_blit_in_trace: Bool
    var trainer_confirmed: Bool
    var record: String
    """Identifier of the run in the document's record section, empty when
    there is none. A route whose level is above `TRANSFER_EVIDENCE_NONE` with an empty
    record is a bug, which `audit` reports."""

    @staticmethod
    def none(route: Int) -> RouteEvidence:
        """No evidence at all, which is the state of every route here."""
        return RouteEvidence(
            route, False, False, False, False, False, False, String("")
        )

    def level(self) -> Int:
        """The highest rung with every rung below it satisfied.

        Monotone on purpose. Evidence about a route that has not been shown
        correct is not evidence, so a missing rung truncates everything above
        it rather than being averaged away.
        """
        if not self.compiled:
            return TRANSFER_EVIDENCE_NONE
        if not self.checksum_ok:
            return TRANSFER_EVIDENCE_COMPILED
        if not self.no_copy_issued:
            return TRANSFER_EVIDENCE_CHECKSUM
        if not self.single_resident_allocation:
            return TRANSFER_EVIDENCE_NO_COPY_ISSUED
        if not self.no_blit_in_trace:
            return TRANSFER_EVIDENCE_NO_SECOND_ALLOCATION
        if not self.trainer_confirmed:
            return TRANSFER_EVIDENCE_NO_BLIT
        return TRANSFER_EVIDENCE_TRAINER

    def enables(self) -> Bool:
        return self.level() >= ENABLE_LEVEL

    def audit(self) -> String:
        """Empty when the evidence is internally consistent, otherwise what
        is wrong with it."""
        if self.level() > TRANSFER_EVIDENCE_NONE and self.record.byte_length() == 0:
            return String(
                "route ",
                route_name(self.route),
                " claims evidence at ",
                evidence_name(self.level()),
                " with no record identifier",
            )
        return String("")


struct EvidenceLedger(Copyable, Movable):
    """One `RouteEvidence` per route.

    Constructed empty. `installed()` is the repository's real state and is
    the only constructor a shipped call site should use; the fieldwise form
    exists so a test can hand the resolver a hypothetical ledger and check
    that the gate opens where it should, without anyone editing the shipped
    one to make a test pass.
    """

    var routes: List[RouteEvidence]

    def __init__(out self):
        self.routes = List[RouteEvidence](capacity=N_ROUTES)
        for route in range(N_ROUTES):
            self.routes.append(RouteEvidence.none(route))

    @staticmethod
    def installed() -> EvidenceLedger:
        """What this repository has established about these routes.

        One run so far, UM-2026-08-15-M4-01 (Apple M4, 16 GB, macOS 26.5.2,
        Mojo 1.0.0, MAX 26.5.0; `bench/results/apple_m4_unified_memory_2026-08-15.md`
        and the record section of `docs/APPLE_UNIFIED_MEMORY.md`). Every rung
        below carries that identifier, and every rung it does not set stays
        False: no route reached `no_second_allocation`, `no_blit`, or
        `trainer`, so nothing here opens the `ENABLE_LEVEL` gate and the
        shipped default is unchanged.

        - `copy_staged`, `copy_direct`: compiled, checksum correct in every
          launch. They issue copies by construction, so `checksum` is their
          ceiling.
        - `map_write`: compiled, checksum correct, no copy issued (Claim 1.5).
          Also measured 45% to 60% slower per round than the staged copy in
          `rewrite` mode, which the ladder does not encode and the record
          does; a rung is evidence of correctness, not of a win.
        - `host_direct`: compiled, and `wrong` in every launch (the kernel
          reads zeros through the host-buffer pointer), so `compiled` is
          all it earns.
        - `wrapped_host_buffer`: not probed, `none`.

        Each later change here is a change to a shipped default that a
        reviewer can see in a diff, and must name the run that earned it.
        """
        var record = String("UM-2026-08-15-M4-01")
        var ledger = EvidenceLedger()
        try:
            ledger.set_route(
                RouteEvidence(
                    ROUTE_COPY_STAGED,
                    True,  # compiled
                    True,  # checksum_ok
                    False,  # no_copy_issued
                    False,  # single_resident_allocation
                    False,  # no_blit_in_trace
                    False,  # trainer_confirmed
                    record,
                )
            )
            ledger.set_route(
                RouteEvidence(
                    ROUTE_COPY_DIRECT,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    record,
                )
            )
            ledger.set_route(
                RouteEvidence(
                    ROUTE_MAP_WRITE,
                    True,
                    True,
                    True,
                    False,
                    False,
                    False,
                    record,
                )
            )
            ledger.set_route(
                RouteEvidence(
                    ROUTE_HOST_DIRECT,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    record,
                )
            )
        except:
            # The route constants above are the module's own and in range;
            # `set_route` cannot raise on them.
            pass
        return ledger^

    def for_route(self, route: Int) raises -> RouteEvidence:
        if route < 0 or route >= N_ROUTES:
            raise Error("unknown transfer route ", route)
        return self.routes[route].copy()

    def level_of(self, route: Int) raises -> Int:
        return self.for_route(route).level()

    def set_route(mut self, evidence: RouteEvidence) raises:
        if evidence.route < 0 or evidence.route >= N_ROUTES:
            raise Error("unknown transfer route ", evidence.route)
        self.routes[evidence.route] = evidence.copy()

    def audit(self) -> String:
        """Every internal inconsistency in the ledger, one per line."""
        var out = String("")
        for route in range(N_ROUTES):
            var line = self.routes[route].audit()
            if line.byte_length() != 0:
                out += line + "\n"
        return out


# ---------------------------------------------------------------------------
# Synchronization obligations
# ---------------------------------------------------------------------------

comptime RETIRE_ON_COPY = 0
"""The host may refill the buffer once the copy that reads it has retired."""

comptime RETIRE_ON_KERNEL = 1
"""The host may refill the buffer only once every kernel that reads it has
retired, which on a shared route is every kernel of the whole tree."""


def retire_event_name(event: Int) -> String:
    if event == RETIRE_ON_COPY:
        return String("copy")
    if event == RETIRE_ON_KERNEL:
        return String("kernel")
    return String("unknown")


@fieldwise_init
struct SyncContract(Copyable, Movable):
    """What a route obliges its owner to do, in the vocabulary
    `gpu_runtime.mojo`'s `HazardTracker` already uses.

    This is the half of a route choice that is not a pointer. It is returned
    with every decision because the two must not be separable: handing a call
    site the shared pointer without the obligations below is how a route that
    benchmarked well becomes a race that only shows up on a busy machine.
    """

    var route: Int

    var publish_is_copy: Bool
    """Whether publishing enqueues a copy. False means `upload_staged` has
    nothing to do on this route."""

    var retire_event: Int
    """What the host must wait for before writing the buffer it fills next.

    This is *not* the same question as `publish_is_copy`, and collapsing the
    two is the modeling error this struct exists to avoid. A copy route fills
    a staging buffer whose only device reader is the copy, so it is free again
    as soon as that copy retires, early in the round, even while kernels are
    still running. Every other route fills a buffer the *kernels* read, so it
    is not free until those retire, at the end of the round. `map_write` is
    the case that proves they are different questions: its publish may well
    cost a transfer, and its next write still has to wait for the kernels."""

    var host_rewrite_needs_drain: Bool
    """Whether refilling that buffer requires a queue drain when device work
    is outstanding. True on every route: the existing `ctx.synchronize()` at
    the top of `stage_gradients` is not removable by a route change, and on
    every route but the copy ones it becomes stronger, not weaker."""

    var note_read_on_publish: Bool
    """Record a device read of the host staging buffer when the copy that
    reads it is enqueued. True only on the copy routes, which are the only
    ones where a copy reads a host buffer of ours."""

    var note_read_on_each_launch: Bool
    """Record a device read at every kernel launch that takes the buffer's
    pointer. This is the flag that changes the dependency model: on a shared
    route the gradient buffer is read by every node's histogram kernel, not
    once per round by one copy, and on `map_write` the mapped device buffer is
    likewise read by every launch between one mapping and the next."""

    var staging_ring_applies: Bool
    """Whether `StagingRing`'s overlap argument still holds.

    True only on the copy routes, and this is the finding most likely to be
    missed. The ring assumes a slot is free once the copy reading it has
    retired, which happens early in a round. On a shared route there is no
    copy and the slot is read by every histogram kernel in the tree, so it is
    not free until the round drains: a two-slot ring buys nothing and, worse,
    would report an overlap that is not there. On `map_write` there is no
    staging slot to rotate at all, since the host writes through the device
    buffer's own mapping.

    **True here is a statement about the dependency model and not a promise of
    overlap on Metal.** `docs/GPU_PORTABILITY.md` section 6.1 establishes, by
    disassembly, that `enqueue_copy` on Metal drains the whole queue before it
    memcpys, so on that backend no copy is ever still in flight and a slot is
    free the instant the copy returns. Section 6.1.1, 2026-08-16, draws the
    consequence: a second slot cannot be the thing that avoids a wait there,
    and `MOJOTREES_GPU_STAGING_SLOTS` above 1 buys nothing on Metal. The flag
    is still right, because it is written against queue ordering rather than
    against one backend, and it is what keeps the ring correct where the copy
    really is asynchronous. What it does not do is predict a saving here.
    """


def sync_contract(route: Int) raises -> SyncContract:
    """The obligations `route` imposes.

    Written as an explicit branch per route family rather than derived from
    one boolean. An earlier version of this function computed all five answers
    from `publishes_by_copy`, which put `map_write` in the copy routes'
    retirement class: its publish may cost a transfer, so that boolean is
    True, but the buffer it writes next is the one the *kernels* read, so its
    wait is a kernel wait. Two different questions had been collapsed onto one
    flag, which is the same error as calling the driver's `enqueues_copy` and
    this module's `publishes_by_copy` the same fact.

    Note that `host_rewrite_needs_drain` is True on every route including the
    default. A route change cannot remove a synchronization that exists
    because the host is about to overwrite memory the device may still be
    reading; it can only change *which event* the wait is for, which is what
    `retire_event` carries. Lane A5's proposal to elide synchronizations and
    any non-default route interact here and must be reviewed together.
    """
    if route < 0 or route >= N_ROUTES:
        raise Error("unknown transfer route ", route)

    if route == ROUTE_COPY_STAGED or route == ROUTE_COPY_DIRECT:
        # Host fills a staging buffer; the only device reader of it is the
        # copy, so the slot frees early and the staging ring's overlap is
        # real.
        return SyncContract(
            route, True, RETIRE_ON_COPY, True, True, False, True
        )

    if route == ROUTE_MAP_WRITE:
        # Host writes through the device buffer's own mapping. There is no
        # staging buffer to rotate, the publish may still cost a transfer,
        # and the next mapping has to wait for every kernel that read that
        # buffer.
        return SyncContract(
            route, True, RETIRE_ON_KERNEL, True, False, True, False
        )

    # host_direct and the unprobed wrapped route: no publish at all, and the
    # kernels read the host buffer directly, so it is live until they retire.
    return SyncContract(
        route, False, RETIRE_ON_KERNEL, True, False, True, False
    )


# ---------------------------------------------------------------------------
# Footprint accounting
# ---------------------------------------------------------------------------


@fieldwise_init
struct Footprint(Copyable, Movable):
    """Bytes a role costs on a route, as *requested*, never as resident.

    The distinction is the whole subject. `host_bytes` and `device_bytes` are
    what mojotrees asked the runtime for. What the system committed, what it
    compressed, what it migrated, and whether two of these numbers name the
    same physical pages are not visible from inside the process, and this
    struct does not pretend otherwise: `resident_bytes_unknown` is always
    True and exists so a caller printing these numbers has to print that too.
    """

    var role: Int
    var route: Int
    var host_bytes: Int
    var device_bytes: Int
    var published_bytes_per_use: Int
    var uses_per_session: Int
    var resident_bytes_unknown: Bool

    def requested_bytes(self) -> Int:
        return self.host_bytes + self.device_bytes

    def published_bytes(self) -> Int:
        """Bytes this role publishes over a session, by whatever mechanism
        the route uses. Not the same as the driver's
        `copy_bytes_issued_total`, which counts only `enqueue_copy` bytes and
        therefore counts zero for `map_write`. See `publishes_by_copy`."""
        return self.published_bytes_per_use * self.uses_per_session


def role_element_bytes(role: Int) raises -> Int:
    if (
        role == ROLE_BINS
        or role == ROLE_VALID_BINS
        or role == ROLE_BATCH_BINS
    ):
        return 1
    if role == ROLE_ROW_SEED:
        return 4
    if role < 0 or role >= N_TRANSFER_ROLES:
        raise Error("unknown buffer role ", role)
    return 4


def role_elements(
    role: Int, n_rows: Int, n_features: Int, n_bins: Int
) raises -> Int:
    """How many elements a role's buffer holds for one dataset shape."""
    if n_rows < 1 or n_features < 1 or n_bins < 1:
        raise Error("footprint needs positive rows, features, and bins")
    if (
        role == ROLE_BINS
        or role == ROLE_VALID_BINS
        or role == ROLE_BATCH_BINS
    ):
        return n_rows * n_features
    if role == ROLE_GRAD or role == ROLE_HESS or role == ROLE_ROW_SEED:
        return n_rows
    if role == ROLE_HIST_OUT:
        # [grad | hess | count] planes, as histogram_gpu.mojo lays them out.
        return 3 * n_features * n_bins
    if role == ROLE_PREDICT_OUT:
        return n_rows
    raise Error("unknown buffer role ", role)


def role_uses(role: Int, n_rounds: Int, n_nodes_per_round: Int) raises -> Int:
    """How many times a session moves this role's bytes.

    The asymmetry here is the reason `ROLE_HIST_OUT` deserves attention that
    `ROLE_BINS` does not: the binned matrix crosses once per session and the
    histogram crosses once per node, so a 200-round fit growing 31 leaves per
    tree moves the (small) histogram about six thousand times and the (large)
    matrix once.
    """
    if n_rounds < 1 or n_nodes_per_round < 1:
        raise Error("footprint needs at least one round and one node")
    if role == ROLE_BINS or role == ROLE_VALID_BINS:
        return 1
    if role == ROLE_BATCH_BINS:
        # Once per scored batch, and a caller scoring in batches calls this
        # once per batch rather than passing a batch count.
        return 1
    if role == ROLE_GRAD or role == ROLE_HESS:
        return n_rounds
    if role == ROLE_ROW_SEED:
        # Once per tree rather than once per round. They differ under
        # multiclass, which grows one tree per class per round, so a
        # multiclass caller passes the tree count as `n_rounds` here.
        return n_rounds
    if role == ROLE_HIST_OUT:
        return n_rounds * n_nodes_per_round
    if role == ROLE_PREDICT_OUT:
        return 1
    raise Error("unknown buffer role ", role)


def role_footprint(
    role: Int,
    route: Int,
    n_rows: Int,
    n_features: Int,
    n_bins: Int,
    n_rounds: Int = 1,
    n_nodes_per_round: Int = 1,
) raises -> Footprint:
    """What one role costs on one route for one dataset shape.

    The host and device sides are counted separately and are never summed
    into a "saving": on a unified pool the interesting question is whether
    two allocations are two sets of physical pages, and that is precisely
    what this cannot see.
    """
    var elems = role_elements(role, n_rows, n_features, n_bins)
    var width = role_element_bytes(role)
    var bytes = elems * width
    var uses = role_uses(role, n_rounds, n_nodes_per_round)
    var copies = publishes_by_copy(route)

    var host_bytes = bytes
    var device_bytes = bytes
    if route == ROUTE_MAP_WRITE:
        # Written through the device buffer's own mapping: no host allocation
        # of ours, one device allocation.
        host_bytes = 0
    elif not copies:
        # One runtime host allocation, handed to the kernel directly.
        device_bytes = 0

    var published = bytes if copies else 0
    return Footprint(
        role,
        route,
        host_bytes,
        device_bytes,
        published,
        uses,
        True,
    )


@fieldwise_init
struct SourceDuplication(Copyable, Movable):
    """How many host-side copies of the caller's data a path holds, before
    any device allocation is counted.

    Mostly this is accounting a route choice does *not* change, and it is
    listed because it is larger than anything a route choice does change.
    Training and validation both hold the caller's `List[UInt8]` and one
    device buffer. Batch scoring holds three copies of the same matrix, and
    that is the one a route choice could reduce, for the reason
    `structural_support` gives.

    On a unified pool every one of these copies is the same DRAM, which is
    why counting them is the memory question and the route is not.
    """

    var caller_copies: Int
    var pinned_copies: Int
    var device_copies: Int

    def total(self) -> Int:
        return self.caller_copies + self.pinned_copies + self.device_copies

    def bytes(self, n_rows: Int, n_features: Int) -> Int:
        return self.total() * n_rows * n_features


def training_bins_duplication() -> SourceDuplication:
    """`GpuHistogramBuilder.__init__`: caller's list plus one device buffer.

    The upload copies straight out of `data.bins`, so there is no pinned
    staging copy on this path.
    """
    return SourceDuplication(1, 0, 1)


def validation_bins_duplication() -> SourceDuplication:
    """`GpuPredictor.upload_validation`: caller's list plus one device buffer,
    the same shape as training.

    The validation buffer is sized to exactly the validation matrix, so the
    upload reads the caller's list directly and needs no staging copy. This
    one is resident *alongside* the training matrix, not instead of it, so a
    fit that scores a held-out set holds four copies of two matrices.
    """
    return SourceDuplication(1, 0, 1)


def batch_scoring_bins_duplication() -> SourceDuplication:
    """`GpuPredictor.upload_bins`: caller's list, the pinned `stage_bins`, and
    the device buffer.

    The third copy is not an oversight. `bins_dev` is sized to the high-water
    batch rather than to this batch, and `enqueue_copy` moves a whole buffer,
    so copying from the caller's exactly-sized list would read past its end.
    Removing the copy means either a sub-range copy (no such API is verified
    here) or sizing the device buffer to each batch, which trades the copy for
    an allocation per batch. This is also the one path where a shared route
    could remove it instead, because the staged bytes are already in a
    runtime allocation the predictor owns.
    """
    return SourceDuplication(1, 1, 1)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@fieldwise_init
struct RouteDecision(Copyable, Movable):
    """One resolved route for one role, with everything a call site needs to
    act on it and everything a report needs to be honest about it."""

    var role: Int
    var requested: Int
    var selected: Int
    var reason: Int
    """`ELIGIBLE` when the request was honored, otherwise why it was not and
    the default was used instead."""

    var ack_unproven: Bool
    """True when the selection rests on `MOJOTREES_GPU_TRANSFER_UNPROVEN=1`
    rather than on evidence. Any number measured under this must be reported
    with it."""

    var evidence_level: Int
    var contract: SyncContract

    def is_default(self) -> Bool:
        return self.selected == DEFAULT_ROUTE


def route_block_reason(
    role: Int,
    requested: Int,
    unified_memory: Bool,
    evidence: EvidenceLedger,
    ack_unproven: Bool = False,
) raises -> Int:
    """Why `role` may not use `requested`, or `ELIGIBLE` when it may.

    The single place the gate is decided. `explain_route` reports this answer
    and `resolve_route` raises on it, so the two cannot disagree about what is
    allowed, which two independent copies of the rules eventually would.

    Order of questions, each of which can only narrow the answer:

    1. The default is always honored, without evidence, on any platform. It
       is what ships and what every other route is measured against.
    2. Structural support (`structural_support`). A route that cannot work
       given how mojotrees owns its data is refused here, and no measurement
       can change that answer.
    3. Platform. A copy-skipping route is refused on a device that does not
       report unified memory.
    4. Evidence. Below `ENABLE_LEVEL` the request is refused unless
       `ack_unproven` is set.

    Raises only on an unknown role or route, which is a programming error
    rather than a policy answer.
    """
    if role < 0 or role >= N_TRANSFER_ROLES:
        raise Error("unknown buffer role ", role)
    if requested < 0 or requested >= N_ROUTES:
        raise Error("unknown transfer route ", requested)
    if requested == DEFAULT_ROUTE:
        return ELIGIBLE

    var support = structural_support(role, requested)
    if support != ELIGIBLE:
        return support

    if not unified_memory and not publishes_by_copy(requested):
        return BLOCK_NOT_UNIFIED_MEMORY

    if evidence.level_of(requested) < ENABLE_LEVEL and not ack_unproven:
        return BLOCK_NO_EVIDENCE

    return ELIGIBLE


def explain_route(
    role: Int,
    requested: Int,
    unified_memory: Bool,
    evidence: EvidenceLedger,
    ack_unproven: Bool = False,
) raises -> RouteDecision:
    """The decision `resolve_route` would make, without raising on a refusal.

    A refused request comes back as the default with `reason` set to why, so a
    diagnostic, a trace line, or a benchmark header can report the gate
    without catching an exception and without a second copy of the rules.

    Nothing that allocates should use this. A call site that asked for a route
    and silently got a different one is the failure `resolve_route` exists to
    prevent; this is for the code that only wants to say what the gate would
    do.
    """
    var level = evidence.level_of(requested)
    var reason = route_block_reason(
        role, requested, unified_memory, evidence, ack_unproven
    )
    if reason != ELIGIBLE:
        return RouteDecision(
            role,
            requested,
            DEFAULT_ROUTE,
            reason,
            False,
            level,
            sync_contract(DEFAULT_ROUTE),
        )
    return RouteDecision(
        role,
        requested,
        requested,
        ELIGIBLE,
        requested != DEFAULT_ROUTE and level < ENABLE_LEVEL,
        level,
        sync_contract(requested),
    )


def resolve_route(
    role: Int,
    requested: Int,
    unified_memory: Bool,
    evidence: EvidenceLedger,
    ack_unproven: Bool = False,
) raises -> RouteDecision:
    """Resolve one role's route, raising when the request cannot be honored.

    The gate itself is `route_block_reason`; this adds the message and the
    refusal. A refused *explicit* request raises rather than quietly returning
    the default, following `device.mojo`, where `device='gpu'` on a machine
    without an accelerator raises instead of silently training on the CPU. A
    caller that wants the survivable behavior asks for the default; a caller
    that only wants to report the gate calls `explain_route`.

    Each refusal names what would have to change, because "route refused" with
    no reason is the kind of message that gets worked around rather than read.
    """
    var reason = route_block_reason(
        role, requested, unified_memory, evidence, ack_unproven
    )
    if reason == BLOCK_NO_EVIDENCE:
        raise Error(
            "transfer route '",
            route_name(requested),
            "' has evidence at '",
            evidence_name(evidence.level_of(requested)),
            "' and needs '",
            evidence_name(ENABLE_LEVEL),
            "'; set MOJOTREES_GPU_TRANSFER_UNPROVEN=1 to run it anyway and"
            " report that flag with any number it produces",
        )
    if reason == BLOCK_NOT_UNIFIED_MEMORY:
        raise Error(
            "transfer route '",
            route_name(requested),
            "' skips the copy, but this device does not report unified"
            " memory; use 'staged'",
        )
    if reason != ELIGIBLE:
        raise Error(
            "transfer route '",
            route_name(requested),
            "' cannot serve the '",
            transfer_role_name(role),
            "' buffer: ",
            transfer_block_name(reason),
            "; see docs/APPLE_UNIFIED_MEMORY.md",
        )
    return explain_route(
        role, requested, unified_memory, evidence, ack_unproven
    )


def env_requested_route() raises -> Int:
    """`MOJOTREES_GPU_TRANSFER`, defaulting to the staged copy.

    Unset is the default. Set to something unparsable raises: a misspelled
    performance knob must not resolve to a different performance path, and
    the same reasoning as `parse_route` applies.
    """
    var s = getenv("MOJOTREES_GPU_TRANSFER")
    if s.byte_length() == 0:
        return DEFAULT_ROUTE
    return parse_route(s)


def env_ack_unproven() -> Bool:
    """`MOJOTREES_GPU_TRANSFER_UNPROVEN=1`."""
    return getenv("MOJOTREES_GPU_TRANSFER_UNPROVEN") == "1"


def resolve_from_env(role: Int, unified_memory: Bool) raises -> RouteDecision:
    """The shipped entry point: environment request against installed
    evidence.

    Returns the default for every role today, because `installed()` is empty
    and the environment variable is unset in every context this repository
    controls.
    """
    return resolve_route(
        role,
        env_requested_route(),
        unified_memory,
        EvidenceLedger.installed(),
        env_ack_unproven(),
    )


# ---------------------------------------------------------------------------
# The session plan: one resolved route per role, as one value
# ---------------------------------------------------------------------------


struct SessionMemoryPlan(Copyable, Movable):
    """Every role's route for one session, resolved once and carried as data.

    This is the shape the rest of the system consumes. `resolve_from_env` and
    `resolve_route` answer for one role, which is what an allocation site
    needs; a *decision report* needs all of them at once, because "which
    transfer strategy is this run using" has eight answers and printing one of
    them is how a reader concludes the wrong thing about the other seven.

    Built with `explain_route`, not `resolve_route`, and the difference is
    deliberate. A plan is a description: a role that cannot honor the
    environment request appears here as the default with `reason` saying why,
    so one structurally-blocked role does not erase the plan for the other
    seven. The refusal still happens, at the allocation site, where
    `resolve_from_env` raises. A report that explains and an allocator that
    refuses are not two policies: `route_block_reason` is the single gate and
    both go through it.

    `unified_memory` is carried because it is an input to every one of those
    answers and because a reader looking at a plan of all-`copy_staged` needs
    to know whether that is the platform's doing or the ledger's.
    """

    var unified_memory: Bool
    var requested: Int
    """The route `MOJOTREES_GPU_TRANSFER` asked for, honored or not."""

    var ack_unproven: Bool
    """`MOJOTREES_GPU_TRANSFER_UNPROVEN=1` was set."""

    var decisions: List[RouteDecision]
    """One entry per role, indexed by the `ROLE_*` code."""

    def __init__(
        out self,
        unified_memory: Bool,
        requested: Int,
        ack_unproven: Bool,
        var decisions: List[RouteDecision],
    ):
        self.unified_memory = unified_memory
        self.requested = requested
        self.ack_unproven = ack_unproven
        self.decisions = decisions^

    @staticmethod
    def staged(unified_memory: Bool = False) raises -> SessionMemoryPlan:
        """The shipped plan: the staged copy for every role.

        Reads no environment, so it is the conservative value a caller uses
        when it has nothing to resolve against. It is also what
        `plan_session_routes` returns in every context this repository
        controls, because `EvidenceLedger.installed()` is empty and
        `MOJOTREES_GPU_TRANSFER` is unset.

        The obligations come from `sync_contract`, not from a literal repeated
        here: two spellings of the staged contract is exactly the drift this
        module was written to prevent.
        """
        var contract = sync_contract(DEFAULT_ROUTE)
        var decisions = List[RouteDecision](capacity=N_TRANSFER_ROLES)
        for role in range(N_TRANSFER_ROLES):
            decisions.append(
                RouteDecision(
                    role,
                    DEFAULT_ROUTE,
                    DEFAULT_ROUTE,
                    ELIGIBLE,
                    False,
                    TRANSFER_EVIDENCE_NONE,
                    contract.copy(),
                )
            )
        return SessionMemoryPlan(
            unified_memory, DEFAULT_ROUTE, False, decisions^
        )

    def for_role(self, role: Int) raises -> RouteDecision:
        if role < 0 or role >= len(self.decisions):
            raise Error("unknown buffer role ", role)
        return self.decisions[role].copy()

    def all_default(self) -> Bool:
        """True when every role resolved to `copy_staged`, which is the state
        this repository ships in and the only state any measurement in it was
        taken under."""
        for i in range(len(self.decisions)):
            if not self.decisions[i].is_default():
                return False
        return True

    def honored_count(self) -> Int:
        """Roles that got the route the environment asked for."""
        var n = 0
        for i in range(len(self.decisions)):
            if self.decisions[i].reason == ELIGIBLE:
                n += 1
        return n

    def any_unproven(self) -> Bool:
        """True when any role's route rests on the acknowledgment rather than
        on evidence. Any number measured under this must be reported with
        it."""
        for i in range(len(self.decisions)):
            if self.decisions[i].ack_unproven:
                return True
        return False

    def needs_kernel_retirement(self) -> Bool:
        """True when any role's buffer is live until the kernels that read it
        retire, rather than until a copy retires.

        The one field of the plan that changes what a *caller* must do rather
        than what it allocates: on a kernel-retirement route the staging ring's
        overlap argument does not hold and the host's next write to that buffer
        has to wait for the round, not for the upload. See `SyncContract`.
        """
        for i in range(len(self.decisions)):
            if self.decisions[i].contract.retire_event == RETIRE_ON_KERNEL:
                return True
        return False

    def report(self) raises -> String:
        """The plan as lines, one per role:

        ```
        transfer.<role> <requested> <selected> <reason> <evidence> <retire_on> <ack>
        ```

        Seven space-separated fields, roles always in declaration order and
        always all present, so a consumer can index rather than search. The
        shape `StartupTrace.report()` uses, for the same reason.
        """
        var out = String("")
        for role in range(len(self.decisions)):
            var d = self.decisions[role].copy()
            out += "transfer." + transfer_role_name(d.role)
            out += " " + route_name(d.requested)
            out += " " + route_name(d.selected)
            out += " " + transfer_block_name(d.reason)
            out += " " + evidence_name(d.evidence_level)
            out += " " + retire_event_name(d.contract.retire_event)
            if d.ack_unproven:
                out += " 1\n"
            else:
                out += " 0\n"
        out += "transfer.unified_memory "
        if self.unified_memory:
            out += "1\n"
        else:
            out += "0\n"
        out += "transfer.requested " + route_name(self.requested) + "\n"
        out += "transfer.honored " + String(self.honored_count()) + "\n"
        out += "transfer.roles " + String(len(self.decisions)) + "\n"
        return out^


def plan_session_routes(unified_memory: Bool) raises -> SessionMemoryPlan:
    """Resolve every role's route for one session, from the environment
    request against the installed evidence.

    Raises only for an unparsable `MOJOTREES_GPU_TRANSFER`, which is
    `env_requested_route`'s rule and not a per-role one: a misspelled
    performance knob must not resolve to a different performance path, and it
    must not resolve to eight of them either.

    Returns the all-`copy_staged` plan in every context this repository
    controls, because `EvidenceLedger.installed()` is empty and the variable
    is unset.
    """
    var requested = env_requested_route()
    var ack = env_ack_unproven()
    var ledger = EvidenceLedger.installed()
    var decisions = List[RouteDecision](capacity=N_TRANSFER_ROLES)
    for role in range(N_TRANSFER_ROLES):
        decisions.append(
            explain_route(role, requested, unified_memory, ledger, ack)
        )
    return SessionMemoryPlan(unified_memory, requested, ack, decisions^)


def describe_session_plan(plan: SessionMemoryPlan) raises -> String:
    """One line for a trace or a bug report, in the shape
    `describe_decision` uses for a single role."""
    var out = String(
        "transfer requested=",
        route_name(plan.requested),
        " honored=",
        plan.honored_count(),
        "/",
        len(plan.decisions),
        " unified_memory=",
    )
    if plan.unified_memory:
        out += "true"
    else:
        out += "false"
    if plan.all_default():
        out += " all_default=true"
    else:
        out += " all_default=false"
    if plan.any_unproven():
        out += " ack_unproven=1"
    return out^


def describe_decision(decision: RouteDecision) -> String:
    """One line for a trace or a bug report.

    Names the refusal reason and the acknowledgment explicitly. A decision
    taken under the acknowledgment is not a measurement of a supported
    configuration, and a decision that fell back to the default for a reason
    is not the same event as one that was asked for."""
    var out = String(
        "role=",
        transfer_role_name(decision.role),
        " requested=",
        route_name(decision.requested),
        " selected=",
        route_name(decision.selected),
        " evidence=",
        evidence_name(decision.evidence_level),
        " retire_on=",
        retire_event_name(decision.contract.retire_event),
    )
    if decision.reason != ELIGIBLE:
        out += " blocked=" + transfer_block_name(decision.reason)
    if decision.ack_unproven:
        out += " ack_unproven=1"
    return out
