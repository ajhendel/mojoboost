"""Persistent GPU runtime: lifecycle, hazards, pooling, residency.

Everything asserted here is device-free by construction. `GpuSession` owns a
`DeviceContext` and cannot be built without an accelerator, but every rule it
enforces lives in a separate value type, so the state machine, the
dependency model, the buffer pool, and the residency ledger are all driven
directly from the host. That is deliberate: a session bug is a bug in when
things are allowed to happen, and the way to test "when" exhaustively,
including every illegal move, is to test it without hardware.

The headline assertion is `test_audit_round_matches_the_current_pipeline`,
which replays a boosting round through the dependency model and compares the
drains the model requires against the drains histogram_gpu.mojo performs
today. It is a claim about the model, not a measurement: nothing here times
anything or asserts that any code is faster.
"""

from std.os import setenv
from std.sys import has_accelerator
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import bin_equal_width
from mojotrees.gpu_runtime import GpuSession
from mojotrees.histogram_gpu import GpuHistogramBuilder

from mojotrees.gpu_runtime import (
    DEFAULT_STAGING_SLOTS,
    KERNEL_HIST_ATOMIC,
    KERNEL_PARTITION,
    MAX_STAGING_SLOTS,
    N_KERNELS,
    N_PHASES,
    N_POOL_SLOTS,
    N_ROLES,
    N_STATES,
    PHASE_ALLOC,
    PHASE_CLEANUP,
    PHASE_COMPILE,
    PHASE_KERNEL,
    PHASE_SYNC,
    PHASE_TRANSFER,
    POOL_ALLOCATE,
    POOL_GROW,
    POOL_REUSE,
    RES_BINS,
    RES_FEAT,
    RES_LEAF,
    RES_OUT,
    RES_STAGE,
    ROLE_TRAIN,
    ROLE_VALID,
    STATE_CLOSED,
    STATE_NEW,
    STATE_OPEN,
    STATE_ROUND,
    STATE_TREE,
    SYNC_HOST_READ,
    SYNC_HOST_WRITE,
    SYNC_TEARDOWN,
    HazardTracker,
    KernelRegistry,
    MatrixIdentity,
    PhaseCounters,
    PoolLedger,
    ResidencyLedger,
    SessionLifecycle,
    StagingRing,
    audit_round,
    bins_fingerprint,
    can_transition,
    env_staging_slots,
    kernel_name,
    model_apply_split,
    model_begin_tree,
    model_build_leaf,
    model_set_features,
    model_upload_gradients,
    phase_name,
    pool_action_name,
    resource_name,
    role_name,
    state_name,
    sync_reason_name,
)


# ---------------------------------------------------------------------------
# Lifecycle state machine
# ---------------------------------------------------------------------------


def test_lifecycle_follows_the_training_shape() raises:
    """The exact sequence a multiclass GPU fit walks: open once, one round
    per boosting iteration, one tree per class inside a round, back to open
    at the end, then closed."""
    var life = SessionLifecycle()
    assert_equal(life.state, STATE_NEW)
    assert_true(not life.is_live())

    life.open()
    assert_equal(life.state, STATE_OPEN)
    assert_true(life.is_live())

    life.begin_round()
    life.begin_tree()
    # Next class in the same round: tree -> tree, no round boundary.
    life.begin_tree()
    life.end_tree()
    # Next boosting round without returning to open.
    life.begin_round()
    life.begin_tree()
    life.end_tree()
    life.end_round()

    assert_equal(life.state, STATE_OPEN)
    assert_equal(life.rounds, 2)
    assert_equal(life.trees, 3)
    assert_equal(life.opens, 1)

    life.close()
    assert_true(life.is_closed())
    assert_true(not life.is_live())


def test_close_is_idempotent() raises:
    """Teardown has to be safe to call from an error path and again from the
    normal one."""
    var life = SessionLifecycle()
    life.open()
    life.close()
    life.close()
    life.close()
    assert_equal(life.state, STATE_CLOSED)


def test_a_session_may_close_from_any_state() raises:
    var states = [STATE_NEW, STATE_OPEN, STATE_ROUND, STATE_TREE]
    for i in range(len(states)):
        assert_true(can_transition(states[i], STATE_CLOSED))


def test_illegal_transitions_raise() raises:
    # Rounds cannot start before the session is open.
    var fresh = SessionLifecycle()
    var raised = False
    try:
        fresh.begin_round()
    except:
        raised = True
    assert_true(raised)

    # A tree needs a round around it.
    var open_only = SessionLifecycle()
    open_only.open()
    raised = False
    try:
        open_only.begin_tree()
    except:
        raised = True
    assert_true(raised)

    # Ending what never began.
    raised = False
    try:
        open_only.end_tree()
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        open_only.end_round()
    except:
        raised = True
    assert_true(raised)

    # A round is not a tree: end_round inside a tree is a mistake, not a
    # shortcut that closes the tree for you.
    var in_tree = SessionLifecycle()
    in_tree.open()
    in_tree.begin_round()
    in_tree.begin_tree()
    raised = False
    try:
        in_tree.end_round()
    except:
        raised = True
    assert_true(raised)


def test_a_closed_session_stays_closed() raises:
    var life = SessionLifecycle()
    life.open()
    life.close()

    var raised = False
    try:
        life.open()
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        life.begin_round()
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        life.require_live()
    except:
        raised = True
    assert_true(raised)


def test_require_names_the_state_it_wanted() raises:
    var life = SessionLifecycle()
    life.open()
    life.require(STATE_OPEN)
    var raised = False
    try:
        life.require(STATE_TREE)
    except:
        raised = True
    assert_true(raised)


def test_transition_table_rejects_out_of_range_states() raises:
    assert_true(can_transition(STATE_NEW, STATE_OPEN))
    assert_true(not can_transition(STATE_NEW, STATE_ROUND))
    assert_true(not can_transition(STATE_OPEN, STATE_TREE))
    assert_true(can_transition(STATE_ROUND, STATE_ROUND))
    assert_true(can_transition(STATE_TREE, STATE_TREE))
    assert_true(not can_transition(STATE_CLOSED, STATE_OPEN))
    assert_true(not can_transition(-1, STATE_OPEN))
    assert_true(not can_transition(STATE_OPEN, N_STATES))


# ---------------------------------------------------------------------------
# The dependency model
# ---------------------------------------------------------------------------


def test_only_real_conflicts_need_a_drain() raises:
    """The whole argument for removing synchronizations: an in-order queue
    means device work never waits, and the host only waits when it touches
    memory the device has not finished with."""
    var h = HazardTracker()

    # Nothing queued: neither direction conflicts.
    assert_true(not h.host_read_hazard(RES_OUT))
    assert_true(not h.host_write_hazard(RES_OUT))
    assert_true(not h.any_pending())

    # A queued write blocks a host read of the same buffer, and only that
    # buffer.
    h.note_device_write(RES_OUT)
    assert_true(h.host_read_hazard(RES_OUT))
    assert_true(h.host_write_hazard(RES_OUT))
    assert_true(not h.host_read_hazard(RES_BINS))

    # A queued read blocks a host overwrite but not a host read.
    var h2 = HazardTracker()
    h2.note_device_read(RES_STAGE)
    assert_true(not h2.host_read_hazard(RES_STAGE))
    assert_true(h2.host_write_hazard(RES_STAGE))

    # Device work following device work needs no host involvement at all.
    var h3 = HazardTracker()
    h3.note_device_write(RES_LEAF)
    h3.note_device_read(RES_LEAF)
    h3.note_device_write(RES_OUT)
    assert_equal(h3.required(), 0)


def test_a_drain_clears_every_resource() raises:
    """`synchronize()` drains the whole queue, so one required wait pays for
    every outstanding hazard. That is why the model counts elided checks."""
    var h = HazardTracker()
    h.note_device_write(RES_OUT)
    h.note_device_read(RES_BINS)
    h.note_device_read(RES_LEAF)

    assert_true(h.sync_for_host_read(RES_OUT))
    assert_equal(h.required(), 1)
    assert_equal(h.syncs_for(SYNC_HOST_READ), 1)
    assert_true(not h.any_pending())

    # Everything else is now free, and the checks that find nothing are the
    # synchronizations today's code performs anyway.
    assert_true(not h.sync_for_host_write(RES_BINS))
    assert_true(not h.sync_for_host_write(RES_LEAF))
    assert_true(not h.sync_for_host_read(RES_OUT))
    assert_equal(h.elided, 3)
    assert_equal(h.required(), 1)


def test_sync_reasons_are_counted_separately() raises:
    var h = HazardTracker()
    h.note_device_write(RES_OUT)
    _ = h.sync_for_host_read(RES_OUT)
    h.note_device_read(RES_STAGE)
    _ = h.sync_for_host_write(RES_STAGE)
    h.sync(SYNC_TEARDOWN)

    assert_equal(h.syncs_for(SYNC_HOST_READ), 1)
    assert_equal(h.syncs_for(SYNC_HOST_WRITE), 1)
    assert_equal(h.syncs_for(SYNC_TEARDOWN), 1)
    assert_equal(h.required(), 3)
    assert_true(h.report().byte_length() > 0)


def test_unknown_resources_are_rejected_and_assumed_unsafe() raises:
    var h = HazardTracker()
    var raised = False
    try:
        h.note_device_read(-1)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        h.note_device_write(999)
    except:
        raised = True
    assert_true(raised)

    # The queries cannot raise, so an unknown resource reports a hazard
    # rather than quietly claiming safety.
    assert_true(h.host_read_hazard(999))
    assert_true(h.host_write_hazard(999))

    raised = False
    try:
        _ = HazardTracker(0)
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# Staging ring
# ---------------------------------------------------------------------------


def test_one_staging_slot_waits_on_every_reuse() raises:
    """Today's single pinned buffer: the host cannot convert the next
    round's gradients until the previous copy has retired."""
    var h = HazardTracker()
    var ring = StagingRing(1)

    assert_true(not model_upload_gradients(h, ring))
    assert_true(model_upload_gradients(h, ring))
    assert_true(model_upload_gradients(h, ring))
    assert_equal(ring.waits, 2)
    assert_equal(h.syncs_for(SYNC_HOST_WRITE), 2)


def test_two_staging_slots_absorb_one_extra_round() raises:
    """A second slot lets the host stage ahead by exactly one round. The
    third staging in a row still waits, which is the honest limit of a
    two-deep ring, not an argument for a deeper one."""
    var h = HazardTracker()
    var ring = StagingRing(2)

    assert_true(not model_upload_gradients(h, ring))
    assert_true(not model_upload_gradients(h, ring))
    assert_true(model_upload_gradients(h, ring))
    assert_equal(ring.waits, 1)
    assert_equal(ring.acquisitions, 3)


def test_any_drain_retires_every_staged_copy() raises:
    """Retirement is derived from the drain count rather than tracked, so a
    drain that happened for some other reason (a histogram download, say)
    frees the staging slots too, and the ring cannot disagree with the
    hazard tracker about it."""
    var h = HazardTracker()
    var ring = StagingRing(1)

    assert_true(not model_upload_gradients(h, ring))
    assert_equal(ring.in_flight(h.required()), 1)

    # A build downloads its histogram, which drains the queue.
    assert_true(model_build_leaf(h, True))
    assert_equal(ring.in_flight(h.required()), 0)

    # So the next staging finds the slot free even with a one-deep ring.
    assert_true(not model_upload_gradients(h, ring))
    assert_equal(ring.waits, 0)


def test_staging_ring_depth_is_bounded() raises:
    var raised = False
    try:
        _ = StagingRing(0)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = StagingRing(MAX_STAGING_SLOTS + 1)
    except:
        raised = True
    assert_true(raised)

    var ring = StagingRing(MAX_STAGING_SLOTS)
    assert_equal(ring.n_slots, MAX_STAGING_SLOTS)
    raised = False
    try:
        ring.mark_in_flight(MAX_STAGING_SLOTS, 0)
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# The modeled pipeline
# ---------------------------------------------------------------------------


def test_device_only_operations_queue_without_waiting() raises:
    """`apply_split` and an unbagged `begin_tree` are pure device work, so
    neither is allowed to cost a drain."""
    var h = HazardTracker()
    assert_true(not model_begin_tree(h, False))
    model_apply_split(h)
    model_apply_split(h)
    assert_equal(h.required(), 0)
    assert_true(h.any_pending())


def test_a_bagged_begin_tree_waits_for_the_kernels_reading_leaf_ids() raises:
    """The bagged reset writes the leaf array from the host, so it has to
    wait for anything still scanning it, and it queues nothing itself."""
    var h = HazardTracker()
    h.note_device_read(RES_LEAF)
    assert_true(model_begin_tree(h, True))
    assert_equal(h.syncs_for(SYNC_HOST_WRITE), 1)
    assert_true(not h.any_pending())

    # With nothing in flight it is free.
    assert_true(not model_begin_tree(h, True))


def test_set_features_is_free_after_a_download() raises:
    """Every tree calls `set_features` first, and the previous tree ended
    with a histogram download, which drained. So the map-to-host write it
    performs never actually has anything to wait for."""
    var h = HazardTracker()
    assert_true(model_build_leaf(h, True))
    assert_true(not model_set_features(h))
    assert_equal(h.elided, 1)


def test_building_a_histogram_always_waits() raises:
    """The one drain the dependency analysis cannot argue away as it
    stands: the host reads the downloaded histogram to search it for a
    split, and the download is device work."""
    var h = HazardTracker()
    assert_true(model_build_leaf(h, True))
    assert_true(model_build_leaf(h, False))
    assert_equal(h.syncs_for(SYNC_HOST_READ), 2)


def test_audit_round_matches_the_current_pipeline() raises:
    """One single-output round over an 8-leaf tree.

    The current code drains once in `stage_gradients` and once per
    `download_raw`, so nine times. The model requires eight: every
    histogram download, and nothing else. The one drain it does not need is
    the gradient staging wait, which the two-slot ring covers.
    """
    var audit = audit_round(8)
    assert_equal(audit.unconditional, 9)
    assert_equal(audit.required, 8)
    assert_equal(audit.staging_waits, 0)
    # The upload and the set_features checks, both of which find nothing.
    assert_equal(audit.elided, 2)


def test_audit_scales_with_multiclass_rounds() raises:
    """One round of a three-class fit grows three trees off one builder, so
    it stages three times and downloads three times as many histograms."""
    var audit = audit_round(8, True, DEFAULT_STAGING_SLOTS, False, 3)
    assert_equal(audit.unconditional, 27)
    assert_equal(audit.required, 24)
    assert_equal(audit.staging_waits, 0)
    assert_equal(audit.elided, 6)


def test_audit_with_one_staging_slot_still_needs_no_waits() raises:
    """A one-deep ring costs nothing here, because the per-node downloads
    already drain between rounds. Deeper staging only starts to pay once
    those downloads are gone, and this asserts that ordering rather than
    letting the two changes be credited to each other."""
    var deep = audit_round(8, True, DEFAULT_STAGING_SLOTS, False, 3)
    var shallow = audit_round(8, True, 1, False, 3)
    assert_equal(shallow.required, deep.required)
    assert_equal(shallow.staging_waits, 0)


def test_audit_of_a_bagged_round() raises:
    """Bagging replaces the memset with a host write, which is one more
    hazard check per tree and, after a download, still no extra drain."""
    var audit = audit_round(8, True, DEFAULT_STAGING_SLOTS, True, 1)
    assert_equal(audit.required, 8)
    assert_equal(audit.elided, 3)


def test_audit_rejects_impossible_rounds() raises:
    var raised = False
    try:
        _ = audit_round(0)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = audit_round(4, True, DEFAULT_STAGING_SLOTS, False, 0)
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# Buffer pool
# ---------------------------------------------------------------------------


def test_pool_reuses_and_never_shrinks() raises:
    """A second fit on the same or smaller data must allocate nothing,
    which is the whole reason the session outlives one fit."""
    var pool = PoolLedger()
    assert_equal(pool.request(0, 1000, 1), POOL_ALLOCATE)
    assert_equal(pool.request(0, 1000, 1), POOL_REUSE)
    assert_equal(pool.request(0, 500, 1), POOL_REUSE)
    assert_equal(pool.capacity_of(0), 1000)

    assert_equal(pool.request(0, 4000, 1), POOL_GROW)
    assert_equal(pool.capacity_of(0), 4000)
    assert_equal(pool.request(0, 4000, 1), POOL_REUSE)

    assert_equal(pool.allocations, 1)
    assert_equal(pool.growths, 1)
    assert_equal(pool.reuses, 3)


def test_pool_reallocates_when_the_element_width_changes() raises:
    """Device buffers are typed. Serving an Int32 request out of a UInt8
    buffer that happens to have the bytes is the aliasing bug this ledger
    exists to prevent."""
    var pool = PoolLedger()
    assert_equal(pool.request(1, 256, 1), POOL_ALLOCATE)
    assert_equal(pool.request(1, 64, 4), POOL_GROW)
    assert_equal(pool.capacity_of(1), 64)
    assert_equal(pool.resident_bytes(), 256)


def test_pool_release_returns_every_slot() raises:
    var pool = PoolLedger()
    _ = pool.request(0, 100, 4)
    _ = pool.request(1, 200, 4)
    assert_equal(pool.resident_bytes(), 1200)
    pool.release_all()
    assert_equal(pool.resident_bytes(), 0)
    # A released slot allocates again rather than reporting a stale reuse.
    assert_equal(pool.request(0, 100, 4), POOL_ALLOCATE)


def test_pool_rejects_nonsense_requests() raises:
    var pool = PoolLedger()
    var raised = False
    try:
        _ = pool.request(N_POOL_SLOTS, 10, 1)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = pool.request(0, 0, 1)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = pool.request(0, 10, 0)
    except:
        raised = True
    assert_true(raised)

    assert_equal(pool.capacity_of(-1), 0)
    assert_equal(pool.capacity_of(N_POOL_SLOTS), 0)


# ---------------------------------------------------------------------------
# Residency
# ---------------------------------------------------------------------------


def test_residency_skips_the_upload_only_for_the_same_matrix() raises:
    var ledger = ResidencyLedger()
    var first = MatrixIdentity(100, 8, 32, UInt64(0x1234))

    assert_true(ledger.admit(ROLE_TRAIN, first))
    assert_true(not ledger.admit(ROLE_TRAIN, first))
    assert_equal(ledger.uploads, 1)
    assert_equal(ledger.reuses, 1)
    assert_equal(ledger.resident_cells(), 800)


def test_same_shape_different_contents_is_not_resident() raises:
    """The failure this guards against is silent: two matrices of the same
    shape aliasing one device copy trains a model on the wrong data."""
    var ledger = ResidencyLedger()
    var a = MatrixIdentity(100, 8, 32, UInt64(1))
    var b = MatrixIdentity(100, 8, 32, UInt64(2))

    assert_true(ledger.admit(ROLE_TRAIN, a))
    assert_true(ledger.admit(ROLE_TRAIN, b))
    assert_equal(ledger.evictions, 1)
    assert_equal(ledger.uploads, 2)
    assert_true(not ledger.is_resident(ROLE_TRAIN, a))
    assert_true(ledger.is_resident(ROLE_TRAIN, b))


def test_training_and_validation_matrices_are_independent() raises:
    """Early stopping scores a held-out matrix every round; it must not
    evict the training matrix to do it."""
    var ledger = ResidencyLedger()
    var train = MatrixIdentity(1000, 16, 32, UInt64(7))
    var valid = MatrixIdentity(200, 16, 32, UInt64(9))

    assert_true(ledger.admit(ROLE_TRAIN, train))
    assert_true(ledger.admit(ROLE_VALID, valid))
    assert_true(not ledger.admit(ROLE_TRAIN, train))
    assert_true(not ledger.admit(ROLE_VALID, valid))
    assert_equal(ledger.evictions, 0)
    assert_equal(ledger.resident_cells(), 1000 * 16 + 200 * 16)

    ledger.evict(ROLE_VALID)
    assert_true(ledger.is_resident(ROLE_TRAIN, train))
    assert_true(not ledger.is_resident(ROLE_VALID, valid))

    ledger.clear()
    assert_equal(ledger.resident_cells(), 0)
    assert_true(ledger.admit(ROLE_TRAIN, train))


def test_residency_rejects_unknown_roles_and_empty_matrices() raises:
    var ledger = ResidencyLedger()
    var raised = False
    try:
        _ = ledger.admit(N_ROLES, MatrixIdentity(10, 2, 8, UInt64(1)))
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        _ = ledger.admit(ROLE_TRAIN, MatrixIdentity(0, 2, 8, UInt64(1)))
    except:
        raised = True
    assert_true(raised)

    assert_true(not ledger.is_resident(N_ROLES, MatrixIdentity.empty()))


def _same_fingerprint(a: UInt64, b: UInt64) -> Bool:
    """UInt64 equality as a plain Bool, so the assertions below take a Bool
    rather than a one-lane mask."""
    if a == b:
        return True
    return False


def test_fingerprint_covers_every_cell_and_the_shape() raises:
    """A sampled fingerprint would let a matrix that differs outside the
    sample reuse another one's device copy, so this checks a single changed
    cell and a reinterpreted shape both move it."""
    var bins = List[UInt8](capacity=64)
    for i in range(64):
        bins.append(UInt8(i % 5))

    var base = bins_fingerprint(bins, 8, 8, 5)
    assert_true(_same_fingerprint(bins_fingerprint(bins, 8, 8, 5), base))
    assert_true(not _same_fingerprint(bins_fingerprint(bins, 4, 16, 5), base))
    assert_true(not _same_fingerprint(bins_fingerprint(bins, 8, 8, 6), base))

    # The last cell, which a strided sample is most likely to miss.
    bins[63] = bins[63] + UInt8(1)
    assert_true(not _same_fingerprint(bins_fingerprint(bins, 8, 8, 5), base))


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------


def test_each_kernel_warms_exactly_once() raises:
    var registry = KernelRegistry()
    assert_true(registry.needs_warm(KERNEL_PARTITION))
    assert_true(registry.mark_warm(KERNEL_PARTITION))
    assert_true(not registry.mark_warm(KERNEL_PARTITION))
    assert_true(not registry.needs_warm(KERNEL_PARTITION))

    assert_true(registry.mark_warm(KERNEL_HIST_ATOMIC))
    assert_equal(registry.warm_count, 2)

    registry.clear()
    assert_equal(registry.warm_count, 0)
    assert_true(registry.needs_warm(KERNEL_PARTITION))

    var raised = False
    try:
        _ = registry.mark_warm(N_KERNELS)
    except:
        raised = True
    assert_true(raised)


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------


def test_disabled_counters_still_count_calls() raises:
    """Call counts are what the lifecycle tests assert on, so they are
    unconditional; the clock is not read at all when tracing is off, which
    is what keeps an untraced fit free of timing overhead."""
    var counters = PhaseCounters(False)
    assert_equal(counters.clock(), 0)

    counters.record(PHASE_TRANSFER, 0)
    counters.record(PHASE_TRANSFER, 0)
    counters.record(PHASE_SYNC, 0)

    assert_equal(counters.calls_of(PHASE_TRANSFER), 2)
    assert_equal(counters.calls_of(PHASE_SYNC), 1)
    assert_equal(counters.calls_of(PHASE_KERNEL), 0)
    assert_equal(counters.total_calls(), 3)
    assert_equal(counters.nanos_of(PHASE_TRANSFER), 0)
    assert_equal(counters.total_nanos(), 0)

    counters.reset()
    assert_equal(counters.total_calls(), 0)


def test_enabled_counters_read_the_clock() raises:
    var counters = PhaseCounters(True)
    var started = counters.clock()
    assert_true(started > 0)
    counters.record(PHASE_ALLOC, started)
    assert_equal(counters.calls_of(PHASE_ALLOC), 1)
    # Elapsed time is whatever it is; the assertion is that it is recorded
    # as a nonnegative duration, not that it is any particular size.
    assert_true(counters.nanos_of(PHASE_ALLOC) >= 0)
    assert_true(counters.report().byte_length() > 0)


def test_counters_reject_unknown_phases() raises:
    var counters = PhaseCounters(False)
    var raised = False
    try:
        counters.record(N_PHASES, 0)
    except:
        raised = True
    assert_true(raised)

    raised = False
    try:
        counters.record(-1, 0)
    except:
        raised = True
    assert_true(raised)

    assert_equal(counters.calls_of(N_PHASES), 0)
    assert_equal(counters.nanos_of(-1), 0)


def test_names_cover_every_constant() raises:
    """A trace is read by a person, so every id has to have a name and an
    unknown id has to say so rather than print a number."""
    assert_equal(phase_name(PHASE_COMPILE), "compile")
    assert_equal(phase_name(PHASE_CLEANUP), "cleanup")
    assert_equal(phase_name(N_PHASES), "unknown")

    assert_equal(resource_name(RES_BINS), "bins")
    assert_equal(resource_name(RES_FEAT), "feat")
    assert_equal(resource_name(-1), "unknown")

    assert_equal(sync_reason_name(SYNC_HOST_READ), "host_read")
    assert_equal(sync_reason_name(SYNC_TEARDOWN), "teardown")
    assert_equal(sync_reason_name(-1), "unknown")

    assert_equal(state_name(STATE_TREE), "tree")
    assert_equal(state_name(N_STATES), "unknown")

    assert_equal(pool_action_name(POOL_REUSE), "reuse")
    assert_equal(pool_action_name(-1), "unknown")

    assert_equal(kernel_name(KERNEL_PARTITION), "partition")
    assert_equal(kernel_name(N_KERNELS), "unknown")

    assert_equal(role_name(ROLE_TRAIN), "train")
    assert_equal(role_name(ROLE_VALID), "valid")
    assert_equal(role_name(N_ROLES), "unknown")


def test_env_overrides() raises:
    """One test owns all environment mutation so no other test sees a dirty
    environment regardless of suite ordering; empty string means unset."""
    _ = setenv("MOJOTREES_GPU_STAGING_SLOTS", "4")
    assert_equal(env_staging_slots(), 4)

    _ = setenv("MOJOTREES_GPU_STAGING_SLOTS", "0")
    assert_equal(env_staging_slots(), 1)

    _ = setenv("MOJOTREES_GPU_STAGING_SLOTS", "9999")
    assert_equal(env_staging_slots(), MAX_STAGING_SLOTS)

    _ = setenv("MOJOTREES_GPU_STAGING_SLOTS", "nonsense")
    assert_equal(env_staging_slots(), DEFAULT_STAGING_SLOTS)

    _ = setenv("MOJOTREES_GPU_STAGING_SLOTS", "")
    assert_equal(env_staging_slots(), DEFAULT_STAGING_SLOTS)

    _ = setenv("MOJOTREES_GPU_TRACE", "1")
    var traced = PhaseCounters.from_env()
    assert_true(traced.enabled)

    _ = setenv("MOJOTREES_GPU_TRACE", "")
    var untraced = PhaseCounters.from_env()
    assert_true(not untraced.enabled)


def test_session_builder_matches_private_context_builder() raises:
    """The one device-bound test in this file: a builder borrowing a
    session's context produces the bit-identical fixed-point histogram a
    private-context builder does, and the session's ledgers record the
    construction. Skips (passing) without an accelerator, so the rest of
    the suite stays device-free."""
    comptime if not has_accelerator():
        print("skipped: no accelerator")
    else:
        var n_rows = 2_000
        var n_features = 4
        var features = List[Float64](capacity=n_rows * n_features)
        for k in range(n_rows * n_features):
            features.append(Float64((k * 2654435761) % 1000) / 1000.0)
        var data = bin_equal_width(features, n_rows, n_features, 32)
        var grad = List[Float64](capacity=n_rows)
        var hess = List[Float64](capacity=n_rows)
        for r in range(n_rows):
            grad.append(Float64((r * 40503) % 997) / 997.0 - 0.5)
            hess.append(1.0)

        var session = GpuSession()
        var shared = GpuHistogramBuilder(session, data)
        var a = shared.build(grad, hess)
        var private = GpuHistogramBuilder(data)
        var b = private.build(grad, hess)
        for i in range(a.n_features * a.n_bins):
            assert_equal(a.grad[i], b.grad[i])
            assert_equal(a.hess[i], b.hess[i])
            assert_equal(a.count[i], b.count[i])

        # The ledgers saw the construction: one training matrix admitted,
        # one allocation per pool slot the builder fills.
        assert_equal(session.residency.uploads, 1)
        assert_equal(session.pool.allocations, 8)
        session.close()
        assert_true(session.life.is_closed())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
