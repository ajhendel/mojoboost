"""The distributed transport layer: framing, session ordering, failure, and
the gather-and-broadcast collective.

Everything here runs in one process against in-memory endpoints and a manual
clock. That is a deliberate limit, not an oversight: the point of this suite
is to cover the protocol logic completely without a port, a process, or a
sleep, so that when a socket endpoint is written the only new thing to test is
the socket.

What that means for what these tests prove. They prove the wire format
round-trips and rejects corruption, that collectives are ordered and that a
frame out of its order fails the run instead of being reduced, that a rank
disagreeing about the world or the job is refused at the handshake, that
timeouts and cancellation and worker loss are terminal, and that the root's
incremental fold equals an ascending-rank-order reduction on data where the
difference is visible in the last bit. They prove nothing about behavior over
a network, because none of this has been over one.

`MemoryEndpoint` is a fake and is used as one. It is primed with the bytes a
correct counterpart would have sent, which is what lets a single-threaded test
drive both sides of a blocking protocol without deadlocking on itself.
"""

from std.testing import assert_equal, assert_true, TestSuite

from mojoboost.collective import agree_equal_ints
from mojoboost.distributed_transport import (
    CHECKPOINT_BYTES,
    FRAME_HEADER_BYTES,
    MSG_CONTRIB,
    MSG_RESULT,
    NS_PER_SECOND,
    OP_BARRIER,
    OP_MAX_INT,
    OP_SUM_F64,
    OP_SUM_INT,
    SESSION_CANCELLED,
    SESSION_FAILED,
    SESSION_READY,
    TRANSPORT_JOB_MISMATCH,
    TRANSPORT_OK,
    TRANSPORT_PEER_LOST,
    TRANSPORT_PROTOCOL_VERSION,
    TRANSPORT_SCHEMA_MISMATCH,
    TRANSPORT_WORLD_MISMATCH,
    CheckpointMeta,
    HandshakeRecord,
    MemoryEndpoint,
    TransportCollective,
    TransportConfig,
    TransportSession,
    absorb_f64,
    adopt_result_f64,
    barrier_frame,
    check_histogram_buffers,
    check_split_agreement,
    contribution_frame_f64,
    contribution_frame_int,
    decode_checkpoint_meta,
    decode_f64_payload,
    decode_frame,
    decode_handshake,
    decode_int_payload,
    encode_checkpoint_meta,
    encode_f64_payload,
    encode_frame,
    encode_handshake,
    encode_int_payload,
    f64_bits,
    f64_from_bits,
    handshake_status,
    histogram_plan,
    local_handshake,
    parse_machine_list,
    reduce_ordered_f64,
    restart_status,
    result_frame_f64,
    resume_session,
    split_digest,
    zigzag_decode,
    zigzag_encode,
)


comptime _JOB = UInt64(0x00A11CE5)


def _machine_list(world_size: Int) -> String:
    var text = String("")
    for r in range(world_size):
        text += "10.0.0." + String(r + 1) + ":" + String(9000 + r) + "\n"
    return text^


def _config(rank: Int, world_size: Int) raises -> TransportConfig:
    return parse_machine_list(_machine_list(world_size), rank, _JOB)


def _ready_session(rank: Int, world_size: Int) raises -> TransportSession:
    """A session that has already agreed a world with every peer, with time
    driven by hand so deadline behavior is deterministic and instant."""
    var config = _config(rank, world_size)
    var schema = config.schema_digest()
    var session = TransportSession(config^)
    session.use_manual_clock(1_000_000)
    var peers = List[HandshakeRecord]()
    for r in range(world_size):
        if r != rank:
            peers.append(
                HandshakeRecord(
                    TRANSPORT_PROTOCOL_VERSION, r, world_size, _JOB, schema, 0
                )
            )
    session.complete_handshake(peers)
    return session^


def _one_endpoint(var endpoint: MemoryEndpoint) -> List[MemoryEndpoint]:
    var out = List[MemoryEndpoint]()
    out.append(endpoint^)
    return out^


# ---------------------------------------------------------------------------
# Integer and byte coding
# ---------------------------------------------------------------------------


def test_zigzag_round_trips_including_negatives() raises:
    """Negatives have to survive the wire because `agree_equal_ints` reduces
    every value alongside its negation, so half of every configuration
    agreement message is negative by construction."""
    var values: List[Int] = [0, 1, -1, 2, -2, 255, -255, 1_000_000, -1_000_000]
    for i in range(len(values)):
        assert_equal(zigzag_decode(zigzag_encode(values[i])), values[i])
    assert_true(
        zigzag_encode(1) != zigzag_encode(-1), "the mapping must be injective"
    )


def test_int_payload_round_trips() raises:
    var buf: List[Int] = [0, -7, 42, -2_147_483_648, 2_147_483_647]
    var payload = encode_int_payload(buf)
    assert_equal(len(payload), 8 * len(buf))
    var got = decode_int_payload(payload)
    assert_equal(len(got), len(buf))
    for i in range(len(buf)):
        assert_equal(got[i], buf[i])


def test_f64_payload_round_trips_exactly() raises:
    """Bit-exact, not almost equal. A transport that loses the last bit of a
    gradient sum breaks the determinism claim in docs/distributed.md section
    6, so the test has to be able to see the last bit."""
    var buf: List[Float64] = [0.0, -0.0, 1.0, -1.0, 0.1, 1e-300, 1e300]
    var payload = encode_f64_payload(buf)
    assert_equal(len(payload), 8 * len(buf))
    var got = decode_f64_payload(payload)
    assert_equal(len(got), len(buf))
    for i in range(len(buf)):
        assert_equal(f64_bits(got[i]), f64_bits(buf[i]))


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def test_frame_round_trips_every_header_field() raises:
    var session = _ready_session(2, 4)
    var header = session.begin(OP_SUM_F64, 3)
    var buf: List[Float64] = [1.5, -2.5, 3.5]
    var bytes = encode_frame(header, encode_f64_payload(buf))
    assert_equal(len(bytes), FRAME_HEADER_BYTES + 24)

    var frame = decode_frame(bytes)
    assert_equal(frame.header.version, TRANSPORT_PROTOCOL_VERSION)
    assert_equal(frame.header.msg_type, MSG_CONTRIB)
    assert_equal(frame.header.job_id, _JOB)
    assert_equal(frame.header.epoch, 0)
    assert_equal(frame.header.seq, 0)
    assert_equal(frame.header.sender, 2)
    assert_equal(frame.header.op, OP_SUM_F64)
    assert_equal(frame.header.n_elements, 3)
    var payload = decode_f64_payload(frame.payload)
    for i in range(3):
        assert_equal(f64_bits(payload[i]), f64_bits(buf[i]))


def test_a_flipped_payload_byte_is_caught() raises:
    var session = _ready_session(1, 2)
    _ = session.begin(OP_SUM_F64, 2)
    var buf: List[Float64] = [1.0, 2.0]
    var bytes = contribution_frame_f64(session, buf)
    bytes[FRAME_HEADER_BYTES + 3] = bytes[FRAME_HEADER_BYTES + 3] + 1

    var message = String("")
    try:
        _ = decode_frame(bytes)
    except e:
        message = String(e)
    assert_true(
        message.find("checksum") >= 0,
        "expected a checksum rejection, got: " + message,
    )


def test_a_corrupt_header_is_caught_on_its_own_frame() raises:
    """The checksum covers the header as well as the payload, so a mangled
    header fails here rather than one frame later, when the reader would
    otherwise blame the next frame's magic."""
    var session = _ready_session(1, 2)
    _ = session.begin(OP_SUM_F64, 2)
    var buf: List[Float64] = [1.0, 2.0]

    var bad_magic = contribution_frame_f64(session, buf)
    bad_magic[0] = bad_magic[0] + 1
    var magic_message = String("")
    try:
        _ = decode_frame(bad_magic)
    except e:
        magic_message = String(e)
    assert_true(
        magic_message.find("magic") >= 0,
        "expected a magic rejection, got: " + magic_message,
    )

    var whole = contribution_frame_f64(session, buf)
    var truncated = List[UInt8]()
    for i in range(len(whole) - 4):
        truncated.append(whole[i])
    var short_message = String("")
    try:
        _ = decode_frame(truncated)
    except e:
        short_message = String(e)
    assert_true(
        short_message.find("length") >= 0,
        "expected a length rejection, got: " + short_message,
    )

    var sender_lied = contribution_frame_f64(session, buf)
    sender_lied[24] = sender_lied[24] + 1
    var sender_message = String("")
    try:
        _ = decode_frame(sender_lied)
    except e:
        sender_message = String(e)
    assert_true(
        sender_message.find("checksum") >= 0,
        "expected a header field to be covered, got: " + sender_message,
    )


def test_a_declared_length_that_disagrees_is_refused_at_encode() raises:
    """Caught on the sender, which is the only side that can still tell which
    of the two numbers was the wrong one."""
    var session = _ready_session(1, 2)
    var header = session.begin(OP_SUM_F64, 5)
    var short: List[Float64] = [1.0, 2.0]
    var payload = encode_f64_payload(short)

    var message = String("")
    try:
        _ = encode_frame(header, payload)
    except e:
        message = String(e)
    assert_true(
        message.find("element count") >= 0,
        "expected an element count rejection, got: " + message,
    )


# ---------------------------------------------------------------------------
# Configuration and handshake
# ---------------------------------------------------------------------------


def test_machine_list_parsing_assigns_ranks_by_file_order() raises:
    var text = String("# the leader\n")
    text += "10.0.0.1:9000\n"
    text += "\n"
    text += "10.0.0.2:9001   # a follower\n"
    text += "10.0.0.3:9002\n"

    var config = parse_machine_list(text, 2, _JOB)
    assert_equal(config.world_size, 3)
    assert_equal(config.rank, 2)
    assert_equal(config.addresses[0].text(), "10.0.0.1:9000")
    assert_equal(config.addresses[1].text(), "10.0.0.2:9001")
    assert_equal(config.addresses[2].text(), "10.0.0.3:9002")
    assert_true(not config.is_root(), "rank 2 is not the root")
    assert_true(_config(0, 3).is_root(), "rank 0 is the root")


def test_a_broken_machine_list_is_refused() raises:
    var cases: List[String] = [
        "10.0.0.1\n",
        "10.0.0.1:0\n",
        "10.0.0.1:70000\n",
        ":9000\n",
        "# only a comment\n",
        "10.0.0.1:9000\n10.0.0.1:9000\n",
    ]
    for i in range(len(cases)):
        var message = String("")
        try:
            _ = parse_machine_list(cases[i], 0, _JOB)
        except e:
            message = String(e)
        assert_true(
            message.byte_length() > 0,
            "expected '" + cases[i] + "' to be refused",
        )


def test_a_rank_outside_the_world_is_refused() raises:
    var message = String("")
    try:
        _ = parse_machine_list(_machine_list(2), 2, _JOB)
    except e:
        message = String(e)
    assert_true(
        message.find("outside the world") >= 0,
        "expected the rank to be refused, got: " + message,
    )


def test_the_schema_digest_covers_the_list_and_not_the_rank() raises:
    """Every rank has to compute the same digest from the same file, and a
    rank started against a rewritten file has to compute a different one."""
    var a = _config(0, 3)
    var b = _config(2, 3)
    assert_equal(a.schema_digest(), b.schema_digest())

    var rewritten = parse_machine_list(
        "10.0.0.1:9000\n10.0.0.2:9001\n10.0.0.9:9002\n", 0, _JOB
    )
    assert_true(
        a.schema_digest() != rewritten.schema_digest(),
        "a rewritten machine list must change the digest",
    )

    var smaller = _config(0, 2)
    assert_true(
        a.schema_digest() != smaller.schema_digest(),
        "a different world size must change the digest",
    )


def test_handshake_round_trips_and_rejects_every_disagreement() raises:
    var config = _config(0, 4)
    var local = local_handshake(config, 0)
    var bytes = encode_handshake(local)
    var back = decode_handshake(bytes)
    assert_equal(back.protocol_version, TRANSPORT_PROTOCOL_VERSION)
    assert_equal(back.rank, 0)
    assert_equal(back.world_size, 4)
    assert_equal(back.job_id, _JOB)
    assert_equal(back.schema, local.schema)
    assert_equal(back.restart_epoch, 0)

    var good = HandshakeRecord(
        TRANSPORT_PROTOCOL_VERSION, 1, 4, _JOB, local.schema, 0
    )
    assert_equal(handshake_status(local, good), TRANSPORT_OK)

    var wrong_version = HandshakeRecord(
        TRANSPORT_PROTOCOL_VERSION + 1, 1, 4, _JOB, local.schema, 0
    )
    assert_equal(
        handshake_status(local, wrong_version), TRANSPORT_SCHEMA_MISMATCH
    )

    var wrong_job = HandshakeRecord(
        TRANSPORT_PROTOCOL_VERSION, 1, 4, _JOB + 1, local.schema, 0
    )
    assert_equal(handshake_status(local, wrong_job), TRANSPORT_JOB_MISMATCH)

    var wrong_world = HandshakeRecord(
        TRANSPORT_PROTOCOL_VERSION, 1, 5, _JOB, local.schema, 0
    )
    assert_equal(handshake_status(local, wrong_world), TRANSPORT_WORLD_MISMATCH)

    var same_rank = HandshakeRecord(
        TRANSPORT_PROTOCOL_VERSION, 0, 4, _JOB, local.schema, 0
    )
    assert_equal(handshake_status(local, same_rank), TRANSPORT_WORLD_MISMATCH)

    var wrong_schema = HandshakeRecord(
        TRANSPORT_PROTOCOL_VERSION, 1, 4, _JOB, local.schema + 1, 0
    )
    assert_equal(
        handshake_status(local, wrong_schema), TRANSPORT_SCHEMA_MISMATCH
    )

    var wrong_epoch = HandshakeRecord(
        TRANSPORT_PROTOCOL_VERSION, 1, 4, _JOB, local.schema, 3
    )
    assert_equal(
        handshake_status(local, wrong_epoch), TRANSPORT_SCHEMA_MISMATCH
    )


def test_two_peers_claiming_one_rank_are_refused() raises:
    var config = _config(0, 3)
    var schema = config.schema_digest()
    var session = TransportSession(config^)
    var peers = List[HandshakeRecord]()
    peers.append(
        HandshakeRecord(TRANSPORT_PROTOCOL_VERSION, 1, 3, _JOB, schema, 0)
    )
    peers.append(
        HandshakeRecord(TRANSPORT_PROTOCOL_VERSION, 1, 3, _JOB, schema, 0)
    )

    var message = String("")
    try:
        session.complete_handshake(peers)
    except e:
        message = String(e)
    assert_true(
        message.find("same rank") >= 0,
        "expected the duplicate rank to be refused, got: " + message,
    )
    assert_equal(session.state, SESSION_FAILED)


def test_a_collective_before_the_handshake_is_refused() raises:
    var session = TransportSession(_config(0, 2))
    var message = String("")
    try:
        _ = session.begin(OP_SUM_F64, 1)
    except e:
        message = String(e)
    assert_true(
        message.find("begin in state init") >= 0,
        "expected the state machine to refuse it, got: " + message,
    )


# ---------------------------------------------------------------------------
# Ordering, deadlines, cancellation, and worker loss
# ---------------------------------------------------------------------------


def test_the_sequence_advances_once_per_collective() raises:
    var session = _ready_session(0, 2)
    assert_equal(session.state, SESSION_READY)
    for expected in range(3):
        assert_equal(session.seq, expected)
        _ = session.begin(OP_SUM_INT, 2)
        session.finish()
    assert_equal(session.seq, 3)
    assert_equal(session.collectives_completed, 3)
    assert_equal(session.elements_reduced, 6)


def test_an_epoch_boundary_restarts_the_sequence() raises:
    var session = _ready_session(0, 2)
    _ = session.begin(OP_SUM_INT, 1)
    session.finish()
    session.next_epoch()
    assert_equal(session.epoch, 1)
    assert_equal(session.seq, 0)

    _ = session.begin(OP_SUM_INT, 1)
    var message = String("")
    try:
        session.next_epoch()
    except e:
        message = String(e)
    assert_true(
        message.find("inside a collective") >= 0,
        "expected the boundary to be refused, got: " + message,
    )


def test_a_frame_from_the_next_collective_is_refused() raises:
    """Requirement 5 of docs/distributed.md section 5, no reordering across
    calls, as a runtime check: one node's histogram cannot overtake the next
    and be summed into the wrong node."""
    var root = _ready_session(0, 2)
    var peer = _ready_session(1, 2)
    _ = root.begin(OP_SUM_F64, 1)
    _ = peer.begin(OP_SUM_F64, 1)
    peer.finish()
    _ = peer.begin(OP_SUM_F64, 1)

    var ahead: List[Float64] = [1.0]
    var frame = contribution_frame_f64(peer, ahead)
    var acc: List[Float64] = [0.0]
    var message = String("")
    try:
        absorb_f64(root, acc, frame, 1)
    except e:
        message = String(e)
    assert_true(
        message.find("sequence") >= 0,
        "expected an out-of-order rejection, got: " + message,
    )
    assert_equal(root.state, SESSION_FAILED)
    assert_equal(acc[0], 0.0)


def test_a_frame_from_the_wrong_sender_is_refused() raises:
    var root = _ready_session(0, 3)
    var peer = _ready_session(1, 3)
    _ = root.begin(OP_SUM_F64, 1)
    _ = peer.begin(OP_SUM_F64, 1)
    var buf: List[Float64] = [1.0]
    var frame = contribution_frame_f64(peer, buf)
    var acc: List[Float64] = [0.0]

    var message = String("")
    try:
        absorb_f64(root, acc, frame, 2)
    except e:
        message = String(e)
    assert_true(
        message.find("expected rank 2") >= 0,
        "expected a sender rejection, got: " + message,
    )


def test_a_frame_from_another_job_is_refused() raises:
    var root = _ready_session(0, 2)

    var stranger_config = parse_machine_list(_machine_list(2), 1, _JOB + 99)
    var stranger_schema = stranger_config.schema_digest()
    var stranger = TransportSession(stranger_config^)
    var stranger_peers = List[HandshakeRecord]()
    stranger_peers.append(
        HandshakeRecord(
            TRANSPORT_PROTOCOL_VERSION, 0, 2, _JOB + 99, stranger_schema, 0
        )
    )
    stranger.complete_handshake(stranger_peers)

    _ = root.begin(OP_SUM_F64, 1)
    _ = stranger.begin(OP_SUM_F64, 1)
    var buf: List[Float64] = [1.0]
    var frame = contribution_frame_f64(stranger, buf)
    var acc: List[Float64] = [0.0]

    var message = String("")
    try:
        absorb_f64(root, acc, frame, 1)
    except e:
        message = String(e)
    assert_true(
        message.find("foreign job id") >= 0,
        "expected a job rejection, got: " + message,
    )


def test_a_buffer_length_disagreement_is_a_schema_failure() raises:
    var root = _ready_session(0, 2)
    var peer = _ready_session(1, 2)
    _ = root.begin(OP_SUM_F64, 2)
    _ = peer.begin(OP_SUM_F64, 3)
    var wide: List[Float64] = [1.0, 2.0, 3.0]
    var frame = contribution_frame_f64(peer, wide)
    var acc: List[Float64] = [0.0, 0.0]

    var message = String("")
    try:
        absorb_f64(root, acc, frame, 1)
    except e:
        message = String(e)
    assert_true(
        message.find("buffer length") >= 0,
        "expected a buffer length rejection, got: " + message,
    )


def test_an_expired_deadline_is_terminal() raises:
    """A timed-out collective cannot be retried: the peers that did answer
    have already advanced their sequence numbers, so the only correct outcome
    is a failure every rank reaches."""
    var session = _ready_session(0, 2)
    _ = session.begin(OP_SUM_F64, 1)
    session.advance_clock(400 * NS_PER_SECOND)

    var message = String("")
    try:
        session.check_deadline()
    except e:
        message = String(e)
    assert_true(
        message.find("deadline") >= 0,
        "expected a deadline failure, got: " + message,
    )
    assert_equal(session.state, SESSION_FAILED)

    var second = String("")
    try:
        _ = session.begin(OP_SUM_F64, 1)
    except e:
        second = String(e)
    assert_true(
        second.find("deadline") >= 0,
        "expected the failure to be sticky, got: " + second,
    )


def test_a_deadline_that_has_not_expired_does_nothing() raises:
    var session = _ready_session(0, 2)
    _ = session.begin(OP_SUM_F64, 1)
    session.advance_clock(NS_PER_SECOND)
    session.check_deadline()
    session.finish()
    assert_equal(session.state, SESSION_READY)


def test_cancellation_is_sticky_and_distinct_from_a_failure() raises:
    var session = _ready_session(0, 2)
    session.cancel("the operator stopped the run")
    assert_equal(session.state, SESSION_CANCELLED)

    var message = String("")
    try:
        _ = session.begin(OP_SUM_F64, 1)
    except e:
        message = String(e)
    assert_true(
        message.find("cancelled") >= 0,
        "expected a cancellation, got: " + message,
    )


def test_a_lost_worker_stops_the_session_and_names_the_lowest_rank() raises:
    var session = _ready_session(0, 4)
    session.mark_lost(3)
    session.mark_lost(2)
    assert_equal(session.lost_rank(), 2)
    assert_equal(session.state, SESSION_FAILED)

    var message = String("")
    try:
        _ = session.begin(OP_SUM_F64, 1)
    except e:
        message = String(e)
    assert_true(
        message.find("vanished") >= 0,
        "expected a peer-loss failure, got: " + message,
    )


# ---------------------------------------------------------------------------
# Ordered reduction
# ---------------------------------------------------------------------------


def test_the_reduction_order_is_ascending_and_it_matters() raises:
    """Float addition is not associative, so ascending rank order is a real
    constraint and not a formality. If these two folds were equal the test
    would be proving nothing."""
    var one: Float64 = 1.0
    var tiny: Float64 = 1e-16

    var big: List[Float64] = [one]
    var small_a: List[Float64] = [tiny]
    var small_b: List[Float64] = [tiny]

    var ascending: List[List[Float64]] = [
        big.copy(),
        small_a.copy(),
        small_b.copy(),
    ]
    var descending: List[List[Float64]] = [
        small_a.copy(),
        small_b.copy(),
        big.copy(),
    ]
    var forward = reduce_ordered_f64(ascending)
    var backward = reduce_ordered_f64(descending)
    assert_true(
        f64_bits(forward[0]) != f64_bits(backward[0]),
        "the two orders must not agree, or the test proves nothing",
    )
    assert_equal(f64_bits(forward[0]), f64_bits((one + tiny) + tiny))


def test_reducing_buffers_of_different_lengths_is_refused() raises:
    var a: List[Float64] = [1.0, 2.0]
    var b: List[Float64] = [1.0]
    var pair: List[List[Float64]] = [a.copy(), b.copy()]
    var message = String("")
    try:
        _ = reduce_ordered_f64(pair)
    except e:
        message = String(e)
    assert_true(
        message.find("differ in length") >= 0,
        "expected a length rejection, got: " + message,
    )


# ---------------------------------------------------------------------------
# The protocol end to end, without a driver
# ---------------------------------------------------------------------------


def test_three_ranks_agree_bit_for_bit_through_the_protocol() raises:
    """A whole collective between three ranks in one process, with no threads
    and no endpoints: contributions encoded, folded at the root in rank order,
    and the root's bytes adopted verbatim by the other two.

    Every rank ends with identical bits, which is requirement 1 of
    docs/distributed.md section 5, and the result equals the ascending-order
    reduction, which is requirement 2.
    """
    var s0 = _ready_session(0, 3)
    var s1 = _ready_session(1, 3)
    var s2 = _ready_session(2, 3)

    var b0: List[Float64] = [1.0, 4.0]
    var b1: List[Float64] = [1e-16, 0.5]
    var b2: List[Float64] = [1e-16, 0.25]

    _ = s0.begin(OP_SUM_F64, 2)
    _ = s1.begin(OP_SUM_F64, 2)
    _ = s2.begin(OP_SUM_F64, 2)

    var f1 = contribution_frame_f64(s1, b1)
    var f2 = contribution_frame_f64(s2, b2)

    var acc = b0.copy()
    absorb_f64(s0, acc, f1, 1)
    absorb_f64(s0, acc, f2, 2)
    var answer = result_frame_f64(s0, acc)

    var got1 = adopt_result_f64(s1, answer)
    var got2 = adopt_result_f64(s2, answer)

    var contributions: List[List[Float64]] = [
        b0.copy(),
        b1.copy(),
        b2.copy(),
    ]
    var expected = reduce_ordered_f64(contributions)
    for i in range(2):
        assert_equal(f64_bits(acc[i]), f64_bits(expected[i]))
        assert_equal(f64_bits(got1[i]), f64_bits(expected[i]))
        assert_equal(f64_bits(got2[i]), f64_bits(expected[i]))

    s0.finish()
    s1.finish()
    s2.finish()
    assert_equal(s0.seq, 1)
    assert_equal(s1.seq, 1)
    assert_equal(s2.seq, 1)


# ---------------------------------------------------------------------------
# The blocking driver
# ---------------------------------------------------------------------------


def test_the_root_folds_in_rank_order_and_broadcasts_identical_bytes() raises:
    var s1 = _ready_session(1, 3)
    var s2 = _ready_session(2, 3)
    _ = s1.begin(OP_SUM_F64, 2)
    _ = s2.begin(OP_SUM_F64, 2)
    var c1: List[Float64] = [1.0, 2.0]
    var c2: List[Float64] = [10.0, 20.0]

    var e1 = MemoryEndpoint(1)
    e1.prime(contribution_frame_f64(s1, c1))
    var e2 = MemoryEndpoint(2)
    e2.prime(contribution_frame_f64(s2, c2))

    var peers = List[MemoryEndpoint]()
    peers.append(e1^)
    peers.append(e2^)
    var comm = TransportCollective[MemoryEndpoint](_ready_session(0, 3), peers^)

    var buf: List[Float64] = [100.0, 200.0]
    comm.allreduce_sum_f64(buf)
    assert_equal(buf[0], 111.0)
    assert_equal(buf[1], 222.0)
    assert_equal(comm.world_size(), 3)
    assert_equal(comm.rank(), 0)
    assert_equal(comm.n_local_ranks(), 1)
    assert_equal(comm.local_rank(0), 0)
    assert_equal(comm.session.seq, 1)

    # The identical buffer went to both peers, which is how bit-identical
    # delivery is achieved rather than hoped for.
    assert_equal(len(comm.peers[0].outbox), len(comm.peers[1].outbox))
    for i in range(len(comm.peers[0].outbox)):
        assert_equal(
            Int(comm.peers[0].outbox[i]), Int(comm.peers[1].outbox[i])
        )
    var sent = decode_frame(comm.peers[0].outbox)
    assert_equal(sent.header.msg_type, MSG_RESULT)
    assert_equal(sent.header.sender, 0)
    assert_equal(sent.header.seq, 0)
    var payload = decode_f64_payload(sent.payload)
    assert_equal(payload[0], 111.0)
    assert_equal(payload[1], 222.0)


def test_a_leaf_sends_its_contribution_and_adopts_the_result() raises:
    var root = _ready_session(0, 3)
    _ = root.begin(OP_SUM_F64, 2)
    var reduced: List[Float64] = [7.0, 8.0]
    var e0 = MemoryEndpoint(0)
    e0.prime(result_frame_f64(root, reduced))

    var comm = TransportCollective[MemoryEndpoint](
        _ready_session(1, 3), _one_endpoint(e0^)
    )
    var buf: List[Float64] = [1.0, 2.0]
    comm.allreduce_sum_f64(buf)
    assert_equal(buf[0], 7.0)
    assert_equal(buf[1], 8.0)

    var sent = decode_frame(comm.peers[0].outbox)
    assert_equal(sent.header.msg_type, MSG_CONTRIB)
    assert_equal(sent.header.sender, 1)
    assert_equal(sent.header.op, OP_SUM_F64)
    var payload = decode_f64_payload(sent.payload)
    assert_equal(payload[0], 1.0)
    assert_equal(payload[1], 2.0)


def test_the_barrier_is_a_real_round_trip() raises:
    """Deliberately not free. A barrier that returned immediately would make
    an epoch boundary a point no rank had actually reached."""
    var root = _ready_session(0, 2)
    _ = root.begin(OP_BARRIER, 0)
    var e0 = MemoryEndpoint(0)
    e0.prime(barrier_frame(root, MSG_RESULT))

    var comm = TransportCollective[MemoryEndpoint](
        _ready_session(1, 2), _one_endpoint(e0^)
    )
    comm.barrier()
    assert_equal(comm.session.seq, 1)

    var sent = decode_frame(comm.peers[0].outbox)
    assert_equal(sent.header.op, OP_BARRIER)
    assert_equal(sent.header.msg_type, MSG_CONTRIB)
    assert_equal(sent.header.n_elements, 0)
    assert_equal(len(sent.payload), 0)


def test_the_driver_carries_negatives_through_a_max_reduction() raises:
    """The real agreement protocol, `agree_equal_ints`, run over the
    transport. It reduces every value alongside its negation, so this is both
    the test that a negative survives the wire and the test that the transport
    is substitutable for `LocalCollective` in the code that already exists."""
    var peer = _ready_session(1, 2)
    _ = peer.begin(OP_MAX_INT, 4)
    # The remote rank claims n_features = 9 where this one says 5.
    var claim: List[Int] = [9, -9, 255, -255]
    var e1 = MemoryEndpoint(1)
    e1.prime(contribution_frame_int(peer, claim))

    var comm = TransportCollective[MemoryEndpoint](
        _ready_session(0, 2), _one_endpoint(e1^)
    )
    var values: List[Int] = [5, 255]
    assert_equal(agree_equal_ints(comm, values), 0)

    var sent = decode_frame(comm.peers[0].outbox)
    var payload = decode_int_payload(sent.payload)
    assert_equal(payload[0], 9)
    assert_equal(payload[1], -5)
    assert_equal(payload[2], 255)
    assert_equal(payload[3], -255)


def test_a_peer_that_vanishes_fails_the_driver() raises:
    """The stream ends where a frame should have been, which is a process that
    exited. The session records the loss before the error escapes, so nothing
    can catch it and issue another collective."""
    var e1 = MemoryEndpoint(1)
    var comm = TransportCollective[MemoryEndpoint](
        _ready_session(0, 2), _one_endpoint(e1^)
    )
    var buf: List[Float64] = [1.0]

    var message = String("")
    try:
        comm.allreduce_sum_f64(buf)
    except e:
        message = String(e)
    assert_true(
        message.find("mid-frame") >= 0, "expected a lost peer, got: " + message
    )
    assert_equal(comm.session.lost_rank(), 1)
    assert_equal(comm.session.state, SESSION_FAILED)
    assert_equal(comm.session.failure_code, TRANSPORT_PEER_LOST)


def test_a_peer_that_stops_answering_times_out() raises:
    var e1 = MemoryEndpoint(1)
    e1.stall_after(0)
    var comm = TransportCollective[MemoryEndpoint](
        _ready_session(0, 2), _one_endpoint(e1^)
    )
    var buf: List[Float64] = [1.0]

    var message = String("")
    try:
        comm.allreduce_sum_f64(buf)
    except e:
        message = String(e)
    assert_true(
        message.find("deadline") >= 0, "expected a timeout, got: " + message
    )
    assert_equal(comm.session.state, SESSION_FAILED)


def test_the_wrong_number_of_endpoints_is_refused() raises:
    var e1 = MemoryEndpoint(1)
    var message = String("")
    try:
        _ = TransportCollective[MemoryEndpoint](
            _ready_session(0, 3), _one_endpoint(e1^)
        )
    except e:
        message = String(e)
    assert_true(
        message.find("expected 2 endpoints") >= 0,
        "expected the topology to be checked, got: " + message,
    )


# ---------------------------------------------------------------------------
# Histogram contract and split agreement
# ---------------------------------------------------------------------------


def test_the_histogram_plan_is_the_documented_cost_model() raises:
    """The cost model in docs/distributed.md section 8: 100 features, 255
    bins, three eight-byte statistics per cell, three reductions per node."""
    var plan = histogram_plan(100, 255)
    assert_equal(plan.cells, 25_500)
    assert_equal(plan.reduces_per_node, 3)
    assert_equal(plan.payload_bytes_per_node, 612_000)
    assert_equal(plan.framing_bytes_per_node, 3 * FRAME_HEADER_BYTES)

    check_histogram_buffers(plan, 25_500, 25_500, 25_500)
    var message = String("")
    try:
        check_histogram_buffers(plan, 25_500, 25_500, 10)
    except e:
        message = String(e)
    assert_true(
        message.find("100 features by 255 bins") >= 0,
        "expected the grid in the message, got: " + message,
    )


def test_split_agreement_is_exact_in_the_gain_bits() raises:
    """A tolerance here would hide exactly the divergence the check exists to
    catch, so two gains one bit apart must not agree."""
    var chosen = split_digest(3, 7, 0.5, True)
    var digests: List[UInt64] = [chosen, chosen, chosen]
    assert_equal(check_split_agreement(digests), -1)

    digests[2] = split_digest(3, 8, 0.5, True)
    assert_equal(check_split_agreement(digests), 2)

    var one_bit_higher = f64_from_bits(f64_bits(0.5) + 1)
    assert_true(
        chosen != split_digest(3, 7, one_bit_higher, True),
        "a one-bit gain difference must be visible",
    )
    assert_true(
        chosen != split_digest(3, 7, 0.5, False),
        "found and not found must differ",
    )


# ---------------------------------------------------------------------------
# Checkpoint and restart
# ---------------------------------------------------------------------------


def test_checkpoint_metadata_round_trips() raises:
    var config = _config(0, 4)
    var meta = CheckpointMeta(
        _JOB, config.schema_digest(), UInt64(0xDEADBEEF), 12, 0, 4, 12
    )
    var bytes = encode_checkpoint_meta(meta)
    assert_equal(len(bytes), CHECKPOINT_BYTES)

    var back = decode_checkpoint_meta(bytes)
    assert_equal(back.job_id, _JOB)
    assert_equal(back.schema, config.schema_digest())
    assert_equal(back.model_digest, UInt64(0xDEADBEEF))
    assert_equal(back.epoch, 12)
    assert_equal(back.seq, 0)
    assert_equal(back.world_size, 4)
    assert_equal(back.n_trees, 12)
    assert_equal(restart_status(back, config), TRANSPORT_OK)


def test_a_restart_refuses_a_changed_world_or_schema() raises:
    """Both are hard requirements. World size because the row partition is a
    function of it, so resuming at a different one would silently re-shard the
    data under a half-built model. Schema because the rank assignment lives in
    the machine list, and a rank resuming against a rewritten list would take
    over another rank's shard."""
    var four = _config(0, 4)
    var meta = CheckpointMeta(
        _JOB, four.schema_digest(), UInt64(1), 5, 0, 4, 5
    )

    assert_equal(restart_status(meta, _config(0, 3)), TRANSPORT_WORLD_MISMATCH)

    var rewritten = parse_machine_list(
        "10.0.0.1:9000\n10.0.0.2:9001\n10.0.0.3:9002\n10.0.0.9:9003\n",
        0,
        _JOB,
    )
    assert_equal(restart_status(meta, rewritten), TRANSPORT_SCHEMA_MISMATCH)

    var other_job = parse_machine_list(_machine_list(4), 0, _JOB + 1)
    assert_equal(restart_status(meta, other_job), TRANSPORT_JOB_MISMATCH)


def test_resuming_starts_the_next_epoch_at_sequence_zero() raises:
    """A checkpoint is only ever taken between boosting rounds, so a resumed
    session starts a fresh epoch rather than trying to agree a mid-epoch
    sequence number across ranks that checkpointed one collective apart."""
    var config = _config(1, 2)
    var meta = CheckpointMeta(
        _JOB, config.schema_digest(), UInt64(9), 5, 3, 2, 5
    )
    var session = resume_session(config^, meta)
    assert_equal(session.epoch, 6)
    assert_equal(session.seq, 0)

    var wrong = CheckpointMeta(_JOB, UInt64(0), UInt64(9), 5, 3, 2, 5)
    var message = String("")
    try:
        _ = resume_session(_config(1, 2), wrong)
    except e:
        message = String(e)
    assert_true(
        message.find("run schema") >= 0,
        "expected a schema refusal, got: " + message,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
