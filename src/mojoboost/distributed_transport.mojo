"""Transport and session state machine for multi-process distributed training.

`collective.mojo` says what a collective has to deliver. This module says how
two processes on two hosts would actually agree on it: the wire frame, the
sequence numbering that makes collectives deterministically ordered, the
handshake that refuses a mismatched world, the deadline and cancellation
rules, what happens when a worker disappears, and the metadata a restart needs
to prove it is resuming the same job.

The layering is deliberate and is the whole point of the file:

- `ByteEndpoint` is the only thing that touches bytes on a wire. It is a
  blocking, ordered, reliable stream of bytes to exactly one peer, which is
  what a connected TCP socket is. An MPI or NCCL adapter, if one is ever
  wanted, implements this trait too and nothing above it changes.
- everything above `ByteEndpoint` is pure logic over `List[UInt8]`: framing,
  checksums, the session state machine, the ordered reduction, the handshake,
  the checkpoint record. All of it is exercised in tests with in-memory
  endpoints and a manual clock, so none of it needs a process or a port to be
  tested.
- `TransportCollective` is the thin blocking driver that composes the two and
  conforms to `Collective`, so `train_distributed` can be handed one without a
  line of `distributed.mojo` changing.
- the runtime section at the end is the only seam anything outside Mojo uses.
  A `RuntimeSpec` describes a rank's world and nothing about the data;
  `open_local_collective` and `open_transport_collective` are the only two ways
  a collective is built; and `transport_available` is the single place that
  answers whether another process can be reached, so every caller refuses for
  the same reason with the same message.

What is implemented and tested, and what is not, is stated plainly because the
distinction matters more here than anywhere else in the repository:

**Implemented and tested in process.** Frame encode and decode, checksum
rejection, the per-collective sequence and epoch numbering, out-of-order and
duplicate rejection, the handshake and schema digest, the root gather and
broadcast protocol including bit-identical delivery, ascending rank order
accumulation, deadline expiry, cancellation, worker loss, machine list
parsing, the histogram all-reduce cost contract, split agreement, and
checkpoint restart compatibility.

**Not implemented.** A socket `ByteEndpoint`. Mojo's standard library exposes
no socket module at the version this repository pins, and this repository has
no foreign function interface precedent to borrow, so writing one here would
mean shipping untested syscall bindings under a protocol that is otherwise
fully tested. The boundary is drawn exactly where that adapter lands, and
docs/DISTRIBUTED_TRANSPORT.md section 7 specifies the syscall sequence it owes
and the hermetic two-process test it has to pass.

**Therefore not claimed.** Nothing here has moved a byte between two
processes or two hosts. No multi-host behavior, no throughput, and no
scalability is claimed. `MemoryEndpoint` is a fake, named as one, and is not
a transport. `HAS_BYTE_ENDPOINT` is False and `transport_available` returns
it, so a caller that asks for a multi-process world is refused with
`TRANSPORT_UNAVAILABLE` before it partitions a row. A socket adapter landing
flips that one name, and nothing above it changes.
"""

from std.memory import bitcast
from std.os import getenv
from std.sys.ffi import external_call
from std.sys.info import CompilationTarget
from std.time import perf_counter_ns, sleep

from .collective import (
    Collective,
    LocalCollective,
    add_into_f64,
    add_into_int,
    max_into_int,
    zeros_f64,
    zeros_int,
)

# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------
#
# Deliberately separate from `collective.mojo`'s STATUS_* codes, which
# describe bad training input. These describe a broken session. They are kept
# apart because they are raised by different layers and mixing them would make
# "the peer went away" indistinguishable from "this shard's weights are
# negative".

comptime TRANSPORT_OK = 0
comptime TRANSPORT_TIMEOUT = 1
comptime TRANSPORT_PEER_LOST = 2
comptime TRANSPORT_SCHEMA_MISMATCH = 3
comptime TRANSPORT_WORLD_MISMATCH = 4
comptime TRANSPORT_JOB_MISMATCH = 5
comptime TRANSPORT_OUT_OF_ORDER = 6
comptime TRANSPORT_FRAME_CORRUPT = 7
comptime TRANSPORT_CANCELLED = 8
comptime TRANSPORT_PROTOCOL_STATE = 9
comptime TRANSPORT_CONFIG_INVALID = 10
comptime TRANSPORT_SPLIT_DISAGREEMENT = 11
comptime TRANSPORT_UNAVAILABLE = 12


def transport_status_message(code: Int) -> String:
    """Text for a transport status code.

    Coarse on purpose, for the same reason `collective.status_message` is: a
    code survives a reduction and a sentence does not, so every rank has to be
    able to produce the identical message from the identical code. Detail that
    is genuinely rank-local rides in the `detail` argument of
    `transport_error`, which is only ever used by the rank that raises first.
    """
    if code == TRANSPORT_OK:
        return "no failure"
    if code == TRANSPORT_TIMEOUT:
        return "a collective did not complete before its deadline"
    if code == TRANSPORT_PEER_LOST:
        return "a peer rank closed or vanished mid-session"
    if code == TRANSPORT_SCHEMA_MISMATCH:
        return "ranks disagree about the protocol version or the run schema"
    if code == TRANSPORT_WORLD_MISMATCH:
        return "ranks disagree about the world size or the rank assignment"
    if code == TRANSPORT_JOB_MISMATCH:
        return "a frame belongs to a different job"
    if code == TRANSPORT_OUT_OF_ORDER:
        return "a collective arrived out of its agreed order"
    if code == TRANSPORT_FRAME_CORRUPT:
        return "a frame failed its magic, length, or checksum check"
    if code == TRANSPORT_CANCELLED:
        return "the session was cancelled"
    if code == TRANSPORT_PROTOCOL_STATE:
        return "a collective was called in the wrong session state"
    if code == TRANSPORT_CONFIG_INVALID:
        return "the transport configuration is invalid"
    if code == TRANSPORT_SPLIT_DISAGREEMENT:
        return "ranks chose different splits for the same node"
    if code == TRANSPORT_UNAVAILABLE:
        return "this build cannot reach another process"
    return "unrecognized transport failure code"


def transport_error(code: Int, rank: Int, detail: String) -> Error:
    """The one error shape this module raises.

    Every message names the code's meaning and the rank the failure is
    attributed to, so a log line from any rank identifies both what broke and
    where, and so tests can assert on a stable prefix.
    """
    return Error(
        "distributed transport failed on rank ",
        rank,
        ": ",
        transport_status_message(code),
        " (",
        detail,
        ")",
    )


# ---------------------------------------------------------------------------
# Byte and integer coding
# ---------------------------------------------------------------------------
#
# Little-endian, fixed width, and written with arithmetic rather than shifts so
# the encoding cannot pick up a host word size or a shift-count type by
# accident. These functions are the reason a frame written by one build is
# readable by another.

comptime _BYTE = 256


def _put_uint(mut out: List[UInt8], value: UInt64, n_bytes: Int):
    var v = value
    for _ in range(n_bytes):
        out.append(UInt8(Int(v % _BYTE)))
        v = v // _BYTE


def _get_uint(buf: List[UInt8], offset: Int, n_bytes: Int) raises -> UInt64:
    if offset < 0 or offset + n_bytes > len(buf):
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "read past the end of a frame"
        )
    var out: UInt64 = 0
    for k in range(n_bytes - 1, -1, -1):
        out = out * _BYTE + UInt64(Int(buf[offset + k]))
    return out


def _put_field(
    mut out: List[UInt8], name: String, value: Int, n_bytes: Int
) raises:
    """A header field that has to fit. Raising here rather than truncating is
    what keeps a 70000-rank world size from being received as rank 4464."""
    if value < 0:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "negative header field " + name
        )
    var limit: UInt64 = 1
    for _ in range(n_bytes):
        limit = limit * _BYTE
    if UInt64(value) >= limit:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "header field " + name + " overflows"
        )
    _put_uint(out, UInt64(value), n_bytes)


def zigzag_encode(value: Int) -> UInt64:
    """Map a signed integer onto an unsigned one so negatives survive the
    wire. Needed because `agree_equal_ints` reduces a value alongside its
    negation, so a plain unsigned cast would corrupt half of every
    configuration agreement message."""
    if value >= 0:
        return UInt64(value) * 2
    return UInt64(-(value + 1)) * 2 + 1


def zigzag_decode(value: UInt64) -> Int:
    var half = Int(value // 2)
    if value % 2 == 0:
        return half
    return -half - 1


def fnv1a32(bytes: List[UInt8], start: Int, end: Int) -> UInt64:
    """32-bit FNV-1a over a byte range.

    A checksum, not a MAC. It catches a truncated read, a misaligned frame
    boundary, and a flipped bit. It does not authenticate anything, and this
    protocol has no authentication: see docs/DISTRIBUTED_TRANSPORT.md section
    9 for why that gates the transport to a trusted network.
    """
    var h: UInt64 = 0x811C9DC5
    for i in range(start, end):
        h = h ^ UInt64(Int(bytes[i]))
        h = (h * 0x01000193) % 0x1_0000_0000
    return h


def digest_ints(values: List[Int]) -> UInt64:
    """64-bit FNV-1a over a list of signed integers.

    Deterministic across ranks and across runs, which is the only property
    asked of it: two ranks that were configured identically must produce the
    same digest, and two ranks that were not must almost certainly produce
    different ones.
    """
    var h: UInt64 = 0xCBF29CE484222325
    for i in range(len(values)):
        var z = zigzag_encode(values[i])
        for _ in range(8):
            h = h ^ (z % _BYTE)
            h = h * 0x0000_0100_0000_01B3
            z = z // _BYTE
    return h


def digest_halves(digest: UInt64) -> List[Int]:
    """Split a 64-bit digest into two non-negative `Int` halves.

    The only reduction every `Collective` offers is over `Int`, and a digest
    does not fit in one portably. Two halves do, and both are non-negative, so
    `agree_equal_ints` can compare them the same way it compares a feature
    count: a digest the ranks disagree about shows up as a disagreement about
    one of its halves, which is enough to fail the run and name the field.
    """
    return [Int(digest % 0x1_0000_0000), Int(digest // 0x1_0000_0000)]


def f64_bits(value: Float64) -> UInt64:
    return value.to_bits().cast[DType.uint64]()


def f64_from_bits(bits: UInt64) -> Float64:
    return bitcast[DType.float64, 1](SIMD[DType.uint64, 1](bits))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

comptime TRANSPORT_PROTOCOL_VERSION = 1

comptime ROOT_RANK = 0

# One second in nanoseconds, the unit every timeout in this module is
# expressed in. Deadlines are monotonic durations, never wall clock times, so
# a clock step on one host cannot expire a collective on another.
comptime NS_PER_SECOND = 1_000_000_000


@fieldwise_init
struct RankAddress(Copyable, Movable):
    """Where one rank listens. Purely descriptive: nothing in this module
    resolves or connects to it, and the socket adapter that will is not
    written."""

    var host: String
    var port: Int

    def text(self) -> String:
        return self.host + ":" + String(self.port)


def parse_address(token: String) raises -> RankAddress:
    """Parse one `host:port` entry."""
    var parts = token.split(":")
    if len(parts) != 2:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID,
            -1,
            "expected host:port, got '" + token + "'",
        )
    var host = String(parts[0])
    if host.byte_length() == 0:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID, -1, "empty host in '" + token + "'"
        )
    var port = Int(String(parts[1]))
    if port < 1 or port > 65535:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID,
            -1,
            "port out of range in '" + token + "'",
        )
    return RankAddress(host^, port)


struct TransportConfig(Copyable, Movable):
    """Everything a rank needs to know before it opens a connection.

    `addresses` is indexed by rank, which is the whole rank assignment: rank r
    is whoever is listening at `addresses[r]`. That makes the assignment a
    property of a file every rank reads rather than of the order processes
    happened to start, which is what lets a restart put the same rank ids back
    on the same shards.
    """

    var rank: Int
    var world_size: Int
    var addresses: List[RankAddress]
    var job_id: UInt64
    var connect_timeout_ns: Int
    var collective_timeout_ns: Int

    def __init__(
        out self,
        rank: Int,
        var addresses: List[RankAddress],
        job_id: UInt64,
        connect_timeout_ns: Int = 30 * NS_PER_SECOND,
        collective_timeout_ns: Int = 300 * NS_PER_SECOND,
    ):
        self.rank = rank
        self.world_size = len(addresses)
        self.addresses = addresses^
        self.job_id = job_id
        self.connect_timeout_ns = connect_timeout_ns
        self.collective_timeout_ns = collective_timeout_ns

    def validate(self) raises:
        """Reject a configuration before anything opens a socket.

        Every check here is one that would otherwise surface as a hang or as a
        wrong answer: a duplicate address means two ranks would answer to the
        same id, and a nonpositive timeout means a deadline that has already
        expired when it is set.
        """
        if self.world_size < 1:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID, self.rank, "world size must be > 0"
            )
        if self.rank < 0 or self.rank >= self.world_size:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                self.rank,
                "rank is outside the world",
            )
        if len(self.addresses) != self.world_size:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                self.rank,
                "one address per rank is required",
            )
        if self.connect_timeout_ns <= 0 or self.collective_timeout_ns <= 0:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                self.rank,
                "timeouts must be positive",
            )
        for i in range(self.world_size):
            for j in range(i + 1, self.world_size):
                if (
                    self.addresses[i].port == self.addresses[j].port
                    and self.addresses[i].host == self.addresses[j].host
                ):
                    raise transport_error(
                        TRANSPORT_CONFIG_INVALID,
                        self.rank,
                        "duplicate address " + self.addresses[i].text(),
                    )

    def is_root(self) -> Bool:
        return self.rank == ROOT_RANK

    def schema_digest(self) -> UInt64:
        """The part of the configuration every rank must agree on.

        Addresses are in it, so a rank started against a stale machine list
        fails the handshake instead of silently talking to whoever answers.
        Timeouts are not, because a slower host is allowed to wait longer.
        """
        var values: List[Int] = [
            TRANSPORT_PROTOCOL_VERSION,
            self.world_size,
        ]
        for i in range(len(self.addresses)):
            values.append(self.addresses[i].port)
            for b in self.addresses[i].host.as_bytes():
                values.append(Int(b))
        return digest_ints(values)


def parse_machine_list(
    text: String, rank: Int, job_id: UInt64
) raises -> TransportConfig:
    """Build a config from LightGBM's machine list shape: one `host:port` per
    line, blank lines skipped, `#` starting a comment.

    Rank order is file order, so every rank reading the same file agrees about
    who is who without any of them being asked.
    """
    var addresses = List[RankAddress]()
    for raw_line in text.split("\n"):
        var line = String(raw_line)
        var before_comment = String(line.split("#")[0])
        for token in before_comment.split():
            addresses.append(parse_address(String(token)))
    if len(addresses) == 0:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID, rank, "the machine list is empty"
        )
    var config = TransportConfig(rank, addresses^, job_id)
    config.validate()
    return config^


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------


@fieldwise_init
struct HandshakeRecord(Copyable, Movable):
    """What one rank announces about itself before the first collective."""

    var protocol_version: Int
    var rank: Int
    var world_size: Int
    var job_id: UInt64
    var schema: UInt64
    var restart_epoch: Int


def local_handshake(
    config: TransportConfig, restart_epoch: Int
) -> HandshakeRecord:
    return HandshakeRecord(
        TRANSPORT_PROTOCOL_VERSION,
        config.rank,
        config.world_size,
        config.job_id,
        config.schema_digest(),
        restart_epoch,
    )


def handshake_status(local: HandshakeRecord, peer: HandshakeRecord) -> Int:
    """Compare a peer's announcement to this rank's, returning a transport
    status code.

    Checked in order of how badly the run is broken, so the reported code is
    the most fundamental disagreement rather than the first field to differ.
    A version mismatch is reported as a version mismatch even though it also
    implies a schema mismatch.
    """
    if peer.protocol_version != local.protocol_version:
        return TRANSPORT_SCHEMA_MISMATCH
    if peer.job_id != local.job_id:
        return TRANSPORT_JOB_MISMATCH
    if peer.world_size != local.world_size:
        return TRANSPORT_WORLD_MISMATCH
    if peer.rank < 0 or peer.rank >= local.world_size:
        return TRANSPORT_WORLD_MISMATCH
    if peer.rank == local.rank:
        return TRANSPORT_WORLD_MISMATCH
    if peer.schema != local.schema:
        return TRANSPORT_SCHEMA_MISMATCH
    if peer.restart_epoch != local.restart_epoch:
        return TRANSPORT_SCHEMA_MISMATCH
    return TRANSPORT_OK


def encode_handshake(record: HandshakeRecord) raises -> List[UInt8]:
    var out = List[UInt8]()
    _put_field(out, "protocol_version", record.protocol_version, 2)
    _put_field(out, "rank", record.rank, 4)
    _put_field(out, "world_size", record.world_size, 4)
    _put_uint(out, record.job_id, 8)
    _put_uint(out, record.schema, 8)
    _put_field(out, "restart_epoch", record.restart_epoch, 4)
    return out^


comptime HANDSHAKE_BYTES = 30


def decode_handshake(buf: List[UInt8]) raises -> HandshakeRecord:
    if len(buf) != HANDSHAKE_BYTES:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "handshake payload has a wrong length"
        )
    return HandshakeRecord(
        Int(_get_uint(buf, 0, 2)),
        Int(_get_uint(buf, 2, 4)),
        Int(_get_uint(buf, 6, 4)),
        _get_uint(buf, 10, 8),
        _get_uint(buf, 18, 8),
        Int(_get_uint(buf, 26, 4)),
    )


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

comptime FRAME_MAGIC = UInt64(0x4D4A4254)  # "MJBT"

# Byte layout, little-endian throughout:
#
#   0  magic        u32
#   4  version      u16
#   6  msg_type     u16
#   8  job_id       u64
#  16  epoch        u32
#  20  seq          u32
#  24  sender       u32
#  28  op           u16
#  30  n_elements   u32
#  34  payload_len  u32
#  38  checksum     u32
comptime FRAME_HEADER_BYTES = 42

# A ceiling on a single frame, so a corrupted length field asks for a bounded
# allocation instead of an unbounded one. 64 MiB of payload is eight million
# doubles, which is a 100-feature 255-bin histogram more than three hundred
# times over.
comptime MAX_PAYLOAD_BYTES = 64 * 1024 * 1024

comptime MSG_HELLO = 1
comptime MSG_CONTRIB = 2
comptime MSG_RESULT = 3
comptime MSG_ABORT = 4

comptime OP_SUM_F64 = 1
comptime OP_SUM_INT = 2
comptime OP_MAX_INT = 3
comptime OP_BARRIER = 4


def op_element_bytes(op: Int) raises -> Int:
    """Payload bytes per element for a reduction op. A barrier carries no
    payload, which is why it is an op here rather than a special case
    everywhere else."""
    if op == OP_SUM_F64 or op == OP_SUM_INT or op == OP_MAX_INT:
        return 8
    if op == OP_BARRIER:
        return 0
    raise transport_error(
        TRANSPORT_FRAME_CORRUPT, -1, "unknown reduction op " + String(op)
    )


def op_name(op: Int) -> String:
    if op == OP_SUM_F64:
        return "sum_f64"
    if op == OP_SUM_INT:
        return "sum_int"
    if op == OP_MAX_INT:
        return "max_int"
    if op == OP_BARRIER:
        return "barrier"
    return "unknown"


@fieldwise_init
struct FrameHeader(Copyable, Movable):
    var version: Int
    var msg_type: Int
    var job_id: UInt64
    var epoch: Int
    var seq: Int
    var sender: Int
    var op: Int
    var n_elements: Int


@fieldwise_init
struct Frame(Copyable, Movable):
    var header: FrameHeader
    var payload: List[UInt8]


def _checksum_over(header_bytes: List[UInt8], payload: List[UInt8]) -> UInt64:
    """Checksum the header fields that precede the checksum, then the payload.

    It covers the header because a frame whose length field was corrupted in
    flight would otherwise be detected only by the next frame failing its
    magic check, which reports the corruption one frame too late and on the
    wrong frame.
    """
    var h = fnv1a32(header_bytes, 0, 38)
    var combined = List[UInt8](capacity=4 + len(payload))
    _put_uint(combined, h, 4)
    for i in range(len(payload)):
        combined.append(payload[i])
    return fnv1a32(combined, 0, len(combined))


def encode_frame(
    header: FrameHeader, payload: List[UInt8]
) raises -> List[UInt8]:
    """Serialize one frame.

    The declared element count and the actual payload length are checked
    against each other here, at the only place that can still tell the truth
    about both, so a length disagreement is a local error on the sender rather
    than a corrupt-frame error on the receiver.
    """
    var element_bytes = op_element_bytes(header.op)
    if len(payload) != header.n_elements * element_bytes:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT,
            header.sender,
            "payload length disagrees with the declared element count",
        )
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, header.sender, "payload exceeds the cap"
        )

    var head = List[UInt8](capacity=FRAME_HEADER_BYTES)
    _put_uint(head, FRAME_MAGIC, 4)
    _put_field(head, "version", header.version, 2)
    _put_field(head, "msg_type", header.msg_type, 2)
    _put_uint(head, header.job_id, 8)
    _put_field(head, "epoch", header.epoch, 4)
    _put_field(head, "seq", header.seq, 4)
    _put_field(head, "sender", header.sender, 4)
    _put_field(head, "op", header.op, 2)
    _put_field(head, "n_elements", header.n_elements, 4)
    _put_field(head, "payload_len", len(payload), 4)
    _put_uint(head, _checksum_over(head, payload), 4)

    var out = List[UInt8](capacity=FRAME_HEADER_BYTES + len(payload))
    for i in range(len(head)):
        out.append(head[i])
    for i in range(len(payload)):
        out.append(payload[i])
    return out^


def frame_payload_len(head: List[UInt8]) raises -> Int:
    """The payload length a header declares, checked before anything reads
    that many bytes. A reader calls this between reading the header and
    reading the body, which is the only moment it can bound the read."""
    if len(head) < FRAME_HEADER_BYTES:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "short frame header"
        )
    if _get_uint(head, 0, 4) != FRAME_MAGIC:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "frame magic does not match"
        )
    var declared = Int(_get_uint(head, 34, 4))
    if declared > MAX_PAYLOAD_BYTES:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT,
            -1,
            "declared payload length exceeds the cap",
        )
    return declared


def decode_frame(buf: List[UInt8]) raises -> Frame:
    """Parse a complete frame, verifying magic, version, lengths, and
    checksum before any field is believed."""
    var declared = frame_payload_len(buf)
    if len(buf) != FRAME_HEADER_BYTES + declared:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "frame length disagrees with header"
        )

    var head = List[UInt8](capacity=FRAME_HEADER_BYTES)
    for i in range(FRAME_HEADER_BYTES):
        head.append(buf[i])
    var payload = List[UInt8](capacity=declared)
    for i in range(declared):
        payload.append(buf[FRAME_HEADER_BYTES + i])

    if _checksum_over(head, payload) != _get_uint(buf, 38, 4):
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "frame checksum does not match"
        )

    var version = Int(_get_uint(buf, 4, 2))
    if version != TRANSPORT_PROTOCOL_VERSION:
        raise transport_error(
            TRANSPORT_SCHEMA_MISMATCH,
            Int(_get_uint(buf, 24, 4)),
            "frame protocol version " + String(version),
        )

    var op = Int(_get_uint(buf, 28, 2))
    var n_elements = Int(_get_uint(buf, 30, 4))
    if declared != n_elements * op_element_bytes(op):
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT,
            Int(_get_uint(buf, 24, 4)),
            "payload length disagrees with the declared element count",
        )

    return Frame(
        FrameHeader(
            version,
            Int(_get_uint(buf, 6, 2)),
            _get_uint(buf, 8, 8),
            Int(_get_uint(buf, 16, 4)),
            Int(_get_uint(buf, 20, 4)),
            Int(_get_uint(buf, 24, 4)),
            op,
            n_elements,
        ),
        payload^,
    )


def encode_f64_payload(buf: List[Float64]) -> List[UInt8]:
    var out = List[UInt8](capacity=8 * len(buf))
    for i in range(len(buf)):
        _put_uint(out, f64_bits(buf[i]), 8)
    return out^


def decode_f64_payload(payload: List[UInt8]) raises -> List[Float64]:
    if len(payload) % 8 != 0:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "f64 payload is not a multiple of 8"
        )
    var out = List[Float64](capacity=len(payload) // 8)
    for i in range(len(payload) // 8):
        out.append(f64_from_bits(_get_uint(payload, 8 * i, 8)))
    return out^


def encode_int_payload(buf: List[Int]) -> List[UInt8]:
    var out = List[UInt8](capacity=8 * len(buf))
    for i in range(len(buf)):
        _put_uint(out, zigzag_encode(buf[i]), 8)
    return out^


def decode_int_payload(payload: List[UInt8]) raises -> List[Int]:
    if len(payload) % 8 != 0:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "int payload is not a multiple of 8"
        )
    var out = List[Int](capacity=len(payload) // 8)
    for i in range(len(payload) // 8):
        out.append(zigzag_decode(_get_uint(payload, 8 * i, 8)))
    return out^


# ---------------------------------------------------------------------------
# Session state machine
# ---------------------------------------------------------------------------

comptime SESSION_INIT = 0
comptime SESSION_READY = 1
comptime SESSION_IN_FLIGHT = 2
comptime SESSION_FAILED = 3
comptime SESSION_CANCELLED = 4
comptime SESSION_CLOSED = 5


def session_state_name(state: Int) -> String:
    if state == SESSION_INIT:
        return "init"
    if state == SESSION_READY:
        return "ready"
    if state == SESSION_IN_FLIGHT:
        return "in_flight"
    if state == SESSION_FAILED:
        return "failed"
    if state == SESSION_CANCELLED:
        return "cancelled"
    if state == SESSION_CLOSED:
        return "closed"
    return "unknown"


struct TransportSession(Movable):
    """The per-rank state machine every collective goes through.

    It owns four things that the reduction itself does not:

    **Ordering.** `epoch` and `seq` number the collectives. `seq` increments
    once per collective and never repeats within an epoch, so a frame that
    arrives with the wrong pair is rejected rather than reduced. Requirement 5
    of docs/distributed.md section 5, no reordering across calls, is this
    field: a histogram that overtakes its predecessor does not silently sum
    into the wrong node, it fails the run.

    **Deadlines.** Every collective gets one, from a monotonic clock and never
    from wall clock time, so a clock step on one host cannot expire a
    collective on another. Tests drive `manual_clock` instead, which is what
    makes timeout behavior testable without waiting.

    **Termination.** `FAILED` and `CANCELLED` are sticky. Once a session
    leaves the ready and in-flight pair it never returns, so a rank that has
    given up cannot rejoin a collective the others are mid-way through and
    contribute stale data. This is requirement 4, fail-stop and not partial,
    expressed as a state machine rather than as a convention.

    **Membership.** `alive` records which ranks are still present. Losing one
    is fatal by design: there is no re-partition and no recovery here, only a
    single agreed failure. Section 7 of docs/distributed.md is explicit that
    fault tolerance is a separate project, and this field exists to fail
    honestly rather than to enable one.
    """

    var config: TransportConfig
    var state: Int
    var epoch: Int
    var seq: Int
    var pending_op: Int
    var pending_n: Int
    var deadline_ns: Int
    var alive: List[Bool]
    var failure_code: Int
    var failure_rank: Int
    var manual_clock: Bool
    var clock_ns: Int
    var collectives_completed: Int
    var elements_reduced: Int

    def __init__(out self, var config: TransportConfig) raises:
        config.validate()
        var alive = List[Bool](capacity=config.world_size)
        alive.resize(config.world_size, True)
        self.config = config^
        self.state = SESSION_INIT
        self.epoch = 0
        self.seq = 0
        self.pending_op = 0
        self.pending_n = 0
        self.deadline_ns = 0
        self.alive = alive^
        self.failure_code = TRANSPORT_OK
        self.failure_rank = -1
        self.manual_clock = False
        self.clock_ns = 0
        self.collectives_completed = 0
        self.elements_reduced = 0

    def now_ns(self) -> Int:
        return self.clock_ns if self.manual_clock else Int(perf_counter_ns())

    def use_manual_clock(mut self, start_ns: Int):
        """Drive time by hand. Tests use this so deadline behavior is
        deterministic and instant; a real run never calls it."""
        self.manual_clock = True
        self.clock_ns = start_ns

    def advance_clock(mut self, delta_ns: Int):
        self.clock_ns += delta_ns

    def _fail(mut self, code: Int, rank: Int, detail: String) -> Error:
        """Record a terminal failure and return the error to raise. Recording
        first is what makes the failure sticky even if the caller swallows the
        error."""
        if self.state != SESSION_CANCELLED:
            self.state = SESSION_FAILED
        if self.failure_code == TRANSPORT_OK:
            self.failure_code = code
            self.failure_rank = rank
        return transport_error(code, rank, detail)

    def _require_usable(self) raises:
        if self.state == SESSION_CANCELLED:
            raise transport_error(
                TRANSPORT_CANCELLED, self.config.rank, "session cancelled"
            )
        if self.state == SESSION_FAILED:
            raise transport_error(
                self.failure_code,
                self.failure_rank,
                "session already failed",
            )
        if self.state == SESSION_CLOSED:
            raise transport_error(
                TRANSPORT_PROTOCOL_STATE, self.config.rank, "session closed"
            )

    def complete_handshake(mut self, peers: List[HandshakeRecord]) raises:
        """Move from `INIT` to `READY` once every other rank has announced a
        compatible world.

        A world is compatible only if all `world_size - 1` peers are present
        and distinct, which is checked here rather than at connect time
        because a peer that connects and then announces rank 2 twice is
        exactly the failure this catches.
        """
        self._require_usable()
        if self.state != SESSION_INIT:
            raise self._fail(
                TRANSPORT_PROTOCOL_STATE,
                self.config.rank,
                "handshake in state " + session_state_name(self.state),
            )
        var local = local_handshake(self.config, self.epoch)
        if len(peers) != self.config.world_size - 1:
            raise self._fail(
                TRANSPORT_WORLD_MISMATCH,
                self.config.rank,
                "expected one handshake per peer",
            )
        var seen = List[Bool](capacity=self.config.world_size)
        seen.resize(self.config.world_size, False)
        seen[self.config.rank] = True
        for i in range(len(peers)):
            var status = handshake_status(local, peers[i])
            if status != TRANSPORT_OK:
                raise self._fail(
                    status, peers[i].rank, "peer handshake was rejected"
                )
            if seen[peers[i].rank]:
                raise self._fail(
                    TRANSPORT_WORLD_MISMATCH,
                    peers[i].rank,
                    "two peers claim the same rank",
                )
            seen[peers[i].rank] = True
        self.state = SESSION_READY

    def begin(mut self, op: Int, n_elements: Int) raises -> FrameHeader:
        """Open a collective and return the header every frame in it carries.

        Also where the deadline is set, so the clock starts when the rank
        enters the collective rather than when it first blocks on a peer.
        """
        self._require_usable()
        if self.state != SESSION_READY:
            raise self._fail(
                TRANSPORT_PROTOCOL_STATE,
                self.config.rank,
                "begin in state " + session_state_name(self.state),
            )
        _ = op_element_bytes(op)
        if n_elements < 0:
            raise self._fail(
                TRANSPORT_PROTOCOL_STATE,
                self.config.rank,
                "negative element count",
            )
        if op == OP_BARRIER and n_elements != 0:
            raise self._fail(
                TRANSPORT_PROTOCOL_STATE,
                self.config.rank,
                "a barrier carries no elements",
            )
        var lost = self.lost_rank()
        if lost >= 0:
            raise self._fail(
                TRANSPORT_PEER_LOST,
                lost,
                "a collective cannot start with a rank missing",
            )
        self.state = SESSION_IN_FLIGHT
        self.pending_op = op
        self.pending_n = n_elements
        self.deadline_ns = self.now_ns() + self.config.collective_timeout_ns
        return self.header(MSG_CONTRIB)

    def header(self, msg_type: Int) -> FrameHeader:
        """The header for the collective currently in flight."""
        return FrameHeader(
            TRANSPORT_PROTOCOL_VERSION,
            msg_type,
            self.config.job_id,
            self.epoch,
            self.seq,
            self.config.rank,
            self.pending_op,
            self.pending_n,
        )

    def check_deadline(mut self) raises:
        """Called between blocking steps. Expiry is terminal: a collective
        that timed out cannot be retried, because the peers that did answer
        have already moved their sequence numbers on."""
        if self.now_ns() > self.deadline_ns:
            raise self._fail(
                TRANSPORT_TIMEOUT,
                self.config.rank,
                "collective " + String(self.seq) + " passed its deadline",
            )

    def validate(
        mut self, header: FrameHeader, expected_msg: Int, expected_sender: Int
    ) raises:
        """Accept or reject an incoming frame for the collective in flight.

        Every field is checked against what this rank is already committed to,
        so a frame from another epoch, another job, another collective, or
        another sender is a failure and never a reduction.
        """
        self._require_usable()
        if self.state != SESSION_IN_FLIGHT:
            raise self._fail(
                TRANSPORT_PROTOCOL_STATE,
                self.config.rank,
                "frame with no collective in flight",
            )
        if header.job_id != self.config.job_id:
            raise self._fail(
                TRANSPORT_JOB_MISMATCH,
                header.sender,
                "frame carries a foreign job id",
            )
        if header.version != TRANSPORT_PROTOCOL_VERSION:
            raise self._fail(
                TRANSPORT_SCHEMA_MISMATCH,
                header.sender,
                "frame carries protocol version " + String(header.version),
            )
        if header.sender != expected_sender:
            raise self._fail(
                TRANSPORT_OUT_OF_ORDER,
                header.sender,
                "expected rank " + String(expected_sender),
            )
        if header.epoch != self.epoch or header.seq != self.seq:
            raise self._fail(
                TRANSPORT_OUT_OF_ORDER,
                header.sender,
                "expected epoch "
                + String(self.epoch)
                + " sequence "
                + String(self.seq)
                + ", got epoch "
                + String(header.epoch)
                + " sequence "
                + String(header.seq),
            )
        if header.msg_type != expected_msg:
            raise self._fail(
                TRANSPORT_OUT_OF_ORDER,
                header.sender,
                "unexpected message type " + String(header.msg_type),
            )
        if header.op != self.pending_op:
            raise self._fail(
                TRANSPORT_OUT_OF_ORDER,
                header.sender,
                "expected op "
                + op_name(self.pending_op)
                + ", got "
                + op_name(header.op),
            )
        if header.n_elements != self.pending_n:
            raise self._fail(
                TRANSPORT_SCHEMA_MISMATCH,
                header.sender,
                "ranks disagree about the buffer length",
            )
        if not self.alive[header.sender]:
            raise self._fail(
                TRANSPORT_PEER_LOST,
                header.sender,
                "a frame arrived from a rank already declared lost",
            )

    def finish(mut self) raises:
        """Close the collective and advance the sequence. Counters here are
        the same instrumentation `LocalCollective` carries, so the cost model
        in docs/distributed.md section 8 is testable over the wire format
        too."""
        if self.state != SESSION_IN_FLIGHT:
            raise self._fail(
                TRANSPORT_PROTOCOL_STATE,
                self.config.rank,
                "finish with no collective in flight",
            )
        self.collectives_completed += 1
        self.elements_reduced += self.pending_n
        self.seq += 1
        self.pending_op = 0
        self.pending_n = 0
        self.state = SESSION_READY

    def next_epoch(mut self) raises:
        """Advance to the next checkpoint epoch and restart sequence
        numbering. Only legal between collectives, because an epoch boundary
        inside one would leave the ranks numbering the same collective
        differently."""
        self._require_usable()
        if self.state != SESSION_READY:
            raise self._fail(
                TRANSPORT_PROTOCOL_STATE,
                self.config.rank,
                "epoch boundary inside a collective",
            )
        self.epoch += 1
        self.seq = 0

    def cancel(mut self, detail: String):
        """Stop the session on purpose. Sticky, and distinct from a failure so
        an operator-requested stop is not reported as a broken peer."""
        self.state = SESSION_CANCELLED
        if self.failure_code == TRANSPORT_OK:
            self.failure_code = TRANSPORT_CANCELLED
            self.failure_rank = self.config.rank
        _ = detail

    def mark_lost(mut self, rank: Int) raises:
        """Declare a rank gone. Terminal for the session by design: there is
        no re-partition and no recovery, only one agreed failure."""
        if rank < 0 or rank >= self.config.world_size:
            raise transport_error(
                TRANSPORT_WORLD_MISMATCH,
                self.config.rank,
                "rank " + String(rank) + " is outside the world",
            )
        self.alive[rank] = False
        if self.failure_code == TRANSPORT_OK:
            self.failure_code = TRANSPORT_PEER_LOST
            self.failure_rank = rank
        if self.state != SESSION_CANCELLED:
            self.state = SESSION_FAILED

    def lost_rank(self) -> Int:
        """The lowest-numbered missing rank, or -1. Lowest so every rank names
        the same one, exactly as `agree_status` does."""
        for r in range(self.config.world_size):
            if not self.alive[r]:
                return r
        return -1

    def close(mut self):
        if self.state == SESSION_READY or self.state == SESSION_INIT:
            self.state = SESSION_CLOSED


# ---------------------------------------------------------------------------
# Ordered reduction
# ---------------------------------------------------------------------------
#
# Requirement 2 of docs/distributed.md section 5, ascending rank order, lives
# in these three functions and nowhere else. The root folds contributions in
# rank order and broadcasts the resulting bytes verbatim, which is also
# requirement 1: every rank ends up holding the root's bits, not its own
# rounding of the same sum.


def reduce_into_f64(mut acc: List[Float64], src: List[Float64]) raises:
    """Fold one contribution into the root's accumulator.

    The arithmetic is `collective.add_into_f64`, not a second copy of it: a
    buffer combined across processes and a buffer combined across the local
    ranks of one process go through the same loop, which is what makes the
    ascending-rank-order requirement one claim rather than two. What this adds
    is the transport's error shape, so a length disagreement between two
    processes is reported as a schema mismatch with a rank attached instead of
    as a bare accumulator error.
    """
    if len(acc) != len(src):
        raise transport_error(
            TRANSPORT_SCHEMA_MISMATCH, -1, "reduction operands differ in length"
        )
    add_into_f64(acc, src)


def reduce_into_int(mut acc: List[Int], src: List[Int], op: Int) raises:
    """The integer reductions, dispatched by op code to the same two loops
    `agree_status` and the local-rank accumulation use."""
    if len(acc) != len(src):
        raise transport_error(
            TRANSPORT_SCHEMA_MISMATCH, -1, "reduction operands differ in length"
        )
    if op == OP_SUM_INT:
        add_into_int(acc, src)
    elif op == OP_MAX_INT:
        max_into_int(acc, src)
    else:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "not an integer reduction op"
        )


def reduce_ordered_f64(
    contributions: List[List[Float64]]
) raises -> List[Float64]:
    """Fold per-rank contributions in ascending rank order.

    This is the reference the root's incremental loop has to match, and the
    only definition of what the sum means: `((c_0 + c_1) + c_2) + ...`, in
    that association, on every rank and every run.
    """
    if len(contributions) == 0:
        raise transport_error(
            TRANSPORT_SCHEMA_MISMATCH, -1, "no contributions to reduce"
        )
    var acc = contributions[0].copy()
    for r in range(1, len(contributions)):
        reduce_into_f64(acc, contributions[r])
    return acc^


def reduce_ordered_int(
    contributions: List[List[Int]], op: Int
) raises -> List[Int]:
    if len(contributions) == 0:
        raise transport_error(
            TRANSPORT_SCHEMA_MISMATCH, -1, "no contributions to reduce"
        )
    var acc = contributions[0].copy()
    for r in range(1, len(contributions)):
        reduce_into_int(acc, contributions[r], op)
    return acc^


# ---------------------------------------------------------------------------
# Protocol steps
# ---------------------------------------------------------------------------
#
# The gather-and-broadcast collective, expressed as pure functions over byte
# buffers. They are separated from the blocking driver below so the protocol
# can be run end to end between several ranks inside one test, with no
# threads, no processes, and no sockets. The driver is then a thin composition
# of these with an endpoint, and the only thing left untested by the pure
# tests is the endpoint itself.


def contribution_frame_f64(
    mut session: TransportSession, buf: List[Float64]
) raises -> List[UInt8]:
    """A non-root rank's contribution to a float reduction."""
    return encode_frame(session.header(MSG_CONTRIB), encode_f64_payload(buf))


def contribution_frame_int(
    mut session: TransportSession, buf: List[Int]
) raises -> List[UInt8]:
    return encode_frame(session.header(MSG_CONTRIB), encode_int_payload(buf))


def barrier_frame(
    mut session: TransportSession, msg_type: Int
) raises -> List[UInt8]:
    return encode_frame(session.header(msg_type), List[UInt8]())


def absorb_f64(
    mut session: TransportSession,
    mut acc: List[Float64],
    frame_bytes: List[UInt8],
    expected_sender: Int,
) raises:
    """Fold one peer's contribution into the root's accumulator.

    The root calls this once per peer in ascending rank order, which is what
    makes the incremental loop equal to `reduce_ordered_f64` over the same
    contributions.
    """
    var frame = decode_frame(frame_bytes)
    session.validate(frame.header, MSG_CONTRIB, expected_sender)
    reduce_into_f64(acc, decode_f64_payload(frame.payload))


def absorb_int(
    mut session: TransportSession,
    mut acc: List[Int],
    frame_bytes: List[UInt8],
    expected_sender: Int,
) raises:
    var frame = decode_frame(frame_bytes)
    session.validate(frame.header, MSG_CONTRIB, expected_sender)
    reduce_into_int(acc, decode_int_payload(frame.payload), frame.header.op)


def result_frame_f64(
    mut session: TransportSession, acc: List[Float64]
) raises -> List[UInt8]:
    """The root's answer. One byte string, sent verbatim to every rank, which
    is how bit-identical delivery is achieved rather than hoped for."""
    return encode_frame(session.header(MSG_RESULT), encode_f64_payload(acc))


def result_frame_int(
    mut session: TransportSession, acc: List[Int]
) raises -> List[UInt8]:
    return encode_frame(session.header(MSG_RESULT), encode_int_payload(acc))


def adopt_result_f64(
    mut session: TransportSession, frame_bytes: List[UInt8]
) raises -> List[Float64]:
    var frame = decode_frame(frame_bytes)
    session.validate(frame.header, MSG_RESULT, ROOT_RANK)
    return decode_f64_payload(frame.payload)


def adopt_result_int(
    mut session: TransportSession, frame_bytes: List[UInt8]
) raises -> List[Int]:
    var frame = decode_frame(frame_bytes)
    session.validate(frame.header, MSG_RESULT, ROOT_RANK)
    return decode_int_payload(frame.payload)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


trait ByteEndpoint:
    """A blocking, ordered, reliable byte stream to exactly one peer.

    This is the entire surface a real transport has to provide, and it is
    deliberately smaller than the collective contract: a connected TCP socket
    satisfies it with `send`, `recv`, `close`, and `SO_RCVTIMEO`. MPI and NCCL
    adapters, if either is ever wanted, implement this and nothing above it
    changes.

    Three requirements, all of which the protocol above depends on:

    - `send_all` writes every byte or raises. A short write that is reported
      as success desynchronizes the stream permanently.
    - `recv_exact` returns exactly `n` bytes or raises. Returning fewer would
      make a frame boundary a guess.
    - bytes arrive in the order they were sent, with none lost, duplicated, or
      reordered. Framing detects violations but cannot repair them.

    `deadline_ns` is an absolute monotonic deadline, not a duration, so the
    same value can be handed to the header read and the body read without the
    body silently getting a fresh timeout.
    """

    def peer_rank(self) -> Int:
        """The rank on the other end."""
        ...

    def is_open(self) -> Bool:
        ...

    def send_all(
        mut self, bytes: List[UInt8], deadline_ns: Int
    ) raises:
        """Write every byte or raise."""
        ...

    def recv_exact(
        mut self, n: Int, deadline_ns: Int
    ) raises -> List[UInt8]:
        """Read exactly `n` bytes or raise."""
        ...

    def close(mut self) raises:
        ...


struct MemoryEndpoint(ByteEndpoint, Copyable, Movable):
    """An in-memory fake endpoint. Not a transport, and not a network.

    It exists so the protocol above can be exercised without a process or a
    port, and so the failure modes a real endpoint has can be produced on
    demand instead of waited for:

    - `stall_at` makes `recv_exact` raise a timeout once that many bytes have
      been consumed, which is a peer that stopped answering
    - closing it makes `recv_exact` raise peer-lost, which is a peer that
      exited
    - an inbox shorter than the requested read raises peer-lost too, which is
      a connection cut mid-frame

    `outbox` keeps everything written, so a test can assert on the exact bytes
    a rank put on the wire, which is how bit-identical delivery is checked.
    """

    var peer: Int
    var inbox: List[UInt8]
    var read_pos: Int
    var outbox: List[UInt8]
    var live: Bool
    var stall_at: Int

    def __init__(out self, peer: Int):
        self.peer = peer
        self.inbox = List[UInt8]()
        self.read_pos = 0
        self.outbox = List[UInt8]()
        self.live = True
        self.stall_at = -1

    def prime(mut self, bytes: List[UInt8]):
        """Append bytes the peer is pretending to have sent."""
        for i in range(len(bytes)):
            self.inbox.append(bytes[i])

    def stall_after(mut self, n_bytes: Int):
        self.stall_at = n_bytes

    def peer_rank(self) -> Int:
        return self.peer

    def is_open(self) -> Bool:
        return self.live

    def send_all(mut self, bytes: List[UInt8], deadline_ns: Int) raises:
        _ = deadline_ns
        if not self.live:
            raise transport_error(
                TRANSPORT_PEER_LOST, self.peer, "write to a closed endpoint"
            )
        for i in range(len(bytes)):
            self.outbox.append(bytes[i])

    def recv_exact(mut self, n: Int, deadline_ns: Int) raises -> List[UInt8]:
        _ = deadline_ns
        if not self.live:
            raise transport_error(
                TRANSPORT_PEER_LOST, self.peer, "read from a closed endpoint"
            )
        if self.stall_at >= 0 and self.read_pos >= self.stall_at:
            raise transport_error(
                TRANSPORT_TIMEOUT, self.peer, "no bytes before the deadline"
            )
        if self.read_pos + n > len(self.inbox):
            raise transport_error(
                TRANSPORT_PEER_LOST, self.peer, "stream ended mid-frame"
            )
        var out = List[UInt8](capacity=n)
        for i in range(n):
            out.append(self.inbox[self.read_pos + i])
        self.read_pos += n
        return out^

    def close(mut self) raises:
        self.live = False


def recv_frame_bytes[
    E: ByteEndpoint
](mut endpoint: E, deadline_ns: Int) raises -> List[UInt8]:
    """Read one complete frame off an endpoint.

    Two reads: the fixed header, then a body whose length the header declares
    and `frame_payload_len` has already bounded. Bounding before allocating is
    the point of the split.
    """
    var head = endpoint.recv_exact(FRAME_HEADER_BYTES, deadline_ns)
    var payload_len = frame_payload_len(head)
    var out = List[UInt8](capacity=FRAME_HEADER_BYTES + payload_len)
    for i in range(len(head)):
        out.append(head[i])
    if payload_len > 0:
        var body = endpoint.recv_exact(payload_len, deadline_ns)
        for i in range(len(body)):
            out.append(body[i])
    return out^


# ---------------------------------------------------------------------------
# The socket endpoint: BSD sockets over libc, by FFI
# ---------------------------------------------------------------------------
#
# This is the one part of the file that leaves Mojo. Everything above it is
# pure logic over `List[UInt8]` and is exercised in process; everything here
# is a direct `external_call` into libc and cannot be exercised without two
# processes and a port.
#
# The C ABI it is written against is documented rather than discovered,
# because the alternative was to leave the transport unopenable. What that
# means concretely, and what a reader must not assume:
#
# - the constant tables below are transcribed from the macOS and Linux
#   headers, per platform, and are `comptime` so the wrong platform's values
#   are never compiled in. Only macOS and Linux are supported;
#   `SOCKETS_SUPPORTED` is False anywhere else and every entry point refuses.
# - `sockaddr_in` differs between the two: macOS leads with a `sin_len` byte,
#   Linux does not. Both layouts are written by `_sockaddr_in`.
# - the integer fields are written little-endian by hand, which is correct on
#   every platform this project targets (x86-64 and arm64) and is stated here
#   rather than assumed silently. The two port and address fields are network
#   byte order and are written big-endian for the same reason.
# - hostnames are not resolved. `getaddrinfo` returns a struct whose field
#   order differs between the two platforms, and getting that wrong is a
#   silent wrong-address rather than a compile error, so a machine list must
#   carry dotted-quad IPv4 (or `localhost`) and is refused with a clear
#   message otherwise.
# - deadlines are real: every send and every receive arms `SO_SNDTIMEO` or
#   `SO_RCVTIMEO` with the time actually left before the caller's absolute
#   deadline, so a peer that stops answering surfaces as `TRANSPORT_TIMEOUT`
#   rather than as a hang. Connect is a bounded retry loop against the same
#   deadline, because a worker generally starts before the root is listening.
# - `SIGPIPE` is suppressed (`MSG_NOSIGNAL` on Linux, `SO_NOSIGPIPE` on
#   macOS) so a peer that exits mid-run is reported as worker loss instead of
#   killing this process.
#
# None of this has been compiled or run in this repository. See
# `transport_validated`.

comptime _IS_MACOS = CompilationTarget.is_macos()
comptime _IS_LINUX = CompilationTarget.is_linux()

comptime SOCKETS_SUPPORTED = _IS_MACOS or _IS_LINUX
"""Whether a socket `ByteEndpoint` is compiled in on this platform."""

comptime AF_INET = 2
comptime SOCK_STREAM = 1
comptime IPPROTO_TCP = 6
comptime TCP_NODELAY = 1
comptime SOCKADDR_IN_BYTES = 16
comptime TIMEVAL_BYTES = 16
comptime LISTEN_BACKLOG = 128

comptime SOL_SOCKET = 0xFFFF if _IS_MACOS else 1
comptime SO_REUSEADDR = 0x0004 if _IS_MACOS else 2
comptime SO_RCVTIMEO = 0x1006 if _IS_MACOS else 20
comptime SO_SNDTIMEO = 0x1005 if _IS_MACOS else 21
comptime SO_NOSIGPIPE = 0x1022  # macOS only; unused on Linux
comptime MSG_NOSIGNAL = 0 if _IS_MACOS else 0x4000

comptime EINTR = 4
comptime EAGAIN = 35 if _IS_MACOS else 11
comptime ECONNREFUSED = 61 if _IS_MACOS else 111
comptime EADDRNOTAVAIL = 49 if _IS_MACOS else 99
comptime ETIMEDOUT = 60 if _IS_MACOS else 110
comptime EHOSTUNREACH = 65 if _IS_MACOS else 113
comptime ENETUNREACH = 51 if _IS_MACOS else 101

comptime CONNECT_RETRY_SECONDS = 0.05
"""How long a worker waits before retrying a refused connection to the root.

Short enough that a root starting a moment later costs a moment, long enough
that a root that never starts is not a spin. The bound is the connect
deadline, not this number."""


def _errno() -> Int:
    """The current thread's `errno`.

    It is a function on both platforms and a differently named one on each,
    which is the whole reason this exists.
    """
    comptime if _IS_MACOS:
        return Int(external_call["__error", UnsafePointer[Int32]]()[])
    comptime if _IS_LINUX:
        return Int(external_call["__errno_location", UnsafePointer[Int32]]()[])
    return 0


def _errno_text(prefix: String, err: Int) -> String:
    return prefix + " (errno " + String(err) + ")"


def _zero_bytes(n: Int) -> List[UInt8]:
    var out = List[UInt8](capacity=n)
    out.resize(n, 0)
    return out^


def sockets_unavailable_detail() -> String:
    return String(
        "this platform has no socket endpoint compiled in: only macOS and"
        " Linux are supported, and the constant tables and struct layouts in"
        " distributed_transport.mojo are written for those two only"
    )


def parse_ipv4(host: String) raises -> UInt32:
    """A dotted-quad IPv4 address as a host-order 32-bit value.

    `localhost` is accepted and means 127.0.0.1, because that is what a
    hermetic two-process test writes in its machine list. Every other name is
    refused rather than resolved: see the note on `getaddrinfo` above.
    """
    var text = String("127.0.0.1") if host == "localhost" else String(host)
    var parts = text.split(".")
    if len(parts) != 4:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID,
            -1,
            "'"
            + host
            + "' is not a dotted-quad IPv4 address. This build does not"
            " resolve hostnames, so a machine list must carry addresses such"
            " as 127.0.0.1:12400",
        )
    var value = UInt32(0)
    for i in range(4):
        var octet: Int
        try:
            octet = Int(String(parts[i]))
        except:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                -1,
                "'" + host + "' has a non-numeric octet",
            )
        if octet < 0 or octet > 255:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                -1,
                "'" + host + "' has an octet outside 0..255",
            )
        value = (value << 8) | UInt32(octet)
    return value


def _sockaddr_in(ip: UInt32, port: Int) -> List[UInt8]:
    """A `struct sockaddr_in` for this platform, as 16 bytes.

    macOS: `{u8 sin_len, u8 sin_family, be16 sin_port, be32 sin_addr, u8[8]}`.
    Linux: `{u16 sin_family, be16 sin_port, be32 sin_addr, u8[8]}`.
    """
    var out = _zero_bytes(SOCKADDR_IN_BYTES)
    comptime if _IS_MACOS:
        out[0] = UInt8(SOCKADDR_IN_BYTES)
        out[1] = UInt8(AF_INET)
    comptime if not _IS_MACOS:
        out[0] = UInt8(AF_INET)
        out[1] = 0
    out[2] = UInt8((port >> 8) & 0xFF)
    out[3] = UInt8(port & 0xFF)
    out[4] = UInt8((ip >> 24) & 0xFF)
    out[5] = UInt8((ip >> 16) & 0xFF)
    out[6] = UInt8((ip >> 8) & 0xFF)
    out[7] = UInt8(ip & 0xFF)
    return out^


def _timeval(ns: Int) -> List[UInt8]:
    """A `struct timeval` for `ns`, never zero.

    A zero `timeval` means "no timeout" to `setsockopt`, which is the exact
    opposite of what a caller passing a deadline that has nearly expired
    means, so it is clamped up to one microsecond instead.
    """
    var total = ns if ns > 0 else 1000
    var sec = total // NS_PER_SECOND
    var usec = (total % NS_PER_SECOND) // 1000
    if sec == 0 and usec == 0:
        usec = 1
    var out = _zero_bytes(TIMEVAL_BYTES)
    for i in range(8):
        out[i] = UInt8((sec >> (8 * i)) & 0xFF)
    for i in range(4):
        out[8 + i] = UInt8((usec >> (8 * i)) & 0xFF)
    return out^


def _setsockopt(
    fd: Int, level: Int, option: Int, mut value: List[UInt8]
) -> Int:
    return Int(
        external_call["setsockopt", Int32](
            Int32(fd),
            Int32(level),
            Int32(option),
            value.unsafe_ptr(),
            UInt32(len(value)),
        )
    )


def _set_flag(fd: Int, level: Int, option: Int, on: Bool) -> Int:
    var value = _zero_bytes(4)
    if on:
        value[0] = 1
    return _setsockopt(fd, level, option, value)


def _set_timeout(fd: Int, option: Int, ns: Int) raises:
    var tv = _timeval(ns)
    if _setsockopt(fd, SOL_SOCKET, option, tv) != 0:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID,
            -1,
            _errno_text("could not set a socket timeout", _errno()),
        )


def _close_fd(fd: Int):
    if fd >= 0:
        _ = external_call["close", Int32](Int32(fd))


def _new_stream_socket() raises -> Int:
    comptime if not SOCKETS_SUPPORTED:
        raise transport_error(
            TRANSPORT_UNAVAILABLE, -1, sockets_unavailable_detail()
        )
    var fd = Int(
        external_call["socket", Int32](
            Int32(AF_INET), Int32(SOCK_STREAM), Int32(0)
        )
    )
    if fd < 0:
        raise transport_error(
            TRANSPORT_PEER_LOST,
            -1,
            _errno_text("could not create a socket", _errno()),
        )
    # Small frames back to back: Nagle would add up to 40ms to every
    # collective, which on a per-node all-reduce is the whole cost.
    _ = _set_flag(fd, IPPROTO_TCP, TCP_NODELAY, True)
    comptime if _IS_MACOS:
        _ = _set_flag(fd, SOL_SOCKET, SO_NOSIGPIPE, True)
    return fd


struct SocketEndpoint(ByteEndpoint, Copyable, Movable):
    """A `ByteEndpoint` over a connected TCP socket. The real transport.

    It holds a file descriptor and nothing else, which is deliberate: it is
    `Copyable` because `TransportCollective` requires it, and a copy therefore
    shares the descriptor rather than duplicating it. For the same reason
    there is no destructor. Closing is explicit, exactly once, through
    `close`, and `TransportCollective.shutdown` is what performs it for a
    real run.

    Every read and write arms the corresponding socket timeout from the
    caller's absolute deadline before it is issued, so `deadline_ns` means
    what the trait says it means: the deadline for the whole operation, not a
    fresh timeout per syscall.
    """

    var peer: Int
    var fd: Int
    var live: Bool

    def __init__(out self, peer: Int, fd: Int):
        self.peer = peer
        self.fd = fd
        self.live = True

    def peer_rank(self) -> Int:
        return self.peer

    def is_open(self) -> Bool:
        return self.live

    def _arm(self, option: Int, deadline_ns: Int) raises:
        var remaining = deadline_ns - Int(perf_counter_ns())
        if remaining <= 0:
            raise transport_error(
                TRANSPORT_TIMEOUT,
                self.peer,
                "the deadline had already passed before the syscall",
            )
        _set_timeout(self.fd, option, remaining)

    def send_all(mut self, bytes: List[UInt8], deadline_ns: Int) raises:
        if not self.live:
            raise transport_error(
                TRANSPORT_PEER_LOST, self.peer, "write to a closed socket"
            )
        var total = len(bytes)
        var sent = 0
        while sent < total:
            self._arm(SO_SNDTIMEO, deadline_ns)
            var n = Int(
                external_call["send", Int64](
                    Int32(self.fd),
                    bytes.unsafe_ptr() + sent,
                    UInt64(total - sent),
                    Int32(MSG_NOSIGNAL),
                )
            )
            if n > 0:
                sent += n
                continue
            var err = _errno()
            if err == EINTR:
                continue
            if err == EAGAIN:
                raise transport_error(
                    TRANSPORT_TIMEOUT,
                    self.peer,
                    "the send deadline expired after "
                    + String(sent)
                    + " of "
                    + String(total)
                    + " bytes",
                )
            self.live = False
            raise transport_error(
                TRANSPORT_PEER_LOST,
                self.peer,
                _errno_text("send failed", err),
            )

    def recv_exact(mut self, n: Int, deadline_ns: Int) raises -> List[UInt8]:
        if n < 0:
            raise transport_error(
                TRANSPORT_FRAME_CORRUPT, self.peer, "negative read length"
            )
        if not self.live:
            raise transport_error(
                TRANSPORT_PEER_LOST, self.peer, "read from a closed socket"
            )
        var out = _zero_bytes(n)
        var got = 0
        while got < n:
            self._arm(SO_RCVTIMEO, deadline_ns)
            var r = Int(
                external_call["recv", Int64](
                    Int32(self.fd),
                    out.unsafe_ptr() + got,
                    UInt64(n - got),
                    Int32(0),
                )
            )
            if r > 0:
                got += r
                continue
            if r == 0:
                # An orderly shutdown mid-frame is a worker that exited, not
                # a short read: there is no more data coming, ever.
                self.live = False
                raise transport_error(
                    TRANSPORT_PEER_LOST,
                    self.peer,
                    "the peer closed the connection after "
                    + String(got)
                    + " of "
                    + String(n)
                    + " bytes",
                )
            var err = _errno()
            if err == EINTR:
                continue
            if err == EAGAIN:
                raise transport_error(
                    TRANSPORT_TIMEOUT,
                    self.peer,
                    "no bytes before the deadline after "
                    + String(got)
                    + " of "
                    + String(n),
                )
            self.live = False
            raise transport_error(
                TRANSPORT_PEER_LOST,
                self.peer,
                _errno_text("recv failed", err),
            )
        return out^

    def close(mut self) raises:
        if not self.live:
            return
        self.live = False
        _close_fd(self.fd)


struct SocketListener(Movable):
    """The root's listening socket, open only for the length of a rendezvous.

    `SO_REUSEADDR` is set because a job restarted immediately after one that
    ended would otherwise be refused for as long as the previous port sits in
    `TIME_WAIT`, which is the common case for a retry and has nothing to do
    with the port being in use.
    """

    var fd: Int
    var port: Int

    def __init__(out self, host: String, port: Int) raises:
        var ip = parse_ipv4(host)
        var fd = _new_stream_socket()
        _ = _set_flag(fd, SOL_SOCKET, SO_REUSEADDR, True)
        var addr = _sockaddr_in(ip, port)
        var rc = Int(
            external_call["bind", Int32](
                Int32(fd), addr.unsafe_ptr(), UInt32(SOCKADDR_IN_BYTES)
            )
        )
        if rc != 0:
            var err = _errno()
            _close_fd(fd)
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                ROOT_RANK,
                _errno_text(
                    "could not bind " + host + ":" + String(port), err
                ),
            )
        if (
            Int(
                external_call["listen", Int32](
                    Int32(fd), Int32(LISTEN_BACKLOG)
                )
            )
            != 0
        ):
            var err = _errno()
            _close_fd(fd)
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                ROOT_RANK,
                _errno_text("could not listen", err),
            )
        self.fd = fd
        self.port = port

    def accept_one(mut self, deadline_ns: Int) raises -> Int:
        """Accept one connection, or raise when the deadline passes.

        `accept` honors `SO_RCVTIMEO` on both supported platforms, which is
        why there is no `poll` here and no non-blocking mode to unwind.
        """
        while True:
            var remaining = deadline_ns - Int(perf_counter_ns())
            if remaining <= 0:
                raise transport_error(
                    TRANSPORT_TIMEOUT,
                    ROOT_RANK,
                    "not every worker connected before the connect deadline",
                )
            _set_timeout(self.fd, SO_RCVTIMEO, remaining)
            var addr = _zero_bytes(SOCKADDR_IN_BYTES)
            var addr_len: List[UInt32] = [UInt32(SOCKADDR_IN_BYTES)]
            var fd = Int(
                external_call["accept", Int32](
                    Int32(self.fd),
                    addr.unsafe_ptr(),
                    addr_len.unsafe_ptr(),
                )
            )
            if fd >= 0:
                _ = _set_flag(fd, IPPROTO_TCP, TCP_NODELAY, True)
                comptime if _IS_MACOS:
                    _ = _set_flag(fd, SOL_SOCKET, SO_NOSIGPIPE, True)
                return fd
            var err = _errno()
            if err == EINTR:
                continue
            if err == EAGAIN:
                raise transport_error(
                    TRANSPORT_TIMEOUT,
                    ROOT_RANK,
                    "not every worker connected before the connect deadline",
                )
            raise transport_error(
                TRANSPORT_PEER_LOST,
                ROOT_RANK,
                _errno_text("accept failed", err),
            )

    def close(mut self):
        _close_fd(self.fd)
        self.fd = -1


def connect_to_root(
    host: String, port: Int, rank: Int, deadline_ns: Int
) raises -> Int:
    """Connect to the root, retrying a refusal until the deadline.

    A refused connection is the expected state, not an error: workers and the
    root are separate processes and nothing orders their starts, so a worker
    that beats the root to the port must wait rather than fail. Refusal,
    timeout, and an unreachable route are retried; anything else is reported
    at once, because retrying a malformed address only delays the message.
    """
    var ip = parse_ipv4(host)
    while True:
        if Int(perf_counter_ns()) >= deadline_ns:
            raise transport_error(
                TRANSPORT_TIMEOUT,
                rank,
                "could not reach the root at "
                + host
                + ":"
                + String(port)
                + " before the connect deadline",
            )
        var fd = _new_stream_socket()
        var remaining = deadline_ns - Int(perf_counter_ns())
        if remaining > 0:
            _set_timeout(fd, SO_SNDTIMEO, remaining)
        var addr = _sockaddr_in(ip, port)
        var rc = Int(
            external_call["connect", Int32](
                Int32(fd), addr.unsafe_ptr(), UInt32(SOCKADDR_IN_BYTES)
            )
        )
        if rc == 0:
            return fd
        var err = _errno()
        _close_fd(fd)
        var retryable = (
            err == ECONNREFUSED
            or err == ETIMEDOUT
            or err == EINTR
            or err == EAGAIN
            or err == EHOSTUNREACH
            or err == ENETUNREACH
            or err == EADDRNOTAVAIL
        )
        if not retryable:
            raise transport_error(
                TRANSPORT_PEER_LOST,
                rank,
                _errno_text(
                    "could not connect to " + host + ":" + String(port), err
                ),
            )
        sleep(CONNECT_RETRY_SECONDS)


# ---------------------------------------------------------------------------
# The blocking collective driver
# ---------------------------------------------------------------------------


struct TransportCollective[E: ByteEndpoint & Copyable & Deinitable](
    Collective, Movable
):
    """A `Collective` over byte endpoints, using gather at the root and
    broadcast back.

    On the root, `peers[i]` is the endpoint to rank `i + 1`. On any other
    rank, `peers` holds exactly one endpoint, to the root. That asymmetry is
    the star topology written down: rank 0 is the reduction point, everyone
    else has one connection.

    Why gather and broadcast rather than a ring: docs/distributed.md section 5
    requirement 1 forbids a scheme that leaves different ranks holding
    different roundings of the same sum, which rules out the usual ring
    all-reduce with per-rank rotation. Here the root folds in ascending rank
    order and every other rank adopts the root's bytes unchanged, so
    bit-identical delivery is structural.

    Communication is `2 * (world_size - 1)` messages per collective against a
    ring's `2 * world_size` of `1 / world_size` the size, so this is the
    latency-optimal and bandwidth-pessimal end of the trade. It is the right
    end for a first transport, where being obviously correct matters more, and
    docs/DISTRIBUTED_TRANSPORT.md section 8 records what replacing it costs.

    `n_local_ranks` is 1: a real transport hosts one rank per process, and the
    growth loop in `distributed.mojo` degenerates to a single iteration, which
    is exactly the case the local-rank loop was written to cover.
    """

    var session: TransportSession
    var peers: List[Self.E]

    def __init__(
        out self, var session: TransportSession, var peers: List[Self.E]
    ) raises:
        var expected = (
            session.config.world_size - 1 if session.config.is_root() else 1
        )
        if len(peers) != expected:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                session.config.rank,
                "expected " + String(expected) + " endpoints",
            )
        self.session = session^
        self.peers = peers^

    def world_size(self) -> Int:
        return self.session.config.world_size

    def rank(self) -> Int:
        return self.session.config.rank

    def n_local_ranks(self) -> Int:
        return 1

    def local_rank(self, index: Int) -> Int:
        return self.session.config.rank

    def _send_to(mut self, index: Int, bytes: List[UInt8]) raises:
        """One write, with a failed peer recorded before the error escapes.

        Recording first is what makes worker loss terminal: the session is
        already `FAILED` by the time any caller sees the error, so nothing can
        catch it and issue another collective.
        """
        self.session.check_deadline()
        var peer = self.peers[index].peer_rank()
        try:
            self.peers[index].send_all(bytes, self.session.deadline_ns)
        except e:
            self.session.mark_lost(peer)
            raise e

    def _recv_from(mut self, index: Int) raises -> List[UInt8]:
        self.session.check_deadline()
        var peer = self.peers[index].peer_rank()
        try:
            return recv_frame_bytes(self.peers[index], self.session.deadline_ns)
        except e:
            self.session.mark_lost(peer)
            raise e

    def _broadcast(mut self, bytes: List[UInt8]) raises:
        """Send the identical buffer to every peer, unmodified, so no rank can
        end up holding a different rounding of the same reduction."""
        for i in range(len(self.peers)):
            self._send_to(i, bytes)

    def _gather_f64(mut self, mut acc: List[Float64]) raises:
        """Fold every peer's contribution in ascending rank order.

        Ascending because `peers[i]` is rank `i + 1` and the loop runs
        forward, and because the read is blocking, so the fold order is the
        rank order regardless of which peer answered first. That independence
        from arrival order is the reproducibility claim in
        docs/distributed.md section 6.
        """
        for i in range(len(self.peers)):
            var sender = self.peers[i].peer_rank()
            var frame = self._recv_from(i)
            absorb_f64(self.session, acc, frame, sender)

    def _gather_int(mut self, mut acc: List[Int]) raises:
        for i in range(len(self.peers)):
            var sender = self.peers[i].peer_rank()
            var frame = self._recv_from(i)
            absorb_int(self.session, acc, frame, sender)

    def allreduce_sum_f64(mut self, mut buf: List[Float64]) raises:
        _ = self.session.begin(OP_SUM_F64, len(buf))
        if self.session.config.is_root():
            var acc = buf.copy()
            self._gather_f64(acc)
            var answer = result_frame_f64(self.session, acc)
            self._broadcast(answer)
            buf = acc^
        else:
            var contribution = contribution_frame_f64(self.session, buf)
            self._send_to(0, contribution)
            var reply = self._recv_from(0)
            buf = adopt_result_f64(self.session, reply)
        self.session.finish()

    def _allreduce_int(mut self, mut buf: List[Int], op: Int) raises:
        _ = self.session.begin(op, len(buf))
        if self.session.config.is_root():
            var acc = buf.copy()
            self._gather_int(acc)
            var answer = result_frame_int(self.session, acc)
            self._broadcast(answer)
            buf = acc^
        else:
            var contribution = contribution_frame_int(self.session, buf)
            self._send_to(0, contribution)
            var reply = self._recv_from(0)
            buf = adopt_result_int(self.session, reply)
        self.session.finish()

    def allreduce_sum_int(mut self, mut buf: List[Int]) raises:
        self._allreduce_int(buf, OP_SUM_INT)

    def allreduce_max_int(mut self, mut buf: List[Int]) raises:
        self._allreduce_int(buf, OP_MAX_INT)

    def request_cancel(mut self, detail: String):
        """Stop this rank's session on purpose.

        Sticky and local: the next collective this rank enters raises
        `TRANSPORT_CANCELLED` instead of sending, and the peers waiting on it
        then fail with a lost peer or a deadline rather than hanging. That is
        cancellation, not a cancellation broadcast, and the difference matters:
        nothing here tells the other ranks *why* they lost this one. A run that
        wants every rank to stop for the same stated reason at the same round
        agrees a control code at a round boundary instead, which is what
        `train_distributed_run` does with a callback's return value.
        """
        self.session.cancel(detail)

    def checkpoint_boundary(mut self) raises:
        """Close one checkpoint epoch and open the next.

        The barrier first, so the epoch advances only from a point every rank
        has reached; the sequence numbering then restarts inside the new epoch,
        which is what makes a frame from before the checkpoint impossible to
        confuse with one from after it.
        """
        self.barrier()
        self.session.next_epoch()

    def shutdown(mut self) raises:
        """Close every endpoint and the session.

        Called on the way out of a run whether or not it succeeded. It does not
        raise on an already-broken session: a rank that is shutting down
        because a peer vanished must still release its own endpoints.
        """
        for i in range(len(self.peers)):
            try:
                self.peers[i].close()
            except:
                pass
        self.session.close()

    def failure_code(self) -> Int:
        """The transport status this session died of, or `TRANSPORT_OK`."""
        return self.session.failure_code

    def collectives_completed(self) -> Int:
        return self.session.collectives_completed

    def elements_reduced(self) -> Int:
        return self.session.elements_reduced

    def barrier(mut self) raises:
        """Same gather and broadcast with an empty payload. Deliberately not
        free: it is the only thing that makes an epoch boundary a point every
        rank has actually reached."""
        _ = self.session.begin(OP_BARRIER, 0)
        if self.session.config.is_root():
            for i in range(len(self.peers)):
                var sender = self.peers[i].peer_rank()
                var frame = self._recv_from(i)
                var decoded = decode_frame(frame)
                self.session.validate(decoded.header, MSG_CONTRIB, sender)
            var answer = barrier_frame(self.session, MSG_RESULT)
            self._broadcast(answer)
        else:
            var contribution = barrier_frame(self.session, MSG_CONTRIB)
            self._send_to(0, contribution)
            var reply = self._recv_from(0)
            var decoded = decode_frame(reply)
            self.session.validate(decoded.header, MSG_RESULT, ROOT_RANK)
        self.session.finish()


# ---------------------------------------------------------------------------
# Histogram all-reduce contract
# ---------------------------------------------------------------------------


@fieldwise_init
struct HistogramPlan(Copyable, Movable):
    """What one tree node costs on the wire, from docs/distributed.md section
    8. Computed rather than asserted so a test can pin the cost model against
    the wire format instead of against a comment."""

    var n_features: Int
    var n_bins: Int
    var cells: Int
    var reduces_per_node: Int
    var payload_bytes_per_node: Int
    var framing_bytes_per_node: Int


def histogram_plan(n_features: Int, n_bins: Int) raises -> HistogramPlan:
    """The three-buffer schedule `allreduce_histogram` actually issues.

    Three reductions per node, not one, because gradients, hessians, and
    counts stay in their own typed buffers and the exactness of the counts is
    then obvious. Packing them is a factor of three fewer round trips at
    identical arithmetic and is left to a transport that has round trips to
    save; section 8 of docs/DISTRIBUTED_TRANSPORT.md keeps the accounting.
    """
    if n_features < 1 or n_bins < 1:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID,
            -1,
            "a histogram needs at least one feature and one bin",
        )
    var cells = n_features * n_bins
    return HistogramPlan(
        n_features, n_bins, cells, 3, 3 * cells * 8, 3 * FRAME_HEADER_BYTES
    )


def check_histogram_buffers(
    plan: HistogramPlan, n_grad: Int, n_hess: Int, n_count: Int
) raises:
    """Refuse a histogram whose three buffers do not describe the same grid.

    Cheap, local, and worth doing before the send rather than after: a
    mismatched buffer reaches the peers as a length disagreement, which is
    correctly detected but is reported against the wrong rank.
    """
    if n_grad != plan.cells or n_hess != plan.cells or n_count != plan.cells:
        raise transport_error(
            TRANSPORT_SCHEMA_MISMATCH,
            -1,
            "histogram buffers do not match "
            + String(plan.n_features)
            + " features by "
            + String(plan.n_bins)
            + " bins",
        )


# ---------------------------------------------------------------------------
# Global split agreement
# ---------------------------------------------------------------------------


def split_digest(feature: Int, bin: Int, gain: Float64, found: Bool) -> UInt64:
    """A digest of one node's chosen split, exact in the gain bits.

    Bits and not a rounded gain, because the claim being checked is that every
    rank computed the identical split from the identical global histogram. A
    tolerance here would hide precisely the divergence the check exists to
    catch.
    """
    var bits = f64_bits(gain)
    var values: List[Int] = [
        feature,
        bin,
        1 if found else 0,
        Int(bits % 0x1_0000_0000),
        Int(bits // 0x1_0000_0000),
    ]
    return digest_ints(values)


def check_split_agreement(digests: List[UInt64]) raises -> Int:
    """Return the lowest rank whose split digest differs from rank 0's, or -1.

    Rank 0 is the reference only so that every rank names the same offender.
    In a correct run this is unreachable: the split is a pure function of the
    all-reduced histogram, so agreement is structural, not negotiated. It is
    here as a cheap assertion over a real transport, where a silently
    corrupted histogram would otherwise produce two different trees and no
    error at all.
    """
    if len(digests) == 0:
        raise transport_error(
            TRANSPORT_SPLIT_DISAGREEMENT, -1, "no split digests to compare"
        )
    for r in range(1, len(digests)):
        if digests[r] != digests[0]:
            return r
    return -1


# ---------------------------------------------------------------------------
# Checkpoint and restart metadata
# ---------------------------------------------------------------------------

comptime CHECKPOINT_MAGIC = UInt64(0x4D4A4243)  # "MJBC"

# magic u32, version u16, job_id u64, schema u64, model_digest u64, then
# epoch, seq, world_size, and n_trees as u32 each.
comptime CHECKPOINT_BYTES = 46


@fieldwise_init
struct CheckpointMeta(Copyable, Movable):
    """What a restart needs to prove it is resuming the same run.

    This is metadata only. Nothing here writes a model, and section 7 of
    docs/distributed.md is explicit that fault tolerance is a separate
    project. What this does is make a wrong restart an error: resuming a
    12-rank run on 8 ranks, or against a different machine list, or from a
    checkpoint of a different job, is refused instead of producing a model
    whose provenance nobody can reconstruct.
    """

    var job_id: UInt64
    var schema: UInt64
    var model_digest: UInt64
    var epoch: Int
    var seq: Int
    var world_size: Int
    var n_trees: Int


def encode_checkpoint_meta(meta: CheckpointMeta) raises -> List[UInt8]:
    var out = List[UInt8](capacity=CHECKPOINT_BYTES)
    _put_uint(out, CHECKPOINT_MAGIC, 4)
    _put_field(out, "version", TRANSPORT_PROTOCOL_VERSION, 2)
    _put_uint(out, meta.job_id, 8)
    _put_uint(out, meta.schema, 8)
    _put_uint(out, meta.model_digest, 8)
    _put_field(out, "epoch", meta.epoch, 4)
    _put_field(out, "seq", meta.seq, 4)
    _put_field(out, "world_size", meta.world_size, 4)
    _put_field(out, "n_trees", meta.n_trees, 4)
    return out^


def decode_checkpoint_meta(buf: List[UInt8]) raises -> CheckpointMeta:
    if len(buf) != CHECKPOINT_BYTES:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "checkpoint record has a wrong length"
        )
    if _get_uint(buf, 0, 4) != CHECKPOINT_MAGIC:
        raise transport_error(
            TRANSPORT_FRAME_CORRUPT, -1, "checkpoint magic does not match"
        )
    var version = Int(_get_uint(buf, 4, 2))
    if version != TRANSPORT_PROTOCOL_VERSION:
        raise transport_error(
            TRANSPORT_SCHEMA_MISMATCH,
            -1,
            "checkpoint protocol version " + String(version),
        )
    return CheckpointMeta(
        _get_uint(buf, 6, 8),
        _get_uint(buf, 14, 8),
        _get_uint(buf, 22, 8),
        Int(_get_uint(buf, 30, 4)),
        Int(_get_uint(buf, 34, 4)),
        Int(_get_uint(buf, 38, 4)),
        Int(_get_uint(buf, 42, 4)),
    )


def restart_status(
    saved: CheckpointMeta, current: TransportConfig
) raises -> Int:
    """Whether a checkpoint may be resumed by this rank's configuration.

    World size and schema are both hard requirements. World size because the
    row partition is a function of it, so resuming at a different one would
    silently re-shard the data underneath a half-built model. Schema because
    the rank assignment lives in the machine list, and a rank that resumes
    against a rewritten list would take over another rank's shard.
    """
    if saved.job_id != current.job_id:
        return TRANSPORT_JOB_MISMATCH
    if saved.world_size != current.world_size:
        return TRANSPORT_WORLD_MISMATCH
    if saved.schema != current.schema_digest():
        return TRANSPORT_SCHEMA_MISMATCH
    if saved.epoch < 0 or saved.n_trees < 0:
        return TRANSPORT_FRAME_CORRUPT
    return TRANSPORT_OK


def resume_session(
    var config: TransportConfig, saved: CheckpointMeta
) raises -> TransportSession:
    """Build a session positioned at a checkpoint's epoch.

    Sequence numbering restarts at zero inside the new epoch rather than
    resuming mid-epoch, because a checkpoint is only ever taken between
    boosting rounds and a resumed sequence number would have to be agreed
    across ranks that may have written their checkpoints one collective apart.
    """
    var status = restart_status(saved, config)
    if status != TRANSPORT_OK:
        raise transport_error(
            status, config.rank, "the checkpoint cannot be resumed here"
        )
    var session = TransportSession(config^)
    session.epoch = saved.epoch + 1
    session.seq = 0
    return session^


# ---------------------------------------------------------------------------
# Zero buffers, re-exported for callers that build contributions by hand
# ---------------------------------------------------------------------------


def zero_contribution_f64(size: Int) -> List[Float64]:
    """A rank with nothing to say still calls every collective, contributing
    zeros. Skipping the call instead is the deadlock docs/distributed.md
    section 4 warns about."""
    return zeros_f64(size)


def zero_contribution_int(size: Int) -> List[Int]:
    return zeros_int(size)


# ---------------------------------------------------------------------------
# The runtime interface
# ---------------------------------------------------------------------------
#
# One narrow seam for everything outside Mojo: the Python bindings, the Dask
# client in python/mojoboost/dask.py, and a future launcher all describe a run
# the same way here, and get back either a working `Collective` or one error
# that says exactly why they cannot have one.
#
# It is deliberately small. A `RuntimeSpec` is the rank, the world, the machine
# list, the job id, and the timeouts, and nothing about the data or the model.
# `open_local_collective` and `open_transport_collective` are the only two ways
# a collective is ever built. `transport_available` is the one place that says
# whether a second process can be reached at all, and today it says no.

comptime RUNTIME_LOCAL = 0
"""Every rank of the world hosted in this process. Real, and the only mode
that runs today."""

comptime RUNTIME_TRANSPORT = 1
"""One rank per process, over TCP. See `transport_available` for whether this
platform has an endpoint, and `transport_validated` for the separate and more
important question of whether anyone has ever run it."""

# Whether a `ByteEndpoint` that can actually reach another process is compiled
# in. `MemoryEndpoint` is a fake and does not count; `SocketEndpoint` is not,
# and this is now exactly the platform predicate for it. It stays a comptime
# name rather than a runtime probe because whether the adapter exists is a
# fact about the build.
comptime HAS_BYTE_ENDPOINT = SOCKETS_SUPPORTED


def runtime_mode_name(mode: Int) -> String:
    if mode == RUNTIME_LOCAL:
        return "local"
    if mode == RUNTIME_TRANSPORT:
        return "transport"
    return "unknown"


def parse_runtime_mode(text: String) raises -> Int:
    if text == "local":
        return RUNTIME_LOCAL
    if text == "transport":
        return RUNTIME_TRANSPORT
    raise transport_error(
        TRANSPORT_CONFIG_INVALID,
        -1,
        "runtime mode must be 'local' or 'transport', got '" + text + "'",
    )


def transport_available() -> Bool:
    """Whether this build can open a session to another process.

    True on macOS and Linux, where `SocketEndpoint` is compiled in, and False
    elsewhere. Stated once here so that every caller refuses for the same
    reason with the same message instead of each discovering the gap its own
    way.

    Available is not the same as validated. See `transport_validated`, which
    is the honest answer to "has this ever worked", and which is False.
    """
    return HAS_BYTE_ENDPOINT


def transport_validated() -> Bool:
    """Whether the socket transport has ever moved a byte between processes.

    False, and deliberately a separate name from `transport_available` rather
    than folded into it, because the two facts are different and conflating
    them is how an untested path gets described as a working one.

    What is true: the framing, ordering, deadline, cancellation, worker loss,
    handshake, and reduction logic above the endpoint are exercised in process
    against `MemoryEndpoint`, and `SocketEndpoint` is a complete BSD socket
    adapter written against the documented C ABI of macOS and Linux.

    What is not true, and is not claimed anywhere: that it has been compiled,
    that a rendezvous has completed, that two processes have trained together,
    or anything at all about multi-host behavior, throughput, or scaling. The
    hermetic two-process procedure in docs/DISTRIBUTED_TRANSPORT.md section 7
    is what turns this to True, and until someone runs it a caller that needs
    a validated transport should gate on this and not on availability.
    """
    return False


def transport_unavailable_detail() -> String:
    return String(
        "this build has no ByteEndpoint that can reach another process."
        " SocketEndpoint is compiled in on macOS and Linux only, because the"
        " syscall constants and the sockaddr_in layout it needs are written"
        " for those two platforms. Configuration, framing, ordering,"
        " deadlines, and the reduction are all validated on every platform,"
        " so a rank can still be configured and refused rather than left to"
        " hang. Train with the local runtime meanwhile"
    )


def transport_unvalidated_detail() -> String:
    return String(
        "the socket transport in this build has never been compiled or run:"
        " no rendezvous has completed and no byte has moved between two"
        " processes. Everything above the endpoint is exercised in process."
        " Run the hermetic two-process procedure in"
        " docs/DISTRIBUTED_TRANSPORT.md section 7 before depending on it"
    )


@fieldwise_init
struct RuntimeCapability(Copyable, Movable):
    """What this build's distributed stack can actually do.

    - `multi_process`: whether two processes can train together on this
      platform, meaning a socket endpoint is compiled in.
    - `validated`: whether that path has ever been compiled and run. False.
      It is separate from `multi_process` on purpose: a caller that must not
      run unproven code gates on this one, and a caller reporting what the
      build supports reports the other. Nothing collapses the two.
    - `local_collective`: whether a world hosted in one process is available.
    - `protocol_version`: the version every rank of a job must agree on.
    - `max_world_size`: the largest world this build can form, or -1 when a
      transport exists and the limit is the machine list's rather than this
      module's.
    - `reason`: empty when `multi_process` is True, and the explanation to
      report otherwise.
    - `caveat`: empty when `validated` is True, and what is unproven
      otherwise. Non-empty today even when `multi_process` is True, which is
      the case this field exists for.
    """

    var multi_process: Bool
    var validated: Bool
    var local_collective: Bool
    var protocol_version: Int
    var max_world_size: Int
    var reason: String
    var caveat: String


def distributed_runtime_capability() -> RuntimeCapability:
    """The one native answer to "can this build train across processes".

    It answers in one place: the Python bindings, the Dask backend, and a
    launcher all report this rather than each carrying its own copy of the
    fact and its own wording. A caller that refuses distributed work should
    refuse on `multi_process` and quote `reason`, and a caller that refuses
    *unproven* distributed work should refuse on `validated` and quote
    `caveat`. Neither should be inferred from this function existing.
    """
    var caveat = transport_unvalidated_detail()
    if transport_validated():
        caveat = String("")
    if transport_available():
        return RuntimeCapability(
            True,
            transport_validated(),
            True,
            TRANSPORT_PROTOCOL_VERSION,
            -1,
            String(""),
            caveat^,
        )
    return RuntimeCapability(
        False,
        False,
        True,
        TRANSPORT_PROTOCOL_VERSION,
        1,
        transport_unavailable_detail(),
        caveat^,
    )


@fieldwise_init
struct RuntimeSpec(Copyable, Movable):
    """How one rank reaches its world, and nothing else.

    `mode` picks between a world hosted in this process and one spread over
    processes. In `RUNTIME_LOCAL`, `addresses` is empty and `rank` is 0,
    because the process is every rank; in `RUNTIME_TRANSPORT`, `addresses` is
    the machine list indexed by rank, which is the rank assignment itself.

    Nothing here describes the data, the objective, or the model. That
    separation is what lets a launcher build this from an environment before it
    has read a row, and what keeps the training entry point in
    `distributed.mojo` from growing a second copy of the wire configuration.
    """

    var mode: Int
    var rank: Int
    var world_size: Int
    var addresses: List[RankAddress]
    var job_id: UInt64
    var connect_timeout_ns: Int
    var collective_timeout_ns: Int
    var restart_epoch: Int

    def validate(self) raises:
        """Refuse a spec before anything is opened.

        Every check here is one whose absence would surface later as a hang, a
        silently wrong rank assignment, or a model whose provenance cannot be
        reconstructed.
        """
        if self.mode != RUNTIME_LOCAL and self.mode != RUNTIME_TRANSPORT:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                self.rank,
                "unknown runtime mode " + String(self.mode),
            )
        if self.world_size < 1:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID, self.rank, "world size must be > 0"
            )
        if self.rank < 0 or self.rank >= self.world_size:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                self.rank,
                "rank is outside the world",
            )
        if self.restart_epoch < 0:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                self.rank,
                "restart epoch must not be negative",
            )
        if self.mode == RUNTIME_LOCAL:
            if len(self.addresses) != 0:
                raise transport_error(
                    TRANSPORT_CONFIG_INVALID,
                    self.rank,
                    "a local runtime has no addresses to connect to",
                )
            if self.rank != 0:
                raise transport_error(
                    TRANSPORT_CONFIG_INVALID,
                    self.rank,
                    "a local runtime hosts every rank and is rank 0",
                )
            return
        if len(self.addresses) != self.world_size:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                self.rank,
                "the machine list has "
                + String(len(self.addresses))
                + " entries for a world of "
                + String(self.world_size),
            )
        _ = self.transport_config()

    def transport_config(self) raises -> TransportConfig:
        """The validated wire configuration, or an error for a local spec."""
        if self.mode != RUNTIME_TRANSPORT:
            raise transport_error(
                TRANSPORT_CONFIG_INVALID,
                self.rank,
                "a local runtime has no wire configuration",
            )
        var config = TransportConfig(
            self.rank,
            self.addresses.copy(),
            self.job_id,
            self.connect_timeout_ns,
            self.collective_timeout_ns,
        )
        config.validate()
        return config^

    def schema_digest(self) raises -> UInt64:
        """The digest every rank of this run must agree on.

        For a transport run it is the machine list's, so a rank started against
        a stale list is refused rather than silently taking over another rank's
        shard. For a local run there is no list and no peer, so it is the mode
        and the world size: enough to stamp a checkpoint with what produced it,
        and honest about carrying nothing more.
        """
        if self.mode == RUNTIME_TRANSPORT:
            return self.transport_config().schema_digest()
        return digest_ints(
            [TRANSPORT_PROTOCOL_VERSION, RUNTIME_LOCAL, self.world_size]
        )

    def describe(self) -> String:
        var text = String(
            "runtime ",
            runtime_mode_name(self.mode),
            " rank ",
            self.rank,
            " of ",
            self.world_size,
        )
        for i in range(len(self.addresses)):
            text += " " + self.addresses[i].text()
        return text^


def local_runtime(world_size: Int, job_id: UInt64 = 0) raises -> RuntimeSpec:
    """A world hosted entirely in this process."""
    var spec = RuntimeSpec(
        RUNTIME_LOCAL,
        0,
        world_size,
        List[RankAddress](),
        job_id,
        30 * NS_PER_SECOND,
        300 * NS_PER_SECOND,
        0,
    )
    spec.validate()
    return spec^


def transport_runtime(
    machines: String,
    rank: Int,
    job_id: UInt64,
    connect_timeout_ns: Int = 30 * NS_PER_SECOND,
    collective_timeout_ns: Int = 300 * NS_PER_SECOND,
    restart_epoch: Int = 0,
) raises -> RuntimeSpec:
    """A multi-process world, from a machine list in LightGBM's shape.

    Building this succeeds even though opening it does not, which is the point:
    a launcher can validate its machine list, its rank assignment, and its
    timeouts long before an endpoint exists, and `open_transport_collective`
    then refuses for one clearly stated reason rather than for a parse error.
    """
    var config = parse_machine_list(machines, rank, job_id)
    var spec = RuntimeSpec(
        RUNTIME_TRANSPORT,
        rank,
        config.world_size,
        config.addresses.copy(),
        job_id,
        connect_timeout_ns,
        collective_timeout_ns,
        restart_epoch,
    )
    spec.validate()
    return spec^


def _env_int(name: String, default: Int) -> Int:
    var s = getenv(name)
    if s.byte_length() == 0:
        return default
    try:
        return Int(s)
    except:
        return default


def runtime_from_env() raises -> RuntimeSpec:
    """The spec a launcher put in this process's environment.

    Read once, here, so that a rank id never comes from the order processes
    happened to start:

        MOJOBOOST_DIST_MODE        local (default) or transport
        MOJOBOOST_DIST_WORLD_SIZE  ranks in the world, local mode only
        MOJOBOOST_DIST_RANK        this process's rank, transport mode only
        MOJOBOOST_DIST_MACHINES    the machine list text, host:port entries
                                   separated by whitespace or newlines
        MOJOBOOST_DIST_JOB_ID      the job id every frame carries
        MOJOBOOST_DIST_TIMEOUT_S   per-collective deadline in seconds

    In transport mode the world size comes from the machine list rather than
    from a variable, so the two cannot disagree.
    """
    var mode_text = getenv("MOJOBOOST_DIST_MODE")
    var mode = RUNTIME_LOCAL
    if mode_text.byte_length() > 0:
        mode = parse_runtime_mode(mode_text)
    var job_raw = _env_int("MOJOBOOST_DIST_JOB_ID", 0)
    if job_raw < 0:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID, -1, "MOJOBOOST_DIST_JOB_ID is negative"
        )
    var job_id = UInt64(job_raw)
    if mode == RUNTIME_LOCAL:
        return local_runtime(_env_int("MOJOBOOST_DIST_WORLD_SIZE", 1), job_id)
    var machines = getenv("MOJOBOOST_DIST_MACHINES")
    if machines.byte_length() == 0:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID,
            -1,
            "transport mode needs MOJOBOOST_DIST_MACHINES",
        )
    var timeout_s = _env_int("MOJOBOOST_DIST_TIMEOUT_S", 300)
    if timeout_s <= 0:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID,
            -1,
            "MOJOBOOST_DIST_TIMEOUT_S must be positive",
        )
    return transport_runtime(
        machines,
        _env_int("MOJOBOOST_DIST_RANK", -1),
        job_id,
        30 * NS_PER_SECOND,
        timeout_s * NS_PER_SECOND,
        _env_int("MOJOBOOST_DIST_RESTART_EPOCH", 0),
    )


def require_transport(spec: RuntimeSpec) raises:
    """Refuse a transport run this build cannot open.

    Called before anything is partitioned or allocated, so a rank configured
    for a world it cannot reach stops at once with a message that says why,
    rather than blocking in a handshake that will never complete.
    """
    if spec.mode != RUNTIME_TRANSPORT:
        return
    if not transport_available():
        raise transport_error(
            TRANSPORT_UNAVAILABLE, spec.rank, transport_unavailable_detail()
        )


def open_local_collective(spec: RuntimeSpec) raises -> LocalCollective:
    """The collective for a world hosted in this process."""
    spec.validate()
    if spec.mode != RUNTIME_LOCAL:
        raise transport_error(
            TRANSPORT_CONFIG_INVALID,
            spec.rank,
            "open_local_collective needs a local runtime spec",
        )
    return LocalCollective(spec.world_size)


def open_transport_collective[
    E: ByteEndpoint & Copyable & Deinitable
](
    spec: RuntimeSpec,
    var peers: List[E],
    peer_handshakes: List[HandshakeRecord],
) raises -> TransportCollective[E]:
    """A session over already-connected endpoints, handshaken and ready.

    The endpoints are an argument rather than something opened here so that
    connecting and handshaking stay separable: `connect_world` performs the
    rendezvous and hands its result to this, and a test that wants to drive
    the blocking driver over `MemoryEndpoint` can compose the same two pieces
    without a port. `open_socket_collective` is the two together.
    """
    require_transport(spec)
    var session = TransportSession(spec.transport_config())
    session.epoch = spec.restart_epoch
    session.complete_handshake(peer_handshakes)
    return TransportCollective[E](session^, peers^)


def session_checkpoint_meta(
    session: TransportSession, model_digest: UInt64, n_trees: Int
) -> CheckpointMeta:
    """Stamp a checkpoint with the session that produced it.

    A free function rather than a method so the checkpoint record stays
    readable from a session a caller already holds, and so the trainer can
    build the same record from a `RuntimeSpec` when it is running without a
    transport at all.
    """
    return CheckpointMeta(
        session.config.job_id,
        session.config.schema_digest(),
        model_digest,
        session.epoch,
        session.seq,
        session.config.world_size,
        n_trees,
    )
