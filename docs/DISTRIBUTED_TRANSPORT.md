# Distributed transport design

Status: protocol and session state machine implemented and tested in process.
No network transport exists. Nothing described here has moved a byte between
two processes or two hosts, and no multi-host behavior, throughput, or
scalability is claimed anywhere in this document.

`docs/distributed.md` covers the training algorithm and says what a collective
has to deliver. This document covers the layer underneath it: how two
processes on two hosts would actually agree on a reduction. The code is
`src/mojotrees/distributed_transport.mojo` and the tests are
`tests/test_distributed_transport.mojo`.

## 1. What is implemented and what is not

This is the first section on purpose. The distinction matters more here than
anywhere else in the repository, because a transport that is described but not
written reads exactly like one that works.

**Implemented, and covered by 39 passing in-process tests.**

- the wire frame: 42-byte header, little-endian fixed-width fields, FNV-1a
  checksum over the header and the payload together
- payload codecs for `Float64` (bit-exact) and `Int` (zigzag, so negatives
  survive)
- the session state machine: epoch and sequence numbering, in-flight tracking,
  sticky failure and cancellation, membership
- the handshake: protocol version, job id, world size, rank uniqueness, and a
  schema digest over the machine list
- machine list parsing and configuration validation
- the gather-and-broadcast collective, both as pure step functions and as a
  blocking driver that conforms to `Collective`
- deadline expiry, worker loss, and their terminal semantics
- the histogram all-reduce cost contract and buffer checks
- global split agreement digests
- checkpoint metadata encoding and restart compatibility rules

**Not implemented.**

- a socket `ByteEndpoint`. Mojo's standard library exposes no socket module at
  the version this repository pins, and the repository has no foreign function
  interface precedent to borrow, so writing one here would mean shipping
  untested syscall bindings underneath a protocol that is otherwise fully
  tested. Section 7 specifies exactly what it owes.
- connection establishment: no listen, no connect, no retry, no discovery
- MPI and NCCL adapters. Both would implement `ByteEndpoint` and neither is
  started; see section 10.
- authentication and encryption. Section 9.
- any integration with `train_distributed`, which needs none. See section 11.

**Therefore not claimed.** That a real cluster would work. That a real
cluster would be fast. That the deadlines are the right values. That the
gather-and-broadcast topology is the right one at scale, which section 8 says
plainly it is not.

`MemoryEndpoint` is a fake, is named as one, and is not a transport.

## 2. Layering

```
distributed.mojo                grows trees, knows nothing about transports
   |
   | Collective  (collective.mojo)
   v
TransportCollective             blocking driver: gather at root, broadcast back
   |
   | protocol steps             pure functions over List[UInt8]
   |   contribution_frame_*     encode this rank's contribution
   |   absorb_*                 fold one peer's frame into the root's sum
   |   result_frame_*           encode the root's answer
   |   adopt_result_*           take the root's bytes verbatim
   |
   | TransportSession           ordering, deadlines, failure, membership
   | frames                     magic, version, checksum, lengths
   |
   | ByteEndpoint               <-- the only thing that touches a wire
   v
MemoryEndpoint (a fake)         TcpEndpoint (not written)
```

Two properties of this layering are load bearing.

**Everything above `ByteEndpoint` is pure logic over byte buffers.** It needs
no process, no port, and no sleep to be tested, which is why the test suite can
cover the protocol completely and instantly. When a socket endpoint is written,
the only new thing to test is the socket.

**`ByteEndpoint` is smaller than `Collective`.** A conforming endpoint is a
blocking, ordered, reliable stream of bytes to exactly one peer: five methods,
all of which a connected TCP socket already provides. Nothing about
histograms, ranks, reductions, or trees appears below that line.

## 3. The wire frame

Little-endian throughout, written with arithmetic rather than shifts so the
encoding cannot pick up a host word size by accident.

```
offset  size  field
     0     4  magic          0x4D4A4254, "MJBT"
     4     2  version        protocol version, currently 1
     6     2  msg_type       hello, contribution, result, abort
     8     8  job_id         which run this frame belongs to
    16     4  epoch          checkpoint epoch
    20     4  seq            collective sequence within the epoch
    24     4  sender         rank that wrote the frame
    28     2  op             sum_f64, sum_int, max_int, barrier
    30     4  n_elements     buffer length in elements
    34     4  payload_len    bytes that follow
    38     4  checksum       FNV-1a 32 over bytes 0..38 and the payload
    42  ...   payload
```

Four decisions worth defending.

**The checksum covers the header, not just the payload.** A corrupted length
field would otherwise be detected only when the next frame failed its magic
check, which reports the corruption one frame late and blames the wrong frame.

**`n_elements` and `payload_len` are both present and are cross-checked.** The
redundancy is caught on the sender in `encode_frame`, which is the only side
that can still tell which of the two numbers was wrong. On the receiver a
disagreement is a corrupt frame.

**`payload_len` is bounded before anything allocates.** `frame_payload_len`
runs between the header read and the body read, and refuses anything over
`MAX_PAYLOAD_BYTES` (64 MiB, which is a 100-feature 255-bin histogram more
than three hundred times over). A corrupted length asks for a bounded
allocation, not an unbounded one.

**Integers are zigzag encoded.** `agree_equal_ints` in `collective.mojo`
reduces every value alongside its negation, so half of every configuration
agreement message is negative by construction. A plain unsigned cast would
corrupt it silently.

## 4. Ordering and the session state machine

`TransportSession` owns four things the reduction itself does not.

**Ordering.** `epoch` and `seq` number the collectives. `seq` advances once per
completed collective and never repeats within an epoch. A frame whose pair does
not match what the receiving rank is committed to is rejected, not reduced.
This is requirement 5 of `docs/distributed.md` section 5, no reordering across
calls, as a runtime check rather than a convention: one node's histogram cannot
overtake the next and be summed into the wrong node.

**Deadlines.** Every collective gets one at `begin`, from a monotonic duration
and never from wall clock time, so a clock step on one host cannot expire a
collective on another. Expiry is terminal: a timed-out collective cannot be
retried, because the peers that did answer have already advanced their sequence
numbers. Tests drive a manual clock, which is what makes timeout behavior
testable without waiting.

**Termination.** `FAILED` and `CANCELLED` are sticky. Once a session leaves the
ready and in-flight pair it never returns, so a rank that has given up cannot
rejoin a collective the others are mid-way through and contribute stale data.
This is requirement 4, fail-stop and not partial, expressed as a state machine.
Cancellation is kept distinct from failure so an operator-requested stop is not
reported as a broken peer.

**Membership.** `alive` records which ranks are present. Losing one is fatal by
design. `lost_rank()` returns the lowest missing rank so every rank names the
same one, matching what `agree_status` already does for validation failures.

States: `INIT`, `READY`, `IN_FLIGHT`, `FAILED`, `CANCELLED`, `CLOSED`. A
collective before the handshake, an epoch boundary inside a collective, and a
`finish` with nothing in flight are all refused.

## 5. The handshake

Before the first collective, every rank announces a `HandshakeRecord`: protocol
version, its own rank, the world size, the job id, a schema digest, and the
restart epoch. `handshake_status` compares a peer's announcement to the local
one and returns a status code, checked in order of how badly the run is broken
so the reported code is the most fundamental disagreement rather than the first
field to differ.

The schema digest is FNV-1a over the protocol version, the world size, and
every address in the machine list. It deliberately covers the addresses, so a
rank started against a stale machine list fails the handshake instead of
silently talking to whoever answers on that port. It deliberately does not
cover the local rank, so every rank computes the same digest from the same
file, and it does not cover timeouts, because a slower host is allowed to wait
longer.

`complete_handshake` additionally requires exactly one record per peer with
distinct ranks, which catches two processes that were both started as rank 2.

## 6. The collective: gather at the root, broadcast back

For a world of size `W`, rank 0 holds `W - 1` endpoints and every other rank
holds one.

```
leaf rank r:            root rank 0:
  send contribution       for i in 1..W-1:          <- ascending, blocking
  receive result            receive contribution
  adopt it verbatim         fold into the accumulator
                          for i in 1..W-1:
                            send the identical bytes
```

**Why not a ring.** Requirement 1 of `docs/distributed.md` section 5 forbids a
scheme that leaves different ranks holding different roundings of the same sum,
which rules out the usual ring all-reduce with per-rank rotation. Here the root
folds and every other rank adopts the root's bytes unchanged, so bit-identical
delivery is structural rather than hoped for. The test asserts the two peers
received byte-for-byte identical buffers.

**Ascending rank order** is requirement 2, and it comes from two facts
together: `peers[i]` is rank `i + 1` and the loop runs forward, and the read is
blocking, so the fold order is rank order regardless of which peer answered
first. That independence from arrival order is the reproducibility claim in
`docs/distributed.md` section 6. The test proves the order is load bearing by
folding `1.0`, `1e-16`, `1e-16` in both directions and asserting the results
differ in the last bit.

**A barrier is a real round trip** with an empty payload. Making it free would
turn an epoch boundary into a point no rank had actually reached.

**`n_local_ranks()` is 1.** A real transport hosts one rank per process, so the
local-rank loop in `distributed.mojo` degenerates to a single iteration, which
is the case that loop was written to cover.

## 7. What a socket endpoint owes

This is the one piece that is not written. It is a `ByteEndpoint`
implementation and nothing more:

```mojo
trait ByteEndpoint:
    def peer_rank(self) -> Int
    def is_open(self) -> Bool
    def send_all(mut self, bytes: List[UInt8], deadline_ns: Int) raises
    def recv_exact(mut self, n: Int, deadline_ns: Int) raises -> List[UInt8]
    def close(mut self) raises
```

Three requirements the protocol above depends on:

1. `send_all` writes every byte or raises. A short write reported as success
   desynchronizes the stream permanently.
2. `recv_exact` returns exactly `n` bytes or raises. Returning fewer makes a
   frame boundary a guess.
3. Bytes arrive in the order they were sent, none lost, duplicated, or
   reordered. Framing detects violations but cannot repair them.

`deadline_ns` is an absolute monotonic deadline, not a duration, so the same
value can be handed to the header read and the body read without the body
silently getting a fresh timeout.

A TCP implementation is the obvious smallest one:

- `socket(AF_INET, SOCK_STREAM, 0)`, `setsockopt(SO_REUSEADDR)`, `bind`,
  `listen`, `accept` on the root; `socket` and `connect` with retry on every
  other rank, against the address the machine list gives for rank 0
- `setsockopt(SO_RCVTIMEO)` and `SO_SNDTIMEO` derived from `deadline_ns` minus
  now, refreshed before each syscall, so a deadline is enforced rather than
  advisory
- `TCP_NODELAY`, because the protocol is request and response and Nagle would
  add a round trip of latency per collective
- loops around `send` and `recv` for short writes and short reads, which are
  normal on a stream socket and are what makes `send_all` and `recv_exact`
  nontrivial
- `close` on both a clean shutdown and any error, and `EINTR` retried rather
  than treated as failure

The blocker is that Mojo's standard library exposes no socket module at the
version this repository pins, so this means `external_call` bindings to libc
plus a hand-built `sockaddr_in`, and the repository has no foreign function
interface anywhere else to model it on. That is a real piece of work with its
own failure modes, and doing it inside this change would have meant an
untested layer underneath a tested one.

**The hermetic two-process test it has to pass.** A shell or Mojo driver that
starts `W` processes on `127.0.0.1` with ports the test allocates, each given
the same machine list and a distinct rank, and asserts:

- a two-rank all-reduce of known buffers returns the identical bytes on both
  ranks, compared as bits and not as floats
- the same run repeated gives the same bits
- world sizes 1, 2, and 3 agree with `LocalCollective` on the same
  contributions
- a rank killed mid-collective makes every survivor fail with the peer-loss
  message naming the killed rank, and none of them hangs
- a rank started against a rewritten machine list is refused at the handshake
- a rank that connects and then stops reading trips the collective deadline
  rather than blocking forever
- the test binds to `127.0.0.1` only, allocates its own ports, and cleans up
  every process it starts, so it can run in CI without a network

Until that test exists and passes, no statement about multi-process operation
belongs in this repository.

## 8. Communication cost, and what this topology costs

`histogram_plan` computes the cost model from `docs/distributed.md` section 8
rather than restating it, so the test pins the model against the wire format:
at 100 features and 255 bins, 25500 cells, three reductions per node, 612000
payload bytes per node, plus 126 bytes of framing.

Three reductions per node and not one, because gradients, hessians, and counts
stay in their own typed buffers and the exactness of the counts is then
obvious. Packing gradient and hessian into one message, or all three into one
buffer of doubles since counts below 2^53 are exact in a double, is a factor of
three fewer round trips at identical arithmetic. It is left to a transport that
has round trips to save.

The gather-and-broadcast topology is the latency-optimal and bandwidth-pessimal
end of the trade, and this should be said plainly rather than discovered later:

| topology | messages per collective | bytes through the root |
| --- | --- | --- |
| gather and broadcast | `2 (W - 1)` | `2 (W - 1) N` |
| ring all-reduce | `2 W` | `2 N` per link |
| reduce-scatter plus all-gather | `2 W` | `2 N` per link |

The root is a bottleneck whose load grows linearly in the world size, so this
topology is right for a first transport, where being obviously correct matters
more than being fast, and wrong for a large one. Replacing it means replacing
the driver, not the framing or the session, and the replacement has to keep
requirement 1: a ring with per-rank rotation leaves different ranks holding
different roundings and cannot be used as written.

## 9. Security

There is none, and that is a gate rather than an omission.

The frame checksum is FNV-1a. It catches a truncated read, a misaligned frame
boundary, and a flipped bit. It authenticates nothing. Any process that can
connect to the root's port and speak the protocol can inject contributions into
a reduction, and there is no transport encryption, so histogram statistics
derived from the training data cross the network in the clear.

This confines a real transport to a trusted network: a single cluster behind a
boundary the operator controls, not the open internet and not a shared
multi-tenant fabric. Adding TLS or a shared-secret MAC is a separate change,
and it belongs after the two-process test in section 7 exists, not before.

## 10. MPI and NCCL

Both would be `ByteEndpoint` implementations and neither is started.

MPI is the awkward one, and the awkwardness is worth recording. MPI already
provides `MPI_Allreduce`, so wrapping it as a byte stream and then rebuilding
an all-reduce on top would be strictly worse than calling it. The right MPI
adapter conforms to `Collective` directly rather than to `ByteEndpoint`, and
has to satisfy requirement 1 itself: `MPI_Allreduce` does not guarantee
bit-identical results across ranks under every implementation and reduction
tree, so an MPI adapter would need `MPI_Reduce` to rank 0 followed by
`MPI_Bcast`, which is the same topology this driver already uses.

NCCL is for GPU buffers and is gated behind the discrete-GPU benchmark in
`docs/distributed.md` section 9. Adding a network layer under a backend that
has not yet been shown to be worth using on one node would be building on an
unproven foundation.

## 11. Relationship to the existing distributed code

Nothing in `distributed.mojo` or `collective.mojo` changes. That is the point
of the collective contract, and it is worth stating as a result rather than as
an intention: `TransportCollective` conforms to `Collective`, so
`train_distributed` and `grow_tree_distributed` accept one today without a line
of edit, and the test suite demonstrates it by running `agree_equal_ints`, the
real configuration agreement protocol, over the transport rather than over a
mock.

The integration this leaves open is not code, it is the two-process test and
the socket underneath it.

## 12. Checkpoint and restart

`CheckpointMeta` is metadata only. Nothing here writes a model, and
`docs/distributed.md` section 7 is explicit that fault tolerance is a separate
project. What this does is make a wrong restart an error rather than a model
whose provenance nobody can reconstruct.

A restart is refused unless the job id, the world size, and the schema digest
all match. World size because the row partition is a function of it, so
resuming at a different one would silently re-shard the data underneath a
half-built model. Schema because the rank assignment lives in the machine list,
and a rank resuming against a rewritten list would take over another rank's
shard.

`resume_session` starts the next epoch at sequence zero rather than resuming
mid-epoch, because a checkpoint is only ever taken between boosting rounds and
a resumed sequence number would have to be agreed across ranks that may have
written their checkpoints one collective apart.

## 13. Testing

`tests/test_distributed_transport.mojo`, 39 tests, all in one process
against in-memory endpoints and a manual clock.

The interesting ones, in the sense of tests that could fail for a real reason:

- three ranks run a whole collective end to end through the pure step
  functions, with no threads and no endpoints, and all three end with
  bit-identical results equal to the ascending-order reduction
- the root driver's two peers receive byte-for-byte identical buffers
- ascending and descending folds of the same three contributions differ, so
  the order requirement is proven to be load bearing rather than asserted
- `agree_equal_ints`, the real agreement protocol from `collective.mojo`, runs
  over the transport and correctly reports which value the ranks disagree about
- a frame from the next collective is refused and the session fails
- a flipped payload byte, a flipped header byte, a bad magic, and a truncated
  frame are each rejected with the right reason
- a peer whose stream ends mid-frame is declared lost, the lowest lost rank is
  named, and the session is terminal afterwards
- a peer that stops answering trips the deadline
- a restart against a rewritten machine list, a different world size, or a
  different job is refused

`MemoryEndpoint` is primed with the bytes a correct counterpart would have
sent, which is what lets a single-threaded test drive both sides of a blocking
protocol without deadlocking on itself. That is also its limit: it proves the
protocol logic and proves nothing about a socket.
