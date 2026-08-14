"""Distributed runtime capability, configuration, and status vocabulary.

Three native modules stand behind this one, and they are at three
different stages:

- `src/mojoboost/collective.mojo` defines what a collective owes a
  distributed trainer, and `LocalCollective`, which delivers it inside one
  process. Real, reachable, and single-process by construction.
- `src/mojoboost/distributed.mojo` grows trees and trains over shards
  against any `Collective`. Real, and generic over the collective, so it
  runs today with the local one.
- `src/mojoboost/distributed_transport.mojo` is the wire protocol two
  processes would agree on: framing, checksums, sequence and epoch
  numbering, the handshake, deadlines, cancellation, worker loss, and the
  restart record. All of it is implemented over `List[UInt8]` and none of
  it has moved a byte between two processes, because the one piece that
  would is a socket `ByteEndpoint`, which does not exist: the pinned Mojo
  exposes no socket module and this repository has no FFI precedent to
  borrow. That module says so in its own header, at length.

So the capability this module reports is deliberately fail-closed: the
configuration and the protocol validate, and multi-process training is
*not* available. A Python caller that asks for distributed training gets
that answer and an explanation, rather than a run that silently trains on
one shard.

What crosses here is configuration text and status codes. No frame, no
buffer, and no endpoint is exposed to Python, and nothing here opens
anything.
"""

from std.python import PythonObject

from binding_support import nonnegative, py_dict, py_str_list

from mojoboost.collective import status_message as collective_status_message
from mojoboost.distributed_transport import (
    TRANSPORT_PROTOCOL_VERSION,
    parse_machine_list,
    transport_status_message as mojo_transport_status_message,
)


# The one fact this module states rather than reads.
#
# There is no native predicate for "a wire transport exists" because there
# is nothing yet to predicate on: a socket `ByteEndpoint` is an absent
# implementation, not a disabled one. Flipping this to True is not the way
# to enable distributed training; the way is to add the endpoint and, with
# it, `distributed_runtime_capability()` in distributed_transport.mojo,
# which this then reads instead. The exact patch is in
# `handoffs/connect_14_bindings.md`.
comptime _HAS_WIRE_TRANSPORT = False

comptime _NO_TRANSPORT_REASON = String(
    "multi-process distributed training is not available in this build: the"
    " wire protocol in src/mojoboost/distributed_transport.mojo is"
    " implemented and exercised in process, but it has no socket endpoint"
    " to run over, so no rank can reach another. Single-process training"
    " over LocalCollective is what this build can run"
)


def distributed_capability() raises -> PythonObject:
    """What this build's distributed stack can actually do.

    Keys:

    - `multi_process`: whether two processes can train together. False in
      every build today, for the reason in `reason`.
    - `local_collective`: whether in-process distributed training over
      `LocalCollective` is available. True: `distributed.train_distributed`
      is generic over the collective and the local one is real.
    - `protocol_version`: the transport protocol's version, which every
      rank of a job must agree on.
    - `max_world_size`: the largest world this build can form. 1 while
      there is no transport, so a caller sizing a cluster gets a number
      rather than a promise.
    - `reason`: empty when `multi_process` is True, and the explanation to
      report otherwise.

    A caller that refuses to run distributed work should refuse on
    `multi_process` and quote `reason`. It should not infer availability
    from the presence of this function: the function exists in order to
    say no.
    """
    var out = py_dict()
    out["multi_process"] = PythonObject(_HAS_WIRE_TRANSPORT)
    out["local_collective"] = PythonObject(True)
    out["protocol_version"] = PythonObject(TRANSPORT_PROTOCOL_VERSION)
    if _HAS_WIRE_TRANSPORT:
        # Unreachable today, and left unanswered rather than guessed: a
        # build with a transport knows its own world limit and must report
        # it from `distributed_runtime_capability()`, not from here.
        out["max_world_size"] = PythonObject(-1)
        out["reason"] = PythonObject("")
    else:
        out["max_world_size"] = PythonObject(1)
        out["reason"] = PythonObject(_NO_TRANSPORT_REASON)
    return out^


def distributed_check_machine_list(
    machines: PythonObject, rank: PythonObject, job_id: PythonObject
) raises -> PythonObject:
    """Validate a LightGBM-shaped machine list and report what it means.

    `machines` is the file's text: one `host:port` per line, blank lines
    skipped, `#` starting a comment. Rank order is file order, so every
    rank reading the same file agrees about who is who without any of them
    being asked.

    Returns `world_size`, the `addresses` in rank order, `is_root`, and
    `schema_digest`, the value every rank must agree on for the handshake
    to succeed. The digest crosses as a decimal string because it is a
    64-bit unsigned value that a signed integer boundary cannot carry, and
    because every consumer of it compares rather than computes.

    Raises with the transport's own message for a malformed entry, a rank
    outside the world, or a duplicate address. Validating costs nothing
    and is worth doing whatever the capability says: a machine list is
    usually written long before a job runs.
    """
    var r = nonnegative(rank, "rank")
    var job = nonnegative(job_id, "job_id")
    var config = parse_machine_list(String(py=machines), r, UInt64(job))
    var addresses = List[String](capacity=config.world_size)
    for i in range(len(config.addresses)):
        addresses.append(config.addresses[i].text())
    var out = py_dict()
    out["world_size"] = PythonObject(config.world_size)
    out["rank"] = PythonObject(config.rank)
    out["is_root"] = PythonObject(config.is_root())
    out["addresses"] = py_str_list(addresses)
    out["schema_digest"] = PythonObject(String(config.schema_digest()))
    return out^


def distributed_status_message(code: PythonObject) raises -> PythonObject:
    """The text for a collective status code: what a distributed trainer
    reports when a shard's input disagrees with the others."""
    return PythonObject(collective_status_message(Int(py=code)))


def transport_status_message(code: PythonObject) raises -> PythonObject:
    """The text for a transport status code: what a session reports when
    it is the connection that broke rather than the input.

    The two vocabularies are deliberately separate, so "the peer went
    away" is never confused with "this shard's weights are negative".
    """
    return PythonObject(mojo_transport_status_message(Int(py=code)))
