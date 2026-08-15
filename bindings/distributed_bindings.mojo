"""Distributed runtime capability, configuration, and status vocabulary.

Three native modules stand behind this one, and they are at three
different stages:

- `src/mojotrees/collective.mojo` defines what a collective owes a
  distributed trainer, and `LocalCollective`, which delivers it inside one
  process. Real, reachable, and single-process by construction.
- `src/mojotrees/distributed.mojo` grows trees and trains over shards
  against any `Collective`. Real, and generic over the collective, so it
  runs today with the local one.
- `src/mojotrees/distributed_transport.mojo` is the wire protocol two
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

from mojotrees.binning import fit_bins
from mojotrees.boosting import BoosterParams
from mojotrees.callback import no_callback
from mojotrees.collective import LocalCollective
from mojotrees.collective import status_message as collective_status_message
from mojotrees.distributed import (
    DataShard,
    DistributedRunOptions,
    partition_rows,
    train_distributed_run,
)
from mojotrees.distributed_gpu import (
    distributed_gpu_available,
    distributed_gpu_gates,
    distributed_gpu_unavailable_detail,
    gate_name,
)
from mojotrees.distributed_strategies import (
    STRATEGY_FEATURE_PARALLEL,
    parse_strategy,
)
from mojotrees.distributed_transport import (
    TRANSPORT_PROTOCOL_VERSION,
    parse_machine_list,
    transport_available,
    transport_validated,
    transport_status_message as mojo_transport_status_message,
)
from mojotrees.model import Model


# `multi_process` is read from the transport module, not stated here:
# `transport_validated()` is the honest answer to "can two processes train
# together", and it is False until the two-process procedure in
# docs/DISTRIBUTED_TRANSPORT.md section 7 has been run. `transport_available()`
# (a socket endpoint is compiled in) is reported beside it so a caller can
# tell "unbuilt" from "unrun".
comptime _NO_TRANSPORT_REASON = String(
    "multi-process distributed training is not validated in this build: the"
    " wire protocol and a socket endpoint exist in"
    " src/mojotrees/distributed_transport.mojo, but no two processes have"
    " trained together yet (transport_validated() is False). Single-process"
    " training over LocalCollective, with tree_learner data, feature, or"
    " voting, is what this build runs"
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
    var validated = transport_validated()
    out["multi_process"] = PythonObject(validated)
    out["transport_available"] = PythonObject(transport_available())
    out["transport_validated"] = PythonObject(validated)
    out["local_collective"] = PythonObject(True)
    out["protocol_version"] = PythonObject(TRANSPORT_PROTOCOL_VERSION)
    var learners: List[String] = ["serial", "data", "feature", "voting"]
    out["tree_learners"] = py_str_list(learners)
    if validated:
        out["max_world_size"] = PythonObject(-1)
        out["reason"] = PythonObject("")
    else:
        out["max_world_size"] = PythonObject(1)
        out["reason"] = PythonObject(_NO_TRANSPORT_REASON)
    return out^


def distributed_gpu_status() raises -> PythonObject:
    """What stands between this build and distributed GPU histogram
    exchange, from src/mojotrees/distributed_gpu.mojo: `available`, the
    `gates` still closed by name, and the `detail` text."""
    var out = py_dict()
    out["available"] = PythonObject(distributed_gpu_available())
    var mask = distributed_gpu_gates()
    var closed = List[String]()
    var bit = 1
    while bit <= mask:
        if mask & bit != 0:
            closed.append(gate_name(bit))
        bit *= 2
    out["gates"] = py_str_list(closed)
    out["detail"] = PythonObject(distributed_gpu_unavailable_detail())
    return out^


def train_local_world[
    features_origin: ImmOrigin, //
](
    features: Span[Float64, features_origin],
    n_rows: Int,
    n_features: Int,
    target: List[Float64],
    objective: Int,
    params: BoosterParams,
    max_bins: Int,
    weights: List[Float64],
    alpha: Float64,
    world_size: Int,
    tree_learner: String,
    top_k: Int,
) raises -> Model:
    """Train over a world of `world_size` ranks hosted in this process, with
    LightGBM's `tree_learner` by name. Data and voting parallel partition the
    rows contiguously across ranks; feature parallel hands every rank the
    whole dataset, which is that mode's arrangement. Binning is the same
    `fit_bins` the single-node trainer uses, so the returned Model predicts
    and saves like any other."""
    if world_size < 1:
        raise Error("num_machines must be at least 1")
    var mapper = fit_bins(features, n_rows, n_features, max_bins)
    var data = mapper.transform(features, n_rows)
    var options = DistributedRunOptions()
    options.tree_learner = parse_strategy(tree_learner)
    options.top_k = top_k
    var shards: List[DataShard]
    if options.tree_learner == STRATEGY_FEATURE_PARALLEL:
        shards = List[DataShard](capacity=world_size)
        for _ in range(world_size):
            shards.append(
                DataShard(data.copy(), target.copy(), weights.copy())
            )
    else:
        shards = partition_rows(data, target, world_size, weights)
    var comm = LocalCollective(world_size)
    var outcome = train_distributed_run(
        shards, objective, params, comm, options, no_callback, alpha
    )
    return Model(mapper^, outcome.model.copy())


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
