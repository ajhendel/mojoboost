# Connect 13: real distributed transport to distributed training

Owned files, and the only files edited:

- `src/mojoboost/collective.mojo` (240 -> 290 lines)
- `src/mojoboost/distributed_transport.mojo` (1844 -> 3220 lines)
- `src/mojoboost/distributed.mojo` (930 -> 1838 lines)
- `handoffs/connect_13_distributed.md` (this file)

A second pass added the piece the first pass named as missing: a real socket
`ByteEndpoint` and the rendezvous over it, so `RUNTIME_TRANSPORT` now opens
instead of refusing. Section 6a is the whole of what that pass added and,
more importantly, what it does *not* prove. Read it before quoting anything
from this file as a distributed capability.

`src/mojoboost/collectives.mojo` does not exist and was not created. Nothing
was committed, staged, reverted, or reformatted. No build, test, benchmark, or
network command was run: everything below is static reading, and every claim
about behavior is a claim about the code as written, not about an observed run.

## 1. Implementations found

Five things claimed a piece of this capability before the change.

| Where | What it is | State before |
| --- | --- | --- |
| `collective.mojo` | the `Collective` trait, `LocalCollective`, `agree_status`, `agree_equal_ints` | authoritative contract, reachable, used by the trainer |
| `distributed.mojo` | data-parallel grower and booster over any `Collective` | authoritative trainer, reachable only if a caller built shards and a collective by hand |
| `distributed_transport.mojo` | frames, session state machine, handshake, ordered reduction, `TransportCollective`, histogram cost plan, split digest, checkpoint record | fully written, entirely unreferenced by the trainer, and unopenable (no `ByteEndpoint` that reaches a process). The second pass added that endpoint; see section 6a |
| `python/mojoboost/dask.py` | `WorldPlan`, `RankAssignment`, `TrainingJob`, backend registry | client-side contract with no backend; `get_backend()` raises `DistributedNotAvailable` |
| `bindings/distributed_bindings.mojo` (another lane, uncommitted at the time) | `distributed_capability()` reporting `_HAS_WIRE_TRANSPORT = False` | a second copy of the availability fact, with its own wording |

`src/mojoboost/distributed_strategies.mojo` (another lane) turned out to be
complementary, not a duplicate: it is feature-parallel and voting-parallel
*cores*, no growth loop and no second `Collective`. It already imported
`hosts_whole_world` from `collective.mojo` and already referenced
`transport_available` in `distributed_transport.mojo` — neither of which
existed yet. Both now exist with exactly those names and semantics, so that
lane's imports resolve rather than needing a patch.

The authoritative implementation selected, per layer: `Collective` for the
contract, `distributed.mojo` for training, `distributed_transport.mojo` for the
wire and the runtime seam. No new trainer, registry, policy engine, or model
representation was introduced.

## 2. Call path before and after

**Before.** A caller had to write, by hand, in this order: build a
`BinnedMatrix`, call `partition_rows`, construct a `LocalCollective`, call
`train_distributed`. Nothing read a rank id, a world size, or a machine list.
`TransportCollective` conformed to `Collective` and therefore *could* have been
passed to `train_distributed`, but no code path constructed one, no endpoint
could open, and the transport's schema digest, histogram plan, split digest,
epoch numbering, and checkpoint record were referenced only by
`tests/parallel/test_distributed_transport.mojo`.

**After.** One entry point:

```
RuntimeSpec (from runtime_from_env / local_runtime / transport_runtime)
  -> run_distributed(spec, shards, objective, params, options, on_iteration)
       -> require_transport(spec)            # refuses a platform with no endpoint
       -> open_local_collective(spec)        # RUNTIME_LOCAL
          or open_socket_collective(spec)    # RUNTIME_TRANSPORT: connect_world
                                             # then open_transport_collective
       -> train_distributed_run(...)         # the one training loop
            -> agree config + schema digest  # one reduction
            -> agree target statuses
            -> resolve_partition -> ShardPlan (+ status agreement)
            -> base score reduction
            -> per round: grad/hess, _grow_tree_distributed,
                          optional split agreement per node,
                          optional global metric + early stopping,
                          optional checkpoint barrier + CheckpointMeta,
                          callback verdict agreed across ranks
       -> DistributedOutcome(model, report)
```

`train_distributed` is now `train_distributed_run` with
`DistributedRunOptions()` and `no_callback`, and returns the `Booster` alone.
Its signature is unchanged.

## 3. Connections completed

Each of these was a written-but-unreferenced piece of the transport that now
sits on the training path.

1. **Rank and world configuration.** `RuntimeSpec` in
   `distributed_transport.mojo` carries mode, rank, world size, machine list,
   job id, timeouts, and restart epoch, and validates all of them before
   anything opens. `runtime_from_env()` reads it from `MOJOBOOST_DIST_MODE`,
   `MOJOBOOST_DIST_WORLD_SIZE`, `MOJOBOOST_DIST_RANK`,
   `MOJOBOOST_DIST_MACHINES`, `MOJOBOOST_DIST_JOB_ID`,
   `MOJOBOOST_DIST_TIMEOUT_S`, `MOJOBOOST_DIST_RESTART_EPOCH`. In transport
   mode the world size comes from the machine list, so the two cannot
   disagree. `local_runtime` and `transport_runtime` are the explicit
   constructors.

2. **Schema negotiation.** `run_schema_digest` / `tree_schema_digest` in
   `distributed.mojo` build a `digest_ints` digest over the binned grid (per
   feature: categorical flag and missing bin), the tree scalars, the
   unsupported mask, the objective, `n_estimators`, the learning rate bits, and
   `alpha`. `_agree_config_and_schema` folds the two 32-bit halves into the
   *existing* configuration reduction, so schema agreement costs no extra
   message. Doubles enter by their bits, so a learning rate differing in the
   last place is a different schema.

3. **Partition metadata.** `ShardPlan` (offsets and row counts per rank) is now
   the single definition of the block boundary: `partition_rows` derives its
   `start`/`rows` from `ShardPlan.contiguous` rather than recomputing
   `r * n // W`. `resolve_partition` negotiates the plan from per-rank row
   counts with one integer reduction, so a rank learns its global row offset
   instead of assuming it. `_partition_statuses` then checks the agreed plan
   against the rows each rank actually holds, via `agree_status`, using the new
   `STATUS_PARTITION_MISMATCH`. `ShardPlan` deliberately mirrors
   `RankAssignment.row_offset` / `n_rows` in `python/mojoboost/dask.py`.

4. **Histogram aggregation.** `allreduce_histogram` now calls `histogram_plan`
   and `check_histogram_buffers` before reducing, so the transport's cost
   contract validates the buffers it describes. A mismatched histogram is
   refused locally by the rank that built it rather than reaching peers as a
   length disagreement attributed to the wrong rank. Three integer comparisons
   per node, no messages.

5. **Global split agreement.** `_agree_split` computes `split_digest` (exact in
   the gain bits) and compares it across ranks with `agree_equal_ints`, at the
   root and both children of every split. Opt-in via
   `DistributedRunOptions.verify_splits`, and a no-op when one process hosts
   the whole world.

6. **Deterministic reduction order.** Unified rather than re-argued:
   `collective.add_into_f64` / `add_into_int` / the new `max_into_int` are now
   the only element-wise reduction loops in the repository, and
   `distributed_transport.reduce_into_f64` / `reduce_into_int` dispatch to them
   while keeping the transport's error shape. Cross-process and cross-local-rank
   folding therefore go through the same code, which makes ascending-rank-order
   one claim instead of two.

7. **Objective and metric aggregation.** `distributed_metric` reduces
   `(sum of per-shard mean loss * rows, sum of rows)` in one two-element
   reduction and divides, reusing `boosting._mean_loss` so the distributed and
   single-node stopping criteria are the same quantity. Empty shards contribute
   nothing rather than dividing by zero. It is not bit-identical to the
   single-node mean and the docstring says so.

8. **Early stopping.** `early_stopping_rounds` / `early_stopping_delta` on the
   global metric. The decision is a function of a reduced scalar, so it is
   unanimous by construction; no rank can leave the loop while another blocks
   in a collective.

9. **Cancellation and failure propagation.** Two mechanisms, both honest about
   what they are. `TransportCollective.request_cancel` makes the session's
   sticky `CANCELLED` state raise at this rank's next collective (peers then see
   a lost peer or a deadline — it is cancellation, not a cancellation
   broadcast, and the docstring says so). Cooperative cancellation is the
   callback seam: `train_distributed_run` takes an `IterationFn` from
   `callback.mojo`, calls it at `BEFORE_ITERATION` and `AFTER_ITERATION`, and
   reduces the returned control code with `allreduce_max_int` so `ABORT` beats
   `STOP` beats `CONTINUE` and every rank acts on the same verdict at the same
   round. A callback that writes `env.params` is refused by
   `_check_params_unchanged` rather than silently ignored.

10. **Timeouts and worker loss.** Already implemented in `TransportSession`;
    now reachable through the runtime seam (`collective_timeout_ns` on
    `RuntimeSpec`, threaded into `TransportConfig`). `shutdown()` closes every
    endpoint and the session, tolerating an already-broken peer.

11. **Checkpoint boundaries.** `checkpoint_every` issues `comm.barrier()` — a
    point every rank must reach — and records a real `CheckpointMeta` carrying
    the job id, the runtime schema digest, a `model_digest` of the trees so far,
    the world size, and the tree count. `model_digest` digests structure and
    leaf-value bits, so it identifies the model rather than labelling it.
    `TransportCollective.checkpoint_boundary()` (barrier then `next_epoch`) and
    `session_checkpoint_meta()` are the transport-side counterparts.

12. **Ranking-group constraints.** `check_group_alignment(plan, groups)`
    implements the rule that a query group must be owned entirely by one rank,
    as a merge over the plan's boundaries. Supplying `options.groups` also sets
    the new `_UNSUPPORTED_RANKING` mask bit, so the run is refused with a
    message that names the reason (a comparison is not a sum) and points at the
    constraint function. The constraint is checked before the refusal, so a
    client partitioning for a future lambdarank transport learns whether its
    partition is legal.

13. **Explicit availability, and explicit non-validation.**
    `HAS_BYTE_ENDPOINT = SOCKETS_SUPPORTED`, `transport_available()`,
    `transport_validated()`, `transport_unavailable_detail()`,
    `transport_unvalidated_detail()`, `TRANSPORT_UNAVAILABLE = 12`,
    `require_transport()`, and `distributed_runtime_capability()` are the
    single place the build answers "can two processes train together" and the
    separate question "has that ever been run". The second answer is False and
    is carried as `RuntimeCapability.validated` plus a non-empty `caveat`.

14. **The socket transport itself.** `SocketEndpoint`, `SocketListener`,
    `connect_to_root`, `connect_world`, and `open_socket_collective` in
    `distributed_transport.mojo`, and the `RUNTIME_TRANSPORT` branch of
    `run_distributed` in `distributed.mojo`. Section 6a.

## 4. The one narrow runtime interface

For bindings, Dask, and any launcher:

```mojo
# distributed_transport.mojo
RUNTIME_LOCAL, RUNTIME_TRANSPORT, HAS_BYTE_ENDPOINT, SOCKETS_SUPPORTED
RuntimeSpec{mode, rank, world_size, addresses, job_id,
            connect_timeout_ns, collective_timeout_ns, restart_epoch}
    .validate() .transport_config() .schema_digest() .describe()
local_runtime(world_size, job_id) -> RuntimeSpec
transport_runtime(machines, rank, job_id, ...) -> RuntimeSpec
runtime_from_env() -> RuntimeSpec
runtime_mode_name / parse_runtime_mode
transport_available() -> Bool          # a socket endpoint is compiled in
transport_validated() -> Bool          # it has moved a byte: False
transport_unavailable_detail() -> String
transport_unvalidated_detail() -> String
distributed_runtime_capability() -> RuntimeCapability
    {multi_process, validated, local_collective, protocol_version,
     max_world_size, reason, caveat}
require_transport(spec)
open_local_collective(spec) -> LocalCollective
open_transport_collective[E](spec, peers, peer_handshakes) -> TransportCollective[E]
connect_world(spec) -> WorldConnection{peers, handshakes}  # opens sockets
open_socket_collective(spec) -> TransportCollective[SocketEndpoint]
session_checkpoint_meta(session, model_digest, n_trees) -> CheckpointMeta

# distributed.mojo
ShardPlan, resolve_partition, check_group_alignment
DistributedRunOptions, DistributedRunReport, DistributedOutcome
run_distributed(spec, shards, objective, params, options, on_iteration, alpha)
train_distributed_run(shards, objective, params, comm, options, on_iteration, alpha)
train_distributed(shards, objective, params, comm, alpha)   # unchanged
distributed_metric, model_digest, run_schema_digest, tree_schema_digest
```

`RuntimeCapability` was named `distributed_runtime_capability()` deliberately:
`bindings/distributed_bindings.mojo` asks for exactly that name in a comment at
its `_HAS_WIRE_TRANSPORT` definition.

## 5. Duplicates fused or quarantined

- **Fused.** Element-wise reduction: `reduce_into_f64` / `reduce_into_int` in
  the transport were a second copy of `add_into_f64` / `add_into_int`; they now
  delegate, and `max_into_int` was added to `collective.mojo` so the maximum has
  one definition too. The transport keeps only the length check and its error
  shape.
- **Fused.** The block-partition boundary formula existed in `partition_rows`
  and nowhere else as a name; it is now `ShardPlan.contiguous` and
  `partition_rows` reads offsets from it.
- **Quarantined, documented.** `zero_contribution_f64` / `zero_contribution_int`
  remain thin aliases of `zeros_f64` / `zeros_int`. They are one line each and
  the transport's docstring already frames them as re-exports; deleting them
  would touch a test import, so they were left alone.
- **Quarantined, cross-lane.** `bindings/distributed_bindings.mojo` carries
  `_HAS_WIRE_TRANSPORT` and `_NO_TRANSPORT_REASON`, a second copy of the
  availability fact. Not mine to edit; patch request in section 7.
- **Deliberately not fused.** `collective.mojo`'s `STATUS_*` and the transport's
  `TRANSPORT_*` codes stay separate. The existing rationale (input errors versus
  session errors are raised by different layers) is right, and merging them
  would make "the peer went away" indistinguishable from "these weights are
  negative".

## 6. Remaining disconnections

Stated plainly, because this is the part that matters most here.

1. **The socket transport is written and has never been run.** It is no longer
   absent, which is the change; it is still unproven, which is the part that
   must be quoted alongside it. Section 6a is the detail. No byte has moved
   between two processes in this repository, and no multi-host capability,
   throughput, or scaling is claimed.
2. **Epoch advance is not driven by the trainer.** `train_distributed_run` is
   generic over `Collective`, which has no `next_epoch`, so a checkpoint
   boundary inside a run is a `barrier()` plus a `CheckpointMeta`, not an epoch
   increment. `TransportCollective.checkpoint_boundary()` does both and is
   callable by a driver *between* runs. Closing this needs either a second
   trait or a run loop that yields at boundaries; adding a method to
   `Collective` is not available, because `tests/test_distributed.mojo` defines
   its own conformer (`_PeerCollective`) and that file is not mine to edit.
3. **`resume_session` is not wired into training.** A restart is validated
   (`restart_status`, `resume_session`) and a checkpoint is now produced with a
   real model digest, but nothing writes the model bytes at a boundary and
   nothing resumes from one. That is the fault-tolerance project
   `docs/distributed.md` section 7 defers.
4. **Distributed ranking, multiclass, quantile/L1, bagging, GOSS, feature
   fraction, categorical, missing bins, monotone, max_depth, extra trees, and
   forced splits are still refused**, now including ranking with its own mask
   bit and message.
5. **The schema digest is enumerated, not reflective.** A `TreeParams` field
   added after this change is not in `_schema_values` and would not trip the
   agreement. Same weakness `_unsupported_mask` documents; extending it is one
   line.
6. **`python/mojoboost/dask.py` still has no backend.** The Mojo side now has
   an entry point a backend could call; nothing on the Python side calls it.
7. **Two of the three new status codes are vocabulary, not yet raised.**
   `STATUS_PARTITION_MISMATCH` is raised through `_partition_statuses`.
   `STATUS_RANKING_GROUPS` and `STATUS_CANCELLED` exist in `status_message` so
   the bindings lane has one vocabulary to map, but the ranking check is local
   (the query sizes are global metadata, identical on every rank, so it raises
   identically without a reduction) and cancellation currently surfaces through
   the transport's `TRANSPORT_CANCELLED` and the callback control code. Either
   route them through `agree_status` or drop them; do not leave them half-used
   for long.

## 6a. The socket transport: what was written, and what it does not prove

### What it is

`src/mojoboost/distributed_transport.mojo` now contains a BSD socket adapter
reached through `external_call`, sitting under the same `ByteEndpoint` trait
the in-process fake sits under, so nothing above it changed.

| Name | What it does |
| --- | --- |
| `SOCKETS_SUPPORTED` | `CompilationTarget.is_macos() or .is_linux()`. `HAS_BYTE_ENDPOINT` is now exactly this. |
| `parse_ipv4(host)` | Dotted quad, plus `localhost` meaning 127.0.0.1. Every other name is refused. |
| `_sockaddr_in(ip, port)` | 16 bytes, hand written, with the leading `sin_len` byte on macOS and not on Linux. |
| `_timeval(ns)` | 8 byte `tv_sec` then 4 byte `tv_usec`, clamped up to 1 microsecond because a zero timeval means block forever. |
| `SocketEndpoint` | `send_all` / `recv_exact` / `close`, arming `SO_SNDTIMEO` / `SO_RCVTIMEO` from the caller's absolute deadline before every syscall, retrying `EINTR`, mapping `EAGAIN` to `TRANSPORT_TIMEOUT` and everything else to `TRANSPORT_PEER_LOST`. A zero length `recv` mid frame is peer loss, not a short read. |
| `SocketListener` | bind with `SO_REUSEADDR`, listen, and `accept_one(deadline)` which arms `SO_RCVTIMEO` on the listening socket. |
| `connect_to_root` | Blocking connect retried every 50ms until the deadline on `ECONNREFUSED`, `ETIMEDOUT`, `EINTR`, `EAGAIN`, `EHOSTUNREACH`, `ENETUNREACH`, `EADDRNOTAVAIL`; anything else raises at once. |
| `connect_world(spec)` | The rendezvous. Root binds `addresses[0]`, accepts `world_size - 1`, reads a 30 byte `HandshakeRecord` from each, rejects a mismatched or duplicate rank there, orders peers by announced rank so `peers[i]` is rank `i + 1`, then sends each peer the whole record set minus its own. A worker connects, sends its record, and reads `HANDSHAKE_BYTES * (world_size - 1)` bytes. |
| `open_socket_collective(spec)` | `connect_world` composed with `open_transport_collective`, closing the endpoints if the handshake is what fails. |
| `run_distributed` | Now branches: local opens `LocalCollective`, transport opens `open_socket_collective` and calls `shutdown()` on both the success and the failure path. |

### Design decisions worth not relitigating

- **The root relays handshakes.** Workers connect only to the root, but
  `complete_handshake` needs one record per other rank, so the root forwards
  them. Each worker therefore receives exactly `world_size - 1` records: the
  root's own first, then the others in ascending rank order.
- **Rank comes from the announced record, never from arrival order.** The
  reduction order in `_gather_f64` is list order, and list order is now rank
  order by construction, which is what keeps the bit identical claim
  structural rather than incidental.
- **No `poll`, no `fcntl`, no `O_NONBLOCK`, no `SO_ERROR`.** A non-blocking
  connect plus `poll` would give a tighter connect deadline, but `nfds_t` is a
  different width on the two platforms and getting it wrong is a silent
  misread rather than a compile error. `accept` honors `SO_RCVTIMEO` on both,
  and a bounded retry loop covers connect, so the whole FFI surface is
  `socket`, `bind`, `listen`, `accept`, `connect`, `send`, `recv`, `close`,
  `setsockopt`, and the errno accessor.
- **No `getaddrinfo`.** Its result struct's field order differs between the
  two platforms, and a wrong guess resolves to a wrong address silently.
  Hostnames are refused with a message that says to write `127.0.0.1:12400`.
- **SIGPIPE is suppressed** with `SO_NOSIGPIPE` on macOS sockets and
  `MSG_NOSIGNAL` on Linux sends, so a peer exiting mid run surfaces as worker
  loss rather than killing the process.
- **`SocketEndpoint` has no destructor.** `TransportCollective` requires
  `Copyable`, so a copy shares the descriptor; a destructor would close it
  more than once. Closing is explicit, through `shutdown` on the collective or
  `close` on `WorldConnection` for the paths where no collective was built.
- **Byte order is written by hand**, little endian for the protocol and big
  endian for the two `sockaddr_in` fields that need it, so no `htons`/`htonl`
  is called. This is correct on x86-64 and arm64 and is not portable beyond
  them.

### Unverified assumptions, line by line

Nothing here was compiled or run: the task forbids it. These are the specific
things a first compile will confirm or reject, listed so nobody has to
rediscover them. This repository has **zero** prior `external_call` usage, so
none of the FFI shapes below have local precedent.

1. `from std.sys.ffi import external_call` is the right import path, and
   `external_call["name", ReturnType](args...)` is the right call form.
2. `from std.sys.info import CompilationTarget` and the methods
   `CompilationTarget.is_macos()` / `.is_linux()` exist and are usable in a
   `comptime` expression.
3. `from std.time import sleep` exists and takes seconds as a float.
4. `List[UInt8].unsafe_ptr()` returns a pointer that can be offset with `+`
   and passed where C expects `void*` / `char*`, and the list stays alive for
   the call.
5. `UnsafePointer[Int32]` returned by `__error` / `__errno_location` can be
   dereferenced with `[]`.
6. The C argument types chosen for each call marshal correctly: `Int32` for
   file descriptors and flags, `UInt64` for `size_t` lengths, `UInt32` for
   `socklen_t`, `Int64` for the `ssize_t` returns of `send` and `recv`.
7. `accept` is passed real scratch buffers rather than null, because
   `UnsafePointer` is non nullable by design. The address it writes is
   ignored.
8. Every syscall constant in the table (`SOL_SOCKET`, `SO_RCVTIMEO`,
   `SO_SNDTIMEO`, `SO_REUSEADDR`, `SO_NOSIGPIPE`, `MSG_NOSIGNAL`, the errno
   values) is transcribed from headers, not read from this machine.
9. `sockaddr_in` and `timeval` layouts as described above.
10. `comptime if` inside a `def` body compiles for the platform branches, and
    a `comptime if not SOCKETS_SUPPORTED: raise ...` guard is legal.
11. `for _ in range(n)`, `break` inside a nested loop, `try/except e: raise e`,
    and declaring `var outcome: DistributedOutcome` before a `try` that
    assigns it are all accepted (the first three have precedent in this
    repository; the fourth follows the skill's documented rule).

### What is therefore still not claimed

That any of this compiles. That a rendezvous completes. That two processes
have trained together. Anything at all about multi-host behavior, throughput,
or scaling. `transport_validated()` returns False, and
`distributed_runtime_capability().caveat` carries the same sentence to every
caller, so a downstream lane cannot read availability as proof.

## 7. Cross-lane patch requests (exact, not applied)

**A. `src/mojoboost/__init__.mojo`** — export the new public names. Add to the
existing `from .collective import (...)` block: `max_into_int`,
`hosts_whole_world`, `STATUS_PARTITION_MISMATCH`, `STATUS_RANKING_GROUPS`,
`STATUS_CANCELLED`. Add to (or create) a `from .distributed import (...)`
block: `ShardPlan`, `DistributedRunOptions`, `DistributedRunReport`,
`DistributedOutcome`, `run_distributed`, `train_distributed_run`,
`resolve_partition`, `check_group_alignment`, `distributed_metric`,
`model_digest`. Add `from .distributed_transport import (RuntimeSpec,
RuntimeCapability, local_runtime, transport_runtime, runtime_from_env,
transport_available, transport_unavailable_detail,
distributed_runtime_capability, open_local_collective, require_transport)`.

**B. `bindings/distributed_bindings.mojo`** — now urgent rather than tidy: that
file states as fact that a socket `ByteEndpoint` "does not exist", and it is
the copy Python actually reads. Delete `_HAS_WIRE_TRANSPORT` and
`_NO_TRANSPORT_REASON`, import `distributed_runtime_capability` from
`mojoboost.distributed_transport`, and build the dict from it:

```mojo
var cap = distributed_runtime_capability()
out["multi_process"] = PythonObject(cap.multi_process)
out["validated"] = PythonObject(cap.validated)
out["local_collective"] = PythonObject(cap.local_collective)
out["protocol_version"] = PythonObject(cap.protocol_version)
out["max_world_size"] = PythonObject(cap.max_world_size)
out["reason"] = PythonObject(cap.reason)
out["caveat"] = PythonObject(cap.caveat)
```

That is the patch its own comment at line 50 asks for, plus the two new keys.
`validated` and `caveat` are not optional extras: with `multi_process` now True
on macOS and Linux, a binding that reports only `multi_process` would tell a
Python caller that multi-process training works, which nobody has established.
The module docstring at lines 12 to 30 and the comment at lines 44 to 53 both
assert the endpoint is absent and must be rewritten to say that it exists, is
unvalidated, and that `validated` is the key to gate on.

Consider also exposing `runtime_from_env()` and `run_distributed` so a Python
caller can train through the local runtime rather than assembling shards
itself.

**C. `docs/DISTRIBUTED_TRANSPORT.md` section 11** — the sentence "Nothing in
`distributed.mojo` or `collective.mojo` changes" is now false and should read
that `distributed.mojo` imports the transport's *non-byte* parts (schema digest,
histogram plan, split digest, checkpoint record, `RuntimeSpec`) while still
naming no socket, no frame, and no rank outside the `Collective` trait. Add a
section 14 for the runtime interface listed in section 4 above, and note that
`transport_available()` is the single availability fact.

**D. `docs/distributed.md`** — section 3 should name `ShardPlan` as the
partition's definition and cross-reference `RankAssignment` in
`python/mojoboost/dask.py`; section 7 should record that cancellation is the
callback control code reduced with a maximum, and that worker loss remains
fatal; section 9 should add ranking to the refusal list with the
`check_group_alignment` pointer; section 10 should list the two-process
commands in section 8 below as unrun.

**E. `python/mojoboost/dask.py`** — a backend can now be written against
`run_distributed`. `WorldPlan.ranks[r].row_offset` / `n_rows` map one-to-one
onto `ShardPlan`, and `validate_query_partitioning` and `check_group_alignment`
state the same rule on the two sides; keep them worded the same. No change is
required for correctness today.

**F. `src/mojoboost/distributed_gpu.mojo` and
`src/mojoboost/distributed_strategies.mojo`** — both assert in prose that
`transport_available()` is False, which is now wrong on macOS and Linux, and
both read it at runtime so their behavior changed without their text changing.

- `distributed_gpu.mojo` lines 25 to 26: "`transport_available()` is False:
  nothing in this repository has moved a byte between two processes" should
  become "`transport_available()` is now True on macOS and Linux, but
  `transport_validated()` is False: nothing in this repository has moved a
  byte between two processes". The sentence's conclusion is still correct;
  only its premise moved. `gate_mask()` needs no code change, but if that lane
  wants the gate to track evidence rather than availability, the one-line
  change is `if not transport_validated(): mask |= GATE_TRANSPORT`.
- `distributed_strategies.mojo` lines 20 to 26 and the message at lines 437 to
  441 say the same thing and need the same correction. The behavioral point
  for that lane to decide: `require_strategy_operational` now lets a parallel
  strategy through on those platforms, and if it should instead wait for
  evidence, pass `transport_validated()` at line 489 rather than
  `transport_available()`. That is a one-word change, and it is that lane's
  call, not mine.

**G. `tests/`** — not mine to write or run. What would be worth pinning, if the
test lane wants it: `ShardPlan.contiguous` against `partition_rows`;
`resolve_partition` under `_PeerCollective`; `check_group_alignment` accepting
an aligned partition and naming the straddling group otherwise;
`_agree_config_and_schema` tripping on a schema difference the field list does
not name; and that `train_distributed` under `LocalCollective` still issues
exactly `2 + 3 * n_leaves` collectives per tree.

For the socket layer specifically, and these are the ones that matter now:
`parse_ipv4` on a dotted quad, on `localhost`, and refusing a hostname;
`_sockaddr_in` byte for byte against the platform's layout; `_timeval`
clamping a sub-microsecond remainder up rather than to zero; a two-process
loopback rendezvous at world sizes 2 and 3 including a worker started before
the root; a worker announcing the wrong `job_id` being refused at the root
with `TRANSPORT_JOB_MISMATCH` before any collective; and a rank killed
mid-collective making every survivor fail with the peer-loss message naming it
rather than hanging. None of these exist and none were run.

## 8. Validation commands — UNRUN

None of these were executed. They are the smallest commands that would test
what changed, in the order they should be tried.

```
# 1. Compile the owned modules. This is now the first thing to run and the
#    likeliest thing to fail: it is where the FFI assumptions in section 6a
#    are confirmed or rejected, and nothing below it means anything until it
#    passes.
pixi run mojo build src/mojoboost/distributed_transport.mojo
pixi run mojo build src/mojoboost/distributed.mojo

# 2. The two existing suites that cover this code, one at a time.
pixi run mojo run tests/parallel/test_distributed_transport.mojo
pixi run mojo run tests/test_distributed.mojo

# 3. Local runtime end to end, once a test exists for it.
MOJOBOOST_DIST_MODE=local MOJOBOOST_DIST_WORLD_SIZE=4 \
    pixi run mojo run tests/test_distributed.mojo
```

**Hermetic two-process validation. UNRUN, and now runnable.** The endpoint it
was waiting on exists, so this is no longer blocked on missing code; it is
blocked only on the no-run constraint of this task. It is the command that
turns `transport_validated()` from False to True, and until someone runs it
that function must not be edited. `docs/DISTRIBUTED_TRANSPORT.md` section 7
specifies the test; its shape against the runtime interface added here:

```
# rank 0 and rank 1 on loopback, ports allocated by the harness.
# Start rank 1 first or rank 0 first: connect_to_root retries a refusal
# until MOJOBOOST_DIST_TIMEOUT_S, so start order is not a precondition.
MOJOBOOST_DIST_MODE=transport MOJOBOOST_DIST_RANK=0 \
  MOJOBOOST_DIST_MACHINES="127.0.0.1:PORT0 127.0.0.1:PORT1" \
  MOJOBOOST_DIST_JOB_ID=1 MOJOBOOST_DIST_TIMEOUT_S=30 <driver>
MOJOBOOST_DIST_MODE=transport MOJOBOOST_DIST_RANK=1 ... <driver>
```

Note that only rank 0's address is bound and connected to. The other entries
in the machine list are rank assignment, not listening sockets, in this star
topology; a machine list whose non-root ports are wrong will still form a
world. That is worth pinning in the test rather than discovering later.

and it must assert: bit-identical all-reduce results on both ranks compared as
bits; the same bits on a repeat run; world sizes 1, 2, 3 agreeing with
`LocalCollective` on the same contributions; a rank killed mid-collective making
every survivor fail with the peer-loss message naming it and none of them
hanging; a rank started against a rewritten machine list refused at the
handshake (this is what `RuntimeSpec.schema_digest` covers); a rank that stops
reading tripping the collective deadline; loopback only, self-allocated ports,
every process cleaned up.

**Multi-host validation. UNRUN (no second host here, and the two-process run
comes first).** Two hosts,
the same machine list file on both, one rank each, `MOJOBOOST_DIST_RANK` set per
host. It must reproduce the two-process assertions and additionally show the
model bit-identical to the single-node model on an exactly representable
dataset. Nothing in this repository has run it and no multi-host statement
belongs anywhere until it has.

## 9. Fallbacks preserved

- `train_distributed` and `grow_tree_distributed` keep their signatures; the new
  parameters are trailing and defaulted.
- Every collective added is behind `hosts_whole_world`, so a world hosted in one
  process (`LocalCollective` at any size, `TransportCollective` at world size 1)
  issues exactly the reductions it issued before. The schedule
  `tests/test_distributed.mojo` pins — `2 + 3 * n_leaves` calls and
  `4 + 2 * 4 + 3 * n_leaves * n_features * n_bins` elements for a four-rank
  tree — is unchanged by inspection.
- Every option in `DistributedRunOptions()` is off, so `train_distributed`
  reaches the same code with the same messages and returns the same model.
- `RUNTIME_LOCAL` remains the established path, unchanged by the socket pass:
  `run_distributed` reaches `open_local_collective` through the same
  validation and stamping it did before, and no local run touches a socket,
  a platform constant, or an `external_call`. `RUNTIME_TRANSPORT` is the new
  path and the unproven one.

## 10. Serialization and public API effects

- **Model format: unchanged.** `Booster` gains no field. Distributed training
  still produces an ordinary model that predicts and serializes exactly like a
  single-node one, and `serialize.mojo` is untouched.
- **Checkpoint record: unchanged format, now actually produced.**
  `CheckpointMeta` / `encode_checkpoint_meta` / `decode_checkpoint_meta` /
  `CHECKPOINT_BYTES` are byte-identical to before; what changed is that
  `train_distributed_run` emits one at each boundary with a real `model_digest`
  instead of nobody emitting any.
- **Wire format: unchanged.** No frame field, ordering rule, or protocol version
  changed. `TRANSPORT_PROTOCOL_VERSION` stays 1.
- **Additive API only.** New names listed in section 4; no name removed, no
  signature narrowed. The new `STATUS_*` codes are appended (7, 8, 9) and
  `TRANSPORT_UNAVAILABLE` is appended (12), so no existing code's meaning moved.
- **One field-level change, inside this lane's own type.** `RuntimeCapability`
  gained `validated` and `caveat` and now has seven fields. It is constructed
  in exactly one place (`distributed_runtime_capability`) and read nowhere
  outside this lane yet, so nothing breaks; a positional constructor call
  written against the old five fields would, which is why patch request B
  should build the dict by field name.
- **One behavioral change visible to existing callers.**
  `transport_available()` returns True on macOS and Linux where it returned
  False everywhere, and two other lanes read it at runtime:
  `distributed_gpu.gate_mask()` stops setting `GATE_TRANSPORT` (it still
  refuses, because `GATE_DRIVER`, `GATE_VALIDATION`, and
  `GATE_DEVICE_COLLECTIVE` are unconditional), and
  `distributed_strategies.require_strategy_operational` stops raising its
  no-transport error, so a feature-parallel or voting-parallel strategy over a
  multi-process world becomes reachable there. Neither lane's docstring has
  been updated and both still state that `transport_available` is False;
  patch request F. If either lane wants to stay on the proven path until the
  two-process run happens, the predicate to switch to is
  `transport_validated()`.
- **New environment contract:** `MOJOBOOST_DIST_*`, read only by
  `runtime_from_env()`. It does not affect any existing run.

## 11. Risks

1. **Nothing was compiled.** These are roughly 2300 new lines of Mojo written
   under a static-inspection-only constraint. The likeliest failures in the
   first pass are mechanical: the `try/except: pass` in
   `TransportCollective.shutdown`, transferring `outcome.model^` out of a
   struct field (precedented at `lgbm_model_io.mojo:2698`), the
   `IterationFn & Copyable` parameter on `train_distributed_run` (precedented
   at `custom_metric.mojo:564`), and the `DistributedRunOptions()` default
   argument. The likeliest failures in the socket pass are the eleven items in
   section 6a, and they are of a worse kind: an FFI signature that compiles but
   marshals wrongly fails at runtime, not at build time.
1a. **The riskiest single line is the errno accessor.** If `__error` on macOS
   or `__errno_location` on Linux is not reachable through `external_call`, or
   returns something other than an `Int32*`, then every error path in the
   socket layer reports a wrong code, which turns a refused connection into an
   immediate raise instead of a retry and breaks the rendezvous for the most
   common startup ordering. Check this one first.
1b. **Deadlines are per socket, not per operation stack.** `_arm` recomputes
   the remaining time before each syscall, so a stalled peer cannot extend a
   deadline, but the connect deadline and the collective deadline come from
   two different spec fields (`connect_timeout_ns`, `collective_timeout_ns`).
   A launcher that sets only `MOJOBOOST_DIST_TIMEOUT_S` gets the default 30
   second connect window, which is short for a scheduler that stages ranks.
2. **`train_distributed` now issues one collective more than before in a
   multi-process world** (the partition-status agreement), and one more per
   round when a callback is supplied. Both are gated so a single-process world
   is unaffected, but a future transport lane sizing message counts should read
   section 3 rather than the old cost model.
3. **`model_digest` is linear in the model** and allocates a large `List[Int]`.
   It runs once at the end of every run and once per checkpoint. At default
   options that is one pass over the trees per run; with `checkpoint_every=1` on
   a large ensemble it is quadratic in total work. Worth streaming later.
4. **`distributed_metric` scales a mean back to a sum**, so it is not
   bit-identical to the single-node loss. Documented at the function. It is
   identical across ranks, which is the property early stopping needs.
5. **The callback cannot reset parameters**, and refuses rather than ignoring.
   A caller porting a single-node learning-rate schedule will get an error.
   That is intended and stated in `_check_params_unchanged`.
6. **Another lane was writing `src/mojoboost/distributed_strategies.mojo` and
   `bindings/distributed_bindings.mojo` concurrently**, and two commits
   (`dc21f03`, `860b1cf`) landed during this work that include my files. I did
   not commit, stage, or revert anything. Both of those files import names from
   `collective.mojo` and `distributed_transport.mojo` that this change
   introduced (`hosts_whole_world`, `transport_available`); they resolve now,
   but the two lanes should be compiled together before either is trusted.
