# Connect 13: real distributed transport to distributed training

Owned files, and the only files edited:

- `src/mojoboost/collective.mojo` (240 -> 290 lines)
- `src/mojoboost/distributed_transport.mojo` (1844 -> 2359 lines)
- `src/mojoboost/distributed.mojo` (930 -> 1803 lines)
- `handoffs/connect_13_distributed.md` (this file)

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
| `distributed_transport.mojo` | frames, session state machine, handshake, ordered reduction, `TransportCollective`, histogram cost plan, split digest, checkpoint record | fully written, entirely unreferenced by the trainer, and unopenable (no `ByteEndpoint` that reaches a process) |
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
       -> require_transport(spec)            # refuses a multi-process spec now
       -> open_local_collective(spec)        # the only mode that opens
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

13. **Explicit unavailability.** `HAS_BYTE_ENDPOINT = False`,
    `transport_available()`, `transport_unavailable_detail()`,
    `TRANSPORT_UNAVAILABLE = 12`, `require_transport()`, and
    `distributed_runtime_capability()` are the single place the build answers
    "can two processes train together". `run_distributed` refuses a transport
    spec before a row is partitioned.

## 4. The one narrow runtime interface

For bindings, Dask, and any launcher:

```mojo
# distributed_transport.mojo
RUNTIME_LOCAL, RUNTIME_TRANSPORT, HAS_BYTE_ENDPOINT
RuntimeSpec{mode, rank, world_size, addresses, job_id,
            connect_timeout_ns, collective_timeout_ns, restart_epoch}
    .validate() .transport_config() .schema_digest() .describe()
local_runtime(world_size, job_id) -> RuntimeSpec
transport_runtime(machines, rank, job_id, ...) -> RuntimeSpec
runtime_from_env() -> RuntimeSpec
runtime_mode_name / parse_runtime_mode
transport_available() -> Bool
transport_unavailable_detail() -> String
distributed_runtime_capability() -> RuntimeCapability
require_transport(spec)
open_local_collective(spec) -> LocalCollective
open_transport_collective[E](spec, peers, peer_handshakes) -> TransportCollective[E]
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

1. **No socket `ByteEndpoint` exists, so nothing multi-process runs.** No byte
   has moved between two processes. No multi-host capability, throughput, or
   scaling is claimed. `MemoryEndpoint` is a fake and is named as one.
   `run_distributed` refuses `RUNTIME_TRANSPORT`; `open_transport_collective`
   refuses too, so it is currently unreachable even from a test (a test that
   wants the driver constructs `TransportCollective` directly, as the existing
   suite does).
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

**B. `bindings/distributed_bindings.mojo`** — delete `_HAS_WIRE_TRANSPORT` and
`_NO_TRANSPORT_REASON`, import `distributed_runtime_capability` from
`mojoboost.distributed_transport`, and build the dict from it:

```mojo
var cap = distributed_runtime_capability()
out["multi_process"] = PythonObject(cap.multi_process)
out["local_collective"] = PythonObject(cap.local_collective)
out["protocol_version"] = PythonObject(cap.protocol_version)
out["max_world_size"] = PythonObject(cap.max_world_size)
out["reason"] = PythonObject(cap.reason)
```

That is the patch its own comment at line 50 asks for. Consider also exposing
`runtime_from_env()` and `run_distributed` so a Python caller can train through
the local runtime rather than assembling shards itself.

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

**F. `tests/`** — not mine to write or run. What would be worth pinning, if the
test lane wants it: `ShardPlan.contiguous` against `partition_rows`;
`resolve_partition` under `_PeerCollective`; `check_group_alignment` accepting
an aligned partition and naming the straddling group otherwise;
`_agree_config_and_schema` tripping on a schema difference the field list does
not name; `run_distributed` refusing a `transport_runtime` spec with
`TRANSPORT_UNAVAILABLE`; and that `train_distributed` under `LocalCollective`
still issues exactly `2 + 3 * n_leaves` collectives per tree.

## 8. Validation commands — UNRUN

None of these were executed. They are the smallest commands that would test
what changed, in the order they should be tried.

```
# 1. Compile the three owned modules (fastest failure surface for the edits).
pixi run mojo build src/mojoboost/distributed_transport.mojo

# 2. The two existing suites that cover this code, one at a time.
pixi run mojo run tests/parallel/test_distributed_transport.mojo
pixi run mojo run tests/test_distributed.mojo

# 3. Local runtime end to end, once a test exists for it.
MOJOBOOST_DIST_MODE=local MOJOBOOST_DIST_WORLD_SIZE=4 \
    pixi run mojo run tests/test_distributed.mojo
```

**Hermetic two-process validation (cannot run: there is no socket endpoint).**
When one exists, `docs/DISTRIBUTED_TRANSPORT.md` section 7 specifies the test.
Its shape against the runtime interface added here:

```
# rank 0 and rank 1 on loopback, ports allocated by the harness
MOJOBOOST_DIST_MODE=transport MOJOBOOST_DIST_RANK=0 \
  MOJOBOOST_DIST_MACHINES="127.0.0.1:PORT0 127.0.0.1:PORT1" \
  MOJOBOOST_DIST_JOB_ID=1 MOJOBOOST_DIST_TIMEOUT_S=30 <driver>
MOJOBOOST_DIST_MODE=transport MOJOBOOST_DIST_RANK=1 ... <driver>
```

and it must assert: bit-identical all-reduce results on both ranks compared as
bits; the same bits on a repeat run; world sizes 1, 2, 3 agreeing with
`LocalCollective` on the same contributions; a rank killed mid-collective making
every survivor fail with the peer-loss message naming it and none of them
hanging; a rank started against a rewritten machine list refused at the
handshake (this is what `RuntimeSpec.schema_digest` covers); a rank that stops
reading tripping the collective deadline; loopback only, self-allocated ports,
every process cleaned up.

**Multi-host validation (cannot run: same reason, plus no hardware).** Two hosts,
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
- `RUNTIME_LOCAL` remains the established path; `RUNTIME_TRANSPORT` is a
  configuration that validates and an open that refuses.

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
- **New environment contract:** `MOJOBOOST_DIST_*`, read only by
  `runtime_from_env()`. It does not affect any existing run.

## 11. Risks

1. **Nothing was compiled.** These are ~1440 new lines of Mojo written under a
   static-inspection-only constraint. The likeliest failures are mechanical:
   the `try/except: pass` in `TransportCollective.shutdown`, transferring
   `outcome.model^` out of a struct field (precedented at
   `lgbm_model_io.mojo:2698`), the `IterationFn & Copyable` parameter on
   `train_distributed_run` (precedented at `custom_metric.mojo:564`), and the
   `DistributedRunOptions()` default argument.
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
