# Connect 15 handoff: Dask on the real distributed backend

Lane: connect the Dask estimators to the native distributed runtime.

## Status, stated plainly

`mojoboost.dask` now has one backend rather than none:
`python/mojoboost/_dask_runtime.py` probes the native distributed runtime,
declares what it can do, launches one rank per worker with the partitions
left where dask put them, and returns rank 0's model bytes. `get_backend()`
falls back to it with no registration step.

**On every build today that fallback refuses**, because the native runtime
is not there to launch: `src/mojoboost/distributed_transport.mojo` has no
socket `ByteEndpoint`, and `bindings/distributed_bindings.mojo` says so in
its own words (`distributed_capability()["multi_process"]` is `False`).
`DaskMojoBoostRegressor(...).fit(...)` therefore raises
`DistributedNotAvailable` before it touches the cluster, quoting the
runtime's reason.

Nothing was run. No test, build, formatter, or cluster. No Dask collection,
scheduler, or worker has executed a line of this. The launch path in
`_dask_runtime` is a proposal in exactly the way `DaskRuntime` already was,
and should be read as one.

Nothing was committed or staged by this lane. See "Shared worktree" at the
end: another lane's whole-tree commit swept `_dask_runtime.py` in while it
was being written.

## Files this lane owns

| path | state |
| --- | --- |
| `python/mojoboost/_dask_runtime.py` | new, ~1600 lines |
| `python/mojoboost/dask.py` | edited |
| `handoffs/connect_15_dask.md` | this file |

No other file was touched. Every change another file needs is a patch
request below.

## Implementations found before writing anything

| where | what it is | verdict |
| --- | --- | --- |
| `python/mojoboost/dask.py` | client-side contract: partition metadata, schema merge, rank plan, backend registry, model ownership, partition-local prediction | authoritative, kept, extended |
| `src/mojoboost/distributed.mojo` | `train_distributed[C: Collective]`, the data-parallel growth loop | authoritative trainer, unreachable from Python |
| `src/mojoboost/collective.mojo` | `Collective` trait, `LocalCollective` (one process) | real, single-process only |
| `src/mojoboost/distributed_transport.mojo` | frames, session state machine, handshake, deadlines, cancellation, worker loss, checkpoint, `TransportCollective`, `MemoryEndpoint` | real in process, no socket, so no cluster |
| `bindings/distributed_bindings.mojo` (Task 14, untracked when read) | `distributed_capability`, `distributed_check_machine_list`, `distributed_status_message`, `transport_status_message` | the capability half of the contract, fused with |
| `bindings/_mojoboost.mojo` | exports `fit*`, `predict*`, `save`/`load`, introspection | no distributed export registered yet |
| `handoffs/task17_dask.md` | the original list of what the transport owes | still accurate, superseded on names |

There was no second Dask adapter, no second backend registry, and no
Python-side trainer to fuse or quarantine. The one duplicate risk this lane
could have created, a Python fit-per-rank-and-average, was not created and
is refused explicitly (see "What this lane refused to do").

## Call path, before and after

**Before**

```
DaskMojoBoost*.fit
  -> get_backend()            -> DistributedNotAvailable (always)
```

**After**

```
DaskMojoBoost*.fit(X, y, ..., eval_set=..., early_stopping_rounds=...)
  -> _validation_arguments()               normalize / refuse orphaned args
  -> get_backend()
       registered backend, else MOJOBOOST_DASK_BACKEND, else
       _dask_runtime.native_runtime_status()
         -> provider.distributed_runtime_info() if present
         -> provider.distributed_capability() otherwise   [exists today]
         -> unavailable, with the runtime's own reason     [today's answer]
  -> runtime.partitions(...) -> validate_partitions -> plan_world
  -> _plan_validation(...)                  one WorldPlan per eval set
  -> require_capabilities(backend, needed)  adds "early_stopping"
  -> TrainingJob(plan, objective, params, label_classes, ranking,
                 validation, eval_metrics, early_stopping_rounds,
                 first_metric_only, timeout)
  -> backend.bind_runtime(runtime)          optional hook
  -> backend.train(job)
       NativeDistributedBackend.train
         _transport_addresses(plan)         host:port per rank
         _check_thread_capacity(client, plan)
         client.submit(_train_rank, ..., workers=[rank.worker])  per rank
           [worker] _concat_blocks -> _arrays.check_X / check_target
           [worker] provider.distributed_worker_open(config)
           [worker] provider.distributed_worker_train(handle, X_addr, ...)
           [worker] provider.save / save_multiclass -> model bytes
           [worker] provider.distributed_worker_close(handle)
       -> NativeModelRef(futures)
  -> take_model_bytes(ref, timeout)         waits on every rank
  -> _install_model(blob, plan, label_classes)
  -> _install_metrics(ref.metrics(), ...)   evals_result_, best_score_, ...
```

Prediction is unchanged and was already connected: `predict` /
`predict_proba` on a Dask collection ship the model bytes and run
`_PartitionPredictor` under `runtime.map_partitions`, parsed once per worker
process. It needs no cluster state and no transport, so it works today.

## Connections completed

1. **One backend, reached without registration.** `get_backend()` resolves
   the native runtime after the registry and the environment variable. A
   registered backend still wins, which is what keeps the test fake and any
   third-party transport working.
2. **Precise unavailability.** `DistributedNotAvailable` now carries the
   runtime's own reason: which entry points are missing, or
   `distributed_capability()`'s `reason` string, or a protocol version
   mismatch, or an unknown capability name. `mojoboost.dask.distributed_status()`
   returns the whole record for a diagnostics report.
3. **Partitions stay put.** Rank tasks are submitted with
   `workers=[rank.worker]`, `allow_other_workers=False`, and the partition
   blocks passed as `distributed.Future` objects inside plain lists and
   dicts, which is what dask walks. The client never gathers X, y, or the
   weights, and never fits.
4. **Categorical metadata reaches the workers.** `WorldPlan.schema` travels
   as `(index, categories)` pairs and is applied through the existing
   `_arrays.check_X(encoders=...)` path, so a rank encodes a pandas frame
   through the fit's tables rather than its own partition's.
5. **Class labels are global.** Worker-side label encoding goes through
   `fit(classes=...)` explicitly (`_encode_labels_through`), never through
   `_arrays.encode_labels`, which derives classes from the rows it is
   handed and would give two ranks two different encodings. A label outside
   the list fails, naming the rank.
6. **Ranking groups reach the workers.** A rank's `group` is the
   concatenation of its own partitions' groups in row order, for training
   and for each eval set, and the client-side rule that a query may not
   straddle a partition boundary is now applied to eval sets too.
7. **Validation and early stopping.** `eval_set`, `eval_names`,
   `eval_sample_weight`, `eval_metric`, `early_stopping_rounds`, and
   `first_metric_only` are accepted, planned onto the training rank layout,
   required to match it worker for worker, carried in `TrainingJob`, and
   passed to the native call. They require the `early_stopping` capability,
   so a backend that cannot reduce a validation loss refuses by name.
8. **Model assembly is unchanged and shared.** Rank 0 serializes with the
   existing `save` / `save_multiclass`; the client rebuilds with the
   existing `load` / `load_multiclass` through `_install_model`. No second
   model format and no second parse path entered the codebase.
9. **Validation history becomes fitted state.** `ref.metrics()` is read
   into `evals_result_`, `best_score_`, `best_iteration_`, `stopped_early_`,
   and `n_iter_`, laid out exactly the way `fit_with_metrics` lays them out
   for a single-machine fit.
10. **Worker-local runtime lifecycle.** A rank registers its session handle
    in a process-local table while it is open and removes it in a
    `finally`, and closes the session on both paths.
11. **Cancellation.** `KeyboardInterrupt` on the client cancels the rank
    futures and calls `client.run(_cancel_sessions, job_id)` so a rank
    already blocked inside a collective is woken through the runtime's own
    cancellation path instead of waiting out a deadline.
12. **Worker failure.** `NativeModelRef.result` waits on every rank, not
    just the root: the first rank to fail cancels the world and raises
    `DistributedRankError` naming the rank, the worker, and the cause.
    `DistributedTimeout` is the same for a fit that outran `timeout`.
13. **A deadlock refused in advance.** Two ranks on a one-thread worker
    cannot both sit in a collective, so `_check_thread_capacity` refuses
    that layout with the fix in the message rather than letting the cluster
    hang at zero CPU.

## What this lane refused to do

- No Python-side training, of any kind. No fit-per-rank-and-average, no
  fit-on-the-client-after-gathering, no "distributed" that is one rank.
- `distributed_capability()["local_collective"]` being true is **not** read
  as availability. One process is not a cluster.
- No retry loop anywhere. A failed rank fails the fit.

## The adapter protocol, and the exact patch requests

`_dask_runtime` resolves a *provider*: the extension module by default,
overridable with `register_runtime_provider(obj)` or
`MOJOBOOST_DISTRIBUTED_PROVIDER=package.module:attribute`. That is what
keeps this lane parallel with Task 13: a different set of names costs one
adapter object, not an edit here.

### Request to Task 14 (bindings), and through it Task 06

`bindings/distributed_bindings.mojo` already exports what the capability
half needs. Two follow-ups:

1. **Register the existing capability function** so Python can see it. In
   `bindings/_mojoboost.mojo` (Task 06 owns the file):

   ```mojo
   from distributed_bindings import (
       distributed_capability,
       distributed_check_machine_list,
       distributed_status_message,
       transport_status_message,
   )
   ...
   m.def_function[distributed_capability]("distributed_capability")
   m.def_function[distributed_check_machine_list](
       "distributed_check_machine_list"
   )
   m.def_function[distributed_status_message]("distributed_status_message")
   m.def_function[transport_status_message]("transport_status_message")
   ```

   Until that lands, `hasattr(_mojoboost, "distributed_capability")` is
   false and `mojoboost.dask` reports "exports no distributed runtime at
   all", which is accurate but less specific than the reason the binding
   already writes.

2. **Add the worker half**, once Task 13 has a transport. Four functions,
   named and shaped as follows. Every argument is a Python scalar, string,
   list, or dict; no pointer to a device buffer or a session object crosses
   the boundary, only opaque integer handles.

   ```
   distributed_worker_open(config: dict) -> Int          # session handle
       config keys, all present, all validated at the boundary:
         protocol_version      Int, must equal 1 or raise
         job_id                Int, 63-bit, identical on every rank
         rank                  Int, 0 <= rank < world_size
         world_size            Int, >= 2
         addresses             List[String], "host:port", indexed by rank
         connect_timeout_ns    Int
         collective_timeout_ns Int
         n_features            Int
         row_offset            Int   this rank's offset in the global order
         n_rows                Int
         objective             String
       Opens this rank's listener, connects the world, runs the handshake,
       and raises with the transport's own message on
       TRANSPORT_WORLD_MISMATCH / SCHEMA_MISMATCH / TIMEOUT.

   distributed_worker_train(handle, X_addr, n_rows, n_features, y_addr,
                            objective, params) -> record
       Same argument shape as `fit`, plus the handle. `params` is the
       estimator's LightGBM-spelled dict, so the existing parameter parsing
       is reused and no parameter policy moves into Python. Keys this
       module adds to it:
         weight_addr        Int, 0 when there are no weights
         categorical_feature List[Int]
         n_classes          Int, 0 for regression and ranking
         group / n_groups   ranking only, this rank's query sizes
         valid_sets         List[dict], each with X_addr, y_addr,
                            weight_addr, n_rows, name, and group/n_groups
                            for a ranker
         eval_names         List[String]
         metric             List[String], may be empty (use the objective's
                            default and report the resolved names)
         early_stopping_rounds Int
         first_metric_only  Int, 0 or 1
       Returns the same six-element record `fit_with_metrics` returns:
         (model handle or 0, flat metric values, n_rounds, best_iteration,
          best_score, stopped_early)
       Flat values are indexed (round * n_valid + valid) * n_metrics +
       metric, which is what the client unpacks. A non-root rank returns 0
       as the model handle.

   distributed_worker_close(handle) -> None    # idempotent
   distributed_worker_cancel(handle) -> None   # callable from another
                                               # thread while train blocks
   ```

   Optionally, `distributed_runtime_info() -> dict` with `available`,
   `reason`, `protocol_version` (1), `capabilities` (names drawn from
   `mojoboost.dask.CAPABILITIES`), `transport`, `multi_host`. It is read in
   preference to `distributed_capability()` when present, and is the only
   way for the runtime to declare which objectives it can actually train
   across ranks. Without it the fallback reads `capabilities` off the
   capability record, and a build reporting `multi_process` true with no
   capability list is refused as unusable.

### Request to Task 13 (native runtime)

- The worker entry points above need a runtime to stand on: a real socket
  `ByteEndpoint`, `TransportCollective` driving `train_distributed`, and a
  model handle at the root. Until the endpoint exists, keep
  `_HAS_WIRE_TRANSPORT` false rather than reporting a world this lane would
  then try to open.
- The capability record must state which of `regression`, `binary`,
  `multiclass`, `ranking`, `categorical`, `weights`, `quantile_l1`,
  `bagging`, `goss`, `class_weight`, `feature_fraction`, `early_stopping`,
  `custom_objective`, `gpu` genuinely work across ranks. Anything absent is
  refused by name on the client, which is the intended behavior; declaring
  a name that does not work is the failure mode to avoid.
- `distributed.mojo`'s `_UNSUPPORTED_*` mask and the capability list must
  not disagree. A parameter the growth loop refuses must not appear as a
  capability.
- Ports: this module assigns rank `r` the port `base + r`, base from
  `MOJOBOOST_DISTRIBUTED_BASE_PORT` (default 24601), host taken from the
  dask worker address. If the runtime would rather be told a port list,
  say so and this module will pass one instead.

### Request to Task 07 (public exports), exact

`python/mojoboost/__init__.py` must keep `mojoboost.dask` lazy. Add a
module-level `__getattr__` (PEP 562) at the **end** of the module body, and
nothing else:

```python
#: Submodules the package answers for without importing them. `mojoboost.dask`
#: subclasses the estimators defined above, so it cannot be imported from the
#: top of this file; and importing dask itself is not something `import
#: mojoboost` may cost. A module-level __getattr__ resolves it on first
#: attribute access instead. Importing mojoboost.dask still does not import
#: dask.
_LAZY_SUBMODULES = ("dask",)


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY_SUBMODULES))
```

Rules that come with it:

- Do **not** add `dask` to `__all__`, and do not re-export
  `DaskMojoBoostRegressor` or any other name from `mojoboost.dask` at the
  top level. They are reached as `mojoboost.dask.X`, the way
  `lightgbm.dask` is.
- Do **not** import `mojoboost._dask_runtime` from `__init__.py`. It is
  imported inside the two functions that need it and importing it eagerly
  would import `mojoboost.dask` eagerly, which is the ordering constraint
  this whole arrangement exists to respect.
- `mojoboost.diagnostics` may usefully report one line; the safe call is
  `mojoboost.dask.distributed_status()` (never raises, never imports dask)
  or `mojoboost._dask_runtime.describe_runtime()`. That is a suggestion,
  not a requirement, and it belongs to whoever owns `diagnostics.py`.
- `python/mojoboost/_public_api_plan.py` already describes this exact
  arrangement (`_LAZY_SUBMODULES = ("dask",)`); this request is that plan,
  spelled as the patch.

### Request to packaging (whoever owns `python/pyproject.toml`)

`[project.optional-dependencies]` wants `dask = ["dask[distributed]"]`.
Nothing else: `dask.py` and `_dask_runtime.py` are inside the `mojoboost`
package and ship already.

## Duplicates fused or quarantined

- **Fused:** the capability probe reads `distributed_capability()`, Task
  14's real binding, rather than inventing a parallel capability record.
  Only the fields that record has no room for come from the optional
  `distributed_runtime_info()`.
- **Fused:** the validation history uses `fit_with_metrics`'s existing
  six-element record shape and `evals_result_`'s existing layout, so there
  is one convention for a metric history rather than two.
- **Fused:** rank 0's model is serialized with the existing `save` and read
  with the existing `load`. No distributed model format exists.
- **Fused:** worker-side data conversion goes through `_arrays.check_X`,
  `check_target`, and `check_sample_weight`, the same boundary a
  single-machine fit uses.
- **Not duplicated:** parameter policy. `job.params` travels to the native
  call in its LightGBM spellings and is parsed by the same Mojo code the
  single-machine entry points use. Python adds only buffer addresses and
  the things with no LightGBM spelling.
- **Nothing quarantined.** No dead glue was found in the owned files.

## Fallbacks preserved

- The backend registry is unchanged and still wins over the native runtime.
  `register_backend`, `clear_backend`, `MOJOBOOST_DASK_BACKEND`, and
  `BytesModelRef` behave exactly as they did.
- `backend.bind_runtime`, `backend.cancel`, and `ref.metrics` are optional
  and reached through `getattr`, so a backend written against version 0 is
  unaffected.
- `BACKEND_PROTOCOL_VERSION` is still 0. The new `TrainingJob` fields are
  empty or zero for a fit with no eval set, which is what such a backend
  sees, and `_check_protocol` refuses a job declaring a version this
  backend does not know.
- Partition-local prediction is untouched and needs no runtime.
- Every rule the adapter enforced before is still enforced, in the same
  place, with the same messages.

## Serialization and public API effects

- **Model bytes:** unchanged format, unchanged reader. A distributed fit
  pickles, saves, and predicts exactly like a local one.
- **Estimator pickling:** unchanged. `__getstate__` still drops the client,
  the runtime, the plan, and the bytes cache. The new fitted attributes
  (`evals_result_`, `best_score_`, `stopped_early_`, `n_iter_`) are the
  single-machine ones, are already in `_Base._FITTED_ATTRS`, and pickle as
  ordinary Python state.
- **New public names in `mojoboost.dask.__all__`:**
  `DistributedRankError`, `DistributedTimeout`, `ValidationPlan`,
  `distributed_status`. `DistributedTimeout` subclasses
  `DistributedRankError`, which subclasses `RuntimeError`.
- **New `TrainingJob` fields:** `validation`, `eval_metrics`,
  `early_stopping_rounds`, `first_metric_only`, all with defaults, all
  before `protocol_version` in the field order (so positional construction
  of `protocol_version` would move; nothing in the tree constructs it
  positionally).
- **Changed `fit` signatures** on all three Dask estimators: the six (seven
  for the ranker) validation keywords are inserted between `sample_weight`
  and `runtime`. They are keyword arguments in practice everywhere in the
  tree and the tests, but a caller passing `runtime` positionally would
  break. No such caller exists in the repository.
- **Changed behavior:** `fit(early_stopping_rounds=...)` without an
  `eval_set` still raises `TypeError` naming the argument, as it did
  before. Note the asymmetry with the single-machine estimators, which
  raise `ValueError` for the same pair; the distributed behavior was
  preserved deliberately rather than aligned, because changing it would be
  a silent API break for the sake of tidiness.
- **`import mojoboost.dask` still imports neither dask nor distributed nor
  numpy.** `_dask_runtime` is imported inside `get_backend` and
  `distributed_status` only, and it imports `distributed` only inside
  `_import_distributed`.

## Remaining disconnections

1. **The native runtime does not exist.** Every launch path is unexercised
   and unreachable. This is the whole of what stands between here and a
   distributed fit.
2. **`evals_result_` needs metric names.** When `eval_metric` names none
   and the backend reports none, `best_score_`, `best_iteration_`, and
   `stopped_early_` are installed and the per-round history is not, because
   the client cannot lay out a table whose columns it cannot name. The fix
   is for `distributed_worker_train` to report the resolved metric names;
   the client already prefers them over what it asked for.
3. **Custom objectives and custom metrics.** A Python callable cannot be
   called from inside a collective on every rank without a design nobody
   has written. `custom_objective` is a capability a backend may declare,
   and a custom `eval_metric` is refused with a message.
4. **GOSS, bagging, and feature fraction** are capability names only. The
   sampling has to be a decision over the global row set, and no rank can
   make it alone; the refusal list belongs to the runtime.
5. **`class_weight="balanced"`** is still refused: it needs a global class
   count the client does not have. A dict works.
6. **Sparse collections.** `_arrays.check_X` is the dense boundary. A dask
   collection of sparse blocks would need the CSC path and a rank-side
   equivalent, and is not handled.
7. **`DaskRuntime` itself** has still never run: worker lookup, partition
   lengths, category extraction, and `map_partitions`'s `meta` are the
   first four things a live cluster will change.
8. **Checkpoint and restart.** `distributed_transport.mojo` has the
   restart record; nothing here asks for a resume, and a lost rank fails
   the fit rather than restarting it.

## Risks

- **The launch path is unverified.** `distributed.Future(key, client=...)`,
  `client.submit(workers=[...], allow_other_workers=False)`,
  `client.run(fn, arg, workers=[...])`'s argument order, and dask's
  traversal of futures inside dicts are all read from the documentation.
  The traversal one is load bearing: the blocks are passed as plain lists
  and dicts precisely because dask does not walk into custom objects, and
  a future that reached a worker unresolved would deadlock the rank task.
- **Port assignment is a guess.** `base + rank` collides with anything else
  on those ports, and a cluster with a restrictive firewall needs the
  environment variable set on every worker and the client.
- **The timeout split is a policy choice**, stated in the module: connect
  gets `min(timeout, 60s)`, a collective gets the whole `timeout`.
- **`_check_thread_capacity` reads `scheduler_info()`** and is skipped
  silently when the scheduler reports something unexpected, so it reduces
  the chance of a deadlock rather than eliminating it.
- **A rank task holds a worker thread for the length of the fit.** That is
  inherent to a blocking collective and is worth saying out loud before
  someone runs this on a shared cluster.
- **Shared worktree:** the concurrent lanes editing `bindings/`,
  `src/mojoboost/`, and `python/mojoboost/__init__.py` were not read
  line by line at the moment of writing, only where they touch this
  contract. `bindings/distributed_bindings.mojo` was untracked and could
  still change.

## Smallest later checks, all UNRUN

None of these were run, and none of them should be run by a lane that
cannot also fix what they break.

```
# Import safety and the refusal message, without a cluster:
pixi run python -c "import mojoboost.dask as d; print(d.distributed_status())"
pixi run python -c "import mojoboost.dask as d; d.DaskMojoBoostRegressor().fit(None, None)"

# The existing contract suite, which is what must not regress:
pixi run pytest python/tests/parallel/test_dask_contract.py -q

# Only once a native runtime exists, a two-worker local cluster:
#   dask scheduler + 2 workers, 1 thread each, one_rank_per_worker=True
#   fit a 2-partition dask array and compare against a single-machine fit
```

The contract suite is the one that matters first: it is 71 tests against a
fake backend and a fake runtime, and everything this lane changed on the
client side is covered by it. Three behaviors it pins were preserved on
purpose and are the things to check if it fails:

- `get_backend()` with nothing registered raises `DistributedNotAvailable`
  whose message contains "not finished". The native probe's
  no-entry-points reason carries that phrase.
- `fit(..., early_stopping_rounds=5)` with no eval set raises `TypeError`
  naming `early_stopping_rounds`.
- `TrainingJob.protocol_version == BACKEND_PROTOCOL_VERSION == 0`.

## Shared worktree

This lane staged, committed, and pushed nothing. Both of its code files
are nevertheless already tracked:

- `python/mojoboost/_dask_runtime.py` first appears in `dc21f03` /
  `860b1cf`, mid-write.
- `python/mojoboost/dask.py`'s changes here were swept into `e6f3959`.

Those are other lanes' whole-tree commits, taken while these files were on
disk, the same way `handoffs/task16_distributed.md` records happening to
`distributed_transport.mojo`. What landed in each is whatever the file held
at that instant, which for `_dask_runtime.py` is an earlier draft than the
one described above. Whoever assembles the round should read the working
tree rather than trusting those commit boundaries, and should expect this
handoff itself to land the same way.
