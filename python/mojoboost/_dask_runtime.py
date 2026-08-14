"""The native distributed runtime behind `mojoboost.dask`.

`mojoboost/dask.py` is the client-side contract: it takes a Dask collection
apart, validates the layout the distributed algorithm needs, plans ranks,
and turns model bytes back into a fitted estimator. It deliberately knows
nothing about how ranks talk to each other. This module is the one piece
that does: it finds the native distributed runtime in the extension module,
declares what that runtime can do, launches one rank per worker with the
data left where dask already put it, and hands the model bytes back through
the version 0 backend protocol.

There is exactly one backend registry, and it is the one in `dask.py`. This
module supplies exactly one backend for it, `NativeDistributedBackend`, and
`mojoboost.dask.get_backend()` falls back to it when the user registered
nothing. Nothing here is a second registry, a second trainer, or a second
model representation.

Nothing here trains in Python
-----------------------------

Every line of the training loop is Mojo. What this module does on the
worker is: concatenate the partitions dask already placed there, in
partition order, put them behind the same column-major float64 boundary
`mojoboost` uses for a single-machine fit, open a runtime session, call one
native entry point, and serialize rank 0's model with the same
`_mojoboost.save` the local estimators use. If the native entry points are
absent, it raises `DistributedNotAvailable` naming what is missing. It
never falls back to fitting on the client, and it never fits rank by rank
and averages, which would be a different algorithm wearing this one's name.

The adapter protocol
--------------------

Task 13 owns the native runtime and Task 14 owns its bindings, and both
were in flight while this was written. So this module does not import a
fixed set of symbols: it resolves a *provider*, an object carrying the five
members below, and the default provider is the extension module itself.

    provider.distributed_runtime_info()          -> record, see below
    provider.distributed_worker_open(config)     -> session handle (int)
    provider.distributed_worker_train(handle, X_addr, n_rows, n_features,
                                      y_addr, objective, params)
                                                 -> the fit record
    provider.distributed_worker_close(handle)    -> None, idempotent
    provider.distributed_worker_cancel(handle)   -> None, from any thread

`distributed_runtime_info()` answers with a mapping (or an object with the
same attributes) carrying at least:

    "available"          -> truthy only when a real transport can open a
                            connection between two processes. The in-memory
                            `MemoryEndpoint` in
                            src/mojoboost/distributed_transport.mojo is a
                            fake and must report false here.
    "reason"             -> why not, when `available` is false. Shown to
                            the user verbatim, so it says what is missing.
    "protocol_version"   -> the adapter version the runtime speaks. This
                            client speaks `ADAPTER_VERSION` and refuses
                            anything else rather than guessing.
    "capabilities"       -> names drawn from `mojoboost.dask.CAPABILITIES`.
                            The runtime declares what it supports; nothing
                            here decides for it.
    "transport"          -> a short name for the transport, for messages.
    "multi_host"         -> whether ranks may live on different hosts.

`distributed_worker_train` returns the same six-element record shape
`_mojoboost.fit_with_metrics` already returns, so the validation history
assembles here exactly the way it does for a single-machine fit:

    (model handle or 0, flat metric values, n_rounds, best_iteration,
     best_score, stopped_early)

A rank other than the root returns 0 as the model handle. See
handoffs/connect_15_dask.md for the exact patch requests behind those
names.

Import safety
-------------

Importing this module imports neither dask nor numpy. `dask.py` imports it
inside the functions that need it, so `import mojoboost.dask` stays as
cheap as it was. `native_runtime_status()` is safe to call anywhere,
including from a diagnostics path on a machine with no cluster: it catches
everything the probe can raise and turns it into an unavailable status with
a reason.
"""

import os
import threading
import time
import uuid
from dataclasses import dataclass, field

from .dask import (
    CAPABILITIES,
    DistributedNotAvailable,
    DistributedRankError,
    DistributedTimeout,
    PartitionError,
)

__all__ = [
    "ADAPTER_VERSION",
    "RUNTIME_ENTRY_POINTS",
    "NativeDistributedBackend",
    "NativeModelRef",
    "RuntimeStatus",
    "clear_runtime_provider",
    "describe_runtime",
    "native_backend",
    "native_runtime_status",
    "register_runtime_provider",
]

#: The version of the worker-side adapter protocol this module speaks. A
#: runtime that reports a different `protocol_version` is refused with a
#: message naming both numbers, because a silent mismatch here means ranks
#: interpreting the same params dict two different ways.
ADAPTER_VERSION = 1

#: The five members a provider must carry. Listed once so the "what is
#: missing" message can name them individually rather than saying that
#: something, somewhere, is absent.
RUNTIME_ENTRY_POINTS = (
    "distributed_runtime_info",
    "distributed_worker_open",
    "distributed_worker_train",
    "distributed_worker_close",
    "distributed_worker_cancel",
)

#: First TCP port a rank listens on. Rank `r` takes `base + r`, which keeps
#: two ranks on one host (legal under `one_rank_per_worker=False`) on
#: distinct ports, as `TransportConfig.validate` requires. A cluster whose
#: firewall wants a different range sets the environment variable on the
#: workers and the client alike.
DEFAULT_BASE_PORT = 24601
_BASE_PORT_ENV = "MOJOBOOST_DISTRIBUTED_BASE_PORT"

#: Seconds a rank waits for the world to connect, and seconds it waits for
#: any one collective. `TransportConfig` takes both in nanoseconds and a
#: `TrainingJob` carries one number, so the split is made here and stated
#: rather than hidden: a fit-wide timeout bounds a collective, and the
#: connect deadline stays short so a cluster that never assembles fails
#: while someone is still watching.
DEFAULT_CONNECT_TIMEOUT = 60.0
DEFAULT_COLLECTIVE_TIMEOUT = 600.0
_CONNECT_TIMEOUT_ENV = "MOJOBOOST_DISTRIBUTED_CONNECT_TIMEOUT"

#: `MOJOBOOST_DISTRIBUTED_PROVIDER` names an alternative provider as
#: `package.module:attribute`, for a build whose runtime does not live in
#: `mojoboost._mojoboost` and for bringing Task 13's runtime up under a
#: different set of names without editing this file.
_PROVIDER_ENV = "MOJOBOOST_DISTRIBUTED_PROVIDER"


# ---------------------------------------------------------------------------
# The provider seam
# ---------------------------------------------------------------------------

_PROVIDER = None
_STATUS = None


def register_runtime_provider(provider):
    """Install the object carrying the five native entry points.

    The default is the extension module, which is what a finished build
    exports them from. This exists so the runtime lane can be brought up
    behind its own names, and so a test double can stand in without a
    cluster.
    """
    global _PROVIDER, _STATUS
    _PROVIDER = provider
    _STATUS = None
    return provider


def clear_runtime_provider():
    """Forget a registered provider and the cached status."""
    global _PROVIDER, _STATUS
    _PROVIDER = None
    _STATUS = None


def _resolve_provider():
    """`(provider, source)`, or `(None, reason)` when there is none.

    Resolution order: whatever `register_runtime_provider` installed, then
    `MOJOBOOST_DISTRIBUTED_PROVIDER`, then the extension module.
    """
    if _PROVIDER is not None:
        return _PROVIDER, "a registered runtime provider"
    spec = os.environ.get(_PROVIDER_ENV, "").strip()
    if spec:
        module_name, _, attribute = spec.partition(":")
        if not module_name or not attribute:
            return None, (
                f"{_PROVIDER_ENV} is {spec!r}, which is not "
                "'package.module:attribute'"
            )
        try:
            import importlib

            module = importlib.import_module(module_name)
            return getattr(module, attribute), f"{_PROVIDER_ENV}={spec}"
        except (ImportError, AttributeError) as exc:
            return None, (
                f"{_PROVIDER_ENV} names {spec!r}, which cannot be "
                f"resolved: {exc}"
            )
    try:
        from . import _mojoboost
    except ImportError as exc:  # pragma: no cover - a broken install
        return None, (
            "the mojoboost extension module cannot be imported "
            f"({exc}), so there is no native distributed runtime to reach"
        )
    return _mojoboost, "the mojoboost extension module"


# ---------------------------------------------------------------------------
# What the runtime says about itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeStatus:
    """Whether the native distributed runtime can be used, and why not.

    `reason` is written to be shown to a user unchanged: it names the
    missing piece rather than describing a category of missing piece.
    """

    available: bool
    reason: str = ""
    source: str = ""
    capabilities: frozenset = frozenset()
    protocol_version: int = 0
    transport: str = ""
    multi_host: bool = False
    missing: tuple = ()

    def require(self):
        """Return self, or raise `DistributedNotAvailable(reason)`."""
        if not self.available:
            raise DistributedNotAvailable(self.reason)
        return self


def native_runtime_status(refresh=False):
    """Whether this build can train across a cluster, cached per process.

    Never raises. A probe that fails for any reason at all comes back as an
    unavailable status carrying the failure, because this is called from
    `get_backend`, from diagnostics, and from every worker at the top of a
    rank task, and none of those want an exception from a capability
    question.
    """
    global _STATUS
    if _STATUS is not None and not refresh:
        return _STATUS
    provider, source = _resolve_provider()
    if provider is None:
        _STATUS = RuntimeStatus(available=False, reason=source, source="")
        return _STATUS
    _STATUS = _status_from_provider(provider, source)
    return _STATUS


def _status_from_provider(provider, source):
    missing = tuple(
        name for name in RUNTIME_ENTRY_POINTS if not hasattr(provider, name)
    )
    if missing:
        return RuntimeStatus(
            available=False,
            source=source,
            missing=missing,
            reason=(
                f"{source} exports no native distributed runtime: "
                f"{', '.join(missing)} "
                f"{'are' if len(missing) > 1 else 'is'} missing. The "
                "transport in src/mojoboost/distributed_transport.mojo is "
                "not finished (it has no socket endpoint), and its "
                "bindings are not built, so this installation cannot train "
                "across a cluster. See handoffs/connect_15_dask.md for the "
                "entry points it owes"
            ),
        )
    try:
        record = provider.distributed_runtime_info()
    except Exception as exc:  # the probe must not raise, ever
        return RuntimeStatus(
            available=False,
            source=source,
            reason=(
                f"{source} has the distributed entry points, but "
                f"distributed_runtime_info() failed: {exc!r}"
            ),
        )
    get = _record_reader(record)
    protocol = _as_int(get("protocol_version"), 0)
    if protocol != ADAPTER_VERSION:
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            reason=(
                f"the native distributed runtime speaks adapter protocol "
                f"{protocol} and this mojoboost speaks {ADAPTER_VERSION}. "
                "They disagree about the shape of the config and params "
                "dicts a rank is opened with, so the fit is refused rather "
                "than guessed at. Install a matching mojoboost on the "
                "client and every worker"
            ),
        )
    transport = str(get("transport") or "unnamed")
    multi_host = bool(get("multi_host"))
    names = tuple(str(name) for name in (get("capabilities") or ()))
    unknown = sorted(set(names) - set(CAPABILITIES))
    if unknown:
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            transport=transport,
            reason=(
                f"the native distributed runtime declares capabilities "
                f"{unknown} that mojoboost.dask does not know. The known "
                f"names are {sorted(CAPABILITIES)}; a name outside them is "
                "a typo or a version mismatch, and reading it as a missing "
                "feature would hide it"
            ),
        )
    if not get("available"):
        reason = str(get("reason") or "").strip()
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            transport=transport,
            multi_host=multi_host,
            capabilities=frozenset(names),
            reason=(
                "the native distributed runtime is present but reports "
                "itself unusable: "
                + (reason or "it gave no reason, which is a runtime bug")
            ),
        )
    if not names:
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            transport=transport,
            multi_host=multi_host,
            reason=(
                "the native distributed runtime declares no capabilities, "
                "so every fit would be refused for a different reason one "
                "at a time. A runtime that can train regression across "
                "ranks declares 'regression'"
            ),
        )
    return RuntimeStatus(
        available=True,
        source=source,
        capabilities=frozenset(names),
        protocol_version=protocol,
        transport=transport,
        multi_host=multi_host,
    )


def _record_reader(record):
    """A `get(name)` over either a mapping or an object with attributes, so
    a binding may answer with whichever is natural on the Mojo side."""
    if hasattr(record, "get"):

        def get(name):
            try:
                return record.get(name)
            except Exception:
                return None

        return get

    def get(name):
        return getattr(record, name, None)

    return get


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def describe_runtime():
    """One line for a diagnostics report."""
    status = native_runtime_status()
    if not status.available:
        return f"distributed: unavailable ({status.reason})"
    hosts = "multi-host" if status.multi_host else "single-host"
    return (
        f"distributed: {status.transport} ({hosts}), capabilities "
        f"{sorted(status.capabilities)}"
    )


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


def native_backend(client=None):
    """The native distributed backend, or `DistributedNotAvailable`.

    Called by `mojoboost.dask.get_backend()` when nothing else is
    registered. The status check happens here, before a collection is
    touched, so an installation without a transport says so immediately
    rather than after persisting a terabyte.
    """
    status = native_runtime_status().require()
    return NativeDistributedBackend(status=status, client=client)


@dataclass(frozen=True)
class _Launch:
    """What every rank of one fit has to agree on.

    Frozen and made of scalars and strings so it pickles to the workers
    unchanged. `job_id` is generated once on the client, which is the only
    place that can give every rank the same value.
    """

    job_id: int
    addresses: tuple
    connect_timeout_ns: int
    collective_timeout_ns: int
    protocol_version: int = ADAPTER_VERSION


class NativeDistributedBackend:
    """The version 0 backend protocol over the native runtime.

    `name`, `capabilities`, and `train(job)` are the protocol. `cancel` and
    `bind_runtime` are optional extensions `mojoboost.dask` calls through
    `getattr`, so a third-party backend that lacks them still works.
    """

    def __init__(self, status=None, client=None, base_port=None):
        self._status = status or native_runtime_status()
        self._client = client
        self._runtime = None
        self._base_port = base_port
        self._live = {}
        self._lock = threading.Lock()

    name = "mojoboost-native"

    @property
    def capabilities(self):
        return self._status.capabilities

    @property
    def status(self):
        return self._status

    def bind_runtime(self, runtime):
        """Take the runtime the estimator is fitting through.

        The client is a property of the collection being fitted, not of the
        backend, and `TrainingJob` deliberately carries no live handle. So
        `mojoboost.dask` hands the runtime over just before `train`, and
        this is where the `distributed.Client` comes from.
        """
        self._runtime = runtime
        return self

    # -- the cluster ------------------------------------------------------

    def _client_for(self):
        if self._client is not None:
            return self._client
        runtime = self._runtime
        getter = getattr(runtime, "client", None)
        if callable(getter):
            client = getter()
            if client is not None:
                return client
        distributed = _import_distributed()
        try:
            return distributed.get_client()
        except (ValueError, RuntimeError) as exc:
            raise DistributedNotAvailable(
                "distributed training needs a dask Client, and there is no "
                f"current one ({exc}). Pass client= to the estimator, or "
                "create a Client before fitting"
            ) from exc

    # -- training ---------------------------------------------------------

    def train(self, job):
        """Launch one rank per `job.plan.ranks` entry and return a model
        reference for rank 0's model.

        The training rows never move to the client. Each rank task is
        submitted to the worker that already holds its partitions, with the
        partitions passed as futures, so dask hands the worker the blocks
        it is already storing.
        """
        self._status.require()
        _check_protocol(job)
        client = self._client_for()
        plan = job.plan
        launch = _Launch(
            job_id=_job_id(),
            addresses=_transport_addresses(plan, self._base_port),
            connect_timeout_ns=_connect_timeout_ns(job.timeout),
            collective_timeout_ns=_collective_timeout_ns(job.timeout),
        )
        _check_thread_capacity(client, plan)
        futures = []
        for rank in plan.ranks:
            futures.append(
                client.submit(
                    _train_rank,
                    rank.rank,
                    launch,
                    _rank_call(job, rank),
                    _rank_blocks(client, job, rank),
                    key=f"mojoboost-rank-{launch.job_id}-{rank.rank}",
                    workers=[rank.worker],
                    allow_other_workers=False,
                    pure=False,
                    retries=0,
                )
            )
        with self._lock:
            self._live[launch.job_id] = (client, tuple(futures), plan)
        return NativeModelRef(
            backend=self,
            client=client,
            futures=tuple(futures),
            plan=plan,
            job_id=launch.job_id,
        )

    def cancel(self, job=None, job_id=None):
        """Stop every rank of a running fit, from the client.

        `mojoboost.dask` calls this when the caller interrupts a fit. Both
        halves matter: cancelling the dask futures stops the tasks from
        being retried or rescheduled, and `client.run` reaches the
        worker-local session registry so a rank already blocked inside a
        collective is woken by the runtime's own cancellation path rather
        than left waiting for a deadline.
        """
        with self._lock:
            entries = (
                list(self._live.items())
                if job_id is None
                else [(job_id, self._live.get(job_id))]
            )
        for identifier, entry in entries:
            if entry is None:
                continue
            client, futures, plan = entry
            _cancel_world(client, identifier, futures, plan)
            self.forget(identifier)

    def forget(self, job_id):
        with self._lock:
            self._live.pop(job_id, None)


def _check_protocol(job):
    version = getattr(job, "protocol_version", None)
    from .dask import BACKEND_PROTOCOL_VERSION

    if version != BACKEND_PROTOCOL_VERSION:
        raise DistributedNotAvailable(
            f"this backend speaks backend protocol "
            f"{BACKEND_PROTOCOL_VERSION} and the job declares {version!r}. "
            "Refusing rather than guessing which fields moved"
        )


# ---------------------------------------------------------------------------
# The model reference
# ---------------------------------------------------------------------------


class NativeModelRef:
    """The three-member model reference over a set of rank futures.

    `result` waits for every rank rather than only for the root: a
    non-root rank that dies leaves the root blocked inside a collective
    until its deadline, and reporting the rank that actually failed is
    worth more than the wait. `release` cancels whatever is still running
    and is idempotent, so `take_model_bytes` may call it in a `finally`
    whether or not the fit succeeded.
    """

    def __init__(self, backend, client, futures, plan, job_id):
        self._backend = backend
        self._client = client
        self._futures = tuple(futures)
        self._plan = plan
        self._job_id = job_id
        self._record = None
        self.released = False
        self.owner = plan.ranks[0].worker if plan.ranks else ""

    def result(self, timeout=None):
        """The bytes of rank 0's model, once every rank has finished."""
        if self.released:
            raise _ownership_error(
                "this model reference was already released"
            )
        payloads = _await_ranks(
            self._client,
            self._futures,
            self._plan,
            self._job_id,
            timeout,
        )
        self._record = payloads[0] if payloads else None
        blob = (self._record or {}).get("model") or b""
        return bytes(blob)

    def metrics(self):
        """The validation history rank 0 reported, or None.

        Read by `mojoboost.dask` after `result` and kept after `release`,
        because it is plain data rather than anything holding a worker.
        """
        return self._record

    def release(self):
        if self.released:
            return
        self.released = True
        futures, self._futures = self._futures, ()
        _cancel_world(self._client, self._job_id, futures, self._plan)
        self._backend.forget(self._job_id)


def _ownership_error(message):
    from .dask import ModelOwnershipError

    return ModelOwnershipError(message)


# ---------------------------------------------------------------------------
# Waiting, cancelling, and losing a worker
# ---------------------------------------------------------------------------


def _await_ranks(client, futures, plan, job_id, timeout):
    """Every rank's payload, in rank order.

    Three ways this ends badly, and each gets its own message: a rank
    raised (the native error, or `KilledWorker` when the worker died), the
    fit ran past its timeout, and the caller interrupted. All three cancel
    the rest of the world first, because a surviving rank sitting in a
    collective is a worker held indefinitely.
    """
    distributed = _import_distributed()
    pending = list(futures)
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        while pending:
            wait_for = None
            if deadline is not None:
                wait_for = max(deadline - time.monotonic(), 0.0)
            try:
                done, not_done = distributed.wait(
                    pending, timeout=wait_for, return_when="FIRST_COMPLETED"
                )
            except TimeoutError:
                _cancel_world(client, job_id, futures, plan)
                raise DistributedTimeout(
                    f"the distributed fit did not finish within {timeout} "
                    f"seconds; {len(pending)} of {len(futures)} ranks were "
                    "still running and every rank has been cancelled"
                ) from None
            for future in done:
                if future.status == "error":
                    _cancel_world(client, job_id, futures, plan)
                    raise _rank_failure(future, futures, plan)
            pending = list(not_done)
    except (KeyboardInterrupt, SystemExit):
        _cancel_world(client, job_id, futures, plan)
        raise
    return [future.result() for future in futures]


def _rank_failure(future, futures, plan):
    """The exception a failed rank future turns into.

    The rank and the worker are named because a distributed traceback
    alone says which task object failed and not which shard of the data it
    was holding, and that is the first thing anyone asks.
    """
    index = list(futures).index(future)
    rank = plan.ranks[index]
    try:
        future.result()
        cause = None
    except BaseException as exc:  # re-raised below as the cause
        cause = exc
    detail = f"{type(cause).__name__}: {cause}" if cause else "no detail"
    error = DistributedRankError(
        f"rank {rank.rank} on {rank.worker} failed, so the fit was "
        f"cancelled on every rank ({detail}). A rank that dies mid-fit "
        "leaves the collective incomplete; there is no partial model to "
        "recover"
    )
    if cause is not None:
        error.__cause__ = cause
    return error


def _cancel_world(client, job_id, futures, plan):
    """Best effort: wake the runtime sessions, then cancel the tasks.

    Both halves are wrapped, because this runs on the failure path and an
    exception raised while cleaning up would replace the error the caller
    actually needs to see.
    """
    workers = sorted({rank.worker for rank in getattr(plan, "ranks", ())})
    if workers:
        try:
            client.run(_cancel_sessions, job_id, workers=workers)
        except Exception:
            pass
    for future in futures:
        try:
            future.cancel()
        except Exception:
            pass


def _check_thread_capacity(client, plan):
    """Refuse a layout that would deadlock before it deadlocks.

    Every rank runs a blocking collective, so all of them have to be
    executing at once. Two ranks on a one-thread worker cannot be, and the
    symptom is a cluster that sits at zero CPU until a deadline expires
    rather than an error. `one_rank_per_worker=False` on a small cluster is
    exactly how someone arrives here.
    """
    counts = {}
    for rank in plan.ranks:
        counts[rank.worker] = counts.get(rank.worker, 0) + 1
    try:
        info = client.scheduler_info().get("workers", {})
    except Exception:  # pragma: no cover - depends on the scheduler
        return
    for worker, wanted in counts.items():
        entry = info.get(worker)
        if not entry:
            continue
        threads = _as_int(entry.get("nthreads"), 0)
        if threads and wanted > threads:
            raise PartitionError(
                f"worker {worker} would run {wanted} ranks but has "
                f"{threads} thread(s). Every rank blocks inside the same "
                "collective, so they have to run at the same time; with "
                "fewer threads than ranks the fit deadlocks rather than "
                "failing. Rebalance the collection so each worker holds "
                "adjacent partitions and use one_rank_per_worker=True, or "
                "start the workers with more threads"
            )


# ---------------------------------------------------------------------------
# From a WorldPlan to a transport config
# ---------------------------------------------------------------------------


def _transport_addresses(plan, base_port=None):
    """`host:port` per rank, in rank order.

    A dask worker address is not a transport address: `tcp://10.0.0.4:38921`
    is the port the worker's own server listens on. A mojoboost rank needs
    a port of its own, and nothing in the plan can invent one, so rank `r`
    takes `base + r`. Two ranks on one host therefore get distinct ports,
    which is what `TransportConfig.validate` requires and what
    `WorldPlan.addresses_unique` being false means in practice.
    """
    base = base_port
    if base is None:
        base = _as_int(
            os.environ.get(_BASE_PORT_ENV, ""), DEFAULT_BASE_PORT
        )
    if not 1 <= base <= 65535 - max(len(plan.ranks) - 1, 0):
        raise PartitionError(
            f"the rank port range starting at {base} does not fit "
            f"{len(plan.ranks)} ranks below 65535; set "
            f"{_BASE_PORT_ENV} to a lower base port"
        )
    addresses = []
    for rank in plan.ranks:
        host = _host_of(rank.worker)
        if not host:
            raise PartitionError(
                f"rank {rank.rank} has worker address {rank.worker!r}, "
                "which carries no host, so no transport address can be "
                "built for it"
            )
        addresses.append(f"{host}:{base + rank.rank}")
    if len(set(addresses)) != len(addresses):
        raise PartitionError(
            f"two ranks would listen on the same address ({addresses}); "
            "this is a bug in the rank plan rather than something to "
            "configure around"
        )
    return tuple(addresses)


def _host_of(worker):
    """The host out of a dask worker address.

    `tcp://10.0.0.4:38921` and `10.0.0.4:38921` both give `10.0.0.4`; an
    inproc or ucx address gives whatever sits in the same position, which
    is the honest answer for a transport that has to connect to it.
    """
    address = str(worker)
    if "://" in address:
        address = address.split("://", 1)[1]
    if address.startswith("["):  # a bracketed IPv6 literal
        end = address.find("]")
        return address[1:end] if end > 0 else ""
    return address.rsplit(":", 1)[0] if ":" in address else address


def _job_id():
    """A job id every rank agrees on, unique per launch.

    A transport job id is a property of the launch and not of the training
    request, which is why `TrainingJob` has no field for it. 63 bits so it
    fits an `Int` on the Mojo side.
    """
    return uuid.uuid4().int >> 65


def _connect_timeout_ns(timeout):
    default = _float_env(_CONNECT_TIMEOUT_ENV, DEFAULT_CONNECT_TIMEOUT)
    if timeout is None:
        seconds = default
    else:
        seconds = min(float(timeout), default)
    return int(max(seconds, 0.0) * 1_000_000_000)


def _collective_timeout_ns(timeout):
    seconds = (
        DEFAULT_COLLECTIVE_TIMEOUT if timeout is None else float(timeout)
    )
    return int(max(seconds, 0.0) * 1_000_000_000)


def _float_env(name, default):
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# What one rank is asked to do
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RankCall:
    """The training request, minus the data, as one rank sees it.

    Everything here is a scalar, a string, a tuple, or a plain dict of
    those, so it pickles to a worker and back with no mojoboost object on
    the wire but the ones defined in this file.
    """

    rank: int
    world_size: int
    row_offset: int
    n_rows: int
    n_features: int
    objective: str
    params: dict
    label_classes: tuple = None
    ranking: bool = False
    categorical: tuple = ()
    feature_names: tuple = None
    group: tuple = None
    eval_names: tuple = ()
    eval_metrics: tuple = ()
    early_stopping_rounds: int = 0
    first_metric_only: bool = False


def _rank_call(job, rank):
    plan = job.plan
    group = None
    if job.ranking:
        group = tuple(
            count
            for index in rank.partitions
            for count in (plan.partitions[index].group or ())
        )
    return _RankCall(
        rank=rank.rank,
        world_size=plan.world_size,
        row_offset=rank.row_offset,
        n_rows=rank.n_rows,
        n_features=plan.n_features,
        objective=job.objective,
        params=dict(job.params),
        label_classes=job.label_classes,
        ranking=bool(job.ranking),
        categorical=tuple(
            (index, tuple(cats)) for index, cats in plan.schema.columns
        ),
        feature_names=plan.feature_names,
        group=group,
        eval_names=tuple(
            valid.name for valid in getattr(job, "validation", ())
        ),
        eval_metrics=tuple(getattr(job, "eval_metrics", ())),
        early_stopping_rounds=int(
            getattr(job, "early_stopping_rounds", 0) or 0
        ),
        first_metric_only=bool(getattr(job, "first_metric_only", False)),
    )


@dataclass(frozen=True)
class _RankBlocks:
    """Futures for the blocks one rank owns, in row order.

    Futures rather than values: dask resolves them on the worker, and the
    worker is the one already holding them, so the training rows never
    reach the client. That is the whole reason this module submits tasks
    instead of gathering partitions and calling a local fit.
    """

    features: tuple = ()
    labels: tuple = ()
    weights: tuple = ()
    validation: tuple = ()  # one _RankBlocks-shaped tuple per eval set


def _rank_blocks(client, job, rank):
    plan = job.plan
    parts = [plan.partitions[index] for index in rank.partitions]
    blocks = _RankBlocks(
        features=_futures_for(client, [part.key for part in parts]),
        labels=_futures_for(client, [part.label_key for part in parts]),
        weights=_futures_for(client, [part.weight_key for part in parts]),
        validation=tuple(
            _validation_blocks(client, valid, rank)
            for valid in getattr(job, "validation", ())
        ),
    )
    return blocks


def _validation_blocks(client, valid, rank):
    """The eval set's blocks for this rank.

    A rank scores the validation rows that live where its training rows do,
    which is why `mojoboost.dask` requires an eval collection with the same
    rank layout: the alternative is shipping validation rows between
    workers every round.
    """
    plan = valid.plan
    owned = plan.ranks[rank.rank]
    parts = [plan.partitions[index] for index in owned.partitions]
    return _RankBlocks(
        features=_futures_for(client, [part.key for part in parts]),
        labels=_futures_for(client, [part.label_key for part in parts]),
        weights=_futures_for(client, [part.weight_key for part in parts]),
    )


def _futures_for(client, keys):
    """Futures for scheduler keys, or `()` when the collection was absent.

    A `None` key means the caller passed no labels or no weights for that
    partition; a partly-filled list is a bug in the runtime's metadata
    rather than something to paper over here.
    """
    keys = list(keys)
    if not keys or all(key is None for key in keys):
        return ()
    if any(key is None for key in keys):
        raise PartitionError(
            "some partitions carry a data key and some do not, so the rank "
            "cannot be given a complete block of rows. This is a runtime "
            "metadata bug: report the collection that produced it"
        )
    distributed = _import_distributed()
    return tuple(distributed.Future(key, client=client) for key in keys)


# ---------------------------------------------------------------------------
# The worker side
# ---------------------------------------------------------------------------

#: Open runtime sessions in this worker process, by job id. A session is a
#: real resource (a listener, peer connections, and the runtime's state
#: machine), so it is registered while it is open and removed in a
#: `finally`. `_cancel_sessions` is the only reason the registry exists:
#: cancellation arrives as a separate task and has to find the session the
#: training task is blocked inside.
_WORKER_SESSIONS = {}
_WORKER_LOCK = threading.Lock()


def _train_rank(rank, launch, call, blocks):
    """One rank of one distributed fit, on the worker that holds its rows.

    Runs on a dask worker. Everything it does is glue: concatenate,
    convert, open, call, close. The training loop is Mojo, and if this
    build has no native runtime it says so here rather than falling back to
    anything.
    """
    provider, source = _resolve_provider()
    status = native_runtime_status(refresh=True)
    if not status.available:
        raise DistributedNotAvailable(
            f"rank {rank} cannot train: {status.reason}. The client's "
            "mojoboost has a distributed runtime and this worker's does "
            "not, or they are different builds"
        )
    from . import _arrays

    features = _concat_blocks(blocks.features, "X", rank)
    if features is None:
        raise DistributedRankError(
            f"rank {rank} was given no feature partitions"
        )
    encoders = {index: list(cats) for index, cats in call.categorical}
    Xb, n_rows, n_features, _ = _arrays.check_X(features, encoders=encoders)
    if n_rows != call.n_rows or n_features != call.n_features:
        raise DistributedRankError(
            f"rank {rank} was planned for {call.n_rows} rows and "
            f"{call.n_features} features but holds {n_rows} by "
            f"{n_features}. The plan and the data disagree, and a rank "
            "that trains on the wrong shape silently corrupts every "
            "reduced histogram"
        )
    labels = _concat_blocks(blocks.labels, "y", rank)
    if labels is None:
        raise DistributedRankError(
            f"rank {rank} was given no labels; distributed fit needs y "
            "partitioned exactly as X is"
        )
    if call.label_classes:
        yb = _encode_labels_through(labels, call.label_classes, n_rows, rank)
    else:
        yb = _arrays.check_target(labels, n_rows)
    weights = _concat_blocks(blocks.weights, "sample_weight", rank)
    wb = None if weights is None else _arrays.check_sample_weight(
        weights, n_rows
    )
    valid_buffers = []
    valid_specs = []
    for index, valid in enumerate(blocks.validation):
        spec, keep = _validation_arrays(
            valid, call, index, n_features, rank, _arrays
        )
        valid_specs.append(spec)
        valid_buffers.append(keep)
    params = _native_params(call, wb, valid_specs, _arrays)
    config = _open_config(rank, launch, call)
    handle = provider.distributed_worker_open(config)
    _register_session(launch.job_id, provider, handle)
    try:
        record = provider.distributed_worker_train(
            handle,
            _arrays.addr(Xb),
            n_rows,
            n_features,
            _arrays.addr(yb),
            call.objective,
            params,
        )
        payload = _payload_from_record(record, call, rank, provider)
    finally:
        # The buffers had to outlive the call above, and the session has to
        # be closed whether the call returned or raised: a session left open
        # holds a listener and a peer connection for the life of the worker.
        del Xb, yb, wb, valid_buffers
        _forget_session(launch.job_id)
        try:
            provider.distributed_worker_close(handle)
        except Exception:
            pass
    return payload


def _open_config(rank, launch, call):
    """The dict `distributed_worker_open` is given.

    It carries only what a session needs to agree with its peers: who is in
    the world, which rank this is, the deadlines, and the shape facts the
    handshake digests. Training parameters travel with the training call
    instead, so a schema mismatch is caught by the handshake rather than
    halfway through the first histogram.
    """
    return {
        "protocol_version": launch.protocol_version,
        "job_id": launch.job_id,
        "rank": rank,
        "world_size": call.world_size,
        "addresses": list(launch.addresses),
        "connect_timeout_ns": launch.connect_timeout_ns,
        "collective_timeout_ns": launch.collective_timeout_ns,
        "n_features": call.n_features,
        "row_offset": call.row_offset,
        "n_rows": call.n_rows,
        "objective": call.objective,
    }


def _native_params(call, wb, valid_specs, _arrays):
    """The params dict the native rank entry parses.

    The estimator's LightGBM-spelled parameters travel unchanged, because
    the binding parses them with the same code the single-machine entry
    points use and parameter policy stays in Mojo. What is added here is
    only what has no spelling in that vocabulary: buffer addresses, the
    global class list, the categorical feature indices, the query group,
    and the validation sets.
    """
    params = dict(call.params)
    params["weight_addr"] = 0 if wb is None else _arrays.addr(wb)
    params["categorical_feature"] = [index for index, _ in call.categorical]
    params["n_classes"] = len(call.label_classes or ()) or 0
    if call.ranking:
        group = list(call.group or ())
        params["group"] = group
        params["n_groups"] = len(group)
    if valid_specs:
        params["valid_sets"] = valid_specs
        params["eval_names"] = list(call.eval_names)
        params["metric"] = list(call.eval_metrics)
        params["early_stopping_rounds"] = call.early_stopping_rounds
        params["first_metric_only"] = int(bool(call.first_metric_only))
    return params


def _validation_arrays(valid, call, index, n_features, rank, _arrays):
    """`(spec, buffers)` for one eval set on this rank.

    The buffers are returned alongside the spec because the spec holds
    their addresses: dropping them before the native call returns would
    hand the trainer freed memory.
    """
    features = _concat_blocks(valid.features, f"eval_set[{index}] X", rank)
    labels = _concat_blocks(valid.labels, f"eval_set[{index}] y", rank)
    if features is None or labels is None:
        raise DistributedRankError(
            f"rank {rank} is missing its rows for eval set {index}; every "
            "rank scores the validation partitions that live with its "
            "training partitions"
        )
    encoders = {i: list(cats) for i, cats in call.categorical}
    Xb, n_rows, got_features, _ = _arrays.check_X(
        features, name=f"eval_set[{index}] X", encoders=encoders
    )
    if got_features != n_features:
        raise DistributedRankError(
            f"eval set {index} has {got_features} features and the "
            f"training data has {n_features}"
        )
    yb = _arrays.check_target(labels, n_rows, name=f"eval_set[{index}] y")
    weights = _concat_blocks(
        valid.weights, f"eval_set[{index}] weight", rank
    )
    wb = None if weights is None else _arrays.check_sample_weight(
        weights, n_rows
    )
    spec = {
        "X_addr": _arrays.addr(Xb),
        "y_addr": _arrays.addr(yb),
        "weight_addr": 0 if wb is None else _arrays.addr(wb),
        "n_rows": n_rows,
    }
    return spec, (Xb, yb, wb)


def _payload_from_record(record, call, rank, provider):
    """The rank's return value: model bytes on the root, history on the
    root, and nothing on anyone else.

    The record shape is the one `_mojoboost.fit_with_metrics` already
    returns, so the history assembles here the way it does for a
    single-machine fit and no second convention enters the codebase.
    """
    values = ()
    n_rounds = 0
    best_iteration = -1
    best_score = float("nan")
    stopped_early = False
    model = 0
    if isinstance(record, (tuple, list)):
        model = record[0]
        if len(record) > 1:
            values = record[1] or ()
        if len(record) > 2:
            n_rounds = _as_int(record[2], 0)
        if len(record) > 3:
            best_iteration = _as_int(record[3], -1)
        if len(record) > 4:
            try:
                best_score = float(record[4])
            except (TypeError, ValueError):
                best_score = float("nan")
        if len(record) > 5:
            stopped_early = bool(record[5])
    else:
        model = record
    if rank != 0:
        return {"model": b"", "rank": rank}
    if not model:
        raise DistributedRankError(
            "rank 0 finished without a model. The runtime returned "
            f"{record!r}, and the root rank is the one that owns the "
            "trained model"
        )
    return {
        "model": _model_bytes(provider, model, call),
        "rank": rank,
        "values": list(values),
        "n_rounds": n_rounds,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "stopped_early": stopped_early,
        "eval_names": list(call.eval_names),
        "eval_metrics": list(call.eval_metrics),
    }


def _model_bytes(provider, model, call):
    """Rank 0's model in the versioned text format `save()` writes.

    The same format the local estimators read, deliberately: the client
    rebuilds the estimator with the existing `load` / `load_multiclass`, so
    a distributed model pickles, saves, and predicts exactly like a local
    one and no new serialization path exists to drift.
    """
    import os as _os
    import tempfile as _tempfile

    multiclass = len(call.label_classes or ()) > 2
    save = getattr(
        provider,
        "save_multiclass" if multiclass else "save",
        None,
    )
    if save is None:  # pragma: no cover - a provider without the exports
        from . import _mojoboost

        save = (
            _mojoboost.save_multiclass if multiclass else _mojoboost.save
        )
    with _tempfile.TemporaryDirectory() as directory:
        path = _os.path.join(directory, "model.mbst")
        save(model, path)
        with open(path, "rb") as handle:
            return handle.read()


def _register_session(job_id, provider, handle):
    with _WORKER_LOCK:
        _WORKER_SESSIONS[job_id] = (provider, handle)


def _forget_session(job_id):
    with _WORKER_LOCK:
        _WORKER_SESSIONS.pop(job_id, None)


def _cancel_sessions(dask_worker=None, job_id=None):
    """Cancel this worker's session for a job. Runs under `client.run`.

    `client.run` passes `dask_worker` and calls this with whatever extra
    arguments were given, so the job id is a keyword with a default rather
    than a positional. Cancelling an unknown job is not an error: a worker
    that already finished its rank has nothing to cancel.
    """
    with _WORKER_LOCK:
        entry = _WORKER_SESSIONS.get(job_id)
    if entry is None:
        return False
    provider, handle = entry
    try:
        provider.distributed_worker_cancel(handle)
    except Exception:
        return False
    return True


def _concat_blocks(blocks, name, rank):
    """One rank's blocks joined in partition order, or None.

    Order preserving by construction: the blocks arrive in the order the
    rank names its partitions, which is the global row order, and a rank's
    partitions are contiguous in it. Anything else would silently permute
    training rows against their labels.
    """
    parts = [block for block in blocks if block is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    if hasattr(first, "iloc") or hasattr(first, "columns"):
        import pandas as pd

        return pd.concat(parts, axis=0, copy=False)
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is optional
        raise DistributedRankError(
            f"rank {rank} holds {len(parts)} {name} blocks and numpy is "
            "not installed on this worker to join them"
        ) from exc
    return np.concatenate([np.asarray(part) for part in parts], axis=0)


def _encode_labels_through(labels, classes, n_rows, rank):
    """Labels as codes into the fit's global class list.

    Never `_arrays.encode_labels`, which derives the classes from the data
    it is given: a rank that happens to hold two of three classes would
    number them 0 and 1 and disagree with every other rank about what class
    1 is. The global list comes from `fit(classes=...)` for exactly this
    reason, and a label outside it is an error naming the rank.
    """
    lookup = {label: index for index, label in enumerate(classes)}
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is optional
        np = None
    values = list(labels) if np is None else list(np.asarray(labels).ravel())
    if len(values) != n_rows:
        raise DistributedRankError(
            f"rank {rank} has {n_rows} rows and {len(values)} labels"
        )
    codes = []
    for value in values:
        try:
            code = lookup[value]
        except (KeyError, TypeError):
            raise DistributedRankError(
                f"rank {rank} holds label {value!r}, which is not in the "
                f"class list the fit was given ({list(classes)!r}). Pass "
                "every label value in classes=, in the order the model "
                "should number them"
            ) from None
        codes.append(float(code))
    if np is None:
        import array as _array

        return _array.array("d", codes)
    return np.asarray(codes, dtype=np.float64)


# ---------------------------------------------------------------------------
# dask, imported nowhere else in this module
# ---------------------------------------------------------------------------


def _import_distributed():
    try:
        import distributed
    except ImportError as exc:
        raise DistributedNotAvailable(
            "distributed training over a cluster needs the dask "
            "distributed scheduler, which is an optional dependency: pip "
            "install 'dask[distributed]'"
        ) from exc
    return distributed
