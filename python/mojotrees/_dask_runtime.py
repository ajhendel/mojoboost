"""The native distributed runtime behind `mojotrees.dask`.

`mojotrees/dask.py` is the client-side contract: it takes a Dask collection
apart, validates the layout the distributed algorithm needs, plans ranks,
and turns model bytes back into a fitted estimator. It deliberately knows
nothing about how ranks talk to each other. This module is the one piece
that does: it finds the native distributed runtime in the extension module,
declares what that runtime can do, launches one rank per worker with the
data left where dask already put it, and hands the model bytes back through
the version 0 backend protocol.

There is exactly one backend registry, and it is the one in `dask.py`. This
module supplies exactly one backend for it, `NativeDistributedBackend`, and
`mojotrees.dask.get_backend()` falls back to it when the user registered
nothing. Nothing here is a second registry, a second trainer, or a second
model representation.

Nothing here trains in Python
-----------------------------

Every line of the training loop is Mojo. What this module does on the
worker is: concatenate the partitions dask already placed there, in
partition order, put them behind the same column-major float64 boundary
`mojotrees` uses for a single-machine fit, open a runtime session, call one
native entry point, and serialize rank 0's model with the same
`_mojotrees.save` the local estimators use. If the native entry points are
absent, it raises `DistributedNotAvailable` naming what is missing. It
never falls back to fitting on the client, and it never fits rank by rank
and averages, which would be a different algorithm wearing this one's name.

The adapter protocol
--------------------

Task 13 owns the native runtime and Task 14 owns its bindings, and both
were in flight while this was written. So this module does not import a
fixed set of symbols: it resolves a *provider*, an object carrying the
members below, and the default provider is the extension module itself.

The capability half already exists. `bindings/distributed_bindings.mojo`
exports `distributed_capability()`, and this module reads exactly the
record it documents:

    provider.distributed_capability() -> {"multi_process": bool,
        "local_collective": bool, "protocol_version": int,
        "max_world_size": int, "reason": str}

`multi_process` is False in every build today and `reason` says why, so
that record is the authority on whether a cluster fit can run at all.
Nothing here second-guesses it, and nothing here reads
`local_collective` as permission to train: a single-process world is not
a distributed fit and this module will not present it as one.

The worker half does not exist yet. These are the entry points a rank
task calls, and `handoffs/connect_15_dask.md (deleted, recover with git log --all --diff-filter=D -- handoffs/connect_15_dask.md)` carries the exact request
for them:

    provider.distributed_worker_open(config)     -> session handle (int)
    provider.distributed_worker_train(handle, X_addr, n_rows, n_features,
                                      y_addr, objective, params)
                                                 -> the fit record
    provider.distributed_worker_close(handle)    -> None, idempotent
    provider.distributed_worker_cancel(handle)   -> None, from any thread

A build that grows a richer record may export `distributed_runtime_info()`
as well, which is read in preference to `distributed_capability()` and
adds the fields the capability record has no room for:

    "available"          -> truthy only when a real transport can open a
                            connection between two processes. The in-memory
                            `MemoryEndpoint` in
                            src/mojotrees/distributed_transport.mojo is a
                            fake and must report false here.
    "reason"             -> why not, when `available` is false. Shown to
                            the user verbatim, so it says what is missing.
    "protocol_version"   -> the adapter version the runtime speaks. This
                            client speaks `ADAPTER_VERSION` and refuses
                            anything else rather than guessing.
    "capabilities"       -> names drawn from `mojotrees.dask.CAPABILITIES`.
                            The runtime declares what it supports; nothing
                            here decides for it.
    "transport"          -> a short name for the transport, for messages.
    "multi_host"         -> whether ranks may live on different hosts.

`distributed_worker_train` returns the same six-element record shape
`_mojotrees.fit_with_metrics` already returns, so the validation history
assembles here exactly the way it does for a single-machine fit:

    (model handle or 0, flat metric values, n_rounds, best_iteration,
     best_score, stopped_early)

A rank other than the root returns 0 as the model handle. See
handoffs/connect_15_dask.md (deleted, recover with git log --all --diff-filter=D -- handoffs/connect_15_dask.md) for the exact patch requests behind those
names.

Import safety
-------------

Importing this module imports neither dask nor numpy. `dask.py` imports it
inside the functions that need it, so `import mojotrees.dask` stays as
cheap as it was. `native_runtime_status()` is safe to call anywhere,
including from a diagnostics path on a machine with no cluster: it catches
everything the probe can raise and turns it into an unavailable status with
a reason.
"""

import os
import threading
import time
import uuid
from dataclasses import dataclass

from .dask import (
    BACKEND_PROTOCOL_VERSION,
    CAPABILITIES,
    DistributedNotAvailable,
    DistributedRankError,
    DistributedTimeout,
    ModelOwnershipError,
    PartitionError,
)

__all__ = [
    "ADAPTER_VERSION",
    "CAPABILITY_ENTRY_POINTS",
    "RUNTIME_ENTRY_POINTS",
    "WORKER_ENTRY_POINTS",
    "NativeDistributedBackend",
    "NativeModelRef",
    "RuntimeStatus",
    "check_machine_list",
    "clear_runtime_provider",
    "describe_runtime",
    "gpu_exchange_status",
    "status_message",
    "native_backend",
    "native_runtime_status",
    "register_runtime_provider",
]

#: The version of the worker-side adapter protocol this module speaks. A
#: runtime that reports a different `protocol_version` is refused with a
#: message naming both numbers, because a silent mismatch here means ranks
#: interpreting the same params dict two different ways.
ADAPTER_VERSION = 1

#: The entry points a rank task calls on the worker. Listed once so the
#: "what is missing" message can name them individually rather than saying
#: that something, somewhere, is absent. None of them exists yet; the
#: request for them is in handoffs/connect_15_dask.md (deleted, recover with git log --all --diff-filter=D -- handoffs/connect_15_dask.md).
WORKER_ENTRY_POINTS = (
    "distributed_worker_open",
    "distributed_worker_train",
    "distributed_worker_close",
    "distributed_worker_cancel",
)

#: The capability records this module reads, in the order it prefers them.
#: `distributed_capability` is what bindings/distributed_bindings.mojo
#: exports today; `distributed_runtime_info` is the richer record a build
#: with a real transport would answer with instead.
CAPABILITY_ENTRY_POINTS = (
    "distributed_runtime_info",
    "distributed_capability",
)

#: Everything a fully connected build exports. Kept for callers that want
#: one list of the contract.
RUNTIME_ENTRY_POINTS = CAPABILITY_ENTRY_POINTS + WORKER_ENTRY_POINTS

#: First TCP port a rank listens on. Rank `r` takes `base + r`, which keeps
#: two ranks on one host (legal under `one_rank_per_worker=False`) on
#: distinct ports, as `TransportConfig.validate` requires. A cluster whose
#: firewall wants a different range sets the environment variable on the
#: workers and the client alike.
DEFAULT_BASE_PORT = 24601
_BASE_PORT_ENV = "MOJOTREES_DISTRIBUTED_BASE_PORT"

#: Seconds a rank waits for the world to connect, and seconds it waits for
#: any one collective. `TransportConfig` takes both in nanoseconds and a
#: `TrainingJob` carries one number, so the split is made here and stated
#: rather than hidden: a fit-wide timeout bounds a collective, and the
#: connect deadline stays short so a cluster that never assembles fails
#: while someone is still watching.
DEFAULT_CONNECT_TIMEOUT = 60.0
DEFAULT_COLLECTIVE_TIMEOUT = 600.0
_CONNECT_TIMEOUT_ENV = "MOJOTREES_DISTRIBUTED_CONNECT_TIMEOUT"

#: `MOJOTREES_DISTRIBUTED_PROVIDER` names an alternative provider as
#: `package.module:attribute`, for a build whose runtime does not live in
#: `mojotrees._mojotrees` and for bringing Task 13's runtime up under a
#: different set of names without editing this file.
_PROVIDER_ENV = "MOJOTREES_DISTRIBUTED_PROVIDER"


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
    `MOJOTREES_DISTRIBUTED_PROVIDER`, then the extension module.
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
        from . import _mojotrees
    except ImportError as exc:  # pragma: no cover - a broken install
        return None, (
            "the mojotrees extension module cannot be imported "
            f"({exc}), so there is no native distributed runtime to reach"
        )
    return _mojotrees, "the mojotrees extension module"


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
    max_world_size: int = 0
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
    probe = None
    for name in CAPABILITY_ENTRY_POINTS:
        if hasattr(provider, name):
            probe = name
            break
    missing = tuple(
        name for name in WORKER_ENTRY_POINTS if not hasattr(provider, name)
    )
    if probe is None:
        return RuntimeStatus(
            available=False,
            source=source,
            missing=CAPABILITY_ENTRY_POINTS + missing,
            reason=(
                f"{source} exports no distributed runtime at all: neither "
                f"{' nor '.join(CAPABILITY_ENTRY_POINTS)} is present, so "
                "there is nothing to ask what it can do. The transport in "
                "src/mojotrees/distributed_transport.mojo is not finished "
                "(it has no socket endpoint) and its bindings are not "
                "registered in this build. See handoffs/connect_15_dask.md (deleted, recover with git log --all --diff-filter=D -- handoffs/connect_15_dask.md) "
                "for the entry points it owes"
            ),
        )
    try:
        record = getattr(provider, probe)()
    except Exception as exc:  # the probe must not raise, ever
        return RuntimeStatus(
            available=False,
            source=source,
            missing=missing,
            reason=(
                f"{source} exports {probe}, but calling it failed: {exc!r}"
            ),
        )
    get = _record_reader(record)
    if probe == "distributed_capability":
        return _status_from_capability(get, source, missing)
    protocol = _as_int(get("protocol_version"), 0)
    if protocol != ADAPTER_VERSION:
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            reason=(
                f"the native distributed runtime speaks adapter protocol "
                f"{protocol} and this mojotrees speaks {ADAPTER_VERSION}. "
                "They disagree about the shape of the config and params "
                "dicts a rank is opened with, so the fit is refused rather "
                "than guessed at. Install a matching mojotrees on the "
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
                f"{unknown} that mojotrees.dask does not know. The known "
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
    if missing:
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            transport=transport,
            multi_host=multi_host,
            capabilities=frozenset(names),
            missing=missing,
            reason=(
                "the native distributed runtime reports itself usable, but "
                f"{source} is missing the entry points a rank task calls: "
                f"{', '.join(missing)}"
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


def _status_from_capability(get, source, missing):
    """A status from `distributed_capability()`, the record that exists.

    Fail closed, and say so in the runtime's own words. Two answers this
    deliberately does not give: `local_collective` being true is not
    availability, because one process is not a cluster and calling it one
    would be the exact misrepresentation this module exists to avoid; and
    a build that reports a transport but exports no worker entry points is
    not available either, because there would be nothing for a rank task
    to call.
    """
    protocol = _as_int(get("protocol_version"), 0)
    max_world = _as_int(get("max_world_size"), 0)
    reason = str(get("reason") or "").strip()
    if not get("multi_process"):
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            max_world_size=max_world,
            missing=missing,
            reason=(
                "the native distributed transport is not finished in this "
                "build, so mojotrees cannot train across a cluster: "
                + (
                    reason
                    or "distributed_capability() reports multi_process "
                    "false and gives no reason, which is a bindings bug"
                )
            ),
        )
    if missing:
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            max_world_size=max_world,
            missing=missing,
            reason=(
                "the native transport reports itself usable, but the "
                "worker entry points a rank task calls are missing from "
                f"{source}: {', '.join(missing)}. The transport is "
                "finished and its worker side is not wired to Python; see "
                "handoffs/connect_15_dask.md (deleted, recover with git log --all --diff-filter=D -- handoffs/connect_15_dask.md)"
            ),
        )
    names = tuple(str(name) for name in (get("capabilities") or ()))
    unknown = sorted(set(names) - set(CAPABILITIES))
    if unknown:
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            max_world_size=max_world,
            reason=(
                f"the native distributed runtime declares capabilities "
                f"{unknown} that mojotrees.dask does not know; the known "
                f"names are {sorted(CAPABILITIES)}"
            ),
        )
    if not names:
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            max_world_size=max_world,
            reason=(
                "the native distributed runtime declares no capabilities, "
                "so every fit would be refused one at a time for a "
                "different reason. A runtime that can train regression "
                "across ranks declares 'regression'"
            ),
        )
    if max_world and max_world < 2:
        return RuntimeStatus(
            available=False,
            source=source,
            protocol_version=protocol,
            max_world_size=max_world,
            capabilities=frozenset(names),
            reason=(
                f"the native distributed runtime reports a maximum world "
                f"size of {max_world}, which is not a cluster. "
                "Single-process training is what MojoTreesRegressor and "
                "the other single-machine estimators already do"
            ),
        )
    return RuntimeStatus(
        available=True,
        source=source,
        capabilities=frozenset(names),
        protocol_version=protocol,
        max_world_size=max_world,
        transport=str(get("transport") or "mojotrees-transport"),
        # The capability record has no `multi_host` field today, so this is
        # false unless a runtime adds one. It is reported, not acted on:
        # nothing here refuses a layout for being multi-host.
        multi_host=bool(get("multi_host")),
    )


def _record_reader(record):
    """A `get(name)` over either a mapping or an object with attributes, so
    a binding may answer with whichever is natural on the Mojo side."""
    if hasattr(record, "get"):

        def get(name, default=None):
            try:
                value = record.get(name)
            except Exception:
                return default
            return default if value is None else value

        return get

    def get(name, default=None):
        return getattr(record, name, default)

    return get


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def check_machine_list(machines, rank=0, job_id=0):
    """Validate a LightGBM-shaped machine list through the transport's own
    parser (`distributed_check_machine_list`) and report what it means.

    `machines` is the file's text (one `host:port` per line, blank lines
    skipped, `#` starting a comment) or a sequence of `host:port` strings.
    Returns a dict with `world_size`, `rank`, `is_root`, `addresses` in
    rank order, and `schema_digest`, the value every rank must agree on for
    the handshake to succeed. Raises `PartitionError` carrying the
    transport's message for a malformed entry, a rank outside the world, or
    a duplicate address. Validating a machine list costs nothing and needs
    no cluster, so it is worth doing long before a job runs.
    """
    if not isinstance(machines, str):
        machines = "\n".join(str(entry) for entry in machines)
    provider, source = _resolve_provider()
    if provider is None:
        raise DistributedNotAvailable(source)
    check = getattr(provider, "distributed_check_machine_list", None)
    if check is None:
        raise DistributedNotAvailable(
            f"{source} does not expose distributed_check_machine_list"
        )
    try:
        record = check(machines, int(rank), int(job_id))
    except Exception as exc:
        raise PartitionError(str(exc)) from None
    return {
        "world_size": int(record["world_size"]),
        "rank": int(record["rank"]),
        "is_root": bool(record["is_root"]),
        "addresses": [str(a) for a in record["addresses"]],
        "schema_digest": str(record["schema_digest"]),
    }


def status_message(code, transport=False):
    """The text for a native status code: a collective status
    (`distributed_status_message`, what a trainer reports when a shard's
    input disagrees with the others) or, with `transport=True`, a transport
    status (`transport_status_message`, what a session reports when it is
    the connection that broke). The two vocabularies are separate on
    purpose. Returns None when this build cannot say."""
    provider, _ = _resolve_provider()
    if provider is None:
        return None
    name = "transport_status_message" if transport else "distributed_status_message"
    hook = getattr(provider, name, None)
    if hook is None:
        return None
    try:
        return str(hook(int(code)))
    except Exception:
        return None


def gpu_exchange_status():
    """What stands between this build and distributed GPU histogram
    exchange, from src/mojotrees/distributed_gpu.mojo through the
    provider's `distributed_gpu_status()`: `{"available": bool, "gates":
    [closed gate names], "detail": str}`. Never raises: a provider without
    the query reports unavailable with the reason in `detail`."""
    provider, source = _resolve_provider()
    query = None if provider is None else getattr(
        provider, "distributed_gpu_status", None
    )
    if query is None:
        return {
            "available": False,
            "gates": [],
            "detail": (
                f"no runtime provider ({source})"
                if provider is None
                else f"{source} does not answer distributed_gpu_status()"
            ),
        }
    try:
        record = query()
        return {
            "available": bool(record["available"]),
            "gates": [str(g) for g in record["gates"]],
            "detail": str(record["detail"]),
        }
    except Exception as exc:  # a capability question never raises
        return {"available": False, "gates": [], "detail": str(exc)}


def describe_runtime():
    """One line for a diagnostics report."""
    status = native_runtime_status()
    if not status.available:
        return f"distributed: unavailable ({status.reason})"
    hosts = "multi-host" if status.multi_host else "single-host"
    gpu = gpu_exchange_status()
    gpu_text = (
        "gpu exchange available"
        if gpu["available"]
        else "gpu exchange closed by " + (", ".join(gpu["gates"]) or "no provider")
    )
    return (
        f"distributed: {status.transport} ({hosts}), capabilities "
        f"{sorted(status.capabilities)}, {gpu_text}"
    )


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


def native_backend(client=None):
    """The native distributed backend, or `DistributedNotAvailable`.

    Called by `mojotrees.dask.get_backend()` when nothing else is
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
    `bind_runtime` are optional extensions `mojotrees.dask` calls through
    `getattr`, so a third-party backend that lacks them still works.
    """

    def __init__(self, status=None, client=None, base_port=None):
        self._status = status or native_runtime_status()
        self._client = client
        self._runtime = None
        self._base_port = base_port
        self._live = {}
        self._lock = threading.Lock()

    name = "mojotrees-native"

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
        `mojotrees.dask` hands the runtime over just before `train`, and
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
                    key=f"mojotrees-rank-{launch.job_id}-{rank.rank}",
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

        `mojotrees.dask` calls this when the caller interrupts a fit. Both
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
        self._collected = False
        self.released = False
        self.owner = plan.ranks[0].worker if plan.ranks else ""

    def result(self, timeout=None):
        """The bytes of rank 0's model, once every rank has finished."""
        if self.released:
            raise ModelOwnershipError(
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
        self._collected = True
        blob = (self._record or {}).get("model") or b""
        return bytes(blob)

    def metrics(self):
        """The validation history rank 0 reported, or None.

        Read by `mojotrees.dask` after `result` and kept after `release`,
        because it is plain data rather than anything holding a worker.
        """
        return self._record

    def release(self):
        """Let go of the ranks. Idempotent, and called on both paths.

        After a finished fit this only drops the futures, which is what
        tells dask the worker may free the rank results. After a failed or
        abandoned one it also cancels, because the ranks may still be
        sitting in a collective.
        """
        if self.released:
            return
        self.released = True
        futures, self._futures = self._futures, ()
        if not self._collected:
            _cancel_world(self._client, self._job_id, futures, self._plan)
        self._backend.forget(self._job_id)


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
    # `distributed.wait` raises its own TimeoutError, which is asyncio's,
    # which is the builtin one only from Python 3.11. Both spellings are
    # caught so a timeout does not escape as an unrelated error on an
    # older interpreter.
    timeouts = tuple(
        {TimeoutError, getattr(distributed, "TimeoutError", TimeoutError)}
    )
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
            except timeouts:
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
    # By identity: `wait` hands back the future objects it was given, and
    # `==` on a Future is not a promise this needs to rely on.
    index = next(
        (i for i, other in enumerate(futures) if other is future), 0
    )
    rank = plan.ranks[index]
    try:
        future.result()
        cause = None
    except BaseException as exc:  # re-raised below as the cause
        cause = exc
    detail = f"{type(cause).__name__}: {cause}" if cause else "no detail"
    # A rank that failed inside the collective carries the native status
    # code; say what it means in the transport's own words.
    for attribute, transport in (
        ("collective_status", False),
        ("transport_status", True),
    ):
        code = getattr(cause, attribute, None)
        if isinstance(code, int):
            text = status_message(code, transport=transport)
            if text:
                detail = f"{detail}; {attribute} {code}: {text}"
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
    is the port the worker's own server listens on. A mojotrees rank needs
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
    # The transport's own parser is the judge of a machine list: duplicate
    # addresses, malformed entries, and a world that does not fit are its
    # rules, applied here once rather than restated in Python.
    check_machine_list(addresses, rank=0, job_id=0)
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
    those, so it pickles to a worker and back with no mojotrees object on
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
    eval_groups: tuple = ()
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
        # A ranker's eval set carries its own query groups, per rank, for
        # the same reason the training data does: NDCG is computed within a
        # query and a rank scores only its own rows.
        eval_groups=tuple(
            tuple(
                count
                for index in valid.plan.ranks[rank.rank].partitions
                for count in (valid.plan.partitions[index].group or ())
            )
            for valid in getattr(job, "validation", ())
        )
        if job.ranking
        else (),
        eval_metrics=tuple(getattr(job, "eval_metrics", ())),
        early_stopping_rounds=int(
            getattr(job, "early_stopping_rounds", 0) or 0
        ),
        first_metric_only=bool(getattr(job, "first_metric_only", False)),
    )


def _rank_blocks(client, job, rank):
    """Futures for the blocks one rank owns, in row order.

    Futures rather than values: dask resolves them on the worker, and the
    worker is the one already holding them, so the training rows never
    reach the client. That is the whole reason this module submits tasks
    instead of gathering partitions and calling a local fit.

    Plain lists and dicts rather than a dataclass, deliberately. Dask walks
    lists, tuples, and dicts looking for the futures in a task's arguments
    and replaces them with their values on the worker; it does not walk
    into an arbitrary object, so a future hidden inside one would arrive on
    the worker as a future the rank task would then have to resolve
    itself, from inside the task that is holding the thread it needs.
    """
    plan = job.plan
    parts = [plan.partitions[index] for index in rank.partitions]
    return {
        "features": _futures_for(client, [part.key for part in parts]),
        "labels": _futures_for(client, [part.label_key for part in parts]),
        "weights": _futures_for(
            client, [part.weight_key for part in parts]
        ),
        "validation": [
            _validation_blocks(client, valid, rank)
            for valid in getattr(job, "validation", ())
        ],
    }


def _validation_blocks(client, valid, rank):
    """The eval set's blocks for this rank.

    A rank scores the validation rows that live where its training rows do,
    which is why `mojotrees.dask` requires an eval collection with the same
    rank layout: the alternative is shipping validation rows between
    workers every round.
    """
    plan = valid.plan
    owned = plan.ranks[rank.rank]
    parts = [plan.partitions[index] for index in owned.partitions]
    return {
        "features": _futures_for(client, [part.key for part in parts]),
        "labels": _futures_for(client, [part.label_key for part in parts]),
        "weights": _futures_for(
            client, [part.weight_key for part in parts]
        ),
    }


def _futures_for(client, keys):
    """Futures for scheduler keys, or `()` when the collection was absent.

    A `None` key means the caller passed no labels or no weights for that
    partition; a partly-filled list is a bug in the runtime's metadata
    rather than something to paper over here.
    """
    keys = list(keys)
    if not keys or all(key is None for key in keys):
        return []
    if any(key is None for key in keys):
        raise PartitionError(
            "some partitions carry a data key and some do not, so the rank "
            "cannot be given a complete block of rows. This is a runtime "
            "metadata bug: report the collection that produced it"
        )
    distributed = _import_distributed()
    return [distributed.Future(key, client=client) for key in keys]


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
    status = native_runtime_status(refresh=True)
    if not status.available:
        raise DistributedNotAvailable(
            f"rank {rank} cannot train: {status.reason}. The client's "
            "mojotrees has a distributed runtime and this worker's does "
            "not, or they are different builds"
        )
    provider, _ = _resolve_provider()
    from . import _arrays

    features = _concat_blocks(blocks.get("features"), "X", rank)
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
    labels = _concat_blocks(blocks.get("labels"), "y", rank)
    if labels is None:
        raise DistributedRankError(
            f"rank {rank} was given no labels; distributed fit needs y "
            "partitioned exactly as X is"
        )
    if call.label_classes:
        yb = _encode_labels_through(labels, call.label_classes, n_rows, rank)
    else:
        yb = _arrays.check_target(labels, n_rows)
    weights = _concat_blocks(blocks.get("weights"), "sample_weight", rank)
    wb = None if weights is None else _arrays.check_sample_weight(
        weights, n_rows
    )
    valid_buffers = []
    valid_specs = []
    for index, valid in enumerate(blocks.get("validation") or ()):
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
    features = _concat_blocks(
        valid.get("features"), f"eval_set[{index}] X", rank
    )
    labels = _concat_blocks(
        valid.get("labels"), f"eval_set[{index}] y", rank
    )
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
        valid.get("weights"), f"eval_set[{index}] weight", rank
    )
    wb = None if weights is None else _arrays.check_sample_weight(
        weights, n_rows
    )
    spec = {
        "X_addr": _arrays.addr(Xb),
        "y_addr": _arrays.addr(yb),
        "weight_addr": 0 if wb is None else _arrays.addr(wb),
        "n_rows": n_rows,
        "name": call.eval_names[index]
        if index < len(call.eval_names)
        else f"valid_{index}",
    }
    if call.ranking:
        group = list(
            call.eval_groups[index]
            if index < len(call.eval_groups)
            else ()
        )
        if sum(group) != n_rows:
            raise DistributedRankError(
                f"rank {rank} holds {n_rows} rows of eval set {index} and "
                f"a query group summing to {sum(group)}"
            )
        spec["group"] = group
        spec["n_groups"] = len(group)
    return spec, (Xb, yb, wb)


def _payload_from_record(record, call, rank, provider):
    """The rank's return value: model bytes on the root, history on the
    root, and nothing on anyone else.

    The record shape is the one `_mojotrees.fit_with_metrics` already
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
        from . import _mojotrees

        save = (
            _mojotrees.save_multiclass if multiclass else _mojotrees.save
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


def _cancel_sessions(job_id, dask_worker=None):
    """Cancel this worker's session for a job. Runs under `client.run`.

    The job id is positional and `dask_worker` is the parameter dask fills
    in by keyword when a run function asks for it, which is the order
    `client.run(_cancel_sessions, job_id, workers=...)` produces.
    Cancelling an unknown job is not an error: a worker that already
    finished its rank has nothing to cancel.
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
    parts = [block for block in (blocks or ()) if block is not None]
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
