"""Dask adapter contracts. Optional, import safe, and not yet a way to
train a model.

    import mojotrees.dask as mbd   # works with or without dask installed

This module is the client side of distributed training: it takes a Dask
collection apart into the metadata a distributed trainer needs, checks that
metadata against what the algorithm in docs/distributed.md actually
requires, assigns ranks to workers, hands the result to a *backend*, and
turns the model that comes back into an ordinary fitted mojotrees
estimator. It does not implement a transport, and it does not train.

Where the training actually happens
-----------------------------------

`mojotrees/_dask_runtime.py` is the one backend this package ships. It
finds the native distributed runtime in the extension module, launches one
rank per worker with the partitions left where dask put them, and hands
rank 0's model bytes back through the protocol below. `get_backend()` falls
back to it when nothing else is registered, so a build whose extension
exports the runtime trains through Mojo with no registration step, and a
build whose extension does not says exactly which entry points are missing.

Today every published build is the second kind: the transport in
src/mojotrees/distributed_transport.mojo has no socket endpoint, so
`DaskMojoTreesRegressor(...).fit(...)` raises `DistributedNotAvailable`
before it touches the cluster, naming what is absent.

Nothing in this module ever trains, in any state. It validates, plans,
negotiates, and reassembles; when there is no backend it refuses. It does
not fit on the client, and it does not fit rank by rank and average, which
would be a different algorithm under this one's name.

What has been exercised is what the tests exercise: metadata validation,
the rank plan, capability negotiation, model ownership, and partition-local
prediction, against a fake backend and a fake runtime. `DaskRuntime` and
the launch path in `_dask_runtime` have never been run against a live
cluster. Read them as proposals.

Optional, and import safe
-------------------------

`import mojotrees.dask` does not import dask, and neither does anything
this module does until a cluster is actually involved. `dask_available()`
answers with `importlib.util.find_spec`, `is_dask_collection()` asks the
object rather than the library, and `_import_dask()` is called from exactly
two places: `DaskRuntime.client` and `DaskRuntime.map_partitions`. Every
dataclass, validator, the rank planner, the backend registry, and
partition-local prediction work with dask absent. That is what makes this
module safe to expose from the package without adding dask to the install.

There is one import this module cannot make lazy: the three Dask
estimators subclass the three single-machine ones, and a base class is
needed when the class statement runs. So `mojotrees.dask` depends on
`mojotrees` being fully initialized, and `mojotrees/__init__.py` must not
import this module at the top of its own body, where the estimators do not
exist yet. It is not a cycle to hide, it is an ordering constraint, and the
way to expose `mojotrees.dask` without tripping over it is a module-level
`__getattr__` (PEP 562) in the package, which resolves the submodule on
first attribute access and costs nothing on `import mojotrees`. The exact
code is in `mojotrees/_public_api_plan.py` and
handoffs/integration_06_python_api.md.

The backend protocol, version 0
-------------------------------

The seam is deliberately narrow, because the thing on the other side of it
is being built at the same time. A backend is any object with:

    backend.name          -> str
    backend.capabilities  -> a container of capability names (CAPABILITIES)
    backend.train(job)    -> a model reference

`job` is a `TrainingJob`: a `WorldPlan` (ranks, workers, row offsets, the
column schema), the objective name, the estimator's hyperparameters under
their LightGBM spellings, the global class labels for a classifier, the
validation plans and early stopping settings, and a timeout. A backend
reads it and never mutates it; every field is frozen.

Three members are optional, called through `getattr` so a backend written
against version 0 still works:

    backend.bind_runtime(runtime)   before train, to reach the cluster
    backend.cancel(job)             when the caller interrupts a fit
    ref.metrics()                   the validation history rank 0 reported

The model reference is any object with:

    ref.owner             -> str, the worker address holding the model
    ref.result(timeout)   -> bytes, mojotrees's versioned text format
    ref.release()         -> None, idempotent

`BytesModelRef` implements it for a backend that already has the bytes in
hand. `register_backend` installs a backend; `MOJOTREES_DASK_BACKEND`
names one as `package.module:attribute` for a process that cannot call
`register_backend` itself.

What the model reference contract buys is stated in `take_model_bytes`:
the adapter gathers once, releases in a `finally`, and never keeps a live
future on the estimator, so a fitted estimator pickles and outlives the
cluster it was trained on.

Partition rules, and why each one is a rule
-------------------------------------------

- **Row counts must be known.** A partition of unknown length cannot be
  given a row offset, and section 3 of docs/distributed.md needs offsets
  to keep the global row order. Persist or compute lengths first.
- **Partitions must be non-empty.** A rank with no rows contributes
  nothing to the histogram all-reduce and still has to answer every
  collective, and the base-score reduction divides by a weight sum.
- **The partition order is the row order.** Rank `r` owns a contiguous
  block of the global row order, so ranks are numbered by partition index,
  never by worker address.
- **One rank per worker needs contiguous partitions.** Concatenating a
  worker's partitions is only order preserving when they are adjacent in
  the global order; when they are not, this module says so instead of
  silently permuting rows. `one_rank_per_worker=False` gives each
  partition its own rank instead.
- **The column schema must agree everywhere.** Column count, column names,
  and the category tables of every categorical column, in the same order,
  because bin 7 of feature 3 has to mean the same thing on every rank.
- **A query may not straddle a partition boundary.** LambdaRank gradients
  are computed within a query, and a rank only sees its own rows.
- **An eval set lands rank for rank on the training data.** A rank scores
  the validation rows on its own worker and the losses are reduced with
  the gradients, so the two collections need the same number of ranks and
  the same worker per rank. Anything else would ship validation rows
  across the cluster every round.

See also
--------

docs/distributed.md for the algorithm and its refusal list,
handoffs/task17_dask.md for the original contract, and
handoffs/connect_15_dask.md for the native entry points the runtime still
owes and the state of each connection made here.
"""

import hashlib
import importlib
import os
from dataclasses import dataclass, field

# The three base classes, and nothing else: this is the whole of what has
# to exist before this module can be imported (see "Optional, and import
# safe" above). Package functions needed only at call time are imported
# inside the functions that need them, so the ordering constraint stays as
# small as the class statements make it.
from . import MojoTreesClassifier, MojoTreesRanker, MojoTreesRegressor

__all__ = [
    "DaskMojoTreesClassifier",
    "DaskMojoTreesRanker",
    "DaskMojoTreesRegressor",
    "BACKEND_PROTOCOL_VERSION",
    "CAPABILITIES",
    "BytesModelRef",
    "CategoricalSchema",
    "DaskRuntime",
    "DistributedNotAvailable",
    "DistributedRankError",
    "DistributedTimeout",
    "ModelOwnershipError",
    "PartitionError",
    "PartitionMeta",
    "RankAssignment",
    "SchemaError",
    "TrainingJob",
    "UnsupportedByBackend",
    "ValidationPlan",
    "WorldPlan",
    "backend_registered",
    "check_machine_list",
    "clear_backend",
    "clear_model_cache",
    "dask_available",
    "distributed_gpu_status",
    "distributed_status",
    "get_backend",
    "is_dask_collection",
    "merge_categorical_schema",
    "partition_group",
    "plan_world",
    "register_backend",
    "require_capabilities",
    "take_model_bytes",
    "validate_partitions",
    "validate_query_partitioning",
]

#: The version of the backend protocol this module speaks. A backend that
#: reads `TrainingJob.protocol_version` and does not recognize it should
#: refuse rather than guess; the fields are documented at the top of this
#: module and nowhere else yet.
BACKEND_PROTOCOL_VERSION = 0

#: Every capability name a backend may declare. The adapter asks for the
#: ones a given fit needs and refuses the fit when the backend does not
#: declare them, so a backend that grows a feature announces it here rather
#: than in this module. Names not in this set are rejected at registration
#: time, which is what keeps a typo from reading as a missing capability.
CAPABILITIES = frozenset(
    {
        "regression",  # single-output regression objectives
        "binary",  # two-class classification
        "multiclass",  # softmax, one tree per class per round
        "ranking",  # lambdarank, needs the query partition rules
        "categorical",  # categorical splits over reduced histograms
        "weights",  # per-row sample weights
        "quantile_l1",  # leaf renewal, which does not all-reduce
        "bagging",  # row subsampling from a global draw
        "goss",  # gradient-based one-side sampling, a global order
        "class_weight",  # per-class weights folded in on the workers
        "feature_fraction",  # column subsampling
        "early_stopping",  # validation loss reduced each round
        "custom_objective",  # a callable objective shipped to the workers
        "gpu",  # distributed training on the GPU backend
    }
)

#: Constructor arguments that describe how to reach the cluster rather than
#: how to train, and so do not travel in `TrainingJob.params`.
_NON_TRAINING_PARAMS = frozenset({"client", "one_rank_per_worker"})


class DistributedNotAvailable(RuntimeError):
    """No distributed backend is registered, or the one named cannot be
    imported. Raised before any cluster work is attempted."""


class PartitionError(ValueError):
    """Partition metadata does not satisfy the layout the distributed
    algorithm needs. The message says which rule and what to do."""


class SchemaError(ValueError):
    """Partitions disagree about their columns, or a categorical column
    has categories the client cannot see."""


class UnsupportedByBackend(RuntimeError):
    """The fit needs a capability the registered backend does not
    declare."""


class ModelOwnershipError(RuntimeError):
    """A model reference did not behave the way the protocol requires: no
    bytes, the wrong type, or a use after release."""


class DistributedRankError(RuntimeError):
    """One rank of a running fit failed, so the whole fit was cancelled.

    A worker that dies mid-fit, a rank whose build has no native runtime,
    and a rank whose rows do not match the plan all arrive here. There is
    no partial model to recover: every other rank is waiting inside a
    collective the dead one will never answer, and the client cancels them
    rather than leaving them there.
    """


class DistributedTimeout(DistributedRankError):
    """A fit ran past `timeout` seconds and every rank was cancelled."""


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def dask_available():
    """Whether `dask` and `distributed` can be imported, without importing
    either of them if they are already known to be absent.

    Importing dask is not free, so nothing in this module does it until a
    cluster is actually involved: the dataclasses, the validators, the rank
    planner, and partition-local prediction all work without it.
    """
    from importlib.util import find_spec

    try:
        return (
            find_spec("dask") is not None
            and find_spec("distributed") is not None
        )
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def _import_dask():
    """`(dask, dask.dataframe or None, distributed)`, or an ImportError
    naming what to install."""
    try:
        import dask
        import distributed
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "mojotrees.dask needs dask and distributed, which are optional "
            "dependencies: pip install 'dask[distributed]'. Importing "
            "mojotrees.dask itself does not require them; reaching a "
            "cluster does."
        ) from exc
    try:  # dask.dataframe is its own optional extra
        import dask.dataframe as dask_dataframe
    except ImportError:  # pragma: no cover - depends on the install
        dask_dataframe = None
    return dask, dask_dataframe, distributed


def is_dask_collection(obj):
    """Whether `obj` is a Dask collection, decided the way dask itself
    decides it and without importing dask.

    `__dask_graph__` is the collection protocol, so this is true for
    arrays, dataframes, series, bags, and anything else that implements it,
    and false for numpy arrays and pandas frames.
    """
    return hasattr(obj, "__dask_graph__")


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoricalSchema:
    """The categorical columns of a training set, as every partition must
    agree they are.

    `columns` maps a feature index to its category table, in the order the
    codes number from 0, exactly as `_arrays.frame_categories` reports it
    for a pandas frame. A column whose categories are `None` is a dask
    "unknown categories" column: dask knows the dtype is categorical but
    not what the categories are, and encoding rows against categories that
    have not been read yet would give each partition its own numbering.
    `merge_categorical_schema` refuses those.
    """

    columns: tuple = ()

    @property
    def indices(self):
        """Feature indices of the categorical columns, ascending."""
        return tuple(index for index, _ in self.columns)

    def encoders(self):
        """`{feature index: [category, ...]}`, the mapping a fitted
        estimator uses to encode a pandas frame at prediction time.

        A model read back from bytes carries which features are
        categorical but not the labels behind their codes, so a distributed
        fit installs this on the estimator itself. Without it, predicting
        on a frame of category dtype would encode through the frame's own
        categories, which may be ordered differently.
        """
        return {index: list(cats) for index, cats in self.columns}

    def __bool__(self):
        return bool(self.columns)


def merge_categorical_schema(partitions):
    """One `CategoricalSchema` for a whole collection, or a `SchemaError`
    saying how the partitions disagree.

    Every partition must declare the same categorical columns with the same
    category tables in the same order. Two partitions that saw different
    subsets of a column's categories are the common cause; in dask that is
    what `.categorize()` on the whole frame exists to fix, and doing it per
    partition is exactly the failure this catches.
    """
    schema = None
    source = None
    for part in partitions:
        columns = tuple(part.categories)
        for index, cats in columns:
            if cats is None:
                raise SchemaError(
                    f"feature {index} of partition {part.index} is a "
                    "categorical column with unknown categories; call "
                    ".categorize() on the collection (or cast the column to "
                    "integer codes) so every partition numbers the "
                    "categories the same way"
                )
        if schema is None:
            schema, source = columns, part.index
            continue
        if columns != schema:
            raise SchemaError(
                f"partitions {source} and {part.index} disagree about their "
                f"categorical columns ({schema!r} against {columns!r}); "
                "every rank has to number a category the same way or a "
                "reduced histogram bin means two different things"
            )
    return CategoricalSchema(columns=schema or ())


# ---------------------------------------------------------------------------
# Partition metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PartitionMeta:
    """What the client knows about one partition of a training collection.

    This is the whole interface between "some Dask collection" and
    everything else in this module, which is why the validators and the
    rank planner take these rather than collections: they are testable
    without a cluster, and a runtime other than dask can produce them.

    - `index` is the partition's position in the global row order, and
      therefore what ranks are numbered by.
    - `worker` is the address the partition lives on, as
      `client.who_has()` reports it.
    - `n_rows` is `None` when the collection has not been persisted or
      computed and its lengths are unknown; that is an error, not a
      default.
    - `feature_names` is `None` for an array and the column names for a
      dataframe.
    - `categories` is `((feature index, categories or None), ...)`, in
      ascending index order.
    - `label_values` is the distinct label values this partition saw. It is
      advisory: nothing populates it and nothing reads it. A classifier's
      global class list is the `classes` argument of `fit`, because a scan
      that missed a rare class would give the ranks two different label
      encodings and no error, and a metadata field cannot promise it saw
      every partition. A runtime that can fill this in honestly may, and a
      backend may log it; neither is a way to skip `classes`.
    - `first_query` and `last_query` are the query ids at the two ends of
      the partition, and `group` its per-query row counts. Ranking only.
    - `key`, `label_key`, and `weight_key` locate the rows themselves: the
      scheduler keys of this partition's feature block, labels, and
      weights. They travel to the backend on `WorldPlan.partitions`,
      because a rank plan that says where the rows are without saying what
      they are called is not enough to fetch them.
    """

    index: int
    worker: str
    n_rows: int = None
    n_features: int = 0
    feature_names: tuple = None
    categories: tuple = ()
    label_values: tuple = ()
    has_weight: bool = False
    first_query: object = None
    last_query: object = None
    group: tuple = None
    key: object = None
    label_key: object = None
    weight_key: object = None


def validate_partitions(partitions):
    """Partition metadata in global row order, checked against the layout
    rules at the top of this module.

    Returns a tuple sorted by `index`. Raises `PartitionError` for a layout
    the algorithm cannot use and `SchemaError` for columns that disagree.
    """
    parts = tuple(sorted(partitions, key=lambda p: p.index))
    if not parts:
        raise PartitionError(
            "the training collection has no partitions; there is nothing to "
            "distribute"
        )
    seen = {}
    for part in parts:
        if part.index in seen:
            raise PartitionError(
                f"two partitions both claim index {part.index}; partition "
                "indices are the global row order and have to be distinct"
            )
        seen[part.index] = part
    expected = list(range(len(parts)))
    if [part.index for part in parts] != expected:
        raise PartitionError(
            f"partition indices {[p.index for p in parts]} are not "
            f"0..{len(parts) - 1}; the row order is the partition order, so "
            "a gap means rows nobody owns"
        )
    for part in parts:
        if not part.worker:
            raise PartitionError(
                f"partition {part.index} has no worker address; persist the "
                "collection (client.persist) before fitting so every "
                "partition has a home worker"
            )
        if part.n_rows is None:
            raise PartitionError(
                f"partition {part.index} has an unknown row count; a rank "
                "needs a row offset in the global order, so compute the "
                "partition lengths first (for a dataframe, "
                "collection.map_partitions(len).compute())"
            )
        if part.n_rows <= 0:
            raise PartitionError(
                f"partition {part.index} is empty; an empty rank still has "
                "to answer every collective and contributes nothing to the "
                "histogram reduction. Repartition to drop empty partitions"
            )
    n_features = parts[0].n_features
    if n_features <= 0:
        raise PartitionError(
            f"partition {parts[0].index} reports {n_features} features"
        )
    for part in parts[1:]:
        if part.n_features != n_features:
            raise SchemaError(
                f"partition {parts[0].index} has {n_features} features and "
                f"partition {part.index} has {part.n_features}"
            )
    names = parts[0].feature_names
    for part in parts[1:]:
        if part.feature_names != names:
            raise SchemaError(
                f"partitions {parts[0].index} and {part.index} disagree "
                f"about their column names ({names!r} against "
                f"{part.feature_names!r})"
            )
    if names is not None and len(names) != n_features:
        raise SchemaError(
            f"{len(names)} column names for {n_features} features"
        )
    merge_categorical_schema(parts)
    return parts


def partition_group(query_ids):
    """Per-query row counts for one partition, from its query id column.

    Delegates to `mojotrees.group_from_query_ids`, so the rule that a
    query's rows must be consecutive is enforced in one place. What this
    adds is the partition-level rule: see `validate_query_partitioning`.
    """
    from . import group_from_query_ids

    return tuple(group_from_query_ids(query_ids))


def validate_query_partitioning(partitions):
    """Check the ranking rules on already-validated partitions.

    Two rules beyond the shared ones:

    - Every partition carries its own `group`, summing to its row count,
      with no empty query. A rank computes lambdarank pairs from the group
      boundaries of its own rows, so a missing or inconsistent group is not
      recoverable on the worker.
    - No query straddles a partition boundary. Pairs are formed within a
      query, and a rank only sees its own rows, so half a query on each of
      two ranks silently changes every gradient that query contributes.
      Shuffle or set the index on the query column so each query lands
      whole.

    Returns the total number of queries.
    """
    total_queries = 0
    previous = None
    for part in partitions:
        if part.group is None:
            raise PartitionError(
                f"partition {part.index} has no query group; a ranker needs "
                "the row count of each query in each partition, in row order"
            )
        group = tuple(int(count) for count in part.group)
        if any(count <= 0 for count in group):
            raise PartitionError(
                f"partition {part.index} has a query with no rows"
            )
        if sum(group) != part.n_rows:
            raise PartitionError(
                f"partition {part.index} has {part.n_rows} rows but its "
                f"query group sums to {sum(group)}"
            )
        if (
            previous is not None
            and previous.last_query is not None
            and part.first_query is not None
            and previous.last_query == part.first_query
        ):
            raise PartitionError(
                f"query {part.first_query!r} spans partitions "
                f"{previous.index} and {part.index}; a query's rows have to "
                "stay on one rank, because lambdarank pairs rows within a "
                "query and a rank sees only its own rows. Shuffle on the "
                "query column so each query lands in one partition"
            )
        total_queries += len(group)
        previous = part
    return total_queries


# ---------------------------------------------------------------------------
# Ranks and the world
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankAssignment:
    """One rank: which worker runs it, which partitions it owns, and where
    its rows sit in the global order."""

    rank: int
    worker: str
    partitions: tuple
    row_offset: int
    n_rows: int


@dataclass(frozen=True)
class WorldPlan:
    """The full rank layout handed to a backend.

    `ranks` is in rank order, which is row order: rank `r` owns the rows
    `[row_offset, row_offset + n_rows)` of the global order, and
    concatenating the ranks in order reproduces the dataset. Section 3 of
    docs/distributed.md explains why that is load bearing rather than
    cosmetic.

    `partitions` is the validated metadata the plan was built from, in
    partition order, so `plan.partitions[i]` is the partition a rank names
    in its own `partitions` tuple. That is how a backend gets from a rank
    to the keys of the data it has to fetch; the plan alone says where the
    rows live and not what they are called.
    """

    ranks: tuple
    n_features: int
    feature_names: tuple = None
    schema: CategoricalSchema = CategoricalSchema()
    partitions: tuple = ()

    @property
    def world_size(self):
        return len(self.ranks)

    @property
    def n_rows(self):
        return sum(rank.n_rows for rank in self.ranks)

    @property
    def workers(self):
        """Worker addresses in rank order, with repeats when more than one
        rank shares a worker."""
        return tuple(rank.worker for rank in self.ranks)

    @property
    def addresses_unique(self):
        """Whether every rank has its own worker address.

        False when `one_rank_per_worker=False` put two ranks on one worker.
        A transport that addresses peers by worker address rather than by
        rank has to check this; that is why it is on the plan rather than
        an assumption.
        """
        return len(set(self.workers)) == len(self.ranks)


def plan_world(partitions, one_rank_per_worker=True):
    """A `WorldPlan` from validated partition metadata.

    With `one_rank_per_worker` (the default), each worker runs one rank
    over all of its partitions concatenated, which is the layout a real
    cluster wants: one process, one histogram, one participant in each
    collective. Concatenation only preserves the row order when a worker's
    partitions are adjacent in it, so a worker holding partitions 0 and 3
    is an error with the fix in the message rather than a quiet permutation
    of the training rows.

    With `one_rank_per_worker=False`, every partition is its own rank, two
    of which may then share an address. That is the layout for a
    single-machine cluster and for tests; `WorldPlan.addresses_unique`
    reports it.
    """
    parts = validate_partitions(partitions)
    schema = merge_categorical_schema(parts)
    groups = []
    if one_rank_per_worker:
        by_worker = {}
        for part in parts:
            by_worker.setdefault(part.worker, []).append(part)
        for worker, owned in by_worker.items():
            indices = [part.index for part in owned]
            span = range(indices[0], indices[-1] + 1)
            if len(indices) != len(span):
                missing = sorted(set(span) - set(indices))
                raise PartitionError(
                    f"worker {worker} holds partitions {indices}, which are "
                    f"not adjacent in the row order (partitions {missing} "
                    "sit between them on other workers). Concatenating them "
                    "into one rank would reorder the training rows. "
                    "Rebalance the collection, or pass "
                    "one_rank_per_worker=False to give every partition its "
                    "own rank"
                )
            groups.append(owned)
        groups.sort(key=lambda owned: owned[0].index)
    else:
        groups = [[part] for part in parts]
    ranks = []
    offset = 0
    for rank, owned in enumerate(groups):
        n_rows = sum(part.n_rows for part in owned)
        ranks.append(
            RankAssignment(
                rank=rank,
                worker=owned[0].worker,
                partitions=tuple(part.index for part in owned),
                row_offset=offset,
                n_rows=n_rows,
            )
        )
        offset += n_rows
    return WorldPlan(
        ranks=tuple(ranks),
        n_features=parts[0].n_features,
        feature_names=parts[0].feature_names,
        schema=schema,
        partitions=parts,
    )


# ---------------------------------------------------------------------------
# The backend seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationPlan:
    """One eval set, planned onto the same ranks as the training data.

    `plan` is a full `WorldPlan` over the validation collection, and its
    rank `r` is required to sit on the same worker as training rank `r`.
    That is what lets a rank score its own validation rows and contribute
    them to the reduction without any validation row moving between
    workers, and it is checked on the client (`_plan_validation`) rather
    than discovered by a backend halfway through round one.
    """

    name: str
    plan: WorldPlan


@dataclass(frozen=True)
class TrainingJob:
    """Everything a backend needs to run one distributed fit.

    Frozen, and every field is either a scalar, a string, or another frozen
    dataclass, so a backend can hold on to it, ship it, or log it without
    the client and the workers disagreeing about it later. `params` is a
    plain dict by necessity and is not to be mutated.

    `validation`, `eval_metrics`, `early_stopping_rounds`, and
    `first_metric_only` are empty or zero for a fit with no eval set, which
    is what a version 0 backend written before they existed sees. A backend
    that declares the `early_stopping` capability reduces each eval set's
    metric across ranks every round and reports the history back through
    `ref.metrics()`.
    """

    plan: WorldPlan
    objective: str
    params: dict = field(default_factory=dict)
    label_classes: tuple = None
    ranking: bool = False
    timeout: float = None
    validation: tuple = ()
    eval_metrics: tuple = ()
    early_stopping_rounds: int = 0
    first_metric_only: bool = False
    protocol_version: int = BACKEND_PROTOCOL_VERSION


class BytesModelRef:
    """A model reference for a backend that already holds the model bytes.

    The protocol exists so a backend can return a handle to a model still
    sitting on a worker and let the client decide when to pull it. A
    backend that has already gathered the bytes wraps them in this instead
    of inventing its own type.
    """

    def __init__(self, blob, owner="<local>"):
        self._blob = blob
        self.owner = owner
        self.released = False

    def result(self, timeout=None):
        if self.released:
            raise ModelOwnershipError(
                "this model reference was already released"
            )
        return self._blob

    def release(self):
        self.released = True
        self._blob = None


def take_model_bytes(ref, timeout=None):
    """Gather a model reference exactly once and release it, always.

    The ownership rule this enforces, from the client side: the adapter
    owns the reference the backend returned, takes the bytes from it once,
    and releases it in a `finally` whether or not that succeeded. Nothing
    downstream holds a live future, which is what lets a fitted estimator
    pickle and outlive its cluster. A backend is free to make `release`
    the thing that frees the worker-side model; it must tolerate being
    called once, and only once, per reference.
    """
    try:
        blob = ref.result(timeout) if timeout is not None else ref.result()
    finally:
        release = getattr(ref, "release", None)
        if release is not None:
            release()
    if isinstance(blob, bytearray):
        blob = bytes(blob)
    if not isinstance(blob, bytes):
        raise ModelOwnershipError(
            f"the backend's model reference returned {type(blob).__name__}, "
            "not the bytes of a saved mojotrees model"
        )
    if not blob:
        raise ModelOwnershipError(
            "the backend's model reference returned no bytes"
        )
    return blob


def _metric_names(eval_metric):
    """`eval_metric` as a tuple of names, in the order it was given.

    Names rather than resolved metrics: which metric an objective defaults
    to, and whether a metric is even compatible with it, belongs to the
    objective registry in Mojo and is decided on the ranks. The client
    passes the names through and lays the reported history out under them.
    """
    if eval_metric is None:
        return ()
    if isinstance(eval_metric, str):
        return (eval_metric,)
    if callable(eval_metric):
        raise ValueError(
            "a custom eval_metric is a Python callable, and a distributed "
            "fit would have to call it once per rank per round from inside "
            "a collective. Name a built-in metric, or fit on one machine"
        )
    names = tuple(str(name) for name in eval_metric)
    if len(set(names)) != len(names):
        raise ValueError(f"eval_metric has repeats: {list(names)!r}")
    return names


def _reader(record):
    """A `get(name)` over a mapping or an object, so a backend may report
    its validation history as whichever is natural for it."""
    if hasattr(record, "get"):
        return record.get
    return lambda name, default=None: getattr(record, name, default)


_BACKEND = None


def register_backend(backend, replace=False):
    """Install the distributed backend this process trains through.

    Checked here rather than at fit time, so a backend that is missing a
    protocol member says so when it is installed: `name`, `capabilities`
    drawn from `CAPABILITIES`, and a callable `train`.
    """
    global _BACKEND
    if _BACKEND is not None and not replace:
        raise DistributedNotAvailable(
            f"a distributed backend ({getattr(_BACKEND, 'name', _BACKEND)!r})"
            " is already registered; pass replace=True to swap it"
        )
    _check_backend(backend)
    _BACKEND = backend
    return backend


def clear_backend():
    """Forget the registered backend. Mostly for tests."""
    global _BACKEND
    _BACKEND = None


def backend_registered():
    """Whether a backend is installed, without importing or resolving
    one."""
    return _BACKEND is not None


def _check_backend(backend):
    for member in ("name", "capabilities", "train"):
        if not hasattr(backend, member):
            raise DistributedNotAvailable(
                f"{backend!r} is not a mojotrees distributed backend: it has "
                f"no {member!r}. The protocol is name, capabilities, and "
                "train(job); see mojotrees/dask.py"
            )
    if not callable(backend.train):
        raise DistributedNotAvailable(
            f"{backend!r}.train is not callable"
        )
    unknown = set(backend.capabilities) - CAPABILITIES
    if unknown:
        raise DistributedNotAvailable(
            f"{backend!r} declares unknown capabilities "
            f"{sorted(unknown)}; the known ones are {sorted(CAPABILITIES)}"
        )
    return backend


def get_backend():
    """The distributed backend, or a `DistributedNotAvailable` explaining
    precisely why there is none.

    Resolution order: whatever `register_backend` installed, then the
    `MOJOTREES_DASK_BACKEND` environment variable as
    `package.module:attribute`, then the native runtime this package ships
    (`mojotrees/_dask_runtime.py`), which is used with no registration step
    on a build whose extension module exports it.

    The native probe is the last step rather than the first so that a
    registered backend always wins, which is what makes the fake backend in
    the tests and a third-party transport possible on a build that has its
    own.
    """
    if _BACKEND is not None:
        return _BACKEND
    spec = os.environ.get("MOJOTREES_DASK_BACKEND", "").strip()
    if spec:
        return _resolve_backend_spec(spec)
    from . import _dask_runtime

    status = _dask_runtime.native_runtime_status()
    if status.available:
        return _check_backend(_dask_runtime.native_backend())
    raise DistributedNotAvailable(
        "mojotrees cannot train across a Dask cluster on this "
        f"installation: {status.reason}. No backend is registered either. "
        "Register one with mojotrees.dask.register_backend(...) or name "
        "one in MOJOTREES_DASK_BACKEND as 'package.module:attribute'. "
        "Train on one machine with MojoTreesRegressor, "
        "MojoTreesClassifier, or MojoTreesRanker meanwhile"
    )


def check_machine_list(machines, rank=0, job_id=0):
    """Validate a LightGBM-shaped machine list (`host:port` per line, or a
    sequence of them) through the native transport and report
    `world_size`, `rank`, `is_root`, `addresses`, and `schema_digest`.
    Raises `PartitionError` with the transport's message. Needs no cluster
    and no dask; see `_dask_runtime.check_machine_list`."""
    from . import _dask_runtime

    return _dask_runtime.check_machine_list(machines, rank=rank, job_id=job_id)


def distributed_gpu_status():
    """What stands between this build and distributed GPU histogram
    exchange: `{"available", "gates", "detail"}` from
    src/mojotrees/distributed_gpu.mojo. Never raises and never imports
    dask."""
    from . import _dask_runtime

    return _dask_runtime.gpu_exchange_status()


def distributed_status():
    """What the native distributed runtime reports about itself.

    A `RuntimeStatus`: whether a cluster fit is possible, the reason when
    it is not, the capabilities the runtime declares, and the transport's
    name. Safe to call anywhere, including with no cluster and no dask
    installed; it never raises and never imports dask.
    """
    from . import _dask_runtime

    return _dask_runtime.native_runtime_status()


def _resolve_backend_spec(spec):
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise DistributedNotAvailable(
            f"MOJOTREES_DASK_BACKEND is {spec!r}, which is not "
            "'package.module:attribute'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DistributedNotAvailable(
            f"MOJOTREES_DASK_BACKEND names {spec!r}, but {module_name!r} "
            f"cannot be imported: {exc}"
        ) from exc
    try:
        backend = getattr(module, attribute)
    except AttributeError as exc:
        raise DistributedNotAvailable(
            f"MOJOTREES_DASK_BACKEND names {spec!r}, but {module_name!r} has "
            f"no {attribute!r}"
        ) from exc
    if isinstance(backend, type):
        # A class is a factory, not a backend: an unbound `train` would take
        # the job as its `self`.
        backend = backend()
    return _check_backend(backend)


def require_capabilities(backend, needed):
    """Refuse a fit the backend has not said it can run.

    The refusal list in section 9 of docs/distributed.md belongs to the
    trainer, not to this module, so it is not copied here: the backend
    declares what it supports and this compares. A backend that grows
    multiclass support announces it by adding a capability, with no change
    on the client.
    """
    missing = sorted(set(needed) - set(backend.capabilities))
    if missing:
        raise UnsupportedByBackend(
            f"the {getattr(backend, 'name', backend)!r} distributed backend "
            f"does not support {', '.join(missing)}. It declares "
            f"{sorted(backend.capabilities)}. Train this model on one "
            "machine, or drop the unsupported settings"
        )
    return tuple(needed)


# ---------------------------------------------------------------------------
# The dask runtime seam
# ---------------------------------------------------------------------------


class DaskRuntime:
    """The only object in this module that calls dask.

    Never run against a live cluster. It is written from the documented
    dask APIs and is the piece most likely to need changing once someone
    tries it; everything else is tested against a fake implementing the
    same three methods, which are the whole seam:

        runtime.partitions(X, y, sample_weight, query_ids) -> PartitionMeta
        runtime.map_partitions(collection, fn, width)      -> collection
        runtime.is_collection(obj)                         -> bool

    Anything satisfying those can stand in, which is how the tests avoid
    starting a cluster and how another scheduler could be adapted later.

    A fourth method, `client()`, is not part of the seam and is not
    required: a backend that has to launch tasks on the cluster looks for
    it and falls back to `distributed.get_client()`. A stand-in runtime
    without one still validates, plans, and predicts.
    """

    def __init__(self, client=None):
        self._client = client

    def client(self):
        """The `distributed.Client` to use: the one passed in, or the
        current default one."""
        if self._client is not None:
            return self._client
        _, _, distributed = _import_dask()
        client = distributed.get_client()
        self._client = client
        return client

    def is_collection(self, obj):
        return is_dask_collection(obj)

    def partitions(self, X, y=None, sample_weight=None, query_ids=None):
        """Partition metadata for a persisted training collection.

        Persisting first is deliberate: worker locations and partition
        lengths are both properties of a materialized collection, and
        asking for them on a lazy one either fails or triggers a second
        computation of the whole graph.
        """
        client = self.client()
        X = client.persist(X)
        keys = _flat_keys(X)
        who_has = client.who_has(X)
        lengths = self._lengths(X, client)
        names = self._feature_names(X)
        n_features = self._n_features(X)
        categories = self._categories(X)
        label_keys = self._aligned_keys(y, len(keys), "y", client)
        weight_keys = self._aligned_keys(
            sample_weight, len(keys), "sample_weight", client
        )
        groups = _query_groups(query_ids, len(keys))
        metas = []
        for index, key in enumerate(keys):
            first_query = last_query = None
            group = None
            if groups is not None:
                group, first_query, last_query = groups[index]
            metas.append(
                PartitionMeta(
                    index=index,
                    worker=_worker_of(who_has, key),
                    n_rows=lengths[index],
                    n_features=n_features,
                    feature_names=names,
                    categories=categories,
                    has_weight=sample_weight is not None,
                    first_query=first_query,
                    last_query=last_query,
                    group=group,
                    key=key,
                    label_key=label_keys[index],
                    weight_key=weight_keys[index],
                )
            )
        return tuple(metas)

    def _aligned_keys(self, collection, n_partitions, name, client):
        """The keys of `collection`, one per feature partition.

        Labels and weights are separate collections, and a distributed
        trainer pairs them with the feature rows by partition. Partitioned
        differently, they would pair row 0 of one with row 0 of another and
        train on labels that belong to other rows, so a mismatch is an
        error here rather than a discovery on a worker.
        """
        if collection is None:
            return (None,) * n_partitions
        if not is_dask_collection(collection):
            raise PartitionError(
                f"{name} is not a Dask collection; distributed training "
                "pairs it with the feature rows partition by partition, so "
                "it has to be partitioned the same way X is"
            )
        keys = _flat_keys(client.persist(collection))
        if len(keys) != n_partitions:
            raise PartitionError(
                f"{name} has {len(keys)} partitions and X has "
                f"{n_partitions}; repartition so each block of rows has its "
                f"{name} on the same worker"
            )
        return tuple(keys)

    def map_partitions(self, collection, fn, width=1):
        """`fn` applied to each partition, as a new lazy collection.

        A dataframe needs a `meta` describing the result and an array needs
        a chunk shape, which is what `width` is for: 1 for one prediction
        per row, `n_classes` for `predict_proba`.
        """
        dask, dask_dataframe, _ = _import_dask()
        if dask_dataframe is not None and isinstance(
            collection, (dask_dataframe.DataFrame, dask_dataframe.Series)
        ):
            import numpy as np
            import pandas as pd

            if width == 1:
                meta = pd.Series(dtype="float64")
            else:
                meta = pd.DataFrame(
                    np.zeros((0, width), dtype="float64")
                )
            return collection.map_partitions(fn, meta=meta)
        chunks = None
        if width != 1:
            chunks = (collection.chunks[0], (width,))
        return collection.map_blocks(
            fn, dtype="float64", chunks=chunks
        )

    def _lengths(self, X, client):
        chunks = getattr(X, "chunks", None)
        if chunks is not None:
            return [
                None if _is_nan(n) else int(n) for n in chunks[0]
            ]
        lengths = X.map_partitions(len).compute()
        return [int(n) for n in lengths]

    def _feature_names(self, X):
        columns = getattr(X, "columns", None)
        if columns is None:
            return None
        return tuple(str(name) for name in columns)

    def _n_features(self, X):
        columns = getattr(X, "columns", None)
        if columns is not None:
            return len(columns)
        shape = getattr(X, "shape", ())
        return int(shape[1]) if len(shape) > 1 else 0

    def _categories(self, X):
        columns = getattr(X, "columns", None)
        dtypes = getattr(X, "dtypes", None)
        if columns is None or dtypes is None:
            return ()
        out = []
        for index, dtype in enumerate(dtypes):
            if getattr(dtype, "categories", None) is None:
                continue
            column = X[columns[index]]
            # dask fills the categories of an uncategorized column with a
            # placeholder rather than leaving them empty, so `.cat.known` is
            # the only reliable answer. Unknown reaches
            # merge_categorical_schema as None and is refused there, with
            # the .categorize() hint.
            known = bool(getattr(getattr(column, "cat", None), "known", True))
            out.append(
                (index, tuple(dtype.categories) if known else None)
            )
        return tuple(out)


def _flat_keys(collection):
    """A dask collection's partition keys, flattened in row order.

    A dataframe's keys are already flat; a 2D array's are one list per row
    block, each holding one key per column block. Column blocking is not
    supported: a rank owns whole rows.
    """
    keys = collection.__dask_keys__()
    flat = []
    for entry in keys:
        if isinstance(entry, list):
            if len(entry) != 1:
                raise PartitionError(
                    "the training array is chunked along its columns as "
                    "well as its rows; a rank owns whole rows, so rechunk "
                    "to one chunk per row block "
                    "(X.rechunk({1: X.shape[1]}))"
                )
            flat.append(entry[0])
        else:
            flat.append(entry)
    return flat


def _worker_of(who_has, key):
    """The first worker address holding `key`, or "".

    `Client.who_has` reports keys as strings while `__dask_keys__` yields
    the tuples they were built from, so both spellings are tried. A key
    nobody holds comes back empty and `validate_partitions` turns that into
    the "persist the collection first" error.
    """
    for candidate in (key, str(key)):
        try:
            addresses = who_has.get(candidate)
        except TypeError:  # an unhashable key spelling
            continue
        if addresses:
            return addresses[0]
    return ""


def _is_nan(value):
    return value != value


def _query_groups(query_ids, n_partitions):
    """Per-partition `(group, first query, last query)` from a list of
    per-partition query id sequences."""
    if query_ids is None:
        return None
    per_partition = list(query_ids)
    if len(per_partition) != n_partitions:
        raise PartitionError(
            f"{len(per_partition)} query id sequences for {n_partitions} "
            "partitions; a ranker needs one per partition, in row order"
        )
    out = []
    for index, ids in enumerate(per_partition):
        ids = list(ids)
        if not ids:
            raise PartitionError(
                f"partition {index} has no query ids; an empty partition is "
                "refused by validate_partitions, and a non-empty one owes a "
                "query id for every row"
            )
        out.append((partition_group(ids), ids[0], ids[-1]))
    return out


# ---------------------------------------------------------------------------
# Partition-local prediction
# ---------------------------------------------------------------------------

#: Deserialized models, keyed by the digest of their bytes and by
#: everything else that changes what the parsed estimator does. A worker
#: process predicts many partitions with the same model, and parsing the
#: model text once per partition would dominate prediction. Keyed by
#: content, so two estimators that hold the same model share an entry and a
#: refit never collides with the model it replaced.
#:
#: An entry owns a native model handle, so this is bounded: a long-lived
#: worker that predicts with model after model would otherwise hold every
#: one of them for the life of the process. Insertion order is the eviction
#: order, oldest first, which is enough for the access pattern this exists
#: for (one model, many partitions, then the next model).
_MODEL_CACHE = {}

#: How many parsed models one worker process keeps. Small on purpose: the
#: case worth caching is the same model across the partitions of one
#: collection, and the cost of a miss is one parse.
_MODEL_CACHE_LIMIT = 4


def clear_model_cache():
    """Drop the per-process cache of deserialized models."""
    _MODEL_CACHE.clear()


@dataclass(frozen=True)
class _PartitionPredictor:
    """A picklable callable that predicts one partition with a model
    carried as bytes.

    Shipping the bytes rather than the estimator is what makes distributed
    prediction independent of the training transport: there is no cluster
    state, no model future, and no requirement that the workers be the ones
    that trained. The same callable works under `map_partitions`,
    `map_blocks`, `client.submit`, or a plain loop, which is how it is
    tested.
    """

    blob: bytes
    kind: str
    multiclass: bool
    method: str = "predict"
    classes: tuple = None
    encoders: tuple = ()
    n_features: int = 0
    kwargs: dict = field(default_factory=dict)

    def model(self):
        """The parsed estimator for this predictor, from the process cache.

        The key is every input `_estimator_from_bytes` reads. `encoders`
        belongs in it as much as the bytes do: two fits of the same trees
        over differently ordered category tables predict different things
        on a pandas frame, and a key that left them out would hand the
        second fit the first one's encoders. The category tables arrive as
        lists, which a dict key may not hold, so they are frozen here rather
        than on the field, where `_estimator_from_bytes` wants them as the
        estimator stores them.
        """
        digest = hashlib.sha256(self.blob).hexdigest()
        key = (
            digest,
            self.kind,
            self.multiclass,
            self.classes,
            tuple((index, tuple(cats)) for index, cats in self.encoders),
        )
        cached = _MODEL_CACHE.get(key)
        if cached is None:
            cached = _estimator_from_bytes(
                self.blob,
                self.kind,
                self.multiclass,
                self.classes,
                dict(self.encoders),
            )
            while len(_MODEL_CACHE) >= _MODEL_CACHE_LIMIT:
                _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
            _MODEL_CACHE[key] = cached
        return cached

    def __call__(self, chunk):
        estimator = self.model()
        return getattr(estimator, self.method)(chunk, **self.kwargs)


_LOCAL_CLASSES = {
    "regressor": MojoTreesRegressor,
    "classifier": MojoTreesClassifier,
    "ranker": MojoTreesRanker,
}


def _estimator_from_bytes(blob, kind, multiclass, classes, encoders):
    """A fitted single-machine estimator from saved model bytes.

    The estimator is the local one, not the Dask one: what a worker holds
    is a model, and the Dask subclass exists only to reach a cluster.
    """
    estimator = _LOCAL_CLASSES[kind]()
    estimator._multiclass = bool(multiclass)
    estimator._model = estimator._model_from_bytes(blob)
    estimator._restore_categorical()
    if encoders:
        estimator._cat_encoders = dict(encoders)
    _install_shape(estimator, kind, classes)
    return estimator


def _install_shape(estimator, kind, classes):
    """The attributes a fitted estimator answers for that model bytes do
    not carry: the feature count, the iteration count, and a classifier's
    labels."""
    from . import _mojotrees

    if estimator._multiclass:
        estimator.n_features_in_ = int(
            _mojotrees.n_features_multiclass(estimator._model)
        )
        estimator.n_classes_ = int(_mojotrees.n_classes(estimator._model))
    else:
        estimator.n_features_in_ = int(
            _mojotrees.n_features(estimator._model)
        )
        if kind == "classifier":
            estimator.n_classes_ = 2
    estimator.best_iteration_ = estimator._num_iterations()
    if kind == "classifier":
        labels = (
            list(classes)
            if classes is not None
            else list(range(estimator.n_classes_))
        )
        if len(labels) != estimator.n_classes_:
            raise ModelOwnershipError(
                f"the backend returned a model over {estimator.n_classes_} "
                f"classes, but the fit declared {len(labels)}: "
                f"{labels!r}"
            )
        # `predict` indexes `classes_` with an array of argmaxes, so it has
        # to be an array wherever numpy is installed, exactly as `load()`
        # leaves it.
        estimator.classes_ = _as_label_array(labels)
    return estimator


def _as_label_array(labels):
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is optional
        return list(labels)
    return np.asarray(labels)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


class _DaskMixin:
    """The distributed half of the three estimators below.

    Each subclasses its single-machine counterpart, so `get_params`,
    `set_params`, `clone`, pickling, the parameter validation, and
    prediction on ordinary arrays are all inherited rather than
    reimplemented. What is added is a `fit` that takes Dask collections and
    a `predict` that returns one.
    """

    #: Which local estimator this wraps, for the partition predictor.
    _kind = None

    def __init__(self, client=None, one_rank_per_worker=True, **kwargs):
        super().__init__(**kwargs)
        self.client = client
        self.one_rank_per_worker = one_rank_per_worker
        self._runtime = None
        self._plan_ = None
        self._model_blob_cache = None

    # -- the model --------------------------------------------------------

    def _model_blob(self):
        """The fitted model as bytes, from the one model this estimator
        owns.

        A fitted estimator holds one model: the handle inside the `Booster`
        on `_booster`, exactly as a single-machine estimator does. The bytes
        a backend handed back are kept next to it as a cache and nothing
        more. They are not pickled and they are not a second source of
        truth; when they are missing, they are written back out of the
        Booster through the same versioned format `save()` uses. So there is
        no state in which this estimator holds two models that could
        disagree, and an unpickled estimator can still predict on a Dask
        collection, which needs bytes to ship to the workers.
        """
        self._require_fitted()
        blob = getattr(self, "_model_blob_cache", None)
        if blob is None:
            blob = self._model_bytes()
            self._model_blob_cache = blob
        return blob

    # -- planning ---------------------------------------------------------

    def _runtime_for(self, runtime):
        """The runtime to use: the one passed in, then the one this
        estimator was fitted through, then a fresh `DaskRuntime`.

        Reusing the fit-time runtime is what lets `predict` reach the same
        cluster without being handed the client again. It does not survive
        pickling, so an unpickled estimator predicting on a collection
        needs a client or a runtime of its own.
        """
        if runtime is not None:
            return runtime
        existing = getattr(self, "_runtime", None)
        if existing is not None:
            return existing
        return DaskRuntime(self.client)

    def _training_params(self):
        return {
            name: value
            for name, value in self.get_params().items()
            if name not in _NON_TRAINING_PARAMS
        }

    def _shared_capabilities(self, partitions, schema):
        """Capabilities every estimator needs to ask about, from the
        settings that are on the shared base."""
        needed = set()
        if schema:
            needed.add("categorical")
        if any(part.has_weight for part in partitions):
            needed.add("weights")
        if self.bagging_fraction != 1.0 or self.subsample not in (None, 1.0):
            needed.add("bagging")
        # Resolved by the estimator rather than re-read here, so the
        # `boosting` / `boosting_type` alias rule has one owner. GOSS keeps
        # the largest gradients and samples the rest, which is a decision
        # over the global row set: a rank sampling its own rows is a
        # different algorithm, not a distributed version of this one.
        if self._resolve_boosting() == "goss":
            needed.add("goss")
        if self.feature_fraction != 1.0 or self.feature_fraction_bynode != 1.0:
            needed.add("feature_fraction")
        # Through `_requested_device` since 2026-08-17, which folds all THREE
        # spellings. This site already read `device` and `device_type`, so it
        # was not the dead-branch bug the multi-target boundary had, but it
        # never consulted `task_type`, and `task_type="GPU"` is CatBoost's own
        # spelling. A capability probe that misses a spelling reports that a
        # cluster needs no GPU for a fit that asked for one.
        if str(self._requested_device()).lower() == "gpu":
            needed.add("gpu")
        return needed

    def _capabilities(self, partitions, schema):
        raise NotImplementedError

    def _reject_unsupported_fit_arguments(self, extra):
        if not extra:
            return
        raise TypeError(
            f"distributed fit does not take {sorted(extra)}. A distributed "
            "fit takes eval_set, eval_names, eval_sample_weight, "
            "eval_metric, early_stopping_rounds, and first_metric_only for "
            "validation, and nothing else beyond the training data: a "
            "per-iteration Python callback would have to run on one client "
            "while every rank waits inside a collective, which is not a "
            "thing this adapter will pretend to do. Fit on one machine "
            "when you need it"
        )

    # -- validation -------------------------------------------------------

    def _validation_arguments(
        self,
        eval_set,
        eval_names,
        eval_sample_weight,
        eval_query_ids,
        early_stopping_rounds,
        eval_metric,
    ):
        """The validation request as a normalized tuple, or `()`.

        Early stopping without an eval set is rejected rather than planned:
        there is nothing to stop on, and a fit that quietly ran to
        `n_estimators` after being asked to stop early would be the worst
        of the available answers. It raises `TypeError` because the
        argument itself is the thing that cannot be honored, which is also
        what this signature did before it took validation at all.
        """
        rounds = int(early_stopping_rounds or 0)
        if not eval_set:
            orphaned = set()
            if rounds > 0:
                orphaned.add("early_stopping_rounds")
            if eval_metric:
                orphaned.add("eval_metric")
            if eval_names:
                orphaned.add("eval_names")
            if eval_sample_weight:
                orphaned.add("eval_sample_weight")
            if eval_query_ids:
                orphaned.add("eval_query_ids")
            if orphaned:
                raise TypeError(
                    f"{sorted(orphaned)} need an eval_set to act on, and "
                    "this fit was given none. Pass "
                    "eval_set=[(X_valid, y_valid)] of Dask collections "
                    "partitioned the way the training data is, or drop the "
                    "argument"
                )
            return ()
        sets = list(eval_set)
        names = list(eval_names or [])
        if names and len(names) != len(sets):
            raise ValueError(
                f"{len(names)} eval_names for {len(sets)} eval_set entries"
            )
        weights = list(eval_sample_weight or [])
        if weights and len(weights) != len(sets):
            raise ValueError(
                f"{len(weights)} eval_sample_weight entries for "
                f"{len(sets)} eval_set entries"
            )
        queries = list(eval_query_ids or [])
        if queries and len(queries) != len(sets):
            raise ValueError(
                f"{len(queries)} eval_query_ids entries for {len(sets)} "
                "eval_set entries"
            )
        out = []
        for index, pair in enumerate(sets):
            try:
                X_valid, y_valid = pair
            except (TypeError, ValueError):
                raise ValueError(
                    "eval_set holds (X, y) pairs of Dask collections; "
                    f"entry {index} is {pair!r}"
                ) from None
            out.append(
                (
                    names[index] if names else f"valid_{index}",
                    X_valid,
                    y_valid,
                    weights[index] if weights else None,
                    queries[index] if queries else None,
                )
            )
        return tuple(out)

    def _plan_validation(self, validation, runtime, plan, ranking):
        """A `ValidationPlan` per eval set, on the training rank layout.

        Every rank scores the validation rows that live on its own worker
        and the losses are reduced, exactly as the gradients are. That only
        works when the eval collection lands rank for rank on the training
        collection: same number of ranks, same worker per rank. Anything
        else would mean shipping validation rows across the cluster every
        round, so it is refused here with the repartition to make instead.
        """
        plans = []
        for name, X_valid, y_valid, weight, queries in validation:
            parts = validate_partitions(
                runtime.partitions(
                    X_valid,
                    y_valid,
                    sample_weight=weight,
                    query_ids=queries,
                )
            )
            if ranking:
                validate_query_partitioning(parts)
            valid_plan = plan_world(parts, self.one_rank_per_worker)
            if valid_plan.n_features != plan.n_features:
                raise SchemaError(
                    f"eval set {name!r} has {valid_plan.n_features} "
                    f"features and the training data has {plan.n_features}"
                )
            if valid_plan.feature_names != plan.feature_names:
                raise SchemaError(
                    f"eval set {name!r} and the training data disagree "
                    f"about their column names ({plan.feature_names!r} "
                    f"against {valid_plan.feature_names!r})"
                )
            if valid_plan.schema.columns != plan.schema.columns:
                raise SchemaError(
                    f"eval set {name!r} and the training data disagree "
                    "about their categorical columns; a category has to "
                    "carry the same code in both or the model scores the "
                    "validation rows against a different encoding"
                )
            if valid_plan.workers != plan.workers:
                raise PartitionError(
                    f"eval set {name!r} lands on workers "
                    f"{list(valid_plan.workers)} and the training data on "
                    f"{list(plan.workers)}. A rank scores the validation "
                    "rows that sit on its own worker, so the two "
                    "collections have to be partitioned the same way and "
                    "persisted together. Repartition the eval set to match "
                    "the training set, or fit without one and score "
                    "afterwards with predict()"
                )
            plans.append(ValidationPlan(name=name, plan=valid_plan))
        return tuple(plans)

    def _install_metrics(self, record, validation, metrics):
        """The validation history a backend reported, as fitted state.

        The names are the single-machine ones (`evals_result_`,
        `best_score_`, `best_iteration_`, `stopped_early_`, `n_iter_`) and
        the layout of `evals_result_` is the single-machine one, because a
        distributed fit that reports its validation differently would be a
        second convention for the same thing. A backend that reports
        nothing leaves all of them as `_install_shape` set them.
        """
        if not record or not validation:
            return
        get = _reader(record)
        values = list(get("values") or ())
        n_rounds = int(get("n_rounds") or 0)
        n_valid = len(validation)
        # The backend's own names win when it reports them: it is the side
        # that resolved a default metric out of the objective, which the
        # client cannot do without duplicating the objective registry.
        metrics = tuple(get("eval_metrics") or metrics)
        n_metrics = len(metrics)
        if n_rounds and n_metrics and len(values) >= (
            n_rounds * n_valid * n_metrics
        ):
            self.evals_result_ = {
                validation[v].name: {
                    metrics[m]: [
                        float(
                            values[(r * n_valid + v) * n_metrics + m]
                        )
                        for r in range(n_rounds)
                    ]
                    for m in range(n_metrics)
                }
                for v in range(n_valid)
            }
            # Round 0 is the base-score-only model, exactly as it is for a
            # single-machine fit, so one fewer round was trained than the
            # history holds entries for.
            self.n_iter_ = max(n_rounds - 1, 0)
        best_iteration = int(get("best_iteration") or -1)
        if best_iteration >= 0:
            self.best_iteration_ = best_iteration
        best_score = get("best_score")
        if best_score is not None:
            self.best_score_ = float(best_score)
        self.stopped_early_ = bool(get("stopped_early"))

    # -- fit --------------------------------------------------------------

    def _distributed_fit(
        self,
        X,
        y,
        sample_weight=None,
        query_ids=None,
        runtime=None,
        timeout=None,
        label_classes=None,
        objective=None,
        ranking=False,
        validation=(),
        eval_metric=None,
        early_stopping_rounds=0,
        first_metric_only=False,
    ):
        """The shared fit path: backend first, then the cluster.

        The backend is resolved before the collection is touched on
        purpose. With no runtime available, which is every published build
        today, a user gets `DistributedNotAvailable` immediately instead of
        after persisting a terabyte.

        Nothing between here and `backend.train` computes anything about
        the data: this validates metadata, plans ranks, and asks. The rows
        stay in the partitions dask put them in, and the only thing that
        comes back to the client is the model.
        """
        backend = get_backend()
        self._reset_fitted()
        self._plan_ = None
        # Not in `_FITTED_ATTRS`, which belongs to the single-machine
        # estimators, so a refit drops the stale bytes here.
        self._model_blob_cache = None
        runtime = self._runtime_for(runtime)
        partitions = validate_partitions(
            runtime.partitions(
                X, y, sample_weight=sample_weight, query_ids=query_ids
            )
        )
        if ranking:
            validate_query_partitioning(partitions)
        plan = plan_world(partitions, self.one_rank_per_worker)
        needed = self._capabilities(partitions, plan.schema)
        valid_plans = ()
        metrics = ()
        if validation:
            needed.add("early_stopping")
            valid_plans = self._plan_validation(
                validation, runtime, plan, ranking
            )
            metrics = _metric_names(eval_metric)
        require_capabilities(backend, needed)
        job = TrainingJob(
            plan=plan,
            objective=objective,
            params=self._training_params(),
            label_classes=label_classes,
            ranking=ranking,
            timeout=timeout,
            validation=valid_plans,
            eval_metrics=metrics,
            early_stopping_rounds=int(early_stopping_rounds or 0),
            first_metric_only=bool(first_metric_only),
        )
        # Optional, so a backend written against version 0 is unaffected:
        # the cluster is a property of the collection being fitted, and a
        # backend that has to reach it gets the runtime here rather than a
        # live handle inside the frozen job.
        bind = getattr(backend, "bind_runtime", None)
        if callable(bind):
            bind(runtime)
        ref = self._train_with_cancellation(backend, job)
        blob = take_model_bytes(ref, timeout)
        self._install_model(blob, plan, label_classes)
        report = getattr(ref, "metrics", None)
        if callable(report):
            self._install_metrics(report(), valid_plans, metrics)
        self._runtime = runtime
        self._plan_ = plan
        return self

    def _train_with_cancellation(self, backend, job):
        """`backend.train(job)`, with an interrupt stopping the cluster.

        Ctrl-C on the client would otherwise leave every rank sitting
        inside a collective until its deadline expired, holding a worker
        each. A backend that declares no `cancel` is left alone, and the
        interrupt propagates either way: this cancels, it does not swallow.
        """
        try:
            return backend.train(job)
        except BaseException:
            cancel = getattr(backend, "cancel", None)
            if callable(cancel):
                try:
                    cancel(job)
                except Exception:
                    pass
            raise

    def _install_model(self, blob, plan, label_classes):
        """Turn the bytes a backend returned into this estimator's fitted
        state.

        The model format carries the trees, the feature count, and which
        features are categorical, and nothing else; the category labels and
        the class labels come from the plan and the label metadata instead,
        which is what lets a distributed fit predict on a pandas frame the
        way a local fit does.
        """
        self._multiclass = bool(
            label_classes is not None and len(label_classes) > 2
        )
        self._model = self._model_from_bytes(blob)
        self._restore_categorical()
        encoders = plan.schema.encoders()
        if encoders:
            self._cat_encoders = encoders
        _install_shape(self, self._kind, label_classes)
        if plan.n_features != self.n_features_in_:
            raise ModelOwnershipError(
                f"the backend returned a model over {self.n_features_in_} "
                f"features, but the training collection has "
                f"{plan.n_features}"
            )
        if plan.feature_names is not None:
            self.feature_names_in_ = list(plan.feature_names)
        # The Booster built above is the model; these are the same bytes
        # kept so that shipping them to the workers does not re-serialize
        # it. See `_model_blob`.
        self._model_blob_cache = blob
        return self

    # -- predict ----------------------------------------------------------

    def _class_labels(self):
        """`classes_` as a plain tuple, or None.

        Spelled out rather than `tuple(self.classes_ or ())`, because
        `classes_` is a numpy array wherever numpy is installed and an
        array is not allowed to answer a truth test.
        """
        labels = getattr(self, "classes_", None)
        if labels is None or len(labels) == 0:
            return None
        return tuple(labels)

    def _distributed_predict(self, X, method, args, kwargs, runtime, width):
        self._require_fitted()
        if args:
            raise TypeError(
                "pass predict options to a Dask collection by keyword; "
                "positional arguments are ambiguous once the call is "
                "shipped to a worker"
            )
        for flag in ("pred_leaf", "pred_contrib"):
            if kwargs.get(flag):
                raise ValueError(
                    f"{flag} is not available on a Dask collection: it "
                    "returns a matrix per row whose width depends on the "
                    "model, which the collection's metadata cannot describe "
                    "here. Compute the partition you want and predict on it "
                    "locally"
                )
        predictor = _PartitionPredictor(
            blob=self._model_blob(),
            kind=self._kind,
            multiclass=self._multiclass,
            method=method,
            classes=self._class_labels(),
            encoders=tuple(sorted(self._cat_encoders.items()))
            if self._cat_encoders
            else (),
            n_features=self.n_features_in_,
            kwargs=dict(kwargs),
        )
        runtime = self._runtime_for(runtime)
        return runtime.map_partitions(X, predictor, width)

    def predict(self, X, *args, runtime=None, **kwargs):
        """Predictions for `X`, as a Dask collection when `X` is one.

        A Dask `X` gets one prediction per row, computed on whichever
        worker holds the partition, with the model shipped as bytes and
        parsed once per worker process. No cluster state and no model
        future is involved, so this works on a cluster that had nothing to
        do with training. Anything else falls through to the
        single-machine `predict`.
        """
        if is_dask_collection(X):
            return self._distributed_predict(
                X, "predict", args, kwargs, runtime, 1
            )
        return super().predict(X, *args, **kwargs)

    def to_local(self):
        """This estimator as its single-machine counterpart.

        The result holds the same model and predicts identically; what it
        does not hold is the client, the rank plan, or any way to reach a
        cluster. It is what to pickle, save, or hand to code that should
        not know a cluster was involved.
        """
        return _estimator_from_bytes(
            self._model_blob(),
            self._kind,
            self._multiclass,
            self._class_labels(),
            dict(self._cat_encoders),
        )

    # -- state ------------------------------------------------------------

    def __getstate__(self):
        """Pickle support. A `Client`, a runtime, and a rank plan describe
        a cluster this estimator is finished with, so none of them travel;
        the model does, exactly as it does for the single-machine
        estimators.

        The model travels once. `_Base.__getstate__` already writes the
        Booster out as bytes, so carrying `_model_blob_cache` as well would
        put the same model in the pickle twice and leave two copies to
        disagree if either were ever edited. It is a cache of the one model,
        and `_model_blob` rebuilds it on demand.
        """
        state = super().__getstate__()
        for name in ("_runtime", "_plan_", "_model_blob_cache"):
            state[name] = None
        state["client"] = None
        return state


class DaskMojoTreesRegressor(_DaskMixin, MojoTreesRegressor):
    """`MojoTreesRegressor` over a Dask cluster, as a contract.

    Every parameter of `MojoTreesRegressor` is accepted and forwarded to
    the backend under its own name, plus:

    - `client`: the `distributed.Client` to use, or `None` for the current
      one.
    - `one_rank_per_worker`: whether a worker's partitions are concatenated
      into a single rank. See `plan_world`.

    `fit` raises `DistributedNotAvailable` until a backend is registered,
    which nothing in mojotrees does yet.
    """

    _kind = "regressor"

    def _capabilities(self, partitions, schema):
        needed = self._shared_capabilities(partitions, schema)
        needed.add("regression")
        if callable(self.objective):
            needed.add("custom_objective")
        elif str(self.objective) in ("quantile", "mae", "regression_l1"):
            # Leaf renewal takes a percentile of the leaf's residuals, and a
            # percentile does not all-reduce. docs/distributed.md, section 9.
            needed.add("quantile_l1")
        return needed

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_metric=None,
        early_stopping_rounds=None,
        first_metric_only=False,
        runtime=None,
        timeout=None,
        **unsupported,
    ):
        """Fit on Dask collections `X` and `y`.

        Validates the partition layout, plans the ranks, negotiates
        capabilities with the registered backend, and installs the model it
        returns. Returns self.

        `eval_set` is a list of `(X, y)` pairs of Dask collections
        partitioned exactly as the training data is, so each rank scores
        the validation rows on its own worker and the loss is reduced
        alongside the gradients. It needs a backend declaring
        `early_stopping`. `evals_result_` is filled in when the metric
        names are known, which means when `eval_metric` names them or the
        backend reports the names it resolved; `best_score_`,
        `best_iteration_`, and `stopped_early_` come back either way.
        """
        self._reject_unsupported_fit_arguments(unsupported)
        validation = self._validation_arguments(
            eval_set,
            eval_names,
            eval_sample_weight,
            None,
            early_stopping_rounds,
            eval_metric,
        )
        objective = (
            "custom" if callable(self.objective) else str(self.objective)
        )
        return self._distributed_fit(
            X,
            y,
            sample_weight=sample_weight,
            runtime=runtime,
            timeout=timeout,
            objective=objective,
            validation=validation,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            first_metric_only=first_metric_only,
        )


class DaskMojoTreesClassifier(_DaskMixin, MojoTreesClassifier):
    """`MojoTreesClassifier` over a Dask cluster, as a contract.

    The class labels are a global fact and the partitions are not: a
    partition that happens to hold one class must still encode its labels
    the way every other rank does. `fit` therefore takes `classes`, the
    global label list, and refuses to infer it from a metadata scan that
    might have missed a rare class.
    """

    _kind = "classifier"

    #: The class list of the fit in progress. Whether a fit needs the
    #: "binary" or the "multiclass" capability depends on it, and
    #: `_capabilities` runs inside `fit`, before there is a `classes_`.
    _pending_classes = None

    def _capabilities(self, partitions, schema):
        needed = self._shared_capabilities(partitions, schema)
        labels = self._pending_classes or ()
        needed.add("multiclass" if len(labels) > 2 else "binary")
        if callable(self.objective):
            needed.add("custom_objective")
        if self.class_weight is not None:
            needed.add("class_weight")
        return needed

    def fit(
        self,
        X,
        y,
        classes,
        sample_weight=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_metric=None,
        early_stopping_rounds=None,
        first_metric_only=False,
        runtime=None,
        timeout=None,
        **unsupported,
    ):
        """Fit on Dask collections `X` and `y`, with `classes` the sorted
        global list of label values.

        Returns self. `classes_` is what was passed, in the order it was
        passed, and the workers encode labels through that order, which is
        why a rank holding an unseen label fails instead of inventing a
        code for it. `eval_set` is as in `DaskMojoTreesRegressor.fit`, and
        its labels are encoded through the same class list.
        """
        self._reject_unsupported_fit_arguments(unsupported)
        if isinstance(self.class_weight, str):
            # "balanced" is one over the class frequency, and the frequency
            # is a global count of rows this client never sees: partition
            # metadata carries which labels a partition holds, not how many
            # rows of each. A per-rank "balanced" would weight the same
            # class differently on every rank.
            raise ValueError(
                f"class_weight={self.class_weight!r} needs the row count of "
                "every class across the whole collection, which the client "
                "does not have and a rank cannot compute from its own rows. "
                "Count the labels yourself and pass the weights as a dict, "
                "or fit on one machine"
            )
        labels = tuple(classes)
        if len(labels) < 2:
            raise ValueError(
                "a classifier needs at least 2 classes; "
                f"classes={list(labels)!r}"
            )
        if len(set(labels)) != len(labels):
            raise ValueError(f"classes has repeats: {list(labels)!r}")
        validation = self._validation_arguments(
            eval_set,
            eval_names,
            eval_sample_weight,
            None,
            early_stopping_rounds,
            eval_metric,
        )
        self._pending_classes = labels
        try:
            return self._distributed_fit(
                X,
                y,
                sample_weight=sample_weight,
                runtime=runtime,
                timeout=timeout,
                label_classes=labels,
                objective="multiclass" if len(labels) > 2 else "binary",
                validation=validation,
                eval_metric=eval_metric,
                early_stopping_rounds=early_stopping_rounds,
                first_metric_only=first_metric_only,
            )
        finally:
            self._pending_classes = None

    def predict_proba(self, X, *args, runtime=None, **kwargs):
        """Class probabilities for `X`, as a Dask collection when `X` is
        one: `n_classes_` columns per row, in `classes_` order."""
        if is_dask_collection(X):
            # Before `n_classes_` is read for the output width, so an
            # unfitted estimator raises NotFittedError rather than
            # AttributeError.
            self._require_fitted()
            return self._distributed_predict(
                X,
                "predict_proba",
                args,
                kwargs,
                runtime,
                int(self.n_classes_),
            )
        return super().predict_proba(X, *args, **kwargs)


class DaskMojoTreesRanker(_DaskMixin, MojoTreesRanker):
    """`MojoTreesRanker` over a Dask cluster, as a contract.

    Ranking adds the one partition rule the other two do not have: a query
    has to land whole on one rank. `fit` takes `query_ids`, one sequence
    per partition in row order, builds each partition's group from it, and
    refuses a query that straddles a boundary. See
    `validate_query_partitioning`.
    """

    _kind = "ranker"

    def _capabilities(self, partitions, schema):
        needed = self._shared_capabilities(partitions, schema)
        needed.add("ranking")
        return needed

    def fit(
        self,
        X,
        y,
        query_ids,
        sample_weight=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_query_ids=None,
        eval_metric=None,
        early_stopping_rounds=None,
        first_metric_only=False,
        runtime=None,
        timeout=None,
        **unsupported,
    ):
        """Fit on Dask collections `X` and `y`, with `query_ids` a sequence
        of per-partition query id columns, in partition order.

        Query ids rather than LightGBM's `group`: a group array alone
        cannot say whether the query at a partition's edge continues into
        the next one, which is the thing that has to be checked.

        An `eval_set` needs `eval_query_ids`, one per eval set, in the same
        shape, and its queries are checked against the same partition rule:
        NDCG is computed within a query, so a query split across two ranks
        would be scored as two shorter ones.

        Returns self.
        """
        self._reject_unsupported_fit_arguments(unsupported)
        if query_ids is None:
            raise ValueError(
                "a distributed ranker needs query_ids: one sequence of "
                "per-row query ids per partition, in partition order"
            )
        if eval_set and not eval_query_ids:
            raise ValueError(
                "a distributed ranker's eval_set needs eval_query_ids: one "
                "sequence of per-partition query id columns per eval set. "
                "Without them a rank cannot form the query groups it "
                "scores NDCG over"
            )
        validation = self._validation_arguments(
            eval_set,
            eval_names,
            eval_sample_weight,
            eval_query_ids,
            early_stopping_rounds,
            eval_metric,
        )
        return self._distributed_fit(
            X,
            y,
            sample_weight=sample_weight,
            query_ids=query_ids,
            runtime=runtime,
            timeout=timeout,
            objective="lambdarank",
            ranking=True,
            validation=validation,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            first_metric_only=first_metric_only,
        )
