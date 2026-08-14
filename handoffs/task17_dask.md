# Task 17 handoff: the Dask adapter contract

Files added by this lane, and nothing else:

- `python/mojoboost/dask.py`
- `python/tests/parallel/test_dask_contract.py`
- `handoffs/task17_dask.md` (this file)

No central or shared file was touched. Nothing was committed or staged.

## Status, stated plainly

`mojoboost.dask` is a contract, not a feature. It imports without dask
installed, validates the metadata a distributed trainer needs, plans ranks,
negotiates capabilities, takes ownership of a model reference, and predicts
partition by partition from model bytes. It does not train, and no
distributed training has been run.

`DaskMojoBoostRegressor.fit(...)` raises `DistributedNotAvailable` on every
installation today, before it touches the cluster, because no backend is
registered and mojoboost ships none.

What has actually been executed: `python/tests/parallel/test_dask_contract.py`,
71 tests, all passing on 2026-08-14 against a fake runtime and a fake
backend, plus locally trained models for the prediction paths.

What has never been executed: `DaskRuntime`, the only class in the module
that calls dask. It is written from the documented dask APIs and should be
read as a proposal. No scheduler, worker, or Dask collection appears
anywhere in the tests.

Do not describe any of the three estimators as supported until the checklist
at the end of this file is complete.

## The backend protocol, version 0

This is the whole seam. A backend is any object with three members:

```python
backend.name          # str, used in error messages
backend.capabilities  # container of names drawn from mojoboost.dask.CAPABILITIES
backend.train(job)    # -> a model reference
```

`job` is a frozen `TrainingJob`:

| field | type | meaning |
| --- | --- | --- |
| `plan` | `WorldPlan` | ranks, workers, row offsets, column schema |
| `objective` | `str` | `regression`, `binary`, `multiclass`, `lambdarank`, `custom`, or a LightGBM regression objective name |
| `params` | `dict` | estimator hyperparameters under their LightGBM spellings, minus `client` and `one_rank_per_worker` |
| `label_classes` | `tuple` or `None` | the global class list for a classifier, in the order labels must be encoded |
| `ranking` | `bool` | whether the query partition rules were checked |
| `timeout` | `float` or `None` | seconds the client will wait for the model |
| `protocol_version` | `int` | `0` today; refuse an unrecognized value rather than guessing |

`plan.ranks` is in rank order, which is row order. Rank `r` owns the global
rows `[row_offset, row_offset + n_rows)`, its `worker` is the address that
holds them, and `partitions` lists the partition indices it concatenates.
`plan.addresses_unique` is False when two ranks share a worker, which is
legal only under `one_rank_per_worker=False`; a transport that addresses
peers by worker address rather than by rank has to check it.

**How a backend reaches the rows.** `plan.partitions` is the validated
partition metadata in partition order, and a rank's `partitions` tuple
indexes into it. Each `PartitionMeta` carries `key`, `label_key`, and
`weight_key`: the scheduler keys of that partition's feature block, its
labels, and its weights. Resolve them through the client the usual way
(`distributed.Future(key)`, `client.who_has`, or `futures_of` on the
collection the caller passed). The adapter checks that labels and weights
have the same partition count as the features, because a mismatch would
pair rows with labels that belong to other rows.

The model reference is any object with:

```python
ref.owner             # str, the worker address holding the model
ref.result(timeout)   # -> bytes of a model saved in mojoboost's text format
ref.release()         # -> None, called exactly once, idempotent
```

`BytesModelRef` implements it for a backend that already holds the bytes.
The client calls `result` once and `release` in a `finally`, so a backend
may free the worker-side model in `release`, and must not require the client
to hold a live future afterwards.

## What Task 16 and the integration lane must expose

Nothing below exists yet. Each item blocks the estimators.

### 1. A Python-visible backend object

The transport in `src/mojoboost/distributed_transport.mojo` is Mojo. Some
Python object has to satisfy the three-member protocol above and reach it.
Two ways, and the choice is the transport lane's:

- a Python class in a new module (`python/mojoboost/_distributed.py`, say)
  that calls new `_mojoboost` entry points, registered by the user with
  `mojoboost.dask.register_backend(...)`, or
- an object exported from anywhere importable and named in
  `MOJOBOOST_DASK_BACKEND` as `package.module:attribute`. A class there is
  called with no arguments and the instance is used.

Whichever it is, it must be constructible on the client, and its `train`
must be callable from the client process. `register_backend` checks the
three members and rejects a capability name outside `CAPABILITIES`.

### 2. Native entry points `train` needs

The adapter deliberately does not say how rows reach the trainer, but the
backend needs at least these, and none of them exists in the extension
module today (`_mojoboost` currently exports `fit`, `fit_multiclass`,
`fit_ranker`, their `_with_metrics` and `_csc` variants, `predict_*`,
`save`, `load`, and the introspection calls, and nothing distributed):

- a worker-side entry that takes this rank's rows, labels, optional weights,
  optional group, a **pre-fit global `BinMapper`**, the rank, the world
  size, and a transport handle, and runs the loop in section 4 of
  `docs/distributed.md`;
- a way to build or ship that global `BinMapper`. Section 9 of
  `docs/distributed.md` is explicit that the prototype refuses to guess bin
  edges, and per-shard binning makes every reduced histogram cell
  meaningless. Either the client fits the mapper on a sample and broadcasts
  it, or the transport implements a distributed quantile sketch. The adapter
  has no opinion, but the backend cannot skip it;
- a rank 0 model export in the same versioned text format `_mojoboost.save`
  writes. The client rebuilds the estimator from those bytes with the
  existing `load` / `load_multiclass`, so a new format would break
  `to_local`, pickling, and distributed prediction at once.

### 3. Mapping a `WorldPlan` onto `TransportConfig`

Read from `src/mojoboost/distributed_transport.mojo` as it stood on
2026-08-14, while that lane was still working, so check it before relying on
any of this.

`TransportConfig.addresses` is indexed by rank, and rank `r` is whoever
listens at `addresses[r]`. That is the same convention `WorldPlan.ranks`
uses, so the two line up positionally with no reordering. Three mismatches
have to be resolved by the backend, not by the adapter:

- **A dask worker address is not a transport address.** `plan.workers[r]` is
  something like `tcp://10.0.0.4:38921`, the worker's own port. `RankAddress`
  wants the `host:port` a mojoboost rank listens on, which does not exist
  until something opens it. The backend has to pick or be given a port per
  rank and build the list; the host is the one in the dask address.
- **`plan.addresses_unique` is False when two ranks share a worker**, which
  is legal under `one_rank_per_worker=False`. `TransportConfig.validate`
  rejects duplicate addresses, so those ranks need distinct ports on the
  same host. Check the flag rather than assuming.
- **`job_id` is the backend's to generate.** `TrainingJob` has no such
  field on purpose: a transport job id is a property of the launch, not of
  the training request, and the client has no way to make one that every
  rank agrees on. Whatever generates it has to give every rank the same
  value.

`TrainingJob.timeout` is seconds as a float and `TransportConfig` takes
nanoseconds in two separate deadlines (`connect_timeout_ns`,
`collective_timeout_ns`). The adapter has one number and no opinion about
how it splits; deciding that is part of writing the backend.

### 4. Capability declarations

`CAPABILITIES` in `python/mojoboost/dask.py` is the vocabulary:
`regression`, `binary`, `multiclass`, `ranking`, `categorical`, `weights`,
`quantile_l1`, `bagging`, `feature_fraction`, `early_stopping`,
`custom_objective`, `gpu`.

The adapter asks for the ones a given fit needs and refuses the fit when the
backend does not declare them. The refusal list in section 9 of
`docs/distributed.md` is deliberately **not** copied into the adapter: the
backend owns it. A first backend matching today's prototype would declare
roughly `{"regression", "binary"}` and nothing else, which makes every other
fit fail with a message naming what is missing.

Add a capability name here only by editing `CAPABILITIES`; an undeclared
name is rejected at registration so a typo cannot read as a missing feature.

### 5. Error and lifecycle semantics the adapter assumes

- `train` raises on failure. Any exception propagates to the caller of
  `fit`, and the estimator is left unfitted (`fit` calls `_reset_fitted`
  before it starts).
- A worker lost mid-fit is `train`'s problem, not the adapter's. Decide
  between restart-from-checkpoint and fail-fast in Task 16, and say which in
  `docs/DISTRIBUTED_TRANSPORT.md`.
- `timeout` is advisory to `train` and is also passed to `ref.result`. A
  backend that ignores it will hang the client.
- Cancellation: not modeled. If Task 16 wants `KeyboardInterrupt` to stop a
  run, `train` has to handle it, and the adapter needs a follow-up.
- The adapter never retries. Nothing here is a retry loop.

### 6. Integration edits this lane did not make

Each of these is in a central or shared file and was left alone on purpose:

- `pyproject.toml`: an optional extra, `dask = ["dask[distributed]"]`, under
  `[project.optional-dependencies]`. `[tool.setuptools] packages =
  ["mojoboost"]` already ships `dask.py`, so packaging needs nothing else.
- `pixi.toml`: `python/tests/parallel/test_dask_contract.py` runs under the
  existing `pytest` feature with no new dependency (it uses fakes). If the
  suite grows a real-cluster test, that needs its own feature with dask and
  distributed, and its own CI job, because it is slow and flaky by nature.
- `python/mojoboost/__init__.py`: no edit is required. `mojoboost.dask` is a
  submodule the user imports by name, the way `lightgbm.dask` is, and
  importing it from `__init__` would undo the import safety this lane is
  built around. If a re-export is wanted later, it must stay lazy.
- `README.md` and `docs/LIGHTGBM_PARITY.md`: a row and a section, once a
  backend exists. Until then the honest statement is the one at the top of
  this file, and README's existing "no distributed performance claim" line
  still holds.
- `docs/distributed.md`: section 9's list and the adapter's capability
  vocabulary should be cross-referenced once a backend declares its first
  set.

## Rules the adapter enforces, and why the transport cares

These are checked on the client, before `train` is called, so the transport
can assume all of them. They come from `docs/distributed.md`, section 3.

1. **Known row counts.** A partition of unknown length cannot get a row
   offset. Persist or compute lengths first.
2. **No empty partitions.** An empty rank still answers every collective and
   contributes nothing to the histogram reduction, and the base score
   reduction divides by a weight sum.
3. **Partition index is row order.** Ranks are numbered by partition index,
   never by worker address.
4. **One rank per worker requires adjacent partitions.** Concatenation is
   order preserving only then. A worker holding partitions 0 and 3 is an
   error with a fix in the message, not a silent permutation.
   `one_rank_per_worker=False` gives each partition its own rank.
5. **One column schema everywhere.** Column count, column names, and every
   categorical column's category table, in the same order. Dask's unknown
   categories are refused with the `.categorize()` hint.
6. **A query never straddles a partition boundary** (ranking only).
   LambdaRank pairs rows within a query and a rank sees only its own rows.

The client also carries what the model format does not: the pandas category
labels behind the codes (`WorldPlan.schema.encoders()`) and the classifier's
class labels (`TrainingJob.label_classes`). The transport must encode labels
through exactly the order in `label_classes`, or `predict` will return the
wrong names for the right rows.

## Known gaps in this lane

- `DaskRuntime.partitions`, `map_partitions`, `_flat_keys`, and the category
  extraction have never run. Expect the first live cluster to change them.
  Three details there are guesses from the dask documentation and are the
  first things to check: `Client.who_has` reports keys as strings while
  `__dask_keys__` yields tuples (`_worker_of` tries both spellings, and a
  miss shows up as the "persist the collection first" error, which would be
  a misleading message if the lookup itself is wrong); a dask dataframe's
  uncategorized column carries placeholder categories rather than empty
  ones, so `.cat.known` is what `_categories` trusts; and
  `map_partitions` passes a `meta` for a dataframe while the partition
  predictor returns a numpy array, which may need wrapping in a Series or
  frame.
- Distributed prediction ships model bytes and parses them once per worker
  process, keyed by content digest (`_MODEL_CACHE`). The cache is unbounded
  and per process; `clear_model_cache()` empties it. Whether a real workload
  needs eviction is unknown.
- `pred_leaf` and `pred_contrib` are refused on a Dask collection: their
  output width depends on the model and the collection metadata cannot
  describe it here.
- Validation sets, early stopping, and callbacks are refused by `fit` with a
  message pointing at section 9. Wiring them up needs one more collective
  and an `evals_result_` reduced across ranks.
- Custom objectives ask for the `custom_objective` capability, which no
  design covers yet: a callable would have to be pickled to every worker and
  called identically per round.
- `sample_weight` reaches the backend as a key per partition
  (`PartitionMeta.weight_key`) and a `has_weight` flag, not as data. How the
  worker turns that key into buffers the Mojo trainer can read is the
  transport's decision, and it is the largest undesigned piece here.

## Checklist before any of the three estimators can be called supported

- [ ] A backend exists, is registered, and declares its capabilities.
- [ ] A global `BinMapper` is fit and shared before partitioning.
- [ ] `train` returns rank 0's model in the format `_mojoboost.save` writes.
- [ ] A two-worker local cluster trains a regressor and the model matches
      single-node training on the same rows, exactly, at world size 1 and
      within a stated tolerance above it.
- [ ] `DaskRuntime` runs against that cluster: worker addresses, partition
      lengths, column names, and category tables all come back correct.
- [ ] Distributed prediction on a dask dataframe and on a dask array both
      return the same values as local prediction.
- [ ] Worker loss during `fit` produces a stated, tested outcome.
- [ ] README and `docs/LIGHTGBM_PARITY.md` say what is supported, with the
      same care section 9 of `docs/distributed.md` uses.

Until every box is checked, the accurate statement is: mojoboost has a Dask
adapter contract and no distributed training.
