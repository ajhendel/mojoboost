"""The Dask adapter's contracts, against fakes rather than a cluster.

Nothing here starts a scheduler, a worker, or a Dask collection. What it
covers is everything `mojoboost.dask` is responsible for on its own side of
the backend protocol: import safety without dask installed, partition
metadata validation, the rank plan, categorical schema agreement, the
ranking partition rules, capability negotiation, model reference ownership,
and partition-local prediction against a model this file trains locally.

What it does not cover, and what no test in this repository covers, is
distributed training: no backend exists yet, so `DaskRuntime` and every
claim about what a worker does are unexercised. See
handoffs/task17_dask.md.
"""

import os
import pickle
import sys
import tempfile
import types

import numpy as np
import pytest

import mojoboost.dask as mbd
from mojoboost import (
    MojoBoostClassifier,
    MojoBoostRanker,
    MojoBoostRegressor,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCollection:
    """A stand-in for a Dask collection: it answers the collection
    protocol and holds its partitions as plain arrays."""

    def __init__(self, chunks):
        self.chunks_ = list(chunks)

    def __dask_graph__(self):
        return {}


class FakeRuntime:
    """The three-method runtime seam, over `FakeCollection`."""

    def __init__(self, metas=(), fail=False):
        self.metas = tuple(metas)
        self.fail = fail
        self.calls = []

    def partitions(self, X, y=None, sample_weight=None, query_ids=None):
        if self.fail:
            raise AssertionError("the runtime should not have been reached")
        self.calls.append(("partitions", sample_weight, query_ids))
        return self.metas

    def map_partitions(self, collection, fn, width=1):
        self.calls.append(("map_partitions", width))
        return [fn(chunk) for chunk in collection.chunks_]

    def is_collection(self, obj):
        return mbd.is_dask_collection(obj)


class FakeBackend:
    """A backend that hands back bytes it was given, and records the job."""

    def __init__(self, blob=b"", capabilities=("regression", "binary")):
        self.name = "fake"
        self.capabilities = frozenset(capabilities)
        self.blob = blob
        self.jobs = []

    def train(self, job):
        self.jobs.append(job)
        return mbd.BytesModelRef(self.blob, owner=job.plan.ranks[0].worker)


def meta(index, worker="tcp://a:1", n_rows=10, **kwargs):
    kwargs.setdefault("n_features", 4)
    return mbd.PartitionMeta(
        index=index, worker=worker, n_rows=n_rows, **kwargs
    )


@pytest.fixture(autouse=True)
def clean_backend(monkeypatch):
    """No registered backend and no environment hook leaks between
    tests."""
    monkeypatch.delenv("MOJOBOOST_DASK_BACKEND", raising=False)
    mbd.clear_backend()
    mbd.clear_model_cache()
    yield
    mbd.clear_backend()
    mbd.clear_model_cache()


def model_bytes(estimator):
    """The saved bytes of a fitted estimator, through the public save."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "model.mbst")
        estimator.save(path)
        with open(path, "rb") as handle:
            return handle.read()


@pytest.fixture(scope="module")
def local_regressor():
    gen = np.random.default_rng(5)
    X = gen.random((120, 4))
    y = X[:, 0] * 2.0 + X[:, 1]
    est = MojoBoostRegressor(n_estimators=5, num_leaves=7).fit(X, y)
    return est, X, y


@pytest.fixture(scope="module")
def local_binary():
    gen = np.random.default_rng(6)
    X = gen.random((120, 4))
    y = (X[:, 0] > 0.5).astype(np.int64)
    est = MojoBoostClassifier(n_estimators=5, num_leaves=7).fit(X, y)
    return est, X, y


@pytest.fixture(scope="module")
def local_multiclass():
    gen = np.random.default_rng(7)
    X = gen.random((120, 4))
    y = np.digitize(X[:, 0], [0.33, 0.66])
    est = MojoBoostClassifier(n_estimators=5, num_leaves=7).fit(X, y)
    return est, X, y


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_module_imports_without_touching_dask():
    # No module-level `import dask`: the names would be module globals.
    assert "dask" not in mbd.__dict__
    assert "distributed" not in mbd.__dict__
    assert isinstance(mbd.dask_available(), bool)


def test_is_dask_collection_needs_no_dask():
    assert mbd.is_dask_collection(FakeCollection([]))
    assert not mbd.is_dask_collection(np.zeros((2, 2)))
    assert not mbd.is_dask_collection([1, 2, 3])


def test_estimators_construct_and_clone_without_a_cluster():
    est = mbd.DaskMojoBoostRegressor(num_leaves=9, one_rank_per_worker=False)
    params = est.get_params()
    assert params["num_leaves"] == 9
    assert params["client"] is None
    assert params["one_rank_per_worker"] is False
    twin = type(est)(**params)
    assert twin.get_params() == params


# ---------------------------------------------------------------------------
# Partition metadata
# ---------------------------------------------------------------------------


def test_validate_partitions_sorts_by_index():
    parts = mbd.validate_partitions([meta(1), meta(0)])
    assert [part.index for part in parts] == [0, 1]


def test_validate_partitions_rejects_empty_collection():
    with pytest.raises(mbd.PartitionError, match="no partitions"):
        mbd.validate_partitions([])


def test_validate_partitions_rejects_duplicate_index():
    with pytest.raises(mbd.PartitionError, match="both claim index"):
        mbd.validate_partitions([meta(0), meta(0)])


def test_validate_partitions_rejects_a_gap_in_the_row_order():
    with pytest.raises(mbd.PartitionError, match="rows nobody owns"):
        mbd.validate_partitions([meta(0), meta(2)])


def test_validate_partitions_rejects_a_partition_with_no_worker():
    with pytest.raises(mbd.PartitionError, match="no worker address"):
        mbd.validate_partitions([meta(0, worker="")])


def test_validate_partitions_rejects_unknown_row_counts():
    with pytest.raises(mbd.PartitionError, match="unknown row count"):
        mbd.validate_partitions([meta(0, n_rows=None)])


def test_validate_partitions_rejects_an_empty_partition():
    with pytest.raises(mbd.PartitionError, match="is empty"):
        mbd.validate_partitions([meta(0, n_rows=0)])


def test_validate_partitions_rejects_a_feature_count_mismatch():
    with pytest.raises(mbd.SchemaError, match="4 features"):
        mbd.validate_partitions([meta(0), meta(1, n_features=5)])


def test_validate_partitions_rejects_disagreeing_column_names():
    left = meta(0, feature_names=("a", "b", "c", "d"))
    right = meta(1, feature_names=("a", "b", "c", "z"))
    with pytest.raises(mbd.SchemaError, match="column names"):
        mbd.validate_partitions([left, right])


def test_validate_partitions_rejects_names_that_do_not_cover_the_columns():
    with pytest.raises(mbd.SchemaError, match="column names for"):
        mbd.validate_partitions([meta(0, feature_names=("a", "b"))])


# ---------------------------------------------------------------------------
# Categorical schema
# ---------------------------------------------------------------------------


def test_merge_categorical_schema_agrees_across_partitions():
    columns = ((1, ("x", "y")),)
    schema = mbd.merge_categorical_schema(
        [meta(0, categories=columns), meta(1, categories=columns)]
    )
    assert schema.indices == (1,)
    assert schema.encoders() == {1: ["x", "y"]}
    assert bool(schema)


def test_merge_categorical_schema_rejects_unknown_categories():
    parts = [meta(0, categories=((2, None),))]
    with pytest.raises(mbd.SchemaError, match="categorize"):
        mbd.merge_categorical_schema(parts)


def test_merge_categorical_schema_rejects_a_reordered_category_table():
    left = meta(0, categories=((1, ("x", "y")),))
    right = meta(1, categories=((1, ("y", "x")),))
    with pytest.raises(mbd.SchemaError, match="disagree"):
        mbd.merge_categorical_schema([left, right])


def test_empty_schema_is_falsey():
    assert not mbd.CategoricalSchema()
    assert mbd.CategoricalSchema().encoders() == {}


# ---------------------------------------------------------------------------
# Ranks and the world
# ---------------------------------------------------------------------------


def test_plan_world_gives_each_worker_one_rank_in_row_order():
    parts = [
        meta(0, worker="tcp://a:1", n_rows=10),
        meta(1, worker="tcp://a:1", n_rows=5),
        meta(2, worker="tcp://b:1", n_rows=7),
    ]
    plan = mbd.plan_world(parts)
    assert plan.world_size == 2
    assert plan.n_rows == 22
    assert plan.workers == ("tcp://a:1", "tcp://b:1")
    assert plan.addresses_unique
    assert [rank.partitions for rank in plan.ranks] == [(0, 1), (2,)]
    assert [rank.row_offset for rank in plan.ranks] == [0, 15]
    assert [rank.n_rows for rank in plan.ranks] == [15, 7]


def test_plan_world_refuses_to_concatenate_non_adjacent_partitions():
    parts = [
        meta(0, worker="tcp://a:1"),
        meta(1, worker="tcp://b:1"),
        meta(2, worker="tcp://a:1"),
    ]
    with pytest.raises(mbd.PartitionError, match="reorder the training rows"):
        mbd.plan_world(parts)


def test_one_rank_per_partition_allows_a_shared_worker():
    parts = [
        meta(0, worker="tcp://a:1", n_rows=3),
        meta(1, worker="tcp://b:1", n_rows=4),
        meta(2, worker="tcp://a:1", n_rows=5),
    ]
    plan = mbd.plan_world(parts, one_rank_per_worker=False)
    assert plan.world_size == 3
    assert not plan.addresses_unique
    assert [rank.row_offset for rank in plan.ranks] == [0, 3, 7]
    assert [rank.rank for rank in plan.ranks] == [0, 1, 2]


def test_plan_world_carries_the_partitions_so_a_backend_can_fetch_them():
    parts = [
        meta(0, worker="tcp://a:1", key=("x", 0), label_key=("y", 0)),
        meta(1, worker="tcp://b:1", key=("x", 1), label_key=("y", 1)),
    ]
    plan = mbd.plan_world(parts)
    assert [part.key for part in plan.partitions] == [("x", 0), ("x", 1)]
    for rank in plan.ranks:
        owned = [plan.partitions[i] for i in rank.partitions]
        assert [part.label_key for part in owned] == [("y", rank.rank)]


def test_worker_lookup_accepts_a_stringified_key():
    key = ("frame", 3)
    assert mbd._worker_of({key: ["tcp://a:1"]}, key) == "tcp://a:1"
    assert mbd._worker_of({str(key): ["tcp://b:1"]}, key) == "tcp://b:1"
    assert mbd._worker_of({}, key) == ""


def test_plan_world_carries_the_schema_and_the_column_names():
    names = ("a", "b", "c", "d")
    parts = [
        meta(0, feature_names=names, categories=((0, ("u", "v")),)),
        meta(1, feature_names=names, categories=((0, ("u", "v")),)),
    ]
    plan = mbd.plan_world(parts)
    assert plan.feature_names == names
    assert plan.n_features == 4
    assert plan.schema.indices == (0,)


# ---------------------------------------------------------------------------
# Ranking partitions
# ---------------------------------------------------------------------------


def test_partition_group_counts_consecutive_queries():
    assert mbd.partition_group([7, 7, 8, 9, 9, 9]) == (2, 1, 3)


def test_partition_group_rejects_a_split_query():
    with pytest.raises(ValueError, match="unbroken run"):
        mbd.partition_group([1, 2, 1])


def test_query_partitioning_accepts_whole_queries():
    parts = [
        meta(0, n_rows=3, group=(2, 1), first_query=1, last_query=2),
        meta(1, n_rows=4, group=(4,), first_query=3, last_query=3),
    ]
    assert mbd.validate_query_partitioning(parts) == 3


def test_query_partitioning_rejects_a_query_across_a_boundary():
    parts = [
        meta(0, n_rows=3, group=(2, 1), first_query=1, last_query=2),
        meta(1, n_rows=4, group=(4,), first_query=2, last_query=2),
    ]
    with pytest.raises(mbd.PartitionError, match="spans partitions"):
        mbd.validate_query_partitioning(parts)


def test_query_partitioning_rejects_a_missing_group():
    with pytest.raises(mbd.PartitionError, match="no query group"):
        mbd.validate_query_partitioning([meta(0, n_rows=3)])


def test_query_partitioning_rejects_a_group_that_does_not_cover_the_rows():
    parts = [meta(0, n_rows=3, group=(2,), first_query=1, last_query=1)]
    with pytest.raises(mbd.PartitionError, match="sums to 2"):
        mbd.validate_query_partitioning(parts)


def test_query_partitioning_rejects_an_empty_query():
    parts = [meta(0, n_rows=3, group=(3, 0), first_query=1, last_query=2)]
    with pytest.raises(mbd.PartitionError, match="query with no rows"):
        mbd.validate_query_partitioning(parts)


# ---------------------------------------------------------------------------
# The backend registry
# ---------------------------------------------------------------------------


def test_no_backend_is_registered_by_default():
    assert not mbd.backend_registered()
    with pytest.raises(mbd.DistributedNotAvailable, match="not finished"):
        mbd.get_backend()


def test_register_and_clear_a_backend():
    backend = FakeBackend()
    assert mbd.register_backend(backend) is backend
    assert mbd.backend_registered()
    assert mbd.get_backend() is backend
    mbd.clear_backend()
    assert not mbd.backend_registered()


def test_registering_twice_needs_replace():
    mbd.register_backend(FakeBackend())
    with pytest.raises(mbd.DistributedNotAvailable, match="already"):
        mbd.register_backend(FakeBackend())
    other = FakeBackend()
    assert mbd.register_backend(other, replace=True) is other


def test_registration_rejects_an_incomplete_backend():
    class NoTrain:
        name = "x"
        capabilities = frozenset()

    with pytest.raises(mbd.DistributedNotAvailable, match="'train'"):
        mbd.register_backend(NoTrain())


def test_registration_rejects_an_unknown_capability():
    with pytest.raises(mbd.DistributedNotAvailable, match="unknown"):
        mbd.register_backend(FakeBackend(capabilities=("teleportation",)))


def test_environment_hook_resolves_a_backend(monkeypatch):
    module = types.ModuleType("fake_mojoboost_backend")
    module.BACKEND = FakeBackend()
    monkeypatch.setitem(sys.modules, "fake_mojoboost_backend", module)
    monkeypatch.setenv(
        "MOJOBOOST_DASK_BACKEND", "fake_mojoboost_backend:BACKEND"
    )
    assert mbd.get_backend() is module.BACKEND


def test_environment_hook_instantiates_a_class(monkeypatch):
    module = types.ModuleType("fake_mojoboost_backend_cls")
    module.Backend = FakeBackend
    monkeypatch.setitem(sys.modules, "fake_mojoboost_backend_cls", module)
    monkeypatch.setenv(
        "MOJOBOOST_DASK_BACKEND", "fake_mojoboost_backend_cls:Backend"
    )
    assert isinstance(mbd.get_backend(), FakeBackend)


@pytest.mark.parametrize(
    "spec, message",
    [
        ("nonsense", "package.module:attribute"),
        ("mojoboost.dask:no_such_attribute", "no 'no_such_attribute'"),
        ("mojoboost_not_a_module:x", "cannot be imported"),
    ],
)
def test_environment_hook_reports_a_bad_spec(monkeypatch, spec, message):
    monkeypatch.setenv("MOJOBOOST_DASK_BACKEND", spec)
    with pytest.raises(mbd.DistributedNotAvailable, match=message):
        mbd.get_backend()


def test_require_capabilities_names_what_is_missing():
    backend = FakeBackend(capabilities=("regression",))
    assert mbd.require_capabilities(backend, {"regression"})
    with pytest.raises(mbd.UnsupportedByBackend) as excinfo:
        mbd.require_capabilities(backend, {"regression", "ranking"})
    assert "ranking" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Model reference ownership
# ---------------------------------------------------------------------------


def test_take_model_bytes_releases_the_reference():
    ref = mbd.BytesModelRef(b"model", owner="tcp://a:1")
    assert mbd.take_model_bytes(ref) == b"model"
    assert ref.released
    with pytest.raises(mbd.ModelOwnershipError, match="already released"):
        ref.result()


def test_take_model_bytes_releases_even_when_the_gather_fails():
    class Broken:
        owner = "tcp://a:1"
        released = False

        def result(self, timeout=None):
            raise TimeoutError("worker went away")

        def release(self):
            type(self).released = True

    Broken.released = False
    with pytest.raises(TimeoutError):
        mbd.take_model_bytes(Broken(), timeout=1.0)
    assert Broken.released


def test_take_model_bytes_rejects_a_non_bytes_model():
    with pytest.raises(mbd.ModelOwnershipError, match="not the bytes"):
        mbd.take_model_bytes(mbd.BytesModelRef("a string"))


def test_take_model_bytes_rejects_an_empty_model():
    with pytest.raises(mbd.ModelOwnershipError, match="no bytes"):
        mbd.take_model_bytes(mbd.BytesModelRef(b""))


def test_take_model_bytes_accepts_a_bytearray():
    assert mbd.take_model_bytes(mbd.BytesModelRef(bytearray(b"ab"))) == b"ab"


# ---------------------------------------------------------------------------
# Fit, against the fakes
# ---------------------------------------------------------------------------


def test_fit_fails_before_the_cluster_when_no_backend_is_registered():
    runtime = FakeRuntime(fail=True)
    est = mbd.DaskMojoBoostRegressor()
    with pytest.raises(mbd.DistributedNotAvailable):
        est.fit(FakeCollection([]), FakeCollection([]), runtime=runtime)
    assert runtime.calls == []


def test_regressor_fit_plans_the_world_and_installs_the_model(
    local_regressor,
):
    local, X, _ = local_regressor
    blob = model_bytes(local)
    backend = FakeBackend(blob=blob, capabilities=("regression",))
    mbd.register_backend(backend)
    parts = [
        meta(0, worker="tcp://a:1", n_rows=60),
        meta(1, worker="tcp://b:1", n_rows=60),
    ]
    runtime = FakeRuntime(parts)
    est = mbd.DaskMojoBoostRegressor(num_leaves=7, n_estimators=5)
    fitted = est.fit(
        FakeCollection([]), FakeCollection([]), runtime=runtime
    )
    assert fitted is est

    job = backend.jobs[0]
    assert job.protocol_version == mbd.BACKEND_PROTOCOL_VERSION
    assert job.objective == "regression"
    assert job.plan.world_size == 2
    assert job.plan.n_rows == 120
    assert job.ranking is False
    assert job.label_classes is None
    # Cluster settings are not hyperparameters.
    assert "client" not in job.params
    assert "one_rank_per_worker" not in job.params
    assert job.params["num_leaves"] == 7

    assert est.n_features_in_ == 4
    assert est.best_iteration_ == local.best_iteration_
    assert np.allclose(est.predict(X), local.predict(X))


def test_fit_refuses_validation_arguments(local_regressor):
    local, _, _ = local_regressor
    mbd.register_backend(
        FakeBackend(blob=model_bytes(local), capabilities=("regression",))
    )
    est = mbd.DaskMojoBoostRegressor()
    with pytest.raises(TypeError, match="early_stopping_rounds"):
        est.fit(
            FakeCollection([]),
            FakeCollection([]),
            runtime=FakeRuntime([meta(0)]),
            early_stopping_rounds=5,
        )


def test_fit_asks_for_the_capabilities_the_settings_need(local_regressor):
    local, _, _ = local_regressor
    backend = FakeBackend(
        blob=model_bytes(local), capabilities=("regression",)
    )
    mbd.register_backend(backend)
    est = mbd.DaskMojoBoostRegressor(bagging_fraction=0.5)
    with pytest.raises(mbd.UnsupportedByBackend, match="bagging"):
        est.fit(
            FakeCollection([]),
            FakeCollection([]),
            runtime=FakeRuntime([meta(0)]),
        )


def test_fit_asks_for_quantile_leaf_renewal(local_regressor):
    local, _, _ = local_regressor
    mbd.register_backend(
        FakeBackend(blob=model_bytes(local), capabilities=("regression",))
    )
    est = mbd.DaskMojoBoostRegressor(objective="quantile")
    with pytest.raises(mbd.UnsupportedByBackend, match="quantile_l1"):
        est.fit(
            FakeCollection([]),
            FakeCollection([]),
            runtime=FakeRuntime([meta(0)]),
        )


def test_fit_asks_for_categorical_support(local_regressor):
    local, _, _ = local_regressor
    mbd.register_backend(
        FakeBackend(blob=model_bytes(local), capabilities=("regression",))
    )
    est = mbd.DaskMojoBoostRegressor()
    parts = [meta(0, categories=((1, ("x", "y")),))]
    with pytest.raises(mbd.UnsupportedByBackend, match="categorical"):
        est.fit(
            FakeCollection([]), FakeCollection([]), runtime=FakeRuntime(parts)
        )


def test_classifier_fit_takes_the_global_class_list(local_binary):
    local, X, _ = local_binary
    backend = FakeBackend(
        blob=model_bytes(local), capabilities=("binary",)
    )
    mbd.register_backend(backend)
    est = mbd.DaskMojoBoostClassifier(n_estimators=5, num_leaves=7)
    est.fit(
        FakeCollection([]),
        FakeCollection([]),
        classes=[0, 1],
        runtime=FakeRuntime([meta(0, n_rows=120)]),
    )
    assert backend.jobs[0].objective == "binary"
    assert backend.jobs[0].label_classes == (0, 1)
    assert list(est.classes_) == [0, 1]
    assert np.array_equal(est.predict(X), local.predict(X))


def test_classifier_fit_rejects_a_degenerate_class_list():
    mbd.register_backend(FakeBackend())
    est = mbd.DaskMojoBoostClassifier()
    with pytest.raises(ValueError, match="at least 2 classes"):
        est.fit(FakeCollection([]), FakeCollection([]), classes=[1])
    with pytest.raises(ValueError, match="repeats"):
        est.fit(FakeCollection([]), FakeCollection([]), classes=[1, 1])


def test_classifier_asks_for_multiclass_when_the_labels_say_so():
    mbd.register_backend(FakeBackend(capabilities=("binary",)))
    est = mbd.DaskMojoBoostClassifier()
    with pytest.raises(mbd.UnsupportedByBackend, match="multiclass"):
        est.fit(
            FakeCollection([]),
            FakeCollection([]),
            classes=[0, 1, 2],
            runtime=FakeRuntime([meta(0)]),
        )


def test_classifier_refuses_a_model_with_the_wrong_class_count(
    local_multiclass,
):
    local, _, _ = local_multiclass
    mbd.register_backend(
        FakeBackend(
            blob=model_bytes(local), capabilities=("binary", "multiclass")
        )
    )
    est = mbd.DaskMojoBoostClassifier()
    with pytest.raises(mbd.ModelOwnershipError, match="classes"):
        est.fit(
            FakeCollection([]),
            FakeCollection([]),
            classes=["a", "b", "c", "d"],
            runtime=FakeRuntime([meta(0)]),
        )


def test_classifier_carries_string_labels_through_a_distributed_fit(
    local_multiclass,
):
    local, X, _ = local_multiclass
    mbd.register_backend(
        FakeBackend(blob=model_bytes(local), capabilities=("multiclass",))
    )
    est = mbd.DaskMojoBoostClassifier()
    est.fit(
        FakeCollection([]),
        FakeCollection([]),
        classes=["a", "b", "c"],
        runtime=FakeRuntime([meta(0, n_rows=120)]),
    )
    assert list(est.classes_) == ["a", "b", "c"]
    codes = np.asarray(local.predict(X))
    assert np.array_equal(
        est.predict(X), np.asarray(["a", "b", "c"])[codes]
    )


def test_ranker_fit_needs_query_ids():
    mbd.register_backend(FakeBackend(capabilities=("ranking",)))
    est = mbd.DaskMojoBoostRanker()
    with pytest.raises(ValueError, match="query_ids"):
        est.fit(FakeCollection([]), FakeCollection([]), query_ids=None)


def test_ranker_fit_checks_the_query_partitioning():
    mbd.register_backend(FakeBackend(capabilities=("ranking",)))
    parts = [
        meta(0, worker="tcp://a:1", n_rows=3, group=(3,),
             first_query=1, last_query=1),
        meta(1, worker="tcp://b:1", n_rows=2, group=(2,),
             first_query=1, last_query=1),
    ]
    est = mbd.DaskMojoBoostRanker()
    with pytest.raises(mbd.PartitionError, match="spans partitions"):
        est.fit(
            FakeCollection([]),
            FakeCollection([]),
            query_ids=[[1, 1, 1], [1, 1]],
            runtime=FakeRuntime(parts),
        )


def test_ranker_fit_passes_the_ranking_objective():
    gen = np.random.default_rng(9)
    X = gen.random((40, 4))
    y = gen.integers(0, 3, size=40)
    local = MojoBoostRanker(n_estimators=3, num_leaves=5).fit(
        X, y, group=[10] * 4
    )
    mbd.register_backend(
        FakeBackend(blob=model_bytes(local), capabilities=("ranking",))
    )
    parts = [
        meta(0, worker="tcp://a:1", n_rows=20, group=(10, 10),
             first_query=1, last_query=2),
        meta(1, worker="tcp://b:1", n_rows=20, group=(10, 10),
             first_query=3, last_query=4),
    ]
    est = mbd.DaskMojoBoostRanker(n_estimators=3, num_leaves=5)
    est.fit(
        FakeCollection([]),
        FakeCollection([]),
        query_ids=[[1] * 10 + [2] * 10, [3] * 10 + [4] * 10],
        runtime=FakeRuntime(parts),
    )
    job = mbd.get_backend().jobs[0]
    assert job.objective == "lambdarank"
    assert job.ranking is True
    assert np.allclose(est.predict(X), local.predict(X))


# ---------------------------------------------------------------------------
# Distributed prediction
# ---------------------------------------------------------------------------


def _fitted_dask_regressor(local, parts=None):
    mbd.register_backend(
        FakeBackend(blob=model_bytes(local), capabilities=("regression",))
    )
    est = mbd.DaskMojoBoostRegressor()
    est.fit(
        FakeCollection([]),
        FakeCollection([]),
        runtime=FakeRuntime(parts or [meta(0, n_rows=120)]),
    )
    return est


def test_predict_maps_over_partitions(local_regressor):
    local, X, _ = local_regressor
    est = _fitted_dask_regressor(local)
    collection = FakeCollection([X[:60], X[60:]])
    out = est.predict(collection)
    assert isinstance(out, list) and len(out) == 2
    assert np.allclose(np.concatenate(out), local.predict(X))


def test_predict_on_an_ordinary_array_stays_local(local_regressor):
    local, X, _ = local_regressor
    est = _fitted_dask_regressor(local)
    assert np.allclose(est.predict(X), local.predict(X))


def test_predict_forwards_keyword_options(local_regressor):
    local, X, _ = local_regressor
    est = _fitted_dask_regressor(local)
    out = est.predict(FakeCollection([X]), num_iteration=2)
    assert np.allclose(out[0], local.predict(X, num_iteration=2))


def test_predict_refuses_positional_options(local_regressor):
    local, X, _ = local_regressor
    est = _fitted_dask_regressor(local)
    with pytest.raises(TypeError, match="by keyword"):
        est.predict(FakeCollection([X]), True)


@pytest.mark.parametrize("flag", ["pred_leaf", "pred_contrib"])
def test_predict_refuses_matrix_output_on_a_collection(local_regressor, flag):
    local, X, _ = local_regressor
    est = _fitted_dask_regressor(local)
    with pytest.raises(ValueError, match=flag):
        est.predict(FakeCollection([X]), **{flag: True})


def test_predict_proba_asks_for_a_column_per_class(local_binary):
    local, X, _ = local_binary
    mbd.register_backend(
        FakeBackend(blob=model_bytes(local), capabilities=("binary",))
    )
    est = mbd.DaskMojoBoostClassifier()
    runtime = FakeRuntime([meta(0, n_rows=120)])
    est.fit(
        FakeCollection([]), FakeCollection([]), classes=[0, 1],
        runtime=runtime,
    )
    out = est.predict_proba(FakeCollection([X]))
    assert ("map_partitions", 2) in runtime.calls
    assert np.allclose(out[0], local.predict_proba(X))


def test_prediction_needs_a_fitted_model():
    from mojoboost import NotFittedError

    est = mbd.DaskMojoBoostRegressor()
    with pytest.raises(NotFittedError):
        est.predict(FakeCollection([np.zeros((2, 4))]))


def test_the_worker_model_cache_parses_a_model_once(local_regressor):
    local, X, _ = local_regressor
    blob = model_bytes(local)
    first = mbd._PartitionPredictor(blob=blob, kind="regressor",
                                    multiclass=False)
    second = mbd._PartitionPredictor(blob=bytes(blob), kind="regressor",
                                     multiclass=False)
    parsed = first.model()
    assert second.model() is parsed
    assert np.allclose(first(X), local.predict(X))
    mbd.clear_model_cache()
    assert first.model() is not parsed


# ---------------------------------------------------------------------------
# Handing the model back
# ---------------------------------------------------------------------------


def test_to_local_returns_a_single_machine_estimator(local_regressor):
    local, X, _ = local_regressor
    est = _fitted_dask_regressor(local)
    plain = est.to_local()
    assert type(plain) is MojoBoostRegressor
    assert np.allclose(plain.predict(X), local.predict(X))


def test_a_fitted_estimator_pickles_without_its_cluster(local_regressor):
    local, X, _ = local_regressor
    est = _fitted_dask_regressor(local)
    est.client = "a client that cannot pickle"
    twin = pickle.loads(pickle.dumps(est))
    assert twin.client is None
    assert twin._runtime is None
    assert np.allclose(twin.predict(X), local.predict(X))
