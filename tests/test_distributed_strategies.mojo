"""Feature-parallel and voting-parallel training over `LocalCollective`.

What is pinned:

- feature parallel reproduces the single-node tree bit for bit at any world
  size, because every rank holds every row and the election reproduces
  `find_best_split`'s scan order (distributed_strategies.mojo, "Determinism")
- a feature-parallel run of `train_distributed_run` equals single-node
  `train` tree for tree
- voting parallel trains, is deterministic, and reduces fewer histogram
  cells per node than data parallel on the same data
- the gates: a parallel mode at world size 1, serial at world size 2, and
  feature parallel over shards that do not all hold every row are refused

Nothing here crosses a process boundary; that is `transport_validated()`'s
claim to make, and it is still False.
"""

from std.math import isfinite
from std.testing import assert_equal, assert_true, TestSuite

from mojotrees.binning import BinnedMatrix, bin_equal_width
from mojotrees.boosting import SQUARED_ERROR, BoosterParams, train
from mojotrees.callback import no_callback
from mojotrees.collective import LocalCollective
from mojotrees.distributed import (
    DataShard,
    DistributedRunOptions,
    grow_tree_distributed,
    partition_rows,
    partition_values,
    train_distributed_run,
)
from mojotrees.distributed_strategies import (
    STRATEGY_DATA_PARALLEL,
    STRATEGY_FEATURE_PARALLEL,
    STRATEGY_SERIAL,
    STRATEGY_VOTING_PARALLEL,
)
from mojotrees.tree import Tree, TreeParams, grow_tree
from support import _uniform


def _dataset(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises -> BinnedMatrix:
    var feats = List[Float64](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            feats.append(_uniform(seed + UInt64(f * 10_007 + r)))
    return bin_equal_width(feats, n_rows, n_features, n_bins)


def _random_gradients(n_rows: Int, seed: UInt64) -> List[Float64]:
    var grad = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(2.0 * _uniform(seed + UInt64(r)) - 1.0)
    return grad^


def _random_hessians(n_rows: Int, seed: UInt64) -> List[Float64]:
    var hess = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        hess.append(0.25 + _uniform(seed + UInt64(r)))
    return hess^


def _targets(n_rows: Int) -> List[Float64]:
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(Float64((r * 5) % 8))
    return y^


def _assert_same_tree(a: Tree, b: Tree) raises:
    assert_equal(a.n_leaves, b.n_leaves)
    assert_equal(len(a.feature), len(b.feature))
    for i in range(len(a.feature)):
        assert_equal(a.feature[i], b.feature[i])
        assert_equal(a.threshold_bin[i], b.threshold_bin[i])
        assert_equal(a.left[i], b.left[i])
        assert_equal(a.right[i], b.right[i])
        assert_equal(a.value[i], b.value[i])
        assert_equal(a.split_gain[i], b.split_gain[i])


def _replicated(
    data: BinnedMatrix, y: List[Float64], world_size: Int
) raises -> List[DataShard]:
    """Every rank holds every row: the feature-parallel arrangement."""
    var shards = List[DataShard](capacity=world_size)
    for _ in range(world_size):
        shards.append(DataShard(data.copy(), y.copy()))
    return shards^


def _replicated_values(
    values: List[Float64], world_size: Int
) -> List[List[Float64]]:
    var out = List[List[Float64]](capacity=world_size)
    for _ in range(world_size):
        out.append(values.copy())
    return out^


def _options(strategy: Int, top_k: Int = 20) -> DistributedRunOptions:
    var options = DistributedRunOptions()
    options.tree_learner = strategy
    options.top_k = top_k
    return options^


def test_feature_parallel_tree_matches_grow_tree() raises:
    var n_rows = 400
    var data = _dataset(n_rows, 6, 16, 1)
    var grad = _random_gradients(n_rows, 77)
    var hess = _random_hessians(n_rows, 991)
    var params = TreeParams(15, 5, 1.0, 1e-3)
    var expected = grow_tree(data, grad, hess, params)
    assert_true(expected.n_leaves > 4, "the tree should actually branch")

    for world_size in [2, 3, 8]:
        var comm = LocalCollective(world_size)
        var got = grow_tree_distributed(
            _replicated(data, _targets(n_rows), world_size),
            _replicated_values(grad, world_size),
            _replicated_values(hess, world_size),
            params,
            comm,
            _options(STRATEGY_FEATURE_PARALLEL),
        )
        _assert_same_tree(expected, got)
        # The candidate all-gather is skipped in process (this process hosts
        # the world), so the only reductions issued are the once-per-tree
        # strategy agreement, and no histogram cell crosses the collective.
        assert_true(
            comm.elements < data.n_features * data.n_bins,
            "feature parallel must not reduce histogram cells",
        )


def test_feature_parallel_run_matches_train() raises:
    var n_rows = 256
    var data = _dataset(n_rows, 5, 12, 41)
    var y = _targets(n_rows)
    var params = BoosterParams(12, 0.1, TreeParams(8, 5, 1.0, 1e-3))
    var expected = train(data, y, SQUARED_ERROR, params)

    var comm = LocalCollective(2)
    var outcome = train_distributed_run(
        _replicated(data, y, 2),
        SQUARED_ERROR,
        params,
        comm,
        _options(STRATEGY_FEATURE_PARALLEL),
        no_callback,
    )
    assert_equal(outcome.model.base_score, expected.base_score)
    assert_equal(len(outcome.model.trees), len(expected.trees))
    for t in range(len(expected.trees)):
        _assert_same_tree(expected.trees[t], outcome.model.trees[t])


def test_voting_parallel_trains_and_reduces_less() raises:
    var n_rows = 512
    var data = _dataset(n_rows, 12, 16, 47)
    var y = _targets(n_rows)
    var params = BoosterParams(6, 0.1, TreeParams(8, 5, 1.0, 1e-3))

    var data_comm = LocalCollective(2)
    var data_outcome = train_distributed_run(
        partition_rows(data, y, 2),
        SQUARED_ERROR,
        params,
        data_comm,
        _options(STRATEGY_DATA_PARALLEL),
        no_callback,
    )
    var voting_comm = LocalCollective(2)
    var voting = train_distributed_run(
        partition_rows(data, y, 2),
        SQUARED_ERROR,
        params,
        voting_comm,
        _options(STRATEGY_VOTING_PARALLEL, top_k=3),
        no_callback,
    )
    assert_equal(len(voting.model.trees), 6)
    for r in range(n_rows):
        assert_true(
            isfinite(voting.model.predict_row(data, r)),
            "voting predictions must be finite",
        )
    assert_true(
        voting_comm.elements < data_comm.elements,
        "voting must reduce fewer cells than data parallel",
    )
    # Deterministic: the same run twice is the same model.
    var again_comm = LocalCollective(2)
    var again = train_distributed_run(
        partition_rows(data, y, 2),
        SQUARED_ERROR,
        params,
        again_comm,
        _options(STRATEGY_VOTING_PARALLEL, top_k=3),
        no_callback,
    )
    for t in range(len(voting.model.trees)):
        _assert_same_tree(voting.model.trees[t], again.model.trees[t])
    _ = data_outcome


def test_gates_refuse_impossible_worlds() raises:
    var n_rows = 64
    var data = _dataset(n_rows, 4, 8, 3)
    var y = _targets(n_rows)
    var params = TreeParams(4, 5, 1.0, 1e-3)
    var grad = _random_gradients(n_rows, 5)
    var hess = _random_hessians(n_rows, 6)

    # A parallel mode needs at least two ranks.
    var one = LocalCollective(1)
    var refused = False
    try:
        _ = grow_tree_distributed(
            _replicated(data, y, 1),
            _replicated_values(grad, 1),
            _replicated_values(hess, 1),
            params,
            one,
            _options(STRATEGY_FEATURE_PARALLEL),
        )
    except:
        refused = True
    assert_true(refused, "feature parallel at world size 1 must be refused")

    # Serial refuses a wider world.
    var two = LocalCollective(2)
    refused = False
    try:
        _ = grow_tree_distributed(
            partition_rows(data, y, 2),
            partition_values(grad, 2),
            partition_values(hess, 2),
            params,
            two,
            _options(STRATEGY_SERIAL),
        )
    except:
        refused = True
    assert_true(refused, "serial at world size 2 must be refused")

    # Feature parallel over row-partitioned shards is a contract violation.
    var three = LocalCollective(2)
    refused = False
    try:
        _ = grow_tree_distributed(
            partition_rows(data, y, 2),
            partition_values(grad, 2),
            partition_values(hess, 2),
            params,
            three,
            _options(STRATEGY_FEATURE_PARALLEL),
        )
    except:
        refused = True
    assert_true(refused, "feature parallel needs every rank to hold every row")

    # Serial at world size 1 is data parallel and works.
    var serial = LocalCollective(1)
    var tree = grow_tree_distributed(
        partition_rows(data, y, 1),
        partition_values(grad, 1),
        partition_values(hess, 1),
        params,
        serial,
        _options(STRATEGY_SERIAL),
    )
    _assert_same_tree(grow_tree(data, grad, hess, params), tree)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
