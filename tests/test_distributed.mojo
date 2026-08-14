"""Data-parallel distributed training: equivalence, determinism, and the
collective contract.

Everything here runs in one process, so these tests validate the algorithm
and not a transport. The equivalence tests are what keep the distributed
grower pinned to the single-node one as the latter evolves.

Two claims are tested separately because they are not the same claim:

- at world size 1 the distributed path is bit-identical to the single-node
  path on any data, since a one-shard reduction adds nothing to zero
- at larger world sizes it is bit-identical whenever the histogram sums are
  exactly representable, and agrees to within accumulated rounding
  otherwise, because sharding regroups a sum that is not associative

Counts are integers and reduce exactly, so count-derived decisions are exact
at any world size on any data.

`_PeerCollective` is a second `Collective` implementation that folds in a
simulated remote rank. It exists so the cross-rank failure paths, which one
process full of healthy ranks can never reach on its own, are actually
exercised. It is also the shape of the conformance suite a real transport
has to pass.
"""

from std.testing import assert_almost_equal, assert_equal, assert_true, TestSuite

from mojoboost.binning import BinnedMatrix, bin_equal_width, fit_bins
from mojoboost.boosting import (
    BINARY_LOGISTIC,
    HUBER,
    L1,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    BoosterParams,
    train,
)
from mojoboost.collective import (
    STATUS_OK,
    STATUS_SHAPE_MISMATCH,
    Collective,
    LocalCollective,
    add_into_f64,
    agree_equal_ints,
    agree_status,
    status_message,
    zeros_int,
)
from mojoboost.distributed import (
    DataShard,
    grow_tree_distributed,
    partition_rows,
    partition_values,
    train_distributed,
)
from mojoboost.serialize import load_model, save_model
from mojoboost.model import Model
from mojoboost.tree import Tree, TreeParams, grow_tree


comptime _TMP_PATH = "./.test_distributed_roundtrip.tmp"


def _splitmix64(state: UInt64) -> UInt64:
    var z = state + 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _uniform(counter: UInt64) -> Float64:
    return Float64(_splitmix64(counter) >> 11) * (1.0 / 9007199254740992.0)


@fieldwise_init
struct _PeerCollective(Collective, Copyable, Movable):
    """A world where this process hosts every rank below `n_local` and one
    simulated remote rank contributes `peer_max` to maximum reductions.

    Sum reductions are the identity, as if the remote rank contributed
    zeros, which is all the validation paths under test need. It exists to
    reach the cross-rank disagreement and remote-failure branches that a
    process full of healthy ranks cannot reach.
    """

    var n: Int
    var n_local_count: Int
    var peer_max: List[Int]

    def world_size(self) -> Int:
        return self.n

    def rank(self) -> Int:
        return 0

    def n_local_ranks(self) -> Int:
        return self.n_local_count

    def local_rank(self, index: Int) -> Int:
        return index

    def allreduce_sum_f64(mut self, mut buf: List[Float64]) raises:
        pass

    def allreduce_sum_int(mut self, mut buf: List[Int]) raises:
        pass

    def allreduce_max_int(mut self, mut buf: List[Int]) raises:
        for i in range(len(buf)):
            if i < len(self.peer_max) and self.peer_max[i] > buf[i]:
                buf[i] = self.peer_max[i]

    def barrier(mut self) raises:
        pass


def _dataset(
    n_rows: Int, n_features: Int, n_bins: Int, seed: UInt64
) raises -> BinnedMatrix:
    var feats = List[Float64](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            feats.append(_uniform(seed + UInt64(f * 10_007 + r)))
    return bin_equal_width(feats, n_rows, n_features, n_bins)


def _exact_gradients(n_rows: Int) -> List[Float64]:
    """Small integers, so every histogram sum is exact in a double and
    regrouping at shard boundaries changes nothing."""
    var grad = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        grad.append(Float64((r * 13) % 7) - 3.0)
    return grad^


def _ones(n_rows: Int) -> List[Float64]:
    var hess = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        hess.append(1.0)
    return hess^


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


def _grow_sharded(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    world_size: Int,
) raises -> Tree:
    var target = List[Float64](capacity=data.n_rows)
    for r in range(data.n_rows):
        target.append(0.0)
    var shards = partition_rows(data, target, world_size)
    var comm = LocalCollective(world_size)
    return grow_tree_distributed(
        shards,
        partition_values(grad, world_size),
        partition_values(hess, world_size),
        params,
        comm,
    )


def test_world_size_one_matches_grow_tree() raises:
    """One shard reduces to adding a histogram to zero, so this holds bit for
    bit on arbitrary data. It is the test that catches the distributed grower
    drifting from the single-node one."""
    var data = _dataset(400, 6, 16, 1)
    var grad = _random_gradients(400, 77)
    var hess = _random_hessians(400, 991)
    var params = TreeParams(15, 5, 1.0, 1e-3)

    var expected = grow_tree(data, grad, hess, params)
    var got = _grow_sharded(data, grad, hess, params, 1)
    assert_true(expected.n_leaves > 4, "the tree should actually branch")
    _assert_same_tree(expected, got)


def test_shard_count_invariance_on_exact_data() raises:
    """With exactly representable gradients every world size gives the same
    tree, including one that does not divide the row count evenly and one
    with more ranks than rows."""
    var n_rows = 300
    var data = _dataset(n_rows, 5, 12, 5)
    var grad = _exact_gradients(n_rows)
    var hess = _ones(n_rows)
    var params = TreeParams(12, 4, 1.0, 1e-3)

    var expected = grow_tree(data, grad, hess, params)
    assert_true(expected.n_leaves > 4, "the tree should actually branch")
    for world_size in [1, 2, 3, 4, 5, 7, 16]:
        _assert_same_tree(
            expected, _grow_sharded(data, grad, hess, params, world_size)
        )


def test_empty_shards_take_part() raises:
    """A rank with no rows contributes zeros and must still call every
    collective. More ranks than rows forces the case."""
    var n_rows = 9
    var data = _dataset(n_rows, 3, 8, 11)
    var grad = _exact_gradients(n_rows)
    var hess = _ones(n_rows)
    var params = TreeParams(6, 1, 1.0, 1e-3)

    var expected = grow_tree(data, grad, hess, params)
    var got = _grow_sharded(data, grad, hess, params, n_rows + 4)
    _assert_same_tree(expected, got)


def test_repeat_growth_is_bit_identical() raises:
    var data = _dataset(200, 4, 10, 21)
    var grad = _random_gradients(200, 3)
    var hess = _random_hessians(200, 4)
    var params = TreeParams(10, 5, 1.0, 1e-3)
    var first = _grow_sharded(data, grad, hess, params, 3)
    var second = _grow_sharded(data, grad, hess, params, 3)
    _assert_same_tree(first, second)


def test_allreduce_schedule_is_one_per_node() raises:
    """The communication schedule is the cost model in docs/distributed.md:
    two reductions to agree on statuses and configuration, then one histogram
    per tree node, which this prototype sends as three typed buffers."""
    var n_rows = 200
    var data = _dataset(n_rows, 4, 10, 33)
    var grad = _exact_gradients(n_rows)
    var hess = _ones(n_rows)
    var params = TreeParams(8, 5, 1.0, 1e-3)

    var target = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        target.append(0.0)
    var shards = partition_rows(data, target, 4)
    var comm = LocalCollective(4)
    var tree = grow_tree_distributed(
        shards,
        partition_values(grad, 4),
        partition_values(hess, 4),
        params,
        comm,
    )
    assert_equal(comm.calls, 2 + 3 * tree.n_leaves)
    assert_equal(
        comm.elements,
        4 + 2 * 4 + 3 * tree.n_leaves * data.n_features * data.n_bins,
    )


def _regression_targets(n_rows: Int) raises -> List[Float64]:
    """Small integers over a power-of-two row count, so the base score is
    exact and round one is bit-identical at any world size."""
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(Float64((r * 5) % 8))
    return y^


def test_train_world_size_one_matches_train() raises:
    var n_rows = 256
    var data = _dataset(n_rows, 5, 12, 41)
    var y = _regression_targets(n_rows)
    var params = BoosterParams(12, 0.1, TreeParams(8, 5, 1.0, 1e-3))

    var expected = train(data, y, SQUARED_ERROR, params)
    var comm = LocalCollective(1)
    var got = train_distributed(
        partition_rows(data, y, 1), SQUARED_ERROR, params, comm
    )
    assert_equal(got.base_score, expected.base_score)
    assert_equal(len(got.trees), len(expected.trees))
    assert_true(len(got.trees) == 12, "every round should produce a tree")
    for t in range(len(expected.trees)):
        _assert_same_tree(expected.trees[t], got.trees[t])


def test_train_binary_world_size_one_matches_train() raises:
    var n_rows = 256
    var data = _dataset(n_rows, 5, 12, 43)
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(1.0 if (r * 7) % 5 < 2 else 0.0)
    var params = BoosterParams(10, 0.1, TreeParams(8, 5, 1.0, 1e-3))

    var expected = train(data, y, BINARY_LOGISTIC, params)
    var comm = LocalCollective(1)
    var got = train_distributed(
        partition_rows(data, y, 1), BINARY_LOGISTIC, params, comm
    )
    assert_equal(got.base_score, expected.base_score)
    assert_equal(len(got.trees), len(expected.trees))
    for t in range(len(expected.trees)):
        _assert_same_tree(expected.trees[t], got.trees[t])


def test_train_sharded_agrees_with_single_node() raises:
    """Past the first round the gradients are arbitrary doubles, so the
    shard-boundary regrouping is visible in the last bits and the claim is a
    tolerance, not bit-identity. The base score stays exact here because the
    targets are small integers over a power-of-two row count."""
    var n_rows = 256
    var data = _dataset(n_rows, 5, 12, 47)
    var y = _regression_targets(n_rows)
    var params = BoosterParams(15, 0.1, TreeParams(8, 5, 1.0, 1e-3))

    var expected = train(data, y, SQUARED_ERROR, params)
    for world_size in [2, 3, 4, 8]:
        var comm = LocalCollective(world_size)
        var got = train_distributed(
            partition_rows(data, y, world_size), SQUARED_ERROR, params, comm
        )
        assert_equal(got.base_score, expected.base_score)
        assert_equal(len(got.trees), len(expected.trees))
        for r in range(n_rows):
            assert_almost_equal(
                got.predict_row(data, r),
                expected.predict_row(data, r),
                atol=1e-9,
            )


def test_train_poisson_and_huber_are_supported() raises:
    var n_rows = 128
    var data = _dataset(n_rows, 4, 10, 53)
    var y = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        y.append(Float64(r % 4))
    var params = BoosterParams(8, 0.1, TreeParams(6, 4, 1.0, 1e-3))

    for objective in [POISSON, HUBER]:
        var expected = train(data, y, objective, params)
        var comm = LocalCollective(3)
        var got = train_distributed(
            partition_rows(data, y, 3), objective, params, comm
        )
        assert_equal(len(got.trees), len(expected.trees))
        for r in range(n_rows):
            assert_almost_equal(
                got.predict_row(data, r),
                expected.predict_row(data, r),
                atol=1e-8,
            )


def test_weights_are_global_not_per_shard() raises:
    """A shard whose weights are all zero is legitimate as long as some other
    shard carries positive weight: the positive-sum requirement is global."""
    var n_rows = 128
    var data = _dataset(n_rows, 4, 10, 59)
    var y = _regression_targets(n_rows)
    var weight = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        weight.append(0.0 if r < n_rows // 4 else 1.0)
    var params = BoosterParams(6, 0.1, TreeParams(6, 4, 1.0, 1e-3))

    var expected = train(data, y, SQUARED_ERROR, params, weight)
    var comm = LocalCollective(4)
    var got = train_distributed(
        partition_rows(data, y, 4, weight), SQUARED_ERROR, params, comm
    )
    assert_equal(got.base_score, expected.base_score)
    assert_equal(len(got.trees), len(expected.trees))
    for r in range(n_rows):
        assert_almost_equal(
            got.predict_row(data, r), expected.predict_row(data, r), atol=1e-9
        )


def test_all_zero_weights_everywhere_raise() raises:
    var n_rows = 32
    var data = _dataset(n_rows, 3, 8, 61)
    var y = _regression_targets(n_rows)
    var weight = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        weight.append(0.0)
    var params = BoosterParams(3, 0.1, TreeParams(4, 2, 1.0, 1e-3))

    var message = String("")
    var comm = LocalCollective(2)
    try:
        _ = train_distributed(
            partition_rows(data, y, 2, weight), SQUARED_ERROR, params, comm
        )
    except e:
        message = String(e)
    assert_true(
        message.find("positive sum") >= 0,
        "expected a global positive-weight-sum error, got: " + message,
    )


def test_distributed_model_round_trips_through_serialization() raises:
    """Sharding changes how histograms are summed and nothing about the model,
    so a distributed model is an ordinary model on disk. No serialization
    format change was needed for this work, and this is what pins that."""
    var n_rows = 128
    var n_features = 4
    var feats = List[Float64](capacity=n_rows * n_features)
    for f in range(n_features):
        for r in range(n_rows):
            feats.append(_uniform(UInt64(f * 10_007 + r) + 67))
    var mapper = fit_bins(feats, n_rows, n_features, 10)
    var data = mapper.transform(feats, n_rows)
    var y = _regression_targets(n_rows)
    var params = BoosterParams(5, 0.1, TreeParams(6, 4, 1.0, 1e-3))

    var comm = LocalCollective(3)
    var booster = train_distributed(
        partition_rows(data, y, 3), SQUARED_ERROR, params, comm
    )
    var model = Model(mapper^, booster^)
    save_model(model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    assert_equal(loaded.booster.base_score, model.booster.base_score)
    assert_equal(len(loaded.booster.trees), len(model.booster.trees))
    for r in range(n_rows):
        var row = List[Float64](capacity=n_features)
        for f in range(n_features):
            row.append(feats[f * n_rows + r])
        assert_equal(loaded.predict(row), model.predict(row))


def test_quantile_and_l1_are_refused() raises:
    var n_rows = 32
    var data = _dataset(n_rows, 3, 8, 71)
    var y = _regression_targets(n_rows)
    var params = BoosterParams(3, 0.1, TreeParams(4, 2, 1.0, 1e-3))

    for objective in [QUANTILE, L1]:
        var message = String("")
        var comm = LocalCollective(2)
        try:
            _ = train_distributed(
                partition_rows(data, y, 2), objective, params, comm
            )
        except e:
            message = String(e)
        assert_true(
            message.find("quantile or L1") >= 0,
            "expected the objective to be refused, got: " + message,
        )


def test_feature_subsampling_is_refused() raises:
    """Unsupported tree parameters are rejected rather than ignored, so an
    unsupported setting is an error and not a quietly different model."""
    var n_rows = 64
    var data = _dataset(n_rows, 4, 8, 73)
    var grad = _exact_gradients(n_rows)
    var hess = _ones(n_rows)
    var params = TreeParams(6, 2, 1.0, 1e-3, 0.0)
    params.feature_fraction = 0.5

    var message = String("")
    try:
        _ = _grow_sharded(data, grad, hess, params, 2)
    except e:
        message = String(e)
    assert_true(
        message.find("feature_fraction") >= 0,
        "expected feature_fraction to be refused, got: " + message,
    )


def test_shape_failure_names_the_lowest_failing_rank() raises:
    """Validation is collective: ranks that were themselves fine raise the
    same error as the ones that were not, so nobody is left blocking in a
    collective a failed rank will never call."""
    var n_rows = 60
    var data = _dataset(n_rows, 3, 8, 79)
    var grad = _exact_gradients(n_rows)
    var hess = _ones(n_rows)
    var params = TreeParams(6, 2, 1.0, 1e-3)

    var target = List[Float64](capacity=n_rows)
    for _ in range(n_rows):
        target.append(0.0)
    var shards = partition_rows(data, target, 4)
    var grads = partition_values(grad, 4)
    var hesses = partition_values(hess, 4)
    # Ranks 1 and 2 are handed the wrong number of gradients.
    grads[1] = List[Float64]()
    grads[2] = List[Float64]()

    var message = String("")
    var comm = LocalCollective(4)
    try:
        _ = grow_tree_distributed(shards, grads, hesses, params, comm)
    except e:
        message = String(e)
    assert_true(
        message.find("rank 1") >= 0,
        "expected the lowest failing rank to be named, got: " + message,
    )
    assert_true(
        message.find(status_message(STATUS_SHAPE_MISMATCH)) >= 0,
        "expected the reason in the message, got: " + message,
    )


def test_remote_rank_failure_stops_every_rank() raises:
    """The local ranks are all fine here. Only the simulated remote rank
    failed, and every local rank still raises."""
    var statuses = zeros_int(2)
    var peer = zeros_int(3)
    peer[2] = STATUS_SHAPE_MISMATCH
    var comm = _PeerCollective(3, 2, peer^)

    var message = String("")
    try:
        agree_status(comm, statuses)
    except e:
        message = String(e)
    assert_true(
        message.find("rank 2") >= 0,
        "expected the remote rank to be named, got: " + message,
    )


def test_healthy_world_agrees_silently() raises:
    var comm = LocalCollective(4)
    agree_status(comm, zeros_int(4))
    assert_equal(agree_equal_ints(comm, [7, 255]), -1)
    assert_equal(comm.calls, 2)


def test_configuration_disagreement_is_detected() raises:
    """Two ranks in different processes given different feature counts. The
    reduction reports which value they disagree about, identically on every
    rank."""
    # The remote rank claims n_features = 9 where this one says 5.
    var peer: List[Int] = [9, -9, 255, -255]
    var comm = _PeerCollective(2, 1, peer^)
    assert_equal(agree_equal_ints(comm, [5, 255]), 0)

    var agreeing_peer: List[Int] = [5, -5, 255, -255]
    var agreeing = _PeerCollective(2, 1, agreeing_peer^)
    assert_equal(agree_equal_ints(agreeing, [5, 255]), -1)


def test_add_into_rejects_a_length_mismatch() raises:
    var acc: List[Float64] = [1.0, 2.0]
    var message = String("")
    try:
        var short: List[Float64] = [1.0]
        add_into_f64(acc, short)
    except e:
        message = String(e)
    assert_true(message.byte_length() > 0, "expected a length mismatch error")
    assert_equal(acc[0], 1.0)


def test_status_messages_are_distinct() raises:
    assert_equal(status_message(STATUS_OK), "no failure")
    assert_true(
        status_message(STATUS_SHAPE_MISMATCH) != status_message(STATUS_OK)
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
