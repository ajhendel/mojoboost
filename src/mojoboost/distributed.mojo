"""Data-parallel distributed training (CPU prototype).

Rows are partitioned across ranks, every rank holds the whole growing model,
and one all-reduce per tree node turns local histograms into the global
histogram every rank then reads to make identical decisions. See
docs/distributed.md for the design; the short version is:

- a rank owns a `DataShard`, a contiguous block of the global row order, and
  never sees another rank's rows
- split selection, the frontier scan, the smaller-child choice, leaf values,
  and the stopping rule are pure functions of all-reduced histograms, so no
  rank can reach a different conclusion and no split needs broadcasting
- which child to build directly and which to derive by subtraction is read
  off the global parent histogram's exact integer counts, so it costs no
  extra message and cannot disagree between ranks
- a rank with no rows still calls every collective, contributing zeros

Everything about how ranks talk to each other lives behind the `Collective`
trait in collective.mojo. Nothing in this file names a transport.

Determinism: for a fixed dataset, partition, and world size this is
bit-identical run to run and rank to rank. It equals single-node training
bit for bit when the histogram sums are exactly representable, and to within
accumulated rounding otherwise, because sharding regroups a floating point
sum that is not associative. Counts are integers and reduce exactly at any
world size.

This is a prototype. It has never run over a network, no distributed
performance is claimed, and the growth loop is a second implementation of
the one in tree.mojo rather than a refactor of it. The world-size-1
equivalence test is what keeps the two from drifting: when the single-node
grower gains a behavior this one lacks, that test fails, which is the point.
Tree parameters this prototype does not implement are rejected rather than
ignored.
"""

from std.math import log

from .binning import BinnedMatrix
from .boosting import (
    BINARY_LOGISTIC,
    HUBER,
    L1,
    POISSON,
    QUANTILE,
    SQUARED_ERROR,
    Booster,
    BoosterParams,
    _clamp_prob,
    _fill_grad_hess,
)
from .collective import (
    STATUS_INVALID_TARGET,
    STATUS_INVALID_WEIGHT,
    STATUS_OK,
    STATUS_SHAPE_MISMATCH,
    Collective,
    add_into_f64,
    add_into_int,
    agree_equal_ints,
    agree_status,
    zeros_f64,
    zeros_int,
)
from .histogram import (
    Histogram,
    build_histogram,
    build_histogram_subset,
    subtract_histogram,
)
from .interaction import extend_branch
from .split import SplitInfo, find_best_split, soft_threshold_l1
from .tree import Tree, TreeParams

# Bit positions in the unsupported-configuration mask agreed across ranks.
#
# The list is enumerated rather than derived, which is the honest weakness of
# a second growth loop: a parameter added to TreeParams or BinnedMatrix after
# this was written will not be in it, and will be ignored instead of refused
# until someone adds it. The world-size-1 equivalence test catches a change
# to default behavior but not the arrival of a new opt-in parameter. That is
# the strongest guarantee a separate loop can give, and it is the argument
# for the histogram-source refactor in docs/distributed.md section 11.
comptime _UNSUPPORTED_FEATURE_FRACTION = 1
comptime _UNSUPPORTED_FEATURE_FRACTION_BYNODE = 2
comptime _UNSUPPORTED_MAX_DEPTH = 4
comptime _UNSUPPORTED_MONOTONE = 8
comptime _UNSUPPORTED_CATEGORICAL = 16
comptime _UNSUPPORTED_MISSING_BIN = 32


struct DataShard(Copyable, Movable):
    """One rank's horizontal slice of the training data.

    `data` holds only this rank's rows, already binned with the bin mapper
    every shard shares. Bin edges are global quantiles, so binning must be
    fit before partitioning: a per-shard mapper would give each rank a
    different meaning for the same bin id and every reduced histogram cell
    would be summing unrelated quantities.

    An empty `weight` means unweighted. A shard may legitimately have zero
    rows.
    """

    var data: BinnedMatrix
    var target: List[Float64]
    var weight: List[Float64]

    def __init__(
        out self,
        var data: BinnedMatrix,
        var target: List[Float64],
        var weight: List[Float64] = [],
    ):
        self.data = data^
        self.target = target^
        self.weight = weight^


def partition_rows(
    data: BinnedMatrix,
    target: List[Float64],
    world_size: Int,
    weight: List[Float64] = [],
) raises -> List[DataShard]:
    """Split a dataset into `world_size` shards, one per rank.

    The partition is contiguous and order preserving: rank r owns rows
    `[r * n // W, (r + 1) * n // W)`, so concatenating shards in ascending
    rank order reproduces the dataset row for row. That is what lets the
    global reduction visit row contributions in the same order the
    single-node trainer does, which is the whole floating point equivalence
    argument. A world size larger than the row count simply yields empty
    shards, which take part in every collective as zero contributions.
    """
    if world_size < 1:
        raise Error("world_size must be positive")
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if len(weight) > 0 and len(weight) != data.n_rows:
        raise Error("sample_weight length must equal n_rows")

    var shards = List[DataShard]()
    for r in range(world_size):
        var start = r * data.n_rows // world_size
        var end = (r + 1) * data.n_rows // world_size
        var rows = end - start

        var bins = List[UInt8](capacity=rows * data.n_features)
        bins.resize(rows * data.n_features, 0)
        for f in range(data.n_features):
            var src = f * data.n_rows + start
            var dst = f * rows
            for i in range(rows):
                bins[dst + i] = data.bins[src + i]

        var shard_target = List[Float64](capacity=rows)
        for i in range(rows):
            shard_target.append(target[start + i])

        var shard_weight = List[Float64]()
        if len(weight) > 0:
            shard_weight = List[Float64](capacity=rows)
            for i in range(rows):
                shard_weight.append(weight[start + i])

        # The categorical spec and the missing-bin table describe the
        # columns, not the rows, so every shard carries the originals.
        # Distributed training refuses both today, but a shard must not
        # quietly lose the metadata that says so.
        shards.append(
            DataShard(
                BinnedMatrix(
                    bins^,
                    rows,
                    data.n_features,
                    data.n_bins,
                    data.cats.copy(),
                    data.missing_bin.copy(),
                ),
                shard_target^,
                shard_weight^,
            )
        )
    return shards^


def partition_values(
    values: List[Float64], world_size: Int
) raises -> List[List[Float64]]:
    """Split a per-row array across ranks on the same block boundaries
    `partition_rows` uses. Gradients and hessians are recomputed every round,
    so they are partitioned separately from the shards themselves."""
    if world_size < 1:
        raise Error("world_size must be positive")
    var out = List[List[Float64]]()
    for r in range(world_size):
        var start = r * len(values) // world_size
        var end = (r + 1) * len(values) // world_size
        var part = List[Float64](capacity=end - start)
        for i in range(start, end):
            part.append(values[i])
        out.append(part^)
    return out^


def _zero_histogram(n_features: Int, n_bins: Int) -> Histogram:
    var size = n_features * n_bins
    return Histogram(
        zeros_f64(size), zeros_f64(size), zeros_int(size), n_features, n_bins
    )


def _accumulate_histogram(mut acc: Histogram, src: Histogram) raises:
    """Add one local rank's histogram into this process's contribution.
    Called in ascending local rank order to match the ascending rank order a
    conforming transport uses across processes."""
    add_into_f64(acc.grad, src.grad)
    add_into_f64(acc.hess, src.hess)
    add_into_int(acc.count, src.count)


def allreduce_histogram[C: Collective](
    mut comm: C, mut hist: Histogram
) raises:
    """Sum a histogram across every rank, leaving the identical result
    everywhere.

    Three reductions rather than one, so that counts stay integers and their
    exactness is obvious. A production transport should pack gradients and
    hessians into one message; see docs/distributed.md section 8.
    """
    comm.allreduce_sum_f64(hist.grad)
    comm.allreduce_sum_f64(hist.hess)
    comm.allreduce_sum_int(hist.count)


def _total_count(hist: Histogram) -> Int:
    """Global row count behind a histogram, from feature 0's bins. Every
    feature's bins sum to the same total."""
    var total = 0
    for b in range(hist.n_bins):
        total += hist.count[b]
    return total


def _left_count(hist: Histogram, feature: Int, bin: Int) -> Int:
    """Rows the split sends left, exactly, from the global parent histogram.
    Integers, so every rank computes the same answer with no extra
    message."""
    var base = feature * hist.n_bins
    var total = 0
    for b in range(bin + 1):
        total += hist.count[base + b]
    return total


def _leaf_value(
    hist: Histogram, lambda_reg: Float64, lambda_l1: Float64
) -> Float64:
    """Newton leaf value with LightGBM's L1 shrinkage, mirroring the
    single-node grower's `_leaf_value`."""
    var g = 0.0
    var h = 0.0
    for b in range(hist.n_bins):
        g += hist.grad[b]
        h += hist.hess[b]
    return -soft_threshold_l1(g, lambda_l1) / (h + lambda_reg)


def _search(
    hist: Histogram,
    n_rows: Int,
    params: TreeParams,
    allowed: List[Bool],
) raises -> SplitInfo:
    """Best split for one node, from the global histogram. Mirrors the
    single-node grower's `_search`; ties resolve by the same ascending
    feature and bin scan order on every rank, so no tie-break protocol is
    needed."""
    if n_rows < 2 * params.min_data_in_leaf or n_rows < 2:
        return SplitInfo(-1, -1, 0.0, False)
    return find_best_split(
        hist,
        lambda_reg=params.lambda_reg,
        min_child_hess=params.min_child_hess,
        min_data_in_leaf=params.min_data_in_leaf,
        lambda_l1=params.lambda_l1,
        allowed=allowed,
    )


struct _DistLeaf(Movable):
    """A grown-but-unsplit leaf. `rows` is one row list per local rank;
    `hist`, `split`, and `count` are global and identical on every rank."""

    var node: Int
    var rows: List[List[Int]]
    var hist: Histogram
    var split: SplitInfo
    var branch: List[Int]
    var count: Int

    def __init__(
        out self,
        node: Int,
        var rows: List[List[Int]],
        var hist: Histogram,
        var split: SplitInfo,
        var branch: List[Int],
        count: Int,
    ):
        self.node = node
        self.rows = rows^
        self.hist = hist^
        self.split = split^
        self.branch = branch^
        self.count = count


def _unsupported_mask(params: TreeParams, shards: List[DataShard]) -> Int:
    """Bit mask of the configuration this prototype does not implement,
    covering both tree parameters and properties of the binned data.
    Rejected rather than ignored, so an unsupported setting is an error and
    not a quietly different model."""
    var mask = 0
    if params.feature_fraction != 1.0:
        mask |= _UNSUPPORTED_FEATURE_FRACTION
    if params.feature_fraction_bynode != 1.0:
        mask |= _UNSUPPORTED_FEATURE_FRACTION_BYNODE
    if params.max_depth > 0:
        mask |= _UNSUPPORTED_MAX_DEPTH
    if params.monotone.is_active():
        mask |= _UNSUPPORTED_MONOTONE
    for i in range(len(shards)):
        if shards[i].data.cats.any_categorical():
            mask |= _UNSUPPORTED_CATEGORICAL
        for f in range(len(shards[i].data.missing_bin)):
            if shards[i].data.missing_bin[f] >= 0:
                mask |= _UNSUPPORTED_MISSING_BIN
                break
    return mask


def _raise_unsupported(mask: Int) raises:
    """Every rank reaches this with the same mask, so it raises everywhere or
    nowhere."""
    if mask & _UNSUPPORTED_FEATURE_FRACTION != 0:
        raise Error(
            "distributed training does not support feature_fraction; the"
            " draw would have to be reproduced identically on every rank"
        )
    if mask & _UNSUPPORTED_FEATURE_FRACTION_BYNODE != 0:
        raise Error(
            "distributed training does not support feature_fraction_bynode;"
            " the draw would have to be reproduced identically on every rank"
        )
    if mask & _UNSUPPORTED_MAX_DEPTH != 0:
        raise Error(
            "distributed training does not support max_depth; the depth cap"
            " is not implemented in the distributed grower"
        )
    if mask & _UNSUPPORTED_MONOTONE != 0:
        raise Error(
            "distributed training does not support monotone constraints; the"
            " output bounds are not propagated by the distributed grower"
        )
    if mask & _UNSUPPORTED_CATEGORICAL != 0:
        raise Error(
            "distributed training does not support categorical features; the"
            " distributed grower searches ordinal thresholds only"
        )
    if mask & _UNSUPPORTED_MISSING_BIN != 0:
        raise Error(
            "distributed training does not support reserved missing bins; the"
            " distributed grower does not route missing values"
        )


def _agree_config[
    C: Collective
](mut comm: C, values: List[Int], names: List[String]) raises:
    """Raise identically on every rank when the ranks were given different
    configuration. One collective."""
    var bad = agree_equal_ints(comm, values)
    if bad >= 0:
        raise Error(
            "distributed training: ranks disagree about ", names[bad]
        )


def _shard_layout(shards: List[DataShard]) -> List[Int]:
    """This process's (n_features, n_bins), or (-1, -1) with no shards, so a
    process holding nothing disagrees with the others instead of silently
    agreeing."""
    if len(shards) == 0:
        return [-1, -1]
    return [shards[0].data.n_features, shards[0].data.n_bins]


def _shape_statuses[
    C: Collective
](
    comm: C,
    shards: List[DataShard],
    grad: List[List[Float64]],
    hess: List[List[Float64]],
) -> List[Int]:
    """Per-local-rank status for the shard shapes. Recorded, not raised, so
    every rank learns about every rank's failure and they all raise the same
    error together."""
    var n_local = comm.n_local_ranks()
    var statuses = zeros_int(n_local)
    if (
        len(shards) != n_local
        or len(grad) != n_local
        or len(hess) != n_local
    ):
        for i in range(n_local):
            statuses[i] = STATUS_SHAPE_MISMATCH
        return statuses^
    for i in range(n_local):
        var n_rows = shards[i].data.n_rows
        if (
            len(grad[i]) != n_rows
            or len(hess[i]) != n_rows
            or shards[i].data.n_features != shards[0].data.n_features
            or shards[i].data.n_bins != shards[0].data.n_bins
        ):
            statuses[i] = STATUS_SHAPE_MISMATCH
    return statuses^


def _grow_tree_distributed[
    C: Collective
](
    shards: List[DataShard],
    grad: List[List[Float64]],
    hess: List[List[Float64]],
    params: TreeParams,
    mut comm: C,
) raises -> Tree:
    """Grow one tree, leaf-wise, over sharded rows. Assumes the caller has
    already agreed configuration and shapes across ranks; the communication
    schedule from here is exactly one histogram all-reduce per tree node."""
    var n_features = shards[0].data.n_features
    var n_bins = shards[0].data.n_bins
    var n_local = len(shards)

    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )

    # Root: every local rank's whole row set, its local histogram summed in
    # ascending rank order, then one all-reduce.
    var root_rows = List[List[Int]]()
    var root_hist = _zero_histogram(n_features, n_bins)
    for i in range(n_local):
        var rows = List[Int](capacity=shards[i].data.n_rows)
        for r in range(shards[i].data.n_rows):
            rows.append(r)
        root_rows.append(rows^)
        _accumulate_histogram(
            root_hist, build_histogram(shards[i].data, grad[i], hess[i])
        )
    allreduce_histogram(comm, root_hist)

    var root_count = _total_count(root_hist)
    var root_branch = List[Int]()
    var root_split = _search(
        root_hist,
        root_count,
        params,
        params.constraints.allowed_features(root_branch),
    )
    var root = tree._add_node(
        _leaf_value(root_hist, params.lambda_reg, params.lambda_l1)
    )

    var frontier = List[_DistLeaf]()
    frontier.append(
        _DistLeaf(
            root, root_rows^, root_hist^, root_split^, root_branch^, root_count
        )
    )
    var n_leaves = 1

    while n_leaves < params.num_leaves:
        # Every input to this scan is global, so every rank picks the same
        # leaf without communicating.
        var best_i = -1
        var best_gain = 0.0
        for i in range(len(frontier)):
            if frontier[i].split.found and frontier[i].split.gain > best_gain:
                best_gain = frontier[i].split.gain
                best_i = i
        if best_i < 0:
            break

        var parent_node = frontier[best_i].node
        var split = frontier[best_i].split.copy()

        # Local work: each rank partitions only its own rows.
        var left_rows = List[List[Int]]()
        var right_rows = List[List[Int]]()
        for i in range(n_local):
            var lr = List[Int]()
            var rr = List[Int]()
            for k in range(len(frontier[best_i].rows[i])):
                var r = frontier[best_i].rows[i][k]
                if shards[i].data.bin_at(r, split.feature) <= split.bin:
                    lr.append(r)
                else:
                    rr.append(r)
            left_rows.append(lr^)
            right_rows.append(rr^)

        # Which child is smaller is read off the global parent histogram's
        # exact counts, so the ranks cannot disagree about which one to
        # build and which to derive by subtraction.
        var n_left = _left_count(
            frontier[best_i].hist, split.feature, split.bin
        )
        var n_right = frontier[best_i].count - n_left

        var left_hist: Histogram
        var right_hist: Histogram
        if n_left <= n_right:
            left_hist = _zero_histogram(n_features, n_bins)
            for i in range(n_local):
                _accumulate_histogram(
                    left_hist,
                    build_histogram_subset(
                        shards[i].data, grad[i], hess[i], left_rows[i]
                    ),
                )
            allreduce_histogram(comm, left_hist)
            right_hist = subtract_histogram(frontier[best_i].hist, left_hist)
        else:
            right_hist = _zero_histogram(n_features, n_bins)
            for i in range(n_local):
                _accumulate_histogram(
                    right_hist,
                    build_histogram_subset(
                        shards[i].data, grad[i], hess[i], right_rows[i]
                    ),
                )
            allreduce_histogram(comm, right_hist)
            left_hist = subtract_histogram(frontier[best_i].hist, right_hist)

        var left_node = tree._add_node(
            _leaf_value(left_hist, params.lambda_reg, params.lambda_l1)
        )
        var right_node = tree._add_node(
            _leaf_value(right_hist, params.lambda_reg, params.lambda_l1)
        )
        tree.feature[parent_node] = split.feature
        tree.threshold_bin[parent_node] = split.bin
        tree.left[parent_node] = left_node
        tree.right[parent_node] = right_node
        tree.split_gain[parent_node] = split.gain

        var branch = extend_branch(frontier[best_i].branch, split.feature)
        var allowed = params.constraints.allowed_features(branch)
        var left_split = _search(left_hist, n_left, params, allowed)
        var right_split = _search(right_hist, n_right, params, allowed)

        frontier[best_i] = _DistLeaf(
            left_node,
            left_rows^,
            left_hist^,
            left_split^,
            branch.copy(),
            n_left,
        )
        frontier.append(
            _DistLeaf(
                right_node,
                right_rows^,
                right_hist^,
                right_split^,
                branch^,
                n_right,
            )
        )
        n_leaves += 1

    tree.n_leaves = n_leaves
    return tree^


def grow_tree_distributed[
    C: Collective
](
    shards: List[DataShard],
    grad: List[List[Float64]],
    hess: List[List[Float64]],
    params: TreeParams,
    mut comm: C,
) raises -> Tree:
    """Grow one tree over rows partitioned across ranks.

    `shards`, `grad`, and `hess` describe this process's local ranks, one
    entry each per `comm.n_local_ranks()`. Every rank returns the same tree.

    Validation is collective: each rank records a status instead of raising,
    the statuses are reduced, and every rank then raises the same error
    naming the lowest-numbered failing rank. A run where one rank raises
    while the others block in a collective it will never call is the failure
    mode this ordering exists to prevent.
    """
    agree_status(comm, _shape_statuses(comm, shards, grad, hess))

    var layout = _shard_layout(shards)
    var mask = _unsupported_mask(params, shards)
    _agree_config(
        comm,
        [layout[0], layout[1], params.num_leaves, mask],
        ["n_features", "n_bins", "num_leaves", "the configuration"],
    )
    _raise_unsupported(mask)
    params.constraints.check_features(layout[0])

    return _grow_tree_distributed(shards, grad, hess, params, comm)


def _objective_supported(objective: Int) -> Bool:
    return (
        objective == SQUARED_ERROR
        or objective == BINARY_LOGISTIC
        or objective == POISSON
        or objective == HUBER
    )


def _target_statuses[
    C: Collective
](comm: C, shards: List[DataShard], objective: Int) -> List[Int]:
    """Per-local-rank status for target and weight validity. Weights are
    only checked for length and sign here: a shard whose weights are all
    zero is fine as long as some other shard has positive weight, so the
    positive-sum check is global and happens after the base score
    reduction."""
    var n_local = comm.n_local_ranks()
    var statuses = zeros_int(n_local)
    if len(shards) != n_local:
        for i in range(n_local):
            statuses[i] = STATUS_SHAPE_MISMATCH
        return statuses^
    for i in range(n_local):
        var n_rows = shards[i].data.n_rows
        if len(shards[i].target) != n_rows:
            statuses[i] = STATUS_SHAPE_MISMATCH
            continue
        var n_weights = len(shards[i].weight)
        if n_weights > 0 and n_weights != n_rows:
            statuses[i] = STATUS_INVALID_WEIGHT
            continue
        var bad_weight = False
        for r in range(n_weights):
            if shards[i].weight[r] < 0.0:
                bad_weight = True
                break
        if bad_weight:
            statuses[i] = STATUS_INVALID_WEIGHT
            continue
        if objective == POISSON:
            for r in range(n_rows):
                if shards[i].target[r] < 0.0:
                    statuses[i] = STATUS_INVALID_TARGET
                    break
    return statuses^


def _distributed_base_score[
    C: Collective
](
    mut comm: C, shards: List[DataShard], objective: Int
) raises -> Float64:
    """Global base score: one reduction of the weighted target sum and the
    weight sum, then the same link transform the single-node trainer
    applies. Every rank holds identical sums afterwards, so the checks below
    the reduction raise on every rank or on none."""
    var sums = zeros_f64(2)
    for i in range(len(shards)):
        var weighted = len(shards[i].weight) > 0
        for r in range(shards[i].data.n_rows):
            var w = shards[i].weight[r] if weighted else 1.0
            sums[0] += w * shards[i].target[r]
            sums[1] += w
    comm.allreduce_sum_f64(sums)

    if sums[1] <= 0.0:
        raise Error("sample_weight must have a positive sum")
    var mean = sums[0] / sums[1]
    if objective == BINARY_LOGISTIC:
        var p = _clamp_prob(mean)
        return log(p / (1.0 - p))
    if objective == POISSON:
        if mean <= 0.0:
            raise Error("poisson requires a positive mean target")
        return log(mean)
    return mean


def train_distributed[
    C: Collective
](
    shards: List[DataShard],
    objective: Int,
    params: BoosterParams,
    mut comm: C,
    alpha: Float64 = 0.9,
) raises -> Booster:
    """Train a boosted ensemble over rows partitioned across ranks.

    `shards` describes this process's local ranks, one per
    `comm.n_local_ranks()`. Every rank returns the same `Booster`, which is
    an ordinary model: it predicts and serializes exactly like one trained on
    a single node, because sharding changes how the histograms are summed and
    nothing about the model.

    Supported objectives are squared error, binary logistic, poisson, and
    huber. Quantile and L1 replace leaf values with a percentile of the
    leaf's residuals, and a percentile is not a sum, so it does not
    all-reduce; approximating it would make distributed training disagree
    with single-node training in a way no tolerance test would catch, so it
    is refused instead. Early stopping and multiclass are not implemented.
    """
    var mask = _unsupported_mask(params.tree, shards)
    var layout = _shard_layout(shards)
    _agree_config(
        comm,
        [
            layout[0],
            layout[1],
            objective,
            params.n_estimators,
            params.tree.num_leaves,
            mask,
        ],
        [
            "n_features",
            "n_bins",
            "the objective",
            "n_estimators",
            "num_leaves",
            "the configuration",
        ],
    )

    # Every value above is now known to be the same on every rank, so these
    # checks raise on every rank or on none.
    if objective == QUANTILE or objective == L1:
        raise Error(
            "distributed training does not support the quantile or L1"
            " objective: both renew leaf values with a percentile of the"
            " leaf's residuals, which is not a sum and does not all-reduce"
        )
    if not _objective_supported(objective):
        raise Error("unknown objective")
    if objective == HUBER and alpha <= 0.0:
        raise Error("huber requires alpha > 0")
    _raise_unsupported(mask)
    params.tree.constraints.check_features(layout[0])

    agree_status(comm, _target_statuses(comm, shards, objective))

    var n_local = len(shards)
    var base_score = _distributed_base_score(comm, shards, objective)

    var raw = List[List[Float64]]()
    for i in range(n_local):
        var r = List[Float64](capacity=shards[i].data.n_rows)
        for _ in range(shards[i].data.n_rows):
            r.append(base_score)
        raw.append(r^)

    var trees = List[Tree]()
    for _ in range(params.n_estimators):
        # Gradients and hessians are a per-row function of local state only.
        var grad = List[List[Float64]]()
        var hess = List[List[Float64]]()
        for i in range(n_local):
            var g = List[Float64]()
            var h = List[Float64]()
            _fill_grad_hess(
                raw[i],
                shards[i].target,
                objective,
                shards[i].weight,
                alpha,
                g,
                h,
            )
            grad.append(g^)
            hess.append(h^)

        var tree = _grow_tree_distributed(shards, grad, hess, params.tree, comm)

        # The tree is global, so this convergence break is unanimous: no
        # rank can leave the loop while another rank stays in it and blocks
        # on a collective.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break

        for i in range(n_local):
            var updated = List[Float64](capacity=len(raw[i]))
            for r in range(shards[i].data.n_rows):
                updated.append(
                    raw[i][r]
                    + params.learning_rate * tree.predict_row(shards[i].data, r)
                )
            raw[i] = updated^
        trees.append(tree^)

    return Booster(trees^, base_score, params.learning_rate, objective)
