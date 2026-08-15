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
trait in collective.mojo. Nothing in this file opens a connection, names a
socket, or knows what a frame is.

What it does now use from distributed_transport.mojo is the part of that module
that is not about bytes: the schema digest, the histogram cost contract, the
split digest, the checkpoint record, and the `RuntimeSpec` that says which
world a process belongs to. Those were written against this algorithm and then
left unreferenced by it, which is what made a validated transport and an
unvalidated training run coexist. `run_distributed` is now the one entry point
that turns a `RuntimeSpec` into a collective and a trained model. A local spec
opens a world hosted in this process; a transport spec goes through the socket
rendezvous in distributed_transport.mojo, which is implemented and which
nobody has yet run.

Two rules keep the additions from changing what a run already does:

- every collective this file added is behind `hosts_whole_world`, so a world
  hosted in one process issues exactly the reductions it always did. There is
  nothing to negotiate with a peer that is this same process, and the values
  a peer would confirm are computed here for every rank already.
- everything a caller has to ask for (split verification, metrics, early
  stopping, checkpoints, callbacks) is off in `DistributedRunOptions()`, so
  `train_distributed` is `train_distributed_run` with the defaults and the two
  produce the same model from the same messages.

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
    _mean_loss,
)
from .callback import (
    ABORT,
    AFTER_ITERATION,
    BEFORE_ITERATION,
    CONTINUE,
    STOP,
    IterationEnv,
    IterationFn,
    no_callback,
)
from .collective import (
    STATUS_INVALID_TARGET,
    STATUS_INVALID_WEIGHT,
    STATUS_OK,
    STATUS_PARTITION_MISMATCH,
    STATUS_SHAPE_MISMATCH,
    Collective,
    add_into_f64,
    add_into_int,
    agree_equal_ints,
    agree_status,
    hosts_whole_world,
    zeros_f64,
    zeros_int,
)
from .distributed_transport import (
    RUNTIME_LOCAL,
    RUNTIME_TRANSPORT,
    CheckpointMeta,
    RuntimeSpec,
    check_histogram_buffers,
    digest_halves,
    digest_ints,
    f64_bits,
    histogram_plan,
    open_local_collective,
    open_socket_collective,
    require_transport,
    split_digest,
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
from .tree_parameters_extra import finish_leaf_output

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
comptime _UNSUPPORTED_FEATURE_FRACTION_BYLEVEL = 64
comptime _UNSUPPORTED_EXTRA_TREES = 128
comptime _UNSUPPORTED_FORCED_SPLITS = 256
comptime _UNSUPPORTED_RANKING = 512


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


struct ShardPlan(Copyable, Movable):
    """Where every rank's rows sit in the global row order.

    `row_offset[r]` is the first global row index rank r owns and `n_rows[r]`
    is how many it owns, so the ranks tile `[0, total_rows)` in ascending rank
    order with no gap and no overlap. That is the same thing
    `RankAssignment.row_offset` and `n_rows` mean on the Dask side of
    python/mojotrees/dask.py, deliberately, so a plan built there and a plan
    negotiated here describe one partition rather than two.

    It matters beyond bookkeeping in three places. The floating point
    equivalence argument is that the reduction visits row contributions in the
    dataset's own order, which is only true if this tiling is the identity on
    the row order. A ranking query group that straddles two ranks cannot be
    scored by either of them, and `check_group_alignment` is that constraint
    written against these offsets. And a restart at a different world size
    re-shards the data underneath a half-built model, which is why the
    checkpoint record carries the world size this plan was built for.
    """

    var row_offset: List[Int]
    var n_rows: List[Int]

    def __init__(out self, var row_offset: List[Int], var n_rows: List[Int]):
        self.row_offset = row_offset^
        self.n_rows = n_rows^

    def world_size(self) -> Int:
        return len(self.n_rows)

    def total_rows(self) -> Int:
        var total = 0
        for r in range(len(self.n_rows)):
            total += self.n_rows[r]
        return total

    def validate(self) raises:
        """Refuse a plan that is not a tiling. Cheap, and worth doing on a plan
        that arrived from outside Mojo rather than trusting it."""
        if len(self.row_offset) != len(self.n_rows):
            raise Error("a shard plan needs one offset per rank")
        if len(self.n_rows) < 1:
            raise Error("a shard plan needs at least one rank")
        var expected = 0
        for r in range(len(self.n_rows)):
            if self.n_rows[r] < 0:
                raise Error("a shard cannot hold a negative row count")
            if self.row_offset[r] != expected:
                raise Error(
                    String(
                        "shard plan is not contiguous: rank ",
                        r,
                        " starts at row ",
                        self.row_offset[r],
                        " where the ranks before it end at ",
                        expected,
                    )
                )
            expected += self.n_rows[r]

    @staticmethod
    def from_counts(counts: List[Int]) raises -> ShardPlan:
        """The plan implied by per-rank row counts, as one prefix sum.

        This is how a plan comes back from a reduction: the counts are the
        thing ranks can contribute, and the offsets follow from them
        identically on every rank without a second message.
        """
        var offsets = List[Int](capacity=len(counts))
        var running = 0
        for r in range(len(counts)):
            offsets.append(running)
            running += counts[r]
        var plan = ShardPlan(offsets^, counts.copy())
        plan.validate()
        return plan^

    @staticmethod
    def contiguous(n_rows: Int, world_size: Int) raises -> ShardPlan:
        """The block partition `partition_rows` builds.

        The one definition of the boundary, so a caller that needs to know
        which rows a rank will own asks this rather than recomputing
        `r * n // W` and drifting from it.
        """
        if world_size < 1:
            raise Error("world_size must be positive")
        if n_rows < 0:
            raise Error("n_rows must not be negative")
        var counts = List[Int](capacity=world_size)
        for r in range(world_size):
            counts.append(
                (r + 1) * n_rows // world_size - r * n_rows // world_size
            )
        return ShardPlan.from_counts(counts)


def resolve_partition[
    C: Collective
](mut comm: C, shards: List[DataShard]) raises -> ShardPlan:
    """The global row partition, agreed across the world.

    Every rank contributes its own row counts into its own slots of a
    world-sized vector and reduces the sum, so each rank ends up holding every
    rank's count and derives the identical offsets from them. A rank therefore
    learns where its rows sit in the global order without being told, which is
    what the ranking-group constraint and the checkpoint record both need.

    One reduction, and only when this process does not already hold every rank:
    a world hosted in one process has all the counts in hand, and reducing them
    would be a round trip to confirm what it just computed.
    """
    var world = comm.world_size()
    var counts = zeros_int(world)
    for i in range(len(shards)):
        var r = comm.local_rank(i)
        if r < 0 or r >= world:
            raise Error("local rank id out of range")
        counts[r] = shards[i].data.n_rows
    if not hosts_whole_world(comm):
        comm.allreduce_sum_int(counts)
    return ShardPlan.from_counts(counts)


def _partition_statuses[
    C: Collective
](comm: C, shards: List[DataShard], plan: ShardPlan) -> List[Int]:
    """Per-local-rank status for the plan against the shards actually held.

    Recorded rather than raised, like every other validation here, so a rank
    whose shard disagrees with the agreed plan does not raise while the rest of
    the world waits in a collective it will never call.
    """
    var n_local = comm.n_local_ranks()
    var statuses = zeros_int(n_local)
    if plan.world_size() != comm.world_size() or len(shards) != n_local:
        for i in range(n_local):
            statuses[i] = STATUS_PARTITION_MISMATCH
        return statuses^
    for i in range(n_local):
        var r = comm.local_rank(i)
        if r < 0 or r >= plan.world_size():
            statuses[i] = STATUS_PARTITION_MISMATCH
        elif plan.n_rows[r] != shards[i].data.n_rows:
            statuses[i] = STATUS_PARTITION_MISMATCH
    return statuses^


def check_group_alignment(plan: ShardPlan, groups: List[Int]) raises:
    """Refuse a ranking partition that splits a query across two ranks.

    A query group's rows are scored against each other, so a rank holding half
    of one can compute neither its NDCG nor its lambdas, and no reduction over
    per-rank sums can repair that: the missing quantity is a comparison, not a
    sum. LightGBM's rule is the same, and section 3 of docs/distributed.md
    states it for this partition.

    `groups` is the global query sizes in row order, which is the same shape
    `groups_from_counts` takes in ranking.mojo. This checks the constraint and
    nothing else; distributed lambdarank is not implemented, and
    `train_distributed_run` refuses it separately. The constraint is worth
    having on its own because it is what a client has to satisfy *before* it
    partitions, and satisfying it after the fact is not possible.
    """
    plan.validate()
    var total = 0
    for g in range(len(groups)):
        if groups[g] <= 0:
            raise Error("a query group must contain at least one row")
        total += groups[g]
    if total != plan.total_rows():
        raise Error(
            String(
                "the query groups cover ",
                total,
                " rows and the shard plan covers ",
                plan.total_rows(),
            )
        )
    # Walk the group boundaries once and require every rank boundary to be one
    # of them. Both sequences are ascending, so this is a merge and not a
    # search.
    var g = 0
    var boundary = 0
    for r in range(1, plan.world_size()):
        var edge = plan.row_offset[r]
        while boundary < edge and g < len(groups):
            boundary += groups[g]
            g += 1
        if boundary != edge:
            raise Error(
                String(
                    "query group ",
                    g - 1,
                    " straddles the boundary between rank ",
                    r - 1,
                    " and rank ",
                    r,
                    " at row ",
                    edge,
                    "; a group must be owned entirely by one rank",
                )
            )


def partition_rows(
    data: BinnedMatrix,
    target: List[Float64],
    world_size: Int,
    weight: List[Float64] = [],
) raises -> List[DataShard]:
    """Split a dataset into `world_size` shards, one per rank.

    The partition is `ShardPlan.contiguous`: contiguous and order preserving,
    so concatenating shards in ascending rank order reproduces the dataset row
    for row. That is what lets the global reduction visit row contributions in
    the same order the single-node trainer does, which is the whole floating
    point equivalence argument. A world size larger than the row count simply
    yields empty shards, which take part in every collective as zero
    contributions.
    """
    if len(target) != data.n_rows:
        raise Error("target length must equal n_rows")
    if len(weight) > 0 and len(weight) != data.n_rows:
        raise Error("sample_weight length must equal n_rows")

    var plan = ShardPlan.contiguous(data.n_rows, world_size)
    var shards = List[DataShard]()
    for r in range(world_size):
        var start = plan.row_offset[r]
        var rows = plan.n_rows[r]

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

    The three buffers are checked against `histogram_plan` first. That is the
    cost contract in distributed_transport.mojo, and checking here is what
    connects it to the schedule it describes rather than leaving it as an
    assertion about a comment: a histogram whose buffers do not describe one
    features-by-bins grid is refused locally, by the rank that built it, rather
    than reaching the peers as a length disagreement attributed to whoever
    happened to receive it. It costs three integer comparisons per node and no
    messages.
    """
    var plan = histogram_plan(hist.n_features, hist.n_bins)
    check_histogram_buffers(
        plan, len(hist.grad), len(hist.hess), len(hist.count)
    )
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
    hist: Histogram,
    lambda_reg: Float64,
    lambda_l1: Float64,
    n_data: Int = 0,
    parent_output: Float64 = 0.0,
    max_delta_step: Float64 = 0.0,
    path_smooth: Float64 = 0.0,
) -> Float64:
    """Newton leaf value with LightGBM's L1 shrinkage, then `max_delta_step`
    and `path_smooth`, mirroring the single-node grower's `_leaf_value`.

    `n_data` is the leaf's *global* row count, the same integer every rank
    reads off the reduced histogram, so the smoothing weight is identical on
    every rank and no extra message is needed to agree on a leaf value.
    """
    var g = 0.0
    var h = 0.0
    for b in range(hist.n_bins):
        g += hist.grad[b]
        h += hist.hess[b]
    var value = -soft_threshold_l1(g, lambda_l1) / (h + lambda_reg)
    if max_delta_step <= 0.0 and path_smooth <= 0.0:
        return value
    return finish_leaf_output(
        value, max_delta_step, path_smooth, n_data, parent_output
    )


def _search(
    hist: Histogram,
    n_rows: Int,
    params: TreeParams,
    allowed: List[Bool],
    parent_output: Float64 = 0.0,
) raises -> SplitInfo:
    """Best split for one node, from the global histogram. Mirrors the
    single-node grower's `_search`; ties resolve by the same ascending
    feature and bin scan order on every rank, so no tie-break protocol is
    needed.

    `params.extra` is forwarded, so `min_gain_to_split`, `feature_contri`,
    and the CEGB split cost decide here exactly as they do on the single-node
    grower. Every input they read is global (the reduced histogram and the
    exact row count), so they cannot make two ranks disagree. The parts of
    the bundle this grower cannot honor are refused up front by
    `_unsupported_mask`, not silently dropped.

    `monotone_penalty` needs no depth here: this grower refuses monotone
    constraints outright, so no feature carries a sign and the penalty is the
    identity by definition.
    """
    if n_rows < 2 * params.min_data_in_leaf or n_rows < 2:
        return SplitInfo(-1, -1, 0.0, False)
    return find_best_split(
        hist,
        lambda_reg=params.lambda_reg,
        min_child_hess=params.min_child_hess,
        min_data_in_leaf=params.min_data_in_leaf,
        lambda_l1=params.lambda_l1,
        allowed=allowed,
        extra=params.extra,
        n_rows=n_rows,
        parent_output=parent_output,
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
    if params.feature_fraction_bylevel != 1.0:
        mask |= _UNSUPPORTED_FEATURE_FRACTION_BYLEVEL
    if params.max_depth > 0:
        mask |= _UNSUPPORTED_MAX_DEPTH
    if params.monotone.is_active():
        mask |= _UNSUPPORTED_MONOTONE
    # The rest of `params.extra` is honored below: `min_gain_to_split`, the
    # per-feature gain multipliers, and the CEGB split cost go into the
    # search, and `max_delta_step` and `path_smooth` into the leaf values.
    # These two cannot be.
    if params.extra.extra_trees:
        mask |= _UNSUPPORTED_EXTRA_TREES
    if not params.extra.forced.is_empty():
        mask |= _UNSUPPORTED_FORCED_SPLITS
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
    if mask & _UNSUPPORTED_FEATURE_FRACTION_BYLEVEL != 0:
        raise Error(
            "distributed training does not support feature_fraction_bylevel;"
            " the draw would have to be reproduced identically on every rank"
        )
    if mask & _UNSUPPORTED_EXTRA_TREES != 0:
        raise Error(
            "distributed training does not support extra_trees; the threshold"
            " draw is keyed by the tree index, which this grower is not given"
        )
    if mask & _UNSUPPORTED_FORCED_SPLITS != 0:
        raise Error(
            "distributed training does not support forced splits; applying"
            " one needs the bin mapper, which this grower is not given"
        )
    if mask & _UNSUPPORTED_RANKING != 0:
        raise Error(
            "distributed training does not support ranking; a query group's"
            " rows are scored against each other, and no reduction over"
            " per-rank sums recovers a comparison. The group-to-rank"
            " constraint is implemented in check_group_alignment, so a client"
            " can partition correctly for a transport that grows lambdarank"
            " trees, but this grower does not"
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


def _push_f64(mut values: List[Int], value: Float64):
    """Append a double to a digest input by its bits, in two halves.

    By its bits because the claim a schema digest supports is that two ranks
    were configured identically, and a learning rate that differs in the last
    place is a different configuration. A tolerance here would hide exactly the
    divergence the digest exists to catch.
    """
    var halves = digest_halves(f64_bits(value))
    values.append(halves[0])
    values.append(halves[1])


def _schema_values(
    shards: List[DataShard], params: TreeParams, world_size: Int
) -> List[Int]:
    """Everything about a run that every rank must have been given the same.

    Enumerated, and that is the same honest weakness `_unsupported_mask` has:
    a parameter added to `TreeParams` after this was written is not in it, so
    two ranks given different values of that parameter would agree here and
    disagree in the model. What the digest buys over the named field list is
    that it is cheap to extend by one line and costs no extra message, so the
    field list can stay short enough to produce good error messages while this
    covers the rest.

    The bin *edges* are not in it and cannot be: they are quantiles of the
    whole dataset, fitted before the partition, and a rank does not hold them.
    What is here is the grid they produced, which is what a reduced histogram
    cell's meaning depends on.
    """
    var values: List[Int] = [world_size]
    if len(shards) == 0:
        values.append(-1)
        values.append(-1)
    else:
        values.append(shards[0].data.n_features)
        values.append(shards[0].data.n_bins)
        for f in range(shards[0].data.n_features):
            var categorical = 0
            if (
                f < len(shards[0].data.cats.is_categorical)
                and shards[0].data.cats.is_categorical[f]
            ):
                categorical = 1
            values.append(categorical)
            if f < len(shards[0].data.missing_bin):
                values.append(shards[0].data.missing_bin[f])
            else:
                values.append(-1)
    values.append(params.num_leaves)
    values.append(params.min_data_in_leaf)
    values.append(params.feature_fraction_seed)
    values.append(params.max_depth)
    _push_f64(values, params.lambda_reg)
    _push_f64(values, params.min_child_hess)
    _push_f64(values, params.lambda_l1)
    _push_f64(values, params.feature_fraction)
    _push_f64(values, params.feature_fraction_bynode)
    _push_f64(values, params.feature_fraction_bylevel)
    _push_f64(values, params.extra.max_delta_step)
    _push_f64(values, params.extra.path_smooth)
    return values^


def tree_schema_digest(
    shards: List[DataShard], params: TreeParams, world_size: Int, mask: Int
) -> UInt64:
    """The digest ranks compare before growing a tree together."""
    var values = _schema_values(shards, params, world_size)
    values.append(mask)
    return digest_ints(values)


def run_schema_digest(
    shards: List[DataShard],
    params: BoosterParams,
    objective: Int,
    alpha: Float64,
    world_size: Int,
    mask: Int,
) -> UInt64:
    """The digest ranks compare before training an ensemble together.

    Also what a checkpoint records as the schema it was taken under, so a
    restart against different parameters is refused rather than resumed into a
    model the parameters no longer describe.
    """
    var values = _schema_values(shards, params.tree, world_size)
    values.append(mask)
    values.append(objective)
    values.append(params.n_estimators)
    _push_f64(values, params.learning_rate)
    _push_f64(values, alpha)
    return digest_ints(values)


def _agree_config_and_schema[
    C: Collective
](
    mut comm: C,
    var values: List[Int],
    var names: List[String],
    digest: UInt64,
) raises:
    """Agree the named fields and the schema digest in one reduction.

    The named fields are first so a disagreement about something a person can
    act on ("ranks disagree about n_features") is reported as that rather than
    as a digest mismatch. The digest is the catch-all underneath, and it rides
    in the same message, so this costs the same one collective the field list
    already cost.

    Under a world hosted in one process the digest is omitted, because this
    process computed it for every rank and there is nobody to have computed a
    different one. That keeps the message schedule of a local run exactly what
    it has always been.
    """
    if not hosts_whole_world(comm):
        var halves = digest_halves(digest)
        values.append(halves[0])
        names.append("the run schema")
        values.append(halves[1])
        names.append("the run schema")
    _agree_config(comm, values, names)


def _agree_split[
    C: Collective
](mut comm: C, split: SplitInfo, node: Int) raises:
    """Check that every rank chose the same split for one node.

    Unreachable in a correct run: the split is a pure function of the
    all-reduced histogram, so agreement is structural and not negotiated. It is
    here for a real transport, where a histogram corrupted in a way the frame
    checksum missed would otherwise grow two different trees and raise nothing
    at all. It costs one integer reduction per node on top of the three the
    histogram already costs, which is why it is opt-in.

    A world hosted in one process returns at once: every rank's split was
    computed by this process from one histogram, so there is no second opinion
    for a reduction to find, and issuing one would only spend a round trip
    confirming a value against itself.
    """
    if hosts_whole_world(comm):
        return
    var digest = split_digest(split.feature, split.bin, split.gain, split.found)
    var bad = agree_equal_ints(comm, digest_halves(digest))
    if bad >= 0:
        raise Error(
            String(
                "distributed training: ranks chose different splits for node ",
                node,
                "; the histogram they read cannot have been identical",
            )
        )


struct DistributedRunOptions(Copyable, Movable):
    """What a distributed run does beyond growing the trees.

    Every field is off by default, and a default-constructed instance is what
    `train_distributed` passes, so the option-free path issues exactly the
    collectives it always did and produces the same model.

    - `verify_splits` adds the per-node split agreement above.
    - `metric_every` scores the global training loss every N rounds, 0 never.
    - `early_stopping_rounds` stops when that loss has not improved by
      `early_stopping_delta` for N consecutive rounds. It implies scoring every
      round. The decision is a function of a reduced scalar, so it is unanimous
      and no rank can leave the loop while another waits in a collective.
    - `checkpoint_every` marks a checkpoint boundary every N rounds: a barrier,
      which every rank must reach, and a `CheckpointMeta` recording what a
      restart would have to match.
    - `groups` is the global ranking query sizes. Supplying them refuses the
      run, because distributed lambdarank is not implemented; they are checked
      against the partition first so the refusal is not the only thing a
      caller learns.
    - `job_id` and `schema` stamp the checkpoint records. `run_distributed`
      fills them from the `RuntimeSpec`; a caller driving the run entry point
      itself may set them directly.
    """

    var verify_splits: Bool
    var metric_every: Int
    var early_stopping_rounds: Int
    var early_stopping_delta: Float64
    var checkpoint_every: Int
    var groups: List[Int]
    var job_id: UInt64
    var schema: UInt64

    def __init__(out self):
        self.verify_splits = False
        self.metric_every = 0
        self.early_stopping_rounds = 0
        self.early_stopping_delta = 0.0
        self.checkpoint_every = 0
        self.groups = List[Int]()
        self.job_id = 0
        self.schema = 0

    def scores_every_round(self) -> Bool:
        return self.early_stopping_rounds > 0

    def check(self) raises:
        if self.metric_every < 0:
            raise Error("metric_every must not be negative")
        if self.early_stopping_rounds < 0:
            raise Error("early_stopping_rounds must not be negative")
        if self.early_stopping_delta < 0.0:
            raise Error("early_stopping_delta must not be negative")
        if self.checkpoint_every < 0:
            raise Error("checkpoint_every must not be negative")


struct DistributedRunReport(Copyable, Movable):
    """What a run did, as opposed to what it produced.

    Identical on every rank, because every field is either a constant of the
    configuration or derived from a reduced value. That is deliberate: a report
    that differed between ranks would be the first thing to make two ranks
    disagree about whether a run succeeded.
    """

    var world_size: Int
    var rounds_run: Int
    var plan: ShardPlan
    var schema: UInt64
    var model_digest: UInt64
    var metric_rounds: List[Int]
    var metric_values: List[Float64]
    var stopped_early: Bool
    var stopped_by_callback: Bool
    var checkpoints: List[CheckpointMeta]

    def __init__(out self, var plan: ShardPlan, schema: UInt64):
        self.world_size = plan.world_size()
        self.rounds_run = 0
        self.plan = plan^
        self.schema = schema
        self.model_digest = 0
        self.metric_rounds = List[Int]()
        self.metric_values = List[Float64]()
        self.stopped_early = False
        self.stopped_by_callback = False
        self.checkpoints = List[CheckpointMeta]()

    def global_rows(self) -> Int:
        return self.plan.total_rows()

    def last_metric(self) raises -> Float64:
        if len(self.metric_values) == 0:
            raise Error("this run scored no metrics")
        return self.metric_values[len(self.metric_values) - 1]


struct DistributedOutcome(Movable):
    """A trained model and the report of the run that produced it."""

    var model: Booster
    var report: DistributedRunReport

    def __init__(
        out self, var model: Booster, var report: DistributedRunReport
    ):
        self.model = model^
        self.report = report^


def distributed_metric[
    C: Collective
](
    mut comm: C,
    shards: List[DataShard],
    raw: List[List[Float64]],
    objective: Int,
    alpha: Float64,
) raises -> Float64:
    """The global training loss, in one reduction.

    The loss is `boosting._mean_loss`, the same function the single-node
    trainer's early stopping reads, so the two stop on the same quantity rather
    than on two definitions of it. Each rank contributes its shard's mean
    scaled by its row count, and the reduced pair is divided at the end, which
    makes the result the row-weighted mean over the whole dataset.

    It is not bit-identical to the single-node mean: scaling a mean back up to
    a sum is not the sum, and the shards regroup the addition anyway. It is
    identical on every rank, which is the property a stopping decision needs.
    Empty shards contribute nothing rather than a division by zero.
    """
    var sums = zeros_f64(2)
    for i in range(len(shards)):
        var n = shards[i].data.n_rows
        if n > 0:
            sums[0] += (
                _mean_loss(raw[i], shards[i].target, objective, alpha)
                * Float64(n)
            )
            sums[1] += Float64(n)
    comm.allreduce_sum_f64(sums)
    if sums[1] <= 0.0:
        raise Error("the world holds no rows to score")
    return sums[0] / sums[1]


def model_digest(model: Booster) -> UInt64:
    """A digest of everything about a fitted ensemble that prediction reads.

    What a checkpoint record identifies. Structure and leaf values by their
    bits, so two models that predict identically digest identically and two
    that differ anywhere do not. Linear in the model, which is why it is
    computed at checkpoint boundaries and not every round.
    """
    var values: List[Int] = [len(model.trees), model.objective]
    _push_f64(values, model.base_score)
    _push_f64(values, model.learning_rate)
    for t in range(len(model.trees)):
        values.append(model.trees[t].n_leaves)
        for i in range(len(model.trees[t].feature)):
            values.append(model.trees[t].feature[i])
            values.append(model.trees[t].threshold_bin[i])
            values.append(model.trees[t].left[i])
            values.append(model.trees[t].right[i])
            _push_f64(values, model.trees[t].value[i])
    return digest_ints(values)


def _grow_tree_distributed[
    C: Collective
](
    shards: List[DataShard],
    grad: List[List[Float64]],
    hess: List[List[Float64]],
    params: TreeParams,
    mut comm: C,
    options: DistributedRunOptions = DistributedRunOptions(),
) raises -> Tree:
    """Grow one tree, leaf-wise, over sharded rows. Assumes the caller has
    already agreed configuration and shapes across ranks; the communication
    schedule from here is exactly one histogram all-reduce per tree node."""
    var n_features = shards[0].data.n_features
    var n_bins = shards[0].data.n_bins
    var n_local = len(shards)

    # The parts of the bundle this grower does honor are validated the way the
    # single-node grower validates them, against this dataset. Every rank runs
    # the same check on the same numbers, so it raises everywhere or nowhere;
    # the parts it does not honor were already refused by `_unsupported_mask`.
    params.extra.check_scalars(params.min_data_in_leaf)
    params.extra.penalties.check_features(n_features)

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

    var max_delta_step = params.extra.max_delta_step
    var path_smooth = params.extra.path_smooth

    var root_count = _total_count(root_hist)
    var root_branch = List[Int]()
    # Covers are the global counts every rank agreed on, so the tree carries
    # the same node covers on every rank. The value is computed before the
    # search because path smoothing makes a candidate's children shrink toward
    # it, exactly as in `tree.grow_tree`; the root has no parent and so smooths
    # toward 0.0.
    var root = tree._add_node(
        _leaf_value(
            root_hist,
            params.lambda_reg,
            params.lambda_l1,
            root_count,
            0.0,
            max_delta_step,
            path_smooth,
        ),
        Float64(root_count),
    )
    var root_split = _search(
        root_hist,
        root_count,
        params,
        params.constraints.allowed_features(root_branch),
        tree.value[root],
    )
    if options.verify_splits:
        _agree_split(comm, root_split, root)

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

        # Both children smooth toward the value the parent already emits, and
        # the row counts are the global ones every rank read off the reduced
        # parent histogram, so the two ranks cannot value a leaf differently.
        var parent_output = tree.value[parent_node]
        var left_node = tree._add_node(
            _leaf_value(
                left_hist,
                params.lambda_reg,
                params.lambda_l1,
                n_left,
                parent_output,
                max_delta_step,
                path_smooth,
            ),
            Float64(n_left),
        )
        var right_node = tree._add_node(
            _leaf_value(
                right_hist,
                params.lambda_reg,
                params.lambda_l1,
                n_right,
                parent_output,
                max_delta_step,
                path_smooth,
            ),
            Float64(n_right),
        )
        tree.feature[parent_node] = split.feature
        tree.threshold_bin[parent_node] = split.bin
        tree.left[parent_node] = left_node
        tree.right[parent_node] = right_node
        tree.split_gain[parent_node] = split.gain

        var branch = extend_branch(frontier[best_i].branch, split.feature)
        var allowed = params.constraints.allowed_features(branch)
        var left_split = _search(
            left_hist, n_left, params, allowed, tree.value[left_node]
        )
        var right_split = _search(
            right_hist, n_right, params, allowed, tree.value[right_node]
        )
        if options.verify_splits:
            _agree_split(comm, left_split, left_node)
            _agree_split(comm, right_split, right_node)

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
    options: DistributedRunOptions = DistributedRunOptions(),
) raises -> Tree:
    """Grow one tree over rows partitioned across ranks.

    `shards`, `grad`, and `hess` describe this process's local ranks, one
    entry each per `comm.n_local_ranks()`. Every rank returns the same tree.

    Validation is collective: each rank records a status instead of raising,
    the statuses are reduced, and every rank then raises the same error
    naming the lowest-numbered failing rank. A run where one rank raises
    while the others block in a collective it will never call is the failure
    mode this ordering exists to prevent.

    The named fields and the schema digest are agreed in the same reduction,
    so a parameter the field list does not name still cannot differ between
    ranks unnoticed. The row partition is not negotiated here: one tree is
    grown from whatever rows the caller handed each rank, and it is
    `train_distributed_run` that owns the partition for a whole run.
    """
    agree_status(comm, _shape_statuses(comm, shards, grad, hess))

    var layout = _shard_layout(shards)
    var mask = _unsupported_mask(params, shards)
    _agree_config_and_schema(
        comm,
        [layout[0], layout[1], params.num_leaves, mask],
        ["n_features", "n_bins", "num_leaves", "the configuration"],
        tree_schema_digest(shards, params, comm.world_size(), mask),
    )
    _raise_unsupported(mask)
    params.constraints.check_features(layout[0])

    return _grow_tree_distributed(shards, grad, hess, params, comm, options)


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


def _agree_control[C: Collective](mut comm: C, code: Int) raises -> Int:
    """Turn one rank's callback verdict into the world's.

    A maximum, so the strictest verdict wins: `ABORT` over `STOP` over
    `CONTINUE`. Every rank then acts on the same code, which is what keeps a
    callback that stops on rank 2 from leaving ranks 0 and 1 waiting in a
    histogram reduction rank 2 will never join. A world hosted in one process
    already has one verdict and reduces nothing.
    """
    if hosts_whole_world(comm):
        return code
    var buf: List[Int] = [code]
    comm.allreduce_max_int(buf)
    return buf[0]


def _check_params_unchanged(
    before: BoosterParams, after: BoosterParams
) raises:
    """Refuse the parameter reset a callback may schedule.

    The single-node loop honors one through `check_resettable`. This one cannot
    yet: `num_leaves`, `n_estimators`, and the learning rate are all in the
    configuration every rank agreed on before the first collective, and
    changing one on some ranks and not others would desynchronize the run in a
    way no reduction would catch. Refusing says so; ignoring the write would
    produce a model the callback believes it shaped.
    """
    if (
        before.n_estimators != after.n_estimators
        or before.learning_rate != after.learning_rate
        or before.tree.num_leaves != after.tree.num_leaves
    ):
        raise Error(
            "distributed training does not support a callback parameter"
            " reset: n_estimators, learning_rate, and num_leaves are part of"
            " the configuration every rank agreed on before the first"
            " collective, so changing one mid-run would desynchronize the"
            " world"
        )


def train_distributed_run[
    C: Collective, K: IterationFn & Copyable
](
    shards: List[DataShard],
    objective: Int,
    params: BoosterParams,
    mut comm: C,
    options: DistributedRunOptions,
    on_iteration: K,
    alpha: Float64 = 0.9,
) raises -> DistributedOutcome:
    """Train a boosted ensemble over rows partitioned across ranks.

    This is the one training loop; `train_distributed` is this with default
    options and an inert callback, so a run with options off and a run without
    them take the same code and send the same messages.

    `shards` describes this process's local ranks, one per
    `comm.n_local_ranks()`. Every rank returns the same `Booster` and the same
    report, because every value in both is either a constant of the
    configuration or derived from a reduced quantity. The model is an ordinary
    one: it predicts and serializes exactly like one trained on a single node,
    because sharding changes how the histograms are summed and nothing about
    the model.

    Supported objectives are squared error, binary logistic, poisson, and
    huber. Quantile and L1 replace leaf values with a percentile of the leaf's
    residuals, and a percentile is not a sum, so it does not all-reduce;
    approximating it would make distributed training disagree with single-node
    training in a way no tolerance test would catch, so it is refused instead.
    Ranking and multiclass are refused as well. Early stopping is implemented
    here, on the global training loss, and off unless asked for.

    The order of the collectives is fixed and every rank issues all of them:
    configuration and schema, then the target statuses, then the partition,
    then the base score, then per round the tree's reductions and whichever of
    the metric, the callback verdict, and the checkpoint barrier are enabled.
    Nothing is skipped on a rank with no rows.
    """
    options.check()

    var mask = _unsupported_mask(params.tree, shards)
    if len(options.groups) > 0:
        mask |= _UNSUPPORTED_RANKING
    var layout = _shard_layout(shards)
    _agree_config_and_schema(
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
        run_schema_digest(
            shards, params, objective, alpha, comm.world_size(), mask
        ),
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

    # The partition, agreed rather than assumed, and then checked against the
    # rows each rank actually holds. Both the ranking constraint and the
    # checkpoint record are stated against it. The check is skipped, not
    # failed, when this process hosts the whole world: the plan was built from
    # these very shards a line earlier, so it cannot disagree with them, and
    # the collective would confirm a value against itself.
    var plan = resolve_partition(comm, shards)
    if not hosts_whole_world(comm):
        agree_status(comm, _partition_statuses(comm, shards, plan))
    if len(options.groups) > 0:
        check_group_alignment(plan, options.groups)

    var n_local = len(shards)
    var base_score = _distributed_base_score(comm, shards, objective)

    var raw = List[List[Float64]]()
    for i in range(n_local):
        var r = List[Float64](capacity=shards[i].data.n_rows)
        for _ in range(shards[i].data.n_rows):
            r.append(base_score)
        raw.append(r^)

    var report = DistributedRunReport(plan^, options.schema)
    # One "validation set", the training rows themselves, and one metric on it:
    # the global loss `distributed_metric` reduces. A callback therefore reads
    # `env.value(0, 0)` here exactly as it reads a validation metric on the
    # single-node path, and reads nothing at all in a run that scores nothing.
    var env = IterationEnv(params.copy(), ["training"], ["loss"])
    var scores_metric = options.metric_every > 0 or options.scores_every_round()
    var best_metric = 0.0
    var best_round = 0
    var have_best = False

    var trees = List[Tree]()
    for i in range(params.n_estimators):
        env.iteration = i
        env.evaluation.clear()
        env.params = params.copy()
        var before = _agree_control(comm, on_iteration(BEFORE_ITERATION, env))
        if before == ABORT:
            raise Error(
                String(
                    "distributed training callback failed in the"
                    " before-iteration phase of round ",
                    i,
                )
            )
        if before == STOP:
            report.stopped_by_callback = True
            break
        _check_params_unchanged(params, env.params)

        # Gradients and hessians are a per-row function of local state only.
        var grad = List[List[Float64]]()
        var hess = List[List[Float64]]()
        for j in range(n_local):
            var g = List[Float64]()
            var h = List[Float64]()
            _fill_grad_hess(
                raw[j],
                shards[j].target,
                objective,
                shards[j].weight,
                alpha,
                g,
                h,
            )
            grad.append(g^)
            hess.append(h^)

        var tree = _grow_tree_distributed(
            shards, grad, hess, params.tree, comm, options
        )

        # The tree is global, so this convergence break is unanimous: no
        # rank can leave the loop while another rank stays in it and blocks
        # on a collective.
        if tree.n_leaves == 1 and abs(tree.value[0]) < 1e-12:
            break

        for j in range(n_local):
            var updated = List[Float64](capacity=len(raw[j]))
            for r in range(shards[j].data.n_rows):
                updated.append(
                    raw[j][r]
                    + params.learning_rate * tree.predict_row(shards[j].data, r)
                )
            raw[j] = updated^
        trees.append(tree^)
        report.rounds_run = len(trees)

        # Scoring, early stopping, and the checkpoint boundary are all
        # decisions about a reduced scalar or an unconditional barrier, so
        # every rank takes the same one at the same round.
        var stop_now = False
        var scored = False
        var due = options.scores_every_round()
        if not due and options.metric_every > 0:
            due = len(trees) % options.metric_every == 0
        if scores_metric and due:
            scored = True
            var value = distributed_metric(comm, shards, raw, objective, alpha)
            report.metric_rounds.append(len(trees))
            report.metric_values.append(value)
            var improved = not have_best
            if not improved:
                improved = value < best_metric - options.early_stopping_delta
            if improved:
                best_metric = value
                best_round = len(trees)
                have_best = True
            if (
                options.early_stopping_rounds > 0
                and len(trees) - best_round >= options.early_stopping_rounds
            ):
                report.stopped_early = True
                stop_now = True

        if options.checkpoint_every > 0 and (
            len(trees) % options.checkpoint_every == 0
        ):
            comm.barrier()
            report.checkpoints.append(
                CheckpointMeta(
                    options.job_id,
                    options.schema,
                    model_digest(
                        Booster(
                            trees.copy(),
                            base_score,
                            params.learning_rate,
                            objective,
                            params.tree.monotone.copy(),
                        )
                    ),
                    len(report.checkpoints),
                    0,
                    comm.world_size(),
                    len(trees),
                )
            )

        # Only this round's score, never the last one that happened to exist: a
        # callback reading `env.value(0, 0)` on a round that scored nothing
        # would otherwise act on a metric from several rounds ago. An empty
        # `evaluation` makes `value` raise, which is the same contract the
        # before-iteration phase already has.
        env.iteration = i
        env.evaluation.clear()
        if scored:
            env.evaluation.append(
                report.metric_values[len(report.metric_values) - 1]
            )
        var after = _agree_control(comm, on_iteration(AFTER_ITERATION, env))
        if after == ABORT:
            raise Error(
                String(
                    "distributed training callback failed in the"
                    " after-iteration phase of round ",
                    i,
                )
            )
        if after == STOP:
            report.stopped_by_callback = True
            break
        if stop_now:
            break

    var model = Booster(
        trees^,
        base_score,
        params.learning_rate,
        objective,
        params.tree.monotone.copy(),
    )
    report.model_digest = model_digest(model)
    return DistributedOutcome(model^, report^)


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

    `train_distributed_run` with the default options and no callback, kept as
    the plain entry point for a caller that wants a model and nothing else.

    The model comes out as a copy, not a transfer. Moving one field out of a
    live struct is a partial move, which this compiler rejects outright: the
    outcome still owns its report, so it still has to be destroyed as a whole.
    The copy is the same one the bindings already take out of a training
    result, and it is a copy of a finished ensemble taken once at the end of a
    whole distributed run, so it costs nothing next to the training itself.
    """
    var outcome = train_distributed_run(
        shards,
        objective,
        params,
        comm,
        DistributedRunOptions(),
        no_callback,
        alpha,
    )
    return outcome.model.copy()


def run_distributed[
    K: IterationFn & Copyable
](
    spec: RuntimeSpec,
    shards: List[DataShard],
    objective: Int,
    params: BoosterParams,
    options: DistributedRunOptions,
    on_iteration: K,
    alpha: Float64 = 0.9,
) raises -> DistributedOutcome:
    """Train through a runtime spec: the one entry point for bindings and Dask.

    It does four things and no more. It refuses a spec this build cannot open,
    before a row is partitioned, so a rank configured for a world it cannot
    reach fails with the reason rather than blocking in a handshake that will
    never complete. It opens the collective the spec names: a world hosted in
    this process for `RUNTIME_LOCAL`, or a rendezvous over sockets for
    `RUNTIME_TRANSPORT`. It stamps the run with the spec's job id and schema
    digest, so a checkpoint records what a restart would have to match. And it
    closes a transport session on the way out, whether the run succeeded or
    raised, because a rank that leaves its descriptors open leaves its peers
    blocked rather than told.

    A caveat that belongs on the entry point rather than only in the handoff:
    the socket path here has never been run. `transport_validated()` in
    distributed_transport.mojo is False, and every claim about multi-process
    or multi-host training waits on the two-process procedure in
    docs/DISTRIBUTED_TRANSPORT.md section 7. What is connected is the whole
    path from spec to rendezvous to reduction; what is unproven is the
    syscall layer at the bottom of it.

    A caller that has already connected its own endpoints, or that wants to
    drive the protocol over `MemoryEndpoint`, composes
    `open_transport_collective` with `train_distributed_run` instead and does
    not come through here.

    Pass `no_callback` from callback.mojo when there is nothing to observe.
    """
    spec.validate()
    require_transport(spec)
    var stamped = options.copy()
    stamped.job_id = spec.job_id
    stamped.schema = spec.schema_digest()
    if spec.mode == RUNTIME_LOCAL:
        var comm = open_local_collective(spec)
        return train_distributed_run(
            shards, objective, params, comm, stamped, on_iteration, alpha
        )
    if spec.mode != RUNTIME_TRANSPORT:
        raise Error(
            "run_distributed opens a local or a transport runtime, and this"
            " spec is neither"
        )
    # Blocks here until every rank of the world has connected and announced
    # itself, or the spec's connect deadline passes.
    var wire = open_socket_collective(spec)
    var outcome: DistributedOutcome
    try:
        outcome = train_distributed_run(
            shards, objective, params, wire, stamped, on_iteration, alpha
        )
    except e:
        # Shut down before the error escapes, and swallow whatever shutting
        # down reports: the failure worth raising is the one that got us here,
        # and a session that already lost a peer will complain about it again.
        try:
            wire.shutdown()
        except:
            pass
        raise e
    wire.shutdown()
    return outcome^
