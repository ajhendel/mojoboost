"""GPU prediction and device-side validation scoring.

Prediction is the other half of the GPU story: `train_gpu` grows trees on the
device (see train_gpu.mojo) but every score it needs, for the training rows
and for any validation set, is still walked tree by tree on the host. This
module moves that walk to the device and keeps the pieces a training loop
needs resident across rounds.

Two shapes of work, one kernel:

- Batch scoring. A whole binned matrix is uploaded once and every row is
  walked through a range of boosting iterations, producing raw scores,
  response-scale predictions, class probabilities, or per-tree leaf
  ordinals. This is what a Python `predict(..., device="gpu")` call needs.
- Incremental validation scoring. A validation matrix, its labels, its
  weights, and a running raw-score vector stay device-resident for a whole
  training run. Each round uploads only the trees that round grew and adds
  their contribution into the resident raw scores, so scoring round `i`
  costs one tree per output rather than `i` of them. A device-side reduction
  then turns those scores into a metric value, which is what early stopping
  watches.

The ensemble crosses to the device as flat arrays (`FlatEnsemble`), not as
`Tree` objects: node fields interleave into one Int32 array with
`NODE_STRIDE` entries per node, child links are absolute indices into that
array, and every tree's categorical bitsets are concatenated into a single
pool. A tree walk is then a pointer chase over one buffer, which is what the
kernel does, one thread per (row, output).

Routing is the same rule everywhere. A categorical node tests set membership
in the node's 256-bit slice of the pool, a row in the node's missing bin
follows the node's default direction, and everything else compares against
the threshold bin. That is `Tree.goes_left` on the host and
`_partition_kernel` on the device, written a third time here against the
flat layout; the tests assert all three agree by comparing predictions.

Precision. Apple GPUs have no Float64, so leaf values, base scores, and the
raw-score accumulator are Float32. Agreement with the CPU predictor is
therefore to Float32 tolerance, not bit-exact, exactly as it is for GPU
histograms. Prediction is still bit-deterministic run to run: one thread
sums one row's trees in ascending iteration order, so no reduction order can
vary. Routing itself is exact, because bins are integers: a row reaches the
same leaf on both backends, and only the sum of the leaf values it collects
rounds differently.

Binning stays on the host. Bin edges are Float64 and a routing decision is
discrete, so binning a raw value against Float32 edges could put a row in a
different bin and change its prediction by a whole leaf value rather than by
a rounding step. `BinMapper.transform` therefore runs host-side (it is
already parallel across features) and only the resulting bins are uploaded.
Device-side binning would need the edge search carried out in a way that
provably agrees with the Float64 one; that is a separate piece of work.

Metric reduction. `_metric_kernel` computes each row's weighted contribution
and reduces it within a threadgroup through shared memory in a fixed tree
order; the host sums the per-threadgroup partials in ascending block order
in Float64 and divides by the total weight, which was validated and summed
in Float64 when the validation set was uploaded. The reduction order is
fixed, so the value is deterministic. The metrics are the ones early
stopping needs (`l2`, `l1`, binary and multiclass log loss, and the two
error rates); their definitions match metrics.mojo term for term, with the
one documented difference in `_clamp32` below.

Not here. Feature contributions (TreeSHAP, contrib.mojo) walk every node of
every tree with a per-node weight vector rather than following one root-to-
leaf path, so they need a different kernel and a different memory budget;
they stay a documented follow-up. Leaf-index prediction is the same walk, so
it is implemented.
"""

from std.gpu import block_dim, block_idx, global_idx, thread_idx
from std.math import exp, log
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from .binning import BinnedMatrix
from .boosting import (
    BINARY_LOGISTIC,
    CROSS_ENTROPY,
    GAMMA,
    POISSON,
    TWEEDIE,
    Booster,
    IterationRange,
    MulticlassBooster,
)
from .categorical import CAT_BITSET_WORDS
from .gpu_tiling import DeviceCaps, derive_block_threads, query_device_caps
from .metrics import check_metric_weight
from .tree import Tree


# One node's fields, interleaved: a walk reads several of them for the same
# node, so one stride keeps them in the same cache line. Slot 7 is the
# node's leaf ordinal (see `Tree.leaf_ordinals`) on a leaf and -1 on an
# internal node; slots 1, 4, and 5 are unused on a leaf.
comptime NODE_STRIDE = 8
comptime NODE_FEATURE = 0
comptime NODE_THRESHOLD = 1
comptime NODE_LEFT = 2
comptime NODE_RIGHT = 3
comptime NODE_MISSING = 4
comptime NODE_DEFAULT_LEFT = 5
comptime NODE_CAT = 6
comptime NODE_LEAF_ORDINAL = 7

# The inverse links `Booster.response` applies, as device codes. The kernel
# takes one of these rather than an objective id so it does not have to
# track the objective registry in boosting.mojo.
comptime RESPONSE_IDENTITY = 0
comptime RESPONSE_SIGMOID = 1
comptime RESPONSE_EXP = 2
comptime RESPONSE_SOFTMAX = 3

# Device metrics, all reported as weighted sums by the kernel and finished
# on the host. ACCURACY variants are reduced rather than the error rates
# themselves because metrics.mojo defines the error rates as one minus the
# accuracy, and subtracting once on the host keeps the two definitions the
# same expression.
comptime METRIC_L2 = 0
comptime METRIC_L1 = 1
comptime METRIC_BINARY_LOG_LOSS = 2
comptime METRIC_MULTICLASS_LOG_LOSS = 3
comptime METRIC_BINARY_ACCURACY = 4
comptime METRIC_MULTICLASS_ACCURACY = 5

# Threads per threadgroup for the metric reduction. Fixed, and a power of
# two, because the shared-memory tree reduction halves the active range and
# the shared array is sized at compile time. It is a warp multiple on every
# supported backend.
comptime REDUCE_BLOCK = 256

# Rows and node ids cross into the kernels as Int32.
comptime MAX_ROWS = Int(Int32.MAX)

# Probability floor for the device log losses. metrics.mojo clamps to 1e-15,
# which Float32 cannot hold away from 1.0: `1 - 1e-15` rounds to exactly 1
# and the log of the complement becomes -inf. 1e-7 is about the Float32
# epsilon, so it is the tightest clamp that still leaves `1 - eps` distinct
# from 1. A device log loss therefore saturates near 16.1 where the host one
# saturates near 34.5; the two agree to Float32 tolerance everywhere the
# clamp does not engage, which is everywhere a probability is not already
# numerically certain.
comptime _F32_PROB_FLOOR = Float32(1e-7)


@always_inline
def _clamp32(p: Float32) -> Float32:
    """`metrics.mojo`'s `_clamp` at Float32 width (see `_F32_PROB_FLOOR`)."""
    if p < _F32_PROB_FLOOR:
        return _F32_PROB_FLOOR
    if p > Float32(1.0) - _F32_PROB_FLOOR:
        return Float32(1.0) - _F32_PROB_FLOOR
    return p


@always_inline
def _sigmoid32(x: Float32) -> Float32:
    """The stable logistic function boosting.mojo's `_sigmoid` computes, at
    Float32 width: the branch keeps `exp` away from overflow on either
    tail."""
    if x >= Float32(0.0):
        var e = exp(-x)
        return Float32(1.0) / (Float32(1.0) + e)
    var e = exp(x)
    return e / (Float32(1.0) + e)


def _predict_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    nodes: MutPointer[Int32, MutAnyOrigin],
    values: MutPointer[Float32, MutAnyOrigin],
    cat_pool: MutPointer[UInt64, MutAnyOrigin],
    tree_root: MutPointer[Int32, MutAnyOrigin],
    base: MutPointer[Float32, MutAnyOrigin],
    scores: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_outputs: Int32,
    iter_begin: Int32,
    iter_end: Int32,
    learning_rate: Float32,
    include_base: Int32,
    accumulate: Int32,
):
    """Raw scores for one (row, output) pair per thread.

    `grid.x` tiles the rows and `grid.y` indexes the output (the class for a
    multiclass ensemble, and the single output otherwise), so the whole
    (row, class) plane is one launch. Trees are stored round-major, so the
    tree of iteration `i` and output `k` is `i * n_outputs + k`, which is the
    layout `MulticlassBooster` documents and the single-output case
    degenerates to.

    `include_base` adds the output's base score, which belongs to iteration 0
    (see `IterationRange`), and `accumulate` adds into the output buffer
    instead of overwriting it, which is how a training loop folds one round's
    trees into a resident raw-score vector.
    """
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nr = Int(n_rows)
    if r >= nr:
        return
    var k = Int(block_idx.y)
    var n_out = Int(n_outputs)

    var acc = Float32(0.0)
    if include_base != 0:
        acc = base[unsafe_offset=k][0]

    for i in range(Int(iter_begin), Int(iter_end)):
        var node = Int(tree_root[unsafe_offset = i * n_out + k][0])
        while True:
            var nb = node * NODE_STRIDE
            var f = Int(nodes[unsafe_offset = nb + NODE_FEATURE][0])
            if f < 0:
                break
            var bin = Int(bins[unsafe_offset = f * nr + r])
            var go_left: Bool
            var cat = Int(nodes[unsafe_offset = nb + NODE_CAT][0])
            if cat >= 0:
                # The node's 256-bit category set, as four words in the
                # shared pool. Bin 0 of a categorical feature (missing,
                # unseen, or dropped) is never a member, so those rows go
                # right, which is what `Tree.goes_left` does.
                var word = cat_pool[unsafe_offset = cat + (bin >> 6)][0]
                go_left = ((word >> UInt64(bin & 63)) & 1) != 0
            elif bin == Int(nodes[unsafe_offset = nb + NODE_MISSING][0]):
                go_left = nodes[unsafe_offset = nb + NODE_DEFAULT_LEFT][0] != 0
            else:
                go_left = bin <= Int(
                    nodes[unsafe_offset = nb + NODE_THRESHOLD][0]
                )
            if go_left:
                node = Int(nodes[unsafe_offset = nb + NODE_LEFT][0])
            else:
                node = Int(nodes[unsafe_offset = nb + NODE_RIGHT][0])
        acc += learning_rate * values[unsafe_offset=node][0]

    var idx = r * n_out + k
    if accumulate != 0:
        scores[unsafe_offset=idx] = scores[unsafe_offset=idx][0] + acc
    else:
        scores[unsafe_offset=idx] = acc


def _leaf_kernel(
    bins: MutPointer[UInt8, MutAnyOrigin],
    nodes: MutPointer[Int32, MutAnyOrigin],
    cat_pool: MutPointer[UInt64, MutAnyOrigin],
    tree_root: MutPointer[Int32, MutAnyOrigin],
    leaves: MutPointer[Int32, MutAnyOrigin],
    n_rows: Int32,
    n_outputs: Int32,
    iter_begin: Int32,
    iter_end: Int32,
):
    """Per-tree leaf ordinals: the same walk as `_predict_kernel`, reporting
    where the row landed instead of what it collected.

    One thread owns one (row, output) pair and writes one entry per iteration
    in the range, so the output is row-major
    `[r * n_iters * n_outputs + (i - iter_begin) * n_outputs + k]`, the
    round-major-within-a-row layout `MulticlassBooster.leaf_indices_bins`
    returns for a single example.
    """
    var r = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var nr = Int(n_rows)
    if r >= nr:
        return
    var k = Int(block_idx.y)
    var n_out = Int(n_outputs)
    var n_iters = Int(iter_end) - Int(iter_begin)

    for i in range(Int(iter_begin), Int(iter_end)):
        var node = Int(tree_root[unsafe_offset = i * n_out + k][0])
        while True:
            var nb = node * NODE_STRIDE
            var f = Int(nodes[unsafe_offset = nb + NODE_FEATURE][0])
            if f < 0:
                break
            var bin = Int(bins[unsafe_offset = f * nr + r])
            var go_left: Bool
            var cat = Int(nodes[unsafe_offset = nb + NODE_CAT][0])
            if cat >= 0:
                var word = cat_pool[unsafe_offset = cat + (bin >> 6)][0]
                go_left = ((word >> UInt64(bin & 63)) & 1) != 0
            elif bin == Int(nodes[unsafe_offset = nb + NODE_MISSING][0]):
                go_left = nodes[unsafe_offset = nb + NODE_DEFAULT_LEFT][0] != 0
            else:
                go_left = bin <= Int(
                    nodes[unsafe_offset = nb + NODE_THRESHOLD][0]
                )
            if go_left:
                node = Int(nodes[unsafe_offset = nb + NODE_LEFT][0])
            else:
                node = Int(nodes[unsafe_offset = nb + NODE_RIGHT][0])
        var slot = (i - Int(iter_begin)) * n_out + k
        leaves[unsafe_offset = r * n_iters * n_out + slot] = nodes[
            unsafe_offset = node * NODE_STRIDE + NODE_LEAF_ORDINAL
        ][0]


def _response_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_outputs: Int32,
    code: Int32,
):
    """Apply the objective's inverse link to a raw-score buffer.

    One thread owns one whole row, because the softmax is a row-wise
    reduction; the element-wise links take the same shape so there is one
    kernel rather than two. `src` and `dst` may be the same buffer: a thread
    reads and writes only its own row.
    """
    var r = Int(global_idx.x)
    if r >= Int(n_rows):
        return
    var n_out = Int(n_outputs)
    var base = r * n_out
    var c = Int(code)

    if c == RESPONSE_SOFTMAX:
        # Max-subtracted softmax, term for term the one `_softmax_inplace`
        # computes on the host.
        var m = src[unsafe_offset=base][0]
        for i in range(1, n_out):
            var v = src[unsafe_offset = base + i][0]
            if v > m:
                m = v
        var total = Float32(0.0)
        for i in range(n_out):
            var e = exp(src[unsafe_offset = base + i][0] - m)
            dst[unsafe_offset = base + i] = e
            total += e
        for i in range(n_out):
            dst[unsafe_offset = base + i] = (
                dst[unsafe_offset = base + i][0] / total
            )
        return

    for i in range(n_out):
        var v = src[unsafe_offset = base + i][0]
        if c == RESPONSE_SIGMOID:
            dst[unsafe_offset = base + i] = _sigmoid32(v)
        elif c == RESPONSE_EXP:
            dst[unsafe_offset = base + i] = exp(v)
        else:
            dst[unsafe_offset = base + i] = v


def _metric_kernel(
    resp: MutPointer[Float32, MutAnyOrigin],
    label: MutPointer[Float32, MutAnyOrigin],
    weight: MutPointer[Float32, MutAnyOrigin],
    partial: MutPointer[Float32, MutAnyOrigin],
    n_rows: Int32,
    n_outputs: Int32,
    metric: Int32,
    has_weight: Int32,
):
    """One threadgroup's share of a metric's weighted sum.

    Each thread scores one row on the response scale, then the threadgroup
    reduces through shared memory by repeated halving, a fixed order that
    does not depend on scheduling. Block `b` writes its partial to
    `partial[b]`; the host sums the partials in ascending order in Float64
    and divides by the total weight, so the division and the final
    accumulation keep full precision even though the per-row terms are
    Float32.
    """
    var s = stack_allocation[
        REDUCE_BLOCK, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var r = Int(block_idx.x) * Int(block_dim.x) + tid
    var n_out = Int(n_outputs)
    var m = Int(metric)

    var term = Float32(0.0)
    if r < Int(n_rows):
        var y = label[unsafe_offset=r][0]
        var w = Float32(1.0)
        if has_weight != 0:
            w = weight[unsafe_offset=r][0]
        if m == METRIC_L2:
            var d = resp[unsafe_offset = r * n_out][0] - y
            term = w * d * d
        elif m == METRIC_L1:
            term = w * abs(resp[unsafe_offset = r * n_out][0] - y)
        elif m == METRIC_BINARY_LOG_LOSS:
            var p = _clamp32(resp[unsafe_offset = r * n_out][0])
            if y > Float32(0.5):
                term = -w * log(p)
            else:
                term = -w * log(Float32(1.0) - p)
        elif m == METRIC_MULTICLASS_LOG_LOSS:
            var c = Int(y)
            term = -w * log(_clamp32(resp[unsafe_offset = r * n_out + c][0]))
        elif m == METRIC_BINARY_ACCURACY:
            var predicted = Float32(0.0)
            if resp[unsafe_offset = r * n_out][0] >= Float32(0.5):
                predicted = Float32(1.0)
            if abs(predicted - y) < Float32(0.5):
                term = w
        elif m == METRIC_MULTICLASS_ACCURACY:
            var argmax = 0
            for i in range(1, n_out):
                if (
                    resp[unsafe_offset = r * n_out + i][0]
                    > resp[unsafe_offset = r * n_out + argmax][0]
                ):
                    argmax = i
            if argmax == Int(y):
                term = w

    s[unsafe_offset=tid] = term
    barrier()

    # Every thread computes the same `active`, so every thread reaches every
    # barrier the same number of times.
    var active = REDUCE_BLOCK // 2
    while active > 0:
        if tid < active:
            s[unsafe_offset=tid] = (
                s[unsafe_offset=tid][0] + s[unsafe_offset = tid + active][0]
            )
        barrier()
        active >>= 1

    if tid == 0:
        partial[unsafe_offset = Int(block_idx.x)] = s[unsafe_offset=0][0]


def response_for_objective(objective: Int) -> Int:
    """The device response code matching `Booster.response` for a built-in
    objective. CUSTOM and every objective without a link map to
    RESPONSE_IDENTITY, which is what `Booster.response` returns for them."""
    if objective == BINARY_LOGISTIC or objective == CROSS_ENTROPY:
        return RESPONSE_SIGMOID
    if objective == POISSON or objective == GAMMA or objective == TWEEDIE:
        return RESPONSE_EXP
    return RESPONSE_IDENTITY


struct FlatEnsemble(Copyable, Movable):
    """A tree ensemble in the flat arrays the prediction kernels walk.

    `nodes` interleaves every tree's nodes at `NODE_STRIDE` Int32 per node,
    with child links already rebased to absolute indices, so a walk needs no
    per-tree offset arithmetic beyond its root. `values` is one Float32 leaf
    value per node (only leaves are read). `cat_pool` concatenates every
    tree's categorical bitsets and `NODE_CAT` holds the absolute offset into
    it, or -1 on a numerical node. `tree_root[t]` is tree t's root node
    index, and trees are round-major: iteration i of output k is tree
    `i * n_outputs + k`.

    `base_scores` has one entry per output and `learning_rate` is the
    ensemble's shrinkage; both are applied by the kernel rather than folded
    in here, so the flat form stays a faithful copy of the trees.
    """

    var nodes: List[Int32]
    var values: List[Float32]
    var cat_pool: List[UInt64]
    var tree_root: List[Int32]
    var base_scores: List[Float32]
    var learning_rate: Float64
    var n_outputs: Int

    def __init__(
        out self,
        var nodes: List[Int32],
        var values: List[Float32],
        var cat_pool: List[UInt64],
        var tree_root: List[Int32],
        var base_scores: List[Float32],
        learning_rate: Float64,
        n_outputs: Int,
    ):
        self.nodes = nodes^
        self.values = values^
        self.cat_pool = cat_pool^
        self.tree_root = tree_root^
        self.base_scores = base_scores^
        self.learning_rate = learning_rate
        self.n_outputs = n_outputs

    def n_trees(self) -> Int:
        return len(self.tree_root)

    def n_nodes(self) -> Int:
        return len(self.values)

    def n_iterations(self) -> Int:
        """Boosting iterations this flat ensemble holds: one iteration is
        one tree per output."""
        return len(self.tree_root) // self.n_outputs


def _append_tree(
    tree: Tree,
    mut nodes: List[Int32],
    mut values: List[Float32],
    mut cat_pool: List[UInt64],
    mut tree_root: List[Int32],
) raises:
    """Append one tree to the flat arrays, rebasing its child links and its
    categorical offsets into the shared node array and bitset pool.

    Children are validated to sit after their parent. Every grower appends
    child nodes after the node it splits, so that always holds for a grown
    or deserialized tree; checking it here is what guarantees the device
    walk terminates, since a cycle in the links would hang the kernel rather
    than raise.
    """
    var n_nodes = len(tree.feature)
    if n_nodes == 0:
        raise Error("cannot predict from a tree with no nodes")
    var node_base = len(values)
    var pool_base = len(cat_pool)
    if node_base + n_nodes > MAX_ROWS:
        raise Error("GPU prediction supports at most 2^31 - 1 ensemble nodes")

    tree_root.append(Int32(node_base))
    for w in range(len(tree.cat_bitset)):
        cat_pool.append(tree.cat_bitset[w])

    var next_ordinal = 0
    for i in range(n_nodes):
        var feature = tree.feature[i]
        var leaf_ordinal = -1
        if feature < 0:
            leaf_ordinal = next_ordinal
            next_ordinal += 1
        else:
            if tree.left[i] <= i or tree.right[i] <= i:
                raise Error(
                    "tree node links must point forward; this ensemble"
                    " cannot be walked on the device"
                )
            if tree.left[i] >= n_nodes or tree.right[i] >= n_nodes:
                raise Error("tree node link out of range")
        var cat = tree.cat_offset[i]
        if cat >= 0:
            if cat + CAT_BITSET_WORDS > len(tree.cat_bitset):
                raise Error("categorical node bitset offset out of range")
            cat += pool_base
        nodes.append(Int32(feature))
        nodes.append(Int32(tree.threshold_bin[i]))
        nodes.append(Int32(node_base + tree.left[i]))
        nodes.append(Int32(node_base + tree.right[i]))
        nodes.append(Int32(tree.missing_bin[i]))
        nodes.append(Int32(1) if tree.default_left[i] else Int32(0))
        nodes.append(Int32(cat))
        nodes.append(Int32(leaf_ordinal))
        values.append(Float32(tree.value[i]))


def flatten_trees(
    trees: List[Tree],
    base_scores: List[Float64],
    n_outputs: Int,
    learning_rate: Float64,
) raises -> FlatEnsemble:
    """Flatten a round-major tree list into the device layout.

    `trees` must hold a whole number of iterations of `n_outputs` trees
    each, `base_scores` one entry per output. An empty tree list is allowed
    and yields a base-score-only ensemble.
    """
    if n_outputs < 1:
        raise Error("n_outputs must be at least one")
    if len(base_scores) != n_outputs:
        raise Error("base_scores must have one entry per output")
    if len(trees) % n_outputs != 0:
        raise Error("tree count must be a multiple of n_outputs")

    var nodes = List[Int32](capacity=NODE_STRIDE * len(trees))
    var values = List[Float32](capacity=len(trees))
    var cat_pool = List[UInt64]()
    var tree_root = List[Int32](capacity=len(trees))
    for t in range(len(trees)):
        _append_tree(trees[t], nodes, values, cat_pool, tree_root)

    var base = List[Float32](capacity=n_outputs)
    for k in range(n_outputs):
        base.append(Float32(base_scores[k]))
    return FlatEnsemble(
        nodes^, values^, cat_pool^, tree_root^, base^, learning_rate, n_outputs
    )


def flatten_booster(booster: Booster) raises -> FlatEnsemble:
    """Flatten a fitted single-output ensemble."""
    var base: List[Float64] = [booster.base_score]
    return flatten_trees(booster.trees, base, 1, booster.learning_rate)


def flatten_multiclass(booster: MulticlassBooster) raises -> FlatEnsemble:
    """Flatten a fitted softmax ensemble. `MulticlassBooster` already stores
    its trees round-major, which is the layout the kernel expects."""
    return flatten_trees(
        booster.trees,
        booster.base_scores,
        booster.n_classes,
        booster.learning_rate,
    )


def _f32_list(values: List[Float64]) -> List[Float32]:
    var out = List[Float32](capacity=len(values))
    for i in range(len(values)):
        out.append(Float32(values[i]))
    return out^


struct GpuPredictor(Movable):
    """Device-resident tree walker.

    Construct once for an ensemble shape (`n_features` and `n_outputs`),
    `upload_ensemble` whenever the trees change, and then either score whole
    binned matrices (`raw_scores`, `response_scores`, `leaf_indices`) or run
    the resident validation path (`set_validation`, `reset_validation`,
    `accumulate_round`, `validation_metric`).

    Batch buffers grow to the largest batch seen and are then reused, so a
    repeated `predict` call over the same shape allocates nothing. The
    validation buffers are separate and sized once by `set_validation`, so
    an ad-hoc prediction during a training run cannot disturb the running
    validation scores.
    """

    var ctx: DeviceContext
    # Flat ensemble, device side.
    var nodes_dev: DeviceBuffer[DType.int32]
    var values_dev: DeviceBuffer[DType.float32]
    var cat_dev: DeviceBuffer[DType.uint64]
    var root_dev: DeviceBuffer[DType.int32]
    var base_dev: DeviceBuffer[DType.float32]
    var n_trees: Int
    var learning_rate: Float64
    # Batch scoring buffers, grown on demand. Transfers are sized by the
    # buffer, not by the batch, so a batch smaller than the high-water mark
    # stages through a pinned host buffer of the buffer's own size rather
    # than having the copy read past the end of the caller's matrix.
    var bins_dev: DeviceBuffer[DType.uint8]
    var stage_bins: HostBuffer[DType.uint8]
    var out_dev: DeviceBuffer[DType.float32]
    var resp_dev: DeviceBuffer[DType.float32]
    var host_out: HostBuffer[DType.float32]
    var row_capacity: Int
    # Resident validation set.
    var valid_bins_dev: DeviceBuffer[DType.uint8]
    var valid_label_dev: DeviceBuffer[DType.float32]
    var valid_weight_dev: DeviceBuffer[DType.float32]
    var valid_raw_dev: DeviceBuffer[DType.float32]
    var valid_resp_dev: DeviceBuffer[DType.float32]
    var part_dev: DeviceBuffer[DType.float32]
    var host_part: HostBuffer[DType.float32]
    var valid_rows: Int
    var valid_blocks: Int
    var valid_weighted: Bool
    var valid_total_weight: Float64
    var n_features: Int
    var n_outputs: Int
    var caps: DeviceCaps
    var block_threads: Int

    def __init__(out self, n_features: Int, n_outputs: Int) raises:
        """Open a device and allocate the placeholder buffers. Nothing is
        scored until `upload_ensemble` has run."""
        if n_features < 1:
            raise Error("GPU prediction requires at least one feature")
        if n_outputs < 1:
            raise Error("GPU prediction requires at least one output")
        self.ctx = DeviceContext()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.caps = query_device_caps(self.ctx)
        self.block_threads = derive_block_threads(self.caps)
        self.n_trees = 0
        self.learning_rate = 1.0
        self.row_capacity = 0
        self.valid_rows = 0
        self.valid_blocks = 0
        self.valid_weighted = False
        self.valid_total_weight = 0.0

        # Zero-length device buffers are not portable, so every buffer that
        # is not sized yet holds a single placeholder element.
        self.nodes_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.values_dev = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.cat_dev = self.ctx.enqueue_create_buffer[DType.uint64](1)
        self.root_dev = self.ctx.enqueue_create_buffer[DType.int32](1)
        self.base_dev = self.ctx.enqueue_create_buffer[DType.float32](n_outputs)
        self.bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](1)
        self.stage_bins = self.ctx.enqueue_create_host_buffer[DType.uint8](1)
        self.out_dev = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.resp_dev = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.host_out = self.ctx.enqueue_create_host_buffer[DType.float32](1)
        self.valid_bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](1)
        self.valid_label_dev = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.valid_weight_dev = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.valid_raw_dev = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.valid_resp_dev = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.part_dev = self.ctx.enqueue_create_buffer[DType.float32](1)
        self.host_part = self.ctx.enqueue_create_host_buffer[DType.float32](1)
        self.ctx.synchronize()

    def synchronize(self) raises:
        """Block until every enqueued device operation has completed."""
        self.ctx.synchronize()

    def n_iterations(self) -> Int:
        """Boosting iterations the uploaded ensemble holds."""
        return self.n_trees // self.n_outputs

    def upload_ensemble(mut self, flat: FlatEnsemble) raises:
        """Replace the device-resident ensemble.

        Call it once for a fitted model, or once per boosting round with that
        round's new trees when driving the incremental validation path: a
        round's flat form is a handful of kilobytes, so re-uploading it is
        cheaper than keeping a growable ensemble buffer on the device.
        """
        if flat.n_outputs != self.n_outputs:
            raise Error("ensemble output count does not match the predictor")
        if flat.n_trees() == 0:
            raise Error("cannot upload an ensemble with no trees")

        self.n_trees = flat.n_trees()
        self.learning_rate = flat.learning_rate
        var pool_len = len(flat.cat_pool)
        if pool_len < 1:
            pool_len = 1

        self.nodes_dev = self.ctx.enqueue_create_buffer[DType.int32](
            len(flat.nodes)
        )
        self.values_dev = self.ctx.enqueue_create_buffer[DType.float32](
            len(flat.values)
        )
        self.cat_dev = self.ctx.enqueue_create_buffer[DType.uint64](pool_len)
        self.root_dev = self.ctx.enqueue_create_buffer[DType.int32](
            len(flat.tree_root)
        )

        self.ctx.enqueue_copy(
            dst_buf=self.nodes_dev, src_ptr=flat.nodes.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.values_dev, src_ptr=flat.values.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.root_dev, src_ptr=flat.tree_root.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.base_dev, src_ptr=flat.base_scores.unsafe_ptr()
        )
        if len(flat.cat_pool) > 0:
            self.ctx.enqueue_copy(
                dst_buf=self.cat_dev, src_ptr=flat.cat_pool.unsafe_ptr()
            )
        # The copies read host memory owned by the caller's `flat`, so they
        # have to complete before this returns.
        self.ctx.synchronize()

    def reserve_rows(mut self, n_rows: Int) raises:
        """Size the batch buffers for `n_rows`. Grows only: a smaller batch
        reuses what a larger one allocated."""
        if n_rows <= self.row_capacity:
            return
        if n_rows > MAX_ROWS:
            raise Error("GPU prediction supports at most 2^31 - 1 rows")
        self.bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](
            n_rows * self.n_features
        )
        self.stage_bins = self.ctx.enqueue_create_host_buffer[DType.uint8](
            n_rows * self.n_features
        )
        self.out_dev = self.ctx.enqueue_create_buffer[DType.float32](
            n_rows * self.n_outputs
        )
        self.resp_dev = self.ctx.enqueue_create_buffer[DType.float32](
            n_rows * self.n_outputs
        )
        self.host_out = self.ctx.enqueue_create_host_buffer[DType.float32](
            n_rows * self.n_outputs
        )
        self.row_capacity = n_rows
        self.ctx.synchronize()

    def _check_matrix(self, data: BinnedMatrix) raises:
        if data.n_features != self.n_features:
            raise Error("matrix feature count does not match the predictor")
        if data.n_rows < 1:
            raise Error("GPU prediction requires at least one row")
        if len(data.bins) != data.n_rows * data.n_features:
            raise Error("binned matrix size must equal n_rows * n_features")

    def _check_ready(self) raises:
        if self.n_trees == 0:
            raise Error("call upload_ensemble before predicting")

    def upload_bins(mut self, data: BinnedMatrix) raises:
        """Upload a binned matrix into the batch buffers.

        The matrix is staged into pinned host memory first because the
        device buffer is sized by the high-water batch rather than by this
        one; the tail beyond this batch's cells is never read by a kernel,
        which only launches `data.n_rows` rows."""
        self._check_matrix(data)
        self.reserve_rows(data.n_rows)
        # Any copy still reading the staging buffer has to finish before it
        # is overwritten.
        self.ctx.synchronize()
        var dst = self.stage_bins.unsafe_ptr()
        var src = data.bins.unsafe_ptr()
        var n_cells = data.n_rows * data.n_features
        for i in range(n_cells):
            dst.unsafe_store(i, src.unsafe_load(i))
        self.ctx.enqueue_copy(dst_buf=self.bins_dev, src_ptr=dst)
        self.ctx.synchronize()

    def _row_blocks(self, n_rows: Int) -> Int:
        return (n_rows + self.block_threads - 1) // self.block_threads

    def _enqueue_scores(
        mut self,
        validation: Bool,
        n_rows: Int,
        rng: IterationRange,
        include_base: Bool,
        accumulate: Bool,
    ) raises:
        """Enqueue the walk over `rng` for `n_rows` rows, reading either the
        batch buffers or the resident validation ones. Does not transfer or
        synchronize."""
        var bins_ptr = (
            self.valid_bins_dev.unsafe_ptr() if validation else self.bins_dev.unsafe_ptr()
        )
        var out_ptr = (
            self.valid_raw_dev.unsafe_ptr() if validation else self.out_dev.unsafe_ptr()
        )
        self.ctx.enqueue_function[_predict_kernel](
            bins_ptr,
            self.nodes_dev.unsafe_ptr(),
            self.values_dev.unsafe_ptr(),
            self.cat_dev.unsafe_ptr(),
            self.root_dev.unsafe_ptr(),
            self.base_dev.unsafe_ptr(),
            out_ptr,
            Int32(n_rows),
            Int32(self.n_outputs),
            Int32(rng.start),
            Int32(rng.stop),
            Float32(self.learning_rate),
            Int32(1) if include_base else Int32(0),
            Int32(1) if accumulate else Int32(0),
            grid_dim=(self._row_blocks(n_rows), self.n_outputs),
            block_dim=self.block_threads,
        )

    def _download_out(
        mut self, n: Int, from_response: Bool = False
    ) raises -> List[Float64]:
        """Copy `n` Float32 scores out of a batch buffer and widen them. The
        widening is exact; the scores were accumulated in Float32."""
        if from_response:
            self.ctx.enqueue_copy(
                dst_ptr=self.host_out.unsafe_ptr(), src_buf=self.resp_dev
            )
        else:
            self.ctx.enqueue_copy(
                dst_ptr=self.host_out.unsafe_ptr(), src_buf=self.out_dev
            )
        self.ctx.synchronize()
        var out = List[Float64](capacity=n)
        var src = self.host_out.unsafe_ptr()
        for i in range(n):
            out.append(Float64(src.unsafe_load(i)))
        return out^

    def raw_scores(
        mut self, data: BinnedMatrix, rng: IterationRange
    ) raises -> List[Float64]:
        """Raw ensemble output for every row of `data` over the iterations in
        `rng`, row-major `[r * n_outputs + k]`.

        The base score belongs to iteration 0, so a range starting there
        includes it and a later one does not, exactly as
        `Booster.predict_raw_bins_range` does on the host."""
        self._check_ready()
        self.upload_bins(data)
        self._enqueue_scores(
            False,
            data.n_rows,
            rng,
            rng.includes_base(),
            False,
        )
        return self._download_out(data.n_rows * self.n_outputs)

    def response_scores(
        mut self, data: BinnedMatrix, rng: IterationRange, response: Int
    ) raises -> List[Float64]:
        """Response-scale predictions: `raw_scores` with the objective's
        inverse link applied on the device.

        `response` is one of the RESPONSE_* codes;
        `response_for_objective` maps a built-in objective to the code
        `Booster.response` would apply. RESPONSE_SOFTMAX over an
        `n_outputs`-wide row is `MulticlassBooster.predict_proba_bins_range`,
        so this is also the probability path."""
        self._check_ready()
        if response < RESPONSE_IDENTITY or response > RESPONSE_SOFTMAX:
            raise Error("unknown response code")
        if response == RESPONSE_SOFTMAX and self.n_outputs < 2:
            raise Error("softmax needs at least two outputs")
        self.upload_bins(data)
        self._enqueue_scores(
            False,
            data.n_rows,
            rng,
            rng.includes_base(),
            False,
        )
        self.ctx.enqueue_function[_response_kernel](
            self.out_dev.unsafe_ptr(),
            self.resp_dev.unsafe_ptr(),
            Int32(data.n_rows),
            Int32(self.n_outputs),
            Int32(response),
            grid_dim=self._row_blocks(data.n_rows),
            block_dim=self.block_threads,
        )
        return self._download_out(
            data.n_rows * self.n_outputs, from_response=True
        )

    def leaf_indices(
        mut self, data: BinnedMatrix, rng: IterationRange
    ) raises -> List[Int]:
        """Per-tree leaf ordinals over `rng`, row-major within a row and
        round-major within an iteration:
        `[r * n_iters * n_outputs + i * n_outputs + k]`. This is the batched
        form of `Booster.leaf_indices_bins` and its multiclass counterpart,
        with the same ordinal numbering (`Tree.leaf_ordinals`).

        The output buffers are allocated per call rather than kept resident:
        their size depends on the iteration range, and leaf prediction is not
        the per-round hot path the score buffers are."""
        self._check_ready()
        self.upload_bins(data)
        var n_iters = rng.n_iterations()
        var n = data.n_rows * n_iters * self.n_outputs
        if n == 0:
            return List[Int]()
        var leaf_dev = self.ctx.enqueue_create_buffer[DType.int32](n)
        var leaf_host = self.ctx.enqueue_create_host_buffer[DType.int32](n)
        self.ctx.enqueue_function[_leaf_kernel](
            self.bins_dev.unsafe_ptr(),
            self.nodes_dev.unsafe_ptr(),
            self.cat_dev.unsafe_ptr(),
            self.root_dev.unsafe_ptr(),
            leaf_dev.unsafe_ptr(),
            Int32(data.n_rows),
            Int32(self.n_outputs),
            Int32(rng.start),
            Int32(rng.stop),
            grid_dim=(self._row_blocks(data.n_rows), self.n_outputs),
            block_dim=self.block_threads,
        )
        self.ctx.enqueue_copy(
            dst_ptr=leaf_host.unsafe_ptr(), src_buf=leaf_dev
        )
        self.ctx.synchronize()
        var out = List[Int](capacity=n)
        var src = leaf_host.unsafe_ptr()
        for i in range(n):
            out.append(Int(src.unsafe_load(i)))
        return out^

    def set_validation(
        mut self,
        data: BinnedMatrix,
        target: List[Float64],
        weight: List[Float64] = [],
    ) raises:
        """Make a validation set device-resident: its bins, its labels, its
        weights, and the running raw-score vector a training loop folds each
        round's trees into.

        `target` is the label per row: the regression target, the {0, 1}
        binary label, or the class index for a multiclass ensemble (labels
        are carried as Float32, which is exact for every class index and for
        any target a Float32 can hold). `weight` follows the metrics.mojo
        contract: empty means unweighted, and a non-empty vector must be
        finite, nonnegative, and have a positive sum, which is validated and
        summed here in Float64 so the metric's divisor keeps full precision.

        The raw scores start at zero; call `reset_validation` with the
        ensemble's base scores before the first round.
        """
        self._check_matrix(data)
        if len(target) != data.n_rows:
            raise Error("target length must equal the validation row count")
        self.valid_total_weight = check_metric_weight(weight, data.n_rows)
        self.valid_rows = data.n_rows
        self.valid_weighted = len(weight) > 0
        self.valid_blocks = (data.n_rows + REDUCE_BLOCK - 1) // REDUCE_BLOCK

        var labels = _f32_list(target)
        var weights = _f32_list(weight)
        var weight_len = len(weights)
        if weight_len < 1:
            weight_len = 1

        self.valid_bins_dev = self.ctx.enqueue_create_buffer[DType.uint8](
            data.n_rows * data.n_features
        )
        self.valid_label_dev = self.ctx.enqueue_create_buffer[DType.float32](
            data.n_rows
        )
        self.valid_weight_dev = self.ctx.enqueue_create_buffer[DType.float32](
            weight_len
        )
        self.valid_raw_dev = self.ctx.enqueue_create_buffer[DType.float32](
            data.n_rows * self.n_outputs
        )
        self.valid_resp_dev = self.ctx.enqueue_create_buffer[DType.float32](
            data.n_rows * self.n_outputs
        )
        self.part_dev = self.ctx.enqueue_create_buffer[DType.float32](
            self.valid_blocks
        )
        self.host_part = self.ctx.enqueue_create_host_buffer[DType.float32](
            self.valid_blocks
        )

        self.ctx.enqueue_copy(
            dst_buf=self.valid_bins_dev, src_ptr=data.bins.unsafe_ptr()
        )
        self.ctx.enqueue_copy(
            dst_buf=self.valid_label_dev, src_ptr=labels.unsafe_ptr()
        )
        if self.valid_weighted:
            self.ctx.enqueue_copy(
                dst_buf=self.valid_weight_dev, src_ptr=weights.unsafe_ptr()
            )
        self.ctx.enqueue_memset(self.valid_raw_dev, Float32(0.0))
        self.ctx.synchronize()

    def reset_validation(mut self, base_scores: List[Float64]) raises:
        """Set every validation row's raw score to the ensemble's per-output
        base score, which is where a boosting run starts. Call once before
        the first round; `accumulate_round` takes it from there."""
        if self.valid_rows == 0:
            raise Error("call set_validation before reset_validation")
        if len(base_scores) != self.n_outputs:
            raise Error("base_scores must have one entry per output")
        var host = self.ctx.enqueue_create_host_buffer[DType.float32](
            self.valid_rows * self.n_outputs
        )
        self.ctx.synchronize()
        var dst = host.unsafe_ptr()
        for r in range(self.valid_rows):
            for k in range(self.n_outputs):
                dst.unsafe_store(
                    r * self.n_outputs + k, Float32(base_scores[k])
                )
        self.ctx.enqueue_copy(dst_buf=self.valid_raw_dev, src_ptr=dst)
        self.ctx.synchronize()

    def accumulate_round(mut self, iteration: Int = 0) raises:
        """Add one iteration of the uploaded ensemble into the resident
        validation raw scores.

        A training loop uploads the round's new trees (one per output) and
        calls this with `iteration=0`, so a round costs one walk of one tree
        per row rather than a walk of the whole ensemble. The base score is
        never added here: `reset_validation` put it in once, which is the
        same place `IterationRange` puts it."""
        self._check_ready()
        if self.valid_rows == 0:
            raise Error("call set_validation before accumulate_round")
        if iteration < 0 or iteration >= self.n_iterations():
            raise Error("iteration out of range for the uploaded ensemble")
        self._enqueue_scores(
            True,
            self.valid_rows,
            IterationRange(iteration, iteration + 1),
            False,
            True,
        )

    def score_validation(mut self, rng: IterationRange) raises:
        """Rewrite the resident validation raw scores from scratch over
        `rng` of the uploaded ensemble, base score included when the range
        starts at 0. The whole-ensemble counterpart of `accumulate_round`,
        for scoring a finished model rather than following a run."""
        self._check_ready()
        if self.valid_rows == 0:
            raise Error("call set_validation before score_validation")
        self._enqueue_scores(
            True,
            self.valid_rows,
            rng,
            rng.includes_base(),
            False,
        )

    def validation_raw(mut self) raises -> List[Float64]:
        """The resident raw scores, row-major `[r * n_outputs + k]`. The
        escape hatch for scoring a metric the device does not implement: the
        host metric suite in metrics.mojo takes exactly this vector."""
        if self.valid_rows == 0:
            raise Error("call set_validation before validation_raw")
        var n = self.valid_rows * self.n_outputs
        var host = self.ctx.enqueue_create_host_buffer[DType.float32](n)
        self.ctx.enqueue_copy(
            dst_ptr=host.unsafe_ptr(), src_buf=self.valid_raw_dev
        )
        self.ctx.synchronize()
        var out = List[Float64](capacity=n)
        var src = host.unsafe_ptr()
        for i in range(n):
            out.append(Float64(src.unsafe_load(i)))
        return out^

    def _reduce_metric(mut self, metric: Int) raises -> Float64:
        """Run the reduction kernel and sum its partials in Float64, in
        ascending block order."""
        self.ctx.enqueue_function[_metric_kernel](
            self.valid_resp_dev.unsafe_ptr(),
            self.valid_label_dev.unsafe_ptr(),
            self.valid_weight_dev.unsafe_ptr(),
            self.part_dev.unsafe_ptr(),
            Int32(self.valid_rows),
            Int32(self.n_outputs),
            Int32(metric),
            Int32(1) if self.valid_weighted else Int32(0),
            grid_dim=self.valid_blocks,
            block_dim=REDUCE_BLOCK,
        )
        self.ctx.enqueue_copy(
            dst_ptr=self.host_part.unsafe_ptr(), src_buf=self.part_dev
        )
        self.ctx.synchronize()
        var total = 0.0
        var src = self.host_part.unsafe_ptr()
        for b in range(self.valid_blocks):
            total += Float64(src.unsafe_load(b))
        return total

    def validation_metric(
        mut self, metric: Int, response: Int
    ) raises -> Float64:
        """Score the resident validation set on the device.

        `response` is the inverse link to apply to the resident raw scores
        first, because every metric in metrics.mojo takes predictions on the
        response scale. The transform writes to a separate buffer, so the raw
        scores stay intact for the next round's `accumulate_round`.

        `metric` is one of the METRIC_* codes. The two error rates are
        reported the way metrics.mojo defines them, as one minus the
        corresponding accuracy.
        """
        if self.valid_rows == 0:
            raise Error("call set_validation before validation_metric")
        if metric < METRIC_L2 or metric > METRIC_MULTICLASS_ACCURACY:
            raise Error("unknown metric code")
        if response < RESPONSE_IDENTITY or response > RESPONSE_SOFTMAX:
            raise Error("unknown response code")
        if (
            metric == METRIC_MULTICLASS_LOG_LOSS
            or metric == METRIC_MULTICLASS_ACCURACY
        ) and self.n_outputs < 2:
            raise Error("multiclass metrics need at least two outputs")

        self.ctx.enqueue_function[_response_kernel](
            self.valid_raw_dev.unsafe_ptr(),
            self.valid_resp_dev.unsafe_ptr(),
            Int32(self.valid_rows),
            Int32(self.n_outputs),
            Int32(response),
            grid_dim=self._row_blocks(self.valid_rows),
            block_dim=self.block_threads,
        )
        var total = self._reduce_metric(metric)
        return total / self.valid_total_weight

    def validation_error(
        mut self, metric: Int, response: Int
    ) raises -> Float64:
        """One minus an accuracy metric, LightGBM's `binary_error` and
        `multi_error`. `metric` must be METRIC_BINARY_ACCURACY or
        METRIC_MULTICLASS_ACCURACY."""
        if (
            metric != METRIC_BINARY_ACCURACY
            and metric != METRIC_MULTICLASS_ACCURACY
        ):
            raise Error("validation_error takes an accuracy metric")
        return 1.0 - self.validation_metric(metric, response)


def predict_gpu(
    booster: Booster, data: BinnedMatrix, rng: IterationRange
) raises -> List[Float64]:
    """One-shot response-scale prediction for a single-output ensemble
    (builds a predictor and uploads everything each call; use `GpuPredictor`
    for repeated prediction)."""
    var predictor = GpuPredictor(data.n_features, 1)
    predictor.upload_ensemble(flatten_booster(booster))
    return predictor.response_scores(
        data, rng, response_for_objective(booster.objective)
    )


def predict_raw_gpu(
    booster: Booster, data: BinnedMatrix, rng: IterationRange
) raises -> List[Float64]:
    """One-shot raw-score prediction for a single-output ensemble."""
    var predictor = GpuPredictor(data.n_features, 1)
    predictor.upload_ensemble(flatten_booster(booster))
    return predictor.raw_scores(data, rng)


def predict_proba_gpu(
    booster: MulticlassBooster, data: BinnedMatrix, rng: IterationRange
) raises -> List[Float64]:
    """One-shot class probabilities for a softmax ensemble, row-major
    `[r * n_classes + k]`."""
    var predictor = GpuPredictor(data.n_features, booster.n_classes)
    predictor.upload_ensemble(flatten_multiclass(booster))
    return predictor.response_scores(data, rng, RESPONSE_SOFTMAX)


def predict_raw_multiclass_gpu(
    booster: MulticlassBooster, data: BinnedMatrix, rng: IterationRange
) raises -> List[Float64]:
    """One-shot per-class raw scores for a softmax ensemble, row-major
    `[r * n_classes + k]`, the scores `predict_proba_gpu` softmaxes."""
    var predictor = GpuPredictor(data.n_features, booster.n_classes)
    predictor.upload_ensemble(flatten_multiclass(booster))
    return predictor.raw_scores(data, rng)
