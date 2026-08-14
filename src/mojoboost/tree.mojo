"""Leaf-wise (best-first) tree growth.

Grows a single regression tree the way LightGBM does. At each step the leaf
with the highest split gain anywhere in the tree is split, until num_leaves
is reached or no leaf has a positive-gain split. For each split, the smaller
child's histogram is built directly and the larger child's is derived by
subtraction from the parent's.

Leaf values use the second-order Newton step with LightGBM's L1 shrinkage:
-T(G) / (H + lambda_l2), where T soft-thresholds the gradient sum by
lambda_l1 (see split.mojo).

`params.max_depth` bounds how deep a leaf may sit without disturbing that
order: growth stays leaf-wise and trees stay unbalanced, a leaf that has
reached the limit simply offers no split and so is never selected. Depth is
counted in edges from the root, so the root is depth 0 and max_depth=1
yields a stump; values <= 0 mean unlimited, as in LightGBM.

Growth also carries each frontier leaf's branch feature set, the features
split on between the root and that leaf. When feature interaction
constraints are configured, the branch set determines which features the
leaf may split on; see interaction.mojo for the rule. With no constraints
the branch sets stay empty and the allow masks stay empty, so the
unconstrained path is unchanged.

Feature subsampling (see sampling.mojo) draws one feature set per tree from
`tree_index` and the seed, and optionally a further set per node. Only the
tree's set is ever accumulated into histograms, so excluded features cost
nothing and sibling subtraction stays exact; the per-node set narrows the
split search on top of that.

Under monotonic constraints (see monotone.mojo) each frontier leaf also
carries the interval its output must lie in. Leaf values are clamped into it,
candidate splits are scored from clamped outputs and rejected when they run
against their feature's constraint, and a split on a constrained feature
divides the parent's interval between its children at the midpoint of their
values. The intervals are not stored on the tree: `node_bounds` recovers them
from a grown tree, because an internal node keeps the value it had when it was
created.
"""

from .binning import BinnedMatrix
from .categorical import (
    CAT_BITSET_WORDS,
    CategoricalParams,
    CategoricalSpec,
    cat_pool_contains,
)
from .histogram import (
    Histogram,
    build_histogram,
    build_histogram_into,
    build_histogram_subset,
    build_histogram_subset_into,
    subtract_histogram,
    subtract_histogram_into,
)
from .parallel import plan_row_blocks, run_row_blocks
from .interaction import InteractionConstraints, extend_branch
from .monotone import (
    MONOTONE_FREE,
    MonotoneConstraints,
    OutputBounds,
    child_bounds,
    midpoint,
    monotone_sign,
)
from .sampling import (
    DEFAULT_FEATURE_FRACTION_SEED,
    select_node_features,
    select_tree_features,
)
from .split import SplitInfo, find_best_split, soft_threshold_l1


struct TreeParams(Copyable, Movable):
    """Tree growth hyperparameters. `lambda_reg` is LightGBM's lambda_l2 and
    `lambda_l1` its lambda_l1; both default to LightGBM's own defaults except
    lambda_reg, which mojoboost defaults to 1.0 (see README). `constraints`
    holds LightGBM's interaction_constraints and defaults to unconstrained.
    `feature_fraction`, `feature_fraction_bynode`, and
    `feature_fraction_seed` are LightGBM's feature subsampling parameters
    (see sampling.mojo); both fractions default to 1.0, which selects every
    feature and leaves the seed with no effect. `max_depth` is LightGBM's
    max_depth: an upper bound on a leaf's depth in edges from the root, with
    values <= 0 meaning unlimited (LightGBM's default is -1). `monotone` holds
    LightGBM's monotone_constraints (see monotone.mojo) and defaults to
    unconstrained. `cat` holds LightGBM's categorical hyperparameters (see
    categorical.mojo) and defaults to LightGBM's own defaults; which features
    are categorical is a property of the binned matrix, not of these
    parameters."""

    var num_leaves: Int
    var min_data_in_leaf: Int
    var lambda_reg: Float64
    var min_child_hess: Float64
    var lambda_l1: Float64
    var constraints: InteractionConstraints
    var feature_fraction: Float64
    var feature_fraction_bynode: Float64
    var feature_fraction_seed: Int
    var max_depth: Int
    var monotone: MonotoneConstraints
    var cat: CategoricalParams

    def __init__(
        out self,
        num_leaves: Int,
        min_data_in_leaf: Int,
        lambda_reg: Float64,
        min_child_hess: Float64,
        lambda_l1: Float64 = 0.0,
        var constraints: InteractionConstraints = InteractionConstraints(),
        feature_fraction: Float64 = 1.0,
        feature_fraction_bynode: Float64 = 1.0,
        feature_fraction_seed: Int = DEFAULT_FEATURE_FRACTION_SEED,
        max_depth: Int = -1,
        var monotone: MonotoneConstraints = MonotoneConstraints(),
        var cat: CategoricalParams = CategoricalParams.default(),
    ):
        self.num_leaves = num_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.lambda_reg = lambda_reg
        self.min_child_hess = min_child_hess
        self.lambda_l1 = lambda_l1
        self.constraints = constraints^
        self.feature_fraction = feature_fraction
        self.feature_fraction_bynode = feature_fraction_bynode
        self.feature_fraction_seed = feature_fraction_seed
        self.max_depth = max_depth
        self.monotone = monotone^
        self.cat = cat^

    @staticmethod
    def default() -> TreeParams:
        # LightGBM defaults (min_child_hess mirrors min_sum_hessian_in_leaf,
        # lambda_l1 defaults to 0, interaction and monotonic constraints to
        # none, both feature fractions to 1.0, and max_depth to -1, as in
        # LightGBM).
        return TreeParams(31, 20, 1.0, 1e-3, 0.0)


struct Tree(Copyable, Movable):
    """Flat-array tree. Node i is internal when feature[i] >= 0; then rows
    with bin(feature[i]) <= threshold_bin[i] go to left[i], the rest to
    right[i]. Node i is a leaf when feature[i] < 0; its output is value[i].

    Missing values are routed by node, not by threshold: a row whose bin
    equals `missing_bin[i]` goes left when `default_left[i]` and right
    otherwise, whatever the threshold says. `missing_bin[i]` is the missing
    bin of node i's split feature, or -1 when that feature reserves none, in
    which case no bin id can match and the threshold decides every row. The
    two arrays are always as long as `feature`, so a tree is self-contained:
    prediction needs no bin mapper to route missing values.

    A node that splits a categorical feature routes by category set instead
    of by threshold: `cat_offset[i] >= 0` is the offset of node i's 256-bit
    set in the flat `cat_bitset` pool, and a row goes left exactly when its
    bin is in that set. Bin 0 of a categorical feature (missing, unseen, or
    dropped) is never in a set, so those rows always go right. `cat_offset`
    is -1 on every numerical node, which is what the whole array holds for a
    model with no categorical features.

    `count[i]` is node i's cover: how many of the rows this tree was grown on
    reach node i. It is the background weighting exact TreeSHAP conditions on
    (see contrib.mojo), so it is kept with the model and serialized rather
    than recomputed. Under bagging or GOSS it counts the sampled rows, which
    are the rows the node's value was fitted from. A tree built without
    counts (a v1 or v2 file, which predate them) carries zeros, which
    `has_node_counts` reports and the contribution API refuses to work
    from."""

    var feature: List[Int]
    var threshold_bin: List[Int]
    var left: List[Int]
    var right: List[Int]
    var value: List[Float64]
    var split_gain: List[Float64]
    var n_leaves: Int
    var default_left: List[Bool]
    var missing_bin: List[Int]
    var cat_offset: List[Int]
    var cat_bitset: List[UInt64]
    var count: List[Float64]

    def __init__(
        out self,
        var feature: List[Int],
        var threshold_bin: List[Int],
        var left: List[Int],
        var right: List[Int],
        var value: List[Float64],
        var split_gain: List[Float64],
        n_leaves: Int,
        var default_left: List[Bool] = [],
        var missing_bin: List[Int] = [],
        var cat_offset: List[Int] = [],
        var cat_bitset: List[UInt64] = [],
        var count: List[Float64] = [],
    ):
        """Omitting the missing-routing arrays (or passing ones of the wrong
        length) builds a tree that routes no missing values, which is what a
        model trained without missing support, or loaded from a v1 file,
        needs. Omitting `cat_offset` likewise builds a tree with no
        categorical nodes, and omitting `count` builds a tree with no node
        covers, which is what a v1 or v2 file describes."""
        var n_nodes = len(feature)
        self.feature = feature^
        self.threshold_bin = threshold_bin^
        self.left = left^
        self.right = right^
        self.value = value^
        self.split_gain = split_gain^
        self.n_leaves = n_leaves
        if len(default_left) == n_nodes:
            self.default_left = default_left^
        else:
            self.default_left = List[Bool](capacity=n_nodes)
            self.default_left.resize(n_nodes, False)
        if len(missing_bin) == n_nodes:
            self.missing_bin = missing_bin^
        else:
            self.missing_bin = List[Int](capacity=n_nodes)
            self.missing_bin.resize(n_nodes, -1)
        if len(cat_offset) == n_nodes:
            self.cat_offset = cat_offset^
        else:
            self.cat_offset = List[Int](capacity=n_nodes)
            self.cat_offset.resize(n_nodes, -1)
        self.cat_bitset = cat_bitset^
        if len(count) == n_nodes:
            self.count = count^
        else:
            self.count = List[Float64](capacity=n_nodes)
            self.count.resize(n_nodes, 0.0)

    def has_node_counts(self) -> Bool:
        """Whether this tree carries node covers (see `count`). False for a
        tree loaded from a v1 or v2 file, which predate them, and for one
        built without them; every grower records them. A grown tree always
        has a positive root cover, so the root alone settles it."""
        return len(self.count) == len(self.feature) and self.count[0] > 0.0

    def check_node_counts(self) raises:
        """Raise unless every node carries a positive cover. Exact
        contribution attribution (contrib.mojo) divides by a node's cover at
        every internal node, so it checks this once per model rather than
        guarding each division."""
        # A tree with no covers at all is the common case worth naming: it
        # came from a v1 or v2 file. Checking it first keeps that diagnosis
        # from being reported as a stray zero at node 0.
        if not self.has_node_counts():
            raise Error(
                "tree carries no node counts: it was loaded from a model file"
                " written before they were recorded (v1 or v2). Retrain, or"
                " re-save the model from a current build, to use feature"
                " contributions."
            )
        for i in range(len(self.count)):
            if not self.count[i] > 0.0:
                raise Error(
                    "tree node ",
                    i,
                    " has a nonpositive cover (",
                    self.count[i],
                    "); feature contributions need every node's training row"
                    " count",
                )

    def _add_node(mut self, value: Float64, count: Float64) -> Int:
        """Append a leaf holding `value`, covered by `count` training rows.
        Every grower goes through here, so node covers cannot be recorded on
        one backend and missed on another."""
        var node = len(self.feature)
        self.feature.append(-1)
        self.threshold_bin.append(-1)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(value)
        # Recorded when the node is split; stays 0.0 for leaves (and for
        # every node of a model loaded from disk).
        self.split_gain.append(0.0)
        # Set when the node is split; a leaf routes nothing.
        self.default_left.append(False)
        self.missing_bin.append(-1)
        self.cat_offset.append(-1)
        self.count.append(count)
        return node

    def _set_split(mut self, node: Int, split: SplitInfo, missing_bin: Int):
        """Record a chosen split on `node`, numerical or categorical. This is
        the only place either grower writes split routing, so the CPU and GPU
        trees carry identical node layouts."""
        self.feature[node] = split.feature
        self.split_gain[node] = split.gain
        if split.is_categorical:
            # A categorical node routes only by its set: no threshold, and no
            # missing bin, since bin 0 already collects the missing rows and
            # is never a set member.
            self.threshold_bin[node] = -1
            self.default_left[node] = False
            self.missing_bin[node] = -1
            self.cat_offset[node] = len(self.cat_bitset)
            for w in range(CAT_BITSET_WORDS):
                self.cat_bitset.append(split.cat_bitset[w])
        else:
            self.threshold_bin[node] = split.bin
            self.default_left[node] = split.default_left
            self.missing_bin[node] = missing_bin

    @always_inline
    def goes_left(self, node: Int, bin: Int) -> Bool:
        """Whether a row in `bin` of node `node`'s split feature goes to the
        left child. A categorical node routes by set membership. Otherwise the
        missing bin follows the node's default direction; -1 never matches a
        real bin, so unaffected nodes fall through to the ordinary threshold
        test."""
        var off = self.cat_offset[node]
        if off >= 0:
            return cat_pool_contains(self.cat_bitset, off, bin)
        if bin == self.missing_bin[node]:
            return self.default_left[node]
        return bin <= self.threshold_bin[node]

    def predict_row(self, data: BinnedMatrix, row: Int) -> Float64:
        var node = 0
        while self.feature[node] >= 0:
            if self.goes_left(node, data.bin_at(row, self.feature[node])):
                node = self.left[node]
            else:
                node = self.right[node]
        return self.value[node]

    def predict_bins(self, bins: List[Int]) -> Float64:
        """Predict one example given its per-feature bin ids."""
        var node = 0
        while self.feature[node] >= 0:
            if self.goes_left(node, bins[self.feature[node]]):
                node = self.left[node]
            else:
                node = self.right[node]
        return self.value[node]

    def depth(self) -> Int:
        """The depth of the deepest leaf, in edges from the root: 0 for a
        single-leaf tree and 1 for a stump. This is the quantity
        `TreeParams.max_depth` bounds."""
        if len(self.feature) == 0:
            return 0
        var best = 0
        var nodes: List[Int] = [0]
        var depths: List[Int] = [0]
        while len(nodes) > 0:
            var node = nodes.pop()
            var d = depths.pop()
            if self.feature[node] < 0:
                if d > best:
                    best = d
                continue
            nodes.append(self.left[node])
            depths.append(d + 1)
            nodes.append(self.right[node])
            depths.append(d + 1)
        return best

    def leaf_index_row(self, data: BinnedMatrix, row: Int) -> Int:
        """The node index of the leaf this row lands in."""
        var node = 0
        while self.feature[node] >= 0:
            if self.goes_left(node, data.bin_at(row, self.feature[node])):
                node = self.left[node]
            else:
                node = self.right[node]
        return node

    def leaf_index_bins(self, bins: List[Int]) -> Int:
        """The node index of the leaf `bins` lands in, given one example's
        per-feature bin ids. The bins counterpart of `leaf_index_row`."""
        var node = 0
        while self.feature[node] >= 0:
            if self.goes_left(node, bins[self.feature[node]]):
                node = self.left[node]
            else:
                node = self.right[node]
        return node

    def leaf_ordinals(self) -> List[Int]:
        """Per-node leaf ordinal: a leaf's rank among this tree's leaves in
        node order, and -1 for every internal node.

        Node ids are an implementation detail (they number internal nodes and
        leaves together, and shift as a tree grows), so leaf prediction
        reports this ordinal instead. It lies in [0, n_leaves), it is fixed
        once the tree is grown, and serialization writes nodes in array order,
        so a saved and reloaded tree assigns exactly the same ordinals. It is
        mojoboost's own numbering: it is not LightGBM's leaf id, and the two
        agree only by coincidence.

        Callers predicting many rows should build this table once per tree and
        index it with `leaf_index_bins`, which is what `leaf_ordinal_bins`
        does for a single example."""
        var out = List[Int](capacity=len(self.feature))
        var next_ordinal = 0
        for i in range(len(self.feature)):
            if self.feature[i] < 0:
                out.append(next_ordinal)
                next_ordinal += 1
            else:
                out.append(-1)
        return out^

    def leaf_ordinal_bins(self, bins: List[Int]) -> Int:
        """The leaf ordinal (see `leaf_ordinals`) this example lands in."""
        var node = self.leaf_index_bins(bins)
        var ordinal = 0
        for i in range(node):
            if self.feature[i] < 0:
                ordinal += 1
        return ordinal


struct _LeafState(Movable):
    """A grown-but-unsplit leaf: its node id, rows, histogram, the best split
    available from it, the features split on between the root and it (empty
    when no interaction constraints are configured), its depth in edges from
    the root, and the interval its output must lie in (unbounded when no
    monotonic constraint above it applies)."""

    var node: Int
    var rows: List[Int]
    var hist: Histogram
    var split: SplitInfo
    var branch: List[Int]
    var depth: Int
    var bounds: OutputBounds

    def __init__(
        out self,
        node: Int,
        var rows: List[Int],
        var hist: Histogram,
        var split: SplitInfo,
        var branch: List[Int] = [],
        depth: Int = 0,
        var bounds: OutputBounds = OutputBounds.unbounded(),
    ):
        self.node = node
        self.rows = rows^
        self.hist = hist^
        self.split = split^
        self.branch = branch^
        self.depth = depth
        self.bounds = bounds^

    def take_hist(mut self) raises -> Histogram:
        """Move this leaf's histogram out, leaving an empty one behind.

        Used when a leaf is split: its histogram is dead the moment sibling
        subtraction has read it, so the buffer goes back to the pool instead
        of being freed with the state it hung off."""
        var out = Histogram.zeroed(0, 0)
        swap(out, self.hist)
        return out^


@fieldwise_init
struct RowPartition(Movable):
    """One node's rows split by a chosen split, each side in ascending row
    order."""

    var left: List[Int]
    var right: List[Int]


def partition_rows(
    data: BinnedMatrix,
    rows: List[Int],
    split: SplitInfo,
    missing_bin: Int,
) raises -> RowPartition:
    """`partition_rows_into` returning freshly allocated sides."""
    var left = List[Int]()
    var right = List[Int]()
    partition_rows_into(left, right, data, rows, split, missing_bin)
    return RowPartition(left^, right^)


def partition_rows_into(
    mut left: List[Int],
    mut right: List[Int],
    data: BinnedMatrix,
    rows: List[Int],
    split: SplitInfo,
    missing_bin: Int,
) raises:
    """Route each of `rows` to the left or right child of `split`.

    A categorical split routes by set membership; otherwise rows in the
    feature's missing bin follow the split's default direction instead of the
    threshold, and a feature with no missing bin (-1) has none of them.

    Two passes: count per block, prefix-sum the counts, then scatter. That
    costs one extra read of the split feature's bins and buys two things a
    single appending pass cannot have. The output lists are allocated once at
    their exact final size instead of doubling, and the passes parallelize
    across row blocks. Block b's rows land at `prefix_left[b]` on the left and
    at `start(b) - prefix_left[b]` on the right, so both sides come out in
    ascending row order whatever the block count: the result is identical to
    the serial single-pass partition, index for index.
    """
    var n = len(rows)
    # Each row is touched twice, and each touch is an indirect load of a bin
    # through a row id, so a row costs roughly three histogram ops rather than
    # one; the estimate is scaled up to match.
    var blocks = plan_row_blocks(n, 3 * n)
    var rows_p = rows.unsafe_ptr()
    var bins_p = data.bins.unsafe_ptr().unsafe_offset(
        split.feature * data.n_rows
    )
    var is_cat = split.is_categorical
    var default_left = split.default_left
    var threshold = split.bin

    @always_inline
    def goes_left(bin: Int) {imm} -> Bool:
        if is_cat:
            return split.goes_left(bin)
        if bin == missing_bin:
            return default_left
        return bin <= threshold

    var left_counts = List[Int](capacity=blocks.n_blocks)
    left_counts.resize(blocks.n_blocks, 0)
    var counts_p = left_counts.unsafe_ptr()

    def count_block(b: Int) {imm}:
        var c = 0
        for i in range(blocks.start(b), blocks.end(b)):
            if goes_left(Int(bins_p.unsafe_load(rows_p.unsafe_load(i)))):
                c += 1
        counts_p.unsafe_store(b, c)

    run_row_blocks(blocks, count_block)

    # Exclusive prefix sum over the per-block left counts, in place.
    var total_left = 0
    for b in range(blocks.n_blocks):
        var c = left_counts[b]
        left_counts[b] = total_left
        total_left += c

    # Sized exactly, so a caller that reuses its buffers across splits keeps
    # whatever capacity it already grew and never reallocates on a smaller
    # node.
    left.resize(total_left, 0)
    right.resize(n - total_left, 0)
    var left_p = left.unsafe_ptr()
    var right_p = right.unsafe_ptr()

    def scatter_block(b: Int) {imm}:
        var start = blocks.start(b)
        var li = counts_p.unsafe_load(b)
        var ri = start - li
        for i in range(start, blocks.end(b)):
            var r = rows_p.unsafe_load(i)
            if goes_left(Int(bins_p.unsafe_load(r))):
                left_p.unsafe_store(li, r)
                li += 1
            else:
                right_p.unsafe_store(ri, r)
                ri += 1

    run_row_blocks(blocks, scatter_block)


struct _HistPool(Movable):
    """Free-list of histogram buffers of one shape.

    Tree growth builds two child histograms per split and drops the parent's,
    so at most `num_leaves + 1` are ever live at once and the buffers can be
    recycled instead of reallocated. Each buffer is three arrays of
    `n_features * n_bins`, which at the default 100 features and 255 bins is
    around 600 KB per node: large enough that the allocator hands back fresh
    pages and faults them in every time. Recycling keeps that cost to the
    first few nodes of the first tree.
    """

    var free: List[Histogram]
    var n_features: Int
    var n_bins: Int

    def __init__(out self, n_features: Int, n_bins: Int):
        self.free = List[Histogram]()
        self.n_features = n_features
        self.n_bins = n_bins

    def take(mut self) raises -> Histogram:
        """A buffer of the pool's shape. Contents are undefined; every
        `_into` builder writes or zeroes before reading."""
        if len(self.free) > 0:
            return self.free.pop()
        return Histogram.zeroed(self.n_features, self.n_bins)

    def give(mut self, var hist: Histogram):
        """Return a buffer. Buffers of another shape are dropped, so a
        mismatched hand-back can never corrupt a later `take`."""
        if hist.matches(self.n_features, self.n_bins):
            self.free.append(hist^)


def _leaf_value(
    hist: Histogram,
    lambda_reg: Float64,
    lambda_l1: Float64 = 0.0,
    feature: Int = 0,
) -> Float64:
    # Totals over one feature's bins: every feature the histogram accumulated
    # has the same bin totals. `feature` must be one of them, which under
    # feature subsampling means a selected feature, since the excluded
    # features' slices are left at zero.
    var base = feature * hist.n_bins
    var g = 0.0
    var h = 0.0
    for b in range(hist.n_bins):
        g += hist.grad[base + b]
        h += hist.hess[base + b]
    return -soft_threshold_l1(g, lambda_l1) / (h + lambda_reg)


def _search(
    hist: Histogram,
    n_rows: Int,
    params: TreeParams,
    allowed: List[Bool] = [],
    features: List[Int] = [],
    depth: Int = 0,
    missing_bins: List[Int] = [],
    monotone: List[Int] = [],
    bounds: OutputBounds = OutputBounds.unbounded(),
    cats: CategoricalSpec = CategoricalSpec.none(),
) raises -> SplitInfo:
    """Best split for one node. `allowed` is the node's interaction-constraint
    allow mask and `features` its subsampled feature ids; empty means every
    feature is a candidate. `depth` is the node's depth in edges from the
    root, checked against `params.max_depth`; both growers pass it
    explicitly, and the 0 default means an unbounded node. `missing_bins` is
    the dataset's per-feature missing-bin table. `monotone` is the active
    monotonic constraint vector (empty when unconstrained) and `bounds` this
    node's output interval. `cats` marks the categorical features, whose
    candidates are category partitions rather than thresholds. Both the CPU
    and the GPU grower go through here, so the two enforce constraints,
    subsampling, the depth limit, missing-value routing, and categorical
    partitioning identically."""
    # A leaf at the depth limit yields no split, which is what stops growth
    # beneath it; leaf-wise selection is otherwise untouched.
    if params.max_depth > 0 and depth >= params.max_depth:
        return SplitInfo(-1, -1, 0.0, False)
    if n_rows < 2 * params.min_data_in_leaf or n_rows < 2:
        return SplitInfo(-1, -1, 0.0, False)
    return find_best_split(
        hist,
        lambda_reg=params.lambda_reg,
        min_child_hess=params.min_child_hess,
        min_data_in_leaf=params.min_data_in_leaf,
        lambda_l1=params.lambda_l1,
        allowed=allowed,
        features=features,
        missing_bins=missing_bins,
        monotone=monotone,
        bounds=bounds,
        cats=cats,
        cat_params=params.cat,
    )


def grow_tree(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
) raises -> Tree:
    """Grow one tree, leaf-wise.

    A non-empty `bag` of row indices (see bagging.mojo) restricts growth to
    those rows: the root histogram covers the bag alone, so every count,
    gain, leaf value, and `min_data_in_leaf` decision beneath it is a bag
    quantity, and the tree is exactly the one this grower would produce on a
    dataset holding only those rows. An empty bag means the full dataset.

    `params.constraints`, when non-empty, restricts each node's split search
    to the features its branch still permits (see interaction.mojo).

    `params.monotone`, when it constrains a feature, makes the grown tree
    monotone in that feature (see monotone.mojo).

    `tree_index` is this tree's position in the ensemble; together with
    `params.feature_fraction_seed` it fixes which features the tree and its
    nodes may split on, so growing the same tree again selects the same
    features no matter what else has been trained.
    """
    params.constraints.check_features(data.n_features)
    params.monotone.check_features(data.n_features)
    # Empty unless a feature is actually constrained, which keeps split search
    # on its unconstrained path and the fit bit-identical.
    var signs = params.monotone.active_signs()
    var tree_features = select_tree_features(
        data.n_features,
        params.feature_fraction,
        params.feature_fraction_seed,
        tree_index,
    )
    # Leaf-value totals must come from a feature the histograms accumulated.
    var value_feature = tree_features[0]
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )

    # Every histogram this tree builds comes from one pool and goes back to it
    # when its leaf is split, so growth allocates a handful of buffers rather
    # than three arrays per node.
    var pool = _HistPool(data.n_features, data.n_bins)

    # The root's row list is the only thing bagging materializes; the full
    # path builds the same list over every row.
    var root_rows: List[Int]
    var root_hist = pool.take()
    if len(bag) == 0:
        root_rows = List[Int](capacity=data.n_rows)
        root_rows.resize(data.n_rows, 0)
        for r in range(data.n_rows):
            root_rows[r] = r
        build_histogram_into(root_hist, data, grad, hess, tree_features)
    else:
        for i in range(len(bag)):
            if bag[i] < 0 or bag[i] >= data.n_rows:
                raise Error("bag row index out of range")
        root_rows = bag.copy()
        build_histogram_subset_into(
            root_hist, data, grad, hess, bag, 0, len(bag), tree_features
        )

    # The root's branch is empty, so its allow mask is the union of every
    # configured group (empty, meaning all features, when unconstrained). The
    # root is always node 0, so its per-node feature draw is fixed too.
    var root_branch = List[Int]()
    var root_split = _search(
        root_hist,
        len(root_rows),
        params,
        params.constraints.allowed_features(root_branch),
        select_node_features(
            tree_features,
            params.feature_fraction_bynode,
            params.feature_fraction_seed,
            tree_index,
            0,
        ),
        depth=0,
        missing_bins=data.missing_bin,
        monotone=signs,
        cats=data.cats,
    )
    var root = tree._add_node(
        _leaf_value(
            root_hist, params.lambda_reg, params.lambda_l1, value_feature
        ),
        Float64(len(root_rows)),
    )

    var frontier = List[_LeafState]()
    frontier.append(
        _LeafState(
            root, root_rows^, root_hist^, root_split^, root_branch^, depth=0
        )
    )
    var n_leaves = 1

    while n_leaves < params.num_leaves:
        # Pick the leaf with the best gain anywhere in the tree.
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

        # Partition the parent's rows by the chosen split. A categorical
        # split routes by set membership; otherwise rows in the feature's
        # missing bin follow the split's default direction instead of the
        # threshold, and a feature with no missing bin has none of them.
        var split_missing_bin = -1 if split.is_categorical else (
            data.missing_bin[split.feature]
        )
        # Each child's rows are handed to its `_LeafState` below, so the two
        # lists cannot be recycled across splits; the `_into` form is used
        # anyway because it sizes them exactly in one shot instead of growing
        # them by doubling.
        var left_rows = List[Int]()
        var right_rows = List[Int]()
        partition_rows_into(
            left_rows,
            right_rows,
            data,
            frontier[best_i].rows,
            split,
            split_missing_bin,
        )

        # The parent's histogram is read once more, by the subtraction below,
        # and is dead after that; moving it out here is what lets its buffer
        # go straight back to the pool.
        var parent_hist = frontier[best_i].take_hist()

        # Histogram subtraction trick: build the smaller child directly.
        var left_hist = pool.take()
        var right_hist = pool.take()
        if len(left_rows) <= len(right_rows):
            build_histogram_subset_into(
                left_hist, data, grad, hess, left_rows, 0, len(left_rows),
                tree_features,
            )
            subtract_histogram_into(right_hist, parent_hist, left_hist)
        else:
            build_histogram_subset_into(
                right_hist, data, grad, hess, right_rows, 0, len(right_rows),
                tree_features,
            )
            subtract_histogram_into(left_hist, parent_hist, right_hist)
        pool.give(parent_hist^)

        # Child values are clamped into the parent's interval, then the
        # interval is divided between the children at their midpoint. Both are
        # no-ops when unconstrained: the interval is unbounded and the split
        # sign is 0, so the values and the children's intervals are the
        # parent's unchanged.
        var parent_bounds = frontier[best_i].bounds.copy()
        var split_sign = monotone_sign(signs, split.feature)
        var left_value = parent_bounds.clamp(
            _leaf_value(
                left_hist, params.lambda_reg, params.lambda_l1, value_feature
            )
        )
        var right_value = parent_bounds.clamp(
            _leaf_value(
                right_hist, params.lambda_reg, params.lambda_l1, value_feature
            )
        )
        if split_sign != MONOTONE_FREE and left_value > right_value:
            # Candidates were scored from the parent's prefix sums while these
            # values come from the child histograms, one of which was derived
            # by subtraction, so the two outputs can invert by a rounding step
            # after passing the candidate check. Collapsing both to their
            # midpoint keeps the ordering exact, and leaves the midpoint (and
            # so both children's intervals) unchanged.
            var mid = midpoint(left_value, right_value)
            left_value = mid
            right_value = mid
        var children = child_bounds(
            parent_bounds, split_sign, left_value, right_value
        )
        var left_node = tree._add_node(left_value, Float64(len(left_rows)))
        var right_node = tree._add_node(right_value, Float64(len(right_rows)))
        tree.left[parent_node] = left_node
        tree.right[parent_node] = right_node
        tree._set_split(parent_node, split, split_missing_bin)

        # Both children inherit the same branch feature set, so they share one
        # allow mask, and both sit one edge below the leaf that was split.
        var branch = extend_branch(frontier[best_i].branch, split.feature)
        var allowed = params.constraints.allowed_features(branch)
        var child_depth = frontier[best_i].depth + 1
        # Each child draws its own per-node feature set from its node id.
        var left_split = _search(
            left_hist,
            len(left_rows),
            params,
            allowed,
            select_node_features(
                tree_features,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                left_node,
            ),
            depth=child_depth,
            missing_bins=data.missing_bin,
            monotone=signs,
            cats=data.cats,
            bounds=children.left,
        )
        var right_split = _search(
            right_hist,
            len(right_rows),
            params,
            allowed,
            select_node_features(
                tree_features,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                right_node,
            ),
            depth=child_depth,
            missing_bins=data.missing_bin,
            monotone=signs,
            cats=data.cats,
            bounds=children.right,
        )

        frontier[best_i] = _LeafState(
            left_node,
            left_rows^,
            left_hist^,
            left_split^,
            branch.copy(),
            depth=child_depth,
            bounds=children.left.copy(),
        )
        frontier.append(
            _LeafState(
                right_node,
                right_rows^,
                right_hist^,
                right_split^,
                branch^,
                depth=child_depth,
                bounds=children.right.copy(),
            )
        )
        n_leaves += 1

    tree.n_leaves = n_leaves
    return tree^


def node_bounds(tree: Tree, monotone: List[Int]) -> List[OutputBounds]:
    """Every node's monotone output interval, recovered from a grown tree.

    Growth does not store the intervals, and it does not have to: an internal
    node still holds the leaf value it carried when it was created, which is
    exactly what its parent's interval was divided at, so one pass down the
    tree reproduces the whole chain. Consumers that rewrite leaf values after
    growth (quantile and L1 leaf renewal in boosting.mojo) clamp into these.

    Returns an empty list when `monotone` is empty, since then no node is
    bounded.
    """
    var bounds = List[OutputBounds]()
    if len(monotone) == 0:
        return bounds^
    var n_nodes = len(tree.feature)
    bounds.resize(n_nodes, OutputBounds.unbounded())
    # Both growers append children after their parent, so node ids increase
    # down the tree and one ascending pass propagates every interval.
    for node in range(n_nodes):
        if tree.feature[node] < 0:
            continue
        var left = tree.left[node]
        var right = tree.right[node]
        var children = child_bounds(
            bounds[node],
            monotone_sign(monotone, tree.feature[node]),
            tree.value[left],
            tree.value[right],
        )
        bounds[left] = children.left.copy()
        bounds[right] = children.right.copy()
    return bounds^
