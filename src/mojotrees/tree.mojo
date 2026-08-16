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
the allow mask is empty for every branch, so the branch set is never read;
growth then leaves the branch sets empty too rather than building a list per
split that nothing consumes, and the unconstrained path allocates nothing
for interaction constraints at all.

Feature subsampling (see sampling.mojo) draws one feature set per tree from
`tree_index` and the seed, then optionally a set per depth and a set per
node, in that order. Only the tree's set is ever accumulated into histograms,
so excluded features cost nothing and sibling subtraction stays exact; the
per-depth and per-node sets narrow the split search on top of that. Both
inner fractions default to 1.0, which passes the tree's set through
untouched -- and when the tree's set is every feature (the default
`feature_fraction`), a node's set is left empty instead of copied, since the
empty list already means "every feature" to the split scan, the CEGB costing
and the profile's cell count. That is the same selection with no per-node
list allocation; a node draws only when some fraction actually narrows it.

Exclusive feature bundling (see efb.mojo) is a histogram layout and nothing
more. `grow_tree` optionally takes a `BundledMatrix`, and then each node's
histogram is accumulated over the bundled matrix -- one column scan per
bundle instead of one per feature, which is the whole saving -- and expanded
straight back into the per-feature shape everything else reads. Rows are
still partitioned by the *original* matrix, because a chosen split names an
original feature and an original bin. So split search, leaf values, sibling
subtraction, and the tree itself are untouched by bundling, and no consumer
-- prediction, serialization, importance, contributions -- needs to know a
plan existed. It is off by default.

The remaining LightGBM tree controls ride on `TreeParams.extra` (see
tree_parameters_extra.mojo). The split-side rules reach the scan through
`_search`; `max_delta_step` and `path_smooth` are applied here, in
`_leaf_value`, because they need a leaf's row count and its parent's finished
output. A leaf is therefore valued before its own split search runs, since
its value is what its candidates' children smooth toward.

Under monotonic constraints (see monotone.mojo) each frontier leaf also
carries the interval its output must lie in. Leaf values are clamped into it,
candidate splits are scored from clamped outputs and rejected when they run
against their feature's constraint, and a split on a constrained feature
divides the parent's interval between its children at the midpoint of their
values. The intervals are not stored on the tree: `node_bounds` recovers them
from a grown tree, because an internal node keeps the value it had when it was
created.
"""

from .apple_cpu_policy import (
    BIN_LAYOUT_AUTO,
    BIN_LAYOUT_FEATURE_MAJOR,
    BIN_LAYOUT_ROW_MAJOR,
    env_bin_layout,
    plan_row_block_count,
)
from .binning import BinnedMatrix
from .cegb import (
    CegbLedger,
    CegbNodeCosts,
    cegb_commit_split,
    cegb_stale_cached_gain,
    check_cegb_grower_support,
    prepare_cegb_node,
)
from .categorical import (
    CAT_BITSET_WORDS,
    CategoricalParams,
    CategoricalSpec,
    cat_pool_contains,
)
from .efb import (
    BundledMatrix,
    columns_for_features,
    expand_bundled_histogram,
)
from .histogram import (
    ConstHessianSettings,
    Histogram,
    build_histogram_into_scratch,
    build_histogram_subset_by_layout_into_scratch,
    choose_bin_layout_timed,
    subtract_histogram_into,
)
from .parallel import (
    DispatchSettings,
    plan_row_blocks_with,
    run_row_blocks,
)
from .phase_profile import (
    HOST_HIST_DISPATCHES,
    HOST_PARTITION_DISPATCHES,
    HOST_SPLIT_SEARCH_DISPATCHES,
    HOST_SUBTRACT_DISPATCHES,
    PROF_HISTOGRAM,
    PROF_HIST_ALLOC,
    PROF_PARTITION,
    PROF_SPLIT_SEARCH,
    PROF_SUBTRACT,
    SCOPE_TREE,
    PhaseProfile,
)
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
    DEFAULT_FEATURE_FRACTION_BYLEVEL,
    DEFAULT_FEATURE_FRACTION_SEED,
    check_feature_fractions,
    check_row_set,
    select_split_features,
    select_tree_features,
)
from .growth_policy import (
    GROW_LEAFWISE,
    GROW_OBLIVIOUS,
    OBLIVIOUS_MAX_DEPTH,
    GrowthSchedule,
    LeafCandidate,
    ObliviousTrace,
    SharedSplitAudit,
    check_grow_policy,
)
from .split import (
    SplitInfo,
    find_best_split,
    find_best_split_shared,
    soft_threshold_l1,
)
from .tree_parameters_extra import ExtraTreeParams, finish_leaf_output


struct TreeParams(Copyable, Movable):
    """Tree growth hyperparameters. `lambda_reg` is LightGBM's lambda_l2 and
    `lambda_l1` its lambda_l1; both default to LightGBM's own default, which
    is 0.0 for each. `lambda_reg` defaulted to 1.0 until 2026-08-16, the last
    tree default in this library that was not LightGBM's; it made every
    benchmark arm carry an explicit `lambda_l2` to keep the two engines on
    one regularizer, and it is stock now, so nothing has to be pinned to
    compare them.

    `constraints`
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
    parameters.

    `feature_fraction_bylevel` is XGBoost's `colsample_bylevel`, a third
    subsampling stage between the tree's set and the node's (see
    sampling.mojo). LightGBM has no equivalent, so it is an extension rather
    than a parity row; 1.0, the default, passes the tree's set through and
    leaves every existing selection bit-identical.

    `extra` is the remaining LightGBM tree controls
    (tree_parameters_extra.mojo): the gain floor, the leaf-output cap and
    smoothing, the extra-trees threshold draw, the monotone penalty, and the
    per-feature gain multipliers and split costs. Its default is inactive, and
    `ExtraTreeParams.is_active()` is what the grower and the split search test
    once per node rather than multiplying by 1.0 per candidate. The two fields
    are appended so that every positional caller of this constructor keeps
    working unchanged.

    `grow_policy` is XGBoost's `grow_policy` (growth_policy.mojo):
    `GROW_LEAFWISE`, the default and LightGBM's growth, commits the best
    split anywhere in the tree next; `GROW_DEPTHWISE` commits every admitted
    split at one depth before any deeper one, `num_leaves` staying a hard
    bound and the last level admitted as a gain-ranked prefix. LightGBM has
    no such switch, so it is an extension rather than a parity row. The
    default leaves every fit bit-identical.

    `GROW_OBLIVIOUS` is CatBoost's symmetric tree: one split is searched
    across a whole level and applied to every leaf of it. `max_depth` binds
    and is REQUIRED (a level count is the only bound there is);
    `num_leaves` does not bind and is ignored, because a level splits
    entirely or not at all. See `growth_policy.mojo` and
    `docs/design/OBLIVIOUS.md`. Only the dense CPU grower
    (`grow_tree_leaves_profiled`) implements it; every other grower refuses
    the code rather than growing a tree that is not symmetric."""

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
    var feature_fraction_bylevel: Float64
    var extra: ExtraTreeParams
    var grow_policy: Int

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
        feature_fraction_bylevel: Float64 = DEFAULT_FEATURE_FRACTION_BYLEVEL,
        var extra: ExtraTreeParams = ExtraTreeParams(),
        grow_policy: Int = GROW_LEAFWISE,
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
        self.feature_fraction_bylevel = feature_fraction_bylevel
        self.extra = extra^
        self.grow_policy = grow_policy

    @staticmethod
    def default() -> TreeParams:
        # LightGBM's stock defaults, every one of them, read from
        # include/LightGBM/config.h at v4.7.0: num_leaves 31,
        # min_data_in_leaf 20, lambda_l2 0.0 (`lambda_reg` here),
        # min_sum_hessian_in_leaf 1e-3 (`min_child_hess`), lambda_l1 0.0.
        # The remaining fields default in the signature above, also to
        # LightGBM's values: max_depth -1, both feature fractions 1.0,
        # interaction and monotonic constraints none.
        #
        # `tools/check_parity.py` asserts every number on this line against
        # LightGBM's, so a default cannot drift off stock without failing a
        # gate. Changing one is a change to every fit that did not set it.
        return TreeParams(31, 20, 0.0, 1e-3, 0.0)


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

    def reserve_nodes(mut self, n_nodes: Int):
        """Size the ten per-node arrays for a tree of at most `n_nodes` nodes.

        `_add_node` appends to ten separately allocated `List`s, each of which
        doubles on its own schedule, so a grower that does not call this pays
        ten independent geometric reallocation sequences per tree: for a
        31-leaf tree that is ten lists times six doublings, and every doubling
        copies what is already there. A grower knows its node budget before it
        starts -- a tree with `num_leaves` leaves has exactly
        `2 * num_leaves - 1` nodes at most -- so one reservation per tree
        replaces all of it.

        Reserve only; it never shrinks, never resizes, and leaves the tree's
        length and contents alone, so calling it changes no value anywhere."""
        if n_nodes <= 0:
            return
        self.feature.reserve(n_nodes)
        self.threshold_bin.reserve(n_nodes)
        self.left.reserve(n_nodes)
        self.right.reserve(n_nodes)
        self.value.reserve(n_nodes)
        self.split_gain.reserve(n_nodes)
        self.default_left.reserve(n_nodes)
        self.missing_bin.reserve(n_nodes)
        self.cat_offset.reserve(n_nodes)
        self.count.reserve(n_nodes)

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
        mojotrees's own numbering: it is not LightGBM's leaf id, and the two
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
    the root, the interval its output must lie in (unbounded when no
    monotonic constraint above it applies), and the forced-split node it owes
    (-1, the default, when it owes none).

    `forced` is how `params.extra.forced` reaches the growth loop: a leaf
    carrying one is split on that node's feature and threshold instead of on
    the best split its histogram offers, and its two children inherit that
    node's own children. Every leaf reaches -1 once the forced tree is
    exhausted, which is where ordinary leaf-wise growth takes over."""

    var node: Int
    var rows: List[Int]
    var hist: Histogram
    var split: SplitInfo
    var branch: List[Int]
    var depth: Int
    var bounds: OutputBounds
    var forced: Int

    def __init__(
        out self,
        node: Int,
        var rows: List[Int],
        var hist: Histogram,
        var split: SplitInfo,
        var branch: List[Int] = [],
        depth: Int = 0,
        var bounds: OutputBounds = OutputBounds.unbounded(),
        forced: Int = -1,
    ):
        self.node = node
        self.rows = rows^
        self.hist = hist^
        self.split = split^
        self.branch = branch^
        self.depth = depth
        self.bounds = bounds^
        self.forced = forced

    def take_hist(mut self) raises -> Histogram:
        """Move this leaf's histogram out, leaving an empty one behind.

        Used when a leaf is split: its histogram is dead the moment sibling
        subtraction has read it, so the buffer goes back to the pool instead
        of being freed with the state it hung off."""
        var out = Histogram.zeroed(0, 0)
        swap(out, self.hist)
        return out^

    def take_rows(mut self) raises -> List[Int]:
        """Move this leaf's row ids out, leaving an empty list behind.

        Used when growth finishes: the frontier's row lists are exactly the
        leaf membership a score update wants (see `LeafMembership`), and
        moving them out is what lets a caller have them without copying.
        Mojo 1.0 forbids partial field moves, so this is a swap, the same
        shape `take_hist` uses for the histogram."""
        var out = List[Int]()
        swap(out, self.rows)
        return out^


struct LeafMembership(Movable):
    """Which training rows each leaf of a freshly grown tree holds.

    `node[l]` is the tree node id of the l-th leaf and `rows[l]` is that
    leaf's training row ids in ascending order, the order
    `partition_rows_into` produces. Both lists have `tree.n_leaves` entries,
    and the row lists partition the rows the tree was grown on: each such row
    appears in exactly one of them, exactly once.

    The grower already had this and used to drop it. A trainer that keeps it
    adds a tree's contribution to the raw scores leaf by leaf --
    `raw[r] += learning_rate * tree.value[node[l]]` for every r in `rows[l]`
    -- instead of walking the whole tree once per row to recover the leaf the
    partition already named. The added quantity is the same Float64 product
    added to the same accumulator, so the raw scores are bit-identical to the
    traversal's; only the route to the leaf changes.

    The leaf value is read from the tree at update time rather than cached
    here, because leaf renewal (`boosting._renew_leaf_values`, for QUANTILE,
    L1, and MAPE) rewrites `tree.value` after growth and before the update.
    Renewal rewrites leaves in place and never renumbers a node, so the node
    ids stay the ones growth handed back.

    `covers_all_rows` is False when the tree was grown on a bag (bagging,
    GOSS, balanced bagging): the row lists then cover the bag alone while
    every row of the dataset still needs the tree's contribution, so a caller
    must fall back to traversal for that tree. It is not a hint. A caller
    that ignores it silently leaves the unbagged rows unscored.
    """

    var node: List[Int]
    var rows: List[List[Int]]
    var covers_all_rows: Bool

    def __init__(out self):
        self.node = List[Int]()
        self.rows = List[List[Int]]()
        self.covers_all_rows = False

    @always_inline
    def n_leaves(self) -> Int:
        return len(self.node)

    def clear(mut self):
        """Drop the previous tree's membership. A trainer keeps one of these
        across a whole fit and refills it each round, so the row lists it
        already grew are freed here rather than accumulating."""
        self.node.clear()
        self.rows.clear()
        self.covers_all_rows = False


@fieldwise_init
struct RowPartition(Movable):
    """One node's rows split by a chosen split, each side in ascending row
    order."""

    var left: List[Int]
    var right: List[Int]


def partition_split_rows(
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
    settings: DispatchSettings = DispatchSettings.unresolved(),
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
    # The last hot per-node dispatch that was still reading the environment.
    # Every other one in the grower takes the fit-scoped snapshot; this one
    # called the live `plan_row_blocks`, so a fit paid one `getenv` sweep --
    # and, above the crossover, a whole `CpuProfile.detect()` -- per split.
    # Unresolved settings fall through to the live path, so the default keeps
    # every caller outside the grower working unchanged.
    var blocks = plan_row_blocks_with(settings, n, 3 * n)
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


def fill_identity_rows(
    mut rows: List[Int],
    n: Int,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """`rows` becomes `[0, 1, ..., n)`, filled over row blocks.

    The grower's root row list, and the one row list in a tree that no
    partition produces. It was also the last per-row pass in the partition
    phase still running on one core: `n` stores per tree, 8 MB of them at a
    million rows, and a hundred rounds of that in a fit.

    Elementwise over disjoint ascending blocks, so `rows[i] == i` whatever the
    block count and whatever `MOJOTREES_NUM_WORKERS` says. Nothing is
    accumulated and nothing is reassociated: the list is identical index for
    index to the one the serial loop wrote, at every worker count and on every
    machine.

    The block plan is charged one op per row. A fill is one store per row with
    no indirection, which is the cheapest per-row work in this file, so it
    reaches the parallel path later than the partition does on the same `n` --
    deliberately, since a fill that fits in the dispatch overhead should stay
    on one core.

    The length is taken without initializing the new elements, which is what
    the caller's next act -- writing every one of them -- makes safe, and it
    is worth a pass: `resize(n, 0)` writes `8 * n` bytes of zeros that the
    fill then immediately overwrites, so the root list cost `16 MB` of stores
    at a million rows where it needs `8 MB`. `Int` is trivially destructible,
    so shrinking this way cannot leak, and no element is readable before the
    fill covers it.
    """
    if n < 0:
        raise Error("row count must be nonnegative")
    rows.resize(unsafe_uninit_length=n)
    if n == 0:
        return
    var p = rows.unsafe_ptr()
    var blocks = plan_row_blocks_with(settings, n, n)

    def fill_block(b: Int) {imm}:
        for i in range(blocks.start(b), blocks.end(b)):
            p.unsafe_store(i, i)

    run_row_blocks(blocks, fill_block)


def _fill_identity_i32(
    mut buf: List[Int32],
    n: Int,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """`buf[0 : n) = [0, n)`, over row blocks. `RowArena.root_identity`'s
    body, split out so the closure captures a pointer into a plain list
    argument rather than into a field of `self`."""
    if n <= 0:
        return
    var p = buf.unsafe_ptr()
    var blocks = plan_row_blocks_with(settings, n, n)

    def fill_block(b: Int) {imm}:
        for i in range(blocks.start(b), blocks.end(b)):
            p.unsafe_store(i, Int32(i))

    run_row_blocks(blocks, fill_block)


def _fill_from_i32(
    mut buf: List[Int32],
    src_rows: List[Int],
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises:
    """`buf[0 : len(src_rows)) = src_rows`, narrowed to `Int32`, over row
    blocks. `RowArena.root_from_bag`'s body, split out for the same reason."""
    var n = len(src_rows)
    if n <= 0:
        return
    var p = buf.unsafe_ptr()
    var s = src_rows.unsafe_ptr()
    var blocks = plan_row_blocks_with(settings, n, n)

    def fill_block(b: Int) {imm}:
        for i in range(blocks.start(b), blocks.end(b)):
            p.unsafe_store(i, Int32(s.unsafe_load(i)))

    run_row_blocks(blocks, fill_block)


@fieldwise_init
struct LeafSpan(Copyable, Movable):
    """One node's rows, as a window into a `RowArena` rather than a list.

    `side` names which of the arena's two buffers holds them: 0 for `a`, 1 for
    `b`. A span is meaningless without it, which is why the three travel
    together and why a leaf state carries a `LeafSpan` and not a bare pair of
    integers.
    """

    var begin: Int
    var count: Int
    var side: Int

    @always_inline
    def end(self) -> Int:
        return self.begin + self.count


@fieldwise_init
struct ArenaPartition(Copyable, Movable):
    """The two child spans `partition_arena_span` produced. Both name the
    buffer the parent's did not, and together they cover the parent's window
    exactly: `left.begin == parent.begin`, `right.begin == left.end()`, and
    `right.end() == parent.end()`."""

    var left: LeafSpan
    var right: LeafSpan


struct RowArena(Movable):
    """A row-id permutation and its double buffer, owned once and reused.

    Growth today allocates two fresh `List[Int]` per split and one fresh
    `List[Int]` of every row per tree. At a million rows that is 8 MB for the
    root plus, over a 31-leaf tree, one write of every internal node's rows
    into a freshly faulted page -- around 48 MB of first-touch traffic and 61
    allocations per tree, times a hundred rounds.

    An arena replaces all of it with two `Int32` arrays of `n_rows`, allocated
    once and never freed until the arena is. Each leaf owns a disjoint window
    of one of them, and a split rewrites the parent's window into *the same
    window of the other buffer*, left side first. Nothing is allocated, and
    the children's spans are `(begin, n_left)` and `(begin + n_left, n -
    n_left)` on the opposite side.

    Two buffers rather than one, which is the whole reason this is not
    `gpu_active_rows.partition_range_host` with the types changed. That
    function scatters into a scratch and then copies the window back so the
    result lands in the buffer it started in, which is what a device-resident
    permutation with one canonical address needs. On the host nothing needs a
    canonical address -- a leaf state can carry which buffer it is in -- so the
    copy-back pass is pure loss: one extra read and one extra write of every
    row at every split, which over a balanced 31-leaf tree at a million rows
    is 40 MB per tree of traffic that buys nothing. Ping-pong deletes it. The
    partition itself is the same algorithm, and both produce the same bytes.

    **Element type.** `Int32` halves every row-id load and store against the
    `List[Int]` lists it replaces, and a bin matrix is indexed by an `Int`
    row anyway so nothing downstream widens. It bounds a dataset at 2^31 rows,
    which is the same bound `gpu_active_rows` already imposes on any fit that
    reaches the device.

    **Order.** The partition is stable and order-preserving: each side comes
    out in ascending position order, exactly the order `partition_rows_into`
    leaves its two lists in. That is not a nicety. Histogram accumulation
    visits a node's rows in the order the row list gives them and sums
    `Float64` in that order, so a partition that permuted a side would move
    every histogram cell below it. The equality is checked element for element
    in `tests/test_cpu_partition.mojo`.
    """

    var a: List[Int32]
    var b: List[Int32]
    var n: Int
    """Rows the arena is currently sized for. The buffers are never shrunk, so
    `len(a)` can exceed this after a smaller dataset."""

    def __init__(out self):
        self.a = List[Int32]()
        self.b = List[Int32]()
        self.n = 0

    def ensure(mut self, n: Int) raises:
        """Size both buffers for `n` rows, growing only.

        A booster holding one arena across a fit allocates here on the first
        tree and never again, which is the point of the type. `n` above 2^31
        is refused rather than truncated into an `Int32`.
        """
        if n < 0:
            raise Error("row count must be nonnegative")
        if n > 2147483647:
            raise Error(
                "row arena holds Int32 row ids and cannot address ",
                n,
                " rows",
            )
        if len(self.a) < n:
            self.a.resize(n, Int32(0))
        if len(self.b) < n:
            self.b.resize(n, Int32(0))
        self.n = n

    def root_identity(
        mut self,
        n: Int,
        settings: DispatchSettings = DispatchSettings.unresolved(),
    ) raises -> LeafSpan:
        """Fill buffer `a` with `[0, n)` and return the root's span.

        The arena form of `fill_identity_rows`, and the same argument: it is
        elementwise over disjoint ascending blocks, so the buffer is identical
        at every worker count. Half the stores of the `List[Int]` form,
        because the ids are `Int32`.
        """
        self.ensure(n)
        _fill_identity_i32(self.a, n, settings)
        return LeafSpan(0, n, 0)

    def root_from_bag(
        mut self,
        bag: List[Int],
        settings: DispatchSettings = DispatchSettings.unresolved(),
    ) raises -> LeafSpan:
        """Fill buffer `a` with `bag` and return the root's span.

        The bagged root. `bag` is ascending and duplicate-free by
        `sampling.check_row_set`, which the grower has already enforced, and
        this copies it position for position, so the arena root is the bag in
        the bag's own order.
        """
        var n = len(bag)
        self.ensure(n)
        _fill_from_i32(self.a, bag, settings)
        return LeafSpan(0, n, 0)

    def row_at(self, span: LeafSpan, i: Int) raises -> Int:
        """The `i`-th row id of `span`. For tests and for the few per-node
        consumers that read a handful of rows; a bulk consumer takes the
        window."""
        if i < 0 or i >= span.count:
            raise Error("row index escapes the span")
        var j = span.begin + i
        return Int(self.a[j]) if span.side == 0 else Int(self.b[j])

    def span_rows(self, span: LeafSpan) raises -> List[Int]:
        """`span` materialized as the `List[Int]` the rest of the package
        still speaks.

        A bridge, and it is worth being exact about what it costs and who
        needs it. Two consumers on the CPU path take a whole node's row ids as
        a `List[Int]` and cannot take a window of `Int32`:

        - `histogram.build_histogram_subset_into_scratch`, which already takes
          a `(rows, row_start, row_count)` window and so needs only an
          `Int32` overload of that window, not a new calling convention;
        - `cegb.prepare_cegb_node` and `cegb.cegb_commit_split`, which read
          rows only when a lazy penalty is configured and return immediately
          otherwise.

        Until those exist, a caller that wants the arena and one of them pays
        `8 * count` bytes and one allocation here, which is exactly the cost
        the arena removed. That is why the grower does not call this on its
        hot path and why this function is not the integration: it is what a
        test uses to compare the arena against the shipped partition, and what
        a CEGB-configured node can use without a new histogram signature.
        """
        if span.begin < 0 or span.count < 0 or span.end() > self.n:
            raise Error("span escapes the arena")
        var out = List[Int](capacity=span.count)
        out.resize(span.count, 0)
        if span.side == 0:
            for i in range(span.count):
                out[i] = Int(self.a[span.begin + i])
        else:
            for i in range(span.count):
                out[i] = Int(self.b[span.begin + i])
        return out^


def _partition_span_into(
    src: List[Int32],
    mut dst: List[Int32],
    begin: Int,
    n: Int,
    data: BinnedMatrix,
    split: SplitInfo,
    missing_bin: Int,
    settings: DispatchSettings,
) raises -> Int:
    """`src[begin : begin + n)` routed into `dst[begin : begin + n)`, left side
    first, returning the left count.

    `partition_arena_span`'s body, with the two buffers named rather than
    selected out of the arena. Split out for two reasons: the closures capture
    pointers into plain list arguments instead of into fields of a struct,
    which is what the origin checker will carry into a parallel closure; and
    the source is `read` while only the destination is `mut`, which states in
    the signature that ping-pong never writes the window it is reading.
    """
    var blocks = plan_row_blocks_with(settings, n, 3 * n)
    var src_p = src.unsafe_ptr()
    var dst_p = dst.unsafe_ptr()
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
            var r = Int(src_p.unsafe_load(begin + i))
            if goes_left(Int(bins_p.unsafe_load(r))):
                c += 1
        counts_p.unsafe_store(b, c)

    run_row_blocks(blocks, count_block)

    # Exclusive prefix sum over the per-block left counts, in place.
    var total_left = 0
    for b in range(blocks.n_blocks):
        var c = left_counts[b]
        left_counts[b] = total_left
        total_left += c

    def scatter_block(b: Int) {imm}:
        var start = blocks.start(b)
        var li = begin + counts_p.unsafe_load(b)
        var ri = begin + total_left + (start - counts_p.unsafe_load(b))
        for i in range(start, blocks.end(b)):
            var r = src_p.unsafe_load(begin + i)
            if goes_left(Int(bins_p.unsafe_load(Int(r)))):
                dst_p.unsafe_store(li, r)
                li += 1
            else:
                dst_p.unsafe_store(ri, r)
                ri += 1

    run_row_blocks(blocks, scatter_block)
    return total_left


def partition_arena_span(
    mut arena: RowArena,
    span: LeafSpan,
    data: BinnedMatrix,
    split: SplitInfo,
    missing_bin: Int,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> ArenaPartition:
    """Route `span`'s rows to the left or right child of `split`, in place.

    `partition_rows_into` with the two freshly allocated `List[Int]` sides
    replaced by one window of the arena's *other* buffer: the left side lands
    at `span.begin` and the right at `span.begin + n_left`, so the two
    children partition the parent's window with no gap and no allocation. The
    parent's own window is left untouched, which is what lets a caller that
    has not finished with it -- a CEGB commit, a diagnostic -- still read it.

    Routing is the same rule, and it is the rule and not a copy of it: a
    categorical split routes by set membership; otherwise rows in the
    feature's missing bin follow the split's default direction instead of the
    threshold, and a feature with no missing bin (-1) has none of them.

    Two passes, same as `partition_rows_into`: count per block, exclusive
    prefix-sum the per-block left counts, then scatter. Block b's rows land at
    `begin + prefix_left[b]` on the left and at `begin + n_left + (local_start
    - prefix_left[b])` on the right. Both sides therefore come out in
    ascending position order whatever the block count -- the result is
    identical to the serial single-pass partition, index for index, and
    identical to `partition_rows_into`'s `left` followed by its `right`, also
    index for index. `tests/test_cpu_partition.mojo` asserts exactly that,
    with integer equality and no tolerance, because it is the property the
    whole design rests on: histogram accumulation sums `Float64` in row-list
    order, so a side that came out permuted would move every histogram cell
    beneath this node and every leaf value under it.

    **Against LightGBM's `DataPartition::Split`.** LightGBM keeps one global
    `indices_` array with a `(leaf_begin_, leaf_count_)` per leaf, which is
    the same structure as one `RowArena` buffer and a `LeafSpan` per leaf.
    Its runner differs from this one in two places, and both differences are
    deliberate here:

    - *Per-thread temporary buffers.* LightGBM's threads each append their
      left and right rows into thread-local arrays, because a thread does not
      know where its rows belong until every earlier thread has finished. This
      partition counts first and prefix-sums the per-block counts, so each
      block knows its exact destination offsets before it writes a single row
      and scatters straight into disjoint ranges of the destination. That
      removes the temporaries rather than reorganizing them: it changes
      neither the block plan (`plan_row_blocks_with`, unchanged, on the
      unchanged three-ops-per-row estimate) nor who owns a buffer, because
      there is no buffer to own. The trade is one extra read of the split
      feature's bins -- one `UInt8` per row -- against one fewer write of
      every row id, which is four bytes.
    - *The final ordered copy-back.* In LightGBM that pass is what reassembles
      the thread-local arrays into `indices_` in block order, and it is what
      makes their partition stable. Here the prefix-summed scatter is already
      the ordered write, so a copy-back would only move the result back to the
      address it started at. `partition_arena_span_inplace` is that variant,
      for a caller that needs the single-array invariant; this one ping-pongs
      instead and hands the children the other buffer, which is one read and
      one write of every row per split cheaper. See that function for the
      arithmetic.

    `settings` is the fit's dispatch snapshot and is threaded to the one
    `plan_row_blocks_with` below, exactly as `partition_rows_into` threads it.
    A partition that dropped it would put a `getenv` sweep and a core
    detection back on the per-split path.
    """
    if span.side != 0 and span.side != 1:
        raise Error("span side must be 0 or 1")
    if span.begin < 0 or span.count < 0 or span.end() > arena.n:
        raise Error("span escapes the arena")
    if split.feature < 0 or split.feature >= data.n_features:
        raise Error("split feature out of range")

    var begin = span.begin
    var n = span.count
    var dst_side = 1 - span.side
    if n == 0:
        return ArenaPartition(
            LeafSpan(begin, 0, dst_side), LeafSpan(begin, 0, dst_side)
        )

    # The two buffers are named by the branch rather than selected into a
    # variable, because `a` and `b` are distinct fields and one of them has to
    # be the mutable argument while the other stays a read-only one.
    var total_left: Int
    if span.side == 0:
        total_left = _partition_span_into(
            arena.a, arena.b, begin, n, data, split, missing_bin, settings
        )
    else:
        total_left = _partition_span_into(
            arena.b, arena.a, begin, n, data, split, missing_bin, settings
        )

    return ArenaPartition(
        LeafSpan(begin, total_left, dst_side),
        LeafSpan(begin + total_left, n - total_left, dst_side),
    )


def _copy_span(
    src: List[Int32],
    mut dst: List[Int32],
    begin: Int,
    n: Int,
    settings: DispatchSettings,
) raises:
    """`dst[begin : begin + n) = src[begin : begin + n)`, over row blocks.
    Elementwise over disjoint ascending blocks, so the window is identical at
    every block count."""
    if n <= 0:
        return
    var src_p = src.unsafe_ptr()
    var dst_p = dst.unsafe_ptr()
    var blocks = plan_row_blocks_with(settings, n, n)

    def copy_block(b: Int) {imm}:
        for i in range(blocks.start(b), blocks.end(b)):
            dst_p.unsafe_store(begin + i, src_p.unsafe_load(begin + i))

    run_row_blocks(blocks, copy_block)


def partition_arena_span_inplace(
    mut arena: RowArena,
    span: LeafSpan,
    data: BinnedMatrix,
    split: SplitInfo,
    missing_bin: Int,
    settings: DispatchSettings = DispatchSettings.unresolved(),
) raises -> ArenaPartition:
    """`partition_arena_span` with LightGBM's ordered copy-back, so both
    children come out on the side their parent was on.

    LightGBM's `DataPartition::Split` finishes by copying the partitioned
    range back into the global `indices_` array, and `gpu_active_rows.
    partition_range_host` does the same for the device's single resident
    permutation. This is that contract: one array is canonical, a leaf's rows
    are always at `indices[begin : begin + count)` of it, and a caller never
    has to track which buffer a span is in. The result is byte for byte what
    `partition_arena_span` produces -- same routing, same stable order, same
    left count -- landed at the same addresses instead of the mirrored ones.

    **It costs a full extra pass over the window**: one read and one write of
    every row id per split. Over a 31-leaf tree at a million rows the sum of
    internal node sizes is around 5 million rows, so at four bytes a row that
    is a **derived bound** of 5e6 x 8 = 40 MB of extra traffic per tree, and
    4 GB over a 100-round fit, buying only that the result lands at the
    address it started at. Nothing on the CPU path needs that: a `_LeafState`
    can carry its span's side in the same word it carries its depth. So
    `partition_arena_span` is the one a grower should call, and this exists
    for a caller that has to hold the single-array invariant -- a diagnostic
    comparing against the device, or a consumer that indexes one canonical
    buffer.
    """
    var got = partition_arena_span(
        arena, span, data, split, missing_bin, settings
    )
    if span.count > 0:
        if span.side == 0:
            _copy_span(arena.b, arena.a, span.begin, span.count, settings)
        else:
            _copy_span(arena.a, arena.b, span.begin, span.count, settings)
    return ArenaPartition(
        LeafSpan(got.left.begin, got.left.count, span.side),
        LeafSpan(got.right.begin, got.right.count, span.side),
    )


struct _HistPool(Movable):
    """Free-list of histogram buffers of one shape.

    Tree growth builds two child histograms per split and drops the parent's,
    so at most `num_leaves + 1` are ever live at once and the buffers can be
    recycled instead of reallocated. Each buffer is three arrays of
    `n_features * n_bins`, which at the default 100 features and 255 bins is
    around 600 KB per node: large enough that the allocator hands back fresh
    pages and faults them in every time. Recycling keeps that cost to the
    first few nodes of the first tree.

    **The free list needs no cap, and this paragraph is here so the idea is
    not re-proposed.** Follow the counts. The first tree allocates on the root
    take and on every split thereafter, so it creates exactly `num_leaves + 1`
    buffers and never more; from the second tree on it creates none, because
    the frontier drain at the end of a tree returns every buffer it held. The
    free list is therefore self-bounding at `num_leaves + 1`, which is the
    same number that is *live* in the frontier at the end of every tree. A cap
    below that would not lower the high-water mark by one byte -- the buffers
    it refused to keep are ones the very next tree has to hold simultaneously
    anyway -- it would only reintroduce the per-tree allocation this pool
    exists to remove.

    The residency it does imply, as a derived bound rather than a
    measurement: `(num_leaves + 1) * 24 * n_features * n_bins` bytes, the 24
    being `Float64` gradient plus `Float64` hessian plus `Int` count per cell.
    At the benchmark shape (31 leaves, 50 features, 255 bins) that is 9.3 MiB
    against a 50 MB binned matrix, and it scales with `num_leaves` rather than
    with rows. If that ceiling ever becomes the problem, the thing to change
    is the frontier holding one histogram per leaf, not this free list.

    Allocation is also not a speed cost here, and that is measured rather than
    argued: the in-run phase profile puts `hist_alloc` at 0.005 percent of the
    serial round at 1,000,000 x 50 (bench/results/cpu_round1_2026-08-16). The
    pool is already doing its job.
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


struct GrowScratch(Movable):
    """The working memory one grower call reuses across every node, and one
    booster reuses across every tree.

    Two buffers live here:

    - `pool`, the histogram free list. `_HistPool`'s own argument is that a
      buffer is three arrays of `n_features * n_bins` -- around 300 KB at 100
      features and 255 bins -- so the allocator hands back fresh pages and
      faults them in every time one is created. Constructing the pool per
      tree defeats exactly that argument: a 100-round fit rebuilt it 100
      times and paid the first-few-nodes cost 100 times. Held here, one pool
      serves a whole fit.
    - `pairs`, the gradient/hessian gather buffer
      `histogram.build_histogram_subset_into_scratch` fills, and behind it
      the row-blocked kernel's private histograms. The non-scratch form
      allocates and frees it per node, which its own docstring says a grower
      should not do. `_hist_full` takes the same list for the root build,
      where there is no gather and the partials sit at offset 0; that build
      is reached once per tree and its share of this buffer is the whole
      `apple_cpu_policy.MAX_ROW_BLOCK_SCRATCH_BYTES` budget at a large fit,
      so holding it here is one allocation and one page-fault pass per fit
      instead of one per tree.

    Neither buffer carries meaning between uses. A pooled histogram's
    contents are undefined on `take` and every `_into` builder writes every
    cell it will read before reading it; `pairs` is written for the current
    node's rows before it is read for them and is only ever grown. So the
    same growth that recycles these within a tree recycles them across trees
    on identical terms: no node can observe another node's leftovers, and a
    fit that shares one scratch produces the tree a fit that allocated a
    fresh one produces, cell for cell.

    Two snapshots live here beside them, `settings` and `bin_layout`. Both
    are environment reads hoisted out of the per-node path: without them a
    grower re-reads `getenv` and re-derives the same answer once per node for
    a decision the fit made before its first tree. Neither can move a cell --
    one is a task count, the other is which array a bin id is loaded from.

    `prepare` makes a scratch safe to hand to a different matrix. It rebuilds
    the pool when the histogram shape changes, so the free list can never
    hold buffers of a shape the new data would misread. `pairs` needs no such
    check: it is sized by row count alone and only grows, and neither snapshot
    depends on the shape.
    """

    var pool: _HistPool
    var pairs: List[Float64]
    var settings: DispatchSettings
    var bin_layout: Int
    var layout_pending: Bool
    """Whether `bin_layout` is still the placeholder rather than an answer.

    True exactly when the fit asked for `auto` and the timed probe has not
    run yet. It is the difference between "feature-major, because that is
    what was asked for" and "feature-major, because nobody has asked the
    question yet", and conflating the two is what left AUTO meaning
    feature-major for a whole round: `bin_layout` alone cannot tell them
    apart, so a probe keyed off it would either never fire or fire on an
    explicit `MOJOTREES_CPU_BIN_LAYOUT=feature`, which is an instruction and
    not a hint.

    It clears on the first node that is offered the probe, taken or declined,
    so a fit asks once and then never again.
    """

    def __init__(out self, n_features: Int, n_bins: Int) raises:
        self.pool = _HistPool(n_features, n_bins)
        self.pairs = List[Float64]()
        # Which copy of the bin ids the node builds read, resolved here for
        # the same reason `settings` is: `MOJOTREES_CPU_BIN_LAYOUT` is read
        # once per scratch instead of once per node, and the per-node call
        # then compares two Ints.
        #
        # `auto` becomes row-major rather than feature-major, and that is the
        # grower's decision to make rather than `resolve_bin_layout`'s: the
        # policy function answers "which layouts are available", and AUTO is
        # its way of saying "the caller has not chosen", which is exactly the
        # question this line answers. It costs nothing when the view does not
        # exist -- `build_histogram_subset_by_layout_into_scratch` degrades to
        # feature-major and reports it -- so on today's default
        # (`binning.env_row_major_bins` off) this line changes nothing at all
        # and the whole row-major path turns on with one environment
        # variable instead of with a code change.
        #
        # `MOJOTREES_CPU_BIN_LAYOUT=feature` is the off switch and reproduces
        # the accumulation that shipped, which is what a bisection wants.
        var requested = env_bin_layout()
        # AUTO maps to FEATURE-MAJOR, and that is a correction rather than a
        # preference. This line used to map AUTO to row-major on the stated
        # reasoning that "on today's default the view does not exist, so this
        # changes nothing at all". `lane/row-major-auto` falsified that
        # premise in the same round: the view is now built by default under a
        # memory budget, so mapping AUTO to row-major here would silently
        # turn every CPU fit under that budget row-major with no timing at
        # all -- the flat default flip that was explicitly declined and
        # replaced with LightGBM's auto rule.
        #
        # **AUTO is not a layout. It means nobody has chosen yet**, and until
        # the timed probe runs the layout that shipped is the one that runs.
        # `resolve_layout_timed` below is that probe: it is the missing half
        # this comment described, and it now exists. What survives from the
        # paragraph above is the placeholder -- a scratch whose probe has not
        # run, or which is never offered one, still runs the layout that
        # shipped, so nothing changes for the growers that do not call it.
        self.bin_layout = (
            BIN_LAYOUT_FEATURE_MAJOR
            if requested == BIN_LAYOUT_AUTO
            else requested
        )
        self.layout_pending = requested == BIN_LAYOUT_AUTO
        # The scheduling environment, read once here and then never again for
        # the life of this scratch. Every dispatch the grower makes -- the
        # histogram builds, the sibling subtractions, the split scans -- takes
        # this snapshot instead of re-reading `getenv` and re-detecting the
        # core counts per call, which is what they all did before.
        #
        # A snapshot does not depend on the histogram shape, so `prepare` does
        # not rebuild it: pointing a scratch at a different matrix does not
        # change what the machine is or what the user asked for.
        #
        # One behaviour change worth naming rather than discovering: an
        # off-ladder `MOJOTREES_CPU_FEATURE_GROUP` is refused by
        # `env_feature_group`, and that refusal now surfaces when the scratch
        # is constructed rather than at the first histogram build. Earlier,
        # and with the same message.
        self.settings = DispatchSettings.resolve()

    def prepare(mut self, n_features: Int, n_bins: Int) raises:
        """Point this scratch at a histogram shape, discarding pooled buffers
        of any other shape."""
        if self.pool.n_features != n_features or self.pool.n_bins != n_bins:
            self.pool = _HistPool(n_features, n_bins)

    def resolve_layout_timed(
        mut self,
        data: BinnedMatrix,
        grad: List[Float64],
        hess: List[Float64],
        rows: List[Int],
        start: Int,
        count: Int,
        features: List[Int],
        const_hessian: Bool,
        const_hessian_env: ConstHessianSettings,
    ) raises:
        """Settle `bin_layout` once per fit, by building this node both ways
        and keeping the faster. LightGBM's `force_row_wise` / `force_col_wise`
        auto rule, on this machine, on this dataset.

        **Once per fit, not once per tree.** A `GrowScratch` is constructed
        by the booster and lives across every tree, so this runs on one node
        of the first tree and the answer is then read from a field 6,100
        times a tree after that. `grow_tree` builds its own scratch and
        therefore probes on its own single tree, which is the correct answer
        for a caller who asked for exactly one tree.

        **It declines rather than guessing**, on any of three grounds, and
        clears `layout_pending` on every one of them so the question is asked
        once:

        - the matrix has no row-major view, so there is nothing to compare;
        - the node does not block, in which case both arms run their
          unblocked variants over a node small enough that the two timings
          are dispatch noise, and a wrong answer would then be pinned for the
          whole fit;
        - `MOJOTREES_CPU_BIN_LAYOUT` named a layout, which is an instruction
          and not a hint. That case never reaches here, because
          `layout_pending` is false unless the request was `auto`, and it is
          restated in `choose_bin_layout_timed` as well.

        **Cost, stated because it is not zero.** The probed node is built
        three times rather than once -- twice by the probe and once for
        real -- and that happens on one node of one tree of the fit. It is
        deliberately not the root: the root has the identity row list, where
        both layouts read sequentially and the question hardly arises, while
        the thousands of child builds that dominate a fit read scattered
        rows, which is where the layouts actually differ. Probing the root
        would time the one node least like the population it is deciding for.

        **It cannot move a cell.** Both builders take the same plan, visit
        the same rows in the same order and add the same Float64 into the
        same cell; only the address the bin id is loaded from differs. The
        probe's own output goes into a pooled buffer that is handed straight
        back, so no node's histogram comes from it.
        """
        if not self.layout_pending:
            return
        # Cleared on every path out, including the declines. A flag that
        # survived a decline would re-probe at every node of the fit.
        self.layout_pending = False
        if not data.has_row_major():
            return
        var n_active = len(features) if len(features) > 0 else data.n_features
        if n_active <= 0 or count <= 0:
            return
        if plan_row_block_count(0, count, data.n_bins, n_active, True) < 2:
            return
        # Three statements and not one, so that no two fields of `self` are
        # borrowed across the same expression: `pool` is borrowed, released,
        # `pairs` is borrowed, released, and only then is `bin_layout`
        # written.
        var probe = self.pool.take()
        var chosen = choose_bin_layout_timed(
            probe,
            self.pairs,
            data,
            grad,
            hess,
            rows,
            start,
            count,
            features,
            const_hessian,
            self.settings,
            const_hessian_env,
        )
        self.pool.give(probe^)
        self.bin_layout = chosen


def _leaf_value(
    hist: Histogram,
    lambda_reg: Float64,
    lambda_l1: Float64 = 0.0,
    feature: Int = 0,
    n_data: Int = 0,
    parent_output: Float64 = 0.0,
    max_delta_step: Float64 = 0.0,
    path_smooth: Float64 = 0.0,
) -> Float64:
    """The value one leaf emits, from its histogram.

    The Newton step `-T(G) / (H + lambda_l2)`, then LightGBM's
    `max_delta_step` cap and `path_smooth` shrinkage toward the parent's
    finished output. Both default to off, in which case this returns the
    Newton step alone and every existing caller is unchanged; `n_data` is the
    leaf's row count, which is what makes the smoothing fade as a leaf grows,
    and `parent_output` is 0.0 at the root.
    """
    # Totals over one feature's bins: every feature the histogram accumulated
    # has the same bin totals. `feature` must be one of them, which under
    # feature subsampling means a selected feature, since the excluded
    # features' slices are left at zero.
    var base = feature * hist.n_bins
    var g = 0.0
    var h = 0.0
    for b in range(hist.n_bins):
        g += hist.grad_at(base + b)
        h += hist.hess_at(base + b)
    var value = -soft_threshold_l1(g, lambda_l1) / (h + lambda_reg)
    if max_delta_step <= 0.0 and path_smooth <= 0.0:
        return value
    return finish_leaf_output(
        value, max_delta_step, path_smooth, n_data, parent_output
    )


def _scan_cells(features: List[Int], n_features: Int, n_bins: Int) -> Int:
    """The cells one node's split scan reads, for `PhaseProfile.charge`.

    A node's feature list is empty exactly when it did not draw one, and an
    empty list means every feature to the scan -- the same convention
    `find_best_split` and `prepare_cegb_node` use. A drawn list is never
    empty (`sampling.selection_count` floors at 2), so the two cases cannot
    be confused. Resolving it here keeps the charge equal to what the scan
    actually reads instead of charging zero for an undrawn node."""
    var n_active = len(features) if len(features) > 0 else n_features
    return n_active * n_bins


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
    node: Int = 0,
    tree_index: Int = 0,
    parent_output: Float64 = 0.0,
    grower_applies_extra: Bool = False,
    cegb: CegbNodeCosts = CegbNodeCosts.inactive(),
    grower_applies_cegb: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
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
    partitioning identically.

    `params.extra` (tree_parameters_extra.mojo) travels with the parameters,
    so the rules that are a function of the histogram, the node's row count,
    and its depth -- `min_gain_to_split`, `monotone_penalty`,
    `feature_contri`, and the CEGB split cost -- are live for every caller
    here with no change on its side.

    The rest of the bundle cannot be: `extra_trees` is keyed by the node id
    and the tree index, and `max_delta_step`/`path_smooth` need the leaf's row
    count and its parent's finished output. A caller that supplies those sets
    `grower_applies_extra`; `tree.grow_tree` does. Any other caller leaves it
    False and an active value is refused here rather than silently drawing
    every node's threshold from node 0's stream, or growing a tree whose
    leaves ignore the cap the caller asked for.

    Exclusive feature bundling never reaches here: `hist` is always a
    per-feature histogram, because `grow_tree` expands a bundled one back into
    that shape before searching it (see `_hist_full`).
    """
    if params.extra.needs_grower_support() and not grower_applies_extra:
        raise Error(
            "extra_trees, max_delta_step, and path_smooth are applied by"
            " tree.grow_tree; this grower does not pass the node id, the"
            " leaf row count, or the parent output that they read. Train on"
            " the dense CPU grower, or leave them at their defaults"
        )
    # The same contract for CEGB, split along the line of what the term
    # needs: the split cost is a function of the node's row count and is live
    # for every caller here, while the coupled and lazy penalties read a
    # ledger that spans the ensemble and are refused for a grower that does
    # not carry one. `cegb` is that grower's prepared per-node costs.
    check_cegb_grower_support(
        params.extra.penalties.cegb, grower_applies_cegb, grower_applies_cegb
    )
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
        extra=params.extra,
        n_rows=n_rows,
        depth=depth,
        node=node,
        tree_index=tree_index,
        parent_output=parent_output,
        cegb=cegb,
        settings=settings,
        # CatBoost's `score_function`. `SCORE_L2` is the field's default and
        # is `find_best_split`'s, so a fit that does not name the parameter
        # makes the identical call it made before this argument was passed.
        # This line and its pair in `_grow_oblivious_levels` are the only two
        # places in the package that carry the choice into a split search;
        # everything else that searches splits is refused or declined by
        # `ExtraTreeParams.is_active()`.
        score_function=params.extra.score_function,
    )


def _expand_bundled(
    mut hist: Histogram, scratch: Histogram, bundled: BundledMatrix,
    features: List[Int],
) raises:
    """Turn a per-bundle histogram in `scratch` into the per-feature histogram
    every consumer downstream expects, in `hist`.

    The buffers are passed as separate lists rather than reached through
    `hist` for the reason `histogram.build_histogram_into` gives: a pointer
    taken from a struct field carries that field's origin. There are two of
    them per histogram now, not three: `_gh` is the interleaved
    `(gradient, hessian)` plane and `_count` is the count.
    """
    expand_bundled_histogram(
        hist._gh,
        hist._count,
        hist.n_bins,
        bundled.plan,
        scratch._gh,
        scratch._count,
        scratch.n_bins,
        features,
    )


def _hist_full(
    mut hist: Histogram,
    mut scratch: Histogram,
    mut pairs: List[Float64],
    data: BinnedMatrix,
    bundled: BundledMatrix,
    grad: List[Float64],
    hess: List[Float64],
    features: List[Int],
    columns: List[Int],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
) raises:
    """Accumulate every row into `hist`, which is always per feature.

    Without bundling that is one call to the ordinary builder. With it, the
    accumulation runs over the bundled matrix into `scratch` -- O(#rows x
    #bundles) instead of O(#rows x #features), which is the whole point of
    bundling -- and the result is expanded back into per-feature shape. The
    two matrices are spelled out rather than selected through a reference,
    because a `BinnedMatrix` chosen by a conditional would be copied.

    `const_hessian` is the caller's declaration that every entry of `hess` is
    exactly `histogram.CONSTANT_HESSIAN`, which is a property of the objective
    and never of the array; `histogram.objective_has_constant_hessian` is the
    predicate that answers it and this grower does not evaluate it, because
    the two things that falsify it -- sample weights and GOSS -- are visible
    at the trainer and not here. It is passed straight through to the builder,
    which produces a bit-identical histogram either way, so it changes how much
    memory traffic the accumulation costs and nothing a consumer can read.

    It reaches the builder on both arms, bundled and not. The bundled arm is
    the one worth stating explicitly: the elision happens inside the
    accumulation over the bundled matrix and the hessian plane is refilled from
    the count before the builder returns, so `scratch` holds three complete
    planes by the time `_expand_bundled` reads it and the expansion sees
    exactly what it would have seen on the three-plane path.

    `pairs` is the grower's scratch list, the same one `_hist_subset` hands
    to the subset builder, and here it carries only the row-blocked kernel's
    private histograms -- the whole-dataset builder reads its derivatives
    sequentially and never gathers, so the `[0, n_sub)` prefix that
    `_hist_subset` reserves for the gather has no counterpart on this path
    and the partials start at offset 0. `build_histogram_into` is this call
    with a fresh empty list in its place: it allocated
    `plan.block_scratch_floats()` Float64 -- the whole
    `MAX_ROW_BLOCK_SCRATCH_BYTES` budget at the root of a large fit -- and
    zero-filled it once per tree, then freed it.

    Nothing reads the zero fill. `_accumulate_blocked_at` zeroes each
    `(block, group)` unit's private slice as the first thing that unit does,
    and the fold reads only slices that pass zeroed, so the buffer's contents
    on entry are irrelevant exactly as they are for `_hist_subset`. That is
    the same contract `build_histogram_into_scratch` documents and the same
    one the subset builder has relied on since the grower started holding
    this list across nodes; the histogram that comes out is the one the
    allocating form produced, cell for cell.
    """
    if not bundled.active:
        build_histogram_into_scratch(
            hist, pairs, data, grad, hess, features, const_hessian, settings,
            const_hessian_env,
        )
        return
    build_histogram_into_scratch(
        scratch, pairs, bundled.data, grad, hess, columns, const_hessian,
        settings, const_hessian_env,
    )
    _expand_bundled(hist, scratch, bundled, features)


def _hist_subset(
    mut hist: Histogram,
    mut scratch: Histogram,
    mut pairs: List[Float64],
    data: BinnedMatrix,
    bundled: BundledMatrix,
    grad: List[Float64],
    hess: List[Float64],
    rows: List[Int],
    start: Int,
    count: Int,
    features: List[Int],
    columns: List[Int],
    const_hessian: Bool = False,
    settings: DispatchSettings = DispatchSettings.unresolved(),
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
    layout: Int = BIN_LAYOUT_FEATURE_MAJOR,
) raises:
    """`_hist_full` for a row subset. Row ids index the original matrix and the
    bundled one identically, because bundling rearranges columns and never
    rows.

    `pairs` is the grower's gradient/hessian gather buffer, held for the whole
    tree (see `GrowScratch`). `build_histogram_subset_into` is the same call
    with a fresh empty list in its place -- it allocates one, hands it over,
    and frees it -- and its docstring says a grower visiting hundreds of nodes
    should hold one instead. Its contents on entry are irrelevant: the gather
    pass writes `[0, 2 * count)` before the accumulation pass reads any of it,
    and `ensure_pair_capacity` only ever grows the buffer. So the histogram
    that comes out is the one the allocating form produces, cell for cell.

    `const_hessian` carries the same declaration `_hist_full` documents, with
    the same guarantee: the builder produces a bit-identical histogram whether
    it is set or not, and setting it wrongly produces a wrong hessian plane, so
    it has to come from the objective at the trainer rather than be guessed
    anywhere on this path.

    `layout` is which copy of the bin ids the inner loop loads from --
    `apple_cpu_policy.BIN_LAYOUT_FEATURE_MAJOR` is the column-wise matrix that
    has always been there, `BIN_LAYOUT_ROW_MAJOR` is
    `BinnedMatrix.build_row_major`'s record array. It is a **scheduling**
    argument and cannot change a cell: both kernels take the same plan from
    `_plan_accumulation`, visit the same rows in the same order, and add the
    same Float64 into the same cell. Only the address the bin id comes from
    differs, which is what `tests/test_row_major_bins.mojo` asserts to the bit
    and `tests/test_grow_bin_layout.mojo` asserts again through a whole grown
    tree.

    A request for the row-major layout on a matrix that has no view degrades
    to feature-major inside `build_histogram_subset_by_layout_into_scratch`
    and says so in its return value, which is why this can be the grower's
    default while `binning.env_row_major_bins` decides separately whether the
    view is worth its memory. The bundled arm degrades for exactly that
    reason: `efb` assembles `bundled.data` itself and never builds a view on
    it, so bundling stays on the layout it always ran.
    """
    if not bundled.active:
        _ = build_histogram_subset_by_layout_into_scratch(
            hist, pairs, data, grad, hess, rows, start, count, layout,
            features, const_hessian, settings, const_hessian_env,
        )
        return
    _ = build_histogram_subset_by_layout_into_scratch(
        scratch, pairs, bundled.data, grad, hess, rows, start, count, layout,
        columns, const_hessian, settings, const_hessian_env,
    )
    _expand_bundled(hist, scratch, bundled, features)


def _check_oblivious(params: TreeParams, data: BinnedMatrix) raises:
    """What `grow_policy = oblivious` refuses, asked once per tree before the
    root histogram rather than part way down a level.

    Every item here is refused rather than half-applied, which is this
    package's rule for a parameter whose meaning does not survive the mode
    (see `find_best_split`'s treatment of `extra_trees` on categoricals).

    - `max_depth` is REQUIRED and is the only bound. A level splits entirely
      or not at all, so `num_leaves` cannot be met exactly and is ignored.
      CatBoost resolves the same collision by overwriting `max_leaves` with
      `1 << depth` and raising if the user set a different value
      (`catboost/private/libs/options/catboost_options.cpp`: "max_leaves
      option works only with lossguide tree growing"); `TreeParams` cannot
      tell a defaulted `num_leaves` from an explicit one, so we ignore it and
      say so here and in the module docstring rather than raise on a value
      the caller never chose.
    - A categorical feature's candidates are category *sets*, ordered by that
      node's own gradient/hessian ratios inside `find_best_categorical_split`.
      There is no one set to share across a level without a different search,
      so the combination is refused rather than searched per leaf and
      averaged into something that is not a partition.
    - Forced splits describe a specific asymmetric binary tree; applying one
      to a single leaf of a level is exactly the thing this mode does not do.
    - `extra_trees` draws one threshold per node, and the CEGB penalties that
      read the ensemble ledger are charged per node. Both are refused by
      `find_best_split_shared` too; asked here so the message arrives before
      any work.
    """
    if params.max_depth <= 0:
        raise Error(
            "grow_policy=oblivious requires max_depth > 0: a symmetric tree"
            " is bounded by its depth and by nothing else (num_leaves does"
            " not bind, because a level splits entirely or not at all)."
            " CatBoost's own default is depth=6"
        )
    if params.max_depth > OBLIVIOUS_MAX_DEPTH:
        raise Error(
            "grow_policy=oblivious refuses max_depth > ",
            OBLIVIOUS_MAX_DEPTH,
            ": depth d costs 2^(d+1) - 1 nodes unconditionally, so ",
            params.max_depth,
            " would build a tree of that size whatever the data does",
        )
    if data.cats.any_categorical():
        raise Error(
            "grow_policy=oblivious is implemented for numerical thresholds"
            " only; a categorical feature is searched as category partitions"
            " whose order comes from one node's own statistics, and a level"
            " shares one split"
        )
    if not params.extra.forced.is_empty():
        raise Error(
            "forced splits describe an asymmetric binary tree and"
            " grow_policy=oblivious applies one split to every leaf of a"
            " level; the two cannot both hold"
        )
    if params.extra.extra_trees:
        raise Error(
            "extra_trees draws one threshold per node and a level of an"
            " oblivious tree has one split for every node in it"
        )
    if params.extra.penalties.cegb.is_active():
        raise Error(
            "the CEGB penalties are charged per node against a ledger that"
            " spans the ensemble; grow_policy=oblivious commits one split per"
            " level and that accounting is not written"
        )


def _grow_oblivious_levels(
    mut profile: PhaseProfile,
    mut scratch: GrowScratch,
    mut tree: Tree,
    mut frontier: List[_LeafState],
    mut bundle_scratch: Histogram,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    tree_features: List[Int],
    tree_columns: List[Int],
    value_feature: Int,
    per_node_draw: Bool,
    signs: List[Int],
    constrained: Bool,
    bundling: BundledMatrix,
    const_hessian: Bool,
    const_h_env: ConstHessianSettings,
    hist_cells: Int,
    tree_index: Int,
) raises -> Int:
    """Grow the rest of an oblivious tree from a frontier holding its root,
    and answer the finished leaf count.

    One level at a time. The level is the whole frontier -- every leaf of an
    oblivious tree sits at the same depth by construction, which this asserts
    rather than assumes -- and it costs exactly:

      1 shared split search  (`split.find_best_split_shared`, one dispatch)
      L partitions           (leaf by leaf, row order preserved)
      L histogram builds     (the smaller child of each pair)
      L subtractions         (the larger child, derived)

    against a depth-wise level's L searches for the same L partitions,
    builds, and subtractions. **Derived bound, not a measurement:** search
    dispatches per tree fall from `num_leaves - 1` to `max_depth`, which at
    depth 6 is 63 to 6. Nothing else about the per-level work changes, so
    this removes fan-out and barrier cost and no arithmetic.

    Leaf numbering and node-id order, which are a cross-backend contract
    --------------------------------------------------------------------
    **Leaf index is the bit pattern of a row's outcomes with the FIRST
    level's outcome as the LEAST significant bit**: a leaf at depth d has
    index `sum_l b_l * 2^l`, where `b_l` is 0 if the row went left at level l
    and 1 if it went right. That is CatBoost's numbering
    (`catboost/private/libs/algo/index_calcer.cpp`:
    `const ui32 splitWeight = 1 << splitParams.Depth;` and
    `indices[doc] += cmpOp(...) * level`), and it is what the device
    implements, so it is not ours to choose. Concretely: the left child keeps
    its parent's index and the right child adds `1 << level_depth`.

    **Node ids are assigned level by level, over the level's leaves in
    ascending LEAF INDEX, left child before right.** That is NOT ascending
    node id: at level 2 the leaves in node-id order are indices 0, 2, 1, 3,
    so the two orders first diverge when level 2's children are created, and
    a depth-3 tree already tells them apart. Host and device must agree on
    node ids or the node-identity test between them fails on trees that are
    both correct, which is why this is written down here rather than left to
    whatever the container did.

    The frontier itself is the same `_LeafState` list the leaf-wise loop
    keeps, with the same convention: a split replaces its parent's slot with
    the left child and appends the right. `leaf_ix` below is maintained in
    exact lockstep with it (slot i keeps the left child's index, the append
    carries the right child's), which is what lets the order above be read
    off the frontier without storing anything in `_LeafState` that the other
    two policies would have to carry. The `_LeafState.split` field is unused
    in this mode -- there is no per-leaf best split -- and is left as "none"
    rather than filled with a number no one may read.

    Empty leaves, which are a property of the mode and not a bug
    ------------------------------------------------------------
    A leaf that was illegal at the chosen candidate is split anyway, and its
    split may send every one of its rows one way. The empty child is a real
    leaf of the tree: it is never reached by a training row, it can be
    reached at prediction time, and it emits 0.0 (clamped into its parent's
    monotone interval). That is CatBoost's answer too -- `CalcAverage` in
    `catboost/private/libs/algo_helpers/online_predictor.h` returns 0 for a
    child of zero weight -- and it also avoids the 0/0 that `_leaf_value`
    would produce at `lambda_l2 = 0`. The consequence a caller has to know:
    an oblivious tree can carry nodes whose cover is 0, so
    `Tree.check_node_counts` (and therefore exact feature contributions)
    will refuse such a tree. Nothing else reads node covers that way.

    CatBoost deletes a level whose every leaf produced an empty child and
    stops growth there (`greedy_tensor_search.cpp`,
    `GetRedundantSplitIdx(GetIsLeafEmpty(...))`). That rule is unreachable
    here and is deliberately not implemented: a chosen split must have a
    strictly positive summed gain, a leaf with an empty child contributes
    either zero (it failed `min_child_hess`/`min_data_in_leaf`) or exactly
    0.0 (both minima are off, and the candidate's two terms then equal the
    parent term), so a level in which every leaf had an empty child sums to
    0.0 and is never chosen. Building the check would be building a gate that
    cannot open.
    """
    var trace = ObliviousTrace.resolve()
    var max_delta_step = params.extra.max_delta_step
    var path_smooth = params.extra.path_smooth
    var n_leaves = len(frontier)
    var level_depth = 0
    if len(frontier) != 1:
        raise Error(
            "oblivious growth starts from the root alone; this frontier holds",
            " ",
            len(frontier),
            " leaves",
        )
    # Leaf index per frontier slot, the root's being 0. Maintained in lockstep
    # with `frontier` below: the split writes the left child's index into the
    # parent's slot and appends the right child's, exactly as the frontier
    # does with the states themselves.
    var leaf_ix = List[Int](capacity=1)
    leaf_ix.append(0)

    while level_depth < params.max_depth:
        # The level in ascending LEAF INDEX, which is the cross-backend
        # contract and is not ascending node id (see the docstring).
        # Selection sort: a level is at most 2^max_depth wide and this keeps
        # the order checkable by eye, which is the argument
        # `growth_policy.rank_level` makes for the same shape.
        var order = List[Int](capacity=len(frontier))
        var last_ix = -1
        while len(order) < len(frontier):
            var best = -1
            for i in range(len(frontier)):
                if leaf_ix[i] <= last_ix:
                    continue
                if best < 0 or leaf_ix[i] < leaf_ix[best]:
                    best = i
            if best < 0:
                break
            last_ix = leaf_ix[best]
            order.append(best)
        if len(order) != len(frontier):
            raise Error("oblivious frontier holds a repeated leaf index")
        for k in range(len(order)):
            if frontier[order[k]].depth != level_depth:
                raise Error(
                    "oblivious frontier leaf ",
                    frontier[order[k]].node,
                    " sits at depth ",
                    frontier[order[k]].depth,
                    " in a level at depth ",
                    level_depth,
                    "; every leaf of a symmetric tree sits at the same depth",
                )

        var level_rows = 0
        for k in range(len(order)):
            level_rows += len(frontier[order[k]].rows)
        # If the level as a whole cannot make two legal children then no leaf
        # in it can, so every candidate would sum to 0.0 and nothing would be
        # chosen. Answering it here costs one comparison instead of a scan.
        if level_rows < 2 * params.min_data_in_leaf or level_rows < 2:
            trace.stop(tree_index, level_depth, String("level too small"))
            break

        # Leaf index 0 is the all-left path, whose node is the first one
        # created at every level, so `order[0]` is also the level's lowest
        # node id and the key below does not depend on the ordering rule.
        var level_node = frontier[order[0]].node

        # One feature draw for the whole level, keyed by its lowest node id.
        # A per-node draw would give the leaves of one level different
        # candidate sets, and they have to agree on one split.
        var level_features = List[Int]()
        if per_node_draw:
            level_features = select_split_features(
                tree_features,
                params.feature_fraction_bylevel,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                level_depth,
                level_node,
            )

        # The level's interaction-constraint mask is the INTERSECTION over
        # its leaves: the shared split lands in every branch, so a feature
        # any one of them forbids is forbidden for the level.
        var allowed = List[Bool]()
        if constrained:
            allowed.resize(data.n_features, True)
            for k in range(len(order)):
                var mask = params.constraints.allowed_features(
                    frontier[order[k]].branch
                )
                if len(mask) == 0:
                    continue
                for f in range(data.n_features):
                    if f >= len(mask) or not mask[f]:
                        allowed[f] = False

        # The level's histograms, moved out of the frontier in the order the
        # split will be applied in. That order is what fixes the cross-leaf
        # sum's addend order, and so its value, independently of the worker
        # count.
        var level_hists = List[Histogram](capacity=len(order))
        var level_bounds = List[OutputBounds](capacity=len(order))
        var level_outputs = List[Float64](capacity=len(order))
        for k in range(len(order)):
            level_hists.append(frontier[order[k]].take_hist())
            level_bounds.append(frontier[order[k]].bounds.copy())
            level_outputs.append(tree.value[frontier[order[k]].node])

        var audit = SharedSplitAudit.none()
        var search_started = profile.clock()
        var split = find_best_split_shared(
            audit,
            level_hists,
            lambda_reg=params.lambda_reg,
            min_child_hess=params.min_child_hess,
            min_data_in_leaf=params.min_data_in_leaf,
            lambda_l1=params.lambda_l1,
            allowed=allowed,
            features=level_features,
            missing_bins=data.missing_bin,
            monotone=signs,
            bounds=level_bounds,
            parent_outputs=level_outputs,
            extra=params.extra,
            n_rows=level_rows,
            depth=level_depth,
            node=level_node,
            tree_index=tree_index,
            settings=scratch.settings,
            # The pair of the line in `_search`: one level's shared split is
            # elected with the same functional a leaf-wise split is.
            # `SCORE_L2` is the default on both sides, so the untouched
            # bundle makes the identical call.
            score_function=params.extra.score_function,
        )
        # One charge for the level, at the level's row count, over the cells
        # the one dispatch read: the per-node figure times the leaves folded
        # into it.
        profile.charge(
            PROF_SPLIT_SEARCH,
            level_rows,
            search_started,
            dispatches=HOST_SPLIT_SEARCH_DISPATCHES,
            cells=len(order)
            * _scan_cells(level_features, data.n_features, data.n_bins),
        )

        if not split.found or not (split.gain > 0.0):
            while len(level_hists) > 0:
                scratch.pool.give(level_hists.pop())
            trace.stop(tree_index, level_depth, String("no positive gain"))
            break
        trace.level(
            tree_index,
            level_depth,
            split.feature,
            split.bin,
            split.gain,
            audit,
        )

        var split_missing_bin = -1 if split.is_categorical else (
            data.missing_bin[split.feature]
        )
        var split_sign = monotone_sign(signs, split.feature)
        var child_depth = level_depth + 1
        # Exactly two nodes per leaf of this level, known before the loop.
        tree.reserve_nodes(len(tree.feature) + 2 * len(order))

        for k in range(len(order)):
            var i = order[k]
            var parent_node = frontier[i].node
            var parent_ix = leaf_ix[i]
            var parent_rows = len(frontier[i].rows)

            var left_rows = List[Int]()
            var right_rows = List[Int]()
            if parent_rows > 0:
                var part_started = profile.clock()
                partition_rows_into(
                    left_rows,
                    right_rows,
                    data,
                    frontier[i].rows,
                    split,
                    split_missing_bin,
                    scratch.settings,
                )
                profile.charge(
                    PROF_PARTITION,
                    parent_rows,
                    part_started,
                    dispatches=HOST_PARTITION_DISPATCHES,
                )

            var builds_left = len(left_rows) <= len(right_rows)
            var built_rows = len(left_rows) if builds_left else len(right_rows)
            var derived_rows = (
                len(right_rows) if builds_left else len(left_rows)
            )
            var alloc_started = profile.clock()
            var left_hist = scratch.pool.take()
            var right_hist = scratch.pool.take()
            profile.charge(
                PROF_HIST_ALLOC,
                built_rows,
                alloc_started,
                cells=2 * hist_cells,
            )
            # A pooled buffer's contents are undefined. The bundled builders
            # write only the sampled features' slices, and a child with no
            # rows is not built at all, so both cases zero the buffer here
            # rather than inherit another node's statistics. The derived
            # sibling is fully written by the subtraction either way.
            var hist_started = profile.clock()
            if builds_left:
                if built_rows == 0 or bundling.active:
                    left_hist.reset()
                if built_rows > 0:
                    _hist_subset(
                        left_hist, bundle_scratch, scratch.pairs, data,
                        bundling, grad, hess, left_rows, 0, len(left_rows),
                        tree_features, tree_columns, const_hessian,
                        scratch.settings, const_h_env, scratch.bin_layout,
                    )
                    profile.note_node()
                    profile.charge(
                        PROF_HISTOGRAM,
                        built_rows,
                        hist_started,
                        dispatches=HOST_HIST_DISPATCHES,
                        slots_per_row=len(tree_columns),
                        cells=hist_cells,
                    )
                var sub_started = profile.clock()
                subtract_histogram_into(
                    right_hist, level_hists[k], left_hist, const_hessian,
                    scratch.settings, const_h_env,
                )
                profile.note_node()
                profile.charge(
                    PROF_SUBTRACT,
                    derived_rows,
                    sub_started,
                    dispatches=HOST_SUBTRACT_DISPATCHES,
                    cells=hist_cells,
                )
            else:
                if bundling.active:
                    right_hist.reset()
                _hist_subset(
                    right_hist, bundle_scratch, scratch.pairs, data, bundling,
                    grad, hess, right_rows, 0, len(right_rows), tree_features,
                    tree_columns, const_hessian, scratch.settings, const_h_env,
                    scratch.bin_layout,
                )
                profile.note_node()
                profile.charge(
                    PROF_HISTOGRAM,
                    built_rows,
                    hist_started,
                    dispatches=HOST_HIST_DISPATCHES,
                    slots_per_row=len(tree_columns),
                    cells=hist_cells,
                )
                var sub_started = profile.clock()
                subtract_histogram_into(
                    left_hist, level_hists[k], right_hist, const_hessian,
                    scratch.settings, const_h_env,
                )
                profile.note_node()
                profile.charge(
                    PROF_SUBTRACT,
                    derived_rows,
                    sub_started,
                    dispatches=HOST_SUBTRACT_DISPATCHES,
                    cells=hist_cells,
                )

            var parent_bounds = frontier[i].bounds.copy()
            var parent_output = tree.value[parent_node]
            # A child with no rows emits 0.0, clamped like any other value so
            # a monotone interval still bounds what it can emit.
            var left_value = parent_bounds.clamp(0.0)
            if len(left_rows) > 0:
                left_value = parent_bounds.clamp(
                    _leaf_value(
                        left_hist,
                        params.lambda_reg,
                        params.lambda_l1,
                        value_feature,
                        len(left_rows),
                        parent_output,
                        max_delta_step,
                        path_smooth,
                    )
                )
            var right_value = parent_bounds.clamp(0.0)
            if len(right_rows) > 0:
                right_value = parent_bounds.clamp(
                    _leaf_value(
                        right_hist,
                        params.lambda_reg,
                        params.lambda_l1,
                        value_feature,
                        len(right_rows),
                        parent_output,
                        max_delta_step,
                        path_smooth,
                    )
                )
            if split_sign != MONOTONE_FREE and left_value > right_value:
                var mid = midpoint(left_value, right_value)
                left_value = mid
                right_value = mid
            var children = child_bounds(
                parent_bounds, split_sign, left_value, right_value
            )
            var left_node = tree._add_node(left_value, Float64(len(left_rows)))
            var right_node = tree._add_node(
                right_value, Float64(len(right_rows))
            )
            tree.left[parent_node] = left_node
            tree.right[parent_node] = right_node
            tree._set_split(parent_node, split, split_missing_bin)

            var branch = List[Int]()
            if constrained:
                branch = extend_branch(frontier[i].branch, split.feature)

            # No per-leaf split is searched in this mode, so the field stays
            # "none" rather than carrying a number no consumer may read.
            frontier[i] = _LeafState(
                left_node,
                left_rows^,
                left_hist^,
                SplitInfo(-1, -1, 0.0, False),
                branch.copy(),
                depth=child_depth,
                bounds=children.left.copy(),
                forced=-1,
            )
            frontier.append(
                _LeafState(
                    right_node,
                    right_rows^,
                    right_hist^,
                    SplitInfo(-1, -1, 0.0, False),
                    branch^,
                    depth=child_depth,
                    bounds=children.right.copy(),
                    forced=-1,
                )
            )
            # The left child keeps its parent's index (its outcome bit at this
            # level is 0) and the right child sets the bit for THIS level,
            # which is the level's own power of two. First level, lowest bit.
            leaf_ix[i] = parent_ix
            leaf_ix.append(parent_ix + (1 << level_depth))
            n_leaves += 1

        while len(level_hists) > 0:
            scratch.pool.give(level_hists.pop())
        level_depth += 1

    return n_leaves


def grow_tree(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
    bundling: BundledMatrix = BundledMatrix.none(),
    const_hessian: Bool = False,
) raises -> Tree:
    """Grow one tree with no CEGB ledger: `grow_tree_with_cegb` with an inert
    one.

    This is the entry point every trainer that does not carry a
    `cegb.CegbLedger` calls, and it is unchanged for all of them. The inert
    ledger is not a silent default: `cegb_penalty_split` is still charged,
    because it reads only the node's row count, while
    `cegb_penalty_feature_coupled` and `cegb_penalty_feature_lazy` are
    *refused* by `check_cegb_grower_support` rather than charged as zero, so
    no caller can get a model that quietly ignored them. A trainer that wants
    them owns a ledger for the whole ensemble and calls `grow_tree_with_cegb`;
    `boosting.fit` and `boosting.fit_multiclass` do.

    A `mut` argument cannot be defaulted, which is why this is a second entry
    point rather than a defaulted parameter.

    `const_hessian` is forwarded unchanged; see `grow_tree_leaves_profiled`,
    which is where the whole family's declaration is documented.
    """
    var ledger = CegbLedger.none()
    return grow_tree_with_cegb(
        data, grad, hess, params, ledger, bag, tree_index, bundling,
        const_hessian,
    )


def grow_tree_with_cegb(
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    mut ledger: CegbLedger,
    bag: List[Int] = [],
    tree_index: Int = 0,
    bundling: BundledMatrix = BundledMatrix.none(),
    const_hessian: Bool = False,
) raises -> Tree:
    """`grow_tree_leaves_profiled` with a profile, a leaf membership, and a
    scratch of its own, the profile reported per tree.

    The signature that shipped, unchanged, and the entry point every caller
    that accumulates nothing across a fit uses, which is all of them except
    `boosting._boost_rounds`. It owns the three things a boosting loop would
    rather own itself:

    - a `PhaseProfile`. With `MOJOTREES_PHASE_PROFILE` unset -- the default
      -- it is off, this costs one `getenv` and one bucket allocation per
      tree, prints nothing, and grows exactly the tree it grew before the
      instrument existed. With it set, one `scope=tree` block is printed per
      tree;
    - a `LeafMembership`. The membership is grown either way -- it is the
      frontier's own row lists, which growth has always built -- so
      discarding it here saves nothing and costs nothing. It exists because
      most trainers only ever wanted the tree, and their call sites are
      unchanged;
    - a `GrowScratch`. This entry point pays one pool construction and one
      gather buffer per tree, which is what growth did before the scratch was
      threaded at all.

    Each of the three has its own further entry point rather than being a
    defaulted parameter, because a `mut` argument cannot be defaulted, which
    is also why the ledger is a second entry point beside `grow_tree`. A
    caller that wants one profile block per fit calls `grow_tree_profiled`; a
    trainer that ends its round by adding the tree's contribution to a raw
    score per row calls `grow_tree_leaves` and adds by leaf instead; a
    boosting loop that wants both holds all three across the fit and calls
    `grow_tree_leaves_profiled`. The four paths differ in nothing a tree can
    observe.
    """
    var profile = PhaseProfile.from_env(SCOPE_TREE, String("grow_tree"))
    var tree = grow_tree_profiled(
        profile, data, grad, hess, params, ledger, bag, tree_index, bundling,
        const_hessian,
    )
    profile.print_report()
    return tree^


def grow_tree_profiled(
    mut profile: PhaseProfile,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    mut ledger: CegbLedger,
    bag: List[Int] = [],
    tree_index: Int = 0,
    bundling: BundledMatrix = BundledMatrix.none(),
    const_hessian: Bool = False,
) raises -> Tree:
    """`grow_tree_leaves_profiled` with a caller-owned profile, and a leaf
    membership and scratch of its own.

    The entry point a caller takes when it accumulates a profile across a fit
    but wants neither the membership nor a scratch that outlives the tree.
    The membership is grown either way and dropped when this returns; the
    scratch is one pool construction and one gather buffer per call, which is
    what growth did before the scratch was threaded at all.

    `const_hessian` is forwarded unchanged; see `grow_tree_leaves_profiled`.
    """
    var leaves = LeafMembership()
    var scratch = GrowScratch(data.n_features, data.n_bins)
    return grow_tree_leaves_profiled(
        profile,
        leaves,
        ledger,
        scratch,
        data,
        grad,
        hess,
        params,
        bag,
        tree_index,
        bundling,
        const_hessian,
    )


def grow_tree_leaves(
    mut leaves: LeafMembership,
    mut ledger: CegbLedger,
    mut scratch: GrowScratch,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
    bundling: BundledMatrix = BundledMatrix.none(),
    const_hessian: Bool = False,
) raises -> Tree:
    """`grow_tree_leaves_profiled` with a profile of its own, reported per
    tree.

    The entry point a caller takes when it wants the leaf membership back and
    a scratch that outlives the tree, but does not accumulate a profile
    across the fit. With `MOJOTREES_PHASE_PROFILE` unset -- the default --
    the profile is off, this costs one `getenv` and one bucket allocation per
    tree, and prints nothing; with it set, one `scope=tree` block is printed
    per tree.

    `const_hessian` is forwarded unchanged; see `grow_tree_leaves_profiled`.
    """
    var profile = PhaseProfile.from_env(SCOPE_TREE, String("grow_tree"))
    var tree = grow_tree_leaves_profiled(
        profile,
        leaves,
        ledger,
        scratch,
        data,
        grad,
        hess,
        params,
        bag,
        tree_index,
        bundling,
        const_hessian,
    )
    profile.print_report()
    return tree^


def grow_tree_leaves_profiled(
    mut profile: PhaseProfile,
    mut leaves: LeafMembership,
    mut ledger: CegbLedger,
    mut scratch: GrowScratch,
    data: BinnedMatrix,
    grad: List[Float64],
    hess: List[Float64],
    params: TreeParams,
    bag: List[Int] = [],
    tree_index: Int = 0,
    bundling: BundledMatrix = BundledMatrix.none(),
    const_hessian: Bool = False,
    const_hessian_env: ConstHessianSettings = ConstHessianSettings.unresolved(),
) raises -> Tree:
    """Grow one tree, leaf-wise by default or depth-wise under
    `params.grow_policy == GROW_DEPTHWISE`.

    `const_hessian_env` is the fit's constant-hessian snapshot
    (`histogram.ConstHessianSettings`), the counterpart of the dispatch
    snapshot `scratch.settings` carries. It is threaded into every histogram
    build and every sibling subtraction below, so a resolved value makes a
    whole tree of nodes read `MOJOTREES_CONST_HESSIAN` and
    `MOJOTREES_CONST_HESSIAN_VERIFY` exactly zero times. The default sentinel
    is resolved **once here**, at the top of the tree, rather than being
    passed down as the sentinel: an unwired caller then pays two `getenv`
    calls per tree instead of two per node, which is the behaviour every
    caller that has not been wired gets and is already a strict improvement on
    the per-node reads. A boosting loop that holds one across the fit passes
    it and pays them once per fit; `parallel.DispatchSettings` states the same
    staging argument for its own `resolved` flag.

    `profile` is the phase and node-size attribution instrument
    (phase_profile.mojo), and it is off unless `MOJOTREES_PHASE_PROFILE` says
    otherwise. An off profile is a Bool test at each of the charge sites below
    and nothing else: no clock is read and no counter is written, so this
    grower's arithmetic, allocation, and iteration order are identical whether
    it is on or off, and so is the tree. What the charges cover is the
    accumulate, the buffer, the subtraction, the partition, and the scan; what
    they cannot cover is documented at the sites that could not reach it.

    `leaves` is filled with the finished frontier's leaf membership (see
    `LeafMembership`) and its previous contents are dropped. Growth partitions
    rows into the children of every split it takes, so at the end the frontier
    holds each leaf's rows already; handing them back costs a move per leaf
    and no copy. A trainer that has them does not have to walk the tree once
    per row to find out where a row landed.

    `scratch` is the caller's histogram pool and gather buffer. Passing the
    same one to every tree of a fit is the point of the argument: see
    `GrowScratch` for why sharing it changes nothing about the trees. The
    `grow_tree`, `grow_tree_with_cegb`, and `grow_tree_profiled` entry points
    construct a fresh one per call, which is what growth did before.

    All three, and `ledger` beside them, are leading `mut` arguments because
    Mojo cannot default a `mut` parameter, so they cannot sit beside `bag`
    and `bundling` at the end. That is also why the grower has four entry
    points rather than one with optional arguments.

    `ledger` is the ensemble's CEGB ledger (cegb.mojo), read once per node to
    cost that node's features and written once per chosen split. It spans the
    whole ensemble rather than one tree, which is CEGB's premise: the model
    pays for a feature once, so a later tree reusing it gets it free. Pass
    `CegbLedger.none()` (or call `grow_tree`) when no CEGB penalty needing a
    ledger is configured.

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

    `params.extra` carries the remaining LightGBM tree controls. This grower
    is the one that applies all of them: the split-side rules go down through
    `_search`, and the leaf-side ones (`max_delta_step`, `path_smooth`) are
    applied here, where a leaf's row count and its parent's finished output
    are both in hand. An inactive bundle, which is the default, leaves every
    decision below exactly as it was.

    `params.extra.forced` is LightGBM's `forcedsplits_filename`, parsed and
    mapped to bins (see `binning.map_forced_splits`). Its nodes are applied
    before any gain-chosen split, in forced-node order, through the same loop
    body: the frontier leaf that owes a forced node is split on that node's
    feature and threshold instead of on its own best candidate, its children
    inherit that node's children, and once the forced tree is exhausted
    leaf-wise growth resumes from the frontier it left behind. A forced node
    records a gain of 0.0, because none was chosen, and routes missing rows
    right, because the document does not say. `ExtraTreeParams.check` has
    already confirmed the forced tree fits inside `num_leaves` and
    `max_depth`, so the loop cannot run out of budget half way through it.

    `bundling` is an exclusive-feature-bundling plan and its matrix (efb.mojo),
    or `BundledMatrix.none()`, which is the default and the fallback. When it
    is active this grower reads **two** matrices and keeps them strictly
    apart:

    - `bundling.data` is what every histogram is accumulated from, which is
      the whole point: one column scan per bundle instead of one per feature.
      The result is expanded back to one slice per original feature
      immediately (`_hist_full`), so nothing past that point sees a bundle;
    - `data`, the original matrix, is what rows are partitioned by, because a
      chosen split names an original feature and an original bin.

    Nothing else changes. Feature subsampling still draws original features
    and `columns_for_features` maps them to the columns that must be
    accumulated; `_search` still receives a per-feature histogram, the
    original `missing_bin` table, the original categorical spec, and the
    original monotone vector; sibling subtraction still works, because the
    expansion is linear; and the `Tree` that comes out is the tree an
    unbundled fit produces, so no consumer of it can tell which matrix built
    the histograms.

    The growth policy decides one thing in this loop: which frontier leaf is
    split next. `GrowthSchedule` (growth_policy.mojo) takes the best gain
    anywhere in the tree under leaf-wise growth, and under depth-wise growth
    plans a depth at a time, admits its positive-gain splits against the
    leaf budget, and hands them out in ascending node id order; the body
    below (partition, sibling subtraction, child values, monotone intervals,
    child search) is the same code either way, so a depth-wise tree enforces
    every constraint exactly as a leaf-wise one does. Forced splits still go
    first in both modes.

    `GROW_OBLIVIOUS` is the exception, and it leaves this loop entirely.
    CatBoost's symmetric tree searches ONE split across a whole level and
    applies it to every leaf of that level, so there is no per-leaf best split
    to pick between and nothing for `GrowthSchedule` to order. It runs in
    `_grow_oblivious_levels` below, which this function hands the root
    frontier to and returns from before the loop; everything before that
    point -- validation, feature selection, the root histogram and value -- is
    shared, and the loop below is untouched. The tree that comes out is an
    ordinary `Tree` with the same split repeated across a level, so predict,
    dump, serialization, model I/O, monotone constraints and interaction
    constraints all work on it without knowing the mode existed. `max_depth`
    binds and is required; `num_leaves` does not bind and is ignored; a
    categorical matrix, forced splits, `extra_trees` and the ledger-reading
    CEGB penalties are refused (`_check_oblivious`).

    `const_hessian` is the caller's declaration that every entry of `hess` is
    exactly `histogram.CONSTANT_HESSIAN`, which lets the histogram builders
    accumulate two planes instead of three and reconstruct the hessian plane
    from the count (see histogram.mojo's module docstring for the exactness
    argument). It is threaded from here to every builder this grower calls and
    to both sibling subtractions, and it is a *declaration*, not an inference:

    - The predicate that answers it is
      `histogram.objective_has_constant_hessian(objective, weighted)`, and this
      grower is handed neither an objective code nor a weight vector, which is
      exactly why it cannot evaluate it. It is `boosting._boost_rounds` and
      `boosting.train_with_valid` that do, next to where they decide whether to
      pass weights.
    - GOSS is the case that makes the argument rather than the weights. A GOSS
      round rescales the sampled small-gradient rows' hessians
      (`goss.apply_goss_scaling`) and leaves the objective code untouched, so a
      squared-error GOSS round has two hessian values and a predicate reading
      only the objective would say the wrong thing. The trainer excludes it.
    - Declaring it for a `hess` that is not constant produces a wrong hessian
      plane silently, with no error and no diagnostic other than
      `MOJOTREES_CONST_HESSIAN_VERIFY=1`.

    Its default is False, so every grower call in this package that does not
    pass it is on the path that shipped. When it is true and correct the tree
    that comes out is byte for byte the tree the three-plane path grows, which
    is the only claim this argument makes: it is an accounting change in the
    accumulation loop and not an approximation.

    Passing it to `subtract_histogram_into` is a separate declaration about
    *two finished histograms* rather than about a row's hessian, and it is
    sound here for an inductive reason worth writing down. Under a true
    declaration every histogram this grower produces holds `Float64(count)` in
    its hessian plane: a directly built one does by the builder's own
    reconstruction, a bundled one does because `efb.expand_bundled_histogram`
    only copies cells and recovers a default bin by subtracting exact integers
    held in Float64, and a derived sibling does because the elided subtraction
    writes `Float64(parent_count - child_count)`. The root establishes the
    property and every node below inherits it, so both operands of every
    subtraction here satisfy what `subtract_histogram_into` asks. Note also
    that the two arms of that function agree even when the declaration is only
    passed to one of them, because `Float64(a) - Float64(b)` and
    `Float64(a - b)` are the same Float64 for integers below 2^53; the elision
    there is a traffic decision and cannot move a bit on its own.
    """
    check_grow_policy(params.grow_policy)
    if params.grow_policy == GROW_OBLIVIOUS:
        _check_oblivious(params, data)
    params.constraints.check_features(data.n_features)
    params.monotone.check_features(data.n_features)
    # CatBoost's one-hot threshold, refused here rather than at the first
    # categorical node. Off by default, so this cannot fire on a fit that
    # never set it.
    params.cat.check_one_hot()
    check_feature_fractions(
        params.feature_fraction,
        params.feature_fraction_bynode,
        params.feature_fraction_bylevel,
    )
    # Rejected before the first histogram rather than part way down a tree,
    # and against this dataset, so a per-feature vector of the wrong length is
    # named here rather than read past its end.
    params.extra.check(
        data.n_features,
        params.num_leaves,
        params.max_depth,
        params.min_data_in_leaf,
    )
    # Whether this call can honor the CEGB penalties that read the ledger,
    # asked once per tree rather than at the first node, so a trainer that did
    # not thread one is told before any histogram is built. `carries_cegb` is
    # also what `_search` is given, so the refusal below and the one inside it
    # are the same test on the same state.
    var cegb_config = params.extra.penalties.cegb.copy()
    var carries_cegb = ledger.is_tracking()
    check_cegb_grower_support(cegb_config, carries_cegb, carries_cegb)
    # A plan is fitted from one matrix and is meaningless against another, so
    # the shapes are checked before the first histogram rather than showing up
    # as a bin id that means the wrong feature.
    if bundling.active:
        if bundling.plan.n_features != data.n_features:
            raise Error(
                "bundling plan and matrix must agree on n_features"
            )
        if bundling.plan.n_rows != data.n_rows:
            raise Error("bundling plan and matrix must agree on n_rows")
        if bundling.data.n_rows != data.n_rows:
            raise Error("bundled matrix must have the same rows")
        if bundling.data.n_features != bundling.plan.n_bundles():
            raise Error("bundled matrix must have one column per bundle")
    var max_delta_step = params.extra.max_delta_step
    var path_smooth = params.extra.path_smooth
    # Empty unless a feature is actually constrained, which keeps split search
    # on its unconstrained path and the fit bit-identical.
    var signs = params.monotone.active_signs()
    # Whether any node's branch feature set is ever read. It is read only by
    # `InteractionConstraints.allowed_features`, which answers "every feature"
    # -- the empty mask -- for any branch when no groups are configured, so
    # with no constraints the sets stay empty and no branch list is built.
    var constrained = not params.constraints.is_empty()
    # The two constant-hessian environment answers, resolved at most once for
    # this tree instead of once per histogram build and once per subtraction.
    # A squared-error round declares a constant hessian, so without this every
    # node pays two `getenv` calls and two `String` allocations to re-derive a
    # decision that was fixed before the fit started; see
    # `histogram.ConstHessianSettings`. A caller that carries one across the
    # fit passes it and nothing here reads the environment at all.
    var const_h_env = const_hessian_env.copy()
    if not const_h_env.resolved:
        const_h_env = ConstHessianSettings.resolve()
    # The `derivative_precision` **parameter**, folded onto whichever
    # snapshot we ended up with. This is the hop that makes the parameter and
    # `MOJOTREES_DERIVATIVE_PRECISION` the same switch instead of two, and it
    # sits here rather than at the trainers because this is the single point
    # every dense CPU grower passes through: `boosting`, `alternate_boosting`
    # (DART and GOSS-boosting), `boosting_rf`, `custom_metric`, `ranking`,
    # `ranking_advanced` and `objective` all reach their histograms from
    # below this line. Folding at each trainer instead would be seven edits
    # that a new trainer could forget; folding here, a trainer cannot.
    #
    # `float64` from either entry wins -- see
    # `histogram.ConstHessianSettings.widened` for why that direction and not
    # the other. It applies on both arms above, the passed snapshot and the
    # per-tree fallback, because a trainer that resolved one for the whole
    # fit did not thereby decline the parameter.
    #
    # This is the histogram read side only. The objective side
    # (`boosting.fill_grad_hess`) still reads the environment and takes no
    # parameter, which is why `ExtraTreeParams.check_derivative_precision`
    # still refuses `float64` set through the parameters: the two halves have
    # to move together or the fit is a third thing that is neither arm.
    const_h_env = const_h_env.widened(
        params.extra.wants_float64_derivatives()
    )
    # `usable` is the pool the fraction draws from, and every caller of
    # `select_tree_features` took its `[]` default, so the parameter existed
    # and no fit ever passed one. `BinnedMatrix.usable_features` says it is
    # every feature unless the fit prefiltered and that passing every feature
    # is the same draw as passing nothing, so this changes no fit today; what
    # it changes is that a prefilter can now reach the sampler at all.
    # `feature_pre_filter` was half-wired for exactly this reason.
    var tree_features = select_tree_features(
        data.n_features,
        params.feature_fraction,
        params.feature_fraction_seed,
        tree_index,
        data.usable_features(),
    )
    # The histogram columns the tree's feature sample requires: the features
    # themselves without bundling, the bundles they sit in with it. A bundle
    # is accumulated when any of its members was sampled; the members that
    # were not ride along in that column and are simply never scanned, which
    # is safe because the expansion recovers a member exactly whatever else
    # shares its column.
    var tree_columns = tree_features.copy()
    if bundling.active:
        tree_columns = columns_for_features(bundling.plan, tree_features)
    # Leaf-value totals must come from a feature the histograms accumulated,
    # which under bundling means one whose slice the expansion wrote.
    var value_feature = tree_features[0]
    # Whether any node needs a feature draw of its own at all.
    #
    # With both inner fractions at 1.0 -- the default -- `select_split_features`
    # copies `tree_features` twice, once for the level draw and once for the
    # node draw, and hands back a list the caller is already holding. When the
    # tree's own set is *every* feature, the empty list means exactly that same
    # set to all three consumers downstream (`_search` and so `find_best_split`,
    # `prepare_cegb_node`, and the profile's cell count, which is resolved
    # through `_scan_cells`), so the two copies are removed by not making them.
    #
    # No bit moves: `find_best_split` reads feature `i_feature` directly under
    # its `use_all` arm and reads `features[i_feature]`, which is `i_feature`
    # for an ascending complete list, otherwise; the scan order, the active
    # count and the tie-breaking are the same either way. `select_tree_features`
    # returns an ascending list without repeats, so `len(tree_features) ==
    # data.n_features` is exactly "the tree may split on every feature".
    #
    # `check_feature_fractions` above has already validated all three fractions
    # for this tree, so skipping the per-node re-validation cannot let a bad
    # value through.
    var per_node_draw = (
        params.feature_fraction_bylevel < 1.0
        or params.feature_fraction_bynode < 1.0
        or len(tree_features) != data.n_features
    )
    var tree = Tree(
        List[Int](), List[Int](), List[Int](), List[Int](),
        List[Float64](), List[Float64](), 0,
    )
    # A leaf-wise tree stops at `num_leaves` leaves, and a binary tree with L
    # leaves has 2L - 1 nodes, so this is an exact upper bound on what
    # `_add_node` will append. One reservation per tree in place of ten
    # independent doubling sequences; see `Tree.reserve_nodes`.
    tree.reserve_nodes(2 * params.num_leaves - 1)

    # The tree's own root row count is the denominator every node size class
    # in this tree is taken against (phase_profile.mojo). Under bagging or
    # GOSS the root holds the sample and not the dataset, and classifying a
    # sampled tree against the dataset would push every node of it down a
    # class or two and make two sampling fractions incomparable.
    var n_root = len(bag) if len(bag) > 0 else data.n_rows
    var hist_cells = data.n_features * data.n_bins
    var tree_started = profile.clock()
    profile.begin_tree(n_root, data.n_rows)

    # Every histogram this tree builds comes from the caller's pool and goes
    # back to it when its leaf is split, so growth allocates a handful of
    # buffers rather than three arrays per node -- and, when the caller keeps
    # one scratch for a whole fit, a handful per fit rather than per tree. The
    # pool's shape is per feature whether or not bundling is on: a bundled
    # accumulation lands in `bundle_scratch` first and is expanded into a
    # pooled buffer, so sibling subtraction, leaf values, and split search all
    # read the shape they always read.
    scratch.prepare(data.n_features, data.n_bins)
    var bundle_scratch = Histogram.zeroed(0, 0)
    if bundling.active:
        bundle_scratch = Histogram.zeroed(
            bundling.data.n_features, bundling.data.n_bins
        )

    # The root's row list is the only thing bagging materializes; the full
    # path builds the same list over every row.
    var root_rows: List[Int]
    # `take` is a pop off the free list once the pool has warmed, and a whole
    # `Histogram.zeroed` allocation before that: three arrays of
    # `n_features * n_bins`, which is where a per-node allocation cost would
    # live if the pool were not doing its job. Charged per cell rather than
    # per row, because it is the same work at any node size. A caller-owned
    # pool warms once for a whole fit rather than once per tree, so this line
    # is a pop from the first node of the second tree onwards.
    var root_alloc_started = profile.clock()
    var root_hist = scratch.pool.take()
    if bundling.active:
        # A pooled buffer's contents are undefined and the expansion writes
        # only the sampled features' slices, so the excluded ones are zeroed
        # here rather than left holding another node's statistics. The
        # unbundled builders zero each slice themselves and need none of this.
        root_hist.reset()
    profile.charge(
        PROF_HIST_ALLOC, n_root, root_alloc_started, cells=hist_cells
    )
    if len(bag) == 0:
        var root_list_started = profile.clock()
        root_rows = List[Int]()
        fill_identity_rows(root_rows, data.n_rows, scratch.settings)
        # The identity permutation is row-list construction, which is the same
        # kind of work a split's two child lists are, so it goes to the same
        # phase rather than disappearing into the unattributed remainder.
        profile.charge(
            PROF_PARTITION,
            n_root,
            root_list_started,
            dispatches=HOST_PARTITION_DISPATCHES,
        )
        var root_hist_started = profile.clock()
        _hist_full(
            root_hist, bundle_scratch, scratch.pairs, data, bundling, grad,
            hess, tree_features, tree_columns, const_hessian,
            scratch.settings, const_h_env,
        )
        profile.note_node()
        profile.charge(
            PROF_HISTOGRAM,
            n_root,
            root_hist_started,
            dispatches=HOST_HIST_DISPATCHES,
            slots_per_row=len(tree_columns),
            cells=hist_cells,
        )
    else:
        # `sampling.check_row_set` is the one place this property is enforced
        # rather than assumed, and everything downstream of the draw -- the
        # subset accumulate, the partition, the node counts -- relies on it.
        check_row_set(bag, data.n_rows)
        var root_list_started = profile.clock()
        root_rows = bag.copy()
        profile.charge(
            PROF_PARTITION,
            n_root,
            root_list_started,
            dispatches=HOST_PARTITION_DISPATCHES,
        )
        var root_hist_started = profile.clock()
        _hist_subset(
            root_hist, bundle_scratch, scratch.pairs, data, bundling, grad,
            hess, bag, 0, len(bag), tree_features, tree_columns,
            const_hessian, scratch.settings, const_h_env, scratch.bin_layout,
        )
        profile.note_node()
        profile.charge(
            PROF_HISTOGRAM,
            n_root,
            root_hist_started,
            dispatches=HOST_HIST_DISPATCHES,
            slots_per_row=len(tree_columns),
            cells=hist_cells,
        )

    # The root's own value is computed before its split search, because path
    # smoothing makes a candidate's children shrink toward it: it is the
    # `parent_output` every candidate at the root is scored with. The root
    # itself has no parent, so it smooths toward 0.0.
    var root = tree._add_node(
        _leaf_value(
            root_hist,
            params.lambda_reg,
            params.lambda_l1,
            value_feature,
            len(root_rows),
            0.0,
            max_delta_step,
            path_smooth,
        ),
        Float64(len(root_rows)),
    )

    # The root's branch is empty, so its allow mask is the union of every
    # configured group (empty, meaning all features, when unconstrained). The
    # root is always node 0, so its per-level and per-node feature draws are
    # fixed too.
    var root_branch = List[Int]()
    # Hoisted out of the `_search` call because the node's costs must be
    # prepared over exactly the feature set the scan will look at: costing
    # more would walk rows for features that are never scanned, and costing
    # fewer makes `CegbNodeCosts.delta_of` raise for a feature the scan asks
    # about.
    var root_features = List[Int]()
    if per_node_draw:
        root_features = select_split_features(
            tree_features,
            params.feature_fraction_bylevel,
            params.feature_fraction_bynode,
            params.feature_fraction_seed,
            tree_index,
            0,
            0,
        )
    var root_costs = prepare_cegb_node(
        cegb_config,
        ledger,
        data.n_features,
        len(root_rows),
        root_rows,
        root_features,
    )
    var root_search_started = profile.clock()
    var root_split = _search(
        root_hist,
        len(root_rows),
        params,
        params.constraints.allowed_features(root_branch),
        root_features,
        depth=0,
        missing_bins=data.missing_bin,
        monotone=signs,
        cats=data.cats,
        node=root,
        tree_index=tree_index,
        parent_output=tree.value[root],
        grower_applies_extra=True,
        cegb=root_costs,
        grower_applies_cegb=carries_cegb,
        settings=scratch.settings,
    )
    # The scan reads one bin at a time over the node's own feature draw, so
    # its cells are that draw's width times `n_bins` and not the buffer's full
    # shape. That distinction is the whole reason `cells` is recorded per
    # charge rather than derived from the dataset. `_scan_cells` resolves the
    # undrawn case, where the empty list means every feature.
    profile.charge(
        PROF_SPLIT_SEARCH,
        len(root_rows),
        root_search_started,
        dispatches=HOST_SPLIT_SEARCH_DISPATCHES,
        cells=_scan_cells(root_features, data.n_features, data.n_bins),
    )

    var frontier = List[_LeafState]()
    frontier.append(
        _LeafState(
            root,
            root_rows^,
            root_hist^,
            root_split^,
            root_branch^,
            depth=0,
            forced=0 if not params.extra.forced.is_empty() else -1,
        )
    )
    var n_leaves = 1

    if params.grow_policy == GROW_OBLIVIOUS:
        # Oblivious growth is not a frontier order (growth_policy.mojo), so it
        # builds no `GrowthSchedule` -- the constructor refuses the code -- and
        # never enters the loop below. Everything ABOVE this line is shared:
        # the validation, the feature selection, the root's rows, histogram
        # and value, and the frontier that holds it. Everything BELOW it is a
        # per-leaf split loop that has no meaning when a level takes one
        # split. Returning from here rather than branching inside that loop is
        # what leaves the default path the path that shipped, instruction for
        # instruction, rather than a path with a policy test in it.
        #
        # The root's own `_search` above is redundant in this mode: a level of
        # one leaf is searched again by `find_best_split_shared`, which for
        # L = 1 is `find_best_split` with the illegal-leaf rule that cannot
        # change a single-leaf answer. That is one extra feature dispatch per
        # tree, against `max_depth` shared searches; removing it means putting
        # a policy test around the shared root block, which is a worse trade.
        n_leaves = _grow_oblivious_levels(
            profile,
            scratch,
            tree,
            frontier,
            bundle_scratch,
            data,
            grad,
            hess,
            params,
            tree_features,
            tree_columns,
            value_feature,
            per_node_draw,
            signs,
            constrained,
            bundling,
            const_hessian,
            const_h_env,
            hist_cells,
            tree_index,
        )
        tree.n_leaves = n_leaves
        leaves.clear()
        leaves.covers_all_rows = len(bag) == 0
        leaves.node = List[Int](capacity=len(frontier))
        leaves.rows = List[List[Int]](capacity=len(frontier))
        for i in range(len(frontier)):
            leaves.node.append(frontier[i].node)
            leaves.rows.append(frontier[i].take_rows())
            scratch.pool.give(frontier[i].take_hist())
        profile.end_tree(tree_started)
        return tree^

    var schedule = GrowthSchedule(params.grow_policy)

    while n_leaves < params.num_leaves:
        # Forced splits go first, in forced-node order. A parent always
        # precedes its children in the forced tree, so ascending order is a
        # valid application order, and `check_budget` has already guaranteed
        # the whole forced tree fits inside num_leaves and max_depth, so this
        # loop cannot run out of budget part way through it.
        var best_i = -1
        var forced_node = -1
        for i in range(len(frontier)):
            if frontier[i].forced >= 0 and (
                forced_node < 0 or frontier[i].forced < forced_node
            ):
                forced_node = frontier[i].forced
                best_i = i
        if best_i < 0:
            # Nothing forced: the growth policy picks (growth_policy.mojo),
            # best gain anywhere in the tree under leaf-wise growth, the
            # planned level's next node under depth-wise growth.
            var cands = List[LeafCandidate](capacity=len(frontier))
            for i in range(len(frontier)):
                cands.append(
                    LeafCandidate(
                        frontier[i].node,
                        frontier[i].depth,
                        frontier[i].split.gain,
                        frontier[i].split.found
                        and frontier[i].split.gain > 0.0,
                    )
                )
            best_i = schedule.next_leaf(
                cands, n_leaves, params.num_leaves, params.max_depth
            )
            if best_i < 0:
                break

        var parent_node = frontier[best_i].node
        var split: SplitInfo
        var left_forced = -1
        var right_forced = -1
        if forced_node >= 0:
            # A forced node's split is the caller's, not the histogram's: it
            # is applied whatever its gain, so it carries a gain of 0.0 (there
            # is no chosen-by-gain value to record, and `gain_importance` will
            # read it as the zero contribution it is). Missing rows go right,
            # because the forced-split document says nothing about them and no
            # scan chose a direction.
            var forced_spec = params.extra.forced.nodes[forced_node].copy()
            split = SplitInfo(
                forced_spec.feature,
                params.extra.forced.bin_at(forced_node),
                0.0,
                True,
                False,
            )
            left_forced = forced_spec.left
            right_forced = forced_spec.right
        else:
            split = frontier[best_i].split.copy()

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
        # Charged at the *parent's* row count, because that is the work: the
        # routing walks the parent's list once and the two child lists it
        # allocates hold exactly those rows between them. Filing it under the
        # children would split one cost across two classes and make the
        # per-row column mean nothing.
        var part_rows = len(frontier[best_i].rows)
        var part_started = profile.clock()
        partition_rows_into(
            left_rows,
            right_rows,
            data,
            frontier[best_i].rows,
            split,
            split_missing_bin,
            scratch.settings,
        )
        profile.charge(
            PROF_PARTITION,
            part_rows,
            part_started,
            dispatches=HOST_PARTITION_DISPATCHES,
        )
        if forced_node >= 0 and (
            len(left_rows) == 0 or len(right_rows) == 0
        ):
            # A gain-chosen split cannot do this: `min_data_in_leaf` and
            # `min_child_hess` reject an empty side before it is ever
            # selected. A forced one can, because it is applied whatever the
            # data does, and an empty child would give a leaf with no rows to
            # value. Naming it is the only honest answer.
            raise Error(
                "forced split node ",
                forced_node,
                " on feature ",
                split.feature,
                " at bin ",
                split.bin,
                " leaves one child empty; the threshold falls outside the"
                " values this node's rows take",
            )

        # The parent's histogram is read once more, by the subtraction below,
        # and is dead after that; moving it out here is what lets its buffer
        # go straight back to the pool.
        var parent_hist = frontier[best_i].take_hist()

        # Histogram subtraction trick: build the smaller child directly. The
        # expansion is linear, so expanding both and subtracting gives what
        # subtracting first and expanding would have: the derived sibling is
        # correct under bundling with no change here.
        # Two buffers per split, and the zeroing pass when bundling needs one.
        # Charged twice at `hist_cells` each, because that is two buffers'
        # worth of allocation and fault-in when the pool is cold, and filed at
        # the built child's size so a class's `hist_alloc` line can be read
        # beside its `histogram` line.
        var builds_left = len(left_rows) <= len(right_rows)
        var built_rows = len(left_rows) if builds_left else len(right_rows)
        var derived_rows = len(right_rows) if builds_left else len(left_rows)
        # The layout probe, offered once per fit on the first child this
        # grower builds directly. `scratch.resolve_layout_timed` declines and
        # clears itself when the fit named a layout, when there is no
        # row-major view, or when the node does not block, so on today's
        # defaults with no view built this is one Bool test per split.
        #
        # Deliberately outside every `profile.clock()` window below: the
        # probe is not this node's histogram and charging it to PROF_HISTOGRAM
        # would put a once-per-fit cost inside a per-node rate. It lands in
        # the profile's unattributed remainder, where it is visible as itself.
        #
        # The bundled arm is not offered it. `efb` assembles `bundling.data`
        # and never builds a row-major view on it, so the probe would time
        # two arms that are the same arm, and `_hist_subset` sends the
        # bundled build to `bundling.data` regardless of what was pinned.
        #
        # Two statements rather than one conditional expression on the row
        # list: `left_rows if builds_left else right_rows` in an argument
        # position materializes a `List[Int]` of the child's rows, which at a
        # large node is hundreds of thousands of elements copied to choose
        # between two names.
        if not bundling.active:
            if builds_left:
                scratch.resolve_layout_timed(
                    data, grad, hess, left_rows, 0, built_rows,
                    tree_features, const_hessian, const_h_env,
                )
            else:
                scratch.resolve_layout_timed(
                    data, grad, hess, right_rows, 0, built_rows,
                    tree_features, const_hessian, const_h_env,
                )
        var alloc_started = profile.clock()
        var left_hist = scratch.pool.take()
        var right_hist = scratch.pool.take()
        if bundling.active:
            # Only the directly built child needs zeroing; the derived one is
            # fully written by the subtraction.
            if len(left_rows) <= len(right_rows):
                left_hist.reset()
            else:
                right_hist.reset()
        profile.charge(
            PROF_HIST_ALLOC, built_rows, alloc_started, cells=2 * hist_cells
        )
        var hist_started = profile.clock()
        if len(left_rows) <= len(right_rows):
            _hist_subset(
                left_hist, bundle_scratch, scratch.pairs, data, bundling,
                grad, hess, left_rows, 0, len(left_rows), tree_features,
                tree_columns, const_hessian, scratch.settings,
                const_h_env, scratch.bin_layout,
            )
            profile.note_node()
            profile.charge(
                PROF_HISTOGRAM,
                built_rows,
                hist_started,
                dispatches=HOST_HIST_DISPATCHES,
                slots_per_row=len(tree_columns),
                cells=hist_cells,
            )
            var sub_started = profile.clock()
            subtract_histogram_into(
                right_hist, parent_hist, left_hist, const_hessian,
                scratch.settings, const_h_env,
            )
            # Filed at the *derived* child's size: it is that child's
            # histogram that comes out, and the point of the trick is that a
            # large sibling costs a per-cell subtraction instead of a per-row
            # accumulate. Reading the two lines side by side is what shows
            # whether that trade is still winning at each size.
            profile.note_node()
            profile.charge(
                PROF_SUBTRACT,
                derived_rows,
                sub_started,
                dispatches=HOST_SUBTRACT_DISPATCHES,
                cells=hist_cells,
            )
        else:
            _hist_subset(
                right_hist, bundle_scratch, scratch.pairs, data, bundling,
                grad, hess, right_rows, 0, len(right_rows), tree_features,
                tree_columns, const_hessian, scratch.settings,
                const_h_env, scratch.bin_layout,
            )
            profile.note_node()
            profile.charge(
                PROF_HISTOGRAM,
                built_rows,
                hist_started,
                dispatches=HOST_HIST_DISPATCHES,
                slots_per_row=len(tree_columns),
                cells=hist_cells,
            )
            var sub_started = profile.clock()
            subtract_histogram_into(
                left_hist, parent_hist, right_hist, const_hessian,
                scratch.settings, const_h_env,
            )
            profile.note_node()
            profile.charge(
                PROF_SUBTRACT,
                derived_rows,
                sub_started,
                dispatches=HOST_SUBTRACT_DISPATCHES,
                cells=hist_cells,
            )
        scratch.pool.give(parent_hist^)

        # Child values are clamped into the parent's interval, then the
        # interval is divided between the children at their midpoint. Both are
        # no-ops when unconstrained: the interval is unbounded and the split
        # sign is 0, so the values and the children's intervals are the
        # parent's unchanged.
        var parent_bounds = frontier[best_i].bounds.copy()
        var split_sign = monotone_sign(signs, split.feature)
        # The cap and the smoothing are applied to the Newton step first and
        # the monotone interval is enforced on the result, which is the order
        # the candidate was scored with (see `split._split_gain`). Both
        # children smooth toward the value the parent already emits.
        var parent_output = tree.value[parent_node]
        var left_value = parent_bounds.clamp(
            _leaf_value(
                left_hist,
                params.lambda_reg,
                params.lambda_l1,
                value_feature,
                len(left_rows),
                parent_output,
                max_delta_step,
                path_smooth,
            )
        )
        var right_value = parent_bounds.clamp(
            _leaf_value(
                right_hist,
                params.lambda_reg,
                params.lambda_l1,
                value_feature,
                len(right_rows),
                parent_output,
                max_delta_step,
                path_smooth,
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

        # The CEGB ledger is written here and nowhere else: after the parent's
        # own split is recorded and before either child is searched, so both
        # children see the ledger this split just updated. That is LightGBM's
        # order. A forced split commits too -- the model computes that feature
        # at prediction time whether a gain chose it or the caller did.
        #
        # `split.feature` is a dataset feature id rather than a bundle id,
        # because this grower expands a bundled histogram back to one slice
        # per original feature before searching it (`_hist_full`) and
        # partitions rows by the original matrix. A grower that searched
        # bundles directly would charge
        # `bundling.plan.charged_feature(split.feature, split.bin, ...)`.
        if carries_cegb:
            var commit = cegb_commit_split(
                ledger, cegb_config, split.feature, frontier[best_i].rows
            )
            if commit.feature_newly_used:
                # Every other leaf's cached candidate was scored while this
                # feature still owed its first-use cost. The model computes it
                # now whatever those leaves do, so the candidates that were
                # charged for it get exactly that charge back. The split and
                # lazy terms belong to their own nodes and a split over here
                # does not touch them.
                for i in range(len(frontier)):
                    if i == best_i:
                        continue
                    frontier[i].split.gain += cegb_stale_cached_gain(
                        commit, frontier[i].split.feature, split.feature
                    )

        # Both children inherit the same branch feature set, so they share one
        # allow mask, and both sit one edge below the leaf that was split.
        #
        # The branch set exists only to be read by
        # `InteractionConstraints.allowed_features`, which returns the empty
        # mask for every branch when no groups are configured. With no
        # constraints the set is therefore never read, and extending it would
        # copy a list per split for a value nothing consumes -- so it is left
        # empty, which is what the module docstring says the unconstrained
        # path does.
        var branch = List[Int]()
        if constrained:
            branch = extend_branch(frontier[best_i].branch, split.feature)
        var allowed = params.constraints.allowed_features(branch)
        var child_depth = frontier[best_i].depth + 1
        # Each child draws its own per-node feature set from its node id, out
        # of the set its depth drew from the tree's.
        var left_features = List[Int]()
        if per_node_draw:
            left_features = select_split_features(
                tree_features,
                params.feature_fraction_bylevel,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                child_depth,
                left_node,
            )
        var left_costs = prepare_cegb_node(
            cegb_config,
            ledger,
            data.n_features,
            len(left_rows),
            left_rows,
            left_features,
        )
        var left_search_started = profile.clock()
        var left_split = _search(
            left_hist,
            len(left_rows),
            params,
            allowed,
            left_features,
            depth=child_depth,
            missing_bins=data.missing_bin,
            monotone=signs,
            cats=data.cats,
            bounds=children.left,
            node=left_node,
            tree_index=tree_index,
            parent_output=left_value,
            grower_applies_extra=True,
            cegb=left_costs,
            grower_applies_cegb=carries_cegb,
            settings=scratch.settings,
        )
        profile.charge(
            PROF_SPLIT_SEARCH,
            len(left_rows),
            left_search_started,
            dispatches=HOST_SPLIT_SEARCH_DISPATCHES,
            cells=_scan_cells(left_features, data.n_features, data.n_bins),
        )
        var right_features = List[Int]()
        if per_node_draw:
            right_features = select_split_features(
                tree_features,
                params.feature_fraction_bylevel,
                params.feature_fraction_bynode,
                params.feature_fraction_seed,
                tree_index,
                child_depth,
                right_node,
            )
        var right_costs = prepare_cegb_node(
            cegb_config,
            ledger,
            data.n_features,
            len(right_rows),
            right_rows,
            right_features,
        )
        var right_search_started = profile.clock()
        var right_split = _search(
            right_hist,
            len(right_rows),
            params,
            allowed,
            right_features,
            depth=child_depth,
            missing_bins=data.missing_bin,
            monotone=signs,
            cats=data.cats,
            bounds=children.right,
            node=right_node,
            tree_index=tree_index,
            parent_output=right_value,
            grower_applies_extra=True,
            cegb=right_costs,
            grower_applies_cegb=carries_cegb,
            settings=scratch.settings,
        )
        profile.charge(
            PROF_SPLIT_SEARCH,
            len(right_rows),
            right_search_started,
            dispatches=HOST_SPLIT_SEARCH_DISPATCHES,
            cells=_scan_cells(right_features, data.n_features, data.n_bins),
        )

        frontier[best_i] = _LeafState(
            left_node,
            left_rows^,
            left_hist^,
            left_split^,
            branch.copy(),
            depth=child_depth,
            bounds=children.left.copy(),
            forced=left_forced,
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
                forced=right_forced,
            )
        )
        n_leaves += 1

    tree.n_leaves = n_leaves

    # The frontier is the leaf set: growth replaces a split leaf's state with
    # its left child and appends its right, so `len(frontier) == n_leaves` at
    # every step and nothing that is still a leaf has left it. Its row lists
    # are moved out rather than copied, and its histograms go back to the pool
    # here rather than being freed with the states, which is what lets a
    # booster-scoped scratch keep serving the next tree.
    leaves.clear()
    leaves.covers_all_rows = len(bag) == 0
    leaves.node = List[Int](capacity=len(frontier))
    leaves.rows = List[List[Int]](capacity=len(frontier))
    for i in range(len(frontier)):
        leaves.node.append(frontier[i].node)
        leaves.rows.append(frontier[i].take_rows())
        scratch.pool.give(frontier[i].take_hist())

    # The tree's whole wall clock, so the report can name what no phase
    # claimed rather than letting the reader assume the phases cover it. Taken
    # after the membership drain above so that the drain is inside the tree's
    # wall time rather than falling off the end of it. What is deliberately
    # outside every phase here: leaf valuation, the monotone clamp, the growth
    # policy's frontier scan, the CEGB bookkeeping, the per-node feature
    # draws, and that drain.
    profile.end_tree(tree_started)
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
