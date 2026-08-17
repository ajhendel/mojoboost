"""Specialized prediction paths that produce exactly the generic walker's
numbers.

Today this file holds one of them: the oblivious (symmetric) evaluation
CatBoost's inference is built on. An oblivious tree asks the same question at
every node of a level, so a row's path is `depth` independent comparisons
rather than a walk. The bits those comparisons produce, with level 0 in the
least significant bit and 0 meaning left, ARE the row's position among the
tree's leaves, so there is no traversal, no pointer chasing, and no
data-dependent load for structure. A depth-6 tree costs six loads of one bin
each and one load of a leaf value, against six dependent loads of a node
record plus six loads of a bin.

What this file is NOT is a second definition of what a tree means. The
per-level predicate here is `Tree.goes_left` rewritten against per-level
copies of the same five fields, in the same order, with the same boundary
condition, and the leaf table is built by walking the tree's own `left` and
`right` arrays. See `ObliviousEnsemble._goes_left` for the line-by-line
correspondence and `_split_matches` for what makes the substitution legal.

Bit-identity, which is the whole reason this is allowed to be on by default
with no switch behind it:

- Same leaf. The plan is built only for a tree that has been VERIFIED, node
  by node, to carry one split per level, so evaluating the level's shared
  predicate answers what `goes_left` would have answered at whichever node
  the row actually stood on. The sequence of directions therefore matches,
  and the sequence of directions is what picks the leaf.
- Same value. `leaf_value` holds `Tree.value` at that leaf, copied, not
  recomputed.
- Same sum. `raw_from_bins` walks the range's trees in ascending order and
  accumulates `s += learning_rate * value`, one tree at a time, which is
  `Booster.predict_raw_bins_range` statement for statement. The TREE axis is
  never reassociated and never split; the only axis anything here splits is
  the row axis, where there is no sum to reassociate.

So the plan is a pure speedup: every gate below (the depth cap, the row
floor, the structural check) decides only how fast the answer arrives, and
none of them can move a Float64. That is what lets `oblivious_plan` take
`n_rows` as an argument without breaking the rule that a scheduling decision
must never change a result.

Not covered, and each falls back to the generic walker rather than guessing:
linear leaves (`LinearEnsemble`, linear_tree.mojo), trees deeper than
`OBLIVIOUS_PLAN_MAX_DEPTH`, ragged trees, and multiclass ensembles.
`MulticlassBooster` is a shape question rather than a hard one -- the same
plan with a class stride -- and it is simply not built here.

Categorical splits ARE covered, and covered rather than excluded on purpose.
A categorical level is a set membership test instead of a threshold compare,
so the per-level operation differs, but a direction is still one bit and the
pack is unaffected; `_split_matches` compares the level's set word for word.
Today `tree._check_oblivious` refuses to GROW a symmetric tree on a matrix
that still offers a categorical column, so no model this library trains can
reach that branch. It is here because the guard is structural: a model
edited, refitted, or imported into this shape would reach it, and a guard
that quietly mis-routes such a model is worse than one that costs four word
compares per level at plan time.
"""

from std.math import isnan
from std.os import getenv
from std.sys.info import simd_width_of

from .binning import BinMapper, BinnedMatrix, POSITIVE_INF
from .boosting import Booster, IterationRange
from .categorical import CAT_BITSET_WORDS, cat_pool_contains
from .parallel import dispatch_rows
from .tree import Tree


# Why a raw plan was refused. Every refusal in this file records one of these
# and every plan carries the FIRST one it hit, because "the number did not
# move" is not a diagnosis: the raw arms below can decline silently, and on
# 2026-08-17 that survived a clean build, a `np.array_equal` bit-identity
# check on two growth policies, and two full benchmark runs, with the only
# symptom being one arm whose time did not change. A declined arm and an arm
# that ran and was no faster are opposite findings with opposite fixes. See
# `PredictTrace`.
comptime REFUSE_NONE = 0
comptime REFUSE_CATEGORICAL_SPLIT = 1
comptime REFUSE_FEATURE_OUT_OF_MATRIX = 2
comptime REFUSE_CATEGORICAL_FEATURE = 3
comptime REFUSE_NEGATIVE_THRESHOLD = 4
comptime REFUSE_MISSING_BIN_IN_RANGE = 5
comptime REFUSE_NODE_MISSING_BIN = 6
comptime REFUSE_NAN_EDGE = 7
comptime REFUSE_SWITCH_OFF = 10
comptime REFUSE_TOO_FEW_ROWS = 11
comptime REFUSE_EMPTY_RANGE = 12
comptime REFUSE_LINEAR_LEAVES = 13
comptime REFUSE_CTR_TABLES = 14
comptime REFUSE_EMPTY_TREE = 15
comptime REFUSE_BACKWARD_LINKS = 16
comptime REFUSE_NOT_OBLIVIOUS = 17
comptime REFUSE_TOO_MANY_NODES = 18


def _yn(b: Bool) -> String:
    """A Bool as "yes"/"no" for the trace.

    Explicit rather than printing the Bool, because nothing else in this
    package prints one and this lane could not build to check that `Bool` is
    `Writable`. A `String` is certain.
    """
    return String("yes") if b else String("no")


def refusal_text(code: Int) -> String:
    """The refusal code as the sentence a person needs.

    A code and not a `String` on the builder itself, because `_raw_split` runs
    once per NODE and a String there would allocate per node on a path whose
    whole point is to allocate nothing.
    """
    if code == REFUSE_NONE:
        return String("none")
    if code == REFUSE_CATEGORICAL_SPLIT:
        return String("categorical split: no threshold to convert")
    if code == REFUSE_FEATURE_OUT_OF_MATRIX:
        return String("split feature is not a column of the raw matrix")
    if code == REFUSE_CATEGORICAL_FEATURE:
        return String("split feature is categorical in the mapper")
    if code == REFUSE_NEGATIVE_THRESHOLD:
        return String("threshold_bin < 0 on a numerical node")
    if code == REFUSE_MISSING_BIN_IN_RANGE:
        return String("mapper missing bin is inside the ordinary bin range")
    if code == REFUSE_NODE_MISSING_BIN:
        return String("node missing bin disagrees with the mapper's")
    if code == REFUSE_NAN_EDGE:
        return String("split edge is NaN, so no comparison can order it")
    if code == REFUSE_SWITCH_OFF:
        return String("MOJOTREES_RAW_PREDICT=0")
    if code == REFUSE_TOO_FEW_ROWS:
        return String("fewer rows than RAW_MIN_ROWS")
    if code == REFUSE_EMPTY_RANGE:
        return String("empty iteration range")
    if code == REFUSE_LINEAR_LEAVES:
        return String("linear leaves")
    if code == REFUSE_CTR_TABLES:
        return String("CTR tables attached to the mapper")
    if code == REFUSE_EMPTY_TREE:
        return String("tree with no nodes")
    if code == REFUSE_BACKWARD_LINKS:
        return String("tree links do not point forward")
    if code == REFUSE_NOT_OBLIVIOUS:
        return String("no oblivious plan (ragged, too deep, or too few rows)")
    if code == REFUSE_TOO_MANY_NODES:
        return String("more than 2^31 - 1 nodes in the iteration range")
    return String("unknown")


# The deepest oblivious tree this file will plan for. Deliberately NOT
# `growth_policy.OBLIVIOUS_MAX_DEPTH`, which is 16 and is the deepest
# symmetric tree the grower will build; the two are different budgets on
# different things and giving them the same value would be a coincidence
# rather than a rule.
#
# The leaf table is `2 ** depth` Float64 per tree and costs
# `2 ** depth * depth` steps to build, so the plan's cost is exponential in
# the depth while the saving per row is only linear in it. Twelve is 4,096
# leaves and 32 KB per tree, which is 3.2 MB across a 100-tree CatBoost-mode
# ensemble; sixteen would be 512 KB per tree and 51 MB across the same
# ensemble, which is why the grower's cap is not this one. CatBoost's default
# depth is 6, where the table is 512 bytes per tree.
#
# **Corrected 2026-08-17: this said "a 360-tree CatBoost-mode ensemble" and
# 360 is not a tree count this library has ever shipped.** `n_estimators`
# defaults to 100 (`python/mojotrees/sklearn.py`), which is LightGBM's
# `num_iterations` and is gated against it by `tools/check_parity.py`, and
# CatBoost mode does NOT raise it: the mode supplies CatBoost's depth, rate,
# bootstrap, scoring and `l2_leaf_reg` and leaves the tree count alone. A
# 2026-08-16 decision to ship symmetric growth at 360 trees was recorded and
# never implemented, and 360 leaked out of that decision into arithmetic here
# as though it were the default. The figures above are recomputed at 100.
#
# A deeper symmetric model is not refused, it simply keeps the generic walker.
# Not measured. It is a budget on the plan, not a statement about where the
# crossover is.
comptime OBLIVIOUS_PLAN_MAX_DEPTH = 12

# Below this many rows the plan is not built at all. Building costs
# `n_trees * (nodes + 2 ** depth * depth)` once; each row then saves roughly
# `n_trees * depth` dependent loads. A one-row predict would pay the whole
# build to save one row's walk, which is a clear loss, and a batch of a few
# dozen rows is too close to call. Sixty-four is the smallest round number
# comfortably past "clear loss".
#
# Not measured either, and it cannot change an output: both arms compute the
# same Float64 (see the module docstring), so this number is free to move.
comptime OBLIVIOUS_MIN_ROWS = 64


def _split_matches(tree: Tree, a: Int, b: Int) -> Bool:
    """Whether internal nodes `a` and `b` route rows by the same rule.

    This is the predicate that makes the whole specialization legal: if it
    holds for every pair on a level, then one copy of the level's routing
    answers for every node on it, and `_goes_left` may be evaluated without
    knowing which node the row stood on.

    It compares every field `Tree.goes_left` reads, and nothing else. For a
    categorical node that means the SET, word for word, and not the offset:
    two nodes of an oblivious tree hold the same set at different offsets in
    the tree's flat `cat_bitset` pool, because `_set_split` appends a fresh
    copy per node. Comparing offsets would reject every oblivious categorical
    tree; comparing words is the question actually being asked.
    """
    if tree.feature[a] != tree.feature[b]:
        return False
    if tree.threshold_bin[a] != tree.threshold_bin[b]:
        return False
    if tree.missing_bin[a] != tree.missing_bin[b]:
        return False
    if tree.default_left[a] != tree.default_left[b]:
        return False
    var ca = tree.cat_offset[a]
    var cb = tree.cat_offset[b]
    if (ca >= 0) != (cb >= 0):
        return False
    if ca >= 0:
        for w in range(CAT_BITSET_WORDS):
            if tree.cat_bitset[ca + w] != tree.cat_bitset[cb + w]:
                return False
    return True


def _oblivious_depth(tree: Tree) -> Int:
    """`tree`'s depth when it is oblivious, and -1 when it is not.

    Structural, and deliberately so. Nothing in a `Tree` records the grow
    policy that produced it, and a flag copied from the training parameters
    would be a claim about how the model was made rather than about what it
    is; a refit, a model edit, or an imported LightGBM file could all falsify
    it. So this walks the tree once, level by level, and asks the model
    itself.

    A level passes when every node on it is internal and every node on it
    routes like the level's first node (`_split_matches`). The tree ends when
    a whole level is leaves; a level that is part leaves and part internal
    nodes is ragged, which is what a leaf-wise or depth-wise grower produces,
    and returns -1.

    Cost is one pass over the reachable nodes, which is the same order as
    walking one row, so it is cheap enough to run per prediction call. A
    single-leaf tree is depth 0 and is oblivious by inspection.
    """
    if len(tree.feature) == 0:
        return -1
    if tree.feature[0] < 0:
        return 0
    var level: List[Int] = [0]
    for d in range(OBLIVIOUS_PLAN_MAX_DEPTH):
        # `level` holds depth d's nodes on entry.
        for i in range(len(level)):
            if tree.feature[level[i]] < 0:
                return -1
            if not _split_matches(tree, level[0], level[i]):
                return -1
        var nxt = List[Int](capacity=2 * len(level))
        for i in range(len(level)):
            nxt.append(tree.left[level[i]])
            nxt.append(tree.right[level[i]])
        level = nxt^
        var leaves = 0
        for i in range(len(level)):
            if tree.feature[level[i]] < 0:
                leaves += 1
        if leaves == len(level):
            return d + 1
        if leaves != 0:
            return -1
    return -1


struct ObliviousEnsemble(Copyable, Movable):
    """A verified oblivious plan for the trees of one iteration range.

    Flat and index-addressed rather than a list of per-tree structs, because
    the hot loop reads one level after another and a flat array is one stride
    rather than one pointer hop per tree.

    Slot arithmetic. Tree `t` of the plan (plan order is range order) has
    `depth[t]` levels starting at slot `level_at[t]`, and `1 << depth[t]` leaf
    values starting at `leaf_at[t]`. Every level slot owns `CAT_BITSET_WORDS`
    words of `lvl_cat_words` whether or not it is categorical, so the set's
    offset is `slot * CAT_BITSET_WORDS` with no indirection; the wasted words
    are 32 bytes per level, which for a hundred depth-6 trees is 19 KB.

    `active` false is the only thing a caller has to test. Everything else is
    meaningless on an inactive plan.
    """

    var active: Bool
    var simple: Bool
    var n_trees: Int
    var depth: List[Int]
    var level_at: List[Int]
    var leaf_at: List[Int]
    var total_levels: Int
    var lvl_feature: List[Int]
    var lvl_threshold: List[Int]
    var lvl_missing: List[Int]
    var lvl_default_left: List[Bool]
    var lvl_is_cat: List[Bool]
    var lvl_cat_words: List[UInt64]
    var leaf_value: List[Float64]

    var raw_ready: Bool
    """Whether `lvl_edge` and `lvl_nan_left` hold the raw-value rewrite of
    every level, so the plan can be evaluated against the caller's Float64
    matrix without binning it first. Filled by `oblivious_raw_plan`; false on
    a plan built by `oblivious_plan` alone, which is bins-only."""

    var lvl_edge: List[Float64]
    """Per level slot, the bin EDGE `lvl_threshold` names. See
    `_raw_split` for why `v <= lvl_edge[s]` selects the same branch as
    `bin(v) <= lvl_threshold[s]` for every non-NaN `v`. Meaningless unless
    `raw_ready`."""

    var lvl_nan_left: List[Int32]
    """Per level slot, 1 when a NaN raw value goes LEFT and 0 when it goes
    right: a constant, because a NaN's bin is a constant of the feature.
    Meaningless unless `raw_ready`.

    0/1 rather than Bool, for the reason binning.mojo:1388 states in the same
    words: the raw-pointer loads that read this in
    `predict_oblivious_raw_batch` are defined for scalar element types only,
    and a `List[Bool]` has no `unsafe_load`. `Int32` rather than `UInt8` or
    `Int` so that this file carries ONE integer width across the plan
    streams, matching `nd_feature` and `nd_child` on `RawEnsemble`.

    `lvl_default_left` and `lvl_is_cat` stay `List[Bool]` on purpose. They
    are read only by ordinary subscript in `_goes_left` and the plan
    builders, never through a pointer, and a `Bool` that is never read
    unsafely is the clearer type."""

    var raw_refuse_code: Int
    """The FIRST `REFUSE_*` reason the raw rewrite declined, or `REFUSE_NONE`.
    Kept so `PredictTrace` can say why rather than leaving a reader to infer
    it from a time that did not move."""

    var raw_refuse_slot: Int
    """The level slot `raw_refuse_code` was raised at, or -1."""

    def __init__(out self):
        """An inactive plan. The generic walker handles everything."""
        self.active = False
        self.simple = False
        self.n_trees = 0
        self.depth = []
        self.level_at = []
        self.leaf_at = []
        self.total_levels = 0
        self.lvl_feature = []
        self.lvl_threshold = []
        self.lvl_missing = []
        self.lvl_default_left = []
        self.lvl_is_cat = []
        self.lvl_cat_words = []
        self.leaf_value = []
        self.raw_ready = False
        self.lvl_edge = []
        self.lvl_nan_left = []
        self.raw_refuse_code = REFUSE_NOT_OBLIVIOUS
        self.raw_refuse_slot = -1

    @always_inline
    def _goes_left(self, slot: Int, bin: Int) -> Bool:
        """`Tree.goes_left` for level slot `slot`, against the level's own
        copy of the five fields that function reads.

        Line for line, and in the same order, which matters: a categorical
        node carries `missing_bin == -1` and `default_left == False`, so the
        two tests below it are inert on it, but reversing the order would
        change what a hypothetical node carrying both would do. The threshold
        test is `<=`, not `<`, which is the boundary the whole model is
        binned against; flipping it would move exactly the rows sitting on a
        bin edge and nothing else.
        """
        if self.lvl_is_cat[slot]:
            return cat_pool_contains(
                self.lvl_cat_words, slot * CAT_BITSET_WORDS, bin
            )
        if bin == self.lvl_missing[slot]:
            return self.lvl_default_left[slot]
        return bin <= self.lvl_threshold[slot]

    @always_inline
    def _leaf_slot_bins(self, t: Int, bins: List[Int]) -> Int:
        """The `leaf_value` index tree `t` sends a row with these bins to.

        The pack puts LEVEL 0's outcome in the LEAST significant bit, 0 for
        left and 1 for right, so the index is
        `sum_l b_l * 2**l`. That is the leaf numbering
        `tree._grow_oblivious_levels` documents as CatBoost's own
        (`index_calcer.cpp`), which is why it is the one used here rather than
        the other way round; `_fill_leaf_values` decodes the same convention.

        It is not the model's LEAF ORDINAL. Ordinals rank leaves in node
        order, and for an oblivious tree node order and leaf-index order
        diverge from level 2 on (`leaf_ordinals` in tree.mojo). This index
        addresses `leaf_value` and nothing else.
        """
        var lo = self.level_at[t]
        var idx = 0
        for level in range(self.depth[t]):
            var slot = lo + level
            var bin = bins[self.lvl_feature[slot]]
            if not self._goes_left(slot, bin):
                idx |= 1 << level
        return self.leaf_at[t] + idx

    @always_inline
    def _leaf_slot_bins_simple(self, t: Int, bins: List[Int]) -> Int:
        """`_leaf_slot_bins` for a plan with no categorical level and no
        routed missing bin, which is the ordinary numerical model.

        `_goes_left` collapses to `bin <= threshold`, so a direction is one
        compare with no branch on structure, and the loop is the branchless
        form the oblivious layout exists for. Same answer as
        `_leaf_slot_bins`; the condition that lets the two tests be dropped
        rather than merely predicted is checked once per level in
        `oblivious_plan`, where `simple` is decided.
        """
        var lo = self.level_at[t]
        var idx = 0
        for level in range(self.depth[t]):
            var slot = lo + level
            var bin = bins[self.lvl_feature[slot]]
            var right = 1 if bin > self.lvl_threshold[slot] else 0
            idx |= right << level
        return self.leaf_at[t] + idx

    @always_inline
    def _leaf_slot_row(self, t: Int, data: BinnedMatrix, row: Int) -> Int:
        """`_leaf_slot_bins` reading a binned matrix directly.

        `data.bin_at(row, f)` is `bins[f * n_rows + row]`, which is the value
        a gather would have put in `bins[f]`, so this reaches the same leaf
        without materializing the row. It also touches only the features the
        plan's levels name rather than all of them.
        """
        var lo = self.level_at[t]
        var idx = 0
        for level in range(self.depth[t]):
            var slot = lo + level
            var bin = data.bin_at(row, self.lvl_feature[slot])
            if not self._goes_left(slot, bin):
                idx |= 1 << level
        return self.leaf_at[t] + idx

    @always_inline
    def _leaf_slot_row_simple(
        self, t: Int, data: BinnedMatrix, row: Int
    ) -> Int:
        """`_leaf_slot_bins_simple` reading a binned matrix directly."""
        var lo = self.level_at[t]
        var idx = 0
        for level in range(self.depth[t]):
            var slot = lo + level
            var bin = data.bin_at(row, self.lvl_feature[slot])
            var right = 1 if bin > self.lvl_threshold[slot] else 0
            idx |= right << level
        return self.leaf_at[t] + idx

    @always_inline
    def raw_from_bins(
        self,
        bins: List[Int],
        base_score: Float64,
        learning_rate: Float64,
        includes_base: Bool,
    ) -> Float64:
        """The raw ensemble score for one already-binned example.

        `Booster.predict_raw_bins_range` statement for statement: the base
        score when the range starts at iteration 0, then one
        `s += learning_rate * value` per tree in ascending range order. The
        multiply and the add stay per tree, so the Float64 accumulation is
        the same sequence of operations on the same values.
        """
        var s = base_score if includes_base else 0.0
        if self.simple:
            for t in range(self.n_trees):
                s += learning_rate * self.leaf_value[
                    self._leaf_slot_bins_simple(t, bins)
                ]
            return s
        for t in range(self.n_trees):
            s += learning_rate * self.leaf_value[
                self._leaf_slot_bins(t, bins)
            ]
        return s

    @always_inline
    def raw_from_row(
        self,
        data: BinnedMatrix,
        row: Int,
        base_score: Float64,
        learning_rate: Float64,
        includes_base: Bool,
    ) -> Float64:
        """`raw_from_bins` for one row of an already binned matrix."""
        var s = base_score if includes_base else 0.0
        if self.simple:
            for t in range(self.n_trees):
                s += learning_rate * self.leaf_value[
                    self._leaf_slot_row_simple(t, data, row)
                ]
            return s
        for t in range(self.n_trees):
            s += learning_rate * self.leaf_value[
                self._leaf_slot_row(t, data, row)
            ]
        return s


def _fill_leaf_values(tree: Tree, d: Int, mut out: List[Float64]):
    """Append tree `t`'s `1 << d` leaf values, indexed by the packed
    direction bits.

    Index `idx`'s bit `level` is level `level`'s direction, 0 for left, which
    is exactly what `_leaf_slot_bins` packs. Rather than assume anything about
    how the grower numbered its nodes, each index is decoded and walked
    through the tree's own `left` and `right` arrays, so the value stored is
    the value the generic walker would have reached by taking those same
    directions. That is the point: the leaf ordering is READ off the tree, not
    asserted about it, so a grower that renumbers its nodes cannot silently
    scramble this table.

    Costs `(1 << d) * d` steps, which is why `OBLIVIOUS_PLAN_MAX_DEPTH` exists.
    """
    var size = 1 << d
    for idx in range(size):
        var node = 0
        for level in range(d):
            var bit = (idx >> level) & 1
            node = tree.right[node] if bit == 1 else tree.left[node]
        out.append(tree.value[node])


def oblivious_plan(
    booster: Booster, rng: IterationRange, n_rows: Int
) -> ObliviousEnsemble:
    """Plan the oblivious evaluation of `booster`'s trees over `rng`, or
    return an inactive plan.

    Inactive, and the generic walker runs instead, when any of these holds:

    - the batch is smaller than `OBLIVIOUS_MIN_ROWS`, or the range is empty,
      so the plan would cost more to build than it saves;
    - the ensemble carries linear leaves, where `Tree.value` is not the whole
      story and only linear_tree.mojo knows what is;
    - any tree in the range is not structurally oblivious, or is deeper than
      `OBLIVIOUS_PLAN_MAX_DEPTH`.

    All or nothing across the range, deliberately. A mixed plan would need a
    per-tree branch in the hot loop and would buy little: an ensemble is grown
    by one policy.

    None of these gates can change a returned prediction. Both arms compute
    the same Float64 (see the module docstring), so this function is free to
    answer differently on two runs of different sizes without the model
    predicting differently.
    """
    var plan = ObliviousEnsemble()
    if n_rows < OBLIVIOUS_MIN_ROWS:
        return plan^
    if rng.stop <= rng.start:
        return plan^
    if booster.linear.is_active():
        return plan^

    var n = rng.stop - rng.start
    var depths = List[Int](capacity=n)
    for i in range(rng.start, rng.stop):
        var d = _oblivious_depth(booster.trees[i])
        if d < 0:
            return plan^
        depths.append(d)

    plan.n_trees = n
    var level_cursor = 0
    var leaf_cursor = 0
    var simple = True
    for j in range(n):
        # A reference, never `var tree = ...`: `Tree` holds Lists and is
        # `Copyable, Movable` rather than `ImplicitlyCopyable`, so binding it
        # by value would be refused, and would copy twelve arrays per tree if
        # it were not.
        ref tree = booster.trees[rng.start + j]
        var d = depths[j]
        plan.depth.append(d)
        plan.level_at.append(level_cursor)
        plan.leaf_at.append(leaf_cursor)

        # One node per level is enough: `_oblivious_depth` has already
        # checked that every node on a level routes identically, so the
        # leftmost spine carries the whole tree's routing.
        var node = 0
        for _ in range(d):
            plan.lvl_feature.append(tree.feature[node])
            plan.lvl_threshold.append(tree.threshold_bin[node])
            plan.lvl_missing.append(tree.missing_bin[node])
            plan.lvl_default_left.append(tree.default_left[node])
            var co = tree.cat_offset[node]
            plan.lvl_is_cat.append(co >= 0)
            if co >= 0:
                for w in range(CAT_BITSET_WORDS):
                    plan.lvl_cat_words.append(tree.cat_bitset[co + w])
            else:
                for _w in range(CAT_BITSET_WORDS):
                    plan.lvl_cat_words.append(UInt64(0))
            # A level is "simple" when `_goes_left` provably collapses to the
            # threshold compare. `missing_bin == -1` is what makes the missing
            # test inert, and it is inert rather than merely unlikely: every
            # bin id a caller can produce is nonnegative (`BinMapper.bin_value`
            # returns an offset into a feature's edges, `BinnedMatrix.bin_at`
            # widens a UInt8), so -1 matches nothing.
            if co >= 0 or tree.missing_bin[node] >= 0:
                simple = False
            node = tree.left[node]

        _fill_leaf_values(tree, d, plan.leaf_value)
        level_cursor += d
        leaf_cursor += 1 << d

    plan.total_levels = level_cursor
    plan.simple = simple
    plan.active = True
    return plan^


def predict_oblivious_batch(
    booster: Booster,
    plan: ObliviousEnsemble,
    data: BinnedMatrix,
    rng: IterationRange,
    raw_score: Bool = False,
) raises -> List[Float64]:
    """One prediction per row of an already binned matrix, through `plan`.

    The row-blocked counterpart of `Booster.predict_batch_range`, and it
    splits the same axis for the same reason: a row reads the plan, which no
    block writes, and writes output slot `r`, which no other row touches.
    Nothing is accumulated across rows, so the outputs are bit-identical to
    the serial form at any block count, and they are bit-identical to
    `Booster.predict_batch_range`'s by the argument in the module docstring.

    `plan.active` is the caller's precondition. An inactive plan would score
    every row against an empty tree list.
    """
    var n = data.n_rows
    var out = List[Float64](capacity=n)
    out.resize(n, 0.0)
    var out_p = out.unsafe_ptr()
    var base = booster.base_score
    var lr = booster.learning_rate
    var with_base = rng.includes_base()
    # Resolved once per call rather than once per row; see
    # `Booster.response_is_identity`. `raw_score` folds in because it already
    # meant "store the raw score", and for an identity link `response(raw)`
    # IS `raw`, so the two arms below are the same store.
    var pass_through = raw_score or booster.response_is_identity()

    def apply(start: Int, end: Int) {imm}:
        for r in range(start, end):
            var raw = plan.raw_from_row(data, r, base, lr, with_base)
            if pass_through:
                out_p.unsafe_store(r, raw)
            else:
                out_p.unsafe_store(r, booster.response(raw))

    # One level is one bin load and one compare, which is about one histogram
    # op, so the level count is the honest estimate; there is no per-row
    # gather to charge for because the plan reads the matrix in place. A
    # scheduling estimate only, like every other one handed to `dispatch_rows`.
    dispatch_rows(apply, n, n * plan.total_levels)
    return out^


# ---------------------------------------------------------------------------
# Raw-value prediction: the same trees, walked against Float64 feature values
# instead of against bin ids.
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. `Model.predict_batch` used to run `BinMapper.transform`
# over the whole scoring matrix and then walk bin ids. That pays the entire
# binning pipeline at inference time: one binary search per CELL (n_rows *
# n_features of them), a full `BinnedMatrix` allocation, and a per-row gather
# of every feature's bin whether or not any tree splits on it. LightGBM and
# CatBoost do none of it -- they compare a raw feature value against a
# Float64 threshold -- and that is the whole of the gap this file closes on
# the leaf-wise path.
#
# THE BIT-IDENTITY ARGUMENT. It is the only thing that makes the substitution
# legal, so it is written out rather than asserted, and `_raw_split` refuses
# every shape it does not cover.
#
# The boundary convention, read off `BinMapper.bin_value` and the `BinMapper`
# docstring. Feature f's edges are `edges[edge_offsets[f] : edge_offsets[f+1]]`,
# STRICTLY INCREASING; call them e[0] < e[1] < ... < e[k-1]. `bin_value` runs
# a binary search for the first index with `value <= edges[mid]` and returns
# it minus the offset. Because the edges are strictly increasing that index is
#
#     bin(v) = #{ i : e[i] < v }                                          (1)
#
# so bin b is the interval (e[b-1], e[b]] -- HALF-OPEN ON THE LEFT, CLOSED ON
# THE RIGHT. Bin 0 is everything at or below e[0], bin k is everything above
# e[k-1], and a value sitting exactly ON an edge belongs to the bin BELOW it.
# A feature with k edges therefore uses k + 1 ordinary bins, 0..k.
#
# The equivalence. For a threshold bin T with 0 <= T <= k-1,
#
#     bin(v) <= T   <=>   #{ i : e[i] < v } <= T   <=>   v <= e[T]        (2)
#
# Forward: if e[T] < v then e[0..T] are all < v, which is T+1 edges, so
# bin(v) >= T+1 > T. Backward: if v <= e[T] then no i >= T has e[i] < v
# (e[i] >= e[T] >= v), so the counted set is contained in {0..T-1} and has at
# most T members. Both directions are exact, and the comparison on the right
# is the SAME Float64 `<=` on the SAME two Float64 values the binary search
# would have performed, so there is no rounding step anywhere in (2). The
# comparison inherits the right-closed convention because `<=` IS that
# convention.
#
# The four boundary cases the argument has to cover, all of them discharged
# by (2) with no special case in the walk:
#
# - v BELOW the first edge: bin(v) = 0 <= T for every T >= 0, and v <= e[0]
#   <= e[T]. Both go left. Agreed.
# - v ABOVE the last edge: bin(v) = k > T for every T <= k-1, and v > e[k-1]
#   >= e[T]. Both go right. Agreed.
# - v exactly ON edge e[T]: (1) counts e[0..T-1] only, so bin(v) = T, and
#   T <= T goes left; `v <= e[T]` is `e[T] <= e[T]`, which goes left too.
#   Agreed. This is the case a `<` would have broken, and it is why the
#   rewrite keeps `<=`.
# - v infinite: -inf is below every finite edge and +inf is above every one,
#   which are the two cases already covered; and if an edge were itself
#   infinite, both arms still run the same `<=` on the same pair.
#
# T OUTSIDE [0, k-1]. `T >= k` cannot be beaten by any ordinary bin, since
# bin(v) <= k <= T always, so the node sends every non-missing row left;
# `POSITIVE_INF` as the edge reproduces that, because `v <= +inf` holds for
# every v including +inf. `T < 0` on a numerical node is not a shape any
# grower here produces (-1 is the leaf-and-categorical filler) and no finite
# or infinite edge reproduces "always right" for v = -inf, so `_raw_split`
# REFUSES it rather than guessing.
#
# MISSING VALUES. Two rules compose and both are handled before any compare.
#
# `bin_value` routes NaN first: to `missing_bin[f]` when the feature reserves
# one, and otherwise it replaces the value with 0.0 and bins that. `fit_bins`
# stores `n_out + 1` there, one PAST the last ordinary bin, so a reserved
# missing bin is k + 1 and lies OUTSIDE [0, k]. `Tree.goes_left` then tests
# `bin == missing_bin[node]` before the threshold and answers
# `default_left[node]`.
#
# Two consequences, and the plan depends on both:
#
# - A NaN's bin is a CONSTANT of the feature, so the direction a NaN takes at
#   a node is a constant of the node. `_raw_split` computes that bin exactly
#   the way `bin_value` would, runs the node's own `goes_left` rule on it, and
#   stores the answer as `nan_left`. Something has to carry it, because
#   `NaN <= edge` is false in IEEE-754, so a NaN falling through to a lone
#   `<=` would silently go right at every node. `RawEnsemble` carries it as
#   the THIRD child of the node rather than as a flag the walk tests: with a
#   NaN edge refused, "neither `v <= edge` nor `v > edge`" identifies a NaN
#   exactly, so the walk indexes the answer instead of branching to it.
# - A NON-NaN value can never take the missing branch, because bin(v) is in
#   [0, k] and a reserved missing bin is k + 1. So dropping the missing test
#   from the compare path removes a branch that could not have fired. To keep
#   that a fact rather than an assumption about who built the model,
#   `_raw_split` refuses any node whose `missing_bin` is neither -1 nor the
#   mapper's own reservation for that feature.
#
# NOT COVERED, and refused rather than approximated: categorical splits (a set
# membership test on a category CODE, which is a table lookup and not a
# threshold), CTR columns (their values are statistics of the training target
# and are not present in the caller's matrix at all), and linear leaves. Each
# leaves the plan inactive and the bins-and-walk path runs unchanged.


# Below this many rows the raw plan is not built. Building costs one pass over
# the ensemble's nodes; the saving is one binary search per cell of the
# scoring matrix, which grows with the rows. Eight is where a 100-tree
# ensemble's few thousand node conversions stop dominating.
#
# Like every other gate in this file it cannot change an output: both arms
# reach the same leaf of the same tree and sum in the same order.
comptime RAW_MIN_ROWS = 8


# The most nodes `raw_plan` will flatten across one iteration range. `Int32`
# node and feature indices are what make the plan twenty-four bytes per node
# (see `RawEnsemble`), and this is the bound that makes the narrowing a
# CHECKED fact rather than an assumption about how large an ensemble gets.
# `gpu_predict._append_tree` bounds its own flatten the same way and for the
# same reason. No ensemble this package can train comes near it; a plan that
# somehow did is refused, not truncated, and the bins-and-walk path answers.
comptime RAW_MAX_NODES = 2147483647


# How much work one row's walk of one tree is worth in the histogram-op
# equivalents `parallel.plan_tasks` compares against its grain. The same
# number `boosting._TRAVERSAL_ROW_OPS` uses, for the same walk, minus the
# gather that path pays and this one does not. A scheduling estimate only.
comptime _RAW_WALK_OPS = 6


def raw_predict_enabled() -> Bool:
    """Whether batch prediction may walk raw feature values.

    `MOJOTREES_RAW_PREDICT=0` forces the bin-and-walk path. Default ON, in the
    `!= "0"` form this repository uses for a default-on switch (compare
    `boosting._leaf_score_update_enabled`), because the two arms are
    bit-identical by the argument above and there is nothing for a user to
    choose between; the switch exists so the two can be measured against each
    other in one process, and it should not outlive that measurement.
    """
    return getenv("MOJOTREES_RAW_PREDICT") != "0"


@fieldwise_init
struct _RawSplit(Copyable, Movable):
    """One node's split rewritten against raw Float64 values, or `ok` false.

    `ok` false is not a failure to compute: it is a shape the rewrite is not
    proved for, and every caller answers it by leaving the plan inactive.
    `reason` is the `REFUSE_*` code, so a caller can say which shape.
    """

    var ok: Bool
    var edge: Float64
    var nan_left: Bool
    var reason: Int


def _raw_split(
    mapper: BinMapper,
    feature: Int,
    threshold_bin: Int,
    node_missing_bin: Int,
    default_left: Bool,
    cat_offset: Int,
) -> _RawSplit:
    """Rewrite one numerical node's routing against raw values.

    Every refusal below is one of the shapes the module comment names as not
    covered, and refusing is what keeps the covered case exact.
    """
    # A categorical node routes by set membership on a category CODE. There is
    # no threshold to convert.
    if cat_offset >= 0:
        return _RawSplit(False, 0.0, False, REFUSE_CATEGORICAL_SPLIT)
    # A CTR column is not a column of the caller's matrix (see
    # `BinMapper.n_total_features`), so no raw value exists to compare.
    if feature < 0 or feature >= mapper.n_features:
        return _RawSplit(False, 0.0, False, REFUSE_FEATURE_OUT_OF_MATRIX)
    if mapper.cats.is_cat(feature):
        return _RawSplit(False, 0.0, False, REFUSE_CATEGORICAL_FEATURE)
    # "Always right" for every value including -inf is not something a
    # threshold compare can express.
    #
    # **This is the refusal a 2026-08-17 review expected to be the one firing
    # on symmetric ensembles, and it is not.** The reasoning was that an
    # oblivious tree stopping before `max_depth` would carry a sentinel level
    # with `threshold_bin = -1`, and that one such level would disqualify the
    # whole ensemble. `tree._grow_oblivious_levels` does not work that way: a
    # level with no legal split, or no positive gain, `break`s out of the
    # level loop and leaves the frontier as leaves, so the tree simply ENDS
    # shallower. It never writes an internal node without a split. The only
    # two writers of a -1 threshold are `Tree._add_node` (a leaf, which never
    # reaches here) and `Tree._set_split`'s categorical arm (caught above).
    if threshold_bin < 0:
        return _RawSplit(False, 0.0, False, REFUSE_NEGATIVE_THRESHOLD)

    var lo = mapper.edge_offsets[feature]
    var k = mapper.edge_offsets[feature + 1] - lo
    var mb = mapper.missing_bin[feature]
    # The missing bin must be the one `fit_bins` reserves, one PAST the last
    # ordinary bin, or no reservation at all. That is what makes "a non-NaN
    # value never takes the missing branch" a fact about this model rather
    # than an assumption about how it was made. `Tree.missing_bin` is the
    # SPLIT FEATURE's missing bin, so it must agree with the mapper's or be
    # the -1 that matches nothing.
    if mb >= 0 and mb <= k:
        return _RawSplit(False, 0.0, False, REFUSE_MISSING_BIN_IN_RANGE)
    if node_missing_bin >= 0 and node_missing_bin != mb:
        return _RawSplit(False, 0.0, False, REFUSE_NODE_MISSING_BIN)

    # Equation (2) of the module comment, and its `T >= k` case.
    var edge = POSITIVE_INF
    if threshold_bin < k:
        edge = mapper.edges[lo + threshold_bin]
    # A NaN edge would make BOTH `v <= edge` and `v > edge` false for every
    # value, NaN and non-NaN alike, so the walk could no longer tell a missing
    # value from an ordinary one and the rewrite would stop matching
    # `bin(v) <= threshold_bin`. `fit_bins` never produces one; this is the
    # same kind of structural guard as the missing-bin checks above, and it is
    # what lets the walk read "neither below nor above" as "NaN".
    if isnan(edge):
        return _RawSplit(False, 0.0, False, REFUSE_NAN_EDGE)

    # `bin_value(feature, NaN)`, statement for statement: the reserved bin
    # when there is one, and otherwise the bin of 0.0.
    var nan_bin = mb
    if mb < 0:
        nan_bin = mapper.bin_value(feature, 0.0)
    # `Tree.goes_left` on that bin, with the categorical arm already excluded
    # above, in the same order and with the same `<=`.
    var nan_left: Bool
    if nan_bin == node_missing_bin:
        nan_left = default_left
    else:
        nan_left = nan_bin <= threshold_bin
    return _RawSplit(True, edge, nan_left, REFUSE_NONE)


struct RawEnsemble(Copyable, Movable):
    """The trees of one iteration range in flat, raw-value form.

    Structure of arrays, one entry per node, with child links already rebased
    to absolute indices in the flat arrays, so a walk touches three contiguous
    tables instead of chasing a `Tree` object per tree and ten `List`s per
    node. `tree_root[t]` is tree t's root.

    **Twenty-four bytes per node across three streams, and the shape of those
    three is the point.** The first version of this struct spread one node
    across SIX `List`s, so a single visit read `nd_feature[node]`,
    `nd_edge[node]`, `nd_nan_left[node]` and then one of `nd_left`/`nd_right`
    at the same index: four unrelated base pointers, four cache lines and four
    TLB entries for one logical record, on a loop whose whole cost is the
    dependent load chain. Three changes fold that down and none of them can
    move a bit:

    - `nd_child` is stride THREE and interleaved, so a node's three
      destinations are 12 adjacent bytes. Slot 0 is the left child, slot 1 the
      right, and slot 2 is the child a NaN takes, which is a CONSTANT of the
      node (`_raw_split.nan_left` picks it at plan time). Baking it into the
      table is what deletes `nd_nan_left` from the hot loop entirely, and it
      turns "compare, then branch to one of two loads" into one indexed load
      from an index the compare already computed.
    - `nd_edge` carries the split edge at an internal node and the LEAF VALUE
      at a leaf. They are never both live: the walk leaves the loop exactly
      when `nd_feature` is negative, and a leaf has no edge. That is one
      Float64 stream where there were two, and the leaf value is still the
      same Float64 the tree stores.
    - `nd_feature` and `nd_child` are `Int32`. A feature index is bounded by
      the matrix and a node index by `RAW_MAX_NODES`, which `raw_plan` checks
      rather than assumes. `nd_edge` stays **Float64**, because the whole
      bit-identity claim is that the comparison and the sum are the same
      Float64 operations the bins path performs.

    Deliberately NOT `gpu_predict.FlatEnsemble`, which was read first and does
    not fit: it stores leaf values as **Float32**, which is right for a device
    walk whose contract is already Float32 accumulation and is fatal here,
    where the whole claim is that the Float64 sum is unchanged. It also keys
    on `threshold_bin`, which is the quantity this path exists to stop
    computing. The layout idea is borrowed; the widths are not.

    `active` false is the caller's only precondition.
    """

    var active: Bool
    var n_trees: Int
    var tree_root: List[Int]
    var nd_feature: List[Int32]
    """The split feature, or -1 at a leaf. Negative is the loop's exit test."""

    var nd_edge: List[Float64]
    """The split edge at an internal node, the leaf value at a leaf."""

    var nd_child: List[Int32]
    """Stride three: `[3 * i]` left, `[3 * i + 1]` right, `[3 * i + 2]` the
    child a NaN takes. All -1 at a leaf, which the walk never reads."""

    var refuse_code: Int
    """The FIRST `REFUSE_*` reason the flatten declined, or `REFUSE_NONE`."""

    var refuse_tree: Int
    """The range-relative tree index `refuse_code` was raised at, or -1."""

    var refuse_node: Int
    """The node index within that tree, or -1."""

    def __init__(out self):
        """An inactive plan. The bin-and-walk path handles everything."""
        self.active = False
        self.n_trees = 0
        self.tree_root = []
        self.nd_feature = []
        self.nd_edge = []
        self.nd_child = []
        self.refuse_code = REFUSE_NONE
        self.refuse_tree = -1
        self.refuse_node = -1

    @staticmethod
    def refused(code: Int, tree: Int, node: Int) -> RawEnsemble:
        """An inactive plan carrying the reason. Every early return in
        `raw_plan` goes through here, so a refusal cannot be silent."""
        var out = RawEnsemble()
        out.refuse_code = code
        out.refuse_tree = tree
        out.refuse_node = node
        return out^


@fieldwise_init
struct PredictTrace(Copyable, Movable):
    """`MOJOTREES_PREDICT_TRACE=1`: one line per batch predict saying which
    arm ran and, when a fast arm declined, the FIRST reason it declined.

    This exists because of a specific failure, and the docstring names the
    failure rather than describing a feature. The raw arms shipped, built
    clean, and passed a `np.array_equal` bit-identity check on both growth
    policies; the symmetric arm's time then did not move, and there was no way
    to tell from outside the process whether the arm had DECLINED or had RUN
    and been no faster. Those are opposite findings with opposite fixes, and
    separating them cost a measuring session that one line of output would
    have ended.

    Off by default in the `== "1"` form this repository uses for a default-off
    switch, and off it costs one `getenv` and a Bool test per predict call.
    """

    var on: Bool

    @staticmethod
    def resolve() -> PredictTrace:
        var s = getenv("MOJOTREES_PREDICT_TRACE")
        return PredictTrace(s == "1" or s == "true" or s == "TRUE")

    def oblivious(self, plan: ObliviousEnsemble, n_rows: Int):
        """Report the symmetric raw arm, which is asked first and, when it
        answers, is the only arm that runs."""
        if not self.on:
            return
        print(
            "predict arm=oblivious_raw rows=",
            n_rows,
            " active=",
            _yn(plan.active),
            " raw_ready=",
            _yn(plan.raw_ready),
            " trees=",
            plan.n_trees,
            " levels=",
            plan.total_levels,
            " refused=",
            refusal_text(plan.raw_refuse_code),
            " at_level_slot=",
            plan.raw_refuse_slot,
            sep="",
        )

    def flat(self, sym: ObliviousEnsemble, plan: RawEnsemble, n_rows: Int):
        """Report the general raw arm, and with it the symmetric arm that
        declined ahead of it.

        Both halves matter. A reader needs to know the symmetric arm was ASKED
        and what it said, not only what ended up running; "the oblivious arm
        is not in this line" is exactly the ambiguity this whole struct exists
        to remove."""
        if not self.on:
            return
        print(
            "predict arm=flat_raw rows=",
            n_rows,
            " oblivious_active=",
            _yn(sym.active),
            " oblivious_raw_ready=",
            _yn(sym.raw_ready),
            " oblivious_refused=",
            refusal_text(sym.raw_refuse_code),
            " at_level_slot=",
            sym.raw_refuse_slot,
            sep="",
        )
        print(
            "predict flat_active=",
            _yn(plan.active),
            " flat_trees=",
            plan.n_trees,
            " flat_refused=",
            refusal_text(plan.refuse_code),
            " at_tree=",
            plan.refuse_tree,
            " at_node=",
            plan.refuse_node,
            sep="",
        )
        if not plan.active:
            print(
                "predict arm=binned_fallback: both raw arms declined, so this"
                " call pays BinMapper.transform"
            )


def raw_plan(
    booster: Booster, mapper: BinMapper, rng: IterationRange, n_rows: Int
) -> RawEnsemble:
    """Flatten `booster`'s trees over `rng` into raw-value form, or return an
    inactive plan.

    All or nothing across the range, and that is the right granularity rather
    than a shortcut worth fixing later.

    A per-TREE plan was asked for on 2026-08-17, so here is why it buys
    nothing. What this path avoids is `BinMapper.transform`, and transform is
    per MATRIX: it bins every cell of the scoring input in one pass. If a
    single tree of a hundred cannot be rewritten, that tree needs bin ids,
    so the transform has to run, so the whole avoided cost is paid anyway --
    and once the `BinnedMatrix` exists the ninety-nine rewritten trees are
    strictly WORSE off walking raw values, because a bin id is a UInt8 and a
    feature value is a Float64, eight times the traffic for the same branch.
    So a per-tree plan turns one refusal into a partial refusal that still
    pays 100% of the cost and then makes the remainder slower. The structure
    would allow it easily (`tree_root` already indexes trees independently);
    the arithmetic is what refuses it.

    The granularity that IS real is the one already in place: the ORDER of the
    two raw arms. A single ragged tree costs an ensemble `oblivious_raw_plan`
    and it still keeps `raw_plan`, because the fall-through in
    `Model.predict_batch` asks the second arm after the first declines. That
    is the fallback ladder doing the work a per-tree plan was meant to do,
    at the granularity where the avoided cost actually lives.
    """
    if not raw_predict_enabled():
        return RawEnsemble.refused(REFUSE_SWITCH_OFF, -1, -1)
    if n_rows < RAW_MIN_ROWS:
        return RawEnsemble.refused(REFUSE_TOO_FEW_ROWS, -1, -1)
    if rng.stop <= rng.start:
        return RawEnsemble.refused(REFUSE_EMPTY_RANGE, -1, -1)
    # `Tree.value` is not the whole leaf for a linear model; only
    # linear_tree.mojo knows what is.
    if booster.linear.is_active():
        return RawEnsemble.refused(REFUSE_LINEAR_LEAVES, -1, -1)
    # A CTR column is a statistic of the training target, not a column of the
    # matrix the caller hands in. Those models keep the transform.
    if mapper.ctr.is_active():
        return RawEnsemble.refused(REFUSE_CTR_TABLES, -1, -1)

    var plan = RawEnsemble()

    var n = rng.stop - rng.start
    for j in range(n):
        # A reference, never `var tree = ...`: `Tree` holds twelve Lists and
        # is `Copyable, Movable` rather than `ImplicitlyCopyable`.
        ref tree = booster.trees[rng.start + j]
        var n_nodes = len(tree.feature)
        if n_nodes == 0:
            return RawEnsemble.refused(REFUSE_EMPTY_TREE, j, -1)
        var base = len(plan.nd_feature)
        # What makes the `Int32` streams a checked fact. Cheap: once per tree.
        if base + n_nodes > RAW_MAX_NODES:
            return RawEnsemble.refused(REFUSE_TOO_MANY_NODES, j, -1)
        plan.tree_root.append(base)
        for i in range(n_nodes):
            var f = tree.feature[i]
            if f < 0:
                # A leaf. The walk reads only `nd_edge` here, which carries the
                # leaf VALUE rather than an edge, but every array keeps its
                # per-node entry so a single index addresses all of them.
                plan.nd_feature.append(Int32(-1))
                plan.nd_edge.append(tree.value[i])
                plan.nd_child.append(Int32(-1))
                plan.nd_child.append(Int32(-1))
                plan.nd_child.append(Int32(-1))
                continue
            # Links must point forward and stay in range. Every grower here
            # appends a child after the node it splits, so this always holds
            # for a grown or deserialized tree; checking it is what makes the
            # `while` in the walk provably terminate, the same reason
            # `gpu_predict._append_tree` checks it before a kernel walk.
            var l = tree.left[i]
            var r = tree.right[i]
            if l <= i or r <= i or l >= n_nodes or r >= n_nodes:
                return RawEnsemble.refused(REFUSE_BACKWARD_LINKS, j, i)
            var s = _raw_split(
                mapper,
                f,
                tree.threshold_bin[i],
                tree.missing_bin[i],
                tree.default_left[i],
                tree.cat_offset[i],
            )
            if not s.ok:
                return RawEnsemble.refused(s.reason, j, i)
            plan.nd_feature.append(Int32(f))
            plan.nd_edge.append(s.edge)
            plan.nd_child.append(Int32(base + l))
            plan.nd_child.append(Int32(base + r))
            # Slot 2: where a NaN goes. `s.nan_left` is `Tree.goes_left` run on
            # this node's own missing bin, so this is the SAME destination the
            # `isnan` branch used to look up per row, resolved once per node.
            if s.nan_left:
                plan.nd_child.append(Int32(base + l))
            else:
                plan.nd_child.append(Int32(base + r))

    plan.n_trees = n
    plan.active = True
    plan.refuse_code = REFUSE_NONE
    return plan^


def predict_raw_batch[
    features_origin: ImmOrigin, //
](
    booster: Booster,
    plan: RawEnsemble,
    features: Span[Float64, features_origin],
    n_rows: Int,
    rng: IterationRange,
    raw_score: Bool = False,
) raises -> List[Float64]:
    """One prediction per row of a RAW column-major matrix
    (`features[f * n_rows + r]`), through `plan`.

    The row-blocked counterpart of `Booster.predict_batch_range`, splitting
    the same axis for the same reason: a row reads the plan, which no block
    writes, and writes output slot `r`, which no other row touches. Nothing is
    accumulated across rows, so the outputs are bit-identical to the serial
    form at any block count.

    Against the bin-and-walk path, the sum is `s = base` then one
    `s += learning_rate * value` per tree in ascending range order, which is
    `Booster.predict_raw_bins_range` statement for statement; the leaf each
    tree contributes is the same leaf by the argument above this function.
    That sentence is a CONSTRAINT and not a description: Float64 addition is
    not associative, so reordering the tree loop would move bits. The body
    below interleaves the WALKS of four trees and leaves the ADDS in ascending
    order, which is the one arrangement that gets the instruction-level
    parallelism without touching the sequence the sentence promises.

    `plan.active` is the caller's precondition.
    """
    var out = List[Float64](capacity=n_rows)
    out.resize(n_rows, 0.0)
    var out_p = out.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    var base = booster.base_score
    var lr = booster.learning_rate
    var with_base = rng.includes_base()
    var n_trees = plan.n_trees

    var root_p = plan.tree_root.unsafe_ptr()
    var nf_p = plan.nd_feature.unsafe_ptr()
    var ed_p = plan.nd_edge.unsafe_ptr()
    var ch_p = plan.nd_child.unsafe_ptr()

    # `booster.response` resolved ONCE per call instead of once per row. See
    # `Booster.response_is_identity`: for an identity link the per-row call
    # returns its argument unchanged, so storing `s` directly is the same
    # Float64, and for every other link the call still happens. `raw_score`
    # folds in here because it already meant "store `s`".
    var pass_through = raw_score or booster.response_is_identity()

    # How many trees one interleaved block walks. After the routing branch was
    # removed, a single walk is a pure DEPENDENT chain -- load the node's
    # feature, load the value, compare, load the child, load ITS feature --
    # and every step waits on the one before it, so the core issues roughly
    # one useful instruction per L1 latency. K independent walks fill those
    # slots with each other's work. Four because the walks are independent
    # (nothing a tree reads is written by another) and because 100 trees, the
    # size this path is measured at, divides by four with no remainder; the
    # remainder loop below is what makes any other count correct rather than
    # merely uncommon.
    comptime K = 4
    var n_blocked = n_trees - (n_trees % K)

    def apply(start: Int, end: Int) {imm}:
        for r in range(start, end):
            var s = base if with_base else 0.0
            var t = 0
            while t < n_blocked:
                # Four cursors, held in registers for the whole block.
                var a0 = root_p.unsafe_load(t)
                var a1 = root_p.unsafe_load(t + 1)
                var a2 = root_p.unsafe_load(t + 2)
                var a3 = root_p.unsafe_load(t + 3)
                var f0 = Int(nf_p.unsafe_load(a0))
                var f1 = Int(nf_p.unsafe_load(a1))
                var f2 = Int(nf_p.unsafe_load(a2))
                var f3 = Int(nf_p.unsafe_load(a3))
                # Step all four while all four are still internal. A leaf's
                # feature is -1, whose sign bit is set, so the OR of the four
                # is negative exactly when at least one has landed: one test
                # rather than four, and no short-circuit branch. When the
                # trees have equal depth -- every complete depth-limited tree,
                # which is the shape this arm is measured on -- they land on
                # the same step and the four tails below run zero times. When
                # they are ragged the tails finish whichever lanes are still
                # walking, so the ANSWER does not depend on the shape at all.
                while (f0 | f1 | f2 | f3) >= 0:
                    var v0 = feat_p.unsafe_load(f0 * n_rows + r)
                    var v1 = feat_p.unsafe_load(f1 * n_rows + r)
                    var v2 = feat_p.unsafe_load(f2 * n_rows + r)
                    var v3 = feat_p.unsafe_load(f3 * n_rows + r)
                    var e0 = ed_p.unsafe_load(a0)
                    var e1 = ed_p.unsafe_load(a1)
                    var e2 = ed_p.unsafe_load(a2)
                    var e3 = ed_p.unsafe_load(a3)
                    var k0 = 0 if v0 <= e0 else (1 if v0 > e0 else 2)
                    var k1 = 0 if v1 <= e1 else (1 if v1 > e1 else 2)
                    var k2 = 0 if v2 <= e2 else (1 if v2 > e2 else 2)
                    var k3 = 0 if v3 <= e3 else (1 if v3 > e3 else 2)
                    a0 = Int(ch_p.unsafe_load(3 * a0 + k0))
                    a1 = Int(ch_p.unsafe_load(3 * a1 + k1))
                    a2 = Int(ch_p.unsafe_load(3 * a2 + k2))
                    a3 = Int(ch_p.unsafe_load(3 * a3 + k3))
                    f0 = Int(nf_p.unsafe_load(a0))
                    f1 = Int(nf_p.unsafe_load(a1))
                    f2 = Int(nf_p.unsafe_load(a2))
                    f3 = Int(nf_p.unsafe_load(a3))
                # The tails. Each is the single-cursor walk, unchanged. The
                # locals carry a lane suffix so that no two of these sibling
                # blocks declare the same name.
                while f0 >= 0:
                    var tv0 = feat_p.unsafe_load(f0 * n_rows + r)
                    var te0 = ed_p.unsafe_load(a0)
                    var tk0 = 0 if tv0 <= te0 else (1 if tv0 > te0 else 2)
                    a0 = Int(ch_p.unsafe_load(3 * a0 + tk0))
                    f0 = Int(nf_p.unsafe_load(a0))
                while f1 >= 0:
                    var tv1 = feat_p.unsafe_load(f1 * n_rows + r)
                    var te1 = ed_p.unsafe_load(a1)
                    var tk1 = 0 if tv1 <= te1 else (1 if tv1 > te1 else 2)
                    a1 = Int(ch_p.unsafe_load(3 * a1 + tk1))
                    f1 = Int(nf_p.unsafe_load(a1))
                while f2 >= 0:
                    var tv2 = feat_p.unsafe_load(f2 * n_rows + r)
                    var te2 = ed_p.unsafe_load(a2)
                    var tk2 = 0 if tv2 <= te2 else (1 if tv2 > te2 else 2)
                    a2 = Int(ch_p.unsafe_load(3 * a2 + tk2))
                    f2 = Int(nf_p.unsafe_load(a2))
                while f3 >= 0:
                    var tv3 = feat_p.unsafe_load(f3 * n_rows + r)
                    var te3 = ed_p.unsafe_load(a3)
                    var tk3 = 0 if tv3 <= te3 else (1 if tv3 > te3 else 2)
                    a3 = Int(ch_p.unsafe_load(3 * a3 + tk3))
                    f3 = Int(nf_p.unsafe_load(a3))
                # THE ONE THING THAT IS NOT FREE TO REORDER. This function's
                # docstring states the sum as "one `s += learning_rate *
                # value` per tree in ascending range order", matching
                # `Booster.predict_raw_bins_range` statement for statement,
                # and Float64 addition is not associative, so the ORDER of
                # these four adds is the bit-identity claim itself. The WALKS
                # above are interleaved; the ADDS stay strictly ascending in
                # `t`, and no cursor reads anything another cursor writes, so
                # nothing about the interleave can reach this sequence.
                s += lr * ed_p.unsafe_load(a0)
                s += lr * ed_p.unsafe_load(a1)
                s += lr * ed_p.unsafe_load(a2)
                s += lr * ed_p.unsafe_load(a3)
                t += K
            # The remainder, `n_trees % K` trees, still in ascending order and
            # still after every blocked tree, because the blocks covered
            # `0 .. n_blocked` exactly.
            while t < n_trees:
                var node = root_p.unsafe_load(t)
                var f = Int(nf_p.unsafe_load(node))
                while f >= 0:
                    var v = feat_p.unsafe_load(f * n_rows + r)
                    var edge = ed_p.unsafe_load(node)
                    # NO `isnan`, and NO branch on the routing decision. The
                    # step below is two Float64 compares, a select over three
                    # constants, and one indexed load.
                    #
                    # IEEE-754 makes every ordered comparison against NaN
                    # false, so "neither at-or-below nor above" identifies a
                    # NaN exactly. `_raw_split` refuses a NaN edge, which is
                    # what makes that an identification and not a guess: with
                    # an ordered edge the three cases below are exhaustive and
                    # mutually exclusive for every Float64 `v`.
                    #
                    #   v <= edge  -> slot 0, the left child.  `Tree.goes_left`
                    #     on `bin(v) <= threshold_bin`, unchanged.
                    #   v >  edge  -> slot 1, the right child. The exact
                    #     complement of the above on the ordered reals, so no
                    #     non-NaN value can reach slot 2.
                    #   neither    -> `v` is NaN -> slot 2, which `raw_plan`
                    #     filled with the left or right child according to
                    #     `nan_left`. That is the SAME lookup the `isnan`
                    #     branch did per row, hoisted to plan time.
                    #
                    # So the leaf reached is the leaf the branching form
                    # reached, for every value including NaN, and the sum below
                    # is untouched. What is gone is a data-dependent branch on
                    # a tree route, which is close to unpredictable, on the
                    # critical path of a dependent load chain.
                    var k = 0 if v <= edge else (1 if v > edge else 2)
                    node = Int(ch_p.unsafe_load(3 * node + k))
                    f = Int(nf_p.unsafe_load(node))
                # `nd_edge` at a leaf is the leaf VALUE; the loop exits exactly
                # when `nd_feature` says leaf, so the two never collide.
                s += lr * ed_p.unsafe_load(node)
                t += 1
            if pass_through:
                out_p.unsafe_store(r, s)
            else:
                out_p.unsafe_store(r, booster.response(s))

    dispatch_rows(apply, n_rows, n_rows * n_trees * _RAW_WALK_OPS)
    return out^


def oblivious_raw_plan(
    booster: Booster, mapper: BinMapper, rng: IterationRange, n_rows: Int
) -> ObliviousEnsemble:
    """`oblivious_plan` with every level rewritten against raw values.

    The structural verification is `oblivious_plan`'s, unchanged and not
    duplicated; this only adds `lvl_edge` and `lvl_nan_left` and sets
    `raw_ready`. A plan whose levels cannot all be rewritten comes back with
    `raw_ready` false and is still usable through `predict_oblivious_batch`,
    so a refusal here costs the binning pass and never a wrong answer.
    """
    var plan = oblivious_plan(booster, rng, n_rows)
    if not plan.active:
        # `raw_refuse_code` is already REFUSE_NOT_OBLIVIOUS from the
        # constructor, which is the honest answer: the structural check said
        # no before the rewrite was ever asked.
        return plan^
    if not raw_predict_enabled():
        plan.raw_refuse_code = REFUSE_SWITCH_OFF
        return plan^
    if mapper.ctr.is_active():
        plan.raw_refuse_code = REFUSE_CTR_TABLES
        return plan^
    for s in range(plan.total_levels):
        # A categorical level carries no threshold; `_raw_split` refuses it on
        # `cat_offset`, which is what the 0/-1 below encodes.
        var co = 0 if plan.lvl_is_cat[s] else -1
        var rs = _raw_split(
            mapper,
            plan.lvl_feature[s],
            plan.lvl_threshold[s],
            plan.lvl_missing[s],
            plan.lvl_default_left[s],
            co,
        )
        if not rs.ok:
            plan.lvl_edge = []
            plan.lvl_nan_left = []
            plan.raw_refuse_code = rs.reason
            plan.raw_refuse_slot = s
            return plan^
        plan.lvl_edge.append(rs.edge)
        # `_RawSplit.nan_left` stays a Bool, which is the right type for a
        # direction; the ENCODING to 0/1 happens here, at the single place
        # the stream is written, so the walker's `!= 0` is the only decode.
        plan.lvl_nan_left.append(Int32(1) if rs.nan_left else Int32(0))
    plan.raw_ready = True
    plan.raw_refuse_code = REFUSE_NONE
    plan.raw_refuse_slot = -1
    return plan^


comptime _OBLIVIOUS_TILE = 4 * simd_width_of[DType.float64]()
"""Rows evaluated together in `predict_oblivious_raw_batch`.

Four vectors' worth, which is `histogram.SIMD_LANES`' rule and for the same
reason: one vector wide leaves the compare, the select and the OR on a single
dependent chain, and four gives the core independent work to overlap while
still fitting the tile's accumulator in one register. It resolves to 8 on a
2-wide Float64 target such as NEON and scales with the target rather than
being pinned to one machine.
"""


def predict_oblivious_raw_batch[
    features_origin: ImmOrigin, //
](
    booster: Booster,
    plan: ObliviousEnsemble,
    features: Span[Float64, features_origin],
    n_rows: Int,
    rng: IterationRange,
    raw_score: Bool = False,
) raises -> List[Float64]:
    """`predict_oblivious_batch` against a RAW column-major matrix, level
    major within a row block.

    The loop is turned inside out relative to the bins version, and that is
    the point. An oblivious level asks ONE question of one feature, so a block
    of rows evaluates it as a straight run over `features[f * n_rows + start
    ..]`, which is contiguous in a column-major matrix, writing one bit each
    into a small per-block index array. No traversal, no dependent load, and
    the only data-dependent address in the whole thing is the leaf gather at
    the end of each tree. That is the shape CatBoost's inference has.

    Bit-identity has two halves and the transposition touches neither.

    - Same leaf. Level `l`'s bit is `0` when the level's predicate says left,
      packed at `1 << l`, which is `_leaf_slot_row_simple`'s pack and
      `_fill_leaf_values`'s decode. The predicate is `_raw_split`'s rewrite of
      the level's own `_goes_left`, exact by the argument above `RAW_MIN_ROWS`.
    - Same sum. `out[r]` is seeded with the base score exactly when
      `rng.includes_base()`, then takes one `+= learning_rate * value` per
      tree in ASCENDING range order, because the tree loop is the outer one.
      That is `Booster.predict_raw_bins_range`'s sequence of Float64
      operations on the same values. Reordering the tree loop would break it;
      reordering the row loop cannot, since rows share no accumulator.
      Holding the running sum in `out` rather than in a register changes no
      bit: an IEEE-754 Float64 add is exact for its inputs either way.

    THE ROW INDEX NEVER REACHES MEMORY. The first version of this loop was
    level major over the whole block and kept the accumulating leaf index in a
    `List[Int]` scratch, which meant a load and a store of `idx[i]` at every
    level of every tree: at depth 6 that is twelve 8-byte touches per row per
    tree against six loads of actual DATA. The row loop is now TILED at
    `_OBLIVIOUS_TILE`, with the tile's index held in one SIMD register across
    all of that tree's levels, so the only traffic left is the column reads
    and one read-modify-write of `out` per tree. Column traffic is unchanged,
    because a tile still reads its columns in ascending row order and the
    tiles run in ascending order too.

    Bit-identity survives the tiling for the reason above, restated in the
    order that matters: a row's index is built from the SAME comparisons
    against the same edges in the same level order, and the OR that packs
    them is integer. What may not move is the ACCUMULATION order per row, and
    it does not: the tree loop is still the outermost, so row `r` takes tree
    0's contribution, then tree 1's, exactly as before. Tiling reorders rows
    against each other, and rows share no accumulator.

    `plan.active and plan.raw_ready` is the caller's precondition.
    """
    var out = List[Float64](capacity=n_rows)
    out.resize(n_rows, 0.0)
    var out_p = out.unsafe_ptr()
    var feat_p = features.unsafe_ptr()
    var lr = booster.learning_rate
    var seed = booster.base_score if rng.includes_base() else 0.0
    # Resolved once per call rather than once per row; see
    # `Booster.response_is_identity`. For an identity link the trailing pass
    # below wrote back exactly what it read, one call and two compares per
    # row, on a loop that had nothing else to do.
    var pass_through = raw_score or booster.response_is_identity()

    # Every plan array the loop reads, hoisted to a raw pointer. `leaf_value`
    # is the one that matters: it was the last `List` subscript left in a
    # function that hoists everything else, and it runs once per row per tree.
    var n_trees = plan.n_trees
    var la_p = plan.level_at.unsafe_ptr()
    var dep_p = plan.depth.unsafe_ptr()
    var lat_p = plan.leaf_at.unsafe_ptr()
    var lvf_p = plan.lvl_feature.unsafe_ptr()
    var lve_p = plan.lvl_edge.unsafe_ptr()
    var lvn_p = plan.lvl_nan_left.unsafe_ptr()
    var lv_p = plan.leaf_value.unsafe_ptr()

    comptime W = _OBLIVIOUS_TILE

    def apply(start: Int, end: Int) {imm}:
        var w = end - start
        if w <= 0:
            return
        for i in range(w):
            out_p.unsafe_store(start + i, seed)
        var w_tiled = w - (w % W)
        for t in range(n_trees):
            var lo = la_p.unsafe_load(t)
            var d = dep_p.unsafe_load(t)
            var lat = lat_p.unsafe_load(t)
            var i = 0
            while i < w_tiled:
                # `acc` is the tile's leaf index, in a REGISTER for the whole
                # tree. Int32 rather than Int because the value is bounded by
                # `1 << OBLIVIOUS_PLAN_MAX_DEPTH`, which is 4096.
                var acc = SIMD[DType.int32, W](0)
                for level in range(d):
                    var slot = lo + level
                    var col = lvf_p.unsafe_load(slot) * n_rows + start + i
                    var ev = SIMD[DType.float64, W](lve_p.unsafe_load(slot))
                    var bitv = SIMD[DType.int32, W](Int32(1 << level))
                    var zero = SIMD[DType.int32, W](0)
                    # EXPLICIT vector width, not a hope that the scalar form
                    # gets vectorized. `unsafe_load[width=W]` without an
                    # `alignment` argument emits the ELEMENT alignment, which
                    # is what an arbitrary `start + i` actually has, so this
                    # is an unaligned vector load and not an assertion about
                    # the caller's buffer.
                    var v = feat_p.unsafe_load[width=W](col)
                    # NO `isnan`, and no branch per ROW. IEEE-754 makes every
                    # ordered comparison against NaN false, and that is enough
                    # to fold the missing case into the compare itself once
                    # the level's `nan_left` -- a CONSTANT of the level, not
                    # of the row -- is hoisted out:
                    #
                    #   nan_left false: `v <= edge` is already false for NaN,
                    #     so the plain compare sends NaN right, which is what
                    #     `nan_left` false means.
                    #   nan_left true:  `v > edge` is also false for NaN, so
                    #     "right when `v > edge`" sends NaN left, which is
                    #     what `nan_left` true means.
                    #
                    # For a non-NaN value the two are the same predicate,
                    # because `<=` and `>` are exact complements on the
                    # ordered reals. The two arms below are the two scalar
                    # arms they replace, lane for lane, with `select` where
                    # the scalar form wrote a conditional expression.
                    #
                    # `v.gt(ev)` AND NOT `v > ev`, which is a Mojo 1.0 spelling
                    # worth one line so the next reader does not "simplify" it
                    # back. The named methods `gt`, `lt`, `le` and friends
                    # return the per-lane `SIMD[DType.bool, W]` mask, which is
                    # the only thing carrying `.select` and `.cast`; the
                    # comparison OPERATORS are constrained to width 1.
                    #
                    # This is a good constraint and not a trap, which is worth
                    # saying because the first reading of it here was that the
                    # operator silently reduced many lanes to one answer. It
                    # does not. On a multi-lane SIMD `v > ev` is a COMPILE
                    # ERROR whose own text names the remedy: "Strict
                    # inequality is only defined for `Scalar`s; did you mean
                    # to use `SIMD.gt(...)`?" So the wrong spelling cannot
                    # produce a wrong answer, only a diagnostic. Verified
                    # against the compiler: `a.gt(b)` on [1,3,1,3] against 2
                    # gives [False, True, False, True].
                    # The
                    # remaining branch is on `lvn_p[slot]`, which is now once
                    # per level per TILE and perfectly predicted, because it
                    # repeats the same pattern for every tile of the block.
                    if lvn_p.unsafe_load(slot) != 0:
                        acc = acc | v.gt(ev).select(bitv, zero)
                    else:
                        acc = acc | v.le(ev).select(zero, bitv)
                # The leaf gather, one lane at a time because the address is
                # data dependent. `comptime for` so the lane index is a
                # constant and the extract is free.
                comptime for k in range(W):
                    var o = start + i + k
                    out_p.unsafe_store(
                        o,
                        out_p.unsafe_load(o)
                        + lr * lv_p.unsafe_load(lat + Int(acc[k])),
                    )
                i += W
            # `w % W` rows, the same predicate written scalar. The index is
            # still a register; a tail row never touches the old scratch
            # either, because there is no longer one to touch.
            while i < w:
                var idx = 0
                for level in range(d):
                    var tslot = lo + level
                    var tv = feat_p.unsafe_load(
                        lvf_p.unsafe_load(tslot) * n_rows + start + i
                    )
                    var tedge = lve_p.unsafe_load(tslot)
                    var tbit = 1 << level
                    if lvn_p.unsafe_load(tslot) != 0:
                        idx = idx | (tbit if tv > tedge else 0)
                    else:
                        idx = idx | (0 if tv <= tedge else tbit)
                var to = start + i
                out_p.unsafe_store(
                    to,
                    out_p.unsafe_load(to) + lr * lv_p.unsafe_load(lat + idx),
                )
                i += 1
        if not pass_through:
            for i in range(w):
                out_p.unsafe_store(
                    start + i, booster.response(out_p.unsafe_load(start + i))
                )

    dispatch_rows(apply, n_rows, n_rows * plan.total_levels)
    return out^
