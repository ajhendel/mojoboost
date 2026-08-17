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

from .binning import BinnedMatrix
from .boosting import Booster, IterationRange
from .categorical import CAT_BITSET_WORDS, cat_pool_contains
from .parallel import dispatch_rows
from .tree import Tree


# The deepest oblivious tree this file will plan for. Deliberately NOT
# `growth_policy.OBLIVIOUS_MAX_DEPTH`, which is 16 and is the deepest
# symmetric tree the grower will build; the two are different budgets on
# different things and giving them the same value would be a coincidence
# rather than a rule.
#
# The leaf table is `2 ** depth` Float64 per tree and costs
# `2 ** depth * depth` steps to build, so the plan's cost is exponential in
# the depth while the saving per row is only linear in it. Twelve is 4,096
# leaves, 32 KB per tree, and 12 MB across a 360-tree CatBoost-mode ensemble;
# sixteen would be 512 KB per tree and 184 MB across the same ensemble, which
# is why the grower's cap is not this one. CatBoost's default depth is 6,
# where the table is 512 bytes per tree.
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

    def apply(start: Int, end: Int) {imm}:
        for r in range(start, end):
            var raw = plan.raw_from_row(data, r, base, lr, with_base)
            if raw_score:
                out_p.unsafe_store(r, raw)
            else:
                out_p.unsafe_store(r, booster.response(raw))

    # One level is one bin load and one compare, which is about one histogram
    # op, so the level count is the honest estimate; there is no per-row
    # gather to charge for because the plan reads the matrix in place. A
    # scheduling estimate only, like every other one handed to `dispatch_rows`.
    dispatch_rows(apply, n, n * plan.total_levels)
    return out^
