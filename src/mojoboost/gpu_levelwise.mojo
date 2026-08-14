"""Frontier and commitment primitives for depth-batched GPU tree growth.

An experimental growth mode, not an optimization of the shipped one. The
leaf-wise growers in `tree.mojo` and `train_gpu.mojo` are untouched by this
lane and remain the only growth mojoboost trains with. See
`levelwise_policy.mojo` for the rules a level-wise grower decides by and
`docs/design/GPU_LEVELWISE.md` for what the mode is expected to cost and to
give up.

Scope of this file
------------------
Everything here is host-side, allocation-light, and free of device imports,
so it compiles and can be reasoned about on a machine with no accelerator.
It holds the two primitives a depth-batched grower needs that leaf-wise
growth has no use for:

  the frontier   `LevelFrontier`, the whole set of nodes at one depth, with
                 each node's row count, branch feature set, monotone
                 interval, and depth. A leaf-wise grower keeps the same
                 fields but has no reason to group them by depth.

  commitment     `plan_level`, which turns one level's search results and
                 one admission mask into the complete set of edits the tree,
                 the device row ranges, and the next frontier all need,
                 decided in a single host pass.

Why commitment needs `child_sums`
---------------------------------
Leaf-wise growth commits one split and immediately builds both children's
histograms, so a child's leaf value comes from the child's own histogram.
Depth-batched growth cannot do that: the whole point is that the level's
children are built together, in one launch, *after* the level has been
committed. So the child values, counts, and monotone intervals have to come
from the parent histogram partitioned by the chosen split, which is what
`child_sums` computes. It is the existing `train_gpu._count_left` rule
(route every bin by `SplitInfo.goes_left`, with the missing bin following
the split's default direction) carried through to the gradient and hessian
sums as well as the counts.

In exact arithmetic this changes nothing. The rows routed left are exactly
the rows sitting in the left-going bins, so summing those bins gives the same
totals as summing over the left child's rows. In floating point the two
summation orders round differently, so a level-wise leaf value is not bit
identical to the leaf-wise value for the same split, and under the GPU's
Float32 histograms the gap is wider. Counts are exact either way, since they
are integer sums.

What is deliberately absent
---------------------------
No kernels, no `DeviceContext`, no batched launch code, and no grower. A
batched histogram build, a batched split search, and a batched partition all
need new device buffers inside `histogram_gpu.mojo` and
`gpu_active_rows.mojo`, which this lane does not own; the design doc
specifies them and the handoff lists them as the integration work. What this
file provides is the host half those launches would be driven from, written
so it can be checked against the leaf-wise growers by reading rather than by
running.

No public parameter is registered. A tree grown this way is an ordinary
`tree.Tree` with its node ids assigned breadth first instead of best first,
so prediction, serialization, inspection, and contributions all work on it
unchanged and unaware.
"""

from .gain import soft_threshold_l1
from .histogram import Histogram
from .interaction import extend_branch
from .levelwise_policy import (
    BUDGET_RANK,
    STOP_RUNNING,
    LevelCandidate,
    admit_level,
    count_eligible,
    depth_permits_split,
    level_capacity,
    level_stop_reason,
    rows_permit_split,
)
from .monotone import (
    MONOTONE_FREE,
    OutputBounds,
    child_bounds,
    midpoint,
    monotone_sign,
)
from .split import SplitInfo


# Bytes one histogram cell occupies. A cell is one (feature, bin) pair, and a
# histogram is `n_features * n_bins` of them.
#
# On the host a `Histogram` keeps three parallel arrays of Float64 gradient,
# Float64 hessian, and Int count. On the device the same cell is three Int32
# fixed-point planes (see `histogram_gpu.GpuHistogramBuilder.out_dev`), which
# is where the factor of two between the two figures comes from.
comptime HOST_HIST_BYTES_PER_CELL = 24
comptime DEVICE_HIST_BYTES_PER_CELL = 12


struct LevelNode(Copyable, Movable):
    """One node of a level, carrying everything growth needs about it that a
    histogram does not hold.

    `node` is its tree node id, which under level-wise growth doubles as its
    device-side leaf id exactly as it does in `train_gpu.grow_tree_gpu`.
    `n_rows` is its cover, taken from its parent's histogram counts rather
    than from a row list, so it is exact. `branch` is the set of features
    split on between the root and it, empty unless interaction constraints
    are configured. `bounds` is the interval its output must lie in,
    unbounded unless a monotone constraint above it applies.
    """

    var node: Int
    var depth: Int
    var n_rows: Int
    var branch: List[Int]
    var bounds: OutputBounds

    def __init__(
        out self,
        node: Int,
        depth: Int,
        n_rows: Int,
        var branch: List[Int] = [],
        var bounds: OutputBounds = OutputBounds.unbounded(),
    ):
        self.node = node
        self.depth = depth
        self.n_rows = n_rows
        self.branch = branch^
        self.bounds = bounds^


struct LevelFrontier(Copyable, Movable):
    """Every node at one depth of a level-wise tree, in node id order.

    Node id order is not an incidental property, it is the layout rule. Ids
    are handed out breadth first, left child before right, in ascending
    parent id order, so a frontier built by appending each committed split's
    two children in `plan_level`'s order comes out sorted with no sorting
    step. `check_sorted` states the invariant for a caller that builds one by
    hand.
    """

    var nodes: List[LevelNode]
    var depth: Int

    def __init__(out self, depth: Int = 0):
        self.nodes = List[LevelNode]()
        self.depth = depth

    @staticmethod
    def root(n_rows: Int) -> LevelFrontier:
        """The depth-0 frontier: node 0 alone, covering the tree's rows (the
        bag's rows under bagging, every row otherwise)."""
        var f = LevelFrontier(0)
        f.nodes.append(LevelNode(0, 0, n_rows))
        return f^

    def n_nodes(self) -> Int:
        return len(self.nodes)

    def is_empty(self) -> Bool:
        return len(self.nodes) == 0

    def total_rows(self) -> Int:
        """Rows this level covers. The live leaves of a tree tile its active
        rows exactly (see gpu_active_rows.mojo), so for a complete frontier
        this is the whole active row count, whatever the depth. That identity
        is the reason a batched level costs one pass over the active rows no
        matter how many nodes it holds."""
        var total = 0
        for i in range(len(self.nodes)):
            total += self.nodes[i].n_rows
        return total

    def node_ids(self) -> List[Int]:
        var out = List[Int](capacity=len(self.nodes))
        for i in range(len(self.nodes)):
            out.append(self.nodes[i].node)
        return out^

    def row_counts(self) -> List[Int]:
        var out = List[Int](capacity=len(self.nodes))
        for i in range(len(self.nodes)):
            out.append(self.nodes[i].n_rows)
        return out^

    def check_sorted(self) raises:
        """Node ids must be strictly ascending and every node must sit at the
        frontier's depth. A frontier that violates either would still grow a
        valid tree, but it would grow a different one from the same data,
        which is exactly the reproducibility this mode has to keep."""
        for i in range(len(self.nodes)):
            if self.nodes[i].depth != self.depth:
                raise Error("frontier holds a node from another depth")
            if i > 0 and self.nodes[i].node <= self.nodes[i - 1].node:
                raise Error("frontier node ids are not strictly ascending")


@fieldwise_init
struct ChildSums(Copyable, Movable, Writable):
    """A parent's gradient, hessian, and row totals divided by a chosen
    split. Counts are exact integers; the two float sums round differently
    from the same quantities summed over a freshly built child histogram (see
    the module docstring)."""

    var left_grad: Float64
    var left_hess: Float64
    var left_count: Int
    var right_grad: Float64
    var right_hess: Float64
    var right_count: Int

    def total_count(self) -> Int:
        return self.left_count + self.right_count

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "ChildSums(left_count=",
            self.left_count,
            ", right_count=",
            self.right_count,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def child_sums(
    hist: Histogram, split: SplitInfo, missing_bin: Int = -1
) raises -> ChildSums:
    """Divide `hist`'s totals by `split`.

    Every bin of the split feature is routed by `SplitInfo.goes_left`, which
    is the same rule `Tree.goes_left`, `RowRouting.goes_left`, and the device
    partition kernel apply, so a child's planned count cannot disagree with
    the number of rows the device actually moves. A row in `missing_bin`
    follows the split's default direction instead of the threshold; a
    categorical split ignores `missing_bin` entirely, because its bin 0
    already collects the missing, unseen, and dropped rows and is never a set
    member.

    This generalizes `train_gpu._count_left`, which takes the same walk for
    the counts alone. A level-wise grower needs the gradient and hessian
    sums too, because it has to produce child leaf values before the
    children's own histograms exist.
    """
    if split.feature < 0 or split.feature >= hist.n_features:
        raise Error("split feature is outside the histogram")
    var base = split.feature * hist.n_bins
    var lg = 0.0
    var lh = 0.0
    var lc = 0
    var rg = 0.0
    var rh = 0.0
    var rc = 0
    for b in range(hist.n_bins):
        var go_left: Bool
        if not split.is_categorical and b == missing_bin:
            go_left = split.default_left
        else:
            go_left = split.goes_left(b)
        var i = base + b
        if go_left:
            lg += hist.grad[i]
            lh += hist.hess[i]
            lc += hist.count[i]
        else:
            rg += hist.grad[i]
            rh += hist.hess[i]
            rc += hist.count[i]
    return ChildSums(lg, lh, lc, rg, rh, rc)


@always_inline
def newton_value(
    grad_sum: Float64, hess_sum: Float64, lambda_reg: Float64,
    lambda_l1: Float64 = 0.0,
) -> Float64:
    """The leaf output `tree._leaf_value` produces, from sums a caller
    already holds rather than from a histogram. Same formula, same L1
    soft-threshold, so a level-wise leaf and a leaf-wise leaf differ only by
    the order their sums were accumulated in."""
    return -soft_threshold_l1(grad_sum, lambda_l1) / (hess_sum + lambda_reg)


struct CommittedSplit(Copyable, Movable):
    """One split of a level, decided and ready to apply.

    Everything three consumers need, resolved once on the host:

      the tree      `parent`, `left`, `right`, `split`, `missing_bin`, both
                    child values, and both child covers
      the device    `parent`, `left`, `right`, `split`, `missing_bin`, and
                    `n_left`, which is the exact left count the partition
                    takes as its `expected_left` so the launch needs no
                    readback
      the frontier  `branch`, `child_depth`, and both child intervals

    `left` is always `right - 1`: children are appended in pairs, as both
    shipped growers append them, so the layout invariant that a node's
    children are consecutive and above it holds under level-wise growth too.
    """

    var parent: Int
    var left: Int
    var right: Int
    var split: SplitInfo
    var missing_bin: Int
    var n_left: Int
    var n_right: Int
    var left_value: Float64
    var right_value: Float64
    var left_bounds: OutputBounds
    var right_bounds: OutputBounds
    var branch: List[Int]
    var child_depth: Int

    def __init__(
        out self,
        parent: Int,
        left: Int,
        right: Int,
        var split: SplitInfo,
        missing_bin: Int,
        n_left: Int,
        n_right: Int,
        left_value: Float64,
        right_value: Float64,
        var left_bounds: OutputBounds,
        var right_bounds: OutputBounds,
        var branch: List[Int],
        child_depth: Int,
    ):
        self.parent = parent
        self.left = left
        self.right = right
        self.split = split^
        self.missing_bin = missing_bin
        self.n_left = n_left
        self.n_right = n_right
        self.left_value = left_value
        self.right_value = right_value
        self.left_bounds = left_bounds^
        self.right_bounds = right_bounds^
        self.branch = branch^
        self.child_depth = child_depth


struct LevelCommit(Copyable, Movable):
    """The complete outcome of deciding one level.

    `splits` are the admitted splits in ascending parent id order, which is
    also the order their children's ids were assigned in. `terminal` holds
    the level's nodes that became leaves, either because they offered no
    split or because the leaf budget did not reach them. `n_leaves_after` is
    the tree's leaf count once the splits are applied, and `stop_reason` is
    one of the `levelwise_policy.STOP_*` constants.
    """

    var splits: List[CommittedSplit]
    var terminal: List[Int]
    var depth: Int
    var n_leaves_after: Int
    var stop_reason: Int

    def __init__(out self, depth: Int, n_leaves_after: Int, stop_reason: Int):
        self.splits = List[CommittedSplit]()
        self.terminal = List[Int]()
        self.depth = depth
        self.n_leaves_after = n_leaves_after
        self.stop_reason = stop_reason

    def n_splits(self) -> Int:
        return len(self.splits)

    def is_done(self) -> Bool:
        """Whether growth ends here. A level that committed nothing has no
        children, so the tree is finished whatever the reason."""
        return len(self.splits) == 0

    def next_frontier(self) raises -> LevelFrontier:
        """The children of this level's committed splits, in the node id
        order `plan_level` assigned them, which is already ascending. Every
        child carries its parent's extended branch set, the interval its
        parent's split gave it, and its own row count."""
        var out = LevelFrontier(self.depth + 1)
        for i in range(len(self.splits)):
            var s = self.splits[i].copy()
            out.nodes.append(
                LevelNode(
                    s.left,
                    s.child_depth,
                    s.n_left,
                    s.branch.copy(),
                    s.left_bounds.copy(),
                )
            )
            out.nodes.append(
                LevelNode(
                    s.right,
                    s.child_depth,
                    s.n_right,
                    s.branch.copy(),
                    s.right_bounds.copy(),
                )
            )
        out.check_sorted()
        return out^


def level_candidates(
    frontier: LevelFrontier,
    splits: List[SplitInfo],
    max_depth: Int,
    min_data_in_leaf: Int,
) raises -> List[LevelCandidate]:
    """The level's offers, one per frontier node, for `admit_level` to rank.

    A node is eligible only when its search found a split whose gain is
    strictly positive and both shape rules pass. The strictly-positive bar is
    the same one both leaf-wise growers set by starting their best gain at
    0.0, so a zero-gain split is refused identically in every mode. The shape
    rules are re-tested here rather than trusted from the search, because a
    grower that prefiltered its launch never called `tree._search` on the
    nodes it dropped and so has no `SplitInfo` from them to trust.
    """
    if len(frontier.nodes) != len(splits):
        raise Error("a level needs one split result per frontier node")
    var out = List[LevelCandidate](capacity=len(splits))
    for i in range(len(splits)):
        var node = frontier.nodes[i].node
        var eligible = (
            splits[i].found
            and splits[i].gain > 0.0
            and depth_permits_split(frontier.nodes[i].depth, max_depth)
            and rows_permit_split(frontier.nodes[i].n_rows, min_data_in_leaf)
        )
        if eligible:
            out.append(LevelCandidate(node, splits[i].gain, True))
        else:
            out.append(LevelCandidate.terminal(node))
    return out^


def plan_level(
    frontier: LevelFrontier,
    splits: List[SplitInfo],
    hists: List[Histogram],
    admitted: List[Bool],
    missing_bins: List[Int],
    signs: List[Int],
    lambda_reg: Float64,
    lambda_l1: Float64,
    next_node_id: Int,
    num_leaves: Int,
    max_depth: Int,
    n_leaves_before: Int,
    n_eligible: Int,
) raises -> LevelCommit:
    """Turn one level's search results into the edits that apply it.

    `admitted` comes from `levelwise_policy.admit_level` and decides
    membership only. Ids are assigned here in ascending parent id order, left
    child then right child, so the tree's layout is a function of the
    frontier and the mask and never of how the gains happened to rank. A
    grower must therefore create its nodes in exactly this order, which for
    both shipped growers means calling `Tree._add_node` twice per entry of
    `splits`, in order; `check_child_ids` is the assertion that it did.

    Child values, covers, and monotone intervals all come from `child_sums`
    on the parent's own histogram, so the level is fully decided before any
    child histogram exists. The monotone treatment is the same three steps
    the leaf-wise growers take and in the same order: clamp both child values
    into the parent's interval, collapse both to their midpoint if a rounding
    step inverted them under an active constraint, then divide the parent's
    interval between the children. Under no constraint every step is the
    identity, exactly as it is today.

    `n_eligible` is how many of the level's nodes could have been split, from
    `levelwise_policy.count_eligible` on the same candidates the mask was
    built from. It is passed in rather than recovered here because the
    difference between it and the number admitted is precisely what tells a
    partial level apart from a dry one, and only the caller that ranked the
    candidates knows it.
    """
    if len(frontier.nodes) != len(splits):
        raise Error("a level needs one split result per frontier node")
    if len(frontier.nodes) != len(hists):
        raise Error("a level needs one histogram per frontier node")
    if len(frontier.nodes) != len(admitted):
        raise Error("a level needs one admission flag per frontier node")
    frontier.check_sorted()

    var commit = LevelCommit(frontier.depth, n_leaves_before, STOP_RUNNING)
    var node_id = next_node_id
    for i in range(len(frontier.nodes)):
        var parent = frontier.nodes[i].node
        if not admitted[i]:
            commit.terminal.append(parent)
            continue

        var split = splits[i].copy()
        if split.feature < 0 or split.feature >= len(missing_bins):
            raise Error("split feature has no missing-bin entry")
        var missing_bin = -1 if split.is_categorical else (
            missing_bins[split.feature]
        )
        var sums = child_sums(hists[i], split, missing_bin)
        # Read out of `split` before it is transferred into the commit below.
        var branch = extend_branch(frontier.nodes[i].branch, split.feature)

        var parent_bounds = frontier.nodes[i].bounds.copy()
        var sign = monotone_sign(signs, split.feature)
        var left_value = parent_bounds.clamp(
            newton_value(
                sums.left_grad, sums.left_hess, lambda_reg, lambda_l1
            )
        )
        var right_value = parent_bounds.clamp(
            newton_value(
                sums.right_grad, sums.right_hess, lambda_reg, lambda_l1
            )
        )
        if sign != MONOTONE_FREE and left_value > right_value:
            var mid = midpoint(left_value, right_value)
            left_value = mid
            right_value = mid
        var children = child_bounds(
            parent_bounds, sign, left_value, right_value
        )

        var left = node_id
        var right = node_id + 1
        node_id += 2
        commit.splits.append(
            CommittedSplit(
                parent,
                left,
                right,
                split^,
                missing_bin,
                sums.left_count,
                sums.right_count,
                left_value,
                right_value,
                children.left.copy(),
                children.right.copy(),
                branch^,
                frontier.nodes[i].depth + 1,
            )
        )

    commit.n_leaves_after = n_leaves_before + len(commit.splits)
    commit.stop_reason = level_stop_reason(
        frontier.depth,
        max_depth,
        commit.n_leaves_after,
        num_leaves,
        n_eligible,
        len(commit.splits),
    )
    return commit^


def decide_level(
    frontier: LevelFrontier,
    splits: List[SplitInfo],
    hists: List[Histogram],
    missing_bins: List[Int],
    signs: List[Int],
    lambda_reg: Float64,
    lambda_l1: Float64,
    next_node_id: Int,
    num_leaves: Int,
    max_depth: Int,
    min_data_in_leaf: Int,
    n_leaves_before: Int,
    budget_mode: Int = BUDGET_RANK,
) raises -> LevelCommit:
    """`level_candidates`, then `admit_level`, then `plan_level`.

    The whole host decision for one depth, in one call, which is the shape
    the mode is built around: a level-wise grower reaches the host once per
    level and this is what it does there.
    """
    var candidates = level_candidates(
        frontier, splits, max_depth, min_data_in_leaf
    )
    var admitted = admit_level(
        candidates, n_leaves_before, num_leaves, budget_mode
    )
    return plan_level(
        frontier,
        splits,
        hists,
        admitted,
        missing_bins,
        signs,
        lambda_reg,
        lambda_l1,
        next_node_id,
        num_leaves,
        max_depth,
        n_leaves_before,
        count_eligible(candidates),
    )


def check_child_ids(commit: LevelCommit, next_node_id: Int) raises:
    """Verify a level's ids are the consecutive block starting at
    `next_node_id`, in ascending parent order, left before right.

    A grower creates its nodes through `Tree._add_node`, which appends, so
    this holds by construction as long as the grower walks `commit.splits` in
    order and creates exactly two nodes per entry. It is checked rather than
    assumed because a mismatch would not fail loudly: the tree would still
    predict, it would simply be a different tree from the one the level
    planned, with children hanging off the wrong parents.
    """
    var expected = next_node_id
    for i in range(len(commit.splits)):
        var s = commit.splits[i].copy()
        if s.left != expected or s.right != expected + 1:
            raise Error("committed child ids are not the consecutive block")
        if i > 0 and s.parent <= commit.splits[i - 1].parent:
            raise Error("committed splits are not in ascending parent order")
        expected += 2


# ----------------------------------------------------------------------
# Sizing. A batched level trades launches for resident histograms, and these
# are the numbers that trade is made against. See the memory section of
# docs/design/GPU_LEVELWISE.md.
# ----------------------------------------------------------------------


def level_histogram_cells(
    n_nodes: Int, n_features: Int, n_bins: Int
) -> Int:
    """Histogram cells one level's batched build writes. A cell is one
    (feature, bin) pair, held as three planes in both layouts."""
    return n_nodes * n_features * n_bins


def level_histogram_bytes(
    n_nodes: Int,
    n_features: Int,
    n_bins: Int,
    bytes_per_cell: Int = DEVICE_HIST_BYTES_PER_CELL,
) -> Int:
    return level_histogram_cells(n_nodes, n_features, n_bins) * bytes_per_cell


def max_level_nodes_for_bytes(
    budget_bytes: Int,
    n_features: Int,
    n_bins: Int,
    bytes_per_cell: Int = DEVICE_HIST_BYTES_PER_CELL,
) raises -> Int:
    """How many nodes one launch group may cover under a memory budget, at
    least 1. This is what `LevelwiseParams.max_level_nodes` would be set from
    on a device whose histogram budget is known."""
    if n_features < 1 or n_bins < 1 or bytes_per_cell < 1:
        raise Error("histogram shape must be positive")
    var per_node = n_features * n_bins * bytes_per_cell
    var n = budget_bytes // per_node
    return n if n > 1 else 1


def peak_resident_nodes(depth: Int, num_leaves: Int) -> Int:
    """Histograms live at once while a level at `depth` produces the level
    below it: the level's own, still needed for the sibling subtraction and
    for `child_sums`, plus its children's.

    A level at `depth` holds `L = level_capacity(depth, num_leaves)` nodes,
    which is the tree's whole leaf count at that moment. It can commit at
    most `num_leaves - L` splits before the budget is spent and at most `L`
    splits because that is how many nodes it has, so it produces at most
    `2 * min(L, num_leaves - L)` children. The peak is therefore maximized at
    `L = num_leaves / 2` and never exceeds `1.5 * num_leaves` histograms.

    That bound is what defuses the obvious memory objection to batching a
    whole level. Leaf-wise growth already holds one histogram per live leaf
    (`tree._HistPool` sizes its free list at `num_leaves + 1`), so at the same
    leaf budget this mode's peak is within a factor of 1.5 of the peak the
    shipped grower already pays. The mode is only a genuinely new memory risk
    when the leaf budget is lifted and `max_depth` alone bounds growth, where
    a complete level at depth 12 is 4096 histograms and, at 100 features and
    256 bins, 1.2 GiB of device memory.
    """
    var here = level_capacity(depth, num_leaves)
    var room = num_leaves - here
    var splits = here if here < room else room
    if splits < 0:
        splits = 0
    return here + 2 * splits


def peak_resident_bytes(
    depth: Int,
    num_leaves: Int,
    n_features: Int,
    n_bins: Int,
    bytes_per_cell: Int = DEVICE_HIST_BYTES_PER_CELL,
) -> Int:
    return level_histogram_bytes(
        peak_resident_nodes(depth, num_leaves),
        n_features,
        n_bins,
        bytes_per_cell,
    )
