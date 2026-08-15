"""The leaf-wise frontier, as a data structure that can be batched over.

`train_gpu.grow_tree_gpu` keeps its frontier as a `List[_GpuLeafState]` and
does four things with it in one loop body: pick the best-gain leaf, commit
that split, create the two children, and compute the children's candidates.
Those four steps are interleaved, so the loop can only ever have one leaf's
worth of device work in flight. This module separates them, because only one
of the four is actually serial.

What is serial and what is not
------------------------------
**Order-independence lemma.** In leaf-wise growth the rows a frontier leaf
owns are decided by the path from the root to that leaf, that is by the
sequence of splits *above* it. They do not depend on the order in which the
other frontier leaves are split, because a split touches only its own leaf's
rows (`GpuActiveRows` makes that structural: a leaf owns a half-open range of
the active-row permutation and its split rewrites only that range). Therefore
a leaf's histogram, and the best split that histogram admits under a fixed
feature set, are invariant to the commit order of every other leaf.

**Corollary.** Any device work whose only inputs are a leaf's row range, the
round's gradients, and the tree's feature set may run at any time after the
leaf exists, in any order. That covers histogram accumulation, the tiled
reduction, sibling subtraction, and even the stable partition of a leaf by
its own chosen split.

**What is genuinely order-dependent**, and therefore stays serial here:

1. Node ids. `Tree._add_node` hands them out in commit order, and
   `select_node_features(..., node)` draws a node's feature subset from its
   id, so with `feature_fraction_bynode < 1.0` a leaf's *candidate* (not its
   histogram) depends on the order. `search_is_order_free` below is the
   predicate for that.
2. Which leaves are split at all. `num_leaves` is a budget, so the order
   decides where growth stops.
3. Nothing else. `depth`, the interaction `branch` and its allow mask, and
   the monotone `OutputBounds` all descend from the ancestor chain, which the
   order cannot change.

So batching is legitimate for construction and illegitimate for commitment,
which is exactly the separation this module encodes: `pending()` returns the
work list a batch may cover, `select_best()` returns the one leaf that may be
committed next, and `plan_commit`/`apply_commit` move the frontier by exactly
one split using the same slot convention the trainer uses today (the left
child takes the parent's slot, the right child is appended), so node ids and
per-node feature draws come out identical.

Speculation
-----------
`speculative_order` names the leaves a batch would cover if the current
candidate ranking held all the way down. That ranking can be wrong: a child
created by an early commit can outrank a leaf further down the list. The
lemma above is what makes being wrong cheap rather than incorrect. Work done
for a leaf that is not committed yet is not wasted, it is early, because that
leaf's rows, histogram, and candidate are the same whenever it is committed.
Two consequences worth stating plainly:

- A speculative *partition* of a leaf that is not committed leaves the leaf's
  row *set* unchanged and only reorders rows inside its own range. Histograms
  are sums over the range, so nothing downstream can observe the reordering.
  And the stable partition is idempotent: running it again on an already
  partitioned range recomputes the same flags, and stability keeps both sides
  in the relative order they are already in, so the second pass is the
  identity. A speculative partition therefore never has to be undone or
  tracked for correctness, only for not paying for it twice.
- A speculative *histogram* stays valid until its leaf is split. It is
  wasted only if `num_leaves` runs out before the leaf is reached.

`SpeculationLedger` counts what actually happened so a benchmark can report
the miss rate rather than a claim about it. Nothing here decides how deep to
speculate; that is a policy question a measurement answers.

What the frontier owns
----------------------
Everything about a live leaf that is not a device buffer, in one place, so a
grower never keeps a second parallel list of any of it:

    offsets and lengths   `row_begin`/`row_count`, the leaf's window into the
                          active-row permutation. `GpuActiveRows.LeafRange` is
                          the same window on the device side, and
                          `GpuActiveRows.check_frontier` holds the two equal
                          rather than assuming they stay so.
    statistics            `LeafStats`: gradient sum, hessian sum, exact row
                          count, plus `set_sibling_stats` for the host half of
                          the subtraction trick.
    completion            `max_leaves`, `status()`, `is_complete()`: whether a
                          tree stopped because its budget ran out or because
                          nothing was left to split, which the trainer's
                          `while` loop cannot currently distinguish.
    multiclass index      `plane`, the class whose gradient plane this tree's
                          leaves read, stamped onto every `LeafWorkItem`.
    device bookkeeping    `hist_slot` (which histogram slot holds this leaf)
                          and `partitioned` (whether its rows are already
                          split by its own candidate).

Scope
-----
Host-side bookkeeping only. No `DeviceContext`, no buffer, no kernel, no
environment read, so the whole frontier story is reasonable, and later
testable, on a machine with no accelerator. The device half is
`gpu_leaf_batching.mojo`, which consumes the `LeafWorkItem` list this module
produces, and `gpu_active_rows.mojo`, which consumes the `CommitPlan`.
"""

from .interaction import extend_branch
from .monotone import (
    MONOTONE_FREE,
    ChildBounds,
    OutputBounds,
    child_bounds,
    midpoint,
    monotone_sign,
)
from .split import SplitInfo


# --- Candidate state ------------------------------------------------------
#
# A frontier leaf is only eligible for commitment once its candidate has been
# computed. The states are explicit rather than implied by a sentinel gain,
# because "no split was found" and "no search has run yet" are different
# facts and the second one must never be read as a gain of zero.

comptime CAND_MISSING = 0
"""No histogram and no search have been run for this leaf yet."""

comptime CAND_PENDING = 1
"""Device work for this leaf has been enqueued and not yet consumed. A
pending leaf is not eligible for commitment, because its gain is not known."""

comptime CAND_READY = 2
"""A split was found. `FrontierCandidate.split.gain` ranks it."""

comptime CAND_NONE = 3
"""The search ran and admitted no split (every candidate failed
`min_child_hess`, `min_data_in_leaf`, the depth limit, the monotone test, or
the interaction mask). The leaf stays a leaf for the rest of the tree."""


comptime NO_SLOT = -1
"""`hist_slot` of a leaf holding no device histogram slot."""


def candidate_state_name(state: Int) -> String:
    if state == CAND_MISSING:
        return String("missing")
    if state == CAND_PENDING:
        return String("pending")
    if state == CAND_READY:
        return String("ready")
    if state == CAND_NONE:
        return String("none")
    return String("invalid")


@fieldwise_init
struct LeafStats(Copyable, Movable):
    """One leaf's gradient sum, hessian sum, and row count.

    The statistics the frontier owns, alongside the offsets and lengths. They
    are what a leaf's output value is computed from and what the sibling
    subtraction operates on, and holding them here is what lets a grower
    derive a sibling without either downloading a histogram or asking the
    device for a second one.

    `count` is exact. The two sums are whatever precision produced them: the
    host split scan sums a downloaded Float64 histogram, the device search
    returns Float32 child statistics. Both are recorded as Float64 and neither
    is claimed to equal the other, which is the same honesty
    `_grow_tree_gpu_device_search` already states about its gains.

    Under bagging or GOSS these are sums over the *bag*, because the frontier
    only ever sees the rows the active-row permutation made live, and under
    GOSS they are sums of already-scaled gradients, because GOSS scaling is
    applied to the gradients before they are uploaded. Nothing here rescales
    anything; see `LeafFrontier`'s row-weight note.
    """

    var sum_grad: Float64
    var sum_hess: Float64
    var count: Int

    @staticmethod
    def zero() -> LeafStats:
        return LeafStats(0.0, 0.0, 0)

    def subtract(self, child: LeafStats) -> LeafStats:
        """`self - child`: the sibling's statistics.

        The host mirror of `gpu_leaf_batching.enqueue_subtract`, and subject
        to the same condition: the two must have been accumulated over the
        same rows-partition (parent = left + right) for the difference to mean
        anything. The count comes out exact; the sums come out to the
        precision of their inputs.
        """
        return LeafStats(
            self.sum_grad - child.sum_grad,
            self.sum_hess - child.sum_hess,
            self.count - child.count,
        )

    def is_empty(self) -> Bool:
        return self.count <= 0


struct FrontierCandidate(Copyable, Movable):
    """One leaf's best split and everything a commit needs from it.

    The child row counts are exact integers, taken from the parent
    histogram's count plane (the host path's `_count_left`) or from the
    device split record's `left.count`/`right.count`. Both are the same
    number counted the same way, which is what lets a commit enqueue the
    partition without waiting for the device to report a left count.

    The child values are the raw Newton values the search produced, before
    the monotone clamp. `plan_commit` applies the clamp and the midpoint
    collapse, so the clamping rule lives in one place whichever search
    produced the candidate.
    """

    var state: Int
    var split: SplitInfo
    var n_left: Int
    var n_right: Int
    var left_value: Float64
    var right_value: Float64
    var parent_value: Float64

    def __init__(out self):
        self.state = CAND_MISSING
        self.split = SplitInfo(-1, -1, 0.0, False)
        self.n_left = 0
        self.n_right = 0
        self.left_value = 0.0
        self.right_value = 0.0
        self.parent_value = 0.0

    @staticmethod
    def none(parent_value: Float64 = 0.0) -> FrontierCandidate:
        """The searched-and-found-nothing candidate."""
        var c = FrontierCandidate()
        c.state = CAND_NONE
        c.parent_value = parent_value
        return c^

    @staticmethod
    def ready(
        var split: SplitInfo,
        n_left: Int,
        n_right: Int,
        left_value: Float64,
        right_value: Float64,
        parent_value: Float64,
    ) raises -> FrontierCandidate:
        var c = FrontierCandidate()
        if not split.found:
            raise Error("a ready candidate must carry a found split")
        if n_left < 0 or n_right < 0:
            raise Error("child row counts must be nonnegative")
        c.state = CAND_READY
        c.split = split^
        c.n_left = n_left
        c.n_right = n_right
        c.left_value = left_value
        c.right_value = right_value
        c.parent_value = parent_value
        return c^

    def is_ready(self) -> Bool:
        return self.state == CAND_READY

    def is_pending(self) -> Bool:
        return self.state == CAND_PENDING

    def needs_work(self) -> Bool:
        """Whether a batch has to cover this leaf. A leaf whose search
        already ran, with or without a split, needs nothing."""
        return self.state == CAND_MISSING

    def gain(self) -> Float64:
        """The ranking key. Zero for every state but `CAND_READY`, which is
        also the value the trainer's `best_gain` starts at, so an unready
        leaf can never win the comparison below."""
        if self.state != CAND_READY:
            return 0.0
        return self.split.gain

    def total_rows(self) -> Int:
        return self.n_left + self.n_right


# --- Frontier leaves ------------------------------------------------------


struct FrontierLeaf(Copyable, Movable):
    """A grown but unsplit leaf.

    `row_begin`/`row_count` is the leaf's window into the device-resident
    active-row permutation, the same `[begin, end)` `LeafRange` carries. It
    is held here as well so a batch can be planned, costed, and checked for
    overlap without touching the device or the range table.

    `hist_slot` is the device histogram slot holding this leaf's histogram,
    or `NO_SLOT` when it holds none (never built, evicted, or already
    consumed). A leaf with no slot can always be rebuilt from its row range,
    which is why eviction is a memory policy and not a correctness question.

    `partitioned` records that the device rows in this leaf's range have
    already been stably partitioned by `candidate.split`. It exists so a
    speculative partition is not paid for twice; because that partition is
    idempotent, a stale False costs a redundant launch and never a wrong
    answer.
    """

    var node: Int
    var row_begin: Int
    var row_count: Int
    var depth: Int
    var branch: List[Int]
    var bounds: OutputBounds
    var hist_slot: Int
    var partitioned: Bool
    var candidate: FrontierCandidate
    var stats: LeafStats
    """This leaf's gradient sum, hessian sum, and row count. `count` is
    maintained by the frontier itself and always equals `row_count`; the two
    sums are `stats_known` only once a search or a subtraction filed them."""

    var stats_known: Bool

    def __init__(
        out self,
        node: Int,
        row_begin: Int,
        row_count: Int,
        depth: Int = 0,
        var branch: List[Int] = [],
        var bounds: OutputBounds = OutputBounds.unbounded(),
    ):
        self.node = node
        self.row_begin = row_begin
        self.row_count = row_count
        self.depth = depth
        self.branch = branch^
        self.bounds = bounds^
        self.hist_slot = NO_SLOT
        self.partitioned = False
        self.candidate = FrontierCandidate()
        self.stats = LeafStats(0.0, 0.0, row_count)
        self.stats_known = False

    def row_end(self) -> Int:
        return self.row_begin + self.row_count

    def is_empty(self) -> Bool:
        return self.row_count <= 0


struct LeafWorkItem(Copyable, Movable):
    """One leaf's entry in a batched device launch.

    This is the whole contract between the frontier and
    `gpu_leaf_batching.mojo`: a row window, where the result goes, and which
    gradient plane it reads. Everything else about the leaf (its node id, its
    branch, its bounds) is host bookkeeping that no kernel needs, so none of
    it crosses into a device table.

    `plane` selects among the gradient/hessian planes when several are
    resident at once. The single-tree round the trainer runs today has one
    plane and passes 0. A multiclass round holds one plane per class; see the
    plane discussion in `gpu_leaf_batching.mojo`.
    """

    var slot: Int
    """Index of this leaf in the frontier it came from, so a downloaded
    result can be routed back without a search."""

    var node: Int
    var row_begin: Int
    var row_count: Int
    var out_slot: Int
    var plane: Int

    def __init__(
        out self,
        slot: Int,
        node: Int,
        row_begin: Int,
        row_count: Int,
        out_slot: Int,
        plane: Int = 0,
    ):
        self.slot = slot
        self.node = node
        self.row_begin = row_begin
        self.row_count = row_count
        self.out_slot = out_slot
        self.plane = plane


# --- Commit planning ------------------------------------------------------


@fieldwise_init
struct CommitPlan(Copyable, Movable):
    """Everything one commit does, computed before anything is mutated.

    Splitting the decision from the mutation is what lets a caller enqueue
    the device half (partition, child histograms) and update the host half in
    either order, and it keeps the monotone clamp, the subtraction choice,
    and the node id assignment in one readable place.

    `build_left` is the subtraction choice: the smaller child is built from
    its own rows and the larger is derived by subtracting it from the parent,
    which is what both growers already do. It is decided by row count alone,
    so it is a pure function of the candidate and cannot vary run to run.
    """

    var slot: Int
    var parent_node: Int
    var left_node: Int
    var right_node: Int
    var left_begin: Int
    var left_count: Int
    var right_begin: Int
    var right_count: Int
    var left_value: Float64
    var right_value: Float64
    var child_depth: Int
    var build_left: Bool
    var branch: List[Int]
    var bounds: ChildBounds
    var split: SplitInfo
    var missing_bin: Int
    """The split feature's missing bin, or -1 when it has none.

    Carried on the plan because the routing rule needs it and the frontier is
    the last place that knows which feature was split. It is what lets
    `GpuActiveRows.apply_commit` build the `RowRouting` itself, so a caller
    partitions a leaf without restating the missing/categorical rule for a
    fourth time. A categorical split ignores it, exactly as
    `RowRouting.from_split` does.
    """

    def smaller_child_node(self) -> Int:
        return self.left_node if self.build_left else self.right_node

    def larger_child_node(self) -> Int:
        return self.right_node if self.build_left else self.left_node

    def smaller_child_rows(self) -> Int:
        return self.left_count if self.build_left else self.right_count

    def smaller_child_slot_is_left(self) -> Bool:
        """Whether the child whose histogram is built occupies the parent's
        frontier slot. The left child always does (see `apply_commit`), so
        this is `build_left` under another name and is spelled out because a
        caller routing a batched result back needs the slot, not the node."""
        return self.build_left


def subtraction_builds_left(n_left: Int, n_right: Int) -> Bool:
    """Which child the histogram is built for, the other coming from the
    subtraction. Ties go to the left child, matching `grow_tree` and
    `grow_tree_gpu`, both of which test `n_left <= n_right`."""
    return n_left <= n_right


# --- Completion -----------------------------------------------------------
#
# Why a tree stopped growing, as a state the frontier owns rather than as the
# shape of a `while` loop in the trainer. Both growers in `train_gpu.mojo`
# leave the same tree for the two reasons below and record neither, so a
# caller cannot tell a tree that ran out of budget from one that ran out of
# splits. It is the difference between "raise num_leaves" and "nothing left to
# find".

comptime FRONTIER_GROWING = 0
"""At least one leaf holds a ready candidate and the budget is not spent."""

comptime FRONTIER_BUDGET_SPENT = 1
"""`max_leaves` reached. Splittable leaves may well remain."""

comptime FRONTIER_NO_CANDIDATE = 2
"""No leaf holds a ready candidate: every search that ran admitted no split,
under the gain floor, the child floors, the depth limit, the monotone test, or
the interaction mask."""

comptime FRONTIER_WORK_PENDING = 3
"""Growth cannot proceed *yet*: some leaf's candidate has never been computed
or is still in flight, so `select_best` may not be believed. A caller sees
this exactly when it has enqueued work it has not consumed."""


def frontier_status_name(status: Int) -> String:
    if status == FRONTIER_GROWING:
        return String("growing")
    if status == FRONTIER_BUDGET_SPENT:
        return String("budget_spent")
    if status == FRONTIER_NO_CANDIDATE:
        return String("no_candidate")
    if status == FRONTIER_WORK_PENDING:
        return String("work_pending")
    return String("unknown")


# --- The frontier ---------------------------------------------------------


struct LeafFrontier(Movable):
    """The live leaves of one tree, in the trainer's slot order.

    Slot order is the tie-breaking rule, so it is preserved exactly: a commit
    overwrites the parent's slot with the left child and appends the right
    child, which is what `grow_tree_gpu` does with
    `frontier[best_i] = ...; frontier.append(...)`. Node ids are handed out
    left then right at commit time, matching two `Tree._add_node` calls in
    that order. Reproducing both is what keeps a batched grower's tree
    identical to the serial one, node for node, rather than merely similar.
    """

    var leaves: List[FrontierLeaf]
    var n_active: Int
    var next_node: Int
    var n_leaves: Int
    var max_leaves: Int
    """The `num_leaves` budget this tree grows under, or 0 for unbounded.
    Owned here so `status()` can tell a tree that spent its budget from one
    that ran out of splits; `select_best` does not consult it, because
    stopping is the caller's decision and this is the caller's fact."""

    var plane: Int
    """Which gradient plane this tree's leaves read: the class index in a
    multiclass round, 0 in a single-tree one.

    Multiclass indexing lives here because it is a property of the tree being
    grown, not of any leaf in it: a K-class round grows K trees per boosting
    iteration, one per class, each over the same row permutation and each
    reading its own plane of a `K * n_rows` gradient buffer.
    `work_items` stamps it onto every batch entry, which is what
    `gpu_leaf_batching.ITEM_PLANE` multiplies by `n_rows` in the kernels, so a
    batch may span classes without the kernels branching on one.

    Row weights are deliberately *not* here. Bagging and GOSS reach the
    frontier only through `n_active` (the bag is the live prefix of the row
    permutation) and through the gradients themselves (GOSS scaling is applied
    before upload). Nothing in this module multiplies a per-row weight, and
    nothing should: two places that scale gradients is one place too many.
    """

    def __init__(out self):
        self.leaves = List[FrontierLeaf]()
        self.n_active = 0
        self.next_node = 0
        self.n_leaves = 0
        self.max_leaves = 0
        self.plane = 0

    def begin_tree(
        mut self, n_active: Int, max_leaves: Int = 0, plane: Int = 0
    ) raises:
        """Start a tree whose root owns `[0, n_active)`, which is the bag
        when bagging is on and every row when it is not.

        `max_leaves` is the `num_leaves` budget, 0 meaning unbounded, and
        `plane` the class index of a multiclass round. Both default to what a
        single-tree unbounded run wants, so an existing caller's
        `begin_tree(n)` is unchanged.
        """
        if n_active < 0:
            raise Error("active row count must be nonnegative")
        if max_leaves < 0:
            raise Error("leaf budget must be nonnegative")
        if plane < 0:
            raise Error("gradient plane must be nonnegative")
        self.leaves.clear()
        self.leaves.append(FrontierLeaf(0, 0, n_active, depth=0))
        self.n_active = n_active
        self.next_node = 1
        self.n_leaves = 1
        self.max_leaves = max_leaves
        self.plane = plane

    def size(self) -> Int:
        return len(self.leaves)

    def leaf(self, slot: Int) raises -> FrontierLeaf:
        """One leaf, copied out. The frontier is a few hundred entries of a
        handful of scalars plus a branch list, so a copy here is cheaper to
        read than a reference with an origin, and the caller cannot mutate
        the frontier behind `apply_commit`'s back."""
        if slot < 0 or slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        return self.leaves[slot].copy()

    def set_candidate(
        mut self, slot: Int, var candidate: FrontierCandidate
    ) raises:
        """Record the candidate a search produced for `slot`. Refuses a
        candidate whose child counts do not add up to the leaf's own rows,
        which is the cheapest available check that the record came from the
        histogram of the leaf it is being filed under."""
        if slot < 0 or slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        if candidate.state == CAND_READY:
            if candidate.total_rows() != self.leaves[slot].row_count:
                raise Error(
                    "candidate child counts do not sum to the leaf's rows"
                )
        self.leaves[slot].candidate = candidate^

    def mark_pending(mut self, slot: Int) raises:
        if slot < 0 or slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        self.leaves[slot].candidate.state = CAND_PENDING

    def assign_slot(mut self, slot: Int, hist_slot: Int) raises:
        if slot < 0 or slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        self.leaves[slot].hist_slot = hist_slot

    def mark_partitioned(mut self, slot: Int) raises:
        if slot < 0 or slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        self.leaves[slot].partitioned = True

    # --- Statistics -------------------------------------------------------

    def set_stats(mut self, slot: Int, stats: LeafStats) raises:
        """File the gradient and hessian sums a search (or a subtraction)
        produced for `slot`.

        Refuses a record whose count disagrees with the leaf's own rows, for
        the same reason `set_candidate` refuses a mismatched child sum: it is
        the cheapest available check that the statistics came from the leaf
        they are being filed under, and a silently misfiled parent would make
        every sibling derived from it wrong.
        """
        if slot < 0 or slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        if stats.count != self.leaves[slot].row_count:
            raise Error(
                "leaf statistics do not cover the leaf's rows"
            )
        self.leaves[slot].stats = stats.copy()
        self.leaves[slot].stats_known = True

    def stats_of(self, slot: Int) raises -> LeafStats:
        if slot < 0 or slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        return self.leaves[slot].stats.copy()

    def stats_known(self, slot: Int) raises -> Bool:
        if slot < 0 or slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        return self.leaves[slot].stats_known

    def set_sibling_stats(
        mut self, parent_stats: LeafStats, built_slot: Int, derived_slot: Int
    ) raises:
        """Derive one child's statistics from the parent's and the other
        child's, host side.

        The bookkeeping half of the subtraction trick, kept next to the
        histogram half in `gpu_leaf_batching.enqueue_subtract` so the two
        cannot drift: whichever child was built, the other's sums are the
        parent's minus it. The count is checked rather than subtracted blind,
        which catches a parent filed against the wrong leaf.
        """
        var built = self.stats_of(built_slot)
        if not self.leaves[built_slot].stats_known:
            raise Error(
                "the built child's statistics must be known before its"
                " sibling is derived from them"
            )
        var derived = parent_stats.subtract(built)
        self.set_stats(derived_slot, derived)

    def slot_of_node(self, node: Int) -> Int:
        """The frontier slot holding `node`, or -1. Node ids are unique across
        a frontier (`check_invariants` holds it), so the answer is unique."""
        for i in range(len(self.leaves)):
            if self.leaves[i].node == node:
                return i
        return -1

    # --- Completion -------------------------------------------------------

    def budget_left(self) -> Int:
        """Leaves this tree may still create, or -1 when unbounded."""
        if self.max_leaves <= 0:
            return -1
        var left = self.max_leaves - self.n_leaves
        if left < 0:
            return 0
        return left

    def status(self) -> Int:
        """Why this frontier can or cannot take another commit.

        The order of the tests is the order a grower's loop asks them in:
        the budget first (a spent budget stops growth whatever the candidates
        say), then work in flight (a pending candidate makes `select_best`
        premature), then whether any candidate is ready at all.
        """
        if self.max_leaves > 0 and self.n_leaves >= self.max_leaves:
            return FRONTIER_BUDGET_SPENT
        for i in range(len(self.leaves)):
            if self.leaves[i].candidate.needs_work():
                return FRONTIER_WORK_PENDING
            if self.leaves[i].candidate.is_pending():
                return FRONTIER_WORK_PENDING
        if self.select_best() < 0:
            return FRONTIER_NO_CANDIDATE
        return FRONTIER_GROWING

    def is_complete(self) -> Bool:
        """Whether this tree is finished: no budget left, or nothing left to
        split. Work still in flight is not completion."""
        var s = self.status()
        return s == FRONTIER_BUDGET_SPENT or s == FRONTIER_NO_CANDIDATE

    def select_best(self) -> Int:
        """The leaf that may be committed next, or -1 when none may be.

        Byte for byte the trainer's rule: a ready candidate with a gain
        strictly greater than the best seen so far wins, the scan runs in
        ascending slot order, and the initial best gain is 0.0, so a
        nonpositive gain never splits anything. Ties therefore go to the
        lowest slot, which is the earliest created leaf.
        """
        var best = -1
        var best_gain = 0.0
        for i in range(len(self.leaves)):
            if (
                self.leaves[i].candidate.state == CAND_READY
                and self.leaves[i].candidate.split.gain > best_gain
            ):
                best_gain = self.leaves[i].candidate.split.gain
                best = i
        return best

    def has_pending(self) -> Bool:
        for i in range(len(self.leaves)):
            if self.leaves[i].candidate.is_pending():
                return True
        return False

    def pending(self) -> List[Int]:
        """Slots whose candidate has never been computed, ascending.

        This is the work list a batch may cover, and it is the whole of it:
        a leaf that has been searched needs nothing, whatever its gain, and a
        leaf whose work is already in flight must not be enqueued twice.
        Ascending order is what makes a batch's item order, and therefore its
        result order, a function of the frontier alone.
        """
        var out = List[Int]()
        for i in range(len(self.leaves)):
            if self.leaves[i].candidate.needs_work():
                out.append(i)
        return out^

    def batch_slots(self, max_items: Int) raises -> List[Int]:
        """The first `max_items` slots a batch may cover, ascending.

        `pending()` capped at what one launch holds, and the cap is taken from
        the front rather than by any ranking, so the batch a frontier offers
        is a function of the frontier alone and two runs of the same tree
        offer the same batch. A grower that wants the *best* leaves in a
        bounded batch ranks them with `speculative_order` and passes that
        list to `work_items` itself; this is the plain, order-free answer.
        """
        if max_items < 1:
            raise Error("a batch holds at least one item")
        var out = List[Int]()
        for i in range(len(self.leaves)):
            if len(out) >= max_items:
                break
            if self.leaves[i].candidate.needs_work():
                out.append(i)
        return out^

    def work_items(
        self, slots: List[Int], plane: Int = -1
    ) raises -> List[LeafWorkItem]:
        """Turn frontier slots into batch entries, in the given order.

        `out_slot` is left as the leaf's current `hist_slot`; a caller that
        allocates slots from a pool overwrites it before launching, which
        `gpu_leaf_batching.assign_batch_slots` is the one way to do without
        restating the pool's rules. A leaf still holding `NO_SLOT` therefore
        produces an item with `out_slot == NO_SLOT`, which `plan_batch`
        refuses outright rather than launching a write to slot -1.

        `plane` defaults to -1, meaning this frontier's own `plane`, which is
        the class index in a multiclass round. Passing one explicitly is for a
        caller assembling a batch that spans classes.

        Refuses a slot list with a repeat, since two entries writing one
        output slice is the one way a batch could corrupt a histogram.
        """
        var p = plane
        if p < 0:
            p = self.plane
        var out = List[LeafWorkItem](capacity=len(slots))
        for i in range(len(slots)):
            var s = slots[i]
            if s < 0 or s >= len(self.leaves):
                raise Error("frontier slot out of range")
            for k in range(i):
                if slots[k] == s:
                    raise Error("a batch may not hold a slot twice")
            out.append(
                LeafWorkItem(
                    s,
                    self.leaves[s].node,
                    self.leaves[s].row_begin,
                    self.leaves[s].row_count,
                    self.leaves[s].hist_slot,
                    p,
                )
            )
        return out^

    def plan_commit(
        self,
        slot: Int,
        monotone_signs: List[Int] = [],
        missing_bin: Int = -1,
    ) raises -> CommitPlan:
        """Everything committing `slot` implies, without mutating anything.

        Node ids are the two `Tree._add_node` would assign next, left first.
        The child values are the candidate's raw values put through the
        parent's interval and then, if the split feature is constrained and
        rounding inverted them, collapsed to their midpoint. That is the same
        clamp-and-divide both growers apply, reproduced here so a batched
        grower cannot drift from them.

        `missing_bin` is the split feature's missing bin, which the caller
        reads out of `BinnedMatrix.missing_bin` (or
        `GpuHistogramBuilder.missing_bin`) and which a categorical split
        ignores. It rides on the plan so the device partition can build its
        own `RowRouting` from it; -1, the default, is the "this feature has no
        missing bin" value every other routing site already uses.
        """
        if slot < 0 or slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        var leaf = self.leaves[slot].copy()
        var cand = leaf.candidate.copy()
        if cand.state != CAND_READY:
            raise Error("only a leaf with a ready candidate may be committed")
        if cand.total_rows() != leaf.row_count:
            raise Error("candidate child counts do not sum to the leaf's rows")

        var split = cand.split.copy()
        var sign = monotone_sign(monotone_signs, split.feature)
        var left_value = leaf.bounds.clamp(cand.left_value)
        var right_value = leaf.bounds.clamp(cand.right_value)
        if sign != MONOTONE_FREE and left_value > right_value:
            var mid = midpoint(left_value, right_value)
            left_value = mid
            right_value = mid
        var bounds = child_bounds(leaf.bounds, sign, left_value, right_value)

        return CommitPlan(
            slot,
            leaf.node,
            self.next_node,
            self.next_node + 1,
            leaf.row_begin,
            cand.n_left,
            leaf.row_begin + cand.n_left,
            cand.n_right,
            left_value,
            right_value,
            leaf.depth + 1,
            subtraction_builds_left(cand.n_left, cand.n_right),
            extend_branch(leaf.branch, split.feature),
            bounds^,
            split^,
            -1 if cand.split.is_categorical else missing_bin,
        )

    def apply_commit(mut self, plan: CommitPlan) raises:
        """Move the frontier by one split.

        The left child takes the parent's slot and the right child is
        appended, which is the trainer's convention and therefore the
        tie-breaking order every later `select_best` depends on. Both
        children start with no candidate and no histogram slot, so they land
        in `pending()` for the next batch.
        """
        if plan.slot < 0 or plan.slot >= len(self.leaves):
            raise Error("frontier slot out of range")
        if plan.parent_node != self.leaves[plan.slot].node:
            raise Error("commit plan was made for a different leaf")
        if plan.left_node != self.next_node:
            raise Error("commit plan node ids are stale")
        if plan.left_count + plan.right_count != self.leaves[
            plan.slot
        ].row_count:
            raise Error("commit plan does not cover the parent's rows")

        self.leaves[plan.slot] = FrontierLeaf(
            plan.left_node,
            plan.left_begin,
            plan.left_count,
            plan.child_depth,
            plan.branch.copy(),
            plan.bounds.left.copy(),
        )
        self.leaves.append(
            FrontierLeaf(
                plan.right_node,
                plan.right_begin,
                plan.right_count,
                plan.child_depth,
                plan.branch.copy(),
                plan.bounds.right.copy(),
            )
        )
        self.next_node += 2
        self.n_leaves += 1

    def total_rows(self) -> Int:
        var total = 0
        for i in range(len(self.leaves)):
            total += self.leaves[i].row_count
        return total

    def max_rows(self) -> Int:
        var m = 0
        for i in range(len(self.leaves)):
            if self.leaves[i].row_count > m:
                m = self.leaves[i].row_count
        return m

    def min_rows(self) -> Int:
        if len(self.leaves) == 0:
            return 0
        var m = self.leaves[0].row_count
        for i in range(1, len(self.leaves)):
            if self.leaves[i].row_count < m:
                m = self.leaves[i].row_count
        return m

    def row_counts(self) -> List[Int]:
        var out = List[Int](capacity=len(self.leaves))
        for i in range(len(self.leaves)):
            out.append(self.leaves[i].row_count)
        return out^

    def check_invariants(self) raises:
        """The live leaves must tile `[0, n_active)` and hold distinct node
        ids and distinct histogram slots.

        The tiling check is the host mirror of
        `LeafRangeTable.check_invariants`, and holding both is what makes a
        disagreement between the frontier and the device range table
        detectable instead of silent. Distinct output slots is the batching
        invariant: two leaves sharing one slot would have one histogram
        overwrite the other inside a single launch.
        """
        var total = 0
        for i in range(len(self.leaves)):
            if self.leaves[i].row_count < 0:
                raise Error("a leaf holds a negative row count")
            if self.leaves[i].stats.count != self.leaves[i].row_count:
                raise Error(
                    "a leaf's statistics do not cover its rows"
                )
            if (
                self.leaves[i].row_begin < 0
                or self.leaves[i].row_end() > self.n_active
            ):
                raise Error("a leaf's row range escapes the active prefix")
            total += self.leaves[i].row_count
            for k in range(i + 1, len(self.leaves)):
                if self.leaves[i].node == self.leaves[k].node:
                    raise Error("two frontier leaves share a node id")
                if (
                    self.leaves[i].hist_slot != NO_SLOT
                    and self.leaves[i].hist_slot == self.leaves[k].hist_slot
                ):
                    raise Error("two frontier leaves share a histogram slot")
                if self.leaves[i].row_count > 0 and self.leaves[
                    k
                ].row_count > 0:
                    if (
                        self.leaves[i].row_begin < self.leaves[k].row_end()
                        and self.leaves[k].row_begin
                        < self.leaves[i].row_end()
                    ):
                        raise Error("two frontier leaves overlap")
        if total != self.n_active:
            raise Error("the frontier does not cover the active prefix")
        if self.n_leaves != len(self.leaves):
            raise Error("leaf count disagrees with the frontier length")


# --- Speculation ----------------------------------------------------------


def search_is_order_free(feature_fraction_bynode: Float64) -> Bool:
    """Whether a leaf's *candidate*, not just its histogram, is independent
    of the commit order.

    A histogram always is (the order-independence lemma). A candidate is too
    exactly when the split search does not read the node id, which is when
    `feature_fraction_bynode` is 1.0 and `select_node_features` returns the
    tree's set unchanged. Below 1.0 the per-node draw hangs off the node id,
    which commit order decides, so a speculatively searched child has to be
    re-searched if the order it assumed did not hold.

    This is the one predicate that decides how much of a batch is free. With
    it true, a whole batch of histograms and searches is safe. With it false,
    the histograms are still safe and only the searches have to follow the
    commits.
    """
    return feature_fraction_bynode >= 1.0


def speculative_order(
    frontier: LeafFrontier, max_commits: Int
) raises -> List[Int]:
    """The slots a batch would cover, in the order best-first would take them
    if no child ever outranked a leaf already on the list.

    Ready candidates sorted by descending gain, ties by ascending slot, which
    is the order repeated `select_best` calls would produce if the frontier
    never gained a leaf. Selection sort over a frontier of at most a few
    hundred leaves, chosen so the tie rule is written out rather than
    delegated to a comparator's stability.

    The list is a prediction, not a decision. `verify_speculation` is what
    turns each entry into a commit or a miss.
    """
    if max_commits < 0:
        raise Error("speculation depth must be nonnegative")
    var ranked = List[Int]()
    var taken = List[Bool](capacity=frontier.size())
    for _ in range(frontier.size()):
        taken.append(False)
    while len(ranked) < max_commits:
        var best = -1
        var best_gain = 0.0
        for i in range(frontier.size()):
            if taken[i]:
                continue
            if (
                frontier.leaves[i].candidate.state == CAND_READY
                and frontier.leaves[i].candidate.split.gain > best_gain
            ):
                best_gain = frontier.leaves[i].candidate.split.gain
                best = i
        if best < 0:
            break
        taken[best] = True
        ranked.append(best)
    return ranked^


def verify_speculation(frontier: LeafFrontier, expected_slot: Int) -> Bool:
    """Whether the next commit really is the slot speculation predicted.

    The only honest check is the serial one: ask the frontier, as it stands
    now, which leaf best-first would take, and compare. A batched grower
    calls this before every commit in a speculative run and stops the run at
    the first False, which is what keeps the committed sequence identical to
    the serial one rather than merely close to it.
    """
    return frontier.select_best() == expected_slot


struct SpeculationLedger(Copyable, Movable):
    """What a speculative run actually did, for a benchmark to report.

    `hits` and `misses` are commits that were and were not predicted.
    `wasted_histograms` counts histograms built for leaves the tree never
    split, which is the only genuinely lost work in a speculative run, and it
    is only lost when `num_leaves` cuts growth short. `redundant_partitions`
    counts partitions enqueued for a leaf already partitioned, which the
    idempotence of the stable partition makes harmless and merely costly.
    """

    var batches: Int
    var items: Int
    var hits: Int
    var misses: Int
    var wasted_histograms: Int
    var redundant_partitions: Int

    def __init__(out self):
        self.batches = 0
        self.items = 0
        self.hits = 0
        self.misses = 0
        self.wasted_histograms = 0
        self.redundant_partitions = 0

    def note_batch(mut self, n_items: Int):
        self.batches += 1
        self.items += n_items

    def note_commit(mut self, predicted: Bool):
        if predicted:
            self.hits += 1
        else:
            self.misses += 1

    def commits(self) -> Int:
        return self.hits + self.misses

    def items_per_batch(self) -> Float64:
        if self.batches == 0:
            return 0.0
        return Float64(self.items) / Float64(self.batches)

    def hit_rate(self) -> Float64:
        var total = self.commits()
        if total == 0:
            return 0.0
        return Float64(self.hits) / Float64(total)

    def report(self) -> String:
        var out = String("batches ") + String(self.batches) + "\n"
        out += "items " + String(self.items) + "\n"
        out += "items_per_batch " + String(self.items_per_batch()) + "\n"
        out += "hits " + String(self.hits) + "\n"
        out += "misses " + String(self.misses) + "\n"
        out += "hit_rate " + String(self.hit_rate()) + "\n"
        out += "wasted_histograms " + String(self.wasted_histograms) + "\n"
        out += (
            "redundant_partitions " + String(self.redundant_partitions) + "\n"
        )
        return out


# --- How many leaves a grower can actually offer --------------------------
#
# The number below is the whole feasibility question for this lane, so it is
# computed rather than asserted. A batching primitive is worth nothing if the
# grower above it never has two leaves to hand it at once.

comptime FEEDER_HOST_SEARCH = 0
"""`grow_tree_gpu`'s default path. One child is built per commit and the
sibling comes from the host subtraction, so a batch is one leaf and batching
is a no-op."""

comptime FEEDER_DEVICE_SEARCH = 1
"""`_grow_tree_gpu_device_search`. Both children are built on the device, so
every commit offers exactly two leaves."""

comptime FEEDER_SPECULATIVE = 2
"""A frontier driven by `speculative_order`. Every commit on the speculation
list offers its children, so a depth-k speculation offers up to 2k leaves,
bounded by the frontier size and the histogram slot pool."""

comptime FEEDER_LEVELWISE = 3
"""Level-wise growth (a separate lane). A level offers all of its leaves at
once, which is the largest batch any grower can produce and the one this
module's device half is shaped for."""


def leaves_per_launch(
    feeder: Int, frontier_size: Int, speculation_depth: Int
) raises -> Int:
    """Leaves one launch can cover under each grower, at a frontier of
    `frontier_size` live leaves.

    Deliberately arithmetic and not a policy. It answers "is there anything
    to batch here", which is the question that has to be answered before any
    benchmark of the kernels below means anything.
    """
    if frontier_size < 1:
        raise Error("a frontier holds at least one leaf")
    if speculation_depth < 1:
        raise Error("speculation depth must be at least one commit")
    if feeder == FEEDER_HOST_SEARCH:
        return 1
    if feeder == FEEDER_DEVICE_SEARCH:
        return 2
    if feeder == FEEDER_SPECULATIVE:
        var offered = 2 * speculation_depth
        if offered > frontier_size:
            return frontier_size
        return offered
    if feeder == FEEDER_LEVELWISE:
        return frontier_size
    raise Error("unknown frontier feeder")


def feeder_name(feeder: Int) -> String:
    if feeder == FEEDER_HOST_SEARCH:
        return String("host_search")
    if feeder == FEEDER_DEVICE_SEARCH:
        return String("device_search")
    if feeder == FEEDER_SPECULATIVE:
        return String("speculative")
    if feeder == FEEDER_LEVELWISE:
        return String("levelwise")
    return String("unknown")
