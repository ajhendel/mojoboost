"""Growth policy: which leaf a grower splits next.

Every grower that keeps a frontier (`tree.grow_tree`,
`tree_sparse.grow_tree_sparse`, and the three loops in `train_gpu.mojo`)
holds its leaves in a list in which a split replaces the parent's slot with
the left child and appends the right child. Each round it describes that
frontier as a list of `LeafCandidate` (node id, depth, best gain, whether a
positive-gain split was found) and asks `GrowthSchedule.next_leaf` for the
slot to split. Everything after the pick (the partition, the child
histograms with sibling subtraction, the split search, the leaf values, the
constraints) is the grower's own per-split body and is the same under both
policies. This module touches no histogram, no device, and no dataset, so
the rules below can be read and reproduced on the host without a GPU.

Two policies, selected by `TreeParams.grow_policy`:

`GROW_LEAFWISE` (default) is LightGBM's growth: one global priority queue
over every live leaf, the best gain anywhere in the tree is split next, ties
go to the lower frontier slot. That is the strict `>` scan every grower ran
before this module existed; the slot order is an artifact of the list (left
child in the parent's slot, right child appended) and it is kept because it
is what every fit before this parameter produced.

`GROW_DEPTHWISE` is XGBoost's `grow_policy=depthwise`: every admitted split
of one depth is committed before any deeper one, so leaves fill level by
level. The parameter name and both spellings are XGBoost's, `lossguide`
being accepted as an alias for leaf-wise. It reaches `params.mojo` as
`grow_policy`, the Python estimators as `grow_policy=`, and the C API
through the parameter string. The tree that comes out is a different tree,
not a faster route to the same one; LightGBM has no such switch and no
parity claim is made. A tree it produces is an ordinary `tree.Tree` and
carries no record of the mode that grew it; the distributed prototype tracks
no depth and rejects the policy.

The three rules that make a depth-wise tree well defined
--------------------------------------------------------
1. **Leaf budget.** Committing a split turns one leaf into two, so it adds
   exactly one leaf. A level offering `E` eligible splits would add `E`
   leaves at once, which can overshoot `num_leaves` in a way leaf-wise
   growth never can. `admit_level` resolves that: `num_leaves` stays a hard
   bound and the last level is admitted as a gain-ranked prefix (or refused
   whole, under `BUDGET_WHOLE_LEVEL`). Growth therefore never produces more
   leaves than `num_leaves`, which is the property model size,
   serialization, and every existing parity statement rest on.

2. **Deterministic order.** Admission ranks siblings against each other, so
   it needs a total order on candidates: gain descending, then node id
   ascending. Node ids are assigned breadth first, left child before right,
   in ascending parent id order, so the order is a function of the tree
   alone and does not depend on container mechanics, launch scheduling, or
   thread count. This is deliberately not the leaf-wise tie rule; that rule
   is correct for a queue popped one at a time and does not survive ranking
   a whole level.

3. **Shape rules first.** `max_depth` and `min_data_in_leaf` are enforced
   where they always were, in `tree._search`, so a leaf past either offers
   no split and is terminal under both policies; the schedule only ever sees
   the result.

What this module does not decide
--------------------------------
Split search itself. Gain, the min-child-hessian test, the per-child
`min_data_in_leaf` test, monotone candidate rejection, interaction allow
masks, categorical partitioning, and missing-value direction all stay in
`split.mojo` and `categorical.mojo`, reached through `tree._search`, so
both policies enforce every one of them identically. A policy changes
*which leaf is split when*, never how a candidate is scored.

What a batched level is, and what it is not
-------------------------------------------
`plan_level` below hands a grower the whole planned level at once instead of
one index at a time. It decides nothing new: it calls `next_leaf` and
returns what `next_leaf` returned, so the admissions, the budget, and the
order are still this module's and a grower that drains the list splits
exactly what a grower calling `next_leaf` between splits would have.

What it is for is the GPU. `train_gpu._device_search_resident` enqueues a
level's partitions and histogram builds back to back and searches every
child in one launch pair, so a level costs one host wait rather than one per
split. That is a launch-count change underneath this order, which is where
`docs/design/GPU_LEVELWISE.md` says batching belongs. The batched histogram
build of that design (one pass over the active row buffer per level, rather
than one per node) is still not built; the multi-leaf kernels it would use
are in `gpu_leaf_batching.mojo`. Its host-side prototype module
(`gpu_levelwise.mojo`, a second commit path that derived child values from
the parent histogram) was removed unused: the growers commit through their
own per-split bodies, and a batched build belongs underneath this order as
a histogram-phase service both policies can call, not as another growth
loop.

Leaf-wise growth cannot batch and is not asked to. Its next pick depends on
the frontier the current split changes, so `plan_level` returns one index
and the grower is the one-split-at-a-time loop it always was.
"""


# `TreeParams.grow_policy`. Leaf-wise is LightGBM's growth and the default;
# depth-wise commits every admitted split of one depth before any deeper one.
comptime GROW_LEAFWISE = 0
comptime GROW_DEPTHWISE = 1


def parse_grow_policy(name: String) raises -> Int:
    """The policy code for a name. XGBoost's spellings: `depthwise`, and
    `lossguide` for leaf-wise; `leafwise` and `leaf_wise`/`depth_wise` are
    accepted so the LightGBM-flavored spelling works too."""
    if name == "leafwise" or name == "leaf_wise" or name == "lossguide":
        return GROW_LEAFWISE
    if name == "depthwise" or name == "depth_wise":
        return GROW_DEPTHWISE
    raise Error(
        "grow_policy must be 'leafwise' (alias 'lossguide') or 'depthwise',"
        " got '",
        name,
        "'",
    )


def grow_policy_name(policy: Int) -> String:
    if policy == GROW_LEAFWISE:
        return String("leafwise")
    if policy == GROW_DEPTHWISE:
        return String("depthwise")
    return String("unknown")


def check_grow_policy(policy: Int) raises:
    if policy != GROW_LEAFWISE and policy != GROW_DEPTHWISE:
        raise Error(
            "grow_policy must be GROW_LEAFWISE (0) or GROW_DEPTHWISE (1),"
            " got ",
            policy,
        )




#
# RANK admits the highest-gain prefix of the level, so `num_leaves` is met
# exactly whenever the tree has that many eligible splits to give. The last
# level is then partial and its membership is gain driven, which is the one
# place level-wise growth ranks siblings against each other.
#
# WHOLE_LEVEL refuses a level it cannot admit entirely, so every level of the
# tree is complete and the leaf count lands on whatever the last full level
# produced (at most `num_leaves`, often well under it). It is the stricter
# reading of "level-wise" and the one to reach for when the point of the run
# is that every leaf sits at the same depth.
comptime BUDGET_RANK = 0
comptime BUDGET_WHOLE_LEVEL = 1

# Why growth stopped. Reported for the level just finished, so a grower can
# record it without reconstructing the reason from the tree.
comptime STOP_RUNNING = 0
comptime STOP_LEAF_BUDGET = 1
comptime STOP_MAX_DEPTH = 2
comptime STOP_DRY_LEVEL = 3
comptime STOP_LEVEL_CAP = 4


@fieldwise_init
struct LeafCandidate(Copyable, Movable, Writable):
    """One frontier leaf's offer, as the growers describe it to the schedule.

    `node` is its tree node id, `depth` its depth in edges from the root,
    `gain` the gain of the best split found for it, and `eligible` whether
    that split may be taken at all. A candidate is eligible only when a
    split was found and its gain is strictly positive (the bar the leaf-wise
    scan always set by starting its best gain at 0.0). An ineligible
    candidate is terminal: it becomes a leaf at its own depth.
    """

    var node: Int
    var depth: Int
    var gain: Float64
    var eligible: Bool

    @staticmethod
    def terminal(node: Int, depth: Int) -> LeafCandidate:
        """A leaf offering nothing, which is what a leaf past the depth
        limit, under the row minimum, or with no positive-gain split is."""
        return LeafCandidate(node, depth, 0.0, False)

    def write_to(self, mut writer: Some[Writer]):
        if not self.eligible:
            writer.write(
                "LeafCandidate(node=",
                self.node,
                ", depth=",
                self.depth,
                ", terminal)",
            )
        else:
            writer.write(
                "LeafCandidate(node=",
                self.node,
                ", depth=",
                self.depth,
                ", gain=",
                self.gain,
                ")",
            )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def count_eligible(candidates: List[LeafCandidate]) -> Int:
    var n = 0
    for i in range(len(candidates)):
        if candidates[i].eligible:
            n += 1
    return n


def rank_level(candidates: List[LeafCandidate]) -> List[Int]:
    """Positions into `candidates` of every eligible candidate, best first.

    The order is gain descending, then node id ascending. Both keys are
    needed: gain alone leaves ties to whatever order the frontier happened to
    be built in, and node id alone would ignore gain entirely. Ties on gain
    are common in practice, not a corner case, because two siblings split on
    the same feature over identical bin totals score identically.

    Selection sort. A level holds at most `num_leaves` nodes, which is 31 by
    default and a few hundred at the top of the useful range, so the
    quadratic cost is beneath measurement. It is also the reading that makes
    the tie rule checkable by eye, which is worth more here than an
    asymptote, and it appends in final order rather than shifting, so the
    result is built once.
    """
    var taken = List[Bool](capacity=len(candidates))
    taken.resize(len(candidates), False)
    var remaining = 0
    for i in range(len(candidates)):
        if candidates[i].eligible:
            remaining += 1

    var order = List[Int](capacity=remaining)
    while len(order) < remaining:
        var best = -1
        for i in range(len(candidates)):
            if taken[i] or not candidates[i].eligible:
                continue
            if best < 0:
                best = i
                continue
            var gain = candidates[i].gain
            var best_gain = candidates[best].gain
            if gain > best_gain:
                best = i
            elif gain == best_gain and candidates[i].node < (
                candidates[best].node
            ):
                best = i
        # `remaining` counted the eligible candidates, so a pass can only
        # fail to find one if the list changed underneath, which it cannot.
        # The guard is here so a wiring mistake ends the loop instead of
        # indexing with -1.
        if best < 0:
            break
        taken[best] = True
        order.append(best)
    return order^


def leaf_budget(n_leaves: Int, num_leaves: Int) -> Int:
    """How many more splits the tree may commit. Each split converts one leaf
    into two, so it spends exactly one unit of budget whatever the level."""
    var room = num_leaves - n_leaves
    return room if room > 0 else 0


def admit_level(
    candidates: List[LeafCandidate],
    n_leaves: Int,
    num_leaves: Int,
    budget_mode: Int = BUDGET_RANK,
) raises -> List[Bool]:
    """Which of a level's candidates are committed, as a mask parallel to
    `candidates`.

    Under BUDGET_RANK this is the highest-gain prefix that fits in the
    remaining budget, taken in `rank_level` order. Under BUDGET_WHOLE_LEVEL a
    level that does not fit entirely is refused entirely, which ends growth
    with the last complete level in place.

    The mask says nothing about the order children are created in. Node ids
    are assigned in ascending parent id order regardless of how the gains
    ranked, so the ranking decides membership and never layout.
    """
    if budget_mode != BUDGET_RANK and budget_mode != BUDGET_WHOLE_LEVEL:
        raise Error(
            "levelwise budget_mode must be BUDGET_RANK or BUDGET_WHOLE_LEVEL"
        )
    var admitted = List[Bool](capacity=len(candidates))
    admitted.resize(len(candidates), False)
    var budget = leaf_budget(n_leaves, num_leaves)
    if budget <= 0:
        return admitted^

    var order = rank_level(candidates)
    if budget_mode == BUDGET_WHOLE_LEVEL:
        if len(order) > budget:
            return admitted^
        for k in range(len(order)):
            admitted[order[k]] = True
        return admitted^

    var take = len(order) if len(order) < budget else budget
    for k in range(take):
        admitted[order[k]] = True
    return admitted^


def count_admitted(admitted: List[Bool]) -> Int:
    var n = 0
    for i in range(len(admitted)):
        if admitted[i]:
            n += 1
    return n


def leaves_after_level(n_leaves: Int, n_admitted: Int) -> Int:
    """Leaf count once a level's admitted splits are applied. One split, one
    leaf: the parent leaves the frontier and two children join it."""
    return n_leaves + n_admitted


def level_stop_reason(
    depth: Int,
    max_depth: Int,
    n_leaves_after: Int,
    num_leaves: Int,
    n_eligible: Int,
    n_admitted: Int,
) -> Int:
    """Why growth stops after the level at `depth`, or STOP_RUNNING when it
    does not. `depth` is the depth of the level just committed, so its
    children sit at `depth + 1`.

    Precedence, most specific first:

      STOP_DRY_LEVEL    the level offered nothing, so there are no children
                        and nothing deeper can exist
      STOP_LEVEL_CAP    the level had eligible splits and none were admitted,
                        which under BUDGET_WHOLE_LEVEL means the level did
                        not fit and growth ends on the last complete one
      STOP_LEAF_BUDGET  `num_leaves` is now spent
      STOP_MAX_DEPTH    the children sit at the depth limit and can offer no
                        split of their own
    """
    if n_eligible == 0:
        return STOP_DRY_LEVEL
    if n_admitted == 0:
        return STOP_LEVEL_CAP
    if n_leaves_after >= num_leaves:
        return STOP_LEAF_BUDGET
    if max_depth > 0 and depth + 1 >= max_depth:
        return STOP_MAX_DEPTH
    return STOP_RUNNING


def stop_reason_name(reason: Int) -> String:
    if reason == STOP_RUNNING:
        return String("running")
    if reason == STOP_LEAF_BUDGET:
        return String("leaf budget")
    if reason == STOP_MAX_DEPTH:
        return String("max depth")
    if reason == STOP_DRY_LEVEL:
        return String("dry level")
    if reason == STOP_LEVEL_CAP:
        return String("level cap")
    return String("unknown")

struct GrowthSchedule(Movable):
    """The order a grower splits its frontier in, under either policy.

    Every grower keeps a frontier list in which a split replaces the parent's
    slot with the left child and appends the right child, so a slot that has
    not been split keeps its index while other slots are split. The schedule
    relies on that: it answers in frontier indices, and under depth-wise
    growth it plans one level at a time as a queue of indices handed out one
    per call, which stay valid across the splits in between.

    Leaf-wise: the eligible candidate with the highest gain, ties to the
    lower frontier index. No state is kept between calls.

    Depth-wise: a level is planned when the queue is empty. Its depth is the
    smallest depth of any eligible leaf; every eligible leaf at that depth is
    a candidate, `admit_level` decides which are committed under the leaf
    budget, and the admitted ones are queued in ascending node id order,
    which is what makes child ids breadth first (`rank_level` decides
    membership, never layout). Leaves at that depth that were not admitted
    are terminal, and so are eligible leaves at a shallower depth, which can
    only exist once the budget is spent. `stop_reason` records why the last
    planned level ended growth, for tracing and tests.

    Forced splits are outside this order: a grower applies them before it
    asks, so by the time the first level is planned the forced tree is
    exhausted and every frontier leaf owes nothing.
    """

    var policy: Int
    var budget_mode: Int
    var queue: List[Int]
    var next: Int
    var level: Int
    var stop_reason: Int

    def __init__(
        out self, policy: Int = GROW_LEAFWISE, budget_mode: Int = BUDGET_RANK
    ):
        self.policy = policy
        self.budget_mode = budget_mode
        self.queue = List[Int]()
        self.next = 0
        self.level = -1
        self.stop_reason = STOP_RUNNING

    def next_leaf(
        mut self,
        candidates: List[LeafCandidate],
        n_leaves: Int,
        num_leaves: Int,
        max_depth: Int,
    ) raises -> Int:
        """The frontier index to split next, or -1 when the tree is
        finished. `candidates` is parallel to the grower's frontier."""
        if self.policy == GROW_LEAFWISE:
            var best = -1
            var best_gain = 0.0
            for i in range(len(candidates)):
                if candidates[i].eligible and candidates[i].gain > best_gain:
                    best_gain = candidates[i].gain
                    best = i
            if best < 0:
                self.stop_reason = STOP_DRY_LEVEL
            return best
        if self.next < len(self.queue):
            var i = self.queue[self.next]
            self.next += 1
            return i
        # Plan the next level: the shallowest depth with anything to offer.
        var level = -1
        for i in range(len(candidates)):
            if candidates[i].eligible and (
                level < 0 or candidates[i].depth < level
            ):
                level = candidates[i].depth
        if level < 0:
            self.stop_reason = STOP_DRY_LEVEL
            return -1
        var at_level = List[LeafCandidate](capacity=len(candidates))
        for i in range(len(candidates)):
            if candidates[i].depth == level:
                at_level.append(candidates[i].copy())
            else:
                at_level.append(
                    LeafCandidate.terminal(
                        candidates[i].node, candidates[i].depth
                    )
                )
        var admitted = admit_level(
            at_level, n_leaves, num_leaves, self.budget_mode
        )
        var n_eligible = count_eligible(at_level)
        var n_admitted = count_admitted(admitted)
        self.level = level
        self.stop_reason = level_stop_reason(
            level,
            max_depth,
            leaves_after_level(n_leaves, n_admitted),
            num_leaves,
            n_eligible,
            n_admitted,
        )
        if n_admitted == 0:
            return -1
        # Ascending node id: repeatedly take the smallest admitted id not yet
        # queued. Levels are at most `num_leaves` wide, so quadratic is fine
        # and it keeps the rule checkable by eye, as `rank_level` does.
        self.queue = List[Int](capacity=n_admitted)
        self.next = 0
        var last_node = -1
        while len(self.queue) < n_admitted:
            var best = -1
            for i in range(len(at_level)):
                if not admitted[i] or at_level[i].node <= last_node:
                    continue
                if best < 0 or at_level[i].node < at_level[best].node:
                    best = i
            if best < 0:
                break
            last_node = at_level[best].node
            self.queue.append(best)
        var i = self.queue[self.next]
        self.next += 1
        return i

    def plan_level(
        mut self,
        candidates: List[LeafCandidate],
        n_leaves: Int,
        num_leaves: Int,
        max_depth: Int,
    ) raises -> List[Int]:
        """Every frontier index this schedule will hand out before it next
        has to look at the frontier, in `next_leaf`'s order.

        The batched reading of `next_leaf`, for a grower that wants to
        enqueue a level's device work back to back and wait once rather than
        wait per split. It is the same order and the same admissions: this
        calls `next_leaf` and returns what it returned, so a grower that
        drains the list splits exactly the leaves, in exactly the sequence,
        that a grower calling `next_leaf` between splits would have.

        The two policies differ in how much can be known ahead:

        - Leaf-wise picks the best gain *anywhere* in the tree, and a split
          changes the frontier it picks from, so nothing past the first pick
          is decided yet. The list is one index, and a grower batching over
          it is the one-split-at-a-time loop it always was.
        - Depth-wise plans a whole level in `next_leaf` (the admissions, the
          leaf budget, and the ascending-node-id order are all decided
          there, once, when the queue is empty) and then hands its queue out
          one index at a time. Those indices stay valid across the splits in
          between, because a split replaces its parent's frontier slot with
          the left child and appends the right, so this returns the level.

        An empty list means growth is finished, which is `next_leaf`
        answering -1; `stop_reason` records why.
        """
        var picks = List[Int]()
        var first = self.next_leaf(candidates, n_leaves, num_leaves, max_depth)
        if first < 0:
            return picks^
        picks.append(first)
        if self.policy == GROW_LEAFWISE:
            return picks^
        # Drain the level `next_leaf` just planned. Asking again once the
        # queue is spent would plan the *next* level against a frontier the
        # caller has not split yet, so the queue bound is the stopping rule
        # rather than a second -1.
        while self.next < len(self.queue):
            picks.append(
                self.next_leaf(candidates, n_leaves, num_leaves, max_depth)
            )
        return picks^
