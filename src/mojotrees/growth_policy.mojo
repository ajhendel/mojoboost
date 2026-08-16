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

Three policies, selected by `TreeParams.grow_policy`. Two of them are frontier
orders and share `GrowthSchedule`; the third is not an order at all and is
described at the end.

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

`GROW_OBLIVIOUS`, which is a different question
-----------------------------------------------
CatBoost's symmetric tree (`EGrowPolicy::SymmetricTree`). Growth is depth
first, but a level is not a set of independent decisions: ONE (feature,
threshold, missing direction) is searched across the whole level and applied
to every leaf of it, so a row's leaf is the bit pattern of its d outcomes and
the tree is symmetric by construction. Verified against
`catboost/private/libs/algo/greedy_tensor_search.cpp`
(`GreedyTensorSearchOblivious`, the `for curDepth < MaxDepth` loop that ends
in one `currentSplitTree.AddSplit(bestSplit)`) and
`catboost/private/libs/algo/index_calcer.cpp` (`splitWeight = 1 << depth`).

Three consequences, all of which this module has to state because they are
what makes the mode not fit `GrowthSchedule`:

1. There is no per-leaf best split to rank, so `rank_level`, `admit_level`
   and the leaf budget have nothing to decide. `GrowthSchedule.__init__`
   REFUSES the code rather than falling through to the depth-wise branch.
2. `max_depth` is the only bound. A level splits entirely or not at all, so
   `num_leaves` cannot be met exactly and is ignored (CatBoost instead
   overwrites `max_leaves` with `1 << depth` and raises if the user set a
   different value: `catboost/private/libs/options/catboost_options.cpp`,
   `"max_leaves option works only with lossguide tree growing"`; we cannot
   tell a defaulted `num_leaves` from an explicit one, so we ignore rather
   than raise, and say so here).
3. A leaf that cannot satisfy `min_data_in_leaf` or
   `min_sum_hessian_in_leaf` at the chosen candidate is split anyway,
   contributing zero to the candidate's summed gain instead of vetoing it.
   `SharedSplitAudit` below records how many leaves that was. **That rule is
   ours and is NOT verified from CatBoost source**: CatBoost's symmetric
   mode has neither parameter, so it never had to define the case. We match
   the GPU device, because host and device must grow the same tree. See
   `CATBOOST_CATALOG.md`, "The per-leaf min-child rule is OURS".

Leaf numbering and node-id order are a cross-backend contract. **Leaf index
is the bit pattern of a row's outcomes with the FIRST level's outcome as the
LEAST significant bit** (`index_calcer.cpp`:
`const ui32 splitWeight = 1 << splitParams.Depth;`), so a left child keeps
its parent's index and a right child at level d adds `1 << d`. **Node ids
are assigned level by level, over the level's leaves in ascending LEAF
INDEX, left child before right** -- which is not ascending node id, since at
level 2 the leaves in node-id order carry indices 0, 2, 1, 3. Host and
device must agree on both or a node-identity test between them fails on
trees that are each correct.

The tree that comes out is an ordinary `tree.Tree` with the same split
repeated across a level, NOT the flat "splits per depth plus a 2^d leaf
table" that CatBoost stores (`catboost/libs/model/model.h`: `TreeSplits`,
`TreeSizes`, `TreeStartOffsets`, `GetLeafValues()[leafId]`). The symmetry is
a property of how the tree was grown, not of how it is stored, and keeping
the representation ordinary is what leaves predict, dump, serialization,
model I/O, monotone constraints and interaction constraints untouched.
"""

from std.os import getenv


# `TreeParams.grow_policy`. Leaf-wise is LightGBM's growth and the default;
# depth-wise commits every admitted split of one depth before any deeper one;
# oblivious is CatBoost's symmetric tree and is not a frontier order at all
# (see `OBLIVIOUS_IS_NOT_A_FRONTIER_ORDER` below).
comptime GROW_LEAFWISE = 0
comptime GROW_DEPTHWISE = 1
comptime GROW_OBLIVIOUS = 2

# `max_depth` is the ONLY bound on an oblivious tree's size: `num_leaves` does
# not bind (see `check_oblivious_params`), so depth d costs 2^(d+1) - 1 nodes
# unconditionally. 16 puts the ceiling at 131071 nodes per tree, which is the
# same depth ceiling CatBoost enforces on `depth`. Refused here rather than
# discovered as an allocation failure part way down a tree.
comptime OBLIVIOUS_MAX_DEPTH = 16


def parse_grow_policy(name: String) raises -> Int:
    """The policy code for a name. XGBoost's spellings: `depthwise`, and
    `lossguide` for leaf-wise; `leafwise` and `leaf_wise`/`depth_wise` are
    accepted so the LightGBM-flavored spelling works too. CatBoost's
    `SymmetricTree` is spelled `oblivious`, with `symmetric` accepted as the
    alias CatBoost users will reach for."""
    if name == "leafwise" or name == "leaf_wise" or name == "lossguide":
        return GROW_LEAFWISE
    if name == "depthwise" or name == "depth_wise":
        return GROW_DEPTHWISE
    if (
        name == "oblivious"
        or name == "symmetric"
        or name == "symmetrictree"
        or name == "symmetric_tree"
    ):
        return GROW_OBLIVIOUS
    raise Error(
        "grow_policy must be 'leafwise' (alias 'lossguide'), 'depthwise', or"
        " 'oblivious' (alias 'symmetric'), got '",
        name,
        "'",
    )


def grow_policy_name(policy: Int) -> String:
    if policy == GROW_LEAFWISE:
        return String("leafwise")
    if policy == GROW_DEPTHWISE:
        return String("depthwise")
    if policy == GROW_OBLIVIOUS:
        return String("oblivious")
    return String("unknown")


def check_grow_policy(policy: Int) raises:
    if (
        policy != GROW_LEAFWISE
        and policy != GROW_DEPTHWISE
        and policy != GROW_OBLIVIOUS
    ):
        raise Error(
            "grow_policy must be GROW_LEAFWISE (0), GROW_DEPTHWISE (1), or"
            " GROW_OBLIVIOUS (2), got ",
            policy,
        )


# Why oblivious growth is a code here and not a `next_leaf` rule.
#
# Leaf-wise and depth-wise both answer the same question -- which of the
# frontier's already-searched leaves is split next -- so they share
# `GrowthSchedule`. Oblivious growth does not ask that question. It searches a
# whole level at once (`split.find_best_split_shared`) and applies the single
# winner to every leaf of that level, so there is no per-leaf best split to
# rank and no admission decision to make: the level splits entirely or not at
# all. `GrowthSchedule.__init__` therefore REFUSES the code rather than
# quietly treating it as depth-wise, which is what its `policy != LEAFWISE`
# branch would otherwise do. Every frontier grower in this package builds a
# `GrowthSchedule` from `params.grow_policy` (`tree_sparse.grow_tree_sparse`,
# the three loops in `train_gpu.mojo`, `train_gpu_sparse.mojo`), so the
# refusal reaches all of them from this one place and a grower that has not
# implemented the mode reports that instead of growing the wrong tree.
comptime OBLIVIOUS_IS_NOT_A_FRONTIER_ORDER = True


@fieldwise_init
struct SharedSplitAudit(Copyable, Movable, Writable):
    """Per-leaf legality accounting for one level's shared split.

    Oblivious growth applies one split to every leaf of a level, so a leaf
    that cannot satisfy `min_data_in_leaf` or `min_sum_hessian_in_leaf` at the
    chosen candidate does not veto it: it contributes zero to the candidate's
    summed gain and is split anyway, possibly into an empty child. That is a
    fact about the tree a reader has to be able to see -- a split that was
    legal for one leaf out of sixteen is a different object from one that was
    legal for all sixteen -- so `split.find_best_split_shared` records it here
    instead of discarding it.

    `n_leaves` is the level's width, `n_illegal` how many of those leaves
    contributed zero at the *chosen* candidate, and `n_scored` the rest. All
    three are 0 when no split was found.
    """

    var n_leaves: Int
    var n_illegal: Int
    var n_scored: Int

    @staticmethod
    def none() -> SharedSplitAudit:
        return SharedSplitAudit(0, 0, 0)

    def all_illegal(self) -> Bool:
        """Whether every leaf of the level contributed zero. Impossible for a
        chosen split: a candidate no leaf could score sums to exactly 0.0 and
        0.0 never beats the running best, which starts there and is compared
        strictly."""
        return self.n_leaves > 0 and self.n_scored == 0

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "SharedSplitAudit(leaves=",
            self.n_leaves,
            ", scored=",
            self.n_scored,
            ", illegal=",
            self.n_illegal,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)




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


@fieldwise_init
struct ObliviousTrace(Copyable, Movable):
    """`MOJOTREES_OBLIVIOUS_TRACE=1`: one line per level of an oblivious tree.

    The per-leaf legality accounting (`SharedSplitAudit`) is a fact about the
    tree that no field of `tree.Tree` has room for -- the representation is
    deliberately an ordinary binary tree and gains no columns for this mode --
    and it is exactly the fact a reader needs, because a split that was legal
    for one leaf of sixteen and one that was legal for all sixteen are
    different objects wearing the same feature and threshold. So it is
    reported rather than dropped: a test reads it from
    `split.find_best_split_shared`'s `audit` argument directly, and a person
    running a fit reads it from here.

    Resolved once per tree, so a traced fit pays one `getenv` per tree and an
    untraced one pays one `getenv` and then a Bool test per level. Off, this
    prints nothing and allocates nothing.
    """

    var on: Bool

    @staticmethod
    def off() -> ObliviousTrace:
        return ObliviousTrace(False)

    @staticmethod
    def resolve() -> ObliviousTrace:
        var s = getenv("MOJOTREES_OBLIVIOUS_TRACE")
        return ObliviousTrace(s == "1" or s == "true" or s == "TRUE")

    def level(
        self,
        tree_index: Int,
        depth: Int,
        feature: Int,
        bin: Int,
        gain: Float64,
        audit: SharedSplitAudit,
    ):
        if not self.on:
            return
        print(
            "oblivious tree=",
            tree_index,
            " level=",
            depth,
            " leaves=",
            audit.n_leaves,
            " scored=",
            audit.n_scored,
            " illegal=",
            audit.n_illegal,
            " feature=",
            feature,
            " bin=",
            bin,
            " gain=",
            gain,
            sep="",
        )

    def stop(self, tree_index: Int, depth: Int, reason: String):
        if not self.on:
            return
        print(
            "oblivious tree=",
            tree_index,
            " level=",
            depth,
            " stop=",
            reason,
            sep="",
        )


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
    ) raises:
        if policy == GROW_OBLIVIOUS:
            # See OBLIVIOUS_IS_NOT_A_FRONTIER_ORDER. Refusing here is what
            # keeps every grower that has not implemented the mode --
            # `tree_sparse`, the three loops in `train_gpu`, and
            # `train_gpu_sparse`, all of which construct a schedule from
            # `params.grow_policy` -- from silently taking the depth-wise
            # branch below and growing a tree that is not symmetric.
            raise Error(
                "grow_policy=oblivious is not a frontier order and has no"
                " GrowthSchedule: a level is searched once and the winner is"
                " applied to every leaf of it, so there is no per-leaf best"
                " split to rank. Only tree.grow_tree implements it; this"
                " grower does not"
            )
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
