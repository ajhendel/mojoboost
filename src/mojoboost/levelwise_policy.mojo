"""Growth policy for depth-batched (level-wise) trees.

This module holds the decisions a level-wise grower makes and nothing else.
It touches no histogram, no device, and no dataset, so the rules below can be
read, reasoned about, and reproduced on the host without a GPU present.

What level-wise growth is here
------------------------------
`tree.grow_tree` and `train_gpu.grow_tree_gpu` grow leaf-wise: one global
priority queue over every live leaf, one split committed per iteration, and
one host decision per split. Level-wise growth replaces that queue with a
*frontier* of all nodes at one depth. Every node at the depth is searched,
the whole depth's splits are committed together, and only then does growth
move down. The tree that comes out is a different tree, not a faster route to
the same one. Nothing in this module tries to reproduce leaf-wise decisions,
and no claim of LightGBM equivalence is made or implied anywhere in this
lane; LightGBM grows leaf-wise and mojoboost's shipped growers grow leaf-wise
with it.

Why the mode exists at all is a machine argument, not a quality argument. A
leaf-wise tree costs `num_leaves` host decisions and `num_leaves` device
launch groups, each sized by one node's rows, which for the small nodes near
the end of a tree is far less work than a GPU wants per launch. A level-wise
tree costs one host decision and one launch group *per depth*, each sized by
every row still live in the tree. The arithmetic per level is the same;
only the launch count, the host synchronization count, and the work per
launch change. See `docs/design/GPU_LEVELWISE.md`.

The three rules that make a level-wise tree well defined
--------------------------------------------------------
1. **Leaf budget.** Committing a split turns one leaf into two, so it adds
   exactly one leaf. A level offering `E` eligible splits would add `E`
   leaves at once, which can overshoot `num_leaves` in a way leaf-wise
   growth never can. `admit_level` resolves that: `num_leaves` stays a hard
   bound and the last level is admitted as a gain-ranked prefix (or refused
   whole, under `BUDGET_WHOLE_LEVEL`). Growth therefore never produces more
   leaves than `num_leaves`, which is the property model size, serialization,
   and every existing parity statement rest on.

2. **Deterministic order.** Level-wise admission ranks siblings against each
   other, so it needs a total order on candidates. The order is gain
   descending, then node id ascending. Node ids are assigned breadth first,
   left child before right, in ascending parent id order, so the order is a
   function of the tree alone and does not depend on container mechanics,
   launch scheduling, or thread count.

   This is deliberately *not* how leaf-wise growth breaks ties. Both shipped
   growers scan their frontier list with a strict `>` and keep the earliest
   entry, which is frontier slot order, an artifact of how the list is
   maintained (the left child overwrites the parent's slot and the right
   child is appended). That rule is correct and stable for a queue popped one
   at a time; it is not a rule that survives ranking a whole level, so this
   module states its own and the design doc records the difference.

3. **Shape rules first.** A node's depth limit and its minimum row count are
   properties of the tree, not of a histogram, so they can be evaluated
   before any histogram is built. `depth_permits_split` and
   `rows_permit_split` mirror the two guards at the top of `tree._search`
   exactly, and they exist here so a level's launch can be narrowed to the
   nodes that could possibly split. A node these reject is terminal: it
   becomes a leaf at its own depth and is never revisited, which is also what
   leaf-wise growth does with it.

What this module does not decide
--------------------------------
Split search itself. Gain, the min-child-hessian test, the per-child
`min_data_in_leaf` test, monotone candidate rejection, interaction allow
masks, categorical partitioning, and missing-value direction all stay in
`split.mojo` and `categorical.mojo`, reached through `tree._search`, so a
level-wise grower enforces every one of them identically to the leaf-wise
growers. Level-wise growth changes *which nodes are searched when* and
*which of the results are taken*, never how a candidate is scored.

Nothing here is registered as a user-facing parameter. `LevelwiseParams` is
held apart from `tree.TreeParams` the way `tree_parameters_extra` holds its
own bundle, so this lane changes no shared struct and adds nothing to
`params.mojo`, the CLI, the C API, the Python layer, or any serialized model.
Level-wise growth is a training-time algorithm choice; a tree it produces is
an ordinary `tree.Tree` and carries no record of the mode that grew it.
"""


# How a level whose eligible splits would overrun the leaf budget is handled.
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

# `max_level_nodes` of 0 means the level is launched whole, however wide it
# is. A positive value caps how many nodes one launch group covers, which is
# the memory control described in the design doc: a level of `n` nodes needs
# `n` histograms resident at once, and a deep level is wide.
comptime UNLIMITED_LEVEL_NODES = 0


struct LevelwiseParams(Copyable, Movable):
    """Configuration of the level-wise mode, held apart from
    `tree.TreeParams` so this lane changes no shared struct. The defaults
    describe the mode as it would be entered with nothing tuned; `is_active`
    is False for a bundle that has not been enabled, so a grower can test it
    once and take its ordinary leaf-wise path.

    `budget_mode` is one of the BUDGET_* constants above.
    `max_level_nodes` is the launch-group cap (UNLIMITED_LEVEL_NODES for
    none); it bounds histogram residency, not the tree, so a level wider than
    the cap is grown in several launch groups and the resulting tree is
    unchanged.
    """

    var enabled: Bool
    var budget_mode: Int
    var max_level_nodes: Int

    def __init__(
        out self,
        enabled: Bool = False,
        budget_mode: Int = BUDGET_RANK,
        max_level_nodes: Int = UNLIMITED_LEVEL_NODES,
    ):
        self.enabled = enabled
        self.budget_mode = budget_mode
        self.max_level_nodes = max_level_nodes

    @staticmethod
    def off() -> LevelwiseParams:
        """The default: leaf-wise growth, exactly as today."""
        return LevelwiseParams()

    @staticmethod
    def on(
        budget_mode: Int = BUDGET_RANK,
        max_level_nodes: Int = UNLIMITED_LEVEL_NODES,
    ) -> LevelwiseParams:
        return LevelwiseParams(True, budget_mode, max_level_nodes)

    def is_active(self) -> Bool:
        return self.enabled

    def check(self) raises:
        if self.budget_mode != BUDGET_RANK and (
            self.budget_mode != BUDGET_WHOLE_LEVEL
        ):
            raise Error(
                "levelwise budget_mode must be BUDGET_RANK or"
                " BUDGET_WHOLE_LEVEL"
            )
        if self.max_level_nodes < 0:
            raise Error("levelwise max_level_nodes must be nonnegative")


@fieldwise_init
struct LevelCandidate(Copyable, Movable, Writable):
    """One frontier node's offer at the level being decided.

    `node` is its tree node id, `gain` the gain of the best split found for
    it, and `eligible` whether that split may be taken at all. A candidate is
    eligible only when a split was found, its gain is strictly positive (the
    same bar both leaf-wise growers set by starting their best gain at 0.0),
    and every shape rule the grower checked passed. An ineligible candidate
    is terminal: it becomes a leaf at its own depth.
    """

    var node: Int
    var gain: Float64
    var eligible: Bool

    @staticmethod
    def terminal(node: Int) -> LevelCandidate:
        """A node offering nothing, which is what a node past the depth
        limit, under the row minimum, or with no positive-gain split is."""
        return LevelCandidate(node, 0.0, False)

    def write_to(self, mut writer: Some[Writer]):
        if not self.eligible:
            writer.write("LevelCandidate(node=", self.node, ", terminal)")
        else:
            writer.write(
                "LevelCandidate(node=",
                self.node,
                ", gain=",
                self.gain,
                ")",
            )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def count_eligible(candidates: List[LevelCandidate]) -> Int:
    var n = 0
    for i in range(len(candidates)):
        if candidates[i].eligible:
            n += 1
    return n


def rank_level(candidates: List[LevelCandidate]) -> List[Int]:
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
    candidates: List[LevelCandidate],
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


@always_inline
def depth_permits_split(depth: Int, max_depth: Int) -> Bool:
    """Mirror of the depth guard at the top of `tree._search`: a node at or
    past `max_depth` offers no split, and `max_depth <= 0` means unlimited,
    as in LightGBM.

    Duplicated here on purpose. `_search` can only answer this after a
    histogram exists, and the whole point of a batched level is to decide
    which nodes belong in the launch *before* building anything. The two must
    agree, and they agree by being the same three-term expression; the
    handoff records this as the one rule this lane copies rather than calls.
    """
    return max_depth <= 0 or depth < max_depth


@always_inline
def rows_permit_split(n_rows: Int, min_data_in_leaf: Int) -> Bool:
    """Mirror of the row guard at the top of `tree._search`: a node needs at
    least two rows, and at least twice `min_data_in_leaf`, before any
    candidate could give both children a legal share. The per-child test is
    stricter and stays in `split.find_best_split`; this is only the cheap
    parent-side rejection that lets an ineligible node be dropped from a
    level's launch."""
    return n_rows >= 2 and n_rows >= 2 * min_data_in_leaf


def prefilter_level(
    nodes: List[Int],
    row_counts: List[Int],
    depth: Int,
    max_depth: Int,
    min_data_in_leaf: Int,
) raises -> List[Int]:
    """The subset of a level's nodes worth searching, in the order given.

    Applying the two shape rules before the level's launch is what keeps a
    batched histogram build and a batched search from spending slots on nodes
    that cannot split. A node dropped here is terminal and its value is
    already known from its own histogram, so nothing about it is left
    undecided.
    """
    if len(nodes) != len(row_counts):
        raise Error("level node and row-count lists must be the same length")
    var out = List[Int]()
    if not depth_permits_split(depth, max_depth):
        return out^
    for i in range(len(nodes)):
        if rows_permit_split(row_counts[i], min_data_in_leaf):
            out.append(nodes[i])
    return out^


def level_capacity(depth: Int, num_leaves: Int) -> Int:
    """How many nodes a level-wise tree can hold at `depth`, given the leaf
    budget. A complete level holds `2**depth` nodes, and the budget caps that
    at `num_leaves`, since every node at the level is a live leaf.

    Shifting past the width of an `Int` is the only trap: a depth of 62 or
    more already exceeds any budget worth naming, so it saturates.
    """
    if depth < 0:
        return 0
    if depth >= 62:
        return num_leaves if num_leaves > 0 else 0
    var width = 1 << depth
    if num_leaves > 0 and width > num_leaves:
        return num_leaves
    return width


def full_level_depth(num_leaves: Int) -> Int:
    """The deepest level a level-wise tree can fill completely under a leaf
    budget of `num_leaves`: the largest `d` with `2**d <= num_leaves`.

    This is the number that makes the mode's headline behavior concrete. At
    the default `num_leaves = 31` a level-wise tree fills depths 0 through 4
    (16 leaves) and then spends its remaining 15 units of budget on a partial
    depth-5 level, so the tree is depth 5 and holds 31 leaves. A leaf-wise
    tree with the same budget is depth 5 only by accident and is usually much
    deeper on one branch and much shallower on another. The two are different
    model classes at identical parameters, which is why matched-parameter
    timing comparisons say nothing (see the design doc).
    """
    if num_leaves < 1:
        return 0
    var d = 0
    var width = 1
    while width * 2 <= num_leaves and d < 62:
        width *= 2
        d += 1
    return d


def effective_max_depth(num_leaves: Int, max_depth: Int) -> Int:
    """The deepest level growth can reach, whichever of the two limits binds.

    Under level-wise growth `num_leaves` is a depth limit in disguise: a
    complete level at depth `d` already holds `2**d` leaves, so the budget
    caps depth at `full_level_depth(num_leaves) + 1`, the extra level being
    the partial one BUDGET_RANK admits. `max_depth`, when positive, caps it
    directly. Leaf-wise growth has no such coupling, which is the practical
    reason a level-wise run wants `max_depth` set explicitly rather than
    inherited from a leaf-wise configuration.
    """
    var by_budget = full_level_depth(num_leaves) + 1
    if max_depth > 0 and max_depth < by_budget:
        return max_depth
    return by_budget


@fieldwise_init
struct LaunchProfile(Copyable, Movable, Writable):
    """How many launch groups and host synchronizations a tree costs.

    This is the quantity the mode changes directly, and unlike a quality
    claim it can be counted rather than benchmarked. A launch group is one
    round of histogram, search, and partition work; a host synchronization is
    one point where the device must finish before the host can decide
    anything.
    """

    var launch_groups: Int
    var host_syncs: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "LaunchProfile(launch_groups=",
            self.launch_groups,
            ", host_syncs=",
            self.host_syncs,
            ")",
        )

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


def leafwise_profile(num_leaves: Int) -> LaunchProfile:
    """A leaf-wise tree's cost. One group per split plus the root, and one
    host synchronization each, because the next split cannot be chosen until
    the last one's histograms are on the host. A 31-leaf tree costs 31."""
    var n = num_leaves if num_leaves > 0 else 1
    return LaunchProfile(n, n)


def levelwise_profile(
    num_leaves: Int, max_depth: Int, max_level_nodes: Int = 0
) -> LaunchProfile:
    """A level-wise tree's cost at the same budget.

    One group per level, plus the root's, and one host synchronization per
    level. With `max_level_nodes` set, a level wider than the cap is split
    into that many groups, so the cap trades launches back for histogram
    residency. At the default 31-leaf budget this is 6 groups against
    leaf-wise's 31, and the gap widens with the budget: 256 leaves costs 9
    against 256.

    These are counts of decision points, not of kernels, and they carry no
    claim about wall clock. The work inside a level-wise group is larger by
    exactly the factor its group count is smaller, so a run that is not
    launch bound will not move.
    """
    var depth = effective_max_depth(num_leaves, max_depth)
    var groups = 1
    for d in range(depth):
        var width = level_capacity(d, num_leaves)
        if max_level_nodes > 0 and width > max_level_nodes:
            groups += (width + max_level_nodes - 1) // max_level_nodes
        else:
            groups += 1
    return LaunchProfile(groups, groups)
