"""Feature interaction constraints.

A constraint set is a list of feature groups. Two features may appear
together on the same root-to-leaf path only if some single group contains
both (and, transitively, every other feature already on that path). This is
how a model is restricted to additive or low-order-interaction structure,
which is often what a domain expert or a regulator actually wants from a
GBDT.

The rule, per node, follows LightGBM's `ColSampler::GetByNode`. Let `B` be
the set of features already split on between the root and this node (its
"branch features"). The features this node may split on are

    allowed(B) = B  union  ( union of every group G with B subset-of G )

Read literally:

- At the root `B` is empty, every group vacuously contains it, and the
  allowed set is the union of all groups.
- Splitting on a feature narrows the allowed set to the groups that still
  contain everything used so far, so a path can never mix features from two
  groups that do not both contain the whole path.
- A feature already used on the branch stays available forever. Re-splitting
  on it adds no new interaction, so no group needs to permit it again.

The invariant this produces, and the one the structural tests check, is:
**every root-to-leaf path's feature set is a subset of at least one
configured group.**

Overlapping groups
------------------
Groups may overlap freely; a feature may appear in any number of them. The
union in the rule above is what makes overlap work: while several groups
still contain the whole branch, every one of them contributes candidates.
With groups `[[0, 1], [1, 2]]`, feature 1 may pair with 0 or with 2, but a
path can never carry 0, 1, and 2 together, because no single group holds
all three.

Unconstrained features
----------------------
There is no such thing here. The allowed set at the root is the union of
the groups, so **a feature listed in no group is never split on at all**
and drops out of the model. This matches LightGBM, and it is the behavior
to be careful with: constraining two of fifty features silently discards
the other forty-eight.

To leave a feature free to interact with everything, add it to every group.
Putting it in a group of its own does something different and usually
unwanted: it may be split on, but once it is, only itself remains allowed
below that split.

An empty constraint set (no groups) means no constraints: every feature is
allowed everywhere. That is the default and it costs nothing at run time.

Scope
-----
Constraints are a training-time restriction on split search, not part of a
fitted model: they change which trees get grown, never how a grown tree is
evaluated. Serialized models therefore carry no constraint record, the same
way they carry no `num_leaves` or `min_data_in_leaf`.
"""


struct InteractionConstraints(Copyable, Movable):
    """Feature groups stored flat: group `g` is
    `group_features[group_offsets[g] : group_offsets[g + 1]]`. A set with no
    groups means unconstrained."""

    var group_features: List[Int]
    var group_offsets: List[Int]
    var n_features: Int

    def __init__(out self):
        """No constraints: every feature may interact with every other."""
        self.group_features = List[Int]()
        self.group_offsets = [0]
        self.n_features = 0

    @staticmethod
    def from_groups(
        groups: List[List[Int]], n_features: Int
    ) raises -> InteractionConstraints:
        """Build from one list of feature indices per group. An empty
        `groups` means unconstrained."""
        var flat = List[Int]()
        var offsets = List[Int](capacity=len(groups) + 1)
        offsets.append(0)
        for g in range(len(groups)):
            for i in range(len(groups[g])):
                flat.append(groups[g][i])
            offsets.append(len(flat))
        return InteractionConstraints.from_flat(flat, offsets, n_features)

    @staticmethod
    def from_flat(
        group_features: List[Int], group_offsets: List[Int], n_features: Int
    ) raises -> InteractionConstraints:
        """Build from the flat representation, validating it. `group_offsets`
        has one more entry than there are groups; it starts at 0, increases
        strictly (no empty group), and ends at `len(group_features)`."""
        if len(group_offsets) < 1:
            raise Error("interaction constraint offsets must not be empty")
        if group_offsets[0] != 0:
            raise Error("interaction constraint offsets must start at 0")
        if group_offsets[len(group_offsets) - 1] != len(group_features):
            raise Error(
                "interaction constraint offsets must end at the flattened"
                " group length"
            )
        var n_groups = len(group_offsets) - 1
        if n_groups > 0 and n_features < 1:
            raise Error(
                "interaction constraints need a positive feature count"
            )
        for g in range(n_groups):
            var start = group_offsets[g]
            var end = group_offsets[g + 1]
            if end <= start:
                raise Error("interaction constraint groups must not be empty")
            for i in range(start, end):
                var f = group_features[i]
                if f < 0 or f >= n_features:
                    raise Error(
                        "interaction constraint feature ",
                        f,
                        " is out of range for ",
                        n_features,
                        " features",
                    )
                for j in range(start, i):
                    if group_features[j] == f:
                        raise Error(
                            "interaction constraint feature ",
                            f,
                            " is repeated within a group",
                        )
        var out = InteractionConstraints()
        out.group_features = group_features.copy()
        out.group_offsets = group_offsets.copy()
        out.n_features = n_features
        return out^

    def n_groups(self) -> Int:
        return len(self.group_offsets) - 1

    def is_empty(self) -> Bool:
        """True when no groups are configured, i.e. no constraints."""
        return self.n_groups() == 0

    def check_features(self, n_features: Int) raises:
        """Raise unless these constraints were built for a dataset with
        `n_features` columns."""
        if self.is_empty():
            return
        if self.n_features != n_features:
            raise Error(
                "interaction constraints were built for ",
                self.n_features,
                " features but the data has ",
                n_features,
            )

    def contains_all(self, group: Int, branch: List[Int]) -> Bool:
        """Whether group `group` contains every feature in `branch`. An
        empty branch is contained in every group."""
        var start = self.group_offsets[group]
        var end = self.group_offsets[group + 1]
        for i in range(len(branch)):
            var found = False
            for j in range(start, end):
                if self.group_features[j] == branch[i]:
                    found = True
                    break
            if not found:
                return False
        return True

    def allowed_features(self, branch: List[Int]) -> List[Bool]:
        """The per-feature allow mask for a node whose branch features are
        `branch` (order and duplicates do not matter; it is read as a set).

        Returns an **empty list** when there are no constraints, which every
        consumer reads as "every feature allowed"; that keeps the
        unconstrained path free of per-node allocation."""
        if self.is_empty():
            return List[Bool]()
        var allowed = List[Bool](capacity=self.n_features)
        allowed.resize(self.n_features, False)
        # Features already on this branch stay available: re-splitting on one
        # introduces no new interaction.
        for i in range(len(branch)):
            allowed[branch[i]] = True
        for g in range(self.n_groups()):
            if not self.contains_all(g, branch):
                continue
            for j in range(self.group_offsets[g], self.group_offsets[g + 1]):
                allowed[self.group_features[j]] = True
        return allowed^


def extend_branch(branch: List[Int], feature: Int) -> List[Int]:
    """The branch feature set of both children of a node split on `feature`.
    Kept duplicate-free so the containment scans stay short; duplicates would
    be harmless but not free.

    This allocates unconditionally -- it has no view of the constraint set --
    so a grower must not call it when `InteractionConstraints.is_empty()`.
    An unconstrained `allowed_features` returns the empty mask for every
    branch, so the set it would build is never read, and building it costs a
    list copy per split for nothing. `tree.grow_tree` gates on that; see the
    `constrained` flag there."""
    var out = branch.copy()
    for i in range(len(out)):
        if out[i] == feature:
            return out^
    out.append(feature)
    return out^
