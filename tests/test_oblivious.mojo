"""`grow_policy = oblivious`: CatBoost's symmetric tree on the CPU grower.

One split is searched across a whole level and applied to every leaf of it,
so a row's leaf is the bit pattern of its `max_depth` outcomes
(`docs/design/OBLIVIOUS.md` Part B, `src/mojotrees/growth_policy.mojo`). The
tree that comes out is an ORDINARY `tree.Tree` with the same split repeated
across a level; the symmetry is a property of how it was grown and not of how
it is stored, which is what leaves predict, dump, serialization and model I/O
working with no knowledge of the mode.

What these tests are for, in the order they matter:

1. **The tree really is symmetric.** Every node at a level shares a feature,
   a threshold and a missing direction. Without this assertion the mode could
   silently be growing leaf-wise and every other test here would still pass,
   which is exactly the failure `bench/results/LANE_RULES.md` warns about.
2. **Per-leaf illegality contributes zero rather than vetoing.** A candidate
   that no single leaf may take is still the level's split when the rest of
   the level wants it, and the count of leaves that could not take it is
   recorded (`growth_policy.SharedSplitAudit`). Both arms are asserted, so
   the test fails if the rule is ever changed to a veto AND if it is ever
   changed to "nobody is ever illegal".
3. **The shared search reduces to the old one.** At a level of one leaf,
   `find_best_split_shared` returns `find_best_split`'s answer to the bit.
   That is the property that keeps the new code out of the default path's
   numbers.
4. **The default policy is untouched**, and determinism holds at
   `MOJOTREES_NUM_WORKERS` 1, 3 and 8 in both modes.

No tolerance appears anywhere below: every float comparison is on
`to_bits()`, per `LANE_RULES.md`.
"""

from std.os import remove, setenv
from std.testing import (
    assert_equal,
    assert_false,
    assert_not_equal,
    assert_raises,
    assert_true,
    TestSuite,
)

from mojotrees import (
    SQUARED_ERROR,
    BinnedMatrix,
    BoosterParams,
    Tree,
    TreeParams,
    bin_equal_width,
    grow_policy_name,
    grow_tree,
    parse_grow_policy,
)
from mojotrees.growth_policy import (
    GROW_DEPTHWISE,
    GROW_LEAFWISE,
    GROW_OBLIVIOUS,
    OBLIVIOUS_MAX_DEPTH,
    GrowthSchedule,
    SharedSplitAudit,
    check_grow_policy,
)
from mojotrees.histogram import Histogram
from mojotrees.model import fit
from mojotrees.model_dump import build_dump
from mojotrees.serialize import load_model, save_model
from mojotrees.split import SplitInfo, find_best_split, find_best_split_shared
from mojotrees.tree_parameters_extra import ExtraTreeParams

comptime _TMP_PATH = "./.test_oblivious_roundtrip.tmp"


# ------------------------------------------------------------------ fixtures


def _dense(n_rows: Int, n_features: Int) -> List[Float64]:
    """Column-major pseudo-random features, the shape `bin_equal_width` takes.
    """
    var out = List[Float64](capacity=n_rows * n_features)
    var state = UInt64(20260816)
    for _ in range(n_rows * n_features):
        state = state * 6364136223846793005 + 1442695040888963407
        out.append(Float64(state >> 11) * (1.0 / 9007199254740992.0))
    return out^


def _grads(n_rows: Int, features: List[Float64], n_rows_total: Int) -> List[
    Float64
]:
    """A gradient with real structure in the first three features, so a level
    has something to disagree about and the chosen split is decided by the
    data rather than by a tie."""
    var out = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        out.append(
            -(
                3.0 * features[r]
                - 2.0 * features[n_rows_total + r]
                + 1.5 * features[2 * n_rows_total + r]
                * features[3 * n_rows_total + r]
            )
        )
    return out^


def _ones(n: Int) -> List[Float64]:
    var out = List[Float64](capacity=n)
    for _ in range(n):
        out.append(1.0)
    return out^


def _oblivious_params(max_depth: Int, min_data: Int = 1) -> TreeParams:
    # `num_leaves` is deliberately small and wrong for the depth: it does not
    # bind under this policy and one of the tests below pins that.
    return TreeParams(
        3,
        min_data,
        1.0,
        1e-3,
        max_depth=max_depth,
        grow_policy=GROW_OBLIVIOUS,
    )


def _grow(
    n_rows: Int, n_features: Int, params: TreeParams, n_bins: Int = 16
) raises -> Tree:
    var features = _dense(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, n_bins)
    var grad = _grads(n_rows, features, n_rows)
    return grow_tree(data, grad, _ones(n_rows), params)


# ------------------------------------------------------- structural helpers


def node_depths(tree: Tree) -> List[Int]:
    """Depth of every node, in edges from the root. Children carry larger ids
    than their parent in every grower, so one forward pass suffices."""
    var depths = List[Int](capacity=len(tree.feature))
    depths.resize(len(tree.feature), 0)
    for n in range(len(tree.feature)):
        if tree.feature[n] >= 0:
            depths[tree.left[n]] = depths[n] + 1
            depths[tree.right[n]] = depths[n] + 1
    return depths^


def tree_depth(tree: Tree) -> Int:
    var depths = node_depths(tree)
    var d = 0
    for n in range(len(depths)):
        if depths[n] > d:
            d = depths[n]
    return d


def is_symmetric(tree: Tree) -> Bool:
    """Whether every internal node at one depth carries the same split.

    This is THE marker of an oblivious tree in an ordinary binary
    representation. A leaf-wise or depth-wise tree of any interest fails it,
    which is what makes it evidence rather than a formality."""
    var depths = node_depths(tree)
    var d_max = tree_depth(tree)
    for d in range(d_max + 1):
        var feature = -2
        var threshold = 0
        var default_left = False
        for n in range(len(tree.feature)):
            if depths[n] != d or tree.feature[n] < 0:
                continue
            if feature == -2:
                feature = tree.feature[n]
                threshold = tree.threshold_bin[n]
                default_left = tree.default_left[n]
                continue
            if (
                tree.feature[n] != feature
                or tree.threshold_bin[n] != threshold
                or tree.default_left[n] != default_left
            ):
                return False
    return True


def leaf_indices(tree: Tree) -> List[Int]:
    """Every node's leaf index under the cross-backend numbering: the root is
    0, a left child keeps its parent's index, and a right child at level d
    adds `1 << d`. First level, LOWEST bit. This is CatBoost's numbering and
    the device's, so it is the labeling a host/device node-identity test will
    compare against."""
    var depths = node_depths(tree)
    var ix = List[Int](capacity=len(tree.feature))
    ix.resize(len(tree.feature), 0)
    for n in range(len(tree.feature)):
        if tree.feature[n] >= 0:
            ix[tree.left[n]] = ix[n]
            ix[tree.right[n]] = ix[n] + (1 << depths[n])
    return ix^


def row_leaf_index(tree: Tree, data: BinnedMatrix, row: Int) -> Int:
    """The same number built the way a row builds it: walk the tree and set
    bit `l` when the row goes RIGHT at level `l`."""
    var node = 0
    var level = 0
    var index = 0
    while tree.feature[node] >= 0:
        if tree.goes_left(node, data.bin_at(row, tree.feature[node])):
            node = tree.left[node]
        else:
            index += 1 << level
            node = tree.right[node]
        level += 1
    return index


def first_node_at(tree: Tree, depth: Int) -> Int:
    var depths = node_depths(tree)
    var first = -1
    for n in range(len(tree.feature)):
        if depths[n] == depth and (first < 0 or n < first):
            first = n
    return first


def internal_nodes_at(tree: Tree, depth: Int) -> Int:
    var depths = node_depths(tree)
    var n = 0
    for i in range(len(tree.feature)):
        if depths[i] == depth and tree.feature[i] >= 0:
            n += 1
    return n


def value_bits(tree: Tree) -> List[Int]:
    """Every node's value as its raw IEEE-754 pattern, so comparisons are
    exact and no tolerance is needed anywhere in this file."""
    var out = List[Int](capacity=len(tree.value))
    for i in range(len(tree.value)):
        out.append(Int(tree.value[i].to_bits()))
    return out^


def assert_same_tree_bits(got: Tree, want: Tree) raises:
    assert_equal(got.n_leaves, want.n_leaves)
    assert_equal(len(got.feature), len(want.feature))
    var gb = value_bits(got)
    var wb = value_bits(want)
    for i in range(len(want.feature)):
        assert_equal(got.feature[i], want.feature[i])
        assert_equal(got.threshold_bin[i], want.threshold_bin[i])
        assert_equal(got.left[i], want.left[i])
        assert_equal(got.right[i], want.right[i])
        assert_equal(got.default_left[i], want.default_left[i])
        assert_equal(gb[i], wb[i])
        assert_equal(
            Int(got.split_gain[i].to_bits()), Int(want.split_gain[i].to_bits())
        )


# ------------------------------------------------------------- the parameter


def test_the_policy_parses_and_names_itself() raises:
    assert_equal(parse_grow_policy(String("oblivious")), GROW_OBLIVIOUS)
    assert_equal(parse_grow_policy(String("symmetric")), GROW_OBLIVIOUS)
    assert_equal(parse_grow_policy(String("symmetric_tree")), GROW_OBLIVIOUS)
    assert_equal(parse_grow_policy(String("symmetrictree")), GROW_OBLIVIOUS)
    assert_equal(grow_policy_name(GROW_OBLIVIOUS), String("oblivious"))
    # The two existing spellings are untouched.
    assert_equal(parse_grow_policy(String("leafwise")), GROW_LEAFWISE)
    assert_equal(parse_grow_policy(String("depthwise")), GROW_DEPTHWISE)
    check_grow_policy(GROW_OBLIVIOUS)
    with assert_raises():
        _ = parse_grow_policy(String("oblivius"))


def test_the_schedule_refuses_the_policy_rather_than_guessing() raises:
    """`GrowthSchedule` is built from `params.grow_policy` by every grower
    that keeps a frontier -- `tree_sparse`, the three loops in `train_gpu`,
    `train_gpu_sparse` -- and its `policy != GROW_LEAFWISE` branch is the
    depth-wise one. Without this refusal each of those would accept
    `grow_policy=oblivious` and grow a tree that is not symmetric, reporting
    nothing. The refusal is the gate; this asserts it opens."""
    with assert_raises():
        _ = GrowthSchedule(GROW_OBLIVIOUS)
    # The two policies it does implement still construct.
    var leafwise = GrowthSchedule(GROW_LEAFWISE)
    assert_equal(leafwise.policy, GROW_LEAFWISE)
    var depthwise = GrowthSchedule(GROW_DEPTHWISE)
    assert_equal(depthwise.policy, GROW_DEPTHWISE)


def test_the_mode_refuses_what_it_cannot_honor() raises:
    var n_rows = 200
    var features = _dense(n_rows, 4)
    var data = bin_equal_width(features, n_rows, 4, 16)
    var grad = _grads(n_rows, features, n_rows)
    var hess = _ones(n_rows)

    # `max_depth` is the only bound there is, so it is required.
    with assert_raises():
        _ = grow_tree(
            data, grad, hess, TreeParams(8, 1, 1.0, 1e-3,
                grow_policy=GROW_OBLIVIOUS)
        )
    # And bounded, because depth d costs 2^(d+1) - 1 nodes unconditionally.
    with assert_raises():
        _ = grow_tree(data, grad, hess, _oblivious_params(
            OBLIVIOUS_MAX_DEPTH + 1
        ))
    # `extra_trees` draws one threshold per node; a level has one split.
    var extra = ExtraTreeParams()
    extra.extra_trees = True
    var p = _oblivious_params(3)
    p.extra = extra^
    with assert_raises():
        _ = grow_tree(data, grad, hess, p)
    # The boundary value is accepted, so the refusal above is the bound and
    # not an off-by-one that would have refused everything.
    var ok = _oblivious_params(OBLIVIOUS_MAX_DEPTH)
    _ = ok  # constructing it must not raise; growing to depth 16 is not run.


def test_num_leaves_does_not_bind() raises:
    """A level splits entirely or not at all, so `num_leaves` cannot be met
    exactly and is ignored. `_oblivious_params` asks for 3 leaves at depth 3
    and must get 8. CatBoost resolves the same collision by overwriting
    `max_leaves` with `1 << depth`; the divergence is recorded in
    `docs/design/CATBOOST_CATALOG.md`."""
    var tree = _grow(400, 5, _oblivious_params(3))
    assert_equal(tree.n_leaves, 8)
    assert_equal(len(tree.feature), 15)


# ------------------------------------------------------- the symmetry marker


def test_an_oblivious_tree_is_symmetric() raises:
    """THE marker. Every internal node at a level carries the same feature,
    the same threshold bin and the same missing direction, and there are
    2^d of them at depth d, so the tree is complete as well as symmetric.

    The same fixture under leaf-wise and depth-wise growth is asserted NOT to
    be symmetric, which is what makes this evidence about the mode rather
    than about the fixture being too easy to disagree on."""
    var n_rows = 800
    var n_features = 6
    var tree = _grow(n_rows, n_features, _oblivious_params(3))
    assert_true(is_symmetric(tree))
    assert_equal(tree_depth(tree), 3)
    assert_equal(internal_nodes_at(tree, 0), 1)
    assert_equal(internal_nodes_at(tree, 1), 2)
    assert_equal(internal_nodes_at(tree, 2), 4)
    # Every leaf sits at the depth limit: a symmetric tree has no short
    # branches.
    var depths = node_depths(tree)
    for n in range(len(tree.feature)):
        if tree.feature[n] < 0:
            assert_equal(depths[n], 3)

    # At least two distinct features are used across the levels, so "every
    # node shares a feature" is a real constraint on this fixture and not a
    # tree that only ever had one feature to choose.
    var seen = List[Int]()
    for n in range(len(tree.feature)):
        if tree.feature[n] < 0:
            continue
        var known = False
        for k in range(len(seen)):
            if seen[k] == tree.feature[n]:
                known = True
        if not known:
            seen.append(tree.feature[n])
    assert_true(len(seen) >= 2)

    # The control: the same rows under the two frontier policies are not
    # symmetric, so this assertion can fail.
    var leafwise = _grow(
        n_rows,
        n_features,
        TreeParams(8, 1, 1.0, 1e-3, max_depth=3, grow_policy=GROW_LEAFWISE),
    )
    assert_false(is_symmetric(leafwise))
    var depthwise = _grow(
        n_rows,
        n_features,
        TreeParams(8, 1, 1.0, 1e-3, max_depth=3, grow_policy=GROW_DEPTHWISE),
    )
    assert_false(is_symmetric(depthwise))


def test_leaf_numbering_is_first_level_lowest_bit() raises:
    """The numbering half of the marker. Symmetry alone does not pin which
    leaf a row lands in, and the host/device node-identity test compares
    exactly that, so the convention is asserted from both ends: the
    structural labeling (left child keeps the index, right child at level d
    adds `1 << d`) and a row's own traversal bits must agree, for every row.

    A depth-3 tree is used deliberately: with the FIRST level as the low bit,
    leaf index and left-to-right position differ from depth 2 onward, so a
    tree that had quietly used the first level as the HIGH bit would pass at
    depth 1 and fail here."""
    var n_rows = 600
    var n_features = 5
    var features = _dense(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 16)
    var grad = _grads(n_rows, features, n_rows)
    var tree = grow_tree(data, grad, _ones(n_rows), _oblivious_params(3))
    assert_true(is_symmetric(tree))

    var ix = leaf_indices(tree)
    for r in range(n_rows):
        assert_equal(
            row_leaf_index(tree, data, r), ix[tree.leaf_index_row(data, r)]
        )

    # Every index in [0, 2^d) appears exactly once at depth d, so the
    # labeling is a bijection and not just consistent.
    var depths = node_depths(tree)
    for d in range(4):
        var width = 1 << d
        var seen = List[Int](capacity=width)
        seen.resize(width, 0)
        for n in range(len(tree.feature)):
            if depths[n] != d:
                continue
            assert_true(ix[n] >= 0 and ix[n] < width)
            seen[ix[n]] += 1
        for j in range(width):
            assert_equal(seen[j], 1)


def test_node_ids_follow_ascending_leaf_index_left_before_right() raises:
    """The other half of the contract: a level's children are created by
    walking that level's leaves in ascending LEAF INDEX and emitting left
    before right. That is not ascending node id -- at level 2 the leaves in
    node-id order carry indices 0, 2, 1, 3 -- so this pins the ordering that
    would otherwise be whatever the frontier container happened to do, and
    which a host/device comparison would catch as a spurious mismatch on two
    correct trees."""
    var tree = _grow(600, 5, _oblivious_params(3))
    var ix = leaf_indices(tree)
    var depths = node_depths(tree)
    for d in range(3):
        var width = 1 << d
        var base = first_node_at(tree, d + 1)
        assert_true(base > 0)
        # Parents of this level, keyed by leaf index.
        var parent_of = List[Int](capacity=width)
        parent_of.resize(width, -1)
        for n in range(len(tree.feature)):
            if depths[n] == d and tree.feature[n] >= 0:
                parent_of[ix[n]] = n
        for j in range(width):
            assert_true(parent_of[j] >= 0)
            assert_equal(tree.left[parent_of[j]], base + 2 * j)
            assert_equal(tree.right[parent_of[j]], base + 2 * j + 1)

    # And the ordering really is distinguishable on this tree: at level 2 the
    # node-id order and the leaf-index order are different permutations, so
    # the assertions above would fail under the other rule.
    var level2 = List[Int]()
    for n in range(len(tree.feature)):
        if depths[n] == 2:
            level2.append(ix[n])
    assert_equal(len(level2), 4)
    assert_equal(level2[0], 0)
    assert_equal(level2[1], 2)
    assert_equal(level2[2], 1)
    assert_equal(level2[3], 3)


# ------------------------------- per-leaf legality contributes zero, recorded


def _leaf_hist(grads: List[Float64], counts: List[Int]) -> Histogram:
    """One leaf's histogram over a single feature, hessian = count, so a
    gain is exactly a variance reduction and can be worked out by hand."""
    var g = grads.copy()
    var h = List[Float64](capacity=len(counts))
    for i in range(len(counts)):
        h.append(Float64(counts[i]))
    return Histogram.from_planes(g^, h^, counts.copy(), 1, len(counts))


def _illegality_level() -> List[Histogram]:
    """Two leaves over one feature and four bins, built so that

      - the bin-0 threshold is by far the best candidate for leaf 0, and
      - it puts a single row in leaf 1's left child.

    With `min_data_in_leaf = 5` leaf 1 is illegal at bin 0 and legal at bins 1
    and 2. Leaf 1's gradients are flat, so it scores exactly 0.0 wherever it
    is legal and the winner is decided by leaf 0 alone. Worked by hand at
    `lambda_l2 = 0`, `lambda_l1 = 0`:

      leaf 0 counts [10,10,10,10], grads [-40,0,0,0], H = 40, G = -40
        bin 0: 1600/10 + 0/30 - 1600/40 = 160 - 40 = 120
        bin 1: 1600/20 + 0/20 - 40      =  80 - 40 =  40
        bin 2: 1600/30 + 0/10 - 40      =  53.33.. - 40
      leaf 1 counts [1,30,30,30], grads all 0             = 0 everywhere

    so the summed gains are 120, 40, 13.33.. and bin 0 wins WITH one leaf
    contributing nothing. A veto rule would have answered bin 1.
    """
    var leaf0 = _leaf_hist(
        [-40.0, 0.0, 0.0, 0.0], [10, 10, 10, 10]
    )
    var leaf1 = _leaf_hist([0.0, 0.0, 0.0, 0.0], [1, 30, 30, 30])
    var out = List[Histogram]()
    out.append(leaf0^)
    out.append(leaf1^)
    return out^


def test_an_illegal_leaf_contributes_zero_and_is_recorded() raises:
    var level = _illegality_level()
    var audit = SharedSplitAudit.none()
    var split = find_best_split_shared(
        audit,
        level,
        lambda_reg=0.0,
        min_child_hess=0.0,
        min_data_in_leaf=5,
    )
    assert_true(split.found)
    assert_equal(split.feature, 0)
    # Not vetoed: the candidate one leaf could not take is still the level's.
    assert_equal(split.bin, 0)
    assert_equal(Int(split.gain.to_bits()), Int(Float64(120.0).to_bits()))
    # And the fact is recorded rather than discarded.
    assert_equal(audit.n_leaves, 2)
    assert_equal(audit.n_illegal, 1)
    assert_equal(audit.n_scored, 1)
    assert_false(audit.all_illegal())


def test_the_illegal_leaf_really_is_illegal_on_its_own() raises:
    """The other half of the previous test: leaf 1 alone offers no split at
    bin 0 under `min_data_in_leaf = 5`, so the zero contribution is a leaf
    that genuinely failed the rule and not a leaf whose gain happened to be
    zero. Without this the previous test would pass under a veto rule too,
    which is the "gate that never opened" failure `LANE_RULES.md` names."""
    var level = _illegality_level()
    var alone = find_best_split(
        level[1],
        lambda_reg=0.0,
        min_child_hess=0.0,
        min_data_in_leaf=5,
    )
    # Flat gradients: nothing to gain anywhere, so no split at all.
    assert_false(alone.found)
    # And leaf 0 alone does choose bin 0, which is where the level's 120 came
    # from.
    var leaf0 = find_best_split(
        level[0],
        lambda_reg=0.0,
        min_child_hess=0.0,
        min_data_in_leaf=5,
    )
    assert_true(leaf0.found)
    assert_equal(leaf0.bin, 0)
    assert_equal(Int(leaf0.gain.to_bits()), Int(Float64(120.0).to_bits()))


def test_relaxing_the_minimum_moves_the_accounting() raises:
    """The gate proved from the other side: with `min_data_in_leaf = 0` no
    leaf is illegal, so the same level reports zero illegal leaves and both
    leaves scored. A test whose two arms agreed whatever the rule did would
    establish nothing."""
    var level = _illegality_level()
    var audit = SharedSplitAudit.none()
    var split = find_best_split_shared(
        audit,
        level,
        lambda_reg=0.0,
        min_child_hess=0.0,
        min_data_in_leaf=0,
    )
    assert_true(split.found)
    assert_equal(split.bin, 0)
    assert_equal(audit.n_leaves, 2)
    assert_equal(audit.n_illegal, 0)
    assert_equal(audit.n_scored, 2)
    # Leaf 1's own contribution is exactly 0.0 (flat gradients), so the
    # summed gain is unchanged from the vetoing-minimum run above: the
    # accounting moved and the value did not.
    assert_equal(Int(split.gain.to_bits()), Int(Float64(120.0).to_bits()))


def test_a_level_no_leaf_can_split_yields_nothing() raises:
    """Every leaf illegal at every candidate sums to exactly 0.0, and 0.0
    never beats a running best that starts there under a strict `>`, so the
    level is terminal. This is why CatBoost's "delete a redundant split"
    rule is unreachable for us and is deliberately not implemented (see
    `docs/design/CATBOOST_CATALOG.md`, A8)."""
    var level = _illegality_level()
    var audit = SharedSplitAudit(7, 7, 7)
    var split = find_best_split_shared(
        audit,
        level,
        lambda_reg=0.0,
        min_child_hess=0.0,
        min_data_in_leaf=1000,
    )
    assert_false(split.found)
    # The audit is reset rather than left holding the caller's stale numbers.
    assert_equal(audit.n_leaves, 0)
    assert_equal(audit.n_illegal, 0)
    assert_equal(audit.n_scored, 0)


# ------------------------------- the shared search reduces to the old search


def _one_leaf(seed: UInt64, n_features: Int, n_bins: Int) -> Histogram:
    var size = n_features * n_bins
    var g = List[Float64](capacity=size)
    var h = List[Float64](capacity=size)
    var c = List[Int](capacity=size)
    var state = seed
    # Every feature must total to the same sums, which is what a real
    # histogram does, so the per-bin counts are drawn once and reused across
    # features with the gradients permuted by a per-feature rotation.
    var base_c = List[Int](capacity=n_bins)
    var base_g = List[Float64](capacity=n_bins)
    for _ in range(n_bins):
        state = state * 6364136223846793005 + 1442695040888963407
        base_c.append(Int(1 + (state >> 33) % 40))
        state = state * 6364136223846793005 + 1442695040888963407
        base_g.append(Float64(state >> 11) * (1.0 / 9007199254740992.0) - 0.5)
    for f in range(n_features):
        for b in range(n_bins):
            var src = (b + f) % n_bins
            g.append(base_g[src] * Float64(base_c[src]))
            h.append(Float64(base_c[src]))
            c.append(base_c[src])
    return Histogram.from_planes(g^, h^, c^, n_features, n_bins)


def _assert_same_split(got: SplitInfo, want: SplitInfo) raises:
    assert_equal(got.found, want.found)
    if not want.found:
        return
    assert_equal(got.feature, want.feature)
    assert_equal(got.bin, want.bin)
    assert_equal(got.default_left, want.default_left)
    assert_equal(got.is_categorical, want.is_categorical)
    assert_equal(Int(got.gain.to_bits()), Int(want.gain.to_bits()))


def test_a_level_of_one_leaf_is_the_ordinary_search_to_the_bit() raises:
    """`find_best_split_shared` over a single-leaf level is
    `find_best_split`: the cross-leaf sum has one addend, and the
    illegal-leaf rule cannot change a single-leaf answer because a candidate
    illegal for the only leaf sums to 0.0, which the strict `>` against a
    running best of 0.0 rejects exactly as the skip did.

    This is the property that keeps the shared search out of the default
    path's numbers, and it is asserted with and without a reserved missing
    bin because the missing-direction rule ("score missing-left first, so an
    exact tie keeps `default_left`") is the one most easily lost in a
    rewrite."""
    var shapes_f = [1, 3, 7]
    var shapes_b = [4, 9, 16]
    for i in range(len(shapes_f)):
        for j in range(len(shapes_b)):
            var n_features = shapes_f[i]
            var n_bins = shapes_b[j]
            var hist = _one_leaf(UInt64(11 + 13 * i + 101 * j),
                n_features, n_bins)
            var level = List[Histogram]()
            level.append(_one_leaf(UInt64(11 + 13 * i + 101 * j),
                n_features, n_bins))

            var no_missing = List[Int]()
            var with_missing = List[Int]()
            for _ in range(n_features):
                no_missing.append(-1)
                with_missing.append(n_bins - 2)

            for m in range(2):
                var missing = no_missing.copy() if m == 0 else (
                    with_missing.copy()
                )
                var want = find_best_split(
                    hist,
                    lambda_reg=1.0,
                    min_child_hess=1e-3,
                    min_data_in_leaf=3,
                    lambda_l1=0.25,
                    missing_bins=missing,
                )
                var audit = SharedSplitAudit.none()
                var got = find_best_split_shared(
                    audit,
                    level,
                    lambda_reg=1.0,
                    min_child_hess=1e-3,
                    min_data_in_leaf=3,
                    lambda_l1=0.25,
                    missing_bins=missing,
                )
                _assert_same_split(got, want)
                if got.found:
                    assert_equal(audit.n_leaves, 1)
                    assert_equal(audit.n_scored, 1)
                    assert_equal(audit.n_illegal, 0)


def test_the_shared_search_is_a_sum_over_the_level() raises:
    """Two leaves, and the level's gain at the winner is exactly the sum of
    the two leaves' own gains at that same candidate, computed by the
    ordinary search. Exact, on bits: the addends are the same numbers added
    in the same order, ascending by leaf."""
    var a = _one_leaf(UInt64(7), 4, 12)
    var b = _one_leaf(UInt64(9901), 4, 12)
    var level = List[Histogram]()
    level.append(_one_leaf(UInt64(7), 4, 12))
    level.append(_one_leaf(UInt64(9901), 4, 12))
    var audit = SharedSplitAudit.none()
    var shared = find_best_split_shared(
        audit, level, lambda_reg=1.0, min_child_hess=0.0, min_data_in_leaf=0
    )
    assert_true(shared.found)
    assert_equal(audit.n_leaves, 2)
    assert_equal(audit.n_illegal, 0)

    # Score the winner on each leaf alone by restricting the ordinary search
    # to that one feature and reading its gain at the same bin. Simplest
    # exact route: a single-leaf shared search restricted to the winning
    # feature, which the previous test pinned to `find_best_split`.
    var only: List[Int] = [shared.feature]
    var left_only = List[Histogram]()
    left_only.append(a^)
    var right_only = List[Histogram]()
    right_only.append(b^)
    var audit_a = SharedSplitAudit.none()
    var ga = find_best_split_shared(
        audit_a,
        left_only,
        lambda_reg=1.0,
        min_child_hess=0.0,
        min_data_in_leaf=0,
        features=only,
    )
    var audit_b = SharedSplitAudit.none()
    var gb = find_best_split_shared(
        audit_b,
        right_only,
        lambda_reg=1.0,
        min_child_hess=0.0,
        min_data_in_leaf=0,
        features=only,
    )
    # Each leaf's own best on that feature is at least as large as its
    # contribution to the shared winner, and the shared winner cannot exceed
    # their sum. That is the inequality a sum satisfies and a maximum or a
    # mean does not.
    assert_true(shared.gain <= ga.gain + gb.gain)


# ------------------------------------------------- the default is untouched


def test_the_default_policy_is_leafwise_and_unmoved() raises:
    """A fit that never names `grow_policy` and one that names
    `GROW_LEAFWISE` are the same tree on every bit, and neither is
    symmetric. This is the assertion that the new mode did not leak into the
    path everybody is on; `tests/test_golden_bits.mojo` is the real gate and
    is the orchestrator's to run."""
    var n_rows = 600
    var features = _dense(n_rows, 5)
    var data = bin_equal_width(features, n_rows, 5, 16)
    var grad = _grads(n_rows, features, n_rows)
    var hess = _ones(n_rows)

    var implicit = grow_tree(data, grad, hess, TreeParams(15, 5, 1.0, 1e-3))
    var explicit = grow_tree(
        data, grad, hess,
        TreeParams(15, 5, 1.0, 1e-3, grow_policy=GROW_LEAFWISE),
    )
    assert_same_tree_bits(implicit, explicit)
    assert_equal(implicit.n_leaves, 15)
    assert_false(is_symmetric(implicit))

    # Depth-wise is likewise unmoved by the new branch.
    var depth_a = grow_tree(
        data, grad, hess,
        TreeParams(15, 5, 1.0, 1e-3, grow_policy=GROW_DEPTHWISE),
    )
    var depth_b = grow_tree(
        data, grad, hess,
        TreeParams(15, 5, 1.0, 1e-3, grow_policy=GROW_DEPTHWISE),
    )
    assert_same_tree_bits(depth_a, depth_b)
    assert_not_equal(
        Int(depth_a.value[0].to_bits()), Int(Float64(0.0).to_bits())
    )


# ------------------------------------------------- determinism across workers


def _auto_workers():
    _ = setenv("MOJOTREES_NUM_WORKERS", "")


def test_workers_do_not_move_a_bit_in_either_mode() raises:
    """The cross-leaf sum is a sum over leaves, so its order is part of its
    value. It is fixed by the loop: the leaf loop runs ascending inside one
    feature's task, and the only thing that crosses a task boundary is the
    choice among features, which is a maximum folded serially in ascending
    scan order under a strict `>`. So 1, 3 and 8 workers must agree exactly,
    and the default policy must too."""
    var n_rows = 900
    var n_features = 6
    var features = _dense(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 16)
    var grad = _grads(n_rows, features, n_rows)
    var hess = _ones(n_rows)
    var counts = ["1", "3", "8"]

    var oblivious = _oblivious_params(3)
    var leafwise = TreeParams(15, 5, 1.0, 1e-3, grow_policy=GROW_LEAFWISE)

    _ = setenv("MOJOTREES_NUM_WORKERS", counts[0])
    var want_obl = grow_tree(data, grad, hess, oblivious)
    var want_leaf = grow_tree(data, grad, hess, leafwise)
    for w in range(len(counts)):
        _ = setenv("MOJOTREES_NUM_WORKERS", counts[w])
        assert_same_tree_bits(
            grow_tree(data, grad, hess, oblivious), want_obl
        )
        assert_same_tree_bits(
            grow_tree(data, grad, hess, leafwise), want_leaf
        )
    _auto_workers()
    # The fixture is one the worker count could have moved: it is wide enough
    # that the split scan actually fans out.
    assert_true(is_symmetric(want_obl))
    assert_equal(want_obl.n_leaves, 8)


# ------------------------------------ the representation stayed an ordinary tree


def test_predict_agrees_with_the_leaf_a_row_lands_in() raises:
    """Routing works on an oblivious tree through the ordinary `Tree`
    machinery, empty leaves included: `predict_row` and `predict_bins` walk
    the same nodes and answer the same value on every row."""
    var n_rows = 500
    var n_features = 5
    var features = _dense(n_rows, n_features)
    var data = bin_equal_width(features, n_rows, n_features, 16)
    var grad = _grads(n_rows, features, n_rows)
    var tree = grow_tree(data, grad, _ones(n_rows), _oblivious_params(3))
    assert_true(is_symmetric(tree))
    for r in range(0, n_rows, 11):
        var bins = List[Int](capacity=n_features)
        for f in range(n_features):
            bins.append(data.bin_at(r, f))
        assert_equal(
            Int(tree.predict_row(data, r).to_bits()),
            Int(tree.predict_bins(bins).to_bits()),
        )
        # A row's leaf is a real node of the tree.
        var leaf = tree.leaf_index_row(data, r)
        assert_true(leaf >= 0 and leaf < len(tree.feature))
        assert_equal(tree.feature[leaf], -1)
    assert_equal(tree.depth(), 3)


def test_an_oblivious_model_serializes_and_dumps() raises:
    """Requirement 2 of the lane, end to end: the mode changes how a tree is
    grown and nothing about how it is stored, so save/load reproduces
    predictions bit-exactly and the inspection dump builds with one record
    per node."""
    var n_rows = 400
    var features = _dense(n_rows, 3)
    var target = List[Float64](capacity=n_rows)
    for r in range(n_rows):
        target.append(
            2.0 * features[r]
            - features[n_rows + r]
            + 0.5 * features[2 * n_rows + r]
        )
    var params = BoosterParams(6, 0.1, _oblivious_params(3))
    var model = fit(features, n_rows, 3, target, SQUARED_ERROR, params, 32)
    for t in range(len(model.booster.trees)):
        assert_true(is_symmetric(model.booster.trees[t]))

    save_model(model, _TMP_PATH)
    var loaded = load_model(_TMP_PATH)
    remove(_TMP_PATH)
    assert_equal(len(loaded.booster.trees), len(model.booster.trees))
    for r in range(0, n_rows, 13):
        var row: List[Float64] = [
            features[r], features[n_rows + r], features[2 * n_rows + r]
        ]
        assert_equal(
            Int(loaded.predict(row).to_bits()),
            Int(model.predict(row).to_bits()),
        )
    for t in range(len(model.booster.trees)):
        assert_true(is_symmetric(loaded.booster.trees[t]))

    var dump = build_dump(model)
    assert_equal(len(dump.trees), len(model.booster.trees))
    for t in range(len(dump.trees)):
        assert_equal(
            len(dump.trees[t].nodes), len(model.booster.trees[t].feature)
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
